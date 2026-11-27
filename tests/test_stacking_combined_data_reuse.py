"""Regression coverage for lossless combined population and split reuse."""
import copy
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import warnings
import weakref
import gc

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestCentroid

import stacking
from utils.stacking.combined_data import CombinedDatasetSession, CombinedRunSplits, source_signatures
from utils.stacking.disk_arrays import DiskSyntheticSource, close_matrix


class CombinedReuseTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.original = self.root / 'original.csv'
        self.original.write_text('fixture\n')
        self.synthetic = self.root / 'synthetic.csv'
        self.synthetic.write_text('fixture\n')
        self.directory = self.root / 'private'
        self.directory.mkdir()
        self.config = stacking.get_default_config()
        self.config['telegram']['enabled'] = False
        self.config['logging']['enabled'] = False
        self.config['explainability']['enabled'] = False
        self.config['stacking']['methods'].update(automl=False, stacking=False)
        self.config['stacking']['memory_management']['spill_min_array_nbytes'] = 0
        self.config['stacking']['model_recovery'] = False
        self.config['stacking']['keep_trained_models'] = False
        self.config['evaluation']['n_jobs'] = 1

    def frame(self, seed=33, dtype=np.float32):
        rng = np.random.RandomState(seed)
        n = 90
        return pd.DataFrame({'b': rng.normal(size=n).astype(dtype),
                             'a': rng.randint(0, 100, n),
                             stacking.LSTM_SOURCE_FILE_COLUMN: ['one']*45 + ['two']*45,
                             stacking.LSTM_ROW_ORDER_COLUMN: np.arange(n)*2,
                             'attack_type': ['A','B','C']*30}, index=np.arange(10, n+10)*2)

    def session(self, frame=None):
        session = CombinedDatasetSession(self.frame() if frame is None else frame,
                                         [self.original], self.directory, source_signatures([self.original]))
        self.addCleanup(session.close)
        return session

    def prepare(self, config):
        return lambda holder, metadata: stacking.prepare_evaluation_data_splits(holder.pop(), config=config, include_lstm_metadata=metadata)

    def assert_frame_values(self, left, right):
        self.assertEqual(list(left.columns), list(right.columns))
        self.assertEqual(list(left.dtypes), list(right.dtypes))
        np.testing.assert_array_equal(left.index, right.index)
        for col in left.columns:
            np.testing.assert_array_equal(left[col], right[col])

    def assert_splits(self, expected, actual):
        for i in range(4):
            self.assertEqual(expected[i].dtype, actual[i].dtype)
            np.testing.assert_array_equal(expected[i], actual[i])
        for field in ('mean_', 'var_', 'scale_', 'n_samples_seen_'):
            np.testing.assert_array_equal(getattr(expected[4], field), getattr(actual[4], field))
        np.testing.assert_array_equal(expected[5].classes_, actual[5].classes_)
        if len(actual) == 8:
            self.assert_frame_values(expected[6], actual[6])
            self.assert_frame_values(expected[7], actual[7])

    def test_snapshot_exact_and_private_and_no_retained_frame(self):
        frame = self.frame()
        expected = frame.copy()
        ref = weakref.ref(frame)
        session = self.session(frame)
        del frame
        gc.collect()
        self.assertIsNone(ref())
        loaded = session.load_original()
        self.assert_frame_values(expected, loaded)
        loaded.iloc[0,0] = 999
        loaded.iloc[0,-1] = 'changed'
        self.assert_frame_values(expected, session.load_original())

    def test_split_reuse_seed_isolation_and_exact_lstm_metadata(self):
        frame = self.frame()
        session = self.session(frame)
        results = []
        for run in (1,2):
            cfg = stacking.build_experiment_run_config(self.config, run)
            expected = stacking.prepare_evaluation_data_splits(frame.copy(), config=cfg, include_lstm_metadata=True)
            cache = CombinedRunSplits(session, run, cfg['evaluation']['random_state'], ['b','a'], True)
            builder = mock.Mock(side_effect=self.prepare(cfg))
            first = cache.get(builder, False)
            self.assertEqual(len(first),6)
            self.assert_splits(expected, first)
            self.assertIsInstance(first[0], np.memmap)
            first[0][0,0] = 999
            first[2][0] = 999
            first[4].mean_[0] = 999
            for array in first[:4]: close_matrix(array)
            second = cache.get(builder, True)
            self.assert_splits(expected, second)
            builder.assert_called_once()
            for array in second[:4]: close_matrix(array)
            cache.close()
            self.assertFalse(cache.directory.exists())
            results.append(expected[0])
        self.assertFalse(np.array_equal(*results))

    def test_disk_ratio_reuse_all_ratios_exact_and_private(self):
        session = self.session()
        synthetic = self.frame(seed=44)
        builder = mock.Mock(return_value=synthetic)
        source = session.get_synthetic([self.synthetic], builder, ['b','a'], 0, 7)
        self.assertIsInstance(source, DiskSyntheticSource)
        self.assertIs(source, session.get_synthetic([self.synthetic], builder, ['b','a'], 0, 7))
        builder.assert_called_once()
        cache = CombinedRunSplits(session, 1, 42, ['b','a'], False)
        X,T,y,yt,scaler,encoder = cache.get(self.prepare(self.config), False)
        for ratio in (.25,.5,.75,1.):
            selected = stacking.select_synthetic_training_samples(synthetic,y,encoder,ratio,config=self.config)
            expected_X = np.concatenate([X,scaler.transform(selected[['b','a']].to_numpy(copy=False))])
            expected_y = encoder.transform(selected['attack_type'])
            choose = mock.Mock(side_effect=lambda: stacking.select_disk_synthetic_training_rows(source,y,encoder,ratio,self.config))
            for attempt in range(2):
                train,test,labels = cache.get_disk_augmented(ratio,X,T,source,scaler,choose,7)
                np.testing.assert_array_equal(expected_X,train)
                np.testing.assert_array_equal(T,test)
                np.testing.assert_array_equal(expected_y,labels)
                train[0,0]=999; test[0,0]=999; labels[0]=999
                for array in (train,test,labels): close_matrix(array)
            choose.assert_called_once()
        cache.close()

    def test_changed_sources_are_rejected(self):
        session = self.session()
        session.get_synthetic([self.synthetic], self.frame, ['b','a'],0,7)
        self.synthetic.write_text('modified synthetic')
        with self.assertRaisesRegex(RuntimeError,'sources changed'):
            session.assert_synthetic_sources([self.synthetic])
        self.original.write_text('modified original')
        with self.assertRaisesRegex(RuntimeError,'sources changed'):
            session.load_original()

    def test_partial_ratio_write_never_published(self):
        session=self.session()
        source=session.get_synthetic([self.synthetic],self.frame,['b','a'],0,7)
        cache=CombinedRunSplits(session,1,42,['b','a'],False)
        X,T,y,yt,scaler,encoder=cache.get(self.prepare(self.config),False)
        choose=lambda: stacking.select_disk_synthetic_training_rows(source,y,encoder,.25,self.config)
        with mock.patch('utils.stacking.combined_data.copy_matrix',side_effect=OSError('injected')):
            with self.assertRaisesRegex(OSError,'injected'):
                cache.get_disk_augmented(.25,X,T,source,scaler,choose,7)
        self.assertIsNone(cache.active_ratio)
        arrays=cache.get_disk_augmented(.25,X,T,source,scaler,choose,7)
        for array in arrays: close_matrix(array)
        cache.close()

    def test_concat_warning_fixed_and_row_order_preserved(self):
        frame=self.frame().drop(columns=[stacking.LSTM_SOURCE_FILE_COLUMN,stacking.LSTM_ROW_ORDER_COLUMN])
        expected=frame.copy()
        with warnings.catch_warnings():
            warnings.simplefilter('error',pd.errors.SettingWithCopyWarning)
            combined,_=stacking.concat_files_into_combined_files_df([(str(self.original),frame,'attack_type',['b','a'],'fixture')],['b','a'],{'A','B','C'},self.config)
        self.assert_frame_values(expected,frame)
        np.testing.assert_array_equal(combined[stacking.LSTM_ROW_ORDER_COLUMN],frame.index)
        np.testing.assert_array_equal(combined['attack_type'],frame['attack_type'])

    def test_specialized_modes_and_opt_out(self):
        self.assertTrue(stacking.sequential_combined_reuse_enabled(self.config))
        for section,key,value in [('stacking','model_recovery',True),('execution','skip_train_if_model_exists',True),('stacking','automl_only',True),('stacking','rerun_cached_experiments',1),('explainability','enabled',True),('evaluation','feature_set_workers',{'full':1})]:
            cfg=copy.deepcopy(self.config); cfg[section][key]=value
            self.assertFalse(stacking.sequential_combined_reuse_enabled(cfg))
        self.config['stacking']['memory_management']['reuse_combined_preprocessing']=False
        self.assertFalse(stacking.sequential_combined_reuse_enabled(self.config))

    def test_incompatible_cache_diagnostic_preserves_rejection(self):
        args = ('combined_files', 'TrainingAugmented@25%_CombinedFiles', 'original_training_augmented_training', .25, ['A','B','C'], 'Full Features', 'Nearest Centroid')
        key = stacking.build_resume_cache_key(*args, False)
        cache = {key: {'n_features': 2, 'features_list': ['a','b'], 'n_samples_train': 81, 'n_samples_test': 19}}
        progress = mock.Mock()
        results = {}
        with mock.patch('builtins.print') as output:
            recovered, counter = stacking.recover_cached_individual_classifier_result(cache, *args, results, 1, 1, progress, expected_n_features=2, expected_feature_names=['b','a'], expected_n_samples_train=90, expected_n_samples_test=18)
        self.assertFalse(recovered)
        self.assertEqual(counter, 1)
        self.assertFalse(cache)
        self.assertFalse(results)
        progress.update.assert_not_called()
        text = ' '.join(str(call) for call in output.call_args_list)
        self.assertIn('training rows: saved=81, active=90', text)
        self.assertIn('test rows: saved=19, active=18', text)
        self.assertIn('ordered feature names:', text)

    def test_real_grid_reads_once_and_matches_previous_results(self):
        paths,aug_paths=[],[]
        for i in range(2):
            for target,seed,prefix in ((paths,10+i,'original'),(aug_paths,20+i,'augmented')):
                path=self.root/f'{prefix}{i}.csv'
                self.frame(seed=seed).drop(columns=[stacking.LSTM_SOURCE_FILE_COLUMN,stacking.LSTM_ROW_ORDER_COLUMN]).rename(columns={'attack_type':'Label'}).to_csv(path,index=False)
                target.append(str(path))
        cfg=copy.deepcopy(self.config)
        cfg['stacking'].update(experiment_runs=2,augmentation_ratios=[.25,.5])
        cfg['stacking']['feature_sets_config'].update(use_full=True,use_ga=True,use_pca=False,use_rfe=False,use_extra_trees=False)
        cfg['execution']['test_data_augmentation']=True
        cfg['stacking']['methods'].update(feature_selection=True,augmentation=True,hyperparameter_optimization=True)
        def run(reuse, name, resume=False):
            config=copy.deepcopy(cfg)
            config['stacking']['memory_management']['reuse_combined_preprocessing']=reuse
            config['stacking']['results_dir']=name
            outputs=[]
            with ExitStack() as mocks:
                patch=lambda name,**kw:mocks.enter_context(mock.patch.object(stacking,name,**kw))
                patch('CONFIG',new=config)
                patch('load_feature_selection_results',return_value=(['a'],None,None,None))
                patch('get_models',side_effect=lambda **kw:{'Nearest Centroid':NearestCentroid(metric='euclidean')})
                patch('build_optimized_hyperparameter_models',side_effect=lambda *a,**kw:({'Nearest Centroid':NearestCentroid(metric='manhattan')},{'Nearest Centroid':{'metric':'manhattan'}}))
                patch('load_augmented_files_for_combined_evaluation',return_value=aug_paths)
                patch('send_telegram_message'); patch('notify_new_best_result_if_applicable')
                patch('persist_per_run_confusion_matrix_artifacts_safely',return_value='fixture-matrix')
                patch('save_combined_files_results_to_csv',side_effect=lambda ref,rows,**kw:outputs.extend(copy.deepcopy(rows)) or self.root)
                patch('save_combined_files_augmentation_comparison'); patch('remove_cache_file')
                patch('write_memory_phase_event'); patch('verbose_output')
                reads=patch('load_dataset',wraps=stacking.load_dataset)
                splits=patch('prepare_evaluation_data_splits',wraps=stacking.prepare_evaluation_data_splits)
                choices=patch('select_disk_synthetic_training_rows',wraps=stacking.select_disk_synthetic_training_rows)
                if resume: patch('fit_classifier_with_progress',side_effect=AssertionError('must resume without fitting'))
                stacking.execute_combined_files_mode_pipeline(paths,'fixture',config=config)
                return outputs,Counter(str(c.args[0]) for c in reads.call_args_list),splits.call_count,choices.call_count
        legacy,old_reads,old_splits,old_choices=run(False,'legacy-results')
        updated,new_reads,new_splits,new_choices=run(True,'new-results')
        self.assertEqual(len(legacy),24); self.assertEqual(len(updated),len(legacy))
        self.assertEqual(new_reads,Counter({p:1 for p in paths+aug_paths}))
        self.assertGreater(sum(old_reads.values()),sum(new_reads.values()))
        self.assertEqual(new_splits,2); self.assertEqual(new_choices,4)
        self.assertGreater(old_splits,new_splits); self.assertGreater(old_choices,new_choices)
        print(f"Fixture counts (previous -> updated): CSV loads {sum(old_reads.values())} -> {sum(new_reads.values())}; original splits {old_splits} -> {new_splits}; synthetic selections {old_choices} -> {new_choices}")
        fields=('experiment_run','experiment_mode','feature_set','hyperparameter_mode','augmentation_ratio','model_name','n_features','n_samples_train','n_samples_test','features_list','accuracy','weighted_precision','weighted_recall','weighted_f1_score','macro_f1_score','fpr','fnr','per_class_f1_scores')
        for previous,current in zip(legacy,updated):
            for field in fields:
                self.assertEqual(previous.get(field),current.get(field),(field,previous.get(field),current.get(field)))
        self.assertFalse(list((self.root/'new-results'/'Array_Cache').glob('FeatureSource_*')))
        resumed,_,_,_=run(True,'new-results',resume=True)
        self.assertEqual(len(resumed),len(updated))


if __name__=='__main__':
    unittest.main()
