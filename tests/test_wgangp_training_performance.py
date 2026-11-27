"""Training performance contracts; CPU always, CUDA and DataParallel when available."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas.io.formats.style
import torch
import wgangp as wg


class TrainingPerformanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.args = SimpleNamespace(csv_path='test.csv', batch_size=32, latent_dim=8,
            critic_steps=2, lambda_gp=10., use_amp=False, _mixed_categorical_groups=[],
            _mixed_numeric_feature_count=4)
        self.config = wg.get_default_config()
        self.globals = (wg.V61_REFERENCE_MOMENTS, wg.V6_REFERENCE_CORRELATIONS, wg.V64_REFERENCE_NUMERIC_QUANTILES, wg.CONFIG)
        wg.V61_REFERENCE_MOMENTS, wg.V6_REFERENCE_CORRELATIONS, wg.V64_REFERENCE_NUMERIC_QUANTILES = {}, {}, {}
        wg.CONFIG = self.config
        self.addCleanup(self.restore_globals)

    def restore_globals(self):
        wg.V61_REFERENCE_MOMENTS, wg.V6_REFERENCE_CORRELATIONS, wg.V64_REFERENCE_NUMERIC_QUANTILES, wg.CONFIG = self.globals

    def make_models(self, device):
        G = wg.Generator(8, 4, 2, [12, 16], 4, 1).to(device)
        D = wg.Discriminator(4, 2, [12, 8], 4).to(device)
        return G, D

    def test_categorical_forward_skips_unused_log_softmax_and_preserves_sampling(self):
        G = wg.Generator(8, 5, 2, [12], 4, 1, categorical_groups=[(2, 5)],
            categorical_class_masks=[np.array([[True, True, False], [False, True, True]])])
        clone = copy.deepcopy(G)
        z, labels = torch.randn(32, 8), torch.arange(32) % 2
        torch.manual_seed(100)
        with patch('torch.log_softmax', wraps=torch.log_softmax) as call:
            actual = G(z, labels)
            self.assertEqual(call.call_count, 0)
        rng = torch.get_rng_state().clone()
        torch.manual_seed(100)
        expected, _ = clone(z, labels, return_log_probabilities=True)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for key, value in G.state_dict().items():
            torch.testing.assert_close(value, clone.state_dict()[key], atol=0, rtol=0)
        G.eval()
        torch.manual_seed(101)
        with patch('torch.log_softmax', wraps=torch.log_softmax) as call:
            inference = G(z, labels)
            self.assertEqual(call.call_count, 1)
        torch.manual_seed(101)
        inference_with_probs, _ = G(z, labels, return_log_probabilities=True)
        torch.testing.assert_close(inference, inference_with_probs, atol=0, rtol=0)

    def check_restricted_backward(self, device, parallel=False):
        G, D = self.make_models(device)
        if parallel:
            G, D = torch.nn.DataParallel(G), torch.nn.DataParallel(D)
        original_g, original_d = copy.deepcopy(G), copy.deepcopy(D)
        labels = torch.arange(32, device=device) % 2
        z = torch.randn(32, 8, device=device)
        wg.set_seed(123)
        loss = -D(G(z, labels), labels).mean()
        loss.backward(inputs=tuple(G.parameters()))
        wg.set_seed(123)
        baseline_loss = -original_d(original_g(z, labels), labels).mean()
        baseline_loss.backward()
        torch.testing.assert_close(loss, baseline_loss, atol=0, rtol=0)
        for a, b in zip(G.parameters(), original_g.parameters()):
            torch.testing.assert_close(a.grad, b.grad, atol=1e-7, rtol=1e-6)
        self.assertTrue(all(p.grad is None for p in D.parameters()))
        self.assertTrue(any(p.grad is not None for p in original_d.parameters()))

    def test_restricted_backward_cpu(self):
        self.check_restricted_backward(torch.device('cpu'))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_restricted_backward_cuda(self):
        self.check_restricted_backward(torch.device('cuda:0'))

    @unittest.skipUnless(torch.cuda.device_count() >= 2, 'Two CUDA devices required')
    def test_restricted_backward_data_parallel(self):
        self.check_restricted_backward(torch.device('cuda:0'), True)

    def test_invalid_gradient_penalty_is_rejected(self):
        class BadCritic(torch.nn.Module):
            def forward(self, x, y):
                return x.sum(1) * float('inf')
        x = torch.randn(32, 4)
        labels = torch.arange(32) % 2
        with patch.object(wg, 'send_exception_via_telegram'), patch.object(wg, 'safe_critical'):
            with self.assertRaisesRegex(FloatingPointError, 'interpolated_scores'):
                wg.gradient_penalty(BadCritic(), x, x.clone(), labels, torch.device('cpu'), self.config, self.args)

    def test_invalid_input_never_updates_optimizer(self):
        G, D = self.make_models(torch.device('cpu'))
        opt = torch.optim.Adam(D.parameters())
        x = torch.randn(32, 4); x[0, 0] = float('nan')
        with patch.object(wg, 'safe_critical'), patch.object(opt, 'step', wraps=opt.step) as step:
            with self.assertRaisesRegex(FloatingPointError, 'real_samples'):
                wg.execute_discriminator_training_steps(G, D, opt, None, x, torch.arange(32)%2, torch.device('cpu'), self.args, self.config, 2, 0, 0)
            step.assert_not_called()
        self.assertEqual(len(opt.state), 0)

    def check_reference_cache(self, device):
        # Mixed numeric and categorical, absent class, small class, rare supported category.
        labels = torch.tensor([0]*19+[1]*17+[2]*3, device=device)
        labels = labels[torch.randperm(len(labels), device=device)]
        x = torch.randn(len(labels), 7, device=device, requires_grad=True)
        groups = [(4, 7)]
        moments = {i: (torch.tensor([.2,.5,.3,.4,.75,.25,0.], device=device),
            torch.ones(7, device=device), torch.tensor([True,False,True,True,False,False,False],device=device)) for i in range(4)}
        correlations = {i:(torch.tensor([0,1,2], device=device), torch.eye(3, device=device)) for i in range(4)}
        quantiles = {i:(torch.zeros(4,4,device=device),torch.ones(4,device=device),torch.ones(4,device=device)) for i in range(4)}
        rows = wg.prepare_fidelity_class_rows(labels)
        self.assertNotIn(3, rows)
        for k,v in rows.items():
            torch.testing.assert_close(v, (labels==k).nonzero(as_tuple=True)[0], atol=0, rtol=0)
        cache = wg.prepare_fidelity_reference_cache(7, device, groups, moments, correlations)
        def objective(data, fast):
            probs = torch.cat((data[:,:4],torch.log_softmax(data[:,4:],1)),1)
            kwargs = dict(class_rows=rows,reference_cache=cache) if fast else {}
            return (wg.class_reference_moment_loss(data,labels,moments,probs,groups,**kwargs)
                   + wg.class_reference_correlation_loss(data,labels,correlations,**kwargs)
                   + wg.class_reference_numeric_quantile_loss(data,labels,quantiles,4,minimum_class_rows=16,**kwargs))
        a, b = objective(x,False), objective(x,True)
        da = torch.autograd.grad(a,x)[0]; db = torch.autograd.grad(b,x)[0]
        torch.testing.assert_close(a,b,atol=0,rtol=0)
        torch.testing.assert_close(da,db,atol=1e-7,rtol=1e-6)

    def test_cached_fidelity_cpu(self): self.check_reference_cache(torch.device('cpu'))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cached_fidelity_cuda(self): self.check_reference_cache(torch.device('cuda:0'))

    def check_finite_scans(self, device):
        a = torch.tensor([torch.finfo(torch.float32).max],device=device)
        b = torch.tensor([1.,2.],device=device)
        c = torch.ones(2,3,device=device).T  # Non-contiguous fallback.
        self.assertTrue(wg.all_finite_tensor_values([a,b,c,torch.tensor(2.)],max_pack_elements=2))
        for invalid in (float('nan'),float('inf'),-float('inf')):
            b[1] = invalid
            self.assertFalse(wg.all_finite_tensor_values([a,b,c],max_pack_elements=2))
            with patch.object(wg,'safe_critical'):
                with self.assertRaisesRegex(FloatingPointError,'value=second'):
                    wg.require_finite_tensor_group([('first',a),('second',b),('third',c)],self.args,1,1,'test')
        self.assertTrue(wg.all_finite_tensor_values([]))

    def test_finite_scans_cpu(self): self.check_finite_scans(torch.device('cpu'))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_finite_scans_cuda(self): self.check_finite_scans(torch.device('cuda:0'))

    def test_optimizer_checks_and_metric_values(self):
        p = torch.nn.Parameter(torch.ones(3)); opt = torch.optim.Adam([p])
        p.sum().backward(); opt.step()
        wg.require_finite_optimizer_state(opt,'adam',self.args,1,1)
        opt.state[p]['exp_avg'][0] = float('nan')
        with patch.object(wg,'safe_critical'):
            with self.assertRaisesRegex(FloatingPointError,'exp_avg'):
                wg.require_finite_optimizer_state(opt,'adam',self.args,1,1)
        history = {key:[] for key in ('steps','loss_D','loss_G','gp','D_real','D_fake','wasserstein')}
        vals = [torch.tensor(x) for x in (.1,.2,.3,4.,3.)]
        result = wg.record_training_step_metrics(4,self.args,history,*vals)
        self.assertEqual(result,tuple(v.item() for v in vals))
        self.assertEqual(history['wasserstein'],[1.])

    def test_gpu_cache_budget_reserves_headroom(self):
        gib = 1024**3
        self.assertTrue(wg.gpu_resident_cache_fits(12*gib,6*gib))
        self.assertFalse(wg.gpu_resident_cache_fits(12*gib,6*gib,max_fraction=.4))
        self.assertFalse(wg.gpu_resident_cache_fits(4*gib,2*gib))
        with self.assertRaises(ValueError): wg.gpu_resident_cache_fits(12*gib,6*gib,max_fraction=1.01)

    def check_paired_critic(self, device, parallel=False):
        _, D = self.make_models(device)
        old = copy.deepcopy(D)
        if parallel: D, old = torch.nn.DataParallel(D), torch.nn.DataParallel(old)
        x,y = torch.randn(32,4,device=device),torch.arange(32,device=device)%2
        fake = torch.randn_like(x)
        a,b = D(x,y,fake); c,d = old(x,y),old(fake,y)
        torch.testing.assert_close(a,c,atol=0,rtol=0); torch.testing.assert_close(b,d,atol=0,rtol=0)
        (b.mean()-a.mean()).backward(); (d.mean()-c.mean()).backward()
        for p,q in zip(D.parameters(),old.parameters()): torch.testing.assert_close(p.grad,q.grad,atol=1e-6,rtol=1e-5)

    def test_paired_critic_cpu(self): self.check_paired_critic(torch.device('cpu'))

    @unittest.skipUnless(torch.cuda.device_count() >= 2, 'Two CUDA devices required')
    def test_paired_critic_data_parallel(self): self.check_paired_critic(torch.device('cuda:0'), True)

    def test_default_provenance_and_journal_compatibility(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'data.csv'; path.write_text('x,Label\n1,A\n2,B\n')
            self.config['wgangp']['csv_path'] = str(path)
            args = wg.ConfigNamespace(self.config)
            args._generation_checkpoint_digest = 'trusted-existing-checkpoint'
            before = wg.build_wgangp_provenance(args)
            journal = wg.generation_journal_identity(args,None,None,np.array([0,1]),['x'],torch.device('cpu'))
            self.assertEqual(journal['release'],'V6.5.31')
            self.assertFalse(self.config['training']['coalesce_critic_forwards'])
            self.config['training']['coalesce_critic_forwards'] = True
            after = wg.build_wgangp_provenance(args)
            self.assertNotEqual(before['sha256'],after['sha256'])
            wg.restore_generation_training_settings(args,self.config,{'provenance':before,'epoch':60})
            self.assertFalse(self.config['training']['coalesce_critic_forwards'])
            wg.verify_wgangp_provenance(before,args,'old checkpoint')


class TrainingFollowupTests(unittest.TestCase):
    def test_resumed_eta_uses_session_work(self):
        self.assertEqual(wg.estimate_remaining_training_seconds(1200, 1, 44), 52800)
        self.assertEqual(wg.estimate_remaining_training_seconds(1200, 9256, 44*9256), 52800)
        self.assertEqual(wg.estimate_remaining_training_seconds(100, 0, 44), 0)
        self.assertEqual(wg.estimate_remaining_training_seconds(100, 100, 0), 0)

    def test_deferred_penalty_matches_original_and_keeps_diagnostics(self):
        D=wg.Discriminator(4,2,[8],4)
        x=torch.randn(16,4); y=torch.arange(16)%2; fake=torch.randn_like(x)
        args=SimpleNamespace(csv_path='test.csv')
        torch.manual_seed(7)
        first=wg.gradient_penalty(D,x,fake,y,torch.device('cpu'),wg.get_default_config(),args)
        torch.manual_seed(7); checks=[]
        second=wg.gradient_penalty(D,x,fake,y,torch.device('cpu'),wg.get_default_config(),args,finite_checks=checks)
        self.assertEqual([name for name,_ in checks],['interpolated_samples','interpolated_scores','interpolated_gradients','interpolated_gradient_norms','gradient_penalty'])
        wg.require_finite_tensor_group(checks,args,1,1,'test')
        torch.testing.assert_close(first,second,atol=0,rtol=0)
        first_grad=torch.autograd.grad(first,tuple(D.parameters()),allow_unused=True)
        second_grad=torch.autograd.grad(second,tuple(D.parameters()),allow_unused=True)
        for a,b in zip(first_grad,second_grad):
            if a is None: self.assertIsNone(b)
            else: torch.testing.assert_close(a,b,atol=0,rtol=0)

    def test_nonfinite_critic_forward_does_not_step(self):
        class InvalidGenerator(torch.nn.Module):
            def forward(self,z,y): return torch.full((len(y),4),float('nan'),device=z.device)
        D=wg.Discriminator(4,2,[8],4); opt=torch.optim.Adam(D.parameters())
        args=SimpleNamespace(csv_path='test.csv',batch_size=16,latent_dim=8,critic_steps=1,lambda_gp=10.,use_amp=False)
        before={k:v.clone() for k,v in D.state_dict().items()}
        with patch.object(wg,'safe_critical'),patch.object(wg,'send_exception_via_telegram'):
            with self.assertRaisesRegex(FloatingPointError,'generated_samples'):
                wg.execute_discriminator_training_steps(InvalidGenerator(),D,opt,None,torch.randn(16,4),torch.arange(16)%2,torch.device('cpu'),args,wg.get_default_config(),2,0,0)
        self.assertFalse(opt.state)
        for k,v in D.state_dict().items(): torch.testing.assert_close(v,before[k],atol=0,rtol=0)

    def test_primary_critic_is_opt_in_and_provenance_isolated(self):
        args=SimpleNamespace(csv_path=None,gpu=None)
        legacy=wg.get_default_config(); primary=copy.deepcopy(legacy); primary['training']['critic_execution']='primary'
        args._generation_provenance_config=legacy
        a=wg.build_wgangp_provenance(args)
        self.assertNotIn('critic_execution',a['identity']['settings'])
        args._generation_provenance_config=primary
        b=wg.build_wgangp_provenance(args)
        self.assertNotEqual(a['sha256'],b['sha256'])
        self.assertEqual(b['identity']['settings']['critic_execution'],'primary')
        with self.assertRaisesRegex(ValueError,'provenance'):
            wg.verify_wgangp_provenance(a,args,'test')
        with self.assertRaisesRegex(ValueError,'automatic multi-GPU'):
            wg.wrap_training_parallel_models(None,None,args,primary,torch.device('cpu'))
        self.assertEqual(wg.wrap_training_parallel_models(None,None,args,legacy,torch.device('cpu')),(None,None))

    def test_cli_primary_critic_overrides_and_explicit_fields(self):
        import sys
        with patch.object(sys,'argv',['wgangp.py','--critic-execution','primary','--coalesce-critic-forwards']):
            config=wg.args_to_config_overrides(wg.parse_args())
        self.assertEqual(config['training']['critic_execution'],'primary')
        self.assertTrue(config['training']['coalesce_critic_forwards'])
        self.assertIn('critic_execution',config['wgangp']['generation_explicit_training_fields'])
        self.assertIn('coalesce_critic_forwards',config['wgangp']['generation_explicit_training_fields'])

    @unittest.skipUnless(torch.cuda.device_count()>=2,'Two CUDA devices required')
    def test_primary_critic_preserves_generator_shards_and_penalty_backward(self):
        device=torch.device('cuda:0'); G=wg.Generator(8,4,2,[12,16],4,1).to(device); D=wg.Discriminator(4,2,[12,8],4).to(device)
        config=wg.get_default_config(); config['training']['critic_execution']='primary'
        G,D=wg.wrap_training_parallel_models(G,D,SimpleNamespace(gpu=None),config,device)
        self.assertIsInstance(G,torch.nn.DataParallel); self.assertNotIsInstance(D,torch.nn.DataParallel)
        x=torch.randn(32,4,device=device); labels=torch.arange(32,device=device)%2
        fake=G(torch.randn(32,8,device=device),labels).detach()
        gp=wg.gradient_penalty(D,x,fake,labels,device,config)
        gp.backward(); self.assertTrue(all(torch.isfinite(p.grad).all() for p in D.parameters() if p.grad is not None))


if __name__ == '__main__': unittest.main()
