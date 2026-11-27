"""Regression coverage for resume-aware training reporting (V6.5.50)."""
import datetime
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas.io.formats.style
import wgangp as wg


class TrainingTimingTests(unittest.TestCase):
    def args(self, start=35, prior=None, epochs=60):
        return SimpleNamespace(epochs=epochs, csv_path='balanced_dataset_ddos.csv',
                               _training_session_start_epoch=start,
                               _training_elapsed_before_session=prior)

    def test_actual_latam_resume_notification(self):
        args = self.args()
        with patch.object(wg, 'training_elapsed_clock', return_value=10000), \
             patch.object(wg, 'send_telegram_message') as send, \
             patch.object(wg, 'format_estimated_finish_time_suffix', return_value=' | Estimated Finish Time: test') as finish:
            self.assertEqual(wg.send_epoch_telegram_notifications(args, True, 41, 70, 10, 168, ''), 80)
        msg = send.call_args.args[1]
        for fragment in ('70% (42/60 epochs)', 'Resumed at epoch: 35/60',
                         'Session progress: 28.0% (7/25 epochs)',
                         'Session elapsed: 2h 43m 52s',
                         'Recorded total training elapsed: unknown', 'ETA: 7h 1m 22s'):
            self.assertIn(fragment, msg)
        finish.assert_called_once_with(9832 * 18 / 7)

    def test_batch_and_epoch_eta_use_same_denominator(self):
        args = self.args(prior=48000)
        with patch.object(wg, 'format_estimated_finish_time_suffix', return_value='') as finish:
            epoch = wg.format_training_progress_timing(args, 42, 1, 9832)
            batch = wg.format_training_progress_timing(args, 42 * 9256, 9256, 9832, 25)
        self.assertEqual(finish.call_args_list[0], finish.call_args_list[1])
        self.assertIn('ETA: 7h 1m 22s', epoch)
        self.assertIn('ETA: 7h 1m 22s', batch)
        self.assertIn('Recorded total training elapsed: 16h 3m 52s', epoch)
        self.assertIn('64792/231400 batches', batch)

    def test_no_fabricated_initial_eta(self):
        args = self.args()
        for completed in (35*9256, 35*9256+1, 35*9256+24):
            with patch.object(wg, 'format_estimated_finish_time_suffix') as finish:
                msg = wg.format_training_progress_timing(args, completed, 9256, 2, 25)
            self.assertIn('ETA: estimating', msg)
            self.assertIn('Estimated Finish Time: estimating', msg)
            finish.assert_not_called()
        with patch.object(wg, 'format_estimated_finish_time_suffix', return_value='') as finish:
            wg.format_training_progress_timing(args, 35*9256+25, 9256, 5, 25)
        finish.assert_called_once_with(5*(25*9256-25)/25)

    def test_fresh_and_completed_training(self):
        fresh = self.args(start=0, prior=None)
        self.assertIn('Recorded total training elapsed: 1h 0m 0s', wg.format_training_progress_timing(fresh, 3, 1, 3600))
        self.assertIn('Session progress: 5.0% (3/60 epochs)', wg.format_training_progress_timing(fresh, 3, 1, 3600))
        for args in (self.args(), self.args(start=60, prior=80000), fresh):
            with patch.object(wg, 'format_estimated_finish_time_suffix', return_value='') as finish:
                msg = wg.format_training_progress_timing(args, 60, 1, 10)
            self.assertIn('Session progress: 100.0%', msg)
            self.assertIn('ETA: 0s', msg)
            finish.assert_called_once_with(0.0)

    def test_invalid_history_is_unknown(self):
        for prior in (None, '', 0, -1, 'bad', float('nan'), float('inf')):
            self.assertIsNone(wg.recorded_training_elapsed(self.args(prior=prior), 123))
        self.assertEqual(wg.recorded_training_elapsed(self.args(prior='1200'), 123), 1323)

    def test_actual_percentage_and_final_notification_once(self):
        args = self.args(start=0, epochs=3)
        config = {'telegram': {'enabled': True, 'progress_pct': 30}}
        enabled, next_notify, interval = wg.init_telegram_progress(args, config, 0)
        with patch.object(wg, 'training_elapsed_clock', return_value=100), patch.object(wg, 'send_telegram_message') as send:
            for epoch in (0, 1, 2, 2):
                next_notify = wg.send_epoch_telegram_notifications(args, enabled, epoch, next_notify, interval, 0, '')
        self.assertEqual(next_notify, 101)
        self.assertEqual(send.call_count, 3)
        for call, pct in zip(send.call_args_list, (33, 66, 100)):
            self.assertIn(f'training progress: {pct}%', call.args[1])
        self.assertEqual(wg.init_telegram_progress(self.args(start=56), config, 56)[1], 100)
        self.assertEqual(wg.init_telegram_progress(self.args(start=60), config, 60)[1], 101)

    def test_multiple_thresholds_crossed_once(self):
        args = self.args(start=0, epochs=2)
        with patch.object(wg, 'training_elapsed_clock', return_value=100), patch.object(wg, 'send_telegram_message') as send:
            next_notify = wg.send_epoch_telegram_notifications(args, True, 0, 10, 10, 0, '')
        self.assertEqual(next_notify, 60)
        send.assert_called_once()
        self.assertIn('50% (1/2 epochs)', send.call_args.args[1])

    def test_monotonic_final_timing_known_unknown_and_already_complete(self):
        for args, expected in ((self.args(prior=48000), 48100), (self.args(), ''),
                               (self.args(start=60, prior=48000), 48000), (self.args(start=0), 100)):
            with patch.object(wg, 'training_elapsed_clock', return_value=1100), \
                 patch.object(wg.time, 'time', side_effect=AssertionError('wall clock used for elapsed time')), \
                 patch('builtins.print'):
                wg.compute_and_store_final_timing(args, 1000, 900)
            self.assertEqual(args._last_training_time, expected)
            self.assertEqual(args._last_file_time, 200)

    def test_finish_time_uses_sao_paulo_calendar(self):
        fixed = datetime.datetime(2026, 11, 27, 11, 49, 38, tzinfo=ZoneInfo('America/Sao_Paulo'))
        with patch.object(wg.datetime, 'datetime', wraps=datetime.datetime) as clock:
            clock.now.return_value = fixed
            label = wg.format_estimated_finish_time_suffix(9832*18/7)
        self.assertEqual(label, ' | Estimated Finish Time: 18h:51m-27/11/2026')
        clock.now.assert_called_once_with(ZoneInfo('America/Sao_Paulo'))

    def test_unknown_final_notification_does_not_convert_blank_to_zero(self):
        args = self.args()
        args._last_training_time = ''
        with patch.object(wg, 'send_telegram_message') as send, patch('builtins.print'):
            wg.send_file_saved_and_timing_messages(args, {})
        self.assertIn('Training: unknown', send.call_args.args[1])


if __name__ == '__main__':
    unittest.main()
