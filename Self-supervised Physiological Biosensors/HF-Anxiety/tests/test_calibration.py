"""Regression coverage for the collapsed resting baseline (no network writes)."""

import importlib.util
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

SERVICE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE))
from calibration import MINIMUM_BASELINE_STDS, baseline_issue, baseline_statistics


# Synthetic minute features exercise the reported failure without committing
# a participant's actual physiological trace or identity to a public repository.
_t = np.arange(10)
_hr = 64 + 4 * np.sin(_t)
RESTING_WINDOWS = np.column_stack([
    _hr, 60000 / _hr, 470 + 100 * np.cos(.6 * _t),
    700 + 130 * np.sin(.7 * _t), np.full(10, 16.25), np.full(10, 6.8),
    37.2 + .04 * np.sin(.8 * _t), .015 + .005 * np.sin(_t),
    .86 + .02 * np.sin(.4 * _t), .01 + .001 * np.cos(_t),
]).astype(np.float32)
OLD_MEAN = [72, 833.33, 560, 740, 14, 4, 36.1, .018, .86, .01]


class BaselineStatisticsTests(unittest.TestCase):
    def test_short_missing_nonfinite_and_flat_sessions_are_rejected(self):
        for values in (None, [], RESTING_WINDOWS[:3], RESTING_WINDOWS[:9],
                       np.full((10, 10), np.nan), np.tile(OLD_MEAN, (10, 1))):
            with self.subTest(values_type=type(values).__name__):
                with self.assertRaises(ValueError):
                    baseline_statistics(values)

    def test_original_collapsed_baseline_requires_recalibration(self):
        self.assertIsNotNone(baseline_issue(MINIMUM_BASELINE_STDS))

    def test_real_minute_variation_replaces_the_tiny_hrv_divisors(self):
        mean, std = baseline_statistics(RESTING_WINDOWS)
        self.assertGreater(std[2], 50)
        self.assertGreater(std[3], 50)
        self.assertIsNone(baseline_issue(std, 10))
        self.assertLess(np.abs((RESTING_WINDOWS - mean) / std).max(), 4)

    def test_invalid_stored_scales_are_rejected_without_invalidating_good_legacy_scales(self):
        _, std = baseline_statistics(RESTING_WINDOWS)
        self.assertIsNone(baseline_issue(std))
        self.assertIsNotNone(baseline_issue(std, 3))
        for invalid in ([None] * 10, [0] * 10, [float('inf')] * 10, [1] * 9):
            self.assertIsNotNone(baseline_issue(invalid))


class EndpointRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Use the real unchanged weights. C1_TEST_MODEL_DIR allows a separately
        # materialized LFS download without modifying the repository artifacts.
        model_dir = Path(os.environ.get('C1_TEST_MODEL_DIR', SERVICE / 'models'))
        required = ['LSTM_AE_FINAL.pth', 'DIRECT_RIDGE_FORECAST_FINAL.pth']
        if any(not (model_dir / f).exists() or (model_dir / f).stat().st_size < 1000
               for f in required):
            raise unittest.SkipTest('Materialize the model LFS weights for endpoint tests.')
        cls.temp = tempfile.TemporaryDirectory()
        app_dir = Path(cls.temp.name)
        (app_dir / 'models').mkdir()
        shutil.copy(SERVICE / 'main.py', app_dir / 'main.py')
        for name in required:
            shutil.copy(model_dir / name, app_dir / 'models' / name)
        for path in (SERVICE / 'models').glob('*_metadata.json'):
            shutil.copy(path, app_dir / 'models' / path.name)
        spec = importlib.util.spec_from_file_location('c1_calibration_regression', app_dir / 'main.py')
        cls.api = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.api)
        cls.api.torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, 'temp'):
            cls.api.db_client.close()
            cls.temp.cleanup()

    def setUp(self):
        self.api.query_api = Mock()
        self.api.write_api = Mock()

    def baseline_record(self, mean, std, **extra):
        return {**dict(zip(self.api.NORM_MEAN_FIELDS, map(float, mean))),
                **dict(zip(self.api.NORM_STD_FIELDS, map(float, std))), **extra}

    @staticmethod
    def table(values):
        return [SimpleNamespace(records=[SimpleNamespace(values=value) for value in values])]

    def prediction(self, norm, windows):
        now = datetime.now(timezone.utc)
        records = []
        for i, values in enumerate(windows):
            time = now - timedelta(minutes=len(windows) - i - 1)
            records.append(SimpleNamespace(
                values=dict(zip(self.api.FEATURE_NAMES, map(float, values))),
                get_time=lambda time=time: time,
            ))
        self.api.query_api.query.side_effect = [self.table([norm]), [SimpleNamespace(records=records)]]
        return self.api.get_escalation_forecast('regression_participant')

    def test_old_baseline_cannot_emit_predictions_or_history_scores(self):
        norm = self.baseline_record(OLD_MEAN, MINIMUM_BASELINE_STDS, reconstruction_error_p90=.25)
        for endpoint in (self.api.get_escalation_forecast, self.api.get_physiological_history):
            self.api.query_api.query.side_effect = None
            self.api.query_api.query.return_value = self.table([norm])
            result = endpoint('regression_participant')
            self.assertEqual(result['status'], 'not_calibrated')
            self.assertNotIn('current_risk_index', result)

    def test_server_recomputes_statistics_and_uses_only_full_real_sequences(self):
        payload = self.api.NormParamsPayload(
            b_mean=OLD_MEAN, b_std=MINIMUM_BASELINE_STDS.tolist(),
            baseline_windows=RESTING_WINDOWS.tolist(),
        )
        with patch.object(self.api.global_ae_model, 'forward', wraps=self.api.global_ae_model.forward) as forward:
            result = self.api.store_norm_params('regression_participant', payload)
        self.assertEqual(result['status'], 'success')
        point = self.api.write_api.write.call_args.kwargs['record']
        mean, std = baseline_statistics(RESTING_WINDOWS)
        self.assertAlmostEqual(point._fields['sdnn_baseline_std'], std[2])
        self.assertAlmostEqual(point._fields['mean_hr_baseline_mean'], mean[0])
        self.assertEqual(point._fields['baseline_window_count'], 10)
        self.assertEqual(forward.call_count, 6)  # ten minutes -> six full five-minute sequences

    def test_short_calibration_is_rejected_before_any_database_write(self):
        payload = self.api.NormParamsPayload(
            b_mean=OLD_MEAN, b_std=MINIMUM_BASELINE_STDS.tolist(),
            baseline_windows=RESTING_WINDOWS[:3].tolist(),
        )
        with self.assertRaises(self.api.HTTPException) as error:
            self.api.store_norm_params('regression_participant', payload)
        self.assertEqual(error.exception.status_code, 400)
        self.api.write_api.write.assert_not_called()

    def test_personal_error_scale_keeps_calm_p90_below_high(self):
        score = self.api._risk_from_reconstruction_error(2.0, 2.0)
        self.assertAlmostEqual(score, 20.0)
        self.assertAlmostEqual(self.api._risk_from_reconstruction_error(0.0, 2.0), 0.0)
        self.assertGreater(self.api._risk_from_reconstruction_error(8.0, 2.0), score)
        with self.assertRaises(ValueError):
            self.api._risk_from_reconstruction_error(float("nan"), 2.0)
        with self.assertRaises(ValueError):
            self.api._risk_from_reconstruction_error(1.0, 0.0)

    def test_recalibrated_resting_replay_and_escalation_remain_distinguishable(self):
        mean, std = baseline_statistics(RESTING_WINDOWS)
        norm = self.baseline_record(mean, std, baseline_window_count=10)
        resting = self.prediction(norm, RESTING_WINDOWS)
        self.assertEqual(resting['status'], 'success')
        self.assertLess(resting['current_risk_index'], 70)
        self.assertLess(max(resting['risk_forecast']), 95)
        # An independent shifted input must still produce a larger model error;
        # no score caps, recentering on live data, or inflated global floors.
        shifted = RESTING_WINDOWS.copy()
        shifted[:, 0] += 25
        shifted[:, 1] = 60000 / shifted[:, 0]
        shifted[:, 4] += 8
        elevated = self.prediction(norm, shifted)
        self.assertEqual(elevated['status'], 'success')
        self.assertGreater(elevated['current_risk_index'], resting['current_risk_index'])
        self.assertGreater(elevated['current_risk_index'], 90)


if __name__ == '__main__':
    unittest.main()
