"""Regression coverage for shared browser provisioning and atomic PNG output."""
import ast
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from PIL import Image

from utils import table_images as images

ROOT = Path(__file__).resolve().parents[1]
EXPORTERS = ('main', 'pca', 'rfe', 'genetic_algorithm', 'hyperparameters_optimization',
             'wgangp', 'stacking', 'dataset_descriptor')


class TableImagesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.addCleanup(setattr, images, '_backend', images._backend)
        images._backend = None
        self.output = self.directory / 'directory with spaces' / 'result.png'
        self.styled = pd.DataFrame({'Feature': ['Rate', 'Dur'], 'F1': [.996123, .987654],
                                    'Parameters': ['x' * 200, '{"n_jobs": 1}']}).style.format({'F1': '{:.4f}'})

    def test_browser_path_is_venv_local(self):
        with patch.object(sys, 'prefix', str(self.directory / 'venv with spaces')):
            with patch.dict(os.environ, {'PLAYWRIGHT_BROWSERS_PATH': '0', 'PIP_USER': '1'}):
                self.assertEqual(images._environment()['PLAYWRIGHT_BROWSERS_PATH'],
                                 str((self.directory / 'venv with spaces' / 'playwright-browsers').resolve()))
                self.assertEqual(images._environment()['PIP_USER'], '0')

    def test_healthy_browser_skips_install(self):
        with patch.object(images, '_ensure_package'), patch.object(images, '_probe') as probe, \
                patch.object(images, '_install_chromium') as install:
            self.assertEqual(images._prepare_browser(), 'managed')
            probe.assert_called_once_with('managed')
            install.assert_not_called()

    def test_missing_browser_downloaded_then_verified(self):
        with patch.object(images, '_ensure_package'), \
                patch.object(images, '_probe', side_effect=[RuntimeError('missing'), None]) as probe, \
                patch.object(images, '_install_chromium') as install:
            self.assertEqual(images._prepare_browser(), 'managed')
            install.assert_called_once()
            self.assertEqual(probe.call_count, 2)

    def test_linux_libraries_repaired_then_verified(self):
        with patch.object(sys, 'platform', 'linux'), patch.object(images, '_ensure_package'), \
                patch.object(images, '_probe', side_effect=[RuntimeError('missing'), RuntimeError('libraries'), None]), \
                patch.object(images, '_install_chromium') as install, \
                patch.object(images, '_install_linux_dependencies') as deps:
            self.assertEqual(images._prepare_browser(), 'managed')
            install.assert_called_once()
            deps.assert_called_once()

    def test_no_download_access_uses_installed_browser(self):
        with patch.object(images, '_ensure_package'), \
                patch.object(images, '_probe', side_effect=[RuntimeError('missing'), None]), \
                patch.object(images, '_install_chromium', side_effect=RuntimeError('network denied')), \
                patch.object(images, '_system_backends', return_value=iter(['channel:chrome'])):
            self.assertEqual(images._prepare_browser(), 'channel:chrome')

    def test_no_browser_uses_verified_native_backend(self):
        with patch.object(images, 'browser_directory', return_value=self.directory), \
                patch.object(images, '_prepare_browser', side_effect=RuntimeError('no privileges')) as setup:
            self.assertEqual(images.ensure_table_images(), 'native')
            self.assertEqual(images.ensure_table_images(), 'native')
            setup.assert_called_once()
            self.assertEqual(images.render_table_image(self.styled, self.output), 'native')
        with Image.open(self.output) as img:
            self.assertEqual(img.format, 'PNG')
            self.assertGreater(img.width, 100)
            self.assertGreater(img.height, 50)

    def test_runtime_browser_failure_falls_back_atomically(self):
        with patch.object(images, 'ensure_table_images', return_value='managed'), \
                patch.object(images, '_browser_subprocess', side_effect=RuntimeError('browser removed')):
            self.assertEqual(images.render_table_image(self.styled, self.output), 'native')
        images._verify_png(self.output)
        self.assertEqual(list(self.output.parent.iterdir()), [self.output])

    def test_genuine_failure_preserves_previous_output(self):
        self.output.parent.mkdir(parents=True)
        self.output.write_bytes(b'previous result')
        with patch.object(images, 'ensure_table_images', return_value='native'), \
                patch.object(images, '_native_render', side_effect=OSError('disk full')):
            with self.assertRaisesRegex(OSError, 'disk full'):
                images.render_table_image(self.styled, self.output)
        self.assertEqual(self.output.read_bytes(), b'previous result')
        self.assertEqual(list(self.output.parent.iterdir()), [self.output])

    def test_style_formatting_and_all_rows_reach_renderer(self):
        captured = []
        real_native = images._native_render
        def record(html, png):
            captured.append(html.read_text(encoding='utf-8'))
            real_native(html, png)
        with patch.object(images, 'ensure_table_images', return_value='native'), \
                patch.object(images, '_native_render', side_effect=record), \
                pd.option_context('styler.render.max_elements', 1):
            images.render_table_image(self.styled, self.output)
        for value in ('0.9961', '0.9877', 'Rate', 'Dur', 'x' * 200):
            self.assertIn(value, captured[0])
        self.assertNotIn('class="data row_trim', captured[0])

    def test_help_does_not_install(self):
        with patch.object(sys, 'argv', ['pca.py', '--help']), patch.object(images, 'ensure_table_images') as ensure:
            images.preflight_table_images()
            ensure.assert_not_called()

    def test_cli_install_commands_use_active_interpreter(self):
        with patch.object(images, '_run') as run:
            images._install_chromium()
            self.assertEqual(run.call_args.args[0], [sys.executable, '-m', 'playwright', 'install', 'chromium'])
        with patch.object(images, '_run', side_effect=[RuntimeError('missing module'), None]) as run, \
                patch.object(sys, 'prefix', str(self.directory)), patch.object(sys, 'base_prefix', 'system'):
            images._ensure_package()
            self.assertEqual(run.call_args.args[0][:5], [sys.executable, '-m', 'pip', 'install', '--no-user'])
            self.assertIn(images._package_requirement(), run.call_args.args[0])

    def test_dependency_setup_wired_to_every_exporter(self):
        for name in EXPORTERS:
            with self.subTest(name=name):
                tree = ast.parse((ROOT / f'{name}.py').read_text(encoding='utf-8'))
                functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
                for function, called in [('main', 'preflight_table_images'), ('export_dataframe_image', 'render_table_image')]:
                    self.assertTrue(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                                        and n.func.id == called for n in ast.walk(functions[function])))
                self.assertFalse(any(isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
                                     and n.value.id == 'dfi' and n.attr == 'export' for n in ast.walk(tree)))
        make = (ROOT / 'Makefile').read_text()
        self.assertIn('"$(PYTHON)" -m utils.table_images', make)

    @unittest.skipUnless(shutil.which('chromium'), 'System Chromium is unavailable')
    def test_real_system_chromium_screenshot(self):
        backend = f'executable:{shutil.which("chromium")}'
        with patch.object(images, 'ensure_table_images', return_value=backend):
            self.assertEqual(images.render_table_image(self.styled, self.output), backend)
        images._verify_png(self.output)

    @unittest.skipUnless(os.environ.get('DDOS_TEST_MANAGED_BROWSER') == '1', 'Requires managed browser download')
    def test_real_managed_browser_install_and_render(self):
        self.assertEqual(images.ensure_table_images(), 'managed')
        self.assertEqual(images.render_table_image(self.styled, self.output), 'managed')
        images._verify_png(self.output)


if __name__ == '__main__':
    unittest.main()
