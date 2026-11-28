"""Catch lazy pandas Styler namespace failures before any training starts."""
import ast
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
AFFECTED = ('pca', 'rfe', 'genetic_algorithm', 'hyperparameters_optimization', 'wgangp')

# Execute actual source definitions with relevant imports in a fresh process,
# without unrelated ML dependencies that might accidentally preload Styler.
PROBE = r'''
import ast
from pathlib import Path
import sys
import tempfile
from typing import get_type_hints
from unittest.mock import patch

import pandas as pd
assert 'pandas.io.formats.style' not in sys.modules, 'Not a cold pandas import'
from utils import table_images
assert 'pandas.io.formats.style' not in sys.modules, 'Renderer preloaded Styler'

path = Path(sys.argv[1])
tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
imports = []
for node in tree.body:
    if isinstance(node, ast.Import):
        if any(alias.name == 'pandas' or alias.name.startswith('pandas.') for alias in node.names):
            imports.append(node)
    elif isinstance(node, ast.ImportFrom):
        if node.module in ('__future__', 'typing', 'pathlib', 'utils.table_images') or (node.module or '').startswith('pandas.'):
            imports.append(node)
functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in
             ('row_style_for_zebra', 'apply_zebra_style', 'export_dataframe_image')]
assert {n.name for n in functions} >= {'apply_zebra_style', 'export_dataframe_image'}
namespace = {'send_exception_via_telegram': lambda *args: None}
exec(compile(ast.Module(body=imports + functions, type_ignores=[]), str(path), 'exec', dont_inherit=True), namespace)
for name in ('apply_zebra_style', 'export_dataframe_image'):
    get_type_hints(namespace[name])
frame = pd.DataFrame({'Feature': ['Rate', 'Dur'], 'F1': [0.98, 0.99]})
styled = namespace['apply_zebra_style'](frame)
from pandas.io.formats.style import Styler
assert isinstance(styled, Styler)
assert 'Rate' in styled.to_html()
with tempfile.TemporaryDirectory() as directory:
    output = Path(directory) / 'results with spaces' / 'table.png'
    with patch.object(table_images, 'ensure_table_images', return_value='native'):
        namespace['export_dataframe_image'](styled, output)
    table_images._verify_png(output)
'''


class StylerImportTests(unittest.TestCase):
    def test_no_lazy_styler_attribute_access_in_repository(self):
        # Inspect executable expressions, not comments or docstrings. Skip venv.
        paths = list(ROOT.glob('*.py'))
        for folder in ('Scripts', 'utils', 'tests'):
            paths.extend((ROOT / folder).rglob('*.py'))
        for path in paths:
            with self.subTest(path=str(path.relative_to(ROOT))):
                tree = ast.parse(path.read_text(encoding='utf-8'))
                for node in ast.walk(tree):
                    if isinstance(node, ast.Attribute) and node.attr == 'Styler':
                        self.assertNotIn(ast.unparse(node),
                                         ('pd.io.formats.style.Styler', 'pandas.io.formats.style.Styler'))

    def test_each_affected_exporter_in_fresh_python(self):
        for name in AFFECTED:
            with self.subTest(module=name):
                result = subprocess.run([sys.executable, '-c', PROBE, str(ROOT / f'{name}.py')],
                                        cwd=ROOT, capture_output=True, text=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
