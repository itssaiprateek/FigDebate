"""Portability checks: stale paths must not leak into supported launches."""
import os
import unittest
from unittest.mock import patch
import project_environment
import run


class PortableSetupTests(unittest.TestCase):
    def test_stale_external_assets_are_overridden(self):
        with patch.dict(os.environ, {'FIGDEBATE_DATA_ROOT': '/old/dataset',
                                     'HF_HUB_CACHE': '/old/cache'}):
            project_environment.configure()
            self.assertEqual(os.environ['FIGDEBATE_DATA_ROOT'],
                             str(project_environment.ROOT / 'dataset'))
            self.assertEqual(os.environ['HF_HUB_CACHE'],
                             str(project_environment.ROOT / '.cache/huggingface/hub'))

    def test_launcher_preserves_arguments_and_exit_code(self):
        with patch('run.python_path') as python, patch('run.configure'), \
             patch('run.subprocess.call', return_value=7) as call, \
             patch('sys.argv', ['run.py', '--feedback-mode', 'disabled', '--selection-only']):
            python.return_value = project_environment.ROOT / 'run.py'
            self.assertEqual(run.main(), 7)
            self.assertEqual(call.call_args.args[0][-3:],
                             ['--feedback-mode', 'disabled', '--selection-only'])
            self.assertEqual(call.call_args.kwargs['cwd'], project_environment.ROOT)


if __name__ == '__main__':
    unittest.main()
