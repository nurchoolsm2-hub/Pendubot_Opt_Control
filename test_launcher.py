"""Regression checks for direct Python startup: python -m unittest test_launcher -v."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import launcher


class LauncherTests(unittest.TestCase):
    def test_existing_server_is_reused_without_installing(self):
        with patch.object(launcher, 'probe_server', return_value=True), \
                patch.object(launcher, 'prepare_environment') as prepare, \
                patch.object(launcher, 'start_server') as start, \
                patch.object(launcher, 'show_interface') as show:
            url = launcher.launch(Path('example'), open_browser=False)
            self.assertEqual(url, 'http://127.0.0.1:8765/')
            prepare.assert_not_called()
            start.assert_not_called()
            show.assert_called_once_with(url, False)

    def test_other_application_is_not_stopped_or_replaced(self):
        with patch.object(launcher, 'probe_server', return_value=False), \
                patch.object(launcher, 'port_in_use', return_value=True), \
                patch.object(launcher, 'start_server') as start:
            with self.assertRaisesRegex(RuntimeError, 'already in use'):
                launcher.launch(Path('example'))
            start.assert_not_called()

    def test_new_server_uses_environment_python_and_waits_for_health(self):
        root = Path('example')
        interpreter = root / '.venv' / 'Scripts' / 'python.exe'
        with patch.object(launcher, 'probe_server', return_value=False), \
                patch.object(launcher, 'port_in_use', return_value=False), \
                patch.object(launcher, 'prepare_environment', return_value=interpreter), \
                patch.object(launcher, 'start_server') as start, \
                patch.object(launcher, 'wait_for_server', return_value=True) as wait, \
                patch.object(launcher, 'show_interface'):
            launcher.launch(root, open_browser=False)
            start.assert_called_once_with(root, interpreter, 8765)
            wait.assert_called_once_with('http://127.0.0.1:8765/', 40)

    def test_failed_start_does_not_claim_ready(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / '.runtime').mkdir()
            (root / '.runtime' / 'error.log').write_text('test startup failure', encoding='utf-8')
            with patch.object(launcher, 'probe_server', return_value=False), \
                    patch.object(launcher, 'port_in_use', return_value=False), \
                    patch.object(launcher, 'prepare_environment'), \
                    patch.object(launcher, 'start_server'), \
                    patch.object(launcher, 'wait_for_server', return_value=False), \
                    patch.object(launcher, 'show_interface') as show:
                with self.assertRaisesRegex(RuntimeError, 'did not become ready'):
                    launcher.launch(root)
                show.assert_not_called()

    def test_browser_failure_still_prints_working_url(self):
        with patch.object(launcher.webbrowser, 'open', side_effect=OSError('no browser')), \
                patch('builtins.print') as output:
            launcher.show_interface('http://127.0.0.1:8765/')
            self.assertIn('Pendubot is ready', output.call_args_list[0].args[0])

    def test_cmd_launchers_do_not_depend_on_powershell(self):
        root = Path(launcher.__file__).resolve().parent
        for name in ('Launch Pendubot.cmd', 'Install Pendubot.cmd'):
            text = (root / name).read_text().lower()
            self.assertNotIn('powershell', text)
            self.assertNotIn('executionpolicy', text)
            self.assertIn('launcher.py', text)


if __name__ == '__main__':
    unittest.main()
