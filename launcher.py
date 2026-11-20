"""Start or install Pendubot directly from Python, without PowerShell scripts."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser

APP_NAME = 'pendubot-control-lab'
DEPENDENCY_CHECK = 'import jax, numpy, scipy, matplotlib, PIL'


def environment_python(root):
    return root / '.venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')


def prepare_environment(root):
    if sys.version_info < (3, 12):
        raise RuntimeError('Pendubot needs Python 3.12 or newer. Install a current Python version and run the installer again.')
    interpreter = environment_python(root)
    if not interpreter.is_file():
        print('Creating the Pendubot Python environment...', flush=True)
        subprocess.run([sys.executable, '-m', 'venv', str(root / '.venv')], check=True)
    check = subprocess.run([str(interpreter), '-c', DEPENDENCY_CHECK],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if check.returncode:
        print('Installing the simulation libraries. This first setup needs internet access...', flush=True)
        subprocess.run([str(interpreter), '-m', 'pip', 'install', '-r', str(root / 'requirements.txt')], check=True)
        subprocess.run([str(interpreter), '-c', DEPENDENCY_CHECK], check=True)
    (root / '.venv' / 'pendubot-ready.txt').write_text('Pendubot libraries installed\n', encoding='utf-8')
    return interpreter


def probe_server(url, timeout=1):
    # Local traffic must stay local even when a system HTTP proxy is configured.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url + 'api/health', timeout=timeout) as response:
            return json.load(response).get('app') == APP_NAME
    except (OSError, ValueError, urllib.error.URLError):
        return False


def port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
        connection.settimeout(.5)
        return connection.connect_ex(('127.0.0.1', port)) == 0


def wait_for_server(url, timeout=40):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if probe_server(url, timeout=.5):
            return True
        time.sleep(.25)
    return False


def start_server(root, interpreter, port):
    runtime = root / '.runtime'
    runtime.mkdir(exist_ok=True)
    options = dict(cwd=str(root), stdin=subprocess.DEVNULL)
    if os.name == 'nt':
        options['creationflags'] = subprocess.CREATE_NO_WINDOW
    else:
        options['start_new_session'] = True
    with (runtime / 'app.log').open('w', encoding='utf-8') as output, \
            (runtime / 'error.log').open('w', encoding='utf-8') as errors:
        process = subprocess.Popen([str(interpreter), str(root / 'interface.py'), '--port', str(port)],
                                   stdout=output, stderr=errors, **options)
    (runtime / 'server.pid').write_text(str(process.pid), encoding='utf-8')
    return process


def show_interface(url, open_browser=True):
    print('Pendubot is ready: ' + url, flush=True)
    if open_browser:
        try:
            if not webbrowser.open(url):
                print('Open the address above in your browser.', flush=True)
        except (OSError, webbrowser.Error):
            print('Open the address above in your browser.', flush=True)


def launch(root, port=8765, open_browser=True, startup_timeout=40):
    url = f'http://127.0.0.1:{port}/'
    if probe_server(url):
        show_interface(url, open_browser)
        return url
    if port_in_use(port):
        raise RuntimeError(f'Port {port} is already in use by another application. '
                           f'Try: python launcher.py --port {port + 1}')
    interpreter = prepare_environment(root)
    print('Starting Pendubot...', flush=True)
    start_server(root, interpreter, port)
    if not wait_for_server(url, startup_timeout):
        log = root / '.runtime' / 'error.log'
        detail = log.read_text(encoding='utf-8', errors='replace').strip() if log.exists() else ''
        if detail:
            print('\nStartup details:\n' + detail, file=sys.stderr)
        raise RuntimeError('The interface did not become ready. See ' + str(log))
    show_interface(url, open_browser)
    return url


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install', action='store_true', help='Prepare libraries without starting the app')
    parser.add_argument('--no-browser', action='store_true', help='Start without opening a browser')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args(argv)
    if not 1024 <= args.port <= 65535:
        parser.error('Choose a port between 1024 and 65535.')
    root = Path(__file__).resolve().parent
    try:
        if args.install:
            prepare_environment(root)
            print('Pendubot is installed. Open Launch Pendubot.cmd to start.', flush=True)
        else:
            launch(root, args.port, not args.no_browser)
    except (OSError, subprocess.CalledProcessError, RuntimeError) as error:
        print('\nPendubot could not start: ' + str(error), file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
