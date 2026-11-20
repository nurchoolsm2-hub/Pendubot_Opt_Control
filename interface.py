"""Local-only parameter editor and real JAX simulation service."""
import argparse
import csv
from dataclasses import asdict
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import secrets
import threading
import traceback
import webbrowser

ROOT = Path(__file__).resolve().parent
TOKEN = secrets.token_urlsafe(32)
LOCK = threading.Lock()
STATE = {'phase': 'idle', 'message': 'Ready', 'result': None}


def number(key, label, value, low, high, step, help='', unit='', integer=False):
    return dict(key=key, label=label, default=value, min=low, max=high, step=step,
                help=help, unit=unit, type='integer' if integer else 'number')


def choice(key, label, value, options, help=''):
    return dict(key=key, label=label, default=value, options=options, help=help, type='choice')


SCHEMA = [
    dict(title='Simulation', note='The optimizer starts at rest, targeting q1 = 180°, q2 = 0°.', fields=[
        choice('method', 'Optimizer', 'ddp', [['ddp','Full DDP'], ['ilqr','iLQR'], ['both','Compare both']]),
        number('duration', 'Swing-up horizon', 6., 1., 12., .1, 'Time available for the planned maneuver.', 's'),
        number('dt', 'Sample interval', .02, .005, .05, .005, 'Horizon must be an exact multiple of this interval.', 's'),
        number('hold', 'Upright hold time', 5., .5, 10., .5, 'Local LQR is applied after the maneuver.', 's'),
        choice('policy', 'Execution controller', 'feedback', [['feedback','Trajectory feedback + LQR'], ['open_loop','Open loop + LQR']]),
        choice('integration', 'Execution integration', 'fine', [['fine','Fine RK4: 10 substeps'], ['coarse','Same RK4 as optimizer']]),
        choice('initial_guess', 'Initial torque guess', 'sine', [['sine','Sine wave'], ['zero','Zero torque']]),
        number('amplitude', 'Guess amplitude', 4., 0., 20., .5, 'Used only for the sine-wave initial guess.', 'N·m'),
        number('period', 'Guess period', 2., .2, 8., .1, 'Used only for the sine-wave initial guess.', 's'),
        number('max_iterations', 'Maximum iterations', 1500, 10, 3000, 50, 'A small budget can stop before convergence.', integer=True),
    ]),
    dict(title='Robot', note='Uniform rods; only the base joint has a motor. COM and inertia update with mass and length.', fields=[
        number('m1','Link 1 mass',1.,.1,10.,.1,unit='kg'),
        number('m2','Link 2 mass',1.,.1,10.,.1,unit='kg'),
        number('l1','Link 1 length',1.,.1,3.,.1,unit='m'),
        number('l2','Link 2 length',1.,.1,3.,.1,unit='m'),
        number('gravity','Gravity',9.81, .1, 20., .01,unit='m/s²'),
    ]),
    dict(title='Cost weights', note='Larger state weights prioritize that error. Larger torque weight discourages motor effort. Q and QN are diagonal.', fields=[
        *[number('q'+str(i), 'Running Q: '+label, value, 0., 10000., .1)
          for i, label, value in [(0,'q1',1.),(1,'q2',1.),(2,'dq1',.1),(3,'dq2',.1)]],
        *[number('qn'+str(i), 'Terminal QN: '+label, value, .001, 100000., 10.)
          for i, label, value in [(0,'q1',2000.),(1,'q2',2000.),(2,'dq1',200.),(3,'dq2',200.)]],
        number('r','Torque penalty r',.01,.0001,10.,.001,'This is a soft penalty; it is not a hard torque limit.'),
        number('lqr_r','Holding LQR: torque weight',.1,.001,10.,.01,'Holding state weights remain diag(100,100,10,10).'),
    ]),
    dict(title='Disturbances', note='Offsets affect execution only. The planned trajectory always starts at rest. Angles below are in radians.', fields=[
        *[number('offset'+str(i), 'Initial offset: '+label, 0., -1., 1., .01, unit=unit)
          for i, label, unit in [(0,'q1','rad'),(1,'q2','rad'),(2,'dq1','rad/s'),(3,'dq2','rad/s')]],
        number('kick_time','Velocity kick time',3.,0.,12.,.1,'Set both kick velocities to zero to disable. Nonzero kicks must occur before swing-up ends.', 's'),
        number('kick1','Kick in dq1',0.,-2.,2.,.05,unit='rad/s'),
        number('kick2','Kick in dq2',0.,-2.,2.,.05,unit='rad/s'),
    ]),
    dict(title='Solver tolerances', note='Convergence requires small objective reduction and a small feedforward correction.', fields=[
        number('relative_tolerance','Relative cost tolerance',1e-8,1e-12,.001,'any'),
        number('control_tolerance','Control correction tolerance',.001,1e-6,.1,'any',unit='N·m'),
    ]),
]
FIELDS = {f['key']: f for group in SCHEMA for f in group['fields']}
DEFAULTS = {k: f['default'] for k,f in FIELDS.items()}


def validate(raw):
    if not isinstance(raw, dict) or set(raw)-set(FIELDS):
        raise ValueError('Unrecognized parameter settings.')
    values = dict(DEFAULTS)
    for key, value in raw.items():
        field = FIELDS[key]
        if field['type'] == 'choice':
            if value not in [option[0] for option in field['options']]:
                raise ValueError('Choose a valid '+field['label']+'.')
        else:
            if isinstance(value, bool) or not isinstance(value, (int,float)) or not math.isfinite(value):
                raise ValueError(field['label']+' must be a finite number.')
            if not field['min'] <= value <= field['max']:
                raise ValueError(f"{field['label']} must be between {field['min']} and {field['max']}.")
            if field['type'] == 'integer' and value != int(value):
                raise ValueError(field['label']+' must be a whole number.')
            if field['type'] == 'integer':
                value = int(value)
            else:
                value = float(value)
        values[key] = value
    for label in ('duration', 'hold'):
        ratio = values[label]/values['dt']
        if abs(ratio-round(ratio)) > 1e-7:
            raise ValueError(('Swing-up horizon' if label == 'duration' else 'Hold time')+' must be an exact multiple of the sample interval.')
    if round(values['duration']/values['dt']) > 1200:
        raise ValueError('Use at most 1,200 swing-up samples: increase the sample interval or shorten the horizon.')
    if (values['kick1'] or values['kick2']) and values['kick_time'] >= values['duration']:
        raise ValueError('Place the nonzero velocity kick before the end of swing-up.')
    return values


def update(message):
    with LOCK:
        STATE['message'] = message


def execute(solver, nominal, settings):
    import jax.numpy as jnp
    import numpy as np
    from scipy.linalg import solve_discrete_are
    from pendubot import GOAL
    cfg = solver.cfg
    jac = np.asarray(solver.derivatives(jnp.r_[GOAL, 0.][None])[0])
    a, b = jac[:, :4], jac[:, 4:]
    q = cfg.dt*np.diag([100.,100.,10.,10.])
    r = cfg.dt*np.array([[settings['lqr_r']]])
    p = solve_discrete_are(a,b,q,r)
    gain = np.linalg.solve(r+b.T@p@b, b.T@p@a)
    x = np.array([settings['offset'+str(i)] for i in range(4)])
    xs, us = [x.tolist()], []
    count = cfg.n + round(settings['hold']/cfg.dt)
    kick_index = round(settings['kick_time']/cfg.dt)
    integrator = solver.fine_step if settings['integration'] == 'fine' else solver.step
    terminated = False
    for k in range(count):
        if k == kick_index and k < cfg.n:
            x[2:] += [settings['kick1'], settings['kick2']]
            xs[-1] = x.tolist()
        if k < cfg.n:
            u = nominal['u'][k].copy()
            if settings['policy'] == 'feedback':
                u -= nominal['gains'][k]@(x-nominal['x'][k])
        else:
            u = -gain@(x-np.asarray(GOAL))
        new_x = np.asarray(integrator(jnp.array(x),jnp.array(u))).copy()
        if not np.all(np.isfinite(new_x)) or np.max(np.abs(new_x)) > 1e5 or not np.all(np.isfinite(u)):
            terminated = True
            break
        x = new_x
        xs.append(x.tolist())
        us.append(float(u[0]))
    arr = np.array(xs)
    e = arr - np.asarray(GOAL)
    e[:,:2] = (e[:,:2]+np.pi)%(2*np.pi)-np.pi
    arrived = len(arr) > cfg.n
    angle = float(np.max(np.abs(e[cfg.n,:2]))) if arrived else None
    velocity = float(np.max(np.abs(e[cfg.n,2:]))) if arrived else None
    hold_angle = float(np.max(np.abs(e[cfg.n:,:2]))) if arrived else None
    hold_velocity = float(np.max(np.abs(e[cfg.n:,2:]))) if arrived else None
    return dict(x=xs, u=us, terminated=terminated, lqr_gain=gain.tolist(),
                actual_kick_time=kick_index*cfg.dt,
                swing_success=bool(arrived and angle < .05 and velocity < .1),
                hold_success=bool(not terminated and arrived and hold_angle < .1 and hold_velocity < .25),
                arrival_angle_error=angle, arrival_velocity_error=velocity,
                hold_max_angle_error=hold_angle, hold_max_velocity=hold_velocity,
                peak_torque=max(map(abs, us), default=0.))


def worker(settings):
    try:
        update('Preparing model and derivatives…')
        import numpy as np
        from pendubot import Config, Solver
        cfg = Config(**{k:settings[k] for k in ('dt','duration','m1','m2','l1','l2','gravity',
                        'r','max_iterations','relative_tolerance','control_tolerance')},
                     q=tuple(settings['q'+str(i)] for i in range(4)),
                     q_terminal=tuple(settings['qn'+str(i)] for i in range(4)))
        solver = Solver(cfg)
        update('Compiling the model for these settings…')
        solver.warmup()
        t = np.arange(cfg.n)*cfg.dt
        u0 = settings['amplitude']*np.sin(2*np.pi*t/settings['period']) if settings['initial_guess']=='sine' else np.zeros(cfg.n)
        methods = ['ddp','ilqr'] if settings['method']=='both' else [settings['method']]
        solutions, comparison = {}, []
        for method in methods:
            update('Optimizing '+method.upper()+'…')
            result = solver.solve(u0,method,progress_callback=lambda i,c,m=method:
                                  update(f'{m.upper()} · iteration {i} · objective {c:.5g}'))
            solutions[method] = result
            comparison.append(dict(method=method,cost=result['cost'],iterations=result['iterations'],
                                   seconds=result['seconds'],status=result['status'],
                                   terminal_error=float(np.linalg.norm(result['x'][-1]-[np.pi,0,0,0])),
                                   peak_torque=float(np.max(np.abs(result['u'])))))
        selected = min(solutions,key=lambda k:solutions[k]['cost'])
        nominal = solutions[selected]
        update('Executing the trajectory and upright hold…')
        execution = execute(solver,nominal,settings)
        payload = dict(settings=settings,comparison=comparison,selected=selected,execution=execution,
                       nominal_x=nominal['x'].tolist(),nominal_u=nominal['u'][:,0].tolist(),
                       history={key:res['history'] for key,res in solutions.items()},
                       example=False,config=asdict(cfg))
        directory = ROOT/'interface_runs'
        directory.mkdir(exist_ok=True)
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        (directory/(stamp+'.json')).write_text(json.dumps(payload,allow_nan=False),encoding='utf-8')
        (directory/'latest.json').write_text(json.dumps(payload,allow_nan=False),encoding='utf-8')
        with LOCK:
            STATE.update(phase='done',message='Simulation finished',result=payload)
    except Exception as exc:
        traceback.print_exc()
        with LOCK:
            STATE.update(phase='error',message='Simulation could not finish: '+str(exc))


def load_example():
    latest = ROOT/'interface_runs'/'latest.json'
    if latest.exists():
        return json.loads(latest.read_text(encoding='utf-8'))
    import numpy as np
    summary = json.loads((ROOT/'results'/'summary.json').read_text(encoding='utf-8'))
    nominal = np.load(ROOT/'results'/'ddp_sine.npz')
    execution = np.load(ROOT/'results'/'execution_nominal_feedback_fine.npz')
    row = next(r for r in summary['comparison'] if r['method']=='ddp' and r['start']=='sine')
    run = next(r for r in summary['execution'] if r['case']=='nominal' and r['policy']=='feedback' and r['integration']=='RK4_dt/10')
    with (ROOT/'results'/'ddp_sine_convergence.csv').open(newline='',encoding='utf-8') as stream:
        reference_history = [dict(iteration=int(row['iteration']),cost=float(row['cost']))
                             for row in csv.DictReader(stream)]
    return dict(settings=DEFAULTS,selected='ddp',example=True,config=summary['config'],history={'ddp':reference_history},
                nominal_x=nominal['x'].tolist(),nominal_u=nominal['u'][:,0].tolist(),
                comparison=[dict(method='ddp',cost=row['cost'],iterations=row['iterations'],
                                 seconds=row['seconds_excluding_compile'],status=row['status'],
                                 terminal_error=row['terminal_error_norm'],peak_torque=row['peak_torque_nm'])],
                execution=dict(x=execution['x'].tolist(),u=execution['u'].tolist(),terminated=False,
                               swing_success=run['swing_success'],hold_success=run['hold_success'],
                               arrival_angle_error=run['arrival_angle_error_rad'],
                               arrival_velocity_error=run['arrival_velocity_error_rad_s'],
                               peak_torque=run['observed_peak_torque_nm']))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def respond(self, data, content_type='application/json; charset=utf-8', status=200):
        if not isinstance(data, bytes):
            data = json.dumps(data, allow_nan=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type',content_type)
        self.send_header('Content-Length',str(len(data)))
        self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff')
        self.end_headers()
        self.wfile.write(data)

    def allowed_host(self):
        return self.headers.get('Host') in (f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}')

    def do_GET(self):
        if not self.allowed_host():
            return self.respond({'error':'Local access only'},status=403)
        if self.path == '/api/health':
            return self.respond({'app':'pendubot-control-lab','version':1})
        if self.path == '/api/schema':
            return self.respond({'groups':SCHEMA,'defaults':DEFAULTS})
        if self.path == '/api/status':
            with LOCK:
                return self.respond({k:v for k,v in STATE.items() if k!='result'})
        if self.path == '/api/result':
            with LOCK:
                return self.respond(STATE['result'])
        if self.path == '/':
            html = (ROOT/'interface.html').read_text(encoding='utf-8').replace('__TOKEN__',TOKEN)
            return self.respond(html.encode('utf-8'),'text/html; charset=utf-8')
        types = {'/interface.js':'text/javascript; charset=utf-8','/interface.css':'text/css; charset=utf-8'}
        if self.path in types:
            return self.respond((ROOT/self.path[1:]).read_bytes(),types[self.path])
        self.respond({'error':'Not found'},status=404)

    def do_POST(self):
        if not self.allowed_host() or self.headers.get('X-App-Token') != TOKEN:
            return self.respond({'error':'Reload the local interface and try again.'},status=403)
        if self.path != '/api/run':
            return self.respond({'error':'Not found'},status=404)
        try:
            size = int(self.headers.get('Content-Length','0'))
            if size <= 0 or size > 20000:
                raise ValueError('Invalid request size.')
            settings = validate(json.loads(self.rfile.read(size)))
        except (ValueError,TypeError) as exc:
            return self.respond({'error':str(exc)},status=400)
        with LOCK:
            if STATE['phase'] == 'running':
                return self.respond({'error':'A simulation is already running.'},status=409)
            STATE.update(phase='running',message='Starting simulation…')
        threading.Thread(target=worker,args=(settings,),daemon=True).start()
        self.respond({'started':True})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--open',action='store_true')
    args = parser.parse_args()
    STATE['result'] = load_example()
    server = ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    url = f'http://127.0.0.1:{args.port}/'
    print('Pendubot Control Lab: '+url,flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
