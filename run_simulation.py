"""Run: python run_simulation.py [--output results] [--max-iterations 1500]."""
import argparse
import csv
import json
from pathlib import Path
import platform
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import solve_discrete_are
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

from pendubot import Config, Solver, GOAL, X0, dynamics, mass_matrix, rk4, energy


def model_checks(solver):
    cfg = solver.cfg
    residuals = [float(jnp.linalg.norm(dynamics(jnp.array([a, b, 0., 0.]), 0., cfg)))
                 for a in (0., np.pi) for b in (0., np.pi)]
    def passive(dt):
        def body(x, _):
            xn = rk4(x, 0., dt, cfg)
            return xn, energy(xn, cfg)
        start = jnp.array([0.5, -0.3, 0.2, -0.1])
        _, es = jax.jit(lambda: jax.lax.scan(body, start, None, length=round(10/dt)))()
        return float(jnp.max(jnp.abs(es-energy(start, cfg))))
    coarse, fine = passive(cfg.dt), passive(cfg.dt/10)
    z = jnp.array([0.4, -0.3, 0.8, -0.4, 0.5])
    jac = np.asarray(solver.derivatives(z[None])[0])
    hess = np.asarray(solver.hessians(z[None])[0])
    eps = 1e-5
    fd = np.column_stack([(np.asarray(solver.step(z[:4]+jnp.eye(5)[i, :4]*eps,
                                                 z[4:]+jnp.eye(5)[i, 4:]*eps))
                           - np.asarray(solver.step(z[:4]-jnp.eye(5)[i, :4]*eps,
                                                    z[4:]-jnp.eye(5)[i, 4:]*eps)))/(2*eps)
                          for i in range(5)])
    hfd = np.stack([(np.asarray(solver.derivatives((z+jnp.eye(5)[i]*eps)[None])[0])
                     - np.asarray(solver.derivatives((z-jnp.eye(5)[i]*eps)[None])[0]))/(2*eps)
                    for i in range(5)], axis=2)
    min_eig = min(float(jnp.min(jnp.linalg.eigvalsh(mass_matrix(jnp.array([0., b]), cfg))))
                  for b in np.linspace(-np.pi, np.pi, 101))
    result = dict(equilibrium_max_residual=max(residuals), mass_matrix_min_eigenvalue=min_eig,
                  energy_drift_coarse_joule=coarse, energy_drift_fine_joule=fine,
                  jacobian_fd_max_error=float(np.max(np.abs(jac-fd))),
                  hessian_fd_max_error=float(np.max(np.abs(hess-hfd))))
    assert max(residuals) < 1e-12 and min_eig > 0
    assert fine < 1e-6 and fine < coarse
    assert result['jacobian_fd_max_error'] < 1e-7
    assert result['hessian_fd_max_error'] < 1e-6
    return result


def make_lqr(solver):
    z = jnp.concatenate((GOAL, jnp.zeros(1)))
    jac = np.asarray(solver.derivatives(z[None])[0])
    a, b = jac[:, :4], jac[:, 4:]
    q = solver.cfg.dt*np.diag([100., 100., 10., 10.])
    r = solver.cfg.dt*np.array([[0.1]])
    p = solve_discrete_are(a, b, q, r)
    gain = np.linalg.solve(r+b.T@p@b, b.T@p@a)
    radius = float(np.max(np.abs(np.linalg.eigvals(a-b@gain))))
    assert radius < 1
    return gain, radius


def execution(solver, nominal, klqr, feedback, fine, case):
    n, dt = solver.cfg.n, solver.cfg.dt
    x = np.zeros(4)
    if case == 'initial_offset':
        x += np.array([0.03, -0.02, 0., 0.])
    xs, us = [x.copy()], []
    termination_time = None
    integrate = solver.fine_step if fine else solver.step
    for k in range(n+round(5/dt)):
        # A velocity jump is an idealized instantaneous disturbance at t=3 s.
        if case == 'velocity_kick' and k == round(3/dt):
            x += np.array([0., 0., 0.15, -0.10])
            xs[-1] = x.copy()
        if k < n:
            u = nominal['u'][k].copy()
            if feedback:
                u -= nominal['gains'][k]@(x-nominal['x'][k])
        else:
            u = -klqr@(x-np.asarray(GOAL))
        with np.errstate(over='ignore', invalid='ignore'):
            x = np.asarray(integrate(jnp.array(x), jnp.array(u))).copy()
        xs.append(x.copy())
        us.append(float(u[0]))
        if not np.all(np.isfinite(x)) or np.max(np.abs(x)) > 1e5:
            termination_time = (k+1)*dt
            xs.extend([np.full(4, np.nan)]*(n+round(5/dt)-k-1))
            us.extend([np.nan]*(n+round(5/dt)-k-1))
            break
    return dict(x=np.asarray(xs), u=np.asarray(us), feedback=feedback, fine=fine, case=case,
                termination_time_s=termination_time)


def errors(x):
    e = np.asarray(x)-np.asarray(GOAL)
    physical = e.copy()
    physical[..., :2] = (physical[..., :2]+np.pi) % (2*np.pi)-np.pi
    return e, physical


def execution_metrics(result, n):
    e, physical = errors(result['x'])
    finite = bool(np.all(np.isfinite(result['x'])))
    angle = float(np.max(np.abs(physical[n, :2])))
    velocity = float(np.max(np.abs(physical[n, 2:])))
    hold_angle = float(np.max(np.abs(physical[n:, :2]))) if finite else None
    hold_velocity = float(np.max(np.abs(physical[n:, 2:]))) if finite else None
    return dict(case=result['case'], policy='feedback' if result['feedback'] else 'open_loop',
                integration='RK4_dt/10' if result['fine'] else 'RK4_dt', finite=finite,
                arrival_angle_error_rad=angle, arrival_velocity_error_rad_s=velocity,
                hold_max_angle_error_rad=hold_angle, hold_max_velocity_rad_s=hold_velocity,
                final_state_error_norm=float(np.linalg.norm(e[-1])) if finite else None,
                observed_peak_torque_nm=float(np.nanmax(np.abs(result['u']))),
                termination_time_s=result['termination_time_s'],
                swing_success=finite and angle < .05 and velocity < .10,
                hold_success=finite and hold_angle < .10 and hold_velocity < .25)


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plots(out, results, best, runs, cfg):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    for key, r in results.items():
        ax.semilogy(np.arange(1, len(r['history'])+1), [h['cost'] for h in r['history']], label=key)
    ax.set(xlabel='Accepted iteration', ylabel='Objective J', title='DDP and iLQR: identical objective and starts')
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    fig.savefig(out/'convergence.png', dpi=160)
    plt.close(fig)
    t = np.arange(cfg.n+1)*cfg.dt
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), layout='constrained', sharex=True)
    axes[0].plot(t, best['x'][:, :2], label=['q1', 'q2'])
    axes[0].axhline(np.pi, color='gray', ls='--', lw=1, label='q1 target')
    axes[0].set(ylabel='Angle (rad)', title=f"Nominal {best['method'].upper()} swing-up")
    axes[1].plot(t, best['x'][:, 2:], label=['dq1', 'dq2'])
    axes[1].set(ylabel='Velocity (rad/s)')
    axes[2].step(t[:-1], best['u'][:, 0], where='post', label='Base motor')
    axes[2].set(xlabel='Time (s)', ylabel='Torque (N m)')
    for ax in axes:
        ax.legend(loc='upper right')
        ax.grid(alpha=.2)
    fig.savefig(out/'trajectory.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), layout='constrained', sharex=True)
    for ax, case in zip(axes, ['nominal', 'initial_offset', 'velocity_kick']):
        for run in runs:
            if run['case'] == case and run['fine']:
                _, e = errors(run['x'])
                ax.plot(np.arange(len(e))*cfg.dt, np.linalg.norm(e[:, :2], axis=1),
                        label='Trajectory feedback + LQR' if run['feedback'] else 'Open loop + LQR')
        ax.axvline(cfg.duration, ls='--', c='gray', label='Switch to local LQR')
        ax.set(ylabel='Angle error norm (rad)', title=case.replace('_', ' ').title())
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    axes[-1].set(xlabel='Time (s)')
    fig.savefig(out/'robustness.png', dpi=160)
    plt.close(fig)


def animation(out, run, cfg):
    # A portable GIF needs no video codec or plotting GUI.
    frames = []
    font = ImageFont.load_default(size=17)
    for k in range(0, len(run['x']), 5):
        x = run['x'][k]
        im = Image.new('RGB', (640, 620), '#101827')
        draw = ImageDraw.Draw(im)
        draw.text((24, 20), 'PENDUBOT | swing-up + stabilization', fill='white', font=font)
        draw.text((24, 50), f't = {k*cfg.dt:.1f} s  |  '+('trajectory feedback' if k < cfg.n else 'local LQR'),
                  fill='#a8b5ca', font=font)
        origin, scale = (320, 330), 112
        draw.line((60, origin[1], 580, origin[1]), fill='#293649', width=1)
        draw.line((origin[0], 90, origin[0], 570), fill='#293649', width=1)
        if np.all(np.isfinite(x)):
            q1, q2 = x[:2]
            p1 = (origin[0]+scale*cfg.l1*np.sin(q1), origin[1]+scale*cfg.l1*np.cos(q1))
            p2 = (p1[0]+scale*cfg.l2*np.sin(q1+q2), p1[1]+scale*cfg.l2*np.cos(q1+q2))
            draw.line((origin, p1), fill='#38bdf8', width=13)
            draw.line((p1, p2), fill='#fb923c', width=13)
            for p in (origin, p1, p2):
                draw.ellipse((p[0]-7, p[1]-7, p[0]+7, p[1]+7), fill='white')
            draw.text((24, 580), f'q1={q1:.3f} rad    q2={q2:.3f} rad', fill='white', font=font)
        frames.append(im)
    frames[0].save(out/'swingup.gif', save_all=True, append_images=frames[1:], duration=100, loop=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='results')
    parser.add_argument('--max-iterations', type=int, default=1500)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    cfg = Config(max_iterations=args.max_iterations)
    solver = Solver(cfg)
    print('Checking physics and discrete-map derivatives...', flush=True)
    checks = model_checks(solver)
    print(json.dumps(checks, indent=2), flush=True)
    compile_start = time.perf_counter()
    print('Compiling JAX kernels (one-time cost)...', flush=True)
    solver.warmup()
    compile_seconds = time.perf_counter()-compile_start
    klqr, radius = make_lqr(solver)
    t = np.arange(cfg.n)*cfg.dt
    starts = {'zero': np.zeros(cfg.n), 'sine': 4*np.sin(2*np.pi*t/2)}
    results, rows, budget_rows = {}, [], []
    for start_name, initial in starts.items():
        for method in ('ddp', 'ilqr'):
            print(f'Optimizing {method.upper()} / {start_name}...', flush=True)
            r = solver.solve(initial, method, verbose=True)
            key = f'{method}_{start_name}'
            results[key] = r
            e, pe = errors(r['x'][-1])
            rows.append(dict(method=method, start=start_name, cost=r['cost'], iterations=r['iterations'],
                             seconds_excluding_compile=r['seconds'], status=r['status'],
                             terminal_error_norm=float(np.linalg.norm(e)),
                             terminal_angle_error_rad=float(np.max(np.abs(pe[:2]))),
                             terminal_velocity_error_rad_s=float(np.max(np.abs(pe[2:]))),
                             peak_torque_nm=float(np.max(np.abs(r['u']))),
                             swing_success=bool(np.max(np.abs(pe[:2])) < .05 and np.max(np.abs(pe[2:])) < .10)))
            assert all(b['cost'] <= a['cost'] for a, b in zip(r['history'], r['history'][1:]))
            np.savez_compressed(out/f'{key}.npz', x=r['x'], u=r['u'], gains=r['gains'])
            if r['history']:
                write_csv(out/f'{key}_convergence.csv', r['history'])
            print(rows[-1], flush=True)
            if len(r['history']) >= 300:
                checkpoint = r['history'][299]
                budget_rows.append(dict(method=method, start=start_name, iteration_budget=300,
                                        cost=checkpoint['cost'], relative_drop=checkpoint['relative_drop'],
                                        correction=checkpoint['correction'],
                                        converged=bool(checkpoint['relative_drop'] < cfg.relative_tolerance
                                                       and checkpoint['correction'] < cfg.control_tolerance)))
    write_csv(out/'comparison.csv', rows)
    if budget_rows:
        write_csv(out/'short_budget_comparison.csv', budget_rows)
    # Select a successful DDP solution first; keep every unsuccessful attempt in comparison.csv.
    candidates = [(k, r) for k, r in results.items() if r['method'] == 'ddp']
    best_key, best = min(candidates, key=lambda pair: pair[1]['cost'])
    runs, execution_rows = [], []
    for fine in (False, True):
        for case in ('nominal', 'initial_offset', 'velocity_kick'):
            for feedback in (False, True):
                r = execution(solver, best, klqr, feedback, fine, case)
                runs.append(r)
                execution_rows.append(execution_metrics(r, cfg.n))
                label = f"{case}_{'feedback' if feedback else 'open_loop'}_{'fine' if fine else 'coarse'}"
                np.savez_compressed(out/f'execution_{label}.npz', x=r['x'], u=r['u'])
    exact = next(r for r in runs if r['case'] == 'nominal' and not r['fine'] and not r['feedback'])
    nominal_replay_error = float(np.max(np.abs(exact['x'][:cfg.n+1]-best['x'])))
    assert nominal_replay_error < 1e-8
    write_csv(out/'execution_comparison.csv', execution_rows)
    selected = next(r for r in runs if r['case'] == 'nominal' and r['fine'] and r['feedback'])
    plots(out, results, best, runs, cfg)
    animation(out, selected, cfg)
    from dataclasses import asdict
    summary = dict(config=asdict(cfg), checks=checks, compilation_seconds=compile_seconds,
                   lqr_gain=klqr.tolist(), lqr_closed_loop_spectral_radius=radius,
                   nominal_replay_max_error=nominal_replay_error, selected=best_key,
                   comparison=rows, short_budget_comparison=budget_rows, execution=execution_rows,
                   versions=dict(python=platform.python_version(), jax=jax.__version__,
                                 numpy=np.__version__, scipy=__import__('scipy').__version__,
                                 matplotlib=matplotlib.__version__), random_seed=None)
    (out/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    make_report(out, summary)
    print(f'Finished. Selected {best_key}; open {out.resolve() / "report.html"}', flush=True)


def make_report(out, summary):
    from html import escape
    def table(rows):
        keys = list(rows[0])
        def val(v):
            if v is None:
                return 'Unavailable'
            return f'{v:.6g}' if isinstance(v, float) else str(v)
        return '<table><tr>'+''.join('<th>'+escape(k.replace('_', ' '))+'</th>' for k in keys)+'</tr>'+''.join(
            '<tr>'+''.join('<td>'+escape(val(row[k]))+'</td>' for k in keys)+'</tr>' for row in rows)+'</table>'
    html = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>Pendubot simulation results</title>
<style>body{font:16px/1.55 system-ui;margin:40px auto;max-width:1120px;padding:0 24px;color:#152238}
h1,h2{line-height:1.2}img{max-width:100%}table{border-collapse:collapse;font-size:12px;display:block;overflow:auto}
th,td{padding:9px;border:1px solid #d8e0ea;text-align:right}th{background:#edf3f9}code{background:#edf3f9;padding:3px}
.note{padding:18px;background:#edf3f9;border-radius:12px}</style>
<h1>Pendubot swing-up and stabilization</h1>
<p>A reproducible numerical simulation using the supplied model, full DDP, iLQR, and sampled local LQR.</p>
<p class="note">This is an ideal, frictionless, unlimited-torque simulation. Terminal penalties are soft.
Results below are measured runs, including failed or unconverged attempts. Physical success and optimizer
convergence are assessed separately. These trials do not establish global convergence or hardware feasibility.</p>
<h2>Setup and success criteria</h2>
<p>Uniform links: m1=m2=1 kg, l1=l2=1 m, COM=0.5 m, COM inertia=1/12 kg m², g=9.81 m/s².
q1 is measured from downward vertical; q2 is relative to link 1. Initial state [0,0,0,0], target [pi,0,0,0].
Horizon 6 s; RK4 step 0.02 s. Q=diag(1,1,0.1,0.1), QN=diag(2000,2000,200,200), r=0.01.
The long horizon permits swing-up; moderate running state penalties favor approaching upright, strong terminal
weights reduce arrival error, and a positive torque penalty discourages excessive actuation.</p>
<p>Swing-up succeeds if each angle error at 6 s is below 0.05 rad and each velocity below 0.10 rad/s.
Holding succeeds if each angle stays below 0.10 rad and each velocity below 0.25 rad/s for the full additional
5 s. Angle errors for physical success are reduced modulo 2pi; optimization uses unwrapped angles as specified.
No random inputs: zero and 4 sin(pi t) N m initial controls are deterministic.</p>
<h2>Model and algorithm verification</h2>CHECKS
<p>Energy drift is the maximum absolute change over a 10 s unforced trajectory starting at [0.5,-0.3,0.2,-0.1].
Finite differences independently check the first and second derivatives of the complete discrete RK4 map.
Only decreasing-cost forward rollouts are accepted.</p>
    <h2>DDP versus iLQR</h2>COMPARISON
<p>Both methods use identical costs, initial controls, horizon and stopping tolerances. Timings exclude explicit
kernel warmup and include derivative evaluation, iteration, and final gain recomputation. Timing is one run per
case on this computer, not a repeated benchmark. First-order JAX startup may still affect timings.</p>
<p>One-time shared kernel warmup: COMPILE s. Relative reduction tolerance 1e-8; maximum unscaled feedforward
correction tolerance 1e-3 N m; both required for ordinary convergence. Maximum MAXITER iterations. A stationary status
means no improving step and a small correction at low regularization. The best-cost DDP trial is selected for
execution: <strong>SELECTED</strong>. See comparison status before treating a trial as converged.</p>
<p>INTERPRETATION</p>
<img src="convergence.png" alt="DDP and iLQR convergence"><img src="trajectory.png" alt="State and torque histories">
<h2>Unsuccessful short-budget optimization attempts</h2>
<p>The table below records the same deterministic runs at 300 accepted iterations. This reproduces an
unsuccessful convergence attempt with a 300-iteration budget: none met both stopping tolerances then.
This is a failure of the convergence test, separate from the question of physically reaching upright.
Allowing more iterations resolved these cases. Intermediate costs alone should not be used to rank final solutions.</p>BUDGET
<h2>Execution and disturbances</h2>
<p>Coarse execution uses the optimizer's RK4 model. Fine execution uses ten RK4 substeps per sample (0.002 s),
holding each control fixed throughout the 0.02 s interval. Feedback uses final nominal trajectory gains.
The initial offset is [0.03,-0.02,0,0]; the separate velocity kick at 3 s is [0,0,0.15,-0.10].
All policies switch at 6 s to the same local discrete LQR: QLQR=diag(100,100,10,10), RLQR=0.1.</p>
<p>Open-loop nominal replay maximum state difference: REPLAY. LQR closed-loop spectral radius: RADIUS.
A stable linearization alone does not prove nonlinear recovery from large disturbances.</p>EXECUTION
<p>In the fine-integration experiments, saved open-loop torques lose upright tracking even without an added
disturbance. Small integration differences are amplified near the unstable equilibrium. Trajectory feedback
passes the arrival and holding criteria for the nominal case and both stated disturbances. These are specific
finite-amplitude tests, not a general robustness guarantee.</p>
<p>Failed open-loop executions are terminated if states become non-finite or their magnitude exceeds 1e5;
the subsequent samples are unavailable. Holding maxima and final errors are unavailable for terminated runs.
Very large observed torques in those failed runs occur after applying local LQR outside its capture region;
they indicate failure and numerical breakdown, not a reliable physical actuator requirement.</p>
<img src="robustness.png" alt="Open loop versus feedback under disturbances">
<h2>Animation</h2><p>Selected nominal DDP trajectory executed using finer integration and feedback, then 5 s of LQR.</p>
<img src="swingup.gif" alt="Pendubot swing-up and local stabilization" width="640">
<h2>Reproduce and inspect</h2><p>Run <code>python run_simulation.py</code> after installing requirements.txt.
Equations and DDP recursion are in DERIVATION.txt; source is in pendubot.py and run_simulation.py.
CSV files contain comparison tables and convergence data; NPZ files contain trajectories, controls and gains.
Full parameters, versions, checks and metrics are in summary.json. Values use SI units.</p>
<p>Sources: the user-supplied project PDF (three pages); <a href="https://docs.jax.dev/en/latest/installation.html">JAX installation</a>;
<a href="https://docs.jax.dev/en/latest/higher-order.html">JAX higher-order derivatives</a>.</p></html>'''
    replacements = dict(CHECKS=table([summary['checks']]), COMPARISON=table(summary['comparison']),
                        MAXITER=str(summary['config']['max_iterations']),
                        BUDGET=table(summary['short_budget_comparison']) if summary['short_budget_comparison'] else '<p>No 300-iteration checkpoint available.</p>',
                        COMPILE=f"{summary['compilation_seconds']:.3f}", SELECTED=summary['selected'],
                        REPLAY=f"{summary['nominal_replay_max_error']:.3g}",
                        RADIUS=f"{summary['lqr_closed_loop_spectral_radius']:.6f}",
                        EXECUTION=table(summary['execution']))
    paired = {start: {r['method']: r for r in summary['comparison'] if r['start'] == start}
              for start in ('zero', 'sine')}
    conclusions = []
    for start, pair in paired.items():
        d, i = pair['ddp'], pair['ilqr']
        conclusions.append(f"For the {start} start, DDP reached J={d['cost']:.6f} in {d['iterations']} iterations "
                           f"({d['seconds_excluding_compile']:.2f} s); iLQR reached J={i['cost']:.6f} in "
                           f"{i['iterations']} iterations ({i['seconds_excluding_compile']:.2f} s).")
    conclusions.append('The additional dynamics curvature did not improve the final objective in these runs, '
                       'and full DDP required more iterations and computation. This is an experimental '
                       'observation for these settings, not a general ordering of the two methods.')
    replacements['INTERPRETATION'] = ' '.join(conclusions)
    for key, value in replacements.items():
        html = html.replace(key, value)
    (out/'report.html').write_text(html, encoding='utf-8')


if __name__ == '__main__':
    main()
