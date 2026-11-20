PENDUBOT SIMULATION - START HERE
===============================

INTERACTIVE INTERFACE (WINDOWS)
Run Install Pendubot.cmd once to create this folder's .venv environment.
Double-click "Launch Pendubot.cmd" to open the local browser interface.
The .cmd launchers start Python directly; PowerShell scripts are not needed.
No Windows execution-policy change is required.
On another computer, run "Install Pendubot.cmd" first (Python 3.12+).
The launcher also checks setup automatically. No account or GPU is needed.

The interface is at http://127.0.0.1:8765/ while the local app is running.
Closing the browser does not stop the app; reopening the launcher reconnects.
The background app ends when Windows is shut down or restarted.

CHANGE PARAMETERS IN THE LEFT PANEL
Simulation: DDP / iLQR / both, horizon, sample interval, upright hold time,
execution policy, fine/coarse integration, initial torque guess, amplitude,
period, and iteration budget.
Robot: link masses, link lengths, and gravity. COM and inertia are derived.
Cost weights: all four diagonal entries of Q and QN, torque penalty r,
and the local LQR torque weight.
Disturbances: initial execution angle/velocity offsets and a velocity kick.
Solver tolerances: relative objective and feedforward correction tolerances.

Click "Run simulation" to apply changes and recompute. While settings are
changed or a run is busy, the animation still shows the previous result.
Use Play, Restart, the time slider and playback speed to inspect motion.
The Optimizer convergence panel plots objective versus accepted iteration.
Choose logarithmic or linear objective scale. Select Compare both before
running to overlay full DDP and iLQR; the plot updates after the run finishes.
Use "Save run data" to export the current settings, trajectories and metrics.
Completed runs are also saved in interface_runs/ and the last one reloads.
Robot and controller settings may cause failures; success is measured in
the results rather than assumed. The optimizer's initial state and upright
target remain [0,0,0,0] and [pi,0,0,0]. Initial offsets affect execution only.
Holding LQR state weights remain diag(100,100,10,10); its gains are computed.

Recommended approach
Use Python + JAX as requested by the project. No robot simulator, CAD,
ROS, or Simulink is needed for this ideal planar two-link model.
Understand the equations first, validate the passive model, then study
full DDP, compare iLQR, and assess feedback and local stabilization.

Files
  pendubot.py          Dynamics, RK4, JAX derivatives, DDP/iLQR solver
  run_simulation.py    Physics checks, experiments, LQR, plots, animation
  DERIVATION.txt       Coordinate definitions, equations and DDP recursion
  requirements.txt    Package requirements
  requirements-lock.txt  Exact package versions used in the verified run
  results/report.html Results, comparison tables, plots and GIF animation
  results/summary.json Full measured metrics and parameters

Run on Windows (PowerShell), from this folder
  python -m venv .venv
  .\.venv\Scripts\python.exe -m pip install -r requirements.txt
  .\.venv\Scripts\python.exe run_simulation.py

Run on macOS/Linux
  python3 -m venv .venv
  .venv/bin/python -m pip install -r requirements.txt
  .venv/bin/python run_simulation.py

Open results/report.html in your browser to view the results. The first
run includes JAX compilation and may take longer. No GPU is required.
To write a new run without replacing the included results:
  python run_simulation.py --output new_results

Parameters to study
Edit Config in pendubot.py for duration, dt, cost weights and iterations.
Edit starts in run_simulation.py to add initial control guesses.
Edit execution() for disturbance amplitude/time; keep the comparison fair.
The supplied equations are frictionless and have no torque saturation.
The animation uses physical angles q1 and q1+q2, not two absolute inputs.

Read the results carefully
- This is a local optimization; a failed start is a useful experimental result.
- Low cost does not guarantee successful upright arrival or stabilization.
- Numerical success criteria are stated in the report and DERIVATION.txt.
- All runs use identical settings across DDP and iLQR.
- A status of iteration_limit means the convergence test was not satisfied.
- Timings are single-run observations, not definitive speed benchmarks.
- Every policy uses the same local LQR after the planned maneuver.
- All controls are sampled and held constant over the sampling interval.
- Values in the report come from the included run; rerunning may slightly
  change timings or trajectories with different hardware or package versions.

Suggested learning sequence
1. Read DERIVATION.txt sections 1-3 and inspect model_checks().
2. Plot an unforced trajectory and inspect energy conservation.
3. Read the DDP recursion alongside Solver._backward().
4. Compare ddp_zero, ilqr_zero, ddp_sine and ilqr_sine.
5. Inspect terminal errors, peak torque, and stopping status, not cost alone.
6. Compare feedback/open-loop under both disturbances and fine integration.
7. Tune one parameter at a time and record successes and failures.

This package is a working foundation for your simulation and understanding.
Review, explain, and adapt it before using it as coursework.

Jupyter code walkthroughs
Three executed notebooks with equations, code explanations, and experiments
are in notebooks/. See notebooks/README.txt for Windows and Ubuntu steps.
Install support: .venv\Scripts\python.exe -m pip install -r requirements-notebooks.txt
Open: .venv\Scripts\python.exe -m jupyterlab notebooks

The 10-slide presentation is in docs/Pendubot Presentation.pptx.
