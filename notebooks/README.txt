PENDUBOT JUPYTER NOTEBOOKS
==========================

Open in this order:
1. 01_Robot_Dynamics_and_Numerics.ipynb
   Coordinates, parameters, mass matrix, dynamics, RK4, energy, derivatives.
2. 02_DDP_and_iLQR_Code_Walkthrough.ipynb
   Cost, rollout, backward/forward passes, regularization, line search,
   stopping criteria, fresh DDP/iLQR runs and convergence/trajectory plots.
3. 03_Feedback_Results_and_Dashboard.ipynb
   Saved reference results, trajectory feedback, upright LQR, disturbance
   tests, animation, dashboard parameters and request-to-result workflow.

Keep notebooks/ inside the project containing pendubot.py and results/.
Each notebook runs independently. Select Restart Kernel and Run All Cells.
Saved outputs are included; rerunning creates fresh outputs in memory.
No notebook overwrites the reference results or starts the dashboard server.
The first JAX compilation can take longer than later calculations.

WINDOWS (from the project folder)
.venv\Scripts\python.exe -m pip install -r requirements-notebooks.txt
.venv\Scripts\python.exe -m jupyterlab notebooks

UBUNTU / WSL (from your Pendubot_Opt_Control project folder)
python3 -m venv .venv-linux
.venv-linux/bin/python -m pip install -r requirements-notebooks.txt
.venv-linux/bin/python -m jupyterlab notebooks --no-browser
Open the localhost URL printed in the terminal in your Windows browser.
Use a Linux Python environment in Ubuntu; a Windows .venv is not portable.

VS CODE
Open the project folder and an .ipynb file. Select the Python interpreter
from the environment where you installed requirements-notebooks.txt.

If a kernel uses the wrong Python, select the matching environment and
restart the kernel. A kernel must have JAX and the other project packages.
No manual global kernel registration is needed when Jupyter is launched
using the environment's Python as shown above.

When changing planning parameters, rebuild Solver and rerun optimization.
Saved results in Notebook 3 always use the settings in summary.json.
This ideal frictionless simulation has no hard motor torque constraint.
