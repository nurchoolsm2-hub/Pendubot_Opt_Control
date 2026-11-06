"""Pendubot dynamics, RK4, full DDP, and iLQR. Angles are in radians.

q1 is measured from downward vertical; q2 is relative to link 1.
Only the base joint is actuated. All derivatives are of the discrete RK4 map.
"""
from dataclasses import dataclass
import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class Config:
    dt: float = 0.02
    duration: float = 6.0
    # Uniform 1 kg, 1 m rods, COM at 0.5 m, COM inertia 1/12 kg m^2.
    m1: float = 1.0
    m2: float = 1.0
    l1: float = 1.0
    l2: float = 1.0
    gravity: float = 9.81
    q: tuple = (1.0, 1.0, 0.1, 0.1)
    q_terminal: tuple = (2000.0, 2000.0, 200.0, 200.0)
    r: float = 0.01
    max_iterations: int = 1500
    relative_tolerance: float = 1e-8
    control_tolerance: float = 1e-3

    @property
    def n(self):
        return int(round(self.duration / self.dt))


GOAL = jnp.array([jnp.pi, 0.0, 0.0, 0.0])
X0 = jnp.zeros(4)


def mass_matrix(q, cfg=Config()):
    lc1, lc2 = cfg.l1 / 2, cfg.l2 / 2
    i1, i2 = cfg.m1 * cfg.l1**2 / 12, cfg.m2 * cfg.l2**2 / 12
    b = cfg.m2 * cfg.l1 * lc2
    d = i2 + cfg.m2 * lc2**2
    a = i1 + cfg.m1 * lc1**2 + cfg.m2 * cfg.l1**2 + d
    return jnp.array([[a + 2*b*jnp.cos(q[1]), d + b*jnp.cos(q[1])],
                      [d + b*jnp.cos(q[1]), d]])


def dynamics(x, u, cfg=Config()):
    q1, q2, v1, v2 = x
    b = cfg.m2 * cfg.l1 * cfg.l2 / 2
    c = jnp.array([-b*jnp.sin(q2)*(2*v1*v2 + v2*v2),
                   b*jnp.sin(q2)*v1*v1])
    g2 = cfg.m2 * cfg.gravity * cfg.l2/2 * jnp.sin(q1+q2)
    g = jnp.array([(cfg.m1*cfg.l1/2 + cfg.m2*cfg.l1)
                   *cfg.gravity*jnp.sin(q1) + g2, g2])
    rhs = jnp.array([jnp.ravel(jnp.asarray(u))[0], 0.0]) - c - g
    # Explicit 2x2 solve avoids an unnecessary general solver in higher AD.
    m = mass_matrix(x[:2], cfg)
    det = m[0, 0]*m[1, 1] - m[0, 1]*m[1, 0]
    acc = jnp.array([m[1, 1]*rhs[0]-m[0, 1]*rhs[1],
                     -m[1, 0]*rhs[0]+m[0, 0]*rhs[1]]) / det
    return jnp.concatenate((x[2:], acc))


def rk4(x, u, dt, cfg=Config()):
    a = dynamics(x, u, cfg)
    b = dynamics(x + dt*a/2, u, cfg)
    c = dynamics(x + dt*b/2, u, cfg)
    d = dynamics(x + dt*c, u, cfg)
    return x + dt/6*(a + 2*b + 2*c + d)


def energy(x, cfg=Config()):
    kinetic = 0.5 * x[2:] @ mass_matrix(x[:2], cfg) @ x[2:]
    potential = -(cfg.m1*cfg.l1/2 + cfg.m2*cfg.l1)*cfg.gravity*jnp.cos(x[0])
    potential -= cfg.m2*cfg.l2/2*cfg.gravity*jnp.cos(x[0]+x[1])
    return kinetic + potential


class Solver:
    def __init__(self, cfg=Config()):
        self.cfg = cfg
        self.Q = jnp.diag(jnp.array(cfg.q, dtype=jnp.float64))
        self.QN = jnp.diag(jnp.array(cfg.q_terminal, dtype=jnp.float64))
        self.step = jax.jit(lambda x, u: rk4(x, u, cfg.dt, cfg))
        self.fine_step = jax.jit(lambda x, u: jax.lax.fori_loop(
            0, 10, lambda _, z: rk4(z, u, cfg.dt/10, cfg), x))

        def packed(z):
            return rk4(z[:4], z[4:], cfg.dt, cfg)
        self.derivatives = jax.jit(jax.vmap(jax.jacfwd(packed)))
        self.hessians = jax.jit(jax.vmap(jax.jacfwd(jax.jacrev(packed))))
        self.rollout = jax.jit(self._rollout)
        self.cost = jax.jit(self._cost)
        self.backward = jax.jit(self._backward)
        self.forward = jax.jit(self._forward)

    def _rollout(self, u, x0=X0):
        def body(x, uk):
            xn = self.step(x, uk)
            return xn, xn
        _, xs = jax.lax.scan(body, x0, u)
        return jnp.concatenate((x0[None, :], xs))

    def _cost(self, x, u):
        e = x - GOAL  # Deliberately unwrapped: exactly the specified quadratic cost.
        return 0.5*self.cfg.dt*(jnp.einsum('ni,ij,nj->', e[:-1], self.Q, e[:-1])
                                + self.cfg.r*jnp.sum(u*u)) + 0.5*e[-1]@self.QN@e[-1]

    def _backward(self, x, u, jac, hess, mu):
        # z=[x,u]; Hessian tensor axes are [output,z,z].
        def body(carry, data):
            vx, vxx, valid = carry
            xi, ui, fz, fzz = data
            a, b = fz[:, :4], fz[:, 4:]
            lx = self.cfg.dt*self.Q@(xi-GOAL)
            lu = self.cfg.dt*self.cfg.r*ui
            hx, hu = lx+a.T@vx, lu+b.T@vx
            curvature = jnp.einsum('i,ijk->jk', vx, fzz)
            hxx = self.cfg.dt*self.Q + a.T@vxx@a + curvature[:4, :4]
            hux = b.T@vxx@a + curvature[4:, :4]
            huu = self.cfg.dt*self.cfg.r + (b.T@vxx@b)[0, 0] + curvature[4, 4]
            regularized = huu + mu
            ok = jnp.isfinite(regularized) & (regularized > 1e-10)
            denom = jnp.where(ok, regularized, 1.0)
            k, gain = hu/denom, hux/denom
            # Substitute du=-k-K dx in the original (unregularized) local model.
            nvx = hx - gain.T@hu - hux.T@k + (gain.T@k)*huu
            nvxx = hxx - gain.T@hux - hux.T@gain + gain.T@gain*huu
            nvxx = (nvxx+nvxx.T)/2
            return (nvx, nvxx, valid & ok & jnp.all(jnp.isfinite(nvxx))), (k, gain)
        terminal = (self.QN@(x[-1]-GOAL), self.QN, jnp.array(True))
        (_, _, valid), (ks, gains) = jax.lax.scan(
            body, terminal, (x[:-1], u, jac, hess), reverse=True)
        return ks, gains, valid

    def _forward(self, x, u, ks, gains, alpha):
        def body(xnew, data):
            xb, ub, k, gain = data
            unew = ub - alpha*k - gain@(xnew-xb)
            xn = self.step(xnew, unew)
            return xn, (xn, unew)
        _, (xs, us) = jax.lax.scan(body, x[0], (x[:-1], u, ks, gains))
        return jnp.concatenate((x[:1], xs)), us

    def warmup(self):
        """Compile every kernel before reporting optimization runtime."""
        u = jnp.zeros((self.cfg.n, 1))
        x = self.rollout(u)
        z = jnp.concatenate((x[:-1], u), axis=1)
        jac, hess = self.derivatives(z), self.hessians(z)
        ks, gains, _ = self.backward(x, u, jac, hess, 1.0)
        self.cost(*self.forward(x, u, ks, gains, 0.1)).block_until_ready()

    def solve(self, initial_u, method="ddp", verbose=False, progress_callback=None):
        if method not in ("ddp", "ilqr"):
            raise ValueError("method must be 'ddp' or 'ilqr'")
        started = time.perf_counter()
        u = jnp.asarray(initial_u, dtype=jnp.float64).reshape(self.cfg.n, 1)
        x = self.rollout(u)
        cost = float(self.cost(x, u))
        if not np.isfinite(cost):
            raise ValueError("Initial control sequence produced non-finite rollout")
        mu, history, status = 1e-3, [], "iteration_limit"
        for iteration in range(self.cfg.max_iterations):
            z = jnp.concatenate((x[:-1], u), axis=1)
            jac = self.derivatives(z)
            hess = self.hessians(z) if method == "ddp" else jnp.zeros((self.cfg.n, 4, 5, 5))
            accepted = False
            for retry in range(16):
                ks, gains, valid = self.backward(x, u, jac, hess, mu)
                if bool(valid):
                    correction = float(jnp.max(jnp.abs(ks)))
                    for alpha in (1., .5, .25, .125, .0625, .03125, .015625, .0078125):
                        xn, un = self.forward(x, u, ks, gains, alpha)
                        cn = float(self.cost(xn, un))
                        if np.isfinite(cn) and cn < cost:
                            accepted = True
                            break
                    if accepted:
                        break
                    if correction < self.cfg.control_tolerance and mu <= 1e-5:
                        status = "stationary"
                        break
                mu *= 10
                if mu > 1e12:
                    break
            if not accepted:
                if status != "stationary":
                    status = "line_search_failed"
                break
            relative_drop = (cost-cn)/max(1., abs(cost))
            history.append(dict(iteration=iteration+1, cost=cn, relative_drop=relative_drop,
                                correction=correction, alpha=alpha, regularization=mu))
            x, u, cost = xn, un, cn
            mu = max(1e-9, mu/3)
            if verbose and (iteration % 50 == 0):
                print(f"  {method} iter {iteration+1:3d}: J={cost:.6g}, du={correction:.3g}", flush=True)
            if progress_callback is not None and (iteration % 20 == 0):
                progress_callback(iteration+1, cost)
            if relative_drop < self.cfg.relative_tolerance and correction < self.cfg.control_tolerance:
                status = "converged"
                break
        # Recompute gains about the FINAL trajectory, never a stale iterate.
        z = jnp.concatenate((x[:-1], u), axis=1)
        jac = self.derivatives(z)
        hess = self.hessians(z) if method == "ddp" else jnp.zeros((self.cfg.n, 4, 5, 5))
        gain_mu = 1e-9
        for _ in range(24):
            _, gains, valid = self.backward(x, u, jac, hess, gain_mu)
            if bool(valid):
                break
            gain_mu *= 10
        else:
            raise RuntimeError("Could not compute finite final feedback gains")
        return dict(x=np.asarray(x), u=np.asarray(u), gains=np.asarray(gains),
                    cost=cost, history=history, status=status, gain_regularization=gain_mu,
                    iterations=len(history), seconds=time.perf_counter()-started, method=method)
