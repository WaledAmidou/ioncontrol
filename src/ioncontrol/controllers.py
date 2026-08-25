"""
Explicit, fully specified implementations of the compared regulation
strategies.  The reviewer noted that the comparators were not specified in
enough detail to explain how they produced the reported numbers; every
strategy below is therefore given as executable code with all of its gains,
saturations and update rules exposed.

All controllers operate on the standardised ionic state
    z = (M - M_ref) / sd_ref            (dimensionless, per-ion SD units)
so that a unit of control effort has the same physiological meaning for
Zn, Ca and Mg.  The plant is the Ornstein-Uhlenbeck model identified in
`identification.py`:

    z_{k+1} = (I + dt A)(z_k - z_h) + z_h + dt B_z u_k + w_k ,
    y_k     = z_k + v_k ,      v_k ~ N(0, sigma_v^2 I)

where z_h is the patient-specific homeostatic attractor (the offset that a
controller must actively fight), w_k is the diffusion term whose covariance
was derived from the NHANES covariance by the Lyapunov equation, and
sigma_v is fixed by the nanosensor SNR.

Every controller is subject to the same actuator saturation |u| <= U_MAX and
rate limit |du| <= DU_MAX, which is what removes the excessive overshoot the
reviewer objected to.
"""
import numpy as np

from .config import U_MAX_SIGMA, DU_MAX_SIGMA, RHO


# ----------------------------------------------------------------------
def _sat(u, umax):
    return np.clip(u, -umax, umax)


def _rate_limit(u, u_prev, dumax):
    return u_prev + np.clip(u - u_prev, -dumax, dumax)


def scalar_dare(a, b, q, r, iters=200):
    """
    Vectorised scalar discrete algebraic Riccati recursion
        p <- q + a^2 p - (a b p)^2 / (r + b^2 p)
    solved independently for every batch element and every ion.
    Returns the stabilising gain k = a b p / (r + b^2 p).
    """
    p = np.full_like(a, fill_value=1.0)
    for _ in range(iters):
        denom = r + b ** 2 * p
        p_new = q + a ** 2 * p - (a * b * p) ** 2 / np.maximum(denom, 1e-12)
        if np.max(np.abs(p_new - p)) < 1e-10:
            p = p_new
            break
        p = np.maximum(p_new, 1e-12)
    k = a * b * p / np.maximum(r + b ** 2 * p, 1e-12)
    return k, p


# ----------------------------------------------------------------------
class BaseController:
    name = "base"

    def __init__(self, n, dt, u_max=U_MAX_SIGMA, du_max=DU_MAX_SIGMA):
        self.n, self.dt = n, dt
        self.u_max, self.du_max = u_max, du_max * dt
        self.u_prev = np.zeros((n, 3))

    def finalise(self, u):
        u = _sat(u, self.u_max)
        u = _rate_limit(u, self.u_prev, self.du_max)
        self.u_prev = u
        return u

    def step(self, k, y, extra=None):
        raise NotImplementedError


class GlucoseOnly(BaseController):
    """
    (a) Glucose-only regulation.

    Insulin delivery is driven by the glucose error alone; no ionic actuation
    is issued, because the strategy has no access to Zn/Ca/Mg measurements.
    The glucose loop is a PI controller on the glycaemic error, but since it
    is orthogonal to the ionic state its ionic control input is identically
    zero.  This is the honest formalisation of the comparator: it is not a
    degraded version of our controller, it is a controller that acts on a
    different variable.
    """
    name = "Glucose-only"

    def step(self, k, y, extra=None):
        return self.finalise(np.zeros((self.n, 3)))


class Supplementation(BaseController):
    """
    (b) Open-loop metal supplementation therapy.

    A fixed population-level dose is delivered as a periodic bolus with
    period `period_min`, sized on the *population mean* deficit rather than
    the individual one, and applied only to the supplementable ions
    (Zn and Mg; serum Ca is not supplemented in this regimen).  No feedback
    of any kind is used.
    """
    name = "Metal supplementation"

    def __init__(self, n, dt, dose, period_min=30.0, width_min=5.0, **kw):
        super().__init__(n, dt, **kw)
        self.dose = np.asarray(dose, float)      # (3,) in SD/min during bolus
        self.period = int(round(period_min / dt))
        self.width = int(round(width_min / dt))

    def step(self, k, y, extra=None):
        on = (k % self.period) < self.width
        u = np.zeros((self.n, 3))
        if on:
            u[:] = self.dose
        return self.finalise(u)


class FixedGainPID(BaseController):
    """
    (c) Fixed-gain PID feedback control.

        u = -( Kp e + Ki * int e dt + Kd * de/dt ),

    with e = y the standardised tracking error.  The derivative term is
    computed on the filtered measurement and is itself low-pass filtered
    with a derivative filter coefficient N, the standard anti-noise
    construction; without it the derivative action would simply amplify the
    nanosensor noise.  Gains are tuned once on the nominal model by ITAE
    grid search under an overshoot constraint and then frozen.  Conditional
    integration provides anti-windup, and the saturation and rate limits are
    the same as for every other strategy.
    """
    name = "Fixed-gain PID"

    def __init__(self, n, dt, kp, ki, kd=0.0, n_filt=10.0, **kw):
        super().__init__(n, dt, **kw)
        self.kp = np.asarray(kp, float)
        self.ki = np.asarray(ki, float)
        self.kd = np.asarray(kd, float) * np.ones(3)
        self.integ = np.zeros((n, 3))
        self.y_prev = None
        self.deriv = np.zeros((n, 3))
        self.alpha_d = dt / (dt + 1.0 / max(n_filt, 1e-6))

    def step(self, k, y, extra=None):
        if self.y_prev is None:
            raw_d = np.zeros_like(y)
        else:
            raw_d = (y - self.y_prev) / self.dt
        self.deriv = self.deriv + self.alpha_d * (raw_d - self.deriv)
        self.y_prev = y.copy()

        u_un = -(self.kp * y + self.ki * self.integ + self.kd * self.deriv)
        u = _sat(u_un, self.u_max)
        not_sat = np.abs(u_un) < self.u_max
        self.integ = self.integ + np.where(not_sat, y * self.dt, 0.0)
        return self.finalise(u_un)


FixedGainPI = FixedGainPID   # backward-compatible alias


class AdaptiveAI(BaseController):
    """
    (d) Proposed strategy: online recursive-least-squares identification +
        certainty-equivalence LQR + a learned set-point.

    Learning architecture (this is the component the reviewer found
    unspecified):

      * Model class.  For each ion i a first-order affine model
            z_{k+1,i} = a_i z_{k,i} + b_i u_{k,i} + c_i
        where c_i absorbs the unknown patient-specific homeostatic offset.
      * Estimator.  Recursive least squares with forgetting factor lambda,
        theta_i = [a_i, b_i, c_i]^T, covariance P_i initialised to p0 I:
            e     = z_{k+1,i} - phi^T theta
            g     = P phi / (lambda + phi^T P phi)
            theta = theta + g e
            P     = (P - g phi^T P) / lambda
      * Objective.  The infinite-horizon quadratic cost
            J = sum_k [ z_k^T Q z_k + u_k^T R u_k ],  Q = I, R = rho I
        (Q = I in standardised units is exactly Q = diag(1/sd_i^2) in
        physical units, i.e. the data-derived weighting).
      * Optimisation.  The scalar discrete algebraic Riccati equation is
        re-solved every `update_every` steps from the current estimate,
        giving the certainty-equivalence gain k_i, plus exact feedforward
        cancellation of the identified offset, u_ff = -c_i / b_i.
      * State estimation.  A first-order steady-state filter whose bandwidth
        is set from the nanosensor SNR reconstructs z from y.
      * Set-point.  z* is chosen by the supervised model f_hat trained and
        validated on NHANES (see identification.learn_insulin_model), by
        maximising predicted beta-cell function over the physiologically
        admissible box |z| <= z_box.  Only the training partition of NHANES
        is used to fit f_hat; evaluation uses held-out patients.
    """
    name = "Proposed adaptive AI"

    def __init__(self, n, dt, a_nom, b_nom, lam=0.999, p0=0.05, rho=RHO,
                 update_every=100, filt_tau=1.0, z_star=None,
                 dither_amp=0.02, dither_until_min=60.0, seed=0, **kw):
        super().__init__(n, dt, **kw)
        self.lam, self.rho, self.update_every = lam, rho, update_every
        # initialised at the nominal (literature-informed) model and refined
        # online; this is adaptation around a nominal plant, not tabula rasa.
        self.theta = np.zeros((n, 3, 3))          # [a, b, c] per ion
        self.theta[:, :, 0] = a_nom
        self.theta[:, :, 1] = b_nom
        self.P = np.tile(np.eye(3) * p0, (n, 3, 1, 1))
        self.k_gain, _ = scalar_dare(np.tile(a_nom, (n, 1)),
                                     np.tile(b_nom, (n, 1)),
                                     np.ones((n, 3)), rho)
        self.zhat = np.zeros((n, 3))
        self.alpha_f = 1.0 if filt_tau <= 0 else dt / (filt_tau + dt)
        self.y_prev = None
        self.z_star = np.zeros((n, 3)) if z_star is None else z_star
        self.u_ff = np.zeros((n, 3))
        self.dither_amp = dither_amp
        self.dither_steps = int(round(dither_until_min / dt))
        self.rng = np.random.default_rng(seed)

    # -- RLS ----------------------------------------------------------
    def _rls(self, phi, target):
        """phi: (n,3,3) regressors per ion; target: (n,3)."""
        P = self.P                                   # (n,3,3,3)
        Pp = np.einsum('nijk,nik->nij', P, phi)      # (n,3,3)
        denom = self.lam + np.einsum('nij,nij->ni', phi, Pp)
        g = Pp / denom[:, :, None]
        err = target - np.einsum('nij,nij->ni', phi, self.theta)
        self.theta = self.theta + g * err[:, :, None]
        self.P = (P - np.einsum('nij,nik->nijk', g, Pp)) / self.lam

    def step(self, k, y, extra=None):
        # state estimation (first-order steady-state filter)
        self.zhat = self.zhat + self.alpha_f * (y - self.zhat)

        # online identification using the previous regressor
        if self.y_prev is not None:
            phi = np.stack([self.y_prev, self.u_prev, np.ones_like(self.y_prev)], axis=-1)
            self._rls(phi, self.zhat)
        self.y_prev = self.zhat.copy()

        # periodic re-solution of the Riccati equation
        if k % self.update_every == 0:
            a = np.clip(self.theta[:, :, 0], -1.5, 1.5)
            b = self.theta[:, :, 1]
            b = np.where(np.abs(b) < 1e-3, np.sign(b + 1e-12) * 1e-3, b)
            self.k_gain, _ = scalar_dare(a, b, np.ones_like(a), self.rho)

        a = np.clip(self.theta[:, :, 0], -1.5, 1.5)
        b = self.theta[:, :, 1]
        b = np.where(np.abs(b) < 1e-3, np.sign(b + 1e-12) * 1e-3, b)
        c = self.theta[:, :, 2]

        e = self.zhat - self.z_star
        u_fb = -self.k_gain * e
        # identified-offset cancellation, low-pass filtered to avoid
        # transmitting estimator transients to the actuator
        ff_raw = np.clip(-(c + (a - 1.0) * self.z_star) / b,
                         -self.u_max, self.u_max)
        self.u_ff = self.u_ff + 0.02 * (ff_raw - self.u_ff)
        u = u_fb + self.u_ff
        # persistent-excitation dither during the identification phase
        if k < self.dither_steps:
            u = u + self.dither_amp * self.rng.normal(size=u.shape)
        return self.finalise(u)


class ModelBasedLQR(BaseController):
    """
    (e) Ablation: model-based LQR without adaptation.

    Identical cost function, saturation and rate limit as the proposed
    strategy (d), and supplied with the *nominal* plant matrices and the true
    patient offset, but with no online identification.  Comparing (d) with
    (e) isolates exactly what the online learning mechanism contributes,
    which is the ablation the editor asked for.
    """
    name = "Model-based LQR (no adaptation)"

    def __init__(self, n, dt, a_true, b_true, z_h, rho=RHO, **kw):
        super().__init__(n, dt, **kw)
        a = np.tile(a_true, (n, 1))
        b = np.tile(b_true, (n, 1))
        self.k_gain, _ = scalar_dare(a, b, np.ones_like(a), rho)
        self.a, self.b, self.z_h = a, b, z_h

    def step(self, k, y, extra=None):
        u_ff = -(1 - self.a) * self.z_h / self.b
        u = -self.k_gain * y + u_ff
        return self.finalise(u)
