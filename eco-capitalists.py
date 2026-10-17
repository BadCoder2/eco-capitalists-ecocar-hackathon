"""EcoCapitalists: model-predictive racing-line planner + grip-aware speed planner.

Only ``Observation.road`` is used (no map), like the event. Pipeline, every control step:

1. **Corridor** - the road model's centerline plus its boundary polylines give the drivable
   band directly; unmapped obstacle cone groups tighten that band on the cheaper side.
2. **Trajectory** - a lateral-offset profile ``q(s)`` relative to the centerline is optimised on
   the 1 m preview grid by damped Gauss-Newton on a weighted squared-curvature cost (weight
   ``v^4``, i.e. squared lateral grip demand, plus a curvature floor), warm-started from the
   previous step and clipped to the corridor. The car's measured pose is a *hard* constraint on
   the first two nodes, so the plan always leaves the car in its real heading and the smoothness
   cost itself does the blending back onto the racing line - no ad-hoc correction term, no kinks.
3. **Speed plan** - forward/backward pass over the merged curvature window (fresh preview plus
   buffered curvature beyond the sensor horizon) under a friction-circle budget; the steering
   actuator's slew limit enters as a curvature-rate speed cap.
4. **Steering** - gain-corrected curvature feedforward at a speed-dependent lookahead on the
   planned path (the anchor already supplies position/heading feedback), slew-limited to the
   steering actuator, with an online estimate of the steering-to-curvature gain.
5. **Throttle / brake** - the planned longitudinal acceleration at the car's position, with an
   integral correction for actuator gain error and a friction-circle cap; pedals never overlap.

Depends only on numpy and ``autocross.api``.
"""
from __future__ import annotations

import math

import numpy as np

from autocross.api import Command, Controller, Observation, TrackInfo, VehicleParams

G = 9.81


def _cross(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]


def _curvature(pts: np.ndarray) -> np.ndarray:
    """Signed Menger curvature at the interior points of a polyline (+ = turning left)."""
    a, b = pts[1:-1] - pts[:-2], pts[2:] - pts[1:-1]
    den = (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
           * np.linalg.norm(pts[2:] - pts[:-2], axis=1))
    return 2.0 * _cross(a, b) / np.maximum(den, 1e-9)


class EcoCapitalists(Controller):
    # ------------------------------------------------------------------ lifecycle
    def on_start(self, track: TrackInfo, vehicle: VehicleParams) -> None:
        super().on_start(track, vehicle)
        c, vp = self.config, vehicle
        self.dt = float(vp.control_dt)
        self.max_steer = float(vp.max_steer)
        self.du_max = float(vp.max_steer_rate) * self.dt          # per-step command slew limit
        self.mu_g = float(vp.mu) * G
        self.a_max = float(vp.max_accel)
        self.b_max = float(vp.max_decel)
        self.v_max = float(vp.max_speed)
        self.L = float(vp.wheelbase)
        self.slew = float(vp.max_steer_rate) * self.max_steer     # rad/s available at the wheels

        # Tunables; every one can be overridden from a controller-config YAML.
        self.grip_lat = float(c.get("grip_lat", 0.70))            # lateral share of the circle
        self.grip_tot = float(c.get("grip_tot", 0.85))            # combined grip budget
        self.brake_gain = float(c.get("brake_gain", 0.90))
        self.accel_gain = float(c.get("accel_gain", 0.95))
        self.half_w = float(vp.width) / 2.0                       # body half-width
        self.half_L = float(vp.length) / 2.0                      # body half-length
        self.wall_margin = float(c.get("wall_margin", 0.15))      # slack on the body footprint
        self.obs_margin = float(c.get("obstacle_margin", 1.00))
        self.obs_reach = float(c.get("obstacle_reach", 0.35))   # extra cover for the front corner
        self.obs_pad = float(c.get("obstacle_pad", 10.0))
        self.obs_lead = float(c.get("obstacle_lead", 20.0))     # how early the avoidance ramps in
        self.obs_settle = float(c.get("obstacle_settle", 12.0))  # where the line must already be clear
        self.horizon = float(c.get("speed_horizon", 70.0))        # curvature / speed lookahead
        self.it_cold = int(c.get("iters_cold", 12))
        self.it_warm = int(c.get("iters_warm", 4))
        self.w_grip = float(c.get("w_grip", 0.0))                 # extra weight on curvature at speed
        self.w_curv = float(c.get("w_curv", 1.0))                 # base curvature weight
        self.w_kcap = float(c.get("w_curv_cap", 20.0))            # enforcement of the curvature ceiling
        self.w_dev = float(c.get("w_dev", 0.02))                  # weak pull back to the centerline
        self.step_max = float(c.get("max_step", 1.0))             # GN trust region, m per iteration
        self.look_t = float(c.get("look_time", 0.25))
        self.look_min = float(c.get("look_min", 2.5))
        self.ki_a = float(c.get("ki_accel", 0.35))                # integral on accel tracking
        self.gain_rate = float(c.get("gain_rate", 0.03))
        self.grip_cmd_cap = float(c.get("grip_cmd_cap", 1.02))    # max commanded grip demand
        self.max_anchor = float(c.get("max_anchor_slope", 0.6))   # cap on the heading anchor slope
        self.w_commit = float(c.get("w_commit", 4.0))            # hold curvature already committed near the horizon
        self.commit_from = float(c.get("commit_from", 16.0))     # ... ramping in from this distance
        self.off_max = float(c.get("max_offcourse", 0.5))        # how far outside the corridor a plan may start
        self.recov_len = float(c.get("recovery_m", 10.0))        # ... and how far it takes to come back
        self.wall_clear = float(c.get("wall_clear", 0.30))        # room left at the cones for tracking error
        self.v_seed_gain = float(c.get("line_speed_seed", 1.0))  # how far the line is sized for the
        # speed the car could carry there rather than the speed it carried last time
        self.k_track = float(c.get("track_gain", 0.3))           # blend of path-tracking into steering
        self.power_knee = float(c.get("power_knee_mps", 20.0))   # above this the drivetrain is power-limited
        self.brake_floor = float(c.get("brake_floor", 0.55))     # share of max braking kept available in corners
        self.k_smooth = int(c.get("curv_smooth", 7))             # curvature low-pass window, preview nodes
        self.k_lead = float(c.get("curv_lead", 0.5))             # share of that window placed ahead
        self.grip_coup = float(c.get("grip_coupling", 1.0))     # how much cornering eats into braking/accel
        self.accel_cut = float(c.get("accel_lat_cut", 0.50))    # how much cornering eats into drive torque

        self.buf_s: np.ndarray | None = None                      # station-frame curvature / speed
        self.buf_k: np.ndarray | None = None
        self.buf_v: np.ndarray | None = None
        self.buf_q: np.ndarray | None = None
        self.buf_obs: np.ndarray | None = None                    # curvature an obstacle demands
        self.prev_progress: float | None = None
        self.steer_cmd = 0.0
        self.int_a = 0.0
        self.a_cmd_prev = 0.0
        self.st_gain = 1.0
        self.plan_upto = -1.0                                    # absolute station the last plan reached
        self._obs_seen: list = []

    def _init_buffers(self, ds: float) -> None:
        self.k_max = 0.9 * math.tan(self.max_steer) / self.L     # steering limit as a curvature
        if self.buf_s is not None and self.buf_s[1] - self.buf_s[0] == ds:
            return
        n = int(self.horizon / ds) + 1
        self.buf_s = np.arange(n, dtype=float) * ds
        self.buf_k = np.zeros(n)
        self.buf_v = np.zeros(n)
        self.buf_q = np.zeros(n)
        self.buf_obs = np.zeros(n)

    # ------------------------------------------------------------------ corridor
    def _corridor(self, obs: Observation):
        """Centerline, unit left normal and lateral bounds on the preview grid. The bounds are the
        cones minus the car's own half-width minus ``wall_clear``: a line the car cannot track is a
        line that ends with the car in the cones, so the clearance is a constraint, not a preference."""
        road = obs.road
        ps = np.asarray(road.preview_s, float)
        cl = np.asarray(road.preview_centerline, float)
        nrm = np.asarray(road.preview_left_boundary, float) - np.asarray(road.preview_right_boundary, float)
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-6)
        m = self.half_w + self.wall_margin + self.wall_clear
        lo = -(np.asarray(road.preview_width_right, float) - m)
        hi = np.asarray(road.preview_width_left, float) - m
        mid, half = 0.5 * (hi + lo), np.maximum(0.5 * (hi - lo), 0.15)   # usable in tight bands
        return ps, cl, nrm, mid - half, mid + half, mid

    def _apply_obstacles(self, obstacles, ps, lo, hi, q_ref):
        """Push the corridor away from unmapped cone groups, and report the curvature that demands.

        The constraint ramps in over ``obs_lead`` ahead of the obstacle and away again over
        ``obs_pad`` behind it. A long approach matters twice over: the line
        curvature it needs is still small, and the reported demand - the shift still outstanding,
        as a circular arc over the distance left - then stays small enough that the speed plan does
        not have to crawl the car through the gap. Ramping in only 10 m out forces a 6 m radius
        manoeuvre, which is a hairpin to the speed plan and costs several seconds per obstacle.
        """
        k_req = np.zeros_like(ps)
        seen = []
        for ob in obstacles:
            d, lmin, lmax = float(ob.distance), float(ob.lateral_min), float(ob.lateral_max)
            if d > ps[-1] + self.obs_lead:
                continue
            j = int(np.argmin(np.abs(ps - max(d, 0.0))))
            q_here = float(q_ref[j]) if q_ref is not None else 0.0
            settle = d - self.obs_settle                           # the line must be clear by here
            qclip = np.clip(q_ref, lo, hi)
            w = np.clip((ps - (d - self.obs_lead)) / max(self.obs_lead - self.obs_settle, 1.0), 0.0, 1.0)
            # The car's front corner reaches to the side of the line by half the wheelbase times the
            # angle it is crossing at. Where the line is moving across to the next bend past the
            # group, that reach is what clips its near cones, so the clearance covers it too.
            sl_j = float(np.gradient(qclip, ps)[j])
            reach = min(self.obs_reach, self.half_L * abs(sl_j) / np.hypot(1.0, sl_j))
            if q_here >= 0.5 * (lmin + lmax):                   # already committed: pass left of it
                band = lmax + self.obs_margin + reach             # lower bound on q
                lo = np.maximum(lo, lo + w * (band - lo))
                short = band - qclip
            else:                                               # pass right of it
                band = lmin - self.obs_margin - reach             # upper bound on q
                hi = np.minimum(hi, hi - w * (hi - band))
                short = qclip - band
            # Only the outstanding shortfall counts. Reading |band - q| instead charges the plan
            # for the clearance it already has, and near the settle point that residual - a few
            # centimetres over the line - divides by a 2 m room and reads as a hairpin: the speed
            # plan braked a clear straight down to 6 m/s.
            room = np.maximum(settle - ps, 2.0)
            need = np.maximum(short, 0.0) * (ps <= settle)
            np.maximum(k_req, np.minimum(8.0 * need / room ** 2, self.k_max), out=k_req)
            seen.append((d, band))
        self._obs_seen = seen
        return lo, hi, k_req

    # ------------------------------------------------------------------ trajectory optimiser
    def _kappa_jac(self, q, C, N):
        """Curvature at nodes 1..n-2 of p = C + N*q, and its derivative wrt q_{i-1}, q_i, q_{i+1}."""
        p = C + N * q[:, None]
        a, b = p[1:-1] - p[:-2], p[2:] - p[1:-1]
        c = a + b
        n0, n1, n2 = N[:-2], N[1:-1], N[2:]
        la = np.maximum(np.linalg.norm(a, axis=1), 1e-6)
        lb = np.maximum(np.linalg.norm(b, axis=1), 1e-6)
        lc = np.maximum(np.linalg.norm(c, axis=1), 1e-6)
        A, D = _cross(a, b), la * lb * lc
        kap = 2.0 * A / D
        z = np.zeros_like(A)
        dA = (-_cross(n0, b), _cross(n1, c), _cross(a, n2))
        da = (-(a * n0).sum(1) / la, (a * n1).sum(1) / la, z)
        db = (z, -(b * n1).sum(1) / lb, (b * n2).sum(1) / lb)
        dc = (-(c * n0).sum(1) / lc, z, (c * n2).sum(1) / lc)
        dk = [2.0 * (dA[j] * D - A * (lb * lc * da[j] + la * lc * db[j] + la * lb * dc[j])) / (D * D)
              for j in range(3)]
        return kap, dk

    def _cap_terms(self, kap, wk, bias, k_cap):
        """Linearisation of the curvature ceiling for the Gauss-Newton step: past the limit the
        weight jumps and the target becomes the limit itself, so an over-tight node is pulled back
        onto the limit rather than only charged a little more for it."""
        over = np.abs(kap) > k_cap
        if not over.any():
            return wk, bias
        return (wk + np.where(over, self.w_kcap, 0.0),
                bias + np.where(over, self.w_kcap * k_cap * np.sign(kap), 0.0))

    def _cost(self, q, C, N, wA, bias, mid, k_cap):
        """Weighted squared curvature, minus the linear term that pulls committed nodes to their
        already-planned curvature, plus the curvature-ceiling violation and a weak pull to the
        corridor centre - the last only decides which way to round a section whose corridor is not
        symmetric; the bounds do the rest."""
        kap, _ = self._kappa_jac(q, C, N)
        exc = np.abs(kap) - k_cap
        dev = q[2:] - mid[2:]
        return float(np.sum(wA[1:-1] * kap * kap - 2.0 * bias * kap
                            + self.w_kcap * np.where(exc > 0.0, exc * exc, 0.0)
                            + self.w_dev * np.dot(dev, dev)))

    def _optimise(self, q, C, N, lo, hi, wA, bias, mid, iters, k_cap):
        """Damped Gauss-Newton with backtracking; nodes 0 and 1 stay pinned to the car's pose."""
        n = q.size
        if n < 6:
            return q
        m = n - 2
        rows = np.arange(m)                      # row r is the curvature at node r+1
        free = np.arange(2, n)
        diag = np.diag_indices(m)
        lam, cost = 1e-2, self._cost(q, C, N, wA, bias, mid, k_cap)
        for _ in range(iters):
            kap, dk = self._kappa_jac(q, C, N)
            wk, bias_i = self._cap_terms(kap, wA[1:-1], bias, k_cap)
            J = np.zeros((m, n))
            J[rows, rows], J[rows, rows + 1], J[rows, rows + 2] = dk
            Jr = J[:, free]
            H = (Jr.T * wk) @ Jr
            g = Jr.T @ (wk * kap - bias_i) + self.w_dev * (q[free] - mid[free])
            scale = max(float(np.mean(np.diag(H))), 1e-3)
            H[diag] += 2.0 * self.w_dev + lam * scale
            try:
                step = -np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                break
            improved = False
            for shrink in (1.0, 0.5, 0.25):
                trial = q.copy()
                trial[free] = q[free] + np.clip(shrink * step, -self.step_max, self.step_max)
                trial[2:] = np.clip(trial[2:], lo[2:], hi[2:])
                new = self._cost(trial, C, N, wA, bias, mid, k_cap)
                if new < cost - 1e-12:
                    q, cost, improved = trial, new, True
                    lam = max(lam * 0.4, 1e-6)
                    break
                lam = min(lam * 6.0, 1e3)
            if not improved:
                break
        return q

    # ------------------------------------------------------------------ speed plan
    def _speed_plan(self, kappa, ds, v0, k_obs=None):
        """Friction-circle forward/backward pass over the curvature window.

        The curvature is low-passed first: a 1 m-scale wiggle in the planned line is something the
        car physically cannot follow (steering actuator plus tyre slip average it out), and reading
        it as a series of hairpins makes the speed plan crawl. ``k_obs`` is not low-passed: it is the
        curvature an obstacle demands, and braking for an obstacle early is the whole point of it."""
        n = kappa.size
        w = max(int(self.k_smooth), 1)
        if w > 1:
            # Forward-biased, not centred. A centred average halves the apparent curvature at a corner
            # entry, which is exactly where the plan has to already be slow; the plan then discovers the
            # corner when it is 10 m away and no longer reachable. Keeping most of the window ahead keeps
            # the noise rejection but leaves time to brake.
            lead = int(round((w - 1) * self.k_lead))
            kappa = np.convolve(np.pad(kappa, (w - 1 - lead, lead), mode="edge"),
                                np.ones(w) / w, "valid")[:n]
        A_lat, A_tot = self.grip_lat * self.mu_g, self.grip_tot * self.mu_g
        ak = np.abs(kappa)
        v_cap = np.sqrt(A_lat / np.maximum(ak, 1e-4))
        if k_obs is not None:
            v_cap = np.minimum(v_cap, np.sqrt(A_lat / np.maximum(k_obs, 1e-4)))
        dk = np.empty(n)
        dk[1:-1] = (ak[2:] - ak[:-2]) / (2.0 * ds)
        dk[0], dk[-1] = ak[1] - ak[0], ak[-1] - ak[-2]
        v_slew = self.slew * (1.0 + (self.L * kappa) ** 2) / (self.L * np.maximum(np.abs(dk), 1e-5))
        v = np.clip(np.minimum(np.minimum(v_cap, v_slew), self.v_max), 0.3, None)
        for i in range(n - 2, -1, -1):                             # braking into the corner
            # the backend clips braking force per axle independently of lateral demand - what hard
            # braking in a corner costs is lateral grip (the car runs wide), not braking authority.
            # ``grip_coup`` scales that trade-off; 1.0 is the strict friction circle.
            lat = self.grip_coup * min(v[i + 1] * v[i + 1] * ak[i], A_lat)
            a_b = self.brake_gain * math.sqrt(max(A_tot * A_tot - lat * lat,
                                                  (self.brake_floor * A_tot) ** 2))
            v[i] = min(v[i], math.sqrt(v[i + 1] * v[i + 1] + 2.0 * a_b * ds))
        v[0] = min(v[0], v0)
        for i in range(1, n):                                      # acceleration out of it
            lat = self.grip_coup * min(v[i - 1] * v[i - 1] * ak[i], A_lat)
            # the drivetrain is traction-limited below the power knee and power-limited above it,
            # so the plan must not promise the full traction figure at high speed. On top of that
            # this car is rear-biased driven (60%): drive torque while cornering is what pushes the
            # rear axle past its slip limit, so the drive budget falls off with lateral demand far
            # faster than the total friction circle - that is the rear_slip metric, not the circle.
            a_d = min(self.accel_gain * self.a_max,
                      self.accel_gain * self.a_max * self.power_knee / max(v[i - 1], self.power_knee),
                      math.sqrt(max(A_tot * A_tot - lat * lat, (0.20 * A_tot) ** 2)))
            a_d *= self._accel_cut(lat, A_lat)
            v[i] = min(v[i], math.sqrt(v[i - 1] * v[i - 1] + 2.0 * a_d * ds))
        return v

    def _accel_cut(self, lat, A_lat):
        """Drive-torque allowance as a function of how much lateral grip is already being used."""
        u = min(lat / A_lat, 1.0)
        return max(1.0 - self.accel_cut * u * u, 1.0 - self.accel_cut)

    def _fallback(self, obs: Observation, v: float) -> Command:
        """No usable road model (course end / perception off): hold course, ease off the throttle."""
        brake = 0.0 if v < 1.0 else min(0.4, v / 20.0)
        return Command(steer=self.steer_cmd, throttle=0.0, brake=brake, target_speed=0.0)

    # ------------------------------------------------------------------ step
    def step(self, obs: Observation) -> Command:
        road = obs.road
        v = float(max(obs.speed, 0.0))
        if road is None or not road.valid or road.preview_s.size < 12:
            return self._fallback(obs, v)

        ps, cl, nrm, lo, hi, mid = self._corridor(obs)
        ds = float(ps[1] - ps[0])
        n_pre, s_end = ps.size, float(ps[-1])
        self._init_buffers(ds)

        # 1. re-express the station-frame buffers for the distance the car actually covered
        first = self.prev_progress is None
        adv = 0.0 if first else float(np.clip(obs.progress - self.prev_progress, 0.0, 8.0))
        self.prev_progress = float(obs.progress)
        for attr in ("buf_k", "buf_v", "buf_q", "buf_obs"):
            buf = getattr(self, attr)
            setattr(self, attr, np.interp(self.buf_s + adv, self.buf_s, buf, right=buf[-1]))

        # 2. warm-start the line, tighten the corridor around obstacles, pin it to the car's pose
        q = np.interp(ps, self.buf_s, self.buf_q, right=self.buf_q[-1])
        # The bound is relaxed near the car. When the car has run outside it the plan has to start
        # there; clipping it to the safe band at the first node would put a kink in the line, and
        # the speed plan reads a kink as a hairpin and crawls the car off the course. The line is
        # brought back over the recovery distance instead of all at once.
        slack = self.off_max * np.exp(-ps / self.recov_len)
        lo, hi = lo - slack, hi + slack
        # The car is 4.8 m long: on a line that changes side quickly it sits at an angle to the
        # cones and its corners reach further sideways than its half-width. The clearance therefore
        # follows the slope of the line the car is actually driving, which is last step's plan.
        sl = np.gradient(np.clip(q, lo, hi), ps)
        extra = np.minimum(self.half_L * np.abs(sl) / np.hypot(1.0, sl),
                           np.maximum(0.5 * (hi - lo) - 0.3, 0.0))
        lo, hi = lo + extra, hi - extra
        # Obstacle bands last. Their clearance is physical - a cone inside the car's footprint is a
        # penalty - while the relaxation above is a licence to *start* outside the safe corridor,
        # not to drive through an obstacle group. Applied before them the relaxation cancels the
        # band exactly where the car is, which is where the clearance is actually needed: the plan
        # asked for 0.39 m past the group and drove it at -0.4.
        lo, hi, k_obs = self._apply_obstacles(road.obstacles, ps, lo, hi, np.clip(q, lo, hi))
        mid = 0.5 * (hi + lo)
        q = np.clip(q, lo, hi)
        q_car = float(np.clip(road.lateral_offset, lo[0], hi[0]))
        slope = math.tan(float(road.heading_error)) + float(road.curvature) * q_car
        q[0] = q_car
        q[1] = float(np.clip(q_car + ds * np.clip(slope, -self.max_anchor, self.max_anchor), lo[1], hi[1]))

        # 3. optimise the line: minimum squared curvature inside the corridor, plus a hold on the
        #    curvature already committed beyond what is still visible. Without the hold the far end
        #    of the window is free, so every step rewrites the corners it has not reached yet: the
        #    line flips side to side and the speed plan inherits the cliffs.
        # The speed the line is sized for. Seeding it with the previous plan's speed is a fixed
        # point in the wrong direction: a tight line gives a slow plan, a slow plan loosens both
        # the grip cap and the speed-weighted curvature penalty, and the line stays tight. Seeding
        # with the speed the car would reach going flat out to each station asks the line for what
        # it would have to look like to hold that speed - which is what a racing line is.
        v_flat = np.minimum(np.sqrt(v * v + 2.0 * self.a_max * ps), self.v_max)
        v_warm = np.maximum(np.interp(ps, self.buf_s, self.buf_v, left=self.buf_v[0], right=self.buf_v[-1]),
                            v_flat * self.v_seed_gain)
        A_lat = self.grip_lat * self.mu_g
        wA = ds * (self.w_grip * (v_warm ** 4) / (A_lat * A_lat) + self.w_curv)
        # The line has to be one the car can actually hold at the speed planned for it: the grip
        # budget at the warm-started speed, and the steering limit, cap its curvature. Without the
        # cap the optimiser kinks the line where the car has run wide, and the speed plan then asks
        # for a deceleration the car cannot deliver.
        k_cap = np.minimum(A_lat / np.maximum(v_warm[1:-1], 1.0) ** 2, self.k_max)
        bias = np.zeros(n_pre - 2)
        if not first:
            kt = np.interp(ps, self.buf_s, self.buf_k, right=self.buf_k[-1])
            d = ps[1:-1]
            ramp = np.clip((d - self.commit_from) / max(s_end - self.commit_from, 1.0), 0.0, 1.0)
            ramp *= d <= self.plan_upto - float(obs.progress)
            wA[1:-1] += self.w_commit * ramp
            bias = self.w_commit * ramp * kt[1:-1]
        self.plan_upto = float(obs.progress) + s_end
        q = self._optimise(q, cl, nrm, lo, hi, wA, bias, mid, self.it_cold if first else self.it_warm, k_cap)

        # 4. the path and its curvature on the preview grid
        path = cl + nrm * q[:, None]
        k_t = np.empty(n_pre)
        k_t[1:-1] = _curvature(path)
        k_t[0], k_t[-1] = k_t[1], k_t[-2]

        # 5. buffer it (beyond the preview the previous plan's curvature stands in for the
        #    not-yet-visible course) and plan speed over the merged window
        self.buf_k[:n_pre] = k_t
        self.buf_q[:n_pre] = q
        self.buf_obs[:n_pre] = k_obs
        if s_end < float(road.range_m) - 0.5 * ds:                # preview truncated: course ends
            self.buf_k[n_pre:] = 0.0
            self.buf_q[n_pre:] = q[-1]
        n_win = int(min(self.buf_s[-1], max(s_end + 25.0, 30.0)) / ds) + 1
        n_win = min(n_win, self.buf_k.size)
        v_plan = self._speed_plan(self.buf_k[:n_win], ds, v, self.buf_obs[:n_win])
        self.buf_v[:n_win] = v_plan

        # 6. steering: curvature feedforward blended with pure-pursuit tracking of our own path.
        #    Feedforward alone cannot drive out an error: the plan is re-anchored on the car every
        #    step, so whatever the car did wrong is absorbed into the plan instead of corrected.
        #    Aiming at the plan point ahead is what makes the car actually follow the line.
        s_ff = min(max(self.look_min, self.look_t * v), s_end - ds)
        j = int(np.clip(round(s_ff / ds), 1, n_pre - 2))
        u_ff = math.atan(self.L * float(k_t[j]) * self.st_gain) / self.max_steer
        tgt = path[j]
        ld = float(max(math.hypot(float(tgt[0]), float(tgt[1])), 1.0))
        u_pp = math.atan(2.0 * self.L * math.sin(math.atan2(float(tgt[1]), float(tgt[0]))) / ld) / self.max_steer
        u = (1.0 - self.k_track) * u_ff + self.k_track * u_pp
        if v > 1.0:                                               # never ask for more grip than exists
            cap = min(math.tan(min(self.grip_cmd_cap * self.mu_g / (v * v), 1.2)) * self.L / self.max_steer, 1.0)
            u = float(np.clip(u, -cap, cap))
        u = float(np.clip(u, self.steer_cmd - self.du_max, self.steer_cmd + self.du_max))
        applied = float(obs.steer)
        self.steer_cmd = u

        # 7. identify the steering-to-curvature gain while the actuator is steady and the car is sliding little
        if v > 6.0 and abs(obs.vy) < 1.0 and abs(applied) > 0.15 and abs(applied - self.steer_cmd) < 2.0 * self.du_max:
            k_ach = obs.yaw_rate / v
            k_cmd = math.tan(applied * self.max_steer) / self.L
            if k_ach * k_cmd > 0.0 and abs(k_cmd) > 0.02:
                r = k_cmd / k_ach
                if 0.5 < r < 2.5:
                    self.st_gain = float(np.clip(self.st_gain + self.gain_rate * (r - self.st_gain), 0.8, 1.4))

        # 8. throttle / brake from the planned longitudinal acceleration, with actuator learning
        a_ref = (v_plan[1] * v_plan[1] - v * v) / (2.0 * ds)
        self.int_a = float(np.clip(self.int_a + (obs.ax - self.a_cmd_prev) * self.dt, -2.5, 2.5))
        a_cmd = a_ref + self.ki_a * self.int_a
        lat = min(v * v * abs(float(k_t[0])), A_lat)
        # The plan budgets longitudinal grip against the lateral demand of the line it is following.
        # The command has to honour the same budget: a deadbeat speed tracker otherwise brakes at full
        # while the corner is already using grip, and ~0.9 of the friction circle is exactly where this
        # car's rear axle lets go (the rear_slip metric). brake_floor keeps a share of braking available
        # at full lateral demand, as the measured envelope shows it can be.
        a_tot = self.grip_tot * self.mu_g
        circle = math.sqrt(max(a_tot * a_tot - lat * lat, 0.0))
        b_avail = self.brake_gain * max(circle, self.brake_floor * a_tot)
        a_pow = self.accel_gain * self.a_max * min(1.0, self.power_knee / max(v, 0.5))
        a_cmd = float(np.clip(a_cmd, -b_avail, min(self.a_max * self.accel_gain, a_pow,
                                                    max(circle, 0.3 * a_tot)) * self._accel_cut(lat, A_lat)))
        self.a_cmd_prev = a_cmd
        if abs(a_cmd) < 0.25:
            throttle = brake = 0.0
        elif a_cmd > 0.0:
            throttle, brake = min(a_cmd / (self.a_max * self.accel_gain), 1.0), 0.0
        else:
            throttle, brake = 0.0, min(-a_cmd / (self.b_max * self.brake_gain), 1.0)
        return Command(steer=u, throttle=throttle, brake=brake, path=path, path_frame="vehicle",
                      target_speed=float(v_plan[1]),
                      debug={"v_plan": float(v_plan[1]), "a_ref": a_ref, "q_car": q_car,
                             "kappa_ff": float(k_t[j]), "st_gain": self.st_gain,
                             "k_obs": float(k_obs[0]), "hi0": float(hi[0]), "lo0": float(lo[0]),
                             "obs_q": float(np.interp(self._obs_seen[0][0], ps, q)) if self._obs_seen else 0.0,
                             "obs_band": float(self._obs_seen[0][1]) if self._obs_seen else 0.0})
