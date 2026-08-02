import os
from typing import Tuple

import numpy as np
import osqp
from scipy import sparse
import matplotlib.pyplot as plt

# Colors
PREDICTION = '#BA4A00'

##################
# MPC Controller #
##################


def solved(dec) -> bool:
    """Did OSQP actually converge?

    Defensive about the osqp version: status_val 1 is OSQP_SOLVED, and some builds report
    'solved' / 'solved inaccurate' only as a string. An unreadable status is treated as
    NOT solved, which costs one relaxation attempt and never a wrong accept.
    """
    info = getattr(dec, "info", None)
    if info is None:
        return False
    val = getattr(info, "status_val", None)
    if val is not None:
        return val == 1
    return str(getattr(info, "status", "")).strip().lower() == "solved"


class MPC:
    def __init__(self, model, N, Q, R, QN, StateConstraints, InputConstraints,
                 ay_max, max_steering_rate, wp_id_offset, use_obstacle_avoidance, use_path_constraints_topic, use_max_kappa_pred=True):
        """
        Constructor for the Model Predictive Controller.
        :param model: bicycle model object to be controlled
        :param N: time horizon | int
        :param Q: state cost matrix
        :param R: input cost matrix
        :param QN: final state cost matrix
        :param StateConstraints: dictionary of state constraints
        :param InputConstraints: dictionary of input constraints
        :param ay_max: maximum allowed lateral acceleration in curves
        :param wp_id_offset: offset for waypoint id to consider control delay
        :param use_obstacle_avoidance: flag to enable obstacle avoidance
        :param use_path_constraints_topic: flag to use path constraints from topic
        :param max_steering_rate: maximum allowed steering rate in rad/s
        """
        # 既存の初期化パラメータ
        self.N = N
        self.Q = Q
        self.R = R
        self.QN = QN
        self.wp_id_offset = wp_id_offset
        self.use_obstacle_avoidance = use_obstacle_avoidance
        self.use_path_constraints_topic = use_path_constraints_topic
        self.model = model
        self.nx = self.model.n_states
        self.nu = 2
        self.state_constraints = StateConstraints
        self.input_constraints = InputConstraints
        self.ay_max = ay_max

        # 追加: ステアリングレート制限関連のパラメータ
        self.max_steering_rate = max_steering_rate
        self.previous_steering = 0.0  # 前回のステア角

        # 追加: ay_maxによる速度制限の方式切り替え
        self.use_max_kappa_pred = use_max_kappa_pred
        # Corridor funnel. The usable e_y band is NOT max_width: update_path_constraints
        # subtracts safety_margin = width/sqrt(2) = 1.061 for width 1.50, so max_width 3.0
        # leaves +/-1.94 m. Step 0 is pinned to the measured e_y by equality, and step 1's
        # e_y is fully determined by it (the e_y row of B is zero, see
        # spatial_bicycle_models.linearize), so once the car is past ~1.94 m the QP has no
        # feasible point at all -- not a poor solution, none. OSQP then returns an
        # unconverged iterate, the command stays at full section speed, and the car
        # accelerates into the wall until stuck_recovery reverses it, still outside the
        # band and pointing wrong, so it re-wedges. Measured: one lap pinballed for 32 s
        # across seven such cycles.
        #
        # So the excursion is not what costs 20-70 s; being unable to PLAN a return is.
        # The funnel admits the car's current offset at the near steps and closes back to
        # the nominal band at funnel_rate per step, which keeps the problem feasible and
        # turns a pinball into a planned rejoin. The cost reference is deliberately left at
        # the original band centre so the objective still pulls toward the line.
        # Env-switched rather than config-switched so the two arms of an A/B differ by
        # nothing except this flag -- routing it through the yaml would also change which
        # config file each arm loads.
        # docker-compose passes `${VAR:-}` for anything unset, i.e. the empty string
        # rather than an absent variable, so os.environ.get's default never fires and
        # float("") raises. Treat blank as unset.
        def _env(name, default):
            raw = (os.environ.get(name) or "").strip()
            return default if raw == "" else raw

        # DEFAULT ON as of the 12-race sweep recorded below. Set MPC_FUNNEL=0 to disable.
        #
        #   arm            laps  6-lap finishes  mean 6-lap total
        #   funnel+anti      68        9/12            338.3
        #   antideadlock     67       10/12            367.6
        #   base             32        4/12            313.1
        #
        # 12 races in the scored format (online3.sh: 3 cars, collisions on, handicap on,
        # ranking on, 6 laps, 480 s), grid rotated every race. base is faster when it
        # finishes but finishes only in grid slot 3 (4/4 there, 2/8 elsewhere).
        self.funnel_enabled = _env("MPC_FUNNEL", "1").lower() not in ("0", "false")
        self.funnel_slack = float(_env("MPC_FUNNEL_SLACK", "0.20"))
        # 0.15, not 0.25: measured to beat both 0.25 and 0.08 (tools/GOAL.md) -- the funnel
        # has to open, but it also has to close, and 0.25 closes too hard while 0.08 barely
        # closes at all.
        self.funnel_rate = float(_env("MPC_FUNNEL_RATE", "0.15"))

        # Lateral error past which corridor segment selection is seeded from the
        # reference waypoint rather than the car. 0 disables the fallback.
        self.recovery_e_y = float(_env("MPC_RECOVERY_E_Y", "1.0"))

        # 既存の初期化
        self.current_prediction = None
        self.debug_funnel = 0.0
        self.infeasibility_counter = 0
        self.last_solved_wp_id = 0
        self.current_control = np.zeros((self.nu*self.N))
        self.optimizer = osqp.OSQP()

        if not self.use_obstacle_avoidance:
            self.model.reference_path.update_simple_path_constraints(
                N,
                self.model.safety_margin)

    def update_v_max(self, v_max: float):
        self.input_constraints['umax'][0] = v_max

    def update_ay_max(self, ay_max: float):
        self.ay_max = ay_max

    def update_wp_id_offset(self, wp_id_offset: int):
        self.wp_id_offset = wp_id_offset

    def update_Q(self, Q: np.ndarray):
        self.Q = Q

    def update_R(self, R: np.ndarray):
        self.R = R

    def update_QN(self, QN: np.ndarray):
        self.QN = QN

    def _init_problem(self, N, safety_margin):
        """
        Initialize optimization problem for current time step with steering rate constraints.
        """
        # 既存の制約設定
        umin = self.input_constraints['umin']
        umax = self.input_constraints['umax']
        xmin = self.state_constraints['xmin']
        xmax = self.state_constraints['xmax']

        # Precompute common terms
        nx_N = self.nx * (N + 1)
        nu_N = self.nu * N

        # LTV System Matrices
        A = np.zeros((nx_N, nx_N))
        B = np.zeros((nx_N, nu_N))

        # Reference vector
        ur = np.zeros(nu_N)
        xr = np.zeros(nx_N)
        uq = np.zeros(N * self.nx)

        # Dynamic constraints
        xmin_dyn = np.kron(np.ones(N + 1), xmin)
        xmax_dyn = np.kron(np.ones(N + 1), xmax)
        umax_dyn = np.kron(np.ones(N), umax)

        # Get curvature predictions
        kappa_pred = np.tan(np.append(np.array(self.current_control[3::self.nu]), self.current_control[-1])) / self.model.length
        # current_control can hold an unconverged OSQP iterate (bounds are only
        # satisfied at convergence), which produced |kappa_pred| up to 0.952
        # against the physical input bound tan(delta_max)/L = 0.668 and pinned
        # vmax_dyn at sqrt(20/0.952) = 4.58 m/s (measured). Clip to the bound.
        kappa_bound = abs(self.input_constraints['umax'][1])
        kappa_pred = np.clip(kappa_pred, -kappa_bound, kappa_bound)

        kappa_ref_horizon = np.zeros(N)
        delta_s_horizon = np.zeros(N)

        # Consider control delay. Snapshot/restore: _init_problem is called again by the
        # relaxation retry, and an unguarded += would build each retry's problem 1.2 m
        # further ahead than the frame x0/e_y0 were measured in.
        wp_id_entry = self.model.wp_id
        self.model.wp_id += self.wp_id_offset

        # Iterate over horizon
        for n in range(N):
            # Get waypoint information
            current_waypoint = self.model.reference_path.get_waypoint(self.model.wp_id + n)
            next_waypoint = self.model.reference_path.get_waypoint(self.model.wp_id + n + 1)
            delta_s = next_waypoint - current_waypoint
            kappa_ref = current_waypoint.kappa
            kappa_ref_horizon[n] = kappa_ref
            delta_s_horizon[n] = delta_s

            # Clip reference velocity
            v_ref = np.clip(current_waypoint.v_ref, self.input_constraints['umin'][0], self.input_constraints['umax'][0])

            # Compute LTV matrices
            f, A_lin, B_lin = self.model.linearize(v_ref, kappa_ref, delta_s)
            A[(n+1) * self.nx: (n+2)*self.nx, n * self.nx:(n+1)*self.nx] = A_lin
            B[(n+1) * self.nx: (n+2)*self.nx, n * self.nu:(n+1)*self.nu] = B_lin

            # Set reference
            ur[n*self.nu:(n+1)*self.nu] = [v_ref, kappa_ref]
            uq[n * self.nx:(n+1)*self.nx] = B_lin.dot([v_ref, kappa_ref]) - f

            # Constrain maximum speed based on curvature
            if self.use_max_kappa_pred:
                max_kappa_pred = np.max(np.abs(kappa_pred[n:]))
                vmax_dyn = np.sqrt(self.ay_max / (np.abs(max_kappa_pred) + 1e-12))
            else:
                vmax_dyn = np.sqrt(self.ay_max / (np.abs(kappa_pred[n]) + 1e-12))
            umax_dyn[self.nu*n] = min(vmax_dyn, umax_dyn[self.nu*n])

            # Enforce the per-waypoint speed profile as a hard input cap.
            # The R[0] term only *pulls* u toward v_ref, and the time cost
            # Q[2] dominates it, so the solver otherwise rides
            # min(umax, vmax_dyn) and ignores v_ref entirely (measured:
            # v_ref0=8.31 while u0=11.11). The flat ref_vel mode only ever
            # worked because update_v_max() turned the section speed into
            # this same constraint globally; there v_ref == umax[0], so this
            # line is a no-op in that mode.
            umax_dyn[self.nu*n] = min(umax_dyn[self.nu*n], v_ref)

        # Debug telemetry for the speed pipeline (read by mpc_controller):
        # v_ref actually fed to the solver at step 0, the dynamic speed cap at
        # step 0 and its minimum over the horizon.
        self.debug_v_ref0 = ur[0]
        self.debug_umax_dyn0 = umax_dyn[0]
        self.debug_umax_dyn_min = float(np.min(umax_dyn[::self.nu]))
        # kappa actually fed to the vmax_dyn formula (prev solution's planned
        # steering, NOT path curvature) and the path's own kappa at step 0.
        self.debug_kpred_max = float(np.max(np.abs(kappa_pred)))
        self.debug_kappa_wp0 = float(
            self.model.reference_path.get_waypoint(self.model.wp_id).kappa)

        # Update path constraints
        if self.use_obstacle_avoidance and not self.use_path_constraints_topic:
            # `pose` seeds which free segment of the corridor gets picked at the head of
            # the horizon (reference_path.py:973); everything downstream follows that
            # choice. Seeding it with the car's real position is right while driving, but
            # once the car is wedged against a wall it selects the pocket the car is stuck
            # in rather than the track, and the plan stays there. Passing None seeds from
            # the waypoint instead, i.e. the way back to the line.
            #
            # Measured 2026-08-02, per-event recovery: 9.9 s median with constraints off,
            # 23.5 s with them computed from the real pose. Threshold is in metres of
            # lateral error; the corridor half-width is about 1.7 m.
            off_line = abs(float(self.model.spatial_state.e_y)) > self.recovery_e_y
            seed_pose = None if off_line else [
                self.model.temporal_state.x, self.model.temporal_state.y,
                self.model.temporal_state.psi]
            ub, lb, _ = self.model.reference_path.update_path_constraints(
                self.model.wp_id + 1, seed_pose,
                N, self.model.length, self.model.width, safety_margin)
        else:
            ref_wp_id = (self.model.wp_id + 1) % len(self.model.reference_path.path_constraints[0])
            # .copy() is load-bearing. set_path_constraints stores (n_wp-1, N) arrays and
            # this indexing returns a ROW VIEW, so the `ub -= safety_margin_diff` below
            # writes through into persistent storage. Nothing restores it: the submission
            # launch path (control/mpc.launch.xml) sets use_obstacle_avoidance=false and
            # never starts path_constraints_provider, so these bounds are built once by
            # update_simple_path_constraints and never republished. Worse, the diff is taken
            # against the constant model.safety_margin every call, so the retry loop ADDS
            # widening instead of setting it -- three retries take a stored bound from 1.90
            # to 3.17 m, permanently, for that waypoint's whole 20-column window, on a
            # circular track. The car then plans through the wall on later laps, and the
            # funnel below silently disables itself at stuck locations because e_y0 no
            # longer exceeds the corrupted ub[0].
            ub = self.model.reference_path.path_constraints[0][ref_wp_id].copy()
            lb = self.model.reference_path.path_constraints[1][ref_wp_id].copy()
            self.model.reference_path.border_cells.current_wp_id = ref_wp_id

            # Update safety margin if provided as argument and different from current value
            if self.model.safety_margin != safety_margin:
                safety_margin_diff = safety_margin - self.model.safety_margin
                ub -= safety_margin_diff
                lb += safety_margin_diff

                infeasible_index = ub < lb
                ub[infeasible_index] = 0.0
                lb[infeasible_index] = 0.0

        # Update dynamic state constraints
        e_y0 = float(self.model.spatial_state.e_y)

        # Keep the cost reference at the ORIGINAL band centre. If it were recomputed after
        # the funnel opens the corridor, the objective would pull the car further out.
        xr_e_y = (lb + ub) / 2

        over = 0.0
        if not self.funnel_enabled:
            pass
        elif e_y0 > ub[0]:
            over = e_y0 - ub[0]
        elif e_y0 < lb[0]:
            over = e_y0 - lb[0]
        if over != 0.0:
            # A relief that decays from k=0 is infeasible exactly in the case it exists
            # for. The plan is effectively a single constant-curvature arc: the
            # steering-rate rows bound consecutive planned curvatures by
            # scaled_steer_rate_max*Ts = 0.0053, so the whole horizon can only sweep 0.107
            # of the +/-0.668 input range. With e_y1 = e_y0 + delta_s*e_psi0 pinned (the
            # e_y row of B is zero) and e_psi unbounded, a car yawed outward keeps moving
            # outward for several steps no matter what u0 is. Decaying relief therefore
            # binds at step 2 for outward yaw beyond ~9 degrees on a straight and ~5 in a
            # curve -- i.e. for every post-contact pose.
            #
            # So forecast the excursion under maximum-effort recovery and keep the bound
            # ahead of it, only then closing at funnel_rate.
            e_psi0 = float(self.model.spatial_state.e_psi)
            sgn = 1.0 if over > 0.0 else -1.0
            psi = max(0.0, sgn * e_psi0)          # outward yaw only
            e_pred = abs(over)
            relief = np.zeros(len(ub))
            kappa_bound = abs(self.input_constraints['umax'][1])
            close_cap = self.funnel_rate / max(delta_s_horizon[0], 1e-6)
            for k in range(len(ub)):
                relief[k] = max(e_pred + self.funnel_slack,
                                abs(over) + self.funnel_slack - self.funnel_rate * k,
                                0.0)
                ds = delta_s_horizon[min(k, N - 1)]
                e_pred += ds * psi
                authority = ds * max(kappa_bound - abs(kappa_ref_horizon[min(k, N - 1)]),
                                     0.0)
                psi = max(psi - authority, -close_cap)
            if over > 0.0:
                ub = ub + relief
            else:
                lb = lb - relief
            self.debug_funnel = float(abs(over))
        else:
            self.debug_funnel = 0.0

        xmin_dyn[0] = xmax_dyn[0] = e_y0
        xmin_dyn[self.nx::self.nx] = lb
        xmax_dyn[self.nx::self.nx] = ub
        xr[self.nx::self.nx] = xr_e_y

        # Get equality matrix
        Ax = sparse.kron(sparse.eye(N + 1), -sparse.eye(self.nx)) + sparse.csc_matrix(A)
        Bu = sparse.csc_matrix(B)
        Aeq = sparse.hstack([Ax, Bu])

        # ステアリングレート制約の行列を構築
        n_rate_constraints = N - 1
        steering_rate_matrix = np.zeros((n_rate_constraints, nx_N + nu_N))

        # ステアリングレート制約の行列を設定
        for i in range(n_rate_constraints):
            # 連続する制御入力間の差分に対する係数を設定
            steering_rate_matrix[i, nx_N + self.nu*i + 1] = -1  # 現在のステア角
            steering_rate_matrix[i, nx_N + self.nu*(i+1) + 1] = 1  # 次のステア角

        # 制約行列の結合
        A_inequality = sparse.vstack([
            sparse.eye(nx_N + nu_N),  # 状態と入力の基本的な制約
            sparse.csc_matrix(steering_rate_matrix)  # ステアリングレート制約
        ])

        # 完全な制約行列
        A_full = sparse.vstack([Aeq, A_inequality], format='csc')

        # 境界制約の構築
        x0 = np.array(self.model.spatial_state[:])
        leq = np.hstack([-x0, uq])
        ueq = leq

        # 入力と状態の制約境界
        lineq_basic = np.hstack([xmin_dyn, np.kron(np.ones(N), umin)])
        uineq_basic = np.hstack([xmax_dyn, umax_dyn])

        # ステアリングレート制約の境界
        max_delta_change = self.max_steering_rate * self.model.Ts
        lineq_rate = -max_delta_change * np.ones(n_rate_constraints)
        uineq_rate = max_delta_change * np.ones(n_rate_constraints)

        # 全ての境界を結合
        l = np.hstack([leq, lineq_basic, lineq_rate])
        u = np.hstack([ueq, uineq_basic, uineq_rate])

        # コスト行列
        P = sparse.block_diag([
            sparse.kron(sparse.eye(N), self.Q),
            self.QN,
            sparse.kron(sparse.eye(N), self.R)
        ], format='csc')

        q = np.hstack([
            -np.tile(np.diag(self.Q.toarray()), N) * xr[:-self.nx],
            -self.QN.dot(xr[-self.nx:]),
            -np.tile(np.diag(self.R.toarray()), N) * ur
        ])

        # オプティマイザの設定
        self.optimizer = osqp.OSQP()
        self.optimizer.setup(P=P, q=q, A=A_full, l=l, u=u, verbose=False)

        # Restore the frame. Without this each relaxation retry builds its problem
        # wp_id_offset further ahead than the frame x0/e_y0 were measured in, and
        # update_prediction then reads the drifted id too.
        self.model.wp_id = wp_id_entry

    def get_control(self) -> Tuple[np.ndarray, float]:
        """
        Get control signal given the current position of the car.
        """
        nx = self.nx
        nu = self.nu

        self.model.get_current_waypoint()

        N = min(self.N, self.model.reference_path.n_waypoints - self.model.wp_id) \
            if not self.model.reference_path.circular else self.N

        self.model.spatial_state = self.model.t2s(
            reference_state=self.model.temporal_state,
            reference_waypoint=self.model.current_waypoint)

        self._init_problem(N, self.model.safety_margin)

        # NOTE: unconverged OSQP iterates (max-iter/infeasible) are still
        # accepted as control here on purpose. Rejecting them and commanding
        # zero was tried and it deadlocks: a car stopped off-line keeps the QP
        # infeasible forever, and the stuck-recovery node never fires because
        # the nominal command is zero. The garbage they used to inject into
        # the speed cap is handled by the kappa_pred clip in _init_problem.
        try:
            dec = self.optimizer.solve()
            control_signals = np.array(dec.x[-N*nu:])

            # The old trigger was `not np.all(control_signals[1::2])` -- the truthiness of
            # floats -- so it only fired when a planned curvature was EXACTLY 0.0, never for
            # the unconverged iterates it exists to handle. But simply asking the solver
            # instead costs far too much: OSQP rarely converges fully here, so the loop ran
            # up to five extra _init_problem + solve rounds every tick and the measured
            # control rate fell from 40 Hz to 10.45 Hz, which wedged the car within one lap
            # (33 stuck events, reverse commanded). _init_problem rebuilds the whole sparse
            # system in a Python loop over the horizon, so it is not cheap enough to repeat.
            #
            # So: retry at most ONCE, and only when the car is genuinely outside the
            # corridor, which is the only case where relaxing the margin can help. The
            # funnel already restores feasibility there, making this a second line of
            # defence rather than the mechanism.
            # MPC_FUNNEL=0 must reproduce the SHIPPED behaviour exactly, including this
            # trigger. `np.all(control_signals[1::2])` is not as dead as it looks: it is
            # False whenever ANY planned curvature is exactly 0.0, which does happen on
            # straights, and the relaxation that follows was evidently doing useful work.
            # Gating the retry on the funnel alone made the control arm of the A/B a third
            # configuration rather than the baseline -- it deadlocked after one lap, twice,
            # with zero stuck-recovery events, which is the signature of a zero command the
            # recovery node cannot see.
            if self.funnel_enabled:
                retry = self.debug_funnel > 0.0 and not solved(dec)
                first = 5          # ONE attempt: range(5, 6). first=4 ran two.
            else:
                retry = not np.all(control_signals[1::2])
                first = 1
            if retry:
                for i in range(first, 6):
                    relaxed_safety_margin = self.model.safety_margin * ((5-i) / 5.0)
                    self._init_problem(N, relaxed_safety_margin)
                    dec = self.optimizer.solve()
                    control_signals = np.array(dec.x[-N*nu:])

                    ok = (solved(dec) if self.funnel_enabled else
                          (self.infeasibility_counter == 0
                           and np.all(control_signals[1::2])))
                    if ok:
                        if self.last_solved_wp_id != self.model.wp_id:
                            print(f"Relaxed safety margin by {relaxed_safety_margin} ({5-i}/5) to solve the problem")
                        break

            # ステア角の計算と保存
            control_signals[1::2] = np.arctan(control_signals[1::2] * self.model.length)
            v = control_signals[0]
            delta = control_signals[1]

            # ステアレートの制限を適用
            max_delta_change = self.max_steering_rate * self.model.Ts
            delta = np.clip(
                delta,
                self.previous_steering - max_delta_change,
                self.previous_steering + max_delta_change
            )
            self.previous_steering = delta

            # 予測の更新
            self.current_control = control_signals
            x = np.reshape(dec.x[:(N+1)*nx], (N+1, nx))
            self.current_prediction = self.update_prediction(x, N)

            u = np.array([v, delta])
            max_delta = np.max(np.abs(control_signals[1:len(control_signals)//3*2:2]))

            if self.infeasibility_counter > (N - 1):
                print(f'Problem solved after {self.infeasibility_counter} infeasible iterations')
            self.infeasibility_counter = 0
            self.last_solved_wp_id = self.model.wp_id

        except (TypeError, ValueError):
            # note: this was 'except TypeError or ValueError' which Python
            # evaluates to 'except TypeError' only.
            id = nu * (self.infeasibility_counter + 1)
            if id + 2 < len(self.current_control):
                u = np.array(self.current_control[id:id+2])
                max_delta = np.abs(u[1])
            else:
                u = np.array([0.0, 0.0])
                max_delta = 0.0

            self.infeasibility_counter += 1

        if self.infeasibility_counter > (N - 1) and self.infeasibility_counter % 100 == 0:
            print('No control signal computed!')

        return u, max_delta

    def update_prediction(self, spatial_state_prediction, N):
        """
        Transform the predicted states to predicted x and y coordinates.
        Mainly for visualization purposes.
        :param spatial_state_prediction: list of predicted state variables
        :return: lists of predicted x and y coordinates
        """

        # Containers for x and y coordinates of predicted states
        x_pred, y_pred = [], []

        # Iterate over prediction horizon
        for n in range(2, N):
            # Get associated waypoint
            associated_waypoint = self.model.reference_path.\
                get_waypoint(self.model.wp_id+n)
            # Transform predicted spatial state to temporal state
            predicted_temporal_state = self.model.s2t(associated_waypoint,
                                            spatial_state_prediction[n, :])

            # Save predicted coordinates in world coordinate frame
            x_pred.append(predicted_temporal_state.x)
            y_pred.append(predicted_temporal_state.y)

        return x_pred, y_pred

    def show_prediction(self, ax):
        """
        Display predicted car trajectory on the provided axis.
        :param ax: Matplotlib axis object to plot on
        """

        if self.current_prediction is not None:
            # ax.scatter(self.current_prediction[0], self.current_prediction[1],
            #            c=PREDICTION, s=5)
            ax.plot(self.current_prediction[0], self.current_prediction[1], c=PREDICTION)
