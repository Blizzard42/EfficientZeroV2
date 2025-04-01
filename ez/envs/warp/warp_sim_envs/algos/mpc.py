# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# MPC toolbox
#
###########################################################################

# from warp.tests.grad_utils import check_tape_safety, check_backward_pass, check_jacobian, function_jacobian_fd
from warp.optim import Adam, SGD
from warp_sim_envs.envs import RenderMode, Environment
from warp_sim_envs.utils import function_jacobian_fd
import sys
import numpy as np
import warp as wp
from enum import Enum

import os

from tqdm import trange

import matplotlib.pyplot as plt

CHECK_NANS = False
DEBUG_PLOTS = True
DEBUG_GRADIENTS = False

# wp.config.verify_cuda = True
# wp.config.mode = "debug"
# wp.config.verify_fp = True


# Types of control point interpolation
class InterpolationMode(Enum):
    INTERPOLATE_HOLD = "hold"
    INTERPOLATE_LINEAR = "linear"
    INTERPOLATE_CUBIC = "cubic"

    def __str__(self):
        return self.value


if DEBUG_PLOTS:
    import pyqtgraph as pg
    from PyQt5 import QtWidgets

    class MainWindow(QtWidgets.QMainWindow):
        def __init__(self, *args, **kwargs):
            super(MainWindow, self).__init__(*args, **kwargs)

            self.layoutWidget = pg.LayoutWidget()
            self.layoutWidget.addWidget

            # self.graphWidget = pg.PlotWidget()
            self.graphWidget1 = pg.GraphicsLayoutWidget()
            self.graphWidget2 = pg.GraphicsLayoutWidget()
            self.layoutWidget.addWidget(self.graphWidget1)
            self.layoutWidget.addWidget(self.graphWidget2)
            self.setCentralWidget(self.layoutWidget)


@wp.kernel
def replicate_maximal_states(
    body_q_in: wp.array(dtype=wp.transform),
    body_qd_in: wp.array(dtype=wp.spatial_vector),
    bodies_per_env: int,
    # outputs
    body_q_out: wp.array(dtype=wp.transform),
    body_qd_out: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()
    env_offset = tid * bodies_per_env
    for i in range(bodies_per_env):
        body_q_out[env_offset + i] = body_q_in[i]
        body_qd_out[env_offset + i] = body_qd_in[i]


@wp.kernel
def replicate_generalized_states(
    joint_q_in: wp.array(dtype=float),
    joint_qd_in: wp.array(dtype=float),
    # outputs
    joint_q_out: wp.array(dtype=float),
    joint_qd_out: wp.array(dtype=float),
):
    tid = wp.tid()
    env_offset_q = tid * joint_q_in.shape[0]
    for i in range(joint_q_in.shape[0]):
        joint_q_out[env_offset_q + i] = joint_q_in[i]
    env_offset_qd = tid * joint_qd_in.shape[0]
    for i in range(joint_qd_in.shape[0]):
        joint_qd_out[env_offset_qd + i] = joint_qd_in[i]


@wp.kernel
def sample_gaussian(
    mean_trajectory: wp.array(dtype=float, ndim=3),
    resampling_noise_std: float,
    num_control_points: int,
    control_dim: int,
    control_limits: wp.array(dtype=float, ndim=2),
    # outputs
    seed: wp.array(dtype=int),
    rollout_trajectories: wp.array(dtype=float, ndim=3),
):
    env_id, point_id, control_id = wp.tid()
    unique_id = (env_id * num_control_points + point_id) * control_dim + control_id
    r = wp.rand_init(seed[0], unique_id)
    mean = mean_trajectory[0, point_id, control_id]
    lo, hi = control_limits[control_id, 0], control_limits[control_id, 1]
    sample = mean + resampling_noise_std * wp.randn(r)
    for i in range(10):
        if sample < lo or sample > hi:
            sample = mean + resampling_noise_std * wp.randn(r)
        else:
            break
    rollout_trajectories[env_id, point_id, control_id] = wp.clamp(sample, lo, hi)
    wp.atomic_add(seed, 0, 1)


@wp.kernel
def interpolate_control_hold(
    control_points: wp.array(dtype=float, ndim=3),
    control_dims: wp.array(dtype=int),
    control_gains: wp.array(dtype=float),
    t: float,
    torque_dim: int,
    # outputs
    torques: wp.array(dtype=float),
):
    env_id, control_id = wp.tid()
    t_id = int(t)
    control = control_points[env_id, t_id, control_id]
    torque_id = env_id * torque_dim + control_dims[control_id]
    torques[torque_id] = control * control_gains[control_id]


@wp.kernel
def interpolate_control_linear(
    control_points: wp.array(dtype=float, ndim=3),
    control_dims: wp.array(dtype=int),
    control_gains: wp.array(dtype=float),
    t: float,
    torque_dim: int,
    # outputs
    torques: wp.array(dtype=float),
):
    env_id, control_id = wp.tid()
    t_id = int(t)
    frac = t - wp.floor(t)
    control_left = control_points[env_id, t_id, control_id]
    control_right = control_points[env_id, t_id + 1, control_id]
    torque_id = env_id * torque_dim + control_dims[control_id]
    action = control_left * (1.0 - frac) + control_right * frac
    torques[torque_id] = action * control_gains[control_id]


@wp.kernel
def interpolate_control_cubic(
    control_points: wp.array(dtype=float, ndim=3),
    control_dims: wp.array(dtype=int),
    control_gains: wp.array(dtype=float),
    t: float,
    dt: float,
    torque_dim: int,
    # outputs
    torques: wp.array(dtype=float),
):
    env_id, control_id = wp.tid()
    t_id = int(t)
    q = t - wp.floor(t)
    c0 = control_points[env_id, t_id, control_id]
    c1 = control_points[env_id, t_id + 1, control_id]
    c2 = control_points[env_id, t_id + 2, control_id]
    c3 = control_points[env_id, t_id + 3, control_id]
    # Eq. 17
    phi_0 = 0.5 * ((c2 - c1) + (c1 - c0)) / dt
    # Eq. 18
    phi_1 = 0.5 * ((c3 - c2) + (c2 - c1)) / dt
    # Eq. 20-23
    a = 2.0 * q**3.0 - 3.0 * q**2.0 + 1.0
    b = (q**3.0 - 2.0 * q**2.0 + q) * dt
    c = -2.0 * q**3.0 + 3.0 * q**2.0
    d = (q**3.0 - q**2.0) * dt
    torque_id = env_id * torque_dim + control_dims[control_id]
    action = a * c1 + b * phi_0 + c * c2 + d * phi_1
    torques[torque_id] = action * control_gains[control_id]


@wp.kernel
def pick_best_trajectory(
    rollout_trajectories: wp.array(dtype=float, ndim=3),
    lowest_cost_id: int,
    # outputs
    best_traj: wp.array(dtype=float, ndim=3),
):
    t_id, control_id = wp.tid()
    best_traj[0, t_id, control_id] = rollout_trajectories[lowest_cost_id, t_id, control_id]


@wp.kernel
def enforce_control_limits(
    control_points: wp.array(dtype=float, ndim=3),
    control_limits: wp.array(dtype=float, ndim=2),
):
    env_id, t_id, control_id = wp.tid()
    lo, hi = control_limits[control_id, 0], control_limits[control_id, 1]
    control_points[env_id, t_id, control_id] = wp.clamp(control_points[env_id, t_id, control_id], lo, hi)


class ModelPredictiveController:

    def __init__(
        self,
        env_fn,
        opt_method="grad",
        resampling_noise_std=0.15,
        num_threads=16,
        control_step=10,
        num_control_points=3,
        traj_length=500,
        optimization_interval=1,
        interpolation_mode=InterpolationMode.INTERPOLATE_LINEAR,
        use_graph_capture=None,
        visualize_rollouts_vs_ref=False,
        **opt_settings,
    ):

        methods = {
            "grad": self.optimize_grad,
            "predsam": self.optimize_predsam,
        }
        if opt_method not in methods:
            raise ValueError(f"Unknown optimization method: {opt_method}")

        def optimize(state):
            methods[opt_method](state, **opt_settings)

        self.optimize = optimize

        self.resampling_noise_std = resampling_noise_std
        self.interpolation_mode = interpolation_mode

        # total number of time steps in the trajectory that is optimized
        self.traj_length = traj_length

        # time steps between control points
        self.control_step = control_step
        # number of control horizon points to interpolate between
        self.num_control_points = num_control_points
        # total number of horizon time steps
        self.horizon_length = self.num_control_points * self.control_step
        # number of trajectories to sample for optimization
        self.num_threads = num_threads
        # number of steps to follow before optimizing again
        self.optimization_interval = optimization_interval

        # whether env_rollout requires gradients
        self.use_diff_sim = opt_method == "grad"

        # create environment for sampling trajectories for optimization
        if hasattr(env_fn, "env_offset"):
            env_fn.env_offset = (0.0, 0.0, 0.0)
        if visualize_rollouts_vs_ref:
            rollout_render_mode = RenderMode.OPENGL
        else:
            rollout_render_mode = RenderMode.NONE
        self.env_rollout: Environment = env_fn(
            num_envs=self.num_threads,
            render_mode=rollout_render_mode,
            episode_frames=self.horizon_length,
            requires_grad=self.use_diff_sim,
        )

        # create environment for visualization and the reference state
        if visualize_rollouts_vs_ref:
            ref_render_mode = RenderMode.NONE
        else:
            ref_render_mode = RenderMode.OPENGL
        self.env_ref: Environment = env_fn(
            num_envs=1,
            episode_frames=self.traj_length,
            render_mode=ref_render_mode,
        )
        self.dof_count = len(self.env_ref.control_input)

        if use_graph_capture is not None:
            self.use_graph_capture = use_graph_capture
        else:
            self.use_graph_capture = wp.get_device(self.device).is_cuda and not CHECK_NANS and not DEBUG_GRADIENTS

        # optimized control points for the current horizon
        self.best_traj = None
        # control point samples for the current horizon
        self.rollout_trajectories = None
        self.global_step = 0

        assert len(self.env_rollout.controllable_dofs) == len(self.env_rollout.control_gains)
        assert len(self.env_rollout.controllable_dofs) == len(self.env_rollout.control_limits)

        # construct Warp array for the indices of controllable dofs
        self.controllable_dofs = wp.array(self.env_rollout.controllable_dofs, dtype=int)
        self.control_gains = wp.array(self.env_rollout.control_gains, dtype=float)
        self.control_limits = wp.array(self.env_rollout.control_limits, dtype=float)

        self.sampling_seed_counter = wp.zeros(1, dtype=int)

        self.controllable_dofs_np = np.array(self.env_rollout.controllable_dofs)

        # CUDA graphs
        self._opt_graph = None

        self._optimizer = None

        self.plotting_app = None
        self.plotting_window = None
        if DEBUG_PLOTS:
            self.plotting_app = QtWidgets.QApplication(sys.argv)
            self.plotting_window = MainWindow()
            self.plotting_window.show()

            # maximum number of threads to visualize in the control sample views
            self.max_plotting_threads = 50

            # get matplotlib colors as list for tab20 colormap
            self.plot_colors = (plt.get_cmap("tab10")(np.arange(10, dtype=int))[:, :3] * 255).astype(int)

            self.data_xs = np.arange(self.num_control_points)
            ys = np.zeros(self.num_control_points)
            self.rollout_plots = []
            self.rollout_plot_axs = []
            num_plots = self.control_dim + 1
            ncols = int(np.ceil(np.sqrt(num_plots)))
            nrows = int(np.ceil(num_plots / float(ncols)))
            # fig, axes = plt.subplots(
            #     ncols=ncols,
            #     nrows=nrows,
            #     constrained_layout=True,
            #     figsize=(ncols * 3.5, nrows * 3.5),
            #     squeeze=False,
            #     sharex=True,
            # )
            for i in range(num_plots):
                p = self.plotting_window.graphWidget1.addPlot(row=i // ncols, col=i % ncols)
                if i == 0:
                    p.setTitle("Cost")
                    # p.setYRange(0.0, 10.0)
                    self.cost_plot = p.plot(
                        np.arange(self.num_threads), np.zeros(self.num_threads), symbol="o", symbolSize=4
                    )
                else:
                    p.setTitle(f"Control {i-1}")
                    p.setYRange(*self.env_rollout.control_limits[i - 1])
                self.rollout_plot_axs.append(p)
            plotting_threads = min(self.num_threads, self.max_plotting_threads)
            for i in range(plotting_threads):
                thread_plots = []
                for j in range(1, self.control_dim + 1):
                    pen = pg.mkPen(color=self.plot_colors[i % len(self.plot_colors)])
                    p = self.rollout_plot_axs[j].plot(self.data_xs, ys, pen=pen, symbol="o", symbolSize=2)
                    thread_plots.append(p)
                self.rollout_plots.append(thread_plots)

            self.data_xs = np.arange(self.num_control_points)
            ys = np.zeros(self.num_control_points)
            self.ref_plots = []
            self.ref_plot_axs = []
            ncols = int(np.ceil(np.sqrt(num_plots)))
            nrows = int(np.ceil(num_plots / float(ncols)))
            # fig, axes = plt.subplots(
            #     ncols=ncols,
            #     nrows=nrows,
            #     constrained_layout=True,
            #     figsize=(ncols * 3.5, nrows * 3.5),
            #     squeeze=False,
            #     sharex=True,
            # )
            scaled_control_limits = (
                np.array(self.env_rollout.control_limits) * np.array(self.env_rollout.control_gains)[:, np.newaxis]
            )
            for i in range(num_plots):
                p = self.plotting_window.graphWidget2.addPlot(row=i // ncols, col=i % ncols)
                if i == 0:
                    p.setTitle("Ref Cost")
                    # p.setYRange(0.0, 10.0)
                    self.ref_cost_plot = p.plot(
                        np.arange(self.num_threads),
                        np.zeros(self.num_threads),
                        pen=pg.mkPen(color=(255, 0, 0)),
                    )
                else:
                    p.setTitle(f"Ref Control {i-1}")
                    p.setYRange(*scaled_control_limits[i - 1])
                self.ref_plot_axs.append(p)
            for j in range(self.control_dim):
                pen = pg.mkPen(color=self.plot_colors[0])
                p = self.ref_plot_axs[j + 1].plot(self.data_xs, ys, pen=pen)
                self.ref_plots.append(p)

    @property
    def control_dim(self):
        return len(self.env_rollout.controllable_dofs)

    @property
    def body_count(self):
        return self.env_ref.bodies_per_env

    @property
    def device(self):
        return self.env_rollout.device

    def allocate_trajectories(self):
        # number of control points that need to be optimized, respective of the type of interpolation
        self.num_control_points_data = self.num_control_points
        if self.interpolation_mode == InterpolationMode.INTERPOLATE_LINEAR:
            self.num_control_points_data += 1
        elif self.interpolation_mode == InterpolationMode.INTERPOLATE_CUBIC:
            self.num_control_points_data += 3
        # optimized control points for the current horizon (3-dimensional to be compatible with rollout trajectories)
        self.best_traj = wp.zeros(
            (1, self.num_control_points_data, self.control_dim),
            dtype=float,
            requires_grad=self.use_diff_sim,
            device=self.device,
        )
        # control point samples for the current horizon
        self.rollout_trajectories = wp.zeros(
            (self.num_threads, self.num_control_points_data, self.control_dim),
            dtype=float,
            requires_grad=self.use_diff_sim,
            device=self.device,
        )
        # cost of each trajectory
        self.rollout_costs = wp.zeros(
            (self.num_threads,), dtype=float, requires_grad=self.use_diff_sim, device=self.device
        )

    def run(self, render_to_usd_file: str = None):
        self.env_rollout.before_simulate()
        self.env_ref.reset()
        self.env_ref.before_simulate()
        self.env_ref_acts = []
        self.env_ref_costs = []
        ref_cost = wp.zeros(1, dtype=float, device=self.device)
        assert len(self.env_ref.state.body_q) == self.body_count
        self.allocate_trajectories()
        self.assign_control_fn(self.env_ref, self.best_traj)
        self.assign_control_fn(self.env_rollout, self.rollout_trajectories)

        self.global_step = 0

        if render_to_usd_file is None:
            self.renderer_usd = None
        else:
            render_to_usd_file = os.path.abspath(render_to_usd_file)
            self.renderer_usd = wp.sim.render.SimRendererUsd(
                self.env_ref.model,
                render_to_usd_file,
                up_axis=self.env_ref.up_axis,
            )
            print(f"Rendering to {render_to_usd_file}...")

        progress = trange(self.traj_length)
        for t in progress:
            # optimize trajectory horizon
            self.optimize(self.env_ref.state)
            # set sim time to zero to execute first optimized action
            self.env_ref.sim_time = 0
            self.env_ref.sim_step = 0
            for _ in range(self.optimization_interval):
                # advance the reference state with the next best action
                self.env_ref.update()
                self.env_ref.render()

                self.global_step += 1

                if self.renderer_usd is not None:
                    self.renderer_usd.begin_frame(t / 60.0)
                    # render state 1 (swapped with state 0 just before)
                    render_state = self.env_ref.next_state
                    self.env_ref.custom_render(render_state, renderer=self.renderer_usd)
                    self.renderer_usd.render(render_state)
                    self.renderer_usd.end_frame()

            ref_cost.zero_()
            # use some fake step and traj_length numbers here to not have discount factor applied
            self.env_ref.evaluate_cost(self.env_ref.state, self.env_ref.control, ref_cost, 1, 1)
            self.env_ref_costs.append(ref_cost.numpy()[0])

            if DEBUG_PLOTS:
                self.ref_cost_plot.setData(np.arange(t + 1), self.env_ref_costs)
            #     fig, axes, ncols, nrows = self._create_plot_grid(self.control_dim)
            #     fig.suptitle("best traj")
            #     best_traj = self.best_traj.numpy()
            #     for dim in range(self.control_dim):
            #         ax = axes[dim // ncols, dim % ncols]
            #         ax.plot(best_traj[0, :, dim])
            #     plt.show()

            progress.set_description(f"cost: {self.last_lowest_cost:.2f} ({self.last_lowest_cost_id})")

        self.env_ref.after_simulate()
        self.env_rollout.after_simulate()

        if self.renderer_usd is not None:
            self.renderer_usd.save()

    def before_optimize(self):
        pass

    def after_optimize(self):
        pass

    def optimize_predsam(self, state):
        # predictive sampling algorithm
        self.before_optimize()
        if self.use_graph_capture:
            if self._opt_graph is None:
                wp.capture_begin()
                self.sample_controls(self.best_traj)
                self.rollout(state, self.rollout_trajectories)
                self._opt_graph = wp.capture_end()
            else:
                wp.capture_launch(self._opt_graph)
        else:
            self.sample_controls(self.best_traj)
            self.rollout(state, self.rollout_trajectories)
        self.pick_best_control()
        self.after_optimize()

    def optimize_grad(
        self,
        state,
        num_opt_steps=20,
        learning_rate=1e-2,
        optimizer="sgd",
        use_fd_grads=False,
        resample_every_step=1,
        **optimizer_settings,
    ):
        assert self.use_diff_sim, "Gradient-based optimization requires differentiable simulation"
        self.before_optimize()

        # gradient-based optimization
        if self._optimizer is None:
            if optimizer == "sgd":
                self._optimizer = SGD(
                    [self.rollout_trajectories.flatten()],
                    lr=learning_rate,
                    nesterov=optimizer_settings.pop("nesterov", False),
                    momentum=optimizer_settings.pop("momentum", 0.0),
                    **optimizer_settings,
                )
                # TODO add these overloads to Warp SGD!
                p = wp.array(dtype=wp.float32)
                wp.overload(wp.optim.sgd.sgd_step_kernel, {"g": p, "b": p, "params": p})
            elif optimizer == "adam":
                self._optimizer = Adam(
                    [self.rollout_trajectories.flatten()],
                    lr=learning_rate,
                    **optimizer_settings,
                )

        if self.global_step == 0 or (resample_every_step > 0 and self.global_step % resample_every_step == 0):
            self.sample_controls(self.best_traj)

        if self.use_graph_capture and not use_fd_grads:
            if self._opt_graph is None or self.env_ref.invalidate_cuda_graph:
                with wp.ScopedCapture() as capture:
                    self.tape = wp.Tape()
                    with self.tape:
                        self.rollout(state, self.rollout_trajectories)
                    self.rollout_costs.grad.fill_(1.0)
                    self.tape.backward()
                    self._optimizer.step([self.rollout_trajectories.grad.flatten()])
                    self.clamp_controls()
                    self.tape.zero()
                self._opt_graph = capture.graph
                self.env_ref.invalidate_cuda_graph = False

            for _ in range(num_opt_steps):
                wp.capture_launch(self._opt_graph)
        else:
            self.sample_controls(self.best_traj)
            for it in range(num_opt_steps):
                # check_tape_safety(
                #     lambda s, r: self.rollout(s, r),
                #     inputs=[state, self.rollout_trajectories],
                #     outputs=[self.rollout_costs])
                # check_jacobian(
                #     lambda controls: self.rollout(state, controls),
                #     inputs=[self.rollout_trajectories],
                #     input_names=["controls"],
                #     output_names=["costs"],
                #     plot_jac_on_fail=True
                # )

                if not use_fd_grads:
                    self.tape = wp.Tape()
                    with self.tape:
                        self.rollout(state, self.rollout_trajectories)

                    self.rollout_costs.grad.fill_(1.0)
                    self.tape.backward()
                else:
                    stride = self.num_control_points_data * self.control_dim

                    def sparsity_pattern(out_id, in_id):
                        print(f"sparsity pattern: {out_id}, {in_id}")
                        return in_id // stride == out_id

                    # jac_fd_raw = np.zeros((self.num_threads, self.num_threads * stride))
                    # for i in range(self.num_threads):
                    #     for j in range(self.num_threads * stride):
                    #         jac_fd_raw[i, j] = int(sparsity_pattern(i, j))
                    # plt.matshow(jac_fd_raw)
                    # plt.show()

                    jac_fd_raw = function_jacobian_fd(
                        lambda controls: self.rollout(state, controls),
                        inputs=[self.rollout_trajectories],
                        eps=1e-3,
                        # sparsity_pattern=sparsity_pattern,
                    )
                    jac_fd = np.empty((self.num_threads, self.num_control_points_data, self.control_dim))
                    for i in range(self.num_threads):
                        jac_fd[i] = jac_fd_raw[i, i * stride : (i + 1) * stride].reshape(
                            (self.num_control_points_data, self.control_dim)
                        )
                    self.rollout_trajectories.grad.assign(jac_fd)

                self._optimizer.step([self.rollout_trajectories.grad.flatten()])
                # print(f"\niter {it} cost:", self.rollout_costs.numpy().flatten())
                # print("\tchecked cost 1:", self.rollout(state, self.rollout_trajectories).numpy())
                # print("\tchecked cost 2:", self.rollout(state, self.rollout_trajectories).numpy())

                # jac_fd = function_jacobian_fd(
                #     lambda controls: self.rollout(state, controls),
                #     inputs=[self.rollout_trajectories],
                #     eps=1e-3)
                # print("\tjac_fd 1:", jac_fd.flatten())
                # jac_fd = function_jacobian_fd(
                #     lambda controls: self.rollout(state, controls),
                #     inputs=[self.rollout_trajectories],
                #     eps=1e-3)
                # print("\tjac_fd 2:", jac_fd.flatten())
                self.rollout_trajectories.assign(
                    self.rollout_trajectories.numpy() - learning_rate * self.rollout_trajectories.grad.numpy()
                )
                # print("\tgrad:", self.rollout_trajectories.grad.numpy().flatten())
                # print("\tcontrols:", self.rollout_trajectories.numpy().flatten())
                self.clamp_controls()
                # print("\tclamped controls:", self.rollout_trajectories.numpy().flatten())
                if not use_fd_grads:
                    self.tape.zero()
        wp.synchronize()
        self.pick_best_control()
        self.after_optimize()

    def pick_best_control(self):
        costs = self.rollout_costs.numpy()
        lowest_cost_id = np.argmin(costs)
        # print(f"lowest cost: {lowest_cost_id}\t{costs[lowest_cost_id]}")
        self.last_lowest_cost_id = lowest_cost_id
        self.last_lowest_cost = costs[lowest_cost_id]
        wp.launch(
            pick_best_trajectory,
            dim=(self.num_control_points_data, self.control_dim),
            inputs=[self.rollout_trajectories, lowest_cost_id],
            outputs=[self.best_traj],
            device=self.device,
        )
        # self.rollout_trajectories[-1].assign(self.best_traj[0])

        # if True or CHECK_NANS:
        #     if np.isnan(self.rollout_trajectories.numpy()).any():
        #         raise RuntimeError("NaN detected in rollout trajectories")
        #     if np.isnan(self.env_rollout.state.body_qd.numpy()).any():
        #         raise RuntimeError("NaN detected in rollout simulation")

        if DEBUG_PLOTS:
            # plot costs, rollout trajectories
            self.cost_plot.setData(np.arange(self.num_threads), costs)
            trajs = self.rollout_trajectories.numpy()
            plotting_threads = min(self.num_threads, self.max_plotting_threads)
            if len(self.env_ref_acts) > 0:
                env_ref_acts = np.array(self.env_ref_acts)
                for j in range(self.control_dim):
                    for i in range(plotting_threads):
                        self.rollout_plots[i][j].setData(self.data_xs, trajs[i, : self.num_control_points, j])
                    self.ref_plots[j].setData(np.arange(len(env_ref_acts)), env_ref_acts[:, j])

    def clamp_controls(self):
        # enforce control limits on the control points
        wp.launch(
            enforce_control_limits,
            dim=[self.num_threads, self.num_control_points_data, self.control_dim],
            inputs=[self.rollout_trajectories, self.control_limits],
            device=self.device,
        )

    def assign_control_fn(self, env, controls):
        # assign environment control application function that interpolates the control points
        def update_control_hold():
            wp.launch(
                interpolate_control_hold,
                dim=(env.num_envs, self.control_dim),
                inputs=[
                    controls,
                    self.controllable_dofs,
                    self.control_gains,
                    env.sim_time / self.control_step / env.frame_dt,
                    self.dof_count,
                ],
                outputs=[env.control_input],
                device=self.device,
            )
            if env == self.env_ref:
                self.env_ref_acts.append(env.control_input.numpy()[self.controllable_dofs_np])
            # if DEBUG_PLOTS and not self.use_graph_capture:
            #     self.control_hist.append(
            #         env.control_input.numpy()[self.controllable_dofs_np].reshape((-1, self.control_dim))
            #     )

        def update_control_linear():
            wp.launch(
                interpolate_control_linear,
                dim=(env.num_envs, self.control_dim),
                inputs=[
                    controls,
                    self.controllable_dofs,
                    self.control_gains,
                    env.sim_time / self.control_step / env.frame_dt,
                    self.dof_count,
                ],
                outputs=[env.control_input],
                device=self.device,
            )
            if env == self.env_ref:
                self.env_ref_acts.append(env.control_input.numpy()[self.controllable_dofs_np])
            # if DEBUG_PLOTS and not self.use_graph_capture:
            #     self.control_hist.append(
            #         env.control_input.numpy()[self.controllable_dofs_np].reshape((-1, self.control_dim))
            #     )

        def update_control_cubic():
            wp.launch(
                interpolate_control_cubic,
                dim=(env.num_envs, self.control_dim),
                inputs=[
                    controls,
                    self.controllable_dofs,
                    self.control_gains,
                    env.sim_time / self.control_step / env.frame_dt,
                    self.control_step * env.frame_dt,
                    self.dof_count,
                ],
                outputs=[env.control_input],
                device=self.device,
            )
            if env == self.env_ref:
                self.env_ref_acts.append(env.control_input.numpy()[self.controllable_dofs_np])
            # if DEBUG_PLOTS and not self.use_graph_capture:
            #     self.control_hist.append(
            #         env.control_input.numpy()[self.controllable_dofs_np].reshape((-1, self.control_dim))
            #     )

        update_fn = {
            InterpolationMode.INTERPOLATE_HOLD: update_control_hold,
            InterpolationMode.INTERPOLATE_LINEAR: update_control_linear,
            InterpolationMode.INTERPOLATE_CUBIC: update_control_cubic,
        }[self.interpolation_mode]

        # cache Environment.before_update to not overwrite it
        before_update = env.before_update
        env.before_update = lambda: (before_update(), update_fn())

    def rollout(self, state, controls):
        self.env_rollout.reset()
        self.rollout_costs.zero_()
        # if DEBUG_PLOTS and not self.use_graph_capture:
        #     self.control_hist = []

        if self.env_rollout.uses_generalized_coordinates:
            wp.launch(
                replicate_generalized_states,
                dim=self.num_threads,
                inputs=[state.joint_q, state.joint_qd],
                outputs=[self.env_rollout.state.joint_q, self.env_rollout.state.joint_qd],
                device=self.device,
            )
        else:
            wp.launch(
                replicate_maximal_states,
                dim=self.num_threads,
                inputs=[state.body_q, state.body_qd, self.body_count],
                outputs=[self.env_rollout.state.body_q, self.env_rollout.state.body_qd],
                device=self.device,
            )

        for t in range(self.horizon_length):
            self.env_rollout.update()
            if CHECK_NANS:
                if np.isnan(self.env_rollout.state.body_qd.numpy()).any():
                    raise RuntimeError("NaN detected in rollout simulation")

            self.env_rollout.evaluate_cost(
                self.env_rollout.state, self.env_rollout.control, self.rollout_costs, t, self.horizon_length
            )
            if CHECK_NANS:
                if np.isnan(self.rollout_costs.numpy()).any():
                    raise RuntimeError("NaN detected in rollout cost")
            if not self.use_graph_capture:
                self.env_rollout.render()

        # if DEBUG_PLOTS and not self.use_graph_capture:
        #     self.control_hist = np.array(self.control_hist)
        #     fig, axes, ncols, nrows = self._create_plot_grid(self.control_dim)
        #     fig.suptitle("joint acts")
        #     for dim in range(self.control_dim):
        #         ax = axes[dim // ncols, dim % ncols]
        #         ax.plot(self.control_hist[:, :, dim], alpha=0.4)
        #     plt.show()
        #     self.control_hist = []

        return self.rollout_costs

    def sample_controls(self, nominal_traj):
        # sample control waypoints around the nominal trajectory
        wp.launch(
            sample_gaussian,
            dim=(self.num_threads, self.num_control_points_data, self.control_dim),
            inputs=[
                nominal_traj,
                self.resampling_noise_std,
                self.num_control_points_data,
                self.control_dim,
                self.control_limits,
                self.sampling_seed_counter,
            ],
            outputs=[self.rollout_trajectories],
            device=self.device,
        )
        # if DEBUG_PLOTS:
        #     fig, axes, ncols, nrows = self._create_plot_grid(self.control_dim)
        #     fig.suptitle("rollout trajectories")
        #     for dim in range(self.control_dim):
        #         ax = axes[dim // ncols, dim % ncols]
        #         ax.plot(self.rollout_trajectories[:, :, dim].numpy().T, alpha=0.2, label="sampled")
        #         ax.plot(nominal_traj[:, dim].numpy(), label="nominal")
        #     plt.show()

        # self.sampling_seed_counter += self.num_threads * self.num_control_points * self.control_dim

    @staticmethod
    def _create_plot_grid(dof):
        ncols = int(np.ceil(np.sqrt(dof)))
        nrows = int(np.ceil(dof / float(ncols)))
        fig, axes = plt.subplots(
            ncols=ncols,
            nrows=nrows,
            constrained_layout=True,
            figsize=(ncols * 3.5, nrows * 3.5),
            squeeze=False,
            sharex=True,
        )
        for dim in range(ncols * nrows):
            ax = axes[dim // ncols, dim % ncols]
            if dim >= dof:
                ax.axis("off")
                continue
            ax.grid()
        return fig, axes, ncols, nrows
