import torch
import warp as wp
import warp.sim
from .utils import cost2reward, eval_timeout
from .rlgames import RlgamesEnvWrapper

BODY_Q_DIM = 7

class ManipulabilityTorchEnvWrapper(RlgamesEnvWrapper):
    def __init__(
        self,
        env,
        render_mode: str = None,
        max_episode_length: int = 50,
        reward_scale: float = 1.0,
        reward_bias: float = 0.0,
        image_width: int = 128,
        image_height: int = 128,
        requires_grad: bool = True,
    ):
        super().__init__(
            env=env,
            render_mode=render_mode,
            max_episode_length=max_episode_length,
            reward_scale=reward_scale,
            reward_bias=reward_bias,
            image_width=image_width,
            image_height=image_height,
            requires_grad=requires_grad,
        )

        if requires_grad:
            assert env.requires_grad, "Environment must be set to require gradients"

        # the state from which we begin the rollout
        self.state = self.env.model.state(requires_grad=False)

        # the state from which to compute the observations for envs not being reset during env.reset(env_ids)
        # self.state = self.env.model.state()

        # self.states = [self.env.model.state() for _ in range(self.env.sim_substeps + 1)]
        self.control = self.env.model.control(requires_grad=False)
        # self.control_input = self.control.joint_act

        self.step_fn = self.create_torch_step_fn()

        if self.env.uses_generalized_coordinates:
            self.q = wp.to_torch(self.state.joint_q)
            self.qd = wp.to_torch(self.state.joint_qd)

        else:
            self.q = wp.to_torch(self.state.body_q)
            self.qd = wp.to_torch(self.state.body_qd)

    def clear_grad(self):
        """cut off the gradient from the current state to previous states"""

        # new_state = self.env.model.state(requires_grad=False)
        # new_state.joint_q.assign(self.state.joint_q)
        # new_state.joint_qd.assign(self.state.joint_qd)
        # new_state.body_q.assign(self.state.body_q)
        # new_state.body_qd.assign(self.state.body_qd)
        # new_state.joint_q.requires_grad = True
        # new_state.joint_qd.requires_grad = True
        # new_state.body_q.requires_grad = True
        # new_state.body_qd.requires_grad = True
        # self.state = new_state

        with torch.no_grad():
            self.q = self.q.clone()
            self.qd = self.qd.clone()

        self.compute_observations(self.state, self.control, self.obs_buf)
        return wp.to_torch(self.obs_buf)

    def reset(self, env_ids=None, grads=False, force_reset=False):
        if grads:
            """
            This function starts collecting a new trajectory from the current states but cuts off the computation graph to the previous states.
            It has to be called every time the algorithm starts an episode and it returns the observation vectors

            Note: force_reset resets all envs and is here for compatibility with rl_games
            """
            self.clear_grad()
            self.compute_observations(self.state, self.control, self.obs_buf)
            return wp.to_torch(self.obs_buf)

        if env_ids is None or force_reset:
            # reset all environemnts
            self.env.reset()
            self.state = self.env.model.state(requires_grad=False)

            self.env.sim_time = 0.0
            self.env.sim_step = 0
            self.step_count = 0
            self.done_buf.fill_(0)
            self.episode_length_buf.fill_(0)

            if self.env.uses_generalized_coordinates:
                self.q = wp.to_torch(self.state.joint_q)
                self.qd = wp.to_torch(self.state.joint_qd)

            else:
                self.q = wp.to_torch(self.state.body_q)
                self.qd = wp.to_torch(self.state.body_qd)

        if env_ids is not None:
            self.env.reset_envs(wp.from_torch(env_ids))

        self.compute_observations(self.state, self.control, self.obs_buf)
        return wp.to_torch(self.obs_buf)

    def render(self, state=None):
        self.env.render(state=state)

    def step_internal(self, ctx, act: wp.array):
        # inner step function, to be captured in a CUDA graph
        self.env.assign_control(act, ctx.control, ctx.states[0])

        if False:
            state = self.env.update()
        else:
            for i in range(self.env.sim_substeps):
                self.env.step(
                    ctx.states[i],
                    ctx.states[i + 1],
                    ctx.control,
                    # eval_collisions=False,
                )
            state = ctx.states[self.env.sim_substeps]
        # print([state.joint_q.ptr for state in ctx.states])
        # print(state.joint_q.ptr)
        # reward is single-step, not cumulative
        ctx.cost_buf.zero_()
        ctx.done_buf.zero_()
        self.env.compute_cost_termination(
            state,
            ctx.control,
            ctx.step_count,
            self.max_episode_length,
            ctx.cost_buf,
            ctx.done_buf,
        )
        # this operation write arrays on the env instance to be used in the next step
        wp.launch(
            eval_timeout,
            dim=self.num_envs,
            inputs=[
                self.max_episode_length,
            ],
            outputs=[
                ctx.timeout_buf,
                ctx.episode_length_buf,
                ctx.done_buf,
            ],
            device=self.device,
            record_tape=False,
        )

        wp.launch(
            cost2reward,
            dim=self.num_envs,
            inputs=[
                ctx.cost_buf,
                self.reward_scale,
                self.reward_bias,
            ],
            outputs=[ctx.rew_buf],
            device=self.device,
        )
        self.compute_observations(state, ctx.control, ctx.obs_buf)
        for object_key in self.env.get_target_keys():
            self.env.query_target_body_q(state, object_key, ctx.target_body_q_buffer[object_key])
        # self.render(state) # This line causes tons of warnings!!! Moved to forward call

        # self.env.reset_envs(ctx.done_buf, state=state) # This line also causes tons of warnings!!! Moved to forward call

        # self.state = state

        ctx.step_count += 1

    def populate_context(self, ctx):
        ctx.states = [self.env.model.state() for _ in range(self.env.sim_substeps + 1)]
        ctx.control = self.env.model.control()
        ctx.control_input = ctx.control.joint_act

        ctx.step_count = self.step_count

        with wp.ScopedDevice(self.device):
            ctx.done_buf = wp.clone(self.done_buf)
            ctx.episode_length_buf = wp.clone(self.episode_length_buf)
            ctx.time_step_buf = wp.clone(self.time_step_buf)
            ctx.timeout_buf = wp.clone(self.timeout_buf)
            ctx.obs_buf = wp.empty(
                (self.num_envs, self.num_obs),
                dtype=wp.float32,
                device=self.device,
                requires_grad=self.requires_grad,
            )
            ctx.act_buf = wp.empty(
                (self.num_envs, self.num_actions),
                dtype=wp.float32,
                requires_grad=self.requires_grad,
            )
            ctx.rew_buf = wp.empty(
                self.num_envs,
                dtype=wp.float32,
                requires_grad=self.requires_grad,
            )
            ctx.cost_buf = wp.empty(
                self.num_envs,
                dtype=wp.float32,
                requires_grad=self.requires_grad,
            )

            ctx.target_body_q_buffer = {}
            for object_key in self.env.get_target_keys():
                ctx.target_body_q_buffer[object_key] = wp.empty(
                    (self.num_envs, BODY_Q_DIM),
                    dtype=wp.float32,
                    requires_grad=self.requires_grad
                )
            # ctx.target_grad_buffer = wp.empty(
            #     (self.num_envs, BODY_Q_DIM, self.num_actions),
            #     dtype=wp.float32,
            #     requires_grad=self.requires_grad
            # )

        ctx.target_grad_buffer = {}
        for object_key in self.env.get_target_keys():
            ctx.target_grad_buffer[object_key] = torch.empty(
                (self.num_envs, BODY_Q_DIM, self.num_actions),
                dtype=torch.float32,
                requires_grad=self.requires_grad,
                device=wp.device_to_torch(self.device)
            )

    def update_from_context(self, ctx):
        # self.state = ctx.states[-1]
        self.control = ctx.control

        self.step_count = ctx.step_count
        self.done_buf = ctx.done_buf
        self.episode_length_buf = ctx.episode_length_buf
        self.time_step_buf = ctx.time_step_buf
        self.timeout_buf = ctx.timeout_buf

        # continue next step from previous last state
        self.state.joint_q.assign(ctx.states[-1].joint_q)
        self.state.joint_qd.assign(ctx.states[-1].joint_qd)
        self.state.body_q.assign(ctx.states[-1].body_q)
        self.state.body_qd.assign(ctx.states[-1].body_qd)

    def create_torch_step_fn(self):
        class IntegratorSimulate(torch.autograd.Function):
            @staticmethod
            def forward(
                ctx,
                q: torch.Tensor,
                qd: torch.Tensor,
                act: torch.Tensor,
            ):
                # initialize data for stepping
                self.populate_context(ctx)

                if self.env.uses_generalized_coordinates:
                    ctx.states[0].joint_q.assign(wp.from_torch(q))
                    ctx.states[0].joint_qd.assign(wp.from_torch(qd))
                else:
                    ctx.states[0].body_q.assign(wp.from_torch(q, dtype=wp.transform))
                    ctx.states[0].body_qd.assign(
                        wp.from_torch(qd, dtype=wp.spatial_vector)
                    )

                # self.clear_grad()

                ctx.tape = wp.Tape()

                extras = {}

                ctx.act_buf.assign(wp.from_torch(act))
                with ctx.tape:
                    self.step_internal(ctx, ctx.act_buf)
                
                self.env.reset_envs(ctx.done_buf, state=self.state) # Check this line!!! Does this do the intended env reset @Rachanon?

                if self.render_mode is not None and self.render_mode != "none":
                    final_state = ctx.states[self.env.sim_substeps]
                    self.render(state=final_state) # Call render on the wrapper instance

                # ctx.tape.visualize(
                #     filename="tape.dot",
                #     track_inputs=[ctx.act_buf],
                #     track_input_names=["act"],
                #     track_outputs=[ctx.rew_buf, ctx.obs_buf, ctx.target_body_q_buffer],
                #     track_output_names=["rew_buf", "obs_buf", "ctx.target_body_q_buffer"],
                #     simplify_graph=True,
                # )

                # Compute each row of the Jacobian
                manipulability = {}
                object_poses = {}

                for object_key in self.env.get_target_keys():
                    for output_index in range(BODY_Q_DIM):
                        # Select which row of the Jacobian we want to compute
                        select_index = torch.zeros((BODY_Q_DIM,), dtype=torch.float32, device=wp.device_to_torch(self.device))
                        select_index[output_index] = 1.0

                        # Assemble selection vector for all environments (can be precomputed)
                        e = wp.from_torch(torch.tile(select_index, (self.num_envs,)).unsqueeze(0), dtype=wp.float32)  # Equivalent to np.tile(select_index, self.num_envs)

                        ctx.tape.backward(grads={ctx.target_body_q_buffer[object_key]: e}) # This line causes all of the warnings!
                        if ctx.act_buf in ctx.tape.gradients:
                            # print("q_grad_i SUCCESFULLY INITIALIZED")
                            q_grad_i = wp.to_torch(ctx.tape.gradients[ctx.act_buf])
                        else:
                            # Key is missing, assume zero gradient
                            print(f"Warning: Gradient for action buffer not found for output_index {output_index}. Assuming zero gradient.")
                            q_grad_i = torch.zeros((self.num_envs, self.num_actions),
                                                        dtype=torch.float32,
                                                        device=wp.device_to_torch(self.device))

                        # Ensure it's a tensor and reshape correctly
                        ctx.target_grad_buffer[object_key][:, output_index, :] = q_grad_i.reshape((self.num_envs, self.num_actions))
                        ctx.tape.zero()

                    manipulability[object_key] = torch.clone(ctx.target_grad_buffer[object_key])
                    object_poses[object_key] = wp.to_torch(ctx.target_body_q_buffer[object_key])
                self.update_from_context(ctx)

                observations = wp.to_torch(ctx.obs_buf)
                rewards = wp.to_torch(ctx.rew_buf)
                dones = wp.to_torch(ctx.done_buf)
                # manipulability = torch.clone(ctx.target_grad_buffer)
                # object_pose = wp.to_torch(ctx.target_body_q_buffer)

                extras = {
                    "obs_before_reset": wp.to_torch(ctx.obs_buf),  # .clone(),
                    "termination": dones,
                    "truncation": dones,
                }
                if hasattr(self.env, "primal"):
                    extras["primal"] = wp.to_torch(self.env.primal)
                else:
                    extras["primal"] = rewards.detach()

                if False and self.requires_grad:
                    extras.update(
                        {
                            "contact_count": self.state.contact_count.clone().detach(),
                            "contact_forces": self.state.contact_f.clone()
                            .detach()
                            .view(self.num_envs, -1, 6),
                            "body_forces": self.state.body_f.clone()
                            .detach()
                            .view(self.num_envs, -1, 6),
                            "accelerations": self.state.body_a_s.clone()
                            .detach()
                            .view(self.num_envs, -1, 6),
                        }
                    )

                # copy warp data to pytorch
                # self.extras["time_outs"] = wp.to_torch(self.timeout_buf).squeeze(-1)
                self.obs_dict["obs"] = observations

                if False:
                    ctx.tape.visualize(
                        filename="tape.dot",
                        track_inputs=[ctx.act_buf],
                        track_input_names=["act"],
                        track_outputs=[ctx.rew_buf, ctx.obs_buf],
                        track_output_names=["rew_buf", "obs_buf"],
                        simplify_graph=False,
                    )
                    import os

                    os.system("dot -Tsvg tape.dot -o tape.svg")
                    os.system("start tape.svg")

                if self.env.uses_generalized_coordinates:
                    next_q = wp.to_torch(ctx.states[-1].joint_q)
                    next_qd = wp.to_torch(ctx.states[-1].joint_qd)
                else:
                    next_q = wp.to_torch(ctx.states[-1].body_q)
                    next_qd = wp.to_torch(ctx.states[-1].body_qd)

                # print("rewards:", self.rew_buf.numpy())
                return (
                    observations,
                    rewards,
                    dones,
                    extras,
                    next_q,
                    next_qd,
                    manipulability,
                    object_poses
                )

            @staticmethod
            def backward(
                ctx,
                adj_obs,
                adj_rew,
                adj_done,
                adj_extras,
                adj_next_q,
                adj_next_qd,
            ):
                # map incoming Torch grads to our output variables
                ctx.obs_buf.grad = wp.from_torch(adj_obs)
                ctx.rew_buf.grad = wp.from_torch(adj_rew)
                if self.env.uses_generalized_coordinates:
                    ctx.states[-1].joint_q.grad = wp.from_torch(adj_next_q)
                    ctx.states[-1].joint_qd.grad = wp.from_torch(adj_next_qd)
                else:
                    ctx.states[-1].body_q.grad = wp.from_torch(
                        adj_next_q, dtype=wp.transform
                    )
                    ctx.states[-1].body_qd.grad = wp.from_torch(
                        adj_next_qd, dtype=wp.spatial_vector
                    )

                ctx.tape.backward()
                act_grad = wp.to_torch(ctx.act_buf.grad).clone()
                if self.env.uses_generalized_coordinates:
                    q_grad = wp.to_torch(ctx.states[0].joint_q.grad).clone()
                    qd_grad = wp.to_torch(ctx.states[0].joint_qd.grad).clone()
                else:
                    q_grad = wp.to_torch(ctx.states[0].body_q.grad).clone()
                    qd_grad = wp.to_torch(ctx.states[0].body_qd.grad).clone()

                # import pdb; pdb.set_trace()

                ctx.tape.zero()

                # return adjoint w.r.t. inputs
                return (q_grad, qd_grad, act_grad)

        return IntegratorSimulate.apply

    def step(self, action: torch.Tensor):
        """
        Step the environments with the given action

        :param action: ([int] or [float]) the action
        :return: ([int] or [float], [float], [bool], dict) observation, reward, done, information
        """
        observations, rewards, dones, extras, self.q, self.qd, manipulability, object_poses = self.step_fn(
            self.q, self.qd, action
        )
        return observations, rewards, dones, extras, manipulability, object_poses

    def close(self):
        self.env.close()
