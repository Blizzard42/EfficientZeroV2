# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Drone environment
#
# Simulation of a quadrotor drone with custom propeller dynamics.
# This example implements a simulation plugin for the SemiImplicitIntegrator
# that computes the propeller forces and torques.
#
###########################################################################

import numpy as np
import warp as wp
import warp.sim
from warp.sim.collide import sphere_sdf, box_sdf, capsule_sdf, cylinder_sdf, cone_sdf, mesh_sdf, plane_sdf

from environment import Environment, run_env, IntegratorType, RenderMode


# air density at sea level
air_density = wp.constant(1.225)  # kg / m^3

carbon_fiber_density = 1750.0  # kg / m^3
abs_density = 1050.0  # kg / m^3


class PropellerData:
    def __init__(
        self,
        thrust: float = 0.109919,
        power: float = 0.040164,
        diameter: float = 0.2286,
        height: float = 0.01,
        max_rpm: float = 6396.667,
        turning_cw: bool = True,
    ):
        """
        Creates an object to store propeller information. Uses default settings of the GWS 9X5 propeller.

        Args:
        thrust: Thrust coefficient.
        power: Power coefficient.
        diameter: Propeller diameter in meters, default is for DJI Phantom 2.
        height: Height of cylindrical area in meters when propeller rotates.
        max_rpm: Maximum RPM of propeller.
        turning_cw: True if propeller turns clockwise, False if counter-clockwise.
        """
        self.thrust = thrust
        self.power = power
        self.diameter = diameter
        self.height = height
        self.max_rpm = max_rpm
        self.turning_direction = 1.0 if turning_cw else -1.0

        # compute max thrust and torque
        revolutions_per_second = max_rpm / 60
        max_speed = revolutions_per_second * wp.TAU  # radians / sec
        self.max_speed_square = max_speed**2

        nsquared = revolutions_per_second**2
        self.max_thrust = self.thrust * air_density * nsquared * self.diameter**4
        self.max_torque = self.power * air_density * nsquared * self.diameter**5 / wp.TAU


@wp.struct
class Propeller:
    body: int
    pos: wp.vec3
    dir: wp.vec3
    thrust: float
    power: float
    diameter: float
    height: float
    max_rpm: float
    max_thrust: float
    max_torque: float
    turning_direction: float
    max_speed_square: float


@wp.kernel
def compute_prop_wrenches(
    props: wp.array(dtype=Propeller),
    controls: wp.array(dtype=float),
    body_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    # outputs
    body_f: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()
    prop = props[tid]
    control = controls[tid]
    tf = body_q[prop.body]
    dir = wp.transform_vector(tf, prop.dir)
    force = dir * prop.max_thrust * control
    torque = dir * prop.max_torque * control * prop.turning_direction
    moment_arm = wp.transform_point(tf, prop.pos) - wp.transform_point(tf, body_com[prop.body])
    torque += wp.cross(moment_arm, force)
    # apply angular damping
    torque *= 0.8
    wp.atomic_add(body_f, prop.body, wp.spatial_vector(torque, force))


@wp.struct
class DragVertex:
    body: int
    pos: wp.vec3
    normal: wp.vec3
    area: float
    drag_coefficient: float
    omnidirectional: wp.bool


@wp.func
def counter_increment(counter: wp.array(dtype=int), counter_index: int, tids: wp.array(dtype=int), tid: int):
    # increment counter, remember which thread received which counter value
    next_count = wp.atomic_add(counter, counter_index, 1)
    tids[tid] = next_count
    return next_count


@wp.func_replay(counter_increment)
def replay_counter_increment(counter: wp.array(dtype=int), counter_index: int, tids: wp.array(dtype=int), tid: int):
    return tids[tid]


@wp.kernel(enable_backward=False)
def count_drag_vertices(
    shape_body: wp.array(dtype=int),
    geo: warp.sim.ModelShapeGeometry,
    # outputs
    vertex_counter: wp.array(dtype=int),
):
    tid = wp.tid()
    shape = tid
    body = shape_body[shape]
    if body == -1:
        return
    geo_type = geo.type[shape]

    if geo_type == wp.sim.GEO_SPHERE:
        wp.atomic_add(vertex_counter, 0, 1)
        return
    if geo_type == wp.sim.GEO_BOX:
        wp.atomic_add(vertex_counter, 0, 6)
        return
    if geo_type == wp.sim.GEO_MESH:
        mesh = wp.mesh_get(geo.source[shape])
        wp.atomic_add(vertex_counter, 0, mesh.indices.shape[0] // 3)
        return
    wp.printf("Unsupported geometry type %d for computing drag vertices\n", geo_type)


@wp.kernel
def generate_drag_vertices(
    shape_body: wp.array(dtype=int),
    shape_X_bs: wp.array(dtype=wp.transform),
    geo: warp.sim.ModelShapeGeometry,
    # outputs
    vertex_counter: wp.array(dtype=int),
    vertex_tids: wp.array(dtype=int),
    drag_vertices: wp.array(dtype=DragVertex),
):
    tid = wp.tid()
    shape = tid
    body = shape_body[shape]
    if body == -1:
        return
    X_bs = shape_X_bs[shape]
    geo_type = geo.type[shape]
    geo_scale = geo.scale[shape]

    if geo_type == wp.sim.GEO_SPHERE:
        radius = geo_scale[0]
        area = 4.0 * wp.PI * radius**2.0
        drag_coefficient = 0.47
        vertex_id = counter_increment(vertex_counter, 0, vertex_tids, tid)
        pos = wp.transform_get_translation(X_bs)
        normal = wp.vec3(0.0)
        omnidirectional = True
        drag_vertices[vertex_id] = DragVertex(
            body,
            pos,
            normal,
            area,
            drag_coefficient,
            omnidirectional,
        )
        return

    if geo_type == wp.sim.GEO_BOX:
        hx, hy, hz = geo_scale[0], geo_scale[1], geo_scale[2]
        drag_coefficient = 1.05
        omnidirectional = False

        # font + back sides
        area = 4.0 * hy * hz
        vertex_id = counter_increment(vertex_counter, 0, vertex_tids, tid)
        pos = wp.transform_point(X_bs, wp.vec3(hx, 0.0, 0.0))
        normal = wp.transform_vector(X_bs, wp.vec3(1.0, 0.0, 0.0))
        drag_vertices[vertex_id] = DragVertex(
            body,
            pos,
            normal,
            area,
            drag_coefficient,
            omnidirectional,
        )
        vertex_id = counter_increment(vertex_counter, 0, vertex_tids, tid)
        pos = wp.transform_point(X_bs, wp.vec3(-hx, 0.0, 0.0))
        normal = wp.transform_vector(X_bs, wp.vec3(-1.0, 0.0, 0.0))
        drag_vertices[vertex_id] = DragVertex(
            body,
            pos,
            normal,
            area,
            drag_coefficient,
            omnidirectional,
        )

        # top + bottom sides
        area = 4.0 * hx * hz
        vertex_id = counter_increment(vertex_counter, 0, vertex_tids, tid)
        pos = wp.transform_point(X_bs, wp.vec3(0.0, hy, 0.0))
        normal = wp.transform_vector(X_bs, wp.vec3(0.0, 1.0, 0.0))
        drag_vertices[vertex_id] = DragVertex(
            body,
            pos,
            normal,
            area,
            drag_coefficient,
            omnidirectional,
        )
        vertex_id = counter_increment(vertex_counter, 0, vertex_tids, tid)
        pos = wp.transform_point(X_bs, wp.vec3(0.0, -hy, 0.0))
        normal = wp.transform_vector(X_bs, wp.vec3(0.0, -1.0, 0.0))
        drag_vertices[vertex_id] = DragVertex(
            body,
            pos,
            normal,
            area,
            drag_coefficient,
            omnidirectional,
        )

        # left + right sides
        area = 4.0 * hx * hy
        vertex_id = counter_increment(vertex_counter, 0, vertex_tids, tid)
        pos = wp.transform_point(X_bs, wp.vec3(0.0, 0.0, hz))
        normal = wp.transform_vector(X_bs, wp.vec3(0.0, 0.0, 1.0))
        drag_vertices[vertex_id] = DragVertex(
            body,
            pos,
            normal,
            area,
            drag_coefficient,
            omnidirectional,
        )
        vertex_id = counter_increment(vertex_counter, 0, vertex_tids, tid)
        pos = wp.transform_point(X_bs, wp.vec3(0.0, 0.0, -hz))
        normal = wp.transform_vector(X_bs, wp.vec3(0.0, 0.0, -1.0))
        drag_vertices[vertex_id] = DragVertex(
            body,
            pos,
            normal,
            area,
            drag_coefficient,
            omnidirectional,
        )
        return

    if geo_type == wp.sim.GEO_MESH:
        mesh = wp.mesh_get(geo.source[shape])
        vertex_count = mesh.indices.shape[0] // 3
        drag_coefficient = 0.5  # XXX there is no drag coefficient for arbitrary triangles
        omnidirectional = False
        for i in range(vertex_count):
            vertex_id = counter_increment(vertex_counter, 0, vertex_tids, tid)
            i0 = mesh.indices[i * 3 + 0]
            i1 = mesh.indices[i * 3 + 1]
            i2 = mesh.indices[i * 3 + 2]
            v0 = mesh.points[i0]
            v1 = mesh.points[i1]
            v2 = mesh.points[i2]
            pos = wp.transform_point(X_bs, (v0 + v1 + v2) / 3.0)
            normal = wp.normalize(wp.cross(v1 - v0, v2 - v0))
            area = wp.length(wp.cross(v1 - v0, v2 - v0)) / 2.0
            drag_vertices[vertex_id] = DragVertex(
                body,
                pos,
                wp.transform_vector(X_bs, normal),
                area,
                drag_coefficient,
                omnidirectional,
            )
        return


@wp.kernel
def compute_drag_wrenches(
    drag_vertices: wp.array(dtype=DragVertex),
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    wind_velocity: wp.vec3,
    # outputs
    body_f: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()
    vertex = drag_vertices[tid]
    tf = body_q[vertex.body]
    vel = body_qd[vertex.body]
    lin_vel = wp.spatial_bottom(vel) - wind_velocity
    ang_vel = wp.spatial_top(vel)
    world_vertex = wp.transform_point(tf, vertex.pos)
    moment_arm = world_vertex - wp.transform_point(tf, body_com[vertex.body])
    world_vertex_vel = lin_vel + wp.cross(moment_arm, ang_vel)
    if vertex.omnidirectional:
        vel_normal = wp.length_sq(world_vertex_vel)
    else:
        world_normal = wp.transform_vector(tf, vertex.normal)
        vel_normal = wp.dot(world_normal, world_vertex_vel)
    if vel_normal > 1e-6:
        force = world_normal * (-vertex.drag_coefficient * vel_normal**2.0 * vertex.area * air_density / 2.0)
        torque = wp.cross(moment_arm, force)
        wp.atomic_add(body_f, vertex.body, wp.spatial_vector(torque, force))


class DroneSimulationPlugin(wp.sim.SemiImplicitIntegratorPlugin):
    def __init__(self, wind_velocity=wp.vec3(0.0)):
        super().__init__()
        self.props = []
        self.props_wp = None
        self.wind_velocity = wind_velocity

    def add_propeller(
        self,
        body: int,
        pos: wp.vec3,
        dir: wp.vec3,
        prop_data: PropellerData,
    ):
        """
        Add a propeller to the scene.
        """
        prop = Propeller()
        prop.body = body
        prop.pos = pos
        prop.dir = wp.normalize(dir)
        for k, v in prop_data.__dict__.items():
            setattr(prop, k, v)
        self.props.append(prop)

    def on_init(self, model, integrator):
        self.props_wp = wp.array(self.props, dtype=Propeller, device=model.device, requires_grad=model.requires_grad)

        # generate drag vertices for all shapes
        self.drag_vertex_counter = wp.zeros(1, dtype=int, device=model.device)
        wp.launch(
            count_drag_vertices,
            dim=model.shape_count,
            inputs=[model.shape_body, model.shape_geo],
            outputs=[self.drag_vertex_counter],
        )
        self.drag_vertex_count = int(self.drag_vertex_counter.numpy()[0])
        self.drag_vertex_tids = wp.zeros(self.drag_vertex_count, dtype=int, device=model.device)
        self.drag_vertices = wp.zeros(self.drag_vertex_count, dtype=DragVertex, device=model.device, requires_grad=model.requires_grad)
        self.drag_vertex_counter.zero_()
        wp.launch(
            generate_drag_vertices,
            dim=model.shape_count,
            inputs=[model.shape_body, model.shape_transform, model.shape_geo],
            outputs=[self.drag_vertex_counter, self.drag_vertex_tids, self.drag_vertices],
        )
        print(f"Generated {self.drag_vertex_count} drag vertices")

    def before_integrate(self, model, state_in, state_out, dt, requires_grad):
        assert self.props_wp is not None, "DroneSimulationPlugin not initialized"

        state_in.body_f.zero_()
        # print(state_in.body_f.numpy())
        # print("control:", state_in.prop_control.numpy())
        wp.launch(
            compute_prop_wrenches,
            dim=len(self.props),
            inputs=[self.props_wp, state_in.prop_control, state_in.body_q, model.body_com],
            outputs=[state_in.body_f],
            device=model.device,
        )
        # print(state_in.body_f.numpy())
        # import numpy as np
        # if np.any(np.isnan(state_in.body_f.numpy())):
        #     raise RuntimeError("NaN in generated body_f", state_in.body_f.numpy())

        wp.launch(
            compute_drag_wrenches,
            dim=self.drag_vertex_count,
            inputs=[self.drag_vertices, state_in.body_q, state_in.body_qd, model.body_com, self.wind_velocity],
            outputs=[state_in.body_f],
        )

    # def after_integrate(self, model, state_in, state_out, dt, requires_grad):
    #     import numpy as np
    #     if np.any(np.isnan(state_out.body_qd.numpy())):
    #         raise RuntimeError("NaN in generated body_qd", state_out.body_qd.numpy())

    def augment_state(self, model, state):
        # add state vector for the propeller control inputs
        state.prop_control = wp.zeros(
            len(self.props), dtype=float, device=model.device, requires_grad=model.requires_grad
        )


@wp.kernel(enable_backward=False)
def update_prop_rotation(
    prop_rotation: wp.array(dtype=float),
    prop_control: wp.array(dtype=float),
    prop_shape: wp.array(dtype=int),
    props: wp.array(dtype=Propeller),
    dt: float,
    shape_transform: wp.array(dtype=wp.transform),
):
    tid = wp.tid()
    prop = props[tid]
    speed = prop_control[tid] * prop.max_speed_square / 5000.0  # a bit slower for better rendering
    wp.atomic_add(prop_rotation, tid, prop.turning_direction * speed * dt)
    shape = prop_shape[tid]
    shape_transform[shape] = wp.transform(prop.pos, wp.quat_from_axis_angle(prop.dir, prop_rotation[tid]))


@wp.kernel
def drone_cost(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    target: wp.vec3,
    prop_control: wp.array(dtype=float),
    step: int,
    horizon_length: int,
    # outputs
    cost: wp.array(dtype=wp.float32),
):
    env_id = wp.tid()
    tf = body_q[env_id]

    pos_drone = wp.transform_get_translation(tf)
    pos_cost = wp.length_sq(pos_drone - target)
    upvector = wp.vec3(0.0, 1.0, 0.0)
    drone_up = wp.transform_vector(tf, upvector)
    upright_cost = wp.length_sq(drone_up - upvector)

    vel_drone = body_qd[env_id]

    # encourage zero velocity
    vel_cost = wp.length_sq(vel_drone)

    control = wp.vec4(prop_control[env_id* 4 + 0], prop_control[env_id* 4 + 1], prop_control[env_id* 4 + 2], prop_control[env_id* 4 + 3])
    control_cost = wp.dot(control, control)

    # time_factor = wp.float(step) / wp.float(horizon_length)
    discount = 0.8 ** wp.float(horizon_length - step - 1) / wp.float(horizon_length) ** 2.0
    # discount = 0.5 ** wp.float(step) / wp.float(horizon_length)
    # discount = 1.0 / wp.float(horizon_length)

    wp.atomic_add(cost, env_id, (1000.0 * pos_cost + 0.05 * control_cost + 0.2 * vel_cost + 10.0 * upright_cost) * discount)


@wp.kernel
def collision_cost(
    body_q: wp.array(dtype=wp.transform),
    obstacle_ids: wp.array(dtype=int, ndim=2),
    shape_X_bs: wp.array(dtype=wp.transform),
    geo: warp.sim.ModelShapeGeometry,
    margin: float,
    weighting: float,
    # outputs
    cost: wp.array(dtype=wp.float32),
):
    env_id, obs_id = wp.tid()
    shape_index = obstacle_ids[env_id, obs_id]

    px = wp.transform_get_translation(body_q[env_id])

    X_bs = shape_X_bs[shape_index]
    X_sb = wp.transform_inverse(X_bs)

    # transform particle position to shape local space
    x_local = wp.transform_point(X_sb, px)

    # geo description
    geo_type = geo.type[shape_index]
    geo_scale = geo.scale[shape_index]

    # evaluate shape sdf
    d = 1.0e6

    if geo_type == wp.sim.GEO_SPHERE:
        d = sphere_sdf(wp.vec3(), geo_scale[0], x_local)

    if geo_type == wp.sim.GEO_BOX:
        d = box_sdf(geo_scale, x_local)

    if geo_type == wp.sim.GEO_CAPSULE:
        d = capsule_sdf(geo_scale[0], geo_scale[1], x_local)

    if geo_type == wp.sim.GEO_CYLINDER:
        d = cylinder_sdf(geo_scale[0], geo_scale[1], x_local)

    if geo_type == wp.sim.GEO_CONE:
        d = cone_sdf(geo_scale[0], geo_scale[1], x_local)

    if geo_type == wp.sim.GEO_MESH:
        mesh = geo.source[shape_index]
        min_scale = wp.min(geo_scale)
        max_dist = margin / min_scale
        d = mesh_sdf(mesh, wp.cw_div(x_local, geo_scale), max_dist)
        d *= min_scale  # TODO fix this, mesh scaling needs to be handled properly

    if geo_type == wp.sim.GEO_SDF:
        volume = geo.source[shape_index]
        xpred_local = wp.volume_world_to_index(volume, wp.cw_div(x_local, geo_scale))
        nn = wp.vec3(0.0, 0.0, 0.0)
        d = wp.volume_sample_grad_f(volume, xpred_local, wp.Volume.LINEAR, nn)

    if geo_type == wp.sim.GEO_PLANE:
        d = plane_sdf(geo_scale[0], geo_scale[1], x_local)

    if d < margin:
        c = margin - d
        wp.atomic_add(cost, env_id, weighting * c)
        # log-barrier function
        # wp.atomic_add(cost, env_id, weighting * wp.log(c))


@wp.kernel(enable_backward=False)
def move_air_molecules(
    env_bounds_lower: wp.vec3,
    env_bounds_upper: wp.vec3,
    wind_velocity: wp.vec3,
    noise_velocity: wp.vec3,
    dt: float,
    seed: int,
    # outputs
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    pos = positions[tid]
    vel = velocities[tid]
    state = wp.rand_init(seed, tid)
    for i in range(3):
        if pos[i] < env_bounds_lower[i] or pos[i] > env_bounds_upper[i]:
            l, u = env_bounds_lower[i], env_bounds_upper[i]
            pos[i] = wp.randf(state, u, l)
    rx = wp.randf(state, -1.0, 1.0) * noise_velocity[0]
    ry = wp.randf(state, -1.0, 1.0) * noise_velocity[1]
    rz = wp.randf(state, -1.0, 1.0) * noise_velocity[2]
    vel = 0.99 * vel + wp.vec3(rx, ry, rz)
    pos += (wind_velocity + vel) * dt
    positions[tid] = pos
    velocities[tid] = vel


class DroneEnvironment(Environment):
    sim_name = "env_drone"

    num_envs = 100

    opengl_render_settings = dict(scaling=3.0, draw_axis=False)
    usd_render_settings = dict(scaling=100.0)

    # use_graph_capture = False

    sim_substeps_euler = 1
    sim_substeps_xpbd = 1

    activate_ground_plane = False

    drone_crossbar_length = 0.2
    drone_crossbar_height = 0.01
    drone_crossbar_width = 0.01

    integrator_type = IntegratorType.EULER

    controllable_dofs = [0, 1, 2, 3]
    control_gains = [0.8] * 4
    control_limits = [(0.1, 1.0)] * 4

    flight_target = wp.vec3(0.0, 0.5, 1.0)

    wind_velocity = wp.vec3(0.5, 0.0, 0.0)

    visualize_air_molecules = True
    air_molecule_bounds_lower = wp.vec3(-10.0, 0.0, -10.0)
    air_molecule_bounds_upper = wp.vec3(10.0, 20.0, 10.0)
    air_molecule_noise_velocity = wp.vec3(3e-3, 3e-3, 3e-3)
    air_molecule_count = 10000

    render_collision_sphere = False

    def __init__(self):
        self.drone_plugin = DroneSimulationPlugin(wind_velocity=self.wind_velocity)
        self.euler_settings["plugins"] = [self.drone_plugin]
        super().__init__()

        # arrays to keep track of propeller rotation just for rendering
        self.prop_rotation = None

        # radius of the collision sphere around the drone
        self.collision_radius = 1.5 * self.drone_crossbar_length

    def setup(self, builder):
        self.prop_shapes = []
        self.obstacle_shapes = []

        def add_prop(drone, i, pos, cw=True):
            normal = wp.vec3(0.0, 1.0, 0.0)
            prop_data = PropellerData(turning_cw=cw)
            self.drone_plugin.add_propeller(drone, pos, normal, prop_data)
            if self.render_mode == RenderMode.OPENGL:
                # add fake propeller geometry
                prop_shape = builder.add_shape_box(
                    drone,
                    pos=pos,
                    hx=prop_data.diameter / 2.0,
                    hy=prop_data.diameter / 25.0,
                    hz=prop_data.diameter / 15.0,
                    density=0.0,
                    has_ground_collision=False,
                    collision_group=i,
                )
                self.prop_shapes.append(prop_shape)

        for i in range(self.num_envs):
            xform = wp.transform(self.env_offsets[i], wp.quat_identity())
            drone = builder.add_body(name=f"drone_{i}", origin=xform)
            # builder.add_shape_box(
            #     drone,
            #     hx=self.drone_crossbar_length * 0.3,
            #     hy=self.drone_crossbar_height * 3.0,
            #     hz=self.drone_crossbar_length * 0.3,
            #     collision_group=i,
            #     density=carbon_fiber_density,
            # )
            builder.add_shape_box(
                drone,
                hx=self.drone_crossbar_length,
                hy=self.drone_crossbar_height,
                hz=self.drone_crossbar_width,
                collision_group=i,
                density=carbon_fiber_density,
            )
            builder.add_shape_box(
                drone,
                hx=self.drone_crossbar_width,
                hy=self.drone_crossbar_height,
                hz=self.drone_crossbar_length,
                collision_group=i,
                density=carbon_fiber_density,
            )
            if self.render_collision_sphere:
                builder.add_shape_sphere(
                    drone,
                    radius=self.collision_radius,
                    density=0.0,
                    collision_group=i,
                )

            add_prop(drone, i, wp.vec3(self.drone_crossbar_length, 0.0, 0.0), False)
            add_prop(drone, i, wp.vec3(-self.drone_crossbar_length, 0.0, 0.0))
            add_prop(drone, i, wp.vec3(0.0, 0.0, self.drone_crossbar_length))
            add_prop(drone, i, wp.vec3(0.0, 0.0, -self.drone_crossbar_length), False)

            obstacle_shapes = [
                # builder.add_shape_plane(
                #     pos=(0.0, -0.5, 0.0),
                #     width=0.0,
                #     length=0.0,
                # ),
                builder.add_shape_capsule(
                    -1,
                    pos=(0.5, 0.5, 0.5),
                    radius=0.15,
                    collision_group=i,
                ),
                # builder.add_shape_capsule(
                #     -1,
                #     pos=(-0.5, 0.5, 0.5),
                #     radius=0.15,
                #     collision_group=i,
                # ),
                # builder.add_shape_capsule(
                #     -1,
                #     pos=(0.5, 0.5, -0.5),
                #     radius=0.15,
                #     collision_group=i,
                # ),
                builder.add_shape_capsule(
                    -1,
                    pos=(-0.5, 0.5, -0.5),
                    radius=0.15,
                    collision_group=i,
                ),
            ]
            self.obstacle_shapes.append(obstacle_shapes)

            self.obstacle_shape_count_per_env = len(obstacle_shapes)

    def before_simulate(self):
        self.obstacle_shapes_wp = wp.array(self.obstacle_shapes, dtype=int, device=self.device)

        if self.visualize_air_molecules:
            lower = np.array(self.air_molecule_bounds_lower)
            upper = np.array(self.air_molecule_bounds_upper)
            pos = np.random.uniform(size=(self.air_molecule_count, 3)) * (upper - lower) + lower
            self.air_molecule_positions = wp.array(pos, dtype=wp.vec3, device=self.device)
            self.air_molecule_velocities = wp.array(
                np.tile(np.array(self.wind_velocity), (self.air_molecule_count, 1)), dtype=wp.vec3, device=self.device
            )
            self.air_molecule_colors = np.tile((0.1, 0.2, 0.8), (self.air_molecule_count, 1))
            self.air_molecule_random_seed = 123

        if self.render_mode != RenderMode.NONE:
            self.prop_shape = wp.array(self.prop_shapes, dtype=int, device=self.device)
            self.prop_rotation = wp.zeros(len(self.prop_shapes), dtype=float, device=self.device)

        if self.render_mode == RenderMode.OPENGL:
            import pyglet

            self.count_target_swaps = 0
            self.possible_targets = [
                wp.vec3(0.0, 0.5, 1.0),
                wp.vec3(1.0, 0.5, 0.0),
                wp.vec3(0.0, 0.5, -1.0),
                wp.vec3(-1.0, 0.5, 0.0),
            ]

            def swap_target(key, modifiers):
                if key == pyglet.window.key.N:
                    self.count_target_swaps += 1
                    self.flight_target = self.possible_targets[self.count_target_swaps % len(self.possible_targets)]
                    self.invalidate_cuda_graph = True

            self.renderer.register_key_press_callback(swap_target)

    # def custom_update(self):
    #     self.state.prop_control.fill_(wp.sin(self.sim_time) * 1e-4 + 0.187)
    # def custom_update(self):
    #     self.state.prop_control.fill_(1.0)

    def custom_render(self, render_state, renderer=None):
        if renderer is None:
            renderer = self.renderer
        for i in range(self.num_envs):
            renderer.render_sphere(
                f"target_{i}",
                self.flight_target + wp.vec3(self.env_offsets[i]),
                wp.quat_identity(),
                radius=0.05,
            )
            # print("target", self.flight_target + self.env_offsets[i])

        if self.visualize_air_molecules:
            wp.launch(
                move_air_molecules,
                dim=self.air_molecule_count,
                inputs=[
                    self.air_molecule_bounds_lower,
                    self.air_molecule_bounds_upper,
                    self.wind_velocity,
                    self.air_molecule_noise_velocity,
                    self.frame_dt,
                    self.air_molecule_random_seed,
                ],
                outputs=[self.air_molecule_positions, self.air_molecule_velocities],
            )
            self.air_molecule_random_seed += 1
            renderer.render_points(
                "air_molecules",
                self.air_molecule_positions.numpy(),
                radius=0.01,
                colors=self.air_molecule_colors,
            )

        if isinstance(renderer, wp.sim.render.SimRendererOpenGL):
            # directly animate shape instances in renderer because model.shape_transform is not considered
            # by the OpenGLRenderer online
            if self.renderer._wp_instance_transforms is None:
                return
            wp.launch(
                update_prop_rotation,
                dim=len(self.prop_shapes),
                inputs=[
                    self.prop_rotation,
                    render_state.prop_control,
                    self.prop_shape,
                    self.drone_plugin.props_wp,
                    self.frame_dt,
                ],
                outputs=[self.renderer._wp_instance_transforms],
                device=self.device,
            )
        if isinstance(renderer, wp.sim.render.SimRendererUsd):
            # update shape transforms in model
            wp.launch(
                update_prop_rotation,
                dim=len(self.prop_shapes),
                inputs=[
                    self.prop_rotation,
                    render_state.prop_control,
                    self.prop_shape,
                    self.drone_plugin.props_wp,
                    self.frame_dt,
                ],
                outputs=[self.model.shape_transform],
                device=self.device,
            )
            tfs = self.model.shape_transform.numpy()
            for i in range(4):
                tf = tfs[i + 2]
                renderer.render_ref(f"body_0_drone_0/shape_{i + 2}", "", pos=None, rot=tf[-4:], scale=None)

    @property
    def control(self):
        # overwrite control property to actuate propellers, not joints
        return self.control.prop_control

    def evaluate_cost(self, state: wp.sim.State, control: wp.sim.Control, cost: wp.array, step: int, horizon_length: int):
        wp.launch(
            drone_cost,
            dim=self.num_envs,
            inputs=[state.body_q, state.body_qd, self.flight_target, control.prop_control, step, horizon_length],
            outputs=[cost],
            device=self.device,
        )
        wp.launch(
            collision_cost,
            dim=(self.num_envs, self.obstacle_shape_count_per_env),
            inputs=[
                state.body_q,
                self.obstacle_shapes_wp,
                self.model.shape_transform,
                self.model.shape_geo,
                self.collision_radius,
                1e4,
            ],
            outputs=[cost],
            device=self.device,
        )
        # import numpy as np
        # if np.any(np.isnan(cost.numpy())):
        #     raise RuntimeError("NaN cost", cost.numpy())


if __name__ == "__main__":
    run_env(DroneEnvironment)
