import warp as wp

from warp.sim import ModelShapeMaterials, ModelShapeGeometry, Model, State


@wp.kernel
def eval_rigid_contact_impulses(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    body_inv_mass: wp.array(dtype=float),
    body_inv_inertia: wp.array(dtype=wp.mat33),
    shape_materials: ModelShapeMaterials,
    geo: ModelShapeGeometry,
    shape_body: wp.array(dtype=int),
    contact_count: wp.array(dtype=int),
    contact_point0: wp.array(dtype=wp.vec3),
    contact_point1: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_shape0: wp.array(dtype=int),
    contact_shape1: wp.array(dtype=int),
    force_in_world_frame: bool,
    dt: float,
    baumgarte_erp: float,
    # outputs
    body_impulses: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()

    count = contact_count[0]
    if tid >= count:
        return

    # retrieve contact thickness, compute average contact material properties
    ke = 0.0  # restitution coefficient
    kd = 0.0  # damping coefficient
    kf = 0.0  # friction coefficient
    mu = 0.0  # coulomb friction
    restitution = 0.0
    mat_nonzero = 0
    thickness_a = 0.0
    thickness_b = 0.0
    shape_a = contact_shape0[tid]
    shape_b = contact_shape1[tid]
    if shape_a == shape_b:
        return
    body_a = -1
    body_b = -1
    if shape_a >= 0:
        mat_nonzero += 1
        ke += shape_materials.ke[shape_a]
        kd += shape_materials.kd[shape_a]
        kf += shape_materials.kf[shape_a]
        mu += shape_materials.mu[shape_a]
        restitution += shape_materials.restitution[shape_a]
        thickness_a = geo.thickness[shape_a]
        body_a = shape_body[shape_a]
    if shape_b >= 0:
        mat_nonzero += 1
        ke += shape_materials.ke[shape_b]
        kd += shape_materials.kd[shape_b]
        kf += shape_materials.kf[shape_b]
        mu += shape_materials.mu[shape_b]
        restitution += shape_materials.restitution[shape_b]
        thickness_b = geo.thickness[shape_b]
        body_b = shape_body[shape_b]
    if mat_nonzero > 0:
        ke = ke / float(mat_nonzero)
        kd = kd / float(mat_nonzero)
        kf = kf / float(mat_nonzero)
        mu = mu / float(mat_nonzero)
        restitution = restitution / float(mat_nonzero)

    # body position in world space
    n = contact_normal[tid]
    bx_a = contact_point0[tid]
    bx_b = contact_point1[tid]
    if body_a >= 0:
        X_wb_a = body_q[body_a]
        X_com_a = body_com[body_a]
        bx_a = wp.transform_point(X_wb_a, bx_a) - thickness_a * n
        r_a = bx_a - wp.transform_point(X_wb_a, X_com_a)

    if body_b >= 0:
        X_wb_b = body_q[body_b]
        X_com_b = body_com[body_b]
        bx_b = wp.transform_point(X_wb_b, bx_b) + thickness_b * n
        r_b = bx_b - wp.transform_point(X_wb_b, X_com_b)

    d = wp.dot(n, bx_a - bx_b)

    if d >= 0.0:
        return

    # compute contact point velocity
    bv_a = wp.vec3(0.0)
    bv_b = wp.vec3(0.0)
    if body_a >= 0:
        inv_I_a = body_inv_inertia[body_a]
        inv_m_a = body_inv_mass[body_a]
        body_v_s_a = body_qd[body_a]
        body_w_a = wp.spatial_top(body_v_s_a)
        body_v_a = wp.spatial_bottom(body_v_s_a)
        if force_in_world_frame:
            bv_a = body_v_a + wp.cross(body_w_a, bx_a)
        else:
            bv_a = body_v_a + wp.cross(body_w_a, r_a)

    if body_b >= 0:
        inv_I_b = body_inv_inertia[body_b]
        inv_m_b = body_inv_mass[body_b]
        body_v_s_b = body_qd[body_b]
        body_w_b = wp.spatial_top(body_v_s_b)
        body_v_b = wp.spatial_bottom(body_v_s_b)
        if force_in_world_frame:
            bv_b = body_v_b + wp.cross(body_w_b, bx_b)
        else:
            bv_b = body_v_b + wp.cross(body_w_b, r_b)

    # # relative velocity
    # v = bv_a - bv_b

    # # print(v)

    # # decompose relative velocity
    # vn = wp.dot(n, v)
    # vt = v - n * vn

    # # contact elastic
    # fn = d * ke

    # # contact damping
    # fd = wp.min(vn, 0.0) * kd * wp.step(d)

    baumgarte_rel_vel = baumgarte_erp * d / dt
    rel_vel = bv_a - bv_b
    normal_rel_vel = wp.dot(n, rel_vel)
    if normal_rel_vel >= 0.0:
        return

    temp1 = inv_I_a * wp.cross(r_a, n)
    temp2 = inv_I_b * wp.cross(r_b, n)
    ang = wp.dot(n, wp.cross(temp1, r_a) + wp.cross(temp2, r_b))
    denom = inv_m_a + inv_m_b + ang
    impulse = -(1.0 + restitution) * normal_rel_vel - baumgarte_rel_vel
    impulse /= denom
    impulse_vector = 0.2 * impulse * n
    if body_a >= 0:
        wp.atomic_add(
            body_impulses,
            body_a,
            wp.spatial_vector(wp.cross(r_a, impulse_vector), impulse_vector),
        )
    if body_b >= 0:
        wp.atomic_sub(
            body_impulses,
            body_b,
            wp.spatial_vector(wp.cross(r_b, impulse_vector), impulse_vector),
        )

    lateral_rel_vel = rel_vel - normal_rel_vel * n
    lateral_rel_vel_norm = wp.length(lateral_rel_vel)
    if lateral_rel_vel_norm < 1e-6:
        return
    friction_impulse_trial = lateral_rel_vel_norm / denom

    friction_coeffcient = mu
    if friction_impulse_trial < impulse * friction_coeffcient:
        friction_impulse = friction_impulse_trial
    else:
        friction_impulse = friction_coeffcient * impulse

    friction_dir = lateral_rel_vel * (1.0 / lateral_rel_vel_norm)
    impulse_vector = 0.2 * friction_impulse * friction_dir
    if body_a >= 0:
        wp.atomic_sub(
            body_impulses,
            body_a,
            wp.spatial_vector(wp.cross(r_a, impulse_vector), impulse_vector),
        )
    if body_b >= 0:
        wp.atomic_add(
            body_impulses,
            body_b,
            wp.spatial_vector(wp.cross(r_b, impulse_vector), impulse_vector),
        )


@wp.kernel
def apply_body_impulses(
    q_in: wp.array(dtype=wp.transform),
    qd_in: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    body_I: wp.array(dtype=wp.mat33),
    body_inv_m: wp.array(dtype=float),
    body_inv_I: wp.array(dtype=wp.mat33),
    deltas: wp.array(dtype=wp.spatial_vector),
    constraint_inv_weights: wp.array(dtype=float),
    dt: float,
    velocity_clipping: float,
    # outputs
    q_out: wp.array(dtype=wp.transform),
    qd_out: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()
    inv_m = body_inv_m[tid]
    if inv_m == 0.0:
        q_out[tid] = q_in[tid]
        qd_out[tid] = qd_in[tid]
        return
    inv_I = body_inv_I[tid]

    tf = q_in[tid]
    delta = deltas[tid]

    p0 = wp.transform_get_translation(tf)
    q0 = wp.transform_get_rotation(tf)

    weight = 1.0
    if constraint_inv_weights:
        inv_weight = constraint_inv_weights[tid]
        if inv_weight > 0.0:
            weight = 1.0 / inv_weight

    dp = wp.spatial_bottom(delta) * (inv_m * weight)
    dq = wp.spatial_top(delta) * weight
    dq = wp.quat_rotate(q0, inv_I * wp.quat_rotate_inv(q0, dq))

    # update orientation
    q1 = q0 + 0.5 * wp.quat(dq * dt, 0.0) * q0
    q1 = wp.normalize(q1)

    # update position
    com = body_com[tid]
    x_com = p0 + wp.quat_rotate(q0, com)
    p1 = x_com + dp * dt
    p1 -= wp.quat_rotate(q1, com)

    q_out[tid] = wp.transform(p1, q1)

    v0 = wp.spatial_bottom(qd_in[tid])
    w0 = wp.spatial_top(qd_in[tid])

    # update linear and angular velocity
    v1 = v0 + dp
    # angular part (compute in body frame)
    wb = wp.quat_rotate_inv(q0, w0 + dq)
    tb = -wp.cross(wb, body_I[tid] * wb)  # coriolis forces
    w1 = wp.quat_rotate(q0, wb + inv_I * tb * dt)

    # XXX this improves gradient stability
    if wp.length(v1) < velocity_clipping:
        v1 = wp.vec3(0.0)
    if wp.length(w1) < velocity_clipping:
        w1 = wp.vec3(0.0)

    qd_out[tid] = wp.spatial_vector(w1, v1)


def eval_rigid_contact_impulses(
    model: Model,
    state_in: State,
    state_out: State,
    dt: float,
    body_impulses: wp.array = None,
    baumgarte_erp: float = 0.1,
    velocity_clipping: float = 1e-4,
):
    if body_impulses is None:
        body_impulses = wp.zeros_like(state_in.body_qd)
    else:
        body_impulses.zero_()
    wp.launch(
        kernel=eval_rigid_contact_impulses,
        dim=model.rigid_contact_max,
        inputs=[
            state_in.body_q,
            state_in.body_qd,
            model.body_com,
            model.body_inv_mass,
            model.body_inv_inertia,
            model.shape_materials,
            model.shape_geo,
            model.shape_body,
            model.rigid_contact_count,
            model.rigid_contact_point0,
            model.rigid_contact_point1,
            model.rigid_contact_normal,
            model.rigid_contact_shape0,
            model.rigid_contact_shape1,
            False,
            dt,
            baumgarte_erp,
        ],
        outputs=[body_impulses],
        device=model.device,
    )

    wp.launch(
        kernel=apply_body_impulses,
        dim=model.body_count,
        inputs=[
            state_in.body_q,
            state_in.body_qd,
            model.body_com,
            model.body_inertia,
            model.body_inv_mass,
            model.body_inv_inertia,
            body_impulses,
            None,
            dt,
            velocity_clipping,
        ],
        outputs=[
            state_out.body_q,
            state_out.body_qd,
        ],
        device=model.device,
    )
