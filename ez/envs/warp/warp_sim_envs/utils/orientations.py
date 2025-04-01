import warp as wp

@wp.func
def rotation_distance_function(
    object_rot: wp.quat,
    target_rot: wp.quat
):
    """Rotation distance function from quat space. Min rotation between 2 orientation, always +"""
    quat_diff = object_rot * wp.quat_inverse(target_rot) #quat_mult(object_rot, quat_conj(target_rot))
    sin_half_angle = wp.sqrt(quat_diff[0]*quat_diff[0] + quat_diff[1]*quat_diff[1] + quat_diff[2]*quat_diff[2]) 
    distance = 2.0 * wp.asin(wp.min(sin_half_angle, 1.0))
    return distance
