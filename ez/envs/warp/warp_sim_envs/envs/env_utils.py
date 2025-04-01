import warp as wp
import numpy as np
from scipy.spatial.transform import Rotation

def set_ground_plane(
    builder,
    pos,
    rot,
    ke: float = None,
    kd: float = None,
    kf: float = None,
    mu: float = None,
    restitution: float = None,
):
    normal = Rotation.from_quat(rot).as_matrix() @ np.array([0., 1., 0.])
    d = np.dot(pos, normal)
    # print(normal, d)
    builder._ground_params = {
        'plane': [*normal, d],
        'pos': pos,
        'rot': rot,
        'width': 0.0,
        'length': 0.0,
        'ke': ke if ke is not None else builder.default_shape_ke,
        'kd': kd if kd is not None else builder.default_shape_kd,
        'kf': kf if kf is not None else builder.default_shape_kf,
        'mu': mu if mu is not None else builder.default_shape_mu,
        'restitution': restitution if restitution is not None else builder.default_shape_restitution
    }