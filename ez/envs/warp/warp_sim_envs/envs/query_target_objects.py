import warp as wp
import warp.sim

from abc import ABC, abstractmethod
from typing import List

class QueryTargetObjects(ABC):

    def get_target_keys(self) -> List[str]:
        """Return list of keys of target"""
        raise NotImplementedError()

    @abstractmethod
    def query_target_body_q(self, state: warp.sim.State, key: str, buffer: wp.array):
        """Should save target object pose into buffer. Dimension of buffer (num_envs, BODY_Q_DIM) where BODY_Q_DIM = 7 (translation (3) + orientation(4)). The key is used to select the objects"""
        raise NotImplementedError()
    