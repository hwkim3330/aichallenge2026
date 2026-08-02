from __future__ import annotations

import numpy as np
from gymnasium import spaces

from context.context_types import StepContext
from observation.interfaces import ObservationBuilder

# Must match tiny_lidar_net_controller's config.yaml so the RL-trained policy
# stays compatible with the deployed inference core.
LIDAR_INPUT_DIM = 750
LIDAR_MAX_RANGE_M = 30.0


class LidarSpeedObservationBuilder(ObservationBuilder):
    @property
    def observation_space(self) -> spaces.Dict:
        return spaces.Dict(
            {
                "lidar": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(LIDAR_INPUT_DIM,),
                    dtype=np.float32,
                ),
                "speed": spaces.Box(
                    low=0.0,
                    high=np.finfo(np.float32).max,
                    shape=(1,),
                    dtype=np.float32,
                ),
            }
        )

    def build(self, context: StepContext) -> tuple[dict[str, np.ndarray], StepContext]:
        raw_ranges = context.env_state.get_value("lidar_ranges")
        if raw_ranges is None or len(raw_ranges) == 0:
            lidar = np.zeros((LIDAR_INPUT_DIM,), dtype=np.float32)
        else:
            ranges = np.array(raw_ranges, dtype=np.float32)
            ranges[np.isnan(ranges)] = 0.0
            ranges[np.isinf(ranges)] = LIDAR_MAX_RANGE_M
            ranges = np.clip(ranges, 0.0, LIDAR_MAX_RANGE_M)

            current_len = len(ranges)
            if current_len > LIDAR_INPUT_DIM:
                idx = np.linspace(0, current_len - 1, LIDAR_INPUT_DIM, dtype=int)
                ranges = ranges[idx]
            elif current_len < LIDAR_INPUT_DIM:
                ranges = np.pad(ranges, (0, LIDAR_INPUT_DIM - current_len), 'constant')

            lidar = (ranges / LIDAR_MAX_RANGE_M).astype(np.float32)

        speed_value = float(context.env_state.get_value("vehicle_speed_mps", 0.0))
        speed = np.array([max(0.0, speed_value)], dtype=np.float32)

        observation = {
            "lidar": lidar,
            "speed": speed,
        }
        return observation, context
