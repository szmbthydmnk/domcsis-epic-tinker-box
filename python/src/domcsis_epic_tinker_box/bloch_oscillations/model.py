from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ModelParams:
    J: float = 1.0
    ht: float = -0.15
    hl: float = 0.15
    dt: float = 0.25
    L: int = 7
    layers_max: int = 40


@dataclass(frozen=True)
class RunConfig:
    backend_mode: str = "ideal"
    fake_backend_name: str | None = None
    initial_state: str = "all_up"
    use_cnot_zz: bool = False
    shots: int = 8192
    optimization_level: int = 1
    use_parallel_u1: bool = False
    trotter_method: str = "even_odd"


def center_index(params: ModelParams) -> int:
    return math.ceil(params.L / 2 - 1)
