from env.tasks.dexplore_inspire import Dexplore_Inspire
from env.tasks.dexplore_leap import Dexplore_Leap
from env.tasks.dexplore_allegro import Dexplore_Allegro
from env.tasks.dexplore_shadow import Dexplore_Shadow
from env.tasks.dexplore_dexhand021 import Dexplore_Dexhand021
from env.tasks.dexplore_distill import Dexplore_Distill
from env.tasks.vec_task_wrappers import VecTaskPythonWrapper, VecTaskDAggerWrapper

from isaacgym import rlgpu

import numpy as np

TASK_REGISTRY = {
    'Dexplore_Inspire': Dexplore_Inspire,
    'Dexplore_Leap': Dexplore_Leap,
    'Dexplore_Allegro': Dexplore_Allegro,
    'Dexplore_Shadow': Dexplore_Shadow,
    'Dexplore_Dexhand021': Dexplore_Dexhand021,
    'Dexplore_Distill': Dexplore_Distill,
}


def parse_task(args, cfg, cfg_train, sim_params, distill=False):
    cfg["seed"] = cfg_train.get("seed", -1)
    cfg["env"]["seed"] = cfg["seed"]
    cfg["env"]["is_test"] = getattr(args, "test", False)

    task_cls = TASK_REGISTRY.get(args.task)
    if task_cls is None:
        raise ValueError(
            f"Unrecognized task: {args.task}\n"
            f"Available tasks: {list(TASK_REGISTRY.keys())}"
        )

    task = task_cls(
        cfg=cfg,
        sim_params=sim_params,
        physics_engine=args.physics_engine,
        device_type=args.device,
        device_id=args.device_id,
        headless=args.headless,
    )

    wrapper_cls = VecTaskDAggerWrapper if distill else VecTaskPythonWrapper
    env = wrapper_cls(
        task,
        args.rl_device,
        cfg_train.get("clip_observations", np.inf),
        cfg_train.get("clip_actions", 1.0),
    )

    return task, env
