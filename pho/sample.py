#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""multi_patch_serial.py - test working with sceneDB, checkout/checking with
multiple patches (run in serial)
"""

import os, sys, shutil
import argparse, time
import socket
import logging
import json

import numpy as np

from forcepho.utils import NumpyEncoder, read_config

from utils import get_superscene, get_patcher
from child import accomplish_task, sampling

try:
    import pycuda
    import pycuda.autoinit
    HASGPU = True
except:
    print("NO PYCUDA")
    HASGPU = False


def do_child(patcher, task, config=None):
    rank = 1
    global logger
    logger = logging.getLogger(f'child-{rank}')

    # --- Event Loop ---
    for _ in range(1):
        # if shutdown break and quit
        if task is None:
            break

        taskID = task["taskID"]
        answer = accomplish_task(patcher, task, config, logger,
                                 method=sampling)

        # --- blocking send to parent, free GPU memory ---
        logger.info(f"Child {rank} sent answer for patch {taskID}")

    return answer


if __name__ == "__main__":

    # --- Arguments ---
    parser = argparse.ArgumentParser()
    # input
    parser.add_argument("--config_file", type=str, default="./hlf_config.yml")
    parser.add_argument("--raw_catalog", type=str, default=None)
    parser.add_argument("--seed_index", type=int, default=-1)
    parser.add_argument("--maxactive_per_patch", type=int, default=None)
    parser.add_argument("--max_fixed", type=int, default=60)
    parser.add_argument("--strict", type=int, default=0)
    parser.add_argument("--tweak_background", type=str, default="")
    # bounds
    parser.add_argument("--minflux", type=float, default=None)
    parser.add_argument("--maxfluxfactor", type=float, default=0)
    # output
    parser.add_argument("--write_residuals", type=int, default=1)
    parser.add_argument("--outbase", type=str, default="../output/multipatchserial/")
    # sampling
    parser.add_argument("--add_barriers", type=int, default=0)
    parser.add_argument("--full_cov", type=int, default=0)
    parser.add_argument("--sampling_draws", type=int, default=None)
    parser.add_argument("--max_treedepth", type=int, default=None)
    parser.add_argument("--warmup", type=int, nargs="*", default=None)
    parser.add_argument("--progressbar", type=int, default=0)
    parser.add_argument("--discard_tuning", type=int, default=1)

    # --- Logger
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger('dispatcher-parent')
    logger.info(f'Starting parent on {socket.gethostname()}')

    # --- Configure ---
    args = parser.parse_args()
    config = read_config(args.config_file, args)
    config.patch_dir = os.path.join(config.outbase, "patches")
    [os.makedirs(a, exist_ok=True) for a in (config.outbase, config.patch_dir)]
    _ = shutil.copy(config.config_file, config.outbase)
    try:
        with open(f"{config.outbase}/config.json", "w") as cfg:
            json.dump(vars(config), cfg, cls=NumpyEncoder)
    except(ValueError):
        logger.info("Config json not written.")

    # --- Get patch dispatcher and maker ---
    sceneDB, bands = get_superscene(config, logger, sqrtq_range=(0.45, 0.99))
    patcher = get_patcher(config, logger)

    # --- checkout scenes in a loop --- (parent)
    seed = config.seed_index
    taskID = 0
    while sceneDB.undone:
        # Try to check out a scene
        ntry = getattr(config, "ntry_checkout", 1000)
        for _ in range(ntry):
            region, active, fixed = sceneDB.checkout_region(seed_index=seed, max_fixed=config.max_fixed)
            seed = -1
            if active is not None:
                if len(active) != len(np.unique(active["source_index"])):
                    logger.error(f"Duplicate source in region!!!")
                    sceneDB.checkin_region(active, fixed, 0)
                else:
                    break
        else:
            logger.error(f'Failed to checkout region')
            break

        big = None

        logger.info(f"Checked out scene centered on source index {active[0]['source_index']} "
                    f"with ID={active[0]['id']}")

        # build the chore
        bounds, cov = sceneDB.bounds_and_covs(active["source_index"])
        chore = {'region': region,
                 'active': active, 'fixed': fixed, 'big': big,
                 'bounds': bounds, 'cov': cov, 'taskID': taskID,
                 'bands': bands, 'shape_cols': sceneDB.shape_cols}

        # submit the task and get the results
        if HASGPU:
            result = do_child(patcher, chore, config)
        else:
            break

        # Check results back in
        sceneDB.checkin_region(result['final'], result['out'].fixed,
                               config.sampling_draws,
                               block_covs=result['covs'],
                               taskID=taskID)
        taskID += 1

        #if result["out"].ncall > 3e5:
        #    logger.error("Something went bad in the tuning, quitting!")
        #    break
        # writeout after every patch in case we have to interrupt
        sceneDB.writeout()

    logger.info(f'SuperScene is done, parent is finished, shutting down.')