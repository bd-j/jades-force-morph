#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""optimal_all.py - Do optimization of source parameters in each patch,
with linear optimization of fluxes conditional on shapes at the end.
"""

import os, sys, shutil
import argparse, time
import socket
import logging
import json

import numpy as np

from forcepho.utils import NumpyEncoder, read_config

from utils import get_superscene, get_patcher
from child import optimization_task

try:
    import pycuda
    import pycuda.autoinit
    HASGPU = True
except:
    print("NO PYCUDA")
    HASGPU = False


def do_child(patcher, task, config=None):
    logger = logging.getLogger(f'child-1')
    # --- Event Loop ---
    for _ in range(1):
        # if shutdown break and quit
        if task is None:
            break

        answer = optimization_task(patcher, task, config, logger)

        # --- blocking send to parent, free GPU memory ---
        logger.info(f"Sent results for patch {task['taskID']} with RA={region.ra}")

    return answer


if __name__ == "__main__":

    # --- Arguments ---
    parser = argparse.ArgumentParser()
    # input
    parser.add_argument("--config_file", type=str, default="./morph_mosaic_config.yml")
    parser.add_argument("--raw_catalog", type=str, default=None)
    parser.add_argument("--seed_index", type=int, default=-1)
    parser.add_argument("--maxactive_per_patch", type=int, default=None)
    parser.add_argument("--max_fixed", type=int, default=60)
    parser.add_argument("--strict", type=int, default=0)
    parser.add_argument("--tweak_background", type=str, default="")
    # bounds
    parser.add_argument("--minflux", type=float, default=None)
    parser.add_argument("--maxfluxfactor", type=float, default=0)
    parser.add_argument("--n_pix_sw", type=float, default=4,
                        help="positional bounds are \pm 0.03'' \\times `n_pix_sw`")
    # output
    parser.add_argument("--write_residuals", type=int, default=1)
    parser.add_argument("--outbase", type=str, default="../output/multipatchserial/")
    # sampling
    parser.add_argument("--use_gradients", type=int, default=1)
    parser.add_argument("--linear_optimize", type=int, default=1)
    parser.add_argument("--gtol", type=float, default=1e-5)
    parser.add_argument("--add_barriers", type=int, default=0)

    # --- Logger ---
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger('dispatcher-parent')
    logger.info(f'Starting parent on {socket.gethostname()}')

    # --- Configure ---
    args = parser.parse_args()
    config = read_config(args.config_file, args)
    config.patch_dir = os.path.join(config.outbase, "patches")
    config.bounds_kwargs["n_pix"] = config.n_pix_sw
    [os.makedirs(a, exist_ok=True) for a in (config.outbase, config.patch_dir)]
    _ = shutil.copy(config.config_file, config.outbase)
    try:
        with open(f"{config.outbase}/config.json", "w") as cfg:
            json.dump(vars(config), cfg, cls=NumpyEncoder)
    except(ValueError):
        logging.info("Config json not written.")

    # --- Get patch dispatcher and maker ---
    sceneDB, bands = get_superscene(config, logger, sqrtq_range=(0.45, 0.99))
    patcher = get_patcher(config, logger)

    # --- Checkout scenes in a loop --- (parent)
    seed = config.seed_index
    taskID = 0
    while sceneDB.undone & (taskID < 100):
        # how many tries before we decide there are no regions to be checked out?
        ntry = getattr(config, "ntry_checkout", 100)
        for _ in range(ntry):
            region, active, fixed = sceneDB.checkout_region(seed_index=seed, max_fixed=config.max_fixed)
            seed = -1
            if active is not None:
                if len(active) != len(np.unique(active["source_index"])):
                    logger.error(f"Duplicate source in region!!!")
                    logger.info(f"seed_index={active[0]['source_index']}, ID={active[0]['id']}")
                    logger.info(f"indices: {active['source_index']}")
                    logger.info(f"IDs: {active['id']}")
                    sceneDB.checkin_region(active, fixed, 0)
                    try:
                        sceneDB.writeout(f"{config.outbase}/scene_with_dupes_task{taskID}.fits")
                    except:
                        pass
                else:
                    break
        else:
            logger.error(f'Failed to checkout region')
            break
        logger.info(f"Checked out scene seeded on source index {active[0]['source_index']} "
                    f"with ID={active[0]['id']}")

        # build the chore
        bounds, cov = sceneDB.bounds_and_covs(active["source_index"])
        chore = {'region': region, 'active': active, 'fixed': fixed,
                 'bounds': bounds, 'cov': cov, 'taskID': taskID,
                 'bands': bands, 'shape_cols': sceneDB.shape_cols}

        # submit the task and get the results
        if HASGPU:
            result = do_child(patcher, chore, config)
        else:
            break

        if "NaN" in result["out"].message:
            logger.error("got a NaN during optimization!!! Quitting.")
            break

        # Check results back in
        sceneDB.checkin_region(result['final'], result['out'].fixed,
                               sceneDB.target_niter,
                               block_covs=result['covs'],
                               new_bounds=result['bounds'],
                               taskID=taskID)
        taskID += 1

        # writeout after every patch in case we have to interrupt
        sceneDB.writeout()
