# -*- coding: utf-8 -*-

import os, json, logging
import numpy as np
from numpy.linalg import LinAlgError

from forcepho.fitting import run_opt, run_lmc
from forcepho.utils import write_residuals

from utils import optimize_linear, dump_design_matrices


def write_to_disk(out, outroot, model, config):

    # --- write the chain and meta-data for this task ---
    outfile = f"{outroot}_samples.h5"
    try:
        out.config = json.dumps(vars(config))
    except(TypeError):
        pass
    out.dump_to_h5(outfile)

    # --- Write image data and residuals if requested ---
    if config.write_residuals:
        outfile = f"{outroot}_residuals.h5"
        q = out.chain[-1, :]  # last position in chain
        residual = model.residuals(q)
        write_residuals(model.patch, outfile, residuals=residual)


def optimization_task(patcher, task, config=None, logger=None):

    # --- unpack all the task variables ---
    region, active, fixed = task['region'], task['active'], task['fixed']
    bounds, cov = task['bounds'], task['cov']
    bands, shape_cols = task['bands'], task['shape_cols']
    sourceID, taskID = active[0]["source_index"], task["taskID"]
    del task

    # --- log receipt ---
    logger.info(f"Received RA {region.ra}, DEC {region.dec} with tag {taskID}")
    logger.info(f"Tag {taskID} has {len(active)} sources seeded on source index {sourceID}")
    if fixed is None:
        logger.info("No fixed sources")

    # --- get pixel data and metadata ---
    patcher.build_patch(region, None, allbands=bands, tweak_background=config.tweak_background)
    logger.info(f"Prepared patch with {patcher.npix} pixels.")
    if config.sampling_draws == 0:
        return patcher

    # --- Prepare model ---
    model, q = patcher.prepare_model(active=active, fixed=fixed,
                                     bounds=bounds, shapes=shape_cols)
    if config.add_barriers:
        logger.info("Adding edge prior to model.")
        from forcepho.priors import ExpBeta
        model._lnpriorfn = ExpBeta(model.transform.lower, model.transform.upper)

    # --- Run fit/optimization ---
    model.ndof = patcher.npix
    model.sampling = False
    opt, scires = run_opt(model, q.copy(), jac=config.use_gradients,
                          disp=True, gtol=config.gtol)

    # --- clean up ---
    model.sampling = True
    model._lnpriorfn = None
    logger.info(f"Optimization complete ({model.ncall} calls).")
    logger.info(f"{scires.message}")
    out, step, stats = opt, None, None
    final, _ = out.fill(region, active, fixed, model, bounds=bounds,
                        step=step, stats=stats, patchID=taskID)
    out.linear_optimized = False

    # --- linear flux optimization conditional on shapes ---
    if config.linear_optimize:
        logger.info(f"Doing linear optimization of fluxes")
        postop = final.copy()
        niter = final["n_iter"][:]
        postop["n_iter"][:] = 0
        old_bounds = bounds.copy()
        try:
            final, bounds = optimize_linear(patcher, postop.copy(), bounds,
                                            fixed=fixed, shape_cols=shape_cols)
            out.linear_optimized = True
        except (LinAlgError, ValueError) as e:
            logger.error(f"Error during linear optimization, skipping!!")
            fn = os.path.join(config.patch_dir, f"patch{taskID}_design.h5")
            logger.info(f"Writing design matrix to {fn}")
            dump_design_matrices(fn, patcher, active, fixed, shape_cols=shape_cols)

        final["n_iter"][:] = niter
        out.final = final
        out.old_bounds = old_bounds
        out.new_bounds = bounds
        out.postop = postop
        # TODO: Add `final` parameter vector as another iteration of chain

    # --- write ---
    outroot = os.path.join(config.patch_dir, f"patch{taskID}")
    logger.info(f"Writing to {outroot}*")
    write_to_disk(out, outroot, model, config)

    # --- develop the payload ---
    payload = dict(out=out, final=final, covs=None, bounds=bounds)

    return payload


def sampling_task(patcher, task, config=None, logger=None):

   # --- unpack all the task variables ---
    region, active, fixed = task['region'], task['active'], task['fixed']
    bounds, cov = task['bounds'], task['cov']
    bands, shape_cols = task['bands'], task['shape_cols']
    sourceID, taskID = active[0]["source_index"], task["taskID"]
    del task

    # --- log receipt ---
    logger.info(f"Received RA {region.ra}, DEC {region.dec} with tag {taskID}")
    logger.info(f"Tag {taskID} has {len(active)} sources seeded on source index {sourceID}")
    if fixed is None:
        logger.info("No fixed sources")

    # --- get pixel data and metadata ---
    patcher.build_patch(region, None, allbands=bands, tweak_background=config.tweak_background)
    logger.info(f"Prepared patch with {patcher.npix} pixels.")
    if config.sampling_draws == 0:
        return patcher

    # --- Prepare model ---
    model, q = patcher.prepare_model(active=active, fixed=fixed,
                                     bounds=bounds, shapes=shape_cols)
    if config.add_barriers:
        logger.info("Adding edge prior to model.")
        from forcepho.priors import ExpBeta
        model._lnpriorfn = ExpBeta(model.transform.lower, model.transform.upper)

    # --- Sample using covariances--- (child)
    model.ndof = patcher.npix
    weight = max(10, active["n_iter"].min())
    logger.info(f"sampling with covariance weight={weight}")
    if config.sampling_draws == 0:
        return patcher
    try:
        discard_tuning = bool(getattr(config, "discard_tuning", True))
        out, step, stats = run_lmc(model, q.copy(),
                                   n_draws=config.sampling_draws,
                                   warmup=config.warmup,
                                   z_cov=cov, full=config.full_cov,
                                   weight=weight,
                                   discard_tuned_samples=discard_tuning,
                                   max_treedepth=config.max_treedepth,
                                   progressbar=getattr(config, "progressbar", False))
    except ValueError as e:
        logger.error(f"Error at constrained parameter q={q}")
        logger.error(f"Error at unconstrained parameter"
                     f"z={model.transform.inverse_transform(q)}")
        raise e

    # --- clean up ---
    logger.info(f"Sampling complete ({model.ncall} calls), preparing output.")
    final, covs = out.fill(region, active, fixed, model, bounds=bounds,
                           step=step, stats=stats, patchID=taskID)

    # --- write ---
    outroot = os.path.join(config.patch_dir, f"patch{taskID}")
    logger.info(f"Writing to {outroot}*")
    write_to_disk(out, outroot, model, config)

    # --- develop the payload ---
    payload = dict(out=out, final=final, covs=covs, bounds=bounds)

    return payload