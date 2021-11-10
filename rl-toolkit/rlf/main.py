import os
from typing import Dict

import attr
import numpy as np

import rlf.rl.utils as rutils


@attr.s(auto_attribs=True, slots=True)
class RunResult:
    prefix: str
    eval_result: Dict = {}


def run_policy(run_settings, runner=None):
    if runner is None:
        runner = run_settings.create_runner()
    end_update = runner.updater.get_num_updates()
    args = runner.args

    if args.ray:
        import ray
        from ray import tune

        # Release resources as they will be recreated by Ray
        runner.close()

        use_config = eval(args.ray_config)
        use_config["cwd"] = os.getcwd()
        use_config = run_settings.get_add_ray_config(use_config)

        rutils.pstart_sep()
        print("Running ray for %i updates per run" % end_update)
        rutils.pend_sep()

        ray.init(local_mode=args.ray_debug)
        tune.run(
            type(run_settings),
            resources_per_trial={"cpu": args.ray_cpus, "gpu": args.ray_gpus},
            stop={"training_iteration": end_update},
            num_samples=args.ray_nsamples,
            global_checkpoint_period=np.inf,
            config=use_config,
            **run_settings.get_add_ray_kwargs()
        )
    else:
        args = runner.args

        if runner.should_load_from_checkpoint():
            runner.load_from_checkpoint()

        if args.eval_only:
            eval_result = runner.full_eval(run_settings.create_traj_saver)
            return RunResult(prefix=args.prefix, eval_result=eval_result)

        start_update = 0
        if args.resume:
            start_update = runner.resume()

        runner.setup()
        print("RL Training (%d/%d)" % (start_update, end_update))

        if runner.should_start_with_eval:
            runner.eval(-1)

        # Initialize outside the loop just in case there are no updates.
        j = 0
        for j in range(start_update, end_update):
            updater_log_vals = runner.training_iter(j)
            if args.log_interval > 0 and (j + 1) % args.log_interval == 0:
                log_dict = runner.log_vals(updater_log_vals, j)
            if args.save_interval > 0 and (j + 1) % args.save_interval == 0:
                runner.save(j)
            if args.eval_interval > 0 and (j + 1) % args.eval_interval == 0:
                runner.eval(j)

        if args.eval_interval > 0:
            runner.eval(j + 1)
        if args.save_interval > 0:
            runner.save(j + 1)

        runner.close()
        # WB prefix of the run so we can later fetch the data.
        return RunResult(args.prefix)
