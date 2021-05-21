import os, sys
import datetime
import dateutil.tz
from rllab import config
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from sandbox.vase.envs.mountain_car_env_x import MountainCarEnvX
from sandbox.vase.envs.cartpole_swingup_env_x import CartpoleSwingupEnvX
from sandbox.vase.envs.double_pendulum_env_x import DoublePendulumEnvX
from sandbox.vase.envs.lunar_lander_x import LunarLanderContinuous
try:
    from sandbox.vase.envs.half_cheetah_env_x import HalfCheetahEnvX
    from sandbox.vase.envs.ant_env_x import AntEnv
except:
    HalfCheetahEnvX = None
    AntEnvX = None
os.environ["THEANO_FLAGS"] = "device=cpu"

from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from rllab.envs.normalized_env import NormalizedEnv

import lasagne.nonlinearities as NL

from sandbox.vase.algos.trpo_expl import TRPO
from rllab.misc.instrument import stub, run_experiment_lite

stub(globals())

def run_trpo_vase(env,nRuns = 20,seed_base=0, sigma_m=0.5, ablation_mode=False):

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

    for seed in range(seed_base,nRuns):

        if env == 'mountaincar':
            mdp = MountainCarEnvX()
            n_itr = 50
            max_path_length = 500
            type = 'classic'
        elif env == 'cartpole':
            mdp = NormalizedEnv(env=CartpoleSwingupEnvX())
            n_itr = 400
            max_path_length = 500
            type = 'classic'
        elif env == 'doublependulum':
            mdp = NormalizedEnv(env=DoublePendulumEnvX())
            n_itr = 400
            max_path_length = 500
            type = 'classic'
        elif env == 'halfcheetah':
            mdp = NormalizedEnv(env=HalfCheetahEnvX())
            n_itr = 600
            max_path_length = 500
            type = 'locomotion'
        elif env == 'ant':
            mdp = NormalizedEnv(env=AntEnv())
            n_itr = 600
            max_path_length = 500
            type = 'locomotion'
        elif env == 'lunarlander':
            mdp = NormalizedEnv(env=LunarLanderContinuous())
            n_itr = 100
            max_path_length = 1000
            type = 'classic'
        else:
            sys.stderr.write("Error! Environment '%s' not recognised\n" % env)
            sys.exit(-1)

        if type == 'classic':
            step_size = 0.01
            replay_pool_size = 100000
            policy_hidden_sizes = (32,)
            unn_n_hidden = [32]
            unn_layers_type=[1, 1]

            baseline = GaussianMLPBaseline(
                env_spec=mdp.spec,
                regressor_args={
                    'hidden_sizes': (32,),
                    'learn_std': False,
                    'hidden_nonlinearity': NL.rectify,
                    'optimizer': ConjugateGradientOptimizer(subsample_factor=1.0)
                }
            )
        else:
            step_size = 0.05
            replay_pool_size = 5000000
            policy_hidden_sizes = (64, 32)
            unn_n_hidden = [64, 64]
            unn_layers_type=[1, 1, 1]

            baseline = LinearFeatureBaseline(
                mdp.spec,
            )

        policy = GaussianMLPPolicy(
            env_spec=mdp.spec,
            hidden_sizes=policy_hidden_sizes,
            hidden_nonlinearity=NL.tanh
        )


        algo = TRPO(
            env=mdp,
            policy=policy,
            baseline=baseline,
            n_itr=n_itr,
            batch_size=5000,
            max_path_length = max_path_length,
            discount = 0.995,
            gae_lambda = 0.95,
            whole_paths=True,
            step_size=step_size,
            eta=1e-4,
            snn_n_samples=10,
            prior_sd=sigma_m,
            subsample_factor=1.0,
            use_replay_pool=True,
            replay_pool_size=replay_pool_size,
            n_updates_per_sample=500,
            unn_n_hidden=unn_n_hidden,
            unn_layers_type=unn_layers_type,
            unn_learning_rate=0.001
        )

        exp_name = "trpo-vase_%s_%04d" % (timestamp, seed + 1)
        if ablation_mode:
            cwd = os.getcwd()
            log_dir = cwd + "/data/local/sigmas/" + env + ("/%.3f/" % sigma_m) + exp_name
        else:
            log_dir = config.LOG_DIR + "/local/" + env +  "/" + exp_name

        run_experiment_lite(
            algo.train(),
            exp_name = exp_name,
            log_dir= log_dir,
            n_parallel=0,
            snapshot_mode="last",
            seed=seed,
            mode="local",
            script="sandbox/vase/experiments/run_experiment_lite.py"
        )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="mountaincar",
                        help='Name of the experiment: mountaincar, cartpole, doublependulum, halfcheetah, ant, or lunarlander.')
    parser.add_argument('--runs', type=int, default=20,
                        help='Number of times to run the experiment')
    parser.add_argument('--seed', type=int, default=0,
                        help='Starting seed for runs')
    parser.add_argument('--sigma', type=float, default=0.5,
                        help='Sigma_m value for VASE')
    parser.add_argument('--ablation', action='store_true',
                        help='Whether to run sigma ablation study')

    args = parser.parse_args(sys.argv[1:])

    run_trpo_vase(args.env, nRuns=args.runs, seed_base=args.seed,  sigma_m=args.sigma, ablation_mode=args.ablation)
