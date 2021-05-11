import os
from sandbox.vase.envs.mountain_car_env_x import MountainCarEnvX
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
os.environ["THEANO_FLAGS"] = "device=cpu"

from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

from sandbox.vase.algos.trpo_expl import TRPO
from rllab.misc.instrument import stub, run_experiment_lite
import itertools

stub(globals())

# Param ranges
seeds = range(20)
mdp = MountainCarEnvX()

for seed in seeds:


    policy = GaussianMLPPolicy(
        env_spec=mdp.spec,
        hidden_sizes=(64, 32),
    )

    baseline = LinearFeatureBaseline(
        mdp.spec,
    )


    algo = TRPO(
        env=mdp,
        policy=policy,
        baseline=baseline,
        n_itr=50,
        batch_size=5000,
        max_path_length = 500,
        discount = 0.995,
        gae_lambda = 0.95,
        whole_paths=True,
        step_size=0.01,
        eta=1e-4,
        snn_n_samples=10,
        prior_sd=0.25,
        subsample_factor=1.0,
        use_replay_pool=True,
        replay_pool_size=100000,
        n_updates_per_sample=500,
        unn_n_hidden=[32],
        unn_layers_type=[1, 1],
        unn_learning_rate=0.001
    )

    run_experiment_lite(
        algo.train(),
        exp_prefix="trpo-vase",
        n_parallel=4,
        snapshot_mode="last",
        seed=seed,
        mode="local",
        script="sandbox/vase/experiments/run_experiment_lite.py"
    )