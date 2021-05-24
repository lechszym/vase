This code is based on [code](https://github.com/openai/vime) for Variational Information Maximizing Exploration (VIME) as presented in Curiosity-driven Exploration in Deep Reinforcement Learning via Bayesian Neural Networks by *R. Houthooft, X. Chen, Y. Duan, J. Schulman, F. De Turck, P. Abbeel* (http://arxiv.org/abs/1605.09674). 

# How to run VASE

To reproduce the results, you should first have [rllab](https://github.com/rllab/rllab) and Mujoco v1.31 configured. Then, run the following commands in the root folder of `rllab`:

```bash
git submodule add -f https://github.com/lechszym/vase.git sandbox/vase
touch sandbox/__init__.py
```

Then you can do the following:
- Execute TRPO+VASE on the Mountain car environment via `python sandbox/vase/experiments/run_rpo_vase.py --env mountaincar`.
