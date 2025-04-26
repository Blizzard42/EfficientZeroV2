from setuptools import setup, find_packages

INSTALL_REQUIRES = [
    # RL
    # do we need gym or create our on format?
    # "gym==0.24.1",
    # "torch",
    # "omegaconf",
    # "hydra-core>=1.1",
    # "rl-games==1.5.2",
    # "usd-core"
    "numpy",
    "warp-lang>=1.0.0",
]


setup(
    name="warp-sim-envs",
    version="0.0.1",
    py_modules=["warp_sim_envs"],
    packages=find_packages(),
    author="NVIDIA Corporation",
    install_requires=INSTALL_REQUIRES,
)
