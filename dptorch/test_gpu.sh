#!/bin/bash
python run_dpgp.py MODEL_NAME=StochasticOptimalGrowthModel +params.active_dim=2 +params.n_agents=100 +params.No_samples=1000 GP_MODEL_NAME=ASGPModel params.learning_rate=0.01 params.iter_per_cycle=1000 EXPECT_OP=MCAdditiveStandardNormal +params.num_cycles=200 hydra.verbose=True +params.n_mc_sample=10 +params.force_cpu=True
