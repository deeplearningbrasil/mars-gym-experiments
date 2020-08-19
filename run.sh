#!/bin/bash

## Fixed Popularity

mars-gym run interaction \
--project trivago.config.fixed_trivago_experiment \
--recommender-module-class trivago.model.SimpleLinearModel \
--recommender-extra-params '{"n_factors": 50, "metadata_size": 147, "window_hist_size": 10, "vocab_size": 120}' \
--bandit-policy-class mars_gym.model.bandit.FixedPolicy \
--bandit-policy-params '{"arg": 2}' \
--data-frames-preparation-extra-params '{"filter_city": "Chicago, USA", "window_hist":10}' \
--learning-rate 0.001 \
--optimizer adam \
--batch-size 200 \
--epochs 250 \
--num-episodes 7 \
--val-split-type random \
--obs-batch-size 1000 \
--full-refit 

# Random

mars-gym run interaction \
--project trivago.config.trivago_experiment \
--recommender-module-class trivago.model.SimpleLinearModel \
--recommender-extra-params '{"n_factors": 50, "metadata_size": 147, "window_hist_size": 10, "vocab_size": 120}' \
--bandit-policy-class mars_gym.model.bandit.RandomPolicy \
--data-frames-preparation-extra-params '{"filter_city": "Chicago, USA", "window_hist":10}' \
--learning-rate 0.001 \
--optimizer adam \
--batch-size 200 \
--epochs 250 \
--num-episodes 7 \
--val-split-type random \
--obs-batch-size 1000 \
--full-refit 

# Egreedy

mars-gym run interaction \
--project trivago.config.trivago_experiment \
--recommender-module-class trivago.model.SimpleLinearModel \
--recommender-extra-params '{"n_factors": 50, "metadata_size": 147, "window_hist_size": 10, "vocab_size": 120}' \
--bandit-policy-class mars_gym.model.bandit.EpsilonGreedy \
--bandit-policy-params '{"epsilon": 0.1}' \
--data-frames-preparation-extra-params '{"filter_city": "Chicago, USA", "window_hist":10}' \
--learning-rate 0.001 \
--optimizer adam \
--batch-size 200 \
--epochs 250 \
--num-episodes 7 \
--val-split-type random \
--obs-batch-size 1000 \
--full-refit 

# LinUcb

mars-gym run interaction \
--project trivago.config.trivago_experiment \
--recommender-module-class trivago.model.SimpleLinearModel \
--recommender-extra-params '{"n_factors": 50, "metadata_size": 147, "window_hist_size": 10, "vocab_size": 120}' \
--bandit-policy-class mars_gym.model.bandit.LinUCB \
--bandit-policy-params '{"alpha": 1e-5}' \
--data-frames-preparation-extra-params '{"filter_city": "Chicago, USA", "window_hist":10}' \
--learning-rate 0.001 \
--optimizer adam \
--batch-size 200 \
--epochs 250 \
--num-episodes 7 \
--val-split-type random \
--obs-batch-size 1000 \
--full-refit 

# custom_lin_ucb

mars-gym run interaction \
--project trivago.config.trivago_experiment \
--recommender-module-class trivago.model.SimpleLinearModel \
--recommender-extra-params '{"n_factors": 50, "metadata_size": 147, "window_hist_size": 10, "vocab_size": 120}' \
--bandit-policy-class mars_gym.model.bandit.CustomRewardModelLinUCB \
--bandit-policy-params '{"alpha": 1e-5}' \
--data-frames-preparation-extra-params '{"filter_city": "Chicago, USA", "window_hist":10}' \
--learning-rate 0.001 \
--optimizer adam \
--batch-size 200 \
--epochs 250 \
--num-episodes 7 \
--val-split-type random \
--obs-batch-size 1000 \
--full-refit 

# Lin_ts

mars-gym run interaction \
--project trivago.config.trivago_experiment \
--recommender-module-class trivago.model.SimpleLinearModel \
--recommender-extra-params '{"n_factors": 50, "metadata_size": 147, "window_hist_size": 10, "vocab_size": 120}' \
--bandit-policy-class mars_gym.model.bandit.LinThompsonSampling \
--bandit-policy-params '{"v_sq": 0.1}' \
--data-frames-preparation-extra-params '{"filter_city": "Chicago, USA", "window_hist":10}' \
--learning-rate 0.001 \
--optimizer adam \
--batch-size 200 \
--epochs 250 \
--num-episodes 7 \
--val-split-type random \
--obs-batch-size 1000 \
--full-refit 


# Adaptive

mars-gym run interaction \
--project trivago.config.trivago_experiment \
--recommender-module-class trivago.model.SimpleLinearModel \
--recommender-extra-params '{"n_factors": 50, "metadata_size": 147, "window_hist_size": 10, "vocab_size": 120}' \
--bandit-policy-class mars_gym.model.bandit.AdaptiveGreedy \
--bandit-policy-params '{"exploration_threshold": 0.7, "decay_rate": 0.0000972907743983833}' \
--data-frames-preparation-extra-params '{"filter_city": "Chicago, USA", "window_hist":10}' \
--learning-rate 0.001 \
--optimizer adam \
--batch-size 200 \
--epochs 250 \
--num-episodes 7 \
--val-split-type random \
--obs-batch-size 1000 \
--full-refit 

# Percentil Adaptive

mars-gym run interaction \
--project trivago.config.trivago_experiment \
--recommender-module-class trivago.model.SimpleLinearModel \
--recommender-extra-params '{"n_factors": 50, "metadata_size": 147, "window_hist_size": 10, "vocab_size": 120}' \
--bandit-policy-class mars_gym.model.bandit.PercentileAdaptiveGreedy \
--bandit-policy-params '{"exploration_threshold": 0.7}' \
--data-frames-preparation-extra-params '{"filter_city": "Chicago, USA", "window_hist":10}' \
--learning-rate 0.001 \
--optimizer adam \
--batch-size 200 \
--epochs 250 \
--num-episodes 7 \
--val-split-type random \
--obs-batch-size 1000 \
--full-refit 

## softmax_explorer

mars-gym run interaction \
--project trivago.config.trivago_experiment \
--recommender-module-class trivago.model.SimpleLinearModel \
--recommender-extra-params '{"n_factors": 50, "metadata_size": 147, "window_hist_size": 10, "vocab_size": 120}' \
--bandit-policy-class mars_gym.model.bandit.SoftmaxExplorer \
--bandit-policy-params '{"logit_multiplier": 5.0}' \
--data-frames-preparation-extra-params '{"filter_city": "Chicago, USA", "window_hist":10}' \
--learning-rate 0.001 \
--optimizer adam \
--batch-size 200 \
--epochs 250 \
--num-episodes 7 \
--val-split-type random \
--obs-batch-size 1000 \
--full-refit 


## Explore the Exploit


mars-gym run interaction \
--project trivago.config.trivago_experiment \
--recommender-module-class trivago.model.SimpleLinearModel \
--recommender-extra-params '{"n_factors": 50, "metadata_size": 147, "window_hist_size": 10, "vocab_size": 120}' \
--bandit-policy-class mars_gym.model.bandit.ExploreThenExploit \
--bandit-policy-params '{"explore_rounds": 1000, "decay_rate": 0.0001872157}' \
--data-frames-preparation-extra-params '{"filter_city": "Chicago, USA", "window_hist":10}' \
--learning-rate 0.001 \
--optimizer adam \
--batch-size 200 \
--epochs 250 \
--num-episodes 7 \
--val-split-type random \
--obs-batch-size 1000 \
--full-refit 
