
mars-gym run data trivago.data PrepareTrivagoSessionsDataFrames

mars-gym run supervised \
--project trivago.config.trivago_experiment \
--recommender-module-class trivago.model.SimpleLinearModel \
--recommender-extra-params '{"n_factors": 10, "metadata_size": 142, "window_hist_size": 5}' \
--data-frames-preparation-extra-params '{"filter_city": "Lausanne, Switzerland"}' 
--epochs 1

mars-gym evaluate supervised --model-task-id SupervisedModelTraining____mars_gym_model_b____ff2ab61e65


# Random

mars-gym run interaction \
--project trivago.config.trivago_experiment \
--recommender-module-class trivago.model.SimpleLinearModel \
--recommender-extra-params '{"n_factors": 50, "metadata_size": 142, "window_hist_size": 10}' \
--bandit-policy-class mars_gym.model.bandit.RandomPolicy \
--bandit-policy-params '{}' \
--data-frames-preparation-extra-params '{"filter_city": "Lausanne, Switzerland", "window_hist":10}' \
--learning-rate 0.001 \
--optimizer adam \
--epochs 250 \
--obs-batch-size 500 \
--batch-size 200 \
--num-episodes 100 \
--val-split-type random \
--full-refit 

# Model

mars-gym run interaction \
--project trivago.config.trivago_experiment \
--recommender-module-class trivago.model.SimpleLinearModel \
--recommender-extra-params '{"n_factors": 50, "metadata_size": 142, "window_hist_size": 10}' \
--bandit-policy-class mars_gym.model.bandit.ModelPolicy \
--bandit-policy-params '{}' \
--data-frames-preparation-extra-params '{"filter_city": "Lausanne, Switzerland", "window_hist":10}' \
--learning-rate 0.001 \
--optimizer adam \
--epochs 250 \
--obs-batch-size 500 \
--batch-size 200 \
--num-episodes 100 \
--val-split-type random \
--full-refit 


# Greedy - RIO

mars-gym run interaction \
--project trivago.config.trivago_experiment \
--recommender-module-class trivago.model.SimpleLinearModel \
--recommender-extra-params '{"n_factors": 50, "metadata_size": 153, "window_hist_size": 10, "vocab_size": 120}' \
--bandit-policy-class mars_gym.model.bandit.EpsilonGreedy \
--bandit-policy-params '{"epsilon": 0.1}' \
--data-frames-preparation-extra-params '{"filter_city": "Rio de Janeiro, Brazil", "window_hist":10}' \
--learning-rate 0.001 \
--optimizer adam \
--loss-function-params '{"clip": 1}' \
--epochs 250 \
--obs-batch-size 500 \
--batch-size 200 \
--num-episodes 1 \
--val-split-type random \
--full-refit 



# Greedy - RIO

mars-gym run supervised \
--project trivago.config.trivago_experiment \
--recommender-module-class trivago.model.SimpleLinearModel \
--recommender-extra-params '{"n_factors": 50, "metadata_size": 153, "window_hist_size": 10, "vocab_size": 120}' \
--data-frames-preparation-extra-params '{"filter_city": "Rio de Janeiro, Brazil", "window_hist":10}' \
--learning-rate 0.001 \
--optimizer adam \
--batch-size 200 \
--epochs 10 

mars-gym evaluate supervised --model-task-id SupervisedModelTraining____mars_gym_model_b___epsilon___0_1__9c3de94e53
