# Geral Default PARAMS

num_episodes=1 
epochs=100 
obs_batch_size=3000 
batch_size=500 
learning_rate=0.001 

## Softmax Recsys CIties

### Train

mars-gym run interaction \
--project trivago.config.trivago_experiment \
--recommender-module-class trivago.model.SimpleLinearModel \
--recommender-extra-params '{"n_factors": 50, "metadata_size": 153, "window_hist_size": 10, "vocab_size": 340}' \
--bandit-policy-class mars_gym.model.bandit.SoftmaxExplorer \
--bandit-policy-params '{"logit_multiplier": 5.0}' \
--data-frames-preparation-extra-params '{"filter_city": "recsys", "window_hist":10}' \
--learning-rate $learning_rate \
--optimizer adam \
--batch-size 200 \
--epochs $epochs \
--num-episodes $num_episodes \
--val-split-type random \
--obs-batch-size $obs_batch_size \
--full-refit --seed $i  


### Evaluation

mars-gym evaluate interaction \
--model-task-id InteractionTraining____mars_gym_model_b___epsilon___0_1__472bcd526f
--fairness-columns '["device_idx", "city_idx", "accessible parking", "accessible hotel", "hotel", "house / apartment", "childcare", "family friendly"]'