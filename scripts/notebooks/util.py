import sys
import os
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import os
import json
import random
import matplotlib.pyplot as plt
import seaborn as sns

from pandas.io.json import json_normalize
from mars_gym.tools.eval_viz.app import fetch_iteraction_results_path, load_all_iteraction_metrics
from mars_gym.tools.eval_viz.plot import plot_line_iteraction, plot_exploration_arm, get_colors, confidence

models_and_legend = {
    "____random_":           ["bandit_policy", "observation"],
    "____fixed_":            ["bandit_policy", "observation"],
    "____lin_ucb_":          ["bandit_policy", "bandit_policy_params.alpha", "full_refit", "val_split_type"],
    "____model_":            ["bandit_policy", "full_refit", "val_split_type"],
    "____custom_lin_ucb_":   ["bandit_policy", "bandit_policy_params.alpha", "full_refit", "val_split_type"],
    "____epsilon_greedy_":   ["bandit_policy", "bandit_policy_params.epsilon", "full_refit", "val_split_type"],
    "____softmax_":          ["bandit_policy", "bandit_policy_params.logit_multiplier", "full_refit", "val_split_type"],
    "____lin_ts_":           ["bandit_policy", "bandit_policy_params.v_sq", "full_refit", "val_split_type"],
    "____percentile_adapt_":       ["bandit_policy", "bandit_policy_params.exploration_threshold", "full_refit", "val_split_type"],
    "____adaptive_":         ["bandit_policy", "bandit_policy_params.exploration_threshold", "bandit_policy_params.decay_rate", "full_refit", "val_split_type"],
    "____explore_then_exp_": ["bandit_policy", "bandit_policy_params.explore_rounds", "bandit_policy_params.decay_rate", "full_refit", "val_split_type"],
}

def list_paths_per_model(input_path):
    models = []

    #for model, legend in models_and_legend.items():
        #print(model)
        #print(legend)
    for root, dirs, files in os.walk(input_path):
        if '/results' in root and 'Interaction' in root:
            for d in dirs:
                #print(os.path.join(root, d))
              if 'mars_gym_model' in d:
                models.append(os.path.join(root, d))
    return models


def load_iteractions_params(iteractions):
  if len(iteractions) == 0:
    return pd.DataFrame()

  dfs = []

  for model in iteractions:

    file_path = os.path.join(model, 'params.json')
    data = []

    #try:
    with open(file_path) as json_file:
        d = json.load(json_file)
        data.append(d)

        df = pd.DataFrame.from_dict(json_normalize(data), orient='columns')

    #except:
    #  df = pd.DataFrame()

    df['iteraction'] = model
    dfs.append(df)

  return pd.concat(dfs)


def load_data_iteractions_metrics(path, sample_size=10000):
    random.seed(42)
    file = os.path.join(path, 'sim-datalog.csv')

    # Count the lines
    num_lines = sum(1 for l in open(file)) - 1

    # Sample size - in this case ~10%
    size = np.min([sample_size, num_lines])  # int(num_lines / 10)

    # The row indices to skip - make sure 0 is not included to keep the header!
    skip_idx = sorted(random.sample(range(1, num_lines), num_lines - size))
    idx = list(set(list(range(num_lines))) - set(skip_idx))

    df = pd.read_csv(file, skiprows=skip_idx)

    df = pd.read_csv(file)  # .reset_index()
    idx = list(range(len(df)))

    df['idx'] = sorted(idx)
    df = df.sort_values("idx")
    return df


def get_metrics_reward(bandits):
    data = []
    for i, p in enumerate(bandits):
        df_metrics = load_data_iteractions_metrics(p)
        r_mean = df_metrics.reward.mean()
        r_reward = df_metrics.reward.sum()
        data.append((i, r_mean, r_reward))
    df_metrics = pd.DataFrame(
        data, columns=['idx', 'r_mean', 'r_reward']).set_index('idx')
    return df_metrics


def group_metrics(df):
    df_g_metrics = df.groupby('bandit').agg({'r_mean': ['mean', 'std'], 'r_reward': [
        'mean', 'std', 'count'], 'model': 'first'})
    df_g_metrics.columns = df_g_metrics.columns.map(
        lambda x: '|'.join([str(i) for i in x]))
    return df_g_metrics


def load_dataset(df_models, bandits, sample_size):
    data = []
    for bandit in bandits:
        input_iteraction = [p.split("/")[-1]
                            for p in df_models.loc[bandit].model_list]
        df_metrics = load_all_iteraction_metrics(input_iteraction, sample_size)
        #params            = load_iteractions_params2(input_iteraction)

        df_metrics = load_all_iteraction_metrics(input_iteraction, sample_size)
        df_metrics['_idx'] = 1
        df_metrics = df_metrics.groupby(['iteraction', 'idx']).sum().fillna(
            0).groupby(level=0).cumsum().reset_index()
        df_metrics['mean_reward'] = (
            df_metrics.reward/df_metrics._idx).fillna(0)
        df_metrics['bandit'] = bandit

        data.append(df_metrics)
    return pd.concat(data, ignore_index=True)


def plot_cum_reward(df, hue='bandit', legend=False, ylim=1):
    plt.figure()
    sns.set(style="darkgrid")

    # Plot the responses for different events and regions
    ax = sns.lineplot(x="idx", y="mean_reward",
                      hue=hue, legend=legend, data=df)
    ax.set_ylim(0, ylim)
    #ax.set(xlabel='Interactions', ylabel='Cumulative Mean Reward',fontsize=20)
    ax.tick_params(labelsize=11)
    ax.set_xlabel('Interactions', fontsize=15)
    ax.set_ylabel('Cumulative Mean Reward', fontsize=15)
    # Put the legend out of the figure
    if legend:
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2,
                   borderaxespad=0., fontsize=12)
    plt.savefig("image_iteraction.pdf", dpi=300, bbox_inches="tight")
    plt.show()
