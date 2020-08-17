import luigi
import torch
import torch.nn as nn
import numpy as np

from typing import Dict, Any, List, Tuple, Union, Type, Callable
from mars_gym.meta_config import ProjectConfig
from mars_gym.model.abstract import RecommenderModule
from mars_gym.torch.init import lecun_normal_init, he_init
import torch.nn.functional as F


class SimpleLinearModel(RecommenderModule):
    def __init__(self,
                 project_config: ProjectConfig,
                 index_mapping: Dict[str, Dict[Any, int]],
                 n_factors: int,
                 window_hist_size: int, 
                 metadata_size: int,
                 vocab_size: int = 88,
                 dropout_prob: int = 0.0, 
                 dropout_module: Type[Union[nn.Dropout, nn.AlphaDropout]] = nn.AlphaDropout,
                 weight_init: Callable = lecun_normal_init):

        super(SimpleLinearModel, self).__init__(project_config, index_mapping)

        self.window_hist_size = window_hist_size
        self.metadata_size = metadata_size

        self.user_embeddings = nn.Embedding(self._n_users, n_factors)
        self.item_embeddings = nn.Embedding(self._n_items, n_factors)
        self.word_embeddings = nn.Embedding(vocab_size, n_factors)

        # Constants
        continuos_size   = 4
        interaction_size = 6
        context_embs     = 4

        # Dropout
        self.dropout: nn.Module = dropout_module(dropout_prob)

        # Dense
        num_dense = continuos_size + (1 * n_factors) + metadata_size  + \
                    (interaction_size * window_hist_size) + \
                    (context_embs * window_hist_size  * n_factors)
             
        self.dense = nn.Sequential(
            nn.Linear(num_dense, int(num_dense/2)),
            nn.SELU(),
            nn.Linear(int(num_dense/2), int(num_dense/4)),
            nn.SELU(),
            nn.Linear(int(num_dense/4), 1)
        )

        # init
        weight_init(self.user_embeddings.weight)
        weight_init(self.item_embeddings.weight)
        #weight_init(self.action_type_embeddings.weight)
        weight_init(self.word_embeddings.weight)

    def init_weights(self, module: nn.Module):
        if type(module) == nn.Linear:
            self.weight_init(module.weight)
            module.bias.data.fill_(0.1)

    def flatten(self, input):
        return input.view(input.size(0), -1)

    def normalize(self, x):
        x = F.normalize(x, p=2, dim=1)
        return x

    def item_dot_history(self, itemA, itemB):
        dot = torch.matmul(self.normalize(itemA.unsqueeze(1)),
                           self.normalize(itemB.permute(0, 2, 1)))
        return self.flatten(dot)

    def forward(self, user_ids, item_ids, pos_item_idx,
                price, sum_action_item_before, is_first_in_impression,
                list_clickout_item_idx,
                list_interaction_item_image_idx, list_interaction_item_info_idx,
                list_interaction_item_rating_idx, list_interaction_item_deals_idx,
                list_search_for_item_idx,
                list_search_for_poi, 
                list_change_of_sort_order,
                list_search_for_destination, 
                list_filter_selection,
                list_metadata):

        # Geral embs
        user_emb = self.user_embeddings(user_ids)
        item_emb = self.item_embeddings(item_ids)

        # Item embs
        clickout_item_emb = self.item_embeddings(list_clickout_item_idx)
        interaction_item_image_emb = self.item_embeddings(
            list_interaction_item_image_idx)
        interaction_item_info_emb = self.item_embeddings(
            list_interaction_item_info_idx)
        interaction_item_rating_emb = self.item_embeddings(
            list_interaction_item_rating_idx)  # less
        interaction_item_deals_emb = self.item_embeddings(
            list_interaction_item_deals_idx)  # less
        search_for_item_emb = self.item_embeddings(list_search_for_item_idx)

        # NLP embs
        search_for_poi_emb = self.word_embeddings(list_search_for_poi)
        search_for_destination_emb = self.word_embeddings(
            list_search_for_destination)
        change_of_sort_order_emb = self.word_embeddings(
            list_change_of_sort_order)
        filter_selection_emb = self.word_embeddings(list_filter_selection)

        context_session_emb = self.flatten(
            torch.cat((search_for_poi_emb, search_for_destination_emb,
                       change_of_sort_order_emb, filter_selection_emb), dim=2))

        # Dot Item X History
        item_dot_clickout_item_emb = self.item_dot_history(
            item_emb, clickout_item_emb)
        item_dot_interaction_item_image_emb = self.item_dot_history(
            item_emb, interaction_item_image_emb)
        item_dot_interaction_item_info_emb = self.item_dot_history(
            item_emb, interaction_item_info_emb)
        item_dot_interaction_item_info_emb = self.item_dot_history(
            item_emb, interaction_item_info_emb)
        item_dot_interaction_item_rating_emb = self.item_dot_history(
            item_emb, interaction_item_rating_emb)
        item_dot_interaction_item_deals_emb = self.item_dot_history(
            item_emb, interaction_item_deals_emb)
        item_dot_search_for_item_emb = self.item_dot_history(
            item_emb, search_for_item_emb)
        #from IPython import embed
        #embed()
        x = torch.cat((item_emb,
                       item_dot_clickout_item_emb,
                       item_dot_interaction_item_image_emb,
                       item_dot_interaction_item_info_emb,
                       item_dot_interaction_item_rating_emb,
                       item_dot_interaction_item_deals_emb,
                       item_dot_search_for_item_emb,
                       is_first_in_impression.float().unsqueeze(1),
                       pos_item_idx.float().unsqueeze(1),
                       sum_action_item_before.float().unsqueeze(1),
                       price.float().unsqueeze(1),
                       list_metadata.float(),
                       context_session_emb), dim=1)

        x = self.dense(x)
        out = torch.sigmoid(x)
        return out
       
