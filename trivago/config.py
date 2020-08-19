
import os
import trivago.data as data
from mars_gym.data.dataset import InteractionsDataset
from mars_gym.meta_config import *

trivago_experiment = ProjectConfig(
    base_dir=data.BASE_DIR,
    prepare_data_frames_task=data.PrepareTrivagoSessionsDataFrames,
    dataset_class=InteractionsDataset,
    user_column=Column("user_idx", IOType.INDEXABLE),
    item_column=Column("item_idx", IOType.INDEXABLE),
    available_arms_column_name="impressions",
    other_input_columns=[
        Column("pos_item_idx", IOType.NUMBER),
        Column("diff_price", IOType.NUMBER),  
        Column("sum_action_item_before", IOType.NUMBER),
        Column("is_first_in_impression", IOType.NUMBER),

        Column("list_reference_clickout_item_idx", IOType.INDEXABLE_ARRAY, same_index_as="item_idx"),
        Column("list_reference_interaction_item_image_idx", IOType.INDEXABLE_ARRAY, same_index_as="item_idx"),
        Column("list_reference_interaction_item_info_idx", IOType.INDEXABLE_ARRAY, same_index_as="item_idx"),
        Column("list_reference_interaction_item_rating_idx", IOType.INDEXABLE_ARRAY, same_index_as="item_idx"),
        Column("list_reference_interaction_item_deals_idx", IOType.INDEXABLE_ARRAY, same_index_as="item_idx"),
        Column("list_reference_search_for_item_idx", IOType.INDEXABLE_ARRAY, same_index_as="item_idx"),

        Column("list_reference_search_for_poi", IOType.INT_ARRAY),
        Column("list_reference_change_of_sort_order", IOType.INT_ARRAY),
        Column("list_reference_search_for_destination", IOType.INT_ARRAY),
        Column("list_reference_filter_selection", IOType.INT_ARRAY),
    ],
    metadata_columns=[
        Column("list_metadata", IOType.INT_ARRAY),
    ],
    output_column=Column("clicked", IOType.NUMBER),
    hist_view_column_name="hist_views",
    hist_output_column_name="hist_clicked",
    recommender_type=RecommenderType.USER_BASED_COLLABORATIVE_FILTERING,
)

fixed_trivago_experiment = ProjectConfig(
    base_dir=data.BASE_DIR,
    prepare_data_frames_task=data.PrepareTrivagoSessionsDataFrames,
    dataset_class=InteractionsDataset,
    user_column=Column("user_idx", IOType.INDEXABLE),
    item_column=Column("item_idx", IOType.INDEXABLE),
    available_arms_column_name="impressions",
    other_input_columns=[
        Column("first_item_idx", IOType.INDEXABLE, same_index_as="item_idx"),
        Column("popularity_item_idx", IOType.INDEXABLE, same_index_as="item_idx"),
        Column("action_type_item_idx", IOType.INDEXABLE, same_index_as="item_idx")
    ],
    metadata_columns=[
    ],
    output_column=Column("clicked", IOType.NUMBER),
    hist_view_column_name="hist_views",
    hist_output_column_name="hist_clicked",
    recommender_type=RecommenderType.USER_BASED_COLLABORATIVE_FILTERING,
)
