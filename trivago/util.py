import pandas as pd
import os
from mars_gym.utils.utils import (
    literal_eval_if_str,
)

def to_array(xs):
    return [literal_eval_if_str(c) if isinstance(literal_eval_if_str(c), int) or isinstance(literal_eval_if_str(c), float) else literal_eval_if_str(c)[0]
            for c in literal_eval_if_str(xs)] if xs is not None else None


def _map_to_pandas(rdds):
    """ Needs to be here due to pickling issues """
    return [pd.DataFrame(list(rdds))]

def toPandas(df, n_partitions=None):
    """
    Returns the contents of `df` as a local `pandas.DataFrame` in a speedy fashion. The DataFrame is
    repartitioned if `n_partitions` is passed.
    :param df:              pyspark.sql.DataFrame
    :param n_partitions:    int or None
    :return:                pandas.DataFrame
    """
    if n_partitions is not None:
        df = df.repartition(n_partitions)
    df_pand = df.rdd.mapPartitions(_map_to_pandas).collect()
    df_pand = pd.concat(df_pand)
    df_pand.columns = df.columns
    return df_pand


def parallelize_dataframe(df, func, n_cores=os.cpu_count()):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df
