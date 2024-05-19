import pandas as pd
import numpy as np


def probs_replace(srs, values):
    '''
    Probabilistic replacement
    Returns a copy of series where target values (e.g. '-', np.nan) are probabilistically replaced with other values. Use on categorical data

    Args:
        srs: pandas series that need values replaced
        values: list (or a single instance) of values (value) to replace

    Returns:
        srs: series with 'values' being probabilisticly replaced
    '''
    if not hasattr(values, '__iter__') or isinstance(values, str):
        values = [values]

    map_series = srs.value_counts().drop(index=values)
    map_series = map_series / map_series.sum()

    srs = srs.copy(deep=True)
    srs[srs.isin(values)] = np.random.choice(map_series.index, size = (srs.isin(values)).sum(), p = map_series.values)
    return srs


def group_occrs(df, column, values, percent=True):
    '''
    Groupby occurences
    Returns frequency of values (e.g. np.nan) in the DataFrame broken down by target column categories

    Args:
        df
        column: column to groupby by
        values:
        percent: whether to present the result as %s (also styles the df)
    
    Returns:
        styled df if percent is True
        df if percent is False
    '''
    if not hasattr(values, '__iter__') or isinstance(values, str):
        values = [values]

    output = pd.concat([df[column], df.drop(column, axis='columns').isin(values)], axis=1).groupby(column).sum()
    if percent:
        return (output / df.shape[0]).style.format('{:,.2%}')\
            .background_gradient(axis=None)
    else:
        return output.style.background_gradient(axis=None)


def na(df, percent = True, verbose = True):
    srs = df.isna().sum()[df.isna().sum() > 0]
    if percent:
       if verbose:
           print('% of NaNs in df:')
       return srs / df.shape[0]
    else:
        if verbose:
           print('# of NaNs in df:')
        return srs


def type_breakdown(df):
    srs = df.dtypes.to_frame().reset_index()
    srs.columns = ['col', 'type']
    return srs.groupby('type').size().sort_values(ascending=False)

