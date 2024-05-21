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


def group_occrs(df : pd.DataFrame,
                column : str  | None = None,
                values = np.nan,
                percent : bool = True):
    '''
    Groupby occurences
    How often do you see `values` in the DataFrame

    Args:
        df: DataFrame. Recommended to use a column-wise slice when working with large dataset
        column: column to groupby by (None, in which case it will group by the first column)
        values: a single value, or an iterable of values (np.nan)
        percent: whether to present the result as %s (True). `True` also styles the df, turning it into custom object
    
    Returns:
        styled df if percent is True
        df if percent is False
    '''
    if not hasattr(values, '__iter__') or isinstance(values, str):
        values = [values]
    
    if column is None:
        column = df.columns[0]

    output = pd.concat([df[column], df.drop(column, axis=1).isin(values)], axis=1).groupby(column).sum()
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


def inspect_mean(df : pd.DataFrame,
                 group : str = 'phone_service',
                 target : str = 'churn_label',
                 include_overall : bool = True,
                 fast_bar : bool = False):
    '''
    Make a table with mean of target 

    Args:
        df: pd.DataFrame, original
        group: name of the column with categories to divide df by ('phone_service')
        target: name of the column with target values ('churn_label')
        include_overall: whether to include the overall of target into table
        fast_bar: whether to plot the bar with ak.fmt_bar() instead of returning the table (False).
        For bar plot modification simply return table with `fast_bar = False` and wrap it around `ak.fmt_bar()`

    Returns:
        pd.DataFrame: if fast_bar is False
        plotly.graph_objects.Figure: if fast_bar is True
    '''

    human_group_name = group.replace('_', ' ')
    out = df.groupby(group, as_index=False)[target].mean()

    if set(out[group].unique().tolist()) == {0, 1}:
        out[group] = out[group].replace({0 : f'No {human_group_name}', 1 : human_group_name.capitalize()})
    
    if include_overall:
        out = pd.concat([out, pd.DataFrame({group : ['Overall'], target : [df[target].mean()]})])

    if fast_bar:
        from aku_utils.plot import fmt_bar
        return fmt_bar(out)
    else:
        return out