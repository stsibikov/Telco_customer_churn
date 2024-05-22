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
    Groupby occurences (count) of values by column

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


def _get_optimal_display_objs(srs):
    '''
    Args: srs: values series from a groupby
    Returns: to_display: # of objects to display
    '''
    srs = srs + srs[0] / len(srs)**2 * srs.index**2 - srs[0] / len(srs) * srs.index / 2
    to_display = srs.idxmin()
    return to_display


def trunc_data_for_display(groupby, min_=4, max_=20) -> pd.DataFrame:
    '''
    Args:
        groupby: result of a groupby (reset index!)
        min_: min objects to plot
        max_: max objects to plot

    Returns:
        truncated (or not) dataframe
    '''
    if groupby.shape[0] <= min_:
        return groupby

    groupby = groupby.sort_values(by=groupby.columns[-1], ascending=False)
    to_display = _get_optimal_display_objs(groupby.iloc[:max_, -1].reset_index(drop=True))
    to_display = min(max(to_display, min_), max_)

    return groupby[:to_display]


def _bin_srs(srs):
    '''
    optimal number of bins is observed via expert (me) analysis
    '''
    nbins = min(10, (srs.nunique()**0.375 + 1.6) // 1)
    srs = pd.cut(srs, nbins)
    return srs


def _group_date(srs, threshold = 60):
    '''
    day, week, month, year - if date range is less than `threshold lst[i]`, then the grouper is lst[i]
    '''
    return srs


def _how_to_treat_col(srs) -> dict:
    '''
    Returns:
        dict with key (str) - value (Any, but mostly bool) of:
            srs - should not modify the length or shuffle series
            needs_sorting
            needs_truncation
            preferred_plot: one of 'bar' (default), 'line'
            any_plot_text
    '''
    #gender binary
    if set(srs.unique()) == {0, 1} and srs.name == 'gender':
        return {'srs' : srs.replace({0 : "Female", 1 : "Male"}), 'needs_sorting' : False, 'needs_truncation' : False, 'preferred_plot' : 'bar'}

    #binary
    if set(srs.unique()) == {0, 1}:
        human_srs_name = srs.name.replace('_', ' ')
        return {'srs' : srs.replace({0 : f"No {human_srs_name}", 1 : f"{human_srs_name.capitalize()}"}), 'needs_sorting' : False, 'needs_truncation' : False, 'preferred_plot' : 'bar'}

    #float or int
    if pd.api.types.is_numeric_dtype(srs):
        if srs.nunique() <= 10:
            # assumed to be ordinal category
            return {'srs' : srs, 'needs_sorting' : True, 'needs_truncation' : False, 'preferred_plot' : 'bar'}
        else:
            srs = _bin_srs(srs).astype('str')
            return {'srs' : srs, 'needs_sorting' : True, 'needs_truncation' : False, 'preferred_plot' : 'bar'}

    if pd.api.types.is_datetime64_any_dtype(srs):
        srs = _group_date(srs)
        return {'srs' : srs, 'needs_sorting' : True, 'needs_truncation' : False, 'skip_size_mode' : True, 'preferred_plot' : 'line', 'any_plot_text' : False}

    if pd.api.types.is_object_dtype(srs):
        return {'srs' : srs, 'needs_sorting' : True, 'needs_truncation' : True, 'preferred_plot' : 'bar'}

    #else (eg intervalindex) treat as object after converting to string
    srs = srs.astype('str')
    return {'srs' : srs, 'needs_sorting' : True, 'needs_truncation' : True, 'preferred_plot' : 'bar'}


def explore(df : pd.DataFrame,
            columns : list,
            modes : list | None = None,
            target : str | None = None,
            aggfunc : str = 'mean',
            sort_any : bool = True,
            trunc_any : bool = True,
            trust_effect_any : bool = True,
            **kwargs) -> None:
    '''
    Explore the dataset with simple visualizations to produce human-oriented graphs.
    Function modifies the data for better readability.

    Args:
        df: original df
        columns: columns to explore (all by default, all but 'target' if it is there)
        modes: how to explore the data, a list of strings (['size']):
            'size': plot size (bar plot) of each column
            'targeted': plot mean (or other aggfunc) of column 'target' grouped by each column.
                Ignored if target is not provided, added by itself if target is provided.
                Yeah, as you can see it doesnt matter if its provided, but this is important for overall structure.
            'scatter': scatter plot a column and target column. Will only work with numeric columns deemed as truly numeric (>10 unique values)
        target: target column to orient bar plots or scatter plots around
        aggfunc: function to aggregate target column by. Ignored if target is not provided
        sort_any: whether to apply sorting to any columns (from `columns`) - the algorithm will do that for some columns (True)
        trunc_any: whether to apply column (from `columns`) truncation to any columns - the algorithm will do that for some columns (True)
        trust_effect_any: whether to apply group trust effect to any columns. Only valid if `targeted` mode is on.

    Returns:
        Nothing. Plots plotly plots, all separate figures
    '''
    # option validating
    if columns is not None:
        for col in columns:
            assert col in df.columns, f'column {col} not found in df'
    
    if target is not None:
        assert target in df.columns, f'target column {target} not found in df'

    # option processing
    if columns is None:
        if target is None:
            columns = df.columns
        else:
            columns = df.drop(target, axis=1).columns

    if modes is None:
        modes = ['size']

    if target is not None and 'targeted' not in modes:
        modes.append('target')

    # modes sorting: size, targeted, scatter
    modes.sort(key=lambda x: {'size' : 1, 'targeted' : 2, 'scatter' : 3}.get(x))

    # start
    for mode in modes:
        for col in columns:
            srs = df[col]
            treat_dict = _how_to_treat_col(srs)
            srs = treat_dict['srs']


    return None