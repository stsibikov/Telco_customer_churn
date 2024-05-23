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
        styled df
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


def uniq(srs, return_ : str = 'set'):
    '''
    unique, no nan

    returns a set by default, just the array of uniques otherwise
    '''
    unique = srs.unique()
    unique = unique[~pd.isnull(unique)]
    if return_ == 'set':
        return set(unique)
    else:
        return unique


def humanize_srs(srs):
    '''
    changes series name and sometimes values to object dtype. For display purpose only.
    '''
    srs_human_name = srs.name.replace('_', ' ')

    from aku_utils.data import uniq
    if uniq(srs) == {0, 1}:
        return pd.Series(np.where(srs == 1, srs_human_name.capitalize(), f'No {srs_human_name}'), name=srs_human_name)
    else:
        return pd.Series(srs, name=srs_human_name)


# def _get_optimal_display_objs(srs):
#     '''
#     Args: srs: values series from a groupby
#     Returns: to_display: # of objects to display
#     '''
#     srs = srs + srs[0] / len(srs)**2 * srs.index**2 - srs[0] / len(srs) * srs.index / 2
#     to_display = srs.idxmin()
#     return to_display


# def trunc_data_for_display(groupby, min_=4, max_=20) -> pd.DataFrame:
#     '''
#     Args:
#         groupby: result of a groupby (reset index!)
#         min_: min objects to plot
#         max_: max objects to plot

#     Returns:
#         truncated (or not) dataframe
#     '''
#     if groupby.shape[0] <= min_:
#         return groupby

#     groupby = groupby.sort_values(by=groupby.columns[-1], ascending=False)
#     to_display = _get_optimal_display_objs(groupby.iloc[:max_, -1].reset_index(drop=True))
#     to_display = min(max(to_display, min_), max_)

#     return groupby[:to_display]


# def _bin_srs(srs):
#     '''
#     optimal number of bins is observed via expert (me) analysis
#     '''
#     nbins = min(10, (srs.nunique()**0.375 + 1.6) // 1)
#     srs = pd.cut(srs, nbins)
#     return srs


# def _group_date(srs, threshold = 60):
#     '''
#     day, week, month, year - if date range is less than `threshold lst[i]`, then the grouper is lst[i]
#     '''
#     return srs


# def _how_to_treat_col(srs) -> dict:
#     '''
#     Returns:
#         dict with key (str) - value (Any, but mostly bool) of:
#             srs - should not modify the length or shuffle series
#             needs_sorting
#             needs_truncation
#             preferred_plot: one of 'bar' (default), 'line'
#             any_plot_text
#     '''
#     #gender binary
#     if set(srs.unique()) == {0, 1} and srs.name == 'gender':
#         return {'srs' : srs.replace({0 : "Female", 1 : "Male"}), 'needs_sorting' : False, 'needs_truncation' : False, 'preferred_plot' : 'bar'}

#     #binary
#     if set(srs.unique()) == {0, 1}:
#         human_srs_name = srs.name.replace('_', ' ')
#         return {'srs' : srs.replace({0 : f"No {human_srs_name}", 1 : f"{human_srs_name.capitalize()}"}), 'needs_sorting' : False, 'needs_truncation' : False, 'preferred_plot' : 'bar'}

#     #float or int
#     if pd.api.types.is_numeric_dtype(srs):
#         if srs.nunique() <= 10:
#             # assumed to be ordinal category
#             return {'srs' : srs, 'needs_sorting' : True, 'needs_truncation' : False, 'preferred_plot' : 'bar'}
#         else:
#             srs = _bin_srs(srs).astype('str')
#             return {'srs' : srs, 'needs_sorting' : True, 'needs_truncation' : False, 'preferred_plot' : 'bar'}

#     if pd.api.types.is_datetime64_any_dtype(srs):
#         srs = _group_date(srs)
#         return {'srs' : srs, 'needs_sorting' : True, 'needs_truncation' : False, 'skip_size_mode' : True, 'preferred_plot' : 'line', 'any_plot_text' : False}

#     if pd.api.types.is_object_dtype(srs):
#         return {'srs' : srs, 'needs_sorting' : True, 'needs_truncation' : True, 'preferred_plot' : 'bar'}

#     #else (eg intervalindex) treat as object after converting to string
#     srs = srs.astype('str')
#     return {'srs' : srs, 'needs_sorting' : True, 'needs_truncation' : True, 'preferred_plot' : 'bar'}


def _infer_col_type(srs : pd.Series) -> str:
    '''
    Returns one of:
        numeric
        object. Series with two unique object values will be counted as object, not binary.
        binary
        low-ordinal
        high-ordinal
        date
    
    Recommended to zip with column list for something like:
        for col, dtype in zip(columns, col_dtypes):
            ...
    '''
    if pd.api.types.is_object_dtype(srs):
        return 'object'

    if uniq(srs) == {0, 1}:
        return 'binary'
    
    if pd.api.types.is_numeric_dtype(srs):
        if uniq(srs) <= 10:
            return 'low-ordinal'
        elif uniq(srs) <= 100:
            return 'high-ordinal'
        else:
            return 'numeric'

    if pd.api.types.is_datetime64_any_dtype(srs):
        return 'date'

    return 'object'


def _explore_mode_size(srs, dtype):
    return



def explore(df : pd.DataFrame,
            target : str | None = None,
            columns : list | None = None,
            aggfunc : str = 'mean',
            modes : list | None = None,
            **kwargs) -> None:
    '''
    Explore the dataset with simple visualizations.
    
    To be used in EDA, after preprocessing. Requires proper typing for, for example, datetime columns,
        and proper values - for example, replacing unnecessary whitespaces in object column that creates more unique values than there really are,
        or replacing string values in ordinal (eg., test score out of 100) column.

    Exploration is done in modes:
        'size': plot bar plots of category sizes. Done for all columns, including the target. Bins numerical columns.
            For object columns, it will sort sizes from biggest to lowest. For object columns with medium+ (>=6) number of values, it will try
            to truncate the data, throwing out categories with low sizes.
        'box-plot': plots box plots of target column values by categories of an 'exploring' column.
            For date columns it will do plots, but the date will be aggregated into a few values.
        'line-plot': plots line plots of mean (or other specified `aggfunc`) of target column by 'exploring' column.
            If 'box-plot' is enabled, this mode will only do plots for date columns.
        'scatter': plots scatter plots of an exploring column and the target.
            The mode will only be on if target column is numeric or high-ordinal (see _infer_col_type())

    Args:
        df: original df
        target: target column to orient bar plots or scatter plots around. Must not be object, or date, 
        columns: columns to explore (all by default)
        aggfunc: aggfunc to be used for target aggregation ('mean')
        modes: manually set modes
        kwargs: kwargs for passing into corresponding plotting methods, eg, plotly.express.bar().
            Must be compatible with all used plotting methods, so not recommending using this.

        It is recommended to only set the target column - the function is aimed at doing everything else itself
    Returns:
        Nothing. Plots figures from plotly
    '''
    #
    # option validation
    #
    if columns is not None:
        for col in columns:
            assert col in df.columns, f'column {col} not found in df'

    if target is not None:
        assert target in df.columns, f'target column {target} not found in df'
        assert pd.api.types.is_numeric_dtype(df[target]) is True, f'target {target} is not binary, ordinal or numeric'
        if columns is not None:
            assert target not in columns, f'target column {target} found in list of columns to explore'

    if modes is not None:
        for mode in modes:
            assert mode in ['size', 'box-plot', 'line-plot', 'scatter'], f"mode {mode} not part of list: ['size', 'box-plot', 'line-plot', 'scatter']"

    #
    # option processing
    #
    if columns is None:
        if target is None:
            columns = df.columns
        else:
            columns = df.drop(target, axis=1).columns

    if target is None and modes is None:
        modes = ['size']
    else:
        modes = ['size', 'box-plot', 'line-plot', 'scatter']

    target_dtype = _infer_col_type(df[target])

    if 'scatter' in modes and target_dtype not in ['numeric', 'high-ordinal']:
        modes.remove('scatter')

    #
    # categorize columns
    #

    col_dtypes = []
    for col in columns:
        col_dtypes.append(_infer_col_type(df[col]))

    #
    # start
    #

    if 'size' in modes:
        for col, dtype in zip(columns, col_dtypes):
            _explore_mode_size(col, dtype)

    return None