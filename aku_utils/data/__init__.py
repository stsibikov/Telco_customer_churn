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


def _get_optimal_display_objs(srs):
    srs = srs + srs[0] / len(srs)**2 * srs.index**2 - srs[0] / len(srs) * srs.index / 2
    to_display = srs.idxmin()
    return to_display


def trunc_data_for_display(groupby, min_=4, max_=20) -> pd.DataFrame:
    '''
    Args:
        groupby: result of a groupby (no index!)
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


def explore(df : pd.DataFrame,
            columns : list | None = None,
            target : str | None = None,
            aggfunc : str = 'mean',
            **kwargs) -> None:
    '''
    Builds groupby plots of all columns from df or the ones specified. Plots are separate.

    This function requires specific, although the most rational, column naming convention:
        * columns are lowered and have spaces or underscores between their names
        * normal naming, so, eg, gender refers to gender - explore() can automatically
            convert values based on naming
    
    Args:
        columns: list of columns to build plots on. if None, builds plots of all columns
        target: if specified, groupby result will aggregate this column.
            If this is specified and `aggfunc` is not, mean will be used
        aggfunc: string of an aggfunc to use with `target`
        kwargs: kwargs for fmt_bar()
    
    Returns:
        None. Draws N plots

    If target is specified with, eg, mean, then the plot would draw the top categories with the top mean
    '''
    from aku_utils.plot import fmt_bar

    def modify_srs_tell_if_trunc(srs) -> tuple:
        '''
        returns series (maybe modified), bool whether to run trunc_data_for_display
        '''
        if pd.api.types.is_float_dtype(srs):
            return pd.cut(srs, 5).astype('str'), False

        if set(srs.unique()) == {0, 1} and srs.name == 'gender':
            return srs.replace({0 : "Female", 1 : "Male"}), False

        if set(srs.unique()) == {0, 1}:
            human_srs_name = srs.name.replace('_', ' ')
            return srs.replace({0 : f"No {human_srs_name}", 1 : f"{human_srs_name.capitalize()}"}), False

        if pd.api.types.is_integer_dtype(srs):
            if srs.nunique() <= 10:
                # assumed to be ordinal category
                return srs, False
            else:
                return pd.cut(srs, 5, precision=0).astype('str'), False

        return srs, True

    if columns is None:
        if target is None:
            columns = df.columns
        else:
            columns = df.drop(target, axis=1).columns

    for col in columns:
        srs, if_trunc = modify_srs_tell_if_trunc(df[col])

        if target is None:
            groupby = pd.Series(srs, name='count').groupby(srs).size().to_frame().reset_index()
        else:
            groupby = pd.concat([srs, df[target]], axis=1).groupby(col, as_index=False).agg({target : aggfunc})

        if if_trunc:
            groupby = trunc_data_for_display(groupby)
        
        #title
        if target is None:
            title = f'Breakdown of {col}'
        else:
            title = f'{aggfunc.capitalize()} of {target} by {col}'


        fig = fmt_bar(groupby, title=title, **kwargs)
        fig.show()
    return None