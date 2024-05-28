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


class Corr_explorer():
    def __init__(self, df, threshold = 0.6):
        '''
        As this is a constructor for class not used by itself, for docs please see corr()
        '''
        corr = df.corr().unstack().to_frame().reset_index()
        corr.columns = ['var1', 'var2', 'corr']

        corr = corr[corr['var1'] != corr['var2']]
        
        corr['abs_corr'] = corr['corr'].abs()
        corr = corr.sort_values(by='abs_corr', ascending=False)
        corr = corr[corr['abs_corr'] > threshold]

        # we use this trick to hide mirrored rows
        corr = corr.iloc[::2]
        self.table = corr

        return None


    def graph(self, figsize : tuple = (12, 10)) -> None:
        import networkx as nx
        import matplotlib.pyplot as plt

        G = nx.from_pandas_edgelist(
            self.table,
            source='var1',
            target='var2',
            edge_attr='abs_corr'
        )

        plt.figure(figsize=figsize)
        plt.tight_layout()

        pos = nx.circular_layout(G)

        edges = nx.draw_networkx_edges(G, pos=pos, edge_color=self.table['abs_corr'], width=4, edge_cmap=plt.cm.viridis_r)
        nodes = nx.draw_networkx_nodes(G, pos=pos, alpha=.5)

        plt.colorbar(edges)

        nx.draw_networkx_labels(G, pos=pos, font_size=10)
        return None


    def top(self) -> pd.DataFrame:
        top_df = pd.DataFrame()
        for col in pd.concat([self.table['var1'], self.table['var2']]).unique():
            srs = self.table.loc[self.table['var1'] == col, 'abs_corr']
            srs = pd.concat([srs, self.table.loc[self.table['var2'] == col, 'abs_corr']])

            top_df = pd.concat([top_df, pd.DataFrame({'var' : [col], 'mean_corr' : [srs.mean()], 'n' : [len(srs)]})])

        top_df = top_df.sort_values(by='n', ascending=False)
        return top_df


def corr(df, threshold=0.6):
    '''
    For when a heatmap is not enough. A constructor for correlation explorer class.

    Args:
        df
        threshold: threshold for absolute correlation to include a var into the table

    Returns:
        Instance of Corr_explorer
    
    Attributes:
        table: df of highly-correlated variables 

    Methods:
        graph: graph relationship between highly-correlated variables with Matplotlib and NetworkX.
            Experimental feature, as 1) no automatic text-wrapping is a available for labels, so it looks bad
            2) the best layout for not-totally-interconnected graph is the circular one, which also looks bad
        top: creates table with var, mean_corr and n columns:
            var: every unique column name (from self.table)
            mean_corr: mean correlation in high correlation pairs, where the particular var is present
            n: number of columns that this feature is highly correlated with
    '''
    return Corr_explorer(df, threshold)