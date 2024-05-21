import plotly
import plotly.express as px

def fmt_bar(df,
            group : str | None = None,
            color : str | None = None,
            value : str | None = None,
            display_values : bool = True,
            display_pct : bool = True,
            title: str | None = None,
            **kwargs) -> plotly.graph_objects.Figure:
    '''
    Make a Plotly barchart from the result of a groupby

    Args:
        df: pd.DataFrame, already groupby'd with as_index=False. Index is ignored in this function
        group: name of the column with primary categories to divide df by (None)
        color: name of the column for `color` parameter for px.bar() (None)
        value: name of the column whose values will be plotted (None)

        It is recommended to not specify columns, in which case: `group` is 1st columns, `value` is the last.
        If df has more than 2 column, which means that there are 2+ category columns, 2nd column in df becomes `color`.
        If df has 2 columns, `color` is same as `group`.

        If you have one of ```group, value, color``` specified, specify others as well

        display_values: whether to display values on bars (True)
        display_pct: whether to display %s on bars (True)
        title: plot title ('Breakdown by age')

        **kwargs: kwargs for plotly.express.bar().
        For other things - like layout - simply do:
            `fig = fmt_bar(...)
            fig.update_layout(...)`

    Returns:
        fig: plotly.graph_objects.Figure
    '''
    if group is None and value is None:
        group = df.columns[0]
        value = df.columns[-1]

    if df.shape[1] > 2 and color is None:
        color = df.drop([group, value], axis=1).columns[0]
    elif color is None:
        color = group

    #format bar annotations
    if display_pct:
        df['%'] = df[value] / df[value].sum()
        if display_values:
            text = [f'{value:.2f}<br>{pct:.2%}' for value, pct in zip(df[value], df['%'])]
        else:
            text = [f'{pct:.2%}' for pct in df['%']]
    else:
        if display_values:
            text = [f'{value:.2f}' for value in df[value]]
        else:
            text = None
  

    fig = px.bar(df, x=group, y=value, color=color, 
                 text=text, template='plotly_white', width=500, height=500, **kwargs)
    
    fig.update_traces(textfont_size=14, textangle=0, cliponaxis=False, textposition='outside')

    fig.update_layout(yaxis={'title':None,'tickfont':{'size':14}}, xaxis={'title':None,'tickfont':{'size':14}}, 
                      margin={'t':60,'b':20,'l':20,'r':20,'pad':0}, showlegend=False, title=title)
    return fig