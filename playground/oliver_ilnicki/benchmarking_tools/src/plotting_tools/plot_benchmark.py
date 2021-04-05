from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def barh_on_benchmark_results(df, x_col, y_col="model_name", logx=False, color_col=None, cmap="tab20", figsize=(10,20), **kwargs):
    """
        produces a barh plot given a dataframe containing benchmark results

        :param: df:         the dataframe
        :param: x_col:      the df column in the x axis, usually some metric to compare the models
        :param: y_col:      the df column in the y axis, usually the model name
        :param: logx:       when true, the x scale will be logarithmic
        :param: color_col:  the column to color the bars in the plot
        :param: cmap:       the colormap used by matplotlib to color the bars in the plot, only used when color_col is set
        :param: figsize:    The figure size passed to matplotlib to draw the plot
        :param: **kwargs:   Additional argument to pass to the plot function
        :returns: matplotlib figure and axes of the plot
    """
    sdf=df.sort_values(x_col, ascending=False)
    fig, ax = plt.subplots()
    if color_col is not None:
        color_col_unique=df[color_col].unique()
        colors=dict(zip(color_col_unique,np.arange(0.0,1.0,1.0/len(color_col_unique))+1.0/len(color_col_unique)))
        sdf.plot.barh(y_col, x_col, figsize=figsize, grid=True, logx=logx, 
                    color=[plt.cm.get_cmap(cmap)(colors[i]) for i in sdf[color_col]], ax=ax, **kwargs)
        ax.legend(handles=[Patch(facecolor=plt.cm.get_cmap(cmap)(colors[i]), edgecolor=plt.cm.get_cmap(cmap)(colors[i]),label=i) 
                        for i in color_col_unique])
    else:
        sdf.plot.barh(y_col, x_col, figsize=figsize, grid=True, logx=logx, **kwargs)
    return fig, ax

