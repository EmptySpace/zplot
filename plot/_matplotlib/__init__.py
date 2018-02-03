# -*- coding:utf-8 -*-

from . import histogram
# import plot
from . import scatter
from . import surface

def boxhist(data, column, **kwargs):
    '''
    Create a boxplot and a histogram out of 'data[column]'

    A figure with two panels is created.

    Input:
     - data : ~pandas.DataFrame
            table containing 'column'
     - column : string
            name of (numerical) column from 'data'

    Output:
     - ~matplotlib.pyplot object

    kwargs
    ------
     - bins : number of bins to use, or vector with bins limits
     - log : if True, use log scale on histogram
    '''
    kw = kwargs

    from matplotlib import pyplot as plt
    fig,axs = plt.subplots(1,2)
    fig.set_size_inches(15,5)

    data[column].plot.box(ax=axs[0])

    bins = kw['bins'] if kw.has_key('bins') else 10
    log = kw['log'] if kw.has_key('log') else False
    data[column].plot.hist(ax=axs[1],bins=bins,log=log)

    return plt


def scatter3d(df, columns):
    '''
    Generate a x,y,z plot from 'df' given three 'columns' names
    '''
    assert len(columns) == 3

    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')

    xl = columns[0] ; xs = df[xl]
    yl = columns[1] ; ys = df[yl]
    zl = columns[2] ; zs = df[zl]

    ax.scatter(xs,ys,zs, marker='o')
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.set_zlabel(zl)

    return plt
