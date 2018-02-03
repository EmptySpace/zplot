from bokeh.plotting import figure

TOOLS="resize,crosshair,pan,wheel_zoom,box_zoom,reset,tap,previewsave,box_select,lasso_select"

def simple(x, y, source=None,
            title='Scatter plot', xlabel=None, ylabel=None,
            color=None, size=5, alpha=0.5, fig=None):
    """
    """
    return_glyph = True

    if source != None:
        xlabel = xlabel or x
        ylabel = ylabel or y

    if fig is None:
        return_glyph = False
        fig = figure(title = title, tools = TOOLS)
        fig.xaxis.axis_label = xlabel
        fig.yaxis.axis_label = ylabel

    # instead of 'circle' one can use other symbol-named functions,
    # or use the Scatter class with options
    if source != None:
        glyph = fig.circle(x, y, source=source,
                            fill_alpha=alpha, line_alpha=alpha,
                            size=size, color=color)
    else:
        glyph = fig.circle(x, y,
                            fill_alpha=alpha, line_alpha=alpha,
                            size=size, color=color)

    if return_glyph:
        return glyph
    return fig

def xy(x,y,title='title',xlabel='x',ylabel='y',colors=None):

    p = simple(x,y,title=title,xlabel=xlabel,ylabel=ylabel,colors=colors)

    from bokeh.models.glyphs import Line
    from bokeh.models import ColumnDataSource, Range1d
    import numpy as np
    _max = max(x.max(),y.max())
    _min = min(x.min(),y.min())
    _x = np.linspace(_min,_max,10)
    _y = np.linspace(_min,_max,10)
    ds = ColumnDataSource(data=dict(x=_x,y=_y))
    line = Line(x='x',y='y')
    p.add_glyph(ds,line)
    p.x_range = Range1d(start=_min,end=_max)
    p.y_range = Range1d(start=_min,end=_max)
    return p
