# -*- coding:utf-8 -*-

# # Put them together. Example:
# p_hists.plot_height = 400
#
# p_boxplot_SR.tools = p_hists.tools
# p_boxplot_SR.x_range = p_hists.x_range
# p_boxplot_SR.plot_height = p_hists.plot_height/2
# p_boxplot_SR.xaxis.axis_label = None
# p_boxplot_SR.ygrid.minor_grid_line_color = None
#
# p_boxplot_SB.tools = p_hists.tools
# p_boxplot_SB.x_range = p_hists.x_range
# p_boxplot_SB.plot_height = p_hists.plot_height/2
# p_boxplot_SB.xaxis.axis_label = None
# p_boxplot_SB.ygrid.minor_grid_line_color = None
#
# from bokeh.plotting import gridplot
# p = gridplot([[p_hists],[p_boxplot_SB],[p_boxplot_SR]])
# show(p)



def _init_figure(tools=None, logscale=False):
    '''
    Init a bokeh figure with some defaults

    Defaults are:
     - tools: PanTool,
              BoxZoomTool,
              WheelZoomTool,
              ResizeTool,
              ResetTool,
              HelpTool

    Input:
     - tools    : list ~bokeh.models.tools.Tool
                If given, add to already defined (i.e, default) set
     - logscale : bool
                If 'True', y-axis will present a log scale

    Output:
     - Figure : ~bokeh.plotting.figure
    '''
    from bokeh.plotting import figure
    from bokeh.models.tools import  PanTool,\
                                    BoxZoomTool,\
                                    WheelZoomTool,\
                                    ResizeTool,\
                                    ResetTool,\
                                    HelpTool
    from bokeh.models.tools import  Tool
    TOOLS = [PanTool(),BoxZoomTool(),WheelZoomTool(),ResizeTool(),ResetTool(),HelpTool()]
    if tools:
        if not isinstance(tools,(list,tuple)):
            tools = [tools]
        for t in tools:
            if not issubclass(t.__class__,Tool):
                continue
            TOOLS.append(t)
    #TOOLS = 'pan,box_zoom,wheel_zoom,crosshair,hover,resize,reset'
    y_type = 'log' if logscale else 'linear'
    fig = figure(tools=TOOLS,y_axis_type=y_type)
    return fig


def bar_stacked(tops, bottoms, bins, label, color, alpha=None, figure=None):
    """
    """
    figure = _bar(tops,bottoms,bins[:-1],bins[1:],
                   label,color,alpha=alpha,figure=figure)
    return figure


def bar_adjacent(counts, lefts, rights, label, color, alpha=None, figure=None):
    """
    """
    bottoms = [0]*len(counts)
    figure = _bar(counts,bottoms,lefts,rights,
                   label,color,alpha=alpha,figure=figure)
    return figure


def _bar(tops,bottoms,lefts,rights,label,color,alpha,figure):
    """
    """
    assert len(tops)==len(bottoms) or isinstance(bottoms,(int,float))
    assert len(lefts)==len(rights)

    if figure is None:
        figure = _init_figure()

    figure.quad(top=tops,
                bottom=bottoms,
                left=lefts,
                right=rights,
                fill_color=color,fill_alpha=alpha,
                legend=label)
    return figure


def bar(counts, bins, label, color, alpha=0.5, figure=None):
    '''
    Generate a bar plot for 'counts' in 'bins'

    If 'figure' is given, it adds the new plot to it.

    Input:
     - counts   : array-like
                Y-axis values
     - bins     : array-like
                Limiting values for x-axis bins
     - label    : string
                Label for this distribution
     - color    : string
                Code defining a color; see ~booq.plot.Colors
     - alpha    : float
                Alpha channel for bars created
     - figure   : ~pandas.plotting.figure

    Output:
     - Figure : ~bokeh.plotting.figure
    '''
    bottoms = [0]*len(counts)
    figure = bar_stacked(tops=counts,
                           bottoms=bottoms,
                           bins=bins,
                           label=label,
                           color=color,
                           alpha=alpha,
                           figure=figure)
    return figure


def step(counts, bins, label, color, figure):
    '''
    Add a step plot to 'figure', inplace of a bar plot

    Input:
     - counts   : array-like
                Y-axis values
     - bins     : array-like
                Limiting values for x-axis bins
     - label    : string
                Label for this distribution
     - color    : string
                Code defining a color; see ~booq.plot.Colors
     - figure   : ~pandas.plotting.figure

    Output:
     - Figure : ~bokeh.plotting.figure
    '''
    _y,_x = histogram2stepfunction(counts,bins)
    figure.line(x=_x,
                y=_y,
                #line_color="#D95B43",line_width=2,
                line_color=color,line_width=2,
                legend='label')

    import numpy as np
    _x = np.diff(bins)/2+bins[:-1]
    figure.circle(x=_x,
                  y=_y,
                  #line_color="#D95B43",line_width=2,
                  line_color=color,line_width=2,
                  fill_color="white",fill_alpha=1,
                  size=9,
                  legend='label')
    return figure


def multihist(df, column, groupby, nbins=10, mode='adjacent', logscale=False):
    '''
    Generate a bar plots from 'column' data , grouped by 'groupby' column

    Input:
     - df       : ~pandas.DataFrame
     - column   : string
                Name of the column to histogram
     - groupby  : string
                Column to use for spliting the samples
     - mode     : 'over', 'stacked', 'adjacent'
                How to disposal the plot bars
     - logscale : bool
                If 'True', Y-axis goes in log-scale

    Output:
     - Figure : ~bokeh.plotting.figure
    '''
    from bokeh.models import CrosshairTool,HoverTool

    import numpy as np

    fig = _init_figure(tools=[CrosshairTool(),HoverTool()],logscale=logscale)

    label = column

    fig.select(CrosshairTool).dimensions = ['height']
    fig.select(HoverTool).mode = 'hline'
    label_tip = "{}: ".format(label)
    fig.select(HoverTool).tooltips = [(label_tip,"$x")]

    fig.xgrid.grid_line_color = None            #vgrid
    fig.ygrid.minor_grid_line_color = 'gray'    #hgrid
    fig.ygrid.minor_grid_line_alpha = 0.5
    fig.ygrid.minor_grid_line_dash = 'dashed'

    fig.yaxis.axis_label = 'Counts'
    fig.xaxis.axis_label = label

    from booq.utils.arrays import binning
    bins = binning(df[column],nbins,spacing='linear')

    from booq.plot import Colors
    groups = df.groupby(groupby)
    ngroups = len(groups)
    colors = Colors.get_colors(ngroups)

    from booq.utils.arrays import histogram
    counts_last = [0]*(len(bins)-1)
    for i,grp_key in enumerate(groups.indices.keys()):
        _data = df.loc[groups.groups[grp_key],(column)]
        _counts,_bins = histogram(_data,bins)

        if mode is 'stacked':
            _counts = [ _counts[_i]+_bot for _i,_bot in enumerate(counts_last) ]
            fig = bar_stacked(_counts,counts_last,_bins,str(i),colors[i],figure=fig)

        elif mode is 'adjacent':
            bins_left,bins_right = [],[]
            for _i in range(len(bins)-1):
                _binsplit = np.linspace(bins[_i],bins[_i+1],ngroups+1)
                bins_left.append(_binsplit[i])
                bins_right.append(_binsplit[i+1])
            fig = bar_adjacent(_counts,bins_left,bins_right,str(i),colors[i],figure=fig)

        else:
            fig = bar(_counts,bins,str(i),colors[i],alpha=float(1)/ngroups,figure=fig)
        last_counts = _counts[:]
        #else:
        #    _data = df.loc[grp.groups[grp_key],(column)]
        #    _counts,_bins = Distro.histogram(_data,bins)
        #    fig = PlotHisto.step(_counts,bins,i,fig)

    return fig

def hist_compare(df, columns, bins, logscale=False):
    '''
    Overplot two histograms

    Plot the histograms (bar-plot and step-plot) for *two* columns of
    a dataframe. The columns are assumed to share the same range of
    values. Nevertheless, the actual range for the X-axis is defined
    by 'bins', which is supposed to be *not* a scalar but the vector
    defining where histograms/columns are to be divided.

    Input:
     - data     : ~pandas.DataFrame
                Dataframe containing (numerical) 'columns'
     - columns  : list of strings
                Name of the columns in 'data'
     - bins     : array-like
                Limiting values for histogram bins
     - logscale : bool
                If 'True', y-axis (counts) is in log-scale

    Output:
     - Figure : ~bokeh.plotting.figure
    '''
    from bokeh.plotting import figure
    from bokeh.models import CrosshairTool,HoverTool

    TOOLS = 'pan,box_zoom,wheel_zoom,crosshair,hover,resize,reset'

    if logscale:
        p_hists = figure(tools=TOOLS,y_axis_type='log')
    else:
        p_hists = figure(tools=TOOLS)
    p_hists.xgrid.grid_line_color = None
    p_hists.ygrid.minor_grid_line_color = 'gray'
    p_hists.ygrid.minor_grid_line_alpha = 0.5
    p_hists.ygrid.minor_grid_line_dash = 'dashed'

    p_hists.select(CrosshairTool).dimensions = ['height']

    p_hists.select(HoverTool).mode = 'hline'
    p_hists.select(HoverTool).tooltips = [("mag: ","$x")]

    x_label = columns[0]
    y_label = columns[1]

    x = df[x_label]
    y = df[y_label]

    hs,b = np.histogram(x,bins=bins,normed=False)

    p_hists.quad(top=hs,
                   bottom=0,
                   left=bins[:-1],
                   right=bins[1:],
                   fill_color="#036564",fill_alpha=0.5,
                   legend=x_label)

    hp,b = np.histogram(y,bins=bins,normed=False)

    hh,bb = histogram2stepfunction(hp,b)

    p_hists.line(x=bb,
                   y=hh,
                   line_color="#D95B43",line_width=2,
                   legend=y_label)

    _b = np.diff(bins)/2+bins[:-1]
    p_hists.circle(x=_b,
                     y=hp,
                     size=9,line_color="#D95B43",line_width=2,
                     fill_color="white",fill_alpha=1,
                     legend=y_label)

    p_hists.yaxis.axis_label = 'Counts'

    return p_hists


def boxplot(df, column, by, mean=True):
    '''
    Generate boxplot from 'column' data divided by 'by' column data

    Input:
     - df       : ~pandas.DataFrame
                Data containing 'column' and 'by' columns
     - column   : string
                Name of the column to plot the (box) distributions
     - by       : string
                Name of the column to use for spliting the data
     - mean     : bool
                If 'True', data will be 'mean'-subtracted

    Output:
     - Figure : ~bokeh.plotting.figure
    '''
    from bokeh.plotting import figure
    from bokeh.models import Range1d

    if mean:
        df['mean'] = df.groupby(by)[column].transform(lambda x:x-x.mean())
    groups = df.groupby(by)
    if mean:
        groups = groups['mean']
    else:
        groups = groups[column]

    # Now a 'try' statement because I don't want to get in to the details when I'm not using Pandas.Categories.
    try:
        # When the 'by' argument is a DF.Categorical instance, we have to do some transforming.
        # Notice that this Categorical (group) labels are strings containing the range of each bin,
        #  so that we get from them the "left,right" values for the 'bins' to use.
        import re
        _bins = set([ re.sub(r'[^\d.]+','',s) for c in df[by].values.categories for s in c.split(',') ])
        _bins = list(_bins)
        _bins.sort()
        _bins = np.asarray(_bins,dtype=np.float)
    except:
        # Now, when I am not using Categories. In particular, now I'm passing "by" as a vector/Series
        # containing numerical labels
        assert False
    _diff = np.diff(_bins)
    _center = _bins[:-1] + _diff/2

    # Generate some synthetic time series for six different categories
    cats = [ s for s,g in groups ]

    # Find the quartiles and IQR foor each category
    q1 = groups.quantile(q=0.25)
    q2 = groups.quantile(q=0.5)
    q3 = groups.quantile(q=0.75)
    iqr = q3 - q1
    upper = q3 + 1.5*iqr
    lower = q1 - 1.5*iqr

    # find the outliers for each category
    def outliers(group):
        cat = group.name
        return group[(group > upper.loc[cat][0]) | (group < lower.loc[cat][0])]
    out = groups.apply(outliers).dropna()

    # Prepare outlier data for plotting, we need coordinate for every outlier.
    outx = []
    outy = []
    for i,cat in enumerate(cats):
        # only add outliers if they exist
        if not out.loc[cat].empty:
            for value in out[cat]:
                outx.append(_center[i])
                outy.append(value)

    p = figure(title="")

    from bokeh.models import FixedTicker
    p.x_range = Range1d(_bins.min(),_bins.max())
    p.xaxis.ticker = FixedTicker(ticks=_center)

    # If no outliers, shrink lengths of stems to be no longer than the minimums or maximums
    qmin = groups.quantile(q=0.00)
    qmax = groups.quantile(q=1.00)
    upper = [ min([x,y]) for (x,y) in zip(qmax,upper) ]
    lower = [ max([x,y]) for (x,y) in zip(qmin,lower) ]

    # stems
    p.segment(_center, upper, _center, q3, line_width=2, line_color="black")
    p.segment(_center, lower, _center, q1, line_width=2, line_color="black")

    # boxes
    p.rect(_center, (q3+q2)/2, _diff/2, q3-q2,    fill_color="#E08E79", line_width=2, line_color="black")
    p.rect(_center, (q2+q1)/2, _diff/2, q2-q1,    fill_color="#3B8686", line_width=2, line_color="black")

    # whiskers (almost-0 height rects simpler than segments)
    p.rect(_center, lower, _diff/4, 0.002, line_color="black")
    p.rect(_center, upper, _diff/4, 0.002, line_color="black")

    # outliers
    p.circle(outx, outy, size=6, color="#F38630", fill_alpha=0.6)

    p.xgrid.grid_line_color = None
    p.ygrid.minor_grid_line_color = 'gray'
    p.ygrid.minor_grid_line_alpha = 0.5
    p.ygrid.minor_grid_line_dash = 'dashed'

    p.xaxis.major_label_text_font_size="12pt"
    p.xaxis.major_label_orientation = -3.14/2

    p.xaxis.axis_label = by
    p.yaxis.axis_label = column if not mean else column + ' (0-mean)'

    return p
