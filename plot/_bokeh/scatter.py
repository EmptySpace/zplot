# -*- coding:utf-8 -*-

def scatter_color(df, x_column, y_column, color_column, color_logscale=False):
    '''
    Plot (scatter) points colored after a column from data
    '''
    from bokeh.plotting import figure
    from bokeh.models import Range1d

    title = "{0} .vs. {1}".format(x_column,y_column)

    TOOLS="pan,wheel_zoom,box_select,lasso_select,box_zoom,reset"

    # create the scatter plot
    p = figure(title = title,
               min_border = 10,
               min_border_left = 50,
               tools=TOOLS)

    p.xaxis.axis_label = x_column
    xmin = df[x_column].min()
    xmax = df[x_column].max()
    p.x_range = Range1d(start = xmin,
                        end = xmax)

    p.yaxis.axis_label = y_column
    ymin = df[y_column].min()
    ymax = df[y_column].max()
    p.y_range = Range1d(start = ymin,
                        end = ymax)

    def scalar2color(vector,logscale=False):
        import numpy as np
        if logscale:
            vector = np.log(1+vector)
        vmin = vector.min()
        vmax = vector.max()
        vfac = 255.0/(vmax-vmin)
        colors = []
        for v in vector:
            if np.isnan(v):
                clr = '#000000'
            else:
                r = int((v-vmin)*vfac)
                g = int(255-((v-vmin)*vfac))
                b = 0
                clr = "#%02x%02x%02x" % (r,g,b)
            colors.append(clr)
        return colors

    xs = df[x_column]
    ys = df[y_column]
    colors = scalar2color(df[color_column],color_logscale)
    p.scatter(xs, ys, size=5, fill_color=colors, fill_alpha=0.5, line_color=None)

    return p


def scatter_kde(x, y, x_label, y_label, lim_scatter=None):
    """
    """
    from bokeh.plotting import figure,gridplot
    from bokeh.models import Range1d

    title = "{} .vs. {}".format(x_label,y_label)

    TOOLS="pan,wheel_zoom,box_select,lasso_select,box_zoom,reset"

    x_min,x_max = x.min(),x.max()
    y_min,y_max = y.min(),y.max()
    if lim_scatter is None:
        _min = min(x_min,y_min)
        _max = max(x_max,y_max)
        lim_scatter = [_min,_max]

    # create the scatter plot
    p_scatter = figure(title = title,
                       min_border = 10,
                       min_border_left = 50,
                       tools=TOOLS)

    p_scatter.x_range = Range1d(start = lim_scatter[0], end = lim_scatter[1])
    p_scatter.xaxis.axis_label = x_label

    p_scatter.y_range = Range1d(start = lim_scatter[0], end = lim_scatter[1])
    p_scatter.yaxis.axis_label = y_label

    p_scatter.line(lim_scatter, lim_scatter, color='black', line_width=2)

    p_scatter.scatter(x, y, size=3, color="#3A5785", alpha=0.5)


    # Kernel Density Estimates
    #
    def kde(values,vmin=None,vmax=None):
        import numpy as np
        from scipy.stats import gaussian_kde
        kern = gaussian_kde(values)
        vmin = vmin if vmin is not None else values.min()
        vmax = vmax if vmax is not None else values.max()
        _cov = kern.covariance[0][0]
        pnts = np.arange(vmin, vmax, 10*_cov)
        kde = kern(pnts)
        return kde,pnts


    # Create the HORIZONTAL plot (former histogram)
    #
    #LINE_ARGS = dict(color="#3A5785", line_color=None)
    #

    x_kde,x_grid = kde(x,x_min,x_max)
    hmax = x_kde.max()*1.1

    p_kde_spec = figure(title = None,
                        plot_width = p_scatter.plot_width,
                        plot_height = p_scatter.plot_height/3,
                        x_range = p_scatter.x_range,
                        y_range = (0, hmax),
                        min_border = 10,
                        min_border_left = 50,
                        toolbar_location = None,
                        tools=TOOLS)
    p_kde_spec.xgrid.grid_line_color = None
    #
    p_kde_spec.line(x_grid, x_kde)#,**LINE_ARGS)


    # Create the VERTICAL plot (former histogram)
    #
    th = 42 # need to adjust for toolbar height, unfortunately
    #

    y_kde,y_grid = kde(y,y_min,y_max)
    vmax = y_kde.max()*1.1

    p_kde_photo = figure(title = None,
                        plot_width = p_scatter.plot_width/3,
                        plot_height = p_scatter.plot_height+th-10,
                        x_range = (0, vmax),
                        y_range = p_scatter.y_range,
                        min_border = 10,
                        min_border_top = th,
                        toolbar_location = None,
                        tools=TOOLS)
    p_kde_photo.ygrid.grid_line_color = None
    #
    p_kde_photo.line(y_kde, y_grid)


    # Let's adjust the borders
    #
    p_kde_photo.min_border_top = 80
    p_kde_photo.min_border_left = 0
    p_kde_photo.min_border_bottom = 50

    p_kde_spec.min_border_top = 10
    p_kde_spec.min_border_right = 10
    p_kde_spec.min_border_left = 80

    p_scatter.min_border_right = 10
    p_scatter.min_border_left = 80
    p_scatter.min_border_bottom = 50


    # Arrange them (the plots) to a regular grid
    #
    layout = gridplot([[p_scatter,p_kde_photo],[p_kde_spec,None]])

    return layout

def scatter_hists(x,y,x_label,y_label,lim_scatter=None):
    """
    """

    from bokeh.plotting import figure,gridplot
    from bokeh.models import Range1d,BoxSelectTool,LassoSelectTool
    import numpy as np

    title = "{} .vs. {}".format(x_label,y_label)

    TOOLS="pan,wheel_zoom,box_select,lasso_select,box_zoom,reset"

    x_min,x_max = x.min(),x.max()
    y_min,y_max = y.min(),y.max()
    if lim_scatter is None:
        _min = min(x_min,y_min)
        _max = max(x_max,y_max)
        lim_scatter = [_min,_max]

    x_bins = np.linspace(_min,_max,100)
    y_bins = x_bins

    # create the scatter plot
    p_hist2d = figure(title=title,
                      x_range = lim_scatter,
                      y_range = lim_scatter,
                      plot_width=600,
                      plot_height=600,
                      min_border=10,
                      min_border_left=50,
                      tools=TOOLS)
    p_hist2d.select(BoxSelectTool).select_every_mousemove = False
    p_hist2d.select(LassoSelectTool).select_every_mousemove = False

    p_hist2d.xaxis.axis_label = x_label
    p_hist2d.yaxis.axis_label = y_label

    p_hist2d.line(lim_scatter, lim_scatter, color='light_gray', line_width=1)

    # We will not plot the usual colorful/heatmap-like histogram, but a size-scaled one
    # So the next steps are to compute the histogram-2d itself, then clean it (no zero-counte)
    #  and (re)define the (x,y) grid to plot the points (scaled by the histogram bins counts).
    #
    hist2d, x_edges, y_edges = np.histogram2d(y, x, bins=(x_bins,y_bins))
    assert np.array_equal(x_edges,x_bins)
    assert np.array_equal(y_edges,y_bins)

    # remove null bins
    hist2d_y_ind, hist2d_x_ind = np.where(hist2d>0)

    y_edges = y_edges[hist2d_y_ind]
    x_edges = x_edges[hist2d_x_ind]
    hist2d = hist2d[hist2d_y_ind, hist2d_x_ind]

    # give them a size; no science here, just the old "looks-good"...
    _hmin = hist2d.min()
    _hmax = hist2d.max()
    hist2d_scale = 2 + 10 * np.log2( (hist2d-_hmin) / (_hmax-_hmin) +1 )

    # a DataFrame makes life easier when using Bokeh
    #
    _df = pd.DataFrame({'spec' :x_edges,
                        'photo':y_edges,
                        'scale':hist2d_scale})

    r_hist2d = p_hist2d.square(x =    _df['spec'],
                               y =    _df['photo'],
                               size = _df['scale'],
                               color = "#3A5785",
                               alpha = 0.5)
    #s = p.scatter(x=xs, y=xp, size=1, color="black")


    # create the horizontal histogram
    x_hist, hedges = np.histogram(x, bins=x_bins,
                                 range=lim_scatter)
    hzeros = np.zeros(len(x_hist)-1)
    hmax = max(x_hist)*1.1

    LINE_ARGS = dict(color="#3A5785", line_color=None)

    p_hist_spec = figure(toolbar_location=None,
                         plot_width=p_hist2d.plot_width,
                         plot_height=200,
                         x_range=p_hist2d.x_range,
                         y_range=(-hmax, hmax),
                         title=None,
                         min_border=10,
                         min_border_left=50,
                         tools=TOOLS)
    p_hist_spec.xgrid.grid_line_color = None

    p_hist_spec.quad(bottom=0, left=hedges[:-1], right=hedges[1:], top=x_hist, color="white", line_color="#3A5785")
    hh1 = p_hist_spec.quad(bottom=0, left=hedges[:-1], right=hedges[1:], top=hzeros, alpha=0.5, **LINE_ARGS)
    hh2 = p_hist_spec.quad(bottom=0, left=hedges[:-1], right=hedges[1:], top=hzeros, alpha=0.1, **LINE_ARGS)


    # create the vertical histogram
    y_hist, vedges = np.histogram(y, bins=y_bins,
                                   range=lim_scatter)
    vzeros = np.zeros(len(y_hist)-1)
    vmax = max(y_hist)*1.1

    th = 42 # need to adjust for toolbar height, unfortunately
    p_hist_photo = figure(toolbar_location=None,
                          plot_width=200,
                          plot_height=p_hist2d.plot_height+th-10,
                          x_range=(-vmax, vmax),
                          y_range=p_hist2d.y_range,
                          title=None,
                          min_border=10,
                          min_border_top=th,
                          tools=TOOLS)
    p_hist_photo.ygrid.grid_line_color = None
    p_hist_photo.xaxis.major_label_orientation = -3.14/2

    p_hist_photo.quad(left=0, bottom=vedges[:-1], top=vedges[1:], right=y_hist, color="white", line_color="#3A5785")
    vh1 = p_hist_photo.quad(left=0, bottom=vedges[:-1], top=vedges[1:], right=vzeros, alpha=0.5, **LINE_ARGS)
    vh2 = p_hist_photo.quad(left=0, bottom=vedges[:-1], top=vedges[1:], right=vzeros, alpha=0.1, **LINE_ARGS)

    p_hist_photo.min_border_top = 80
    p_hist_photo.min_border_left = 0

    p_hist_spec.min_border_top = 10
    p_hist_spec.min_border_right = 10

    p_hist2d.min_border_right = 10

    layout = gridplot([[p_hist2d,p_hist_photo],
                       [p_hist_spec,None]])

    def update(attr, old, new):
        inds = np.array(new['1d']['indices'])
        if len(inds) == 0 or len(inds) == len(x_edges):
            hhist1, hhist2 = hzeros, hzeros
            vhist1, vhist2 = vzeros, vzeros
        else:
            neg_inds = np.ones_like(x, dtype=np.bool)
            neg_inds[inds] = False
            hhist1, _ = np.histogram(x[inds_x], bins=hedges)
            vhist1, _ = np.histogram(y[inds_xp], bins=vedges)
            hhist2, _ = np.histogram(x[neg_inds_x], bins=hedges)
            vhist2, _ = np.histogram(y[neg_inds_y], bins=vedges)

        hh1.data_source.data["top"]   =  hhist1
        hh2.data_source.data["top"]   = -hhist2
        vh1.data_source.data["right"] =  vhist1
        vh2.data_source.data["right"] = -vhist2

    r_hist2d.data_source.on_change('selected', update)

    return layout
