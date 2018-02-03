# -*- coding:utf-8 -*-

import inspect
import pandas as pd

import bokeh
from bokeh.charts import defaults

from . import scatter

class PlotScatter:
    x = 'ra'
    y = 'dec'
    _columns = [ x,y ]

    NUMAX = 16

    def __init__(self):
        from collections import OrderedDict
        self.datasources = OrderedDict()

        self.fig = None

        self.set_palette()
        self._set_tools()
        self._set_device()

    def _set_tools(self):
        _tools = [  'pan',
                    'box_zoom',
                    'wheel_zoom',
                    'resize',
                    'reset',
                    'box_select',
                    'lasso_select',
                    # 'crosshair',
                    # 'tap',
                    'previewsave',
                    'help']
        self.tools = _tools

    @staticmethod
    def _set_device():
        from bokeh.io import output_file,output_notebook
        from booq.plot import get_device
        dev = get_device()
        if dev == 'notebook':
            output_notebook()
        elif dev == 'screen':
            output_file('/tmp/plot.html')
        else:
            output_file(dev)

    def set_palette(self,categorical=False):
        if not categorical:
            from bokeh.palettes import viridis,magma
            _vir = viridis(self.NUMAX)
            _mag = magma(self.NUMAX)
            colors = _mag[:]
            colors[1::2] = _vir[1::2]
            from random import shuffle,seed
            seed(1234567)
            shuffle(colors)
        else:
            from bokeh.palettes import Category20
            colors = Category20[20]

        self.palette = colors[:]

    def _get_color(self,ds_label):
        ds_index = self.datasources[ds_label].get('index')
        return self.palette[ds_index]

    def _create_figure(self):
        from bokeh.plotting import figure

        x_label = self.x
        y_label = self.y
        fig = figure(title='Scatter',
                     x_axis_label=x_label,
                     y_axis_label=y_label,
                     height = 300,
                     width = 600,
                     tools=self.tools,
                     toolbar_location="above",
                     toolbar_sticky=False)
        self.fig = fig

    def plot(self,ds_label):
        if not self.fig:
            self._create_figure()
        self._update_figure(ds_label)

    def _update_figure(self, ds_label):
        ds = self.datasources[ds_label]
        if 'data' in ds.keys():
            glyph = self._scatter(ds)
        elif 'region' in ds.keys():
            glyph = self._region(ds)
        self.datasources[ds_label].get('glyphs').append(glyph)

    def _scatter(self,ds,kind='simple'):
        import inspect
        assert isinstance(kind,str)
        fplot = eval('scatter.{}'.format(kind))
        assert inspect.isfunction(fplot)

        ds_label = ds['label']
        source = ds.get('data')
        color = self._get_color(ds_label)

        glyph = fplot(x = self.x,
                    y = self.y,
                    source = source,
                    color=color,
                    fig=self.fig)
        return glyph

    def _region(self,ds):
        ds_label = ds['label']
        source = ds['region']
        color = self._get_color(ds_label)
        alpha = 0.1

        if 'quad' in source.tags:
            from bokeh.models.glyphs import Quad
            glyph = Quad(left="left", right="right",
                        top="top", bottom="bottom",
                        fill_color=color, fill_alpha=alpha)
            self.fig.add_glyph(source,glyph)

        elif 'ellipse' in source.tags:
            from bokeh.models.glyphs import Ellipse
            glyph = Ellipse(x="x", y="y", width="width", height="height",
                            angle="angle",
                            fill_color=color, fill_alpha=alpha)
            self.fig.add_glyph(source, glyph)
        return glyph


    def add_dataset(self, data, label, x=None, y=None):
        from bokeh.models import ColumnDataSource

        xs = x or self.x
        ys = y or self.y

        _data = {
            self.x    : data[xs],
            self.y    : data[ys]
            }

        ds = ColumnDataSource(data=_data)
        ds_index = len(self.datasources)
        ds_label = '_'.join(label.split())
        i = None
        while ds_label in self.datasources:
            if not i:
                i = 2
                ds_label += '_{}'.format(i)
            else:
                i += 1
                ds_label = '{}{}'.format(ds_label[-1],str(i))

        self.datasources[ds_label] = { 'label':ds_label, 'data':ds,
                                        'index':ds_index, 'glyphs':[] }
        return ds_label

    def add_region(self, region, label):
        '''
        - To define a circle:
        region = {'center':(xc,yc), 'radius':rc}

        - To define a square:
        region = {'center':(xc,yc), 'side':d}

        - To define a rectangule:
        region = {'center':(xc,yc), 'sides':(dh,dv)}

        - To define a polygon:
        region = [(x1,y1), (x2,y2), (x3,y3), ...]
        '''
        reg_type = None

        def quad(center,hside,vside):
            from bokeh.models import ColumnDataSource
            xcenter,ycenter = center
            ds = ColumnDataSource(tags = ['quad'],
                data = dict(
                    top = [ycenter + vside/2],
                    bottom = [ycenter - vside/2],
                    left = [xcenter - hside/2],
                    right = [xcenter + hside/2]
                )
            )
            return ds

        def ellipse(center,radius,vaxis_ratio=1,angle=0):
            from bokeh.models import ColumnDataSource
            xcenter,ycenter = center
            ds = ColumnDataSource(tags = ['ellipse'],
                data = dict(
                    x = [xcenter],
                    y = [ycenter],
                    width = [radius],
                    height = [radius * vaxis_ratio],
                    angle = [angle]
                )
            )
            return ds

        if 'center' in region.keys():
            reg_center = [ float(v) for v in region.get('center') ]

            if 'radius' in region.keys():
                reg_type = 'circle'
                reg_radius = float(region.get('radius'))
                ds = ellipse(reg_center,reg_radius)
            elif 'side' in region.keys():
                reg_type = 'square'
                reg_side = float(region.get('side'))
                ds = quad(reg_center,reg_side,reg_side)
            else:
                assert 'sides' in region.keys()
                reg_type = 'rectangule'
                _sides = region.get('sides')
                reg_hedge = float(_sides[0])
                reg_vedge = float(_sides[1])
                ds = quad(reg_center,reg_hedge,reg_vedge)
        else:
            reg_type = 'polygon'
            reg_nodes = [ (float(x),float(y)) for x,y in region ]

        reg_label = '_'.join(label.split())
        ds_index = len(self.datasources)
        self.datasources[reg_label] = { 'label':reg_label, 'region':ds,
                                        'index':ds_index, 'glyphs':[] }
        return reg_label

    def grid(self,on=True):
        if self.fig:
            self.fig.xgrid.visible = on
            self.fig.ygrid.visible = on

    def show(self):
        from bokeh.io import show
        if self.fig:
            show(self.fig)

    # @property
    # def notebook(self):
    #     from bokeh.io import output_notebook
    #     output_notebook()


# =====================================================================


from booq.plot import Color
Colors = Color

def histogram2stepfunction(hist,bins):
    hist = hist.tolist()
    hh = hist+hist
    hh[::2] = hist
    hh[1::2] = hist
    hist = hh[:]
    bins = bins.tolist()
    bb = bins[:-1]+bins[1:]
    bb.sort()
    bins = bb[:]
    assert len(hist)==len(bins)
    return hist,bins

class Distro:

    @staticmethod
    def binning(vector,nbins,xmin=None,xmax=None,spacing='linear'):
        """
        """
        import numpy as np
        spacing_function = {'linear' : np.linspace}

        if spacing is not 'linear':
            spacing = 'linear'

        xmin = vector.min() if xmin is None else xmin
        xmax = vector.max() if xmax is None else xmax
        nbins = 10 if not (nbins >= 1) else nbins

        bins = spacing_function[spacing](xmin,xmax,nbins)
        return bins


    @staticmethod
    def histogram(vector,bins):
        """
        """
        import numpy as np

        h,b = np.histogram(vector,bins=bins,normed=False)
        assert np.array_equal(b,bins)
        return h,b


class PlotHisto:
    x = 'bins'
    y = 'oops'
    _columns = [ x,y ]

    NUMAX = 16

    def __init__(self):
        from collections import OrderedDict
        self.datasources = OrderedDict()

        self.fig = None

        self.set_palette()
        # self._set_tools()
        self._set_device()

    # def _set_tools(self):
    #     _tools = [  'pan',
    #                 'box_zoom',
    #                 'wheel_zoom',
    #                 'resize',
    #                 'reset',
    #                 'box_select',
    #                 'lasso_select',
    #                 # 'crosshair',
    #                 # 'tap',
    #                 'previewsave',
    #                 'help']
    #     self.tools = _tools

    def show(self):
        from bokeh.io import show
        if self.fig:
            show(self.fig)

    @staticmethod
    def _set_device(plotfile='histo.html'):
        from bokeh.io import output_file,output_notebook
        from booq.plot import get_device
        dev = get_device()
        if dev == 'notebook':
            output_notebook()
        elif dev == 'screen':
            output_file(plotfile)
        else:
            output_file(dev)

    def set_palette(self,categorical=False):
        if not categorical:
            from bokeh.palettes import viridis,magma
            _vir = viridis(self.NUMAX)
            _mag = magma(self.NUMAX)
            colors = _mag[:]
            colors[1::2] = _vir[1::2]
            from random import shuffle,seed
            seed(1234567)
            shuffle(colors)
        else:
            from bokeh.palettes import Category20
            colors = Category20[20]

        self.palette = colors[:]

    def _get_color(self,ds_label):
        ds_index = self.datasources[ds_label].get('index')
        return self.palette[ds_index]

    def _create_figure(self):
        self.fig = self.init_figure()

    def plot(self,ds_label):
        if not self.fig:
            self._create_figure()
        self._update_figure(ds_label)

    def _update_figure(self, ds_label):
        ds = self.datasources[ds_label]
        if 'data' in ds.keys():
            glyph = self._histo(ds)
        elif 'region' in ds.keys():
            assert False, "Not implemented yet"
            # glyph = self._region(ds)
        self.datasources[ds_label].get('glyphs').append(glyph)

    def _histo(self,ds,kind='simple'):
        ds_label = ds['label']
        source = ds.get('data')
        color = self._get_color(ds_label)
        df = source.to_df()
        glyph = hist_compare(df,columns=self.x,bins=None,logscale=False,
                            fig=self.fig,color=color,label=ds_label)
        return glyph

    @staticmethod
    def init_figure(tools = None,logscale=False):
        """
        """
        from bokeh.plotting import figure
        from bokeh.models.tools import  PanTool,\
                                        BoxZoomTool,\
                                        WheelZoomTool,\
                                        ResizeTool,\
                                        ResetTool,\
                                        HelpTool,\
                                        CrosshairTool,HoverTool
        from bokeh.models.tools import  Tool
        TOOLS = [PanTool(),BoxZoomTool(),WheelZoomTool(),ResizeTool(),
                    ResetTool(),HelpTool(),CrosshairTool(),HoverTool()]
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

    @staticmethod
    def bar(counts,bins,label,color,alpha=0.5,figure=None):
        """
        """
        bottoms = [0]*len(counts)
        figure = PlotHisto.bar_stacked(tops=counts,
                                       bottoms=bottoms,
                                       bins=bins,
                                       label=label,
                                       color=color,
                                       alpha=alpha,
                                       figure=figure)
        return figure

    @staticmethod
    def bar_stacked(tops,bottoms,bins,label,color,alpha=1,figure=None):
        """
        """
        figure = PlotHisto._bar(tops,bottoms,bins[:-1],bins[1:],
                               label,color,alpha=alpha,figure=figure)
        return figure

    @staticmethod
    def bar_adjacent(counts,lefts,rights,label,color,alpha=1,figure=None):
        """
        """
        bottoms = [0]*len(counts)
        figure = PlotHisto._bar(counts,bottoms,lefts,rights,
                               label,color,alpha=alpha,figure=figure)
        return figure

    @staticmethod
    def _bar(tops,bottoms,lefts,rights,label,color,alpha,figure):
        """
        """
        assert len(tops)==len(bottoms) or isinstance(bottoms,(int,float))
        assert len(lefts)==len(rights)

        if figure is None:
            figure = init_figure()
        tops = list(tops)
        bottoms = list(bottoms)
        lefts = list(lefts)
        rights = list(rights)

        figure.quad(top=tops,
                    bottom=bottoms,
                    left=lefts,
                    right=rights,
                    fill_color=color,fill_alpha=alpha,
                    legend=label)

        return figure

    @staticmethod
    def step(counts,bins,label,color,figure):
        """
        """
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


    @staticmethod
    def multihist(df,column,groupby,nbins=10,mode='over',logscale=False):
        """
        mode = [over,stacked,adjacent]
        """
        from bokeh.models import CrosshairTool,HoverTool

        import numpy as np

        fig = PlotHisto.init_figure(tools=[CrosshairTool(),HoverTool()],logscale=logscale)

        label = column

        fig.select(CrosshairTool).dimensions = 'height'
        fig.select(HoverTool).mode = 'hline'
        label_tip = "{}: ".format(label)
        fig.select(HoverTool).tooltips = [(label_tip,"$x")]

        fig.xgrid.grid_line_color = None            #vgrid
        fig.ygrid.minor_grid_line_color = 'gray'    #hgrid
        fig.ygrid.minor_grid_line_alpha = 0.5
        fig.ygrid.minor_grid_line_dash = 'dashed'

        fig.yaxis.axis_label = 'Counts'
        fig.xaxis.axis_label = label

        bins = Distro.binning(df[column].dropna(),nbins,spacing='linear')

        groups = df.groupby(groupby)
        ngroups = len(groups)
        colors = Colors.get_colors(ngroups)

        counts_last = [0]*(len(bins)-1)
        for i,grp_key in enumerate(groups.indices.keys()):
            _data = df.loc[groups.groups[grp_key],(column)].dropna()
            _counts,_bins = Distro.histogram(_data,bins)

            if mode is 'stacked':
                _counts = [ _counts[_i]+_bot for _i,_bot in enumerate(counts_last) ]
                fig = PlotHisto.bar_stacked(_counts,counts_last,_bins,str(i),colors[i],figure=fig)

            elif mode is 'adjacent':
                bins_left,bins_right = [],[]
                for _i in range(len(bins)-1):
                    _binsplit = np.linspace(bins[_i],bins[_i+1],ngroups+1)
                    bins_left.append(_binsplit[i])
                    bins_right.append(_binsplit[i+1])
                fig = PlotHisto.bar_adjacent(_counts,bins_left,bins_right,str(i),colors[i],figure=fig)

            else:
                fig = PlotHisto.bar(_counts,bins,str(i),colors[i],alpha=float(1)/ngroups,figure=fig)
            last_counts = _counts[:]
            #else:
            #    _data = df.loc[grp.groups[grp_key],(column)]
            #    _counts,_bins = Distro.histogram(_data,bins)
            #    fig = PlotHisto.step(_counts,bins,i,fig)

        return fig

    def add_dataset(self, data, label, x=None):
        from bokeh.models import ColumnDataSource

        xs = x or self.x

        _data = {
            self.x    : data[xs]
            }

        ds = ColumnDataSource(data=_data)
        ds_index = len(self.datasources)
        ds_label = '_'.join(label.split())
        i = None
        while ds_label in self.datasources:
            if not i:
                i = 2
                ds_label += '_{}'.format(i)
            else:
                i += 1
                ds_label = '{}{}'.format(ds_label[-1],str(i))

        self.datasources[ds_label] = { 'label':ds_label, 'data':ds,
                                        'index':ds_index, 'glyphs':[] }
        return ds_label



def hist_compare(df,columns,bins=None,logscale=False,fig=None,color=None,label=None):
    import numpy as np

    return_glyph = True if fig!=None else False

    if isinstance(columns,str):
        x_label = columns
        y_label = None
    else:
        x_label = columns[0]
        y_label = columns[1] if len(columns)>1 else None

    x = df[x_label].dropna()
    y = df[y_label].dropna() if y_label else None

    if bins == None:
        bins = np.linspace(x.min(),x.max())
    hs,b = np.histogram(x,bins=bins,normed=False)

    from bokeh.plotting import figure
    from bokeh.models import CrosshairTool,HoverTool


    if fig == None:
        TOOLS = 'pan,box_zoom,wheel_zoom,crosshair,hover,resize,reset'
        if logscale:
            fig = figure(tools=TOOLS,y_axis_type='log')
        else:
            fig = figure(tools=TOOLS)
    fig.xgrid.grid_line_color = None
    fig.ygrid.minor_grid_line_color = 'gray'
    fig.ygrid.minor_grid_line_alpha = 0.5
    fig.ygrid.minor_grid_line_dash = 'dashed'

    fig.select(CrosshairTool).dimensions = 'height'

    fig.select(HoverTool).mode = 'hline'
    ttlabel = "{}: ".format(x_label)
    fig.select(HoverTool).tooltips = [(ttlabel,"$x")]

    color = color if color != None else "#036564"
    glyph = fig.quad(top=hs,
                   bottom=0,
                   left=bins[:-1],
                   right=bins[1:],
                   fill_color=color,fill_alpha=0.5,
                   legend=label)

    if y:
        hp,b = np.histogram(y,bins=bins,normed=False)

        hh,bb = histogram2stepfunction(hp,b)

        fig.line(x=bb,
                       y=hh,
                       line_color="#D95B43",line_width=2,
                       legend=y_label)

        _b = np.diff(bins)/2+bins[:-1]
        fig.circle(x=_b,
                         y=hp,
                         size=9,line_color="#D95B43",line_width=2,
                         fill_color="white",fill_alpha=1,
                         legend=y_label)

    fig.yaxis.axis_label = 'Counts'

    if return_glyph:
        return glyph
    return fig


def histboxgrid(histplot_central, boxplot_one, boxplot_two):
    p_hists = histplot_central
    p_boxplot_SB = boxplot_one
    p_boxplot_SR = boxplot_two

    p_hists.plot_height = 400

    p_boxplot_SR.toolbar = p_hists.toolbar
    p_boxplot_SR.x_range = p_hists.x_range
    p_boxplot_SR.plot_height = p_hists.plot_height//2
    p_boxplot_SR.xaxis.axis_label = None
    p_boxplot_SR.ygrid.minor_grid_line_color = None

    p_boxplot_SB.toolbar = p_hists.toolbar
    p_boxplot_SB.x_range = p_hists.x_range
    p_boxplot_SB.plot_height = p_hists.plot_height//2
    p_boxplot_SB.xaxis.axis_label = None
    p_boxplot_SB.ygrid.minor_grid_line_color = None

    from bokeh.plotting import gridplot
    grid = gridplot([[p_hists],[p_boxplot_SB],[p_boxplot_SR]])
    return grid



# =====================================================================


class PlotBox:

    @staticmethod
    def boxplot(df,column,by,mean=True):
        """
        """
        from bokeh.plotting import figure
        from bokeh.models import Range1d
        from numpy import concatenate

        if mean:
            _mean = df.groupby(by)[column].transform('mean')
            df['mean'] = df[column] - _mean
            del _mean
        groups = df.groupby(by)
        if mean:
            groups = groups['mean']
        else:
            groups = groups[column]

#         # Now a 'try' statement because I don't want to get in to the details when I'm not using Pandas.Categories.
#         try:
#             # When the 'by' argument is a DF.Categorical instance, we have to do some transforming.
#             # Notice that this Categorical (group) labels are strings containing the range of each bin,
#             #  so that we get from them the "left,right" values for the 'bins' to use.
#             import re
#             # The following line is not necessary for pandas 0.20
#             #_bins = set([ re.sub(r'[^\d.]+','',s) for c in df[by].values.categories for s in c.split(',') ])
#             # instead, we go for:
#             _cats = data['bins_SR'].values.categories
#             _bins = set(concatenate((_cats.right,_cats.left)))
#             _bins = list(_bins)
#             _bins.sort()
#             _bins = np.asarray(_bins,dtype=np.float)
#         except:
#             # Now, when I am not using Categories. In particular, now I'm passing "by" as a vector/Series
#             # containing numerical labels
#             assert False
#         _diff = np.diff(_bins)
#         _center = _bins[:-1] + _diff/2

        measures = groups.agg(['count'])
        index = measures.index.categories
        measures['width'] = index.right - index.left
        measures['center'] = (index.left + index.right)/2

        x_range = Range1d(index.left[0],index.right[-1])

        # Find the quartiles and IQR foor each category
        measures['q1'] = groups.quantile(q=0.25)
        measures['median'] = groups.median()#quantile(q=0.5)
        measures['q3'] = groups.quantile(q=0.75)
        iqr = measures.q3 - measures.q1
        measures['top'] = measures.q3 + 1.5*iqr
        measures['bottom'] = measures.q1 - 1.5*iqr

        # find the outliers for each category
        def outliers(group):
            cat = group.name
            return group[(group > measures.loc[cat,'top']) | (group < measures.loc[cat,'bottom'])]
        out = groups.apply(outliers).dropna()
        print('out',out)

        # Prepare outlier data for plotting, we need coordinate for every outlier.
        cats = [ s for s,g in groups ]
        print(cats)
        print(index)
        outx = []
        outy = []
        for i,cat in enumerate(cats):
            # only add outliers if they exist
            if not out.loc[cat].empty:
                for value in out[cat]:
                    outx.append(measures.loc[cat,'center'])
                    outy.append(value)

        # If no outliers, shrink lengths of stems to be no longer than the minimums or maximums
        qmin = groups.quantile(q=0.00)
        qmax = groups.quantile(q=1.00)
        upper = [ min([x,y]) for (x,y) in zip(qmax,measures['top'].values) ]
        lower = [ max([x,y]) for (x,y) in zip(qmin,measures['bottom'].values) ]



#         measures['upper_center'] = (measures.q2+measures.q3)/2
#         measures['upper_height'] = measures.q3-measures.q2
#         measures['lower_height'] = measures.q2-measures.q1
        measures['height'] = measures.q3-measures.q1
#         measures['lower_center'] = (measures.q2+measures.q1)/2

        measures = measures.reset_index()
        del measures[by]

        from bokeh.models import ColumnDataSource
#         _data = dict(
#                     center=measures.center,
#                     width=measures.width/2,
#                     upper_center=(measures.q2+measures.q3)/2,
#                     upper_height=measures.q3-measures.q2,
#                     lower_center=(measures.q2+measures.q1)/2,
#                     lower_height=measures.q2-measures.q1,
#                     top=measures.upper,
#                     bottom=measures.lower,
#                     q3=measures.q3,
#                     q2=measures.q2,
#                     q1=measures.q1
#                     )
#         _data = pandas.DataFrame(_data)
#         boxes = ColumnDataSource(_data)

        p = figure(title="")

        from bokeh.models import FixedTicker
        p.x_range = x_range
        p.xaxis.ticker = FixedTicker(ticks=measures.center.values)

        measures = measures.dropna(how='any')
        print(measures)
        boxes = ColumnDataSource(measures)

        # boxes
        p.rect(source=boxes, x='center', y='median',
                width='width', height='height',
                fill_color='blue', fill_alpha=0.1,
                line_width=2, line_color="black")

        # middle
        p.circle(source=boxes, x='center', y='median', size=10,
                fill_alpha=0, line_width=2, line_color="black")
        p.segment(x0=measures['center']-measures['width']/4,y0=measures['median'], line_width=3,
                  x1=measures['center']+measures['width']/4,y1=measures['median'], line_color="red")

        # stems
        p.segment(source=boxes, x0='center', y0='q3', x1='center', y1='top',
                line_color="blue")
        p.segment(source=boxes, x0='center', y0='q1', x1='center', y1='bottom',
                line_color="blue")

        # whiskers (almost-0 height rects simpler than segments)
        p.inverted_triangle(source=boxes, x='center', y='top', size=10)
        p.triangle(source=boxes, x='center', y='bottom', size=10)

        # outliers
        p.circle_x(outx, outy, size=6, color="#F38630", fill_alpha=0.6)

        p.xgrid.grid_line_color = None
        p.ygrid.minor_grid_line_color = 'gray'
        p.ygrid.minor_grid_line_alpha = 0.5
        p.ygrid.minor_grid_line_dash = 'dashed'

        p.xaxis.major_label_text_font_size="12pt"
        p.xaxis.major_label_orientation = -3.14/2

        p.xaxis.axis_label = by
        p.yaxis.axis_label = column if not mean else column + ' (0-mean)'

        return p
