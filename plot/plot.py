# -*- coding:utf-8 -*-
import astropy
import bokeh

class Plot(object):
    """
    2D plot base class

    Plot holds a reference to a datasource and figure.
    Its duty is to project the data values into figure axes; the units
    from the figure pov has more like a labeling role. So, data properties
    are all controlled by datasource, figure should only know which columns
    to use for values, colors and other labels.

    As a 2D plot, 'x' and 'y' are main quantities here after 'data'.
    """
    _data = None
    _figure = None
    _file = 'plot.html'
    def __init__(self,data,x=None,y=None,xunit=None,yunit=None):
        """
        Init Plot. See `~set_data` for parameters info.
        """
        self._setup(data,x,y,xunit,yunit)

    def _setup(self,data,x,y,xunit,yunit):
        self.init_figure()
        self.set_data(data,x,y,xunit,yunit)

    def init_figure(self):
        from bokeh.plotting import figure as figureb
        self._figure = figureb()   # default tools are set automatically

    def set_data(self,data,x,y,xunit,yunit,label=None):
        """
        Configure data and metadata to Plot

        Input:
        - data : `~pandas.DataFrame` or `booq.plot.datasource.Q2dDataSource`
            data columns labeled according to x,y arguments
        - x : str
            label of a column in data, to be used as "x" axis
        - y : str
            label of a column in data, to be used as "y" axis
        - xunit : str or `~astropy.units.Unit`
            unit to set "xaxis" with
        - yunit : str or `~astropy.units.Unit`
            unit to set "yaxis" with
        """
        from .datasource import Q2dDataSource
        if label is None:
            label = 'nolabel'
        if x is None:
            x = 'x'
        if y is None:
            y = 'y'
        if data is None:
            label = 'fake'
            data = Utils.fake_data(x,y)
        self._data = Q2dDataSource(label,data,x,y,xunit,yunit)

    def draw(self):
        self.set_output(self._file)
        self._assembly()

    def _assembly(self):
        """
        _assembly is run by self.draw() to put things together
        """
        self._figure.xaxis.axis_label = "{} ({})".format(self.xlabel,self.xunit.to_string())
        self._figure.yaxis.axis_label = "{} ({})".format(self.ylabel,self.yunit.to_string())

    def data(self):
        return self._data.datasource()

    @property
    def xlabel(self):
        return self._data.xlabel

    @property
    def ylabel(self):
        return self._data.ylabel

    @property
    def xunit(self):
        return self._data.xunit

    @property
    def yunit(self):
        return self._data.yunit

    def set_xlabel(self,label):
        self._xlabel = label

    def set_ylabel(self,label):
        self._ylabel = label

    def set_xunit(self,unit):
        self._data.set_xunit(unit)

    def set_yunit(self,unit):
        self._data.set_yunit(unit)

    def has_data(self):
        return self._data.has_data()

    def has_axis(self,axis):
        if not self.has_data():
            return False
        return self._data.has_column(axis)

    def set_output(self,filename):
        from bokeh.io import output_file as bofile
        bofile(filename)

    def display(self):
        self.draw()
        self.show()

    def show(self):
        from bokeh.io import show as showb
        showb(self._figure)


class Scatter(Plot):
    """
    """
    def __init__(self,data=None,x=None,y=None,xunit=None,yunit=None):
        super(Scatter,self).__init__(data,x,y,xunit,yunit)

    def _setup(self,data,x,y,xunit,yunit):
        super(Scatter,self)._setup(data,x,y,xunit,yunit)

    def draw(self):
        super(Scatter,self).draw()
        self._figure.circle(source=self.data(),
                            x=self.xlabel,
                            y=self.ylabel)
