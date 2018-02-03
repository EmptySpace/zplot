# -*- coding:utf-8 -*-

"""
.. module:: wavebands
    :synopsis: Multiwavelength data plot

.. moduleauthor:: Carlos Brandt <carloshenriquebrandt@gmail.com>
"""

from astropy import units

from elements import PointElement,LineElement

from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib import cm
# from matplotlib import rc
# rc('text',usetex=True)

from numpy import diff,log10,mean,sqrt

# Module to create a plot for surveys spectral sensitivity, all wavebands:
# - Radio
# - Millimeter
# - Infrared
# - Optical
# - UV
# - EUV
# - X-ray
# - Gamma-ray
#
# Let's write down the structure that defines the (wave)bands.
# This is temporary, it should go to a config file in the near future.
# Each entry in WAVEBANDS (structure/config) is headed by the waveband name
#  (as the key) pointing to a dictionary with the following key/values:
#  - unit : (length,frequency/energy) unit in which 'min/max' are expressed
#  - min  : limiting minimum value
#  - max  : limiting maximum value
class Config:
    from collections import OrderedDict
    WAVEBANDS = OrderedDict({
                    'radio' : {
                        'unit' : 'mm',
                        'min' : 10,
                        'max' : 1000 # support inf/nan
                    },
                    'millimeter' : {
                        'unit' : 'mm',
                        'min' : 0.1,
                        'max' : 10
                    },
                    'infrared' : {
                        'unit' : 'um',
                        'min' : 1,
                        'max' : 100
                    },
                    'optical' : {
                        'unit' : 'angstrom',
                        'min' : 3000,
                        'max' : 10000
                    },
                    'uv' : {
                        'unit' : 'nm',
                        'min' : 100,
                        'max' : 300
                    },
                    'euv' : {
                        'unit' : 'nm',
                        'min' : 10,
                        'max' : 100
                    },
                    'xray' : {
                        'unit' : 'angstrom',
                        'min' : 0.1,
                        'max' : 100
                    },
                    'gammaray' : {
                        'unit' : 'angstrom',
                        'min' : 0.001, # support inf/nan
                        'max' : 0.01
                    },
    })
    MATHFUNCS = {   'linear': lambda x:x,
                    'log'   : log10,
                }

class Waveband(object):
    """
    Waveband definition, limits and adequate units
    """
    _frequency_unitname = 'Hz'
    _energy_unitname = 'eV'

    def __init__(self,name,min,max,unit):
        super(Waveband,self).__init__()
        assert isinstance(name,str) and name is not ''
        self._name = name
        self._unit = units.Unit(unit,parse_strict='silent')
        _min = float(min)
        _max = float(max)
        assert _min < _max
        self._min = _min
        self._max = _max
        self._range = [float(min), float(max)] * self._unit

        self._unit_nu = units.Unit(self._frequency_unitname)
        self._unit_E = units.Unit(self._energy_unitname)

    def __str__(self):
        n = self.name
        mn = float(self.min.value)
        mx = float(self.max.value)
        u = str(self.unit)
        return '%15s %.3e %.3e %s' % (n,mn,mx,u)

    @property
    def name(self):
        """
        Waveband name
        """
        return self._name

    @property
    def unit(self):
        """
        Return the default unit
        """
        return self._unit

    def min(self,unit=None):
        """
        Return the lower limiting value
        """
        _m = min(self.limits(unit))
        return _m

    def max(self,unit=None):
        """
        Return the upper limiting value
        """
        _m = max(self.limits(unit))
        return _m

    def limits(self,unit=None):
        """
        Return waveband limits as a astropy.units.Quantity pair
        """
        if unit is None:
            return self._range
        _lim =  self.convert(self._range,units.Unit(unit))
        _lmin = min(_lim)
        _lmax = max(_lim)
        _q = units.Quantity([_lmin,_lmax])
        return _q

    def limits_wavelength(self,unit=None):
        """
        Convenient function to retrieve limits in the default wavelength unit
        """
        _unit = self._unit if unit is None else units.Unit(unit)
        return self.limits(_unit)

    def limits_frequency(self,unit=None):
        """
        Convenient function to retrieve limits in the default frequency unit
        """
        _unit = self._unit_nu if unit is None else units.Unit(unit)
        return self.limits(_unit)

    def limits_energy(self,unit=None):
        """
        Convenient function to retrieve limits in the default energy unit
        """
        _unit = self._unit_E if unit is None else units.Unit(unit)
        return self.limits(_unit)

    @staticmethod
    def convert(quantities,unit):
        """
        Convenient function to convert between equivalent quantities
        """
        return (quantities).to(unit,equivalencies=units.spectral())

class Wavebands(list):
    def __init__(self,wavebands=None):
        super(Wavebands,self).__init__()
        self._unit = None
        if wavebands is not None:
            assert isinstance(wavebands,dict)
            self.init(wavebands)

    def init(self,wavebands):
        for band,lims in wavebands.items():
            _wb = Waveband(band,lims['min'],lims['max'],lims['unit'])
            self.append(_wb)


    def append(self,band):
        assert isinstance(band,Waveband)
        super(Wavebands,self).append(band)

    @property
    def min(self):
        """
        Retrieves the minimum among all wavebands
        """
        _min = float('+inf')
        for _b in self:
            _min = min(_min,_b.min(self._unit))
        assert _min >= 0.0, "_min is %f" % _min
        return _min

    @property
    def max(self):
        _max = float('-inf')
        for _b in self:
            _max = max(_max,_b.max(self._unit))
        assert _max > 0.0, "_max is %f" % _max
        return _max

    def __getUnit__(self):
        return self._unit

    def __setUnit__(self,unit=None):
        if unit is not None:
            unit = units.Unit(unit)
        self._unit = unit

    unit = property(__getUnit__,__setUnit__,
            doc="Get/Set default unit")

# The core of the plot function -- annotation and axis cloning -- were
#  taken from stackoverflow's post by Joe Kington:
#  http://stackoverflow.com/a/3919443/687896
# I should eveolve the function in the near future to use Matplotlib's
#  AxesGrid toolkit, like in this example:
#  http://matplotlib.org/mpl_toolkits/axes_grid/users/overview.html#axisartist-with-parasiteaxes
#
def wavebands(surveysData,title='Surveys depth',
                xunit='GHz',xlabel='log \nu',
                yunit='erg/s.cm2',ylabel='Depth',yscale='log',
                **kwargs):
    """
    Plot

    Args:
        - surveysData : ElementList
        - title : str
            Plot title
        - xunit : str
            Unit to use on the x-axis. Should be a frequency, wavelength or energy.
        - xlabel : str
            Label for 'x', whitout unit information
        - yunit : str
            Unit to use on the y-axis.
        - ylabel : str
            Label for 'y', whitout unit information
        - kwargs available:
            - draw_legend : True|False
            - draw_labels : True|False

    Returns:
        - a matplotlib.pyplot object

    """

    draw_legend = kwargs['draw_legend'] if kwargs.has_key('draw_legend') else None
    draw_labels = kwargs['draw_labels'] if kwargs.has_key('draw_labels') else None

    config = Config()

    # Setup the plot
    fig = plt.figure()
    fig.set_size_inches(18, 9, forward=True)
    ax = fig.add_subplot(111)
    # Give ourselves a bit more room at the bottom
    plt.subplots_adjust(bottom=0.25)
    # Put some color on it..
    cmap = cm.get_cmap('spectral_r')

    # set_yscale()
    if yscale=='log':
        plt.yscale('log')

    # set_grid(on)
    ax.yaxis.grid(True)

    # set_labels(title,x,y)
    ax.set_xlabel(r'%s(%s)' % (xlabel,xunit))
    ax.set_ylabel(r'%s(%s)' % (ylabel,yunit))
    ax.set_title(r'%s' % title)

    # Construct the wavebands (x) domain
    #
    bands = Wavebands(config.WAVEBANDS)
    bands.unit = xunit

    # set_xlim()
    from math import log10 as mlog10
    x_max = int(mlog10(bands.max.value))
    x_min = 0
    ax.set_xlim(x_min,x_max)

    # Create a function to be used after for normalization of x/colors
    normalize = lambda x,x_min=x_min,x_max=x_max: ((x-x_min)/(x_max-x_min))

    # Finally, plot the data
    # x_data = []
    # y_data = []
    # eLabel = []
    for i,elem in enumerate(surveysData):
        x = elem.x.asunit(xunit).value
        y = elem.y.asunit(yunit).value
        x = log10(x)
        v = normalize(mean(x))
        c = cmap(v)
        if isinstance(elem,LineElement):
            pl, = ax.plot(x, y, lw=5, linestyle='-', color=c)
        else:
            assert isinstance(elem,PointElement)
            pl, = ax.plot(x, y, marker='*', color=c)
        # x_data.append(mean(x))
        # y_data.append(mean(y))
        # eLabel.append(elem.label)
        # Put some labels over the points
        #TODO: autofix overwriten labels
        pl.set_label(elem.label)
        if draw_labels:
            x = mean(x)
            y = mean(y)
            _yfrac = 0.25*y
            if yscale is not 'log':
                assert yscale is 'linear'
                _yfrac = 0.5
            y += _yfrac
            ax.text(x, y, elem.label, ha="center", va="center", size=8, fontweight='bold')#, bbox=bbox_props)

    _ymin,_ymax = ax.get_ylim()
    _ymin *= 0.9
    # _ymin = 0
    _ymax *= 1.1
    ax.set_ylim(_ymin,_ymax)

    draw_wavebands = True
    if draw_wavebands:
        y = (_ymin,_ymax)
        for b in bands:
            x = mlog10(b.max(xunit).value)
            v = normalize(x)
            if v < x_max:
                c = cmap(v)
                ax.plot([x,x],y,lw=2,linestyle='--',color=c,alpha=0.25,zorder=0)

    # The blok below does some good work over this 'overwritten' stuff; from:
    #   [http://stackoverflow.com/q/8850142/687896]
    # There is other way of doing it using networkx:
    #   [http://stackoverflow.com/q/14938541/687896]
    if False:
        txt_height = 0.04*(plt.ylim()[1] - plt.ylim()[0])
        txt_width = 0.02*(plt.xlim()[1] - plt.xlim()[0])
        #Get the corrected text positions, then write the text.
        text_positions = get_text_positions(x_data, y_data, txt_width, txt_height)
        text_plotter(x_data, y_data, text_positions, eLabel, ax, txt_width, txt_height)


    if draw_legend:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.draw()

    return plt

# --------------------------------------------------------------------------
def get_text_positions(x_data, y_data, txt_width, txt_height):
    a = zip(y_data, x_data)
    text_positions = y_data[:]
    for index, (y, x) in enumerate(a):
        local_text_positions = [i for i in a if i[0] > (y - txt_height)
                            and (abs(i[1] - x) < txt_width * 2) and i != (y,x)]
        if local_text_positions:
            sorted_ltp = sorted(local_text_positions)
            if abs(sorted_ltp[0][0] - y) < txt_height: #True == collision
                differ = diff(sorted_ltp, axis=0)
                a[index] = (sorted_ltp[-1][0] + txt_height, a[index][1])
                text_positions[index] = sorted_ltp[-1][0] + txt_height
                for k, (j, m) in enumerate(differ):
                    #j is the vertical distance between words
                    if j > txt_height * 2: #if True then room to fit a word in
                        a[index] = (sorted_ltp[k][0] + txt_height, a[index][1])
                        text_positions[index] = sorted_ltp[k][0] + txt_height
                        break
    return text_positions

def text_plotter(x_data, y_data, text_positions, labels, axis,txt_width,txt_height):
    for x,y,t,l in zip(x_data, y_data, text_positions, labels):
        axis.text(x - txt_width, 1.01*t, '%s'%(l),rotation=0, color='blue')
        if y != t:
            axis.arrow(x, t,0,y-t, color='red',alpha=0.3, width=txt_width*0.1,
                       head_width=txt_width, head_length=txt_height*0.5,
                       zorder=0,length_includes_head=True)

# --------------------------------------------------------------------------
