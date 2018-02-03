# -*- coding:utf-8 -*-

"""
Plotting package
"""

from booq.plot._matplotlib.sky import scatter_sky
from .color import Color

from .mybokeh import PlotScatter,PlotHisto

__PLOT_DEV='screen'
__PLOT_NAME=None
def set_device(dev):
    '''
    Input:
     - dev : string
        Name of the device to use or a file name
        Options are: 'notebook', 'screen' or a filename
    '''
    global __PLOT_DEV
    global __PLOT_NAME
    if not dev in ['notebook', 'screen']:
        # it is a filename
        __PLOT_DEV = 'file'
        __PLOT_NAME = str(dev) or 'output.html'
    else:
        __PLOT_DEV = dev
        __PLOT_NAME = None
def get_device():
    if __PLOT_NAME is None:
        return __PLOT_DEV
    return __PLOT_NAME
