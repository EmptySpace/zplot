# -*- coding:utf-8 -*-
import bokeh
import astropy

class QMetadata(object):
    """
    This class looks like an Axes
    """
    _unit = None
    _label = None
    def __init__(self,label,unit):
        '''
        Give a name and a unit (~::astropy.units)
        '''
        assert isinstance(label,str)
        Utils = UtilsUnits
        self._unit = Utils.unit(unit)
        self._label = label
    @property
    def unit(self):
        return self._unit
    @property
    def label(self):
        return self._label

Axis = QMetadata
