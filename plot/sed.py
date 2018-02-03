# -*- coding:utf-8 -*-
"""Spectral Energy Distribution plot"""

import astropy
import bokeh

from .plot import Scatter
class SED(Scatter):
    """
    Specialization of Plot for SEDs

    The SED plot is a particular scatter plot, where 'x','y' axes are
    "energy" and "flux" (respectively) of an object's (foton) emission.

    Spectral Energy Distributions can span the electromagnetic energy range
    or just a particular waveband, and display different combinations of
    "energy-flux" units, depending on the particular kind of event/object
    being analyzed or the user's background. From my background, I'd say
    "energy" is usually expressed in units of frequency ('Hz'), and flux
    as 'mW/m2'.
    """
    _xunit_kind = 'Hertz'
    _yunit_kind = 'mW/m2'

    def __init__(self,data=None,x=None,y=None,xunit='Hertz',yunit='mW/m2'):
        super(SED,self).__init__(data,x,y,xunit,yunit)

    def _setup(self,data,x,y,xunit,yunit):
        """
        Basics of a (null) object setup
        """
        super(SED,self)._setup(data,x,y,xunit,yunit)
        self.set_xunit(xunit)
        # print "\nSED.SETUP-/",self.data().data,self.xlabel,self.ylabel
        self.set_yunit(yunit)
        # print "\nSED.SETUP-2",self.data().data,self.xlabel,self.ylabel
        self.set_xlabel(self.xunit.physical_type)
        self.set_ylabel(self.yunit.physical_type)

    def set_xunit(self,unit):
        if self._xunit_is_equivalent(unit):
            super(SED,self).set_xunit(unit)

    def set_yunit(self,unit):
        if self._yunit_is_equivalent(unit):
            super(SED,self).set_yunit(unit)

    def _xunit_is_equivalent(self,unit):
        from .utils import UtilsUnits as Utils
        from astropy.units import spectral
        assert not Utils.is_dimensionless(self.xunit)
        _a = Utils.are_equivalents(unit, self._xunit_kind, spectral())
        _b = Utils.are_equivalents(unit, self.xunit,       spectral())
        return _a and _b

    def _yunit_is_equivalent(self,unit):
        from .utils import UtilsUnits as Utils
        assert not Utils.is_dimensionless(self.yunit)
        _a = Utils.are_equivalents(unit, self._yunit_kind)
        _b = Utils.are_equivalents(unit, self.yunit)
        return _a and _b
