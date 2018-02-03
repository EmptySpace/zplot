# -*- coding:utf-8 -*-

# from collections import namedtuple
# _Point = namedtuple('Point','id, x, xunit, y, yunit',verbose=False)
# _Line = namedtuple('Line','id, xi, xf, xunit, yi, yf, yunit',verbose=False)

def fabric(label,x,y,xunit=None,yunit=None):
    if not (isinstance(x,(tuple,list)) or isinstance(y,(tuple,list))):
        # It's a "point"
        return PointElement(label,x,y,xunit,yunit)
    else:
        # Considered to be a "line"
        return LineElement(label,x,y,xunit,yunit)


from astropy.units import Quantity,Unit,spectral
class MyQuantity(Quantity):

    def __init__(self,coord,unit):
        self.q = Quantity(coord,unit)

    @property
    def value(self):
        return self.q.value

    @property
    def unit(self):
        return self.q.unit

    def set_value(self,value):
        assert isinstance(value,(float,int))
        _unit = self.q.unit
        self.q = Quantity(value,_unit)

    def set_unit(self,unit):
        assert isinstance(unit,(str,Unit))
        _newQ = self.q.to(unit)
        self.q = _newQ

    def asunit(self,unit):
        _q = self.q.to(unit,equivalencies=spectral())
        return _q

class Element(object):
    """
    """

    def __init__(self,label,x,y,xunit,yunit):
        super(Element,self).__init__()
        self._x = MyQuantity(x,xunit)
        self._y = MyQuantity(y,yunit)
        self._label = label

    @property
    def label(self):
        return self._label

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    def set_x(self,x):
        self._x.set_value(x)

    def change_xunit(self,unit):
        self._x.set_unit(unit)

    def set_y(self,y):
        self._y.set_value(y)

    def change_yunit(self,unit):
        self._y.set_unit(unit)


class PointElement(Element):
    def __init__(self,label,x,y,xunit,yunit):
        assert isinstance(x,(float,int)) and isinstance(y,(float,int))
        assert isinstance(xunit,str) and isinstance(yunit,str)
        super(PointElement,self).__init__(label,x,y,xunit,yunit)


class LineElement(Element):
    def __init__(self,label,x,y,xunit,yunit):
        assert isinstance(x,(tuple,list)) and isinstance(y,(tuple,list))
        assert isinstance(xunit,str) and isinstance(yunit,str)
        super(LineElement,self).__init__(label,x,y,xunit,yunit)


class ElementList(list):
    def __init__(self):
        super(ElementList,self).__init__()
        self._xunit = None
        self._yunit = None

    def append(self,element):
        assert isinstance(element,Element)
        super(ElementList,self).append(element)
    add = append
    
    def set_xunit(unit):
        _u = Unit(unit)
        self._xunit = _u

    def set_yunit(unit):
        _u = Unit(unit)
        self._yunit = _u
