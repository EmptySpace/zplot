# -*- coding:utf-8 -*-
import astropy
import bokeh

class UtilsUnits:

    @staticmethod
    def unit(unit):
        from astropy import units as u
        if isinstance(unit,u.Unit):
            # nothing to do
            return unit
        if unit is None:
            # undefined unit
            return u.one
        if isinstance(unit,str):
            if not unit.strip():
                # undefined unit
                return u.one
        return u.Unit(unit,parse_strict='warn')

    @staticmethod
    def are_equivalents(unit1,unit2,equivalencies=None):
        Utils = UtilsUnits
        _u1 = Utils.unit(unit1)
        _u2 = Utils.unit(unit2)
        return _u1.is_equivalent(_u2,equivalencies=equivalencies)

    @staticmethod
    def is_dimensionless(unit):
        Utils = UtilsUnits
        _u = Utils.unit(unit)
        return _u.is_equivalent(1)


    @staticmethod
    def convert(data,old_unit,new_unit):
        if not len(data):
            return None
        Utils = UtilsUnits
        if not Utils.are_equivalents(old_unit,new_unit):
            return None
        _uo = Utils.unit(old_unit)
        _un = Utils.unit(new_unit)
        _do = data * _uo
        _dn = _do.to(_un)
        return _dn

class UtilsDS:

    @staticmethod
    def data2source(data):
        from bokeh.models import ColumnDataSource
        from pandas import DataFrame
        if data is None:
            return None
        assert isinstance(data,(dict,DataFrame))
        if not isinstance(data,ColumnDataSource):
            _df = data
            if isinstance(data,dict):
                _df = Utils.dict2df(data)
            source = ColumnDataSource(_df)
        else:
            source = data
        return source

class Utils:

    @staticmethod
    def label(label):
        try:
            return str(label)
        except:
            return ""

    @staticmethod
    def dict2df(dcy):
        assert isinstance(dcy,dict)
        from pandas import DataFrame
        df = DataFrame(dcy)
        return df

    @staticmethod
    def fake_data(x='x',y='y'):
        xy = dict({x:[1],y:[1]})
        return xy
