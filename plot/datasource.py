# -*- coding:utf-8 -*-
import bokeh
import astropy

class DataSource(object):
    """
    """
    _data = None
    _label = None
    _description= None

    def __init__(self,label,data=None):
        if data != None:
            self.set_data(data)
        self.__set_label(label)

    def __set_label(self,label):
        self._label = label
    def __get_label(self):
        return self._label
    label = property(__get_label,doc="Get DS's label")

    def set_data(self,data):
        ds = UtilsDS.data2source(data)
        self._data = ds

    def datasource(self):
        return self._data

    def get_data(self,column=None):
        df = self._data.to_df()
        if column:
            df = df[column]
        return df
    # data = property(get_data, doc="Get data' DataFrame")

    def data_from_column(self,column):
        return self.get_data(column)

    def has_column(self,column):
        return column in self._data.column_names

    def has_data(self):
        return bool(len(self._data.column_names))


class QDataSource(DataSource):
    """
    """
    _columns = None

    def __init__(self,label,data=None):
        super(QDataSource,self).__init__(label,data)


from .axis import QMetadata
class Q2dDataSource(QDataSource):
    """
    """
    def __init__(self,label,data,x,y,xunit,yunit):
        """
        Keeps data,columns and units under control

        Input:
        - label :   str
                    name for this dataset
        - data :    dict, pandas.DataFrame
                    data set
        - x :       str
                    name for the column (x) found in data
        - y :       str
                    name for the column (y) found in data
        - xunit :   str, astropy.unit.Unit
                    name/unit for column x
        - yunit :   str, astropy.unit.Unit
                    name/unit for column y
        """
        super(Q2dDataSource,self).__init__(label,data)
        assert self.has_column(x) and self.has_column(y)
        self._x = QMetadata(x,xunit)
        self._y = QMetadata(y,yunit)

    def set_data(self,data,xunit=None,yunit=None):
        super(Q2dDataSource,self).set_data(data)
        if xunit != None:
            self._x = QMetadata(self.xlabel,xunit)
        if yunit != None:
            self._y = QMetadata(self.ylabel,yunit)

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def xlabel(self):
        # print "xlabel at Q2d",self.x.label
        return self.x.label

    @property
    def ylabel(self):
        # print "ylabel at Q2d",self.y.label
        return self.y.label

    @property
    def xunit(self):
        return self.x.unit

    @property
    def yunit(self):
        return self.y.unit

    def set_xunit(self,unit):
        Utils = UtilsUnits
        _data = Utils.convert(self.xdata(),self.xunit,unit)
        self.set_xdata(_data,unit)

    def set_yunit(self,unit):
        Utils = UtilsUnits
        _data = Utils.convert(self.ydata(),self.yunit,unit)
        self.set_ydata(_data,unit)

    def xdata(self):
        return self._cdata(self.xlabel)

    def ydata(self):
        return self._cdata(self.ylabel)

    def _cdata(self,column):
        return self.data_from_column(column)

    def set_xdata(self,data,unit):
        # Notice that x-label cannot be change, yet
        _data = self._cdata_set(self.xlabel,data)
        self.set_data(_data,xunit=unit)

    def set_ydata(self,data,unit):
        # Notice that y-label cannot be change, yet
        _data = self._cdata_set(self.ylabel,data)
        self.set_data(_data,yunit=unit)

    def _cdata_set(self,column,data):
        _data = self.get_data()
        _data[column] = data
        return _data
