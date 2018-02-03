from bokeh.charts import Histogram, Bar
import numpy as np
import pandas as pd

def simple(counts,bins=20):
    h,b = np.histogram(counts,bins=bins)
    df = pd.DataFrame({'c':counts})
    p = Histogram(counts,bins)
    return p,df

def group(data,values,color):
    p = Histogram(data,values=values,color=color)
    return p
    
def bar(data,label,values,groupBy=None):
    """
    Bar chart of data.

    Input:
     - data : data frame
     - label : column name to use as label/bins
     - values : column name to aggregate
     - groupBy : column name to group data in each bin
    """
    p = Bar(data,label=label,values=values,agg='count',group=groupBy)
    return p
