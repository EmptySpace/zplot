from bokeh.charts import BoxPlot

def simple(data,label,values):
    p = BoxPlot(data, values=values, label=label, outliers=True)
    return p
