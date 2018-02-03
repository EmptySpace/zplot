import math
import numpy
import matplotlib
from matplotlib import pyplot

#import bit

# ---

def barhist(hist,bins,vmarks=[],log=False,xlabel='X',ylabel='%'):
    """
    Plot histogram with vertical line marks

    Output:
     - plt  : pyplot object. plt.show() will give you the histogram.

    """
    plt = matplotlib.pyplot;

    width = 0.7*(bins[1]-bins[0])
    centers = (bins[:-1]+bins[1:])/2.
    plt.grid(True)
    plt.bar(centers, hist, width=width, align='center', color='g', log=log, alpha=0.9)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    ymax = hist.max()*1.01
    ymin = -ymax/50.

    for i,iv in enumerate(vmarks):
        ymaxn = ymax + i%2*(ymin*2)
        plt.vlines(iv,ymin,ymaxn,color='r')
        plt.text(iv,ymaxn,"%.1f"%(iv))

    return plt

simple = barhist;
#hist = barhist

# --

def imhist(img,bins=100,vmarks=[],log=False,xlabel='X',ylabel='%'):
    """
    Plot histogram from image array
    #TODO This function should not remain here.

    Output:
     - plt  : pyplot object. plt.show() will give you the histogram.

    """
    np = numpy
    hist,bins = np.histogram(img.flatten(),bins=bins,normed=True)
    #hist,bins = image.Distros.histogram(img,bins=bins,normed=True)

    return barhist(hist,bins,vmarks,log,xlabel,ylabel)

# --

def boxhist(data,nbins=None,outliers=True,legend=None,xlabel="X",ylabel="Normed counts",title="Distribution plot"):
    """
    Plots data distribution in two different ways: a boxplot and (over) a histogram.


    Input:
     - data       < [] > : A list of data values or (for comparative plots) a list of datasets(vectors)
     - nbins     < int > : Number of bins for histogram. If None (default) nbins will be in somewhere between 10 and 100
     - outliers < bool > : If False, outliers (points) will not be shown
     - legend     < [] > : Data label for each dataset
     - xlabel    < str > : Label for X axis
     - ylabel    < str > : Label for Y axis
     - title     < str > : Figure title

    Output:
     - fig < plt.figure() > : Matplotlib figure instance

    ---
    """
    np = numpy;
    plt = matplotlib.pyplot;

#    if type(data) is not np.ndarray:
#        data = np.array(data);
    try:
        ND = data.ndim;
        if ND == 2:
            ND = data.shape[1];
    except:
        ND = len(data);

    colors=['b', 'r', 'y', 'k', 'g', 'm', 'c'];

    # Bins width:
    #
    def binswidth(dados):
        width = 3.49 * np.std(dados) * math.pow(dados.size,-1/3.);
        return width;
    ##
    if not nbins:
        _dat = np.asarray(data);
        _w = binswidth(_dat.ravel());
        _min = _dat.ravel().min();
        _max = _dat.ravel().max();
        nbins = (_max-_min)/_w;
        nbins = min(100,max(10,nbins));
        nbins = int(nbins)
        del _dat,_w,_min,_max;

    # Initialize the plot environment:
    fig,(ax1,ax2) = plt.subplots(2,1,sharex=True);

    # BOXPLOT
    bpD = ax1.boxplot(data,vert=0,sym='*',patch_artist=True,notch=1)#,positions=bp_positions[:-1],bootstrap=None);
    _box = bpD['boxes'];
    _med = bpD['medians'];
    _cap = bpD['caps'];
    _wsk = bpD['whiskers'];
    _flr = bpD['fliers'];

    # Fix the colors for each boxplot plotted..
    xmin = [];
    xmax = [];
    for i in range(ND):
        _box[i].set(color=colors[i],linewidth=2,alpha=0.5);
        _wsk[2*i].set(color=colors[i])#,linestyle=':');
        _wsk[2*i+1].set(color=colors[i])#,linestyle=':');
        #_med[i].set(color=colors[i]);
        #_cap[2*i].set(color=colors[i],linewidth=2);
        #_cap[2*i+1].set(color=colors[i],linewidth=2);
        xmin.extend(_wsk[i].get_xdata());
        xmax.extend(_wsk[i+1].get_xdata());

    # HISTOGRAM
    num,bin,p = ax2.hist(data, bins=nbins, normed=True, rwidth=0.8,
                            label=legend, color=colors[:ND], align='mid',
                            histtype='bar',alpha=0.5,linewidth=1)#,range=xrange);

    # Extra elements on the plots.
    # first, the median lines
    _median_x_bxplt = _med[0].get_xdata()
    _median_x = _median_x_bxplt[0]
    ax2.axvline(_median_x, label="median",
                linewidth=2,  linestyle='--', markerfacecolor=colors[:ND])
    ax2.xaxis.set_ticks(list(ax2.xaxis.get_majorticklocs()) + [_median_x])

    # Adjust the plots
    fig.subplots_adjust(hspace=0);
    ax1.yaxis.set_visible(False);
    #ax1.xaxis.set_visible(False);
    ax1.xaxis.set_ticks_position('top');
    ax2.xaxis.set_ticks_position('bottom');
    ax1.xaxis.grid(True)
    ax2.yaxis.grid(True)
    ax1.patch.set_facecolor('0.9');
    ax2.patch.set_facecolor('0.9');

    if not outliers:
        ax2.set_xlim((min(xmin),max(xmax)));

    fig.suptitle(title)
    ax2.set_ylabel(ylabel)
    ax2.set_xlabel(xlabel)
    ax2.legend(loc='upper right')

    return fig;

boxplot = boxhist
