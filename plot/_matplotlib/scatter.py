#!/usr/bin/env python

#
# Example boxplot code
#

from matplotlib import pyplot as plt;
import numpy as np;
import math;

nyticks = 4;

# ---
def scalled(points,sizes,colors):
    """
    Plots a scatter graphic in a X,Y axes with sized symbols.

    Input:
     - points  <[(,)]> : a list of tuples of X,Y datapoints
     - sizes   <[]>    : a list of numbers for symbols sizing
     - colors  <[]>    : a list of numbers for symbols colors
    
    Output:
     - fig  <pyplot.figure()>  : A matplotlib figure instance
        Save (fig.savefig()) or show (fig.show()) the object.

    ---
    """

    if len(points) != len(sizes)  and  len(sizes) != 0:
        return False;

    RA,DEC = zip(*points);
    RA_min = min(RA);
    RA_max = max(RA);
    DEC_min = min(DEC);
    DEC_max = max(DEC);

    _ra = abs(RA_max-RA_min)/20;
    _dec = abs(DEC_max-DEC_min)/20;

    sizes = np.asarray(sizes)
    sizes = 10*(sizes/sizes.min())**2;
    sizes = sizes.astype('int32').tolist();

    x,y = zip(*points);
    
    # Sort the data just to get it(the plot) right done when 
    # 'scatter' run through datasets.
    _tosort = zip(sizes,colors,x,y);
    _tosort.sort();
    _tosort.reverse();
    sizes,colors,x,y = zip(*_tosort);
    #
    
    plt.set_cmap('hot');
    fig = plt.figure();
    ax = fig.add_subplot(111);
    ax.patch.set_facecolor('0.9');
    ax.grid(True);
    ax.set_xlabel('RA');
    ax.set_ylabel('DEC');
    ax.set_title('Massive objects scatter (position) plot')

    ax.scatter(x,y,s=sizes,c=colors,marker='o',edgecolor='black',alpha=0.75) #,color='blue');

#    fig.show();

    return fig;


# ---
