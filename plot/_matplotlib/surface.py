#
"""Module for image bright surfaces visualization/analysis"""

import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter #, FixedLocator
from matplotlib import pyplot
import numpy


# ---

def profile(img):
    """
    Returns a Pyplot object with a 3D projection of 'img'
    
    Input:
     - img  ndarray : Image array (ndim=2,dtype=float)
    
    Output:
     - plt  <matplotlib.pyplot instance>
    
    ---
    """
    np = numpy;
    plt = matplotlib.pyplot;
    
    x,y = img.shape
    x = np.arange(x)
    y = np.arange(y)
    X,Y = np.meshgrid(x,y)
    extent = (X.min(), X.max(), Y.min(), Y.max())
    
    # Twice as wide as it is tall.
    fig = plt.figure(figsize=plt.figaspect(0.5))
    
    S = np.log10(img+1)
    s_min = S.min()
    s_max = S.max()
    ds = (s_max-s_min)/20
    
    ax = fig.add_subplot(1,1,1, projection='3d')    
    surf = ax.plot_surface(X,Y,S,rstride=1,cstride=1,cmap=cm.jet,linewidth=0,antialiased=False)
    ax.set_zlim3d(s_min-ds,s_max+ds)
#    ax.set_title('3D projection, log10 scale. Max: %s, Min: %s'%(s_max,s_min))

    fig.colorbar(surf, shrink=0.5, aspect=5)

    return plt


def bright(img):
    """
    Returns a Pyplot object with a 3D projection of 'img'
    
    Input:
     - img  ndarray : Image array (ndim=2,dtype=float)
    
    Output:
     - plt  <matplotlib.pyplot instance>
    
    ---
    """
    np = numpy;
    plt = matplotlib.pyplot;
    
    x,y = img.shape
    x = np.arange(x)
    y = np.arange(y)
    X,Y = np.meshgrid(x,y)
    extent = (X.min(), X.max(), Y.min(), Y.max())
    
    # Twice as wide as it is tall.
    #fig = plt.figure(figsize=plt.figaspect(0.5))
    fig = plt.figure()
    
    S = np.log10(img+1)
    s_min = S.min()
    s_max = S.max()
    ds = (s_max-s_min)/20
    
    ax = fig.add_subplot(1,1,1)
    ax.imshow(S[::-1,:],extent=extent)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    plt.colorbar()

    return plt


profile3D = profile
profile2D = bright


def overlay(img,regions):
    """
    """
    import pymorph;
    plt = pyplot;
    
    over_img = pymorph.overlay(img,regions)
    plt.imshow(over_img)
    
    return plt
