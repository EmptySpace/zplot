import sys

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter #, FixedLocator
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from astropy.io import fits

#from .. import image


# --

#def sersic():
#
#    X = np.arange(-5, 5, 0.1)
#    Y = np.arange(-5, 5, 0.1)
#    X, Y = np.meshgrid(X, Y)
#
#    R = np.sqrt(X**2 + Y**2)
#    R_s = R/R.max()
#    n_s = 2.
#    b_s = 1.
#    S = -b_s*(np.power(R_s,1./n_s) - 1)
#
#    return X,Y,S
#
## --
#def gauss_kern():
#    """ Returns a normalized 2D gauss kernel array for convolutions """
#    h1 = 5
#    h2 = 5
#    x, y = np.mgrid[0:h2, 0:h1]
#    x = x-h2/2
#    y = y-h1/2
#    sigma = 2.
#    g = np.exp( -( x**2 + y**2 ) / (2*sigma**2) );
#    return g / g.sum()

# --
def prof_33D(img):

    x,y = img.shape
    x = np.arange(0,x)
    y = np.arange(0,y)
    X,Y = np.meshgrid(x,y)
    extent = (X.min(), X.max(), Y.min(), Y.max())

    # Twice as wide as it is tall.
    fig = plt.figure(figsize=plt.figaspect(0.5))

    S = np.log10(img+1)
    s_min = S.min()
    s_max = S.max()
    ds = (s_max-s_min)/20

    Z = image.grad(img)**2
    Z = np.log10(np.sqrt(Z)+1)
    z_min = Z.min()
    z_max = Z.max()
    dz = (z_max-z_min)/20

    #---- First subplot - 3D
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    surf = ax.plot_surface(X, Y, S, rstride=1, cstride=1, cmap=cm.jet, linewidth=0, antialiased=False)
    ax.set_zlim3d(s_min-ds, s_max+ds)
    ax.set_title('Arc (HST) surface')

    #---- Second subplot - 2D
    ax = fig.add_subplot(2, 2, 2)
    ax.imshow(S[::-1,:],extent=extent)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    #fig.colorbar(surf, shrink=0.5, aspect=5)


    #---- First subplot
    ax = fig.add_subplot(2, 2, 3, projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet, linewidth=0, antialiased=False)
    ax.set_zlim3d(z_min-dz, z_max+dz)
    ax.set_title('Arc surface gradient')

    #---- Second subplot
    ax = fig.add_subplot(2, 2, 4)
    ax.imshow(Z[::-1,:],extent=extent)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')


    return plt

# --
if __name__ == "__main__":

    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option('-c','--conv', default=False,
                      dest='conv', action='store_true',
                      help="Convolve?")
    parser.add_option('-o','--out', default='profile.png',
                      dest='out', help="Output filename")
    opts,args = parser.parse_args()

    if not args:
        parser.print_help()
        sys.exit()

    img = fits.getdata(args[0])

    # Convolve --------
    if opts.conv:
        g = gauss_kern()
        img = signal.convolve(img,g,mode='valid')

    plot = prof_33D(img)
    plot.savefig(opts.out)

