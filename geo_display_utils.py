"""utils for displaying geos"""
from __future__ import print_function,division,absolute_import
import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
from ylm_utils import reconstruct_from_alm
from polygon_utils import get_healpix_pixelation
#def plot_reconstruction(m,ax,pixels,reconstruct):
#    """display reconstruction on basemap m"""
#    lats = (pixels[:,0]-np.pi/2.)*180/np.pi
#    lons = pixels[:,1]*180/np.pi
#    x,y = m(lons,lats)
#    #have to switch because histogram2d considers y horizontal, x vertical
#    #fig = plt.figure(figsize=(10,5))
#    minC = np.min(reconstruct)
#    maxC = np.max(reconstruct)
#    bounds = np.linspace(minC,maxC,10)
#    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
#    #ax = fig.add_subplot(121)
#    H1,yedges1,xedges1 = np.histogram2d(y,x,100,weights=reconstruct)
#    X1, Y1 = np.meshgrid(xedges1, yedges1)
#    pc1 = ax.pcolormesh(X1,Y1,-H1,cmap='gray')
#    ax.set_aspect('equal')
def plot_reconstruction(reconstruction,title,fig,cmap='Greys',cbar=True,notext=False):
    """plot the reconstruction of pixels"""
    #hp.mollview(reconstruction,title=title,fig=fig,hold=True,cmap=cmap,cbar=cbar,notext=notext,xsize=2000,margins=[0.,0.,0.,0.])
    hp.mollview(reconstruction,title=title,fig=fig,hold=True,cmap=cmap,cbar=cbar,notext=notext)
    hp.graticule(dmer=360,dpar=360,alpha=0)

def reconstruct_and_plot(geo,l_max,pixels,title,fig,cmap='Greys',do_round=False,cbar=True,notext=False):
    """plot the area enclosed by a geo on a pixelated map"""
    alms = geo.get_alm_table(l_max)
    reconstruction = reconstruct_from_alm(l_max,pixels[:,0],pixels[:,1],alms)
    if do_round:
        reconstruction = np.round(reconstruction)
    plot_reconstruction(reconstruction,title,fig,cmap,cbar=cbar,notext=notext)

def display_geo(geo,l_max,res_healpix=5,title='geo display',fig=None,cmap='Greys',display=True,do_round=False,cbar=True,notext=False):
    """plot the area enclosed by a geo on a pixelated map"""
    pixels = get_healpix_pixelation(res_healpix)
    alms = geo.get_alm_table(l_max)
    reconstruction = reconstruct_from_alm(l_max,pixels[:,0],pixels[:,1],alms)
    if fig is None:
        fig = plt.figure()
    if do_round:
        reconstruction = np.round(reconstruction)

    plot_reconstruction(reconstruction,title,fig,cmap,cbar=cbar,notext=notext)
    if display:
        plt.show(fig)
    return fig
