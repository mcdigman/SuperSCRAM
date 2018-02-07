"""approximate a circular geometry with PolygonGeo"""
import numpy as np
from polygon_geo import PolygonGeo
#exact radius for 1000 deg^2 r=np.arccos(1.-5.*np.pi/324.)=0.3126603700269391 as n_x->infinity
#if n_x=100 for 1000 deg^2 r=0.31275863997971481
class CircleGeo(PolygonGeo):
    """approximately circular geometry centered at (0,0) approximated as PolygonGeo"""
    def __init__(self,zs,C,radius,n_x,z_fine,l_max,poly_params):
        """radius r in radians,n_x steps to approximate (larger=>better approximation),
        rest of arguments as in PolygonGeo"""
        self.radius=radius
        self.n_x = n_x
        theta_in = 0.
        phi_in = 0.
        thetas = np.full(self.n_x,self.radius)
        phis = np.linspace(0.,2.*np.pi,self.n_x)
        thetas[-1] = thetas[0]
        phis[-1] = phis[0]

        PolygonGeo.__init__(self,zs,thetas,phis,theta_in,phi_in,C,z_fine,l_max,poly_params)
