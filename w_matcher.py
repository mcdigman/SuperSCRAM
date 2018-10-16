"""class to match 'effective' w and growth factors in order to emulate arbitrary w(z) behavior"""
from __future__ import division,print_function,absolute_import
from builtins import range
from warnings import warn

from scipy.interpolate import InterpolatedUnivariateSpline,RectBivariateSpline
from scipy.integrate import cumtrapz
import numpy as np

import cosmopie as cp
class WMatcher(object):
    """matcher object for emulating arbitrary w(z) with effective constant w and growth factor"""
    def __init__(self,C_fid,wmatcher_params):
        """ C_fid: the fiducial CosmoPie, w(z) irrelevant because it will be ignored
            wmatcher_params:
                w_step: resolution of w grid
                w_min: minimum w to consider
                w_max: maximum w to consider
                a_step: resolution of a grid
                a_min: minimum a to consider
                a_max: maximum a to consider
        """
        #appears only use of C_fid to extract cosmology
        #CosmoPie appears to be used to extract G_norm,G, and Ez
        self.C_fid = C_fid
        self.cosmo_fid = self.C_fid.cosmology.copy()
        self.cosmo_fid['w'] = -1.
        self.cosmo_fid['de_model'] = 'constant_w'

        self.w_step = wmatcher_params['w_step']
        self.w_min = wmatcher_params['w_min']
        self.w_max = wmatcher_params['w_max']
        self.ws = np.arange(self.w_min,self.w_max,self.w_step)
        self.n_w = self.ws.size

        self.a_step = wmatcher_params['a_step']
        self.a_min = wmatcher_params['a_min']
        self.a_max = wmatcher_params['a_max']
        self.a_s = np.arange(self.a_max,self.a_min-self.a_step/10.,-self.a_step)
        self.n_a = self.a_s.size

        self.zs = 1./self.a_s-1.
        self.cosmos = np.zeros(self.n_w,dtype=object)

        self.integ_Es = np.zeros((self.n_w,self.n_a))
        self.Gs = np.zeros((self.n_w,self.n_a))


        for i in range(0,self.n_w):
            self.cosmos[i] = self.cosmo_fid.copy()
            self.cosmos[i]['w'] = self.ws[i]
            C_i = cp.CosmoPie(cosmology=self.cosmos[i],p_space=self.cosmo_fid['p_space'],silent=True)
            E_as = C_i.Ez(self.zs)
            self.integ_Es[i] = cumtrapz(1./(self.a_s**2*E_as)[::-1],self.a_s[::-1],initial=0.)
            self.Gs[i] = C_i.G(self.zs)
        self.G_interp = RectBivariateSpline(self.ws,self.a_s[::-1],self.Gs[:,::-1],kx=3,ky=3)

        self.ind_switches = np.argmax(np.diff(self.integ_Es,axis=0)<0,axis=0)+1
        #there is a purely numerical issue that causes the integral to be non-monotonic, this loop eliminates the spurious behavior
        for i in range(1,self.n_a):
            if self.ind_switches[i]>1:
                if self.integ_Es[self.ind_switches[i]-1,i]-self.integ_Es[0,i]>=0:
                    self.integ_Es[0:(self.ind_switches[i]-1),i] = self.integ_Es[self.ind_switches[i]-1,i]
                else:
                    raise RuntimeError( "Nonmonotonic integral, solution is not unique at "+str(self.a_s[i]))

        self.integ_E_interp = RectBivariateSpline(self.ws,self.a_s[::-1],self.integ_Es,kx=3,ky=3)

    #accurate to within numerical precision
    #could reduce reliance on padding
    def match_w(self,C_in,z_match,n_pad=3):
        """ match effective constant w as in casarini paper
            require some padding so can get very accurate interpolation results, 2 works 3 is better"""
        z_match = np.asanyarray(z_match)
        a_match = 1./(1.+z_match)
        E_in = C_in.Ez(self.zs)
        integ_E_in = cumtrapz(1./(self.a_s**2*E_in)[::-1],self.a_s[::-1],initial=0.)
        integ_E_in_interp = InterpolatedUnivariateSpline(self.a_s[::-1],integ_E_in,k=3,ext=2)
        integ_E_targets = integ_E_in_interp(a_match)
        w_grid1 = np.zeros(a_match.size)
        for itr in range(0,z_match.size):
            iE_vals = self.integ_E_interp(self.ws,a_match[itr]).T[0]-integ_E_targets[itr]
            iG = np.argmax(iE_vals<=0.)
            if iG-n_pad>=0 and iG+n_pad<self.ws.size:
                w_grid1[itr] = InterpolatedUnivariateSpline(iE_vals[iG-n_pad:iG+n_pad][::-1],self.ws[iG-n_pad:iG+n_pad:][::-1],k=2*n_pad-1,ext=2)(0.)
            else:
                warn("w is too close to edge of range, using nearest neighbor w, consider expanding w range")
                w_grid1[itr] = self.ws[iG]
        return w_grid1

    def match_growth(self,C_in,z_in,w_in):
        """matches the redshift dependence of the growth factor for the input model to the equivalent model with constant w"""
        a_in = 1./(1.+z_in)
        G_norm_ins = C_in.G_norm(z_in)
        n_z_in = z_in.size
        pow_mult = np.zeros(n_z_in)
        for itr in range(0,n_z_in):
            pow_mult[itr] = (G_norm_ins[itr]/(self.G_interp(w_in[itr],a_in[itr])/self.G_interp(w_in[itr],1.)))**2
        #return multiplier for linear power spectrum from effective constant w model
        return pow_mult

    def growth_interp(self,w_in,a_in):
        """get an interpolated growth factor for a given w, a_in is a vector"""
        return self.G_interp(w_in,a_in,grid=False).T

#    def match_scale(self,z_in,w_in):
#        """match scaling (ie sigma8) for the input model compared to the fiducial model, not used"""
#        n_z_in = z_in.size
#        pow_scale = np.zeros(n_z_in)
#        G_fid = self.C_fid.G(0)
#        for itr in range(0,n_z_in):
#            pow_scale[itr]=(self.G_interp(w_in,1./(1.+z_in))/G_fid)**2
#        #return multiplier for linear power spectrum from effective constant w model
#        return pow_scale
