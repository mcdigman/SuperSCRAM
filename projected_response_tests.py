import numpy as np
import shear_power as sp
from geo import rect_geo
import defaults
from cosmopie import CosmoPie

if __name__=='__main__':

    d=np.loadtxt('camb_m_pow_l.dat')
    k=d[:,0]; P=d[:,1]

    C=CosmoPie(k=k,P_lin=P,cosmology=defaults.cosmology)

    Theta1=[np.pi/4.,5.*np.pi/16.]
    Phi1=[0.,np.pi/12.]
    zs=np.array([.9,1.0])
    ls = np.arange(2,3000)

    geo1=rect_geo(zs,Theta1,Phi1,C)
    
    omega_s = geo1.angular_area()
    
    sp1 = sp.shear_power(k,C,zs,ls,omega_s=omega_s,pmodel='dc_halofit',P_in=P)
    
