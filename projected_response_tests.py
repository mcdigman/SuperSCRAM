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
    zs=np.array([0.1,0.4,0.7,1.0,1.3,1.6])

    ls = np.arange(2,3000)

    geo1=rect_geo(zs,Theta1,Phi1,C)
    
    omega_s = geo1.angular_area()
    
    z_fine = np.arange(0.1,2.0,0.01)
    len_params = defaults.lensing_params
    len_params['sigma'] = 0.1
    sp1 = sp.shear_power(k,C,z_fine,ls,omega_s=omega_s,pmodel='dc_halofit',P_in=P,params=len_params)
    import matplotlib.pyplot as plt
    ax = plt.subplot(111)
    plt.grid()
    for i in range(0,geo1.rbins.shape[0]):
        rbin = geo1.rbins[i]
        q1_dC = sp.q_shear(sp1,rbin[0],rbin[1])
        dc_ddelta = sp.Cll_q_q(sp1,q1_dC,q1_dC).Cll() 
        print dc_ddelta
        ax.loglog(ls,dc_ddelta)
    ax.legend(['1','2','3','4','5'])
    plt.show()
