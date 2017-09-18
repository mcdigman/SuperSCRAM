import numpy as np
from shear_power import ShearPower,Cll_q_q
from cosmopie import CosmoPie
import defaults
from scipy.interpolate import SmoothBivariateSpline,interp2d
from lensing_weight import QShear
import sys

if __name__=='__main__':
    omega_s = 0.02
    ls = np.arange(2,3000)

    d = np.loadtxt('camb_m_pow_l.dat')
    k_in = d[:,0]
    P_in = d[:,1]



    C = CosmoPie(defaults.cosmology)
    C.k = k_in
    C.P_lin = P_in
    len_params = defaults.lensing_params.copy()
    len_params['z_bar']=1.0
    len_params['sigma']=0.4
   
    z_test_res1 = 0.001
    #z_test_res2 = 0.01
    zs_test1 = np.arange(len_params['z_min_integral'],len_params['z_max_integral'],z_test_res1)
    #zs_test2 = np.arange(len_params['z_min_integral'],len_params['z_max_integral'],z_test_res2)

    dC_ddelta1 = ShearPower(C,zs_test1,ls,omega_s=omega_s,pmodel=len_params['pmodel_dO_ddelta'],mode='dc_ddelta')
    #dC_ddelta2 = ShearPower(k_in,C,zs_test2,ls,omega_s=omega_s,pmodel=len_params['pmodel_dO_ddelta'])
    
    sp1 = ShearPower(C,zs_test1,ls,omega_s=omega_s,pmodel=len_params['pmodel_O'],mode='power')
   # sp2 = ShearPower(k_in,C,zs_test2,ls,omega_s=omega_s,pmodel=len_params['pmodel_O'])

    
    z_min1 = 0.8
    z_max1 = 1.0
    chi_min1 = C.D_comov(z_min1)
    chi_max1 = C.D_comov(z_max1)

    z_min2 = 1.6
    z_max2 = 1.8
    chi_min2 = C.D_comov(z_min2)
    chi_max2 = C.D_comov(z_max2)

    QShear1_1 = QShear(dC_ddelta1,chi_min1,chi_max1)
    QShear1_2 = QShear(dC_ddelta1,chi_min2,chi_max2)

    #QShear2_1 = QShear(dC_ddelta2,chi_min1,chi_max1)
    #QShear2_2 = QShear(dC_ddelta2,chi_min2,chi_max2)

    ss_1 = Cll_q_q(sp1,QShear1_1,QShear1_2).Cll()
    #check that \partial C^{ij}/\partial \bar{\delta}(zs) \propto 1/(width of zs bin)*z^i*C^{ij}, where z^i ~ average z of closer z bin
    import matplotlib.pyplot as plt
    for z_ind in xrange(5,100,10):
        ind_min = 200
        ind_max = ind_min+z_ind
        dC_ss_integrand_1 = Cll_q_q(dC_ddelta1,QShear1_1,QShear1_2).Cll(chi_min=dC_ddelta1.chis[ind_min],chi_max=dC_ddelta1.chis[ind_max])
        ax = plt.subplot(111)
        ax.loglog(ls,dC_ss_integrand_1/ss_1/(dC_ddelta1.zs[ind_max]-dC_ddelta1.zs[ind_min]))
        print np.average(dC_ss_integrand_1/ss_1/(dC_ddelta1.zs[ind_max]-dC_ddelta1.zs[ind_min])*(z_min1+z_max1)/2.)
        #dC_ss_1 = Cll_q_q(dC_ddelta1,QShear1_1,QShear1_2).Cll()

    #dC_ss_integrand_2 = Cll_q_q(dC_ddelta2,QShear2_1,QShear2_2).Cll_integrand()

    #dz_1s = np.hstack((dC_ddelta1.zs[0],np.diff(dC_ddelta1.zs)))/np.hstack((dC_ddelta1.chis[0],np.diff(dC_ddelta1.chis)))
    #dz_2s = np.hstack((dC_ddelta2.zs[0],np.diff(dC_ddelta2.zs)))/np.hstack((dC_ddelta2.chis[0],np.diff(dC_ddelta2.chis)))

    #ss_2 = Cll_q_q(sp2,QShear2_1,QShear2_2).Cll()
 
    plt.show()

    sys.exit()
    for z_val in np.arange(0.1,1.3,0.1):
        zdiff1 = np.abs(zs_test1-z_val)
       # zdiff2 = np.abs(zs_test2-z_val)
        z_ind1 = np.argmin(zdiff1)
       # z_ind2 = np.argmin(zdiff2)
    
        test_rat1 = dC_ss_integrand_1[z_ind1]/ss_1*(z_min1+z_max1)/2.
       # test_rat2 = dC_ss_integrand_2[z_ind2]/ss_2
        print z_ind1
        print dC_ss_integrand_1[z_ind1]
        ax = plt.subplot(111)
        ax.loglog(ls,test_rat1)
      #  ax.loglog(ls,test_rat2)
    plt.show()
