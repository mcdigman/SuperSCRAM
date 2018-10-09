"""compares gaussian covariance with cosmolike"""
from __future__ import absolute_import,division,print_function
from builtins import range
import numpy as np
from scipy.integrate import cumtrapz
import scipy.special as spp
import pytest
from cosmopie import CosmoPie
from shear_power import ShearPower,Cll_q_q
import defaults
from lensing_weight import QShear
import matter_power_spectrum as mps
from sw_survey import SWSurvey
from circle_geo import CircleGeo

def test_cosmolike_agreement():
    """test agreement with cosmolike"""
    base_dir = './'
    input_dir = base_dir+'test_inputs/cosmolike_1/'
    cosmo_results = np.loadtxt(input_dir+'cov_results_7.dat')
    cosmo_shear = np.loadtxt(input_dir+'shear_shear_cosmo_3.dat')
    cosmo_nz = np.loadtxt(input_dir+'n_z_2.dat')

    camb_params = defaults.camb_params
    camb_params['force_sigma8'] = True
    camb_params['leave_h'] = False
    camb_params['npoints'] = 1000
    camb_params['minkh'] = 1.1e-4
    camb_params['maxkh'] = 100.
    camb_params['kmax'] = 1.
    power_params = defaults.power_params.copy()
    power_params.camb = camb_params

    RTOL = 3.*10**-2
    ATOL = 10**-10

    cosmology_cosmolike = { 'Omegab'  :0.04868,#fixed
                            'Omegabh2':0.02204858372,#calculated
                            'Omegach2':0.12062405128,#calculated
                            'Omegamh2':0.142672635,
                            'OmegaL'  :0.685,#fixed
                            'OmegaLh2':0.310256,#calculated
                            'Omegam'  :.315, #fixed
                            'H0'      :67.3,
                            'sigma8'  :.829,#fixed
                            'h'       :0.673,#fixed
                            'Omegak'  :0.0,#fixed
                            'Omegakh2':0.0,
                            'Omegar'  :0.0,#fixed
                            'Omegarh2':0.0,
                            'ns'      :0.9603,#fixed
                            'tau'     :0.067,
                            'Yp'      :None,
                            'As'      :2.143*10**-9,
                            'LogAs'   :np.log(2.143*10**-9),
                            'w'       :-1.,
                            'de_model':'constant_w',
                            'mnu'     :0.
                          }

    C = CosmoPie(cosmology_cosmolike,p_space='basic')
    lmin_cosmo = 20
    lmax_cosmo = 5000
    n_b = 20
    area_cosmo = 1000. #deg^2
    fsky_cosmo = area_cosmo/41253. #41253 deg^2/sky
    n_gal_cosmo = 10.*(180**2/np.pi**2)*3600 #gal/rad^2
    sigma_e_cosmo = 0.27*np.sqrt(2)
    tomo_bins_cosmo = 4
#    amin_cosmo = 0.2

    n_s = sigma_e_cosmo**2/(2*n_gal_cosmo)
    print("n_s",n_s)
    #format input results
    lbin_cosmo_1 = cosmo_results[:,0]
#    lbin_cosmo_2 = cosmo_results[:,1]
#    lmid_cosmo_1 = cosmo_results[:,2]
#    lmid_cosmo_2 = cosmo_results[:,3]
    zbin_cosmo_1 = cosmo_results[:,4]
    zbin_cosmo_2 = cosmo_results[:,5]
    zbin_cosmo_3 = cosmo_results[:,6]
    zbin_cosmo_4 = cosmo_results[:,7]
    c_g_cosmo_in = cosmo_results[:,8]
    c_ssc_cosmo_in = cosmo_results[:,9]

    c_g_cosmo = np.zeros((tomo_bins_cosmo,tomo_bins_cosmo,tomo_bins_cosmo,tomo_bins_cosmo),dtype=object)
    c_ssc_cosmo = np.zeros((tomo_bins_cosmo,tomo_bins_cosmo,tomo_bins_cosmo,tomo_bins_cosmo),dtype=object)
    itr = 0
    while itr<lbin_cosmo_1.size:
        zs = np.array([int(zbin_cosmo_1[itr]),int(zbin_cosmo_2[itr]),int(zbin_cosmo_3[itr]),int(zbin_cosmo_4[itr])])
        loc_mat1 = np.zeros((n_b,n_b))
        loc_mat2 = np.zeros((n_b,n_b))
        for l1 in range(0,n_b):
            for l2 in range(0,n_b):
                loc_mat1[l1,l2] = c_g_cosmo_in[itr]
                loc_mat2[l1,l2] = c_ssc_cosmo_in[itr]
                itr+=1

        c_g_cosmo[zs[0],zs[1],zs[2],zs[3]] = loc_mat1
        c_ssc_cosmo[zs[0],zs[1],zs[2],zs[3]] = loc_mat2
    n_side = int(spp.binom(tomo_bins_cosmo+1,2))
    c_g_cosmo_flat = np.zeros((n_side*n_b,n_side*n_b))
    c_ssc_cosmo_flat = np.zeros((n_side*n_b,n_side*n_b))
    itr1 = 0
    #sanity check formatting of input results from cosmolike
    for i1 in range(tomo_bins_cosmo):
        for i2 in range(i1,tomo_bins_cosmo):
            itr2 = 0
            for i3 in range(tomo_bins_cosmo):
                for i4 in range(i3,tomo_bins_cosmo):
                    assert np.all(c_g_cosmo[i1,i2,i3,i4]==c_g_cosmo[i3,i4,i1,i2].T)
                    assert np.all(c_g_cosmo[i1,i2,i3,i4]==c_g_cosmo[i1,i2,i4,i3].T)
                    assert np.all(c_g_cosmo[i1,i2,i3,i4]==c_g_cosmo[i2,i1,i3,i4].T)
                    assert np.all(c_g_cosmo[i1,i2,i3,i4]==c_g_cosmo[i2,i1,i4,i3].T)
                    assert np.all(c_ssc_cosmo[i1,i2,i3,i4]==c_ssc_cosmo[i3,i4,i1,i2].T)
                    assert np.all(c_ssc_cosmo[i1,i2,i3,i4]==c_ssc_cosmo[i1,i2,i4,i3].T)
                    assert np.all(c_ssc_cosmo[i1,i2,i3,i4]==c_ssc_cosmo[i2,i1,i3,i4].T)
                    assert np.all(c_ssc_cosmo[i1,i2,i3,i4]==c_ssc_cosmo[i2,i1,i4,i3].T)
                    c_g_cosmo_flat[itr1:itr1+n_b,itr2:itr2+n_b] = c_g_cosmo[i1,i2,i3,i4]
                    c_ssc_cosmo_flat[itr1:itr1+n_b,itr2:itr2+n_b] = c_ssc_cosmo[i1,i2,i3,i4]
                    if itr2!=itr1:
                        c_g_cosmo_flat[itr2:itr2+n_b,itr1:itr1+n_b] = c_g_cosmo_flat[itr1:itr1+n_b,itr2:itr2+n_b].T
                        c_ssc_cosmo_flat[itr2:itr2+n_b,itr1:itr1+n_b] = c_ssc_cosmo_flat[itr1:itr1+n_b,itr2:itr2+n_b].T
                    itr2+=n_b
            itr1+=n_b
    assert np.all(c_g_cosmo_flat==c_g_cosmo_flat.T)
    assert np.all(c_ssc_cosmo_flat==c_ssc_cosmo_flat.T)

    Cll_sh_sh_cosmo = np.zeros((tomo_bins_cosmo,tomo_bins_cosmo),dtype=object)
    itr = 0
    for i in range(tomo_bins_cosmo):
        for j in range(i,tomo_bins_cosmo):
            Cll_sh_sh_cosmo[i,j] = cosmo_shear[:,itr+1]
            if not i==j:
                Cll_sh_sh_cosmo[j,i] = Cll_sh_sh_cosmo[i,j]
            itr += 1

    l_starts = np.zeros(n_b)
    log_dl = (np.log(lmax_cosmo)-np.log(lmin_cosmo))/n_b
    l_mids = np.zeros(n_b)
    dls = np.zeros(n_b)

    for i in range(l_mids.size):
        l_starts[i] = np.exp(np.log(lmin_cosmo)+i*log_dl)
        l_mids[i] = np.exp(np.log(lmin_cosmo)+(i+0.5)*log_dl)
        dls[i] = (np.exp(np.log(lmin_cosmo)+(i+1.)*log_dl)- np.exp(np.log(lmin_cosmo)+i*log_dl))


    z_fine = cosmo_nz[:,0]
    n_z = cosmo_nz[:,1]
    #prevent going to 0 badly
    z_fine[0] +=0.00001
    cum_n_z = cumtrapz(n_z,z_fine,initial=0.)
    cum_n_z = cum_n_z/cum_n_z[-1]
    z_bin_starts = np.zeros(tomo_bins_cosmo)
#    r_bins = np.zeros((tomo_bins_cosmo,2))
    for i in range(0,tomo_bins_cosmo):
        z_bin_starts[i] = np.min(z_fine[cum_n_z>=1./tomo_bins_cosmo*i])

    P_in = mps.MatterPower(C,power_params)
    k_in = P_in.k
    C.k = k_in
    C.P_lin = P_in
    #theta0 = np.pi/4.
    #theta1 = np.pi/2.
    #theta_in = np.pi/3.
    #phi0 = np.pi/4.
    #phi1 = np.pi/4.+np.pi/4.*0.5443397550644993
    #phi_in = np.pi/4.+0.01

    #thetas = np.array([theta0,theta1,theta1,theta0,theta0])
    #phis = np.array([phi0,phi0,phi1,phi1,phi0])
    z_coarse = np.hstack([z_bin_starts,2.])
    #geo1 = PolygonGeo(z_coarse,thetas,phis,theta_in,phi_in,C,z_fine.copy(),40,{'n_double':30})
    geo1 = CircleGeo(z_coarse,C,0.31275863997971481,100,z_fine.copy(),40,{'n_double':30})
    assert np.isclose((geo1.angular_area()*180**2/np.pi**2),1000.)
    r_bins = geo1.rbins
    z_bins = geo1.zbins


    len_params = defaults.lensing_params.copy()
    len_params['smodel'] = 'custom_z'
    len_params['n_gal'] = n_gal_cosmo
    len_params['sigma2_e'] = sigma_e_cosmo**2
    len_params['l_min'] = np.min(l_starts)
    len_params['l_max'] = np.max(l_starts)
    len_params['n_l'] = l_starts.size
    len_params['pmodel'] = 'halofit'
    #test lensing observables
    sp = ShearPower(C,z_fine,fsky_cosmo,len_params,mode='power',ps=n_z)
    qs = np.zeros(tomo_bins_cosmo,dtype=object)
    for i in range(qs.size):
        qs[i] = QShear(sp,z_bins[i,0],z_bins[i,1])
    Cll_sh_sh = np.zeros((tomo_bins_cosmo,tomo_bins_cosmo),dtype=object)
    ratio_means = np.zeros((tomo_bins_cosmo,tomo_bins_cosmo))
    for i in range(qs.size):
        for j in range(qs.size):
            Cll_sh_sh[i,j] = Cll_q_q(sp,qs[i],qs[j]).Cll()
            ratio_means[i,j] = np.average(Cll_sh_sh[i,j]/Cll_sh_sh_cosmo[i,j])
            assert np.allclose(Cll_sh_sh[i,j],Cll_sh_sh_cosmo[i,j],rtol=RTOL,atol=ATOL)

    print("average Cll ratio: ",np.average(ratio_means))
    print("mse Cll: ",np.linalg.norm(np.linalg.norm(1.-Cll_sh_sh/Cll_sh_sh_cosmo))/(Cll_sh_sh.size*Cll_sh_sh[0,0].size))

    c_g_flat = np.zeros_like(c_g_cosmo_flat)
    c_ssc_flat = np.zeros_like(c_g_cosmo_flat)

    #first test individually by manually building covariance matrix
    itr1 = 0
    for i1 in range(tomo_bins_cosmo):
        for i2 in range(i1,tomo_bins_cosmo):
            itr2 = 0
            for i3 in range(tomo_bins_cosmo):
                for i4 in range(i3,tomo_bins_cosmo):
                    ns = np.array([0.,0.,0.,0.])
                    if i1==i3:
                        ns[0] = n_s
                    if i1==i4:
                        ns[1] = n_s
                    if i2==i4:
                        ns[2] = n_s
                    if i2==i3:
                        ns[3] = n_s
                    qs_in = np.array([qs[i1],qs[i2],qs[i3],qs[i4]])
                    c_g_flat[itr1:itr1+n_b,itr2:itr2+n_b] = np.diagflat(sp.cov_g_diag(qs_in,ns))
                    #c_ssc_flat[itr1:itr1+n_b,itr2:itr2+n_b] = c_ssc[i1,i2,i3,i4]
                    if not itr2==itr1:
                        c_g_flat[itr2:itr2+n_b,itr1:itr1+n_b] = c_g_flat[itr1:itr1+n_b,itr2:itr2+n_b].T
                    #    c_ssc_flat[itr2:itr2+n_b,itr1:itr1+n_b] = c_ssc_flat[itr1:itr1+n_b,itr2:itr2+n_b].T
                    itr2+=n_b
            itr1+=n_b
    assert np.all(c_g_flat==c_g_flat.T)
    assert np.allclose(c_g_flat,c_g_cosmo_flat,atol=ATOL,rtol=RTOL)
    assert np.abs(1.-np.average(ratio_means))<0.01
    assert np.all(c_ssc_flat==c_ssc_flat.T)

    #test with SWSurvey pipeline with 1000 sq deg spherical rectangle geo
    sw_params = {'needs_lensing':True,'cross_bins':True}
    observable_list = np.array(['len_shear_shear'])
    sw_survey = SWSurvey(geo1,'c_s',C,sw_params,observable_list=observable_list,len_params=len_params,ps=n_z)
#    C_pow = sw_survey.len_pow.C_pow

    c_g_sw = sw_survey.get_non_SSC_sw_covar_arrays()[0]
    nonzero_mask = c_g_sw!=0.
    rat_c = c_g_cosmo_flat[nonzero_mask]/c_g_sw[nonzero_mask]
#    rat_m = c_g_flat[nonzero_mask]/c_g_sw[nonzero_mask]
    assert np.allclose(c_g_sw,c_g_flat)
    assert np.allclose(c_g_sw,c_g_cosmo_flat,atol=ATOL,rtol=RTOL)

    print("mean squared diff covariances:"+str(np.linalg.norm(1.-rat_c)/rat_c.size))


    print("PASS: all assertions passed")
#    do_plot = True
#    if do_plot:
#        import matplotlib.pyplot as plt
#        ax = plt.subplot(111)
#        bin1 = 0
#        bin2 = 0
#        bin3 = 1
#        bin4 = 0
#        ax.loglog(l_mids,np.diag(c_g_flat[0:n_b,0:n_b]))
#        ax.loglog(l_mids,np.diag(c_g_cosmo_flat[0:n_b,0:n_b]))
#        #ax.loglog(l_mids,np.diag(c_g[bin1,bin2,bin3,bin4]))
#        #ax.loglog(l_mids,np.diag(c_g_cosmo[bin1,bin2,bin3,bin4]))
#        #ax.loglog(l_mids,Cll_sh_sh[bin1,bin2])
#        #ax.loglog(l_mids,Cll_sh_sh_cosmo[bin1,bin2])
#        #print(Cll_sh_sh[bin1,bin2]/Cll_sh_sh_cosmo[bin1,bin2])
#        plt.show()

if __name__=='__main__':
    pytest.cmdline.main(['cosmolike_comparison_test.py'])
