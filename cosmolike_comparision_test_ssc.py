"""intended to provide comparison with cosmolike ssc term, currently broken"""
import numpy as np
from cosmopie import CosmoPie
from scipy.integrate import cumtrapz
import defaults
from camb_power import camb_pow
import scipy.special as spp
from sph_klim import SphBasisK
from polygon_geo import PolygonGeo
from sw_survey import SWSurvey
from lw_survey import LWSurvey
from Super_Survey import SuperSurvey

if __name__ == '__main__':
    base_dir = './'
    input_dir = base_dir+'test_inputs/cosmolike_1/'
    cosmo_results = np.loadtxt(input_dir+'cov_results_7.dat')
    cosmo_shear = np.loadtxt(input_dir+'shear_shear_cosmo_3.dat')
    cosmo_nz = np.loadtxt(input_dir+'n_z_2.dat')

    RTOL = 3.*10**-2
    ATOL = 10**-10
    lmin_cosmo = 20
    lmax_cosmo = 5000
    nbins_cosmo = 20
    area_cosmo = 1000 #deg^2
    fsky_cosmo = area_cosmo/41253. #41253 deg^2/sky
    n_gal_cosmo = 10.*(1./2.908882e-04)**2 #gal/deg^2
    sigma_e_cosmo = 0.27*np.sqrt(2)
    tomo_bins_cosmo = 4
    amin_cosmo = 0.2

    len_params = defaults.lensing_params.copy()
    len_params['smodel'] = 'custom_z'
    len_params['n_gal'] = n_gal_cosmo
    len_params['sigma2_e'] = sigma_e_cosmo**2

    camb_params = defaults.camb_params
    camb_params['force_sigma8']=True
    k_in,P_in=camb_pow(defaults.cosmology_cosmolike,camb_params=camb_params)
    C=CosmoPie(defaults.cosmology_cosmolike,P_lin=P_in,k=k_in,p_space='overwride')


    n_s = sigma_e_cosmo**2/(2*n_gal_cosmo)
    print n_s
    #n_s = 0.
    lbin_cosmo_1 = cosmo_results[:,0]
    lbin_cosmo_2 = cosmo_results[:,1]
    lmid_cosmo_1 = cosmo_results[:,2]
    lmid_cosmo_2 = cosmo_results[:,3]
    zbin_cosmo_1 = cosmo_results[:,4]
    zbin_cosmo_2 = cosmo_results[:,5]
    zbin_cosmo_3 = cosmo_results[:,6]
    zbin_cosmo_4 = cosmo_results[:,7]
    cov_g_cosmo = cosmo_results[:,8]
    cov_ssc_cosmo = cosmo_results[:,9]

    cov_g_mat_cosmo = np.zeros((tomo_bins_cosmo,tomo_bins_cosmo,tomo_bins_cosmo,tomo_bins_cosmo),dtype=object)
    cov_ssc_mat_cosmo = np.zeros((tomo_bins_cosmo,tomo_bins_cosmo,tomo_bins_cosmo,tomo_bins_cosmo),dtype=object)
    itr = 0
    while itr<lbin_cosmo_1.size:
        zs = np.array([int(zbin_cosmo_1[itr]),int(zbin_cosmo_2[itr]),int(zbin_cosmo_3[itr]),int(zbin_cosmo_4[itr])])
        loc_mat1 = np.zeros((nbins_cosmo,nbins_cosmo))
        loc_mat2 = np.zeros((nbins_cosmo,nbins_cosmo))
        for l1 in xrange(0,nbins_cosmo):
            for l2 in xrange(0,nbins_cosmo):
                loc_mat1[l1,l2] = cov_g_cosmo[itr]
                loc_mat2[l1,l2] = cov_ssc_cosmo[itr]
                itr+=1

        cov_g_mat_cosmo[zs[0],zs[1],zs[2],zs[3]] = loc_mat1
        cov_ssc_mat_cosmo[zs[0],zs[1],zs[2],zs[3]] = loc_mat2
        #cov_g_mat_cosmo[zs[0],zs[1],zs[3],zs[2]] = loc_mat
        #cov_g_mat_cosmo[zs[1],zs[0],zs[2],zs[3]] = loc_mat
        #cov_g_mat_cosmo[zs[0],zs[0],zs[3],zs[2]] = loc_mat
        #cov_g_mat_cosmo[zs[2],zs[3],zs[0],zs[1]] = loc_mat
        #cov_g_mat_cosmo[zs[3],zs[2],zs[0],zs[1]] = loc_mat
        #cov_g_mat_cosmo[zs[2],zs[3],zs[1],zs[0]] = loc_mat
        #cov_g_mat_cosmo[zs[3],zs[2],zs[1],zs[0]] = loc_mat
    n_side = int(spp.binom(tomo_bins_cosmo+1,2))
    cov_g_mat_cosmo_flat = np.zeros((n_side*nbins_cosmo,n_side*nbins_cosmo))
    cov_ssc_mat_cosmo_flat = np.zeros((n_side*nbins_cosmo,n_side*nbins_cosmo))
    itr_out = 0
    for i1 in xrange(tomo_bins_cosmo):
        for i2 in xrange(i1,tomo_bins_cosmo):
            itr_in = 0
            for i3 in xrange(tomo_bins_cosmo):
                for i4 in xrange(i3,tomo_bins_cosmo):
                    assert np.all(cov_g_mat_cosmo[i1,i2,i3,i4]==cov_g_mat_cosmo[i3,i4,i1,i2].T)
                    assert np.all(cov_g_mat_cosmo[i1,i2,i3,i4]==cov_g_mat_cosmo[i1,i2,i4,i3].T)
                    assert np.all(cov_g_mat_cosmo[i1,i2,i3,i4]==cov_g_mat_cosmo[i2,i1,i3,i4].T)
                    assert np.all(cov_g_mat_cosmo[i1,i2,i3,i4]==cov_g_mat_cosmo[i2,i1,i4,i3].T)
                    assert np.all(cov_ssc_mat_cosmo[i1,i2,i3,i4]==cov_ssc_mat_cosmo[i3,i4,i1,i2].T)
                    assert np.all(cov_ssc_mat_cosmo[i1,i2,i3,i4]==cov_ssc_mat_cosmo[i1,i2,i4,i3].T)
                    assert np.all(cov_ssc_mat_cosmo[i1,i2,i3,i4]==cov_ssc_mat_cosmo[i2,i1,i3,i4].T)
                    assert np.all(cov_ssc_mat_cosmo[i1,i2,i3,i4]==cov_ssc_mat_cosmo[i2,i1,i4,i3].T)
                    cov_g_mat_cosmo_flat[itr_out:itr_out+nbins_cosmo,itr_in:itr_in+nbins_cosmo] = cov_g_mat_cosmo[i1,i2,i3,i4]
                    cov_ssc_mat_cosmo_flat[itr_out:itr_out+nbins_cosmo,itr_in:itr_in+nbins_cosmo] = cov_ssc_mat_cosmo[i1,i2,i3,i4]
                    if itr_in!=itr_out:
                        cov_g_mat_cosmo_flat[itr_in:itr_in+nbins_cosmo,itr_out:itr_out+nbins_cosmo] = cov_g_mat_cosmo_flat[itr_out:itr_out+nbins_cosmo,itr_in:itr_in+nbins_cosmo].T
                        cov_ssc_mat_cosmo_flat[itr_in:itr_in+nbins_cosmo,itr_out:itr_out+nbins_cosmo] = cov_ssc_mat_cosmo_flat[itr_out:itr_out+nbins_cosmo,itr_in:itr_in+nbins_cosmo].T
                    itr_in+=nbins_cosmo
            itr_out+=nbins_cosmo
    assert np.all(cov_g_mat_cosmo_flat==cov_g_mat_cosmo_flat.T)
    assert np.all(cov_ssc_mat_cosmo_flat==cov_ssc_mat_cosmo_flat.T)
                    #if i1<=i2 and i3<=i4
#                    cov_g_mat_cosmo[i1,i2,i3,i4] = np.zeros((nbins_cosmo,nbins_cosmo))
#                    for l1 in xrange(nbins_cosmo):
#                        for l2 in xrange(nbins_cosmo):



    Cll_shear_shear_cosmo = np.zeros((tomo_bins_cosmo,tomo_bins_cosmo),dtype=object)
    itr = 0
    for i in xrange(tomo_bins_cosmo):
        for j in xrange(i,tomo_bins_cosmo):
            #print i,j
            Cll_shear_shear_cosmo[i,j] =cosmo_shear[:,itr+1]
            if not i==j:
                Cll_shear_shear_cosmo[j,i] = Cll_shear_shear_cosmo[i,j]
            itr += 1

    l_starts = np.zeros(nbins_cosmo)
    log_dl = (np.log(lmax_cosmo)-np.log(lmin_cosmo))/nbins_cosmo
    l_mids = np.zeros(nbins_cosmo)
    dls = np.zeros(nbins_cosmo)

    for i in xrange(l_mids.size):
        l_starts[i] = np.exp(np.log(lmin_cosmo)+i*log_dl)
        l_mids[i] = np.exp(np.log(lmin_cosmo)+(i+0.5)*log_dl)
        dls[i] = (np.exp(np.log(lmin_cosmo)+(i+1.)*log_dl)- np.exp(np.log(lmin_cosmo)+i*log_dl))


    z_fine = cosmo_nz[:,0]
    n_z = cosmo_nz[:,1]
    cum_n_z = cumtrapz(n_z,z_fine,initial=0.)
    z_bin_starts = np.zeros(tomo_bins_cosmo)
    chi_bins = np.zeros((tomo_bins_cosmo,2))
    z_fine[0] +=0.00001
    for i in xrange(0,tomo_bins_cosmo):
        z_bin_starts[i] = np.min(z_fine[cum_n_z>=1./tomo_bins_cosmo*i])

    for i in xrange(0,tomo_bins_cosmo):
        if i==tomo_bins_cosmo-1:
            chi_next = C.D_comov(np.max(z_fine))
        else:
            chi_next = C.D_comov(z_bin_starts[i+1])
        chi_bins[i] = np.array([C.D_comov(z_bin_starts[i]),chi_next])

    r_max=C.D_comov(np.max(z_fine))
    #theta0=0.
    #theta1=np.pi/2.
    #phi0=0.
    #phi1=5.*np.pi**2/162. #gives exactly a 1000 square degree field of view
    #phi2=2.*np.pi/3.
    #phi3=phi2+(phi1-phi0)
    #theta1s = np.array([theta0,theta1,theta1,theta0,theta0])
    #phi1s = np.array([phi0,phi0,phi1,phi1,phi0])
    #theta_in1 = np.pi/8.
    #phi_in1 = 4.*np.pi**2/162.
    #theta2s = np.array([theta0,theta1,theta1,theta0,theta0])
    #phi2s = np.array([phi2,phi2,phi3,phi3,phi2])
    #theta_in2 = np.pi/8.
    #phi_in2 = phi2+(phi_in1-phi0)
    #theta0=(1.25814+np.pi/2.)
    #approximate circular 1000 square degree geo
    theta0=np.pi-np.arccos(1.-5.*np.pi/324.)
    phi0 = 0.
    #theta1s = np.array([theta0,theta0,theta0,theta0,theta0,theta0,theta0,theta0])
    #phi1s = np.array([phi0,phi0+np.pi/3.,phi0+2.*np.pi/3.,phi0+np.pi,phi0+4.*np.pi/3.,phi0+5.*np.pi/3.,6.*np.pi/3.,phi0])[::-1]
    #theta2s = np.array([theta0,theta0,theta0,theta0,theta0,theta0,theta0,theta0])
    #phi2s = np.array([phi0,phi0+np.pi/3.,phi0+2.*np.pi/3.,phi0+np.pi,phi0+4.*np.pi/3.,phi0+5.*np.pi/3.,6.*np.pi/3.,phi0])[::-1]
    n_steps = 120
    theta1s = np.zeros(n_steps+2)+theta0
    theta2s = theta1s
    phi1s = np.zeros(n_steps+2)
    phi1s[0] = phi0
    phi1s[-1] = phi0
    for itr in xrange(1,n_steps+1):
        phi1s[itr] = phi0+itr*2.*np.pi/n_steps
    phi1s = phi1s[::-1]
    phi2s = phi1s
    theta_in1 = theta0+0.1
    phi_in1 = phi0-0.01
    theta_in2 = theta0+0.1
    phi_in2 = phi0-0.01

    zs=np.hstack([z_bin_starts,np.max(z_fine)])
    #d=np.loadtxt('camb_m_pow_l.dat')
    #k_in=d[:,0]; P_in=d[:,1]
    l_max = 50
    geo1 = PolygonGeo(zs,theta1s,phi1s,C,z_fine,l_max,defaults.polygon_params)
    geo2 = PolygonGeo(zs,theta2s,phi2s,C,z_fine,l_max,defaults.polygon_params)

    lenless_defaults = defaults.sw_survey_params.copy()
    lenless_defaults['needs_lensing'] = False

    survey_1 = SWSurvey(geo1,'survey1',C,l_mids,defaults.sw_survey_params,observable_list = defaults.sw_observable_list,cosmo_param_list = np.array([],dtype=object),cosmo_param_epsilons=np.array([]),len_params=len_params,ps=n_z)
    survey_2 = SWSurvey(geo1,'survey2',C,l_mids,lenless_defaults,observable_list = np.array([]),cosmo_param_list = np.array([],dtype=object),cosmo_param_epsilons=np.array([]),len_params=len_params,ps=n_z)



    M_cut=10**(12.5)


    surveys_sw=np.array([survey_1])


    geos = np.array([geo1,geo2])
    l_lw=np.arange(0,30)
    k_cut = 0.009
    n_zeros=49

    basis=SphBasisK(r_max,C,k_cut,defaults.basis_params,l_ceil=100)

    survey_3 = LWSurvey(geos,'lw_survey1',basis,C=C,ls = l_lw,params=defaults.lw_survey_params,observable_list=defaults.lw_observable_list,dn_params=defaults.dn_params)
    surveys_lw=np.array([survey_3])

    SS=SuperSurvey(surveys_sw, surveys_lw,r_max,l_mids,n_zeros,k_in,basis,P_lin=P_in,C=C,get_a=False,do_unmitigated=True,do_mitigated=False)
