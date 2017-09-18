import numpy as np
from cosmopie import CosmoPie
from scipy.integrate import cumtrapz,trapz
from shear_power import ShearPower
from shear_power import Cll_q_q
import defaults
from lensing_weight import QShear
from camb_power import camb_pow
import scipy.special as spp
import matter_power_spectrum as mps

if __name__ == '__main__':
    base_dir = './'
    input_dir = base_dir+'test_inputs/cosmolike_1/'
    cosmo_results = np.loadtxt(input_dir+'cov_results_7.dat')
    cosmo_shear = np.loadtxt(input_dir+'shear_shear_cosmo_3.dat')
    cosmo_nz = np.loadtxt(input_dir+'n_z_2.dat')

    RTOL = 3.*10**-2
    ATOL = 10**-10
    C=CosmoPie(defaults.cosmology_cosmolike)
    lmin_cosmo = 20
    lmax_cosmo = 5000
    nbins_cosmo = 20
    area_cosmo = 1000 #deg^2
    fsky_cosmo = area_cosmo/41253. #41253 deg^2/sky
    n_gal_cosmo = 10.*(1./2.908882e-04)**2 #gal/deg^2
    sigma_e_cosmo = 0.27*np.sqrt(2)
    tomo_bins_cosmo = 4
    amin_cosmo = 0.2

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
                    assert(np.all(cov_g_mat_cosmo[i1,i2,i3,i4]==cov_g_mat_cosmo[i3,i4,i1,i2].T))
                    assert(np.all(cov_g_mat_cosmo[i1,i2,i3,i4]==cov_g_mat_cosmo[i1,i2,i4,i3].T))
                    assert(np.all(cov_g_mat_cosmo[i1,i2,i3,i4]==cov_g_mat_cosmo[i2,i1,i3,i4].T))
                    assert(np.all(cov_g_mat_cosmo[i1,i2,i3,i4]==cov_g_mat_cosmo[i2,i1,i4,i3].T))
                    assert(np.all(cov_ssc_mat_cosmo[i1,i2,i3,i4]==cov_ssc_mat_cosmo[i3,i4,i1,i2].T))
                    assert(np.all(cov_ssc_mat_cosmo[i1,i2,i3,i4]==cov_ssc_mat_cosmo[i1,i2,i4,i3].T))
                    assert(np.all(cov_ssc_mat_cosmo[i1,i2,i3,i4]==cov_ssc_mat_cosmo[i2,i1,i3,i4].T))
                    assert(np.all(cov_ssc_mat_cosmo[i1,i2,i3,i4]==cov_ssc_mat_cosmo[i2,i1,i4,i3].T))
                    cov_g_mat_cosmo_flat[itr_out:itr_out+nbins_cosmo,itr_in:itr_in+nbins_cosmo] = cov_g_mat_cosmo[i1,i2,i3,i4]
                    cov_ssc_mat_cosmo_flat[itr_out:itr_out+nbins_cosmo,itr_in:itr_in+nbins_cosmo] = cov_ssc_mat_cosmo[i1,i2,i3,i4]
                    if not itr_in==itr_out:
                        cov_g_mat_cosmo_flat[itr_in:itr_in+nbins_cosmo,itr_out:itr_out+nbins_cosmo] = cov_g_mat_cosmo_flat[itr_out:itr_out+nbins_cosmo,itr_in:itr_in+nbins_cosmo].T
                        cov_ssc_mat_cosmo_flat[itr_in:itr_in+nbins_cosmo,itr_out:itr_out+nbins_cosmo] = cov_ssc_mat_cosmo_flat[itr_out:itr_out+nbins_cosmo,itr_in:itr_in+nbins_cosmo].T
                    itr_in+=nbins_cosmo
            itr_out+=nbins_cosmo
    assert(np.all(cov_g_mat_cosmo_flat==cov_g_mat_cosmo_flat.T))
    assert(np.all(cov_ssc_mat_cosmo_flat==cov_ssc_mat_cosmo_flat.T))
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

    #d=np.loadtxt('camb_m_pow_l.dat')
    #k_in=d[:,0]; P_in=d[:,1]
    camb_params = defaults.camb_params
    camb_params['force_sigma8']=True
    camb_params['leave_h'] =False
    camb_params['npoints']=1000
    camb_params['minkh']=1.1e-4
    camb_params['maxkh']=100.
    camb_params['kmax']=1.
    #k_in,P_in=camb_pow(defaults.cosmology_cosmolike,camb_params=camb_params)
    P_in = mps.MatterPower(C)
    k_in =P_in.k
    C.k = k_in
    C.P_lin = P_in

    import matplotlib.pyplot as plt
    len_params = defaults.lensing_params.copy()
    len_params['smodel'] = 'custom_z' 
    len_params['n_gal'] = n_gal_cosmo
    len_params['sigma2_e'] = sigma_e_cosmo**2
    
    sp = ShearPower(C,z_fine,l_mids,fsky_cosmo,pmodel='halofit',mode='power',ps=n_z,params=len_params)
    qs = np.zeros(tomo_bins_cosmo,dtype=object)
    for i in xrange(qs.size):
        qs[i] = QShear(sp,chi_bins[i,0],chi_bins[i,1])
    Cll_shear_shear = np.zeros((tomo_bins_cosmo,tomo_bins_cosmo),dtype=object)
    ratio_means = np.zeros((tomo_bins_cosmo,tomo_bins_cosmo))
    for i in xrange(qs.size):
        for j in xrange(qs.size):
            Cll_shear_shear[i,j] = Cll_q_q(sp,qs[i],qs[j]).Cll()
            ratio_means[i,j] = np.average(Cll_shear_shear[i,j]/Cll_shear_shear_cosmo[i,j])
            assert(np.allclose(Cll_shear_shear[i,j],Cll_shear_shear_cosmo[i,j],rtol=RTOL,atol=ATOL))
    #print ratio_means
    print np.average(ratio_means)


    #for i in xrange(dls.size):
        #print "l,dl,frac",l_mids[i],dls[i],1./(fsky_cosmo*(2.*l_mids[i]+1.)*dls[i])

    cov_g_mat_flat = np.zeros_like(cov_g_mat_cosmo_flat)
    cov_ssc_mat_flat = np.zeros_like(cov_g_mat_cosmo_flat)

    itr_out = 0
    for i1 in xrange(tomo_bins_cosmo):
        for i2 in xrange(i1,tomo_bins_cosmo):
            itr_in = 0
            for i3 in xrange(tomo_bins_cosmo):
                for i4 in xrange(i3,tomo_bins_cosmo):
                    ns = np.array([0.,0.,0.,0.])
                    if i1==i3:
                        ns[0] = n_s#/trapz(sp.ps*(sp.chis>=chi_bins[z1,0])*(sp.chis<=chi_bins[z1,1])*1.,sp.chis)
                    if i1==i4:
                        ns[1] = n_s#/trapz(sp.ps*(sp.chis>=chi_bins[z1,0])*(sp.chis<=chi_bins[z1,1])*1.,sp.chis)
                    if i2==i4:
                        ns[2] = n_s#/trapz(sp.ps*(sp.chis>=chi_bins[z2,0])*(sp.chis<=chi_bins[z2,1])*1.,sp.chis)
                    if i2==i3:
                        ns[3] = n_s#/trapz(sp.ps*(sp.chis>=chi_bins[z2,0])*(sp.chis<=chi_bins[z2,1])*1.,sp.chis)
                    cov_g_mat_flat[itr_out:itr_out+nbins_cosmo,itr_in:itr_in+nbins_cosmo] = np.diagflat(sp.cov_g_diag(np.array([qs[i1],qs[i2],qs[i3],qs[i4]]),ns,delta_ls=dls,ls=l_mids))
                    #cov_ssc_mat_flat[itr_out:itr_out+nbins_cosmo,itr_in:itr_in+nbins_cosmo] = cov_ssc_mat[i1,i2,i3,i4]
                    if not itr_in==itr_out:
                        cov_g_mat_flat[itr_in:itr_in+nbins_cosmo,itr_out:itr_out+nbins_cosmo] = cov_g_mat_flat[itr_out:itr_out+nbins_cosmo,itr_in:itr_in+nbins_cosmo].T
                    #    cov_ssc_mat_flat[itr_in:itr_in+nbins_cosmo,itr_out:itr_out+nbins_cosmo] = cov_ssc_mat_flat[itr_out:itr_out+nbins_cosmo,itr_in:itr_in+nbins_cosmo].T
                    itr_in+=nbins_cosmo
            itr_out+=nbins_cosmo
    assert(np.all(cov_g_mat_flat==cov_g_mat_flat.T))
    assert(np.allclose(cov_g_mat_flat,cov_g_mat_cosmo_flat,atol=ATOL,rtol=RTOL))
    #assert(np.all(cov_ssc_mat_flat==cov_ssc_mat_flat.T))
   
#    cov_g_mat = np.zeros((tomo_bins_cosmo,tomo_bins_cosmo,tomo_bins_cosmo,tomo_bins_cosmo),dtype=object)
#    cov_g_rat_means = np.zeros_like(cov_g_mat)
#    for z1 in xrange(0,tomo_bins_cosmo): 
#        print "z1 adj trap",z1,n_s/trapz(sp.ps*(sp.chis>=chi_bins[z1,0])*(sp.chis<=chi_bins[z1,1])*1.,sp.chis),trapz(sp.ps*(sp.chis>=chi_bins[z1,0])*(sp.chis<=chi_bins[z1,1])*1.,sp.chis)
#        for z2 in xrange(0,tomo_bins_cosmo): 
#            for z3 in xrange(0,tomo_bins_cosmo): 
#                for z4 in xrange(0,tomo_bins_cosmo): 
#                    ns = np.array([0.,0.,0.,0.])
#                    if z1==z3:
#                        ns[0] = n_s#/trapz(sp.ps*(sp.chis>=chi_bins[z1,0])*(sp.chis<=chi_bins[z1,1])*1.,sp.chis)
#                    if z1==z4:
#                        ns[1] = n_s#/trapz(sp.ps*(sp.chis>=chi_bins[z1,0])*(sp.chis<=chi_bins[z1,1])*1.,sp.chis)
#                    if z2==z4:
#                        ns[2] = n_s#/trapz(sp.ps*(sp.chis>=chi_bins[z2,0])*(sp.chis<=chi_bins[z2,1])*1.,sp.chis)
#                    if z2==z3:
#                        ns[3] = n_s#/trapz(sp.ps*(sp.chis>=chi_bins[z2,0])*(sp.chis<=chi_bins[z2,1])*1.,sp.chis)
#                    
#                    cov_g_mat[z1,z2,z3,z4] = np.diagflat(sp.cov_g_diag(np.array([qs[z1],qs[z2],qs[z3],qs[z4]]),ns,delta_ls=dls,ls=l_mids))
#                    if not isinstance(cov_g_mat_cosmo[z1,z2,z3,z4],int):
#                        if z1==0 and z2==0 and ((z3==3 and z4 ==0) or (z3 ==0 and z4==3)):
#                            print np.diag(cov_g_mat[z1,z2,z3,z4])/np.diag(cov_g_mat_cosmo[z1,z2,z3,z4])
#                        cov_g_rat_means[z1,z2,z3,z4] = np.average(np.diag(cov_g_mat[z1,z2,z3,z4])/np.diag(cov_g_mat_cosmo[z1,z2,z3,z4]))
#                    else: 
#                        print "conf",z1,z2,z3,z4
##                 if not isinstance(cov_g_mat_cosmo[z1,z2,z4,z3],int):
##                        cov_g_rat_means[z1,z2,z4,z3] = np.average(np.diag(cov_g_mat[z1,z2,z3,z4])/np.diag(cov_g_mat_cosmo[z1,z2,z4,z3]))
##                    if not isinstance(cov_g_mat_cosmo[z2,z1,z3,z4],int):
##                        cov_g_rat_means[z2,z1,z3,z4] = np.average(np.diag(cov_g_mat[z1,z2,z3,z4])/np.diag(cov_g_mat_cosmo[z2,z1,z3,z4]))
##                    if not isinstance(cov_g_mat_cosmo[z2,z1,z4,z3],int):
##                        cov_g_rat_means[z2,z1,z4,z3] = np.average(np.diag(cov_g_mat[z1,z2,z3,z4])/np.diag(cov_g_mat_cosmo[z2,z1,z4,z3]))
#                    if cov_g_rat_means[z1,z2,z3,z4]>5:
#                        print "large ",z1,z2,z3,z4
#                    if cov_g_rat_means[z1,z2,z3,z4]<0.5 and not cov_g_rat_means[z1,z2,z3,z4]==0:
#                        print "small ",z1,z2,z3,z4
#                    #cov_g_mat[z1,z2,z4,z3] = cov_g_mat[z1,z2,z3,z4]
#                    #cov_g_mat[z2,z1,z3,z4] = cov_g_mat[z1,z2,z3,z4]
#                    #cov_g_mat[z2,z1,z4,z3] = cov_g_mat[z1,z2,z3,z4]
    import matplotlib.pyplot as plt
    ax = plt.subplot(111)
    bin1 = 0
    bin2 = 0
    bin3 = 1
    bin4 = 0
    ax.loglog(l_mids,np.diag(cov_g_mat_flat[0:nbins_cosmo,0:nbins_cosmo]))
    ax.loglog(l_mids,np.diag(cov_g_mat_cosmo_flat[0:nbins_cosmo,0:nbins_cosmo]))
    #ax.loglog(l_mids,np.diag(cov_g_mat[bin1,bin2,bin3,bin4]))
    #ax.loglog(l_mids,np.diag(cov_g_mat_cosmo[bin1,bin2,bin3,bin4]))
    #ax.loglog(l_mids,Cll_shear_shear[bin1,bin2])
    #ax.loglog(l_mids,Cll_shear_shear_cosmo[bin1,bin2])
    #print(Cll_shear_shear[bin1,bin2]/Cll_shear_shear_cosmo[bin1,bin2])
    plt.show()

