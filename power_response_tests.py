"""check code matches a plot from chiang&wagner arxiv:1403.3411v2 figure 4-5"""
import numpy as np
from scipy.interpolate import interp1d
import power_response as shp
import defaults
import cosmopie as cp
import matter_power_spectrum as mps
COSMOLOGY_CHIANG = {'Omegabh2' :0.023,
                    'Omegach2' :0.1093,
                    'Omegamh2' : 0.1323,
                    'OmegaL'   : 0.73,
                    'OmegaLh2' : 0.3577,
                    'Omegam'   : .27,
                    'H0'       : 70.,
                    'sigma8'   : .7913,
                    'h'        :0.7,
                    'Omegak'   : 0.0, # check on this value
                    'Omegakh2' : 0.0,
                    'Omegar'   : 0.0,
                    'Omegarh2' : 0.0,
                    'ns'       : 0.95,
                    'w'        : -1.,
                    'de_model' : 'constant_w',
                    'tau'      : None,
                    'Yp'        :None,
                    'As'        : None,
                    'LogAs'   : None,
                    'mnu'     :0.
                   }

def test_power_derivative():
    """test that the power derivatives agree with chiang&wagner arxiv:1403.3411v2 figure 4-5"""
    power_params = defaults.power_params.copy()
    power_params.camb['force_sigma8'] = True
    power_params.camb['leave_h'] = False
    C = cp.CosmoPie(cosmology=COSMOLOGY_CHIANG,p_space='basic')
    #d = np.loadtxt('camb_m_pow_l.dat')
    #k_in = d[:,0]
    epsilon = 0.00001
    #k_a,P_a = cpow.camb_pow(cosmo_a)
    P_a = mps.MatterPower(C,power_params)
    k_a = P_a.k
    C.k = k_a
    k_a_h = P_a.k/C.cosmology['h']

    d_chiang_halo = np.loadtxt('test_inputs/dp_1/dp_chiang_halofit.dat')
    k_chiang_halo = d_chiang_halo[:,0]*C.cosmology['h']
    dc_chiang_halo = d_chiang_halo[:,1]
    dc_ch1 = interp1d(k_chiang_halo,dc_chiang_halo,bounds_error=False)(k_a)
    d_chiang_lin = np.loadtxt('test_inputs/dp_1/dp_chiang_linear.dat')
    k_chiang_lin = d_chiang_lin[:,0]*C.cosmology['h']
    dc_chiang_lin = d_chiang_lin[:,1]
    dc_ch2 = interp1d(k_chiang_lin,dc_chiang_lin,bounds_error=False)(k_a)
    d_chiang_fpt = np.loadtxt('test_inputs/dp_1/dp_chiang_oneloop.dat')
    k_chiang_fpt = d_chiang_fpt[:,0]*C.cosmology['h']
    dc_chiang_fpt = d_chiang_fpt[:,1]
    dc_ch3 = interp1d(k_chiang_fpt,dc_chiang_fpt,bounds_error=False)(k_a)

    zbar = np.array([1.])
    dcalt1,p1a = shp.dp_ddelta(P_a,zbar,C=C,pmodel='linear',epsilon=epsilon)
    dcalt2,p2a = shp.dp_ddelta(P_a,zbar,C=C,pmodel='halofit',epsilon=epsilon)
    dcalt3,p3a = shp.dp_ddelta(P_a,zbar,C=C,pmodel='fastpt',epsilon=epsilon)

    mask_mult = (k_a_h>0.)*(k_a_h<0.4)
    rat_halofit = (dc_ch1/abs(dcalt2/p2a))[mask_mult]
    rat_linear = (dc_ch2/abs(dcalt1/p1a))[mask_mult]
    rat_fpt = (dc_ch3/abs(dcalt3/p3a))[mask_mult]


    #TODO what is wrong with halofit
    k_a_halofit = k_a_h[mask_mult][~np.isnan(rat_halofit)]
    k_a_linear = k_a_h[mask_mult][~np.isnan(rat_linear)]
    k_a_fpt = k_a_h[mask_mult][~np.isnan(rat_fpt)]
    dkh = 0.05
    halofit_bins = np.zeros(7)
    linear_bins = np.zeros(7)
    fpt_bins = np.zeros(7)
    for itr in range(1,8):
        mask_loc_hf = (k_a_halofit<dkh*(itr+1.))*(k_a_halofit>=dkh*itr)
        mask_loc_lin = (k_a_linear<dkh*(itr+1.))*(k_a_linear>=dkh*itr)
        mask_loc_fpt = (k_a_fpt<dkh*(itr+1.))*(k_a_fpt>=dkh*itr)
        halofit_bins[itr-1] =  np.average(rat_halofit[~np.isnan(rat_halofit)][mask_loc_hf])
        linear_bins[itr-1] =  np.average(rat_linear[~np.isnan(rat_linear)][mask_loc_lin])
        fpt_bins[itr-1] =  np.average(rat_fpt[~np.isnan(rat_fpt)][mask_loc_fpt])
    assert np.all(np.abs(halofit_bins-1.)<0.02)
    assert np.all(np.abs(linear_bins-1.)<0.02)
    assert np.all(np.abs(fpt_bins-1.)<0.02)

class PowerDerivativeComparison1(object):
    """replicate chiang&wagner arxiv:1403.3411v2 figure 4-5
    note that the mean averaged over 1 oscillation should match as should the phase of the oscillations,
    but the amplitude of the oscillations does not match because we are not convolving with a window function"""
    def __init__(self):
        """ do power derivative comparison"""
        #camb_params = defaults.camb_params.copy()
        power_params = defaults.power_params.copy()
        power_params.camb['force_sigma8'] = True
        power_params.camb['leave_h'] = False
        C = cp.CosmoPie(cosmology=COSMOLOGY_CHIANG,p_space='basic')
        #d = np.loadtxt('camb_m_pow_l.dat')
        #k_in = d[:,0]
        epsilon = 0.00001
        #k_a,P_a = cpow.camb_pow(cosmo_a)
        P_a = mps.MatterPower(C,power_params)
        k_a = P_a.k
        C.k = k_a
        k_a_h = P_a.k/C.cosmology['h']

        d_chiang_halo = np.loadtxt('test_inputs/dp_1/dp_chiang_halofit.dat')
        k_chiang_halo = d_chiang_halo[:,0]*C.cosmology['h']
        dc_chiang_halo = d_chiang_halo[:,1]
        dc_ch1 = interp1d(k_chiang_halo,dc_chiang_halo,bounds_error=False)(k_a)
        d_chiang_lin = np.loadtxt('test_inputs/dp_1/dp_chiang_linear.dat')
        k_chiang_lin = d_chiang_lin[:,0]*C.cosmology['h']
        dc_chiang_lin = d_chiang_lin[:,1]
        dc_ch2 = interp1d(k_chiang_lin,dc_chiang_lin,bounds_error=False)(k_a)
        d_chiang_fpt = np.loadtxt('test_inputs/dp_1/dp_chiang_oneloop.dat')
        k_chiang_fpt = d_chiang_fpt[:,0]*C.cosmology['h']
        dc_chiang_fpt = d_chiang_fpt[:,1]
        dc_ch3 = interp1d(k_chiang_fpt,dc_chiang_fpt,bounds_error=False)(k_a)
        do_plots = False
        if do_plots:
            import matplotlib.pyplot as plt
        zbar = np.array([3.])
        dcalt1,p1a = shp.dp_ddelta(P_a,zbar,C=C,pmodel='linear',epsilon=epsilon)
        dcalt2,p2a = shp.dp_ddelta(P_a,zbar,C=C,pmodel='halofit',epsilon=epsilon)
        dcalt3,p3a = shp.dp_ddelta(P_a,zbar,C=C,pmodel='fastpt',epsilon=epsilon)
        if do_plots:
            ax = plt.subplot(221)
            plt.xlim([0.,0.4])
            plt.ylim([1.2,3.2])
            plt.grid()
            plt.title('z=3.0')
            ax.plot(k_a_h,abs(dcalt1/p1a))
            ax.plot(k_a_h,abs(dcalt2/p2a))
            ax.plot(k_a_h,abs(dcalt3/p3a))

        zbar = np.array([2.])
        dcalt1,p1a = shp.dp_ddelta(P_a,zbar,C=C,pmodel='linear',epsilon=epsilon)
        dcalt2,p2a = shp.dp_ddelta(P_a,zbar,C=C,pmodel='halofit',epsilon=epsilon)
        dcalt3,p3a = shp.dp_ddelta(P_a,zbar,C=C,pmodel='fastpt',epsilon=epsilon)

        if do_plots:
            ax = plt.subplot(222)
            plt.xlim([0.,0.4])
            plt.ylim([1.2,3.2])
            plt.grid()
            plt.title('z=2.0')
            ax.plot(k_a_h,abs(dcalt1/p1a))
            ax.plot(k_a_h,abs(dcalt2/p2a))
            ax.plot(k_a_h,abs(dcalt3/p3a))


        zbar = np.array([1.])
        dcalt1,p1a = shp.dp_ddelta(P_a,zbar,C=C,pmodel='linear',epsilon=epsilon)
        dcalt2,p2a = shp.dp_ddelta(P_a,zbar,C=C,pmodel='halofit',epsilon=epsilon)
        dcalt3,p3a = shp.dp_ddelta(P_a,zbar,C=C,pmodel='fastpt',epsilon=epsilon)
        if do_plots:
            ax = plt.subplot(223)
            plt.xlim([0.,0.4])
            plt.ylim([1.2,3.2])
            plt.grid()
            plt.title('z=1.0')
            ax.set_xlabel('k h Mpc^-1')
            ax.set_ylabel('dln(P)/ddeltabar')
            ax.plot(k_a_h,abs(dcalt1/p1a))
            ax.plot(k_a_h,abs(dcalt2/p2a))
            ax.plot(k_a_h,abs(dcalt3/p3a))
            ax.plot(k_a_h,dc_ch1)
            ax.plot(k_a_h,dc_ch2)
            ax.plot(k_a_h,dc_ch3)
            plt.legend(['linear','halofit','fastpt','halo_chiang',"lin_chiang","fpt_chiang"],loc=4)
        mask_mult = (k_a_h>0.)*(k_a_h<0.4)
        rat_halofit = (dc_ch1/abs(dcalt2/p2a))[mask_mult]
        rat_linear = (dc_ch2/abs(dcalt1/p1a))[mask_mult]
        rat_fpt = (dc_ch3/abs(dcalt3/p3a))[mask_mult]
        #ax.plot(abs(dcalt1/p1a)-dc_ch2)


        zbar = np.array([0.])
        dcalt1,p1a = shp.dp_ddelta(P_a,zbar,C=C,pmodel='linear',epsilon=epsilon)
        dcalt2,p2a = shp.dp_ddelta(P_a,zbar,C=C,pmodel='halofit',epsilon=epsilon)
        dcalt3,p3a = shp.dp_ddelta(P_a,zbar,C=C,pmodel='fastpt',epsilon=epsilon)
        if do_plots:
            ax = plt.subplot(224)
            plt.xlim([0.,0.4])
            plt.ylim([1.2,3.2])
            plt.grid()
            plt.title('z=0.0')
            ax.plot(k_a_h,abs(dcalt1/p1a))
            ax.plot(k_a_h,abs(dcalt2/p2a))
            ax.plot(k_a_h,abs(dcalt3/p3a))
        #plt.legend(['linear','halofit','fastpt'],loc=4)
        if do_plots:
            plt.show()
        #TODO what is wrong with halofit
        k_a_halofit = k_a_h[mask_mult][~np.isnan(rat_halofit)]
        k_a_linear = k_a_h[mask_mult][~np.isnan(rat_linear)]
        k_a_fpt = k_a_h[mask_mult][~np.isnan(rat_fpt)]
        dkh = 0.05
        halofit_bins = np.zeros(7)
        linear_bins = np.zeros(7)
        fpt_bins = np.zeros(7)
        for itr in range(1,8):
            mask_loc_hf = (k_a_halofit<dkh*(itr+1.))*(k_a_halofit>=dkh*itr)
            mask_loc_lin = (k_a_linear<dkh*(itr+1.))*(k_a_linear>=dkh*itr)
            mask_loc_fpt = (k_a_fpt<dkh*(itr+1.))*(k_a_fpt>=dkh*itr)
            halofit_bins[itr-1] =  np.average(rat_halofit[~np.isnan(rat_halofit)][mask_loc_hf])
            linear_bins[itr-1] =  np.average(rat_linear[~np.isnan(rat_linear)][mask_loc_lin])
            fpt_bins[itr-1] =  np.average(rat_fpt[~np.isnan(rat_fpt)][mask_loc_fpt])
        #print np.abs(halofit_bins-1.)
        #print np.abs(linear_bins-1.)
        #print np.abs(fpt_bins-1.)
        fails = 0
        if np.all(np.abs(halofit_bins-1.)<0.02):
            print "PASS: smoothed z=1 halofit matches chiang"
        else:
            fails+=1
            print "FAIL: smoothed z=1 halofit does not match chiang"
        if np.all(np.abs(linear_bins-1.)<0.02):
            print "PASS: smoothed z=1 linear matches chiang"
        else:
            fails+=1
            print "FAIL: smoothed z=1 linear does not match chiang"
        if np.all(np.abs(fpt_bins-1.)<0.02):
            print "PASS: smoothed z=1 fastpt matches chiang"
        else:
            fails+=1
            print "FAIL: smoothed z=1 fastpt does not match chiang"
        if fails==0:
            print "PASS: all tests satisfactory"
        else:
            print "FAIL: "+str(fails)+" tests unsatisfactory"
        #plt.plot(k_a_h[mask_mult][~np.isnan(rat_halofit)],rat_halofit[~np.isnan(rat_halofit)])
        #plt.plot(k_a_h[mask_mult][~np.isnan(rat_linear)],rat_linear[~np.isnan(rat_linear)])
        #plt.show()
if __name__=='__main__':
    PowerDerivativeComparison1()
