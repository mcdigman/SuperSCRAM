"""demonstration of ability to compute super sample variance term in a geometry"""
from __future__ import print_function,division,absolute_import
from builtins import range
from time import time
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from cosmopie import CosmoPie
import defaults
from sph_klim import SphBasisK
import matter_power_spectrum as mps
from half_sky_geo import HalfSkyGeo

if __name__=='__main__':
    time0 = time()
    camb_params = { 'npoints':2000,
                    'minkh':1.1e-4,
                    'maxkh':1.476511342960e+02,
                    'kmax':1.476511342960e+02,
                    'leave_h':False,
                    'force_sigma8':False,
                    'return_sigma8':False,
                    'accuracy':1,
                    'pivot_scalar':0.002
                  }
    print("main: building cosmology")
    power_params = defaults.power_params.copy()
    power_params.camb = camb_params
    C = CosmoPie(defaults.cosmology.copy(),p_space='jdem')
    P_lin = mps.MatterPower(C,power_params)
    C.set_power(P_lin)
    time1 = time()

    #x_cut = 527.
    #l_max = 511
    #x_cut = 360
    #l_max = 346
    #x_cut = 150
    #l_max = 139
    #x_cut = 30
    #l_max = 23
    x_cut = 93
    l_max = 84

    param_use = 7
    do_plot = False
    if param_use==1:
        #matches to  0.918455921357 for l_max=340,x_cut=360, not monotonic
        theta0 = np.pi/2.-np.pi*0.21644180767757945
        theta1 = np.pi/2.+np.pi*0.21644180767757945
        phi0 = 0.
        phi1 = np.pi*0.4328860177816256

        z_coarse = np.array([0.001,0.005])
        z_max = 0.0055
        z_fine = np.arange(0.00001,0.2,0.00001)

        variance_pred1 = 6.94974e-01
        variance_pred2 = 6.47471e-01
    elif param_use==2:
        #matches to 0.984883733033 for l_max=340,x_cut=360, monotonic
        theta0 = np.pi/2.-np.pi*0.005549789940450755
        theta1 = np.pi/2.+np.pi*0.005549789940450755
        phi0 = 0.
        phi1 = np.pi*0.011099641481580144

        z_coarse = np.array([0.9,1.0])
        z_fine = np.arange(0.0005,1.,0.0005)
        z_max = 1.05

        variance_pred1 = 1.32333e-02
        variance_pred2 = 1.38770e-02
    elif param_use==3:
        #matches to  0.256984236659 for l_max=340,x_cut=360, monotonic but convergence appears poor
        #matches to 0.442356296589 for l_max=515, x_cut=
        theta0 = np.pi/2.-np.pi/1801.87
        theta1 = np.pi/2.+np.pi/1801.87
        phi0 = 0.
        phi1 = np.pi/900.93

        z_coarse = np.array([0.99525,1.0])
        z_fine = np.arange(0.0005,1.,0.0005)
        z_max = 1.05

        variance_pred1 = 1.18262e+00
        variance_pred2 = 1.18253e+00
    elif param_use==4:
        #matches to 0.982422823102 side 0.985672230237 volume  for l_max=340,x_cut=360, monotonic, convergence looks good
        theta0 = np.pi/2.-np.pi*0.025320951131222628
        theta1 = np.pi/2.+np.pi*0.025320951131222628
        phi0 = 0.
        phi1 = np.pi*0.05064218331592479

        z_coarse = np.array([0.8102,1.0])
        z_fine = np.arange(0.0005,1.,0.0005)
        z_max = 1.05

        variance_pred1 = 4.91106e-04
        variance_pred2 = 4.89487E-04
    elif param_use==5:
        #20000 deg^2
        #matches to side 1.17896133281 vol 0.914043964767 for l_max=340,x_cut=360, appears converged
        theta0 = np.pi/2.-np.pi*0.48476681987396752
        theta1 = np.pi/2.+np.pi*0.48476681987396752
        phi0 = 0.
        phi1 = np.pi*0.96953902048583762

        z_coarse = np.array([0.9,1.3])
        z_fine = np.arange(0.0005,1.,0.0005)
        z_max = 1.301

        #by sides
        variance_pred1 = 2.88548E-07
        #by volume
        variance_pred2 = 3.10674E-07
    elif param_use==6:
        #2000 deg^2
        #matches to side 1.17896133281 vol 0.914043964767 for l_max=340,x_cut=360, appears converged
        theta0 = np.pi/2.-np.pi*0.12306206198097261
        theta1 = np.pi/2.+np.pi*0.12306206198097261

        phi0 = 0.
        phi1 = np.pi*0.24612548990671312

        z_coarse = np.array([0.9,1.3])
        z_fine = np.arange(0.0005,1.,0.0005)
        z_max = 1.301

        #by sides
        variance_pred1 = 5.50604e-06
        #by volume
        variance_pred2 = 4.95809e-06
    elif param_use==7:
        #20000 deg^2
        #matches to side 1.17896133281 vol 0.914043964767 for l_max=340,x_cut=360, appears converged
        theta0 = np.pi/2.-np.pi*0.48476681987396752
        theta1 = np.pi/2.+np.pi*0.48476681987396752
        phi0 = 0.
        phi1 = np.pi*0.96953902048583762

        z_coarse = np.array([0.2,1.3])
        z_fine = np.arange(0.0005,1.,0.0005)
        z_max = 1.301

        #by sides
        variance_pred1 = 1.12606e-07
        #by volume
        variance_pred2 = 1.50551e-07
    thetas = np.array([theta0,theta1,theta1,theta0,theta0])
    phis = np.array([phi0,phi0,phi1,phi1,phi0])
    theta_in = (theta1+theta0)/2.
    phi_in = (phi1+phi0)/2.

    l_sw = np.logspace(np.log(30),np.log(5000),base=np.exp(1.),num=20)

    print("main: building geometries")
    polygon_params = defaults.polygon_params.copy()
    polygon_params['n_double'] = 80
#    geo1 = PolygonGeo(z_coarse,thetas,phis,theta_in,phi_in,C,z_fine,l_max,polygon_params)
    #geo1 = LSSTGeoSimpl(z_coarse,C,z_fine,l_max,polygon_params) #best con 2.51447459e-07
    #z_coarse = np.arange(0.2,3.01,0.2)#np.array([0.2,0.4])
    z_coarse = np.array([0.2,0.4])
    #z_max = np.max(z_coarse)
    z_max = 3.01
    #z_fine = np.arange(0.0001,z_max,0.0001)
    z_fine = np.linspace(0.001,z_max,2)

    print("main: building basis")
    basis_params = defaults.basis_params.copy()
    basis_params['n_bessel_oversample'] = 400000
    basis_params['x_grid_size'] = 100000

    r_max = C.D_comov(z_max)
    k_cut = x_cut/r_max
    k_tests = np.linspace(20./r_max,k_cut,25)
    n_basis = np.zeros(k_tests.size)
    variances = np.zeros((k_tests.size,z_coarse.size-1,z_coarse.size-1))

    time4 = time()
    basis = SphBasisK(r_max,C,k_cut,basis_params,l_ceil=l_max,needs_m=False)
    time5 = time()
    time2 = time()
    #geo1 = LSSTGeoSimpl(z_coarse,C,z_fine,l_max,polygon_params,phi0=0.,phi1=0.9202821591024097,deg0=-59,deg1=-10)#WFIRSTGeo(z_coarse,C,z_fine,l_max,polygon_params) #best con 2.43117008e-06
    #geo1 = FullSkyGeo(z_coarse,C,z_fine)
    geo1 = HalfSkyGeo(z_coarse,C,z_fine)
    #geo1 = FullSkyPixelGeo(z_coarse,C,z_fine,l_max,6)
    time3 = time()

    #geo1 = WFIRSTGeo(z_coarse,C,z_fine,l_max,polygon_params) #best con 2.43117008e-06
    #geo1 = CircleGeo(z_coarse,C,0.45509788222857989,100,z_fine,l_max,polygon_params)
    #geo1 = LSSTGeoSimpl(z_coarse,C,z_fine,l_max,defaults.polygon_params.copy()) #best con 2.43117008e-06
    print('main: r diffs',np.diff(geo1.rs))
    print('main: theta width',(geo1.rs[1]+geo1.rs[0])/2.*(theta1-theta0))
    #print('main: phi width',(geo1.rs[1]+geo1.rs[0])/2.*(phi1-phi0)*np.sin((theta1+theta0)/2))
    print('main: phi width',(geo1.rs[1]+geo1.rs[0])/2.*(phi1-phi0))
#    geo2 = PolygonGeo(z_coarse,thetas,phis+phi1,theta_in,phi_in+phi1,C,z_fine,l_max,defaults.polygon_params.copy())
#    #geo1 = RectGeo(z_coarse,np.array([theta0,theta1]),np.array([phi0,phi1]),C,z_fine)
#    #geo2 = RectGeo(z_coarse,np.array([theta0,theta1]),np.array([phi0,phi1])+phi1,C,z_fine)



    for i in range(0,k_tests.size):
        print("r_max,k_cut,j",r_max,k_tests[i])
        n_basis[i] = np.sum(basis.C_id[:,1]<=k_tests[i])
        variances[i,:,:] = basis.get_variance(geo1,k_cut_in=k_tests[i])
        print("main: with k_cut="+str(k_tests[i])+" size="+str(n_basis[i])+" got variance="+str(variances[i,0,0]))

    print("main: getting variance")
    variance_res = basis.get_variance(geo1)
    time6 = time()
    print("main: finished geo in "+str(time3-time2)+" s")
    print("main: finished basis in "+str(time5-time4)+" s")
    print("main: finished getting in "+str(time6-time5)+" s")
    print("main: finished all in "+str(time6-time0)+" s")
    times = np.array([time0,time1,time2,time3,time4,time5,time6])

    if do_plot:
        import matplotlib.pyplot as plt
        for j in range(0,z_coarse.size-1):
            plt.plot(n_basis,variances[:,j,j])
        plt.show()

    r_width = np.diff(geo1.rs)[0]
    theta_width = (geo1.rs[1]+geo1.rs[0])/2.*(theta1-theta0)
    phi_width = (geo1.rs[1]+geo1.rs[0])/2.*(phi1-phi0)
    volume = 4./3.*np.pi*(geo1.rs[1]**3-geo1.rs[0]**3)*geo1.angular_area()/(4.*np.pi)
    square_equiv = volume**(1./3.)
    print('main: r diffs',r_width*C.h)
    print('main: theta width',theta_width*C.h)
    #print('main: phi width',(geo1.rs[1]+geo1.rs[0])/2.*(phi1-phi0)*np.sin((theta1+theta0)/2))
    print('main: phi width',phi_width*C.h)
    print('main: square side length',square_equiv*C.h)
    print("main: variance is "+str(variance_res))
    #print("main: variance/predicted by sides is "+str(variances[-1]/variance_pred1))
    #print("main: variance/predicted by volume is "+str(variances[-1]/variance_pred2))
    #print("main: rate of change of variance is "+str((variances[-1]-variances[-2])/(k_tests[-1]-k_tests[-2])))
    #approx_deriv = np.average((np.diff(variances)/np.diff(n_basis))[-5::])
    approx_deriv = (variances[-1,0,0]-variances[-5,0,0])/(n_basis[-1]-n_basis[-5])
    estim_change = approx_deriv*n_basis[-1]*2.
    estim_converge = estim_change/variances[-1,0,0]
    print("main: estimate variance converged to within an error of"+str(estim_converge*100.)+"%, first approx of true value ~"+str(estim_change+variances[-1,0,0]))
    converge_approx = InterpolatedUnivariateSpline(n_basis,variances[:,0,0]/variances[-1,0,0],k=3,ext=2)
    if n_basis[-1]>=40000:
        print("main: estimated convergence of 40000 element basis: "+str(converge_approx(40000)))
    do_dump = True
    if do_dump:
        import dill
        results = [z_coarse,z_fine,k_cut,l_max,camb_params,power_params,l_sw,z_max,r_max,k_tests,n_basis,variances,variance_res,r_width,theta_width,phi_width,volume,square_equiv,times]
        dump_f = open('dump_var_con.pkl','w')
        dill.dump(results,dump_f)
        dump_f.close()
