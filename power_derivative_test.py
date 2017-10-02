import numpy as np
from shear_power import ShearPower,Cll_q_q
from cosmopie import CosmoPie
import defaults
from lensing_weight import QShear
import matter_power_spectrum as mps

if __name__=='__main__':
    omega_s = 0.02
    ls = np.arange(2,3000)

    #d = np.loadtxt('camb_m_pow_l.dat')
    #k_in = d[:,0]
    #P_in = d[:,1]


    C = CosmoPie(defaults.cosmology)
    P_in = mps.MatterPower(C,camb_params=defaults.camb_params)
    k_in = P_in.k
    C.P_lin = P_in
    C.k = k_in
    #C.k = k_in
    #C.P_lin = P_in
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
    r_min1 = C.D_comov(z_min1)
    r_max1 = C.D_comov(z_max1)

    z_min2 = 1.6
    z_max2 = 1.8
    r_min2 = C.D_comov(z_min2)
    r_max2 = C.D_comov(z_max2)

    QShear1_1 = QShear(dC_ddelta1,r_min1,r_max1)
    QShear1_2 = QShear(dC_ddelta1,r_min2,r_max2)

    #QShear2_1 = QShear(dC_ddelta2,r_min1,r_max1)
    #QShear2_2 = QShear(dC_ddelta2,r_min2,r_max2)

    ss_1 = Cll_q_q(sp1,QShear1_1,QShear1_2).Cll()
    #check that \partial C^{ij}/\partial \bar{\delta}(zs) \propto 1/(width of zs bin)*z^i*C^{ij}, where z^i ~ average z of closer z bin
    do_plots = False
    if do_plots:
        import matplotlib.pyplot as plt
    fails = 0
    do_dz_test=True
    if do_dz_test:
        results1_0 = np.zeros((5,ls.size))
        results1_test = np.zeros((5,ls.size))
        for z_ind in xrange(10,35,5):
            ind_min = 200
            ind_max = ind_min+z_ind
            dC_ss_1 = Cll_q_q(dC_ddelta1,QShear1_1,QShear1_2).Cll(chi_min=dC_ddelta1.chis[ind_min],chi_max=dC_ddelta1.chis[ind_max])
            #dC_ss_1 = Cll_q_q(dC_ddelta1,QShear1_1,QShear1_2).Cll()
            ss_1_loc = Cll_q_q(sp1,QShear1_1,QShear1_2).Cll(chi_min=dC_ddelta1.chis[ind_min],chi_max=dC_ddelta1.chis[ind_max])
            #ax.loglog(ls,dC_ss_integrand_1/ss_1/(dC_ddelta1.zs[ind_max]-dC_ddelta1.zs[ind_min]))
            #results[(z_ind-10)/5] = dC_ss_integrand_1/ss_1/(dC_ddelta1.zs[ind_max]-dC_ddelta1.zs[ind_min])*(dC_ddelta1.zs[ind_max]+dC_ddelta1.zs[ind_min])/2.
            results1_test[(z_ind-10)/5] = dC_ss_1/ss_1/(dC_ddelta1.zs[ind_max]-dC_ddelta1.zs[ind_min])#*(z_min1+z_max1)/2.#*(dC_ddelta1.zs[ind_max]+dC_ddelta1.zs[ind_min])/2.
            results1_0[(z_ind-10)/5] = dC_ss_1/ss_1#*(z_min1+z_max1)/2.#*(dC_ddelta1.zs[ind_max]+dC_ddelta1.zs[ind_min])/2.
            #ax.loglog(ls,dC_ss_integrand_1/ss_1/(dC_ddelta1.zs[ind_max]-dC_ddelta1.zs[ind_min])*(dC_ddelta1.zs[ind_max]+dC_ddelta1.zs[ind_min])/2.)

            #ax.loglog(ls,dC_ss_1/ss_1/(dC_ddelta1.zs[ind_max]-dC_ddelta1.zs[ind_min])*(z_min1+z_max1)/2.)
            #a##x.loglog(ls,dC_ss_integrand_1/ss_1_loc)#(dC_ddelta1.zs[ind_max]-dC_ddelta1.zs[ind_min]))
            #print np.average(dC_ss_1/ss_1/(dC_ddelta1.zs[ind_max]-dC_ddelta1.zs[ind_min])*(z_min1+z_max1)/2.)
            #dC_ss_1 = Cll_q_q(dC_ddelta1,QShear1_1,QShear1_2).Cll()
        results_norm1_test = results1_test/results1_test[0] 
        results_norm1_0 = results1_0/results1_0[0] 
        mean_abs_error1_test = np.average(np.abs(1.-results_norm1_test),axis=1)
        mean_abs_error1_0 = np.average(np.abs(1.-results_norm1_0),axis=1)
        print "avg relative test error with, without delta z component: "+str(np.mean(mean_abs_error1_test[1::]))+","+str(np.mean(mean_abs_error1_0[1::]))
        if np.all((mean_abs_error1_0[1::]/mean_abs_error1_test[1::])>10.):
            print "PASS: fit significantly better with delta z dependent component than without"
        elif np.any((mean_abs_error1_0[1::]/mean_abs_error1_test[1::])<1.):
            print "FAIL: fit worse with z dependent component than without"
            fails+=1
        else:
            print "SOFT PASS: fit at least somewhat better with delta z dependent component than without"
        if np.all(mean_abs_error1_test[1::]<0.03):
            print "PASS: testing of delta z component within acceptable error"
        else:
            print "FAIL: testing of delta z component outside acceptable error"
            fails+=1
        if do_plots:
            ax = plt.subplot(111)
            ax.loglog(ls,results1_test.T)
            plt.show()

    do_zavg_test2 = True
    if do_zavg_test2:
        results2_test = np.zeros((5,ls.size))
        results3_0 = np.zeros((5,ls.size))
        results2_0 = np.zeros((5,ls.size))
        z_test2 = np.linspace(0.8,0.9,6)
        #import matplotlib.pyplot as plt
        for itr in range(0,5):
            QShear1_itr =  QShear(dC_ddelta1,C.D_comov(z_test2[itr]),C.D_comov(z_test2[itr+1]))
            ss_itr = Cll_q_q(sp1,QShear1_itr,QShear1_2).Cll()
            dC_ss_itr = Cll_q_q(dC_ddelta1,QShear1_itr,QShear1_2).Cll(chi_min=dC_ddelta1.chis[200],chi_max=dC_ddelta1.chis[210])
            results2_test[itr] = dC_ss_itr/ss_itr*np.average(z_test2[itr:itr+2])
            results3_0[itr] = dC_ss_itr*np.average(z_test2[itr:itr+2])
            results2_0[itr] = dC_ss_itr/ss_itr#*np.average(z_test2[itr:itr+2])
         #   ax = plt.subplot(111)
         #   ax.loglog(ls,results2_test[itr])
        results_norm2_test = results2_test/results2_test[0] 
        results_norm2_0 = results2_0/results2_0[0] 
        mean_abs_error2_test = np.average(np.abs(1.-results_norm2_test),axis=1)
        mean_abs_error2_0 = np.average(np.abs(1.-results_norm2_0),axis=1)
        print "avg relative test error with, without z component: "+str(np.mean(mean_abs_error2_test[1::]))+","+str(np.mean(mean_abs_error2_0[1::]))
        if np.all((mean_abs_error2_0[1::]/mean_abs_error2_test[1::])>10.):
            print "PASS: fit significantly better with z dependent component than without"
        elif np.any((mean_abs_error2_0[1::]/mean_abs_error2_test[1::])<1.):
            print "FAIL: fit worse with z dependent component than without"
            fails+=1
        else:
            print "SOFT PASS: fit at least somewhat better with z dependent component than without"

        #results_norm3_test = results3_test/results3_test[0] 
        results_norm3_0 = results3_0/results3_0[0] 
        #mean_abs_error3_test = np.average(np.abs(1.-results_norm3_test),axis=1)
        mean_abs_error3_0 = np.average(np.abs(1.-results_norm3_0),axis=1)
        print "avg relative test error with, without C^ll component: "+str(np.mean(mean_abs_error2_test[1::]))+","+str(np.mean(mean_abs_error3_0[1::]))
        if np.all((mean_abs_error3_0[1::]/mean_abs_error2_test[1::])>10.):
            print "PASS: fit significantly better with C^ll dependent component than without"
        elif np.any((mean_abs_error3_0[1::]/mean_abs_error2_test[1::])<1.):
            print "FAIL: fit worse with C^ll dependent component than without"
            fails+=1
        else:
            print "SOFT PASS: fit at least somewhat better with z dependent component than without"
        if np.all(mean_abs_error2_test[1::]<0.01):
            print "PASS: testing of C^ll and z component within acceptable error"
        else:
            print "FAIL: testing of C^ll and z component outside acceptable error"
            fails+=1


        if do_plots:
            ax = plt.subplot(111)
            ax.loglog(ls,results2_test.T)
            plt.show()
    if fails==0:
        print "PASS: All checks satisfactory"
    else:
        print "Fail: "+str(fails)+" checks unsatisfactory"
    #dC_ss_integrand_2 = Cll_q_q(dC_ddelta2,QShear2_1,QShear2_2).Cll_integrand()

    #dz_1s = np.hstack((dC_ddelta1.zs[0],np.diff(dC_ddelta1.zs)))/np.hstack((dC_ddelta1.chis[0],np.diff(dC_ddelta1.chis)))
    #dz_2s = np.hstack((dC_ddelta2.zs[0],np.diff(dC_ddelta2.zs)))/np.hstack((dC_ddelta2.chis[0],np.diff(dC_ddelta2.chis)))

    #ss_2 = Cll_q_q(sp2,QShear2_1,QShear2_2).Cll()
    do_zavg_test1=False 
    if do_zavg_test1:
        dC_ss_integrand_1 = Cll_q_q(dC_ddelta1,QShear1_1,QShear1_2).Cll()
        import matplotlib.pyplot as plt
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
