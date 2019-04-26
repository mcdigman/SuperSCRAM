"""rebuild results from testing"""
from __future__ import print_function,division,absolute_import
from builtins import range
from scipy.interpolate import InterpolatedUnivariateSpline
import dill
if __name__=='__main__':
    dump_f = open('record/var_run_t6/dump_var_1652353_t6.pkl','r')
    #dump_f = open('record/var_run_t5/dump_var_1652352_t5.pkl','r')
    results = dill.load(dump_f)
    [z_coarse,z_fine,k_cut,l_max,camb_params,power_params,l_sw,z_max,r_max,k_tests,n_basis,variances,variance_res,r_width,theta_width,phi_width,volume,square_equiv,times] = results
    converge_approx = InterpolatedUnivariateSpline(n_basis,variances[:,0,0]/variances[-1,0,0],k=3,ext=2)

    print(variance_res)
    approx_deriv = variances[-1,0,0]*converge_approx.derivative(1)(n_basis[-1])
    estim_change = approx_deriv*n_basis[-1]*2.
    estim_converge = estim_change/variances[-1,0,0]

    #approx_deriv = (variances[-1,0,0]-variances[-2,0,0])/(n_basis[-1]-n_basis[-2])
    #estim_change = approx_deriv*n_basis[-1]*2.
    #estim_converge = estim_change/variances[-1,0,0]
    print("main: estimate variance converged to within an error of "+str(estim_converge*100.)+"%, first approx of true value ~"+str(estim_change+variances[-1,0,0]))
    if n_basis[-1]>=40000:
        print("main: estimated convergence of 40000 element basis: "+str(converge_approx(40000)))

    do_plot = True
    if do_plot:
        import matplotlib.pyplot as plt
        for j in range(0,z_coarse.size-1):
            plt.plot(n_basis,variances[:,j,j])
        plt.show()
