"""reconsitute the dill pickle from wfirst_embed_demo.py"""
from __future__ import division,print_function,absolute_import
from builtins import range
import dill
import numpy as np
import matplotlib.pyplot as plt
from super_survey import make_standard_ellipse_plot,make_ellipse_plot
from multi_fisher import get_eig_set

if __name__=='__main__':
    do_plots = True
    #dump_f = open('record/run_x80_con104/dump_x80_1642415_con104.pkl','r')
    #dump_f = open('record/run_x80_con96/dump_x80_1642451_con96.pkl','r')
    #dump_f = open('record/run_x80_con92/dump_x80_1642455_con92.pkl','r')
    #dump_f = open('record/run_x80_con95/dump_x80_1642452_con95.pkl','r')
    #dump_f = open('record/run_x80_con149/dump_x80_1652683_con149.pkl','r')
    #dump_f = open('record/run_x80_con142/dump_x80_1650772_con142.pkl','r')
    #dump_f = open('record/run_x80_con148/dump_x80_1652687_con148.pkl','r')
    #dump_f = open('record/run_x80_con174/dump_x80_1653719_con174.pkl','r')
    #dump_f = open('record/run_x80_con178/dump_x80_1659312_con178.pkl','r')
    dump_f = open('record/run_x80_con179/dump_x80_1660833_con179.pkl','r')
    #dump_f = open('record/run_x80_con180/dump_x80_1660797_con180.pkl','r')
    dump_set = dill.load(dump_f)
    f_set_nopriors = np.array([[None,None,None],[None,None,None],[None,None,None]])
    f_set = np.array([[None,None,None],[None,None,None],[None,None,None]])
#    f_set_nopriors[0][2] = dump_set[0]
#    f_set_nopriors[1][2] = dump_set[1]
#    f_set_nopriors[2][2] = dump_set[2]
#    f_set[0][2] = dump_set[3]
#    f_set[1][2] = dump_set[4]
#    f_set[2][2] = dump_set[5]
#    cosmo_par_list = dump_set[6]
    eig_set = np.array([None,None])
    eig_set_ssc = np.array([None,None])
    [f_set_nopriors[0][2],f_set_nopriors[1][2],f_set_nopriors[2][2],f_set[0][2],f_set[1][2],f_set[2][2],cosmo_par_list,eig_set[1],eig_set_ssc[1],lw_param_list,n_params_lsst,power_params,nz_params_wfirst_lens,sw_observable_list,lw_observable_list,sw_params,len_params,x_cut,l_max,zs,zs_lsst,z_fine,mf_params,basis_params,cosmo_par_eps,cosmo,poly_params,f_set_nopriors[0][1],f_set_nopriors[1][1],f_set_nopriors[2][1],sw_to_par_array] = dump_set
#    eig_set[1] = dump_set[7]
#    eig_set_ssc[1] = dump_set[8]
#    power_params = dump_set[11]
#    zs = dump_set[19]
#    zs_lsst = dump_set[20]
#    z_fine = dump_set[21]
#    mf_params = dump_set[22]
#    basis_params = dump_set[23]
#    cosmo_par_eps = dump_set[24]
#    cosmo = dump_set[25]
#    poly_params = dump_set[26]
#
#    eig_set_priors = get_eig_set(f_set,False,False)
#
#    dchi2 = 2.3
#    cov_g_inv = f_set_nopriors[0][2].get_fisher()
#    chol_g = f_set_nopriors[0][2].get_cov_cholesky()
#    u_no_mit = eig_set[1][0][1]
#    v_no_mit = np.dot(chol_g,u_no_mit)
#    of_no_mit = np.dot(cov_g_inv,v_no_mit)
#
#    c_rot_eig_no_mit = np.dot(of_no_mit.T,np.dot(f_set_nopriors[1][2].get_covar(),of_no_mit))
#    c_rot_eig_mit = np.dot(of_no_mit.T,np.dot(f_set_nopriors[2][2].get_covar(),of_no_mit))
#    c_rot_eig_g = np.dot(of_no_mit.T,np.dot(f_set_nopriors[0][2].get_covar(),of_no_mit))
#    opacities = np.array([1.,1.,1.])
#    colors = np.array([[0,1,0],[1,0,0],[0,0,1]])
#    pnames = np.array(['p7','p6','p5','p4','p3','p2','p1'])
#    names = np.array(['g','no mit','mit'])
#    boxes = np.array([5.,5.,5.,5.,5.,5.,5.,])
#    cov_set_1 = np.array([c_rot_eig_g,c_rot_eig_no_mit,c_rot_eig_mit])
#    cov_set_2 = np.array([c_rot_eig_g[-2:,-2:],c_rot_eig_no_mit[-2:,-2:],c_rot_eig_mit[-2:,-2:]])
#    direction_norm = np.array([1.,cosmo['Omegamh2'],cosmo['Omegabh2'],cosmo['OmegaLh2'],1.,1.,1.])
#    #print("main: most contaminated direction: ",of_no_mit[:,-1]/direction_norm)
#    print("contaminated prod w/o, w priors ",np.product(eig_set[1][0][0]),np.product(eig_set_priors[1][0][0]))
#    print("mitigated prod w/o, w priors ",np.product(eig_set[1][1][0]),np.product(eig_set_priors[1][1][0]))
#    print("log(det(C_g)) w/o, w priors ",np.log(np.linalg.det(f_set_nopriors[0][2].get_covar())),np.log(np.linalg.det(f_set[0][2].get_covar())))
#    print("log(det(C_ssc)) w/o, w priors ",np.log(np.linalg.det(f_set_nopriors[1][2].get_covar())),np.log(np.linalg.det(f_set[1][2].get_covar())))
#    print("log(det(C_ssc_mit)) w/o, w priors ",np.log(np.linalg.det(f_set_nopriors[2][2].get_covar())),np.log(np.linalg.det(f_set[2][2].get_covar())))
#    if do_plots:
#        fig1 = make_standard_ellipse_plot(f_set,cosmo_par_list)
#        plt.show(fig1)
#        fig2 = make_ellipse_plot(cov_set_2,colors,opacities,names,boxes[-2:],pnames[-2:],dchi2,1.05,False,'equal',2.,(4,4),0.17,0.99,0.99,0.05)
#        plt.show(fig2)
#        dump_set = [f_set_nopriors[0][2],f_set_nopriors[1][2],f_set_nopriors[2][2],f_set[0][2],f_set[1][2],f_set[2][2],cosmo_par_list,eig_set[1],eig_set_ssc[1],lw_param_list,n_params_wfirst,power_params,nz_params_wfirst_lens,sw_observable_list,lw_observable_list,sw_params,len_params,x_cut,l_max,zs,zs_lsst,z_fine,mf_params,basis_params,cosmo_par_eps,cosmo,poly_params]
#TODO need sw correlation matrix
import prior_fisher as pf
import defaults
if cosmo['de_model'] == 'jdem':
    f_g_w0wa = pf.project_w0wa(f_set_nopriors[0][2].get_fisher(),defaults.prior_fisher_params.copy(),np.array([]))[0]
    f_g_w0 = pf.project_w0(f_set_nopriors[0][2].get_fisher(),defaults.prior_fisher_params.copy(),np.array([]))[0]
    eig_g_w0wa  = np.linalg.eigh(f_g_w0wa)[0]
elif cosmo['de_model'] == 'w0wa':
    f_g_w0wa = f_set_nopriors[0][2].get_fisher()
    f_g_w0 = pf.project_w0wa_to_w0(f_set_nopriors[0][2].get_fisher(),defaults.prior_fisher_params.copy(),np.array([]))[0]
    eig_g_w0wa  = np.linalg.eigh(f_g_w0wa)[0]
elif cosmo['de_model'] == 'constant_w':
    f_g_w0 = f_set_nopriors[0][2].get_fisher()

eig_g_w0  = np.linalg.eigh(f_g_w0)[0]
