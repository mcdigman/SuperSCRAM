r"""test that the shear shear power spectrum derivatives have some expected functional dependencies,
specifically \partial C^{ij}/\partial \bar{\delta}(zs) \propto 1/(width of zs bin)*z^i*C^{ij},
where z^i ~ average z of closer z bin"""
#pylint: disable=W0621
from __future__ import division,print_function,absolute_import
from builtins import range
import numpy as np
import pytest
from shear_power import ShearPower,Cll_q_q
from cosmopie import CosmoPie
import defaults
from lensing_weight import QShear
import matter_power_spectrum as mps

@pytest.fixture()
def util_set():
    """get set of things needed for tests"""
    omega_s = 0.02
    power_params = defaults.power_params.copy()
    C = CosmoPie(defaults.cosmology.copy(),p_space='jdem')
    P_in = mps.MatterPower(C,power_params)
    C.set_power(P_in)

    len_params = defaults.lensing_params.copy()
    len_params['z_bar'] = 1.0
    len_params['sigma'] = 0.4
    len_params['l_min'] = 30
    len_params['l_max'] = 3000
    len_params['n_l'] = 1000
    len_params['n_gal'] = 118000000*6.

    z_test_res1 = 0.001
    zs_test1 = np.arange(0.0005,2.,z_test_res1)

    dC_ddelta1 = ShearPower(C,zs_test1,omega_s,len_params,mode='dc_ddelta')

    sp1 = ShearPower(C,zs_test1,omega_s,len_params,mode='power')
    ls = sp1.l_starts


    z_min1 = 0.8
    z_max1 = 1.0
#    r_min1 = C.D_comov(z_min1)
#    r_max1 = C.D_comov(z_max1)

    z_min2 = 1.6
    z_max2 = 1.8
#    r_min2 = C.D_comov(z_min2)
#    r_max2 = C.D_comov(z_max2)

    QShear1_1 = QShear(dC_ddelta1,z_min1,z_max1)
    QShear1_2 = QShear(dC_ddelta1,z_min2,z_max2)

    ss_1 = Cll_q_q(sp1,QShear1_1,QShear1_2).Cll()
    return [C,dC_ddelta1,QShear1_1,QShear1_2,ss_1,ls,sp1]

def test_dz(util_set):
    """test correct dependence on bin width"""
    dC_ddelta1 = util_set[1]
    QShear1_1 = util_set[2]
    QShear1_2 = util_set[3]
    ss_1 = util_set[4]
    ls = util_set[5]

    results1_0 = np.zeros((5,ls.size))
    results1_test = np.zeros((5,ls.size))
    for z_ind in range(10,35,5):
        ind_min = 200
        ind_max = ind_min+z_ind
        dC_ss_1 = Cll_q_q(dC_ddelta1,QShear1_1,QShear1_2).Cll(dC_ddelta1.zs[ind_min],dC_ddelta1.zs[ind_max])
        results1_test[int((z_ind-10)/5)] = dC_ss_1/ss_1/(dC_ddelta1.zs[ind_max]-dC_ddelta1.zs[ind_min])
        results1_0[int((z_ind-10)/5)] = dC_ss_1/ss_1

    results_norm1_test = results1_test/results1_test[0]
    results_norm1_0 = results1_0/results1_0[0]
    mean_abs_error1_test = np.average(np.abs(1.-results_norm1_test),axis=1)
    mean_abs_error1_0 = np.average(np.abs(1.-results_norm1_0),axis=1)
    assert np.all((mean_abs_error1_0[1::]/mean_abs_error1_test[1::])>10.)
    assert np.all(mean_abs_error1_test[1::]<0.04)

def test_zavg(util_set):
    """test proportional to power spectrum and z average"""
#    C = util_set[0]
    dC_ddelta1 = util_set[1]
    QShear1_2 = util_set[3]
    ls = util_set[5]
    sp1 = util_set[6]

    results2_test = np.zeros((5,ls.size))
    results3_0 = np.zeros((5,ls.size))
    results2_0 = np.zeros((5,ls.size))
    z_test2 = np.linspace(0.8,0.9,6)
    for itr in range(0,5):
        QShear1_itr =  QShear(dC_ddelta1,z_test2[itr],z_test2[itr+1])
        ss_itr = Cll_q_q(sp1,QShear1_itr,QShear1_2).Cll()
        dC_ss_itr = Cll_q_q(dC_ddelta1,QShear1_itr,QShear1_2).Cll(z_min=dC_ddelta1.zs[200],z_max=dC_ddelta1.zs[210])
        results2_test[itr] = dC_ss_itr/ss_itr*np.average(z_test2[itr:itr+2])
        results3_0[itr] = dC_ss_itr*np.average(z_test2[itr:itr+2])
        results2_0[itr] = dC_ss_itr/ss_itr
    results_norm2_test = results2_test/results2_test[0]
    results_norm2_0 = results2_0/results2_0[0]
    mean_abs_error2_test = np.average(np.abs(1.-results_norm2_test),axis=1)
    mean_abs_error2_0 = np.average(np.abs(1.-results_norm2_0),axis=1)
    assert np.all((mean_abs_error2_0[1::]/mean_abs_error2_test[1::])>10.)
    results_norm3_0 = results3_0/results3_0[0]
    mean_abs_error3_0 = np.average(np.abs(1.-results_norm3_0),axis=1)
    assert np.all((mean_abs_error3_0[1::]/mean_abs_error2_test[1::])>10.)
    assert np.all(mean_abs_error2_test[1::]<0.01)

if __name__=='__main__':
    pytest.cmdline.main(['power_derivative_tests.py'])
