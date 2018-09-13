"""test extrap_utils module"""
#pylint: disable=W0621
from __future__ import absolute_import,division,print_function
from builtins import range
import numpy as np
import pytest
from extrap_utils import power_law_extend

pow_test_list = [[0.03,3.02],[0.02,-1.05],[-0.05,0.2],[-1.6,-3.92]]
@pytest.fixture(params=pow_test_list)
def power_law_low(request):
    """iterate over low end power law parameters"""
    return request.param

@pytest.fixture(params=pow_test_list)
def power_law_high(request):
    """iterate over high end parameters"""
    return request.param

def test_power_extend_same(power_law_low):
    """test power law extend"""
    mult = power_law_low[0]
    exp = power_law_low[1]
    x_in = np.arange(0.05,1.07,0.0011)
    x_out = np.arange(0.02,1.9,0.0013)
    f_in = mult*x_in**exp
    f_out1 = mult*x_out**exp
    f_out2 = power_law_extend(x_in,f_in,x_out,k=2)
    assert np.allclose(f_out1,f_out2)

def test_power_extend2(power_law_low,power_law_high):
    """test power law extend different power laws"""
    x_in = np.arange(0.05,1.07,0.00011)
    x_out = np.arange(0.02,1.9,0.00013)
    mult_low = power_law_low[0]
    exp_low = power_law_low[1]
    mult_high = power_law_high[0]
    exp_high = power_law_high[1]
    f_in = mult_low*x_in**exp_low
    f_in[x_in>0.5] = mult_high*x_in[x_in>0.5]**exp_high
    f_out1 = mult_low*x_out**exp_low
    f_out1[x_out>0.5] = mult_high*x_out[x_out>0.5]**exp_high
    f_out2 = power_law_extend(x_in,f_in,x_out,k=2)
    #cut out the break where spline may misbehave
    assert np.allclose(f_out1[x_out>0.5+0.005],f_out2[x_out>0.5+0.005])
    assert np.allclose(f_out1[x_out<0.5-0.005],f_out2[x_out<0.5-0.005])

if __name__=='__main__':
    pytest.cmdline.main(['extrap_util_tests.py'])
