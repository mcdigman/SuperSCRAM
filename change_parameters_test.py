"""test change_parameter.py"""
import pytest
import numpy as np
import cosmopie as cp
import defaults
from change_parameters import rotate_jdem_to_lihu,rotate_lihu_to_jdem
def test_change_params():
    """test rotation function work"""
    C_fid = cp.CosmoPie(defaults.cosmology.copy(),'jdem')


    f_set_in1 = np.zeros(3,dtype=object)
    for i in range(0,3):
        f_set1 = np.random.rand(6,6)
        f_set1 = np.dot(f_set1.T,f_set1)
        f_set1 = f_set1+np.diag(np.random.rand(6))
        f_set_in1[i] = f_set1
    f_set_in2 = rotate_jdem_to_lihu(f_set_in1,C_fid)
    f_set_in3 = rotate_lihu_to_jdem(f_set_in2,C_fid)
    f_set_in4 = rotate_jdem_to_lihu(f_set_in3,C_fid)
    for i in range(0,3):
        assert np.allclose(f_set_in1[i],f_set_in3[i])
        assert np.allclose(f_set_in2[i],f_set_in4[i])

if __name__=='__main__':
    pytest.cmdline.main(['change_parameters_test.py'])
