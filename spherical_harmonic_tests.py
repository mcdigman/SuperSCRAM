import numpy as np
import sph_functions as sph
import polygon_pixel_geo as ppg
import pytest

class ThetaPhiList:
    def __init__(self,key):
        if key==1:
            n_side = 100
            theta_need=np.linspace(0.,np.pi,n_side)
            phi_need=np.linspace(0.,2.*np.pi,n_side)
            self.thetas = np.zeros(n_side**2)
            self.phis = np.zeros(n_side**2)
            for i in range(0,n_side):
                for j in range(0,n_side):
                    self.thetas[n_side*i+j] = theta_need[j]
                    self.phis[n_side*i+j] = phi_need[i]
        else:
            raise ValueError("unrecognized key: "+str(key))

@pytest.fixture(params=[1])
def theta_phis(request):
    return ThetaPhiList(request.param)

def test_Y_R_agreement1(theta_phis):
    thetas = theta_phis.thetas.copy()
    phis = theta_phis.phis.copy()
    l_max = 30
    Y_r_1s,ls,ms = ppg.get_Y_r_table(l_max,thetas,phis)
    Y_r_2s = np.zeros_like(Y_r_1s)
    for itr in range(0,ls.size):
        Y_r_2s[itr] = sph.Y_r(ls[itr],ms[itr],thetas,phis)
    assert(np.allclose(Y_r_1s,Y_r_2s))


if __name__=='__main__':
        pytest.cmdline.main(['spherical_harmonic_tests.py'])

