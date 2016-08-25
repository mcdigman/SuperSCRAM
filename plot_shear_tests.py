import numpy as np
import power_response as shp
import defaults 
import cosmopie as cp
import camb_power as cpow

#replicate chiang&wagner arxiv:1403.3411v2 figure 4-5
class PowerDerivativeComparison1:
    def __init__(self):
        C=cp.CosmoPie(cosmology=defaults.cosmology_chiang)
        d = np.loadtxt('camb_m_pow_l.dat')
        k_in = d[:,0]
        zs = np.arange(0.1,1.0,0.1)
        ls = np.arange(1,5000)
        epsilon = 0.0001

        cosmo_a = defaults.cosmology_chiang.copy()
        k_a,P_a = cpow.camb_pow(cosmo_a)
        
        import matplotlib.pyplot as plt

        zbar = 3.
        ax = plt.subplot(221)
        plt.xlim([0.,0.4])
        plt.ylim([1.2,3.2])
        plt.grid()
        plt.title('z=3.0')
        dcalt1,p1a = shp.dp_ddelta(k_a,P_a,zbar,pmodel='linear')
        dcalt2,p2a = shp.dp_ddelta(k_a,P_a,zbar,pmodel='halofit')
        dcalt3,p3a = shp.dp_ddelta(k_a,P_a,zbar,pmodel='fastpt')
        ax.plot(k_a,abs(dcalt1/p1a))
        ax.plot(k_a,abs(dcalt2/p2a))
        ax.plot(k_a,abs(dcalt3/p3a))

        zbar = 2.
        ax = plt.subplot(222)
        plt.xlim([0.,0.4])
        plt.ylim([1.2,3.2])
        plt.grid()
        plt.title('z=2.0')
        dcalt1,p1a = shp.dp_ddelta(k_a,P_a,zbar,pmodel='linear')
        dcalt2,p2a = shp.dp_ddelta(k_a,P_a,zbar,pmodel='halofit')
        dcalt3,p3a = shp.dp_ddelta(k_a,P_a,zbar,pmodel='fastpt')
        ax.plot(k_a,abs(dcalt1/p1a))
        ax.plot(k_a,abs(dcalt2/p2a))
        ax.plot(k_a,abs(dcalt3/p3a))
        

        zbar = 1.
        ax = plt.subplot(223)
        plt.xlim([0.,0.4])
        plt.ylim([1.2,3.2])
        plt.grid()
        plt.title('z=1.0')
        dcalt1,p1a = shp.dp_ddelta(k_a,P_a,zbar,pmodel='linear')
        dcalt2,p2a = shp.dp_ddelta(k_a,P_a,zbar,pmodel='halofit')
        dcalt3,p3a = shp.dp_ddelta(k_a,P_a,zbar,pmodel='fastpt')
        ax.plot(k_a,abs(dcalt1/p1a))
        ax.plot(k_a,abs(dcalt2/p2a))
        ax.plot(k_a,abs(dcalt3/p3a))
        

        zbar = 0.
        ax = plt.subplot(224)
        plt.xlim([0.,0.4])
        plt.ylim([1.2,3.2])
        plt.grid()
        plt.title('z=0.0')
        dcalt1,p1a = shp.dp_ddelta(k_a,P_a,zbar,pmodel='linear')
        dcalt2,p2a = shp.dp_ddelta(k_a,P_a,zbar,pmodel='halofit')
        dcalt3,p3a = shp.dp_ddelta(k_a,P_a,zbar,pmodel='fastpt')
        ax.plot(k_a,abs(dcalt1/p1a))
        ax.plot(k_a,abs(dcalt2/p2a))
        ax.plot(k_a,abs(dcalt3/p3a))
        
        plt.legend(['linear','halofit','fastpt'],loc=4)
        plt.show()
        
if __name__=='__main__':
    PowerDerivativeComparison1()
