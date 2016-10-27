import numpy as np
from numpy import pi 
from hmf import ST_hmf
import sys
from lw_observable import LWObservable
from fisher_matrix import fisher_matrix

#def volume(r1,r2,geo):
#    if geo['type']='rectangular' :
#        
#    3
#    result=(phi2-phi1)*(np.cos(theta1)- np.cos(theta2))*(r2**3-r1**3)/3. 
#    return result 

class DNumberDensityObservable(LWObservable):
    def __init__(self,bins,geos,params,survey_id, C,basis,ddelta_bar_ddelta_alpha_list): 
        
        '''
            data should hold: 
            the redshift and spatial regions 
            and the minimum masss
        
        '''
     
        min_mass = params['M_cut']
        LWObservable.__init__(self,geos,params,survey_id,C)
        
        geo1 = geos[0]
        geo2 = geos[1]
    
        bin1 = bins[0]
        bin2 = bins[1]
        self.mf=ST_hmf(self.C)
        self.basis = basis 
    
        V1 = geo1.volumes[bin1]
        V2 = geo2.volumes[bin2]
        #self.DO_a=np.zeros(ddelta_bar_ddelta_alpha_list.size)
           
        #d1s = ddelta_bar_ddelta_alpha_list[0]
        #d2s = ddelta_bar_ddelta_alpha_list[1]
        z_min1 = geo1.zbins[bin1][0]
        z_max1 = geo1.zbins[bin1][1]
        z_fine1 = geo1.z_fine
        self.dn_ddelta_bar1 = np.zeros((z_fine1.size,1))
        for i in range(0,z_fine1.size):
            #z_avg=(geo1.zbins[bin1][0] + geo1.zbins[bin1][1])/2.
            
            z_i = z_fine1[i]
            if z_i< z_min1:
                continue
            if z_i>z_max1:
                break
            
            n_avg=self.mf.n_avg(min_mass, z_i)
            self.dn_ddelta_bar1[i]=self.mf.bias_avg(min_mass,z_i)
        d1 = self.basis.D_O_I_D_delta_alpha(geo1,self.dn_ddelta_bar1,use_r=False)

        z_min2 = geo2.zbins[bin2][0]
        z_max2 = geo2.zbins[bin2][1]
        z_fine2 = geo2.z_fine
        self.dn_ddelta_bar2 = np.zeros((z_fine2.size,1))
        for i in range(0,z_fine2.size):
            #z_avg=(geo1.zbins[bin1][0] + geo1.zbins[bin1][1])/2.
            
            z_i = z_fine2[i]
            if z_i< z_min2:
                continue
            if z_i>z_max2:
                break
            
            self.dn_ddelta_bar2[i]=self.mf.bias_avg(min_mass,z_i)
        d2 = self.basis.D_O_I_D_delta_alpha(geo2,self.dn_ddelta_bar2,use_r=False)
        self.DO_a = d2-d1

        n_avgs = np.zeros(z_fine1.size)
        for i in range(0,z_fine1.size):
            z_i = z_fine1[i]
            if z_i< z_min1:
                continue
            if z_i>z_max1:
                break

            n_avgs[i]=self.mf.n_avg(min_mass, z_i)

        n_avg = np.trapz(n_avgs,z_fine1)/(z_max1-z_min1)
        #n_avg = self.mf.n_avg(min_mass,(z_min1+z_max1)/2.)

        #self.DO_a=self.dn_ddelta_bar*(d1s[bin1]-d2s[bin2])
          #  V1=volume(r_min[i-1],r_max[i-1],geo1)
          #  V2=volume(r_min[i-1],r_max[i-1],geo2)
            
        self.Nab=(1./V1+1./V2)*(n_avg)
            
       
        self.v=self.DO_a
        #self.F_alpha_beta=np.outer(v,v)*1./self.Nab 
        
           
    #TODO check zbins        
   # def get_dO_a_ddelta_bar():
   #     return self.dn_ddelta_bar

    def get_dO_a_ddelta_bar(self):
        return self.DO_a
        
    def get_F_alpha_beta(self):
        return np.outer(self.v,self.v)*1./self.Nab
                
        

        
if __name__=="__main__":
    print 'hello'
    
    from defaults import cosmology 
    from cosmopie import C
    d=np.loadtxt('Pk_Planck15.dat')
    k=d[:,0]; P=d[:,1]
    
    Omega_s1=18000
    z_bins1=np.array([.1,.2,.3])
    N1=np.array([1e6, 5*1e7]) 
   
    
    S1={'name':'LSST', 'sky_frac': Omega_s1, 'z_bins': z_bins1, 'N': N1}
    
    Omega_s2=18000/2.
    z_bins2=np.array([.1,.2,.35,.5])
    N2=np.array([1e7, 5*1e7]) 
  
    S2={'name':'WFIRST', 'sky_frac': Omega_s2, 'z_bins': z_bins2, 'N': N2}
    
    n_obs_dict=[S1, S2]
    
    cp=CosmoPie(cosmology, k=k, P_lin=P)
    r_max=cp.D_comov(4)
    l_alpha=np.array([1,2,3,4,5])
    n_zeros=3
    
    from sph import sph_basis
    geo=sph_basis(r_max,l_alpha,n_zeros,k,P)
    
    O=DO_n(n_obs_dict,1e15,cp,geo)
    
    
