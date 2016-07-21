import numpy as np
from numpy import pi 
from hmf import ST_hmf
import sys

#def volume(r1,r2,geo):
#    if geo['type']='rectangular' :
#        
#    3
#    result=(phi2-phi1)*(np.cos(theta1)- np.cos(theta2))*(r2**3-r1**3)/3. 
#    return result 

class DO_n:
    def __init__(self, data,min_mass, CosmoPie,basis): 
        
        '''
            data should hold: 
            the redshift and spatial regions 
            and the minimum masss
        
        '''
        
        zbins=data['zbins']
        geo1=data['geo1']
        geo2=data['geo2']
            
        self.mf=ST_hmf(CosmoPie)
        n_zbins=zbins.size
        
    
        self.DO_a=np.zeros((n_zbins-1,basis.C_size))
        self.Nab=np.zeros(n_zbins-1)
           
         
        for i in range(1,n_zbins):
            z_avg=(zbins[i-1] + zbins[i])/2.
            
            n_avg=self.mf.n_avg(min_mass, z_avg)
            x=self.mf.bias_avg(min_mass,z_avg)
            print "x",x
            d1=basis.D_delta_bar_D_delta_alpha(geo1.rs[i-1],geo1.rs[i],geo1)
            d2=basis.D_delta_bar_D_delta_alpha(geo2.rs[i-1],geo2.rs[i],geo2)
            
            self.DO_a[i-1]=x*(d1-d2)
            V1 = geo1.volumes[i-1] 
            V2 = geo2.volumes[i-1] 
          #  V1=volume(r_min[i-1],r_max[i-1],geo1)
          #  V2=volume(r_min[i-1],r_max[i-1],geo2)
            
            self.Nab[i-1]=(1/V1+1/V2)/n_avg
            
       
        self.F_alpha_beta=np.zeros(zbins.size-1,dtype=object)
        for i in range(zbins.size-1):
            v=self.DO_a[i]
            self.F_alpha_beta[i]=np.outer(v,v)*1./self.Nab[i]
           
        
    def dn_ddelta(self):
        return self.DO_a
        
    def Fisher_alpha_beta(self):
        return self.F_alpha_beta 
                
        

        
if __name__=="__main__":
    print 'hello'
    
    from defaults import cosmology 
    from cosmopie import CosmoPie
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
    
    
