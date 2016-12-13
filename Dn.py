import numpy as np
from numpy import pi 
from hmf import ST_hmf
from nz_candel import NZCandel
import sys
from lw_observable import LWObservable
from fisher_matrix import fisher_matrix


class DNumberDensityObservable(LWObservable):
    def __init__(self,bins,geos,params,survey_id, C,basis,ddelta_bar_ddelta_alpha_list,nz_params): 
        
        '''
            data should hold: 
            the redshift and spatial regions 
            and the minimum masss
        
        '''
        print("Dn: initializing") 
        min_mass = params['M_cut']
        self.variable_cut = params['variable_cut']
        LWObservable.__init__(self,geos,params,survey_id,C)
        
        self.geo1 = geos[0]
        self.geo2 = geos[1]

        self.mf=ST_hmf(self.C)
        self.nzc = NZCandel(nz_params)
    
        bin1 = bins[0]
        bin2 = bins[1]
        self.basis = basis 
        self.bounds1 = self.geo1.fine_indices[bin1]
        self.bounds2 = self.geo2.fine_indices[bin2]
        range1 = range(self.bounds1[0],self.bounds1[1])
        range2 = range(self.bounds2[0],self.bounds2[1])

        #TODO check if interpolation needed
        if self.variable_cut: 
            self.n_avgs1 = self.nzc.get_nz(self.geo1)
            self.n_avgs2 = self.nzc.get_nz(self.geo2)
        else:
            self.n_avgs1 = np.zeros(self.geo1.z_fine.size)
            self.n_avgs2 = np.zeros(self.geo2.z_fine.size)
            for i in range(0,self.geo1.z_fine.size):
                self.n_avgs1[i] = self.mf.n_avg(min_mass,self.geo1.z_fine[i])
            for i in range(0,self.geo2.z_fine.size):
                self.n_avgs2[i] = self.mf.n_avg(min_mass,self.geo2.z_fine[i])

        V1 = self.geo1.volumes[bin1]
        V2 = self.geo2.volumes[bin2]

        if self.variable_cut:
            self.M_cuts1 = self.nzc.get_M_cut(self.mf,self.geo1)
            self.M_cuts2 = self.nzc.get_M_cut(self.mf,self.geo2)
            
        self.dn_ddelta_bar1 = np.zeros((self.geo1.z_fine.size))
        self.dn_ddelta_bar2 = np.zeros((self.geo2.z_fine.size))
        #self.DO_a=np.zeros(ddelta_bar_ddelta_alpha_list.size)
           
        #d1s = ddelta_bar_ddelta_alpha_list[0]
        #d2s = ddelta_bar_ddelta_alpha_list[1]

        #TODO vectorize, make sure to replace n_avgs with 0 outside range if necessary
        if self.variable_cut: 
            for i in range1:
                self.dn_ddelta_bar1[i]=self.mf.bias_avg(self.M_cuts1[i],self.geo1.z_fine[i])
            for i in range2:
                self.dn_ddelta_bar2[i]=self.mf.bias_avg(self.M_cuts2[i],self.geo2.z_fine[i])
        else: 
            for i in range1:
                self.dn_ddelta_bar1[i]=self.mf.bias_avg(min_mass,self.geo1.z_fine[i])
            for i in range2:
                self.dn_ddelta_bar2[i]=self.mf.bias_avg(min_mass,self.geo2.z_fine[i])
        print "Dn: getting d1,d2"
        #multiplier for integrand,TODO maybe better way
        self.integrand1 = np.expand_dims(self.dn_ddelta_bar1*self.geo1.r_fine**2,axis=1)
        self.d1=self.basis.D_O_I_D_delta_alpha(self.geo1,self.integrand1,use_r=True)/(self.geo1.r_fine[range1[1]]**3-self.geo1.r_fine[range1[0]]**3)*3.
        
        self.integrand2 = np.expand_dims(self.dn_ddelta_bar1*self.geo2.r_fine**2,axis=1)
        #self.d1 = self.basis.D_O_I_D_delta_alpha(self.geo1,self.dn_ddelta_bar1)#.T*self.geo1.r_fine**2*C.D_comov_dz(self.geo1.z_fine),use_r=True)/(self.geo1.r_fine[range1[1]]**3-self.geo1.r_fine[range1[0]]**3)*3.
        self.d2=self.basis.D_O_I_D_delta_alpha(self.geo2,self.integrand2,use_r=True)/(self.geo2.r_fine[range2[1]]**3-self.geo2.r_fine[range2[0]]**3)*3.
        #self.d2 = self.basis.D_O_I_D_delta_alpha(self.geo2,self.dn_ddelta_bar2)#.T*self.geo1.r_fine**2*C.D_comov_dz(self.geo1.z_fine),use_r=True)/(self.geo2.r_fine[range2[1]]**3-self.geo1.r_fine[range2[0]]**3)*3.
        self.DO_a = self.d2-self.d1
        
        #TODO handle correctly if n avg not same
        #TODO averaging w and wo trapezoidal rule different 
        #self.n_avg1 = np.trapz(self.n_avgs1[range1],self.geo1.z_fine[range1])/(self.geo1.z_fine[range1[-1]]-self.geo1.z_fine[range1[0]])
        #self.n_avg2 = np.trapz(self.n_avgs2[range2],self.geo2.z_fine[range2])/(self.geo2.z_fine[range2[-1]]-self.geo2.z_fine[range2[0]])
        self.n_avg1 = np.trapz((self.geo1.r_fine**2*self.n_avgs1)[range1],self.geo1.r_fine[range1])/(self.geo1.r_fine[range1[1]]**3-self.geo1.r_fine[range1[0]]**3)*3.
        self.n_avg2 = np.trapz((self.geo2.r_fine**2*self.n_avgs2)[range2],self.geo2.r_fine[range2])/(self.geo1.r_fine[range1[1]]**3-self.geo1.r_fine[range1[0]]**3)*3.
        #self.n_avg2 = np.average(self.n_avgs2[range2])
        

        #n_avg = self.mf.n_avg(min_mass,(z_min1+z_max1)/2.)

        #self.DO_a=self.dn_ddelta_bar*(d1s[bin1]-d2s[bin2])
          #  V1=volume(r_min[i-1],r_max[i-1],self.geo1)
          #  V2=volume(r_min[i-1],r_max[i-1],self.geo2)
        #TODO handle overlapping geometries 
        self.Nab=(self.n_avg1/V1+self.n_avg2/V2)
        #self.Nab=(1./V1+1./V2)*self.n_avg1
            
       
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
    
    
