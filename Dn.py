"""Sample implementation of a long wavelength mitigation strategy, for the difference in galaxy number densities
between two survey geometries, as described in the paper"""

from warnings import warn
from hmf import ST_hmf
from nz_candel import NZCandel
from nz_lsst import NZLSST
from lw_observable import LWObservable
from algebra_utils import trapz2

import numpy as np
import fisher_matrix as fm

class DNumberDensityObservable(LWObservable):
    def __init__(self,geos,params,survey_id, C,basis,nz_params1,nz_params2):
        """
            An observable for the difference in galaxy number density between two bins
            inputs:
                geos: a numpy array of Geo objects
                params: a dict of parameters
                survey_id: an id for the associated LWSurvey
                C: as CosmoPie object
                nz_params1,nz_params2: parameters for the nz_matcher objects to get
        """
        print("Dn: initializing")
        min_mass = params['M_cut']
        self.variable_cut = params['variable_cut']
        LWObservable.__init__(self,geos,params,survey_id,C)
        self.fisher_type=False

        self.geo1 = geos[0]
        self.geo2 = geos[1]
        self.nz_params1=nz_params1
        self.nz_params2=nz_params2

        #there is a bug in spherical_geometry that causes overlap to fail if geometries are nested and 1 side is identical, handle this case unless they fix it
        try:
            self.overlap_fraction = self.geo1.get_overlap_fraction(self.geo2)
        except:
            warn('spherical_geometry overlap failed, assuming total overlap')
            if self.geo1.angular_area()<=self.geo2.angular_area():
                self.overlap_fraction = 1.
            else:
                self.overlap_fraction = self.geo2.angular_area()/self.geo1.angular_area()

        self.mf=ST_hmf(self.C)
        self.nzc_1 = NZCandel(self.nz_params1)
        self.nzc_2 = NZLSST(self.nzc_1.z_grid,self.nz_params2)
        self.n_bins = self.geo1.fine_indices.shape[0]
        if not self.n_bins==self.geo2.fine_indices.shape[0]:
            warn("size of geo1 "+str(self.n_bins)+" incompatible with size of geo2 "+str(self.geo2.fine_indices.shape[0]))
        self.Nab_i = np.zeros((self.n_bins,self.n_bins))
        self.vs = np.zeros((self.n_bins,basis.get_size()))


        #TODO check if interpolation needed
        if self.variable_cut:
            self.n_avgs1 = self.nzc_1.get_nz(self.geo1)
            self.n_avgs2 = self.nzc_2.get_nz(self.geo2)
        else:
            self.n_avgs1 = np.zeros(self.geo1.z_fine.size)
            self.n_avgs2 = np.zeros(self.geo2.z_fine.size)
            for i in xrange(0,self.geo1.z_fine.size):
                self.n_avgs1[i] = self.mf.n_avg(min_mass,self.geo1.z_fine[i])
            for i in xrange(0,self.geo2.z_fine.size):
                self.n_avgs2[i] = self.mf.n_avg(min_mass,self.geo2.z_fine[i])

        if self.variable_cut:
            self.M_cuts1 = self.nzc_1.get_M_cut(self.mf,self.geo1)
            self.M_cuts2 = self.nzc_2.get_M_cut(self.mf,self.geo2)

        for itr in xrange(0,self.n_bins):
            self.basis = basis
            self.bounds1 = self.geo1.fine_indices[itr]
            self.bounds2 = self.geo2.fine_indices[itr]
            range1 = np.array(range(self.bounds1[0],self.bounds1[1]))
            range2 = np.array(range(self.bounds2[0],self.bounds2[1]))


            V1 = self.geo1.volumes[itr]
            V2 = self.geo2.volumes[itr]
            VInt = self.geo1.volumes[itr]*self.overlap_fraction

            #TODO pull out of loop
            self.dn_ddelta_bar1 = np.zeros((range1.size))
            self.dn_ddelta_bar2 = np.zeros((range2.size))
            #self.DO_a=np.zeros(ddelta_bar_ddelta_alpha_list.size)

            #d1s = ddelta_bar_ddelta_alpha_list[0]
            #d2s = ddelta_bar_ddelta_alpha_list[1]

            if self.variable_cut:
                #for i in range1:
                self.dn_ddelta_bar1=self.mf.bias_n_avg(self.M_cuts1[range1],self.geo1.z_fine[range1])
                self.dn_ddelta_bar2=self.mf.bias_n_avg(self.M_cuts2[range2],self.geo2.z_fine[range2])
                #for i in range2:
                 #   self.dn_ddelta_bar2[i]=self.mf.bias_avg(self.M_cuts2[i],self.geo2.z_fine[i])
            else:
                mass_array1 = np.full(range1.shape,min_mass)
                self.dn_ddelta_bar1=self.mf.bias_n_avg(mass_array1,self.geo1.z_fine[range1])
                mass_array2 = np.full(range2.shape,min_mass)
                self.dn_ddelta_bar2=self.mf.bias_n_avg(mass_array2,self.geo2.z_fine[range2])
            print "Dn: getting d1,d2"
            #multiplier for integrand,TODO maybe better way
            self.integrand1 = np.expand_dims(self.dn_ddelta_bar1*self.geo1.r_fine[range1]**2,axis=1)
            self.d1=self.basis.D_O_I_D_delta_alpha(self.geo1,self.integrand1,use_r=True,range_spec=range1)/(self.geo1.r_fine[range1[-1]]**3-self.geo1.r_fine[range1[0]]**3)*3.

            self.integrand2 = np.expand_dims(self.dn_ddelta_bar1*self.geo2.r_fine[range2]**2,axis=1)
            self.d2=self.basis.D_O_I_D_delta_alpha(self.geo2,self.integrand2,use_r=True,range_spec=range2)/(self.geo2.r_fine[range2[-1]]**3-self.geo2.r_fine[range2[0]]**3)*3.
            self.DO_a = self.d2-self.d1

            self.n_avg1 = trapz2((self.geo1.r_fine**2*self.n_avgs1)[range1],self.geo1.r_fine[range1])/(self.geo1.r_fine[range1[-1]]**3-self.geo1.r_fine[range1[0]]**3)*3.
            self.n_avg2 = trapz2((self.geo2.r_fine**2*self.n_avgs2)[range2],self.geo2.r_fine[range2])/(self.geo2.r_fine[range2[-1]]**3-self.geo2.r_fine[range2[0]]**3)*3.


            #use min because assume all galaxies detected in the deeper survey, so np.min([self.n_avg1,self.n_avg2]) should be n_both
            #TODO ensure well behaved if overlap is total
            #TODO enable different type of cutoff for LSST like and WFIRST like surveys
            Nab_itr = self.n_avg1/V1+self.n_avg2/V2-2.*np.min([self.n_avg1,self.n_avg2])*VInt/(V1*V2)
            if Nab_itr == 0.:
                warn('Dn: variance had a value which was exactly 0; mitigation disabled for axis '+str(itr))
                self.Nab_i[itr,itr] = 0.
                self.vs[itr]=np.zeros(self.DO_a.flatten().shape)
            else:
                self.Nab_i[itr,itr]=1./Nab_itr
                self.vs[itr]=self.DO_a.flatten()


        self.Nab_f = fm.FisherMatrix(np.sqrt(self.Nab_i),input_type=fm.REP_CHOL_INV)

    def get_rank(self):
        """get the rank of the perturbation, ie the number of vectors summed in F=v^Tv"""
        return self.n_bins

    def get_dO_a_ddelta_bar(self):
        """get the observables response to a density fluctuation"""
        return self.DO_a

    def get_fisher(self):
        """get the fisher matrix"""
        return self.Nab_f.project_fisher(self.vs)

    def get_perturbing_vector(self):
        """get decomposition of fisher matrix as F=v^Tv"""
        return np.dot( np.sqrt(self.Nab_i),self.vs)


#if __name__=="__main__":
#    print 'hello'
#
#    from defaults import cosmology
#    from cosmopie import CosmoPie
#    d=np.loadtxt('Pk_Planck15.dat')
#    k=d[:,0]; P=d[:,1]
#
#    Omega_s1=18000
#    z_bins1=np.array([.1,.2,.3])
#    N1=np.array([1e6, 5*1e7])
#
#
#    S1={'name':'LSST', 'sky_frac': Omega_s1, 'z_bins': z_bins1, 'N': N1}
#
#    Omega_s2=18000/2.
#    z_bins2=np.array([.1,.2,.35,.5])
#    N2=np.array([1e7, 5*1e7])
#
#    S2={'name':'WFIRST', 'sky_frac': Omega_s2, 'z_bins': z_bins2, 'N': N2}
#
#    n_obs_dict=[S1, S2]
#
#    cp=CosmoPie(cosmology, k=k, P_lin=P)
#    r_max=cp.D_comov(4)
#    l_alpha=np.array([1,2,3,4,5])
#    n_zeros=3
#
#    from sph_klim import SphBasisK
#    geo=SphBasisK(r_max,l_alpha,n_zeros,k,P)
#    #TODO write demo
#    #O=DO_n(n_obs_dict,1e15,cp,geo)
