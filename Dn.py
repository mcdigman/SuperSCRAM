"""Sample implementation of a long wavelength mitigation strategy,
for the difference in galaxy number densities
between two survey geometries, as described in the paper"""

from warnings import warn
import numpy as np

from hmf import ST_hmf
from nz_candel import NZCandel
from nz_wfirst import NZWFirst
from nz_lsst import NZLSST
from lw_observable import LWObservable
from algebra_utils import trapz2
from polygon_geo import PolygonGeo
from polygon_pixel_geo import PolygonPixelGeo
from polygon_union_geo import PolygonUnionGeo
from polygon_pixel_union_geo import PolygonPixelUnionGeo

import fisher_matrix as fm
class DNumberDensityObservable(LWObservable):
    """An observable for the difference in galaxy number density between two bins"""
    def __init__(self,geos,params,survey_id,C,basis,nz_params,mf_params):
        """ inputs:
                geos: a numpy array of Geo objects
                        the [geo1,geo2], where geo2 will be used for mitigation
                        of covariance in geo1 geometry
                params: a dict of parameters,
                        nz_select: 'CANDELS','WFIRST','LSST' to use for NZMatcher
                survey_id: an id for the associated LWSurvey
                C: as CosmoPie object
                nz_params: parameters needed by NZMatcher object
                mf_params: params needed by ST_hmf
        """
        print "Dn: initializing"
        LWObservable.__init__(self,geos,params,survey_id,C)
        self.fisher_type = False
        self.basis = basis
        self.nz_select = params['nz_select']
        self.nz_params = nz_params

        #self.geo2 should be area in mitigation survey but not in original survey
        if isinstance(geos[0],PolygonGeo):
            self.geo2 = PolygonUnionGeo(np.array([geos[1]]),np.array([geos[0]]))
        elif isinstance(self.geos[0],PolygonPixelGeo):
            self.geo2 = PolygonPixelUnionGeo(np.array([geos[1]]),np.array([geos[0]]))
        else:
            raise ValueError('unrecognized type for geo1')

        #self.geo1 should be intersect of mitigation survey and original survey
        if np.isclose(geos[0].get_overlap_fraction(geos[1]),1.):
            self.geo1 = geos[0]
        #elif np.isclose(geos[0].get_overlap_fraction(geos[1]),0.):
        #    self.geo1 = PolygonUnionGeo(np.array([geos[0]]),np.array([geos[1]]))
        else:
            if isinstance(geos[0],PolygonGeo):
                raise RuntimeError('partial overlap not yet implemented')
        #        self.geo1 = PolygonUnionGeo(np.array([geos[0]]),np.array([self.geo2]))
            elif isinstance(self.geos[0],PolygonPixelGeo):
                self.geo1 = PolygonPixelUnionGeo(np.array([geos[0]]),np.array([self.geo2]))
            else:
                raise ValueError('unrecognized type for geo1')

        #should be using r bin structure of mitigation survey
        self.r_fine = self.geo2.r_fine
        self.z_fine = self.geo2.z_fine
        assert np.all(self.r_fine==geos[1].r_fine)
        assert np.all(self.z_fine==geos[1].z_fine)

        self.mf = ST_hmf(self.C,params=mf_params)

        if self.nz_select == 'CANDELS':
            self.nzc = NZCandel(self.nz_params)
        elif self.nz_select == 'WFIRST':
            self.nzc = NZWFirst(self.nz_params)
        elif self.nz_select == 'LSST':
            self.nzc = NZLSST(self.z_fine,self.nz_params)
        else:
            raise ValueError('unrecognized nz_select '+str(self.nz_select))

        self.n_bins = self.geo2.fine_indices.shape[0]


        self.n_avgs = self.nzc.get_nz(self.geo2)
        self.M_cuts = self.nzc.get_M_cut(self.mf,self.geo2)
        self.dn_ddelta_bar = self.mf.bias_n_avg(self.M_cuts,self.z_fine)/C.h**3 #TODO PRIORITY fix so not negative
        self.integrand = np.expand_dims(self.dn_ddelta_bar*self.r_fine**2,axis=1)
        #note effect of mitigation converged to ~0.3% if cut off integral at for z>1.5, 10% for z>0.6,20% for z>0.5
        r_vols = 3./np.diff(self.geo2.rbins**3)
        n_avg_integrand = self.r_fine**2*self.n_avgs

        self.Nab_i = np.zeros(self.n_bins)
        self.vs = np.zeros((self.n_bins,basis.get_size()))
        #self.n_avg_c = np.zeros(self.n_bins)
        #self.b_ns = np.zeros(self.n_bins)
        #self.b_avgs = np.zeros(self.n_bins)
        #self.m_avgs = np.zeros(self.n_bins)
        #self.biases = np.diag(self.mf.bias(self.M_cuts,self.z_fine))

        #NOTE this whole loop could be pulled apart with a small change in sph_klim
        for itr in xrange(0,self.n_bins):
            bounds1 = self.geo2.fine_indices[itr]
            range1 = np.arange(bounds1[0],bounds1[1])#np.array(range(bounds1[0],bounds1[1]))

            print "Dn: getting d1,d2"
            #multiplier for integrand
            V1 = self.geo1.volumes[itr]
            V2 = self.geo2.volumes[itr]
            assert V1>=0 and V2>=0


            #r_vol = 1./(self.r_fine[range1[-1]]**3-self.r_fine[range1[0]]**3)*3.
            #TODO check number densities sensible
            #TODO why does increasing n_avg DECREASE the information?
            #TODO should be a bin which goes to 0
            n_avg = r_vols[itr]*trapz2(n_avg_integrand[range1],self.r_fine[range1])
            #b_n = r_vols[itr]*trapz2(n_avg_integrand[range1],self.r_fine[range1])
            #bias = r_vols[itr]*trapz2(n_avg_integrand[range1],self.r_fine[range1])
            #self.n_avg_c[itr] = n_avg
         

            #self.b_ns[itr] = r_vols[itr]*trapz2(self.integrand[range1],self.r_fine[range1])
            #self.b_avgs[itr] = r_vols[itr]*trapz2(self.biases[range1]*self.r_fine[range1]**2,self.r_fine[range1])
            #self.m_avgs[itr] = r_vols[itr]*trapz2(self.M_cuts[range1]*self.r_fine[range1]**2,self.r_fine[range1])
            

            if V1 == 0 or V2 == 0:
                continue
            elif n_avg==0.:
                warn('Dn: variance had a value which was exactly 0; fixing inverse to np.inf '+str(itr))
                self.Nab_i[itr] = np.inf
            else:
                d1 = self.basis.D_O_I_D_delta_alpha(self.geo1,self.integrand,use_r=True,range_spec=range1)
                d2 = self.basis.D_O_I_D_delta_alpha(self.geo2,self.integrand,use_r=True,range_spec=range1)
                DO_a = (d2-d1)*r_vols[itr]
                Nab_itr = n_avg*(1./V1+1./V2)
                self.Nab_i[itr] = 1./Nab_itr
                self.vs[itr] = DO_a.flatten()

    def get_rank(self):
        """get the rank of the perturbation, ie the number of vectors summed in F=v^Tv"""
        return self.n_bins

#    def get_dO_a_ddelta_bar(self):
#        """get the observables response to a density fluctuation"""
#        return self.vs

    def get_fisher(self):
        """get the fisher matrix, cholesky is sqrt"""
        Nab_f = fm.FisherMatrix(np.diagflat(np.sqrt(self.Nab_i)),input_type=fm.REP_CHOL_INV)
        return Nab_f.project_fisher(self.vs)

    def get_perturbing_vector(self):
        """get decomposition of fisher matrix as F=sigma2*v^Tv"""
        return self.vs,self.Nab_i
