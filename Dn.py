"""Sample implementation of a long wavelength mitigation strategy,
for the difference in galaxy number densities
between two survey geometries, as described in the paper"""
from __future__ import division,print_function,absolute_import
from builtins import range

from warnings import warn
import numpy as np
from scipy.ndimage import gaussian_filter1d

from hmf import ST_hmf
from nz_candel import NZCandel
from nz_wfirst import NZWFirst
from nz_lsst import NZLSST
from nz_constant import NZConstant
from lw_observable import LWObservable
from algebra_utils import trapz2
from polygon_geo import PolygonGeo
from pixel_geo import PixelGeo
from polygon_pixel_geo import PolygonPixelGeo
from polygon_union_geo import PolygonUnionGeo
from polygon_pixel_union_geo import PolygonPixelUnionGeo
#LSST sigma/(1+z)<0.05 required 0.02 goal for i<25.3 (lsst science book 3.8.1)
import fisher_matrix as fm
#NOTE if doing photozs tomo must go to z=0 or photoz uncertainty
#can actually add information because it provides information at z=0
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
        print("Dn: initializing")
        LWObservable.__init__(self,geos,params,survey_id,C)
        self.fisher_type = False
        self.basis = basis
        self.nz_select = params['nz_select']
        self.nz_params = nz_params

        #self.geo2 should be area in mitigation survey but not in original survey
        #TODO this is a bit hackish, but works for now
        if isinstance(geos[0],PolygonGeo):
            if isinstance(geos[1],PolygonGeo):
                self.geo2 = PolygonUnionGeo(np.array([geos[1]]),np.array([geos[0]]),zs=geos[1].zs,z_fine=geos[1].z_fine)
            elif isinstance(geos[1],PolygonUnionGeo):
                self.geo2 = PolygonUnionGeo(geos[1].geos,np.append(geos[0],geos[1].masks),zs=geos[1].zs,z_fine=geos[1].z_fine)
            else:
                raise ValueError('unsupported type for geo2')
        elif isinstance(self.geos[0],PixelGeo):
#            if isinstance(geos[1],PolygonPixelUnionGeo):
#                self.geo2 = PolygonPixelUnionGeo(geos[1].geos,np.append(geos[0],geos[1].masks),zs=geos[1].zs,z_fine=geos[1].z_fine)
            if isinstance(geos[1],PixelGeo):
                self.geo2 = PolygonPixelUnionGeo(np.array([geos[1]]),np.array([geos[0]]),zs=geos[1].zs,z_fine=geos[1].z_fine)
            else:
                raise ValueError('unsupported type for geo2')
            #if isinstance(geos[1],PolygonPixelGeo):
            #    self.geo2 = PolygonPixelUnionGeo(np.array([geos[1]]),np.array([geos[0]]))
            #else:
            #    raise ValueError('unsupported type for geo2')
        else:
            raise ValueError('unsupported type for geo1')

        #self.geo1 should be intersect of mitigation survey and original survey
        if np.isclose(geos[0].get_overlap_fraction(geos[1]),1.):
            self.geo1 = geos[0]
        else:
            raise RuntimeError('partial overlap not yet implemented')
#            if isinstance(geos[0],PolygonGeo):
#                raise RuntimeError('partial overlap not yet implemented')
#        #        self.geo1 = PolygonUnionGeo(np.array([geos[0]]),np.array([self.geo2]))
#            elif isinstance(self.geos[0],PolygonPixelGeo) or isinstance(self.geos[0],PolygonPixelUnionGeo):
#                self.geo1 = PolygonPixelUnionGeo(np.array([geos[0]]),np.array([self.geo2]))
#            else:
#                raise ValueError('unrecognized type for geo1')

        #should be using r bin structure of mitigation survey
        self.r_fine = self.geo2.r_fine
        self.z_fine = self.geo2.z_fine
        assert np.all(self.r_fine==geos[1].r_fine)
        assert np.all(self.z_fine==geos[1].z_fine)
        assert np.all(geos[0].z_fine==geos[1].z_fine)
        assert np.all(geos[0].r_fine==geos[1].r_fine)
        dz = self.z_fine[2]-self.z_fine[1]

        print(mf_params)
        self.mf = ST_hmf(self.C,params=mf_params)

        if self.nz_select == 'CANDELS':
            self.nzc = NZCandel(self.nz_params)
        elif self.nz_select == 'WFIRST':
            self.nzc = NZWFirst(self.nz_params)
        elif self.nz_select == 'LSST':
            self.nzc = NZLSST(self.z_fine,self.nz_params)
        elif self.nz_select == 'constant':
            self.nzc = NZConstant(self.geo2,self.nz_params)
        else:
            raise ValueError('unrecognized nz_select '+str(self.nz_select))

        self.n_bins = self.geo2.fine_indices.shape[0]


        self.n_avgs = self.nzc.get_nz(self.geo2)
        self.dNdzs = self.n_avgs/self.geo2.dzdr*self.r_fine**2
        self.M_cuts = self.nzc.get_M_cut(self.mf,self.geo2)
        self.dn_ddelta_bar = self.mf.bias_n_avg(self.M_cuts,self.z_fine)/C.h**3
        self.integrand = np.expand_dims(self.dn_ddelta_bar*self.r_fine**2,axis=1)
        #note effect of mitigation converged to ~0.3% if cut off integral at for z>1.5, 10% for z>0.6,20% for z>0.5
        self.r_vols = 3./np.diff(self.geo2.rbins**3)
        self.n_avg_integrand = self.r_fine**2*self.n_avgs

        self.Nab_i = np.zeros(self.n_bins)
        self.vs = np.zeros((self.n_bins,basis.get_size()))
        self.b_ns = np.zeros(self.n_bins)
        self.n_avg_bin = np.zeros(self.n_bins)
        self.bias = self.dn_ddelta_bar/self.n_avgs
        self.sigma0 = params['sigma0']
        self.z_extra = np.hstack([self.z_fine,np.arange(self.z_fine[-1]+dz,self.z_fine[-1]+params['n_extend']*self.sigma0*(1.+self.z_fine[-1]),dz)])
        self.integrands_smooth = np.zeros((self.z_fine.size,self.n_bins))

        self.V1s = np.diff(self.geo2.rs**3)/3.*self.geo1.angular_area()
        self.V2s = self.geo2.volumes
        assert np.all(self.V1s>=0.)
        assert np.all(self.V2s>=0.)

        #NOTE this whole loop could be pulled apart with a small change in sph_klim
        for itr in range(0,self.n_bins):
            bounds1 = self.geo2.fine_indices[itr]
            range1 = np.arange(bounds1[0],bounds1[1])

            print("Dn: getting d1,d2")
            #multiplier for integrand
            n_avg = self.r_vols[itr]*trapz2(self.n_avg_integrand[range1],self.r_fine[range1])
            self.n_avg_bin[itr] = n_avg

            assert n_avg>=0.

            if self.V1s[itr] == 0. or self.V2s[itr] == 0.:
                continue
            elif n_avg==0.:
                warn('Dn: variance had a value which was exactly 0; fixing inverse to np.inf '+str(itr))
                self.Nab_i[itr] = np.inf
            else:
                self.b_ns[itr] = self.r_vols[itr]*trapz2(self.integrand[range1],self.r_fine[range1])
                #need a bit extra z so does not reflect back from right boundary
                dN_wind = np.zeros(self.z_extra.size)
                dN_wind[range1] = self.dNdzs[range1]
                sigma = self.sigma0*(1.+self.geo2.zs[itr])/(self.z_fine[2]-self.z_fine[1])
                dN_smooth = gaussian_filter1d(dN_wind,sigma,mode='mirror',truncate=10.)
                dN_smooth = dN_smooth[0:self.z_fine.size]
                print("tot acc",np.trapz(dN_smooth,self.z_fine)/np.trapz(dN_wind,self.z_extra))
                print("outside",np.trapz(dN_smooth[range1],self.z_fine[range1])/np.trapz(dN_wind,self.z_extra))
                n_smooth = dN_smooth/self.r_fine**2*self.geo2.dzdr
                bn_smooth = n_smooth*self.bias
                integrand_smooth = np.expand_dims(bn_smooth*self.r_fine**2,axis=1)
                self.integrands_smooth[:,itr] = integrand_smooth[:,0]
                d1 = self.basis.D_O_I_D_delta_alpha(self.geo1,integrand_smooth,use_r=True)
                d2 = self.basis.D_O_I_D_delta_alpha(self.geo2,integrand_smooth,use_r=True)
                DO_a = (d2-d1)*self.r_vols[itr]
                Nab_itr = n_avg*(1./self.V1s[itr]+1./self.V2s[itr])
                self.Nab_i[itr] = 1./Nab_itr
                self.vs[itr] = DO_a.flatten()
            d1s_alt = self.basis.D_O_I_D_delta_alpha(self.geo1,self.integrands_smooth,use_r=True)
            d2s_alt = self.basis.D_O_I_D_delta_alpha(self.geo2,self.integrands_smooth,use_r=True)
            self.DO_a_alt = ((d2s_alt-d1s_alt).T*self.r_vols).T
            self.vs_alt = self.DO_a_alt.flatten()

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
