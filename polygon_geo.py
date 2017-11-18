"""Module for a spherical polygon Geo, defined by vertices with great circle edges"""
from warnings import warn
import numpy as np
import scipy as sp
import spherical_geometry.vector as sgv
from spherical_geometry.polygon import SphericalPolygon
import spherical_geometry.great_circle_arc as gca
#from polygon_pixel_geo import get_Y_r_dict
from alm_utils_mpmath import get_Y_r_dict,get_Y_r_dict_central 
import alm_utils as au
from geo import Geo
import defaults
#get an exact spherical polygon geo
class PolygonGeo(Geo):
    #TODO check order of thetas,phis
    #vertices should be specified such that the clockwise oriented contour contains the area
    #theta_in and phi_in do not actually determine the inside, but are necessary for now if generating intersects with SphericalPolygon
    def __init__(self,zs,thetas,phis,theta_in,phi_in,C,z_fine,l_max,poly_params=defaults.polygon_params):
        """create a spherical polygon defined by vertices
                inputs:
                    zs: the tomographic z bins
                    thetas,phis: an array of theta values for the edges in radians, last value should be first for closure, edges will be clockwise
                    theta_in,phi_in: a theta and phi known to be outside, needed for finding intersect for now
                    C: a CosmoPie object
                    z_fine: the resolution z slices
                    l_max: the maximum l to compute the alm table to
                    poly_params: a dict of parameters
        """

        self.poly_params = poly_params
        self.n_double = poly_params['n_double']
        self.theta_in = theta_in
        self.phi_in = phi_in
        self.l_max = l_max

        #maximum alm already available, only a00 available at start
        self._l_max = 0
        self.n_v = thetas.size-1

        self.bounding_theta = thetas-np.pi/2. #to radec
        self.bounding_phi = np.pi-phis
        self.bounding_xyz = np.asarray(sgv.radec_to_vector(self.bounding_phi,self.bounding_theta,degrees=False)).T

        #this gets correct internal angles with specified vertex order (copied from spherical_polygon clas)
        self.internal_angles = 2.*np.pi-np.hstack([gca.angle(self.bounding_xyz[:-2], self.bounding_xyz[1:-1], self.bounding_xyz[2:], degrees=False),gca.angle(self.bounding_xyz[-2], self.bounding_xyz[0], self.bounding_xyz[1], degrees=False)])

        Geo.__init__(self,zs,C,z_fine)

        self.sp_poly = get_poly(thetas,phis,theta_in,phi_in)
        print "PolygonGeo: area calculated by SphericalPolygon: "+str(self.sp_poly.area())
        print "PolygonGeo: area calculated by PolygonGeo: "+str(self.angular_area())+" sr or "+str(self.angular_area()*(180./np.pi)**2)+" deg^2"

        self.alm_table = {(0,0):self.angular_area()/np.sqrt(4.*np.pi)}


        self.z_hats = np.zeros((self.n_v,3))
        self.y_hats = np.zeros_like(self.z_hats)
        self.xps = np.zeros_like(self.z_hats)

        self.betas = np.zeros(self.n_v)
        self.betas_alt = np.zeros(self.n_v)
        self.theta_alphas = np.zeros(self.n_v)
        self.omega_alphas = np.zeros(self.n_v)
        self.gamma_alphas = np.zeros(self.n_v)

        self.internal_angles1 = np.zeros(self.n_v)
        self.internal_angles2 = np.zeros(self.n_v)
        self.internal_angles3 = np.zeros(self.n_v)
        self.internal_angles4 = np.zeros(self.n_v)
        self.internal_angles5 = np.zeros(self.n_v)
        self.angle_opps = np.zeros(self.n_v)

        for itr1 in xrange(0,self.n_v):
            itr2 = itr1+1
            pa1 = self.bounding_xyz[itr1] #vertex 1
            pa2 = self.bounding_xyz[itr2] #vertex 2
            cos_beta12 = np.dot(pa1,pa2) #cos of angle between pa1 and pa2
            #TODO check consistent beta12 and beta12_alt
            self.betas[itr1] = np.arccos(cos_beta12) #angle between pa1 and pa2
            cross_12 = np.cross(pa2,pa1)
            sin_beta12 = np.linalg.norm(cross_12) #magnitude of cross product
            assert(np.isclose(sin_beta12,np.sin(self.betas[itr1])))
            self.betas_alt[itr1] = np.arcsin(sin_beta12)
            #TODO  don't use isclose here
            if np.isclose(self.betas[itr1],0.):
                print "PolygonGeo: side length 0, directions unconstrained, picking directions arbitrarily"
                #z_hat is not uniquely defined here so arbitrarily pick one orthogonal to pa1, #TODO: probably a better way to do this
                arbitrary = np.zeros(3)
                if not np.isclose(np.abs(pa1[0]),1.):
                    arbitrary[0] = 1.
                elif not np.isclose(np.abs(pa1[1]),1.):
                    arbitrary[1] = 1.
                else:
                    arbitrary[2] = 1.
                cross_12 = np.cross(arbitrary,pa1)
                sin_beta12 = np.linalg.norm(cross_12)
            #TODO check if isclose is right
            elif np.isclose(self.betas[itr1],np.pi):
                raise RuntimeError("PolygonGeo: Spherical polygons with sides of length pi are not uniquely determined")
            self.z_hats[itr1] = cross_12/sin_beta12 #direction of cross product
            #three euler rotation angles #TODO check, especially signs
            if not (np.isclose(self.z_hats[itr1,1],0.) and np.isclose(self.z_hats[itr1,0],0.)):
                self.theta_alphas[itr1] = -np.arccos(self.z_hats[itr1,2])
                y1 = np.cross(self.z_hats[itr1],pa1)
                self.y_hats[itr1] = y1
                assert(np.allclose(pa1*np.cos(self.betas[itr1])-y1*np.sin(self.betas[itr1]),pa2))
                assert(np.allclose(np.cross(pa1,y1),self.z_hats[itr1]))
                self.xps[itr1] = np.array([self.z_hats[itr1][1]*pa1[0]-self.z_hats[itr1][0]*pa1[1],self.z_hats[itr1][1]*y1[0]-self.z_hats[itr1][0]*y1[1],0.])

                #self.gamma_alphas[itr1] = np.arctan2(self.z_hats[itr1,1],self.z_hats[itr1,0])-np.pi/2.
                self.gamma_alphas[itr1] = np.mod(np.arctan2(-self.z_hats[itr1,0],self.z_hats[itr1,1]),2.*np.pi)
                self.omega_alphas[itr1] = -np.arctan2(self.xps[itr1,1],self.xps[itr1,0])
                #self.omega_alphas[itr1] = np.mod(np.arccos(np.dot(pa1,np.array([np.cos(self.gamma_alphas[itr1]),np.sin(self.gamma_alphas[itr1]),0]))),2.*np.pi)
            else:
                self.omega_alphas[itr1] = 0.
                self.gamma_alphas[itr1] = np.mod(np.arctan2(pa1[1],pa1[0]),2.*np.pi)
                #need to handle the case where z||z_hat separately (so don't divide by 0)
                if self.z_hats[itr1,2]<0:
                    print "PolygonGeo: setting theta_alpha to pi at "+str(itr1)
                    #TODO check
                    self.theta_alphas[itr1] = np.pi
                else:
                    print "PolygonGeo: setting theta_alpha to 0 at "+str(itr1)
                    self.theta_alphas[itr1] = 0.


        self.expand_alm_table(l_max)
        print "PolygonGeo: finished initialization"

    #use angle defect formula (Girard's theorem) to get a00
    def angular_area(self):
        return np.sum(self.internal_angles)-(self.n_v-2.)*np.pi

    #TODO test: probably bug somehow is giving float to get_Y_r_dict
    def expand_alm_table(self,l_max):
        ls = np.arange(np.int(self._l_max)+1,np.int(l_max)+1)
        n_l = ls.size
        if n_l==0:
            return
        d_alm_table1 = np.zeros(n_l,dtype=object)
        Y_r_dict = get_Y_r_dict(np.max(ls),np.zeros(1)+np.pi/2.,np.zeros(1))
        #Y_r(l,m,pi/2.,0.) has an analytic solution, use it
        Y_r_dict2 = get_Y_r_dict_central(l_max)
        #assert(np.allclose(Y_r_dict.values(),Y_r_dict2.values()))
        #self.factorials = sp.misc.factorial(np.arange(0,2*np.max(ls)+1))
        for l_itr in xrange(0,ls.size):
            ll=ls[l_itr]
            d_alm_table1[l_itr] = np.zeros((2*ll+1,self.n_v))
            for mm in xrange(0,ll+1):
                if mm>ll-1:
                    prefactor = 0.
                else:
                    #if ll-mm is odd or mm<0, then Y_r(ll,mm,pi/2,0)=0 anayltically, enforce
                    if mm<0 or  (mm-ll-1) % 2 == 1:
                        prefactor = 0.
                    else:
                        prefactor = (-1)**mm*np.sqrt((4.*ll**2-1.)*(ll-mm)*(ll+mm)/2.)/(ll*(-1+ll+2*ll**2))*Y_r_dict[(ll-1,mm)]
                #else:
                #    prefactor=0
                if not np.all(np.isfinite(prefactor)):
                    #print ll,mm,Y_r_dict[(ll-1,mm)]
                    raise ValueError('prefactor is nan at l='+str(ll)+',m='+str(mm))
                if mm==0:
                    d_alm_table1[l_itr][ll+mm] = np.sqrt(2.)*prefactor*self.betas
                else:
                    d_alm_table1[l_itr][ll+mm] = prefactor*np.sqrt(2.)/mm*np.sin(self.betas*mm)
                    d_alm_table1[l_itr][ll-mm] = prefactor*np.sqrt(2.)/mm*(1.-np.cos(self.betas*mm))
                #if (ll-mm) % 2 == 1:
                    #d_alm_table1[l_itr][ll+mm]*= 0.
                    #d_alm_table1[l_itr][ll-mm]*= 0.
            #if l_itr % 2 ==1:
            #    print "zerocand",d_alm_table1[l_itr] 
                #d_alm_table1[l_itr]=np.zeros((2*ll+1,self.n_v)) 

        d_alm_table2 = au.rot_alm_z(d_alm_table1,self.omega_alphas,ls)
        d_alm_table3 = au.rot_alm_x(d_alm_table2,self.theta_alphas,ls,n_double=self.n_double)
        d_alm_table4 = au.rot_alm_z(d_alm_table3,self.gamma_alphas,ls)

        for l_itr in xrange(0,ls.size):
            ll = ls[l_itr]
            for mm in xrange(0,ll+1):
                if mm==0:
                    self.alm_table[(ll,mm)] = np.sum(d_alm_table4[l_itr][ll+mm])
                else:
                    self.alm_table[(ll,mm)] = np.sum(d_alm_table4[l_itr][ll+mm])
                    self.alm_table[(ll,-mm)] =np.sum(d_alm_table4[l_itr][ll-mm])
        self._l_max = l_max

    def a_lm(self,l,m):

        if l>self._l_max:
            print "PolygonGeo: l value "+str(l)+" exceeds maximum precomputed l "+str(self._l_max)+",expanding table"
            self.expand_alm_table(l)

        alm = self.alm_table.get((l,m))
        if alm is None:
            raise RuntimeError('PolygonGeo: a_lm generation failed for unknown reason at l='+str(l)+',m='+str(m))
        return alm

    #TODO make robust for type of poly2
    #TODO avoid relying on SphericalPolygon
    def get_overlap_fraction(self,geo2):
        #there is a bug in spherical_geometry that causes overlap to fail if geometries are nested and 1 side is identical, handle this case unless they fix it
        try:
            result = self.sp_poly.overlap(geo2.sp_poly)
        except:
            warn('spherical_geometry overlap failed, assuming total overlap')
            if self.angular_area()<=geo2.angular_area():
                result = 1.
            else:
                result = geo2.angular_area()/self.angular_area()
        return result

    def surface_integral(self,function):
        raise  NotImplementedError, "Implement this if anything actually needs it"

#Note these are spherical polygons so all the sides are great circles (not lines of constant theta!)
#So area will differ from integral if assuming constant theta
#vertices must have same first and last coordinate so polygon is closed
#last point is arbitrary point inside because otherwise 2 polygons possible.
#Behavior may be unpredictable if the inside point is very close to an edge or vertex.
def get_poly(theta_vertices,phi_vertices,theta_in,phi_in):
    bounding_theta = theta_vertices-np.pi/2. #to radec
    bounding_phi = phi_vertices
    bounding_xyz = np.asarray(sgv.radec_to_vector(bounding_phi,bounding_theta,degrees=False)).T
    inside_xyz = np.asarray(sgv.radec_to_vector(phi_in,theta_in-np.pi/2.,degrees=False))

    sp_poly = SphericalPolygon(bounding_xyz,inside=inside_xyz)
    return sp_poly

def check_mutually_orthonormal(vectors):
    fails = 0
    for itr1 in xrange(0,vectors.shape[0]):
        for itr2  in xrange(0,vectors.shape[0]):
            prod =np.sum(vectors[itr1]*vectors[itr2],axis=1)
            if itr1==itr2:
                if not np.allclose(prod,np.zeros(vectors[itr1].shape[0])+1.):
                    warn("normality failed on vector pair: "+str(itr1)+","+str(itr2))
                    print prod
                    fails+=1
            else:
                if not np.allclose(prod,np.zeros(vectors[itr1].shape[0])):
                    warn("orthogonality failed on vector pair: "+str(itr1)+","+str(itr2))
                    print prod
                    fails+=1
    return fails
if __name__=='__main__':
    from cosmopie import CosmoPie

    poly_params = defaults.polygon_params.copy()
    l_max = 20
    zs = np.array([0.01,1.01])
    z_fine = np.arange(0.01,1.05,0.01)
    C = CosmoPie(defaults.cosmology)

    #phis = np.array([-19.,7.,25.,-19.])/180.*np.pi
    #thetas = np.pi/2.+np.array([-50,-35.,-55.,-50.])/180.*np.pi
    #phis = np.array([-19.,7.,25.,-19.])/180.*np.pi
    #thetas = np.pi/2.+np.array([-50,-35.,-55.,-50.])/180.*np.pi
    #phis = np.array([-19.,7.,-11.,7.,25.,5.,24.,43.,-19.])/180.*np.pi
    #thetas = np.array([-50.,-55.,-35.,-35.,-55.,-78.,-78.-78.,-55.,-50.])/180.*np.pi+np.pi/2.
    #phis = np.array([-19.,7.,7.,25.,-11.,7.,7.,25.,25.,43.,5.,24.,24.,50.,43.,50.,-19.])*np.pi/180.
    #thetas = np.array([-50.,-35.,-55.,-35.,-35.,-20.,-35.,-19.,-55.,-15.8,-78.,-55.,-78.,-55.,-55.,-40.,-50.])*np.pi/180.+np.pi/2.
    #phis = np.array([-19.,7.,25.,-19.])/180.*np.pi
    #thetas = np.pi/2.+np.array([-50,-35.,-55.,-50.])/180.*np.pi
    #theta_in = np.pi/2.-40./180.*np.pi
    #phi_in = 0./180.*np.pi
    #phis = np.array([-11.,7.,25.,43.,-11.])/180.*np.pi
    #thetas = np.array([-20.,-35.,-19.
    phis = np.array([-19.,-19.,-11.,-11.,7.,25.,25.,43.,43.,50.,50.,50.,24.,5.,5.,7.,7.,-19.])*np.pi/180.
    thetas = np.array([-50.,-35.,-35.,-19.,-19.,-19.,-15.8,-15.8,-40.,-40.,-55.,-78.,-78.,-78.,-55.,-55.,-50.,-50.])*np.pi/180.+np.pi/2.
    phi_in = 7./180.*np.pi
    theta_in = -35.*np.pi/180.+np.pi/2.
    poly_geo = PolygonGeo(zs,thetas,phis,theta_in,phi_in,C,z_fine,l_max=l_max,poly_params=poly_params)
    
    n_fill = 20
    theta_high = np.pi/2.+5.*np.pi/180.
    theta_low = np.pi/2.-65.*np.pi/180.
    theta_high_fill = np.full(n_fill,theta_high)
    theta_low_fill = np.full(n_fill,theta_low)
    theta2s = np.hstack([[theta_high],theta_high_fill,[theta_high,theta_low],theta_low_fill,[theta_low,theta_high]])
    #phi_high1 = 252.7*np.pi/180.-3.*np.pi
    #phi_high2 = 232.5*np.pi/180.-np.pi
    #phi_low1 = 181.5*np.pi/180.-np.pi#phi_high2
    #phi_low2 = 223.*np.pi/180.-3*np.pi
    #phi_high1 = 186.3285*np.pi/180.-2.*np.pi
    #phi_high2 = 174.67*np.pi/180.-0.*np.pi
    #phi_low1 = 174.67*np.pi/180.-0.*np.pi#phi_high2
    #phi_low2 = 186.3285*np.pi/180.-2*np.pi
#    phi_high1 = 160.*np.pi/180.-2.*np.pi
#    phi_high2 = 160.*np.pi/180.-0.01
    #phi_low1 = 160.*np.pi/180.-0.01
    #phi_low2 = 160.*np.pi/180.-2.*np.pi
#    phi_high_fill = np.linspace(phi_high1,phi_high2,n_fill+2)[1:-1]
    #phi_low_fill = phi_high_fill[::-1]
    #phi_low_fill = np.linspace(phi_low1,phi_low2,n_fill+2)[1:-1]
#    phi2s = np.hstack([[phi_high1],phi_high_fill,[phi_high2,phi_low1],phi_low_fill[::-1],[phi_low2,phi_high1]])-np.pi/2.
    #    theta2s = np.array([np.pi/4.,3.*np.pi/4.,3*np.pi/4.,np.pi/4.,np.pi/4.])
    #    phi2s = np.array([0.,0.,3.0740962890559151,3.0740962890559151,0.])-3.0740962890559151/2.
    #phi2s*=3.0981128
#    theta_in2 = 3.*np.pi/8.
#    phi_in2 = 0.
    #phi2_high1_d = =160.-360.
    #phi2_high2_d = =160.-0.01
    theta2r_high_fill = np.full(n_fill,5.)
    theta2r_low_fill =np.full(n_fill, -65.)
    phi2r_high_fill = np.linspace(180.-360.,180.-10.,n_fill)
    phi2r_low_fill = phi2r_high_fill[::-1] 
    theta2rs = np.hstack([theta2r_high_fill,theta2r_low_fill,theta2r_high_fill[0]])
    phi2rs = np.hstack([phi2r_high_fill,phi2r_low_fill,phi2r_high_fill[0]])

    theta2s= np.zeros_like(theta2rs)
    phi2s= np.zeros_like(theta2rs)
    from astropy.coordinates import SkyCoord
    for itr in xrange(0,theta2rs.size):
        coord_gal = SkyCoord(phi2rs[itr], theta2rs[itr], frame='icrs', unit='deg')
        theta2s[itr] = coord_gal.geocentrictrueecliptic.lat.rad+np.pi/2. 
        phi2s[itr] = coord_gal.geocentrictrueecliptic.lon.rad
    theta_in2= SkyCoord(0.,0.,frame='icrs',unit='deg').geocentrictrueecliptic.lat.rad+np.pi/2.
    phi_in2 = SkyCoord(0.,0.,frame='icrs',unit='deg').geocentrictrueecliptic.lon.rad
     
    poly_geo2 = PolygonGeo(zs,theta2s,phi2s,theta_in2,phi_in2,C,z_fine,l_max=l_max,poly_params=poly_params)

    thetar_high_fill = np.full(n_fill,20.)
    thetar_low_fill =np.full(n_fill, -20.)
    phir_high_fill = np.linspace(160.-360.,160.-20.,n_fill)
    phir_low_fill = np.linspace(160.-360.,160.-20.,n_fill)[::-1]
    thetars = np.hstack([thetar_high_fill,thetar_low_fill,thetar_high_fill[0]])
    phirs = np.hstack([phir_high_fill,phir_low_fill,phir_high_fill[0]])

    thetas_mask= np.zeros_like(thetars)
    phis_mask= np.zeros_like(thetars)
    for itr in xrange(0,thetars.size):
        coord_gal = SkyCoord(phirs[itr], thetars[itr], frame='galactic', unit='deg')
        thetas_mask[itr] = coord_gal.geocentrictrueecliptic.lat.rad+np.pi/2. 
        phis_mask[itr] = coord_gal.geocentrictrueecliptic.lon.rad
    theta_in_mask = SkyCoord(0.,0.,frame='galactic',unit='deg').geocentrictrueecliptic.lat.rad+np.pi/2.
    phi_in_mask = SkyCoord(0.,0.,frame='galactic',unit='deg').geocentrictrueecliptic.lon.rad
    mask_geo = PolygonGeo(zs,thetas_mask,phis_mask,theta_in_mask,phi_in_mask,C,z_fine,l_max=l_max,poly_params=poly_params)
    
    import polygon_union_geo as pug
    union_geo = pug.PolygonUnionGeo(np.array([poly_geo2]),np.array([mask_geo],dtype=object)) 

    import matplotlib.pyplot as plt
    from mpl_toolkits.basemap import Basemap
    m  = Basemap(projection='moll',lon_0=0)
    do_plot1 = False
    if do_plot1:
       # poly_geo.sp_poly.draw(m,color='red')
        poly_geo2.sp_poly.draw(m,color='blue')
        mask_geo.sp_poly.draw(m,color='green')
        union_geo.union_mask.draw(m,color='red')
        plt.show()

    do_plot3 = False
    if do_plot3:
        #poly_geo.sp_poly.draw(m,color='red')
        poly_geo2.sp_poly.draw(m,color='blue')
        mask_geo.sp_poly.draw(m,color='red')
        union_geo.union_geo.sp_poly.draw(m,color='green')
        plt.show()

    do_reconstruct = True
    if do_reconstruct:
        from alm_utils import reconstruct_from_alm
        from polygon_pixel_geo import PolygonPixelGeo
        import matplotlib.colors as colors
        pp_geo2 = PolygonPixelGeo(zs,theta2s,phi2s,theta_in2,phi_in2,C,z_fine,l_max=l_max,res_healpix=6)
        pp_geo = PolygonPixelGeo(zs,thetas,phis,theta_in,phi_in,C,z_fine,l_max=l_max,res_healpix=6)
        pp_mask_geo = PolygonPixelGeo(zs,thetas_mask,phis_mask,theta_in_mask,phi_in_mask,C,z_fine,l_max=l_max,res_healpix=6)
        #mask = ((pp_geo2.contained*1.-pp_mask_geo.contained*1.)>0)
        from polygon_pixel_geo import get_healpix_pixelation,contains_points
        all_pixels = get_healpix_pixelation(res_choose=6)
        mask =  contains_points(all_pixels,union_geo.union_mask)
        #union_geo.union_mask.draw(m,color='red')
        totals_poly = reconstruct_from_alm(l_max,pp_geo2.all_pixels[:,0],pp_geo2.all_pixels[:,1],union_geo.alm_table.copy())
        print "mean squared reconstruction error/point = ",np.linalg.norm(totals_poly-mask)/mask.size
    #    m = Basemap(projection='moll',lon_0=0)
        lats = (pp_geo2.all_pixels[:,0]-np.pi/2.)*180/np.pi
        lons = pp_geo2.all_pixels[:,1]*180/np.pi
        x,y=m(lons,lats)
        #have to switch because histogram2d considers y horizontal, x vertical
        fig = plt.figure(figsize=(10,5))
        #minC = np.min([totals_poly,totals_pp])
        #maxC = np.max([totals_poly,totals_pp])
        minC =np.min(totals_poly)
        maxC = np.max(totals_poly)
        bounds = np.linspace(minC,maxC,10)
        norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
        ax = fig.add_subplot(121)
        H1,yedges1,xedges1 = np.histogram2d(y,x,100,weights=totals_poly)
        X1, Y1 = np.meshgrid(xedges1, yedges1)
        pc1 = ax.pcolormesh(X1,Y1,-H1,cmap='gray')
        ax.set_title("PolygonGeo reconstruction")
        ax.set_aspect('equal')
        #fig.colorbar(pc1,ax=ax)
        #m.plot(x,y,'bo',markersize=1)
        pp_geo2.sp_poly.draw(m,color='blue')
        do_poly=True
        if do_poly:
            ax = fig.add_subplot(122)
            H2,yedges2,xedges2 = np.histogram2d(y,x,100,weights=mask)
            X2, Y2 = np.meshgrid(xedges2, yedges2)
            ax.pcolormesh(X2,Y2,-H2,cmap='gray')
            ax.set_aspect('equal')
            #m.plot(x,y,'bo',markersize=1)
            ax.set_title("PolygonPixelGeo mask")
            union_geo.union_mask.draw(m,color='red')
            #pp_geo2.sp_poly.draw(m,color='red')
        plt.show()
    from astropy import units as u
    from astropy.coordinates import SkyCoord
#if __name__=='__main__':
#    from polygon_pixel_geo import PolygonPixelGeo,reconstruc_from_alm
#    from geo import RectGeo
#    from cosmopie import CosmoPie
#    from time import time
#    toffset = np.pi/2.
#    #theta0=np.pi/6+toffset
#    #theta1=np.pi/3.+toffset
#    #theta0=2.82893228356285
#    theta0=np.pi-np.arccos(1.-5.*np.pi/324.)
#    #theta0=(1.25814+toffset)
#    theta1=np.pi/2.+toffset
#    theta2=np.pi/3.+0.1+toffset
#    theta3=theta2-np.pi/3.
#    theta4 = theta3+np.pi/6.
#    offset=0.
#    phi0=0.+offset
#    phi1 = 4.*np.pi/3.+offset
#    #phi1 = np.pi/3.+offset
#    phi2 = phi1+np.pi/2.
#    phi3 = phi2-np.pi/6.
#    phi4 = phi3-np.pi/3.
#    n_steps = 120
#    thetas = np.zeros(n_steps+2)+theta0
#    phis = np.zeros(n_steps+2)
#    phis[0] = phi0
#    phis[-1] = phi0
#    for itr in xrange(1,n_steps+1):
#        phis[itr] = phi0+itr*2.*np.pi/n_steps
#    phis = phis[::-1]
#    #thetas = np.array([theta0,theta0,theta0,theta0,theta0,theta0,theta0,theta0])
#    #phis = np.array([phi0,phi0+np.pi/3.,phi0+2.*np.pi/3.,phi0+np.pi,phi0+4.*np.pi/3.,phi0+5.*np.pi/3.,6.*np.pi/3.,phi0])[::-1]
#    #thetas = np.array([theta0,theta1,theta1,theta0,theta0])
#    #phis = np.array([phi0,phi0,phi1,phi1,phi0])
#    #thetas = np.array([theta0,theta1,theta1,theta0,theta0])
#    #phis = np.array([phi0,phi0,phi1,phi1,phi0])
#    #thetas = np.array([theta0,theta1,theta1,theta2,theta0,theta3,theta3,theta4,theta0])
#    #phis = np.array([phi0,phi0,phi1,phi1,phi2,phi3,phi4,phi4,phi0])
#    #theta_in = np.pi/4.+toffset
#    #phi_in = np.pi/6.+offset
#    theta_in = theta0+0.1
#    phi_in = phi0-0.1
#    res_choose = 6
#    res_choose2 = 7
#    l_max = 50
#
#
#    #some setup to make an actual geo
#    d=np.loadtxt('camb_m_pow_l.dat')
#    k=d[:,0]; P=d[:,1]
#    C=CosmoPie(k=k,P_lin=P,cosmology=defaults.cosmology)
#    zs=np.array([.01,1.01])
#    z_fine = np.arange(defaults.lensing_params['z_min_integral'],np.max(zs),defaults.lensing_params['z_resolution'])
#
#
#    t0 = time()
#    poly_params = defaults.polygon_params.copy()
#    poly_geo = PolygonGeo(zs,thetas,phis,theta_in,phi_in,C,z_fine,l_max=l_max,poly_params=poly_params)
#    t1 = time()
#    print "PolygonGeo initialized in time: "+str(t1-t0)
#    pp_geo = PolygonPixelGeo(zs,thetas,phis,theta_in,phi_in,C,z_fine,l_max=l_max,res_healpix=res_choose)
#    t2 = time()
#    print "PolygonPixelGeo initialized at res "+str(res_choose)+" in time: "+str(t2-t1)
#    pp_geo2 = PolygonPixelGeo(zs,thetas,phis,theta_in,phi_in,C,z_fine,l_max=l_max,res_healpix=res_choose2)
#    t3 = time()
#    print "PolygonPixelGeo initialized at res "+str(res_choose2)+" in time: "+str(t3-t2)
#    r_geo = RectGeo(zs,np.array([theta0,theta1]),np.array([phi0,phi1]),C,z_fine)
#    alm1 = r_geo.get_alm_array(10)[0]
#    alm2 = poly_geo.get_alm_array(10)[0]
#    print "max diff from rect "+str(np.max(np.abs(alm2-alm1)))
#    nt = poly_geo.n_v
#
#    my_table = poly_geo.alm_table.copy()
#    #get RectGeo to cache the values in the table
#    #for ll in xrange(0,l_max+1):
#    #    for mm in xrange(0,ll+1):
#    #        r_geo.a_lm(ll,mm)
#    #        if mm>0:
#    #            r_geo.a_lm(ll,-mm)
#    #r_alm_table = r_geo.alm_table
#    #reconstruct at higher resolution to mitigate resolution effects in determining accuracy
#    totals_pp= reconstruct_from_alm(l_max,pp_geo2.all_pixels[:,0],pp_geo2.all_pixels[:,1],pp_geo.alm_table)
#    totals_poly = reconstruct_from_alm(l_max,pp_geo2.all_pixels[:,0],pp_geo2.all_pixels[:,1],my_table)
#    avg_diff = np.average(np.abs(totals_pp-totals_poly))
#    print "mean absolute difference between pixel and exact geo reconstruction: "+str(avg_diff)
#    poly_error = np.sqrt(np.average(np.abs(totals_poly-pp_geo2.contained*1.)**2))
#    pp_error = np.sqrt(np.average(np.abs(totals_pp-pp_geo2.contained*1.)**2))
#    print "rms reconstruction error of exact geo: "+str(poly_error)
#    print "rms reconstruction error of pixel geo at res "+str(res_choose)+": "+str(pp_error)
#    #can be negative if res_choose2=res_choose due to pixelation effects
#    print "improvement in rms reconstruction accuracy: "+str((pp_error-poly_error)/pp_error*100)+"%"
#
#    #totals_alm = reconstruct_from_alm(l_max,pp_geo.all_pixels[:,0],pp_geo.all_pixels[:,1],r_alm_table)
#    try_plot=True
#    do_poly=True
#    if try_plot:
#               #try:
#            from mpl_toolkits.basemap import Basemap
#            import matplotlib.pyplot as plt
#            m = Basemap(projection='moll',lon_0=0)
#            #m.drawparallels(np.arange(-90.,120.,30.))
#            #m.drawmeridians(np.arange(0.,420.,60.))
#            #restrict = totals_recurse>-1.
#            lats = (pp_geo2.all_pixels[:,0]-np.pi/2.)*180/np.pi
#            lons = pp_geo2.all_pixels[:,1]*180/np.pi
#            x,y=m(lons,lats)
#            #have to switch because histogram2d considers y horizontal, x vertical
#            fig = plt.figure(figsize=(10,5))
#            ax = fig.add_subplot(121)
#            H1,yedges1,xedges1 = np.histogram2d(y,x,100,weights=totals_pp)
#            X1, Y1 = np.meshgrid(xedges1, yedges1)
#            pc1 = ax.pcolormesh(X1,Y1,-H1,cmap='gray')
#            ax.set_aspect('equal')
#            ax.set_title("PolygonPixelGeo reconstruction")
#            #fig.colorbar(pc1,ax=ax)
#            #m.plot(x,y,'bo',markersize=1)
#            pp_geo2.sp_poly.draw(m,color='red')
#            if do_poly:
#                ax = fig.add_subplot(122)
#                H2,yedges2,xedges2 = np.histogram2d(y,x,100,weights=1.*totals_poly)
#                X2, Y2 = np.meshgrid(xedges2, yedges2)
#                ax.pcolormesh(X2,Y2,-H2,cmap='gray')
#                ax.set_aspect('equal')
#                #m.plot(x,y,'bo',markersize=1)
#                ax.set_title("PolygonGeo reconstruction")
#                pp_geo2.sp_poly.draw(m,color='red')
#            plt.show()

