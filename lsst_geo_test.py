"""test approximate a circular geometry with PolygonGeo"""
from __future__ import print_function,division,absolute_import
from builtins import range
import numpy as np
from cosmopie import CosmoPie
import defaults
from premade_geos import WFIRSTGeo,LSSTGeoSimpl,LSSTGeo,LSSTPixelGeo,WFIRSTPixelGeo
from polygon_union_geo import PolygonUnionGeo
from polygon_pixel_union_geo import PolygonPixelUnionGeo
from polygon_utils import get_healpix_pixelation
from polygon_display_utils import display_geo
from ylm_utils import reconstruct_from_alm
from copy import deepcopy

if __name__=='__main__':
    do_plot = True
    do_approximate_checks = False

    cosmo_fid =  defaults.cosmology.copy()
    C = CosmoPie(cosmo_fid,'jdem')
    l_max = 10
    zs = np.array([0.01,1.])
    z_fine = np.arange(0.01,1.0001,0.01)

#    radius = 0.3126603700269391
#    n_x = 100
#    error_old = 10000000.
#    error_new = 1000000.
#    area_goal = 1000.
#    while error_new<error_old:
#        geo1 = CircleGeo(zs,C,radius,n_x,z_fine,l_max,{'n_double':30})
#        area1 = 2.*np.pi*(1.-np.cos(radius))
#        area2 = geo1.angular_area()
#        radius = radius*np.sqrt(area_goal/(geo1.angular_area()*180**2/np.pi**2))
#        error_old = error_new
#        error_new = np.abs(area2*180**2/np.pi**2-area_goal)
#    print("radius for n_x="+str(n_x)+" 1000 deg^2="+str(radius))
    do_simpl_test = True
    if do_simpl_test:
        geo1 = LSSTGeoSimpl(zs,C,z_fine,l_max,{'n_double':30},phi0=0.,phi1=0.9202821591024097*0.8*1.0383719267257006*1.0000197370387798*1.0000197370387798*0.99998029455615*0.9999999844187037*0.9999999999876815*0.9999999999999916*1.0000000000000029,deg0=-49,deg1=-20)
        print(0.31587654497768103/geo1.angular_area())
        import sys
        sys.exit()
        geo2 = LSSTGeo(zs,C,z_fine,l_max,{'n_double':30})
        #geo2 = LSSTGeoSimpl(zs,C,z_fine,l_max,{'n_double':30},phi1=2.0*2.9030540874480577)
        #geo3 = WFIRSTGeo(zs,C,z_fine,l_max,{'n_double':30})
        geo4 = PolygonUnionGeo(geo2.geos,np.append(geo1,geo2.masks))
        print(geo4.angular_area(),geo2.angular_area()-geo1.angular_area(),geo4.angular_area()-geo2.angular_area()+geo1.angular_area())
        #assert geo1.angular_area()<geo2.angular_area()
        #assert geo3.angular_area()<geo1.angular_area()
        if do_plot:
            display_geo(geo4,l_max)
#            from mpl_toolkits.basemap import Basemap
#            import matplotlib.pyplot as plt
#            m = Basemap(projection='moll',lon_0=0)
#            geo1.sp_poly.draw(m,color='red')
#            geo2.sp_poly.draw(m,color='blue')
#            geo3.sp_poly.draw(m,color='green')
#            plt.show()
    import sys
    sys.exit()
    do_lsst_test = False
    if do_lsst_test:
        geo4 = LSSTGeo(zs,C,z_fine,l_max,{'n_double':80})
        res_healpix_high = 6
        res_healpix_low = 5
        geo5 = LSSTPixelGeo(zs,C,z_fine,l_max,res_healpix_high)
        geo6 = LSSTPixelGeo(zs,C,z_fine,l_max,res_healpix_low)
        l_max_high = 50
        assert geo4.union_mask.area()<=geo4.masks[0].angular_area()
        assert np.isclose(geo4.union_pos.area(),geo4.geos[0].angular_area())
        assert np.isclose(geo4.angular_area(),geo4.union_pos.area()-geo4.union_mask.area())
        assert geo4.angular_area()<=geo4.geos[0].angular_area()
        assert geo5.angular_area()<=geo5.geos[0].angular_area()
        assert geo6.angular_area()<=geo6.geos[0].angular_area()
        assert np.isclose(geo5.angular_area(),geo4.angular_area(),rtol=1.e-3,atol=1.e-3)
        assert np.isclose(geo6.angular_area(),geo4.angular_area(),rtol=1.e-3,atol=1.e-3)
        assert np.isclose(geo6.angular_area(),geo5.angular_area(),rtol=1.e-3,atol=1.e-3)
        alms5_0 = geo5.get_alm_table(8)
        alms4_0 = geo4.get_alm_table(8)
        alms5_array_0 = geo5.get_alm_array(8)
        alms4_array_0 = geo4.get_alm_array(8)
        alms5_1 = geo5.get_alm_table(10)
        alms4_1 = geo4.get_alm_table(10)
        alms5_array_1 = geo5.get_alm_array(10)
        alms4_array_1 = geo4.get_alm_array(10)
        alms6_array_2 = geo6.get_alm_array(l_max_high)
        alms5_array_2 = geo5.get_alm_array(l_max_high)
        alms4_array_2 = geo4.get_alm_array(l_max_high)
        alms6_2 = geo6.get_alm_table(l_max_high)
        alms5_2 = geo5.get_alm_table(l_max_high)
        alms4_2 = geo4.get_alm_table(l_max_high)
        #a few basic self consistency checks
        assert np.all(alms5_array_2[0][0:alms5_array_1[0].size]==alms5_array_1[0])
        assert np.all(alms4_array_2[0][0:alms4_array_1[0].size]==alms4_array_1[0])
        assert np.all(alms5_array_2[0][0:alms5_array_0[0].size]==alms5_array_0[0])
        assert np.all(alms4_array_2[0][0:alms4_array_0[0].size]==alms4_array_0[0])
        for key in list(alms5_0):
            assert alms5_2[key]==alms5_0[key]
        for key in list(alms4_0):
            assert alms4_2[key]==alms4_0[key]
        for key in list(alms5_1):
            assert alms5_2[key]==alms5_1[key]
        for key in list(alms4_1):
            assert alms4_2[key]==alms4_1[key]
        for itr in range(0,alms5_array_2[0].size):
            assert alms5_2[(alms5_array_2[1][itr],alms5_array_2[2][itr])]==alms5_array_2[0][itr]
        for itr in range(0,alms4_array_2[0].size):
            assert alms4_2[(alms4_array_2[1][itr],alms4_array_2[2][itr])]==alms4_array_2[0][itr]
        #check inter geo consistency
        assert np.allclose(alms5_array_2,alms4_array_2,atol=1.e-3,rtol=1.e-3)
        assert np.allclose(alms5_array_2,alms6_array_2,atol=5.e-3,rtol=5.e-3)
        assert np.allclose(alms4_array_2,alms6_array_2,atol=5.e-3,rtol=5.e-3)
        
        pixels = get_healpix_pixelation(res_healpix_low)
        reconstruct4 = reconstruct_from_alm(l_max_high,pixels[:,0],pixels[:,1],alms4_2)
        reconstruct5 = reconstruct_from_alm(l_max_high,pixels[:,0],pixels[:,1],alms5_2)
        reconstruct6 = reconstruct_from_alm(l_max_high,pixels[:,0],pixels[:,1],alms6_2)
        assert np.allclose(reconstruct4,reconstruct5,atol=4.e-2,rtol=1.e-4)
        assert np.allclose(reconstruct4,reconstruct6,atol=4.e-1,rtol=1.e-4)
        assert np.allclose(reconstruct5,reconstruct6,atol=4.e-1,rtol=1.e-4)
        
        if do_plot:
            from mpl_toolkits.basemap import Basemap
            import matplotlib.pyplot as plt
            from polygon_display_utils import plot_reconstruction
            import healpy as hp
            m = Basemap(projection='moll',lon_0=0)
            fig = plt.figure(figsize=(10,5))
            #ax = fig.add_subplot(121)
            #plot_reconstruction(m,ax,pixels,reconstruct6)
            #geo4.union_pos.draw(m,color='red')
            #geo4.union_mask.draw(m,color='blue')
            #plt.show()
            ax = fig.add_subplot(221)
            #plot_reconstruction(m,ax,pixels,geo6.contained*1.)
            #hp.mollview(geo6.contained*1.,title='Ground Truth',fig=fig,hold=True,cmap='Greys')
            plot_reconstruction(geo6.contained*1.,'Ground Truth',fig)
            ax = fig.add_subplot(222)
            #hp.mollview(reconstruct4,title='Exact Solution',fig=fig,hold=True,cmap='Greys')
            plot_reconstruction(reconstruct4,'Exact Solution',fig)
            #plot_reconstruction(m,ax,pixels,reconstruct4)
            ax = fig.add_subplot(223)
            plot_reconstruction(reconstruct5,'High Res',fig)
            #hp.mollview(reconstruct4,title='High Res',fig=fig,hold=True,cmap='Greys')
            #plot_reconstruction(m,ax,pixels,reconstruct5)
            ax = fig.add_subplot(224)
            plot_reconstruction(reconstruct6,'Low Res',fig)
            #hp.mollview(reconstruct4,title='Low Res',fig=fig,hold=True,cmap='Greys')
            #geo4.union_pos.draw(m,color='red')
            #plot_reconstruction(m,ax,pixels,reconstruct6)
            #geo4.union_mask.draw(m,color='blue')
            abs_error = np.abs(reconstruct6-geo6.contained*1.)
            mse = np.sqrt(np.average(abs_error**2))
    do_wfirst_test=False
    if do_wfirst_test:
        geo4 = WFIRSTGeo(zs,C,z_fine,l_max,{'n_double':80})
        res_healpix_high = 8
        res_healpix_low = 7
        geo5 = WFIRSTPixelGeo(zs,C,z_fine,l_max,res_healpix_high)
        geo6 = WFIRSTPixelGeo(zs,C,z_fine,l_max,res_healpix_low)
        l_max_high = 84
        assert np.isclose(geo5.angular_area(),geo4.angular_area(),rtol=1.e-3,atol=1.e-3)
        assert np.isclose(geo6.angular_area(),geo4.angular_area(),rtol=1.e-3,atol=1.e-3)
        assert np.isclose(geo6.angular_area(),geo5.angular_area(),rtol=1.e-3,atol=1.e-3)
        alms5_0 = geo5.get_alm_table(8)
        alms4_0 = geo4.get_alm_table(8)
        alms5_array_0 = geo5.get_alm_array(8)
        alms4_array_0 = geo4.get_alm_array(8)
        alms5_1 = geo5.get_alm_table(10)
        alms4_1 = geo4.get_alm_table(10)
        alms5_array_1 = geo5.get_alm_array(10)
        alms4_array_1 = geo4.get_alm_array(10)
        alms6_array_2 = geo6.get_alm_array(l_max_high)
        alms5_array_2 = geo5.get_alm_array(l_max_high)
        alms4_array_2 = geo4.get_alm_array(l_max_high)
        alms6_2 = geo6.get_alm_table(l_max_high)
        alms5_2 = geo5.get_alm_table(l_max_high)
        alms4_2 = geo4.get_alm_table(l_max_high)
        #a few basic self consistency checks
        assert np.all(alms5_array_2[0][0:alms5_array_1[0].size]==alms5_array_1[0])
        assert np.all(alms4_array_2[0][0:alms4_array_1[0].size]==alms4_array_1[0])
        assert np.all(alms5_array_2[0][0:alms5_array_0[0].size]==alms5_array_0[0])
        assert np.all(alms4_array_2[0][0:alms4_array_0[0].size]==alms4_array_0[0])
        for key in list(alms5_0):
            assert alms5_2[key]==alms5_0[key]
        for key in list(alms4_0):
            assert alms4_2[key]==alms4_0[key]
        for key in list(alms5_1):
            assert alms5_2[key]==alms5_1[key]
        for key in list(alms4_1):
            assert alms4_2[key]==alms4_1[key]
        for itr in range(0,alms5_array_2[0].size):
            assert alms5_2[(alms5_array_2[1][itr],alms5_array_2[2][itr])]==alms5_array_2[0][itr]
        for itr in range(0,alms4_array_2[0].size):
            assert alms4_2[(alms4_array_2[1][itr],alms4_array_2[2][itr])]==alms4_array_2[0][itr]
        #check inter geo consistency
        assert np.allclose(alms5_array_2,alms4_array_2,atol=1.e-3,rtol=1.e-3)
        assert np.allclose(alms5_array_2,alms6_array_2,atol=5.e-3,rtol=5.e-3)
        assert np.allclose(alms4_array_2,alms6_array_2,atol=5.e-3,rtol=5.e-3)
        
        pixels = get_healpix_pixelation(res_healpix_low)
        reconstruct4 = reconstruct_from_alm(l_max_high,pixels[:,0],pixels[:,1],alms4_2)
        reconstruct5 = reconstruct_from_alm(l_max_high,pixels[:,0],pixels[:,1],alms5_2)
        reconstruct6 = reconstruct_from_alm(l_max_high,pixels[:,0],pixels[:,1],alms6_2)
        assert np.allclose(reconstruct4,reconstruct5,atol=4.e-2,rtol=1.e-4)
        assert np.allclose(reconstruct4,reconstruct6,atol=4.e-1,rtol=1.e-4)
        assert np.allclose(reconstruct5,reconstruct6,atol=4.e-1,rtol=1.e-4)
        
        if do_plot:
            from mpl_toolkits.basemap import Basemap
            import matplotlib.pyplot as plt
            #from polygon_display_utils import plot_reconstruction
            import healpy as hp
            m = Basemap(projection='moll',lon_0=0)
            fig = plt.figure(figsize=(10,5))
            #ax = fig.add_subplot(121)
            #plot_reconstruction(m,ax,pixels,reconstruct6)
            #geo4.union_pos.draw(m,color='red')
            #geo4.union_mask.draw(m,color='blue')
            #plt.show()
            ax = fig.add_subplot(221)
            #plot_reconstruction(m,ax,pixels,geo6.contained*1.)
            hp.mollview(geo6.contained*1.,title='Ground Truth',fig=fig,hold=True,cmap='Greys')
            ax = fig.add_subplot(222)
            hp.mollview(reconstruct4,title='Exact Solution',fig=fig,hold=True,cmap='Greys')
            #plot_reconstruction(m,ax,pixels,reconstruct4)
            ax = fig.add_subplot(223)
            hp.mollview(reconstruct4,title='High Res',fig=fig,hold=True,cmap='Greys')
            #plot_reconstruction(m,ax,pixels,reconstruct5)
            ax = fig.add_subplot(224)
            hp.mollview(reconstruct4,title='Low Res',fig=fig,hold=True,cmap='Greys')
            #geo4.union_pos.draw(m,color='red')
            #plot_reconstruction(m,ax,pixels,reconstruct6)
            #geo4.union_mask.draw(m,color='blue')
            abs_error = np.abs(reconstruct6-geo6.contained*1.)
            mse = np.sqrt(np.average(abs_error**2))

    
    do_wfirst_lsst_test = True
    if do_wfirst_lsst_test:
        res_healpix_high = 5
        res_healpix_low = 4
        l_max_high = 30
        geo_wfirst = WFIRSTGeo(zs,C,z_fine,l_max,{'n_double':80})
        geo_lsst = LSSTGeo(zs,C,z_fine,l_max,{'n_double':80})
        geo_wfirst_pp1 = WFIRSTPixelGeo(zs,C,z_fine,l_max,res_healpix_low)
        geo_lsst_pp1 = LSSTPixelGeo(zs,C,z_fine,l_max,res_healpix_low)
        geo_wfirst_pp2 = WFIRSTPixelGeo(zs,C,z_fine,l_max,res_healpix_high)
        geo_lsst_pp2 = LSSTPixelGeo(zs,C,z_fine,l_max,res_healpix_high)
        geo4 = PolygonUnionGeo(geo_lsst.geos,np.append(geo_wfirst,geo_lsst.masks)) 
        geo5 = PolygonPixelUnionGeo(geo_lsst_pp2.geos,np.append(geo_wfirst_pp2,geo_lsst_pp2.masks)) 
        geo6 = PolygonPixelUnionGeo(geo_lsst_pp1.geos,np.append(geo_wfirst_pp1,geo_lsst_pp1.masks)) 
        assert np.isclose(geo4.angular_area(),geo_lsst.angular_area()-geo_wfirst.angular_area())
        assert np.isclose(geo5.angular_area(),geo_lsst_pp2.angular_area()-geo_wfirst_pp2.angular_area())
        assert np.isclose(geo6.angular_area(),geo_lsst_pp1.angular_area()-geo_wfirst_pp1.angular_area())
        #geo4 = #WFIRSTGeo(zs,C,z_fine,l_max,{'n_double':80})
        #geo5 = WFIRSTPixelGeo(zs,C,z_fine,l_max,res_healpix_high)
        #geo6 = WFIRSTPixelGeo(zs,C,z_fine,l_max,res_healpix_low)
        #assert geo4.union_mask.area()<=geo4.masks[0].angular_area()
        assert np.isclose(geo4.union_pos.area(),geo4.geos[0].angular_area())
        assert np.isclose(geo4.angular_area(),geo4.union_pos.area()-geo4.union_mask.area())
        assert geo4.angular_area()<=geo4.geos[0].angular_area()
        assert geo5.angular_area()<=geo5.geos[0].angular_area()
        assert geo6.angular_area()<=geo6.geos[0].angular_area()
        assert np.isclose(geo5.angular_area(),geo4.angular_area(),rtol=1.e-3,atol=1.e-3)
        assert np.isclose(geo6.angular_area(),geo4.angular_area(),rtol=1.e-3,atol=1.e-3)
        assert np.isclose(geo6.angular_area(),geo5.angular_area(),rtol=1.e-3,atol=1.e-3)
        alms6_0 = deepcopy(geo6.get_alm_table(8))
        alms5_0 = deepcopy(geo5.get_alm_table(8))
        alms4_0 = deepcopy(geo4.get_alm_table(8))
        alms6_array_0 = deepcopy(geo6.get_alm_array(8))
        alms5_array_0 = deepcopy(geo5.get_alm_array(8))
        alms4_array_0 = deepcopy(geo4.get_alm_array(8))
        alms6_1 = deepcopy(geo6.get_alm_table(10))
        alms5_1 = deepcopy(geo5.get_alm_table(10))
        alms4_1 = deepcopy(geo4.get_alm_table(10))
        alms6_array_1 = deepcopy(geo6.get_alm_array(10))
        alms5_array_1 = deepcopy(geo5.get_alm_array(10))
        alms4_array_1 = deepcopy(geo4.get_alm_array(10))
        alms6_array_2 = deepcopy(geo6.get_alm_array(l_max_high))
        alms5_array_2 = deepcopy(geo5.get_alm_array(l_max_high))
        alms4_array_2 = deepcopy(geo4.get_alm_array(l_max_high))
        alms_wfirst_array = deepcopy(geo_wfirst.get_alm_array(l_max_high))
        alms_lsst_array = deepcopy(geo_lsst.get_alm_array(l_max_high))
        alms_wfirst_pp1_array = deepcopy(geo_wfirst_pp1.get_alm_array(l_max_high))
        alms_lsst_pp1_array = deepcopy(geo_lsst_pp1.get_alm_array(l_max_high))
        alms_wfirst_pp2_array = deepcopy(geo_wfirst_pp2.get_alm_array(l_max_high))
        alms_lsst_pp2_array = deepcopy(geo_lsst_pp2.get_alm_array(l_max_high))
        alms6_2 = deepcopy(geo6.get_alm_table(l_max_high))
        alms5_2 = deepcopy(geo5.get_alm_table(l_max_high))
        alms4_2 = deepcopy(geo4.get_alm_table(l_max_high))
        alms6_3 = deepcopy(geo6.get_alm_table(10))
        alms5_3 = deepcopy(geo5.get_alm_table(10))
        alms4_3 = deepcopy(geo4.get_alm_table(10))
        alms6_array_3 = deepcopy(geo6.get_alm_array(10))
        alms5_array_3 = deepcopy(geo5.get_alm_array(10))
        alms4_array_3 = deepcopy(geo4.get_alm_array(10))
        #a few basic self consistency checks
        assert np.all(alms6_array_3[0]==alms6_array_1[0])
        assert np.all(alms5_array_3[0]==alms5_array_1[0])
        assert np.all(alms4_array_3[0]==alms4_array_1[0])
        assert np.all(alms6_array_2[0][0:alms5_array_1[0].size]==alms6_array_1[0])
        assert np.all(alms5_array_2[0][0:alms5_array_1[0].size]==alms5_array_1[0])
        assert np.all(alms4_array_2[0][0:alms4_array_1[0].size]==alms4_array_1[0])
        assert np.all(alms6_array_2[0][0:alms5_array_0[0].size]==alms6_array_0[0])
        assert np.all(alms5_array_2[0][0:alms5_array_0[0].size]==alms5_array_0[0])
        assert np.all(alms4_array_2[0][0:alms4_array_0[0].size]==alms4_array_0[0])
        assert sorted(list(alms4_1))==sorted(list(alms5_1))
        assert sorted(list(alms4_1))==sorted(list(alms6_1))
        assert sorted(list(alms4_2))==sorted(list(alms5_2))
        assert sorted(list(alms4_2))==sorted(list(alms6_2))
        assert sorted(list(alms4_0))==sorted(list(alms5_0))
        assert sorted(list(alms4_0))==sorted(list(alms6_0))
        assert sorted(list(alms4_3))==sorted(list(alms5_3))
        assert sorted(list(alms4_3))==sorted(list(alms6_3))
        assert sorted(list(alms4_1))==sorted(list(alms4_3))
        assert sorted(list(alms5_1))==sorted(list(alms5_3))
        assert sorted(list(alms6_1))==sorted(list(alms6_3))
        assert alms6_array_0[0].size==len(list(alms6_0))
        assert alms5_array_0[0].size==len(list(alms5_0))
        assert alms4_array_0[0].size==len(list(alms4_0))
        assert alms6_array_1[0].size==len(list(alms6_1))
        assert alms6_array_1[0].size==len(list(alms6_1))
        assert alms5_array_1[0].size==len(list(alms5_1))
        assert alms4_array_2[0].size==len(list(alms4_2))
        assert alms5_array_2[0].size==len(list(alms5_2))
        assert alms4_array_2[0].size==len(list(alms4_2))
        assert alms6_array_3[0].size==len(list(alms6_3))
        assert alms5_array_3[0].size==len(list(alms5_3))
        assert alms4_array_3[0].size==len(list(alms4_3))
        #assert np.all(alms5_array_3[0]==alms5_array_1[0])
        #assert np.all(alms4_array_3[0]==alms4_array_1[0])
        for key in list(alms6_0):
            assert alms6_2[key]==alms6_0[key]
        for key in list(alms5_0):
            assert alms5_2[key]==alms5_0[key]
        for key in list(alms4_0):
            assert alms4_2[key]==alms4_0[key]
        for key in list(alms6_1):
            assert alms6_2[key]==alms6_1[key]
            assert alms6_3[key]==alms6_1[key]
        for key in list(alms5_1):
            assert alms5_2[key]==alms5_1[key]
            assert alms5_3[key]==alms5_1[key]
        for key in list(alms4_1):
            assert alms4_2[key]==alms4_1[key]
        for itr in range(0,alms6_array_2[0].size):
            assert alms6_2[(alms6_array_2[1][itr],alms6_array_2[2][itr])]==alms6_array_2[0][itr]
        for itr in range(0,alms5_array_2[0].size):
            assert alms5_2[(alms5_array_2[1][itr],alms5_array_2[2][itr])]==alms5_array_2[0][itr]
        for itr in range(0,alms4_array_2[0].size):
            assert alms4_2[(alms4_array_2[1][itr],alms4_array_2[2][itr])]==alms4_array_2[0][itr]
        #check inter geo consistency
        if do_approximate_checks:
            assert np.allclose(alms5_array_2,alms4_array_2,atol=1.e-3,rtol=1.e-3)
            assert np.allclose(alms5_array_2,alms6_array_2,atol=5.e-3,rtol=5.e-3)
            assert np.allclose(alms4_array_2,alms6_array_2,atol=5.e-3,rtol=5.e-3)

        #use complete intersection
        assert np.allclose(alms4_array_2[0],alms_lsst_array[0]-alms_wfirst_array[0])
        assert np.allclose(alms5_array_2[0],alms_lsst_pp2_array[0]-alms_wfirst_pp2_array[0])
        assert np.allclose(alms6_array_2[0],alms_lsst_pp1_array[0]-alms_wfirst_pp1_array[0])
        
        pixels = get_healpix_pixelation(res_healpix_low)
        reconstruct4 = reconstruct_from_alm(l_max_high,pixels[:,0],pixels[:,1],alms4_2)
        reconstruct5 = reconstruct_from_alm(l_max_high,pixels[:,0],pixels[:,1],alms5_2)
        reconstruct6 = reconstruct_from_alm(l_max_high,pixels[:,0],pixels[:,1],alms6_2)
        if do_approximate_checks:
            assert np.allclose(reconstruct4,reconstruct5,atol=4.e-2,rtol=1.e-4)
            assert np.allclose(reconstruct4,reconstruct6,atol=4.e-1,rtol=1.e-4)
            assert np.allclose(reconstruct5,reconstruct6,atol=4.e-1,rtol=1.e-4)
        
        if do_plot:
            #from mpl_toolkits.basemap import Basemap
            import matplotlib.pyplot as plt
            from polygon_display_utils import plot_reconstruction,reconstruct_and_plot
            import healpy as hp
            #m = Basemap(projection='moll',lon_0=0)
            fig = plt.figure(figsize=(10,5))
            ax = fig.add_subplot(221)
            plot_reconstruction(geo6.contained*1.,'Ground Truth',fig)
            ax = fig.add_subplot(222)
            plot_reconstruction(reconstruct4,'Exact Solution',fig)
            ax = fig.add_subplot(223)
            plot_reconstruction(reconstruct5,'High Res',fig)
            ax = fig.add_subplot(224)
            plot_reconstruction(reconstruct6,'Low Res',fig)
            #reconstruct_and_plot(geo6,l_max_high,pixels,'Low Res',fig)
            abs_error = np.abs(reconstruct6-geo6.contained*1.)
            mse = np.sqrt(np.average(abs_error**2))
            print("mse: ",mse)
