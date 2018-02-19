"""pixelated geometry"""
import numpy as np
from geo import Geo
import ylm_utils as ylmu

#same pixels at every redshift.
class PixelGeo(Geo):
    """generic pixelated geometry"""
    def __init__(self,zs,pixels,C,z_fine,l_max,hard_l_max=np.inf):
        """pixelated geomtery
            inputs:
                zs: tomographic z bins
                pixels: pixels in format np.array([(theta,phi,area)]), area in steradians
                C: CosmoPie object
                z_fine: the fine z slices
                hard_l_max: absolute maximum possible l to resolve
        """
        self.pixels = pixels
        self.hard_l_max = hard_l_max


        Geo.__init__(self,zs,C,z_fine)

        self._l_max = 0
        self.alm_table[(0,0)] = np.sum(self.pixels[:,2])/np.sqrt(4.*np.pi)
        self.alm_table,_,_,self.alm_dict = self.get_a_lm_table(l_max)
        self._l_max = l_max

#    def surface_integral(self,function):
#        """do the surface integral by summing over values at the discrete pixels"""
#        total = 0.
#        for i in xrange(0,self.pixels.shape[0]):
#            total+=function(self.pixels[i,0],self.pixels[i,1])*self.pixels[i,2] #f(theta,phi)*A
#        return total

#    def a_lm(self,l,m):
#        """vectorized a_lm computation relies on vector Y_r"""
#        alm = self.alm_table.get((l,m))
#        if alm is None:
#            alm = np.sum(Y_r(l,m,self.pixels[:,0],self.pixels[:,1])*self.pixels[:,2])
#            self.alm_table[(l,m)] = alm
#        return alm
    def angular_area(self):
        return np.sum(self.pixels[:,2])

    def a_lm(self,l,m):
        """a(l,m) if not precomputed, regenerate table up to specified l, otherwise read it out of the table
            assume constant pixel area"""
        if l>self._l_max:
            print "PixelGeo: l value "+str(l)+" exceeds maximum precomputed l "+str(self._l_max)+",expanding table"
            self.alm_table,_,_,self.alm_dict = self.get_a_lm_table(l)
            self._l_max = l
        alm = self.alm_table.get((l,m))
        if alm is None:
            raise RuntimeError("PixelGeo: alm evaluated to None at l="+str(l)+",m="+str(m)+". l,m may exceed highest available Ylm")
        return alm

    def get_a_lm_table(self,l_max):
        """get table of a(l,m) below l_max"""
        if l_max>self.hard_l_max:
            raise ValueError('requested l '+str(l_max)+' exceeds resolvable l limit '+str(self.hard_l_max))
        return ylmu.get_a_lm_table(l_max,self.pixels[:,0],self.pixels[:,1],self.pixels[0,2])
