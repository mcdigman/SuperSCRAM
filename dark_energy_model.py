import numpy as np
from scipy.interpolate import interp1d,InterpolatedUnivariateSpline
#TODO why mix interp1d and InterpolatedUnivariateSpline
class DarkEnergyModel(object):
    """class for computing parameters related to a specific dark energy model"""
    def __init__(self):
        return
    def w_of_z(self,z):
        raise NotImplementedError('w(z) must be implemented in subclass')
    def de_mult(self,z):
        raise NotImplementedError('de mult must be implemented in subclass')

class DarkEnergyConstant(DarkEnergyModel):
    """w(z)=constant dark energy model"""
    def __init__(self,w):
        """w: constant w"""
        self.w = w
        DarkEnergyModel.__init__(self)

    def w_of_z(self,z):
        if isinstance(z,np.ndarray) and z.size>1:
            assert np.all(np.diff(z)>0.)
        return np.full_like(z,self.w)
    def de_mult(self,z):
        if isinstance(z,np.ndarray) and z.size>1:
            assert np.all(np.diff(z)>0.)
        return (z+1.)**(3.*(1.+self.w))

class DarkEnergyW0Wa(DarkEnergyModel):
    """w(z)=constant dark energy model"""
    def __init__(self,w0,wa):
        """w: constant w"""
        self.w0 = w0
        self.wa = wa
        DarkEnergyModel.__init__(self)

    def w_of_z(self,z):
        if isinstance(z,np.ndarray) and z.size>1:
            assert np.all(np.diff(z)>0.)
        return self.w0+(1.-1./(1.+z))*self.wa
    def de_mult(self,z):
        if isinstance(z,np.ndarray) and z.size>1:
            assert np.all(np.diff(z)>0.)
        return np.exp(-3.*self.wa*z/(1.+z))*(1.+z)**(3.*(1.+self.w0+self.wa))

class DarkEnergyJDEM(DarkEnergyModel):
    """36 bin piecewise constant dark energy model"""
    def __init__(self,ws_in,a_step,w_base):
        #piecewise constant approximation over 36 values with z=0.025 spacing
        self.ws_in = ws_in
        self.a_step = a_step
        self.w = w_base

        #z grid for dark energy interpolation
        a_de = np.arange(1.,a_step,-a_step)
        z_de = 1./a_de-1.
        z_de[0] = 0. #enforce to avoid numeric precision issue

        zs_max = 0.025*np.arange(1,37)/(1-0.025*np.arange(1,37))
        zs_min = 0.025*np.arange(0,36)/(1-0.025*np.arange(0,36))
        #default to value of w over edge
        #TODO enforce consistent use of default w, ie defaulting so at higher z just use last z given
        ws = np.full(z_de.size,self.w)
        itr = 0
        for i in xrange(0,z_de.size):
            if z_de[i]>=zs_max[itr]:
                itr+=1
                if itr>=zs_max.size:
                    break
            if z_de[i]>=zs_min[itr] and z_de[i]<zs_max[itr]:
                ws[i] = ws_in[itr]

        self.w_interp_jdem = interp1d(z_de,ws)
        de_step = de_exp_const_w(zs_max,ws_in)-de_exp_const_w(zs_min,ws_in)
        de_step_sum = np.full(de_step.size+1,0.)
        de_step_sum[1::] += np.cumsum(de_step)
        de_exponent_true = de_exp_const_w(z_de,ws)
        for itr in xrange(0,36):
            de_exponent_true[(zs_min[itr]<=z_de)*(z_de<zs_max[itr])]+=de_step_sum[itr]-de_exp_const_w(zs_min[itr],ws_in[itr])
        de_exponent_true[z_de>=zs_max[-1]]+=de_step_sum[-1]-de_exp_const_w(zs_max[-1],self.w)
        #TODO needless logging and exping?
        self.de_true_interp = InterpolatedUnivariateSpline(z_de,np.exp(3.*de_exponent_true),k=3,ext=2)
        DarkEnergyModel.__init__(self)

    def w_of_z(self,z):
        z = np.asanyarray(z)
        if not (np.any(z>=9.) or np.any(z<0)):
            result = self.w_interp_jdem(z)
        else:
            result = np.zeros_like(z)
            result[z>=9.] = self.w
            result[z<0.] = self.w
            result[(z>=0.)*(z<9.)] = self.w_interp_jdem(z[(z>=0.)*(z<9.)])
        if isinstance(z,np.ndarray) and z.size>1:
            assert np.all(np.diff(z)>0.)
        return result

    def de_mult(self,z):
        z = np.asanyarray(z)
        if not (np.any(z<0) or np.any(z>=9.)):
            return self.de_true_interp(z)
        result = np.zeros_like(z)
        result[z<0.] = (z[z<0.]+1.)**(3.*(1.+self.w))
        result[(z>=0.)*(z<9.)] = self.de_true_interp(z[(z>=0.)*(z<9.)])
        result[z>=9.] = np.exp(3.*(de_exp_const_w(z[z>=9.],self.w)-de_exp_const_w(9.,self.w)+np.log(self.de_true_interp(9.))/3.))
        if isinstance(z,np.ndarray) and z.size>1:
            assert np.all(np.diff(z)>0.)
        return result

def de_exp_const_w(z,w):
    return np.log((z+1.)**(3.*(1.+w)))/3.
def de_mult_const(w,z):
    """multiplier for de term if w constant"""
    return (z+1.)**(3.*(1.+w))
def de_w_const(w,z):
    """w function for de term if w constant"""
    return np.full_like(z,w)
def de_mult_w0wa(w0,wa,z):
    """multiplier for de term if w0wa"""
    return np.exp(-3.*wa*z/(1.+z))*(1.+z)**(3.*(1.+w0+wa))
def de_w_w0wa(w0,wa,z):
    """w(z) for de term if w0wa"""
    return w0+(1.-1./(1.+z))*wa
