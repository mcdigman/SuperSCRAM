"""convenience functions for extrapolation"""
from __future__ import division,print_function,absolute_import
from builtins import range
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
def power_law_extend(x_in,f_in,x_out,k=3,extend_limit=None):
    """take f_in(x_in) and extrapolate with power laws to f_in(x_out)
    interpolates with spline order k inside
    extend_limit is multiplier to limit maximum extrapolation without error"""
    if np.any(x_in<=0):
        raise ValueError('extrapolation routine not equipped for negative inputs')

    if extend_limit is not None:
        #don't extrapolate excessively
        if x_in[0]/x_out[0] > 2.:
            raise ValueError('min x in '+str(x_in[0])+' and min x out '+str(x_in[0])+' differ too much')
        if x_out[-1]/x_in[-1] > 2.:
            raise ValueError('max x in '+str(x_in[-1])+' and max x out '+str(x_out[-1])+' differ too much')

    if not np.array_equal(x_in,x_out):
        f_out = np.zeros(x_out.size)
        mask_low = x_in[0]<=x_out
        mask_high =  x_out<=x_in[-1]
        mask_mid = mask_low & mask_high
        f_out[mask_mid] = InterpolatedUnivariateSpline(x_in,f_in,k=k,ext=2)(x_out[mask_mid])
        exp_low = np.log(f_in[1]/f_in[0])/np.log(x_in[1]/x_in[0])
        mult_low = f_in[0]/x_in[0]**exp_low
        exp_high = np.log(f_in[-1]/f_in[-2])/np.log(x_in[-1]/x_in[-2])
        mult_high = f_in[-1]/x_in[-1]**exp_high
        f_out[~mask_low] = mult_low*x_out[~mask_low]**exp_low
        f_out[~mask_high] = mult_high*x_out[~mask_high]**exp_high
        return f_out
    else:
        return f_in.copy()
