import numpy as np
import defaults
import FASTPTcode.FASTPT as FASTPT
import halofit as hf
from camb_power import camb_pow
class MatterPower:
    def __init__(self,cosmo_in,P_lin=None,k_in=None,params=defaults.matter_power_params,camb_params=defaults.camb_params):
        if P_lin is None or k_in is None:
            camb_pow(
         
