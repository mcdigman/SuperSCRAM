''' 
    This module uses impo to import talk_to_class from the folder 
    class/python 
'''

import numpy as np 
import imp 

code=imp.load_source("talk_to_class","./class/python/talk_to_class.py") 

# return the class_interface object 
def class_objects(cosmology=None):
    return code.class_interface(cosmology)




# test run 
X=class_objects()
k=np.logspace(-2,2,1000)
P=X.linear_power(k) 
    