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



if __name__=="__main__":


    X=class_objects()
    k=np.logspace(-4,np.log10(5),50)
    z=np.array([0,1])
    P=X.linear_power(k) 
    #P2=X.nl_power(k,z) 

    import matplotlib.pyplot as plt
    ax=plt.subplot(111)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    ax.plot(k,P)
    #ax.plot(k,P2)
    #ax.plot(k,Pkz[:,0])
    plt.show()

    