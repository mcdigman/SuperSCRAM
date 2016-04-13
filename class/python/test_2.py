import numpy as np 
import class_interface 
from time import time 
import sys

JB=np.loadtxt('/Users/joe/Dropbox/Super_Sample_Covariance/SSCM/dump/P_full2_JAB_z0symm_II.dat')
	

h=.6774
#import SPT
print 'size', JB.shape


k_sample=np.logspace(np.log(.000001),np.log(1000),10000)
import class_interface as iclass 
power=iclass.class_items()
	
from scipy.interpolate import interp1d 
P_sample=power.linear_power(k_sample)
P_L=interp1d(k_sample,P_sample, bounds_error=False, fill_value=0)

#id=np.arange(0,JB.shape[0])
#id=id[0::5]
#JB=JB[id,:]

k=JB[:,0]

P=P_L(k*h)*h**3

P2=power.linear_power(k*h)*h**3 


import matplotlib.pyplot as plt
	
ax=plt.subplot(131)
ax.set_xscale('log')
ax.set_yscale('log')


ax.plot(k,P, color='green')
ax.plot(k,P2, color='red')
#ax.plot(k,power.linear_power(k/h)*h**3, color='red')
ax.plot(JB[:,0],JB[:,1],'--')

	
ax=plt.subplot(132)
ax.set_xscale('log')
ax.plot(JB[:,0],P/JB[:,1])


ax=plt.subplot(133)
ax.set_xscale('log')

ax.plot(JB[:,0],P2/JB[:,1], color='red')



plt.show()