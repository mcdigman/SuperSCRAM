import numpy as np 
import class_interface 
from time import time 
import sys

JB=np.loadtxt('/Users/joe/Dropbox/Super_Sample_Covariance/SSCM/dump/P_full2_JAB_z0symm_II.dat')
	

#id=id[0::20]
#id=id[0::100]
#JB=JB[id,:]
	
#import SPT
print 'size', JB.shape
k=JB[:,0]
r=np.linspace(.00001,1000,600)
mu=np.linspace(-1,1,500)

def F_2(r,mu):
	F=(7*mu+3*r-10*r*mu**2)/(1+r**2-2*r*mu)/(14.*r)
	F[np.isnan(F)]=0 
	return F
		

def SPT(k): 
	q=np.logspace(np.log(1e-5), np.log(10), 2000)
	mu=np.linspace(-1,.98,250)
	
	import class_interface as iclass 
	
	t1=time()
	power=iclass.class_items()
	P_q, P_kq=power.linear_power_grid(k,q,mu) 
	t2=time()
	print 'time', t2-t1 
	print 'shape', P_kq.shape
	
	from scipy.integrate import trapz 
	
	P=np.zeros(k.size)
	for i in range(k.size):
		I=np.zeros(q.size)
		for j in range(q.size): 
			I[j]=trapz(P_kq[i,j,:]*F_2(q[j]/k[i],mu)**2*mu)
			
		P[i]=1/(2*np.pi**2)*trapz(q**3*I*P_q,np.log(q)) 	
	
	return P


h=.6774	

id=np.arange(0,JB.shape[0])
id=id[0::100]
JB=JB[id,:]


P_22=SPT(JB[:,0]*h)
P_22=P_22*h**3

print P_22

import matplotlib.pyplot as plt
	
ax=plt.subplot(121)
ax.set_xscale('log')
ax.set_yscale('log')


	
ax.plot(JB[:,0],P_22)

ax.plot(JB[:,0],P_22*2)
ax.plot(JB[:,0],JB[:,2],'--')
	
ax=plt.subplot(122)
ax.set_xscale('log')
ax.plot(JB[:,0],(P_22-JB[:,2])/JB[:,2])
plt.show()