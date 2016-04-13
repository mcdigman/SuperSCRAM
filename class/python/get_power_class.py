import numpy as np
import sys
from time import time 
from classy import Class

JB=np.loadtxt('/Users/joe/Dropbox/Super_Sample_Covariance/SSCM/dump/P_full2_JAB_z0symm_II.dat')

#h=0.6774
#omch2=0.1188
#ombh2=0.02230
#sigma8=0.816 #(this is


class class_items:
	
	def __init__(self, params=None):
	
		if params is None:
			# default cosmology 
			self.params={
 			'output' : 'tCl lCl,mPk, mTk', 
 			'l_max_scalars' : 2000, 
 			'z_pk' : 0, 
 			'A_s': 2.3e-9, 
 			'n_s' : 0.9624, 
 			'h' : 0.6774,
 			'omega_b' : 0.02230,
 			'omega_cdm': 0.1188, 
 			'k_pivot' : 0.05,
			'A_s' : 2.142e-9,
			'n_s' : 0.9667,
			'P_k_max_1/Mpc' : 500.0,
			'N_eff' :3.04,
			'Omega_fld' : 0,
			'YHe' : 0.2453,
			'z_reio' : 8.8}

			
		
		else:
			self.params=params
	
		self.cosmo=Class()
		self.cosmo.set(self.params)
		self.cosmo.compute()

	def linear_power(self,k):
		
		if k > 100: 
			P=self.cosmo.pk(100,0)*(100.**3)/k**3
			return P
				
				
		if k < .0001:
			return self.cosmo.pk(.0001,0)/(.0001)*k
				
		
		return self.cosmo.pk(k,0)



if __name__=='__main__':
	power=class_items()
	
	h=.6774
	#k=np.logspace(np.log10(.0001), np.log10(1), 100)
	#k=np.logspace(np.log10(.02), np.log10(.4), 1000)
	#k=np.logspace(np.log10(.00001), np.log10(10), 50) 
	k=JB[:,0]*h
	print 'max', np.max(k)
	def F_2(r,mu):
		return (7*mu+3*r-10*r*mu**2)/(1+r**2-2*r*mu)
		
	def P_22(r,mu,k):
		return power.linear_power(k*r)*F_2(r,mu)**2*power.linear_power(k*np.sqrt(1+r**2-2*r*mu))
	
	P=np.zeros(k.size)	
	t1=time()
	for i in range(k.size):
		P[i]=power.linear_power(k[i])
	t2=time()
	
	print 'time to make power', t2-t1
	
	import class_interface as iclass 
	
	power=iclass.class_items()
	t1=time()
	P2=power.linear_power(k)
	t2=time()
	
	print 'time to make power', t2-t1
	import matplotlib.pyplot as plt
	
	ax=plt.subplot(121)
	ax.set_xscale('log')
	ax.set_yscale('log')
	
	
	ax.plot(k/h,P*h**3)
	ax.plot(k/h,P2*h**3)
	ax.plot(JB[:,0],JB[:,1],'--')
	
	ax=plt.subplot(122)
	ax.set_xscale('log')
	ax.plot(k,(P*h**3)/JB[:,1])
	plt.show()
	
#	print x.linear_power(k)
	

# 		
# 	def SPT(k):
# 		return 1
		
				
				
				
				
										

	
		

		