import numpy as np 
import class_interface 
from time import time 
import sys

JB=np.loadtxt('/Users/joe/Dropbox/Super_Sample_Covariance/SSCM/dump/P_full2_JAB_z0symm_II.dat')
	


	
#import SPT
print 'size', JB.shape
k=JB[:,0]
r=np.linspace(.00001,1000,600)
mu=np.linspace(-1,1,500)

def F_2(r,mu):
	if (r==0):
		return 0 
	if ((1+r**2-2*r*mu)==0):
		return 0 
		
	F=(7*mu+3*r-10*r*mu**2)/(1+r**2-2*r*mu)/(14.*r)
	#F[np.isnan(F)]=0 
	return F
	
def F_2_symm(r,mu):
	F=(7*mu+3*r-10*r*mu**2)/(1+r**2-2*r*mu)/(14.*r)
	F[np.isnan(F)]=0 
	return F
	
def S_2(r,mu): 
	S=(r**2-2*r*mu+mu**2)/(1+r**2-2*r*mu) -1/3. 
	S[np.isnan(S)]=0 
	return S
	
def F_13(r):
	F=( 12/r**4 -158/r**2 + 100 - \
	42*r**2+3/r**5*(7*r**2+2)*(r**2-1)**3*np.log(np.absolute(r+1)/np.absolute(r-1)) )
	F[np.isnan(F)]=0 
	return F
	
# def F_2(q,k,mu):
# 	psi=np.absolute(k-q)
# 	F=5/7. + .5*(q*k*mu-q**2)/q/psi*(q/psi + psi/q) + 2/7.* (q*k*mu-q**2)**2/q**2/psi**2
# 	F[np.isnan(F)]=0 
# 	return F
		

def SPT(k): 
	print 'size', k.size 
	q=np.logspace(np.log10(1e-10), np.log10(1000), 100000)
	
	import class_interface as iclass 
	
	t1=time()
	power=iclass.class_items()
	P=power.linear_power(q) 

	t2=time()
	print 'time', t2-t1 	
	from scipy.interpolate import interp1d
	from scipy.integrate import trapz 
	from scipy.integrate import romb
	from scipy.integrate import dblquad 
	from scipy.integrate import quad 
	P_L=interp1d(q,P, bounds_error=False, fill_value=0)
	

	
	P_22=np.zeros(k.size)
	P_S2=np.zeros(k.size)
	
	N_mu=501
	P_mu=np.zeros(N_mu)
	
	N_q=5000
	q_min=1e-7
	q_max=800
	eps=.05
	
	def I_22(r,mu,args):
		k=args	
		psi=1+r**2 -2*r*mu
		
		return r**2*P_L(r*k)*P_L(k*psi)*F_2(r,mu)**2 
		
	P_22=np.zeros(k.size)
	
	def I_22(k):
					
		def I_r(r):
			r=10**r 
			#print 'r,k' 
			#print r, k 
			
			def I_mu(mu):
				
				#print 'r, mu', r, mu 
				psi=1+r**2-2*r*mu
				#print 'check'
				#print P_L(k*psi)*F_2(r,mu)**2 
				return P_L(k*psi)*F_2(r,mu)**2 
				
			I,_=quad(I_mu, -1,1)
			#print 'ok I', I
			return r**2*P_L(k*r)*I
			
		I,_=quad(I_r, -6,3)
		return I 
		

			
	for a in range(k.size):
		P_22[a]=I_22(k[a]) 
		print 'a', a 
	
		
	print 'total time', t3-t1
	return P_22*k**3/(2*np.pi**2) 
	



h=.6774	

id=np.arange(0,JB.shape[0])
id=id[0::50]
JB=JB[id,:]


P_22=SPT(JB[:,0]*h)
P_22=P_22*h**3



import matplotlib.pyplot as plt
	
ax=plt.subplot(141)
ax.set_xscale('log')
ax.set_yscale('log')


	
ax.plot(JB[:,0],P_22)
#ax.plot(JB[:,0],2*P_13)



ax.plot(JB[:,0],JB[:,2],'--')
ax.plot(JB[:,0],JB[:,3],'--',color='black')

#plt.axvline(.35)
#plt.axvline(.02)

	
ax=plt.subplot(142)
ax.set_xscale('log')
#ax.plot(JB[:,0],(P_22-JB[:,2])/JB[:,2])
ax.plot(JB[:,0],P_22/JB[:,2])




plt.show()