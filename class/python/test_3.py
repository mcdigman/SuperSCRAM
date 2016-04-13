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
	F=(7*mu+3*r-10*r*mu**2)/(1+r**2-2*r*mu)/(14.*r)
	F[np.isnan(F)]=0 
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
	
	import matplotlib.pyplot as plt
	ax=plt.subplot(111)
# 	ax.set_xscale('log')
# 	ax.set_yscale('log')
#	ax.set_ylim(-100,100)
	print 'ok'
	from scipy.integrate import trapz 
 	r=np.linspace(.95,1.05,100)
 	r2=np.logspace(np.log10(.95),np.log10(1.05),100)
 	r3=np.logspace(np.log10(1.05),np.log10(.95),100)
 	
 	print r2
 	print r3 
 	ax.plot(r, F_2(r,1), color='black')
 	ax.plot(r, F_2(r,.99))
 	ax.plot(r2, F_2(r2,1), color='red')
	#ax.plot(q,P)
	
	
	print 'test'
	print trapz(F_2(r,1),r)
	print trapz(F_2(r2,1),r2)
	print trapz(F_2(r3,1),r3)
	print .5*(trapz(F_2(r3,1),r3) - trapz(F_2(r2,1),r2) )
 	plt.show()
	
	t2=time()
	print 'time', t2-t1 	
	from scipy.interpolate import interp1d
	from scipy.integrate import trapz 
	P_L=interp1d(q,P, bounds_error=False, fill_value=0)
	

	
	P_22=np.zeros(k.size)
	P_S2=np.zeros(k.size)
	
	N_mu=300
	P_mu=np.zeros(N_mu)
	q=np.logspace(np.log10(1e-4),np.log10(20),1500)
	
	def make_mu(r):
		if ( ( r > .98) and (r < 1.02)):
			
			return np.linspace(-1,.999,N_mu)
		else:
			return np.linspace(-1,.999,N_mu)
			
	I_13=np.zeros(k.size)
	mu=np.linspace(-.999,.999,N_mu)
	for a in range(k.size):
	
		I_r_22=np.zeros(q.size)
		I_r_S2=np.zeros(q.size)
		
		r=q/k[a]
		I_13[a]=k[a]**3*P_L(k[a])*trapz(r**2*P_L(k[a]*r)*F_13(r),r)
		for b in range(q.size):
			
			#mu=make_mu(r[b])
			psi=np.sqrt(1 + r[b]**2 - 2*r[b]*mu) 
			#print 'max min', np.min(psi), np.max(psi) 
			P_mu=P_L(k[a]*psi)
			I_22=trapz(P_mu*F_2(r[b],mu)**2, mu) 
			I_S2=trapz(P_mu*S_2(r[b],mu)**2, mu) 
			
			I_r_22[b]=r[b]**3*P_L(q[b])*I_22
			I_r_S2[b]=r[b]**2*P_L(q[b])*I_S2
			
		P_22[a]=trapz(I_r_22,np.log(r)) 
		P_S2[a]=trapz(I_r_S2,r) 
		
	
		#print a 
	t3=time()
	print 'total time', t3-t1
	return P_22*k**3/(2*np.pi**2) , I_13/252./(2*np.pi)**2,P_S2*k**3/(2*np.pi**2)
	



h=.6774	

id=np.arange(0,JB.shape[0])
id=id[0::20]
JB=JB[id,:]


P_22,P_13,P_S2=SPT(JB[:,0]*h)
P_22=P_22*h**3
P_13=P_13*h**3
P_S2=P_S2*h**3

print P_22

#H=np.loadtxt('/Users/joe/Dekstop/OSU_WORK_Stream_Velocities/Pk_z2.4.dat')

import matplotlib.pyplot as plt
	
ax=plt.subplot(141)
ax.set_xscale('log')
ax.set_yscale('log')


	
ax.plot(JB[:,0],P_22)
ax.plot(JB[:,0],2*P_13)



ax.plot(JB[:,0],JB[:,2],'--')
ax.plot(JB[:,0],JB[:,3],'--',color='black')

#plt.axvline(.35)
#plt.axvline(.02)

	
ax=plt.subplot(142)
ax.set_xscale('log')
#ax.plot(JB[:,0],(P_22-JB[:,2])/JB[:,2])
ax.plot(JB[:,0],P_22/JB[:,2])

ax=plt.subplot(143)
ax.set_xscale('log')
ax.set_yscale('log')


ax.plot(JB[:,0],P_S2/4.)
ax.plot(JB[:,0],JB[:,6],'--')

ax=plt.subplot(144)
ax.set_xscale('log')
#ax.set_yscale('log')


ax.plot(JB[:,0],P_S2/4./JB[:,6])


plt.show()