import numpy as np
from numpy import pi 
import sys



class halofitPk(object):
		''' python version of Halofit, orginally presented in Smith 2003. Bibtex entry for Smith 2003
				et al. is below. 
				
				@MISC{2014ascl.soft02032P,
				author = {{Peacock}, J.~A. and {Smith}, R.~E.},
				title = "{HALOFIT: Nonlinear distribution of cosmological mass and galaxies}",
				howpublished = {Astrophysics Source Code Library},
				year = 2014,
				month = feb,
				archivePrefix = "ascl",
				eprint = {1402.032},
				adsurl = {http://adsabs.harvard.edu/abs/2014ascl.soft02032P},
				adsnote = {Provided by the SAO/NASA Astrophysics Data System}
				}
		
		'''
		def __init__(self,z,cosmology):
				
				for key in cosmology:
					 self.Omegam=cosmology['Omegam']
					 self.OmegaL=cosmology['OmegaL']
					 # note that capital O in Omegas will correspond to the present
					 # day values 
					 self.h=cosmology['h']
					 self.sig8=cosmology['sigma8']

				self.H_0=100*self.h
				self.z=z 
				self.gams=.21# what is the number ?

				self.spectral_parameters()
			

		
		def spectral_parameters(self):

			# get the evolved omega matter and omega Lambda 
			self.Om=self.omega_m(self.z)
			self.OL=self.omega_L(self.z)

			growth=self.G(self.z,self.Om, self.OL)
			growth_norm=self.G(0,self.Omegam, self.OmegaL)

			#normalize the growth factor 
			growth=growth/growth_norm

			self.amp=growth/(1 +self.z)

			xlogr1=-2
			xlogr2=3.5 

			''' iterate to determine the wavenumber where
					nonlinear effects become important (rknl), 
					the effectice spectral index (rneff),
					the second derivative of the power spectrum at rknl (rncur) 
			''' 
			while True:
				rmid = 10**( 0.5*(xlogr2+xlogr1) )
						
				sig,d1,d2 = self.wint(rmid)
						
				diff = sig-1.0
						
				if diff > 0.001:
					xlogr1 = np.log10(rmid)
					continue
				elif diff < -0.001:
					xlogr2 = np.log10(rmid)
					continue
				else:
					self.rknl = 1./rmid
					self.rneff = -3-d1
					self.rncur = -d2
								
					break
		
		def E(self,z):
			# for a flat universe with no radiation 
			return self.Omegam*(1+z)**3 + self.OmegaL
				 
		def omega_m(self,z):
			# \Omega_m(z) = \Omega_{m,0}(1+z)^3 H_0^2/H(z)^2
			#return self.Omegam*(1+z)**3/self.E(z)
			a = 1./(1.+z)
			Ok = 1.-self.Omegam-self.OmegaL
			omega_t = 1.0 - Ok / (Ok + self.OmegaL*a*a + self.Omegam/a)
			return omega_t * self.Omegam / (self.Omegam + self.OmegaL*a*a*a) 
		
		def omega_L(self,z):
			# \Omega_L(z) = \Omega_{L,0} H_0^2/H(z)^2
			#return self.OmegaL/self.E(z) 
			a = 1./(1.+z)
			Ok = 1.-self.Omegam-self.OmegaL
			omega_t = 1.0 - Ok / (Ok + self.Omegam*a*a + self.Omegam/a)
			return omega_t*self.OmegaL / (self.OmegaL+ self.Omegam/a/a/a)


		def G(self,z,Om,OL):
			# Growth factor, using Carrol, Press & Turner (1992) 
			# approximation 
			return 2.5*Om/(Om**4/7. - OL + (1 + Om/2.)*(1 + Om/70.))

		def wint(self,r):
			'''
			fint the effectice spectral quantities
			rkn;, rneff, rncurr.  Uses a Gaussian filter at the 
			to determine the scale at which the variance is unity, to 
			fing rknl. rneff is defined as the first derivative of the 
			variance, calculated at the nonlinear wavenumber. rncur si the 
			second derivative of variance at rknl. 
			'''

			nint=3000.
			t = ( np.arange(nint)+0.5 )/nint
			y = 1./t - 1.
			rk = y
			d2 = self.D2_L(rk)
			x2 = y*y*r*r
			w1=np.exp(-x2)
			w2=2*x2*w1
			w3=4*x2*(1-x2)*w1

			mult = d2/y/t/t
				
			sum1 = np.sum(w1*mult)/nint
			sum2 = np.sum(w2*mult)/nint
			sum3 = np.sum(w3*mult)/nint
				
			sig = np.sqrt(sum1)
			d1  = -sum2/sum1
			d2  = -sum2*sum2/sum1/sum1 - sum3/sum1
				
			return sig,d1,d2
		


		def D2_L(self,rk):
			# the linear power spectrum 
			rk=np.asarray(rk)
			return self.amp*self.amp*self.p_cdm(rk)

		def p_cdm(self,rk):
			# unormalized power spectrum 

			rk = np.asarray(rk)
			p_index = 1.
			rkeff=0.172+0.011*np.log(self.gams/0.36)*np.log(self.gams/0.36)
			q=1.e-20 + rk/self.gams
			q8=1.e-20 + rkeff/self.gams
			tk=1/(1+(6.4*q+(3.0*q)**1.5+(1.7*q)**2)**1.13)**(1/1.13)
			tk8=1/(1+(6.4*q8+(3.0*q8)**1.5+(1.7*q8)**2)**1.13)**(1/1.13)
			return self.sig8*self.sig8*((q/q8)**(3.+p_index))*tk*tk/tk8/tk8

		def D2_NL(self,rk,return_components = False):
				"""
				halo model nonlinear fitting formula as described in 
				Appendix C of Smith et al. (2002)
				"""
				rk = np.asarray(rk)
				rn    = self.rneff
				rncur = self.rncur
				rknl  = self.rknl
				plin  = self.D2_L(rk)
				om_m  = self.Om
				om_v  = self.OL
				
				gam=0.86485+0.2989*rn+0.1631*rncur
				a=10**(1.4861+1.83693*rn+1.67618*rn*rn+0.7940*rn*rn*rn+\
							 0.1670756*rn*rn*rn*rn-0.620695*rncur)
				b=10**(0.9463+0.9466*rn+0.3084*rn*rn-0.940*rncur)
				c=10**(-0.2807+0.6669*rn+0.3214*rn*rn-0.0793*rncur)
				xmu=10**(-3.54419+0.19086*rn)
				xnu=10**(0.95897+1.2857*rn)
				alpha=1.38848+0.3701*rn-0.1452*rn*rn
				beta=0.8291+0.9854*rn+0.3400*rn**2
				
				if abs(1-om_m) > 0.01: #omega evolution
						f1a=om_m**(-0.0732)
						f2a=om_m**(-0.1423)
						f3a=om_m**(0.0725)
						f1b=om_m**(-0.0307)
						f2b=om_m**(-0.0585)
						f3b=om_m**(0.0743)       
						frac=om_v/(1.-om_m) 
						f1=frac*f1b + (1-frac)*f1a
						f2=frac*f2b + (1-frac)*f2a
						f3=frac*f3b + (1-frac)*f3a
				else:         
						f1=1.0
						f2=1.0
						f3=1.0

				y=(rk/rknl)
				
				ph = a*y**(f1*3)/(1+b*y**(f2)+(f3*c*y)**(3-gam))
				ph /= (1+xmu*y**(-1)+xnu*y**(-2))
				pq = plin*(1+plin)**beta/(1+plin*alpha)*np.exp(-y/4.0-y**2/8.0)
				
				pnl=pq+ph

				if return_components:
						return pnl,pq,ph,plin
				else:
						return pnl

		def P_NL(self,k,return_components = False):
			if(return_components):
				pnl,pq,ph,plin=self.D2_NL(k,return_components)
				return pnl/k**3*(2*pi**2), pq, ph, plin/k**3*(2*pi**2)
			else:
				return D2_NL(k,return_components)/k**3*(2*pi**2)

				

if __name__=="__main__":

		cosmology={'Omegabh2' :0.02230,
				 'Omegach2' :0.1188,
					 'Omegamh2' : 0.14170,
				 'OmegaL'   : .6911,
				 'Omegam'   : .3089,
				 'H0'       : 67.74, 
				 'sigma8'   : .8159, 
				 'h'        :.6774, 
				 'Omegak'   : 0.0, # check on this value 
				 'Omegar'   : 0.0 # check this value too
				 }

		HF=halofitPk(0,cosmology)
		h=.6774
		d1=np.loadtxt('test_data/class_halofit_z0.dat')
		k=d1[:,0]; P1=d1[:,1]
		#k=np.logspace(-2,2,500)
		
		P,pq,ph,Plin=HF.P_NL(k, return_components=True)
	
		import matplotlib.pyplot as plt

		ax=plt.subplot(111)
		ax.set_xscale('log')
		ax.set_yscale('log')
		

				
		ax.plot(k,P, label='halofit')
		ax.plot(k,Plin, label='linear')
		ax.plot(k/h,P1*h**3, label='class halofit')
		
		plt.legend(loc=1)
		plt.grid()
		plt.show()
