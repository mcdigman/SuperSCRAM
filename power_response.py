import numpy as np
from FASTPTcode import FASTPT
from halofit import halofitPk as halofit

def derv(x,f):
    return np.gradient(f)/np.gradient(x)

class power_response:
    def __init__(self,k,P_lin,cosmopie):
    
        self.CosmoPie=cosmopie
        
        self.k=k
        self.P_lin=P_lin
        
    def linear_response(self,z):
        # first get d ln P/ d bar{\delta}
        P=self.CosmoPie.G_norm(z)**2*self.P_lin
        dp=68/21. - 1/3.*derv(np.log(self.k), np.log(self.k**3*P))
        return dp/P
        
    def SPT_response(self,z):
        P=self.P_lin
        G=self.CosmoPie.G_norm(z)
        fastpt=FASTPT.FASTPT(self.k,-2,low_extrap=-5,high_extrap=5,n_pad=500) 
        P_spt=fastpt.one_loop(P,C_window=.65)*G**4
        spt=G**2*P + P_spt
        
        dp=68/21. - 1/3.*derv(np.log(self.k), np.log(self.k**3*P)) + 26/21.*P_spt/spt
        # also return P for a check 
        return dp/spt, spt 
    
    def halofit_response(self,z,delta_bar_0=0.01):
        sig_8=self.CosmoPie.sigma_r(0,8)
        sig_8_new=(1+13/21.*delta_bar_0)*sig_8
        
        P_new=sig_8_new/sig_8*self.P_lin
        
        hf1=halofit(self.k,p_lin=self.P_lin,C=self.CosmoPie())
        hf2=halofit(self.k,p_lin=P_new,C=Self.CosmoPie())
        
        P_hf=hf1.P_NL(k)
        
        dphf=(np.log(hf2.P_NL(k))-np.log(P_hf))/(13/21*delta_bar_0)
        
        dp=(13/21.*dphf + 2 - 1/3.*derv(np.log(k**3*P_hf),np.log(k)))/P_hf
        
        return dp, P_hf
        
        
if __name__=="__main__":
    
    d=np.loadtxt('Pk_Planck15.dat')
    k=d[:,0]; P=d[:,1]
    
    from cosmopie import CosmoPie
    
    CosmoPie=CosmoPie(k=k, P_lin=P)
    print k[-1], P[-1]
    hf1=halofit(k,p_lin=P)
    sys.exit()

    
    PR=power_response(k,P,CosmoPie)
    dp_lin=PR.linear_response(0)
    dp_spt,spt=PR.SPT_response(0)
    dp_hf, P_hf=PR.halofit_response(0)
    
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
    
    fig=plt.figure(figsize=(14,10))
    ax=fig.add_subplot(111)
    ax.set_xlim(0,1)
    ax.set_xlabel(r'$k$ [$h$/Mpc]', size=30)
    ax.set_ylabel(r'$ d\; \ln P/d\bar{\delta}$', size=30)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', width=2, length=10)
    ax.tick_params(axis='both', which='minor', width=1, length=5)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%2.2f'))
    ax.xaxis.labelpad = 20

    
   
    ax.plot(k,dp_lin*P, lw=4, color='black', label='linear')
    ax.plot(k,dp_spt*spt,'--', lw=4, color='black', label='SPT')
    ax.plot(k,dp_hf*P_hf,lw=2, color='black', label='halofit')
    
    
    plt.legend(loc=4, fontsize=30)
    plt.grid()
    plt.show()
        
    
        