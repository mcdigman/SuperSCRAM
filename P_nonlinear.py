import numpy as np
from FASTPTcode import FASTPT 


class P_nonlinear():
    
    def __init__(self,P_lin,z,k,C_pie,types=['fastpt']):
        ''' 
           P_lin : the linear power spectrum for a given cosmology 
           z : an array of redshifts
           k : the wave vector grid the power spectrum is sampled on 
           C_pie : is a class made with cosmopie class that contians relevant things (like growth factor)
        '''
       
        self.G=C_pie.G_array(z)
        self.P_lin=P_lin
        self.k=k
        for i in range(len(types)):
            if types[i]=='fastpt':
                self.fastpt=FASTPT.FASTPT(k,-2,n_pad=500)
        
    def SPT(self):
        P_out=np.zeros((self.k.size,self.G.size))
        P_spt=self.fastpt.one_loop(P,C_window=.65) 
        #maybe should use the fastpt P_lin too 
        for i in range(self.G.size):
            P_out[:,i]=self.G[i]**2*self.P_lin+ self.G[i]**4*P_spt
        
        return P_out 
    
    def bias(self,bias_params):
        #maybe bias params should be a matrix to account for different redshifts. 
        b1,b2,bs=bias_params
        P_out=np.zeros((self.k.size,self.G.size))
        Power, Pd1d2, Pd2d2, Pd1s2, Pd2s2, Ps2s2, sig4=self.fastpt.P_bias(P,C_window=.65) 
        for i in range(self.G.size):
            #P_out[:,i]=self.G[i]**2*self.P_lin+ self.G[i]**4*P_spt
            print 'finish this part'
        
    
    


if __name__=="__main__":
    
    import cosmopie
    C_pie=cosmopie.CosmoPie()
    z=np.array([0,.5,.9])
    
    d=np.loadtxt('Pk_Planck15.dat')
    k=d[:,0]; P=d[:,1]
    
    PNL=P_nonlinear(P,z,k,C_pie)
    x=PNL.SPT()
    import matplotlib.pyplot as plt
    
    ax=plt.subplot(111)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    ax.plot(k, x[:,0])
    ax.plot(k, x[:,1])
    ax.plot(k, x[:,2])
    
    plt.show()
    
    