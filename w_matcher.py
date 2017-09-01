import cosmopie as cp
import defaults
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline,RectBivariateSpline,SmoothBivariateSpline
from scipy.integrate import cumtrapz
from warnings import warn
class WMatcher:
    def __init__(self,C_fid,wmatcher_params=defaults.wmatcher_params):
        self.C_fid = C_fid
        self.cosmo_fid = C_fid.cosmology.copy()
        self.cosmo_fid['w']=-1
        self.cosmo_fid['de_model']='constant_w'

        self.w_step = wmatcher_params['w_step'] 
        self.w_min = wmatcher_params['w_min']
        self.w_max = wmatcher_params['w_max']
        self.ws = np.arange(self.w_min,self.w_max,self.w_step)
        self.n_w = self.ws.size 

        self.a_step = wmatcher_params['a_step'] 
        self.a_min = wmatcher_params['a_min']
        self.a_max = wmatcher_params['a_max']
        self.a_s = np.arange(self.a_min,self.a_max,self.a_step)
        self.n_a = self.a_s.size 

        self.zs = 1./self.a_s-1.#np.arange(self.a_min,self.a_max,self.a_step)

        self.Cs = np.zeros(self.n_w,dtype=object)
        self.cosmos = np.zeros(self.n_w,dtype=object)

        #self.E_as = np.zeros((self.n_w,self.n_a))
        self.integ_Es = np.zeros((self.n_w,self.n_a))
        self.Gs = np.zeros((self.n_w,self.n_a))
            

        for i in xrange(0,self.n_w):
           self.cosmos[i] = self.cosmo_fid.copy()
           self.cosmos[i]['w'] = self.ws[i]
           self.Cs[i] = cp.CosmoPie(cosmology=self.cosmos[i],silent=True)
           E_as = self.Cs[i].Ez(1./self.a_s-1.)
           self.integ_Es[i] = cumtrapz(1./(self.a_s**2*E_as),self.a_s,initial=0.)
           #self.integ_Es[i] = InterpolatedUnivariateSpline(self.integ_as,integ_Es_i,k=2,ext=2)(self.a_s) 
           self.Gs[i] = self.Cs[i].G(self.zs)
        self.G_interp = RectBivariateSpline(self.ws,self.a_s,self.Gs,kx=2,ky=2)
        #self.integ_E_interp = RectBivariateSpline(self.ws,self.a_s,self.integ_Es,kx=2,ky=2)

        self.ind_switches = np.argmax(np.diff(self.integ_Es,axis=0)<0,axis=0)+1
        #there is a purely numerical issue that causes the integral to be non-monotonic, this loop eliminates the spurious behavior
        for i in xrange(1,self.n_a):
            if self.ind_switches[i]>1:
                if self.integ_Es[self.ind_switches[i]-1,i]-self.integ_Es[0,i]>=0:
                    self.integ_Es[0:(self.ind_switches[i]-1),i]=self.integ_Es[self.ind_switches[i]-1,i]
                else:
                    raise RuntimeError( "Nonmonotonic integral, solution is not unique at "+str(self.a_s[i]))

        self.integ_E_interp = RectBivariateSpline(self.ws,self.a_s,self.integ_Es,kx=2,ky=2)
             
    #accurate to within numerical precision
    #match effective constant w as in casarini paper
    def match_w(self,C_in,z_match):
        a_match = 1./(1.+z_match)
        E_in = C_in.Ez(1./self.a_s-1.)    
        integ_E_in = cumtrapz(1./(self.a_s**2*E_in),self.a_s,initial=0.)
        integ_E_in_interp = InterpolatedUnivariateSpline(self.a_s,integ_E_in,k=2,ext=2)
        integ_E_targets = integ_E_in_interp(a_match)
        w_grid1 = np.zeros(a_match.size)
        for itr in xrange(0,z_match.size):
            iE_vals = self.integ_E_interp(self.ws,a_match[itr]).T[0]-integ_E_targets[itr]
            iG = np.argmax(iE_vals<=0.)
            #require some padding so can get very accurate interpolation results
            if iG-2>=0 and iG+2<self.ws.size:
                w_grid1[itr] = InterpolatedUnivariateSpline(iE_vals[iG-2:iG+2][::-1],self.ws[iG-2:iG+2:][::-1],k=2)(0.) 
            else:
                warn("w is too close to edge of range, using nearest neighbor w, consider expanding w range")
                w_grid1[itr] = self.ws[iG]
        return w_grid1


    #does not work
#    def match_w2(self,C_in,z_match):
#        a_match = 1./(1.+z_match)
#        E_in = C_in.Ez(1./self.integ_as-1.)    
#        integ_E_in_init = cumtrapz(1./(self.integ_as**2*E_in),self.integ_as,initial=0.)
#        integ_E_in_interp = InterpolatedUnivariateSpline(self.integ_as,integ_E_in_init,k=2,ext=2)
#        integ_E_in = integ_E_in_interp(self.a_s)
#        integ_E_targets = integ_E_in_interp(a_match)
#        w_grid1 = np.zeros(a_match.size)
#        for itr in xrange(0,a_match.size):
#            for i in xrange(0,self.n_a):
#                if a_match[itr]<self.a_s[i]:
#                    print "searching at a="+str(a_match[itr])+" to match ai="+str(self.a_s[i])+" i="+str(i)
#                    if self.ind_switches[i]>1:
#                        interp1 = InterpolatedUnivariateSpline(self.integ_Es[self.ind_switches[i]-1::,i][::-1],self.ws[self.ind_switches[i]-1::][::-1],k=2,ext=2)
#                    else:
#                        interp1 = InterpolatedUnivariateSpline(self.integ_Es[0::,i][::-1],self.ws[0::][::-1],k=2,ext=2)
#                    print "need iE="+str(integ_E_targets[itr])+" range ("+str(np.min(self.integ_Es[0::,i][::-1]))+","+str(np.max(self.integ_Es[0::,i][::-1]))+")"
#                    print "would have between w="+str(interp1(integ_E_in[i-1]))+" and w="+str(interp1(integ_E_in[i]))
#                    w_grid1[itr] = interp1(integ_E_targets[itr])
#                    print "at i="+str(i)+" found w="+str(w_grid1[itr])+ " for itr="+str(itr)
#                    break
#        return w_grid1
#
#    #fairly accurate, would be more accurate if used better interpolation 
#    def match_w3(self,C_in,z_match,width_interp=3):
#        a_match = 1./(1.+z_match)
#        E_in = C_in.Ez(1./self.a_s-1.)    
#        integ_E_in = cumtrapz(1./(self.a_s**2*E_in),self.a_s,initial=0.)
#        integ_E_in_interp = InterpolatedUnivariateSpline(self.a_s,integ_E_in,k=3,ext=2)
#        w_grid = np.zeros(self.n_a)
#        
#        #for silencing excessive warnings
#        warned1 = 0
#        warned2 = 0
#
#        #start from 1 because there is a ghost cell at start because cumtrapz is 0 in first cell
#        for i in xrange(1,self.n_a):
#            if (self.integ_Es[:,i][-1]>=integ_E_in[i] or self.integ_Es[:,i][0]<=integ_E_in[i]):
#                #if np.any(1./(1.+z_match))
#                last_val = np.abs(integ_E_in[i]-self.integ_Es[0,i])
#                last_dir = np.sign(integ_E_in[i]-self.integ_Es[0,i])
#                dir_st = np.sign(integ_E_in[i]-self.integ_Es[:,i])
#                diff_st = np.sign(np.diff(self.integ_Es[:,i]))
#                possible_w_is = np.array([],dtype=int)
#                for itr in xrange(1,self.n_w):
#                    this_dir = dir_st[itr]
#                    if ((not this_dir==last_dir) and (not last_dir==0)) :#and last_dir==diff_st[itr-1]): 
#                        possible_w_is=np.hstack((possible_w_is,np.array([itr])))
#                    elif this_dir==0:
#                        possible_w_is=np.hstack((possible_w_is,np.array([itr])))
#                    elif (last_dir==diff_st[itr-1]): #and not diff_st[itr]==diff_st[itr-1]):
#                        possible_w_is=np.hstack((possible_w_is,np.array([itr])))
#                    last_dir = this_dir
#                if possible_w_is.size==0:
#                    warn('failed')
#                    
#                possible_ws = self.ws[possible_w_is] 
#
#                if possible_ws.size==0 and self.a_s[i]>=np.min(a_match) and self.a_s[i]<=np.max(a_match):
#                    print "crashing at i="+str(i)+", wanted iE="+str(integ_E_in[i])+" max="+str(np.max(self.integ_Es[:,i]))+" min="+str(np.min(self.integ_Es[:,i]))+" max-iE="+str(np.max(self.integ_Es[:,i])-integ_E_in[i])+" min-iE="+str(np.min(self.integ_Es[:,i])-integ_E_in[i])
#                    raise ValueError('unable to find minima, extrapolation probably needed,try increasing range')
#                else:
#                    possible_w_is = np.array([self.ind_switches[i]])
#                    possible_ws = self.ws[possible_w_is]
#
#                if possible_ws.size>1 and self.a_s[i]>np.min(a_match) and self.a_s[i]<np.max(a_match):
#                    if warned1==0:
#                        print "warning, wanted iE="+str(integ_E_in[i])+" max="+str(np.max(self.integ_Es[:,i]))+" min="+str(np.min(self.integ_Es[:,i]))
#                        warn('ambiguity between w vals '+str(possible_ws)+' choosing closest to '+str(C_in.w_interp(1./self.a_s[i]-1.)))
#                    warned1+=1
#                w_choose_i = possible_w_is[np.argmin(np.abs(possible_ws-C_in.w_interp(1./self.a_s[i]-1.)))]
#            
#                i_great = np.argmin(self.ws[w_choose_i]>self.ws)
#                
#            else:
#                i_great = np.argmin(integ_E_in[i]<self.integ_Es[:,i])
#            #use linear interpolation between nearest points sampled
#            #average over forward and backwards because of uneven x spacing
#            #note may be inaccurate if (self.integ_Es[i_great,i]-self.integ_Es[i_great-1,i])~0,
#            #TODO could probably do somewhat better with higher order approximation but must make sure function values are unique
#             
#            if i_great>0 and i_great<self.n_w-1:
#                w_1 = self.ws[i_great-1]+(integ_E_in[i]-self.integ_Es[i_great-1,i])*(self.ws[i_great]-self.ws[i_great-1])/(self.integ_Es[i_great,i]-self.integ_Es[i_great-1,i])
#                w_2 = self.ws[i_great]+(integ_E_in[i]-self.integ_Es[i_great,i])*(self.ws[i_great+1]-self.ws[i_great])/(self.integ_Es[i_great+1,i]-self.integ_Es[i_great,i])
#                w_grid[i] = (w_1+w_2)/2.
#                if w_grid[i]<np.min(self.ws) or w_grid[i]>np.max(self.ws):
#                    print "as i="+str(i)+" w="+str(w_grid[i])+" off w1="+str(w_1)+" w2="+str(w_2)+" ig="+str(i_great)+" ws[i_g]="+str(self.ws[i_great])+" ws[i_g-1]="+str(self.ws[i_great-1])+" ws[i_g+1]="+str(self.ws[i_great+1])+" integ_Es[ig,i]="+str(self.integ_Es[i_great,i])+" integ_Es[ig+1,i]="+str(self.integ_Es[i_great+1,i])
#                    if i==142:
#                        import matplotlib.pyplot as plt
#                        plt.plot(self.ws,self.integ_Es[:,i])
#                        plt.show()
#            else:
#                w_grid[i] = self.ws[i_great]
#                if self.a_s[i]>np.min(a_match) and self.a_s[i]<np.max(a_match):
#                    if warned2==0:
#                        warn('extrapolation may be necessary')
#                    warned2+=1
#        #extrapolate for ghost cell
#        #TODO bisection corrections are probably a good idea  
#        #TODO check extrapolation
#        w_grid[0] = 2.*w_grid[1]-w_grid[2]
#        #return interpolated effective constant w
#        return InterpolatedUnivariateSpline(self.a_s,w_grid,k=1,ext=2)(1./(1.+z_match))
    #matches the redshift dependence of the growth factor for the input model to the equivalent model with constant w
    def match_growth(self,C_in,z_in,w_in):
        a_in = 1./(1.+z_in)
        G_norm_ins = C_in.G_norm(z_in)
        n_z_in = z_in.size
        pow_mult = np.zeros(n_z_in)
        #G_norm_fid = self.C_fid.G_norm(z_in)
        #TODO vectorize correctly
        for itr in xrange(0,n_z_in):
            pow_mult[itr]=(G_norm_ins[itr]/(self.G_interp(w_in[itr],a_in[itr])/self.G_interp(w_in[itr],1.)))**2
        #return multiplier for linear power spectrum from effective constant w model
        return pow_mult

    #get an interpolated growth factor for a given w, a_in is a vector 
    def growth_interp(self,w_in,a_in):
        return self.G_interp(w_in,a_in).T

    #match scaling (ie sigma8) for the input model compared to the fiducial model
    def match_scale(self,z_in,w_in):
#        a_in = 1./(1.+z_in)
#        G_norm_ins = C_in.G_norm(z_in)
        n_z_in = z_in.size
        pow_scale = np.zeros(n_z_in)
        G_fid = self.C_fid.G(0)
        #TODO vectorize correctly
        for itr in xrange(0,n_z_in):
            pow_scale[itr]=(self.G_interp(w_in[itr],1.)/G_fid)**2
        #return multiplier for linear power spectrum from effective constant w model
        return pow_scale
        

if __name__=='__main__':
    cosmo_start = defaults.cosmology.copy()
    cosmo_start['w'] = -1
    cosmo_start['de_model']='constant_w'
    cosmo_match_a = cosmo_start.copy()
    cosmo_match_a['de_model']='w0wa'

    do_w_int_test=True
    if do_w_int_test:
        cosmo_match_a['w0'] = -1.
        cosmo_match_a['wa'] = 0.
        cosmo_match_a['w'] = -1.

        wmatcher_params = defaults.wmatcher_params.copy()
        C_start = cp.CosmoPie(cosmology=cosmo_start)
        C_match_a = cp.CosmoPie(cosmology=cosmo_match_a)

        wm = WMatcher(C_start)
        a_grid = np.arange(1.001,0.001,-0.01)
        zs = 1./a_grid-1.

      #  zs = np.arange(0.,20.,0.5)

        w1s = wm.match_w(C_match_a,zs)

        #w2s = wm.match_w2(C_match_a,zs)
        #w3s = wm.match_w3(C_match_a,zs)
        #print "rms discrepancy methods 1 and 3="+str(np.linalg.norm(w3s-w1s)/w1s.size)
        print "rms discrepancy method 1 and input="+str(np.linalg.norm(w1s-C_match_a.w_interp(a_grid))/w1s.size)
        import matplotlib.pyplot as plt
        a_s = 1./(1.+zs)
        plt.plot(a_s,w1s)
        #plt.plot(a_s,w3s)
        plt.show()


    do_match_casarini=False
    if do_match_casarini:
        cosmo_match_a['w0'] = -1.2
        cosmo_match_a['wa'] = 0.5
        cosmo_match_a['w'] = -1.2

        cosmo_match_b = cosmo_match_a.copy()
        cosmo_match_b['w0'] = -0.6
        cosmo_match_b['wa'] = -1.5
        cosmo_match_b['w'] = -0.6


        wmatcher_params = defaults.wmatcher_params.copy()
        C_start = cp.CosmoPie(cosmology=cosmo_start)
        C_match_a = cp.CosmoPie(cosmology=cosmo_match_a)
        C_match_b = cp.CosmoPie(cosmology=cosmo_match_b)

        wm = WMatcher(C_start)
        zs = np.arange(0.,1.51,0.05)

        ws_a = wm.match_w(C_match_a,zs)
        ws_b = wm.match_w(C_match_b,zs)

        #w1s,w2s = wm.match_w2(C_match,zs)
        pow_mults_a = wm.match_growth(C_match_a,zs,ws_a)
        pow_mults_b = wm.match_growth(C_match_b,zs,ws_b)
        #print wm.match_w(C_start,np.array([1.0]))
#        import matplotlib.pyplot as plt
        a_s = 1./(1.+zs)
        #plt.plot(a_s,ws)
        #should match arXiv:1601.07230v3 figure 2
        plt.plot(zs,(cosmo_match_a['w0']+(1-a_s)*cosmo_match_a['wa']))
        plt.plot(zs,ws_a)
        plt.ylim([-1.55,-0.4])
        plt.xlim([1.5,0.])
        plt.show()

        plt.plot(zs,(cosmo_match_b['w0']+(1-a_s)*cosmo_match_b['wa']))
        plt.plot(zs,ws_b)
        plt.ylim([-1.55,-0.4])
        plt.xlim([1.5,0.])
        plt.show()

        plt.plot(zs,np.sqrt(pow_mults_a)*0.83)
        plt.plot(zs,np.sqrt(pow_mults_b)*0.83)
        plt.xlim([1.5,0.])
        plt.ylim([0.8,0.9])
        plt.show()
        #plt.plot(wm.w_Es)
        #plt.show()
