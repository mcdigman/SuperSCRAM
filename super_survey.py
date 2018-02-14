"""class for manipulating and plotting the Super Sample term"""
from time import time

import numpy as np

import multi_fisher as mf
#TODO maybe make possible to serialize/pickle or at least provide a standardized way of storing run information
#TODO examine possibile platform dependence bug with ruby
class SuperSurvey(object):
    """ This class holds and returns information for all surveys
    """
    def __init__(self,surveys_sw,surveys_lw,basis,C,prior_params,get_a=False,do_mitigated=True,do_unmitigated=True):
        r"""master class for mitigation analysis
            inputs:
                surveys_sw: an array of SWSurveys
                surveys_lw: an array of LWSurveys (different mitigations)
                basis: an LWBasis object
                C: a CosmoPie object
                prior_params: parameters needed for PriorFisher
                get_a: get (v.T).C_lw.v where v=\frac{\partial\bar{\delta}}{\delta_\alpha}
                do_mitigated: if False don't get mitigated covariances
                do_unmitigate: if False don't get unmitigated covariances"""
        t1 = time()

        self.get_a = get_a
        self.do_mitigated = do_mitigated
        self.do_unmitigated = do_unmitigated

        self.surveys_lw = surveys_lw
        self.surveys_sw = surveys_sw
        self.N_surveys_sw = surveys_sw.size
        self.N_surveys_lw = surveys_lw.size
        print'SuperSurvey: this is the number of surveys', self.N_surveys_sw, self.N_surveys_lw

        self.C = C

        self.basis = basis
        self.N_O_I = 0
        self.N_O_a = 0

        for i in xrange(self.N_surveys_sw):
            self.N_O_I = self.N_O_I + surveys_sw[i].get_N_O_I()

        for i in xrange(self.N_surveys_lw):
            self.N_O_a = self.N_O_a + surveys_lw[i].get_N_O_a()

        print 'SuperSurvey: there are '+str(self.N_O_I)+' short wavelength and '+str(self.N_O_a)+' long wavelength observables'

        #TODO support multiple sw surveys with MultiFisher
        self.multi_f = mf.MultiFisher(basis,surveys_sw[0],surveys_lw,prior_params,needs_a=self.get_a,do_mit=self.do_mitigated)

        self.f_set_nopriors = self.multi_f.get_fisher_set(include_priors=False)
        self.f_set = self.multi_f.get_fisher_set(include_priors=True)
        self.eig_set = self.multi_f.get_eig_set(self.f_set_nopriors)
        self.eig_set_ssc = self.multi_f.get_eig_set(self.f_set_nopriors,ssc_metric=True)
        #TODO get this before changing shape in multi_fisher
        if self.get_a:
            self.a_vals = self.multi_f.get_a_lw()
            print "SuperSurvey: mitigated run gave a="+str(self.a_vals)
        else:
            self.a_vals = np.zeros(2)

        t2 = time()
        print 'SuperSurvey: all done'
        print 'SuperSurvey: run time', t2-t1

    def print_standard_analysis(self):
        """"print a standard set of analyses"""
        print "SuperSurvey: Summary analysis"
        print "----------------------------------------------------"
        print "----------------------------------------------------"
        print "top 2 eigenvalues without mitigation using gaussian metric: "+str(self.eig_set[1,0][0][-1])+","+str(self.eig_set[1,0][0][-2])
        print "top 2 eigenvalues with mitigation using gaussian metric: "+str(self.eig_set[1,1][0][-1])+","+str(self.eig_set[1,1][0][-2])
        print "bottom 2 eigenvalues with mitigation using unmitigated metric: "+str(self.eig_set_ssc[1,1][0][0])+","+str(self.eig_set_ssc[1,1][0][1])
        print "number eigenvalues>1.00000001 without mitigation using gaussian metric: "+str(np.sum(np.abs(self.eig_set[1,0][0])>1.00000001))
        print "number eigenvalues>1.00000001 with mitigation using gaussian metric: "+str(np.sum(np.abs(self.eig_set[1,1][0])>1.00000001))
        print "number eigenvalues<0.99999999 with mitigation using unmitigated metric: "+str(np.sum(np.abs(self.eig_set_ssc[1,1][0])<0.99999999))
        #print "components of eigenvectors, sorted by descending importance, eigenvectors sorted by descending contamination without mitigation"
        #for i in xrange(0,6):
        #    comp_no_mit = lambda itr1,itr2: cmp(np.abs(self.eig_set[1,0][1][itr1,-1-i]),np.abs(self.eig_set[1,0][1][itr2,-1-i]))
        #    list_no_mit = np.arange(self.eig_set[1,0][0].size)
        #    list_no_mit = sorted(list_no_mit,comp_no_mit)
        #    print "descending order of component along worst direction without mitigation using gaussian metric: "+str(self.surveys_sw[0].cosmo_par_list[list_no_mit][::-1])
        #for i in xrange(0,6):
        #    comp_mit = lambda itr1,itr2: cmp(np.abs(self.eig_set[1,1][1][itr1,-1-i]),np.abs(self.eig_set[1,1][1][itr2,-1-i]))
        #    list_mit = np.array(np.arange(self.eig_set[1,1][0].size))
        #    list_mit = np.array(sorted(list_mit,comp_mit))
        #    print "descending order of component along worst direction with mitigation using gaussian metric: "+str(self.surveys_sw[0].cosmo_par_list[list_mit][::-1])
        #for i in xrange(0,6):
        #    comp_ssc = lambda itr1,itr2: cmp(np.abs(self.eig_set_ssc[1,1][1][itr1,i]),np.abs(self.eig_set_ssc[1,1][1][itr2,i]))
        #    list_ssc = np.arange(self.eig_set_ssc[1,1][0].size)
        #    list_ssc = np.array(sorted(list_ssc,comp_ssc))
        #    print "descending order of component along most improved direction with mitigation using unmitigated metric: "+str(self.surveys_sw[0].cosmo_par_list[list_ssc][::-1])
#        dchi2 = 2.3
#        width1s_g,width2s_g,angles_g,areas_g = get_ellipse_specs(self.f_set_nopriors[0][2].get_covar(),dchi2=dchi2)
#        width1s_no_mit,width2s_no_mit,angles_no_mit,areas_no_mit = get_ellipse_specs(self.f_set_nopriors[1][2].get_covar(),dchi2=dchi2)
#        width1s_mit,width2s_mit,angles_mit,areas_mit = get_ellipse_specs(self.f_set_nopriors[2][2].get_covar(),dchi2=dchi2)
#        rat_no_mit_g = areas_no_mit/areas_g
#        rat_no_mit_g[np.isnan(rat_no_mit_g)] = 0.
#        rat_mit_g = areas_mit/areas_g
#        rat_mit_g[np.isnan(rat_mit_g)] = 0.
#        rat_mit_no_mit = areas_mit/areas_no_mit
#        rat_mit_no_mit[np.isnan(rat_mit_no_mit)] = 0.
        align1 = np.dot(self.eig_set[1,0][1][:,-1],self.eig_set[1,1][1][:,-1])
        align2 = np.dot(self.eig_set[1,0][1][:,-1],self.eig_set_ssc[1,1][1][:,0])
        align3 = np.dot(self.eig_set[1,0][1][:,-2],self.eig_set[1,1][1][:,-2])
        align4 = np.dot(self.eig_set[1,0][1][:,-2],self.eig_set_ssc[1,1][1][:,1])
        print "alignment of most contaminated direction before and most contaminated direction after mitigation: "+str(align1)
        print "alignment of most contaminated direction before mitigation and most improved direction: "+str(align2)
        print "alignment of second most contaminated direction before and second most contaminated direction after mitigation: "+str(align3)
        print "alignment of second most contaminated direction before mitigation and second most improved direction: "+str(align4)



        print "----------------------------------------------------"
        print "----------------------------------------------------"

    def make_standard_ellipse_plot(self,c_extra=None,include_base=True,dchi2=2.3,include_diag=True):
        """make a standardized ellipse plot for the object"""
        if c_extra is None:
            c_extra = np.zeros((0,self.surveys_sw[0].cosmo_par_list.size,self.surveys_sw[0].cosmo_par_list.size))
        extra_colors = np.zeros((c_extra.size,3))
        for i in xrange(0,c_extra.size):
            extra_colors[i] = np.random.rand(3)

        if include_base:
            no_mit_color = np.array([1.,0.,0.])
            mit_color = np.array([0.,1.,0.])
            g_color = np.array([0.,0.,1.])
            color_set = np.vstack([np.array([mit_color,no_mit_color,g_color]),extra_colors])
            opacity_set = np.full(3+c_extra.size,1.)#np.array([1.0,1.0,1.0])
            cov_set = np.vstack([np.array([self.f_set[2][2].get_covar(),self.f_set[1][2].get_covar(),self.f_set[0][2].get_covar()]),c_extra])
            label_set = np.hstack([np.array(["ssc+mit+g","ssc+g","g"]),np.full(c_extra.shape[0],'extra')])
        else:
            color_set = extra_colors
            opacity_set = np.full(c_extra.size,1.)#np.array([1.0,1.0,1.0])
            cov_set = c_extra
            label_set = np.full(c_extra.shape[0],'extra')
        #box_widths = np.array([0.015,0.005,0.0005,0.005,0.1,0.05])*3.
        #cov_set = np.array([SS.covs_params[1],SS.covs_params[0],SS.covs_g_pars[0]])
        make_ellipse_plot(cov_set,color_set,opacity_set,label_set,'adaptive',self.surveys_sw[0].cosmo_par_list,dchi2=dchi2,include_diag=include_diag)


def get_ellipse_specs(covs,dchi2=2.3):
    """Get the widths and angles for plotting covariance ellipses from a covariance matrix covs, with width dchi2"""
    a_s = np.zeros_like(covs)
    b_s = np.zeros_like(covs)
    #dchi2 = 2.3
    alpha = np.sqrt(dchi2)
    angles = np.zeros_like(covs)
    areas = np.zeros_like(covs)
    for i in xrange(0,covs.shape[0]):
        for j in xrange(0,covs.shape[1]):
            #cf arxiv:0906.4123 for explanation
            a_s[i,j] = np.sqrt((covs[i,i]+covs[j,j])/2.+np.sqrt((covs[i,i]-covs[j,j])**2/4.+covs[i,j]**2))
            b_s[i,j] = np.sqrt((covs[i,i]+covs[j,j])/2.-np.sqrt((covs[i,i]-covs[j,j])**2/4.+covs[i,j]**2))
            angles[i,j] = np.arctan2(2.*covs[i,j],covs[i,i]-covs[j,j])/2.
    width1s = a_s*alpha
    width2s = b_s*alpha
    areas = np.pi*width1s*width2s
    return width1s,width2s,angles,areas


FORMATTED_LABELS = {"ns":"$n_s$",
                    "Omegamh2":r"$\Omega_m h^2$",
                    "OmegaLh2":r"$\Omega_{de} h^2$",
                    "Omegabh2":r"$\Omega_b h^2$",
                    "LogAs":"$ln(A_s)$",
                    "As":"$A_s$",
                    "sigma8":r"$\sigma_8$",
                    "w0":"$w_0$",
                    "wa":"$w_a$",
                    "w":"w",
                    "Omegam":r"$\Omega_m$",
                    "OmegaL":r"$\Omega_{de}$",
                    "Omegab":r"$\Omega_b$",
                    "h":"h",
                    "H0":"H_0",
                    "Omegach2":r"$\Omega_c h^2$",
                    "Omegac":r"$\Omega_c$"
                   }
def make_ellipse_plot(cov_set,color_set,opacity_set,label_set,box_widths,cosmo_par_list,dchi2,adaptive_mult=1.05,include_diag=True):
    """make the plot of error ellipses given the set of covariance matrices"""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.patches import Ellipse
    n_p = cosmo_par_list.size
    if include_diag:
        _,ax_list = plt.subplots(n_p,n_p)
    else:
        _,ax_list = plt.subplots(n_p-1,n_p-1)
    n_c = cov_set.shape[0]
    n_p = cosmo_par_list.shape[0]
    width1_set = np.zeros((n_c,n_p,n_p))
    width2_set = np.zeros((n_c,n_p,n_p))
    angle_set = np.zeros((n_c,n_p,n_p))
    area_set = np.zeros((n_c,n_p,n_p))
    for itr3 in xrange(0,n_c):
        width1_set[itr3],width2_set[itr3],angle_set[itr3],area_set[itr3] = get_ellipse_specs(cov_set[itr3],dchi2=dchi2)
    if not isinstance(box_widths,np.ndarray) and box_widths=="adaptive":
        box_widths = np.zeros(n_p)
        for itr3 in xrange(0,n_c):
            #TODO make sigmas more selectable don't hardcode the 3
            box_widths = np.max(np.array([box_widths,3.*np.sqrt(np.diag(cov_set[itr3]))]),axis=0)
        box_widths*=adaptive_mult

    xbox_widths = box_widths
    ybox_widths = box_widths
    for itr1 in xrange(0,n_p):
        for itr2 in xrange(0,n_p):
            if not include_diag:
                if itr1==itr2 or itr2==0 or itr1==n_p-1:
                    continue
                else:
                    if n_p==2:
                        ax = ax_list
                    else:
                        ax = ax_list[itr2-1,itr1]
            else:
                if n_p==1:
                    ax = ax_list
                else:
                    ax = ax_list[itr2,itr1]
            #ax.set_aspect('equal')
            ax.set_axisbelow(True)
            if itr2<itr1:
                ax.axis('off')
                continue
            param1 = cosmo_par_list[itr1]
            param2 = cosmo_par_list[itr2]
            fid_point = np.array([0.,0.])

            es = np.zeros(n_c,dtype=object)
            ybox_width = 0.
            for itr3  in xrange(0,n_c):
                es[itr3] = Ellipse(fid_point,width1_set[itr3][itr1,itr2],width2_set[itr3][itr1,itr2],angle=180./np.pi*angle_set[itr3][itr1,itr2],label=label_set[itr3])
                if itr1==itr2:
                    xs = np.linspace(-xbox_widths[itr1]/2.,xbox_widths[itr1]/2.,200)
                    sigma = np.sqrt(cov_set[itr3][itr1,itr1])#width1_set[itr3][itr1,itr1]
                    ax.plot(xs,1./np.sqrt(2.*np.pi*sigma**2)*np.exp(-xs**2/(2.*sigma**2)),color=color_set[itr3])
                    ybox_width = np.max(np.array([ybox_width,np.max(1./np.sqrt(2.*np.pi*sigma**2)*np.exp(-xs**2/(2.*sigma**2)))*1.05]))
                elif itr1<itr2:
                    ax.add_artist(es[itr3])
                es[itr3].set_clip_box(ax.bbox)
                es[itr3].set_alpha(opacity_set[itr3])
                es[itr3].set_edgecolor(color_set[itr3])
                es[itr3].set_facecolor(color_set[itr3])
                es[itr3].set_fill(False)

            if itr1<itr2:
                xbox_width = xbox_widths[itr1]
                ybox_width = ybox_widths[itr2]
            elif itr1==itr2:
                xbox_width = xbox_widths[itr1]
            nticks = 2
            tickrange = 0.7
            formatter = ticker.FormatStrFormatter("%.1e")
            if itr1<=itr2:
                xtickspacing = xbox_width*tickrange/nticks
                xticks = np.arange(-tickrange/2*xbox_width,tickrange/2*xbox_width+0.01*xtickspacing,xtickspacing)
                ax.xaxis.set_major_locator(ticker.FixedLocator(xticks))
                ax.xaxis.set_major_formatter(formatter)
                ax.set_xlim(-xbox_width/2.,xbox_width/2.)


            if itr1<=itr2:
                if itr1==itr2:
                    ytickspacing = (ybox_width)*tickrange/nticks
                    yticks = np.arange(ybox_width/2.-tickrange/2*ybox_width,ybox_width/2.+tickrange/2*ybox_width+0.01*ytickspacing,ytickspacing)
                else:
                    ytickspacing = (ybox_width)*tickrange/nticks
                    yticks = np.arange(-tickrange/2*ybox_width,tickrange/2*ybox_width+0.01*ytickspacing,ytickspacing)
                ax.yaxis.set_major_locator(ticker.FixedLocator(yticks))
                ax.yaxis.set_major_formatter(formatter)
                if itr1==itr2:
                    ax.set_ylim(0.,ybox_width)
                else:
                    ax.set_ylim(-ybox_width/2.,ybox_width/2.)
                ax.tick_params(axis='both',labelsize='small',labelbottom='off',labelleft='off',labeltop='off',labelright='off')

            ax.grid()

            if itr1==itr2==0:
                param2_pretty = FORMATTED_LABELS.get(param2)
                if param2_pretty is None:
                    param2_pretty = param2
                ax.set_ylabel("$\\Delta$"+str(param2_pretty),fontsize=8)
                ax.legend(handles=es.tolist(),loc=2,prop={'size':6})
            if itr1==0 and itr1<itr2:
                param2_pretty = FORMATTED_LABELS.get(param2)
                if param2_pretty is None:
                    param2_pretty = param2
                ax.set_ylabel("$\\Delta$"+str(param2_pretty),fontsize=8)
                ax.tick_params(axis='y',labelsize=6,labelleft='on',pad=0.2,bottom='on',left='on',right='on',top='on',direction='inout')
            if itr2==n_p-1:
                param1_pretty = FORMATTED_LABELS.get(param1)
                if param1_pretty is None:
                    param1_pretty = param1
                ax.set_xlabel("$\\Delta$"+str(param1_pretty),fontsize=8)
                ax.tick_params(axis='x',labelsize=6,labelbottom='on',pad=0.2,bottom='on',left='on',right='on',top='on',direction='inout')
            if itr1==itr2:
                if itr2==n_p-1:
                    ax.tick_params(axis='both',labelsize=6,labelright='off',labeltop='off',pad=0.2,bottom='on',left='on',right='on',top='on',direction='inout')
                elif itr2==0:
                    ax.tick_params(axis='both',labelsize=6,labelright='off',labeltop='off',labelleft='on',pad=0.2,bottom='on',left='on',right='off',top='on',direction='inout')
                else:
                    ax.tick_params(axis='both',labelsize=6,labelright='off',labeltop='off',labelbottom='on',pad=0.2,bottom='on',left='on',right='off',top='on',direction='inout')
    plt.subplots_adjust(wspace=0.,hspace=0.,left=0.1,right=0.9,top=0.9,bottom=0.1)
    plt.show()
    #TODO add correlation matrix functionality
