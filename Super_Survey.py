import numpy as np
#from talk_to_class import class_objects 
from FASTPTcode import FASTPT 
from polygon_pixel_geo import polygon_pixel_geo
from polygon_geo import polygon_geo

from cosmopie import CosmoPie 
#import sph_basis as basis
from sph_klim import sph_basis_k
from time import time 
from Dn import DNumberDensityObservable
import Dn
import shear_power as sh_pow
from hmf import ST_hmf

import sys
from geo import rect_geo
from algebra_utils import ch_inv ,get_inv_cholesky
import defaults
import fisher_matrix as fm
import lensing_observables as lo
import copy
from sw_survey import SWSurvey
from lw_survey import LWSurvey
import planck_fisher
from warnings import warn
class super_survey:
    ''' This class holds and returns information for all surveys
    ''' 
    def __init__(self, surveys_sw, surveys_lw, r_max, l, n_zeros,k,basis,C,P_lin=None,get_a=False,do_mitigated=True,do_unmitigated=True):
    
      
        '''
        1) surveys is an array containing the informaiton for all the surveys.
        2) r_max is the maximum radial super mode 
        3) l angular super modes
        4) cosmology is the comological parameters etc. 
        ''' 
                         
        t1=time()

        self.get_a = get_a
        self.do_mitigated = do_mitigated
        self.do_unmitigated = do_unmitigated

        self.surveys_lw = surveys_lw
        self.surveys_sw = surveys_sw
        self.N_surveys_sw=surveys_sw.size
        self.N_surveys_lw=surveys_lw.size
        #self.geo1_lw=surveys_lw[0]['geo1']     
        #self.geo2_lw=surveys_lw[0]['geo2']     
        #self.zs_lw=surveys_lw[0]['zs']
        print'Super_Survey: this is the number of surveys', self.N_surveys_sw, self.N_surveys_lw
           
        self.C=C
         
          #k_cut = 0.25 #converge to 0.0004
        self.basis = basis 
        self.N_O_I=0
        self.N_O_a=0   
        self.O_I=np.array([], dtype=object)
    
        #self.O_a=np.array([], dtype=object)
        for i in range(self.N_surveys_sw): 
            survey=surveys_sw[i]
            #TODO why add N_O_I from different surveys?
            self.N_O_I=self.N_O_I + survey.get_N_O_I()          
        #TODO temp     
        for i in range(self.N_surveys_lw):
          
            survey=self.surveys_lw[i]
            self.N_O_a=self.N_O_a + survey.get_N_O_a()
            #self.O_a=np.append(self.O_a,survey['O_a'])
              
        print('Super_Survey: there are '+str(self.N_O_I)+' short wavelength and '+str(self.N_O_a)+' long wavelength observables')
        #self.O_a_data=self.get_O_a()
        #self.O_I_data=self.get_O_I(k,P_lin)
        self.O_I_data = self.get_O_I_all()
        self.F_0 = self.basis.get_fisher()
        if self.do_mitigated:
            #self.cov_mit,self.a_mit=self.get_SSC_covar(mitigation=True)     
            self.covs_sw,self.a_vals=self.get_SSC_covar(mitigation=True)     
            #self.cov_no_mit,self.a_no_mit=self.get_SSC_covar(mitigation=False)     
            if self.get_a:
                print "Super_Survey: mitigated run gave a="+str(self.a_vals)
        elif self.do_unmitigated:
            self.covs_sw,self.a_vals=self.get_SSC_covar(mitigation=False)     
            print "Super_Survey: unmitigated run gave a="+str(self.a_no_mit)
        else:
            self.covs_sw=np.array([],dtype=object)
            self.a_vals = np.array([],dtype=object)

        self.covs_params = self.surveys_sw[0].get_cov_tot_parameters(self.covs_sw[0])
        #if self.do_mitigated:
         #   self.c_mit_params = self.surveys_sw[0].get_cov_tot_parameters(self.cov_mit[0])
            #self.c_no_mit_params = self.surveys_sw[0].get_cov_tot_parameters(self.cov_no_mit[0])
        #elif self.do_unmitigated:
            #self.c_no_mit_params = self.surveys_sw[0].get_cov_tot_parameters(self.cov_no_mit[0])
            #self.f_no_mit = fm.fisher_matrix(self.cov_no_mit[0].get_total_covar(),input_type=fm.REP_COVAR,fix_input=False)
            #self.f_no_mit_params = self.surveys_sw[0].get_fisher_parameters(self.f_no_mit) 
            #self.f_mit = fm.fisher_matrix(self.cov_mit[0].get_total_covar(),input_type=fm.REP_COVAR,fix_input=False)
            #self.f_mit_params = self.surveys_sw[0].get_fisher_parameters(self.f_mit) 
        #TODO iterate
        self.covs_g_params = self.surveys_sw[0].get_cov_tot_parameters(self.covs_sw[0],gaussian_only=True)
        #self.f_gaussian = fm.fisher_matrix(self.cov_no_mit[0].get_gaussian_covar(),input_type=fm.REP_COVAR,fix_input=True)
        #self.f_gaussian_params = self.surveys_sw[0].get_fisher_parameters(self.f_gaussian)

        t2=time()
        print 'Super_Survey: all done'
        print 'Super_Survey: run time', t2-t1
     
     
    def get_SSC_covar(self,mitigation=True):
         
        result=np.zeros_like(2,dtype=object)
        #TODO handle F_loc differently
        if mitigation:
            print "Super_Survey: getting SSC covar with mitigation"
            F_loc = copy.deepcopy(self.F_0)
            #F_loc = self.basis.get_fisher(allow_caching=True)
            for i in range(0,self.N_surveys_lw):
                self.surveys_lw[i].fisher_accumulate(F_loc)
            F_loc.switch_rep(fm.REP_CHOL)
        else:
            F_loc = self.F_0
            print "Super_Survey: getting SSC covar without mitigation"

        Cov_SSC = np.zeros(self.N_surveys_sw,dtype=object) 
        F_SSC = np.zeros((2,self.N_surveys_sw),dtype=object) 
        a_SSC = np.zeros((2,self.N_surveys_sw),dtype=object) 

        for i in range(0,self.N_surveys_sw):
            survey=self.surveys_sw[i]
            if mitigation:
                Cov_SSC[i] = survey.get_covars(np.array([self.F_0,F_loc]),self.basis)
            else:
                Cov_SSC[i] = survey.get_covars(np.array([self.F_0]),self.basis)
            
            if self.get_a:
                T=self.basis.D_delta_bar_D_delta_alpha(survey.geo,tomography=True)[0]
                a_SSC[i]=F_loc.contract_covar(T.T,T,identical_inputs=True)
                print "Super_Survey: a is "+str(a_SSC[i])+" for survey #"+str(i)
        self.F_fin = F_loc
        return Cov_SSC,a_SSC

    def get_O_a(self):
        D_O_a=np.zeros(self.N_O_a,dtype=object)
        print 'Super_Survey: I have this many long wavelength observables', self.N_O_a 
        for i in range(self.N_O_a):
            O_a=self.O_a[i]
            for key in O_a:
                  
                if key=='number density':
                    data=O_a['number density']
                    n_obs=np.array([data[0],data[1]])
                    mass=data[2]
                    print n_obs, mass
                    X=DNumberDensityObservable(self.surveys_lw[0],defaults.dn_params,'lw1',self.C,self.basis,defaults.nz_params)
                    D_O_a[i] = X.Fisher_alpha_beta()
                    print "Super_Survey: min D_O_a element for lw observable #"+str(i)+": "+str(np.min(D_O_a[i][0]))
                    print "Super_Survey: D_O_a for lw observable #"+str(i)+": ",D_O_a[i][0]
        return D_O_a  

    def get_O_I_all(self):
        O_I_list = np.zeros(self.N_O_I,dtype=object)
        N_O_I = 0
        for i in range(0,self.N_surveys_sw):
            survey = self.surveys_sw[i]
            N_O_I_survey = survey.get_N_O_I()
            O_I_list[N_O_I:N_O_I+N_O_I_survey] = survey.get_O_I_list()
        return O_I_list
def get_ellipse_specs(covs,dchi2=2.3):
    a_s = np.zeros_like(covs)
    b_s = np.zeros_like(covs)
    #dchi2 = 2.3
    alpha =np.sqrt(dchi2)
    angles = np.zeros_like(covs)
    areas = np.zeros_like(covs)
    for i in range(0,covs.shape[0]):
        for j in range(0,covs.shape[1]):
            #cf arxiv:0906.4123 for explanation
            a_s[i,j] = np.sqrt((covs[i,i]+covs[j,j])/2.+np.sqrt((covs[i,i]-covs[j,j])**2/4.+covs[i,j]**2))
            b_s[i,j] = np.sqrt((covs[i,i]+covs[j,j])/2.-np.sqrt((covs[i,i]-covs[j,j])**2/4.+covs[i,j]**2))
            angles[i,j] = np.arctan2(2.*covs[i,j],covs[i,i]-covs[j,j])/2.
    width1s = a_s*alpha
    width2s = b_s*alpha
    areas = np.pi*width1s*width2s
    return width1s,width2s,angles,areas

def make_ellipse_plot(cov_set,color_set,opacity_set,label_set,box_widths,cosmo_param_list,C,dchi2,adaptive_mult=1.05):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.patches import Ellipse
    n_p = cosmo_param_list.size
    fig,ax_list = plt.subplots(n_p,n_p)
    #width1s_g,width2s_g,angles_g,areas_g = get_ellipse_specs(SS.c_g_params,dchi2=dchi2)
    #width1s_no_mit,width2s_no_mit,angles_no_mit,areas_no_mit = get_ellipse_specs(SS.c_no_mit_params,dchi2=dchi2)
    #width1s_mit,width2s_mit,angles_mit,areas_mit = get_ellipse_specs(SS.c_mit_params,dchi2=dchi2)
    n_c = cov_set.shape[0]
    n_p = cosmo_param_list.shape[0]
    width1_set = np.zeros((n_c,n_p,n_p))
    width2_set = np.zeros((n_c,n_p,n_p))
    angle_set = np.zeros((n_c,n_p,n_p))
    area_set = np.zeros((n_c,n_p,n_p))
    for itr3 in range(0,n_c):
        width1_set[itr3],width2_set[itr3],angle_set[itr3],area_set[itr3] = get_ellipse_specs(cov_set[itr3],dchi2=dchi2)
    if box_widths is "adaptive":
        box_widths = np.zeros(n_p)
        for itr3 in range(0,n_c):
            box_widths = np.max(np.array([box_widths,2.*np.sqrt(np.diag(cov_set[itr3]))]),axis=0)
        box_widths*=adaptive_mult

    xbox_widths = box_widths
    ybox_widths = box_widths
    for itr1 in range(0,n_p):
        for itr2 in range(0,n_p): 
            ax = ax_list[itr2,itr1]
            param1 = cosmo_param_list[itr1] 
            param2 = cosmo_param_list[itr2] 
            fid_point = np.array([C.cosmology[param1],C.cosmology[param2]])

            #TODO check sense of rotation
            es = np.zeros(n_c,dtype=object)
            for itr3  in range(0,n_c):
                es[itr3] = Ellipse(fid_point,width1_set[itr3][itr1,itr2],width2_set[itr3][itr1,itr2],angle=180./np.pi*angle_set[itr3][itr1,itr2],label=label_set[itr3])
                ax.add_artist(es[itr3])
                es[itr3].set_clip_box(ax.bbox)
                es[itr3].set_alpha(opacity_set[itr3])
                es[itr3].set_edgecolor(color_set[itr3])
                es[itr3].set_facecolor(color_set[itr3])
                es[itr3].set_fill(False)

            xbox_width = xbox_widths[itr1]
            ybox_width = ybox_widths[itr2]
            nticks = 3
            tickrange =0.7
            xtickspacing = xbox_width*tickrange/nticks
            ytickspacing = ybox_width*tickrange/nticks
            xticks = np.arange(fid_point[0]-tickrange/2*xbox_width,fid_point[0]+tickrange/2*xbox_width+0.01*xtickspacing,xtickspacing)
            yticks = np.arange(fid_point[1]-tickrange/2*ybox_width,fid_point[1]+tickrange/2*ybox_width+0.01*ytickspacing,ytickspacing)
            #ax.set_title('1-sigma contours')
            ax.xaxis.set_major_locator(ticker.FixedLocator(xticks))
            ax.yaxis.set_major_locator(ticker.FixedLocator(yticks))
            formatter=ticker.FormatStrFormatter("%.4f")
            ax.xaxis.set_major_formatter(formatter)
            ax.yaxis.set_major_formatter(formatter)
            ax.set_xlim(fid_point[0]-xbox_width/2.,fid_point[0]+xbox_width/2.)
            ax.set_ylim(fid_point[1]-ybox_width/2.,fid_point[1]+ybox_width/2.)
            ax.tick_params(axis='both',labelsize='small',labelbottom='off',labelleft='off',labeltop='off',labelright='off')
            #ax.ticklabel_format(useOffset=False)

            ax.grid()
            if itr1==itr2==0:
                ax.legend(handles=[es[0],es[1],es[2]],loc=2,prop={'size':6})
            if itr1==0:
                ax.set_ylabel(param2,fontsize=8)
                ax.tick_params(axis='y',labelsize=8,labelleft='on')
            if itr1==n_p-1:
                ax.tick_params(axis='y',labelsize=8,labelright='on')
            if itr2==0:
                ax.tick_params(axis='x',labelsize=8,labeltop='on')
             #   ax.ticklabel_format(axis='x',useOffset=True)
            if itr2==n_p-1:
                ax.set_xlabel(param1,fontsize=8)
                ax.tick_params(axis='x',labelsize=8,labelbottom='on')
             #   ax.ticklabel_format(axis='x',useOffset=True)

    plt.subplots_adjust(wspace=0.,hspace=0.)
    plt.show()
      
if __name__=="__main__":
    t1 = time()
    z_max=1.05; l_max=50 

    #d=np.loadtxt('Pk_Planck15.dat')
    #d=np.loadtxt('camb_m_pow_l.dat')
    #k=d[:,0]; P=d[:,1]
    #TODO check possible h discrepancy
    camb_params = defaults.camb_params.copy()
    camb_params['kmax'] = 10.
    camb_params['npoints'] = 1000
    cosmo_fid = defaults.cosmology_jdem.copy()
    C=CosmoPie(cosmology=cosmo_fid,p_space='jdem',needs_power=True,camb_params=camb_params)
    #C=CosmoPie(cosmology=defaults.cosmology,p_space='basic',needs_power=True)
    k,P=C.get_P_lin()
    r_max=C.D_comov(z_max)
    print 'this is r max and l_max', r_max , l_max

    #theta0=0.
    #theta1=np.pi/2.
    theta0=np.pi/4.
    theta1=np.pi/2.
    phi0=0.
    #phi1=5.*np.pi**2/162. #gives exactly a 1000 square degree field of view
    phi1 = 5*np.pi**2/(np.sqrt(2)*81.)
    phi2=np.pi/3.
    phi3=phi2+(phi1-phi0)
    theta1s = np.array([theta0,theta1,theta1,theta0,theta0])
    phi1s = np.array([phi0,phi0,phi1,phi1,phi0])
    theta_in1 = 3.*np.pi/8.
    #phi_in1 = 4.*np.pi**2/162.
    phi_in1 = 5*np.pi**2/(2.*np.sqrt(2)*162.)
    theta2s = np.array([theta0,theta1,theta1,theta0,theta0])
    phi2s = np.array([phi2,phi2,phi3,phi3,phi2])
    theta_in2 = 3.*np.pi/8.
    phi_in2 = phi2+(phi_in1-phi0)
    #theta0=np.pi/16.
    #theta1=np.pi/2.
    #phi0=0.
    #phi1=np.pi/6.
    #phi2=2.*np.pi/3.
    #phi3=5.*np.pi/6.

    #theta1s = np.array([theta0,theta1,theta1,theta0,theta0])
    #phi1s = np.array([phi0,phi0,phi1,phi1,phi0])
    #theta_in1 = np.pi/8.
    #phi_in1 = np.pi/12.
    #theta2s = np.array([theta0,theta1,theta1,theta0,theta0])
    #phi2s = np.array([phi2,phi2,phi3,phi3,phi2])
    #theta_in2 = np.pi/8.
    #phi_in2 = np.pi/12+2.*np.pi/3.
    res_choose = 6



    Theta1=[theta0,theta1]
    Phi1=[phi0,phi1]
    Theta2=[theta0,theta1]
    Phi2=[phi2,phi3]

    #zs=np.array([.4,0.8,1.2])
    zs=np.array([.6,0.8,1.01])
    z_fine = np.arange(defaults.lensing_params['z_min_integral'],np.max(zs),defaults.lensing_params['z_resolution'])
    #zbins=np.array([.2,.6,1.0])
    #l=np.logspace(np.log10(2),np.log10(3000),1000)
    l_sw = np.logspace(np.log(30),np.log(5000),base=np.exp(1.),num=40)
    #l_sw = np.arange(0,50)
    use_poly=True
    use_poly2=False
    if use_poly:
        if use_poly2:
            geo1 = polygon_geo(zs,theta1s,phi1s,C,z_fine,l_max=l_max,poly_params=defaults.polygon_params)
            geo2 = polygon_geo(zs,theta2s,phi2s,C,z_fine,l_max=l_max,poly_params=defaults.polygon_params)
        else:
            geo1 = polygon_pixel_geo(zs,theta1s,phi1s,theta_in1,phi_in1,C,z_fine,l_max=l_max,res_healpix=res_choose)  
            geo2 = polygon_pixel_geo(zs,theta2s,phi2s,theta_in2,phi_in2,C,z_fine,l_max=l_max,res_healpix=res_choose)  
    else:
        geo1=rect_geo(zs,Theta1,Phi1,C,z_fine)
        geo2=rect_geo(zs,Theta2,Phi2,C,z_fine)
    
    loc_lens_params = defaults.lensing_params.copy()
    loc_lens_params['z_min_dist'] = np.min(zs)
    loc_lens_params['z_max_dist'] = np.max(zs)
    
    #TODO put in defaults
    lenless_defaults = defaults.sw_survey_params.copy()
    lenless_defaults['needs_lensing'] = False

    cosmo_param_list = np.array(['ns','Omegamh2','Omegabh2','OmegaLh2','LogAs','w'])
    cosmo_param_epsilons = np.array([0.002,0.0005,0.0001,0.0005,0.1,0.1])
    #cosmo_param_list = np.array(['LogAs','w'])
    #cosmo_param_epsilons = np.array([0.001,0.001])

    param_priors = planck_fisher.get_w0_projected(params=defaults.planck_fisher_params)
    #cosmo_param_list = np.array(['Omegamh2','Omegabh2'])
    #cosmo_param_epsilons = np.array([0.001,0.001])
    #cosmo_param_list = np.array(['Omegamh2','Omegabh2','ns','h','sigma8'])
    #note that currently (poorly implemented derivative) in jdem, OmegaLh2 and LogAs are both almost completely unconstrained but nondegenerate, while in basic, h and sigma8 are not constrained but are almost completely degenerate
    survey_1 = SWSurvey(geo1,'survey1',C=C,ls=l_sw,params=defaults.sw_survey_params,observable_list = defaults.sw_observable_list,cosmo_param_list = cosmo_param_list,cosmo_param_epsilons=cosmo_param_epsilons,len_params=loc_lens_params,param_priors=param_priors) 
    #survey_2 = SWSurvey(geo1,'survey2',C=C,ls=l_sw,params=defaults.sw_survey_params,observable_list = defaults.sw_observable_list,len_params=loc_lens_params) 
 
    #survey_1 = SWSurvey(geo1,'survey1',C=C,ls=l_sw,params=defaults.sw_survey_params,observable_list = np.array([]),len_params=loc_lens_params) 
    survey_2 = SWSurvey(geo1,'survey2',C=C,ls=l_sw,params=lenless_defaults,observable_list = np.array([]),cosmo_param_list = np.array([],dtype=object),cosmo_param_epsilons=np.array([]),len_params=loc_lens_params,param_priors=param_priors) 
     
     

    M_cut=10**(12.5)
     
     
    surveys_sw=np.array([survey_1])
    
     
    geos = np.array([geo1,geo2])
    #geos = np.array([geo1])
    l_lw=np.arange(0,30)
    n_zeros=49
    k_cut = 0.005
            
    basis=sph_basis_k(r_max,C,k_cut,l_ceil=100)

    survey_3 = LWSurvey(geos,'lw_survey1',basis,C=C,ls = l_lw,params=defaults.lw_survey_params,observable_list=defaults.lw_observable_list,dn_params=defaults.dn_params)
    surveys_lw=np.array([survey_3])
     
     
    print 'main: this is r_max: '+str(r_max)
     
    SS=super_survey(surveys_sw, surveys_lw,r_max,l_sw,n_zeros,k,basis,P_lin=P,C=C,get_a=False,do_unmitigated=True,do_mitigated=True)

    t2 = time()
    print "main: total run time "+str(t2-t1)+" s"
     
    #print "fractional mitigation: ", SS.a_no_mit/SS.a_mit     
    rel_weights1 = SS.basis.D_delta_bar_D_delta_alpha(SS.surveys_sw[0].geo,tomography=True)[0]*np.dot(SS.F_0.get_cov_cholesky(),SS.basis.D_delta_bar_D_delta_alpha(SS.surveys_sw[0].geo,tomography=True)[0])
    rel_weights2 = SS.basis.D_delta_bar_D_delta_alpha(SS.surveys_sw[0].geo,tomography=True)[0]*np.dot(SS.F_fin.get_cov_cholesky(),SS.basis.D_delta_bar_D_delta_alpha(SS.surveys_sw[0].geo,tomography=True)[0])
    #np.savetxt('rel_weights_k_026.txt',rel_weights)

 #   ax.plot(rel_weights)
 #   plt.show()
    
#     cov_ss = SS.cov_no_mit[0].get_gaussian_covar()#np.diag(SS.O_I_data[0]['shear_shear']['covariance'][0,0])
#    c_ss = SS.O_I_data[0]
   #chol_cov = get_inv_cholesky(cov_ss)
    #mat_retrieved = (np.identity(chol_cov.shape[0])+np.dot(np.dot(chol_cov,SS.cov_no_mit[0,0]),chol_cov.T))
    #eig_ret = np.linalg.eigvals(mat_retrieved)
    #SS_eig =  SS.cov_no_mit[0].get_SS_eig()[0]
    #SS_eig_mit =  SS.cov_mit[0].get_SS_eig()[0]
    try:
        SS_eigs = SS.covs_sw[0].get_SS_eig()
        chol_gauss = np.linalg.cholesky(SS.covs_sw[0].get_gaussian_covar())
        print "main: unmitigated lambda1,2: "+str(SS_eigs[0][0][-1])+","+str(SS_eigs[0][0][-2])
        print "main: mitigated lambda1,2: "+str(SS_eigs[1][0][-1])+","+str(SS_eigs[1][0][-2])
        print "n lambda>1.00000001: "+str(np.sum(np.abs(SS_eigs[0][0][0])>1.00000001))
    except:
        warn('Eig value failed')
#    print "SS eigenvals:",SS_eig[0]
#    print "(S/N)^2 gaussian: ",np.dot(np.dot(c_ss,np.linalg.inv(cov_ss)),c_ss)
    #print "(S/N)^2 gaussian+no mitigation: ",np.dot(np.dot(c_ss,np.linalg.inv(np.diagflat(cov_ss)+SS.cov_no_mit[0,0])),c_ss)
    #print "(S/N)^2 gaussian+mitigation: ",np.dot(np.dot(c_ss,np.linalg.inv(np.diagflat(cov_ss)+SS.cov_mit[0,0])),c_ss)
    #ax.loglog(l_sw,cov_ss)
    #ax.loglog(l,np.diag(SS.cov_mit[0,0])/c_ss**2*l/2)
   # ax.loglog(l,(np.diag(SS.cov_mit[0,0])))
    #ax.loglog(l_sw,(np.diag(SS.cov_no_mit[0,0])/cov_ss))
    #ax.legend(['cov','mit','no mit'])
    #plt.show()
    print 'r diffs',np.diff(geo1.rs)
    print 'theta width',(geo1.rs[1]+geo1.rs[0])/2.*(Theta1[1]-Theta1[0])
    print 'phi width',(geo1.rs[1]+geo1.rs[0])/2.*(Phi1[1]-Phi1[0])*np.sin((Theta1[1]+Theta1[0])/2)
    ax_ls = np.hstack((l_sw,l_sw))
    
    v= SS.surveys_sw[0].get_dO_I_dparameter_array()
    eig_params = SS.covs_sw[0].get_SS_eig_param(v)
    #eig_nm = SS.cov_no_mit[0].get_SS_eig_param(v)
    #eig_m = SS.cov_mit[0].get_SS_eig_param(v)

    ellipse_plot=True
    if ellipse_plot:
        no_mit_color = np.array([1.,0.,0.])
        mit_color = np.array([0.,1.,0.])
        g_color = np.array([0.,0.,1.])
        color_set = np.array([mit_color,no_mit_color,g_color])
        opacity_set = np.array([1.0,1.0,1.0])
        box_widths = np.array([0.015,0.005,0.0005,0.005,0.1,0.05])
        dchi2 = 2.3
        cov_set = np.array([SS.covs_params[1],SS.covs_params[0],SS.covs_g_params[0]])
        label_set = np.array(["ssc+mit+g","ssc+g","g"])
        make_ellipse_plot(cov_set,color_set,opacity_set,label_set,'adaptive',cosmo_param_list,C,dchi2=dchi2)


    chol_plot=False
    if chol_plot:
        ax = plt.subplot(111)
        for itr in range(1,5):
            #ax.plot(ax_ls,ax_ls*(ax_ls+1.)*np.dot(chol_gauss,SS_eig[1][:,-itr]))
            ax.plot(np.dot(chol_gauss,SS_eig[1][:,-itr]))
    
        ax.legend(['1','2','3','4','5'])
        plt.show()
    #TODO make testing module for this
    test_perturbation=False
    pert_test_fails = 0
    if test_perturbation:
        #TOLERANCE below which an eigenvalue less than TOLERANCE*max eigenvalue is considered 0
        REL_TOLERANCE = 10**-8
        f0 = SS.F_0.get_F_alpha_beta()
        f1 = SS.F_fin.get_F_alpha_beta()
        if not np.all(f0.T==f0):
            pert_test_fails+=1
            warn("unperturbed fisher matrix not symmetric, unacceptable")
        if not np.all(f1.T==f1):
            pert_test_fails+=1
            warn("perturbed fisher matrix not symmetric, unacceptable")
        #get eigenvalues and set numerically zero values to 0 
        eigf0 = np.linalg.eigh(f0)[0]
        eigf0[np.abs(eigf0)<REL_TOLERANCE*np.max(np.abs(eigf0))]=0.
        eigf1 = np.linalg.eigh(f1)[0]
        eigf1[np.abs(eigf1)<REL_TOLERANCE*np.max(np.abs(eigf1))]=0.
        #check positive semidefinite
        if np.any(eigf0<0.):
            pert_test_fails+=1
            warn("unperturbed fisher matrix not positive definite within tolerance, unacceptable")
        if np.any(eigf1<0.):
            pert_test_fails+=1
            warn("perturbed fisher matrix not positive definite within tolerance, unacceptable")

        #check nondecresasing
        diff_eig = eigf1-eigf0
        diff_eig[np.abs(diff_eig)<REL_TOLERANCE*np.max(np.abs(diff_eig))] = 0
        if np.any(diff_eig<0):
            pert_test_fails+=1
            warn("some eigenvalues decreased within tolerance, unacceptable")
        
        #check interlace theorem satisfied (eigenvalues cannot be reordered by more than rank of perturbation)
        n_offset = SS.surveys_lw[0].get_total_rank()
        rolled_eig = (eigf1[::-1][n_offset:eigf0.size]-eigf0[::-1][0:eigf0.size-n_offset])
        rolled_eig[np.abs(rolled_eig)<REL_TOLERANCE*np.max(np.abs(rolled_eig))]=0.  
        if np.any(rolled_eig>0):
            pert_test_fails+=1
            warn("some eigenvalues fail interlace theorem, unacceptable")


        c0 = SS.F_0.get_covar()
        c1 = SS.F_fin.get_covar()
        if not np.allclose(c0,c0.T):
            pert_test_fails+=1
            warn("unperturbed covariance not symmetric, unacceptable")
        #TODO investigate possible deviation from exactness here
        if not np.allclose(c1,c1.T):
            warn("perturbed covariance not symmetric, unacceptable")
        eigc0 = np.linalg.eigh(c0)[0]
        eigc1 = np.linalg.eigh(c1)[0]
        if np.any(eigc0<0):
            pert_test_fails+=1
            warn("unperturbed covariance not positive semidefinite, unacceptable")
        if np.any(eigc1<0):
            pert_test_fails+=1
            warn("perturbed covariance not positive semidefinite, unacceptable")
        fdiff_eigc = (eigc1-eigc0)/eigc0
        fdiff_eigc[np.abs(fdiff_eigc)<REL_TOLERANCE] = 0.
        if np.any(fdiff_eigc>0):
            pert_test_fails+=1
            warn("some covariance eigenvalues increase, unacceptable")
        
        if pert_test_fails==0:
            print "All fisher matrix sanity checks passed"
        else:
            warn(str(pert_test_fails)+" fisher matrix sanity checks failed")
    test_eigs = False
    eig_test_fails = 0
    if test_eigs:
        REL_TOLERANCE = 10**-8
        c_sscs = SS.covs_sw[0].get_ssc_covar()
        c_ssc0 = c_sscs[0]
        if not np.allclose(c_ssc0,c_ssc0.T):
            eig_test_fails+=1
            warn("unperturbed result covariance not symmetric, unacceptable")
        c_ssc1 = c_sscs[1]
        if not np.allclose(c_ssc1,c_ssc1.T):
            eig_test_fails+=1
            warn("perturbed result covariance not symmetric, unacceptable")
        eigsys_ssc0 = np.linalg.eigh(c_ssc0)
        eigsys_ssc1 = np.linalg.eigh(c_ssc1)
        eig_ssc0 = eigsys_ssc0[0].copy()
        eig_ssc1 = eigsys_ssc1[0].copy()
        eig_ssc0[np.abs(eig_ssc0)<np.max(np.abs(eig_ssc0))*REL_TOLERANCE]=0
        eig_ssc1[np.abs(eig_ssc0)<np.max(np.abs(eig_ssc0))*REL_TOLERANCE]=0
        if np.any(eig_ssc0<0):
            eig_test_fails+=1
            warn("unperturbed result cov not positive semidefinite, unacceptable")
        if np.any(eig_ssc1<0):
            eig_test_fails+=1
            warn("perturbed result cov not positive semidefinite, unacceptable")
        cg = SS.cov_no_mit[0].get_gaussian_covar()
        eigsys_cg = np.linalg.eigh(cg)
        eig_cg = eigsys_cg[0].copy()
        eig_mitprod = np.real(np.linalg.eig(np.dot(np.linalg.inv(c_ssc0+cg),c_ssc1+cg))[0])
        eig_mitprod[np.abs(eig_mitprod-1.)<REL_TOLERANCE] = 1.
        if np.any(eig_mitprod>1):
            eig_test_fails+=1
            warn("mitigation making covariance worse, unacceptable")
        n_offset = SS.surveys_lw[0].get_total_rank()
        if np.sum(eig_mitprod<1.)>n_offset:
            eig_test_fails+=1
            warn("mitigation changing too many eigenvalues, unacceptable")
        eig_diff = eig_ssc1-eig_ssc0
        eig_diff[np.abs(eig_diff)<np.max(np.abs(eig_diff))*REL_TOLERANCE] = 0.
        if np.any(eig_diff>0):
            eig_test_fails+=1
            warn("mitigation making covariance worse, unacceptable")

        if eig_test_fails==0:
            print "All eigenvalue sanity checks passed"
        else:
            warn(str(pert_test_fails)+" eigenvalue sanity checks failed")

