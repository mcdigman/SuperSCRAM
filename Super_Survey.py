from time import time
from warnings import warn

import numpy as np

from polygon_pixel_geo import PolygonPixelGeo
from polygon_geo import PolygonGeo
from sph_klim import SphBasisK
from geo import RectGeo
from sw_survey import SWSurvey
from lw_survey import LWSurvey

import defaults
import fisher_matrix as fm
import prior_fisher
import matter_power_spectrum as mps
import multi_fisher as mf
import cosmopie as cp
#TODO maybe make possible to serialize/pickle or at least provide a standardized way of storing run information
#TODO examine possibile platform dependence bug with ruby
class SuperSurvey:
    ''' This class holds and returns information for all surveys
    '''
    def __init__(self, surveys_sw, surveys_lw, basis,C,get_a=False,do_mitigated=True,do_unmitigated=True):
        """master class for mitigation analysis
            inputs:
                surveys_sw: an array of SWSurveys
                surveys_lw: an array of LWSurveys (different mitigations)
                basis: an LWBasis object
                C: a CosmoPie object
                get_a: get (v.T).C_lw.v where v=\frac{\partial\bar{\delta}}{\delta_\alpha}
                do_mitigated: if False don't get mitigated covariances
                do_unmitigate: if False don't get unmitigated covariances"""
        t1=time()

        self.get_a = get_a
        self.do_mitigated = do_mitigated
        self.do_unmitigated = do_unmitigated

        self.surveys_lw = surveys_lw
        self.surveys_sw = surveys_sw
        self.N_surveys_sw=surveys_sw.size
        self.N_surveys_lw=surveys_lw.size
        print'SuperSurvey: this is the number of surveys', self.N_surveys_sw, self.N_surveys_lw

        self.C=C

          #k_cut = 0.25 #converge to 0.0004
        self.basis = basis
        self.N_O_I=0
        self.N_O_a=0

        #self.O_a=np.array([], dtype=object)
        for i in xrange(self.N_surveys_sw):
            self.N_O_I=self.N_O_I + surveys_sw[i].get_N_O_I()

        for i in xrange(self.N_surveys_lw):
            self.N_O_a=self.N_O_a + surveys_lw[i].get_N_O_a()

        print('SuperSurvey: there are '+str(self.N_O_I)+' short wavelength and '+str(self.N_O_a)+' long wavelength observables')

        #TODO support multiple sw surveys with MultiFisher
        de_prior_params = defaults.prior_fisher_params
        self.multi_f = mf.MultiFisher(basis,surveys_sw[0],surveys_lw,prior_params=de_prior_params,needs_a=self.get_a,do_mit=self.do_mitigated)

#        self.f_set_nopriors = self.multi_f.get_fisher_set(include_priors=False)
#        self.f_set = self.multi_f.get_fisher_set(include_priors=True)
#        self.eig_set = self.multi_f.get_eig_set(self.f_set_nopriors)
#        self.eig_set_ssc = self.multi_f.get_eig_set(self.f_set_nopriors,ssc_metric=True)
        #TODO get this before changing shape in multi_fisher
        if self.get_a:
            self.a_vals = self.multi_f.get_a_lw()
            print "SuperSurvey: mitigated run gave a="+str(self.a_vals)
        else:
            self.a_vals = np.zeros(2)

        t2=time()
        print 'SuperSurvey: all done'
        print 'SuperSurvey: run time', t2-t1


def get_ellipse_specs(covs,dchi2=2.3):
    """Get the widths and angles for plotting covariance ellipses from a covariance matrix covs, with width dchi2"""
    a_s = np.zeros_like(covs)
    b_s = np.zeros_like(covs)
    #dchi2 = 2.3
    alpha =np.sqrt(dchi2)
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

def make_ellipse_plot(cov_set,color_set,opacity_set,label_set,box_widths,cosmo_par_list,C,dchi2,adaptive_mult=1.05):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.patches import Ellipse
    n_p = cosmo_par_list.size
    fig,ax_list = plt.subplots(n_p,n_p)
    #width1s_g,width2s_g,angles_g,areas_g = get_ellipse_specs(SS.c_g_params,dchi2=dchi2)
    #width1s_no_mit,width2s_no_mit,angles_no_mit,areas_no_mit = get_ellipse_specs(SS.c_no_mit_params,dchi2=dchi2)
    #width1s_mit,width2s_mit,angles_mit,areas_mit = get_ellipse_specs(SS.c_mit_params,dchi2=dchi2)
    n_c = cov_set.shape[0]
    n_p = cosmo_par_list.shape[0]
    width1_set = np.zeros((n_c,n_p,n_p))
    width2_set = np.zeros((n_c,n_p,n_p))
    angle_set = np.zeros((n_c,n_p,n_p))
    area_set = np.zeros((n_c,n_p,n_p))
    for itr3 in xrange(0,n_c):
        width1_set[itr3],width2_set[itr3],angle_set[itr3],area_set[itr3] = get_ellipse_specs(cov_set[itr3],dchi2=dchi2)
    if box_widths=="adaptive":
        box_widths = np.zeros(n_p)
        for itr3 in xrange(0,n_c):
            box_widths = np.max(np.array([box_widths,2.*np.sqrt(np.diag(cov_set[itr3]))]),axis=0)
        box_widths*=adaptive_mult

    xbox_widths = box_widths
    ybox_widths = box_widths
    for itr1 in xrange(0,n_p):
        for itr2 in xrange(0,n_p):
            ax = ax_list[itr2,itr1]
            param1 = cosmo_par_list[itr1]
            param2 = cosmo_par_list[itr2]
            fid_point = np.array([C.cosmology[param1],C.cosmology[param2]])

            #TODO check sense of rotation
            es = np.zeros(n_c,dtype=object)
            for itr3  in xrange(0,n_c):
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
     #TODO add correlation matrix functionality
if __name__=="__main__":
    t1 = time()
    #TODO make sure input z cannot exceed z_max in geo
    z_max=1.35
    l_max=50

    #d=np.loadtxt('Pk_Planck15.dat')
    #d=np.loadtxt('camb_m_pow_l.dat')
    #k=d[:,0]; P=d[:,1]
    #TODO check possible h discrepancy
    camb_params = defaults.camb_params.copy()
    camb_params['force_sigma8']=False
    camb_params['kmax'] = 10.
    camb_params['npoints'] = 1000
    cosmo_fid = defaults.cosmology_jdem.copy()
    cosmo_fid['w']=-1.
    cosmo_fid['w0'] = cosmo_fid['w']
    cosmo_fid['wa'] = 0.
    cosmo_fid['de_model'] = 'constant_w'
    if cosmo_fid['de_model'] == 'jdem':
        for i in xrange(0,36):
            cosmo_fid['ws36_'+str(i)] = cosmo_fid['w']

    C=cp.CosmoPie(cosmology=cosmo_fid,p_space='jdem',camb_params=camb_params)
    #C=cp.CosmoPie(cosmology=defaults.cosmology,p_space='basic',needs_power=True)
    #k,P=C.get_P_lin()
    P=mps.MatterPower(C,camb_params=camb_params)
    k=P.k
    C.P_lin=P
    C.k=k
    #TODO check comoving distance is working
    r_max=C.D_comov(z_max)
    print 'this is r max and l_max', r_max , l_max

    #theta0=0.
    #theta1=np.pi/2.
    theta0=np.pi/4.
    theta1=np.pi/2.
    phi0=0.
    #phi1=5.*np.pi**2/162. #gives exactly a 1000 square degree field of view
    phi1 = 5*np.pi**2./(np.sqrt(2.)*81.)*2.3033
    phi2=np.pi/3.
    phi3=phi2+(phi1-phi0)
    theta1s = np.array([theta0,theta1,theta1,theta0,theta0])
    phi1s = np.array([phi0,phi0,phi1,phi1,phi0])-phi1/2.
    theta_in1 = 3.*np.pi/8.
    #phi_in1 = 4.*np.pi**2/162.
    phi_in1 = 5*np.pi**2/(2.*np.sqrt(2)*162.)

    #theta2s = np.array([theta0,theta1,theta1,theta0,theta0])
    #phi2s = np.array([phi2,phi2,phi3,phi3,phi2])
    #theta_in2 = 3.*np.pi/8.
    #phi_in2 = phi2+(phi_in1-phi0)

    theta2s = np.array([theta0,3./2.*theta1,3./2.*theta1,theta0,theta0])
    phi2s = phi1s.copy()
    phi2s*=4.*0.7745282
    theta_in2=theta_in1
    phi_in2 = phi_in1

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
    res_choose = 7



    Theta1=[theta0,theta1]
    Phi1=[phi0,phi1]
    Theta2=[theta0,theta1]
    Phi2=[phi2,phi3]

    #zs=np.array([.4,0.8,1.2])
    #zs=np.array([.6,0.8,1.01])
    #zs=np.array([0.2,0.43,.63,0.9,1.178125])
    zs=np.array([0.2,0.43,.63,0.9, 1.3])
    z_fine = np.arange(defaults.lensing_params['z_min_integral'],np.max(zs),defaults.lensing_params['z_resolution'])
    #z_fine = np.linspace(defaults.lensing_params['z_min_integral'],np.max(zs),590)
    #zbins=np.array([.2,.6,1.0])
    #l=np.logspace(np.log10(2),np.log10(3000),1000)
    l_sw = np.logspace(np.log(30),np.log(5000),base=np.exp(1.),num=40)
    #l_sw = np.arange(0,50)
    use_poly=True
    use_poly2=True
    if use_poly:
        if use_poly2:
            geo1 = PolygonGeo(zs,theta1s,phi1s,theta_in1,phi_in1,C,z_fine,l_max=l_max,poly_params=defaults.polygon_params)
            geo2 = PolygonGeo(zs,theta2s,phi2s,theta_in2,phi_in2,C,z_fine,l_max=l_max,poly_params=defaults.polygon_params)
        else:
            geo1 = PolygonPixelGeo(zs,theta1s,phi1s,theta_in1,phi_in1,C,z_fine,l_max=l_max,res_healpix=res_choose)
            geo2 = PolygonPixelGeo(zs,theta2s,phi2s,theta_in2,phi_in2,C,z_fine,l_max=l_max,res_healpix=res_choose)
    else:
        geo1=RectGeo(zs,Theta1,Phi1,C,z_fine)
        geo2=RectGeo(zs,Theta2,Phi2,C,z_fine)

    loc_lens_params = defaults.lensing_params.copy()
    loc_lens_params['z_min_dist'] = np.min(zs)
    loc_lens_params['z_max_dist'] = np.max(zs)
    loc_lens_params['pmodel_O'] = 'halofit'
    loc_lens_params['pmodel_dO_ddelta'] = 'halofit'
    loc_lens_params['pmodel_dO_dpar'] = 'halofit'

    #TODO put in defaults
    lenless_defaults = defaults.sw_survey_params.copy()
    lenless_defaults['needs_lensing'] = False
    if cosmo_fid['de_model'] == 'w0wa':
        cosmo_par_list = np.array(['ns','Omegamh2','Omegabh2','OmegaLh2','LogAs','w0','wa'])
        cosmo_par_epsilons = np.array([0.002,0.0005,0.0001,0.0005,0.1,0.01,0.07])
    elif cosmo_fid['de_model'] == 'constant_w':
        cosmo_par_list = np.array(['ns','Omegamh2','Omegabh2','OmegaLh2','LogAs','w'])
        cosmo_par_epsilons = np.array([0.002,0.0005,0.0001,0.0005,0.1,0.01])
    elif cosmo_fid['de_model'] == 'jdem':
        cosmo_par_list = ['ns','Omegamh2','Omegabh2','OmegaLh2','LogAs']
        cosmo_par_list.extend(cp.JDEM_LIST)
        cosmo_par_list = np.array(cosmo_par_list,dtype=object)
        #TODO does not really seem to be converging in small epsilon, maybe interpolation or camb issue?
        cosmo_par_epsilons = np.full(41,0.5)
        cosmo_par_epsilons[0:5] = np.array([0.002,0.0005,0.0001,0.0005,0.1])

    else:
        raise ValueError('not prepared to handle '+str(cosmo_fid['de_model']))

    #cosmo_par_list = np.array(['LogAs','w'])
    #cosmo_par_epsilons = np.array([0.001,0.001])
    #TODO eliminate
    #param_priors = prior_fisher.get_w0wa_projected(params=defaults.prior_fisher_params)
    #param_priors = prior_fisher.get_w0_projected(params=defaults.prior_fisher_params)
    #cosmo_par_list = np.array(['Omegamh2','Omegabh2'])
    #cosmo_par_epsilons = np.array([0.001,0.001])
    #cosmo_par_list = np.array(['Omegamh2','Omegabh2','ns','h','sigma8'])
    #note that currently (poorly implemented derivative) in jdem, OmegaLh2 and LogAs are both almost completely unconstrained but nondegenerate, while in basic, h and sigma8 are not constrained but are almost completely degenerate
    survey_1 = SWSurvey(geo1,'survey1',C=C,ls=l_sw,params=defaults.sw_survey_params,observable_list = defaults.sw_observable_list,cosmo_par_list = cosmo_par_list,cosmo_par_epsilons=cosmo_par_epsilons,len_params=loc_lens_params)
    #survey_2 = SWSurvey(geo1,'survey2',C=C,ls=l_sw,params=defaults.sw_survey_params,observable_list = defaults.sw_observable_list,len_params=loc_lens_params)

    #survey_1 = SWSurvey(geo1,'survey1',C=C,ls=l_sw,params=defaults.sw_survey_params,observable_list = np.array([]),len_params=loc_lens_params)
    #survey_2 = SWSurvey(geo1,'survey2',C=C,ls=l_sw,params=lenless_defaults,observable_list = np.array([]),cosmo_par_list = np.array([],dtype=object),cosmo_par_epsilons=np.array([]),len_params=loc_lens_params,param_priors=param_priors)



    M_cut=10**(12.5)


    surveys_sw=np.array([survey_1])


    geos = np.array([geo1,geo2])
    #geos = np.array([geo1])
    #l_lw=np.arange(0,30)
    #n_zeros=49
    #k_cut = 0.005
    #k_cut = 0.016
    k_cut = 0.019
    #about biggest possible, take 414 sec
    #k_cut = 0.0214
    #k_cut = 0.03

    basis=SphBasisK(r_max,C,k_cut,l_ceil=100)
    survey_3 = LWSurvey(geos,'lw_survey1',basis,C=C,params=defaults.lw_survey_params,observable_list=defaults.lw_observable_list,dn_params=defaults.dn_params)
    surveys_lw=np.array([survey_3])


    print 'main: this is r_max: '+str(r_max)
    #TODO do not need basis as an argument
    SS=SuperSurvey(surveys_sw, surveys_lw,basis,C=C,get_a=True,do_unmitigated=True,do_mitigated=True)

    t2 = time()
    print "main: total run time "+str(t2-t1)+" s"

    #print "fractional mitigation: ", SS.a_no_mit/SS.a_mit
    #rel_weights1 = SS.basis.D_delta_bar_D_delta_alpha(SS.surveys_sw[0].geo,tomography=True)[0]*np.dot(SS.multi_f.get_fisher(mf.f_spec_no_mit,mf.f_return_lw)[0].get_cov_cholesky(),SS.basis.D_delta_bar_D_delta_alpha(SS.surveys_sw[0].geo,tomography=True)[0])
    #rel_weights2 = SS.basis.D_delta_bar_D_delta_alpha(SS.surveys_sw[0].geo,tomography=True)[0]*np.dot(SS.multi_f.get_fisher(mf.f_spec_mit,mf.f_return_lw)[0].get_cov_cholesky(),SS.basis.D_delta_bar_D_delta_alpha(SS.surveys_sw[0].geo,tomography=True)[0])


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
        #mit_eigs_sw = SS.f_set[2][1].get_cov_eig_metric(SS.f_set[0][1])
        #no_mit_eigs_sw = SS.f_set[1][1].get_cov_eig_metric(SS.f_set[0][1])

        #mit_eigs_par = SS.f_set[2][2].get_cov_eig_metric(SS.f_set[0][2])
        #no_mit_eigs_par = SS.f_set[1][2].get_cov_eig_metric(SS.f_set[0][2])
        mit_eigs_sw = SS.eig_set[0,1]
        no_mit_eigs_sw = SS.eig_set[0,0]
        mit_eigs_par = SS.eig_set[1,1]
        no_mit_eigs_par = SS.eig_set[1,0]
        #TODO check not including priors in eigenvalues
        #TODO check eigenvalue interlace for this projection
        print "main: unmitigated sw lambda1,2: "+str(no_mit_eigs_sw[0][-1])+","+str(no_mit_eigs_sw[0][-2])
        print "main: mitigated sw lambda1,2: "+str(mit_eigs_sw[0][-1])+","+str(mit_eigs_sw[0][-2])
        print "main: n sw mit lambda>2.00000001: "+str(np.sum(np.abs(mit_eigs_sw[0])>2.00000001))
        print "main: n sw no mit lambda>2.00000001: "+str(np.sum(np.abs(no_mit_eigs_sw[0])>2.00000001))
        print "main: unmitigated par lambda1,2: "+str(no_mit_eigs_par[0][-1])+","+str(no_mit_eigs_par[0][-2])
        print "main: mitigated par lambda1,2: "+str(mit_eigs_par[0][-1])+","+str(mit_eigs_par[0][-2])
        print "main: n par mit lambda>2.00000001: "+str(np.sum(np.abs(mit_eigs_par[0])>2.00000001))
        print "main: n par no mit lambda>2.00000001: "+str(np.sum(np.abs(no_mit_eigs_par[0])>2.00000001))
        v_no_mit_par = np.dot(SS.f_set_nopriors[0][2].get_cov_cholesky(),no_mit_eigs_par[1])
        v_mit_par = np.dot(SS.f_set_nopriors[0][2].get_cov_cholesky(),mit_eigs_par[1])

        v_no_mit_sw = np.dot(SS.f_set_nopriors[0][1].get_cov_cholesky(),no_mit_eigs_sw[1])
        v_mit_sw = np.dot(SS.f_set_nopriors[0][1].get_cov_cholesky(),mit_eigs_sw[1])

        test_v=False
        if test_v:
            m_mat_no_mit_par = np.identity(mit_eigs_par[0].size)+np.dot(SS.f_set_nopriors[1][2].get_covar(),SS.f_set_nopriors[0][2].get_fisher())
            m_mat_mit_par = np.identity(mit_eigs_par[0].size)+np.dot(SS.f_set_nopriors[2][2].get_covar(),SS.f_set_nopriors[0][2].get_fisher())

            if not np.allclose(np.dot(m_mat_no_mit_par,v_no_mit_par)/no_mit_eigs_par[0],v_no_mit_par):
                warn('some no mit eig vectors may be bad')
            if not np.allclose(np.dot(m_mat_mit_par,v_mit_par)/mit_eigs_par[0],v_mit_par):
                warn('some mit eig vectors may be bad')


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
    #TODO add back priors
    get_hold_mats = False
    if get_hold_mats:
        no_prior_hold = SS.f_set[0][2].get_fisher()
        if C.de_model == 'jdem':
            no_prior_project = prior_fisher.project_w0wa(no_prior_hold)

    print 'main: r diffs',np.diff(geo1.rs)
    print 'main: theta width',(geo1.rs[1]+geo1.rs[0])/2.*(Theta1[1]-Theta1[0])
    print 'main: phi width',(geo1.rs[1]+geo1.rs[0])/2.*(Phi1[1]-Phi1[0])*np.sin((Theta1[1]+Theta1[0])/2)
    ax_ls = np.hstack((l_sw,l_sw))

    #v= SS.surveys_sw[0].get_dO_I_dpar_array()
    #eig_nm = SS.cov_no_mit[0].get_SS_eig_param(v)
    #eig_m = SS.cov_mit[0].get_SS_eig_param(v)
    ellipse_plot_setup = True
    if ellipse_plot_setup:
        no_mit_color = np.array([1.,0.,0.])
        mit_color = np.array([0.,1.,0.])
        g_color = np.array([0.,0.,1.])
        color_set = np.array([mit_color,no_mit_color,g_color])
        opacity_set = np.array([1.0,1.0,1.0])
        box_widths = np.array([0.015,0.005,0.0005,0.005,0.1,0.05])
        dchi2 = 2.3
        #cov_set = np.array([SS.covs_params[1],SS.covs_params[0],SS.covs_g_pars[0]])
        cov_set = np.array([SS.f_set[2][2].get_covar(),SS.f_set[1][2].get_covar(),SS.f_set[0][2].get_covar()])
        label_set = np.array(["ssc+mit+g","ssc+g","g"])

    ellipse_plot=False
    if ellipse_plot:
        make_ellipse_plot(cov_set,color_set,opacity_set,label_set,'adaptive',cosmo_par_list,C,dchi2=dchi2)


#    chol_plot=False
#    if chol_plot:
#        import matplotlib.pyplot as plt
#        ax = plt.subplot(111)
#        chol_gauss = SS.f_set[0][2].get_cov_choleksy()#np.linalg.cholesky(SS.covs_sw[0].get_gaussian_covar())
#        #eig_pars = SS.covs_sw[0].get_SS_eig_param(v)
#        eig_pars = np.zeros(2,dtype=object)
#        eig_pars[0] = SS.f_set[1][2].get_cov_eig_metric(SS.covs_g_pars)
#        eig_pars[1] = SS.f_set[2][2].get_cov_eig_metric(SS.covs_g_pars)
#        for itr in xrange(1,5):
#            #ax.plot(ax_ls,ax_ls*(ax_ls+1.)*np.dot(chol_gauss,SS_eig[1][:,-itr]))
#            #ax.plot(np.dot(chol_gauss,SS_eig[1][:,-itr]))
#            #TODO might just delete this
#            ax.plot(np.dot(chol_gauss,eig_pars[:,-itr]))
#
#        ax.legend(['1','2','3','4','5'])
#        plt.show()
    #TODO make testing module for this
    test_perturbation = False
    pert_test_fails = 0
    if test_perturbation:
        #TOLERANCE below which an eigenvalue less than TOLERANCE*max eigenvalue is considered 0
        REL_TOLERANCE = 10**-8
        f0 = SS.multi_f.get_fisher(mf.f_spec_no_mit,mf.f_return_lw)[0].get_fisher()
        f1 = SS.multi_f.get_fisher(mf.f_spec_mit,mf.f_return_lw)[0].get_fisher()
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

        c0 = SS.multi_f.get_fisher(mf.f_spec_no_mit,mf.f_return_lw)[0].get_covar()
        c1 = SS.multi_f.get_fisher(mf.f_spec_mit,mf.f_return_lw)[0].get_covar()
        if not np.all(c0==c0.T):
            pert_test_fails+=1
            warn("unperturbed covariance not symmetric, unacceptable")
        if not np.all(c1==c1.T):
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
        #c_sscs = SS.multi_f.get_fisher(mf.f_spec_SSC_no_mit,mf.f_return_sw).get_covar()#SS.covs_sw[0].get_ssc_covar()
        c_ssc0 = SS.multi_f.get_fisher(mf.f_spec_SSC_no_mit,mf.f_return_sw)[1].get_covar()#SS.covs_sw[0].get_ssc_covar()
        #c_ssc0 = c_sscs[0]
        if not np.allclose(c_ssc0,c_ssc0.T):
            eig_test_fails+=1
            warn("unperturbed result covariance not symmetric, unacceptable")
        #c_ssc1 = c_sscs[1]
        c_ssc1 = SS.multi_f.get_fisher(mf.f_spec_SSC_mit,mf.f_return_sw)[1].get_covar()#SS.covs_sw[0].get_ssc_covar()
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
        cg = SS.f_set_nopriors[0][1].get_covar()
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
            print "All sw eigenvalue sanity checks passed"
        else:
            warn(str(pert_test_fails)+" eigenvalue sanity checks failed")

        #TODO investigate positive semidefiniteness of jdem eigenvalues
        do_eig_interlace_check = False
        if do_eig_interlace_check:
            eig_interlace_fails_mit = 0
            eig_interlace_fails_no_mit = 0
            n_sw = mit_eigs_sw[0].size
            n_par = mit_eigs_par[0].size
            d_n = n_sw-n_par
            eig_l_mit_par = mit_eigs_par[0][::-1]
            eig_l_no_mit_par = no_mit_eigs_par[0][::-1]
            eig_l_mit_sw = mit_eigs_sw[0][::-1]
            eig_l_no_mit_sw = no_mit_eigs_sw[0][::-1]
            for i in xrange(0,n_par):
                if eig_l_mit_par[i]>eig_l_mit_sw[i]:
                    eig_interlace_fails_mit+=1
                if eig_l_no_mit_par[i]>eig_l_no_mit_sw[i]:
                    eig_interlace_fails_no_mit+=1
                if eig_l_mit_par[i]<eig_l_mit_sw[i+d_n]:
                    eig_interlace_fails_mit+=1
                if eig_l_no_mit_par[i]<eig_l_no_mit_sw[i+d_n]:
                    eig_interlace_fails_no_mit+=1
            if eig_interlace_fails_mit==0 and eig_interlace_fails_no_mit==0:
                print "All parameter eigenvalue interlace tests passed"
            else:
                warn(str(eig_interlace_fails_mit)+" mitigation and "+str(eig_interlace_fails_no_mit)+" no mitigation failures in parameter eigenvalue interlace tests")
