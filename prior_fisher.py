"""Load the prior fisher matrix from the jdem fom working group report and project to the necessary parameter space"""
import numpy as np
import fisher_matrix as fm
#jdem order is ns,Omegamh2,Omegabh2,Omegakh2,OmegaLh2,deltaGamma,deltaM,deltaG0,LogAs,ws36_0,ws36_1,...
JDEM_LABELS = np.hstack([np.array(['ns','Omegamh2','Omegabh2','Omegakh2','OmegaLh2','deltaGamma','deltaM','deltaG0','LogAs']),np.array(['ws36_'+str(_i) for _i in xrange(0,36)])])
class PriorFisher(object):
    """class to manage reading in prior fisher matrix"""
    def __init__(self,de_model,params,fisher_in=None,labels_in=None):
        """ de_model: 'jdem','constant_w','w0wa',or 'none'
            params:
                row_strip: indexes of rows to remove from source file
                fisher_source: source file
                n_full: how many total params there are
                n_row: how many de params there are
                z_step: z spacing of de params for jdem
            fisher_in: if not None, use this matrix instead of reading from file
            labels_in: array of labels for the axis elements"""
        self.params = params
        self.de_model = de_model
        #read in matrix
        if fisher_in is None:
            self.unprocessed_mat = read_prior_fisher(self.params)
            self.unprocessed_labels = JDEM_LABELS
        else:
            self.unprocessed_mat = fisher_in
            if labels_in is None:
                raise ValueError('labels_in must be set if fisher_in is set')
            self.unprocessed_labels = labels_in

        #strip unwanted rows
        self.stripped_mat,self.stripped_labels = fix_elements(self.unprocessed_mat,params,self.unprocessed_labels)

        #project to correct de model
        if de_model=='jdem':
            self.processed_mat = self.stripped_mat
            self.processed_labels = self.stripped_labels.copy()
        elif de_model=='constant_w':
            self.processed_mat,self.processed_labels = project_w0(self.stripped_mat,params,self.stripped_labels)
        elif de_model=='w0wa':
            self.processed_mat,self.processed_labels = project_w0wa(self.stripped_mat,params,self.stripped_labels)
        elif de_model=='none':
            self.processed_mat,self.processed_labels = project_no_de(self.stripped_mat,params,self.stripped_labels)
        else:
            raise ValueError('unrecognized de_model '+str(de_model))

        self.fisher_matrix = fm.FisherMatrix(self.processed_mat,input_type=fm.REP_FISHER,initial_state=fm.REP_FISHER,fix_input=True)

    def get_fisher(self):
        """return the FisherMatrix object"""
        return self.fisher_matrix
    def get_labels(self):
        """return the labels for the axes for fisher_matrix"""
        return self.processed_labels

def read_prior_fisher(params):
    """read in a matrix of priors from a source file"""
    prior_fisher_loc = params['fisher_source']
    fisher_d = np.loadtxt(prior_fisher_loc)
    row_is = fisher_d[:,0].astype(int)
    col_is = fisher_d[:,1].astype(int)
    n_row = np.max(row_is)+1
    fisher_mat = np.zeros((n_row,n_row))
    for itr in xrange(0,row_is.size):
        fisher_mat[row_is[itr],col_is[itr]] = fisher_d[itr,2]
    fisher_mat = (fisher_mat+fisher_mat.T)/2.
    return fisher_mat

def fix_elements(fisher_mat,params,labels):
    r""" fix parameters not of interest in read in prior fisher matrix
        several of the parameters (\omega_k,\delta\gamma, \delta M,ln(G_0))
        are not of interest to us at this time (all but \delta M could be incorporated straightforwardly)
    """
    rows = params['row_strip']
    #fix element by stripping row and column
    #delete larges indices first so lower indices unaffected
    rows = np.sort(rows)[::-1]
    #make sure does not change actual array
    fisher_new = fisher_mat.copy()
    labels_new = labels.copy()
    for i in xrange(0,rows.size):
        fisher_new = np.delete(fisher_new,(rows[i]),axis=0)
        fisher_new = np.delete(fisher_new,(rows[i]),axis=1)
        labels_new = np.delete(labels_new,(rows[i]),axis=0)
    return fisher_new,labels_new

def project_w0wa(fisher_mat,params,labels):
    """project jdem parametrization to w0wa"""
    n_de = params['n_de']
    n_new = fisher_mat.shape[0]-(n_de-2) #will project 36 dark energy entries to just 2
    fisher_new = np.zeros((n_new,n_new))
    #all the bins have constant change wrt w0
    project_mat = np.zeros((n_new,fisher_mat.shape[0]))
    project_mat[0:n_new-2,0:n_new-2] = np.identity(n_new-2)
    dwzdw0 = np.zeros(n_de)+1.
    project_mat[n_new-2,n_new-2::] = dwzdw0
    #z_step = params['z_step']
    #zs = z_step*np.arange(0,n_de)/(1-z_step*np.arange(0,n_de))
    #TODO make sure doing this correctly
    #a_s = 1.-np.arange(0,36)*0.025+0.0125
    #zs = 1./(1.+a_s)
    #change in wa is z*a
    dwzdwa = np.arange(0,36)*0.025+0.0125
    project_mat[n_new-1,n_new-2::] = dwzdwa
    fisher_new = np.dot(np.dot(project_mat,fisher_mat),project_mat.T)
    fisher_new = (fisher_new+fisher_new.T)/2.
    labels_new = np.hstack([labels[0:n_new-2].copy(),'w0','wa'])
    return fisher_new,labels_new

def project_w0(fisher_mat,params,labels):
    """project jdem parametrization to w0"""
    n_de = params['n_de']
    n_new = fisher_mat.shape[0]-(n_de-1) #will project 36 dark energy entries to just 1
    fisher_new = np.zeros((n_new,n_new))
    #all the bins have constant change wrt w0
    #made full length for testing purposes
    project_mat = np.zeros((fisher_mat.shape[0],fisher_mat.shape[0]))
    project_mat[0:n_new-1,0:n_new-1] = np.identity(n_new-1)
    dwzdw0 = np.zeros(n_de)+1.
    project_mat[n_new-1,n_new-1::] = dwzdw0
    #TODO are project_mats transposed?
    fisher_new = np.dot(np.dot(project_mat,fisher_mat),project_mat.T)
    fisher_new = fisher_new[0:n_new,0:n_new]
    fisher_new = (fisher_new+fisher_new.T)/2.
    labels_new = np.hstack([labels[0:n_new-1],'w'])
    return fisher_new,labels_new

def project_no_de(fisher_mat,params,labels):
    """project out all dark energy parameters"""
    n_de = params['n_de']
    n_new = fisher_mat.shape[0]-(n_de) #will project 36 dark energy entries to 0
    fisher_new = np.zeros((n_new,n_new))
    #all the bins have constant change wrt w0
    #project_mat = np.zeros((n_new,fisher_mat.shape[0]))
    fisher_new = fisher_mat[0:n_new,0:n_new]
    #project_mat[0:n_new,0:n_new] = np.identity(n_new)
    labels_new = labels[0:n_new].copy()
    return (fisher_new+fisher_new.T)/2.,labels_new

#def get_jdem_projected(params):
#    """get matrix in jdem parametrization"""
#    fisher_mat = read_prior_fisher(params=params)
#    fisher_strip,labels_strip = fix_elements(fisher_mat,params,JDEM_LABELS)
#    return fisher_strip,labels_strip
#
#def get_no_de_projected(params):
#    """get matrix with no de"""
#    fisher_mat = read_prior_fisher(params=params)
#    fisher_strip,labels_strip = fix_elements(fisher_mat,params,JDEM_LABELS)
#    fisher_project,labels_project = project_no_de(fisher_strip,params,labels_strip)
#    return fisher_project,labels_project
#
#def get_w0wa_projected(params):
#    """get matrix with w0wa parametrization"""
#    fisher_mat = read_prior_fisher(params=params)
#    fisher_strip,labels_strip = fix_elements(fisher_mat,params,JDEM_LABELS)
#    fisher_project,labels_project = project_w0wa(fisher_strip,params,labels_strip)
#    return fisher_project,labels_project
#
#def get_w0_projected(params):
#    """get matrix with constant w parametrization"""
#    fisher_mat = read_prior_fisher(params=params)
#    fisher_strip,labels_strip = fix_elements(fisher_mat,params,JDEM_LABELS)
#    fisher_project,labels_project = project_w0(fisher_strip,params,labels_strip)
#    return fisher_project,labels_project
