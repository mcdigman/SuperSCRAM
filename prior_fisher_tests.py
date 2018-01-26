"""test the prior_fisher module"""
#pylint: disable=W0621
import pytest
import numpy as np
#import defaults
#from prior_fisher import fix_elements,read_prior_fisher,project_w0,project_w0wa,PriorFisher,JDEM_LABELS
from prior_fisher import PriorFisher,JDEM_LABELS
EIG_SMALL = 1e-7
class PriorFisherTest(object):
    """class for managing test PriorFisher objects"""
    def __init__(self,key,de_type):
        """key: numerical key"""
        prior_fisher_params = { 'row_strip'     :np.array([3,5,6,7]),
                                'fisher_source' :'data/F_Planck_tau0.01.dat',
                                'n_full'        :45,
                                'n_de'          :36,
                                'z_step'        :0.025
                              }
        self.key = key
        self.de_type = de_type
        if key==1:
            self.fisher_prior = PriorFisher(de_type,prior_fisher_params)
        elif key==2:
            mat = np.outer(np.arange(1,46),np.arange(1,46))
            self.fisher_prior = PriorFisher(de_type,prior_fisher_params,fisher_in=mat,labels_in=JDEM_LABELS)
        else:
            raise ValueError('unrecognized key '+str(key))

test_list = [1,2]
type_list = ['constant_w','w0wa','jdem','none']
@pytest.fixture(params=type_list)
def de_type(request):
    """iterate through de models"""
    return request.param

@pytest.fixture(params=test_list)
def prior_fisher(request,de_type):
    """iterate through tests"""
    return PriorFisherTest(request.param,de_type)

def test_strip(prior_fisher):
    """test fixing elements works"""
    original = prior_fisher.fisher_prior.unprocessed_mat.copy()
    params = prior_fisher.fisher_prior.params.copy()
    original_labels = prior_fisher.fisher_prior.unprocessed_labels.copy()
    stripped_labels = prior_fisher.fisher_prior.stripped_labels.copy()
    stripped = prior_fisher.fisher_prior.stripped_mat.copy()
    row_strip = np.sort(params['row_strip'].copy())[::-1]

    n_removed = row_strip.size
    assert original.shape[0]-n_removed==stripped.shape[0]
    assert original.shape[1]-n_removed==stripped.shape[1]
    assert original_labels.shape[0]-n_removed==stripped_labels.shape[0]
    stripped_2 = original.copy()
    stripped_labels_2 = original_labels.copy()
    for index in row_strip:
        stripped_2 = np.vstack([stripped_2[0:index,:],stripped_2[index+1:,:]])
        stripped_2 = np.hstack([stripped_2[:,0:index],stripped_2[:,index+1:]])
        stripped_labels_2 = np.hstack([stripped_labels_2[0:index],stripped_labels_2[index+1:]])
    assert np.all(stripped_2==stripped)
    assert np.all(stripped_labels_2==stripped_labels)

    if prior_fisher.key==2:
        index_list = np.hstack([1,2,3,5,np.arange(9,46)])
        stripped_true = np.outer(index_list,index_list)
        print stripped_true[0:5,0:5]
        print stripped[0:5,0:5]
        print original[0:5,0:5]
        assert np.all(stripped_true==stripped)
    assert np.all(stripped.T==stripped)
    eig_strip = np.linalg.eigh(stripped)
    assert np.all(eig_strip[0][np.abs(eig_strip[0])>EIG_SMALL]>0.)

def test_de(prior_fisher):
    """test projecting de"""
    stripped = prior_fisher.fisher_prior.stripped_mat.copy()
#    params = prior_fisher.fisher_prior.params.copy()
    stripped_labels = prior_fisher.fisher_prior.stripped_labels.copy()
    processed_labels = prior_fisher.fisher_prior.processed_labels.copy()
    processed = prior_fisher.fisher_prior.processed_mat.copy()
    processed_2 = prior_fisher.fisher_prior.get_fisher().get_fisher().copy()
    assert np.all(processed==processed_2)
    assert np.all(stripped[0:5,0:5]==processed[0:5,0:5])
    assert np.all(stripped_labels[0:5]==processed_labels[0:5])
    if prior_fisher.fisher_prior.de_model=='constant_w':
        assert processed_labels.shape[0]==6
        assert processed.shape[0]==6
        assert processed.shape[1]==6
        assert np.all(processed_labels==np.array(['ns','Omegamh2','Omegabh2','OmegaLh2','LogAs','w']))
    elif prior_fisher.fisher_prior.de_model=='w0wa':
        assert processed_labels.shape[0]==7
        assert processed.shape[0]==7
        assert processed.shape[1]==7
        assert np.all(processed_labels==np.array(['ns','Omegamh2','Omegabh2','OmegaLh2','LogAs','w0','wa']))
    elif prior_fisher.fisher_prior.de_model=='jdem':
        assert processed_labels.shape[0]==41
        assert processed.shape[0]==41
        assert processed.shape[1]==41
        assert np.all(processed_labels[0:5]==np.array(['ns','Omegamh2','Omegabh2','OmegaLh2','LogAs']))
        for i in xrange(0,36):
            assert processed_labels[5+i]=='ws36_'+str(i)
        assert np.all(processed==stripped)
        assert np.all(processed_labels==stripped_labels)
    elif prior_fisher.fisher_prior.de_model=='none':
        assert processed_labels.shape[0]==5
        assert processed.shape[0]==5
        assert processed.shape[1]==5
        assert np.all(processed_labels==np.array(['ns','Omegamh2','Omegabh2','OmegaLh2','LogAs']))

    assert np.all(processed.T==processed)
    eig_process = np.linalg.eigh(processed)
    assert np.all(eig_process[0][np.abs(eig_process[0])>EIG_SMALL]>0.)

#TODO is ther a useful eigenvalue or similar test?
#def test_multi_project():
#    """test that information is removed each time matrix projected down"""
#    prior_fisher_params = {  'row_strip'     :np.array([3,5,6,7]),
#                            'fisher_source' :'data/F_Planck_tau0.01.dat',
#                            'n_full'        :45,
#                            'n_de'          :36,
#                            'z_step'        :0.025
#                         }
#    prior0 = PriorFisher('none',prior_fisher_params)
#    prior1 = PriorFisher('constant_w',prior_fisher_params)
#    prior2 = PriorFisher('w0wa',prior_fisher_params)
#    prior3 = PriorFisher('jdem',prior_fisher_params)
#    eig0 = np.linalg.eigh(prior0.get_fisher().get_fisher())[0]
#    eig1 = np.linalg.eigh(prior1.get_fisher().get_fisher())[0]
#    eig2 = np.linalg.eigh(prior2.get_fisher().get_fisher())[0]
#    eig3 = np.linalg.eigh(prior3.get_fisher().get_fisher())[0]
#    assert np.all(eig0<=eig1[-5:])
#    assert np.all(eig0<=eig2[-5:])
#    assert np.all(eig0<=eig3[-5:])
#    assert np.all(eig1[-5:]<=eig2[-5:])
#    #assert np.all(eig1[-1:]<=eig3[-1:])
#    #assert np.all(eig2[-1:]<=eig3[-1:])


if __name__=='__main__':
    pytest.cmdline.main(['prior_fisher_tests.py'])
#if __name__=='__main__':
#    param_1 = defaults.prior_fisher_params.copy()
#    param_1['row_strip'] = np.array([3,5,6,7])
#    fisher_mat = read_prior_fisher(params=param_1)
#    assert fisher_mat.shape==(45,45)
#    assert np.all(fisher_mat==fisher_mat.T)
#    fisher_strip = fix_elements(fisher_mat,param_1)
#    assert fisher_strip.shape==(41,41)
#    param_2 = defaults.prior_fisher_params.copy()
#    param_2['row_strip'] = np.array([5,6,7])
#    param_3 = defaults.prior_fisher_params.copy()
#    param_3['row_strip'] = np.array([3])
#    fisher_strip_567 = fix_elements(fisher_mat,params=param_2)
#    fisher_strip_3567 = fix_elements(fisher_strip_567,params=param_3)
#    assert np.all(fisher_strip_3567==fisher_strip)
#    #check stripping worked as expected
#    assert np.all(fisher_strip==fisher_strip.T)
#    assert np.all(fisher_strip_567==fisher_strip_567.T)
#    n_r = 5
#    n_c = 3
#    n_skip = 0
#    assert np.all(fisher_strip_567[0:n_r-n_skip,0:n_r-n_skip]==fisher_mat[0:n_r-n_skip,0:n_r-n_skip])
#    assert np.all(fisher_strip_567[0:n_r-n_skip,n_r::]==fisher_mat[0:n_r-n_skip,n_r+n_c::])
#    assert np.all(fisher_strip_567[n_r::,n_r::]==fisher_mat[n_r+n_c::,n_r+n_c::])
#
#    n_r = 3
#    n_c = 1
#    n_skip = 0
#    assert np.all(fisher_strip_3567[0:n_r-n_skip,0:n_r-n_skip]==fisher_strip_567[0:n_r-n_skip,0:n_r-n_skip])
#    assert np.all(fisher_strip_3567[0:n_r-n_skip,n_r::]==fisher_strip_567[0:n_r-n_skip,n_r+n_c::])
#    assert np.all(fisher_strip_3567[n_r::,n_r::]==fisher_strip_567[n_r+n_c::,n_r+n_c::])
#
#    fisher_project,project_mat = project_w0(fisher_strip,params=param_1,return_project=True)
#    assert fisher_project.shape==(6,6)
#    #projection matrix must be idempotent, have determinant 0, trace=dimension of subspace projected 2,all eigenvalues 0 or 1
#    assert np.all(np.dot(project_mat,project_mat)==project_mat)
#    assert np.linalg.det(project_mat)==0.
#    assert np.trace(project_mat)==6
#    project_eig = np.linalg.eig(project_mat)[0]
#    assert np.all((project_eig==0) | (project_eig==1))
#
#    fisher_project2,project_mat2 = project_w0(fisher_mat,params=param_1,return_project=True)
#    assert np.all(np.dot(project_mat2,project_mat2)==project_mat2)
#    assert np.linalg.det(project_mat2)==0.
#    assert np.trace(project_mat2)==10
#    project_eig2 = np.linalg.eig(project_mat2)[0]
#    assert np.all((project_eig2==0) | (project_eig2==1))
#
#    fisher_strip2 = fix_elements(fisher_project2)
#    assert np.all(fisher_strip2==fisher_project)
