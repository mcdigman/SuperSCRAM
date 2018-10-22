"""test the prior_fisher module"""
#pylint: disable=W0621
from __future__ import print_function,division,absolute_import
from builtins import range
import pytest
import numpy as np
import prior_fisher as pf
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
        print(stripped_true[0:5,0:5])
        print(stripped[0:5,0:5])
        print(original[0:5,0:5])
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
        for i in range(0,36):
            assert processed_labels[5+i]=='ws36_'+str(i).zfill(2)
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

def test_consistency():
    """do some consistency checks for the projection"""
    prior_fisher_params = { 'row_strip'     :np.array([3,5,6,7]),
                            'fisher_source' :'data/F_Planck_tau0.01.dat',
                            'n_full'        :45,
                            'n_de'          :36,
                            'z_step'        :0.025
                          }
    mat = np.random.rand(45,45)
    mat = np.dot(mat.T,mat)
    priors = np.random.rand(45,45)
    priors = np.dot(priors.T,priors)
    assert np.all(np.linalg.eigh(mat)[0]>=0.)
    assert np.all(np.linalg.eigh(priors)[0]>=0.)

    fp1 = PriorFisher('jdem',prior_fisher_params,fisher_in=mat,labels_in=JDEM_LABELS)
    fp2 = PriorFisher('w0wa',prior_fisher_params,fisher_in=mat,labels_in=JDEM_LABELS)
    fp3 = PriorFisher('constant_w',prior_fisher_params,fisher_in=mat,labels_in=JDEM_LABELS)
    res0 = fp3.get_fisher().get_fisher()
    res1 = pf.project_w0wa_to_w0(fp2.get_fisher().get_fisher(),prior_fisher_params,fp2.processed_labels)[0]
    res2 = pf.project_w0(fp1.get_fisher().get_fisher(),prior_fisher_params,fp1.processed_labels)[0]
    res3_int = pf.project_w0wa(fp1.get_fisher().get_fisher(),prior_fisher_params,fp1.processed_labels)[0]
    res3 = pf.project_w0wa_to_w0(res3_int,prior_fisher_params,fp2.processed_labels)[0]
    assert np.allclose(res3_int[0:6,0:6],res3)
    assert np.all(np.linalg.eigh(res0)[0]>=0.)
    assert np.all(np.linalg.eigh(res1)[0]>=0.)
    assert np.all(np.linalg.eigh(res2)[0]>=0.)
    assert np.all(np.linalg.eigh(res3_int)[0]>=0.)
    assert np.all(np.linalg.eigh(res3)[0]>=0.)
    assert np.allclose(res0,res1)
    assert np.allclose(res0,res2)
    assert np.allclose(res0,res3)
    assert np.allclose(fp2.get_fisher().get_fisher(),res3_int)
    assert np.isclose(np.linalg.det(res0),np.linalg.det(res1))
    assert np.isclose(np.linalg.det(res0),np.linalg.det(res2))
    assert np.isclose(np.linalg.det(res0),np.linalg.det(res3))
    priors = priors
    fp1p = PriorFisher('jdem',prior_fisher_params,fisher_in=priors,labels_in=JDEM_LABELS)
    fp2p = PriorFisher('w0wa',prior_fisher_params,fisher_in=priors,labels_in=JDEM_LABELS)
    fp3p = PriorFisher('constant_w',prior_fisher_params,fisher_in=priors,labels_in=JDEM_LABELS)
    res0p = fp3p.get_fisher().get_fisher()
    res1p = pf.project_w0wa_to_w0(fp2p.get_fisher().get_fisher(),prior_fisher_params,fp2p.processed_labels)[0]
    res2p = pf.project_w0(fp1p.get_fisher().get_fisher(),prior_fisher_params,fp1p.processed_labels)[0]
    res3_intp = pf.project_w0wa(fp1p.get_fisher().get_fisher(),prior_fisher_params,fp1p.processed_labels)[0]
    res3p = pf.project_w0wa_to_w0(res3_intp,prior_fisher_params,fp2p.processed_labels)[0]
    assert np.allclose(res3_intp[0:6,0:6],res3p)
    assert np.all(np.linalg.eigh(res0p)[0]>=0.)
    assert np.all(np.linalg.eigh(res1p)[0]>=0.)
    assert np.all(np.linalg.eigh(res2p)[0]>=0.)
    assert np.all(np.linalg.eigh(res3_intp)[0]>=0.)
    assert np.all(np.linalg.eigh(res3p)[0]>=0.)
    assert np.allclose(res0p,res1p)
    assert np.allclose(res0p,res2p)
    assert np.allclose(res0p,res3p)
    assert np.allclose(fp2p.get_fisher().get_fisher(),res3_intp)
    assert np.isclose(np.linalg.det(res0p),np.linalg.det(res1p))
    assert np.isclose(np.linalg.det(res0p),np.linalg.det(res2p))
    assert np.isclose(np.linalg.det(res0p),np.linalg.det(res3p))


    alt_prior_fisher_params = { 'row_strip'     :np.array([]),
                                'fisher_source' :'data/F_Planck_tau0.01.dat',
                                'n_full'        :41,
                                'n_de'          :36,
                                'z_step'        :0.025
                              }
    prior_mat = priors+mat
    fp1c = PriorFisher('jdem',prior_fisher_params,fisher_in=prior_mat,labels_in=JDEM_LABELS)
    fp2c = PriorFisher('w0wa',prior_fisher_params,fisher_in=prior_mat,labels_in=JDEM_LABELS)
    fp3c = PriorFisher('constant_w',prior_fisher_params,fisher_in=prior_mat,labels_in=JDEM_LABELS)
    fp1c2 = PriorFisher('jdem',alt_prior_fisher_params,fisher_in=fp1p.get_fisher().get_fisher()+fp1.get_fisher().get_fisher(),labels_in=fp1.processed_labels)
    fp2c2 = PriorFisher('w0wa',alt_prior_fisher_params,fisher_in=fp1p.get_fisher().get_fisher()+fp1.get_fisher().get_fisher(),labels_in=fp1.processed_labels)
    fp3c2 = PriorFisher('constant_w',alt_prior_fisher_params,fisher_in=fp1p.get_fisher().get_fisher()+fp1.get_fisher().get_fisher(),labels_in=fp1.processed_labels)
    sum_in1 = fp1p.get_fisher().get_fisher()+fp1.get_fisher().get_fisher()
    sum_in2 = fp2p.get_fisher().get_fisher()+fp2.get_fisher().get_fisher()
    sum_in3 = fp3p.get_fisher().get_fisher()+fp3.get_fisher().get_fisher()
    res4 = fp1c.get_fisher().get_fisher()
    res5 = fp2c.get_fisher().get_fisher()
    res6 = fp3c.get_fisher().get_fisher()
    res7 = pf.project_w0wa_to_w0(fp2c.get_fisher().get_fisher(),alt_prior_fisher_params,fp2c.processed_labels)[0]
    res8 = pf.project_w0wa(fp1c.get_fisher().get_fisher(),alt_prior_fisher_params,fp3c.processed_labels)[0]
    res9 = pf.project_w0wa_to_w0(res8,alt_prior_fisher_params,fp2c.processed_labels)[0]
    res10 = fp1c2.get_fisher().get_fisher()
    res11 = fp2c2.get_fisher().get_fisher()
    res12 = fp3c2.get_fisher().get_fisher()
    assert np.allclose(res8[0:6,0:6],res9)
    assert np.all(np.linalg.eigh(res4)[0]>=0.)
    assert np.all(np.linalg.eigh(res5)[0]>=0.)
    assert np.all(np.linalg.eigh(res6)[0]>=0.)
    assert np.all(np.linalg.eigh(res7)[0]>=0.)
    assert np.all(np.linalg.eigh(res8)[0]>=0.)
    assert np.all(np.linalg.eigh(res9)[0]>=0.)
    assert np.all(np.linalg.eigh(res10)[0]>=0.)
    assert np.all(np.linalg.eigh(res11)[0]>=0.)
    assert np.all(np.linalg.eigh(res12)[0]>=0.)
    assert np.allclose(res4,sum_in1)
    assert np.allclose(res5,sum_in2)
    assert np.allclose(res6,sum_in3)
    assert np.allclose(res6,res7)
    assert np.allclose(res5,res8)
    assert np.allclose(res6,res9)
    assert np.allclose(res4,res10)
    assert np.allclose(res5,res11)
    assert np.allclose(res6,res12)

    #interlace tests
    eig2 = np.linalg.eigh(res2)[0]
    eig3 = np.linalg.eigh(res3)[0]
    eig_diff = (eig2[::-1][1:eig3.size]-eig3[::-1][0:eig3.size-1])
    eig2p = np.linalg.eigh(res2p)[0]
    eig3p = np.linalg.eigh(res3p)[0]
    eig_diffp = (eig2p[::-1][1:eig3p.size]-eig3p[::-1][0:eig3p.size-1])
    assert np.all(eig_diffp<=0.)
    eig2c = np.linalg.eigh(res5)[0]
    eig3c = np.linalg.eigh(res6)[0]
    eig_diffc = (eig2c[::-1][1:eig3c.size]-eig3c[::-1][0:eig3c.size-1])
    assert np.all(eig_diffc<=0.)




if __name__=='__main__':
    pytest.cmdline.main(['prior_fisher_tests.py'])
