import numpy as np
import defaults
def read_planck_fisher(params = defaults.planck_fisher_params):
    planck_fisher_loc = params['fisher_source']
    fisher_d = np.loadtxt(planck_fisher_loc)
    row_is = fisher_d[:,0].astype(int)
    col_is = fisher_d[:,1].astype(int)
    n_row = np.max(row_is)+1
    fisher_mat = np.zeros((n_row,n_row))
    for itr in range(0,row_is.size):
        fisher_mat[row_is[itr],col_is[itr]] = fisher_d[itr,2]
    #NOTE: element 8,14 is not identical to 14,8??? very close though,just symmetrize to eliminate the problem
    fisher_mat=(fisher_mat+fisher_mat.T)/2.
    return fisher_mat

#several of the parameters (\omega_k,\delta\gamma, \delta M,ln(G_0)) are not of interest to us at this time (all but \delta M could be incorporated straightforwardly)
def fix_elements(fisher_mat,params=defaults.planck_fisher_params):
    rows = params['row_strip']
    #fix element by stripping row and column
    #delete larges indices first so lower indices unaffected
    rows = np.sort(rows)[::-1]
    #make sure does not change actual array
    fisher_new = fisher_mat.copy()
    for i in range(0,rows.size):
        fisher_new=np.delete(fisher_new,(rows[i]),axis=0)
        fisher_new=np.delete(fisher_new,(rows[i]),axis=1)
    return fisher_new

def project_w0wa(fisher_mat,params=defaults.planck_fisher_params):
    n_de = params['n_de']
    n_new= fisher_mat.shape[0]-(n_de-2) #will project 36 dark energy entries to just 2
    fisher_new=np.zeros((n_new,n_new))
    #all the bins have constant change wrt w0
    project_mat = np.zeros((n_new,fisher_mat.shape[0]))
    project_mat[0:n_new-2,0:n_new-2] = np.identity(n_new-2)
    dwzdw0 = np.zeros(n_de)+1.
    project_mat[n_new-2,n_new-2::] = dwzdw0
    z_step = params['z_step']
    zs = z_step*np.arange(0,n_de)/(1-z_step*np.arange(0,n_de))
    #change in wa is z*a
    dwzdwa =zs/(1.+zs)
    project_mat[n_new-1,n_new-2::] = dwzdwa
    fisher_new= np.dot(np.dot(project_mat,fisher_mat),project_mat.T)
    return fisher_new


def project_w0(fisher_mat,params=defaults.planck_fisher_params,return_project=False):
    n_de = params['n_de']
    n_new= fisher_mat.shape[0]-(n_de-1) #will project 36 dark energy entries to just 1
    fisher_new=np.zeros((n_new,n_new))
    #all the bins have constant change wrt w0
    #made full length for testing purposes
    project_mat = np.zeros((fisher_mat.shape[0],fisher_mat.shape[0]))
    project_mat[0:n_new-1,0:n_new-1] = np.identity(n_new-1)
    dwzdw0 = np.zeros(n_de)+1.
    project_mat[n_new-1,n_new-1::] = dwzdw0
    fisher_new= np.dot(np.dot(project_mat,fisher_mat),project_mat.T)
    fisher_new = fisher_new[0:n_new,0:n_new]
    if return_project:
        return fisher_new,project_mat
    else:
        return fisher_new

def get_w0wa_projected(params=defaults.planck_fisher_params):
    fisher_mat = read_planck_fisher(params=params)
    fisher_strip = fix_elements(fisher_mat,params=params)
    fisher_project = project_w0wa(fisher_strip,params=params)
    return fisher_project

def get_w0_projected(params=defaults.planck_fisher_params):
    fisher_mat = read_planck_fisher(params=params)
    fisher_strip = fix_elements(fisher_mat,params=params)
    fisher_project = project_w0(fisher_strip,params=params)
    return fisher_project

if __name__=='__main__':
    param_1 = defaults.planck_fisher_params.copy()
    param_1['row_strip']=np.array([3,5,6,7])
    fisher_mat=read_planck_fisher(params=param_1)
    assert(fisher_mat.shape==(45,45))
    assert(np.all(fisher_mat==fisher_mat.T))
    fisher_strip = fix_elements(fisher_mat,param_1)
    assert(fisher_strip.shape==(41,41))
    param_2 = defaults.planck_fisher_params.copy()
    param_2['row_strip']=np.array([5,6,7])
    param_3 = defaults.planck_fisher_params.copy()
    param_3['row_strip']=np.array([3])
    fisher_strip_567 = fix_elements(fisher_mat,params=param_2)
    fisher_strip_3567 = fix_elements(fisher_strip_567,params=param_3)
    assert(np.all(fisher_strip_3567==fisher_strip))
    #check stripping worked as expected
    assert(np.all(fisher_strip==fisher_strip.T))
    assert(np.all(fisher_strip_567==fisher_strip_567.T))
    n_r = 5
    n_c = 3
    n_skip = 0
    assert(np.all(fisher_strip_567[0:n_r-n_skip,0:n_r-n_skip]==fisher_mat[0:n_r-n_skip,0:n_r-n_skip]))
    assert(np.all(fisher_strip_567[0:n_r-n_skip,n_r::]==fisher_mat[0:n_r-n_skip,n_r+n_c::]))
    assert(np.all(fisher_strip_567[n_r::,n_r::]==fisher_mat[n_r+n_c::,n_r+n_c::]))

    n_r = 3
    n_c = 1
    n_skip = 0
    assert(np.all(fisher_strip_3567[0:n_r-n_skip,0:n_r-n_skip]==fisher_strip_567[0:n_r-n_skip,0:n_r-n_skip]))
    assert(np.all(fisher_strip_3567[0:n_r-n_skip,n_r::]==fisher_strip_567[0:n_r-n_skip,n_r+n_c::]))
    assert(np.all(fisher_strip_3567[n_r::,n_r::]==fisher_strip_567[n_r+n_c::,n_r+n_c::]))

    fisher_project,project_mat = project_w0(fisher_strip,params=param_1,return_project=True) 
    assert(fisher_project.shape==(6,6))
    #projection matrix must be idempotent, have determinant 0, trace=dimension of subspace projected 2,all eigenvalues 0 or 1
    assert(np.all(np.dot(project_mat,project_mat)==project_mat))
    assert(np.linalg.det(project_mat)==0.)
    assert(np.trace(project_mat)==6)
    project_eig = np.linalg.eig(project_mat)[0]
    assert(np.all((project_eig==0) | (project_eig==1)))
   
    fisher_project2,project_mat2 = project_w0(fisher_mat,params=param_1,return_project=True)
    assert(np.all(np.dot(project_mat2,project_mat2)==project_mat2))
    assert(np.linalg.det(project_mat2)==0.)
    assert(np.trace(project_mat2)==10)
    project_eig2 = np.linalg.eig(project_mat2)[0]
    assert(np.all((project_eig2==0) | (project_eig2==1)))

    fisher_strip2 = fix_elements(fisher_project2)
    assert(np.all(fisher_strip2==fisher_project))

    fisher_project3= get_w0_projected(params=param_1)
    assert(np.all(fisher_project3==fisher_project))
    import matplotlib.pyplot as plt
    plt.imshow(np.log(np.abs(fisher_project)))
    plt.show()
