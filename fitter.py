# Copyright Haotian Yang, modified by Hao Zhu, 2020-09
# Fit facial 3DMM from 68 3D-keypoints
import numpy as np
import json, tqdm
from scipy.optimize import minimize
from scipy.linalg import orthogonal_procrustes

class fitter:
    def __init__(self, 
                 fn_params = "./predefine_data/factor_847_50_52.json", 
                 fn_core = "./predefine_data/core_847_50_52.npy"):
        
        # read params
        with open(fn_params, 'r') as f:
            content_dict = json.load(f)
        self.id_mean = np.array(content_dict['id_mean'])
        self.id_var = np.array(content_dict['id_var'])
        self.lm_index_full = np.array(content_dict['lm_index_full'])
        self.factors_id_0 = np.array(content_dict['factors_id_0'])
        self.faces = np.array(content_dict['faces'])
        
        # read core
        self.core_tensor = np.load(fn_core).transpose((2, 1, 0))
        for i in range(51):
            self.core_tensor[:, i + 1, :] = self.core_tensor[:, i + 1, :] - self.core_tensor[:, 0, :]
    
    def get_faces(self):
        return self.faces
        
    def optimize_rigid_pos(self, recon_verts, tar_verts):
        tar_center = np.mean(tar_verts, axis=0)
        recon_center = np.mean(recon_verts, axis=0)
        tar_verts_centered = tar_verts - tar_center
        recon_verts_centered = recon_verts - recon_center
        scale_recon = np.linalg.norm(recon_verts_centered) / np.linalg.norm(tar_verts_centered)
        recon_verts_centered = recon_verts / scale_recon
        translate = tar_center
        rotation, _ = orthogonal_procrustes(tar_verts_centered, recon_verts_centered)
        return 1 / scale_recon, translate, rotation


    def compute_res_id(self, id, id_matrix, scale, trans, rot_matrix, tar_verts):
        id_matrix = id_matrix.reshape(-1, id.shape[0])
        recon_verts = id_matrix.dot(id).reshape((-1, 3))
        recon_verts = recon_verts.ravel()
        return np.linalg.norm(recon_verts - tar_verts) ** 2 + 20 * \
               (id - self.id_mean).dot(np.diag(1 / self.id_var)).dot(np.transpose([id - self.id_mean]))


    def optimize_identity(self, scale, trans, rot_matrix, id, exp, core_tensor, tar_verts):
        id_matrix = np.tensordot(core_tensor, exp, axes=([1], [0])).ravel()
        tar_verts = tar_verts.ravel()
        result = minimize(self.compute_res_id, id, method='L-BFGS-B', 
                          args=(id_matrix, scale, trans, rot_matrix, tar_verts),
                          options={'maxiter': 100})
        return result.x

    def compute_res_exp(self, exp, exp_matrix, scale, trans, rot_matrix, tar_verts):
        exp_matrix = exp_matrix.reshape(-1, exp.shape[0] + 1)
        exp_full = np.ones(52)
        exp_full[1:52] = exp
        recon_verts = exp_matrix.dot(exp_full).reshape((-1, 3))
        recon_verts = recon_verts.ravel()
        return np.linalg.norm(recon_verts - tar_verts) ** 2

    def optimize_expression(self, scale, trans, rot_matrix, id, exp, core_tensor, tar_verts):
        exp_matrix = np.dot(core_tensor, id).ravel()
        tar_verts = tar_verts.ravel()
        bounds = []
        for i in range(exp.shape[0] - 1):
            bounds.append((0, 1))
        result = minimize(self.compute_res_exp, exp[1:52], method='L-BFGS-B', bounds=bounds,
                          args=(exp_matrix, scale, trans, rot_matrix, tar_verts), 
                          options={'maxiter': 100})
        exp_full = np.ones(52)
        exp_full[1:52] = result.x
        return exp_full
        
    # input is 68 x 3 numpy array or list
    def fit(self, lm_3d):
        
        lm_3d = np.asarray(lm_3d)        
        
        lm_core_tensor = self.core_tensor[self.lm_index_full]        

        # initialize with any id
        id = self.factors_id_0 
        
        # intialize with neutral expression
        exp = np.zeros(52)
        exp[0] = 1 
        
        recon_lm = lm_core_tensor.dot(id).dot(exp).reshape((-1, 3))
        #for optimize_loop in tqdm.trange(5):
        for optimize_loop in range(5):
            scale, trans, rot_matrix = self.optimize_rigid_pos(recon_lm, lm_3d)
            tar_verts_align = lm_3d.copy()
            tar_verts_align = rot_matrix.T.dot((tar_verts_align - trans).T / scale).T + \
                              np.mean(recon_lm, 0)
            id = self.optimize_identity(scale, trans, rot_matrix, id, exp, 
                                        lm_core_tensor, tar_verts_align)
            exp = self.optimize_expression(scale, trans, rot_matrix, id, exp, 
                                           lm_core_tensor, tar_verts_align)
            recon_lm = lm_core_tensor.dot(id).dot(exp).reshape((-1, 3))

        recon_verts = self.core_tensor.dot(id).dot(exp).reshape((-1, 3))
        scale, trans, rot_matrix = self.optimize_rigid_pos(recon_lm, lm_3d)
        recon_verts = rot_matrix.dot((recon_verts - np.mean(recon_lm, 0)).T * scale).T + trans
        
        return recon_verts, rot_matrix, trans, scale

# demo main
def main():
    # initialize
    fit_model = fitter()
    
    # read landmarks
    lm_3d = np.zeros((68, 3))
    with open('./data/kp_3d.txt') as f:
        for i, line in enumerate(f.readlines()):
            lm_3d[i] = list(map(float, line.split()))
    
    # fit
    verts = fit_model.fit(lm_3d)
    faces = fit_model.get_faces()
    
    # save out
    with open('data/fit.obj', "w") as f:
        for vert in verts:
            f.write("v %.6f %.6f %.6f\n" % (vert[0], vert[1], vert[2]))
        for face in faces:
            f.write("f %d %d %d\n" % (face[0], face[1], face[2]))

if __name__ == "__main__":
    main()
