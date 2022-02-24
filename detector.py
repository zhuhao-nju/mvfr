# Copyright Hao Zhu, 2020-09
# get 3d keypoints from multi-view facial images

import face_alignment, cv2, os, json, camera, tqdm
import numpy as np
from scipy.optimize import least_squares

class detector:
    def __init__(self, fn_params, idx_list, img_format, img_dir, debug = False):
        
        self.debug = debug
        self.kp_num = 68
        self.img_dir = img_dir
        self.idx_list = idx_list
        self.img_format = img_format
        
        with open(fn_params, 'r') as f:
            cam_params = json.load(f)
        self.cam_list = []        
        
        for idx in idx_list:
            K = np.array(cam_params['%s_K' % idx])
            Rt = np.array(cam_params['%s_Rt' % idx])
            h_src = cam_params['%s_height' % idx]
            w_src = cam_params['%s_width' % idx]
            cam = camera.CamPara(Rt = Rt, K = K, img_size = (w_src, h_src))
            self.cam_list.append(cam)
        
        # initialize fa
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, 
                                               flip_input=False)
        
    # get the closest point to input rays
    def locate(self, rays):
        
        ray_start_positions = []
        for ray in rays:
            ray_start_positions.append(ray[0])
        starting_P = np.stack(ray_start_positions).mean(axis=0).ravel()

        # solver with least squares
        result = least_squares(self.distance, starting_P, kwargs={'rays': rays})
        
        return result

    # calculate the distance between a point and each ray
    def distance(self, P, rays):
       
        dims = len(rays[0][0])
        errors = np.full([len(rays)*dims,1], np.inf)
        
        # calculate the errors
        for i, _ in enumerate(rays):
            S, D = rays[i]
            t_P = D.dot((P - S).T)/(D.dot(D.T))
            if t_P > 0:
                errors[i*dims:(i+1)*dims, 0] = (P - (S + t_P * D)).T
            else:
                errors[i*dims:(i+1)*dims, 0] = (P - S).T
                
        return errors.ravel()

    # detect 2d key points for multi-view images using face_alignment
    def get_kp2d(self, sc_ratio = 0.1):
        kp2d_list = []
        #for idx in tqdm.tqdm(self.idx_list):
        for i, idx in enumerate(self.idx_list):
            src_img = cv2.imread(self.img_dir + "%s%s" % (idx, self.img_format[i]))
            src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
            src_img_sc = cv2.resize(src_img, (int(src_img.shape[1]*sc_ratio), 
                                              int(src_img.shape[0]*sc_ratio)))
            preds = self.fa.get_landmarks(src_img_sc)

            if preds is None:
                kp2d_list.append([])
            elif len(preds) != 1:
                kp2d_list.append([])
            else:
                kp2d_list.append(preds[0]/sc_ratio)
            
            # only for debug
            if self.debug is True:
                src_img_draw = src_img_sc.copy()
                for kp in preds[0]:
                    cv2.circle(src_img_draw, tuple(kp), 5, (255, 0, 0), -1)
                src_img_draw = cv2.cvtColor(src_img_draw, cv2.COLOR_BGR2RGB)
                cv2.imwrite("./test_data/tmp_%s.png" % idx, src_img_draw)
            
        return kp2d_list
    
    def solve_3d(self, kp2d_list = None):
        
        if kp2d_list is None:
            kp2d_list = self.get_kp2d()
        
        if len(kp2d_list) < 2:
            return []
        # make N ray list
        kp3d_list = []
        for kp_idx in range(self.kp_num):
            my_rays = []
            for view_num in range(len(kp2d_list)):
                if len(kp2d_list[view_num])>0:
                    ray_center = self.cam_list[view_num].get_camcenter()
                    ray_point = self.cam_list[view_num].inv_project(\
                                kp2d_list[view_num][kp_idx], 1)
                    ray_dir = ray_point - ray_center

                    # build rays
                    my_rays.append((ray_center, ray_dir))

            # solve for 3D kp
            result = self.locate(my_rays)
            kp3d_list.append(result['x'].tolist())
        if self.debug is False:
            return kp3d_list
        else:
            return kp3d_list, kp2d_list

# demo main
def main():
    
    img_name_list = os.listdir("../test_data/mvkp_result/images/")
    idx_list = [int(fn.split('.')[0]) for fn in img_name_list]
    
    kp_detector = detector(fn_params = "./data/1_neutral_mv_params.json", 
                           idx_list = idx_list,
                           img_dir = "../test_data/mvkp_result/images/")
    
    # read kp list
    #kp_list = np.load("../test_data/mvkp_result/825_test_kp.npy")
    
    # solve for 3d key points
    kp3d_list = kp_detector.solve_3d()
    
    for kp3d in kp3d_list:
        print("%f %f %f" % (kp3d[0], kp3d[1], kp3d[2]))
    
    print("Done")
    
if __name__ == "__main__":
    main()
