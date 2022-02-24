import numpy as np

# Basic camera projection and inv-projection
class CamPara():
    def __init__(self, K=None, Rt=None, img_size = [200, 200]):
        if K is None:
            K = np.array([[500, 0, 112],
                          [0, 500, 112],
                          [0, 0, 1]])
        else:
            K = np.array(K)
        if Rt is None:
            Rt = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0]])
        else:
            Rt = np.array(Rt)
        R = Rt[:, :3]
        t = Rt[:, 3]
        self.cam_center = -np.dot(R.transpose(), t)

        # compute projection and inv-projection matrix
        self.proj_mat = np.dot(K, Rt)

        self.K = K
        self.Rt = Rt

    def get_camcenter(self):
        return self.cam_center

    # def get_center_dir(self):
    #    return self.center_dir

    def project(self, p_xyz):
        p_xyz = np.double(p_xyz)
        p_uv_1 = np.dot(self.proj_mat, np.append(p_xyz, 1))
        if p_uv_1[2] == 0:
            return 0
        p_uv = (p_uv_1 / p_uv_1[2])[:2]
        return p_uv

    # inverse projection
    def inv_project(self, p_uv, depth):
        '''
        znear = 0.1
        zfar = 100
        P = np.zeros((4, 4))
        P[0][0] = 2.0 * self.K[0, 0] / width
        P[1][1] = 2.0 * self.K[0, 0] / height
        P[0][2] = 1.0 - 2.0 * self.K[0, 2] / width
        P[1][2] = 2.0 * self.K[1, 2] / height - 1.0
        P[3][2] = -1.0
        if zfar is None:
            P[2][2] = -1.0
            P[2][3] = -2.0 * znear
        else:
            P[2][2] = (zfar + znear) / (znear - zfar)
            P[2][3] = (2 * zfar * znear) / (znear - zfar)

        p_uv = np.double(p_uv)
        p_uv[1] = -p_uv[0] / height * 2 + 1
        p_uv[0] = p_uv[1] / width * 2 - 1
        p_uv = np.append(p_uv, -(znear + zfar - (2 * znear * zfar) / depth) / (znear - zfar))
        p_uv = np.append(p_uv, 1) * depth
        p_xyz = np.linalg.inv(P).dot(p_uv)
        p_xyz[2] = -p_xyz[2]
        p_xyz[[0, 1]] = p_xyz[[1, 0]]
        real_xyz = self.Rt[:, 0:3].T.dot(p_xyz[0:3] - self.Rt[:, 3])
        '''
        p_uv = np.double(p_uv)
        p_uv[0] = p_uv[0]
        p_xyz = np.append(p_uv, 1) * depth
        p_xyz = np.linalg.inv(self.K).dot(p_xyz)
        real_xyz = self.Rt[:, 0:3].T.dot(p_xyz[0:3] - self.Rt[:, 3])

        return real_xyz

