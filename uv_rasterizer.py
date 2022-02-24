# Debug version
# bug fixed in 2021/04/22, zhuhao
from objloader import *
from psbody.mesh import Mesh
import pickle, tqdm, cv2
import openmesh as om

# rasterize from mesh to uv space with a certain attributes, 
# some parameters are pre-computed for fast processing
class uv_rasterizer:
    def __init__(self, uv_size = 1024, enable_frontal = False,
                 ref_mesh_dirname = "../data/pre_defined/ref_face_fill.obj",
                 front_face_dirname = "./data_gen/front_face_indices.pkl",
                ):
        self.uv_size = uv_size
        self.enable_frontal = enable_frontal
        self.ref_mesh = OBJ(ref_mesh_dirname)
        self.predef_pixels_pos = None
        self.predef_bc = None
        self.predef_v_idx = None
        
        if self.enable_frontal is True:
            with open(front_face_dirname, 'rb') as f:
                self.face_indices = pickle.load(f)
        else:
            self.face_indices = list(range(len(self.ref_mesh.faces)))
        
        self.init_compute()
        
    def init_compute(self):
        predef_pixels_pos = []
        predef_bc = []
        predef_v_idx = []
        for i, face_idx in enumerate(tqdm.tqdm(self.face_indices)):
            face = self.ref_mesh.faces[int(face_idx)]
            ver_indices, _, texture_coords, _ = face
            texture_shape = (self.uv_size, self.uv_size)
            t1_x = self.ref_mesh.texcoords[texture_coords[0] - 1][0] * texture_shape[1]
            t1_y = self.ref_mesh.texcoords[texture_coords[0] - 1][1] * texture_shape[0]
            t2_x = self.ref_mesh.texcoords[texture_coords[1] - 1][0] * texture_shape[1]
            t2_y = self.ref_mesh.texcoords[texture_coords[1] - 1][1] * texture_shape[0]
            t3_x = self.ref_mesh.texcoords[texture_coords[2] - 1][0] * texture_shape[1]
            t3_y = self.ref_mesh.texcoords[texture_coords[2] - 1][1] * texture_shape[0]
            
            v0 = np.array([t2_x - t1_x, t2_y - t1_y], dtype=np.float64)
            v1 = np.array([t3_x - t1_x, t3_y - t1_y], dtype=np.float64)

            if max(abs(t1_x - t2_x), abs(t2_x - t3_x), abs(t3_x - t1_x), abs(t1_y - t2_y), 
                   abs(t2_y - t3_y),abs(t3_y - t1_y)) > texture_shape[0] / 10:
                print("Warning: too large faces in UV coordinate found.")
                continue
            
            for j in range(min(int(t1_x), int(t2_x), int(t3_x))-1, max(int(t1_x), int(t2_x), int(t3_x))+1):
                for k in range(min(int(t1_y), int(t2_y), int(t3_y))-1, max(int(t1_y), int(t2_y), int(t3_y))+1):
                    v2 = np.array([j + 0.5 - t1_x, k + 0.5 - t1_y], dtype=np.float64)
                    v_1_1 = v1.dot(v1)
                    v_2_0 = v2.dot(v0)
                    v_1_0 = v1.dot(v0)
                    v_2_1 = v2.dot(v1)
                    v_0_0 = v0.dot(v0)
                    denominator = v_0_0 * v_1_1 - v_1_0 * v_1_0
                    if denominator==0:
                        #print("three texcoords: (%f %f) (%f %f) (%f %f)" % 
                        #      (t1_x, t1_y, t2_x, t2_y, t3_x, t3_y))
                        continue
                    u = (v_1_1 * v_2_0 - v_1_0 * v_2_1) / denominator
                    v = (v_0_0 * v_2_1 - v_1_0 * v_2_0) / denominator
                    if 0 <= u <= 1 and 0 <= v <= 1 and u + v <= 1:
                        predef_pixels_pos.append((texture_shape[0] - k, j))
                        predef_bc.append((u, v))
                        predef_v_idx.append((ver_indices[0] - 1, 
                                             ver_indices[1] - 1, ver_indices[2] - 1))
                        
        self.predef_pixels_pos = np.array(predef_pixels_pos)
        self.predef_bc = np.array(predef_bc)
        self.predef_v_idx = np.array(predef_v_idx)
        
    # vert_values should be N x C array or list
    def compute_vv(self, vert_values):
        vert_values = np.asarray(vert_values)
        if len(vert_values.shape) == 1:
            self.channel = 1
        elif len(vert_values.shape) == 2:
            self.channel = vert_values.shape[1]
        else:
            print("vert_values should be N x C array or list")
            return -1
        
        source_pixels_3D = vert_values[self.predef_v_idx[:, 0]] + \
            self.predef_bc[:, 0][:, np.newaxis] * (vert_values[self.predef_v_idx[:, 1]] - \
            vert_values[self.predef_v_idx[:, 0]]) + self.predef_bc[:, 1][:, np.newaxis] * \
            (vert_values[self.predef_v_idx[:, 2]] - vert_values[self.predef_v_idx[:, 0]])
        
        uv_values = np.zeros((self.uv_size, self.uv_size, self.channel), 
                             dtype = np.float32)
        for i in range(len(self.predef_pixels_pos)):
            if 0 <= self.predef_pixels_pos[i][0] < self.uv_size and \
               0 <= self.predef_pixels_pos[i][1] < self.uv_size:
                uv_values[self.predef_pixels_pos[i][0], 
                          self.predef_pixels_pos[i][1]] = source_pixels_3D[i]    
        return uv_values
    
    def compute_mvmap(self, gt_mesh_filename, base_mesh_filename, max_dist=0.1, smooth_norm_kernel=21):
        # read gt mesh
        gt_mesh = om.read_trimesh(gt_mesh_filename)

        # get gt_verts, gt_faces
        gt_verts = np.array(gt_mesh.points())
        gt_faces = np.array(gt_mesh.face_vertex_indices())

        # get gt_psb_mesh
        gt_mesh_psb = Mesh(v=gt_verts, f=gt_faces)
        gt_mesh_aabb = gt_mesh_psb.compute_aabb_tree()

        # read base mesh
        base_mesh = om.read_trimesh(base_mesh_filename)

        # get base_verts, base_faces
        base_verts = np.array(base_mesh.points())
        base_faces = np.array(base_mesh.face_vertex_indices())

        # get base normals
        base_mesh.request_vertex_normals()
        base_mesh.update_normals()
        base_norms = base_mesh.vertex_normals()

        # get uv_pos
        uv_pos = base_verts[self.predef_v_idx[:, 0]] + self.predef_bc[:, 0][:, np.newaxis] * \
                 (base_verts[self.predef_v_idx[:, 1]] - base_verts[self.predef_v_idx[:, 0]]) + \
                 self.predef_bc[:, 1][:, np.newaxis] * (base_verts[self.predef_v_idx[:, 2]] - \
                                                   base_verts[self.predef_v_idx[:, 0]])

        # get uv_norm
        uv_norm = base_norms[self.predef_v_idx[:, 0]] + self.predef_bc[:, 0][:, np.newaxis] * \
                  (base_norms[self.predef_v_idx[:, 1]] - base_norms[self.predef_v_idx[:, 0]]) + \
                  self.predef_bc[:, 1][:, np.newaxis] * (base_norms[self.predef_v_idx[:, 2]] - \
                                                    base_norms[self.predef_v_idx[:, 0]])
        uv_norm = uv_norm / np.linalg.norm(uv_norm, axis=1)[:, np.newaxis]
        
        # generate normal map / position and mask map
        pos_map = np.zeros((self.uv_size, self.uv_size, 3), dtype = np.float32)
        norm_map = np.zeros((self.uv_size, self.uv_size, 3), dtype = np.float32)
        mask_map = np.zeros((self.uv_size, self.uv_size, 3), dtype = np.uint8)
        for i in tqdm.trange(len(self.predef_pixels_pos)):
            if 0 <= self.predef_pixels_pos[i][0] < self.uv_size and \
               0 <= self.predef_pixels_pos[i][1] < self.uv_size:
                # assign value to normal map
                
                # assign value to position map
                pos_map[self.predef_pixels_pos[i][0], 
                        self.predef_pixels_pos[i][1]] = uv_pos[i]
                
                # assign value to normal map
                norm_map[self.predef_pixels_pos[i][0], 
                         self.predef_pixels_pos[i][1]] = uv_norm[i]

                # assign value to mask map
                mask_map[self.predef_pixels_pos[i][0], 
                         self.predef_pixels_pos[i][1]] = 1
                
        # smooth normal map if enabled
        if smooth_norm_kernel != 0:
            # smooth (use inpainting to ignore background)
            mask = (np.sum(np.abs(norm_map), 2)==0).astype(np.uint8)
            norm_map_ip = np.stack((cv2.inpaint(norm_map[:,:,0], mask, 5, cv2.INPAINT_NS), 
                                    cv2.inpaint(norm_map[:,:,1], mask, 5, cv2.INPAINT_NS), 
                                    cv2.inpaint(norm_map[:,:,2], mask, 5, cv2.INPAINT_NS)), 2)
            norm_map_sm = cv2.GaussianBlur(norm_map_ip, (smooth_norm_kernel, smooth_norm_kernel), 0)
            norm_map_sm[norm_map==0] = 0
            norm_map = norm_map_sm
            
            # update norm_uv
            for idx, pixels_pos in enumerate(self.predef_pixels_pos):
                uv_norm[idx] = norm_map[pixels_pos[0], pixels_pos[1]]
            
        # compute nearest along normal
        mv_map = np.zeros((self.uv_size, self.uv_size), dtype = np.float32)
        
        if max_dist != None:
            distances, f_idxs, pos = gt_mesh_aabb.nearest_alongnormal(uv_pos, uv_norm)
            for i in tqdm.trange(len(self.predef_pixels_pos)):
                if 0 <= self.predef_pixels_pos[i][0] < self.uv_size and \
                   0 <= self.predef_pixels_pos[i][1] < self.uv_size:
                    # assign value to movement map
                    if (distances[i] == 1e+100) or (distances[i]>max_dist):
                        mv_map[self.predef_pixels_pos[i][0], 
                               self.predef_pixels_pos[i][1]] = 0
                    elif np.dot(uv_norm[i], (pos[i] - uv_pos[i]))>0:
                        mv_map[self.predef_pixels_pos[i][0], 
                               self.predef_pixels_pos[i][1]] = distances[i]
                    else:
                        mv_map[self.predef_pixels_pos[i][0], 
                               self.predef_pixels_pos[i][1]] = -distances[i]

        return mv_map, pos_map, norm_map
