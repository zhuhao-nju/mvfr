import cv2, tqdm, torch
import numpy as np
import openmesh as om
import matplotlib.pyplot as plt
#from psbody.mesh import Mesh

def reshape_multiview_tensors(images_tensor, cams_tensor):
    # Careful here! Because we put single view and multiview together,
    # the returned tensor.shape is 5-dim: [B, num_views, C, W, H]
    # So we need to convert it back to 4-dim [B*num_views, C, W, H]
    # Don't worry PIFuNet will handle multi-view cases
    # [BV, C, H, W]
    images_tensor = images_tensor.view(
        images_tensor.shape[0] * images_tensor.shape[1],
        images_tensor.shape[2],
        images_tensor.shape[3],
        images_tensor.shape[4]
    )
    # [BV, 2, 4, 4]
    cams_tensor = cams_tensor.view(
        cams_tensor.shape[0] * cams_tensor.shape[1],
        cams_tensor.shape[2],
        cams_tensor.shape[3],
        cams_tensor.shape[4]
    )

    return images_tensor, cams_tensor

def read_npy(path):

    npy = np.load(path)

    return npy

def read_camera(K, Rt):
    
    cam = np.zeros((2, 4, 4))
    # extrinsic
    cam[0, 0:3, 0:4] = Rt
    cam[0, 3, 3] = 1
    # intrinsic 
    cam[1, 0:3, 0:3] = K
    
    return cam

def read_image(path):
    
    image = cv2.imread(path)
    
    return image

def resize_camera(cam, scale):
    
    # intrinsic
    new_cam = np.copy(cam)
    new_cam[1,0:2,:] = cam[1,0:2,:] * scale
    
    return new_cam

def resize_image(image, scale, interpolation="linear"):
    
    if interpolation == 'linear':
        return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    if interpolation == 'nearest':
        return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    
def transpose_camera(cam):
    
    new_cam = np.copy(cam)
    # x, y exchange
    new_cam[1,0,:] = cam[1,1,:]
    new_cam[1,1,:] = cam[1,0,:]
    
    return new_cam

def transpose_image(image):
    
    new_image = cv2.transpose(image)
    
    return new_image

def view_loss(loss_list, view_interval=0, mod="sqrt"):
    if view_interval == 0:
        view_interval = len(loss_list)
    
    loss_list = np.array(loss_list)
    #
    length = loss_list.shape[0]
    interval = length // view_interval
    length_new = view_interval
    loss_list = np.reshape(loss_list[:length_new * interval], (length_new, interval))
    print(loss_list.shape)
    loss_list = np.mean(loss_list, axis=1)
    
    if mod == "sqrt":
        loss_list = np.sqrt(loss_list)
    elif mod == "none":
        loss_list = loss_list
    else:
        raise Exception("Wrong mod")
    
    plt.plot(loss_list)
    
    return

def save_npy(path, arr):

    np.save(path, arr)

    return 

def save_mesh(path, verts, faces):
    with open(path, 'w') as file:
        for v in verts:
            file.write('v %.6f %.6f %.6f\n' % (v[0], v[1], v[2]))
        for f in faces:
            f_plus = f + 1
            file.write('f %d %d %d\n' % (f_plus[0], f_plus[2], f_plus[1]))
    print("save mesh to %s"%(path))

    return

def save_sampled_points(path, points, prob):
    """
    Save the visualization of sampling to a ply file.
    Red points represent positive predictions.
    Green points represent negative predictions.
    :param path: path to save
    :param points: [3, N] array of points
    :param prob: [N] array of predictions in the range [0~1]
    """
    # [N]
    r = (prob > 0.5) * 255
    g = (prob < 0.5) * 255
    b = np.zeros(prob.shape)
    
    # [N, 3]
    to_save = np.concatenate([points, r, g, b], axis=0).T
    
    return np.savetxt(path,
                      to_save,
                      fmt='%.6f %.6f %.6f %d %d %d',
                      comments='',
                      header=(
                          'ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header').format(
                          points.shape[1])
                      )

def make_trimesh(verts, faces, texture=None, compute_vn = False):
    # if vertex index starts with 1, make it start with 0
    if np.min(faces) == 1:
        faces = np.array(faces)
        faces = faces - 1
    
    # make a mesh
    mesh = om.TriMesh()

    # transfer verts and faces
    for i in range(len(verts)):
        mesh.add_vertex(verts[i])
        if texture:
            #mesh.set_texcoord2D(mesh.vertex_handle(verts[i]), texture[i])
            mesh.set_texcoord2D(mesh.vertex_handle(i), texture[i])
    for i in range(len(faces)):
        a = mesh.vertex_handle(faces[i][0])
        b = mesh.vertex_handle(faces[i][1])
        c = mesh.vertex_handle(faces[i][2])
        mesh.add_face(a,b,c)

    # compute vert_norms
    if compute_vn is True:
        mesh.request_vertex_normals()
        mesh.update_normals()

    return mesh


def uv2mesh(uv_map, vt=False, uv_norm_map=None, mv_map=None):
    h = uv_map.shape[0]
    w = uv_map.shape[1]
    verts, faces = [], []
    texture = []
    
    # make verts list
    vert_idx_map = np.zeros((h, w), dtype = np.int)
    
    count = 0
    for i in range(h):
        for j in range(w):
            if np.all(np.equal(uv_map[i, j], np.array([0, 0, 0]))):
                vert_idx_map[i, j] = -1
            else:
                verts.append(uv_map[i, j])
                texture.append(np.array( [j / w, 1 - i / h] ))
                vert_idx_map[i, j] = count
                count += 1
    
    # make face list
    for i in tqdm.trange(1, h-1):
        for j in range(1, w-1):
            if mv_map is None:
                if np.all(np.equal(uv_map[i, j], np.array([0, 0, 0]))):
                    continue
                if np.all(np.equal(uv_map[i+1, j], np.array([0, 0, 0]))):
                    continue
                if np.all(np.equal(uv_map[i, j+1], np.array([0, 0, 0]))):
                    continue
                if np.all(np.equal(uv_map[i+1, j+1], np.array([0, 0, 0]))):
                    continue
            else:
                if  mv_map[i, j]==0:
                    continue
                if  mv_map[i+1, j]==0:
                    continue
                if  mv_map[i, j+1]==0:
                    continue
                if  mv_map[i+1, j+1]==0:
                    continue

            if uv_norm_map is None:
                faces.append([vert_idx_map[i,j], vert_idx_map[i+1, j], vert_idx_map[i, j+1]])
                faces.append([vert_idx_map[i+1, j+1], vert_idx_map[i, j+1], vert_idx_map[i+1, j]])
            else:
                # face 1
                face_dir = np.cross(uv_map[i, j] - uv_map[i, j+1], 
                                    uv_map[i, j+1] - uv_map[i+1, j])
                if face_dir.dot(uv_norm_map[i, j])>0:
                    faces.append([vert_idx_map[i,j], vert_idx_map[i+1, j], vert_idx_map[i, j+1]])
                else:
                    faces.append([vert_idx_map[i,j], vert_idx_map[i, j+1], vert_idx_map[i+1, j]])
                
                # face 2
                face_dir = np.cross(uv_map[i+1, j+1] - uv_map[i+1, j], 
                                    uv_map[i+1, j] - uv_map[i, j+1])
                if face_dir.dot(uv_norm_map[i, j])>0:
                    faces.append([vert_idx_map[i+1, j+1], vert_idx_map[i, j+1], vert_idx_map[i+1, j]])
                else:
                    faces.append([vert_idx_map[i+1, j+1], vert_idx_map[i+1, j], vert_idx_map[i, j+1]])
                    
    # make openmesh mesh
    if vt:
        mesh = make_trimesh(verts, faces, texture)
    else:
        mesh = make_trimesh(verts, faces)
    return mesh

def compute_chamfer(mesh_A, mesh_B):
    
    # B to A
    mesh_A_aabb = Mesh(v=mesh_A.vertices, f=mesh_A.faces).compute_aabb_tree()
    _, closests_B2A = mesh_A_aabb.nearest(mesh_B.vertices)
    errors_B2A = np.linalg.norm(mesh_B.vertices - closests_B2A, axis=1)
    
    # A to B
    mesh_B_aabb = Mesh(v=mesh_B.vertices, f=mesh_B.faces).compute_aabb_tree()
    _, closests_A2B = mesh_B_aabb.nearest(mesh_A.vertices)
    errors_A2B = np.linalg.norm(mesh_A.vertices - closests_A2B, axis=1)
    
    errors_all = np.concatenate((errors_A2B, errors_B2A))
    chamfer_dist = np.mean(errors_all[~np.isnan(errors_all)])

    return chamfer_dist

