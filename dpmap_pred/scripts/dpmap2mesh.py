from objloader import OBJ
import numpy as np
import trimesh
import os
import tqdm
from PIL import Image
import time


def subdiv(verts, tris, texcoords=None, face_index=None):
    if face_index is None:
        face_index = np.arange(len(tris))
    else:
        face_index = np.asanyarray(face_index)

    # the (c, 3) int array of vertex indices
    tris_subset = tris[face_index]

    # find the unique edges of our faces subset
    edges = np.sort(trimesh.remesh.faces_to_edges(tris_subset), axis=1)
    unique, inverse = trimesh.remesh.grouping.unique_rows(edges)
    # then only produce one midpoint per unique edge
    mid = verts[edges[unique]].mean(axis=1)
    mid_idx = inverse.reshape((-1, 3)) + len(verts)

    # the new faces_subset with correct winding
    f = np.column_stack([tris_subset[:, 0],
                         mid_idx[:, 0],
                         mid_idx[:, 2],
                         mid_idx[:, 0],
                         tris_subset[:, 1],
                         mid_idx[:, 1],
                         mid_idx[:, 2],
                         mid_idx[:, 1],
                         tris_subset[:, 2],
                         mid_idx[:, 0],
                         mid_idx[:, 1],
                         mid_idx[:, 2]]).reshape((-1, 3))
    # add the 3 new faces_subset per old face
    new_faces = np.vstack((tris, f[len(face_index):]))
    # replace the old face with a smaller face
    new_faces[face_index] = f[:len(face_index)]

    new_vertices = np.vstack((verts, mid))

    if texcoords is not None:
        texcoords_mid = texcoords[edges[unique]].mean(axis=1)
        new_texcoords = np.vstack((texcoords, texcoords_mid))
        return new_vertices, new_faces, new_texcoords

    return new_vertices, new_faces


def dpmap2verts(verts, tris, texcoords, dpmap, scale=0.914):
    dpmap = np.array(dpmap).astype(int)
    normals = np.zeros(verts.shape)
    tri_verts = verts[tris]
    n0 = np.cross(tri_verts[::, 1] - tri_verts[::, 0], tri_verts[::, 2] - tri_verts[::, 0])
    n0 = n0 / np.linalg.norm(n0, axis=1)[:, np.newaxis]
    for i in range(tris.shape[0]):
        normals[tris[i, 0]] += n0[i]
        normals[tris[i, 1]] += n0[i]
        normals[tris[i, 2]] += n0[i]
    normals = normals / np.linalg.norm(normals, axis=1)[:, np.newaxis]

    pos_u = dpmap.shape[0] - (texcoords[:, 1] * dpmap.shape[0]).astype(int)
    pos_v = (texcoords[:, 0] * dpmap.shape[1]).astype(int)
    pos_u[pos_u >= dpmap.shape[0]] = dpmap.shape[0] - 1
    pos_v[pos_v >= dpmap.shape[1]] = dpmap.shape[1] - 1
    verts += normals * (dpmap[pos_u, pos_v] - 32768)[:, np.newaxis] / 32768 * scale
    return verts


def main():
    in_dir = '/media/xyz/RED31/mvfr_released/demo/pred/reg_mesh/'
    dpmap_dir = '/media/xyz/RED31/mvfr_released/demo/pred/dp_map/'
    out_dir = '/media/xyz/RED31/mvfr_released/demo/pred/meso_mesh/'

    names = [name for name in os.listdir(in_dir) if name.split('.')[-1] == 'obj']
    for name in tqdm.tqdm(names):
        # if os.path.exists(out_dir + name):
        #     continue
        mesh = OBJ(in_dir + name)

        # start = time.time()

        tris = []
        vert_texcoords = np.zeros((len(mesh.vertices), 2))
        for face in mesh.faces:
            vertices, normals, texture_coords, material = face
            tris.append([vertices[0] - 1, vertices[1] - 1, vertices[2] - 1])
            for i in range(len(vertices)):
                vert_texcoords[vertices[i] - 1] = mesh.texcoords[texture_coords[i] - 1]
        tris = np.array(tris)

        verts = np.array(mesh.vertices)
        raw_verts = verts.copy()
        verts = verts / np.linalg.norm(verts[12280] - verts[12320]) * 60

        for _ in range(3):
            verts, tris, vert_texcoords = subdiv(verts, tris, vert_texcoords)

        dpmap = Image.open(dpmap_dir + 'texture_' + name[4:10] + '.png')
        dpmap = np.array(dpmap)[1600 - 1024:1600 + 1024, 2048 - 1024:2048 + 1024]

        verts = dpmap2verts(verts, tris, vert_texcoords, dpmap)
        verts = verts / 60 * np.linalg.norm(raw_verts[12280] - raw_verts[12320])


        with open(out_dir + name[4:10] + '.obj', 'w') as f:
            for i in range(len(verts)):
                f.write("v %.6f %.6f %.6f\n" % (verts[i][0], verts[i][1], verts[i][2]))
            for i in range(len(vert_texcoords)):
                f.write("vt %.6f %.6f\n" % (vert_texcoords[i][0], vert_texcoords[i][1]))
            for tri in tris:
                tri += 1
                f.write("f %d/%d %d/%d %d/%d\n" % (tri[0], tri[0], tri[1], tri[1], tri[2], tri[2]))


if __name__ == '__main__':
    main()
