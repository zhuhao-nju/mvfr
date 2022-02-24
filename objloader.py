# Author:Yang HT
# DATE:2018/10/26

import os.path
import numpy as np


def MTL(filename):
    contents = {}
    mtl = None
    for line in open(filename, "r"):
        if line.startswith('#'): continue
        values = line.split()
        if not values: continue
        if values[0] == 'newmtl':
            mtl = contents[values[1]] = {}
        elif mtl is None:
            raise ValueError('mtl file doesn\'t start with newmtl stmt')
        elif values[0] == 'map_Kd':
            # load the texture referred to by this declaration
            mtl[values[0]] = values[1]
        else:
            mtl[values[0]] = map(float, values[1:])
    return contents


class OBJ:
    def __init__(self, filename, swapyz=False):
        """Loads a Wavefront OBJ file. """
        self.vertices = []
        self.normals = []
        self.texcoords = []
        self.faces = []
        self.adjacent_list = []

        material = None
        for line in open(filename, "r"):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            if values[0] == 'v':
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = v[0], v[2], v[1]
                self.vertices.append(v)
            elif values[0] == 'vn':
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = v[0], v[2], v[1]
                self.normals.append(v)
            elif values[0] == 'vt':
                self.texcoords.append(list(map(float, values[1:3])))
            elif values[0] in ('usemtl', 'usemat'):
                material = values[1]
            # elif values[0] == 'mtllib':
            #    self.mtl = MTL(os.path.dirname(filename) + '/' + os.path.basename(values[1]))
            elif values[0] == 'f':
                face = []
                texcoords = []
                norms = []
                for v in values[1:]:
                    w = v.split('/')
                    face.append(int(w[0]))
                    if len(w) >= 2 and len(w[1]) > 0:
                        texcoords.append(int(w[1]))
                    else:
                        texcoords.append(0)
                    if len(w) >= 3 and len(w[2]) > 0:
                        norms.append(int(w[2]))
                    else:
                        norms.append(0)
                self.faces.append((face, norms, texcoords, material))

    def get_adjacent(self, index):
        if not self.adjacent_list:
            adjacent_list = [[] for i in range(len(self.vertices))]
            for face in self.faces:
                face_vertices, face_normals, face_texture_coords, material = face
                adjacent_list[face_vertices[0] - 1].append(face_vertices[1] - 1)
                adjacent_list[face_vertices[0] - 1].append(face_vertices[2] - 1)
                adjacent_list[face_vertices[1] - 1].append(face_vertices[0] - 1)
                adjacent_list[face_vertices[1] - 1].append(face_vertices[2] - 1)
                adjacent_list[face_vertices[2] - 1].append(face_vertices[0] - 1)
                adjacent_list[face_vertices[2] - 1].append(face_vertices[1] - 1)

            adjacent_list = list(map(set, adjacent_list))
            self.adjacent_list = list(map(list, adjacent_list))
        return self.adjacent_list[index]
