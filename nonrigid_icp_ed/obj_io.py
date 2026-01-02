
from dataclasses import dataclass, field
from typing import TypeAlias, Optional, List
from pathlib import Path

import numpy as np
import open3d as o3d


@dataclass
class ObjMtl:
    name: str = "default_mat_objio"
    Ka: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    Kd: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    Ks: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    Tr: float = 1.0
    illum: int = 2
    Ns: float = 0.0
    map_Kd: Optional[str] = None

    def __str__(self) -> str:
        lines = [f"newmtl {self.name}"]
        if len(self.Ka) == 3:
            lines.append(f"Ka {self.Ka[0]} {self.Ka[1]} {self.Ka[2]}")
        if len(self.Kd) == 3:
            lines.append(f"Kd {self.Kd[0]} {self.Kd[1]} {self.Kd[2]}")
        if len(self.Ks) == 3:
            lines.append(f"Ks {self.Ks[0]} {self.Ks[1]} {self.Ks[2]}")
        lines.append(f"Tr {self.Tr}")
        lines.append(f"illum {self.illum}")
        lines.append(f"Ns {self.Ns}")
        if self.map_Kd:
            lines.append(f"map_Kd {self.map_Kd}")
        return "\n".join(lines) + "\n"


@dataclass
class ObjMesh:
    verts: np.array = field(default_factory=np.array)
    uvs: np.array = field(default_factory=np.array)
    normals: np.array = field(default_factory=np.array)
    indices: np.array = field(default_factory=np.array)
    uv_indices: np.array = field(default_factory=np.array)
    normal_indices: np.array = field(default_factory=np.array)
    vert_colors: np.array = field(default_factory=np.array)
    mtl_path: str = field(default_factory=str)
    mtls: list[ObjMtl] = field(default_factory=[ObjMtl])
    mtl_per_faces: dict = field(default_factory=dict)

    def eunsureNumpy(self):
        for k, v in self.__dict__.items():
            if not isinstance(getattr(self, k), np.ndarray):
                if "mtl" in k:
                    continue
                elif "indices" in k:
                    setattr(self, k, np.asarray(v, dtype=np.int32))
                else:
                    setattr(self, k, np.asarray(v, dtype=float))

    def ensureList(self):
        for k, v in self.__dict__.items():
            if isinstance(getattr(self, k), np.ndarray):
                setattr(self, k, v.tolist())

    def recomputeNormals(self):
        POLY_NUM = len(self.indices[0])

        def normalize(v):
            return v / np.linalg.norm(v, axis=1, keepdims=True)

        if POLY_NUM >= 3:
            v10 = normalize(
                self.verts[self.indices[..., 1]] - self.verts[self.indices[..., 0]]
            )
            v20 = normalize(
                self.verts[self.indices[..., 2]] - self.verts[self.indices[..., 0]]
            )
            face_normals_tmp = normalize(np.cross(v10, v20, axis=1))
        else:
            raise Exception("")
        if POLY_NUM == 3:
            face_normals = face_normals_tmp
        else:
            # (FACE_NUM, POLY_NUM, 3)
            def pca(X: np.array):
                F, P, _ = X.shape
                demean = X - X.mean(axis=1, keepdims=True)
                cov = np.einsum("ikj,ikl->ijl", demean, demean)
                vals, vecs = np.linalg.eig(cov)
                min_index = np.argsort(vals, axis=1)
                vecs = vecs.transpose(0, 2, 1)
                n = np.take_along_axis(vecs, min_index[..., None], axis=1)
                return n[:, 0]

            face_verts = self.verts[self.indices]
            face_normals = pca(face_verts)
            flipped_mask = np.einsum("ij,ij->i", face_normals_tmp, face_normals) < 0
            face_normals[flipped_mask] *= -1.0

        counts = np.zeros(self.verts.shape[0], dtype=int)
        normal_sum = np.zeros_like(self.verts)
        for fid, fn in enumerate(face_normals):
            for i in range(POLY_NUM):
                vid = self.indices[fid][i]
                counts[vid] += 1
                normal_sum[vid] += fn
        invalid_mask = counts <= 0
        counts[invalid_mask] = 1
        normals = normal_sum / counts[..., None]
        normals = normalize(normals)
        self.normals = normals

        self.normal_indices = self.indices


ObjIoVec: TypeAlias = list[float] | list[list[float]] | np.ndarray
ObjIoIndices: TypeAlias = list[int] | list[list[int]] | np.ndarray


def loadMtls(mtl_path: str):
    mtls = []
    with open(mtl_path, "r") as fp:
        for line in fp:
            line = line.strip()
            if line.startswith("#"):
                continue
            splitted = line.split(" ")
            if len(splitted) < 2:
                continue
            start = splitted[0]
            data = splitted[1:]

            def to_float(data):
                return [float(x) for x in data]

            if start == "newmtl":
                mat = ObjMtl(name=data[0])
                mtls.append(mat)
            elif start == "Ka":
                mtls[-1].Ka = to_float(data)
            elif start == "Kd":
                mtls[-1].Kd = to_float(data)
            elif start == "Ks":
                mtls[-1].Ks = to_float(data)
            elif start == "Tr":
                mtls[-1].Tr = to_float(data)[0]
            elif start == "illum":
                mtls[-1].illum = int(to_float(data)[0])
            elif start == "Ns":
                mtls[-1].Ns = to_float(data)[0]
            elif start == "map_Kd":
                mtls[-1].map_Kd = data[0]
    return mtls


def loadObjSimple(obj_path: str):
    num_verts = 0
    num_uvs = 0
    num_normals = 0
    num_indices = 0
    verts = []
    uvs = []
    normals = []
    vert_colors = []
    indices = []
    uv_indices = []
    normal_indices = []
    base_dir = Path(obj_path).parent
    mtl_path = None
    mtls = []
    mtl_file_name = None
    mtl_per_faces = {}
    current_mtl_name = None
    with open(obj_path, "r") as f:
        lines = f.read()
    for line in lines.splitlines():
        vals = line.split()
        if len(vals) == 0:
            continue
        if vals[0] == "v":
            v = [float(x) for x in vals[1:4]]
            verts.append(v)
            if len(vals) == 7:
                vc = [float(x) for x in vals[4:7]]
                vert_colors.append(vc)
            num_verts += 1
        if vals[0] == "vt":
            vt = [float(x) for x in vals[1:3]]
            uvs.append(vt)
            num_uvs += 1
        if vals[0] == "vn":
            vn = [float(x) for x in vals[1:4]]
            normals.append(vn)
            num_normals += 1
        if vals[0] == "f":
            v_index = []
            uv_index = []
            n_index = []
            valid_face = False
            for f in vals[1:]:
                w = f.split("/")
                if num_verts > 0:
                    v_index.append(int(w[0]) - 1)
                if num_uvs > 0:
                    uv_index.append(int(w[1]) - 1)
                if num_normals > 0:
                    if len(w) > 2:
                        n_index.append(int(w[2]) - 1)
                    else:
                        print("no normal index")
                        n_index.append(int(w[0]) - 1)
            if len(v_index) > 0:
                indices.append(v_index)
                valid_face = True
            if len(uv_index) > 0:
                uv_indices.append(uv_index)
                valid_face = True
            if len(n_index) > 0:
                normal_indices.append(n_index)
                valid_face = True
            if valid_face:
                num_indices += 1
                if current_mtl_name is not None:
                    mtl_per_faces[current_mtl_name].append(num_indices - 1)
        if vals[0] == "mtllib":
            mtl_file_name = vals[1]
            mtl_path = str((base_dir / mtl_file_name).expanduser().absolute())
        if vals[0] == "usemtl":
            current_mtl_name = vals[1]
            if current_mtl_name not in mtl_per_faces.keys():
                mtl_per_faces[current_mtl_name] = []
    if mtl_path is not None:
        mtls = loadMtls(str(mtl_path))
    else:
        mtls = [ObjMtl()]
        mtl_per_faces[mtls[0].name] = list(range(len(indices)))
    return (
        verts,
        uvs,
        normals,
        indices,
        uv_indices,
        normal_indices,
        vert_colors,
        mtl_path,
        mtls,
        mtl_per_faces,
    )


def loadObj(obj_path: str, is_numpy: bool = True):
    mesh = ObjMesh(*loadObjSimple(obj_path))
    if is_numpy:
        mesh.eunsureNumpy()
    else:
        mesh.ensureList()
    return mesh


def saveMtl(mtl_path: str, mtls: list[ObjMtl]):
    with open(mtl_path, "w") as fp:
        for mtl in mtls:
            fp.write(str(mtl))
            fp.write("\n")


def saveObjSimple(
    obj_path: str,
    verts: ObjIoVec,
    indices: ObjIoVec,
    uvs: ObjIoVec = [],
    normals: ObjIoVec = [],
    uv_indices: ObjIoVec = [],
    normal_indices: ObjIoVec = [],
    vert_colors: ObjIoVec = [],
    mtl_path: str = "",
    mtls: list[ObjMtl] = [],
    mtl_per_faces: dict = {},
):
    f_out = open(str(obj_path), "w")
    f_out.write("####\n")
    f_out.write("#\n")
    f_out.write("# verts: %s\n" % (len(verts)))
    f_out.write("# Faces: %s\n" % (len(indices)))
    f_out.write("#\n")
    f_out.write("####\n")
    if mtl_path != "":
        mtl_file_name = Path(mtl_path).name
        f_out.write(f"mtllib {mtl_file_name}\n")
    for vi, v in enumerate(verts):
        vertstr = "v %s %s %s" % (v[0], v[1], v[2])
        if len(vert_colors) > 0:
            color = vert_colors[vi]
            vertstr += " %s %s %s" % (color[0], color[1], color[2])
        vertstr += "\n"
        f_out.write(vertstr)
    f_out.write("# %s verts\n\n" % (len(verts)))
    for uv in uvs:
        uvstr = "vt %s %s\n" % (uv[0], uv[1])
        f_out.write(uvstr)
    f_out.write("# %s uvs\n\n" % (len(uvs)))
    for n in normals:
        nStr = "vn %s %s %s\n" % (n[0], n[1], n[2])
        f_out.write(nStr)
    f_out.write("# %s normals\n\n" % (len(normals)))
    current_mtl = None
    face_mtl = [None for _ in range(len(indices))]
    for k, v in mtl_per_faces.items():
        for idx in v:
            face_mtl[idx] = k
    for fi, v_index in enumerate(indices):
        if face_mtl[fi] is not None and current_mtl != face_mtl[fi]:
            f_out.write("usemtl " + face_mtl[fi] + "\n")
            current_mtl = face_mtl[fi]
        fStr = "f"
        for fvi, v_indexi in enumerate(v_index):
            fStr += " %s" % (v_indexi + 1)
            if len(uv_indices) > 0:
                fStr += "/%s" % (uv_indices[fi][fvi] + 1)
            if len(normal_indices) > 0:
                fStr += "/%s" % (normal_indices[fi][fvi] + 1)
        fStr += "\n"
        f_out.write(fStr)
    f_out.write("# %s faces\n\n" % (len(indices)))
    f_out.write("# End of File\n")
    f_out.close()

    if mtl_path != "":
        saveMtl(mtl_path, mtls)


def saveObj(obj_path: str, mesh: ObjMesh, mtl_path: str = ""):
    if mtl_path == "":
        mtl_path = str(Path(obj_path).parent / (Path(obj_path).stem + ".mtl"))
    saveObjSimple(
        obj_path,
        mesh.verts,
        mesh.indices,
        mesh.uvs,
        mesh.normals,
        mesh.uv_indices,
        mesh.normal_indices,
        mesh.vert_colors,
        mtl_path,
        mesh.mtls,
        mesh.mtl_per_faces,
    )


def removeByVid(attrs_list: ObjIoVec, indices: ObjIoIndices, to_remove_vids: ObjIoVec):
    indices_ = []
    table = {}
    count = 0
    to_keep_vids = []
    for i in range(len(attrs_list[0])):
        if i in to_remove_vids:
            table[i] = None
            continue
        to_keep_vids.append(i)
        table[i] = count
        count += 1

    for face in indices:
        ok = True
        new_face = []
        for vid in face:
            if table[vid] is None:
                ok = False
                break
            new_face.append(table[vid])
        if not ok:
            continue
        indices_.append(new_face)

    attrs_list_ = []
    for attrs in attrs_list:
        attrs_list_.append(attrs[to_keep_vids])
    return np.asarray(attrs_list_), np.asarray(indices_)


def load_obj_as_open3d(obj_path: str):
    """
    Load an OBJ file without any processing.
    """
    # NOTE: DO NOT USE o3d.io.read_triangle_mesh() becasue it may split .obj's vertices to make unconnected triangles.
    obj_mesh = loadObj(str(obj_path))
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(obj_mesh.verts)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(obj_mesh.indices)
    return o3d_mesh
