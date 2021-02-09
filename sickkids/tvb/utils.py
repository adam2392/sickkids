import os
import zipfile

import numpy as np
import scipy


def compute_closest_vertex(verts, regmap, ch_xyz):
    verts_dist_ch_xyz = scipy.spatial.distance.cdist(ch_xyz, verts)

    closest_regs = []
    for ich in range(ch_xyz.shape[0]):
        # get the closest vertex that is not an unknown brain region
        ch_dist_to_verts = verts_dist_ch_xyz[ich, :]

        # np argsort this
        sorted_idx = np.argsort(ch_dist_to_verts)
        for idx in sorted_idx:
            closest_vert = regmap[idx]
            if closest_vert != -1:
                break

        closest_regs.append(closest_vert)

    return closest_regs


def read_surf(directory: str, use_subcort: bool):
    """Read surface files from TVB dataset.

    Parameters
    ----------
    directory : str
        The path for the TVB dataset.
    use_subcort : bool
        Whether or not to use the subcortical regions.
    Returns
    -------

    """
    # Shift to account for 0 - unknown region, not included later
    reg_map_cort = (
        np.genfromtxt((os.path.join(directory, "region_mapping_cort.txt")), dtype=int)
        - 1
    )
    reg_map_subc = (
        np.genfromtxt(
            (os.path.join(directory, "region_mapping_subcort.txt")), dtype=int
        )
        - 1
    )

    with zipfile.ZipFile(os.path.join(directory, "surface_cort.zip")) as zip:
        with zip.open("vertices.txt") as fhandle:
            verts_cort = np.genfromtxt(fhandle)
        with zip.open("normals.txt") as fhandle:
            normals_cort = np.genfromtxt(fhandle)
        with zip.open("triangles.txt") as fhandle:
            triangles_cort = np.genfromtxt(fhandle, dtype=int)

    with zipfile.ZipFile(os.path.join(directory, "surface_subcort.zip")) as zip:
        with zip.open("vertices.txt") as fhandle:
            verts_subc = np.genfromtxt(fhandle)
        with zip.open("normals.txt") as fhandle:
            normals_subc = np.genfromtxt(fhandle)
        with zip.open("triangles.txt") as fhandle:
            triangles_subc = np.genfromtxt(fhandle, dtype=int)

    vert_areas_cort = compute_vertex_areas(verts_cort, triangles_cort)
    vert_areas_subc = compute_vertex_areas(verts_subc, triangles_subc)

    if not use_subcort:
        return (verts_cort, normals_cort, vert_areas_cort, reg_map_cort)
    else:
        verts = np.concatenate((verts_cort, verts_subc))
        normals = np.concatenate((normals_cort, normals_subc))
        areas = np.concatenate((vert_areas_cort, vert_areas_subc))
        regmap = np.concatenate((reg_map_cort, reg_map_subc))

        return (verts, normals, areas, regmap)


def compute_triangle_areas(vertices, triangles):
    """Calculates the area of triangles making up a surface."""
    tri_u = vertices[triangles[:, 1], :] - vertices[triangles[:, 0], :]
    tri_v = vertices[triangles[:, 2], :] - vertices[triangles[:, 0], :]
    tri_norm = np.cross(tri_u, tri_v)
    triangle_areas = np.sqrt(np.sum(tri_norm ** 2, axis=1)) / 2.0
    triangle_areas = triangle_areas[:, np.newaxis]
    return triangle_areas


def compute_vertex_areas(vertices, triangles):
    """Compute vertex areas of all the triangles."""
    triangle_areas = compute_triangle_areas(vertices, triangles)
    vertex_areas = np.zeros((vertices.shape[0]))
    for triang, vertices in enumerate(triangles):
        for i in range(3):
            vertex_areas[vertices[i]] += 1.0 / 3.0 * triangle_areas[triang]
    return vertex_areas
