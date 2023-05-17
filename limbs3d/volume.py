import open3d as o3d
import numpy as np
import pyvista as pv
import pymeshfix as mf
import os
import sys


# Surface reconstruction関数(サンプルの通り。意味なかったかもですが、、Depthだけいじってます。。。)
def get_surface(pcd,depth=10,vis=False,auto_normals=True):
    
    if(auto_normals):
        pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))  # invalidate existing normals
        pcd.estimate_normals()
        # 法線の確認時にON
        if(vis):
            o3d.visualization.draw_geometries([pcd], point_show_normal=True)
        pcd.orient_normals_consistent_tangent_plane(100)
        if(vis):
            o3d.visualization.draw_geometries([pcd], point_show_normal=True)
    
    with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth)
    #print(mesh)
    
    vertices_to_remove = densities < np.quantile(densities, 0.01)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    print(mesh)
    if(vis):
        o3d.visualization.draw_geometries([mesh])
    return mesh

# 穴埋め関数 by pyvista
def get_fix(mesh,vis=False):

    if isinstance(mesh,o3d.cpu.pybind.geometry.TriangleMesh):
        o3d.io.write_triangle_mesh("tmp.ply",mesh)
        mesh=pv.read("tmp.ply")
        os.remove("tmp.ply")


    if not isinstance(mesh,pv.core.pointset.PolyData):
        print("Wrong file format")
        sys.exit(1)

    fix = mf.MeshFix(mesh)
    holes = fix.extract_holes()

    fix.repair(verbose=True)

    if vis:
        fix.mesh.plot()
    
    return fix.mesh
    
# 体積算出関数
def get_vol(mesh):
    if not isinstance(mesh,pv.core.pointset.PolyData):
        print("Wrong file format")
        sys.exit(1)
    
    volume = mesh.volume
    # [cm^3] にするために 0.001をかける
    #vol  = volume*0.001
    
    return volume