import open3d as o3d
import numpy as np
import copy
import time



# ICP関数
def draw_registration_result(source, target, transformation,vis=True):
    source_out = copy.deepcopy(source)
    source_out.transform(transformation)
    source_temp = copy.deepcopy(source)
    source_temp.transform(transformation)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    #print(type(source_temp))
    if(vis):
        o3d.visualization.draw_geometries([source_temp, target_temp])
    return source_out,target

def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def prepare_dataset(voxel_size,source,target):    
    #draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(45000000, 10000))
    return result
    
#fast grobal registration add@22.01.28
def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                    target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    print(":: Apply fast global registration with distance threshold %.3f" \
            % distance_threshold)
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result

def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size,result_ransac):
    distance_threshold = voxel_size * 0.4
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result



## 実際にICPを実行するための関数
def get_icp(sou,tar,voxel_size = 30,vis=False):
#     source, target, source_down, target_down, source_fpfh, target_fpfh = \
#             prepare_dataset(voxel_size,source_r,target_r)
    source, target, source_down, target_down, source_fpfh, target_fpfh =             prepare_dataset(voxel_size,sou,tar)

    result_ransac = execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                voxel_size)
    #print(result_ransac)
    o3d.geometry.PointCloud.estimate_normals(source, search_param = o3d.geometry.KDTreeSearchParamHybrid( radius = 0.1, max_nn = 50))
    o3d.geometry.PointCloud.estimate_normals(target, search_param = o3d.geometry.KDTreeSearchParamHybrid( radius = 0.1, max_nn = 50))

    draw_registration_result(source_down, target_down,result_ransac.transformation,vis=vis)

    result_icp = refine_registration(source, target, source_fpfh, target_fpfh,voxel_size,result_ransac)
    print(result_icp)
    #draw_registration_result(sour, targ, result_icp.transformation)
    result_source,result_target =     draw_registration_result(sou, tar, result_icp.transformation,vis=vis) #result_source/target に一致させた点群保存
    
    return result_source,result_target

#Fast grobal registrationによるICP @22.01.28
def get_icpf(sou,tar,voxel_size = 10,vis=False):
    start = time.time()
    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(voxel_size,sou,tar)
    result_fast = execute_fast_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                voxel_size)
    print("Fast global registration took %.3f sec.\n" % (time.time() - start))
    print(result_fast)
    
    draw_registration_result(source_down, target_down, result_fast.transformation,vis=vis)

    o3d.geometry.PointCloud.estimate_normals(source, search_param = o3d.geometry.KDTreeSearchParamHybrid( radius = 0.1, max_nn = 50))
    o3d.geometry.PointCloud.estimate_normals(target, search_param = o3d.geometry.KDTreeSearchParamHybrid( radius = 0.1, max_nn = 50))

    result_icp = refine_registration(source, target, source_fpfh, target_fpfh,voxel_size,result_fast)
    print(result_icp)

    result_source,result_target =  draw_registration_result(sou, tar, result_icp.transformation,vis=vis) #result_source/target に一致させた点群保存

    return result_source,result_target

#global registrationなし。元の座標がスタート @22.01.30
def get_icp_keep(source,target,voxel_size = 10,vis=False):

    if not source.has_normals():
        source.estimate_normals()
    if not target.has_normals():
        target.estimate_normals()

    trans_init=np.array([[1., 0., 0., 0.],
                        [0., 1., 0., 0.],
                        [0., 0., 1., 0.],
                        [0., 0., 0., 1.]])

    distance_threshold = voxel_size * 0.4
    result_icp = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    print(result_icp)

    result_source,result_target =  draw_registration_result(source, target, result_icp.transformation,vis=vis) #result_source/target に一致させた点群保存

    print(result_icp.transformation)

    return result_source,result_target