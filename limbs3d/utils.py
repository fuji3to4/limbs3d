import open3d as o3d
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import copy



# x,y,z座標のポリゴン作成（原点中心）
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size = 100,origin=[0, 0, 0])

def locate_reset(pcd):
    b = pcd.get_axis_aligned_bounding_box()
    min = b.get_min_bound()
    pcd_l = pcd.translate([-min[0],-min[1],-min[2]])
    #o3d.visualization.draw_geometries([pcd_l,mesh_frame])
    return pcd_l

def centerYZ(pcd,withX=False):
    pcd2=copy.deepcopy(pcd)
    data=np.array(pcd2.points)
    base_mean=[np.mean(data[:,0]),np.mean(data[:,1]),np.mean(data[:,2])]

    for i in range(len(data)):
        if(withX):
            data[i,0] = data[i,0] - base_mean[0] 
        data[i,1] = data[i,1] - base_mean[1]
        data[i,2] = data[i,2] - base_mean[2]
    pcd2.points=o3d.utility.Vector3dVector(data)

    return pcd2

# ply保存関数
def save_ply(filename,pcd):
    filename2 = filename + ".ply"
    o3d.io.write_point_cloud(filename2,pcd)
    print(filename2+"で保存")

# 左右の脚を同時にply保存する関数（separateフォルダ）
def rl_save(filename,pcd1,pcd2):
    r_filename = "./separate/"+filename + "_right.ply"
    l_filename = "./separate/"+filename + "_left.ply"
    o3d.io.write_point_cloud(r_filename,pcd1)
    o3d.io.write_point_cloud(l_filename,pcd2)
    print(r_filename+"と"+l_filename+"で保存")

# PCA関数
def get_pca(pcd):
    color = pcd.colors
    pca = PCA(n_components=3)
    X = np.asarray(pcd.points)
    pca.fit(X)
    Y = pca.transform(X)
    
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(Y)
    pcd2.colors = color   ## 色情報を戻す
    
    return pcd2

# crop関数
def get_crop(pcd,box):
    cropped = pcd.crop(box) 
    o3d.visualization.draw_geometries([cropped,mesh_frame],window_name="Crop Done")
    return cropped


# 指定したx/y/z座標の点以下を切り取る関数
def remove(pcd,x,y,z,vis=False):
    box = pcd.get_axis_aligned_bounding_box()
    box.color = (1,0,1)
    # 切り取る範囲表示（Boxの外側が除去される）
    box_c = box.translate([x,y,z])
    if(vis):
        o3d.visualization.draw_geometries([pcd,box_c,mesh_frame])
    # 切り取り実行
    pcd_c = pcd.crop(box_c)
    if(vis):
        o3d.visualization.draw_geometries([pcd_c,mesh_frame])
    print("CROP DONE\n")
    # 切り取り範囲のBoxも出力　←　Boxを使いまわせるため
    return pcd_c,box_c

# 基準点からxｍｍ以下の部分を切り取る
def ground_crop(pcd,x,vis=False):
    box = pcd.get_axis_aligned_bounding_box()
    cutp = box.max_bound
    cutp = cutp[0]
#     print(cutp)
    ref,box = remove(pcd,-cutp+x,0,0,vis=vis)
    #o3d.visualization.draw_geometries([ref,mesh_frame],window_name=str(cutp)+"でカット")
    return ref

# topとbottomの間を切り取る
def crop_between_top_and_bottom(pcd,top,bottom,vis=False):
    box = pcd.get_axis_aligned_bounding_box()
    max=box.get_max_bound()
    max[0] = top
    min=box.get_min_bound()
    min[0] = bottom
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min, max_bound=max)
    bbox.color=(1,0,1)
    pcd1=pcd.crop(bbox)
    if(vis):
        o3d.visualization.draw_geometries([pcd,mesh_frame,bbox])
        o3d.visualization.draw_geometries([pcd1,mesh_frame,bbox])
    return pcd1

# ヒストグラム描く関数
def get_hist(pcd):
    points = np.asarray(pcd.points)
    x_p = points[:,[0]]
    ## ヒストグラム書く
    hist = plt.hist(x_p, bins=600,color=["lightblue"])  
    n, bins, pathces = hist #度数,階級値,その他？

    # 階級値100未満で最大の度数の階級値を
    rem_x = np.argmax(n[0:100])

    print("Remove point is "+str(rem_x))

    pathces[np.argmax(n[0:100])].set_facecolor("red")
    plt.show()
    # 除去したいx座標を返す
    return rem_x

def compare_models(source, target):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    #print(type(source_temp))
    o3d.visualization.draw_geometries([source_temp, target_temp,mesh_frame])


