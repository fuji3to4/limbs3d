import open3d as o3d
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import copy
import math
import scipy
from ellipse import LsqEllipse

from limbs3d.utils import get_hist, mesh_frame, get_pca, crop_between_top_and_bottom,centerYZ


def separate2legs(pcd,vis=False,floor=True):

    #主成分分析
    pca_pcd = get_pca(pcd)

    #床に合わせる
    if floor:
        pca_pcd=align2Floor(pca_pcd)

    points = np.asarray(pca_pcd.points)
    #足の向きが逆になっているものを180度回転 @22.01.27
    #尖度を利用
    #床が邪魔をすることがあるので、大きく(20mm)床を削除して片足のみにしてから尖度を求める@22.10.31
    pcd1=deleteFloor(pca_pcd,vis=vis,distance_threshold=5)
    pcd2=pcd1.select_by_index(np.where(np.asarray(pcd1.points)[:,0] >=20)[0])
    pcd3=getMainClusterDBSCAN(pcd2,eps=5, min_points=5, print_progress=True,vis=vis)
    points_nofloor = np.asarray(pcd3.points)
    if(scipy.stats.skew(points_nofloor[:,2]) < 0):
        R=pca_pcd.get_rotation_matrix_from_xyz([np.pi,0,0])
        pca_pcd.rotate(R, center=(0, 0, 0))

    if vis:    
        o3d.visualization.draw_geometries([pca_pcd,mesh_frame],window_name="PCA check")

    y_p = points[:,1]
    ## ヒストグラム書く
    if vis: 
        hist = plt.hist(y_p, bins=600,color=["lightblue"])  
        n, bins, pathces = hist

        rem_n = np.argmin(n[200:400]) #ざっと中心部分で最小値
        rem_y = bins[rem_n+200]
        #pathces[rem_n+200].set_facecolor("red")
        plt.vlines(rem_y,0,np.max(n),"red")
        plt.show()
    else:
        n, bins = np.histogram(y_p, bins=600)
        rem_n = np.argmin(n[200:400]) 
        rem_y = bins[rem_n+200]


    left_leg=pca_pcd.select_by_index(np.where(y_p <= rem_y)[0])
    right_leg=pca_pcd.select_by_index(np.where(y_p > rem_y)[0])
    
    if vis:    
        o3d.visualization.draw_geometries([left_leg,mesh_frame],window_name="left_leg")
        o3d.visualization.draw_geometries([right_leg,mesh_frame],window_name="right_leg")

    return left_leg,right_leg

## Floor detection & alignment@22.01.27
def detectFloor(pcd,distance_threshold=3,vis=False):
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                            ransac_n=3,
                                            num_iterations=1000)
    [a, b, c, d] = plane_model
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    if vis:
        inlier_cloud = pcd.select_by_index(inliers)
        inlier_cloud.paint_uniform_color([1.0, 0, 0])
        outlier_cloud = pcd.select_by_index(inliers, invert=True)
        o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud,mesh_frame],
                                            zoom=0.8,
                                            front=[-0.4999, -0.1659, -0.8499],
                                            lookat=[2.1813, 2.0619, 2.0999],
                                            up=[0.1204, -0.9852, 0.1215])

    return plane_model,inliers


def vector_angle(u, v):
    return np.arccos(np.dot(u,v) / (np.linalg.norm(u)* np.linalg.norm(v)))

def floor2YZplane(pcd,plane_model):

    _pcd =copy.deepcopy(pcd)

    [a, b, c, d] = plane_model
    _pcd.translate((d/a,0,0))

    # Calculate rotation angle between plane normal & x-axis
    x_axis = (1,0,0)
    rotation_angle = vector_angle(plane_model[:3], x_axis)

    # Calculate rotation axis
    plane_normal_length = math.sqrt(b**2 + c**2)
    u1 = c / plane_normal_length
    u2 = -b / plane_normal_length
    rotation_axis = (0, u1, u2)

    # Generate axis-angle representation
    #optimization_factor = 1
    #axis_angle = tuple([x * rotation_angle * optimization_factor for x in rotation_axis])
    axis_angle = np.array(rotation_axis) * rotation_angle

    # Rotate point cloud
    R = _pcd.get_rotation_matrix_from_axis_angle(axis_angle)
    _pcd.rotate(R, center=(0,0,0))

    return _pcd

def align2Floor(pcd,vis=False,distance_threshold=5):
    plane_model, inliers =detectFloor(pcd,vis=vis,distance_threshold=distance_threshold)
    pcd2=floor2YZplane(pcd,plane_model)

    if vis:
        o3d.visualization.draw_geometries([pcd,mesh_frame],window_name="Input")
        get_hist(pcd)
        o3d.visualization.draw_geometries([pcd2,mesh_frame],window_name="align2Floor")
        get_hist(pcd2)

    return pcd2

def deleteFloor(pcd,distance_threshold=5,vis=False):
    plane_model, inliers =detectFloor(pcd,vis=vis,distance_threshold=distance_threshold)
    pcd = pcd.select_by_index(inliers, invert=True)
    
    if vis:
        o3d.visualization.draw_geometries([pcd,mesh_frame],window_name="deleteFloor")

    return pcd

#Noise remove(get main cluster by DBSCAN)
def getMainClusterDBSCAN(pcd,eps=5, min_points=10, print_progress=True,vis=False):
    pcd_tmp=copy.deepcopy(pcd)
    labels = np.array(pcd_tmp.cluster_dbscan(eps=eps, min_points=min_points, print_progress=print_progress))
    u, counts =np.unique(labels,return_counts=True)
    pcd_out=pcd_tmp.select_by_index(np.where(labels==u[np.argmax(counts)])[0])

    
    if vis:
        max_label = labels.max()
        print(f"point cloud has {max_label + 1} clusters")
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        pcd_tmp.colors = o3d.utility.Vector3dVector(colors[:, :3])
        o3d.visualization.draw_geometries([pcd_tmp])
    
    return pcd_out


#remove hidden points inside leg 
def removeHiddenPoints(pcd,dx=100,xmargin=50,dth=0.25,vis=False):
    pcd_tmp=copy.deepcopy(pcd)
    #o3d.visualization.draw_geometries([pcd,limbs3d.mesh_frame])
    crd=np.asarray(pcd_tmp.points)

    #Centering YZ coordinate
    crd[:,1]=crd[:,1]-crd[:,1].mean()
    crd[:,2]=crd[:,2]-crd[:,2].mean()
    pcd_tmp.points= o3d.utility.Vector3dVector(crd)
    #o3d.visualization.draw_geometries([pcd,limbs3d.mesh_frame])

    #Define camera view point
    x_len = crd[:,0].max()
    y_len=crd[:,1].max()-crd[:,1].min()
    z_len=crd[:,2].max()-crd[:,2].min()
    len = y_len if y_len > z_len else z_len
    #dth=0.5
    preset=set()
    for xcame in np.arange(xmargin,x_len,dx):
        for th in np.arange(0,2,dth):
            #th=1.5
            #Define camera view point
            camera = [xcame, np.cos(np.pi*th)*len, np.sin(np.pi*th)*len]
            radius = x_len * 100

            _, pt_map = pcd_tmp.hidden_point_removal(camera, radius)

            preset=preset|set(pt_map)
            if(vis):
                pcdvis = pcd_tmp.select_by_index(list(preset))
                o3d.visualization.draw_geometries([pcdvis,mesh_frame])

    pcd2 = pcd_tmp.select_by_index(list(preset))

    return pcd2




#円柱座標系変換
def transCylinCoord(data,deg=True):
    radi=np.sqrt(data[:,1]**2 + data[:,2]**2)
    cosi=data[:,1]/radi
    rad=np.arccos(cosi)
    rad[data[:,2]<0]=-rad[data[:,2]<0]
    if(deg):
        rad=np.rad2deg(rad)+180

    return np.array([data[:,0],radi, rad])


#高さと円周角のGridにおける半径の平均値の計算
#return grid平均値, gridに含まれるpointcloudのID
def gridRadiusMean(high_array,radius_array,degree_array,maxh,dh=5,dt=5):
    #全範囲における半径の平均を求める

    #aveに初めから0を入れておく
    ave=np.zeros((int(maxh/dh),int(360/dt)))
    ave_ids= np.array([[None for j in range(int(360/dt))] for i in range(int(maxh/dh))])

    #ave=np.matrix()
    i=0
    for h in range(0,maxh,dh):
        j=0
        for t in range(0,360,dt):
            id=(high_array>=h) & (high_array<h+dh) & (degree_array>=t) & (degree_array<t+dt)
            #idにTrueとFalseが入っている
            
            #trueのradiusを出してくれるはず
            r_id=radius_array[id]
            #print(sum(id),h,t)
            ave_ids[i][j]=np.where(id)
            if len(r_id) != 0:
                #r_aveにr_idの平均値を入れる
                r_ave=r_id.mean()
                #aveに平均値を配列としていれる
                ave[i][j]=r_ave
            
            else:
                ave[i][j]=0
            j=j+1
        
        i=i+1

    return {'ave':ave,'ave_ids':ave_ids}
        

#pointcloudに色付け
def colorMap2Pcd(pcd,Z,ave_ids,cmap='jet',vmin=None,vmax=None,vis=True):

    pcd_temp=copy.deepcopy(pcd)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    colors = cm.get_cmap(cmap)(norm(Z))
    colors = colors[:,:,0:3]

    pcd_colors=np.asarray(pcd_temp.colors)

    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            for id in ave_ids[i][j]:
                pcd_colors[id]=colors[i][j]

    pcd_temp.colors= o3d.utility.Vector3dVector(pcd_colors)

    if vis:
        o3d.visualization.draw_geometries([pcd_temp])

    return pcd_temp

def perimeter_from_ellipse_fitting(pcd, length,window=1,vis=False):

    #pcd2=pcd.select_by_index(np.intersect1d(np.where(np.asarray(pcd.points)[:,0] <=length+window/2)[0],np.where(np.asarray(pcd.points)[:,0] >=length-window/2)[0]))
    pcd2=crop_between_top_and_bottom(pcd,top=length+window/2,bottom=length-window/2)
    pcd3=centerYZ(pcd2)

    data=np.array(pcd3.points)
    P = np.array([data[:,1], data[:,2]]).T
    reg = LsqEllipse().fit(P)
    center, width, height, phi = reg.as_parameters()

    if width > height:
        a=width
        b=height
    else:
        a=height
        b=width

    
    dt=0.00001
    perimeter=0
    for t in np.arange(0,0.5+dt,dt):
        perimeter += 4*a*np.sqrt(1-(1-b**2/a**2)*np.sin(t*np.pi) ** 2) * dt*np.pi

    if(vis):

        print(f'center: {center[0]:.3f}, {center[1]:.3f}')
        print(f'width: {width:.3f}')
        print(f'height: {height:.3f}')
        print(f'phi: {phi:.3f}')

        fig = plt.figure(figsize=(6, 6))
        ax = plt.subplot()
        ax.axis('equal')
        #ax.plot(X1, X2, 'ro', zorder=1)
        ax.scatter(P[:,0], P[:,1], label='measure data',s=2)
        ellipse = Ellipse(
            xy=center, width=2*width, height=2*height, angle=np.rad2deg(phi),
            edgecolor='r', fc='None', lw=1, label='Fit', zorder=2
        )
        ax.add_patch(ellipse)

        plt.xlabel('$X_1$')
        plt.ylabel('$X_2$')

        plt.legend()
        plt.show()
    
    return perimeter

def perimeter_by_savgol(pcd, length,window=5,graph=False):
    pcd2=crop_between_top_and_bottom(pcd,top=length+window/2,bottom=length-window/2)
    if(np.asarray(pcd2.points).shape[0] <= 5):
        return 0
    pcd3=centerYZ(pcd2)

    cylin=transCylinCoord(np.array(pcd3.points),deg=False)
    cylin=cylin.T
    cylin=cylin[np.argsort(cylin[:,2])]
    cylin2=copy.deepcopy(cylin)
    cylin2[:,2]=cylin2[:,2]+2*np.pi
    cylin_double=np.vstack([cylin,cylin2])

    X=cylin_double[:,2]
    Y=cylin_double[:,1]


    sv=scipy.signal.savgol_filter(Y,int(Y.size/10)+1,5,deriv=0)

    xx=sv*np.cos(X)
    yy=sv*np.sin(X)
    xx=xx[np.where(X >= 0)[0][0]:np.where(X >= 2*np.pi)[0][1]]
    yy=yy[np.where(X >= 0)[0][0]:np.where(X >= 2*np.pi)[0][1]]
    peri=0
    for i in range(0,xx.size-1):
        peri+=np.sqrt((xx[i+1]-xx[i])**2 + (yy[i+1]-yy[i])**2)



    if(graph):
        data=np.array(pcd3.points)

        fig = plt.figure(figsize=(6, 6))
        ax = plt.subplot()
        ax.axis('equal')
        #ax.plot(X1, X2, 'ro', zorder=1)
        ax.scatter(data[:,1], data[:,2], label='measure data',s=2)
        ax.plot(xx, yy, color="orange",label='Fit')
        plt.xlabel('$X_1$')
        plt.ylabel('$X_2$')

        plt.legend()
        plt.show()

    return peri


def window_average(X,Y,w=0.2*np.pi,dx=0.01*np.pi,ini=0,end=2*np.pi):

    xmean=np.arange(ini,end,dx)
    ymean=[]
    for i in xmean:
        bool_x=(X >= i-w/2) & (X < i+w/2)
        if(bool_x.sum() != 0):
            premean=Y[bool_x].mean()
            ymean.append(premean)
        else:
            ymean.append(premean)
    ymean=np.asarray(ymean)

    return [xmean,ymean]


def perimeter_by_winave(pcd, length,window=5,graph=False,w=0.2*np.pi,dx=0.01*np.pi):
    pcd2=crop_between_top_and_bottom(pcd,top=length+window/2,bottom=length-window/2)
    pcd3=centerYZ(pcd2)

    cylin=transCylinCoord(np.array(pcd3.points),deg=False)
    cylin=cylin.T
    cylin=cylin[np.argsort(cylin[:,2])]
    cylin2=copy.deepcopy(cylin)
    cylin2[:,2]=cylin2[:,2]+2*np.pi
    cylin_double=np.vstack([cylin,cylin2])

    X=cylin_double[:,2]
    Y=cylin_double[:,1]
    
    winave=window_average(cylin_double[:,2],cylin_double[:,1],w=w,dx=dx)

    xx=winave[1]*np.cos(winave[0])
    yy=winave[1]*np.sin(winave[0])
    xx=np.append(xx,xx[0])
    yy=np.append(yy,yy[0])
    peri=0
    for i in range(0,xx.size-1):
        peri+=np.sqrt((xx[i+1]-xx[i])**2 + (yy[i+1]-yy[i])**2)

    #peri+=np.sqrt((xx[0]-xx[xx.size-1])**2 + (yy[0]-yy[xx.size-1])**2)

    if(graph):
        data=np.array(pcd3.points)

        fig = plt.figure(figsize=(6, 6))
        ax = plt.subplot()
        ax.axis('equal')
        #ax.plot(X1, X2, 'ro', zorder=1)
        ax.scatter(data[:,1], data[:,2], label='measure data',marker=".",s=0.1)
        ax.plot(xx, yy, color="orange",label='Fit')
        plt.xlabel('$X_1$')
        plt.ylabel('$X_2$')

        plt.legend()
        plt.show()

    return peri
    
    
    

