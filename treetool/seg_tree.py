"""
MIT License - Copyright (c) 2021 porteratzo
このファイルは、Open3Dを使用するように全面的に書き換えられました。
"""
import open3d as o3d
import numpy as np

def floor_remove_o3d(pcd, distance_threshold=0.2, ransac_n=3, num_iterations=1000):
    """
    Open3DのRANSACを用いて点群から地面を分離する。

    Args:
        pcd (open3d.geometry.PointCloud): 入力点群
        distance_threshold (float): RANSACで平面とみなすための点と平面の最大距離
        ransac_n (int): 平面を推定するために使用する点の数
        num_iterations (int): RANSACの反復回数

    Returns:
        tuple: (非地面点群, 地面点群) のタプル (いずれも open3d.geometry.PointCloud)
    """
    plane_model, inlier_indices = pcd.segment_plane(distance_threshold=distance_threshold,
                                                    ransac_n=ransac_n,
                                                    num_iterations=num_iterations)
    
    ground_cloud = pcd.select_by_index(inlier_indices)
    non_ground_cloud = pcd.select_by_index(inlier_indices, invert=True)
    
    return non_ground_cloud, ground_cloud

def estimate_normals_o3d(pcd, search_radius=0.2, max_nn=30):
    """
    Open3Dを用いて点群の法線を推定する。
    """
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=search_radius, max_nn=max_nn))
    pcd.orient_normals_to_align_with_direction(orientation_reference=np.array([0., 0., 1.]))
    
def segment_cylinder_o3d(pcd, distance_threshold=0.1, ransac_n=3, num_iterations=100, min_radius=0.05, max_radius=0.5):
    """
    RANSACを用いて点群から円筒をセグメンテーションする。
    Open3D 0.17.0にはこの機能がないため、ここではプレースホルダーとして機能します。
    実際の応用では、この部分をカスタムRANSAC等で実装する必要があります。
    """
    # この関数は、現在のOpen3Dのバージョンではダミーとして機能します。
    # 代わりに、最大の平面を検出してそれを幹の代表と見なすことで、処理を続行させます。
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                         ransac_n=ransac_n,
                                         num_iterations=num_iterations)
    
    if len(inliers) > 10:
        inlier_cloud = pcd.select_by_index(inliers)
        center = inlier_cloud.get_center()
        
        # 簡易的に半径を計算（正確ではない）
        distances = np.linalg.norm(np.asarray(inlier_cloud.points)[:, :2] - center[:2], axis=1)
        radius = np.mean(distances)

        # PCLの7パラメータ形式 [center_x, y, z, axis_x, y, z, radius] に合わせる
        # ここでは垂直な円筒を仮定
        model_coeffs = [center[0], center[1], center[2], 0, 0, 1, radius]
        return inliers, model_coeffs
        
    return [], []