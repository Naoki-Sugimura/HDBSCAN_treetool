import pclpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import open3d as o3d
import treetool.seg_tree as seg_tree
import treetool.utils as utils
import treetool.tree_tool as tree_tool
from mpl_toolkits.mplot3d import Axes3D
import pyvista as pv

# 点群データの読み込み
def load_point_cloud(file_path):
    PointCloud = pclpy.pcl.PointCloud.PointXYZ()
    pclpy.pcl.io.loadPCDFile(file_path, PointCloud)
    return PointCloud

# 点群をボクセル化する関数
def voxelize_point_cloud(PointCloud, voxel_size):
    voxelized = seg_tree.voxelize(PointCloud.xyz, voxel_size)
    
    # voxelizedがnumpy配列として返されることを期待する
    print(f"Voxelized shape: {voxelized.shape}")
    return voxelized

# pclpyのPointCloudをNumPy配列に変換する関数
def pcl_to_numpy(pcl_point_cloud):
    # 点群データがすでにNumPy配列であればそのまま返す
    if isinstance(pcl_point_cloud, np.ndarray):
        return pcl_point_cloud
    # pclpyのPointCloudからNumPy配列を取得
    np_points = np.array(pcl_point_cloud.xyz)
    
    if np_points.shape[1] != 3:
        raise ValueError(f"Expected point cloud data with shape (N, 3), but got shape {np_points.shape}")
    
    return np_points

# pclpyのPointCloudをOpen3DのPointCloudに変換する関数
def pcl_to_o3d(np_points):
    # Open3DのPointCloudに変換
    o3d_points = o3d.geometry.PointCloud()
    o3d_points.points = o3d.utility.Vector3dVector(np_points)  # Open3DのPointCloudに変換
    return o3d_points


# 点群可視化の関数（Open3D）
def visualize_point_cloud_o3d(PointCloud_list, voxel_size, tree_ids=None):
    """
    可視化関数：点群をOpen3Dで表示し、tree_idsが指定されていればIDも表示する
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    for i, PointCloud in enumerate(PointCloud_list):
        np_points = pcl_to_numpy(PointCloud)  # pclpyのPointCloudをNumPy配列に変換
        
        # Open3Dに変換
        o3d_point_cloud = pcl_to_o3d(np_points)
        vis.add_geometry(o3d_point_cloud)

        # tree_idsが存在する場合、IDを表示
        if tree_ids is not None:
            tree_id = tree_ids[i]  # ID取得
            tree_center = np.mean(np_points, axis=0)  # 木の中心を計算

            # 木の中心に小さな球体を表示
            label = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)  # 小さな球
            label.paint_uniform_color([1.0, 0.0, 0.0])  # 赤色
            label.translate(tree_center)  # 球体を木の中心に配置

            vis.add_geometry(label)

    vis.run()
    vis.destroy_window()

def debug_visualize_point_cloud_o3d(PointCloud, tree_centers=None, tree_ids=None):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    np_points = pcl_to_numpy(PointCloud)
    o3d_point_cloud = pcl_to_o3d(np_points)
    vis.add_geometry(o3d_point_cloud)
    
    if tree_centers and tree_ids:
        for center, tree_id in zip(tree_centers, tree_ids):
            label = o3d.geometry.TriangleMesh.create_sphere(radius=0.3)
            label.paint_uniform_color([1, 0, 0])
            label.translate(center)
            vis.add_geometry(label)
    
    vis.run()
    vis.destroy_window()

def visualize_stem_groups_open3d(stem_groups, PointCloud):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    np_points = pcl_to_numpy(PointCloud)
    o3d_point_cloud = pcl_to_o3d(np_points)
    vis.add_geometry(o3d_point_cloud)
    
    geometries = []

    for group in stem_groups:
        cloud_np = group["cloud"]
        center = group["center"]

        # 点群
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud_np)
        pcd.paint_uniform_color([0, 1, 0])
        geometries.append(pcd)

        # 重心（赤い球）
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        sphere.translate(center)
        sphere.paint_uniform_color([1.0, 0, 0])
        geometries.append(sphere)
        vis.add_geometry(sphere)

    vis.run()
    vis.destroy_window()
    # o3d.visualization.draw_geometries(geometries)

def visualize_detected_trees_open3d(detected_trees, PointCloud):
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # 点群の変換・表示
    np_points = pcl_to_numpy(PointCloud)
    o3d_point_cloud = pcl_to_o3d(np_points)
    # o3d_point_cloud.paint_uniform_color([0.5, 0.5, 0.5])  # グレー
    vis.add_geometry(o3d_point_cloud)

    # 幹の円柱と重心（球）を追加
    for (xc, yc, r) in detected_trees:
        # 赤い円柱で幹を表現
        height = 0.1  # 仮の幹高さ
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=r, height=height)
        cylinder.translate([xc, yc, 0.0])
        cylinder.paint_uniform_color([1.0, 0.0, 0.0])
        vis.add_geometry(cylinder)

        # 青い球で幹の中心（1.2mの高さ）を可視化
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        sphere.translate([xc, yc, 1.2])
        sphere.paint_uniform_color([0.0, 0.0, 1.0])
        vis.add_geometry(sphere)

    vis.run()
    vis.destroy_window()

def visualize_slice_and_circles(slice_points, detected_circles, z_height=2.5, point_color=[0.2, 0.8, 0.2]):

    # 入力 shape に応じて z を追加
    if slice_points.shape[1] == 2:
        points_3d = np.hstack([slice_points, np.full((slice_points.shape[0], 1), z_height)])
    elif slice_points.shape[1] == 3:
        points_3d = slice_points.copy()
    else:
        raise ValueError("slice_points must have shape (N, 2) or (N, 3)")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)

    colors = np.tile(np.array([point_color]), (len(points_3d), 1))
    pcd.colors = o3d.utility.Vector3dVector(colors)

    geometries = [pcd]

    for (xc, yc, r) in detected_circles:
        theta = np.linspace(0, 2 * np.pi, 100)
        circle_points = np.array([[xc + r * np.cos(t), yc + r * np.sin(t), z_height] for t in theta])
        lines = [[i, i + 1] for i in range(len(circle_points) - 1)]
        lines.append([len(circle_points) - 1, 0])
        circle = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(circle_points),
            lines=o3d.utility.Vector2iVector(lines)
        )
        circle.paint_uniform_color([1.0, 0.0, 0.0])
        geometries.append(circle)

    o3d.visualization.draw_geometries(geometries)


def visualize_trees_with_pyvista(trees, tree_ids, diameters, heights):
# def visualize_trees_with_pyvista(trees, tree_ids, diameters):
    """
    Trees を可視化し、各木の中心近くに ID, DBH, 座標を表示する。
    
    Args:
        trees: List[np.ndarray] - 各木の点群データ。
        tree_ids: List[int] - 各木の識別番号。
        diameters: List[float] - 各木の胸高直径（DBH）。
    """
    # 全ての木の Z 座標を取得
    all_z_coords = np.concatenate([tree[:, 2] for tree in trees])

    # 全体の Z 座標範囲を計算
    z_min = all_z_coords.min()
    z_max = all_z_coords.max()

    plotter = pv.Plotter()

    for tree, tree_id, diameter, height in zip(trees, tree_ids, diameters, heights):
    # for tree, tree_id, diameter in zip(trees, tree_ids, diameters):
        # 点群データが numpy.ndarray の場合、そのまま使用
        if isinstance(tree, np.ndarray):
            cloud_points = tree
        else:
            raise ValueError("Expected tree to be a numpy.ndarray, but got type: {}".format(type(tree)))

        # Z 座標を取得
        z_coords = cloud_points[:, 2]

        # 全体の Z 座標範囲に基づいて正規化
        normalized_z = (z_coords - z_min) / (z_max - z_min)

        # カラーマップを適用
        colors = plt.cm.rainbow(normalized_z)[:, :3]  # RGBA の A を除外

        # 点群をプロット
        cloud = pv.PolyData(cloud_points)
        cloud["colors"] = colors  # カラー情報を追加
        plotter.add_mesh(cloud, scalars="colors", rgb=True, point_size=5, render_points_as_spheres=True)

        # 木の中心（重心）を計算
        tree_center = np.mean(cloud_points, axis=0)
        x, y, z = tree_center  # 座標を展開

        # ラベルを作成 (ID, DBH, 座標を表示)
        # label = f"ID: {tree_id}\nDBH: {diameter:.2f} m\n({x:.2f}, {y:.2f}, {z:.2f})"
        # ラベルを作成 (ID, DBH, 樹高を表示)
        label = f"ID: {tree_id}\nDBH: {diameter:.2f} m\nHeight: {height:.2f} m"
        plotter.add_point_labels([tree_center], [label], point_size=20, text_color='white')

    plotter.show()


# 樹木検出処理を行う関数
def process_trees(PointCloudV):
    My_treetool = tree_tool.treetool(PointCloudV)
    
    # ステップ1: 地面除去
    My_treetool.step_1_remove_floor()
    # デバッグ: non_ground_cloud と ground_cloud の確認
    print(f"non_ground_cloud: {getattr(My_treetool, 'non_ground_cloud', 'Not Set')}")
    print(f"ground_cloud: {getattr(My_treetool, 'ground_cloud', 'Not Set')}")
    
    if hasattr(My_treetool, 'non_ground_cloud') and hasattr(My_treetool, 'ground_cloud'):
        # 両方がセットされていれば可視化
        visualize_point_cloud_o3d([My_treetool.non_ground_cloud, My_treetool.ground_cloud], voxel_size=0.01)
        visualize_point_cloud_o3d([My_treetool.non_ground_cloud], voxel_size=0.3)
        visualize_point_cloud_o3d([My_treetool.ground_cloud], voxel_size=0.3)
    else:
        print("non_ground_cloud or ground_cloud is not set correctly.")
    
    # ステップ2: 法線フィルタリング
    My_treetool.step_2_normal_filtering(verticality_threshold=0.08, curvature_threshold=0.10)
    visualize_point_cloud_o3d([My_treetool.filtered_points.xyz, 
                               My_treetool.filtered_points.xyz + My_treetool.filtered_normals * 0.05], voxel_size=0.3)
    
    My_treetool.step_2_5_detect_trees(use_ransac=True)
    # My_treetool.step_2_5_detect_trees()
    # ステップ3: ユークリッドクラスタリング
    My_treetool.step_3_cluster_trees(tolerance=0.1, min_cluster_size=40, max_cluster_size=6000000)
    visualize_point_cloud_o3d(My_treetool.cluster_list, voxel_size=0.3)
    visualize_slice_and_circles(My_treetool.sliced_points, My_treetool.detected_circles)
    visualize_detected_trees_open3d(My_treetool.detected_trees,PointCloudV)
    
    # ステップ4: 幹のグループ化
    My_treetool.step_4_group_stems(max_distance=0.4)
    visualize_point_cloud_o3d(My_treetool.complete_Stems, voxel_size=0.2)
    visualize_stem_groups_open3d(My_treetool.stem_groups,PointCloudV)
    
    # ステップ5: 樹木の高さ調整
    My_treetool.step_5_get_ground_level_trees(lowstems_height=5, cutstems_height=5)
    visualize_point_cloud_o3d(My_treetool.low_stems, voxel_size=0.2)
    
    # ステップ6: 円筒モデルによる樹木モデリング
    My_treetool.step_6_get_cylinder_tree_models(search_radius=0.1)
    visualize_point_cloud_o3d([i['tree'] for i in My_treetool.finalstems] + My_treetool.visualization_cylinders, voxel_size=0.2)
    
    # ステップ7: 楕円フィット
    My_treetool.step_7_ellipse_fit()

     # 樹木のIDとDBHを取得
    tree_ids = list(range(1, len(My_treetool.finalstems) + 1))
    diameters = [i['final_diameter'] for i in My_treetool.finalstems]
    heights = [i.get("height", None) for i in My_treetool.finalstems] 
    
    # 結果をCSVに保存
    My_treetool.save_results(save_location='results/myresults.csv')
    print("Results saved to 'results/myresults.csv'")

    # 各木を可視化（Open3D）
    # visualize_point_cloud_o3d([i['tree'] for i in My_treetool.finalstems], voxel_size=0.2, tree_ids=tree_ids)

    # 各木を可視化（matplotlib）
    # visualize_with_matplotlib([i['tree'] for i in My_treetool.finalstems], tree_ids)
    
    # visualize_trees_with_pyvista([i['tree'] for i in My_treetool.finalstems], tree_ids)
    # 各木を可視化 (PyVista)
    visualize_trees_with_pyvista(
        trees=[i['tree'] for i in My_treetool.finalstems],
        tree_ids=tree_ids,
        diameters=diameters,
        heights=heights
    )

    tree_ids = list(range(1, len(My_treetool.finalstems) + 1))
    tree_centers = [np.mean(i['tree'], axis=0) for i in My_treetool.finalstems]
    return My_treetool, tree_centers, tree_ids

    # return My_treetool

def main():
    # 点群データの読み込み
    # file_directory = 'data/downsampledlesscloudEURO2.pcd'  # このパスを実際のPCDファイルのパスに変更してください
    # file_directory = 'data/rotated_test.pcd'
    # file_directory = 'data/place1_1.pcd'
    # file_directory = 'data/field.pcd'
    # file_directory = 'data/scana5left.pcd'
    # file_directory = 'data/4.pcd'
    # file_directory = 'data/50th_mid.pcd'
    # file_directory = 'data/50th_trim.pcd'
    file_directory = 'data/sample5.pcd'
    # file_directory = 'data/downsample5_0.05.pcd'
    # file_directory = 'data/dense_forest_with_gaps.pcd'
    PointCloud = load_point_cloud(file_directory)
    
    # 点群データのボクセル化
    PointCloudV = voxelize_point_cloud(PointCloud, voxel_size=0.01)
    print(f"PointCloudV shape after voxelization: {PointCloudV.shape}")
    
    # 点群データをOpen3Dで可視化
    # visualize_point_cloud_o3d([PointCloudV], voxel_size=0.3)
    My_treetool, tree_centers, tree_ids = process_trees(PointCloudV)
    debug_visualize_point_cloud_o3d(PointCloud, tree_centers, tree_ids)
    
    # 樹木検出処理
    # My_treetool = process_trees(PointCloudV)
    
    # 結果の確認
    print("Detected tree data and DBH saved to CSV.")
    
if __name__ == "__main__":
    main()
