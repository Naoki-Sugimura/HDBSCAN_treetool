import numpy as np
import open3d as o3d
import treetool.tree_tool as tree_tool
import pyvista as pv
import matplotlib.pyplot as plt
import random

def load_point_cloud_o3d(file_path):
    """Open3Dで点群を読み込む"""
    pcd = o3d.io.read_point_cloud(file_path)
    if not pcd.has_points():
        raise ValueError(f"Could not read point cloud from file: {file_path}")
    return pcd

def visualize_final_colored_clusters(original_pcd, final_clusters):
    """
    元の点群データに、最終的なクラスタリング結果を色付けしてPyVistaで表示する。

    Args:
        original_pcd (open3d.geometry.PointCloud): 元の（ダウンサンプリングされた）点群データ。
        final_clusters (list of numpy.ndarray): 最終的に検出された幹クラスタのリスト。
    """
    print("\n--- Visualizing Final Result on Original Point Cloud ---")
    
    points = np.asarray(original_pcd.points)
    # まず全ての点を灰色で初期化
    colors = np.full((points.shape[0], 3), [0.6, 0.6, 0.6])

    if not final_clusters:
        print("No clusters to color.")
    else:
        # 高速な近傍探索のため、元の点群からKDTreeを作成
        pcd_tree = o3d.geometry.KDTreeFlann(original_pcd)
        
        # 各クラスタにユニークな色を割り当てるためのカラーマップを準備
        cmap = plt.get_cmap('viridis', len(final_clusters))

        print("Coloring final clusters...")
        for i, cluster in enumerate(final_clusters):
            if cluster.size == 0:
                continue
            
            # カラーマップから色を取得 (RGBA -> RGB)
            color = cmap(i)[:3]
            
            # クラスタ内の各点について、元の点群で最も近い点のインデックスを探し、色を塗る
            for point in cluster:
                # search_knn_vector_3dは、指定された点に最も近い1つの点を探す
                [k, idx, _] = pcd_tree.search_knn_vector_3d(point, 1)
                if k > 0:
                    colors[idx[0]] = color
    
    # PyVistaで可視化
    pv_cloud = pv.PolyData(points)
    pv_cloud['colors'] = colors

    print("Launching final visualization window...")
    plotter = pv.Plotter(window_size=[1600, 1200])
    # 点のサイズを大きくする (例: 5)
    plotter.add_mesh(pv_cloud, scalars='colors', rgb=True, point_size=5, render_points_as_spheres=True)
    plotter.add_axes()
    plotter.background_color = 'white'
    plotter.show()


def process_trees(pcd):
    """Open3Dの点群オブジェクトを入力として樹木検出処理全体を実行する"""
    My_treetool = tree_tool.treetool(np.asarray(pcd.points))
    
    print("--- STEP 1: Ground Removal ---")
    My_treetool.step_1_remove_floor()
    # --- 可視化1: 地面除去の結果 ---
    print("Visualizing: Ground (Green) and Non-Ground (Blue) points. Close window to continue.")
    My_treetool.ground_cloud.paint_uniform_color([0.1, 0.9, 0.1])
    My_treetool.non_ground_cloud.paint_uniform_color([0.2, 0.2, 0.9])
    o3d.visualization.draw_geometries([My_treetool.ground_cloud, My_treetool.non_ground_cloud], window_name="Step 1: Ground Removal")

    print("\n--- STEP 2: Normal Filtering ---")
    My_treetool.step_2_normal_filtering()

    print("\n--- STEP 2.5: Detect Trees ---")
    My_treetool.step_2_5_detect_trees()
    # --- 可視化2: 樹木検出の結果 ---
    print("Visualizing: Sliced points (Green) and detected tree centers (Red spheres).")
    slice_pcd = o3d.geometry.PointCloud()
    slice_pcd.points = o3d.utility.Vector3dVector(My_treetool.sliced_points)
    slice_pcd.paint_uniform_color([0.2, 0.8, 0.2])
    spheres = []
    for x, y, r in My_treetool.detected_trees:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=r if r > 0.01 else 0.1)
        sphere.translate([x, y, 1.3])
        sphere.paint_uniform_color([1.0, 0.1, 0.1])
        spheres.append(sphere)
    o3d.visualization.draw_geometries([slice_pcd] + spheres, window_name="Step 2.5: Tree Detection")

    print("\n--- STEP 3: Rough Cluster Trees ---")
    My_treetool.step_3_cluster_trees()

    print("\n--- STEP 3.5: Refine Trunks with Cylinder Model ---")
    My_treetool.step_3_5_refine_trunks_with_cylinder_model()
    # --- 可視化3: 純化された幹のクラスタリング結果 ---
    if My_treetool.cluster_list:
        print("Visualizing: Refined tree trunks (each in a different color).")
        clusters_pcd_list = []
        for cluster_np in My_treetool.cluster_list:
            cluster_pcd = o3d.geometry.PointCloud()
            cluster_pcd.points = o3d.utility.Vector3dVector(cluster_np)
            cluster_pcd.paint_uniform_color([random.random(), random.random(), random.random()])
            clusters_pcd_list.append(cluster_pcd)
        o3d.visualization.draw_geometries(clusters_pcd_list, window_name="Step 3.5: Refined Trunks")

    print("\n--- STEP 4 & 5: Group and Filter Stems ---")
    My_treetool.step_4_group_stems()
    My_treetool.step_5_get_ground_level_trees()

    print("\n--- STEP 6 & 7: Modeling and Fitting ---")
    My_treetool.step_6_get_cylinder_tree_models()
    My_treetool.step_7_ellipse_fit()
    
    # --- 結果の保存 ---
    My_treetool.save_results(save_location='results/myresults.csv')
    print("\nResults saved to 'results/myresults.csv'")
    
    return My_treetool


def main():
    file_directory = 'data/sample5.pcd'
    
    pcd = load_point_cloud_o3d(file_directory)
    print(f"Loaded {len(pcd.points)} points.")
    
    voxel_size = 0.05
    pcd_down = pcd.voxel_down_sample(voxel_size)
    print(f"Downsampled to {len(pcd_down.points)} points.")
    
    # --- 樹木検出プロセスの実行 ---
    My_treetool = process_trees(pcd_down)

    # --- 最終結果の可視化 ---
    if hasattr(My_treetool, 'low_stems') and My_treetool.low_stems:
        visualize_final_colored_clusters(pcd_down, My_treetool.low_stems)
    else:
        print("No final clusters found to visualize.")

    print("\nTree detection and visualization process finished.")

if __name__ == "__main__":
    main()
