"""
MIT License - Copyright (c) 2021 porteratzo
このファイルは、pclpyへの依存をなくし、Open3Dを使用するように全面的に書き換えられました。
"""
import numpy as np
import pandas as pd
import treetool.seg_tree as seg_tree
import treetool.utils as utils
from ellipse import LsqEllipse
import os
import open3d as o3d
from skimage.measure import CircleModel
import hdbscan

class treetool:
    def __init__(self, point_cloud_np=None):
        self.point_cloud = None
        self.non_ground_cloud = None
        self.ground_cloud = None
        self.filtered_points = None
        
        if point_cloud_np is not None:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_cloud_np)
            self.point_cloud = pcd

    def set_point_cloud(self, point_cloud_np):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud_np)
        self.point_cloud = pcd

    def step_1_remove_floor(self, distance_threshold=0.2, ransac_n=3, num_iterations=1000):
        if self.point_cloud is None: return
        self.non_ground_cloud, self.ground_cloud = seg_tree.floor_remove_o3d(
            self.point_cloud, distance_threshold, ransac_n, num_iterations)
        print(f"Ground removal complete. Non-ground points: {len(self.non_ground_cloud.points)}")

    def step_2_normal_filtering(self, search_radius=0.2, max_nn=30, verticality_threshold=0.2):
        if self.non_ground_cloud is None: return
        
        pcd_copy = o3d.geometry.PointCloud(self.non_ground_cloud)
        seg_tree.estimate_normals_o3d(pcd_copy, search_radius, max_nn)

        if not pcd_copy.has_normals(): return

        normals = np.asarray(pcd_copy.normals)
        dot_product = np.abs(np.dot(normals, np.array([0, 0, 1])))
        
        # 法線が水平に近い（地面に垂直）点を選択
        mask = dot_product < verticality_threshold
        self.filtered_points = pcd_copy.select_by_index(np.where(mask)[0])
        print(f"Normal filtering complete. Filtered points: {len(self.filtered_points.points)}")

    def step_2_5_detect_trees(self, height=1.3, tol=0.5, hdbscan_min_cluster_size=20):
        if self.filtered_points is None: return

        points_np = np.asarray(self.filtered_points.points)
        mask = (points_np[:, 2] > height - tol) & (points_np[:, 2] < height + tol)
        slice_points = points_np[mask]
        self.sliced_points = slice_points
        
        if len(slice_points) < hdbscan_min_cluster_size:
            print("Not enough points in slice for HDBSCAN.")
            self.detected_trees = []
            return

        points_2d = slice_points[:, :2]
        clusterer = hdbscan.HDBSCAN(min_cluster_size=hdbscan_min_cluster_size, allow_single_cluster=True)
        labels = clusterer.fit_predict(points_2d)

        self.detected_trees = []
        for label in np.unique(labels[labels != -1]):
            cluster_points_2d = points_2d[labels == label]
            xc, yc = np.mean(cluster_points_2d, axis=0)
            # 半径は円フィッティングをしていないので、後で計算するかダミー値
            distances = np.linalg.norm(cluster_points_2d - [xc, yc], axis=1)
            radius = np.mean(distances)
            self.detected_trees.append((xc, yc, radius))
        print(f"STEP2.5 Detected {len(self.detected_trees)} unique trees.")

    def step_3_cluster_trees(self, min_cluster_size=40, initial_radius=0.5):
        if not hasattr(self, 'detected_trees') or not self.detected_trees: return
            
        points_np = np.asarray(self.filtered_points.points)
        kdtree = o3d.geometry.KDTreeFlann(self.filtered_points)
        final_clusters = []

        for xc, yc, r in self.detected_trees:
            [k, idx, _] = kdtree.search_radius_vector_3d([xc, yc, 1.3], initial_radius)
            if k < min_cluster_size: continue
            
            initial_points = points_np[idx, :]
            
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, allow_single_cluster=True)
            labels = clusterer.fit_predict(initial_points)
            
            unique_labels = np.unique(labels[labels != -1])
            if len(unique_labels) > 0:
                largest_label = max(unique_labels, key=lambda l: np.sum(labels == l))
                final_clusters.append(initial_points[labels == largest_label])
        
        self.cluster_list = final_clusters
        print(f"STEP3 Clustered {len(self.cluster_list)} trees.")

    def step_3_5_refine_trunks_with_cylinder_model(self, distance_threshold=0.1, ransac_n=10, num_iterations=100):
        """
        円筒モデル法を用いて、幹のクラスタを純化する。
        step_3で見つけた大まかなクラスタから円筒モデルを推定し、そのモデルにフィットする点のみを
        non_ground_cloud全体から再収集する。
        """
        if not hasattr(self, 'cluster_list') or not self.cluster_list:
            print("No rough clusters found to refine.")
            return

        print("\n--- STEP 3.5: Refining Trunks with Cylinder Models ---")
        
        non_ground_points_np = np.asarray(self.non_ground_cloud.points)
        refined_trunk_clusters = []

        # step_3で見つかった各クラスタをループ
        for i, rough_cluster in enumerate(self.cluster_list):
            if len(rough_cluster) < ransac_n:
                continue

            # NumPy配列からOpen3DのPointCloudオブジェクトを作成
            pcd_cluster = o3d.geometry.PointCloud()
            pcd_cluster.points = o3d.utility.Vector3dVector(rough_cluster)
            
            # RANSACで円筒モデルを推定（seg_treeのダミー関数を使用）
            # この関数は中心、軸、半径を含むモデル係数を返す
            inliers, model_coeffs = seg_tree.segment_cylinder_o3d(
                pcd_cluster,
                distance_threshold=distance_threshold,
                ransac_n=ransac_n,
                num_iterations=num_iterations
            )
            
            # モデルが推定できた場合のみ処理を続行
            if model_coeffs:
                # モデルパラメータを抽出
                center = np.array(model_coeffs[0:3])
                axis = np.array(model_coeffs[3:6])
                radius = model_coeffs[6]

                # 軸ベクトルがほぼ0の場合はスキップ
                if np.linalg.norm(axis) < 1e-6:
                    continue

                # non_ground_cloudの全ての点と、推定された円筒の軸との距離を計算
                # 3D空間での点と直線の距離の公式を使用
                p1 = center - axis * 10 # 軸上の点1
                p2 = center + axis * 10 # 軸上の点2
                
                # ブロードキャストを用いて効率的に計算
                d = np.linalg.norm(np.cross(p2 - p1, non_ground_points_np - p1), axis=1) / np.linalg.norm(p2 - p1)
                
                # 距離が半径（+少しの許容誤差）以内にある点をすべて選択
                # これにより、葉や枝ではなく、幹に属する点だけが選ばれる
                refined_mask = d < (radius + 0.05) # 半径に5cmの許容誤差
                
                # 十分な点があれば、純化されたクラスタとして追加
                if np.sum(refined_mask) > 50:
                    refined_trunk_clusters.append(non_ground_points_np[refined_mask])

        # 純化されたクラスタで元のクラスタリストを上書き
        self.cluster_list = refined_trunk_clusters
        print(f"Refined {len(self.cluster_list)} trunks using cylinder models.")

    def step_4_group_stems(self, max_distance=0.4):
        if not hasattr(self, 'cluster_list') or not self.cluster_list: return
        self.complete_Stems = self.cluster_list # このバージョンではグループ化を簡略化
        print(f"STEP4: Grouping simplified. Found {len(self.complete_Stems)} stems.")

    def step_5_get_ground_level_trees(self, lowstems_height=5):
        if not hasattr(self, 'complete_Stems') or not self.complete_Stems: return
            
        ground_points = np.asarray(self.ground_cloud.points)
        A = np.c_[np.ones(ground_points.shape[0]), ground_points[:, :2], np.prod(ground_points[:, :2], axis=1), ground_points[:, :2] ** 2]
        self.ground_model_c, _, _, _ = np.linalg.lstsq(A, ground_points[:, 2], rcond=None)

        self.stems_with_ground = []
        for stem in self.complete_Stems:
            center = np.mean(stem, axis=0)
            X, Y = center[:2]
            Z = np.dot(np.c_[1, X, Y, X * Y, X**2, Y**2], self.ground_model_c).item()
            
            # 地面より低い点がある幹のみを保持
            if np.min(stem[:, 2]) < (Z + lowstems_height):
                 self.stems_with_ground.append([stem, [X, Y, Z]])
        
        self.low_stems = [s[0] for s in self.stems_with_ground]
        print(f"STEP5: {len(self.low_stems)} stems remain after ground-level filtering.")

    def step_6_get_cylinder_tree_models(self):
        self.finalstems = []
        for stem, ground_info in self.stems_with_ground:
            center = np.mean(stem, axis=0)
            radius = np.mean(np.linalg.norm(stem[:, :2] - center[:2], axis=1))
            height = np.max(stem[:, 2]) - ground_info[2]
            
            model = [center[0], center[1], center[2], 0, 0, 1, radius] # 垂直を仮定
            
            self.finalstems.append({
                "tree": stem, "model": model,
                'ground': ground_info[2], "height": height
            })
        print(f"STEP6: Simplified modeling for {len(self.finalstems)} stems.")
        
    def step_7_ellipse_fit(self):
        if not hasattr(self, 'finalstems'): return
        
        for i in self.finalstems:
            points = i["tree"]
            model = i["model"]
            R = utils.rotation_matrix_from_vectors(model[3:6], [0, 0, 1])
            centered_tree = points - model[:3]
            corrected_cyl = centered_tree @ R.T

            try:
                reg = LsqEllipse().fit(corrected_cyl[:, :2])
                _, a, b, _ = reg.as_parameters()
                # 楕円周長ラマヌジャン近似
                ellipse_circumference = np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b)))
                ellipse_diameter = ellipse_circumference / np.pi
            except Exception:
                ellipse_diameter = model[6] * 2

            i["final_diameter"] = max(ellipse_diameter, model[6] * 2)
        print("STEP7: Ellipse fitting complete.")

    def save_results(self, save_location="results/myresults.csv"):
        if not hasattr(self, 'finalstems') or not self.finalstems: return
        os.makedirs(os.path.dirname(save_location), exist_ok=True)
        data = {
            "ID": list(range(1, len(self.finalstems) + 1)),
            "X": [i["model"][0] for i in self.finalstems],
            "Y": [i["model"][1] for i in self.finalstems],
            "Z": [i["model"][2] for i in self.finalstems],
            "DBH": [i["final_diameter"] for i in self.finalstems],
            "Height": [i.get("height", 0) for i in self.finalstems]
        }
        pd.DataFrame(data).to_csv(save_location, index=False)