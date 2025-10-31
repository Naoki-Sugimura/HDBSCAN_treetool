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
from scipy.spatial.distance import pdist, squareform # カスタム距離関数のためにscipyをインポート

class treetool:
    def __init__(self, point_cloud_np=None):
        self.point_cloud = None
        self.non_ground_cloud = None
        self.ground_cloud = None
        self.filtered_points = None
        
        # --- 追加: カスタムHDBSCANのために法線と曲率を格納 ---
        self.kdtree = None
        self.points_curvature = None # 各点の疑似曲率を格納
        self.points_normals = None   # 各点の法線ベクトルを格納
        # ---------------------------------------------------
        
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

    # tree_tool.py の step_2_normal_filtering 関数全体を置き換え
    def step_2_normal_filtering(self, search_radius=0.2, max_nn=30, verticality_threshold=0.2):
        if self.non_ground_cloud is None: return
        
        pcd_copy = o3d.geometry.PointCloud(self.non_ground_cloud)
        seg_tree.estimate_normals_o3d(pcd_copy, search_radius, max_nn)

        if not pcd_copy.has_normals(): return

        normals = np.asarray(pcd_copy.normals)
        points = np.asarray(pcd_copy.points) # 点群座標
        
        # --- 曲率の事前計算と法線フィルタリング ---
        kdtree = o3d.geometry.KDTreeFlann(pcd_copy)
        curvatures = np.zeros(points.shape[0])
        
        # 各点pの近傍点を使って曲率を計算
        for i in range(points.shape[0]):
            [k, idx, _] = kdtree.search_hybrid_vector_3d(points[i], search_radius, max_nn)
            # 曲率計算のためには最低でも4点以上必要
            if k > 3:
                neighbors = points[idx]
                curvatures[i] = utils.get_curvature(neighbors)

        dot_product = np.abs(np.dot(normals, np.array([0, 0, 1])))
        
        # 法線が水平に近い（地面に垂直）点を選択
        mask = dot_product < verticality_threshold
        
        filtered_indices = np.where(mask)[0]
        self.filtered_points = pcd_copy.select_by_index(filtered_indices)
        
        # フィルタリングされた点に対応するデータのみを保存
        # KDTreeはfiltered_pointsに対して構築する
        self.kdtree = o3d.geometry.KDTreeFlann(self.filtered_points)
        self.points_curvature = curvatures[filtered_indices]
        self.points_normals = normals[filtered_indices]

        print(f"Normal filtering and curvature pre-calculation complete. Filtered points: {len(self.filtered_points.points)}")


    # tree_tool.py の step_2_5_detect_trees 関数全体を置き換え
    def step_2_5_detect_trees(self, height=2.5, tol=0.5, hdbscan_min_cluster_size=60):
        if self.filtered_points is None: return

        points_np = np.asarray(self.filtered_points.points)
        
        mask = (points_np[:, 2] > height - tol) & (points_np[:, 2] < height + tol)
        slice_points = points_np[mask]
        self.sliced_points = slice_points
        
        if len(slice_points) < hdbscan_min_cluster_size:
            print("Not enough points in slice for HDBSCAN.")
            self.detected_trees = []
            return

        points_3d = slice_points[:, :3] 
        original_indices = np.where(mask)[0]
        
        # スライスされた点に対応する曲率と法線を取得
        slice_curvatures = self.points_curvature[original_indices]
        slice_normals = self.points_normals[original_indices]
        
        # --- カスタムHDBSCANのパラメータ ---
        ALPHA = 0.1                            # α: 曲率差d_curve(p,q)に適用する正の重み
        MAX_SHORTEST_DISTANCE = 0.05           # 2直線が「交わったと見なす」最短距離 (5cm)
        MIN_DISTANCE_TO_PQ = 0.30              # 交点からPとQまでの距離が満たすべき最低距離 (30cm)
        
        # --- 【追加】カスタム距離調整のためのパラメータ ---
        KAPPA_ZERO_THRESHOLD = 1e-4            # 曲率を「ゼロ」と見なすしきい値
        EPSILON_DIFF = 1e-4                    # 曲率の差を「ゼロ」と見なすしきい値
        LARGE_PSEUDO_DIST = 3                  # 両曲率ゼロの場合に与える大きな距離 (分離)
        BETA_INTERSECT = 0.5                   # β: 交差条件を満たした場合の追加ペナルティ
        
        # *元のBOOST_FACTORは、カスタム距離がゼロに近い場合の例外処理 4-B に利用し、距離を短縮する
        BOOST_FACTOR = 0.001
        # ---------------------------------------------------

        # 点のXY座標を抽出 (P_xy = [Xp, Yp], Q_xy = [Xq, Yq])
        points_xy = points_3d[:, :2]
        
        # 法線ベクトルのXY成分を抽出 (N_xy = [Nx, Ny])
        normals_xy = slice_normals[:, :2]

        # --- XY平面の直線の交点/最短距離と交点を求める関数 (変更なし) ---
        def shortest_distance_and_intersection_2D(P_xy, Np_xy, Q_xy, Nq_xy):
            """
            XY平面上で、Pを通る方向ベクトルr(-Np_xy)の直線と、Qを通る方向ベクトルs(-Nq_xy)の直線の
            交点（または最短距離）を計算する。
            """
            p = P_xy
            q = Q_xy
            r = -Np_xy 
            s = -Nq_xy
            r_cross_s = r[0] * s[1] - r[1] * s[0]
            q_minus_p = q - p
            
            if abs(r_cross_s) < 1e-6:
                shortest_dist = abs(q_minus_p[0] * r[1] - q_minus_p[1] * r[0]) / np.linalg.norm(r)
                return shortest_dist, np.full(2, np.nan)

            t = (q_minus_p[0] * s[1] - q_minus_p[1] * s[0]) / r_cross_s
            intersection = p + t * r
            return 0.0, intersection


        # --- 距離行列の手動計算 ---
        N = len(points_3d)
        distance_matrix = np.zeros((N, N))
        
        for i in range(N):
            for j in range(i + 1, N):
                u = points_3d[i] # Pの座標 (3D)
                v = points_3d[j] # Qの座標 (3D)

                # 1. dist(p,q): ユークリッド距離
                dist_pq = np.linalg.norm(u - v)
                
                # 2. d_curve(p,q)
                kappa_p = slice_curvatures[i]
                kappa_q = slice_curvatures[j]
                d_curve_pq = np.abs(kappa_p - kappa_q) 
                
                # 3. β項の計算
                BETA_TERM = 0.0 # 初期値: β
                
                # b. XY平面での交差判定
                P_xy = points_xy[i]
                Q_xy = points_xy[j]
                Np_xy = normals_xy[i]
                Nq_xy = normals_xy[j]
                
                shortest_dist_xy, intersection_xy = shortest_distance_and_intersection_2D(
                    P_xy, Np_xy, Q_xy, Nq_xy
                )

                # c. 交差判定 (10cmマージン) と 30cm 距離条件
                if shortest_dist_xy <= MAX_SHORTEST_DISTANCE:
                    if not np.any(np.isnan(intersection_xy)):
                        dist_to_P = np.linalg.norm(intersection_xy - P_xy)
                        dist_to_Q = np.linalg.norm(intersection_xy - Q_xy)
                        
                        if dist_to_P >= MIN_DISTANCE_TO_PQ and dist_to_Q >= MIN_DISTANCE_TO_PQ:
                            # 条件を満たした場合: 違う幹である可能性が高いため、距離ペナルティ(BETA_TERM)を加算
                            BETA_TERM = BETA_INTERSECT
                
                # --- 4. x(p,q) のカスタム距離計算ロジック (足し算形式) ---
                
                x_pq = 0.0 # 初期化

                # 4-A. 最優先：両曲率がゼロに近い場合の処理 (ノイズ/地面として分離)
                if abs(kappa_p) < KAPPA_ZERO_THRESHOLD and abs(kappa_q) < KAPPA_ZERO_THRESHOLD:
                    # 両方平坦（地面やノイズ）なので、クラスタリングから分離
                    x_pq = LARGE_PSEUDO_DIST 

                # 4-B. 次点：曲率の差が非常に小さい場合の処理 (幹として結合を促進)
                elif d_curve_pq < EPSILON_DIFF:
                    # 曲率が似ているので結合を強く促進
                    # ゼロ除算回避ロジックを流用し、距離を大幅に短縮する
                    x_pq = dist_pq - BOOST_FACTOR 
                    
                    # x_pq = max(0.0, dist_pq - BOOST_FACTOR) としても良いが、max(0.0, x_pq) が最後に実行されるため省略
                
                # 4-C. 通常のカスタム距離計算 (提案された式: x = dist + α*d_curve + β)
                else:
                    # 曲率差 d_curve(p,q) が小さいほど、ALPHA*d_curve 項が小さくなり、結合が促進される。
                    # BETA_TERM (交差ペナルティ) は、交差条件を満たせば距離を伸ばし、分離を促進する。
                    x_pq = dist_pq + ALPHA * d_curve_pq + BETA_TERM
                
                # 距離は常に非負である必要がある
                distance = max(0.0, x_pq)
                
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance

        # HDBSCANに**事前計算された距離行列**を渡す (変更なし)
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=hdbscan_min_cluster_size, 
            metric='precomputed', 
            allow_single_cluster=True
        )
        labels = clusterer.fit_predict(distance_matrix)

        # 後続の処理は既存のまま (変更なし)
        self.detected_trees = []
        for label in np.unique(labels[labels != -1]):
            cluster_points_3d = points_3d[labels == label]
            
            xc, yc, zc = np.mean(cluster_points_3d, axis=0)
            distances_2d = np.linalg.norm(cluster_points_3d[:, :2] - [xc, yc], axis=1)
            radius = np.mean(distances_2d)
            
            self.detected_trees.append((xc, yc, radius))
        print(f"STEP2.5 Detected {len(self.detected_trees)} unique trees using 3D HDBSCAN with custom curvature and 2D intersection metric.")


    # --- step_3 以降は、以前の修正と機能に基づき、HDBSCANのメトリック変更に関係のない部分を維持 ---

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
        if not hasattr(self, 'cluster_list') or not self.cluster_list:
            print("No rough clusters found to refine.")
            return

        print("\n--- STEP 3.5: Refining Trunks with Cylinder Models ---")
        
        non_ground_points_np = np.asarray(self.non_ground_cloud.points)
        refined_trunk_clusters = []

        for i, rough_cluster in enumerate(self.cluster_list):
            if len(rough_cluster) < ransac_n:
                continue

            pcd_cluster = o3d.geometry.PointCloud()
            pcd_cluster.points = o3d.utility.Vector3dVector(rough_cluster)
            
            inliers, model_coeffs = seg_tree.segment_cylinder_o3d(
                pcd_cluster,
                distance_threshold=distance_threshold,
                ransac_n=ransac_n,
                num_iterations=num_iterations
            )
            
            if model_coeffs:
                center = np.array(model_coeffs[0:3])
                axis = np.array(model_coeffs[3:6])
                radius = model_coeffs[6]

                if np.linalg.norm(axis) < 1e-6:
                    continue

                p1 = center - axis * 10
                p2 = center + axis * 10
                
                d = np.linalg.norm(np.cross(p2 - p1, non_ground_points_np - p1), axis=1) / np.linalg.norm(p2 - p1)
                
                refined_mask = d < (radius + 0.05)
                
                if np.sum(refined_mask) > 50:
                    refined_trunk_clusters.append(non_ground_points_np[refined_mask])

        self.cluster_list = refined_trunk_clusters
        print(f"Refined {len(self.cluster_list)} trunks using cylinder models.")

    def step_4_group_stems(self, max_distance=0.4):
        if not hasattr(self, 'cluster_list') or not self.cluster_list: return
        self.complete_Stems = self.cluster_list
        print(f"STEP4: Grouping simplified. Found {len(self.complete_Stems)} stems.")

    def step_5_get_ground_level_trees(self, lowstems_height=5):
        if not hasattr(self, 'complete_Stems') or not self.complete_Stems: return
            
        ground_points_z = np.asarray(self.ground_cloud.points)[:, 2]
        if ground_points_z.size == 0:
            print("DEBUG (STEP 5): Warning: No ground points found. Skipping step.")
            self.stems_with_ground = []
            return
            
        sorted_z = np.sort(ground_points_z)
        N = int(len(sorted_z) * 0.1)
        if N == 0: N = 1
        
        ground_z_reference = np.mean(sorted_z[:N])
        
        self.ground_model_c = [0] * 6
        
        self.stems_with_ground = []
        for i, stem in enumerate(self.complete_Stems):
            X, Y = np.mean(stem, axis=0)[:2]
            Z = ground_z_reference
            
            min_stem_z = np.min(stem[:, 2])
            
            if min_stem_z < (Z + lowstems_height):
                self.stems_with_ground.append([stem, [X, Y, Z]])
        
        self.low_stems = [s[0] for s in self.stems_with_ground]
        print(f"STEP5: {len(self.low_stems)} stems remain after ground-level filtering.")

    def step_6_get_cylinder_tree_models(self):
        self.finalstems = []
        
        SLICE_THICKNESS = 0.1
        MIN_POINTS_PER_SLICE = 10
        MAX_SEARCH_HEIGHT = 25.0
        
        for idx, (stem, ground_info) in enumerate(self.stems_with_ground):
            ground_z = ground_info[2]
            relative_z = stem[:, 2] - ground_z
            max_stem_relative_z = np.max(relative_z)
            
            search_limit = min(max_stem_relative_z, MAX_SEARCH_HEIGHT)
            
            last_valid_slice_height = 0.0
            
            for h in np.arange(SLICE_THICKNESS, search_limit + 0.001, SLICE_THICKNESS):
                lower_bound = h - SLICE_THICKNESS
                upper_bound = h
                
                slice_mask = (relative_z >= lower_bound) & (relative_z < upper_bound)
                point_count = np.sum(slice_mask)
                
                if point_count >= MIN_POINTS_PER_SLICE:
                    last_valid_slice_height = upper_bound

            height = last_valid_slice_height 

            center = np.mean(stem, axis=0)
            radius = np.mean(np.linalg.norm(stem[:, :2] - center[:2], axis=1))
            model = [center[0], center[1], center[2], 0, 0, 1, radius] 
            
            self.finalstems.append({
                "tree": stem, "model": model,
                'ground': ground_z, "height": height 
            })
            
            print(f"  FINAL CALCULATED HEIGHT: {height:.3f} m")
            
        print(f"STEP6: Modified height calculation (0.1m slice, min 10 points, max 25m search). Simplified modeling for {len(self.finalstems)} stems.")
        
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
                ellipse_circumference = np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b)))
                ellipse_diameter = ellipse_circumference / np.pi
            except Exception:
                ellipse_diameter = model[6] * 2

            i["final_diameter"] = max(ellipse_diameter, model[6] * 2)
        print("STEP7: Ellipse fitting complete.")
        print(f"\nFinal Result: Detected {len(self.finalstems)} trees.")

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