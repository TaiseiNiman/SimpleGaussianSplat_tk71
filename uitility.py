import pycolmap
import torch
import numpy as np
import os
from PIL import Image
import torchvision
import scipy
from torch.utils.data import DataLoader, Dataset
import kornia.metrics as metrics
import math
import gc

class Utilities():
    #gpuメモリ監視
    def gpu_mem(tag=""):
        alloc = torch.cuda.memory_allocated() / 1024**3
        reserv = torch.cuda.memory_reserved() / 1024**3
        print(f"[{tag}] GPU使用中: {alloc:.2f} GB / 予約済: {reserv:.2f} GB")
    
    #明示的なメモリ開放 - > 引数が参照渡しのように働くテンソルなどでのみ有効
    def mem_refresh(val,del_bool,col_bool = True):# del_bool -> Trueなら削除,そうでないならNone代入
        if(del_bool):
            del val
        else:
            val = None
        if(col_bool):
            gc.collect()
            torch.cuda.empty_cache()
     
    #高速なソート関数
    def batch_sort_by_column(A: np.ndarray, column_index: int = 1) -> np.ndarray:
        """
        (N, M, P) テンソルの N バッチそれぞれについて、
        P 軸の指定された列の値を基準に M 軸を並べ替える。
        """
        N, M, P = A.shape
        
        # 1. 各バッチの並べ替え基準 (形状: (N, M))
        sort_keys = A[:, :, column_index]
        
        # 2. 各バッチの並べ替えインデックスを取得 (np.argsortをバッチ適用)
        # np.argsortは最後の軸に対して適用されるため、(N, M) 形状の sort_keys で実行すると、
        # 各 N ごとに並べ替えインデックスが取得される。
        sorted_indices = np.argsort(sort_keys, axis=1)

        # 3. 高度なインデックス作成のためのバッチインデックスの準備
        # 形状 (N, 1) の配列 (0, 1, 2, ...) を作成し、(N, M) にブロードキャストさせる。
        # これにより、並べ替え時に「どのバッチから取るか」を指定できる。
        batch_indices = np.arange(N)[:, None]

        # 4. 高度なインデックスを使用して一度に並べ替えを実行
        # A[batch_indices, sorted_indices, :]
        # これは A[i, sorted_indices[i], :] をすべての i について並列実行する
        A_sorted = A[batch_indices, sorted_indices, :]
        
        return A_sorted
    
    #n個の近傍点のユークリッド距離の平均を計算する
    def kyori(n, cloud):#could->(n,m)
        c0,c1 = cloud.shape
        gaus = torch.norm((cloud[None,:,:] - cloud[:,None,:]), dim=2,keepdim=False)
        #行ベクトルを昇順にソート
        gaus = gaus.sort(dim=1)
        #近傍なn個の点のユークリッド距離平均を計算
        uclid_n = torch.mean(gaus[:,0:n],dim=1,keepdim=True).repeat(1,3)
        return uclid_n # -> (n,1)
    
    def kyori2(n,cloud, batch_size=2000):
        N = cloud.shape[0]
        uclid = torch.zeros((N, 1), device=cloud.device)
        for i in range(0, N, batch_size):
            Utilities.gpu_mem("batch_before")
            end = min(i + batch_size, N)
            dists = torch.cdist(cloud[i:end], cloud)
            dists, _ = dists.sort(dim=1)
            uclid[i:end, :] = torch.mean(dists[:, :n], dim=1, keepdim=True)
            Utilities.gpu_mem("batch_after")
        return uclid.repeat(1, 3)

    
    #方向ベクトルから球面調和関数を計算
    # def direction_to_spherical_harmonics(d: np.ndarray, L_max: int = 1) -> np.ndarray:
    #     """
    #     単位方向ベクトル d から、指定された最大次数 (L_max) までの
    #     球面調和関数 (SH) の基底関数の値を計算する。

    #     Args:
    #         d (np.ndarray): 単位方向ベクトル (x, y, z)。
    #         L_max (int): SHの最大次数 (例: 1, 2, 3)。

    #     Returns:
    #         np.ndarray: 計算されたSH基底関数の値の配列 (総係数数 x 1)。
    #     """
    #     #shapeの形状を取得
    #     d0,d1,d2 = d.shape
    #     # 1. 方向ベクトルの正規化
    #     norm = np.linalg.norm(d,axis=1,keepdims=True) 
    #     norm[norm == 0] = 1
    #     d = d / norm   
        
    #     # 2. 直交座標 (x, y, z) を球面座標 (theta, phi) に変換
    #     # 慣例として、theta (極角) は z軸からの角度 [0, pi]
    #     # phi (方位角) は x軸からの角度 [0, 2*pi]
        
    #     # phi: x-y平面での角度 (numpy.arctan2(y, x))
    #     phi = np.array(np.arctan2(d[:,:,1], d[:,:,0])).reshape(d0,d1)
        
    #     # theta: z軸からの角度 (numpy.arccos(z))
    #     theta = np.array(np.arccos(d[:,:,2])).reshape(d0,d1) 

    #     # 3. SH基底関数の値を計算し、格納
    #     sh_values = []
        
    #     for l in range(L_max + 1):
    #         for m in range(-l, l + 1):
    #             # sph_harm(m, l, phi, theta) の順で引数を渡す
    #             # scipy.special.sph_harm は複素数を返すため、実部 (real) のみを使用
    #             sh_value = scipy.special.sph_harm(m, l, phi, theta).real
    #             sh_values.append(sh_value)

    #     return np.array(sh_values).reshape(d0,-1,d1) #(基底関数,画像,ガウス)→(画像,基底関数,ガウス)
    
    
    def legendre_P(l, m, x):
        """
        陪ルジャンドル多項式 P_l^m(x) を再帰的に計算
        x: (...,) torch.Tensor
        """
        # m=l の初期値
        pmm = torch.ones_like(x)
        if m > 0:
            somx2 = torch.sqrt(torch.clamp((1 - x) * (1 + x), min=0.0)) + 1e-12
            fact = 1.0
            for i in range(1, m + 1):
                pmm = -pmm * fact * somx2
                fact += 2.0

        if l == m:
            return pmm

        pmmp1 = x * (2*m + 1) * pmm
        if l == m + 1:
            return pmmp1

        pll = torch.zeros_like(x)
        for ll in range(m + 2, l + 1):
            pll = ((2*ll - 1) * x * pmmp1 - (ll + m - 1) * pmm) / (ll - m)
            pmm, pmmp1 = pmmp1, pll

        return pll


    def real_spherical_harmonics(l, m, theta, phi):
        """
        実数版球面調和関数 Y_l^m(theta, phi)
        theta, phi: (...,) torch.Tensor (autograd対応)
        """
        K = math.sqrt((2*l + 1) / (4*math.pi) *
                    math.factorial(l - abs(m)) / math.factorial(l + abs(m)))
        P = Utilities.legendre_P(l, abs(m), torch.cos(theta).clamp(-1.0, 1.0))

        if m > 0:
            return math.sqrt(2) * K * P * torch.cos(m * phi)
        elif m < 0:
            return math.sqrt(2) * K * P * torch.sin(-m * phi)
        else:
            return K * P


    def direction_to_spherical_harmonics_torch(d: torch.Tensor, L_max: int = 1):
        """
        d: (M, N, 3) 単位方向ベクトル群
            - M: 画像数
            - N: 各画像の点の数
        戻り値:
            Y: (M, N, num_basis) 各点における球面調和関数の基底値
        """
        # 正規化
        d0,d1,d2 = d.shape
        d = d / torch.norm(d, dim=-1, keepdim=True).clamp_min(1e-8)
        x, y, z = d[..., 0], d[..., 1], d[..., 2]

        #  球面座標に変換
        phi = torch.atan2(y, x)                     # (M, N)
        theta = torch.acos(z.clamp(-1, 1))          # (M, N)

        #  各次数 l,m ごとに基底を構築
        Y_list = []
        for l in range(L_max + 1):
            for m in range(-l, l + 1):
                Ylm = Utilities.real_spherical_harmonics(l, m, theta, phi)  # (M, N)
                Y_list.append(Ylm)

        # すべての基底を結合 → (M, N, num_basis)
        Y = torch.stack(Y_list, dim=-1)
        return Y.reshape(d0,d1,-1)
    
    def qvec_to_rotmat_torch(qvec: torch.Tensor) -> torch.Tensor:
        """
        qvec: (4,) tensor [qx, qy, qz, qw]
        returns: (3,3) rotation matrix
        """
        q1, q2, q3, q0 = qvec
        R = torch.tensor([
            [1 - 2*(q2**2 + q3**2), 2*(q1*q2 - q0*q3),     2*(q1*q3 + q0*q2)],
            [2*(q1*q2 + q0*q3),     1 - 2*(q1**2 + q3**2), 2*(q2*q3 - q0*q1)],
            [2*(q1*q3 - q0*q2),     2*(q2*q3 + q0*q1),     1 - 2*(q1**2 + q2**2)]
        ], device=qvec.device)
        return R
    
    # def qvec_to_rotmat_torch_batch(qvec: torch.Tensor) -> torch.Tensor:
    #     """
    #     qvec: (N, 4) tensor [qx, qy, qz, qw]
    #     returns: (N, 3, 3) rotation matrices
    #     """
    #     # 各成分を展開（ブロードキャスト可能）
    #     q1 = qvec[:, 0]
    #     q2 = qvec[:, 1]
    #     q3 = qvec[:, 2]
    #     q0 = qvec[:, 3]

    #     # 要素をまとめて計算（バッチ演算）
    #     R = torch.stack([
    #         torch.stack([1 - 2*(q2**2 + q3**2),  2*(q1*q2 - q0*q3),      2*(q1*q3 + q0*q2)], dim=1),
    #         torch.stack([2*(q1*q2 + q0*q3),      1 - 2*(q1**2 + q3**2),  2*(q2*q3 - q0*q1)], dim=1),
    #         torch.stack([2*(q1*q3 - q0*q2),      2*(q2*q3 + q0*q1),      1 - 2*(q1**2 + q2**2)], dim=1)
    #     ], dim=1).permute(2, 0, 1)  # -> (N, 3, 3)

    #     return R
    
    def qvec_to_rotmat_torch_batch(qvec: torch.Tensor) -> torch.Tensor:
        """
        qvec: (N, 4) tensor [qx, qy, qz, qw]
        returns: (N, 3, 3) rotation matrices
        """
        qx, qy, qz, qw = qvec[:, 0], qvec[:, 1], qvec[:, 2], qvec[:, 3]

        # 各行を (N, 3) としてまとめる
        row0 = torch.stack([1 - 2*(qy**2 + qz**2),
                            2*(qx*qy - qw*qz),
                            2*(qx*qz + qw*qy)], dim=1)

        row1 = torch.stack([2*(qx*qy + qw*qz),
                            1 - 2*(qx**2 + qz**2),
                            2*(qy*qz - qw*qx)], dim=1)

        row2 = torch.stack([2*(qx*qz - qw*qy),
                            2*(qy*qz + qw*qx),
                            1 - 2*(qx**2 + qy**2)], dim=1)

        # 各行を縦方向に結合して (N, 3, 3)
        R = torch.stack([row0, row1, row2], dim=1)

        return R
    
    #カメラ座標からピクセル座標系へのヤコビアン行列を計算
    def pixel_jacobian_batch(K, XYZ):
        """
        K:   (M, 3, 3) 内部パラメータ行列（画像ごと）
        XYZ: (M, N, 3) 各画像におけるカメラ座標点群
        return: (M, N, 2, 3) ピクセル座標に対するヤコビアン行列
        """
        device = K.device
        dtype = K.dtype

        # 各画像の焦点距離 fx, fy を抽出 (M,1)
        fx = K[:, 0, 0].unsqueeze(1)  # (M,1)
        fy = K[:, 1, 1].unsqueeze(1)  # (M,1)

        # 各画像における点群座標 (M,N)
        X = XYZ[..., 0]  # (M,N)
        Y = XYZ[..., 1]
        Z = XYZ[..., 2].clamp_min(10**(-2))  # Z=0除外防止

        # 出力テンソル確保
        M, N = XYZ.shape[:2]
        J = torch.zeros((M, N, 2, 3), dtype=dtype, device=device)

        # 各成分をブロードキャストで一括計算
        J[..., 0, 0] = fx / Z
        J[..., 0, 1] = 0
        J[..., 0, 2] = -fx * X / (Z ** 2)
        J[..., 1, 0] = 0
        J[..., 1, 1] = fy / Z
        J[..., 1, 2] = -fy * Y / (Z ** 2)

        return J
    
    #長方形の座標テンソルを計算する
    def make_rect_points(a, b):
        xs = torch.arange(a[0], b[0] + 1, device=a.device,dtype=torch.int32)
        ys = torch.arange(a[1], b[1] + 1, device=a.device,dtype=torch.int32)
        xx, yy = torch.meshgrid(xs, ys, indexing='xy')
        return torch.stack((xx.flatten(), yy.flatten()), dim=1)  # (num_points, 2)
    
    def make_rect_points_batch(box_start: torch.Tensor, box_end: torch.Tensor):
        """
        box_start, box_end: (N, 2) テンソル
        各行 i に対して [x_min, y_min], [x_max, y_max]
        それぞれの矩形に含まれる全座標点をまとめて返す
        """
        x_min, y_min = box_start[:, 0], box_start[:, 1]
        x_max, y_max = box_end[:, 0], box_end[:, 1]

        # 各矩形の幅と高さ
        widths  = x_max - x_min + 1
        heights = y_max - y_min + 1

        # 各矩形ごとの総点数
        num_points = (widths * heights).to(torch.int64)
        offsets = torch.cat([
            torch.zeros(1, dtype=torch.int64, device=box_start.device),
            torch.cumsum(num_points[:-1], dim=0)
        ])
        print(num_points.dtype, num_points)
        print(torch.cumsum(num_points[:-1], dim=0))
        print(offsets)

        total_points = int(num_points.sum().item())  # ← ここだけvmap外なのでitem()使用可

        # 出力テンソルを準備
        coords = torch.empty((total_points, 2), dtype=torch.int32, device=box_start.device)

        # forループを残すが軽量（数千〜数万矩形程度ならGPUでも高速）
        # vmap禁止エリアを避け、全テンソル操作で処理
        for i in range(box_start.shape[0]):
            xs = torch.arange(x_min[i], x_max[i] + 1, device=box_start.device, dtype=torch.int32)
            ys = torch.arange(y_min[i], y_max[i] + 1, device=box_start.device, dtype=torch.int32)
            xx, yy = torch.meshgrid(xs, ys, indexing="xy")
            rect = torch.stack((xx.flatten(), yy.flatten()), dim=1)
            coords[offsets[i]:offsets[i] + num_points[i]] = rect

        return coords

    
    def make_rect_points_parallel(start: torch.Tensor, end: torch.Tensor):
        """
        各 (x_min, y_min), (x_max, y_max) から長方形座標を並列生成
        並列処理計算量オーダー？ メモリ O(Σ w_i h_i)
        """
        device = start.device
        wh = end - start + 1  # 各矩形の幅・高さ
        widths, heights = wh[:, 0], wh[:, 1]
        N = start.shape[0]

        # 各矩形内の総点数
        num_points = widths * heights
        total_points = int(num_points.sum().item())

        # --- 各矩形に対応するgroup idを作る ---
        rect_ids = torch.repeat_interleave(torch.arange(N, device=device), num_points)

        # --- 各グループ内の「ローカル座標インデックス」を割り当て ---
        # 0..w_i*h_i-1 → (iy, ix) = divmod(idx, w_i)
        local_idx = torch.arange(total_points, device=device)
        local_offset = torch.repeat_interleave(torch.cumsum(num_points, 0) - num_points, num_points)
        idx_in_rect = local_idx - local_offset

        ix = idx_in_rect % widths[rect_ids]
        iy = idx_in_rect // widths[rect_ids]

        # --- 実際の (x, y) 座標へ変換 ---
        x_coords = start[rect_ids, 0] + ix
        y_coords = start[rect_ids, 1] + iy

        return torch.stack((x_coords, y_coords), dim=1)
    
    #グループ化された累積積を計算する
    # def grouped_comprod(A, G):
        
    #     inv = torch.unique(G, return_inverse=True)[1]
    #     sorted_inv,index = torch.sort(inv)
    #     A_sorted = A[index]
    #     A_sorted_cumprod = torch.cumprod(A_sorted,dim=0)
    #     A_sorted_grouped_prod = torch_scatter.scatter_mul(A_sorted ,sorted_inv,dim=0)
    #     cumprod_max = torch_scatter.scatter_min(A_sorted_cumprod,sorted_inv,dim=0) / A_sorted_grouped_prod
    #     count = torch_scatter.scatter_add(torch.ones_like(sorted_inv,device="cuda",dtype=torch.float32), sorted_inv)
    #     cumprod_max_interleave = torch.repeat_interleave(cumprod_max, count, dim=0)
    #     return (A_sorted_cumprod / cumprod_max_interleave)[torch.argsort(index)]
        
    
    # def grouped_cumprod(A,G, G_unique = False):#G_uniqueはkeys = G[:,0] * (MAX_VAL + 1) + G[:,1]で変換
    #     # 例
    #     # A = torch.tensor([1, 2, 3, 4, 5, 6, 7], dtype=torch.float32, device="cuda")
    #     # G = torch.tensor([[1,1],
    #     #                 [1,2],
    #     #                 [1,1],
    #     #                 [1,2],
    #     #                 [1,3],
    #     #                 [1,1],
    #     #                 [1,3]])
    #     #結果
    #     #   tensor([ 1.0000,  2.0000,  3.0000,  8.0000,  5.0000, 18.0000, 35.0000])

    #     # 各行ベクトルを整数キーに変換 Gはint32以上の整数値でなければならない
    #     if(G_unique) :
    #         keys = G[:,0] * (10000 + 1) + G[:,1]
    #         # 例: tensor([1000100,  11000]) 最大で10000までしか対応できない
    #     else: 
    #         keys = G

    #     # key順にソート
    #     sorted_keys, sort_idx = torch.sort(keys)
    #     A_sorted = A[sort_idx]

    #     # keyの変化を検出して「グループ開始点」を求める
    #     key_change = torch.cat([torch.tensor([True], device=A.device),
    #                             sorted_keys[1:] != sorted_keys[:-1]])

    #     # グループごとの「セグメントID」を生成
    #     group_id = key_change.cumsum(dim=0) - 1  # 0始まり

    #     # グループごとの累積積をベクトル化で取る
    #     #    (同じgroup_idの中で累積積が進む)
    #     #    まず、group_idの境界で累積積をリセットするためにトリックを使う
    #     cum_A = torch.cumprod(A_sorted, dim=0)
    #     # しかし group_id ごとにリセットしたい → 差分を使う

    #     # (1) 同じgroup_idの前要素を1で埋めるマスク
    #     reset_mask = torch.cat([torch.tensor([0], device=A.device), (group_id[1:] != group_id[:-1]).int()])
    #     # (2) リセット位置を使って補正
    #     logA = torch.log(A_sorted + 1e-12)
    #     cumlog = torch.cumsum(logA * (1 - reset_mask.cumsum(0).diff(prepend=torch.tensor([0], device=A.device))), dim=0)
    #     cumprod_grouped = torch.exp(cumlog)

    #     # 元の順序に戻す
    #     unsort_idx = torch.argsort(sort_idx)
    #     return cumprod_grouped[unsort_idx] # 例 tensor([ 1.0000,  2.0000,  3.0000,  8.0000,  5.0000, 18.0000, 35.0000])
    
    #2×2専用の行列の逆行列を閉形式でも計算
    def invert_2x2_batch(A: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        A: (..., 2, 2) のテンソル（例: (画像数, 点群数, 2, 2)）
        eps: 正則化項（数値安定化のため）
        
        返り値: A_inv (..., 2, 2) 同じshape
        """
        assert A.shape[-2:] == (2, 2), "入力は2x2行列である必要があります"

        # 単位行列（eps * I）
        I = torch.eye(2, device=A.device, dtype=A.dtype).expand_as(A)
        A_reg = A   # 正則化

        # 各成分を取り出し
        a = A_reg[..., 0, 0]
        b = A_reg[..., 0, 1]
        c = A_reg[..., 1, 0]
        d = A_reg[..., 1, 1]

        # 行列式
        det = a * d - b * c

        # det が 0 に近いときは安定化
        det = det + eps

        # 逆行列の閉形式
        inv = torch.empty_like(A_reg)
        inv[..., 0, 0] =  d / det
        inv[..., 0, 1] = -b / det
        inv[..., 1, 0] = -c / det
        inv[..., 1, 1] =  a / det

        return inv
    
    # def split_by_cumsum_parallel(x, limit):
    #     csum = torch.cumsum(x, dim=0)
    #     # limit を超えた箇所を検出
    #     over = (csum > limit)
    #     # 超えた箇所で「バッチID」が増えるようなラベルを作る
    #     batch_ids = torch.cumsum(over.int(), dim=0)
    #     # ただし最初の要素の処理調整
    #     batch_ids = batch_ids - batch_ids.min()

    #     # 各バッチごとの要素数を数える
    #     unique_ids, counts = torch.unique(batch_ids, return_counts=True)
    #     return counts
    
    def split_by_cumsum_parallel(x, limit):
        
        csum = torch.cumsum(x, dim=0)
        
        # limitを超えるたびにリセットするためのオフセット計算
        # floor_divideにより、どの「グループ」に属するかを整数化
        group_id = torch.floor_divide(csum, limit)
        
        # 各グループごとの要素数をカウント
        _, counts = torch.unique(group_id, return_counts=True)
        return counts
    

    def render_gaussian_batch(
        mean_pixel_zsort, gausian_boxsize_zsort,
        L_d_zsort, opacity_zsort, variance_inverse_zsort,
        shape_width, shape_height
    ):
        pixel_image_batch = []

        for batch_i in range(mean_pixel_zsort.shape[0]):  # ← バッチ単位は必要
            # ---------- (1) マスク ----------
            gx = gausian_boxsize_zsort[batch_i, :, 0]
            gy = gausian_boxsize_zsort[batch_i, :, 1]
            mx = mean_pixel_zsort[batch_i, :, 0]
            my = mean_pixel_zsort[batch_i, :, 1]

            mask = (
                (gx != 0)
                & (mx - gx < shape_width)
                & (mx + gx > 0)
                & (my - gy < shape_height)
                & (my + gy > 0)
            )

            mean_pixel_batch = mean_pixel_zsort[batch_i][mask]
            gausian_boxsize = gausian_boxsize_zsort[batch_i][mask]
            L_d_batch = L_d_zsort[batch_i][mask]
            opacity_batch = opacity_zsort[batch_i][mask]
            variance_inv_batch = variance_inverse_zsort[batch_i][mask]

            # ---------- (2) start/end点 ----------
            box_start = torch.clamp(mean_pixel_batch - gausian_boxsize, min=0)
            box_end   = torch.clamp(mean_pixel_batch + gausian_boxsize, max=torch.tensor([shape_width, shape_height], device=mean_pixel_batch.device))
            box_wh    = (box_end - box_start + 1).to(torch.int32)
            box_sizes = torch.prod(box_wh, dim=1)

            # ---------- (3) rect座標生成（ベクトル化） ----------
            # 各ガウスの長方形範囲を展開する
            # 例: box_start[i]=[x0,y0], box_wh[i]=[w,h]
            # → rect = [x0+ix, y0+iy] for ix∈[0,w), iy∈[0,h)
            x_offsets = torch.arange(box_wh[:, 0].max(), device=box_wh.device)
            y_offsets = torch.arange(box_wh[:, 1].max(), device=box_wh.device)
            grid_x, grid_y = torch.meshgrid(x_offsets, y_offsets, indexing="xy")
            grid = torch.stack([grid_x, grid_y], dim=-1)  # (Wx, Wy, 2)

            # 長方形ごとにmask化
            mask_x = (x_offsets[None, :] < box_wh[:, 0, None])
            mask_y = (y_offsets[None, :] < box_wh[:, 1, None])
            mask2d = mask_x[:, :, None] & mask_y[:, None, :]
            rects = (box_start[:, None, None, :] + grid[None, :, :, :])
            rects = rects[mask2d].view(-1, 2)  # (Σ w_i*h_i, 2)

            # ---------- (4) ベクトル化ガウスカーネル計算 ----------
            # 各ガウスに属するpixelのインデックス
            group_ids = torch.repeat_interleave(
                torch.arange(len(box_sizes), device=rects.device), box_sizes
            )
            diff = rects - mean_pixel_batch[group_ids]
            diff = diff.unsqueeze(1)  # (N,1,2)
            var = variance_inv_batch[group_ids]  # (N,2,2)
            tmp = torch.bmm(diff, var)           # (N,1,2)
            G = torch.exp(-0.5 * (tmp * diff).sum(dim=(1,2), keepdim=True))  # (N,1)

            # ---------- (5) 不透明度合成 ----------
            op = opacity_batch[group_ids]
            Ld = L_d_batch[group_ids]
            unti = 1 - op * G.squeeze()
            keys = rects[:,0] * (shape_width+1) + rects[:,1]
            T0 = Utilities.grouped_cumprod(unti, keys) / unti.clamp(min=1e-8)

            pixel_batch = T0[:,None] * Ld * op * G

            # ---------- (6) scatter加算 ----------
            pixel_img = torch.zeros((shape_height+1, shape_width+1, 3), device=rects.device)
            pixel_img.index_put_(
                (rects[:,1].long(), rects[:,0].long()),
                pixel_batch,
                accumulate=True
            )

            pixel_image_batch.append(pixel_img)

        return torch.stack(pixel_image_batch, dim=0)
    
    def get_expon_lr_func(
        lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
    ):
        """
        Copied from Plenoxels

        Continuous learning rate decay function. Adapted from JaxNeRF
        The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
        is log-linearly interpolated elsewhere (equivalent to exponential decay).
        If lr_delay_steps>0 then the learning rate will be scaled by some smooth
        function of lr_delay_mult, such that the initial learning rate is
        lr_init*lr_delay_mult at the beginning of optimization but will be eased back
        to the normal learning rate when steps>lr_delay_steps.
        :param conf: config subtree 'lr' or similar
        :param max_steps: int, the number of steps during optimization.
        :return HoF which takes step as input
        """

        def helper(step):
            if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
                # Disable this parameter
                return 0.0
            if lr_delay_steps > 0:
                # A kind of reverse cosine decay.
                delay_rate = lr_delay_mult + (1 - lr_delay_mult) * torch.sin(
                    0.5 * torch.pi * torch.clip(step / lr_delay_steps, 0, 1)
                )
            else:
                delay_rate = 1.0
            t = torch.clip(step / max_steps, 0, 1)
            log_lerp = torch.exp(torch.log(lr_init) * (1 - t) + torch.log(lr_final) * t)
            return delay_rate * log_lerp

        return helper

   