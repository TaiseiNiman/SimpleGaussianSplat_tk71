import torch 
import sys
print("Executable:", sys.executable)
print("Site-packages path:", sys.path)
# if torch.cuda.is_available():
#     print("true")
# else:
#     print("false")
 
# print(torch.__version__)
# print(torch.version.cuda)
# print(torch.backends.cudnn.version())
# print(torch.cuda.is_available())  

print(torch.cuda.get_device_properties(0))
 


def grouped_cartesian_product_efficient(x: torch.Tensor, y: torch.Tensor, group: torch.Tensor):
    """
    グループごとのデカルト積 (完全ベクトル化, O(Σ M_i²) メモリ)
    x, y : (N,)
    group : (N,) 連続または非連続のグループID (整数)
    """
    # ---- グループ分布 ----
    uniq, counts = torch.unique_consecutive(group, return_counts=True)
    offsets = torch.cat((torch.tensor([0], device=x.device), torch.cumsum(counts, 0)))

    # ---- 各グループ i の (M_i × M_i) の組み合わせインデックスを構成 ----
    # 各ブロックのサイズは M_i^2
    M2 = counts * counts
    total_pairs = int(M2.sum().item())

    # 各グループごとにインデックス範囲を連続配置
    # 各ブロック内で (i,j) = (0..M_i-1, 0..M_i-1)
    inner_idx = torch.arange(total_pairs, device=x.device)
    group_id = torch.repeat_interleave(torch.arange(len(counts), device=x.device), M2)

    # グループ i 内でのオフセットを求める
    inner_idx_in_group = inner_idx - torch.repeat_interleave(torch.cumsum(M2, 0) - M2, M2)
    Mi = counts[group_id]

    i_idx = inner_idx_in_group // Mi
    j_idx = inner_idx_in_group %  Mi

    # グループごとのオフセットを加算してグローバルインデックス化
    start = offsets[group_id]
    xi = x[start + i_idx]
    yj = y[start + j_idx]

    return torch.stack((xi, yj), dim=1)

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

