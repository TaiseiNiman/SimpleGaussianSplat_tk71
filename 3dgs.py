# ================================================================
#  Gaussian Splatting (Unofficial Implementation)
#  Copyright (c) 2025 Taisei Kuwana
#  Master's Student (M1), Graduate School of Engineering,
#  Nagoya Institute of Technology, Japan, cmk14002@ict.nitech.ac.jp
#  Licensed under the MIT License.
# ================================================================

import pycolmap
import torch
import numpy as np
import os
from PIL import Image
from torchvision import transforms
import scipy
from torch.utils.data import DataLoader, Dataset
import kornia.metrics as metrics
import math

#メイン関数の定義
class Program():
    #コールバック
    def __call__(self):
        self.main()
    #ユーザ入力の処理.
    def __init__(self):
        print("このプログラムはガウシアンスプラッティングのpython非公式実装(cudaカーネル不使用).")
        print("このプログラムはコンソールアプリケーションとして動作します.")
        print("この実装は,colmap点群データの色情報を使用しません.")
    #プログラム本体
    def main(self):
        print("学習を開始します.")
        #ここにプログラムを書く.
        #コントロールクラスのインスタンス作成
        control = Control()
        #学習開始
        control.learning()
        #学習結果を出力する画面の表示


#main関数呼び出し
Program()()

#---------------------------------------これより下はクラスの定義------------------------------------------
#コントロールクラス
class Control():
    def __init__(self):
        self._mode = input("簡易モード(細かくパラメータ指定しない)を選びますか?(Y/N)") == ("Y" or "y" or "YES" or "yes")
        if(self.mode) : #簡易モード すでにユーザがパラメータを指定しているならそちらが優先される
            self.learning_numbers = 35000
            self.batch_size_rate = 0.1
            self.uclid_mean = 3
            self.o_init = -1.0
            self.loss_lamda = 0.2
            self.densify_until_iter = 0
            self.grad_delta_upper_limit = 1e-12
            self.limit_upper_grad = 0.005
            self.dense_percent = 0.1
            self.prunning_opacity_min = 0.01
            self.reset_opacity_min = 0.01
    #modeを変更する
    def changing_mode(self,mode) : # mode = True -> 簡易モード
        self._mode = mode
    #学習
    def learning(self):
        try:
            self.colmap_root_dir = input("colmapのディレクトリパスを指定してください:") or self.colmap_root_dir
            initial_xyz_tensor, P, K, wh, image_samples = Load_colmap_data_from_binaries(self.colmap_root_dir).convert_to_tensors()
            self.batch_size_rate = input("学習画像全体に対するバッチ数の割合を指定してください(標準0.1):") or self.batch_size_rate if self.mode != True else self.batch_size_rate
            self.batch_size = round(image_samples.shape[0] * self.batch_size_rate)
            self.learning_numbers = input("学習回数を指定してください(標準35000):") or self.learning_numbers if self.mode != True else self.learning_numbers
            #パラメータ初期化
            self.gaus_mean = initial_xyz_tensor
            self.gaus_point_numbers = initial_xyz_tensor.shape[0]
            self.variance_q = torch.zeros(self.gaus_point_numbers,4)
            self.variance_q[:,3] = 1
            self.uclid_mean_point_number = input("分散スケール初期値のユークリッド平均で用いる点群の数を指定してください(標準3):") or self.uclid_mean_point_number if self.mode != True else self.uclid_mean_point_number
            self.variance_scale = torch.zeros(self.gaus_point_numbers,3)
            self.variance_scale = torch.log_(Utilities.kyori(self.uclid_mean_point_number,self.gaus_mean))
            self.o_init = input("不透明度αはシグモイド関数σ(o)を介して定められる.oの初期値を指定してください(標準-1.0から-5.0程度):") or self.o_init if self.mode != True else self.o_init
            self.gaus_point_o = torch.zeros(self.gaus_point_numbers,1) + self.o_init
            #損失関数の係数λの初期化
            self.loss_lamda = input("損失関数の係数λの初期値を指定してください(標準0.2):") or self.loss_lamda if self.mode != True else self.loss_lamda
            #ここですべてのユーザ入力値を指定
            self.densify_until_iter = input("ガウシアンの複製と分割を終了するイテレーション数を指定してください(標準):") or self.densify_until_iter if self.mode != True else self.densify_until_iter
            self.densify_from_iter = input("ガウシアンの複製と分割を開始するイテレーション数を指定してください(標準):") or self.densify_from_iter if self.mode != True else self.densify_from_iter
            self.densification_interval = input("ガウシアンの複製と分割のイテレーション間隔を指定してください(標準):") or self.densification_interval if self.mode != True else self.densification_interval
            self.opacity_reset_interval_per_densification = input("不透明度のリセット間隔をガウス分割・複製間隔に対する割合で指定してください(標準0 = リセットしない):") or self.opacity_reset_interval_per_densification if self.mode != True else self.opacity_reset_interval_per_densification
            self.grad_delta_upper_limit = input("勾配値の変化を検出する上限の値を指定してください(標準1e-12):") or self.grad_delta_upper_limit if self.mode != True else self.grad_delta_upper_limit
            self.limit_upper_grad = input("ガウス分割と複製を行う上限値(標準0.005):") or self.grad_delta_upper_limit if self.mode != True else self.grad_delta_upper_limit
            self.prunning_opacity_min = input("削除を行う不透明度の下限値(標準0.01):") or self.prunning_opacity_min if self.mode != True else self.prunning_opacity_min
            self.dense_percent = input("???を指定してください(標準0.1):") or self.grad_delta_upper_limit if self.mode != True else self.grad_delta_upper_limit
            self.reset_opacity_min = input("リセットする不透明度の値(標準0.01):") or self.reset_opacity_min if self.mode != True else self.reset_opacity_min
            #モデルインスタンス作成
            self.GS_model_param = GS_model_with_param(self.gaus_mean,self.variance_q,self.variance_scale,self.gaus_point_o,self.grad_delta_upper_limit)
            #イテレーション開始
            learning_numbers_per_epoch = self.learning_numbers / self.image_samples.shape[0] #１エポックあたりの学習回数
            
            for iter_i in iter(torch.arange(round(learning_numbers_per_epoch))):
                #
                it = DataLoader(
                GS_dataset(P,K,wh,image_samples),   # Datasetのインスタンス
                batch_size=self.batch_size,     # バッチサイズ（1回に取り出すデータ数）
                shuffle=True,      # データのシャッフル（エポックごと）
                num_workers=2,     # 並列でデータをロードするスレッド数
                drop_last=False,   # 最後の中途半端なバッチを捨てるか
            )
                for batch_i, (it_P,it_K,it_wh,it_image_sample) in enumerate(it):
                    #ガウシアンスプラッティングによる画像の出力
                    model_images = self.GS_model_param(it_P,it_K,it_wh)
                    #損失関数を計算
                    loss_d_ssim =  1 - metrics.ssim(model_images, it_image_sample, max_val=1.0, window_size=11).mean()
                    loss_1 = torch.nn.functional.l1_loss(model_images, it_image_sample, reduction='mean')
                    self.GS_model_param.train_step((1-self.loss_lamda) * loss_1 + self.loss_lamda * loss_d_ssim)
                    #ガウシアンの複製・分割・削除と不透明度のリセットを行う
                    # イテレーション数を計算
                    iteration = iter_i * math.Ceil(self.image_samples.shape[0] / self.batch_size) + batch_i + 1
                    if iteration <= self.densify_until_iter:
                        #

                        if iteration >= self.densify_from_iter and (iteration - self.densify_from_iter) % self.densification_interval == 0:
                            self.GS_model_param.densify_and_clone(it.get_camera_extent())
                        
                        if (iteration - self.densify_from_iter) % (self.opacity_reset_interval_per_densification or math.nan) == 0 or iteration == self.densify_from_iter:
                            self.GS_model_param.opacity = torch.sigmoid_(self.reset_opacity_min)
                    

        except Exception as e:
            print("エラー:", e)
            print("初めからやり直します.入力は保存されています.問題のある入力のみ変更してください.")
            self.learning()


#3dgsデータセットの定義
class GS_dataset(torch.utils.data.Dataset):
    def __init__(self,P,K,wh,image_sample):
        self.P = P
        self.K =  K
        self.wh =  wh
        self.image_sample =  image_sample
    def __len__(self):
        return 4
    def __getitem__(self):
        return [self.P, self.K, self.wh, self.image_sample]
    def get_camera_extent(self): #カメラの平均位置から最も遠い位置のカメラとの距離を求める
        return 0

#入力と出力のテンソルのshapeを必ず同じ形にする
class Utilities():
    
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
        gaus = np.sqrt(np.sum((cloud[None,:,:] - cloud[:,None,:])**2,axis=2,keepdims=False))
        #行ベクトルを昇順にソート
        gaus.sort(axis=1)
        #近傍なn個の点のユークリッド距離平均を計算
        uclid_n = np.mean(gaus[:,n],axis=1,keepdims=True)
        return uclid_n # -> (n,1)
    
    #方向ベクトルから球面調和関数を計算
    def direction_to_spherical_harmonics(d: np.ndarray, L_max: int = 1) -> np.ndarray:
        """
        単位方向ベクトル d から、指定された最大次数 (L_max) までの
        球面調和関数 (SH) の基底関数の値を計算する。

        Args:
            d (np.ndarray): 単位方向ベクトル (x, y, z)。
            L_max (int): SHの最大次数 (例: 1, 2, 3)。

        Returns:
            np.ndarray: 計算されたSH基底関数の値の配列 (総係数数 x 1)。
        """
        #shapeの形状を取得
        d0,d1,d2 = d.shape
        # 1. 方向ベクトルの正規化
        norm = np.linalg.norm(d,axis=1,keepdims=True) 
        norm[norm == 0] = 1
        d = d / norm   
        
        # 2. 直交座標 (x, y, z) を球面座標 (theta, phi) に変換
        # 慣例として、theta (極角) は z軸からの角度 [0, pi]
        # phi (方位角) は x軸からの角度 [0, 2*pi]
        
        # phi: x-y平面での角度 (numpy.arctan2(y, x))
        phi = np.array(np.arctan2(d[:,:,1], d[:,:,0])).reshape(d0,d1)
        
        # theta: z軸からの角度 (numpy.arccos(z))
        theta = np.array(np.arccos(d[:,:,2])).reshape(d0,d1) 

        # 3. SH基底関数の値を計算し、格納
        sh_values = []
        
        for l in range(L_max + 1):
            for m in range(-l, l + 1):
                # sph_harm(m, l, phi, theta) の順で引数を渡す
                # scipy.special.sph_harm は複素数を返すため、実部 (real) のみを使用
                sh_value = scipy.special.sph_harm(m, l, phi, theta).real
                sh_values.append(sh_value)

        return np.array(sh_values).reshape(d0,-1,d1) #(基底関数,画像,ガウス)→(画像,基底関数,ガウス)

#学習モデルとパラメータモデル 
class GS_model(torch.Module):
    #optimizerの変更
    def changing_optimizer(self,lr): #lrはテンソル
        self._optimizer = torch.optim.SGD(self.parameters(recurse=False), lr=lr)
    #パラメータのrequired_gradを一括変更
    def changing_required_grad(self,bool): #bool = Trueなら勾配計算する,Falseなら計算しない
        for param in iter(self.named_parameters(recurse=False)):
            param.required_grad(bool)
    #パラメータのgradを一括でリセット
    def reset_grad(self):
        self._optimizer.zero_grad(set_to_none=True)
    #パラメータあるいはスーパーパラメータあるいはウルトラパラメータの最適化
    def train_step(self, loss, count_param):
            loss.backward()
            temp = torch.clone(self[count_param].grad)
            self[count_param].grad -= self["__"+count_param+"_grad_before_one"]
            self._optimizer.step()
            self[count_param].grad = temp
            self.param_iter_update(count_param)
            self._optimizer.zero_grad(seto_to_none=False)
            return self
    #パラメータのgradの回数の更新を行う
    def param_iter_update(self,parameter_names):
        p = self.named_parameters(recurse=False)
        for name,key in iter(p) :
            if key.grad != None and name in parameter_names :
                grads_delta_norm = key.grad.norm(dim=1) - self["__"+name+"_grads_before_one"].norm(dim=1) if self[name+"_grads_before_one"] != None else key.grad.norm(dim=1)
                mask_norm = (grads_delta_norm > self.grad_delta_upper_limit)
                self["__"+name+"_grads_iter"] = self["__"+name+"_grads_iter"] + mask_norm.int() if self["__"+name+"_grads_iter"] != None else mask_norm.int()
                self["__"+name+"_grads_before_one"] = key.grad
    #パラメータのgradの積算の平均値のノルムを計算
    def param_grads_per_iter_norm(self,param_name): #param_name = パラメータ名
        #
        return (self[param_name] / self["__"+param_name+"_grads_iter"][:,None]).norm(dim=1)[:,None]
        
        
class GS_model_with_ultra_param(GS_model):
    def __init__(self, grad_delta_upper_limit,grad_threshold,percent_dense,lr=0.1):
        self.grad_delta_upper_limit = torch.Parameter(grad_delta_upper_limit)
        self.grad_threshold = torch.Parameter(grad_threshold)
        self.percent_dense = torch.Parameter(percent_dense)
        self._optimizer = self.changing_optimizer(lr)
        #安全のために最初は勾配計算されないように設定
        self.changing_required_grad(False)


class GS_model_with_super_param(GS_model):
    def __init__(self, mean_lr, others_lr,prunning_min_opacity, lr=0.1):
        #self.mean_lr = torch.Parameter(mean_lr)
        #self.others_lr = torch.Parameter(others_lr)
        self.prunning_min_opacity = prunning_min_opacity
        self._optimizer = self.changing_optimizer(lr)
        #安全のために最初は勾配計算されないように設定
        self.changing_required_grad(False)


#コンポジションモデル,ウルトラ及びスーパーパラメータクラスをコンポジションで継承
class GS_model_with_param(GS_model):
    # mean -> (ガウス数,3(x,y,zの順))
    # variance_q -> (ガウス数,4(i,j,k,wの順))
    # variance_scale -> (ガウス数,3)
    # opacity -> (ガウス数,1)
    # color -> (ガウス数,基底関数の数,3(x,y,zの順))
    def __init__(self, mean, variance_q, variance_scale, opacity, grad_delta_upper_limit,grad_threshold,percent_dense, c_00=1.77, L_max=3, lr=0.1):
        self.ultra = GS_model_with_ultra_param(grad_delta_upper_limit,grad_threshold,percent_dense)
        self.super = GS_model_with_super_param(lr,lr)
        self.mean = torch.Parameter(mean)
        self.variance_q = torch.Parameter(variance_q)
        self.variance_scale = torch.Parameter(variance_scale)
        self.opacity = torch.Parameter(opacity)
        #sh学習係数を初期化
        self.color = torch.Parameter(torch.zeros(self.mean.size(0),(L_max+1)**2,3))
        self.color[:,0,:] = c_00 #ベース色のみ中間色に設定
        self._optimizer = self.changing_optimizer(lr)
        #
        self._L_max = L_max
        
    # def densify_and_split(self, grads_param_name, scene_extent, N=2):
    #     n_init_points = self.mean.shape[0]
    #     # Extract points that satisfy the gradient condition
    #     padded_grad = torch.zeros((n_init_points), device="cuda")
    #     selected_pts_mask = torch.where(self.param_grads_per_iter_norm(grads_param_name) >= self.ultra.grad_threshold, True, False)
    #     selected_pts_mask = torch.logical_and(selected_pts_mask,
    #                                           torch.max(torch.exp(self.variance_scale), dim=1).values > self.ultra.percent_dense*scene_extent)

    def densify_and_clone(self, grads_param_name, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(self.param_grads_per_iter_norm(grads_param_name) >= self.ultra.grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(torch.exp_(self.variance_scale), dim=1).values > self.ultra.percent_dense*scene_extent)
        
        self.mean[selected_pts_mask].grad = 0
        self.mean = torch.vstack(self.mean,self.mean[selected_pts_mask])
        #
        self.variance_q = torch.vstack(self.variance_q,self.variance_q[selected_pts_mask])
        self.variance_scale = torch.vstack(self.variance_scale,self.variance_scale[selected_pts_mask])
        self.opacity = torch.vstack(self.opacity,self.opacity[selected_pts_mask])
        self.color = torch.vstack(self.color,self.color[selected_pts_mask])

    def densify_and_prune(self, extent):

        prune_mask = (torch.sigmoid_(self.opacity) < self.super.prunning_min_opacity).squeeze()

        big_points_ws = torch.max(torch.exp_(self.variance_scale), dim=1).values > 0.1 * extent
        prune_mask = torch.logical_or(prune_mask, big_points_ws)
        #ガウスを削除
        self.mean = self.mean[prune_mask == False]
        self.variance_q = self.variance_q[prune_mask == False]
        self.variance_scale = self.variance_scale[prune_mask == False]
        self.opacity = self.opacity[prune_mask == False]
        self.color = self.color[prune_mask == False]

        torch.cuda.empty_cache()
    
    # P -> (画像数,3,4)
    # K -> (画像数,3,3)
    # wh -> (画像数,2(width,heightの順))
    def forward(self, input, super_param=[0.1,0.1]):#input = 入力, super_param = 超パラメータ
        P, K, wh = input
        mean_lr, others_lr = super_param
        #shape_image = 画像数, shape_gausian = ガウス分布の数
        shape_gausian, shape_image, shape_wh = [self.mean.shape[0], P.shape[0], wh[0,0]*wh[0,1]]

        #ガウス位置のカメラ座標mean_cameraを計算.←ガウスの位置meanをカメラ座標系に座標変換
        m1, P1 = [shape_gausian,shape_image]
        homo_mean = np.hstack([self.mean,np.ones((m1,1))])[None, :, None, :] #同次座標変換
        mean_camera = (homo_mean @ np.transpose(P,(0,2,1))[:, None, :, :]).reshape(P1,m1,3) #カメラ座標xyzに変換
        #ガウス位置を画像面に投影(線形近似ではない)
        mean_pixel_homo = (mean_camera @ np.transpose(K,(0,2,1))[:, None, :, :]).reshape(P1,m1,3)
        mean_pixel = mean_pixel_homo[:,:,0:2] / mean_pixel_homo[:,:,2][mean_pixel_homo != 0]

        #共分散行列の回転行列をクォータニオンから計算
        #単位クォータニオンに正則化
        q = self.variance_q/np.sqrt(np.sum(self.variance_q**2,axis=1,keepdims=True))
        #回転行列に変換
        rotate = scipy.spatial.transform.rotation.from_quat(q)
        #共分散行列のスケールをvariance_scaleから計算
        #スケールの正則化
        s = np.exp(self.variance_scale)
        #スケールの各成分を対角に配置する対角行列を計算
        # (1,3,3) * (ガウス数,1,3) - > (ガウス数,3,3)
        s_diag = np.eye(3)[None,:,:] * s[:,None,:]
        #共分散行列を計算
        variance = rotate @ s_diag @ np.transpose(s_diag,(0,2,1)) @ np.transpose(rotate,(0,2,1))
        #共分散行列をカメラ座標系に変換
        variance_camera = P[:,None,:,0:3] @ variance[None,:,:,:] @ np.transpose(P,(0,2,1))[:,None,0:3,:]
        #カメラ座標系からピクセル座標形に線形近似するヤコビ行列Jを計算
        J = np.zeros(shape_image,shape_gausian,2,3)
        #ヤコビアン行列Jの(0,0),(0,2),(1,1),(1,2)成分を計算
        # J -> (画像,ガウス数,2,3)
        J[:,:,0,0] = K[:,None,:,:][:,:,0,0] / mean_camera[:,:,None,:][:,:,0,2][mean_camera!=0]
        J[:,:,1,1] = K[:,None,:,:][:,:,1,1] / mean_camera[:,:,None,:][:,:,0,2][mean_camera!=0]
        J[:,:,0,2] = -K[:,None,:,:][:,:,0,2] * mean_camera[:,:,None,:][:,:,0,0] / (mean_camera[:,:,None,:][:,:,0,2][mean_camera!=0]**2)
        J[:,:,1,2] = -K[:,None,:,:][:,:,1,2] * mean_camera[:,:,None,:][:,:,0,1] / (mean_camera[:,:,None,:][:,:,0,2][mean_camera!=0]**2)
        #共分散行列をピクセル座標系へ変換(線形近似)
        # variance_pixel -> (画像数,ガウス数,2,2)
        variance_pixel = J[:,None,:,:] @ variance_camera[None,:,:,:] @ np.transpose(J,(0,2,1))[:,None,:,:]
        #共分散行列を固有値分解
        #lamda -> (画像数,ガウス数,2)
        #vector -> (画像数,ガウス数,2,2)
        lamda,vector = np.linalg.eigh(variance_pixel)
        #画像面に投影された楕円ガウス分布の99.7%ボックスサイズwidth,heightを計算
        #gausian_boxsize -> (画像数,ガウス数,2(width,heightの順))
        gausian_boxsize = 3 * np.sqrt(vector**2 @ np.abs(lamda)[:,:,:,None]).reshape(shape_image,shape_gausian,2)

        #色L(d)を計算
        gamma_lm = Utilities.direction_to_spherical_harmonics(-mean_camera,self._L_max)
        dim_image, dim_basis, dim_gausian = gamma_lm.shape
        #球面調和関数を構成する-1. 基底関数を対角成分とする行列をもったテンソルを作る.
        # (1,1,基底,基底) * (画像,ガウス,基底,1) -> (画像,ガウス,基底,基底)
        diagonal_matrix = np.eye(dim_basis)[None,None,:,:] * gamma_lm[:,:,:,None]
        #-2. テンソル同士の行列積をかけてx,y,z成分ごとに合計し球面調和関数を得る.
        # (画像,ガウス,基底,基底) @ (1,ガウス,基底,3) -> (画像,ガウス,基底,3) 
        # -> (画像,ガウス,1,3) -> (画像,ガウス,3)
        L_d = np.sum(diagonal_matrix @ self.color[None,:,:,:],axis=2,keepdims=True).reshape(-1,dim_gausian,3)
        
        #ピクセル行列の計算のパイプライン
        #ピクセルマスクの作成
        #x_pixel,y_pixel
        x_pixel = np.arange(wh[0,0])
        y_pixel = np.arange(wh[0,1])
        #ピクセル格子点の座標行列を取得（デカルト積を用いる）
        pixel_uv = np.column_stack([np.repeat(x_pixel, len(y_pixel)), np.tile(y_pixel, len(x_pixel))])
        #broadcast
        x_pixel_broadcast = x_pixel[None,None,None,:]
        y_pixel_broadcast = y_pixel[None,None,:,None]
        pixel_uv_brod = pixel_uv[None,None,:,:]
        p_x = mean_pixel[:,:,0][:,:,None,:]
        gb_x = gausian_boxsize[:,:,0][:,:,None,:]
        p_y = mean_pixel[:,:,1][:,:,:,None]
        gb_y = gausian_boxsize[:,:,1][:,:,:,None]
        #ガウシアン99.7%ボックスのマスク
        gausian_box_mask_x = (p_x - gb_x <= x_pixel_broadcast) & (x_pixel_broadcast <= p_x + gb_x)
        gausian_box_mask_y = (p_y - gb_y <= y_pixel_broadcast) & (y_pixel_broadcast <= p_y + gb_y)
        #z深度=0を除外するマスク(幾何学的には画像面の無限遠点に相当する)
        z_zero_mask = (mean_pixel_homo[:,:,2] != 0)
        #固有値のいずれかが0となる場合を除外するマスク(正則にならないので逆行列を計算できない)
        #マスクの計算
        msk = gausian_box_mask_x & gausian_box_mask_y & z_zero_mask[:,:,None,None]
        
        #ガウスカーネルの計算
        mean_k = (pixel_uv_brod - mean_pixel[:,:,None,:])[:,:,:,None,:]
        variance_inverse_k = np.linalg.inv(variance_pixel)[:,:,None,:,:]
        Gaus_karnel = np.exp(-0.5 * mean_k @ variance_inverse_k @ np.transpose(mean_k,(0,2,1)))
        #深度zでパラメータを昇順にソート
        #深度インデックスの計算
        z_index = np.argsort(mean_camera[:,:,2],axis=1)
        #逆深度インデックスの計算
        z_index_reverse = np.argsort(z_index,axis=1)
        #batchのためのブロードキャスト配列の作成
        batch_index = np.arange(shape_image)[:,None]
        #不透明度パラメータをzソート
        opacity_batch_zsort = torch.sigmoid_(self.opacity)[None,:,:][np.zeros(shape_image),:,:][batch_index,z_index,:]
        #透過率T_oの計算
        #透明度(1-不透明度)の相乗を計算するためにopacity_batchをブロードキャストする.
        opacity_batch_broadcast = opacity_batch_zsort[:,None,:,:] + np.zeros(shape_image,shape_wh,shape_gausian,shape_gausian)
        #ガウスカーネルで重みづけ
        #ガウスカーネルをブロードキャスト 
        # !maskするときは,まずブロードキャストしてshapeを同じにするべきです.
        # !なぜなら,ブロードキャストとmaskを併用するとブロードキャストされる前にmaskされてしまう.
        # !また,A[mask] = (B[mask] + C[mask]) @ D[mask]...という形に計算される前にマスクをかけないと,
        # !maskを満足しない要素に対しても計算が行われる.maskの使用で生じる非連続なメモリ計算によるオーバーヘッドを考慮して,
        # 極端に疎なテンソルに対してはmaskをかける,そうでなければ可算や乗算のような単純な計算では全要素で計算するという方法を用いるべきです.
        # とりあえず全てmaskかけて計算した.テンソルとマスクの要素数を比較して割合を求めて,条件分岐する式を書き加えればよいだけ.
        Gaus_kernel_broadcast_to_opacity = np.broadcast_to(Gaus_karnel.reshape(shape_image,shape_wh,shape_gausian)[:,:,None,:],opacity_batch_broadcast.shape)
        msk_broadcast_to_opacity = np.broadcast_to(msk.reshape(shape_image,shape_wh,shape_gausian)[:,:,None,:],opacity_batch_broadcast.shape)
        opacity_kernel = np.zeros(shape_image,shape_wh,shape_gausian,shape_gausian)
        # ガウスカーネルをzソート←忘れてた.
        # ４階のテンソルに対して第3軸(つまり最後の軸)に対してz_indexで指定された順序でソートを行う
        # 非常に煩雑な計算で,これは修正する必要があると思う
        z_index_broadcast = np.broadcast_to(z_index[:,None,None,:],opacity_batch_broadcast)[:,:,0,:]
        z_index_broadcast_reverse = np.argsort(z_index_broadcast,axis=3)
        z_index_broadcast_0 = np.broadcast_to(np.arange(shape_image)[:,None,None], z_index_broadcast)
        z_index_broadcast_1 = np.broadcast_to(np.arange(shape_wh)[None,:,None], z_index_broadcast)
        Gaus_kernel_broadcast_to_opacity_zsort = Gaus_kernel_broadcast_to_opacity[z_index_broadcast_0,z_index_broadcast_1,:,z_index_broadcast]

        opacity_kernel[msk_broadcast_to_opacity] = np.tril(opacity_batch_broadcast,k=-1)[msk_broadcast_to_opacity] * Gaus_kernel_broadcast_to_opacity_zsort[msk_broadcast_to_opacity]
        #透過率T_oを計算しそれを逆zソートして元の順番に戻す.
        # T_o -> (画像数,ガウス数,1)
        T_o = np.broadcast_to(np.prod(1 - opacity_kernel,axis=3)[:,:,None,:],opacity_batch_broadcast)[z_index_broadcast_0,z_index_broadcast_1,:,z_index_broadcast_reverse]
        #ピクセル輝度pixelを計算
        #maskを適用するためにブロードキャスト
        #maskをA[mask]=B[mask]+C[mask]という形で計算すれば、maskで除外された要素は計算されないので、要素計算が重い場合に高速に計算できる。
        #たとえコードが長くなっても最適化のためにやるべきです。
        opacity_batch_broadcast_r = opacity_batch_broadcast[:,:,0,:][:,:,:,None] + np.zeros(shape_image,shape_wh,shape_gausian,3)
        L_d_broadcast_r = np.broadcast_to(L_d[:,None,:,:],opacity_batch_broadcast_r)
        Gaus_kernel_broadcast_to_opacity_r = np.broadcast_to(Gaus_kernel_broadcast_to_opacity[:,:,0,:][:,:,:,None],opacity_batch_broadcast_r)
        T_o_r = np.broadcast_to(T_o[:,:,0,:][:,:,:,None],opacity_batch_broadcast_r)
        msk_r = np.broadcast_to(msk_broadcast_to_opacity[:,:,0,:][:,:,:,None],opacity_batch_broadcast_r)
        pixel_r = np.zeros(opacity_batch_broadcast_r.shape)
        pixel_r[msk_r] = opacity_batch_broadcast_r[msk_r] * L_d_broadcast_r[msk_r] * Gaus_kernel_broadcast_to_opacity_r[msk_r] * T_o_r[msk_r]
        #ピクセル行列pixelの計算 pixel -> (画像数,ガウス数,チャンネル数(RGB=3),height,width)
        pixel = np.sum(pixel_r,axis=2).reshape(shape_image,3,wh[0,1],wh[0,0])
        #クリッピングする
        pixel_clipping = torch.clamp(pixel, min=0.0, max=1.0) 
        #戻り値を返す.
        return pixel_clipping

    
    def culling_param(self):
        
        return self
    def cloning_param(self):

        return self
    def splitting_param(self):

        return self


class Image_to_transform():
    def __init__(self,root_dir,path):
        try:
            # ルートパスと結合して絶対パスを構築
            full_path = os.path.join(root_dir, path)
            # 画像を読み込み
            self.img = Image.open(full_path)
            print(f"画像 ({path}) の読み込みに成功しました.")
            
        except Exception as e:
            print(f"画像 {path} の処理中にエラーが発生しました: {e}")  
            self.img = None  
    def convert_to_pil(self):#画像をPIL形式に変換する.
        self.img = transforms.ToPILImage()(self.img)
        return self
    def convert_to_torch_tensor(self):#画像をテンソル形式に変換
        self.img = transforms.ToTensor()(self.img)
        return self
    def get_data(self):
        return self.img
    def __call__(self):
        return self.get_data()
    

class Load_colmap_data():
    def __init__(self,root_dir):
        self.root_dir = root_dir
    def get_data(self):
        return [self.camera,self.images,self.points3d]

    def convert_to_tensors(self):
        #
        images_data, points3d_data, cameras_data = self.get_data()

        # 1. 初期点群 (ガウスの位置) の抽出と変換
        xyz_list = np.array([v.xyz for v in points3d_data.values()])
        
        # NumPyリストをPyTorchテンソルに変換
        initial_xyz_tensor = torch.from_numpy(xyz_list, dtype=torch.float32)
        print(f"Initial XYZ Tensor Shape: {initial_xyz_tensor.shape}") 
        
        # 2. データセットとして使うために,入力と比較画像をテンソルに変換 (PyTorchのDataLoaderで使うため)
        # ----------------------------------------------------
        R_matrix , T_vector, image_wh, image_K, image = []
        
        # ImagesデータからRとTを抽出し、行列に変換するカスタム関数が必要
        # (ここでは簡略化のため、qvecとtvecを直接リスト化)
        for img_id in images_data:
            image = images_data[img_id]
            # クォータニオンを回転行列に変換する処理（外部モジュールが必要）をここで行う
            R_matrix.append(pycolmap.qvec_to_rotmat(image.qvec))
            T_vector.append(image.tvec)
            # 1. 対応するカメラIDを取得
            cam_id = image.camera_id
            # 2. cameras_dataから内部パラメータオブジェクトを取得
            camera = cameras_data[cam_id]
            # 3. パラメータの抽出
            # COLMAPはレンズモデル（Pinhole, Radial, Simple_Radialなど）ごとにパラメータ形式が異なる.
            model_name = ['PINHOLE', 'SIMPLE_PINHOLE','SIMPLE_RADIAL','RADIAL','BROWN']
            if camera.model_name in model_name[1:3]:
                fx, cx, cy = camera.params[0:3]
                fy = fx
            else:
                fx, fy, cx, cy = camera.params[0:3]
            #画像パスを配列に追加
            image.append(Image_to_transform(self.root_dir,image.name).convert_to_pil().convert_to_torch_tensor()())
            #外部パラメータkを作成し配列に追加
            image_K.append([[fx,0,cx],[0,fy,cy],[0,0,1]])
            #画像の縦横を作成し配列に追加
            image_wh.append([camera.width,camera.height]) 

        #画像サンプル配列をテンソルに変換
        image_samples = torch.from_numpy(np.array(image),dtype=torch.float32)
        #回転と並進を結合しテンソルに変換
        P = torch.from_numpy(np.hstack((np.array(R_matrix),np.array(T_vector))),dtype=torch.float32)
        #外部パラメータKをテンソルに変換
        K = torch.from_numpy(np.array(image_K),dtype=torch.float32)
        #画像の縦横をテンソルに変換
        wh = torch.from_numpy(np.array(image_wh),dtype=torch.float32)

        return [initial_xyz_tensor, P, K, wh, image_samples]


class Load_colmap_data_from_binaries(Load_colmap_data):
    def __init__(self,root_dir):
        super().__init__(root_dir=root_dir)
        # ファイルパスの定義
        cameras_file = os.path.join(self.root_dir, 'cameras.bin')
        images_file = os.path.join(self.root_dir, 'images.bin')
        points3d_file = os.path.join(self.root_dir, 'points3D.bin')
        try:
            # cameras.bin: カメラ内部パラメータ（レンズモデル、焦点距離、主点など）
            cameras = pycolmap.read_cameras_binary(cameras_file)
            print(f"Cameras loaded: {len(cameras)} unique camera models found.")
            
            # images.bin: カメラ外部パラメータ（画像名、姿勢（Q/T）、対応するカメラID）
            images = pycolmap.read_images_binary(images_file)
            print(f"Images/Poses loaded: {len(images)} images found.")

            # points3D.bin: 3D点群データ（位置、色、観測情報）
            points3d = pycolmap.read_points3d_binary(points3d_file)
            print(f"3D Points loaded: {len(points3d)} points found.")

        except FileNotFoundError as e:
            print(f"Error: COLMAP file not found. Please check path and file existence: {e}")
            self.cameras,self.images,self.points3d = [None, None, None]
                    
        self.cameras,self.images,self.points3d = [cameras, images, points3d]

# ----------------------------------------------------
# 実行例（使用方法）
# ----------------------------------------------------
# COLMAPの出力フォルダを指定
# colmap_dir = 'path/to/colmap/output/0' 

# # データを読み込み
# cameras_data, images_data, points3d_data = load_colmap_data(colmap_dir)

# # データ構造の確認（例: 最初の画像の姿勢）
# if images_data:
#     # 最初の画像のキーは通常 1 から始まる
#     first_image_id = min(images_data.keys()) 
#     first_image = images_data[first_image_id]
#     
#     print("\n--- First Image Data ---")
#     print(f"Image Name: {first_image.name}")
#     print(f"Rotation (Quaternion): {first_image.qvec}")
#     print(f"Translation (T): {first_image.tvec}")
#     print(f"Camera ID: {first_image.camera_id}")