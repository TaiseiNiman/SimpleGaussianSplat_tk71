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
from uitility import Utilities


#3dgsデータセットの定義
class GS_dataset(torch.utils.data.Dataset):
    def __init__(self,P,K,wh,image_sample):
        self.P = P
        self.K =  K
        self.wh =  wh
        self.image_sample =  image_sample
    def __len__(self):
        return 4
    def __getitem__(self, idx):
        return [self.P[idx], self.K[idx], self.wh[idx], self.image_sample[idx]]
    def get_camera_extent(self): #カメラの平均位置から最も遠い位置のカメラとの距離を求める
        #カメラの並進ベクトルを取り出す
        camera_T = self.P[:,:,3]
        #カメラの平均位置
        camera_mean = torch.mean(camera_T,dim=0)
        #計算
        camera_max = torch.max((camera_mean[None,:] - camera_T).norm(dim=1)).item()
        return camera_max

#入力と出力のテンソルのshapeを必ず同じ形にする


#学習モデルとパラメータモデル 
class _GS_model(torch.nn.Module):
    #GS_model_with_paramのみ利用可能.
    def __init__(self,caller=None):#callerはキーワード引数
        super().__init__()
        if not isinstance(caller,GS_model_with_param):
            raise PermissionError("GS_model_with_paramのみ利用可能です.")
    #optimizerの変更
    def changing_optimizer(self,lr): #lrはテンソル
        self._optimizer = torch.optim.SGD(self.parameters(recurse=False), lr=lr)
    #パラメータのrequired_gradを一括変更
    def changing_required_grad(self,bool): #bool = Trueなら勾配計算する,Falseなら計算しない
        for param in iter(self.parameters(recurse=False)):
            param.requires_grad_(bool)
    #パラメータのgradを一括でリセット
    def reset_grad(self):
        self._optimizer.zero_grad(set_to_none=True)
    #勾配計算
    def backward(self, loss):
        #パラメーターの勾配計算オン
        self.changing_required_grad(True)
        #勾配計算
        loss.backward()
        #パラメーターの勾配計算オフ
        self.changing_required_grad(False)
        return self
    #パラメータあるいはスーパーパラメータあるいはウルトラパラメータの最適化
    def train_step(self):
        self._optimizer.step()
        self.reset_grad()
        return self
        
        
class _GS_model_with_ultra_param(_GS_model):
    def __init__(self, grad_delta_upper_limit,grad_threshold,percent_dense,variance_pixel_tile_max_width, lr=0.1, caller=None):
        #GS_model_with_paramのみ利用しないとエラーが出る.
        super().__init__(caller)
        self.grad_delta_upper_limit = torch.nn.Parameter(torch.tensor(grad_delta_upper_limit,device="cuda",dtype=torch.float32))
        self.grad_threshold = torch.nn.Parameter(torch.tensor(grad_threshold,device="cuda",dtype=torch.float32))
        self.percent_dense = torch.nn.Parameter(torch.tensor(percent_dense,device="cuda",dtype=torch.float32))
        self.variance_pixel_tile_max_width = torch.nn.Parameter(torch.tensor(torch.logit(variance_pixel_tile_max_width),device="cuda",dtype=torch.float32))
        self.changing_optimizer(lr)
        #最初は勾配計算されないように設定
        self.changing_required_grad(False)


class _GS_model_with_super_param(_GS_model):
    def __init__(self, mean_lr, others_lr,prunning_min_opacity, lr=0.1, caller=None):
        #GS_model_with_paramのみ利用しないとエラーが出る.
        super().__init__(caller)
        #self.mean_lr = torch.nn.Parameter(mean_lr)
        #self.others_lr = torch.nn.Parameter(others_lr)
        self.prunning_min_opacity = torch.nn.Parameter(torch.tensor(prunning_min_opacity,device="cuda",dtype=torch.float32))
        self.changing_optimizer(lr)
        #最初は勾配計算されないように設定
        self.changing_required_grad(False)


#コンポジションモデル,ウルトラ及びスーパーパラメータクラスをコンポジションで継承
class GS_model_with_param(_GS_model):
    # mean -> (ガウス数,3(x,y,zの順))
    # variance_q -> (ガウス数,4(i,j,k,wの順))
    # variance_scale -> (ガウス数,3)
    # opacity -> (ガウス数,1)
    # color -> (ガウス数,基底関数の数,3(x,y,zの順))
    def __init__(self, mean, variance_q, variance_scale, opacity, grad_delta_upper_limit,grad_threshold,percent_dense,prunning_min_opacity,variance_pixel_tile_max_width, c_00=1.77, L_max=3, lr=0.1):
        #インスタンスをはじく
        super().__init__(self)
        self.ultra = _GS_model_with_ultra_param(grad_delta_upper_limit,grad_threshold,percent_dense,variance_pixel_tile_max_width, caller=self)
        self.super = _GS_model_with_super_param(lr,lr,prunning_min_opacity,caller=self)
        self.mean = torch.nn.Parameter(mean)
        self.variance_q = torch.nn.Parameter(variance_q)
        self.variance_scale = torch.nn.Parameter(variance_scale)
        self.opacity = torch.nn.Parameter(opacity)
        #sh学習係数を初期化
        self.color = torch.zeros((self.mean.size(0),(L_max+1)**2,3),device="cuda",dtype=torch.float32)
        self.color[:,0,:] = c_00 #ベース色のみ中間色に設定
        self.color = torch.nn.Parameter(self.color)
        self.changing_optimizer(lr)
        self._L_max = L_max
        #パラメーターとして使用しないローカル変数
        self.mean_grads_norm = torch.zeros(self.mean.shape[0],device="cuda")
        self.mean_grads_iter = torch.clone(self.mean_grads_norm)
        #勾配計算されないように設定
        self.changing_required_grad(False)
        
    # def densify_and_split(self, grads_param_name, scene_extent, N=2):
    #     n_init_points = self.mean.shape[0]
    #     # Extract points that satisfy the gradient condition
    #     padded_grad = torch.zeros((n_init_points), device="cuda")
    #     selected_pts_mask = torch.where(self.param_grads_per_iter_norm(grads_param_name) >= self.ultra.grad_threshold, True, False)
    #     selected_pts_mask = torch.logical_and(selected_pts_mask,
    #                                           torch.max(torch.exp(self.variance_scale), dim=1).values > self.ultra.percent_dense*scene_extent)
    
    #パラメータのgradの回数の更新を行う
    #これは厳密には正しくない実装である.
    #なぜならば,損失関数にパラメーターテンソルの特定の成分が含まれていない場合,それに対応する勾配テンソルの成分には0が代入されるからである.
    #それに対して,勾配値が非常に小さな値つまりアンダーフローする場合は浮動小数点数のように最小正規化数が代入されるのではなくflush to zeroつまり0が代入される.
    #この両者を区別できないので,実際は非常に小さな勾配値が計算されたつまりその成分は損失関数の経路に存在していたにもかかわらず,勾配積算値の積算回数に含まれないことになる.
    #しかしながら,非常に小さな勾配値自体は勾配値を積算するときに無視できるはずだから,積算回数だけ減らされることになり本当の勾配値積算値の平均と同じかそれより大きくなり
    #ガウス分布の複製・分割をする方向に作用するので,計算が増加するが学習品質には影響を及ぼさないはずである.
    def param_iter_update(self):
        if self.mean.grad != None:
            grads_norm = self.mean.grad.norm(dim=1)
            self.mean_grads_norm += grads_norm
            self.mean_grads_iter += (grads_norm != 0).int()
    #パラメータのgradの積算の平均値のノルムを計算
    def param_grads_per_iter_norm(self): #param_name = パラメータ名
        #積算回数が0の成分のみ1に置き換える
        grads_iter = self.mean_grads_iter + (self.mean_grads_iter == 0).int()
        return self.mean_grad_norm / grads_iter
    
    def densify_and_clone(self, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(self.param_grads_per_iter_norm() >= self.ultra.grad_threshold, True, False)
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
    
    # P -> (画像数,3,4)
    # K -> (画像数,3,3)
    # wh -> (画像数,2(width,heightの順))
    def forward(self, P, K, wh):#input = 入力, super_param = 超パラメータ
        #shape_image = 画像数, shape_gausian = ガウス分布の数
        shape_gausian, shape_image, shape_wh, shape_width, shape_height = [self.mean.shape[0], P.shape[0], wh[0,0]*wh[0,1], wh[0,0], wh[0,1]]

        #ガウス位置のカメラ座標mean_cameraを計算.←ガウスの位置meanをカメラ座標系に座標変換
        m1, P1 = [shape_gausian,shape_image]
        Utilities.gpu_mem()
        homo_mean = torch.hstack((self.mean,torch.ones((m1,1),device="cuda")))[None, :, :] #同次座標変換
        Utilities.gpu_mem()
        mean_camera = homo_mean @ torch.transpose(P,1,2) #[:, None, :, :]).reshape(P1,m1,3) カメラ座標xyzに変換
        Utilities.gpu_mem()
        #ガウス位置を画像面に投影(線形近似ではない)
        mean_pixel_homo = mean_camera @ torch.transpose(K,1,2) #[:, None, :, :]).reshape(P1,m1,3)
        Utilities.gpu_mem()
        mean_pixel = mean_pixel_homo[:,:,0:2] / mean_pixel_homo[:,:,2][:,:,None].clamp_min(1e-8) #[mean_pixel_homo != 0]
        Utilities.gpu_mem()
        #共分散行列の回転行列をクォータニオンから計算
        #単位クォータニオンに正則化
        q = self.variance_q / torch.norm(self.variance_q,dim=1,keepdim=True).clamp_min(1e-8)
        #回転行列に変換
        Utilities.gpu_mem()
        rotate = Utilities.qvec_to_rotmat_torch_batch(q)
        #共分散行列のスケールをvariance_scaleから計算
        #スケールの正則化
        Utilities.gpu_mem()
        s = torch.exp(self.variance_scale)
        #スケールの各成分を対角に配置する対角行列を計算
        # (1,3,3) * (ガウス数,1,3) - > (ガウス数,3,3)
        Utilities.gpu_mem()
        s_diag = torch.eye(3,dtype=torch.float32,device="cuda")[None,:,:] * s[:,None,:]
        #共分散行列を計算
        Utilities.gpu_mem()
        variance = rotate @ s_diag @ torch.transpose(s_diag,1,2) @ torch.transpose(rotate,1,2)
        #共分散行列をカメラ座標系に変換
        Utilities.gpu_mem()
        variance_camera = P[:,None,:,0:3] @ variance[None,:,:,:] @ torch.transpose(P,1,2)[:,None,0:3,:]
        #カメラ座標系からピクセル座標形に線形近似するヤコビ行列Jを計算
        Utilities.gpu_mem()
        J = Utilities.pixel_jacobian_batch(K,mean_camera) 
        #ヤコビアン行列Jの(0,0),(0,2),(1,1),(1,2)成分を計算 torch.zeros(shape_image,(shape_gausian,2,3),device="cuda")
        # J -> (画像,ガウス数,2,3)
        Utilities.gpu_mem()
        # J[:,:,0,0] = K[:,None,:,:][:,:,0,0] / mean_camera[:,:,None,:][:,:,0,2][mean_camera!=0]
        # J[:,:,1,1] = K[:,None,:,:][:,:,1,1] / mean_camera[:,:,None,:][:,:,0,2][mean_camera!=0]
        # Utilities.gpu_mem()
        # J[:,:,0,2] = -K[:,None,:,:][:,:,0,2] * mean_camera[:,:,None,:][:,:,0,0] / (mean_camera[:,:,None,:][:,:,0,2][mean_camera!=0]**2)
        # J[:,:,1,2] = -K[:,None,:,:][:,:,1,2] * mean_camera[:,:,None,:][:,:,0,1] / (mean_camera[:,:,None,:][:,:,0,2][mean_camera!=0]**2)
        #共分散行列をピクセル座標系へ変換(線形近似)
        # variance_pixel -> (画像数,ガウス数,2,2)
        variance_pixel = J @ variance_camera @ torch.transpose(J,2,3)
        #J[:,None,:,:] @ variance_camera[None,:,:,:] @ torch.transpose(J,(0,2,1))[:,None,:,:]
        Utilities.gpu_mem()
        #共分散行列を固有値分解
        #lamda -> (画像数,ガウス数,2)
        #vector -> (画像数,ガウス数,2,2)
        lamda,vector = torch.linalg.eigh(variance_pixel.to("cpu"))
        vector = vector.to("cuda")
        lamda = lamda.to("cuda")
        Utilities.gpu_mem()
        #画像面に投影された楕円ガウス分布の99.7%ボックスサイズwidth,heightを計算
        #gausian_boxsize -> (画像数,ガウス数,2(width,heightの順))
        gausian_boxsize = 3 * torch.sqrt(vector**2 @ torch.abs(lamda)[:,:,:,None]).reshape(shape_image,shape_gausian,2)

        #色L(d)を計算
        Utilities.gpu_mem()
        gamma_lm = Utilities.direction_to_spherical_harmonics_torch(-mean_camera,self._L_max)
        dim_image, dim_gausian, dim_basis = gamma_lm.shape
        #球面調和関数を構成する-1. 基底関数を対角成分とする行列をもったテンソルを作る.
        # (1,1,基底,基底) * (画像,ガウス,基底,1) -> (画像,ガウス,基底,基底)
        Utilities.gpu_mem()
        diagonal_matrix = torch.eye(dim_basis,device="cuda",dtype=torch.float32)[None,None,:,:] * gamma_lm[:,:,:,None]
        #-2. テンソル同士の行列積をかけてx,y,z成分ごとに合計し球面調和関数を得る.
        # (画像,ガウス,基底,基底) @ (1,ガウス,基底,3) -> (画像,ガウス,基底,3) 
        # -> (画像,ガウス,1,3) -> (画像,ガウス,3)
        Utilities.gpu_mem()
        L_d = torch.sum(diagonal_matrix @ self.color[None,:,:,:],dim=2,keepdim=True).reshape(-1,dim_gausian,3)
        
        #
        variance_inverse = Utilities.invert_2x2_batch(variance_pixel)
        
        #ピクセル値算出のためにzソート
        z_index = torch.argsort(mean_camera[:,:,2],dim=1)
        image_index = torch.arange(shape_image)[:,None]
        #
        mean_camera_zsort = mean_camera[image_index,z_index,:]
        #
        opacity_zsort = torch.sigmoid_(self.opacity).unsqueeze(0).repeat(shape_image,1,1)[image_index,z_index,:]
        mean_pixel_zsort = mean_pixel[image_index,z_index,:]
        variance_inverse_zsort = variance_inverse[image_index,z_index,:]
        L_d_zsort = L_d[image_index,z_index,:]
        gausian_boxsize_zsort = gausian_boxsize[image_index,z_index,:]
        print(f"opacity.shape = {opacity_zsort.shape}")
        print(f"mean_pixel_zsort.shape = {mean_pixel_zsort.shape}")
        print(f"variance_inverse_zsort.shape = {variance_inverse_zsort.shape}")
        print(f"L_d_zsortity.shape = {L_d_zsort.shape}")
        print(f"gausian_boxsize_zsort.shape = {gausian_boxsize_zsort.shape}")
        
        Utilities.gpu_mem()
        pixel_image_batch =[]
        ptile_maxwidth = torch.sqrt(shape_wh)*torch.sigmoid(self.variance_pixel_tile_max_width)
        for batch_i in range(shape_image):
            #
            pixel_gausian_batch = []
            index = 0
            #画像に映らないガウシアンを取り除く
            gausian_boxsize_batch = gausian_boxsize_zsort[batch_i,:,:]
            box_startpoint_x = (mean_pixel_zsort[batch_i,:,:][:,0] - gausian_boxsize_batch[:,0]).clamp(max=shape_width,min=0)
            box_startpoint_y = (mean_pixel_zsort[batch_i,:,:][:,1] - gausian_boxsize_batch[:,1]).clamp(max=shape_height,min=0)
            box_startpoint = torch.hstack((box_startpoint_x[:,None],box_startpoint_y[:,None]))
            box_endpoint_x = (mean_pixel_zsort[batch_i,:,:][:,0] + gausian_boxsize_batch[:,0]).clamp(max=shape_width,min=0)
            box_endpoint_y = (mean_pixel_zsort[batch_i,:,:][:,1] + gausian_boxsize_batch[:,1]).clamp(max=shape_height,min=0)
            box_endpoint = torch.hstack((box_endpoint_x[:,None],box_endpoint_y[:,None]))
            Utilities.gpu_mem()
            torch.cuda.empty_cache()
            #各ガウシアンの長方形ピクセル座標を2次元配列(N,2)として返す
            box_startpoint_int = box_startpoint.to(torch.int32)
            box_endpoint_int = box_endpoint.to(torch.int32)
            
            #マスク
            #ガウシアンのvariance_pixelが99.7%タイルが画像に対してある幅以上となるものは全てその幅まで縮小する.
            # 理由：variance_pixelは線形一次近似であって,ガウス中心に対して十分近傍なピクセル値に対してしか有効でないはずだからである.
            # 具体的に画像に対して何パーセントとすべきかは、これ自体をハイパーパラメータとして学習するか公式の実装コードを読み解くしかない.
            
            boxsize_xy = (box_endpoint_int - box_startpoint_int).clamp(max=ptile_maxwidth)
            boxsize_mask = (boxsize_xy[:, 0] != 0) | (boxsize_xy[:, 1] != 0) 
            boxsize = torch.prod((boxsize_xy[boxsize_mask] + 1), dim=1) 
            print(f"boxsize.shape = {boxsize.shape}")
            mean_pixel_batch = mean_pixel_zsort[batch_i,:,:][boxsize_mask]
            L_d_batch = L_d_zsort[batch_i,:,:][boxsize_mask]
            opacity_batch = opacity_zsort[batch_i,:,:][boxsize_mask]
            variance_inverse_batch = variance_inverse_zsort[batch_i,:,:][boxsize_mask]
            #バッチ数を1024^3*6 / 80 = 0.075GB以下となるように確定する
            gausian_batch = torch.cumsum(Utilities.split_by_cumsum_parallel(boxsize/1024, (1024**3*6/80)/1024),dim=0)
            
            
            for i in range(len(gausian_batch)):
                # index = start/2
                start = gausian_batch[i-1].item() if i != 0 else 0
                end = gausian_batch[i]
                rects = Utilities.make_rect_points_parallel(box_startpoint_int, box_endpoint_int)
                Utilities.gpu_mem()
                print(f"box_startpoint_int.shape = {box_startpoint_int.shape}")
                print(f"box_endpoint_int.shape = {box_endpoint_int.shape}")
                #mean_pixel,opacity,L_dをガウシアン長方形ボックスの形に複製&１次元配列化
                mean_pixel_batch_boxcopy = torch.repeat_interleave(mean_pixel_batch[start:end,:], boxsize, dim=0)
                # mean_pixel_batch_boxcopy = torch.repeat_interleave(mean_pixel_batch, boxsize, dim=0)
                print(f"mean_pixel_batch_boxcopy.shape = {mean_pixel_batch_boxcopy.shape}")
                Utilities.gpu_mem()
                print(f"L_d_zsortity.shape = {L_d_zsort[batch_i,start:end,:].shape}")
                L_d_boxcopy = torch.repeat_interleave(L_d_batch[start:end,:], boxsize, dim=0)
                Utilities.gpu_mem()
                opacity_boxcopy = torch.repeat_interleave(opacity_batch[start:end,:], boxsize, dim=0)
                Utilities.gpu_mem()
                variance_inverse_boxcopy = torch.repeat_interleave(variance_inverse_batch[start:end,:], boxsize, dim=0)
                Utilities.gpu_mem()
                Gaus_karnel = torch.exp(-0.5 * (rects - mean_pixel_batch_boxcopy)[:,None,:] @ variance_inverse_boxcopy @ (rects - mean_pixel_batch_boxcopy)[:,:,None])
                Utilities.gpu_mem()
                keys = rects[:,0] * (10000 + 1) + rects[:,1]
                unti_opacity = 1 - opacity_boxcopy.reshape(-1) * Gaus_karnel.reshape(-1)
                Utilities.gpu_mem()
                T_0 = Utilities.grouped_cumprod(unti_opacity,keys) / unti_opacity.clamp(min=1e-8)
                Utilities.gpu_mem()
                pixel_batch = T_0[:,None] * L_d_boxcopy * opacity_boxcopy * Gaus_karnel.reshape(-1,1) 
                Utilities.gpu_mem()
                # (H, W) 行列を作成
                pixel_batch_sum = torch.zeros((shape_height.to(torch.int32)+1, shape_width.to(torch.int32)+1, 3), dtype=torch.float32,device="cuda")
                Utilities.gpu_mem()

                # 各座標に A を加算（同じ座標なら合算）
                pixel_batch_sum.index_put_(
                    (rects[:, 1].long(), rects[:, 0].long()),
                    pixel_batch,
                    accumulate=True
                )
                Utilities.gpu_mem()
                
                #
                if(index != 0) :
                    pixel_gausian_batch.append(pixel_gausian_batch[index-1] + pixel_batch_sum)
                    #index-1とpixel_batch_sumをnoneにする
                    pixel_gausian_batch[index-1] = 0
                    
                else :
                    pixel_gausian_batch.append(pixel_batch_sum)
                    
                pixel_batch_sum = 0
                gc.collect()
                torch.cuda.empty_cache()
                index += 1
            #追加
            if len(pixel_gausian_batch) > 0:
                pixel_image_batch.append(pixel_gausian_batch[-1])
                pixel_gausian_batch[-1] = 0

                
        
        #ピクセル行列の計算のパイプライン
        #ピクセルマスクの作成
        #x_pixel,y_pixel
        # x_pixel = torch.arange(wh[0,0],device="cuda",dtype=torch.float32)
        # y_pixel = torch.arange(wh[0,1],device="cuda",dtype=torch.float32)
        # #ピクセル格子点の座標行列を取得（デカルト積を用いる）
        # Utilities.gpu_mem()
        # # pixel_uv = torch.column_stack([x_pixel[None,:].repeat(len(y_pixel), 1), torch.tile(y_pixel, len(x_pixel))])
        # n, m = int(wh[0,1].item()), int(wh[0,0].item())
        # i = torch.arange(n,device="cuda",dtype=torch.float32).unsqueeze(1).expand(n, m)  # shape (n, m)
        # j = torch.arange(m,device="cuda",dtype=torch.float32).unsqueeze(0).expand(n, m)  # shape (n, m)
        # pixel_uv = torch.stack((i, j), dim=2)[..., [1,0]] #要素は(x,y)の順であることに注意
        # pixel_uv_brod = pixel_uv[None,None,:,:,:]
        # #broadcast
        # Utilities.gpu_mem()
        # x_pixel_broadcast = x_pixel[None,None,:]
        # y_pixel_broadcast = y_pixel[None,None,:]
        # Utilities.gpu_mem()
        # p_x = mean_pixel[:,:,0][:,:,None]
        # Utilities.gpu_mem()
        # gb_x = gausian_boxsize[:,:,0][:,:,None]
        # p_y = mean_pixel[:,:,1][:,:,None]
        # gb_y = gausian_boxsize[:,:,1][:,:,None]
        # Utilities.gpu_mem()
        # #ガウシアン99.7%ボックスのマスク
        # gausian_box_mask_x = (p_x - gb_x <= x_pixel_broadcast) and (x_pixel_broadcast <= p_x + gb_x)
        # gausian_box_mask_y = (p_y - gb_y <= y_pixel_broadcast) and (y_pixel_broadcast <= p_y + gb_y)
        # #z深度=0を除外するマスク(幾何学的には画像面の無限遠点に相当する)
        # Utilities.gpu_mem()
        # # z_zero_mask = (mean_pixel_homo[:,:,2] != 0)
        # #固有値のいずれかが0となる場合を除外するマスク(正則にならないので逆行列を計算できない)
        # #マスクの計算
        # Utilities.gpu_mem()
        # msk = gausian_box_mask_x[:,:,None,:] & gausian_box_mask_y[:,:,:,None]
        
        # #ガウスカーネルの計算
        # mean_k = (pixel_uv_brod - mean_pixel[:,:,None,None,:]) #[:,:,:,None,:]
        # Utilities.gpu_mem()
        # variance_inverse_k = torch.linalg.inv(variance_pixel)[:,:,None,None,:,:]
        # Utilities.gpu_mem()
        # Gaus_karnel = torch.exp(-0.5 * mean_k[:,:,:,:,None,:] @ variance_inverse_k @ mean_k[:,:,:,:,:,None])
        # #深度zでパラメータを昇順にソート
        # #深度インデックスの計算
        # Utilities.gpu_mem()
        # z_index = torch.argsort(mean_camera[:,:,2],dim=1)
        # #逆深度インデックスの計算
        # Utilities.gpu_mem()
        # z_index_reverse = torch.argsort(z_index,dim=1)
        # #batchのためのブロードキャスト配列の作成
        # Utilities.gpu_mem()
        # batch_index = torch.arange(shape_image)[:,None]
        # #不透明度パラメータをzソート
        # Utilities.gpu_mem()
        # opacity_batch_zsort = torch.sigmoid_(self.opacity)[None,:,:][torch.zeros(shape_image),:,:][batch_index,z_index,:]
        # #透過率T_oの計算
        # #透明度(1-不透明度)の相乗を計算するためにopacity_batchをブロードキャストする.
        # Utilities.gpu_mem()
        # opacity_batch_broadcast = opacity_batch_zsort[:,None,:,:] + torch.zeros(shape_image,shape_wh,shape_gausian,shape_gausian)
        # #ガウスカーネルで重みづけ
        # #ガウスカーネルをブロードキャスト 
        # # !maskするときは,まずブロードキャストしてshapeを同じにするべきです.
        # # !なぜなら,ブロードキャストとmaskを併用するとブロードキャストされる前にmaskされてしまう.
        # # !また,A[mask] = (B[mask] + C[mask]) @ D[mask]...という形に計算される前にマスクをかけないと,
        # # !maskを満足しない要素に対しても計算が行われる.maskの使用で生じる非連続なメモリ計算によるオーバーヘッドを考慮して,
        # # 極端に疎なテンソルに対してはmaskをかける,そうでなければ可算や乗算のような単純な計算では全要素で計算するという方法を用いるべきです.
        # # とりあえず全てmaskかけて計算した.テンソルとマスクの要素数を比較して割合を求めて,条件分岐する式を書き加えればよいだけ.
        # Utilities.gpu_mem()
        # Gaus_kernel_broadcast_to_opacity = torch.broadcast_to(Gaus_karnel.reshape(shape_image,shape_wh,shape_gausian)[:,:,None,:],opacity_batch_broadcast.shape)
        # Utilities.gpu_mem()
        # msk_broadcast_to_opacity = torch.broadcast_to(msk.reshape(shape_image,shape_wh,shape_gausian)[:,:,None,:],opacity_batch_broadcast.shape)
        # Utilities.gpu_mem()
        # opacity_kernel = torch.zeros(shape_image,shape_wh,shape_gausian,shape_gausian)
        # # ガウスカーネルをzソート←忘れてた.
        # # ４階のテンソルに対して第3軸(つまり最後の軸)に対してz_indexで指定された順序でソートを行う
        # # 非常に煩雑な計算で,これは修正する必要があると思う
        # z_index_broadcast = torch.broadcast_to(z_index[:,None,None,:],opacity_batch_broadcast)[:,:,0,:]
        # Utilities.gpu_mem()
        # z_index_broadcast_reverse = torch.argsort(z_index_broadcast,axis=3)
        # z_index_broadcast_0 = torch.broadcast_to(torch.arange(shape_image)[:,None,None], z_index_broadcast)
        # Utilities.gpu_mem()
        # z_index_broadcast_1 = torch.broadcast_to(torch.arange(shape_wh)[None,:,None], z_index_broadcast)
        # Gaus_kernel_broadcast_to_opacity_zsort = Gaus_kernel_broadcast_to_opacity[z_index_broadcast_0,z_index_broadcast_1,:,z_index_broadcast]
        # Utilities.gpu_mem()

        # opacity_kernel[msk_broadcast_to_opacity] = torch.tril(opacity_batch_broadcast,k=-1)[msk_broadcast_to_opacity] * Gaus_kernel_broadcast_to_opacity_zsort[msk_broadcast_to_opacity]
        # #透過率T_oを計算しそれを逆zソートして元の順番に戻す.
        # # T_o -> (画像数,ガウス数,1)
        # Utilities.gpu_mem()
        # T_o = torch.broadcast_to(torch.prod(1 - opacity_kernel,axis=3)[:,:,None,:],opacity_batch_broadcast)[z_index_broadcast_0,z_index_broadcast_1,:,z_index_broadcast_reverse]
        # #ピクセル輝度pixelを計算
        # #maskを適用するためにブロードキャスト
        # #maskをA[mask]=B[mask]+C[mask]という形で計算すれば、maskで除外された要素は計算されないので、要素計算が重い場合に高速に計算できる。
        # #たとえコードが長くなっても最適化のためにやるべきです。
        # Utilities.gpu_mem()
        # opacity_batch_broadcast_r = opacity_batch_broadcast[:,:,0,:][:,:,:,None] + torch.zeros(shape_image,shape_wh,shape_gausian,3)
        # L_d_broadcast_r = torch.broadcast_to(L_d[:,None,:,:],opacity_batch_broadcast_r)
        # Utilities.gpu_mem()
        # Gaus_kernel_broadcast_to_opacity_r = torch.broadcast_to(Gaus_kernel_broadcast_to_opacity[:,:,0,:][:,:,:,None],opacity_batch_broadcast_r)
        # T_o_r = torch.broadcast_to(T_o[:,:,0,:][:,:,:,None],opacity_batch_broadcast_r)
        # Utilities.gpu_mem()
        # msk_r = torch.broadcast_to(msk_broadcast_to_opacity[:,:,0,:][:,:,:,None],opacity_batch_broadcast_r)
        # pixel_r = torch.zeros(opacity_batch_broadcast_r.shape)
        # Utilities.gpu_mem()
        # pixel_r[msk_r] = opacity_batch_broadcast_r[msk_r] * L_d_broadcast_r[msk_r] * Gaus_kernel_broadcast_to_opacity_r[msk_r] * T_o_r[msk_r]
        # #ピクセル行列pixelの計算 pixel -> (画像数,ガウス数,チャンネル数(RGB=3),height,width)
        # Utilities.gpu_mem()
        # pixel = torch.sum(pixel_r,axis=2).reshape(shape_image,3,wh[0,1],wh[0,0])
        # #クリッピングする
        # Utilities.gpu_mem()
        # pixel_clipping = torch.clamp(pixel, min=0.0, max=1.0) 
        # #戻り値を返す.
        return 

    
    def culling_param(self):
        
        return self
    def cloning_param(self):

        return self
    def splitting_param(self):

        return self