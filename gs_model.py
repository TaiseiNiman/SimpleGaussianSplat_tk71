import torch
import numpy as np
from PIL import Image
import torch.utils.data
import kornia.metrics as metrics
import gc
from uitility import Utilities
import grouped_cumprod


#3dgsデータセットの定義
class GS_dataset(torch.utils.data.Dataset):
    def __init__(self,P,K,wh,image_sample):
        self.P = P
        self.K =  K
        self.wh =  wh
        self.image_sample =  image_sample
    def __len__(self):
        return len(self.P)
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
    
    def requirs_grad_reaf_node(self):
        for name, param in self.named_parameters():
            if param.is_leaf and param.requires_grad:
                print(f"Leaf param: y{name}, shape={param.shape}")
        
        
class _GS_model_with_ultra_param(_GS_model):
    def __init__(self, grad_delta_upper_limit,grad_threshold,percent_dense,variance_pixel_tile_max_width, lr=0.1, caller=None):
        #GS_model_with_paramのみ利用しないとエラーが出る.
        super().__init__(caller)
        self.grad_delta_upper_limit = torch.nn.Parameter(torch.tensor(grad_delta_upper_limit,device="cuda",dtype=torch.float32))
        self.grad_threshold = torch.nn.Parameter(torch.tensor(grad_threshold,device="cuda",dtype=torch.float32))
        self.percent_dense = torch.nn.Parameter(torch.tensor(percent_dense,device="cuda",dtype=torch.float32))
        self.variance_pixel_tile_max_width = torch.nn.Parameter(torch.logit(torch.tensor(variance_pixel_tile_max_width,device="cuda",dtype=torch.float32)))
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
        color = torch.zeros((self.mean.size(0),(L_max+1)**2,3),device="cuda",dtype=torch.float32)
        color[:,0,:] = c_00 #ベース色のみ中間色に設定
        self.color = torch.nn.Parameter(color)
        self.changing_optimizer(lr)
        self._L_max = L_max
        #パラメーターとして使用しないローカル変数
        self.mean_grads_norm = torch.zeros(self.mean.shape[0],device="cuda")
        self.mean_grads_iter = torch.clone(self.mean_grads_norm)
        #勾配計算されないように設定
        self.requirs_grad_reaf_node()
        
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
        return self.mean_grads_norm / grads_iter
    
    def densify_and_clone(self, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(self.param_grads_per_iter_norm() >= self.ultra.grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(torch.exp(self.variance_scale), dim=1).values > self.ultra.percent_dense*scene_extent)
        
        #self.mean[selected_pts_mask].grad = 0
        self.mean_grads_norm = torch.hstack((self.mean_grads_norm,self.mean_grads_norm[selected_pts_mask]))
        self.mean_grads_iter = torch.hstack((self.mean_grads_iter,self.mean_grads_iter[selected_pts_mask]))
        self.mean = torch.nn.Parameter(torch.vstack((self.mean,self.mean[selected_pts_mask])))
        #
        self.variance_q = torch.nn.Parameter(torch.vstack((self.variance_q,self.variance_q[selected_pts_mask])))
        self.variance_scale = torch.nn.Parameter(torch.vstack((self.variance_scale,self.variance_scale[selected_pts_mask])))
        self.opacity = torch.nn.Parameter(torch.vstack((self.opacity,self.opacity[selected_pts_mask])))
        self.color = torch.nn.Parameter(torch.vstack((self.color,self.color[selected_pts_mask])))

    def densify_and_prune(self, extent, lr):
        
        self.densify_and_clone(extent)
        
        prune_mask = (torch.sigmoid(self.opacity) < self.super.prunning_min_opacity).squeeze(1)

        big_points_ws = torch.max(torch.exp(self.variance_scale), dim=1).values > 0.1 * extent
        prune_mask = torch.logical_or(prune_mask, big_points_ws)
        #ガウスを削除
        self.mean_grads_norm = self.mean_grads_norm[prune_mask == False]
        self.mean_grads_iter = self.mean_grads_iter[prune_mask == False]
        self.mean = torch.nn.Parameter(self.mean[prune_mask == False])
        self.variance_q = torch.nn.Parameter(self.variance_q[prune_mask == False])
        self.variance_scale = torch.nn.Parameter(self.variance_scale[prune_mask == False])
        self.opacity = torch.nn.Parameter(self.opacity[prune_mask == False])
        self.color = torch.nn.Parameter(self.color[prune_mask == False])
        
        self.changing_optimizer(lr)
    
    # P -> (画像数,3,4)
    # K -> (画像数,3,3)
    # wh -> (画像数,2(width,heightの順))
    def forward(self, P, K, wh):#input = 入力, super_param = 超パラメータ
        #shape_image = 画像数, shape_gausian = ガウス分布の数
        shape_gausian, shape_image, shape_wh, shape_width, shape_height = [self.mean.shape[0], P.shape[0], (wh[0,0]*wh[0,1]).to(torch.int32), wh[0,0].to(torch.int32), wh[0,1].to(torch.int32)]
        #オーバーフロー及びアンダーフロー
        self.requirs_grad_reaf_node()
        float32_max = torch.finfo(torch.float32).max
        float32_min = torch.finfo(torch.float32).min
        int32_max = torch.iinfo(torch.int32).max
        int32_min = torch.iinfo(torch.int32).min
        #ガウス位置のカメラ座標mean_cameraを計算.←ガウスの位置meanをカメラ座標系に座標変換
        self.requirs_grad_reaf_node()
        m1, P1 = [shape_gausian,shape_image]
        Utilities.gpu_mem()
        homo_mean = torch.hstack((self.mean,torch.ones((m1,1),device="cuda")))[None, :, :] #同次座標変換
        Utilities.gpu_mem()
        mean_camera = homo_mean @ torch.transpose(P,1,2) #[:, None, :, :]).reshape(P1,m1,3) カメラ座標xyzに変換
        self.requirs_grad_reaf_node()
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
        variance_pixel = (J @ variance_camera @ torch.transpose(J,2,3)).clamp(max=float32_max/1000,min=float32_min/1000)
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
        opacity_zsort = torch.sigmoid(self.opacity).unsqueeze(0).repeat(shape_image,1,1)[image_index,z_index,:]
        mean_pixel_zsort = mean_pixel[image_index,z_index,:].clamp(max=int32_max/1000,min=int32_min/1000).to(torch.int32)
        variance_inverse_zsort = variance_inverse[image_index,z_index,:]
        L_d_zsort = L_d[image_index,z_index,:]
        ptile_maxwidth = torch.sqrt(shape_wh)*torch.sigmoid(self.ultra.variance_pixel_tile_max_width)
        gausian_boxsize_zsort = gausian_boxsize[image_index,z_index,:].clamp(max=ptile_maxwidth*10).to(torch.int32)
        print(f"opacity.shape = {opacity_zsort.shape}")
        print(f"mean_pixel_zsort.shape = {mean_pixel_zsort.shape}")
        print(f"variance_inverse_zsort.shape = {variance_inverse_zsort.shape}")
        print(f"L_d_zsortity.shape = {L_d_zsort.shape}")
        print(f"gausian_boxsize_zsort.shape = {gausian_boxsize_zsort.shape}")
        
    #     test = Utilities.render_gaussian_batch(
    #     mean_pixel_zsort, gausian_boxsize_zsort,
    #     L_d_zsort, opacity_zsort, variance_inverse_zsort,
    #     shape_width, shape_height
    # )
        
        Utilities.gpu_mem()
        pixel_image_batch =[]
        for batch_i in range(shape_image):
            #
            pixel_gausian_batch = []
            anti_opacity_max = []
            pixel_keys_max = []
            index = 0
            #ガウシアンボックス幅が０となるガウシアンを取り除く
            boxsize_mask = (gausian_boxsize_zsort[batch_i, :, 0] != 0) & (mean_pixel_zsort[batch_i,:,0] - gausian_boxsize_zsort[batch_i, :, 0] < shape_width) & (mean_pixel_zsort[batch_i,:,0] + gausian_boxsize_zsort[batch_i, :, 0] > 0) & (mean_pixel_zsort[batch_i,:,1] - gausian_boxsize_zsort[batch_i, :, 1] < shape_height) & (mean_pixel_zsort[batch_i,:,1] + gausian_boxsize_zsort[batch_i, :, 1] > 0)  
            gausian_boxsize_wh = gausian_boxsize_zsort[batch_i,:,:][boxsize_mask]
            mean_pixel_batch = mean_pixel_zsort[batch_i,:,:][boxsize_mask]
            L_d_batch = L_d_zsort[batch_i,:,:][boxsize_mask]
            opacity_batch = opacity_zsort[batch_i,:,:][boxsize_mask]
            variance_inverse_batch = variance_inverse_zsort[batch_i,:,:][boxsize_mask]
            
            #endpointおよびstartpointを作成
            box_startpoint_x = (mean_pixel_batch[:,0] - gausian_boxsize_wh[:,0]).clamp(max=shape_width,min=0)
            box_startpoint_y = (mean_pixel_batch[:,1] - gausian_boxsize_wh[:,1]).clamp(max=shape_height,min=0)
            box_startpoint = torch.hstack((box_startpoint_x[:,None],box_startpoint_y[:,None]))
            box_endpoint_x = (mean_pixel_batch[:,0] + gausian_boxsize_wh[:,0]).clamp(max=shape_width,min=0)
            box_endpoint_y = (mean_pixel_batch[:,1] + gausian_boxsize_wh[:,1]).clamp(max=shape_height,min=0)
            box_endpoint = torch.hstack((box_endpoint_x[:,None],box_endpoint_y[:,None]))
            gausian_boxsize_batch = torch.prod((box_endpoint - box_startpoint + 1), dim=1)
            print(f"boxsize.shape = {gausian_boxsize_batch.shape}")
            #バッチ数を1024^3*6 / 80 = 0.075GB以下となるように確定する
            gausian_batch = torch.cumsum(Utilities.split_by_cumsum_parallel(gausian_boxsize_batch/1024, (1024**3*6/640)/1024),dim=0)
            Utilities.gpu_mem()
            torch.cuda.empty_cache()
            #各ガウシアンの長方形ピクセル座標を2次元配列(N,2)として返す
            # box_startpoint_int = box_startpoint.to(torch.int32)
            # box_endpoint_int = box_endpoint.to(torch.int32)
            
            #マスク
            #ガウシアンのvariance_pixelが99.7%タイルが画像に対してある幅以上となるものは全てその幅まで縮小する.
            # 理由：variance_pixelは線形一次近似であって,ガウス中心に対して十分近傍なピクセル値に対してしか有効でないはずだからである.
            # 具体的に画像に対して何パーセントとすべきかは、これ自体をハイパーパラメータとして学習するか公式の実装コードを読み解くしかない.
            
            
            
            for i in range(len(gausian_batch)):
                # index = start/2
                start = gausian_batch[i-1].item() if i != 0 else 0
                end = gausian_batch[i]
                rects = Utilities.make_rect_points_parallel(box_startpoint[start:end,:], box_endpoint[start:end,:]).to(torch.int32)
                torch.cuda.empty_cache()
                Utilities.gpu_mem()
                print(f"box_startpoint_int.shape = {box_startpoint[start:end,:].shape}")
                print(f"box_endpoint_int.shape = {box_endpoint[start:end,:].shape}")
                #ボックスコピー
                mean_pixel_batch_boxcopy,L_d_boxcopy,opacity_boxcopy,variance_inverse_boxcopy = self.boxcopy(gausian_boxsize_batch[start:end],mean_pixel_batch[start:end,:].to(torch.float32),L_d_batch[start:end,:],opacity_batch[start:end,:],variance_inverse_batch[start:end,:])
                Gaus_karnel = torch.exp(-0.5 * (rects.to(torch.float32) - mean_pixel_batch_boxcopy)[:,None,:] @ variance_inverse_boxcopy @ (rects.to(torch.float32) - mean_pixel_batch_boxcopy)[:,:,None])
                torch.cuda.empty_cache()
                Utilities.gpu_mem()
                # self.cuda_kernel.cuda_kernel.grouped_cumprod()
                if(len(pixel_keys_max) > 0):
                    #
                    
                    #
                    unique_keys, inv = torch.unique(torch.cat((pixel_keys_max[-1],rects),dim=0),dim=0, return_inverse=True)
                    #ソート
                    pixel_index_tensor,pixel_index = torch.sort(inv)
                    #
                    unti_opacity = torch.cat((anti_opacity_max[-1],(1 - opacity_boxcopy.reshape(-1) * Gaus_karnel.reshape(-1))),dim=0).clamp(min=1e-8)[pixel_index]
                    T_0_r_nosorted = custom_autograd_grouped_cumprod.apply(unti_opacity,pixel_index_tensor,box_startpoint[start:end,:],box_endpoint[start:end,:],gausian_boxsize_batch[start:end],mean_pixel_batch[start:end,:].to(torch.float32),opacity_batch[start:end,:],variance_inverse_batch[start:end,:],pixel_keys_max[-1],anti_opacity_max[-1])
                    argsorted_index = torch.argsort(pixel_index)
                    T_0_r = T_0_r_nosorted[argsorted_index] 
                    T_0_r_nosorted = None
                    anti_opacity_max.append(torch.zeros_like(unique_keys[:,0],device="cuda",dtype=torch.float32).scatter_reduce(0, inv, T_0_r, reduce="amin", include_self=False))
                    pixel_keys_max.append(unique_keys)
                    anti_opacity_max[-2] = 0
                    pixel_keys_max[-2] = 0
                    T_0 = (T_0_r / unti_opacity[argsorted_index])[len(pixel_keys_max[-1]):]
                    T_0_r = None
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                else:
                    
                    unique_keys, inv = torch.unique(rects,dim=0, return_inverse=True)
                    #ソート
                    pixel_index_tensor,pixel_index = torch.sort(inv)
                    unti_opacity = (1 - opacity_boxcopy.reshape(-1) * Gaus_karnel.reshape(-1)).clamp(min=1e-8)[pixel_index]
                    T_0_r_nosorted = custom_autograd_grouped_cumprod.apply(unti_opacity,pixel_index_tensor,box_startpoint[start:end,:],box_endpoint[start:end,:],gausian_boxsize_batch[start:end],mean_pixel_batch[start:end,:].to(torch.float32),opacity_batch[start:end,:],variance_inverse_batch[start:end,:])
                    # loss = torch.sum((T_0_r_nosorted**2 / 2),dim=0)
                    # self.backward(loss)
                    argsorted_index = torch.argsort(pixel_index)
                    T_0_r = T_0_r_nosorted[argsorted_index] 
                    T_0_r_nosorted = None
                    anti_opacity_max.append(torch.zeros_like(unique_keys[:,0],device="cuda",dtype=torch.float32).scatter_reduce(0, inv, T_0_r, reduce="amin", include_self=False))
                    pixel_keys_max.append(unique_keys)
                    T_0 = T_0_r / unti_opacity[argsorted_index]
                    T_0_r = None
                    gc.collect()
                    torch.cuda.empty_cache()
                
                
  
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
                print(f"今{batch_i+1}番目の画像のガウシアンバッチ{i+1}枚目が終わりました.")
            #追加
            if len(pixel_gausian_batch) > 0:
                pixel_image_batch.append(pixel_gausian_batch[-1])
                pixel_gausian_batch[-1] = 0
        
        a = 0
        return torch.stack(pixel_image_batch, dim=0)[:,1:,1:,:].reshape(-1,3,shape_height,shape_width)
    
    def boxcopy(self,boxsize,*tensors):
        output = []
        for tensor in tensors:
            output.append(torch.repeat_interleave(tensor, boxsize, dim=0))
        return output
    
    def create_T_0(self,unti_opacity,keys):
        #
        unique_keys, inv = torch.unique(keys, return_inverse=True)
        #ソート
        pixel_index_tensor,pixel_index = torch.sort(inv)
        return (self.cuda_kernel.grouped_cumprod(unti_opacity[pixel_index],pixel_index_tensor)[torch.argsort(pixel_index)] / unti_opacity)


#forwardメソッドでピクセル座標系におけるガウス位置,精度行列,不透明度、放射輝度から画像テンソルを出力し、backwardでその勾配を計算する
class custom_autograd_grouped_cumprod(torch.autograd.Function):
    #rectsを計算する
    @staticmethod
    def _create_rects(startpoint,endpoint):
        return Utilities.make_rect_points_parallel(startpoint, endpoint).to(torch.int32)
    #boxcopyを計算する
    @staticmethod
    def _create_boxcopy(boxsize,*tensors):
        output = []
        for tensor in tensors:
            output.append(torch.repeat_interleave(tensor, boxsize, dim=0))
        return output
    #ガウスカーネルを計算する:
    @staticmethod
    def _create_gause_kernel(rects,mean,variance_inverse):
        return torch.exp(-0.5 * (rects.to(torch.float32) - mean)[:,None,:] @ variance_inverse @ (rects.to(torch.float32) - mean)[:,:,None])
    #テンソルをソートする:
    @staticmethod
    def _sort_tensor(index,*tensors):
        output = []
        for tensor in tensors:
            output.append(tensor[index])
        return output
    #アルファブレンドTを計算する:
    @staticmethod
    def _create_alpha_brend():
    
    #各キーのアルファブレンドの最大値とキーを返す.
    @staticmethod
    def _create_alpha_brend():
    
    #アルファブレンドを追加する.
    @staticmethod
    def _cat_alpha_brend():
    
    @staticmethod
    def forward(ctx, unti_opacity_sorted, pixel_index, box_startpoint,box_endpoint,boxsize,mean,opacity,variance_inverse,pixel_max=None,anti_opacity_max=None):
        #結果配列
        output = torch.zeros_like(unti_opacity_sorted)
        #cudaカーネルでグループごとの累積積を計算し,output配列に代入
        grouped_cumprod.grouped_cumprod_forward(unti_opacity_sorted,pixel_index.to(torch.int32),output)
        #
        ctx.save_for_backward(box_startpoint,box_endpoint,boxsize,mean,opacity,variance_inverse,pixel_max,anti_opacity_max)
        return output
    @staticmethod
    def backward(ctx,grad):
        startpoint,endpoint,boxsize,mean,opacity,variance_inverse,pixel_max,anti_opacity_max = ctx.saved_tensors
        rects = Utilities.make_rect_points_parallel(startpoint, endpoint).to(torch.int32)
        #ボックスコピー
        mean_pixel_batch_boxcopy,opacity_boxcopy,variance_inverse_boxcopy = boxcopy(boxsize,mean.to(torch.float32),opacity,variance_inverse)
        Gaus_karnel = torch.exp(-0.5 * (rects.to(torch.float32) - mean_pixel_batch_boxcopy)[:,None,:] @ variance_inverse_boxcopy @ (rects.to(torch.float32) - mean_pixel_batch_boxcopy)[:,:,None])
        torch.cuda.empty_cache()
        Utilities.gpu_mem()
        if(pixel_max):
            #
            unti_opacity = torch.cat((anti_opacity_max,(1 - opacity_boxcopy.reshape(-1) * Gaus_karnel.reshape(-1))),dim=0)
            #
            unique_keys, inv = torch.unique(torch.cat((pixel_max,rects),dim=0), dim=0, return_inverse=True)
            #ソート
            pixel_index_tensor,pixel_index = torch.sort(inv)
            pixel_index_tensor_len = torch.zeros_like(unique_keys[:,0],device="cuda",dtype=torch.int32).scatter_reduce(0, pixel_index_tensor, torch.ones_like(pixel_index_tensor,device="cuda",dtype=torch.int32), reduce="sum", include_self=False).cumsum(dim=0).to(torch.int32)
            T_0_r = torch.zeros_like(unti_opacity)
            grouped_cumprod.grouped_cumprod_forward(unti_opacity[pixel_index],pixel_index_tensor.to(torch.int32),T_0_r)
            gradin = torch.zeros_like(unti_opacity)
            grouped_cumprod.grouped_cumprod_backward(unti_opacity[pixel_index],T_0_r,grad,pixel_index_tensor.to(torch.int32),gradin,pixel_index_tensor_len)
            T_0_r = None
            gc.collect()
            torch.cuda.empty_cache()
            
        else:
            #
            unti_opacity = (1 - opacity_boxcopy.reshape(-1) * Gaus_karnel.reshape(-1))
            #
            unique_keys, inv = torch.unique(rects, dim=0, return_inverse=True)
            #ソート
            pixel_index_tensor,pixel_index = torch.sort(inv)
            pixel_index_tensor_len = torch.zeros_like(unique_keys[:,0],device="cuda",dtype=torch.int32).scatter_reduce(0, pixel_index_tensor, torch.ones_like(pixel_index_tensor,device="cuda",dtype=torch.int32), reduce="sum", include_self=False).cumsum(dim=0).to(torch.int32)
            T_0_r = torch.zeros_like(unti_opacity)
            grouped_cumprod.grouped_cumprod_forward(unti_opacity[pixel_index],pixel_index_tensor.to(torch.int32),T_0_r)
            gradin = torch.zeros_like(unti_opacity)
            grouped_cumprod.grouped_cumprod_backward(unti_opacity[pixel_index],T_0_r,grad,pixel_index_tensor.to(torch.int32),gradin,pixel_index_tensor_len)
            T_0_r = None
            gc.collect()
            torch.cuda.empty_cache()
        
        return gradin,None,None,None,None,None,None,None,None,None

def boxcopy(boxsize,*tensors):
        output = []
        for tensor in tensors:
            output.append(torch.repeat_interleave(tensor, boxsize, dim=0))
        return output