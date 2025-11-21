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
from gs_model import GS_dataset, GS_model_with_param
from gs_load_colmap import Load_colmap_data_from_binaries, Image_to_transform
from gs_visualizer import Visualizer
import time


#コントロールクラス
class Control():
    def __init__(self):
        self._mode = input("簡易モード(細かくパラメータ指定しない)を選びますか?(Y/N)").lower() in ["y", "yes"]
        if(self._mode) : #簡易モード すでにユーザがパラメータを指定しているならそちらが優先される
            self.colmap_root_dir = None
            self.learning_numbers = 30_000
            self.batch_size_rate = 0.04
            self.uclid_mean_point_number = 3
            self.opacity = 0.1
            self.loss_lamda = 0.2
            self.densify_until_iter = round(self.learning_numbers / 2)
            self.grad_delta_upper_limit = 1e-12
            self.limit_upper_grad = 0.0004
            self.dense_percent = 0.01
            self.prunning_opacity_min = 0.005
            self.reset_opacity_min = 0.01
            self.densify_from_iter = 500
            self.densification_interval = 100
            self.opacity_reset_interval = 3000
            self.variance_pixel_tile_max_width = 0.04
            self.learning_rate = 0.001
            
            self.position_lr_init = 0.00016
            self.position_lr_final = 0.0000016
            self.position_lr_delay_mult = 0.01
            self.position_lr_max_steps = 30_000
            self.feature_lr = 0.0025
            self.opacity_lr = 0.025
            self.scaling_lr = 0.005
            self.rotation_lr = 0.001
            
            # self.colmap_root_dir = None
            # self.learning_numbers = 30_000
            # self.batch_size_rate = 0.1
            # self.uclid_mean_point_number = 3
            # self.o_init = -1.0
            # self.loss_lamda = 0.2
            # self.densify_until_iter = round(self.learning_numbers / 2)
            # self.grad_delta_upper_limit = 1e-12
            # self.limit_upper_grad = 0.005
            # self.dense_percent = 0.1
            # self.prunning_opacity_min = 0.01
            # self.reset_opacity_min = 0.01
            # self.densify_from_iter = 100
            # self.densification_interval = 100
            # self.opacity_reset_interval_per_densification = 0
            # self.variance_pixel_tile_max_width = 0.04
            # self.learning_rate = 0.2
            
            #     self.iterations = 30_000
            # self.position_lr_init = 0.00016
            # self.position_lr_final = 0.0000016
            # self.position_lr_delay_mult = 0.01
            # self.position_lr_max_steps = 30_000
            # self.feature_lr = 0.0025
            # self.opacity_lr = 0.025
            # self.scaling_lr = 0.005
            # self.rotation_lr = 0.001
            # self.exposure_lr_init = 0.01
            # self.exposure_lr_final = 0.001
            # self.exposure_lr_delay_steps = 0
            # self.exposure_lr_delay_mult = 0.0
            # self.percent_dense = 0.01
            # self.lambda_dssim = 0.2
            # self.densification_interval = 100
            # self.opacity_reset_interval = 3000
            # self.densify_from_iter = 500
            # self.densify_until_iter = 15_000
            # self.densify_grad_threshold = 0.0002
            # self.depth_l1_weight_init = 1.0
            # self.depth_l1_weight_final = 0.01
            # self.random_background = False
            # self.optimizer_type = "default"
   
    #modeを変更する
    def changing_mode(self,mode) : # mode = True -> 簡易モード
        self._mode = mode
    #学習
    def learning(self):
        try:
            self.colmap_root_dir = input("colmapのディレクトリパスを指定してください:") or self.colmap_root_dir if self._mode != True else self.colmap_root_dir
            colmap_data = Load_colmap_data_from_binaries(self.colmap_root_dir)
            root_dir = colmap_data.root_dir
            initial_xyz_tensor, P, K, wh, image_samples = colmap_data.convert_to_tensors()
            #メモリを明示的に開放
            Utilities.mem_refresh(colmap_data,True)
            self.batch_size_rate = input("学習画像全体に対するバッチ数の割合を指定してください(標準0.1):") or self.batch_size_rate if self._mode != True else self.batch_size_rate
            self.batch_size = 3 #round(len(image_samples) * self.batch_size_rate)
            self.learning_numbers = input("学習回数を指定してください(標準35000):") or self.learning_numbers if self._mode != True else self.learning_numbers
            self.learning_rate = input("確率降下勾配法の学習率を指定してください(標準0.1):") or self.learning_rate if self._mode != True else self.learning_rate
            #パラメータ初期化
            self.gaus_mean = initial_xyz_tensor
            self.gaus_point_numbers = initial_xyz_tensor.shape[0]
            self.variance_q = torch.zeros((self.gaus_point_numbers,4),device="cuda")
            self.variance_q[:,3] = 1
            self.uclid_mean_point_number = input("分散スケール初期値のユークリッド平均で用いる点群の数を指定してください(標準3):") or self.uclid_mean_point_number if self._mode != True else self.uclid_mean_point_number
            self.variance_scale = torch.log_(Utilities.kyori2(self.uclid_mean_point_number,self.gaus_mean))
            self.opacity = input("不透明度αはシグモイド関数σ(o)を介して定められる.oの初期値を指定してください(標準-1.0から-5.0程度):") or self.opacity if self._mode != True else self.opacity
            self.gaus_point_o = torch.zeros((self.gaus_point_numbers,1),device="cuda") + torch.logit(torch.tensor([self.opacity],device="cuda",dtype=torch.float32))[None,:]
            #損失関数の係数λの初期化
            self.loss_lamda = input("損失関数の係数λの初期値を指定してください(標準0.2):") or self.loss_lamda if self._mode != True else self.loss_lamda
            #ここですべてのユーザ入力値を指定
            self.densify_until_iter = input("ガウシアンの複製と分割を終了するイテレーション数を指定してください(標準):") or self.densify_until_iter if self._mode != True else self.densify_until_iter
            self.densify_from_iter = input("ガウシアンの複製と分割を開始するイテレーション数を指定してください(標準50):") or self.densify_from_iter if self._mode != True else self.densify_from_iter
            self.densification_interval = input("ガウシアンの複製と分割のイテレーション間隔を指定してください(標準):") or self.densification_interval if self._mode != True else self.densification_interval
            self.opacity_reset_interval = input("不透明度のリセット間隔を指定してください(自然数でなければならない。標準0 = リセットしない):") or self.opacity_reset_interval if self._mode != True else self.opacity_reset_interval
            self.grad_delta_upper_limit = input("勾配値の変化を検出する上限の値を指定してください(標準1e-12):") or self.grad_delta_upper_limit if self._mode != True else self.grad_delta_upper_limit
            self.limit_upper_grad = input("ガウス分割と複製を行う上限値(標準0.005):") or self.limit_upper_grad if self._mode != True else self.limit_upper_grad
            self.prunning_opacity_min = input("削除を行う不透明度の下限値(標準0.01):") or self.prunning_opacity_min if self._mode != True else self.prunning_opacity_min
            self.dense_percent = input("???を指定してください(標準0.1):") or self.dense_percent if self._mode != True else self.dense_percent
            self.reset_opacity_min = input("リセットする不透明度の値(標準0.01):") or self.reset_opacity_min if self._mode != True else self.reset_opacity_min
            self.variance_pixel_tile_max_width = input("ピクセル共分散行列の%タイルの最大幅を指定してください.(標準0.04):") or self.variance_pixel_tile_max_width if self._mode != True else self.variance_pixel_tile_max_width
            #モデルインスタンス作成
            self.GS_model_param = GS_model_with_param(
                self.gaus_mean,self.variance_q,self.variance_scale,
                self.gaus_point_o,self.grad_delta_upper_limit,self.limit_upper_grad,
                self.dense_percent,self.prunning_opacity_min,
                self.variance_pixel_tile_max_width,
                self.position_lr_init,
                self.position_lr_final,
                self.position_lr_delay_mult,
                self.position_lr_max_steps,
                self.feature_lr,self.opacity_lr,
                self.scaling_lr,self.rotation_lr,
                lr = self.learning_rate)
            #データセットのインスタンス作成
            self.GS_dataset = GS_dataset(P,K,wh,image_samples)
            #カメラの平均距離から最も遠いカメラの距離
            self.camera_extent = self.GS_dataset.get_camera_extent()
            #レンダリング画像出力用のウィンドウ作成
            # self.vis = Visualizer()
            
            #イテレーション開始
            learning_numbers_per_epoch = self.learning_numbers / len(image_samples) #１エポックあたりの学習回数
            #処理時間計測
            iter_time = 0
            #lossから収束しているか判断
            loss_sum = 0
            for iter_i in iter(torch.arange(round(learning_numbers_per_epoch),device="cuda",dtype=torch.float32)):
                #
                it = DataLoader(
                self.GS_dataset,   # Datasetのインスタンス
                batch_size=self.batch_size,     # バッチサイズ（1回に取り出すデータ数）
                shuffle=True,      # データのシャッフル（エポックごと）
                num_workers=0,     # 並列でデータをロードするスレッド数
                drop_last=False,   # 最後の中途半端なバッチを捨てるか
            )
                for batch_i, (it_P,it_K,it_wh,it_image_sample) in enumerate(it):
                    time_start = time.time()
                    #パラメーターの勾配計算オン
                    self.GS_model_param.changing_required_grad(True)
                    #ガウシアンスプラッティングによる画像の出力
                    model_images,it_image_sample,grad_iter = self.GS_model_param(it_P,it_K,it_wh,list(it_image_sample))
                    #画像パスから学習画像をgpuメモリに配置
                    images_tensor = torch.empty((0,3,it_wh[0,1].to(torch.int32).item(),it_wh[0,0].to(torch.int32).item()))
                    for img in it_image_sample:
                        image_tensor = Image_to_transform(root_dir,img).convert_to_torch_tensor()[None,:,:,:]
                        images_tensor = torch.cat((images_tensor,image_tensor),dim=0)
                    images_tensor_gpu = images_tensor.to("cuda")
                    #損失関数を計算
                    loss_d_ssim =  1 - metrics.ssim(model_images, images_tensor_gpu, max_val=1.0, window_size=11).mean()
                    loss_1 = torch.nn.functional.l1_loss(model_images, images_tensor_gpu, reduction='mean')
                    loss = (1-self.loss_lamda) * loss_1 + self.loss_lamda * loss_d_ssim
                    #自動微分による勾配計算
                    # torch.autograd.set_detect_anomaly(True)
                    self.GS_model_param.backward(loss)
                    #ガウス分布の位置の勾配値の積算回数の更新
                    self.GS_model_param.param_iter_update(grad_iter)
                    #確率的勾配降下法によって最適化
                    self.GS_model_param.train_step()
                    # イテレーション数を計算
                    iteration = iter_i * math.ceil(len(image_samples) / self.batch_size) + batch_i + 1
                    print(f"現在の学習回数は{iteration}です.")
                    #lossの平均
                    loss_sum += loss
                    print(f"lossの平均: {loss_sum / iteration}")
                    #学習率の更新
                    self.GS_model_param.lr["mean"] = self.GS_model_param.mean_lr_setfunc(iteration)
                    #学習率をoptimizerに設定
                    self.GS_model_param.changing_optimizer()
                    #ガウシアンの複製・分割・削除と不透明度のリセットを行う
                    if iteration <= self.densify_until_iter and iteration >= self.densify_from_iter:
                        
                        if (iteration % self.densification_interval == 0):
                            self.GS_model_param.densify_and_prune(self.camera_extent, self.learning_rate)
                        
                    if ( self.opacity_reset_interval != 0 and (iteration % self.opacity_reset_interval == 0) ):
                            self.GS_model_param.reset_opacity(self.reset_opacity_min)
                            
                    #処理時間を表示
                    time_end = time.time()
                    print(f"第{iteration}回の処理時間は、{time_end - time_start}秒です。") 
                    iter_time += (time_end - time_start) / 60
                    print(f"総学習時間は,{iter_time}分です.")
                    #レンダリング画像表示
                    # if batch_i + 1 == self.batch_size:
                    # if(model_images != None): self.vis.update(model_images[0,:,:,:])
                    if((model_images != None) and (iteration % 1 == 0)):
                        tensor_img = model_images[0,:,:,:].detach().cpu()   # GPU → CPU ＆ graph 切断
                        # PIL画像に変換
                        img_pil = torchvision.transforms.functional.to_pil_image(tensor_img)

                        # 保存
                        img_pil.save(os.path.join("rendering_images", f"output{iteration}.png"))
                        
                    #パラメータを保存
                    if (iteration % 1000 == 0):
                        torch.save(self.GS_model_param.mean,"mean.pt")
                        torch.save(self.GS_model_param.variance_q,"variance_q.pt")
                        torch.save(self.GS_model_param.variance_scale,"variance_scale.pt")
                        torch.save(self.GS_model_param.opacity,"opacity.pt")
                        torch.save(self.GS_model_param.color,"color.pt")
                    #ガウス点群数を表示
                    print(f"ガウス点群数は、{self.GS_model_param.mean.size(0)}点数です.")
                    #gpuメモリの解放
                    Utilities.mem_refresh(images_tensor_gpu,False)

                    
                    
                    

        except Exception as e:
            print("エラー:", e)
            print("初めからやり直します.入力は保存されています.問題のある入力のみ変更してください.")
            self.learning()

