import pycolmap
import torch
import os
from PIL import Image
import torchvision

class Image_to_transform():
    def __init__(self,root_dir,path):
        try:
            # ルートパスと結合して絶対パスを構築
            full_path = os.path.join(root_dir, "images", path)
            # 画像を読み込み
            self.img = Image.open(full_path)
            # print(f"画像 ({path}) の読み込みに成功しました.")
            
        except Exception as e:
            print(f"画像 {path} の処理中にエラーが発生しました: {e}")  
            self.img = None  

    # def convert_to_pil(tensor):#画像をPIL形式に変換する.
    #     return torchvision.transforms.ToPILImage()(tensor)
    def convert_to_torch_tensor(self):#画像をテンソル形式に変換
        return torchvision.transforms.functional.to_tensor(self.img)
    def get_data(self):
        return self.img
    def __call__(self):
        return self.get_data()
    

class Load_colmap_data():
    def __init__(self,root_dir=None):
        if root_dir == None:
            self.root_dir = os.path.join(os.path.dirname(__file__), "colmap")
        else:
            self.root_dir = os.path.join(os.path.dirname(__file__), root_dir)
        #コルマップパスの指定
        self.dir_path = os.path.join(self.root_dir, "sparse", "0")
        
    def get_data(self):
        return [self.cameras,self.images,self.points3d]

    
    # def convert_from_text_to_bin(self,input_name,output_name):
    #     """
    #     Convert COLMAP text model (cameras.txt, images.txt, points3D.txt)
    #     to binary model (cameras.bin, images.bin, points3D.bin)
    #     """
    #     input_path = os.path.join(self.dir_path,input_name)
    #     output_path = os.path.join(self.dir_path,output_name)
    #     if not os.path.exists(output_path):
            
    #         if os.path.exists(input_path):
                
    #             os.makedirs(output_path)
    #             pycolmap.convert_model(
    #                 input_path=input_path,
    #                 output_path=output_path,
    #                 input_type="TXT",
    #                 output_type="BIN"
    #             )
                
    #         else:
    #             raise FileNotFoundError(f"colmapのtxt,binファイルが存在しません:{input_path},{output_path}")
            
    
    def convert_to_tensors(self):
        #
        cameras_data, images_data, points3d_data = self.get_data()

        # 1. 初期点群 (ガウスの位置) の抽出と変換
        xyz_list = torch.tensor([v.xyz for v in points3d_data.values() if v.xyz is not None],dtype=torch.float32,device="cuda")
        print(f"Initial XYZ Tensor Shape: {xyz_list.shape}") 
        
        # 2. データセットとして使うために,入力と比較画像をテンソルに変換 (PyTorchのDataLoaderで使うため)
        # ----------------------------------------------------
        P = torch.empty((0,3,4),device="cuda")
        K = torch.empty((0,3,3),device="cuda")
        wh = torch.empty((0,2),device="cuda")
        images_name_list = []
        
        # COLMAPはレンズモデル（Pinhole, Radial, Simple_Radialなど）ごとにパラメータ形式が異なる.
        model_name = ['PINHOLE', 'SIMPLE_PINHOLE','SIMPLE_RADIAL','RADIAL','BROWN']
        
        # ImagesデータからRとTを抽出し、行列に変換するカスタム関数が必要
        # (ここでは簡略化のため、qvecとtvecを直接リスト化)
        for img_id in images_data:
            image = images_data[img_id]
            # クォータニオンを回転行列に変換する処理（外部モジュールが必要）をここで行う
            #回転行列R
            # image_R = image.cam_from_world()
            # #並進ベクトルT
            # image_T = torch.tensor(image.tvec,device="cuda")[:,None]
            #回転行列P(斉次座標ではない)
            image_P = torch.tensor(image.cam_from_world().matrix(),dtype=torch.float32,device="cuda")[None,:,:]
            #回転行列Pを追加
            P = torch.cat((P,image_P),dim=0)
            # 1. 対応するカメラIDを取得
            cam_id = image.camera_id
            # 2. cameras_dataから内部パラメータオブジェクトを取得
            camera = cameras_data[cam_id]
            # 3. パラメータの抽出
            if camera.model in model_name[1:3]:
                fx = fy = camera.params[0]
                cx, cy = camera.params[1:3]
            else:
                fx, fy, cx, cy = camera.params[0:4]
            #画像パスを配列に追加
            images_name_list.append(image.name)
            #外部パラメータK
            image_K = torch.tensor([[fx,0,cx],[0,fy,cy],[0,0,1]],dtype=torch.float32,device="cuda")[None,:,:]
            #外部パラメータkを作成し配列に追加
            K = torch.cat((K,image_K),dim=0)
            #画像の縦横を作成し配列に追加
            image_wh = torch.tensor([camera.width,camera.height],dtype=torch.float32,device="cuda")[None,:]
            wh = torch.cat((wh,image_wh),dim=0)

        return [xyz_list, P, K, wh, images_name_list]


class Load_colmap_data_from_binaries(Load_colmap_data):
    def __init__(self,root_dir=None):
        super().__init__(root_dir=root_dir)
        try:
            if not os.path.exists(self.dir_path):
                raise FileNotFoundError(f"Path not found: {self.dir_path}")

            recon = pycolmap.Reconstruction(self.dir_path)
            print(f"Cameras: {len(recon.cameras)}")
            print(f"Images: {len(recon.images)}")
            print(f"Points3D: {len(recon.points3D)}")
            
        except Exception as e:
            print(f"Error reading COLMAP data: {e}")
            self.cameras, self.images, self.points3d = [None, None, None]
            return 
                    
        self.cameras,self.images,self.points3d = [recon.cameras, recon.images, recon.points3D]


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