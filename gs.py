# ================================================================
#  Gaussian Splatting (Unofficial Implementation)
#  Copyright (c) 2025 Taisei Kuwana
#  Master's Student (M1), Graduate School of Engineering,
#  Nagoya Institute of Technology, Japan, cmk14002@ict.nitech.ac.jp
#  Licensed under the MIT License.
# ================================================================
from gs_module import Control

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

