スライド2
目的： SfM等で得られたワールド座標 $\mathbf{𝑟}_𝑤$ のガウスを、カメラの外部パラメータ $(\mathbf{𝑅}, \mathbf{𝑇})$ を用いてカメラ座標 $\mathbf{𝑟}_𝑐$ に写像する。

変換式 (アフィン変換):
$\mathbf{𝑟}_𝑐 = \mathbf{𝑅} \mathbf{𝑟}_𝑤 + \mathbf{𝑇}$

ガウスの不変性： ガウス分布はアフィン変換に対して閉じており、変換後もガウス分布の形を保つ。

パラメータの変換：
平均ベクトル: $\boldsymbol{\mu}_c = \mathbf{R} \boldsymbol{\mu}_w + \mathbf{T}$
共分散行列: $\boldsymbol{\Sigma}_c = \mathbf{R} \boldsymbol{\Sigma}_w \mathbf{R}^\mathrm{T}$

スライド3
射影の必要性： カメラ座標 $\mathbf{r}_c = (x, y, z)$ を画像平面 $\mathbf{r}_m = (u, v)$ に変換する。

元の射影式 (非線形):
$\mathbf{r}_m = (\frac{f_x x}{z}, \frac{f_y y}{z})^{T}$
（$z$ による除算が非線形要素）

問題点： 非線形変換はガウスの形を厳密には保たない。

解決策（線形近似）： ガウス中心 $\boldsymbol{\mu}_c$ の周りでヤコビ行列 ($\mathbf{J}$) による一次近似を実行し、射影を線形化する。

結果： 画像座標 $\mathbf{r}_m$ における共分散行列 $\boldsymbol{\Sigma}_m$ を計算。
$$\boldsymbol{\Sigma}_m = \mathbf{J} \boldsymbol{\Sigma}_c \mathbf{J}^\mathrm{T}$$

スライド6
学習パラメータ：位置 ($\boldsymbol{\mu}_i$)、共分散 ($\boldsymbol{\Sigma}_i$)、放射特性（SH係数）、不透明度 ($c_i$)。

$\boldsymbol{\Sigma}$ の制約： $\boldsymbol{\Sigma}$ が常に正定値となるよう、回転行列 ($\mathbf{R}$) とスケーリング行列 ($\mathbf{S}$) に分解して学習。
($\boldsymbol{\Sigma} = \mathbf{R} \mathbf{S} \mathbf{S}^\mathrm{T} \mathbf{R}^\mathrm{T}$)

損失関数 ($L_\text{recon}$): 再構成品質を評価するため、L1損失とD-SSIM損失の複合損失を使用。
$$L_{recon} = (1 - \lambda) L_1 + \lambda L_{\text{D-SSIM}}$$

最適化手法： AdamなどのSGDにより、画像再構成誤差が最小となるよう全パラメータを更新。

スライド12

・学習パラメータ ω_i に対する損失の偏微分：
$$
  \frac{∂L}{∂ω_i} = \frac{∂L}{∂μ_i'} \left(\frac{∂μ_i'}{∂α_i}\right)\left(\frac{∂α_i}{∂ω_i}\right)
           + \frac{∂L}{∂Σ_i'} \left(\frac{∂Σ_i'}{∂α_i}\right)\left(\frac{∂α_i}{∂ω_i}\right)
           + \frac{∂L}{∂L_H} \left(\frac{∂L_H}{∂α_i}\right)\left(\frac{∂α_i}{∂ω_i}\right)
$$
【各項の導出】
$$
\frac{∂μ_i'}{∂α_i} = \log(R_G) \exp(α_i \log R_G) μ_i + T_G  \\
\frac{∂α_i}{∂ω_i}= β α_i (1 - α_i) \\
\frac{∂Σ_i'}{∂α_i} = \log(R_G) \exp(α_i \log R_G) Σ_i \exp(α_i \log R_G^T)
             + \exp(α_i \log R_G) Σ_i \exp(α_i \log R_G^T) \\
\frac{∂L_H}{∂α_i} = -Σ_i \log\left(\frac{α_i}{1 - α_i}\right)
$$
→ これにより、連続かつ可微分な偏導関数を得て安定した最適化が可能。
スライド14

動的物体に追従するガウス中心 $μ_i$ は 動的物体の回転・並進$R_G, T_G$ で表現できる
$$
\mu'_i = R_G\mu_i+T_G
$$
静的／剛体点を連続補間で統一的に扱う
$
\mu'_i = \left(1-a_i\right)\mu_i+a_i\left(R_G\mu_i+T_G\right)
$
ここで,ガウス分布が静的物体に属するなら$a_i=0$,動的物体に属するなら$a_i=1$である.
つまり,
$
\mu'_i = \begin{Bmatrix}
\mu_i \ \ (a_i=0) \\
R_G\mu_i+T_G \ \ (a_i=1)
\end{Bmatrix}
$
剛体所属度 $a_i$ をシグモイド関数で連続表現し微分可能な関数とする
→　勾配計算が可能
$
a_i =  \frac{1}{1+e^{-w_i}}
$
図
剛体の回転を対数写像して線形補間することで,物理的な意味をもたせる.
$
\mu'_i = \exp\{a_i\log(R_G)\}\mu_i+a_iT_G
$


スライド15

ガウス分布が動的物体に追従するなら,その回転はガウス分布の共分散行列自体も回転させる.
[証明]
確率変数(の位置)を剛体に追従させたとき,確率密度自体は変化しないと仮定すれば,
$x'_i=\exp\{a_i\log(R_G)\}x_i+a_iT_G $の期待値Eは,
$
E[\exp\{a_i\log(R_G)\}x_i+a_iT_G] = \exp\{a_i\log(R_G)\}E[x_i]+a_iT_GE[1]
$
ここで,定義より$x_i$は平均$\mu_i$,共分散$\sum_i$の正規分布に従うので,
$
E[x'_i] = \exp\{a_i\log(R_G)\}\mu_i+a_iT_G
$
すなわち$\mu'_i$
また,$x'_i$と$\mu'_i$の偏差の2乗の期待値$E[(x'_i-\mu'_i)(x'_i-\mu'_i)^T]$は,
$
E[(x'_i-\mu'_i)(x'_i-\mu'_i)^T] = E[\exp\{a_i\log(R_G)\}(x_i-\mu_i)(x_i-\mu_i)^T\exp\{a_i\log(R_G)\}^T]
$
$
= \exp\{a_i\log(R_G)\}E[(x_i-\mu_i)(x_i-\mu_i)^T]\exp\{a_i\log(R_G)\}^T
$ 
$E[(x_i-\mu_i)(x_i-\mu_i)^T] = \sum_i$だから,
$
E[(x'_i-\mu'_i)(x'_i-\mu'_i)^T] = \exp\{a_i\log(R_G)\}\sum_i\exp\{a_i\log(R_G)\}^T
$
これは$x'_i$が従うガウス分布の共分散行列を表す

次に共分散行列∑_iは,一般に対称行列かつ正定値であり固有値行列D_iと回転行列R_iに固有値分解できる.
$
\sum_{i} = R_iD_iR^T_i
$
$
D_i = \begin{pmatrix}
λ_1 & 0 & 0 \\
0 & λ_2 & 0 \\
0 & 0 & λ_3 
\end{pmatrix}
$
$
R_i = (p_1,p_2,p_3)
$
としたとき,$p_1,p_2,p_3$はそれぞれ楕円体の軸の方向ベクトル,
$\sqrt{λ_1},\sqrt{λ_2},\sqrt{λ_3}$はそれぞれの方向ベクトルに対応する軸の長さ

平均が$\mu_i$,共分散行列$\sum_i$が固有値分解の形,
$
\sum_i  = I\begin{pmatrix}
λ_1 & 0 & 0 \\
0 & λ_2 & 0 \\
0 & 0 & λ_3 
\end{pmatrix}I^T = \begin{pmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{pmatrix} \begin{pmatrix}
λ_1 & 0 & 0 \\
0 & λ_2 & 0 \\
0 & 0 & λ_3 
\end{pmatrix}\begin{pmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{pmatrix}^T
$
で与えられるガウス分布を考える.
$I$は$\sum_i$の固有ベクトルの向きがそれぞれx,y,z軸であることを意味する.
このガウス分布の中心を$\mu_i \to R_i\mu_i$と変換すると,
上記よりその共分散行列もまた$I\sum_iI^T \to R_iI \sum_iI^T(R_i)^T = (R_iI) \sum_i(R_iI)^T$と変化する.
$R_i$は回転行列でありこれを$I$の左からかけることは$I$を$R_i$だけ回転することを意味し,
すなわち固有値ベクトル(x,y,z軸の向き)をそれぞれ$R_i$だけ回転したことを意味する.

今,
Σ_i′ = R_G(α_i) Σ_i R_G(α_i)^T
$
剛体回転の影響を分布形状に反映し動的変化を再現

さらに

スライド１６

各剛体のR, Tをグローバル座標で統一的に定義

所属度をソフトマックス関数で多変数表現

ガウス中心：
μ_i′ = exp(Σ a_ij log R_ij) μ_i + Σ a_ij T_ij

共分散：
Σ_i′ = exp(Σ a_ij log R_ij) Σ_i exp(Σ a_ij log R_ij)^T


複数剛体に追従するガウス分布を統合的に再構成

各剛体のR, Tをグローバル座標で統一的に定義

所属度をソフトマックス関数で多変数表現

ガウス中心：
$
μ_i′ = \exp\left(Σ a_{ij} \log R_{ij}\right) μ_i + Σ a_{ij} T_{ij}
$
共分散：
$
Σ_i′ = \exp\left(Σ a_{ij} \log (R_{ij})\right) Σ_i \exp(Σ a_{ij} \log (R_{ij}))^T
$
複数剛体に追従するガウス分布を統合的に再構成