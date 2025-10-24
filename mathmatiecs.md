## これは,ガウシアンスプラフティング実装において用いる数学の公式・定理の説明とその証明を行うものである.
参考にしてほしい.

### 全微分可能なスカラー関数の極小点の十分条件を与える定理
$n$次元列ベクトル$\bold{p}$の全微分可能なスカラー値関数$ f: \mathbb{R^n}\to \mathbb{R},\ \bold{p} \to f(\bold{p}) $
に対して以下が成り立つことは,$\ \bold{p}=\bold{p_0}$で$f$が極小点を取るための十分条件である.
$$
\begin{aligned}
(1):\ &\nabla f = \bold{0} \\
(2):\ &\forall \bold{x} \in \mathbb{R^n},\ \bold{x}^T\frac{∂\nabla f}{∂\bold{p}^T}\bold{x} \gt 0 
\end{aligned}
$$
ここで,ベクトルは全て列ベクトルである.
#### [証明]
スカラー関数$f$の全微分$df$は,
$$
\displaystyle \mathrm{d}f = \bold{p}^T\bold{\nabla} f \tag{1}
$$
$\bold{p}=\bold{p_0}$で$f$が極小値を取る十分条件は,
ベクトル$\bold{p}$の微小量$\mathrm{d}\bold{p}$を一変数$t$についての媒介変数表示
$$
\mathrm{d}\bold{p} = \mathrm{d}\bold{g}(t) \tag{2}
$$
で与えたとき,
$$
 \frac{\mathrm{d}\bold{g}^T}{\mathrm{d}t}\frac{\partial}{\partial\bold{p}}\left[\frac{\frac{\mathrm{d}\bold{g}^T}{\mathrm{d}t}\frac{\mathrm{\partial}f}{\mathrm{\partial}\bold{p}}}{\sqrt{\left(\frac{\mathrm{d}\bold{g}^T}{\mathrm{d}t}\frac{\mathrm{d}\bold{g}}{\mathrm{d}t}\right)}}\right] \gt 0 \tag{3}
$$

を満たし,かつ$f$の勾配ベクトル$ \frac{\partial f}{\partial \bold{p}} = \bold{0}$となることである.
ここで,$\frac{\mathrm{d}\bold{g}}{\mathrm{d}t} ≠ \bold{0}$であって,$\bold{g}(t)=\bold{p_0}$を満たす$t$が存在しその点で2階微分可能な
任意のベクトル値関数$\bold{g}:\mathbb{R}\to\mathbb{R^n}$について,式(3)は常に成り立たなければならない.
実際は,$\bold{g}(t)=\bold{p_0}$の近傍で式(3)が成り立てば十分であるが,
近傍の議論が大変なので省略した.本定理を用いるときは,単位円の内側について考えれば十分である.
式(3)の左辺は,方向微分を同じ方向で微分したもの(分母を取り払っていることに注意)であり,
1変数関数の場合の極小値を持つ十分条件は,その関数の二階微分が0より
大きくなることであることを考えば容易に理解できる.
次にこれをより簡単な形に同値変形していく.
式(3)の左辺を分数関数の微分法を用いて以下のように変形する.
$$
 \frac{\mathrm{d}\bold{g}^T}{\mathrm{d}t}\frac{\partial}{\partial\bold{p}}\left[\frac{\frac{\mathrm{d}\bold{g}^T}{\mathrm{d}t}\frac{\mathrm{\partial}f}{\mathrm{\partial}\bold{p}}}{\sqrt{\left(\frac{\mathrm{d}\bold{g}^T}{\mathrm{d}t}\frac{\mathrm{d}\bold{g}}{\mathrm{d}t}\right)}}\right] 
 =  \frac{\mathrm{d}\bold{g}^T}{\mathrm{d}t}\frac{\sqrt{\left(\frac{\mathrm{d}\bold{g}^T}{\mathrm{d}t}\frac{\mathrm{d}\bold{g}}{\mathrm{d}t}\right)}\frac{\partial}{\partial\bold{p}}\left[\frac{\mathrm{d}\bold{g}^T}{\mathrm{d}t}\frac{\mathrm{\partial}f}{\mathrm{\partial}\bold{p}}\right]-\frac{\mathrm{d}\bold{g}^T}{\mathrm{d}t}\frac{\mathrm{\partial}f}{\mathrm{\partial}\bold{p}}\frac{\frac{\partial}{\partial\bold{p}}\left[\frac{\mathrm{d}\bold{g}^T}{\mathrm{d}t}\frac{\mathrm{\partial}\bold{g}}{\mathrm{\partial}\bold{p}}\right]}{2\sqrt{\left(\frac{\mathrm{d}\bold{g}^T}{\mathrm{d}t}\frac{\mathrm{d}\bold{g}}{\mathrm{d}t}\right)}}}{\frac{\mathrm{d}\bold{g}^T}{\mathrm{d}t}\frac{\mathrm{d}\bold{g}}{\mathrm{d}t}} \tag{4}
$$
ここで式(4)右辺の分母は$\frac{\mathrm{d}\bold{g}^T}{\mathrm{d}t}\frac{\mathrm{d}\bold{g}}{\mathrm{d}t} \geqq 0$だから,式(3)の両辺にそれを乗じても一般性を失わない.
また積の微分法と$\frac{\partial f}{\partial \bold{p}} = \bold{0}$を用いると,

$$
\frac{\partial}{\partial\bold{p}}\left[\frac{\mathrm{d}\bold{g}^T}{\mathrm{d}t}\frac{\mathrm{\partial}f}{\mathrm{\partial}\bold{p}}\right] = \frac{\partial}{\partial\bold{p}}\left[\frac{\mathrm{d}\bold{g}^T}{\mathrm{d}t}\right]\frac{\mathrm{\partial}f}{\mathrm{\partial}\bold{p}}+\frac{\partial}{\partial\bold{p}^T}\left[\frac{\mathrm{\partial}f}{\mathrm{\partial}\bold{p}}\right]\frac{\mathrm{d}\bold{g}}{\mathrm{d}t} = \frac{\partial}{\partial\bold{p}^T}\left[\frac{\mathrm{\partial}f}{\mathrm{\partial}\bold{p}}\right]\frac{\mathrm{d}\bold{g}}{\mathrm{d}t} \tag{5}
$$

$$
\frac{\mathrm{d}\bold{g}^T}{\mathrm{d}t}\frac{\mathrm{\partial}f}{\mathrm{\partial}\bold{p}}\frac{\frac{\partial}{\partial\bold{p}}\left[\frac{\mathrm{d}\bold{g}^T}{\mathrm{d}t}\frac{\mathrm{\partial}\bold{g}}{\mathrm{\partial}\bold{p}}\right]}{2\sqrt{\left(\frac{\mathrm{d}\bold{g}^T}{\mathrm{d}t}\frac{\mathrm{d}\bold{g}}{\mathrm{d}t}\right)}} = \bold{0}\tag{6}
$$
式(5),(6)を式(4)に代入すると,

$$
\frac{\mathrm{d}\bold{g}^T}{\mathrm{d}t}\frac{\partial}{\partial\bold{p}^T}\left[\frac{\mathrm{\partial}f}{\mathrm{\partial}\bold{p}}\right]\frac{\mathrm{d}\bold{g}}{\mathrm{d}t} \gt 0 \tag{7}
$$
は式(3)の同値変形である.
また,$\sqrt{\left(\frac{\mathrm{d}\bold{g}^T}{\mathrm{d}t}\frac{\mathrm{d}\bold{g}}{\mathrm{d}t}\right)}$はスカラーであって,仮定より0より大きいから取り払っても一般性を失わない.
$\frac{\mathrm{d}\bold{g}}{\mathrm{d}t}$は,点$\bold{p}=\bold{p_0}$から任意の方向を向いた方向ベクトルのスカラー倍であって,スカラー倍の影響はそれが0より大きいから取り払っても大きくしても一般性を失わないことに注意すると,
結局式(3)は,極小値の候補から任意の方向を向いた単位方向ベクトル$\bold{v}$ 全てに対して,
$$
\bold{v}^T\frac{\partial}{\partial\bold{p}^T}\left[\frac{\mathrm{\partial}f}{\mathrm{\partial}\bold{p}}\right]\bold{v} \gt 0 \tag{8}
$$
を満たすことと同値であり,
$$
\forall \bold{x} \in \mathbb{R^n}, \ \ \bold{x}^T\frac{\partial}{\partial\bold{p}^T}\left[\frac{\mathrm{\partial}f}{\mathrm{\partial}\bold{p}}\right]\bold{x} \gt 0 \tag{9}
$$
を満たすことと同値である.
$\frac{\partial}{\partial\bold{p}^T}\left[\frac{\mathrm{\partial}f}{\mathrm{\partial}\bold{p}}\right]$はヘッシアン行列と呼ばれるもので,ヤコビ行列$J$と以下の関係がある.
$$
J\left[ \bold{\nabla} f \right] = \frac{\partial}{\partial\bold{p}^T}\left[\frac{\mathrm{\partial}f}{\mathrm{\partial}\bold{p}}\right] \tag{10}
$$
本定理の式(2)は$f$の勾配のヤコビ行列$J\left[ \bold{\nabla} f \right]$あるいはヘッシアン行列が正定値であることを要求するものであり,しかも偏微分の順序を交換できるなら正定値かつ対称行列になり,固有値が全て正となる固有値行列と固有ベクトル行列に分解(固有値分解)できる.
### 備考
単位方向ベクトル$\bold{v}$は,二変数スカラー関数に対しては極座標系,
三変数に関しては,球面座標系を考えることで容易に構成できるが,一般の超球面座標系を与える方法を説明する.ただし,方向ベクトルの集合全体に全射ではあるが単射ではないようなものを構成する.
理由は,一対一に対応させるのが大変で,それを必要する場合はそこまでないからである.
それよりも,直感的に理解できる構成を重視した.証明は面倒なので省略.
$\bold{v} = (x_1,x_2,x_3,...,x_n) \in \{\bold{r}\in \mathbb{R^n},\|\bold{r}\|_2 = 1\}$であって,$\bold{v}$を第一軸成分の単位ベクトル$\bold{e_1}$との内積を取れば,それは$\bold{v}$の第一軸成分を表しているから,
$$
x_1 =  \bold{e_1}\cdot \bold{v} \tag{1}
$$
$\bold{v} - x_1\bold{e_1}$と第二成分の単位ベクトル$\bold{e_2}$との内積を取れば,それは$\bold{v}$の第二軸成分を表しているから,
$$
x_2 =  \bold{e_2}\cdot \left(\bold{v} - x_1\bold{e_1}\right) = \bold{e_2}\cdot\bold{v}-x_1\bold{e_1}\cdot\bold{e_2} \tag{2}
$$
これを繰り返していけば,構成していける.
面倒になったので中断.








<!--$$
\frac{df}{\sqrt{(dx)^2+(dy)^2}} = \frac{1}{\sqrt{1+\left(\frac{dy}{dx}\right)^2}}\left(\frac{∂f}{∂x}+\frac{∂f}{∂y}\frac{dy}{dx}\right)=g
$$
$$
\frac{dg}{\sqrt{(dx)^2+(dy)^2}} = \frac{1}{\sqrt{1+\left(\frac{dy}{dx}\right)^2}}\left(\frac{∂g}{∂x}+\frac{∂g}{∂y}\frac{dy}{dx}\right) \gt 0
$$
$$
\frac{∂g}{∂x} = -\frac{\frac{dy}{dx}\frac{d^2y}{dx^2}}{\sqrt{\left(1+\left(\frac{dy}{dx}\right)^2\right)^3}}\left(f_x+f_y\frac{dy}{dx}\right) + \frac{1}{\sqrt{1+\left(\frac{dy}{dx}\right)^2}}\left(f_{xx}+f_{xy}\frac{dy}{dx}+f_y\frac{d^2y}{dx^2}\right)
$$
$$
\frac{∂g}{∂y} = -\frac{\frac{d^2y}{dx^2}}{\sqrt{\left(1+\left(\frac{dy}{dx}\right)^2\right)^3}}\left(f_x+f_y\frac{dy}{dx}\right) + \frac{1}{\sqrt{1+\left(\frac{dy}{dx}\right)^2}}\left(f_{xy}+f_{yy}\frac{dy}{dx}+\frac{f_y}{\frac{dy}{dx}}\frac{d^2y}{dx^2}\right)
$$
$$
\frac{dg}{\sqrt{(dx)^2+(dy)^2}} = -\frac{2\frac{dy}{dx}\frac{d^2y}{dx^2}}{\left(1+\left(\frac{dy}{dx}\right)^2\right)^2}\left(f_x+f_y\frac{dy}{dx}\right)
$$
$$
+\frac{1}{1+\left(\frac{dy}{dx}\right)^2}\left(2f_{xy}\frac{dy}{dx}+2f_y\frac{d^2y}{dx^2} + f_{yy}\left(\frac{dy}{dx}\right)^2+f_{xx}\right)
$$-->