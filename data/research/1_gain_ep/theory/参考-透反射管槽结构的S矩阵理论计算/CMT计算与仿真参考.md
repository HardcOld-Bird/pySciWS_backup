## 附录 A 衍射模式分布图

对于图 A（a）所示的周期性超构表面，由于其固有的周期性，系统支持多个对应于不同衍射模式的反射和透射通道。通过空间傅里叶变换，可将该周期结构从实空间转换至动量空间，从而得到如图 A（b）所示的动量空间中的衍射模式分布图。该分布图由自由空间中波数 $k(k=2\pi/\lambda,\lambda=f/c_{0}$ 为入射波波长，f 为入射波频率， $c_{0}$ 为声速）和入射波 x 方向波矢分量 $k_{x}$ （沿超构表面方向的波矢， $k_{x}=k\sin\theta_{i},\quad\theta_{i}$ 为入射角）张成的动量空间构成。

![](images/64b425d20ac6a2ac595cb5fa22995177d28cc547affb46e56462eec16e4b9cd7.jpg)

![](images/0ee1aecb34618282ff3a15cb6fe87155f47524e334c5d8c0cd54876c934c9b32.jpg)
图 A (a) 实空间中超构表面由不同衍射模式叠加而成的散射（包括反射和透射）现象。(b) 动量空间中超构表面的衍射模式分布图。

衍射模式分布图由反射和透射 Wood 异常（WA）线构成，其数学表达式为 $k_{x} + m(2\pi/D) = \pm n_{0}k$ （其中 m 为表示衍射模式阶次的任意整数。 $n_{0}$ 为背景介质的折射率，在本文的研究中背景介质为空气， $n_{0} = 1$ 。当波矢位于第 m 阶（ $m = 0, \pm 1, \pm 2, \ldots$ ）WA 线上方时，第 m 阶次衍射模式的传播反射波开始出现。衍射模式分布图以声锥线 $k = (1/n_{0})k_{x}$ 为边界，该线以下区域仅存在表面倏逝波。由于此时 $k_{x} > k$ ，散射（反射或透射）波矢垂直于超构表面的分量为虚数（ $k_{\perp} = \sqrt{k^{2} - k_{x}^{2}}$ ），故这些散射波模式在垂直于超构表面方向会急剧衰减，只能沿超构表面表面传播。WA 线将衍射模式分布图划分为支持不同衍射模式组合的参数区域，如图 A（b）所示。对于本文的研究内容而言，本文仅关注双端口反射超构表面（第 2、3 章）和四端口透射超构表面（第 4 章），即图 A（b）中绿色区域所示仅支持 3 个衍射模式的参数区域。在这一区域中，在左斜入射（右斜入射）情况下，只会存在第 0 和 -1 阶（第 0 和 +1 阶）两个衍射传播模式。根据前面提到的 k 与 $k_{x}$ 的变换关系，衍射模式分布图中的任一坐标点 $(k_{x}, k)$ 均可映射至实空间中的入射角 $\theta_{i}$ 与频率 f。通过模式耦合理论（CMT），可计算出衍射模式分布图中各衍射阶次反射系数的幅值与相位谱，该计算结果仅取决于超构表面的具体几何参数。

## 附录 B 模式耦合理论（Coupled Mode Theory, CMT）

## B. 1 反射型梯度相位超构表面

如图 B.1 所示，一个梯度相位超构表面的每个周期由 S 个矩形管槽填充而成。每个管槽的宽度为 w，高度为 h，其底部为硬边界，其中填充等效介质（第 s 个管槽中填充介质的折射率为 $n_{s}$ ）。这些管槽被宽度为 d 的硬边界壁分割开来，故超构表面的周期长度为 $D = Na = N(w + d)$ 。

![](images/c845d148941654373c68737d8685ba83e94bca5fa98bc75cbea593c770de8805.jpg)
图 B.1 反射型无损梯度相位超构表面结构图，可以利用模式耦合理论（CMT）求解其中的传播波衍射模式。

对于该结构，计算域被分为两个区域：区域Ⅰ（有入射波和反射波传播的上自由空间）和区域Ⅱ（超构表面内部结构空间），可以写出这两个区域的总声压场：

$$
P ^ {\mathrm{I}} = \exp (i k _ {x} ^ {- 1} x + i k _ {z} ^ {- 1} z) + \sum_ {m} r _ {m} \exp \left(i k _ {x} ^ {m} x - i k _ {z} ^ {m} z\right),\tag{B. 1.1}
$$

$$
\begin{array}{r} P _ {s} ^ {\mathrm{II}} = \mathrm{rect} \left(\frac {x - x _ {s}}{w}\right) \sqrt {2 - \delta_ {0 , p}} \cos \left[ k _ {x s} ^ {p} \left(x - x _ {s} + \frac {w}{2}\right) \right] [ a _ {s} ^ {+} \exp (i k _ {z s} ^ {p} z) + \\ a _ {s} ^ {-} \exp (- i k _ {z s} ^ {p} (z - h)) ], \quad (n = 1, 2, \dots N), \end{array}\tag{B. 1.2}
$$

其中 $P_{s}^{II}$ 是第 s 个管槽中的声压场， $k_{x}^{m}=k_{x}^{in}+m(2\pi/D)$ 和 $k_{z}^{m}=\sqrt{k^{2}-(k_{x}^{m})^{2}}$ 是第 m 阶次衍射模式的 x 和 z 分量波矢。需要注意的是，考虑到梯度相位项的影响，在第 2 章使用一个修正编号 m'代表衍射阶次 m，两者关系为 $m'=m-1$ 。在本研究中，只考虑附录 A 中提到的双端口反射超构表面，其只支持 -1, 0, +1 总共 3 阶传播波衍射模式，为了更加精确，本节在计算中考虑了 21 个衍射模式。这些更高阶的模式，虽然都是倏逝波模式，仍然会通过其近场衰减效应影响总声压场。rect(.)为矩形函数，确保在管槽之间的硬边界壁中不存在声压场， $x_{s}$ 是第 s 个管槽的中心位置， $a_{s}^{+}$ 和 $a_{s}^{-}$ 为管槽中正向声波与反向声波的复振幅。 $k_{xs}^{p}=p\frac{\pi}{w}$ 和 $k_{zs}^{p}=\sqrt{(n_{s}k)^{2}-(k_{xs}^{p})^{2}}$ 为管槽中第 p 个波导模式的 x 和 z 分量波矢。由于管槽宽度 w 远小于波长（深亚波长结构），所以在管槽中只考虑平面波模式（p=0），由此公式（B.1.2）可以简化为：

$$
\begin{array}{r} P _ {s} ^ {\mathrm{II}} = \mathrm{rect} \left(\frac {x - x _ {s}}{w}\right) [ a _ {s} ^ {+} \mathrm{exp} (i n _ {s} k z) + a _ {s} ^ {-} \mathrm{exp} (- i n _ {s} k (z - h)) ], \\ (s = 1, 2, \ldots S), \end{array}\tag{B. 1.3}
$$

通过利用声压与振动速度之间的关系： $v_{z} = \frac{-1}{iw\rho}\frac{\partial P}{\partial z}$ ，有：

$$
v ^ {\mathrm{I}} = \frac {- 1}{\omega \rho_ {0}} [ k _ {z, - 1} \mathrm{exp} (i k _ {x} ^ {- 1} x + i k _ {z} ^ {- 1} z) - \sum_ {m} k _ {z} ^ {m} r _ {m} \mathrm{exp} (i k _ {x} ^ {m} x - i k _ {z} ^ {m} z) ],\tag{B.1.4}
$$

$$
v _ {s} ^ {\mathrm{II}} = - \mathrm{rect} \Big (\frac {x - x _ {s}}{w} \Big) \frac {n _ {s} k}{\omega \rho_ {s}} [ a _ {s} ^ {+} \mathrm{exp} (i n _ {s} k z) - a _ {s} ^ {-} \mathrm{exp} \big (- i n _ {s} k (z - h) \big) ],
$$

$$
(s = 1, 2, \dots S),\tag{B. 1.5}
$$

其中 $\rho_{s} = n_{s}\rho_{0}$ ，由于管槽为硬边界单向封闭结构，有 $v_{s}^{\mathrm{II}}(z = h) = 0$ ，则

$$
a _ {s} ^ {-} = a _ {s} ^ {+} \exp (i n _ {s} k h).\tag{B. 1.6}
$$

假设 $a_{s}^{+}=a_{s}$ ，并结合公式（B.1.4）到（B.1.6），有

$$
P _ {s} ^ {\mathrm{II}} = \operatorname{rect} \left(\frac {x - x _ {s}}{w}\right) a _ {s} [ \exp (i n _ {s} k z) + \exp (- i n _ {s} k (z - 2 h)) ], (s = 1, 2, \dots S),\tag{B. 1.7}
$$

$$
v _ {s} ^ {\mathrm{II}} = \operatorname{rect} \left(\frac {x - x _ {s}}{w}\right) \frac {- k}{\omega \rho_ {0}} a _ {s} [ \exp (i n _ {s} k z) - \exp (- i n _ {s} k (z - 2 h)) ], (s = 1, 2, \dots S),\tag{B. 1.8}
$$

利用区域 I 和区域 II 交界处 z=0 的声压和振动速度连续性条件，有

$$
\exp (i k _ {x} ^ {- 1} x) + \sum_ {m} r _ {m} \exp (i k _ {x} ^ {m} x) = \mathrm{rect} \left(\frac {x - x _ {s}}{w}\right) a _ {s} [ 1 + \exp (2 i n _ {s} k h) ],\tag{B. 1.9}
$$

$$
\frac {1}{\omega \rho_ {0}} [ k _ {z} ^ {- 1} \mathrm{exp} (i k _ {x} ^ {- 1} x) - \sum_ {m} k _ {z} ^ {m} r _ {m} \mathrm{exp} (i k _ {x} ^ {m} x) ] =
$$

$$
\mathrm{rect} \left(\frac {x - x _ {s}}{w}\right) \frac {k}{\omega \rho_ {0}} a _ {s} [ 1 - \exp (2 i n _ {s} k h) ].\tag{B. 1.10}
$$

在公式（B.1.9）两边对 $x$ 进行积分，积分区域为 $x_{s} - \frac{w}{2} < x < x_{s} + \frac{w}{2}$ （第 $s$ 个管槽区域），可得

$$
\exp (i k _ {x} ^ {- 1} x _ {s}) \mathrm{sinc} \frac {k _ {x} ^ {- 1} w}{2} + \sum_ {m} r _ {m} \exp (i k _ {x} ^ {m} x _ {s}) \mathrm{sinc} \frac {k _ {x} ^ {m} w}{2} = a _ {s} [ 1 + \exp (2 i n _ {s} k h) ].\tag{B. 1.11}
$$

在公式（B.1.10）两边同乘 $\exp (-ik_x^m x_s)$ 并在 $0 < x < D$ 区域（超构表面的周期）对 $x$ 进行积分，则有

$$
\delta_ {- 1, m} - r _ {m} = \sum_ {n} \frac {k}{k _ {z} ^ {m}} \frac {w}{D} \exp (- i k _ {x} ^ {m} x _ {s}) \operatorname{sinc} \frac {k _ {x} ^ {m} w}{2} a _ {s} [ 1 - \exp (2 i n _ {s} k h) ].\tag{B. 1.12}
$$

通过公式（B.1.11）和（B.1.12），可以得到一个线性方程组，其维度为考虑的衍射模式数×一个周期内的管槽数。解该方程组，即可得到每个衍射模式的复反射系数 $r_{m}$ 。

## B. 2 非厄米管槽式反射型超构表面

基于 1.3.3 节中关于散射奇异点的内容，要在超构表面系统中构建散射奇异点，最关键的是要得到超构表面系统的散射矩阵，即 S 矩阵，然后将 S 矩阵的约当块形式作为目标函数对超构表面结构的几何参数进行优化调制。为了得到超构表面 S 矩阵中的元素，需要通过超构表面的参数解出每一阶次的衍射模式。对于双端口散射系统，需要得到的则是 0 阶次和 ±1 阶次的衍射模式，即镜面反射和逆反射系数。与 B.1 节类似，这些系数可以通过模式耦合理论（CMT）求解，然而与 B.1 节不同的是，在此本节舍弃梯度相位表面的假设，而是采用 1.2.3 节中所介绍的非局域管槽型结构。也就是说，超构表面结构的每个周期由多个不规则排列的单向封闭硬边界管槽构成。另外，为了更加精确，本节在这一方法中考虑管槽中除平面波基础模式之外的高阶模式。同时，为了加入非厄米扰动，本节会在管槽中引入虚声速，作为超构表面系统的本征损耗。

![](images/b2f6167d8551531d3618cace735322b89a74221485badb653e2839d46068dbf5.jpg)
图 B.2 管槽式反射型超构表面示意图，其散射行为可以通过模式耦合理论得到。

由图 B.2 所示，当平面声波入射到非局域管槽式反射型超构表面时，与 B.1 节中的推导类似，可以写出上半自由空间中的总声场：

$$
p _ {t} (x, z) = A _ {i} \exp [ i (k _ {i} \sin \theta_ {i} x + k _ {i} \cos \theta_ {i} z) ] + A _ {i} \sum_ {m} r _ {m} \exp [ i (k _ {x} ^ {m} x - k _ {z} ^ {m} z) ],\tag{B.2.1}
$$

其中 $A_{i}$ 为入射波的振幅， $r_{m}$ 表示第 $m$ 阶次衍射模式的反射系数， $k_{x}^{m} = k_{i}\sin \theta_{i} + mG$ ， $k_{z}^{m} = \sqrt{k_{i}^{2} - k_{x,m}^{2}}$ 为 $m$ 阶反射波的 $x$ 和 $z$ 方向波矢分量。单向封闭管槽中的声场可以写为：

$$
p _ {g} ^ {s} (x, z) = \sum_ {k} a _ {k s} \cos \alpha_ {k s} (x - x _ {s}) \bigl [ \exp (i \beta_ {k s} z) + \exp \bigl (- i \beta_ {k s} (z + 2 d _ {s}) \bigr) \bigr ],\tag{B. 2.2}
$$

$p_{g}^{s}$ 是一个周期内第 s 个管槽中的声场， $a_{ks}$ 为第 s 个管槽内第 k 阶波导模式的幅值， $\alpha_{ks}$ 和 $\beta_{ks}$ 是第 s 个管槽内第 k 阶波导模式的 x 和 y 方向波矢分量， $\alpha_{ks}=k\pi/w_{s}$ ， $\beta_{ks}=\sqrt{k_{s}^{\prime2}-\alpha_{ks}^{2}}$ ， $k_{s}^{\prime}$ 为第 s 个管槽中的总波数，本节用第 s 个管槽中声速的虚部 $c_{i}^{s}$ 作为损耗，则有 $k_{s}^{\prime}=k_{i}/(1+ic_{i}^{s})$ ， $w_{s}$ ， $d_{s}$ 和 $x_{s}$ 分别为第 s 个管槽的宽度，深度和初始位置，自由空间和管槽中的声速实部为 343 m/s。通过利用声压和速度之间的关系 $v=\frac{-1}{i\omega\rho}\frac{\partial p}{\partial z}$ ，有

$$
\begin{array}{r l} & v _ {t} (x, z) = \frac {- A _ {i}}{\omega \rho} [ k _ {i} \mathrm{cos} \theta_ {i} \mathrm{exp} (i (k _ {i} \mathrm{sin} \theta_ {i} x + k _ {i} \mathrm{cos} \theta_ {i} z) \\ & \qquad - \sum_ {m} k _ {z} ^ {m} r _ {m} \mathrm{exp} \big (i (k _ {x} ^ {m} x - k _ {z} ^ {m} z) \big) ], \\ & v _ {g} ^ {s} (x, z) = \frac {- \beta_ {k s}}{\omega \rho} \sum_ {k} a _ {k s} \mathrm{cos} \alpha_ {k l} (x - x _ {s}) \\ & \qquad \big [ \mathrm{exp} (i \beta_ {k s} z) + \mathrm{exp} \big (- i \beta_ {k s} (z + 2 d _ {s}) \big) \big ], \end{array}\tag{B. 2.3}
$$

(B.2.4)

其中介质密度 $\rho = 1.28\ kg/m^{3}$ 。利用管槽和上半自由空间交界处的声压和速度连续性条件，可以写出矩阵乘法形式的线性方程组：

$$
\left[ \begin{array}{c} P _ {1} \\ V _ {1} \end{array} \right] A _ {0} = \left[ \begin{array}{c c} - P _ {2} & P _ {3} \\ - V _ {2} & V _ {3} \end{array} \right] \left[ \begin{array}{c} A _ {1} \\ A _ {2} \end{array} \right],\tag{B. 2.5}
$$

其中 $A_{0}=[0,\cdots,A_{i},\cdots,0]^{T}$ 代表入射声波的幅值， $A_{1}=A_{i}[r_{-M},\cdots,r_{0},\cdots,r_{M}]^{T}$ 代表 $2M+1$ 个衍射模式下的反射波幅值， $A_{2}=[A^{1},\cdots,A^{s},\cdots,A^{S}]^{T}$ ，其中 $A^{s}=A_{i}[\alpha_{1s},\cdots,\alpha_{ks},\cdots,\alpha_{KS}]^{T}$ 代表了管槽内波导模式的幅值，S是一个周期内管槽的数量，K是管槽内波导模式的数量。 $P_{1},P_{2},P_{3},V_{1},V_{2},V_{3}$ 等元素可以通过与正交函数相乘，以积分形式给出。

$$
P _ {1} ^ {s} (k) = \frac {1}{w _ {s}} \int_ {x _ {s}} ^ {x _ {s} + w _ {s}} \exp {(i k _ {i} \sin \theta_ {i} x)} \mathrm{cos} \alpha_ {k s} (x - x _ {s}) d x,\tag{B. 2.6}
$$

$$
P _ {2} ^ {s} (k, m) = \frac {1}{w _ {s}} \int_ {x _ {s}} ^ {x _ {s} + w _ {s}} \exp {(i k _ {x} ^ {m} x)} \mathrm{cos} \alpha_ {k s} (x - x _ {s}) d x,\tag{B. 2.7}
$$

$$
\begin{array}{l} {P _ {3} ^ {s} (k _ {1}, k _ {2}) = \frac {1}{w _ {s}} \big [ 1 + \exp \big (- 2 i \beta_ {k _ {2} s} d _ {s} \big) \big ]} \\ {\int_ {x _ {s}} ^ {x _ {s} + w _ {s}} \cos \alpha_ {k _ {1 s}} (x - x _ {s}) \cos \alpha_ {k _ {2 s}} (x - x _ {s}) d x,} \end{array}\tag{B. 2.8}
$$

$$
V _ {1} ^ {s} (m) = - \frac {k _ {i} \cos \theta_ {i}}{D} \int_ {0} ^ {D} \exp [ i (k _ {i} \sin \theta_ {i} + k _ {x} ^ {m}) x ] d x,\tag{B. 2.9}
$$

$$
V _ {2} ^ {s} (m _ {1}, m) = \frac {k _ {y , m}}{D} \int_ {0} ^ {D} \exp \left[ i \bigl (k _ {x} ^ {m} - k _ {x} ^ {m _ {1}} \bigr) x \right] d x,\tag{B.2.10}
$$

$$
\begin{array}{r l} & V _ {3} ^ {s} (m, k) = - \frac {\beta_ {k s}}{D} \big [ 1 - \exp \big (- 2 i \beta_ {k _ {2} s} d _ {s} \big) \big ] \\ & \int_ {x _ {s}} ^ {x _ {s} + w _ {s}} \cos \alpha_ {k _ {s}} (x - x _ {s}) \exp (- i k _ {x} ^ {m} x) d x, \end{array}\tag{B.2.11}
$$

$m\in 2M + 1$ 表示反射波模式阶次， $m_{1}\in 2M + 1$ 是 $m$ 的正交阶。将公式(B.2.6)-(B.2.11)带入到公式(B.2.5)中的矩阵乘法中，可以解出每个衍射模式下的反射系数 $r_{m}$ ，从而得到上半自由空间的散射场。与B.1节类似，对于双端口散射系统，即使上半自由空间只有三个传播波模式（+1，0，-1阶次），为了结果的精确性本节仍然考虑20个以上的衍射模式（M=10）以及30个以上的管槽中波导模式（K=30）。

## B. 3 透射型超构表面

图 B.3 展示了第 4 章中使用的超构表面构型：“Z”型透射型非局域超构表面，其一个周期由若干透射通道和反射管槽组成。为了求解这一超构表面的各阶次衍射模式，本节将 B.2 节中反射型超构表面的模式耦合理论进行修正，引入了透射通道对声波的作用，并且同时考虑上半空间的反射和下半空间的透射。

![](images/fcfdcef7a064ddfcd97da389c899c6afb9705e7a6c0f880c62bfe164bd27bc48.jpg)
图 B.3 “Z” 型厄米超构表面示意图，其由周期性的透射通道和反射管槽构成，利用模式耦合理论（CMT）可以求解该结构的各阶次衍射模式的反射和透射系数。

首先，将公式（4.5）改写为类似公式（B.2.1）那样的衍射模式形式：

$$
p ^ {\mathrm{u(d)}} (x, z) = \sum_ {m} e ^ {i k _ {x} ^ {m} x} \left[ \psi_ {m} ^ {\mathrm{u(d)} +} e ^ {i k _ {z} ^ {m} z} + \psi_ {m} ^ {\mathrm{u(d)} -} e ^ {- i k _ {z} ^ {m} z} \right],\tag{B. 3.1}
$$

其中 $\Psi_{m}^{u+}$ 和 $\Psi_{m}^{d-}$ 是上方和下方端口的第 m 阶入射波的复振幅， $\Psi_{m}^{u-}=r_{m}^{u}\Psi_{m}^{u+}+t_{m}^{d}\Psi_{m}^{d-}$ 和 $\Psi_{m}^{d+}=t_{m}^{u}\Psi_{m}^{u+}+r_{m}^{d}\Psi_{m}^{d-}$ 则是上方和下方端口的第 m 阶出射波（反射或透射）复振幅。 $r_{m}^{u(d)}$ 和 $t_{m}^{u(d)}$ 表示上（下）半空间中第 m 阶衍射模式下的反射和透射系数， $k_{x}^{m}=k\sin\theta_{i}+mG,\quad\theta_{i}$ 为入射角，本节只考虑超构表面上方端口的单色平面波入射，即 $\Psi_{0}^{u+}=1,\quad\Psi_{\pm1,\pm2,\cdots}^{u+}=0$ ，以及 $\Psi_{m}^{d-}=0$ （对于任意 m）。显然可以通过调制k， $\theta_{i}$ 和D控制可传播的反射和透射模式的数量。在某一角度的单色平面波入射时，如果仅有四个可传播模式存在：上半空间的两个反射模式和下半空间的两个透射模式，一个四端口透射超构表面系统被建立起来。

由于声压和振动速度之间的关系 $v=\frac{-1}{i\omega\rho}\frac{\partial p}{\partial z}$ ，z方向的振速可以写为

$$
v (x, z) = \frac {- 1}{\omega \rho} \sum_ {m} k _ {z} ^ {m} e ^ {i k _ {x} ^ {m} x} \Big [ \psi_ {m} ^ {\mathrm{u(d)} +} e ^ {i k _ {z} ^ {m} z} - \psi_ {m} ^ {\mathrm{u(d)} -} e ^ {- i k _ {z} ^ {m} z} \Big ],\tag{B. 3.2}
$$

其中 $\rho$ 为背景介质的密度。半封闭反射管槽和透射通道中的声压和z方向振速为多个波导模式的叠加：

$$
p _ {c} (x, z) = \sum_ {s} \cos \alpha_ {s c} (x - x _ {c}) \big [ H _ {s c} ^ {+} e ^ {i \beta_ {s c} z} + H _ {s c} ^ {-} e ^ {- i \beta_ {s c} z} \big ],\tag{B. 3.3}
$$

$$
v _ {c} (x, z) = \frac {- \beta_ {s c}}{\omega \rho} \sum_ {s} \cos \alpha_ {s c} (x - x _ {c}) \big [ H _ {s c} ^ {+} e ^ {i \beta_ {s c} z} - H _ {s c} ^ {-} e ^ {- i \beta_ {s c} z} \big ],\tag{B. 3.4}
$$

$$
p _ {l} ^ {\mathrm{u(d)}} (x, z) = \sum_ {s} \cos \alpha_ {s l} \Big (x - x _ {l} ^ {\mathrm{u(d)}} \Big) \Big [ H _ {s l} ^ {\mathrm{u(d)+}} e ^ {i \beta_ {s l} z} + H _ {s l} ^ {\mathrm{u(d)-}} e ^ {- i \beta_ {s l} z} \Big ],\tag{B. 3.5}
$$

$$
v _ {l} ^ {\mathrm{u(d)}} (x, z) = \frac {- \beta_ {s l}}{\omega \rho} \sum_ {s} \cos \alpha_ {s l} \Big (x - x _ {l} ^ {\mathrm{u(d)}} \Big) \Big [ H _ {s l} ^ {\mathrm{u(d)+}} e ^ {i \beta_ {s l} z} - H _ {s l} ^ {\mathrm{u(d)-}} e ^ {- i \beta_ {s l} z} \Big ],\tag{B. 3.6}
$$

其中 $p_{c}$ 和 $v_{c}$ （ $p_{l}^{\mathrm{u(d)}}$ 和 $v_{l}^{\mathrm{u(d)}}$ ）为在透射通道[超构表面上（下）表面第l个反射管槽]中的声压场和z方向振速场。 $H_{sc}^{\pm}$ （ $H_{sl}^{\pm}$ ）为透射通道（第l个反射管槽）中第s个波导模式的幅值。 $\alpha_{sc}=s\pi/w_{c}$ （ $\alpha_{sl}=s\pi/w_{l}^{\mathrm{u(d)}}$ ）， $\beta_{sc}=\sqrt{k^{2}-\alpha_{sc}^{2}}$ （ $\beta_{sl}=\sqrt{k^{2}-\alpha_{sl}^{2}}$ ） $w_{c}$ （ $w_{l}^{\mathrm{u(d)}}$ ）和 $x_{c}$ （ $x_{l}^{\mathrm{u(d)}}$ ）表示透射通道[超构表面上（下）表面第l个反射管槽]的宽度和x维度上的起始位置。

通过利用与 B.1 节和 B.2 节中内容相同的超构表面与上下自由空间交界处的声压和振速连续性条件，有

$$
p ^ {\mathrm{u(d)}} = \left\{ \begin{array}{l l} p _ {c} & x _ {c} <   x <   x _ {c} + w _ {c}, \\ p _ {l} ^ {\mathrm{u(d)}} & x _ {l} ^ {\mathrm{u(d)}} <   x <   x _ {l} ^ {\mathrm{u(d)}} + w _ {l} ^ {\mathrm{u(d)}}, l = 1, \dots , L, \end{array} \right.\tag{B. 3.7}
$$

$$
v ^ {\mathrm{u(d)}} = \left\{ \begin{array}{l l} {v _ {c}} & {x _ {c} <   x <   x _ {c} + w _ {c},} \\ {v _ {l} ^ {\mathrm{u(d)}}} & {x _ {l} ^ {\mathrm{u(d)}} <   x <   x _ {l} ^ {\mathrm{u(d)}} + w _ {l} ^ {\mathrm{u(d)}}, l = 1, \dots , L,} \\ {0} & {\text {其他区域}} \end{array} \right.\tag{B. 3.8}
$$

首先考虑上边界的连续性条件, 设置上边界位置为 z = 0 , 将公式 (B.3.1) - (B.3.6) 带入到公式 (B.3.7) 和 (B.3.8) 的第一个等式中, 在对等式两边同时乘以正交波导模式项后, 可以得到:

$$
M _ {1 1} ^ {C} (\mathbf {\Psi^ {u +}} + \mathbf {\Psi^ {u -}}) = M _ {2 1} ^ {C} (\mathbf {H} _ {\mathbf {C}} ^ {+} + \mathbf {H} _ {\mathbf {C}} ^ {-}),
$$

(B. 3.9)

$$
M _ {1 2} ^ {G \mathrm{u}} (\mathbf {\Psi^ {u +}} + \mathbf {\Psi^ {u -}}) = M _ {2 2} ^ {G \mathrm{u}} (\mathbf {H} _ {\mathbf {G}} ^ {\mathrm{u+}} + \mathbf {H} _ {\mathbf {G}} ^ {\mathrm{u-}}),\tag{B. 3.10}
$$

其中 $\Psi^{u\pm}=\left[\psi_{-M}^{u\pm},\cdots,\psi_{m}^{u\pm},\cdots,\psi_{M}^{u\pm}\right]^{T}$ 代表了2M+1阶次的输入和输出（反射加透射）声波复振幅。 $H_{c}^{\pm}=\left[H_{1c}^{\pm},\cdots,H_{sc}^{\pm},\cdots,H_{Sc}^{\pm}\right]^{T}$ 和 $H_{G}^{u\pm}=\left[H_{1}^{u\pm},\cdots,H_{l}^{u\pm},\cdots,H_{L}^{u\pm}\right]^{T}$ （ $H_{l}^{u\pm}=\left[H_{1l}^{u\pm},\cdots,H_{sl}^{u\pm},\cdots,H_{Sl}^{u\pm}\right]^{T}$ ）代表了在透射通道和反射管槽内与上自由空间交界处的波导模式幅值，S和L是波导模式和反射管槽的数量。 $M_{12}^{Gu}=$

$[M_{12}^{1\mathrm{u}},\dots ,M_{12}^{lu},\dots ,M_{12}^{Lu}]^T$ 和 $M_{22}^{Gu} = \begin{bmatrix} M_{22}^{1\mathrm{u}} & 0 & \dots & \dots & 0\\ 0 & \ddots & 0 & \dots & \vdots \\ \vdots & 0 & M_{22}^{lu} & 0 & \vdots \\ \vdots & \dots & 0 & \ddots & 0\\ 0 & \dots & 0 & & M_{22}^{Lu} \end{bmatrix}$ ，其中的元素为

$$
M _ {1 1} ^ {C} (s, m) = \frac {1}{w _ {c}} \int_ {x _ {c}} ^ {x _ {c} + w _ {c}} e ^ {i k _ {x} ^ {m} x} \cos \alpha_ {s c} (x - x _ {c}) \mathrm{d} x,\tag{B.3.11}
$$

$$
M _ {1 2} ^ {l u} (s, m) = \frac {1}{w _ {l} ^ {u}} \int_ {x _ {l} ^ {u}} ^ {x _ {l} ^ {u} + w _ {l} ^ {u}} e ^ {i k _ {x} ^ {m} x} \cos \alpha_ {s l} (x - x _ {l} ^ {u}) d x,\tag{B. 3.12}
$$

$$
M _ {2 1} ^ {C} (k _ {1}, k _ {2}) = \frac {1}{w _ {c}} \int_ {x _ {c}} ^ {x _ {c} + w _ {c}} \cos \alpha_ {k _ {1} c} (x - x _ {c}) \cos \alpha_ {k _ {2} c} (x - x _ {c}) \mathrm{d} x,\tag{B.3.13}
$$

$$
M _ {2 2} ^ {l \mathrm{u}} (k _ {1}, k _ {2}) = \frac {1}{w _ {l} ^ {\mathrm{u}}} \int_ {x _ {l} ^ {\mathrm{u}}} ^ {x _ {l} ^ {\mathrm{u}} + w _ {l} ^ {\mathrm{u}}} \cos \alpha_ {k _ {1} l} (x - x _ {l} ^ {\mathrm{u}}) \cos \alpha_ {k _ {2} l} (x - x _ {l} ^ {\mathrm{u}}) \mathrm{d} x,\tag{B.3.14}
$$

其中 $k_{1(2)}=1,\cdots,K$ ，将公式（B.3.1）-（B.3.6）带入到公式（B.3.7）和（B.3.8）的第二个等式中，同样在对等式两边同时乘以正交平面波模式项后，可以得到：

$$
M _ {2 2} ^ {l u} (k _ {1}, k _ {2}) = \frac {1}{w _ {l} ^ {u}} \int_ {x _ {l} ^ {u}} ^ {x _ {l} ^ {u} + w _ {l} ^ {u}} \cos \alpha_ {k _ {1} l} (x - x _ {l} ^ {u}) \cos \alpha_ {k _ {2} l} (x - x _ {l} ^ {u}) d x,\tag{B.3.15}
$$

其中 $N_{22}^{Gu}=[N_{22}^{1u},\cdots,N_{22}^{lu},\cdots,N_{22}^{Lu}]$ ，公式（B.3.15）中各元素为

$$
N _ {1} (m, m _ {1}) = - \frac {k _ {z , m}}{D} \int_ {0} ^ {D} e ^ {i (k _ {x, m} - k _ {x, m _ {1}}) x} \mathrm{d} x,\tag{B.3.16}
$$

$$
N _ {2 1} ^ {C} (m, s) = - \frac {\beta_ {s c}}{D} \int_ {x _ {c}} ^ {x _ {c} + w _ {c}} \cos \alpha_ {s c} (x - x _ {c}) e ^ {- i k _ {x} ^ {m} x} \mathrm{d} x,\tag{B.3.17}
$$

$$
N _ {2 2} ^ {l u} (m, s) = - \frac {\beta_ {s l}}{D} \int_ {x _ {l} ^ {u (d)}} ^ {x _ {l} ^ {u} + w _ {l} ^ {u}} \cos \alpha_ {s l} (x - x _ {l} ^ {u}) e ^ {- i k _ {x} ^ {m} x} d x,\tag{B.3.18}
$$

其中 $m \in 2M + 1$ 表示反射（透射）模式的阶次， $m_{1} \in 2M + 1$ 为 $m$ 阶模式的正交模式。同样地，假设超构表面下表面位置 $z = 0$ ，将公式（B.3.1）-（B.3.6）带入到公式 B.3.7）和（B.3.8）中，可以得到与上述公式类似的方程：

$$
M _ {1 1} ^ {C} \big (\mathbf {\Psi^ {d +}} + \mathbf {\Psi^ {d -}} \big) = M _ {2 1} ^ {C} (\mathbf {H ^ {\prime}} _ {\mathbf {C}} ^ {+} + \mathbf {H ^ {\prime}} _ {\mathbf {C}} ^ {-}),\tag{B.3.19}
$$

$$
M _ {1 2} ^ {G \mathrm{d}} \big (\mathbf {\Psi^ {d +}} + \mathbf {\Psi^ {d -}} \big) = M _ {2 2} ^ {G \mathrm{d}} \big (\mathbf {H _ {G} ^ {d +}} + \mathbf {H _ {G} ^ {d -}} \big),\tag{B. 3.20}
$$

$$
N _ {1} \big (\pmb {\Psi} ^ {\mathbf {d} +} - \pmb {\Psi} ^ {\mathbf {d} -} \big) = N _ {2 1} ^ {C} (\mathbf {H ^ {\prime}} _ {\mathbf {C}} ^ {+} - \mathbf {H ^ {\prime}} _ {\mathbf {C}} ^ {-}) + N _ {2 2} ^ {G \mathrm{d}} \big (\mathbf {H} _ {\mathbf {G}} ^ {\mathbf {d} +} - \mathbf {H} _ {\mathbf {G}} ^ {\mathbf {d} -} \big),\tag{B.3.21}
$$

其中 $M_{12}^{Gd}$ ， $M_{22}^{Gd}$ 和 $N_{22}^{Gd}$ 具有与 $M_{12}^{Gu}$ ， $M_{22}^{Gu}$ 和 $N_{22}^{Gu}$ 类似的表达式，不过其中的参数换为超构表面下边界反射管槽的几何参数。 $\mathbf{H}_{\mathbf{G}}^{\mathbf{d} \pm}$ 和 $\mathbf{H}_{\mathbf{C}}^{\prime \pm}$ 为下表面反射管槽与透射通道与自由空间交界处的波导模式幅值，其与 $\mathbf{H}_{\mathbf{G}}^{\mathbf{u} \pm}$ 和 $\mathbf{H}_{\mathbf{C}}^{\pm}$ 可以通过相位矩阵 $\mathbf{T}_{\mathbf{c}}$ 和 $\mathbf{T}_{\mathbf{l}}^{\mathbf{u}(\mathbf{d})}$ 联系起来：

$$
\mathbf {H} _ {\mathbf {C}} ^ {\prime +} = \mathbf {T} _ {\mathbf {c}} \cdot \mathbf {H} _ {\mathbf {C}} ^ {+}, \mathbf {H} _ {\mathbf {C}} ^ {-} = \mathbf {T} _ {\mathbf {c}} \cdot \mathbf {H} _ {\mathbf {C}} ^ {\prime -}\tag{B. 3.22}
$$

$$
\mathbf {H} _ {\mathrm{l}} ^ {\mathrm{u} -} = \mathbf {T} _ {\mathrm{l}} ^ {\mathrm{u}} \cdot \mathbf {H} _ {\mathrm{l}} ^ {\mathrm{u} +}, \mathbf {H} _ {\mathrm{l}} ^ {\mathrm{d} +} = \mathbf {T} _ {\mathrm{l}} ^ {\mathrm{d}} \cdot \mathbf {H} _ {\mathrm{l}} ^ {\mathrm{d} -}\tag{B. 3.23}
$$

其中 $\mathbf{T}_{\mathbf{c}}$ 和 $\mathbf{T}_{\mathbf{l}}^{\mathbf{u}(\mathbf{d})}$ 为对角矩阵，其对角元 $\mathbf{T}_{\mathbf{c}}(s,s) = e^{i\beta_{sc}h_c}$ ， $\mathbf{T}_{\mathbf{l}}^{\mathbf{u}(\mathbf{d})}(s,s) = e^{i2\beta_{sl}h_l^{\mathbf{u}(\mathbf{d})}}$ ， $h_c$ 和 $h_l^{\mathbf{u}(\mathbf{d})}$ 透射通道和上（下）表面反射管槽的高度。定义 $T_G^{\mathbf{u}(\mathbf{d})} = \begin{bmatrix} T_1^{\mathbf{u}(\mathbf{d})} & \cdots & 0 \\ \vdots & \ddots & \vdots \\ 0 & \cdots & T_L^{\mathbf{u}(\mathbf{d})} \end{bmatrix}$ ，可以得到六个线性方程组：

$$
M _ {1 1} ^ {C} (\mathbf {\Psi^ {u +}} + \mathbf {\Psi^ {u -}}) = M _ {2 1} ^ {C} (\mathbf {H} _ {\mathbf {C}} ^ {+} + \mathbf {T _ {c}} \mathbf {H ^ {\prime}} _ {\mathbf {C}} ^ {-}),\tag{B. 3.24}
$$

$$
M _ {1 1} ^ {C} \big (\mathbf {\Psi^ {d +}} + \mathbf {\Psi^ {d -}} \big) = M _ {2 1} ^ {C} (\mathbf {T _ {c}} \mathbf {H _ {C} ^ {+}} + \mathbf {H _ {C} ^ {\prime -}}),\tag{B. 3.25}
$$

$$
M _ {1 2} ^ {G \mathrm{u}} (\mathbf {\Psi} + \mathbf {\Psi^ {u -}}) = M _ {2 2} ^ {G \mathrm{u}} (I + T _ {G} ^ {\mathrm{u}}) \mathbf {H} _ {\mathbf {G}} ^ {\mathbf {u +}},\tag{B. 3.26}
$$

$$
M _ {1 2} ^ {G \mathrm{d}} \big (\pmb {\Psi} ^ {\mathbf {d} +} + \pmb {\Psi} ^ {\mathbf {d} -} \big) = M _ {2 2} ^ {G \mathrm{d}} \big (T _ {G} ^ {\mathrm{d}} + I \big) \mathbf {H} _ {\mathbf {G}} ^ {\mathbf {d} -},\tag{B. 3.27}
$$

$$
N _ {1} (\mathbf {\Psi^ {u +}} - \mathbf {\Psi^ {u -}}) = N _ {2 1} ^ {C} (\mathbf {H _ {C} ^ {+}} - T _ {c} \mathbf {H _ {C} ^ {\prime -}}) + N _ {2 2} ^ {G u} (I - T _ {G} ^ {u}) \mathbf {H _ {G} ^ {u +}},\tag{B. 3.28}
$$

$$
N _ {1} \big (\mathbf {\Psi^ {d +}} - \mathbf {\Psi^ {d -}} \big) = N _ {2 1} ^ {C} (T _ {c} \mathbf {H _ {C} ^ {+}} - \mathbf {H _ {C} ^ {\prime -}}) + N _ {2 2} ^ {G \mathrm{d}} \big (T _ {G} ^ {\mathrm{d}} - I \big) \mathbf {H _ {G} ^ {d -}},\tag{B.3.29}
$$

其中 I 为幺正矩阵，将以上线性方程组写为矩阵形式：

$$
\left[ \begin{array}{c c c c c c} - M _ {1 1} ^ {C} & 0 & M _ {2 1} ^ {C} & M _ {2 1} ^ {C} T _ {c} & 0 & 0 \\ 0 & - M _ {1 1} ^ {C} & M _ {2 1} ^ {C} T _ {c} & M _ {2 1} ^ {C} & 0 & 0 \\ - M _ {1 2} ^ {\text {Gu}} & 0 & 0 & 0 & M _ {2 2} ^ {\text {Gu}} (I + T _ {G} ^ {\text {u}}) & 0 \\ 0 & - M _ {1 2} ^ {\text {Gd}} & 0 & 0 & 0 & M _ {2 2} ^ {\text {Gd}} \big (T _ {G} ^ {\text {d}} + I \big) \\ N _ {1} & 0 & N _ {2 1} ^ {C} & - N _ {2 1} ^ {C} T _ {c} & N _ {2 2} ^ {\text {Gu}} (I - T _ {G} ^ {\text {u}}) & 0 \\ 0 & N _ {1} & - N _ {2 1} ^ {C} T _ {c} & N _ {2 1} ^ {C} & 0 & N _ {2 2} ^ {\text {Gd}} \big (T _ {G} ^ {\text {d}} - I \big) \end{array} \right] \left[ \begin{array}{c} \boldsymbol {\Psi} ^ {\mathbf {u} -} \\ \boldsymbol {\Psi} ^ {\mathbf {d} +} \\ \mathbf {H} _ {\mathbf {C}} ^ {+} \\ \mathbf {H} _ {\mathbf {G}} ^ {\mathbf {u} +} \\ \mathbf {H} _ {\mathbf {C}} ^ {' -} \\ \mathbf {H} _ {\mathbf {G}} ^ {\mathbf {d} -} \end{array} \right] = \left[ \begin{array}{c} M _ {1 1} ^ {C} \boldsymbol {\Psi} ^ {\mathbf {u} +} \\ M _ {1 1} ^ {C} \boldsymbol {\Psi} ^ {\mathbf {d} -} \\ M _ {1 2} ^ {\text {Gu}} \boldsymbol {\Psi} ^ {\mathbf {u} +} \\ M _ {1 2} ^ {\text {Gd}} \boldsymbol {\Psi} ^ {\mathbf {d} -} \\ N _ {1} \boldsymbol {\Psi} ^ {\mathbf {u} +} \\ N _ {1} \boldsymbol {\Psi} ^ {\mathbf {d} -} \end{array} \right]\tag{B. 3.30}
$$

在入射条件 $\Psi^{u+}=[0,\cdots,1,\cdots,0]^{T}$ 和 $\Psi^{d-}=[0,\cdots,0]^{T}$ 下，通过公式（B.3.30），即可得到各阶衍射模式下的反射和透射系数 $r_{m}^{u(d)}$ 和 $t_{m}^{u(d)}$ 。与第3章相同，为了更精确的结果，本节考虑了超过20个衍射模式和超过30个管槽和通道内波导模式。

## 附录 C 仿真设置

图 C 展示了本文研究内容中所有超构表面结构的有限元仿真模型设置，均采用 COMSOL Multiphysics 有限元仿真软件中的声学模块，研究接口为频域，所获得的散射场结果为平面波入射激励下的稳态不含时解（第 4 章中涉及时空涡旋的瞬态仿真设置已在图 4.17（a）中给出）。在所有模型中，研究区域填充材料为空气，在未涉及等效折射率调制的情况下，声速设置为 $c_{0} = 343 \, m/s$ ，密度设置为 $\rho_{0} = 1.29 \, kg/m^{3}$ 。

![](images/bfe57c0449db43385c6f38742e9c4dba14afd5f063a87cf48d875136896f420c.jpg)

![](images/5920e01941bae34c88ea6df6fa2b3ee77402035fe8231843c4899feeea638f4b.jpg)

![](images/a09aeb0eec42d4d1451099898c1723284c6560009fac877485d791e6b653e7a9.jpg)
图 C (a-b) 本文研究内容中在无限周期假设下的反射型和透射型超构表面有限元仿真模型设置。(c) 在有限周期假设下，与实验条件保持一致的透（反）射型超构表面有限元仿真模型设置。

图 C（a）为所有反射型超构表面模型的仿真设置。模型上方设置完美匹配层（Perfectly Matched Layer，PML），用于模拟无限散射空间；结构与 PML 之间为背景压力场，其场定义为平面波入射 $P_{i}=e^{i(-k_{x}x+k_{y}y)}$ ， $k_{x}$ 和 $k_{y}$ 分别为平行和垂直于超构表面的波矢分量。结构两侧施加 Floquet 周期性条件，其 Floquet 波矢为 $k_{x}$ 和 $k_{y}$ ，用于模拟超构表面在无限周期假设下的散射行为。

本文涉及的反射型超构表面主要包括三种结构。第一种是梯度相位等效介质结构，由多个高度和宽度相同、折射率呈梯度分布的反射管槽组成，其不同折射率通过对声速的修改实现，两者关系为 $c_{i}=c_{0}/n_{i}$ ， $n_{i}$ 为第i个管槽的折射率。由于梯度折射率等效介质本质上提供了离散的梯度声程，因此也可采用具有梯度高度分布的反射管槽结构作为替代，即第二种结构，此时反射管槽的高度与折射率的关系为 $h_{i}=h_{0}n_{i}$ ，其中 $h_{0}$ 为任意参考高度，代表超构表面为入射波前提供的初始相位。前两种结构为第2章的研究内容，第3章涉及第三种结构——非厄米非局域反射管槽结构，该结构具有经过优化的不规则管槽分布，且管槽介质具有本征损耗。在这一模型中，本文通过在管槽介质的声速中加入虚部来实现损耗效果，定义为 $c_{i}=c_{0}+ic_{i}$ 。本文在距离结构较远（以排除倏逝场干扰）的扫描线上设置探针，收集入射场与反射场的分布数据，并对其进行空间傅里叶变换，即可求得平面波入射下超构表面结构各阶衍射模式的反射系数。

图 C（b）展示了第 4 章中非局域透射型超构表面的仿真设置，与反射模型类似，区别在于增加了与上半区域镜像对称的下半区域，同样通过设置扫描线探针获得透射系数。图 C（c）为本文所有实验的类比仿真设置，即模拟实验条件下的有限周期模型。其研究区域为边角为入射角 $\theta_{i}$ 的等腰三角形（透射模型则包括下半镜像区域）。超构表面结构与实际样品参数一致，根据实验条件确定三角形区域的大小，在扬声器阵列区域设置相同长度的平面波激励边界，其余边界条件设为散射（吸收）边界，以模拟实验中的吸声棉介质。由于该模型的求解结果为总散射场，为获得反射场和透射场，需设置一个类似模型，将超构表面结构替换为散射（吸收）边界以得到入射场，再用总场结果减去入射场结果，即可得到反射场或透射场的分布图案。该仿真结果对实验具有重要的指导作用。
