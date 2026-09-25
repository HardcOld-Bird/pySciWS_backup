# Perturbative Analysis of Spectral Singularities and Their Optical Realizations

Ali Mostafazadeh and Saber Rostamzadeh

Departments of Mathematics and Physics, Koç University,

Sarıyer 34450, Istanbul, Turkey

amostafazadeh@ku.edu.tr

We develop a perturbative method of computing spectral singularities of a Schrödinger operator defined by a general complex potential that vanishes outside a closed interval. These can be realized as zero-width resonances in optical gain media and correspond to a lasing effect that occurs at the threshold gain. Their time-reversed copies yield coherent perfect absorption of light that is also known as antilasering. We use our general results to establish the exactness of the n-th order perturbation theory for an arbitrary complex potential consisting of n delta-functions, obtain an exact expression for the transfer matrix of these potentials, and examine spectral singularities of complex barrier potentials of arbitrary shape. In the context of optical spectral singularities, these correspond to inhomogeneous gain media.

Pacs numbers: 03.65.-w, 03.65.Nk, 42.25.Bs, 24.30.Gd

## I. INTRODUCTION

A spectral singularity is a well-known mathematical concept $[1]$ with an interesting physical counterpart; it corresponds to a zero-width resonance $[2]$ . As shown in Refs. $[2, 3]$ , such resonances can be realized in optical systems consisting of a gain medium. They give rise to a particular lasing effect that occurs at the threshold gain $[4]$ . Since the publication of $[2]$ there has been a growing interest in the study of the physical applications of spectral singularities $[3–8]$ . In particular, it turns out that a time-reversed copy of an optical spectral singularity (OSS) $[2]$ corresponds to a coherent perfect absorption of light $[6]$ . This is the basic physical phenomenon occurring in an antilaser $[9]$ .

The investigation of OSSs that is conducted in $[2-4]$ relies on the assumption that the gain (attenuation) coefficient is constant throughout the gain (loss) region(s). This is a simplifying condition that is almost impossible to fulfill in practice. In order to have more realistic models displaying an OSS we need to develop computational techniques for the cases that the optically active medium is inhomogeneous. The first step in this direction is taken in $[7]$ , where a semiclassical method of calculating spectral singularities is employed. The purpose of the present paper is to develop an alternative method of exploring spectral singularities that is based on perturbation theory. This is especially useful, because for most optically active media, the imaginary part of the effective potential, that is responsible for the emergence of an OSS, is several orders of magnitude smaller than its real part $[10]$ .

Consider the time-independent Schödinger equation

$$
- \psi^ {\prime \prime} (x) + v (x) \psi (x) = k ^ {2} \psi (x),\tag{1}
$$

where $x \in \mathbb{R}$ and $v$ is a complex-valued potential that vanishes outside the interval [0, 1]. We can express the general solution of (1) as

$$
\psi (x) = \left\{ \begin{array}{c l} A _ {-} e ^ {i k x} + B _ {-} e ^ {- i k x} & \text { for } \quad x <   0, \\ A _ {0}   \phi_ {1} (x; k) + B _ {0}   \phi_ {2} (x; k) & \text { for } \quad 0 \leq x \leq 1, \\ A _ {+} e ^ {i k x} + B _ {+} e ^ {- i k x} & \text { for } \quad x > 1, \end{array} \right.\tag{2}
$$

where $A_{\pm}, B_{\pm}, A_0, B_0$ are complex coefficients and $\phi_1(\cdot; k)$ and $\phi_2(\cdot, k)$ are solutions of (1) on the interval [0, 1] that are determined by the initial conditions [7]:

$$
\phi_ {1} (0; k) = 1, \quad \phi_ {1} ^ {\prime} (0; k) = - i k, \quad \phi_ {2} (0; k) = 1, \quad \phi_ {2} ^ {\prime} (0; k) = 0.\tag{3}
$$

Spectral singularities are given by the real zeros of the $M_{22}$ entry of the transfer matrix M of the system [2]. This is the $2 \times 2$ matrix M that satisfies $\begin{bmatrix} A_{+} \\ B_{+} \end{bmatrix} = M \begin{bmatrix} A_{-} \\ B_{-} \end{bmatrix}$ . Demanding that $\psi$ is a continuously differentiable function (throughout R) and using (3), we find the following expression for the transfer matrix.

$$
\mathbf {M} = \frac {1}{2 i k} \left[ \begin{array}{c c} - e ^ {- i k} [ \Gamma_ {1 +} (k) - 2 \Gamma_ {2 +} (k) ] & e ^ {- i k} \Gamma_ {1 +} (k) \\ e ^ {i k} [ \Gamma_ {1 -} (k) - 2 \Gamma_ {2 -} (k) ] & - e ^ {i k} \Gamma_ {1 -} (k) \end{array} \right],\tag{4}
$$

where

$$
\Gamma_ {j \pm} (k) := \phi_ {j} ^ {\prime} (1; k) \pm i k \phi_ {j} (1; k).\tag{5}
$$

In view of (4) spectral singularities are the real zeros of the Jost function $\Gamma_{1-}$ , [7].

In order to develop a perturbative method of computing spectral singularities we consider potentials of the form

$$
v (x) := \left\{ \begin{array}{c c} v _ {0} (x) + \epsilon   v _ {1} (x) & \text { for } 0 \leq x \leq 1, \\ 0 & \text { otherwise }, \end{array} \right.\tag{6}
$$

where $v_{0}$ is an exactly solvable potential, $\epsilon$ is a real perturbation parameter, and $v_{1}$ is an arbitrary potential. Our aim is to use perturbation theory to compute the solution $\phi_{1}(\cdot;k)$ of the Schrödinger equation (1), the Jost function $\Gamma_{1-}$ , and its real zeros.

The plan of the paper is as follows. In Section 2 we derive a perturbative series expansion for $\phi_1(\cdot ;k)$ and discuss how it can be used to locate the spectral singularities. In Section 3 we use our perturbative method to examine potentials involving one or more delta functions. Here we establish the exactness of the perturbation theory and give an explicit formula for the transfer matrix of the system. In Section 4, we apply our method to an arbitrary piecewise continuous complex barrier potential. In Section 5, we use the results of Section 4 to study the OSS of an infinite planar slab gain medium whose gain/loss coefficient varies along the normal direction. Finally, in Section 6 we give a summary of our findings and concluding remarks.

## II. PERTURBATIVE CALCULATION OF SPECTRAL SINGULARITIES

We start our analysis by introducing the differential operator

$$
L := \frac {d ^ {2}}{d x ^ {2}} - v _ {0} (x) + k ^ {2},\tag{7}
$$

and note that $\phi_1(\cdot ;k)$ and $\phi_2(\cdot ;k)$ are solutions of the differential equation

$$
L \phi (x) = \epsilon v _ {1} (x) \phi (x),\tag{8}
$$

on the interval $[0,1]$ that are uniquely determined by the initial conditions (3). To construct a perturbative solution of (8) we insert the ansatz

$$
\phi (x) = \sum_ {j = 0} ^ {\infty} \phi^ {(j)} (x) \epsilon^ {j}\tag{9}
$$

in (8) and demand that it holds term by term in powers of $\epsilon$ . This yields

$$
L \phi^ {(0)} (x) = 0,\tag{10}
$$

$$
L \phi^ {(\ell)} (x) = v _ {1} (x) \phi^ {(\ell - 1)} (x),\tag{11}
$$

where $\ell=1,2,3,\cdots$ . We can express the general solution of the latter equation in the form

$$
\phi^ {(\ell)} (x) = a _ {\ell} \phi_ {1} ^ {(0)} (x) + b _ {\ell} \phi_ {2} ^ {(0)} (x) + \int_ {0} ^ {x} G (x, y) v _ {1} (y) \phi^ {(\ell - 1)} (y) d y,\tag{12}
$$

where $a_{\ell}, b_{\ell}$ are constant coefficients, $\phi_1^{(0)}$ and $\phi_2^{(0)}$ are linearly-independent solutions of (10), and $G$ is the Green's function for the operator $L$ that can be expressed as [11]:

$$
G (x, y) = \frac {\phi_ {1} ^ {(0)} (y) \phi_ {2} ^ {(0)} (x) - \phi_ {2} ^ {(0)} (y) \phi_ {1} ^ {(0)} (x)}{\phi_ {1} ^ {(0)} (y) \phi_ {2} ^ {(0) ^ {\prime}} (y) - \phi_ {2} ^ {(0)} (y) \phi_ {1} ^ {(0) ^ {\prime}} (y)}.\tag{13}
$$

A convenient choice, that determines $\phi_{1}^{(0)}$ and $\phi_{2}^{(0)}$ in a unique manner, is to demand that they also satisfy the initial conditions (3), i.e.,

$$
\phi_ {1} ^ {(0)} (0) = 1, \quad \phi_ {1} ^ {(0) ^ {\prime}} (0) = - i k, \quad \phi_ {2} ^ {(0)} (0) = 1, \quad \phi_ {2} ^ {(0) ^ {\prime}} (0) = 0.\tag{14}
$$

In this case the denominator of the Green's function that coincides with the Wronskian of $\phi_1^{(0)}$ and $\phi_2^{(0)}$ takes the value $ik$ , and we find

$$
G (x, y) = i k ^ {- 1} \left[ \phi_ {1} ^ {(0)} (x) \phi_ {2} ^ {(0)} (y) - \phi_ {2} ^ {(0)} (x) \phi_ {1} ^ {(0)} (y) \right].\tag{15}
$$

Another advantage of imposing (14) is that it identifies the zeroth-order term in the perturbative expansion of $\phi_{j}(\cdot;k)$ with $\phi_{j}^{(0)}(\cdot;k)$ . Moreover, if we denote the higher order terms in this expansion by $\phi_{j}^{(\ell)}(\cdot;k)\epsilon^{\ell}$ (with $\ell\geq1$ ), so that

$$
\phi_ {j} (x; k) = \sum_ {\ell = 0} ^ {\infty} \phi_ {j} ^ {(\ell)} (x; k) \epsilon^ {\ell}, \quad \phi_ {j} ^ {(0)} (x; k) = \phi_ {j} ^ {(0)} (x),\tag{16}
$$

then (3) and (14) imply that $\phi_j^{(\ell)}(0;k) = \phi_j^{(\ell)'}(0;k) = 0$ for $\ell \geq 1$ . A direct consequence of this relation is that the coefficients $a_{\ell}$ and $b_{\ell}$ of (12) vanish, and we can use (12) to derive the formula:

$$
\phi_ {j} ^ {(\ell)} (x _ {\ell}; k) = \int_ {0} ^ {x _ {\ell}} d x _ {\ell - 1} \int_ {0} ^ {x _ {\ell - 1}} d x _ {\ell - 2} \dots \int_ {0} ^ {x _ {1}} d x _ {0} \phi_ {j} ^ {(0)} (x _ {0}) \prod_ {m = 1} ^ {\ell} G (x _ {m}, x _ {m - 1}) v _ {1} (x _ {m - 1}),\tag{17}
$$

that holds for all $\ell \geq 1$ and $x_{\ell} \in [0,1]$ .

We can use (16) to obtain a perturbative expansion for the Jost functions (5):

$$
\Gamma_ {j \pm} (k) = \sum_ {\ell = 0} ^ {\infty} \Gamma_ {j \pm} ^ {(\ell)} (k) \epsilon^ {\ell}.\tag{18}
$$

In view of (5), (16), and (17), we have

$$
\begin{array}{r l} \Gamma_ {j \pm} ^ {(0)} (k) = & \phi_ {j} ^ {(0) ^ {\prime}} (1) \pm i k \phi_ {j} ^ {(0)} (1), \\ \Gamma_ {j \pm} ^ {(\ell)} (k) = & \int_ {0} ^ {1} d x _ {\ell} \int_ {0} ^ {x _ {\ell}} d x _ {\ell - 1} \dots \int_ {0} ^ {x _ {2}} d x _ {1} \phi_ {j} ^ {(0)} (x _ {1}) \times \\ & \Big \{\left[ G ^ {\prime} (1, x _ {\ell}) \pm i k G (1, x _ {\ell}) \right] v _ {1} (x _ {\ell}) \prod_ {m = 1} ^ {\ell - 1} G (x _ {m + 1}, x _ {m}) v _ {1} (x _ {m}) \Big \}, \end{array}\tag{19}
$$

(20)

where

$$
G ^ {\prime} (x, y) := \partial_ {x} G (x, y) = i k ^ {- 1} \left[ \phi_ {1} ^ {(0) ^ {\prime}} (x) \phi_ {2} ^ {(0)} (y) - \phi_ {2} ^ {(0) ^ {\prime}} (x) \phi_ {1} ^ {(0)} (y) \right].\tag{21}
$$

Next, we recall that in general $v_{0}$ and $v_{1}$ involve a set of complex coupling constants $(\mathfrak{z}_{1},\mathfrak{z}_{2},\cdots,\mathfrak{z}_{n})$ . These together with the perturbation parameter $\epsilon$ are the parameters that enter in the expression (18) for the Jost solutions. Spectral singularities are given by the real values of k for which

$$
\Gamma_ {1 -} (k) = 0.\tag{22}
$$

This is a complex equation involving n complex variables, $z_{1}, z_{2}, \cdots, z_{n}$ , and two real variables, $\epsilon, k$ .

If the unperturbed potential $v_{0}$ is real-valued or more generally does not support a spectral singularity, the emergence of spectral singularities is a consequence of the perturbation $\epsilon v_{1}$ . Otherwise the presence of the latter leads to small changes in the location of the spectral singularities of $v_{0}$ that are given by the real solutions of

$$
\Gamma_ {1 -} ^ {(0)} (k) = 0.\tag{23}
$$

In this case, we can again adopt perturbative theory to construct solutions of (22) using those of (23), [12]. Before, we explore the details of this construction, we consider a situation where perturbation theory yields the exact solution of the problem.

## III. ARRAY OF COMPLEX DELTA-FUNCTION POTENTIALS

Consider the case that $v_{0}(x)=0$ and the perturbation involves an array of Dirac delta-functions [13–16]:

$$
v _ {1} (x) = \sum_ {i = 1} ^ {n} \mathfrak {z} _ {i} \delta (x - a _ {i}),\tag{24}
$$

where $z_{1}, z_{2}, \cdots, z_{n}$ are complex coupling constants, and $a_{1}, a_{2}, \cdots, a_{n}$ are arbitrary positive numbers satisfying

$$
0 <   a _ {1} <   a _ {2} <   \dots <   a _ {n} <   1.\tag{25}
$$

Because $v_{0}=0$ , Eqs. (7), (10), (14), (15), and (21) give

$$
\phi_ {1} ^ {(0)} (x) = e ^ {- i k x}, \qquad \qquad \phi_ {2} ^ {(0)} (x) = \cos (k x),\tag{26}
$$

$$
G (x, y) = \frac {\sin [ k (x - y) ]}{k}, \qquad G ^ {\prime} (x, y) = \cos [ k (x - y) ].\tag{27}
$$

If we substitute (24), (26), and (27) in (17), and use the properties of the delta-function to perform the relevant integrals, we find

$$
\phi_ {j} ^ {(\ell)} (x; k) = \sum_ {i = 1} ^ {n} Z _ {j i} ^ {(\ell)} \sin [ k (x - a _ {i}) ] \theta (x - a _ {i}),\tag{28}
$$

where $\ell\geq1$ , for all $i_{\ell}=1,2,\cdots,n$ and j=1,2,

$$
Z _ {j i _ {\ell}} ^ {(\ell)} := k ^ {- \ell} \mathfrak {z} _ {i _ {\ell}} \sum_ {i _ {1}, i _ {2}, \dots , i _ {\ell - 1} = 1} ^ {n} \phi_ {j} ^ {(0)} (a _ {i _ {1}}) \prod_ {m = 1} ^ {\ell - 1} \mathfrak {z} _ {i _ {m}} \sin [ k (a _ {i _ {m + 1}} - a _ {i _ {m}}) ] \theta (a _ {i _ {m + 1}} - a _ {i _ {m}}),\tag{29}
$$

and $\theta$ stands for the Heaviside step function,

$$
\theta (x) = \left\{ \begin{array}{l l} 1 & \text { for } x \geq 0, \\ 0 & \text { for } x <   0. \end{array} \right.\tag{30}
$$

In view of (25) and (30), the product on the right-hand side of (29) vanishes identically, if $i_{m+1} \leq i_{m}$ . Hence

$$
Z_{ji_{\ell}}^{(\ell)}(k) = \theta (n - \ell)  k^{-\ell}\mathfrak{z}_{i_{\ell}}\sum_{\substack{i_{1} <   i_{2} <   \dots <  i_{\ell -1} = 1}}^{i_{\ell}}\phi_{j}^{(0)}(a_{i_{1}})\prod_{m = 1}^{\ell -1}\mathfrak{z}_{i_{m}}\sin [k(a_{i_{m + 1}} - a_{i_{m}})],\tag{31}
$$

and as a result $\phi_j^{(\ell)}(x;k) = 0$ for all $\ell > n$ . This proves the following theorem.

Theorem: For a point interaction consisting of n delta-functions with arbitrary centers and possibly complex coupling constants, the n-th order perturbation theory is exact.

Next, we compute the Jost functions (18). Using (24), (26), and (27) in (19) and (20), we find

$$
\Gamma_ {1 +} ^ {(0)} (k) = 0, \qquad \Gamma_ {1 -} ^ {(0)} (k) = - 2 i e ^ {- i k}, \qquad \Gamma_ {2 \pm} ^ {(0)} (k) = \pm i k e ^ {\pm i k},\tag{32}
$$

$$
\Gamma_ {j \pm} ^ {(\ell)} (k) = \theta (n - \ell) \left(\frac {\pm 1}{2 i k}\right) ^ {\ell - 1} \sum_ {i _ {1} <   i _ {2} <   \dots <   i _ {\ell} = 1} ^ {n} \Omega_ {i _ {1} j \pm} \prod_ {p = 1} ^ {\ell} \mathfrak {z} _ {i _ {p}} \prod_ {m = 1} ^ {\ell - 1} \left[ 1 - e ^ {\mp 2 i k (a _ {i _ {m + 1}} - a _ {i _ {m}})} \right],\tag{33}
$$

where $\ell\geq1$ , and for all $i_{1}=1,2,\cdots,\ell$ ,

$$
\Omega_ {i _ {1} 1 +} := e ^ {i k (1 - 2 a _ {i _ {1}})}, \qquad \Omega_ {i _ {1} 1 -} := e ^ {- i k}, \qquad \Omega_ {i _ {1} 2 \pm} := \frac {1}{2} \left(1 + e ^ {\mp 2 i k a _ {i _ {1}}}\right) e ^ {\pm i k}.\tag{34}
$$

Substituting (26) in this relation, using the result in (4), and setting $\epsilon = 1$ , we obtain the following expressions for the entries of the transfer matrix corresponding to the potential (24):

$$
M _ {1 1} = 1 + \sum_ {\ell = 1} ^ {n} (2 i k) ^ {- \ell} \sum_ {i _ {1} <   i _ {2} <   \dots <   i _ {\ell} = 1} ^ {n} \prod_ {p = 1} ^ {\ell} \mathfrak {z} _ {i _ {p}} \prod_ {m = 1} ^ {\ell - 1} \left[ 1 - e ^ {- 2 i k (a _ {i _ {m + 1}} - a _ {i _ {m}})} \right]\tag{35}
$$

$$
M _ {1 2} = \sum_ {\ell = 1} ^ {n} (2 i k) ^ {- \ell} \sum_ {i _ {1} <   i _ {2} <   \dots <   i _ {\ell} = 1} ^ {n} e ^ {- 2 i k a _ {i _ {1}}} \prod_ {p = 1} ^ {\ell} \mathfrak {z} _ {i _ {p}} \prod_ {m = 1} ^ {\ell - 1} \left[ 1 - e ^ {- 2 i k (a _ {i _ {m + 1}} - a _ {i _ {m}})} \right]\tag{36}
$$

$$
M _ {2 1} = \sum_ {\ell = 1} ^ {n} (- 2 i k) ^ {- \ell} \sum_ {i _ {1} <   i _ {2} <   \dots <   i _ {\ell} = 1} ^ {n} e ^ {2 i k a _ {i _ {1}}} \prod_ {p = 1} ^ {\ell} \mathfrak {z} _ {i _ {p}} \prod_ {m = 1} ^ {\ell - 1} \left[ 1 - e ^ {2 i k (a _ {i _ {m + 1}} - a _ {i _ {m}})} \right]\tag{37}
$$

$$
M _ {2 2} = 1 + \sum_ {\ell = 1} ^ {n} (- 2 i k) ^ {- \ell} \sum_ {i _ {1} <   i _ {2} <   \dots <   i _ {\ell} = 1} ^ {n} \prod_ {p = 1} ^ {\ell} \mathfrak {z} _ {i _ {p}} \prod_ {m = 1} ^ {\ell - 1} \left[ 1 - e ^ {2 i k (a _ {i _ {m + 1}} - a _ {i _ {m}})} \right].\tag{38}
$$

It is instructive to examine the cases $n = 1,2$ .

For $n = 1$ , we have $v_{1} = \mathfrak{z}_{1}\delta (x - a_{1})$ , and (35)-(38) give

$$
M _ {1 1} = 1 - \frac {i \mathfrak {z} _ {1}}{2 k}, \qquad M _ {1 2} = - \frac {i \mathfrak {z} _ {1} e ^ {- 2 i k a _ {1}}}{2 k}, \qquad M _ {2 1} = \frac {i \mathfrak {z} _ {1} e ^ {2 i k a _ {1}}}{2 k}, \qquad M _ {2 2} = 1 + \frac {i \mathfrak {z} _ {1}}{2 k}.\tag{39}
$$

This agrees with the results of [17]. In particular, a spectral singularity occurs for imaginary values of $\mathfrak{z}_1$ and is given by $k = -i_{\mathfrak{z}_1} / 2$ , as envisaged in [15] and shown in [16].

For $n = 2$ , we have $v_{1} = \mathfrak{z}_{1}\delta (x - a_{1}) + \mathfrak{z}_{2}\delta (x - a_{2})$ , and (35)-(38) give

$$
M _ {1 1} = 1 - \frac {i (\mathfrak {z} _ {1} + \mathfrak {z} _ {2})}{2 k} - \frac {\mathfrak {z} _ {1} \mathfrak {z} _ {2} [ 1 - e ^ {- 2 i k (a _ {2} - a _ {1})} ]}{4 k ^ {2}},\tag{40}
$$

$$
M _ {1 2} = - \frac {i (\mathfrak {z} _ {1} e ^ {- 2 i k a _ {1}} + \mathfrak {z} _ {2} e ^ {- 2 i k a _ {2}})}{2 k} - \frac {\mathfrak {z} _ {1} \mathfrak {z} _ {2} (e ^ {- 2 i k a _ {1}} - e ^ {- 2 i k a _ {2}})}{4 k ^ {2}},\tag{41}
$$

$$
M _ {2 1} = \frac {i (\mathfrak {z} _ {1} e ^ {2 i k a _ {1}} + \mathfrak {z} _ {2} e ^ {2 i k a _ {2}})}{2 k} - \frac {\mathfrak {z} _ {1} \mathfrak {z} _ {2} (e ^ {2 i k a _ {1}} - e ^ {2 i k a _ {2}})}{4 k ^ {2}},\tag{42}
$$

$$
M _ {2 2} = 1 + \frac {i (\mathfrak {z} _ {1} + \mathfrak {z} _ {2})}{2 k} - \frac {\mathfrak {z} _ {1} \mathfrak {z} _ {2} [ 1 - e ^ {2 i k (a _ {2} - a _ {1})} ]}{4 k ^ {2}}.\tag{43}
$$

Spectral singularities are therefore given by the real values of $k$ for which the right-hand side of (43) vanishes. Again this is in complete agreement with the results of [16].

For the cases that $a_{i} := i/(n+1)$ and $z_{1} = z_{2} = \cdots = z_{n}$ , the potential v is locally periodic, and we can use the results of [14] to compute the transfer matrix of the system. We have checked by explicit calculation for small values of n that (35) – (38) give the same expression for the transfer matrix as the one derived in [14].

## IV. COMPLEX BARRIER POTENTIALS

Consider the potentials of the form (6) where for all $x \in [0,1]$ ,

$$
v _ {0} (x) := \mathfrak {z} _ {1}, \qquad v _ {1} (x) := \mathfrak {z} _ {2} f (x),\tag{44}
$$

$\mathfrak{z}_{1}$ and $\mathfrak{z}_{2}$ are complex coupling constants, $f:[0,1]\to\mathbb{C}$ is an integrable function satisfying $\int_{0}^{1}dx|f(x)|\leq1$ , and $|\epsilon\mathfrak{z}_{2}|\ll|\mathfrak{z}_{1}|$ . As explained in Ref. [7], these potentials appear in the study of the OSS of an infinite planar slab gain medium with gain coefficient changing along the normal direction to the slab.

In order to determine the spectral singularities of the potentials given by (6) and (44), we first use (7), (10), (14), (15), and (21) to compute

$$
\phi_ {1} ^ {(0)} (x) = \cos (\mathfrak {n} k x) - i \mathfrak {n} ^ {- 1} \sin (\mathfrak {n} k x), \quad \phi_ {2} ^ {(0)} (x) = \cos (\mathfrak {n} k x),\tag{45}
$$

$$
G (x, y) = (\mathfrak {n} k) ^ {- 1} \sin [ \mathfrak {n} k (x - y) ], \quad G ^ {\prime} (x, y) - i k G (x, y) = \phi_ {1} ^ {(0)} (x - y),\tag{46}
$$

where

$$
\mathfrak {n} := \sqrt {1 - \frac {\mathfrak {z} _ {1}}{k ^ {2}}}.\tag{47}
$$

Clearly, $\mathfrak{n} = 0$ marks a singularity of our construction that we will avoid. If we substitute (45) in (19), we find

$$
\begin{array}{r c l} \Gamma_ {1 -} ^ {(0)} (k) = F _ {0} (\mathfrak {n}, k) & := & - \mathfrak {n} ^ {- 1} k \left[ (\mathfrak {n} ^ {2} + 1) \sin (\mathfrak {n} k) + 2 i \mathfrak {n} \cos (\mathfrak {n} k) \right] \\ & = & \frac {k (\mathfrak {n} + 1) ^ {2} e ^ {i \mathfrak {n} k}}{2 i \mathfrak {n}} \left[ e ^ {- 2 i \mathfrak {n} k} - \left(\frac {\mathfrak {n} - 1}{\mathfrak {n} + 1}\right) ^ {2} \right]. \end{array}\tag{48}
$$

This is consistent with the results of Ref. [4], because it implies that for $\epsilon = 0$ , that corresponds to a constant complex barrier potential, the spectral singularities are determined by the equation:

$$
e ^ {- 2 i \mathfrak {n} k} - \left(\frac {\mathfrak {n} - 1}{\mathfrak {n} + 1}\right) ^ {2} = 0.\tag{49}
$$

Similarly, using (20), (45) and (46), we obtain for all $\ell \geq 1$ ,

$$
\Gamma_ {1 -} ^ {(\ell)} (k) = \mathfrak {z} _ {2} ^ {\ell} F _ {\ell} (\mathfrak {n}, k)\tag{50}
$$

where

$$
F _ {\ell} (\mathfrak {n}, k) := (\mathfrak {n} k) ^ {1 - \ell} \int_ {0} ^ {1} d x _ {\ell} \int_ {0} ^ {x _ {\ell}} d x _ {\ell - 1} \dots \int_ {0} ^ {x _ {2}} d x _ {1} \xi (\mathfrak {n}, k, x _ {1}) \prod_ {m = 1} ^ {\ell - 1} \sin [ \mathfrak {n} k (x _ {m + 1} - x _ {m}) ] \prod_ {p = 1} ^ {\ell} f (x _ {p}),\tag{51}
$$

$$
\begin{array}{r c l} \xi (\mathfrak {n}, k, x) & := & \phi_ {1} ^ {(0)} (x) \phi_ {1} ^ {(0)} (1 - x) \\ & = & \frac {1}{2} \left\{\left(1 + \frac {1}{\mathfrak {n} ^ {2}}\right) \cos (\mathfrak {n} k) - \frac {2 i \sin (\mathfrak {n} k)}{\mathfrak {n}} + \left(1 - \frac {1}{\mathfrak {n} ^ {2}}\right) \cos [ \mathfrak {n} k (2 x - 1) ] \right\}. \end{array}\tag{52}
$$

In particular,

$$
F _ {1} (\mathfrak {n}, k) = \int_ {0} ^ {1} d x   \xi (\mathfrak {n}, k, x) f (x).\tag{53}
$$

In the remainder of this section we explore the application of perturbation theory for treating the following equation whose real solutions yield the spectral singularities.

$$
\Gamma_ {1 -} (k) = \sum_ {\ell = 0} ^ {\infty} F _ {\ell} (\mathfrak {n}, k)   \mathfrak {z} _ {2} ^ {\ell}   \epsilon^ {\ell} = 0.\tag{54}
$$

Suppose that we have a spectral singularity for the unperturbed potential, i.e., when $\mathfrak{z}_{2}=0$ . Let $(\mathfrak{n}_{0},k_{0})$ be the value of $(\mathfrak{n},k)$ at which this spectral singularity is realized. If we turn on the perturbation, i.e., set $\epsilon_{Z2}\neq0$ , this spectral singularity occurs for a new value $(\mathfrak{n}_{\star},k_{\star})$ of $(\mathfrak{n},k)$ . Our aim is to express $k_{\star}$ and $n_{\star}$ as power series in the perturbation parameter $\epsilon$ ,

$$
k _ {\star} = \sum_ {m = 0} ^ {\infty} k _ {m} \epsilon^ {m}, \qquad \mathfrak {n} _ {\star} = \sum_ {m = 0} ^ {\infty} \mathfrak {n} _ {m} \epsilon^ {m},\tag{55}
$$

and determine the coefficients $k_{m}$ and $\mathfrak{n}_m$ , that respectively take real and complex values. To do this we expand $F_{\ell}(\mathfrak{n},k)$ in a power series about $(\mathfrak{n}_0,k_0)$ ,

$$
F _ {\ell} (\mathfrak {n}, k) = \sum_ {p, q = 0} ^ {n} F _ {\ell p q} (\mathfrak {n} - \mathfrak {n} _ {0}) ^ {p} (k - k _ {0}) ^ {q},\tag{56}
$$

$$
F _ {\ell p q} := \frac {1}{p ! q !} \frac {\partial^ {p + q} F _ {\ell} (\mathfrak {n} _ {0} , k _ {0})}{\partial \mathfrak {n} _ {0} ^ {p} \partial k _ {0} ^ {q}},\tag{57}
$$

![](images/5e9e09d760d3f0e8a5c9581380ba718779f7a9a71b2c97303e5afab78af536ff.jpg)
FIG. 1: (Color online) Schematic view of an infinite planar slab of gain material of thickness L that is aligned in the x-y plane.

substitute $k = k_{\star}$ and $n = n_{\star}$ in (54), and use (55) and (56) to express the resulting equation in the form

$$
\sum_ {j = 1} ^ {\infty} \mathfrak {c} _ {j} \epsilon^ {j} = 0,\tag{58}
$$

where $c_{j}$ are complex coefficients depending on $k_{m}$ , $n_{m}$ , and $z_{2}$ . For example,

$$
\begin{array}{l} \mathfrak {c} _ {1} = F _ {0 1 0}   \mathfrak {n} _ {1} + F _ {0 0 1}   k _ {1} + F _ {1 0 0}   \mathfrak {z} _ {2}, \\ \mathfrak {c} _ {2} = F _ {0 1 0}   \mathfrak {n} _ {2} + F _ {0 0 1}   k _ {2} + F _ {0 2 0}   \mathfrak {n} _ {1} ^ {2} + F _ {0 1 1}   \mathfrak {n} _ {1} k _ {1} + F _ {0 0 2}   k _ {1} ^ {2} \\ \qquad + (F _ {1 1 0}   \mathfrak {n} _ {1} + F _ {1 0 1}   k _ {1}) \mathfrak {z} _ {2} + F _ {2 0 0}   \mathfrak {z} _ {2} ^ {2}. \end{array}\tag{59}
$$

(60)

Finally, we enforce (58) by demanding that $c_{j}=0$ for all $j\geq1$ . Because $c_{j}$ involves $k_{m}$ and $n_{m}$ with $m\leq j$ , in this way we obtain an infinite set of algebraic equations for $k_{m}$ and $n_{m}$ that we can solve iteratively.

For example, in view of (59) and (60), $c_{1}=0$ and $c_{2}=0$ give

$$
\begin{array}{r c l} F _ {0 1 0}   \mathfrak {n} _ {1} + F _ {0 0 1}   k _ {1} & = & - F _ {1, 0, 0}   \mathfrak {z} _ {2}, \\ F _ {0 1 0}   \mathfrak {n} _ {2} + F _ {0 0 1}   k _ {2} & = & - \left(F _ {0 2 0}   \mathfrak {n} _ {1} ^ {2} + F _ {0 1 1}   \mathfrak {n} _ {1} k _ {1} + F _ {0 0 2}   k _ {1} ^ {2}\right) \\ & & - (F _ {1 1 0}   \mathfrak {n} _ {1} - F _ {1 0 1}   k _ {1}) \mathfrak {z} _ {2} - F _ {2 0 0}   \mathfrak {z} _ {2} ^ {2}, \end{array}\tag{61}
$$

(62)

respectively. In general if n and k are independent complex and real variables, we can choose $k_{m} = 0$ and use the above method to compute $n_{m}$ . However, as we see in the next section, there are situations of physical interest where n depends on k and another real variable g. In this case, our method produces a perturbative calculation of k and g.

## V. OSSS OF AN INHOMOGENEOUS PLANAR SLAB GAIN MEDIUM

A simple model supporting OSSs is an infinite planar slab gain medium $[3, 4]$ . Suppose we choose a coordinate system in which the slab is aligned parallel to the x-y plane and has a thickness L, as demonstrated in Figure 1. Then the wave equation associated with this system admits a solution that propagates along the z-axis and corresponds to the following expression for the electric field $[7]$ .

$$
\vec {E} (z, t) = E _ {0} e ^ {- i \omega t} \Psi (z) \hat {e} _ {x},
$$

where $E_{0}$ is a constant coefficient, $\omega$ is the angular frequency of the wave, $\hat{e}_{x}$ is the unit vector pointing along the positive x-axis, $\Psi$ is a solution of the Schrödinger equation,

$$
- \Psi^ {\prime \prime} (z) + V (z) \Psi (z) = \frac {\omega^ {2}}{c ^ {2}} \Psi (z),\tag{63}
$$

$$
V (z) := \left\{ \begin{array}{c l} \frac {\omega^ {2} [ 1 - n (\omega , z) ^ {2} ]}{c ^ {2}} & \text { for } | z | \leq L / 2, \\ 0 & \text { for } | z | > L / 2, \end{array} \right.\tag{64}
$$

c is the speed of light in vacuum, and $n(\omega,z)$ is the complex refractive index of the medium. As shown in Ref. [7], by a simple change of variables z and $\omega$ , namely

$$
z \to x := \frac {z}{L} + \frac {1}{2}, \quad \omega \to k := \frac {L \omega}{c},\tag{65}
$$

we can map (63) and (64) to (1) and (6), respectively. This means that we can use our general results to compute the effects of inhomogeneity of the gain medium (z-dependence of the refractive index) provided that we express the perturbation parameter $\epsilon$ , the coupling constants $z_{1}$ and $z_{2}$ , and the function f entering (44) in terms of the physical parameters of the system. This requires the knowledge of the dispersion relation that determines the dependence of the complex refractive index on $\omega$ and z.

If we consider a gain medium that is obtained by doping a host medium of refraction index $n_{0}$ and modeled as a two-level atomic system with lower and upper level population densities $N_{l}$ and $N_{u}$ , resonance frequency $\omega_{0}$ , and damping coefficient $\gamma$ , then

$$
n ^ {2} (\omega , z) = n _ {0} ^ {2} - \frac {\hat {\omega} _ {p} (z) ^ {2}}{\hat {\omega} ^ {2} - 1 + i \hat {\gamma} \hat {\omega}},\tag{66}
$$

where $\hat{\omega} := \omega / \omega_0$ , $\hat{\gamma} := \gamma / \omega_0$ , $\omega_p^2 := (N_l - N_u)e^2 / (m_e\varepsilon_0)$ , $e$ is electron's charge, and $m_e$ is its mass. Furthermore, we have [4]

$$
\hat {\omega} _ {p} (z) ^ {2} = 2 \hat {\gamma} \kappa (z) \sqrt {n _ {0} ^ {2} + \kappa (z) ^ {2}} \approx 2 \hat {\gamma} n _ {0} \kappa (z) = - \frac {n _ {0} \hat {\gamma} \lambda_ {0} g (z)}{2 \pi}, \quad \kappa (z) := - \frac {\lambda_ {0} g (z)}{4 \pi},\tag{67}
$$

where $\lambda_{0} := 2\pi c/\omega_{0}$ is the resonance wavelength, $g(z)$ is the effective gain coefficient (gain coefficient minus loss coefficient) at the resonance frequency, and we have made use of the fact that for all known gain media, $|\kappa(z)| \ll n_{0}$ . Substituting (67) in (66) gives

$$
n ^ {2} (\omega , z) = n _ {0} ^ {2} + \frac {n _ {0} \hat {\gamma} \lambda_ {0} g (z)}{2 \pi (\hat {\omega} ^ {2} - 1 + i \hat {\gamma} \hat {\omega})},\tag{68}
$$

Next, we assume that the pumping intensity decays exponentially inside the sample. This implies that if we pump it from the left-hand side, we have [7],

$$
g (z) = \left(g _ {0} + \alpha\right) e ^ {- \nu (\frac {z}{L} + \frac {1}{2})} - \alpha \quad \text { for } \quad | z | <   \frac {L}{2},\tag{69}
$$

where $g_{0} := g(-\frac{L}{2})$ , $\alpha$ is the attenuation coefficient at the resonance frequency that coincides with the largest allowed value of $g_{0}$ , and $\nu$ is the decay constant that specifies the exponential decay of the intensity of the pumping beam inside the slab. If we pump the sample from both sides (double pumping), we instead find [7]

$$
g (z) = \left[ \frac {g _ {0} + \alpha}{\cosh (\frac {\nu}{2})} \right] \cosh (\frac {\nu z}{L}) - \alpha \quad \mathrm{for} \quad | z | <   \frac {L}{2}.\tag{70}
$$

Putting all these information together and making the change of variable (65), we can reduce the problem of finding the OSSs of the above system to locating the spectral singularities of the potential (6) with $v_{0}$ and $v_{1}$ given by (44)

and the following choices for $z_{1}$ , $z_{2}$ , $\epsilon$ , and $f(x)$ , [7].

$$
\mathfrak {z} _ {1} := \left(\frac {2 \pi L \hat {\omega}}{\lambda_ {0}}\right) ^ {2} \left[ 1 - n _ {0} ^ {2} + \frac {\hat {\gamma} n _ {0} \lambda_ {0} g _ {0} \mathfrak {z}}{2 \pi} \right], \qquad \mathfrak {z} _ {2} := 2 \pi \hat {\omega} ^ {2} \left(\frac {g _ {0}}{\alpha} + 1\right) \mathfrak {z},\tag{71}
$$

$$
\epsilon := \left\{ \begin{array}{l l} \lambda_ {0} ^ {- 1} L ^ {2} \alpha \hat {\gamma} n _ {0} \nu & \text { for   single   pumping }, \\ \lambda_ {0} ^ {- 1} L ^ {2} \alpha \hat {\gamma} n _ {0} \nu^ {2} & \text { for   double   pumping }, \end{array} \right.\tag{72}
$$

$$
f (x) := \left\{ \begin{array}{c l} \frac {e ^ {- \nu x} - 1}{\nu} & \text {for single pumping,} \\ \frac {\cosh [ \nu (x - \frac {1}{2}) ] - \cosh (\frac {\nu}{2})}{\nu^ {2} \cosh (\frac {\nu}{2})} & \text {for double pumping,} \end{array} \right.\tag{73}
$$

where we have introduced:

$$
\mathfrak {z} := \frac {1}{1 - \hat {\omega} ^ {2} - i \hat {\gamma} \hat {\omega}}.\tag{74}
$$

We also note that according to (47) and (71),

$$
\mathfrak {n} = \sqrt {n _ {0} ^ {2} - \frac {\hat {\gamma} n _ {0} \lambda_ {0} g _ {0} \mathfrak {z}}{2 \pi}} = \sqrt {n _ {0} ^ {2} - \frac {\hat {\gamma} n _ {0} \lambda_ {0} g _ {0} \lambda^ {2}}{2 \pi (\lambda^ {2} - i \hat {\gamma} \lambda_ {0} \lambda - \lambda_ {0} ^ {2})}},\tag{75}
$$

where $\lambda$ stands for the wavelength of the wave, i.e.,

$$
\lambda := \frac {2 \pi c}{\omega} = \frac {2 \pi L}{k} = \frac {\lambda_ {0}}{\hat {\omega}},\tag{76}
$$

Because the physical parameters of practical interest are the wavelength $\lambda$ and the gain coefficient $g_{0}$ , we fix all the other physical quantities and study the effect of inhomogeneity of the medium on the $\lambda$ and $g_{0}$ values associated with spectral singularities. These we respectively denote by $\lambda_{\star}$ and $g_{\star}$ and expand in power series in the perturbation parameter $\epsilon$ ,

$$
\lambda_ {\star} = \lambda_ {(0)} + \sum_ {m = 1} ^ {\infty} \lambda_ {m} \epsilon^ {m}, \quad g _ {\star} = g _ {(0)} + \sum_ {m = 1} ^ {\infty} g _ {m} \epsilon^ {m}.\tag{77}
$$

Here $\lambda_{(0)}$ and $g_{(0)}$ are respectively the values of $\lambda$ and $g_{0}$ associated with the spectral singularity of the unperturbed potential that appears for $n = n_{0}$ and $k = k_{0}$ . As shown in Ref. [4], these are labeled by a mode number m and can be computed by substituting (68) and $\hat{\omega} = \lambda_{0}/\lambda$ in (49) and finding the real values of $\lambda$ and $g_{0}$ that satisfy (49). In view of (55) and (76), it is easy to see that

$$
k _ {0} = \frac {2 \pi L}{\lambda_ {(0)}}, \qquad k _ {1} = - \frac {2 \pi L \lambda_ {1}}{\lambda_ {(0)} ^ {2}} = - \frac {k _ {0} \lambda_ {1}}{\lambda_ {(0)}}.\tag{78}
$$

Next, we view $\mathfrak{n}$ as a function of $\lambda$ and $g_0$ , and identify it with its Taylor series about $(\lambda_{(0)}, g_{(0)})$ . In light of (77), this gives for the value of $\mathfrak{n}$ at $(\lambda_{\star}, g_{\star})$ the second equation in (55), i.e., $\mathfrak{n}_{\star} = \sum_{m=0}^{\infty} \mathfrak{n}_m \epsilon^m$ , and allows for identifying the coefficients $\mathfrak{n}_m$ in terms of those of the power series (77) for $\lambda_{\star}$ and $g_{\star}$ . In particular, $\mathfrak{n}_0$ is the value of $\mathfrak{n}$ at $(\lambda_{(0)}, g_{(0)})$ , i.e.,

$$
\mathfrak {n} _ {0} := \sqrt {1 - \frac {\mathfrak {z} (1)}{k _ {0} ^ {2}}},\tag{79}
$$

$$
\mathfrak {z} _ {(1)} := \mathfrak {z} _ {1} \Big | _ {\lambda = \lambda_ {(0)}, g _ {0} = g _ {(0)}} = - \left(\frac {2 \pi L}{\lambda_ {(0)}}\right) ^ {2} \left[ n _ {0} ^ {2} - 1 - \frac {\hat {\gamma} n _ {0} \lambda_ {0} \lambda_ {(0)} ^ {2} g _ {(0)}}{2 \pi (\lambda_ {(0)} ^ {2} - i \hat {\gamma} \lambda_ {0} \lambda_ {(0)} - \lambda_ {0} ^ {2})} \right],\tag{80}
$$

and $\mathfrak{n}_1$ is given by

$$
\mathfrak {n} _ {1} := \mathfrak {n} _ {1, 0} \lambda_ {1} + \mathfrak {n} _ {0, 1} g _ {1},\tag{81}
$$

$$
\mathfrak {n} _ {1, 0} := \left. \frac {\partial \mathfrak {n}}{\partial \lambda} \right| _ {\lambda = \lambda_ {(0)}, g _ {0} = g _ {(0)}} = \frac {\hat {\gamma} n _ {0} \lambda_ {0} ^ {2} \lambda_ {(0)} (2 \lambda_ {0} + i \hat {\gamma} \lambda_ {(0)}) g _ {(0)}}{4 \pi \mathfrak {n} _ {0} (\lambda_ {(0)} ^ {2} - i \hat {\gamma} \lambda_ {0} \lambda_ {(0)} - \lambda_ {0} ^ {2}) ^ {2}},\tag{82}
$$

$$
\mathfrak {n} _ {0, 1} := \left. \frac {\partial \mathfrak {n}}{\partial g _ {0}} \right| _ {\lambda = \lambda_ {(0)}, g _ {0} = g _ {(0)}} = \frac {- \hat {\gamma} n _ {0} \lambda_ {0} \lambda_ {(0)} ^ {2}}{4 \pi \mathfrak {n} _ {0} (\lambda_ {(0)} ^ {2} - i \hat {\gamma} \lambda_ {0} \lambda_ {(0)} - \lambda_ {0} ^ {2})}.\tag{83}
$$

Now, we recall that the spectral singularities are obtained from the equation

$$
\Gamma_ {1 -} = \sum_ {\ell = 1} ^ {\infty} F _ {\ell}   \mathfrak {z} _ {2} ^ {\ell}   \epsilon^ {\ell} = 0.\tag{84}
$$

In order to use perturbation theory to solve this equation for $\lambda$ and $g_{0}$ , we need to express $F_{\ell}$ and $\mathfrak{z}_{2}$ as functions of $\lambda$ , $g_{0}$ , and $\epsilon$ , substitute their Taylor series expansion about $(\lambda_{(0)}, g_{(0)})$ in $\sum_{\ell=1}^{\infty} F_{\ell}\mathfrak{z}_{2}^{\ell}\epsilon^{\ell}$ , and then set the coefficients of the resulting powers series to zero. As we will see below, for most realistic situations the first-order perturbation theory gives highly reliable results. Therefore, we outline the details of the calculation of the first-order corrections to $\lambda_{(0)}$ and $g_{(0)}$ , namely $\lambda_{1}$ and $g_{1}$ .

Determination of $\lambda_{1}$ and $g_{1}$ requires the following expansions of $F_{0}$ , $F_{1}$ , and $\mathfrak{z}_{2}$ .

$$
F _ {0} = F _ {(0)} + (X \lambda_ {1} + Y g _ {1}) \epsilon + \mathcal {O} (\epsilon^ {2}), \quad F _ {1} = F _ {(1)} + \mathcal {O} (\epsilon^ {1}), \quad \mathfrak {z} _ {2} = \mathfrak {z} _ {(2)} + \mathcal {O} (\epsilon^ {1}),\tag{85}
$$

where $\mathcal{O}(\epsilon^{\ell})$ stands for the terms of order $\epsilon^{\ell}$ and higher, and

$$
F _ {(0)} := F _ {0} \left(\mathfrak {n} _ {0}, k _ {0}\right) = F _ {0 0 0}, \quad X := \mathfrak {n} _ {1 0} F _ {0 1 0} - \frac {2 \pi L F _ {0 0 1}}{\lambda_ {(0)} ^ {2}},\tag{86}
$$

$$
Y := \mathfrak {n} _ {0 1} F _ {0 1 0}, \qquad F _ {(1)} := F _ {1} (\mathfrak {n} _ {0}, k _ {0}) = F _ {1 0 0},\tag{87}
$$

$$
\mathfrak {z} _ {(2)} := \left. \mathfrak {z} _ {2} \right| _ {\lambda = \lambda_ {(0)}, g _ {0} = g _ {(0)}} = \frac {2 \pi \lambda_ {0} ^ {2} (g _ {(0)} + \alpha)}{\alpha (\lambda_ {(0)} ^ {2} - i \hat {\gamma} \lambda_ {0} \lambda_ {(0)} - \lambda_ {0} ^ {2})},\tag{88}
$$

In the derivation of these formulas we have made use (55), (57), (71), (74), (78), and (81).

Next, we observe that because for $(\mathfrak{n}, k) = (\mathfrak{n}_{0}, k_{0})$ we have a spectral singularity, $F_{(0)} = 0$ . In view of this relation, (48), and (57), we can calculate

$$
F _ {0 1 0} = \frac {\mathfrak {z} _ {(1)} + i 2 k _ {0}}{\mathfrak {n} _ {0}} = \frac {1}{\mathfrak {n} _ {0}} \left[ \mathfrak {z} _ {(1)} + \frac {4 \pi i L}{\lambda_ {(0)}} \right], \qquad F _ {0 0 1} = - k _ {0} (\mathfrak {n} _ {0} ^ {2} - 1) = \frac {\mathfrak {z} _ {(1)}}{k _ {0}} = \frac {\lambda_ {(0)} \mathfrak {z} _ {(1)}}{2 \pi L}.\tag{89}
$$

These together with (86) and (88) give

$$
X = \frac {\mathfrak {n} _ {1 0}}{\mathfrak {n} _ {0}} \left[ \mathfrak {z} _ {(1)} + \frac {4 \pi i L}{\lambda_ {(0)}} \right] - \frac {\mathfrak {z} _ {(1)}}{\lambda_ {(0)}}, \qquad Y = \frac {\mathfrak {n} _ {0 1}}{\mathfrak {n} _ {0}} \left[ \mathfrak {z} _ {(1)} + \frac {4 \pi i L}{\lambda_ {(0)}} \right].\tag{90}
$$

Furthermore, in light of $F_{(0)} = 0$ and (84) - (88), we can write the equation determining $\lambda_{1}$ and $g_{1}$ as

$$
X \lambda_ {1} + Y g _ {1} = - F _ {1 0 0} \mathfrak {z} _ {(2)}.\tag{91}
$$

This is a complex linear equation involving two real unknowns. Therefore, we can easily solve it to obtain:

$$
\lambda_ {1} = - \frac {\operatorname{Im} (F _ {1 0 0 \mathfrak {z} (2)} Y ^ {*})}{\operatorname{Im} (X Y ^ {*})}, \qquad g _ {1} = \frac {\operatorname{Im} (F _ {1 0 0 \mathfrak {z} (2)} X ^ {*})}{\operatorname{Im} (X Y ^ {*})},\tag{92}
$$

where “Im(·)” stands for the imaginary part of its argument.

Next, we examine $F_{100}$ that encodes all the information about the inhomogeneity of the gain medium. According to (51),

$$
F _ {1 0 0} = F _ {1} (\mathfrak {n} _ {0}, k _ {0}) = \frac {1}{\mathfrak {n} _ {0} k _ {0}} \int_ {0} ^ {1} d x \xi (\mathfrak {n} _ {0}, k _ {0}, x) f (x),\tag{93}
$$

where $\xi$ is given by (52). With the help of (49) we have been able to simplify the expression for $\xi(\mathfrak{n}_{0}, k_{0}, x)$ and find

$$
\xi (\mathfrak {n} _ {0}, k _ {0}, x) = \left(1 - \frac {1}{\mathfrak {n} _ {0} ^ {2}}\right) \cos^ {2} \left[ \mathfrak {n} _ {0} k _ {0} (x - \frac {1}{2}) \right].\tag{94}
$$

Substituting this equation and (73) in (93), evaluating the resulting integrals, and using (49) to simplify the outcome we obtain

$$
F _ {1 0 0} = \frac {(1 - \frac {1}{\mathfrak {n} _ {0} ^ {2}}) \Big [ (1 - e ^ {- \nu}) [ 4 i + k _ {0} (1 + \mathfrak {n} _ {0} ^ {2}) ] - k _ {0} \nu + 2 i \nu^ {3} - 4 k _ {0} ^ {3}   \mathfrak {n} _ {0} ^ {3} (\mathfrak {n} _ {0} ^ {2} - 1) (\nu + e ^ {- \nu} - 1) \Big ]}{2 k _ {0} ^ {3}   \mathfrak {n} _ {0} ^ {4}   \nu^ {2} (4 k _ {0} ^ {2}   \mathfrak {n} _ {0} ^ {2} + \nu^ {2})},\tag{95}
$$

![](images/fe0f7dca7f1d759e17949a172c79d486e8a46a333e1f516d3ee6b7bfa7e89f06.jpg)

![](images/a85e070c7dac43b80f4ad15177b8b6461bcc60679273494ce88f62994978012b.jpg)
FIG. 2: (Color online) Location of the optical spectral singularities considered in Table I for $0 \leq \nu \leq 0.5$ . The left- and right-hand figures respectively show the behavior of OSS for the singly and doubly pumped samples. The displayed dots that mark the boundaries of each of the curves of the OSSs correspond to $\nu = 0$ and $\nu = 0.5$ . As we increase $\nu$ the location of OSSs in the $\lambda-g_{0}$ plane moves upwards on the given curves.

for a singly-pumped sample, and

$$
F _ {1 0 0} = \frac {(\mathfrak {n} _ {0} - 1) (\mathfrak {n} _ {0} + 1) ^ {2} [ - 2 + \mathfrak {n} _ {0} ^ {2} (\nu - 2) - \mathfrak {n} _ {0} \nu ]}{2 k _ {0} ^ {2} \mathfrak {n} _ {0} ^ {7} \nu (4 k _ {0} ^ {2} \mathfrak {n} _ {0} ^ {2} + \nu^ {2})},\tag{96}
$$

for a doubly-pumped sample.

Now, we can use (72), (80), (90), (92), (95), and (96) to calculate the first-order corrections to the wavelength and gain coefficient of the OSSs, i.e., $\lambda_1\epsilon$ and $g_{1}\epsilon$ .

As a concrete example, consider a semiconductor gain medium with the following specifications that is also studied in Refs. [4, 7].

$$
n _ {0} = 3. 4, \quad L = 3 0 0 \mu \mathrm{m}, \quad \lambda_ {0} = 1 5 0 0 \mathrm{nm}, \quad \hat {\gamma} = 0. 0 2, \quad \alpha = 2 0 0 \mathrm{cm} ^ {- 1}, \quad 0 \leq \nu \leq 0. 5.\tag{97}
$$

For this sample $\epsilon\approx0.8\nu\leq0.4$ for single pumping and $\epsilon\approx0.8\nu^{2}\leq0.2$ for double pumping. In particular for physically realistic situations where $\nu\precsim0.1$ , we have $\epsilon^{2}\precsim6.4\times10^{-3}$ and $\epsilon^{2}\precsim4.0\times10^{-4}$ for single and double pumping, respectively. This shows that the first-order perturbation theory produces a highly reliable description of the OSSs for this system.

In the remainder of this section, we report the results of the first-order perturbative calculation of $g_{\star}$ and $\lambda_{\star}$ for the first five OSSs that appear as we increase the intensity of the pumping beam starting from zero. These have a wavelength that is closest to the resonance wavelength of the sample, $\lambda_{0} = 1500$ nm. We label them using the mode number m that is introduced in [7] and takes values between 1358 and 1362, with 1360 corresponding to the OSS with closest wavelength to $\lambda_{0}$ . Table I gives numerical values of $\lambda_{\star}$ and $g_{\star}$ for these OSSs and five different values of the decay constant $\nu$ . As we increase $\nu$ , $\lambda_{\star}$ remains essentially unchanged while $g_{\star}$ increases. This is particularly pronounced for a singly-pumped sample. It confirms the semiclassical results obtained in [7].

Figure 2 shows the curves traced by $(\lambda_{\star}, g_{\star})$ as we change $\nu$ for each of the OSSs considered in Table I. The fact that these curved are essentially vertical line segments shows that $\lambda_{\star}$ does not depend on $\nu$ , while the opposite is true for $g_{\star}$ , especially for the singly-pumped sample. The displayed dots that mark the lower and upper boundaries of each line segment respectively correspond to $\nu = 0$ and $\nu = 0.5$ . As we increase $\nu$ , the location of each OSS in the $\lambda-g_{0}$ plane moves upwards along the corresponding line segment.

<table><tr><td colspan="2"></td><td colspan="2">Single Pumping</td><td colspan="2">Double Pumping</td></tr><tr><td>m</td><td> $\nu$ </td><td> $\lambda$ (nm)</td><td> $g_0$ (cm $^{-1}$ )</td><td> $\lambda$ (nm)</td><td> $g_0$ (cm $^{-1}$ )</td></tr><tr><td rowspan="5">1362</td><td>0.0</td><td>1497.561770810</td><td>41.53101</td><td>1497.561770810</td><td>41.53101</td></tr><tr><td>0.1</td><td>1497.561770785</td><td>43.45261</td><td>1497.561770784</td><td>41.56407</td></tr><tr><td>0.2</td><td>1497.561770716</td><td>45.25128</td><td>1497.561770707</td><td>41.66286</td></tr><tr><td>0.3</td><td>1497.561770609</td><td>46.93581</td><td>1497.561770579</td><td>41.82620</td></tr><tr><td>0.5</td><td>1497.561770304</td><td>49.99447</td><td>1497.561770180</td><td>42.33818</td></tr><tr><td rowspan="5">1361</td><td>0.0</td><td>1498.389018373</td><td>40.91032</td><td>1498.389018373</td><td>40.91032</td></tr><tr><td>0.1</td><td>1498.389018341</td><td>42.83283</td><td>1498.389018339</td><td>40.94324</td></tr><tr><td>0.2</td><td>1498.389018251</td><td>44.63206</td><td>1498.389018239</td><td>41.04159</td></tr><tr><td>0.3</td><td>1498.389018115</td><td>46.31686</td><td>1498.389018073</td><td>41.20423</td></tr><tr><td>0.5</td><td>1498.389017715</td><td>49.37529</td><td>1498.389017554</td><td>41.71399</td></tr><tr><td rowspan="5">1360</td><td>0.0</td><td>1499.999983312</td><td>40.40905</td><td>1499.999983312</td><td>40.40905</td></tr><tr><td>0.1</td><td>1499.999983275</td><td>42.33379</td><td>1499.999983220</td><td>40.44217</td></tr><tr><td>0.2</td><td>1499.999983205</td><td>44.13541</td><td>1499.999983115</td><td>40.54115</td></tr><tr><td>0.3</td><td>1499.999983098</td><td>45.82273</td><td>1499.999983003</td><td>40.70480</td></tr><tr><td>0.5</td><td>1499.999982791</td><td>48.88649</td><td>1499.999982512</td><td>41.21777</td></tr><tr><td rowspan="5">1359</td><td>0.0</td><td>1501.475689102</td><td>40.79650</td><td>1501.475689102</td><td>40.79650</td></tr><tr><td>0.1</td><td>1501.475689077</td><td>42.72315</td><td>1501.475689075</td><td>40.82968</td></tr><tr><td>0.2</td><td>1501.475689007</td><td>44.52660</td><td>1501.475688997</td><td>40.92881</td></tr><tr><td>0.3</td><td>1501.475688899</td><td>46.21566</td><td>1501.475688689</td><td>41.09272</td></tr><tr><td>0.5</td><td>1501.475688590</td><td>49.28266</td><td>1501.475688464</td><td>41.60649</td></tr><tr><td rowspan="5">1358</td><td>0.0</td><td>1502.670951310</td><td>41.63220</td><td>1502.670951310</td><td>41.63220</td></tr><tr><td>0.1</td><td>1502.670951286</td><td>43.56043</td><td>1502.670951282</td><td>41.65321</td></tr><tr><td>0.2</td><td>1502.670951220</td><td>45.36542</td><td>1502.670951211</td><td>41.76466</td></tr><tr><td>0.3</td><td>1502.670951118</td><td>47.05600</td><td>1502.670951089</td><td>41.92890</td></tr><tr><td>0.5</td><td>1502.670950826</td><td>50.12593</td><td>1502.670950707</td><td>42.44373</td></tr></table>

TABLE I: The wavelength $\lambda_{\star}$ and gain coefficient $g_{\star}$ for the five OSSs that are generated by pumping the semiconductor gain medium (97). m is the mode number labeling these OSSs [7]. $\nu$ is the decay constant for the intensity of the pumping beam(s) inside the sample.

## VI. CONCLUDING REMARKS

In this article we have outlined a general method for carrying out a perturbative calculation of the transfer matrix for a general complex potential that vanishes outside a closed interval. This allows for a systematic characterization of the spectral singularities of this class of potentials. This turns out to be particularly suitable for the study of optical spectral singularities and their time-reversed analogs that respectively correspond to lasing at threshold gain and antilasing.

As an application of our general results, we examined the problem of computing the transfer matrix for a potential consisting of finitely many Dirac delta-functions that are centered at arbitrary points and have arbitrary complex coupling constants. We showed that perturbation theory gives an exact expression for the transfer matrix for this system.

Next, we considered an arbitrary complex perturbation of a constant (complex) barrier potential and applied our method to compute the effect of the perturbation on the spectral singularities. A physical realization of this model is in the description of threshold lasing associated with an infinite planar slab of gain material. For this system the gain coefficient that is proportional to the intensity of the pumping beam decays exponentially as the beam penetrates the medium. This in turn makes the gain medium inhomogeneous. Our method allows for an essentially analytic calculation of the effect of this inhomogeneity on the location of optical spectral singularities. Our results confirm those obtained using the method of Ref. [7] that is based on the semiclassical approximation. Compared with this method ours has the advantage of being applicable in every spectral range. In particular, we can use it in the spectral ranges comparable with the length scale of the system. This is, for example, the case in the recent study of unidirectional invisibility [18]. Our method allows for a thorough analysis of this and much more general optical systems displaying threshold lasing, antilasing, and unidirectional invisibility.

Acknowledgments: This work has been supported by the Scientific and Technological Research Council of Turkey (TÜBİTAK) in the framework of the project no: 110T611 and the Turkish Academy of Sciences (TÜBA). We wish to thank Aref Mostafazadeh for his help in preparing Figure 1 and Ali Serpengüzel for his careful reading of the first draft of this paper and making many invaluable remarks.

[1] M. A. Naimark, Trudy Moscow. Mat. Obsc. 3, 181 (1954) in Russian, English translation: Amer. Math. Soc. Transl. (2), 16, 103 (1960); R. R. D. Kemp, Canadian J. Math. 10, 447 (1958); J. Schwartz, Comm. Pure Appl. Math. 13, 609 (1960); G. Sh. Guseinov, Pramana. J. Phys. 73, 587 (2009).

[2] A. Mostafazadeh, Phys. Rev. Lett. 102, 220402 (2009).

[3] A. Mostafazadeh, Phys. Rev. A 80, 032711 (2009).

[4] A. Mostafazadeh, Phys. Rev. A 83, 045801 (2011).

[5] Z. Ahmed, J. Phys. A 42, 472005 (2009); S. Longhi, Phys. Rev. B 80, 165125 (2009) and Phys. Rev. A 81, 022102 (2010); B. F. Samsonov, J. Phys. A 44, 392001 (2011).

[6] S. Longhi, Physics 3, 61 (2010); Phys. Rev. A 82, 031801 (2010); and Phys. Rev. A 83, 055804 (2011).

[7] A. Mostafazadeh, Phys. Rev. A 84, 023809 (2011).

[8] A. Mostafazadeh and M. Sarisaman, Phys. Lett. A 375, 3387 (2011), and Proc. R. Soc. A, to appear.

[9] Y. D. Chong, L. Ge, H. Cao, and A. D. Stone, Phys. Rev. Lett. 105, 053901 (2010); W. Wan, Y. Chong, L. Ge, H. Noh, A. D. Stone, and H. Cao, Science 331, 889 (2011); Y. D. Chong, L. Ge, and A. D. Stone, Phys. Rev. Lett. 106, 093902 (2011); S. Longhi, Phys. Rev. A 82, 031801 (2010), Phys. Rev. A 83, 055804 (2011), and Phys. Rev. Lett. 107, 033901 (2011); L. Ge, Y. D. Chong,, S. Rotter, H. E. Türeci, and A. D. Stone, Phys. Rev. A 84, 023820 (2011).

[10] This is because the imaginary part of this potential is proportional to the imaginary part of the complex refractive index of the medium which is even for the high gain/loss material is must smaller than its real part. For typical numerical values see for example [3, 4, 7, 8].

[11] W. E. Boyce, R. C. DiPrima, Elementary Differential Equations and Boundary Value Problems, John Wiley and Sons, New Jersey, (2005).

[12] For sufficiently large $\epsilon$ , the presence of $\epsilon v_{1}$ may also lead to the creation of new spectral singularities.

[13] D. Kiang, Am. J. Phys. 42, 785 (1974).

[14] D. J. Griffiths and C. A. Steinke, Am. J. Phys. 69, 137 (2001).

[15] A. Mostafazadeh, J. Phys. 39, 13495 (2006).

[16] A. Mostafazadeh and H. Mehri-Dehnavi, J. Phys. A 42, 125303 (2009).

[17] A. Mostafazadeh, Pramana, J. Phys. 73, 269 (2009).

[18] Z. Lin, H. Ramezani, T. Eichelkraut, T. Kottos, H. Cao, and D. N. Christodoulides, Phys. Rev. Lett. 106, 213901 (2011).
