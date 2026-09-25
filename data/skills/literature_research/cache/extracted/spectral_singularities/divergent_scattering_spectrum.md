# Spectral Singularities of Complex Scattering Potentials and Infinite Reflection and Transmission Coefficients at real Energies

Ali Mostafazadeh

Department of Mathematics, Koç University,

Sarıyer 34450, Istanbul, Turkey

amostafazadeh@ku.edu.tr

Spectral singularities are spectral points that spoil the completeness of the eigenfunctions of certain non-Hermitian Hamiltonian operators. We identify spectral singularities of complex scattering potentials with the real energies at which the reflection and transmission coefficients tend to infinity, i.e., they correspond to resonances having a zero width. We show that a wave guide modeled using such a potential operates like a resonator at the frequencies of spectral singularities. As a concrete example, we explore the spectral singularities of an imaginary $\mathcal{PT}$ -symmetric barrier potential and demonstrate the above resonance phenomenon for a certain electromagnetic waveguide.

Pacs numbers: 03.65.-w, 03.65.Nk, 11.30.Er, 42.25.Bs, 42.79.Gn

## I. INTRODUCTION

Complex PT-symmetric potentials [1] that have a real spectrum are interesting, because one can restore the Hermiticity of the corresponding Hamiltonian and uphold unitarity by modifying the inner product of the Hilbert space [2-4]. It is usually believed that one can similarly treat every non-Hermitian Hamiltonian $H$ that has a real and discrete spectrum. This is actually true provided that $H$ has a complete set of eigenvectors [2]. For the cases that the spectrum is discrete the lack of completeness is associated with the presence of exceptional points. These correspond to situations where two or more eigenvalues together with their eigenvectors coalesce. This phenomenon is known to have physically observable consequences [5]. It also plays an important role in the study of open quantum systems particularly in relation with the resonance states [6, 7]. For the cases that the spectrum has a continuous part, there is another mathematical obstruction for the completeness of the eigenvectors called a “spectral singularity” [22]. The purpose of the present article is to describe the physical meaning and a possible practical application of spectral singularities.

Spectral singularities were discovered by Naimark [8] and subsequently studied by mathematicians in the 1950's and 1960's [9]. The mechanism by which spectral singularities spoil the completeness of the eigenfunctions and their difference with exceptional points are discussed in [10].

Spectral singularities of complex PT-symmetric and non-PT-symmetric scattering potentials have been studied in [10, 11]. In this article we shall examine the spectral singularities of the imaginary potential [12, 13]:

$$
v _ {\mathfrak {a}, \mathfrak {z}} (\mathrm{x}) = \left\{ \begin{array}{c c} i _ {\mathfrak {z}} & \text { for } - \mathfrak {a} <   \mathrm{x} <   0, \\ - i _ {\mathfrak {z}} & \text { for } 0 <   \mathrm{x} <   \mathfrak {a}, \\ 0 & \text { otherwise }, \end{array} \right.\tag{1}
$$

with $a \in R^{+}$ and $z \in R - \{0\}$ , that has applications in modeling certain electromagnetic waveguides [12].

## II. SPECTRAL SINGULARITIES

Consider a complex scattering potential $v(x)$ that decays rapidly as $|x| \to \infty$ [23]. Suppose that the continuous spectrum of the Hamiltonian $H = -\frac{d^{2}}{dx^{2}} + v(x)$ is $[0, \infty)$ , and for each $k \in R^{+}$ let $\psi_{k\pm}(x)$ denote the solutions of the eigenvalue equation $H\psi(x) = k^{2}\psi(x)$ satisfying the asymptotic boundary conditions:

$$
\psi_ {k \pm} (x) \rightarrow e ^ {\pm i k x} \mathrm{as} x \rightarrow \pm \infty ,\tag{2}
$$

i.e., the Jost solutions. A spectral singularity of $H$ (or $v$ ) is a point $k_{\star}^{2}$ of the continuous spectrum of $H$ such that the $\psi_{k_{\star}\pm}$ are linearly-dependent, i.e., they have a vanishing Wronskian, $\psi_{k_{\star}} + \psi_{k_{\star -}}^{\prime} - \psi_{k_{\star}} - \psi_{k_{\star +}}^{\prime} = 0$ , [10].

Clearly the continuous spectrum of $H$ is doubly-degenerate. To make this explicit, we use $\psi_k^\mathfrak{g}$ with $k\in \mathbb{R}^+$ and $\mathfrak{g}\in \{1,2\}$ to denote a general solution of the eigenvalue equation $H\psi (x) = k^{2}\psi (x)$ . Furthermore, because $v(x)\to 0$ as $x\to \pm \infty$ , we have

$$
\psi_ {k} ^ {\mathfrak {g}} \rightarrow A _ {\pm} ^ {\mathfrak {g}} e ^ {i k x} + B _ {\pm} ^ {\mathfrak {g}} e ^ {- i k x} \mathrm{as} x \rightarrow \pm \infty ,\tag{3}
$$

where $A_{\pm}^{g}$ and $B_{\pm}^{g}$ are complex coefficients. A quantity of interest is the transfer matrix $\mathbf{M}(k)$ that is defined by $\left(\begin{array}{c}A_{+}^{\mathfrak{g}}\\B_{+}^{\mathfrak{g}}\end{array}\right)=\mathbf{M}(k)\left(\begin{array}{c}A_{-}^{\mathfrak{g}}\\B_{-}^{\mathfrak{g}}\end{array}\right)$ . Among its useful properties are the identity $\det\mathbf{M}(k)=1$ and the following theorem.

Theorem 1: $k_{\star}^{2} \in R^{+}$ is a spectral singularity of H if and only if either $-k_{\star}$ or $k_{\star}$ is a real zero of the entry $M_{22}(k)$ of $\mathbf{M}(k)$ , [10].

Next, consider the left- and right-going scattering solutions of $H\psi(x)=k^{2}\psi(x)$ that we denote by $\psi_{k}^{l}$ and $\psi_{k}^{r}$ , respectively. They satisfy [15]

$$
\psi_ {k} ^ {l} (x) \to \left\{ \begin{array}{c l} N _ {l} \left(e ^ {i k x} + R ^ {l} e ^ {- i k x}\right) & \text {as} x \to - \infty , \\ N _ {l} T ^ {l} e ^ {i k x} & \text {as} x \to + \infty , \end{array} \right.\tag{4}
$$

$$
\psi_ {k} ^ {r} (x) \to \left\{ \begin{array}{c l} N _ {r} T ^ {r} e ^ {- i k x} & \text {as} x \to - \infty , \\ N _ {r} \left(e ^ {- i k x} + R ^ {r} e ^ {i k x}\right) & \text {as} x \to + \infty , \end{array} \right.\tag{5}
$$

where $N_{l}, N_{r}, R^{l}, R^{r}, T^{l}$ and $T^{r}$ are complex coefficients. $N_{l}, N_{r}$ are normalization constants, $|R^{l}|^{2}, |R^{r}|^{2}$ are the left and right reflection coefficients, and $|T^{l}|^{2}, |T^{r}|^{2}$ are the left and right transmission coefficients, respectively. Comparing (4) and (5) with (2), we see that $\psi_{k}^{l}$ and $\psi_{k}^{r}$ are respectively proportional to the Jost solutions $\psi_{k+}$ and $\psi_{k-}$ . Therefore, at a spectral singularity, $k_{\star}^{2}$ , the scattering solutions $\psi_{k}^{l}$ and $\psi_{k}^{r}$ become linearly-dependent. In view of (4) and (5), this is possible only if $R^{l}, R^{r}, T^{l}$ and $T^{r}$ tend to infinity as $k \to k_{\star}$ . The converse of this statement is also true:

Theorem 2: $k_{\star}^{2} \in R^{+}$ is a spectral singularity of H if and only if the left and right reflection and transmission coefficients tend to infinity as $k \to k_{\star}$ or $k \to -k_{\star}$ .

The following is an explicit proof of this theorem.

Comparing (4) and (5) with (3), we can determine the coefficients $A_{\pm}^{g}$ and $B_{\pm}^{g}$ for $\psi_{k}^{l}$ and $\psi_{k}^{r}$ and use them to express $R^{l}, R^{r}, T^{l}$ and $T^{r}$ in terms of the entries of the transfer matrix $\mathbf{M}(k)$ . This yields

$$
T ^ {l} = 1 / M _ {2 2} (k), \quad R ^ {l} = - M _ {2 1} (k) / M _ {2 2} (k),\tag{6}
$$

$$
T ^ {r} = 1 / M _ {2 2} (k), \quad R ^ {r} = M _ {1 2} (k) / M _ {2 2} (k),\tag{7}
$$

where we have employed $\det\mathbf{M}(k)=1$ . As seen from (6) and (7), at a spectral singularity, where $M_{22}(k)$ vanishes, $R^{l}, R^{r}, T^{l}$ and $T^{r}$ diverge. The converse holds because $M_{12}$ and $M_{21}$ are entire functions (lacking singularities).

Another curious consequences of (6) and (7), is the identity: $T^l = T^r$ . This is derived in [14] using a different approach, but is usually overlooked. See for example [15].

Next, we examine the $S$ -matrix of the system: $\mathbf{S} = \left( \begin{array}{cc}T^l & R^r\\ R^l & T^r \end{array} \right)$ , [15]. In view of (6), (7), and $\det \mathbf{M}(k) = 1$ , the eigenvalues of $\mathbf{S}$ are given by $s_{\pm} = (1\pm \sqrt{1 - M_{11}(k)M_{22}(k)}) / M_{22}(k)$ . At a spectral singularity $s_{+}$ diverges while $s_{-}\to M_{11}(k) / 2$ . This suggests identifying spectral singularities with certain type of resonances. Indeed, in view of Theorem 2 and Siegert's definition of resonance states [16], they correspond to resonances with a vanishing width (real energy).

## III. PT-SYMMETRIC BARRIER POTENTIAL

Consider the Hamiltonian operator $H = -\frac{d^{2}}{dx^{2}} + v_{\mathfrak{a},\mathfrak{z}}(x)$ with $v_{\mathfrak{a},\mathfrak{z}}(x)$ given by (1). Because $v_{\mathfrak{a},\mathfrak{z}}(x) = 0$ for $|x| > \mathfrak{a}$ , the results of Section II apply to $v_{\mathfrak{a},\mathfrak{z}}$ . The determination of the eigenfunctions [13, 17] of H and the corresponding transfer matrix $\mathbf{M}(k)$ is a straightforward calculation. Here we report the result of the calculation of $M_{22}(k)$ :

$$
M _ {2 2} (k) = e ^ {2 i \mathfrak {a} k} [ f _ {1} (k) - i f _ {2} (k) ] / \sqrt {1 + y ^ {2}},\tag{8}
$$

where $f_{1}$ and $f_{2}$ are real-valued functions given by

$$
f _ {1} (k) = \sqrt {1 + y ^ {2}} | \cos (\mathfrak {a} k w) | ^ {2} - | \sin (\mathfrak {a} k w) | ^ {2},\tag{9}
$$

$$
f _ {2} (k) = \Re \left[ \sqrt {1 + i y} (2 - i y) \sin (\mathfrak {a} k w) \cos (\mathfrak {a} k w ^ {*}) \right],\tag{10}
$$

$y := \mathfrak{z}/k^2$ , $w := \sqrt{1 - iy}$ , and $\Re$ means “real part of”.

According to Theorem 1 and Eq. (8), $k^2 \in \mathbb{R}^+$ is a spectral singularity of $v_{a,\mathfrak{z}}$ if and only if $f_1(k) = 0$ and $f_2(k) = 0$ . If we insert (9) and (10) in these equations and divide their both sides by $|\cos(\mathfrak{ak}w)|^2$ , we find

$$
| \tan (\mathfrak {a} k w) | ^ {2} = \sqrt {1 + y ^ {2}},\tag{11}
$$

$$
\tan (\mathfrak {a} k w) = - \left[ \frac {\sqrt {1 - i y} (2 + i y)}{\sqrt {1 + i y} (2 - i y)} \right] \tan (\mathfrak {a} k w) ^ {*}.\tag{12}
$$

Now, we multiply both sides of (12) by $\tan(\mathfrak{a}kw)$ , use (11), $\cos(2\theta)=(1-\tan^{2}\theta)/(1+\tan^{2}\theta)$ and $w=\sqrt{1-iy}$ , to obtain $\cos(2\mathfrak{a}k\sqrt{1-iy})=-(1+4y^{-2})+2iy^{-1}$ . This equation is equivalent to

$$
\cos r \cosh q = - (1 + 4 y ^ {- 2}),\tag{13}
$$

$$
\sin r \sinh q = 2 y ^ {- 1},\tag{14}
$$

where

$$
q := \mathfrak {a} k \sqrt {2 (\sqrt {y ^ {2} + 1} - 1)} \operatorname{sgn} (y),\tag{15}
$$

$$
r := \mathfrak {a} k \sqrt {2 (\sqrt {y ^ {2} + 1} + 1)},\tag{16}
$$

$\operatorname{sgn}(y)$ denotes the sign of $y$ , and we have employed the identities $\sin \left(\frac{\tan^{-1}y}{2}\right) = \operatorname{sgn}(y)\sqrt{\frac{1}{2}\left[1 - (y^2 + 1)^{-1/2}\right]}$ and $\cos \left(\frac{\tan^{-1}y}{2}\right) = \sqrt{\frac{1}{2}\left[1 + (y^2 + 1)^{-1/2}\right]}$ .

Next, we solve for $y^{-1}$ in (14), substitute the resulting expression in (13), and use the identities $\sinh^{2}q = \cosh^{2}q - 1$ and $\cos^{2}r = 1 - \sin^{2}r$ to obtain a quadratic equation for $\cosh q$ with solutions

$$
\cosh q = \frac {1}{2} (- 1 \pm \sqrt {2 \cos (2 r) - 1}) \cot r \csc r.\tag{17}
$$

To ensure that the right-hand side of this equation is real, we must have $\cos(2r) \geq \frac{1}{2}$ . Furthermore according to (13), $\cos(r) < 0$ . These imply

$$
| r - (2 n + 1) \pi | \leq \frac {\pi}{6}, \quad \text { for   some   integer } n.\tag{18}
$$

Under this condition the right-hand side of (17) is greater than 1. Hence, $q = \pm q_{\pm}(r)$ , where $q_{\pm}(r) := \cosh^{-1}\left[\cot r \csc r(\pm \sqrt{2\cos(2r) - 1} - 1)/2\right]$ . If we set $q = \pm q_{\pm}(r)$ in (14) and solve for $y$ , we find $y = \pm \operatorname{sgn}(\sin r)y_{\pm}(r)$ , where $y_{\pm} := 2|\sin r \sinh q_{\pm}(r)|^{-1}$ . Inserting this expression for $y$ in (15) and (16) and solving for $q$ give $q = \pm \tilde{q}_{\pm}(r)$ , where $\tilde{q}_{\pm}(r) := r\operatorname{sgn}(\sin r)\sqrt{\frac{\sqrt{y_{\pm}(r)^2 + 1} - 1}{\sqrt{y_{\pm}(r)^2 + 1} + 1}}$ . The spectral singularities correspond to the values of $r$ for which $q_{\pm}(r) = \tilde{q}_{\pm}(r)$ . These are transcendental equations admitting simple numerical treatments. It turns out that $q_{+}(r) = \tilde{q}_{+}(r)$ does not have a real solution fulfilling (18), while $q_{-}(r) = \tilde{q}_{-}(r)$ has two solutions $\pm r_n$ for each choice of $n$ in (18). Table 1 lists the numerical values of $r_n$ for various choices of $n$ . It turns out that $r_n > 0$ and $r_{-n} = -r_{n+1}$ for $n > 0$ .

<table><tr><td>n</td><td>rn</td><td>yn</td><td>akn</td><td> $\mathfrak{a}^2_{\mathfrak{z}n}$ </td></tr><tr><td>0</td><td>2.64390700</td><td>1.82765566</td><td>1.06468255</td><td>2.07173713</td></tr><tr><td>1</td><td>9.11655393</td><td>0.71364271</td><td>4.31823693</td><td>13.3074170</td></tr><tr><td>2</td><td>15.4804556</td><td>0.49008727</td><td>7.52928304</td><td>27.7830976</td></tr><tr><td>10</td><td>65.8884385</td><td>0.17167639</td><td>32.8243878</td><td>184.971084</td></tr><tr><td>100</td><td>631.445619</td><td>0.02901727</td><td>315.689592</td><td>2891.85852</td></tr></table>

TABLE I: $r_{n}$ , $y_{n}$ , $k_{n}$ , and $z_{n}$ are respectively the numerical values of $r, y_{-}$ , k and z that correspond to spectral singularities. For $n \geq 0$ , $k_{n}$ and $z_{n}$ are increasing functions of n.

Next, we insert $y = \pm \text{sgn}(\sin r)y_{-}(r)$ in (16) and use the identity $\mathfrak{a}^{2}\mathfrak{z} = (\mathfrak{a}k)^{2}y$ to obtain $k = g(r)$ and $\mathfrak{a}^{2}\mathfrak{z} = \pm g(r)^{2}\text{sgn}(\sin r)y_{-}(r)$ , where $g(r) := r/\sqrt{2(\sqrt{y_{-}(r)^{2}+1}+1)}$ . Setting $r = \pm r_{n}$ in these relations gives the values $(\mathfrak{a}k_{n}$ and $\mathfrak{a}^{2}\mathfrak{z}_{n})$ of ak and $a^{2}z$ that are associated with spectral singularities. We list some of these values in Table I. Because ak and $a^{2}z$ are odd functions of r and $r_{-n} = -r_{n+1}$ for n > 0, we have $k_{-n} = -k_{n+1}$ and $z_{-n} = -z_{n+1}$ . According to Table I, the smallest values of $|a|k|$ and $a^{2}|z|$ for which a spectral singularity occurs are respectively $ak_{0} \approx 1.06$ and $a^{2}z_{0} \approx 2.07$ . Using more accurate values for $ak_{n}$ and $a^{2}z_{n}$ that we do not report here, we have checked that $|M_{22}(k_{n})| < 10^{-9}$ for $|n| \leq 20$ and $|n| = 10^{2}, 10^{3}, 10^{4}$ .

For the system we considered in this section, each value of $a^{2}z$ can support at most one spectral singularity (either the latter does not exist or it exists for a single energy value).

## IV. A PT-SYMMETRIC WAVEGUIDE

Consider a rectangular waveguide with perfectly conducting walls that is aligned along the z-axis and has height $2\beta$ as depicted in Figure 1. Suppose that the region $|z| < \alpha$ inside the waveguide is filled with an atomic gas, and a laser beam shining along the y-direction in the region $-\alpha < z < 0$ is used to excite the resonant atoms and produce a population inversion. In this way $-\alpha < z < 0$ and $0 < z < \alpha$ serve as gain and loss regions respectively, and the relative permittivity at the resonance frequency takes the form $\varepsilon(z) = 1 + \frac{i\omega_{p}^{2}\operatorname{sgn}(z)}{2\delta\omega}$ for $|z| < \alpha$ and $\varepsilon(z) = 1$ for $|z| \geq \alpha$ , where $\omega, \omega_{p}$ and $\delta$ are respectively the frequency of the wave, plasma frequency, and the damping constant, [12]. Alternatively, $\varepsilon(z) = 1 - v_{\alpha,\mathfrak{s}/\omega}(z)$ where $\mathfrak{s} := \omega_{p}^{2}/(2\delta)$ . In [12], the authors use an approximation scheme to reduce Maxwell's equations for this system to the Schrödinger equation for the barrier potential (1). Here we offer an exact treatment to examine singularities of the reflection and transmission coefficients for this waveguide.

Let $\hat{i},\hat{j},\hat{k}$ be the unit vectors along the $x-$ , $y-$ and $z$ -axes, $\mathfrak{K} := \omega/c$ , $m \in \mathbb{Z}^{+}$ , $\mathfrak{K}_{m} := \pi m/(2\beta)$ , $\chi_{m}(x)) := \sin[\mathfrak{K}_{m}(x+\beta)]$ , and $\kappa := \sqrt{\mathfrak{K}^{2}-\mathfrak{K}_{m}^{2}}$ . Then $\vec{E} (\vec{r},t) = \Re \left(e^{-i\omega t}[-i\omega \chi_m(x)\phi (z)]\hat{j}\right),$ and $\vec{B} (\vec{r},t) = \Re \left(e^{-i\omega t}[\chi_m(x)\phi '(z)\hat{i} -\chi_m'(x)\phi (z)\hat{k} ]\right),$ are transverse electric (TE) waves satisfying the boundary conditions for the waveguide and solving Maxwell's equations provided that

![](images/a2905cfae0ec7835232f97910fd23ad9d30597d6957dc4f245592999ae1c1390.jpg)
FIG. 1: Cross section of a waveguide with gain (+) and loss (-) regions in the x-z plane. Arrows labeled by I, R, and T represent the incident, reflected, and transmitted waves.

$$
\phi^ {\prime \prime} (z) + [ \mathfrak {K} ^ {2} \varepsilon (z) - \mathfrak {K} _ {m} ^ {2} ] \phi (z) = 0,\tag{19}
$$

and $\phi$ and $\phi'$ are continuous functions on the z-axis [24]. For $|\Re| > \Re_{n}$ , the solution of (19) has the form (3) with z and $\kappa$ playing the roles of x and k respectively. This allows us to define a transfer matrix $\mathbf{M}(\kappa)$ for this system and introduce the right and left transmission and reflection amplitudes, $T^{l,r}$ and $R^{l,r}$ , associated with (19). These satisfy (6) and (7), and diverge whenever $M_{22}(\kappa) = 0$ .

It is not difficult to see that the right and left reflection and transmission amplitudes for the propagating TE wave coincide with $T^{l,r}$ and $R^{l,r}$ , respectively. Therefore, if we can tune the frequency $\omega$ of the incoming wave to the frequency $\omega_{\star}$ of a spectral singularity, then the amplitude of the wave will diverge as $\omega \rightarrow \omega_{\star}$ . In practice, this means that sending in a wave of frequency $\omega \approx \omega_{\star}$ will induce outgoing (transmitted and reflected) waves of considerably enhanced amplitude. The waveguide then uses a part of the energy of the laser beam to produce and emit a more intensive electromagnetic wave. Note that this effect is fundamentally different from the resonance effects associated with exciting resonance modes of a cavity resonator. Unlike the latter that has a geometric origin, the spectral singularity-related resonance effect relies on the existence of a localized region with a complex permittivity (a complex scattering potential).

The calculation of the transfer matrix $\mathbf{M}(\kappa)$ defined by (19) is analogous to that of the PT-symmetric barrier potential. In fact, $M_{22}(\kappa)$ takes the form (8) provided that we set: $k = \kappa$ , $a = \alpha$ , and $y = \mathfrak{s}\mathfrak{K}/(c\kappa^{2})$ . In particular, we can determine the values of $\omega$ and s for which the above resonance phenomenon occurs by setting $\kappa = k_{n}$ and $\mathfrak{s}\mathfrak{K}/(c\kappa^{2}) = \pm y_{n}$ . This yields $\omega = \omega_{n,m}$ and $s = s_{n,m}$ where $\omega_{n,m} := c\sqrt{k_{n}^{2} + \mathfrak{K}_{m}^{2}}$ and $s_{n,m} := \pm c k_{n}^{2} y_{n} / \sqrt{k_{n}^{2} + \mathfrak{K}_{m}^{2}} = \pm c^{2} z_{n} / \omega_{n,m}$ . For m = 1, $\hbar\omega_{p} = 0.2$ eV, $\hbar\delta = 1.25$ eV, we attain the spectral singularity with n = 0 for $\hbar\omega = \hbar\omega_{0,1} = 5$ eV, $\alpha \approx 1004$ nm and $\beta \approx 62$ nm. Figure 2 shows the graphs of the logarithm of the transmission and reflection coefficients as functions of $\omega/\omega_{0,1}$ for this case. The location and height of the pick representing the spectral singularity are highly sensitive to the values of $\alpha$ and $\beta$ .

![](images/c501eba6a30171dbe3a2647c1e12458b8826dff19485ee73e1120e9476210c5c.jpg)

![](images/20899901180f3386ee64ec0affc9415fc6868fbc8498ccc0277227b4cb069ef4.jpg)
FIG. 2: Graphs of $\log_{10}(|T^{r,l}|^{2})$ (full blue curves), $\log_{10}(|R^{l}|^{2})$ (dotted red curves), and $\log_{10}(|R^{r}|^{2})$ (dashed green curves) as a function of $\omega/\omega_{0,1}$ , for m=1, n=0, $\hbar\omega_{0,1}=5$ eV, $\hbar\omega_{p}=0.2$ eV, $\hbar\delta=1.25$ eV. For the figure on the left (right) $\alpha=1004.17$ nm, $\beta=62.0464$ nm ( $\alpha=1004$ nm, $\beta=62$ nm).

We close this section by noting that similar PT-symmetric waveguides have been considered in [18, 19]. These differ from the one we studied in that in our case the permittivity changes along the direction of the propagation of the wave (z-axis). This is essential for realizing the spectral singularity-related resonance effect.

## V. CONCLUDING REMARKS

In this article, we offered for the first time a simple physical interpretation for the spectral singularities of complex scattering potentials. In the framework of pseudo-Hermitian quantum mechanics [4], where one de-

[1] C. M. Bender and S. Boettcher, Phys. Rev. Lett. 80, 5243 (1998).

[2] A. Mostafazadeh, J. Math. Phys. 43, 2814 and 3944 (2002).

[3] C. M. Bender, D. C. Brody and H. F. Jones, Phys. Rev. Lett. 89, 270401 (2002); Erratum: 92, 119902 (2004).

[4] A. Mostafazadeh and A. Batal, J. Phys. A 37, 11645 (2004); A. Mostafazadeh, preprint arXiv: 0810.5643.

[5] C. Dembowski, H.-D. Gräf, H. L. Harney, A. Heine, W. D. Heiss, H. Rehfeld, and A. Richter, Phys. Rev. Lett. 86, 787 (2001); C. Dembowski, B. Dietz, H.-D. Gräf, H. L. Harney, A. Heine, W. D. Heiss, and A. Richter, Phys. Rev. E 69, 056216 (2004).

[6] E. Narevicius and N. Moiseyev, Phys. Rev. Lett. 81, 2221 (1998) and 84, 1681 (2000).

[7] M. Müller and I. Rotter, J. Phys. A 41, 244018 (2008).

[8] M. A. Naimark, Amer. Math. Soc. Transl. (2), 16, 103 (1960).

[9] R. R. D. Kemp, Canadian J. Math. 10, 447 (1958); J. Schwartz, Comm. Pure Appl. Math. 13, 609 (1960); G. Guseinov, private communications.

[10] A. Mostafazadeh and H. Mehri-Dehnavi, J. Phys. A 42,

fines unitary quantum systems with a non-Hermitian Hamiltonian by modifying the inner product of the Hilbert space, the presence of spectral singularities is an unsurmountable obstacle $[10]$ . In contrast, in the standard applications of non-Hermitian Hamiltonians, spectral singularities are interesting objects to study, because they correspond to scattering states (with real energy) that nevertheless behave like resonant states.

We explored the spectral singularities of a PT-symmetric potential $v_{a,3}$ that admits a realization in the form of a waveguide. We obtained the values of the physical parameters of the waveguide and the propagating TE wave for which the system displays the resonance behavior associated with the spectral singularities of $v_{a,3}$ .

Our results call for a more extensive investigation of the spectral singularities of complex scattering potentials that can be realized experimentally. This should provide means for the observation of the resonance effect that is foreseen in this article. Another line of research is to explore the spectral singularities of complex periodic potentials $[20]$ . A more basic problem is to study the consequences of spectral singularities for the implementation of quantum scattering theory. Similarly to exceptional points, presence of spectral singularities leads to subtleties associated with the existence of an appropriate resolution of identity $[21]$ . This problem may be avoided for scattering wave packets obtained by superposing eigenfunctions of the Hamiltonian. A general treatment of this problem requires a separate investigation.

Acknowledgements: I wish to thank Hossein Merhi-Dehnavi for helpful discussions. This work was supported by the Scientific and Technological Research Council of Turkey (TÜBİTAK) in the framework of the project no: 108T009 and by Turkish Academy of Sciences (TÜBA).

125303 (2009).

[11] B. F. Samsonov, J. Phys. A 38, L397 and L571 (2005); A. Mostafazadeh, J. Phys. A 39, 13506 (2006).

[12] A. Ruschhaupt, F. Delgado, and J. G. Muga, J. Phys. A 38, L171 (2005).

[13] A. Mostafazadeh, J. Math. Phys. 46, 102108 (2005).

[14] Z. Ahmed, Phys. Rev. A 64, 042716 (2001).

[15] J. G. Muga, J. P. Palao, B. Navarro, and I. L. Egusquiza, Phys. Rep. 395, 357 (2004).

[16] A. J. F. Siegert, Phys. Rev. 56, 750 (1939).

[17] H. Uncu and E. Demiralp, Phys. Lett. A 359, 190 (2006).

[18] S. Klaiman, Y. Günther, and N. Moiseyev, Phys. Rev. Lett. 101, 080402 (2008).

[19] Z. H. Musslimani, K. G. Makris, R. El-Ganainy, and D. N. Christodoulides, Phys. Rev. Lett. 100, 030402 (2008); K. G. Makris, R. El-Ganainy, D. N. Christodoulides and Z. H. Musslimani, Phys. Rev. Lett. 100, 103904 (2008).

[20] M. G. Gasymov, Func. Anal. Appl. 44, 11 (1980).

[21] A. V. Sokolov, A. A. Andrianov, and F. Cannata, J. Phys. A 39, 10207 (2006).

[22] Unlike exceptional points, spectral singularities are not

associated with the coalescence of eigenfunctions. Indeed, regardless of the presence of a spectral singularity, to each point in the continuous spectrum there corresponds two linearly independent eigenfunctions [10].

[23] Specifically, suppose that $\int_{-\infty}^{\infty}(1 + |x|)|v(x)|dx < \infty$ .

[24] This is true independently of the width of the waveguide.
