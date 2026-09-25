# Supplementary Information for "Absorbing Exceptional Point of Coherent Vortex Beams"

Hua Ding $^{1\dagger}$ , Quansen Wang $^{1\dagger}$ , Xu Wang $^{1*}$ , Yong Li $^{1*}$

$^{1}$ Institute of Acoustics, Tongji University, Shanghai, China.

\*Corresponding author(s). E-mail(s): xuwang@tongji.edu.cn; yongli@tongji.edu.cn; †These authors contributed equally to this work.

## 8 Contents

9 S1 Structure parameters of the metamaterial at CPA EP 2
10 S2 Experiment Analysis and Measurement Methods 2
11 S3 Details and discussions for CCPA EP 3
12 S4 Enhanced sensitivity of CCPA EP 8
13 S5 Circumferential mode decomposition theory 10

# S1 Structure parameters of the metamaterial at CPA EP

Part of the optimized metamaterial structure parameters depicted in Fig. 2A are listed in Table 1. The spacing between two cavities is $\Delta\theta = 3.28^{\circ}$ .

Table 1 Geometry parameters of the metamaterial

<table><tr><td>Number</td><td> $\theta_i$ </td><td> $R_i$ </td><td> $L_i$ </td></tr><tr><td>1</td><td>25.94°</td><td>74.00 mm</td><td>6.30 mm</td></tr><tr><td>2</td><td>60.33°</td><td>82.26 mm</td><td>9.40 mm</td></tr></table>

## 18 S2 Experiment Analysis and Measurement Methods

## 19 1. Vortex beam excitation

The vortex modes in a cylindrical waveguide can be excited by a loudspeaker array arranged circumferentially. The acoustic pressure $p(r,\varphi,z)$ in the waveguide, excited by a monopole source, can be expressed as:

$$
p = \sum_ {m = - \infty} ^ {\infty} \sum_ {n = 0} ^ {\infty} A _ {m, n} J _ {m} (k _ {m, n} r) e ^ {i k _ {z} z} e ^ {i m \varphi}.\tag{S1}
$$

When the infinitely long rigid-walled waveguide system contains $N_{s}$ monopole sources, the coefficient $A_{m,n}$ for the mode $(m,n)$ is given by

$$
A _ {m, n} = \frac {\omega \rho_ {0}}{2 k _ {z} C _ {m , n} ^ {2}} \sum_ {i = 1} ^ {N _ {s}} q _ {i} J _ {m} (k _ {m, n} R _ {0}) e ^ {- i k _ {z} z _ {i}} e ^ {- i m \varphi_ {i}}.\tag{S2}
$$

$q_{i}$ denotes the complex intensity of the $i$ -th monopole source. $C_{m,n}^{2}$ represents the normalization coefficient, which can be obtained by

$$
\mathrm{X} _ {m, n} ^ {2} = \pi R _ {0} ^ {2} \left(1 - \frac {m ^ {2}}{k _ {m , n} R _ {0}}\right) [ J _ {m} (k _ {m, n} R _ {0}) ] ^ {2}.\tag{S3}
$$

27 The modal coefficients in the duct satisfy

$$
\left( \begin{array}{c c c c} \chi_ {1 1} & \chi_ {1 2} & \dots & \chi_ {1 N _ {s}} \\ \chi_ {2 1} & \chi_ {2 2} & \dots & \chi_ {2 N _ {s}} \\ \vdots & \vdots & \ddots & \vdots \\ \chi_ {N _ {m} 1} & \chi_ {N _ {m} 2} & \dots & \chi_ {N _ {m} N _ {s}} \end{array} \right) \left( \begin{array}{c} q _ {1} \\ q _ {2} \\ \vdots \\ q _ {N _ {s}} \end{array} \right) = \left( \begin{array}{c} A _ {1} \\ A _ {2} \\ \vdots \\ A _ {N _ {m}} \end{array} \right)\tag{S4}
$$

where $\chi_{N_{m}N_{s}}$ represents the weight coefficient of the $N_{m}$ -th mode contributed by the $N_{s}$ -th monopole source. The Eq. (S4) accounts for all the propagating modes in the duct and can be simplified to

$$
\chi \mathbf {Q} = \mathbf {A}.\tag{S5}
$$

Selective excitation of the target modes in a cylindrical waveguide necessitates the suppression of all non-target modes by enforcing vanishing modal coefficients for undesired modes. Given the weighting matrix $\chi$ and modal coefficients vector A, the complex source intensity vector Q can be determined by Eq. (S5). The amplitude and phase information of each source in the array used for excitation are thereby obtained.

## 2. Experimental sound field mode calculation

The total acoustic field in the waveguide comprises multiple modes of incident and reflected waves:

$$
p _ {t} = \sum_ {M} J _ {M} \left(k _ {M, n} R _ {0}\right) e ^ {i M \varphi} \left(A _ {M} ^ {\mathrm{i}} e ^ {i k _ {z} z} + A _ {M} ^ {\mathrm{o}} e ^ {- i k _ {z} z}\right).\tag{S6}
$$

According to Eq. (S16), the mode coefficient of the total acoustic field with mode m can be expressed as:

$$
C _ {t} = J _ {m} \left(k _ {r, m} R _ {0}\right) \left(A ^ {\mathrm{i}} e ^ {j k _ {z, m} z} + A ^ {\mathrm{o} _ {1}} e ^ {- j k _ {z, m} z} + A ^ {\mathrm{o} _ {2}} e ^ {- j k _ {z, - m} z}\right).\tag{S7}
$$

Here, the scattering matrix can be solved by the double boundary method, and the matrix equation can be expressed as:

$$
\begin{array}{r} \left( \begin{array}{c} C _ {t _ {1}} ^ {1} \\ C _ {t _ {1}} ^ {2} \\ C _ {t _ {2}} ^ {1} \\ C _ {t _ {2}} ^ {2} \end{array} \right) = \left( \begin{array}{c c c c} J _ {m _ {1}} e ^ {j k _ {z, m _ {1}} z _ {1}} & J _ {m _ {1}} e ^ {- j k _ {z, m _ {1}} z _ {1}} & 0 & J _ {m _ {2}} e ^ {- j k _ {z, m _ {2}} z _ {1}} \\ J _ {m _ {1}} e ^ {j k _ {z, m _ {1}} z _ {2}} & J _ {m _ {1}} e ^ {- j k _ {z, m _ {1}} z _ {2}} & 0 & J _ {m _ {2}} e ^ {- j k _ {z, m _ {2}} z _ {2}} \\ 0 & J _ {m _ {1}} e ^ {- j k _ {z, m _ {1}} z _ {1}} & J _ {m _ {2}} e ^ {j k _ {z, m _ {2}} z _ {2}} & J _ {m _ {2}} e ^ {- j k _ {z, m _ {2}} z _ {1}} \\ 0 & J _ {m _ {1}} e ^ {- j k _ {z, m _ {1}} z _ {2}} & J _ {m _ {2}} e ^ {j k _ {z, m _ {2}} z _ {2}} & J _ {m _ {2}} e ^ {- j k _ {z, m _ {2}} z _ {2}} \end{array} \right) \left( \begin{array}{c} A _ {m _ {1}} ^ {\mathrm{i}} \\ A _ {m _ {1}} ^ {\mathrm{o}} \\ A _ {m _ {2}} ^ {\mathrm{i}} \\ A _ {m _ {2}} ^ {\mathrm{o}} \end{array} \right). \\ J _ {m _ {1}} = J _ {m _ {1}} \left(k _ {r, m _ {1}} R _ {0}\right), \quad J _ {m _ {2}} = J _ {m _ {2}} \left(k _ {r, m _ {2}} R _ {0}\right). \end{array}\tag{S8}
$$

The superscript of mode coefficient C denotes the number of circles in the microphone array, while the subscript indicates the two different boundary conditions set at the horn end. $z_{1(2)}$ denotes the distance from the first (second) microphone array to the sample. In this design, we choose an open boundary and an impedance boundary provided by a layer of melamine foam. From Eq. (S8), the incident and reflected amplitudes A of modes $m_{1} = +1$ and $m_{2} = -1$ can be calculated, and reflection coefficients are obtained by the ratio of the corresponding mode amplitudes.

## 50 S3 Details and discussions for CCPA EP

51 For CPA, we have

$$
r _ {+ +} r _ {- - } = r _ {+ -} r _ {- +},\tag{S9}
$$

in which all elements are non-zero. The eigenvalues of the scattering matrix established based on chirality are

$$
\lambda_ {1, 2} = \frac {1}{2} \left(r _ {+ +} + r _ {- - } \pm \sqrt {(r _ {+ +} - r _ {- - }) ^ {2} + 4 r _ {+ -} r _ {- +}}\right).\tag{S10}
$$

a
![](images/a04d4656950edae44a47c77ad1dafc1462093cb345649d6e832be2b18a0e99b4.jpg)

b
![](images/344fc3175e2dfaafed676a2ffa4bb807126f9d04cf2bd3b9ce79a1b7d435be1e.jpg)
Fig. S1 The absorption efficiency with regard to the incidence energy ratio of each mode. $\tau = \frac{\left|A_{+}^{\mathrm{i}}\right|^{2}}{\left|A_{+}^{\mathrm{i}}\right|^{2} + \left|A_{-}^{\mathrm{i}}\right|^{2}}$ is defined as the ratio of the mode $m = +1$ incident energy flow to the

total incident energy flow, and the incident energy flux ratio of the mode m = -1 is given by $1 - \tau$ . a, The absorption response of $\Delta_{i}$ to different incident energy flux ratios. The red and blue arrow-labeled points correspond to the red star and blue diamond in Fig. 4b. b, The absorption response of $\beta$ to different incident energy flux ratios. The red and blue arrow-labeled points correspond to the red star and blue hexagon in Fig. 4c.

Under the condition of CPA, the degeneracy requirement is $\lambda_{1,2}=0$ , which yields Eq.(2). The absorption coefficient can be expressed as

$$
\alpha = 1 - \frac {\left| A _ {+} ^ {\mathrm{o}} \right| ^ {2} + \left| A _ {-} ^ {\mathrm{o}} \right| ^ {2}}{\left| A _ {+} ^ {\mathrm{i}} \right| ^ {2} + \left| A _ {-} ^ {\mathrm{i}} \right| ^ {2}},\tag{S11}
$$

$$
\begin{array}{l l} _ {5 6} & \text {where A_{+} ^{\mathrm{i}} = A^{\mathrm{i}} e^{i\varphi_ {1}} and A_{-} ^{\mathrm{i}} = A^{\mathrm{i}} e^{i\varphi_ {2}} . Given |A_{+}^{\mathrm{i}}|^{2} = |A_{-}^{\mathrm{i}}|^{2} = A_{0}^{2} and \Theta =} \\ _ {5 7} & \frac {\left| A _ {+} ^ {\mathrm{o}} \right| ^ {2} + \left| A _ {-} ^ {\mathrm{o}} \right| ^ {2}}{\left| A _ {+} ^ {\mathrm{i}} \right| ^ {2} + \left| A _ {-} ^ {\mathrm{i}} \right| ^ {2}}, \text {we can deduce that} \\ & \Theta = \frac {\left(\left| r _ {+ +} \right| ^ {2} + \left| r _ {- +} \right| ^ {2} + \left| r _ {+ -} \right| ^ {2} + \left| r _ {- - } \right| ^ {2}\right) A _ {0} ^ {2} + \left(2 \left| r _ {+ +} \right| \left| r _ {+ -} \right| e ^ {i \Delta_ {+}} + 2 \left| r _ {- +} \right| \left| r _ {- - } \right| e ^ {i \Delta_ {-}}\right) A _ {0} ^ {2} e ^ {i \Delta_ {\mathrm{i}}}}{2 A _ {0} ^ {2}} \\ & = r ^ {2} \left[ 2 + \left(e ^ {i \Delta_ {+}} + e ^ {i \Delta_ {-}}\right) e ^ {i \Delta_ {\mathrm{i}}} \right], \end{array}\tag{S12}
$$

where $\Delta_{+}=\arg(r_{++})-\arg(r_{+-})$ and $\Delta_{-}=\arg(r_{-+})-\arg(r_{--})$ . The eigenvector of the CCPA EP discussed in the main text is $v_{1,2}=\begin{pmatrix}-i\\1\end{pmatrix}$ , and in this case, the CCPA EP has $\Delta_{\pm}=\Delta_{i}=\pi/2$ . The other CCPA EP corresponding to $v_{1,2}^{\prime}=\begin{pmatrix}i\\1\end{pmatrix}$ has $\Delta_{\pm}=\Delta_{i}=-\pi/2$ .

![](images/b7656f0a79d013ec51b0d6a233d2e2ffa1874a3693b29c10b87b7feda2c29420.jpg)
Fig. S2 Simulated and experimental sound field distributions for total reflection points. a,b, The sound field distributions at the blue diamond and hexagonal points in Fig. 4, respectively.

For the sake of simplicity, this work takes the equal energy flux incidence as an example to realize CCPA EP. As the equal energy flux incidence condition is destroyed $\left(\left|A_{+}^{\mathrm{i}}\right|^{2}\neq\left|A_{-}^{\mathrm{i}}\right|^{2}\right)$ , the changes of sound absorption efficiency under different incidence phase differences $\Delta_{i}$ and structure rotation angle $\beta$ are demonstrated in Fig. S1.
Considering that the system response associated with $\beta$ is the same within the ranges $[-\pi,0]$ and $[0,\pi]$ , the absorption coefficient spectra are only presented for the range $[0,\pi]$ . Different energy flux ratios lead to significant changes of absorption efficiency.
Whereas in fact the design of CPA allows for the adjustment of the incident energy flux ratio according to the response of the structure, which may inspire the study of CPA EP under unequal energy flux conditions.

![](images/3c502ac4f7fbd760702edc791070a812f8da7305a99dcc273ce0ff4678f8a614.jpg)

![](images/c4a447a4b646379dca09c22638c4b8629f3116ef688804d91f4ca6d7e4261d7b.jpg)

![](images/ccbbfdf9a539da16a9d182426d9f5e1c5044a5eaf08d6c4b70b6e5338f68174c.jpg)

![](images/ad489fb8e7d86ab17b7db0acc57ffd2e4a7f209f034d62e295fc518d69a4b57b.jpg)

![](images/e9d3846503f1b44d2c57a6e0c19ecd0d52e2f4c86b000c16d96866ccb8c24ed5.jpg)

![](images/4cd6886e9750ad42147dfea5fbd6896f0bc7a5f9df41ebbeca187358fa45e486.jpg)

![](images/da52d8b465ed2712d85d3e69489d1335e67050f02706f6a9d006e083fadb8959.jpg)

![](images/fb0f35aaa0d977251c5d410bfc1a431466c5da7c5f2fe3bde7aa12ce3cea1087.jpg)

![](images/344d253efabd90f420bc737048028135e1877c7e0f967402ca79f0b44abce5b5.jpg)

![](images/c446d44440024143ab9e28816b1c262fad6f5f89614d3cc596f3ceeae272e098.jpg)

![](images/bfe9fcb1ebf8b89d9e8319814e5c3fcadcc33b1a091cd5c07cbe6895523445ce.jpg)
Fig. S3 The CPA EP corresponding to the eigenvector $v_{1,2}^{\prime}=\binom{i}{1}$ . a-c, $\beta$ -rotation operation ( $\beta=\pi/2$ ). d-f, mirror-plane operation. a,d, Reflectivity spectra. b,e, Absorption spectra for dual-mode (black) and single-mode (blue and red) incidence. c,f, Pressure field patterns of sound pressure p at CPA EP.

![](images/89e6fb0c3d2d0e3075efd3f853fdb9b02bd3a656edeebfb5d1ca2cf31115be15.jpg)

![](images/e72f65e038b37285026f6775d5b99573ac7c40b39a6ea90c09534a2c77b24d71.jpg)

![](images/b655f4b81d6ea43aaaa506218e5a1a502e94e5773baa3b6b4ba8134f55b0bedb.jpg)

![](images/d7604911649caa16122c7a93dee1c8334f6e36af6fa4d5301ea967b25880d69c.jpg)
Fig. S4 Two manipulation schemes for the CPA EP corresponding to the eigenvector $v_{1,2}^{\prime}=\binom{i}{1}$ . a, Modulate incident phase difference. As the incident phase difference $\Delta_{i}$ varies from $-\pi$ to $\pi$ , the absorption coefficient undergoes a continuous transition from 0 to 1, and the reflection phase difference spans the full range $[-\pi,\pi]$ . The maximum and minimum absorption are achieved at $\Delta_{\pm}=-\pi/2$ and $\Delta_{\pm}=\pi/2$ , respectively. b, Rotate the metamaterial. The absorption coefficient and reflection phase difference $\Delta_{\pm}$ exhibit identical variation patterns over the intervals $\beta\in[-\pi,0]$ and $\beta\in[0,\pi]$ . The red diamond and hexagon mark the CPA EP, while the blue pentagrams indicate the points of total reflection which correspond to the initial CPA EP configuration.

According to Eq. S12, it can be inferred that, with all other conditions held constant, the system exhibits total reflection when $\Delta_{i} = -\pi/2$ , which is determined by the coherent properties of sound waves. Another way provided by the geometric phase to achieve the same reflective effect is to rotate the metamaterial by $\pi/2$ or $-\pi/2$ . The sound field distributions in Fig. S2 further illustrate the total reflection efficiency of the system. The rotational symmetry of the metamaterial and the chirality of the system facilitate EP manipulation. The observed total reflection points in Fig. S2b manifest scattering features of CCPA EPs when excited through non-eigenchannel, demonstrating EP state transformation. As evidenced in Fig. S3, the system excited by the eigenvector $v_{1,2}^{\prime} = \begin{pmatrix} i \\ 1 \end{pmatrix}$ simultaneously demonstrates reflection coefficient degeneracy and coherent perfect absorption at $f_{0}$ . Spectral characteristics in the vicinity of this CCPA EP exhibits identical to the initial CCPA EP.

The rotational operation and the mirror-symmetry operation define two distinct methods for EP transitions. The scattering matrix after the rotation operation is $S_{R} = XRXSR^{-1}$ , which is obtained by the action of the rotation operator R and the inversion operator $\mathbf{X} := \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$ . The mirror operator $\mathbf{M} = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$ , when applied to the scattering matrix $\mathbf{S}$ and $\mathbf{S}_{\mathrm{R}}$ , yields $\mathbf{S}_{\mathrm{M}} = \mathbf{MSM}^{-1} = \begin{pmatrix} r_{--} & r_{-+} \\ r_{+-} & r_{++} \end{pmatrix}$ , $\mathbf{S}_{\mathrm{RM}} = \mathbf{MS}_{\mathrm{R}}\mathbf{M}^{-1} = \begin{pmatrix} r_{--}e^{i2\beta} & r_{-+} \\ r_{+-} & r_{++}e^{-i2\beta} \end{pmatrix}$ . When $\beta = \pi/2$ with all other system parameters fixed, the scattering matrices satisfy $\mathbf{S} = \mathbf{S}_{\mathrm{RM}}$ and $\mathbf{S}_{\mathrm{R}} = \mathbf{S}_{\mathrm{M}}$ , indicating that the system recovers its original EP state through spatial symmetry transformations. Specifically, the system converges to the identical CPA EP shown in Fig. S3 upon both $\beta$ -rotation ( $\beta = \pi/2$ ) and mirror-plane operation. As expected, with variation of $\Delta_{\mathrm{i}}$ and $\beta$ , this CCPA EP maintains similar response characteristics (Fig. S4) to the initial CCPA EP. Furthermore, if the rotation and mirror operations are performed successively, the system will return to the initial CCPA EP (Fig. S5).

![](images/eee5b18cbefde77292c22f5d4ddb1a58823274fdc9492aa94e959546138149f4.jpg)

![](images/e64c7eabbb1be51cedec00b0f85989bcd532c1f376e19697cad844ff682d9fd8.jpg)

![](images/ed5a1d69056ad024f6d6bb4291bbed5b4031e219eaebb214e5e5d9e472a82fb0.jpg)

![](images/1f0fe8ea8a0e710afd00428cc2b1a7b87ff960cd8709f4a44c68a5060f533387.jpg)

![](images/93dfe97c5403e17d1dedd4f1ca4810f2311d1a5f2971f77e8a175068390b2676.jpg)
Fig. S5 Scattering characteristics of the CPA EP [corresponds to the eigenvector $v_{1,2} = \begin{pmatrix} -i \\ 1 \end{pmatrix}$ ] after rotation and mirror-flip operations. a, Reflectivity spectra. b, Absorption spectra for dual-mode (black) and single-mode (blue and red) incidence. c, Pressure field patterns of sound pressure p at CPA EP.

## S4 Enhanced sensitivity of CCPA EP

The higher-order mode channels via chiral waves provides a pathway to enhanced sensitivity of CCPA EPs. Equation (4) can be generalized to higher-order TC modes and is expressed as $\mathbf{S}_{\mathrm{R}}=\begin{pmatrix}r_{++}e^{-i2m\beta}&r_{+-}\\r_{-+}&r_{--}e^{i2m\beta}\end{pmatrix}$ . The requirement of higher rotational symmetry in the metamaterial to support higher-order modes enhances the sensitivity of the corresponding eigenvalues to rotational operations. As an example, the higher-order chiral wave modes with TC = ±2 are considered. The metamaterial possessing $C_{4}$ symmetry enables the propagation of modes with TC = ±2. The corresponding structural parameters are detailed in Table 2. The reflection and absorption coefficients are shown in the Fig. S6. At the operational frequency of 3600 Hz, the system exhibits coherent perfect absorption at an exceptional point, with the reflection coefficient $r_{++} = 0.45 + 0.21i$ .

Table 2 Geometry parameters of the metamaterial for TC = ±2.

<table><tr><td>Number</td><td> $\theta_i$ </td><td> $R_i$ </td><td> $L_i$ </td></tr><tr><td>1</td><td>16.32°</td><td>67.72mm</td><td>14.81mm</td></tr><tr><td>2</td><td>35.28°</td><td>70.91mm</td><td>21.77mm</td></tr><tr><td>3</td><td>19.02°</td><td>65.77mm</td><td>7.73mm</td></tr></table>

![](images/0eaef2ed270481495d29bb6cd86f909a49621c5e37dccd0cc366e69a9efc37fe.jpg)
Fig. S6 The CCPA EP for $\mathbf{TC} = \pm 2$ . a, Reflectivity spectra. b, Absorption spectra for dual-mode (black) and single-mode (blue and red) incidence. c, Pressure field patterns of sound pressure $p$ at CPA EP.

As discussed in the main text, the CCPA EP with TC = 2 exhibits enhanced sensitivity to $\beta$ perturbations, manifested by its absorption coefficient undergoing four full cycles within a $2\pi$ variation of $\beta$ (Fig. S7). The intrinsic properties of the system change accordingly. For the CCPA EP with TC = $\pm1$ , the eigenvalues undergo two cycles during a full $2\pi$ rotation of the metamaterial (Fig. 5a). For the TC = $\pm2$ case, the eigenvalue traverses four cycles within the same rotational period (Fig. 5b).

Within the interval of $\theta\in[0,\pi/9]$ , the sensitivity of the EP demonstrates significant enhancement (Fig. 5c). This demonstrates the feasibility of leveraging higher-order TC channels to enhance the sensitivity of CPA EP.

![](images/4ae4a238fd19cf431e2117c2a18ed58a199fb5d60deb26944276a04dbbcc7fb8.jpg)

![](images/902ff746f5561f6e877cbd4646c5fb2e2b0641dbafdea58c379b0427cfc87484.jpg)

b
![](images/d8f01de7a603c74d1d42847465fdc601f4f7d363c1ae42399c538c0dbc57b676.jpg)

![](images/c6a7c17eb4fdf7691c8605dbfbc1ad53ca8f2fcea44b879974c3f32940f09b9b.jpg)
Fig. S7 The two absorption modulation responses of CCPA EP with $TC = \pm2$ . a, Rotating one of the incident vortices, $|+\rangle$ , by a phase of $\Delta_{i}$ (the left panel). The degree of absorption undergoes a continuous transition from 0 to 100% as $\Delta_{i}$ completes a full $2\pi$ cycle (the left panel). During the process, the system itself characterized by $\Delta_{\pm}$ remains unchanged (the right panel). The red pentagram and the blue diamond indicate the maximum and minimum absorption, respectively, corresponding to "on" and "off" of the symbol of the CCPA-EP. b, Rotating the metamaterial and introducing a geometry phase to the system. The degree of absorption undergoes four cycles of transition from 0 to 100% during a full $2\pi$ rotation, underpinning the $C_{4}$ -rotational symmetry of the system. The red pentagram and the blue hexagon mark two different CCPA-EPs, where one exhibits perfect absorption while the other results in total reflection under the same coherent inputs.

## S5 Circumferential mode decomposition theory

The circumferential mode decomposition theory is a fundamental method for analyzing the acoustic field in cylindrical waveguide systems. This discrete analytical approach leveraging spatial sampling data enables rapid determination of vortex acoustic field mode composition in waveguides. Azimuthal Fourier expansion of the sound field is

$$
p (\varphi) = \sum_ {m} C _ {m} e ^ {i m \varphi}.\tag{S13}
$$

23 Multiply both sides of Eq. (S5) by $e^{im\varphi}$ and integrate over $[0,2\pi]$ :

$$
\int_ {0} ^ {2 \pi} p (\varphi) e ^ {- i M \varphi} d \varphi = \int_ {0} ^ {2 \pi} \left(\sum_ {m} C _ {m} e ^ {i m \varphi}\right) e ^ {- i M \varphi} d \varphi .\tag{S14}
$$

The coefficient $C_{m}$ can be obtained using the orthogonality relationship of functions

$$
C _ {M} = \frac {1}{2 \pi} \int_ {0} ^ {2 \pi} p (\varphi) e ^ {- j M \varphi} d \varphi .\tag{S15}
$$

In the case of discrete sampling in simulations and experiments, the integral in Eq. S15 can be rewritten as

$$
C _ {M} = \frac {1}{N} \sum_ {n} p (\varphi_ {n}) e ^ {- j M \varphi_ {n}},\tag{S16}
$$

where $\varphi_{n} = \frac{2\pi n}{N}$ is the azimuthal angle of the n-th measurement point. As higher-order propagating modes ( $|m| > 1$ ) are suppressed in the design, the circumferential mode decomposition for $m = \pm1$ can be accomplished by placing microphones at four measurement points (N = 4). As the mode order increases, the number of microphones also increases correspondingly. For example, for $m = \pm2$ , six microphones are arranged circumferentially to achieve accurate mode decomposition.
