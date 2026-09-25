# Controlling Sound in Non-Hermitian Acoustic Systems

Zhongming Gu, $^{1,2,\ddagger}$ He Gao, $^{1,2,\ddagger}$ Pei-Chao Cao, $^{3,\ddagger}$ Tuo Liu $^{\textcircled{ID}}$ , $^{1,2}$ Xue-Feng Zhu, $^{3,*}$ and Jie Zhu $^{1,2,\dagger}$

$^{1}$ Department of Mechanical Engineering, The Hong Kong Polytechnic University, Hung Hom, Kowloon, Hong Kong SAR, China

$^{2}$ The Hong Kong Polytechnic University Shenzhen Research Institute, Shenzhen 518057, China $^{3}$ School of Physics and Innovation Institute, Huazhong University of Science and Technology, Wuhan 430074, China

![](images/4f539309dcb73b8992da72afa7d7e40124faf267a5c11245c51f5d7462d863ce.jpg)

(Received 15 June 2021; revised 30 August 2021; accepted 3 September 2021; published 4 November 2021)

The concept of non-Hermitian physics has inspired numerous intriguing phenomena in classical wave systems. By introducing judicious arrangements of gain and loss media, non-Hermitian modulation in acoustic metamaterials can give rise to many extraordinary wave-matter interactions that cannot be achieved naturally, which have been verified theoretically and experimentally by a series of previous works. Here, we aim to review recent developments in this topic. First of all, we explain the basic concepts and mathematical tools to deal with non-Hermitian acoustics by studying some pedagogical examples. Then, we present some interesting works that demonstrate the superior abilities of non-Hermitian modulation in wave manipulation. Also, we pay attention to the study of topological systems with non-Hermitian modulation, with a special emphasis on nontrivial topological states induced by non-Hermiticity alone. Finally, we present an outlook on the potential directions and applications for future works. We hope this review can provide a better understanding to stimulate research on acoustics beyond the Hermitian regime.

DOI: 10.1103/PhysRevApplied.16.057001

## I. INTRODUCTION

The Hermitian Hamiltonian plays a key role in the study of modern physics and is widely used to describe a closed physical system with respect to the conservation of energy. However, the practical system is more complex and may contact the ambient environment to some degree or have inevitable dissipation naturally, which, in turn, renders the Hermitian description unsuitable in the presence of gain or loss modulation. Lately, the non-Hermitian description, originating from the study of atomic nucleus by Gamow [1], has attracted ongoing attention in research of the nonconservation system within both classical and quantum physics [2–7]. Particularly, a unique class of non-Hermitian system with balanced gain and loss, also known as the parity-time (PT) symmetric system, is proved to hold an exceptional point (EP), where two or more eigenstates emerge in the real part of frequency, while they start to diverge in the imaginary part [8]. Around the EP, a plethora of intriguing wave phenomena are observed, such as unidirectional invisibility [9,10], enhanced sensing [11–13], coherent perfect absorption [14-16], and asymmetric mode switching [17-20].

Acoustics can be regarded as a feasible and versatile platform to verify non-Hermitian concepts $[21–26]$ , which not only provides a better understanding but also brings non-Hermitian physics closer to real applications. First, analogous to quantum physics, the effective non-Hermitian Hamiltonian can be employed to describe the acoustic coupled system with well-designed sound leakage or additional loss $[21,27]$ . Additionally, due to simple fabrication with three-dimensional printing technology, such engineered systems with non-Hermitian modulation can be conveniently measured to exhibit the complicated ideas of non-Hermitian physics $[28,29]$ . On the other hand, the intriguing wave behavior enabled by non-Hermiticity can contribute to the development of acoustic metamaterials beyond the Hermitian restriction and inspire acoustic devices with more unexpected functionalities $[30–39]$ . Some interesting works were proposed to show the extraordinary abilities of the acoustic metasurface with non-Hermitian modulation in manipulating the wave front asymmetrically $[32,40–46]$ . Moreover, asymmetric acoustic absorption, for which waves incident from one side are absorbed perfectly and waves incident from the opposite side are reflected totally, can be obtained when the system is tuned to work around the EP, which may have potential applications in many areas of acoustic engineering [47-52]. In addition, the non-Hermitian topological insulator can also be studied on an acoustic platform [53-56]. With the careful arrangement of gain and loss, nontrivial topological features can be probed in a macroscopic structure with the advantages of facile fabrication and measurement.

This paper aims to provide an insight into recent works that use an acoustic platform to verify the concepts of non-Hermitian physics and inspire acoustic wave manipulations with non-Hermitian modulation. In Sec. II, we introduce the basic concept of non-Hermitian physics by using a closed-cavity system and an open scattering system. Then, some experimental works related to more complex models are studied with the help of multiple cavities or active elements. In Sec. III, we review acoustic wave manipulation enabled by non-Hermiticity, including asymmetric transmission and reflections, asymmetric absorption around the EPs, and some other interesting applications. In Sec. IV, we present recent works on topological insulators with non-Hermitian modulations in acoustic systems of two types. The first one is that the sample works as a topological insulator in the Hermitian regime, while showing more topological features with the existence of non-Hermiticity. The second one is the topological properties of the system induced by non-Hermiticity alone. In Sec. V, we provide an outlook on the further study of non-Hermitian acoustics and give a conclusion.

## II. THE CONCEPT OF NON-HERMITIAN ACOUSTICS AND BEYOND

## A. The non-Hermitian acoustics in a closed-cavity system

First, we consider two resonators that are connected by a narrow tube in the Hermitian case. Air, with a density $\rho_{0}$ and speed of sound $c_{0}$ , is sealed in the closed system. For a single resonator, there exists a fixed dipole mode with a certain frequency. However, for two resonators linked by a hollow tube, coupling will be introduced to split the dipole mode into an even mode and an odd mode [29], as shown in the upper panel of Fig. 1(a). Typically, the strength of the coupling can be adjusted by changing the position and size of the tube. Here, for simplicity, we fix the position of the tube at the top end of both resonators, which can give rise to strong coupling. Then, a variable, d, that describes the dimension of the opening of the tube is employed to change the coupling strength. The relationship, plotted in the lower panel of Fig. 1(a), shows that a larger value of d, associated with stronger coupling, can lead to a larger splitting of the eigenmodes in the spectrum.

(a)
![](images/b6eaf7cdc20f09d84698a91da2c6b4ced45b2a8e93539806d7ff3a3c5f509a7a.jpg)

(b) Non-Hermitian, lossy
![](images/f02f1cd919487692a2fbcdefa5f3db2e99c01f09c6e1cf92faf6653298100950.jpg)

![](images/b3e671219534c2452f7cf1a97279f07e7ee93fd4e8c79cfda46a9a5a83482b03.jpg)

(c) Non-Hermitian, PT
![](images/3668eb8243a866bd23070d9610bdbebc008f15c58234cbd802092707441bc057.jpg)

![](images/349461338c055d9469c7e662f30b01db1893e2904422c0fd89330045edc41f12.jpg)

![](images/a0df61c97b2fb64316c1c76406a0f71cbdc9dd75ed59eaf8fe95589b86f9542f.jpg)
FIG. 1. Non-Hermitian acoustics in the closed-cavity system. Two identical acoustic resonators with eigenfrequency $\omega_{0}$ are connected by a narrow tube. Existence of the tube can introduce coupling to split the eigenmode, while the amplitude of coupling can be adjusted by changing the size of the opening of the tube, d. (a) Upper panel shows, in the Hermitian case, two passive resonators (depicted in blue) are connected by a narrow tube. Dipole mode with eigenfrequency $\omega_{0}$ will be split into odd mode and even mode with frequencies of $\omega_{+}$ and $\omega_{-}$ . Lower panel shows the change of splitting eigenmodes as a function of d. (b) Upper panel shows, in the non-Hermitian case, a passive resonator (depicted in blue) and a lossy resonator (depicted in green) connected by a narrow tube. Fields localized in one of the resonators in the PT-broken phase. Lower panel shows the real and imaginary parts of eigenfrequency that change as a function of non-Hermitian modulation, $\gamma$ . (c) Upper panel shows, in the non-Hermitian case, an amplified resonator (depicted in yellow) and a lossy resonator (depicted in green) connected by a narrow tube. Lower panel shows real and imaginary parts of the eigenfrequency that change as a function of non-Hermitian modulation, $\gamma$ .

By adding gain or loss in the resonator, we can extend the system to the non-Hermitian regime. At a given value of $d$ , the upper panel of Fig. 1(b) shows that a passive resonator and a lossy resonator are connected by a hollow tube. The lossy part has parameters of $\rho_{L} = \rho_{0}$ and $c_{L} = c_{0}(1 + i\gamma)$ , where variable $\gamma$ denotes non-Hermiticity. In a practical realization, loss can be introduced by adding an absorptive material or sound leakage in the resonator. Under non-Hermitian modulation, the system may pass through the EP that links the PT-symmetric phase and PT-broken phase. Different from the Hermitian case, the acoustic field is localized in either of the resonators in the PT-broken phase. The two split modes can refer to a more-loss mode and a less-loss mode. It can also be reflected in the evolution of the real and imaginary parts of the eigenfrequency, as shown in the lower panel of Fig. 1(b). Non-Hermitian degeneracy can be clearly observed, where the real part of the eigenfrequency merges while the imaginary part separates. Beyond the EP, one branch of the imaginary part becomes larger and the other branch approaches zero with a small imaginary part of the eigenfrequency. In addition, the system can also have non-Hermitian modulation with balanced gain and loss, as shown in the upper panel of Fig. 1(c). In the simulations, the gain medium has parameters of $\rho_{G} = \rho_{0}$ and $c_{G} = c_{0}(1 - i\gamma)$ , while, in the realization, the active elements can be used to mimic the gain effect [43,57,58]. Similar to the lossy case, the two localized modes can refer to a gain mode and a loss mode. The system also undergoes a transition from the exact PT phase to a broken PT phase, as plotted in the lower panel of Fig. 1(c). In this case, two branches of the imaginary part of the eigenfrequency are symmetric with respect to the coordinate zero point. The one on the positive part represents loss modulation, and the other branch on the negative part represents gain modulation.

The progress discussed above can be described by the coupled-mode theory $[27,59]$ . Since the two resonators are identical, the evolution of the eigenfrequency induced by reciprocal coupling and non-Hermitian modulation can be studied in the time domain by solving the coupled-mode equations as follows:

$$
\frac {d a _ {1}}{d t} = i \omega_ {1} a _ {1} - \gamma_ {1} a _ {1} + i \kappa a _ {2},\tag{1}
$$

$$
\frac {d a _ {2}}{d t} = i \omega_ {2} a _ {2} - \gamma_ {2} a _ {2} + i \kappa a _ {1},\tag{2}
$$

where $a_{1}$ and $a_{2}$ indicate the amplitudes of the field in the two resonators. $\kappa$ indicates the strength of reciprocal coupling. $\omega_{1}$ and $\omega_{2}$ indicate the resonant frequencies of the two resonators and should be equal. $\gamma_{1}$ and $\gamma_{2}$ indicate non-Hermitian modulation imposed on the resonators. Assuming that the time dependence of the form is $e^{i\omega t}$ , we can obtain an expression for the eigenfrequency of the coupled cavities:

$$
\omega_ {\pm} = \omega_ {0} + i \frac {\gamma_ {1} + \gamma_ {2}}{2} \pm \sqrt {\kappa^ {2} - \frac {(\gamma_ {1} - \gamma_ {2}) ^ {2}}{4}}.\tag{3}
$$

In the Hermitian case with $\gamma_{1}=\gamma_{2}=0$ , Eq. (3) will degenerate to $\omega_{\pm}=\omega_{0}\pm\kappa$ , which implies that the coupling has only a modulation on the real part of the eigenfrequencies. The eigenfrequencies of the coupled cavities bifurcate as the strength of coupling increases, which is in good agreement with the results shown in Fig. 1(a). However, in the non-Hermitian case with solely loss modulation, for example, $\gamma_{1}=0$ and $\gamma_{2}>0$ , Eq. (3) can be reformulated as $\omega_{\pm}=\omega_{0}+i(\gamma_{2}/2)\pm\sqrt{\kappa^{2}-(\gamma_{2}^{2}/4)}$ . Under this circumstance, increasing the value of $\gamma_{2}$ will have an impact on the eigenfrequencies in the real and imaginary parts simultaneously. The real part merges gradually, while the imaginary part increases before the EP. At the critical point where $\kappa^{2}-\gamma_{2}^{2}/4=0$ , the two branches of the real part coalesce and the imaginary part begins to bifurcate into two branches with more absorption and less absorption. Particularly, in the non-Hermitian case with balanced gain and loss, like $\gamma_{2}=-\gamma_{1}>0$ , Eq. (3) can be further simplified to $\omega_{\pm}=\omega_{0}\pm\sqrt{\kappa^{2}-\gamma_{1}^{2}}$ . Accordingly, increasing the value of $\gamma_{1}$ will change the eigenfrequency in the complex realm. Similarly, the real part merges and the imaginary part bifurcates at the EP. However, in the exact PT phase, the imaginary part keeps zero rigorously while in the PT-broken phase. The two branches of the imaginary part bifurcate symmetrically, with one referring to amplification and the other referring to absorption.

On the basis of Eqs. (1) and (2), we can also obtain the effective Hamiltonian of the system in matrix form:

$$
H = \left( \begin{array}{c c} \omega_ {0} - i \gamma_ {1} & - \kappa \\ - \kappa & \omega_ {0} - i \gamma_ {2} \end{array} \right).\tag{4}
$$

By solving the eigenvalues of this matrix, the expression of the eigenfrequencies of the coupled-cavity system can also be deduced. In fact, analogous with the tight-binding theory, which is widely used in condensed-matter physics, the effective Hamiltonian has become a versatile tool that plays a key role in studying topological acoustics and non-Hermitian acoustics, which we discuss later in Sec. V.

## B. The non-Hermitian acoustics in an open scattering system

Now, we utilize a straight pipe with balanced gain and loss modulations to study non-Hermitian acoustics in the open scattering system $[60]$ . In contrast to the closed-cavity system with discrete resonant frequencies, the open acoustic waveguide can support continuous propagation

modes, which can be well studied by the transmission and reflection properties. Figure 2(a) shows this one-dimensional PT-symmetric structure with incident waves from both sides. The background medium is air with acoustic parameters $\rho_{0}$ and $c_{0}$ , and the parameters of the gain and loss medium are $\rho_{g} = \rho_{0}$ and $c_{g} = c_{0}(1 - i\gamma)$ and $\rho_{l} = \rho_{0}$ and $c_{l} = c_{0}(1 + i\gamma)$ , respectively. Both areas have length L. By considering the continuous conditions of sound pressure and flow, the transmission coefficient, t, and reflection coefficient, r, can be deduced easily, as follows:

$$
r _ {\mathrm{L}} = \frac {(m _ {g} - 1) A _ {g} + (m _ {g} + 1) B _ {g}}{(m _ {g} + 1) A _ {g} + (m _ {g} - 1) B _ {g}},\tag{5}
$$

$$
r _ {\mathrm{R}} = \frac {(m _ {l} - 1) A _ {l} + (m _ {l} + 1) B _ {l}}{(m _ {l} + 1) A _ {l} + (m _ {l} - 1) B _ {l}},\tag{6}
$$

$$
t = \frac {2 m _ {g}}{(m _ {g} + 1) A _ {g} + (m _ {g} - 1) B _ {g}},\tag{7}
$$

where

$$
A _ {g (l)} = \frac {[ m _ {l (g)} + m _ {g (l)} ] [ 1 + m _ {l (g)} ] e ^ {i k _ {l (g)} L} + [ m _ {l (g)} - m _ {g (l)} ] [ 1 - m _ {l (g)} ] e ^ {- i k _ {l (g)} L}}{4 m _ {l (g)}} e ^ {i k _ {g (l)} L}
$$

and

$$
B _ {g (l)} = \frac {[ m _ {l (g)} - m _ {g (l)} ] [ 1 + m _ {l (g)} ] e ^ {i k _ {l (g)} L} + [ m _ {l (g)} + m _ {g (l)} ] [ 1 - m _ {l (g)} ] e ^ {- i k _ {l (g)} L}}{4 m _ {l (g)}} e ^ {- i k _ {g (l)} L}
$$

with $m_{g} = m_{l}^{*} = 1 - i\gamma$ and $k_{g(l)} = k_{0}/m_{g(l)}$ . The subscript L (R) represents waves incident from the left (right) side. Different from the passive system obeying the energy-conservation principle $T + R = 1$ , with $T \equiv |t|^{2}$ and $R \equiv |r|^{2}$ , the essence of the PT-symmetric structure will lead to a generalized conservation relationship:

$$
\sqrt {R _ {L} R _ {R}} = | T - 1 |.\tag{8}
$$

One of the most interesting properties of this relationship is allowing the existence of unidirectional reflectionless behavior for the incidence from a certain direction, while the transmission for both sides remains unitary. These scattering properties can also be verified in simulations by using the finite-element method, as shown in Fig. 2(b). The reflection for the right incidence vanishes several times, related to the reflectionless phenomenon, in a wide frequency band. At these frequencies, the transmission coefficients for both sides are equal to one, while the reflection coefficient for the left incidence has nonzero values. In fact, these special points are mathematically meaningful and can be regarded as the EPs of the scattering matrix constructed by the following relationships:

$$
t I ^ {+} + r _ {R} I ^ {-} = O ^ {+},
$$

$$
r _ {\mathrm{L}} I ^ {+} + t I ^ {-} = O ^ {-},\tag{9}
$$

(10)

(11)

where I and O represent the input and output signals, respectively. The superscript + (−) indicates that the wave propagates from left (right) to right (left). Then, the scattering matrix can be defined as

$$
S _ {1} = \left( \begin{array}{c c} t & r _ {R} \\ r _ {L} & t \end{array} \right).
$$

The eigenvalues of $S_{1}$ can be expressed as $\lambda_{1,2}=t\pm r_{L}r_{R}$ . According to the simulations, we plot the logarithm of the modulus of the eigenvalues of $S_{1}$ (black lines) in Fig. 2(c). Phenomenally, the emergence of the EPs has good agreement with the frequencies where T=1. A simple criterion can be derived that the change of the transmission is consistent with the transition of the PT phase. When T<1, the system is in the PT-symmetric phase. When T>1, the system is in the PT-broken phase. This conclusion can also be deduced by another format of eigenvalues. By substituting Eq. (8), we can have $\lambda_{1,2}=t\left[1\pm i\sqrt{(1-T)/T}\right]$ . It is obvious that, when T<1, the eigenvalues are unimodular and nondegenerate, associated with the PT-symmetric phase. However, when T>1, the eigenvalues are nonunimodular and degenerate, associated with the PT-broken phase.

![](images/530b0993a5ea9768944ce0b2e962e9d77e5d032f9842fefef5930819def48e3f.jpg)

![](images/5f3aab54a5c3d3ded6384dfc6b9a581f7f5f57e3d7b6f2ab74215ed7e518ab44.jpg)

![](images/10a14440f00f459f8e51546e237e4b2cb51201a75e4646d9b417a7eba0275ea1.jpg)

Notably, there is an alternative definition of the scattering matrix, $S_{2}$ , where the transmission coefficients are on the off-diagonal terms [60]. Consequently, the eigenvalues of the scattering matrix need to be rewritten as $\lambda_{1,2} = \left[r_{L} + r_{R} \pm \sqrt{(r_{L} - r_{R})^{2} + 4t^{2}}\right]/2$ . In contrast to $S_{1}$ having a series of EPs with increasing frequency, $S_{2}$ has only one EP that links the PT-symmetric phase and PT-broken phase, as shown in Fig. 2(c) by the red line. This EP is stable and moves only when the non-Hermitian modulation changes. Moreover, the unidirectional reflectionless propagation will not be bound to appear at the EP. In fact, if we add the hard-boundary condition to the PT-symmetric structure shown in Fig. 2(a), we will have a closed non-Hermitian system with discrete eigenfrequencies in the complex frequency domain. Its complex energy spectrum also shows a transition between real and complex conjugate pairs, which is consistent with the description of the eigenvalues of $S_{2}$ . Thus, we can say that $S_{2}$ is of physical significance that reflects the PT phase transition of the system, in a similar way to the effective non-Hermitian Hamiltonian of the closed-cavity system.

## C. High-order EPs observed in a multiple-cavity system

Beyond the two-state system that possesses a single EP determined by a simple $2 \times 2$ non-Hermitian matrix, the high-order EPs and more-complex non-Hermitian physics can be found in the multiple-cavity system with non-Hermitian modulation [61–64]. Ding et al. extended this concept to the four coupled acoustic resonators with asymmetric losses [28], as shown in Fig. 3(a). The four cavities are placed in a square arrangement and connected to the adjacent one with small tubes. As described in the inset of Fig. 3(a), the top two cavities, D and C, have the same

FIG. 2. Non-Hermitian acoustics in an open scattering system. (a) Two areas with the same length, L = 5 cm, are located in the middle of a straight pipe symmetrically. One (depicted in yellow) has the gain modulation $[\rho_{g} = \rho_{0} \text{ and } c_{g} = c_{0}(1 - i\gamma)]$ and the other one (depicted in green) has the loss modulation $[\rho_{l} = \rho_{0} \text{ and } c_{l} = c_{0}(1 + i\gamma)]$ . Acoustic waves can propagate from both sides. (b) Transmission and reflection coefficients are retrieved from the scattering properties. (c) Eigenvalues of the scattering matrix in two different formulas versus frequency.

eigenfrequency, $\omega_{1}$ , while the bottom two cavities, A and B, have the same eigenfrequency, $\omega_{2}$ . Also, the strength of the coupling linking A and D, as well as B and C, is set as t. The strength of the coupling linking A and B, as well as D and C, is set as $\kappa$ . The intrinsic loss induced by the thermoviscous effect of air in each cavity is $\Gamma_{0}$ , while the additional loss utilized in the cavities on the off-diagonal term is $\Delta\Gamma$ . In experiments, an absorptive sponge is employed as the additional loss and a little putty is added to eliminate the swing of the resonant frequency caused by the sponge. Then, the effective non-Hermitian Hamiltonian can be expressed by a $4 \times 4$ matrix as

$$
H = \left( \begin{array}{c c c c} \omega_ {2} - i \Gamma_ {0} & \kappa & 0 & t \\ \kappa & \omega_ {2} - i \Gamma & t & 0 \\ 0 & t & \omega_ {1} - i \Gamma_ {0} & \kappa \\ t & 0 & \kappa & \omega_ {1} - i \Gamma \end{array} \right),\tag{12}
$$

where $\Gamma = \Gamma_{0} + \Delta\Gamma$ . The eigenvalues of this matrix can be solved easily,

$$
\tilde {\omega} _ {j} = \omega_ {0} - i \frac {\Gamma + \Gamma_ {0}}{2} \pm \frac {1}{2} \sqrt {\Delta_ {1} \pm 4 \sqrt {\Delta_ {2}}}, j = 1, 2, 3, 4,\tag{13}
$$

where $\omega_{0}=(\omega_{1}+\omega_{2})/2$ , $\Delta_{1}=-(\Delta\Gamma)^{2}+4\kappa^{2}+4t^{2}+(\Delta\omega)^{2}$ , and $\Delta_{2}=4\kappa^{2}t^{2}+\kappa^{2}(\Delta\omega)^{2}-(\Delta\Gamma)^{2}[(\Delta\omega)^{2}/4]$ with $\Delta\omega=\omega_{1}-\omega_{2}$ . Similar to Eq. (3), Eq. (13) shows the evolution of the EPs and can plot a rich physical picture in the parameter spaces of $(\Delta\omega)^{2}/(4\kappa^{2})$ and $t^{2}/\kappa^{2}$ , as presented in Fig. 3(b). Obviously, the coalescence of the eigenstates occurs under the following three conditions: (1) $\Delta_{1}\pm4\sqrt{\Delta_{2}}=0$ , $\Delta_{1}\neq0$ , and $\Delta_{2}\neq0$ ; (2) $\Delta_{1}\neq0$ and $\Delta_{2}=0$ ; and (3) $\Delta_{1}=\Delta_{2}=0$ . According to the coalescence of the EPs, the whole picture can be divided into five regions. In the domain of class I (depicted in gray), related to the large $\Delta\omega$ , the four coupled cavities can be divided into two subsystems that have independent responses to variable coupling and show two separated trajectories of EPs under non-Hermitian modulations. With decreasing $(\Delta\omega/2\kappa)^{2}$ , the energy spectra of the two subsystems will overlap and generate much richer EP physics. In the domain of class II (depicted in blue), the EPs of the two subsystems can still be identified clearly, while there are two additional EPs determined by the condition $\Delta_{1}-4\sqrt{\Delta_{2}}=0$ . In the domain of class III (depicted in green), the EPs of the two subsystems have the same real part of the eigenfrequency, while there are also two additional EPs determined by the conditions $\Delta_{1}+4\sqrt{\Delta_{2}}=0$ and $\Delta_{1}-4\sqrt{\Delta_{2}}=0$ . In Fig. 3(b), the yellow solid line, separating class II from class I, and the red solid line, separating class II from class III, show the coalescence of two EPs and three EPs. The white dashed line shows the state inversion at degeneracy determined by $4\kappa^{2}=4t^{2}+(\Delta\omega)^{2}$ in the Hermitian region, which divides class II and class III into two parts with different topological properties.

(a)
![](images/579735f4cfdca0ec031b61d670a1acd9308f2e872e5243c2d1c9edeb0f95a2b6.jpg)

(b)
![](images/d333b43b15144ab609a1139e88712ccfafea4f6a0c88cd0c22b071c81549d11f.jpg)

(c)
![](images/7d36d2f9b931d174127b6e59673d64065e4aaf59da4f638d1d73937e7c15f140.jpg)

(d)
![](images/acfb7426621ea99bdbe83df03d67e3abc37c7ab30319141ee6b5e1efbfddae03.jpg)

(e)
![](images/c0d1e41a5ca27fea05d64c72eb901bb84a9b3f53a428da0398b383abeb02b961.jpg)
FIG. 3. Non-Hermitian acoustics in multiple-cavity systems. (a) Four-state system that consists of four coupled acoustic resonators with asymmetric loss modulation used to exhibit multiple EPs. (b) Formation pattern of multiple EPs is mapped in the parameter space. (c) Three-state system that consists of three coupled acoustic resonators with non-Hermitian modulation used to realize the hybrid topological invariant. (d),(e) Eigenvalue Riemann surfaces show the encircling of the exceptional nexus in different parameter spaces. (a),(b) Reprinted with permission. © 2016 APS. (c),(d) Reprinted with permission. © 2020 AAAS.

Additionally, Tang et al. proposed a ternary acoustic cavity system to verify the exceptional nexus with a hybrid topological invariant [21]. Three identical acoustic cavities are connected by small tubes, as shown in Fig. 3(c). With flexible modulations to each cavity, the effective non-Hermitian Hamiltonian of the system can be expressed as

$$
\begin{array}{l} H = (\omega_ {0} - i \gamma_ {0}) \mathbf {I} + \kappa \\ \times \left[ \begin{array}{c c c} \sqrt {2} i (1 + \Lambda) & - 1 & 0 \\ - 1 & i \Xi & - 1 \\ 0 & - 1 & - \sqrt {2} i (1 + \Lambda) \end{array} \right], \end{array}\tag{14}
$$

where $\omega_{0}$ and $\gamma_{0}$ are the resonant frequency and intrinsic loss of each cavity, respectively. $\kappa$ is the strength of coupling between the cavities. I is the identity matrix. $\Xi = \delta_{f} + i\sigma_{A}$ and $\Lambda = \delta_{g} + i\sigma_{B}$ indicate additional tuning imposed on the middle cavity and the other cavities, respectively, where $\sigma_{A}$ and $\sigma_{B}$ represent onsite detuning of the resonant frequencies; $\delta_{f}$ and $\delta_{g}$ represent non-Hermitian modulation. Notably, extra absorptive materials are added to the cavities to introduce a global loss bias for the realization of a gain effect in the experiment.

For a three-state system expressed by this matrix, a third-order EP (EP3) would exist at $(\Xi,\Lambda)=(0,0)$ . By

encircling the EP3 in the eigenvalue Riemann surfaces, Berry phases accumulating on different cyclic paths will indicate distinct winding numbers that are associated with different topological notions. Figures 3(d) and 3(e) show the encircling of the EP3 on the $\Xi$ plane ( $\Lambda = 0$ ) and $\Lambda$ plane ( $\Xi = 0$ ) respectively. Since the EP3 can be regarded as the coalescence of two or more second-order EPs (EP2), the system may provide several routes to encircle the EP3 under different non-Hermitian modulations. In the case of the $\Xi$ plane, it will take three complete cycles to return to the starting point, indicating a winding number of $w_{\Xi} = -2/3$ . However, in the case of the $\Lambda$ plane, there will be two different paths to encircle the EP3. One starts from the upper state, which will take two cycles to return to the starting point. The other one starts from the middle state, which will take just one cycle. Thus, the winding number can be confirmed as $w_{\Lambda} = -1$ for the case of the $\Lambda$ plane. With experimental verification, this ternary acoustic cavity system can provide hybrid topological invariants with distinctive winding numbers under non-Hermitian modulations.

(a)
![](images/831ddf01ed6dcc836bca3ec215ebde2e66ff9b06a5232e4fd6dc021f9f6fbf07.jpg)

(b)
![](images/0edcc3dc298386df996b225e584c5e1ad4a34d27c23895c96495f24e43f7c4c8.jpg)
(g)

(c)
![](images/6e6c1ea94e9393f25283f12dfc2095a1d9fb4046b5a6e3ba570aead948887790.jpg)

![](images/db5016f09d501b4d2c251b338dfd7545ef5b3c036412908fb1f999ae8bb82de1.jpg)

## D. Active approaches to realizing PT acoustics

![](images/bfbad8836b6f791008acb7ea544ad2fa1d4c7cd61cf3bda058d6c9d50d0dbbd3.jpg)

Since the PT acoustics shows the versatile ability of wave manipulation in the one-dimensional (1D) waveguide, many efforts are devoted to study the PT-symmetric structure with balanced gain and loss. Zhu et al. studied the extraordinary scattering characteristics of the acoustic PT medium $[65]$ , as shown in Fig. 4(a). Such a periodic structure with balanced gain (red blocks) and loss (green blocks) can lead to unidirectional reflectionless transmission. Specifically, the waves incident from the left side will pass through unimpeded, while waves incident from the right side will have total transmission and reflection simultaneously. Based on these intriguing properties, they also predicted a non-Hermitian-enabled acoustic cloak with unidirectional invisibilities. Soon after the theoretical work, several experiments were conducted to verify these unexpected wave phenomena $[46,57,58]$ . Figure 4(b) shows the active approach to realizing a PT-invisible acoustic senor by using two acoustic transducers with well-designed non-Foster electrical circuits $[57]$ . One of the transducers is assigned to absorb the incident acoustic waves, which inevitably causes some reflections and scatterings. The other transducer is designed to work as the time-reversed counterpart of the first one, which can mimic the gain medium that amplifies the acoustic waves. This specific combination can let the waves propagate freely in a prescribed direction while producing an additional total reflection in the opposite direction. Another interesting work with an ultrathin PT acoustic cloak $[30]$ is shown in Figs. 4(c) and 4(d). Figure 4(c) illustrates that the metasurface structure consists of lossy Helmholtz resonators (left part) and the loudspeaker array with amplified circuits (right part). The pressure field for waves impinging on the left part is plotted in Fig. 4(d). It clearly shows that incident acoustic waves will be absorbed first, then be reproduced perfectly after passing through the whole metasurface structure. Since the model has symmetric distributions of gain and loss modulation, the cloaking performance retains extremely strong unidirectionality.

![](images/51d3864561d3e4b879975cba3d9282fe1b46641ecf38f06264845192579c819b.jpg)

![](images/6fe0e1f45fac30f4a36ec2f89f67b7910409eaa70eef78afe2b93944ae76c471.jpg)
FIG. 4. PT acoustics with gain and loss modulation. (a) Schematic diagram of acoustic PT-symmetric structure. (b) Invisible acoustic sensor realized by active elements with PT symmetry. (c) Acoustic cloak enabled by the PT-symmetric metasurface. Blue dashed line represents the lossy part, while the red dashed line represents the amplified parts. (d) Acoustic pressure field of the rigid rod cloaked by the PT-symmetric metasurface. (e) Experimental approach to accessing the EP via active acoustic elements. (f) Conceptual diagrams of the incidences from the loss side and gain side. (g) Transmission and reflection coefficients for incidences from the loss side and gain side versus varying spacing between them. (a) Reprinted with permission. © 2014 APS. (b) Reprinted with permission. © 2015 Springer Nature. (c), (d) Reprinted with permission. © 2019 AAAS. (e)–(g) Reprinted with permission. © 2016 Springer Nature.

Figures 4(e)–4(g) show a flexible method to control the EP in an active configuration with balanced gain and loss [58]. As shown in Fig. 4(e), the loss part is sound leakage induced by some slits curved in the waveguide, and the gain part is a pair of coherent loudspeaker arrays that generate directed acoustic waves. Two calibrated unidirectional microphones are mounted on both sides of the sample to measure the transmission and reflection coefficients. First, the scattering field of the loss part alone is tested to calculate the complex refractive index of the loss. Then, the effective refractive index of the gain part can be confirmed, and the loudspeaker arrays are modulated to satisfy the condition of balanced gain and loss. A claim of this work is that two EPs can be obtained by changing the gap between the loss part and gain part. By varying this spacing, the transmission and reflection coefficients for the incidences from both sides are plotted in Fig. 4(g). The dip of the reflection spectrum, related to the EP, can be observed with wave illumination from the loss part and gain part.

## III. WAVE MANIPULATIONS WITH NON-HERMITIAN MODULATION

## A. Non-Hermitian metasurface

A metasurface, which is well known for the characteristics of subwavelength thickness, is a type of artificial structure that shows unexpected abilities in wave-front engineering $[66–69]$ . Most of the previous works on acoustic metasurfaces study the Hermitian domain where loss is ignored $[70–72]$ . However, loss is inevitable in nature for acoustic waves and may harm the performance of acoustic devices. Li et al. proposed a tunable metasurface structure with judiciously designed loss distribution to realize asymmetric transmission [40], as shown in Fig. 5(a). For a lossless metasurface, the transmission properties of the propagation modes can be obtained with the generalized Snell law:

$$
(\sin \theta_ {t} - \sin \theta_ {i}) k _ {0} = \xi + n G,\tag{15}
$$

where $\theta_{t}$ and $\theta_{i}$ are the refraction angle and incidence angle, respectively. $k_{0}$ is the wave number in the free space. $\xi$ denotes the phase gradient of the metasurface. n indicates the diffraction order, and G represents the reciprocal lattice corresponding to the period of units of the metasurface. Due to destructive interference between adjacent units, the diffraction orders associated with the period of the gradient phase can take the values of n = 0 and n = -2 for the positive and negative directions, respectively. Thus, diffraction induced by the periodic structure of the metasurface has an influence on the transmission property in the negative direction, since the term nG equals zero in the positive direction. Acoustic waves traveling in the negative direction will encounter a strong absorption within the units due to the multiple reflection process, while waves traveling in the positive direction will have high-efficiency transmission, since the grating effect is negligible. Pressure fields for the loss and lossless cases are calculated at an incident angle of $25^{\circ}$ . For comparison, the lossless case, shown in the top panel of Fig. 5(b), clearly demonstrates symmetric transmissions from both incident directions. However, in the loss case with an optimal value of 0.14, asymmetric transmission, where waves propagating along the negative direction are absorbed and waves incident from the positive direction have a high transmission, can be observed in the middle panel of Fig. 5(b). For a better understanding, the two dominant propagating modes of the asymmetric transmitted fields are extracted and plotted in the bottom panel of Fig. 5(b), since other higher-order modes have no contribution to the far field. It is obvious that the propagating mode along the negative direction will vanish and leave an empty field. At the same time, the propagating mode along the positive direction has little influence on the transmission. This work opens a non-Hermitian avenue to realize robust wave manipulation with the metasurface structure.

Following this transmission-type acoustic metasurface, Wang et al. extended this concept to realize asymmetric reflection based on a metasurface with non-Hermitian modulation $[41]$ , as shown in Fig. 5(c). By engineering the metasurface with the predesigned phase gradient and period, when the acoustic waves impinge at a specific angle, the diffraction orders of reflection can be restricted to the specular reflection mode and the retroreflection mode. By adding absorptive materials to one unit of the metasurface in a period, non-Hermitian modulation can be introduced into the model. Thus, a two-port system can be constructed to hold an EP analogous to the scattering properties of the 1D waveguide with PT-symmetric potentials that we discuss above. The scattering properties of this two-dimensional (2D) metasurface can be described by

(a)
![](images/c14300cb3f81d664e92e0b58147fee75a6d00af874b51a063d6c8d81f8f887e5.jpg)

![](images/9d6047b1472bfa7d177cbba2356ff26769d3ee4b99843c62120f566ffbe9ff7a.jpg)

![](images/56b11dec92be6c0661655e3300cb5509422ac9e4cb027dbe2a9cc144b41dba05.jpg)
(d)

![](images/4ad637212585c92c61667244e74cc19066ee2b86548116c22d5d80c9a6405595.jpg)
FIG. 5. Non-Hermitian modulation enables asymmetric transmissions and reflections through the metasurface structure. (a) Illustration of tunable asymmetric transmission induced by loss modulations. (b) Calculated acoustic pressure fields of positive and negative transmissions for three cases: gradient-index metasurface without loss modulation, gradient-index metasurface with a loss of 0.14, and ±1 orders of the diffraction mode with a loss of 0.14. (c) Schematic diagram of asymmetric reflection at the EP. (d) Scattering fields for high-efficiency asymmetric reflection with which the wave can be retroreflected from one side while being absorbed perfectly from the opposite side. (a),(b) Reprinted with permission. © 2017 APS. (c),(d) Reprinted with permission. © 2019 APS.

$$
\binom{p _ {f} ^ {R}}{p _ {b} ^ {L}} = S \binom{p _ {f} ^ {L}}{p _ {b} ^ {R}},\tag{16}
$$

where p indicates the complex acoustic pressure. The subscript f (b) represents the forward (backward) propagating waves. The superscript L (R) denotes the left (right) side of the normal direction of the metasurface. By solving the eigenvalues of the scattering matrix, S, we can obtain an EP, where the phenomenon of asymmetric reflection in 2D space can be observed, as depicted in Fig. 5(d). The top panel shows the simulated and measured acoustic fields of the waves that travel from the left side. The acoustic energy is retroreflected along the reverse incident direction, which is similar to total reflection in the 1D waveguide. However, when the wave travels from the right side, as shown in the bottom panel of Fig. 5(d), the incident energy is mostly absorbed, which is similar to the reflectionless property in the 1D waveguide. Notably only loss modulation is considered in this model. Thus, dissipation cannot be compensated for by gain modulation, which leaves low efficiency for the specular reflection mode.

In addition to asymmetric transmission and reflection discussed above, metamaterials with non-Hermitian modulation can achieve other interesting scattering phenomena $[73–75]$ . With the help of the periodic PT-symmetric potentials, the structure can provide a unidirectional wave vector, q, with amplitude $|q| = 2\pi/T$ , where T is the period of the modulation. For the forward incidence, the wave-vector matching condition, $k_{r} = q + k_{i}$ , can be approached when the system works at the EP. Here, $k_{i}$ and $k_{r}$ denote the incident and reflection wave vectors, respectively. However, complex modulation cannot provide any matched wave-vector condition for the backward incidence. Based on this property, Fig. 6(a) shows unidirectional acoustic focusing by extending the unidirectional reflectionless transmission from the 1D waveguide to the 2D cylindrical system [44]. To access the EP in the passive acoustic system, the complex refractive-index distribution is imposed on the structure to meet the requirement that PT-symmetric modulation is even in the real part and odd in the imaginary part. For the incidence from the bottom side, the specular reflection emerges at the interface of the metamaterial structure, which is determined by the matched wave-vector condition. The reflected waves at the concave interface can generate a focused field. For the incidence from the top side, the metamaterial structure cannot fulfill the wave-vector continuous condition and leads to no reflection. Thus, the asymmetric acoustic focusing enabled by the periodic PT-symmetric metamaterial can be observed, in both simulations and experiments, as shown in Fig. 6(a). In the experimental measurement, modulation of the real and imaginary parts of the complex refractive index is achieved by using curved grooves and sound-leakage meshes, respectively.

This unidirectional wave-vector manipulation can also be applied to the oblique incidence situation that realizes the single-sided acoustic beam splitting at the EP [76], as shown in Fig. 6(b). The acoustic wave incident from the left side will split into two parts. One is the specular reflection that relates to unidirectional wave-vector matching, and the other one is the direct transmission that passes through the metamaterial crystal. On the contrary, for incidence from the top side, only the direct transmission exists, since the structure cannot support a real reflection wave vector.

## B. The absorption properties at the EP

Owing to asymmetric wave behavior at the EP, some works related to the extraordinary absorption properties have been proposed theoretically and experimentally. Figure 7(a) shows two Helmholtz resonators with intrinsic losses that are loaded to a waveguide [22]. When the system works at the EP, the two adjacent coherent perfect absorptions (CPA) can be obtained in the parameter space of frequency and loss. In a more complex model with three resonators loaded on the waveguide, a broadened and flat CPA can also be theoretically verified. Moreover, Li et al. presented a compact structure with deep subwavelength thickness to realize asymmetric sound absorption at low frequencies [77]. Two small necks with different lengths are embedded in a cavity to construct a coupled system, as shown in Fig. 7(b). Thus, non-Hermitian modulation can be introduced by adjusting the structural parameters of the neck to induce different intrinsic losses, without the need for absorptive materials. The transmission and reflection coefficients from left and right incidences are plotted in Figs. 7(c) and 7(d) respectively. At the EP, there is an extremely asymmetric reflection. Meanwhile, transmissions from both sides are nearly identical and retain a low value. Underlying these scattering properties, asymmetric absorption can be deduced, according to the conservation of energy. Compared with the single-port system that absorbs acoustic waves from one direction, this two-port scheme provides not only the asymmetric scattering properties but also the functionality of ventilation.

![](images/0c38d4c8b0c89ab45a79f00e41ab296fc6d58a9b487dfa0b0d2d6c02896f43d5.jpg)

(b)
![](images/5156ad002a3b527731f3290b461173aeacbf166e5f1a58ef8650205f32c19969.jpg)
FIG. 6. Non-Hermitian modulation enables unidirectional wave-vector manipulation. (a) Unidirectional acoustic focusing by using the passive PT-symmetric metamaterial structure. Left and right parts are simulated and measured fields, respectively. Acoustic wave incident from the bottom is reflected and generates a focused field. Acoustic wave incident from the top side is absorbed due to dissipation in the PT-symmetric potentials. (b) Single-sided acoustic beam splitter under balanced gain and loss modulation. Acoustic wave incident from the left (black solid line) will be split into two beams (red and green solid lines). For the wave incident from the top side, there will be total transmission without wave-front distortion. (a) Reprinted with permission. © 2018 APS. (b) Reprinted with permission. © 2020 APS.

Apart from the work discussed in Fig. 7(b), there is another approach to realize asymmetric absorption by obtaining the topological interface state and EP simultaneously [49], as shown in Fig. 7(e). Inspired by the Aubry-Andre-Harper model, comblike acoustic resonators with additional embedded loss are used to construct topologically protected unidirectional reflectionless propagation, which is a signature of the EP. Notably, the nontrivial topology of the model is established in a synthetic dimension of the structural parameters. Since the transmission can be restricted with a long chain, unidirectional reflectionless propagation will lead to a nearly asymmetric absorption. The measured absorption coefficients for both incident directions are plotted in Fig. 7(f). Waves propagating from one side can have total absorption around the EP, while waves traveling from the opposite side can hardly be absorbed, indicating a strong reflection.

## C. Other interesting applications

Conventionally, the disorder scatterer is always considered to be an obstacle to wave propagation, which leads to huge reflection and wave-front distortion. However, Rivet et al. proposed a non-Hermitian router to achieve a constant pressure field in the disordered system, where the amplitude of the pressure is fixed while the phase may vary [78]. In Fig. 8(a), a comparison of the absolute pressure in a disordered system with and without non-Hermitian modulation is plotted. Here, disorder is introduced via variations of the mass density along the propagation direction, as depicted in gray. When the acoustic wave travels from the left side in the Hermitian case, most of the acoustic energy is reflected and leaves a low transmission. Also, the intensity is nonuniform in the domain with varied density. On the contrary, the intensity can remain constant in the same system under non-Hermitian modulation. This intriguing property can be investigated by assuming that the pressure field in the system has a constant amplitude, $p_{j+1} = e^{ik\varphi_j}p_j$ , with $j$ denoting a series of checkpoints

![](images/583f44913f74b71f8e047e1efde469a0dc2b9690f52804e928644c86de26c0cd.jpg)

![](images/2f1df4517380d6a1051469902d189061e4e50106da607d3b2dde00ed9923a044.jpg)

![](images/7ae19782d5d2cbecf1cddf69ce9ee2f19877bccdd6250a9901c2831f7d7ab1e5.jpg)

![](images/729ed998a71880ea049570fbc0a8d84fd8feed32412036b7dfa4b8b06d4045da.jpg)

(e)
![](images/2f64551c7decebfd743f35f720542899f078e02f38d8f1e99646768b952dbe07.jpg)

(f)
![](images/468648f8c70693acd4398f456ffc097c1a65ef9f2d432285862cde387c3a7d29.jpg)
FIG. 7. Asymmetric absorption around the EP. (a) Two Helmholtz resonators are loaded on the waveguide. (b) Asymmetric absorption via two small tubes embedded in the square cavity. Transmission and reflection coefficients of the sample shown in (b) for left incidence (c) and right incidence (d). (e) Experimental setup of coupled resonators. (f) Asymmetric absorption of the sample shown in (e) versus frequency for both incidence directions. (a) Reprinted with permission. © 2017 APS. (b)—(d) Reprinted with permission. © 2021 Springer. (e), (f) Reprinted with permission. © 2018 APS.

![](images/36cca676b43ac9333796b6375773eb47a0fc11c1fd3c8539d3287072abd761a3.jpg)

![](images/fada8cafbb0bb33330e61a0d37d81f98b21ca336de48061a63397fd3450c3e60.jpg)

(c)
![](images/20de6d54a72ea8cf1b83bb6753d71fb99bbeaf8ace842e1b3fa6f4dcb2946f68.jpg)

(d)
![](images/b399d46f717f02b9a90c18a5d2f23888df34624a25f5de5138a7ba9c52a2048d.jpg)

(e)
![](images/fb808ee72e590a07eba01dadf3d1e030107112f02e65871a320f669ac11ef7a8.jpg)
FIG. 8. (a) Continuous-pressure fields of the disordered system with and without non-Hermitian modulation. (b) Discretized model of the disordered system with and without non-Hermitian modulation. (c) Picture of the passive PT-symmetric ring cavity. (d) Amplitude of pressure in the cavity with and without non-Hermitian modulation. (e) Phase of pressure in the cavity with and without non-Hermitian modulation. (a),(b) Reprinted with permission. © 2018 Springer Nature. (c)–(e) Reprinted with permission. © 2020 Springer Nature.

along the propagation direction. By introducing a transfer matrix, $M_{j} = [A_{j}, B_{j}; C_{j}, D_{j}]$ , that connects adjacent checkpoints, the requirement for acoustic impedance for the constant pressure field can be expressed as

$$
Z _ {j} = \frac {p _ {j} - A _ {j} p _ {j + 1} - B _ {j} q _ {j + 1}}{C _ {j} p _ {j + 1} + D _ {j} q _ {j + 1}},\tag{17}
$$

where q is the volume flow. For the description mentioned in Ref. [71], the imaginary part of $Z_{j}$ represents spatially varying scattering caused by disorder, and the real part of $Z_{j}$ represents non-Hermitian modulation that can compensate for the scattering part. The physical realization of this model can be constructed via an acoustic waveguide that contains several irregular inclusions, such as resonators, hard spheres, and labyrinth channels. In the top panel of Fig. 8(b), the simulated pressure field clearly shows that a strong reflection is generated in the waveguide that contains various scattering obstacles in the Hermitian regime. However, if the transfer matrix, $M_{j}$ , is deduced by the scattering properties of the obstacles, the required acoustic impedance can be determined according to Eq. (17). Thus, non-Hermitian modification, which could further be implemented by the active elements in the experiment, can be applied to this waveguide with many defects to preserve the constant-amplitude pressure profile, as shown in the bottom panel of Fig. 8(b).

Another interesting application is an acoustic emitter with a reversed eigenmode $[79]$ . The constructed structure is shown in Fig. 8(c). For a normal ring cavity, the pressure field excited by a single source can be considered as the superposition of two coupled eigenstates, associated with the clockwise (CW) mode and counterclockwise (CCW) mode. Under non-Hermitian modulation, the two eigenmodes of the ring cavity will degenerate to the CW mode or CCW mode purely because the effective non-Hermitian Hamiltonian becomes a defective matrix with a coalesced eigenstate $[80]$ . The other eigenstate can be regarded as a missing dimension. However, by introducing a single source to interact with PT-symmetric potentials, the pressure field coupled to the missing dimension can be excited in the cavity. The explicit transfer relation of the CW mode and CCW mode can be expressed as

$$
\frac {a _ {\mathrm{CCW}}}{a _ {\mathrm{CW}}} = 1 - \frac {\chi_ {b a} (\varphi_ {0})}{i \Delta - \gamma_ {\mathrm{tot}}},\tag{18}
$$

where $a_{CCW}$ and $a_{CW}$ denote the complex amplitudes of the CCW mode and CW mode, respectively. $\Delta$ is the detuning parameter that deviates from the cavity's resonant frequency. $\chi_{ba}(\varphi_{0})$ is the position-dependent coupling strength, with $\varphi_{0}$ being the azimuthal angle of the ring cavity. $\gamma_{tot}$ is the total decay rate induced by non-Hermitian modulation. At the resonant frequency where $\Delta = 0$ , this relationship indicates that the transfer ratio may vanish at $\chi_{ba}(\varphi_{0}) = -\gamma_{\mathrm{tot}}$ , when the source is placed at a specific position and a pure CW mode can be obtained that is related to the missing dimension. This chirality-reversal phenomenon can be verified by the pressure field in the ring cavity, as plotted in Figs. 8(d) and 8(e). Compared with the standing-wave field excited in the normal cavity, the amplitude of the field in the cavity with non-Hermitian modulation can retain a quasi-constant profile. The evolution of the phase also indicates a propagating mode that increases with the azimuthal angle.

## IV. NON-HERMITIAN TOPOLOGICAL INSULATOR

## A. Topological insulator with non-Hermitian modulation

Topological phases discovered in quantum systems have triggered extensive explorations of various platforms, from photonics, acoustics, and mechanics to electrics $[81–86]$ . These systems are commonly assumed to be Hermitian, despite non-Hermiticity naturally existing in a wide range of systems. Recently, non-Hermitian Hamiltonians have attracted growing attention and are incorporated into Hermitian systems, which enable rich and unique topological features with notably different bulk-boundary correspondence from the Hermitian cases $[61,87–94]$ .

The second-order topological insulator can be observed in an acoustic platform with square sonic crystals that contain four fluidic cylinders embedded in air $[55]$ , as shown in Fig. 9(a). By expanding or shrinking these concentric cylinders, the topological properties of the structure can be adjusted from the trivial phase to the nontrivial phase. In the Hermitian regime, the bulk-boundary correspondence, known as topological corner states as well as edge states, can be obtained in the nontrivial band gaps of the real part of the eigenfrequency, while the imaginary part of the eigenfrequency of these confined states remains zero, since no gain or loss is considered in the system. By adding complex values to the effective-density parameters of cylinders, non-Hermitian modulation can be imposed on the system in two specific arrangements, where gain atoms and loss atoms are placed in parallel or diagonally. Under non-Hermitian modulation, these real parts of the frequency of excited confined eigenstates remain identical to those in the Hermitian case, whereas the imaginary part may diverge into the gain branch and loss branch, which refers to the sourcelike and sink states. Moreover,

Fig. 9(b) clearly shows that these localized eigenstates preserve the horizontal and vertical profiles based on non-Hermitian modulation with a parallel arrangement. For the diagonal arrangement, the localized eigenstates preserve the diagonal profiles accordingly.

Another straightforward example is the non-Hermitian valley Hall topological insulator in acoustic crystals $[95]$ . The acoustic realization of valley Hall transport is verified theoretically and experimentally in Hermitian cases. The basic unit, with a hexagonal lattice structure, as shown in Fig. 9(c), can be employed to support the valley states $K_{1}$ and $K_{2}$ , which are located at the extrema of the nontrivial band gaps. The gray and white circles represent the fluidic cylinders under non-Hermitian modulations with the imaginary part of mass density, $-\gamma i$ and $n\gamma i$ , respectively. n = 1.4 is used to compensate for the difference between the radii of these two cylinders to obtain approximately balanced gain and loss. By changing the sign of $\gamma$ , gain or loss modulation imposed on the cylinders can be exchanged. Figure 9(d) shows the amplitude and phase fields of $K_{1}$ and $K_{2}$ states that indicate the acoustic valley vortex with the topological charges of +1 and -1, respectively. In a bulk structure with a triangular shape, as shown in Fig. 9(e), a point source with different chirality can be used to excite the $K_{1}$ and $K_{2}$ states. Depending on gain or loss modulation (the sign of $\gamma$ ) in the system, the valley states $K_{1}$ and $K_{2}$ can be tuned to present the attenuating or amplifying states, compared with that in the Hermitian case.

## B. Topological insulator induced by non-Hermiticity alone

Very recently, a few experimental works have been conducted to fabricate acoustic non-Hermitian topological insulators. More significantly, the topological phase transitions are induced only by non-Hermiticity $[29]$ , instead of conventional structural parameters. Figure 10(a) is a photograph of a 1D non-Hermitian topological system, which is composed of 12 coupled resonators. The basic unit cell contains four rectangular resonators connected with thin waveguides. Additional loss is introduced by drilling some air holes in the resonator and then inserting some sound-absorbing materials. The topological phase of the chain system is defined by the biorthogonal polarization, p, based on the non-Abelian Berry connection. If no additional loss is considered, the system is trivial (p = 0). When additional loss is applied to the middle two sites of each unit cell, a topological band gap will be opened and a topological phase transition can be obtained (p = 0.5). The nontrivial topology can be confirmed from the measured dispersions. In the bulk dispersions, as shown in Fig. 10(b), two peaks can be observed, revealing the existence of the band gap. But in the edge response, as shown in Fig. 10(c), a peak appears in the band gap, which corresponds to the topological edge state.

![](images/0a257e6786cafcceabed641f0449a295d269bc9a0501c91a2af4f7685f1b2456.jpg)

(b)
![](images/4966f1f2dbb4472dc46da3831e9574c1151d1174f11d0045ddf4fd6cbbc5041b.jpg)

![](images/e50cef046a569b9de664d9d6e19367163bcf94cbf897664fc5bae34ae39c0289.jpg)

(d)
![](images/9969c346a2b59bd799f52f180e4cffbe5ec21cbb1335e5d90067788957b8997e.jpg)

![](images/8d33949793c3799baf2d7fea4b5a324c0759560ed1da50c7d9041926e5c6c4c8.jpg)
FIG. 9. Acoustic topological insulator with non-Hermitian modulation. (a) Acoustic second-order topological insulator composed of the cylinders embedded in the background medium. Each unit contains four cylinders, and by expanding or shrinking these four cylinders, the topological property of the sample can be adjusted. Green and yellow denote the cylinders with gain and loss modulation respectively. (b) Simulated eigenmodes of edge states (the upper panel) and corner states (the lower panel). (c) Acoustic valley topological insulator composed of a hexagonal lattice. Gray and white circles represent the cylinders with gain or loss modulations respectively. (d) Amplitude and phase fields for the valley states. (e) Intensity fields of valley states excited by a chiral source under the non-Hermitian modulation. (a),(b) Reprinted with permission. © 2019 APS. (c)–(e) Reprinted with permission. © 2018 APS.

Moreover, non-Hermiticity can play a significant role in higher-order topological insulators $[56]$ . Similarly to the 1D non-Hermitian topological system, by carefully designing the loss configuration, a 2D non-Hermitian topological insulator is fabricated, as depicted in Fig. 10(d). This lattice contains $3 \times 3$ unit cells, the basic unit cell of which is composed of 16 resonators coupled with identical waveguides. With the introduced loss distribution in Fig. 10(d) (the sites with sound-absorbing materials can introduce additional loss), the lattice transitions from a trivial gapless phase to a nontrivial gapped phase, which is a topological quadrupole. Owing to the nontrivial quadrupole moment, the in-gap corner states are observed in the sample, as shown in Fig. 10(e), in which the acoustic energy in the four corners is much higher than other positions. Also, in the topological edge states plotted in Fig. 10(f), the intensity along the edge of the sample is relatively higher than the bulk. This non-Hermiticity-induced topological phase can be extended to design other types of topological insulators in various platforms, such as photonics, electronics, and mechanics.

## C. Non-Hermitian skin effect in acoustics

The bulk-boundary correspondence induced by non-Hermiticity has attracted increasing research interest in many fields of physics $[87]$ . Recent theoretical works have predicted that there exists a unique feature, where the intensities of all eigenstates are localized at the boundaries, also known as the non-Hermitian skin effect $[96–100]$ . This intriguing wave behavior can be realized by a typical tight-binding model, i.e., one-dimensional Su-Schrieffer-Heeger model with non-Hermitian modulation in the coupling $[101]$ . The basic unit with nonreciprocal coupling is shown in Fig. 11(a), which has two identical acoustic cavities with resonant frequency $\omega_{0}$ . The effective non-Hermitian Hamiltonian of this model can be described as

$$
H (k) = \left( \begin{array}{c c} 0 & \kappa_ {1} + \gamma + \kappa_ {2} e ^ {- i k} \\ \kappa_ {1} - \gamma + \kappa_ {2} e ^ {i k} & 0 \end{array} \right),\tag{19}
$$

where $\kappa_{1}$ and $\kappa_{2}$ are the intracell and intercell couplings, respectively. $\gamma$ is the non-Hermitian component that determines gain or loss modulation based on its sign. Consider a finite-chain containing 10 units. By solving the eigenenergy of the system under open-boundary conditions, Fig. 11(b) clearly shows that all the bulk eigenstates are localized at one edge boundary. When $\gamma > 0$ , the eigenenergies of all modes will be confined at the right side [red line in Fig. 11(b)]. On the contrary, the eigenenergies will be localized at the left side [blue line in Fig. 11(b)].

(a)
![](images/424e626efc1e56c932de8d5f20f5f1ad6b51094a9801304bd9cf335d0e22d56c.jpg)

(b)
![](images/053b81444d19bbf1f2ad4c0664c50eda2b85f9c7112adc6b9665bd997b8ca6da.jpg)

(c)
![](images/dcd3d378f5df40167405cf3bb4e3b6355403a6fda213defc1861c841d7e346fa.jpg)

(d)
![](images/48e53d7f8f9e161f91bc0f318b9e39facdfd94e8f857dabb88dc4b328f864149.jpg)

(e)
![](images/53993db3931a6643ec7f230766ba99c5b352cfd5d78302a0aca1055d6b230b33.jpg)

![](images/f459ac7afd7d42650e26240057f34469c1574211484008462ca9111baf50b44f.jpg)
FIG. 10. Non-Hermiticity-induced topological insulators. (a) 1D non-Hermitian topological insulator coupled with acoustic resonators. Simulated and measured spectra in one of the (b) bulk and (c) edge resonators. (d) 2D non-Hermitian topological insulator coupled with acoustic resonators. Measured intensity profile for (e) corner mode and (f) edge mode. (a) –(c) Reprinted with permission. © 2020 APS. (d)–(f) Reprinted with permission. © 2021 Springer Nature.

Recently, some experimental works have been conducted to demonstrate the non-Hermitian skin effect in the acoustic platform $[102,103]$ . Zhang et al. proposed a scheme based on coupled acoustic resonators $[103]$ . By utilizing the active elements to mimic the gain effect, bipolar localization can be observed, which indicates a topological notion of twisted winding with oppositely oriented loops. Another approach to realize the two-dimensional non-Hermitian skin effect is based on coupled ring cavities in a passive way $[102]$ . With a judicious modification to decouple the spin-up and spin-down states of the system, additional loss can be imposed to achieve spin-dependent wave propagation in the structure.

## V. OUTLOOK AND CONCLUSION

Recent advances in non-Hermitian acoustics already show the flexibility of wave manipulations in many aspects, as we discuss above. Ongoing research into non-Hermitian physics can provide continuous intellectual support for the study of non-Hermitian acoustics. In particular, parallel to the development of non-Hermitian optics, we envision that hot topics inspired by this research may emerge in the following directions.

![](images/4700914060ac9f44102e9be691c0bc871fcf512b71c58e7f5edc86e8bf79becf.jpg)

![](images/b6091c0096c98a8c0cb2434ec22b2f9565ada6d65f498fde067ab683bc4a905d.jpg)
FIG. 11. Non-Hermitian skin effect in acoustics. (a) Sketch of the basic unit with nonreciprocal coupling. (b) Intensity distribution of the eigenstates of the lattice structure with 20 sites.

## A. Nonlinear non-Hermitian acoustics

The interplay between nonlinearity and non-Hermiticity may overcome many unconventional wave phenomena $[104]$ . Typically, the transmissions for both incidence directions of one-dimensional scattering systems are identical, which is guaranteed by the reciprocal principle, despite the system being located in the PT-symmetric phase or broken phase. By increasing the input to bring the system into the nonlinear regime, strong nonreciprocity enabled by enhanced nonlinearity can be observed at the PT-broken phase $[105]$ . Additionally, with unique properties at the saturation point, tunable asymmetric or nonreciprocal transmission can be realized within the PT-symmetric system with nonlinear parameters $[106]$ . Lately, a local nonlinear medium has been employed to provide a feasible tool to manipulate the global topological and non-Hermitian properties of the whole sample $[107]$ . Indeed, the acoustic nonlinear effect is small among natural materials. However, by using active nonlinear elements $[108]$ , nonlinear non-Hermitian acoustics can be a versatile platform for a variety of wave-matter interactions $[57,109–111]$ .

## B. Anti-PT symmetric acoustics

Recently, anti-PT symmetric physics has attracted considerable attention, from quantum systems to wave systems $[112–116]$ and diffusion systems $[117,118]$ . Different from the PT-symmetric configuration that holds the complex potential $V(-x)=V^{*}(x)$ , the counterpart of the anti-PT-symmetric configuration satisfies $V(-x)=-V^{*}(x)$ and has imaginary couplings between connected resonators. Accordingly, the EPs mediated in the anti-PT symmetric system can exhibit different features of the eigenfrequency gradually splitting in the real part and merging in the imaginary part upon changing the detuning parameters. Such a system is of great significance in acoustics and could be applied to realize a flat total transmission band $[119]$ , ultrasensitive sensing $[13]$ , and constant refraction $[115]$ .

To conclude, we discuss the fundamental principles of non-Hermitian acoustics and review recent advances in its applications for many aspects of acoustic manipulations. Benefitting from the development of non-Hermitian physics, acoustic metamaterials with non-Hermitian modulation can excite various extraordinary wave phenomena, such as high-order EPs, asymmetric scattering, and acoustic vortexes. In particular, with the help of active elements, an exact PT-symmetric system with suitable gain effect can be implemented, which could lead to robust non-Hermitian properties. Moreover, we discuss studies on non-Hermitian topological insulators within the acoustic platform. Finally, we expect that non-Hermitian acoustics may show its vigor and vitality with the combination of nonlinearity and anti-PT symmetry.

## ACKNOWLEDGMENTS

J.Z. acknowledges support from the Research Grants Council of Hong Kong SAR (Grants No. C6013-18G and No. AoE/P-502/20) and the National Natural Science Foundation of China (Grant No. 11774297). X.F.Z. acknowledges support from the National Natural Science Foundation of China (Grants No. 11674119 and No. 11690032) and the Fundamental Research Funds for the Central Universities, China (HUST, Grant No. 2019JYCXJJ038).

[1] G. Gamow, Zur quantentheorie des atomkernes, Z. Phys. 51, 204 (1928).

[2] L. Feng, R. El-Ganainy, and L. Ge, Non-Hermitian photonics based on parity-time symmetry, Nat. Photonics 11, 752 (2017).

[3] R. El-Ganainy, K. G. Makris, M. Khajavikhan, Z. H. Musslimani, S. Rotter, and D. N. Christodoulides, Non-Hermitian physics and PT symmetry, Nat. Phys. 14, 11 (2018).

[4] R. El-Ganainy, M. Khajavikhan, D. N. Christodoulides, and S. K. Ozdemir, The Dawn of non-Hermitian optics, Commun. Phys. 2, 37 (2019).

[5] S. K. Gupta, Y. Zou, X. Y. Zhu, M. H. Lu, L. J. Zhang, X. P. Liu, and Y. F. Chen, Parity-time symmetry in non-Hermitian complex optical media, Adv. Mater. 32, e1903639 (2020).

[6] S. K. Ozdemir, S. Rotter, F. Nori, and L. Yang, Parity-time symmetry and exceptional points in photonics, Nat. Mater. 18, 783 (2019).

[7] N. Moiseyev, Non-Hermitian Quantum Mechanics (Cambridge University Press, Cambridge, 2011).

[8] A. Krasnok and A. Alù, Parity-Time Symmetry and Exceptional points: A Tutorial, arXiv:2103.08135.

[9] Z. Lin, H. Ramezani, T. Eichelkraut, T. Kottos, H. Cao, and D. N. Christodoulides, Unidirectional Invisibility Induced by PT-Symmetric Periodic Structures, Phys. Rev. Lett. 106, 213901 (2011).

[10] H. Li, A. Mekawy, A. Krasnok, and A. Alu, Virtual Parity-Time Symmetry, Phys. Rev. Lett. 124, 193901 (2020).

[11] W. Chen, S. Kaya Ozdemir, G. Zhao, J. Wiersig, and L. Yang, Exceptional points enhance sensing in an optical microcavity, Nature 548, 192 (2017).

[12] Q. Zhong, J. Ren, M. Khajavikhan, D. N. Christodoulides, S. K. Ozdemir, and R. El-Ganainy, Sensing With Exceptional Surfaces in Order to Combine Sensitivity with Robustness, Phys. Rev. Lett. 122, 153902 (2019).

[13] J. M. P. Nair, D. Mukhopadhyay, and G. S. Agarwal, Enhanced Sensing of Weak Anharmonicities Through Coherences in Dissipatively Coupled Anti-PT Symmetric Systems, Phys. Rev. Lett. 126, 180401 (2021).

[14] K. Pichler, M. Kuhmayer, J. Bohm, A. Brandstotter, P. Ambichl, U. Kuhl, and S. Rotter, Random anti-lasing through coherent perfect absorption in a disordered medium, Nature 567, 351 (2019).

[15] S. Longhi, PT-symmetric laser absorber, Phys. Rev. A 82, 031801(R) (2010).

[16] W. R. Sweeney, C. W. Hsu, S. Rotter, and A. D. Stone, Perfectly Absorbing Exceptional Points and Chiral Absorbers, Phys. Rev. Lett. 122, 093901 (2019).

[17] J. Doppler, A. A. Mailybaev, J. Böhm, U. Kuhl, A. Girschikm, F. Libisch, T. J. Milburn, P. Rabl, N. Moiseyev, and S. Rotter, Dynamically encircling an exceptional point for asymmetric mode switching, Nature 537, 76 (2016).

[18] A. Li, J. Dong, J. Wang, Z. Cheng, J. S. Ho, D. Zhang, J. Wen, X.-L. Zhang, C. T. Chan, A. Alù, C.-W. Qiu, and L. Chen, Hamiltonian Hopping for Efficient Chiral Mode Switching in Encircling Exceptional Points, Phys. Rev. Lett. 125, 187403 (2020).

[19] X. L. Zhang, T. Jiang, and C. T. Chan, Dynamically encircling an exceptional point in anti-parity-time symmetric systems: Asymmetric mode switching for symmetry-broken modes, Light Sci. Appl. 8, 88 (2019).

[20] Q. Liu, S. Li, B. Wang, S. Ke, C. Qin, K. Wang, W. Liu, D. Gao, P. Berini, and P. Lu, Efficient Mode Transfer on a Compact Silicon Chip by Encircling Moving Exceptional Points, Phys. Rev. Lett. 124, 153903 (2020).

[21] W. Tang, X. Jiang, K. Ding, Y. X. Xiao, Z. Q. Zhang, C. T. Chan, and G. Ma, Exceptional nexus with a hybrid topological invariant, Science 370, 1077 (2020).

[22] V. Achilleos, G. Theocharis, O. Richoux, and V. Pagneux, Non-Hermitian acoustic metamaterials: Role of exceptional points in sound absorption, Phys. Rev. B 95, 144303 (2017).

[23] N. J. R. K. Gerard and Y. Jing, Loss in acoustic metasurfaces: A blessing in disguise, MRS Commun. 10, 32 (2020).

[24] Y. Auregan and V. Pagneux, PT-Symmetric Scattering in Flow Duct Acoustics, Phys. Rev. Lett. 118, 174301 (2017).

[25] R. Fleury, D. L. Sounas, and A. Alu, Parity-Time symmetry in acoustics: Theory, devices, and potential applications, IEEE J. Sel. Top. Quantum Electron 22, 121 (2016).

[26] S. Puri, J. Ferdous, A. Shakeri, A. Basiri, M. Dubois, and H. Ramezani, Tunable Non-Hermitian Acoustic Filter, Phys. Rev. Appl. 16, 014012 (2021).

[27] D. N. Maksimov, A. F. Sadreev, A. A. Lyapina, and A. S. Pilipchuk, Coupled mode theory for acoustic resonators, Wave Motion 56, 52 (2015).

[28] K. Ding, G. Ma, M. Xiao, Z. Q. Zhang, and C. T. Chan, Emergence, Coalescence, and Topological Properties of Multiple Exceptional Points and Their Experimental Realization, Phys. Rev. X 6, 021007 (2016).

[29] H. Gao, H. R. Xue, Q. Wang, Z. M. Gu, T. Liu, J. Zhu, and B. L. Zhang, Observation of topological edge states induced solely by non-hermiticity in an acoustic crystal, Phys. Rev. B 101, 180303(R) (2020).

[30] H. X. Li, M. Rosendo-Lopez, Y. F. Zhu, X. D. Fan, D. Torrent, B. Liang, J. C. Cheng, and J. Christensen, Ultrathin acoustic parity-time symmetric metasurface cloak, Research 2019, 8345683 (2019).

[31] R. Thevamaran, R. M. Branscomb, E. Makri, P. Anzel, D. Christodoulides, T. Kottos, and E. L. Thomas, Asymmetric acoustic energy transport in non-Hermitian metamaterials, J. Acoust. Soc. Am. 146, 863 (2019).

[32] Y. Yang, H. Jia, S. Wang, P. Zhang, and J. Yang, Diffraction control in a non-Hermitian acoustic grating, Appl. Phys. Lett. 116, 213501 (2020).

[33] S. R. Craig, P. J. Welch, and C. Shi, Non-Hermitian complementary acoustic metamaterials for lossy barriers, Appl. Phys. Lett. 115, 051903 (2019).

[34] H. Long, Y. Cheng, J. Tao, and X. Liu, Perfect absorption of low-frequency sound waves by critically coupled subwavelength resonant system, Appl. Phys. Lett. 110, 023502 (2017).

[35] T. Magariyachi, H. Arias Casals, R. Herrero, M. Botey, and K. Staliunas, PT-symmetric Helmholtz resonator dipoles for sound directivity, Phys. Rev. B 103, 094201 (2021).

[36] M. Rosendo-López, A. Merkel, and J. Christensen, PT symmetric sonic crystals: From asymmetric echoes to supersonic speeds, Europhys. Lett. 124, 34001 (2018).

[37] H. Yang, X. Zhang, D. Zhao, Y. Liu, J. Guo, Y. Yao, and F. Wu, The self-collimation effect induced by non-Hermitian acoustic systems, Appl. Phys. Lett. 114, 133503 (2019).

[38] L. Cao, Z. Yang, Y. Xu, S.-W. Fan, Y. Zhu, Z. Chen, Y. Li, and B. Assouar, Flexural wave absorption by lossy gradient elastic metasurface, J. Mech. Phys. Solids 143, 104052 (2020).

[39] Z. Hou, H. Ni, and B. Assouar, PT-Symmetry for Elastic Negative Refraction, Phys. Rev. Appl. 10, 044071 (2018).

[40] Y. Li, C. Shen, Y. Xie, J. Li, W. Wang, S. A. Cummer, and Y. Jing, Tunable Asymmetric Transmission via Lossy Acoustic Metasurfaces, Phys. Rev. Lett. 119, 035501 (2017).

[41] X. Wang, X. Fang, D. Mao, Y. Jing, and Y. Li, Extremely Asymmetrical Acoustic Metasurface Mirror at the Exceptional Point, Phys. Rev. Lett. 123, 214302 (2019).

[42] Y. Yang, H. Jia, Y. Bi, H. Zhao, and J. Yang, Experimental Demonstration of an Acoustic Asymmetric Diffraction Grating Based on Passive Parity-Time-Symmetric Medium, Phys. Rev. Appl. 12, 034040 (2019).

[43] T. T. Koutserimpas, E. Rivet, H. Lissek, and R. Fleury, Active Acoustic Resonators With Reconfigurable Resonance Frequency, Absorption, and Bandwidth, Phys. Rev. Appl. 12, 054064 (2019).

[44] T. Liu, X. Zhu, F. Chen, S. Liang, and J. Zhu, Unidirectional Wave Vector Manipulation in Two-Dimensional Space with an All Passive Acoustic Parity-Time-Symmetric Metamaterials Crystal, Phys. Rev. Lett. 120, 124502 (2018).

[45] F. Ju, Y. Tian, Y. Cheng, and X. Liu, Asymmetric acoustic transmission with a lossy gradient-index metasurface, Appl. Phys. Lett. 113, 121901 (2018).

[46] J. Christensen, M. Willatzen, V. R. Velasco, and M. H. Lu, Parity-Time Synthetic Phononic Media, Phys. Rev. Lett. 116, 207601 (2016).

[47] H. Long, Y. Cheng, and X. Liu, Asymmetric absorber with multiband and broadband for low-frequency sound, Appl. Phys. Lett. 111, 143502 (2017).

[48] C. Shen, J. Li, X. Peng, and S. A. Cummer, Synthetic exceptional points and unidirectional zero reflection in non-Hermitian acoustic systems, Phys. Rev. Mater. 2, 125203 (2018).

[49] W. Zhu, X. Fang, D. Li, Y. Sun, Y. Li, Y. Jing, and H. Chen, Simultaneous Observation of a Topological Edge State and Exceptional Point in an Open and Non-Hermitian Acoustic System, Phys. Rev. Lett. 121, 124501 (2018).

[50] H. Long, Y. Cheng, T. Zhang, and X. Liu, Wide-angle asymmetric acoustic absorber based on one-dimensional lossy bragg stacks, J. Acoust. Soc. Am. 142, EL69 (2017).

[51] A. Merkel, V. Romero-Garcia, J. P. Groby, J. Li, and J. Christensen, Unidirectional zero sonic reflection in passive PT-symmetric willis media, Phys. Rev. B 98, 201102(R) (2018).

[52] S. An, T. Liu, S. Liang, H. Gao, Z. Gu, and J. Zhu, Unidirectional invisibility of an acoustic multilayered medium with parity-time-symmetric impedance modulation, J. Appl. Phys. 129, 175106 (2021).

[53] Z. Gu, H. Gao, T. Liu, S. Liang, S. An, Y. Li, and J. Zhu, Topologically Protected Exceptional Point With Local Non-Hermitian Modulation in an Acoustic Crystal, Phys. Rev. Appl. 15, 014025 (2021).

[54] M. Rosendo López, Z. Zhang, D. Torrent, and J. Christensen, Multiple scattering theory of non-Hermitian sonic second-order topological insulators, Commun. Phys. 2, 132 (2019).

[55] Z. Zhang, M. Rosendo Lopez, Y. Cheng, X. Liu, and J. Christensen, Non-Hermitian Sonic Second-Order Topological Insulator, Phys. Rev. Lett. 122, 195501 (2019).

[56] H. Gao, H. Xue, Z. Gu, T. Liu, J. Zhu, and B. Zhang, Non-Hermitian route to higher-order topology in an acoustic crystal, Nat. Commun. 12, 1888 (2021).

[57] R. Fleury, D. Sounas, and A. Alu, An invisible acoustic sensor based on parity-time symmetry, Nat. Commun. 6, 5905 (2015).

[58] C. Shi, M. Dubois, Y. Chen, L. Cheng, H. Ramezani, Y. Wang, and X. Zhang, Accessing the exceptional points of parity-time symmetric acoustics, Nat. Commun. 7, 11110 (2016).

[59] S. Huang, T. Liu, Z. Zhou, X. Wang, J. Zhu, and Y. Li, Extreme Sound Confinement From Quasibound States in the Continuum, Phys. Rev. Appl. 14, 021001 (2020).

[60] L. Ge, Y. D. Chong, and A. D. Stone, Conservation relations and anisotropic transmission resonances in one-dimensional PT-symmetric photonic heterostructures, Phys. Rev. A 85, 023802 (2012).

[61] E. J. Bergholtz, J. C. Budich, and F. K. Kunst, Exceptional topology of non-Hermitian systems, Rev. Mod. Phys. 93, 015005 (2021).

[62] M. A. Miri and A. Alu, Exceptional points in optics and photonics, Science 363, eaar7709 (2019).

[63] Q. Zhong, J. Kou, S. K. Ozdemir, and R. El-Ganainy, Hierarchical Construction of Higher-Order Exceptional Points, Phys. Rev. Lett. 125, 203602 (2020).

[64] K. Ding, G. Ma, Z. Q. Zhang, and C. T. Chan, Experimental Demonstration of an Anisotropic Exceptional Point, Phys. Rev. Lett. 121, 085702 (2018).

[65] X. F. Zhu, H. Ramezani, C. Z. Shi, J. Zhu, and X. Zhang, PT-Symmetric Acoustics, Phys. Rev. X 4, 031042 (2014).

[66] B. Assouar, B. Liang, Y. Wu, Y. Li, J.-C. Cheng, and Y. Jing, Acoustic metasurfaces, Nat. Rev. Mater. 3, 460 (2018).

[67] H. Ge, M. Yang, C. Ma, M.-H. Lu, Y.-F. Chen, N. Fang, and P. Sheng, Breaking the barriers: Advances in acoustic functional materials, Natl. Sci. Rev. 5, 159 (2018).

[68] G. Ma and P. Sheng, Acoustic metamaterials: From local resonances to broad horizons, Sci. Adv. 2, e1501595 (2016).

[69] F. Zangeneh-Nejad and R. Fleury, Active times for acoustic metamaterials, Rev. Phys. 4, 100031 (2019).

[70] Y. Li, B. Liang, Z. M. Gu, X. Y. Zou, and J. C. Cheng, Reflected wavefront manipulation based on ultrathin planar acoustic metasurfaces, Sci. Rep. 3, 2546 (2013).

[71] H. Gao, Z. Gu, S. Liang, S. An, T. Liu, and J. Zhu, Coding Metasurface for Talbot Sound Amplification, Phys. Rev. Appl. 14, 054067 (2020).

[72] Z.-M. Gu, B. Liang, X.-Y. Zou, and J.-C. Cheng, Broadband diffuse reflections of sound by metasurface with random phase response, Europhys. Lett. 111, 64003 (2015).

[73] R. Fleury, D. L. Sounas, and A. Alu, Negative Refraction and Planar Focusing Based on Parity-Time Symmetric Metasurfaces, Phys. Rev. Lett. 113, 023903 (2014).

[74] J. Luo, J. Li, and Y. Lai, Electromagnetic Impurity-Immunity Induced by Parity-Time Symmetry, Phys. Rev. X 8, 031035 (2018).

[75] M. Lawrence, N. Xu, X. Zhang, L. Cong, J. Han, W. Zhang, and S. Zhang, Manifestation of PT Symmetry Breaking in Polarization Space with Terahertz Metasurfaces, Phys. Rev. Lett. 113, 093901 (2014).

[76] T. Liu, G. Ma, S. Liang, H. Gao, Z. Gu, S. An, and J. Zhu, Single-sided acoustic beam splitting based on parity-time symmetry, Phys. Rev. B 102, 014306 (2020).

[77] D. Li, S. Huang, Y. Cheng, and Y. Li, Compact asymmetric sound absorber at the exceptional point, Sci. China Phys. Mech. Astron. 64, 244303 (2021).

[78] E. Rivet, A. Brandstötter, K. G. Makris, H. Lissek, S. Rotter, and R. Fleury, Constant-pressure sound waves in non-Hermitian disordered media, Nat. Phys. 14, 942 (2018).

[79] H.-Z. Chen, T. Liu, H.-Y. Luan, R.-J. Liu, X.-Y. Wang, X.-F. Zhu, Y.-B. Li, Z.-M. Gu, S.-J. Liang, H. Gao, L. Lu, L. Ge, S. Zhang, J. Zhu, and R.-M. Ma, Revealing the missing dimension at an exceptional point, Nat. Phys. 16, 571 (2020).

[80] L. Feng, Z. J. Wong, R. M. Ma, Y. Wang, and X. Zhang, Single-mode laser by parity-time symmetry breaking, Science 346, 972 (2014).

[81] G. Ma, M. Xiao, and C. T. Chan, Topological phases in acoustic and mechanical systems, Nat. Rev. Phys. 1, 281 (2019).

[82] C. K. Chiu, J. C. Y. Teo, A. P. Schnyder, and S. Ryu, Classification of topological quantum matter with symmetries, Rev. Mod. Phys. 88, 035005 (2016).

[83] Z. Yang, F. Gao, X. Shi, X. Lin, Z. Gao, Y. Chong, and B. Zhang, Topological Acoustics, Phys. Rev. Lett. 114, 114301 (2015).

[84] X. Zhang, M. Xiao, Y. Cheng, M.-H. Lu, and J. Christensen, Topological sound, Commun. Phys. 1, 97 (2018).

[85] N. P. Armitage, E. J. Mele, and A. Vishwanath, Weyl and dirac semimetals in three-dimensional solids, Rev. Mod. Phys. 90, 015001 (2018).

[86] J. K. Asbóth, L. Oroszlány, and A. Pályi, A Short Course on Topological Insulators: Band Structure and Edge States in one and two Dimensions (Springer, Cham, 2016).

[87] Y. Ashida and M. Ueda, Non-Hermitian Physics, arXiv:2006.01837.

[88] K. Kawabata, K. Shiozaki, M. Ueda, and M. Sato, Symmetry and Topology in Non-Hermitian Physics, Phys. Rev. X 9, 041015 (2019).

[89] K. Kawabata, S. Higashikawa, Z. Gong, Y. Ashida, and M. Ueda, Topological unification of time-reversal and particle-hole symmetries in non-Hermitian physics, Nat. Commun. 10, 297 (2019).

[90] S. Yao and Z. Wang, Edge States and Topological Invariants of Non-Hermitian Systems, Phys. Rev. Lett. 121, 086803 (2018).

[91] D. Leykam, K. Y. Bliokh, C. Huang, Y. D. Chong, and F. Nori, Edge Modes, Degeneracies, and Topological Numbers in Non-Hermitian Systems, Phys. Rev. Lett. 118, 040401 (2017).

[92] H. Zhao, X. Qiao, T. Wu, B. Midya, S. Longhi, and L. Feng, Non-Hermitian topological light steering, Science 365, 1163 (2019).

[93] M. Li, X. Ni, M. Weiner, A. Alù, and A. B. Khanikaev, Topological phases and nonreciprocal edge states in non-Hermitian floquet insulators, Phys. Rev. B 100, 045423 (2019).

[94] L. E. F. Foa Torres, Perspective on topological states of non-Hermitian lattices, J. Phys. Mater. 3, 014002 (2019).

[95] M. Wang, L. Ye, J. Christensen, and Z. Liu, Valley Physics in Non-Hermitian Artificial Acoustic Boron Nitride, Phys. Rev. Lett. 120, 246601 (2018).

[96] N. Okuma, K. Kawabata, K. Shiozaki, and M. Sato, Topological Origin of Non-Hermitian Skin Effects, Phys. Rev. Lett. 124, 086801 (2020).

[97] F. Song, S. Yao, and Z. Wang, Non-Hermitian Skin Effect and Chiral Damping in Open Quantum Systems, Phys. Rev. Lett. 123, 170401 (2019).

[98] L. Li, C. H. Lee, S. Mu, and J. Gong, Critical non-Hermitian skin effect, Nat. Commun. 11, 5491 (2020).

[99] L. Xiao, T. Deng, K. Wang, G. Zhu, Z. Wang, W. Yi, and P. Xue, Non-Hermitian bulk–boundary correspondence in quantum dynamics, Nat. Phys. 16, 761 (2020).

[100] Z. Lin, S. Ke, X. Zhu, and X. Li, Square-root non-Bloch topological insulators in non-Hermitian ring resonators, Opt. Express 29, 8462 (2021).

[101] X. Zhu, H. Wang, S. K. Gupta, H. Zhang, B. Xie, M. Lu, and Y. Chen, Photonic non-Hermitian skin effect and non-Bloch bulk-boundary correspondence, Phys. Rev. Res. 2, 013280 (2020).

[102] X. Zhang, J.-H. Jiang, M.-H. Lu, and Y.-F. Chen, Observation of higher-order non-Hermitian skin effect, Nat. Commun. 12, 5377 (2021).

[103] L. Zhang, Y. Ge, Y.-J. Guan, Q. Chen, Q. Yan, F. Chen, R. Xi, Y. Li, D. Jia, S.-Q. Yuan, H.-X. Sun, H. Chen, and B. Zhang, Acoustic non-Hermitian skin effect from twisted winding topology, arXiv:2104.08844.

[104] V. V. Konotop, J. Yang, and D. A. Zezyulin, Nonlinear waves in PT-symmetric systems, Rev. Mod. Phys. 88, 035002 (2016).

[105] B. Peng, S. K. Ozdemir, F. Lei, F. Monifi, M. Gianfreda, G. L. Long, S. Fan, F. Nori, C. M. Bender, and L. Yang, Parity–time-symmetric whispering-gallery microcavities, Nat. Phys. 10, 394 (2014).

[106] O. V. Shramkova, K. G. Makris, D. N. Christodoulides, and G. P. Tsironis, Nonlinear scattering by non-Hermitian multilayers with saturation effects, Phys. Rev. E 103, 052205 (2021).

[107] S. Xia, D. Kaltsas, D. Song, I. Komis, J. Xu, A. Szameit, H. Buljan, K. G. Makris, and Z. Chen, Nonlinear tuning of PT symmetry and non-Hermitian topological states, Science 372, 72 (2021).

[108] B. I. Popa and S. A. Cummer, Non-reciprocal and highly nonlinear active acoustic metamaterials, Nat. Commun. 5, 3398 (2014).

[109] X. Wen, A. Fan, W. Y. Tam, J. Zhu, F. Lemoult, M. Fink, and J. Li, Asymmetric frequency conversion with acoustic non-Hermitian space-time varying metamaterial, arXiv:2011.01006.

[110] C. Cho, X. Wen, N. Park, and J. Li, Digitally virtualized atoms for acoustic metamaterials, Nat. Commun. 11, 251 (2020).

[111] C. Cho, X. Wen, N. Park, and J. Li, Acoustic willis meta-atom beyond the bounds of passivity and reciprocity, Commun. Phys. 4, 82 (2021).

[112] P. Peng, W. Cao, C. Shen, W. Qu, J. Wen, L. Jiang, and Y. Xiao, Anti-parity–time symmetry with flying atoms, Nat. Phys. 12, 1139 (2016).

[113] F. Zhang, Y. Feng, X. Chen, L. Ge, and W. Wan, Synthetic Anti-PT Symmetry in a Single Microcavity, Phys. Rev. Lett. 124, 053901 (2020).

[114] Y. Choi, C. Hahn, J. W. Yoon, and S. H. Song, Observation of an anti-PT-symmetric exceptional point and energy-difference conserving dynamics in electrical circuit resonators, Nat. Commun. 9, 2182 (2018).

[115] F. Yang, Y.-C. Liu, and L. You, Anti-PT symmetry in dissipatively coupled optical systems, Phys. Rev. A 96, 053845 (2017).

[116] Y. Yang, Y. P. Wang, J. W. Rao, Y. S. Gui, B. M. Yao, W. Lu, and C. M. Hu, Unconventional Singularity in Anti-Parity-Time Symmetric Cavity Magnonics, Phys. Rev. Lett. 125, 147202 (2020).

[117] Y. Li, Y.-G. Peng, L. Han, M.-A. Miri, W. Li, M. Xiao, X.-F. Zhu, J. Zhao, A. Alù, S. Fan, and C.-W. Qiu, Anti-parity-time symmetry in diffusive systems, Science 364, 170 (2019).

[118] P. Cao, Y. Li, Y. Peng, C. Qiu, and X. Zhu, High-Order exceptional points in diffusive systems: Robust APT symmetry against perturbation and phase oscillation at APT symmetry breaking, ES Energy Environ. 7, 48 (2019).

[119] L. Ge and H. E. Türeci, AntisymmetricPT-photonic structures with balanced positive- and negative-index materials, Phys. Rev. A 88, 053810 (2013).
