# High-Order Perfect Absorption in the Absence of Exceptional Point

Huisheng Xu $^{1}$ , Luojia Wang $^{2}$ , Luqi Yuan $^{2,*}$ and Liang Jin $^{1,\dagger}$

$^{1}$ School of Physics, Nankai University, Tianjin 300071, China

$^{2}$ State Key Laboratory of Photonics and Communications, School of Physics and Astronomy, Shanghai Jiao Tong University, Shanghai, 200240, China

![](images/2a1996510c1d32b9360d9ae00136abf963b55636ba871c7965a8a86d70337b92.jpg)

(Received 9 October 2025; revised 16 January 2026; accepted 11 March 2026; published 31 March 2026)

High-order perfect absorption of coherent input has recently attracted significant attention due to its broadband absorption capacity. However, the realization of a high-order perfect absorber relies on the exceptional point (EP) to coalesce the scattering zeros. Here, we present a general scattering framework and achieve the high-order perfect absorber in the absence of an EP. We consider the asynchronous coherent input, where a spatial delay introduces a momentum-dependent phase factor beyond the amplitude and phase control in synchronous coherent input. This new degree of freedom enables active control of the momentum-dependent output, effectively reshaping the absorption line shape necessary for the high-order perfect absorber. Remarkably, despite the absence of an EP, the proposed high-order perfect absorber exhibits significant response to the perturbations in the delay length. Our findings provide insights for the delay-induced momentum-sensitive interference phenomenon and offer a new route for wave control.

DOI: 10.1103/nkls-pgkf

Introduction—Coherent perfect absorption (CPA) is a hallmark non-Hermitian interference phenomenon $[1-8]$ . When properly superposed input waves from different ports arrive simultaneously at a scattering center, they can be completely absorbed by a lossy medium. CPA has been extensively explored across diverse platforms, stimulating applications ranging from photocurrent enhancement to optical data processing $[9,10]$ . However, CPA typically has a narrow bandwidth, where a slight frequency shift can destroy the resonance and limit its utility. To overcome this limitation, exceptional points (EPs) have been incorporated into CPA $[11-13]$ . When two or more purely incoming wave solutions coalesce at a real frequency, the resulting high-order perfect absorber, known as a CPA EP, exhibits a broadened absorption line shape. This phenomenon has been predicted and observed in optical $[14-16]$ , electronic $[17]$ , and acoustic systems $[18]$ , offering strong potential for broadband absorption. Moreover, this high-order perfect absorber inherits the sensitivity of EPs, enabling EP-based sensing applications $[19-21]$ .

Fundamentally, the broadband absorption of high-order perfect absorbers corresponds to a high-order zero of the scattering output and does not necessitate the presence of

EPs. Notably, the output after scattering depends on both the scattering center and the incident waves. In conventional CPA $[1-3]$ , the incident waves are always synchronous and controlled solely by their amplitudes and phases. Such input corresponds to a zero-eigenvalue eigenmode of the scattering matrix $[9]$ , thereby revealing only the properties of the scattering center. This raises the fundamental question of whether there are additional degrees of freedom in the incident waves that can be exploited to independently manipulate the scattering output. Addressing this question is essential for a comprehensive understanding of the interplay between the scattering center and the incident waves, and for realizing high-order perfect absorbers within a more general framework.

In this Letter, we establish a generalized scattering framework for asynchronous coherent input and demonstrate the high-order perfect absorption in the absence of EPs. The asynchronous coherent input introduces a spatial delay as a new degree of freedom for manipulating wave interference. The incorporated delay brings a momentum-dependent dynamical phase factor, enabling active control of the scattering output through the incident waves, which cannot be achieved by conventional amplitude and phase modulation. Using this mechanism, we design a high-order perfect absorber with sextic absorption line shape, which arises from the delay-induced destructive interference instead of the coalescence of scattering zeros. Since the resulting high-order absorber relies on the delay rather than an EP, it exhibits a strong absorption response to variations in delay length without spectral splitting from perturbations in system parameters. Our delay framework highlights the asynchronous wave interference in the non-Hermitian scattering and opens new opportunities for wave manipulation and coherent control.

Delay formalism—We briefly review the concept of Wigner delay $[22,23]$ , a fundamental scattering phenomenon underlying the characterization of asynchronous coherent input. Intuitively, a wave packet would reflect directly at the boundary of a potential barrier following a geometric path. However, the actual reflection involves a phase gradient across the momentum of the wave packet, shifting its center by a delay length L, as if it experienced a spatial delay relative to the ideal path. This delay length is quantified as

$$
L = d \arg [ r (k) ] / d k,\tag{1}
$$

where $r(k)$ is the reflection coefficient and k is the momentum of the wave packet. The Wigner delay has been extensively studied in optical fiber [24], acoustic system [25], time-varying media [26], and synthetic lattice [27], with applications in precise measurement [28] and interferometry [29].

We address the inverse problem of what the influence of delay on the interference phenomenon is if it is introduced to the coherent input. We simply consider a two-port scattering system. In contrast to the Wigner delay that characterizes delay after scattering, we adapt this framework to describe relative delay in coherent input waves before scattering. The input waves in the two ports are represented as $a = (a_{1}, a_{2})^{\mathrm{T}}$ with $a_{1}: a_{2} = 1: \eta$ . Conventionally, the two components arrive simultaneously at the scattering center without any relative delay, and $\eta = |\eta| e^{i\varphi}$ describes their amplitude ratio and phase difference [Fig. 1(a)]. Here, we consider the input wave in port 2 having a delayed arrival with length l along the propagation direction [Fig. 1(b)]. Inspired by the characterization of delay in Eq. (1), the delay length l relates to the phase of $\eta$ through its gradient $l = d \arg(\eta)/dk$ . Given that l is a k independent constant, we obtain $\eta$ after an integration,

$$
\eta = | \eta | e ^ {i \varphi} e ^ {i k l}.\tag{2}
$$

Notably, the delay introduces an additional momentum-dependent dynamical phase factor $e^{ikl}$ .

We further consider the impact of delay on the output. Applying the scattering matrix to the input wave vector, the output wave vector is obtained $b = (b_{1}, b_{2})^{\mathrm{T}} = S(k)a$ , where $S(k)$ is the scattering matrix. The momentum-dependent output for a renormalized input $a = 1/\sqrt{1 + |\eta|^{2}}(1, |\eta|e^{i\varphi}e^{ikl})^{\mathrm{T}}$ in Eq. (2) is

$$
\binom{b _ {1}}{b _ {2}} = \frac {1}{\sqrt {1 + | \eta | ^ {2}}} \binom{s _ {1 1} (k) + s _ {1 2} (k) | \eta | e ^ {i \varphi} e ^ {i k l}}{s _ {2 1} (k) + s _ {2 2} (k) | \eta | e ^ {i \varphi} e ^ {i k l}},\tag{3}
$$

![](images/b082de5d96929d0de97ef900f6fae15940ddfc9931f9e33272258a76b1e7d917.jpg)
FIG. 1. (a) Coherent input without relative delay, where the coherent control only involves amplitude ratio and phase difference. (b) Coherent input with relative delay, where the delay length l causes an extra momentum-dependent phase $e^{ikl}$ . Right panels: delay-induced modification of the momentum-dependent output.

where $s_{qp}(k)$ is the scattering coefficient describing the output in port q for the input in port p.

Notably, the dependence of output on the momentum k stems from both the scattering coefficients $s_{qp}(k)$ and the delay-induced dynamical phase $e^{ikl}$ . Therefore, the delay provides an additional control parameter that actively manipulates the momentum dependence of the output. This leads to novel scattering phenomena that are absent under synchronous coherent input. For example, the delay facilitates a momentum-sensitive modulation of absorption spectrum, where tuning the delay length l can enhance the absorption without relying on an EP. To illustrate this effect, we consider the derivatives of the output components $b_{1}$ and $b_{2}$ with respect to the momentum k. In the context of perfect absorption $b_{1}=b_{2}=0$ , the derivatives are

$$
\frac {d}{d k} b _ {1} = \frac {s _ {1 1} (k)}{\sqrt {1 + | \eta | ^ {2}}} \left[ \frac {d}{d k} \ln \frac {s _ {1 1} (k)}{s _ {1 2} (k)} - i l \right],\tag{4}
$$

$$
\frac {d}{d k} b _ {2} = \frac {s _ {2 1} (k)}{\sqrt {1 + | \eta | ^ {2}}} \left[ \frac {d}{d k} \ln \frac {s _ {2 1} (k)}{s _ {2 2} (k)} - i l \right].\tag{5}
$$

By contrast to the first-order perfect absorber with only $b_{1(2)} = 0$ [Fig. 1(a)], the derivative $db_{1(2)}/dk = 0$ corresponds to the second-order perfect absorber with a broadened absorption line shape [Fig. 1(b)]. Therefore, high-order perfect absorbers are enabled by the delay through tuning the derivatives to zero [30], without the need for an EP.

In the following, we propose high-order perfect absorbers based on the delay framework and demonstrate the crucial role of delay in the wave interference.

Synthetic frequency lattice—CPA offers a powerful tool for rectifying energy flow. However, in discrete lattices, the absorption of an initial excitation is incomplete, as the excitation inevitably spreads in momentum space while the absorption bandwidth remains narrow $[17,31]$ . This limitation becomes particularly severe for excitations with compact spatial profiles constrained by the lattice size [32-35]. Recently, synthetic frequency lattices offer a promising way to overcome this lattice-size constraint [36-38], with frequency modes effectively serving as discrete lattice sites [39-41].

![](images/a4f4c6e6d1fee8fde2a0567f55a018c3d2d220f7f50ad7f8e918bb16b957cc50.jpg)

![](images/73dccc809d9e98d6ee7f5babe2254f92c91e5496b864b8c025634cdd50eb25fc.jpg)
FIG. 2. (a) Schematic illustration of a ring resonator coupled to a two-level dissipative atom. The ring resonator is dynamically modulated via an electro-optic modulator (EOM). (b) The synthetic frequency lattice for (a). (c) Eigenvalue intensity $|r(k) + t(k)|^2$ of the scattering matrix. The black line marks the zeros of the eigenvalue. (d) The normalized output $|b_1|^2$ and $|b_2|^2$ versus $k$ for $\gamma = \kappa = 2J$ . The inset displays the double-logarithmic plot near $k = \pi / 2$ , with $\delta k = k - \pi / 2$ . (e) Trajectories of three purely incoming wave solutions at fixed $\kappa = 2J$ as $\gamma / J$ varying from 0 to 3.

Figure 2(a) shows a design of the dynamically modulated ring resonator coupled to a dissipative two-level atom [42-44]. An electro-optic modulator, driven by a sinusoidal external source $-2J\cos(\Omega t)$ , is placed within the resonator. The dynamic refractive index modulation induces coupling between adjacent resonant modes $\omega_{n}=\omega_{0}+n\Omega$ , and creates a uniform synthetic frequency lattice [45-47]. The two-level atom with ground state $|g\rangle$ and excited state $|e\rangle$ is separated by the frequency $\omega_{0}$ , so the transition between $|g\rangle$ and $|e\rangle$ is resonantly coupled to the zeroth resonant mode with the coupling strength $\kappa$ [48]. The atom also possesses an intrinsic dissipation rate $\gamma$ . Such a two-level atom can be implemented using a Cs atom coupled to a whispering-gallery-mode resonator [49], or alternatively in superconducting platforms using artificial atoms, such as superconducting qubits coupled to transmission-line resonators [43,44]. The Hamiltonian of this model is

$$
\begin{array}{l} H = \omega_ {n} \sum_ {n} \hat {c} _ {n} ^ {\dagger} \hat {c} _ {n} - 2 J \cos (\Omega t) \sum_ {n} (\hat {c} _ {n} ^ {\dagger} \hat {c} _ {n + 1} + \hat {c} _ {n + 1} ^ {\dagger} \hat {c} _ {n}) \\ \qquad + (\omega_ {0} - i \gamma) | e \rangle \langle e | + \kappa (\hat {c} _ {0} ^ {\dagger} | g \rangle \langle e | + \hat {c} _ {0} | e \rangle \langle g |), \end{array}\tag{6}
$$

where $\hat{c}_{n}^{\dagger}\left(\hat{c}_{n}\right)$ is the creation (annihilation) operator for the nth resonant mode.

Under rotating wave approximation, the above Hamiltonian describes a uniform tight-binding chain side-coupled with a dissipation [Fig. 2(b)]. The dynamics of photon transport is characterized by the equations of motion for the single excitation (Supplemental Material Sec. A [50]),

$$
i \frac {\mathrm{d} f _ {n}}{\mathrm{d} t} = \omega_ {0} f _ {n} - J f _ {n - 1} - J f _ {n + 1}, \quad (| n | > 0),\tag{7}
$$

$$
i \frac {\mathrm{d} f _ {0}}{\mathrm{d} t} = \omega_ {0} f _ {0} - J f _ {- 1} - J f _ {1} + \kappa f _ {\gamma},\tag{8}
$$

$$
i \frac {\mathrm{d} f _ {\gamma}}{\mathrm{d} t} = (\omega_ {0} - i \gamma) f _ {\gamma} + \kappa f _ {0},\tag{9}
$$

where $f_{n}$ describes the wave amplitude of the photon in the nth lattice site while the atom is in the ground state, $|g\rangle$ , and $f_{\gamma}$ denotes the wave amplitude of the atom being excited to $|e\rangle$ by the propagating photon. We emphasize that the above equations of motion are generic and can be realized in various realistic platforms beyond synthetic dimensions (Supplemental Material Sec. B [50]).

In the elastic scattering process, the wave amplitudes have the form of $f_{n} = \psi_{n} e^{-i\omega t}$ , where $\psi_{n}$ is the steady-state wave amplitude. As the system has the inversion symmetry, the reflection and transmission are symmetric [60], with $s_{11}(k) = s_{22}(k) = r(k)$ and $s_{21}(k) = s_{12}(k) = t(k)$ , being independent of the input direction. By setting $\psi_{n} = e^{ikn} + r(k)e^{-ikn}$ for $n \leq 0$ and $\psi_{n} = t(k)e^{ikn}$ for $n \geq 0$ , the scattering coefficients are obtained (Supplemental Material Sec. C [50]),

$$
r (k) = t (k) - 1 = - \frac {\kappa^ {2}}{2 J \gamma \sin (k) + 2 i J ^ {2} \sin (2 k) + \kappa^ {2}}.\tag{10}
$$

The corresponding scattering matrix $S(k)$ is nonunitary due to the dissipation [61–63]. The eigenvalues of $S(k)$ consist of a momentum-dependent component $r(k) + t(k)$ and a constant value of -1. Figure 2(c) plots $|r(k) + t(k)|^{2}$ as a function of momentum k. At the resonant input $k = \pi/2$ for $\kappa^{2} = 2J\gamma$ , the eigenvalue reaches zero, as indicated by the black line. Notably, a zero eigenvalue of the scattering matrix identifies a perfect absorption [9]. The associated eigenvector is $a = 1/\sqrt{2}(1, 1)^{\mathrm{T}}$ , leading to $b = S(k)a = 0$ . This indicates that the resonant waves synchronously input from opposite directions with identical amplitude and phase are perfectly absorbed by the atom (Supplemental Material Sec. D [50]).

Without loss of generality, we show the normalized output $|b_{1}|^{2}$ and $|b_{2}|^{2}$ as a function of momentum k for $\gamma = \kappa = 2J$ in Fig. 2(d). The perfect absorption displays a quadratic line shape, as evidenced by a slope of 2 in the logarithmic plot [the inset of Fig. 2(d)]. The quadratic line shape suggests that the perfect absorption is first-order and occurs without the coalescence of scattering zeros [11]. We analytically continue k to the complex plane and solve for the purely incoming wave solutions [64,65]. Figure 2(e) displays three effective complex solutions for k (Supplemental Material Sec. E [50]), only one of which moves to the real axis (red dot) at $\gamma = \kappa = 2J$ .

High-order perfect absorption—The perfect absorption shown in Fig. 2(d) exhibits a narrow bandwidth, that is, the output wave near $k = \pi/2$ deviates from zero rapidly. A broadband absorption spectrum of high-order perfect absorber resolves this issue [14]. We utilize our delay formalism to create the high-order perfect absorption with an EP being absent.

From the delay formalism, the outputs are obtained by substituting Eq. (10) into Eq. (3). Perfect absorption occurs when the coherent input waves have equal amplitude $|\eta| = 1$ and satisfy the phase-matching condition $e^{i\varphi}e^{ikl} = 1$ . Compared with the scheme without delay [14–21], the phase matching is now jointly tuned by the relative phase $\varphi$ and the delay length l. Notably, the output derivatives can be tuned to zero via the delay, allowing the active reshaping of the absorption line shape to realize high-order perfect absorption. Under perfect absorption $b_{1} = b_{2} = 0$ , the output derivatives are shown in Figs. 3(a) and 3(b). Remarkably, when $l = 2J/\gamma$ and $l = -2J/\gamma$ , the derivatives vanish ( $db_{1}/dk = 0$ and $db_{2}/dk = 0$ ), giving rise to second-order perfect absorption in ports 1 and 2, respectively. Furthermore, in the case $\gamma = \kappa = 2J$ , an even higher-order perfect absorption is possible. Specifically, when l = 1 and l = -1, the second derivatives also vanish ( $d^{2}b_{1}/dk^{2} = 0$ and $d^{2}b_{2}/dk^{2} = 0$ ), leading to a third-order perfect absorption in ports 1 and 2. As shown in Fig. 3(c), the absorption bandwidth varies with delay and system parameters. Increasing the order of perfect absorption via tuning the delay results in an ultrawide absorption bandwidth around $k = \pi/2$ , where the output approaches zero over a broad momentum range. Figures 3(d) and 3(e) show the outputs for ports 1 and 2 with delay length l = 1. Unlike the delay-free case in Fig. 2(d), where $|b_{1}|^{2}$ and $|b_{2}|^{2}$ are identical, the introduction of delay induces third-order perfect absorption in port 1 and first-order perfect absorption in port 2. Notably, the broadened absorption bandwidth can also be characterized in terms of the incident frequency, with the absorption order identical to that extracted from the incident momentum [66].

(a)
![](images/e5d96349f26be20cf1673d319db59973f21fdc0577af201803bbbfeea118d9b9.jpg)

(b)
![](images/73032791a8cc72f15191e8421b80bb5ecf7d6ae1bd3aab3bc2b4c5cc8b881840.jpg)
(d)

![](images/2ff3cd791ef6f8b252783d94e54ad6042582e478d4308c76694f58a919b4f34c.jpg)

![](images/f1a20fb3db32438622c77a228c580ffbcd60270d505117ce181d3dbf1110a3b6.jpg)

(e)
![](images/dc2bd1a3424ac129c7fdca88966b0f5ca2937777a2fb177cfa4984f5655ab171.jpg)

(f)
![](images/7cf5eeeb962db31a8f5dc73ac21926a863e2b88516239ac878e4d878c7a38316.jpg)
FIG. 3. (a),(b) Derivatives of $b_{1}$ and $b_{2}$ under perfect absorption $b_{1} = b_{2} = 0$ . (c) Comparison of absorption line shapes $|b_1|^2$ near $k = \pi / 2$ for a set of parameters marked in (a). (d),(e) Output intensities for ports 1 and 2, exhibiting third-order (slope 6) and first-order (slope 2) absorption for $\gamma = \kappa = 2J$ and $\lambda = 1$ . (f) Time evolution of the amplitude $|\phi|$ for two coherent input wave packets, initially centered at $n_{-} = -50$ and $n_{+} = 51$ , with $\sigma = 10$ .

The proposed high-order perfect absorption arises from the actively introduced delay in the coherent input, which fundamentally differs from that achieved from the coalescence of scattering zeros (Supplemental Material Sec. F [50]). The delay scheme highlights the interplay between the momentum dependences of scattering coefficients and the input. By contrast, conventional scheme from the coalescence of scattering zeros solely depends on the scattering center [11-13].

To verify the proposed high-order perfect absorption, we perform the time evolution of coherent input with a relative delay. The initial excitation is a superposition of two counterpropagating Gaussian wave packets,

$$
| \phi \rangle = \frac {1}{\sqrt {1 + | \eta | ^ {2}}} (| \phi_ {-} \rangle + | \eta | e ^ {i \varphi} | \phi_ {+} \rangle),\tag{11}
$$

where $\left|\phi_{\pm}\right\rangle=\sum_{n}e^{-(n-n_{\pm})^{2}/(2\sigma^{2})}e^{\mp ik_{c}(n-n_{\pm})}/\sqrt[4]{\pi\sigma^{2}}|n\rangle$ [67–70]. The wave packets $\left|\phi_{-}\right\rangle$ and $\left|\phi_{+}\right\rangle$ are centered at the modes $n_{-}$ and $n_{+}$ with relative amplitude $|\eta|$ , phase $e^{i\varphi}$ , and delay length $l=|n_{+}|-|n_{-}|$ . Here, $\sigma$ controls their width and $k_{c}$ is the central momentum. This excitation can be implemented in the synthetic frequency lattice by temporally shaping the intraresonator field within a single round trip, where the temporal position of the injected pulse determines the momentum (Supplemental Material Sec. G [50]). In Fig. 3(f), the two wave packets are shown to be nearly perfectly absorbed by the dissipative atom. The incomplete absorption is attributed to the spreading of Gaussian wave packets near $k_{c}=\pi/2$ in the momentum space.

The distinct outputs in ports 1 and 2 shown in Fig. 3(f) imply different orders of perfect absorption. To quantify this difference, we calculate the residual intensity of the wave packets after scattering in ports 1 and 2, denoted as $I_{1}$ and $I_{2}$ , respectively. Through analytical derivation (Supplemental Material Sec. H [50]), we find that the residual intensity of the wave packets after the nth-order perfect absorption scales with the wave packet width as

$$
I _ {s} \propto \sigma^ {- 2 n},\tag{12}
$$

where the subscript s = 1, 2 denotes the port. In Fig. 4(a), we plot the residual intensities at both ports for varying packet widths $\sigma$ [71]. The scaling laws confirm the third-order perfect absorption in port 1 and the first-order perfect absorption in port 2. We highlight that the enhanced performance of high-order perfect absorption originates from its ultrawide absorption bandwidth.

![](images/229918367491584d0eae5936ae39deb8bf9f3e44c028db9b83154de2377e0a13.jpg)

(c)
![](images/f9945c3df999e617d85f8dca2cdcd94dad9b2ab81c02ab974942bb39f26b380d.jpg)

![](images/5dda2572be557347ae600f9eb8d83e0378952b04d1d0ac736fe70d78ec979b1c.jpg)
FIG. 4. (a) Scaling behaviors of residual intensity for third-order and first-order absorption. (b) Absorption in decibels for ports 1 and 2 as a function of delay l, exhibiting peaks at l = 1 and l = -1, respectively. (c) $\chi_{1}$ and $\chi_{2}$ versus the delay perturbation $\delta l$ near l = 1 [gray region in (a)].

Delay modulation—In the EP-based schemes $[72–77]$ , achieving high-order perfect absorption requires tuning the system parameters to reach the EPs, and tiny parameter variations near the EP induce large splittings in the absorption spectrum $[19–21]$ . By contrast, the delay-based scheme does not work at an EP, and the perturbations of system parameters do not lead to spectrum splitting [Fig. 2(c)]. Instead, our proposed high-order perfect absorption depends on the delay, and small variations in the delay can induce a strong absorption response without relying on the EP mechanism $[78,79]$ .

We consider the variation of absorption under different delays. To clearly present the variations, we express the residual intensity in decibels as $-10\log_{10}[I_{1(2)}]$ . In Fig. 4(b), we plot the absorptions in ports 1 and 2 as functions of the delay length l under the condition of perfect absorption. Peaks in $I_{1}$ at l=1 and $I_{2}$ at l=-1 are observed. This rapid intensity decrease indicates the delay-induced significant enhancement of absorption. We define the $\chi_{s}$ as the change in absorption (in decibels) with respect to the perturbation in the delay,

$$
\chi_ {s} = \frac {- 1 0 \log_ {1 0} (I _ {s} ^ {\prime} / I _ {s})}{\delta l},\tag{13}
$$

where $I_{s}$ ( $I_{s}^{\prime}$ ) denotes the residual intensity before (after) introducing the perturbation $\delta l$ . Figure 4(c) shows $\chi_{s}$ as a function of $\delta l$ near the absorption peak at l=1. A significant increase in $\chi_{s}$ near the third-order perfect absorption is clearly observed, and the enhancement originates from the sharp transition near the high-order perfect absorption peak in the scaling residual intensity from $\sigma^{-2}$ to $\sigma^{-6}$ . The change in the scaling exponent introduces a dominant term $20\Delta n\log_{10}\sigma/\delta l$ in $\chi_{s}$ , where $\Delta n$ is the change in the order of perfect absorption (Supplemental Material Sec. I [50]). Therefore, the modulation of delay alters the order of perfect absorption, and induces a significant change in the absorption. Moreover, the response can be further enlarged by increasing the wave packet width. The strong absorption response has the potential for sensing applications, but a full assessment would take noise into account (Supplemental Material Sec. J [50]). Further systematic analysis on the signal-to-noise ratio has profound meaning [80–82].

Conclusion—Traditionally, coherent input assumes synchronous excitation, with control confined to the amplitude and phase. Here, we introduce the concept of asynchronous coherent input and establish a generalized scattering framework. We show that delay, as a new degree of freedom, reshapes the absorption line shape, enables ultrabroadband perfect absorption, and enhances the response to perturbations. The broader role of delay in coherent input overturns the limitation that dissipative systems without CPA EPs cannot achieve broadband absorption, and provides a new paradigm for designing high-order absorbers. Our findings offer valuable insights for the asynchronous wave interference engineering, paving the way for designing advanced photonic devices with tailored absorption profiles. The delay mechanism also enables flexible control of reflection and transmission line shapes $[83–86]$ , and can be generalized to multiport systems $[20,87,88]$ . While this Letter focuses on spatial delay, the underlying principles are directly applicable to temporal delay $[27,89,90]$ . Furthermore, the delay may stimulate intriguing interference effects and transport phenomena for giant atoms coupled to multimode waveguides $[43]$ .

Acknowledgments—This work is supported by the National Natural Science Foundation of China (Grant No. 12525502, No. 124B2079, No. 12475021, and No. 12204304). L.J. also acknowledge the support from Quantum Science and Technology-National Science and Technology Major Project (Grant No. 2024ZD0301000). L.Y. also acknowledge the support from National Key R&D Program of China (Grant No. 2023YFA1407200).

Data availability—The data that support the findings of this article are not publicly available. The data are available from the authors upon reasonable request.

[1] Y. D. Chong, L. Ge, H. Cao, and A. D. Stone, Coherent perfect absorbers: Time-reversed lasers, Phys. Rev. Lett. 105, 053901 (2010).

[2] S. Longhi, PT-symmetric laser absorber, Phys. Rev. A 82, 031801(R) (2010).

[3] W. Wan, Y. Chong, L. Ge, H. Noh, A. D. Stone, and H. Cao, Time-reversed lasing and interferometric control of absorption, Science 331, 889 (2011).

[4] H. Noh, Y. Chong, A. D. Stone, and H. Cao, Perfect coupling of light to surface plasmons by coherent absorption, Phys. Rev. Lett. 108, 186805 (2012).

[5] Y. Sun, W. Tan, H.-Q. Li, J. Li, and H. Chen, Experimental demonstration of a coherent perfect absorber with PT phase transition, Phys. Rev. Lett. 112, 143903 (2014).

[6] C. Hang, G. Huang, and V. V. Konotop, Tunable spectral singularities: Coherent perfect absorber and laser in an atomic medium, New J. Phys. 18, 085003 (2016).

[7] J. Jeffers, Nonlocal coherent perfect absorption, Phys. Rev. Lett. 123, 143602 (2019).

[8] W. Gou, T. Chen, D. Xie, T. Xiao, T.-S. Deng, B. Gadway, W. Yi, and B. Yan, Tunable nonreciprocal quantum transport through a dissipative Aharonov-Bohm ring in ultracold atoms, Phys. Rev. Lett. 124, 070402 (2020).

[9] D. G. Baranov, A. Krasnok, T. Shegai, A. Alù, and Y. Chong, Coherent perfect absorbers: Linear control of light with light, Nat. Rev. Mater. 2, 17064 (2017).

[10] C. Yan, M. Pu, J. Luo, Y. Huang, X. Li, X. Ma, and X. Luo, Coherent perfect absorption of electromagnetic wave in subwavelength structures, Opt. Laser Technol. 101, 499 (2018).

[11] W. R. Sweeney, C. W. Hsu, S. Rotter, and A. D. Stone, Perfectly absorbing exceptional points and chiral absorbers, Phys. Rev. Lett. 122, 093901 (2019).

[12] V. Achilleos, G. Theocharis, O. Richoux, and V. Pagneux, Non-Hermitian acoustic metamaterials: Role of exceptional points in sound absorption, Phys. Rev. B 95, 144303 (2017).

[13] H. S. Xu, L. C. Xie, and L. Jin, High-order spectral singularity, Phys. Rev. A 107, 062209 (2023).

[14] C. Wang, W. R. Sweeney, A. D. Stone, and L. Yang, Coherent perfect absorption at an exceptional point, Science 373, 1261 (2021).

[15] S. Soleymani, Q. Zhong, M. Mokim, S. Rotter, R. El-Ganainy, and Ş. K. Özdemir, Chiral and degenerate perfect absorption on exceptional surfaces, Nat. Commun. 13, 599 (2022).

[16] H. Hörner, L. Wild, Y. Slobodkin, G. Weinberg, O. Katz, and S. Rotter, Coherent perfect absorption of arbitrary wavefronts at an exceptional point, Phys. Rev. Lett. 133, 173801 (2024).

[17] S. Suwunnarat, Y. Tang, M. Reisner, F. Mortessagne, U. Kuhl, and T. Kottos, Non-linear coherent perfect absorption in the proximity of exceptional points, Commun. Phys. 5, 5 (2022).

[18] Y.-F. Xia, Z.-X. Xu, Y.-T. Yan, A. Chen, J. Yang, B. Liang, J.-C. Cheng, and J. Christensen, Observation of coherent perfect acoustic absorption at an exceptional point, Phys. Rev. Lett. 135, 067001 (2025).

[19] Y. Feng, Y. Wang, Z. Li, and T. Li, Enhanced sensing and broadened absorption with higher-order scattering zeros, Opt. Express 32, 32283 (2024).

[20] D. Yan, A. S. Shalin, Y. Wang, Y. Lai, Y. Xu, Z. H. Hang, F. Cao, L. Gao, and J. Luo, Ultrasensitive higher-order exceptional points via non-Hermitian zero-index materials, Phys. Rev. Lett. 134, 243802 (2025).

[21] Y.-D. Hu, Y.-P. Wang, R.-C. Shen, Z.-Q. Wang, W.-J. Wu, and J. Q. You, Synthetically enhanced sensitivity using higher-order exceptional point and coherent perfect absorption, arXiv:2401.01613.

[22] E. P. Wigner, Lower limit for the energy derivative of the scattering phase shift, Phys. Rev. 98, 145 (1955).

[23] F. T. Smith, Lifetime matrix in collision theory, Phys. Rev. 118, 349 (1960).

[24] W. Xiong, P. Ambichl, Y. Bromberg, B. Redding, S. Rotter, and H. Cao, Spatiotemporal control of light transmission through a multimode fiber with strong mode coupling, Phys. Rev. Lett. 117, 053901 (2016).

[25] B. Orazbayev, M. Malléjac, N. Bachelard, S. Rotter, and R. Fleury, Wave-momentum shaping for moving objects in heterogeneous and dynamic media, Nat. Phys. 20, 1441 (2024).

[26] S. A. Ponomarenko, J. Zhang, and G. P. Agrawal, Goos-Hänchen shift at a temporal boundary, Phys. Rev. A 106, L061501 (2022).

[27] C. Qin, S. Wang, B. Wang, X. Hu, C. Liu, Y. Li, L. Zhao, H. Ye, S. Longhi, and P. Lu, Temporal Goos-Hänchen shift in synthetic discrete-time heterolattices, Phys. Rev. Lett. 133, 083802 (2024).

[28] M. Strauß, A. Carmele, J. Schleibner, M. Hohn, C. Schneider, S. Höfling, J. Wolters, and S. Reitzenstein, Wigner time delay induced by a single quantum dot, Phys. Rev. Lett. 122, 107401 (2019).

[29] M. Han, J.-B. Ji, C. S. Leung, K. Ueda, and H. J. Wörner, Separation of photoionization and measurement-induced delays, Sci. Adv. 10, eadj2629 (2024).

[30] Similarly, when the higher-order derivatives of the output components vanish, i.e., $d^{m}b_{1(2)} / dk^{m} = 0$ for all integers $m < n$ , the system realizes an $n$ th-order perfect absorber with an even broader absorption line shape.

[31] A. Müllers, B. Santra, C. Baals, J. Jiang, J. Benary, R. Labouvie, D. A. Zezyulin, V. V. Konotop, and H. Ott, Coherent perfect absorption of nonlinear matter waves, Sci. Adv. 4, eaat6539 (2018).

[32] R. W. Robinett, Quantum wave packet revivals, Phys. Rep. 392, 1 (2004).

[33] J. Wenner, Y. Yin, Y. Chen, R. Barends, B. Chiaro, E. Jeffrey, J. Kelly, A. Megrant, J. Y. Mutus, C. Neill, P. J. J. O'Malley, P. Roushan, D. Sank, A. Vainsencher, T. C. White, A. N. Korotkov, A. N. Cleland, and J. M. Martinis, Catching time-reversed microwave coherent state photons with 99.4% absorption efficiency, Phys. Rev. Lett. 112, 210501 (2014).

[34] M. Asano, K. Y. Bliokh, Y. P. Bliokh, A. G. Kofman, R. Ikuta, T. Yamamoto, Y. S. Kivshar, L. Yang, N. Imoto, Ş. K. Özdemir, and F. Nori, Anomalous time delays and quantum weak measurements in optical micro-resonators, Nat. Commun. 7, 13488 (2016).

[35] Z. Dong, H. Li, T. Wan, Q. Liang, Z. Yang, and B. Yan, Quantum time reflection and refraction of ultracold atoms, Nat. Photonics 18, 68 (2024).

[36] L. Yuan, Q. Lin, M. Xiao, and S. Fan, Synthetic dimension in photonics, Optica 5, 1396 (2018).

[37] M. Ehrhardt, S. Weidemann, L. J. Maczewsky, M. Heinrich, and A. Szameit, A perspective on synthetic dimensions in photonics, Laser Photonics Rev. 17, 2200518 (2023).

[38] D. Yu, W. Song, L. Wang, R. Srikanth, S. Kaushik Sridhar, T. Chen, C. Huang, G. Li, X. Qiao, X. Wu, Z. Dong, Y. He, M. Xiao, X. Chen, A. Dutt, B. Gadway, and L. Yuan, Comprehensive review on developments of synthetic dimensions, Photonics Insights 4, R06 (2025).

[39] M. Zhang, B. Buscaino, C. Wang, A. Shams-Ansari, C. Reimer, R. Zhu, J. M. Kahn, and M. Lončar, Broadband electro-optic frequency comb generation in a lithium niobate microring resonator, Nature (London) 568, 373 (2019).

[40] A. Dutt, Q. Lin, L. Yuan, M. Minkov, M. Xiao, and S. Fan, A single photonic cavity with two independent physical synthetic dimensions, Science 367, 59 (2020).

[41] H. Xu, Z. Dong, L. Yuan, and L. Jin, Probing bulk band topology from time boundary effect in synthetic dimension, Phys. Rev. Lett. 134, 163801 (2025).

[42] H. Xiao, L. Wang, Z.-H. Li, X. Chen, and L. Yuan, Bound state in a giant atom-modulated resonators system, npj Quantum Inf. 8, 80 (2022).

[43] L. Du, Y. Zhang, J.-H. Wu, A. F. Kockum, and Y. Li, Giant atoms in a synthetic frequency dimension, Phys. Rev. Lett. 128, 223602 (2022).

[44] R. Chai, G. Cai, Q. Xie, H. Wu, and Y. Li, Single-photon routing induced by giant atoms in a synthetic frequency dimension, Phys. Rev. A 112, 033724 (2025).

[45] Z.-A. Wang, Y.-T. Wang, X.-D. Zeng, J.-M. Ren, W. Liu, X.-H. Wei, Z.-P. Li, Y.-Z. Yang, N.-J. Guo, L.-K. Xie, J.-Y. Liu, Y.-H. Ma, J.-S. Tang, Z.-W. Zhou, C.-F. Li, and G.-C. Guo, On-chip photonic simulating band structures toward arbitrary-range coupled frequency lattices, Phys. Rev. Lett. 133, 233805 (2024).

[46] W. Liu, X. Su, C. Li, C. Zeng, B. Wang, Y. Wang, Y. Ding, C. Qin, J. Xia, and P. Lu, Reconfigurable chiral edge states in synthetic dimensions on an integrated photonic chip, Phys. Rev. Lett. 134, 143801 (2025).

[47] R. Ye, G. Li, S. Wan, X. Xue, P.-Y. Wang, X. Qiao, L. Wang, H. Li, S. Liu, J. Wang, R. Ma, F. Bo, Y. Zheng, C.-H. Dong, L. Yuan, and X. Chen, Construction of various time-varying Hamiltonians on thin-film lithium niobate chip, Phys. Rev. Lett. 134, 163802 (2025).

[48] Different from synthetic giant-atom models [42-44], here we assume the free spectral range $\Omega$ is sufficiently large so the atomic transition cannot efficiently couple to other resonant modes with $n \neq 0$ .

[49] X. Zhou, H. Tamura, T.-H. Chang, and C.-L. Hung, Coupling single atoms to a nanophotonic whispering-gallery-mode resonator via optical guiding, Phys. Rev. Lett. 130, 103601 (2023).

[50] See Supplemental Material at http://link.aps.org/supplemental/10.1103/nkls-pgkf for more detailed information on the synthetic frequency lattice, the high-order perfect absorber in realistic systems, the scattering coefficients, the atomic excited state population, the scattering zeros, the comparison between EP-based and delay-based high-order perfect absorber, the Gaussian wave packets, the residual intensity, the absorption response to the delay, and the shot noise in the output, which includes Refs. [51–59].

[51] H. Ramezani, Y. Wang, E. Yablonovitch, and X. Zhang, Unidirectional perfect absorber, IEEE J. Sel. Top. Quantum Electron. 22, 115 (2016).

[52] S. Longhi, Quantum-optical analogies using photonic structures, Laser Photonics Rev. 3, 243 (2009).

[53] H. Ramezani, H.-K. Li, Y. Wang, and X. Zhang, Unidirectional spectral singularities, Phys. Rev. Lett. 113, 263905 (2014).

[54] L. Jin and Z. Song, Incident direction independent wave propagation and unidirectional lasing, Phys. Rev. Lett. 121, 073901 (2018).

[55] P.-O. Löwdin, Studies in perturbation theory. IV. Solution of eigenvalue problem by projection operator formalism, J. Math. Phys. (N.Y.) 3, 969 (1962).

[56] L. Jin and Z. Song, Partitioning technique for discrete quantum systems, Phys. Rev. A 83, 062118 (2011).

[57] Y. He, Z. Dong, G. Li, P. Yu, X. Wu, X. Chen, and L. Yuan, Observing momentum conservation at temporal interfaces in synthetic frequency dimension, Sci. Adv. 11, eadz5445 (2025).

[58] C. W. J. Beenakker, Thermal radiation and amplified spontaneous emission from a random medium, Phys. Rev. Lett. 81, 1829 (1998).

[59] Y. D. Chong, H. Cao, and A. D. Stone, Noise properties of coherent perfect absorbers and critically coupled resonators, Phys. Rev. A 87, 013843 (2013).

[60] L. Jin and Z. Song, Symmetry-protected scattering in non-Hermitian linear systems, Chin. Phys. Lett. 38, 024202 (2021).

[61] C. C. Wojcik, H. Wang, M. Orenstein, and S. Fan, Universal behavior of the scattering matrix near thresholds in photonics, Phys. Rev. Lett. 127, 277401 (2021).

[62] H. S. Xu and L. Jin, Pseudo-Hermiticity protects the energy-difference conservation in the scattering, Phys. Rev. Res. 5, L042005 (2023).

[63] Y. Shou, D. Wang, Y. Wang, Q.-K.-L. Huang, H. Chen, W. Yu, R. Ju, H. Chen, and Y. Li, Resonant and scattering exceptional points in non-Hermitian metasurfaces, npj Nanophotonics 2, 29 (2025).

[64] R. Jost and A. Pais, On the scattering of a particle by a static potential, Phys. Rev. 82, 840 (1951).

[65] Y. D. Chong, L. Ge, and A. D. Stone, PT-symmetry breaking and laser-absorber modes in optical scattering systems, Phys. Rev. Lett. 106, 093902 (2011).

[66] The nth-order perfect absorption in terms of incident momentum leads to $b_{1,2}(k) \propto (\delta k)^{n}$ . Using the dispersion relation $\omega(k) = \omega_{0} - 2J \cos(k)$ , each momentum k corresponds to a well-defined incident frequency $\omega$ , giving $b_{1,2}(\omega) \propto (\delta \omega / v_{g})^{n}$ , where $v_{g} = d\omega / dk = 2J \sin(k)$ is the group velocity.

[67] A. Alberucci, C. P. Jisha, M. Monika, U. Peschel, and S. Nolte, Wave manipulation via delay-engineered periodic potentials, Phys. Rev. Res. 4, 043162 (2022).

[68] O. Y. Long, K. Wang, A. Dutt, and S. Fan, Time reflection and refraction in synthetic frequency dimension, Phys. Rev. Res. 5, L012046 (2023).

[69] Y. Ren, K. Ye, Q. Chen, F. Chen, L. Zhang, Y. Pan, W. Li, X. Li, L. Zhang, H. Chen, and Y. Yang, Observation of momentum-gap topology of light at temporal interfaces in a time-synthetic lattice, Nat. Commun. 16, 707 (2025).

[70] H. Wang, C. Guo, and S. Fan, Spatiotemporal steering of nondiffracting wave packets, Phys. Rev. Lett. 134, 073803 (2025).

[71] Here, we require $\sigma$ to be sufficiently large such that the wave packet in momentum space is well localized around $k_{\mathrm{c}}$ . In this regime, the output components can be reliably expanded in the Taylor series about $k_{\mathrm{c}}$ , and the scaling law in Eq. (12) holds, as derived in Supplemental Material Sec. H [50].

[72] J. Wiersig, Enhancing the sensitivity of frequency and energy splitting detection by using exceptional points: Application to microcavity sensors for single-particle detection, Phys. Rev. Lett. 112, 203901 (2014).

[73] W. Chen, Ş. K. Özdemir, G. Zhao, J. Wiersig, and L. Yang, Exceptional points enhance sensing in an optical microcavity, Nature (London) 548, 192 (2017).

[74] H. Hodaei, A. U. Hassan, S. Wittek, H. Garcia-Gracia, R. El-Ganainy, D. N. Christodoulides, and M. Khajavikhan, Enhanced sensitivity at higher-order exceptional points, Nature (London) 548, 187 (2017).

[75] Q. Zhong, J. Ren, M. Khajavikhan, D. N. Christodoulides, Ş. K. Özdemir, and R. El-Ganainy, Sensing with exceptional surfaces in order to combine sensitivity with robustness, Phys. Rev. Lett. 122, 153902 (2019).

[76] Q. Zhong, J. Kou, Ş. K. Özdemir, and R. El-Ganainy, Hierarchical construction of higher-order exceptional points, Phys. Rev. Lett. 125, 203602 (2020).

[77] T. Chen, D. Zou, Z. Zhou, R. Wang, Y. Feng, H. Sun, and X. Zhang, Ultra-sensitivity in reconstructed exceptional systems, Natl. Sci. Rev. 11, nwae278 (2024).

[78] Y. Chu, Y. Liu, H. Liu, and J. Cai, Quantum sensing with a single-qubit pseudo-Hermitian system, Phys. Rev. Lett. 124, 020501 (2020).

[79] L. Xiao, Y. Chu, Q. Lin, H. Lin, W. Yi, J. Cai, and P. Xue, Non-Hermitian sensing in the absence of exceptional points, Phys. Rev. Lett. 133, 180801 (2024).

[80] M. Zhang, W. Sweeney, C. W. Hsu, L. Yang, A. D. Stone, and L. Jiang, Quantum noise theory of exceptional point amplifying sensors, Phys. Rev. Lett. 123, 180501 (2019).

[81] R. Kononchuk, J. Cai, F. Ellis, R. Thevamaran, and T. Kottos, Exceptional-point-based accelerometers with

enhanced signal-to-noise ratio, Nature (London) 607, 697 (2022).

[82] W. Ding, X. Wang, and S. Chen, Fundamental sensitivity limits for non-Hermitian quantum sensors, Phys. Rev. Lett. 131, 160801 (2023).

[83] O. Jamadi, B. Real, K. Sawicki, C. Hainaut, A. González-Tudela, N. Pernet, I. Sagnes, M. Morassi, A. Lemaître, L. Le Gratiet, A. Harouri, S. Ravets, J. Bloch, and A. Amo, Reconfigurable photon localization by coherent drive and dissipation in photonic lattices, Optica 9, 706 (2022).

[84] H. S. Xu and L. Jin, Robust incoherent perfect absorption, Phys. Rev. Res. 6, L022006 (2024).

[85] C. Guo and S. Fan, Passivity constraints on the relations between transmission, reflection, and absorption eigenvalues, Phys. Rev. B 110, 205431 (2024).

[86] T. M. Blessan, B. Real, C. Druelle, C. Fournier, A. M. d. l. Heras, A. González-Tudela, I. Sagnes, A. Harouri, L. Le Gratiet, A. Lemaître, S. Ravets, J. Bloch, C. Hainaut, and A. Amo, Directional transport and nonlinear localization of light in a one-dimensional driven-dissipative photonic lattice, Phys. Rev. Res. 7, 033283 (2025).

[87] H. S. Xu and L. Jin, Coherent resonant transmission, Phys. Rev. Res. 4, L032015 (2022).

[88] C. Guo, D. A. B. Miller, and S. Fan, Unitary control of multiport wave transmission, Phys. Rev. A 111, 023507 (2025).

[89] A. Farhi, A. Mekawy, A. Alù, and D. Stone, Excitation of absorbing exceptional points in the time domain, Phys. Rev. A 106, L031503 (2022).

[90] Z. Dong, X. Chen, and L. Yuan, Extremely narrow band in moiré photonic time crystal, Phys. Rev. Lett. 135, 033803 (2025).
