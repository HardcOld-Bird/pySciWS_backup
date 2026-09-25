# Spin-Selective Topological Effects without Encircling Exceptional Points

Shun Wan, $^{1,*}$ Yuze Hu $^{\textcircled{i},2,*,\dagger}$ Ran Huang $^{\textcircled{i},3,*}$ Shiru Song, $^{1}$ Hui Yang, $^{4}$ Weibao He, $^{2}$ Siyang Hu, $^{1}$ Ziheng Ren, $^{1}$ Zhongyi Yu, $^{1}$ Yunlan Zuo, $^{5}$ Yulong Zhang, $^{1}$ Dongsheng Yang $^{\textcircled{i},2}$ Xiang'ai Cheng, $^{1}$ Franco Nori $^{\textcircled{i},3,6}$ Hui Jing, $^{2,4,\ddagger}$ and Tian Jiang $^{\textcircled{i},1,2,7,\S}$

$^{1}$ College of Advanced Interdisciplinary Studies, National University of Defense Technology, Changsha 410073, People's Republic of China

$^{2}$ College of Science, National University of Defense Technology, Changsha 410073, People's Republic of China $^{3}$ Quantum Information Physics Theory Research Team, Center for Quantum Computing (RQC), RIKEN, Wakoshi, Saitama 351-0198, Japan

$^{4}$ Key Laboratory of Low-Dimensional Quantum Structures and Quantum Control of Ministry of Education, Hunan Normal University, Changsha 410081, People's Republic of China

$^{5}$ School of Physics and Chemistry, Hunan First Normal University, Changsha 410205, People's Republic of China $^{6}$ Department of Physics, University of Michigan, Ann Arbor, Michigan 48109-1040, USA $^{7}$ Hunan Research Center of the Basic Discipline for Physical States, Changsha 410073, People's Republic of China

(Received 6 August 2025; revised 30 October 2025; accepted 8 January 2026; published 3 February 2026)

Exceptional points (EPs), namely non-Hermitian spectral singularities, enable unconventional light-matter interactions, leading to intriguing phenomena, such as chiral mode transfer, state flip, and chiral phase accumulation. Yet considerable previous EP effects in topological photonics based on a standard strategy, i.e., evolution encircling EPs, require fine tuning of multiple parameters. To go beyond this approach for designing unconventional topological devices with more convenient tunabilities, it is essential to study how to control open trajectories by tuning fewer or even a single parameter without encircling EPs. This route remains largely unexplored. Here, we demonstrate $2\pi$ optical phase control with open evolution trajectories by tuning a single geometrical parameter of the structure in a terahertz metasurface featuring a pair of EPs in a complex-frequency space, while only $\pi$ phase control can be achieved with open trajectories in the metasurface with a single EP. We find that such full $2\pi$ -phase control without encircling EPs is realized due to the open trajectories passing between the two EPs with identical chirality but opposite topological charges. Furthermore, we demonstrate topological spin-selective beam deflection and realize dynamical control of EP positions in the complex-frequency space by tuning the angle of split-ring resonators. Our Letter drives the field of EP topological photonics into a broader regime with open trajectory approaches, opening up perspectives for achieving unconventional photonic devices that are more controllable with exciting applications, such as novel wavefront control and polarization multiplexing.

DOI: 10.1103/92m4-dhtc

Recent advances in non-Hermitian physics have opened numerous opportunities for basic research and engineering applications $[1-9]$ . Central to this field is the concept of exceptional points (EPs), which are spectral singularities characterized by the coalescence of eigenvalues and eigenstates simultaneously, exhibiting nontrivial Riemann sheet topologies $[10-15]$ . These unique features of EPs have enabled the observation of various intriguing phenomena, including perfect absorption $[16-19]$ , ultrasensitive sensing $[20-27]$ , and nonreciprocal wave propagation $[28,29]$ . EPs have been experimentally observed and harnessed across diverse classical and quantum platforms, including electronic circuits [30], condensed exciton-polaritons [31], thermal atoms [32], and superconducting qubits [33].

Encircling EPs in photonic systems becomes an attractive strategy for realizing chiral phase accumulation $[34–36]$ , asymmetric mode switching $[37–43]$ , and the state-flip effect $[31,44]$ , with applications ranging from optical switches to wavefront control devices. However, previous experimental realizations of encircling EPs generally require precise controls of multiple parameters, such as resonant frequency, coupling strengths, and loss or damping rates of systems. Recent studies have demonstrated asymmetric mode switching with a closed trajectory in the vicinity of an EP, rather than encircling it $[45]$ . Moreover, chiral transmission can occur with an open trajectory by tuning multiple parameters $[46]$ . A fundamental question arises whether open evolution trajectories

Dual-EP metasurface

without encircling EPs can enable any counterintuitive EP effects by tuning fewer or even a single parameter.

![](images/0528af2895025fb61fafccda2ab7846c319e58cc7b5769e1d9543ab6376671ba.jpg)

Here, we experimentally demonstrate topological spin-selective $2\pi$ -phase effects by tuning a single geometrical parameter of the structure of a terahertz metasurface without encircling EPs. We achieve full $2\pi$ -phase accumulation along open trajectories in a terahertz metasurface consisting of two nonorthogonal modes. To analyze the topological characteristics of the system, we study the distributions of the optical phase and amplitude, as well as the positions of EPs in a complex-frequency space. We find that the EP positions can be tuned by altering the angle $\theta$ of the splitting resonators (SRRs) in the metasurface. More interestingly, an EP pair with identical chirality but opposite topological charges can emerge in a complex plane and enable the accumulation of full $2\pi$ phase with an open path between this EP pair, namely continuous tuning of $\theta$ at a specific complex frequency. Furthermore, we experimentally demonstrate spin-selective beam deflection by utilizing such topological phase effects. Our results release previous conditions of $2\pi$ -phase accumulation with the control of multiple parameters to encircling EPs, which opens a new frontier at the junction of two emerging fields in physics, namely terahertz metasurface and non-Hermitian optics based on complex-frequency excitations. This provides potential applications in wavefront control technology, polarization multiplexing, and optical information encryption.

In conventional studies, chiral EPs can be achieved by tuning the coupling strength between a pair of orthogonally coupled modes and frequency detuning, which leads to the collapse of the transmission matrices' eigenstate with a fixed circular polarization (CP) vector $[47–50]$ . Full $2\pi$ -phase control can be achieved by traversing a closed path around an EP [Fig. 1(a)], a spin-selective effect rooted in the intrinsic chirality of non-Hermitian systems at EPs $[34]$ . This paradigm necessitates closed-loop trajectories in parameter space, while open trajectories near the EP undergo a maximum phase modulation around $\pi$ [Figs. 1(b) and 1(c)].

Figures 1(d)–1(f) illustrate our implementation of topological phase control without encircling EPs. The proposed system consists of a metal cut-wire and an SRR that is rotated by an angle $\theta$ about its center [Fig. 1(d)]. The variation in $\theta$ redistributes the resonant components of the SRR along the $x$ and $y$ directions, thereby altering the coupling coefficient between the two dipoles. The illustration in Fig. 1(d) provides a macroscopic schematic of periodic arrayed metasurfaces with different SRR orientations. More information about materials and geometrical parameters of our fabricated metasurfaces can be found in Sec. S1 of Supplemental Material (SM) [51]. Notably, two EPs emerge in the space spanned by frequency $f$ and geometrical parameter $\theta$ [Fig. 1(e)]. The $2\pi$ -phase control is achieved along open trajectories connecting these paired EPs [Fig. 1(f)].

Single-EP systems

![](images/ce718af9ddd876c1c99736ba3c64148c2140f22b128d38e7a93fd6246675d180.jpg)

![](images/2e0f870cd9c5868e1d37d36deeb372ed22dbe19a04a522dcb71c243b4110e92e.jpg)

![](images/47a4d496c87448b81cda0ecdf789318cc6eb6c9fcfd0abf5a7600f634bba299d.jpg)

(e)
![](images/347aa4003b8c07b3ab613f17f365208af99dc41e48581f67ab73e0cb633d3916.jpg)

(c)
![](images/cbdcfa8fe953d8afaa034dce3d2f9f79e5e1122ec95d3d8cdd17a0704e4a9c6d.jpg)

(f)
![](images/d9c3049ffba41c2cc8f2ac91829c10e48cd08a649d40e8003aba594a34b5b1f1.jpg)
FIG. 1. Full $2\pi$ -phase engineering without encircling EPs. (a) The spectral response of the system in the parameter space defined by structural geometric parameters or physical variables (X and Y). A full $2\pi$ -phase control can be achieved by traversing a closed path encircling an EP (tuning two parameters), where the blue dots (black arrows) indicate the starting (ending) points. (b), (c) The maximum phase modulation achievable for an open trajectory near an EP is around $\pi$ . (d) The structure diagram of the nonorthogonal metasurface consists of a metal cut-wire and a split-ring resonator (SRR). Here, $\theta$ is the angle between the x axis and the gap of the SRR. The illustration presents a macroscopic schematic of periodic arrayed metasurfaces with varying SRR orientations. (e), (f) The spectral response of the metasurface. Two EPs with same chirality emerge in the parameter space spanned by frequency f and $\theta$ . Between these EPs, full $2\pi$ -phase accumulation occurs along an open path.

The EP complex-square-root topology of the eigenvalue surface can be observed by varying two parameters in real-frequency space. Given the intrinsic complex eigenvalue spectrum of non-Hermitian systems, we employ analytical continuation into complex-frequency space $(f \rightarrow f + if')$ to characterize their topological features. This approach is equivalent to introducing virtual gain or loss into the system and can be implemented by using incident signals whose amplitudes grow or decay exponentially in time [54–58]. Through this approach, the response characteristics of the system in higher dimensions are provided, uncovering two distinct types of singularities: zeros (where the function vanishes) and poles (where it diverges) [17,59–62].

These phase singularities generate phase vortices with quantized topological charges,

$$
q = \frac {1}{2 \pi} \oint_ {C _ {l}} \nabla_ {f} \phi (f) \cdot \mathrm{d} f,\tag{1}
$$

with $q = +1$ for a zero and $q = -1$ for a pole, and $\phi(f)$ denotes the phase of the metasurface response function. To investigate the evolution of these phase singularities in the complex plane, we establish an analytical model based on temporal coupled-mode theory combined with orthogonal decomposition [51]. We show that the zeros of the $CP$ conversion coefficient are equivalent to the formation conditions of EPs. Moreover, EPs exist in pairs, each associated with a degeneracy of eigenstates corresponding to either left- or right-handed circular polarization (LHCP, RHCP), denoted as $\mathrm{EP_L}$ and $\mathrm{EP_R}$ , respectively [51].

Based on the analysis of complex-frequency zeros and poles, the phase accumulation in resonant metasurfaces can be interpreted from a topological perspective. For example, using temporal coupled-mode theory, we demonstrate that the Pancharatnam-Berry phase can be interpreted as originating from the self-rotation of a pole in the complex-frequency plane $[51]$ . Furthermore, while a $2\pi$ -phase shift is typically obtained by encircling a singularity or crossing a branch cut between a zero and a pole $[63,64]$ , we demonstrate both theoretically and experimentally a novel mechanism for achieving a spin-selective $2\pi$ -phase shift through the motion of a zero-pole pair.

Topological charges ensure the robustness of EPs and poles, preventing their disappearance as $\theta$ changes [65,66]. As a result, EPs trace out continuous trajectories in complex-frequency space, known as exceptional lines. Figures 2(a) and 2(b) illustrate two exceptional lines with opposite chirality, that is, the evolution of the zeros of CP conversion coefficients $t_{R\rightarrow L}$ (red line) and $t_{L\rightarrow R}$ (blue line) as $\theta$ varies [where the subscript $R\rightarrow L$ represents transmitted CP conversion from RHCP (R) to LHCP (L)]. The exceptional line of $EP_{R}$ intersects the plane of $\mathrm{Im}(f)=0$ from two different directions, giving rise to two real-frequency EPs. In contrast, the lines of poles (black line) and $EP_{L}$ (blue line) remain confined below the real axis in the complex-frequency plane.

Our experimental verification uses a $\theta$ -graded metasurface array on quartz, where the SRR orientation is systematically varied in $5^{\circ}$ increments across the array. These samples are characterized with polarization-resolved terahertz time-domain spectroscopy. Frequency-domain spectra are obtained via fast Fourier transform of the time-domain signals. Figures 2(c) and 2(d) present the trajectories of EPs and poles in a two-dimensional plane. A truncated complex-frequency excitation can be generated at low frequencies using an arbitrary waveform generator, whereas at high frequencies an effective complex-frequency excitation can be synthesized through real-frequency drives [17,54].

(b)
![](images/3aa87296fbb6b8f440dcf1d257bfef288dabfc8db75a42c2f1ab46695c63b2f6.jpg)

![](images/1606dc659751c7fd2511f92bdf8439a7cfa335f26806f73761e220446b2544d7.jpg)

(c)
(d)
![](images/b627bfe86651353d8413cb8515d01c083155d3e6eb3dbbefb21dcce67e0ddd81.jpg)

![](images/d55e00c3fbc7f974727b57dafc4c2a9162e10b59eec2a07b409dbe1d6e1f7895.jpg)
FIG. 2. Topological evolution of EPs and poles. (a),(b) Evolution of EPs and poles in the complex-frequency plane of CP conversion coefficients $t_{R\rightarrow L}$ (a) and $t_{L\rightarrow R}$ (b). The black circles indicate the complex-frequency bound states in the continuum (c-BIC). (c),(d) The evolution of zeros and poles of $t_{R\rightarrow L}$ and $t_{L\rightarrow R}$ in a two-dimensional plane defined by the real and imaginary parts of the complex frequency, where the experimental results are extracted by fitting the measured spectral data.

Accordingly, in our terahertz-frequency experiments, we employ the latter approach, fitting the experimentally measured real-frequency spectra with temporal coupled-mode theory to reconstruct the locations of zeros and poles in the complex-frequency plane $[17,51]$ . From theoretical simulations of the transmission spectra, the system parameters are further obtained, including resonant frequency $(f_{1,2})$ , intrinsic loss rate $(\Gamma_{1,2})$ , radiation loss $(\gamma_{1,2})$ , and coupling coefficient $(\kappa)$ , and analytically continue the fitted response to complex-frequency space. Zeros and poles are then located as isolated singularities of the fitted response in the complex plane.

The resulting zero and pole trajectories exhibit good agreement with the theoretical predictions, confirming the reliability of our measurement and fitting procedures. A slight deviation (0.023 THz) between experimental and theoretical EP positions is observed, which arises primarily from discrepancies between the sample-dependent quartz refractive index relative to the simulated parameters.

Mirror symmetry at $\theta=0^{\circ}$ or $180^{\circ}$ forces $t_{R\to L}=t_{L\to R}$ , inducing a diabolic point. Such a point belongs to a nondefective EP, a straightforward extension of the Hermitian degeneracy point [6]. It is worth noting that zeros merge with poles at $\theta=0^{\circ}$ to form a complex-frequency bound state in the continuum (BIC) characterized by a diverging radiative Q factor [67–70]. The complex-frequency BIC stems from the interference cancellation of the two resonances.

![](images/67419bef708aeaa5a3bbab488d7a062d24a825959bc525dc9f8daa4869f2d363.jpg)

![](images/1d2b339d7da00f36cc00c68486c87696f2f98c9a46b1724c240098c82eaa0e4e.jpg)

![](images/07a7ddd327ee685681cf60ad3eade2e766be2ff2ef5b51ffb72236afff0c7737.jpg)
FIG. 3. Spin-selective phase accumulation without encircling EPs. (a),(b) Simulated and experimental spectra of $t_{R \to L}$ in the parameter space defined by the real frequency and $\theta$ , showing a pair of EPs with the same chirality but opposite topological charges. (c) The phase profiles at $f = 0.44 \mathrm{THz}$ and $0.42 \mathrm{THz}$ exhibit a full $2\pi$ -phase accumulation as $\theta$ changes, while $t_{L \to R}$ exhibits no significant variation, highlighting the chiral property of the phase modulation.

Symmetry breaking within $0^{\circ} < \theta < 180^{\circ}$ causes the complex-frequency BIC to split into a pair of EPs with opposite chirality. Because of its proximity to a pole, the singularity strength of $EP_{L}$ is significantly weakened, and it may collide and annihilate with the pole at certain critical angles. This annihilation erases local topological features while preserving global topological charge conservation. In contrast, exceptional line of $EP_{R}$ first ascends into the upper half of the complex plane and subsequently returns to the lower half, which means two homochiral EPs emerge within the space defined by real frequency and $\theta$ . We conclude that the two EPs observed at real frequencies correspond to the intersections of an exceptional line with the real axis from two different directions.

Notably, although zeros in the complex-frequency plane carry a topological charge of +1, their effective charge in a specific projection plane can appear inverted, depending on the direction in which they are traversed. Consequently, one of the EPs may acquire an effective topological charge of -1. For experimental feasibility, we choose the real-frequency plane as the basis to validate our theoretical predictions. Figures 3(a) and 3(b) present the simulated and experimental spectra of $t_{R\rightarrow L}$ as functions of real frequency and $\theta$ , revealing a pair of EPs with opposite topological charges. Importantly, a full $2\pi$ -phase accumulation along an open trajectory defined by continuously varying $\theta$ between the two EPs can be achieved [Fig. 3(c)]. If the following two conditions are satisfied, a spin-selective $2\pi$ -phase accumulation is ensured for any open path: (i) two singularities with opposite topological charges exist [e.g., in our system, the condition of the imaginary part of the complex frequency is -0.003 THz < Im(f) < 0.016 THz [51]]; (ii) an open path of sufficient length passes between the two singularities. Therefore, this robust spin-selective $2\pi$ -phase accumulation is topologically protected. We provide further discussion of the topologically protected phase behavior induced by zero-pole motion in Sec. S5 of SM [51]. Besides, the distinct trajectories of the two exceptional lines result in asymmetric amplitude and phase responses: $t_{R\rightarrow L}$ exhibits strong phase modulation and spectral zeros, whereas $t_{L\rightarrow R}$ shows neither.

To demonstrate a practical implementation, we design a phase-gradient metasurface functioning as a spin-selective beam deflector. The unit cell [Fig. 4(a)] shares the same geometrical parameters, differing only in the SRR rotation angle. Figure 4(b) schematically illustrates the operational principle through simulated CP beam deflection, emphasizing the deflection of a single CP conversion beam by the metasurface. The experimental results shown in Figs. 4(c) and 4(d) reveal asymmetric deflection of two CP conversion beams, with a maximum efficiency of 7% and a power ratio of approximately 4.5:1. Theoretically, the efficiency can be significantly enhanced by introducing a metallic back plate $[51]$ . Because of mirror symmetry, the eigenstates of EPs and the phase accumulation of the CP conversion coefficient become opposite when $\theta$ varies between $180^{\circ}$ and $360^{\circ}$ . For verification, we also made mirror-symmetric samples, and the results obtained are contrary, further proving the correctness of our theory $[51]$ .

(a)
![](images/4785fc8819164ffb5a7064b3ece1c0d4abd907284c6d22b616a0f12f2399d73c.jpg)

![](images/91ebd780b84168f7247734420c7a71881e12623518afc4edc7c0b5fb2d756947.jpg)

(c)
![](images/738f036f6c201d384267c1aae1c3e315cf5a499525a97bbcfb665678284a8980.jpg)

(d)
![](images/61d1a4136ea6eb7627de2d9b9ad76b1d5c8bc986e123ceed7e8d397fe68ba8be.jpg)
FIG. 4. Spin-selective beam deflector without encircling EPs. (a) Top view and side view of the unit structure of the spin-selective beam deflector. $\alpha$ denotes the deflection angle of the incident beam. (b) Schematic of the designed metasurface, which achieves spin-selective beam deflection. (c), (d) Measured asymmetric deflection for the two CP-conversion channels, $T_{R\rightarrow L}$ (c) and $T_{L\rightarrow R}$ (d).

In summary, we have experimentally demonstrated a paradigm-shifting approach to non-Hermitian topological effects by achieving $2\pi$ optical phase control through open trajectories by tuning a single geometrical parameter in a terahertz metasurface, bypassing the conventional need to encircle EPs. Analytical extension of the transmission matrix into complex-frequency space reveals that EP pairs with opposite topological charges occur when an exceptional line crosses a given reference plane from opposite directions. We show that open trajectories connecting these EPs enable full phase accumulation, a phenomenon inaccessible in systems containing only one EP. Our experimental realization of spin-selective beam deflection highlights the practical viability of this approach for wavefront manipulation and polarization-multiplexed devices. Our system provides the potential for studying strong coupling phenomena with resonance hybridization, such as BICs $[67–69]$ , Fano resonance $[71]$ , electromagnetically induced transparency $[72]$ , and phase accumulation control $[73]$ . Looking forward, our findings pave the way for novel applications in optical information encryption, terahertz communications, and advanced polarization control, while offering a versatile platform for exploring complex-frequency dynamics and topological phenomena in metasurfaces.

Acknowledgments—The authors thank Jinhui Shi and Yicheng Li in Harbin Engineering University for their contributions to theoretical discussions, Weiqiang Ding and SixCarbon Technology (Shenzhen) for their contributions to sample manufacturing, and Lei Du in National University of Defense Technology for his contributions to drawing. This work is supported by National Key R&D Program of China (No. 2024YFE0102400 and No. 2020YFB2205800), National Natural Science Foundation of China (No. 62305384, No. 62075240, and No. 11935006), the Youth Innovation Talent Incubation Foundation of the National University of Defense Technology (No. 2023-lxy-fhij-007), the Science and Technology Innovation Program of Hunan Province (Grants No. 2020RC4047 and No. 2025ZYJ001), and Hunan provincial major sci-tech program (No. 2023ZJ1010). R.H. is supported by the RIKEN Special Postdoctoral Researchers (SPDR) program. F.N. is supported in part by the Japan Science and Technology Agency (JST) [via the CREST Quantum Frontiers program Grant No. JPMJCR24I2, the Quantum Leap Flagship Program (Q-LEAP), and the Moonshot R&D Grant No. JPMJMS256E].

Data availability—The data that support the findings of this article are not publicly available. The data are available from the authors upon reasonable request.

[1] A. Li, H. Wei, M. Cotrufo, W. Chen, S. Mann, X. Ni, B. Xu, J. Chen, J. Wang, S. Fan et al., Nat. Nanotechnol. 18, 706 (2023).

[2] Z. Rao, C. Meng, Y. Han, L. Zhu, K. Ding, and Z. An, Nat. Phys. 20, 1904 (2024).

[3] M. Reisenbauer, H. Rudolph, L. Egyed, K. Hornberger, A. V. Zasedatelev, M. Abuzarli, B. A. Stickler, and U. Delić, Nat. Phys. 20, 1629 (2024).

[4] L. Feng, R. El-Ganainy, and L. Ge, Nat. Photonics 11, 752 (2017).

[5] X. Feng, T. Wu, Z. Gao, H. Zhao, S. Wu, Y. Zhang, L. Ge, and L. Feng, Nat. Photonics 19, 264 (2025).

[6] K. Ding, C. Fang, and G. Ma, Nat. Rev. Phys. 4, 745 (2022).

[7] R. El-Ganainy, K. G. Makris, M. Khajavikhan, Z. H. Musslimani, S. Rotter, and D. N. Christodoulides, Nat. Phys. 14, 11 (2018).

[8] C. E. Rüter, K. G. Makris, R. El-Ganainy, D. N. Christodoulides, M. Segev, and D. Kip, Nat. Phys. 6, 192 (2010).

[9] J. Zhang, G. Xia, C. Wu, T. Chen, Q. Zhang, Y. Xie, W. Su, W. Wu, C. Qiu, P. Chen et al., Nat. Commun. 16, 301 (2025).

[10] E. J. Bergholtz, J. C. Budich, and F. K. Kunst, Rev. Mod. Phys. 93, 015005 (2021).

[11] C. Coulais, R. Fleury, and J. van Wezel, Nat. Phys. 17, 9 (2021).

[12] B. Peng, Ş. Özdemir, S. Rotter, H. Yilmaz, M. Liertzer, F. Monifi, C. Bender, F. Nori, and L. Yang, Science 346, 328 (2014).

[13] Ş. K. Özdemir, S. Rotter, F. Nori, and L. Yang, Nat. Mater. 18, 783 (2019).

[14] L. Qiao, W. Zhang, and K. Shi, Chin. Phys. Lett. 41, 120301 (2024).

[15] S. Weimann, M. Kremer, Y. Plotnik, Y. Lumer, S. Nolte, K. G. Makris, M. Segev, M. C. Rechtsman, and A. Szameit, Nat. Mater. 16, 433 (2017).

[16] W. R. Sweeney, C. W. Hsu, S. Rotter, and A. D. Stone, Phys. Rev. Lett. 122, 093901 (2019).

[17] C. Wang, W. R. Sweeney, A. D. Stone, and L. Yang, Science 373, 1261 (2021).

[18] H. Hörner, L. Wild, Y. Slobodkin, G. Weinberg, O. Katz, and S. Rotter, Phys. Rev. Lett. 133, 173801 (2024).

[19] D. G. Baranov, A. Krasnok, T. Shegai, A. Alù, and Y. Chong, Nat. Rev. Mater. 2, 1 (2017).

[20] H. Hodaei, A. U. Hassan, S. Wittek, H. Garcia-Gracia, R. El-Ganainy, D. N. Christodoulides, and M. Khajavikhan, Nature (London) 548, 187 (2017).

[21] J.-H. Park, A. Ndao, W. Cai, L. Hsu, A. Kodigala, T. Lepetit, Y.-H. Lo, and B. Kanté, Nat. Phys. 16, 462 (2020).

[22] W. Chen, Ş. Kaya Özdemir, G. Zhao, J. Wiersig, and L. Yang, Nature (London) 548, 192 (2017).

[23] R. Kononchuk, J. Cai, F. Ellis, R. Thevamaran, and T. Kottos, Nature (London) 607, 697 (2022).

[24] J. Xu, Y. Mao, Z. Li, Y. Zuo, J. Zhang, B. Yang, W. Xu, N. Liu, Z. J. Deng, W. Chen et al., Nat. Nanotechnol. 19, 1472 (2024).

[25] Y.-P. Ruan, J.-S. Tang, Z. Li, H. Wu, W. Zhou, L. Xiao, J. Chen, S.-J. Ge, W. Hu, H. Zhang et al., Nat. Photonics 19, 109 (2025).

[26] M. P. Hokmabadi, A. Schumer, D. N. Christodoulides, and M. Khajavikhan, Nature (London) 576, 70 (2019).

[27] Q. Zhong, J. Ren, M. Khajavikhan, D. N. Christodoulides, Ş. K. Özdemir, and R. El-Ganainy, Phys. Rev. Lett. 122, 153902 (2019).

[28] B. Peng, Ş. K. Özdemir, F. Lei, F. Monifi, M. Gianfreda, G. L. Long, S. Fan, F. Nori, C. M. Bender, and L. Yang, Nat. Phys. 10, 394 (2014).

[29] Y. Choi, C. Hahn, J. W. Yoon, S. H. Song, and P. Berini, Nat. Commun. 8, 14154 (2017).

[30] Y. Choi, C. Hahn, J. W. Yoon, and S. H. Song, Nat. Commun. 9, 2182 (2018).

[31] T. Gao, E. Estrecho, K. Bliokh, T. Liew, M. Fraser, S. Brodbeck, M. Kamp, C. Schneider, S. Höfling, Y. Yamamoto et al., Nature (London) 526, 554 (2015).

[32] Z. Zhang, F. Zhang, Z. Xu, Y. Hu, H. Bao, and H. Shen, Phys. Rev. Lett. 133, 133601 (2024).

[33] M. Abbasi, W. Chen, M. Naghiloo, Y. N. Joglekar, and K. W. Murch, Phys. Rev. Lett. 128, 160401 (2022).

[34] Q. Song, M. Odeh, J. Zúñiga-Pérez, B. Kanté, and P. Genevet, Science 373, 1133 (2021).

[35] H. Qin, Z. Yang, P.-S. Huang, X. Mu, S.-H. Huang, Y. Shi, W. Zhao, B. Li, J. Zhou, J. Zúñiga-Pérez et al., Nat. Commun. 16, 2656 (2025).

[36] Z. Yang, P.-S. Huang, Y.-T. Lin, H. Qin, J. Zúñiga-Pérez, Y. Shi, Z. Wang, X. Cheng, M.-C. Tang, S. Han et al., Nat. Commun. 15, 232 (2024).

[37] J. Doppler, A. A. Mailybaev, J. Böhm, U. Kuhl, A. Girschik, F. Libisch, T. J. Milburn, P. Rabl, N. Moiseyev, and S. Rotter, Nature (London) 537, 76 (2016).

[38] Z. Feng and X. Sun, Phys. Rev. Lett. 129, 273601 (2022).

[39] W. Liu, Y. Wu, C.-K. Duan, X. Rong, and J. Du, Phys. Rev. Lett. 126, 170506 (2021).

[40] Q. Liu, S. Li, B. Wang, S. Ke, C. Qin, K. Wang, W. Liu, D. Gao, P. Berini, and P. Lu, Phys. Rev. Lett. 124, 153903 (2020).

[41] A. Li, W. Chen, H. Wei, G. Lu, A. Alù, C.-W. Qiu, and L. Chen, Phys. Rev. Lett. 129, 127401 (2022).

[42] I. I. Arkhipov, F. Minganti, A. Miranowicz, Ş. K. Özdemir, and F. Nori, Phys. Rev. Lett. 133, 113802 (2024).

[43] K. Li, S. Wang, H. Chen, J. Zeng, and J. Wang, Chin. Optic. Lett. 23, 111301 (2025).

[44] M. S. Ergoktas, S. Soleymani, N. Kakenov, K. Wang, T. B. Smith, G. Bakan, S. Balci, A. Principi, K. S. Novoselov, S. K. Ozdemir et al., Science 376, 184 (2022).

[45] H. Nasari, G. Lopez-Galmiche, H. E. Lopez-Aviles, A. Schumer, A. U. Hassan, Q. Zhong, S. Rotter, P. LiKamWa, D. N. Christodoulides, and M. Khajavikhan, Nature (London) 605, 256 (2022).

[46] X. Shu, Q. Zhong, K. Hong, O. You, J. Wang, G. Hu, A. Alù, S. Zhang, D. N. Christodoulides, and L. Chen, Light Sci. Appl. 13, 65 (2024).

[47] \S. Baek, S. H. Park, D. Oh, K. Lee, S. Lee, H. Lim, T. Ha, H. S. Park, S. Zhang, L. Yang et al., Light Sci. Appl. 12, 87 (2023).

[48] W. He, Y. Hu, Z. Ren, S. Hu, Z. Yu, S. Wan, X. Cheng, and T. Jiang, Adv. Sci. 10, 2304972 (2023).

[49] M. Lawrence, N. Xu, X. Zhang, L. Cong, J. Han, W. Zhang, and S. Zhang, Phys. Rev. Lett. 113, 093901 (2014).

[50] W. He, S. Wan, Y. Zuo, S. Hu, Z. Ren, Z. Yu, D. Yang, X. Cheng, K. Xia, Y. Hu et al., Phys. Rev. Lett. 134, 106901 (2025).

[51] See Supplemental Material at http://link.aps.org/supplemental/10.1103/92m4-dhtc for details of theoretical calculations, parameters of the metasurface, details of complex-frequency analysis, supplementary simulation and experimental results, and other related discussions, which includes Refs. [52,53].

[52] M. Kang, J. Chen, and Y. D. Chong, Phys. Rev. A 94, 033834 (2016).

[53] S. Fan, W. Suh, and J. D. Joannopoulos, J. Opt. Soc. Am. A 20, 569 (2003).

[54] F. Guan, X. Guo, K. Zeng, S. Zhang, Z. Nie, S. Ma, Q. Dai, J. Pendry, X. Zhang, and S. Zhang, Science 381, 766 (2023).

[55] F. Guan, X. Guo, S. Zhang, K. Zeng, Y. Hu, C. Wu, S. Zhou, Y. Xiang, X. Yang, Q. Dai et al., Nat. Mater. 23, 506 (2024).

[56] S. Kim, S. Lepeshov, A. Krasnok, and A. Alù, Phys. Rev. Lett. 129, 203601 (2022).

[57] S. Kim, Y.-G. Peng, S. Yves, and A. Alù, Phys. Rev. X 13, 041024 (2023).

[58] S. Kim, A. Krasnok, and A. Alù, Science 387, eado4128 (2025).

[59] A. Krasnok, D. Baranov, H. Li, M.-A. Miri, F. Monticone, and A. Alú, Adv. Opt. Photonics 11, 892 (2019).

[60] Z. Sakotic, P. Stankovic, V. Bengin, A. Krasnok, A. Alú, and N. Jankovic, Laser Photonics Rev. 17, 2200308 (2023).

[61] V. Grigoriev, A. Tahri, S. Varault, B. Rolly, B. Stout, J. Wenger, and N. Bonod, Phys. Rev. A 88, 011803(R) (2013).

[62] V. Grigoriev, S. Varault, G. Boudarham, B. Stout, J. Wenger, and N. Bonod, Phys. Rev. A 88, 063805 (2013).

[63] R. Colom, E. Mikheeva, K. Achouri, J. Zuniga-Perez, N. Bonod, O.J. Martin, S. Burger, and P. Genevet, Laser Photonics Rev. 17, 2200976 (2023).

[64] E. Mikheeva, R. Colom, K. Achouri, A. Overvig, F. Binkowski, J.-Y. Duboz, S. Cueff, S. Fan, S. Burger, A. Alù et al., Optica 10, 1287 (2023).

[65] X. Zhao, J. Wang, W. Liu, L. Shi, and J. Zi, Phys. Rev. Lett. 135, 046203 (2025).

[66] X. Yin, T. Inoue, C. Peng, and S. Noda, Phys. Rev. Lett. 130, 056401 (2023).

[67] Y. Zeng, G. Hu, K. Liu, Z. Tang, and C.-W. Qiu, Phys. Rev. Lett. 127, 176101 (2021).

[68] M. Liu, C. Zhao, Y. Zeng, Y. Chen, C. Zhao, and C.-W. Qiu, Phys. Rev. Lett. 127, 266101 (2021).

[69] J. Fan, Z. Li, Z. Xue, H. Xing, D. Lu, G. Xu, J. Gu, J. Han, and L. Cong, Opto-Electron. Sci. 2, 230006 (2023).

[70] A. E. Miroshnichenko, S. Flach, and Y. S. Kivshar, Rev. Mod. Phys. 82, 2257 (2010).

[71] M. F. Limonov, M. V. Rybin, A. N. Poddubny, and Y. S. Kivshar, Nat. Photonics 11, 543 (2017).

[72] S. H. Mousavi, A. B. Khanikaev, J. Allen, M. Allen, and G. Shvets, Phys. Rev. Lett. 112, 117402 (2014).

[73] J. Y. Kim, J. Park, G. R. Holdman, J. T. Heiden, S. Kim, V. W. Brar, and M. S. Jang, Nat. Commun. 13, 2103 (2022).
