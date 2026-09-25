# Observation of Hybrid Degenerate Point in Projected Non-Hermitian Metasurfaces

Jingyi Chen $^{ID}$ , $^{*}$ Zhiling Zhou, $^{*}$ Yu Xiao $^{ID}$ , Nengyin Wang, Xu Wang $^{ID}$ , $^{\dagger}$ and Yong Li $^{ID}$ Institute of Acoustics, Tongji University, Shanghai 200092, People's Republic of China

(Received 11 April 2025; revised 29 July 2025; accepted 25 August 2025; published 11 September 2025)

Degeneracy appears ubiquitously in physical systems. Exceptional points, the non-Hermitian degeneracies, have unveiled numerous novel phenomena that have no counterparts in Hermitian degeneracies—diabolic points. Here, we observe a hybrid degenerate point (HP) in a projected non-Hermitian system, namely a subsystem obtained by projecting a metasurface characterized by a Hermitian scattering matrix. In the projected space, the metasurface is found naturally anchored at the merging point of two exceptional curves, each carrying opposite chirality. The HP exhibits a unique topology that encodes features of both Hermitian and non-Hermitian degeneracies, thereby exhibiting linear and square-root sensitivities simultaneously. We conceptually validate and experimentally confirm the projected HP using a passive and lossless acoustic metagrating, uncovering its unique singular behavior: the pronounced anisotropic sensitivity to perturbations. Our findings highlight the potential for exploring degenerate states via the projective Hilbert space and pave the way for extreme wave manipulation in open spaces.

DOI: 10.1103/9tdx-lcm5

Degenerate states arise in physical systems due to inherent symmetry. In Hermitian systems, degeneracy manifests as “diabolic points” (DPs) [1], indicating degenerate eigenvalues with orthogonal eigenvectors for the corresponding Hamiltonian. In contrast, non-Hermitian systems [2,3] host exotic degenerate states known as exceptional points (EPs) [4], where eigenvalues and eigenvectors coalesce simultaneously, bringing about far-reaching physical consequences [5–9]. Compared to the linear splitting of eigenvalues at the DP as a function of perturbation strength ( $\varepsilon$ ), a system operating at an $N$ th-order EP ( $N$ -fold degeneracy) shows a splitting induced by the perturbation that scales as $\varepsilon^{1/N}$ [10–12]. Thus, under small perturbations, the sensitivity of a non-Hermitian system far exceeds that of a Hermitian one, providing promising possibilities for sensing [8,13]. However, the ultrahigh sensitivity requires extremely fine modulation of non-Hermitian parameters such as gain and loss [14,15].

Non-Hermitian modulations further allow for the evolution of degenerate states $[16–18]$ . Typically, two EPs carrying opposite chirality can annihilate into a DP $[19,20]$ , while two EPs with the same chirality may either merge into a higher-order EP $[17,21–23]$ , or a hybrid degenerate point (HP) $[17,24,25]$ . HPs have been demonstrated to extend the conventional understanding of non-Hermitian degeneracy $[6,26–28]$ by their anisotropic responses. However, observations of the evolution among these degenerate states (DPs, EPs, and HPs) are rather challenging owing to the grueling demand for precise and comprehensive control over the non-Hermitian parameter space, further compounded by the need for dynamic tunability $[19,27]$ . This becomes a stumbling block in the exploration and broader application of these degenerate states.

Although Hermitian systems have long been considered to host only trivial degeneracies—DPs—it is essential to consider that subsystems within these systems can be locally non-Hermitian, potentially encoding projected EPs. This raises an intriguing question: can a fully Hermitian system, treated as an integrated whole, exhibit EP-related phenomena despite such features being localized within its non-Hermitian subsystems? In this Letter, we explore this question through a passive (gain-free) and lossless (free of intrinsic loss) acoustic metagrating. While metagratings are typically non-Hermitian due to their interaction with the environment through wave scattering, our design ensures energy conservation before and after scattering, further supporting finely controlled distribution of energy flux across various scattering channels. As a result, the metagrating is described by a scattering matrix (S-matrix) that is both unitary and Hermitian. Intriguingly, under projection, the initially expected EP manifests not simply as an EP, but as an HP naturally anchored at the merging of two exceptional curves (ECs) with opposite chirality, without the need for deliberate tuning of non-Hermitian parameters. Building on existing work, this is, to the best of our knowledge, the first observation of an HP in open space, distinct from those realized in cavities or waveguides $[24,25,29]$ . A further point of distinction lies in our approach, which circumvents the need for intrinsic energy dissipation—a non-Hermitian parameter that is challenging to fine-tune—to realize an HP. Additionally, the HP observed here arises from the merging of two opposite-chiral EPs from two ECs, setting it apart from previously reported cases on a single EC involving the coalescence of either the same- [17,24,25] or opposite-chiral EPs [29,30]. We demonstrate that, rooted within the intricacies of the projected non-Hermiticity, the behavior of the metagrating encodes features of both Hermitian degeneracy (from the perspective of winding behaviors of the eigenvalues) and non-Hermitian degeneracy (from the perspective of the collapsed eigenvector space), thereby displaying distinct sensitivity to perturbations along different directions in the parameter space (Fig. 1).

![](images/bb5a6e16f2d8a91a8f83be7957c7eea93ba76dbef395d148d72261a81ec9354c.jpg)
FIG. 1. Degenerate states. Eigenvalue responses in the two-dimensional parameter space for (a) a DP in a Hermitian system, (b) an HP in a projected non-Hermitian system, and (c) an EP in a conventional non-Hermitian system. The splittings of eigenvalues are shown in gray projection planes.

We consider a metagrating functioning as a four-channel scattering system, as illustrated in Fig. 2(a) (see theoretical framework in Supplemental Material (SM) [31], Sec. A and design methods in Sec. B). These four independent channels span a four-dimensional space, with the scattering behavior characterized by a fourth-order S-matrix (see details on constructing the matrices in End Matter). In our design, the metagrating supports an asymmetric scattering behavior, functioning as a retroreflector $\left(\theta_{i}=45^{\circ},\theta_{r}=45^{\circ}\right)$ when sound waves impinge from the upper left and as a negative refractor $\left(\theta_{i}=-45^{\circ},\theta_{t}=-45^{\circ}\right)$ for waves from the upper right. The mirror symmetry of the metagrating ensures symmetric scattering behavior for waves incident from the upper and lower planes. By further leveraging the wave-manipulation capacity of metagrating to fine-tune the energy flux distribution among these channels, the system can be characterized by a unitary and Hermitian S-matrix $\left(\mathbf{S}_{I}\right)$ as shown in the inset of Fig. 2(a) (see details on designing a Hermitian $S_{I}$ in SM [31], Sec. B).

For the designed metagrating, we project $S_{I}$ onto a plane spanned by the upper two channels, resulting in a projected matrix $S_{I}^{\prime}$ , a block of $S_{I}$ [inset of Fig. 2(b)]. Notably, $S_{I}^{\prime} = S_{p}^{\prime} \odot S_{p}^{\prime *}$ , where $S_{p}^{\prime}$ is the S-matrix in sound pressure, $\odot$ denotes the Hadamard product, and \* means conjugate. $S_{p}^{\prime} := \left( \begin{array}{cc}s_{o} & s_{e}^{R}\\ s_{e}^{L} & s_{o} \end{array} \right)$ contains four scattering coefficients of sound pressure for different modes. The diagonal scattering coefficient $s_{o}$ represents ordinary (specular) reflections under left and right incidence, which is identical due to reciprocity, while the antidiagonal coefficients $s_{e}^{L}$ and $s_{e}^{R}$ represent the extraordinary retroreflections. Through the projection operation, $S_{I}^{\prime}$ , as a Jordan block, exhibits non-Hermitian degeneracy with a degenerate eigenvalue ( $\lambda_{1,2}=0$ ) and a defective eigenspace represented by $\vec{\nu}_{1,2}=[0,1]^{T}$ (T representing transposition). Obviously, such a non-Hermitian degeneracy also occurs for $S_{p}^{\prime}$ (see details in SM [31], Sec. C). However, our investigation below reveals that this is not an EP observed previously via a truly non-Hermitian system relying on either fine-tuned energy dissipation [6,27] or nonreciprocity [32], but a scattering HP found for the first time.

(a)
![](images/8ff4f88d012ac181c9f5ceac998c68f61f101bcc6af437cfc2797004bc8e6d59.jpg)

(c)
![](images/edcf50001792ed1bfdfb558b271f045141f3dfb390b996d532313b1a91966414.jpg)

(e)
![](images/88481744fa556da0d46ce80ad205c74696ffab2c1fb0849a0b03c3831c8065b9.jpg)

(f)
![](images/bc08ca103bee201ba5c729e0cc819aba381bba01bf619312795402c1a2662f6c.jpg)

![](images/e0165c866613090b7e5a25a859fd558725d174f38e8c28f4ec1527ec7f39ba23.jpg)

![](images/d7a858ba570088ee8b4ba7160f32ec096ddd14c23b261860bf53a0176340f224.jpg)

![](images/bd93e5559c27e7287833b896fb7a486f3881bdaa50ee0042ac38775618f7926b.jpg)

![](images/ba82db0e332ffa0d575d13da5473d7587c4a519b5d83b39de8dda84b00708c03.jpg)

![](images/43cbd9f37427d4ae13ad8318e20bb099da8255a24e129259cf58187075ad06b8.jpg)

![](images/579622c4aaf5b6e02492147c8909d1ea9248c4305a19a2118e3fda1a165ed117.jpg)
FIG. 2. Schematic illustration of the projected HP and its topological characteristics. (a) A four-channel metagrating characterized by a unitary and Hermitian scattering matrix $\mathbf{S}_I$ . (b) The projected matrix $\mathbf{S}_I'$ , as a block of $\mathbf{S}_I$ , holding the HP and characterizing the local behavior of the metagrating in the upper-half domain. (c) Evolution of non-Hermitian degenerate states in a 3D parameter space defined by $\varepsilon_1$ , $\varepsilon_2$ , and $\varepsilon_3$ . The HP is the merging of two ECs, indicated by the blue and red curves. (d) Logarithmic representation of the variations of coefficient $s_e^R$ on $\Lambda_1$ , $\Lambda_2$ , and $\Lambda_3$ along the dotted lines shown in (c). (e) Imaginary part of the eigenvalue surfaces on $\Lambda_1$ , $\Lambda_2$ , and $\Lambda_3$ . (f) The discriminant fields $\vec{D}$ (arrows) and their norm (color map) on $\Lambda_1$ , $\Lambda_2$ , and $\Lambda_3$ .

To reveal the uniqueness of the degeneracy encoded in the projected non-Hermitian system, we calculate its trajectory in a three-dimensional (3D) parameter space defined by the relative deviation of the depths of two rigid-end grooves $(\pmb{\varepsilon}_{1,2})$ and the width of the connecting channel $(\pmb{\varepsilon}_{3})$ of a metaunit (see details in SM [31], Sec. D). As illustrated in Fig. 2(c), the HP hosted by the metagrating is naturally the merging point of two ECs anchored at the origin of this parameter space. By tuning the virtual loss $(\pmb{\varepsilon}_{3})$ [33], we focus on the states before, at, and after the emergence of the HP on three planes $(\Lambda_{1}, \Lambda_{2}, \text{and } \Lambda_{3})$ , as shown in Fig. 2(c). Figure 2(d) confirms the approaching, merging, and resplitting of the paired EPs, evidenced by the zeros of $s_{e}^{R}$ that signify the occurrence of the non-Hermitian degeneracy. Physically, the projected HP, is underpinned by the symmetry of the metasurface (see details of the symmetry requirement in SM [31], Sec. E).

We further calculate the topological structures of the eigenvalue surfaces on these planes [Fig. 2(e)]. The projected matrix $S_{p}^{\prime}$ on these planes shows two-sheet manifolds with distinct topologies, featuring either a pair of branch-point singularities (EPs) connected by an imaginary Fermi arc [20] on $\Lambda_{1}$ and $\Lambda_{3}$ [Fig. 2(e), the upper and lower panels] or an isolated HP on $\Lambda_{2}$ (the middle panel). The distinct topologies of the EP and HP are further confirmed by the eigenvalue evolutions via adiabatically tuning the system encircling the singularities along closed loops. Encircling an EP exchanges the eigenvalues, and the system returns to its original eigenstate after only two cycles of evolution [Fig. 2(e), the upper panel]—a symbol of the “topological half-integer charge” [16] carried by a standard second-order EP. In contrast, encircling the HP along loop $\Gamma_{2}$ returns an eigenstate to itself (the middle panel), similar to encircling the paired EPs shown in the lower panel (see numerical simulation results in SM [31], Sec. F).

Unlike DPs, non-Hermitian degeneracies inherently carry chirality [34], which can be quantified by discriminant fields, $\vec{D} = \nabla_{\vec{\epsilon}}[\mathrm{Im}(\ln \Omega)]$ , where $\Omega$ is the discriminant of the characteristic polynomial of $\mathbf{S}_p'$ (see the detailed mathematical discussion of discriminant fields in SM [31], Sec. G). Integrating $\vec{D}$ over a closed loop enclosing degenerate points produces a topological-invariant discriminant number, which determines the chirality [35-37]. On $\Lambda_{1}$ and $\Lambda_{3}$ planes, the paired EPs enclose two nonzero vorticity fluxes with opposite winding directions, indicating opposite chirality [Fig. 2(f), the upper and lower panels]. In contrast, the HP on the $\Lambda_{2}$ plane represents the merging of EPs with opposite chirality, leading to zero net vorticity (the middle panel). Notably, before (the lower panel) and after (the upper panel) the fusion of paired EPs into an HP, no chirality exchange occurs between $\mathrm{EP}_1$ and $\mathrm{EP}_2$ , implying that these two ECs do not cross during the evolution, but are tangent to each other; see the topologies of the projected HP in different parameter planes in SM [31], Sec. H.

Figures 3(a) and 3(b) show the calculated logarithmic amplitude and phase of $s_e^R$ on planes $\Lambda_2$ and $\Lambda_3$ (see results for $s_o$ and $s_e^L$ in SM [31], Sec. I). On plane $\Lambda_2$ , at the HP, both $s_o$ and $s_e^R$ approach zero at the origin ( $\pmb{\varepsilon}_1 = 0, \pmb{\varepsilon}_2 = 0$ ), signifying singular points with phases that cannot be defined. However, encircling $s_o$ results in a phase accumulation of $\pm 2\pi$ (SM [31], Fig. S11), while no phase accumulation occurs for $s_e^R$ [Fig. 3(a), the right panel]. The unique phase diagram around $s_e^R$ underpins the occurrence of the HP on the $\Lambda_2$ plane. In contrast, on plane $\Lambda_3$ , the original zero of $s_e^R$ splits into a pair of zeros [Fig. 3(b)], explaining the paired EPs observed on the $\Lambda_3$ plane. Encircling each of these zeros results in a phase accumulation of $\pm 2\pi$ , while encircling both yields no phase accumulation, which confirms that the HP on plane $\Lambda_2$ comes from the merging of EPs on plane $\Lambda_3$ .

The unique topology of the HP endows the metagrating with both robustness and sensitivity (see details in SM [31], Sec. J). Importantly, this results in an interesting consequence: anisotropic sensitivity response of asymmetric scattering behavior. We calculate the splitting of imaginary eigenvalues along different directions, as illustrated by the gray dashed line in Fig. 3(a), with the perturbation strength $\varepsilon$ (see details in SM [31], Sec. K). Along the direction $\overrightarrow{e_s}$ , the splitting follows a $\sqrt{\varepsilon}$ dependence [Fig. 3(c), the left panel], which is a symbol of a standard second-order EP. In contrast, when the metagrating is disturbed along another direction $\overrightarrow{e_l}$ , the eigenvalue splitting is linear (the right panel)—a feature long sought to be closely tied to a Hermitian system operating around a DP. Therefore, the HP manifests itself as an anisotropic EP, exhibiting distinct dispersion relations along different directions in the parameter plane $\Lambda(\pmb{\varepsilon}_1,\pmb{\varepsilon}_2)$ . Such anisotropy can be further confirmed by the phase rigidities ( $r_i$ ) [21,38,39], characterizing the orthogonality of the $i$ th eigenvector (see details in SM [31], Sec. L). At the HP, the phase rigidity approaches zero, signifying the deficiency of the eigenspace [Fig. 3(d)]. When slightly deviating from this point, the phase rigidity displays either square-root [Fig. 3(d), the left panel] or linear [Fig. 3(d), the right panel] sensitivity, depending on the direction of the perturbation.

(a) $_{0}$
![](images/b54c309c677ab46c7d942a2bd704b5a95d443be82bd446a36a3db4943fac9af7.jpg)
(c)

![](images/65f8426bd013ed056cf4358d42ac4e694a255418e850ba041f1b6a5eefc24a86.jpg)

![](images/a2c1b8e4bde40263df2e8be9fbfbf8d935bd8287a35465bc99edf81a636ec7ca.jpg)

(b)
![](images/cd0020f537152c533517cc42c0fdda0767a8ac650c311ce5fc41feaf939ddf9d.jpg)

![](images/ec46c6b71926ed99629401560625249853a4178aa7907be5c1c21d00c88156c2.jpg)
(d).

![](images/7c462fa1e2afff3836b71bec11ee0e690a0be065f62d47c0d0f4ce311bfedeb2.jpg)

![](images/5906103018894e4a5c19abbacabfbf6b8839c730dda22b8850e42ea159c0f91e.jpg)

![](images/6080d7ba70d094bdaf1814dd5a8e198785b52ada4e73603d576e9ef2eba5305a.jpg)
FIG. 3. (a),(b) Logarithmic amplitude (left panels) and phase (right panels) plots of $s_e^R$ around (a) the HP on plane $\Lambda_2$ and (b) paired EPs on $\Lambda_3$ . (c) Imaginary eigenvalues through the HP along directions $\overrightarrow{e_l}$ and $\overrightarrow{e_s}$ indicated by the gray dashed lines in (a). (d) Phase rigidity of eigenvectors along the corresponding directions. The insets show the logarithmic plots.

Although the non-Hermitian subsystem has been verified to hold the projected HP, an interesting question is how this relates to the scattering behavior of the whole metagrating. To uncover this, we perform simulations of two sets of metagratings, each perturbed along $\vec{e}_{s}$ and $\vec{e}_{l}$ , respectively. Perturbation strength ( $\varepsilon$ ) is incrementally increased for each set. The simulated results of the splitting eigenvalues [Fig. 4(b), solid curves] confirm the pronounced anisotropic sensitivity of the metagrating at the projected HP.

Experiments are carried out to verify these findings (see details about experiments in the SM [31], Sec. M). As illustrated in Fig. 4(a), the experimental setup contains a two-dimensional (2D) waveguide surrounded by a layer of melamine foam to suppress the undesired reflection

(a)

from its lateral boundaries. The operating frequency $f = 3430$ Hz. With a step of $0.01 \, \mathrm{m}$ , two scanning regions of $0.2 \times 0.2 \, \mathrm{m}^2$ were measured, corresponding to the left and right ports [marked by the boxes in Figs. 4(c)-4(e)].

Two sets of samples, fabricated using 3D printing technology, are perturbed along $\vec{e}_l$ and $\vec{e}_s$ , respectively. For each set, we selected six representative cases with $\varepsilon$ gradually increased from 0.025 to 0.15. An unperturbed metagrating is tested for comparison. Each sample underwent ten independent measurements to ensure experimental repeatability (see details of measurements and spectral analyses in SM [31], Sec. N). The experimental results, represented by the dots (the statistical mean of multiple measurements) and the error bars, are closely aligned with the simulation data [Fig. 4(b)]. Figures 4(c)–4(e) further show the measured scattering fields of unperturbed and two perturbed (with $\varepsilon = 0.15$ ) metagratings, which agree well with the simulation results. It can be observed that perturbations along $\vec{e}_s$ result in a more “chaotic” scattering field [Fig. 4(e)], where those undesired diffraction modes are no longer suppressed. In contrast, perturbations along $\vec{e}_l$ maintain a more ordered scattering pattern, underscoring the directional robustness of the metagrating at the projected HP [Fig. 4(d)]. Figures 4(f)–4(h) quantitatively compare the simulated and experimental data along the centerlines of the scanning regions [white dashed lines shown in Fig. 4(c)]. The simulated and measured results confirm the strongly anisotropic sensitivity behavior (see additional experimental results in SM [31], Sec. N).

(b)
![](images/4a8f80edcfaec82ceb0971a3f263ef9ded388dba1109309e5a91dffdcdc5c1eb.jpg)

![](images/5e77c01ff78de56cac2bd61cd44091f387a04717f16b465d57bf74ad2088ec65.jpg)

![](images/996b4bf9401c7b137dc26df10b05b96d21cbbae9692175f36f1ba78162c7c6a9.jpg)

(d)
![](images/940916b5765157b4edad15218294c6698795940d2429f4d85da8a87b47c2e22d.jpg)

![](images/4010028728e405cbd7961697e5e6053fc34f57c19554b7ef89e449eabc4220cc.jpg)
FIG. 4. Experimental verification of the scattering behavior of the metagrating at the projected HP. (a) The experimental platform, with the inset showing three periods of the fabricated metagrating sample. (b) Anisotropic sensitivity along two different directions when perturbations are introduced. The blue line represents square-root sensitivity along direction $\overrightarrow{e_s}$ , whereas the red line signifies linear sensitivity along direction $\overrightarrow{e_l}$ . The dots with error bars depict the experimental data. (c)-(e) Simulated and experimental results of the unperturbed case (c) and the perturbed cases (with $\varepsilon = 0.15$ ) along the linear (d) and square-root (e) directions, respectively. The upper and lower panels show the cases for left and right incidences, respectively. White arrows indicate the propagating direction, while the less-visible arrows indicate the suppressed retroreflection waves. (f)-(h) Simulated (lines) and measured (dots) sound pressure along the centerline of the scanning regions I and II.

To conclude, we have conceptually demonstrated and experimentally validated a scattering HP in a projected non-Hermitian system, where the HP naturally occurs without the need for tuning any non-Hermitian parameters. Notably, although the HP is a local singular feature embedded in the subsystem, the entire metasurface, as a whole, exhibits related enhanced sensitivity and even anisotropic sensitivity when subjected to perturbations. Considering the ubiquity that a subsystem of a Hermitian system can be non-Hermitian, our findings may provide an important basis for the exploration of non-Hermitian concepts in a more general framework, shedding light on the potential of enhanced sensing by the local behavior of Hermitian devices. For example, the coexistence of square-root and linear responses may enable compensation and calibration mechanisms for wave manipulation: distinct from conventional EP-based sensors that rely solely on high sensitivity at the cost of robustness, the linear direction in our system may provide a trade-off that allows for stable operation under external fluctuations.

Acknowledgments—This work was supported by the National Key R&D Program of China (Grants No. 2020YFA0211400 and No. 2022YFA1404400), the Scientific Research Innovation Capability Support Project for Young Faculty (Grant No. ZYGXQNJSKYCXNLZCXM-D8), the National Science Foundation of China (Grant No. 12474463), the Shanghai Pilot Program for Basic Research, Xiaomi Young Talents Program, and the Fundamental Research Funds for the Central Universities.

Data availability—The data that support the findings of this Letter are not publicly available. The data are available from the authors upon reasonable request.

[1] M. V. Berry and M. Wilkinson, Diabolical points in the spectra of triangles, Proc. R. Soc. A 392, 15 (1984).

[2] R. El-Ganainy, K. G. Makris, M. Khajavikhan, Z. H. Musslimani, S. Rotter, and D. N. Christodoulides,

Non-Hermitian physics and PT symmetry, Nat. Phys. 14, 11 (2018).

[3] X. Wang, R. Dong, Y. Li, and Y. Jing, Non-local and non-Hermitian acoustic metasurfaces, Rep. Prog. Phys. 86, 116501 (2023).

[4] E. J. Bergholtz, J. C. Budich, and F. K. Kunst, Exceptional topology of non-Hermitian systems, Rev. Mod. Phys. 93, 015005 (2021).

[5] B. Peng, S. k. Özdemir, M. Liertzer, W. Chen, J. Kramer, H. Yılmaz, J. Wiersig, S. Rotter, and L. Yang, Chiral modes and directional lasing at exceptional points, Proc. Natl. Acad. Sci. U.S.A. 113, 6845 (2016).

[6] X. Wang, X. Fang, D. Mao, Y. Jing, and Y. Li, Extremely asymmetrical acoustic metasurface mirror at the exceptional point, Phys. Rev. Lett. 123, 214302 (2019).

[7] Z. Zhou, B. Jia, N. Wang, X. Wang, and Y. Li, Observation of perfectly-chiral exceptional point via bound state in the continuum, Phys. Rev. Lett. 130, 116101 (2023).

[8] W. Chen, S. K. Özdemir, G. Zhao, J. Wiersig, and L. Yang, Exceptional points enhance sensing in an optical microcavity, Nature (London) 548, 192 (2017).

[9] Q. Song, M. Odeh, J. Zúñiga-Pérez, B. Kanté, and P. Genevet, Plasmonic topological metasurface by encircling an exceptional point, Science 373, 1133 (2021).

[10] T. Katō, Perturbation Theory for Linear Operators (Springer, Berlin, 1995).

[11] J. Wiersig, Enhancing the sensitivity of frequency and energy splitting detection by using exceptional points: Application to microcavity sensors for single-particle detection, Phys. Rev. Lett. 112, 203901 (2014).

[12] J. Wiersig, Sensors operating at exceptional points: General theory, Phys. Rev. A 93, 033809 (2016).

[13] H. Hodaei, A. U. Hassan, S. Wittek, H. Garcia-Gracia, R. El-Ganainy, D. N. Christodoulides, and M. Khajavikhan, Enhanced sensitivity at higher-order exceptional points, Nature (London) 548, 187 (2017).

[14] X. Zhu, H. Ramezani, C. Shi, J. Zhu, and X. Zhang, PT-symmetric acoustics, Phys. Rev. X 4, 031042 (2014).

[15] C. Shi, M. Dubois, Y. Chen, L. Cheng, H. Ramezani, Y. Wang, and X. Zhang, Accessing the exceptional points of parity-time symmetric acoustics, Nat. Commun. 7, 11110 (2016).

[16] H. Zhou, C. Peng, Y. Yoon, C. W. Hsu, K. A. Nelson, L. Fu, J. D. Joannopoulos, M. Soljačić, and B. Zhen, Observation of bulk Fermi arc and polarization half charge from paired exceptional points, Science 359, 1009 (2018).

[17] K. Ding, G. Ma, M. Xiao, Z. Q. Zhang, and C. T. Chan, Emergence, coalescence, and topological properties of multiple exceptional points and their experimental realization, Phys. Rev. X 6, 021007 (2016).

[18] B. Zhen, C. W. Hsu, Y. Igarashi, L. Lu, I. Kaminer, A. Pick, S.-L. Chua, J. D. Joannopoulos, and M. Soljačić, Spawning rings of exceptional points out of Dirac cones, Nature (London) 525, 354 (2015).

[19] F. Ding, Y. Deng, C. Meng, P.C.V. Thrane, and S.I. Bozhevolnyi, Electrically tunable topological phase transition in non-Hermitian optical MEMS metasurfaces, Sci. Adv. 10, eadl4661 (2024).

[20] M. Król, I. Septembre, P. Oliwa, M. Kędziora, K. Łempicka Mirek, M. Muszyński, R. Mazur, P. Morawiak, W. Piecek, P. Kula, W. Bardyszewski, P. G. Lagoudakis, D. D. Solnyshkov,

G. Malpuech, B. Piętka, and J. Szczytko, Annihilation of exceptional points from different Dirac valleys in a 2D photonic system, Nat. Commun. 13, 5340 (2022).

[21] W. Tang, X. Jiang, K. Ding, Y.-X. Xiao, Z.-Q. Zhang, C. T. Chan, and G. Ma, Exceptional nexus with a hybrid topological invariant, Science 370, 1077 (2020).

[22] K. Ding, Z. Q. Zhang, and C. T. Chan, Coalescence of exceptional points and phase diagrams for one-dimensional PT-symmetric photonic crystals, Phys. Rev. B 92, 235310 (2015).

[23] W. Tang, K. Ding, and G. Ma, Realization and topological properties of third-order exceptional lines embedded in exceptional surfaces, Nat. Commun. 14, 6660 (2023).

[24] K. Ding, G. Ma, Z. Q. Zhang, and C. T. Chan, Experimental demonstration of an anisotropic exceptional point, Phys. Rev. Lett. 121, 085702 (2018).

[25] X.-L. Zhang and C. T. Chan, Hybrid exceptional point and its dynamical encircling in a two-state system, Phys. Rev. A 98, 033810 (2018).

[26] X. Fang, N. Wang, W. Wu, W. Wang, X. Yin, X. Wang, and Y. Li, Extreme wave manipulation via non-Hermitian metagratings on degenerated states, Phys. Rev. Appl. 19, 054003 (2023).

[27] X. Fang, N. Gerard, Z. Zhou, H. Ding, N. Wang, B. Jia, Y. Deng, X. Wang, Y. Jing, and Y. Li, Observation of higher-order exceptional points in a non-local acoustic metagrating, Commun. Phys. 4, 271 (2021).

[28] T. Liu, X. Zhu, F. Chen, S. Liang, and J. Zhu, Unidirectional wave vector manipulation in two-dimensional space with an all passive acoustic parity-time-symmetric metamaterials crystal, Phys. Rev. Lett. 120, 124502 (2018).

[29] W. Tang, K. Ding, and G. Ma, Direct measurement of topological properties of an exceptional parabola, Phys. Rev. Lett. 127, 034301 (2021).

[30] H. Shen, B. Zhen, and L. Fu, Topological band theory for non-Hermitian Hamiltonians, Phys. Rev. Lett. 120, 146402 (2018).

[31] See Supplemental Material at http://link.aps.org/supplemental/10.1103/9tdx-lcm5 for theoretical model, details of the optimization procedure, parameters of the metasurface, details of simulations, discussions about the scattering properties at the singularities, mathematical definition of discriminant number, symmetry conditions for the emergence of the projected HP, and details of experiments' setup, fabricated experimental samples, and additional experiments.

[32] J. Zhang, J. Guo, H. Wang, D. Tang, and D. Shen, Single longitudinal mode lasing near the exceptional point in a fiber laser using a tunable isolator, Opt. Lett. 47, 2222 (2022).

[33] H. Zhou, M. Jiang, J. Zhu, Y. Li, Q. Li, Y. Wang, C. Qiu, and Y. Wang, Underwater scattering exceptional point by metasurface with fluid-solid interaction, Adv. Funct. Mater. 34, 2404282 (2024).

[34] W. Heiss and H. Harney, The chirality of exceptional points, Eur. Phys. J. D 17, 149 (2001).

[35] R. Su, E. Estrecho, D. Biegańska, Y. Huang, M. Wurdack, M. Pieczarka, A. G. Truscott, T. C. H. Liew, E. A. Ostrovskaya, and Q. Xiong, Direct measurement of a non-Hermitian topological invariant in a hybrid light-matter system, Sci. Adv. 7, eabj8905 (2021).

[36] Z. Yang, A. P. Schnyder, J. Hu, and C.-K. Chiu, Fermion doubling theorems in two-dimensional non-Hermitian systems for Fermi points and exceptional points, Phys. Rev. Lett. 126, 086401 (2021).

[37] Y.-X. Xiao, K. Ding, R.-Y. Zhang, Z. H. Hang, and C. T. Chan, Exceptional points make an astroid in non-Hermitian Lieb lattice: Evolution and topological protection, Phys. Rev. B 102, 245144 (2020).

[38] I. Rotter, A non-Hermitian Hamilton operator and the physics of open quantum systems, J. Phys. A 42, 153001 (2009).

[39] E. N. Bulgakov, I. Rotter, and A. F. Sadreev, Phase rigidity and avoided level crossings in the complex energy plane, Phys. Rev. E 74, 056204 (2006).

## End Matter

Details on constructing the scattering matrix—By adjusting the metagrating period D, we ensure that only two diffraction channels, belonging to zero and first orders, are allowable in both upper and lower planes under $45^{\circ}$ oblique incidence. According to the labeled channels illustrated in Fig. 2(a), such a four-port scattering system can be described by

$$
\left( \begin{array}{c} I _ {4} ^ {\text {out}} \\ I _ {3} ^ {\text {out}} \\ I _ {2} ^ {\text {out}} \\ I _ {1} ^ {\text {out}} \end{array} \right) = \mathbf {S} _ {I} \left( \begin{array}{c} I _ {1} ^ {\text {in}} \\ I _ {2} ^ {\text {in}} \\ I _ {3} ^ {\text {in}} \\ I _ {4} ^ {\text {in}} \end{array} \right), \quad \mathbf {S} _ {I} := \left( \begin{array}{c c c c} T _ {e} ^ {U L} & T _ {0} ^ {U R} & R _ {0} ^ {D R} & R _ {e} ^ {D L} \\ T _ {0} ^ {U L} & T _ {e} ^ {U R} & R _ {e} ^ {D R} & R _ {0} ^ {D L} \\ R _ {0} ^ {U L} & R _ {e} ^ {U R} & T _ {e} ^ {D R} & T _ {0} ^ {D L} \\ R _ {e} ^ {U L} & R _ {0} ^ {U R} & T _ {0} ^ {D R} & T _ {e} ^ {D L} \end{array} \right).\tag{A1}
$$

In Eq. (A1), for states before and after scattering, I represents sound intensity in each channel, the superscripts distinguish between the input and the output, and the subscripts mark the channels labeled in Fig. 2(a); for the scattering matrix $S_{I}$ , T/R represents intensity coefficients of transmission and reflection, superscripts U/D denote incidence from the upper and lower sides, L/R indicate incidence from the left and right of the surface normal, and subscripts o/e represent ordinary (zero order) and extraordinary (first order) scattering. The rationale for constructing $S_{I}$ in this way is based on the fact that the trivial matrix (identity matrix) corresponds to the trivial scattering case (in absence of the metagrating). All grooves in the metaunit are sufficiently wide so that the thermoviscous effect and inherent loss can be neglected. The passive and lossless nature of the metagrating ensures a unitary evolution of sound scattering.
