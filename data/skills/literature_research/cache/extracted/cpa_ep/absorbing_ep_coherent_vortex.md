# Absorbing Exceptional Point of Coherent Vortex Beams

Yong Li
yongli@tongji.edu.cn

Tongji University https://orcid.org/0000-0001-8049-9128

Hua Ding
Tongji University

Xu Wang
Tongji University https://orcid.org/0000-0001-5114-4897

## Article

Keywords: Coherent perfect absorption, Exceptional point conversion, Sensitivity enhancement, Chiral waves

Posted Date: September 29th, 2025

DOI: https://doi.org/10.21203/rs.3.rs-7674170/v1

License: © This work is licensed under a Creative Commons Attribution 4.0 International License. Read Full License

Additional Declarations: There is NO Competing Interest.

# Absorbing Exceptional Point of Coherent Vortex Beams

Hua Ding $^{1\dagger}$ , Quansen Wang $^{1\dagger}$ , Xu Wang $^{1*}$ , Yong Li $^{1*}$

$^{1}$ Institute of Acoustics, Tongji University, Shanghai, China.

\*Corresponding author(s). E-mail(s): xuwang@tongji.edu.cn; yongli@tongji.edu.cn; †These authors contributed equally to this work.

## Abstract

Coherent perfect absorption at exceptional point (CPA-EP) is of great interest in wave systems for its unique absorption behavior. However, most CPA-EPs achieved so far are confined to plane-wave systems, limiting their versatility. Here, we realize CPA-EPs in chiral spaces via topological vortex beams, using metamaterials with tailored symmetry. We show that chiral CPA-EPs arise from the delicate interplay of mode channels encoding different topological charges (TCs), and TC channels supported by orbital angular momentum significantly boost the channel capacity of CPA-EPs. Notably, through rotational operations, the chiral system enables dynamic switching of absorption capacity and state transformation of CPA-EPs, in which the introduced geometric phase provides a crucial route for enhancing sensitivity via lower-order EPs. Crucially, we further demonstrate the unique coexistence of sensitivity (in EP state) and robustness (in CPA state) within the chiral space. Our findings broaden CPA-EP research and enable advanced devices on chiral wave systems.

Keywords: Coherent perfect absorption, Exceptional point conversion, Sensitivity enhancement, Chiral waves

## 1 Introduction

Exceptional points (EPs) in non-Hermitian systems, manifesting as the coalescence of eigenvalues and eigenvectors $[1, 2]$ , provide a crucial platform for versatile counterintuitive phenomena such as unidirectional reflectiveness $[3–6]$ , extreme asymmetric scattering $[7–9]$ , nontrivial topological behaviors $[10–13]$ , etc. Particularly, systems operating in the vicinity of EPs exhibit pronounced sensitivity to perturbations $[14–17]$ , demonstrating their unique capability in high-precise sensing applications $[15, 18]$ . Thus, the concept of EPs as well as their potential applications attract extensive studies crossing various physical systems, encompassing microwaves $[19–22]$ , plasmonics $[14, 23]$ , photonics $[1, 24]$ , and acoustics $[6, 25–27]$ . The concept of EP also empowers coherent perfect absorption (CPA)—a special absorption state driven by the interplay of multi-channel inputs that causes energy trapping and dissipation $[20, 28, 29]$ . The resultant CPA-EPs, as a special absorbing EP, exhibit distinct properties in stark contrast to those resonance EPs suffered from intrinsic instability $[20, 30, 31]$ .

Previously observed CPA-EPs typically rely on plane waves, whose sensitivity, however, is limited by the number of available scattering channels. This intrinsic constraint inevitably necessitates complex mechanisms for enhancing EP sensitivity, including higher-order EPs[17, 26] and nonlinear effects[32, 33]. In contrast, structured waves, inherently carrying richer transmission channels and additional symmetries owing to their higher degree of freedom, can remarkably broaden the versatility of CPA-EPs and open new routes for sensitivity enhancement. Vortex beams, a specialized class of chiral beams featuring unique spiral phase distribution, carry orbital angular momentum (OAM) and encode topological charges (TCs) [34, 35]. The infinite-dimensional Hilbert space inherent to OAM transcends the limitations of finite-state architectures utilizing plane waves [36]. Thus, vortices serve as a powerful means for novel physical phenomena and intriguing functionalities [34, 37–39]. In particular, OAMs carried by vortices pave a promising way to unlock desired multidimensional channels and overcome the limitations of plane-wave-based CPA-EPs.

In this work, we report the chiral CPA-EP (CCPA-EP), which is a unique absorbing degeneracy state that suppresses reflections in all TC channels. Such CCPA-EP arises from the delicate interplay between paired vortices carrying opposite TCs. The singular scattering behavior enhances the sensitivity of CCPA-EP in a special way. To demonstrate this, we design metamaterials with tailored rotational symmetry and broken mirror symmetry. Simulations and experiments confirm that perfect absorption occurs when vortex modes with opposite TCs are coherently excited. At CCPA-EP, the system's response in terms of absorption shows sensitivity to both the input signal as well as the matematerial itself, providing two distinct yet compensated phase-based mechanisms for absorption modulation. The former enables flexible control of absorption for making the EP state hidden, while the geometric phase induced by the latter facilitates the transformation of CCPA-EPs. Notably, this geometry phase in chiral space boosts the manipulation capacity of CPA-EPs, allowing a significantly enhanced sensitivity via a lower-order EP, in contrast to existing routes relying on higher-order EPs that may inevitably involve the issue of growing system complexity.

![](images/a91e98448d2eecfea5d9eb8f7d5349e7bd872693a053617bb54ecae4445122c5.jpg)
Fig. 1 Schematics of a chiral system at the CPA-EP. a, CCPA-EP: both conditions—degeneracy of eigenvalues and zeroing of eigenvalues—coinciding at the same real frequency. b, Perfect absorption via coherent excitations of vortex modes $|+\rangle$ and $|- \rangle$ with a delicate phase difference. In contrast, a single-mode incidence of either $|+\rangle$ or $|- \rangle$ yields strong reflections. c, Rotation operation in chiral space induced geometric phase for CCPA-EP state switching. Specific coherent excitation causes one of the EPs to exhibit perfect absorption, while the other EP is completely hidden in the absorption spectrum.

## 2 Results

As conceptually illustrated in Fig. 1a, the CCPA-EP features the coalescence of scattering zeros positioning on the real frequency axis, where scattering zeros are eigenvalues of zero of the scattering matrix characterizing the reflection behavior of different-mode vortices. Such a unique degenerate state underscores perfect absorption that occurs only under the coherent incidence of vortices carrying different TC numbers. In contrast, strong reflections (containing different modes) are generated when the system is excited by a single-mode vortex (Fig. 1b). As will be elucidated in the latter part of this paper, the inherent symmetry in chiral space leads to the emergence of CCPA-EPs in pairs, with their eigenvectors being mutually orthogonal. This results in a coherent input corresponding to one CCPA-EP eigenvalue inducing perfect absorption, while the other CCPA-EP remains completely hidden in the absorption spectrum (Fig. 1c). This unique behavior in chiral space enables state switching between the paired CCPA-EPs — along with switching of the corresponding absorption spectrum — through the introduction of a geometric phase by rotating the metamaterial by an angle $\beta$ .

In this work, acoustic vortex beams propagating in a cylindrical waveguide along the z-axis are represented as $p(m,n) = A_{m,n}J_{m}(k_{m,n}r)e^{ik_{z}z}e^{im\varphi}$ , where $A_{m,n}$ donates the pressure amplitude of mode $(m,n)$ , $J_{m}(k_{m,n}r)$ is the m-th order Bessel function, $k_{m,n}$ and $k_{z} = \sqrt{k_{0}^{2} - k_{m,n}^{2}}$ are transverse and axial wave numbers, and $k_{0}$ is the wavenumber in free space. The mode of vortices can be characterized by the TC number m ( $m = 0, \pm1, \cdots, \pm\infty$ ) and radial index n ( $n = 0, 1, \cdots, \infty$ ). Without loss of generality, here vortex modes carrying opposite TCs (TC = $\pm1$ ) are chosen, denoted as states $|+\rangle$ and $|- \rangle$ , respectively.

In line with the definition of chirality, a $2 \times 2$ scattering matrix is formulated, i.e. $S := \begin{pmatrix} r_{++} & r_{+-} \\ r_{-+} & r_{--} \end{pmatrix}$ , which connects the incident and reflected vortices in this two-port

$$
\binom{A _ {+} ^ {\mathrm{o}}}{A _ {-} ^ {\mathrm{o}}} = \mathbf {S} \binom{A _ {+} ^ {\mathrm{i}}}{A _ {-} ^ {\mathrm{i}}},\tag{1}
$$

where A represents complex amplitude of vortices, with the superscript i (o) denoting incident (reflected) waves and the subscript $+(-)$ indicating the state $|+\rangle(|-\rangle)$ . Note that the anti-diagonal elements of S, $r_{+-}$ and $r_{-+}$ , represent ordinary reflections, which undergo OAM inversion and are equal to each other owing to reciprocity [40]. The diagonal ones ( $r_{++}$ and $r_{--}$ ) are coefficients of extraordinary reflections, indicating anomalous chiral wave scattering preserving the chirality. Acoustic metamaterials, owing to their remarkable ability to manipulate wave-matter interactions at the subwavelength level, facilitate the engineering of the chiral wave scattering behavior by tuning the four elements of S towards a CCPA-EP. In principle, for a scattering system described by such a 2nd-order matrix S, a CCPA-EP state necessitates

$$
r _ {+ +} = - r _ {- - } = \pm i r _ {+ - (- +)}.\tag{2}
$$

Equation (2) permits the construction of a pair of CCPA-EPs (see the plus-minus sign of the equation). The paired CCPA-EPs feature the same degenerate eigenvalue $(\lambda_{1,2} = \lambda_{1,2}^{\prime} = 0)$ and orthogonal degenerate eigenvectors $[v_{1,2} = (-i, 1)^{\mathrm{T}}$ and $v_{1,2}^{\prime} = (i, 1)^{\mathrm{T}}$ , T representing conjugate]. Note that the two eigenvectors of S represent two specific superpositions of states $|+\rangle$ and $|- \rangle$ for which the waveform remains invariant after scattering.

It is worth noting that the reflection behavior described by Eq. (1) indicates that the system output is the hybridization of input vortices, i.e. $A_{+}^{o} = r_{++}A_{+}^{i} + r_{+-}A_{-}^{i}$ and $A_{-}^{o} = r_{-+}A_{+}^{i} + r_{--}A_{-}^{i}$ . Considering the identical amplitudes in the four elements (only having difference in phases) as specified in Eq. (2), the absorption coefficient $\alpha := 1 - \frac{\left|A_{+}^{o}\right|^{2} + \left|A_{-}^{o}\right|^{2}}{\left|A_{+}^{i}\right|^{2} + \left|A_{-}^{i}\right|^{2}}$ can be written as

$$
\alpha = 1 - r ^ {2} \left[ 2 + \left(e ^ {i \varDelta_ {+}} + e ^ {i \varDelta_ {-}}\right) e ^ {i \varDelta_ {\mathrm{i}}} \right],\tag{3}
$$

where r is the amplitude of these coefficients, $\Delta_{+}(\Delta_{-})$ denotes the phase difference between the two coefficients contributing output state $|+\rangle(|-\rangle)$ , i.e. $\Delta_{+}:=arg(r_{++})-arg(r_{+-})$ and $\Delta_{-}:=arg(r_{-+})-arg(r_{--})$ , and $\Delta_{i}$ represents the relative phase between the incident states, i.e. $\Delta_{i}:=arg(A_{-}^{i})-arg(A_{+}^{i})$ . Equation (3) indicates that, for such a system operating at the CCPA-EP (with a specific $\Delta_{\pm}$ ), coherent excitation of both vortices with balanced input power and the optimized relative phase ( $\Delta_{i}$ ) will result in perfect absorption, i.e. $\alpha = 1$ . Alternatively, perfect absorption can only be achieved by the synergy of inputs ( $\Delta_{i}$ ) and system response ( $\Delta_{\pm}$ ). Notably, CCPA-EP predicted by Eq. (3) indicates that perfect absorption can not only be satisfied by high dissipation (r = 0), a common way for absorber design, but also be fulfilled by setting the terms in the square bracket of Eq. (3) to zero, manifesting complete sound trapping resulted from the interplay of mode channels with different TCs.

![](images/a87dc63053fe8bbf70242e82ee74f61d6841463661ce417833d831a636838164.jpg)

![](images/a7dbcc0f97b1d378f7575c62b16835777468246f92762fce07ecda0513e52cdb.jpg)
Fig. 2 Chiral metamaterials at CCPA-EP. a, Schematics of a metamaterial with $C_{2}$ rotational symmetry. The upper panel shows the air regions bounded by the metamaterial and the lower panel shows the metamaterial. b, Trajectories of eigenvalues $\lambda_{i}$ (i = 1, 2) on the complex plane. The cross-sectional color map represents the acoustic energy reflectivity R, which is related to the system absorption as $\alpha := 1 - |R|^{2}$ . The blue dot represents the (degenerate) zeros of the system corresponding to the absorbing EP, i.e. CCPA-EP, while the red dot represents the (degenerate) poles, indicating the resonant EP.

The capabilities of metamaterials on flexible wave manipulation facilitate the realization of CCPA-EP. Here, we design a metamaterial consisting of two toroidal cavities. As illustrated in Fig. 2a, the designed metamaterial hosts a $C_2$ rotational symmetry (with respect to the $z$ axis) and broken mirror symmetry (with respect to the $x$ and $y$ axes). Such a symmetry configuration effectively suppresses undesired modes while streamlining the design through modal purity enhancement [34]. In this work, the radius $R_0$ of the cylinder waveguide is chosen as 50 mm and the operating frequency is set at $f_0 = 3000\mathrm{Hz}$ . The span angles $(\theta_i)$ , radii $(R_i)$ , and depths $(L_i)$ of the ith cavity $(i = 1,2)$ , together with the cavity spacing $(\Delta \theta)$ , are tunable to customize the four elements to construct the sought-after scattering matrix S hosting a CCPA-EP (see Supplementary Section 1 for detailed parameters of the metamaterial).

Akin to traditional EPs, non-Hermitian perturbations can drive the evolution of the system away from CCPA-EP. To characterize this, we introduce a loss perturbation via the imaginary part of the effective sound speed $(c_{\mathrm{i}})$ in the metamaterial. Note that a positive or negative $c_{i}$ represents a loss or gain medium, respectively [1]. As illustrated in Fig. 2b, both the degeneracy of eigenvalues, necessary for EP, and the zeroing of eigenvalues, necessary for CPA, coincide at the same real frequency, signifying the sought-after CCPA-EP (the crossing of two eigenvalue lines at the blue dot indicating zero). Note that such a degeneracy also occurs for the resonance state (the red dot), which is consistent with that found previously [20]. The unique CCPA-EP, observed in chiral systems for the first time, presents intriguing EP-related behavior not previously observed, as will be demonstrated experimentally below.

The designed metamaterial provides a reflection coefficient $r_{++} = 0.35 - 0.34i$ at the operating frequency $f_{0}$ . Note that the prerequisite of CCPA-EP by Eq. (2) endows the system with a unique characteristic — equal reflectivities across all channels, i.e.

d
![](images/30e39081b0353111727ed9a1b97c42fffddf73abdfb69dda26513634d6a941ba.jpg)

![](images/8b154f0e553e83a3f7c35c49fe7ab99e52462ca20297ac566b2b69d4d956f75e.jpg)

![](images/e62948cac5ce3cce239be31bbe636aa2768e21b3b9bcac2a0a8f0a726ed509fe.jpg)

![](images/bc2fd0666d0a6989cc8668a94739fcaba1d9d42a1ae0e36ff2802516a5458d66.jpg)

![](images/f2a121794c47fcdbf4f5bf41b00c66ca45ecbc15cce16684544840535efd3d76.jpg)

![](images/2aa8ce1b4c1560395453fa5b453a5ffda7ecedc3c98d1a3b1d43ddc0f20fca3e.jpg)

![](images/2449c5cd4064ec3d8185b7d90fbf99ea68dd44bfef3057b3d89a8468d00766e6.jpg)

![](images/a4a3d50916be6283ce018ba174a7ad273dfe08f28a102d9c9940324236f53107.jpg)
Fig. 3 Scattering characteristics at CCPA EP and experimental verification. a, Numerical simulation (line) and experimental (dots) reflectivity spectra. b, The experimental platform and the metamaterial sample. c, Left panel: absorption spectra under coherent (dual mode) excitation, with an inset showing the energy contribution of each incident mode to the total incident energy. Middle and right panels: upper sections show absorption spectra for single-mode excitation, and bottom sections display the corresponding energy ratios of incident and reflected modes. Purple and blue bars represent simulated and experimental results, respectively. d, Incident and reflected sound field distributions of coherent incidence at CCPA-EP. The displayed sectional plane (marked by the dashed circle) is located at a distance of 1.4 wavelengths from the metamaterial. Middle section: incident sound fields. Right section: reflected sound fields.

$|r_{++}| = |r_{--}| = |r_{+-}| = |r_{-+}|$ . To prove this, numerical simulations and experimental measurements are carried out to investigate the variation of reflection coefficients on frequency, where the CCPA-EP is demonstrated by the identical amplitudes of the four coefficients at $f_0$ (Fig. 3a). The experimental configuration is presented in Fig. 3b. A speaker array is used to precisely generate high-purity chiral states $|+\rangle$ and $|- \rangle$ . A microphone array, arranged both angularly and axially, is used for acoustic field decomposition to analyze the reflected modes so that scattering matrix S can be calculated for each element. For a more detailed introduction on experiments, readers are referred to the Supplementary Section 2. As illustrated in Fig. 3a, the experimental results agree well with the numerical simulations (errors may originate from defects of the metamaterial sample fabricated by 3D printing technology).

Figure 3c shows the absorption spectrum of the system under the coherent excitation of balance-strength states $|+\rangle$ and $|- \rangle$ with a delicate phase difference of $\Delta_{\mathrm{i}} = \pi /2$ , corresponding to the eigenstate of S. At $f_0$ , the simulation prediction shows an absorption coefficient exceeding 0.99 while the experimental result remarkably reaches 0.98 (the left panel in Fig. 3c, the inset: incident energy flux proportion of distinct modes), in stark contrast to the cases under only single-mode vortex incidence (the upper middle and upper right panels). The lower right panel in Fig. 3c presents the incident and reflected energy proportions of each mode under single-mode excitation, showing consistency between experimental and simulation results. Figure 3d displays the incident and reflected fields of the real part of sound pressure at CCPA-EP. The measured amplitude and phase profiles, obtained via field scanning with a scanning area $50~\mathrm{mm}\times 50~\mathrm{mm}$ and a scanning step of $2\mathrm{mm}$ , demonstrate excellent agreement with numerical simulations. These experimental results further validate the scattering zero of the metamaterial at the CCPA-EP (see Supplementary Section 2 for experimental details).

For EPs in non-Hermitian systems, loss modulation usually involves complicated reconfigurations of structural parameters, thereby placing a challenge for fine-tuning the non-Hermiticity approaching the degeneracy. In contrast, the coherent excitations at CCPA-EP offer more strategies for EP-related absorption manipulation. Among these strategies, adjusting the strength ratio of states $|+\rangle$ and $|- \rangle$ is one; however, this cannot achieve a complete manipulation of absorption from 0 to 100% (see related discussions in Supplementary Section 3). On the other hand, phase modulations serve as a promising method not only enabling flexible switching of the symbol of CCPA-EPs—perfect absorption—on and off but also allowing the state transformation between different CCPA-EPs.

First, the degree of absorption at CCPA-EP is sensitive to the relative phase of the inputs. To characterize this, the incident vortex with TC = 1 is rotated by an angle $\Delta_{i}$ , which effectively adds an additional phase to the input state $|+\rangle$ (Fig. 4a). During the variation of $\Delta_{i}$ , the input changes, but the system characterized by S and $\Delta_{\pm}$ maintains (Fig. 4b), causing the symbol of CCPA-EP—perfect absorption under coherent incidences—to be hidden. The case $\Delta_{i} = \pi/2$ corresponds to the input eigenstate $v_{1,2} = (-i,1)^{\mathrm{T}}$ necessary to achieve perfect absorption, while total reflection occurs when $\Delta_{i} = -\pi/2$ . In Fig. 4b, the maximum and minimum absorptions are denoted by the red pentagram and the blue diamond, respectively, corresponding to "on" and "off" of the symbol of the CCPA-EP. Thus, the absorption efficiency undergoes a continuous evolution from 0 to 100% as $\Delta_{i}$ completes a full $2\pi$ cycle (Fig. 4b), implying a dynamic on-off control of the symbol of CCPA-EPs, which paves an effective route for signal phase detection by the degree of absorption at CCPA-EP.

The regulation of absorption capacity of the system can also be controlled by $\Delta_{\pm}$ , which closely relates to the geometric phase [41, 42] of the metamaterial. To this end, the metamaterial is rotated around the z axis by an angle $\beta$ (Fig. 4a), while the system input is maintained with the delicate phase difference of $\Delta_{i} = \pi/2$ .

![](images/fd54af0985e3cacff2eae28ddb51533f2a256e48830fae4d4ed44a831a511e64.jpg)
Fig. 4 Absorption modulations and CCPA-EP state transformation via rotation operations. a, The two phase-based mechanisms for absorption modulations: rotating one of the incident vortices, $|+\rangle$ , by a phase of $\Delta_{\mathrm{i}}$ , and rotating the metamaterial by an angle $\beta$ . b,c, The changes in reflection phase difference $(\Delta_{\pm})$ and absorption coefficient induced by these two approaches, respectively. The red pentagram and the blue diamond (hexagon) indicate the maximum and minimum absorption, respectively. d, CCPA-EP state transformation illustrated on the Bloch sphere. On the sphere, the elevation angle $(\tau)$ and azimuth angle $(\Delta_{\mathrm{i}})$ represent mode contributions and relative phase between states, respectively. The eigenstates associated with the CCPA-EP pair are located at the intersections of the equatorial line and the $y$ -axis. Structural rotation by $\beta = \pi /2$ drives the CCPA-EP state transformation. Coherent excitation corresponding to the eigenstate of one of the CCPA-EPs causes coherent perfect absorption of that EP, while it is completely decoupled from the other, resulting in near-zero absorption for the other EP. In lower panels, lines denote simulations and dots represent experimental results.

This corresponds to applying a rotation operator $\mathbf{R}$ to the original system, where the scattering matrix after rotation is expressed as $\mathbf{S}_{\mathbf{R}} = \begin{pmatrix} r_{++}e^{-i2\beta} & r_{+-} \\ r_{-+} & r_{--}e^{i2\beta} \end{pmatrix}$ . As implied by $\mathbf{S}_{\mathbf{R}}$ , an additional geometric phase will be introduced when the incident mode is converted to the reflected mode preserving the same chirality under the action of unitary rotational transformation $\mathbf{R} = \begin{pmatrix} e^{i\beta} & 0 \\ 0 & e^{-i\beta} \end{pmatrix}$ (see more details in Supplementary Section 3). As shown in Fig. 4c, the influence of the additional geometric phase causes $\Delta_{\pm}$ to undergo a change from $-\pi$ to $\pi$ , thereby inducing a variation in the absorption coefficient from 0 to 1. Both $\Delta_{\pm}$ and absorption coefficient complete two cycles of transition during a full $2\pi$ rotation, underpinning the $C_2$ -rotational symmetry of the system. Surprisingly, the total reflection represented by the diamond markers in Fig. 4c corresponds to another CCPA-EP. This result is due to that the coherent inputs adopted here are fully decoupled from the eigenstate $v_{1,2}' = (i,1)^{\mathrm{T}}$ of this CCPA-EP, thus resulting in a zero absorption. As such, the induced geometry phase effectively tunes the system's intrinsic characteristics, allowing the state transformation between different CCPA-EPs.

![](images/6538e02c673895b384f48b1aabc228f7bc3b4d7286b986dd83a77d609edcb406.jpg)
Fig. 5 Sensitivity of different TC-ordered CCPA-EPs. a,b, Variation trajectories of the eigenvalues around CCPA-EPs of $\mathrm{TC} = \pm 1$ and $\mathrm{TC} = \pm 2$ as $\beta$ evolves. Solid and dashed lines denote $\lambda_{1}$ and $\lambda_{2}$ , respectively. Left panel: real part of eigenvalues. Right panel: imaginary part of eigenvalues. The insets show the cross-sectional view of the corresponding metamaterials on the $x - y$ plane. c, The triangles represent the difference between absolute values of eigenvalues ( $\Delta \lambda = |\lambda_1| - |\lambda_2|$ ) calculated by simulations. Solid lines show corresponding fitting curves. The purple and pink curves (symbols) correspond to $\mathrm{TC} = \pm 2$ and $\mathrm{TC} = \pm 1$ , respectively.

The Bloch sphere provides an intuitive representation for superpositions of the two OAM states $|\pm\rangle$ . The eigenstates corresponding to the aforementioned CCPA-EP pair are located at the intersection of the y-axis (on this axis states $|\pm\rangle$ being orthonormal to each other) and the equatorial line (on this line states $|\pm\rangle$ hosting identical amplitude) of the Bloch sphere (Fig. 4d, the upper panel). Generally, achieving EP-state conversions typically relied on structural redesigns [19]. In contrast, the geometric phases introduced here provide a structure-preserving method for CCPA-EP conversion. As shown in Fig. 4d, after a metamaterial's rotation of $\pi/2$ , the state converts from EP1 to EP2. The original EP1 now has no response to the excitation of $v'$ (the middle panel), whereas EP2 exhibits perfect absorption for the identical excitation (the lower panel). Note that the mirror-inversion operation also facilitates EP conversion[43], which, combined with rotational operation, enables the restoration of system to its initial state (see details in Supplementary Section 3).

The result shown in Fig. 4 — absorption spectrum evolving twice during the $2\pi$ -cycle rotation of the metamaterial — is a direct consequence of rotational symmetry $C_2$ . It can be straightforwardly expected that, for a metamaterial hosting higher symmetry group $(C_T)$ , the absorption of the system will undergo $T$ cyclic variations. This important result indicates a higher rate of change in absorption, underscoring a promising paradigm for enhancing the sensitivity associated with a lower-order EP by leveraging the additional degree of freedom (rotation symmetry) in chiral space. This is in stark contrast to previously sensitive-enhancing route relying on higher-order EP from complex systems that may inevitably involve the issue of growing system complexity[17]. To validate this, we showcase a metamaterial hosting $C_4$ rotational symmetry and underpinning a CCPA-EP for vortex modes with $\mathrm{TC} = \pm 2$ (see detailed metamaterial parameters and scattering properties in Supplementary Section 4). We compare its sensitivity in terms of the variation of eigenvalues with the one mentioned above with $C_{2}$ rotational symmetry by tuning geometry phase $\beta$ . Intriguingly, during the variation of the geometric phase ( $\beta$ ), the two eigenvalues of the system undergo alternating changes: while one eigenvalue changes, the other remains zero (Figs. 5a and b). This result, which can be theoretically predicted by $\lambda_{1,2}=r_{++}\left(-i\sin2\beta\pm i|\sin2\beta|\right)$ for $S_{R}$ , holds significance in two key aspects. Firstly, it demonstrates that the geometric phase periodically modulates the EP state (where the two eigenvalues coalesce), while the CPA state is continuously maintained (one eigenvalue always being zero). Secondly, and more importantly, it reveals that under geometric phase modulation, the two eigenvalues individually exhibit sensitivity (the changing eigenvalue) and robustness (the invariant eigenvalue), which are two distinct features now coexisting in the same system. These distinctive characteristics, which are unobserved in plane-wave systems to the best of our knowledge, underscore the uniqueness of CPA-EPs in chiral spaces. Remarkably, when $\beta$ is considered as a perturbation (i.e., with small variations), the degree of eigenvalue splitting in these two metamaterials exhibits a distinct difference: despite both hosting second-order EPs, the $C_{4}$ -symmetric metamaterial demonstrates significantly enhanced sensitivity by leveraging the higher rotational symmetry in chiral space (Fig. 5c). This remarkable result—enhanced sensitivity via a higher-order TC—may pave an important route for advanced sensing applications by lower-order EPs.

## Discussion

Exploring the concept of CPA-EP in chiral space brings about unique absorbing states arising from the delicate interplay between paired vortices carrying opposite TCs. The chiral space opens an avenue to manipulate CPA-EPs through additional degree of freedom—rotation operations. In general, the two phase-related modulation strategies induced via rotation operations exhibit both distinctions and interconnections. Essentially, variation on the phase of coherently incident vortices preserves the intrinsic properties of the system, whereas structural rotation alters the system by inducing geometric phase. The latter inspires an effective strategy for CCPA-EP state transitions. Nevertheless, from the perspective of system response, since both modulation mechanisms ultimately affect the related phase of scattered vortices, they exhibit functional equivalence in governing the absorption behavior. The $C_{2}$ rotational symmetry for TC = ±1 imposes a factor-of-two relation between the variation of $\Delta_{i}$ (phase difference of input) and $\beta$ (geometry phase of the metamaterial) for the same absorption. As we have demonstrated, this relationship scales by a factor of order number T in higher-symmetry $C_{T}$ -symmetric systems, paving an important route for sensitivity enhancing via higher-order TCs rather than higher-order EPs. In practical implementations, these two phase manipulation approaches enable the compensation arising from random defects in either one of signal sources or system itself by adjusting the other, thereby providing an important way for the robustness of CCPA-EP both spectrally and spatially. These unique characteristics, involving both sensitivity and robustness, intrinsically linked to rotational symmetry of chiral space, empower CPA-EP based wave manipulations.

In summary, our work realizes CPA-EPs in chiral systems by harnessing TC mode channels formed by OAM-carrying vortices. We systematically analyze and experimentally verify their intrinsic properties and scattering characteristics. Moreover, we show that the introduction of TCs enables rotational operations to realize on-off dynamic control of perfect absorption as well as state transformation between different CCPA-EPs. The rotational operation associated with TC modes further offers a novel approach to enhancing sensitivity via a lower-order EP, in which the two eigenvalues alternatively exhibit contradictory features—robustness and sensitivity, respectively. This work enhances the versatility of CPA-EPs, expanding avenues for the development and applications of highly sensitive non-Hermitian chiral devices.

## Methods

## Numerical Simulations

Calculations and optimizations of the metamaterial are carried out using numerical simulations based on the finite element method. The background pressure field is assigned the sound pressure of the target mode as the excitation source. In addition to setting perfectly matched layers at the port near the sound source to eliminate the influence of multiple reflection waves, hard boundaries are imposed at all interfaces between the ambient air domain and the entire model, including the waveguide and metamaterial structure. The air domain parameters are set to sound speed $c_{0} = 343$ m/s and static density $\rho = 1.21$ kg/m $^{3}$ . The thermal-viscous boundary layers are implemented within the structural cavities to mimic the intrinsic dissipative losses of the metamaterial. The analysis of incident and reflection modes is calculated by the circumferential mode decomposition method (see Supplementary Section 5).

## Experimental setup

As shown in Fig. 3b, the cylindrical waveguide assembly in the experimental platform is equipped with four loudspeakers uniformly distributed along the circumference at one end to generate the vortex beams with target modes, while the other end is loaded with the metamaterial sample which is fabricated through stereolithography using photosensitive resin. Eight microphones are evenly distributed in two circles to simultaneously decompose the incident and scattered field information. The speaker array is driven by a PXI Multifunction I/O Device (National Instruments, type PXIe-9263), which supports four output channels. The measured signals from microphones (GRAS, type 46BD) are transmitted to another PXI Multifunction I/O Device (National Instruments, type PXIe-4497) for data processing.

## Sample Fabrication

The metamaterial sample in Fig. 3b is fabricated using 3D printing technology. The stereo lithography apparatus (SLA) employed offers a precision of $0.1\mathrm{mm}$ . The material used for the sample is photosensitive resin, with density $\rho_{\mathrm{m}} = 1160\mathrm{kg / m^3}$ , Young's modulus $E_{\mathrm{m}} = 2450\mathrm{MPa}$ , and Poisson's ratio $\nu_{\mathrm{m}} = 0.41$ . The wall thickness of the resin sample employed in the experimental measurements is set to $3\mathrm{mm}$ .

Data availability. All data are available in the main text or the supplementary information.

Supplementary information.

Acknowledgements. This work was supported by the National Key R&D Program of China (Grant Nos. 2022YFA1404400 and 2022YFE0208000), the National Science Foundation of China (Grant No. 12474463, and 124B2087), the Scientific Research Innovation Capability Support Project for Young Faculty (Grant No.ZYGXQNJSKYCXNLZCXM-D8), the Shanghai Pilot Program for Basic Research, the Xiaomi Young Talents Program, and the Fundamental ResearchFunds for the Central Universities.

Author contributions. Conceptualization: Y.L., X.W., H.D., Q.-S.W. Simulation and experiment: H.D., Q.-S.W. Supervision: Y.L., X.W. Writing—original draft: H.D. Writing—review & editing: H.D., Q.-S.W., X.W., Y.L.

Competing interests. The authors declare no competing interests.

## References

[1] Miri, M.-A. & Alù, A. Exceptional points in optics and photonics. Science 363, eaar7709 (2019).

[2] Özdemir, S. K., Rotter, S., Nori, F. & Yang, L. Parity–time symmetry and exceptional points in photonics. Nat. Mater. 18, 783–798 (2019).

[3] Feng, L. et al. Experimental demonstration of a unidirectional reflectionless parity-time metamaterial at optical frequencies. Nat. Mater. 12, 108-113 (2012).

[4] Zhao, W. et al. Exceptional points induced by unidirectional coupling in electronic circuits. Nat. Commun. 15, 9907 (2024).

[5] Huang, Y., Shen, Y., Min, C., Fan, S. & Veronis, G. Unidirectional reflectionless light propagation at exceptional points. Nanophotonics 6, 977-996 (2017).

[6] Shen, C., Li, J., Peng, X. & Cummer, S. A. Synthetic exceptional points and unidirectional zero reflection in non-hermitian acoustic systems. Phys. Rev. Mater. 2, 125203 (2018).

[7] Doppler, J. et al. Dynamically encircling an exceptional point for asymmetric mode switching. Nature 537, 76-79 (2016).

[8] Shu, X. et al. Chiral transmission by an open evolution trajectory in a non-hermitian system. Light: Sci. Appl. 13, 65 (2024).

[9] Wang, X., Fang, X. S., Mao, D. X., Jing, Y. & Li, Y. Extremely asymmetrical acoustic metasurface mirror at the exceptional point. Phys. Rev. Lett. 123, 214302 (2019).

[10] Tang, W. et al. Exceptional nexus with a hybrid topological invariant. Science 370, 1077-1080 (2020).

[11] Yoon, J. W. et al. Time-asymmetric loop around an exceptional point over the full optical communications band. Nature 562, 86–90 (2018).

[12] Tang, W., Ding, K. & Ma, G. Realization and topological properties of third-order exceptional lines embedded in exceptional surfaces. Nat. Commun. 14, 6660 (2023).

[13] Ding, K., Ma, G., Xiao, M., Zhang, Z. Q. & Chan, C. T. Emergence, coalescence, and topological properties of multiple exceptional points and their experimental realization. Phys. Rev. X 6, 021007 (2016).

[14] Park, J.-H. et al. Symmetry-breaking-induced plasmonic exceptional points and nanoscale sensing. Nat. Phys. 16, 462–468 (2020).

[15] Chen, W., Özdemir, S. K., Zhao, G., Wiersig, J. & Yang, L. Exceptional points enhance sensing in an optical microcavity. Nature 548, 192–196 (2017).

378 [16] Rechtsman, M. C. Optical sensing gets exceptional. Nature 548, 161-162 (2017).

[17] Hodaei, H. et al. Enhanced sensitivity at higher-order exceptional points. Nature 548, 187–191 (2017).

[18] Mao, W., Fu, Z., Li, Y., Li, F. & Yang, L. Exceptional–point-enhanced phase sensing. Sci. Adv. 10, eadl5037 (2024).

[19] Peng, B. et al. Chiral modes and directional lasing at exceptional points. Proc. Natl. Acad. Sci. U.S.A. 113, 6845–6850 (2016).

[20] Wang, C., Sweeney, W. R., Stone, A. D. & Yang, L. Coherent perfect absorption at an exceptional point. Science 373, 1261–1265 (2021).

[21] Lee, H. et al. Chiral exceptional point and coherent suppression of backscattering in silicon microring with low loss mie scatterer. eLight 3, 20 (2023).

[22] Wang, C. Q. et al. Electromagnetically induced transparency at a chiral exceptional point. Nat. Phys. 16, 334–340 (2020).

[23] Song, Q., Odeh, M., Zuniga-Perez, J., Kante, B. & Genevet, P. Plasmonic topological metasurface by encircling an exceptional point. Science 373, 1133–1137 (2021).

[24] Feng, X. et al. Non-hermitian hybrid silicon photonic switching. Nat. Photonics 19, 264–270 (2025).

[25] Chen, H.-Z. et al. Revealing the missing dimension at an exceptional point. Nat. Phys. 16, 571-578 (2020).

[26] Fang, X. et al. Observation of higher-order exceptional points in a non-local acoustic metagrating. Commun. Phys. 4, 271 (2021).

[27] Zhou, H. T. et al. Underwater scattering exceptional point by metasurface with fluid-solid interaction. Adv. Funct. Mater. 34, 2404282 (2024).

[28] Chong, Y. D., Ge, L., Cao, H. & Stone, A. D. Coherent perfect absorbers: Time-reversed lasers. Phys. Rev. Lett. 105, 053901 (2010).

[29] Baranov, D. G., Krasnok, A., Shegai, T., Alù, A. & Chong, Y. Coherent perfect absorbers: linear control of light with light. Nat. Rev. Mater. 2, 17064 (2017).

[30] Sweeney, W. R., Hsu, C. W., Rotter, S. & Stone, A. D. Perfectly absorbing exceptional points and chiral absorbers. Phys. Rev. Lett. 122, 093901 (2019).

[31] Hörner, H. et al. Coherent perfect absorption of arbitrary wavefronts at an exceptional point. Phys. Rev. Lett. 133, 173801 (2024).

[32] Bai, K. et al. Nonlinearity-enabled higher-order exceptional singularities with ultra-enhanced signal-to-noise ratio. Natl. Sci. Rev. 10, nwac259 (2023).

[33] Bai, K. et al. Nonlinear exceptional points with a complete basis in dynamics. Phys. Rev. Lett. 130, 266901 (2023).

[34] Zhou, Z., Jia, B., Wang, N., Wang, X. & Li, Y. Observation of perfectly-chiral exceptional point via bound state in the continuum. Phys. Rev. Lett. 130, 116101 (2023).

[35] Zhang, Z. et al. Tunable topological charge vortex microlaser. Science 368, 760–763 (2020).

[36] Jiang, X., Liang, B., Cheng, J. C. & Qiu, C. W. Twisted acoustics: Metasurface-enabled multiplexing and demultiplexing. Adv. Mater. 30, 1800257 (2018).

[37] Hentschel, M., Schäferling, M., Duan, X., Giessen, H. & Liu, N. Chiral plasmonics. Sci. Adv. 3, e1602735 (2017).

[38] Sha, X. et al. Chirality tuning and reversing with resonant phase-change metasurfaces. Sci. Adv. 10, eadn9017 (2024).

[39] Yang, Q., Wen, X., Li, Z., You, O. & Zhang, S. Gigantic tellegen responses in metamaterials. Nat. Commun. 16, 151 (2025).

[40] Zou, Z., Lirette, R. & Zhang, L. Orbital angular momentum reversal and asymmetry in acoustic vortex beam reflection. Phys. Rev. Lett. 125, 074301

(2020).

[41] Xie, X. et al. Generalized pancharatnam-berry phase in rotationally symmetric meta-atoms. Phys. Rev. Lett. 126, 183902 (2021).

[42] Zhang, K. et al. Geometric phase in twisted topological complementary pair. Adv. Sci. 10, 2304992 (2023).

[43] Yang, Z. J. et al. Creating pairs of exceptional points for arbitrary polarization control: asymmetric vectorial wavefront modulation. Nat. Commun. 15, 232 (2024).

## Supplementary Files

This is a list of supplementary files associated with this preprint. Click to download.

\- Supplementary.pdf
