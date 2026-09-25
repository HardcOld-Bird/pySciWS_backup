# Photonic exceptional points in engineered materials and their emerging applications

Haoye Qin $^{1,7}$ , Wenjing Lv $^{2,7}$ , Zhe Zhang $^{3}$ , Zijin Yang $^{2}$ , Jue Li $^{2}$ , Mengyao Li $^{2}$ , Bo Li $^{2}$ , Ji Zhou $^{4}$ , Romain Fleury $^{3}$ , Patrice Genevet $^{5}$ , Qinghua Song $^{2}$ & Cheng-Wei Qiu $^{1,6}$

## Abstract

Enabled by the coalescence of eigenvalues and eigenstates, exceptional points (EPs) in non-Hermitian photonic systems have revolutionized the control of light–matter interactions and sparked growing interest across diverse material and structural platforms. This Review synthesizes advances in engineered materials that harness EPs across three key domains: band EPs in dielectric photonic crystals, wherein radiation-induced loss transforms Hermitian degeneracies into exceptional rings and bulk Fermi arcs; scattering EPs in hybrid dielectric or lossy metasurfaces, enabling unidirectional reflectionless propagation; and Jones EPs in plasmonic and anisotropic materials, which exploit chiral degeneracies for asymmetric scattering and holographic multiplexing. We highlight emerging phenomena in dynamic EP control using tunable materials such as graphene, phase-change media and micro-electromechanical systems, which enable real-time modulation, topological phase transitions and the direct observation of non-Hermitian braiding. The topological properties of EPs, manifested in phase accumulation and half-integer polarization charges, support key applications in wavefront shaping and singular optics. New frontiers involve the integration of EPs with other concepts, including bound states in the continuum, Dirac points, nonreciprocity and magnetic tunability. Bridging non-Hermitian physics with material-engineered platforms paves the way for adaptive photonic devices, topological meta-architectures and machine learning-driven designs, charting a path towards next-generation nanophotonics.

Sections

Introduction
EPs in engineered materials
Band EPs in photonic crystals
Scattering EPs in metasurfaces
Jones EPs
Emerging applications of EPs
Outlook

## Introduction

The study of exceptional points (EPs), exotic singularities in non-Hermitian systems, has redefined our understanding of light–matter interactions $^{1-5}$ . Unlike Hermitian systems governed by energy conservation, non-Hermitian systems embrace the realities of gain, loss and environmental coupling, enabling phenomena such as eigenvalue coalescence and chiral dynamics $^{6-9}$ (Box 1). At an EP, two or more optical modes merge in both frequency and eigenstate, creating a degeneracy whereby the response of a system becomes hypersensitive to perturbations $^{3,10-12}$ . This unique trait allows devices operating near EPs to detect minute changes in their surroundings, such as single molecules or strain $^{12-14}$ , with unprecedented precision. Furthermore, EPs can introduce direction-dependent wave propagation, asymmetric

## Box 1 | Exceptional points in matrices

The non-Hermitian nature of photonic systems — arising from material gain or loss and radiative coupling — enables exceptional points (EPs). Mathematically, these are degeneracies at which eigenvalues and eigenvectors coalesce, corresponding to the appearance of nontrivial Jordan blocks (size>1) in the Jordan form of a non-diagonalizable complex matrix $^{175}$ . For photonics, such complex-valued matrices lie in Hamiltonians H and scattering matrices S: for finite coupled resonators, EPs are described in H with the coalescence of eigenmodes in resonators $^{176}$ ; for periodic photonic structures, EPs emerge from $H(\mathbf{k})$ within the Brillouin zone $^{40}$ ; in the context of photonic scattering, non-unitary scattering matrices can also give rise to EPs $^{177}$ by modelling responses among waveguides, spatial scattering channels, photonic modes and polarizations that can be referred to as Jones matrices. For Hermitian systems, H is Hermitian whereas S satisfies unitarity: both are diagonalizable square matrices by similarity transformation. Therefore, to achieve EPs, breaking Hermiticity is necessary. However, it is not sufficient for EPs: for example, parity-time symmetric systems $^{3}$ can support diagonalizable real-valued eigenvalue spectra for H and S. According to matrix theory, any complex square matrix can be transformed into its Jordan canonical form. An s×s Jordan block $J_{s}(\lambda)$ is a square matrix in which the eigenvalue $\lambda$ is repeated on the main diagonal with ones on the super diagonal and zeros everywhere else, expressed as

$$
J _ {s} (\lambda) = \left[ \begin{array}{c c c} \lambda & 1 & 0 \\ 0 & \ddots & 1 \\ 0 & 0 & \lambda \end{array} \right].
$$

The emergence of nontrivial Jordan blocks $(s>1)$ renders the matrix non-diagonalizable — this is the necessary and sufficient condition for EPs $^{178}$ . The size of the Jordan block s represents the number of coalescing eigenstates, defining the order of the EPs. The enhanced sensitivity of EPs can be explained from the perturbation theory of matrices performed around these points $^{179,180}$ . For diabolic points, which are locally a double cone formed by two sheets under a perturbation on parameter $\alpha$ , the gap between the final eigenvalue and the original one can be expressed by a series:

$$
\delta \lambda = \sum_ {j = 1} ^ {\infty} c _ {j} \cdot (\delta a) ^ {j}.
$$

optical responses and topological robustness $^{15-17}$ . These properties position EPs as transformative tools for sensing, wave manipulation and optical processing, but their practical implementation requires precise control over gain, loss and coupling – a challenge largely addressed by engineered materials $^{18,19}$ .

From a materials standpoint, non-Hermitian photonics emerges from how different material platforms provide and balance these key parameters. Semiconductors and quantum wells deliver optical gain for amplifying modes, whereas plasmonic and metallic structures supply controllable absorption loss. Low-loss dielectrics, such as Si or $TiO_{2}$ , create radiation-dominated environments for efficient mode coupling, whereas magnetic and light-emitting materials introduce non-reciprocity and EP-assisted lasing. In parallel, adaptive media including

At an EP of order s, the usual Taylor series fails; the correct expansion is a modified series with fractional powers:

$$
\delta \lambda = \sum_ {j = 1} ^ {\infty} c _ {j} \cdot (\delta a) ^ {\frac {j}{s}}.
$$

The modification on power terms from $(\delta a)^{j}$ to $(\delta a)^{\frac{j}{s}}$ exhibit the s-order enhanced sensitivity. Furthermore, beyond the eigenvalue properties at and around EPs, the orthogonality of eigenmodes at EPs is also a focus: they collapse into a single self-orthogonal state. In Hermitian systems, both the Hamiltonian H and the scattering matrix S are normal matrices $(AA^{\dagger}=A^{\dagger}A)$ . The unitary similarity diagonalizes them, and eigenvectors are orthogonal. In non-Hermitian settings, orthogonality is replaced by biorthogonality $^{7,181}$ . Define right and left eigenvectors by

$$
\begin{array}{r l} & A | R _ {n} \rangle = \lambda_ {n} | R _ {n} \rangle , \\ & \langle L _ {n} | A = \lambda_ {n} \langle L _ {n} |, \end{array}
$$

where n is the index for eigenvalues and eigenvectors. If A is diagonalizable with nondegenerate eigenvalues, they can be chosen to be biorthonormal:

$$
\langle L _ {m} | R _ {n} \rangle = \delta_ {\mathrm{mn}}, \sum_ {n} | R _ {n} \rangle \langle L _ {n} | = I, A = \lambda_ {n} \sum_ {n} | R _ {n} \rangle \langle L _ {n} |.
$$

At an EP, the coalesced eigenvector is self-orthogonal:

$$
\langle L _ {E P} | R _ {E P} \rangle = 0.
$$

For photonic applications, this property has been exploited for unidirectional polarization conversions and observed in metasurfaces. The self-orthogonality also gives rise to two indices $^{182}$ that are widely used in numerical studies to identify EPs, as EPs form a measure-zero set. They are phase rigidity (O at EPs) and the Petermann factor (diverging at EPs), defined as

$$
\text { Phase   rigidity }: r _ {n} = \frac {| \langle L _ {n} | R _ {n} \rangle |}{\sqrt {\langle R _ {n} | R _ {n} \rangle \langle L _ {n} | L _ {n} \rangle}} \in [ 0, 1 ],
$$

Petermann factor: $K_{n}=\frac{\langle R_{n}|R_{n}\rangle\langle L_{n}|L_{n}\rangle}{|\langle L_{n}|R_{n}\rangle|^{2}}$ .

## EPs in metasurfaces and PhCs

![](images/c38983270dff71b51901276fcf8f425943ec4fcfc7c5d875376476be8d6aff6a.jpg)
Fig. 1 | Photonic EPs in engineered materials and their emerging applications. Engineered photonic structures including photonic crystals (PhCs), metasurfaces and other designed materials serve as versatile platforms for realizing non-Hermitian exceptional points (EPs). Around an EP, the eigenvalues E form characteristic Riemann surfaces, exhibiting the simultaneous coalescence of their real and imaginary parts under parameter tuning. Photonic EPs can be
broadly classified as band EPs, scattering EPs and Jones EPs, reflecting different representations of non-Hermitian degeneracies. Emerging applications span dynamic control of EPs, non-Hermitian braiding, topological properties, wavefront engineering and special EP states, establishing EPs as a unifying framework for next-generation photonic functionalities.

graphene, phase-change materials and micro-electromechanical system (MEMS) architectures enable dynamic tuning of complex refractive index and coupling, transforming static resonances into reconfigurable EP platforms. Collectively, these engineered materials establish the foundation of non-Hermitian photonics by offering controllable pathways to introduce, tune and balance gain, loss, coupling and the other degrees of freedom, therefore shaping optical singularities and topological photonic behaviours.

Engineered materials, such as metasurfaces and photonic crystals (PhCs), which are artificial structures composed of, respectively, subwavelength 'meta-atoms' or structured unit cells, as well as other designer materials, enable the tailoring of the behaviour of light in ways unattainable with natural materials. By designing the geometry, arrangement and composition of these meta-atoms, metamaterials achieve exotic optical responses, such as negative refraction, near-zero permittivity and electromagnetic cloaking $^{20,21}$ . Beyond static designs, modern metamaterials integrate dynamic elements including liquid crystals, tunable semiconductors or phase-change materials $^{22}$ , enabling real-time control over their optical properties $^{23-25}$ . This adaptability makes them ideal platforms for embedding non-Hermitian features, such as spatially balanced gain and loss, that are essential for creating and stabilizing EPs. Their subwavelength architecture also amplifies light–matter interactions, enhancing phenomena such as plasmonic resonances and photonic bandgap engineering. As a result, metamaterials serve as versatile playgrounds for testing theoretical concepts and translating them into functional devices, from superlenses to ultracompact sensors $^{21,26}$ .

The convergence of non-Hermitian theory with materials-by-design approaches marks a pivotal moment for photonics. Recent progress in inverse-designed nanostructures, hybrid integration of active and passive media, and ultrafast measurement techniques has elevated EP physics from theoretical abstraction to an experimentally accessible framework for functional device engineering. Viewing EPs through the framework of materials design establishes a coherent basis for linking non-Hermitian physics with practical photonic implementations. By highlighting how material composition, structural geometry and tunability together dictate the distribution of gain, loss and coupling, this Review elucidates how engineered platforms transform theoretical constructs into experimentally accessible phenomena. Building on this perspective, the following sections examine how EPs emerge within engineered material systems and how these platforms enable new regimes of photonic functionality.

## EPs in engineered materials

The synergy between EPs and engineered materials such as meta-surfaces and PhCs unlocks considerable opportunities in photonics (Fig. 1). Metamaterials provide the spatial and dynamic control needed to engineer EPs in tailored systems, such as PT-symmetric lattices or coupled-resonator arrays $^{27,28}$ . These hybrid platforms exploit the tunability of metamaterials to achieve precise eigenvalue coalescence while leveraging the sensitivity and chirality of EPs for advanced applications. For instance, EP-enhanced metamaterial sensors can detect biomarkers at ultralow concentrations $^{29-32}$ , whereas topological encircling of EPs can accumulate $2\pi$ phase for diverse wavefront controls $^{17,33,34}$ . Challenges in scalability and fabrication remain $^{35}$ , but advances in nanophotonics and computational design are rapidly closing these gaps. By merging the flexibility of metamaterials with the singular physics of EPs, this interdisciplinary frontier promises to redefine photonic technologies, pushing the boundaries of what light can achieve $^{36}$ .

This Review examines current research progress on photonic EPs realized with engineered materials by categorizing them into three primary domains: band EPs, arising in momentum space within photonic band structures; scattering EPs, generated by degeneracies in scattering matrices; and Jones EPs, manifesting as chiral degeneracies in polarization-selective scattering systems dictated by Jones matrices. We further explore emerging applications of EP-related phenomena based on such platforms and novel opportunities, such as the dynamic control of EPs using tunable metamaterials, wherein recent advances have enabled the experimental observation of nontrivial winding charge. Non-Hermitian braiding around EPs has emerged as a tool for a more in-depth exploration of non-Hermitian physics and for applications associated with non-Abelian operations. The topological properties of EPs are analysed across parameter space and momentum space, and under external control parameters, underscoring their robustness as an important concept in topological photonics, especially in the regime of wavefront engineering through the exceptional topological phase. Finally, we highlight novel proposals to achieve special EP states by constructively interacting EPs with other singularities, and we outline new frontiers in EP research, including efforts to integrate EPs with other cutting-edge concepts in photonics, non-Hermitian physics and beyond.

## Band EPs in photonic crystals

The study of energy-band EPs in PhCs marks a key advancement in exploring non-Hermitian physics within open wave systems. Unlike Hermitian degeneracies (such as Dirac and Weyl points) $^{37-39}$ , which exist in closed systems with orthogonal eigenmodes, EPs emerge from eigenvalue and eigenstate coalescence in non-Hermitian regimes. These degeneracies in PhCs, mostly driven by radiation loss $^{40}$ , reshape light–matter interactions, polarization control and resonance behaviours. Their formation as band EPs or exceptional rings introduces novel paradigms for tailoring optical phenomena in non-Hermitian platforms.

A seminal breakthrough was the demonstration of band EPs in an all-dielectric PhC with negligible material loss $^{40}$ . Through the engineering of an accidental degeneracy between dipole and quadrupole modes at the Brillouin zone's $\Gamma$ point, the system formed a linear Dirac cone under Hermitian conditions $^{41}$ . However, radiation-induced non-Hermiticity, which is an inherent property of open systems, transformed this Dirac cone into an exceptional ring, a closed contour of EPs in momentum space (Fig. 2a). This exceptional ring originates from the disparity between the radiation rates of the modes around the $\Gamma$ point: the radiative dipole mode merges with the quadrupole mode with a relatively lower radiation loss, inducing a bifurcation in the complex eigenvalue spectrum. Inside the ring, the real parts of the eigenvalues flatten into dispersionless bands, whereas the imaginary parts form degenerate profiles outside the ring. At the ring, the EP condition is satisfied, with eigenmodes collapsing into a single self-orthogonal state (Box 1), a behaviour experimentally confirmed via angle-resolved reflectivity measurements. This work underscores radiation coupling to free space in open systems as a universal non-Hermitian mechanism, distinct from material-dependent loss or gain.

a
![](images/e89917a54dff91548c1c3fcba806e998ba7a73d155840eb9cc04096c351274a3.jpg)

![](images/8e024bde7c5573ed349ace6ff3993840b668028ff984b9dfbabf206d060bfd91.jpg)
b

![](images/450b17455f06a6ea245a5b803651c9418723f842668c6e17819dde24951efdf6.jpg)

![](images/dae1671480154262dfe4884a5a37e73fe1f0655f31ec7eeb254bfda9d1060c93.jpg)

c
![](images/65065d519d9cfcb9b1086e14cbe74ed3ba0ff2cdcb390ec029b221ea136173ab.jpg)

![](images/c60b3f998f53b230d9910885b09b3aa14ed770a2c7d2445cb1e2f79c3d5f784b.jpg)
Fig. 2 | Band EPs. a, Exceptional ring generated in momentum space from Hermitian degeneracy by introducing free-space radiation loss in a photonic crystal (PhC) with finite thickness. By tuning the hole radius r, the conditions for the appearance of exceptional points (EPs) are met, and the real (left) and imaginary (right) parts of the eigenvalues become degenerate, respectively, inside and outside a ring in the wavevector space (yellow shading). On the ring of EPs, both the real and the imaginary parts are degenerate. b, Paired EPs connected by a bulk Fermi arc (blue) in a PhC slab with a rhombic lattice of elliptical air holes, whereby radiation loss splits a Dirac point into an EP pair (top; blue dots). At frequencies higher and lower than the EP frequency, closed isofrequency

![](images/9566bb4f1ae2dcda5670e17bd0ccf20e3fca8d21e0d772889f7ad5f49d8701c9.jpg)

![](images/e0804409fa9a31f1ea257a138367e49992db9f61510117876bb3e66e0d7c1ddd.jpg)

d
![](images/880ecbfbd60ccd8db873159dd98555af2747e23d36a73b773fefd993ebc7ea04.jpg)

![](images/0328206e8de43a7325e1b5758c2c8e2e3c471ebd6d99d01413915c0bf5d8ba66.jpg)
contours arise (bottom; red and yellow). c, By inserting periodic breaks into one sublattice of a helical waveguide array (left; top right), non-Hermiticity is introduced through enhanced radiation loss, causing an isolated Weyl point to expand into a Weyl exceptional ring carrying Berry charge. d, Far-field polarization singularities associated with bulk Fermi arcs, wherein both paired EPs and polarization singularities induce topological polarization charges in the radiated fields. Stars and dots denote circular-polarization singularities and EPs, respectively. Panel a reprinted from ref. 40, Springer Nature Limited. Panel b reprinted with permission from ref. 42, AAAS. Panel c reprinted from ref. 43, Springer Nature Limited. Panel d reprinted with permission from ref. 44, PNAS.

In a rhombic lattice with elliptical air holes, radiation loss can split a Dirac point into a pair of EPs, which generate open-ended bulk Fermi arcs $^{42}$ . The bulk Fermi arcs form open-ended isofrequency contours (Fig. 2b) that reside in bulk dispersions, fundamentally different from the surface Fermi arcs connecting the Weyl points in 3D Hermitian systems.

The experimental extension of band EPs to 3D systems was realized by creating a Weyl exceptional ring through the introduction of non-Hermiticity into a helical waveguide array $^{43}$ (Fig. 2c). Periodic breaks in one sublattice induced controlled radiation loss, transforming a Hermitian type-II Weyl point into a continuous EP ring, which preserves the original quantized Berry charge but replaces discrete Weyl crossings with a coalesced eigenmode ring characterized by square-root eigenvalue dispersion in momentum space. Unlike conventional Weyl points, Weyl exceptional rings are distributed Berry flux sources, enabling novel non-Hermitian topological transitions under loss-tuning. Transmission spectroscopy confirmed Fermi arc-like features and extremely small transverse intensity transport velocity, pivotal for tunable photonic applications.

Band EPs in PhCs also leave distinct signatures in the far-field polarization topology. In a system related to the one in Fig. 2b with a rhombic lattice and modified elliptical air holes hosting a bulk Fermi arc, subtle connections were established between the Berry phase of near-field Bloch modes and polarization singularities in their far-field radiation $^{44}$ . Despite an invariant nontrivial $\pi$ Berry phase, the far-field polarization charges are variable in a discontinuous and contour-dependent manner, and they can even become trivial. Such complex charge evolutions are mediated by additional circular-polarization singularities (C points, Fig. 2d), ensuring global charge conservation consistent with Berry phase invariance.

Band EPs offer unprecedented control over photonic density of states and resonance engineering $^{45,46}$ . The dispersionless real bands inside the exceptional ring enable flat bands with high density of states $^{47-49}$ , ideal for enhancing Purcell factors in spontaneous emission or single-photon sources. The abrupt transition in loss dispersion near EPs also provides a mechanism for high-performance, large-area single-mode photonic crystal lasers, wherein mode competition is suppressed by selective radiation leakage. In metasurfaces, chiral emission near band EPs has been leveraged for polarization-selective devices $^{50}$ . Room-temperature EP-driven polariton lasing has also been demonstrated, assisted by enhanced local density of states beneficial for polariton condensation $^{51}$ . With engineered perovskite metasurfaces supporting exciton–polariton states and band EPs, ultrafast all-optical modulation of transmission was realized with giant modulation depth $^{52}$ .

Experimental studies of band EPs rely on precision fabrication and advanced characterization techniques. Angle-resolved spectroscopy and polarization-resolved imaging are key to probing eigenvalue coalescence and far-field polarization singularities $^{53-55}$ . However, practical challenges persist, such as disentangling radiation loss from material loss in hybrid systems, or stabilizing EPs in the presence of fabrication disorder. Resonant scattering and surface roughness introduce uncertainties in eigenvalue measurements, necessitating advanced compensation techniques $^{30,40}$ . Emerging perovskites and hybrid dielectric-plasmonic PhCs offer tunable non-Hermiticity through controllable material losses and exciton–photon coupling. Incorporating active gain layers or electro-optic modulators introduces dynamic access to exceptional rings, expanding band EP engineering to flexible and low-cost materials systems. In parallel, inverse-design approaches can enable the tailoring of band structures and decay characteristics by optimizing the dielectric landscape of PhCs $^{46}$ , allowing EP locations to be predetermined within the Brillouin zone and, thereby, reducing experimental uncertainty and design complexity.

Beyond radiation-induced non-Hermiticity, material gain-loss modulation in PT-symmetric PhCs provides an alternative route to realizing band EPs. By introducing balanced gain and loss into supercell PhC designs, exceptional contours can emerge in momentum space at thresholdless PT phase transitions, enabling systematic band-structure engineering without relying on radiative leakage $^{56}$ . In a non-Hermitian bilayer photonic system incorporating particle gain and loss $^{57}$ , two exceptional rings and an exceptional concentric ring were discovered, and their global evolution was characterized through a unified topological framework. Symmetry-preserving gain-loss perturbations applied to 3D PhCs hosting Hermitian nodal-line band structures have been shown to transform these nodal lines into 2D exceptional surfaces in momentum space, enabling EP-based control of dispersion, density of states and light-matter interactions in non-Hermitian PhCs $^{58}$ . Additionally, numerical studies of a symmetry-protected exceptional ring $^{59}$ in a PhC with negative-index media $^{60}$ have shown that the negativity of the permittivity and the permeability results in an indefinite generalized eigenvalue problem, providing the mathematical foundation for the emergence of symmetry-protected non-Hermitian band degeneracies.

## Scattering EPs in metasurfaces

Scattering EPs – which are situated within the framework of scattering matrices (Box 1) – enable unconventional phenomena such as unidirectional reflectionless propagation, asymmetric scattering suppression and enhanced sensitivity $^{61-65}$ . The general reciprocal two-port scattering matrix is expressed as $S = [t, r_{b}; r_{f}, t]$ , which dictates the relations among incoming and outgoing waves or fields, whereas $r_{f}$ and $r_{b}$ are the complex reflection coefficients for light incidence from the left (forward direction) and right (backward direction), respectively $^{66,67}$ . Owing to reciprocity, the transmission coefficients are identical and are denoted by t. Eigenvalues of this scattering matrix are now $\lambda_{\pm}^{S} = t \pm \sqrt{r_{b} r_{f}}$ , and an EP occurs when $r_{b} r_{f} = 0$ , which is possible only under non-unitary S (Box 1), indicating unidirectional reflectionless light propagation. Scaling scattering EPs to large-scale structures through coating glass wafers with several nanoscale layers enabled unidirectional reflectionless light transport at the macroscale $^{68}$ (Fig. 3a). Specifically, the device consisted of a wafer-scale multilayer thin-film stack composed of alternating absorbing (amorphous silicon) and non-absorbing (silica) dielectric layers, where the EP was reached by engineering the layer thicknesses and loss contrast to suppress reflection from one side. Under sunlight, the wafer imaged only in backward reflection, and this reflection asymmetry exhibited good angular robustness near the unidirectional EP.

Similarly, a top-down asymmetric metasurface consisting of two vertically stacked silver ring resonators demonstrated angle-insensitive unidirectional suppression of reflection (Fig. 3b), sustaining EP phenomena across $\pm25^{\circ}$ incidence angles $^{69}$ . Passive scattering EPs were obtained in a non-Hermitian metasurface by spatially tailoring the loss distributions $^{70}$ (Fig. 3c). A tri-meta-atom supercell achieved an EP by balancing interleaved lossy and lossless regions. At the EP, the metasurface exhibited unidirectional retroreflection: 96% efficiency from the right port but complete suppression from the left port. The leakage loss from slits in metallic backplanes enabled precise amplitude modulation, crucial for eigenvalue coalescence.

b
![](images/c8ed6c0b54a6b7f5771a76f4fa96944c275d137738b9a161e62182bedb6f6cc6.jpg)

![](images/42960bd36cc6580dc2b508ec5888b9e480f33069aa265068ba37a0354b61ca50.jpg)
Fig. 3 | Scattering EPs. a, Large-scale scattering exceptional point (EP) realized using a wafer-scale multilayer thin-film structure composed of alternating absorbing and non-absorbing dielectric layers (top) that enable unidirectional reflectionless light propagation, as exemplified by the photographs taken in forward and backward directions with a band-pass filter centred at 520 nm (bottom). b, Unidirectional reflectionless propagation at an EP in a non-ideal PT-symmetric metasurface composed of two silver ring resonators embedded in a photopolymer matrix. c, Scattering EP realized through spatially engineered loss in a passive metasurface, wherein a tri-meta-atom supercell with selectively introduced lossy elements drives the scattering matrix to an EP, resulting

Scattering EPs were also extended to the visible spectrum using a bilayer design $^{71}$ . A $TiO_{2}$ metagrating atop a silicon subwavelength grating leveraged interlayer loss to isolate lightwave behaviours: 88% retroreflection was obtained for left-side incidence and 85% absorption for right-side incidence at 532 nm (Fig. 3d). The upper metagrating enabled directional control, whereas the lower lossy grating regulated absorption via duty-cycle tuning. Multiple scattering processes between layers allowed eigenvalue coalescence at the EP while maintaining high efficiency, representing a critical advance for nanophotonic applications. Graphene-metamaterial hybrids also showed promise for tunable scattering EPs and EP encircling, assisted by the Fano resonance arising from the coupling between metallic silver strips and graphene nanoribbons $^{72}$ . A dual-band chiral $VO_{2}$ metasurface was proposed to realize tunable EPs via phase transitions, hinting at multi-frequency EP control $^{73}$ .

![](images/243da55b91c35eb9545ee704f742aec8b5bd6e82ffdea1f8ff278bba04aee5da.jpg)

![](images/8d5ac5c0219b1b0344d105eab2a57c3ce2e31a308aef2b827cefab040d4f8b78.jpg)
in extreme angular asymmetry and unidirectional retroreflection. d, High-efficiency scattering EP operating in the visible spectrum, realized in a bilayer metasurface composed of a dielectric metagrating and an interlayer lossy subwavelength grating. At the EP, the structure exhibits near-unity retroreflection for illumination from one side and strong absorption for illumination from the opposite side. Panel a reprinted with permission from ref. 68, © Optical Publishing Group. Panel b reprinted with permission from ref. 69, © Optical Publishing Group. Panel c reprinted with permission from ref. 70, ACS. Panel d reprinted from ref. 71, CC BY 4.0.

Embedding EPs within spatiotemporally modulated or nonreciprocal materials can convert them into tunable nodes for signal routing, isolation and dynamic information transfer $^{74}$ . Extending these principles to quantum, thermal and stochastic regimes will open new possibilities for correlated photon emission, radiative heat regulation and entropy-based sensing. As inverse design and large-area fabrication mature, scattering EPs are poised to evolve from proof-of-concept demonstrations to programmable, multifunctional interfaces that link optical, electronic and thermal channels across integrated systems.

## Jones EPs

Recent advances in plasmonic metasurfaces composed of subwavelength nanostructured meta-atoms have provided a fertile ground for exploring EPs in the context of Jones matrices $^{75,76}$ , which characterize polarization conversions. Although these phenomena could be interpreted within broader scattering frameworks, we specifically focus on Jones matrix EPs, as their recent resurgence has been particularly impactful in polarization-engineered metadevices, especially for subwavelength-scale structure engineering. We review key developments in Jones EP research, focusing on their realization, unique properties and applications in asymmetric polarization control, dynamic wavefront shaping and full-colour holography.

Jones matrices on the reflection side $J = [r_{LL} r_{LR}; r_{RL} r_{RR}]$ encode the polarization response of metasurfaces under circularly polarized light, where $r_{ij}$ represents polarization conversion from j to i. For a planar structure under normal incidence, there is generally the constraint $r_{LL} = r_{RR}$ , and EPs manifest when one of the off-diagonal terms, that is, one of the cross-polarization conversion coefficients vanishes: $r_{LR} r_{RL} = 0$ , leading to a singularity in the parameter space $^{17}$ .

a
![](images/754cf951e4d1f95ff64857ec58b26f98fc32f3b63207f1d28aaafe9cc52a90ae.jpg)
b

c
![](images/eeae8d4f984f4262dec6b941d099064cc3c23328eca8b5f89edb08d8b8c9a630.jpg)

![](images/00013008d24b1f81452c0aa29fe1a31658d432bb3b8f346052474fcee5404b87.jpg)

![](images/e246d65f03b34aed5652ad0d7f3b968c9ea9aa56e5b53880f3e22a2b9867a9a4.jpg)

Fig. 4 | Jones EPs. a, Reflective Jones exceptional point (EP) realized in a plasmonic metasurface based on a metal-insulator-metal architecture with coupled rod-L-shaped antennas. Under normal incidence, the EP is identified by the vanishing amplitude of one cross-polarization channel in the circular basis ( $r_{\text{LR}} = 0$ ), indicating the coalescence of eigenvalues and eigenstates of the Jones matrix. b, Transmissive Jones EP in a PT-symmetric terahertz metasurface composed of coupled silver and lead resonators, wherein loss imbalance and tunable coupling induce a chiral degeneracy and a phase transition between PT-symmetric and PT-symmetry-broken regimes. The top schematics depict the corresponding evolution of the eigen-polarization states, with eigenstate coalescence into a circularly polarized state at the EP. c, A non-Hermitian metasurface composed of split-ring resonators, described
![](images/73cc100f686eec2310228332d8332aa83d91b059c9f59813adc45a0635da93b3.jpg)

At this point, the eigenstates of the Jones matrix coalesce into a single chiral eigen-polarization $^{77}$ . An easy-to-fabricate plasmonic platform working in the visible regions that enables precise control of near-field coupling through antenna geometry and spacing has been realized using a metal–insulator–metal structure and an engineered top layer with coupled rod-L shapes $^{17}$ (Fig. 4a). At the EP, one of the cross-polarization conversion channels reaches a zero singularity, such that right-circularly polarized (RCP) light is preserved without conversion to left-circularly polarized (LCP) light, a hallmark of a non-Hermitian degeneracy. Similarly, a transmissive Jones EP was identified in a terahertz metasurface (Fig. 4b) with coupled PT-symmetric resonators made of different metals, whereby loss imbalance between silver and lead resonators drove the EP and enabled the observation of eigen-polarization evolution and the phase transition between PT-symmetric and PT-symmetry-broken regimes through coupling distance variation $^{27}$ .

Similarly, by tuning radiation frequency and inter-resonator coupling in a non-Hermitian metasurface composed of orthogonally oriented split-ring resonators, it was possible to map the parameter space of an EP, observing unique phenomena such as eigenstate swapping and asymmetric transmission of circularly polarized light $^{78}$ (Fig. 4c). To advance the concept of Jones EPs, mirror-related plasmonic meta-atoms were combined to create pairs of circularly polarized EPs (Fig. 4d), which were further integrated with Pancharatnam–Berry phase encoding to realize asymmetric vectorial holography $^{79,80}$ . By superimposing orthogonal EP eigenmodes, full-polarization vectorial metasurfaces capable of projecting holograms with arbitrary polarization states were achieved $^{79}$ . Engineered rectangular bars have also been demonstrated with twins of Jones EPs with topological phases $^{81}$ . Unlike conventional Pancharatnam–Berry metasurfaces, their EP-pair design eliminates redundant images, enabling high-fidelity polarization multiplexing for optical encryption and data storage. In the meanwhile, various set-ups have been proposed based on plasmonic metasurfaces and coupled ring resonators $^{29,82-93}$ , enabling enhanced sensing, polarization detection or control, and chirality decoupled holography beyond the Pancharatnam–Berry phase $^{94}$ . Jones EPs can go beyond normal incidence and planarity to have unequal co-polarization coefficients $^{95}$ , generating a more general matrix and arbitrary coalesced eigenstates on the Poincaré sphere for EPs beyond circular polarizations $^{96}$ .

d
![](images/f533c0d044809a18803019d27a4d2f7d015e0ae7785c496c6dfdb819013431c1.jpg)
by a Jones transmission matrix in the circular-polarization basis, illustrating asymmetric polarization conversion between right-circularly polarized (RCP) and left-circularly polarized (LCP) light – the conversion from LCP to RCP ( $T_{\mathrm{RL}}$ ) is substantially lower than the conversion from RCP to LCP ( $T_{\mathrm{LR}}$ ) – and eigenstate coalescence at the EP. d, Paired Jones EPs formed by combining mirror-related plasmonic meta-atoms produce circularly polarized degenerate eigenstates of opposite handedness, enabling chiral-selective wavefront control and asymmetrical polarization responses. The EPs are indicated by stars on the Poincaré sphere, with $S_1$ , $S_2$ and $S_3$ representing the Stokes parameters. Panel a reprinted with permission from ref. 17, AAAS. Panel b reprinted with permission from ref. 27, APS. Panel c reprinted from ref. 78, CC BY 4.0. Panel d reprinted with permission from ref. 80, ACS.

Future developments in Jones EPs are expected to extend beyond two-state polarization degeneracies towards multichannel and higher-order regimes, wherein several eigenstates coalesce simultaneously (Box 1). Such multichannel Jones EPs could emerge in materials or metasurfaces with multiple coupled anisotropic axes wherein the Jones matrix generalizes to a higher-dimensional matrix. In parallel, entering the nonlinear regime through Kerr effects, saturable absorption and harmonic generation may transform Jones EPs into power-dependent chirality and nonlinear phase accumulation. Together, these directions suggest a new frontier in which Jones EPs evolve from simple two-mode singularities to complex degeneracies governed by the interplay between materials symmetry, nonlinearity and higher-order non-Hermitian topology.

## Emerging applications of EPs Dynamic control of EPs

Recent advances in the dynamic control of EPs have leveraged electrically tunable platforms to overcome challenges in precisely accessing and manipulating non-Hermitian singularities. Notably, a graphene-based metamaterial demonstrated real-time voltage-driven EP transitions, enabling topological control of light intensity and phase by tuning loss imbalance and detuning frequency, accompanied by Berry phase accumulation during parameter loops enclosing the EP $^{97}$ (Fig. 5a).

In the optical domain, a voltage-controlled topological phase transition between EPs and diabolic points was achieved using a MEMS-integrated chiral metasurface $^{98}$ , wherein electrical actuation of a piezoelectric MEMS mirror continuously tuned the air gap and near-field coupling, enabling rapid reconfigurability and robust polarization conversion with minimal voltage steps (Fig. 5b). In the terahertz regime, the need for single-device dynamic control to mitigate fabrication errors inherent in passive metasurfaces was highlighted, and gated graphene metasurfaces were proposed as a viable platform for precise parameter modulation in terahertz non-Hermitian systems $^{28}$ (Fig. 5c). Ultrafast terahertz switching between anti-chiral Jones EPs with high modulation depth and picosecond speed was obtained by decoupling two spin eigenstates $^{99}$ . Moreover, in situ active control of EP chirality has been achieved using an exceptional-line metasurface (Fig. 5d).

Selective chirality inversion was realized solely through light-induced loss without structural modifications, thereby enabling ultrafast picosecond-scale switching under transient perturbation $^{100}$ . Current implementations primarily target terahertz frequencies using optically controlled loss in metasurfaces $^{99,101}$ , whereas optical phase-change materials are attracting renewed interest for active EP metadevices in the visible and mid-infrared ranges.

EP-related non-Hermitian phase transitions enable arbitrary, robust light steering in reconfigurable non-Hermitian junctions, wherein chiral topological states propagate at gain-loss domain interfaces through the strategic interplay of non-Hermitian and topological physics $^{102}$ . By incorporating nonlinearity, both static edge modes and dynamic phase transitions involving EPs are achieved on timescales of hundreds of picoseconds while preserving topological protection against fabrication disorder $^{103}$ . These advances demonstrate the synergistic interplay between topology and non-Hermiticity, enabling dynamic control capabilities that transcend conventional static systems.

Applications span topological sensors, adaptive polarization devices and EP-enhanced light–matter interactions, promising breakthroughs in topological optoelectronics $^{33,104,105}$ . Such dynamic EP control via electrically tunable metasurfaces represents a paradigm shift in photonics, offering unprecedented precision in probing non-Hermitian phenomena $^{106-108}$ . As these platforms evolve, they will unlock new functionalities in light manipulation, paving the way for devices that harness the topological and chiral properties of EPs $^{105,109,110}$ . A crucial next step is to realize deterministic, multidimensional and multiphysics modulation of EPs through advances in responsive material platforms. In such systems, spatial, temporal and spectral degrees of freedom can be dynamically coupled by integrating diverse actuation mechanisms, including electro-optic, thermo-optic, magneto-optic and optomechanical approaches, within a unified control framework. This material-enabled programmability could lead to the emergence of Floquet EPs, as well as synthetic topological phases that encode temporal nature, nonreciprocal energy flow and coherent state evolution across extended parameter spaces.

## Non-Hermitian braiding

Braiding phenomena in non-Hermitian systems can be exemplified by the complex-valued Bloch energies E of a two-band non-Hermitian lattice tracing intertwined trajectories in the combined $(\mathrm{Re}(E), \mathrm{Im}(E), k)$ space as the crystal momentum traverses the Brillouin zone. Because the two ends of the Brillouin zone are equivalent, these trajectories close to form braids whose topology corresponds to knots or links that characterize the global evolution of the band structure. Non-Hermitian braiding was realized by varying a periodic control parameter, with an EP serving as the topological transition between non-braiding and braiding phases $^{109,111,112}$ . Such complex-energy braiding was experimentally demonstrated in a synthetic Floquet coupled-resonator platform (Fig. 5e), wherein temporal modulation of phase and amplitude using electro-optic modulators emulated a non-Hermitian lattice $^{111}$ . This approach enabled direct visualization and control of braided eigenvalue trajectories, establishing a concrete link between non-Hermitian band topology and knot theory, with relevance to applications in both classical and quantum systems.

This concept has been extended to metasurfaces focusing on Jones matrices $^{109}$ , demonstrating picosecond-scale manipulation of eigenspectrum braids using reconfigurable non-Hermitian metasurfaces. Through femtosecond infrared pulse excitation of a photoconductive semiconductor terahertz metasurface, ultrafast switching between distinct braiding topologies was achieved, enabling dynamic transitions from the Solomon link to either the trefoil knot or the Hopf link.

![](images/5d285454ce2eb013c9f26d30410ecdef4ac06e9ba1f8c62f1bf9642671db4ae8.jpg)

![](images/7d66019b55f948d2a3a63707509813cf9a8de20a92359488f47ce81e497878c0.jpg)

c
![](images/6a2ad20e5643ce415c9d51b14d761388dab36c6b1c68e0f57b98f6768bd7d388.jpg)
b

e
![](images/824e1121a9eb523e82dbdf498c4bf98d7d9d82ee1459a3b6bde3d7e6a15227d9.jpg)
Fig. 5 | Dynamic control and braiding of EP. a, A tunable two-parameter framework to realize exceptional point (EP) devices for topological engineering of terahertz light. The voltage-controlled encircling of the EP results in state exchange. b, Electrically tunable topological phase transition in a non-Hermitian metasurface with a micro-electromechanical system-controlled gold mirror. c, Non-Hermitian gated graphene metasurface for studying dynamics around chiral EPs. d, Optically controlled chirality switch of an EP through loss engineering. The eigenstate of the system at the EP is left-handed circularly

Building on recent advances in non-Abelian physics $^{8,113}$ , non-Hermitian braiding in three-band (or higher) non-Abelian systems promises especially rich phenomena. Two complementary forms can be distinguished. The first form is band braiding, similar to the cases above for two-band braiding: non-Abelian non-Hermitian band braiding is the global momentum-space evolution of multiple non-Hermitian bands across the Brillouin zone, wherein the complex-energy sheets exchange in a manner characterized by braid topology $^{8,114-116}$ . The second form is parameter-space EP braiding: the permutation of eigenmodes induced by adiabatic loops in control-parameter space that encircle EPs, whose arrangement and connectivity themselves exhibit braid structure under parameter evolution $^{117,118}$ .

d
![](images/51ea1f6544f390d047ebb427b2ada0c89137bf3ee679d3ea27886e8e8236dfab.jpg)

![](images/c867ca006f1cae0e3b7d064e51b1514de401c1dfdc1d45712e9929d3417a9c15.jpg)

![](images/2df2c099488cac2a2164afd4c252dbd379d4cb6905516eb9a2b8521cd8471d8c.jpg)

![](images/14f714fd8930f802f3af5aaff1431ed7a7e8025b0b13ef33b5dd85dc4981b338.jpg)

![](images/13eb6f436fd803b3a398c0fcb19f65921e9e209a1221f46a12f261c0aea4616c.jpg)
polarized at a pump power of 40 mW and right-handed circularly polarized at 270 mW. e, Non-Hermitian braiding in a synthetic dimension in the $(\mathrm{Re}(E), \mathrm{Im}(E), k)$ space, with E denoting the complex-valued energies and k the momentum. The left schematic shows the experimental set-up, which involves two fibre optical ring resonators with the same free spectral range $\Omega$ . Panel a reprinted with permission from ref. 97, AAAS. Panel b reprinted from ref. 98, CC BY 4.0. Panel c reprinted from ref. 28, CC BY 4.0. Panel d reprinted with permission from ref. 100, APS. Panel e reprinted from ref. 111, Springer Nature Limited.

Together, these advances in non-Hermitian engineable material platforms around EPs establish a pathway for realizing braiding operations in photonics and beyond. The synergistic combination of gain-loss contrast, temporal tunability and reconfigurable coupling, enabled by semiconductors, plasmonic hybrid systems and adaptive metasurfaces, provides advanced control over wave propagation, energy transfer and signal processing. These platforms not only deepen our understanding of the topological physics of braids $^{117,119}$ but also lay the groundwork for non-Abelian photonic manipulation and quantum computing architectures $^{120,121}$ , wherein braiding provides a robust mechanism for state evolution and error-resilient information encoding.

## Topological properties

Topological features associated with EPs have emerged as pivotal tools for manipulating nontrivial light–matter interactions in non-Hermitian systems $^{8,122-124}$ . The topological encircling of a zero singularity associated with an EP in parameter space enables a robust and continuous $2\pi$ phase accumulation by varying geometries, corresponding to a nontrivial topological charge carried by one of the cross-polarization channels. This property underpins polarization-dependent metaholography in non-Hermitian metasurfaces (Fig. 6a), in which wavefront control is governed by topological winding and is, therefore, insensitive to local deviations $^{17,21}$ . This is the same system we highlighted in Fig. 4a.

Direct observation of winding number switching around EPs was achieved by detecting a $2\pi$ phase accumulation in the reflection signals of the graphene metasurface we discussed in Fig. 5a (Fig. 6b), reflecting a switch in the winding number (from 0 to 1) as the system dynamically encircled an EP $^{97}$ . The concept of gate-tunable winding numbers was further advanced in the graphene device shown in Fig. 5c, wherein electrical control allowed precise reconstruction of Riemann surfaces near EPs, enabling direct modulation of the winding number via gate voltage $^{28}$ (Fig. 6c).

The half-charged nature of bulk Fermi arcs was experimentally validated in the system in Fig. 2b, linking paired EPs in momentum space to half-integer polarization charges in far-field radiation (Fig. 6d). This phenomenon, a hallmark of non-Hermitian topology, was observed via polarimetry along the bulk Fermi arc, revealing mode-switching behaviour in the band structure and an analogy to a Möbius strip $^{37}$ . Collectively, these studies underscore the interplay between EPs, topology and non-Hermitian dynamics $^{8,122}$ , offering transformative strategies for sensing, optoelectronics and chiral photonics.

## Wavefront engineering

In wavefront engineering, Jones EPs hold great potential for applications such as vortex generation and asymmetric holography by leveraging the topological phase singularities they create $^{17,96,125,126}$ . Under the circular-polarization basis and normal incidence, EPs typically manifest as a zero value in one of the off-diagonal terms in the Jones matrix, such as in the cross-polarized conversion channel $r_{LR}$ . This zero represents the complete suppression of polarization conversion from LCP to RCP, creating a singularity point equivalent to perfect absorption.

Building on these principles, a new class of functional devices that exploit the robust topological phase termed exceptional topological phase, obtained by encircling EPs in parameter space, has been realized $^{17}$ . In the meta-structure introduced in Fig. 4a, the structural dimensions along two orthogonal directions are independently tuned to control the desired x-polarized and y-polarized resonances, thereby constructing a zero point in the reflection channel. By leveraging this property, the decoupling of two circularly polarized light beams was obtained: when an RCP beam is incident on the structure, a holographic image of the letter 'C' appears at the designed 30° angle. Conversely, when an LCP beam is incident, no holographic image appears, demonstrating a highly selective polarization response (Fig. 6e).

Owing to the intrinsic handedness of EPs, current EP-based systems are restricted to operating at a specific circular-polarization state, limiting their versatility in engineering arbitrary polarization states. By contrast, Pancharatnam–Berry phase encoding imparts phases of equal magnitude but opposite sign to the two circular polarizations through rotation of polarization-converting elements $^{94,127}$ . Operating precisely at an EP suppresses one polarization conversion channel, allowing the Pancharatnam–Berry phase to be asymmetrically encoded onto the remaining channel. To translate this specific asymmetric response into enhanced wavefront control, the enantiomer of a coupled rot-L-shaped meta-structure was obtained through a general mirror-symmetry strategy $^{79}$ . A mirrored structure resulted in the degeneracy of the eigenstates flipping from RCP to LCP, corresponding to the north and south poles of the Poincaré sphere, respectively. These two singularities give rise to phase vortices characterized by opposite topological charges in parameter space.

Therefore, through the use of a pair of EPs, one can surmount the limitation of circular-polarization states in the output. By precisely tuning the amplitude ratio and phase difference across multiple rows of meta-structures, full polarization-state coverage over the Poincaré sphere can be achieved while preserving the asymmetric imaging properties intrinsic to EP-based systems, unlocking new possibilities for topological wavefront engineering $^{79}$ . Asymmetric full-colour vectorial holography was further demonstrated by combining EP pairs with a wavelength-multiplexing strategy $^{80}$ (Fig. 6f), a strategy with potential for applications in information security and virtual reality. More broadly, the concept of encircling zeros for wavefront engineering is not limited to non-Hermitian EPs, and it has recently been extended to Hermitian Jones matrices to realize co-polarization topologically protected phase modulation for beam steering, focusing and vortex generation $^{128}$ .

The inherent topological protection offered by EPs provides a fundamentally new approach to robust wavefront-shaping devices $^{17,96}$ , wherein phase singularities and their associated vortex structures remain stable against perturbations that would compromise conventional designs. This topologically enabled resilience marks a critical transition from fragile precision optics to inherently stable systems, positioning EP-based wavefront engineering as a practical platform for real-world applications wherein reliability and performance consistency are essential.

## Special EPs and new frontiers

Recently, the focus has shifted towards combining EPs with other singularities, such as bound states in the continuum (BICs) $^{129-132}$ , Dirac points and multiple degeneracies, resulting in the formation of novel EPs with enhanced properties.

For example, in exceptional BICs, several BICs are merged into one EP, forming a new kind of singularity $^{133}$ (Fig. 7a). The resulting state inherits properties from both BIC and EP: it does not radiate and shows extremely high sensitivity to perturbations, making it promising for the realization of exceptional sensing at the nanoscale. Recently, the coexistence of topological BICs and Jones EPs was proposed and verified in an all-dielectric metasurface-embedded PhC, wherein the embedded geometry preserved the momentum-space topology while inducing chiral Jones EPs for polarization-selective responses $^{134}$ . This integration reveals a practical route towards singular optical devices combining robustness, chirality and enhanced sensitivity within a compact dielectric platform. Dirac EPs represent a unique class of spectral degeneracies that bridge two fundamental concepts: Dirac points, characteristic

## Review article

a
![](images/5cb88b439dff6da3a4894e4c06f7629c0898cc36e68dbec7cdbadc4830bb9522.jpg)

b
![](images/77d1ec437a51fc313fc6590dfcb0cb240553c305609488a30bac1e89d4433c68.jpg)

e
![](images/fdd0e26620f29420b98af0346161992caa015d206365c881004163167218fe9c.jpg)

c
![](images/d9fda914f671d669fe7a81963891dc503ce032bf91868814340a28c64ec23c6b.jpg)

![](images/1e69733b4804cc6e00a7181bfe37e42965587886916cd7ef7e5f4d613282e9e2.jpg)

![](images/cae731eccaabaa5f15a53864b60ee475866663d2492263cb5f393329fd4398f9.jpg)

![](images/955d93851619847fa699e031f973e02643108fed2cd20fea6439d9ba96045bfc.jpg)

d
![](images/95970569b7e1f3468d5dc00f3da77e5e2c031eb4e6aa0d1de6383ee6fd0f8415.jpg)

![](images/87a6e287ba3140c6624a45a5de377462e50a910953017b003f7f6a273b4ca8b3.jpg)

f
![](images/b854371c9f3aaa8613612fb9c0f715e424d3267b9171899a0e4567b58125b562.jpg)

Fig. 6 | Topological properties and wavefront engineering of EPs.

a, Topological encircling of an exceptional point (EP) in parameter space to realize meta-holograms by exploiting the accumulated $2\pi$ phase. Here, $L_{1}$ and $L_{2}$ denote the geometric lengths of coupled antennas, which together form a 2D parameter space for encircling the reflection zero on one cross-polarization channel of the Jones matrix.

b, Direct observation of winding number (w) topological switching around an EP, extracted from the phase evolution of the reflected terahertz pulse measured at different time delays after the application of the gate voltage.

c, Gate-tunable winding number indicating the number of times the system winds around the EP. Left: phase of the complex cross-polarized transmission coefficient $t_{RL}$ as a function of frequency and gate voltage. The phase surface exhibits a topological singularity associated with the EP. Right: corresponding trajectories of $t_{RL}$ in the complex plane for different gate voltages: encircling the EP produces a nontrivial phase winding that defines a gate-controlled topological winding number.

d, Half-charged polarization winding associated with the bulk Fermi arc, revealing the fractional topological nature

![](images/5b487de0df94c25282b2368349e3e29fcd0de65989a36557050cfb0e1259214a.jpg)

![](images/1d0b65c642030f8145143e062e9ef401109210127a7d01160ea4a51600834763.jpg)

![](images/3bd814d16a9c5a5cc83749e83f78b3d7fa5b413dd9fb4bcfa0b9881f6b637bd8.jpg)

of paired EPs. In momentum space, the long axis of the polarization ellipse rotates along the bulk Fermi arc, accumulating a total rotation of $180^{\circ}$ . e, Asymmetric meta-holography enabled by the exceptional topological phase around a Jones EP with chiral degeneracy (top), wherein the phase modulation is topologically distinct for the two cross-polarization channels (middle and bottom). f, Asymmetric full-colour vectorial holography using EP pairs of opposite handedness. Top: a metasurface composed of EP-paired meta-atoms enabling asymmetric holographic reconstruction. Bottom: experimental full-colour vectorial holograms of a dice, wherein the three images represent independently encoded intensity, polarization azimuth and polarization ellipticity. LCP, left-circularly polarized; RCP, right-circularly polarized. Panel a reprinted with permission from ref. 17, AAAS. Panel b reprinted with permission from ref. 97, AAAS. Panel c reprinted from ref. 28, CC BY 4.0. Panel d reprinted with permission from ref. 42, AAAS. Panel e reprinted with permission from ref. 17, AAAS. Panel f reprinted with permission from ref. 80, ACS.

of Hermitian systems, and EPs, inherent to non-Hermitian physics $^{135,136}$ . The existence of Dirac EPs was experimentally confirmed through the observation of real eigenvalue coalescence near the degeneracy and coalesced eigenstates at the EP itself (Fig. 7b). These degeneracies exhibit conical dispersion in parameter space, forming a characteristic Dirac-like cone structure while maintaining PT symmetry without spontaneous symmetry breaking. This distinctive combination of properties enables transformative applications in quantum control and topological photonics, including robust mode manipulation, adiabatic state transfer and loss-immune topological transport in non-Hermitian systems. When subject to differential loss, a Dirac point splits into a pair of EPs in the band structure while simultaneously inducing a pair of C points with opposite handedness in the far-field radiation $^{137}$ (Fig. 7c). By breaking the corresponding mirror symmetries, these Dirac point-induced C points can be independently controlled, establishing a correspondence between branch point singularities and polarization singularities and enabling new routes for polarization control and band engineering.

Theoretical proposals for a new type of singular Jones EP – named the super chiral EP $^{138}$ or zero-eigenvalue EP $^{139}$ – were put forward, using vanishing diagonal terms while maintaining the EP condition, enabling complete asymmetric polarization conversion with maximum circular dichroism. The corresponding matrix is of the form $[0;c\ 0]$ featuring three vanishing elements, where c is the only nonvanishing component. This results in coalesced eigenvalues at zero and coalesced eigenstates at the circular polarization. This kind of EP corresponds to coherent perfect absorption owing to the existence of a zero-amplitude eigenvalue $^{140-143}$ . Singular degeneracies can also emerge from projected non-Hermitian subsystems embedded within globally Hermitian scattering metasurfaces $^{144}$ . In particular, the observation of a hybrid degenerate point shows that projecting a unitary and Hermitian scattering matrix onto a reduced channel subspace can naturally anchor a degeneracy at the merging point of two exceptional curves with opposite chirality, without requiring material loss or gain.

New frontiers of EPs involve interfacing with other concepts, including magneto-optical materials, nonreciprocity $^{145,146}$ , lasing $^{147,148}$ and machine learning-empowered design $^{149}$ . The magneto-optical effect has been incorporated into EP photonic systems, wherein the coalescence gives rise to a characteristic square-root dependence of eigenfrequency splitting on perturbation strength $^{10}$ , which greatly enhances magneto-optical interactions and frequency splitting near EPs compared with the linear response of conventional systems $^{150}$ . In a magnetically controlled nonreciprocal system (Fig. 7d), EPs can be accessed and controlled with nonreciprocal scattering, establishing magnonics as a versatile platform for exploring non-Hermitian band theory $^{151}$ . Coupled lasing cavities embedded in a PhC enable lasing EPs and EP tracking above the lasing threshold (Fig. 7e), wherein they become branch points of a nonlinear dynamical system $^{152}$ . Furthermore, machine learning approaches $^{153}$ can be integrated into metasurface design to facilitate the realization of EPs, reducing manual design complexity and enabling the development of smart metasurfaces $^{82}$ . This will greatly ease the structure designs to reach the conditions for higher-order EPs in a high-dimensional parameter space $^{154}$ . A laser operating sufficiently close to an EP may spontaneously generate multi-spectral, multi-modal instabilities, resulting in a self-starting frequency comb: an EP comb $^{155}$ . EP combs offer opportunities for developing on-chip ultrafast light sources, benefiting applications such as precision spectroscopy, high-speed communications, and frequency metrology. Besides, the enhancement of magnonic frequency combs through EPs was demonstrated in a coupled pump-induced magnon mode and Kittel mode system $^{156}$ . This approach enables low-power comb generation with high density, bypassing traditional nonlinear limitations and providing optimized frequency combs for signal processing and sensitive detection. EPs can also enable ultracompact light storage, as demonstrated in an optical microcavity through nonlinear Brillouin scattering (Fig. 7f). The EP-induced abrupt dispersion transition in PT-symmetric optical–acoustic hybrid modes led to a critical slow-to-fast light transition, allowing a pulse to be stored for up to half a millisecond at room temperature $^{157}$ . This compact and integrable approach holds great promise for on-chip optical buffering and quantum information processing.

![](images/9f52a16663f49886a8f18d89a6ea4d259795c8ef1f6a62fae46ccf5059443c9c.jpg)
Fig. 7 | Special EPs and new frontiers. a, Exceptional bound states in the continuum combining concepts of exceptional points (EPs) and bound states in the continuum (BICs), forming coalesced nonradiative states. Two coupled bound states in the continuum (BIC 1 and BIC 2) interact with coupling strength $\kappa$ , while remaining decoupled from the radiation continuum (crossed arrows). One mode experiences intrinsic loss, $\gamma^{\mathrm{int}}$ . The plot shows the real (blue) and imaginary (red) parts of the eigenfrequencies as a function of $\kappa$ , with their coalescence marking the EP. b, Dirac EP exhibiting real-valued eigenspectrum but coalesced eigenstates. c, When differential loss is finite, a Dirac point induces an EP pair in the band structure and a C-point pair in the far-field radiation, whose loci evolve in momentum space with varying differential loss. This behaviour can be realized in 1D, tilted 1D and 2D photonic crystal (PhC) slabs (right panel; top, middle and bottom, respectively). d, An EP in a magnetically controlled nonreciprocal system
composed of two yttrium iron garnet (YIG) spheres side-coupled to a microwave transmission line, wherein an external magnetic field tunes magnon resonances to induce reflectionless scattering states and their coalescence at the EP. e, Lasing at an EP in coupled PhC nanocavities. f, Light can be stored near an EP in a microsphere cavity owing to EP-induced strong dispersion in a nonlinear optical–acoustic hybrid system. When the control (blue) and signal (red) light meet the phase-matching condition, a forward Brillouin acoustic wave (green) is excited, and the group velocity of the signal is reduced to 0 m s $^{-1}$ within the microsphere. LCP, left-circularly polarized; RCP, right-circularly polarized; SBS, stimulated Brillouin scattering. Panel a reprinted from ref. 133, CC BY 4.0. Panel b rerpinted with permission from ref. 174, Jiangfeng Du. Panel c reprinted with permission from ref. 137, APS. Panel d reprinted from ref. 151, Springer Nature Limited. Panel e reprinted from ref. 152, CC BY 4.0. Panel f reprinted from ref. 157, CC BY 4.0.

Looking ahead, the convergence of materials engineering and non-Hermitian design is poised to transform EP research from a conceptual framework in physics into a foundation for next-generation photonic functionalities. Continued progress in magnetic, light-emitting and adaptive materials will enable precise control over gain-loss symmetry, dispersion and mode coupling at the nanoscale, and data-driven inverse design approaches will accelerate the discovery of unconventional EP configurations across the materials design space. Together, these developments mark the emergence of a materials-by-design paradigm, in which EPs evolve from abstract mathematical entities into engineered attributes intrinsic to reconfigurable and multifunctional photonic materials.

## Outlook

We anticipate that future advancements in metamaterial-based EPs will harness emerging phase-change materials, such as $VO_{2}$ (refs. 158,159), GeSbTe (refs. 160,161), perovskite $^{162}$ and magneto-optical materials, including yttrium iron garnet $^{147,163}$ and nickel $^{164,165}$ , to achieve dynamic control and chirality switching with nontrivial topological nature within optical wavelengths. Temporal metamaterials could further enable adaptive tunability and facilitate the exploration of higher-dimensional non-Hermitian light–matter interactions $^{22,166-168}$ . Achieving controllable and localized thermal emission by exploiting EPs in thermal metamaterials could be a promising prospect for thermal control and energy harvesting.

Additionally, higher-order topological charges and interactions among multiple EPs remain largely underexplored, offering both fundamental insights into topological physics and practical strategies for manipulating light propagation. The special degeneracy exceptional nexus $^{169}$ , which is not only a higher-order EP but also the cusp singularity of multiple exceptional arcs, has been shown to carry distinct winding numbers on different complex planes, denoted as the hybrid charge. With the rapid advancement of artificial intelligence $^{149,153,170-172}$ , next-generation research frontiers will probably integrate artificial intelligence-driven metamaterial design for EPs and EP-assisted machine learning, paving the way for intelligent photonic systems and flat optics platforms. Ultimately, the convergence of non-Hermitian physics and materials science will redefine how optical systems are conceived, designed and realized $^{173}$ . As advances in nanofabrication, material synthesis and computational design continue, EPs may evolve into designable states of matter within metasurfaces, PhCs and other engineerable materials including plasmonic, dielectric, tunable and phase-change platforms, wherein dispersion, symmetry and loss are precisely orchestrated. These materials will not only enable reconfigurable and topologically protected functionalities but also serve as universal testbeds for non-Hermitian engineering across optical, acoustic and quantum domains. This convergence signifies a paradigm shift towards programmable, topology-governed material systems, establishing a unifying framework that bridges physics, materials science and next-generation information technologies.

At the same time, several challenges intrinsic to non-Hermitian photonic platforms continue to define the landscape of EP research. A prominent limitation arises from the tension between enhanced parametric sensitivity and operational robustness: as EPs amplify modal response, the associated eigenstate coalescence also increases vulnerability to fabrication disorder, material inhomogeneity and environmental fluctuations. This sensitivity becomes particularly consequential in large-area metasurfaces and densely integrated photonic circuits, in which collective or spatially distributed EPs can be sought. At the materials and device level, realizing scalable, low-power and spatially resolved control of gain-loss distributions at optical frequencies remains a persistent challenge. Practical implementations are often constrained by optical absorption, thermal crosstalk, limited switching endurance and integration compatibility, which together restrict the speed, bandwidth and stability of dynamically reconfigurable EP systems. In parallel, although higher-order and interacting EPs have been extensively explored theoretically, their experimental realization remains comparatively scarce, reflecting the difficulty of simultaneously controlling multiple coupled modes, dispersion, symmetry and non-Hermitian perturbations within high-dimensional parameter spaces. More broadly, the field lacks design frameworks that systematically connect non-Hermitian topology with realistic material dispersion, fabrication tolerance and system-level functionality. Progress towards scalable and application-relevant EP platforms, therefore, tends to depend on approaches that move beyond isolated spectral singularities towards architectures in which robustness, programmability and functionality emerge collectively. In this context, the continued maturation of EP-based engineered materials will be shaped not only by advances in material synthesis and nanofabrication but also by the development of design paradigms that treat non-Hermiticity as an integral and engineable degree of freedom.

Published online: 23 February 2026

## References

1. Li, Z. et al. Non-Hermitian electromagnetic metasurfaces at exceptional points. Prog. Electromagn. Res. 171, 1–20 (2021).

2. Li, A. et al. Exceptional points and non-Hermitian photonics at the nanoscale. Nat. Nanotechnol. 18, 706–720 (2023).

3. Özdemir, ŞK., Rotter, S., Nori, F. & Yang, L. Parity–time symmetry and exceptional points in photonics. Nat. Mater. 18, 783–798 (2019).

4. Miri, M.-A. & Alù, A. Exceptional points in optics and photonics. Science 363, eaar7709 (2019).

5. El-Ganainy, R. et al. Non-Hermitian physics and PT symmetry. Nat. Phys. 14, 11–19 (2018).

6. El-Ganainy, R., Khajavikhan, M., Christodoulides, D. N. & Ozdemir, S. K. The dawn of non-Hermitian optics. Commun. Phys. 2, 37 (2019).

7. Ashida, Y., Gong, Z. & Ueda, M. Non-Hermitian physics. Adv. Phys. 69, 249–435 (2020).

8. Ding, K., Fang, C. & Ma, G. Non-Hermitian topology and exceptional-point geometries. Nat. Rev. Phys. 4, 745–760 (2022).

9. Doppler, J. et al. Dynamically encircling an exceptional point for asymmetric mode switching. Nature 537, 76–79 (2016).

10. Wiersig, J. Review of exceptional point-based sensors. Photonics Res. 8, 1457–1467 (2020).

11. Xu, J. et al. Single-cavity loss-enabled nanometrology. Nat. Nanotechnol. 19, 1472–1477 (2024).

12. Mao, W., Fu, Z., Li, Y., Li, F. & Yang, L. Exceptional-point-enhanced phase sensing. Sci. Adv. 10, eadl5037 (2024).

13. Chen, W., Kaya Özdemir, Ş, Zhao, G., Wiersig, J. & Yang, L. Exceptional points enhance sensing in an optical microcavity. Nature 548, 192–196 (2017).

14. Li, Z., Prasad, C. S., Wang, X., Zhang, D. & Naik, G. V. Sensing beyond the exceptional point for high detectivity. ACS Photonics 11, 2954–2960 (2024).

15. Huang, Y., Shen, Y., Min, C., Fan, S. & Veronis, G. Unidirectional reflectionless light propagation at exceptional points. Nanophotonics 6, 977–996 (2017).

16. Peng, B. et al. Chiral modes and directional lasing at exceptional points. Proc. Natl Acad. Sci. USA 113, 6845–6850 (2016).

17. Song, Q., Odeh, M., Zúñiga-Pérez, J., Kanté, B. & Genevet, P. Plasmonic topological metasurface by encircling an exceptional point. Science 373, 1133–1137 (2021).

18. Sun, J. & Zhou, J. Metamaterials: the art in materials science. Engineering 44, 145–161 (2024).

19. Qiu, C.-W., Zhang, T., Hu, G. & Kivshar, Y. Quo vadis, metasurfaces? Nano Lett. 21, 5461–5474 (2021).

20. Liu, Y. & Zhang, X. Metamaterials: a new frontier of science and technology. Chem. Soc. Rev. 40, 2494–2507 (2011).

21. Schulz, S. A. et al. Roadmap on photonic metasurfaces. Appl. Phys. Lett. 124, 260701 (2024).

22. Bentata, F. et al. Spatially-controlled planar guided crystallization of low-loss phase change materials for programmable photonics. Adv. Mater. 38, e06609 (2026).

23. Cui, T., Bai, B. & Sun, H.-B. Tunable metasurfaces based on active materials. Adv. Funct. Mater. 29, 1806692 (2019).

24. Jung, C., Lee, E. & Rho, J. The rise of electrically tunable metasurfaces. Sci. Adv. 10, eado8964 (2024).

25. Gu, T., Kim, H. J., Rivero-Baleine, C. & Hu, J. Reconfigurable metasurfaces towards commercial success. Nat. Photonics 17, 48–58 (2023).

26. Kuznetsov, A. I. et al. Roadmap for optical metasurfaces. ACS Photonics 11, 816–865 (2024).

27. Lawrence, M. et al. Manifestation of PT symmetry breaking in polarization space with terahertz metasurfaces. Phys. Rev. Lett. 113, 093901 (2014).

28. Baek, S. et al. Non-Hermitian chiral degeneracy of gated graphene metasurfaces. Light Sci. Appl. 12, 87 (2023).

29. Jin, B. et al. High-performance terahertz sensing at exceptional points in a bilayer structure. Adv. Theory Simul. 1, 1800070 (2018).

30. Park, J.-H. et al. Symmetry-breaking-induced plasmonic exceptional points and nanoscale sensing. Nat. Phys. 16, 462–468 (2020).

31. Nag Chowdhury, B., Lahiri, P., Johnson, N. P., De La Rue, R. M. & Lahiri, B. Exceptional-point-enhanced superior sensing using asymmetric coupled-lossy-resonator based optical metasurface. Laser Photonics Rev. 19, 2401661 (2024).

32. Wang, L. et al. Resonant exceptional points sensing in terahertz metasurfaces. Appl. Phys. Lett. 124, 131701 (2024).

33. Li, T. et al. Chip-scale metaphotonic singularities: topological, dynamical, and practical aspects. Chip 3, 100109 (2024).

34. Chen, J. et al. Continuous lines of topological singularities in metasurface scattering matrices: from nodal to exceptional. ACS Photonics 12, 3208–3216 (2025).

35. Su, V.-C., Chu, C. H., Sun, G. & Tsai, D. P. Advances in optical metasurfaces: fabrication and applications [Invited]. Opt. Express 26, 13148 (2018).

36. Chen, Z. & Segev, M. Highlighting photonics: looking into the next decade. eLight 1, 2 (2021).

37. Özdemir, ŞK. Fermi arcs connect topological degeneracies. Science 359, 995–996 (2018).

38. Huang, X., Lai, Y., Hang, Z. H., Zheng, H. & Chan, C. T. Dirac cones induced by accidental degeneracy in photonic crystals and zero-refractive-index materials. Nat. Mater. 10, 582–586 (2011).

39. Lu, L. et al. Experimental observation of Weyl points. Science 349, 622–624 (2015).

40. Zhen, B. et al. Spawning rings of exceptional points out of Dirac cones. Nature 525, 354–358 (2015).

41. Sakoda, K. Proof of the universality of mode symmetries in creating photonic Dirac cones. Opt. Express 20, 25181–25194 (2012).

42. Zhou, H. et al. Observation of bulk Fermi arc and polarization half charge from paired exceptional points. Science 359, 1009–1012 (2018).

43. Cerjan, A. et al. Experimental realization of a Weyl exceptional ring. Nat. Photonics 13, 623–628 (2019).

44. Chen, W., Yang, Q., Chen, Y. & Liu, W. Evolution and global charge conservation for polarization singularities emerging from non-Hermitian degeneracies. Proc. Natl Acad. Sci. USA 118, e2019578118 (2021).

45. Deng, Z.-L., Li, F.-J., Li, H., Li, X. & Alù, A. Extreme diffraction control in metagratings leveraging bound states in the continuum and exceptional points. Laser Photonics Rev. 16, 2100617 (2022).

46. Lin, Z., Pick, A., Lončar, M. & Rodriguez, A. W. Enhanced spontaneous emission at third-order Dirac exceptional points in inverse-designed photonic crystals. Phys. Rev. Lett. 117, 107402 (2016).

47. Yang, Y. et al. Photonic flatband resonances for free-electron radiation. Nature 613, 42–47 (2023).

48. Vicencio Poblete, R. A. Photonic flat band dynamics. Adv. Phys. X 6, 1878057 (2021).

49. Mao, X.-R., Shao, Z.-K., Luan, H.-Y., Wang, S.-L. & Ma, R.-M. Magic-angle lasers in nanostructured Moiré superlattice. Nat. Nanotechnol. 16, 1099–1105 (2021).

50. Kolkowski, R., Kovaios, S. & Koenderink, A. F. Pseudochirality at exceptional rings of optical metasurfaces. Phys. Rev. Res. 3, 023185 (2021).

51. Masharin, M. A. et al. Room-temperature exceptional-point-driven polariton lasing from perovskite metasurface. Adv. Funct. Mater. 33, 2215007 (2023).

52. Masharin, M. A. et al. Giant ultrafast all-optical modulation based on exceptional points in exciton-polariton perovskite metasurfaces. ACS Nano 18, 3447–3455 (2024).

53. Wang, J. et al. Optical bound states in the continuum in periodic structures: mechanisms, effects, and applications. Photonics Insights 3, R01 (2024).

54. Zhang, Y. et al. Observation of polarization vortices in momentum space. Phys. Rev. Lett. 120, 186103 (2018).

55. Zhang, Y. et al. Momentum-space imaging spectroscopy for the study of nanophotonic materials. Sci. Bull. 66, 824–838 (2021).

56. Cerjan, A., Raman, A. & Fan, S. Exceptional contours and band structure design in parity-time symmetric photonic crystals. Phys. Rev. Lett. 116, 203902 (2016).

57. Wang, H. et al. Exceptional concentric rings in a non-Hermitian bilayer photonic system. Phys. Rev. B 100, 165134 (2019).

58. Zhou, H., Lee, J. Y., Liu, S. & Zhen, B. Exceptional surfaces in PT-symmetric non-Hermitian photonic systems. Optica 6, 190 (2019).

59. Isobe, T., Yoshida, T. & Hatsugai, Y. Topological band theory of a generalized eigenvalue problem with Hermitian matrices: symmetry-protected exceptional rings with emergent symmetry. Phys. Rev. B 104, L121105 (2021).

60. Isobe, T., Yoshida, T. & Hatsugai, Y. A symmetry-protected exceptional ring in a photonic crystal with negative index media. Nanophotonics 12, 2335–2346 (2023).

61. Kang, M., Zhang, T., Zhao, B., Sun, L. & Chen, J. Chirality of exceptional points in bianisotropic metasurfaces. Opt. Express 29, 11582–11590 (2021).

62. Wang, C., Sweeney, W. R., Stone, A. D. & Yang, L. Coherent perfect absorption at an exceptional point. Science 373, 1261–1265 (2021).

63. Zhou, H.-T. et al. Underwater scattering exceptional point by metasurface with fluid-solid interaction. Adv. Funct. Mater. 34, 2404282 (2024).

64. Zhou, Z., Jia, B., Wang, N., Wang, X. & Li, Y. Observation of perfectly-chiral exceptional point via bound state in the continuum. Phys. Rev. Lett. 130, 116101 (2023).

2π-phase retardation in non-Hermitian metasurfaces. Laser Photonics Rev. 17, 2200976 (2023).

66. Mikheeva, E. et al. Asymmetric phase modulation of light with parity-symmetry broken metasurfaces. Optica 10, 1287 (2023).

67. Ge, L., Chong, Y. D. & Stone, A. D. Conservation relations and anisotropic transmission resonances in one-dimensional PT-symmetric photonic heterostructures. Phys. Rev. A 85, 023802 (2012).

68. Feng, L. et al. Demonstration of a large-scale optical exceptional point structure. Opt. Express 22, 1760–1767 (2014).

69. Gu, X. et al. Unidirectional reflectionless propagation in a non-ideal parity-time metasurface based on far field coupling. Opt. Express 25, 11778 (2017).

70. Dong, S. et al. Loss-assisted metasurface at an exceptional point. ACS Photonics 7, 3321–3327 (2020).

71. He, T. et al. Scattering exceptional point in the visible. Light Sci. Appl. 12, 229 (2023).

72. Liu, Q. et al. Exceptional points in Fano-resonant graphene metamaterials. Opt. Express 25, 7203–7212 (2017).

73. Gao, F., Zhou, J., Liu, H., Deng, J. & Yan, B. Topological metasurface of tunable, chiral $VO_{2}$ -based system with exceptional points in the dual band. J. Appl. Phys. 135, 063104 (2024).

74. Feng, X. et al. Non-Hermitian hybrid silicon photonic switching. Nat. Photonics 19, 264–270 (2025).

75. Collett, E. Field Guide to Polarization (SPIE, 2005).

76. Balthasar Mueller, J. P., Rubin, N. A., Devlin, R. C., Groever, B. & Capasso, F. Metasurface polarization optics: independent phase control of arbitrary orthogonal states of polarization. Phys. Rev. Lett. 118, 113901 (2017).

77. Kang, M., Chen, J. & Chong, Y. D. Chiral exceptional points in metasurfaces. Phys. Rev. A 94, 033834 (2016).

78. Park, S. H. et al. Observation of an exceptional point in a non-Hermitian metasurface. Nanophotonics 9, 1031–1039 (2020).

79. Yang, Z. et al. Creating pairs of exceptional points for arbitrary polarization control: asymmetric vectorial wavefront modulation. Nat. Commun. 15, 232 (2024).

80. Yang, Z. et al. Asymmetric full-color vectorial meta-holograms empowered by pairs of exceptional points. Nano Lett. 24, 844–851 (2024).

81. Wu, X., Zhao, X., Lin, Y., Lin, F. & Fang, Z. Twins of exceptional points with opposite chirality for non-Hermitian metasurfaces. ACS Photonics 11, 2054–2060 (2024).

83. Gao, F. et al. High-performance full-Stokes polarization detection at exceptional point in a non-Hermitian metasurface. Appl. Phys. Lett. 123, 011705 (2023).

84. Gao, F., Liu, H., Zhou, J., Deng, J. & Yan, B. The exceptional point of PT-symmetry metasurface: topological phase studies and highly sensitive refractive index sensing applications. J. Appl. Phys. 134, 093104 (2023).

85. Hu, S., Wang, C., Du, S., Han, Z. & Gu, C. Dynamic and polarization-independent wavefront control based on hybrid topological metasurfaces. Nano Lett. 24, 2041–2047 (2024).

86. Indu Krishna, K. N. & Roy Chowdhury, D. Thin film sensing near exceptional point utilizing terahertz plasmonic metasurfaces. N. J. Phys. 26, 053033 (2024).

87. Leung, H. M. et al. Exceptional point-based plasmonic metasurfaces for vortex beam generation. Opt. Express 28, 503–510 (2020).

88. Li, J., Fu, J., Liao, Q. & Ke, S. Exceptional points in chiral metasurface based on graphene strip arrays. J. Opt. Soc. Am. B 36, 2492–2498 (2019).

89. Li, Y. et al. Bifunctional sensing based on an exceptional point with bilayer metasurfaces. Opt. Express 31, 492–501 (2023).

90. Li, Z. et al. Parity-time symmetry transition and exceptional points in terahertz metal–graphene hybrid metasurface with switchable transmission and reflection characteristics. Phys. Chem. Chem. Phys. 25, 6510–6518 (2023).

91. Li, Y. et al. Independent control of circularly polarized light with exceptional topological phase coding metasurfaces. Photonics Res. 12, 534–542 (2024).

92. Li, H. et al. Nonlocal metasurface with chiral exceptional points in the telecom-band. Nano Lett. 24, 2087–2093 (2024).

93. Zhao, X. et al. Mode-interference-induced chiral exceptional points in momentum space. Laser Photonics Rev. 18, 2301257 (2024).

94. Xie, X. et al. Generalized Pancharatnam-Berry phase in rotationally symmetric meta-atoms. Phys. Rev. Lett. 126, 183902 (2021).

95. Menzel, C., Rockstuhl, C. & Lederer, F. Advanced Jones calculus for the classification of periodic metamaterials. Phys. Rev. A 82, 053811 (2010).

96. Qin, H. et al. Sphere of arbitrarily polarized exceptional points with a single planar metasurface. Nat. Commun. 16, 2656 (2025).

97. Ergoktas, M. S. et al. Topological engineering of terahertz light using electrically tunable exceptional point singularities. Science 376, 184–188 (2022).

98. Ding, F., Deng, Y., Meng, C., Thrane, P. C. V. & Bozhevolnyi, S. I. Electrically tunable topological phase transition in non-Hermitian optical MEMS metasurfaces. Sci. Adv. 10, eadl4661 (2024).

99. Yu, Z. et al. Creating anti-chiral exceptional points in non-Hermitian metasurfaces for efficient terahertz switching. Adv. Sci. 11, 2402615 (2024).

100. He, W. et al. Loss-enabled chirality inversion in terahertz metasurfaces. Phys. Rev. Lett. 134, 106901 (2025).

101. Wang, L. et al. Photoswitchable exceptional points derived from bound states in the continuum. Light Sci. Appl. 14, 377 (2025).

102. Zhao, H. et al. Non-Hermitian topological light steering. Science 365, 1163–1166 (2019).

103. Dai, T. et al. Non-Hermitian topological phase transitions controlled by nonlinearity. Nat. Phys. 20, 101–108 (2024).

104. Ha, S. T. et al. Optoelectronic metadevices. Science 386, eadm7442 (2024).

105. Lee, H. et al. Chiral exceptional point enhanced active tuning and nonreciprocity in micro-resonators. Light Sci. Appl. 14, 45 (2025).

106. Chen, P.-Y. & Jung, J. PT symmetry and singularity-enhanced sensing based on photoexcited graphene metasurfaces. Phys. Rev. Appl. 5, 064018 (2016).

107. Farhat, M., Yang, M., Ye, Z. & Chen, P.-Y. PT-symmetric absorber-laser enables electromagnetic sensors with unprecedented sensitivity. ACS Photonics 7, 2080–2088 (2020).

108. Park, S. H., Xia, S., Oh, S.-H., Avouris, P. & Low, T. Accessing the exceptional points in a graphene plasmon-vibrational mode coupled system. ACS Photonics 8, 3241–3248 (2021).

109. Hu, Y. et al. Ultrafast control of braiding topology in non-Hermitian metasurfaces. Preprint at https://doi.org/10.48550/arXiv.2410.16756 (2024).

110. Zheludev, N. I. & Kivshar, Y. S. From metamaterials to metadevices. Nat. Mater. 11, 917–924 (2012).

111. Wang, K., Dutt, A., Wojcik, C. C. & Fan, S. Topological complex-energy braiding of non-Hermitian bands. Nature 598, 59–64 (2021).

112. Tong, S. et al. Observation of Floquet-Bloch braids in non-Hermitian spatiotemporal lattices. Phys. Rev. Lett. 134, 126603 (2025).

113. Yang, Y. et al. Non-Abelian physics in light and sound. Science 383, eadf9621 (2024).

114. Guo, C.-X., Chen, S., Ding, K. & Hu, H. Exceptional non-Abelian topology in multiband non-Hermitian systems. Phys. Rev. Lett. 130, 157201 (2023).

115. Long, Y., Xue, H. & Zhang, B. Unsupervised learning of topological non-Abelian braiding in non-Hermitian bands. Nat. Mach. Intell. 6, 904–910 (2024).

116. Zhang, Q. et al. Experimental characterization of three-band braid relations in non-Hermitian acoustic lattices. Phys. Rev. Res. 5, L022050 (2023).

117. Patil, Y. S. S. et al. Measuring the knot of non-Hermitian degeneracies and non-commuting braids. Nature 607, 271–275 (2022).

118. Guria, C. et al. Resolving the topology of encircling multiple exceptional points. Nat. Commun. 15, 1369 (2024).

119. Wojcik, C. C., Wang, K., Dutt, A., Zhong, J. & Fan, S. Eigenvalue topology of non-Hermitian band structures in two and three dimensions. Phys. Rev. B 106, L161401 (2022).

120. Zhang, X.-L. et al. Non-Abelian braiding on photonic chips. Nat. Photonics 16, 390–395 (2022).

121. Bonesteel, N. E., Hormozi, L., Zikos, G. & Simon, S. H. Braid topologies for quantum computation. Phys. Rev. Lett. 95, 140503 (2005).

122. Parto, M., Liu, Y. G. N., Bahari, B., Khajavikhan, M. & Christodoulides, D. N. Non-Hermitian and topological photonics: optics at an exceptional point. Nanophotonics 10, 403–423 (2021).

123. Wang, H. et al. Topological physics of non-Hermitian optics and photonics: a review. J. Opt. 23, 123001 (2021).

124. Nasari, H., Pyrialakos, G. G., Christodoulides, D. N. & Khajavikhan, M. Non-Hermitian topological photonics. Opt. Mater. Express 13, 870–885 (2023).

125. Song, Q., Liu, X., Qiu, C.-W. & Genevet, P. Vectorial metasurface holography. Appl. Phys. Rev. 9, 011311 (2022).

126. Shi, Y. et al. Optical manipulation with metamaterial structures. Appl. Phys. Rev. 9, 031303 (2022).

127. Deng, Z.-L. & Li, G. Metasurface optical holography. Mater. Today Phys. 3, 16–32 (2017).

128. Li, J. et al. Exploiting hidden singularity on the surface of the Poincaré sphere. Nat. Commun. 16, 5953 (2025).

129. Azzam, S. I. & Kildishev, A. V. Photonic bound states in the continuum: from basics to applications. Adv. Opt. Mater. 9, 2001469 (2021).

130. Hsu, C. W., Zhen, B., Stone, A. D., Joannopoulos, J. D. & Soljačić, M. Bound states in the continuum. Nat. Rev. Mater. 1, 16048 (2016).

131. Qin, H. et al. Disorder-assisted real-momentum topological photonic crystal. Nature 639, 602–608 (2025).

132. Qin, H. et al. Arbitrarily polarized bound states in the continuum with twisted photonic crystal slabs. Light Sci. Appl. 12, 66 (2023).

133. Canós Valero, A., Sztranyovszky, Z., Muljarov, E. A., Bogdanov, A. & Weiss, T. Exceptional bound states in the continuum. Phys. Rev. Lett. 134, 103802 (2025).

134. Qin, H. et al. Metasurface-embedded topological photonic crystal. Laser Photonics Rev. https://doi.org/10.1002/lpor.202501032 (2025).

135. Rivero, J. H. D., Feng, L. & Ge, L. Imaginary gauge transformation in momentum space and Dirac exceptional point. Phys. Rev. Lett. 129, 243901 (2022).

136. Wu, Y., Zhu, D., Wang, Y., Rong, X. & Du, J. Experimental observation of Dirac exceptional points. Phys. Rev. Lett. 134, 153601 (2025).

137. Wang, J., Liu, J., Hu, P., Jiang, Q. & Han, D. Topological polarization singularities induced by non-Hermitian Dirac points. Phys. Rev. B 111, 035430 (2025).

138. Li, H. et al. Manifestation of super chiral exceptional points in a plasmonic metasurface. Photonics Res. 12, 2863–2872 (2024).

139. Oh, D. et al. Complete asymmetric polarization conversion at zero-eigenvalue exceptional points of non-Hermitian metasurfaces. Nanophotonics 13, 4409–4416 (2024).

140. Baranov, D. G., Krasnok, A., Shegai, T., Alù, A. & Chong, Y. Coherent perfect absorbers: linear control of light with light. Nat. Rev. Mater. 2, 17064 (2017).

141. Qin, H., Zhang, Z., Wang, J. & Fleury, R. Topological hysteretic winding for temporal anti-lasing. Nat. Commun. 16, 6189 (2025).

142. Ramezani, H., Wang, Y., Yablonovitch, E. & Zhang, X. Unidirectional perfect absorber. IEEE J. Sel. Top. Quantum Electron. 22, 115–120 (2016).

143. Jin, L. & Song, Z. Incident direction independent wave propagation and unidirectional lasing. Phys. Rev. Lett. 121, 073901 (2018).

144. Chen, J. et al. Observation of hybrid degenerate point in projected non-Hermitian metasurfaces. Phys. Rev. Lett. 135, 116601 (2025).

145. Caloz, C. et al. Electromagnetic nonreciprocity. Phys. Rev. Appl. 10, 047001 (2018).

146. Mahmoud, A. M., Davoyan, A. R. & Engheta, N. All-passive nonreciprocal metastructure. Nat. Commun. 6, 8359 (2015).

147. Bahari, B. et al. Nonreciprocal lasing in topological cavities of arbitrary geometries. Science 358, 636–640 (2017).

148. You, J. W. et al. Topological metasurface: from passive toward active and beyond. Photonics Res. 11, B65–B102 (2023).

149. Qian, C., Kaminer, I. & Chen, H. A guidance to intelligent metamaterials and metamaterials intelligence. Nat. Commun. 16, 1154 (2025).

150. Ruan, Y.-P. et al. Observation of loss-enhanced magneto-optical effect. Nat. Photonics 9, 109–115 (2024).

151. Rao, Z. et al. Braiding reflectionless states in non-Hermitian magnonics. Nat. Phys. 20, 1904–1911 (2024).

152. Ji, K. et al. Tracking exceptional points above the lasing threshold. Nat. Commun. 14, 8304 (2023).

153. Xue, Z. et al. Fully forward mode training for optical neural networks. Nature 632, 280–286 (2024).

## Review article

154. Fu, P. et al. Achieving higher-order exceptional points in a terahertz metasurface. Nano Lett. 25, 3773–3780 (2025).

155. Gao, X., He, H., Sobolewski, S., Cerjan, A. & Hsu, C. W. Dynamic gain and frequency comb formation in exceptional-point lasers. Nat. Commun. 15, 8618 (2024).

156. Wang, C. et al. Enhancement of magnonic frequency combs by exceptional points. Nat. Phys. 20, 1139–1144 (2024).

157. Zhu, Y. et al. Storing light near an exceptional point. Nat. Commun. 15, 8101 (2024).

158. Guo, T. et al. Durable and programmable ultrafast nanophotonic matrix of spectral pixels. Nat. Nanotechnol. 19, 1635–1643 (2024).

159. Tripathi, A. et al. Tunable Mie-resonant dielectric metasurfaces based on $VO_{2}$ phase-transition materials. ACS Photonics 8, 1206–1213 (2021).

160. Chu, C. H. et al. Active dielectric metasurface based on phase-change medium. Laser Photonics Rev. 10, 986–994 (2016).

161. Sha, X. et al. Chirality tuning and reversing with resonant phase-change metasurfaces. Sci. Adv. 10, eadn9017 (2024).

162. Tian, J. et al. Phase-change perovskite microlaser with tunable polarization vortex. Adv. Mater. 35, 2207430 (2023).

163. Lv, W. et al. Robust generation of intrinsic C points with magneto-optical bound states in the continuum. Sci. Adv. 10, eads0157 (2024).

164. Kim, D., Baucour, A., Choi, Y.-S., Shin, J. & Seo, M.-K. Spontaneous generation and active manipulation of real-space optical vortices. Nature 611, 48–54 (2022).

165. Kim, D. et al. Dynamic realization of emergent high-dimensional optical vortices. Nat. Commun. 16, 9788 (2025).

166. Engheta, N. Four-dimensional optics using time-varying metamaterials. Science 379, 1190–1191 (2023).

167. Galiffi, E. et al. Photonics of time-varying media. Adv. Photonics 4, 014002 (2022).

168. Yin, S., Galiffi, E. & Alù, A. Floquet metamaterials. eLight 2, 8 (2022).

169. Tang, W. et al. Exceptional nexus with a hybrid topological invariant. Science 370, 1077–1080 (2020).

170. Zenbaa, N. et al. A universal inverse-design magnonic device. Nat. Electron. 8, 106–115 (2025).

171. Zheng, H. et al. Multichannel meta-imagers for accelerating machine vision. Nat. Nanotechnol. 19, 471–478 (2024).

172. Nadell, C. C., Huang, B., Malof, J. M. & Padilla, W. J. Deep learning for accelerated all-dielectric metasurface design. Opt. Express 27, 27523–27535 (2019).

173. Wu, N. et al. Intelligent nanophotonics: when machine learning sheds light. eLight 5, 5 (2025).

174. Fadelli, I. The first experimental observation of Dirac exceptional points. Phys.org https://phys.org/news/2025-04-experimental-dirac-exceptional.html (2025).

175. Horn, R. A. & Johnson, C. R. Matrix Analysis (Cambridge Univ. Press, 2012).

176. Haus, H. A. & Huang, W. Coupled-mode theory. Proc. IEEE 79, 1505-1518 (1991).

177. Chong, Y. D., Ge, L. & Stone, A. D. PT-symmetry breaking and laser-absorber modes in optical scattering systems. Phys. Rev. Lett. 106, 093902 (2011).

178. Bergholtz, E. J., Budich, J. C. & Kunst, F. K. Exceptional topology of non-Hermitian systems. Rev. Mod. Phys. 93, 015005 (2021).

179. Heiss, W. D. The physics of exceptional points. J. Phys. A Math. Theor. 45, 444016 (2012).

180. Schucan, T. H. & Weidenmüller, H. A. The effective interaction in nuclei and its perturbation expansion: an algebraic approach. Ann. Phys. 73, 108–135 (1972).

181. Bender, C. M. & Hook, D. W. PT -symmetric quantum mechanics. Rev. Mod. Phys. 96, 045002 (2024).

182. Wiersig, J. Petermann factors and phase rigidities near exceptional points. Phys. Rev. Res. 5, 033042 (2023).

## Acknowledgements

H.Q. and Z.Z. thank J. Wang and Q. Chen for the helpful discussions. C.-W.Q. acknowledges the financial support of the Ministry of Education, Republic of Singapore (grant numbers A-8002152-00-00, A-8002458-00-00 and A-8003643-00-00), and the Competitive Research Program Award (NRF-CRP26-2021-0004 and NRF-CRP30-2023-0003) from the National Research Foundation, Prime Minister's Office, Singapore. Q.S. acknowledges funding support from the National Natural Science Foundation of China (no. 12474388) and the Guangdong Basic and Applied Basic Research Foundation (no. 2025A1515011483).

## Author contributions

C.-W.Q., P.G., R.F., H.Q., Q.S., B.L. and J.Z. discussed the content of the Review. H.Q., W.L., Z.Z., M.L., Z.Y., J.L. and Q.S. wrote the first draft. All authors reviewed and edited the final manuscript.

## Competing interests

The authors declare no conflict of interest.

## Additional information

Peer review information Nature Reviews Materials thanks Liang Feng for his contribution to the peer review of this work.

Publisher's note Springer Nature remains neutral with regard to jurisdictional claims in published maps and institutional affiliations.

Springer Nature or its licensor (e.g. a society or other partner) holds exclusive rights to this article under a publishing agreement with the author(s) or other rightsholder(s); author self-archiving of the accepted manuscript version of this article is solely governed by the terms of such publishing agreement and applicable law.

© Springer Nature Limited 2026
