Article

# Non-Hermitian topological whispering gallery

https://doi.org/10.1038/s41586-021-03833-4

Received: 29 January 2021

Accepted: 16 July 2021

Published online: 29 September 2021

Check for updates

Bolun Hu $^{1}$ , Zhiwang Zhang $^{1✉}$ , Haixiao Zhang $^{1}$ , Liyang Zheng $^{2}$ , Wei Xiong $^{1}$ , Zichong Yue $^{1}$ , Xiaoyu Wang $^{1}$ , Jianyi Xu $^{1}$ , Ying Cheng $^{1✉}$ , Xiaojun Liu $^{1✉}$ & Johan Christensen $^{2✉}$

In 1878, Lord Rayleigh observed the highly celebrated phenomenon of sound waves that creep around the curved gallery of St Paul's Cathedral in London $^{1,2}$ . These whispering-gallery waves scatter efficiently with little diffraction around an enclosure and have since found applications in ultrasonic fatigue and crack testing, and in the optical sensing of nanoparticles or molecules using silica microscale toroids. Recently, intense research efforts have focused on exploring non-Hermitian systems with cleverly matched gain and loss, facilitating unidirectional invisibility and exotic characteristics of exceptional points $^{3,4}$ . Likewise, the surge in physics using topological insulators comprising non-trivial symmetry-protected phases has laid the groundwork in reshaping highly unconventional avenues for robust and reflection-free guiding and steering of both sound and light $^{5,6}$ . Here we construct a topological gallery insulator using sonic crystals made of thermoplastic rods that are decorated with carbon nanotube films, which act as a sonic gain medium by virtue of electro-thermoacoustic coupling. By engineering specific non-Hermiticity textures to the activated rods, we are able to break the chiral symmetry of the whispering-gallery modes, which enables the out-coupling of topological ‘audio lasing’ modes with the desired handedness. We foresee that these findings will stimulate progress in non-destructive testing and acoustic sensing.

Understanding topological phases in non-Hermitian settings has become a thriving area in many research fields such as condensed-matter physics, cold atoms physics, and classical optics and acoustics. Specifically, efforts are motivated by the quest to expand on exclusive topological phases in non-Hermitian settings, which have no Hermitian counterpart $^{7,8}$ . Among the most acclaimed examples are the non-Hermitian skin effect, the unconventional non-Bloch bulk-boundary correspondence and topological lasers $^{9-18}$ . Topological lasers constitute an appealing area in terms of applications, in that topologically resistant edge states—which are robust against imperfections and fabrication defects—would be set to laser when combined with optically active media. Lasers are fundamentally non-Hermitian, but have lately gained this unprecedented benefit owing to the added ingredient of topology reported in active Su–Schrieffer–Heeger arrays $^{19,20}$ , coupled ring resonators $^{21,22}$ , exciton–polariton insulators $^{23}$ and topological photonic crystals $^{24,25}$ .

Creating non-Hermiticity for sound waves in terms of a gain medium let alone an equivalent laser is highly challenging; yet loudspeakers with appropriate gain circuits or the acoustoelectric effect using piezo semiconductors could do the job $^{3,26}$ . However, a more flexible and tunable approach is posed by employing thermoacoustics where fluctuating Joule heating is converted into sound, for example, when an alternating current is applied to a conductor $^{27}$ . This electro-thermoacoustic coupling has been proven to be notably efficient in carbon nanotube (CNT) films owing to their low heat capacitance and thermal inertia, permitting broadband and high-pressure acoustic wave generation $^{28,29}$ (Methods). Here we construct a whispering-gallery (WG) insulator using a triangular sonic crystal made of acrylonitrile butadiene styrene (ABS) rods that is capable of sustaining topological valley edge states along its interface that are protected by the underlying lattice symmetry. The valley degree of freedom was first used in condensed-matter physics, but has since found use in artificial lattices in which the suppression of intervalley scattering enables robust and compact guiding of light or sound $^{30-32}$ . To get the acoustic valley edge states to 'lase', a topological lattice needs to be implemented by means of an acoustic gain medium, which we accomplished by pasting CNT films around the insulator rods. Our experimental measurements reveal how the WG mode chirality can be broken—that is, we split the symmetry among the clockwise (CW) and the counterclockwise (CCW) resonances by adequately tuning the electrically imposed non-Hermiticity. Not only do these topological WG modes revolve around the enclosed sonic insulator through its complex edge states, but we also demonstrate the capability to out-couple amplified and focused sound emission at audible frequencies.

## Non-Hermitian topological lattice

Before we analyse the electrically assisted thermoacoustic generation of sound in a topological WG insulator, we discuss topological modes in lattices executing acoustic gain. In doing so, we theoretically introduce an active fluid layer of thickness t whose real effective mass density is that of air $\rho_{0}$ , whereas the imaginary component is controlled through a non-Hermiticity gain factor $\beta$ , that is, $\rho = (1 + i\beta)\rho_{0}$ (Methods). In a time-reversal symmetric setting, we consider a triangular sonic lattice in air comprising rigid cylinder trimers that are coated by the active layer as illustrated in Fig. 1a. This so-called kagome lattice $^{33,34}$ facilitates a valley pseudospin that can be retrieved prototypically by breaking the spatial-inversion symmetry to access opposite Berry curvatures at the Brillouin zone corners. At moderate gain levels, the topological properties can indeed be well characterized by a quasi-Hermitian topological valley-Chern number (Methods). Hence, when the trimers are arranged upright, the topological valley-Chern index is $C^{(K,K')} = \pm \frac{1}{2}$ ; however, when inverted, the index becomes $C^{(K,K')} = \mp \frac{1}{2}$ . In Fig. 1b, we compute the corresponding complex band diagram involving gain with $\beta = 0.05$ (solid (dashed) curves represent the real (imaginary) bands). The gapped Dirac cone features a pair of pseudospin valley states $K_{1}/K_{2}$ comprising negative imaginary eigenfrequencies that account for sound amplification. Interestingly, when the non-Hermiticity is steadily increased, the real component of the bulk states shows no effect (Fig. 1c); however, a linear thresholdless growth of their imaginary components against $\beta$ is clearly seen.

![](images/42532da133eadfb6e7b0e261c5ebe2a116870a0e640ce9d6701e5ef870ae2619.jpg)

![](images/35c2197f4afaf33571bf24b11201f5f052371f3af239eaa01a679e160c76c92a.jpg)

![](images/c920934eb4305f4ee37ab81d2807bf3f991db0ee38017195ba1cde4c753033aa.jpg)

![](images/2781011c06a10eddf1ebdc0be586b97d33a298bfc8ae491dde9cb8b40616887b.jpg)

![](images/6749670e5e2787d4c4cd1f0ad96853fdb2885a7a7282b91811e1bef74b00ddf9.jpg)
Fig. 1 | Complex band diagram of a sonic topological insulator with acoustic gain. a, Illustration of the non-Hermitian topological lattice. The inset shows the primitive cell, composed of three rigid cylinders that are covered by an active layer of thickness $t = 0.1r$ . b, Finite-element simulations show the real (solid lines) and imaginary (dashed lines) bands in the presence of gain via the non-Hermiticity factor $\beta = 0.05$ . The red and blue coloured curves represent the first and second bands, respectively. c, Increasing the factor $\beta$ shows how the real valley states $K_1$ (red lines) and $K_2$ (blue lines) remain spectrally unaffected; their imaginary part (dashed lines), however, grows linearly in amplification strength. d, Valley-edge dispersion for a sonic insulator with a zigzag-type interface, which separates two topological lattices of opposite valley-Hall phase. Grey dots and purple curves represent the real bulk and valley-projected edge states, respectively. The shaded blue region marks the topological bandgap. The inset shows a schematic of the interface. e, Hermitian (grey) versus non-Hermitian (coloured dots) in-gap edge states along the imaginary frequency axis. Insets: valley edge states at their corresponding values of $\beta$ .

Whether amplifying acoustic states also sustain at interfaces is best demonstrated by designing a non-zero valley-Chern index, $\Delta C^{(K,K^{\prime})}=C_{\mathrm{I}}^{(K,K^{\prime})}-C_{\mathrm{II}}^{(K,K^{\prime})}\neq0$ , across the interface between two adjacent insulators (I and II) as depicted in the inset of Fig. 1d. The real band diagram shows the well known valley-projected edge states inside the bandgap. Moreover, Fig. 1e shows the imaginary counterpart, which indicates that topological edge states undergo sound amplification along the interface in the presence of gain ( $\beta\neq0$ ) by virtue of their negative values (Im(f)<0) and enhanced edge-state intensities.

## Experimental realization

We used three-dimensional (3D) printing to make the rods out of ABS plastic, including two voids to act as the moulds into which electrodes were cast, which connect to the CNT films that were pasted around the rods (Fig. 2a). In the last step, several rods were assembled into a triangular lattice and mounted to an electrical circuit board, as depicted in Fig. 2b (Methods). With an appropriately applied time-varying current, each coated rod operates as an acoustic source through the electro-thermoacoustic coupling. Moreover, as shown in Fig. 2, we are able to electrically invoke a phase delay to each rod in the unit cell, which has the advantage of enabling both phase and amplitude control over extended finite configurations. The thermoacoustic emission of sound simulated in Fig. 2c illustrates the case in which all rods emit sound coherently with zero phase delay $\phi = 3(\phi_{j+1} - \phi_j) = 0$ with $j = 1, 2$ , which results in a zero relative phase emanating each triangle edge as measurements show in Fig. 2d. However, with deliberate phase jumps within the unit cell, that is, $\phi_1 = 0$ , $\phi_2 = 2\pi/3$ and $\phi_3 = -2\pi/3$ , according to the underlying lattice symmetry, the acoustic edge emissions acquire equivalent relative phase advancements as shown in Fig. 2e, f. These findings imply that an appropriate gain texture within the truncated lattice facilitates deterministic edge emission control (see Supplementary Video 1 for the time evolution), which is the cornerstone of non-Hermitian chiral symmetry breaking as we discuss in the following.

## Whispering-gallery mode splitting

In Fig. 3a, we show the fabricated finite triangular lattice capable of hosting interface valley states. These states are borne from the underlying lattice symmetry and run along the domain wall (green dashed line) across which upright ( $\Delta < 0$ ) and inverted ( $\Delta > 0$ ) cylinder trimers are arranged (here $\Delta$ symbolizes the geometrical perturbation $^{34}$ ). Moreover, as the magnification near the domain wall in Fig. 3b shows, only a narrow area is considered, the cylinders of which are coated by the acoustic gain medium. Conventional WG modes have degenerate CW and CCW resonances, the frequencies of which are identical. As shown in Fig. 2, acoustic waves emanating from the edges of a triangular lattice can be readily controlled through appropriately phase-engineered gain in the unit cell. Along the topologically non-trivial interface that encompasses a triangular circumference (Fig. 3a), we are able to launch valley-projected edge states using a three-channel signal generator (Methods) where the electrical phase and amplitude can be precisely controlled to power the CNT-film-decorated rods surrounding the topological domain wall. After multiple acoustic round trips along the circumference of length L, the phase is accumulated from each of the three edges of the WG insulator. Thus, several WG resonance triplets are generated through the non-Hermitian phase arrangement:

$$
\begin{array}{l} k _ {0} L / 3 = 2 \pi m \\ k _ {\pm} L / 3 = 2 \pi m \pm 2 \pi / 3, \end{array}\tag{1}
$$

where m is an integer, and $k_{0}$ and $k_{\pm}$ correspond to the achiral, CW (−) and CCW (+) WG wavenumbers, respectively (Methods).

d
c
![](images/2584a1f0876f1ca10653b2f880cd94587c796b0c5cd5982f4c1c702570f2ed1c.jpg)
$\phi = 0$

![](images/920bb87397100b7dd1279f75f544a9564109e12e7e954db25f5ea31015cfd873.jpg)
e

![](images/c3ce6a7c96118b47f808e092ae3b9d6f3ed72f037803c579e60e7974f6aa5e57.jpg)

$\phi = 2\pi$
![](images/b16f726688ef4d326fd8f0d5e2b27bc53a50ec2a4598fd14ba78c8eaba22dcde.jpg)

![](images/a551b90c7aaefe5c8b4da51e298652c1af089f17645beac27a557d31a0ef5666.jpg)
Fig. 2 | Assembly and non-Hermitian phase engineering, a, ABS rods are 3D-printed with moulds into which contacts are cast, whereupon CNT films are pasted around each rod. b, Several groups including six unit cells are assembled in a triangular truncated lattice geometry while being electrically connected to a circuit board. Within each unit cell, we are able to ascribe a specific emission phase of the electro-thermoacoustically generated sound field. c, All three gain-assisted rods emit sound simultaneously with zero delay among them, that is, $\phi = 3(\phi_{j+1} - \phi_j) = 0$ with $j = 1, 2$ , as shown by the phase maps of the pressure field at $f = 8.8\mathrm{kHz}$ within the topological bandgap. d, Simulations (Sim.) and

![](images/7ff905853e9c8640b05f0ec038a76ba1cd30f8fa8c6182f37ffa7d31f39010e2.jpg)
experiments (Exp.) depict equal phases emanating from the edges at the lattice near-fields. e, By contrast, now the phase increment acquires $2\pi/3$ to assume a full gain cycle of $\phi = 2\pi$ . f, Here, the breaking of chirality is made possible through asymmetric phase textures, giving rise to different edge radiation phases. The insets in c, e show a magnification of the edge near-fields. In between c and e, the unit cell is depicted, comprising the individual phase contributions $\phi_{j}$ . Blue shaded regions in d, f represent the frequency range of the topological bandgap. Experimental data in d, f are represented as mean ± s.d. of six independent measurements.

We experimentally detect the WG modes by averaging the signal probed by microphones at several points along the domain wall, followed by quantification of the amplification factor compared with a straight interface (Methods). When all coated cylinders are set to emit sound simultaneously ( $\phi = 0$ ), the excited in-gap valley edge state breathes in the absence of angular momentum (achiral) at frequency $f_{0} = 8,995\mathrm{Hz}$ (Fig. 3c). Theoretically, we show that at that frequency, only identical bidirectional modes exist, unlike the chiral modes at $f_{-} = 8,885\mathrm{Hz}$ and $f_{+} = 9,099\mathrm{Hz}$ , whose forward and backward wave components both destructively interfere (Methods). Contrary to this, the breathing mode $(f_0)$ ceases to exist at $\phi = 2\pi$ , whereas the two in-gap chiral WG modes each acquire a complete handedness, that is, the chiral symmetry is broken and the $f_{-}(f_{+})$ WG resonance is set to emit topologically spinning sound in a strict CW (CCW) fashion. The corresponding detected and simulated amplification factors are shown in Fig. 3d, e at gain advancement phases of $\phi = \pi$ and $\phi = 2\pi$ , respectively. Additional measurements in Fig. 3f again clearly show how the amplification of the breathing mode decays with growing phase $\phi$ in contrast to the two growing chiral modes. To visualize this behaviour, we map the acoustic pressure of the WG lattice for the said resonances. With gain phases from $\phi = 0$ to $\phi = 2\pi$ , Fig. 3g-i clearly shows both chiral and achiral edge-state confinements at the respective WG modes (see Supplementary Video 2 for the time evolution). The influence of the inherent viscosity and the robustness of the structure are studied carefully in Methods.

## Chirality routing

We add a router to the topological WG insulator to make the out-coupling of the non-Hermitian WG modes through either of its two ports possible, as shown in Fig. 4. Moreover, this approach allows for the separation of acoustic signals of opposite propagation chiralities (Methods). Spectrally, we both compute and observe the pressure amplitude at both router ports (Fig. 4a for output 1 and Fig. 4b for output 2) when the gain phase is chosen to be $\phi = 2\pi$ . The $f_{-}$ WG mode shows a remarkable peak when measured at output 1 owing to the conservation of the CW angular momentum preserved at the router. By contrast, sound only emerges at output 2 when CCW-polarized by virtue of a strong peak at $f_{+}$ . Compared with the scenario of coherent emission ( $\phi = 0$ ) at which sound radiates as collimated beams from both ports through the WG breathing mode at $f_{0}$ (Fig. 4c), waves emanate through selective output ports in the form of highly directive beams in accordance with their polarization. In particular, the CW (CCW)-polarized WG mode $f_{-}(f_{+})$ , which is carried by the respective valley projected edge state, emerges highly focused from output 1 (output 2), as seen in Fig. 4d. This functionality is displayed through simulated spatial acoustic energy maps at two different gain phase structures, as shown in Fig. 4e. Here, directive sound beams split symmetrically through both ports at $f_0(\phi = 0)$ and selectively in response to the propagation chirality at $f_{\pm}(\phi = 2\pi)$ . We anticipate that the highly directional and electrically controlled amplified sound beams could lead to an improved image quality by reducing the effects of poor resolution and speckles, which may have an impact in non-destructive testing and acoustic sensing.

![](images/4413412a1e06cf5c4568d42f0459c754105dce4c6d4a0f17b86cfe1adc678d88.jpg)

![](images/8c677445815b562d6366431a1b443c4df4131d58f5124d9a6f38db1d5528e4f7.jpg)

![](images/9cfbfdb5ecfb7ee4c7fbafc0570d1cf4340604561adf299e909f35283b3c7ccf.jpg)

![](images/255fb48a1738bc9f8dee37159f1cb8fc93422fa62c3e3482c622a3fc4817643e.jpg)

![](images/cb4dbccec92caa01ca482052f0a52a8a7d17b3c70a6c889da231f91bb20b8a8c.jpg)

![](images/441645270e3f00cc8010c29939578117d2608ca3a3d489b56acfa37bef7e23bc.jpg)

![](images/317cf100aae4c61c2230e63723d5150623983576a98ac7043f41ffefa55a6391.jpg)
Fig. 3 | Topological WG mode splitting. a, Photograph of the fabricated topological WG insulator, including a triangular domain wall (green dashed line). Insets show photographs of the cylinder trimers wrapped without or with CNT films. b, The cyan zone in a, highlighting the domain wall, that is, the topological interface involving coated and bare rods. c–e, For three different gain-phase textures $-\phi = 0$ (c), $\phi = \pi$ (d) and $\phi = 2\pi$ (e)—spectrally resolved amplification factors are shown through both simulations and experiments. $f_{-} = 8,885$ Hz, $f_{0} = 8,995$ Hz and $f_{+} = 9,099$ Hz represent three eigenfrequencies

## Discussion and outlook

In summary, we have used the electro-thermoacoustic coupling from CNT films as a gain medium in sonic valley-Hall lattices to set up non-Hermitian topological edge states. On the basis of this approach, we devised a whispering-gallery resonator whose conventional degenerate resonances are mode split and controlled through the phase that is imprinted on the gain elements. We collect these mode-split whispering-gallery modes by routing highly collimated audible beams on the basis of their chiralities. We foresee that the combination of topologically protected acoustic resonances with non-Hermitian ingredients in the form of gain may enable new technological avenues beyond the already far-reaching scientific implications. Amplified steering and guiding of sound in a topologically robust environment may improve acoustic communication systems. When scaled to micrometre and nanometre scales, electromechanical filtering in broadband cellular telecommunication networks has the potential to capitalize on non-Hermitian gigahertz topology. Also, combining sonic lattices with CNT films may facilitate engineered active control.

![](images/c799339547c060b9abfc860ae577dd7362b9ad63dc294bab0bbb84f4215ee2e6.jpg)
corresponding to the WG mode order m=27. The colour bar represents the intensity $\pm I_{x}$ along the lower domain wall of the rightward-/leftward-going wave, that is, the CCW/CW mode. f, For several gain phases $\phi$ , the amplification factors among the three in-gap modes are experimentally evaluated and fitted (Fit.). g-i, Chirality and field control of the three resonances with $f_{-}(\mathbf{g})$ , $f_{0}(\mathbf{h})$ and $f_{+}(\mathbf{i})$ at three different gain-phase textures. The acoustic simulations are conducted for finite topological WG insulators as depicted in a. Experimental data in f are represented as mean $\pm$ s.d. of four independent measurements.

![](images/66c2f5e2adb2630840c87ea0ec5148677dc8ba5d3c81d90dd98c24af12e53484.jpg)

![](images/00e6e831b49de16958a12ebee923e3e62304fdd09a68fcde6c4e655690066302.jpg)

![](images/331a8861c0633f93994c16c2117fc49e1f1e037d855bc1c642f4834e8121dc6d.jpg)
Fig. 4 | Routing of amplified topological WG modes. a, b, Simulated (curves) and experimentally measured (dots) pressure amplitude spectra at two spatially distant ports: output 1 (a) and output 2 (b). The gain phase is set to $\phi = 2\pi$ . Inset: picture of the WG lattice including the two-port router. c, d, Simulated directional far-field radiation patterns of the out-coupled
topological chiral and achiral WG modes at gain phases $\phi = 0$ (c) and $\phi = 2\pi$ (d). e, Acoustic energy maps of the WG lattice are shown for the corresponding modes at their respective frequencies $f_{-} = 8,889$ Hz, $f_{0} = 9,001$ Hz and $f_{+} = 9,106$ Hz. The inherent viscosity $\mu = 2.9072 \times 10^{-5}$ Pa s and the thermal conductivity $\chi = 0.0258$ W (m K) $^{-1}$ of air are considered in the simulations.

## Online content

Any methods, additional references, Nature Research reporting summaries, source data, extended data, supplementary information, acknowledgements, peer review information; details of author contributions and competing interests; and statements of data and code availability are available at https://doi.org/10.1038/s41586-021-03833-4.

1. Lord Rayleigh The Theory of Sound Vol. II, 1st edn (MacMillan, 1878).

2. Lord Rayleigh CXII. The problem of the whispering gallery. Phil. Mag. 20, 1001–1004 (1910).

3. Fleury, R., Sounas, D. L. & Alù, A. Parity–time symmetry in acoustics: theory, devices, and potential applications. IEEE J. Sel. Top. Quantum Electron. 22, 121–129 (2016).

4. Gupta, S. K. et al. Parity–time symmetry in non-Hermitian complex optical media. Adv. Mater. 32, 1903639 (2020).

5. Zhang, X., Xiao, M., Cheng, Y., Lu, M.-H. & Christensen, J. Topological sound. Commun. Phys. 1, 97 (2018).

6. Khanikaev, A. B. et al. Photonic topological insulators. Nat. Mater. 12, 233–239 (2013).

7. Gong, Z. et al. Topological phases of non-Hermitian systems. Phys. Rev. X 8, 031079 (2018).

8. Foa Torres, L. E. F. Perspective on topological states of non-Hermitian lattices. J. Phys. Mater. 3, 014002 (2019).

9. Lee, T. E. Anomalous edge state in a non-Hermitian lattice. Phys. Rev. Lett. 116, 133903 (2016).

10. Wang, M., Ye, L., Christensen, J. & Liu, Z. Valley physics in non-Hermitian artificial acoustic boron nitride. Phys. Rev. Lett. 120, 246601 (2018).

11. Zhang, Z., López, M. R., Cheng, Y., Liu, X. & Christensen, J. Non-Hermitian sonic second-order topological insulator. Phys. Rev. Lett. 122, 195501 (2019).

12. Zhao, H. et al. Non-Hermitian topological light steering. Science 365, 1163–1166 (2019).

13. Yao, S. & Wang, Z. Edge states and topological invariants of non-Hermitian systems. Phys. Rev. Lett. 121, 086803 (2018).

14. Song, F., Yao, S. & Wang, Z. Non-Hermitian topological invariants in real space. Phys. Rev. Lett. 123, 246801 (2019).

15. Okuma, N., Kawabata, K., Shiozaki, K. & Sato, M. Topological origin of non-Hermitian skin effects. Phys. Rev. Lett. 124, 086801 (2020).

16. Xiao, L. et al. Non-Hermitian bulk-boundary correspondence in quantum dynamics. Nat. Phys. 16, 761–766 (2020).

17. Helbig, T. et al. Generalized bulk-boundary correspondence in non-Hermitian topoelectrical circuits. Nat. Phys. 16, 747–750 (2020).

18. Weidemann, S. et al. Topological funneling of light. Science 368, 311–314 (2020).

19. Parto, M. et al. Edge-mode lasing in 1D topological active arrays. Phys. Rev. Lett. 120, 113901 (2018).

20. St-Jean, P. et al. Lasing in topological edge states of a one-dimensional lattice. Nat. Photon. 11, 651–656 (2017).

21. Zhao, H. et al. Topological hybrid silicon microlasers. Nat. Commun. 9, 981 (2018).

22. Bandres, M. A. et al. Topological insulator laser: experiments. Science 359, eaar4005 (2018).

23. Klembt, S. et al. Exciton-polariton topological insulator. Nature 562, 552–556 (2018).

24. Bahari, B. et al. Nonreciprocal lasing in topological cavities of arbitrary geometries. Science 358, 636–640 (2017).

25. Zeng, Y. et al. Electrically pumped topological laser with valley edge modes. Nature 578, 246–250 (2020).

26. Hutson, A. R., McFee, J. H. & White, D. L. Ultrasonic amplification in CdS. Phys. Rev. Lett. 7, 237–239 (1961).

27. Arnold, H. & Crandall, I. The thermophone as a precision source of sound. Phys. Rev. 10, 22–38 (1917).

28. Xiao, L. et al. Flexible, stretchable, transparent carbon nanotube thin film loudspeakers. Nano Lett. 8, 4539–4545 (2008).

29. Aliev, A. E., Lima, M. D., Fang, S. & Baughman, R. H. Underwater sound generation using carbon nanotube projectors. Nano Lett. 10, 2374–2380 (2010).

30. Ma, T. & Shvets, G. All-Si valley-Hall photonic topological insulator. New J. Phys. 18, 025012 (2016).

31. Ye, L. et al. Observation of acoustic valley vortex states and valley-chirality locked beam splitting. Phys. Rev. B 95, 174106 (2017).

32. Lu, J. et al. Observation of topological valley transport of sound in sonic crystals. Nat. Phys. 13, 369–374 (2017).

33. Ni, X., Gorlach, M. A., Alù, A. & Khanikaev, A. B. Topological edge states in acoustic kagome lattices. New J. Phys. 19, 055002 (2017).

34. Zhang, Z. et al. Directional acoustic antennas based on valley-Hall topological insulators. Adv. Mater. 30, 1803229 (2018).

Publisher's note Springer Nature remains neutral with regard to jurisdictional claims in published maps and institutional affiliations.

© The Author(s), under exclusive licence to Springer Nature Limited 2021

## Article

## Methods

Numerical simulations, device fabrications and measurements
The numerical results presented in this work were calculated using the commercial finite-element-method simulation software COMSOL Multiphysics. In the band-structure and pressure-field calculations of the lossless system, physical models were established and analysed in the pressure acoustic module, including the detailed structures with actual geometric dimensions. The inherent viscosity $\mu=2.9072\times10^{-5}$ Pa s and the thermal conductivity $\chi=0.0258\;W\;(\mathrm{m}\;\mathrm{K})^{-1}$ of air were introduced to the simulations conducted by the thermoviscous acoustic module when the inherent viscosity of the background air is taken into consideration. The boundaries of the 3D-printed core cylinders can be modelled as hard-wall boundary conditions owing to the large acoustic impedance mismatch between the matrix medium of air and the printing materials of ABS. The standard parameters used for air were the mass density $\rho_{0}=1.2059\;kg\;m^{-3}$ and sound speed $c_{0}=343.58\;m\;s^{-1}$ under an ambient pressure of 1 atm at a temperature of 21 °C. In each basic unit cell as shown in Fig. 1a, there were three identical cylindrical rods with a radius r=0.30 cm, placed on the vertexes of an equilateral triangle, with the adjacent centre-to-centre interval D=0.69 cm. The lattice constant is a=2.17 cm. The largest mesh element size was lower than one-tenth of the shortest incident wavelength. We refined the meshes surrounding the thin meta-fluid layer with thickness t=0.1r. The effective acoustic gain of the meta-fluid layer shown in Fig. 1 was modelled through a positive imaginary part of the mass density $i\beta\rho_{0}\;(\rho_{0}\;for\;air)$ . The active CNT film attached to each gain rod can be regarded as a thermal-acoustic gain, which functions as a vibrating surface with the normal displacement $d_{j}=d_{0}e^{i\phi_{j}}=d_{0}e^{i(j-1)\frac{\phi}{3}}$ , where j=1,2,3 is the number of rods in the unit cell and $d_{0}=0.01r$ represents the initial displacement. In the band-structure calculations, Floquet periodic boundary conditions were implemented at the boundaries of the periodic cells or strips. In the pressure-field calculations, plane-wave radiation conditions were imposed at the exterior facing the air domain to eliminate interference from reflected waves. To obtain the amplification ratio, we first averaged the absolute pressure fields along the infinite straight topological waveguides with the frequency ranging from 8,840 Hz to 9,160 Hz as the reference, which can be labelled as $P_{0}$ . After introducing the topological WG insulator, the averaged absolute pressure fields along the bottom domain wall were calculated as P. Then, the amplification ratio can be obtained as $P/P_{0}$ ; the results are plotted in Fig. 3c–e.

The assembled device was composed of two parts: the precision-fabricated gain elements and the customized printed circuit boards (PCBs). Figure 2a illustrates the fabrication work flow of a single element. We used CNT films synthesized by a high-temperature chemical-vapour-deposition method from the same batch to guarantee the uniformity. The CNT films were cut into small rectangular pieces. We experimentally measured the sheet resistance of each film using a four-probe meter (type RTS-2, which complies with ASTM F84) and screened out the films with the sheet resistance ranging from $1.3 \Omega sq^{-1}$ to $1.6 \Omega sq^{-1}$ , which fulfil the experimental requirement. Each piece of rectangular CNT film was pasted tightly around the surface of the rigid cylindrical cores, which were 3D-printed with heat-resistant ABS plastic. The diameter and height of cylinder were 6 mm and 14 mm, respectively. Two edges of the CNT film were fixed inside the two vertical slots, inside which we also cast silver-paste contacts. These contacts were soldered to a double-layer top PCB (yellow or blue in photos in Figs. 2–4), which were coupled to the six-layer bottom PCB motherboard (green) via connectors, from which the programmable circuit is controlled to achieve full electro-thermoacoustic control.

During the measurements, one plate of Plexiglass was used to cover the whole sample. In this scenario, the 2D approximation was applicable as the planar waveguide supported the propagating mode uniformly along the rod axis for the wavelengths under consideration. Cone-shaped sound absorbing foams were mounted around the testing area to minimize the boundary reflections from open space. As shown in Extended Data Fig. 1, the excitation signals were controlled by two clock-synchronized arbitrary waveform generators (NI PXIe-5423) through pre-programmed LabVIEW software, and were amplified by the power amplifiers (YAMAHA P-5000). Afterwards, we calibrated the excitation signals to eliminate the phase and amplitude offset generated by the asynchronous power amplifiers. Local pressure fields and the phases were measured by inserting 1/4-inch condensed microphones (GRAS 40PH) into the top plate at the designated positions. The output signals of the microphones were acquired by a digitizer (NI PXI-4499). In the experiments, we carried out accurate temperature control of the device with the help of an infrared thermal imager (FOTRIC 348) and a thermocouple thermometer (Fluke 52-II).

## Effective Hamiltonian

Let us begin with the effective Hamiltonian in Hermitian systems without considering a gain medium. Based on the k·p method $^{35,36}$ , the wave dynamics around the Dirac cone at the K point is given by

$$
\Delta \mathcal {H} _ {\mathrm{D}} \boldsymbol {\Psi} = \lambda_ {\mathrm{D}} \boldsymbol {\Psi},\tag{2}
$$

where $\Psi$ and $\lambda_{D}$ represent the wave functions and the reduced Dirac frequency respectively. We demonstrate that the existence of the Dirac cone at the K point in our sonic crystal is protected by the $C_{3v}$ symmetry. From the schematic of the unit cell shown in Extended Data Fig. 2a, it is clear that a $C_{3}$ rotational symmetry (blue arrows) exists together with three mirrors $M_{1}$ , $M_{2}$ and $M_{3}$ , leading to the appearance of the doubly degenerate Dirac cones at the boundaries of the first Brillouin zone, as can be seen from the band diagram shown in Extended Data Fig. 2b. Considering this symmetry protection, the reduced Hamiltonian around the K point can be mapped into a standard massless Dirac Hamiltonian $\Delta\mathcal{H}_{\mathrm{D}} = v_{\mathrm{D}}(\Delta k_{x}\sigma_{x} + \Delta k_{y}\sigma_{y})$ , and $\lambda_{D} = f - f_{D}$ with f the sound wave frequency and $f_{D}$ the Dirac frequency; $v_{D}$ is the group velocity; $\Delta k_{i} (i = x, y)$ is the distance from the Dirac points; and $\sigma_{i} (i = x, y)$ represents the Pauli matrix. As depicted in Extended Data Fig. 2d, rotating the cylinders in the unit cell by a non-zero angle of $\theta \neq n\pi/3 (n = 1, 2, 3)$ reduces the symmetry from $C_{3v}$ to $C_{3}$ . In this case, we demonstrate that the Dirac cones at the K/K' points will be gapped as shown in the band diagram of Extended Data Fig. 2e with $\theta = -\pi/6$ . The rotation operation introduces perturbations and leads to an effective mass term in the Dirac Hamiltonian

$$
\Delta \mathcal {H} \boldsymbol {\Psi} = (\Delta \mathcal {H} _ {\mathrm{D}} + b _ {0} \sigma_ {z}) \boldsymbol {\Psi} = \lambda \boldsymbol {\Psi},\tag{3}
$$

where $b_{0}$ is the effective mass caused by the symmetry reduction and $\lambda = f - f_{D} - a_{0}$ , with $a_{0}$ a bias quantity to slightly shift the frequency due to the perturbation of symmetry reduction. Then the dispersion relations near the K point can be derived by

$$
f = f _ {\mathrm{D}} + a _ {0} \pm \sqrt {v _ {\mathrm{D}} ^ {2} (\Delta k _ {x} ^ {2} + \Delta k _ {y} ^ {2}) + b _ {0} ^ {2}}.\tag{4}
$$

Here taking the case with the rotation $\theta = -\pi /6$ as an example, the corresponding parameters $v_{\mathrm{D}}\approx 29.72\mathrm{ms}^{-1},a_0\approx 45.95\mathrm{Hz}$ and $b_{0}\approx 525.25\mathrm{Hz}$ can be retrieved from the numerical data in Extended Data Fig. 2e. Accordingly, the theoretically predicted dispersion relations near the K point from the k-p method are added into Extended Data Fig. 2b, e as marked by black solid lines, showing good agreement with the numerical data from the finite-element method.

Next, we coat the rigid cylinders by the gain medium with the factor $\beta$ . Note that the rotational and mirror symmetries of the system cannot be broken by the introduced non-Hermiticity. As a result, the existence of the Dirac cone at the K point with $\theta = 0$ and the gapped valley states with $\theta = -\pi/6$ still can be predicted when the acoustic gain is considered, as shown in the band diagrams of Extended Data Fig. 2c, f, respectively. The small amount of gain with $\beta = 0.05$ can be considered as a small perturbation on the wave dynamics around the Dirac cone, which suggests the governing equation under $C_{3v}$ symmetry to be

$$
\Delta \mathcal {H} _ {\mathrm{D}} ^ {\prime} \boldsymbol {\Psi} = (\Delta \mathcal {H} _ {\mathrm{D}} + \mathrm{i} \gamma_ {0} \sigma_ {0}) \boldsymbol {\Psi} = \lambda_ {\mathrm{D}} \boldsymbol {\Psi}.\tag{5}
$$

The gain term is approximated by $\gamma_{0}\approx t_{0}\beta$ where $t_{0}\approx-439.4$ is obtained from numerical data. As the gain term $i\gamma_{0}\sigma_{0}$ is purely diagonal, we can rewrite the governing equation as

$$
\Delta \mathcal {H} _ {\mathrm{D}} \pmb {\Psi} = \lambda_ {\mathrm{D}} ^ {\prime} \pmb {\Psi},\tag{6}
$$

with $\lambda_{\mathrm{D}}^{\prime} = \lambda_{\mathrm{D}} - \mathrm{i}\gamma_{0}$ . When comparing equation (2) with equation (6), it is apparent that the systems share the same form except that $\lambda^{\prime}$ contains an imaginary part stemming from gain. At this stage thus, it can be concluded that in the presence of gain $\Delta \mathcal{H}_{\mathrm{D}}$ remains Hermitian; however, the eigenvalues will show wave amplification.

Subsequently, on reducing the symmetry $C_{3v}$ to $C_{3}$ to gap the Dirac cone, we obtain

$$
\Delta \mathcal {H} ^ {\prime} \pmb {\Psi} = (\Delta \mathcal {H} _ {\mathrm{D}} + b _ {0} ^ {\prime} \sigma_ {z}) \pmb {\Psi} = \lambda^ {\prime} \pmb {\Psi},\tag{7}
$$

with the modified mass term $b_{0}^{\prime}=b_{0}+\mathrm{i}\gamma_{2}$ and $\lambda^{\prime}=\lambda-\mathrm{i}(\gamma_{0}+\gamma_{1})$ . For $\beta=0.05$ , we find that $\gamma_{0}\approx-21.97\ Hz$ , $\gamma_{1}\approx0.78\ Hz$ and $\gamma_{2}\approx6.19\ Hz$ . Equation (7) also shows a similar form compared to equation (3) except that the effective mass and the eigenfrequency are modified by gain. We then obtain the following dispersion relation

$$
f _ {\pm} = f _ {D} + a _ {0} + \mathrm{i} (\gamma_ {0} + \gamma_ {1}) \pm \sqrt {\nu_ {D} ^ {2} (\Delta k _ {x} ^ {2} + \Delta k _ {y} ^ {2}) + (b _ {0} + \mathrm{i} \gamma_ {2}) ^ {2}}.\tag{8}
$$

Accordingly, the predicted dispersion relations (real part) for $\beta=0.05$ around the Dirac point that are marked by black solid lines in Extended Data Fig. 2c, f show good agreement with the numerical data from the finite-element method.

## Topological invariant

We first consider the Hermitian system, that is, without gain. According to the effective Hamiltonian derived from the k·p method, a massive Dirac Hamiltonian $\Delta\mathcal{H}=v_{\mathrm{D}}(\Delta k_{x}\sigma_{x}+\Delta k_{y}\sigma_{y})+b_{0}\sigma_{z}$ is introduced by breaking the mirror symmetry. Owing to the time-reversal symmetry, the massive Dirac Hamiltonians at the K and K' points satisfy the relation $\Delta\mathcal{H}(K)\rightarrow-\Delta\mathcal{H}(K')$ . Hence, the effective mass has the opposite sign at the K and K' points with $M_{K}=-M_{K'}$ . The non-zero mass term induces a local quadratic Berry curvature around the K/K' points, which can be integrated at each valley as the valley-Chern index $^{32,37}$ .

$$
C _ {\mathrm{K}, \mathrm{K} ^ {\prime}} = \frac {1}{2} \operatorname{sgn} (M _ {\mathrm{K}, \mathrm{K} ^ {\prime}}).\tag{9}
$$

Taking the sonic crystal with $\theta = -\pi /6$ as an example (shown in Extended Data Fig. 2d), the effective mass terms $M_{\mathrm{K}} = b_0$ and $M_{\mathrm{K}^{\prime}} = -b_0$ at the K and $\mathbf{K}^{\prime}$ valleys guarantee the topological valley-Chern index as $C_{\mathrm{K}}^{\mathrm{I}} = \frac{1}{2}$ and $C_{\mathrm{K}^{\prime}}^{\mathrm{I}} = -\frac{1}{2}$ , respectively. Note that the zero Chern number $C = C_{\mathrm{K}} + C_{\mathrm{K}^{\prime}}$ is preserved under time-reversal symmetry. However, the mass term changes the sign when the unit is rotated to $\theta = \pi /6$ , which results in the topological valley-Chern index as $C_{\mathrm{K}}^{\mathrm{II}} = -\frac{1}{2}$ and $C_{\mathrm{K}^{\prime}}^{\mathrm{II}} = \frac{1}{2}$ . Therefore, for a domain wall separating these two regions, the topological invariant difference across the domain wall can be obtained as

$$
\Delta C _ {\mathrm{K}} = C _ {\mathrm{K}} ^ {\mathrm{II}} - C _ {\mathrm{K}} ^ {\mathrm{I}} = - 1, \quad \Delta C _ {\mathrm {K^ {\prime}}} = C _ {\mathrm {K^ {\prime}}} ^ {\mathrm{II}} - C _ {\mathrm {K^ {\prime}}} ^ {\mathrm{I}} = 1.\tag{10}
$$

According to the bulk-boundary correspondence, there will exist the topological edge states with the negative velocity projected by K valley, leading to the CW WG modes in the proposed device. The CCW WG modes are supported by the topological edge states with the positive velocity projected by K' valley.

Lastly, for the non-Hermitian systems with $\beta = 0.05$ , the Dirac dispersion is described by equation (6) comprising a Hermitian Hamiltonian but imaginary eigenvalue $\lambda'$ accounting for gain. When the $C_{3v}$ symmetry is reduced to $C_3$ , the effective mass terms at the K and K' points are slightly modified to $M_{\mathrm{K}} = b_0 + \mathrm{i}\gamma_2$ and $M_{\mathrm{K}'} = -b_0 - \mathrm{i}\gamma_2$ . In our system, the parameters of $b_0 = 525.25\mathrm{Hz}$ and $\gamma_{2} \approx 6.19\mathrm{Hz}$ satisfy the relations $\gamma_{2} \ll b_{0}, M_{\mathrm{K}} \approx b_{0}$ and $M_{\mathrm{K}'} \approx -b_{0}$ . Consequently, the introduced gain with $\beta = 0.05$ can be regarded as a small perturbation to the system, the topological properties of which are thus sufficiently well characterized by a quasi-Hermitian topological valley-Chern number.

## Comparison between the meta-fluid and thermal-acoustic gain models

Here we demonstrate that the ideal meta-fluid layer and the thermal-acoustic layer we developed show the same effect for realizing the enhanced sound field. Indeed, Extended Data Fig. 3 shows the enhancement of the outward scattering field amplitude by a single cylinder under the inward radiation of coaxial cylindrical waves, before and after introducing the meta-fluid layer (thermal-acoustic gain layer). To obtain the scattering enhancement, the scattering field from a bare rigid rod ( $\beta=0$ ) is first determined as a reference and then compared with covering the rod with a meta-fluid or thermal-acoustic layer ( $\beta>0$ ). The corresponding scattering field distributions at $f=9.1$ kHz and $\beta=0.05$ , calculated with both the meta-fluid and the thermal-acoustic gain model, are shown in Extended Data Fig. 3a, b, respectively. Extended Data Fig. 3c depicts the evolution of scattering enhancement with gain factor $\beta$ at the frequency of 9.1 kHz, in which it can be clearly seen that the scattering fields using both the meta-fluid model and the thermal-acoustic gain model are enhanced linearly with the increase of $\beta$ and show almost identical variation. In other words, such scattering fields characterizing the sound field enhancement via a thermal-acoustic layer can be analogously described through a meta-fluid layer. Extended Data Fig. 3d shows the evolution of the scattering enhancement with frequency at the gain factor of $\beta=0.05$ . Both the results obtained through the meta-fluid model and the thermal-acoustic gain model increase slightly with frequency, which can be neglected in practice due to the small increment. The evolutions confirm the equivalence of the two models at different frequencies. Note that the amplitude of the heat source is set as $q=\beta A_{0}f$ , with $A_{0}$ the thermal energy flow into air and $f$ the frequency, because the pressure radiation of CNT film depends linearly with frequency $^{38}$ . The radius of the rod is set to $r=3$ mm, and the thickness of the layer of the meta-fluid/heat source is set to $t=0.1r$ . We calculate the average amplitude of the scattering sound field within a $\lambda$ -wide ring located 2 $\lambda$ away from the surface of the rod.

## Characterizations of CNT film

We compare the excitation properties of the CNT film and the regular paper basin speaker through experiments as shown in Extended Data Fig. 4a. In the experiment, a sinusoidal electric signal was used to stimulate the CNT film and the speaker, separately. The acoustic and vibration signals were measured through a microphone (GRAS 40PH) and a laser vibrometer (Polytec PSV-500-HV). For comparison, we kept the diaphragm of the speaker localized at the same place as the CNT film to keep the distance between the measured object and microphone/vibrometer unchanged. Besides, it is worth noting that the heating effect is proportional to $^{27}$

$$
R I ^ {2} = R I _ {0} ^ {2} \sin^ {2} (2 \pi f t) = \frac {R I _ {0} ^ {2}}{2} [ 1 - \cos (4 \pi f t) ],\tag{11}
$$

with the supplied alternating electrical signal $I = I_{0} \sin(2\pi ft)$ and R for the film's resistance. As a result, the frequency of the input electrical signal should be set as f/2 (4.4 kHz, for example) if the required acoustic radiation from the CNT film with the frequency $f(8.8 \text{ kHz})$ is needed in the experiments. The experimentally measured acoustic and vibration signals are described in Extended Data Fig. 4b, c, respectively. We can clearly observe that the acoustic pressure amplitudes are almost the

## Article

same in both situations. However, the surface vibration displacement of the CNT film is almost zero but not for the paper basin speaker. This indicates that the CNT film generates an acoustic signal through thermoacoustic coupling rather than the traditional surface vibrations. When the alternating current (sinusoidal signal in our experiment) passes through the surface of the CNT film possessing a low heat capacity, the heat transmits to the ambient air rapidly and the air around the CNT film is expanded and compressed periodically, which generates acoustic waves.

In Extended Data Fig. 4d, we measured the electrical impedance of the CNT film, which can be expressed as $Z = |Z| \mathrm{e}^{-\mathrm{i}\theta_Z}$ , within the frequency range from $40\mathrm{Hz}$ to $20\mathrm{kHz}$ through a precision impedance analyser (Agilent 4294A). The results shown in Extended Data Fig. 4e clearly indicate that the impedance $|Z|$ of the CNT film stays consistent around $1.181 \pm 0.005\Omega$ . The phase of the impedance $\theta_Z$ remains at $0^{\circ} \pm 1^{\circ}$ , demonstrating that the value of the capacitive reactance and inductive reactance are zero. In short, the CNT film is a pure resistive element with a certain resistance in the frequency range from $40\mathrm{Hz}$ to $20\mathrm{kHz}$ .

In addition, the radiative spreading of single coated rod was measured at the frequencies of 8 kHz, 8.5 kHz and 9 kHz, which is illustrated in Extended Data Fig. 4f. The approximately omnidirectional directivity confirms good uniformity of the curved CNT film.

## Non-Hermitian mode splitting

The CW and CCW modes that are highly localized along the domain wall of the topological WG insulator can be equivalently treated as waves that revolve around a triangular enclosure. Thus, the entire path of the domain wall encompasses three connected sections, that is, three connected waveguides of equal cross-section as shown in Extended Data Fig. 5. As the CW/CCW modes are highly localized, their propagation along the domain wall can be simplified as the transmission of 1D waveguide modes described by a 1D Helmholtz equation

$$
\frac {\mathrm{d} ^ {2} p}{\mathrm{d} x ^ {2}} + k ^ {2} p = 0,\tag{12}
$$

where $k = \omega/v$ with angular frequency $\omega$ and sound velocity v in the waveguide, and p being the pressure. On the basis of equation (12), the pressure at one end of the waveguide is a function of the other end. For instance, the pressure at junction 1 relates to junctions 2 and 3 as

$$
p _ {2} = \cos (k l) p _ {1} - \mathrm{i} \sin (k l) p _ {1}, \quad p _ {3} = \cos (k l) p _ {1} + \mathrm{i} \sin (k l) p _ {1},\tag{13}
$$

where l is the length of each waveguide. For the triangular domain wall with a perimeter L, we have l = L/3. From equation (13), we obtain the relation

$$
p _ {2} + p _ {3} = 2 \cos (k l) p _ {1}.\tag{14}
$$

Taking all the junctions into account, the following governing equation of the acoustic waveguide model can be derived

$$
\left( \begin{array}{c c c} 0 & \frac {1}{2} & \frac {1}{2} \\ \frac {1}{2} & 0 & \frac {1}{2} \\ \frac {1}{2} & \frac {1}{2} & 0 \end{array} \right) \left( \begin{array}{c} p _ {1} \\ p _ {2} \\ p _ {3} \end{array} \right) = \Omega \left( \begin{array}{c} p _ {1} \\ p _ {2} \\ p _ {3} \end{array} \right),\tag{15}
$$

where $\Omega = \cos(kl)$ . Equation (15) has three eigenvalues $\Omega = 1$ , $\Omega = -1/2$ and $\Omega = 1/2$ with the following WG modes:

$$
k _ {0} L / 3 = 2 m \pi , \quad k _ {+} L / 3 = \frac {2 \pi}{3} + 2 m \pi , \quad k _ {-} L / 3 = - \frac {2 \pi}{3} + 2 m \pi ,\tag{16}
$$

where m is an integer. Given the velocity v in the waveguide, the corresponding eigenfrequencies are obtained through $f = k\nu/2\pi$ .

Next we consider that the pressure at each junction is the sum of the CW and the CCW modes $p_{i}=A\mathrm{e}^{-\mathrm{i}kx_{i}}+B\mathrm{e}^{\mathrm{i}kx_{i}}$ where $A(B)$ is the amplitude of the CCW (CW) mode. By choosing $p_{1}=A+B$ as the reference, the other two junctions obtain the following expressions

$$
p _ {2} = A \mathrm{e} ^ {\mathrm{i} \varphi - \mathrm{i} k l} + B \mathrm{e} ^ {\mathrm{i} \varphi + \mathrm{i} k l}, p _ {3} = A \mathrm{e} ^ {\mathrm{i} \zeta - 2 \mathrm{i} k l} + B \mathrm{e} ^ {\mathrm{i} \zeta + 2 \mathrm{i} k l},\tag{17}
$$

where $\varphi(\zeta)$ is the accumulated phase from junction 1 to junction 2 (3) as collected through gain. As shown in Extended Data Fig. 5, it should be noted that the accumulated phase $\varphi$ from junction 1 to 2 can be obtained from the radiative phase $\alpha_{3}$ along edge 3 and $\alpha_{2}$ along edge 2 according to the relation $\varphi = \alpha_{3} - \alpha_{2}$ . In the same way, the accumulated phase $\zeta$ from junction 1 to 3 satisfies the relation $\zeta = \alpha_{1} - \alpha_{2}$ . In matrix form the pressure fields read

$$
\left( \begin{array}{c} p _ {1} \\ p _ {2} \\ p _ {3} \end{array} \right) = U \binom{A}{B}, \text {with} U = \left( \begin{array}{c c} 1 & 1 \\ \mathrm{e} ^ {\mathrm{i} \varphi - \mathrm{i} k l} & \mathrm{e} ^ {\mathrm{i} \varphi + \mathrm{i} k l} \\ \mathrm{e} ^ {\mathrm{i} \zeta - 2 \mathrm{i} k l} & \mathrm{e} ^ {\mathrm{i} \zeta + 2 \mathrm{i} k l} \end{array} \right).\tag{18}
$$

When substituting those junctions pressure fields into equation (15) and multiplying $U^{\dagger}/3$ (where $\dagger$ denotes the conjugate transpose) on both sides, the following equation is obtained

$$
Q \binom{A}{B} = \Omega \left( \begin{array}{c c} 1 & s \\ s ^ {*} & 1 \end{array} \right) \binom{A}{B}, \text {with} Q = \left( \begin{array}{c c} q _ {1 1} & q _ {1 2} \\ q _ {1 2} ^ {*} & q _ {2 2} \end{array} \right).\tag{19}
$$

Therein, the matrix elements are expressed as

$$
s = \frac {1}{3} (1 + e ^ {2 i k l} + e ^ {4 i k l}),\tag{20a}
$$

$$
q _ {1 1} = \frac {1}{3} [ \cos (\varphi - \zeta + k l) + \cos (\varphi - k l) + \cos (\zeta - 2 k l) ],\tag{20b}
$$

$$
q _ {1 2} = \frac {1}{3} \left[ e ^ {3 i k l} \cos (\varphi - \zeta) + e ^ {i k l} \cos \varphi + e ^ {2 i k l} \cos \zeta \right],\tag{20c}
$$

$$
q _ {2 2} = \frac {1}{3} [ \cos (\varphi - \zeta - k l) + \cos (\varphi + k l) + \cos (\zeta + 2 k l) ].\tag{20d}
$$

Now we demonstrate that once the frequency and the phases $\varphi$ and $\zeta$ are given, the mode behaviour can be analysed accordingly.

Case I. For $k = k_{-}$ , equation (19) simply reduces to

$$
Q \binom{A}{B} = \Omega \binom{A}{B},\tag{21}
$$

with eigenvalue $\Omega = -1/2$ . When the gain phase $\phi = 2\pi$ , which results in the phases $\varphi = -2\pi/3$ and $\zeta = 2\pi/3$ , is introduced into the system as has been accomplished in this work, equation (21) has the non-trivial solution

$$
\binom{A}{B} = \binom{\mathbf {0}}{\mathbf {1}}.\tag{22}
$$

Equation (22) directly suggests that the CCW mode is completely suppressed $(A=0)$ while the CW mode $(B=1)$ only is allowed to propagate along the domain wall, which is shown in the last plot of Fig. 3g. On the contrary, changing the signs to $\varphi=2\pi/3$ and $\zeta=-2\pi/3$ results in $[A;B]=[1;0]$ , which leads to a reversed chirality. Note that for the introduced gain phase $\phi=0$ accompanied by the resulted phases $\varphi=\zeta=0$ , the CW and CCW chiral modes cannot be excited. In this case, the governing equation, that is, equation (15), has the following eigenmodes at eigenvalue $\Omega = -1/2$

$$
\left( \begin{array}{c} p _ {1} \\ p _ {2} \\ p _ {3} \end{array} \right) = \left( \begin{array}{c} - 1 \\ 0 \\ 1 \end{array} \right) \text {or} \left( \begin{array}{c} p _ {1} \\ p _ {2} \\ p _ {3} \end{array} \right) = \left( \begin{array}{c} - 1 \\ 1 \\ 0 \end{array} \right).\tag{23}
$$

The components of these eigenmodes indicate that these modes can be excited only when two of the junctions possess the opposite signs. However, for $\varphi=\zeta=0$ , the pressures at the three junctions have identical signs. Hence, the WG modes cannot be supported.

Case II. For $k = k_{+}$ , equation (19) reduces to the identical form as equation (21). However, the introduced phases $\varphi = -2\pi/3$ and $\zeta = 2\pi/3$ result in the non-trivial solution

$$
\binom{A}{B} = \binom{1}{0}.\tag{24}
$$

Thus, the CW mode is forbidden while the CCW mode is allowed along the domain wall. Likewise, as before, this scenario is reversed when the signs of the phases are changed. According to the analysis in case I, the WG modes do not exist in this case under the introduced phases $\varphi = \zeta = 0$ , either.

Case III. For $k = k_{0}$ , equation (19) can be rewritten as

$$
(\xi - \Omega) (A + B) = 0,\tag{25}
$$

where $\zeta=1/3[\cos(\varphi-\zeta)+\cos\varphi+\cos\zeta]$ . Equation (25) indicates that the non-trivial solution only exists when $\xi=\Omega$ , that is, $\varphi=\zeta=0$ , where the CCW and CW modes are coupled and thus the chirality of the WG mode is inhibited. The introduced gain phase $\phi=2\pi$ leads to $\xi\neq\Omega$ , indicating the trivial solution where the breathing modes emerge destructively.

In our study, the effective perimeter of the WG is $L = 1.539 \, \text{m}$ and the velocity $v = 170.89 \, \text{m} \, \text{s}^{-1}$ is calculated from the slope of dispersion relations of the topological edge states shown in Fig. 1d. Substituting these parameters into equation (16) leads to corresponding eigenfrequencies $f_{-} = 8,883 \, \text{Hz}, f_{0} = 8,994 \, \text{Hz}$ and $f_{+} = 9,105 \, \text{Hz}$ , at the order $m = 27$ . Supplementary Video 2 visualizes the dynamic fields of these three modes.

## Influence of the inherent viscosity

Taking the inherent loss of the system into consideration, Extended Data Fig. 6 illustrates the simulated results of the topological WG mode splitting, where the dynamic viscosity of air $\mu = 2.9072 \times 10^{-5}$ Pa s and the thermal conductivity $\chi = 0.0258$ W (m K) $^{-1}$ were introduced in the thermoviscous acoustic module of COMSOL Multiphysics. Extended Data Fig. 6a–c illustrates the corresponding spectrally resolved amplification factors and Extended Data Fig. 6d–f depicts the pressure-field distributions at the respective resonances with different gain-phase textures $\phi = 0$ , $\phi = \pi$ and $\phi = 2\pi$ . Compared with the lossless cases in Fig. 3, we demonstrate that the unavoidable loss in the systems leads to the decrease of the amplification factors accompanied with negligible frequency shifts. However, we emphasize that the topological WG mode splitting is perfectly preserved. The excited in-gap valley edge states breath in the absence of angular momentum (achiral) at frequency $f_0$ with $\phi = 0$ but the propagating CW/CCW topological WG modes can be found at frequency $f_- / f_+$ with $\phi = 2\pi$ , which agrees well with the lossless cases in the main text.

## Physical mechanism of the selective emission

To separate the acoustic signals with opposite propagation chiralities, we replace the left-bottom part of the sample in Fig. 3, which is composed of inverted trimers, by the structures consisting of upright trimers as marked by the red dashed frame in Extended Data Fig. 7a. In this case, a router with two outputs is formed to make the out-coupling of the non-Hermitian WG modes possible. We demonstrate that the valley-selective emission phenomenon roots in valley-projected physics. In Fig. 1d, one can see that the topological edge states with positive/negative group velocity are projected from the K'/K valley. Accordingly, the topological WG mode with CW chirality at $f_{-}$ is projected from the K valley as shown in Extended Data Fig. 7b, whereas Extended Data Fig. 7c illustrates the K' valley-projected topological WG mode with CCW chirality at $f_{+}$ . After introducing the router with the two ports, only the K valley-projected edge states are extracted in Extended Data Fig. 7b. Note that the sound transmission towards output 2, which should be projected by K' valley, is inhibited at $f_{-}$ due to the mismatch of the projected momentum. This results in a highly directional sound beam emanating through output 1. This efficient coupling is made possible by the matching of the parallel component of the specific valley wavevector on to the equifrequency curve in free space $^{34}$ as illustrated in Extended Data Fig. 7b. The simulated directivity of the radiated energy beam shows good agreement with the theoretical prediction as marked by the white arrow. Contrary to this, the K' valley-projected topological WG mode of CCW chirality at $f_{+}$ as shown in Extended Data Fig. 7c now emerges from output 2 by the same momentum conserving principle.

## Thermogram of the topological WG insulator and temperature influence

To further demonstrate the thermal-acoustic effect in the system, we measured the thermogram of the topological WG insulator before and after introducing the electric activation using an infrared thermal imager (FOTRIC 348). Extended Data Fig. 8a shows a photo of the imaging area. As illustrated in Extended Data Fig. 8b, the sample and the background air have identical temperatures when the system is passive (off). However, when the electric signal is turned on, the temperature in the CNT area is much higher than the surrounding passive cylinders as is clearly shown in Extended Data Fig. 8c. Extended Data Fig. 8d illustrates how the background and sample temperature evolves in time during the cyclical excitation of the rods. All measurements were conducted under relatively steady temperatures ranging from 21 °C to 22 °C, as indicated by the purple area in Extended Data Fig. 8d, which we monitored with the help of an infrared thermal imager and a thermocouple thermometer (Fluke 52-II). Based on this control, we aimed to predict the source for the observed frequency shifts at resonance. The relationship between the sound speed and temperature in the air can be expressed as

$$
c = \sqrt {\gamma R T},\tag{26}
$$

where R, $\gamma$ and T represent the molar gas constant, adiabatic exponent and temperature in kelvin, respectively. At finite temperature variations, we assume that the wavelength of the WG modes remain constant

$$
\lambda = c (T) / f (T) = c \left(T _ {0}\right) / f \left(T _ {0}\right).\tag{27}
$$

Here, $f(T_{0})$ and $c(T_{0})$ represent the peak frequency and sound speed at $T_{0}=293.15\;K$ (20 °C in Celsius, the temperature used in the simulations). At marginal temperature variations, that is, $\left|\frac{T-T_{0}}{T_{0}}\right|<0.02$ , we linearize the temperature-induced frequency shift according to

$$
f (T) = f \left(T _ {0}\right) \sqrt {\frac {T}{T _ {0}}} = f \left(T _ {0}\right) \left[ 1 + \frac {\alpha \left(T - T _ {0}\right)}{c \left(T _ {0}\right)} \right],\tag{28}
$$

with $\alpha=0.607$ . As Extended Data Fig. 8e shows, numerically simulated temperature variations induce the theoretically predicted linear frequency shift among the three WG modes of interest.

## Robustness of the system

To verify the robustness of the system, we construct a fractal geometry as shown in Extended Data Fig. 9a. The corresponding energy distribution at the specific frequency supporting a strict CCW spinning of sound is illustrated in Extended Data Fig. 9b. The dynamic pressure-field maps (see Supplementary Video 3 for the time evolution) depict not only the CCW polarization but also the strong sound confinements along the perimeter, which is a convincing indicator of the topological resilience along the curved and sharp path.

Next, the influence of instabilities of the excited sound signals from the CNT films is considered, which also reflect the possible sample fabrication irregularities. The ideal excited sound signals from each gain unit in the primitive cell can be expressed as

$$
S _ {j} = S _ {0} \mathrm{e} ^ {\mathrm{i} (j - 1) \phi / 3}, j = 1, 2, 3\tag{29}
$$

as shown by the solid curves in Extended Data Fig. 9c, e. First, a phase disturbance of $\delta\phi$ , randomly distributed within the range from $-0.2 < \delta\phi < 0.2$ is introduced. We test 20 groups of $\delta\phi$ values and find that the WG modes remain spectrally unaffected. The corresponding amplification spectrum at $\phi = 2\pi$ , for a randomly chosen case, as marked by the dashed curves in Extended Data Fig. 9c, is shown in Extended Data Fig. 9d. Subsequently, an equivalent amplitude disturbance of $\delta S$ is also randomly distributed within the range from $-0.2S_0 < \delta S < 0.2S_0$ ( $S_0$ is the original source amplitude) as depicted in Extended Data Fig. 9e, f. In both cases, the amplification factor, although slightly altered, remains high even in the presence of high disturbances.

To respect the underlying $C_{3v}$ symmetry, we introduced geometrical defects both at the corners and the side of the topological WG insulator. We concentrated at one specific gain texture, that is, $\phi = 2\pi$ , and conducted numerical simulations to compute the WG modes including defects comprising gainless, displaced (moved a distance of 0.08a) and expanded (increased the centre-to-centre distance by 0.05a) rods along the domain wall as shown in Extended Data Fig. 10a. Extended Data Fig. 10b shows the spectrally simulated amplification factors under the discussed six types of perturbation, showing that the frequency and amplitude of both the CW and the CCW modes coincide well with unperturbed results presented in Fig. 3. In particular, the field maps in Extended Data Fig. 10a show that the CCW resonance $f_{+}$ remains entirely unaffected by the introduced defects.

From the above group of defects, we chose to fabricate three sets of units containing displaced cylinder trimers with shifts in the range of $\Delta d = 0.04a - 0.10a$ . Identically defected unit cells were thus introduced at the corners and afterwards at the centre edges of the WG lattice as illustrated in Extended Data Fig. 10c. Thus, the six spectral measurements of the amplification factors as shown in Extended Data Fig. 10d show a remarkable immunity to the added defects by virtue of almost identical data. As a result of even more pronounced perturbations, our system that builds on valley-projected physics cannot longer be protected due to the Anderson localizations $^{39}$ . However, recent studies have reported that increasing the disorder could indeed induce a topological phase transition. Such systems are coined topological Anderson insulators and provide novel transport properties $^{40-42}$ .

## Data availability

The data that support the findings of this study are available from the corresponding authors on reasonable request.

## Code availability

The code used to calculate the results for this work is available from the corresponding authors on reasonable request.

35. Mei, J., Wu, Y., Chan, C. T. & Zhang, Z.-Q. First-principles study of Dirac and Dirac-like cones in phononic and photonic crystals. Phys. Rev. B 86, 035141 (2012).

36. Makwana, M. P. & Craster, R. V. Geometrically navigating topological plate modes around gentle and sharp bends. Phys. Rev. B 98, 184105 (2018).

37. Ochiai, T. Photonic realization of the $(2+1)$ -dimensional parity anomaly. Phys. Rev. B 86, 075152 (2012).

38. Vesterinen, V., Niskanen, A. O., Hassel, J. & Helisto, P. Fundamental efficiency of nanothermophones: modeling and experiments. Nano Lett. 10, 5020–5024 (2010).

39. Anderson, P. W. Absence of diffusion in certain random lattices. Phys. Rev. 109, 1492–1505 (1958).

40. Stützer, S. et al. Photonic topological Anderson insulators. Nature 560, 461–465 (2018).

41. Liu, G.-G. et al. Topological Anderson insulator in disordered photonic crystals. Phys. Rev. Lett. 125, 133603 (2020).

42. Zangeneh-Nejad, F. & Fleury, R. Disorder-induced signal filtering with topological metamaterials. Adv. Mater. 32, 2001034 (2020).

Acknowledgements This work was supported by the National Basic Research Program of China (2017YFA0303702), NSFC (12074183, 11922407, 11904035, 11834008, 11874215 and 12104226) and the Fundamental Research Funds for the Central Universities (020414380181). Z.Z. acknowledges the support from the China National Postdoctoral Program for Innovative Talents (BX20200165) and the China Postdoctoral Science Foundation (2020M681541). L.Z. acknowledges support from the CONEX-Plus programme funded by Universidad Carlos III de Madrid and the European Union's Horizon 2020 research and innovation programme under Marie Sklodowska-Curie grant agreement 801538. J.C. acknowledges support from the European Research Council (ERC) through the Starting Grant 714577 PHONOMETA and from the MINECO through a Ramón y Cajal grant (grant number RYC-2015-17156).

Author contributions Y.C. initiated the project and conceived the idea. Y.C., X.L. and J.C. guided the research. B.H., Z.Z., H.Z., L.Z. and J.C. carried out the theoretical analyses. Z.Z. and L.Z. developed the Hamiltonian model. B.H., Z.Z. and H.Z. conducted finite-element-method simulations, designed the experimental setup and conducted the measurements. W.X., Z.Y., X.W. and J.X. assisted with sample fabrication. Z.Z., Y.C. and J.C. wrote the manuscript. All the authors contributed to the discussions of the results and the manuscript preparation.

![](images/f5ff39acf79589ba35db6e0b389467d840f90a85155664b967faf876d10e1655.jpg)

Extended Data Fig. 1 | Experimental setup for modulating the acoustic gain. The CNT films wrapped around cylinders are connected with the electrical input and thus play the role of acoustic gain thanks to electro-thermoacoustic coupling.

## Article

![](images/3e9253cb95c3638acdb2648d1c04d18b032f93af64ca2150238705af11a00c53.jpg)

![](images/ef3c88b26619c4224f2b9d765f7810e16d7791811d1a569bf24f7a6077d086c1.jpg)

c
![](images/1a1adf9d133406bc9d20014087edf02957b3ed4c33a1cb41d3b459d9d1ce299a.jpg)
f

Extended Data Fig. 2 | Symmetry analysis and the dispersion relations calculated from the k-p method. a, Schematic of the unit cell under $C_{3v}$ symmetry. b, c, Band diagrams of the $C_{3v}$ -symmetric sonic crystal with $\beta = 0$ (b) and with $\beta = 0.05$ (c). Coloured circles and solid curves epitomize the
![](images/d7b5a853ae14afface05f0663dba32250ab1bad331070a3b090e474ef13eb02d.jpg)

![](images/c071533547cc1d80abe557e0aeb7cbe37e272fa8650419efb0d4507faf94c1bf.jpg)

![](images/e7e154af629b0cb1fade931bdf1a019b97aab0d2bc25f93080995381a427b068.jpg)
calculated results from the finite-element method and the k-p method, respectively. d-f, Same as a-c, but for the sonic crystal preserved under $C_{3}$ symmetry with the rotation angle $\theta = -\pi/6$ .

d
a
![](images/7dab906269c8e6d37976610f9509e6e3dff020a34e371de87fcbaf3032864b9a.jpg)
b

![](images/6b7f5cff65e0660d989777cec936ffd9376aa15d91d4160c736ce15b6fc9aa7a.jpg)
Extended Data Fig. 3 | Comparison between amplification through the meta-fluid and thermal-acoustic gain. a, b, Simulated scattering pressure fields under the inward radiation of coaxial cylindrical waves by the meta-fluid model (a) and the thermal-acoustic gain model (b). c, Enhancement of the

![](images/394128d6541561707850c4f63c7bddd155197feeb1bb322c39974df9ce7fd54c.jpg)

![](images/4c362de7337a4a6e85e666aa8c10d3298b8c57b9cdfde5e9b8a3ad9aa89609a2.jpg)
scattering pressure fields calculated by the meta-fluid model (purple line) and the thermal-acoustic gain model (orange dots) with different $\beta$ at f=9.1 kHz. d, Corresponding frequency dependence at $\beta=0.05$ .

![](images/d46143fbe9d17f81e72b5bab3888802a6ed0174bbe3d32aaecf27a84e94f1ab9.jpg)
Extended Data Fig. 4 | Characterizations of the CNT film. a, Experimental setup for measuring the sound pressure and the surface vibration displacement. b, Acoustic pressure amplitude spectra measured near the CNT film (solid curve) and the loudspeaker (dashed curve). c, Vibration displacements spectra measured by laser vibrometry. The solid and dashed curves represent surface
displacement on the CNT film and the traditional paper basin loudspeaker, respectively. d, Photograph of the measurement setup for electrical impedance analysis. Inset: enlarged view of the single sample. e, Experimentally measured amplitude and phase of the impedance curve. f, Experimentally measured directivity pattern.

![](images/a4eb237fecca0f406c810d87e7ea5ad51ec7bcc5f4df7d572723b2580bec508e.jpg)
Extended Data Fig. 5 | Physical model of the topological WG insulator. The proposed topological domain wall can be regarded as a triangular acoustic waveguide, and the phase along each edge is labelled as $\alpha_{j}$ , with j=1, 2 and 3.

![](images/ae8912fb155c8dda9601db39926449d9ac652237f5b75f50b14d31d33113b9c2.jpg)
Extended Data Fig. 6 | The influence of loss on the topological WG insulator. a–c, Spectrally resolved amplification factors through simulations considering the inherent loss with three different gain-phase textures: $\phi = 0$ (a),
$\phi = \pi (\mathbf{b})$ and $\phi = 2\pi (\mathbf{c})$ . d–f, Pressure-field distributions and their chiralities of the three resonances at three different gain-phase textures corresponding to the frequencies $f_{-} = 8,889 \text{ Hz (d)}, f_{0} = 8,999 \text{ Hz (e)}$ and $f_{+} = 9,103 \text{ Hz (f)}$ .

f\_
a
![](images/95b54213f1e71acb0ce65a75b6e3ff131df460f07c83d76444576ce7a76bdd31.jpg)

![](images/4def7d76c48f120a31bc5dc37182a75cf91976736d3a52df8052aed83160af41.jpg)

b
![](images/b9f7ff094eb563167c20d6393a815bdc2268d61fc9be7c8d79c4d85122c449ef.jpg)

![](images/448ce75f2b7936f17d365c604f6abb71b5a5debbf3020523c1a91038ed176dc7.jpg)
Extended Data Fig. 7 | Valley chirality-selective sound emissions of the topological WG insulator. a, Left: illustration of the designed device for the out-coupling of the chiral WG modes. Right: enlarged view of the router. The insets in the right panel show photographs of the cylinder trimers wrapped without or with CNT films. b, c, Momentum space analysis of the out-coupled K
valley-projected topological WG mode of CW chirality at frequency $f_{-}=8,889\ Hz$ (b) and K' valley-WG mode of CCW chirality at frequency $f_{+}=9,106\ Hz$ (c). The white solid hexagon represents the first Brillouin zone and the white dashed circle shows the equi-frequency contour in air. Ambient thermal colour represents the corresponding simulated sound energy fields.

![](images/279d2139092c31f45bd0a35d39fb46d968f69657855db9c39eeb36d0825fdca6.jpg)

![](images/c372260e863e39e285400eee6aa5288fee775f41121248edf71dcda9c8eed2e0.jpg)

![](images/a1f55b4e72a57f2c56fd6280f066b33cb39a88df6be8fb5c4d9dde92480114dc.jpg)
background (orange curve) with time during the measurements. The shaded area corresponds to the temperature range of 21–22 °C in the experiments. e, The frequency shifts of the peaks corresponding to $f_{-}, f_{0}$ and $f_{+}$ under the variation of the temperature. Lines and dots represent the theoretical and simulated results.

Extended Data Fig. 8 | Thermogram of the device and influence of temperature. a, Photograph of the sample. b, c, Corresponding thermogram of the sample without (passive; b) and with (active; c) applied electric control. In a–c, the left column shows the entire sample and the right column shows the enlarged view of the partial sample outlined by the dashed frame. d, Temperature evolutions of air near the CNT film (blue curve) and in the

a
![](images/b3face0ec7e5bb0fc022816ea00ce938036e53cadf45bf9bb60cfd1409c82ec3.jpg)

![](images/8bac86921194e372c4a11087ad2a9773b75275825dc7e07ea9026ba7a775211b.jpg)

![](images/2ca1cae176068de71212b47ae46fc9a4618e726b07540f2bea3e4039a3bf8803.jpg)

b
![](images/57f653a8d213e71c21136f9ddd818805c276d1c694c1ed191b288f157185002a.jpg)

![](images/865cb5c911019e333f33ada7be64f7d2d9dfff9696098ee91cf3c235deb7c345.jpg)

![](images/b80bc37d6d3377a25efae8a8fba7843c0879d46353a80a39da652700c0de12c0.jpg)
Extended Data Fig. 9 | Other types of active topological gallery and robustness against disturbances. a, Schematic of the WG with a snowflake-shaped domain wall. b, Energy distributions of the CCW WG mode with $\phi = 2\pi$ . c, Introducing phase disturbances. The solid curves in orange, light
blue and purple represent the undistorted gain signals, and the dashed curves represent the distorted gain signals. d, Amplification spectrum including phase inhomogeneities with $\phi = 2\pi$ . e, f, Same as c, d, but amplitude disturbances are introduced instead of phase disturbances.

![](images/0e7dd48a9d1b5db3094b86a5ede5f8ec6313c2569cd9aa79f63fd8aefa6e3b5f.jpg)

b
![](images/8f020bea4cd20a8bed3d46d6252ffd1cbc81435fa94e18582995d7a4216e472c.jpg)

c
![](images/4cbd59cd0aca2b45830594a302d58b45bc7b645adbb6565d3107e35e24f50137.jpg)

![](images/aca6e9638bb6748c55f5674dfa44a27bda0ba4ea549be9b8384fe4eff65eef83.jpg)

![](images/58665351f6c00a804cc6b307fbe9aa7eda76f470daa552734e294562dc4d231b.jpg)
Extended Data Fig. 10 | Robustness of the topological WG insulator against the geometric defects. a, b, Numerical defect analysis comprising one defective unit cell at each corner or side of the structure. At a gain-phase texture of $\phi = 2\pi$ , we simulate the pressure fields of the system including defective units, that is, gainless, displaced or expanded cylinders (a) together with their corresponding spectral amplification factors (b). c, Schematic of the
sample where the red and blue highlighted units label the perturbed rods located at the sides and corners, respectively. d, In the experiments, we chose three sets of perturbation displacements $\Delta d = 0.04a - 0.10a$ with $\phi = 2\pi$ , whose measured amplification factors include both corner (top) and side (bottom) defects.
