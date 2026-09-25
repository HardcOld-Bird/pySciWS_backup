# Causal-Constraint Broadband Sound Absorption under Isothermal Process

Chuanhao Ge, $^{*}$ Nengyin Wang, $^{*}$ Xu Wang $^{ID}$ , $^{\dagger}$ and Yong Li $^{ID}$

Institute of Acoustics, Tongji University, Shanghai 200092, China

(Received 17 December 2024; revised 31 March 2025; accepted 16 May 2025; published 10 June 2025)

Causality, a cornerstone of physical laws, fundamentally links a system's structural characteristics to its wave interaction properties, such as the minimum thickness of acoustic absorbers required for specific absorption spectra. Traditional causality principles for sound absorption are derived under the assumption of adiabatic sound propagation. In this Letter, we propose a generalized causal framework that incorporates isothermal processes, accounting for nonslip boundary conditions at the fluid-solid interface. These conditions introduce velocity and temperature gradients, challenging the conventional adiabatic assumption. To validate our framework, we analyze two distinct absorber types: a metamaterial with multiresonant units and a metafoam with multilayer double-porosity structures. Our theoretical and experimental studies reveal that absorber thickness can exceed the adiabatic limit, being instead governed by isothermal constraints. This paradigm shift deepens the understanding of sound absorption mechanisms and paves the way for designing high-performance acoustic devices that approach fundamental performance limits.

DOI: 10.1103/jwvm-ntts

Accurately determining the equation of state (EOS) of a fluid is essential for understanding its dynamic behavior. Historically, different EOS assumptions—either isothermal or adiabatic—have resulted in contrasting predictions for sound propagation in air. Newton's isothermal model predicted a sound speed at 298 m/s [1], whereas Laplace's adiabatic model, which incorporates thermodynamic effects, corrected this estimate to 348 m/s [2]. This discrepancy highlights the critical impact of a medium's thermodynamic characteristics on its response to acoustic waves.

Compared to air, sound propagation in materials is significantly more complex due to their intricate internal structures and the resulting air-solid interfaces. This complexity shapes the acoustic responses of materials, influencing their sound absorption and scattering characteristics. Underpinning this behavior is the principle of causality, a fundamental concept that links specific absorption spectra to the minimum achievable material thickness $[3,4]$ . Therefore, accurately defining the EOS is essential for theoretically establishing the physical limits of acoustic absorbers' performance.

Metamaterials and metasurfaces $[5–15]$ exploit local resonances to strengthen wave-matter interactions, enhancing sound energy density within the structures, thereby significantly improving sound absorption $[3,16–22]$ . The adiabatic assumption is often adopted as the theoretical foundation for optimizing acoustic performance by balancing absorption bandwidth and material thickness $[22–30]$ . However, sound propagation inherently requires a medium, and the continuity equation for this medium enforces a nonslip boundary condition at fluid-solid interfaces. This boundary condition leads to the formation of viscous and thermal boundary layers, characterized by velocity and temperature gradients $[31,32]$ . Consequently, heat exchange and viscous effects become increasingly significant, particularly in the low-frequency regime approaching the static limit, where the thickness of the boundary layers becomes comparable to or exceeds the characteristic dimensions of the material structure. Under these conditions, sound propagation transitions toward an isothermal process, challenging the conventional adiabatic assumption and highlighting the necessity for a more comprehensive theoretical framework.

In this Letter, through theoretical analysis, numerical simulations, and experimental validation, we revisit the role of the EOS in describing sound propagation within absorbing materials and propose a generalized causal framework based on isothermal processes, which reveals that the minimum thickness of acoustic absorbers can surpass the thickness limits predicted by adiabatic models. To prove the universality of our isothermal framework on governing the behavior of sound absorption, various types of absorbers are investigated. Among them, two distinct absorbers, a metamaterial (a structure with multiresonant units) and a metafoam (a material featuring double porosity), provide exceptional low-frequency and ultrabroadband sound absorption, with thicknesses closely matching the theoretical limit predicted by our isothermal framework. These findings establish a new theoretical foundation for the design of advanced sound-absorbing devices surpassing existing performance limits.

(a)
![](images/3954c0422323b4a5bec3b775fb50c944d23eef857ef3507d4ecabbc7549c2815.jpg)

(b)
![](images/65c821ec35bc16fb07f8265d40f5a1040d972a63a60c53f448f8c102cf68d3f5.jpg)

(c)
![](images/f0cdccfbe1c8b53d9bf94ff01d6dd4a0851a2f1e1fd5be1c9a341c6fa7dcfc99.jpg)
FIG. 1. (a) The acoustic boundary layer at the fluid-solid interface of a cylindrical cavity used as an example. Sound propagates forth and back in the cavity along the z direction. $T_{0}$ and $v_{0}$ denote the ambient temperature and free-stream particle velocity, respectively. The black dashed line indicates the central axis of the cylindrical cavity, while the blue and gray solid lines represent the temperature and particle velocity gradients along the r direction, respectively. (b) Frequency-dependent transition of the effective bulk modulus of air within the cavity, from the isothermal modulus $B_{0}$ at low frequencies to the adiabatic modulus $B_{T}$ at high frequencies. (c) Complex wavelength plane illustration of the frequency response of acoustic materials, where the causal-constrained integration contour comprises a semicircle of radius $|\lambda|$ and the real axis, which extends to $|\lambda| \to \infty$ in the static limit.

When sound propagates near a structural boundary [Fig. 1(a)], an acoustic boundary layer forms due to the nonslip boundary condition at the fluid-solid interface [31-33]. This boundary layer comprises two key components: a viscous boundary layer and a thermal boundary layer. The viscous boundary layer, characterized by a thickness $\delta_{\mathrm{v}} = \sqrt{\mu / \pi\rho_0f}$ , arises from shear forces within the fluid, where $\mu$ is the dynamic viscosity, $\rho_0$ is the fluid density, and $f$ is the sound frequency. The thermal boundary layer, with a thickness $\delta_{\mathrm{th}} = \sqrt{\kappa / \pi\rho_0fC_p}$ , results from wall-fluid heat exchange, where $\kappa$ is the thermal conductivity and $C_{\mathrm{p}}$ is the heat capacity at constant pressure. Together, these layers define the acoustic boundary layer, which plays a critical role in shaping sound propagation near fluid-solid interfaces.

The acoustic boundary layer exerts a frequency-dependent influence on the effective bulk modulus of air in the vicinity of structural boundary [Fig. 1(b)]. At low frequencies where the boundary layer thickness is comparable to or exceeds the material dimensions, the bulk modulus approaches the isothermal limit, $B_{T} = P$ , where P is the ambient pressure. As the frequency increases, the boundary layer becomes negligible relative to the structural scale, leading to a transition toward the adiabatic regime. In the high-frequency limit, the bulk modulus approaches the adiabatic value, $B_{0} = \gamma P$ , where $\gamma$ is the specific heat ratio. This transition from isothermal to adiabatic behavior underscores the critical role of acoustic boundary layer in shaping the thermodynamic response of acoustic materials.

Fundamentally, for linear time-invariant systems, the principle of causality establishes a relationship between the absorption spectrum and the minimum achievable material thickness, which in acoustics can be is expressed as

$$
L _ {\min} = \frac {1}{4 \pi^ {2}} \frac {B _ {\text { eff }}}{B _ {0}} \left| \int_ {0} ^ {\infty} \ln [ 1 - A (\lambda) ] \mathrm{d} \lambda \right|,\tag{1}
$$

where $\lambda$ is the sound wavelength in air, $A(\lambda)$ is the absorption coefficient, and $B_{eff}$ represents the effective bulk modulus of the sound-absorbing material. Here, $B_{\mathrm{eff}}^{-1} = \varphi B_{\mathrm{air}}^{-1} + (1 - \varphi) B_{\mathrm{solid}}^{-1}$ , which is collectively determined by the bulk modulus of air in the material ( $B_{air}$ ), the bulk modulus of the solid structural matrix of the material ( $B_{solid}$ ), and the volume ratio of air to material ( $\varphi$ ) (see details in Supplemental Material, Sec. I [34]). Note that the integration of Eq. (1) covers the whole wavelength range, extending to the long-wave limit ( $\lambda \to \infty$ ). Accordingly, $B_{air}$ is evaluated in the static limit where the isothermal process dominates, i.e., $B_{air} = B_{T}$ .

As evidenced by Eq. (1), a smaller $B_{eff}$ allows a lower theoretical minimum thickness, enabling more efficient sound absorption with less material. This motivates the design strategy for achieving $\varphi = 1$ (so that $B_{eff} \rightarrow B_{T}$ ). Incorporating the effects of the boundary layer into the material's response under the causality constraint reveals the isothermal nature of the acoustic process within the material [Fig. 1(c)]. By accounting for these effects, we derive a generalized causal relationship for airborne sound absorption devices (see Supplemental Material, Sec. I [34] for a detailed derivation):

$$
L _ {\lim} = \frac {1}{4 \pi^ {2}} \frac {1}{\gamma} \left| \int_ {0} ^ {\infty} \ln [ 1 - A (\lambda) ] d \lambda \right|.\tag{2}
$$

This generalized relationship demonstrates that the limiting thickness can exceed the value predicted under the adiabatic assumption by a factor of $1/\gamma$ , thereby expanding the theoretical boundaries of acoustic absorber design.

Upon the built isothermal framework, two factors determine the thickness of an absorber: the volume ratio $\varphi$ and the entire absorption spectrum $A(\lambda)$ (not just the spectrum in the range of interest). Under this guidance, we design an broadband acoustic metamaterial [Fig. 2(a)] (see design details in Supplemental Material, Sec. II [34]). The metamaterial consists of four cascaded neck-embedded Helmholtz resonators and achieves an average absorption coefficient of 0.94 over a broad frequency range from 250 to $2000\mathrm{Hz}$ [Fig. 2(b)]. The structural thickness of the metamaterial, $150.2\mathrm{mm}$ , closely matches the isothermal limit of $149.3\mathrm{mm}$ predicted by Eq. (2), while being significantly smaller than the adiabatic minimum of $209.1\mathrm{mm}$ [Fig. 2(a)]. The thickness agreement between the structure and isothermal limit confirms isothermal sound propagation under static conditions, validating our causal framework. Note that although the isothermal process at the long-wave limit exerts negligible influence on the metamaterial's behavior at its operating frequency, it cannot be ignored as it determines the metamaterial limit thickness according to Eq. (1). To approach the ideal $B_{\mathrm{eff}}$ , the wall thickness of the metamaterial used here was reduced to $0.1\mathrm{mm}$ to maximize $\varphi$ (see Supplemental Material, Sec. II.A [34]). This ultrathin design made experimental fabrication impractical. To address this, we supplemented a series of experimental studies with fabricable absorber samples featuring varying $\varphi$ (see Supplemental Material, Sec. III [34]). Experimental results from these samples further corroborate the validity of the isothermal framework.

(a)
![](images/6db10b673dcbd1331638aeba2d36bf1c9732088dafefe65c4d225009185d427c.jpg)

(b)
![](images/f24f5fb0c7d3754faf6375af5c330e09de6691da68c527900df9cedf599e3163.jpg)
FIG. 2. (a) Schematic diagram of the designed metamaterial consisting of four cascaded neck-embedded Helmholtz resonators, and the comparison of the corresponding minimum thicknesses predicted by the causality principles under adiabatic and isothermal assumptions. (b) Absorption spectrum of the metamaterial (red line) and the ratio of the viscous boundary layer thickness to the structural size (blue line). Here, w = 2.5 mm denotes half of the mean side length of the resonators' cross section.

Metamaterials require a certain structural thickness to maintain stability, which prevents their practical realization approaching the predicted theoretical limit. In contrast, conventional porous materials intrinsically have high porosity and high $\varphi$ [35,36]. However, absorber designs based on porous materials sometimes exhibit “pseudo-beyond-causal” behavior when analyzed using traditional causality model under adiabatic assumption. To validate the universality of the proposed isothermal framework, this class of absorbers is also investigated. Here, we employ the Johnson-Champoux-Allard model $[37,38]$ , which assumes a rigid porous framework saturated with a viscous fluid (see details of the model in Supplemental Material, Sec. IV. A $[34]$ ). Given that these micropores are primarily effective at high frequencies, drawing on the idea of metamaterials, we introduce artificially engineered structures inside these porous materials. The material thus formed is a hierarchical structure with scale-separated micropores and macroscopic holes $[Fig. 3(b)]$ . This design leverages the complementary effects of macroscale and microscale porosity to enhance acoustic performance. Calculations of the surface impedance reveal significant modulation in both acoustic resistance and reactance for this double-porosity material compared to conventional foams (i.e., single-porosity materials) $[Fig. 3(c)]$ , thereby facilitating its sound absorption owing to the well impedance matching with the surrounding air (see Supplemental Material, Sec. IV. C $[34]$ )

(a)
![](images/735e232ce0397b4ffaae231d3e8dda5208db738b8603a130c930746cca350fa3.jpg)
(b)

![](images/a153dbb6b55b9b8a4d5b7c3380a92f73267183d6bb39ea0cad5f56a3672a0995.jpg)

![](images/50070f9ece1abe528bc89759ea162ebcf68051883d0e6f471539a43ba0446a48.jpg)

(c)
![](images/2d94d33b57ec23bc770ae81cbe16a18216ab2b1b9b560da60d2bc27441228fec.jpg)
(d)

![](images/0f100a310299fb0910e886ebdc83d418cf72caa85b63f2a9d32ad0e99fdc6cbd.jpg)

![](images/f4b40fe2da04ecec345a47e22d27b099731f21158ca6672d7a8b9e726fbee912.jpg)

(e)
![](images/74f0096f78aa57eb6c41bd8d02c3bc3fbe1c2814e09743deaf113b1d69176c29.jpg)
FIG. 3. (a) The melamine foam used as an example of porous materials. The micrograph shows its internal intricacies with rich fluid-solid boundaries. (b) A schematic of the double-porosity material based on melamine foams, where r is the radius of macroscopic hole. (c) Acoustic resistance and reactance of the double-porosity material with varying r (dashed lines) compared to the melamine foam (solid lines). Real (d) and imaginary (e) parts of $B_{eff}/B_{T}$ . The green and yellow dashed lines correspond to the cases with only macroscopic hole and melamine foam, respectively. The red solid lines correspond to the double-porosity material. $f_{tp}$ and $f_{tm}$ are thermal characteristic frequencies of the macroscopic hole and micropores, respectively [39].

The double-porosity material features an intentionally tuned effective bulk modulus that varies with frequency (see detailed calculations in Supplemental Material, Sec. IV. B [34]). The scale separation between macroscopic holes and micropores leads to a “double-characteristic-frequency behavior” [Figs. 3(d) and 3(e)] [39]. At ultralow frequencies ( $f \ll f_{\text{tp}}$ , $f_{\text{tp}}$ less than 0.1 Hz), the real part of the bulk modulus remains low, reflecting high compressibility and the dominance of isothermal processes in both micropores and macroscopic holes [Fig. 3(d)]. As frequency increases ( $f_{\text{tp}} < f < f_{\text{tm}}$ ), the real part of the bulk modulus increases, which represents the incompressibility of the air in macroscopic holes and thus manifests a gradual transition toward an adiabatic process. At higher frequencies ( $f \gg f_{\text{tm}}$ ), the entire system transitions to an adiabatic regime as the material’s effective bulk modulus approaches its maximum.

This frequency-dependent transition from isothermal to adiabatic behavior is driven by the progressive “stiffening” of the material with frequency. Simultaneously, the imaginary part of the bulk modulus, which governs sound energy dissipation, exhibits two distinct local maxima corresponding to the thermal characteristic frequencies of the macroscopic holes ( $f_{tp}$ ) and the micropores ( $f_{tm}$ ) [Fig. 3(e)]. Note that this dual-response mechanism enables the material to emulate airlike behavior at low frequencies and the typical characteristics of porous materials at high frequencies, thereby achieving efficient broadband sound absorption across a wide frequency range.

Based on the double-porosity materials, we construct a metafoam consisting of a stack of five double-porosity layers, each layer with a thickness of 100 mm and featuring different hole sizes [Fig. 4(a)] (see geometries of the sample and experimental setups in Supplemental Material, Sec. V. A [34]). As illustrated in Fig. 4(b), the metafoam exhibits superior low-frequency and broadband sound absorption compared to a conventional foam of the same thickness, maintaining an absorption coefficient above 0.9 starting from 56 Hz.

To gain deeper insights into the underlying mechanism of the designed metafoam, we define a dimensionless quantity $\varsigma(\lambda)=1/(4\pi^{2}\gamma)|\ln[1-A(\lambda)]|$ , which, as a function of wavelength, depends solely on the sound absorption coefficient A (where higher A corresponds to larger $\varsigma$ values). Then Eq. (2) can be simplified as $L_{\lim}=\int_{0}^{\infty}\varsigma(\lambda)\mathrm{d}\lambda$ , showing that the limiting thickness $L_{\lim}$ corresponds to the area enclosed under the $\varsigma(\lambda)$ curve. Figure 4(c) plots $\varsigma(\lambda)$ for both the metafoam and the conventional foam. At very low frequencies, boundary-layer effects dominate the causal-constraint thickness due to the long wavelengths, though their contribution to sound absorption remains modest. The introduction of macroscopic holes mitigates the “undesired” ultra-low-frequency modest absorption by its airlike behavior [Fig. 3(d)]. Moreover, these designed macroscopic holes induce resonancelike states, which significantly enhances the absorption performance by shifting the excessive response $[22]$ and optimizing thickness utilization (see Supplemental Material, Fig. S8 $[34]$ ), thereby broadening the bandwidth of strong absorption. The response characteristics exhibited in the complex frequency plane also witness the very difference of these two materials [Figs. 4(d) and 4(e)]. Compared to conventional foams, the metafoam exhibits paired poles and zeros in the complex frequency plane, resulting from the resonancelike states induced by macroscopic holes. Crucially, these zeros and poles are all locate in the lower half of the complex frequency plane, ensuring that the designed metafoam reaches the limiting thickness dictated by the causality constraint (the theoretical and experimental criteria for determining whether the absorber reaches the minimum thickness are detailed in Supplemental Material, Secs. VI. A and VI. B $[34]$ ). Moreover, the thickness of the metafoam, 500 mm, surpasses the adiabatic minimum thickness of 682.5 mm and closely aligns with the causal limit of 487.5 mm, incorporating isothermal processes [Fig. 4(a)] (see explanations for slight discrepancies in thickness in Supplemental Material, Sec. V. D [34]).

(a)
(b)
![](images/53d01cddc9e45786ffdc63cefdf087317e49ed68fe3e95da4d5b7086f8160d9e.jpg)

![](images/eb67a46da12860402311358d1832701bfc25962ee667ebf5b0ac65920fa21c98.jpg)

![](images/d0e92df230efb3428fcca7f854b810acfccacf8bbaa17aa33b5711d5c64cadb3.jpg)
FIG. 4. (a) Schematic of the designed metafoam. The structural thickness is 500 mm, surpassing the limit thickness of 682.5 mm predicted by traditional causality principles based on adiabatic assumptions. (b) Absorption spectra of the metafoam compared to a same-height foam. (c) Plots of the dimensionless quantity $\varsigma(\lambda)$ illustrating the wavelength-dependent response of the foam and the metafoam. Reflection coefficients in the complex frequency plane of the foam (d) and the metafoam (e). $f_{i}$ is the imaginary frequency, and the green contour line represents absorption coefficient equal to 0.9. Here, $r'$ is reflection coefficient.

In conclusion, we have developed a generalized causal framework that incorporates isothermal processes, thereby extending the conventional understanding of causality in wave-matter interactions. This framework has been rigorously validated through theoretical analysis and experimental demonstrations using meticulously designed metamaterials and metafoam absorbers, both achieving near-perfect low-frequency broadband sound absorption. Our findings advance the theoretical foundation of sound absorption mechanisms and offer practical strategies for designing lightweight, high-efficiency acoustic absorbers with broad applications.

Acknowledgments—This Letter was supported by the Shanghai Pilot Program for Basic Research, the Xiaomi Young Talents Program, the Fundamental Research Funds for the Central Universities, and the Shanghai 3-year Action Plan (No. GWVI-11.1-37).

Data availability—The data that support the findings of this Letter are not publicly available. The data are available from the authors upon reasonable request.

[1] I. Newton, Philosophiae Naturalis Principia Mathematica (G. Brookman, London, 1833).

[2] P. S. Laplace, Théorie Analytique des Probabilités (Courcier, Paris, 1820).

[3] M. Yang, S. Chen, C. Fu, and P. Sheng, Optimal sound-absorbing structures, Mater. Horiz. 4, 673 (2017).

[4] K. N. Rozanov, Ultimate thickness to bandwidth ratio of radar absorbers, IEEE Trans. Antennas Propag. 48, 1230 (2000).

[5] Z. Liu, X. Zhang, Y. Mao, Y. Y. Zhu, Z. Yang, C. T. Chan, and P. Sheng, Locally resonant sonic materials, Science 289, 1734 (2000).

[6] N. Fang, D. Xi, J. Xu, M. Ambati, W. Srituravanich, C. Sun, and X. Zhang, Ultrasonic metamaterials with negative modulus, Nat. Mater. 5, 452 (2006).

[7] Y. Li, B. Liang, Z. M. Gu, X. Y. Zou, and J. C. Cheng, Reflected wavefront manipulation based on ultrathin planar acoustic metasurfaces, Sci. Rep. 3, 2546 (2013).

[8] G. Ma, M. Yang, S. Xiao, Z. Yang, and P. Sheng, Acoustic metasurface with hybrid resonances, Nat. Mater. 13, 873 (2014).

[9] B. Assouar, B. Liang, Y. Wu, Y. Li, J.-C. Cheng, and Y. Jing, Acoustic metasurfaces, Nat. Rev. Mater. 3, 460 (2018).

[10] H. Ge, M. Yang, C. Ma, M.-H. Lu, Y.-F. Chen, N. Fang, and P. Sheng, Breaking the barriers: Advances in acoustic functional materials, Natl. Sci. Rev. 5, 159 (2018).

[11] C. Shao, Y. Zhu, H. Long, C. Liu, Y. Cheng, and X. Liu, Metasurface absorber for ultra-broadband sound via overdamped modes coupling, Appl. Phys. Lett. 120, 083504 (2022).

[12] Q. Wang, P. del Hougne, and G. Ma, Controlling the spatiotemporal response of transient reverberating sound, Phys. Rev. Appl. 17, 044007 (2022).

[13] S. Sergeev, R. Fleury, and H. Lissek, Ultrabroadband sound control with deep-subwavelength plasmacoustic metalayers, Nat. Commun. 14, 2874 (2023).

[14] L. Huang, S. Huang, C. Shen, S. Yves, A. S. Pilipchuk, X. Ni, S. Kim, Y. K. Chiang, D. A. Powell, J. Zhu, Y. Cheng, Y. Li, A. F. Sadreev, A. Alù, and A. E. Miroshnichenko, Acoustic resonances in non-Hermitian open systems, Nat. Rev. Phys. 6, 11 (2024).

[15] Z. Su, Q. Wang, Z.-G. Chen, and M.-H. Lu, Hybrid porous Helmholtz resonator for low-frequency broadband absorption, Phys. Rev. Appl. 22, 044032 (2024).

[16] S. Huang, Z. Zhou, D. Li, T. Liu, X. Wang, J. Zhu, and Y. Li, Compact broadband acoustic sink with coherently coupled weak resonances, Sci. Bull. 65, 373 (2020).

[17] S. Qu, N. Gao, A. Tinel, B. Morvan, V. Romero-Garcia, J. P. Groby, and P. Sheng, Underwater metamaterial absorber with impedance-matched composite, Sci. Adv. 8, eabm4206 (2022).

[18] N. Wang, C. Zhou, S. Qiu, S. Huang, B. Jia, S. Liu, J. Cao, Z. Zhou, H. Ding, J. Zhu, and Y. Li, Meta-silencer with designable timbre, Int. J. Extrem. Manuf. 5, 025501 (2023).

[19] X. Wang, R. Dong, Y. Li, and Y. Jing, Non-local and non-Hermitian acoustic metasurfaces, Rep. Prog. Phys. 86, 116501 (2023).

[20] S. Huang, Y. Li, J. Zhu, and D. P. Tsai, Sound-absorbing materials, Phys. Rev. Appl. 20, 010501 (2023).

[21] S. Qu, M. Yang, T. Wu, Y. Xu, N. Fang, and S. Chen, Analytical modeling of acoustic exponential materials and physical mechanism of broadband anti-reflection, Mater. Today Phys. 44, 101421 (2024).

[22] Z. Zhou, S. Huang, D. Li, J. Zhu, and Y. Li, Broadband impedance modulation via non-local acoustic metamaterials, Natl. Sci. Rev. 9, nwab171 (2022).

[23] Y. Li and B.M. Assouar, Acoustic metasurface-based perfect absorber with deep subwavelength thickness, Appl. Phys. Lett. 108, 063502 (2016).

[24] C. Zhang and X. H. Hu, Three-dimensional single-port labyrinthine acoustic metamaterial: Perfect absorption with large bandwidth and tunability, Phys. Rev. Appl. 6, 064025 (2016).

[25] Y. Zhu, A. Merkel, K. Donda, S. Fan, L. Cao, and B. Assouar, Nonlocal acoustic metasurface for ultrabroadband sound absorption, Phys. Rev. B 103, 064102 (2021).

[26] M. Yang and P. Sheng, Sound absorption structures: From porous media to acoustic metamaterials, Annu. Rev. Mater. Res. 47, 83 (2017).

[27] H. Y. Mak, X. Zhang, Z. Dong, S. Miura, T. Iwata, and P. Sheng, Going beyond the causal limit in acoustic absorption, Phys. Rev. Appl. 16, 044062 (2021).

[28] S. Yu, J. Ni, Z. Zhou, S. Xu, D. Li, Y. Li, and J. Qiu, Perfect broadband sound absorption on a graphene-decorated porous system with dual-3d structures, ACS Appl. Mater. Interfaces 14, 28145 (2022).

[29] A. Maddi, C. Olivier, G. Poignand, G. Penelet, V. Pagneux, and Y. Auregan, Frozen sound: An ultra-low frequency

and ultra-broadband non-reciprocal acoustic absorber, Nat. Commun. 14, 4028 (2023).

[30] S. Yu, W. Guo, Z. Zhou, Y. Li, and J. Qiu, Rough-endoplasmic-reticulum-like hierarchical composite structures for efficient mechanical-electromagnetic wave-energy attenuation, Adv. Funct. Mater. 34, 2312835 (2024).

[31] G. Kirchhoff, Ueber den Einfluss der Wärmeleitung in einem Gase auf die Schallbewegung, Ann. Phys. (N.Y.) 210, 177 (1868).

[32] J. W. S. B. Rayleigh, The Theory of Sound (MacMillan, London, 1896), Vol. 2.

[33] G. P. Ward, R. K. Lovelock, A. R. J. Murray, A. P. Hibbins, J. R. Sambles, and J. D. Smith, Boundary-layer effects on acoustic transmission through narrow slit cavities, Phys. Rev. Lett. 115, 044302 (2015).

[34] See Supplemental Material at http://link.aps.org/supplemental/10.1103/jwvm-ntts for demonstrations of

the isothermal process, details of the metamaterial and metafoam, verification of generalizability, and analysis of the minimum causal thickness.

[35] J. Allard and N. Atalla, Propagation of Sound in Porous Media: Modelling Sound Absorbing Materials (John Wiley Sons, New York, 2009).

[36] L. Cao, Q. Fu, Y. Si, B. Ding, and J. Yu, Porous materials for sound absorption, Compos. Commun. 10, 25 (2018).

[37] D. L. Johnson, J. Koplik, and R. Dashen, Theory of dynamic permeability and tortuosity in fluid-saturated porous media, J. Fluid Mech. 176, 379 (1987).

[38] Y. Champoux and J.-F. Allard, Dynamic tortuosity and bulk modulus in air-saturated porous media, J. Appl. Phys. 70, 1975 (1991).

[39] X. Olny and C. Boutin, Acoustic wave propagation in double porosity media, J. Acoust. Soc. Am. 114, 73 (2003).
