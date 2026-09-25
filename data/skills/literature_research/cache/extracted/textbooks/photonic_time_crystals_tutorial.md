This is an electronic reprint of the original article.

This reprint may differ from the original in pagination and typographic detail.

Asgari, Mohammad M.; Garg, Puneet; Wang, Xuchen; Mirmoosa, Mohammad S.; Rockstuhl, Carsten; Asadchy, Viktar

Theory and applications of photonic time crystals : a tutorial

Published in:
Advances in Optics and Photonics

DOI: 10.1364/AOP.525163

Published: 31/12/2024

Document Version
Publisher's PDF, also known as Version of record

Please cite the original version:
Asgari, M. M., Garg, P., Wang, X., Mirmoosa, M. S., Rockstuhl, C., & Asadchy, V. (2024). Theory and applications of photonic time crystals : a tutorial. Advances in Optics and Photonics, 16(4), 958-1063. https://doi.org/10.1364/AOP.525163

Advances in Optics and Photonics

# Theory and applications of photonic time crystals: a tutorial

MOHAMMAD M. ASGARI, $^{1,\dagger}$ PUNEET GARG, $^{2,\dagger}$ iD XUCHEN WANG, $^{3,4,\dagger}$ MOHAMMAD S. MIRMOOSA, $^{5,\dagger}$ iD CARSTEN ROCKSTUHL, $^{2,4,6}$ AND VIKTAR ASADCHY $^{1,*}$ iD

$^{1}$ Department of Electronics and Nanoengineering, Aalto University, Maarintie 8, 02150 Espoo, Finland $^{2}$ Institute of Theoretical Solid State Physics, Karlsruhe Institute of Technology, Kaiserstr. 12, 76131 Karlsruhe, Germany

$^{3}$ Qingdao Innovation and Development Base, Harbin Engineering University, Qingdao 266400, China $^{4}$ Institute of Nanotechnology, Karlsruhe Institute of Technology, Kaiserstr. 12, 76131 Karlsruhe, Germany $^{5}$ Department of Physics and Mathematics, University of Eastern Finland, Yliopistokatu 7, 80130 Joensuu, Finland

$^{6}$ carsten.rockstuhl@kit.edu $^{\dagger}$ These authors contributed equally to this work. $^{*}$ viktar.asadchy@aalto.fi

Received April 1, 2024; revised September 7, 2024; accepted September 8, 2024; published 22 November 2024

This tutorial offers a comprehensive overview of photonic time crystals: artificial materials whose electromagnetic properties are periodically modulated in time at scales comparable to the oscillation period of light while remaining spatially uniform. Being the temporal analogs to traditional photonic crystals, photonic time crystals differ in that they exhibit momentum bandgaps instead of energy bandgaps. The energy is not conserved within momentum bandgaps, and eigenmodes with exponentially growing amplitudes exist in the momentum bandgap. Such properties make photonic time crystals a fascinating novel class of artificial materials from a basic science and applied perspective. This tutorial gives an overview of the fundamental electromagnetic equations governing photonic time crystals and explores the ground-breaking physical phenomena they support. Based on these properties, we also oversee the diverse range of applications they unlock. Different material platforms suitable for creating photonic time crystals are discussed and compared. Furthermore, we elaborate on the connections between wave amplification in photonic time crystals and parametric amplification mechanisms in electrical circuits and nonlinear optics. Numerical codes for calculating the band structures of photonic time crystals using two approaches, the plane wave expansion method and the transfer matrix method, are provided. This tutorial will be helpful for readers with physics or engineering backgrounds. It is designed to serve as an introductory guide for beginners and to establish a reference baseline reflecting the current understanding for researchers in the field. © 2024 Optica Publishing Group

1. Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 960
2. Eigenmodes in PTCs . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 964
2.1. Governing Equations . . . . . . . . . . . . . . . . . . . . . 965
2.1a. Description of a Linear, Isotropic, Spatially Homogeneous, Time-Invariant, and Dispersive Medium 965
2.1b. Extension to a Time-Varying Medium 966
2.1c. Assumption of an Instantaneous Response 968
2.1d. Band Structure Analysis Based on the Plane Wave Expansion 968
2.1e. Band Structure Analysis Based on the Transfer Matrix Method 973
2.2. Electromagnetic Effects Inside the Momentum Bandgap 979
3. Aspects of Realistic PTCs 981
3.1. Effects of Temporal Dispersion 982
3.1a. Helmholtz Equation in a PTC Made From Dispersive Media 983
3.1b. Time-Varying Plasma Frequency 985
3.1c. Time-Varying Resonance Frequency 986
3.1d. Estimating the Size of Momentum Bandgaps 989
3.2. Spatially Finite PTCs 992
3.3. Temporally Finite PTCs 995
3.4. Effects of Anisotropy in PTCs 997
3.5. Defects in PTCs 999
3.6. Effects of Disorder 1000
3.7. Nonlinear PTCs 1002
4.Relations to Other Parametric Systems: Similarity and Distinction 1004
4.1. Parametric Amplification in Electrical Circuits 1004
4.2. Parametric Amplification in Nonlinear Optics 1007
4.3. Parametric Amplification in PTCs 1009
4.4. Comparison of Time Crystals and PTCs 1013
5.Material Platforms to Realize PTCs 1016
5.1. Transmission Lines 1016
5.2. Optical Materials 1018
5.3. 2D Platforms 1020
5.4.Mimicking PTCs With Other Material Platforms 1022
6.Potential Applications of PTCs 1026
6.1.Realizing Thresholdless Lasers Using PTCs 1026
6.2.Enhancing the Emission Rate of Radiation by Free Electrons 1029
6.3.Controlling the Spectral Flow of Light 1031
6.4.Advanced Optical Absorbers Beyond the Rozanov Bound 1033
6.5.Enhanced Resolution Imaging 1034
7.Introducing Spatial Periodicity in PTCs: Spatiotemporal Photonic Crystals 1035
7.1.Traveling-Wave Modulation 1035
7.2.Arbitrary Spatiotemporal Modulation 1038
7.3.Enhancing the Size of a Momentum Bandgap Using Resonant ST-PCs 1040
7.4.Topological Aspects of PTCs 1042
7.5.Nonlinear ST-PCs 1045
8.Future Outlook 1045
9.Concluding Remarks 1048
Funding 1050
Acknowledgments 1050
Disclosures 1050
Data Availability 1050
References 1050

# Theory and applications of photonic time crystals: a tutorial

MOHAMMAD M. ASGARI, PUNEET GARG, XUCHEN WANG, MOHAMMAD S. MIRMOOSA, CARSTEN ROCKSTUHL, AND VIKTAR ASADCHY

## 1. INTRODUCTION

Materials constitute the cornerstone for most technological and many societal developments. It is by no means a surprise that periods of humanity are named according to the materials that shaped them. While we had to consider those materials given to us by nature for a long time, we got increasingly used to the fact that tailor-made materials with properties on demand come in reach by combining intrinsic materials with a suitable geometry or structure. In the context of the tutorial at hand, the material properties at stake are those that govern the interaction of electromagnetic fields or light with matter, and we are interested in electromagnetic or optical materials. In particular, the ever-evolving demands of contemporary society call for artificial materials that can control light propagation in a way inaccessible to natural materials.

Over the past several decades, a diverse spectrum of artificial composites and material systems has emerged. This includes but is not limited to photonic crystals $[1]$ , metamaterials $[2]$ , metasurfaces $[3]$ , nanocolloids $[4]$ , and two-dimensional (2D) materials $[5]$ . The materials explored in these endeavors have subsequently found extensive utility in various industrial applications. A unifying feature of these advanced materials is their inherent spatial inhomogeneity, which accounts for their complex electromagnetic properties and sharply distinguishes them from naturally occurring, spatially uniform materials.

Among the categories mentioned previously, photonic crystals are particularly noteworthy, as they are arguably the most thoroughly investigated material platform, well-established, and application-ready. In the most basic one-dimensional (1D) geometry, photonic crystals constitute a multilayer structure with a periodicity $P_{m}$ comparable to the light wavelength. In its simplest form, each period comprises alternating layers of two distinct materials characterized by permittivities $\varepsilon_{1}$ and $\varepsilon_{2}$ (see illustration in Fig. 1(a)). When discussing the propagation of light in such a system, we usually ask: what do the eigenmodes look like, and what is the dispersion relation of the associated eigenvalues? Here, the eigenmodes are the elementary solutions to Maxwell's equations in such a medium without external sources. Once they are known, an arbitrary solution to Maxwell's equation can be written as a superposition of such eigenmodes weighted with suitable amplitudes. In extension, the dispersion relation is a governing equation that relates the parameters that characterize the eigenmodes. For a plane wave, being the eigenmode of the homogeneous space, the parameters are the frequency and the wave vector components.

Now, a fundamental principle of physics, the Noether theorem, says that for every continuous symmetry of a physical system, there exists a conserved quantity. Owing to their continuous time-translational symmetry, the frequency of the eigenmode of photonic crystals remains conserved. However, because of only the discrete space-translational symmetries, their wavenumbers k are conserved only up to the addition of a multiple of the reciprocal lattice vector $2\pi N/P_{m}$ [1, p. 35], with N being an

Figure 1

![](images/0bc8337e59e6b149d2a2cd9a2c2a3c991ed73b9427b2f91a058448bf9a13ba7c.jpg)

![](images/9894ffe8f171150d75b32dac19cb452b61f6931f2c92052fc749dde208ab2f05.jpg)

![](images/5d601ea0e435b0fd5eb67793344680c76e16c60b5ce1d37a77b99696da9ca1b5.jpg)
(b)

![](images/7acf5e471db7e82def0e45123a4ade682de7fffc01e1326192725743098d9af8.jpg)
(d)
(a) Schematics of a conventional 1D photonic (space) crystal where permittivity $\varepsilon$ is modulated along one spatial coordinate while it is constant in time. Moreover, the crystal is uniform along the other two spatial directions. (b) Characteristic band diagram of a 1D photonic crystal. Here, the dispersion relation is shown, expressing the functional dependency of the frequency on the real part of the wavenumber $\Re(k)$ that expresses the phase variation of the eigenmode along the direction of the periodic material modulation. The third coordinate axis denotes the imaginary part of wavenumber $\Im(k)$ . For simplicity, only the two lowest bands (shown in blue) are depicted inside the Brillouin zone (whose edge is indicated with the dashed line). There is a frequency domain, the energy bandgap (shown in light gray), in which no propagating solutions to Maxwell's equations exist. Inside the energy bandgap, two eigenmodes are also supported (shown in red and green), but they decay in space along the positive and negative spatial directions, respectively. (c) Schematics of a photonic time crystal where the permittivity is a periodic function of time only. Otherwise, the spatial distribution of the permittivity is uniform in all three directions. (d) Characteristic band diagram of a photonic time crystal. The third coordinate axis denotes the imaginary part of frequency $\Im(\omega)$ . Inside the momentum bandgap, two eigenmodes are supported: one exponentially growing (shown in red) and one exponentially decaying (shown in green) in time.

integer. Consequently, just as the photonic crystal is periodic in space, expressed as $\varepsilon(z + P_{\mathrm{m}}) = \varepsilon(z)$ for the 1D version, the dispersion relation exhibits a periodicity in the reciprocal space, expressed as $\omega(k + 2\pi/P_{\mathrm{m}}) = \omega(k)$ . For a finite permittivity contrast, i.e., $\varepsilon_{1} \neq \varepsilon_{2}$ , eigenmodes that differ by $2\pi/P_{m}$ in wavenumber (momentum) have lifted frequency degeneracy, resulting in a frequency (energy) bandgap (see illustration in Fig. 1(b)).

Remarkably, these photonic bandgaps have been known since 1887 [6]. Most importantly, light with frequencies within these photonic bandgaps cannot propagate inside the photonic crystal. Indeed, within such a bandgap, two eigenmodes exist but with complex wavenumbers (see Fig. 1(b)). These eigenmodes are evanescent and decay exponentially in space in either the positive or negative spatial directions along the crystal's periodicity. Those eigenmodes have a nonvanishing amplitude after excitation only close to the photonic crystal interface, and the requirement on an exponential decay explains which of the two eigenmodes is excited.

The significance of frequency bandgaps cannot be overstated, as they fundamentally drive the practical applications of 2D and three-dimensional (3D) photonic crystals. Examples of such applications include optical fibers, light-emitting diodes, solar cells, and biosensors, among others $[7–12]$ . However, it is not just the bandgap that has drawn attention. The promise to tailor the dispersion relation and, with that, the isofrequency surfaces, holds the key to controlling the light propagation comprehensively.

In special relativity and electrodynamics, the interwoven relationship between time and space coordinates naturally invites exploration into the behavior of light in materials with temporal inhomogeneity. These sophisticated materials, variously termed “temporal,” “dynamic,” or “time-varying” in the scientific literature, exhibit properties that vary over time on scales comparable to the oscillation period of the incident electromagnetic wave $[13–18]$ . It is crucial to differentiate these rapidly modulated time-varying systems from adiabatically slowly varying tunable and reconfigurable materials $[19]$ . The latter always operate in the steady-state regimes, possibly switching between several states. In sharp contrast, time-varying systems, as considered here, always operate in the transient regime due to the ultrafast modulation, which enables novel, unique phenomena in light–matter interactions.

Considering the possibility of rapidly modulating material properties, the concept of photonic crystals can be extended to a temporal context. We will consider a spatially uniform material but with material parameters, such as the permittivity, that oscillate periodically in time with a period $T_{m}$ (as illustrated in Fig. 1(c)). Such materials, which possess continuous space-translational and discrete time-translational symmetries, have been categorized recently as “photonic time crystals,” “temporal photonic crystals,” “pure-time crystals,” etc. [15,20,21]. For the purposes of this tutorial, we use the term “photonic time crystals” (PTCs), and it is these PTCs that we focus on here.

Please note the subtle but important choice of terminology of a photonic time crystals. That is done to distinguish PTCs from the newly introduced “time crystals,” which are quantum many-body systems that spontaneously violate time-translation symmetry in the presence of a drive that is uncorrelated with the system’s periodicity $[22]$ . As we often encountered at conferences or more general discussion questions to which extent PTCs are related to these “time crystals,” we wanted to emphasize upfront that there is no relation between them.

Intriguingly, the band structure of PTCs is analogous to that of spatial photonic crystals but rotated by $90^{\circ}$ in the $\omega-k$ plane [23] (see Fig. 1(d)). In the following sections, we explain in more detail how to derive this dispersion relation and elaborate on its physical meaning. However, as an introduction, we wanted to highlight that in PTCs, the wavenumbers of eigenmodes are conserved thanks to their spatial uniformity. In contrast, their frequencies are conserved only up to the addition of a multiple of $2\pi N/T_{m}$ . Unlike conventional photonic crystals, which feature energy bandgaps, PTCs exhibit “momentum bandgaps.” Notably, these momentum bandgaps host two eigenmodes with complex frequencies (see Fig. 1(d)): one of these eigenmodes decays over time while the other undergoes amplification. Such mode amplification is allowable in PTCs since they are non-energy-conserving systems, i.e., non-Hermitian, that can extract energy from the modulation source to permit such an amplification. The phenomenon of momentum bandgaps represents one of the principal drivers for the recent rapid growth of scientific interest in PTCs. However, despite most research on PTCs has been emerging only in the past five years, examining and recognizing the foundational historical developments that have led to this new field remains crucial.

The seminal work on what is now known today as PTCs was first published in 1958 by Morgenthaler [24]. He investigated wave propagation in a bulk spatially uniform material with time-varying permittivity and permeability. While ground-breaking, his work was limited to solving special cases of temporal modulation. Around the same period, a separate line of research began to examine wave propagation in systems characterized by a spatiotemporal modulation. This research opened up possibilities for parametric amplification in diverse systems such as transmission lines $[25,26]$ , guided-mode systems $[27–30]$ , and nonlinear optical materials $[31–34]$ . Another pivotal contribution to exploring PTCs was made by Holberg and Kunz $[35]$ . They focused on materials with a distinct type of permittivity modulation, resulting in eigenmodes that could be analytically described using Mathieu functions. Importantly, they were among the first to demonstrate theoretically eigenmode amplification within a momentum bandgap. However, even earlier discussions of momentum bandgaps and the complex-frequency modes they support can be attributed to a 1958 publication by Sturrock $[36]$ . Sturrock employed general kinematic principles to analyze wave functions in propagating systems. Subsequent studies further explored dispersion relations with momentum bandgaps in systems with space–time modulations $[37–39]$ . Other influential early works in the area of PTCs addressed various aspects. Examples are the excitation problem of the crystal $[40]$ , momentum bandgaps at time-varying impedance boundaries $[41]$ , finite spatial extents of PTCs $[42,43]$ , and self-modulated crystals based on electron plasmas $[44]$ . In the early 21st century, the field witnessed significant expansion with the introduction of a generalized framework by the group of O'Reilly $[23]$ . This framework encompassed wave propagation in bulk materials subject to a spatiotemporal rectangular-shaped modulation, and it posited that energy and momentum bandgaps are specific examples within a broader category of forbidden bandgaps. Finally, the group of Halevi explored different aspects of wave propagation and parametric resonances in PTCs, including those with temporal modulation of a general form $[20,45,46]$ , as well as performed the first experimental observation of the momentum bandgap $[47,48]$ .

Over the past decade, the field of PTCs has witnessed significant expansion, garnering substantial interest from the research community. This growth is attributed to the following two primary factors.

(i) The distinct and intrinsically significant effects PTCs exhibit on light–matter interactions have been recognized. Analogous to how traditional photonic crystals offer remarkable spatial light concentration and suppress the spontaneous emission of quantum emitters within the crystal $[49–52]$ , PTCs, when operating in optical domains proximal to electronic transitions in solids, have the potential to intensify interaction with matter strongly. This has been evidenced by the ability of PTCs to amplify the spontaneous emission from excited atoms $[53]$ , support subluminal Cherenkov radiation $[54,55]$ , facilitate superluminal momentum-gap solitons $[56]$ , and host a temporal counterpart of Anderson localization $[57–59]$ , among other phenomena.

(ii) Current advancements in material science reveal great potential for fabricating the first PTCs that operate within the optical domain $[60]$ . The realization of such PTCs necessitates temporal permittivity modulations at exceptionally high frequencies, typically twice that of the light probing the response, and with sufficiently large amplitudes $[20,21,61]$ . Experimental findings suggest that transparent conductive oxides (TCOs), specifically indium tin oxide (ITO) and aluminum-doped zinc oxide (AZO), could meet both criteria when operated in the epsilon-near-zero (ENZ) domain $[62–67]$ .

However, to reach the successful synthesis of PTCs from these materials, two primary challenges remain to be addressed: (i) the high pumping power requirements that could induce thermal degradation of the modulated material $[61]$ and (ii) the inadvertent excitation of a “dynamic-grating” nonlinear effect, potentially hindering the observation of the PTC state $[68]$ . Notwithstanding these obstacles, ongoing research endeavors in this domain persist, and alternative promising methodologies for constructing optical PTCs continue to emerge $[69–72]$ .

In this tutorial, we present the governing physics and potential applications of PTCs in an educational manner, accessible to readers without a solid background in this field. We start with the description of the eigenmodes in PTCs and the properties of the momentum bandgaps (Section 2). This is followed by considerations of practical aspects of realistic PTCs, such as effects of frequency dispersion, topology, finite spatial and temporal extent of the crystal (Section 3). Next, Section 4 investigates the relations between PTCs and other related concepts, including parametric amplification in distributed systems, nonlinear optics, and quantum time crystals. Material candidates for implementation of PTCs are reviewed in Section 5. Section 6 describes the implications and potential applications of PTCs. Finally, in Section 7, we explore the additional opportunities in PTCs provided by introducing spatial modulation. We finish this tutorial with an outlook and concluding remarks in the two last sections. For the convenience of the reader, we attached as Supplementary Material two numerical codes for calculating the band structures of photonic time crystals using the plane wave expansion method (Code 1, Ref. [73]) and the transfer matrix method (Code 2, Ref. [74]).

Although this is the first, to the best of the authors' knowledge, in-depth tutorial article focusing on PTCs, it is important to acknowledge other related reviews and perspectives. The theory and applications of general time-modulated materials and systems were reviewed in Refs. [17,75–80]. Future visions and challenges for PTCs were overviewed in perspectives [81–83]. Discussion on potential material platforms for PTCs can be found in Refs. [60,84]. Physics and applications of space-time metamaterials and metasurfaces were described in Refs. [15–18,85–87].

Finally, it is important to note that throughout this article, time-harmonic oscillations in the form $e^{j\omega t}$ are assumed according to the conventional electrical engineering notation [88,89]. Moreover, all the field quantities in the frequency domain are marked with the tilde symbol “\~” on top to differentiate them from the same quantities in the time domain. For example, the electric fields in the two domains are denoted as $\mathbf{E}(t)$ and $\tilde{\mathbf{E}}(\omega)$ . Material parameters, in contrast, are identified with their specific domain through explicit notation in the argument brackets. We strive to be consistent and do not wish to drop any of the arguments, as the arguments tell us something about the space in which they live. For example, $\varepsilon(t)$ describes a material that is nondispersive but has time-varying properties; in contrast, $\varepsilon(\omega)$ would describe a stationary but dispersive material.

## 2. EIGENMODES IN PTCs

This section introduces the basic properties of PTCs from first principles. It is structured into two sections. In the first section, we derive the general solution of Maxwell's equations in a PTC (see the illustration of the geometry in Fig. 1(c)). We stress upfront that two different approaches are chosen. One of them solves Maxwell's equations fully in the Fourier domain, whereas the other exploits a transfer matrix technique. By discussing the details of the solutions, we overview the concepts of the momentum bandgap and other electromagnetic effects inside the momentum bandgap in the second subsection of this section. The rigorous mathematical and physical frameworks developed here are pivotal in their own right and serve as prerequisites for comprehending the complex phenomena discussed in subsequent sections of this tutorial.

## 2.1. Governing Equations

This section considers the basic concepts and concisely describes the time- and frequency-domain constitutive relations associated with a linear time-varying medium. Although we outline constitutive relations applicable for the general temporally nonlocal (dispersive) medium, we focus here, in Section 2, on the instantaneous (dispersionless) response when deriving the time-domain wave equation and plotting the band structure of an exemplary medium. We stress that this assumption of an instantaneous response in this section is only made to permit a more transparent discussion of the effects linked to the time modulation. The band structure analysis of time-varying media in the presence of frequency dispersion is explored in Section 3, when more realistic aspects of PTCs are discussed.

In the following, we also outline two different methods for calculating the band structure of a periodically time-varying medium based on the Floquet theorem. The Floquet theorem is sometimes also referred to as the Bloch theorem. Both theorems reveal that in periodic systems, wave functions can be expressed as a product of a periodic function and a simpler, well-understood function (like a plane wave). While the Bloch theorem is historically applied for spatially periodic systems (like electronic or photonic crystals), the Floquet theorem is typically applied for temporally periodic systems (like certain quantum or classical wave problems, including PTCs) [90]. One of the methods for calculating the band structure of a periodically time-varying medium is based on the plane-wave expansion [20]. The other method exploits a transfer matrix formalism, sometimes called the ABCD-matrix formalism [21]. The former method is suitable for PTCs with a continuous and smooth variation of the time-dependent permittivity, i.e., ideally in a sinusoidal fashion. In contrast, the latter method is convenient when a stepwise change in the permittivity is considered. The results should be independent of the chosen method, but the numerical convenience strongly differs depending on the modulation type. In addition, traditional numerical techniques to solve Maxwell's equations can be used to obtain the band structure, of course [91].

## 2.1a. Description of a Linear, Isotropic, Spatially Homogeneous, Time-Invariant, and Dispersive Medium

To set the stage, we remind the reader of the constitutive relation for most natural materials. These constitutive relations must supplement the Maxwell equations to make them solvable.

In nature, we encounter a temporal nonlocal response as an inherent characteristic of any material that stems from inertia. Specifically, it implies that a response, such as the induced polarization density $\mathbf{P}(\mathbf{r},\mathbf{t})$ (here r is the spatial dependence), always experiences a delay to an action, such as the applied electric field $\mathbf{E}(\mathbf{r},\mathbf{t})$ or magnetic field $\mathbf{H}(\mathbf{r},\mathbf{t})$ . Combined with causality, we assert that the response depends on the action only in the past, and the delay time between the response and the action is always positive. Moreover, for a material whose properties do not depend on time, i.e., it is a time-invariant medium, the absolute times do not matter but only the time difference between action and response. In electromagnetic theory, this fundamental principle defines the conventional constitutive relation. In the time domain, this constitutive relation must be written as a convolution integral, in which the polarization density is found as the convolution of the impulse response function $R(\mathbf{r},\mathbf{t}')$ with the electric field. For simplicity, we assume an isotropic medium. Anisotropic materials will matter in Section 3.4. Under all these assumptions, the polarization is expressed as

$$
\mathbf {P} (\mathbf {r}, t) = \varepsilon_ {0} \int_ {0} ^ {+ \infty} R (\mathbf {r}, t ^ {\prime}) \mathbf {E} (\mathbf {r}, t - t ^ {\prime}) \mathrm{d} t ^ {\prime}.\tag{1}
$$

Here, $t'$ is the delay time between the action and the response, i.e., the electric field and the polarization density. The second variable t is the observation time. The response function expresses the induced polarization density for a punctual excitation in time. It is the response of the medium, and the total polarization is just the sum, or in a continuous manner, the integral, of the response due to all the excitations from the past. We assumed that the medium is not bianisotropic. A bianisotropy would have caused an electric response in the polarization density from the magnetic field.

Indeed, the above convolution integral properly models a causal and nonlocal response in time (see, e.g., Refs. [92] and [93, p. 330]). It should be noted that throughout the paper, we assume a spatially local material response. It implies that the induced polarization only depends on the electric field at the same spatial location. However, the theory could be further generalized to spatially nonlocal materials. For a spatially homogeneous medium, as assumed here for the moment, there is also no explicit space dependency of the response function, i.e., $R(\mathbf{r}, t') = R(t')$ .

Expressing constitutive relations in the time domain on the base of such a convolution is to some extent inconvenient. Therefore, we go to a reciprocal space for linear systems by Fourier transforming all involved quantities. The electric field, for example, can be written as

$$
\mathbf {E} (\mathbf {r}, t) = \frac {1}{2 \pi} \int_ {- \infty} ^ {+ \infty} \tilde {\mathbf {E}} (\mathbf {r}, \omega) \exp (j \omega t) d \omega .\tag{2}
$$

Then, we can write the constitutive relation as a product in the frequency domain, which reads as

$$
\tilde {\mathbf {P}} (\mathbf {r}, \omega) = \varepsilon_ {0} \chi (\omega) \tilde {\mathbf {E}} (\mathbf {r}, \omega),\tag{3}
$$

where the susceptibility $\chi(\omega)$ is the Fourier transform of the response function $R(t')$ . Again, for an isotropic medium, this is a scalar function. For an anisotropic medium, it would be a tensor.

It remains to be mentioned that the electric flux density, also called the electric displacement field, in the time domain is the sum of the electric field multiplied by the vacuum permittivity and the polarization density,

$$
\mathbf {D} (\mathbf {r}, t) = \epsilon_ {0} \mathbf {E} (\mathbf {r}, t) + \mathbf {P} (\mathbf {r}, t).\tag{4}
$$

In the frequency domain, it reads as

$$
\tilde {\mathbf {D}} (\mathbf {r}, \omega) = \epsilon_ {0} \tilde {\mathbf {E}} (\mathbf {r}, \omega) + \tilde {\mathbf {P}} (\mathbf {r}, \omega).\tag{5}
$$

So far, so conventional. An intriguing question arises when considering a linear medium that is not temporally invariant. For example, the number of atoms per unit volume, the damping coefficient, or the resonance frequency of atoms' response varies over time (see, e.g., Fig. 2). In this case, the invariance under time translation is broken, and the medium becomes nonstationary (in other words, if the cause shifts in time, the macroscopic response does not shift by the same amount of time). Foundational questions arise: How does the constitutive relation change in the frequency domain? And how does this change affect the dispersion curves for plane wave solutions of Maxwell's equations? In the following, we are going to answer these questions.

## 2.1b. Extension to a Time-Varying Medium

For a homogeneous, time-varying dielectric medium that is linear and causal, we write the general nonlocal constitutive relation between the electric flux density $\mathbf{D}(\mathbf{r}, t)$ and

Figure 2

![](images/deabd21ed54718880fecb57937c9003c45fb556e941536cc64dac231c048e689.jpg)
Conceptional representation of a linear time-varying artificial medium. Here, we consider the medium to be made from periodically arranged meta-atoms that change, in this specific example, their properties in time. For that, the meta-atoms are described as harmonic oscillators with a time-varying spring constant $\kappa_{\mathrm{s}}(t)$ .

the electric field $\mathbf{E}(\mathbf{r},t)$ as

$$
\mathbf {D} (\mathbf {r}, t) = \varepsilon_ {0} \mathbf {E} (\mathbf {r}, t) + \varepsilon_ {0} \int_ {0} ^ {+ \infty} R (t ^ {\prime}, t) \mathbf {E} (\mathbf {r}, t - t ^ {\prime}) d t ^ {\prime}.\tag{6}
$$

Compared with Eq. (1), here, the response function depends on the two time variables $t'$ and t. The former expresses the memory of the material, and the latter expresses the explicit time dependency of the material properties.

In Eq. (6), the electric field at each point in space can be an arbitrary function of time. Therefore, we can write the electric field again as the inverse Fourier transform of $\tilde{\mathbf{E}}(\mathbf{r},\omega)$ , and we replace the electric field in the constitutive relation (see Eq. (6)) by the above expression. Consequently, we achieve an important expression for the instantaneous electric flux density, which reads as

$$
\mathbf {D} (\mathbf {r}, t) = \frac {\varepsilon_ {0}}{2 \pi} \int_ {- \infty} ^ {+ \infty} \varepsilon_ {\mathrm{T}} \left(\omega^ {\prime}, t\right) \tilde {\mathbf {E}} \left(\mathbf {r}, \omega^ {\prime}\right) \exp \left(j \omega^ {\prime} t\right) \mathrm{d} \omega^ {\prime}.\tag{7}
$$

Here, we introduced the temporal complex dielectric function $\varepsilon_{\mathrm{T}}(\omega', t) = 1 + \chi_{\mathrm{T}}(\omega', t)$ , and the temporal complex susceptibility is defined as

$$
\chi_ {\mathrm{T}} \left(\omega^ {\prime}, t\right) = \int_ {0} ^ {+ \infty} R \left(t ^ {\prime}, t\right) \exp \left(- j \omega^ {\prime} t ^ {\prime}\right) \mathrm{d} t ^ {\prime}.\tag{8}
$$

The subscript “T” should remind the reader that this quantity lives partially in time and partially in the frequency domain. It can be seen that the complex susceptibility $\chi_{\mathrm{T}}(\omega', t)$ is merely the Fourier transform of the response function $R(t', t)$ with respect to the delay time $t'$ . These definitions and expressions (discussed in Refs. [78,92] and recently reviewed in Ref. [94]) closely mirror those employed in the study of linear variable networks, originally introduced in 1950 [95].

According to Eq. (7), we explicitly observe that by taking the second Fourier transform, we fully leave the time domain and can investigate light–matter interactions utterly in the frequency domain. However, this second Fourier transform should be done concerning the time variable t. Therefore, we must apply a different angular frequency notation rather than $\omega'$ . Here, we use the letter $\omega$ (that we employed before for the time-invariant case) for this purpose. Knowing that $\exp(j\omega't)$ gives rise to a shift in the frequency domain, we conclude that [92]

$$
\tilde {\mathbf {D}} (\mathbf {r}, \omega) = \frac {\varepsilon_ {0}}{2 \pi} \int_ {- \infty} ^ {+ \infty} \varepsilon \left(\omega^ {\prime}, \omega - \omega^ {\prime}\right) \tilde {\mathbf {E}} \left(\mathbf {r}, \omega^ {\prime}\right) d \omega^ {\prime}.\tag{9}
$$

Consequently, in this equation, there are two angular frequencies. One of them $(\omega')$ refers to the temporal nonlocality or dispersion, and the presence of the other one $(\omega)$ is due to the time-variance of the material and Fourier transform concerning the observation time.

## 2.1c. Assumption of an Instantaneous Response

To simplify the discussion in this section, we assume that the response of the material is instantaneous. In that case, we say that the medium is dispersionless, i.e., it shows the same response at every frequency.

A medium has an instantaneous (dispersionless) response if the induced electric flux density depends only on the electric field at the observation time t. From this point of view, there is no temporal nonlocality, and the response function must be represented by a Dirac delta distribution, allowing a temporal local response. For that, the argument of the Dirac delta distribution should be only the time variable $t'$ . However, since the system is time-varying, the response function continues to include another term depending on the time variable t. Accordingly, we eventually write the response function as $R(t', t) = \delta(t')\chi(t)$ . By having this expression and applying Eq. (6), in the time domain, we see that $\mathbf{D}(\mathbf{r}, t) = \varepsilon_{0}\varepsilon(t)\mathbf{E}(\mathbf{r}, t)$ , where $\varepsilon(t) = 1 + \chi(t)$ .

Since in a PTC, $\varepsilon(t)$ is a periodic function with a temporal period $T_{m}$ (see Fig. 1(c)), we note that $\varepsilon(t + T_{\mathrm{m}}) = \varepsilon(t)$ . Due to this property, we expand $\varepsilon(t)$ into the Fourier series:

$$
\varepsilon (t) = \sum_ {p} \epsilon_ {p} e ^ {j p \omega_ {\mathrm{m}} t},\tag{10}
$$

in which $\epsilon_{p}$ are the Fourier coefficients, and $\omega_{m}=2\pi/T_{m}$ denotes the angular modulation frequency.

In the end, to briefly mention another feature of the instantaneous response for the interested reader, we refer to the modification of Eq. (9). In fact, due to the presence of the Dirac delta distribution that we explained previously, the relative permittivity $\varepsilon(\omega',\omega)$ does not include the independent variable $\omega'$ , and it is only a function of $\omega$ . In this scenario, Eq. (9) becomes a convolution integral, and the electric flux density $\tilde{\mathbf{D}}(\mathbf{r},\omega)$ is proportional to the convolution of the relative permittivity $\varepsilon(\omega-\omega')$ and the electric field $\tilde{\mathbf{E}}(\mathbf{r},\omega')$ in the frequency domain.

## 2.1d. Band Structure Analysis Based on the Plane Wave Expansion

To derive the band structure of periodically time-varying media with an instantaneous response, we use in this subsection the plane wave expansion [20]. According to the Floquet theorem [78, Section 4.2], the electric field can be written as a product of the exponential function $e^{j\omega_{\mathrm{F}}t}$ and a function periodic in time, where the periodicity is the same as that of the time-varying material properties. Here, $\omega_{\mathrm{F}}$ is called the Floquet angular frequency. Using the Fourier series, we can express the electric field for an eigenwave propagating along the $+z$ direction inside a PTC as

$$
\mathbf {E} (\mathbf {r}, t) = \sum_ {n} E _ {n} (\omega_ {\mathrm{F}}) e ^ {j \omega_ {n} t} e ^ {- j k z} \mathbf {a} _ {x},\tag{11}
$$

in which $\omega_{n} = \omega_{F} + n\omega_{m}$ is the nth frequency harmonic, $E_{n}(\omega_{\mathrm{F}})$ is the amplitude of the frequency harmonic $\omega_{n}$ at the Floquet angular frequency $\omega_{F}$ , and k represents the phase constant (wavenumber). Note that the phase constant is fixed for all these harmonics and equals k. That is because the material modulation occurs in time, whereas the susceptibility and permittivity are uniform in space. Hence, the phase constant is a conserved quantity. The polarization of the field was arbitrarily assumed to be in the x direction. In general, in the spatially homogeneous medium, the eigenmodes are elliptically polarized. Moreover, for convenience, we have chosen here a complex notation. However, it should be stated explicitly that the experimentally observable field corresponds only to the real part of that quantity.

In addition to the Floquet theorem, we also need to infer the wave equation, and, certainly, we need to start from Maxwell's equations in the absence of sources, which are written for our study as

$$
\nabla \times \mathbf {E} (\mathbf {r}, t) = - \mu_ {0} \frac {\partial \mathbf {H} (\mathbf {r} , t)}{\partial t}, \quad \nabla \cdot \mathbf {E} (\mathbf {r}, t) = 0,\tag{12a}
$$

$$
\nabla \times \mathbf {H} (\mathbf {r}, t) = \varepsilon_ {0} \frac {\partial}{\partial t} [ \varepsilon (t) \mathbf {E} (\mathbf {r}, t) ], \quad \nabla \cdot \mathbf {H} (\mathbf {r}, t) = 0.\tag{12b}
$$

Regarding Eq. (12a), we apply the curl operator to both sides of the equation, and, subsequently, we use the information of Eq. (12b) about the curl of the magnetic field as well as the information about the divergences of the fields that are zero. After traditional algebraic manipulations, we achieve the wave equation

$$
\nabla^ {2} \mathbf {E} (\mathbf {r}, t) - \frac {1}{c ^ {2}} \frac {\partial^ {2}}{\partial t ^ {2}} \Bigl (\varepsilon (t) \mathbf {E} (\mathbf {r}, t) \Bigr) = 0.\tag{13}
$$

Here, $c = 1/\sqrt{\varepsilon_{0}\mu_{0}}$ represents the speed of light in vacuum. Substituting Eqs. (10) and (11) into this equation results in the master equation for finding the band structure of the PTC. In Eq. (13), the Laplacian operator is easily replaced by the square of the phase constant because this operator is about space variations (i.e., $\nabla \rightarrow -jk$ and $\nabla^{2} \rightarrow -k^{2}$ ). The master equation is derived as follows:

$$
\sum_ {n} \sum_ {p} \frac {[ \omega_ {\mathrm{F}} + (n + p) \omega_ {\mathrm{m}} ] ^ {2}}{c ^ {2}} \epsilon_ {p} E _ {n} e ^ {j [ \omega_ {\mathrm{F}} + (n + p) \omega_ {\mathrm{m}} ] t} = k ^ {2} \sum_ {n} E _ {n} e ^ {j \omega_ {n} t}.\tag{14}
$$

On the left-hand side of Eq. (14), we shift the index $n$ to $n - p$ . Consequently, Eq. (14) can be simplified to

$$
\sum_ {n} \sum_ {p} \frac {(\omega_ {\mathrm{F}} + n \omega_ {\mathrm{m}}) ^ {2}}{c ^ {2}} \epsilon_ {p} E _ {n - p} e ^ {j \omega_ {n} t} = k ^ {2} \sum_ {n} E _ {n} e ^ {j \omega_ {n} t}.\tag{15}
$$

We can see from Eq. (15) that both sides share the same basis $e^{j\omega_n t}$ . Therefore, the summation in terms of $n$ can be removed. By shifting now the index $p$ to $n - p$ , we infer that

$$
\sum_ {p} \frac {(\omega_ {\mathrm{F}} + n \omega_ {\mathrm{m}}) ^ {2}}{c ^ {2}} \epsilon_ {n - p} E _ {p} - k ^ {2} E _ {n} \delta_ {p n} = 0,\tag{16}
$$

where $\delta_{pn}$ is the Kronecker delta function, which is valid for each index n [20]. Equation (16) could also be written in a matrix form where a square matrix of infinite size $\overline{\overline{U}}$ is multiplied by a vector $\overline{V}$ describing the field amplitudes of harmonics for a given Floquet frequency $\omega_{F}$ . We can write Eq. (16) in the following form:

$$
\underbrace {\left[ \begin{array}{c c c c c} \ddots & \ddots & \ddots & \ddots & \ddots \\ \ddots & \zeta_ {- 1} \epsilon_ {0} - k ^ {2} & \zeta_ {- 1} \epsilon_ {- 1} & \zeta_ {- 1} \epsilon_ {- 2} & \ddots \\ \ddots & \zeta_ {0} \epsilon_ {1} & \zeta_ {0} \epsilon_ {0} - k ^ {2} & \zeta_ {0} \epsilon_ {- 1} & \ddots \\ \ddots & \zeta_ {1} \epsilon_ {2} & \zeta_ {1} \epsilon_ {1} & \zeta_ {1} \epsilon_ {0} - k ^ {2} & \ddots \\ \ddots & \ddots & \ddots & \ddots & \ddots \end{array} \right]} _ {\overline {{\overline {{U}}}} (\omega_ {\mathrm{F}}, k)} \cdot \underbrace {\left[ \begin{array}{c} \vdots \\ E _ {- 1} \\ E _ {0} \\ E _ {1} \\ \vdots \end{array} \right]} _ {\overline {{V}}} = 0,\tag{17}
$$

in which $\zeta_{n}(\omega_{\mathrm{F}}) = (\omega_{\mathrm{F}} + n\omega_{\mathrm{m}})^{2} / c^{2}$ .

We observe that the square of the phase constant $k^{2}$ is accompanied by the Fourier coefficient $\epsilon_{0}$ (not to be confused with vacuum permittivity $\varepsilon_{0}$ ), and both appear only in the diagonal elements of the matrix. In most PTCs, amplitudes of high-order harmonics decay rapidly with the harmonic number. Therefore, it is possible to truncate the square matrix to that of a finite size, i.e., $(2N+1)\times(2N+1)$ , where N means considering the harmonics from $-N\leq n\leq+N$ .

Equation (17) defines an eigenvalue problem and has a nontrivial solution if the determinant of the square matrix vanishes, i.e., $\det\left[\overline{\overline{U}}(\omega_{\mathrm{F}}, k)\right]=0$ . It allows us to calculate the band structure, demonstrating the relation between $\omega_{F}$ and k. From Eq. (17), one can observe an important property of the band structure, its periodicity. Indeed, one can replace all $\zeta_{n}(\omega_{\mathrm{F}})$ in the determinant of the matrix with equivalent terms $\zeta_{n-1}(\omega_{\mathrm{F}}+\omega_{\mathrm{m}})$ . The resulting determinant will be the same as the initial one (due to the infinite size of the matrix) with the only change that all parameters depend on $\omega_{F}+\omega_{m}$ . Thus, the eigenvalue solution of the new determinant $\det\left[\overline{\overline{U}}(\omega_{\mathrm{F}}+\omega_{\mathrm{m}}, k)\right]$ must be equal to that of the initial determinant $\det\left[\overline{\overline{U}}(\omega_{\mathrm{F}}, k)\right]$ , which leads to

$$
k (\omega_ {\mathrm{F}} + \omega_ {\mathrm{m}}) = k (\omega_ {\mathrm{F}}).\tag{18}
$$

Relation (18) brings us to the important implication: the band structure of a PTC is periodic in $\omega_{F}$ with the period being the modulation frequency $\omega_{m}$ . Moreover, from Eq. (17) one can find that eigenmode solutions for two different eigenfrequencies $\omega_{F}$ and $\omega_{F} + q\omega_{m}$ ( $q \in Z$ ) are related to each other through the relation [20]

$$
E _ {n} (\omega_ {\mathrm{F}} + q \omega_ {\mathrm{m}}) = E _ {n - q} (\omega_ {\mathrm{F}}).\tag{19}
$$

Therefore, without loss of information, it is sufficient to consider only the first period of the band structure $\omega_{\mathrm{F}} \in [-\omega_{\mathrm{m}} / 2, \omega_{\mathrm{m}} / 2]$ , known as the first Brillouin zone of the PTC.

As the simplest but illustrative example, consider a sinusoidal permittivity modulation defined as $\varepsilon(t)=\varepsilon_{\mathrm{av}}(1+m_{\varepsilon}\cos(\omega_{\mathrm{m}}t))$ , where $\varepsilon_{av}$ denotes the time-averaged permittivity, and $m_{\varepsilon}$ is the relative amplitude of the permittivity modulation ranging from 0 to 1. The band structure, depicted in Fig. 3(a), is computed using Eq. (17). The exact numerical parameters are indicated in the figure legend, and we plot the band structure for a frequency interval $\omega_{F}\in[-2\omega_{m},2\omega_{m}]$ . From Fig. 3(a), it is evident that the band structure is periodic in the frequency domain.

The most important feature of PTCs is the bandgap that appears in the momentum domain, as shown in Fig. 3(a). What we see here is the hallmark of a PTC. We see that for a certain momentum interval, Maxwell's equations do not have a propagating solution at any possible real frequency. We will elaborate on that momentum bandgap in much more detail below. Afterward, we discuss the eigenmodes and return to Figs. 3(b)–(d).

![](images/a290a4c78d8c24eb2a8728b433e98d7bbf47655128dd8a3acca8c54228c8876c.jpg)

![](images/a60e0adc7da4c32a94c1af6c04b9a1bda2f32502039c6652856a823e96307772.jpg)
(a) Band structure of the PTC calculated using Eq. (17). The permittivity of the PTC is given by $\varepsilon(t) = \varepsilon_{\mathrm{av}}(1 + m_{\varepsilon}\cos(\omega_{\mathrm{m}}t))$ , where $\varepsilon_{\mathrm{av}} = 5$ and $m_{\varepsilon} = 0.6$ . Distribution of the normalized harmonic magnitudes for different eigenmodes in the PTC. The eigenfrequencies of these modes are: (b) $\omega_{\mathrm{F}} = \omega_{A} + \omega_{\mathrm{m}}$ , (c) $\omega_{\mathrm{F}} = \omega_{A}$ , and (d) $\omega_{\mathrm{F}} = \omega_{B}$ . Note that in (b), (c), and (d), the harmonic distribution is asymmetric with respect to the fundamental harmonic. This asymmetry is caused by the asymmetry of the matrix $\overline{\overline{U}}$ in Eq. (17) since $\omega_{\mathrm{F}} \neq \omega_{\mathrm{m}} / 2$ ( $k$ is outside the momentum bandgap).

The size of the aforementioned momentum bandgap is affected by the modulation depth $m_{\varepsilon}$ . Figure 4 shows the band structure in the first Brillouin zone for three different modulation amplitudes of the same PTC as discussed before. The plots were created using Code 1 attached as Supplementary Material [73]. Within the so-called empty-lattice approximation, i.e., when $m_{\varepsilon} \rightarrow 0$ (upper plot in the figure), the dispersion relation of the PTC can be obtained from the corresponding dispersion relation of the same material without temporal modulations upon “folding” the dispersion relation into the first Brillouin zone. This folding of the bands occurs due to the periodicity of the band structure given by Eq. (18). Once a band reaches one zone edge (e.g., at $\omega_{m}/2$ ), through the translation by $-\omega_{m}$ , it emerges and continues from the opposite edge $-\omega_{m}/2$ . Therefore, due to the periodicity, different branches of the dispersion relation cross at the edge of the Brillouin zone, i.e., when $\omega = \pm\omega_{m}/2$ (see the upper plot in Fig. 4). In the same plot, in the region of small values of wavenumber k, one can see a linear dispersion emerging from $\omega_{F} = 0$ . This is the effective-medium regime, where the modulation frequency of the material properties is much higher than the frequency of the considered light. The light would effectively see a homogeneous medium, and the slope of the dispersion relation reflects the effective properties. In this specific case, the permittivity of the effective medium corresponds to the average permittivity relative to which a modulation occurs.

When the modulation amplitude $m_{\varepsilon}$ is noticeable (see the two bottom plots in Fig. 4), an avoided crossing occurs at the Brillouin zone edge, and a bandgap in the momentum domain opens (shown with the gray shaded region). The curvature of the bands near the bandgap is not linear. In this region of k, the light starts to probe the temporal modulation of the material properties of the PTC. Such a momentum bandgap is the hallmark of PTCs. With the increase in the modulation amplitude, the bandgap size increases. Interestingly, for a sinusoidal modulation with only one frequency, only one momentum bandgap is formed, that is, between the first and second bands. If higher order harmonics are added to the modulation function $\varepsilon(t)$ , more bandgaps will be open at higher momenta. In particular, when the modulation function is stepwise, i.e., it has an infinite number of nonzero Fourier coefficients in its spectrum given by Eq. (10), infinitely many momentum bandgaps open, as discussed in the next subsection. The wave phenomena occurring inside the bandgap are explained in Section 2.2.

![](images/24c70e7369a4302d5ea8677a1f0abd87bd453d6c415aff10d212472139025a1f.jpg)
Band structure of a PTC calculated for three different values of the modulation depth. The permittivity of the PTC is given by $\varepsilon(t) = \varepsilon_{\mathrm{av}}(1 + m_{\varepsilon} \cos(\omega_{\mathrm{m}} t))$ where $\varepsilon_{av} = 5$ . The shaded regions depict the momentum domain in which a bandgap exists.

Looking at Fig. 4, one can observe that the band structure is mirror-symmetric with respect to the $\omega_{\mathrm{F}} = 0$ axis. The reason can be directly observed from Eq. (17). If $\omega_{\mathrm{F}}$ is an eigenfrequency, i.e., $\det \left[\overline{\overline{U}} (\omega_{\mathrm{F}},k)\right] = 0$ , then $\det \left[\overline{\overline{U}} (-\omega_{\mathrm{F}},k)\right] = 0$ must be satisfied, since $\zeta_n(\omega_{\mathrm{F}}) = \zeta_{-n}(-\omega_{\mathrm{F}})$ . Therefore, we have $k(\omega_{\mathrm{F}}) = k(-\omega_{\mathrm{F}})$ . It should be mentioned, however, that this degeneracy does not hold in a general case of PTC, which could consist of, e.g., magneto-optical materials and possess an asymmetric band structure. For example, the emergence of asymmetric band structures in spatial photonic crystals is a well-explored topic [96,97].

Next, we discuss the eigenmodes of Eq. (17). For a given $k$ (see the dashed line in Fig. 3(a)), there are infinitely many eigenfrequencies that could be grouped into two sets: $\omega_{A,n} = \omega_A + n\omega_{\mathrm{m}}$ and $\omega_{B,n} = \omega_B + n\omega_{\mathrm{m}}$ , where $n \in \mathbb{Z}$ . These two arrays of harmonics are marked by red and green dots in Fig. 3(a), respectively. For each eigenfrequency belonging to $\omega_{A,n}$ , one can derive from Eq. (17) a formally different eigenmode characterized by complex amplitudes of each harmonic $E_n(\omega_{\mathrm{F}})$ (see Eq. (11)). However, as it is clear from Eq. (19), the spectral weights of different frequency harmonics $E_n(\omega_{\mathrm{F}})$ remain the same for all these eigenmodes with the only change in relabeling the harmonic numbers [1, p. 47]. This is depicted in Figs. 3(b) and (c), where the amplitudes of the different harmonics contributing to the eigenmodes are compared for two chosen eigenfrequencies $\omega_A$ and $\omega_A + \omega_{\mathrm{m}}$ . Indeed, the harmonic spectra for $\omega_A + \omega_{\mathrm{m}}$ are the same as for $\omega_A$ if shifted by one order to the left. In Fig. 3(c), the harmonic with order $n = 0$ corresponds to the Floquet frequency $\omega = \omega_A$ in the PTC. The corresponding harmonic amplitude distribution $E_n$ for the eigenfrequencies $\omega_{B,n}$ is depicted in Fig. 3(d). In the considered example of the PTC, it is equivalent to that shown in Fig. 3(c) under the sign flip of the harmonic number n. However, in the general case, they could be different. Thus, for a given wavenumber k, the electric field inside a PTC can be viewed as a superposition of eigenmodes (11) with known amplitudes $E_{n}(\omega_{\mathrm{F}})$ . This solution can be used in problems where the wavenumber k in the PTC is fixed to a single value, e.g., the problem of time interface between free space and PTC (see Section 3.3 for more details). However, in the general case, this solution is not complete, as discussed in the following.

From Fig. 4, one can observe that for a given Floquet frequency $\omega_{F}$ , there is a discrete but infinite number of solutions for wavenumber k. We denote them by $k_{p}$ , where $p \in Z$ . Therefore, $k_{p}$ is generally a function of $\omega_{F}$ , and we will write it as $k_{p}(\omega_{\mathrm{F}})$ . Index p can be used to indicate the number of the band in the photonic band structure. Thus, the general solution for the electric field inside a PTC can be written in the following form [20]:

$$
\mathbf {E} (\mathbf {r}, t) = \sum_ {p = 1} ^ {\infty} \sum_ {n = - \infty} ^ {\infty} E _ {p, n} (\omega_ {\mathrm{F}}) e ^ {j \omega_ {n} t} e ^ {- j k _ {p} (\omega_ {\mathrm{F}}) z} \mathbf {a} _ {x},\tag{20}
$$

where $E_{p,n}(\omega_{\mathrm{F}})$ are the eigenmodes complex amplitudes. This solution for the electric field should be used when the wavenumber of the eigenmodes inside the PTC is not fixed, as it is in the scenario of spatially finite PTCs (see Section 3.2 for more details).

Finally, we wish to stress that the band structure with such a technique can also be analyzed while simultaneously considering a time-varying permeability $[98]$ .

## 2.1e. Band Structure Analysis Based on the Transfer Matrix Method

In the previous subsection, we discussed how to obtain the dispersion relation of the eigenmodes in a PTC directly from the wave equation. The quantities that entered that description are the Fourier coefficients describing the time-dependent and periodic modulation of the permittivity. Naturally, such an approach is convenient for a smoothly varying periodic modulation. Ideally, a sinusoidal modulation is considered so that only three Fourier coefficients are sufficient to describe the permittivity modulation. In contrast, stepwise modulations are more demanding to describe because of the many Fourier coefficients that must be considered. For such problems, an alternative approach is much more suitable. It is usually referred to as the transfer-matrix method or the ABCD-matrix method. It allows us to propagate the fields through one unit cell in time by slicing the medium into time segments with constant material properties and connecting the solutions in the different time segments using suitable interface conditions. By imposing additional Floquet-periodic boundary conditions onto time segments corresponding to one period, we can solve for the dispersion relation $[99,100]$ . The method is well known when solving light propagation in spatially stratified media $[101]$ , and its dual version was adapted recently for PTCs $[21,102–105]$ .

Before developing the calculation method for the band structure, it is important to introduce the concept of temporal interface conditions. The corresponding expressions for a spatial interface are well known. As a reminder, let us consider a spatial interface between two different materials. We assume that each material occupies one half-space along the x-coordinate, and the interface shall be at $x = x_{0}$ . Provided that there are no electric and magnetic surface currents flowing at the interface, the tangential electric and magnetic fields must be continuous when transiting through the interface [106, Chapter 7.3.6]. In addition, the tangential component of the wave vector is a preserved quantity, as the interface is assumed to be translational invariant in the corresponding spatial directions. To satisfy those interface conditions, reflection and refraction must happen.

Analogous to a spatial interface, its temporal counterpart is an interface occurring in time between two materials with different properties. It signifies an abrupt transition at time moment $t = t_{0}$ from one material property to another. The materials continue to be spatially uniform, i.e., the properties are assumed to be identical at each spatial coordinate inside the media. Integrating the Maxwell curl equations around the switching time, i.e., from $t_{0}^{-} = t_{0} - s$ to $t_{0}^{+} = t_{0} + s$ and taking the limit of $s \to 0$ , we have

$$
\lim _ {s \rightarrow 0} \int_ {t _ {0} - s} ^ {t _ {0} + s} \frac {\partial \mathbf {B} (\mathbf {r} , t)}{\partial t} d t = - \lim _ {s \rightarrow 0} \int_ {t _ {0} - s} ^ {t _ {0} + s} \nabla \times \mathbf {E} (\mathbf {r}, t) d t,\tag{21a}
$$

$$
\lim _ {s \rightarrow 0} \int_ {t _ {0} - s} ^ {t _ {0} + s} \frac {\partial \mathbf {D} (\mathbf {r} , t)}{\partial t} d t = \lim _ {s \rightarrow 0} \int_ {t _ {0} - s} ^ {t _ {0} + s} (- \mathbf {J} (\mathbf {r}, t) + \nabla \times \mathbf {H} (\mathbf {r}, t)) d t.\tag{21b}
$$

Since the field and source $(\mathbf{E}(\mathbf{r},t),\mathbf{H}(\mathbf{r},t),\text{and}\mathbf{J}(\mathbf{r},t))$ are finite, the integration on the right-hand side of Eq. (21a) and Eq. (21b) is zero [17]. Therefore, their left-hand sides of the equations are also zero, indicating the continuity of field fluxes across the switching time moment $t_{0}$ . This continuity corresponds to the interface condition, and it reads as

$$
\mathbf {D} (\mathbf {r}, t = t _ {0} ^ {+}) = \mathbf {D} (\mathbf {r}, t = t _ {0} ^ {-}),\tag{22a}
$$

$$
\mathbf {B} (\mathbf {r}, t = t _ {0} ^ {+}) = \mathbf {B} (\mathbf {r}, t = t _ {0} ^ {-}).\tag{22b}
$$

To ensure the temporal interface conditions in Eq. (22) at the switching moment, waves undergo phenomena similar to spatial reflection and refraction, referred to as temporal reflection and refraction $[107–109]$ . Reflected and refracted waves at spatial interfaces are separated in space since they propagate in opposite directions and in different half-spaces. However, temporal reflected and refracted waves are not separated spatially. Time refraction has been observed experimentally recently at optical $[64,66,67,110]$ , mid-infrared $[63]$ , and terahertz frequencies $[111]$ , but time reflection proves more challenging due to the difficulty in altering optical properties strongly and instantaneously. Analogously to the spatial interface case, generating strong time reflections requires a strong impedance mismatch at the temporal boundary, which translates into the necessity to have the strong amplitude of the temporal modulations (currently very challenging at optical frequencies) $[17,112]$ . Therefore, time reflection has been observed only at microwave frequencies for electromagnetic waves $[113–115]$ , as well as for water waves $[116]$ , cold atoms $[117]$ , and synthetic dimensions $[118]$ .

When periodically repeating spatial interfaces, a photonic crystal is created, and the energy bandgaps are formed through reflections at consecutive spatial interfaces. The impossibility of traveling waves within a frequency bandgap inside a photonic crystal is usually explained as an interference phenomenon. Consider a plane wave illuminating from outside a 1D photonic crystal. Then, light will experience a sequence of reflections at the spatial interfaces. The optical thickness of the layers is now suitably adjusted so that reflected fields in the backward direction constructively interfere whereas they destructively interfere in the forward direction, forbidding light propagation inside the photonic crystal.

Similarly, when temporal interfaces are repeated periodically in time by jumping between two values of the permittivity, a PTC is created, and the momentum bandgap emerges. The momentum bandgaps are formed due to multiple instances of temporal reflection/refraction at these temporal interfaces $[17]$ . Thus, the band structure of PTCs can be ascertained through these temporal interface conditions $[103,119,120]$ . The derivation of the band structure can parallel the approach for spatial photonic crystals $[101]$ .

![](images/e260c8c839d6e20dda47827e9815ff2ad3f9c53a4afcabc77f42549ece5e2c49.jpg)
Schematic of a PTC made from a sequence of temporal slabs with constant permittivities. Such a structure is perfectly suitable for applying the ABCD-matrix method in the calculation of the band structure. Arrows denote the propagation directions of plane waves denoted by corresponding complex amplitudes $a_{n}$ , $b_{n}$ , $c_{n}$ , and $d_{n}$ .

Recall that in calculating the band structure for photonic crystals, we essentially establish a relation that expresses how the (continuous) tangential components of the field evolve from one interface to the next interface. The field is written between the interfaces as a superposition of a forward- and a backward-propagating field. A $2 \times 2$ matrix can express the evolution of the field from one interface to the following interface. Multiplication of a sequence of matrices allows us to express how the field evolves across one unit cell. Finally, imposing Bloch-periodic boundary conditions, i.e., requiring that the amplitude is the same and only the phase varies by the predefined Bloch phase, allows us to disclose the dispersion relation. A similar procedure will be put in place in case of a PTC.

For that, let us consider a homogeneous, nonmagnetic, isotropic, bulk material. For definiteness, we assume an x-polarized plane wave propagating along the z direction. As shown in Fig. 5, the medium experiences a periodic stepwise temporal permittivity modulation with a period of $T_{m}$ , i.e., $\varepsilon(t + T_{\mathrm{m}}) = \varepsilon(t)$ . Let us consider the nth period of the time evolution, where $(n - 1)T_{\mathrm{m}} < t < nT_{\mathrm{m}}$ . The permittivity variation in this period is divided into two temporal segments or slabs, with duration $t_{1}$ and $t_{2}$ , where $t_{1} + t_{2} = T_{m}$ :

$$
\varepsilon (t) = \left\{ \begin{array}{l l} \varepsilon_ {2}, & (n - 1) T _ {\mathrm{m}} <   t <   (n - 1) T _ {\mathrm{m}} + t _ {2} \\ \varepsilon_ {1}, & n T _ {\mathrm{m}} - t _ {1} <   t <   n T _ {\mathrm{m}} \end{array} \right..\tag{23}
$$

In the time segment filled with yellow color in Fig. 5, the permittivity is constant and equal to $\varepsilon_{1}$ for a duration of $t_1$ . The forward and backward complex amplitudes of the plane waves in this temporal segment can be denoted as $a_{n}$ and $b_{n}$ . The frequency of the wave is $\omega_{1}$ . Therefore, we choose an ansatz for the electric flux in the media as a sum of forward- and backward-propagating waves in time, which can be written as

$$
D _ {x} (z, t) = \left[ a _ {n} e ^ {j \omega_ {1} (t - n T _ {\mathrm{m}})} + b _ {n} e ^ {- j \omega_ {1} (t - n T _ {\mathrm{m}})} \right] e ^ {- j k z}, \quad n T _ {\mathrm{m}} - t _ {1} <   t <   n T _ {\mathrm{m}}.\tag{24}
$$

The corresponding magnetic flux is the time derivative of the electric flux, which can be calculated by $\nabla \times \mathbf{B}(\mathbf{r}, t) = \mu_{0} \frac{\partial \mathbf{D}(\mathbf{r}, t)}{\partial t}$ . The magnetic flux can be calculated as

$$
B _ {y} (z, t) = \frac {\omega_ {1} \mu_ {0}}{k} \left[ a _ {n} e ^ {j \omega_ {1} (t - n T _ {\mathrm{m}})} - b _ {n} e ^ {- j \omega_ {1} (t - n T _ {\mathrm{m}})} \right] e ^ {- j k z}, \quad n T _ {\mathrm{m}} - t _ {1} <   t <   n T _ {\mathrm{m}}.\tag{25}
$$

In the time segment filled with blue color in Fig. 5, the permittivity is constant and equal to $\varepsilon_{2}$ , and the forward and backward complex amplitudes of the propagating plane waves in this temporal segment are denoted as $c_{n}$ and $d_{n}$ . The frequency of the wave in this second segment is $\omega_{2}$ . A frequency change ( $\omega_{1} \neq \omega_{2}$ ) occurs because the spatial homogeneity requires the conservation of the momentum [79]. Still, the dispersion relation of each material needs to be satisfied, suggesting that frequency and momentum are related by

$$
k ^ {2} = \frac {\omega_ {1 , 2} ^ {2}}{c ^ {2}} \varepsilon_ {1, 2}.\tag{26}
$$

Therefore, we have the basic relation between the two frequencies $\omega_{1}\sqrt{\varepsilon_{1}}=\omega_{2}\sqrt{\varepsilon_{2}}$ . The electric and magnetic fluxes in that time segment (filled with blue color) can be likewise written as a superposition of forward and backward waves,

$$
D _ {x} (z, t) = \left[ c _ {n} e ^ {j \omega_ {2} (t - n T _ {\mathrm{m}})} + d _ {n} e ^ {- j \omega_ {2} (t - n T _ {\mathrm{m}})} \right] e ^ {- j k z}, \quad (n - 1) T _ {\mathrm{m}} <   t <   (n - 1) T _ {\mathrm{m}} + t _ {2}\tag{27}
$$

and

$$
B _ {y} (z, t) = \frac {\omega_ {2} \mu_ {0}}{k} \left[ c _ {n} e ^ {j \omega_ {2} (t - n T _ {\mathrm{m}})} - d _ {n} e ^ {- j \omega_ {2} (t - n T _ {\mathrm{m}})} \right] e ^ {- j k z}, \quad (n - 1) T _ {\mathrm{m}} <   t <   (n - 1) T _ {\mathrm{m}} + t _ {2}.\tag{28}
$$

Having spelled out these ansatz for the fields, we need to connect them by imposing the interface conditions given by Eq. (22). First, we consider the field continuity at the switch moment of $t = (n - 1)T_{\mathrm{m}}$ . According to Eq. (24) and Eq. (27), the continuity of $D_{x}$ can be written as the following equation:

$$
a _ {n - 1} + b _ {n - 1} = c _ {n} e ^ {- j \omega_ {2} T _ {\mathrm{m}}} + d _ {n} e ^ {j \omega_ {2} T _ {\mathrm{m}}}.\tag{29}
$$

Combining Eq. (25) and Eq. (28), the continuity of magnetic flux at $t = (n - 1)T_{\mathrm{m}}$ requires that

$$
a _ {n - 1} - b _ {n - 1} = \frac {\omega_ {2}}{\omega_ {1}} c _ {n} e ^ {- j \omega_ {2} T _ {\mathrm{m}}} - \frac {\omega_ {2}}{\omega_ {1}} d _ {n} e ^ {j \omega_ {2} T _ {\mathrm{m}}}.\tag{30}
$$

Equations (29) and (30) can be arranged as a matrix operation,

$$
\left[ \begin{array}{c c} 1 & 1 \\ 1 & - 1 \end{array} \right] \cdot \left[ \begin{array}{c} a _ {n - 1} \\ b _ {n - 1} \end{array} \right] = \left[ \begin{array}{c c} e ^ {- j \omega_ {2} T _ {\mathrm{m}}} & e ^ {j \omega_ {2} T _ {\mathrm{m}}} \\ \frac {\omega_ {2}}{\omega_ {1}} e ^ {- j \omega_ {2} T _ {\mathrm{m}}} & - \frac {\omega_ {2}}{\omega_ {1}} e ^ {j \omega_ {2} T _ {\mathrm{m}}} \end{array} \right] \cdot \left[ \begin{array}{c} c _ {n} \\ d _ {n} \end{array} \right].\tag{31}
$$

Similarly, utilizing the flux continuity at the time moment of $t = nT_{m} - t_{1}$ , we obtain another matrix equation,

$$
\left[ \begin{array}{c c} e ^ {- j \omega_ {1} t _ {1}} & e ^ {j \omega_ {1} t _ {1}} \\ \frac {\omega_ {1}}{\omega_ {2}} e ^ {- j \omega_ {1} t _ {1}} & - \frac {\omega_ {1}}{\omega_ {2}} e ^ {j \omega_ {1} t _ {1}} \end{array} \right] \cdot \left[ \begin{array}{c} a _ {n} \\ b _ {n} \end{array} \right] = \left[ \begin{array}{c c} e ^ {- j \omega_ {2} t _ {1}} & e ^ {j \omega_ {2} t _ {1}} \\ e ^ {- j \omega_ {2} t _ {1}} & - e ^ {j \omega_ {2} t _ {1}} \end{array} \right] \cdot \left[ \begin{array}{c} c _ {n} \\ d _ {n} \end{array} \right].\tag{32}
$$

Combining Eqs. (31) and (32), we obtain the fields at two consecutive periods (temporal unit cells) to be connected as

$$
\left[ \begin{array}{c} a _ {n - 1} \\ b _ {n - 1} \end{array} \right] = \left[ \begin{array}{c c} A & B \\ C & D \end{array} \right] \cdot \left[ \begin{array}{c} a _ {n} \\ b _ {n} \end{array} \right],\tag{33}
$$

where the matrix elements are given by

$$
A = e ^ {- j \omega_ {1} t _ {1}} \left[ \cos \omega_ {2} t _ {2} - \frac {j}{2} \left(\frac {\omega_ {2}}{\omega_ {1}} + \frac {\omega_ {1}}{\omega_ {2}}\right) \sin \omega_ {2} t _ {2} \right],\tag{34a}
$$

$$
B = - \frac {j}{2} e ^ {j \omega_ {1} t _ {1}} \left(\frac {\omega_ {2}}{\omega_ {1}} - \frac {\omega_ {1}}{\omega_ {2}}\right) \sin \omega_ {2} t _ {2},\tag{34b}
$$

$$
C = B ^ {*} = \frac {j}{2} e ^ {- j \omega_ {1} t _ {1}} \left(\frac {\omega_ {2}}{\omega_ {1}} - \frac {\omega_ {1}}{\omega_ {2}}\right) \sin \omega_ {2} t _ {2},\tag{34c}
$$

$$
D = A ^ {*} = e ^ {j \omega_ {1} t _ {1}} \left[ \cos \omega_ {2} t _ {2} + \frac {j}{2} \left(\frac {\omega_ {2}}{\omega_ {1}} + \frac {\omega_ {1}}{\omega_ {2}}\right) \sin \omega_ {2} t _ {2} \right].\tag{34d}
$$

One can see that Eq. (34) and Eq. (33) possess the exact dual resemblance to those derived for the spatial case in Ref. [101]. The duality is apparent under the replacement of $\omega \leftrightarrow k$ and $t \leftrightarrow z$ . Before we continue, we draw the reader's attention to one point. From Eq. (34), we see that if $\omega_2 t_2$ is identical with $p\pi$ ( $p = \pm 1, \pm 2, \ldots$ ), the matrix elements $B$ and $C$ vanish, and the elements $A$ and $D$ are described by $e^{\pm j\omega_1 t_1} \cos(p\pi)$ . In other words, the ABCD-matrix is diagonalized with elements whose absolute value is unity. Hence, it adds only an additional phase to the wave propagating between the corresponding time slab.

Thus, the ABCD-matrix connects the amplitudes of the fields at two neighboring temporal unit cells of the PTC. The ABCD-matrix can be expressed explicitly once the permittivities in the two temporal segments $\varepsilon_{1}$ and $\varepsilon_{2}$ and their durations $t_{1}$ and $t_{2}$ are given. Naturally, more complicated unit cells consisting of more than two temporal segments can be considered using this ABCD method. In essence, it would boil down to a multiplication of an increasing number of matrices. Although closed-form expressions are challenging to obtain, a numerical evaluation is particularly easy. Moreover, even continuous profiles can be handled. It would require slicing the temporal profile into very short intervals within which the permittivity can be assumed constant. Independent of these details, we can always obtain a $2 \times 2$ ABCD-matrix that links the amplitudes of the same waves in two consecutive periods.

To obtain the dispersion relation, we must impose that the amplitudes of the waves from one unit cell to the next unit cell are preserved, and the only permissible change is an additional phase accumulation. According to the Floquet theorem, the fields in the neighboring temporal unit cells have a phase difference (same for all the frequency harmonics) of $e^{j(\omega_{\mathrm{F}}+p\omega_{\mathrm{m}})T_{\mathrm{m}}} = e^{j\omega T_{\mathrm{m}}}$ (here, $\omega_{F}$ is the Floquet frequency and p is an arbitrary integer), given as

$$
\left[ \begin{array}{c} a _ {n} \\ b _ {n} \end{array} \right] = e ^ {j \omega_ {\mathrm{F}} T _ {\mathrm{m}}} \left[ \begin{array}{c} a _ {n - 1} \\ b _ {n - 1} \end{array} \right].\tag{35}
$$

Combining Eq. (33) and Eq. (35), we obtain the equation

$$
\underbrace {\left[ \begin{array}{c c} A & B \\ C & D \end{array} \right]} _ {\overline {{\overline {{M}}}} (k)} \cdot \left[ \begin{array}{c} a _ {n} \\ b _ {n} \end{array} \right] = e ^ {- j \omega_ {\mathrm{F}} T _ {\mathrm{m}}} \left[ \begin{array}{c} a _ {n} \\ b _ {n} \end{array} \right].\tag{36}
$$

Note that, in this equation, the matrix elements A, B, C, and D are functions of $\omega_{1}$ and $\omega_{2}$ which, in turn, are functions of momentum k (see Eq. (26)). Therefore, Eq. (36) is an eigenequation relating Floquet frequencies $\omega_{F}$ to wavenumber k. The above eigenequation can be solved by requiring that

$$
\det \left[ \overline {{\overline {{M}}}} (k) - e ^ {- j \omega_ {\mathrm{F}} T _ {\mathrm{m}}} \overline {{\overline {{I}}}} \right] = 0,\tag{37}
$$

where $\overline{\overline{M}}(k)$ is the ABCD-matrix. To satisfy the above equation, we have

$$
e ^ {- j \omega_ {\mathrm{F}} T _ {\mathrm{m}}} = \frac {A + D}{2} \pm \sqrt {\frac {(A + D) ^ {2}}{4} - 1}.\tag{38}
$$

Note that Eq. (38) is derived using the fact that $AD - BC = 1$ , which can be verified by Eqs. (34). Since $D = A^{*}$ , $A + D$ is a real number. The ranges of wavenumber $k$ where $|A + D| < 2$ correspond to eigenmodes with purely real Floquet frequencies (allowed bands):

$$
\omega_ {\mathrm{F}} (k) = \mp \frac {1}{T _ {\mathrm{m}}} \left[ \cos^ {- 1} \left(\frac {A + D}{2}\right) - 2 p \pi \right], \quad p \in \mathbb {Z}.\tag{39}
$$

These eigenmodes are propagating waves.

![](images/20cb4a161efbf78655d500373a1849e39d1ee3fd0d4d61dec4e4ab5b66bfdad1.jpg)
Band structure of PTCs with periodic stepwise modulation where $\varepsilon_{1}=5$ and $\varepsilon_{2}=1$ . For real values of wavenumber k, complex-valued eigenfrequencies are plotted. The momentum bandgaps are shown with shaded regions.

In contrast, for wavenumbers k where $|A + D| > 2$ , we obtain from Eq. (38) that the eigenmodes have complex Floquet frequencies. For wavenumbers k that satisfy $A + D < -2$ ,

$$
\begin{array}{c} \mathfrak {R} [ \omega_ {\mathrm{F}} (k) ] = \frac {\omega_ {\mathrm{m}} (2 p + 1)}{2}, \quad p \in \mathbb {Z} \\ \mathfrak {I} [ \omega_ {\mathrm{F}} (k) ] = \frac {1}{T _ {\mathrm{m}}} \ln \left[ \left(- \frac {A + D}{2} \mp \sqrt {\frac {(A + D) ^ {2}}{4} - 1}\right) \right]. \end{array}\tag{40}
$$

For $A + D > 2$

$$
\begin{array}{c} \mathfrak {R} [ \omega_ {\mathrm{F}} (k) ] = \omega_ {\mathrm{m}} p, \quad p \in \mathbb {Z} \\ \mathfrak {I} [ \omega_ {\mathrm{F}} (k) ] = \frac {1}{T _ {\mathrm{m}}} \ln \left[ \left(\frac {A + D}{2} \pm \sqrt {\frac {(A + D) ^ {2}}{4} - 1}\right) \right]. \end{array}\tag{41}
$$

The $\pm$ signs determine the sign of the imaginary part of the eigenfrequency. These solutions correspond to the forbidden bands (momentum bandgaps) of the PTC. The band edges occur when $|A + D| = 2$ .

To plot the complex solutions of the band structure, we choose $\varepsilon_{1}=5$ and $\varepsilon_{2}=1$ , $t_{1}=t_{2}=T_{m}/2$ . For a given real k, the eigenfrequencies $\omega$ are generally complex. The real and imaginary parts of $\omega$ are calculated using Eqs. (39)–(41) and are shown in Fig. 6. The plot was created using Code 2 attached as Supplementary Material [74]. Unlike sinusoidal modulation, where only one bandgap appears, stepwise modulation generates a series of bandgaps. The next section discusses the physical meaning of the complex eigenfrequency.

Using the transfer matrix method, it is also possible to extract the amplitudes of each harmonic for a given eigenmode, that is, $E_{p,n}(\omega_{\mathrm{F}})$ in Eq. (20). In the plane-wave expansion method, the calculation of the harmonics spectrum is straightforward, as the eigenmode can be obtained directly by finding the eigenvector of Eq. (17). In contrast, when the ABCD-matrix technique is used, calculating the harmonic spectrum becomes not so straightforward, but can be computed by Fourier transforming the eigenfield in one temporal unit cell $T_{m}$ of the PTC. Next, we briefly introduce the method to calculate the eigenmode.

For a specific wavenumber k, by solving the eigenvalue equation (36), the Floquet frequencies $\omega_{F}$ can be obtained. Note that there are multiple discrete solutions of $\omega_{F}$ . Choosing one solution of $\omega_{F}$ with a spacing of $p\omega_{m}$ , we can obtain the corresponding eigenvector $[a_{n}, b_{n}]^{T}$ of Eq. (36), which corresponds to the amplitudes of forward and backward waves in the time section when $\varepsilon(t) = \varepsilon_{1}$ as shown in Fig. 5. Using Eq. (32), the amplitudes of forward and backward waves in the next time segment $[c_{n}, d_{n}]^{T}$ can be calculated. Knowing $a_{n}, b_{n}, c_{n}$ , and $d_{n}$ , the total field in one temporal unit can be obtained from Eqs. (24) and (27). For a purely real $\omega_{F}$ , the total field has a constant amplitude in one temporal unit cell. Therefore, one can directly take the Fourier transform of this time-domain field and obtain the harmonics spectrum corresponding to a chosen $\omega_{F}$ . The calculated spectrum is the same as that determined by solving the eigenvectors from Eq. (17). For a complex $\omega_{F}$ (when k is located inside the bandgap), the total field is exponentially growing or decaying in time with a factor of $e^{\Im(\omega_{F})t}$ . In this case, by normalizing the field by this exponential factor before taking the Fourier transform, we can obtain the harmonics amplitudes of the eigenmode. Therefore, the methods based on transfer matrix and plane-wave expansion provide the same information. Which of the two methods is more suitable depends on the specific temporal modulation profile of $\varepsilon(t)$ .

## 2.2. Electromagnetic Effects Inside the Momentum Bandgap

In the previous section, we discussed the band structure and mentioned the presence of the bandgaps along the momentum axis. In this section, we elaborate on one enticing feature associated with such bandgaps, which results in important consequences, at least from the application perspective. An important question is: What happens to an electromagnetic wave in a PTC whose wavenumber is located inside the momentum bandgap?

To quantitatively understand the wave effects within the momentum bandgap, we examine Fig. 6 in further detail. We select the wavenumber $k$ inside the first momentum bandgap (the first shaded region) shown in the figure. The eigenfrequencies, in this case, are complex-valued. The real parts of the eigenfrequencies remain constant, $\Re(\omega_{\mathrm{F}}) = \frac{(2p + 1)\omega_{\mathrm{m}}}{2}$ , where $p \in \mathbb{Z}$ , whereas the imaginary part delineates an ellipse in the band structure, as shown in Fig. 6. For a given $k$ inside the momentum bandgap, there are both positive and negative solutions for the imaginary part, $\Im(\omega_{\mathrm{F}}) = \pm \omega_{\mathrm{im}}$ with $\omega_{\mathrm{im}} > 0$ , denoted as $S_1$ and $S_2$ , respectively, in the figure. The solution with $\Im(\omega_{\mathrm{F}}) = -\omega_{\mathrm{im}}$ signifies fields that are exponentially growing, with the rate of $e^{+\omega_{\mathrm{im}}t}$ [121]. This is the salient property of PTCs, which allows them to amplify light in time. Remember that the system we describe here is not passive and the external energy supply comes from the mechanism responsible for the temporal modulation of the PTC. Therefore, the energy conservation principle is not violated since we deal with an open and active system. In Section 4.1, we discuss a simple physical picture of how the energy from the temporal modulations is transferred to the signal wave to support its growing amplitude.

The second solution for the eigenfrequencies with $\mathfrak{I}(\omega_{\mathrm{F}})=+\omega_{\mathrm{im}}$ corresponds to waves that are exponentially decaying over time at the rate of $e^{-\omega_{im}t}$ . It is difficult to probe these modes experimentally as they decay rapidly, over the time scale of the wave oscillations. Therefore, inside the momentum bandgap of the PTC, typically only the dominant (growing in time) modes are considered. Nevertheless, under proper conditions, the PTC can be made to operate in a phase-sensitive regime, allowing the excitation of only the decaying mode [122]. This can be accomplished, e.g., by making

## Figure 7

![](images/3b98842341da9359564b92f61704b036917d62df69fdf94168f32b9bf95db4ff.jpg)
(a)

![](images/7cf8aa505448edea4258cb51ee64f2e7203e310c6365ed4153b4e8c7488cb525.jpg)
(b)
(a) Spectrum of harmonics amplitudes of the eigenmode whose k is located in the center of the momentum bandgap and the eigenfrequency $\omega_{F}$ is complex-valued with $\Re(\omega_{\mathrm{F}})=\frac{\omega_{\mathrm{m}}}{2}$ . The harmonics amplitudes are normalized such that $\sum_{n}|E_{n}(\omega_{\mathrm{F}})|^{2}=1$ . The parameters of the periodic stepwise modulation are the same as those mentioned in the caption of Fig. 6. (b) Illustration depicting propagation directions and relative amplitudes of the harmonics shown in (a). Note that although frequencies of different harmonics are different, their wavelengths are the same inside the PTC since they share the same wavenumber.

PTC subwavelength and/or by adding a back reflector to ensure the standing pattern of the signal wave.

As was discussed in the introduction, there is a beautiful duality symmetry between PTCs and conventional photonic crystals (see Fig. 1). Photonic crystals also host two types of eigenmodes within the energy bandgap: one decaying in the positive and one decaying in the negative spatial direction. However, if the photonic crystal is semi-infinite (consider, e.g., the interface between free space and the photonic crystal), only one mode is physical because the other mode corresponds to an exponentially growing field in the direction away from the interface, which is impossible due to energy conservation. In contrast, in PTCs, both eigenmodes are always physical because PTCs are not bound by the energy conservation law.

Now, let us analyze the harmonics' weights $E_{n}(\omega_{\mathrm{F}})$ of the eigenmodes inside the bandgap (see Eq. (11)). Similarly to the analysis performed for plotting Fig. 3(a), we find for each harmonic of a given eigenmode its magnitude and phase. Figure 7(a) illustrates the harmonics amplitude distribution of the eigenmode whose wavenumber is selected at the center of the momentum bandgap $k_{\mathrm{bg}}$ and the eigenfrequency has the real part equal to $\Re (\omega_{\mathrm{F}}) = \frac{\omega_{\mathrm{m}}}{2}$ . The frequencies of all the harmonics $\omega_{n}$ share the same imaginary part $\pm \omega_{\mathrm{im}} = \pm 0.108\omega_{\mathrm{m}}$ . Note that the distribution of $E_{n}(\omega_{\mathrm{F}})$ is the same for decaying and growing eigenmodes. From Fig. 7(a), it is evident that the eigenmode encompasses harmonics with amplitudes symmetrically distributed across the spectrum. It is important to note that this symmetry property always holds when the wavenumber falls inside the momentum bandgap. This is different from the case we analyzed in Fig. 3(c), where the wavenumber was outside the bandgap, and the harmonics amplitude distribution was asymmetric.

Let us consider a pair of the dominant harmonics. The -1st harmonic, i.e., $\Re (\omega_{-1}) = -\omega_{\mathrm{m}} / 2$ , possesses the same amplitude as the fundamental harmonic, i.e., $\Re (\omega_0) = \omega_{\mathrm{m}} / 2$ . Thus, this degenerate pair of dominant harmonics has real frequencies with opposite signs. Since the two harmonics share the same momentum $k_{\mathrm{bg}}$ , their phase velocities are opposite $\nu_{\mathrm{ph}} = \pm \frac{\omega_{\mathrm{m}}}{2k_{\mathrm{bg}}}$ . Therefore, this pair of dominant harmonics represents a standing wave [98] whose amplitude grows or decays exponentially over time due to the nature of the complex eigenfrequency. From Eq. (11) we obtain

$$
E (z, t) = | E _ {0} | e ^ {\omega_ {\mathrm{im}} t} e ^ {- j k _ {\mathrm{bg}} z} \left[ e ^ {j \phi_ {0}} e ^ {j \frac {\omega_ {\mathrm{m}}}{2} t} + e ^ {j \phi_ {- 1}} e ^ {- j \frac {\omega_ {\mathrm{m}}}{2} t} \right],\tag{42}
$$

where we neglected the decaying eigenmode and wrote only the dominant harmonics with n = 0 and n = -1; $\phi_{0}$ and $\phi_{-1}$ are the phases of the complex amplitudes $E_{0}$ and $E_{-1}$ , respectively. The instantaneous real-valued electric field $E_{\mathrm{inst}}(z,t) = \Re[E(z,t)]$ then reads

$$
E _ {\mathrm{inst}} (z, t) = 2 | E _ {0} | e ^ {\omega_ {\mathrm{im}} t} \cos \left[ k _ {\mathrm{bg}} z - \frac {\phi_ {0} + \phi_ {- 1}}{2} \right] \cos \left[ \frac {\omega_ {\mathrm{m}}}{2} t + \frac {\phi_ {0} - \phi_ {- 1}}{2} \right].\tag{43}
$$

The standing wave pattern of the instantaneous field in Eq. (43) is apparent as the spatial and temporal variations of the electric field are decoupled. The nodes of the electric field are separated from one another by the integer of $\pi/k_{bg}$ . Nevertheless, the wave given by Eq. (43) is not a conventional standing wave since its amplitude $2|E_{0}|e^{\omega_{im}t}$ is quickly growing in time.

For other higher-order harmonic pairs with the same wavenumber, such as $\pm\frac{3\omega_{m}}{2}$ , the behavior is analogous: they form a standing wave pattern with smaller amplitudes (see Fig. 7(a)) that grow exponentially at the same rate $e^{\omega_{im}t}$ as the fundamental pair. As the harmonic order increases, their amplitudes diminish, and their contributions are negligible.

Finally, it is important to discuss the shape of the imaginary part of eigenfrequency inside the bandgap (as shown in Fig. 7(a)). For PTCs consisting of lossless material, inside the momentum bandgap, the imaginary part of eigenfrequency forms an ellipse which is symmetrically located with respect to the wavenumber axis ( $\omega_{F} = 0$ ). When considering PTCs whose material includes dissipation, the imaginary-frequency ellipse will be shifted upward in the considered $e^{j\omega t}$ convention. Thus, for a sufficiently high amount of loss for a given modulation strength, all the solutions of the imaginary eigenfrequency become positive. Such a regime could be useful for enhancing light absorption [122]. Furthermore, the shape of the imaginary eigenfrequency contours inside the momentum bandgap is not always elliptical and can be engineered to some degree, providing interesting possibilities for predefined direction-dependent amplification of plane waves [69].

## 3. ASPECTS OF REALISTIC PTCs

In the previous section, we introduced the fundamentals of PTCs and analyzed their eigenmodes while assuming the most basic scenario, i.e., that of a dispersionless and linear material. Moreover, the PTC was infinitely extended in space and time, and the crystal lattice was perfectly periodic in time.

These idealistic assumptions were highly beneficial to getting a glimpse into the fundamental properties of PTCs and obtaining closed-form analytical expressions for some defining quantities. In particular, the dispersion relation could be explicitly written out, and properties such as the growth or decay rate of the amplitudes of the eigenmodes within the momentum bandgap could be obtained. However, many more features characterize realistic systems. We need to consider them for a reliable prediction, also in the light of possible experimental observations. Therefore, the following section is written with the purpose of considering an increasing number of aspects that cause a deviation of the idealistic toward a more realistic description of light propagation in PTCs. We admit at this point that the more realistic description naturally leads to a more complex description. This gives rise to further aspects at the technical and scientific level.

We start by elaborating on the effect of temporal dispersion, continuing with the consideration of spatially finite PTCs, and, finally, considering temporally finite PTCs. These should be the most important aspects we must accommodate in a more realistic setting. However, there were other assumptions that do not necessarily apply and which give rise to interesting effects. For example, the intrinsic material might be characterized by an anisotropy, whereas we have assumed isotropic materials so far. Furthermore, the time variation might deviate from a perfect periodic one, allowing us to introduce temporal defects. These defects are discussed in a dedicated section, leading straight to the question of the general effect of the disorder on the properties of the considered PTCs. Finally, we also discuss a deviation from a linear response, and we consider nonlinear PTCs at the end of this section.

## 3.1. Effects of Temporal Dispersion

In the preceding section, we derived the wave equation and determined the band structure for a PTC. That derivation of the dispersion relation has been based on the assumption of an instantaneous response in time that leads to a vanishing dispersion of the materials in the frequency domain. The materials were characterized by a nondispersive susceptibility or permittivity, i.e., it has not been a function of the frequency. This assumption is generally nonphysical and only approximately valid for systems exhibiting minimal temporal variations and/or insignificant temporal dispersion. An example includes materials such as lithium niobate and silicon in their transparency frequency regions $[123,124]$ . To operate in the transparency region, all the relevant resonances that cause dispersion in the material must be located at frequencies far away from the operational frequency considered for the PTC. These resonances are usually electronic transitions in the material in the ultraviolet range or vibrational excitations in the infrared.

However, the modulation depth of the material parameters in these dispersionless materials is typically very low, ranging from $10^{-4}$ to $10^{-3}$ [125]. That small modulation depth is detrimental to observing notable momentum bandgaps and the possible amplification of eigenmodes. There are two ways to overcome this limitation. First, we need to operate close to a resonance frequency of the material. An electron plasma is a typical example of a material where this is possible. In passing, we note that the plasma is a particular case where the resonance frequency is zero as there is no restoring force, but the argument applies as well. In perspective, operating close to a resonance frequency permits a substantial modification of the material properties. However, it also comes along with dispersion and absorption.

Second, we can operate in a regime where even a tiny change in the material properties might have a great effect on how light propagates in such a material. For example, operating in the ENZ regime would be one of the options, where a small absolute change in the permittivity tremendously affects the light propagation. Such operation is possible at visible or near-infrared frequencies in TCOs $[126]$ . Yet, these materials exhibit strong dispersion at relevant frequencies, particularly in the ENZ region where the modulation depth is significant. Consequently, incorporating temporal dispersion (that should be considered from the equation of motion $[92,127,128]$ ) is crucial in analyzing PTCs $[92,129–133]$ . This section aims to consider the properties of PTCs in the presence of material dispersion for a more realistic description.

We start by deriving the Helmholtz equation in the frequency domain under the assumption of a dispersive medium. Then, in two dedicated subsections, we consider two specific examples of periodic temporal modulation in a dispersive PTC: a change in the plasma frequency and a change in the resonance frequency. It should be noted that the plasma frequency corresponds to a change in the number density and/or effective mass of the particles that constitute the material.

## 3.1a. Helmholtz Equation in a PTC Made From Dispersive Media

In general, in the absence of a time modulation, material properties can be discussed on phenomenological grounds either using a Drude model (applicable to free electrons in a metal or a plasma) or using a Lorentz model (relevant to bound resonances such as electronic or vibrational resonances in materials made from atoms or molecules). Of course, artificial materials, i.e., metamaterials, can also be considered, where artificially structured meta-atoms cause the dispersion. The dispersion of arbitrary materials can always be written as a superposition of a Drude term and a finite number of Lorentzian oscillators. Therefore, the material dispersion up to an arbitrary degree of precision can be expressed with these two different models. Consequently, even though applied and discussed here for some specific types of dispersions, the approach in this subsection can be used for all kinds of materials. Moreover, even though we consider only the case where a few selected parameters of the models are time-dependent, similar derivation and considerations can be done for all kinds of parameters.

To derive general expressions while considering a given constitutive relation in the frequency domain, we start from Maxwell's equations that have been Fourier-transformed regarding both time variables involved in the response function to obtain the Helmholtz equation. The Maxwell equations in the frequency domain read as [134]

$$
\nabla \times \tilde {\mathbf {E}} (\mathbf {r}, \omega) = - j \omega \mu_ {0} \tilde {\mathbf {H}} (\mathbf {r}, \omega),\tag{44}
$$

$$
\nabla \times \tilde {\mathbf {H}} (\mathbf {r}, \omega) = j \omega \tilde {\mathbf {D}} (\mathbf {r}, \omega),\tag{45}
$$

$$
\nabla \cdot \tilde {\mathbf {D}} (\mathbf {r}, \omega) = 0,\tag{46}
$$

$$
\nabla \cdot \tilde {\mathbf {B}} (\mathbf {r}, \omega) = 0\tag{47}
$$

that is supplemented by the constitutive relation in Eq. (9). As a reminder, it reads as

$$
\tilde {\mathbf {D}} (\mathbf {r}, \omega) = \frac {\varepsilon_ {0}}{2 \pi} \int_ {- \infty} ^ {+ \infty} \varepsilon (\mathbf {r}, \omega^ {\prime}, \omega - \omega^ {\prime}) \tilde {\mathbf {E}} (\mathbf {r}, \omega^ {\prime}) d \omega^ {\prime}.\tag{48}
$$

To derive the Helmholtz equation, we apply the curl operator to both sides of the Faraday law, i.e., the first of the four Maxwell equations, i.e., Eq. (44). Using the vector algebra relation $\nabla \times \nabla \times \mathbf{V}(\mathbf{r}) = \nabla (\nabla \cdot \mathbf{V}(\mathbf{r})) - \nabla^{2}\mathbf{V}(\mathbf{r})$ , in which $\mathbf{V}(\mathbf{r})$ is an arbitrary differentiable vector field in space, we obtain $\nabla (\nabla \cdot \tilde{\mathbf{E}}(\mathbf{r}, \omega)) - \nabla^{2}\tilde{\mathbf{E}}(\mathbf{r}, \omega) = -j\omega\mu_{0}\nabla \times \tilde{\mathbf{H}}(\mathbf{r}, \omega)$ .

On the other hand, the curl of $\tilde{\mathbf{H}}(\mathbf{r},\omega)$ is related to the electric flux density through the Ampère–Maxwell law, i.e., the second of the Maxwell equations, i.e., Eq. (45). By plugging the Ampère–Maxwell law and the constitutive relation into the modified Faraday law, we deduce that

$$
\begin{array}{l} \nabla \Big (\nabla \cdot \tilde {\mathbf {E}} (\mathbf {r}, \omega) \Big) - \nabla^ {2} \tilde {\mathbf {E}} (\mathbf {r}, \omega) \\ \qquad - \frac {\omega^ {2} \mu_ {0} \varepsilon_ {0}}{2 \pi} \int_ {- \infty} ^ {+ \infty} \varepsilon (\mathbf {r}, \omega^ {\prime}, \omega - \omega^ {\prime}) \tilde {\mathbf {E}} (\mathbf {r}, \omega^ {\prime}) d \omega^ {\prime} = 0. \end{array}\tag{49}
$$

However, this is not yet the final expression. In addition, Gauss law, i.e., the third of the Maxwell equations, i.e. Eq. (46), states that the divergence of the electric flux density $\tilde{\mathbf{D}}(\mathbf{r},\omega)$ should be zero. Let us recall that $\nabla \cdot (f(\mathbf{r})\mathbf{V}(\mathbf{r})) = \nabla f(\mathbf{r}) \cdot \mathbf{V}(\mathbf{r}) + f(\mathbf{r})\nabla \cdot \mathbf{V}(\mathbf{r})$ .

Here, $f(\mathbf{r})$ is an arbitrary differentiable scalar function. By using this algebraic rule, we find the Gauss law in the form

$$
\int_ {- \infty} ^ {+ \infty} \varepsilon (\mathbf {r}, \omega^ {\prime}, \omega - \omega^ {\prime}) \nabla \cdot \tilde {\mathbf {E}} (\mathbf {r}, \omega^ {\prime}) d \omega^ {\prime} = - \int_ {- \infty} ^ {+ \infty} \nabla \varepsilon (\mathbf {r}, \omega^ {\prime}, \omega - \omega^ {\prime}) \cdot \tilde {\mathbf {E}} (\mathbf {r}, \omega^ {\prime}) d \omega^ {\prime}.\tag{50}
$$

If we wish to discuss again the eigenmodes of a PTC, but now in the presence of dispersion, Eqs. (49) and (50) must be solved simultaneously.

As our purpose is to calculate the eigenmodes of the Maxwell equations, we shall assume from now on that the space is filled homogeneously with the same material. The permittivity will not depend on the spatial coordinate, i.e., $\varepsilon(\mathbf{r},\omega',\omega-\omega')=\varepsilon(\omega',\omega-\omega')$ , and the gradient of the permittivity vanishes. Therefore, we have

$$
\int_ {- \infty} ^ {+ \infty} \varepsilon (\omega^ {\prime}, \omega - \omega^ {\prime}) \nabla \cdot \tilde {\mathbf {E}} (\mathbf {r}, \omega^ {\prime}) \mathrm{d} \omega^ {\prime} = 0.\tag{51}
$$

At this point, we use the separation of variables method $[134]$ , assuming that the electric field is written as the product of two functions: $\tilde{\mathbf{E}}(\mathbf{r},\omega)=\mathbf{R}(\mathbf{r})G(\omega)$ . One of them will only depend on the spatial coordinate. The other will only depend on frequency. As we show later, this assumption is valid, and the general solution for the field is a superposition of all possible eigenfunctions written as the product of the two mentioned functions.

If we substitute this ansatz into Eq. (51), we observe that the term $\nabla \cdot \mathbf{R}(\mathbf{r})$ can be taken out of the integral, and two possibilities arise that guarantee that the equation continues to apply. The first possibility is that $\int_{-\infty}^{+\infty} \varepsilon(\omega', \omega - \omega') G(\omega') \, \mathrm{d}\omega' = 0$ . However, this possibility cannot be realized, as we see from Eq. (49). The second possibility requires that $\nabla \cdot \mathbf{R}(\mathbf{r}) = 0$ meaning that $\nabla \cdot \tilde{\mathbf{E}}(\mathbf{r}, \omega) = 0$ . Accordingly, Eq. (49) is simplified, and we finally infer that

$$
\nabla^ {2} \tilde {\mathbf {E}} (\mathbf {r}, \omega) + \frac {\omega^ {2} \mu_ {0} \varepsilon_ {0}}{2 \pi} \int_ {- \infty} ^ {+ \infty} \varepsilon (\omega^ {\prime}, \omega - \omega^ {\prime}) \tilde {\mathbf {E}} (\mathbf {r}, \omega^ {\prime}) \mathrm{d} \omega^ {\prime} = 0.\tag{52}
$$

This is the Helmholtz equation for dispersive time-varying media.

As a quick sanity check, for the conventional time-invariant case, the temporal complex permittivity corresponds to $\varepsilon_{\mathrm{T}}(\omega', t) = \varepsilon(\omega')$ . Thus, the Fourier transform gives rise to $\varepsilon(\omega', \omega) = 2\pi\varepsilon(\omega')\delta(\omega)$ . Substituting this relation into Eq. (52) and knowing that $\int_{-\infty}^{+\infty} f(\omega')\delta(\omega - \omega') \, \mathrm{d}\omega' = f(\omega)$ , we obtain $\nabla^{2}\tilde{\mathbf{E}}(\mathbf{r}, \omega) + k^{2}\tilde{\mathbf{E}}(\mathbf{r}, \omega) = 0$ ( $k^{2} = \omega^{2}\mu_{0}\varepsilon_{0}\varepsilon(\omega)$ ), which is the classical textbook result [89].

Equation (52) is general and accurate for any linear time-varying causal medium. However, in the following, we concentrate on the particular scenario of periodic temporal modulation applicable to PTCs. Hence, we can use the Floquet theorem given by Eq. (11) and introduce an alternative version for Eq. (52). In fact, if we take the expression given for the electric field and apply it to Eq. (7), the electric flux density is simplified to

$$
\mathbf {D} (\mathbf {r}, t) = \sum_ {n} \varepsilon_ {0} \varepsilon_ {\mathrm{T}} (\omega_ {n}, t) E _ {n} e ^ {j \omega_ {n} t} e ^ {- j k z} \mathbf {a} _ {x}.\tag{53}
$$

This equation expresses that the amplitude of the electric flux density for each harmonic $D_{n}$ is time-dependent ( $D_{n} = \varepsilon_{0}\varepsilon_{\mathrm{T}}(\omega_{n}, t)E_{n}$ ). As is clear, this is because the temporal complex relative permittivity $\varepsilon_{\mathrm{T}}(\omega_{n}, t)$ is a function of time. By having Eq. (53) and employing time-domain Maxwell's equations, one can deduce the desired equation as

$$
\nabla^ {2} \mathbf {E} (\mathbf {r}, t) - \frac {1}{c ^ {2}} \frac {\partial^ {2}}{\partial t ^ {2}} \sum_ {n} \varepsilon_ {\mathrm{T}} (\omega_ {n}, t) E _ {n} e ^ {j \omega_ {n} t} e ^ {- j k z} \mathbf {a} _ {x} = 0.\tag{54}
$$

Indeed, regarding periodic modulation, Eq. (54) is the time-domain version of Eq. (52), which was illustrated fully in the frequency domain. Both equations are valid, and we can use one of them depending on the nature of the problem under study. For example, next, we study the problem when the plasma frequency periodically changes in time, and we show that Eq. (54) is quite convenient for calculating the corresponding band structure. On the other hand, in the subsection that follows, we discuss a periodically modulated resonance frequency, in which we employ Eq. (52).

## 3.1b. Time-Varying Plasma Frequency

Here, we consider a PTC made of a dispersive bulk material whose plasma frequency depends on time and varies continuously. The temporal discontinuity in such media in which the plasma frequency rapidly transits once from one value to another has been studied carefully in different works such as Ref. [135]. It should be noted that changing the plasma frequency can be accomplished by changing the particle density in the system or their effective mass. As discussed previously, we can distinguish a Drude model that applies to the description of free electrons. In this case, the plasma frequency corresponds to the density of free-charge carriers, such as the number of electrons per unit volume [62–65,136,137].

Alternatively, we can consider a Lorentz model that applies to the description of bound resonances. The permittivity of the Lorentz model in the static case reads as

$$
\varepsilon (\omega) = 1 + \chi (\omega) = 1 + \frac {\omega_ {\mathrm{p0}} ^ {2}}{\omega_ {\mathrm{r0}} ^ {2} - \omega^ {2} + j \gamma \omega},\tag{55}
$$

where $\omega_{r0}$ is the natural or resonance frequency of the material, and $\gamma$ represents a phenomenological damping constant. Moreover, the plasma frequency is defined as $\omega_{\mathrm{p0}}^{2}=Ne^{2}/(\varepsilon_{0}m_{\mathrm{e}})$ with N being the number of electrons per unit volume, e the electron charge, and $m_{e}$ the electron mass. Please note that when we set $\omega_{r0}=0$ , we restore the phenomenological dielectric function of the Drude model. This reflects that there is no restoring force on the electrons as opposed to the bound resonance.

Instead of a natural material, we can also think of an artificial material, i.e., a metamaterial made from structured unit cells. Then, we can consider a metamaterial made from N meta-atoms per unit volume, and the polarizability of the metamaterial at the effective level is described by one of the discussed phenomenological models.

If only the number of electrons, the number of polarizable atoms, or the number of meta-atoms per unit volume changes in time (i.e., $N_{e}(t)$ ), the interaction of the individual electrons/atoms/meta-atoms with the electric and magnetic fields is the same as the case of a static medium. Mathematically, we can state that the response function of the electron/atom/meta-atom and the corresponding polarizability are unaffected. By assuming that the electrons/atoms/meta-atoms interact weakly with each other, the time-dependent complex susceptibility is expressed as a product of a time-varying function, describing the change of density in time and the Lorentzian dispersion.

In this scenario, the frequency dispersion due to the temporal nonlocality is not affected, which means that in Eq. (55), we substitute $\omega_{p0}^{2}$ by a time-varying function $\omega_{\mathrm{p}}^{2}(t)$ , and, consequently, we write $\varepsilon_{\mathrm{T}}(\omega', t) = 1 + \omega_{\mathrm{p}}^{2}(t)/(\omega_{\mathrm{r0}}^{2} - \omega'^{2} + j\gamma\omega')$ (a detailed discussion is given in Ref. [92]). Note that for a periodic temporal modulation, $\omega_{\mathrm{p}}^{2}(t)$ is a summation of the constant $\omega_{p0}^{2}$ and a temporal function which fluctuates around zero. Conventionally, this fluctuation is in the form of a sinusoidal function, and the temporal complex relative permittivity is expressed as $\varepsilon_{\mathrm{T}}(\omega', t) = 1 + \chi(\omega') [1 + m_{\mathrm{p}} \cos(\omega_{\mathrm{m}} t)]$ . Here, $m_{p}$ is the relative modulation strength. The reason to modulate the plasma frequency “sinusoidally” is only for simplicity. In general, as before, one can work with any arbitrary periodic function and employ the theory of Fourier series (see Eq. (10)). Accordingly, the expression inside the above bracket should be revised so that we have

$$
\varepsilon_ {\mathrm{T}} (\omega^ {\prime}, t) = 1 + \chi^ {\prime} (\omega^ {\prime}) \sum_ {m} f _ {m} \mathrm{e} ^ {j m \omega_ {\mathrm{m}} t},\tag{56}
$$

where $\chi'(\omega') = \chi(\omega')/\omega_{\mathrm{p0}}^{2}$ and $\omega_{\mathrm{p}}^{2}(t) = \sum_{m} f_{m} \mathrm{e}^{jm \omega_{m} t}$ ( $f_{m}$ denotes the Fourier coefficients). Here, let us continue with this general form since our goal is to elucidate the effect of dispersion by obtaining an expression similar to Eq. (16).

By substituting the expression for $\varepsilon_{\mathrm{T}}(\omega', t)$ written in Eq. (56) into Eq. (54), we find the following equation that connects the wavenumber k to the corresponding Floquet frequency $\omega_{F}$ :

$$
\sum_ {n} \sum_ {m} \frac {(\omega_ {\mathrm{F}} + n \omega_ {\mathrm{m}}) ^ {2}}{c ^ {2}} \chi^ {\prime} (\omega_ {m}) f _ {n - m} E _ {m} e ^ {j \omega_ {n} t} = \sum_ {n} \left(k ^ {2} - \frac {\omega_ {n} ^ {2}}{c ^ {2}}\right) E _ {n} e ^ {j \omega_ {n} t}.\tag{57}
$$

To solve this equation, we must ensure that the coefficients corresponding to each harmonic frequency $\omega_{n}$ are equal on both the right and left sides of the equation. This requirement leads to simplifying the above equation and reducing it to

$$
\sum_ {m} \frac {(\omega_ {\mathrm{F}} + n \omega_ {\mathrm{m}}) ^ {2}}{c ^ {2}} \chi^ {\prime} (\omega_ {m}) f _ {n - m} E _ {m} - \left(k ^ {2} - \frac {\omega_ {n} ^ {2}}{c ^ {2}}\right) E _ {n} \delta_ {n m} = 0.\tag{58}
$$

If we compare Eq. (58) with Eq. (16), we observe that both are similar. The key difference is that in Eq. (58), the field amplitudes $E_{m}$ are now multiplied by an additional factor $\chi'(\omega_{m})$ . That factor is a constant value in the case of nondispersive time-varying materials that we studied in the previous section. Indeed, in the expression above, if $\chi(\omega_{m}) = \omega_{\mathrm{p0}}^{2}/\omega_{\mathrm{r0}}^{2}$ , which results in $\chi'(\omega_{m}) = 1/\omega_{\mathrm{r0}}^{2}$ , we obtain exactly Eq. (16). As we did before for Eq. (16), the left-hand side of Eq. (58) can also be represented as the multiplication of a square matrix and field-amplitude vector, which is a one-column matrix (see Eq. (17)). To determine the band structure, again, the determinant of such a square matrix must be zero.

We have shown the band structure in Fig. 8 to see the dispersion effect. Accordingly, the plasma frequency (i.e., more precisely, the number of polarizable atoms) is modulated in the case of nondispersive (blue curve) and dispersive (orange curve) dielectric media. We observe that in the dispersive case, the size of the bandgap is considerably larger which is a favorable feature for amplification of waves. In addition, for the same Floquet frequency, there is a shift in the bandgap, which is due to the dispersion of the medium. The band structure of a dispersive Lorentz media with stepwise periodically modulated plasmon frequency can be calculated using the transfer matrix method $[138]$ .

## 3.1c. Time-Varying Resonance Frequency

Next, we elucidate how to calculate the band structure of a Lorentzian dielectric material described previously with a time-varying resonance frequency (here, the damping coefficient and the plasma frequency are assumed time-invariant). This scenario is significantly different from that in the previous subsection. Whereas the change of the plasma frequency requires the change of the density of the free electrons or that of the polarizable entities, the change in the resonance frequency preserves the number of particles but changes the properties of the individual oscillators. That can be implemented by various means. On the one hand, the intrinsic resonance frequency of a bulk material can be temporally modulated, for instance, by applying a strong dynamic electric bias $[139]$ . However, this method can be practically challenging. On the other hand, an effective resonance frequency can be modulated using spatially structured meta-atoms, as discussed in Ref. $[69]$ . These meta-atoms can be as simple as dielectric spheres or more advanced unit cells.

![](images/6d254d7b080b1954a3e888aee8edf78779b33ed27c22ee8902c56220b5435a33.jpg)
Band structures for a dispersive (orange curve) and a nondispersive (blue curve) PTC. The dispersion is given by the Lorentz model (assuming that the damping coefficient is zero), and the temporal complex relative permittivity is described by $\varepsilon_{\mathrm{T}}(\omega', t) = 1 + \chi(\omega')(1 + m_{\mathrm{p}} \cos(\omega_{\mathrm{m}} t))$ . The nondispersive relative permittivity is assumed as $\varepsilon(t) = \varepsilon_{\mathrm{av}}(1 + m_{\varepsilon} \cos(\omega_{\mathrm{m}} t))$ where $\varepsilon_{av} = 5$ , and $m_{\varepsilon} = 0.2$ . For nondispersive case, we set $\omega_{p0}/\omega_{r0} = 2$ , which ensures the DC permittivity is same as the $\varepsilon_{av}$ . The modulation frequency is $\omega_{m} = \omega_{r0}$ .

Regarding such an artificial material, if the properties of the individual particle or meta-atom vary, the response function and the associated dipole polarizability are certainly modified, and, accordingly, the temporal complex susceptibility is strikingly revised (see Ref. [92] for a complete discussion about this subject). Therefore, we cannot simply write the Lorentzian dispersion and only change the parameters in time in the corresponding model, as we did in the previous subsection. Instead, we need to explicitly derive the response function and the corresponding polarizability for such a meta-atom. Subsequently, we can calculate the dispersion relation based on the wave equations described previously. This is what we do in this subsection.

In the following, we consider that the resonance frequency is temporally modulated as $\omega_{\mathrm{r}}(t)$ . It is worth mentioning that a similar derivation could be used to obtain the band structure of a PTC with a time-varying damping coefficient. We start with the second-order differential equation that describes the polarization density of the Lorentzian material with the resonance frequency being modulated in time [92]:

$$
\frac {\mathrm{d} ^ {2} \mathbf {P} (t)}{\mathrm{d} t ^ {2}} + \gamma \frac {\mathrm{d} \mathbf {P} (t)}{\mathrm{d} t} + \omega_ {\mathrm{r}} ^ {2} (t) \mathbf {P} (t) = \varepsilon_ {0} \omega_ {\mathrm{p}} ^ {2} \mathbf {E} (t),\tag{59}
$$

in which we suppose that the resonance frequency is sinusoidally modulated as $\omega_{\mathrm{r}}^{2}(t)=\omega_{\mathrm{r}0}^{2}[1+m_{\mathrm{r}}\cos(\omega_{\mathrm{m}}t)]$ . This modulation function is only for simplicity regarding the next steps. Accordingly, by taking the Fourier transform regarding the observation time variable t from both sides, we derive that

$$
(\omega_ {\mathrm{r0}} ^ {2} - \omega^ {2} + j \gamma \omega) \tilde {\mathbf {P}} (\omega) + \frac {m _ {\mathrm{r}} \omega_ {\mathrm{r0}} ^ {2}}{2} \left[ \tilde {\mathbf {P}} (\omega - \omega_ {\mathrm{m}}) + \tilde {\mathbf {P}} (\omega + \omega_ {\mathrm{m}}) \right] = \varepsilon_ {0} \omega_ {\mathrm{p}} ^ {2} \tilde {\mathbf {E}} (\omega).\tag{60}
$$

We see that if there is no modulation (i.e., $m_{r}=0$ ), we obtain the conventional Lorentzian dispersion as $\tilde{\mathbf{P}}(\omega)=\varepsilon_{0}\omega_{\mathrm{p}}^{2}/(\omega_{\mathrm{r}0}^{2}-\omega^{2}+j\gamma\omega)\tilde{\mathbf{E}}(\omega)$ that was already expressed in Eq. (55). Equation (60) connects the polarization density to the electric field in the frequency domain. However, on the other hand, these vectors are also related to each other through the definition of the transfer function in both time and frequency domains. Indeed, based on the definition (see Eq. (9)), we have

$$
\tilde {\mathbf P} (\omega) = \frac {\varepsilon_ {0}}{2 \pi} \int_ {- \infty} ^ {+ \infty} \chi (\omega - \omega^ {\prime}, \omega^ {\prime}) \tilde {\mathbf E} (\omega^ {\prime}) d \omega^ {\prime},\tag{61}
$$

in which $\chi(\omega,\omega')$ is the Fourier transform of the response function. As seen and discussed before, there are two angular frequency variables: one is due to the dispersion property (the delay time between the response and excitation), and the other is due to the temporal modulation of the material, which in this case would be caused by the time-varying resonance frequency.

Equation (61) is general and holds for an arbitrary temporal modulation function. Since we consider a periodic time modulation here, we apply the Floquet theorem. Hence, in the frequency domain $(\omega)$ , the electric field from Eq. (11) is rewritten as

$$
\tilde {\mathbf {E}} (\omega) = 2 \pi \sum_ {\ell} \mathbf {E} _ {\ell} \delta (\omega - \Omega_ {\ell}) e ^ {- j k z},\tag{62}
$$

where $\Omega_{\ell} = \omega_{F} + \ell\omega_{m}$ . By having the electric field as the summation of Dirac delta distributions, which all correspond to one single value of the phase constant k, and by using Eq. (61), we deduce the polarization density in terms of the susceptibility:

$$
\tilde {\mathbf {P}} (\omega) = \varepsilon_ {0} \sum_ {\ell} \mathbf {E} _ {\ell} \chi \left(\omega - \Omega_ {\ell}, \Omega_ {\ell}\right).\tag{63}
$$

Now, it may be clear that our next step is to substitute the above equation into Eq. (60), and in this way, we achieve an equation that provides an expression for the susceptibility. Therefore, we infer that

$$
\begin{array}{r l} & {\left(\omega_ {\mathrm{r0}} ^ {2} - \omega^ {2} + j \gamma \omega\right) \sum_ {\ell} \mathbf {E} _ {\ell} \chi (\omega - \Omega_ {\ell}, \Omega_ {\ell}) + \frac {m _ {\mathrm{r}} \omega_ {\mathrm{r0}} ^ {2}}{2} \sum_ {\ell} \mathbf {E} _ {\ell} \chi (\omega - \Omega_ {\ell} - \omega_ {\mathrm{m}}, \Omega_ {\ell})} \\ & {+ \frac {m _ {\mathrm{r}} \omega_ {\mathrm{r0}} ^ {2}}{2} \sum_ {\ell} \mathbf {E} _ {\ell} \chi (\omega - \Omega_ {\ell} + \omega_ {\mathrm{m}}, \Omega_ {\ell}) = \omega_ {\mathrm{p}} ^ {2} \sum_ {\ell} \mathbf {E} _ {\ell} \delta (\omega - \Omega_ {\ell}).} \end{array}\tag{64}
$$

To solve this equation, we multiply the right- and left-hand sides by a function in the form of a Dirac delta distribution: $\delta (\omega -\Omega_{\ell '})$ in which $\Omega_{\ell^{\prime}} = \omega_{\mathrm{F}} + \ell^{\prime}\omega_{\mathrm{m}}$ , and, subsequently, we integrate over all possible angular frequencies $\omega$ . As a consequence, concerning each integer value for $\ell$ and $\ell^{\prime}$ , we express eventually that

$$
\begin{array}{r l} & {\left(\omega_ {\mathrm{r0}} ^ {2} - \Omega_ {\ell^ {\prime}} ^ {2} + j \gamma \Omega_ {\ell^ {\prime}}\right) \chi (\Omega_ {\ell^ {\prime}} - \Omega_ {\ell}, \Omega_ {\ell}) + \frac {m _ {\mathrm{r}} \omega_ {\mathrm{r0}} ^ {2}}{2} \chi (\Omega_ {\ell^ {\prime}} - \Omega_ {\ell} - \omega_ {\mathrm{m}}, \Omega_ {\ell})} \\ & {+ \frac {m _ {\mathrm{r}} \omega_ {\mathrm{r0}} ^ {2}}{2} \chi (\Omega_ {\ell^ {\prime}} - \Omega_ {\ell} + \omega_ {\mathrm{m}}, \Omega_ {\ell}) = \omega_ {\mathrm{p}} ^ {2} \delta_ {\ell^ {\prime} \ell},} \end{array}\tag{65}
$$

where $\delta_{\ell^{\prime}\ell}$ is the Kronecker delta. As mentioned, the above equation has been written for each integer value $\ell$ . Thus, we can fix this value ( $\ell$ ) and accordingly repeat the above equation enough times $(2N+1$ times) by changing the integer value of $\ell'$ $(-N<\ell'<+N)$ . From this perspective, Eq. (63) indicated that a square matrix $\overline{\chi}$ is formed, which connects the polarization density at each Floquet angular frequency to the electric field. The columns of this matrix correspond to each value of $\ell$ , and the rows correspond to each value of $\ell'$ . Ideally, this matrix is infinite in size (i.e., $N\to\infty$ ). However, we can consider lower and upper limitations for $\ell$ and $\ell'$ . This is because, at each Floquet angular frequency, only a few values are important and play a role if the modulation strength is not strong. In the above, fixing $\ell$ and changing $\ell'$ , in fact, gives one column of the matrix $\overline{\chi}$ . In the next step, we change the integer value $\ell$ and again try to make the other column of the matrix. After performing this analysis for enough values of $\ell$ , the whole square matrix is constructed. Having this matrix, using Eq. (63), knowing that $\tilde{\mathbf{D}}(\omega)=\varepsilon_{0}\tilde{\mathbf{E}}(\omega)+\tilde{\mathbf{P}}(\omega)$ , and, eventually, applying Maxwell's equations, one can deduce the band structure as explained in Section 2.1 of Ref. [134].

## 3.1d. Estimating the Size of Momentum Bandgaps

The size of the momentum bandgap is crucial because it determines the range of light momenta (wavenumber) in an incident pulse that can be amplified in a PTC. One key factor for the size of the momentum bandgap is the relative modulation depth of the material parameter, such as permittivity, $m_{\varepsilon}$ . By increasing the modulation depth, the size of the momentum bandgap can be expanded. However, this is not the only factor. Engineering the material dispersion is an alternative means to widen the bandgap [69], being especially important in those parts of the frequency spectrum where reaching a strong material modulation depth is very challenging [61].

In order to find the size of the bandgap, one needs to plot the photonic band diagram. Nevertheless, calculating the band structure of a PTC with arbitrary general material dispersion, although possible, could be a difficult task, requiring a numerical solution of the matrix-type wave Eq. (58). There exists a simple yet powerful approach to estimating the size of the bandgap based on a closed-form analytical solution $[46,140,141]$ . The approach provides a good qualitative and quantitative description of the bands in the vicinity of the momentum bandgap and is based on the assumption of the small modulation amplitude. When the modulation depth is small (e.g., $m_{p} \ll 1$ for material with plasma frequency modulated), the higher-order harmonics excited inside the bandgap of the PTC are very weak and can be neglected, simplifying the description only to two dominant harmonics, 0th and -1st. This assumption simplifies the mathematics significantly and makes it possible to solve the band structure analytically, providing more physical insights into the bandgap formation.

To exemplify such an analysis, we consider in the following a PTC with a sinusoidally modulated plasma frequency with modulation amplitude $m_{p} = 0.2$ . The band structure of this PTC obtained by solving Eq. (58) is plotted in Fig. 9(a) (see blue solid line). The size of the bandgap is given by the distance between the two intersection points of the horizontal line representing the Floquet frequency $\omega_{F} = \omega_{m}/2$ with the photonic bands. Note that here, we have a rather small modulation depth. Therefore, the higher-order harmonics can be neglected. The main harmonics are excited at $\omega_{0} = \omega_{m}/2$ and $\omega_{-1} = -\omega_{m}/2$ , respectively. By considering only these two modes, the matrix in Eq. (58) can be reduced to a $2 \times 2$ matrix resulting in the wave equation in the form

$$
\left[ \begin{array}{c c} [ 1 + \chi (\omega_ {\mathrm{F}} - \omega_ {\mathrm{m}}) ] (\omega_ {\mathrm{F}} - \omega_ {\mathrm{m}}) ^ {2} - k ^ {2} c ^ {2} & m _ {\mathrm{p}} \chi (\omega_ {\mathrm{F}}) (\omega_ {\mathrm{F}} - \omega_ {\mathrm{m}}) ^ {2} / 2 \\ m _ {\mathrm{p}} \chi (\omega_ {\mathrm{F}} - \omega_ {\mathrm{m}}) \omega_ {\mathrm{F}} ^ {2} / 2 & [ 1 + \chi (\omega_ {\mathrm{F}}) ] \omega_ {\mathrm{F}} ^ {2} - k ^ {2} c ^ {2} \end{array} \right] \cdot \left[ \begin{array}{c} E _ {- 1} \\ E _ {0} \end{array} \right] = 0.\tag{66}
$$

The band structure predicted by Eq. (66) is displayed in Fig. 9(a) by red circles. It is seen that the two band structures calculated using Eq. (58) and Eq. (66) almost overlap, showing the high accuracy of the method based on the weak-modulation approximation. From the eigenvalue problem (66), we can obtain the bandgap size. This can be done by substituting $\omega_{F} = \omega_{m}/2$ and requiring that the determinant of the matrix vanishes. Then, we can obtain two positive solutions of k,

![](images/c1f270c3a6ab785e932c5b688ec9f25cbf74400a0634e1a6f7ef3b906fb0706a.jpg)

![](images/9184e5af704d335a508cae20288c27892802413f239e00e6db274656687f9672.jpg)
Comparison of the band structures of a dispersive PTC in the proximity of the momentum bandgap calculated using rigorous Eq. (58) (the truncated matrix size in the calculation is $9 \times 9$ ) and using Eq. (66) based on the weak-modulation approximation (the matrix has $2 \times 2$ size). The modulation strength is equal to (a) $m_{p} = 0.2$ and (b) $m_{p} = 0.9$ . The permittivity modulation function is $\hat{\varepsilon}_{\mathrm{T}}(\omega, t) = 1 + \omega_{\mathrm{p}}^{2}(t)/(\omega_{\mathrm{r0}}^{2} - \omega^{2} + j\gamma\omega)$ , where $\omega_{\mathrm{p}}^{2}(t) = \omega_{\mathrm{p0}}^{2}[1 + m_{\mathrm{p}} \cos(\omega_{\mathrm{m}}t)]$ , $\omega_{p0} = 3.5\omega_{r0}$ , $\omega_{m} = 0.2\omega_{r0}$ , and $\gamma = 0$ .

$$
k _ {\pm} = \frac {\omega_ {\mathrm{m}}}{2 c} \sqrt {1 + \chi (\omega_ {\mathrm{m}} / 2) [ 1 \pm m _ {\mathrm{p}} / 2 ]},\tag{67}
$$

corresponding to the edges of the dispersion relation that form the bandgap. The bandgap size is then given by $|k_{+}-k_{-}|$ . Interestingly, the described approach for estimating the bandgap size can provide a reasonably good qualitative description even for strong modulations. As one can see from Fig. 9(b), even when $m_{p}=0.9$ , the approximate approach predicts a shifted but qualitatively similar band structure near the bandgap. The bandgap size has a small deviation compared with that calculated using the exact solution.

As one can see from (67), the two edges of the momentum bandgap, $k_{+}$ and $k_{-}$ , correspond to the wave vectors of two stationary (time-invariant) media calculated at $\omega = \omega_{m}/2$ . One medium is characterized by the susceptibility $\chi(\omega)(1 + m_{\mathrm{p}}/2)$ , whereas the other is characterized by $\chi(\omega)(1 - m_{\mathrm{p}}/2)$ [69]. This simple but powerful observation points us to the fact that the bandgap size can be predicted even without solving Eq. (66). Instead, it can be done by simply plotting the dispersion relations of two auxiliary time-invariant media.

The above-mentioned rule is not limited to PTCs where the time-varying parameter is the plasma frequency [69]. Generally speaking, it can be used for PTCs with other time-varying physical quantity $q(t) = q_0(1 + m_q\cos (\omega_{\mathrm{m}}t))$ , where $m_{q}$ is the modulation depth and $q$ could be permittivity, permeability, etc. (see Fig. 10(a)). To apply the rule, we first plot the dispersion curves of the two time-invariant auxiliary materials described by $q_{1} = (1 + \frac{m_{q}}{2})q_{0}$ and $q_{2} = (1 - \frac{m_{q}}{2})q_{0}$ . Their conceptual dispersion relations $k_{1}(\omega)$ and $k_{2}(\omega)$ are sketched in Fig. 10(b). Next, we draw a horizontal line corresponding to $\omega = \omega_{m}/2$ and find the two points where it intersects with the dispersion curves of the two auxiliary materials. The intersection points with the curves are $[k_{1}(\omega_{\mathrm{m}}/2), \omega_{\mathrm{m}}/2]$ and $[k_{2}(\omega_{\mathrm{m}}/2), \omega_{\mathrm{m}}/2]$ . The horizontal coordinates of these two points precisely define the edges of the momentum bandgap for a PTC modulated according to $q(t)$ . Thus, the bandgap width equals to $|k_{2}(\omega_{\mathrm{m}}/2) - k_{1}(\omega_{\mathrm{m}}/2)|$ .

![](images/97ed613d8032dc7c1f1521f33cecaa1acdcdd8325fc3027c50036a38932ef6f5.jpg)
(a)

![](images/37fa606c62f23b4af6f3cab2165e4a15a8849c489c69b2f7538e24b1e4ca72f0.jpg)
(b)
Simple rule for estimating the size of the momentum bandgap in a PTC. (a) A sinusoidal modulation form is applied to quantity $q(t)$ . (b) Conceptual dispersion relation curves $k_{1}(\omega)$ and $k_{2}(\omega)$ of two auxiliary time-invariant materials with $q(t) = q_{1}$ and $q(t) = q_{2}$ , respectively. The size of the bandgap of the PTC whose modulation function is shown in (a) equals the distance $|k_{2} - k_{1}|$ between two crossing points shown in (b).

This rule is very useful, especially in scenarios where the PTC is made of a material with complicated frequency dispersion. Since it is visual and does not imply determining the band structure of the PTC, it can be used not only to estimate the bandgap size of a given PTC (analysis purpose) but also to predict what material dispersion one would need to obtain the bandgap with desired characteristics (synthesis purpose). For example, in Ref. [69], it was found that a highly dispersive material, e.g., a resonant metamaterial, can drastically enhance the size of the momentum bandgap for a given modulation depth.

Altogether, this discussion finalizes the consideration of material dispersion in the context of PTCs. It remains to emphasize that material dispersion can always be considered in the analysis by suitable modifications of the theoretical framework. Material dispersion is particularly important when describing realistic materials, so when realistic predictions on observable quantities shall be made. However, assuming nondispersive material is also of scientific value. The simplification allows us to obtain a better glimpse into basic effects, which is particularly useful when exploring fundamental phenomena. Therefore, the reader will find in this tutorial and in the general literature discussions that assume a nondispersive but time-varying permittivity. Such assumptions are acceptable provided that the potential limitations are recognized.

After discussing the material dispersion aspect of a realistic PTC, we consider two other aspects that characterize a more realistic system in the following sections. First, we consider spatially finite PTCs. Second, we consider temporally finite PTCs.

## 3.2. Spatially Finite PTCs

While analyzing the eigenmodes of PTCs with infinite spatial extent is fundamentally important for understanding their physics, in real applications, the PTCs always have spatial boundaries. In optical applications, the size of the experimental sample can be tens of wavelengths or even larger; thus, it can be treated as a nearly infinite-sized structure. However, on many other occasions, including those at microwave frequencies, large-size PTCs may not be feasible. Therefore, analyzing how the finite spatial extent affects the properties of PTCs is crucial.

For finite-sized PTCs, spatial boundaries exist between the PTCs and the background medium. Under external excitation, the scattering from the PTCs can be analyzed using a conventional mode-matching method. By listing all the eigenmodes inside and outside the PTCs, and then applying spatial boundary conditions, the amplitudes of all the eigenmodes inside and outside the PTCs can be solved uniquely. This approach enables the calculation of reflection, transmission, and the fields inside the PTCs $[20]$ .

Next, we consider an example to briefly explain the procedures for solving the reflection and transmission from a finite-size PTC. The time-varying slab of length L is shown in Fig. 11(a). The permittivity inside the slab is modulated periodically in time as $\varepsilon(t)$ . Outside the slab, the permittivity is $\varepsilon_{A}$ for the range z<0 and $\varepsilon_{B}$ for z>L. In this case, two spatial boundaries exist at z=0 and z=L. The external excitation is an x-polarized plane wave illuminating the PTC along the z direction. Figure 11(b) shows the band structure of the infinite PTC. An incident plane wave with frequency $\omega_{inc}$ excites inside the PTC multiple eigenmodes with different wavenumber k but the same eigenfrequency $\omega_{inc}$ . Each of these eigenmodes consists of an infinite number of frequency harmonics $\omega_{i}+n\omega_{m}$ denoted as points with the same color in the figure. Thus, all the frequency harmonics marked with points in Fig. 11(b) are excited inside the PTCs, although the external excitation had a unique wavenumber k.

We label each eigenmode with index p according to its wavenumber. The first eigenmode with p = 1 contains n frequency harmonics, with frequencies $\omega_{i} + n\omega_{m}$ . Using Eq. (17), we can solve the amplitude relations among all the harmonics of a given eigenmode $E_{p=1,n}(\omega_{\mathrm{inc}})$ . A similar process applies to other eigenmodes. It should be noted that negative k values, corresponding to backward waves traveling in the -z direction, are also excited but are not shown in the band structure.

Although each eigenmode has a fixed relation of harmonic amplitudes, different eigenmodes can have different amplitude weights. For the pth eigenmode, we denote the weights for forward and backward waves as $C_{p}$ and $D_{p}$ , respectively. The field inside the PTCs slab is a superposition of all the possible harmonics,

$$
\mathbf {E} _ {\mathrm{sl}} (z, t) = \sum_ {p = 1} ^ {\infty} \sum_ {n = - \infty} ^ {\infty} \left[ C _ {p} e ^ {- j k _ {p} (\omega_ {\mathrm{inc}}) z} + D _ {p} e ^ {j k _ {p} (\omega_ {\mathrm{inc}}) z} \right] E _ {p, n} (\omega_ {\mathrm{inc}}) e ^ {j \omega_ {n} t} \mathbf {a} _ {x}.\tag{68}
$$

Outside the time-varying slab, the reflected field $\mathbf{E}_{\mathrm{r}}(z,t)$ and transmitted field $\mathbf{E}_{\mathrm{t}}(z,t)$ consist of infinite frequency harmonics, i.e., $\omega_{inc} + n\omega_{m}$ . These harmonics must respect the dispersion relation of the stationary background media, $k_{n}^{A,B}(\omega_{inc}) = \sqrt{\varepsilon_{A,B}}(\omega_{inc} + n\omega_{m})/c$ . The fields are a superposition of these frequency harmonics, with unknown harmonic amplitudes to be solved later.

After defining the fields inside and outside the PTCs, we can apply two spatial boundary conditions at z = 0 and z = L to compute all the unknown coefficients:

$$
[ \mathbf {E}, \mathbf {H} ] _ {\mathrm{i}} (0, t) + [ \mathbf {E}, \mathbf {H} ] _ {\mathrm{r}} (0, t) = [ \mathbf {E}, \mathbf {H} ] _ {\mathrm{sl}} (0, t),\tag{69a}
$$

$$
[ \mathbf {E}, \mathbf {H} ] _ {\mathrm{sl}} (L, t) = [ \mathbf {E}, \mathbf {H} ] _ {\mathrm{t}} (L, t).\tag{69b}
$$

![](images/f6b98ddce8a1cd80d12ec1a4ae6b5a1e8ea649b9eb9794c29d6e8b1cd9886578.jpg)
(a)

![](images/8f0c1b16683268f2dfdd0a5397a8a0ae17999df3ff3a987d31c9cd2fa00d528e.jpg)
(b)
(a) Dielectric slab characterized by a time-varying permittivity $\varepsilon(t)=\varepsilon_{\mathrm{av}}(1+m_{\varepsilon}\cos(\omega_{\mathrm{m}}t))$ . The slab has a thickness of L and is surrounded by semi-infinite media characterized by $\varepsilon_{A}$ and $\varepsilon_{B}$ , respectively. In the specific example, we consider $\varepsilon_{av}=5$ and $m_{\varepsilon}=0.6$ . (b) Band structure of the time-varying material. The points of the same color denote frequency harmonics of a single eigenmode of the PTC. When the PTC is illuminated at a spatial boundary by a plane wave with frequency $\omega_{inc}$ , harmonics of different frequencies and wavenumbers denoted by the colorful dots in the figure are excited.

This analysis reveals that a slab, when illuminated by a plane harmonic wave $\omega_{i}$ , effectively becomes a polychromatic light source, radiating at frequencies $\omega_{i} + n\omega_{m}$ . We have thus established the general equations necessary to determine the reflection and transmission coefficients for the generated harmonics.

Using the method of eigenmode expansion combined with conditions of continuity of fields at the spatial boundaries, one can solve the scattering or eigenfields problem of PTCs with different shapes. Typical examples include finite planar slabs $[35,45,46,142–144]$ as discussed before, and sphere particles $[140]$ . In the latter scenario, vector spherical harmonics are employed for the spatial component of the wave functions instead of plane waves, as outlined previously. Nevertheless, aside from this distinction, the overall approach remains unchanged.

As discussed earlier, spatially infinite PTCs exhibit momentum bandgaps, with their widths being linearly proportional to the modulation depth when the modulation is weak. This raises the pertinent question of the conditions under which spatially finite PTCs support bandgaps and the factors determining their widths. It was demonstrated in Refs. [46,140] that while parametric amplification always occurs, it is typically finite. The exponential growth, that is, parametric oscillations, only happen at specific conditions for the modulation depth, and radiation and dissipation losses. In particular, for low or moderate modulation depths, finite PTCs exhibit exponential amplification only when $\omega_{\mathrm{m}} / 2$ is close to the resonance frequency $\omega_{\mathrm{r}}$ of some mode inside the same material without temporal modulations. This translates into the requirement that $\Delta \omega = |\omega_{\mathrm{m}} / 2 - \omega_{\mathrm{r}}| \ll \omega_{\mathrm{m}} / 2$ . The resonance mode can be, for example, a Fabry-Perot mode [46] or Mie resonance mode [140]. The qualitative description of this process can be given by the temporal coupled mode theory (see more details in Section 4.3). When a finite PTC with low modulation depth is illuminated by incidence at a frequency close to $\omega_{m}/2$ , one can use the weak-modulation approximation and describe the system by merely two coupled quasi-normal modes. They both oscillate at frequency $\omega_{m}/2$ and are described by temporal envelopes $a_{1}(t)$ and $a_{2}(t)$ . The coupled-mode equations for this case then read [140]

$$
\frac {\mathrm{d}}{\mathrm{d} t} a _ {1} (t) = [ - j \Delta \omega - \gamma_ {\mathrm{tot}} ] a _ {1} (t) - j \eta a _ {2} ^ {*} (t),\tag{70a}
$$

$$
\frac {\mathrm{d}}{\mathrm{d} t} a _ {2} ^ {*} (t) = [ j \Delta \omega - \gamma_ {\mathrm{tot}} ] a _ {2} ^ {*} (t) + j \eta^ {*} a _ {1} (t),\tag{70b}
$$

where $\eta$ is the coupling parameter that is linearly proportional to the modulation depth $m_{\varepsilon}$ , $\gamma_{tot}$ is the total decaying rate caused by possible radiation and/or dissipation loss in the system, and “\*” denotes the complex conjugate operation. The condition of parametric amplification $(a_{1,2}(t)$ grow exponentially) can be solved from Eqs. (70). The condition is strongly related to the modulation depth. The threshold modulation depth that induces the parametric amplification of such finite-sized PTCs is expressed as

$$
m _ {\varepsilon} ^ {\mathrm{thr}} \propto \gamma_ {\mathrm{tot}} + \frac {\Delta \omega^ {2}}{2 \gamma_ {\mathrm{tot}}}.\tag{71}
$$

Equation (71) demonstrates that a higher decay rate of the mode necessitates a greater modulation depth to achieve parametric oscillation. Furthermore, a larger deviation of the half-modulation frequency from the resonance frequency also requires an increased modulation depth.

The above theory can be applied to various different spatially finite PTC geometries. One example is the time-varying dielectric slab discussed in Ref. [35]. It was found in Ref. [35] that if exponentially increasing waves are to be produced, the modulation depth must exceed a certain critical value. This critical value is dependent on the thickness of the slab and on the relation of the permittivities of the slab and the surrounding medium. For the case that the modulation depth is less than the critical value, the transmission and reflection coefficients can be calculated by the mode expansion method as displayed in Eq. (68) combined with interface conditions in Eq. (69).

Another example, a sphere with time-varying charge carrier density $N(t)$ , is shown in Fig. 12(a). Figure 12(b) shows the minimum modulation depth that can provide exponential amplification (parametric oscillations) for different resonant modes of the time-varying dielectric sphere. When the sphere radius R increases, higher-order electric (denoted as $\alpha_{N}$ ) and magnetic (denoted as $\alpha_{M}$ ) modes can provide parametric oscillations. Those higher-order multipolar modes possess higher quality factors (lower $\gamma_{tot}$ ) and, therefore, they have lower thresholds of modulation amplitudes for achieving parametric oscillations.

Note that the exponential field growth indicates an unstable system. In other words, the threshold value of modulation depth in Ref. $[140]$ and slab thickness in Ref. $[35]$ separate stable and nonstable regimes. However, even when we are above the threshold (unstable), the growth is always limited in practice. In particular, we quickly reach the nonlinear regime, and then the system becomes detuned from the optimal conditions. For example, in recent experimental work on metasurface-based PTCs $[145]$ , a finite-sized metasurface was modulated, and parametric oscillations occurred, but finite gain and stable performance were observed.

It should be finally noted that the term “exponentially growing” indicates instability. In this case, calculating the scattering amplitudes using the mode expansion method is nonphysical, as these quantities are defined in the frequency domain, assuming the system is stable. The stability issue is also discussed in Ref. [146].

Figure 12

![](images/89fa254ce0175c23c710d3973c20650dadac304e1db673babcfb9e2c4cb304d6.jpg)
(a)

![](images/a8a56038ceb20c61b394f366fc7ec662665d618b2aad36f085c480b3e128225b.jpg)
(b)
(a) Spherical particle with a time-modulated bulk carrier density $N(t)$ illuminated by incident light. Temporal modulation leads to parametric Mie resonances with simultaneous scattered-field amplification and the possibility of far-field pattern manipulation. Figure 1 reprinted with permission from Asadchy et al., Phys. Rev. Appl. 18, 054065, 2022, Ref. [140]. Copyright (2022) by the American Physical Society. (b) Threshold values of the modulation depth $m_{N}$ that provide parametric oscillations at fixed frequency $\omega_{m}/2$ for different multipolar modes in the time-modulated sphere versus its radius R. It can be seen that the higher-order modes, as sustained at higher radii, require a lower modulation depth to observe parametric oscillations due to their higher quality factors (lower radiation losses). Figure 3 reprinted with permission from Asadchy et al. Phys. Rev. Appl. 18, 054065, 2022 Ref. [140]. Copyright (2022) by the American Physical Society.

## 3.3. Temporally Finite PTCs

By definition, PTCs imply a system whose material properties are modulated periodically in time. Such periodic modulation extends for all times t, i.e., $-\infty < t < \infty$ . However, to probe the emerging features of the PTCs, we can only consider a finite number of cycles of such periodic time modulation. Theoretically, this requires solving the Maxwell equations with temporal interface conditions.

In Fig. 13(a), a temporally finite PTC, i.e., a temporal slab, is shown. Such a temporal slab consists of $p$ cycles of a stepwise temporal modulation (see Fig. 13(a)). During one period of the modulation, the permittivity of the medium takes the values $\varepsilon_{1}$ (for time interval $t_1$ ) and $\varepsilon_{2}$ (for time interval $t_2$ ) (see Fig. 13(a)). The slab exists only for a finite time, i.e., $0 < t < pT_{\mathrm{m}}$ . Furthermore, we assume the permittivity of the medium for times $t < 0$ and $t > pT_{\mathrm{m}}$ to be $\varepsilon_{\mathrm{s}}$ . To solve the scattering problem for such a temporal slab, the transfer matrix method [102] can be used as outlined here in Section 2.1e. Such an approach is helpful, as knowing the temporal matching matrix $\overline{\overline{J}}_{\varepsilon_{\mathrm{a}} \to \varepsilon_{\mathrm{b}}}$ of the interface at which the permittivity jumps from some $\varepsilon_{\mathrm{a}}$ to some $\varepsilon_{\mathrm{b}}$ and the ABCD-matrix of the temporal modulation (see Eq. (33)) is sufficient to construct the effective transfer-matrix $\overline{\overline{T}}$ of the scattering structure shown in Fig. 13(a). Note that we can compute the temporal matching matrix $\overline{\overline{J}}_{\varepsilon_{\mathrm{a}} \to \varepsilon_{\mathrm{b}}}$ using the continuity of the fields $D_x, B_y$ (see Eqs. (24), (25)) and Eqs. (27), (28)) at the temporal interface $\varepsilon_{\mathrm{a}} \to \varepsilon_{\mathrm{b}}$ as

$$
\overline {{\overline {{J}}}} _ {\varepsilon_ {\mathrm{a}} \to \varepsilon_ {\mathrm{b}}} = \left[ \begin{array}{c c} 1 + \frac {\omega_ {\mathrm{a}}}{\omega_ {\mathrm{b}}} & 1 - \frac {\omega_ {\mathrm{a}}}{\omega_ {\mathrm{b}}} \\ 1 - \frac {\omega_ {\mathrm{a}}}{\omega_ {\mathrm{b}}} & 1 + \frac {\omega_ {\mathrm{a}}}{\omega_ {\mathrm{b}}} \end{array} \right],\tag{72}
$$

![](images/8b3ccec000e64948b15250b11f59a1dbdcb06542c17e50d88769d95cceb5bc0d.jpg)
(b)

![](images/92e1a0f549cbc7a53ffee315c2f02cada1cefeac903f7b5b113c8f1ca6cbf3e9.jpg)
(a) Temporal slab consisting of a stepwise temporal modulation. (b) Transmittance $T$ of the temporal slab (in the inset) as a function of the detuning parameter $\delta \omega$ for various values of chirping coefficient $F_{\mathrm{s}}$ . Here, $\varepsilon_{\mathrm{av}} = 1$ . (b) Reprinted from [149] under a Creative Commons license.

where $\frac{\omega_{a}}{\omega_{b}} = \sqrt{\frac{\varepsilon_{b}}{\varepsilon_{a}}}$ (see Eq. (26)). Here, Eq. (72) is similar to Eq. (3) in Ref. [102]. Finally, we can write the effective transfer-matrix $\overline{T}$ of the temporal slab shown in Fig. 13(a) as [102, Eq. (11)]

$$
\overline {{\overline {{T}}}} = \overline {{\overline {{J}}}} _ {\varepsilon_ {2} \to \varepsilon_ {\mathrm{s}}} \cdot \overline {{\overline {{J}}}} _ {\varepsilon_ {2} \to \varepsilon_ {1}} ^ {(- 1)} \cdot \overline {{\overline {{M}}}} ^ {(- p)} \cdot \overline {{\overline {{J}}}} _ {\varepsilon_ {\mathrm{s}} \to \varepsilon_ {1}}.\tag{73}
$$

Here, the transfer matrix $\overline{T}$ connects the forward- and backward-propagating fields $f_{\mathrm{s}}(pT_{\mathrm{m}}^{+})$ , $b_{\mathrm{s}}(pT_{\mathrm{m}}^{+})$ at time $t = pT_{m}^{+}$ to the forward- and backward-propagating fields $f_{\mathrm{s}}(0^{-})$ , $b_{\mathrm{s}}(0^{-})$ at time $t = 0^{-}$ as (see Fig. 13(a))

$$
\left[ \begin{array}{c} f _ {\mathrm{s}} (p T _ {\mathrm{m}} ^ {+}) \\ b _ {\mathrm{s}} (p T _ {\mathrm{m}} ^ {+}) \end{array} \right] = \overline {{\overline {{T}}}} \cdot \left[ \begin{array}{c} f _ {\mathrm{s}} (0 ^ {-}) \\ b _ {\mathrm{s}} (0 ^ {-}) \end{array} \right].\tag{74}
$$

Having calculated the transfer matrix $\overline{T}$ , one can easily calculate the optical observables, i.e., transmittance (T), reflectance (R), and absorbance (A) of the underlying scattering structure [102]. In particular, the transmittance and reflectance for a forward-propagating incident field with amplitude $f_{\mathrm{s}}(0^{-})$ at the time $t = 0^{-}$ is given by $T = \left|\overline{\overline{T}}_{11}\right|^{2}$ and $R = \left|\overline{\overline{T}}_{21}\right|^{2}$ , respectively. These expressions are obtained from Eq. (74) by considering that $b_{\mathrm{s}}(0^{-}) = 0$ , i.e., the absence of the backward wave at t < 0 due to causality [17]. Furthermore, the transmittance and reflectance for a backward-propagating incident field with amplitude $b_{\mathrm{s}}(0^{-})$ at the time $t = 0^{-}$ is given by $T = \left|\overline{\overline{T}}_{22}\right|^{2}$ and $R = \left|\overline{\overline{T}}_{12}\right|^{2}$ , respectively (here due to causality, $f_{\mathrm{s}}(0^{-}) = 0$ ). Moreover, the absorbance A can be computed as A = 1 - R - T. Note that in Fig. 13(a), a temporal stepwise system is considered. However, one can use the transfer matrix method for an arbitrary temporal modulation, as shown in Ref. [100]. Furthermore, the transfer matrix method can also be used to homogenize the finite PTCs shown in Fig. 13(a) using an eigenmode-based approach as discussed in Ref. [147]. Moreover, the transfer matrix method is also useful to study the photon squeezing in the time-varying media [148].

As an application of the finite PTCs, the authors of Ref. [149] consider the transfer matrices of more complex temporal slabs using the Möbius transformation method. In particular, they investigate the reflectance and transmittance of temporally finite slabs when the modulation frequency $\omega_{\mathrm{m}}$ and/or the modulation strength $m_{\varepsilon}$ becomes time-dependent, that is $\varepsilon(t) = \varepsilon_{\mathrm{av}}[1 + m_{\varepsilon}(t)\cos \omega_{\mathrm{m}}(t)t]$ . Note that the former phenomenon is known as chirping and the latter is known as apodization of the temporal modulation. In the following, we review the effects of a linear chirp on the temporal modulation. Note that a linearly chirped modulation is important to consider, as the chirping of light pulses occurs inevitably in dispersive systems due to the group velocity dispersion [150]. Therefore, in the systems that involve temporal modulation of $\varepsilon$ using optical pumps, such a chirping effect becomes crucial. Furthermore, as outlined later, a linearly chirped modulation leads to an increase in the amplification bandwidth of the temporal slabs [149]. Such a change in the bandwidth can, in principle, be used to control the shape of the light pulses scattered off the temporal slabs.

As shown in the inset of Fig. 13(b), a temporal slab with a temporally perturbed modulation frequency $\omega_{\mathrm{m}}(F_s,t)$ is considered. Note that the modulation frequency $\omega_{\mathrm{m}}(F_s,t)$ depends on $F_{s}$ as $\omega_{\mathrm{m}}(F_s,t) = \omega_{\mathrm{m0}} + 2F_{\mathrm{s}}t / \Delta t^2$ . Here, $\omega_{\mathrm{m0}}$ is a constant modulation frequency over which the linear chirping is applied, $F_{s}$ is the chirping coefficient, and $\Delta t$ is the total duration of the considered temporal slab. Here, the total duration of the slab is taken as $\Delta t = 20T_{\mathrm{m}}$ . Having chirped the temporal modulation, the transmittance of the temporal slab is calculated for different values of the chirping coefficient $F_{s}$ . Such transmittance as a function of the detuning parameter $\delta \omega$ is shown in Fig. 13(b). Note that $\delta \omega$ corresponds to the detuning of the incident frequency $\omega_{\mathrm{inc}}$ from $\frac{\omega_{\mathrm{m}}(F_{\mathrm{s}},t)}{2}$ , i.e., $\delta \omega = \omega_{\mathrm{inc}} - \frac{\omega_{\mathrm{m}}(F_{\mathrm{s}},t)}{2}$ . From Fig. 13(b), we observe that as $F_{s}$ increases (chirping increases), the maximum value of the transmittance $T$ decreases. However, the $k$ -bandwidth over which $T$ is substantially higher than unity increases. Therefore, we conclude that the linear chirping of the temporal modulation increases the amplification bandwidth at the cost of the maximum value of the amplification of the incident fields.

As a summary, we have been elaborating in the last two sections on aspects that matter for realistic systems, i.e., their finiteness. The assumption of an infinite spatial and temporal extent cannot hold up in reality, and we must consider the finiteness in both dimensions. In the following section, we would like to lift one more assumption made up to this point, which does not always hold in practical realizations of PTCs. It concerns the isotropy of the considered material.

## 3.4. Effects of Anisotropy in PTCs

Up to now, only isotropic materials have been considered. Isotropic materials are characterized by a scalar susceptibility or permittivity. In contrast, in this section, we discuss PTCs made from anisotropic materials. Examples of anisotropic materials include various types of crystals. The anisotropy implies that the material properties are described by tensorial quantities. The induced polarization density does not

## Figure 14

![](images/17be9f8d7925459defce8c479a73be70f26b9b37ae38beafb2233a1229b09b9b.jpg)

![](images/85330af95b4932e979a71510302907a5a9f99fd95f7c505bacb8cc60b49489cf.jpg)

![](images/047ede9b56f736ef5383d2f655c261b34631f02dd9e8d5794ce35a683fb6c7dd.jpg)
(d)

![](images/64ddee1dc741ee8fed9bc7de8b80887d001437eabeb71d63a87effccc9c50817.jpg)
(a) Permittivity modulation in an anisotropic PTC. The permittivity oscillates between that of an uniaxial anisotropic medium and an isotropic medium where $\varepsilon = 1$ , $\varepsilon_{\parallel} = 4$ , $\varepsilon_{\perp} = 25$ , $t_{1} = t_{2} = T_{m}2$ , and $k_{0} = 2\pi/T_{m}c_{0}$ . (b) Permittivity ellipsoid of the PTC during the times when it is anisotropic and described by permittivity tensor $\overline{\varepsilon}_{2}$ . The arrows depict the principal axes of the material permittivity. The incidence xz-plane is shown in pink. (c) Band structure for the ordinary light. The yellow lines depict the first and second bands, whereas the yellow arrows denote the bandgaps. The bandgaps $\Delta k_{x}$ and $\Delta k_{z}$ have the same widths. (d) Band structure for the extraordinary light. The bands depend on the propagation direction. In particular, the lowest-order bandgaps in the $k_{x}$ and $k_{z}$ directions are different. (a), (c), and (d) Figure 1 reprinted with permission from Li et al., Phys. Rev. Lett. 130, 093803, 2023 Ref. [151]. Copyright (2023) by the American Physical Society.

need to have the same orientation as the electric field that induces it. Moreover, the isofrequency surfaces of the dispersion relation in anisotropic materials usually have ellipsoidal shapes. As such, the length of the wavenumber depends on the direction of propagation.

Anisotropic PTCs combine the anisotropy and periodic temporal modulations, resulting in more complex and exotic light–matter interaction phenomena, including light emission manipulation and control of radiative energy distribution in space. In the following, we concentrate on the most fundamental configuration of anisotropic PTCs explored in Ref. [151], which is constructed by alternating in time between two types of lossless, nonmagnetic media periodically. The first medium is isotropic with scalar permittivity $\varepsilon$ , whereas the second medium is an uniaxial crystal with permittivity tensor $\overline{\bar{\varepsilon}}_{2}$ , as shown in Fig. 14(a). For simplicity, material dispersion is disregarded, and the principal axes of the uniaxial crystal are aligned to the coordinate system, which can be seen in Fig. 14(b). In this coordinate system, the permittivities of the PTC at the two temporal states are given by $\overline{\bar{\varepsilon}}_{1} = \varepsilon\overline{\bar{I}}$ and $\overline{\bar{\varepsilon}}_{2} = \text{diag}(\varepsilon_{\perp}, \varepsilon_{\perp}, \varepsilon_{\parallel})$ , where $\overline{\bar{I}}$ is the identity matrix.

It is well known that in uniaxial anisotropic media, a light beam that propagates not along the optical axis can be split into two rays with orthogonal polarization. The first ray has the polarization orthogonal to the optical axis. It is called ordinary because it “sees” the same material permittivity $\varepsilon_{\parallel}$ independent of its incident direction. The second ray is referred to as extraordinary because it is direction-dependent, that is, depending on the propagation direction, it experiences a material permittivity within the range $[\varepsilon_{\parallel};\varepsilon_{\perp}]$ . Likewise, the polarization degeneracy of the eigenmodes in an anisotropic PTC is lifted. For a given incidence plane, there are two distinct photonic band structures: for ordinary and extraordinary light. Following Ref. [151], let us choose the incidence plane to be parallel to the $xz$ -plane. From Fig. 14(b), we can see that the ordinary light has transverse-electric (TE) or $s$ -polarization. On the other hand, the extraordinary light has transverse-magnetic (TM) or $p$ -polarization.

Since the light of orthogonal polarizations does not mix, the photonic band structure can be calculated using the transfer matrix method, analogously to how it was described in Section 2.1e. Performing the necessary derivations, one can obtain two independent eigenvalue equations similar to (37), one for the ordinary and another for the extraordinary light. Solving these equations yields two band structures of the anisotropic PTC. Figures 14(c) and (d) show the band structures for a specific PTC configuration with a setting described in the figure caption. One can see from Fig. 14(c) that for the ordinary light, the band structure does not depend on the direction of propagation since it has radial symmetry in the $k_x - k_z$ plane. In other words, this band structure is similar to an isotropic PTC. Indeed, the lowest-order bandgaps in both the $k_x$ and $k_z$ directions, marked with yellow color, have the same widths ( $\Delta k_x = \Delta k_z$ ). Note that in the figure, both real and imaginary parts of the eigenfrequency contours are plotted using different color maps. In contrast, the band structure for the extraordinary light is not symmetric in the $k_x - k_z$ plane, as shown in Fig. 14(d). Here, the lowest-order bandgaps $\Delta k_x$ and $\Delta k_z$ are different. Thus, light amplification now depends strongly on its propagation direction. This regime of anisotropic PTCs provides exciting opportunities to achieve direction-dependent light amplification.

Furthermore, anisotropic PTCs possess another unique feature. It is well known that in a time-invariant medium, at the rest frame, a static (position- and time-invariant) charge cannot generate electromagnetic radiation. However, this is not true anymore if the stationary charge is positioned inside an anisotropic PTC $[151]$ . When the medium switches between the isotropic and anisotropic states, the static charge induces propagating electromagnetic waves. It appears that this phenomenon of DC-to-AC conversion is unique to anisotropic PTCs and was not achieved previously in the isotropic counterparts. A more comprehensive discussion on the phenomenon of electromagnetic radiation from a charge located inside a PTC is given in Section 6.2.

In addition to Ref. [151], we note that the concept of PTC has also been extended to biaxial anisotropic materials. It has been shown that specific nonuniform plane waves can experience broad momentum bandgaps even with small modulation depths in such materials [71]. Recently, the study of wave propagation in time-varying anisotropic media has been linked to the concept of twistronics in condensed matter physics, serving as its temporal counterpart [152]. Condensed-matter twistronics involves the study of twisted 2D material bilayers and the effect of the twisting angle on their electrical properties. The photonic research outlined in Ref. [152] investigates light propagation through a spatially unbounded anisotropic medium experiencing a temporal jump in the relative permittivity tensor. This jump results in the creation of a new anisotropic medium which is a rotated version of the original. It was discovered that such temporal jumps lead to frequency conversion, the extent of which significantly depends on the propagation direction of the initial wave, the rotation angle, and the initial values of the material parameters.

## 3.5. Defects in PTCs

The defects in photonic crystals have given rise to various interesting applications such as enhanced light emission $[153]$ , photonic-crystal-based beam splitters $[154]$ , and ultrasensitive sensors $[155]$ . Similarly, the defects in PTCs may also lead to many intriguing physical effects. Their consideration is a further endeavor when studying more realistic PTCs.

In Ref. [103], the effect of temporal defects during the periodic temporal modulation of PTCs has been studied. In Fig. 15(a), a PTC with a time-periodic permittivity $\varepsilon$ has been shown without any defects. Note that the permittivity $\varepsilon$ fluctuates between $\varepsilon_{1} = 3$ for time interval $t_1 = 1$ fs and $\varepsilon_{2} = 1$ (see Fig. 15(a)). To introduce a temporal defect, the time-periodic modulation is perturbed for the time interval $t_\mathrm{d}$ during which the permittivity takes a different value as compared with the values during the periodic modulation (see Fig. 15(b)). Such a value of the permittivity during the time interval $t_\mathrm{d}$ is denoted by $\varepsilon_\mathrm{d}$ . After the time interval $t_\mathrm{d}$ , the permittivity modulation of the PTC returns to its earlier periodic state as shown in Fig. 15(b). In the following, we assume that $\varepsilon_\mathrm{d} = 1$ and $t_\mathrm{d} = 1$ fs.

The photonic band structures of the PTCs characterize its optical response. Therefore, it is important to note the changes introduced in the band structures of the PTCs in the presence of temporal defects. In Figs. 15(c) and (d), the imaginary part of the eigenfrequency $\omega$ as a function of the eigenwavenumber $k$ of the considered PTC is shown in the absence and presence of the temporal defect, respectively. From Fig. 15(c), we observe the existence of nonzero imaginary parts of the eigenfrequency $\Im (\omega_{\mathrm{F}})$ in two different spectral regions ( $k$ regions). Therefore, there exist two momentum bandgaps in the considered spectral range. However, from Fig. 15(d), in the presence of the defect, we observe that within the bandgaps, there exist certain $k$ values for which $\Im (\omega_{\mathrm{F}})$ goes to zero, leading to the splitting of each bandgap. Such splitting is evident as each main lobe in Fig. 15(c) splits into two main lobes in Fig. 15(d). Furthermore, we note that in addition to the main lobes in Fig. 15(d), we observe certain sidelobes of $\Im (\omega_{\mathrm{F}})$ outside the momentum bandgap. Note that despite the fact that the time modulation is not strictly periodic anymore, it is still possible to calculate the band structure of the defective PTC. Such a calculation can be easily done by considering the effect of defect in terms of a defect matrix while calculating the band structure of the PTC using the transfer matrix approach [103, Eq. (2)].

As a next step, the effect of the aforementioned features due to defects on the transmissivity T of the PTC is investigated. As mentioned in Section 3.3, we need a temporally finite PTC to calculate such transmissivity. Therefore, in the following, we assume the defect-free PTC to have 10 temporal unit cells. Further, the defective PTC is taken such that the resulting scattering structure has five temporal unit cells on both sides of the temporal defect (see Fig. 15(b)). The transmissivity of such a finite defective PTC is shown in Figs. 15(e) and (f). In Fig. 15(e), we observe two maxima of the transmittance T of the PTC in the absence of the defect. They occur due to the presence of two momentum bandgaps in the considered spectral range (see Figs. 15(c) and (d)). However, in the presence of the defect, a minima of T exists sandwiched between the maxima. This can be explained by the vanishing values of $\Im(\omega_{\mathrm{F}})$ at the spectral locations of the minima of T (see Fig. 15(d)).

Such temporal defects endow PTCs with additional degrees of freedom. Therefore, the amplification within the momentum bandgaps can be manipulated more effectively by exploiting the defects.

## 3.6. Effects of Disorder

The propagation of electromagnetic waves in temporally disordered media is another crucial aspect with regard to realistic PTCs. Such disordered PTCs correspond to those photonic systems that are spatially homogeneous, but their material properties change randomly as a function of time. In Ref. [58], the interaction of such disordered PTCs with an incident pulse was studied. Figure 16(a) shows the permittivity $\varepsilon(t)$

## Figure 15

(a)
![](images/8fe354dbeeeef39f6462d03fcee4f37d27f0b526f9a80977330490cd68c17cf3.jpg)

(b)
![](images/176a6599fdf5d0fd46177040fb4314e48bb0291256748e84559e8c8d8b681ca6.jpg)

![](images/2f693d36e124e30f554efba3fcd352254d7edc8ca9250cb769ac1262b42a0506.jpg)

![](images/78ea80b15b7b25fb376da0e15dda956686a6a6c133f440b62fb62e9f3d82fefa.jpg)

![](images/220dc50375559b5ecb2546f3b327fa27b305d1e66cc68dd49566a39ec53a76f9.jpg)

![](images/4bf61f86ca206536df65ecb2f6163750c650b241a6c42e9592141a12eb331e94.jpg)
(a) PTC consisting of a time-periodic stepwise permittivity modulation. (b) PTC with a temporal defect. Here, the periodic nature of the temporal modulation of the PTC is broken for the time duration $t_{d}$ due to a temporal defect with permittivity $\varepsilon_{d}$ . (c), (d) Imaginary part of the eigenfrequency $\omega$ as a function of the eigenwavenumber k (band structure) in the absence and presence of the defect, respectively. (e), (f) Transmittance T of the PTC in the absence and presence of the defect, respectively. Note that $k_{0} = 2\pi/(cT_{\mathrm{m}})$ . Figures 1, 2, and 3 reprinted with permission from Sadhukhan and Ghosh, Phys. Rev. A 108, 023511, 2023 Ref. [103]. Copyright (2023) by the American Physical Society.

of the disordered PTC under consideration. The permittivity of such a disordered PTC consists of equal time segments of duration T. During each such segment, the permittivity can be written as $\varepsilon = 2 + A \times U[-1, 1]$ . Here, A represents the disorder magnitude, and U is a uniform distribution. Next, the propagation of a Gaussian pulse is studied through such a disordered system (see Fig. 16(b)). Note that we assume the full width at half maximum (FWHM) of the pulse to be 200 T. Further, the central frequency of the pulse is taken as $2\pi/(5T)$ . In the following, the group velocity $v_{g}$ and pulse energy of the Gaussian pulse are examined for different disorder amplitudes A. Figures 16(c) and (d) show the variation of the group velocity and energy of the pulse, respectively, as a function of the propagation time t. From Fig. 16(c), it can be seen that the group velocity of the pulse goes down exponentially as a function of the propagation time. Further, Fig. 16(d) shows an exponential growth of the pulse energy in the disordered systems with the propagation time. As discussed in Ref. [58], a further investigation of the temporally disordered systems reveals a strong dependence of the emerging effects on the band structure of the PTC. Moreover, effects analogous to the Anderson localization are also reported in the disordered PTCs [58].

In addition, some interesting applications of the disordered media are discussed in Ref. [156]. Specifically, the temporal disorder was utilized to tailor light scattering from spatially homogeneous time-varying structures. The time-varying permittivity of the considered disordered system can be written as $\varepsilon(t) = \varepsilon_{\mathrm{av}}(1 + \Delta\varepsilon(t))$ . Here, $\Delta\varepsilon(t)$ incorporates the temporal disorder. First, in Ref. [156] the authors discussed how temporally disordered systems can be used to attain unidirectional scattering. Figures 16(e) and (f) show the permittivity profiles of the engineered temporal systems that exhibit negligible backward and forward scattering for an incident frequency $\omega_{\mathrm{inc}}$ , respectively. Note that the incident frequency $\omega_{\mathrm{inc}}$ is such that $\omega_{\mathrm{inc}} = kc / \sqrt{\varepsilon_{\mathrm{av}}}$ . Here, $k$ is the magnitude of the incident wave vector. Next, a transition from ordered temporal modulation (PTC regime) to uncorrelated temporal disorder has been

## Figure 16

![](images/9365fad88f4541a58c31b5a102d7c54e48bf2217de28e133f7aa0e90ffe0291d.jpg)

![](images/c90d1e0390e3142af0fa11798e60c5f5ec7cb02a8b26f23eca7708a0d48c4dc5.jpg)

![](images/6b2b2aa3a28e72d7c771df7780968c13053798a1d0013e1a14cc1cdbebe670ba.jpg)

(d),
![](images/ae94f886c74597be54087cbcd49fc0a7d27ebe7631b4a0cba2717a5dffc9a5b5.jpg)

![](images/e8cd71da1828c59a44a34c732dd4665edc565b320fd1233fccd776b5045f7539.jpg)

![](images/fd6a7867122efefcb56ecfe15f94c7db62f82616c3c9586cc6c4a138a9b924c9.jpg)

![](images/548307bd50cadfe575e988e9a0f12de6f75672d1823f193379070b02c46e4986.jpg)
(a) Permittivity $\varepsilon$ of a temporally disordered system as a function of time t. (b) Propagation of a Gaussian pulse in a spatially homogeneous medium with temporally disordered permittivity shown in (a). (c) Group velocity $v_{g}$ and (d) energy of the Gaussian pulse shown in (b) as a function of time. (e), (f) Realized permittivity disorder $\Delta\varepsilon$ (gray areas) and the corresponding scattering intensities $|D_{sca}|^{2}$ (solid lines) of the systems that suppress backward and forward scattering, respectively. (g), (h) Forward- and backward-scattered powers ( $P_{FW}$ , $P_{BW}$ ) a function of wave momenta k, respectively, of a device designed to suppress forward scattering. Here, cases A, B, and D represent perfectly ordered, intermediate, and near-Poisson temporal modulations, respectively. Here, $t_{\mathrm{inc}} = 2\pi/\omega_{\mathrm{inc}}$ , $D_{\mathrm{sca}}(t)$ is the scattered electric displacement field, and $C_{0}$ is a normalization constant that depends on the correlation of $\varepsilon(t)$ . (a)–(d) Figures 1 and 2 reprinted with permission from Sharabi et al., Phys. Rev. Lett. 126, 163902, 2021, Ref. [58]. Copyright (2021) by the American Physical Society. (e)–(h) Reprinted from [156] under a Creative Commons license.

studied. Such an investigation reveals that the uncorrelated disorder increases the momentum bandwidth over which nonzero backward scattering can be attained while maintaining negligible forward scattering [see Figs. 16(g) and (h)]. Note that such a negligible forward scattering is maintained by inversely designing the temporal disorder.

## 3.7. Nonlinear PTCs

Finally, we discuss in this section the propagation of electromagnetic fields in the PTCs formed inside nonlinear media (see also a relevant discussion for photonic space–time crystals in Section 7.5). The permittivity of such nonlinear PTCs has a time-periodic linear part and a stationary nonlinear contribution term. One example of a nonlinear PTC supporting Kerr nonlinearity has the form $\varepsilon(t,|\mathbf{E}|^{2})=\varepsilon_{1}(t)+\chi^{(3)}|\mathbf{E}|^{2}$ [56]. Here, $\varepsilon_{1}(t)$ is the time-periodic linear permittivity given by $\varepsilon_{1}(t)=\varepsilon_{\mathrm{av}}[1+m_{\varepsilon}\cos(\omega_{\mathrm{m}}t)]$ , $\chi^{(3)}$ is the third-order nonlinear susceptibility, and E is the electric field of light. Such a nonlinear PTC has been studied in detail in Ref. [56]. In particular, such PTCs are shown to support solitonic solutions inside their momentum bandgap. Solitons refer to those wave pulses that maintain their shapes while traveling through a medium despite the dispersion of the medium [157]. Note that the solitons found inside the momentum bandgap (k-gap) of the PTCs differ from the solitons found inside the energy bandgaps ( $\omega$ -gap) of the spatial photonic crystals (see Figs. 17(a) and (b)). First, the solitons

## Figure 17

![](images/f9af675445f1796bea4065f4ff83626666f6070661d9825f9632c45c970a47d6.jpg)

![](images/bb856c960267dfc3a2e513a9ee402e21f41f4ec41f843258e18e9370a25f827d.jpg)

(c)
![](images/a635d5060f4ec727984cc53484538612de00ca95e60474c0e6e4466ed4984027.jpg)

![](images/8504e2970f22aba9aa440103ac975647d0e1c7f14b39abec1305bb92ad958e0a.jpg)
(a), (b) Soliton solutions inside the $\omega$ -gap and k-gap, respectively. Here, the vertical axis denotes the intensity of the soliton wave pulse. (c) Spatiotemporal dynamics of k-gap solitons generated by an input Gaussian pulse (shown by solid red curve). (d) Temporal profiles of the solitons generated by the input pulse in (b) at different spatial locations. Figures 1 and 2 reprinted with permission from Pan et al., Phys. Rev. Lett. 130, 233801, 2023, Ref. [56]. Copyright (2023) by the American Physical Society.

existing inside the $\omega$ -gaps are finite wave packets in space, but they have a plane wave dependence in time $t$ (see Fig. 17(a)). Further, their group velocity, $v_{\mathrm{g}} = 0$ due to the fact that the edge states of $\omega$ -gap satisfy $\frac{\partial\omega}{\partial k} = 0$ .

In contrast, the $k$ -gap solitons are finite wave packets in time, but they have a plane wave dependence in space (see Fig. 17(b)). Further, their group velocity, $\nu_{\mathrm{g}} = \infty$ due to the fact the edge states of the $k$ -gap satisfy $\frac{\partial k}{\partial \omega} = 0$ . Such infinite group velocity indicates a superluminal behavior of the $k$ -gap solitons as they travel faster than light inside the nonlinear PTCs. Figure 17(c) shows the generation of such a $k$ -gap soliton. To produce the results in the figure, the units are chosen such that the speed of light in vacuum $c = 1$ , the modulation frequency is $\omega_{\mathrm{m}} = 4\pi$ , the modulation strength $m_{\varepsilon} = 0.12$ , and the third-order nonlinear susceptibility is $\chi^{(3)} = \varepsilon_0\varepsilon_{\mathrm{av}}^3 /300$ . Further, in Fig. 17(d), the temporal profiles of the gap soliton excited by the input pulse in Fig. 17(c) for different spatial locations are plotted. The intensity of the solitons in Fig. 17(d) first grows as a function of time. Such growth is due to the exponentially growing modes inside the $k$ -gap. However, after attaining a certain peak, the intensity decays with time. Such decay is driven by the nonlinearities of the system that transfer the power from the growing modes to the decaying modes of the $k$ -gap. Furthermore, as noted in Ref. [56], the infinite group velocity of the solitons does not violate Einstein's causality principle. This is because, in active media, the information velocity is defined by the velocity of the leading edge of the wave packet. Therefore, the group velocity, which quantifies the motion of the center of the wave packet, does not relate to the velocity at which information travels. Of course, as shown numerically, the velocity of the leading edge of the wave packet does not exceed the speed of light $[56]$ . Finally, recently in Ref. $[158]$ Kiselev and Pan showed that the interplay between a time-varying permittivity and nonlinearity induces broken spatial and time translation symmetries in the PTCs.

## 4. RELATIONS TO OTHER PARAMETRIC SYSTEMS: SIMILARITY AND DISTINCTION

In the rapidly evolving field of photonics, distinguishing PTCs from related phenomena is essential for a comprehensive understanding. This discussion clarifies the nuances that differentiate PTCs, such as their unique temporal and frequency characteristics, from similar-sounding or similar-appearing phenomena in other fields of electrical engineering, optics, and condensed matter physics. By elucidating these distinctions and parallels, we facilitate deeper insights into the behavior and applications of PTCs, contributing to a more accurate scientific discourse. Such an analysis not only aids in theoretical comprehension but also guides practical advancements, ensuring that researchers and practitioners accurately apply these concepts to innovate in different areas.

In general, PTCs are material systems that involve parametric amplification effects. In particular, light with a wavenumber located inside the momentum bandgap of a PTC is parametrically amplified. In a general sense, parametric amplification is called “parametric” because it involves the modulation of a system’s parameter (such as capacitance, inductance, or permittivity) to achieve signal amplification. This process does not add energy directly to the signal, but rather, through the periodic variation of the system parameter, energy is transferred from an external source to a signal that is amplified. Parametric amplification can occur in various physical systems, such as electrical circuits, nonlinear optical materials, transmission lines, and mechanical parametric pendula. In the following sections of this section, we discuss and compare parametric amplification effects in different systems: electrical circuits in Section 4.1, nonlinear materials in Section 4.2, and PTCs in Section 4.3.

Finally, in Section 4.4 we compare PTCs with the recently discovered “time crystals” in condensed matter physics.

## 4.1. Parametric Amplification in Electrical Circuits

Parametric amplification, a cornerstone in modern circuit design, exploits the principle of parameter modulation to amplify signals. This mechanism, distinct from direct energy transfer seen in traditional amplifiers, modulates a system parameter (such as capacitance or inductance) periodically at a specific frequency. This modulation can transfer energy from the power supply to the signal, resulting in amplification. That concept of parametric amplification, first discovered in 1892 [159], has found profound applications ranging from telecommunications to quantum computing. In radio and microwave engineering, it enhances signal strength while preserving phase information, which is essential for high-fidelity communications.

Let us consider the parametric amplification phenomenon in the most simple electric circuit consisting of a time-varying capacitance $C(t)$ connected to a voltage source $v(t)$ , as shown in Fig. 18(a). This source is time-harmonic and described by voltage $v(t) = V_0\cos (\omega_{\mathrm{inc}}t + \phi_{\mathrm{inc}})$ , where $\phi_{\mathrm{inc}}$ is an arbitrary phase. First, let us consider for simplicity a scenario where the capacitance is modulated in time in a stepwise manner (see the red curve in Fig. 18(c)). Such time dependency could be mechanically induced by altering the spacing between a capacitor's plates. When these plates are moved apart or closer together, the capacitance decreases or increases correspondingly. Consider a scenario where at the moment $t = t_0$ , when the voltage across the capacitor reaches its peak value $v_{1}$ , the plates are instantly separated. This action necessitates work against the attractive force existing between the charged plates. Consequently, the capacitance shifts from $C_{1} = C_{av} + \Delta C/2$ to $C_{2} = C_{av} - \Delta C/2$ . It is convenient to introduce the relative capacitance change $\delta_{C} = C_{1}/C_{2} > 1$ . It is critical to note that during this transition, the electric charge on the capacitor remains unchanged [160, p.389]. As a result, the voltage across the capacitor must rise to $v_{2} = \delta_{C}v_{1}$ .

![](images/10f92e0d2b249c0d2da22c9bafa89c7da99bcc866f8373e5a85cedef6b80cd7a.jpg)
(a) Electrical circuit with a time-varying capacitance. (b) Circuit equivalent to that in (a) where the time-varying capacitance is replaced by a parallel connection of a time-invariant capacitance $C_{eq}$ and a resistance $R_{eq}$ . The resistance can have positive and negative values depending on the phase $\phi_{inc}$ . (c) Time evolution of the voltage in the circuit shown in (a) for the case of a stepwise capacitance modulation. Phase $\phi_{inc}$ is chosen such that there is parametric amplification in the circuit. (d) Same as (c) but where the phase is shifted by $\pi/4$ , which leads to a parametric de-amplification.

Accordingly, the energy stored in the capacitor, expressed as $C(t)v(t)^{2}/2$ , increases by $\delta_{C}$ . This increase in energy originates from the mechanical work done in separating the capacitor plates. Next, at the time moment when the voltage across the capacitor is zero, the plates are pushed together. However, this time, no energy is stored or released, as the charge at the capacitor plates is zero. Likewise, when the voltage reaches its maximum negative value, the plates are moved apart once again, leading to another increase in the capacitor's energy (refer to Fig. 18(c)). Thus, the energy in the circuit escalates throughout one complete oscillation cycle, resulting in parametric amplification (here, the modulated “parameter” is the capacitance C). The energy growth in the circuit is exponential. One can notice that to have an energy growth without any energy decrease within each cycle of the voltage (i.e., for the time interval $\Delta t = 2\pi/\omega_{inc}$ ), it is important to modulate the capacitance at twice the frequency, i.e., $2\omega_{inc}$ . Importantly, the modulation of the capacitance must be synchronized with the voltage in the circuit to obtain the maximum parametric amplification.

One can follow the same logic and consider the case when the voltage oscillation is shifted by a $\pi/4$ phase with respect to the voltage signal in the above example. Then, the power in the circuit would decrease twice per cycle (i.e., for $\Delta t = 2\pi/\omega_{inc}$ ) and never increase, as shown in Fig. 18(d). This regime is called parametric de-amplification [161].

Next, let us conduct a more detailed analysis of parametric amplification and de-amplification in the circuit. To simplify the analysis, instead of a stepwise capacitance modulation, we assume a time-harmonic modulation of the form $C(t) = C_{\mathrm{av}}(1 + m_{\mathrm{C}} \cos(\omega_{\mathrm{m}} t))$ , where we assume upfront that $\omega_{m} = 2\omega_{inc}$ . Note that we choose the modulation phase to be zero because only the relative phase between the modulation function and the voltage is important and is described by $\phi_{inc}$ in $v(t) = V_{0} \cos(\omega_{\mathrm{inc}} t + \phi_{\mathrm{inc}})$ . The alternating electric current flowing through the capacitance can be calculated using $i(t) = \mathrm{d}[C(t)v(t)]/\mathrm{d}t$ . By substituting inside this expression the capacitance and voltage functions and dropping the term oscillating at $3\omega_{inc}$ (practically, it is accomplished by adding a frequency filter), we arrive at the electric current consisting of two terms oscillating at $\omega_{inc}$ :

$$
i (t) = - C _ {\mathrm{av}} V _ {0} \omega_ {\mathrm{inc}} \sin (\omega_ {\mathrm{inc}} t + \phi_ {\mathrm{inc}}) - \frac {m _ {\mathrm{C}} C _ {\mathrm{av}} V _ {0}}{2} \omega_ {\mathrm{inc}} \sin (\omega_ {\mathrm{inc}} t - \phi_ {\mathrm{inc}}) = i _ {1} (t) + i _ {2} (t).\tag{75}
$$

The first term $i_{1}(t)$ is the electric current associated with a conventional linear and time-invariant capacitance $C_{eq}$ . It is easy to see that the time-averaged dissipated power due to this current term is always zero. That is the case because $v(t)$ and $i_{1}(t)$ have a $90^{\circ}$ phase shift, i.e., $\langle P(t)\rangle_{t} = \langle v(t)i_{1}(t)\rangle_{t} = 0$ .

The second term $i_{2}(t)$ is drastically different because it is a consequence of the temporal modulation. For determining the contribution of this second term, the phase $\phi_{inc}$ plays a significant role. The power dissipated at the capacitor for this term is given by

$$
\langle P (t) \rangle_ {t} = \frac {m _ {\mathrm{C}} C _ {\mathrm{av}} V _ {0} ^ {2} \omega_ {\mathrm{inc}}}{4} \sin 2 \phi_ {\mathrm{inc}}.\tag{76}
$$

Thus, the current term $i_{2}(t)$ for certain values of $\phi_{inc}$ leads to nonzero dissipated or accumulated power in the time-varying capacitor like it had some equivalent resistance $R_{eq}$ . Therefore, we can view the circuit in Fig. 18(a) as equivalent to a parallel connection of some static capacitance $C_{eq}$ and the static resistance $R_{eq}$ , as shown in Fig. 18(b). The resistance can be determined from $\langle P(t)\rangle_{t} = \frac{V_{0}^{2}}{2R_{eq}}$ . One can see that depending on the value of $\phi_{inc}$ , the resistance

$$
R _ {\mathrm{eq}} = \frac {2}{\omega_ {\mathrm{inc}} m _ {\mathrm{C}} C _ {\mathrm{av}} \sin 2 \phi_ {\mathrm{inc}}}\tag{77}
$$

can have different signs. The equivalent time-invariant capacitance is given by $C_{eq} = C_{av} + \frac{m_{C}C_{av}}{2} \cos 2\phi_{inc}$ . Note that, in general, $C_{eq} \neq C_{av}$ . Next, let us consider three special cases. First, when $\phi_{inc} = -\pi/4 + \pi p (p \in Z)$ , the equivalent resistance is negative $R_{eq} = -\frac{2}{\omega_{inc}m_{C}C_{av}}$ and $C_{eq} = C_{av}$ , resulting in parametric amplification in the circuit. On the other hand, when $\phi_{inc} = \pi/4 + \pi p$ , $R_{eq} = \frac{2}{\omega_{inc}m_{C}C_{av}} > 0$ , and $C_{eq} = C_{av}$ , there is a parametric de-amplification in the circuit. Finally, when $\phi_{inc} = \pi p/2$ , we obtain that the equivalent parameters are given by $R_{eq} \to \infty$ and $C_{eq} = C_{av} + (-1)^{p}\frac{m_{C}C_{av}}{2}$ . This means that instead of the equivalent resistor, we have an open circuit in Fig. 18(b), and no power accumulates or dissipates in the circuit. In this case, $\langle P(t)\rangle_{t} = 0$ .

The parametric amplification occurs when the modulation frequency is exactly twice the signal frequency $\omega_{m} = 2\omega_{inc}$ . For that reason, the process is referred to as degenerate [160, Section 11.4]. In the context of the considered circuit, this terminology means that $i_{1}(t)$ and $i_{2}(t)$ oscillate at the same signal frequency $\omega_{inc}$ , i.e., the two current modes degenerate regarding their frequency. Degenerate parametric amplification in electrical circuits is a process that is strongly sensitive to the phase difference of the modulation function and the signal [162, Ch. 11], as one can see from Eq. (77). In fact, this statement is very general and applies to other degenerate parametric physical systems. For example, as we show in Section 4.2, degenerate parametric amplification in nonlinear optics strongly depends on the relative phase difference between the signal and pump photons. Further, amplification inside the momentum bandgap in PTCs is typically studied in the case when the signal frequency $\omega_{F}$ is located in the first Brillouin zone, that is, in the degenerate regime when $\omega_{m} = 2\omega_{F}$ . Therefore, strictly speaking, PTCs are also phase-sensitive. However, as discussed in Section 4.3, this phase-sensitivity additionally depends on the spatial distribution of the signal wave inside the PTC. In fact, in most scenarios of degenerate PTCs, the phase-sensitivity vanishes unless the crystal is engineered in a specific way. This makes parametric amplification in PTCs different compared with that in other systems. A qualitative explanation of this difference was made in the supplementary information of Ref. [145], where a PTC was viewed as a transmission line comprising a cascade of circuits similar to that in Fig. 18(a). Although amplification/de-amplification in each circuit is phase-sensitive, in total, the energy of the standing mode (since it is inside the momentum bandgap) inside the transmission line grows exponentially. A more rigorous explanation of the phase-sensitivity properties of PTCs is given in Section 4.3.

Phase-sensitivity of parametric amplification often leads to difficulties in practical setups, as one needs to control the modulation and signal phases carefully. In such scenarios, engineers opt for nondegenerate amplification, that is, when $\omega_{m} \neq 2\omega_{inc}$ . It should be noted that nondegenerate amplification is qualitatively different from the degenerate one [160, p. 388] and is a phase-insensitive process. On the other hand, the phase-sensitivity of the degenerate parametric amplification also has very important applications, such as for generating squeezed states of light. These states have reduced noise in one quadrature of the electromagnetic field at the expense of increased noise in the other, allowing for measurements that surpass the quantum noise limit. This is invaluable in fields where noise reduction can significantly enhance performance, such as in quantum metrology and gravitational wave detection.

## 4.2. Parametric Amplification in Nonlinear Optics

In the previous section, we considered parametric amplification in a single lumped circuit element. However, more related to PTCs are distributed (bulk) systems where the modulation is in the traveling-wave form. Among others, they include transmission lines and nonlinear optical materials, as was pointed out in the Introduction. In this section, we overview coupled-mode equations for optical parametric amplification (difference-frequency generation process) in materials with a second-order nonlinearity. We start with the general (nondegenerate) case and conclude with the special case of degenerate parametric amplification. In Section 4.3, we derive coupled-mode equations for light propagation inside the momentum bandgap of a PTC and compare them with those in the present section.

Let us consider the difference-frequency generation process shown in Fig. 19(a). There, a pump wave at the frequency $\omega_{\mathrm{pump}}$ and a signal wave at the frequency $\omega_{\mathrm{s}}$ interact in a lossless optical medium with a $\chi^{(2)}$ nonlinearity. Out of that interaction, they produce an output idler wave at frequency $\omega_{\mathrm{i}} = \omega_{\mathrm{pump}} - \omega_{\mathrm{s}}$ . For simplicity, we assume that the pump wave is strong, and we neglect back-action from the generated signal and idler onto the pump. This implies that it is undepleted by the nonlinear interaction so that we can treat its amplitude as constant over distance $z$ . The photon description of the interaction of the three optical waves in the considered process is shown in Fig. 19(b). The spatial dependence of the amplitude of the pump wave propagating along the $z$ direction is given by $a_{\mathrm{pump}}(z) = A_{\mathrm{pump}}\exp (-jk_{\mathrm{pump}}z)$ , where $A_{\mathrm{pump}}$ is the constant pump amplitude and $k_{\mathrm{pump}}$ is the wavenumber of the pump wave. For the signal and idler waves $a_{\mathrm{s}}(z) = A_{\mathrm{s}}(z)\exp (-jk_{\mathrm{s}}z)$ and $a_{\mathrm{i}}(z) = A_{\mathrm{i}}(z)\exp (-jk_{\mathrm{i}}z)$ where the amplitudes are a slowly varying function in space compared with the fast oscillating exponential function. The well-known coupled-mode equations expressing how the amplitudes of the three waves are related to one another inside the nonlinear material along the $z$ direction read as [157, Sect. 2.8]

$$
\begin{array}{l} \frac {\mathrm{d} A _ {\mathrm{s}}}{\mathrm{d} z} = - j \eta_ {\mathrm{OPA}} A _ {\mathrm{i}} ^ {*} e ^ {- j \Delta k z}, \\ \frac {\mathrm{d} A _ {\mathrm{i}} ^ {*}}{\mathrm{d} z} = j \eta_ {\mathrm{OPA}} ^ {*} \frac {\omega_ {\mathrm{i}} n _ {\mathrm{s}}}{\omega_ {\mathrm{s}} n _ {\mathrm{i}}} A _ {\mathrm{s}} e ^ {j \Delta k z}, \end{array}\tag{78}
$$

where $\Delta k = k_{pump} - k_{s} - k_{i}$ , the coupling coefficient is $\eta_{\mathrm{OPA}} = \omega_{\mathrm{s}} \chi^{(2)} A_{\mathrm{pump}} / (cn_{\mathrm{s}})$ , $n_{s}$ , and the $n_{i}$ denote the refractive indices of the medium for the signal and idler waves.

![](images/55f9567c4337d621ecbc0faebfbd520045095cbaa2f94bb1afe14f8a10b87398.jpg)

![](images/3283b4dc31eb96898a162db0d97d4ca7bd693204729fa7e811d74c9f4b251100.jpg)
(a) Difference-frequency generation process resulting in optical parametric amplification along the spatial coordinate z. Typically, the idler frequency is absent at the input. (b) Photon description of the interaction of three optical waves in the process shown in (a). (c) Spatial evolution of the envelope amplitudes of the signal and idler waves in the assumption of perfect phase-matching $\Delta k = 0$ and an undepleted pump. The initial conditions are given by $A_{\mathrm{s}}(0) \neq 0$ and $A_{\mathrm{i}}^{*}(0) = 0$ .

Subscript “OPA” refers to the optical parametric amplification process. In the expression for the coupling coefficient, $\chi^{(2)}$ is the second-order nonlinear susceptibility. From the first equation in Eq. (78), one can see that the increase in signal amplitude $A_{s}$ along z is proportional to the amplitude of the idler wave $A_{i}^{*}$ and the pump wave amplitude $A_{pump}$ (hidden inside coefficient $\eta_{OPA}$ ). Note that the idler amplitude appears complex-conjugated because the product of the two time dependencies $\exp(j\omega_{pump}t)$ of the pump amplitude and $\exp(-j\omega_{i}t)$ of the complex conjugate of the idler amplitude results in the time dependence $\exp[j\omega_{pump}t - j\omega_{i}t] = \exp(j\omega_{s}t)$ of the signal wave. Therefore, from Eq. (78), the presence of a field at frequency $\omega_{i}$ stimulates the downward transition from $\omega_{pump}$ that leads to the generation of the $\omega_{s}$ field. Likewise, the $\omega_{s}$ wave stimulates the generation of the $\omega_{i}$ wave. Hence, the generation of the signal wave reinforces the generation of the idler wave and vice versa, leading to the exponential growth of each wave.

The optical parametric amplification process requires nearly perfect phase matching $\Delta k \approx 0$ since otherwise, the coupling between the signal and idler waves gets out of phase rapidly as light travels along the nonlinear material. The requirement of phase matching

$$
k _ {\mathrm{pump}} = k _ {\mathrm{s}} + k _ {\mathrm{i}}\tag{79}
$$

in practice usually translates into the necessity to engineer the birefringence of the nonlinear material, but a detailed consideration would go beyond the scope of this tutorial.

Assuming the typical initial conditions $A_{\mathrm{s}}(0) \neq 0$ and $A_{\mathrm{i}}^{*}(0) = 0$ , the solution of the differential equations (78) reads [157, Section 2.8]

$$
\begin{array}{l} A _ {\mathrm{s}} (z) = A _ {\mathrm{s}} (0) \cosh \left[ | \eta_ {\mathrm{OPA}} | \sqrt {\frac {\omega_ {\mathrm{i}} n _ {\mathrm{s}}}{\omega_ {\mathrm{s}} n _ {\mathrm{i}}} z} \right], \\ A _ {\mathrm{i}} ^ {*} (z) = \frac {j}{\eta_ {\mathrm{OPA}}} A _ {\mathrm{s}} (0) \sinh \left[ | \eta_ {\mathrm{OPA}} | \sqrt {\frac {\omega_ {\mathrm{i}} n _ {\mathrm{s}}}{\omega_ {\mathrm{s}} n _ {\mathrm{i}}} z} \right]. \end{array}\tag{80}
$$

The nature of this solution is shown in Fig. 19(c). Note that the signal and the idler fields experience monotonic growth and that asymptotically, each field experiences exponential growth. Importantly, this exponential growth is independent of the phase difference of the pump and signal waves (the phase of $A_{\mathrm{pump}}(z)$ does not affect the value of $|\eta_{\mathrm{OPA}}|$ appearing in Eq. (80), which determines the mode amplitude).

In the special case of degenerate parametric amplification, i.e., when $\omega_{s} = \omega_{i}$ , coupled-mode equations (78) simplify to

$$
\begin{array}{l} \frac {\mathrm{d} A _ {\mathrm{s}}}{\mathrm{d} z} = - j \eta_ {\mathrm{OPA}} A _ {\mathrm{s}} ^ {*}, \\ \frac {\mathrm{d} A _ {\mathrm{s}} ^ {*}}{\mathrm{d} z} = j \eta_ {\mathrm{OPA}} ^ {*} A _ {\mathrm{s}}, \end{array}\tag{81}
$$

where perfect phase matching was assumed $(\Delta k = 0)$ . Nevertheless, the degenerate scenario is qualitatively very different. Now, the excitations $A_{\mathrm{s}}(0)$ and $A_{\mathrm{s}}^{*}(0)$ lie in the same frequency band and thus determine jointly the input signal excitation. In other words, one cannot, in practice, satisfy the previously assumed initial conditions $A_{\mathrm{s}}(0) \neq 0$ and $A_{\mathrm{s}}^{*}(0) = 0$ . Therefore, it is convenient to rewrite two decoupled mode envelopes as [160, Section 11.4]

$$
\begin{array}{r} A _ {\mathrm{OPA}} ^ {(1)} (z) = \frac {1}{2} \left[ A _ {\mathrm{s}} (z) e ^ {- j \psi_ {\mathrm{OPA}} / 2} + A _ {\mathrm{s}} ^ {*} (z) e ^ {j \psi_ {\mathrm{OPA}} / 2} \right], \\ A _ {\mathrm{OPA}} ^ {(2)} (z) = - \frac {1}{2} \left[ A _ {\mathrm{s}} (z) e ^ {- j \psi_ {\mathrm{OPA}} / 2} - A _ {\mathrm{s}} ^ {*} (z) e ^ {j \psi_ {\mathrm{OPA}} / 2} \right], \end{array}\tag{82}
$$

where we defined the phase $\psi_{OPA}$ according to $j\eta_{OPA} = |\eta_{OPA}|e^{j\psi_{OPA}}$ as related to the phase of the pump wave.

Then, the solution of Eq. (81) in the basis of the decoupled mode envelopes Eq. (82) reads as

$$
\left[ \begin{array}{c} A _ {\mathrm{OPA}} ^ {(1)} (z) \\ A _ {\mathrm{OPA}} ^ {(2)} (z) \end{array} \right] = \left[ \begin{array}{c c} e ^ {- | \eta_ {\mathrm{OPA}} | z} & 0 \\ 0 & e ^ {| \eta_ {\mathrm{OPA}} | z} \end{array} \right] \cdot \left[ \begin{array}{c} A _ {\mathrm{OPA}} ^ {(1)} (0) \\ A _ {\mathrm{OPA}} ^ {(2)} (0) \end{array} \right].\tag{83}
$$

These equations predict an exponential spatial growth of $A_{\mathrm{OPA}}^{(2)}$ and an exponential decay of $A_{\mathrm{OPA}}^{(1)}$ . The two decoupled modes are $90^{\circ}$ out of phase. In contrast to Eq. (80), now the phase of the pump wave $A_{pump}$ affects the light propagation inside the nonlinear material, providing the possibility for both parametric amplification and de-amplification. Indeed, this phase affects $\psi_{OPA}$ and, therefore, also the values of the amplitudes $A_{\mathrm{OPA}}^{(1)}(0)$ and $A_{\mathrm{OPA}}^{(2)}(0)$ , as seen from Eq. (82).

Thus, degenerate parametric amplification in materials with $\chi^{(2)}$ nonlinearity is phase-sensitive, similarly to that in electrical circuits discussed in Section 4.1. In the next section, we analyze the coupled-mode equations for the dominant modes inside the momentum bandgap of a PTC and discuss the differences and similarities with the considered case.

## 4.3. Parametric Amplification in PTCs

In this section, we derive the temporal coupled-mode theory for light inside an infinite (both in space and time) PTC with negligible frequency dispersion, in particular, when its wavenumber resides inside the momentum bandgap. The derivations are similar to those published in Refs. $[140, Supplementary Material]$ and $[163]$ . In contrast to frequency-domain analysis, where eigenmodes within the momentum bandgap are described by complex frequencies, here we analyze temporal evolution in the time domain. Although both approaches yield the same solution for wave propagation inside the PTC, the time-domain analysis allows for a clearer comparison of wave processes in PTCs and nonlinear optics.

We start with the assumption that inside the bandgap, there are two dominant modes, the 0th and the -1st, oscillating at frequency $\omega_{\mathrm{m}} / 2$ and $-\omega_{\mathrm{m}} / 2$ , respectively (see illustration in Fig. 20(a)). As discussed in Section 2.2, this approximation typically works well (see also Fig. 7(a)). Moreover, the proposed theory can be further generalized to cases where the number of dominant modes is four or more. Therefore, from $(11)$ , we write the real-valued electric field of an eigenmode in the following form:

$$
\mathbf {E} (z, t) = A _ {1} (t) e ^ {j \omega_ {\mathrm{m}} / 2 t} e ^ {- j k z} \mathbf {a} _ {x} + A _ {2} ^ {*} (t) e ^ {- j \omega_ {\mathrm{m}} / 2 t} e ^ {- j k z} \mathbf {a} _ {x} + \mathrm{c.c.},\tag{84}
$$

where “c.c.” denotes the complex conjugated terms, k denotes the real-valued wavenumber of the mode that is assumed to be inside the momentum bandgap, and $A_{1}(t)$ and $A_{2}^{*}(t)$ are the slowly varying (compared with $\exp(j\omega_{\mathrm{m}}/2t)$ ) unknown temporal mode envelopes which define the temporal evolution of the modes inside the momentum bandgap. Note that in Eq. (84) we write the $A_{2}^{*}(t)$ -mode as complex conjugated to follow the same notation as in Section 4.2 for an easier comparison. We aim to determine $A_{1}(t)$ and $A_{2}^{*}(t)$ .

As in the previous subsections and sections, we assume here that the permittivity is modulated according to $\varepsilon(t) = \varepsilon_{\mathrm{av}}(1 + m_{\varepsilon}\cos (\omega_{\mathrm{m}}t + \phi_{\mathrm{m}})) = \varepsilon_{\mathrm{av}}(1 + m_{\varepsilon}(e^{j\omega_{\mathrm{m}}t + j\phi_{\mathrm{m}}} + e^{-j\omega_{\mathrm{m}}t - j\phi_{\mathrm{m}}}) / 2)$ with some arbitrary modulation phase $\phi_{\mathrm{m}}$ . By substituting the modulation function and the ansatz equation (84) into the wave equation (13) and dropping the higher-order harmonics oscillating at $3\omega_{\mathrm{m}} / 2$ , we obtain [140, Supplementary Material]

$$
\begin{array}{r l} - \frac {k ^ {2} c ^ {2} A _ {1} (t)}{\varepsilon_ {\mathrm{av}}} & = \frac {\mathrm{d} ^ {2} A _ {1} (t)}{\mathrm{d} t ^ {2}} + 2 j \frac {\omega_ {\mathrm{m}}}{2} \frac {\mathrm{d} A _ {1} (t)}{\mathrm{d} t} - \frac {\omega_ {\mathrm{m}} ^ {2}}{4} A _ {1} (t) \\ & + \frac {m _ {\varepsilon} e ^ {j \phi_ {\mathrm{m}}}}{2} \left[ \frac {\mathrm{d} ^ {2} A _ {2} ^ {*} (t)}{\mathrm{d} t ^ {2}} + 2 j \frac {\omega_ {\mathrm{m}}}{2} \frac {\mathrm{d} A _ {2} ^ {*} (t)}{\mathrm{d} t} - \frac {\omega_ {\mathrm{m}} ^ {2}}{4} A _ {2} ^ {*} (t) \right], \end{array}\tag{85}
$$

$$
\begin{array}{l} - \frac {k ^ {2} c ^ {2} A _ {2} (t)}{\varepsilon_ {\mathrm{av}}} = \frac {\mathrm{d} ^ {2} A _ {2} (t)}{\mathrm{d} t ^ {2}} + 2 j \frac {\omega_ {\mathrm{m}}}{2} \frac {\mathrm{d} A _ {2} (t)}{\mathrm{d} t} - \frac {\omega_ {\mathrm{m}} ^ {2}}{4} A _ {2} (t) \\ \qquad + \frac {m _ {\varepsilon} e ^ {j \phi_ {\mathrm{m}}}}{2} \left[ \frac {\mathrm{d} ^ {2} A _ {1} ^ {*} (t)}{\mathrm{d} t ^ {2}} + 2 j \frac {\omega_ {\mathrm{m}}}{2} \frac {\mathrm{d} A _ {1} ^ {*} (t)}{\mathrm{d} t} - \frac {\omega_ {\mathrm{m}} ^ {2}}{4} A _ {1} ^ {*} (t) \right]. \end{array}\tag{86}
$$

The complex conjugate terms in Eq. (84) yield the same equations as Eqs. (85) and (86). It should be noted that from a single wave equation, we ended up with two equations, i.e., Eqs. (85) and (86), because even in the parametric degenerate case, photons at frequencies $\omega_{m}/2$ and $-\omega_{m}/2$ should be distinguished. This leads to one coupled equation for each kind of photon.

## Figure 20

![](images/01d7a31594ca66454f2f334dc6f5337a556ee56d7c0691e02a03b27dfde16151.jpg)

(b)
![](images/18927a16e4fc816d7fe81ef3c433395dd6db934583b197189ae6c4eb40860e12.jpg)

(c)
![](images/9b318d08093d2256a149cd5c2c45a4f14e3acbf95dd34865a611c4153deb9706.jpg)
(a) Two dominant modes inside a momentum bandgap of a PTC. They correspond to plane waves at frequency $\omega_{m}/2$ propagating in the opposite directions. (b) Temporal evolution of the envelope amplitudes of the two modes shown in (a). The initial conditions are given by $A_{1}(0) \neq 0$ and $A_{2}^{*}(0) = 0$ (fields before the temporal modulation started corresponded to a single propagating wave). (c) Phase-sensitivity analysis of a PTC with initial conditions in the form $A_{2}^{*}(0) = A_{1}(0)e^{j\phi_{\mathrm{init}}}$ (fields before the temporal modulation started corresponded to a standing wave).

Now, we can use the fact that the same medium without temporal modulation, that is, when $\Delta\varepsilon\to0$ and $\varepsilon(t)=\varepsilon_{\mathrm{av}}$ , has the dispersion relation of the conventional form

$$
k ^ {2} c ^ {2} = \varepsilon_ {\mathrm{av}} \omega_ {\mathrm{st}} ^ {2},\tag{87}
$$

which can be observed, e.g., from Fig. 3. Here, we add a small imaginary part to the eigenfrequency in the stationary medium $\omega_{\mathrm{st}} = \frac{\omega_{\mathrm{m}}}{2} + jg$ , where $g \ll \frac{\omega_{\mathrm{m}}}{2}$ , to consider a possible small absorption in the medium. Note that $k$ is assumed to be real-valued. By substituting $\frac{\omega_{\mathrm{m}}}{2} = \omega_{\mathrm{st}} - jg$ inside Eq. (86) and using Eq. (87), we obtain [140, Supplementary Material]

$$
\begin{array}{l} \frac {\mathrm{d} ^ {2} A _ {1} (t)}{\mathrm{d} t ^ {2}} + 2 j (\omega_ {\mathrm{st}} - j g) \frac {\mathrm{d} A _ {1} (t)}{\mathrm{d} t} + g (2 j \omega_ {\mathrm{st}} + g) A _ {1} (t) \\ + \frac {m _ {\varepsilon} e ^ {j \phi_ {\mathrm{m}}}}{2} \left[ \frac {\mathrm{d} ^ {2} A _ {2} ^ {*} (t)}{\mathrm{d} t ^ {2}} + 2 j (\omega_ {\mathrm{st}} - j g) \frac {\mathrm{d} A _ {2} ^ {*} (t)}{\mathrm{d} t} - (\omega_ {\mathrm{st}} - j g) ^ {2} A _ {2} ^ {*} (t) \right] = 0, \end{array}\tag{88}
$$

$$
\begin{array}{l} \frac {\mathrm{d} ^ {2} A _ {2} (t)}{\mathrm{d} t ^ {2}} + 2 j (\omega_ {\mathrm{st}} - j g) \frac {\mathrm{d} A _ {2} (t)}{\mathrm{d} t} + g (2 j \omega_ {\mathrm{st}} + g) A _ {2} (t) \\ + \frac {m _ {\varepsilon} e ^ {j \phi_ {\mathrm{m}}}}{2} \left[ \frac {\mathrm{d} ^ {2} A _ {1} ^ {*} (t)}{\mathrm{d} t ^ {2}} + 2 j (\omega_ {\mathrm{st}} - j g) \frac {\mathrm{d} A _ {1} ^ {*} (t)}{\mathrm{d} t} - (\omega_ {\mathrm{st}} - j g) ^ {2} A _ {1} ^ {*} (t) \right] = 0. \end{array}\tag{89}
$$

Next, by using $g \ll |\omega_{st}|$ and subsequently imposing the slowly varying envelope approximation, which requires that $\frac{\mathrm{d}A_{1,2}(t)}{\mathrm{d}t} \ll \omega_{\mathrm{st}} A_{1,2}(t)$ and $\mathrm{d}^{2} A_{1,2}(t)/\mathrm{d}t^{2} \ll \omega_{\mathrm{st}} (\mathrm{d} A_{1,2}(t)/\mathrm{d}t)$ , we obtain

$$
\begin{array}{l} \frac {\mathrm{d}}{\mathrm{d} t} A _ {1} (t) = - g A _ {1} (t) - j \eta_ {\mathrm{PTC}} A _ {2} ^ {*} (t), \\ \frac {\mathrm{d}}{\mathrm{d} t} A _ {2} ^ {*} (t) = - g A _ {2} ^ {*} (t) + j \eta_ {\mathrm{PTC}} ^ {*} A _ {1} (t), \end{array}\tag{90}
$$

where

$$
\eta_ {\mathrm{PTC}} = \frac {m _ {\varepsilon} \omega_ {\mathrm{st}}}{4} e ^ {j \phi_ {\mathrm{m}}}.\tag{91}
$$

We can see that coupled-mode equations of a PTC inside the bandgap (90) closely resemble those of a degenerate parametric amplification in nonlinear optics, Eq. (81) (especially in the case of lossless medium when g = 0). In the present case, the roles of signal and idler photons are played by the 0th and -1st harmonics, respectively. Instead of evolution in space along the z-axis, the modes' envelopes now evolve in time.

It should be noted that Eqs. (90) were derived assuming that the two dominant modes in the PTC have the same frequencies, corresponding to the degenerate case (see Eq. (84)). In the nondegenerate case, the coupled-mode equations will have an exponential factor similar to that in Eq. (78) with the argument proportional to the frequency difference of the two dominant modes [163]. This implies that PTCs also have a phase-matching condition. However, in contrast to Eq. (81), here the phase mismatch occurs in time rather than space. Nevertheless, satisfying the phase-matching condition in PTCs is much simpler in practice since one only needs to ensure that the signal frequency equals exactly half the modulation frequency (no need to use the effect of birefringence).

Next, making the replacement $j\eta_{\mathrm{PTC}} = |\eta_{\mathrm{PTC}}|e^{j\psi_{\mathrm{PTC}}}$ , we solve system Eq. (90) regarding $A_{1}(t)$ and $A_{2}^{*}(t)$ :

$$
\binom{A _ {1} (t)}{A _ {2} ^ {*} (t)} = e ^ {- g t} \left( \begin{array}{c c} \cosh | \eta_ {\mathrm{PTC}} | t & - e ^ {j \psi_ {\mathrm{PTC}}} \sinh | \eta_ {\mathrm{PTC}} | t \\ - e ^ {- j \psi_ {\mathrm{PTC}}} \sinh | \eta_ {\mathrm{PTC}} | t & \cosh | \eta_ {\mathrm{PTC}} | t \end{array} \right) \cdot \binom{A _ {1} (0)}{A _ {2} ^ {*} (0)}.\tag{92}
$$

Here, $A_{1}(0)$ and $A_{2}^{*}(0)$ are the initial conditions for the mode envelope amplitudes. The zero within the function arguments denotes an arbitrarily chosen reference point in time, highlighting the relative nature of the time axis that lacks an absolute scale. In contrast to Eq. (81), where the signal and idler waves in the degenerate scenario were inevitably coupled in the initial excitation, in the PTC case, modes $A_{1}(t)$ and $A_{2}^{*}(t)$ are oppositely propagating plane waves along the z-axis, which could be easily decoupled and excited separately. Specifically, it is feasible to configure a PTC such that, prior to introducing temporal modulations, a single plane wave propagates in the medium with minimized reflections at any spatial interface. This setup corresponds to a temporally finite PTC, as discussed in Section 3.3. In this setup, initial conditions such as $A_{1}(0) \neq 0$ and $A_{2}^{*}(0) = 0$ are naturally attainable. Then, the solution is given by

$$
\begin{array}{l} A _ {1} (t) = A _ {1} (0) e ^ {- g t} \cosh | \eta_ {\mathrm{PTC}} | t, \\ A _ {2} ^ {*} (t) = - A _ {1} (0) e ^ {- g t} e ^ {- j \psi_ {\mathrm{PTC}}} \sinh | \eta_ {\mathrm{PTC}} | t. \end{array}\tag{93}
$$

Figure 20(b) depicts this solution. It is clear that similar to Eq. (80), both $A_{1}(t)$ and $A_{2}^{*}(t)$ modes experience monotonic exponential growth assuming that $g$ is sufficiently small (compare with Fig. 19(c)). However, this exponential growth is independent of the modulation phase $\phi_{\mathrm{m}}$ . Therefore, PTCs provide a phase-insensitive parametric amplification under the chosen initial conditions, even in the degenerate case. This is in stark contrast to the optical parametric amplification considered in Section 4.2. One can also observe from Eq. (93) that after a sufficiently long time duration ( $|\eta_{\mathrm{PTC}}|t \gg 1$ ), the modes become related as $A_{2}^{*}(t) = A_{1}(t)e^{-j\psi_{\mathrm{PTC}} + j\pi}$ . This fact is discussed further at the end of the section.

Nevertheless, one can also identify initial conditions that provide phase-sensitive amplification inside the bandgap of a PTC. To see this, let us diagonalize the matrix in Eq. (92). By doing this and replacing mode envelopes according to

$$
\begin{array}{r} A _ {\mathrm{PTC}} ^ {(1)} (t) = \frac {1}{2} \left[ A _ {1} (t) e ^ {- j \psi_ {\mathrm{PTC}} / 2} + A _ {2} ^ {*} (t) e ^ {j \psi_ {\mathrm{PTC}} / 2} \right], \\ A _ {\mathrm{PTC}} ^ {(2)} (t) = - \frac {1}{2} \left[ A _ {1} (t) e ^ {- j \psi_ {\mathrm{PTC}} / 2} - A _ {2} ^ {*} (t) e ^ {j \psi_ {\mathrm{PTC}} / 2} \right], \end{array}\tag{94}
$$

we obtain the final solution for the modified mode envelopes:

$$
\left[ \begin{array}{c} A _ {\mathrm{PTC}} ^ {(1)} (t) \\ A _ {\mathrm{PTC}} ^ {(2)} (t) \end{array} \right] = e ^ {- g t} \left[ \begin{array}{c c} e ^ {- | \eta_ {\mathrm{PTC}} | t} & 0 \\ 0 & e ^ {| \eta_ {\mathrm{PTC}} | t} \end{array} \right] \cdot \left[ \begin{array}{c} A _ {\mathrm{PTC}} ^ {(1)} (0) \\ A _ {\mathrm{PTC}} ^ {(2)} (0) \end{array} \right].\tag{95}
$$

One can note a striking equivalence of Eq. (95) with Eq. (83) when $g = 0$ .

If one wants to find the solution where a PTC regardless of phase $\phi_{\mathrm{m}}$ always exhibits parametric de-amplification (even when $g = 0$ ), one needs to satisfy $A_{\mathrm{PTC}}^{(2)}(0) = 0$ and $A_{\mathrm{PTC}}^{(1)}(0) \neq 0$ . Then, from Eq. (94), one finds that the initial condition must be of the form $A_2^*(0) = e^{-j\psi_{\mathrm{PTC}}} A_1(0)$ , which implies that prior to the temporal modulations, there was a standing-wave excitation in the medium.

To find the mode envelope evolution for an arbitrary initial standing-wave excitation, let us substitute in Eq. (95) initial conditions in the form $A_2^*(0) = A_1(0)e^{j\phi_{\mathrm{init}}}$ with some arbitrary phase $\phi_{init}$ . The solution then reads

$$
\begin{array}{l} A _ {\mathrm{PTC}} ^ {(1)} (t) = \frac {1}{2} e ^ {- g t - | \eta_ {\mathrm{PTC}} | t} \left[ e ^ {- j \psi_ {\mathrm{PTC}} / 2} + e ^ {j \phi_ {\mathrm{init}}} e ^ {j \psi_ {\mathrm{PTC}} / 2} \right] A _ {1} (0), \\ A _ {\mathrm{PTC}} ^ {(2)} (t) = - \frac {1}{2} e ^ {- g t + | \eta_ {\mathrm{PTC}} | t} \left[ e ^ {- j \psi_ {\mathrm{PTC}} / 2} - e ^ {j \phi_ {\mathrm{init}}} e ^ {j \psi_ {\mathrm{PTC}} / 2} \right] A _ {1} (0). \end{array}\tag{96}
$$

Although the first mode is always decaying, the second (dominant) mode can be exponentially growing given that $g<|\eta_{PTC}|$ and the expression in the square brackets is nonzero. Figure 20(c) depicts the time evolution of the dominant mode in the case when g=0. We can see that the phase $\psi_{PTC}$ (which is a function of the modulation phase $\phi_{m}$ according to Eq. (91)) strongly affects the parametric amplification in the PTC. Thus, parametric amplification in a PTC with the initial excitation corresponding to a standing wave is strongly phase-sensitive and is similar to that in the degenerate optical parametric amplifier described by Eq. (83). Interestingly, this fact of the phase-sensitivity was exploited in Ref. [122]. In practice, standing-wave initial excitation can be naturally achieved when the PTC is surrounded at least from one side by a highly reflective surface, such as metal backing.

It is interesting to look back at the solution given by Eq. (93) for the initial condition when $A_{1}(0) \neq 0$ and $A_{2}^{*}(0) = 0$ . As was mentioned before, after the transient time duration $t_{\mathrm{trans}}$ such that $|\eta_{\mathrm{PTC}}|t_{\mathrm{trans}} \gg 1$ , the modes inside the PTC form a standing wave described by $A_{2}^{*}(t) = A_{1}(t)e^{-j\psi_{\mathrm{PTC}} + j\pi}$ . In the present notation of a standing wave $A_{2}^{*}(t_{\mathrm{trans}}) = A_{1}(t_{\mathrm{trans}})e^{i\phi_{\mathrm{init}}}$ , this state is described by $\psi_{\mathrm{PTC}} + \phi_{\mathrm{init}} = \pi$ . It corresponds to the fastest growing (purple) line in Fig. 20(c).

Thus, in contrast to the degenerate difference-frequency generation in nonlinear optics, we see that for PTCs, parametric amplification (measured after a sufficient number of wave cycles) depends on the modulation phase only in a special case of initial excitation, that is, when a pure standing wave was inside the material prior to the temporal modulations. When the initial excitation corresponds to a single plane wave, amplification in PTCs is phase-insensitive. This distinction in phase-sensitivity compared with that in nonlinear optics arises because, in the latter scenario, signal and idler photons at the input stage are indistinguishable. Conversely, in the context of PTCs, one can readily generate an input that comprises exclusively one mode (single propagating wave). In addition, as mentioned, the phase-matching condition in PTCs is much easier to satisfy in practice than that in the optical parametric amplification process. Finally, while in the former process the amplification occurs in space, in the latter process it occurs in time. This could be an advantage of PTCs in some situations, as they can provide high amplification even in compact geometries.

Finally, it should be noted that in a recent work $[163]$ , an important comparison was made between the PTC regime and the process of backward optical parametric amplification for the case of transverse pumping $[164,165]$ . It was pointed out that although the geometry and the eigenmodes of the medium are similar in both cases, the two parametric processes have qualitative differences due to the different initial conditions. In particular, the backward optical parametric amplification process has a finite growth of light energy (due to the oscillatory nature of light propagation), whereas there is an exponential growth of the modes in PTCs.

## 4.4. Comparison of Time Crystals and PTCs

PTCs are fundamentally different from time crystals in condensed matter physics. In this subsection, we first explain the basic concepts of time crystals and then discuss their differences from PTCs.

Time crystals, first conceived by Frank Wilczek in 2012 [22], are to some extent dual to spatial crystals. While spatial time crystals have a periodic pattern in space, and they break the continuous spatial translational symmetry due to the periodic repeating patterns, in time crystals, the particles oscillate periodically in time in a harmonic manner, which breaks continuous time translational symmetry [22]. Time crystals represent a new phase of matter, which is of fundamental importance to material science. As they bear similarity to the PTCs considered in this tutorial, we would like to elaborate on them shortly to distinguish them clearly.

The originally proposed time crystals by Frank Wilczek were supposed to oscillate at their ground state in thermal equilibrium in a closed system, without energy exchange with the external surroundings. Such oscillation is spontaneous and is caused by the interactions of the particles themselves. In time crystals, the Hamiltonian does not change in time, whereas the particles periodically oscillate in time and persist forever. However, it was later demonstrated by Watanabe and Oshikawa that such a self-sustained motion quantum system is fundamentally forbidden $[166]$ since it violates the laws of thermodynamics and quantum mechanics. In 2017, Yao et al. pointed out that in a nonequilibrium open system pumped by a periodic drive, the persistent oscillation of particles is possible $[167]$ . In other words, if energy exchange with the external environment (open system) is allowed, the idea of a time crystal can still be realized. These are called discrete time crystals because they exhibit discrete time translational symmetry. External periodic forces should drive discrete time crystals. This force keeps the system out of equilibrium. However, the system's response to this driving force exhibits a periodicity that is different (usually an integer multiple) from that of the driving force itself. This incoherence of the particle oscillation and the external pump indicates that the oscillation of particles is mainly caused by the interactions of particles themselves rather than directly driven by the external pump. It is a kind of semi-spontaneous oscillation. The most important feature of a discrete time crystal is that it is robust to the interaction strength among particles and the imperfection in the driving pulses $[167,168]$ , showing the characteristics of a new phase of matter.

Discrete time crystals with external periodic driving are different from the originally proposed time crystal by Wilczek, which is a spontaneous process without any dependence on external driving. Although Wilczek's time crystals are impossible to realize, it is possible to step closer to them if the external driving is present (to respect the no-go theorem and provide energy to the system) but is independent of time, for example, driven by a DC pump. This is called a continuous time crystal because the pump is not periodically repeating but continuous and constant in time. With such a time-invariant pump, the particles oscillate periodically, which is purely determined by the interactions among particles. The oscillation is independent of the pump, which only compensates for the intrinsic loss of the material during oscillation. A continuous time crystal has been observed experimentally in many quantum systems (see, e.g., [169]).

A recent work demonstrated that continuous time crystals can form in a classical mechanic–photonic coupled system $[170]$ . There, the researchers created a 2D array of plasmonic metamolecules (see Fig. 21). These are tiny, gold-coated resonant structures placed on mechanically flexible nanowires cut from a semiconductor membrane. The nanowires were illuminated with laser light at a frequency that matched the plasmonic resonance of the metamolecules and many orders of magnitude higher than the frequency of mechanical nanowire oscillations, effectively representing a continuous pump. The continuous beam of laser light heats the nanowires, causing them to start oscillating due to thermal expansion. These oscillations are random at first but become quickly synchronized across the array of nanowires due to the effect of spontaneous synchronization. This synchronization is a key feature, as it represents the ordered, repeating pattern in time that represents a time crystal. As the nanowires oscillate, they change probe light transmission through the system. The synchronized oscillations can be observed directly by measuring these changes in light reflectivity.

![](images/ab8e12eb15e1c100f3e0dde51f9902a11e19b687c3ec9fab8482e261d362511e.jpg)
(a) Artistic representation of the fundamental unit of a classical continuous time crystal, synthesized within a photonic metamaterial. (b) Scanning electron microscope imagery capturing the entire 2D array of metamolecules. When illuminated with coherent light (depicted schematically by the overlaid laser spot), the system transitions into a state characterized by persistent, synchronized oscillations of nanowires. (a) and (b) Reprinted from [170] under a Creative Commons license.

It is important to note that such a system emulates a continuous time crystal because the oscillations induced in the metamaterial are at a much lower frequency than the pump light itself. This allows the system to effectively “average out” the pump’s periodicity over the time scale of the oscillations, making the pump appear continuous. As a result, the metamaterial transitions to a state showing persistent and synchronized oscillations, which are the defining features of a continuous time crystal. This innovative approach extends the concept of time crystals to photonic systems, where the interaction between light and matter can create new phases of matter with unique temporal properties.

The combination of both spatial and temporal translational symmetry breaking defines a space–time crystal, exhibiting periodicity in both space and time. This phenomenon has recently been observed experimentally in a magnon platform $[171]$ . In this magnonic system, the oscillations of each particle in space have different phases, leading to a periodic modulation in both dimensions. This modulation results in the formation of a magnonic band structure due to the back folding of modes at the Brillouin zone boundaries of the space–time crystal. The study also demonstrates interactions between magnons and the space–time crystal, leading to lattice scattering and the generation of ultrashort spin waves that cannot be described by classical dispersion relations for linear spin wave excitation.

Unlike condensed-matter “time crystals,” a PTC is a medium with time-varying macroscopic optical properties. To achieve these time-varying periodic optical properties, an external periodic pump is required, similar to discrete time crystals. However, from a microscopic view, in discrete time crystals, particles generally oscillate at a frequency different from that of the external pump. The interactions among the atoms themselves mainly determine this oscillation. In contrast, in PTCs, the material's particles are driven directly by an external pump and oscillate in synchronization with it.

## 5. MATERIAL PLATFORMS TO REALIZE PTCs

The experimental realization of PTCs is challenging, but recent advances have shown promising potential for the fabrication of such devices. We review different types of material platforms that are being used for the realization of PTCs and the observation of their wave phenomena. We start this section by discussing initially a material platform operating at radio frequencies where PTCs can be implemented using transmission lines. The possibility of reliably changing their properties on time scales comparable to the oscillation period of the electromagnetic fields renders them excellent candidates to study many of the fundamental effects that were just discussed on experimental grounds. Then, we discuss material platforms to observe comparable effects at optical frequencies. Next, we concentrate on metasurfaces made from time-varying photonic materials: so, essentially, thin films only. They are easier to realize and interrogate experimentally, which allows for a better understanding of the properties of PTCs. In the last section, we highlight that not just electromagnetic fields can propagate in systems with time-varying material properties. Other waves, such as acoustic, elastic, or water waves, can also be used to study waves in time-varying media. Finally, more exotic material systems supporting synthetic dimensions can be envisioned, which we discuss here briefly. This should provide a comprehensive overview of the material systems used to study PTCs.

## 5.1. Transmission Lines

Modulating the material parameters, such as the permittivity, remains challenging in many frequency ranges. As discussed in Section 3.1c, the change in resonance frequency would require a change in the properties of the atoms that make up the materials. It is easy to appreciate that this is rather demanding. However, when instead of an actual atom, a meta-atom is considered, and if the required modulation frequency of the material properties is fairly small and accessible by electronic means, we can get excellent control over a system that offers us the desired effects.

Indeed, at radio frequencies, it is possible to use time-varying transmission lines to emulate a time-varying medium $[47,48,114,172]$ . That is possible because the governing equations that express the voltage along the circuit can be mapped to an ordinary wave equation in optics. Moreover, modulating a transmission line is reasonably easy by considering a time-dependent capacitance as part of the transmission line. Voltage-controlled varactor diodes provide such a time-dependent capacitance. With that, all ingredients are at hand to study PTCs.

Figure 22 shows a typical transmission line that supports the propagation of voltage–current waves. The transmission line is composed of distributed inductances (in series) L with the unit of farads per meter (F/m) and capacitance (shunt) C with the unit of henries per meter (H/m) [88, Section 2.1]. The series inductance L represents the total self-inductance of the two conductors of the transmission line, and the shunt capacitance C is the capacitance due to the close proximity of the two conductors. By multiplying both quantities by $\Delta z$ (infinitesimal length along the transmission line), we would obtain conventional (lumped) capacitance and inductance. For simplicity, we assume the system is lossless, and the capacitance is modulated in time as a function of $C(t)$ .

## Figure 22

![](images/627116ab986310102dc629b659af1f64422d0f195752e92e00dcdd71a0f59e85.jpg)
Transmission line with lumped elements extended along the z direction. The distributed capacitance is modulated in time. The transmission line is electromagnetically equivalent to a material with time-varying permittivity. In other words, voltage waves propagate along the transmission line in the same manner as electromagnetic waves propagate in bulk material with time-varying permittivity. Here, $\Delta z$ is the length of an infinitesimal section of the transmission line.

According to Kirchhoff's voltage and current laws, the circuit displayed in Fig. 22 must satisfy the following relations,

$$
V (z, t) - L \Delta z \frac {\partial I (z , t)}{\partial t} - V (z + \Delta z, t) = 0,\tag{97a}
$$

and

$$
I (z, t) - \Delta z \frac {\partial}{\partial t} [ C (t) V (z + \Delta z, t) ] - I (z + \Delta z, t) = 0.\tag{97b}
$$

Taking the limit of $\Delta z\to 0$ , Eqs. (97) can be rearranged in differential forms, and becomes

$$
\frac {\partial V (z , t)}{\partial z} = - L \frac {\partial I (z , t)}{\partial t}\tag{98a}
$$

and

$$
\frac {\partial I (z , t)}{\partial z} = - \frac {\partial [ C (t) V (z , t) ]}{\partial t}.\tag{98b}
$$

The above equations have exactly the same form as the Faraday and Ampére laws in Maxwell's equations when written for a wave propagating in a bulk medium with absolute permeability $\mu$ and permittivity $\varepsilon(t)$ along the $z$ direction with a given polarization:

$$
\frac {\partial E (z , t)}{\partial z} = - \mu \frac {\partial H (z , t)}{\partial t},\tag{99a}
$$

$$
\frac {\partial H (z , t)}{\partial z} = - \frac {\partial [ \varepsilon (t) E (z , t) ]}{\partial t}.\tag{99b}
$$

Thus, the governing electromagnetic equations for a lossless transmission line with time-varying distributed capacitance correspond to those for bulk dielectric material with time-varying permittivity. The equivalence of the circuit components and the material parameters suggests that

$$
C (t) \sim \varepsilon (t), L \sim \mu .\tag{100}
$$

Indeed, this equivalence allows us to study some of the wave phenomena of bulk PTCs using 1D transmission lines with time-varying parameters. For example, the momentum bandgap of a PTC was observed experimentally using transmission lines in Refs. [47,48]. In those works, the transmission line was formed by a microstrip line whose lumped capacitances were realized with varactors (tunable capacitors). The setup comprised a finite-sized PTC consisting of eight unit cells, and the varactors were

Figure 23

microstrip line loaded with varactors
![](images/79d5a0348768bd244007b713bb6e247f0d01d79656b00c3c727d76f90966ba21.jpg)
(a)

![](images/0896fd6bf32698c135ee2e9a568573a879cc72df2178f41222ed691fb0dad96e.jpg)
(b)
Experimental demonstration of PTCs using a microstrip line loaded with varactors. (a) Photograph of an experiment prototype. (b) Experimentally determined band structure. The band structure is determined by measuring the allowed wavenumbers $k_{1}$ and $k_{2}$ when altering the input signal frequency $\omega_{0}$ . Here, a is the spatial period of the transmission line. (a) © 2016 IEEE. Reprinted, with permission, from Reyes-Ayona and Halevi, IEEE Trans. Microwave Theory Tech. 64, 3449–3459 (2016) [48]. (b) Reprinted with permission from Reyes-Ayona and Halevi, Appl. Phys. Lett. 107, 074101 (2015) [47]. Copyright 2015, AIP Publishing LLC.

controlled with the same signal to ensure the in-phase time variation of all components (see Fig. 23(a)). That should guarantee the spatial homogeneity of the PTC. In the reported experiment, a voltage signal wave was launched along the transmission line whose frequency $\omega = \omega_{0}$ could be varied. Then, the phases of the voltage in each unit cell are measured to determine the allowed eigenwavenumbers $k_{1}$ and $k_{2}$ that belong to the two lowest bands of the band diagram (see Fig. 23(b)). By varying the signal frequency (for a given modulation frequency), the dispersion relation was extracted experimentally, leading to the functional dependency between $\omega$ and k. The band structure contains a momentum bandgap in the momentum space. Whereas these experiments convincingly demonstrate the core aspect of PTCs, the possibility of wave amplification inside the momentum bandgap has not been reported in these earlier works.

## 5.2. Optical Materials

The creation of PTCs operating at optical frequencies is very challenging since the modulation must be at optical speeds and have sufficiently strong amplitude $[60]$ . Indeed, the relative width of the momentum bandgaps in a PTC is proportional to the relative modulation strength, as was discussed in Section 3.1d. Various mechanisms can conceivably modify the refractive index in bulk materials. However, the majority, including electro-optic, acousto-optic, and thermo-optic mechanisms, are constrained by their operational speeds $[125]$ . Consequently, these methods are deemed unsuitable for realizing PTCs at optical frequencies.

Optical modulation has emerged as the most prominent method in this context. The most natural way to change the refractive index at optical speeds is probably to exploit the Kerr-type nonlinearity. The Kerr nonlinearity is a third-order nonlinear effect occurring in many centrosymmetric materials. In the Kerr effect, a strong pump beam changes the refractive index of a nonlinear medium through which it propagates. By changing the pump intensity in time (e.g., by using pulsed pump excitations), one can obtain modulation of the refractive index of the material. When a weak probe light with electric field $E(\omega')$ propagates through the nonlinear medium illuminated by a pump beam with amplitude $E(\omega)$ , we obtain the nonlinear polarization $P^{\mathrm{NL}}(\omega) = 6\epsilon_0\chi^{(3)}(\omega' = \omega' + \omega - \omega)|E(\omega)|^2E(\omega')$ [157, Section 4.1], where $\chi^{(3)}$ is the third-order nonlinear susceptibility. In this context, a cross coupling between the pump and probe is observed, and the refractive index of the medium experienced by the probe light can be expressed as $n = n_0 + 2n_2|E(\omega)|^2$ (note that $n$ here denotes the refractive index and should not be confused with the harmonic number), where $n_0$ is the linear and $n_2 = \frac{3\chi^{(3)}}{2n_0}$ is the nonlinear refractive index. The optical Kerr effect is nearly instantaneous in the sense that the response of the material's refractive index to changes in pump intensity occurs extremely rapidly (on the femtosecond level and faster). Thus, to induce a PTC in the nonlinear material, one needs to be able to change the intensity of the pump beam quickly in time so that it leads to the periodic modulation of the refractive index. However, there are two challenges in practical setups with nonlinear materials. The first is related to the fact that most materials have small nonlinear susceptibilities. Indeed, the maximum observed change of the relative modulation strength of the refractive index is in the range of $1\%$ , even at power densities as high as $1\mathrm{TW/cm^2}$ [173]. With such modulation depths, the momentum bandgap becomes negligible. The second challenge is to change the pump intensity at optical speed in a periodic manner. Although ultrashort pulse excitation is nowadays technologically accessible, making a train of many such pulses with a high duty cycle (high repetition rate) is not feasible. Current technology allows obtaining repetition rates in trains of pulses up to $10\mathrm{GHz}$ [174].

Therefore, so far, the experimental developments have been limited to single time interfaces (requiring a single ultrashort pump pulse) in a special type of materials supporting large modulation strengths, so-called ENZ materials $[175]$ . Since permittivity and refractive index are related as $n = \sqrt{\varepsilon}$ , differentiating n with respect to $\varepsilon$ , we can find that $\Delta n = \frac{\Delta \varepsilon}{2\sqrt{\varepsilon}}$ . From that expression, we see that since $\varepsilon$ approaches zero in ENZ materials, small variations in $\Delta \varepsilon$ can induce significant changes in $\Delta n$ $[62]$ . The ENZ regime occurs close to the plasma frequency of a material whose dielectric function is described by a Drude model. However, to use this effect, one needs to ensure that the imaginary part of the material permittivity is sufficiently small. Typical materials discussed in the literature for that purpose are TCOs $[176]$ such as ITO and AZO and others $[62,65,126,177–179]$ . They have the benefit that the ENZ region can be suitably tuned, but it is roughly at telecommunication wavelengths, which is an asset for potential applications. In the ENZ wavelength region, the imaginary part of the permittivity in TCOs is in the range of 0.2–0.4 $[62,65,126]$ . The nonlinear refractive index of ITO at its ENZ wavelength of 1240 nm reaches $n_{2} = 0.11 \, cm^{2}/GW$ $[62]$ , whereas for AZO at 1390 nm $n_{2} = 3.07 \times 10^{-4} \, cm^{2}/GW$ $[65]$ . A comprehensive overview of nonlinear refractive indices of different materials can be found in Ref. $[84]$ .

It should be noted that the nonlinearities, represented by $n_{2}$ , in TCOs originate differently compared to those in conventional Kerr nonlinear materials. This distinction arises from the nonparabolic dispersion of the conduction band, which leads to a change in the average effective mass of the electron sea due to intraband absorption [68,180]. Significant changes in refractive index $\Delta n/n$ , of the order of 100%, have indeed been observed in these materials [178]. In early experimental studies, rapid changes (on the order of a few hundred femtoseconds) in the refractive index were observed [64,110,181]. However, these changes were still two to three orders of magnitude slower than necessary for achieving PTCs in the optical (infrared or visible) regime. As a result, this type of nonlinearity was initially termed “slow” [68]. More recent theoretical work has suggested that refractive index changes in TCOs could be nearly instantaneous [182,183]. Subsequent experimental studies have confirmed such ultrafast modulations, showing excitation and relaxation times of the refractive index in ITO in the order of 5–10 fs and 10–20 fs, respectively [66,184], limited only by the pump-pulse duration. These findings represent a significant step toward the implementation of PTCs at optical frequencies.

Periodic optical temporal modulation of ENZ materials using the nonlinearity mechanism described above (arising from the nonparabolic dispersion) remains impractical due to unrealistically high power requirements for modulating TCOs at optical speeds. It was theoretically predicted that to obtain a moderately wide momentum bandgap, one would need pump power density on the order of tens of TW/cm $^{3}$ even for PTCs operating in the terahertz frequency range [61]. Such intense pump power, compounded by the nonzero dissipation inherent to the materials, would invariably result in degradation of the materials, precluding the observation of PTCs.

Furthermore, TCOs are also capable of supporting another type of nonlinearities where temporal modulations are generated entirely through optical means via the third-order susceptibility, $\chi^{(3)}$ , in the four-wave mixing process [68]. Specifically, in degenerate four-wave mixing, two counter-propagating pump beams interact with a signal wave to generate a third-order nonlinear polarization [185]. During this process, the two pump waves combine to create a uniform intensity pattern oscillating at the frequency $2\omega$ . This oscillation modulates the material's refractive index at the same frequency, thereby offering the potential to form a PTC. However, such effects are typically overshadowed by other more dominant nonlinear phenomena, such as the “slow dynamic grating” [61,68,186].

To summarize, the realization of optical PTCs remains a significant challenge, necessitating further experimental and theoretical exploration. We wish to highlight several innovative approaches that have been proposed recently to facilitate the observation of PTCs within the optical domain. The first approach involves utilizing a second-order nonlinear process, where a standing pump wave at the second harmonic frequency $2\omega$ interacts with counter-propagating waves at the fundamental frequency $[61,164,165]$ . This mechanism is instantaneous and remains unaffected by the spurious “dynamic-grating” effects. In addition, the pump wave is confined within a resonant cavity, enhancing the modulation depth of the material. The second strategy focuses on inducing intrinsic or structural resonances within the time-modulated material, strongly enhancing the system’s quality factor $[69]$ . This method significantly reduces the required modulation depth (two orders of magnitude), enabling the implementation of PTCs that possess considerable momentum bandgaps in low-loss materials characterized by Kerr-type nonlinearity. Furthermore, recently, in Ref. $[187]$ it was proposed that the PTC behavior can be mediated by phonon squeezing. Finally, in recent works, it was suggested that the large momentum bandgap can be obtained in low-loss systems supporting biaxial anisotropy $[71]$ or longitudinal phonon modes $[72]$ .

The discussion presented above is related to PTCs with time-modulated permittivity. However, the concept of PTCs extends beyond just variations of permittivity over time. Some theoretical studies have explored the modulation of other material parameters, for instance, varying the permeability $[45]$ and magneto-optical coefficients $[188]$ . In such scenarios, the modulation of these parameters also leads to a momentum bandgap, paving the way for more complex and exotic effects. Despite this theoretical advancement, practically implementing modulation of parameters other than permittivity remains a significant challenge, particularly within optical frequency ranges.

## 5.3. 2D Platforms

In the previous sections, we have discussed PTCs made of time-varying volumetric (3D) materials or 1D transmission lines that emulate them. These were the original types of PTCs proposed. However, the concept of PTCs can be also applied to 2D systems. In a 2D material system, the modulation is applied to surface properties, such as surface conductivity or impedance. In contrast to bulk media, which support plane-wave propagation in all three spatial dimensions, the 2D PTCs have negligible thickness and sustain eigenmodes that propagate along the surface. This leads to a restricted dimensionality of the problem since the surface waves can propagate only along two directions (in specific settings, propagation in only one direction is permissible). This dimensional reduction simplifies the implementation of PTCs significantly, as modulation is required only on the surface, exempting the need to modulate along the thickness direction.

The periodic fast temporal modulation of material properties in a 2D material system, probably, for the first time were considered in Ref. [189]. The authors studied a suspended graphene layer as shown in Fig. 24(a), where the layer is located in the $xz$ plane. Without a temporal modulation, the graphene sheet supports TM-polarized surface plasmon polaritons as its eigenmode. The top panel of Fig. 24(a) shows that under a dipole source excitation, a TM-polarized surface plasmon polaritons is generated and propagates in the outward direction as indicated by the yellow arrows. First, the authors considered a simple case of a temporal jump, that is, when the surface conductivity was instantaneously switched to another value at a specific moment. Therefore, time-reflected and time-transmitted waves were generated, as indicated by white and yellow arrows in the bottom panel of Fig. 24(a). This effect is similar to the temporal jump of bulk media, as discussed in Section 2.1e, but happens in a 2D system here. The amplitudes of temporal reflection and transmission can be derived using temporal boundary conditions [189]. Next, the authors of Ref. [189] considered a temporal cascade of such temporal jumps resulting in a periodic stepwise modulation of the surface conductivity. The modulation frequency was twice the frequency of the surface plasmon polaritons. It was discovered that such modulation leads to constructive interference between time-reflected and time-transmitted waves, resulting in an exponentially growing standing-wave pattern, as depicted in Fig. 24(b). Similar results have been observed more recently in Refs. [190-192].

At the time of those initial investigations, the amplification was not attributed to the momentum bandgap as the band structure of such time-varying surfaces has not been investigated in depth. Only recently was the band structure of a time-varying material surface calculated $[145]$ . An artistic illustration of the experimental setup with an actual implementation of that surface is shown in Fig. 24(c). In that work, it was assumed that the surface is capacitive, and the surface capacitance varies periodically in time as a function of $C(t) = C_{\mathrm{av}}[1 + m_{\mathrm{C}} \cos(\omega_{\mathrm{m}} t)]$ , where $C_{av}$ is the average surface capacitance. The band structure for the TE-polarized surface wave supported by the time-varying surface is shown in Fig. 24(d). It was revealed that a momentum bandgap is open (see the yellow shaded region). Inside the momentum bandgap, the eigenfrequencies $(\ldots, \omega_{-2}, \omega_{-1}, \omega_{0}, \omega_{1}, \ldots)$ are complex. Some modes $(\omega_{-1}, \omega_{0})$ correspond to surface modes (below the dotted light line in the figure), whereas others correspond to free-space propagating modes (above the light line). This implies that modulating a surface boundary can provide amplification of the surface and free-space propagating modes simultaneously. This observation suggests that the reduction in the dimensionality of PTCs does not compromise their characteristic momentum bandgap and the consequent amplification effects. Instead, it underscores an advantage over bulk media, limited to amplifying only the propagating plane waves.

Due to the dimensionality reduction, 2D PTCs are much easier to realize than 3D PTCs. This eliminates the need for uniform modulation of the material parameters into the third dimension (thickness direction), a requirement for bulk media. For

## Figure 24

(a)
![](images/717c33bea6fbb9b6cd607e761d0f5df13ba738b6ff1f594dfbfc68e77b9a77dd.jpg)

(b)
![](images/5b86a0cdd2607b6a04108842b454d935af6dd96f3fc8a67d5a6f8229440e5ddd.jpg)

(c)
![](images/bd905a07fe476ec8d2a1846d6bee04ff740228de4269b4c267a6186e6b5cbcc1.jpg)

(d)
![](images/c21d414e63d2e734b0cfef5315469f14d15128f88249c61fdd719dd0feb48f8a.jpg)
(a) Propagation of a surface plasmon polariton on a graphene sheet (xz plane) before (top) and after (bottom) a temporal jump of the surface conductivity of graphene. The arrows represent the propagating directions. (b) Field growth of a graphene plasmon polariton as a function of time when applying a periodic stepwise modulation of surface conductivity. The field is recorded at one spatial point on the surface. The blue and red curves correspond to real and imaginary parts of the z component of the magnetic field. (c) Metasurface realization of a PTC. Due to the periodic temporal modulation, both surface modes and propagating modes become exponentially growing inside the momentum bandgap. (d) Band structure of the metasurface in (c). The orange dashed line represents the dispersion curve of a stationary (nonmodulated) capacitive surface. (a) and (b) Figures 2 and 4 reprinted with permission from Wilson et al., Phys. Rev. B 98, 081411, 2018 Ref. [189]. Copyright (2018) by the American Physical Society. (c) and (d) Reprinted from [145] under a Creative Commons license.

this reason, the authors in Ref. [145] implemented the time-varying homogeneous surface in the microwave region using a metasurface. The capacitive metasurface was composed of periodic metallic patches with effective capacitance of $C_{\mathrm{av}}$ (see Fig. 24(c)). The metallic patches were connected with varactors. By applying uniform periodical voltage signals on the varactors, the effective capacitance of the whole metasurface can be modulated in time. The experimental results observed near-30-dB amplification of surface wave in the center of the momentum bandgap.

## 5.4. Mimicking PTCs With Other Material Platforms

Electromagnetic waves exhibit similarities with various types of waves, such as mechanical, phonon, and sound waves. This similarity allows the concept of PTCs, characterized by the periodic repetition of electromagnetic properties in time, to be extended to other wave physics domains. For instance, a material with time-modulated acoustic properties may be considered an acoustic time crystal. This section explores a range of proposed materials that align with the broader concept of “wave time crystals.” The study of wave time crystals within various physics domains serves a dual purpose. First, alternative material platforms may offer more practicality, thereby facilitating testing fundamental concepts shared by all wave time crystals. Second, such advancements could pave the way for novel technological applications in diverse technological fields.

One of the early realizations of the wave time crystal has been proposed in Ref. [193] using an acoustic system. Figure 25(a) depicts the experimental setup, consisting of a rotating cylinder with four chambers. One chamber is filled with $\mathrm{CO}_{2}$ , whereas the remaining three contain air.

An acoustic source placed on the right side of the cylinder emits waves along the $+x$ direction, and a detector is located on the left side. Rotating the cylinder alters the mass density of the medium through which the wave propagates over time. Figure 25(b) shows the temporal variation of the mass density that the incident acoustic waves experience, varying between the densities of $\rho_{\mathrm{air}}$ and $\rho_{\mathrm{CO_2}}$ , which $\rho_{\mathrm{CO_2}}$ is 1.53 times than $\rho_{\mathrm{air}}$ . The acoustic band structure for this time-varying acoustic medium is presented in Fig. 25(c), illustrating the first Brillouin zone. Since in the experiment the modulation frequency $\omega_{\mathrm{m}}$ was limited to the value of $\omega_{\mathrm{m}} / 2\pi = 9.3~\mathrm{Hz}$ , being much smaller than the incident acoustic frequency $\omega_{\mathrm{m}} \ll \omega_{\mathrm{inc}}$ , the dominant excited harmonics had very large values of the normalized wavenumber $kc / \omega_{\mathrm{m}}$ . In other words, the dominant harmonics belonged to very high-order bands shown in Fig. 25(c) (note that the diagram is discontinued at $kc / \omega_{\mathrm{m}} = 1.5$ for clarity). At such high-order bands, momentum bandgap does not open, as is shown in Fig. 4. The authors of Ref. [193] measured experimentally the transmission spectra of such an acoustic time crystal. For an incident frequency of $\omega_{\mathrm{inc}} / 2\pi = 3\mathrm{kHz}$ , the transmission spectrum is depicted in Fig. 25(d). One can see the main peak, corresponding to the incident wave frequency, surrounded by additional side harmonics separated from one

## Figure 25

![](images/9441a78da058b3a95ac432f0c7eaae99f953bac9a81a2365e14aaf2ba20c8dc3.jpg)

(b)
![](images/33fcf2ad0ff2dddf3fba09fccfbf3982b2d99a1b57c1edf89dacf27f253aecf9.jpg)

![](images/0fd5a0d736aa98020bc1eb8f31827921d959336fbfbd1ece928c0ce79b85acf3.jpg)

(d)
![](images/f258d193da10cee4090716bd21e543632ca0282e378a37bfc469317a49fa04cf.jpg)
(a) Experimental setup of an acoustic time crystal consisting of a rotating cylinder with four chambers through which acoustic waves propagate along the x direction. (b) Theoretical temporal variation of the mass density of the acoustic time crystal. In fact, a stepwise modulation in practice is hard to obtain in the explored setup, but it continues to be a good approximation. (c) Band structure of the acoustic time crystal. Note the discontinuity of the x axis. The spectral domain shows the experimentally relevant bands because the modulation frequency is rather small compared with the operational frequency where the response of the acoustic time crystal is probed. (d) Transmission spectrum of the acoustic time crystal when illuminated by waves with a frequency of 3.0 kHz. (a)–(d) Reprinted from [193] under a Creative Commons license.

another by $\omega_{m}/2\pi$ distance. It should be mentioned that the proposed acoustic time crystal, despite its straightforward configuration and absence of any nonlinear effects, possesses a substantial disadvantage of limited feasible modulation frequencies due to the mechanical rotating parts.

Another platform for observing physics of wave time crystals is based on water waves. A possible experimental setup, proposed in Ref. [59] to study the physics of time-varying medium using water waves, can be seen in Fig. 26(a). The setup consists of a grounded water tank that is conductive. An electrode is placed at a distance $d$ above the water surface. The electrode is connected to a high-voltage amplifier. A Faraday instability, which manifests as parametric water waves, can be observed by modulating the gravitational acceleration. The gravitational potential is periodically changed by applying a sequence of voltage pulses to the electrode, which exerts an electrostatic force on the water surface. This system couples hydrodynamics and electrostatics. Therefore, instead of modulating the gravitational acceleration through physical displacement, we can use electrostatic modulation to achieve this effect [194]. A comb-like modulation of potential in time is applied to the electrode on the top of the water to implement the time crystal. A light source illuminates a checkerboard pattern below the water tank, and with a camera above the setup, the evolution of Faraday waves can be captured. A band structure can be extracted as the potential is modulated in time in the system and uniform in space. The band structure with the specific parameters of $d = 5\mathrm{mm}$ and the amplitude of the Dirac comb potential equals $8\mathrm{kV}$ is shown in Fig. 26(b). There are two bandgaps in the band structure.

In Refs. [195,196], phononic time crystals were analyzed both theoretically and experimentally. These studies focused on a 1D phononic lattice characterized by time-periodic elastic properties. This lattice was composed of magnetic masses (cylindrical bar magnets) that repel each other, with their grounding stiffness modulated by electrical coils that are driven by periodically varying electrical signals. For weak excitations (in the linear regime), the authors observed the generation of momentum bandgaps. Furthermore, it was discovered that such phononic time crystals can support what are termed wavenumber breathers [195]. Unlike classical breathers, which are spatially localized and time-periodic solutions to nonlinear lattice differential equations found across photonics, phononics, and electrical systems [197], wavenumber breathers are localized in time and exhibit periodicity in space. This positions them as the dual counterparts to the classical breathers.

PTCs have also been investigated in elastic and electromechanical waveguides $[198,199]$ . For instance, when the stiffness is modulated in time, the complex frequency has been demonstrated. In such a platform, the modulation is implemented experimentally by an array of piezoelectric patches shunted through a negative electrical capacitance controlled by a switching circuit $[198]$ .

So far, in this section, we have discussed the possible implementation of wave time crystals based on water, elastic, and acoustic waves, in which the modulation frequency was rather small. Let us investigate other platforms to find possible realizations of PTCs at optical frequencies. In that context, the concept of synthetic dimension is a powerful tool for analyzing a higher-dimensional system using a lower-dimensional system. For instance, arranging and coupling certain states of a system makes it possible to form a 1D lattice in the synthetic space [200]. One example of such a system is a ring resonator supporting resonant modes at a discrete set of frequencies (equally spaced from one another). By time-modulating the permittivity of the ring, it is possible to couple the neighboring resonance modes, mimicking the tight-binding model in

## Figure 26

![](images/e5bdac7927a240117865c89689becbd0663ea5b4fa8172bf9eebb6a074391cbe.jpg)
(b)

![](images/c95ac157c0efe0624e527d6ad19e0c45e43a70df7fd67ffac277ff26a710faef.jpg)
(a) Experimental setup to study time crystals in water waves. The time crystals for water waves are enabled by the modulation of a potential above the water surface to excite a Faraday wave. A Faraday wave refers to standing waves that appear on the surface of a liquid in a container undergoing vertical oscillations. The vertical oscillation is induced in this experimental setup by an electrode placed above the water surfaces and on which a sequence of voltage pulses is applied. The sequence of these pulses takes the form of a comb of Dirac delta distribution. The applied electric field exerts an attractive force on the water surface that modifies the wave speed, forming a time crystal for these specific water waves. The water waves are imaged by a camera above the setup that takes images of a checkerboard underneath the water tank. From the deformation of the checkerboard, the water waves can be reconstructed. (b) Band structure of the system exemplarily for $d = 5 \mathrm{~mm}$ and an amplitude of $8 \mathrm{kV}$ and a frequency of $60 \mathrm{~Hz}$ for the Dirac comb excitation. Real (red) and imaginary (blue) parts of the dispersion relation can be seen. Let $\Gamma = 2\nu k^2$ , where $\nu$ represents the kinematic viscosity of the liquid. The eigenvalue is expressed in the form $\exp(i\mu - \Gamma)$ , where $\mu$ is a complex number. Clearly visible momentum gaps form. Reprinted with permission from [59].

the synthetic dimension. Moreover, one can configure a 2D synthetic lattice with a different arrangement and coupling [201,202].

Recent proposals suggest leveraging synthetic dimensions, specifically the 1D Su–Schrieffer–Heeger (SSH) lattice model, to explore phenomena such as time reflection and time refraction [117]: phenomena essential for constructing PTCs. Such a lattice contains two energy bands, and an abrupt change in the lattice leads to a new band structure after the time boundary. The modes in that modified band structure contain information on the propagation characteristics of the time-reflected and time-refracted wave packets. In Ref. [117], the SSH model was implemented as a momentum–space lattice. Here, the sign of the group velocity can be changed by altering the sign of the hopping parameters, effectively changing the sign of the refractive index. The experimental realization has been based on a Bose–Einstein condensate of a cloud of ultracold atoms that forms the momentum states as a synthetic dimension. These states can be coupled through a two-photon Bragg transition, using two laser beams with opposite momentum and one containing multiple frequencies. By adjusting the intensity and phase of the laser beams, it is possible to control the coupling coefficient and implement variable coupling, which enables the observation of time reflection and time reflection [117]. Notably, the coupling between each state can be controlled independently in this platform.

Time reflection and refraction were also observed recently in photonic platforms supporting synthetic dimensions $[118]$ . Importantly, it was found that modulation at microwave frequencies (as low as 0.72 MHz) was sufficient to observe time-boundary effects for optical waves in the synthetic frequency dimension. In that work, a two-leg ladder model comprising two energy bands was considered. Such a system can be described by a dedicated model that features two coupled sites, characterized by a coupling coefficient. In addition, the model accommodates a self-coupling at each site. Exciting the system in one particular mode and applying an abrupt change to the coupling coefficient within sites causes a new band structure. The new band structure sustains modes into which the incident wave packet can be time-reflected and time-refracted. This model can be physically realized with two ring resonators with frequency modes representing each site of the ladder lattice. As mentioned before, each resonator is made from time-varying permittivity that couples the modes. The coupling between sites happens with a directional coupler between two resonators. By applying a variation in the amplitude of modulation in the ring resonator, the time boundary occurs, and the reflected and refracted waves are observable. In conclusion, the two platforms mentioned above allow us to obtain strong time reflections. Therefore, they can be also used for creating a PTC using synthetic dimensions.

## 6. POTENTIAL APPLICATIONS OF PTCs

The flow of the tutorial has been chosen in such a way that we introduced the basics and some advanced properties of PTCs first. Then, we discussed specific material systems and spectral domains that allow us to implement PTCs. In the following section, we discuss various applications of PTCs. It is important to note that the field of PTCs is now predominantly in the realm of fundamental research, with applications of PTCs still being relatively limited. Nevertheless, these opportunities stimulate future research efforts, and we should keep them in mind. It should also be noted that the strict requirements for rapid temporal modulation have hindered the optical realization of PTCs. Consequently, at the same time, research has been expanding in the direction of photonic space–time crystals, where these stringent requirements are to some extent relaxed. The applications of space–time crystals are discussed in Section 7.

## 6.1. Realizing Thresholdless Lasers Using PTCs

One of the crucial subjects worth studying concerning any periodic structure or, more general, photonic material is the investigation of the radiation of a source placed inside or close to it. The photonic materials may significantly affect the local density of states that change, in turn, the radiation properties of a source compared with the case that the source is in a vacuum. In addition to metamaterials, this subject has attracted particular attention in the photonic crystal community. In fact, for spatial photonic crystals, this study resulted in influential novel discoveries. It was shown that if a 3D periodic dielectric structure has an electromagnetic bandgap that overlaps with the transition frequency of an emitter placed into the photonic material, spontaneous emission from the source can be suppressed entirely $[51]$ . There will be simply no electromagnetic mode in space to which an excited emitter could release its energy. The idea is quite general and can also be extended to other types of photonic materials. For example, an emitter whose specific transition is given by different multipolar contributions can be placed and oriented in close proximity to an individual plasmonic nanoantenna so that its emission is suppressed. In this case, the emission along the different multipolar contributions can be brought into a destructive interference $[203]$ . Modifying the spontaneous emission of emitters thanks to a structured photonic environment led to many applications, e.g., in the context of solar cells, light-emitting devices, or displays $[204,205]$ .

Hence, it will be reasonable to scrutinize the problem for PTCs: How is the emission of electromagnetic waves (or light) by a source embedded in a PTC modified? For instance, one can think about the point dipole radiation $[53]$ . Importantly, it is intriguing to see how the radiation depends on the excitation frequency, which determines the oscillation of the dipole moment. This is a valid question due to the existence of momentum bands and gaps in a PTC. Here, we discuss the problem initially classically and afterwards quantum-optically, as done in Ref. $[53]$ . We work out that radiation is exponentially amplified if the emission is linked to the momentum bandgap.

After having discussed the details of the dispersion relation in depth in the previous sections, we should be familiar with the fact that the imaginary part of the eigenfrequency, which expresses the growth in time, is strongest at the frequency corresponding to half the modulation frequency. Therefore, the longer the evolution of the field, the stronger that frequency dominates over all the others. Of course, there are also exponentially decaying fields that vanish after enough time has elapsed. However, from a classical perspective, we shall appreciate that as soon as there is a finite projection of some incident field onto the mode that will exponentially grow in time, the amplitude of that mode will dominate the field eventually. It is worth mentioning that the presence of an exponentially growing and decaying solution is in contrast to spatial photonic crystals in which the excitation within the bandgap results in two decaying waves in space (evanescent waves).

However, what happens if we excite with a frequency associated with the passband? Nevertheless, in this scenario, we must keep in mind that we switch on the source at a specific moment. The frequency spectrum launched into the system will not correspond to a single frequency, as that would require a time-harmonic source that oscillates forever and will oscillate forever. Any possible source will consist of a time-harmonic signal modulated with some envelope function. Then, the spectrum launched into the system corresponds to the Fourier transform of that envelope function convoluted with the Dirac-delta distribution at the signal frequency.

To be precise, we consider a point dipole that is turned on at a specific moment: $\mathbf{J}(\mathbf{r},t)=\mathbf{J}_{0}\delta^{3}(\mathbf{r})\exp(j\omega_{0}t)\theta(t)$ . Here, $\delta^{3}(\mathbf{r})$ represents the 3D Dirac delta distribution, and $\theta(t)$ is a Heaviside step function describing that the electric current density is generated at t=0. Such a dipole excites modes with all possible frequencies, including the exponentially growing gap modes. Of course, predominantly, it emits radiation at the carrier frequency, but the multiplication with the Heaviside step function also initiates other frequencies. The emission from such a source leads to fields amplified constantly over time, drawing energy from the modulated material parameters.

To validate the above statements, we use COMSOL Multiphysics software to simulate the emission from a point dipole located in a time-varying material. Figure 27(a) depicts the geometry of the problem. The permittivity is modulated harmonically in time. The dipole is considered to emit radiation at one of the three different carrier frequencies. These frequencies are denoted as $\omega_0 = \omega_{\mathrm{m}} / 6$ , $\omega_{\mathrm{m}} / 4$ , and $\omega_{\mathrm{m}} / 3$ , where $\omega_0$ and $\omega_{\mathrm{m}}$ are the carrier frequency and modulation frequency, respectively. Note that the dipole source is turned on at $t = 0$ in the simulation. Figure 27(b) shows the field evolution as a function of time for $t > 15T_{\mathrm{m}}$ where $T_{\mathrm{m}}$ is the reference time defined as $T_{\mathrm{m}} = 2\pi /\omega_{\mathrm{m}}$ . The time variation is switched on $15T_{\mathrm{m}}$ after the source has been switched on. There are a few aspects we can notice. Initially and after the time modulation kicks in, at a very low amplitude, the fields of the dipole source oscillate at the frequency corresponding to the carrier frequency of the source (see the inset of Fig. 27(b)). This makes sense because the source predominantly emits fields of such a frequency. Outside the momentum bandgap, the eigenmodes of the medium correspond to time-harmonic fields.

Figure 27

(a)
![](images/3e14c58dc1ab9d43a696168e5089b522e4dba40689288b437ec3eea0d7f5d244.jpg)

![](images/63b78d4ba8230b8595194f64548ab976baa9f3e4474e0905f868c90af1684ed7.jpg)
(b)

![](images/de9deaba0ee5fded79110b1bc26297cc4ac51273692db8aadf9cae9b68c29d60.jpg)
(c)
(a) Schematic view and (b), (c) simulation results for a point dipole that is embedded in a PTC. In the simulation, the relative permittivity of PTC is set as $\varepsilon(t)=3(1-0.2\cos(\omega_{\mathrm{m}}t))$ . (b) Evolution of the radiated field in time for three different excitation frequencies. Here, $\omega_{0}$ denotes the excitation frequency, and $\omega_{m}$ represents the modulation frequency. The simulation domain is shown in the inset picture (the dipole source is located in the center, and the probing point is at the right side of the source). (c) Evolution of the radiated field in time for different time intervals $\Delta t$ between the excitation and modulation. Here, the incident frequency is $\omega_{0}=\omega_{m}/3$ . The COMSOL Multiphysics software has been used to achieve these numerical results.

However, with time passing, we note that an exponentially growing field emerged. The frequency of that ever-increasing field corresponds to half the modulation frequency $\omega_{m}/2$ . Independent of the chosen carrier frequency $\omega_{0}$ , this established field oscillation will always have the same frequency. This makes perfect sense in light of the permissible eigenmodes in such a PTC, as it corresponds to the eigenmode with the strongest exponential growth in time. At one moment, it simply dominates all other frequency components. We can think of it as the onset of lasing. It is crucial to consider the dispersion relation in our analysis. The field inside the PTC can always be written as a superposition of the eigenmodes. After launching the source and switching on the time modulation of the material parameter, we have a temporal interface between a stationary medium and the PTC. After switching on the source and after switching on the time modulation where temporal refraction occurs, we excite nearly all possible modes. The exact amplitude depends on the details of the source (in our case, it is mostly the consideration of a Heaviside step function used to switch on the source), but it will be finite at nearly all frequencies. The fields propagate outward as time progresses and are absorbed in the perfectly matched layer (PML) region surrounding the computational domain of interest. Adding PML is solely for establishing a radiation boundary and is not related to the physics of PTCs. Still, even though the initial amplitude will be exponentially suppressed after some time, the amplitude of all the frequencies of the eigenmodes inside the momentum bandgap will grow exponentially once the modulation is switched on. Eventually, the field at the frequency half the modulation frequency is strongest, i.e., the field at $\omega_{m}/2$ . This is precisely what we see in the simulation results.

In the described simulation, the corresponding host medium is modulated at a later moment with respect to the source excitation moment. Hence, it is intriguing to see how the time difference between the excitation and modulation moments also affects the amplification phenomenon. The expectation suggests that the longer we wait, the stronger all the frequencies are damped (because of the dissipation in the PMLs) that do not correspond to the carrier frequency. Therefore, the later we switch on the time modulation, the longer we need to wait until the exponentially growing field corresponding to the momentum bandgap compares in amplitude to the time-harmonic carrier signal that is neither damped nor amplified. Thus, we fix the excitation frequency to $\omega_{0} = \omega_{m}/3$ and simulate the field where we switch on the time modulation of the material properties after some time interval since the dipole source was turned on, which is indicated by $\Delta t$ in Fig. 27(c). As expected, it is clearly seen that as $\Delta t$ becomes larger, the amplification becomes noticeably later. The exponentially damped spurious frequencies need more time to get amplified and reach the value where they are dominant.

The classical picture provided above about the radiation of a classical point dipole can be extended to the emission of light by atoms that are in excited states by using the principles of quantum electrodynamics. However, the problem is not straightforward. When we have a static medium, the excited atom will decay to the ground state which is known as spontaneous emission. On the other hand, for the PTC, it is challenging to analyze this type of emission due to the generated photons by the PTC regardless of the frequency of the atomic transition. If we neglect the effect of the gap modes, the spontaneous emission rate is expressed as $[53]$

$$
\gamma = \frac {V}{\pi \hbar^ {2}} \sum_ {m} k _ {m} ^ {2} \left| V _ {f} ^ {m} \right| ^ {2} \left| \frac {\partial \omega}{\partial k} \right| _ {k = k _ {m}} ^ {- 1},\tag{101}
$$

in which $V_{f}^{m}$ is the coupling constant between the initial and the final Floquet eigenstates through the interaction Hamiltonian, and $k_{m}$ denotes the wave number of the mode corresponding to the mth harmonic of the atomic transition. This equation shows that when we are very close to the band edge of the PTC, the spontaneous emission rate becomes very small (ideally zero). This is because, at the band edge, the slope of the dispersion curve is basically vertical ( $\partial\omega/\partial k \rightarrow \infty$ ). This issue has interesting consequences. Since the exited atom does not decay into the lower state, it will always remain in the excited state. It is important to remark that this theoretical conclusion has been criticized by Ref. [206]. It claims that the vertical slope of the dispersion relation at the band edge does not guarantee zero spontaneous emission rate in non-Hermitian systems. In fact, if the analysis (Eq. (101)) properly considers the non-Hermiticity of the PTC, the spontaneous emission rate is nonzero at the band edge [207].

The above results show the application of PTCs in lasing, where we can provide a resonator by locating mirrors on both sides of the PTC. The length of the cavity must be large enough compared with the desired wavelength. The existence of a saturation mechanism gives rise finally to a stable monochromatic emission. Importantly, such emission can be engineered because we can control the modulation system, giving us the possibility to have a tunable laser. We would like to mention that in Ref. [208], a model based on a four-level system has been used to demonstrate this possibility of having a (threshold-free) lasing operation.

## 6.2. Enhancing the Emission Rate of Radiation by Free Electrons

In the previous section, we described radiation from mainly point dipoles and explained the corresponding mechanism. However, the radiating source embedded in a PTC or close to a PTC can differ. This subject becomes important if the PTC can dramatically affect the radiation phenomenon regarding those sources such that it removes fundamental limitations that exist when the same source is placed in conventional static materials. Accordingly, in addition to point dipoles, the interaction of free electrons with PTCs and investigation of the radiation from these free electrons has also attracted attention $[54,209]$ . Within a bulk unbounded static medium, if a free electron moves with a constant velocity, it emits in the form of Cherenkov radiation. This radiation mechanism requires that the constant velocity of the electron is greater than the phase velocity of the electromagnetic wave propagating in the static medium. Therefore, the question is: Does the PTC eliminate such a condition? Indeed, it is shown that free electrons moving in a spatially homogeneous PTC radiate spontaneously even if the constant velocity is below the Cherenkov threshold as illustrated in Fig. 28 $[54]$ . This is an interesting result. Of course, another benefit is that similar to the radiation from the point dipole explained above, if the wave vector is located within the momentum gap, we expect that the radiation is enhanced exponentially (for that, the energy is provided by the modulation).

To describe the physics, we can model the moving electron classically and define an electric current density expressed as $\mathbf{J}(\mathbf{r},t)=\delta(\mathbf{r}_{\perp})\delta(z-\beta ct)\mathbf{a}_{z}$ , in which $r_{\perp}$ refers to the coordinates transverse to z, and $v=\beta c$ determines the constant velocity of the electron. This expression assumes that the electron is moving in the z direction. On the other hand, based on the theory of PTCs explained above, each Floquet frequency is accompanied by an ideally infinite number of harmonics. Thus, according to this fact about harmonics and the expression for the electric current density, we can conclude that the optimum scenario for radiation from a free electron is when $k_{z}\beta c=\omega_{F}+m\omega_{m}$ . Here, $k_{z}$ is the z-component of the wave vector (recall that $\omega_{F}$ is the Floquet frequency, $\omega_{m}$ is the modulation frequency, and m represents the order of the harmonic). This equality is a phase-matching condition, which plays a significant role. It means that the temporal modulation provides the electron with the energy to interact with frequencies that are higher or lower than the Floquet frequency of the radiated modes. This has important consequences. As mentioned, concerning a static medium, the electron emits if its constant velocity is larger than the phase velocity of modes in the medium. However, now, regarding a PTC, the situation is different. Due to this phase-matching condition, we can have interaction with lower harmonics $(m<0)$ that allow the free electron with a lower speed to emit (although we are in the regime where Cherenkov radiation cannot exist). For example, in Ref. [54], the phase-matching condition is met for the order m = -1, and, therefore, the radiation occurs (i.e., $k_{z}\beta c = \omega_{F} - \omega_{m}$ ). Temporal modulation generates harmonics that have small eigenfrequencies for a given wavenumber $k_{z}$ . Therefore, these harmonics have reduced phase velocity $\omega/k_{z}$ , enabling slow electrons to radiate. Above the Cherenkov threshold, one can expect that the electron emits ordinary Cherenkov radiation that happens in a time-invariant medium as well. However, it can also radiate to higher harmonics with m>0.

![](images/bc2c70510aa7f79a678fe354d331360bcb9bbfdd90f8dca757d8dff5ba6877e7.jpg)

(b)
![](images/85b6dacc220d267dc7e73bc50b5e5c7990fc40db5088f5dcd906a7fa38e61f79.jpg)
Radiation of a free electron with a finite velocity, which is located in a PTC. (a) Schematic view of the evolution of radiation and electron position as time moves forward. The permittivity of the medium is periodically changing in time. (b) Finite-difference time-domain (FDTD) simulation result for the amplitude of the magnetic field of the electromagnetic radiation. Here, the free electron is moving in a PTC of sinusoidal modulation starting at $t = T_{1}$ and ending at $t = T_{2}$ . The simulation was done for the scenario in which the electron velocity is below the Cherenkov threshold. (a) and (b) Reprinted from [54] under a Creative Commons license.

Hence, in conclusion, there are two different regimes of radiation: the “subluminal” regime, where only lower harmonics contribute to the process, and the “superluminal” regime, in which all the harmonics can be present in addition to the fundamental harmonic $m = 0$ . In the end, we would like to mention two points. First, if a free electron in a PTC starts to radiate (within the subluminal or superluminal regime), the velocity cannot be constant. Due to the radiation, the electron speed must immediately change in time (which means that the acceleration is not zero). From this point of view, the same scenario may happen as the flash point dipole (described in the previous section). In other words, such a source may excite all the modes including the gap modes, which results in the exponentially growing fields in any case. However, a more rigorous and sophisticated theory is needed to validate this statement. Second, the electron with the constant velocity should not necessarily be embedded in a PTC in order to radiate. It can be in the vacuum and travel close to such a time-varying system [55,210].

In the above, we focused on the radiation from a moving electron. Based on electrostatics, we know that an electron or any charge with zero velocity, which is placed in a static medium, cannot radiate. Such a stationary charge generates only static electric fields in the medium. How about locating this stationary charge in a PTC? For a static isotropic medium, the corresponding electric flux density is not a function of the relative permittivity of the medium. Thus, definitely, in a conventional isotropic PTC, the charge does not emit as well. The electric flux density is still time-independent. However, in the case of a static anisotropic medium, unlike an isotropic medium, we see that the electric flux density depends on the components of the permittivity tensor. Hence, one can think now about what happens if those components become a function of time. It is shown that if the stationary charge is placed in an anisotropic PTC, the electric flux density becomes time-dependent, and the time derivative of the electric flux density does not become zero. As a consequence, as demonstrated in Ref. [151], for such a stationary charge that is within an anisotropic PTC, we have a radiation phenomenon.

## 6.3. Controlling the Spectral Flow of Light

PTCs are interesting not only from the perspective of light amplification inside their momentum bandgaps but also from the perspective of spectral transformations they can provide for a given signal. As was demonstrated in Fig. 3, the spectrum of excited harmonics in a typical PTC is nearly symmetric with respect to the fundamental frequency (n = 0). Nevertheless, this symmetry can be broken in PTCs whose real and imaginary parts of permittivity are periodically modulated in time with a specific phase difference [211]. These PTCs with non-Hermitian modulation functions can provide control over the scattering response in the frequency space, e.g., fully upconverting or downconverting in frequency incident signals. Such “one-way” frequency conversion can be used for creating optical magnetless isolators. Indeed, consider a tandem of two PTCs with non-Hermitian modulation designed such that the left one upconverts and the right one downconverts incident signal of a given frequency $\omega_{i}$ (see the geometry

## Figure 29

![](images/c5a8fb1f46640123a975ca778fb3f8f2a37eabd2af1227978fe9c26496fc7342.jpg)
(a) Reciprocal tandem of two PTC slabs (shown in green) with real-valued periodically time-varying permittivities (Hermitian modulation). The two PTC slabs are separated by a HPF (shown as a blue box). Light transmission for illuminations from the left and from the right is the same. Due to the Hermitian modulation function, incident signal at $\omega_{i}$ experiences symmetric upconversion and downconversion to $\omega_{i} + \omega_{m}$ and $\omega_{i} - \omega_{m}$ , respectively. (b) Same as in (a) but both PTC slabs have non-Hermitian modulation (both real and imaginary parts of permittivity are periodically modulated at different regions of the slabs depicted in green and orange). Light can be transmitted only from left to right, whereas it is fully blocked by the filter when propagating in the opposite direction. (c) Asymmetric frequency conversion can be used for “frequency” rerouting of incident waves providing the opportunity for direction-dependent optical cloaking. The object (shown in blue) has a small transparency window for blue wavelengths. By upconverting incident light to these wavelengths using the PTC, one can obtain high light transmission through the object. The transmitted light is then downconverted to the original incident frequency. © 2021 IEEE. Reprinted, with permission, from Hayran and Monticone, 2021 Fifteenth International Congress on Artificial Materials for Novel Wave Phenomena (Metamaterials), pp. 153–155, 2021 [211].

in the upper part of Fig. 29(b)). The two PTCs are separated by a high-pass filter (HPF) transmitting only $\omega_{\mathrm{i}} + \omega_{\mathrm{m}}$ harmonic, shown as a blue box in the figure. In this scenario, the light can propagate through the tandem of the PTCs only from left to right, as shown in the bottom of Fig. 29(b). A similar nonreciprocal effect due to one-way frequency conversion was also reported previously in other optical systems [212,213]. On the other hand, if the PTCs with Hermitian modulation (only the real part of permittivity is modulated) form a tandem, it will be reciprocal, as shown in Fig. 29(a).

Moreover, it was suggested in Ref. [211] that the same PTCs with non-Hermitian modulation functions can be used for the optical cloaking of objects. As is shown in Fig. 29(c), if an object is made out of a dispersive material, it can be hidden for a given illumination by shifting the incidence spectrum into a frequency domain for which the object is transparent. Then, the light will weakly interact with the object. After shifting the frequency back to the original value, the incident field continues to propagate as if there would have been no object. This “frequency rerouting” could possibly allow large objects to be rendered invisible, overcoming conventional cloaking limitations.

However, it is important to acknowledge that this cloak would not work for the light coming from the opposite direction.

## 6.4. Advanced Optical Absorbers Beyond the Rozanov Bound

In this section, we discuss PTCs with specifically designed losses and their role in achieving ultrawideband absorption that exceeds the Rozanov limit. In 2000, Rozanov [214] established that the absorption bandwidth of a lossy material slab is related to its thickness. If a time-invariant material slab has a reflection spectrum denoted as $\Gamma (\lambda)$ , then the integral of $\ln |\Gamma (\lambda)|$ across the entire wavelength spectrum is bounded above by a value related to the slab's thickness $d$ and slab permeability $\mu_{\mathrm{s}}$ . This limit is expressed as follows:

$$
\left| \int_ {0} ^ {+ \infty} \ln | \Gamma (\lambda) |, d \lambda \right| \leq 2 \pi^ {2} \mu_ {s} d.\tag{102}
$$

The left-hand side of Eq. (102) is called the Rozanov integral $I_{R}$ .

The Rozanov bound in Eq. (102) is derived under the premise that materials are linear and time-invariant. Therefore, a natural possible strategy to surpass the Rozanov limit is to introduce material temporal modulations. For example, in Refs. [215-217], the authors used temporal switching of a lossy material, to go beyond the Rozanov limit. However, temporal switching requires precise synchronization of the switching events, which is challenging to achieve in practice.

A promising solution, as proposed in Ref. [122], is to apply continuous (periodic) modulation on the dissipative material. Here, the material is dispersive following the Lorentz dispersion relation. The plasma frequency is modulated according to $\omega_{\mathrm{p}}^{2}(t) = \omega_{\mathrm{p0}}^{2}[1 + m_{\mathrm{p}}\sin (\omega_{\mathrm{m}}t + \phi_{\mathrm{m}})]$ , which effectively modulates the permittivity continuously. This continuous modulation results in the formation of a PTC. The band structure of such a lossy PTC can be analyzed using the Floquet theorem, as presented in Section 2.1d. With the proper loss factor, a momentum bandgap is formed in the band structure, depicted in Fig. 30(a). Unlike lossless PTCs, the two complex eigenfrequencies within this bandgap indicate decaying modes due to the negative imaginary part of the eigenfrequencies ( $e^{-i\omega t}$ convention is used in Ref. [122]), as illustrated in the lower panel of Fig. 30(a). To overcome the Rozanov limit, one needs to predominantly excite the lossier eigenmode (shown by the line with blue triangles in the figure). For that, the authors of Ref. [122] position a time-varying dielectric slab on a metal plate, as shown in Fig. 30(b). As was discussed in Section 4.3, then the PTC becomes phase-sensitive to the excitation, and by carefully adjusting the modulation phase, it is possible to selectively excite one of its two eigenmodes. Since the mode with larger attenuation is excited and dominant, the absorptance in the material can be higher than in its time-invariant counterpart. Figure 30(c) displays the Rozanov integral $I_R$ as a function of modulation parameters (phase $\phi_{\mathrm{m}}$ and frequency $\omega_{\mathrm{m}}$ ). One can see that with appropriate phase selection, for any modulation frequency, performance beyond the Rozanov limit is achievable. Thus, PTCs can be used to create ultrathin broadband absorbers of electromagnetic radiation. Such absorbers are of great importance for many applications, especially at gigahertz and millimeter-wave regimes.

Recently, it was proposed that PTCs under proper configuration can be used for creating customizable multi-band absorbers $[218]$ . Such absorbers are capable of absorbing light at a predefined number of frequency bands, forming a frequency comb. It was demonstrated, in particular, that by changing the incidence angle or the temporal modulation function, it is possible to control the spectral spacing between the neighboring bands in the comb. This functionality can find applications for the control of thermal emission, direction-selective filters, and switches.

## Figure 30

(a)
![](images/8aa035a631d889dd80d4a0f31c63e37c0ea4611e2886296fe5a04a56a1da9f31.jpg)
(b)

![](images/cafefcec2d45d64e5d38769fd3fdd993aa73b3080852693abe657ff75165c7d5.jpg)

![](images/11d2b47a1acd4d3ebdd75bb768dd11e1fb03425a842c8b66f8edbfbee2add367.jpg)

(c)
![](images/bbb94f45a780e5cce69e65fc7490a8224f1dfa54b7d946edc2d79336562e489c.jpg)
Normalized modulation frequency, $\omega_{m}d/2\pi c$
(a) Band structure of a lossy PTC, calculated from the Floquet theorem. The modulation depth here is $m_{p} = 1$ . The momentum bandgap in the band structure induced by the periodic temporal modulation (upper panel) results in the imaginary part of the eigenfrequency splitting into two distinct values within the gap (lower panel). Colored triangular markers denote results obtained using the coupled mode theory. (b) Structure of the advanced absorber is made of a time-varying dielectric slab and a metal ground plane. The incoming pulse has a finite size in the space and time domain. (c) Rozanov integral values, $I_{R}$ , for a range of modulation parameters. The modulation depth here is $m_{p} = 0.4$ . Figure 2 reprinted with permission from Hayran and Monticone, Phys. Rev. Appl. 21, 044007, 2023 Ref. [122]. Copyright (2023) by the American Physical Society.

## 6.5. Enhanced Resolution Imaging

Recently, it has also been suggested that PTCs could find applications in the field of imaging for resolution enhancement $[219]$ . In order to obtain an image with resolution not bounded by the diffraction limit, one needs to preserve the information about its high spatial frequencies (wavenumbers) $[220]$ . Since the light components with high spatial frequencies reside below the light line of the medium where light propagates, they get quickly attenuated in space and cannot reach the image sensor. A perfect lens based on a material slab with a negative refractive index was proposed to solve this problem and restore high-resolution details of an image through amplification of evanescent modes inside the slab $[220]$ . However, materials with negative refractive index inevitably possess dissipation loss, which diminishes the perfect lens effect. Due to the inherent amplification nature of PTCs inside the momentum bandgaps, in Ref. $[219]$ , it was suggested to exploit PTCs for enhancing high spatial frequencies of the 2D Fourier expansion of an image. Moreover, it was found that by additionally applying aperiodic perturbations to the temporal modulations of the PTC, the image resolution can be further enhanced.

## 7. INTRODUCING SPATIAL PERIODICITY IN PTCs: SPATIOTEMPORAL PHOTONIC CRYSTALS

This last section of this tutorial shall widen its scope, and we concentrate here on photonic crystals with optical properties periodic in space and time. As such, we call them spatiotemporal photonic crystals (ST-PCs). To a certain extent, this can be considered as a generalization of concepts previously considered as independent. Initially, we consider a specific kind of ST-PCs where the modulation is in the form of a traveling wave. Then, we summarize the main contributions toward a generalization of this concept. We elaborate in this context particularly on the aspect of how a suitably structured spatial photonic crystal can enhance the effects associated with a temporal modulation. That is important because it shows that even though we may experimentally achieve only a small to modest modulation of the material properties in time, especially at optical frequencies, these effects can be enhanced by suitably structuring the time-varying material. We finish this section with discussions on topological aspects of ST-PCs and the properties of nonlinear ST-PCs.

## 7.1. Traveling-Wave Modulation

In ST-PCs, the material properties are not only locally periodic in time but also periodic in space. The most simple ST-PCs have a traveling wave modulation form. It assumes that the material properties are modulated as a traveling wave in a general form, which is equivalent to a moving media. As we show in this section, PTCs represent a special case of a broader class of artificial materials, that is, ST-PCs $[23]$ .

Under the assumption that the traveling wave modulation is propagating into the +z direction, such modulation can be written in a Fourier series as

$$
\varepsilon (z, t) = \sum_ {p} a _ {p} e ^ {- j p (k _ {\mathrm{m}} z - \omega_ {\mathrm{m}} t)},\tag{103}
$$

where $k_{m}$ and $\omega_{m}$ are modulation wavenumber and frequency, respectively, and $\omega_{m}/k_{m} = v_{m}$ is the speed of modulation wave. When an x-polarized plane wave characterized by the frequency $\omega$ and wave vector k, in the form of $A(z,t) = A_{0}e^{-j(kz-\omega t)}$ ( $A_{0}$ is the wave amplitude), travels through such a modulated media with modulation $\varepsilon(z,t)$ , due to the frequency mixing effect, the electric field inside the media should contain infinite numbers of harmonics. This can be mathematically explained by multiplying $\varepsilon(z,t)$ and $A(z,t)$ and generating harmonics in the form of

$$
\mathbf {E} (z, t) = \sum_ {n} E _ {n} e ^ {j n (\omega_ {\mathrm{m}} t - k _ {\mathrm{m}} z)} e ^ {j (\omega t - k z)} \mathbf {a} _ {x} = \sum_ {n} E _ {n} e ^ {j (\omega_ {n} t - k _ {n} z)} \mathbf {a} _ {x},\tag{104}
$$

where the generated harmonics have an equal order in space and time, $k_{n} = k + nk_{m}$ and $\omega_{n} = \omega + n\omega_{m}$ . Note that Eq. (104) is an application of Floquet's theorem for periodicity in both space and time. The electric field of the eigenwave in such a media must respect the wave equation

$$
\nabla \times \nabla \times \mathbf {E} (z, t) + \mu_ {0} \frac {\partial^ {2}}{\partial t ^ {2}} [ \varepsilon (z, t) \mathbf {E} (z, t) ] = 0.\tag{105}
$$

Substituting (104) and (103) into (105), the band structure can be calculated by using mode-matching analysis [221].

The traveling wave that modulates the material properties propagates at a speed $v_{m} = \frac{\omega_{m}}{k_{m}}$ . Therefore, we can classify the possible scenarios for the dispersion relation into three types, depending on the relation between the modulation speed $v_{m}$ and the speed of light in the same medium in the absence of temporal modulations $v_{ph} = c/\sqrt{a_{0}}$ , where $a_{0}$ is the zeroth Fourier coefficient in (103) which is the temporal average of the dielectric function [222–224]. These scenarios are typically referred to as subluminal ( $v_{m}<v_{ph}$ ), luminal ( $v_{m}=v_{ph}$ ), and superluminal ( $v_{m}>v_{ph}$ ). We can start the discussion in the limit of a vanishing modulation frequency ( $\omega_{m}\rightarrow0$ ) and a nonzero spatial modulation frequency ( $k_{m}\neq0$ ), resulting in vanishing speed $v_{m}$ . Then, the material under consideration becomes a traditional (spatial) photonic crystal. The band structure of such material under weak spatial modulation represents merely a folded dispersion curve, which represents the dispersion relation of a material when its periodic structure causes the original dispersion relation to “fold” into a smaller region, typically within the first Brillouin zone [1, p. 146]. This folded dispersion curve is seen in the red curves in Fig. 31(a). The band structure is periodic with respect to k with a period $k_{m}$ , as depicted by the reciprocal vector of ( $k_{m},0$ ) in the figure. At the band crossings, the energy (frequency) bandgaps appear.

![](images/37e653c7f1bc063e0f9115e43c3f1e5d0f5b339ab25e688788af220fa0559979.jpg)

![](images/499d50d9aa8602d4fddbd04e57862b612718f2f92d100b68a0bf1813258a45c3.jpg)

![](images/1f8cf41bdf27d4499cfc0d41383d445e57dbb7ab0515c79c41483d8765898296.jpg)

![](images/32e982a2b050157db02234e2a91f37e4f886958aa13031bcc93692d2d1700c53.jpg)
(a) Dispersion relation of a photonic spatial crystal where $\omega_{m}/k_{m}=0$ with $\omega_{m}=0$ and $k_{m}\neq0$ . (b) Same dispersion relation but now for a ST-PC with a traveling-wave modulation. Here, the modulation wave propagates at speeds lower than the phase velocity in the time-invariant medium, i.e., $\omega_{m}/k_{m}<v_{ph}$ . (c) Same as in (b) but for the superluminal scenario ( $\omega_{m}/k_{m}>v_{ph}$ ). (d) Dispersion relation of a PTC with $\omega_{m}/k_{m}=\infty$ . The bandgaps between red and blue bands correspond to energy and momentum bandgaps, respectively. Here, “PA” means parametric amplification. Figure 1 reprinted with permission from Galiffi et al., Phys. Rev. Lett. 123, 206101, 2019 Ref. [223]. Copyright (2019) by the American Physical Society.

When the speed of the modulation wave increases but still being smaller than $v_{ph}$ , the band structure under weak modulation approximation is formed by folding the dispersion curve of a stationary material into a new tilted Brillouin zone [23] with a reciprocal vector ( $k_{m}$ , $\omega_{m}$ ) (see the red arrow in Fig. 31(b)). A tilted Brillouin zone occurs when there is both spatial and temporal modulation in the material. This causes the original Brillouin zone (which is defined for purely spatial modulation) to tilt due to the additional temporal component. Likewise, at the crossings, energy bandgaps are formed. One can see that in this case, the band structure is asymmetric with respect to the frequency and momentum axes. One of the most important physical effects induced by the asymmetric band structure is nonreciprocity [225]. The asymmetric band structure of a typical traveling-wave modulated media is shown in Fig. 32(a). When the excitation frequency is $\omega_{0}$ (indicated by the black dashed line in the figure), it corresponds to a real wavenumber in the negative k domain and a complex wavenumber in the positive k domain. The real eigenwavenumber means that the wave can propagate through the media without attenuation, whereas the complex eigenwavenumber means that the wave exponentially decays in space. Therefore, as shown in Fig. 32(b), the device works as a wave isolator. For an incident wave coming from the right-hand side (corresponding to negative k), the wave passes through the media. For incidence from the left-hand side (corresponding to positive k), the wave is strongly attenuated. Similar nonreciprocal effects have been reported in traveling-wave modulated metasurfaces platform [86,124,226–228].

![](images/ad2256dcc0b0acb092d1efffbf36143c74acdc41581a3fde3369aecf77ce7397.jpg)
(a) Asymmetric energy bandgaps in a subluminal traveling-wave modulated ST-PC. The vertical axis represents the frequency, and the horizontal axis represents either real (solid lines) or imaginary (dotted circles) part of the propagation constant k. (b) Nonreciprocal wave propagating through the same crystal of finite size. Figure 1 reprinted with permission from Chamanara et al., Phys. Rev. B 96, 155409, 2017 Ref. [221]. Copyright (2017) by the American Physical Society.

When $v_{m}>v_{ph}$ , the modulation is superluminal. The conceptual band structure for this scenario is shown in Fig. 31(c). Similar to the subluminal modulation, the band structure is asymmetric, which means that such modulation can also induce nonreciprocity. However, in contrast to the subluminal case, for superluminal modulation, the crossing of bands generates a momentum bandgap. Therefore, superluminal modulation can induce nonreciprocal amplification [229].

Regardless of whether the modulation is subluminal or superluminal, it was discovered in Ref. [230] that a space–time modulated metamaterial with both permittivity and permeability modulation can always generate the Fresnel drag effect, typically observed in moving media. This finding suggests that space–time traveling wave modulation can be represented by effective bianisotropic parameters with nonreciprocal magnetoelectric coupling, which can, in turn, be mapped to a moving homogeneous medium. A more complete study of moving interface between two materials were considered in a recent paper [231].

Between the subluminal and superluminal regimes, there is a special scenario that the modulation phase velocity is equal to the speed of light, which is called luminal, i.e., $v_{m} = v_{ph}$ . This is an exceptional case where exotic wave effects can happen [223]. As shown in Fig. 33(a), the reciprocal vector (green arrow) aligns with the dispersion curve of a stationary material. The band structure of a space–time varying media is formed by folding the dispersion curve of the stationary material by reciprocal vectors. The crossing of the bands results in that all the forward-traveling states are degenerate in a broadband region and therefore strongly coupled. Therefore, luminal modulation can induce broadband nonreciprocal amplification [223]. Moreover, a plane wave propagating through a luminally modulated medium can be transformed into a pulse

## Figure 33

![](images/e711d5e1ef1ecb94c145394a2a0c3fa216f934784fc602602552425de0b46bfe.jpg)

![](images/f8845af1a0c3b0c90840d1c2dae11ede64ddb9b2b52fd00bae8b84ef809326d7.jpg)

![](images/4816345481669884c3eb2580df77e09dc82dbb0943a97234e4d5863a9dbaa3b1.jpg)
(a) Band degeneracy for the case of luminal traveling-wave modulation. (b) Field amplitude at the right boundary of the ST-PC at different time moments. (c) Permittivity variation at different time moments. (d) Luminal grating when illuminated with a continuous wave at low frequency. The output is a compressed pulse train. (b) and (c) Figure 2 reprinted with permission from Galiffi et al., Phys. Rev. Lett. 123, 206101, 2019 Ref. [223]. Copyright (2019) by the American Physical Society. (d) Reprinted from [232] under a Creative Commons license.

train, as illustrated in Fig. 33(d). This phenomenon is elucidated by the dynamics presented in Fig. 33(b), where the field captured at the output face $(x = d)$ of the ST-PC exhibits modulation-induced variations. Specifically, at instances when $\omega_{m}t = \pi/2$ , the signal undergoes exponential growth, whereas at $\omega_{m}t = 3\pi/2$ , it experiences exponential suppression. This effect stems from the observations in Fig. 33(c), indicating that field amplitudes within the range $-\pi/2 < \omega_{m}t < \pi/2$ encounter a reduced permittivity and, consequently, a heightened phase velocity, leading to wave compression and amplification. Conversely, amplitudes within the range $\pi/2 < \omega_{m}t < 3\pi/2$ are subjected to increased permittivity, resulting in decreased phase velocity and field attenuation.

Revisiting Fig. 31, in the scenario where $k_{\mathrm{m}} = 0$ , the modulation velocity becomes infinite. This situation means that in a PTC, the material's properties change uniformly over time without any repeating patterns in space. Thus, we can see that PTCs represent a special scenario of a broader notion of ST-PCs.

## 7.2. Arbitrary Spatiotemporal Modulation

The traveling-wave modulation, as discussed in the previous section, is a special type of space–time modulation function. In general, the modulation function of an ST-PC does not necessarily need to be in the traveling-wave form. Indeed, the modulation can be an arbitrary periodic function with a temporal period $T_{m}$ and spatial period $P_{m}$ , respectively. The scattering property of such generalized space–time periodic media was studied, e.g., in Ref. [233]. Next, we aim to derive the eigenvalue problem of such generalized space–time modulation, following the steps in Ref. [234]. For isotropic and nondispersive materials, Maxwell's curl equations simplify to

$$
\frac {\partial}{\partial x} \left[ \begin{array}{c} \mathbf {E} \\ \mathbf {H} \end{array} \right] = - \frac {\partial}{\partial t} \left[ \begin{array}{c c} 0 & \mu (x, t) \\ \varepsilon (x, t) & 0 \end{array} \right] \left[ \begin{array}{c} \mathbf {E} \\ \mathbf {H} \end{array} \right].\tag{106}
$$

Assuming the permittivity and permeability are modulated with identical spatial and temporal periods, the material parameter matrix can be expressed as a 2D Fourier

series:

$$
\left[ \begin{array}{c c} 0 & \mu (x, t) \\ \varepsilon (x, t) & 0 \end{array} \right] = \sum_ {m, n} \overline {{\overline {{U}}}} _ {m, n} e ^ {j (n k _ {\mathrm{m}} x - m \omega_ {\mathrm{m}} t)},\tag{107}
$$

where $k_{m} = 2\pi/P_{m}$ and $\omega_{m} = 2\pi/T_{m}$ represent the spatial and temporal angular modulation frequencies, respectively, and $\overline{\overline{U}}_{m,n}$ are $2 \times 2$ anti-diagonal matrix storing the Fourier coefficients of permittivity and permeability, with n and m being integers that range from $-\infty$ to $+\infty$ .

Using the Bloch–Floquet theorem, the solutions of Eq. (106) can be written as

$$
\left[ \begin{array}{c} \mathbf {E} \\ \mathbf {H} \end{array} \right] = e ^ {j (k _ {\mathrm{B}} x - \omega_ {\mathrm{F}} t)} \sum_ {m, n} \boldsymbol {\Psi} _ {m, n} e ^ {j (n k _ {\mathrm{m}} x - m \omega_ {\mathrm{m}} t)},\tag{108}
$$

where $k_{B}$ and $\omega_{F}$ are the Bloch wavenumber and Floquet frequency of the Bloch–Floquet mode, respectively, and $\Psi_{m,n}$ is $2 \times 1$ column vector containing the harmonic amplitudes of electric and magnetic fields for harmonic index $(m,n)$ . Substituting Eqs. (108) and (107) into Eq. (106) gives

$$
\sum_ {m, n} (k _ {\mathrm{B}} + n k _ {\mathrm{m}}) \pmb {\Psi} _ {m, n} = \sum_ {m, n} (\omega_ {\mathrm{F}} + m \omega_ {\mathrm{m}}) \sum_ {p, q} \overline {{\overline {{U}}}} _ {m - p, n - q} \pmb {\Psi} _ {p, q}.\tag{109}
$$

In principle, the ranges of $m, n, p, q$ can extend from $-\infty$ to $+\infty$ . However, to derive a finite number of equations, these indices are truncated. Specifically, the spatial order is limited to $\{n, q\} \in [-Q, Q]$ and the temporal order to $\{m, p\} \in [-P, P]$ , where $P$ and $Q$ are positive integers. Consequently, Eq. (109) corresponds to a finite set of equations. For each field quantity (electric or magnetic field), the number of equations is $S = (2P + 1)(2Q + 1)$ . These equation sets can be expressed through matrix operations. By defining convolution matrices $\overline{\overline{C}}_{\varepsilon,\mu}$ and a 2S-dimensional column vector of field Fourier components $\Phi = (\mathbf{E}_a, \mathbf{H}_a)^T$ (here, $\mathbf{E}_a$ and $\mathbf{H}_a$ are $S$ -dimensional row vector storing the Fourier amplitudes of electric and magnetic waves), the matrix operation is formulated as follows:

$$
[ \overline {{\overline {{I}}}} _ {2} \otimes (k _ {\mathrm{B}} \overline {{\overline {{I}}}} _ {S} + \overline {{\overline {{G}}}}) ] \boldsymbol {\Phi} = [ \overline {{\overline {{I}}}} _ {2} \otimes (\omega_ {\mathrm{F}} \overline {{\overline {{I}}}} _ {S} + \overline {{\overline {{W}}}}) ] \cdot \left[ \begin{array}{c c} 0 & \overline {{\overline {{C}}}} _ {\mu} \\ \overline {{\overline {{C}}}} _ {\varepsilon} & 0 \end{array} \right] \boldsymbol {\Phi}.\tag{110}
$$

Here, the symbol $\otimes$ represents the Kronecker product. The subscript of the unit matrix $\overline{\overline{I}}$ represents its dimensionality. Matrices $\overline{\overline{G}}$ and $\overline{\overline{W}}$ are square matrices of size $S$ which are defined as $\overline{\overline{G}} = \overline{\overline{I}}_{2P + 1}\otimes \mathrm{diag}(nk_{\mathrm{m}})$ and $\overline{\overline{W}} = \mathrm{diag}(m\omega_{\mathrm{m}})\otimes \overline{\overline{I}}_{2Q + 1}$ . Matrices $\overline{\overline{C}}_{\varepsilon ,\mu}$ are $S$ -dimensional square matrices. After rearranging (110), the eigenvalue problem can be established,

$$
\left[ \begin{array}{c c} 0 & \overline {{\overline {{C}}}} _ {\mu} \\ \overline {{\overline {{C}}}} _ {\varepsilon} & 0 \end{array} \right] ^ {- 1} \cdot \left[ \begin{array}{c c} k _ {\mathrm{B}} \overline {{\overline {{I}}}} _ {S} + \overline {{\overline {{G}}}} & - \overline {{\overline {{W}}}} \cdot \overline {{\overline {{C}}}} _ {\mu} \\ - \overline {{\overline {{W}}}} \cdot \overline {{\overline {{C}}}} _ {\varepsilon} & k _ {\mathrm{B}} \overline {{\overline {{I}}}} _ {S} + \overline {{\overline {{G}}}} \end{array} \right] \boldsymbol {\Phi} = \omega_ {\mathrm{F}} \boldsymbol {\Phi}.\tag{111}
$$

The above derivation originally obtained in Ref. [234] is for the most general periodic space–time modulation. However, it is important to consider also a special scenario where the modulation function of the permittivity can be separated in both space and time variables, i.e., $\varepsilon(x,t)=\varepsilon_{x}(x)\varepsilon_{t}(t)$ . The method to analytically determine the band structure under the separable modulation function of this form was presented in Refs. [235,236]. With such modulation, it is possible to generate the so-called “mixed” bandgaps which show the features of both momentum and energy bandgaps [236]. Figure 34(a) shows that under proper space–time modulation, the energy and momentum bandgaps overlap in the frequency–momentum plane. In the overlapping region, the eigenmode features complex frequency and complex wavenumber. This results in unique propagation phenomena where exponential growth induced by temporal modulation and exponential decay caused by spatial modulation can counteract each other. When the temporal growing mode dominates, its momentum and energy exponentially grow, as shown in Fig. 34(b). Moreover, under very specific modulation parameters, the counteracting forces of growing in time, and decaying in space are exactly matched, causing the bandgap to close, as shown in Fig. 34(c). Inside this closed mixed bandgap, a pulse wave stops and expands its width in space, maintaining the constant amplitude. This results in the linear growth of its momentum and energy, as shown in Fig. 34(d).

![](images/1967b1e5d01e81b69caa5bddcb8c2a442bb425f2ab390f5ed9fea89a3caccd87.jpg)

![](images/e49058b934e99fb51de4ccab6d586bb8df1e62b77e75dddeeb59b1c9cb6a5d21.jpg)

![](images/467ee844e9f2341a9e05c68f4e8be5e89656f453e9cc698466b2fc5f1e76b971.jpg)

![](images/c4bc20f22694582ebcf1c7c819db11d260750a6b102cca579932ef08159a1edb.jpg)
(a) Mixed momentum–energy bandgap. The spatiotemporal permittivity modulation is in the form of $\varepsilon(x,t)=\varepsilon_{x}(x)\varepsilon_{t}(t)$ , where $\varepsilon_{x}(x)=(1+2B)/[1+B(1+\cos(k_{\mathrm{m}}x))]$ with B=0.5 and $\varepsilon_{t}(t)=1+A[1+\cos(\omega_{\mathrm{m}}t)]$ with A=0.5. (b) Exponential growth of momentum M and energy E of the eigenmode inside the mixed bandgap with the modulation starting at t=0. (c) Closed mixed bandgap. Here, the spatial and temporal variation functions must be inversely proportional to each other. (d) Linear growth of momentum and energy in the closed mixed bandgap with the modulation starting at t=0. Reprinted with permission from [236]. © Optica Publishing Group.

## 7.3. Enhancing the Size of a Momentum Bandgap Using Resonant ST-PCs

Another class of ST-PCs with a separable spatiotemporal modulation are those whose susceptibility is of the form $\chi(\mathbf{r},t)=\chi_{\mathrm{r}}(\mathbf{r})\chi_{\mathrm{t}}(t)$ . Here, $\chi_{\mathrm{r}}(\mathbf{r})$ corresponds to the spatial part of the susceptibility encoding the spatial modulation of the system. Further, $\chi_{\mathrm{t}}(t)$ is the temporal part of the susceptibility encoding the temporal modulation of the system. Note that the modulation of susceptibility in such a form is qualitatively different from the modulation of permittivity in the same form discussed in the previous section. One example of a system having such a separable spatiotemporal profile of susceptibility is a metasurface made from a periodic arrangement of time-varying spheres embedded in vacuum (see Fig. 35(a)) [69,237]. Here, each sphere is assumed to have a time-dependent susceptibility $\chi_{\mathrm{t}}(t)=\chi_{\mathrm{st}}[1+M_{\mathrm{s}}\cos(\omega_{\mathrm{m}}t)]$ . Here, $\chi_{st}$ is the electrical susceptibility of the static spheres, $M_{s}$ is the modulation depth, and $\omega_{m}$ is the modulation frequency. Moreover, the susceptibility variation of the ST-PC due to the spatial arrangement of the spheres can be physically expressed in the form of a space-dependent susceptibility $\chi_{\mathrm{r}}(\mathbf{r})$ .

As discussed earlier, the existence of momentum bandgaps is one of the key features of PTCs $[20]$ . However, the PTCs often require large modulation depths of the material parameters to show discernible bandgaps $[61]$ . As was discussed in Section 5.2, such a requirement of large modulation depth poses two major challenges. First, the modulation of material parameters with high modulation depths requires enormously high pump powers. Second, the intrinsic losses of the material may lead to its rapid thermal damage in the presence of such high-power pumps. To overcome these challenges, the structural resonances of ST-PCs can be harnessed. This is because they lower the required modulation depths drastically to attain large momentum bandgaps $[69,238]$ .

(a)
![](images/1ad7e99f659eea2570d791ce4a6055c4168ceea223993982f4fa58efde975623.jpg)

![](images/1ef3f4d2fcf3ad466346da7000cfe2dc9bcc8380549e7a963c9602540ec8c2ae.jpg)

(c)
![](images/ffbd0c1a7e6ee1a67025c01e2f6ad545bb382da8fbf9b9ae28b0dec984d54fa2.jpg)

![](images/b5a242e89203e7219c3fefff33440f8b7a05637f08969688ac31ec5959bc13f8.jpg)

![](images/40e763a0fa9b7ebc1dd77ace3537273066368f0fc4f5d649f6a940e2f2f65684.jpg)
(a) ST-PC based on a metasurface consisting of dielectric spheres with time-varying material susceptibility. (b) Band structure of the time-invariant metasurface. (c) Dipolar Mie coefficients of an isolated static sphere. (d)–(f) Band structure of the nonresonant ST-PC with the modulation frequency $\omega_{m1}$ . (g)–(i) Band structure of the resonant ST-PC with the modulation frequency $\omega_{m2}$ . Note that a represents the spatial period of the metasurface in (a). Further, $S_{min}$ is a metric based on the T-matrix of the ST-PC. A local minimum of $S_{min}$ determines the spectral location of an eigenmode of the ST-PC (see Ref. [69] for details). Reprinted from [69] under a Creative Commons license.

In Ref. [69], the metasurface-based design of the ST-PC shown in Fig. 35(a) was used to attain large momentum bandgaps for very low material modulation strengths. In the following, we discuss the proposed method in Ref. [69]. First, the band structure of a static metasurface (i.e., with $M_{\mathrm{s}} = 0$ ) is plotted in Fig. 35(b). Here, the flatbands appear as prominent features. These flatbands occur due to the dipolar Mie resonances of the static spheres of the metasurface (see Fig. 35(c)). Next, the temporal modulation was switched on with the modulation depth as low as $M_{s} = 0.01$ . In the first (nonresonant) scenario, the modulation frequency was chosen as $\omega_{\mathrm{m}} = \omega_{\mathrm{m1}}$ . Note that such a configuration of the metasurface corresponds to a nonresonant case because $\omega_{\mathrm{m1}}$ is spectrally far away from the locations of the flatbands (see Fig. 35(b)). The band structure for such a nonresonant ST-PC is plotted in Fig. 35(d). From Figs. 35(e) and (f), we observe a small momentum bandgap.

In the second (resonant) scenario, the modulation frequency was chosen as $\omega_{m} = \omega_{m2}$ , while keeping the same modulation strength $M_{s} = 0.01$ . Note that, since $0.5\omega_{m1}$ is at the spectral location of one of the flat bands in Fig. 35(b), such a configuration corresponds to a resonant case. In Fig. 35(g), the band structure of such a resonant

ST-PC is plotted. From Figs. 35(h)–(i), a wide momentum bandgap is observed. As reported in Ref. [69], such structural resonances of the metasurface lead to the enhancement of the bandgap size by a factor of 350.

## 7.4. Topological Aspects of PTCs

The understanding of topological phases of photonic systems has given rise to various interesting phenomena in optics such as unidirectional light propagation $[239]$ and unidirectional lasing $[240]$ . An important characteristic of the topological phases of matter is that they are robust against defects and disorders in the underlying system $[241]$ . Therefore, studying the topology of physical systems is crucial for classifying them for stability against such defects and disorders.

Recently, the topological phases of the PTCs have also been studied $[21,242,243]$ . Since the PTCs correspond to a 1D periodic system (with periodicity about temporal dimension), the topological invariant of interest in such a system is the Zak phase $[244]$ . The Zak phase of each band of a PTC is given by $[21, Eq. (5)]$

$$
\theta_ {p} ^ {\mathrm{Zak}} = \int_ {- \pi / T _ {\mathrm{m}}} ^ {\pi / T _ {\mathrm{m}}} \mathrm{d} \omega_ {\mathrm{F}} \left[ j \int_ {0} ^ {T _ {\mathrm{m}}} \mathrm{d} t \varepsilon (t) D _ {p, \omega_ {\mathrm{F}}} ^ {*} (t) \frac {\partial D _ {p , \omega_ {\mathrm{F}}} (t)}{\partial \omega_ {\mathrm{F}}} \right].\tag{112}
$$

Here, $\varepsilon(t)$ is the time-periodic permittivity of the PTC, p is the band index (lowest band index being p = 0), and $D_{p,\omega_{\mathrm{F}}}(t)$ is the displacement field of the pth band of the PTC at the Floquet frequency $\omega_{F}$ . Note that the quantity $\theta_{p}^{Zak}$ in Eq. (112) is the adaptation of the spatial Zak phase to the temporally periodic systems [21]. Using Eq. (112), we can calculate the Zak phases of the bands of the PTC with the same material modulation as shown in Fig. 1(b). The Zak phases of the bands of the considered PTC are shown beside the corresponding bands in Fig. 36(a).

The Zak phase $\theta_{p}^{Zak}$ dictates the sign of the relative phase between the reflected and transmitted waves for an incident excitation that falls inside the momentum bandgap of the considered PTC [21]. For this purpose, consider a temporal slab made from a finite number of cycles of a stepwise modulation (see Fig. 13(a)). Further, let us assume an incident excitation to the temporal slab to be $E_{\mathrm{inc}} = E_{0}e^{i(\omega_{\mathrm{inc}}t - k_{\mathrm{inc}}z)}$ . Furthermore, let us assume that $k_{inc}$ falls within one of the momentum bandgaps of the PTC formed by the infinite periodic extension of the considered stepwise permittivity profile shown in Fig. 13(a). Further, from Eq. (74), we know that a plane wave incident to the temporal slab gives rise to reflected and transmitted plane waves. Let r and t to be the complex reflection and transmission coefficients of the reflected and transmitted plane waves, respectively. Moreover, let the phase difference between r and t to be $\phi_{s}$ , i.e., $\angle r - \angle t = \phi_{s}$ . Then the sign of $\phi_{s}$ is dictated by the Zak phase $\theta_{p}^{Zak}$ as [21]

$$
\operatorname{sgn} \left(\phi_ {s}\right) = \eta (- 1) ^ {l + s} \exp \left(j \sum_ {p = 1} ^ {s - 1} \theta_ {p} ^ {\text { Zak }}\right).\tag{113}
$$

Here, s is the gap number (lowest gap number being 1), $\eta = \operatorname{sgn}(1 - \varepsilon_{1}/\varepsilon_{2})$ , and l is the number of bands below gap s (see Fig. 36(a)).

Next, the authors of Ref. [21] studied the topological edge state of the system that consists of two different PTCs with different topology occurring one after the other (see Fig. 36(b)). Note that the first PTC has $\varepsilon_{1} = 3$ for time $t_1 = 0.5T_{\mathrm{m}}$ and $\varepsilon_{2} = 1$ for time $t_2 = 0.5T_{\mathrm{m}}$ . On the other hand, the second PTC has $\varepsilon_{1} = 1$ for time $t_1 = 0.5T_{\mathrm{m}}$ and $\varepsilon_{2} = 3$ for time $t_2 = 0.5T_{\mathrm{m}}$ . Further, the interface of the two PTCs occurs at the time $t_\mathrm{edge} = 8T$ . Clearly, the two PTCs shown in Fig. 36(b) have the same bandgaps but different topologies. For such a system, in Ref. [21] a topological edge state was reported as an eigenstate. The corresponding edge state is shown in Fig. 36(c). From Fig. 36(c), we observe that for the topological edge state, the field $|D|$ increases exponentially up to time $t = t_{edge}$ followed by an exponential decay afterwards. Further, after the field amplitude has decayed it starts increasing again as a function of time. Note that such a topological edge state is robust with respect to the defects and impurities in the underlying time-varying system.

(a)
![](images/2c0560867838b900ec9ad6c8052b9d3744269ae0b56ffe9f1c595b59b54c6629.jpg)

![](images/327530dac9ebd32ff47fc6f92f77bb72e85bc67c064dff2f0bef4b128faa4967.jpg)

![](images/97e494521b2a0ea37ca7c6f43a68d155390e1a6579ee959b9458f8b5720b664f.jpg)
(a) Band structure of a PTC with stepwise modulation. Here, Zak phases $\theta_{p}^{Zak}$ of the bands are shown beside the relevant bands. (b) Permittivity profile $\varepsilon(t)$ as a function of time t of the composite system made from two PTC that occur one after another. (c) Variation of the displacement field amplitude $|D|$ as a function of time t of the topological edge state that occurs as an eigenmode of the composite system shown in (b). Reprinted with permission from [21]. © 2018 The Optical Society.

Furthermore, the authors of Ref. [242] discussed the conditions for the topological phase transitions in PTCs. Let us again consider a PTC made from the stepwise modulations of the permittivity as shown in Fig. 1(a). Let us assume that $\varepsilon_{2} = 4$ and $\varepsilon_{1} = g\varepsilon_{2}$ , where $g$ characterizes the modulation strength of the PTC. The size $\Delta k$ of the second momentum bandgap of the PTC (lying between the bands $p = 1$ and $p = 2$ ) as a function of $g$ and the modulation frequency $f_{\mathrm{m}} = \omega_{\mathrm{m}} / 2\pi$ is plotted in Fig. 37 [242]. From Fig. 37, we find that when $g = f_{\mathrm{m}} / (1 - f_{\mathrm{m}})$ (dashed curve), the gap size $\Delta k$ is zero. Therefore, the topological phase transition occurs whenever the condition $g = f_{\mathrm{m}} / (1 - f_{\mathrm{m}})$ is satisfied. This implies that the bands $p = 1$ and $p = 2$ have different topological phases $\theta_p^{\mathrm{Zak}}$ on either side of the dashed line. As discussed earlier, the authors of Ref. [242] suggested that such a topological phase transition can be probed by measuring the phase difference $\phi_s$ between the complex reflection and transmission coefficients of the temporal slabs formed by truncating the modulation of the PTC in time (see Fig. 11). In contrast to Refs. [21,242], the authors of Ref. [245] studied the topological phases (see Eq. (112)) of the PTCs with continuous profiles of $\varepsilon(t)$ such as sinusoidal and exponential modulation. Moreover, in Ref. [246] Lin et al. discussed the effects of temporal defects on the topological features of the PTCs.

## Figure 37

![](images/20b1a0ecad371c1a4c100edd4d478779a270bb8604a4ff7a109a026d92162ce9.jpg)
Variation of the second momentum bandgap size $\Delta k$ of the PTC as a function of the parameter g and the modulation frequency $f_{m} = \omega_{m}/2\pi$ . Here, the dotted black line corresponds to the condition of topological phase transition as the gap size vanishes, i.e., $\Delta k \rightarrow 0$ . Reprinted with permission from [242]. © 2019 The Optical Society.

## Figure 38

![](images/9a0f9389493c7bf8263ce1588a93ed832cceabe85f8689880d836c3f89a1d245.jpg)
(b)

![](images/933009905116d67f2d30b4daffa50e602d65881e8b2cb76fe058f4f331b489a5.jpg)
(a) Unit cell of an ST-PC formed by cylindrical ring resonators. (b) Band structure of the ST-PC shown in (a) when the spatiotemporal modulation of the ring resonators is turned off, i.e., $\omega_{m}^{\pm}=0$ (solid blue lines), and when the spatiotemporal modulation of the ring resonator at the center of the unit cell is turned on, i.e., $\omega_{m}^{+}\neq0$ , $\omega_{m}^{-}=0$ (dotted blue lines). Figure 6 reprinted with permission from Serra and Silveirinha, Phys. Rev. B 107, 035133, 2023 Ref. [247]. Copyright (2023) by the American Physical Society.

Similar to PTCs, the topological effects in ST-PCs have also been studied. The authors of Ref. [247] studied the ST-PCs made from the inclusions that are subjected to rotating wave modulation (see Fig. 38(a)). The unit cell of the ST-PC consists of cylindrical ring resonators. Here, each resonator is subject to a spatiotemporal modulation of the permittivity and permeability given by $\varepsilon = \varepsilon (\phi -\omega_{\mathrm{m}}t)$ and $\mu = \mu (\phi -\omega_{\mathrm{m}}t)$ , respectively. Here, $\phi$ is the azimuthal angle in the cylindrical coordinates and $\omega_{\mathrm{m}}$ is the angular frequency of the rotating spatiotemporal modulation. For such spatiotemporal crystals, the relevant topological invariant to study is the gap Chern number [247,248]. Note that, here, the term "gap" refers to the energy bandgaps arising in the ST-PCs. The gap Chern number for a specific bandgap of the considered ST-PCs can be defined as [247]

$$
C _ {\mathrm{gap}} = \frac {1}{2 \pi} \iint_ {\mathrm{BZ}} F _ {\mathbf k} \mathrm{d} ^ {2} {\mathbf k}.\tag{114}
$$

Here, BZ refers to the spatial Brillouin zone of ST-PC, and $F_{\mathrm{k}}$ corresponds to the Berry curvature associated with the bandgap. Note that $F_{\mathrm{k}}$ can be computed using the Green's function of the ST-PC (see Ref. [247] for more details). Next, to utilize the aforementioned gap Chern number, the band structure of the ST-PC is plotted (see Fig. 38(b)). In Fig. 38(b), the solid blue lines correspond to the case when the spatiotemporal modulations of the ring resonators are turned off (i.e., $\omega_{\mathrm{m}}^{\pm} = 0$ ). On the other hand, the dotted blue lines correspond to the case when the spatiotemporal modulation of the ring resonators at the center of the unit cell of the ST-PC is turned on (i.e., $\omega_{m}^{+} \neq 0$ , $\omega_{m}^{-} = 0$ ). From Fig. 38(b), we observe that the spatiotemporal modulation opens an energy bandgap. The gap Chern number of the bandgap is then calculated using Eq. (114). The value of the gap Chern number turns out to be $C_{gap} = -1$ . Furthermore, it was shown in Ref. [247] that upon manipulating the relative signs of $\omega_{m}^{+}$ and $\omega_{m}^{-}$ , one can engineer the sign of $C_{gap}$ .

Further, in Ref. [249], the authors engineered nontrivial topological phases in ST-PCs. They showed the emergence of scattering immune topological edge states in such ST-PCs.

## 7.5. Nonlinear ST-PCs

In addition to the linear ST-PC, the ST-PCs made from nonlinear media have also been studied. In particular, in Ref. [250], the emergence of gap soliton solutions of the underlying nonlinear wave equation satisfied by such nonlinear ST-PCs was demonstrated. The considered system is assumed to have a space–time-dependent linear refractive index $n(z,t)$ (see Fig. 39(a)). Therefore, the linear part of the electric polarization is written as $P_{\mathrm{L}} = [n^{2}(z,t) - 1]E$ . Further, the system is assumed to have a third-order nonlinearity such that the nonlinear electric polarization is written as $P_{NL} = \chi_{NL}E^{3}$ . Here, $\chi_{NL}$ , is the third-order nonlinear susceptibility of the considered system. Such systems support mixed energy–momentum bandgaps (see Fig. 39(b)). Note that the band structure shown in Fig. 39(b) exhibits a bandgap with respect to both Floquet frequency $\omega_{F}$ (energy bandgap) and Bloch wavenumber $k_{B}$ (momentum bandgap). Moreover, the soliton solutions found inside such mixed bandgaps are termed spatiotemporal gap solitons.

The applicability of the developed method to find the spatiotemporal gap soliton solutions in Ref. [250] is shown in Figs. 39(c) and (d). The authors of Ref. [250] used the rotated coordinate system $(p,q)$ to compute the soliton solutions (see Fig. 39(a)). Note that in Figs. 39(c) and (d), $\tau$ quantifies the propagation length of the soliton as it travels. On the other hand, $\xi$ quantifies the localization of the soliton solution (see the inset of Fig. 39(c)). In Fig. 39(c), the dynamics of the intensity of an initial gap soliton in the absence of the ST-PC are shown. Here, the inset of Fig. 39(c) shows the input profile of the initial soliton. Note that the blue (red) curve denotes the forward-(backward-)propagating envelope of the electric fields of the soliton. From Fig. 39(c), we observe that in the absence of the ST-PC, the forward and backward components of the initial soliton separate and do not interact as they propagate. Hence, expectedly, the initial soliton does not preserve its localization property. On the other hand, Fig. 39(d) demonstrates such intensity dynamics for the case when the soliton is coupled to the ST-PC. From Fig. 39(d), the undisturbed soliton dynamics is observed. Therefore, the initial soliton stays well localized as it propagates inside the ST-PC. Such undisturbed dynamics convincingly show the robustness of the proposed method to find the soliton solutions inside the mixed bandgaps of the ST-PCs [250].

## 8. FUTURE OUTLOOK

After elaborating on all the work done in the context of PTCs and their foundational principles, in this short section we summarize aspects that we, as an entire community, shall address in the future to develop the topic of PTCs further. With the purpose of avoiding wrong impressions, we wish to fully acknowledge that much of the research has been driven so far by theoretical and computational studies. At the same time, the first experiments that address specific aspects of PTCs were reported, while many have yet to be done. Therefore, the outlook is written in a humble manner in that we do not wish to suggest that PTCs will change the world. Still, working on PTCs is fascinating and it is interesting to disclose new physical effects in these materials.

Figure 39

![](images/f6190de199549ee7e64a02a5b2d5fb5d897f74b83e973f4706c407fde257165a.jpg)

![](images/6f2e0655aef666ee9ecdf52ca178948626bee59bdaace285ca7703c94812aa9e.jpg)

![](images/d297a57606b4df59f03029e44f05a7acc9c7227fecf79976dfc90fc87d672db3.jpg)

![](images/ab4c8fc316cf0d9b75e6204e7067bdc10a685b25eade8da6d5533a7b51e24a8a.jpg)
(a) ST-PC made from a nonlinear material. (b) Band structure of the nonlinear ST-PC. (c) Contour plot of the total intensity of an initial gap soliton as it propagates in the absence of the ST-PC (the inset shows the forward and backward intensity profiles of the input with blue and red curves, respectively. (d) Corresponding contour plot of the total intensity in the presence of the ST-PC. Here, $(\tau,\xi)$ quantifies the propagation length and localization of the gap soliton computed with respect to the rotated coordinate system $(p,q)$ shown in (a), respectively. Figures 1 and 2 reprinted with permission from Biancalana et al., Phys. Rev. A 77, 011801, 2008 Ref. [250]. Copyright (2008) by the American Physical Society.

First, we shall continue to explore the degrees of freedom PTCs offer us to control light propagation. We shall do so out of intellectual curiosity to shift the boundaries that define our current understanding of the world surrounding us. But of course, also with an eye on possible applications and devices that exploit these phenomena in their design. When looking back in time, we can rationalize how novel research themes in the context of nanophotonics were established. Most new trends emerged by writing down the constitutive relations and asking us how these constitutive relations can be modified to enlarge the space of possible effects. One line of future developments in the field of PTCs signifies the transition from isotropic to anisotropic materials, further toward bianisotropic $[251–253]$ and eventually nonlocal $[254]$ and magnetic materials $[255]$ . In a generalization, higher-order nonlocal materials can be explored. In addition, assuming materials to be periodic in space gave significant momentum to many subfields of optics by considering them as photonic crystals. Adding a time variation to those material properties, generally to all, adds many novel opportunities. While most of these previously considered extensions concentrated on the spatial propagation characteristics of light propagation, a time variation adds control over the spectral composition of the field to the portfolio. Compared with nonlinear optical effects that provide similar capabilities, the advantage of a time modulation would be much more rational control of the spectral composition. The spectral content of the light is controlled deterministically by changing the waveform and the period of the time modulation on demand. In addition to exploring these aspects, we would like to motivate the wider community to explore other possible extensions in the constitutive relations.

When thinking about more practical aspects, the most urgent work that needs to be accomplished is the establishment of an ordinary PTC characterized by a time-dependent permittivity at optical frequencies. But generally, it would be great to explore more experimental systems that permit the observation of effects offered by PTCs. This holds for experimental platforms operating at different frequency domains and for experimental platforms that allow us to access wave phenomena outside that of electromagnetics and optics. Acoustics or fluid dynamics would be two examples. In addition, unconventional material platforms could be explored. For example, scattering at high index spheres at extremely low frequencies was explored experimentally using voids filled with water, with water having a permittivity as high as 80 up to 10 GHz [256]. The system invites exploration because voids can be created in a stretchable rubber material, similar to balloons filled with water. The time variation of such systems might be feasible with some mechanical efforts. It would constitute an excellent implementation for some of the effects described in Section 7.3.

However, the biggest challenge remains to establish reliable material platforms that allow us to observe the described effects at optical frequencies. We need to push the range of frequencies where PTCs were demonstrated toward the visible or at least toward the infrared domain, where many applications that would benefit from a deterministic control over the spectral control of light exist. The first steps along these lines were done by all-optical fast tuning of material properties, but many open questions remain. For example, it is not just the switching from one material property to another that matters, which permits the observation of time reflection and time refraction. It is the periodic modulation between two states of the material on time scales comparable to the oscillation period of light that needs to be accomplished. After the switching, in most cases, thermal or electronic processes are responsible for driving the material back to its original state. These are slow processes that need to be accelerated. Therefore, we need to find ways to change material properties on optical time scales between two states that, hopefully, differ substantially. And even if the change in the material properties is only modest, we can exploit our understanding of how to enhance the light–matter interaction thanks to a suitable structured spatial environment (see Section 7.3), by choosing suitable frequency domains, or a combination thereof. ST-PCs could be one solution [69]. In addition, operating in the ENZ domain was already exploited [61]. But other schemes continue to be uncharted, for example, operating in an integrated photonic system close to the cutoff frequency of a guided mode [257] or exploiting propagating surface plasmon polaritons [258]. A part of these explorations should also concern the delineation from nonlinear optical effects that have already been discussed in the past [163]. While, in some specific situations, the nonlinear polarization can be written so that it appears to be a time-varying material property, equating both effects under all circumstances would be misleading. Therefore, elaborating on the differences and exploring the unique aspects would be crucial.

Furthermore, it is reasonable to anticipate additional discoveries of new light–matter interaction phenomena in PTCs. In just the past five years, a wealth of significant fundamental studies on PTCs has surfaced. These studies encompass a range of phenomena, including among others the amplification of spontaneous emission from excited atoms, subluminal Cherenkov radiation, superluminal momentum-gap solitons, and temporal Anderson localization. Much like traditional photonic crystals have played a pivotal role in quantum optics, PTCs hold the potential to achieve similar or even greater significance due to their capacity for extreme light–matter interactions $[259]$ . However, realizing this potential necessitates PTCs to operate within the optical domain close to electronic transitions in solids. But, certainly, realizing PTCs in the infrared part of the spectrum or terahertz frequencies would be also exciting and important. There, vibrational and rotational excitations in molecules and phonons in solids do play a role.

With time passing and progress being made concerning the different material platforms with which we can realize PTCs operating at different frequencies, we can ask further questions. For example, how to actually design and later realize PTCs so that they offer predefined optical functionalities on demand, the focus here should be on spectral control and field amplification. This is the vast field of inverse design, where major developments are witnessed in nearly all fields of science. Most notably, the notion of differential programming attracted increasing attention. It would allow us to use gradient-based optimization to design functional devices easily. Therefore, when setting up computational tools to describe PTCs, this aspect should be considered from the beginning in implementing computational routines. Of course, also techniques from the field of machine learning and artificial intelligence shall be exploited. However, the benefit of such techniques compared with traditional approaches needs to be demonstrated.

Moreover, emphasis shall be put on exploring possible novel effects of PTCs on theoretical, computational, and experimental grounds. A prime example could be in the context of synthetic dimensions $[200]$ . Synthetic dimensions in photonics involve creating additional degrees of freedom for light propagation beyond the conventional three spatial dimensions. PTCs offer unique possibilities by exploiting the frequency as a synthetic dimension, where the temporal periodicity of PTCs allows us to address it in a highly efficient manner, possibly also driven by some of the inverse design techniques just described. This can enable studies of high-dimensional topological phenomena and complex light–matter interactions. In addition, the temporal modulation in PTCs provides a means to dynamically control the properties of these synthetic dimensions so that topological properties can be adjusted. By exploiting the synthetic dimensions, PTCs can enhance light–matter interactions, enabling efficient photon–photon interactions and nonlinear processes that are meaningful for a future quantum information processing architecture. With that, PTCs can simulate higher-dimensional physical or quantum systems that are challenging to study in conventional settings, providing a platform for exploring new physics.

Finally, working toward more compact, ideally fully integrated PTCs would be fantastic. Currently, the experimental schemes are great at providing proof for the principles, but they are unlikely to constitute a base for future applications in, e.g., wireless communication systems. That domain could especially benefit from the possibility of transducing information among different frequencies and amplifying weak signals in unconventional manners. However, bulky experimental schemes are unlikely to be attractive, and fully integrated schemes would be necessary to make practical use of PTCs in the long run. However, before that level is reached, many more questions need to be answered, and we hope that some of the readers of this tutorial will be among those who deliver the answers.

## 9. CONCLUDING REMARKS

In this short conclusion, we wish to wrap up our tutorial on PTCs. We started by appreciating that a PTC consists of a spatially homogeneous material whose properties periodically change in time. While there is no clear delineation, it is assumed that the oscillation period of the property compares to the oscillation period of the probe light. Or at least, being comparable in the order of magnitude allows us to observe many of the effects usually attributed to PTCs. While permittivity is a common property to modulate in these materials, other material properties can also be varied over time to achieve similar phenomena. When implemented in different physical systems, it would be a parameter describing the system that appears in the governing equations, which is on equal footing as the permittivity for optical phenomena. Motivated by experiments, we gave examples such as transmission lines, where the capacitance of the respective LC circuit was time-varying. Then, the description of how voltage waves propagate along the transmission line is the same as that of a wave equation in a homogeneous medium, and the time-varying capacitance emerges instead of the time-varying permittivity.

To explore the fundamental properties of PTCs, we outlined two computational techniques that can also be used for analytical explorations. On the one hand, we showed how to find elementary solutions to Maxwell's equations in reciprocal space, both in space and time. On the other hand, an ABCD transfer-matrix technique can be used for similar purposes. Both techniques are beneficial for specific modulation profiles. The former is for sinusoidal modulations, while the latter is for modulations where the material property jumps between discrete values. Both methods make the same predictions, and it is instead a question of convenience which method to choose. The techniques can be used to explore key features of PTCs.

On the one hand, PTCs sustain momentum gaps for a sufficiently strong modulation. It implies that for a given frequency, no propagating wave exists for a specific range of momenta. Inside the momentum gaps, two inhomogeneous solutions exist to the wave equation. One is exponentially decaying in time, while the other is exponentially growing. Next, we have explored multiple aspects of PTCs that are relevant for realistic systems, leading also to a more complex description. Dispersion, the finiteness in space and time, an anisotropy, possibly nonlinearities, or deviations from the perfect time-harmonic modulation of the material properties were discussed.

In the following sections, we elaborated on the possibilities of implementing PTCs. First, we distinguished PTCs from other physical systems with similar effects, especially specific nonlinear processes. The different material platforms that were discussed provide access to PTCs in different spectral domains. Finally, we elaborated on the effects of light–matter interaction and possible applications of PTCs. We emphasized applications that exploit the critical properties of PTCs but also other aspects that look appealing.

Considering that many of the fundamental effects in PTCs were just explored, we expect many more applications to emerge shortly. However, we emphasize that the field of PTCs itself is relatively nascent, and many primary effects still await demonstration. But independent of these future achievements, we hope to have convinced the reader in the tutorial that the topic is fascinating and full of potential. The motivation to explore PTCs lies in their unique manipulation of light in time and, with that, also in the frequency domain, offering a novel playground for exploring fundamental physics and pushing the boundaries of optical technologies. Though in its infancy, this field holds the promise of revolutionary applications, ranging from advanced communication systems to ground-breaking quantum computing platforms.

The potential to control and manipulate light in intricate ways opens doors to uncharted territories in photonics. We envision that the continued research in PTCs will lead to the development of more efficient, faster, and compact photonic devices, which could transform the landscape of technology and industry. Furthermore, the interplay of PTCs with nonlinear optics and quantum phenomena presents a rich vein of research that could yield unprecedented insights into the nature of light–matter interactions.

In conclusion, the journey into the realm of PTCs is not just a pursuit of practical applications but a venture into the depths of scientific curiosity and innovation. It is a field where each discovery paves the way for new questions and deeper understanding, inviting researchers to continually push the frontiers of what is possible.

## FUNDING

Tekniikan Edistämissäätiö; Research Council of Finland (PREIN, decision number 346529, Aalto University); Research Council of Finland (356797); Helmholtz Association (Materials Systems Engineering); Bundesministerium für Bildung und Forschung; Carl-Zeiss-Stiftung (CZF-Focus@HEiKA Program); Deutsche Forschungsgemeinschaft (258734477-SFB 1173, EXC-2082/1-390761711).

## ACKNOWLEDGMENTS

The authors would like to thank Mr. Bahman Amrahi for the fruitful discussions about synthetic dimensions. The authors would like to thank all past and current co-workers in their groups that have contributed to the discussion and the understanding of photonic time crystals. In particular, we would like to thank Grigorii Ptitcyn, Theodosios D. Karamanos, Aristeidis Lamprianidis, Sergei Tretyakov, Shanhui Fan, and Mohamed Mostafa.

## DISCLOSURES

The authors declare no conflicts of interest.

## DATA AVAILABILITY

No data were generated or analyzed in the presented research.

## REFERENCES

1. J. D. Joannopoulos, S. G. Johnson, J. N. Winn, et al., Photonic Crystals: Molding the Flow of Light - Second Edition (Princeton University Press, 2008).

2. C. Simovski and S. Tretyakov, An Introduction to Metamaterials and Nanophotonics (Cambridge University Press, 2020).

3. K. Achouri and C. Caloz, Electromagnetic Metasurfaces: Theory and Applications (John Wiley & Sons, 2021).

4. M. Sanchez-Dominguez and C. Rodriguez-Abreu, Nanocolloids: A Meeting Point for Scientists and Technologists (Elsevier, 2016).

5. K. S. Novoselov, A. Mishchenko, A. Carvalho, et al., “2D materials and van der Waals heterostructures,” Science 353, aac9439 (2016).

6. L. Rayleigh, “XVII. On the maintenance of vibrations by forces of double frequency, and on the propagation of waves through a medium endowed with a periodic structure,” London Edinburgh Philos. Mag. J. Sci. 24, 145–159 (1887).

7. J. C. Knight, T. A. Birks, P. S. J. Russell, et al., “All-silica single-mode optical fiber with photonic crystal cladding,” Opt. Lett. 21, 1547–1549 (1996).

8. J. J. Wierer, A. David, and M. M. Megens, “III-nitride photonic-crystal light-emitting diodes with high extraction efficiency,” Nat. Photonics 3, 163–169 (2009).

9. W. Liu, H. Ma, and A. Walsh, “Advance in photonic crystal solar cells,” Renewable Sustainable Energy Rev. 116, 109436 (2019).

10. C. Fenzl, T. Hirsch, and O. S. Wolfbeis, “Photonic crystals for chemical sensing and biosensing,” Angew. Chem. Int. Ed. 53, 3318–3335 (2014).

11. J. Münzberg, A. Vetter, F. Beutel, et al., “Superconducting nanowire single-photon detector implemented in a 2D photonic crystal cavity,” Optica 5, 658–665 (2018).

12. A. Bielawny, C. Rockstuhl, F. Lederer, et al., “Intermediate reflectors for enhanced top cell performance in photovoltaic thin-film tandem cells,” Opt. Express 17, 8439–8446 (2009).

13. J. A. Richards, Analysis of Periodically Time-Varying Systems (Springer Science & Business Media, 2012).

14. D. K. Kalluri, Electromagnetics of Time Varying Complex Media: Frequency and Polarization Transformer (CRC Press, 2018).

15. C. Caloz and Z.-L. Deck-Léger, “Spacetime metamaterials—part I: general concepts,” IEEE Trans. Antennas Propag. 68, 1569–1582 (2019).

16. C. Caloz and Z.-L. Deck-Léger, “Spacetime metamaterials—part II: theory and applications,” IEEE Trans. Antennas Propag. 68, 1583–1598 (2020).

17. E. Galiffi, R. Tirole, S. Yin, et al., “Photonics of time-varying media,” Adv. Photonics 4, 014002 (2022).

18. N. Engheta, “Four-dimensional optics using time-varying metamaterials,” Science 379, 1190–1191 (2023).

19. Q. He, S. Sun, and L. Zhou, “Tunable/reconfigurable metasurfaces: physics and applications,” Research 2019, 1849272 (2019).

20. J. R. Zurita-Sánchez, P. Halevi, and J. C. Cervantes-González, “Reflection and transmission of a wave incident on a slab with a time-periodic dielectric function $\varepsilon(t)$ ,” Phys. Rev. A 79, 053821 (2009).

21. E. Lustig, Y. Sharabi, and M. Segev, “Topological aspects of photonic time crystals,” Optica 5, 1390–1395 (2018).

22. F. Wilczek, “Quantum time crystals,” Phys. Rev. Lett. 109, 160401 (2012).

23. F. Biancalana, A. Amann, A. V. Uskov, et al., “Dynamics of light propagation in spatiotemporal dielectric structures,” Phys. Rev. E 75, 046607 (2007).

24. F. Morgenthaler, “Velocity modulation of electromagnetic waves,” IEEE Trans. Microwave Theory Tech. 6, 167–172 (1958).

25. A. L. Cullen, “A travelling-wave parametric amplifier,” Nature 181, 332 (1958).

26. P. K. Tien and H. Suhl, “A traveling-wave ferromagnetic amplifier,” Proc. IRE 46, 700–706 (1958).

27. J.-C. Simon, “Action of a progressive disturbance on a guided electromagnetic wave,” IEEE Trans. Microwave Theory Tech. 8, 18–29 (1960).

28. S. I. Averkov and N. S. Stepanov, “Wave propagation in systems with a traveling parameter,” Izv. Vyssh. Uchebn. Zaved., Radiofiz 2, 203–212 (1959).

29. A. Oliner and A. Hessel, “Wave propagation in a medium with a progressive sinusoidal disturbance,” IEEE Trans. Microwave Theory Tech. 9, 337–343 (1961).

30. L. A. Ostrovskii and N. S. Stepanov, “Nonresonance parametric phenomena in distributed systems,” Radiophys. Quantum Electron. 14, 387–419 (1971).

31. J. A. Armstrong, N. Bloembergen, J. Ducuing, et al., “Interactions between light waves in a nonlinear dielectric,” Phys. Rev. 127, 1918–1939 (1962).

32. R. H. Kingston, “Parametric amplification and oscillation at optical frequencies,” Proc. Inst. Rádió Eng. 50, 472 (1962).

33. N. M. Kroll, “Parametric amplification in spatially extended media and application to the design of tuneable oscillators at optical frequencies,” Phys. Rev. 127, 1207–1211 (1962).

34. S. A. Akhmanov and R. V. Khokhlov, “Concerning one possibility of amplification of light waves,” Zh. Eksp. Teor. Fiz. 43, 351–353 (1962).

35. D. Holberg and K. Kunz, “Parametric properties of fields in a slab of time-varying permittivity,” IEEE Trans. Antennas Propag. 14, 183–194 (1966).

36. P. A. Sturrock, “Kinematics of growing waves,” Phys. Rev. 112, 1488–1503 (1958).

37. E. Cassedy, “Temporal instabilities in traveling-wave parametric amplifiers (correspondence),” IEEE Trans. Microwave Theory Techn. 10, 86–87 (1962).

38. E. Cassedy and A. Oliner, “Dispersion relations in time-space periodic media: part I-stable interactions,” Proc. IEEE 51, 1342–1359 (1963).

39. E. Cassedy, “Dispersion relations in time-space periodic media part II–unstable interactions,” Proc. IEEE 55, 1154–1168 (1967).

40. L. Felsen and G. Whitman, "Wave propagation in time-varying media," IEEE Trans. Antennas Propag. 18, 242-253 (1970).

41. E. Cassedy, “Waves guided by a boundary with time-space periodic modulation,” Proc. Inst. Electr. Eng. 112, 269–279 (1965).

42. R. Fante, “Transmission of electromagnetic waves into time-varying media,” IEEE Trans. Antennas Propag. 19, 417–424 (1971).

43. F. Harfoush and A. Taflove, “Scattering of electromagnetic waves by a material half-space with a time-varying conductivity,” IEEE Trans. Antennas Propag. 39, 898–906 (1991).

44. E. Yablonovitch, “Self-phase modulation of light in a laser-breakdown plasma,” Phys. Rev. Lett. 32, 1101–1104 (1974).

45. J. S. Martínez-Romero, O. Becerra-Fuentes, and P. Halevi, “Temporal photonic crystals with modulations of both permittivity and permeability,” Phys. Rev. A 93, 063813 (2016).

46. J. S. Martínez-Romero and P. Halevi, “Parametric resonances in a temporal photonic crystal slab,” Phys. Rev. A 98, 053852 (2018).

47. J. R. Reyes-Ayona and P. Halevi, “Observation of genuine wave vector (k or $\beta$ ) gap in a dynamic transmission line and temporal photonic crystals,” Appl. Phys. Lett. 107, 074101 (2015).

48. J. R. Reyes-Ayona and P. Halevi, “Electromagnetic wave propagation in an externally modulated low-pass transmission line,” IEEE Trans. Microwave Theory Tech. 64, 3449–3459 (2016).

49. V. P. Bykov, “Spontaneous emission in a periodic structure,” Soviet Journal of Experimental and Theoretical Physics 35, 269–273 (1972).

50. V. P. Bykov, “Spontaneous emission from a medium with a band spectrum,” Sov. J. Quantum Electron. 4, 861–871 (1975).

51. E. Yablonovitch, “Inhibited spontaneous emission in solid-state physics and electronics,” Phys. Rev. Lett. 58, 2059–2062 (1987).

52. S. John, “Strong localization of photons in certain disordered dielectric superlattices,” Phys. Rev. Lett. 58, 2486–2489 (1987).

53. M. Lyubarov, Y. Lumer, A. Dikopoltsev, et al., “Amplified emission and lasing in photonic time crystals,” Science 377, 425–428 (2022).

54. A. Dikopoltsev, Y. Sharabi, M. Lyubarov, et al., “Light emission by free electrons in photonic time-crystals,” Proc. Natl. Acad. Sci. 119, e2119705119 (2022).

55. X. Gao, X. Zhao, X. Ma, et al., “Free electron emission in vacuum assisted by photonic time crystals,” J. Phys. D: Appl. Phys. 57, 315112 (2024).

56. Y. Pan, M.-I. Cohen, and M. Segev, “Superluminal k-gap solitons in nonlinear photonic time crystals,” Phys. Rev. Lett. 130, 233801 (2023).

57. R. Carminati, H. Chen, R. Pierrat, et al., “Universal statistics of waves in a random time-varying medium,” Phys. Rev. Lett. 127, 094101 (2021).

58. Y. Sharabi, E. Lustig, and M. Segev, “Disordered photonic time crystals,” Phys. Rev. Lett. 126, 163902 (2021).

59. B. Apffel, S. Wildeman, A. Eddi, et al., “Time localization of energy in disordered time-modulated systems,” Phys. Rev. Lett. 128, 094503 (2022).

60. S. Saha, O. Segal, C. Fruhling, et al., “Photonic time crystals: a materials perspective,” Opt. Express 31, 8267–8273 (2023).

61. Z. Hayran, J. B. Khurgin, and F. Monticone, “ $\hbar\omega$ versus $\hbar k$ : dispersion and energy constraints on time-varying photonic materials and time crystals,” Opt. Mater. Express 12, 3904–3917 (2022).

62. M. Z. Alam, I. De Leon, and R. W. Boyd, “Large optical nonlinearity of indium tin oxide in its epsilon-near-zero region,” Science 352, 795–797 (2016).

63. J. Bohn, T. S. Luk, S. Horsley, et al., “Spatiotemporal refraction of light in an epsilon-near-zero indium tin oxide layer: frequency shifting effects arising from interfaces,” Optica 8, 1532–1537 (2021).

64. Y. Zhou, M. Z. Alam, M. Karimi, et al., “Broadband frequency translation through time refraction in an epsilon-near-zero material,” Nat. Commun. 11, 2180 (2020).

65. L. Caspani, R. Kaipurath, M. Clerici, et al., “Enhanced nonlinear refractive index in $\varepsilon$ -near-zero materials,” Phys. Rev. Lett. 116, 233901 (2016).

66. E. Lustig, O. Segal, S. Saha, et al., “Time-refraction optics with single cycle modulation,” Nanophotonics 12, 2221–2230 (2023).

67. R. Tirole, S. Vezzoli, D. Saxena, et al., “Second harmonic generation at a time-varying interface,” Nature Commun. 15, 7752 (2024).

68. J. B. Khurgin, M. Clerici, and N. Kinsey, “Fast and slow nonlinearities in epsilon-near-zero materials,” Laser Photonics Rev. 15, 2000291 (2021).

69. X. Wang, P. Garg, M. Mirmoosa, et al., “Unleashing infinite momentum bandgap using resonant material systems,” Nature Photon. (2024), https://doi.org/10.1038/s41566-024-01563-3.

70. E. E. Narimanov, “Ultrafast optical modulation by virtual interband transitions,” arXiv (2023).

71. J. Dong, S. Zhang, H. He, et al., “Non-uniform wave momentum bandgap in biaxial anisotropic photonic time crystals,” arXiv (2024).

72. S. Zhang, J. Dong, H. Li, et al., “Longitudinal optical phonons in photonic time crystals containing a stationary charge,” Phys. Rev. B 110, L100306 (2024).

73. X. Wang, “Photonic-time-crystal band structure calculation based on the plane wave expansion method,” figshare (2024), https://doi.org/10.6084/m9.figshare.26407996.

74. X. Wang, “Photonic-time-crystal band structure calculation based on the transfer matrix method,” figshare (2024), https://doi.org/10.6084/m9.figshare.26407999.

75. A. B. Shvartsburg, “Optics of nonstationary media,” Phys.-Usp. 48, 797–823 (2005).

76. L. Yuan, A. Dutt, and S. Fan, “Synthetic frequency dimensions in dynamically modulated ring resonators,” APL Photonics 6, 071102 (2021).

77. Z. Hayran and F. Monticone, “Using time-varying systems to challenge fundamental limitations in electromagnetics: overview and summary of applications,” IEEE Antennas Propag. Mag. 65, 29–38 (2023).

78. G. Ptitcyn, M. S. Mirmoosa, A. Sotoodehfar, et al., “A tutorial on the basics of time-varying electromagnetic systems and circuits: historic overview and basic concepts of time-modulation,” IEEE Antennas Propag. Mag. 65, 10–20 (2023).

79. A. Ortega-Gomez, M. Lobet, J. E. Vázquez-Lozano, et al., “Tutorial on the conservation of momentum in photonic time-varying media,” Opt. Mater. Express 13, 1598–1608 (2023).

80. S. Yin, E. Galiffi, G. Xu, et al., “Scattering at temporal interfaces: an overview from an antennas and propagation engineering perspective,” IEEE Antennas Propag. Mag. 65, 21–28 (2023).

81. E. Lustig, O. Segal, S. Saha, et al., “Photonic time-crystals-fundamental concepts,” Opt. Express 31, 9165–9170 (2023).

82. R. Won, “It’s a matter of time,” Nat. Photonics 17, 209–210 (2023).

83. A. Boltasseva, V. M. Shalaev, and M. Segev, “Photonic time crystals: from fundamental insights to novel applications: opinion,” Opt. Mater. Express 14, 592–597 (2024).

84. M. Lobet, N. Kinsey, I. Liberal, et al., “New horizons in near-zero refractive index photonics and hyperbolic metamaterials,” ACS Photonics 10, 3805–3820 (2023).

85. A. M. Shaltout, V. M. Shalaev, and M. L. Brongersma, “Spatiotemporal light control with active metasurfaces,” Science 364, eaat3100 (2019).

86. S. Taravati and G. V. Eleftheriades, “Microwave space–time-modulated metasurfaces,” ACS Photonics 9, 305–318 (2022).

87. N. Engheta, “Metamaterials with high degrees of freedom: space, time, and more,” Nanophotonics 10, 639–642 (2021).

88. D. M. Pozar, Microwave Engineering (John Wiley & Sons, 2012).

89. D. K. Cheng, Field and Wave Electromagnetics (Addison Wesley, 1983).

90. J. S. Blakemore, Solid State Physics (Cambridge University Press, 1985).

91. Z. Ozer, A. M. Mamedov, and E. Ozbay, “BaTiO $_{3}$ based photonic time crystal and momentum stop band,” Ferroelectrics 557, 105–111 (2020).

92. M. Mirmoosa, T. Koutserimpas, G. Ptitcyn, et al., “Dipole polarizability of time-varying particles,” New J. Phys. 24, 063004 (2022).

93. J. D. Jackson, Classical Electrodynamics (Wiley, 1999).

94. T. T. Koutserimpas and F. Monticone, “Time-varying media, dispersion, and the principle of causality,” Opt. Mater. Express 14, 1222–1236 (2024).

95. L. A. Zadeh, “Frequency analysis of variable networks,” Proc. IRE 38, 291–299 (1950).

96. A. Figotin and I. Vitebsky, “Nonreciprocal magnetic photonic crystals,” Phys. Rev. E 63, 066609 (2001).

97. Z. Yu, Z. Wang, and S. Fan, “One-way total reflection with one-dimensional magneto-optical photonic crystals,” Appl. Phys. Lett. 90, 121133 (2007).

98. J. S. Martínez-Romero and P. Halevi, “Standing waves with infinite group velocity in a temporally periodic medium,” Phys. Rev. A 96, 063831 (2017).

99. J. G. Gaxiola-Luna and P. Halevi, “Temporal photonic (time) crystal with a square profile of both permittivity $\varepsilon$ (t) and permeability $\mu$ (t),” Phys. Rev. B 103, 144306 (2021).

100. M. Chegnizadeh, K. Mehrany, and M. Memarian, “General solution to wave propagation in media undergoing arbitrary transient or periodic temporal variations of permittivity,” J. Opt. Soc. Am. B 35, 2923–2932 (2018).

101. P. Yeh, A. Yariv, and C.-S. Hong, “Electromagnetic propagation in periodic stratified media. I. general theory,” J. Opt. Soc. Am. 67, 423–438 (1977).

102. D. Ramaccia, A. Alù, A. Toscano, et al., “Temporal multilayer structures for designing higher-order transfer functions using time-varying metamaterials,” Appl. Phys. Lett. 118, 101901 (2021).

103. S. Sadhukhan and S. Ghosh, “Defect in photonic time crystals,” Phys. Rev. A 108, 023511 (2023).

104. A. M. Shaltout, J. Fang, A. V. Kildishev, et al., “Photonic time-crystals and momentum band-gaps,” in CLEO: QELS\_Fundamental Science (Optica Publishing Group, 2016), pp. FM1D–4.

105. T. T. Koutserimpas and R. Fleury, “Electromagnetic waves in a time periodic medium with step-varying refractive index,” IEEE Trans. Antennas Propag. 66, 5300–5307 (2018).

106. D. J. Griffiths, Introduction to Electrodynamics (Cambridge University Press, 2017).

107. Y. Sivan and J. B. Pendry, “Time reversal in dynamically tuned zero-gap periodic systems,” Phys. Rev. Lett. 106, 193902 (2011).

108. Y. Xiao, D. N. Maywar, and G. P. Agrawal, “Reflection and transmission of electromagnetic waves at a temporal boundary,” Opt. Lett. 39, 574–577 (2014).

109. X. Wang, M. S. Mirmoosa, and S. A. Tretyakov, “Controlling surface waves with temporal discontinuities of metasurfaces,” Nanophotonics 12, 2813–2822 (2023).

110. A. Shaltout, M. Clerici, N. Kinsey, et al., “Doppler-shift emulation using highly time-refracting TCO layer,” in 2016 Conference on Lasers and Electro-Optics (CLEO) (IEEE, 2016), pp. 1–2.

111. F. Deng, F. Zhu, X. Zhou, et al., “Frequency-selective terahertz wave amplification by a time-boundary-engineered Huygens metasurface,” arXiv (2024).

112. L. Bar-Hillel, A. Dikopoltsev, A. Kam, et al., “Time-refraction and time-reflection above critical angle for total internal reflection,” Phys. Rev. Lett. 132, 263802 (2024).

113. T. R. Jones, A. V. Kildishev, M. Segev, et al., “Time-reflection of microwaves by a fast optically-controlled time-boundary,” Nature Commun. 15, 6786 (2024).

114. H. Moussa, G. Xu, S. Yin, et al., “Observation of temporal reflection and broadband frequency translation at photonic time interfaces,” Nat. Phys. 19, 863–868 (2023).

115. E. Galiffi, G. Xu, S. Yin, et al., “Broadband coherent wave control through photonic collisions at time interfaces,” Nat. Phys. 19, 1703–1708 (2023).

116. D. Peng, Y. Fan, R. Liu, et al., “Time-reversed water waves generated from an instantaneous time mirror,” J. Phys. Commun. 4, 105013 (2020).

117. Z. Dong, H. Li, T. Wan, et al., “Quantum time reflection and refraction of ultracold atoms,” Nat. Photonics 18, 68–73 (2024).

118. O. Y. Long, K. Wang, A. Dutt, et al., “Time reflection and refraction in synthetic frequency dimension,” Phys. Rev. Res. 5, L012046 (2023).

119. E. Lustig, Y. Sharabi, and M. Segev, “Topology of photonic time-crystals,” in 2018 Conference on Lasers and Electro-Optics (CLEO) (Optica Publishing Group, 2018), paper FM3Q.3.

120. M. Salem and C. Caloz, “Temporal photonic crystals: causality versus periodicity,” in 2015 International Conference on Electromagnetics in Advanced Applications (ICEAA) (IEEE, 2015), pp. 490–493.

121. J. Gaxiola-Luna and P. Halevi, “Growing fields in a temporal photonic (time) crystal with a square profile of the permittivity $\varepsilon(t)$ ,” Appl. Phys. Lett. 122, 011702 (2023).

122. Z. Hayran and F. Monticone, “Beyond the Rozanov bound on electromagnetic absorption via periodic temporal modulations,” Phys. Rev. Appl. 21, 044007 (2023).

123. D. Barton, M. Lawrence, and J. Dionne, “Wavefront shaping and modulation with resonant electro-optic phase gradient metasurfaces,” Appl. Phys. Lett. 118, 071104 (2021).

124. X. Guo, Y. Ding, Y. Duan, et al., “Nonreciprocal metasurface with space–time phase modulation,” Light: Sci. Appl. 8, 123 (2019).

125. I. A. Williamson, M. Minkov, A. Dutt, et al., “Integrated nonreciprocal photonic devices with dynamic modulation,” Proc. IEEE 108, 1759–1784 (2020).

126. N. Kinsey, C. DeVault, J. Kim, et al., “Epsilon-near-zero Al-doped ZnO for ultrafast switching at telecom wavelengths,” Optica 2, 616–622 (2015).

127. G. Ptitcyn, M. S. Mirmoosa, and S. A. Tretyakov, “Time-modulated meta-atoms,” Phys. Rev. Res. 1, 023014 (2019).

128. M. S. Mirmoosa, G. A. Ptitcyn, R. Fleury, et al., “Instantaneous radiation from time-varying electric and magnetic dipoles,” Phys. Rev. A 102, 013503 (2020).

129. D. M. Solís and N. Engheta, “Functional analysis of the polarization response in linear time-varying media: a generalization of the Kramers–Kronig relations,” Phys. Rev. B 103, 144303 (2021).

130. J. Sloan, N. Rivera, and M. Soljačić, “Dispersion in photonic time crystals,” in CLEO: QELS Fundamental Science (Optica Publishing Group, 2020), pp. FTh4A–4.

131. J. Sloan, N. Rivera, J. D. Joannopoulos, et al., “Optical properties of dispersive time-dependent materials,” ACS Photonics 11, 950–962 (2024).

132. A. Sotoodehfar, M. S. Mirmoosa, and S. A. Tretyakov, “Waves in linear time-varying dielectric media,” in 2022 16th European Conference on Antennas and Propagation (EuCAP) (IEEE, 2022), pp. 1–5.

133. S. Horsley, E. Galiffi, and Y.-T. Wang, “Eigenpulses of dispersive time-varying media,” Phys. Rev. Lett. 130, 203803 (2023).

134. G. Ptitcyn, A. Lamprianidis, T. Karamanos, et al., “Floquet–Mie theory for time-varying dispersive spheres,” Laser Photonics Rev. 17, 2100683 (2023).

135. D. M. Solís, R. Kastner, and N. Engheta, “Time-varying materials in the presence of dispersion: plane-wave propagation in a Lorentzian medium with temporal discontinuity,” Photonics Res. 9, 1842–1853 (2021).

136. M. S. Mirmoosa, M. S. M. Mollaei, G. A. Ptitcyn, et al., “Time-varying plasmonic particles,” in 2021 Fifteenth International Congress on Artificial Materials for Novel Wave Phenomena (Metamaterials) (IEEE, 2021), pp. 272–274.

137. X. Ye, Y. Wang, J. Yao, et al., “Floquet modeling of surface-wave amplification in two-dimensional photonic time crystals,” Phys. Rev. B 109, 165304 (2024).

138. F. Feng, N. Wang, and G. P. Wang, “Temporal transfer matrix method for Lorentzian dispersive time-varying media,” Appl. Phys. Lett. 124, 101701 (2024).

139. J. a. C. Serra and M. G. Silveirinha, “Homogenization of dispersive space–time crystals: anomalous dispersion and negative stored energy,” Phys. Rev. B 108, 035119 (2023).

140. V. Asadchy, A. Lamprianidis, G. Ptitcyn, et al., “Parametric Mie resonances and directional amplification in time-modulated scatterers,” Phys. Rev. Appl. 18, 054065 (2022).

141. T. T. Koutserimpas, “Parametric amplification interactions in time-periodic media: coupled waves theory,” J. Opt. Soc. Am. B 39, 481–489 (2022).

142. J. R. Zurita-Sánchez and P. Halevi, “Resonances in the optical response of a slab with time-periodic dielectric function $\varepsilon(t)$ ,” Phys. Rev. A 81, 053834 (2010).

143. J. Valdez-García and P. Halevi, “Parametric resonances in a photonic time crystal with periodic square modulation of its permittivity $\varepsilon(t)$ ,” Phys. Rev. A 109, 063517 (2024).

144. D. Globosits, J. Hüpfl, and S. Rotter, “A photonic Floquet scattering matrix for wavefront-shaping in time-periodic media,” Phys. Rev. A 110, 053515 (2024).

145. X. Wang, M. S. Mirmoosa, V. S. Asadchy, et al., “Metasurface-based realization of photonic time crystals,” Sci. Adv. 9, eadg7541 (2023).

146. M. Salehi, M. Memarian, and K. Mehrany, “Parametric amplification and instability in time-periodic dielectric slabs,” Opt. Express 31, 2911–2930 (2023).

147. P. Garg, A. G. Lamprianidis, S. Rahman, et al., “Two-step homogenization of spatiotemporal metasurfaces using an eigenmode-based approach,” Opt. Mater. Express 14, 549–563 (2024).

148. J. Echave-Sustaeta, F. J. García-Vidal, and P. A. Huidobro, “Photon squeezing in photonic time crystals,” arXiv (2024).

149. S. F. Koufidis, T. T. Koutserimpas, and M. W. McCall, “Temporal analog of Bragg gratings,” Opt. Lett. 48, 4500–4503 (2023).

150. J.-C. Diels and W. Rudolph, Ultrashort Laser Pulse Phenomena Fundamentals, Techniques, and Applications on a Femtosecond Time Scale (Academic Press, 1996).

151. H. Li, S. Yin, H. He, et al., “Stationary charge radiation in anisotropic photonic time crystals,” Phys. Rev. Lett. 130, 093803 (2023).

152. G. Ptitcyn and N. Engheta, “Temporal twistronics,” arXiv (2024).

153. S. Noda, A. Chutinan, and M. Imada, “Trapping and emission of photons by a single defect in a photonic bandgap structure,” Nature 407, 608–610 (2000).

154. M. Bayindir, B. Temelkuran, and E. Ozbay, “Photonic-crystal-based beam splitters,” Appl. Phys. Lett. 77, 3902–3904 (2000).

155. A. Aly, B. Mohamed, M. Al-Dossari, et al., “Ultra-sensitive pressure sensing capabilities of defective one-dimensional photonic crystal,” Sci. Rep. 13, 18876 (2023).

156. J. Kim, D. Lee, S. Yu, et al., “Unidirectional scattering with spatial homogeneity using correlated photonic time disorder,” Nat. Phys. 19, 726–732 (2023).

157. R. W. Boyd, Nonlinear Optics (Academic Press, Inc., 2008).

158. E. I. Kiselev and Y. Pan, “Symmetry breaking and spatiotemporal pattern formation in photonic time crystals,” arXiv (2024).

159. G. F. FitzGerald, “On the driving of electromagnetic vibrations by electromagnetic and electrostatic engines,” The Scientific Writings of the Late George Francis FitzGerald, J. Lamor, pp. 277–281 (Longmans, Green, and Co., 1902).

160. H. A. Haus, Electromagnetic Noise and Quantum Optical Measurements (Springer Science & Business Media, 2012).

161. B. Yurke, L. Corruccini, P. Kaminsky, et al., “Observation of parametric amplification and deamplification in a Josephson parametric amplifier,” Phys. Rev. A 39, 2519–2533 (1989).

162. Y. Yamamoto, Fundamentals of Noise Processes (Cambridge University Press, 2004).

163. J. B. Khurgin, “Photonic time crystals and parametric amplification: similarity and distinction,” ACS Photonics 11, 2150–2159 (2024).

164. Y. J. Ding, S. J. Lee, and J. B. Khurgin, “Transversely pumped counterpropagating optical parametric oscillation and amplification,” Phys. Rev. Lett. 75, 429–432 (1995).

165. L. Lanco, S. Ducci, J.-P. Likforman, et al., “Semiconductor waveguide source of counterpropagating twin photons,” Phys. Rev. Lett. 97, 173901 (2006).

166. H. Watanabe and M. Oshikawa, “Absence of quantum time crystals,” Phys. Rev. Lett. 114, 251603 (2015).

167. N. Y. Yao, A. C. Potter, I.-D. Potirniche, et al., “Discrete time crystals: rigidity, criticality, and realizations,” Phys. Rev. Lett. 118, 030401 (2017).

168. S. Choi, J. Choi, R. Landig, et al., “Observation of discrete time-crystalline order in a disordered dipolar many-body system,” Nature 543, 221–225 (2016).

169. A. Greilich, N. Kopteva, A. Kamenskii, et al., “Robust continuous time crystal in an electron–nuclear spin system,” Nat. Phys. 20, 631 (2024).

170. T. Liu, J.-Y. Ou, K. F. MacDonald, et al., “Photonic metamaterial analogue of a continuous time crystal,” Nat. Phys. 19, 986–991 (2023).

171. N. Träger, P. Gruszecki, F. Lisiecki, et al., “Real-space observation of magnon interaction with driven space–time crystals,” Phys. Rev. Lett. 126, 057201 (2021).

172. H. Kazemi, M. Y. Nada, T. Mealy, et al., “Exceptional points of degeneracy induced by linear time-periodic variation,” Phys. Rev. Appl. 11, 014007 (2019).

173. B. Borchers, C. Brée, S. Birkholz, et al., “Saturation of the all-optical Kerr effect in solids,” Opt. Lett. 37, 1541–1543 (2012).

174. A. Bartels, D. Heinecke, and S. A. Diddams, “10-GHz self-referenced optical frequency comb,” Science 326, 681 (2009).

175. D. Fomra, A. Ball, S. Saha, et al., “Nonlinear optics at epsilon near zero: from origins to new materials,” Appl. Phys. Rev. 11, 011317 (2024).

176. W. Jaffray, S. Stengel, F. Biancalana, et al., “Spatio-spectral optical fission in time-varying subwavelength layers,” arXiv, (2024).

177. M. Clerici, N. Kinsey, C. DeVault, et al., “Controlling hybrid nonlinearities in transparent conducting oxides via two-colour excitation,” Nat. Commun. 8, 15829 (2017).

178. S. Saha, B. T. Diroll, J. Shank, et al., “Broadband, high-speed, and large-amplitude dynamic optical switching with yttrium-doped cadmium oxide,” Adv. Funct. Mater. 30, 1908377 (2020).

179. M. Li, S. Biswas, C. U. Hail, et al., “Refractive index modulation in monolayer molybdenum diselenide,” Nano Lett. 21, 7602–7608 (2021).

180. R. Secondo, J. Khurgin, and N. Kinsey, “Absorptive loss and band non-parabolicity as a physical origin of large nonlinearity in epsilon-near-zero materials,” Opt. Mater. Express 10, 1545–1560 (2020).

181. V. Bruno, S. Vezzoli, C. DeVault, et al., “Broad frequency shift of parametric processes in epsilon-near-zero time-varying media,” Appl. Sci. 10, 1318 (2020).

182. I.-W. Un, S. Sarkar, and Y. Sivan, “Electronic-based model of the optical nonlinearity of low-electron-density Drude materials,” Phys. Rev. Appl. 19, 044043 (2023).

183. J. B. Pendry, “An avalanche model for femtosecond optical response,” arXiv (2024).

184. R. Tirole, S. Vezzoli, E. Galiffi, et al., “Double-slit time diffraction at optical frequencies,” Nat. Phys. 19, 999–1002 (2023).

185. Y. Shen, “Basic considerations of four-wave mixing and dynamic gratings,” IEEE J. Quantum Electron. 22, 1196–1203 (1986).

186. H. Eichler, “Laser-induced grating phenomena,” Opt. Acta 24, 631–642 (1977).

187. M. H. Michael, S. Haque, L. Windgaetter, et al., “Photonic time-crystalline behaviour mediated by phonon squeezing in $Ta_{2}NiSe_{5}$ ,” Nat. Commun. 15, 3638 (2024).

188. H. He, S. Zhang, J. Qi, et al., “Faraday rotation in nonreciprocal photonic time-crystals,” Appl. Phys. Lett. 122, 051703 (2023).

189. J. Wilson, F. Santosa, M. Min, et al., “Temporal control of graphene plasmons,” Phys. Rev. B 98, 081411 (2018).

190. E. I. Kiselev and Y. Pan, “Light controlled THz plasmonic time varying media: momentum gaps, entangled plasmon pairs, and pulse induced time reversal,” arXiv (2023).

191. A. Shirokova, A. Maslov, and M. Bakunov, “Surface plasmon transformation on dynamic graphene with a periodic modulation of carrier density,” Phys. Rev. B 108, 245139 (2023).

192. K.-H. Kim and O. Kang-Hyok, “Graphene plasmonic time crystals,” Phys. Status Solidi RRL 11, 2400116 (2024).

193. X. Dong, Y. Ye, B. Wang, et al., “Experimental demonstration of the acoustic frequency conversions by temporal phononic crystals,” arXiv (2013).

194. K. Ward, S. Matsumoto, and R. Narayanan, “The electrostatically forced Faraday instability: theory and experiments,” J. Fluid Mech. 862, 696–731 (2019).

195. C. Chong, B. Kim, E. Wallace, et al., “Modulation instability and wavenumber bandgap breathers in a time layered phononic lattice,” Phys. Rev. Res. 6, 023045 (2024).

196. B. L. Kim, C. Chong, S. Hajarolasvadi, et al., “Dynamics of time-modulated, nonlinear phononic lattices,” Phys. Rev. E 107, 034211 (2023).

197. S. Flach and A. V. Gorbach, “Discrete breathers—advances in theory and applications,” Phys. Rep. 467, 1–116 (2008).

198. G. Trainiti, Y. Xia, J. Marconi, et al., “Time-periodic stiffness modulation in elastic metamaterials for selective wave filtering: theory and experiment,” Phys. Rev. Lett. 122, 124301 (2019).

199. Y. Xia, E. Riva, M. I. Rosa, et al., “Experimental observation of temporal pumping in electromechanical waveguides,” Phys. Rev. Lett. 126, 095501 (2021).

200. L. Yuan, Q. Lin, M. Xiao, et al., “Synthetic dimension in photonics,” Optica 5, 1396–1405 (2018).

201. A. Dutt, Q. Lin, L. Yuan, et al., “A single photonic cavity with two independent physical synthetic dimensions,” Science 367, 59–64 (2020).

202. T. Ozawa and H. M. Price, “Topological quantum matter in synthetic dimensions,” Nat. Rev. Phys. 1, 349–357 (2019).

203. E. Rusak, J. Straubel, P. Gladysz, et al., “Enhancement of and interference among higher order multipole transitions in molecules near a plasmonic nanoantenna,” Nat. Commun. 10, 5775 (2019).

204. D. M. Callahan, J. N. Munday, and H. A. Atwater, “Solar cell light trapping beyond the ray optic limit,” Nano Lett. 12, 214–218 (2012).

205. M. G. Abebe, A. Abass, G. Gomard, et al., “Rigorous wave-optical treatment of photon recycling in thermodynamics of photovoltaics: perovskite thin-film solar cells,” Phys. Rev. B 98, 075141 (2018).

206. J. Park, H. C. Park, K. Lee, et al., “Comment on ‘Amplified emission and lasing in photonic time crystals’,” arXiv (2022).

207. J. Park, K. Lee, R.-Y. Zhang, et al., “Spontaneous emission decay and excitation in photonic temporal crystals,” arXiv (2024).

208. K. Xu, M. Fang, J. Feng, et al., “Thresholdless laser based on photonic time crystals,” Research Square, (2024).

209. A. Dikopoltsev, Y. Sharabi, S. Tsesses, et al., “Free-electrons radiation in a photonic time crystal,” in 2020 Conference on Lasers and Electro-Optics (CLEO) (IEEE, 2020), pp. 1–2.

210. J.-F. Zhu, A. Nussupbekov, W. Zhou, et al., “Smith-Purcell radiation from time grating,” arXiv, (2023).

211. Z. Hayran and F. Monticone, “Controlling the spectral flow of light in non-Hermitian photonic time crystals,” in 2021 Fifteenth International Congress on Artificial Materials for Novel Wave Phenomena (Metamaterials) (IEEE, 2021), pp. 153–155.

212. T. T. Koutserimpas and R. Fleury, “Nonreciprocal gain in non-Hermitian time-Floquet systems,” Phys. Rev. Lett. 120, 087401 (2018).

213. E. Li, B. J. Eggleton, K. Fang, et al., “Photonic Aharonov–Bohm effect in photon–phonon interactions,” Nat. Commun. 5, 3225 (2014).

214. K. N. Rozanov, “Ultimate thickness to bandwidth ratio of radar absorbers,” IEEE Trans. Antennas Propag. 48, 1230–1234 (2000).

215. H. Li and A. Alù, “Temporal switching to extend the bandwidth of thin absorbers,” Optica 8, 24–29 (2021).

216. C. Firestein, A. Shlivinski, and Y. Hadad, “Absorption and scattering by a temporally switched lossy layer: going beyond the Rozanov bound,” Phys. Rev. Appl. 17, 014017 (2022).

217. X. Yang, E. Wen, and D. F. Sievenpiper, “Broadband time-modulated absorber beyond the Bode-Fano limit for short pulses by energy trapping,” Phys. Rev. Appl. 17, 044003 (2022).

218. R.-y. Dong, S. Wang, J.-H. Zou, et al., “Tunable and controllable multi-channel time-comb absorber based on continuous photonic time crystals,” Opt. Lett. 48, 2627–2630 (2023).

219. Z. Manzoor and S. Taravati, “Enhanced resolution imaging by aperiodically perturbed photonic time crystals,” in 2020 Conference on Lasers and Electro-Optics (CLEO) (IEEE, 2020), pp. 1–2.

220. J. B. Pendry, “Negative refraction makes a perfect lens,” Phys. Rev. Lett. 85, 3966–3969 (2000).

221. N. Chamanara, S. Taravati, Z.-L. Deck-Léger, et al., “Optical isolation based on space–time engineered asymmetric photonic band gaps,” Phys. Rev. B 96, 155409 (2017).

222. Z.-L. Deck-Léger, N. Chamanara, M. Skorobogatiy, et al., “Uniform-velocity spacetime crystals,” Adv. Photonics 1, 056002 (2019).

223. E. Galiffi, P. Huidobro, and J. B. Pendry, “Broadband nonreciprocal amplification in luminal metamaterials,” Phys. Rev. Lett. 123, 206101 (2019).

224. J. Pendry, E. Galiffi, and P. Huidobro, “Gain in time-dependent media—a new mechanism,” J. Opt. Soc. Am. B 38, 3360–3366 (2021).

225. Z. Yu and S. Fan, “Complete optical isolation created by indirect interband photonic transitions,” Nat. Photonics 3, 91–94 (2009).

226. X. Wang, A. Diaz-Rubio, H. Li, et al., “Theory and design of multifunctional space–time metasurfaces,” Phys. Rev. Appl. 13, 044040 (2020).

227. A. E. Cardin, S. R. Silva, S. R. Vardeny, et al., “Surface-wave-assisted nonreciprocity in spatio-temporally modulated metasurfaces,” Nat. Commun. 11, 1469 (2020).

228. Y. Hadad, D. L. Sounas, and A. Alù, “Space-time gradient metasurfaces,” Phys. Rev. B 92, 100304 (2015).

229. S. Lee, J. Park, H. Cho, et al., “Parametric oscillation of electromagnetic waves in momentum band gaps of a spatiotemporal crystal,” Photonics Res. 9, 142–150 (2021).

230. P. A. Huidobro, E. Galiffi, S. Guenneau, et al., “Fresnel drag in space–time-modulated metamaterials,” Proc. Natl. Acad. Sci. 116, 24943–24948 (2019).

231. Z. Li, X. Ma, Z.-L. Deck-Léger, et al., “Wave-medium interactions in dynamic matter and modulation systems,” arXiv (2024).

232. J. Pendry, E. Galiffi, and P. Huidobro, “Gain mechanism in time-dependent media,” Optica 8, 636–637 (2021).

233. S. Taravati and G. V. Eleftheriades, “Generalized space–time-periodic diffraction gratings: theory and applications,” Phys. Rev. Appl. 12, 024026 (2019).

234. J. Park and B. Min, “Spatiotemporal plane wave expansion method for arbitrary space–time periodic photonic media,” Opt. Lett. 46, 484–487 (2021).

235. J. C. González, J. C. Miñano, and P. Benítez, “Mode analysis of a class of spatiotemporal photonic crystals,” arXiv (2010).

236. Y. Sharabi, A. Dikopoltsev, E. Lustig, et al., “Spatiotemporal photonic crystals,” Optica 9, 585–592 (2022).

237. P. Garg, A. G. Lamprianidis, D. Beutel, et al., “Modeling four-dimensional metamaterials: a T-matrix approach to describe time-varying metasurfaces,” Opt. Express 30, 45832–45847 (2022).

238. J. B. Khurgin, “Energy and power requirements for alteration of the refractive index,” Laser Photonics Rev. 18, 2300836 (2024).

239. Z. Wang, Y. Chong, J. Joannopoulos, et al., “Observation of unidirectional backscattering-immune topological electromagnetic states,” Nature 461, 772–775 (2009).

240. B. Bahari, A. Ndao, F. Vallini, et al., “Nonreciprocal lasing in topological cavities of arbitrary geometries,” Science 358, 636–640 (2017).

241. M. Segev and M. A. Bandres, “Topological photonics: where do we go from here?” Nanophotonics 10, 425–434 (2021).

242. J. Ma and Z.-G. Wang, “Band structure and topological phase transition of photonic time crystals,” Opt. Express 27, 12914–12922 (2019).

243. Y. Long, L. Zou, L. Yu, et al., “Inverse design of topological photonic time crystals via deep learning,” Opt. Mater. Express 14, 2032–2039 (2024).

244. J. Zak, “Berry’s phase for energy bands in solids,” Phys. Rev. Lett. 62, 2747–2750 (1989).

245. R.-Y. Dong, Y.-M. Liu, J.-Y. Sui, et al., “Band structure and temporal topological edge state of continuous photonic time crystals,” IEEE Trans. Antennas Propag. 72, 674–682 (2024).

246. M. Lin, S. Ahmed, M. Jamil, et al., “Temporally-topological defect modes in photonic time crystals,” Opt. Express 32, 9820–9836 (2024).

247. J. C. Serra and M. G. Silveirinha, “Rotating spacetime modulation: topological phases and spacetime Haldane model,” Phys. Rev. B 107, 035133 (2023).

248. L. Lu, J. Joannopoulos, and M. Soljacic, “Topological photonics,” Nat. Photonics 8, 821–829 (2014).

249. J. C. Serra and M. G. Silveirinha, “Engineering topological phases with a traveling-wave spacetime modulation,” arXiv (2023).

250. F. Biancalana, A. Amann, and E. P. O'Reilly, "Gap solitons in spatiotemporal photonic crystals," Phys. Rev. A 77, 011801 (2008).

251. M. S. Mirmoosa, M. H. Mostafa, A. Norrman, et al., “Time interfaces in bianisotropic media,” Phys. Rev. Res. 6, 013334 (2024).

252. M. H. M. Mostafa, M. Mirmoosa, M. Sidorenko, et al., “Temporal interfaces in complex electromagnetic materials: an overview,” Opt. Mater. Express 14, 1103 (2024).

253. S. F. Koufidis, T. T. Koutserimpas, F. Monticone, et al., “Light propagation in time-periodic bi-isotropic media,” arXiv (2024).

254. C. Rizza, G. Castaldi, and V. Galdi, “Nonlocal effects in temporal metamaterials,” Nanophotonics 11, 1285–1295 (2022).

255. L. Shaposhnikov, E. Barredo-Alamilla, F. Wilczek, et al., “Probing ultrafast magnetization dynamics via synthetic axion fields,” arXiv (2024).

256. R. E. Jacobsen, S. Arslanagić, and A. V. Lavrinenko, “Water-based devices for advanced control of electromagnetic waves,” Appl. Phys. Rev. 8(4), 041304 (2021).

257. B. Edwards, A. Alù, M. E. Young, et al., “Experimental verification of epsilon-near-zero metamaterial coupling and energy squeezing using a microwave waveguide,” Phys. Rev. Lett. 100, 033903 (2008).

258. Y. Li, A. Nemilentsau, and C. Argyropoulos, “Resonance energy transfer and quantum entanglement mediated by epsilon-near-zero and other plasmonic waveguide systems,” Nanoscale 11, 14635–14647 (2019).

259. M. S. Mirmoosa, T. Setälä, and A. Norrman, “Quantum theory of wave scattering from electromagnetic time interfaces,” arXiv (2023).

![](images/1801d0121e0cae05f8d6dc329938c5f765292a5610bb0d399f09e7af0d181753.jpg)

Mohammad Mahdi Asgari received his B.Sc. degree in Electrical Engineering (Telecommunications) from Babol Noshirvani University of Technology, Mazandaran, Iran in 2018 and his M.Sc. in Electrical Engineering (Field and Waves) from the Sharif University of Technology, Tehran, Iran in 2021. He has been a doctoral researcher in the Department of Electronics and Nanoengineering at Aalto University, Espoo, Finland, since 2022. His main research topics are metamaterials, time-varying systems, and inverse design.

![](images/1912f2685dbed6595684307286f9adb56943cd71ea7c3331a90c493ffd779f8c.jpg)

Puneet Garg received his Bachelor's degree in Physics (Honors) from St. Stephen's College, India, in 2020. He then moved to Karlsruhe Institute of Technology (KIT), Germany, to pursue a Master's degree in Optics and Photonics, which he received in 2022. Currently, he works as a doctoral researcher at the Institute of Theoretical Solid State Physics at KIT. His research interests include the theoretical and numerical investigation of time-varying metamaterials.

![](images/7cec6ad1196763bdfbf6efcb317fff211ec1ef2dcad61733546bf2869685fac8.jpg)

Xuchen Wang received a B.Sc. degree in optical information science and technology from Northwestern Polytechnical University, Xi'an, China, in 2011, a master's degree from the Department of Optical Engineering, Zhejiang University, Hangzhou, China, in 2014, and a Ph.D. degree (Hons.) from the Department of Electronics and Nanoengineering, School of Electrical Engineering, Aalto University, Aalto, Finland, in 2020. He worked as a Radio Frequency Engineer at Huawei (Shanghai, China), and TP-Link

(Shenzhen, China) from 2014 to 2016. He worked as a Post-Doctoral Researcher at Karlsruhe Institute of Technology, Germany from 2022 to 2023. He is currently working as a Professor in the Harbin Engineering University, China.

![](images/bcf0a4b7af38887bdfdaec5ba7999f1d2b70b7f71c314bcd37fdc0e6b26300e5.jpg)

Mohammad Sajjad Mirmoosa received a B.Sc. degree from the Shahid Bahonar University of Kerman, Kerman, Iran, in 2011, and the M.Sc. and D.Sc. degrees from Aalto University, Aalto, Finland, in 2013 and 2017, respectively, all in Electrical Engineering. He is currently with the Department of Physics and Mathematics, University of Eastern Finland, as a Project Researcher. His main research interests include theories of classical and quantum electromagnetism.

![](images/7766927c987429704530fabcf0910c6de8418941fb536adf628e9c466fcf65b7.jpg)

Carsten Rockstuhl received a Ph.D. from the University of Neuchaâtel, Neuchaâtel, Switzerland in 2004. After a Postdoc. period at AIST in Tsukuba, Japan, he has been since 2005 with the Friedrich Schiller University of Jena, Jena, Germany. In 2013, he was appointed a full professor at the Karlsruhe Institute of Technology, Karlsruhe, Germany. He works on many aspects in the context of theoretical and computational nano-optics. He serves the community as an editor with multiple journals. Moreover, he is a member of the Karlsruhe School of Optics and Photonics, where he currently acts as the dean of study, the Max Planck School of Photonics, and is a fellow of Optica.

![](images/cf0aac05947c3b8d1605f98a9341929fb32e97b47dc5cab8115300597578f1ea.jpg)

Viktar Asadchy received his Diploma and M.Sc. degrees in Physics from Gomel State University, Belarus, in 2013 and 2014, respectively. In 2017, he obtained his D.Sc. degree in Electrical Engineering from Aalto University, Finland. From 2019 to 2022, he served as a Postdoctoral Fellow at the Department of Electrical Engineering, Stanford University, CA, USA. He is currently an Assistant Professor in the Department of Electronics and Nanoengineering at Aalto University, Finland. He is an Elected Associate

Member of URSI. His primary research interests include metasurfaces, reconfigurable intelligent surfaces, metamaterials, photonic crystals, time-varying systems, and nanophotonics.
