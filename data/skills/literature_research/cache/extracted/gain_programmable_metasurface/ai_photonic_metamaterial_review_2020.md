# Toward Intelligent Metasurfaces: The Progress from Globally Tunable Metasurfaces to Software-Defined Metasurfaces with an Embedded Network of Controllers

Odysseas Tsilipakos, $^{*}$ Anna C. Tasolamprou, Alexandros Pitilakis, Fu Liu, Xuchen Wang, Mohammad Sajjad Mirmoosa, Dimitrios C. Tzarouchis, Sergi Abadal, Hamidreza Taghvaee, Christos Liaskos, Ageliki Tsiolaridou, Julius Georgiou, Albert Cabellos-Aparicio, Eduard Alarcón, Sotiris Ioannidis, Andreas Pitsillides, Ian F. Akyildiz, Nikolaos V. Kantartzis, Eleftherios N. Economou, Costas M. Soukoulis, Maria Kafesaki, and Sergei Tretyakov

Metasurfaces, the ultrathin, 2D version of metamaterials, have recently attracted a surge of attention for their capability to manipulate electromagnetic waves. Recent advances in reconfigurable and programmable metasurfaces have greatly extended their scope and reach into practical applications. Such functional sheet materials can have enormous impact on imaging, communication, and sensing applications, serving as artificial skins that shape electromagnetic fields. Motivated by these opportunities, this progress report provides a review of the recent advances in tunable and reconfigurable metasurfaces, highlighting the current challenges and outlining directions for future research. To better trace the historical evolution of tunable metasurfaces, a classification into globally and locally tunable metasurfaces is first provided along with the different physical addressing mechanisms utilized. Subsequently, coding metasurfaces, a particular class of locally tunable metasurfaces in which each unit cell can acquire discrete response states, is surveyed, since it is naturally suited to programmatic control. Finally, a new research direction of software-defined metasurfaces is described, which attempts to push metasurfaces toward unprecedented levels of functionality by harnessing the opportunities offered by their software interface as well as their inter- and intranetwork connectivity and establish them in real-world applications.

## 1. Introduction

Metasurfaces (MSs) $^{[1]}$ are artificial ultrathin sheet materials that constitute the 2D version of metamaterials. $^{[2]}$ They are composed of subwavelength meta-atoms (unit cells) arranged on a typically planar surface. The building blocks can be made of diverse material compositions (dielectric, metallic, semiconducting) which are arranged into various geometries to allow for the desired response to electromagnetic waves. Due to their subwavelength periodicity, metasurfaces can be characterized by effective, homogenized surface material parameters, such as electric and magnetic sheet conductivities or impedances. $^{[3]}$ Depending on the resonant properties and the shape of the meta-atoms, MSs can provide fascinating properties and capabilities for electromagnetic wave control, $^{[1,4]}$ such as perfect absorption, $^{[5]}$ anomalous reflection, $^{[6]}$ wavefront shaping, $^{[7]}$ polarization control, $^{[8]}$ and dispersion engineering. $^{[9]}$

Metasurfaces can be distinguished in 1-port (reflection) or 2-port (transmission) type structures. Their unit-cell response is described in terms of (local) complex reflection and transmission coefficients. Control over these complex coefficients, locally or globally, can give rise to a multitude of tunable functionalities and respective applications, for example, absorber, polarizer, anomalous reflector, etc. Fundamental physical properties of the resonant unit cells underlie the operation regimes, that is, when a reflective metasurface can operate as a tunable absorber/ reflector $^{[10,11]}$ or when a transmissive metasurface is transparent/absorptive, $^{[12]}$ as well as the transitions between them, $^{[13]}$ as can be shown with, for example, coupled mode theory.

Historically, early metasurfaces were used to provide a fixed electromagnetic response for a specific incident wave (frequency, incidence direction, polarization) that was assumed a priori known. It was soon realized that the capability of dynamically tuning the metasurface properties, either to change the desired output functionality or to adapt it to an input wave of different characteristics, would greatly enhance the potential of metasurfaces for practical applications, giving rise to tunable metasurfaces. $^{[14-17]}$ Further developing the concept of tunability toward reconfigurable and programmatically controlled metasurfaces unfolded the promise of artificial intelligent sheet materials that can dynamically shape electromagnetic fields and have enormous impact on imaging, communication, and sensing applications. Excellent reviews on metasurfaces have appeared in the literature, covering the underlying physics, the different applications and functionalities, the approaches to tunability, and the opportunities of nonlinearity and non-reciprocity. $^{[1,4,16-21]}$ In refs. [1,4,18,19], the authors highlight the broad range of functionalities and applications that can be performed by metasurfaces despite their subwavelength thickness. Ref. [20] provides an account of nonlinear metasurfaces and in ref. [21] the progress on gradient metasurfaces is reported. In refs. [16,17], the authors focus on tunable metasurfaces and provide comprehensive reports of the different physical mechanisms and the relevant research.

![](images/a675fb51d2953b73cf7157284c27a1c920776ade4befc9077881f2e717d838f4.jpg)

Odysseas Tsilipakos received the Diploma and Ph.D. degrees from the Department of Electrical and Computer Engineering, Aristotle University of Thessaloniki (AUTh), Greece, in 2008 and 2013, respectively. From 2014 to 2015, he was a postdoctoral research fellow with AUTh. Since 2016 he is a Postdoctoral

Researcher at the Institute of Electronic Structure and Laser in Foundation for Research and Technology Hellas. His current research interests include metasurfaces and metamaterials, plasmonics and nanophotonics, nonlinear optics, graphene and 2D photonic materials, and theoretical and computational electromagnetics.

![](images/3e01b91e86eda22f4060f60bb8e8b94aabd3c502c54f779423acde8562e6c57c.jpg)

Sergei Tretyakov received the Dipl. Eng.-Physicist, Ph.D., and D.Sc. degrees in radiophysics from Saint Petersburg State Technical University, Saint Petersburg, Russia, in 1980, 1987, and 1995, respectively. From 1980 to 2000, he was with the Radiophysics Department, Saint Petersburg State Technical University. He is

currently a professor of radio science with the Department of Electronics and Nanoengineering, Aalto University, Espoo, Finland. His current research interests include electromagnetic field theory, complex media electromagnetics, metamaterials, and microwave engineering.

In this Progress Report, we present a review up to the very recent research advances on tunable and reconfigurable metasurfaces, covering thoroughly the field of coding and programmable metasurfaces. Subsequently, we go beyond that to highlight prospective research directions toward software-defined metasurfaces and identify the emerging research challenges. The paper is organized as follows: In Section 2, we first discuss globally tunable metasurfaces, where all unit cells are collectively controlled in the same way, tracing the historical evolution of tunable metasurfaces from its very beginning. We provide a comprehensive classification based on the physical mechanisms used for the tuning, namely, electro-optical, optical, magnetic, and thermal tuning. Next, in Section 3 we discuss locally tunable metasurfaces where each unit cell can be independently tuned, allowing for reconfigurability and the ability to change between different functionalities. Amongst the various tuning mechanisms employed to construct locally tunable metasurfaces, we mostly focus on lumped electronic elements controlled with voltage, as this approach is naturally compatible with the coding (Section 4) and software-defined metasurfaces (Section 5) that follow. In Section 4, we describe the important class of coding metasurfaces, in which each unit cell can offer discrete amplitude/phase response states, that act as “bits” to construct a “code” implemented on the metasurface. By modifying the individual settings of the unit cells, the code can be varied and the coding metasurface can implement (and switch between) different functionalities at will. Finally, in Section 5, we proceed to present our view on the future of programmable metasurfaces and highlight the emerging research challenges. We highlight the new research direction of software-defined metasurfaces (SDMs), whose unit cells are equipped with controllers instead of mere actuators that possess actuating, communication, and sensing capabilities. The controllers communicate with each other by means of an intrasurface network and then with a software application through a gateway, allowing for full programmatic control and interconnectivity. Such SDMs aim to bridge the gap between academic research and real-world applications and supply even more functionalities such as synergetic actuation, sensing, adaptation, and resilience to faults (self-healing). We envision that SDMs can have a transformational effect in a broad range of practical applications, ultimately serving as interconnected intelligent surfaces that form an internet of materials for shaping electromagnetic field distributions.

## 2. Global Tunability at the Metasurface Level

Tunability can be achieved by introducing in the metasurface composition materials or components (e.g., lumped electronic elements) whose electromagnetic properties can be modified by means of external stimuli. Depending on the material/element introduced, there are different means of applying the driving signal, namely, through a low-frequency electric or magnetic bias field, using an optical pump beam, a voltage source, or changing the ambient temperature. By externally modulating this applied stimulus, the electromagnetic properties (e.g., permittivity or conductivity of a material) can be varied, thus leading to a modification of the effective electric and/or magnetic response of the metasurface and the tuning of the electromagnetic output. The most elementary form of tuning concerns the entire metasurface as a whole: all meta-atoms constituting the metasurface are tuned collectively in the same way. This type of tuning is termed “global tuning” and will be thoroughly discussed in this section using a classification based on the physical tuning mechanism employed, that is, electro-optical, optical, magnetic, thermal, or voltage-based tuning. Note that these mechanisms are not specific to globally tunable metasurfaces; we use globally tunable examples to present them, since these structures were the first to be historically demonstrated.

## 2.1. Electro-Optical Tuning

We first discuss electro-optical tuning, that is, changing the high-frequency (“optical”) electromagnetic material properties by varying a low-frequency electric field across the material. This is typically achieved by applying a voltage between appropriate electrodes. By properly adjusting the bias voltage, modulation of the phase and amplitude response can be demonstrated. Liquid crystals and graphene are two of the most prominent materials of this class that have found numerous applications in metasurfaces; they will be discussed in detail in what follows. In addition, the use of electro-optic polymers with high electro-optic coefficient $^{[22,23]}$ or conducting oxides, which have found many applications in NIR and optical frequencies, $^{[24-28]}$ are other interesting alternatives for demonstrating electro-optically tunable metasurfaces.

## 2.1.1. Liquid Crystals

Liquid crystals (LCs) belong to a special class of soft condensed matter; they are characterized by fluidity but at the same time they exhibit properties commonly met in crystalline solids, such as anisotropy. Depending on the strength of ambient factors and stimuli, such as external electric/magnetic field, temperature and pressure, LCs can be found in different phases, enabling their use for constructing tunable metasurfaces. The most common phase of LCs is the nematic phase, in which the long-range order is of orientational type, that is, the elongated molecules of the LC tend to align in the same average orientation referred to as the nematic director. The direction of the nematic director defines anisotropic optical properties of LCs, characterized by the ordinary and extraordinary refractive indices; thus, controlling the nematic director allows for shaping the optical properties. Note that the voltage-dependent birefringence and the dependence of the refractive index on the polarization and propagation direction of light are fascinating and useful features of nematic LCs, which have led to the massive development of the optical display technology over the last decades.

It was soon realized that the mature technology of nematic LCs could be exploited for the realization of tunable metasurfaces and metadevices. $^{[31,32]}$ Due to their liquid character, nematic LCs can be easily infiltrated into various structures where they can provide modulation of the refractive index. Additionally, nematic LC devices typically require low power consumption, since the operation is, in most applications, capacitive in nature with negligible current flow associated with the applied voltage. Furthermore, the already-present conducting parts (such as metallic patches) can undertake the role of the electrodes, thus no additional electrodes are required in these metasurfaces. On the negative side, nematic LCs are characterized by a relatively slow response time, which prohibits their use when ultrafast modulation is required.

Molecules forming the nematic liquid crystal phase at room temperature are typically a few nanometers long along the extraordinary optical axis and 1 nm or less perpendicularly to it, while a liquid crystal collection of molecules is safely assumed as a continuum in domains of the order of 100 nm wide or less. The thickness of the layers along with the electrode lateral dimensions determine the minimum in-plane control due to capacitive fringing effects, which should be taken into consideration for engineering the tunable electromagnetic functions. Initial works on LC-based metamaterials were conducted in the microwave regime. One of the first works demonstrated the electrically tunable negative permeability by infiltrating nematic LCs in a periodic array of split ring resonators. It was shown that the resonant frequency can be continuously and reversibly adjusted by modulating the applied electric field. $^{[33]}$ Later, implementations with wires and fishnet structures in the microwave regime $^{[34,35]}$ were demonstrated. Following these microwave experiments, electro-optical metamaterials with LC inclusions of deeply subwavelength thickness have been demonstrated in the terahertz and near infrared (NIR) regimes. $^{[30,36,37]}$ Note that LC absorption bands exist in the far-IR end mid-IR, $^{[38]}$ posing limitations to the targeted operation frequency.

![](images/2355053638ef7d24fbffe04ebd14d5f00ffc8ce01bc2e9ed43fc9dd5574178c1.jpg)
Figure 1. a) Liquid crystal-based metasurface absorber. Reproduced with permission. $^{[5]}$ Copyright 2013, American Physical Society. b) Metasurface absorption function for THz applications based on the liquid crystal induced spatial modulation. Reproduced with permission. $^{[29]}$ Copyright 2014, Wiley-VCH. c) Multifunctional photonic switch in a nanostructured metasurface based on the electrically controllable liquid-crystal load. Reproduced under the terms of a Creative Commons Attribution CC-BY license. $^{[30]}$ Copyright 2015, The Authors, published by Wiley-VCH.

To date, nematic LCs have been exploited for a variety of applications. In tunable absorption operation, incorporating LCs in proper locations within the unit cell has been shown to offer dynamic control over the resonance frequency and the absorption bandwidth. $^{[5]}$ For example, with the unit cell shown in Figure 1a, a modification of the absorption by 30% at 2.62 THz, as well as tuning of the resonant absorption over 4% bandwidth have been demonstrated. Nematic-LC-based metasurfaces have been also examined for metadevices targeting spatial modulation applications. $^{[29]}$ As an example shown in Figure 1b, a tunable metasurface made of $6 \times 6$ pixels is demonstrated, where the response of each pixel is modulated by electronically controlling the orientation of the LC. In addition, in Figure 1c, a multifunctional photonic switch with a resonance response controlled both in terms of its magnitude and wavelength has been demonstrated. Other proposed applications involve electrically tunable phase modulators and polarization converters, $^{[11,39]}$ cloaks with the capability of real-time control of invisibility, $^{[40]}$ and tunable frequency-selective surfaces. $^{[41,42]}$ Moreover, as LCs exhibit a large anisotropy under modulation, they have been exploited in hyperbolic metamaterial implementations offering electric-field-driven toggling between elliptic and hyperbolic dispersion relations, and between normal and negative refraction. $^{[43]}$

## 2.1.2. Graphene

Graphene is a 2D (one-atom-thick) material that exhibits superior properties such as ultrahigh carrier mobility and exceptional electrical conductivity. Over the past decade, graphene has been the subject of intensive research for its potential applications in future electronic and photonic devices. Due to its unique band structure, $^{[44]}$ the conductivity of graphene can be dynamically modulated by electric, magnetic and optical means, which opens an avenue for the realization of tunable electric and optoelectronic devices. $^{[45,46]}$

It is well known that a pristine graphene layer interacts weakly with incident light due to its monoatomic thickness, limiting its applications in the control of wave propagation. To address this problem, in the pioneering work, $^{[51]}$ a graphene layer is patterned into periodic disks to create localized plasmon resonance and therefore enhance the light–matter interactions in graphene. Following this step, several works have demonstrated efficient wave manipulation with nanostructured graphene metasurfaces, and focused on the tunability achieved by varying the Fermi level of graphene by means of electric gating. Important demonstrations include the modulation of wave amplitude $^{[52,53]}$ and phase $^{[54,55]}$ in both reflection and transmission modes particularly for tunable absorption operation. $^{[56-59]}$ For example, in the representative work, $^{[53]}$ graphene is patterned into an array of nanoribbons to excite localized plasmon resonances in the terahertz band; the transmission peak can shift to higher frequencies when increasing the carrier concentration of graphene. In ref. [47], by asymmetrically structuring the graphene meta-atom in two orthogonal directions (x and y), the phase shift of TE- or TM-polarized wave can be tuned by the external voltage, which realizes dynamic control of polarization states (Figure 2a).

For most of the above designs, there usually exists a trade-off between the quality of graphene (carrier mobility or relaxation time) and the efficiency of devices, especially for those functionalities where the ohmic losses in graphene are not desirable. For example, in refs. [47,52], graphene mobility is assumed to be as high as 10 000 cm $^{2}$ V $^{-1}$ s $^{-1}$ to suppress the losses in graphene. Although even higher mobilities have been measured in suspended or hexagonal boron nitride supported graphene with rather careful treatment, the typical reported mobilities of processed graphene are around 1000 cm $^{2}$ V $^{-1}$ s $^{-1}$ , which limits the practicality of these designs.[60] In order to enhance the light-matter interaction in graphene with low mobility, graphene is integrated with auxiliary structures such as metallic or dielectric metasurfaces. Such structures can tightly localize incident fields and therefore enhance the light-matter interaction in graphene.[48–50,61–63] For instance, in ref. [48], graphene nanoribbons are embedded in the capacitive gaps of metallic strips, and strongly localized surface plasmons are excited in graphene due to the presence of the metallic gap. As a result, complete absorption in graphene was experimentally observed with an on-off modulation efficiency of 95.9% in reflection mode (Figure 2b). In ref. [49], a simpler structure is proposed, where a graphene sheet is placed directly on a metallic high impedance surface without structural patterning (Figure 2c). The metallic patches not only reduce the effective resistance of graphene, but also provide a high impedance surface which is essential for Salisbury-screen-type absorbers. The absorption frequency can be dynamically tuned from 4 THz to 12 THz when changing the Fermi level of graphene from 0.1 eV to 1.0 eV. In a different application, the phase shift of two orthogonal linearly polarized waves can be tuned and thus the polarization state of the incident wave can be dynamically controlled. $^{[61-63]}$ In a recent work $^{[50]}$ researchers demonstrated gate-controlled circular dichroism (CD) and optical activity (OA) in graphene incorporated chiral metasurface (Figure 2d). It was shown that by changing the conductivity of graphene, the transmission of a right-handed circular polarized (RCP) wave can be efficiently modulated without affecting the other circular polarization. Furthermore, the rotation angle of the transmitted linearly polarized wave can be also actively controlled.

![](images/878582c0f8bf6994317b9d77404c4af4ebfffa3727c3f4389e2dfa2b7738ddef.jpg)

![](images/c553b201069bf5038ff596138f7e2d6ad8618f5b7c893d9adc34af808ef62fa1.jpg)

![](images/ba47819376bb1db837eb28a01569b5aba8526929b9dcb52f0871587a3b1179a8.jpg)

![](images/76fdfab626d2de5d3c8ec380835ca17ec689c6f2eb6e458c73c554562fbac404.jpg)
Figure 2. Tunable metasurfaces comprising graphene. a) Metasurface with patterned graphene for enhancing light–matter interactions. Application to the polarization control of scattered waves. Reproduced with permission. $^{[47]}$ Copyright 2013, The Optical Society. b) Graphene–metal hybrid structures for amplitude tunability. Reproduced with permission. $^{[48]}$ Copyright 2018, American Chemical Society. c) Graphene–metal hybrid structures for frequency-tunable absorbers. Reproduced with permission. $^{[49]}$ Copyright 2019, IEEE. d) Graphene–metal hybrid structures for tunable polarizers. Reproduced with permission. $^{[50]}$ Copyright 2017, The Authors, some rights reserved; exclusive licensee American Association for the Advancement of Science. Distributed under a Creative Commons Attribution NonCommercial License 4.0 (CC BY-NC) (http://creativecommons.org/licenses/by-nc/4.0/).

## 2.2. Optical Tuning

Optical illumination can be also employed for making tunable metasurfaces by introducing light-sensitive materials in the metasurface composition. In what follows, we focus on pump-probe approaches where a dedicated pump beam is used to change the properties of materials, such as photoconductive semiconductors and graphene. Apart from semiconductors and graphene, alternative materials can be exploited for realizing optically tunable metasurfaces. For example, optomechanical approaches can be used to create adaptive loads in metasurfaces; for example, by using specialized materials such as the poly disperse red 1 acrylate (PDR1A) which is able to expand when exposed to circularly polarized (CP) light and can quickly return to its previous state when exposed to linearly polarized (LP) light. $^{[64]}$ Integrating this material in capacitor-like structures can offer variable reactive response $^{[64,65]}$ and be exploited, for example, for tunable absorbing metasurfaces. $^{[65]}$ In addition, optical tuning can be achieved by exploiting the binary isomeric states of an ethyl red switching layer upon light stimulation, leading to polarization control at optical wavelengths. $^{[66]}$

## 2.2.1. Semiconductors

For implementing metasurfaces tunable with light, photoconductive semiconductor materials constitute a prime candidate, with the most prominent examples being Si and GaAs. Their conductivity can be tuned by photoexciting carriers with an infrared pump beam, providing a means for tuning the metasurface response. Various implementations operating in the THz band have been proposed, primarily based on the split-ring resonator (SRR) meta-atom. On one hand, a semiconductor (GaAs) can be employed as the substrate material. $^{[67,68]}$ As the conductivity increases under excitation, the resonance is quenched, modulating the amplitude of the response. On the other hand, semiconductor materials (Si) can be also used to construct resonant meta-atom inclusions. $^{[69,70]}$ In this case, carrier photoexcitation effectively modifies the resonator geometry: depending on the pump fluence, the semiconducting section can switch from a dielectric-like to a conductor-like state. As a result, the fundamental resonance can experience a red shift $^{[69]}$ or a transition to a different, blue-shifted, resonance can be triggered $^{[70]}$ (Figure 3a). Apart from SRRs, Si-connected patches for transitioning from a cut-wire array (exhibiting Lorentzian resonant response) to a wire-grating geometry (showing Drude response) have been also proposed. $^{[71]}$ In addition, semiconductor inclusions have been also used in chiral metasurfaces for tuning the dichroism and optical activity or switching the metasurface handedness. $^{[72,73]}$

## 2.2.2. Graphene

Besides the electric control discussed in Section 2.1.2, graphene can be also tuned optically as its conductivity at terahertz frequencies can be modified by an optical pump which may affect the carrier concentration and energy distribution. $^{[75]}$ Over the past few years, a variety of optically stimulated graphene-based tunable metasurfaces have been proposed. These include distributed Bragg resonators for radiation amplification and lasing, $^{[76]}$ parity-time symmetric systems for sensing applications, $^{[77]}$ and a graphene-based flat absorber with ultrafast modulation which has been experimentally demonstrated $^{[74]}$ (Figure 3b). The material properties of graphene can be modified also through nonlinear self-action. Graphene has shown great promise for nonlinear applications by exploiting the geometry-induced second-order $^{[78]}$ and the intrinsic third-order $^{[79]}$ nonlinear susceptibilities, and by relying on single- $[^{80}]$ and multichannel $[^{81}]$ effects.

![](images/376808b1ca7298019ee74eb75a9fa157cfb5679f90c51ba35d9ebdd29a460bac.jpg)

![](images/8fbfb0095f76cba50e56a9ce6d203109fcff58389a5c565238d7167fe3f54e52.jpg)

![](images/c7627123816ee9c195da21d89babac78d8709545b243e1a7d30bf41396a17723.jpg)

![](images/1b8324fbf5febbed70c54d8e4e1fb7c5937ab3b7d1b196b883af7e6b5ba9366e.jpg)
Figure 3. Globally tuned metasurfaces by means of optical beams. a) Metasurface with photoconductive Si inclusions in SRR gaps for operation in the THz range. Upon irradiance with an infrared pump beam, carriers are photoexcited triggering a transition to a different, blue-shifted, resonance. Reproduced with permission. $^{[70]}$ Copyright 2011, American Physical Society. b) Graphene-based flat absorber with ultrafast response: The infrared pump beam excites carriers changing the conductivity of graphene. Reproduced with permission. $^{[74]}$ Copyright 2019, American Chemical Society. https://pubs.acs.org/doi/10.1021/acsphotonics.8b01595. Request for further permissions to the excerpted material should be directed to the ACS.

## 2.3. Magnetic Tuning

Magnetically responsive materials and structures have also attracted research interest because they can nearly instantaneously respond to external magnetic stimuli in a contactless manner. One of the most well understood means of magnetic tunability in the microwave regime relies on the ferromagnetic resonance (FMR) arising in ferrite materials such as yttrium iron garnets (YIG). $^{[82]}$ The addition of ferrite materials biased by an externally applied static or slowly varying magnetic field can give rise to a Lorentz-type resonance in the effective permeability, leading to high positive and negative values of $\mathrm{Re}\{\mu_{\mathrm{eff}}\}$ . $^{[82,83]}$ The characteristics of the FMR are controlled by the parameters of the ferrite material (relative permittivity $\varepsilon_{r}$ , gyromagnetic ratio $\gamma$ , saturation magnetization $M_{s}$ , linewidth $\Delta H$ ), sample dimensions, and the amplitude of the applied bias magnetic field. Specifically, for mm-thick YIG ferrite structures, the FMR frequency can be significantly shifted in the X-band (8–12 GHz) by relatively weak bias magnetic field (few kOe or, equivalently, few hundred mT), allowing for tunable operations. Various metasurfaces utilizing ferrites as tuning elements have been designed and fabricated, most notably for tunable perfect absorption, $^{[84,85]}$ with the resonant frequency being controlled by an external magnetic bias (Figure 4a,b).

![](images/c088b83dab3f3850de6c7cb68c425a9e5321728eb1582602e2b75c524ec41b94.jpg)
(a)

![](images/c2c97f00bf27ce4665ccc1b9234f03f1f54e8bd2c2eac0f129cf6db5cab39ef7.jpg)
(b)

![](images/e10503022c8786e24f0ad565665cb3b58f595e4ecc7294ec37c82053a0809c1f.jpg)

![](images/77cea51e20d57625d6c855dd09cde4e471b75db41031f2a1b2c8a16648d80fe5.jpg)
Figure 4. a) Ferrite-rod based absorptive metasurface and b) experimentally measured magnetic-field tunable resonance. Reproduced with permission. $^{[85]}$ Copyright 2016, AIP publishing. c,d) Nonreciprocal ferrite-based metasurface exhibiting unidirectional c) perfect absorption or d) reflection. Reproduced with permission. $^{[86]}$ Copyright 2012, EPLA.

Magnetic materials also provide a platform for realizing nonreciprocal phenomena, resulting from the breaking of time-reversal symmetry due to the existence of the external magnetic field. For example, magnetically controllable nonreciprocal Goos–Hänchen shift has been predicted in plasmonic gradient reflective metasurfaces, $^{[87]}$ whereas ferrites can be used in devices exploiting magnetic surface plasmons to implement tunable unidirectional absorptive metasurfaces, $^{[86]}$ as illustrated in Figure 4c,d.

Finally, magnetic-responsive structures without ferrite materials have been proposed, with examples including THz metamaterials and filters incorporating semiconductor resonant structures $^{[88]}$ and bio-inspired microplate arrays for tuned optics. $^{[89]}$ The latter is cast as a magnetically responsive metasurface with local tunability: by carefully programming the strength and spatial distribution of the external magnetic field used in the assembly process, one can attain precise positioning of a single nano-sized object in a locally magnetized area, apart from globally synchronized motion of a large collection of objects.

## 2.4. Thermal Tuning

Heat can be another stimulus for tunable metasurfaces. For efficient thermally tunable metasurfaces, the meta-atoms or the substrate of the metasurfaces should be made of materials exhibiting strong sensitivity to temperature variations. In consequence of such sensitivity, the electromagnetic response of the metasurface significantly changes as the temperature varies and we obtain a capability to tune the metasurface response. In this context, phase change materials (PCMs) $^{[90]}$ have been recognized as promising candidates. $^{[91-94]}$ The PCMs can be switched between two different phases exhibiting different optical properties. $^{[90]}$ As a characteristic example of PCMs, vanadium dioxide (VO $_{2}$ ) shows a metal-to-insulator phase transition under thermal variation. The transition temperature of VO $_{2}$ is a bit higher than room temperature. $^{[95]}$ This transition from dielectric to metal state has been used for tuning the resonance frequency, the amplitude and/or polarization of the scattered fields. For example, in the hybrid SRR-VO $_{2}$ device, $^{[91]}$ an array of SRRs is lithographically fabricated on a VO $_{2}$ film (Figure 5a). When the ambient temperature increases, the permittivity of the VO $_{2}$ substrate changes, causing a change in the capacitance of each SRR and, consequently, shifting the resonance frequency of the metasurface. The temperature of VO $_{2}$ is controlled by applying voltage which induces local heating as current flows through the VO $_{2}$ film. Furthermore, VO $_{2}$ can be utilized in designing quarter-wave plates for switching the circular polarization of the transmitted wave at two different frequencies in the THz regime. $^{[92]}$

(a)
![](images/3e9358b580c21a0be024e821506a54e5cca2cef36c00e9d2c8a77d05432dda0d.jpg)

![](images/3c072bc008844cac78f850faaeb1e1b41940b2a0e78b87378f08ed982438af05.jpg)
(c)

![](images/e8881cbab682da350364aa2694312349dcc772e2035b69beb0bd6a347c761e2f.jpg)
(d)

![](images/39e1bdb5d411395a2a22c55e6ed3d518431e54cfb13240eb97d66a9f943d4863.jpg)

![](images/10bc1c1d4b545b5ec9a3530bded8eba7244c958b4ca557e43a860e57d1081f0c.jpg)
Figure 5. a) Dynamically tunable split ring resonator metasurface with $VO_{2}$ substrate. Reproduced with permission. $^{[91]}$ Copyright 2008, AIP Publishing. b) An all-dielectric metasurface whose temperature determines the reflection and transmission amplitudes. Reproduced with permission. $^{[96]}$ Copyright 2017, Wiley-VCH. c) THz transmission amplitude spectra of a 180-nm thick YBCO superconductor metasurface at various temperatures. Reproduced with permission. $^{[97]}$ Copyright 2010, American Physical Society. d) Thermally tunable metasurface support structure consisting of alternating reconfigurable and nonreconfigurable bimaterial bridges. Reproduced with permission. $^{[98]}$ Copyright 2011, American Chemical Society.

Apart from PCMs, semiconductors and superconductors have been also utilized for this purpose, although their thermal sensitivity is not strong enough for a phase change. The temperature-dependent refractive index of semiconductors can give rise to dynamic tuning of metasurfaces. $^{[96,99,100]}$ For example, the change of temperature shifts the Mie resonances of high-index meta-atom constituents $^{[99]}$ and, consequently, results in significant changes in the metasurface functionality. $^{[99,100]}$ In ref. [96], an all-dielectric metasurface made of silicon disks is shown to provide a red shift of 0.1 nm per °C change in the near infrared spectrum due to the thermo-optical effect on the refractive index of silicon (Figure 5b). High-temperature superconductors have been also employed for tuning the resonance frequency and strength since the conductivity of superconductors is temperature dependent. In ref. [97], it was shown that the resonance of a metasurface made of an array of SRRs, which is fabricated by 180-nm thick $YBa_{2}Cu_{3}O_{7-\delta}$ films ( $\delta = 0.05$ ), experiences a red shift while the transmission amplitude is increased by 60% as the temperature rises from T = 20 K to T = 100 K (Figure 5c).

The impact of temperature-varying materials can be also exploited for mechanically reconfigurable photonic structures $^{[98]}$ (Figure 5d). For example, when the support beam of a metasurface is made of two materials with different thermal expansion coefficients, the beam can provide mechanical bending in response to temperature variations, thus modulating the metasurface response. It was shown that the transmission amplitude increases by 51% as the metasurface temperature changes from T = 76 K to T = 270 K. If the temperature reduces to T = 76 K again, the transmission returns to its initial value, meaning that the tunable metasurface possess reversible operation. $^{[98]}$

## 2.5. Lumped Element Tuning with Voltage

Tunable lumped electronic elements, for which the input impedance can be tuned by means of DC voltage signals (biasing), provide a powerful way for realizing tunable metasurfaces, particularly for microwave implementations. Characteristic examples of such components are variable resistors, PIN switch diodes, and P-N varactor diodes. Tunable metasurfaces are realized by incorporating lumped elements in the metatom configuration and addressing them with proper biasing signals. When all the elements are collectively controlled by modulating the global bias voltage, the surface impedance of the metasurface is uniformly modified and can lead to the tuning of a preselected functionality.

## 2.5.1. Switch Diodes

A PIN switch diode has two states, “ON” (conducting) and “OFF” (insulating), which can be accessed by forward bias and zero- (or reverse-) bias, respectively. This enables switchable metasurfaces that can toggle between two distinct operating states. For example, in ref. [101] a metasurface whose response can be switched between total absorption and full reflection by tuning the bias voltage has been proposed. The state of the diode influences the surface impedance of the metasurface to match (total absorption) or mismatch (total reflection) with the free-space impedance. In a different design, $^{[102]}$ both the scattering and polarization properties are modified. When the diodes are in the “on” state, the linearly polarized wave that is impinging on the metasurface is reflected retaining the same polarization. On the other hand, when the diodes are toggled to the “off” state, the incident wave is transmitted with perfect polarization conversion to the orthogonal state.

## 2.5.2. Varactor Diodes

Another type of lumped elements introduced in metasurfaces is P-N varactor diodes (tunable capacitors), whose reactance can be adjusted (associated with a relatively small change of resistance) in a continuous fashion by varying the reverse bias voltage. $^{[103]}$ This is in contrast with the two discrete states achievable with switch diodes. By addressing the varactors collectively with the same voltage one can achieve frequency tunability. $^{[104,105]}$ One of the most actively investigated functionalities is tunable perfect absorption; modifying the reverse biasing voltage of the varactors leads to a shift of the resonance frequency and, consequently, the spectral position of perfect absorption. $^{[106,107]}$ Usually, capacitances in the order of few pico-Farads (0.5–5 pF) are typically required for this purpose and can be obtained with commercially available diodes using moderate reverse bias voltages of the order of 0–15 V. In addition, off-the-shelf varactors come in packages of few cubic millimeters, making it possible to integrate them in GHz metasurafaces.

In the same spirit, one may tune the capacitance of microelectromechanical systems (MEMS), by using the piezoelectric effect. This route has been exploited for example to demonstrate continuously tunable high impedance metasurfaces $^{[108]}$ that can act as perfect (artificial) magnetic conductors. Finally, the voltage required to tune a varactor can be generated optically, when a photodiode is illuminated by white light using common LED and focusing optics. $^{[109]}$

In Table 1, we provide a brief summary of the wealth of tuning mechanisms and materials that have been exploited in globally tunable metasurfaces. The same tuning mechanisms can be utilized for building locally tunable metasurfaces, which will be discussed in Section 3. Notice that most tuning mechanisms can cover a very wide range of operating frequencies. Voltage-controlled lumped elements, on the other hand, are mainly limited to the GHz regime. However, their importance is significant since they are naturally compatible with local tuning and are the predominant tuning mechanism for implementing coding and software-defined metasurfaces, which will be discussed in Sections 4 and 5, respectively.

## 3. Local Tunability at the Unit-Cell Level

In this section, we discuss local tunability at the unit-cell level. Contrary to globally tuned metasurfaces, local tunability refers to the case where each constitutive unit cell of the metasurface is independently tuned. This allows for achieving additional functionalities that rely on the spatial modulation of the surface impedance (e.g., wavefront control, holography) and also the ability to dynamically switch between functionalities, that is, reconfigurability.

In principle, local tuning can be achieved by utilizing all the tuning mechanisms presented in Section 2 for globally tunable metasurfaces. For example, based on the phased-array principle, liquid crystals have been used for constructing voltage-driven locally tunable metasurfaces for beam steering. $^{[111]}$ A fishnet metamaterial loaded with nematic LCs has been demonstrated as a voltage-tunable gradient-index lens in the microwave regime. $^{[112]}$ Following a similar approach with graphene as the locally tunable material, anomalous wave deflection has been shown in reflection $^{[113]}$ and transmission $^{[114]}$ modes. Devices for beam steering or lensing based on magnetic materials have been also proposed, provided that a spatial profiling is imparted by spatially modulating the external magnetic field. $^{[115]}$ Recent optical local-tuning approaches include focusing white light from LED arrays on photodiode arrays embedded on the back

Table 1. Summary of different physical tuning mechanisms in examples of globally tunable metasurfaces indicating the operation frequency regime, the material employed, and the function achieved.

<table><tr><td>Tuning mechanism</td><td>Operation regimes</td><td>Material</td><td>Function</td><td>Operating frequency</td><td>Ref.</td></tr><tr><td rowspan="3">Electro-optic</td><td rowspan="3">GHz to NIR</td><td>Liquid crystals</td><td>Negative permeability</td><td>8–10 GHz</td><td>[33,35]</td></tr><tr><td>Graphene</td><td>Tunable absorber</td><td>0.5–10 THz</td><td>[58,59]</td></tr><tr><td>TCO (ITO)</td><td>Amplitude modulator</td><td>1.55 μm</td><td>[24]</td></tr><tr><td rowspan="4">Optical</td><td rowspan="4">GHz to Optical</td><td>Semiconductor (Si)</td><td>Optical activity/dichroism</td><td>5–10 THz</td><td>[73]</td></tr><tr><td>Semiconductor (GaAs)</td><td>Amplitude modulator</td><td>1 μm</td><td>[110]</td></tr><tr><td>Graphene</td><td>Sensing</td><td>1.5 THz</td><td>[77]</td></tr><tr><td>Polymer (PDR1A)</td><td>Tunable absorber</td><td>GHz</td><td>[65]</td></tr><tr><td rowspan="4">Thermal</td><td rowspan="4">THz to NIR</td><td> $VO_2$ </td><td>Absorb, polarize</td><td>0.5–1.5 THz</td><td>[91,92]</td></tr><tr><td>Semiconductor (Si, InSb)</td><td>Spectral filter</td><td>1-10 μm</td><td>[96,99]</td></tr><tr><td>Superconductor (YBCO)</td><td>Spectral filter</td><td>0.6 THz</td><td>[97]</td></tr><tr><td> $Au/Si_3N_4$ </td><td>Spectral filter</td><td>1.5 μm</td><td>[98]</td></tr><tr><td rowspan="3">Magnetic</td><td rowspan="3">GHz to Optical</td><td>Ferrite (YIG)</td><td>Absorb, non-reciprocal</td><td>5–10 GHz</td><td>[84,86,87]</td></tr><tr><td>Semiconductor (InAs)</td><td>Tunable absorber</td><td>0.1–5 THz</td><td>[88]</td></tr><tr><td>PDMS/Fe</td><td>Tunable reflector</td><td>0.7–1 μm</td><td>[89]</td></tr><tr><td rowspan="3">Lumped electronic elements</td><td rowspan="3">MHz, GHz</td><td>PIN switch diode</td><td>Amplitude modulator</td><td>3.5 GHz</td><td>[101]</td></tr><tr><td>PIN switch diode</td><td>Polarization control</td><td>2 GHz</td><td>[102]</td></tr><tr><td>P-N varactor diode</td><td>Tunable absorber</td><td>2.5–5.5 GHz</td><td>[107]</td></tr></table>

![](images/8f0afd0c885ad9956823e57da141c8c7b12b2cf38652c3474db15938c3f7c2ce.jpg)
(a)

![](images/2c0238ee127ff5e49e7b58de93cb79fa3a4f93b166b1e08c3d0c49ba6f318b04.jpg)
(b)

![](images/457336bd51765faa8520b6fb81faced49393b551c5835730e03c4a6b156d6ac1.jpg)
(c)

![](images/79361dc001ffcfc78f307ab54e5dd53c610f124a351eef626552c6281741cfb3.jpg)
(d)

![](images/37a5ceccc0386b324cb76320a7aa7af116ea5bc7d9b93601bb6a2ea9f4950082.jpg)
(e)

![](images/3d9abb9156633a989fb0a04048346a8c80059909cd8c33fbd8559ead092d18fc.jpg)
(f)
Figure 6. Local tunability and the realized functionalities. a) Dielectric-rod metasurface based on elliptic ceramic cylinders for reconfigurable wavefront manipulation. Reproduced with permission. $^{[119]}$ Copyright 2018, The Authors, published by Wiley-VCH. b) A tunable metasurface realized by embedding varactors. Multiple functions such as splitting, steering, and polarization conversion are realized by controlling the varactors in each column. Reproduced with permission. $^{[120]}$ Copyright 2017, Wiley-VCH. c) Metasurface with embedded varactors for implementing a reconfigurable metalens for operation in transmission based on Huygens' principle. Reproduced with permission. $^{[121]}$ Copyright 2017, Wiley-VCH. d) A tunable metasurface realizing dynamic holography by changing the pattern of the applied voltage on the varactors in each unit cell. Reproduced from ref. [122] under Creative Commons Attribution 4.0 International License (http://creativecommons.org/licenses/by/4.0/). Copyright 2017, The Authors, published by Springer Nature. e) Comparison of the different local tuning approaches using diodes, varactors, and a chip supplying continuously varying complex input impedance (i.e., variable R and C). f) A tunable metasurface realized by embedding RC chips in each unit cell that can supply continuous tuning in the entire complex plane. It can provide tunable perfect absorption and tunable wavefront manipulation functionalities. Adapted with permission. $^{[123]}$ Copyright 2019, American Physical Society.

side of a metasurface; the photodiodes generate the voltage required to bias lumped varactors that, in turn, modify the unit cell responses, enabling functionalities such as cloaking, illusion, and vortex beam generation. $^{[109,116]}$ In addition, meta-atoms that experience certain mechanical movement under applied voltage $^{[108,117,118]}$ have been used for designing metasurfaces with tunable functionalities.

Each tuning mechanism is associated with characteristic traits and constraints, dictating its applicability for locally tunable metasurfaces depending on the application requirements and the operation frequency. For example, optical and thermal control can be prone to crosstalk between unit cells, as local control requires to localise and isolate the control signals between different unit cells. Regarding modulation speed, thermal diffusion has response times in the millisecond range, whereas free-carrier generation and recombination in photoconductive semiconductors, for example, can take place in nanosecond or even picosecond time scales. Apart from the tuning mechanisms, certain functionalities impose additional design constraints. For instance, reconfigurable wavefront manipulation typically requires an achievable phase span of $2\pi$ in the complex reflection or transmission coefficients. This large phase span, usually, translates into a significant modulation of the material parameters around the rest value. This can be achieved, for example, with ceramic ferroelectric materials featuring ample tunability (15% in permittivity) with DC bias or temperature $^{[119]}$ (Figure 6a).

A practical approach to the local control of metasurfaces that avoids crosstalk altogether is the use of voltage-tunable lumped elements discussed in Section 2.5. In addition, they are naturally suited to programmatic control with a computer which is very pertinent to software-defined metasurfaces discussed in Section 5. Thus, in the remainder of this section we focus on this tuning mechanism. Lumped elements gather numerous advantages: (i) switch diodes and varactors are relatively compact and can be integrated in the metasurface without disturbing the morphological features of the meta-atoms, (ii) they operate in a low voltage regime, and (iii) have a local effect on the meta-atom without interfering with other cells causing crosstalk, as mentioned above. These features are particularly attractive for locally tunable metasurfaces, and have lead to a broad range of demonstrations and applications. For example, in ref. [120], a voltage profile is applied on varactors along one direction. In this way, a specific 1D phase profile can be actively imprinted on the metasurface. This enables a range of tunable applications, such as tunable anomalous reflection (steering), $^{[117,118,120]}$ tunable polarization control, and beam splitting, $^{[120]}$ see Figure 6b. More generally, full 2D patterns can be imprinted on the metasurface. In this case, all individual varactors can be controlled independently and therefore one can in principle obtain any desirable pattern of the bias distribution, which has enabled researchers to demonstrate dynamic imaging by changing the focal point following the Huygens principle, $^{[121]}$ as shown in Figure 6c, and dynamic holography, $^{[122]}$ as shown in Figure 6d.

Noting that metasurfaces can be characterized by a complex (spatially modulated) surface impedance, the usage of switching diodes as the lumped elements allows for a binary approach to the local properties, that is, obtaining two phase/amplitude states for the unit cell response. On the other hand, the usage of varactors enables one to achieve a continuous variation of the reactive (imaginary) part of the equivalent surface impedance, associated with a small change of the resistive (real) part. These facts are schematically illustrated in Figure 6e. In order to exploit the full capability of the available tuning space, both the real and imaginary parts of the surface impedance need to be made continuously tunable. $^{[123,124]}$ This allows for complete tuning in the entire complex plane, schematically depicted with the gray area in Figure 6e, and, thus, space-granular and, ideally, independent control over both the amplitude and phase of the transmitted/reflected wave. Following this rationale, tunable anomalous reflection, as well as tunable perfect absorption with the same metasurface have been demonstrated in ref. [123], as schematically shown in Figure 6f. This level of control can be practically accomplished by integrating in the unit cells a tunable chip that provides continuously tunable resistance and reactance. $^{[125,126]}$ Note that ultimately addressing the individual components inside tunable chips is usually performed with quantized voltage levels, which, however, can be selected at will depending on the application and can be in principle as many as required for a smooth quasi-continuous coverage of the impedance space. $^{[127]}$

## 4. Coding and Programmable Metasurfaces

In Section 3, we briefly discussed the difference between digital and analog control. For example, switch diodes allow for strictly digital control since they possess two discrete states, whereas varactor diodes or inherently analog means of tuning such as liquid crystals can offer continuous variation of the response. Analog tuning and the ability to continuously cover a range of reflection/transmission amplitudes and phases gives maximum freedom in functionality. Digital control, on the other hand, permits to use simpler actuation elements and simplifies the corresponding driving circuits. In addition, the potential of digital control lies in that it can offer a systematic way of constructing quasi-continuous tunability by grouping more "bits" into "words" to construct more states. An interesting approach for obtaining quasi-continuous control via discrete states consists in clustering unit cells with a binary response in groups of two, three, or more (N), forming composite building blocks (which should still remain subwavelength in size) that can offer $2^{N}$ surface impedance states. A similar approach is based on embedding more diodes in each unit cell maintaining individual control over each of them. These approaches have been termed as coding or digital metasurfaces. $^{[128]}$ Given an adequate number of discrete states, practically any desired response can be encoded on the metasurface, by a proper spatial arrangement of the discrete states. This provides a powerful and intuitive design perspective, draws a parallelism with information theory, and opens new ways to model, compose, and design advanced metasurfaces.

The concept of coding metasurfaces was originally introduced with fixed, static designs that could not be tuned after fabrication. In this case, the discrete states are made by employing (or combining) unit cells with distinct geometry. $^{[129-132]}$ As shown in the comprehensive summary of Table 2, this first family of coding metasurfaces proved the validity of the concept for a wide set of cases. Most designs have been experimentally validated at microwave frequencies, $^{[129,132,133]}$ with some demonstrations also in the terahertz band. $^{[130,131,134]}$ Basic functionalities such as radar cross section (RCS) reduction, $^{[133]}$ focusing, $^{[135]}$ or anomalous reflection, $^{[136]}$ as well as more advanced functionalities such as vorticity control $^{[137]}$ and conversion $^{[138]}$ have been achieved with high performance while relying on a maximum of 8 different states (3-bit coding). The presented static designs also showcase how the coding approach enables the realization of complex behaviors. Coding patterns can be obtained analytically and then convoluted or “added” to other patterns to give rise to non-trivial effects such as steering of multiple beams, $^{[139]}$ simultaneous control of surface and space waves, $^{[140]}$ or the superposition of two desired functionalities. $^{[132]}$ This is indicated in the second column of Table 2, along with the optimization method (if any) used to specify the coding pattern that maximizes performance for the selected functionality. This is especially useful in the case of programmable/tunable coding metasurfaces, and will be discussed again later in this section.

Moving beyond static designs, the coding metasurface approach provides a natural match for realizing the programmable/reconfigurable metasurface paradigm in a systematic and scalable way. When built using locally tunable elements, each unit cell can be seen as one bit or word while the metasurface can be elegantly described as a state (bit/word) matrix which can be digitally controlled through reconfigurable logic, for example, a Field-Programmable Gate Array (FPGA). $^{[128]}$ This concept of tunable/programmable coding metasurfaces has been exemplified with several works in a variety of functionalities, as summarized in Table 3. Most papers make use of diodes for operation in the GHz range $^{[150]}$ ; however, demonstrations at higher frequencies are also appearing, such as in the mmWave band based on liquid crystals, $^{[151]}$ in the THz band based on graphene $^{[152-154]}$ and in the NIR based on indium tin oxide (ITO). $^{[155]}$ Note that the number of experimental works showcasing this approach is smaller than for static designs due to a number of manufacturing challenges. Experimental programmable prototypes are mostly focused in the microwave regime using diodes, as seen in Table 3. Embedding locally tunable actuators in the unit cells becomes more complex for higher frequencies, due to the granularity required and the space limitations dictated by the subwavelength nature of the unit cells. The approach relying on integrating diodes cannot be scaled well beyond mmWave frequencies, where liquid

Table 2. Summary of static coding metasurfaces in the literature, arranged by year of publication. The operating frequency and type of bit coding are indicated. Optimization techniques refer to the design of the coding pattern.

<table><tr><td>Function</td><td>Design approach</td><td>Frequency</td><td>Bit-coding</td><td>Year</td><td>Ref.</td></tr><tr><td>Low scattering</td><td>Hybrid</td><td>6–14 GHz</td><td>3-bit</td><td>2014</td><td>[129]</td></tr><tr><td>Low reflection</td><td>Particle swarm</td><td>1–2 THz</td><td>3-bit</td><td>2015</td><td>[130]</td></tr><tr><td>Wave diffusion</td><td>Particle swarm</td><td>0.8–1.7 THz</td><td>Multibit</td><td>2015</td><td>[131]</td></tr><tr><td>Beam splitting</td><td>FFT</td><td>10 GHz</td><td>1-bit</td><td>2016</td><td>[132]</td></tr><tr><td>Wave diffusion</td><td>Simulated annealing</td><td>4–8 GHz</td><td>1-bit</td><td>2016</td><td>[141]</td></tr><tr><td>RCS reduction</td><td>Genetic algorithm</td><td>4–18 GHz</td><td>1-bit</td><td>2016</td><td>[133]</td></tr><tr><td>Beam steering</td><td>Convolution</td><td>1 THz</td><td>2-bit</td><td>2016</td><td>[134]</td></tr><tr><td>Polarized wave manipulation</td><td>Anisotropic</td><td>1 THz</td><td>1 and 2-bit</td><td>2016</td><td>[142]</td></tr><tr><td>RCS reduction</td><td>Analytic</td><td>10 GHz</td><td>1-bit</td><td>2017</td><td>[143]</td></tr><tr><td>Focusing</td><td>Analytic</td><td>0.225–0.3 THz</td><td>3-bit</td><td>2017</td><td>[135]</td></tr><tr><td>Vorticity</td><td>Convolution</td><td>15 GHz</td><td>3-bit</td><td>2017</td><td>[137]</td></tr><tr><td>Vorticity</td><td>Analytic</td><td>15 GHz</td><td>2 and 3-bit</td><td>2017</td><td>[144]</td></tr><tr><td>Multibeam steering</td><td>Addition theorem</td><td>9 GHz</td><td>Multibit</td><td>2018</td><td>[139]</td></tr><tr><td>RCS reduction</td><td>Simulated annealing</td><td>6–20 GHz</td><td>1-bit</td><td>2018</td><td>[145]</td></tr><tr><td>RCS reduction</td><td>Genetic algorithm</td><td>2–20 GHz</td><td>1-bit</td><td>2018</td><td>[146]</td></tr><tr><td>Anomalous reflection</td><td>Analytic</td><td>10 GHz</td><td>1 and 2-bit</td><td>2018</td><td>[136]</td></tr><tr><td>Wave modulation</td><td>Genetic algorithm</td><td>8–12 GHz</td><td>3-bit</td><td>2018</td><td>[147]</td></tr><tr><td>Vorticity conversion</td><td>Analytic</td><td>16 GHz</td><td>3-bit</td><td>2019</td><td>[138]</td></tr><tr><td>RCS reduction</td><td>Analytic</td><td>12–30 GHz</td><td>1 and 2-bit</td><td>2019</td><td>[148]</td></tr><tr><td>Vorticity control</td><td>Analytic</td><td>9–15 GHz</td><td>2-bit</td><td>2019</td><td>[149]</td></tr><tr><td>Surface-space wave control</td><td>Analytic</td><td>10 GHz</td><td>2-bit</td><td>2019</td><td>[140]</td></tr></table>

crystals have been proposed instead. $^{[151]}$ In the THz band, graphene can enable the programmable approach, $^{[152-154]}$ but practical local biasing methods for micrometer-sized graphene strips remains a challenge. Even at microwave frequencies, metasurfaces employing PIN diodes need to take into account the parasitics introduced by the package and soldering processes, which change the effective circuit models. $^{[156]}$ Moreover, unit cell designs with PIN diodes may need to account for fabrication mismatches that lead to impedances deviating from the nominal value. $^{[122]}$ Another open issue with diode-based designs is that the switching speed of commercial diodes can be a limiting factor in certain applications. $^{[157]}$ Finally, these metasurfaces require of a large number of feeding lines, which affect the reliability of the system. This has led to alternative proposals that simplify the design by relying on mechanical means of tuning. $^{[158]}$

Table 3. Summary of programmable metasurface works in the literature, sorted by publication date. The operating frequency and the tuning material/element employed are dictated. Works with an experimental demonstration are marked with “Exp”. Optimization techniques refer to the design of the coding pattern.

<table><tr><td>Function</td><td>Design approach</td><td>Tuning mechanism</td><td>Frequency</td><td>Bit-coding</td><td>Year</td><td>Ref.</td></tr><tr><td>RCS reduction</td><td>Optimization</td><td>Pin-diode (Exp)</td><td>7–10 GHz</td><td>1 and 2-bit</td><td>2014</td><td>[128]</td></tr><tr><td>Beam shaping</td><td>Analytic</td><td>Pin-diode (Exp)</td><td>8–10 GHz</td><td>1-bit</td><td>2016</td><td>[156]</td></tr><tr><td>Multifunction</td><td>Genetic Algorithm, IDFT</td><td>Pin-diode (Exp)</td><td>9-12 GHz</td><td>1-bit</td><td>2016</td><td>[159]</td></tr><tr><td>Imaging</td><td>Analytic</td><td>Pin-diode (Exp)</td><td>9–10 GHz</td><td>2-bit</td><td>2016</td><td>[160]</td></tr><tr><td>Hologram</td><td>Gerchberg-Saxton</td><td>Pin-diode (Exp)</td><td>7.8 GHz</td><td>1-bit</td><td>2017</td><td>[122]</td></tr><tr><td>Harmonic beam steering</td><td>Binary particle swarm</td><td>Pin-diode (Exp)</td><td>8–11 GHz</td><td>Multibit</td><td>2018</td><td>[161]</td></tr><tr><td>Scattering control</td><td>Binary Bat</td><td>Graphene</td><td>1–1.5 THz</td><td>1-bit</td><td>2018</td><td>[152]</td></tr><tr><td>Focusing</td><td>Analytic</td><td>Graphene</td><td>2 THz</td><td>2-bit</td><td>2019</td><td>[153]</td></tr><tr><td>Vorticity control</td><td>Analytic</td><td>Graphene</td><td>2 THz</td><td>Multibit</td><td>2019</td><td>[154]</td></tr><tr><td>Nonreciprocal reflection</td><td>Analytic</td><td>Pin-diode (Exp)</td><td>10 GHz</td><td>2-bit</td><td>2019</td><td>[157]</td></tr><tr><td>Wavefront control</td><td>Analytic</td><td>Step motor (Exp)</td><td>4–6 GHz</td><td>1 and 2-bit</td><td>2019</td><td>[158]</td></tr><tr><td>Beam scanning</td><td>Analytic</td><td>Pin-diode</td><td>28 GHz</td><td>1-bit</td><td>2019</td><td>[150]</td></tr><tr><td>Beam steering</td><td>Analytic</td><td>Liquid crystal (Exp)</td><td>30 GHz</td><td>2-bit</td><td>2019</td><td>[151]</td></tr><tr><td>Beamforming</td><td>Machine learning</td><td>Pin-diode (Exp)</td><td>10 GHz</td><td>1-bit</td><td>2020</td><td>[162]</td></tr></table>

The design of a programmable coding metasurfaces has two main steps. First, a tunable unit cell is designed that can implement a discrete set of states, appropriate to the application at hand. Then, based on the available states, the coding pattern of the metasurface is developed, to which end several approaches have been used as seen in the second column of Table 3 (and Table 2). The required coding pattern can emerge from well-known physical principles and, in many cases, approximate solutions can be analytically formulated. As an example, consider the case (functionality) of a beam-steering gradient metasurface in reflection, implemented via a reconfigurable reflection-phase spatial profile. In this case, the coding set (i.e., the set of possible unit cell states) is built by picking points where the unit cell provides high reflection amplitude and reflection phases that are equally distributed in the $2\pi$ span, such as multiples of $2\pi / 2^{N_{\mathrm{bit}}}$ ( $N_{\mathrm{bit}}$ is the number of control bits). This coding set is used by the FPGA to implement the phase profile required in each instance, a linear phase gradient in this particular case. However, in more complex scenarios, closed-form expressions are not easy or possible to derive. Fortunately, for these cases the coding approach is very well suited to modeling and optimization techniques used in areas such as signal processing or complex systems. For instance, programmable metasurfaces that implement reconfigurable holograms have been proposed using the conventional Gerchberg-Saxton algorithm to obtain the coding pattern that leads to a particular holographic image.[122] Furthermore, genetic algorithms (GAs) work best in large but bounded parameter-spaces, and are thus well suited for few-bit, large-sized coding metasurfaces. Therefore, one can approach the coding pattern problem with such optimization techniques. As a result, GA,[133,146,163] but also particle-swarm,[130,131] simulated annealing,[141,145] or Binary Bat[152] algorithms have been used in multiple cases. Although most of these optimization techniques have been tested in static designs with non-tunable unit cells, the approach is directly applicable to programmable metasurfaces that need to be dynamically reconfigured in order to successfully switch between functionalities. Finally, machine learning methods coupled to optimization algorithms have been proposed not only to reduce the time devoted to design optimal unit cells,[164] but also to program the metasurfaces for complex functionalities, for example, beamforming,[162] real-time imaging,[165] or self-adaptive microwave cloaking.[166]

Another design tool that has been employed in coding metasurfaces is the physical information of the electric field distribution in the far field which can be approximately recovered by the (fast) Fourier transform (FFT) of the 2D metasurface code. $^{[132]}$ This process, exemplified in Figure 7, opens the door to different possibilities in the modeling and optimization of metasurfaces. For instance, in ref. [159], the authors use this feature to reduce the complexity of their optimization algorithm from $(MN)^{2}$ to $MN \times \log(MN)$ for a metasurface with $M \times N$ unit cells. Additionally, Cui et al. $^{[132]}$ define the entropy of the coding metasurface via conventional information theory methods, which is proven to effectively capture the complexity of the code required to achieve a given functionality. In this context, simple specular reflection is associated with zero-entropy coding, whereas random scattering requires random coding and it is thus associated with high entropy. In ref. [134], the multiplication of two spatial codes is shown to lead to the convolution of its scattering patterns. Wu et al. $^{[139]}$ relying on the linearity property of the Fourier transform demonstrated that the sum of two metasurface code matrices leads to the addition of its respective scattering patterns in phase-coded metasurfaces. Subsequently, this approach has been revisited for amplitude/phase-coded metasurfaces and exemplified for asymmetric power divider applications. $^{[167]}$ These perspectives paved the way to the fast encoding of multiple beams with arbitrary requirements $^{[134,139]}$ or multiple orthogonal functions within the same beam. $^{[144]}$ By similar analytical approaches, the scaling laws and design rules of RCS reduction metasurfaces have been derived. $^{[143]}$ Finally, a fault tolerance analysis has been performed in ref. [168], where a comprehensive error model is introduced and the impact of errors on the programming of the metasurfaces is evaluated.

![](images/10ecc3c260af56c5bf0fcbc4ece927e57bbe22bd2908fa7c42ec522f7b38912f.jpg)
Figure 7. A metasurface encoded with coding sequence 010101... for beam splitting applications. a) Full-wave numerical simulation results. b) Theoretical calculation results by the Fourier transform. c) The process to obtain the polar far-field pattern from the coding pattern of a metasurface using FFT. Reproduced with permission from ref. [132] under Creative Commons Attribution-NonCommercial-NoDerivs 4.0 International License (http://creativecommons.org/licenses/by-nc-nd/4.0/). Copyright 2016, The Authors, published by Springer Nature.

Beyond spatial coding which enables functionalities like wavefront manipulation, $^{[142,147,153,169]}$ angular momentum conversion, $^{[154]}$ and polarization control, $^{[137,144]}$ space-time-coding digital metasurfaces have been also proposed for harnessing the advantages of temporal modulation, $^{[161]}$ as illustrated in Figure 8. Applying temporal modulation on coding metasurfaces can generate new harmonics, acting as a frequency mixer. $^{[170]}$ In addition, it can be used to break reciprocity, thus providing an alternative to nonlinear metasurfaces for enabling such functionalities. More specifically, by periodically modulating the unit cells in time, higher and lower harmonic frequencies are generated, separated by a frequency spacing that is dictated by the frequency of modulation. $^{[170]}$ The modulation speed is limited by the underlying tuning mechanism and the response time of the system. By combining time modulation with space modulation more powerful control can be achieved, as frequency and wavefront control can be combined. For example, by carefully designing the space-time control signals, the generated harmonics can be directed to different scattering channels. $^{[161]}$ On the other hand, temporal modulation of the metasurface unit cells can break time-reversal symmetry and therefore realize nonreciprocal wave propagation. $^{[171-173]}$ For instance, an incident plane wave with angle $\theta_{1}$ and frequency $f_{1}$ can be anomalously reflected at angle $\theta_{2}$ and frequency $f_{2}$ by a space-time encoded metasurface. $^{[157]}$ Finally, by properly choosing the space-time modulation functions, classical nonreciprocal electromagnetic devices, for example, isolators, circulators, and phase shifters can be integrated on a single metasurface platform. $^{[174]}$

![](images/7295d24ac14fc3b5a7af5cc6c95a360e60f72512b5fcf6d8445e5755022ae62d.jpg)
Figure 8. Space-time-coding digital metasurface concept. The operation of the metasurface elements is controlled in the time domain with discrete phase or amplitude states. This coding method enables electromagnetic wave manipulations in both space and frequency domains. Reproduced from ref. [161] under Creative Commons Attribution 4.0 International License (http://creativecommons.org/licenses/by/4.0/). Copyright 2018, The Authors, published by Springer Nature.

## 5. Software-Defined Metasurfaces with Networked Controllers

The functionality and potential of reconfigurable and programmable metasurfaces can be extended even further by transforming the actuators in each unit cell (e.g., diodes in typical microwave implementations) into “controllers” with extra capabilities. These include the capability to sense the impinging wave, communicate with each other, and process information, thus forming a smart network of interconnected controllers. This intra-metasurface network can communicate with the outside world via a “gateway” which conforms with standard communication protocols (e.g., WiFi, Ethernet, Bluetooth) and can be accessed by a software application running on a computer, enabling efficient metasurface control and reconfiguration even by non-specialists. Moreover, multiple metasurfaces can be interfaced through the software application, forming an inter-metasurface network. Third-party devices can be incorporated in the inter-metasurface network, compatible to the emerging Internet-of-Things paradigm. $^{[175]}$ The concept outlined above is schematically illustrated in Figure 9.

Harnessing these extra capabilities can be a decisive step toward massive deployment of functional metasurfaces. The capability of distributing and exchanging commands between the gateway and the interconnected controllers can be utilized to make the metasurface resilient to faults, as information can be rerouted to avoid damaged actuators or reach the intended controllers even if a set of connections fail, as analyzed in ref. [176]. The capability of obtaining distributed sensory measurements, for example, current intensity over actuators, can allow to determine the impinging wave and, subsequently, configure the state of the actuators accordingly so as to achieve the desirable function. An example that determines the incidence direction of a plane wave by monitoring the absorbed power in the metasurface has been proposed in ref. [177].

![](images/23567bcf42214a54049bb1153f95406aec1228df37d84176da5699c8f7e5eddd.jpg)
Figure 9. Concept of software-defined metasurfaces with an embedded network of controllers that provide actuation, sensing, and information processing capabilities. Metasurfaces can communicate with a software application via a gateway and form cooperating inter-metasurface networks.

This new paradigm of programmable metasurfaces has been termed software-defined metasurfaces (SDMs). $^{[178]}$ The end vision is to bring forth an Internet-of-Metamaterials, with automated control over the electromagnetic behavior of materials. As an example of such possibilities, the idea of coating large objects in a setting (e.g., walls and ceilings in an office area) with SDMs has given rise to the concept of Programmable Wireless Environments (PWEs). $^{[179]}$ Within PWEs, the wireless channel is shaped and optimized via software to match exactly the physical attributes and communication objectives of users. $^{[180,181]}$ Mitigation of path loss, multipath fading, and Doppler effects has been reported. $^{[182]}$ In the communications community, the use of tunable metasurfaces, also referred to as reconfigurable intelligent surfaces (RIS), is actively discussed $^{[183,184]}$ for the control and optimization of the wireless propagation environment.

## 5.1. Network of Interconnected Controllers

The actual implementation of an SDM may vary considerably, depending on the desired operation frequency, the physical working principle of the actuators, the capabilities of the controllers, and their network. Regarding the controller network, the main distinctions are between wireless or wired implementation (copper interconnections or photonic/plasmonic waveguides for higher frequencies $^{[185]}$ ) and between synchronous or asynchronous communication within the network. $^{[186]}$

An example of a microwave metasurface designed for operation at 5 GHz comprising integrated controllers communicating with the aid of wireless transceivers is depicted in Figure 10. The wireless connectivity has the advantages of reduced network distance between controllers and support of natural data broadcast (opposed to the point-to-point nature of the wired approach), which helps data dissemination and the implementation of distributed processing tasks. The metasurface consists of a multilayer printed circuit board $^{[125,126]}$ and is designed to operate in reflection mode (Figure 10a). The controllers incorporate a wireless transceiver and an antenna and lie behind the metasurface ground plane so as to avoid interfering with the metasurface operation (Figure 10b). For the same reason, communication between the controllers is conducted with millimeter waves. Three communication scenarios are illustrated in Figure 10c–e; they are examined in detail in ref. [187] with respect to performance metrics of path loss, mean delay, and delay spread. More specifically, the communication channel can be confined in different layers to avoid excessive power attenuation by forming a dedicated parallel-plate waveguide layer (Figure 10d), or placing the antennas directly inside the metasurface layer (Figure 10e).

## 5.2. Gateway and Software Application Components

Software-defined metasurfaces exploit computer science and communications concepts to facilitate programmable operation and compatibility with existing networks, pushing metasurfaces into greater functionality and applicability. The gateway implements communication protocols connecting the intra- and intermetasurface “worlds” as follows: On one hand, it participates as a peer to the intra-metasurface network and therefore it can send and receive data to/from the controllers. On the other hand, it provides mainstream connectivity to the outside world by translating the data generated by the internal network of the SDM to common protocols (e.g., WiFi, Bluetooth). The software application is the component that completes the software-defined aspect of SDMs. $^{[188]}$ The main aim is to make the metasurface operation easy to integrate into applications and systems. To this end, the software application component implements software libraries that enable interaction with multiple SDMs in a physics-abstracting manner. As SDMs evolve, it is expected to incorporate optimization techniques, $^{[189]}$ heuristics, $^{[152]}$ and neural network approaches, $^{[184]}$ for determining the required settings of the actuators in order to switch between supported functionalities and execute them.

![](images/5dce3210d1d7b7c7902566ae8219cd3e8ca12fa4712c3aaffd8600aa0c2fb5f4.jpg)

![](images/c5d676157c658a7d563ee937d9ab250fe5e0eba36e82c9e1816de26355ff1ea5.jpg)

(d) Communication in Dedicated Layer
(e) Communication in Metasurface Layer
![](images/c548ede0a55878b79b3bba1d09af040f80cdff47a22bd9567adf22115e02af62.jpg)
Figure 10. Example of microwave programmable metasurface with a an embedded network of controllers. a) Schematic of metasurface operating at 5 GHz under oblique plane wave incidence. b) bottom-view of unit cell illustrating the controller for programmable operation. Unit cell side-views illustrating the communication channels for mmWave communication between controllers: c) Communication in the controller layer, d) communication in a dedicated parallel-plate waveguide, and e) communication inside the metasurface layer. Adapted under the terms of a Creative Commons Attribution CC-BY license from ref. [187]. Copyright 2019, The Authors, published by IEEE.

## 6. Conclusion

Metasurfaces have shown a great potential as functional artificial skins that can shape and control electromagnetic waves. To this end, metasurfaces need to depart from the early demonstrations of fixed, static electromagnetic response and enter the tunable, reconfigurable, and programmable regimes.

We have provided a comprehensive survey of the progress in tunable metasurfaces from basic globally tunable structures to fully reconfigurable structures, which realize local control over the amplitude, phase, and polarization of electromagnetic waves. We have thoroughly presented the very recent advances in the field of coding metasurfaces, in which discrete amplitude/phase states are used to construct a code that can implement (and switch between) different functionalities at will. Coding metasurfaces borrow concepts from information theory to achieve complex functionalities, are well suited to optimization techniques, and are naturally compatible with programmatic control. In addition, we have outlined the new research direction of software-defined metasurfaces which attempts to harness the extra capabilities offered by an integrated network of controllers pushing metasurfaces toward unprecedented levels of functionality and real-world applications. Several challenges arise in the implementation of SDMs, providing exciting opportunities that can drive research for years to come. These are indicatively: i) the careful design of the metasurface in order to accommodate the communication channel between the controllers and enable efficient and cost-effective communication while avoiding interference with the metasurface operation, and ii) the implementation of smart controllers that encompass the actuators along with transceivers for communication and sensors, while respecting the subwavelength unit cell dimensioning of metasurfaces that enables their broad functionality. The physical implementation largely depends on the frequency of operation; different tuning mechanisms, design strategies, and limitations become relevant for different frequency regimes.

As reconfigurable metasurfaces continue to mature and evolve, we envision that software-defined metasurfaces can have a transformational effect on a broad range of applications including communications, imaging, sensing, and wireless power transfer.

## Acknowledgements

This work was supported by the European Union's Horizon 2020 research and innovation programme-Future Emerging Topics (FETOPEN) under grant agreement No 736876 (VISORSURF). Financial support by the National Priorities Research Program grant No. NPRP9-383-1-083 from the Qatar National Research Fund is also acknowledged. O.T. acknowledges the financial support of the Stavros Niarchos Foundation within the framework of the project ARCHERS (“Advancing Young Researchers’ Human Capital in Cutting Edge Technologies in the Preservation of Cultural Heritage and the Tackling of Societal Challenges”).

## Conflict of Interest

The authors declare no conflict of interest.

## Keywords

electromagnetic wave control, functional sheet materials, metasurfaces, programmable, reconfigurable, software-defined metasurfaces, tunable

Received: May 12, 2020
Published online: July 8, 2020

[1] S. B. Glybovski, S. A. Tretyakov, P. A. Belov, Y. S. Kivshar, C. R. Simovski, Phys. Rep. 2016, 634, 1.

[2] C. M. Soukoulis, M. Wegener, Nat. Photonics 2011, 5, 523.

[3] V. Asadchy, A. Díaz-Rubio, D.-H. Kwon, S. Tretyakov, Surface Electromagnetics, With Applications in Antenna, Microwave, and Optical Engineering, Cambridge University Press, Cambridge, MA 2019, pp. 30–65.

[4] H.-T. Chen, A. J. Taylor, N. Yu, Rep. Prog. Phys. 2016, 79, 076401.

[5] D. Shrekenhamer, W.-C. Chen, W. J. Padilla, Phys. Rev. Lett. 2013, 110, 177403.

[6] N. Yu, P. Genevet, M. A. Kats, F. Aieta, J.-P. Tetienne, F. Capasso, Z. Gaburro, Science 2011, 334, 333.

[7] C. Pfeiffer, A. Grbic, Phys. Rev. Lett. 2013, 110, 197401.

[8] S. Kruk, B. Hopkins, I. I. Kravchenko, A. Miroshnichenko, D. N. Neshev, Y. S. Kivshar, APL Photonics 2016, 1, 030801.

[9] O. Tsilipakos, T. Koschny, C. M. Soukoulis, ACS Photonics 2018, 5, 1101.

[10] C. Qu, S. Ma, J. Hao, M. Qiu, X. Li, S. Xiao, Z. Miao, N. Dai, Q. He, S. Sun, L. Zhou, Phys. Rev. Lett. 2015, 115, 235503.

[11] B. Vasić, D. Zografopoulos, G. Isić, R. Beccherelli, R. Gajić, Nanotechnology 2017, 28, 124002.

[12] Y. Li, J. Lin, H. Guo, W. Sun, S. Xiao, L. Zhou, Adv. Opt. Mater. 2020, 8, 1901548.

[13] L. Cong, P. Pitchappa, C. Lee, R. Singh, Adv. Mater. 2017, 29, 1700733.

[14] I. V. Shadrivov, D. N. Neshev, World Scientific Handbook of Metamaterials and Plasmonics, World Scientific, Singapore 2017; Ch. 9, pp. 387–418.

[15] F. Liu, A. Pitilakis, M. S. Mirmoosa, O. Tsilipakos, X. Wang, A. C. Tasolamprou, S. Abadal, A. Cabellos-Aparicio, E. Alarcon, C. Liaskos, N. V. Kantartzis, M. Kafesaki, E. N. Economou, C. M. Soukoulis, S. Tretyakov, presented at IEEE Int. Symp. on Circuits and Systems, Florence, Italy, 2018, p. 8351817.

[16] Q. He, S. Sun, L. Zhou, Research 2019, 2019, 1849272.

[17] S. Sun, Q. He, J. Hao, S. Xiao, L. Zhou, Adv. Opt. Photonics 2019, 11, 380.

[18] O. Quevedo-Teruel, H. Chen, A. Díaz-Rubio, G. Gok, A. Grbic, G. Minatti, E. Martini, S. Maci, G. V. Eleftheriades, M. Chen, N. I. Zheludev, N. Papasimakis, S. Choudhury, Z. A. Kudyshev, S. Saha, H. Reddy, A. Boltasseva, V. M. Shalaev, A. V. Kildishev, D. Sievenpiper, C. Caloz, A. Alù, Q. He, L. Zhou, G. Valerio, E. Rajo-Iglesias, Z. Sipus, F. Mesa, R. Rodríguez-Berral, F. Medina, V. Asadchy, S. T. C. Craeye, J. Opt. 2019, 21, 073002.

[19] Q. He, S. Sun, S. Xiao, L. Zhou, Adv. Opt. Mater. 2018, 6, 1800415.

[20] A. E. Minovich, A. E. Miroshnichenko, A. Y. Bykov, T. V. Murzina, D. N. Neshev, Y. S. Kivshar, Laser Photonics Rev. 2015, 9, 195.

[21] N. M. Estakhri, A. Alù, J. Opt. Soc. Am. B 2016, 33, A21.

[22] L. R. Dalton, P. A. Sullivan, D. H. Bale, Chem. Rev. 2010, 110, 25.

[23] C.-Y. Lin, X. Wang, S. Chakravarty, B. Lee, W. Lai, J. Luo, A.-Y. Jen, R. Chen, Appl. Phys. Lett. 2010, 97, 093304.

[24] Y.-W. Huang, H. W. H. Lee, R. Sokhoyan, R. A. Pala, K. Thyagarajan, S. Han, D.-P. Tsai, H. A. Atwater, Nano Lett. 2016, 16, 5319.

[25] D. C. Zografopoulos, G. Sinatkas, E. Lotfi, L. A. Shahada, M. A. Swillam, E. E. Kriezis, R. Beccherelli, App. Phys. A 2018, 124, 105.

[26] J. Park, J.-H. Kang, X. Liu, M. Brongersma, Sci. Rep. 2015, 5, 15754.

[27] A. Anopchenko, L. Tao, C. Arndt, H. Lee, ACS Photonics 2018, 5, 2631.

[28] K. Shi, R. Haque, B. Zhao, R. Zhao, Z. Lu, Opt. Lett. 2014, 39, 4978.

[29] S. Savo, D. Shrekenhamer, W. J. Padilla, Adv. Opt. Mater. 2014, 2, 275.

[30] O. Buchnev, N. Podoliak, M. Kaczmarek, N. I. Zheludev, V. A. Fedotov, Adv. Opt. Mater. 2015, 3, 674.

[31] M. V. Gorkunov, A. E. Miroshnichenko, Y. S. Kivshar, Springer Ser. Math. Sci. 2015, 200, 237.

[32] N. I. Zheludev, Y. S. Kivshar, Nat. Mater. 2012, 11, 917.

[33] Q. Zhao, L. Kang, B. Du, B. Li, J. Zhou, H. Tang, X. Liang, B. Zhang, Appl. Phys. Lett. 2007, 90, 011112.

[34] F. Zhang, Q. Zhao, W. Zhang, J. Sun, J. Zhou, D. Lippens, Appl. Phys. Lett. 2010, 97, 134103.

[35] F. Zhang, W. Zhang, Q. Zhao, J. Sun, K. Qiu, J. Zhou, D. Lippens, Opt. Express 2011, 19, 1563.

[36] M. Decker, C. Kremers, A. Minovich, I. Staude, A. E. Miroshnichenko, D. Chigrin, D. N. Neshev, C. Jagadish, Y. S. Kivshar, Opt. Express 2013, 21, 8879.

[37] G. Isić, B. Vasić, D. Zografopoulos, R. Beccherelli, R. Gajić, Phys. Rev. Appl. 2015, 3, 064007.

[38] P. Kula, N. Bennis, P. Marć, P. Harmata, K. Gacioch, P. Morawiak, L. Jaroszewicz, Opt. Mater. 2016, 60, 209.

[39] L. Wang, S. Ge, W. Hu, M. Nakajima, Y. Lu, Opt. Mater. Express 2017, 7, 2023.

[40] G. Pawlik, K. Tarnowski, W. Walasik, A. C. Mitus, I. C. Khoo, Opt. Lett. 2012, 37, 1847.

[41] J. A. Bossard, X. Liang, L. Li, S. Yun, D. H. Werner, B. Weiner, T. Mayer, P. Cristman, A. Diaz, I. Khoo, IEEE Trans. Antennas Propag. 2008, 56, 1308.

[42] D. Franklin, Y. Chen, A. Vazquez-Guardado, S. Modak, J. Boroumand, D. Xu, S.-T. Wu, D. Chanda, Nat. Commun. 2015, 6, 7337.

[43] G. Pawlik, K. Tarnowski, W. Walasik, A. C. Mitus, I. C. Khoo, Opt. Lett. 2014, 39, 1744.

[44] K. S. Novoselov, A. K. Geim, S. V. Morozov, D. Jiang, M. I. Katsnelson, I. V. Grigorieva, S. V. Dubonos, A. A. Firsov, Nature 2005, 438, 197.

[45] A. Vakil, N. Engheta, Science 2011, 332, 1291.

[46] M. Tamagnone, A. Fallahi, J. R. Mosig, J. Perruisseau-Carrier, Nat. Photon. 2014, 8, 556.

[47] H. Cheng, S. Chen, P. Yu, J. Li, L. Deng, J. Tian, Opt. Lett. 2013, 38, 1567.

[48] S. Kim, M. S. Jang, V. W. Brar, K. W. Mauser, L. Kim, H. A. Atwater, Nano Lett. 2018, 18, 971.

[49] X. Wang, S. A. Tretyakov, IEEE Trans. Antennas Propag. 2018, 67, 2452.

[50] T.-T. Kim, S. S. Oh, H.-D. Kim, H. S. Park, O. Hess, B. Min, S. Zhang, Sci. Adv. 2017, 3, e1701377.

[51] S. Thongrattanasiri, F. H. Koppens, F. J. G. De Abajo, Phys. Rev. Lett. 2012, 108, 047401.

[52] Y. Fan, N.-H. Shen, T. Koschny, C. M. Soukoulis, ACS Photonics 2015, 2, 151.

[53] L. Ju, B. Geng, J. Horng, C. Girit, M. Martin, Z. Hao, H. A. Bechtel, X. Liang, A. Zettl, Y. R. Shen, F. Wang, Nat. Nanotechnol. 2011, 6, 630.

[54] P.-Y. Chen, A. Alu, IEEE Trans. Terahertz Sci. Technol. 2013, 3, 748.

[55] T. Guo, C. Argyropoulos, Opt. Lett. 2016, 41, 5592.

[56] H. Chen, W.-B. Lu, Z.-G. Liu, J. Zhang, A.-Q. Zhang, B. Wu, IEEE Trans. Microwave Theory Tech. 2018, 66, 3807.

[57] E. Galiffi, J. Pendry, P. Huidobro, ACS Nano 2018, 12, 1006.

[58] X. Chen, Z. Tian, Y. Lu, Y. Xu, X. Zhang, C. Ouyang, J. Gu, J. Han, W. Zhang, Adv. Opt. Mater 2020, 8, 1900660.

[59] N. Kakenov, O. Balci, T. Takan, V. Ozkan, H. Altan, C. Kocabas, ACS Photonics 2016, 3, 1531.

[60] F. J. Garcia de Abajo, ACS Photonics 2014, 1, 135.

[61] N. Dabidian, S. Dutta-Gupta, I. a. Kholmanov, K. Lai, F. Lu, J. Lee, M. Jin, S. Trendafilov, A. Khanikaev, B. Fallahazad, E. Tutuc, M. A. Belkin, G. Shvets, Nano Lett. 2016, 16, 3607.

[62] Z. Miao, Q. Wu, X. Li, Q. He, K. Ding, Z. An, Y. Zhang, L. Zhou, Phys. Rev. X 2015, 5, 041027.

[63] M. C. Sherrott, P. W. C. Hon, K. T. Fountaine, J. C. Garcia, S. M. Ponti, V. W. Brar, L. A. Sweatlock, H. A. Atwater, Nano Lett. 2017, 17, 3027.

[64] J. Georgiou, K. Kossifos, M. Antoniades, A. Jaafar, N. Kemp, presented at 2018 IEEE Int. Symp. on Circuits and Systems, IEEE, Florence, Italy 2018, pp. 1–5.

[65] K. Kossifos, M. Antoniades, J. Georgiou, A. Jaafar, N. Kemp, presented at 2018 IEEE Int. Symp. on Circuits and Systems, IEEE, Florence, Italy 2018, pp. 1–5.

[66] M.-X. Ren, W. Wu, W. Cai, B. Pi, X.-Z. Zhang, J.-J. Xu, Light Sci. Appl. 2017, 6, e16254.

[67] W. J. Padilla, A. J. Taylor, C. Highstrete, M. Lee, R. D. Averitt, Phys. Rev. Lett. 2006, 96, 107401.

[68] I. Chatzakis, L. Luo, J. Wang, N.-H. Shen, T. Koschny, J. Zhou, C. M. Soukoulis, Phys. Rev. B 2012, 86, 125110.

[69] H.-T. Chen, J. F. O'Hara, A. K. Azad, A. J. Taylor, R. D. Averitt, D. B. Shrekenhamer, W. J. Padilla, Nat. Photonics 2008, 2, 295.

[70] N.-H. Shen, M. Massaouti, M. Gokkavas, J.-M. Manceau, E. Ozbay, M. Kafesaki, T. Koschny, S. Tzortzakis, C. M. Soukoulis, Phys. Rev. Lett. 2011, 106, 037403.

[71] J. E. Heyes, W. Withayachumnankul, N. K. Grady, D. R. Chowdhury, A. K. Azad, H.-T. Chen, Appl. Phys. Lett. 2014, 105, 181108.

[72] J. Zhou, D. R. Chowdhury, R. Zhao, A. K. Azad, H.-T. Chen, C. M. Soukoulis, A. J. Taylor, J. F. O'Hara, Phys. Rev. B 2012, 86, 035448.

[73] G. Kenanakis, R. Zhao, N. Katsarakis, M. Kafesaki, C. M. Soukoulis, E. N. Economou, Opt. Express 2014, 22, 12149.

[74] A. C. Tasolamprou, A. D. Koulouklidis, C. Daskalaki, C. P. Mavidis, G. Kenanakis, G. Deligeorgis, Z. Viskadourakis, P. Kuzhir, S. Tzortzakis, M. Kafesaki, E. N. Economou, C. M. Soukoulis, ACS Photonics 2019, 6, 720.

[75] P. A. George, J. Strait, J. Dawlaty, S. Shivaraman, M. Chandrashekhar, F. Rana, M. G. Spencer, Nano Lett. 2008, 8, 4248.

[76] V. V. Popov, O. V. Polischuk, S. A. Nikitov, V. Ryzhii, T. Otsuji, M. S. Shur, J. Opt. 2013, 15, 114009.

[77] P.-Y. Chen, J. Jung, Phys. Rev. Appl. 2016, 5, 064018.

[78] Q. Ren, J. W. You, N. C. Panoiu, Phys. Rev. B 2019, 99, 205404.

[79] B. Jin, T. Guo, C. Argyropoulos, J. Opt. 2017, 19, 094005.

[80] T. Christopoulos, O. Tsilipakos, N. Grivas, E. E. Kriezis, Phys. Rev. E 2016, 94, 062219.

[81] T. Christopoulos, O. Tsilipakos, G. Sinatkas, E. E. Kriezis, Phys. Rev. B 2018, 98, 235421.

[82] Y. He, P. He, N. Sun, V. Harris, C. Vittoria, IEEE Trans. Magn. 2006, 42, 2852.

[83] G. Dewar, New J. Phys. 2005, 7, 161.

[84] Y. Yong-Jun, H. Yong-Jun, W. Guang-Jun, Z. Jing-Ping, S. Hai-Bin, O. Gordon, Chinese Phys. B 2012, 21, 038501.

[85] M. Lei, N. Feng, Q. Wang, Y. Hao, S. Huang, K. Bi, J. Appl. Phys. 2016, 119, 244504.

[86] J. Yu, H. Chen, Y. Wu, S. Liu, Europhys. Lett. 2012, 100, 47007.

[87] H. Wu, Q. Luo, H. Chen, Y. Han, X. Yu, S. Liu, Phys. Rev. A 2019, 99, 033820.

[88] A. E. Serebryannikov, A. Lakhtakia, E. Ozbay, Ann. Phys. 2017, 530, 1700252.

[89] S. Liu, Y. Long, C. Liu, Z. Chen, K. Song, Adv. Opt. Mater. 2017, 5, 1601043.

[90] S. Raoux, Annu. Rev. Mater. Res. 2009, 39, 25.

[91] T. Driscoll, S. Palit, M. M. Qazilbash, M. Brehm, F. Keilmann, B.-G. Chae, S.-J. Yun, H.-T. Kim, S. Y. Cho, N. M. Jokerst, D. R. Smith, D. N. Basov, Appl. Phys. Lett. 2008, 93, 024101.

[92] D. Wang, L. Zhang, Y. Gu, M. Q. Mehmood, Y. Gong, A. Srivastava, L. Jian, T. Venkatesan, C.-W. Qiu, M. Hong, Sci. Rep. 2015, 5, 15020.

[93] J.-H. Shin, K. Moon, E. S. Lee, I.-M. Lee, K. H. Park, Nanotechnology 2015, 26, 315203.

[94] N. A. Butakov, I. Valmianski, T. Lewi, C. Urban, Z. Ren, A. A. Mikhailovsky, S. D. Wilson, I. K. Schuller, J. A. Schuller, ACS Photonics 2018, 5, 371.

[95] Z. Shao, X. Cao, H. Luo, P. Jin, NPG Asia Mater. 2018, 10, 581.

[96] M. Rahmani, L. Xu, A. E. Miroshnichenko, A. Komar, R. Camacho-Morales, H. Chen, Y. Zárate, S. Kruk, G. Zhang, D. N. Neshev, Y. S. Kivshar, Adv. Funct. Mater. 2017, 27, 1700580.

[98] J. Y. Ou, E. Plum, L. Jiang, N. I. Zheludev, Nano Lett. 2011, 11, 2142.

[97] H.-T. Chen, H. Yang, R. Singh, J. F. O'Hara, A. K. Azad, S. A. Trugman, Q. X. Jia, A. J. Taylor, Phys. Rev. Lett. 2010, 105, 247402.

[99] T. Lewi, H. A. Evans, N. A. Butakov, J. A. Schuller, Nano Lett. 2017, 17, 3940.

[100] P. P. Iyer, R. A. DeCrescent, T. Lewi, N. Antonellis, J. A. Schuller, Phys. Rev. Appl. 2018, 10, 044029.

[101] B. Zhu, Y. Feng, J. Zhao, C. Huang, T. Jiang, Appl. Phys. Lett. 2010, 97, 051906.

[102] Z. Tao, X. Wan, B. C. Pan, T. J. Cui, Appl. Phys. Lett. 2017, 110, 121901.

[103] J. Zhao, Q. Cheng, J. Chen, M. Q. Qi, W. X. Jiang, T. J. Cui, New J. Phys. 2013, 15, 043049.

[104] C. Mias, J. H. Yap, IEEE Trans. Antennas Propag. 2007, 55, 1955.

[105] S. N. Burokur, J.-P. Daniel, P. Ratajczak, A. de Lustrac, Appl. Phys. Lett. 2010, 97, 064101.

[106] Z. Luo, J. Long, X. Chen, D. Sievenpiper, Appl. Phys. Lett. 2016, 109, 071107.

[107] H. K. Kim, D. Lee, S. Lim, Appl. Opt. 2016, 55, 4113.

[108] D. Chicherin, S. Dudorov, D. Lioubtchenko, V. Ovchinnikov, S. Tretyakov, A. V. Räisänen, Microwave Opt. Technol. Lett. 2006, 48, 2570.

[109] X. G. Zhang, W. X. Jiang, H. L. Jiang, Q. Wang, H. W. Tian, L. Bai, Z. J. Luo, S. Sun, Y. Luo, C.-W. Qiu, T. J. Cui, Nat. Electron. 2020, 3, 165.

[110] M. R. Shcherbakov, S. Liu, V. V. Zubyuk, A. Vaskin, P. P. Vabishchevich, G. Keeler, T. Pertsch, T. V. Dolgova, I. Staude, I. Brener, A. A. Fedyanin, Nat. Commun. 2017, 8, 17.

[111] B. Vasić, G. Isić, R. Beccherelli, D. C. Zografopoulos, IEEE J. Sel. Top. Quantum Electron. 2020, 26, 7701609.

[112] M. Maasch, M. Roig, C. Damm, R. Jakoby, IEEE Antennas Wirel. Propag. Lett. 2014, 13, 1581.

[113] E. Carrasco, M. Tamagnone, J. R. Mosig, T. Low, J. Perruisseau-Carrier, Nanotechnology 2015, 26, 134002.

[114] H. Cheng, S. Chen, P. Yu, W. Liu, Z. Li, J. Li, B. Xie, J. Tian, Adv. Opt. Mater. 2015, 3, 1744.

[115] H. Yang, T. Yu, Q. Wang, M. Lei, Sci. Rep. 2017, 7, 5441.

[116] I. V. Shadrivov, P. V. Kapitanova, S. I. Maslovski, Y. S. Kivshar, Phys. Rev. Lett. 2012, 109, 083902.

[117] D. F. Sievenpiper, J. H. Schaffner, H. J. Song, R. Y. Loo, G. Tangonan, IEEE Trans. Antennas Propag. 2003, 51, 2713.

[118] D. Chicherin, S. Dudorov, M. Sterner, J. Oberhammer, A. V. Raisanen, presented at 33rd Int. Conf. on Infrared, Millimeter and Terahertz Waves, Pasadena, CA 2008, pp. 1–3.

[119] O. Tsilipakos, A. C. Tasolamprou, T. Koschny, M. Kafesaki, E. N. Economou, C. M. Soukoulis, Adv. Opt. Mater. 2018, 6, 1800633.

[120] C. Huang, C. Zhang, J. Yang, B. Sun, B. Zhao, X. Luo, Adv. Opt. Mater. 2017, 5, 1700485.

[121] K. Chen, Y. Feng, F. Monticone, J. Zhao, B. Zhu, T. Jiang, L. Zhang, Y. Kim, X. Ding, S. Zhang, A. Alù, C.-W. Qiu, Adv. Mater. 2017, 29, 1606422.

[122] L. Li, T. Jun Cui, W. Ji, S. Liu, J. Ding, X. Wan, Y. B. Li, M. Jiang, C.-W. Qiu, S. Zhang, Nat. Commun. 2017, 8, 197.

[123] F. Liu, O. Tsilipakos, A. Pitilakis, A. C. Tasolamprou, M. S. Mirmoosa, N. V. Kantartzis, D.-H. Kwon, J. Georgiou, K. Kossifos, M. A. Antoniades, M. Kafesaki, C. M. Soukoulis, S. A. Tretyakov, Phys. Rev. Appl. 2019, 11, 044024.

[124] O. Tsilipakos, F. Liu, A. Pitilakis, A. C. Tasolamprou, D.-H. Kwon, M. S. Mirmoosa, N. V. Kantartzis, E. N. Economou, M. Kafesaki, C. M. Soukoulis, S. A. Tretyakov, presented at 12th Int. Congress on Artificial Materials for Novel Wave Phenomena, Espoo, Finland 2018, pp. 392–394.

[125] A. Pitilakis, O. Tsilipakos, F. Liu, K. M. Kossifos, A. C. Tasolamprou, D.-H. Kwon, M. S. Mirmoosa, D. Manessis, N. V. Kantartzis, C. Liaskos, M. A. Antoniades, J. Georgiou, C. M. Soukoulis, M. Kafesaki, S. A. Tretyakov, arXiv:2003.08654 [physics.app-ph], 2020.

[126] K. M. Kossifos, L. Petrou, G. Varnava, A. Pitilakis, O. Tsilipakos, F. Liu, P. Karousios, A. Tasolamprou, M. Seckel, D. Manessis, N. V. Kantartzis, D.-H. Kwon, M. A. Antoniades, J. Georgiou, IEEE Access 2020, 8, 92986.

[127] H. Taghvaee, S. Abadal, A. Pitilakis, O. Tsilipakos, A. Tasolamprou, C. K. Liaskos, M. Kafesaki, N. V. Kantartzis, A. Cabellos-Aparicio, E. Alarcón, IEEE Access 2020, 8, 105320.

[128] T. J. Cui, M. Q. Qi, X. Wan, J. Zhao, Q. Cheng, K. T. Lee, J. Y. Lee, S. Seo, L. J. Guo, Z. Zhang, Z. You, D. Chu, Light: Sci. Appl. 2014, 3, e218.

[129] K. Wang, J. Zhao, Q. Cheng, D. S. Dong, T. J. Cui, Sci. Rep. 2014, 4, 10.

[130] D. S. Dong, J. Yang, Q. Cheng, J. Zhao, L. H. Gao, S. J. Ma, S. Liu, H. B. Chen, Q. He, W. W. Liu, Z. Fang, L. Zhou, T. J. Cui, Adv. Opt. Mater. 2015, 3, 1405.

[131] L.-H. Gao, Q. Cheng, J. Yang, S.-J. Ma, J. Zhao, S. Liu, H.-B. Chen, Q. He, W.-X. Jiang, H.-F. Ma, Q.-Y. Wen, L.-J. Liang, B.-B. Jin, W.-W. Liu, L. Zhou, J.-Q. Yao, P.-H. Wu, T.-J. Cui, Light: Sci. Appl. 2015, 4, e324.

[132] T.-J. Cui, S. Liu, L.-L. Li, Light: Sci. Appl. 2016, 5, e16172.

[133] S. J. Li, X. Y. Cao, L. M. Xu, L. J. Zhou, H. H. Yang, J. F. Han, Z. Zhang, D. Zhang, X. Liu, C. Zhang, Y. J. Zheng, Y. Zhao, Sci. Rep. 2016, 5, 1.

[134] S. Liu, T. J. Cui, L. Zhang, Q. Xu, Q. Wang, X. Wan, J. Q. Gu, W. X. Tang, M. Qing Qi, J. G. Han, W. L. Zhang, X. Y. Zhou, Q. Cheng, Adv. Sci. 2016, 3, 1.

[135] H. Yi, S. W. Qu, B. J. Chen, X. Bai, K. B. Ng, C. H. Chan, Sci. Rep. 2017, 7, 2.

[136] S. Liu, T. J. Cui, A. Noor, Z. Tao, H. C. Zhang, D. B. Guo, Y. Yang, X. Y. Zhou, Light: Sci. Appl. 2018, 7, 18008.

[137] L. Zhang, S. Liu, L. Li, T. J. Cui, ACS Appl. Mater. Interfaces 2017, 9, 36447.

[138] G. Ding, K. Chen, X. Luo, J. Zhao, T. Jiang, Y. Feng, Phys. Rev. Appl. 2019, 11, 044043.

[139] R. Y. Wu, C. B. Shi, S. Liu, W. Wu, T. J. Cui, Adv. Opt. Mater. 2018, 6, 1701236.

[140] M. Wang, H. F. Ma, L. W. Wu, S. Sun, W. X. Tang, T. J. Cui, Adv. Opt. Mater. 2019, 7, 1900478.

[141] Y. Zhao, X. Cao, J. Gao, Y. Sun, H. Yang, X. Liu, Y. Zhou, T. Han, W. Chen, Sci. Rep. 2016, 6, 1.

[142] S. Liu, T. J. Cui, Q. Xu, D. Bao, L. Du, X. Wan, W. X. Tang, C. Ouyang, X. Y. Zhou, H. Yuan, H. F. Ma, W. X. Jiang, J. Han, W. Zhang, Q. Cheng, Light: Sci. Appl. 2016, 5, e16076.

[143] M. Moccia, S. Liu, R. Y. Wu, G. Castaldi, A. Andreone, T. J. Cui, V. Galdi, Adv. Opt. Mater. 2017, 5, 1700455.

[144] Q. Ma, C. B. Shi, G. D. Bai, T. Y. Chen, A. Noor, T. J. Cui, Adv. Opt. Mater. 2017, 5, 1.

[145] L. Jidi, X. Cao, Y. Tang, S. Wang, Y. Zhao, X. Zhu, Radioengineering 2018, 27, 394.

[146] S. Sui, H. Ma, J. Wang, Y. Pang, M. Feng, Z. Xu, S. Qu, J. Phys. D: Appl. Phys. 2018, 51, 6.

[147] Y. Zhou, G. Zhang, H. Chen, P. Zhou, X. Wang, L. Zhang, L. Zhang, Sci. Rep. 2018, 8, 8672.

[148] Q. Zheng, Y. Li, Y. Pang, J. Wang, H. Chen, S. Qu, M. Feng, J. Zhang, Opt. Commun. 2019, 430, 189.

[149] S. Iqbal, H. A. Madni, S. Liu, L. Zhang, T. J. Cui, Mater. Res. Express 2019, 6, 125804.

[150] Q. Xiao, Y. Z. Zhang, S. Iqbal, X. Wan, T. J. Cui, presented at Proc. of the ISAP '19, EAAP, Graz, AT 2019.

[151] Q. Wang, X. G. Zhang, H. W. Tian, W. X. Jiang, D. Bao, H. L. Jiang, Z. J. Luo, L. T. Wu, T. J. Cui, Adv. Theory Simul. 2019, 2, 1900141.

[152] A. Momeni, K. Rouhi, H. Rajabalipanah, A. Abdolali, Sci. Rep. 2018, 8, 6200.

[153] S. E. Hosseininejad, K. Rouhi, M. Neshat, R. Faraji-Dana, A. Cabellos-Aparicio, S. Abadal, E. Alarcón, Sci. Rep. 2019, 9, 2868.

[154] K. Rouhi, H. Rajabalipanah, A. Abdolali, Carbon 2019, 149, 125.

[155] R. Sabri, A. Forouzmand, H. Mosallaei, Opt. Express 2020, 28, 3464.

[156] X. Wan, M. Q. Qi, T. Y. Chen, T. J. Cui, Sci. Rep. 2016, 6, 20663.

[157] L. Zhang, X. Q. Chen, R. W. Shao, J. Y. Dai, Q. Cheng, G. Castaldi, V. Galdi, T. J. Cui, Adv. Mater. 2019, 31, 1904069.

[158] S. Liu, L. Zhang, G. D. Bai, T. J. Cui, Sci. Rep. 2019, 9, 1809.

[159] H. Yang, X. Cao, F. Yang, J. Gao, S. Xu, M. Li, Sci. Rep. 2016, 6, 35692.

[160] Y. B. Li, L. L. Li, B. B. Xu, W. Wu, R. Y. Wu, X. Wan, Q. Cheng, T. J. Cui, Sci. Rep. 2016, 6, 23731.

[161] L. Zhang, X. Q. Chen, S. Liu, Q. Zhang, J. Zhao, J. Y. Dai, G. D. Bai, X. Wan, Q. Cheng, G. Castaldi, V. Galdi, T. J. Cui, Nat. Commun. 2018, 9, 4334.

[162] T. Shan, M. Li, presented at Proc. of the AP-S/URSI '19, Atlanta, GA 2020.

[163] S. Sui, H. Ma, Y. Lv, J. Wang, Z. Li, J. Zhang, Z. Xu, S. Qu, Opt. Express 2018, 26, 1443.

[164] Q. Zhang, C. Liu, X. Wan, L. Zhang, S. Liu, Y. Yang, T. J. Cui, Adv. Theory Simul. 2018, 2, 1800132.

[165] L. Li, H. Ruan, C. Liu, Y. Li, Y. Shuang, A. Alù, C. W. Qiu, T. J. Cui, Nat. Commun. 2019, 10, 1082.

[166] C. Qian, B. Zheng, Y. Shen, L. Jing, E. Li, L. Shen, H. Chen, Nat. Photonics 2020, 14, 383.

[167] H. Rajabalipanah, A. Abdolali, J. Shabanpour, A. Momeni, arXiv:1901.04063 [physics.app-ph], 2019.

[168] H. Taghvaee, A. Cabellos-Aparicio, J. Georgiou, S. Abadal, IEEE J. Em. Sel. Top. C 2020, 10, 62.

[169] C. Huang, B. Sun, W. Pan, J. Cui, X. Wu, X. Luo, Sci. Rep. 2017, 7, 42302.

[170] M. M. Salary, S. Farazi, H. Mosallaei, Adv. Opt. Mater. 2019, 7, 1900843.

[171] D. L. Sounas, A. Alù, Nat. Photonics 2017, 11, 774.

[172] C. Caloz, A. Alù, S. Tretyakov, D. Sounas, K. Achouri, Z.-L. Deck-Léger, Phys. Rev. Appl. 2018, 10, 047001.

[173] V. Asadchy, M. S. Mirmoosa, A. Díaz-Rubio, S. Fan, S. A. Tretyakov, arXiv:2001.04848 [physics.app-ph], 2020.

[174] X. Wang, A. Díaz-Rubio, H. Li, S. A. Tretyakov, A. Alù, Phys. Rev. Appl. 2020, 13, 044040.

[175] A. Mihovska, M. Sarkar, New Advances in the Internet of Things, Springer, New York 2018, pp. 105–118.

[176] D. Kouzapas, C. Skitsas, T. Saeed, V. Soteriou, M. Lestas, A. Philippou, S. Abadal, C. Liaskos, L. Petrou, J. Georgiou, A. Pitsillides, J. Syst. Architect. 2020, 106, 101703.

[177] C. Liaskos, G. Pyrialakos, A. Pitilakis, S. Abadal, A. Tsioliaridou, A. Tasolamprou, O. Tsilipakos, N. Kantartzis, S. Ioannidis, E. Alarcon, A. Cabellos, M. Kafesaki, A. Pitsillides, K. Kossifos, J. Georgiou, I. F. Akyildiz, arXiv:1907.04811 [eess.SP], 2019.

[178] C. Liaskos, A. Tsioliaridou, A. Pitsillides, I. F. Akyildiz, N. V. Kantartzis, A. X. Lalas, X. Dimitropoulos, S. Ioannidis, M. Kafesaki, C. M. Soukoulis, IEEE Trans. Circuits Syst. 2015, 15, 12.

[179] C. Liaskos, A. Tsioliaridou, A. Pitsillides, S. Ioannidis, I. Akyildiz, Commun. ACM 2018, 61, 30.

[180] C. Liaskos, S. Nie, A. Tsioliaridou, A. Pitsillides, S. Ioannidis, I. Akyildiz, IEEE Commun. Mag. 2018, 56, 162.

[181] R. Mehrotra, R. I. Ansari, A. Pitilakis, C. Liaskos, N. Kantartzis, A. Pitsillides, presented at Proc. of IEEE SPECTS, IEEE, Abu Dhabi 2019.

[182] C. Liaskos, A. Tsioliaridou, S. Nie, A. Pitsillides, S. Ioannidis, I. F. Akyildiz, IEEE/ACM Transactions on Networking 2019, 27, 1696.

[183] E. Basar, M. D. Renzo, J. D. Rosny, M. Debbah, M.-S. Alouini, R. Zhang, IEEE Access 2019, 7, 116753.

[184] M. Di Renzo, M. Debbah, D.-T. Phan-Huy, A. Zappone, M.-S. Alouini, C. Yuen, V. Sciancalepore, G. C. Alexandropoulos, J. Hoydis, H. Gacanin, J. de Rosny, A. Bounceur, G. Lerosey, M. Fink, EURASIP J. Wirel. Comm. 2019, 2019, 129.

[185] S. Papaioannou, K. Vyrsokinos, D. Kalavrouziotis, G. Giannoulis, D. Apostolopoulos, H. Avramopoulos, F. Zacharatos, K. Hassan, J.-C. Weeber, L. Markey, A. Dereux, A. Kumar, S. I. Bozhevolnyi, A. Suna, O. G. de Villasante, T. Tekin, M. Waldow, O. Tsilipakos, A. Pitilakis, E. E. Kriezis, N. Pleros, in Merging Plasmonics and Silicon Photonics Towards Greener and Faster “Network-on-Chip” Solutions for Data Centers and High-Performance Computing Systems (Ed: K. Y. Kim), IntechOpen, London, UK 2012.

[186] L. Petrou, P. Karousios, J. Georgiou, presented at Proc. of the ISCAS '18, Florence, Italy, 2018.

[187] A. C. Tasolamprou, A. Pitilakis, S. Abadal, O. Tsilipakos, X. Timoneda, H. Taghvaee, M. S. Mirmoosa, F. Liu, C. Liaskos, A. Tsioliaridou, S. Ioannidis, N. V. Kantartzis, D. Manessis, J. Georgiou, A. Cabellos-Aparicio, E. Alarcon, A. Pitsillides, I. Akyildiz, S. A. Tretyakov, E. N. Economou, M. Kafesaki, C. M. Soukoulis, IEEE Access 2019, 7, 122931.

[188] C. Liaskos, G. G. Pyrialakos, A. Pitilakis, S. Abadal, A. Tsioliaridou, A. C. Tasolamprou, O. Tsilipakos, N. V. Kantartzis, S. Ioannidis, E. Alarcón, A. Cabellos-Aparicio, M. Kafesaki, A. Pitsillides, K. M. Kossifos, J. Georgiou, I. F. Akyildiz, Initial UML definition of the HyperSurface Compiler middle-ware. European Commission Project VISORSURF: Accepted Public Deliverable D2.2, 31-Dec-2017, http://www.visorsurf.eu/m/VISORSURF-D2.2.pdf (accessed: December 2017).

[189] L.-S. Shu, S.-J. Ho, S.-Y. Ho, Intelligent Data Engineering and Automated Learning: Lecture Notes in Computer Science, Springer, New York 2003.
