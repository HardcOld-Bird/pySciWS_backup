REVIEW

# Non-local and non-Hermitian acoustic metasurfaces

To cite this article: Xu Wang et al 2023 Rep. Prog. Phys. 86 116501

View the article online for updates and enhancements.

## You may also like

\- Ultrathin broadband acoustic reflection metasurface based on meta-molecule clusters
Y B Wang, C R Luo, Y B Dong et al.

\- Active optical metasurfaces: comprehensive review on physics, mechanisms, and prospective applications
Jingyi Yang, Sudip Gurung, Subhajit Bej et al.

\- Localization and amplification of Rayleigh waves by topological elastic metasurfaces
Weijian Zhou and Zheng Fan

Review

# Non-local and non-Hermitian acoustic metasurfaces

Xu Wang $^{1,3}$ , Ruizhi Dong $^{1,3}$ , Yong Li $^{1,*}$ and Yun Jing $^{2,*}$

$^{1}$ Institute of Acoustics, School of Physics Science and Engineering, Tongji University, Shanghai 200092, People's Republic of China

$^{2}$ Graduate Program in Acoustics, The Pennsylvania State University, University Park, PA 16802, United States of America

E-mail: yongli@tongji.edu.cn and yqj5201@psu.edu

Received 30 March 2023, revised 3 August 2023
Accepted for publication 21 September 2023
Published 10 October 2023

Corresponding editor: Dr Masud Mansuripur

![](images/099f33dca4cb8994f0e0bc7c417dca86f85b4d9e6a6a4a29e5c78aa88afc7d1c.jpg)

## Abstract

Acoustic metasurfaces are at the frontier of acoustic functional material research owing to their advanced capabilities of wave manipulation at an acoustically vanishing size. Despite significant progress in the last decade, conventional acoustic metasurfaces are still fundamentally limited by their underlying physics and design principles. First, conventional metasurfaces assume that unit cells are decoupled and therefore treat them individually during the design process. Owing to diffraction, however, the non-locality of the wave field could strongly affect the efficiency and even alter the behavior of acoustic metasurfaces. Additionally, conventional acoustic metasurfaces operate by modulating the phase and are typically treated as lossless systems. Due to the narrow regions in acoustic metasurfaces' subwavelength unit cells, however, losses are naturally present and could compromise the performance of acoustic metasurfaces. While the conventional wisdom is to minimize these effects, a counter-intuitive way of thinking has emerged, which is to harness the non-locality as well as loss for enhanced acoustic metasurface functionality. This has led to a new generation of acoustic metasurface design paradigm that is empowered by non-locality and non-Hermicity, providing new routes for controlling sound using the acoustic version of 2D materials. This review details the progress of non-local and non-Hermitian acoustic metasurfaces, providing an overview of the recent acoustic metasurface designs and discussing the critical role of non-locality and loss in acoustic metasurfaces. We further outline the synergy between non-locality and non-Hermiticity, and delineate the potential of using non-local and non-Hermitian acoustic metasurfaces as a new platform for investigating exceptional points, the hallmark of non-Hermitian physics. Finally, the current challenges and future outlook for this burgeoning field are discussed.

Keywords: acoustic metasurfaces, metagratings, non-locality, non-Hermiticity, topological systems, exceptional points, asymmetric response

(Some figures may appear in colour only in the online journal)

## 1. Introduction

Sound, along with light, is a classical wave that plays a crucial role in our perception of the world. As a result, the ability to manipulate sound at will has long been a sought-after goal. For centuries, various natural materials have been utilized to control the propagation of sound. For those natural materials, the underlying wave-matter interactions usually become prominent only when the material thickness is comparable to the operating wavelength. Thus, controlling sound presents a challenge due to its characteristics in wavelength. To better understand this, we can compare audible sound with visible light. The wavelength of visible light is typically hundreds of nanometers, whereas for a typical audible sound at 1 kHz, it is measured in decimeters. This dramatic difference in wavelength means that materials that shield us from the Sun can be manifested as a film, while materials that offer noise-proofing require a wall. Additionally, the visible light spectrum (ranging from 400 nm to 760 nm) is relatively narrow, covering less than an octave. In contrast, audible sound (ranging from 20 Hz to 20 kHz) spans up to ten octaves, with wavelengths ranging from decameters to centimeters. Consequently, it is inherently challenging to develop a mechanism for consistent sound control across such a wide range. This has led to acoustics, the study of the propagation of sound and vibrational waves, becoming an old yet still fast-growing discipline in physics.

For sound, a general picture of wave-matter interaction entails sound striking a surface, where transmission, absorption, and reflection occur. The coefficients relevant to this process (transmittance, absorptance, and reflectance) indicate the amount of energy going into transmission, absorption, or reflection, and are heavily dependent on the surface's acoustic properties. These metrics illustrate the effectiveness of sound absorption (by high absorptance) and/or sound insulation (by high reflectance and low transmittance), which are the most crucial means in noise control. However, these typical wave-matter interactions rely solely on the manipulation of the amplitude of sound. Conversely, the manipulation of phase is a less commonly used technique in traditional acoustics for controlling sound.

The utility of phase control in sound can be exemplified by the use of an acoustic diffusor. Sound can be reflected in two different ways, specular reflection by a large flat surface or scattering by a rough or non-flat surface. Diffuse reflection, a special type of scattering where the acoustic energy is scattered in a uniform manner, plays a vital role in room acoustics. In the 1970s, Schroeder proposed an acoustic diffusor that is a period structure, where each period comprises a number of wells (grooves) with carefully chosen depths [1, 2]. When sound is reflected by the diffusor, the local phase shifts provided by the wells collectively change the wavefront of the reflection in a predictable and desired manner and thus producing the so-called optimum diffuse reflection. While the idea of Schroeder diffusers does not originate from the quantum mechanical band theory of solids, they essentially function as 'reflection phase gratings' that scatter sound in a manner similar to how crystal lattices scatter electromagnetic waves. In this sense, Schroeder diffusers can be considered as an archetype in the rapidly growing fields of acoustic metamaterials and metasurfaces (which we will elaborate on), due to their intriguing capability of manipulating the phase of sound and its periodic nature.

Although the manipulation of sound is an endeavor that dates back centuries, the use of natural materials has been extremely limited in its effectiveness. However, around three decades ago, inspired by the quantum mechanical band theory of solids in which electronic waves interact with a lattice of periodically arranged atomics to form energy bands separated by bandgaps, man-made crystals for electromagnetic and acoustics waves, i.e. photonic $[3]$ and phononic $[4]$ crystals, were proposed to control light and sound utilizing scattering from rationally designed periodic structures. However, the lattice constants of these artificial crystals must be on the order of the wavelength being manipulated. As a result, phononic crystals designed for the audible regime remain too bulky due to the large wavelength of acoustic waves $[5]$ .

The emergence of acoustic metamaterials addressed the sample size issue $[6]$ . Local resonance can greatly enhance the wave-matter interaction, allowing for much smaller materials to control sound waves with large wavelengths. Metamaterials are a specific category of artificial materials whose wave functionalities arise from the combined effects of their locally resonant constituent units (meta-units). These macroscopic composites possess a three-dimensional (3D) cellular architecture, often period in nature (though not required), that is designed to produce an optimized response to a specific excitation. Acoustic metamaterials commonly incorporate sub-wavelength meta-units, resulting in a significantly smaller material size than phononic crystals.

The capabilities of metamaterials to control waves at the macroscopic scale arise from the methodologically designed and arranged meta-units with a mesoscopic size. Since the meta-units are typically in a deeply subwavelength scale, the overall structure can be considered an effective medium possessing homogeneous material properties. In this way, the acoustic metamaterial (designed for sound in gas and fluid) behaves like a continuous (homogeneous) material that can be primarily characterized by two acoustical properties: the mass density $\rho$ and the bulk modulus $\kappa$ . Furthermore, in wave theory, the characteristics of a medium can be described by its effective wave speed. The speed of sound and light are respectively given by $\sqrt{\kappa/\rho}$ and $\sqrt{1/\epsilon\mu}$ , and there is a mapping of $\rho\to\epsilon$ and $\kappa\to\mu^{-1}$ , where $\epsilon$ and $\mu$ are the dielectric constant and magnetic permeability [7]. In an acoustic metamaterial, the two constitutive parameters $(\rho,\kappa)$ can take unusual values (for example, negative, zero, close to divergent, or strongly anisotropic). The related intriguing phenomena include negative refraction [8, 9], subwavelength imaging [10], and cloaking [11, 12], which cannot be attained by natural materials. For a detailed review on acoustic metamaterials, the readers are referred to [7, 13–17].

Despite the achievements of acoustic metamaterials, they remain bulky materials that typically possess periodicity in 3D. As a result, tremendous efforts have been made to manipulate waves using only one or a few layers of man-made composites, resulting in the emergence of two-dimensional (2D) metamaterials known as metasurfaces. These structures are particularly appealing due to their significantly reduced size (thickness) compared to metamaterials. The first internationally recognized metasurface is known as the gradient-phase (or gradient-index) metasurface (GPM), which is a periodic flat surface providing linearly-varying local phase shift to the incident wave to redirect the reflection (refraction) in a way beyond what is allowed by the traditional law of reflection (refraction) [18]. Although proposed long before the term 'metasurfaces' was coined, the aforementioned Schroeder diffusors share many similarities with GPMs. For example, they are both acoustic engineered surfaces functioned as phase gratings. However, they are also designed based on vastly different principles: a 'randomly' arranged local phase modulation (for Schroder diffusors) causes a dispersed wave field, whereas an orderly arranged one (for GPMs) leads to a directional wave field. Over the last decade, acoustic metasurfaces have emerged as a highly researched area, capturing significant attention due to their remarkable ability to manipulate wave-matter interactions at the subwavelength level [19]. These metasurfaces now come in various designs, tailored to specific application scenarios, and are being increasingly explored in diverse fields. Among the wide range of applications, acoustic metasurfaces have demonstrated exceptional performance in anomalous transmission and reflection [20–22], hologram rendering [23, 24], acoustic absorption [25, 26], and isolation [27] for effective noise reduction [16]. They have also shown promise in real-time acoustic communication using vortex beams [28, 29], beam focusing and directional transmission for high-precision particle manipulation [30], sound diffusing techniques for manipulating scattering fields [31], as well as controlling reverberating sound in architectural acoustics [32]. Detailed reviews for classical acoustic metasurfaces can be found in [15, 17, 19].

In the following sections, we will introduce the recent advances in acoustic metasurfaces, covering works related to sound waves in air (gases), liquids, and solid bodies (including mechanical vibrations). To make this article accessible to a wider scientific community, we have structured and written the paper in a manner that can be understood not only by specialists in metasurfaces, but also those who are new to this field and may not have expert knowledge of the subject. To this end, we begin by introducing the basic principles of traditional metasurfaces (i.e. GPMs) in section 2. We then move on to discuss their limitations and the efforts to address these issues, which have led to the development of non-local and non-Hermitian metasurfaces. These novel structures embrace the previously perceived limitations and offer brand-new functionalities to acoustic metasurfaces. We have also included topics that may not directly fall into the category of non-local and non-Hermitian acoustic metasurfaces, but are necessary to cover in order to provide fundamental background knowledge highly relevant to the past, present, and future of non-local and non-Hermitian acoustic metasurfaces. These topics include bianisotropic acoustic metasurfaces, power-flow-conformal acoustic metasurfaces, non-Hermitian acoustics in zero-dimensional (0D) and one-dimensional (1D) systems, and non-Hermitian acoustics in topological systems.

## 2. Traditional acoustic metasurfaces

To provide a self-contained review on a topic related to acoustic metasurfaces, we start with the basic physics underpinning the traditional acoustic metasurface, which also provides fundamental knowledge needed to understand non-local and non-Hermitian acoustic metasurfaces. Classic wave theory informs that the reflection (refraction) of plane waves impinging upon a planar boundary separating two media obey the law of reflection (refraction). These physical laws predict equal angles of incidence and reflection, as well as the relationship between the angles of incidence and refraction depending on the difference between the refractive indices of the two media. The concept of metasurfaces generalize these physical laws, leading to reflection (refraction) by a flat interface that can be redirected in a controllable manner $[18]$ . To provide an intuitive picture of the theory, we take reflection as an example by discussing reflection-type acoustic metasurfaces, which can be readily extended to refraction following the same principle.

An incident plane wave is directed towards a flat interface located on the z=0 plane (figure 1). Consider the surface to be no longer homogenous so that sound reflected at two adjacent points $A(x,0)$ and $C(x+\mathrm{d}x,0)$ on the surface undergo different phase shifts, denoted as $\varphi$ and $\varphi+d\varphi$ , respectively. A wavefront is a plane consisting of points that move in phase. Thus, between the incident and reflected wavefronts (indicated as AB and $B^{\prime}C$ ) as shown in figure 1, the phase accumulated along these two nearby paths must be identical, so that

$$
k \overline {{{A B ^ {\prime}}}} + \varphi = k \overline {{{B C}}} + \varphi + \mathrm{d} \varphi ,\tag{1}
$$

where k is the wavenumber of sound in the background medium. Using simple geometrical relationship yields

$$
k \sin (\theta_ {r}) d x + \varphi = k \sin (\theta_ {i}) d x + \varphi + d \varphi .\tag{2}
$$

Equation (2) can be simplified as

$$
k \left[ \sin \left(\theta_ {r}\right) - \sin \left(\theta_ {i}\right) \right] = \frac {\mathrm{d} \varphi}{\mathrm{d} x}.\tag{3}
$$

This equation governs the relationship between the angles of incidence and reflection, which is determined by the gradient of phase shift along the surface. This is well known as the generalized law of reflection [18]. For a homogenous surface (i.e. $d\varphi/dx=0$ ), equation (3) reduces to a simple form, i.e. $\theta_{r}=\theta_{i}$ , indicating a specular reflection. Crucially, equation (3) suggests that the reflection from a flat surface can be arbitrarily tuned, and such an anomalous reflection arises from the inhomogeneity of the surface featuring a linearly varying profile, which constitutes the simplest form of metasurfaces. The meta-units of the metasurface are the key to realizing the flexible control of reflection (refraction), as they can be designed to provide the desired gradient phase shifts (covering a $2\pi$ shift in a period). Separating the derivatives (dφ and dx) and integrating both sides of equation (3) further yield

![](images/3282e95ab7f1d24b0751b79475be971b4213952a91c5fa571efdc4182d711c8d.jpg)
Figure 1. Schematic diagram of sound reflection and refraction occurring at an inhomogeneous surface.

$$
\varphi (x) = k \delta x + C,\tag{4}
$$

where $\varphi(x)$ is the phase-shift profile along the surface, $\delta = \sin(\theta_{r}) - \sin(\theta_{i})$ , and $C \in R$ is an integral constant. Equations (3) and (4) suggest that a desired anomalous reflection (i.e. a desired $\delta$ ) hinges on the gradient of phase shift imparted by the metasurface, rather than the phase shift itself, since C can be arbitrarily chosen to give rise to multiple different $\varphi(x)$ given a fixed $\delta$ .

This early version of the generalized law of reflection works in most cases for redirecting the wave. However, under some special conditions (especially with large incident angles), sound will be redirected to a direction that cannot be predicted by equation (3) [33]. To understand this, let us consider another intrinsic property of metasurfaces: periodicity. Now consider the two points $A$ and $C$ in figure 1 not being adjacent but rather spaced with a distance that is exactly one period ( $D$ ). Consequently, equation (3) should be rewritten as

$$
k D \left[ \sin \left(\theta_ {r}\right) - \sin \left(\theta_ {i}\right) \right] = D \frac {\mathrm{d} \varphi}{\mathrm{d} x} + 2 m \pi ,\tag{5}
$$

where $m \in N$ . Equation (5) is obtained by multiplying D on both sides of equation (3), while the additional term $(2m\pi)$ indicates that reflections from A and B should be in phase. Equation (5) can be further simplified by dividing D on both sides, yielding [34, 35]

$$
k \left[ \sin \left(\theta_ {r}\right) - \sin \left(\theta_ {i}\right) \right] = \frac {\mathrm{d} \varphi}{\mathrm{d} x} + m G,\tag{6}
$$

where $G = \frac{2\pi}{D}$ is the amplitude of reciprocal lattice vector for such a periodic structure. Notice that the phase shift $\varphi$ provided by a GPM covers a $2\pi$ range in a period so that $\frac{d\varphi}{dx} = \frac{2\pi}{D}$ . Therefore, equation (6) can be rewritten as

$$
k \left[ \sin \left(\theta_ {r}\right) - \sin \left(\theta_ {i}\right) \right] = m ^ {\prime} G,\tag{7}
$$

which is the grating equation, where $m' = m + 1$ represents the diffraction order. This reveals that a GPM, in essence, is a diffraction grating. The number of allowable diffraction channels, associated with different diffraction orders, is determined by the period, D, while the energy distributed among these channels is dictated by the specific phase-shift profile in a period. GPMs are a special type of diffraction grating whose phase-shift profile features a linear gradient. As a result, the anomalous reflection governed by equation (3) directs the energy into the diffraction of order 1 ( $m' = 1$ ), rather than the normal one ( $m' = 0$ ) for specular reflection. However, such a diffraction order ( $m' = 1$ ) may no longer be allowed to propagate for some large-angle incidence, since the reflection term, i.e. $\sin(\theta_{r})$ , in equation (3) may become complex. Physically, this means that, when the angle of incidence is greater than a critical angle, the anomalous diffraction corresponding to $m' = 1$ is an evanescent wave, which travels along the interface [33].

The same physical principle discussed above can be also applied to other types of acoustic metasurfaces. For example, the GPMs provide a phase shift that yields a gradient in the Cartesian coordinates, while a phase-shift gradient in the azimuthal direction enables a metasurface to generate vortex sound carrying orbital angular momentum (OAM) $[36, 37]$ . Moreover, a linear phase-shift metasurface (i.e. GPM) redirects sound but preserves its wavefront, while a nonlinear phase-shift metasurface is able to reshape the wavefront, leading to beam focusing $[38]$ , beam bending $[39]$ , and diffuse reflection $[40]$ . It should also be pointed out that sound-absorbing metasurfaces do not share the same working principles as the sound scattering (reflection-type or transmission-type) metasurfaces. They typically do not act as a phase-control structure, but instead aim to effectively absorb sound by dramatically reducing the thickness of the structure. The progress of these sound absorbing metasurfaces will be elaborated on in section 5.2. Note that, in this set of traditional metasurfaces, the dissipation, radiation coupling, and acoustic-structure interaction, which all can be classified as non-Hermitian effects $[41–43]$ and non-local effects $[44]$ , have already been partly considered. As we will see in the following, these effects are in fact the key for a new generation of acoustic metasurfaces.

## 3. Non-local acoustic metasurfaces

## 3.1. Limitation of traditional metasurfaces

For conventional acoustic metasurfaces, exemplified by the GPMs, the capabilities to mold the flow of sound hinge on their carefully designed and arranged meta-units at the sub-wavelength scale along the surface. These meta-units are tailored to provide a precise local response to external excitations. Importantly, these building blocks are designed individually, and then assembled in a particular pattern to form the metasurface. This design strategy naturally only takes local acoustic effects into account, whereas the waves under manipulation experience non-local effects, owing to diffraction—an intrinsic property underpinning wave propagation. These subwavelength meta-units inevitably affect the radiation emitted by neighboring units, leading to the radiation coupling between densely-placed units. Traditional acoustic metasurfaces do not account for this non-local effect, which is expected to impact the overall response of the metasurface. This impact is typically demonstrated by a rapid decrease in beam redirecting efficiency that occurs as the steering angle deviates from the specular reflection angle. In the first part of this section, we provide a historical account of the efforts made towards this issue, which ultimately resulted in the breakthrough on non-local acoustic metasurfaces. As will be shown, non-local acoustic metasurfaces deliberately utilize coupling and engineered non-locality to achieve ultra-high efficiency wave field manipulations, in contrast to the common approach that would minimize non-local effects. For a detailed review of the development of non-local metasurfaces in electromagnetics, interested readers can refer to $[45]$ . Besides non-local metasurfaces, other solutions to address the efficiency issue are also discussed in sections 3.2 and 3.3 for completeness.

(a)
![](images/45e1fe87c00eb4eba2e11aede2e701fb5d175e2453dc13c0e508cb666a8c87c1.jpg)

(b)
![](images/04a3a93177523d4aa706afbe96155320478b773212d277228648b7c19d3a0d90.jpg)

![](images/e85fa1e1d92d8b69fcac777f88dae475c2d40ec4358b28389918689f4ca0a062.jpg)

(d)
![](images/9695d7f322c4ddc0f2466183a53a2e0bb8ca730824cb872a107c5a65e2a49da7.jpg)
Figure 2. (a) Efficiency of the GPM as a function of the angle of reflection under normal incidence. (b)-(d) Schematics of the reflected fields of a normal incident wave when (b) $\theta_r = 10^\circ$ , (c) $\theta_r = 30^\circ$ and (d) $\theta_r = 60^\circ$ . The efficiency can be seen to drop as the angle of reflection increases.

To better illustrate the limitations of traditional GPMs, we revisit the generalized law of reflection, i.e. equation (3). While this equation predicts the angle of reflection resulted from the GPM, it does not provide information about the efficiency of reflection. In the following, we will attempt to calculate the efficiency as a function of the angle of reflection. For an inhomogeneous surface possessing a phase-shift profile described by equation (4), the local reflection coefficient r (the ratio between the reflected wave pressure and incident wave pressure) can be written as

$$
r = \mathrm{e} ^ {i \varphi (x)},\tag{8}
$$

where i is the imaginary unit. Equation (8) suggests a complete reflection (manifested by the amplitude of r being one) since we assume the surface provides only phase shift to the incident wave, without changing its amplitude. The reflection coefficient is related to the surface impedance as

$$
r = \frac {Z _ {s} - Z _ {i}}{Z _ {s} + Z _ {i}},\tag{9}
$$

where $Z_{i} = \rho_{0}c_{0}/\cos\theta_{i}$ is the specific acoustic impedance of the incident wave at the metasurface, with $\rho_{0}$ and $c_{0}$ being the density and sound speed of the background media, respectively. Combing equations (8) and (9), the impedance of the metasurface can be written as [46]

$$
Z _ {s} (x) = i \frac {\rho_ {0} c _ {0}}{\cos \theta_ {i}} \cot \frac {\varphi (x)}{2}.\tag{10}
$$

Such a reactive impedance (purely imaginary) indicates that a GMP depicted by equation (3) is a passive (without amplification) and lossless (without dissipation) boundary. We then calculate the efficiency of the anomalous reflection from a boundary with the impedance described by equation (10) under different angles of reflection, following the approach in [46]. The reflected wave fields are computed using COMSOL Multiphysics. The angle of incidence is assumed to be a constant $(\theta_{i} = 0^{\circ})$ . It can be found from figure 2 that the efficiency of the metasurface governed by the generalized law of reflection decreases as the angle of reflection increases, owing to the fact that parasitic diffraction modes are generated, which take the energy away. In fact, to ensure perfect reflection for the order $m' = 1$ , such a metasurface should be characterized by the following impedance [46]

$$
Z _ {s} (x) = \rho_ {0} c _ {0} \frac {1 + \sqrt {\frac {\cos \theta_ {i}}{\cos \theta_ {r}}} \mathrm{e} ^ {i \varphi (x)}}{\cos \theta_ {i} - \sqrt {\cos \theta_ {r}} \cos \theta_ {i} \mathrm{e} ^ {i \varphi (x)}},\tag{11}
$$

which now turns out to be no longer imaginary, but complex, of which the positive (negative) real part implies that such a metasurface yields local loss (gain).

Such a fundamental limitation of traditional passive metasurfaces was first discovered in optics, showing that local phase compensation is essentially insufficient to realize perfect control of reflection and refraction $[47, 48]$ . It was also shown that the optimal phase distribution that maximizes the transformation efficiency, in the case of passive reflection-type metasurfaces, largely deviates from the linear phase distribution underpinning the GPM design $[47, 48]$ . For acoustics, the drawback in the efficiency of manipulations of reflection and refraction by conventional local metasurfaces has also been observed $[46]$ . It was found that ideal anomalous refraction is possible only if the metasurface is bianisotropic, while ideal anomalous reflection requires a strongly non-local response, which allows energy channeling along the metasurface $[46]$ .

![](images/592ce95f4edf11dcb43dacc14e58f198ffbb9ade9ef10d1024be20d2d6f41d31.jpg)

(c)
![](images/92e232c0a530e64b866a0633f425a31a69988c2f97a332d3a25cf8b4e3f47757.jpg)
(d)

(b)
![](images/b41e351b1926e1195d050c95348b8194fa68fb69826f4b2ebe381788cfb46024.jpg)

![](images/bb130ef6df2811cba712d3048937011602cfcd885afea8556178c80faf7b0d2c.jpg)

![](images/379475ee49e56c7c57f3d08a200fa4dfef4127c5f32531f98dd1391215ca2776.jpg)
(e)

![](images/95257c2a475ac5908f09e9cde0b3c91793d0ac2333471231cfbf9ff52e69192c.jpg)

![](images/1268446bcb8f255a096cc346b0a8647dd593cb7956016c74db473b56178cf09f.jpg)
Figure 3. (a) Schematics of a transmission-type bianisotropic metasurface for perfect anomalous sound refraction. Each bianisotropic meta-unit consists of a straight channel equipped with four side-loaded resonators. (b) Simulated and experimental results of the pressure fields of the bianisotropic metasurface with $\theta_{i} = 0^{\circ}$ and $\theta_{t} = 80^{\circ}$ . (c) Schematics of a power-flow-conformal metasurface, which is a curved metasurface for perfect anomalous sound reflection. (d) Distribution of the sound intensity vector (left) and the potential function of sound intensity (right) for the case with $\theta_{i} = 0^{\circ}$ and $\theta_{r} = 70^{\circ}$ . White lines (right panel) represent the level curves of potentials, i.e. the curves parallel to the intensity vector (left panel) at every point. (e) The realized power-flow-conformal metasurface using rigidly-ended grooves. Red lines indicate surfaces modeled as hard boundaries following the shape of a contour line marked by the white dashed line in the right panel of (d). Panels (a) and (b) are reproduced from [55]. CC BY 4.0. Panels (c)-(e) are reprinted from [58]. © The Authors, some rights reserved; exclusive licensee AAAS. Distributed under a CC BY-NC 4.0 Reprinted with permission from AAAS.

## 3.2. Bianisotropic acoustic metasurfaces

Bianisotropic meta-units were first used for simultaneous control of reflection and refraction $[49]$ before they were put forward as a solution to increase the metasurface efficiency $[46, 48]$ . In electromagnetic metamaterials, bianisotropy, or magnetoelectric coupling, enables the coupling of magnetic and electric phenomena at the subwavelength scale $[50]$ . As an analogy, Willis coupling has been explored in elastodynamics, which states that, in some inhomogeneous media, stress not only relates to strain but also couples to velocity, while momentum density is not only determined by velocity but also coupled to strain $[51, 52]$ . Willis coupling can be extended to describe the interaction between acoustic pressure and particle velocity, which leads to bianisotropic responses where the scattered fields are different depending on the direction of incident waves $[53, 54]$ .

To achieve a perfect refraction without parasitic diffractions using lossless acoustic metasurfaces, it is necessary for the metasurface to be bianisotropic. This requirement can be fulfilled by implementing a multi-layer design strategy. As a proof of concept, a bianisotropic acoustic metasurface composed of three layers of membranes featuring perfect refraction was numerically demonstrated $[46]$ . Besides using membranes, bianisotropic metasurfaces can be also realized by meta-units consisting of straight channels with multiple side-loaded resonators $[55–57]$ . For example, Li et al showed numerically and experimentally that bianisotropic acoustic metasurfaces can fully redirect normally incident waves with angles of refraction of $60^{\circ}$ , $70^{\circ}$ , and $80^{\circ}$ on the transmission side $[55]$ (figures 3(a) and (b)). Significantly improved energy efficiencies (93%, 96%, and 91%) over those of the conventional GPM-based design (89%, 58%, and 35%), were demonstrated at these large angles.

## 3.3. Power-flow-conformal acoustic metasurfaces

Bianisotropy was proposed as a key factor for high efficiency refraction control, while the requirements for achieving high efficiency reflections are more complicated $[46, 48]$ . To achieve a perfect anomalous reflection, the impedance of a flat metasurface should be complex, as indicated by equation $(11)$ . Díaz-Rubio et al $[58]$ , however, found that such a constraint of complex impedance can be relaxed, as long as the metasurface is non-flat (figures 3(c)–(e)). By analyzing the power flow distribution of the desired incident and reflected wave fields, it was revealed that the level curve of power-flow potential $[58]$ implies a purely imaginary (reactive) impedance when it acts as a boundary. In this way, lossless metasurfaces can be constructed by conventional passive groove-type meta-units, where the specific spatial profile of the metasurface follows the level curve of power-flow potential. This curved-surface metasurface is therefore called the power-flow-conformal metasurface. Such a groove-type power-flow-conformal metasurface, by adding a metallic grid in the groove, can be further used to achieve a dual-physics device that can simultaneously route electromagnetic and acoustic waves with high efficiency, where the acoustic impedance is controlled by the groove depth, while the electromagnetic wave impedance is controlled by the metal grid $[59]$ . In addition to showing the capability of high efficiency wave reflection control, power-flow-conformal metasurfaces have also found applications for high efficiency acoustic focusing $[60]$ and beam splitting $[61]$ .

![](images/1c85f272f925619c622107d5022ef9f351b56c7942456b5e514962d81bd94d8a.jpg)
Figure 4. (a) Schematics of a non-local reflection-type metasurface utilizing physical coupling. Only the air domain inside the meta-units is shown. (b) Simulated reflection fields from a metasurface consisting of four meta-units in each unit cell, without (left panel) and with (right panel) physical coupling. (c) Schematics of a non-local transmission-type metasurface utilizing radiation coupling. (d) Local sound intensity in the vicinity of the metasurface. The arrows are strongly distorted in the area very close to the metasurface, showing a significant non-local effect. (e) Simulated (top panel) and measured (bottom panel) transmitted pressure fields from the metasurface with $\theta_{i} = 0^{\circ}$ and $\theta_{t} = 72^{\circ}$ . The white arrows show the directions of the refracted waves. Panels (a) and (b) are reprinted (figure) with permission from [62], Copyright (2019) by the American Physical Society. Panels (c)-(e) are reprinted (figure) with permission from [64], Copyright (2019) by the American Physical Society.

## 3.4. Non-local effects from physical coupling

As shown earlier, achieving perfect anomalous reflection requires a non-local metasurface with a complex impedance, where the real part of the impedance indicates local gain (negative) or loss (positive). An important scientific question to consider is whether it is possible to achieve perfect reflection control using a metasurface without any gain and/or loss. Power-flow-conformal metasurfaces $[58–61]$ have been identified to be a solution, but they require a curved surface profile according to the desired energy flow distribution, which comes at a cost: since the power flow distribution varies with different incident and reflected wave fields, such a solution may only work exclusively for some specific angles of incidence and reflection. On the other hand, it has been suggested that one may not need real gain and/or loss to achieve a boundary with complex impedance. The local gain and loss associated with the negative and positive real part of the impedance can be in fact fulfilled by allowing coupling between adjacent meta-units. Specifically, coupling provides a means for the acoustic energy to tunnel from one meta-unit to another. The energy outflowing from the meta-unit represents local loss, while the energy inflowing into the meta-unit can be regarded as local gain.

Following this concept, Quan and Alù [62] showed that coupling can be introduced by a physical channel connecting neighboring meta-units on a metasurface, and therefore such a design is characterized by its intrinsic non-locality (figures 4(a) and (b)). Such a non-local metasurface, where the non-locality is dominantly contributed by physical coupling, enables unitary efficiency for extreme beam steering unattainable by conventional GPMs. The non-locality can be readily engineered by the geometry of the physical connecting channels. The authors further extended the 1D design $[62]$ to a 2D case and created different physical connecting channels in different directions along the surface. In this way, anisotropy can be induced through detuned non-locality in different directions. This enables extreme anisotropic responses for sound traveling tangential to the metasurface $[63]$ .

## 3.5. Non-local effects from radiation coupling

Adding physical channels to connect neighboring meta-units is a rather intuitive and straightforward way to induce a non-local effect. However, a metasurface is usually made up of densely packed meta-units, and creating small channels between these meta-units appears to be a daunting task and can lead to an intricate structure that is difficult to fabricate and fraught with thermoviscous dissipation.

Recall that in section 3.1, it is pointed out that the radiation coupling between meta-units compromise the overall performance of a GPM in terms of the efficiency. While radiation coupling presents a challenge for conventional metasurface designs, it can be harnessed to achieve the necessary complex impedance (equation (11)) for non-local metasurfaces. In this case, rather than attempting to minimize the radiation coupling, we instead embrace radiation coupling as a powerful tool to tune energy exchange among meta-units. These non-local metasurfaces utilizing radiation coupling hence do not require the use of connecting physical channels.

The first type of non-local metasurfaces based on radiation coupling can be regarded as a modification from the conventional GPMs $[46, 65]$ . The design follows a two-step scheme. In the first step, an array of lossless meta-units are designed individually following the local design strategy, which produces a gradient-phase shift. Since mutual radiation coupling naturally arises when the meta-units are assembled together, the second step is to further finetune the geometry of meta-units, by which the non-locality from radiation coupling is considered to enhance the global performance of the metasurface. Such a non-local metasurface features the same number of meta-units per period as GPMs (i.e. dense units, in contrast to the other type with sparse units to be introduced below), while the geometry of the meta-unit slightly deviates from those determined by the first step of the design scheme following the generalized law of reflection/refraction [46, 65].

![](images/0e4079f14e17a6006bc14306082b938e79c49d88216d5ed989683314a537354d.jpg)
Figure 5. (a) Schematics of an acoustic reflection-type metagrating composed of a surface with periodically etched grooves having two units in a period. (b) A retroreflective metagrating functioning as a carpet cloak. The two figures on the right show the scattering fields without and with metagrating-based carpet cloak under normal incidence. (c) A transmission-type metagrating splits an incident wave into different desired directions with arbitrary power flow partition. (d) and (e) Sound focusing from (d) a local metasurface and (e) a metagrating integrating the power-flow-conformal concept. The latter shows a stronger focus indicated by the color bar. Panel (a) is reprinted (figure) with permission from [69], Copyright (2019) by the American Physical Society. Panel (b) is reprinted (figure) with permission from [75], Copyright (2019) by the American Physical Society. Panel (c) is reprinted (figure) with permission from [79], Copyright (2019) by the American Physical Society. Panels (d) and (e) are reprinted (figure) with permission from [81], Copyright (2021) by the American Physical Society.

Another type of metasurfaces that fully consider the non-local effect from radiation coupling are metagratings. Although metasurfaces can be studied from a diffraction-grating perspective $[66]$ , metasurfaces and gratings are still distinct from each other. Compared to the wave scatterers in conventional gratings, meta-units in metasurfaces are far more complicated, manifested by their microstructures having more tunable geometrical parameters. Metasurfaces also typically have more than one meta-units in a period, in comparison with conventional gratings which only have one unit cell in a period. The field of electromagnetics first saw the fusion of metasurface and grating that leveraged their individual benefits, leading to the introduction of the term ‘metagrating’ $[67]$ . Metagratings enrich the physics of gratings: these elaborately engineered scatters (meta-units) provide another degree of freedom to suppress parasitic diffraction modes. On the other hand, metagratings reform the metasurfaces: metagratings no longer consider the individual response but the collective behavior of all constituent units, taking full use of the non-locality from coupling effect. Compared to GPMs, a metagrating manifests itself as a sparse-unit design (figure 5(a)). Acoustic reflection-type metagrating can be realized by using just a few grooves (or curling grooves), which shows their capabilities to redirect reflection in a highly efficient manner [68–70]. It is found that only one or two grooves are required for the most typical applications of anomalous reflectors, while a moderate number (e.g. five) of grooves are required for more sophisticated applications [68]. In addition, using such metagratings as boundaries forms an open waveguide. By suppressing all diffraction modes and completely reflecting incident waves without leakage, acoustic waves can be guided along an arbitrary path [71].

Retroreflection is a form of anomalous reflection in which the reflected wave travels in the same path as the incident wave. Carpet cloaking is an application that can be facilitated by retroreflection. Specifically, metagratings can be designed to restore the wave scattering of the object beneath them to render the object ‘invisible’ from detection $[72]$ . Compared to GPM-based strategies $[73, 74]$ , metagratings offer a significantly simplified approach to designing carpet cloaks. For example, it is possible to use a metagrating with only one meta-unit per period to efficiently cloak an object below $[75–77]$ (figure 5(b)). Additionally, a metagrating supporting retroreflections can also be utilized for levitation and contactless movement of particles $[78]$ .

Besides the reflection-type metagratings, transmission-type metagrating can be realized by using a meta-unit possessing one straight channel as well as several grooves etched on both sides of a plate $[64]$ (figures 4(c)–(e)). Sound passes the metagrating through the straight channel, which serves as a secondary source for the transmission side, while the grooves on the top and bottom surfaces are responsible for modulating the reflected and transmitted wave fields, respectively. By harnessing the coupling between the channel and grooves, such a metagrating further enables an asymmetric behavior on the reflection side, giving rise to perfect transmission or retroreflection, depending on the sign of the angle of incidence ( $\pm\theta_{i}$ ) [79]. For metagratings, non-locality plays a vital role in their unparalleled wavefront capabilities, which stem from the lateral energy exchange along the surface caused by the evanescent wave modes inside the meta-units [64, 80].

In addition to beam deflection, metagratings have also shown promises in wavefront reshaping, including splitting $[69, 79]$ , focusing $[81–83]$ , and even a combination of different functions $[84, 85]$ . High efficiency wave splitting was demonstrated by metagratings containing only two or four grooves per period $[69]$ , which can split an incident wave into beams with different desired directions and arbitrary power flow partition (figure 5(c)). On the other hand, by combing the power-flow-conformal concept and non-local metagrating $[81]$ , or by combining multiple metagratings with different periodicities $[82, 83]$ , integrating-based meta-lens engender increased focusing effect, overcoming the limited efficiency for a traditional lens (figures 5(d) and (e)). In addition, taking advantage of their simpler configurations (compared to GPMs), metagratings operating at ultrasound frequencies $[78, 82, 83]$ (even MHz frequencies $[86]$ ) are also more compatible with current fabrication techniques than conventional metasurfaces.

Compared to traditional gratings, the engineered meta-units provide additional degrees of freedom to suppress parasitic diffraction modes. A representative example is metagratings constructed by using bianisotropic meta-units. This type of metagrating effectively combines the strengths of metagratings, which feature sparse units and non-locality, and that of the bianisotropic microstructure, which provides an asymmetric response. Note that the term bianisotropy used here is different from that mentioned in section 3.2, where binanisotropy is used to describe a feature in transmission-type metasurfaces (with dense units) which support asymmetric responses on opposite surfaces of the metasurfaces (i.e. above and below the metasurface) [45, 53]. Whereas the bianisotropic reflection-type metagrating introduced here is concerned with asymmetric responses on opposite sides of the same surface normal (e.g. two incident waves that come from $+\theta_{i}$ and $-\theta_{i}$ ). Quan et al [87] showcased such an anisotropic metagrating composed of a single-layer identical scatterers placed above a rigid surface, in which bianisotropy is realized by breaking the spatial symmetry of meta-units (figures 6(a) and (b)). Their design reroutes a normally incident wave to extreme angles with unitary efficiency in reflection. It was further shown that such a metagrating can be reconfigured (by varying the orientation of meta-units and the distance above the rigid surface) to cater to a wide range of angles and frequencies [88, 89]. Without the rigid surface, a bianisotropic metagrating using a single-layer of identical meta-units with broken spatial symmetry can realize both asymmetric wave transmission and reflection [90], while a multi-layer bianisotropic metagrating gives rise to asymmetric beam splitting [91]. In addition to the above-mentioned 1D cases for wave control, bianisotropic meta-units can be periodically arranged in 2D, providing a powerful tool to manipulate waves in higher dimensions, such as to achieve asymmetric lateral sound beaming [92].

(a)
(c)
(b)
![](images/4b3d6bb50cbe537d234066a4b015dca8e7c27252d25dae42281e52194b5c4e2a.jpg)

![](images/2386d09f6b8dc8fa62368760e8ef046f5205bd9f43f5e3b4f69aa014f3146497.jpg)

(d)
![](images/6dcfd7d6d4bf890bd65276b220558908bc5f697a935cc134bc499c59d5b05821.jpg)

![](images/ec19ac243e521e90f435ebf54888147bc212b2df7ada0af808b8f135394bd643.jpg)

![](images/1881f94c77db04a7d2f66470bc86485787d16e4d31c894df90bb8ab08818b9cd.jpg)

(f)
![](images/2fa87655a7677ae808bb2c9fe08b47a95c355c5246591eb18463b8b55f14a8e3.jpg)
Figure 6. (a) Schematics of a reflection-type bianisotropic metagrating composed of identical asymmetric acoustic scatterers (i.e. one unit in a period). (b) Reflection spectrum for different diffraction-order channels. At 2404 Hz, all energy is reflected into the order -1 channel and unitary reflection is achieved.

(c) Schematics of a transmission-type metagrating as an acoustic vortex generator. The metagrating is composed of a small hole at the center and the surrounding grooves on both surfaces. (d) Simulated results for generating acoustic vortices with topological charges of 2 (upper panel) and 3 (lower panel). For each row, the left panel shows the distribution of the real part of the scattered pressure field along the axial direction; the middle and right panels show the distribution of the phase (normalized by $\pi$ ) and amplitude on the cross-section, respectively. (e) Schematics of the non-local design for a underwater acoustic metagrating. The metagrating is composed of periodic meta-units and supports 0th, $\pm1$ st, $\ldots$ , $\pm$ Nth order diffraction (left panel). Driven by the objective function of reflection efficiency, a parameter optimization procedure with genetic algorithm is applied for the inverse design of the metasurface, by which the non-local interaction between the meta-units induced by fluid-structure interaction is considered (right panel). (f) A case for perfect anomalous reflection with $\theta_{i}=0^{\circ}$ and $\theta_{r}=81^{\circ}$ utilizing the optimal design method, where the white arrows indicate the local power intensity vector of the reflected acoustic pressure fields. Panels (a) and (b) are reprinted (figure) with permission from [87], Copyright (2018) by the American Physical Society. Panels (c) and (d) are reprinted (figure) with permission from [93], Copyright (2021) by the American Physical Society. Panels (e) and (f) are reprinted (figure) with permission from [94], Copyright (2021) by the American Physical Society.

In contrast to plane waves, vortex beams carry OAM and therefore enrich the wave-matter interaction physics, which have shown great potential applications $[95–97]$ . While metasurfaces have been largely used to redirect plane (or quasi-plane) waves in the linear space (Cartesian coordinates), the same concept can be introduced into the angular space (polar coordinates) to generate and control acoustic vortices. For example, sound vortices can be realized in a cylindrical waveguide with angular-GPM made of fanlike resonators $[96]$ . The incident vortex can be either transmitted or reflected depending on the integer parity of the meta-units according to the integer parity theory $[98]$ . Besides the gradient-phase-based design, a vortex beam generator can also be realized via acoustic metagratings $[93]$ . One such a metagrating has a small cylindrical hole in the center, as well as sparse fanlike grooves etched on both sides of the metagraing (figures 6(c) and (d)). By harnessing the non-locality, one can fully control the topological charge of OAM carried by the acoustic vortex with a high efficiency $[93]$ .

The emerging non-local metasurfaces also have a particular meaning for underwater acoustics. Because of the considerably higher characteristic impedance of water than air, thin walls of meta-units can no longer be considered acoustically rigid in water. Therefore, fluid-structure interaction becomes far more important than for airborne sound and it could result in strong non-local interaction between the meta-units in water. This has made underwater metasurfaces a new playground for exploiting the physics and applications of non-local effects $[76, 78, 83, 89, 94, 99–102]$ . For non-local underwater metagratings, the non-local interaction between the meta-units induced by fluid-structure interaction can be carefully engineered via an optimization method (e.g. genetic algorithm) (figures 6(e) and (f)). In this way, by using meta-units such as cylindrical scatterers $[99, 100]$ , cylinders with slits $[89]$ , side-branched Helmholtz resonators (HRs) $[94]$ , and grooves $[76, 78, 101, 102]$ , various functions including anomalous reflection and transmission, retroreflection, beam splitting, and highly asymmetric transmission can be realized for underwater sound with desired efficiency.

It should be pointed out that a non-local metasurface may share both types of coupling. Consider the example shown in figures 4(c)–(e). The grooves on each side of the metasurface facilitate radiation coupling among themselves, whereas the straight channel serves as a physical coupling that connects the reflection and transmission sides. The same is true for the one shown in figures 6(e) and (f), as such a metagrating in fact embraces both the radiation coupling among the meta-units as well as the physical coupling induced by wave-structure interaction.

## 4. Non-Hermitian acoustic metasurfaces

## 4.1. Introduction of non-Hermitian acoustics

Conventional acoustic metasurfaces operate by modulating the phase and are typically treated as lossless systems. However, thermoviscous losses in acoustic metasurfaces are inevitable due to their subwavelength microstructures (narrow regions such as channels or grooves) [41] and these losses could become a dominating factor once amplified by acoustic resonances. Such naturally occurring losses cannot be ignored in acoustic metasurfaces, and if left unchecked, they could undercut the performance of acoustic metasurfaces. Furthermore, efforts on addressing the efficiency of conventional metasurfaces have pointed out the importance of complex surface impedances $[46]$ . Taken together, these observations imply that acoustic metasurfaces should be regarded as inherently lossy systems, a characterization that places them within the rapidly expanding field of non-Hermitian physics.

The concept of non-Hermitian physics was originated from quantum mechanics, in which the Hermiticity of a Hamiltonian that describes the conservation of energy in a closed system plays a fundamental role. However, actual physical systems are not always conservative since they can interact with the surrounding environment, leading to non-Hermitian systems. Non-Hermitian Hamiltonians are applicable to a wide range of systems such as (but not limited to) systems with open boundaries, as well as systems with gain and/or loss. It had been a well-accepted notion that a quantum-mechanical system (more precisely, its Schrödinger operator) must be Hermitian in order for the system to possess entirely real-valued energy spectra. In 1998, Bender and Boettcher [103] discovered something quite counter-intuitive. They theoretically showed that, for a non-Hermitian system, such as one that can be described by the 1D Schrödinger equation with a complex potential $V(x)$

$$
i \frac {\partial}{\partial t} \psi (x, t) = - \frac {\partial^ {2}}{\partial x ^ {2}} \psi (x, t) + V (x) \psi (x, t),\tag{12}
$$

its spectrum of the Schrödinger operator $-\partial_{xx} + V(x)$ can be entirely real-valued if the system is parity-time (PT)-symmetric. In other words, if this equation is invariant under the combined action of parity $\mathcal{P}(x \to -x)$ and time-reversal $\mathcal{T}(t \to -t, i \to -i)$ , the energy level E of the eigenstates $\psi(x, t) = u(x)\mathrm{e}^{iEt}$ , where $u(x)$ is the stationary wavefunction, can in principle be real-valued under a certain threshold. In this case, one can show that a necessary (albeit not sufficient) condition for this complex potential to be PT-symmetric is

$$
V (x) = V ^ {*} (- x),\tag{13}
$$

where $*$ represents complex conjugate. While the exploration of PT-symmetry in actual quantum systems presents a challenge, electromagnetic and acoustic wave systems have emerged as fertile ground for fruitful investigation of PT-symmetry concepts, leading to the discovery of new mechanisms and opportunities to control classical waves. To have a PT-symmetric quantum-mechanical system, it is necessary to have complex potentials with even real and odd imaginary parts as functions of position. The same is true in electromagnetics and acoustics, leading to the requirement of complex indices of refraction, and the imaginary part of the index can be induced by gain or loss. In other words, the complex refractive index function $n(x) = n_{R}(x) + i n_{I}(x)$ (or, correspondingly, the effective dielectric permittivity function in electromagnetics, and effective sound speed function in acoustics) now plays the role of a complex potential $V(x)$ , where $n_{R}(x)$ represents the refractive index distribution while $n_{I}(x)$ characterizes the gain and loss profiles within the medium. In this case, $\mathcal{PT}$ -symmetry implies that

(a)

$$
n _ {R} (x) = n _ {R} (- x),\tag{14}
$$

and

$$
n _ {I} (x) = - n _ {I} (- x).\tag{15}
$$

Hence, the refractive index function must be even, whereas the gain (loss) profile should be an odd function of position. These equations hold for acoustic waves in fluids $[104]$ , and solids $[105]$ , where the refractive index is determined by $n = c/c_{0}$ (c and $c_{0}$ being wave speed in the meta-unit and background medium). More explicitly, since c are determined by the two constitutive terms, i.e. the density and modulus of elasticity (for example $c = \sqrt{\kappa/\rho}$ for waves in fluids, and $c = \sqrt{E/\rho}$ for longitudinal waves in solids with E being the Young's modulus), such a designated reflective index can be realized by building a meta-unit with the corresponding effective modulus or/and effective density, following the effective medium theory. What makes electromagnetics and acoustics an ideal platform for PT-symmetry is that all these three ingredients (refractive index, gain, and loss) can be readily engineered to achieve the needed balance. For those interested in non-Hermitian acoustics, detailed reviews are available $[106, 107]$ . In this review, we specifically focus on non-Hermitian physics in scattering (2D) systems, while briefly mentioning other systems (0D, 1D, and topological systems) for a self-contained presentation.

Materials possessing gain and loss play a vital role in building an acoustic PT-symmetric system. However, in contrast to lossy materials, acoustic gain media are absent in nature. It is therefore the most challenge part to realize PT-symmetric acoustics involving a gain material that can be precisely tuned. One of the most common solutions is to utilize loudspeakers with elaborately designed feedback circuits to artificially amplify the acoustic signal $[108–111]$ . Another common approach is to induce feed-back controlled piezoelectric elements (PZT for example) based on the electromechanical coupling effect $[112–118]$ , which is more practical for manipulating elastic waves where speaker are not suitable $[113, 114, 116–119]$ . Besides these active sound generating units with feedback control, acoustic gain can also be realized by leveraging flow-induced sound $[120]$ , based on the hydrodynamic instability theory. Other solutions include utilizing electromagnets $[121]$ , optomechanical effect $[122–128]$ , and electro-thermoacoustic coupling $[129]$ .

## 4.2. The minimal model

To offer an intuitive understanding of non-Hermitian acoustics, we begin by introducing a minimal model. This model can aid readers in comprehending several fundamental concepts that underlie non-Hermitian acoustics. The minimal model is a two-level system, which in acoustics, can be physically realized by two identical resonant cavities coupled by a connecting tube (figure 7). The tube can be also used to adjust the coupling strength. By leveraging the temporal coupled mode theory [130], such a coupled-cavity acoustic system can be described by

![](images/23bf206a77aeb3353b75fe43d29d2f969fcf8752790acec80690465f65bd583d.jpg)
Figure 7. Schematics and the eigenfrequencies of (a) a Hermitian and (b) a non-Hermitian acoustic system comprising two coupled cavities, where $\kappa$ represents the coupling, $\gamma_{1}$ and $\gamma_{2}$ represent local loss/gain in the left and right cavities, respectively. The splitting of the eigenfrequencies grows as $\kappa$ increases in the Hermitian system. For the non-Hermitian system, the eigenfrequencies experience a transition point (EP, marked by the star) from the PT-symmetric phase (real-valued eigenfrequencies) to the PT-broken phase (complex-valued eigenfrequencies). Insets feature the eigenmodes of the coupled cavities for (a) the Hermitian case from non-coupling to strong coupling conditions, and (b) the non-Hermitian case from PT-symmetric regime, EP, to PT-broken regime. In the lower panels, the eigenfrequencies (the curves) and the eigenmodes (the insets) of two-cavity systems are calculated using the eigenfrequency analysis module of COMSOL Multiphysics. The degeneracy at EP results in only one eigenmode. In the PT-broken regime, one of the eigenmodes exists in the cavity with gain and experiences amplification, while the other one exists in the cavity with loss and experiences attenuation.

$$
\frac {\mathrm{d} a _ {1}}{\mathrm{d} t} = i \omega_ {0} a _ {1} + \gamma_ {1} a _ {1} + i \kappa_ {1 2} a _ {2},\tag{16}
$$

and

$$
\frac {\mathrm{d} a _ {2}}{\mathrm{d} t} = i \omega_ {0} a _ {2} + \gamma_ {2} a _ {2} + i \kappa_ {2 1} a _ {1},\tag{17}
$$

where $\omega_{0}$ is the resonant frequency of the two identical cavities (denoted as A and B), $a_{1}$ and $a_{2}$ are the mode amplitudes in the two cavities, $\kappa_{12}\left(\kappa_{21}\right)$ is the coupling representing hopping from cavity B (A) to cavity A (B), and $\gamma_{1}$ and $\gamma_{2}$ are the onsite potential (local gain/loss) in cavities A and B. The equations above can be rewritten in a matrix form, as

$$
- i \frac {\mathrm{d}}{\mathrm{d} t} \psi = \mathbf {H} \psi ,\tag{18}
$$

where $\psi=\left[\begin{array}{cc}a_{1}&a_{2}\end{array}\right]^{T}$ (the superscript T being transposition). This equation is in direct analog with the temporal Schrödinger equation, and the effective Hamiltonian for the coupled-cavity acoustic system is

$$
\mathbf {H} = \left[ \begin{array}{c c} \omega_ {0} - i \gamma_ {1} & \kappa_ {1 2} \\ \kappa_ {2 1} & \omega_ {0} - i \gamma_ {2} \end{array} \right],\tag{19}
$$

whose eigenvalues take the form

$$
\omega_ {\pm} = \omega_ {0} - i \frac {\gamma_ {1} + \gamma_ {2}}{2} \pm \sqrt {\kappa_ {1 2} \kappa_ {2 1} - \frac {\left(\gamma_ {1} - \gamma_ {2}\right) ^ {2}}{4}}.\tag{20}
$$

First consider a Hermitian case where the two cavities are passive and lossless, i.e. $\gamma_{1,2}=0$ , as well as reciprocal, i.e. $\kappa_{12}=\kappa_{21}$ . The eigenvalues are then simplified to $\omega_{\pm}=\omega_{0}\pm\kappa(\kappa=\kappa_{12}=\kappa_{21})$ , indicating that the eigenfrequencies are real values and the splitting of the two eigenvalues are determined by the coupling strength, i.e. $\Delta\omega=2\kappa$ . Generally, the separation of the eigenfrequencies increases as the coupling strength grows (figure 7(a)).

Once the system becomes non-Hermitian $(\gamma_{1,2} \neq 0)$ , equation (20) indicates that the eigenfrequencies, in general, are no longer real, but complex. However, when $\gamma_{1,2} = \pm\gamma$ , a special set of solutions emerge where the Hamiltonian yields

$$
\mathbf {H} = \left[ \begin{array}{c c} \omega_ {0} - i \gamma & \kappa \\ \kappa & \omega_ {0} + i \gamma \end{array} \right],\tag{21}
$$

with the eigenvalues

$$
\omega_ {\pm} = \omega_ {0} \pm \sqrt {\kappa^ {2} - \gamma^ {2}},\tag{22}
$$

where $\gamma$ is positive real. Interestingly, this Hamiltonian's eigenvalues can be real even though the system is non-Hermitian. In this case, the onsite potentials $\gamma_{1,2}$ in the two cavities have the same amplitude but opposite signs, where the positive one represents a local gain while the negative one indicates a local loss. This refers to a particular acoustic system with balanced gain and loss: a PT-symmetric configuration.

With this particular configuration, whether the spectrum is real still depends on the relative strength of the coupling $(\kappa)$ in regard to the gain/loss $(\gamma)$ . To illustrate this more clearly, let us gradually increase the gain/loss amplitude $\gamma$ . When $\gamma < \kappa$ , the square root term in equation (22) is real, so does the eigenfrequencies $\omega_{\pm}$ : real $(\omega_{\pm}) = \omega_0 \pm \sqrt{\kappa^2 - \gamma^2}$ and $\operatorname{Im}(\omega_{\pm}) = 0$ . This is known as the PT-symmetric phase. When the gain/loss continuous to increase so that $\gamma > \kappa$ , the eigenvalues would show a distinct picture: the square root term in equation (22) now becomes imaginary, leading to a complex eigen-spectrum with a degenerate real part, i.e. real $(\omega_{\pm}) = \omega_0$ , and a bifurcated imaginary part, i.e. $\operatorname{Im}(\omega_{\pm}) = \pm i\sqrt{\gamma^2 - \kappa^2}$ (figure 7(b)). Now the system transitions to a so-called PT-broken phase. Between the PT-symmetric and PT-broken phases, there is a critical point where phase transition happens, i.e. $\gamma = \kappa$ . In this case, $\omega_{\pm} = \omega_0$ , indicating a simultaneous degenerate real part and imaginary part of eigenfrequencies. This pinpoints the PT-symmetry breaking threshold, which is the phase transition point known as the exception points (EPs). In general, EPs are singularities in parameter space characterized by the defective Hamiltonian and uniquely supported by non-Hermitian systems, where both the eigenvalues $(\lambda_1 = \lambda_2 = \omega_0)$ and eigenvectors $\left(\nu_1 = \nu_2 = \left[-i\gamma/k-1\right]^T\right)$ of the system's Hamiltonian simultaneously coalesce [131, 132].

Building upon the minimal model, we further identify the difference between non-Hermitian acoustics and quantum systems. In quantum mechanics, the PT-symmetry was proposed under the premise that the observable of any quantum mechanical operator should be real. For the Schrödinger operator (the Hamiltonian), the observable is energy. Therefore, it is only physically meaningful for a quantum mechanical system to operate in a PT-symmetric phase, which ensure a real spectrum. However, for acoustics (and electromagnetics), the eigenvalues of the effective Hamiltonian can be complex, in which the real part refers to the resonant frequency and the imaginary part comes from the loss or gain depending on its sign. This innate difference, therefore, renders the non-Hermitian concept even more intriguing in the acoustic platform. Mathematically, for any non-Hermitian system, the EP (of order n) exists as long as the Hamiltonian matrix, under a similarity transformation, can be shown to contain an $n \times n$ Jordan block J [133, 134]

$$
\mathbf {J} = \left[ \begin{array}{c c c c} \omega_ {0} & 1 & & \\ & \omega_ {0} & \ddots & \\ & & \ddots & 1 \\ & & & \omega_ {0} \end{array} \right],\tag{23}
$$

and the PT-symmetric Hamiltonian described by equation (21) is a similar matrix of a $2 \times 2$ Jordan block. This requirement can be typically satisfied for a general effective Hamiltonian depicted by equation (19) in two ways: one is with unbalanced local loss/gain ( $\gamma_{1} \neq -\gamma_{2}$ ) [135], while the other is with unidirectional (non-reciprocal) coupling ( $\kappa_{12} = 0$ , or $\kappa_{21} = 0$ ) [136]. These are the two main approaches applied to non-Hermitian acoustic systems approaching an EP. This more relaxed requirement for effective Hamiltonians in contrast to that in quantum systems enables EP-related features to be fruitfully explored by classical acoustic systems considering only loss or non-reciprocity. A simple example is given here to illustrate this point. Assuming the same coupled-cavity system with (detuned) losses of $\gamma_{1}$ and $\gamma_{2} = \gamma_{1} + \Delta\gamma$ . The system is also assumed to be gainless and reciprocal. The Hamiltonian can be written as [135]

$$
\mathbf {H} = \left[ \begin{array}{c c} \omega_ {0} - i \gamma_ {1} & \kappa \\ \kappa & \omega_ {0} - i \gamma_ {2} \end{array} \right],\tag{24}
$$

whose eigenvalues yield

$$
\omega_ {\pm} = \omega_ {0} - i \frac {\gamma_ {1} + \gamma_ {2}}{2} \pm \sqrt {\kappa^ {2} - \frac {\Delta \gamma^ {2}}{4}}.\tag{25}
$$

It can be shown that an EP arises when $\Delta\gamma=2|\kappa|$ , at which point $\omega_{\pm}=\omega_{0}-i\frac{\gamma_{1}+\gamma_{2}}{2}$ . This is a clear evidence that EPs can be realized in an acoustic system with only loss.

## 4.3. Non-Hermitian acoustics in closed systems (0D)

As shown by the minimal model, the physics of non-Hermitian acoustics can be demonstrated in a coupled-cavity system,

(a)

which is isolated from the surrounding environment while the non-Hermiticity comes from onsite gain and loss. In this case, acoustic gain and loss can be achieved by utilizing loudspeakers and various energy dissipation mechanisms, while the coupling can be fine-tuned predominately by the width and position of the tube connecting the cavities $[137]$ .

(c)

Within the field of non-Hermitian physics, there are numerous intriguing phenomena that can be attributed to the existence of EPs. As shown by the minimal model, the collapse of the $2 \times 2$ effective Hamiltonian matrix (equation (19)) from a two-cavity system leads to a 2nd-order EP [123, 138]. It is thus reasonable to expect that coupling more cavities can lead to the attainment of multiple EPs, or even higher-order EPs. Indeed, Ding et al [135] built a passive non-Hermitian system, which is a four-state system by using four coupled cavities with detuned losses (figure 8(a)). They found that multiple EPs can emerge, and as the system parameters vary, these EPs can collide and merge, leading to higher-order EP singularities whose topological characteristics are much richer than those observed in a two-state system (figure 8(b)). It was also found that, the coalescence of multiple 2nd-order EPs having the same chirality can lead to higher-order EPs [135], while the coalescence of two EPs with opposite chiralities results in a diabolic point [139]. Furthermore, in some cases, non-Hermitian systems can also carry anisotropic EPs [135, 140, 141]. In this case, different singular behaviors are observed when the EP is approached from different directions in the parameter space [141], a behavior that is different from those observed in common (isotropic) EPs.

The abundance of system parameters in a coupled-cavity system provides a sufficient degree of freedom for continuously varying EPs in a higher-dimensional parameter space, leading to the emergence of exceptional arcs $[143]$ , exceptional rings $[144]$ , and exceptional surfaces $[145]$ . The potential offered by the coupled-cavity system to achieve these outcomes could bring about a significant advancement in our fundamental understanding of non-Hermitian physics. For example, by using a three-coupled-cavity system characterized by a 3rd-order non-Hermitian Hamiltonian, Tang et al showed a peculiar EP, which is not only of higher order, but is also an exceptional nexus acting as the cusp singularity of multiple exceptional arcs (figures 8(c) and (d)) $[142]$ .

## 4.4. Non-Hermitian acoustics in waveguides (1D)

A typical 1D non-Hermitian acoustic system can be achieved by periodically repeating the minimal model (two coupled cavities with gain and/or loss), resulting in a 1D acoustic chain with two sublattices in a period. In fact, such an acoustic chain can be viewed as either a closed system (with periodic boundary conditions) or an open system (with open boundary conditions). The latter case is more practical for experimental validation and has the potential for real-world applications. It can be essentially considered as a 1D chain of a finite number of periodic cells having two opposite ends that can be both connected to 1D waveguides. We are particularly interested in the exotic scattering behavior associated with the underlying non-Hermiticity in such open systems.

(b)
![](images/5fea41f6f8adfc8b0d29c759ca125c620501990421369a53d4558d2d80b90c01.jpg)

![](images/46fe808ea9810ef37b4e482439f09e01703c9bb59c347c05e0b746c894201396.jpg)
(d)

![](images/6a5df14ad8b81499fc319e16ad7a13c3da0c84618870bccab9adfba75f152bd8.jpg)

![](images/94d11c0b1492384bbfb11010d22cf5c5c63a60544213f160a6a75a51a22f54bf.jpg)
Figure 8. (a) A four-coupled-cavity system. The system can be regarded as combing two pairs of the minimal models of figure 7. The four cavities are labeled A–D in the inset, where A and B (C and (D) form a pair with resonant frequency $\omega_{2}$ ( $\omega_{1}$ ) with $\kappa(t)$ being the intra-(inter-) pair coupling. (b) The phase diagram of the four-coupled-cavity system by varying the difference of resonant frequencies ( $\Delta\omega = \omega_{1} - \omega_{2}$ ) and the coupling t. The solid red curve marks the coalescence of three EPs, while the solid yellow line marks the coalescence of two EPs. These lines separate the ( $\Delta\omega$ , t) parameter space into three regions marked by gray, blue, and green, each with a unique EP formation pattern. The white dashed line marks the state inversion line that further separates subclasses ‘a’ and ‘b’. (c) A three-coupled-cavity system hosting a 3rd-order EP, where $\delta_{A}$ and $\delta_{B}$ are the detuning in the respective sites, and $\delta_{g}$ and $\delta_{f}$ are the on-site loss or gain. (d) The evolution EPs in the 3D parameter space of $\delta_{g}$ , $\delta_{A}$ and $\delta_{B}$ . The 3rd-order EP acts as a cusp where multiple arcs of 2nd-order EPs merge. Panels (a) and (b) are reproduced from [135]. CC BY 3.0. Panels (c) and (d) are from [142]. Reprinted with permission from AAAS.

Prior to delving into the 1D acoustic systems, it is worth noting that non-Hermitian representations are classified under two distinct terminologies (H-matrix and S-matrix representations). Non-Hermiticity can be described by effective Hamiltonian exemplified by equation (19), and hence is referred to as the H-matrix representation hereafter. H-matrix gives an intuitive picture of the resonant frequencies (eigenvalues) and the wavefunctions (eigenvectors) of a system. On the other hand, the non-Hermicity in open systems are usually encoded in the scattering matrices (S-matrix representation). The eigenvalues in the S-matrix, to the best of our knowledge, have not been defined with clear physical meaning, while the eigenvectors are associated with particular inputs. For example, for a 1D waveguide system, the eigenvector represents a particular combination of inputs from left and right ends [65, 146]. At the EP where the matrix is defective, the H-matrix representation indicates a degeneracy of the resonant modes, while the S-matrix representation, as will be shown in the following, leads to extremely asymmetric scattering behaviors (distinct responses from different inputs). These two representations can be adopted in the same system, and the relationship between the H- and S- matrices has been discussed in some works $[147–149]$ . However, in general, these two representations are distinct. For example, for a 1D cavity chain with N periods of two-coupled cavities connected to waveguides at both ends, the size of the H-matrix depends on the number of the modes in the chain (when considering only the fundamental resonance of each cavity, H-matrix size is $2N \times 2N$ ), while the size of the S-matrix is determined by the number of propagating modes in the waveguide (only the right- and left-going plane waves are permitted below the cutoff frequency of a 1D waveguide; S-matrix size is $2 \times 2$ ). One can clearly see the fundamental difference between the S- and H- matrices by the size, since the number of resonant modes (in the chain) has no direct correlation with the number of scattering modes allowed (in the waveguides).

In general, a PT-symmetric 1D system features an asymmetric response from opposite sides, and such asymmetric behavior reaches an extreme at the EP, where unidirectional transparency or unidirectional reflectionlessness can be observed $[104, 108, 150]$ . Beyond EPs, in the PT-broken phase, all bulk modes can be localized at a given boundary, which is known as the non-Hermitian skin effects (NHSEs) derived from the unconventional bulk-boundary correspondence in non-Hermitian systems $[114, 118, 151–155]$ . As will be discussed in section 4.6, NHSE is another important feature of the non-Hermitian system. Moreover, in the PT-broken phase, a pair of particular points, referring to as the coherent perfect absorber (CPA) and laser points have been observed $[156–160]$ . The CPA-laser points occur when a pole and a zero of the eigenvalues of the Hamiltonian coexist at a specific frequency, which offers a promising route for wave modulation with capabilities of dramatical amplification and perfect absorption.

Many early works exploring non-Hermitian acoustics are carried out in 1D waveguide systems. Zhu et al [104] first introduced the non-Hermitian concept to acoustics by studying a finite 1D periodic acoustic medium, with complex parameters featuring balanced and alternative loss and gain, where unidirectional reflectionlessness (also known as unidirectional transparency) are numerically demonstrated at the EP (figure 9(a)). This unique scattering behavior was then observed in experiments, by utilizing feed-back-controlled loudspeakers [108] or piezoelectric elements [112] as the gain medium.

Although EP-related phenomena have been demonstrated in PT-symmetric acoustic systems $[104, 108]$ , PT-symmetry is not a necessary condition for realizing the EP, as we have demonstrated at the end of section 4.2. A passive scattering system with only loss, which does not respect the PT-symmetry, can still yield EPs associated with the coalescence of eigenvalues and eigenvectors of the S-matrix (figures 9(b)–(e)). There are different ways to realize a passive non-Hermitian system with EPs. One is a modification from the PT-symmetric framework by eliminating the gain. In this case, PT-symmetry can be reestablished through the gauge transformation $[161]$ , where the average loss bias can be regarded as the loss of the background medium while the lossless part now behaves a relative ‘gain’. Such a configuration is therefore also defined to be passive PT-symmetric [161]. Following this idea, Liu et al [146] showed that a passive system consisting of four periods, with loss modulation in each, can achieve unidirectional reflectionless wave propagation (figure 9(c)), though unidirectional transparency is absent in this system because of the lack of true gain (see the details of constructing a passive-PT-symmetric configuration through refractive index modulations following the four steps shown in figure 9(b): PT-symmetry with complex exponential modulation, approximated PT-symmetry with square-wave modulation; passive PT-symmetry with truncated complex square-wave modulation; and modified passive PT-symmetric configuration with separated real and imaginary part of refractive index modulations).

![](images/85b483f633e55651f037f6e9aad1082f063b3c0784f425818f8d4fc1e96d9da2.jpg)

(d)
![](images/43bf4e6536be997baeb2c5a0fc9e4f324b7292c8a417763e16e792ba0618284e.jpg)

![](images/cb4bb9210c098487d1a7e0410506bf42e561d824fe5e342e3caaa4719b344853.jpg)

![](images/4de34c0b925ae7d95d69760003b7218c14cd5580a36c6e5b1dcebaa9b4575580.jpg)
Figure 9. PT-symmetric and passive PT symmetric 1D non-Hermitian acoustic systems at EPs. (a) A PT-symmetric system possessing alternative loss and gain regions and an asymmetric wave behavior of unidirectional transparency. (b) A passive PT-symmetric system realized through the following four-step refractive index modulations: I. PT-symmetry with complex exponential modulation; II. Approximated PT-symmetry with square-wave modulation; III. Passive PT-symmetry with truncated complex square-wave modulation; IV. Modified passive PT-symmetric configuration with separated real and imaginary part of refractive index modulations. The red (blue) curves denote the modulation of real (imaginary) part of the refractive index. (c) A passive PT-symmetric system with the real (red) and imaginary (blue) parts of the refractive index modulated following IV in (b), which gives rise to unidirectional reflectionlessness. (d) Eigenvalues and (e) eigenvectors of the S-matrix, where lines and circles correspond to PT- and passive PT-symmetric configurations (II and IV in (b)). Panel (a) is reproduced from [104]. CC BY 3.0. Panels (b)-(e) are reprinted (figure) with permission from [146], Copyright (2018) by the American Physical Society.

Another approach for realizing EPs in passive systems is by leveraging the Willis coupling, as asymmetric response is a key feature manifested by the non-Hermitian system at the EP.

Indeed, Willis coupling media have shown their bianisotropic (i.e. asymmetric) responses in the context of metasurfaces $[87–91]$ . The majority of Willis coupling media that have been studied to date are lossless and do not provide access to EPs. Once loss is induced, EPs can be systematically synthesized in lossy Willis coupling media, giving rise to unidirectional reflectionless wave propagation of sound $[162, 163]$ , elastic waves $[164]$ , and water waves $[165]$ .

It should be reiterated that, mathematically, EPs just require the S-matrix (similar to equation $(19)$ ) to follow a certain form (equation $(23)$ ). This requirement can be readily satisfied for a coupled system with unbalanced local loss/gain in sublattices (detuning $\gamma_{1}$ and $\gamma_{2}$ ) or unbalanced (nonreciprocal) coupling among sublattices ( $\kappa_{12}=0$ , or $\kappa_{21}=0$ ). Such a mathematical condition suggests the universality of non-Hermiticity in an unbalanced system that is not built from the special techniques described in the above two paragraphs (see a simple model using a pair of forks with different losses $[166]$ ).

Non-Hermitian acoustics, including the PT-symmetric and passive PT-symmetric systems, provide a rich source of opportunities for the development of impactful applications. For example, the unique characteristics of non-Hermitian systems at EPs may find promising applications in acoustic sensing. Traditional sensors inevitably perturb the wave field, by its own physical presence (absorbing part of signal energy) or creating a shadow. By taking advantage of unidirectional transparency, a PT-invisible acoustic sensor can be realized. As shown by Fleury et al [109], such an invisible sensor can be formed by a pair of loudspeakers loaded with properly tailored electrical circuits, in which one loudspeaker is operated as a sensor by loading it with an absorptive circuit, while the other forms an acoustic gain element. On the other hand, the ultra-high sensitivity of a system at EP essentially amplifies any small parametric variations, a useful property for sensor applications [167]. It has been shown that a non-Hermitian system operating in close vicinity of an EP, with the eigenfrequencies of the relevant non-Hermitian modes being highly sensitive to perturbations to the system [117, 118, 126, 168], has potential applications in ultrasonic inspection, noise detection, and vibration accelerometer. Moreover, unidirectional transparency stemming from non-Hermitian acoustic systems can also be utilized to realize an acoustic directional cloak [104, 110]. In a similar way, such unidirectional transparency can be utilized to suppress the scattering from impurity, realized by a design combing zero-index and PT-symmetry [111, 169, 170].

Another application of particular interest based on non-Hermitian acoustics is directional absorption. Directional absorption in the waveguide can be realized at EPs $[156, 171–173]$ , which features extremely asymmetric response showcased by perfect absorption and strong reflection from opposite incidences for both sound $[171]$ and elastic waves $[173]$ . In addition, the non-Hermicity also helps tune the absorption spectrum, leading to an acoustic filter where undesired frequencies are absorbed and the desired ones are reflected $[174]$ .

Besides those applications achieved at EPs, in the PT-broken phase, the CPA-laser points also brings about intriguing functionalities. In particular, CPA satisfies the boundary condition of a particular scattering problem with no outgoing waves, giving rise to perfect absorption when sound incidents form both sides $[156, 159]$ . On the other hand, its time reversal counterpart, i.e. laser point, engenders an unidirectional wave amplifier (acoustic lasing) $[122, 124]$ . At the CPA-laser point, Lan et al further realized acoustic logic gates and amplifier $[158]$ . In this way, CPA-laser point has been exploited as a novel way to control sound amplification, absorption and signal gating.

## 4.5. Non-Hermitian acoustics in metasurfaces (2D)

As has been demonstrated, incorporating non-Hermitian physics into 1D scattering systems (such as waveguides) offers significant potential for manipulating waves in a unique and unprecedented way. Nevertheless, this unconventional approach for controlling waves primarily emphasizes the modulation of amplitude, as evidenced by the manipulation of transmission, reflection, and absorption coefficients. Introducing non-Hermiticity to higher-dimensional scattering systems could markedly expand the scope for shaping wave fields. Acoustic metasurfaces, as a higher-dimensional scattering system, constitute an ideal platform for leveraging the rich physics of non-Hermiticity, thanks to their open characteristics (radiation loss) and the thermoviscous effect in meta-units (dissipation loss).

To begin, we note that conventional lossless GPMs exhibit angular-symmetric scattering behavior under opposite oblique incidences $(\pm\theta_{i})$ to the surface normal, though the two incident waves experience fundamentally different physics [98]. Compared to a certain incidence $(\theta_{i})$ , the scattering from the opposite one $(- \theta_{i})$ is a higher-order diffraction involving multiple reflections within the metasurface. As a result, it can be expected that there will be a stark contrast in the scattering behavior under opposite incidences once loss is induced, since multiple reflections in meta-units significantly enhance sound absorption. This was shown by a transmission-type GPM with tailored losses in all meta-units, where the anomalous refraction is dramatically reduced under oblique incidence while that under opposite incidence is only moderately decreased [175] (figure 10(a)). This indicates a significant asymmetric scattering brought about by non-Hermiticity, which can be observed in a wide range of angles [177, 178]. By leveraging loss modulations, non-Hermitian metasurfaces show not only asymmetric refraction with respect to the surface normal [175, 177, 178], but also with respect to the surface tangential (space above and below the metasurface) [179, 180]. Moreover, transmission-type non-Hermitian metasurfaces show intriguing capabilities including, but not limited to reflection immunity (significantly enhanced transmission efficiency by suppressing the back scattering through impurity-embedded [181] or impedance-mismatched lossy [169] media) as well as negative refractions [160, 182].

Non-Hermiticity also empowers reflection-type acoustic metasurfaces, giving rise to anomalous reflection and strong absorption for opposite incident angles $[65, 183–185]$ . Building upon earlier studies on lossy transmission-type

(a)
![](images/595f681b947d3fe3a7c4b0f9c45a771343c8e884afeb842ab46c8b5f019307d0.jpg)
(b)

![](images/8692a7f0299a397cfdb5b336a31d44d7586f6011853eb7b211a076fe136f2565.jpg)

(c)
![](images/0d8ebc7f3b464a5695e5b7fc78cbb93473b604d285a534a3ccd64dca47ebb861.jpg)
Figure 10. (a) A non-Hermitian transmission-type metasurface with the non-Hermiticity induced by loss in the meta-units. The upper panel shows the asymmetric transmission behavior when waves obliquely come from left and right. The anomalous refraction is dramatically reduced under right-side incidence while that under left-side incidence is only moderately decreased. The lower panel shows the sample of the metasurface. (b) Asymmetric reflection from a non-Hermitian metasurface at the EP. The upper panels show the sample of metasurface constructed by five rigid-end grooves and a leaky groove in a period (left) and the experimental setup (right). The lower panels show the numerical and experimental results of scattering fields. Left-side incidence causes strong retroreflection (left) while right-side incidence results in near-perfect absorption. (c) A sound-absorbing non-Hermitian metasurface which utilizes the mechanism of multiple-reflection-enhanced absorption. Panel (a) is reprinted (figure) with permission from [175], Copyright (2017) by the American Physical Society. Panel (b) is reprinted (figure) with permission from [65], Copyright (2019) by the American Physical Society. Panel (c) is reprinted (figure) with permission from [176], Copyright (2018) by the American Physical Society.

acoustic metasurfaces [175], Wang et al further demonstrated a reflection-type non-Hermitian acoustic metasurface and observed its extreme behavior at the EP [65] (figure 10(b)). Compared to previous non-Hermitian works where all meta-units are designed to be lossy [175, 183–185], only one meta-unit is induced with a small amount of loss in this work. It is found that such a tiny loss is negligible for positive-angle incidence, leading to a highly efficient $r \rightarrow 1$ retroreflection, while it dramatically changes the system behavior when sound comes from the opposite side of the surface normal (negative-angle incidence), resulting in a near-perfect absorption $(r \rightarrow 0)$ [65].

Since non-Hermitian reflection-type metasurfaces engender strong absorption under certain incidence $[65, 183–185]$ , they can serve as 2D angle-dependent acoustic absorbers $[179]$ . As shown by $[65]$ , perfect absorption can be achieved with a small amount of loss in only one meta-unit, when the wave incidents from a certain angle. In this way, non-resonant yet highly-efficient sound absorbers can be achieved based on non-Hermitian metasurfaces $[186]$ , which leverage the strongly enhanced wave-matter interaction arising from multiple internal reflections during the higher-order diffraction process $[176, 187]$ (figure 10(c)). Several works have also attempted to broaden the bandwidth of sound absorption using this mechanism $[187, 188]$ .

Non-Hermitian metasurfaces not only demonstrate their advantageous properties in extreme wave redirecting, but also facilitate wave front shaping. Compared to those traditional lossless metasurfaces (even those non-local ones) whose meta-units provide only phase modulations, losses encoded in the non-Hermitian metasurfaces intrinsically offer a new dimension to control waves with both phase and amplitude. Typically, the phase and amplitude modulations supported by a meta-unit are interdependent and can therefore impede the overall performance. Zhu et al [189] showed that it is possible to control the reflected wave field with independent and arbitrary reflection amplitude and phase, and they demonstrated multi-plane hologram using non-Hermitian acoustic metasurfaces outperforming the ones based on conventional phase-controlled methods (figure 11(a)). Such non-Hermitian metasurfaces further enable acoustic holograms via simultaneous amplitude-phase control at multiple planes [190] or multiple frequencies [191, 192].

Non-Hermitian metasurfaces are also capable of focusing and splitting with either direction-selectivity $[146, 179, 194, 195]$ or enhanced focus intensity $[181, 196]$ . Given that a 1D non-Hermitian lattice supports unidirectional reflectionlessness at the EP, the unidirectional focusing can be achieved by curving the original planar profile of a 1D structure $[146]$ (figure 11(b)). The authors from the same group further showed a directional acoustic beam splitter, which splits obliquely incident sound from a specific side but is totally transparent for sound incident from the opposite side $[193]$ (figure 11(c)).

## 4.6. Non-Hermitian acoustics in topological systems

In this section, we briefly discuss non-Hermitian acoustics in topological systems, as this topic could lead to future development of acoustic metasurfaces. There is a close tie between non-Hermitian physics with another emerging field, topological physics. Both concepts were originated from quantum mechanics and are now being extensively studied in classical wave systems. Topological systems are characterized by the topological phase and often associated with topological protection against disorders $[197–202]$ . A typical 1D non-Hermitian chain can be constructed by periodically

![](images/69994c9505283cd8a5802d45888590be1a370d5fcf3103583a66fe15e57b16f0.jpg)
Figure 11. (a) Acoustic hologram based on non-Hermitian metasurfaces. Schematic diagram of a metasurface where leaky loss is used to introduce the non-Hermiticity and to modulate the amplitude (left panel) and the 3D illustration of a meta-unit (middle panel). The simulated and experimentally measured holographic images based on the non-Hermitian metasurface (right panel). (b) Schematics of directional wave focusing based on passive PT-symmetric metasurfaces (left panel), and the simulated and measured acoustic energy density fields for the forward and backward incidences at 3000 Hz, demonstrating unidirectional sound reflection and focusing (right panel). (c) Schematic of the PT-symmetric multilayered metasurface (left panel). The single-sided acoustic beam splitting effect is illustrated by the simulated acoustic pressure and energy density fields (middle panel), as well as the pressure amplitude distributions at the input and output ports when waves incident from top and left (right panel). The metasurface is transparent to incident acoustic waves from port $I_{2}$ (blue and white arrows) but works as a beam splitter for those from port $I_{1}$ (red arrows). Panel (a) is reproduced from [189]. CC BY 4.0. Panel (b) is reprinted (figure) with permission from [146], Copyright (2018) by the American Physical Society. Panel (c) is reprinted (figure) with permission from [193], Copyright (2020) by the American Physical Society.

repeating the minimal model, which is similar to the classical 1D Su–Schrieffer–Heeger (SSH) model used to study topological physics in 1D lattices [203, 204]. Both 1D models can be treated as two-level systems with unbalanced modulation on the two sublattices (noted as A and B). For example, for non-Hermitian systems, this unbalanced modulation can be detuned loss/gain on A and B ( $\gamma_{A} \neq \gamma_{B}$ ) or detuned forward and backward coupling ( $\kappa_{AB} \neq \kappa_{BA}$ ), while for the SSH model, this unbalanced modulation is the detuned intercell and intracell coupling ( $\kappa_{A_{n}B_{n}} \neq \kappa_{B_{n}A_{n+1}}$ , n referring to the nth cell in the lattice). Non-Hermiticity, thus, in this regard, can find its connection with topological physics, and topological modes (TMs) and EPs can be simultaneously observed in the same system (but under different boundary conditions) [205].

Despite the similarity between non-Hermitian and topological systems, fundamental differences exist. Traditional topological systems are commonly assumed to be Hermitian, featuring a real-valued spectrum, which is in stark contrast to the non-Hermitian ones. Non-Hermitian systems thus have unique topological properties not attainable in their Hermitian counterpart, owing to their complex-valued spectra $[140, 207–209]$ . Non-Hermiticity, therefore, challenges the conventional wisdom of bulk-boundary correspondence, a fundamental principle of topological physics established for Hermitian systems $[151, 152, 208–211]$ , while bringing about new perspectives on topological physics and new ways to control waves. We briefly introduce the works on acoustic non-Hermitian topological systems. For those who are interested, there are comprehensive reviews available on non-Hermitian topological physics $[133, 167, 212]$ and topological acoustics $[200–202]$ .

Acoustic non-Hermitian topological systems can be mainly classified into two categories. The first category involves the inclusion of non-Hermitian features in existing acoustic topological systems, which modifies the topological states in the presence of non-Hermiticity. For example, for an acoustic Weyl semimetal, the Weyl points will spread to Weyl exceptional rings in the presence of non-Hermiticity introduced by gain and loss $[144, 213]$ . Incorporating non-Hermiticity also modifies the bulk-boundary correspondence for topological insulators, leading to the so-called non-Hermitian skin effect (NHSE) which has been actively researched in the following areas of acoustics: (1) NHSE modifies the wave functions of TMs by adding directional responses, leading to chiral transportation $[214]$ and directional amplification/decay of TMs $[129, 215]$ ; (2) NHSE endows the topological edge modes with further spatial confinement, where directional transport owing to NHSEs results in wave accumulated at certain corners and hence forming a higher-order topological insulators supporting corner states $[153, 216, 217]$ (figures 12(a) and (b)); (3) NHSE can modify the wave functions of TMs by delocalization, which leads to a topological state spreading in the bulk and hence is called the Extended State in a Localized Continuum $[218]$ ; (4) More generally, there is a competition between NHSEs and TMs, and thus NHSE can be used to arbitrarily morph the wavefunctions $[154]$ .

The second category encompasses topological systems build by non-Hermiticity alone. Non-Hermicity not only can modify an existing topological system, but also can be used to build a topological system $[206, 219–221]$ . Gao et al showed that topological edge states can be induced solely by non-Hermiticity in a passive 1D acoustic crystal, which is a chain of coupled resonators with tunable loss $[219]$ . By extending this concept into 2D, they further showed a non-Hermitian higher-order topological insulator, where the unequal loss opens a topological bulk bandgap that is populated with gapped edge states and in-gap corner states (figures 12(c)–(e)) $[206]$ .

![](images/bf0faeb6f632c60ede562b7b8ec2715d4ce0f6609c253b8f29ff51f283a39708.jpg)

(a)
![](images/ca04719befe7f9868055a0c41dd6ca100a5cb652da480aa56192f4bbec11ef9d.jpg)

![](images/cc7d780ee663b508b95c9d84268760245b29fe2dbe07a5323bcfedc47d4d421e.jpg)
(c)

![](images/4c1d56266891fe6987f15cedf57cd745cbdf003236c859c8d91d01922fc2f39e.jpg)

(d)
![](images/e7638554455d5909324b39d886ca43fb8d04164ca066cb0f3e9b3bcbce453f54.jpg)

(e)
![](images/b9b12dd5a8fb1b31e4904faa5767944d3b2547c4ebfba26678f1be67ea904fc4.jpg)

![](images/83d8491714206fa6ea7357ae943ad8f71c6834529fcf85da1daed28867bedcd9.jpg)

![](images/7dbde9d581ac7f3c25f6da598853d4f30f640d74d9a362da9e43b558d0802162.jpg)
Figure 12. (a), (b) A non-Hermitian topological system where NHSE endows the topological edge modes with further spatial confinement. (a) Calculated eigen-spectrum for an acoustic topological insulator (without loss induced) with open boundary conditions in both the x and y directions (left panel). There are eight corner states (dark red), sixteen edge states (dark blue), and eight bulk states (gray). The corresponding acoustic wavefunctions for the bulk, edge, and corner states (marked by arrows) are presented (right panel). (b) The same as (a), but with acoustic loss induced. The higher-order non-Hermitian skin effect is manifested by the emergence of spin-polarized, corner-like skin-modes. (c)–(e) Topological edge states induced solely by non-Hermiticity in 2D acoustic crystals. (c) Photo of a 3D printed lattice with $12 \times 12$ resonators. (d) Measured acoustic intensity spectra for the sample in (c). Red, yellow, and blue curves represent the corner, edge, bulk spectra measured at the resonators labeled as '1', '2', and '3' in (c). The energy of the corner state is significantly higher than the edge state and the bulk state. (e) Measured intensity profiles at the peaks of the corner, edge, bulk spectra, denoted by the red (2142 Hz), yellow (2164 Hz), and blue (2170 Hz) dashed lines in (b), respectively. Both the height and color of each bar indicate the power strength. Panels (a) and (b) are reproduced from [153]. CC BY 4.0. Panels (c)–(e) are reproduced from [206]. CC BY 4.0.

## 5. Non-local and non-Hermitian acoustic metasurfaces

Acoustic metasurfaces incorporating both non-local and non-Hermitian effects, similar to their counterparts in electromagnetics $[222]$ , have surpassed those with only one type of effect, opening up new avenues for manipulating sound. For 1D cases, a non-Hermitian acoustic system usually features onsite modulations, i.e. local gain and/or loss. The cases where such modulation is non-local have also been investigated. For a non-Hermitian elastic waveguide with piezoelectric feedback actuation, the voltage applied to each PZT actuator is proportional to the voltage detected by a sensor a few units away. This is referred to as a non-local feedback scheme $[223]$ . The induced non-locality splits the Bloch bands into multiple bands with interchanging wave attenuation or amplification behavior that is tunable by the non-locality of the feedback scheme $[224]$ .

In the following, we conduct a systematic review of the efforts made in combining non-local and non-Hermitian effects for sound manipulations. We categorize these acoustic metasurfaces based on their primary functionalities, which are either sound scattering or sound absorption, depending on whether they are designed for sound field shaping (through tailored reflection, refraction, focus, etc) or sound energy elimination, respectively. However, it is essential to acknowledge that a strict differentiation between scattering and absorbing metasurfaces may not always be feasible. This is because introducing non-Hermiticity in the form of loss may inherently provide a scattering metasurface (e.g. GPM) with the ability to absorb sound. Nevertheless, the classification approach employed here offers an intuitive interpretation of the underlying working mechanisms, which is applicable to the majority of non-local and non-Hermitian acoustic metasurface.

## 5.1. Non-local and non-Hermitian scattering metasurfaces

The synergy between non-locality and non-Hermicity in higher-dimensional acoustic systems creates even more intriguing opportunities for wave manipulation. An example of this is a non-Hermitian (lossy) metasurface with fluid-structure coupling among its meta-units, which is a non-local effect. This metasurface allows for simultaneous phase and amplitude modulation of its meta-units, resulting in precise manipulation of waterborne sound $[225]$ . Another example is the non-local and non-Hermitian acoustic metasurfaces enabling high-efficiency negative refraction $[226]$ .

A 1D PT-symmetric acoustic lattice is typically built by periodically assigned gain and loss units. When such a bipartite lattice is subject to incident waves, it in fact resembles an acoustic grating, giving rise to asymmetric diffractions (distinct responses for opposite angles of incidence $\pm\theta_{i}$ ) based on the specific PT-symmetric configuration involved, including passive PT-symmetry [227]. Following the same analogue, asymmetric diffractions can also arise from an acoustic grating with unbalanced loss/gain modulation, and hence they can be observed in a general non-Hermitian metagrating without PT-symmetry [228]. A particular case is a transmission-type lossless metagrating that allows sound coming from two opposite angles to be either perfectly transmitted or retroreflected [79]. This metagrating can be also viewed as a reflection-type surface by considering only the plane on the incident wave side and the transmitted wave as a form of leaky loss from which non-Hermicity is introduced.

An acoustic metasurface can host multiple diffraction orders. Such a metasurface can be intrinsically characterized by an S-matrix whose size is dependent on the number of diffraction orders. An S-matrix whose size is higher than $2 \times 2$ serves as a versatile platform for exploiting higher-order EPs that have attracted increased attention in recent years due to their enhanced sensitivity and distinct topological features [135, 229]. However, to achieve higher-order EPs using GPMs is very challenging, if possible at all [65]. This is due to the fact that GPMs do not offer the flexibility to distribute the energy arbitrarily among different diffraction orders, because fundamentally, the diffraction order is exclusively determined by a single degree of freedom, i.e. the phase-gradient term $d\varphi/dx$ (see equation (3) as an example for the reflection-type GPM). Hence, they lack precise and simultaneous control over the multiple diffraction modes depicted by the corresponding S-matrices, which is instrumental for achieving higher-order EPs. In contrast, metagratings are much better equipped to provide a full control of multiple diffraction modes. For example, Fang et al showed that lossy acoustic metagratings can be used to engineer the S-matrix as a way to explore higher-order EPs in free space [134] (figure 13(a)).

(a)
![](images/dd0b1ea369eebf6b598d93233054b59a0bc9fc021c671e7ac88de43110d705e1.jpg)
Figure 13. (a) Schematic diagram of a non-local and non-Hermitian acoustic metagrating at the higher-order EP (upper panel). The metagrating supports the left, normal and right channels. At the higher-order EP, the metagrating retro-reflects the wave incident from left, but almost completely absorbs those from the other two directions. Experimental demonstration of the non-local and non-Hermitian acoustic metagrating exhibiting a higher-order EP (lower panel). (b) Non-reciprocal metagratings (upper left) without and (upper right) with bias flow induced. (Lower panels) Without the flow, the reflected power is steered symmetrical towards left and right channels for a normally incident wave, whereas when the flow velocity is $3\mathrm{ms}^{-1}$ , the reflection shows a non-reciprocal behavior. Panel (a) is reproduced from [134]. CC BY 4.0. Panel (b) is reproduced from [230]. CC BY 4.0.

We have pointed out that the synergy between non-locality and non-Hermiticity could endow the metasurface with extreme wave manipulation capabilities such as ultra-high-contrast directional wave anomalous diffraction, i.e. perfect reflection/refraction under certain inputs while near-perfect absorption for other inputs. This type of behaviors are characterized by an exotic EP, named perfect EP [134, 231]. In stark contrast to a conventional EP which mathematically requires a zero response from one input and a nonzero response from another input, the perfect EP has an extreme nonzero response, reaching $100\%$ output in efficiency. This behavior is extraordinary since for conventional non-Hermitian metasurfaces, it is extremely challenging to have near-perfect reflectivity/transmission owing to the inevitable absorption from the induced loss. Therefore, for a system with a perfect EP (2nd-order EP for example), the induced loss should be so insignificant that nearly no absorption can be observed under a certain input; whereas, perfect absorption under the opposite input should be realized merely relying on such tiny loss, which resorts to dramatically enhanced wave-matter interaction arising from, for example, multiple reflections [65] or bound states in the continua (BICs) [231].

Non-Hermiticity characterized by equation (19) can be constructed in different ways. Besides unbalanced gain and loss (detuned $\gamma_{1}$ and $\gamma_{2}$ ), nonreciprocal hopping (detuned $\kappa_{12}$ and $\kappa_{21}$ ) is another means, though less studied, that has been explored by researchers [154]. The acoustic wave equation for a linear and time-invariant medium exhibits a fundamental property known as reciprocity, which states that the pressure remains the same if the source and receiver simply exchange their locations. Three common approaches to break reciprocity in sound propagation are based on using acoustic nonlinearity [232, 233], flows [234], and time-modulation [235–237]. For the second approach, introducing a moving medium locally [238] in acoustic metasurfaces results in two-faced and independent wavefront manipulations (acoustic focusing, sound absorption, acoustic diffusion, and beam splitting) for two opposite incidences. Whereas, inducing a moving medium globally [99] endows a metasurface with nonreciprocal reflection behavior between its multiple diffraction channels (with diffraction angles of $0^{\circ}$ and $\pm45^{\circ}$ ). The third approach imparts an effective unidirectional momentum in time to break the reciprocity. Non-Hermitian metasurfaces built in this way have been demonstrated for exotic sound steering from non-reciprocal mode transitions, showcased by unidirectional evanescent wave conversion and non-reciprocal upconversion focusing [239].

Non-reciprocity also enriches Willis-coupling-based metagratings. Traditional Willis coupling arising from geometrically asymmetric structures respects reciprocity. A nonreciprocal Willis coupling can be realized by geometrically symmetric structures with external flow. Quan et al [230] demonstrated this by a symmetric subwavelength scatterer biased by angular momentum introduced by air flow, which enables non-reciprocal wave redirection from a metasurface consisting of an array of such non-reciprocal Willis coupling scatterers (figure 13(b)). When these non-reciprocal Willis coupling scatterers are assembled in a 2D array, NHSE can be observed [240]. Even without external flow, asymmetric wave propagation can be realized by a static Willis-coupling metagrating with inversion-symmetry-breaking meta-units [90]. Moreover, when built by active scatterers, such active acoustic Willis metasurfaces show their advantages including, but not limited to, highly non-reciprocal broadband sound transport [241], efficient sound barrier with strong sound attenuation, bandwidth, and shielded volume [242], as well as independent control of the transmission and reflection of waves [243].

(a)

## 5.2. Non-local and non-Hermitian absorbing metasurfaces

Acoustic scattering metasurfaces, including the traditional ones such as GPMs, are in general non-Hermitian systems, whose non-Hermiticity stems from radiation loss of meta-units enabling energy exchange with the surrounding environment. On the other hand, sound absorber is another type of non-Hermitian system where non-Hermiticity comes from dissipation. To effectively reduce noise, sound absorbers have been extensively studied (see detailed reviews on acoustic absorbers $[244–247]$ ). Conventional sound-absorbing metasurfaces usually combine several detuned resonators to achieve broadband sound absorption. Specifically, this can be realized by combing multiple Fabry–Pérot (FP) channels $[25, 248, 249]$ or HRs $[250–252]$ . While these acoustic metasurfaces can achieve a good amount of sound absorption, they encounter the same issue as traditional scattering metasurfaces. For sound absorption, the meta-units in a conventional metasurface are carefully tailored to provide strong resonances under the critical coupling condition to enable a high absorption at the target frequency. In this way, these sound-absorbing meta-units are designed individually, and then assembled in a particular pattern to form the metasurfaces $[244]$ . As a result, the local design strategy does not take into account the coupling among these meta-units, which can have a substantial impact on the overall response of the absorbing metasurface $[253]$ . This impact is typically manifested by lower efficiency or shifted absorption peaks. This issue becomes a major barrier for the pursuit of sound absorbers with consistently near-unity sound absorption within a broadband frequency.

The last decade has witnessed the tremendous success of sound absorbing metasurfaces by harnessing the non-local coupling effect among their meta-units. In fact, even before the very first paper on acoustic non-local metasurface was proposed $[46]$ , non-local coupling between lossy resonant units for sound absorption has been proposed $[254]$ (figure 14(a)). This study demonstrated that by coupling two distinct resonators, a hybrid resonance mode is formed, resulting in an absorber (with a thickness of 40 mm) that yields over 99% energy absorption at the central frequency of 511 Hz and has a 50% absorption bandwidth of 140 Hz. The absorption peak observed is not due to the resonance of either of the individual resonators, but rather arises from a combined effect of both resonators. Such a strategy can be extended to more absorption peaks by combing a greater number of coupled resonators $[257]$ .

Similar to some non-local scattering metasurfaces $[62, 63]$ , the coupling effect can be strengthened by a physical channel connecting neighboring meta-units in a non-local sound-absorbing metasurface. One such example is a metasurface comprising multiple FP channels with a bridge structure attached to the channel array and hence connecting all the channels $[255]$ . Here, the non-locality is introduced by the bridge structure (figure 14(b)). The authors compared the cases with and without the bridge structure, revealing that non-locality mainly have three advantages: optimization of the effective acoustic impedances toward impedance matching conditions, frequency shift of the FP resonance, and enhancement of coupling effects between neighboring meta-units. The advantages outlined above contribute to the enhanced bandwidth and efficiency of the acoustic meta-absorber, resulting in ultrabroadband sound absorption (with an average absorption coefficient exceeding 0.9 across a range from 600 to 2600 Hz) despite its comparatively small thickness (6.8 cm, or $\lambda/9$ for the lowest frequency).

![](images/30d2f7c9afaea0f165f68ff05b12cc91d4c4d1334201c176055eb807a8414238.jpg)

(b)
![](images/1ce54d576196d77d7b04c58042002aea7d07356bf84151c97214c0b4ea318299.jpg)

![](images/f5438081a763b352409c9e325e732e0551a5ee3c6a9ab12d4910024b633f497d.jpg)
(c)

![](images/8d7ffc05eb5aaf255c3643bd6611b7047dee44914372362876f90a6d60dbab3a.jpg)

![](images/bc29d75ec0cc949b0ecbfa8a0f23e24bc3d6e51062f53878dc090ea3a3626d47.jpg)

![](images/8b9a4274ac084e8be13da992a87bd97097abee2304acd600d379054951043b51.jpg)

![](images/dd2720cb4ca9fb4e1ea9b8b5fed3afd6675c616119b67687853747a08b1cd251.jpg)

![](images/5f7c485fa8e7abc01b3aac1fd2a72d2587975efc5aadb2b651e87e7c07ee681e.jpg)
Figure 14. (a) Pressure field and velocity field of a pair of coupled resonators at the hybrid resonance mode. The high absorption is not due to the resonance of either of the individual resonators, but rather arises from a combined effect of both (upper panel). The reflection coefficient, absorption coefficient, and both the real and imaginary parts of the impedance of the absorber (lower panel). (b) Multi-channel sound absorbers whose meta-units are without bridge (O, local metasurface), with cavity (C, local metasurface), and with bridge (B, non-local metasurface), respectively (upper panel). The simulated and experimental absorption coefficients for the O/C/B cases for the frequency range of 500–3000 Hz (lower panel). (c) A non-local metasurface formed by 25 imperfect resonators (upper left). Complex frequency plane illustration of the metasurface (upper right). The calculated (lines) and measured (circles) acoustic resistance and reactance of the metasurface (lower left). Theoretical (black line) and experimental (black circles) absorption coefficients of the hybrid metasurface (lower right). The colored lines represent the calculated absorption coefficients of the 25 individual NEHRs, respectively. These colored lines indicate low absorption from each individual resonator; however, the coherent coupling among these resonators results in a substantial overall enhancement of absorption. This enhanced coupling achieves quasi-perfect absorption within the frequency range of 297–475 Hz. Panel (a) is reprinted from [254], with the permission of AIP Publishing. Panel (b) is reprinted (figure) with permission from [255], Copyright (2021) by the American Physical Society. Panel (c) is reprinted from [256], Copyright (2020), with permission from Elsevier.

On the other hand, the inherent radiation coupling can be also utilized for tuning the overall performance of a sound-absorbing metasurface. In a study by Peng et al [250], a sound-absorbing metasurface was developed using periodically arranged supercells consisting of honeycomb units. Each honeycomb unit was designed as a HR with the same cavity geometry, but varying neck sizes. Such a design can achieve 90% sound absorption from 600 to 1000 Hz with a thickness less than 30 mm. The authors pointed out that the broadband sound absorption stems from the radiation coupling between different meta-units rather than merely the superposition of the individual behavior of each meta-unit [250]. Radiation coupling has been also investigated for sound absorption by Huang et al [256]. The authors showed that, by leveraging the radiation coupling, a quasi-perfect broadband sound absorber can be realized without utilizing prefect resonators (a resonator that can achieve perfect sound absorption at its resonance frequency), but rather with coherently coupled ‘weak resonances’ (resonators with low sound absorption peaks) (figure 14(c)). In this way, a metasurface consisting of embedded-necked Helmholtz resonators (ENHRs) was designed which resulted in an averaged absorption coefficient of 0.957 for a frequency band of 870–3224 Hz with a metasurface thickness of only 39 mm. This design is analogous to a competitive football team that does not rely solely on the individual skills of its players, but rather on their ability to collaborate effectively. Such a strategy of coupling imperfect resonators has been widely adopted in various studies [256, 258–263]. In particular, by coherently coupling modified ENHRs, Zhou et al further achieved a ultrabroadband absorption with an average sound absorption of 0.93 from 320 Hz to 6400 Hz with a metasurface thickness of only 100 mm [262]. For these non-local absorbers, coupling between over-damped modes has shown its superiority to under-damped or critical-coupled states in sound absorption [262, 264]. Moreover, coupled resonators via radiation can be adopted for the grazing-incidence case, showcased by an acoustic meta-liner for abating tunnel noise with a causality-governed minimal thickness [265].

## 6. Concluding remarks and outlook

In this review, the latest progress of acoustic metasurfaces is introduced, with the emphasis on their underlying working mechanisms involving non-locality and non-Hermiticity. For a self-contained presentation, we first introduce the fundamental principle of traditional metasurfaces in section 2, which also provides the backdrop for non-local and/or non-Hermitian acoustic metasurfaces. Section 3 is focused on the drawback of traditional metasurfaces in neglecting the non-local effect among meta-units, and the approaches that have been put forward to address this issue. The non-local effect from radiation coupling among the meta-units inevitably affects the performance of a traditional metasurface following the ‘individually-designed-then-assembled’ paradigm. Rather than trying to mitigate the non-local effect, recent efforts have embraced it, resulting in the development of non-local metasurfaces that can redirect waves with high efficiency, surpassing traditional metasurfaces. In section 4, we discuss another limitation of traditional metasurfaces, which is their failure to account for dissipation that may occur due to the thermoviscous effect within subwavelength meta-units. This can undermine the foundation of traditional metasurfaces, which are based solely on the framework of phase modulation. Such intrinsic dissipation in acoustics, however, is now being harnessed to empower acoustic metasurfaces as a new design dimension, resulting in highly asymmetric responses for reflection, transmission, and absorption. Section 5 centers on recent attempts to merge non-local and non-Hermitian effects in acoustic metasurfaces, which have led to even more intriguing wave behaviors.

Due to the unparalleled level of sound control achieved with surface-confined components, non-local and non-Hermitian acoustic metasurfaces hold the potential to expand into even broader horizons beyond those discussed in this review article. As a rapidly expanding area of research, acoustic metasurfaces are poised to have a significant impact on several promising yet largely unexplored fields, which we outline below.

(1) Topological acoustic metasurfaces Incorporating topological physics into acoustic metasurfaces may provide yet another interesting avenue to enrich our capabilities to manipulate the scattering of sound. A 2D topological acoustic lattice usually only considers the in-plane (tangential to the lattice surface) sound propagation, and are commonly assumed to be Hermitian. While a few works have recently studied non-Hermitian acoustic topological systems, the non-Hermiticity is introduced by dissipation loss [206]. We expect that radiation loss, which involves out-of-plane sound propagation, can provide an alternative approach to induce non-Hermiticity in a topological acoustic system. This can be essentially viewed as a leaky topological metasurface interacting with the surrounding environment. The interplay between topological states and the background medium has led to a series of exciting phenomena [266–268]. By incorporating the topological concept into a metasurface, we may pave the way for robustly rerouting wave scattering [269]. One possible challenge is that radiation loss, not only contributes to non-Hermiticity, but also plays a role in coupling among meta-units (as in radiation coupling). A unified framework that can precisely characterize loss and coupling due to radiation would be a key to the success of this development.

(2) Singular acoustic metasurfaces Extremely sensitive anomalous sound scattering is another area that could be explored by the next-generation acoustic metasurfaces. It is known that, for realistic physical systems, the occasional occurrence of singularities usually gives birth to exotic characteristics. There are typically two types of singularities, EPs and BICs. EP is a singular point related to the defective characteristic matrix (H-matrix or S-matrix), while BIC is a singular point in its far-field radiation phase $[270]$ . Enhanced sensitivity usually occurs when a metasurface approaching a singularity in its parameter space. Recent studies have demonstrated the extreme sensitivity of wave scattering around EP $[134, 231]$ as well as BIC $[271]$ . The emergence of these singular features facilitate wave diffraction with enhanced spectrum selectivity and hence could lead to promising applications for selective wavefront filtering and sensing. However, combing such singularities, especially those from BICs, in acoustic metasurfaces has been largely unexplored.

(3) Sound diffusers On the application front, sound diffusers can be also greatly benefited from the development of acoustic metasurfaces. While acoustic metasurfaces have already been used to design thinner sound diffusers $[31, 272]$ than commercial ones, these designs are still based on the conventional metasurface design paradigm, where non-locality and non-Hermitian are either absent or not fully taken advantage of. Taking inspiration from recent works regarding non-local acoustic metasurfaces, we envision that it is possible to achieve smaller and broader bandwidth sound diffusers by leveraging physical or radiation-based coupling. Furthermore, due to their performance requirements, sound diffusers naturally host a significant number of diffraction orders ( $\geqslant7$ ), making them an ideal platform for implementing higher-order EPs. This, in turn, could enable functionalities that are unattainable by conventional sound diffusers that do not consider the loss in the ‘equation’ during the design process. For example, we could envision a non-local and non-Hermitian metasurface that would only diffusely reflect the sound at a particular angle of incidence. At all other angles of incidence, most scattered energy will be suppressed through absorption, so that the diffuser behaves like a sound absorber. Such a highly unidirectional sound diffuser could be useful for a number of applications in studios and concert halls, among other spaces where control on the diffuse reflection is highly desirable. Major challenges for this effort, however, are the engineering of the large size S-matrix due to the high number of diffraction orders that need to be considered, as well as the precise measurements of 2D diffusers, which typically require an anechoic chamber and a 3D goniometer.

(4) Artificial-intelligence-empowered metasurfaces The intricate structural complexity of acoustic metasurfaces poses great challenges in precisely tuning the non-local and non-Hermitian effects. Nevertheless, achieving precise control over the non-locality and non-Hermiticity within the high-dimensional parameter space inherent in the advanced structure of acoustic metasurfaces is of paramount importance for accessing singular points like EPs and BICs. In the present day, the rapid advancement of artificial intelligence has the potential to turn these once-deemed ‘mission impossible’ tasks into reality $[15, 273, 274]$ . Deep learning methods, a subset of artificial intelligence, provide a promising avenue for optimizing computers’ inferential capabilities to account for non-local and non-Hermitian effects by ‘learning’ from past datasets or experiences. However, these strategies may face a challenge due to the limited training data size, as the data is typically obtained from costly simulations. As a solution, the integration of artificial intelligence with our current knowledge of non-local and non-Hermitian metasurfaces holds promise. The latter can serve as a physical constraint, enabling us to reduce the size of the training set and facilitate the learning process $[275]$ . We believe that this integration will undoubtedly invigorate research and promote the development of interdisciplinary approaches in this field.

(5) Active and tunable acoustic metasurfaces Although non-local and/or non-Hermitian metasurfaces hold promise for the extraordinary manipulation of acoustic waves, their practical applicability in dynamic acoustic conditions remains challenging. Once these complex meta-units are constructed, they lack the ability to be readily adjusted to accommodate changes in the acoustic environment, such as variations in incident angles, frequencies, or wavefronts. Active and tunable meta-units have demonstrated significant capabilities in acoustic crystals and metamaterials $[276]$ . Current attempts to include active and tunable elements in acoustic metasurfaces involve designing mechanical structures and units that can achieve local structural adjustment and limited task switching $[190, 277]$ . On the other hand, active meta-units naturally play a vital role in the field of non-Herimitian acoustics, for both achieving the acoustic gain medium to form a PT-symmetric system $[108–111]$ as well as for inducing bias flow to form a non-reciprocal system $[230, 234]$ . We anticipate that the capabilities of non-local and non-Hermitian metasurfaces can be significantly expanded by incorporating self-adaptive capabilities and coding-control features into these active meta-units. Notably, a recent study explored a metasurface that can detect acoustic wave properties, allowing for precise control and adjustment of its active elements to achieve optimal acoustic absorption at deep-subwavelength thickness $[278]$ . Looking ahead, we envision the emergence of next-generation metasurfaces that possess the remarkable ability to automatically adjust their properties and behaviors. This advancement would enable metasurfaces to be smartly controlled, revolutionizing the field of sound manipulation and acoustic applications

(6) Nonlinear acoustic metasurfaces In the realm of acoustic metasurfaces, the majority of exotic phenomena have been demonstrated within the linear acoustic region. However, it is important to note that the propagation of acoustic waves, governed by the Navier–Stokes equation, inherently possesses nonlinear characteristics. Although acoustic nonlinear effects are typically weak, specific conditions like resonances can render the nonlinear acoustic properties non-negligible, warranting exploration and exploitation. In traditional metasurfaces, efforts have been made to harness the nonlinearity of meta-units to achieve intriguing functionalities. Examples include second-harmonic tailoring and demultiplexing $[279]$ , frequency conversion $[280]$ , and ultra-low-frequency insulation $[281]$ . However, for existing non-local and non-Hermitian metasurfaces, the focus has predominantly been on their linear behavior, with relatively limited exploration of their nonlinear features. Therefore, we expect that by capitalizing on the sharp resonances present in metasurfaces, such as due to BICs, the nonlinear effects can be effectively harnessed within a small sample size and at a moderate input energy level. This endeavor could yield fruitful insights and applications in the realm of non-local and non-Hermitian physics.

In conclusion, while significant progress has been made in the development of acoustic non-local and non-Hermitian metasurfaces, challenges and opportunities still exist. We anticipate that the emergence of new ideas in the field of acoustic metasurfaces, beyond those reviewed here, will lead to breakthroughs and enable more powerful abilities for rerouting acoustic waves in a compact, flexible, and highly efficient manner.

## Data availability statement

All data that support the findings of this study are included within the article (and any supplementary files).

## Acknowledgments

Prof. Li thanks the support from the National Key R&D Program of China (Grant Nos. 2020YFA0211400 and 2020YFA0211402), the Fundamental Research Funds for the Central Universities (Grant No. 020414380195), the National Natural Science Foundation of China (Grant No. 12074286), and the Shanghai Science and Technology Committee (Grant No. 21JC1405600). Dr Wang thanks the support from the National Science Foundation of China (Grant No. 12074288). Dr Jing thanks the support from NSF CMMI (Grant No. 1951221).

## ORCID iD

Yong Li https://orcid.org/0000-0001-8049-9128

## References

[1] Schroeder M R 1975 Diffuse sound reflection by maximum-length sequences J. Acoust. Soc. Am. 57 149–50

[2] Schroeder M R 1979 Binaural dissimilarity and optimum ceilings for concert halls: more lateral sound diffusion J. Acoust. Soc. Am. 65 958–63

[3] Yablonovitch E 1987 Inhibited spontaneous emission in solid-state physics and electronics Phys. Rev. Lett. 58 2059–62

[4] Hussein M I, Leamy M J and Ruzzene M 2014 Dynamics of phononic materials and structures: historical origins, recent progress and future outlook Appl. Mech. Rev. 66 040802

[5] Martínez-Sala R, Sancho J, Sánchez J V, Gómez V, Llinares J and Meseguer F 1995 Sound attenuation by sculpture Nature 378 241

[6] Liu Z, Zhang X, Mao Y, Zhu Y Y, Yang Z, Chan C T and Sheng P 2000 Locally resonant sonic materials Science 289 1734–6

[7] Ma G and Sheng P 2016 Acoustic metamaterials: from local resonances to broad horizons Sci. Adv. 2 e1501595

[8] Liang Z and Li J 2012 Extreme acoustic metamaterial by coiling up space Phys. Rev. Lett. 108 114301

[9] Xie Y, Popa B I, Zigoneanu L and Cummer S A 2013 Measurement of a broadband negative index with space-coiling acoustic metamaterials Phys. Rev. Lett. 110 175501

[10] Shen C, Xie Y, Sui N, Wang W, Cummer S A and Jing Y 2015 Broadband acoustic hyperbolic metamaterial Phys. Rev. Lett. 115 254301

[11] Zhang S, Xia C and Fang N 2011 Broadband acoustic cloak for ultrasound waves Phys. Rev. Lett. 106 024301

[12] Cummer S A, Popa B I, Schurig D, Smith D R, Pendry J, Rahm M and Starr A 2008 Scattering theory derivation of a 3D acoustic cloaking shell Phys. Rev. Lett. 100 024301

[13] Cummer S A, Christensen J and Alù A 2016 Controlling sound with acoustic metamaterials Nat. Rev. Mater. 1 16001

[14] Ge H, Yang M, Ma C, Lu M H, Chen Y F, Fang N and Sheng P 2018 Breaking the barriers: advances in acoustic functional materials Natl Sci. Rev. 5 159–82

[15] Chen A L, Wang Y S, Wang Y F, Zhou H T and Yuan S M 2022 Design of acoustic/elastic phase gradient metasurfaces: principles, functional elements, tunability and coding Appl. Mech. Rev. 74 020801

[16] Gao N, Zhang Z, Deng J, Guo X, Cheng B and Hou H 2022 Acoustic metamaterials for noise reduction: a review Adv. Mater. Technol. 7 2100698

[17] Jin Y, Pennec Y, Bonello B, Honarvar H, Dobrzynski L, Djafari-Rouhani B and Hussein M I 2021 Physics of surface vibrational resonances: pillared phononic crystals, metamaterials and metasurfaces Rep. Prog. Phys. 84 086502

[18] Yu N, Genevet P, Kats M A, Aieta F, Tetienne J P, Capasso F and Gaburro Z 2011 Light propagation with phase discontinuities: generalized laws of reflection and refraction Science 334 333–7

[19] Assouar B, Liang B, Wu Y, Li Y, Cheng J C and Jing Y 2018 Acoustic metasurfaces Nat. Rev. Mater. 3 460–72

[20] Tang K, Qiu C, Ke M, Lu J, Ye Y and Liu Z 2014 Anomalous refraction of airborne sound through ultrathin metasurfaces Sci. Rep. 4 6517

[21] Li Y, Jiang X, Li R Q, Liang B, Zou X Y, Yin L L and Cheng J C 2014 Experimental realization of full control of reflected waves with subwavelength acoustic metasurfaces Phys. Rev. Appl. 2 064002

[22] Zhao J, Li B, Chen Z and Qiu C W 2013 Manipulating acoustic wavefront by inhomogeneous impedance and steerable extraordinary reflection Sci. Rep. 3 2537

[23] Xie Y, Shen C, Wang W, Li J, Suo D, Popa B I, Jing Y and Cummer S A 2016 Acoustic holographic rendering with two-dimensional metamaterial-based passive phased array Sci. Rep. 6 35437

[24] Melde K, Mark A G, Qiu T and Fischer P 2016 Holograms for acoustics Nature 537 518–22

[25] Yang M, Chen S, Fu C and Sheng P 2017 Optimal sound-absorbing structures Mater. Horiz. 4 673–80

[26] Li Y and Assouar B M 2016 Acoustic metasurface-based perfect absorber with deep subwavelength thickness Appl. Phys. Lett. 108 063502

[27] Dong R, Mao D, Wang X and Li Y 2021 Ultrabroadband acoustic ventilation barriers via hybrid-functional metasurfaces Phys. Rev. Appl. 15 024044

[28] Zhang C, Jiang X, He J, Li Y and Ta D 2023 Spatiotemporal acoustic communication by a single sensor via rotational Doppler effect Adv. Sci. 10 e2206619

[29] Wu K, Liu J J, Ding Y J, Wang W, Liang B and Cheng J C 2022 Metamaterial-based real-time communication with high information density by multipath twisting of acoustic wave Nat. Commun. 13 5171

[30] Dong H W, Shen C, Zhao S D, Qiu W, Zheng H, Zhang C, Cummer S A, Wang Y S, Fang D and Cheng L 2022 Achromatic metasurfaces by dispersion customization for ultra-broadband acoustic beam engineering Natl Sci. Rev. 9 nwac030

[31] Zhu Y, Fan X, Liang B, Cheng J and Jing Y 2017 Ultrathin acoustic metasurface-based schroeder diffuser Phys. Rev. X 7 021034

[32] Wang Q, del Hougne P and Ma G 2022 Controlling the spatiotemporal response of transient reverberating sound Phys. Rev. Appl. 17 044007

[33] Xie Y, Wang W, Chen H, Konneker A, Popa B I and Cummer S A 2014 Wavefront modulation and subwavelength diffractive acoustics with an acoustic metasurface Nat. Commun. 5 5553

[34] Sun S, He Q, Xiao S, Xu Q, Li X and Zhou L 2012 Gradient-index meta-surfaces as a bridge linking propagating waves and surface waves Nat. Mater. 11 426–31

[35] Liu B, Zhao W and Jiang Y 2016 Apparent negative reflection with the gradient acoustic metasurface by integrating supercell periodicity into the generalized law of reflection Sci. Rep. 6 38314

[36] Jiang X, Li Y, Liang B, Cheng J C and Zhang L 2016 Convert acoustic resonances to orbital angular momentum Phys. Rev. Lett. 117 034301

[37] Jin Y, Kumar R, Poncelet O, Mondain-Monval O and Brunet T 2019 Flat acoustics with soft gradient-index metasurfaces Nat. Commun. 10 143

[38] Li Y, Liang B, Gu Z M, Zou X Y and Cheng J C 2013 Reflected wavefront manipulation based on ultrathin planar acoustic metasurfaces Sci. Rep. 3 2546

[39] Li Y, Jiang X, Liang B, Cheng J C and Zhang L 2015 Metascreen-based acoustic passive phased array Phys. Rev. Appl. 4 024003

[40] Fan X D, Zhu Y F, Liang B, Yang J, Yang J and Cheng J C 2017 Ultra-broadband and planar sound diffuser with high uniformity of reflected intensity Appl. Phys. Lett. 111 103502

[41] Ward G P, Lovelock R K, Murray A R, Hibbins A P, Sambles J R and Smith J D 2015 Boundary-layer effects on acoustic transmission through narrow slit cavities Phys. Rev. Lett. 115 044302

[42] Henríquez V C, García-Chocano V M and Sánchez-Dehesa J 2017 Viscothermal losses in double-negative acoustic metamaterials Phys. Rev. Appl. 8 014029

[43] Jiang X, Li Y and Zhang L 2017 Thermoviscous effects on sound transmission through a metasurface of hybrid resonances J. Acoust. Soc. Am. 141 EL363

[44] Gerard NJ R K, Li Y and Jing Y 2018 Investigation of acoustic metasurfaces with constituent material properties considered J. Appl. Phys. 123 124905

[45] Shastri K and Monticone F 2022 Nonlocal flat optics Nat. Photon. 17 36–47

[46] Díaz-Rubio A and Tretyakov S A 2017 Acoustic metasurfaces for scattering-free anomalous reflection and refraction Phys. Rev. B 96 125409

[47] Estakhri N M and Alù A 2016 Wave-front transformation with gradient metasurfaces Phys. Rev. X 6 041008

[48] Asadchy V S, Albooyeh M, Tcvetkova S N, Díaz-Rubio A, Ra'di Y and Tretyakov S A 2016 Perfect control of

reflection and refraction using spatially dispersive metasurfaces Phys. Rev. B 94 075142

[49] Koo S, Cho C, Jeong J H and Park N 2016 Acoustic omni meta-atom for decoupled access to all octants of a wave parameter space Nat. Commun. 7 13012

[50] Lindell I V, Sihvola A H, Tretyakov S A and Viitanen A J 1994 Electromagnetic Waves in Chiral and Bi-Isotropic Media (Artech House)

[51] Willis J R 1985 The nonlocal influence of density variations in a composite Int. J. Solids Struct. 21 805–17

[52] Milton G W and Willis J R 2007 On modifications of Newton's second law and linear continuum elastodynamics Proc. R. Soc. A 463 855–80

[53] Sieck C F, Alù A and Haberman M R 2017 Origins of Willis coupling and acoustic bianisotropy in acoustic metamaterials through source-driven homogenization Phys. Rev. B 96 104303

[54] Muhlestein M B, Sieck C F, Wilson P S and Haberman M R 2017 Experimental evidence of Willis coupling in a one-dimensional effective material element Nat. Commun. 8 15625

[55] Li J, Shen C, Díaz-Rubio A, Tretyakov S A and Cummer S A 2018 Systematic design and experimental demonstration of bianisotropic metasurfaces for scattering-free manipulation of acoustic wavefronts Nat. Commun. 9 1342

[56] Su G and Liu Y 2020 Amplitude-modulated binary acoustic metasurface for perfect anomalous refraction Appl. Phys. Lett. 117 221901

[57] Li J, Song A and Cummer S A 2020 Bianisotropic acoustic metasurface for surface-wave-enhanced wavefront transformation Phys. Rev. Appl. 14 044012

[58] Díaz-Rubio A, Li J, Shen C, Cummer S A and Tretyakov S A 2019 Power flow-conformal metamirrors for engineering wave reflections Sci. Adv. 5 eaau7288

[59] Díaz-Rubio A and Tretyakov S 2020 Dual-physics metasurfaces for simultaneous manipulations of acoustic and electromagnetic waves Phys. Rev. Appl. 14 014076

[60] Peng X, Li J, Shen C and Cummer S A 2021 Efficient scattering-free wavefront transformation with power flow conformal bianisotropic acoustic metasurfaces Appl. Phys. Lett. 118 061902

[61] Yan P Y, Zhu X F, Chen D C and Wu D J 2021 Perfect multiple splitting with arbitrary power distribution by acoustic metasurfaces Europhys. Lett. 134 48003

[62] Quan L and Alù A 2019 Passive acoustic metasurface with unitary reflection based on nonlocality Phys. Rev. Appl. 11 054077

[63] Quan L and Alù A 2019 Hyperbolic sound propagation over nonlocal acoustic metasurfaces Phys. Rev. Lett. 123 244303

[64] Hou Z, Fang X, Li Y and Assouar B 2019 Highly efficient acoustic metagrating with strongly coupled surface grooves Phys. Rev. Appl. 12 034021

[65] Wang X, Fang X, Mao D, Jing Y and Li Y 2019 Extremely asymmetrical acoustic metasurface mirror at the exceptional point Phys. Rev. Lett. 123 214302

[66] Larouche S and Smith D R 2012 Reconciliation of generalized refraction with diffraction theory Opt. Lett. 37 2391–3

[67] Ra'di Y, Sounas D L and Alù A 2017 Metagratings: beyond the limits of graded metasurfaces for wave front control Phys. Rev. Lett. 119 067404

[68] Torrent D 2018 Acoustic anomalous reflectors based on diffraction grating engineering Phys. Rev. B 98 060101

[69] Ni H, Fang X, Hou Z, Li Y and Assouar B 2019 High-efficiency anomalous splitter by acoustic meta-grating Phys. Rev. B 100 104104

[70] Chen S, Fan Y, Yang F, Sun K, Fu Q, Zheng J and Zhang F 2021 Coiling-up space metasurface for high-efficient and wide-angle acoustic wavefront steering Front. Mater. 8 790987

[71] Hu Y, Zhang Y, Su G, Zhao M, Li B, Liu Y and Li Z 2022 Realization of ultrathin waveguides by elastic metagratings Commun. Phys. 5 62

[72] Li J and Pendry J B 2008 Hiding under the carpet: a new strategy for cloaking Phys. Rev. Lett. 101 203901

[73] Faure C, Richoux O, Félix S and Pagneux V 2016 Experiments on metasurface carpet cloaking for audible acoustics Appl. Phys. Lett. 108 064103

[74] Esfahlani H, Karkar S, Lissek H and Mosig J R 2016 Acoustic carpet cloak based on an ultrathin metasurface Phys. Rev. B 94 014302

[75] Jin Y, Fang X, Li Y and Torrent D 2019 Engineered diffraction gratings for acoustic cloaking Phys. Rev. Appl. 11 011004

[76] He J, Jiang X, Ta D and Wang W 2020 Experimental demonstration of underwater ultrasound cloaking based on metagrating Appl. Phys. Lett. 117 091901

[77] Song A, Sun C, Xiang Y and Xuan F Z 2022 Switchable acoustic metagrating for three-channel retroreflection and carpet cloaking Appl. Phys. Express 15 024002

[78] Zeng L S, Shen Y X, Fang X S, Li Y and Zhu X F 2021 Experimental realization of ultrasonic retroreflection tweezing via metagratings Ultrasonics 117 106548

[79] Cao S and Hou Z 2019 Angular-asymmetric transmitting metasurface and splitter for acoustic waves: combining the coherent perfect absorber and a laser Phys. Rev. Appl. 12 064016

[80] Li X, Dong D, Liu J, Liu Y and Fu Y 2022 Perfect retroreflection assisted by evanescent guided modes in acoustic metagratings Appl. Phys. Lett. 120 151701

[81] Xie H and Hou Z 2021 Nonlocal metasurface for acoustic focusing Phys. Rev. Appl. 15 034054

[82] Chiang Y K, Quan L, Peng Y, Sepehrirahnama S, Oberst S, Alù A and Powell D A 2021 Scalable metagrating for efficient ultrasonic focusing Phys. Rev. Appl. 16 064014

[83] He J, Jiang X, Zhao H, Zhang C, Zheng Y, Liu C and Ta D 2021 Broadband three-dimensional focusing for an ultrasound scalpel at megahertz frequencies Phys. Rev. Appl. 16 024006

[84] Fu Y, Cao Y and Xu Y 2019 Multifunctional reflection in acoustic metagratings with simplified design Appl. Phys. Lett. 114 053502

[85] Wang Y, Cheng Y and Liu X 2019 Modulation of acoustic waves by a broadband metagrating Sci. Rep. 9 7271

[86] Melnikov A, Koble S, Schweiger S, Chiang Y K, Marburg S and Powell D A 2022 Microacoustic metagratings at ultra-high frequencies fabricated by two-photon lithography Adv. Sci. 9 e2200990

[87] Quan L, Ra'di Y, Sounas D L and Alù A 2018 Maximum Willis coupling in acoustic scatterers Phys. Rev. Lett. 120 254301

[88] Chiang Y K, Oberst S, Melnikov A, Quan L, Marburg S, Alù A and Powell D A 2020 Reconfigurable acoustic metagrating for high-efficiency anomalous reflection Phys. Rev. Appl. 13 064067

[89] Bernard S, Chikh-Bled F, Kourchi H, Chati F and Léon F 2022 Broadband negative reflection of underwater acoustic waves from a simple metagrating: modeling and experiment Phys. Rev. Appl. 17 024059

[90] Craig S R, Su X, Norris A and Shi C 2019 Experimental realization of acoustic bianisotropic gratings Phys. Rev. Appl. 11 061002

[91] Fu Y Y, Tao J Q, Song A L, Liu Y W and Xu Y D 2020 Controllably asymmetric beam splitting via gap-induced

diffraction channel transition in dual-layer binary metagratings Front. Phys. 15 52502

[92] Su X and Banerjee D 2021 Asymmetric lateral sound beaming in a Willis medium Phys. Rev. Res. 3 033080

[93] Hou Z, Ding H, Wang N, Fang X and Li Y 2021 Acoustic vortices via nonlocal metagratings Phys. Rev. Appl. 16 014002

[94] Zhou H T, Fu W X, Wang Y F and Wang Y S 2021 High-efficiency ultrathin nonlocal waterborne acoustic metasurface Phys. Rev. Appl. 15 044046

[95] Cao J, Yang K, Fang X, Guo L, Li Y and Cheng Q 2021 Holographic tomography of dynamic three-dimensional acoustic vortex beam in liquid Appl. Phys. Lett. 119 143501

[96] Fu Y, Shen C, Zhu X, Li J, Liu Y, Cummer S A and Xu Y 2020 Sound vortex diffraction via topological charge in phase gradient metagratings Sci. Adv. 6 eaba9876

[97] Jimenez N, Groby J P and Romero-Garcia V 2021 Spiral sound-diffusing metasurfaces based on holographic vortices Sci. Rep. 11 10217

[98] Fu Y, Shen C, Cao Y, Gao L, Chen H, Chan C T, Cummer S A and Xu Y 2019 Reversal of transmission and reflection based on acoustic metagratings with integer parity design Nat. Commun. 10 2326

[99] Fan L and Mei J 2021 Acoustic metagrating circulators: nonreciprocal, robust and tunable manipulation with unitary efficiency Phys. Rev. Appl. 15 064002

[100] Fan L and Mei J 2020 Metagratings for waterborne sound: various functionalities enabled by an efficient inverse-design approach Phys. Rev. Appl. 14 044003

[101] Fan L and Mei J 2021 Multifunctional waterborne acoustic metagratings: from extraordinary transmission to total and abnormal reflection Phys. Rev. Appl. 16 044029

[102] Yu G, Qiu Y, Li Y, Wang X and Wang N 2021 Underwater acoustic stealth by a broadband 2-bit coding metasurface Phys. Rev. Appl. 15 064064

[103] Bender C M and Boettcher S 1998 Real spectra in non-Hermitian Hamiltonians having PT symmetry Phys. Rev. Lett. 80 5243–6

[104] Zhu X, Ramezani H, Shi C, Zhu J and Zhang X 2014 PT-symmetric acoustics Phys. Rev. X 4 031042

[105] Yi J, Ma Z, Xia R, Negahban M, Chen C and Li Z 2022 Structural periodicity dependent scattering behavior in parity-time symmetric elastic metamaterials Phys. Rev. B 106 014303

[106] Gu Z, Gao H, Cao P C, Liu T, Zhu X F and Zhu J 2021 Controlling sound in non-Hermitian acoustic systems Phys. Rev. Appl. 16 057001

[107] Gerard NJ R K and Jing Y 2020 Loss in acoustic metasurfaces: a blessing in disguise MRS Commun. 10 32–41

[108] Shi C, Dubois M, Chen Y, Cheng L, Ramezani H, Wang Y and Zhang X 2016 Accessing the exceptional points of parity-time symmetric acoustics Nat. Commun. 7 11110

[109] Fleury R, Sounas D and Alù A 2015 An invisible acoustic sensor based on parity-time symmetry Nat. Commun. 6 5905

[110] Li H X, Rosendo-Lopez M, Zhu Y F, Fan X D, Torrent D, Liang B, Cheng J C and Christensen J 2019 Ultrathin acoustic parity-time symmetric metasurface cloak Research 2019 8345683

[111] Rivet E, Brandstötter A, Makris K G, Lissek H, Rotter S and Fleury R 2018 Constant-pressure sound waves in non-Hermitian disordered media Nat. Phys. 14 942–7

[112] Christensen J, Willatzen M, Velasco V R and Lu M H 2016 Parity-time synthetic phononic media Phys. Rev. Lett. 116 207601

[113] Wu Q, Chen Y and Huang G 2019 Asymmetric scattering of flexural waves in a parity-time symmetric metamaterial beam J. Acoust. Soc. Am. 146 850

[114] Jin Y, Zhong W, Cai R, Zhuang X, Pennec Y and Djafari-Rouhani B 2022 Non-Hermitian skin effect in a phononic beam based on piezoelectric feedback control Appl. Phys. Lett. 121 022202

[115] Zhou Y, Yang Z Z, Peng Y Y and Zou X Y 2022 Parity–time symmetric acoustic system constructed by piezoelectric composite plates with active external circuits Chin. Phys. B 31 064304

[116] Hou Z and Assouar B 2018 Tunable elastic parity-time symmetric structure based on the shunted piezoelectric materials J. Appl. Phys. 123 085101

[117] Rosa M I N, Mazzotti M and Ruzzene M 2021 Exceptional points and enhanced sensitivity in PT-symmetric continuous elastic media J. Mech. Phys. Solids 149 104325

[118] Cai R, Jin Y, Li Y, Rabczuk T, Pennec Y, Djafari-Rouhani B and Zhuang X 2022 Exceptional points and skin modes in non-Hermitian metabeams Phys. Rev. Appl. 18 014067

[119] Alamri S, Li B, McHugh G, Garafolo N and Tan K T 2019 Dissipative diatomic acoustic metamaterials for broadband asymmetric elastic-wave transmission J. Sound Vib. 451 120–37

[120] Auregan Y and Pagneux V 2017 PT-symmetric scattering in flow duct acoustics Phys. Rev. Lett. 118 174301

[121] Bender C M, Berntson B K, Parker D and Samuel E 2013 Observation of PT phase transition in a simple mechanical system Am. J. Phys. 81 173–9

[122] Jing H, Ozdemir S K, Lu X Y, Zhang J, Yang L and Nori F 2014 PT-symmetric phonon laser Phys. Rev. Lett. 113 053604

[123] Xu X W, Liu Y X, Sun C P and Li Y 2015 Mechanical $\mathcal{PT}$ symmetry in coupled optomechanical systems Phys. Rev. A 92 013852

[124] Poshakinskiy A V, Poddubny A N and Fainstein A 2016 Multiple quantum wells for $\mathcal{PT}$ -symmetric phononic crystals Phys. Rev. Lett. 117 224302

[125] Xu H, Mason D, Jiang L and Harris J G 2016 Topological energy transfer in an optomechanical system with exceptional points Nature 537 80–83

[126] Kononchuk R and Kottos T 2020 Orientation-sensed optomechanical accelerometers based on exceptional points Phys. Rev. Res. 2 023252

[127] Djorwe P, Pennec Y and Djafari-Rouhani B 2019 Exceptional point enhances sensitivity of optomechanical mass sensors Phys. Rev. Appl. 12 024002

[128] Tchounda S R M, Djorwé P, Engo S G N and Djafari-Rouhani B 2023 Sensor sensitivity based on exceptional points engineered via synthetic magnetism Phys. Rev. Appl. 19 064016

[129] Hu B et al 2021 Non-Hermitian topological whispering gallery Nature 597 655–9

[130] Maksimov D N, Sadreev A F, Lyapina A A and Pilipchuk A S 2015 Coupled mode theory for acoustic resonators Wave Motion 56 52–66

[131] Rotter I 2009 A Non-Hermitian Hamilton operator and the physics of open quantum systems J. Phys. A: Math. Theor. 42 153001

[132] Heiss W D 2012 The physics of exceptional points J. Phys. A: Math. Theor. 45 444016

[133] Bergholtz E J, Budich J C and Kunst F K 2021 Exceptional topology of non-Hermitian systems Rev. Mod. Phys. 93 015005

[134] Fang X, Gerard N J R K, Zhou Z, Ding H, Wang N, Jia B, Deng Y, Wang X, Jing Y and Li Y 2021 Observation of higher-order exceptional points in a non-local acoustic metagrating Commun. Phys. 4 271

[135] Ding K, Ma G, Xiao M, Zhang Z Q and Chan C T 2016 Emergence, coalescence and topological properties of multiple exceptional points and their experimental realization Phys. Rev. X 6 021007

[136] Wang S, Hou B, Lu W, Chen Y, Zhang Z Q and Chan C T 2019 Arbitrary order exceptional point induced by photonic spin-orbit interaction in coupled resonators Nat. Commun. 10 832

[137] Chen Z G, Wang L, Zhang G and Ma G 2020 Chiral symmetry breaking of tight-binding models in coupled acoustic-cavity systems Phys. Rev. Appl. 14 024023

[138] Shin Y, Kwak H, Moon S, Lee S B, Yang J and An K 2016 Observation of an exceptional point in a two-dimensional ultrasonic cavity of concentric circular shells Sci. Rep. 6 38826

[139] Berry M V and Wilkinson M 1997 Diabolical points in the spectra of triangles Proc. R. Soc. A 392 15–43

[140] Shen H, Zhen B and Fu L 2018 Topological band theory for non-Hermitian Hamiltonians Phys. Rev. Lett. 120 146402

[141] Ding K, Ma G, Zhang Z Q and Chan C T 2018 Experimental demonstration of an anisotropic exceptional point Phys. Rev. Lett. 121 085702

[142] Tang W, Jiang X, Ding K, Xiao Y X, Zhang Z Q, Chan C T and Ma G 2020 Exceptional nexus with a hybrid topological invariant Science 370 1077–80

[143] Tang W, Ding K and Ma G 2021 Direct measurement of topological properties of an exceptional parabola Phys. Rev. Lett. 127 034301

[144] Liu J J, Li Z W, Chen Z G, Tang W, Chen A, Liang B, Ma G and Cheng J C 2022 Experimental realization of Weyl exceptional rings in a synthetic three-dimensional non-Hermitian phononic crystal Phys. Rev. Lett. 129 084301

[145] Tang W, Ding K and Ma G 2022 Realization and topological properties of third-order exceptional lines embedded in exceptional surfaces (arXiv:2211.15921)

[146] Liu T, Zhu X, Chen F, Liang S and Zhu J 2018 Unidirectional wave vector manipulation in two-dimensional space with an all passive acoustic parity-time-symmetric metamaterials crystal Phys. Rev. Lett. 120 124502

[147] Ge L, Chong Y D and Stone A D 2012 Conservation relations and anisotropic transmission resonances in one-dimensional PT-symmetric photonic heterostructures Phys. Rev. A 85 023802

[148] Novitsky A, Lyakhov D, Michels D, Pavlov A A, Shalin A S and Novitsky D V 2020 Unambiguous scattering matrix for non-Hermitian systems Phys. Rev. A 101 043834

[149] Wang C, Sweeney W R, Stone A D and Yang L 2021 Coherent perfect absorption at an exceptional point Science 373 1261–5

[150] Yi J, Negahban M, Li Z, Su X and Xia R 2019 Conditionally extraordinary transmission in periodic parity-time symmetric phononic crystals Int. J. Mech. Sci. 163 105134

[151] Okuma N, Kawabata K, Shiozaki K and Sato M 2020 Topological origin of non-Hermitian skin effects Phys. Rev. Lett. 124 086801

[152] Zhang L et al 2021 Acoustic non-Hermitian skin effect from twisted winding topology Nat. Commun. 12 6297

[153] Zhang X, Tian Y, Jiang J H, Lu M H and Chen Y F 2021 Observation of higher-order non-Hermitian skin effect Nat. Commun. 12 5377

[154] Wang W, Wang X and Ma G 2022 Non-Hermitian morphing of topological modes Nature 608 50–55

[155] Brandenbourger M, Locsin X, Lerner E and Coulais C 2019 Non-reciprocal robotic metamaterials Nat. Commun. 10 4608

[156] Achilleos V, Theocharis G, Richoux O and Pagneux V 2017 Non-Hermitian acoustic metamaterials: role of exceptional points in sound absorption Phys. Rev. B 95 144303

[157] Ji W Q, Wei Q, Zhu X F, Wu D J and Liu X J 2019 Extraordinary acoustic scattering in a periodic PT-symmetric zero-index metamaterials waveguide Europhys. Lett. 125 58002

[158] Lan J, Wang L, Zhang X, Lu M and Liu X 2020 Acoustic multifunctional logic gates and amplifier based on passive parity-time symmetry Phys. Rev. Appl. 13 034047

[159] Gu Z, Liu T, Gao H, Liang S, An S and Zhu J 2021 Acoustic coherent perfect absorber and laser modes via the non-Hermitian dopant in the zero index metamaterials J. Appl. Phys. 129 234901

[160] Lan J, Zhang X, Wang L, Lai Y and Liu X 2020 Bidirectional acoustic negative refraction based on a pair of metasurfaces with both local and global PT-symmetries Sci. Rep. 10 10794

[161] Guo A, Salamo G J, Duchesne D, Morandotti R, Volatier-Ravat M, Aimez V, Siviloglou G A and Christodoulides D N 2009 Observation of PT-symmetry breaking in complex optical potentials Phys. Rev. Lett. 103 093902

[162] Shen C, Li J, Peng X and Cummer S A 2018 Synthetic exceptional points and unidirectional zero reflection in non-Hermitian acoustic systems Phys. Rev. Mater. 2 125203

[163] Merkel A, Romero-García V, Groby J P, Li J and Christensen J 2018 Unidirectional zero sonic reflection in passive PT-symmetric Willis media Phys. Rev. B 98 201102

[164] Liu Y, Liang Z, Zhu J, Xia L, Mondain-Monval O, Brunet T, Alù A and Li J 2019 Willis metamaterial on a structured beam Phys. Rev. X 9 011040

[165] Meng Y, Hao Y, Guenneau S, Wang S and Li J 2021 Willis coupling in water waves New J. Phys. 23 073004

[166] Thevamaran R, Branscomb R M, Makri E, Anzel P, Christodoulides D, Kottos T and Thomas E L 2019 Asymmetric acoustic energy transport in non-Hermitian metamaterials J. Acoust. Soc. Am. 146 863

[167] Ding K, Fang C and Ma G 2022 Non-Hermitian topology and exceptional-point geometries Nat. Rev. Phys. 4 745–60

[168] Zhao D, Shen Y, Zhang Y, Zhu X and Yi L 2016 Bound states in one-dimensional acoustic parity-time-symmetric lattices for perfect sensing Phys. Lett. A 380 2698–702

[169] Zhang H, Zhang Y, Liu X, Bao Y and Zhao J 2022 Acoustic impurity shielding induced by a pair of metasurfaces respecting $\mathcal{PT}$ symmetry Phys. Rev. B 106 094101

[170] Zhang H, Zhang Y, Liu X, Bao Y and Zhao J 2022 Step-wise constant-amplitude waves in non-Hermitian disordered media AIP Adv. 12 065217

[171] Li D, Huang S, Cheng Y and Li Y 2021 Compact asymmetric sound absorber at the exceptional point Sci. China Phys. Mech. Astron. 64 244303

[172] Lee T, Nomura T, Dede E M and Iizuka H 2020 Asymmetric loss-induced perfect sound absorption in duct silencers Appl. Phys. Lett. 116 214101

[173] Li X, Yu Z, Iizuka H and Lee T 2022 Experimental demonstration of extremely asymmetric flexural wave absorption at the exceptional point Extreme Mech. Lett. 52 101649

[174] Puri S, Ferdous J, Shakeri A, Basiri A, Dubois M and Ramezani H 2021 Tunable non-Hermitian acoustic filter Phys. Rev. Appl. 16 014012

[175] Li Y, Shen C, Xie Y, Li J, Wang W, Cummer S A and Jing Y 2017 Tunable asymmetric transmission via lossy acoustic metasurfaces Phys. Rev. Lett. 119 035501

[176] Shen C and Cummer S A 2018 Harnessing multiple internal reflections to design highly absorptive acoustic metasurfaces Phys. Rev. Appl. 9 054009

[177] Ju F, Tian Y, Cheng Y and Liu X 2018 Asymmetric acoustic transmission with a lossy gradient-index metasurface Appl. Phys. Lett. 113 121901

[178] Song X, Chen T, Zhu J, Ding W, Liang Q and Wang X 2020 Broadband and broad-angle asymmetric acoustic transmission by unbalanced excitation of surface evanescent waves based on single-layer metasurface Phys. Lett. A 384 126419

[179] Lee T and Iizuka H 2019 Acoustic resonance coupling for directional wave control: from angle-dependent absorption to asymmetric transmission New J. Phys. 21 043030

[180] Liu B and Jiang Y 2018 Controllable asymmetric transmission via gap-tunable acoustic metasurface Appl. Phys. Lett. 112 173503

[181] Craig S R, Welch P J and Shi C 2019 Non-Hermitian complementary acoustic metamaterials for lossy barriers Appl. Phys. Lett. 115 051903

[182] Fleury R, Sounas D L and Alù A 2016 Parity-time symmetry in acoustics: theory, devices and potential applications IEEE J. Sel. Top. Quantum Electron. 22 121–9

[183] Song A, Li J, Peng X, Shen C, Zhu X, Chen T and Cummer S A 2019 Asymmetric absorption in acoustic metamirror based on surface impedance engineering Phys. Rev. Appl. 12 054048

[184] Ju F, Zou X, Qian S Y and Liu X 2021 Asymmetric acoustic retroflection with a non-Hermitian metasurface mirror Appl. Phys. Express 14 124001

[185] Kim M S, Lee W, Park C I and Oh J H 2020 Elastic wave energy entrapment for reflectionless metasurface Phys. Rev. Appl. 13 054036

[186] Liu T, Liang S, Chen F and Zhu J 2018 Inherent losses induced absorptive acoustic rainbow trapping with a gradient metasurface J. Appl. Phys. 123 091702

[187] Cao L, Yang Z, Xu Y, Fan S W, Zhu Y, Chen Z, Li Y and Assouar B 2020 Flexural wave absorption by lossy gradient elastic metasurface J. Mech. Phys. Solids 143 104052

[188] Cao L, Zhu Y, Wan S, Zeng Y and Assouar B 2022 On the design of non-Hermitian elastic metamaterial for broadband perfect absorbers Int. J. Eng. Sci. 181 103768

[189] Zhu Y, Hu J, Fan X, Yang J, Liang B, Zhu X and Cheng J 2018 Fine manipulation of sound via lossy metamaterials with independent and arbitrary reflection amplitude and phase Nat. Commun. 9 1632

[190] Fan S W, Zhu Y, Cao L, Wang Y F, Li Chen A, Merkel A, Wang Y S and Assouar B 2020 Broadband tunable lossy metasurface with independent amplitude and phase modulations for acoustic holography Smart Mater. Struct. 29 105038

[191] Zhu Y, Gerard N J R K, Xia X, Stevenson G C, Cao L, Fan S, Spadaccini C M, Jing Y and Assouar B 2021 Systematic design and experimental demonstration of transmission-type multiplexed acoustic metaholograms Adv. Funct. Mater. 31 2101947

[192] Zhu Y and Assouar B 2019 Systematic design of multiplexed-acoustic-metasurface hologram with simultaneous amplitude and phase modulations Phys. Rev. Mater. 3 045201

[193] Liu T, Ma G, Liang S, Gao H, Gu Z, An S and Zhu J 2020 Single-sided acoustic beam splitting based on parity-time symmetry Phys. Rev. B 102 014306

[194] Yang H, Zhang X, Liu Y, Yao Y, Wu F and Zhao D 2019 Novel acoustic flat focusing based on the asymmetric response in parity-time-symmetric phononic crystals Sci. Rep. 9 10048

[195] Stojanoska K and Shen C 2022 Non-Hermitian planar elastic metasurface for unidirectional focusing of flexural waves Appl. Phys. Lett. 120 241701

[196] Gao F, Xie J, Peng Y, Yan B, Liu E, Peng P, Li H, Jiang J and Liu J 2020 Subwavelength acoustic focusing within multi-breadth bands with a window-shape metasurface Europhys. Lett. 132 38003

[197] Chiu C K, Teo J C Y, Schnyder A P and Ryu S 2016 Classification of topological quantum matter with symmetries Rev. Mod. Phys. 88 035005

[198] Armitage N P, Mele E J and Vishwanath A 2018 Weyl and Dirac semimetals in three-dimensional solids Rev. Mod. Phys. 90 015001

[199] Yang Z, Gao F, Shi X, Lin X, Gao Z, Chong Y and Zhang B 2015 Topological acoustics Phys. Rev. Lett. 114 114301

[200] Zhang X, Xiao M, Cheng Y, Lu M H and Christensen J 2018 Topological sound Commun. Phys. 1 97

[201] Ma G, Xiao M and Chan C T 2019 Topological phases in acoustic and mechanical systems Nat. Rev. Phys. 1 281–94

[202] Xue H, Yang Y and Zhang B 2022 Topological acoustics Nat. Rev. Mater. 7 974–90

[203] Su W P, Schrieffer J R and Heeger A J 1979 Solitons in polyacetylene Phys. Rev. Lett. 42 1698–701

[204] Asbóth J, Oroszlány L and Pályi A 2016 A Short Course on Topological Insulators: Band Structure and Edge States in One and Two Dimensions (Lecture Notes in Physics) (Springer International Publishing)

[205] Zhu W, Fang X, Li D, Sun Y, Li Y, Jing Y and Chen H 2018 Simultaneous observation of a topological edge state and exceptional point in an open and non-Hermitian acoustic system Phys. Rev. Lett. 121 124501

[206] Gao H, Xue H, Gu Z, Liu T, Zhu J and Zhang B 2021 Non-Hermitian route to higher-order topology in an acoustic crystal Nat. Commun. 12 1888

[207] Leykam D, Bliokh K Y, Huang C, Chong Y D and Nori F 2017 Edge modes, degeneracies and topological numbers in non-Hermitian systems Phys. Rev. Lett. 118 040401

[208] Yao S and Wang Z 2018 Edge states and topological invariants of non-Hermitian systems Phys. Rev. Lett. 121 086803

[209] Gong Z, Ashida Y, Kawabata K, Takasan K, Higashikawa S and Ueda M 2018 Topological phases of non-Hermitian systems Phys. Rev. X 8 031079

[210] Kawabata K, Shiozaki K, Ueda M and Sato M 2019 Symmetry and topology in non-Hermitian physics Phys. Rev. X 9 041015

[211] Wang K, Dutt A, Yang K Y, Wojcik C C, Vuckovic J and Fan S 2021 Generating arbitrary topological windings of a non-Hermitian band Science 371 1240–5

[212] Coulais C, Fleury R and van Wezel J 2020 Topology and broken Hermiticity Nat. Phys. 17 9–13

[213] He L, Li Y, Djafari-Rouhani B and Jin Y 2023 Hermitian and non-Hermitian Weyl physics in synthetic three-dimensional piezoelectric phononic beams Phys. Rev. Res. 5 023020

[214] Pino J D, Slim J J and Verhagen E 2022 Non-Hermitian chiral phononics through optomechanically induced squeezing Nature 606 82–87

[215] Wang M, Ye L, Christensen J and Liu Z 2018 Valley physics in non-Hermitian artificial acoustic boron nitride Phys. Rev. Lett. 120 246601

[216] Zhang Z, Rosendo López M, Cheng Y, Liu X and Christensen J 2019 Non-Hermitian sonic second-order topological insulator Phys. Rev. Lett. 122 195501

[217] Rosendo López M, Zhang Z, Torrent D and Christensen J 2019 Multiple scattering theory of non-Hermitian sonic second-order topological insulators Commun. Phys. 2 132

[218] Wang W, Wang X and Ma G 2022 Extended state in a localized continuum Phys. Rev. Lett. 129 264301

[219] Gao H, Xue H, Wang Q, Gu Z, Liu T, Zhu J and Zhang B 2020 Observation of topological edge states induced

solely by non-Hermiticity in an acoustic crystal Phys. Rev. B 101 180303

[220] Ghatak A, Brandenbourger M, van Wezel J and Coulais C 2020 Observation of non-Hermitian topology and its bulk-edge correspondence in an active mechanical metamaterial Proc. Natl Acad. Sci. USA 117 29561–8

[221] Zhang K, Zhang X, Wang L, Zhao D, Wu F, Yao Y, Xia M and Guo Y 2021 Observation of topological properties of non-Hermitian crystal systems with diversified coupled resonators chains J. Appl. Phys. 130 064502

[222] Monticone F, Valagiannopoulos C A and Alù A 2016 Parity-time symmetric nonlocal metasurfaces: all-angle negative refraction and volumetric imaging Phys. Rev. X 6 041018

[223] Rosa M I N and Ruzzene M 2020 Dynamics and topology of non-Hermitian elastic lattices with non-local feedback control interactions New J. Phys. 22 053004

[224] Braghini D, Villani L G G, Rosa M I N and Arruda J R D F 2021 Non-Hermitian elastic waveguides with piezoelectric feedback actuation: non-reciprocal bands and skin modes J. Phys. D: Appl. Phys. 54 285302

[225] Zhou H T, Fu W X, Li X S, Wang Y F and Wang Y S 2022 Loosely coupled reflective impedance metasurfaces: precise manipulation of waterborne sound by topology optimization Mech. Syst. Signal Process. 177 109228

[226] Hou Z, Ni H and Assouar B 2018 PT-symmetry for elastic negative refraction Phys. Rev. Appl. 10 044071

[227] Yang Y, Jia H, Bi Y, Zhao H and Yang J 2019 Experimental demonstration of an acoustic asymmetric diffraction grating based on passive parity-time-symmetric medium Phys. Rev. Appl. 12 034040

[228] Yang Y, Jia H, Wang S, Zhang P and Yang J 2020 Diffraction control in a non-Hermitian acoustic grating Appl. Phys. Lett. 116 213501

[229] Hodaei H, Hassan A U, Wittek S, Garcia-Gracia H, El-Ganainy R, Christodoulides D N and Khajavikhan M 2017 Enhanced sensitivity at higher-order exceptional points Nature 548 187–91

[230] Quan L, Yves S, Peng Y, Esfahlani H and Alù A 2021 Odd Willis coupling induced by broken time-reversal symmetry Nat. Commun. 12 2615

[231] Zhou Z, Jia B, Wang N, Wang X and Li Y 2023 Observation of perfectly-chiral exceptional point via bound state in the continuum Phys. Rev. Lett. 130 116101

[232] Popa B I and Cummer S A 2014 Non-reciprocal and highly nonlinear active acoustic metamaterials Nat. Commun. 5 3398

[233] Coulais C, Sounas D and Alù A 2017 Static non-reciprocity in mechanical metamaterials Nature 542 461–4

[234] Fleury R, Sounas D L, Sieck C F, Haberman M R and Alù A 2014 Sound isolation and giant linear nonreciprocity in a compact acoustic circulator Science 343 516–9

[235] Oudich M, Deng Y, Tao M and Jing Y 2019 Space-time phononic crystals with anomalous topological edge states Phys. Rev. Res. 1 033069

[236] Li J, Jing Y and Cummer S A 2022 Nonreciprocal coupling in space-time modulated systems at exceptional points Phys. Rev. B 105 L100304

[237] Wan S, Cao L, Zhu Y, Oudich M and Assouar B 2021 Nonreciprocal sound propagation via cascaded time-modulated slab resonators Phys. Rev. Appl. 16 064061

[238] Zhu Y, Cao L, Merkel A, Fan S W, Vincent B and Assouar B 2021 Janus acoustic metascreen with nonreciprocal and reconfigurable phase modulations Nat. Commun. 12 7089

[239] Chen Z, Peng Y, Li H, Liu J, Ding Y, Liang B, Zhu X F, Lu Y, Cheng J and Alù A 2021 Efficient nonreciprocal mode transitions in spatiotemporally modulated acoustic metamaterials Sci. Adv. 7 eabj1198

[240] Cheng W and Hu G 2022 Acoustic skin effect with non-reciprocal Willis materials Appl. Phys. Lett. 121 041701

[241] Zhai Y, Kwon H S and Popa B I 2019 Active Willis metamaterials for ultracompact nonreciprocal linear acoustic devices Phys. Rev. B 99 220301

[242] Popa B I, Zhai Y and Kwon H S 2018 Broadband sound barriers with bianisotropic metasurfaces Nat. Commun. 9 5299

[243] Chen Y, Li X, Hu G, Haberman M R and Huang G 2020 An active mechanical Willis meta-layer with asymmetric polarizabilities Nat. Commun. 11 3681

[244] Yang M and Sheng P 2017 Sound absorption structures: from porous media to acoustic metamaterials Annu. Rev. Mater. Res. 47 83–114

[245] Arenas J P and Crocker M J 2010 Recent trends in porous sound-absorbing materials Sound Vib. 44 12–17

[246] Cao L, Fu Q, Si Y, Ding B and Yu J 2018 Porous materials for sound absorption Compos. Commun. 10 25–35

[247] Qu S and Sheng P 2022 Microwave and acoustic absorption metamaterials Phys. Rev. Appl. 17 047001

[248] Zhang C and Hu X 2016 Three-dimensional single-port labyrinthine acoustic metamaterial: perfect absorption with large bandwidth and tunability Phys. Rev. Appl. 6 064025

[249] Jiang X, Liang B, Li R Q, Zou X Y, Yin L L and Cheng J C 2014 Ultra-broadband absorption by acoustic metamaterials Appl. Phys. Lett. 105 243505

[250] Peng X, Ji J and Jing Y 2018 Composite honeycomb metasurface panel for broadband sound absorption J. Acoust. Soc. Am. 144 EL255

[251] Jimenez N, Romero-Garcia V, Pagneux V and Groby J P 2017 Rainbow-trapping absorbers: broadband, perfect and asymmetric sound absorption by subwavelength panels for transmission problems Sci. Rep. 7 13595

[252] Romero-García V, Theocharis G, Richoux O, Merkel A, Tournat V and Pagneux V 2016 Perfect and broadband acoustic absorption by critically coupled sub-wavelength resonators Sci. Rep. 6 19519

[253] Ji J, Li D, Li Y and Jing Y 2020 Low-frequency broadband acoustic metasurface absorbing panels Front. Mech. Eng. 6 586249

[254] Li J, Wang W, Xie Y, Popa B I and Cummer S A 2016 A sound absorbing metasurface with coupled resonators Appl. Phys. Lett. 109 091908

[255] Zhu Y, Merkel A, Donda K, Fan S, Cao L and Assouar B 2021 Nonlocal acoustic metasurface for ultrabroadband sound absorption Phys. Rev. B 103 064102

[256] Huang S, Zhou Z, Li D, Liu T, Wang X, Zhu J and Li Y 2020 Compact broadband acoustic sink with coherently coupled weak resonances Sci. Bull. 65 373–9

[257] Ryoo H and Jeon W 2018 Perfect sound absorption of ultra-thin metasurface based on hybrid resonance and space-coiling Appl. Phys. Lett. 113 121903

[258] Cai X, Guo Q, Hu G and Yang J 2014 Ultrathin low-frequency sound absorbing panels based on coplanar spiral tubes or coplanar Helmholtz resonators Appl. Phys. Lett. 105 121901

[259] Ren Z, Cheng Y, Chen M, Yuan X and Fang D 2022 A compact multifunctional metastructure for Low-frequency broadband sound absorption and crash energy dissipation Mater. Des. 215 110462

[260] Li Z, Li X, Wang Z and Zhai W 2023 Multifunctional sound-absorbing and mechanical metamaterials via a decoupled mechanism design approach Mater. Horiz. 10 75–87

[261] Zeng K, Li Z, Guo Z, Liang X and Wang Z 2022 Acoustic metamaterial for highly efficient low-frequency

impedance modulation by extensible design Extreme Mech. Lett. 56 101855

[262] Zhou Z, Huang S, Li D, Zhu J and Li Y 2022 Broadband impedance modulation via non-local acoustic metamaterials Natl Sci. Rev. 9 nwab171

[263] Liu L, Xie L X, Huang W, Zhang X J, Lu M H and Chen Y F 2022 Broadband acoustic absorbing metamaterial via deep learning approach Appl. Phys. Lett. 120 251701

[264] Shao C, Zhu Y, Long H, Liu C, Cheng Y and Liu X 2022 Metasurface absorber for ultra-broadband sound via over-damped modes coupling Appl. Phys. Lett. 120 083504

[265] Ding H, Wang N, Qiu S, Huang S, Zhou Z, Zhou C, Jia B and Li Y 2022 Broadband acoustic meta-liner with metal foam approaching causality-governed minimal thickness Int. J. Mech. Sci. 232 107601

[266] Lee K Y, Yoon S, Song S H and Yoon J W 2022 Topological beaming of light Sci. Adv. 8 eadd8349

[267] Gorlach M A, Ni X, Smirnova D A, Korobkin D, Zhirihin D, Slobozhanyuk A P, Belov P A, Alù A and Khanikaev A B 2018 Far-field probing of leaky topological states in all-dielectric metasurfaces Nat. Commun. 9 909

[268] Zhang Z, Tian Y, Wang Y, Gao S, Cheng Y, Liu X and Christensen J 2018 Directional acoustic antennas based on valley-Hall topological insulators Adv. Mater. 30 1803229

[269] Song Q, Odeh M, Zúñiga-Pérez J, Kanté B and Genevet P 2021 Plasmonic topological metasurface by encircling an exceptional point Science 373 1133–7

[270] Zhen B, Hsu C W, Lu L, Stone A D and Soljacic M 2014 Topological nature of optical bound states in the continuum Phys. Rev. Lett. 113 257401

[271] Deng Z L, Li F J, Li H, Li X and Alù A 2022 Extreme diffraction control in metagratings leveraging bound states in the continuum and exceptional points Laser Photon. Rev. 16 2100617

[272] Jimenez N, Cox T J, Romero-Garcia V and Groby J P 2017 Metadiffusers: deep-subwavelength sound diffusers Sci. Rep. 7 5389

[273] Jin Y, He L, Wen Z, Mortazavi B, Guo H, Torrent D, Djafari-Rouhani B, Rabczuk T, Zhuang X and Li Y 2022 Intelligent on-demand design of phononic metamaterials Nanophotonics 11 439–60

[274] Muhammad, Kennedy J and Lim C W 2022 Machine learning and deep learning in phononic crystals and metamaterials—a review Mater. Today Commun. 33 104606

[275] Karniadakis G E, Kevrekidis I G, Lu L, Perdikaris P, Wang S and Yang L 2021 Physics-informed machine learning Nat. Rev. Phys. 3 422–40

[276] Wang Y S, Chen W, Wu B, Wang Y Z and Wang Y F 2020 Tunable and active phononic crystals and metamaterials Appl. Mech. Rev. 72 040801

[277] Fan S W, Zhao S D, Chen A L, Wang Y F, Assouar B and Wang Y S 2019 Tunable broadband reflective acoustic metasurface Phys. Rev. Appl. 11 044038

[278] Sergeev S, Fleury R and Lissek H 2023 Ultrabroadband sound control with deep-subwavelength plasmacoustic metalayers Nat. Commun. 14 2874

[279] Lin Z, Zhang Y, Wang K W and Tol S 2022 Anomalous wavefront control via nonlinear acoustic metasurface through second-harmonic tailoring and demultiplexing Appl. Phys. Lett. 121 201703

[280] Jeon G J and Oh J H 2021 Nonlinear acoustic metamaterial for efficient frequency down-conversion Phys. Rev. E 103 012212

[281] Fang X, Wen J, Bonello B, Yin J and Yu D 2017 Ultra-low and ultra-broad-band nonlinear acoustic metamaterials Nat. Commun. 8 1288
