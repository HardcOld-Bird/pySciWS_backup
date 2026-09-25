REVIEW SUMMARY

METAMATERIALS

# Tunable structured light with flat optics

Ahmed H. Dorrah\* and Federico Capasso\*

BACKGROUND: Structuring the degrees of freedom of light—including its phase, amplitude, and polarization—has opened new frontiers in science and technology alike. Adaptive cameras, microscopes, portable and wearable devices, optical communications, and laser machining are only a few of the domains that have evolved over the past decade owing to the advances in wavefront-shaping platforms. Flat optics composed of subwavelength-spaced optical scatterers—also known as metasurfaces or meta-optics—are key enabling tools for structured light not only for their compact footprint and complementary metal-oxide semiconductor (CMOS) compatibility but also because of their versatility and custom design. Although flat optics, or at least in its first generation, has led to the development of effects like anomalous refraction and diffraction-limited focusing, new classes of metasurfaces can now mold the flow of light in much more complex ways. Dispersion engineering and polarization optics are two prominent areas in which the metasurface's ability to spatially manipulate each wavelength and/or polar ization state, independently, cannot be paralleled using bulk optical components. At the heart of these developments is a carefully engineered light-matter interaction at the level of the met-atom, which allows a passive metasurface to produce this complex response.

ADVANCES: The ability to manipulate light in different ways, depending on its properties, is intriguing because it allows a passive device to produce many functions without the need for active switching—that is, light itself can be used as an optical control knob. For example, the same flat optic may behave as a lens or a mirror depending on the incidence angle of light. Likewise, by changing light's polarization, a metasurface can switch between different holograms or modify its focal length. In this spirit, a new generation of meta-optics can now perform parallel processing of the polarization of input light in the transverse plane or in 3D, reducing the function of many polarizers and waveplates into a single optical component that can be integrated in

![](images/75ee1571a03ce3c4ed1a77d144fe350074236d91b5229a4d3a427486abbc6869.jpg)

Tunable structured light with static meta-optics. Different properties of input light may act as control knobs for tuning the optical response of the metasurface. These degrees of freedom include the angle of incidence (A), polarization state (B), orbital angular momentum (or spatial structure in general) (C), wavelength (D), and intensity level (manifested in a nonlinear interaction) (E). By changing one or more of these properties at the input of the metasurface, one can obtain a different light pattern at the output (depicted by the different puzzle pieces of Harvard University's logo). This tunability relies on an intricate light-matter interaction at the level of the meta-atom, which often cannot be replicated by other conventional wavefront shaping platforms.

polarimeters and cellphone cameras. The spatial phase distribution of incoming light is another degree of freedom that can also be used as a switch, allowing a static flat optic to project different holograms by varying the helicity of the incident wavefront or its phase profile in general. Moreover, the ability to impart different phase and/or amplitude profiles on different wavelengths, independently, has enabled a wide class of versatile metalenses and compact pulse-shaping tools. Harnessing the nonlinear interaction of light with meta-atoms has also enabled multiwavelength holography on high harmonic-generated signals in addition to an asymmetric response. This versatility has made flat optics an ideal platform for the generation of structured light and has inspired many applications. The figure depicts the use of a metasurface as a multipurpose device (akin to a Swiss knife) that can mix and match output light by tuning the five above-mentioned control knobs, without the need for any complex circuitry to drive the meta-atoms. Tunable behavior of this kind relies on an intricate light-matter interaction at the nanoscale, which is often difficult to replicate with other wavefront-shaping tools. With such tunability, many existing technologies can be dramatically miniaturized, enabling compact spectrometers, polarization-sensitive cameras, lightweight augmented reality and virtual reality headsets, and biomedical devices.

OUTLOOK: As the area of structured light matures, the quest for more sophisticated metasurfaces is also on the rise, aided by advanced nanofabrication, powerful computation capabilities for revealing new meta-atom libraries, and recent developments in actively tunable materials for time-varying control. Structured light with tunable metasurfaces is poised to reveal new functionalities and to replace conventional optical systems with on-chip photonic components. This includes integrating metasurfaces in laser cavities, Fabry-Perot resonators, fiber-based devices, and active wavefront-shaping tools. With the emerging trends in inverse design and topology optimization, new standardized protocols for large-scale multilayer metasurface fabrication and innovative material platforms will push the limits of multifunctional meta-optics and structured light from 2D to 3D and from static to animate, thus tackling the open challenges in this wide field of research and unlocking many new paths.

REVIEW

METAMATERIALS

# Tunable structured light with flat optics

Ahmed H. Dorrah\* and Federico Capasso\*

Flat optics has emerged as a key player in the area of structured light and its applications, owing to its subwavelength resolution, ease of integration, and compact footprint. Although its first generation has revolutionized conventional lenses and enabled anomalous refraction, new classes of meta-optics can now shape light and dark features of an optical field with an unprecedented level of complexity and multifunctionality. Here, we review these efforts with a focus on metasurfaces that use different properties of input light—angle of incidence and direction, polarization, phase distribution, wavelength, and nonlinear behavior—as optical knobs for tuning the output response. We discuss ongoing advances in this area as well as future challenges and prospects. These recent developments indicate that optically tunable flat optics is poised to advance adaptive camera systems, microscopes, holograms, and portable and wearable devices and may suggest new possibilities in optical communications and sensing.

Harnessing the properties of light defines how we experience the world around us. High-resolution microscopes, long-range telescopes, fast cameras, and spectrometers are a few tools that have shaped our understanding of the Universe, from the atomic to the astrophysical scale. At the heart of these developments lies an optical component that controls the properties of light as it interacts with matter. Several efforts have been made to manipulate light using artificial materials that do not naturally exist in bulk, inspired by nature's ability to structure light through diffraction, interference, and scattering—as observed, for example, in the colorful wings of butterflies (1). Early attempts toward this goal (from the fourth century) have relied on doping glass with metallic nanoparticles to modify its optical properties, producing intriguing structured coloration owing to the plasmonic resonance of the nanoparticles, as displayed in Roman artifacts and French cathedrals (2). Similarly, periodic multilayer stacks of alternating dielectrics in 1D (3) or 2D or 3D—that is, photonic crystals (4–6)—exhibit extreme reflection properties that deviate from their constituent materials. Engineering the optical response of matter by structuring its geometry or material composition provides new routes toward harnessing the properties of light in unconventional ways (7–9), similar to how semiconductor heterostructures have revolutionized optoelectronics. Metamaterials have emerged as new platforms for controlling light by virtually realizing anomalous values of effective permittivity and permeability (10, 11), altering the output electromagnetic response on demand (12, 13).

Platforms of this nature are composed of periodic unit cells, referred to as meta-atoms, which are typically made of metallic or dielectric scatterers that are tightly packed in a lattice-like arrangement with subwavelength separation $(14)$ .

First introduced at microwave frequencies, the widespread use of metamaterials in optics was hampered by the challenges associated with their fabrication, a task that mandates 3D printing at the nanoscale. To surmount this obstacle, metasurfaces (also known as planar or flat optics or meta-optics) have evolved as a promising wavefront-shaping candidate thanks to their monolithic integration, compact design, and subwavelength control (15–20). Their earlier demonstrations in the optical regime have relaxed basic laws of physics (21–24) and introduced innovative mechanisms for wavefront tilting (25, 26), holography (27–29), and diffraction-limited focusing (30, 31). Notably, metasurfaces have gained wide popularity not only for their compact form factor and complementary metal-oxide semiconductor (CMOS) compatibility but also because of their unprecedented control of polarization (32, 33) and dispersion engineering (34). The former has enabled point-by-point polarization transformations at the nanoscale (35, 36), whereas the latter has allowed broadband focusing and multiwavelength holography (37). These are nontrivial advances that, to date, cannot be replicated with conventional polarization optics or other wavefront-shaping tools. As this area of research matured, the complexity of metasurface design also progressed, enabling more sophisticated functions and a tunable output response. For instance, a wide class of active metasurfaces can produce time-varying behavior that can be precisely controlled with external stimuli, bringing new physics and applications (38–40). Notably, static metasurfaces, with the proper design, can also change their output response by tuning the properties of input light (41). Versatile devices of this kind can realize fast all-optical switching without the need for electronic circuitry to modulate the output response, thereby saving footprint, complexity, and cost.

In this review, we take a closer look at these static devices, focusing on passive meta-optics that can tune their output behavior in response to changing one or more degrees of freedom of input light. Devices of this nature typically rely on an intricate light-matter interaction at the meta-atom level, which cannot be easily replicated by other platforms (like, for example, spatial light modulators). This tunability is often realized by different types of resonances (Mie scattering, Fano resonances, bound states in the continuum) (42) or shape birefringent meta-atoms (35, 36) and free-form topologies (43, 44), thus adding to the existing structured-light toolkit enabled by digital holography (45). We discuss possible technological consequences of these metasurface-based devices, their anticipated challenges, and open areas for innovation. Besides structuring light, we also highlight new methods for structuring the dark (namely, phase singularities) with flat optics and hint to its applications.

## Angle-dependent and directional response

Angle dependence is usually an undesired feature in conventional lenses, holograms, and beam-steering applications because it manifests in the form of diffractive loss, distortion, or coma aberration. Independent control of each input wave vector can mitigate these effects and enable new functions. However, achieving this task is not straightforward because, on a macroscopic scale, metasurfaces (similar to Fresnel lenses) resemble diffraction gratings, which are angle sensitive, whereas on a microscopic level, the response of simple meta-atom geometries (like cylindrical nanopillars) is typically maintained over a wide range of angles. Local and nonlocal metasurfaces tackle this dilemma in conceptually different ways. The former acts on incident light, point by point, in real space (e.g., gradient metasurfaces), whereas the latter relies on neighbor-to-neighbor meta-atom interactions and often uses sharp resonances to collectively modify the output response in momentum space (akin to photonic crystal slabs) $(46)$ . Early examples of local metasurfaces, with angle dependence, used U-shaped meta-atoms made of amorphous silicon supported on a reflective aluminum mirror with a thin spacer of silicon dioxide in between $(47)$ . In this configuration, the meta-atoms act as a multimode resonator, imparting different phase delays on a discrete set of incidence angles. An angle-multiplexed metasurface that generates two distinct holograms for $0^{\circ}$ and $30^{\circ}$ incident light (wavelength $\lambda = 915$ nm) was thus demonstrated. Another

approach exploits the coupling between neighbor meta-atoms to create multifunctional devices that, depending on the excitation angle, may project a different hologram with phase (48) or complex-amplitude modulation in the visible range (49) and can act as a mirror or a half-wave plate in the near-infrared (NIR) range (50). Figure 1A shows a configuration that relies on mutual coupling between meta-atoms (comprising metal-insulator-metal) to achieve an angle-dependent response (49), enabling different holograms to be generated in phase and amplitude by changing the angle of incidence (Fig. 1B).

Additionally, dielectric metasurfaces that combine interleaved superpixels with the concept of detour phase have been used to expand the number of angle-dependent functions: for instance, generating four distinct holograms in response to four distinct angles of incidence (51). The coupling between meta-atoms is highly sensitive to the surrounding environment, suggesting unexplored mechanisms for sensing. For instance, a nonlocal germanium-based metasurface has been used to realize highly sensitive and sharp resonances, with spectral position controlled by the incidence angle of mid-infrared (IR) light (Fig. 1C) (52). This angle-multiplexed approach delivers more than 200 resonances for angles between $13^{\circ}$ and $60^{\circ}$ with a spectral coverage between 1100 and $1800~\mathrm{cm}^{-1}$ and a spectral resolution below $5\mathrm{cm}^{-1}$ . Analyte molecules that match the angular position of the resonances incur strong modulation of the far-field response through surface-enhanced near-field effects. By detecting the angle-resolved reflectance signal from such metasurfaces before and after coating with the analyte molecules, the full spectral content of the molecular absorption fingerprint can be retrieved. Another approach uses off-axis multiplexed holograms to project distinct images on a discrete set of incidence angles (53). Note that obtaining a continuous angle-tunable function is a more daunting task because it mandates controlling the phase derivative with respect to the incident angle. Complex functions of this kind can be achieved using free-form metasurfaces. For example, Fig. 1D shows a topology-optimized polarization meta-optic that modifies its birefringence depending on the incidence angle (44). Light incident on this device would witness linear, circular, or elliptical birefringence depending on its angle (Fig. 1E). Inverse-designed devices of this type use the nonlocal interaction between adjacent meta-atoms to directly operate on incident light in momentum space. Besides an angle-dependent response, nonlocal metasurfaces have enabled many other complex functions such as performing mathematical operations (54, 55), all-optical analog image processing and edge detection (56, 57), and space compression (58, 59), as well as externally tuned laser cavities (60). The latter relies on supercell metasurfaces whose diffraction orders can be efficiently and independently controlled at extreme angles.

![](images/971093b77884f034f4c0ba9a3bd35138be3fce925d2596a0dd791325c24f5d99.jpg)

![](images/0ac436d5338a05c74e17248d6914f0ef1e155ad36df0726a7797c1cb8cd7831f.jpg)
Fig. 1. Angle-dependent and directional metasurfaces. (A) Coupled metal-insulator-metal meta-atom geometries in which the electric energy density distribution depends on the illumination angle, $\theta$ . Reproduced with permission from (49). (B) A metasurface composed of these unit cells can impart different functions for different $\theta$ , such as “nanoprinting” images (channels 1 and 2) and holographic images (channels 3 and 4) (top). Measured data for two independent complex-amplitude holograms (at $\lambda = 633$ nm) for $\theta = 0^{\circ}$ and $30^{\circ}$ (bottom). Reproduced with permission from (49). (C) Germanium-based dielectric metasurface with a high quality factor produces sharp resonances with distinct resonance frequency for each incidence angle over a broad range. Strong near-field coupling between the dielectric resonators and the molecular vibrations of a nearby analyte induces a pronounced attenuation of the resonance line shape that is associated with the vibrational absorption bands. This configuration is suited for surface-enhanced mid-IR molecular absorption spectroscopy. A simple implementation is depicted on the far right. a.u., arbitrary units; $\Delta l(k)$ , normalized intensity as a function of the imaginary part of the complex refractive index. Images reproduced from (52). (D) Free-form amorphous silicon nanostructures patterned on top of a glass substrate exhibit an angle-dependent polarization response at
1550 nm (left). Here, the nanostructure's height (H) = 1500 nm and center-to-center separation (U) = 600 nm. A SEM image of a fabricated sample is shown on the right. Scale bar is 1 $\mu$ m. Images reproduced from (44). (E) On the left, the arrows represent the angle-dependent birefringence axis visualized in polarization space. Different colors refer to different angles of incidence. By varying the angle, the device can be continuously tuned between linear and elliptical birefringence (right). Images reproduced from (44). S, Stokes components. (F) A directional metasurface can project one face of Roman god Janus while totally concealing the other, depending on the direction of input light. k, wave vector. Reproduced with permission from (63). (G) Janus metasurfaces can be implemented by making use of plasmonic helical meta-atoms with an asymmetric chiral response. Using focused ion beam milling with different doses of gallium, the depth of the spiral groove can be adiabatically increased along the red dashed arrows, as shown in the SEM images. Scale bars are 100 nm. Images reproduced from (62). (H) Direction-controlled polarization-encrypted data storage. Right-hand circularly polarized (RCP) light at 800 nm can generate a specific QR code in forward transmission, whereas linearly polarized (LP) light creates a distinct image in reverse. Scale bar is 10 $\mu$ m. Images reproduced from (62).

Besides the angle of incidence, the direction of propagation (forward or backward) has been exploited for tuning the response of static metasurfaces. Commonly referred to as Janus metasurfaces, these directional devices exhibit different coloration $(61)$ or asymmetric transmission $(62, 63)$ by reversing the direction of illumination from the front to the back side. Figure 1F shows a schematic of a metasurface that projects one face of Roman god Janus while totally concealing the other depending on the direction of input light $(63)$ . This has been realized using meta-atoms made of cascaded subwavelength anisotropic impedance sheets. By introducing a gradual rotational in each sheet, linearly polarized light will undergo asymmetric transmission. Similar behavior can be achieved with helical meta-atoms (Fig. 1G), which exhibit circular dichroism as large as 0.72 with forward transmission, owing to spin-dependent mode coupling, while incurring giant linear dichroism up to 0.87 in reverse, with high selectivity for the azimuthal angle of linearly polarized light $(62)$ . Despite their 3D chiral geometry, these meta-atoms are fabricated with single-step focused ion beam milling. By creating a metasurface with two meta-atom enantiomers of specific rotation angles, direction-controlled polarization-encrypted holography can be realized. For instance, Fig. 1H shows a scenario in which a binary QR code image is displayed in the forward direction under right-hand circularly polarized light, whereas a distinct grayscale image is generated in the backward direction under linear polarization. This scheme can be useful in data encryption, optical storage, and information processing. To this end, angle dependence and directionality can be combined to generate distinct holograms in reflection and transmission simultaneously (64, 65).

## Polarization-switchable behavior

Unlike conventional bulk polarizers and waveplates, which are components that control light's polarization globally, the metasurface counterparts enable point-by-point polarization transformations at the nanoscale because of the shape-dependent birefringence of their meta-atoms. Consider, for instance, the rectangular nanofin shown in Fig. 2A. Light transmitted through this "nanoantenna" will experience two independent phase delays along the major and minor axes, depending on the dimensions of the nanofin. Additionally, the nanofin's angular orientation—that is, its birefringence axis—can be rotated, thereby offering a third degree of freedom that typically manifests as a Pancharatnam-Berry phase (66). Replicating this scheme with other wavefront-shaping methods such as spatial light modulators (which only operate on one polarization at a time) is challenging because it requires multiple interactions with incident light or cascading two or more devices (36, 67). Metasurfaces thus stand out as a powerful tool for polarization control and vector-beam generation (35, 36). Earlier attempts, besides polarization gratings (68, 69), have relied on subwavelength Pancharatnam-Berry microstripe gratings at a wavelength of $10.6\mu \mathrm{m}$ for wavefront tilting (26) and vector-beam generation (70) and have exploited plasmonic metasurfaces for manipulating light's chirality (71, 72). In parallel with these efforts, exotic classes of structured light have been created through spin-orbit coupling (73). Liquid crystal q-plates are one popular manifestation of the latter in which conjugate pairs of vortex beams—donut-shaped structured light with helical phase profiles, on-axis singularity, and orbital angular momentum (OAM) (74)—are generated while reversing the chirality of incident circularly polarized light (75). Another approach based on shared aperture antenna arrays and asymmetric harmonic resonances from metal-insulator-metal meta-atoms has enabled spin-controlled structured light (76). Reflective-type plasmonic nanoantennae have also been used to generate polarization-dependent dual images over a broad band in the visible range (77). In addition, dielectric metasurfaces made of elliptically shaped meta-atoms have been used to construct independent pairs of vectorial modes (radially and azimuth ally polarized light) in response to input x and y linearly polarized light in the visible range $(32, 78)$ and have enabled spatial-mode multiplexing at the telecom (1550 nm) wavelength, using linear polarization as a switch $(79, 80)$ . More generally, other states of polarization such as circular and elliptical have been used to switch between two independent phase holograms $(33)$ or two different vortex beams of arbitrary topological charge (helicity), enabling arbitrary spin-orbit coupling $(81)$ ; the latter device is known as a J-plate. There are now numerous examples of polarization control with metasurfaces, from polarization converters to polarization-dependent holograms, varifocal lenses, nanoprinting, and vortex-beam generators, which are comprehensively reviewed in $(35, 36)$ .

We highlight more recent meta-optics with complex polarization-switchable behavior; one example of this is a metasurface that generates a far-field profile that performs parallel polarization analysis of incident light (Fig. 2B) (82). The intensity of each polarization state (denoted by the green arrows) obeys Malus's law, that is, the intensity is proportional to the projection (dot product) of the incident polarization onto that particular state. Hence, this class of metasurfaces, dubbed Jones matrix holograms, enables visual full-Stokes polarimetry of input light by pictorially reading the projected pattern. Figure 2C shows another metasurface that makes use of superpixels composed of four rectangular nanofins to project two complex-amplitude holograms in response to any input pair of orthogonal polarizations (83, 84). Although this multifunctional behavior is limited to two orthogonal polarization channels, chirality-assisted phase modulation deploys cascaded metasurface layers to decouple all four components of the Jones matrix that describe each meta-atom (85). Using this approach, four distinct wavefronts are encoded on the input-output polarization channels L-L, L-R, R-L, and R-R (L-R denotes left- and right-handed circular polarization at the input and output of the device, respectively). Figure 2D shows a metadeflector that uses this concept, steering an input wavefront to four different directions by decoupling all input-output copolarized and cross-polarized channels (85). Similarly, by sandwiching a birefringent metasurface between two polarizers, seven different images have been encrypted on different input-output polarization channels at an 800-nm wavelength (86). Multichannel holography has also been enhanced by including wavelength as an additional degree of freedom (87). Notably, parallel polarization processing and analysis may devise new techniques for polarization characterization. For instance, full Stokes polarimetry [which requires at least four intensity measurements of input light onto four different analyzers (88)] can be replaced with a single metasurface

A
![](images/ef0041cd645270b6ce75167df806a55acd1e2c2cd338a0c4330cb3c8efbf6796.jpg)

![](images/e204ee12902b9ebdbd9805e775344e0ea9cc2b0134938768694a0356bcc59a16.jpg)

![](images/15191dcc1de1d86be1cface01b42850d71cf5818eae15ab3ade4e68b1a273995.jpg)

![](images/7ed7116232bb7e896fad0cb86645cf1f42e432c37d9ed4450a75130afcef0c2e.jpg)

E
![](images/e0ba3a3d73204f96f710f8cd7d2256f7e798ca50ebe6fc7ce1d326cece0eab0e.jpg)
H
G

F
![](images/70b98c0ca2b1d634caca452deb1b327923fc91bf9e9ba753aae137eefb13641b.jpg)

![](images/9a7e66164fb8ff8f66a61a023e3fb269e445fd9168b53d92592f66c3a1ef5625.jpg)

![](images/0e0392a6ad45fb6c5e3473c4078959b9d821e9e1b718eb94fcfdc5e1d3945596.jpg)

![](images/bcd1ccad18952894d2bcc0c584f81193e14db1a39b15e393d88547277858e233.jpg)

![](images/6e2386b67b6c9527c0f7c600ffcdd9c6fbe0c941e1534ff0a1835574d1e36fdf.jpg)
J

(Fig. 2E) (89). This metagrating projects input light onto four diffraction orders, each analyzing for a different polarization in the far field. The underlying principle relies on matrix Fourier optics, which recasts plane-wave decomposition in a two-by-two matrix form, allowing each plane-wave term in the sum to be weighted by a Jones matrix (as opposed to a complex scalar) coefficient (89). This formalism generalizes a wide body of work that seeks to generate vectorial structured light under the assumption of a fixed input polarization (90–95) by lifting that constraint. It also mitigates unwanted diffractive losses and limited functionality of interlaced metagratings (18, 96–98). By interfacing matrix gratings with a commercial CMOS sensor, the spatial polarization profile of a scene can be imaged in real time, enabling a polarization-sensitive camera that provides extra information compared with raw intensity images (Fig. 2F).

K
Fig. 2. Metasurface with polarization-dependent response. (A) A birefringent metasurface composed of dielectric nanofins. SEM image of a small section is shown. Scale bar is 500 nm. Images reproduced from (82). (B) Polarization-analyzing hologram. Under coherent illumination $j_{in}$ , light will be projected into specific far-field pixels depending on the input polarization. Each drawing on the screen acts as an “analyzer” that will exhibit an intensity level for its shown polarization state (depicted by the green arrows) in accordance with Malus’s law. J, spatially varying Jones matrix. Images reproduced from (82). (C) Polarization-dependent complex-amplitude holograms. Far-field images of a “Penrose triangle” (top) and “Möbius strip” (bottom) are generated (at $\lambda = 530$ nm) in response to x- and y-polarized input light, respectively. Scale bars are 200 $\mu$ m. LP, linear polarization. Images reproduced from (83). (D) Chirality-assisted metadeflector can steer the input beam into different directions under four input-output polarization channels of circularly polarized light: L-L, L-R, R-L, and R-R (from top to bottom). Here, L-L denotes input LCP–output LCP light. The device consists of five metallic layers and four dielectric substrates. Simulated and measured normalized far-field patterns are shown at 10 GHz. LHCP, left-handed circular polarization; RHCP, right-handed circular polarization. Images reproduced from (85). (E) Matrix gratings can analyze for many polarization states in parallel at the far field. Incident light will distribute its energy onto the four orders subject to Malus’s law. Integrating this meta-optic with a conventional CMOS sensor enables real-time polarization imaging with no moving parts. Images
![](images/76b3c7db51728ad76ea46670a6464664f4b24b7676673c94b7b3a2dc57cb275c.jpg)
reproduced from (89). (F) Polarization imaging makes use of the measured $S_{3}$ Stokes component. In each example, raw sensor acquisition, $S_{0}$ (intensity image), and $S_{3}$ are shown. Examples include 3D glasses with opposite circular polarizers and a laser-cut acrylic piece that is stressed by hand-squeezing, displaying stress birefringence in its $S_{3}$ image. The black-to-white scale bar represents normalized intensity; the blue-to-red scale bar represents the value of the chiral Stokes component. Images reproduced from (89). (G) Longitudinally variable polarization optics perform many polarization operations, simultaneously, along the optical path. The black arrows depict the virtual principal axis orientation of the polarizing element at each z plane. Images reproduced from (99). $\tilde{F}$ , target polarization transformation. (H) Longitudinal profiles of the generated “pencil-like” beam for each incident polarization that is depicted in the inset. The on-axis intensity distribution continuously shifts away from the source by rotating the input polarization from $0^{\circ}$ to $90^{\circ}$ . Images reproduced from (99). (I) Polarization-switchable TAM plates enable arbitrary spin-orbit coupling in 3D, mapping any pair of orthogonal polarizations ( $|\lambda^{+}\rangle$ and $|\lambda^{-}\rangle\rangle$ ) into two propagation-dependent vortices with varying OAM state, $\ell$ . Images reproduced from (100). (J) Optical micrographs of the devices. Images reproduced from (100). (K) Versatile TAM plates can control light’s polarization and OAM along the optical path. The arrows depict the evolution of output polarization under x-polarized (top) and y-polarized (bottom) illumination. Images reproduced from (100).

Besides controlling polarization in the transverse plane (in 2D), a new class of meta-optics has enabled parallel polarization transformations along the optical path. In this case, a single meta-optic can mimic an arrangement of many polarization optics cascaded in series (Fig. 2G) (99). Light incident on this device is converted into a quasi-nondiffracting (pencil-like) beam that changes its polarization state as if encountering different polarizers and waveplates at each z plane thereafter. For example, Fig. 2H exhibits the response of a meta-optic that analyzes for a set of linear polarizations between $0^{\circ}$ and $90^{\circ}$ along the z direction, as depicted by the red arrows. By rotating the input linear polarization from $0^{\circ}$ to $90^{\circ}$ , the centroid of the output beam is continuously translated along the axial direction, following the projection on the red arrows and in accordance with Malus's law (99). This polarization-switchable axicon may find use in optical trapping applications in which rapid polarization modulation at the input can modify the output scattering forces and radiation pressure on demand. Besides polarization, this design methodology can be generalized to control light's OAM along the propagation direction. Figure 2I shows the response of total angular momentum (TAM) plates (100). By switching the input polarization between any two specified orthogonal bases, the device produces two distinct optical vortices in which the OAM state $\ell$ (wavefront helicity) varies with propagation, as signified by the change in the beam's diameter. The optical micrographs of these devices (Fig. 2J) express very characteristic patterns. Simultaneous and independent control over both components of angular momentum (spin and orbit) has also been achieved (Fig. 2K) (100). In this case, the vortex beam changes its OAM state and rotates its polarization state adiabatically as a function of z, regardless of the incident polarization. Notably, the evolution in angular momentum occurs only locally over the central region of the beam because of spatial multimode beating (i.e., controlled interference between copropagating modes with different wave vectors). The total OAM is conserved when integrated across the entire transverse plane. 3D structured light can be used in refractive index sensing (101, 102), free-space optical communications (103), and micromanipulation at multiple planes (104).

## Tuning with structured light

Because a photon's spin is constrained to two degrees of freedom, polarization-switchable metasurfaces are primarily limited to mapping two orthogonal polarizations to two output wavefront profiles. Multichannel holography (86, 87) can relax this constraint by decoupling different input-output polarization channels, but with additional bulk optics and an upper limit on the number of generated holograms, as described in the previous section. To overcome these challenges, OAM holography with metasurfaces has emerged as a versatile wavefront-shaping tool (105–107). It relies on mapping an orthogonal set of vortex beams (modes with helical phasefront) to an arbitrarily large number of output holograms (constrained by the aperture size). Figure 3, A to C, depicts three variations: OAM-conserving, -selective, and -multiplexing meta-holograms (105). OAM-conserving holograms produce pixelated images while preserving the OAM property of incident OAM beams in each pixel of the reconstructed image (Fig. 3A). To achieve this, the holographic image is spatially sampled by an OAM-dependent 2D Dirac comb function to avoid spatial overlap of the helical wavefront kernel at the image plane (upon its convolution with the hologram). OAM-selective holograms, on the other hand, are sensitive to the input OAM mode because they project a holographic image in response to a particular helical mode. This is realized by adding a spiral phase $e^{-i\ell \phi}$ on an OAM-conserving meta-hologram, where $e$ is Euler's number, $i$ is the imaginary unit, and $\phi$ is the azimuthal phase. In this case, the target image will only be projected if the conjugate mode $e^{i\ell \phi}$ is incident on the metasurface. Notably, adding multiple OAM-selective holograms (designed to be sensitive to different OAM numbers) on the same metasurface will create an OAM-multiplexing meta-hologram that maps different OAM states to distinct holographic images (Fig. 3C). Gallium nitride (GaN) metasurfaces were used to demonstrate these three types of holograms in the visible range (632 nm), by mapping four helical modes to four different holograms (105).

Furthermore, owing to their orthogonality and unbounded size, many OAM modes can be multiplexed by a single meta-hologram while maintaining high spatial resolution. For example, OAM states ranging from an $\ell$ of -50 to 50 impinging on an OAM-multiplexing meta-hologram can sequentially address 200 OAM-dependent orthogonal holographic image frames (100 images at two different axial positions, z), allowing an all-optical holographic video display without any mechanical scanning (Fig. 3D) (106). Here, the metasurface is made of a 3D-printed polymer and can achieve complex-amplitude modulation at 633 nm. Notably, more than one hologram can be encoded on the same OAM state by using the input-output polarization channel as an additional degree of freedom (Fig. 3E) (108).

Besides OAM, more general wavefront distributions can serve as control knobs to project distinct holographic images (Fig. 3F) (109). By illuminating a silicon metasurface with a judiciously engineered wavefront at $633~\mathrm{nm}$ , the encrypted image can be displayed, whereas under uniform illumination, random noise is generated. This approach relies on dividing the target phase profile of the holographic image into two distributions that are imposed on the metasurface and the illumination beam. Even though the metasurface is static, the incident beam contains a large parameter space that can alter (reprogram) the output beam, suggesting new techniques for information security and authentication. Similarly, an all-optical solution for secret key sharing has been proposed using cascaded meta-surface holography (110). This configuration can be used to split and share encrypted holographic information across multiple metasurface layers (Fig. 3G). A set of metasurfaces are designated as “shares.” Each of them contains an encoded phase-only Fourier hologram, which can be reconstructed in the far field upon illumination with circularly polarized light, serving as specific identifiers for each metasurface. Meanwhile, when two metasurfaces are stacked 100 $\mu$ m apart, the illumination of the cascaded configuration creates a new holographic image that is distinct from the two single-layer holograms. This cascaded holographic image can be used as an optical secret that is only revealed if both metasurfaces are stacked and can be extended to a larger set of shares. The concept relies on the fact that the same cascaded phase mask can be built up through combinations of different single phase masks without revealing any information about the shared secret within the single-layer images. It has been implemented with an iterative gradient optimization approach that uses the “automatic differentiation” feature of machine learning. In the future, a reprogrammable version of this scheme can be realized by combining a metasurface as the main share with a digitally addressed spatial light modulator as the deputy share. In addition to structured light, complex patterns of structured dark can potentially be used as metasurface knobs. For example, beyond the 1D singularity carried by OAM modes, 2D singularity sheets have been engineered by minimizing the real and imaginary components of the field over an arbitrarily chosen geometry, as shown in Fig. 3H (111). By maximizing the phase gradient normal to the region of interest, heart-shaped singularity sheets have been generated. Similarly, vectorial singularity sheets, over which the polarization state is undefined, have also been constructed. These degrees of freedom provide additional knobs for meta-holograms and may suggest new sensing schemes owing to their extreme sensitivity to perturbation (112).

## Multiwavelength control

Dispersion, or wavelength dependence, is a profound property of all optical materials. It imposes a fundamental limit on achromatic focusing, broadband holography, and high-rate data transmission. Decades of effort have attempted to tailor dispersion by tuning the chemical composition (refractive index) of matter, which is often a daunting and nonscalable task. Metasurfaces provide a more flexible route to dispersion engineering through shaping the geometry of meta-atoms, thereby altering their effective refractive index in a reproducible manner at the nanoscale. Dispersion-engineered metasurfaces can emulate the dispersion properties of refractive or diffractive optics, on demand, and have subsequently emerged as a promising platform for achromatic focusing, wavelength-dependent holography, multifunctional devices, and pulse shaping, as comprehensively reviewed in $(20, 34, 37)$ . Note that achromaticity is dominated by the phase and group delay (first-order dispersion) of the phase shifters. For example, to maintain a broad-band performance, the phase shifters should exhibit smooth dispersive responses by avoiding sharp resonances. On the other hand, operating near sharp resonances can effectively decouple the phases at different wavelengths, allowing a multifunctional response.

![](images/f23418b6d2d68a708c3b2718d6fa1f9a19e3689a00f936ddfc267bd547ff39a7.jpg)
Fig. 3. Structured light as a metasurface knob. (A) OAM-conserving holograms can project a pixelated image where each pixel carries the same OAM as the source, as depicted by the arrows in the spatial frequency (k-space) domain (top right). The corresponding phase ( $\phi$ ) and intensity (l) distributions of single pixels in the reconstructed images are shown at the bottom right. Pseudocolors visualize different OAM modes. Images reproduced from (105). (B) OAM-selective holograms can project an image in response to a specific input OAM mode. The phase and intensity of each pixel in the reconstructed holographic images are shown on the right. High intensity is achieved in each pixel if the incident OAM mode matches the specified design ( $\ell = -2$ ). Images reproduced from (105). (C) OAM-multiplexing holograms can reconstruct many OAM-dependent holographic images. Images reproduced from (105). (D) OAM holography can realize time-varying image frames by mapping a range of input OAM states (from $\ell = -50$ to 50) to output images, creating a movie. This can occur at multiple z planes along the optical axis. Here, the metasurface is fabricated using 3D laser printing, as depicted in the inset. The nanofin's height (H) and rotation ( $\theta$ ) provide independent control of amplitude and phase. Reprinted by permission from Springer Nature (106). (E) Polarization-encrypted
OAM holography makes use of the input-output polarization channels (white arrows) to encode additional images on each OAM state. The black-to-white scale bar represents normalized intensity. Reproduced with permission from $(108)$ . (F) Encrypted meta-holograms project the desired image only when illuminated with the correct light structure and generate random noise otherwise. Images reproduced from $(109)$ . (G) Secret key sharing using “master share” metasurface holograms (set A) and three “deputy share” metasurface holograms (set B). Every metasurface in each set can project a distinct Fourier hologram. By stacking two metasurfaces from sets A and B, a new holographic image is generated. Each metasurface is represented by a specific color. Images reproduced from $(110)$ . (H) Heart-shaped optical singularities can be engineered by creating an isosurface of low field intensity. The phase gradient is maximized in the directions indicated by the arrows (left). A titanium dioxide metasurface made of cylindrical nanofins can generate the heart-shaped dark patterns (middle), whereas a birefringent metasurface composed of rectangular nanofins can create vectorial singularity sheets with undefined polarization azimuth (far right), as indicated by the black and white lines. Images reproduced from $(111)$ .

Multiwavelength control has been realized with interleaved superpixels, guided-mode resonances, coupled meta-atoms, and stacked metasurfaces, among others. Figure 4A shows one implementation of the former in which a dielectric metasurface manipulates the phase response of red, green, and blue wavelengths independently (113). The meta-molecule consists of three kinds of silicon nanoblocks, each imparting a geometric phase on a specific wavelength by varying the in-plane orientation of each waveplate-like nanofin. Despite its straightforward implementation, spatial interleaving imposes an upper limit on efficiency and introduces undesired meta-atom coupling, which degrades the image quality, producing ghost images and unwanted diffraction orders. To overcome these limitations, vertically stacked metasurfaces have been proposed where each layer is optimized for a particular incoming wavelength, enabling achromatic focusing and a multiwavelength response, albeit at the expense of complex fabrication requirements (114). These challenges can be mitigated by using dispersion-engineered metasurfaces such as the one depicted in Fig. 4B (115). This scheme makes use of a reflective metasurface to effectively double the propagation phase per pillar compared with operating in transmission. It also introduces wavelength-dependent guided-mode resonances, which arise when incident light couples to the leaky surface modes and reradiates into free space through phase matching. This creates rapid phase gradients around the resonances, enabling broad phase coverage while decoupling the phases imparted on each wavelength. Using this approach, red, green, and blue wavelengths have been mapped to three different phase profiles (Fig. 4B). Bilayer metasurfaces can further expand the design space, enabling wavelength-selective holography with complex-amplitude modulation, as depicted in Fig. 4C (116).

Metasurface-assisted configurations have also been used in pulse shaping and spatiotemporal light control. For example, Fig. 4D depicts a frequency-gradient metasurface created by combining a frequency comb source (centered at $\lambda = 720$ nm) with a passive metasurface (made of silicon nanopillars on sapphire substrate) to achieve ultrafast dy namic beam steering without any mechanical components (41). The underlying concept relies on mapping each incoming spectral line into a spatial optical mode with a different wave vector. Because the spectral lines are phase-locked, their individual spatial patterns constructively interfere to generate a 4D optical pattern in which the spatial light intensity distribution naturally evolves in time (steering $25^{\circ}$ in 8 ps).

Figure 4E shows another metasurface-based setup for 1D pulse shaping, realized by placing a metasurface made of polycrystalline silicon nanopillars at the focal plane of a Fourier-transform—that is, spectral dispersing-recombining—pulse synthesizer (117). The metasurface provides independent phase and amplitude control for each spectral component of a 10-fs pulse centered at 800 nm, thereby tailoring its temporal characteristics at will. More compact setups for pulse shaping, without the use of diffraction gratings, can be achieved through direct interaction with a single metasurface (Fig. 4F) (118). In this case, a broadband pulse compressor nanocoating compensates for the group delay dispersion of up to 2-mm-thick fused silica glass in the visible-to-NIR spectral region with a bandwidth of up to 80 nm. The group delay characteristics are determined by geometric properties of the nanopillars rather than their material dispersion, as shown in Fig. 4F, providing a versatile approach that can be adapted to different spectral regions and applications.

## Nonlinear metasurfaces

The use of light's degrees of freedom as control knobs can naturally be extended to harness nonlinear interactions. Note that obtaining a strong nonlinear response from optically thin structures requires much stronger light-matter interactions than natural bulk nonlinear media. Nonlinear metasurfaces tackle this dilemma by judiciously structuring the shape and size of the nanoscatterers, thereby enhancing the nonlinear effects over very small volumes by relaxing phase-matching and symmetry rules (119, 120). Earlier work relied on the excitation of localized surface plasmon polariton resonances to strongly enhance the electromagnetic field in the vicinity of the meta-atoms—an effect that can be tailored by designing the meta-atom geometries. Configurations of this type often use metal-dielectric interfaces where the nonlinearities stem from the asymmetry of the potential, confining the electrons at the material interface (121). For instance, Fig. 5A shows a meta-atom made of a split-ring resonator, which provides strong anisotropic response in the linear regime as well as highly efficient second-harmonic generation (SHG) (122). Circularly polarized light with spin $-\sigma$ interacting with this meta-atom (of angular orientation $\phi$ ) will accumulate a geometric phase of $2\sigma\phi$ ( $\sigma$ is $\pm1$ depending on the input polarization handedness), whereas the SHG signals with spin $\sigma$ and $-\sigma$ will acquire a Berry phase of $\sigma\phi$ and $3\sigma\phi$ , respectively. Thus, by designing orientation angles of each meta-atom, three different spatial phase distributions can be imparted on the fundamental, co- and cross-polarized SHG under left-handed circular polarization (LCP), as shown in Fig. 5B. Here, we refer to the Berry phase acquired by the SHG signal as the nonlinear Berry phase. Moreover, nonlinear metasurfaces based on multilayered V-shaped gold nanoantennas have enabled polarization-multiplexed holography on the third harmonic-generated (THG) signal ( $\lambda = 4:22$ nm), making it possible to project different images at multiple propagation distances by changing the incoming polarization (123).

Recently, all-dielectric metasurfaces have received much attention because of their low optical loss, strong field overlap, and systematic control over their dispersion properties, as reviewed in $(120)$ . Here, we highlight one application that made use of a silicon metal lens to enhance the nonlinear conversion efficiency for the THG process $(124)$ . Figure 5C depicts the proposed nonlinear lens scheme. Here, the fundamental wave (1550 nm) incorporates the object information, whereas the nonlinear image formed on the other side of the lens (517 nm) deviates in terms of size and location from the linear regime, obeying a modified Gaussian lens equation $(124)$ . Figure 5D shows scanning electron microscope (SEM) images of the fabricated lens in addition to an L-shaped aperture imaged using this lens. The longitudinal profiles of the fundamental and THG beams are shown in Fig. 5E. Notably, the THG image formed using this nonlinear scheme carries additional information about the spatial coherence of light emitted from the object, inferred from the high-order spatial correlations from different features of that object (not shown here).

More-complex behavior such as directional transmission based on asymmetric parametric conversion has been recently demonstrated. Devices of this kind can produce images in the visible spectral range when illuminated by infrared radiation while projecting different and independent images by reversing the direction of illumination (Fig. 5F) (125). Here, the resonators consist of two layers of materials: amorphous silicon and silicon nitride (Fig. 5G). Silicon has a higher refractive index and nonlinear susceptibility compared with silicon nitride. In this case, “forward” illumination leads to predominantly magnetic dipole-type scattering, whereas under “backward” illumination, the scattering is dominated by an electric dipole. The field enhancements within the nanoresonator are substantially higher for the forward excitation than they are for the backward direction. This leads to a drastic difference in brightness of the generated third-harmonic signal. Asymmetric harmonic generation can thus be used to generate two different images depending on the direction of illumination (Fig. 5H). This functionality can go beyond parametric generation of light and may find application in asymmetric generation of entangled photon states as well as realizing nonreciprocal behavior and optical isolation at the nanoscale.

![](images/b6e5ea8bbfaac9300ee8c65945bff61bf3d8ec9aff78c909fb9d9f2520239fa4.jpg)

A
![](images/4c54c19b37f2372ac9f703510fb1177ff80b04ebb6f1e9d6e83e5ff578c7d5fe.jpg)
B

![](images/3db9912076f1a97b8eeaeb317903821dd0ff2b186bc35c5af44e172a33524b7d.jpg)

C
![](images/ca5e748c9a12eb2d9cb8e3245b9bd898165e200db06720e8c507a8c50974eaab.jpg)

![](images/5965b57d81b54ad1a34a55e540756681c6dc06f6d40fa143e4847cf304dd4c47.jpg)

D
E
![](images/80b708340255acd8d6920966ebe97177765e9893ff421cf424113b8bd1d26876.jpg)

![](images/b573c9f6dda3e67e4ab6acef9e8d9406b3353efbfd64e13be4fbbb0b39b88bc3.jpg)

![](images/fa1b3d62dc79594642464fb799fd622cbe1744afbc82076f8362a5b8c98573fb.jpg)

![](images/b2daf486fe292a13003a54d5f6f19cceecc8292c55dead2176c7786f28353783.jpg)

![](images/7356423a5320b394d3a3ede4ba3cbd53d938be4ccc9581a88511390404764978.jpg)

![](images/97112a5576d1d3bd20fc94aa0f16f3f4241f2a436195683327a62e4e889614ab.jpg)

![](images/8ec9285c00c207894345f83aecb67260654c5de5e8c5042b20ad394c4f90d7f7.jpg)

![](images/6c11d214d698c8bc6cdeb982af03cf895e6120c1d2f1b39bdbf56bcec88536b7.jpg)
Fig. 4. Metasurface optics with multiwavelength control. (A) A highly dispersive meta-hologram can independently project distinct red, green, and blue images to reconstruct a holographic flower. The device relies on meta-molecules consisting of silicon nano blocks, each optimized for one wavelength, multiplexed altogether to form the metasurface. Partial SEM images of the fabricated metasurfaces are shown. Scale bar is 1 $\mu$ m. P, periodicity of the metamolecule. Reprinted with permission from (113). Copyright 2016 American Chemical Society. (B) Wavelength-controlled beam generator focuses incoming red, green, and blue Gaussian beams into OAM states $\ell$ of 0, 1, and 2, respectively. The device relies on phase shifters made of titanium dioxide (TiO $_{2}$ ) square nanopillars on top of a silver (Ag) substrate with a thin layer of silicon dioxide (SiO $_{2}$ ) in between. The reflection phase can be controlled by adjusting the width of the nanopillar. Measured output intensity profiles are shown on the right (in 1D and 2D) for each incident wavelength. Scale bars are 2 $\mu$ m. H, height; U, unit-cell length; W, width of the nanopillar. Reprinted with permission from (115). Copyright 2016 American Chemical Society. (C) Bilayer meta-holograms made of amorphous silicon nanopillars can project two independent (complex-amplitude modulated) logos in response to two different wavelengths, 1180 and 1680 nm. Images reproduced from (116). (D) Interaction between an input frequency-comb source and a passive
metasurface where each spectral line is mapped to a spatial optical mode, via diffraction and focusing, generating a spatiotemporal optical pattern with time-varying tilt for beam-steering applications. a, incoming pulse composed of a discrete superposition of weighted spectral lines; d, spatial separation between the output spatial optical modes; r, position; t, time. Images reproduced from (41). (E) Ultrafast pulse shaping using metasurface-based Fourier transform. The setup consists of a pair of diffraction gratings, two parabolic mirrors, and a metasurface that is divided into N superpixels along the x direction to tailor the temporal characteristics of the output pulse. Here, N is 660 superpixels, each 34 $\mu$ m in size. Temporal profiles of the targeted and measured output pulse are depicted at the bottom for a positively chirped input pulse (left) and a transform-limited input pulse (right). Images reproduced from (117). (F) Metasurface-based pulse compressor. Ultrashort pulses typically suffer from normal chromatic dispersion in transparent materials, which leads to pulse stretching. To mitigate these effects, a nanocoating (purple) can be applied to thin substrate to shorten elongated pulses or to thick optics to compensate for its group delay dispersion. Images reproduced from (118). (G) Measured and simulated compressor group delay profile compared with a fused silica substrate. The shaded regions mark spectral regions with different values of group delay dispersion. Images reproduced from (118).

## Summary and outlook

Over the past decade, the use of flat optics has been widely extended from wavefront shaping and focusing to more sophisticated manipulation of structured light, owing to rich meta-atom libraries, accurate full-wave simulations, and precise nanofabrication. Complex configurations of single- and multilayer meta-atoms have enabled multifunctional behavior by simply changing one or more degrees of freedom of input light. This ability allows a static monolithically integrated photonic component to rapidly switch its behavior without the need for active circuitry. We have highlighted recent progress in this field as well as its possible applications in holography, structured-light generation, and polarization control. Meta-surfaces with all-optical control knobs will serve as a key player in augmented reality and virtual reality (AR and VR) devices, 3D displays, and drone and automotive light detection and ranging (LIDAR) systems, owing to their light

![](images/6b43bc9ceddfb37671c366efae4db376174a865ec31d26c790ac4c5f6bdbb9ad.jpg)
B

![](images/a938879d6a01e77be416044afd125eb82a3ee89bdd12834112782dff2f7ee191.jpg)

![](images/67b312e33d5c22fc796baec084e609785da518e5193a876303213dccc1f54ceb.jpg)

![](images/3ef4d44e7564c102bd5254f8bd5e6da585b15df5c917014fc66d0362bb0540cc.jpg)
G

![](images/dc7988a72137358d85c46393e99e3aafba03583e288446e4eda9e806caf64af0.jpg)

C
![](images/3219c9aa422f4f47c41652987791d4f816119351f24579c4d699c2b5892169e0.jpg)

![](images/869b41903d5e16b25f0d372fa754229f614cea5fbfe0c5a3d5929c37ac5f51ab.jpg)

![](images/fc496a6f7ed9a123bb093eb7b1fdaad3947d77388aa2a63bec46c8dcea6eb3da.jpg)

D
E
![](images/d25a16840731de775925d1ee0d5539cf56b3b3a4de868cea790a2c6727576c26.jpg)

![](images/497812f763dc0dde385d53772dd5ffe1f55c42746186a1781d19dc4746ec09af.jpg)
H

![](images/23e0e7fbeddda5d3a3107af37994de2eed57a6640c967307b8ebe9903f3a2f44.jpg)

![](images/08a2c44200a1c5df338af2bf8a23f36d3912099f6d2f5c7be5fb78ca2a4ec33f.jpg)
Fig. 5. Nonlinear metasurface holography and imaging. (A) Linear and nonlinear Berry phases from a gold split-ring resonator with an orientation angle of $\phi$ . While the transmitted fundamental cross-circularly polarized (CP) component accumulates a linear geometric phase of $2\sigma \phi$ , CP and cross-CP components of the SHG incur nonlinear geometric phases of $\sigma \phi$ and $3\sigma \phi$ , respectively. $\omega$ , angular frequency. Images reproduced from (122). (B) The linear channel of the meta-hologram projects the letter "X" at IR wavelengths, whereas the second-harmonic channels encode the letters "R" and "L" on different circular polarization states. SEM images of the device and its output response are shown. Images reproduced from (122). (C) Schematic of image formation with a regular lens compared with that with a nonlinear lens made of $\chi^{(3)}$ material, where the size and location of the formed image are modified. a, position of the object; A, height of the object; b, image position; B, image height; F, focal length. Reproduced with permission from (124). (D) An SEM image of the fabricated all-dielectric nonlinear metalens is shown on the left. A micrograph of the fabricated L-shaped aperture placed at an object distance $-300~{\mu\mathrm{m}}$ in front of the metalens is shown on the right. Although the conventional lens equation predicts image formation at infinity, the focal plane and the formed
image of the THG lie at a z of 300 and 450 $\mu$ m at the back focal plane of the metalens, respectively, governed by the generalized Gaussian lens equation. Reproduced with permission from (124). (E) Measured longitudinal intensity profiles of the fundamental (red) and THG (green) beams for the fundamental wavelength of 1550 nm. Reproduced with permission from (124). (F) Asymmetric parametric generation of images with a nonlinear metasurface. Different and independent THG images are generated in transmission, depending on the direction of illumination. Reproduced with permission from (125). (G) The metasurface is made of an anisotropic cylindrical nanoresonator consisting of silicon (gray) and silicon nitride (green) embedded into glass (left). Arrows visualize forward and backward excitation. Near-field distributions of the electric field at the fundamental ( $\lambda = 1475$ nm) and third-harmonic signal ( $\lambda = 492$ nm) for the forward and backward directions show high transmission contrast (right). Yellow lines mark contours of the bilayer nanoresonator. Reproduced with permission from (125). (H) SEM images of the meta-hologram. Its nonlinear optical response detected in transmission at the third-harmonic frequency is shown on the right under forward and backward excitation at a wavelength of 1475 nm. Reproduced with permission from (125).

weight, ease of integration, and compact footprint. For instance, an autonomous vehicle can make use of metasurface-based polarization cameras to distinguish between different features on a road environment (e.g., water versus mud puddles) without the need for expensive and bulky camera systems $(89)$ . Additionally, nonlocal metalenses can be deployed in AR and VR wearable devices to reflect NIR light onto the iris for accurate eye tracking while transmitting visible light, ensuring unperturbed vision of the outside world $(126)$ . Moreover, integrating multifunctional metasurfaces with quantum emitters can enable efficient single-photon quantum sources and quantum-entangled photon states, whereas combining metasurfaces with single-photon-sensitive charge-coupled device cameras could allow multiple timeframe imaging and fast quantum measurements (127). Furthermore, complex states of classical and nonclassical structured light can be generated with metasurface-assisted laser cavities (128). The list goes on, with numerous applications that can make use of multifunctional flat optics, from laser beam shaping to optical communications, biomedical sensing, and imaging (129).

Nevertheless, several open challenges are still underway in metasurface research and applications. At the physical layer, topology-optimized meta-atom libraries seek to expand the function of forward-designed nanoantennas by exploring a larger design space of free-form geometries (44). Inverse-design and machine-learning techniques will be key tools for addressing this challenge by searching nonintuitive design spaces (130, 131). More accurate models that account for the metaatom coupling, in the absence of periodic boundary conditions, are also being tested now (132). Additionally, bilayer metasurfaces are being investigated as means for enhancing the diffraction efficiency (133) and realizing more versatile polarization control by relaxing the Jones matrix symmetry constraint imposed by their single-layer counterpart (116, 123). With regard to mass production, several fabrication techniques are being developed to realize large area and conformal metasurfaces. From a light-wave standpoint, we expect that more attention will be given to the full vectorial nature of nonparaxial light (e.g., to control its longitudinal field component) as an additional degree of freedom (134). Likewise, optical coherence will be more heavily investigated as a fundamental property of light that acts as an additional control knob if combined with engineering of the point spread function of the metasurface. In the near future, we envision that hybrid configurations of static and active metasurfaces will be integrated to realize the best of both worlds: high spatial resolution and time-varying response. With this versatile level of control, pushing structured light from 2D to 3D and from frozen to animate will help unlock a rich palette of optical phenomena throughout the next decade, from the atomic to the astrophysical scale.

## REFERENCES AND NOTES

1. P. Vukusic, J. R. Sambles, Photonic structures in biology. Nature 424, 852–855 (2003). doi: 10.1038/nature01941; pmid: 12917700

2. N. Capobianco et al., The Grande Rose of the Reims Cathedral: An eight-century perspective on the colour management of medieval stained glass. Sci. Rep. 9, 3287 (2019). doi: 10.1038/s41598-019-39740-y; pmid: 30824744

3. L. Rayleigh, XVII. On the maintenance of vibrations by forces of double frequency, and on the propagation of waves through a medium endowed with a periodic structure. London Edinb. Dublin Philos. Mag. J. Sci. 24, 145–159 (1887). doi: 10.1080/14786448708628074

4. E. Yablonovitch, Photonic band-gap structures. J. Opt. Soc. Am. B 10, 283 (1993). doi: 10.1364/JOSAB.10.000283

5. E. Yablonovitch, Inhibited spontaneous emission in solid-state physics and electronics. Phys. Rev. Lett. 58, 2059–2062 (1987). doi: 10.1103/PhysRevLett.58.2059; pmid: 10034639

6. S. John, Strong localization of photons in certain disordered dielectric superlattices. Phys. Rev. Lett. 58, 2486–2489 (1987). doi: 10.1103/PhysRevLett.58.2486; pmid: 10034761

7. P. Lalanne, D. Lemercier-lalanne, On the effective medium theory of subwavelength periodic structures. J. Mod. Opt. 43, 2063–2085 (1996). doi: 10.1080/09500349608232871

8. G. W. Milton, Reformulating the Problem of Finding Effective Tensors (Cambridge Univ. Press, 2002), pp. 245–270.

9. V. G. Veselago, The electrodynamics of substances with simultaneously negative values of $\epsilon$ and $\mu$ . Sov. Phys. Usp. 10, 509–514 (1968). doi: 10.1070/PU1968v010n04ABEH003699

10. R. A. Shelby, D. R. Smith, S. Schultz, Experimental verification of a negative index of refraction. Science 292, 77–79 (2001). doi: 10.1126/science.1058847; pmid: 11292865

11. J. Yao et al., Optical negative refraction in bulk metamaterials of nanowires. Science 321, 930 (2008). doi: 10.1126/science.1157566; pmid: 18703734

12. R. W. Ziolkowski, N. Engheta, Introduction, History, and Selected Topics in Fundamental Theories of Metamaterials (Wiley, 2006), chap. 1, pp. 1–41.

13. M. Kadic, G. W. Milton, M. van Hecke, M. Wegener, 3D metamaterials. Nat. Rev. Phys. 1, 198–210 (2019). doi: 10.1038/s42254-018-0018-y

14. A. M. Urbas et al., Roadmap on optical metamaterials. J. Opt. 18, 093005 (2016). doi: 10.1088/2040-8978/18/9/093005

15. A. V. Kildishev, A. Boltasseva, V. M. Shalaev, Planar photonics with metasurfaces. Science 339, 1232009 (2013). doi: 10.1126/science.1232009; pmid: 23493714

16. N. Yu, F. Capasso, Flat optics with designer metasurfaces. Nat. Mater. 13, 139–150 (2014). doi: 10.1038/nmat3839; pmid: 24452357

17. D. Lin, P. Fan, E. Hasman, M. L. Brongersma, Dielectric gradient metasurface optical elements. Science 345, 298–302 (2014). doi: 10.1126/science.1253213; pmid: 25035488

18. H.-T. Chen, A. J. Taylor, N. Yu, A review of metasurfaces: Physics and applications. Rep. Prog. Phys. 79, 076401 (2016). doi: 10.1088/0034-4885/79/7/076401; pmid: 27308726

19. P. Genevet, F. Capasso, F. Aieta, M. Khorasaninejad, R. Devlin, Recent advances in planaroptics: From plasmonic to dielectric metasurfaces. Optica 4, 139 (2017). doi: 10.1364/OPTICA.4.000139

20. S. M. Kamali, E. Arbabi, A. Arbabi, A. Faraon, A review of dielectric optical metasurfaces for wavefront control. Nanophotonics 7, 1041–1068 (2018). doi: 10.1515/nanoph-2017-0129

21. T. W. Ebbesen, H. J. Lezec, H. F. Ghaemi, T. Thio, P. A. Wolff, Extraordinary optical transmission through sub-wavelength hole arrays. Nature 391, 667–669 (1998). doi: 10.1038/35570

22. J. B. Pendry, Negative refraction makes a perfect lens. Phys. Rev. Lett. 85, 3966–3969 (2000). doi: 10.1103/PhysRevLett.85.3966; pmid: 11041972

23. H. Liu, P. Lalanne, Microscopic theory of the extraordinary optical transmission. Nature 452, 728–731 (2008). doi: 10.1038/nature06762; pmid: 18401405

24. N. Yu et al., Light propagation with phase discontinuities: Generalized laws of reflection and refraction. Science 334, 333–337 (2011). doi: 10.1126/science.1210713; pmid: 21885733

25. P. Lalanne, S. Astilean, P. Chavel, E. Cambril, H. Launois, Design and fabrication of blazed binary diffractive elements with sampling periods smaller than the structural cutoff. J. Opt. Soc. Am. A 16, 1143 (1999). doi: 10.1364/JOSAA16.001143

26. Z. Bomzon, G. Biener, V. Kleiner, E. Hasman, Space-variant Pancharatnam-Berry phase optical elements with computer-generated subwavelength gratings. Opt. Lett. 27, 1141–1143 (2002). doi: 10.1364/OL.27.001141; pmid: 18026387

27. M. Ozaki, J. Kato, S. Kawata, Surface-plasmon holography with white-light illumination. Science 332, 218–220 (2011). doi: 10.1126/science.1201045; pmid: 21474756

28. P. Genevet, F. Capasso, Holographic optical metasurfaces: A review of current progress. Rep. Prog. Phys. 78, 024401 (2015). doi: 10.1088/0034-4885/78/2/024401; pmid: 25609665

29. R. Zhao, L. Huang, Y. Wang, Recent advances in multidimensional metasurfaces holographic technologies. PhotoniX 1, 20 (2020). doi: 10.1186/s43074-020-00020-y

30. M. Khorasaninejad et al., Metalenses at visible wavelengths: Diffraction-limited focusing and subwavelength resolution imaging. Science 352, 1190–1194 (2016). doi: 10.1126/science.aaf6644; pmid: 27257251

31. M. Khorasaninejad, F. Capasso, Metalenses: Versatile multifunctional photonic components. Science 358, eaam8100 (2017). doi: 10.1126/science.aam8100; pmid: 28982796

32. A. Arbabi, Y. Horie, M. Bagheri, A. Faraon, Dielectric metasurfaces for complete control of phase and polarization with subwavelength spatial resolution and high transmission. Nat. Nanotechnol. 10, 937–943 (2015). doi: 10.1038/nnano.2015.186; pmid: 26322944

33. J. P. Balthasar Mueller, N. A. Rubin, R. C. Devlin, B. Groever, F. Capasso, Metasurface polarization optics: Independent phase control of arbitrary orthogonal states of polarization. Phys. Rev. Lett. 118, 113901 (2017). doi: 10.1103/PhysRevLett.118.113901; pmid: 28368630

34. W. T. Chen, A. Y. Zhu, F. Capasso, Flat optics with dispersion engineered metasurfaces. Nat. Rev. Mater. 5, 604–620 (2020). doi: 10.1038/s41578-020-0203-3

35. Y. Hu et al., All-dielectric metasurfaces for polarization manipulation: Principles and emerging applications. Nanophotonics 9, 3755–3780 (2020). doi: 10.1515/nanoph-2020-0220

36. N. Rubin, Z. Shi, F. Capasso, Polarization in diffractive optics and metasurfaces. Adv. Opt. Photonics 13, 836 (2021). doi: 10.1364/AOP.439986

37. L. Huang, S. Zhang, T. Zentgraf, Metasurface holography: From fundamentals to applications. Nanophotonics 7, 1169–1190 (2018). doi: 10.1515/nanoph-2017-0118

38. A. L. Holsteen, A. F. Cihan, M. L. Brongersma, Temporal color mixing and dynamic beam shaping with silicon metasurfaces. Science 365, 257–260 (2019). doi: 10.1126/science.aax5961; pmid: 31320534

39. A. M. Shaltout, V. M. Shalaev, M. L. Brongersma, Spatiotemporal light control with active metasurfaces. Science 364, eaat3100 (2019). doi: 10.1126/science.aat3100; pmid: 31097638

40. X. Guo, Y. Ding, Y. Duan, X. Ni, Nonreciprocal metasurface with space-time phase modulation. Light Sci. Appl. 8, 123 (2019). doi: 10.1038/s41377-019-0225-z; pmid: 31871675

41. A. M. Shaltout et al., Spatiotemporal light control with frequency-gradient metasurfaces. Science 365, 374–377 (2019). doi: 10.1126/science.aax2357; pmid: 31346064

42. W. Liu, Z. Li, H. Cheng, S. Chen, Dielectric resonance-based optical metasurfaces: From fundamentals to applications. iScience 23, 101868 (2020). doi: 10.1016/j.isci.2020.101868; pmid: 33319185

43. K. Wu, P. Coquet, Q. J. Wang, P. Genevet, Modelling of free-form conformal metasurfaces. Nat. Commun. 9, 3494 (2018). doi: 10.1038/s41467-018-05579-6; pmid: 30154424

44. Z. Shi et al., Continuous angle-tunable birefringence with freeform metasurfaces for arbitrary polarization conversion. Sci. Adv. 6, eaba3367 (2020). doi: 10.1126/sciadv.aba3367; pmid: 32537506

45. A. Forbes, M. de Oliveira, M. R. Dennis, Structured light. Nat. Photonics 15, 253–262 (2021). doi: 10.1038/s41566-021-00780-4

46. A. C. Overvig, S. C. Malek, N. Yu, Multifunctional nonlocal metasurfaces. Phys. Rev. Lett. 125, 017402 (2020). doi: 10.1103/PhysRevLett.125.017402; pmid: 32678662

47. S. M. Kamali et al., Angle-multiplexed metasurfaces: Encoding independent wavefronts in a single metasurface under different illumination angles. Phys. Rev. X 7, 041056 (2017). doi: 10.1103/PhysRevX.7.041056

48. X. Zhang et al., Ultrahigh-capacity dynamic holographic displays via anisotropic nanoholes. Nanoscale 9, 1409–1415 (2017). doi: 10.1039/C6NR07854K; pmid: 28074963

49. S. Wan et al., Angular-multiplexing metasurface: Building up independent-encoded amplitude/phase dictionary for angular illumination. Adv. Opt. Mater. 9, 2101547 (2021). doi: 10.1002/adom.202101547

50. X. Zhang et al., Controlling angular dispersions in optical metasurfaces. Light Sci. Appl. 9, 76 (2020). doi: 10.1038/s41377-020-0313-0; pmid: 32411361

51. J. Jang, G.-Y. Lee, J. Sung, B. Lee, Independent multichannel wavefront modulation for angle multiplexed meta-holograms. Adv. Opt. Mater. 9, 2100678 (2021). doi: 10.1002/adom.202100678

52. A. Leitis et al., Angle-multiplexed all-dielectric metasurfaces for broadband molecular fingerprint retrieval. Sci. Adv. 5, eaaw2871 (2019). doi: 10.1126/sciadv.aaw2871; pmid: 31123705

53. E. Wang et al., Complete control of multichannel, angle-multiplexed, and arbitrary spatially varying polarization fields. Adv. Opt. Mater. 8, 1901674 (2020). doi: 10.1002/adom.201901674

54. A. Silva et al., Performing mathematical operations with metamaterials. Science 343, 160–163 (2014). doi: 10.1126/science.1242818; pmid: 24408430

55. C. Guo, M. Xiao, M. Minkov, Y. Shi, S. Fan, Photonic crystal slab laplace differentiator. Optica 5, 251 (2017). doi: 10.1364/OPTICA.5.000251

56. H. Kwon, D. Sounas, A. Cordaro, A. Polman, A. Alù, Nonlocal metasurfaces for optical signal processing. Phys. Rev. Lett. 121, 173004 (2018). doi: 10.1103/PhysRevLett.121.173004; pmid: 30411907

57. Y. Zhou, H. Zheng, I. I. Kravchenko, J. Valentine, Flat optics for image differentiation. Nat. Photonics 14, 316–323 (2020) doi: 10.1038/s41566-020-0591-3

58. C. Guo, H. Wang, S. Fan, Squeeze free space with nonlocal flat optics. Optica 7, 1133 (2020). doi: 10.1364/OPTICA.392978

59. O. Reshef et al., An optic to replace space and its application towards ultra-thin imaging systems. Nat. Commun. 12, 3512 (2021). doi: 10.1038/s41467-021-23358-8; pmid: 34112771

60. C. Spägele et al., Multifunctional wide-angle optics and lasing based on supercell metasurfaces. Nat. Commun. 12, 3787 (2021). doi: 10.1038/s41467-021-24071-2; pmid: 34145275

61. T. Kim et al., Asymmetric optical camouflage: Tuneable reflective colour accompanied by the optical Janus effect. Light Sci. Appl. 9, 175 (2020). doi: 10.1038/s41377-020-00413-5; pmid: 33088492

62. Y. Chen, X. Yang, J. Gao, 3D Janus plasmonic helical nanoapertures for polarization-encrypted data storage. Light Sci. Appl. 8, 45 (2019). doi: 10.1038/s41377-019-0156-8; pmid: 31098013

63. K. Chen et al., Directional Janus metasurface. Adv. Mater. 32, e1906352 (2020). doi: 10.1002/adma.201906352; pmid: 31746042

64. G. Yoon, D. Lee, K. T. Nam, J. Rho, "Crypto-display" in dual-mode metasurfaces by simultaneous control of phase and spectral responses. ACS Nano 12, 6421–6428 (2018). doi: 10.1021/acsnano.8b01344; pmid: 29924588

65. Z. Li et al., Full-space cloud of random points with a scrambling metasurface. Light Sci. Appl. 7, 63 (2018). doi: 10.1038/s41377-018-0064-3; pmid: 30245810

66. E. Cohen et al., Geometric phase from Aharonov–Bohm to Pancharatnam–Berry and beyond. Nat. Rev. Phys. 1, 437–449 (2019). doi: 10.1038/s42254-019-0071-1

67. Q. Hu, C. He, M. J. Booth, Arbitrary complex retarders using a sequence of spatial light modulators as the basis for adaptive polarisation compensation. J. Opt. 23, 065602 (2021). doi: 10.1088/2040-8986/abed33

68. L. Nikolova, T. Todorov, Diffraction efficiency and selectivity of polarization holographic recording. Opt. Acta 31, 579–588 (1984). doi: 10.1080/713821547

69. J. Tervo, J. Turunen, Paraxial-domain diffractive elements with 100% efficiency based on polarization gratings. Opt. Lett. 25, 785–786 (2000). doi: 10.1364/OL.25.000785; pmid: 18064183

70. Z. Bomzon, G. Biener, V. Kleiner, E. Hasman, Radially and azimuthally polarized beams generated by space-variant dielectric subwavelength gratings. Opt. Lett. 27, 285–287 (2002). doi: 10.1364/OL.27.000285; pmid: 18007778

71. Y. Zhao, A. Alu, Manipulating light polarization with ultrathin plasmonic metasurfaces. Phys. Rev. B 84, 205428 (2011). doi: 10.1103/PhysRevB.84.205428

72. N. Yu et al., A broadband, background-free quarter-wave plate based on plasmonic metasurfaces. Nano Lett. 12, 6328–6333 (2012). doi: 10.1021/nl303445u; pmid: 23130979

73. K. Y. Bliokh, F. J. Rodriguez-Fortuno, F. Nori, A. V. Zayats, Spin-orbit interactions of light. Nat. Photonics 9, 796–808 (2015). doi: 10.1038/nphoton.2015.201

74. A. M. Yao, M. J. Padgett, Orbital angular momentum: Origins, behavior and applications. Adv. Opt. Photonics 3, 161 (2011). doi: 10.1364/AOP.3.000161

75. L. Marrucci, C. Manzo, D. Paparo, Optical spin-to-orbital angular momentum conversion in inhomogeneous anisotropic media. Phys. Rev. Lett. 96, 163905 (2006). doi: 10.1103/PhysRevLett.96.163905; pmid: 16712234

76. E. Maguid et al., Photonic spin-controlled multifunctional shared-aperture antenna array. Science 352, 1202–1206 (2016). doi: 10.1126/science.aaf3417; pmid: 27103668

77. W.-T. Chen et al., High-efficiency broadband meta-hologram with polarization-controlled dual images. Nano Lett. 14, 225–230 (2014). doi: 10.1021/nl403811d; pmid: 24329425

78. S. Kruk et al., Invited article: Broadband highly efficient dielectric meta-devices for polarization control. APL Photonics 1, 030801 (2016). doi: 10.1063/1.4949007

79. S. Kruk et al., Transparent dielectric metasurfaces for spatial mode multiplexing. Laser Photonics Rev. 12, 1800031 (2018). doi: 10.1002/lpor.201800031

80. E. Nazemosadat et al., Dielectric broadband metasurfaces for fiber mode multiplexed communications. Adv. Opt. Mater. 7, 1801679 (2019). doi: 10.1002/adom.201801679

81. R. C. Devlin, A. Ambrosio, N. A. Rubin, J. P. B. Mueller, F. Capasso, Arbitrary spin-to-orbital angular momentum conversion of light. Science 358, 896–901 (2017). doi: 10.1126/science.aao5392; pmid: 29097490

82. N. A. Rubin, A. Zaidi, A. H. Dorrah, Z. Shi, F. Capasso, Jones matrix holography with metasurfaces. Sci. Adv. 7, eabg7488 (2021). doi: 10.1126/sciadv.abg7488; pmid: 34389537

83. M. Liu et al., Multifunctional metasurfaces enabled by simultaneous and independent control of phase and amplitude for orthogonal polarization states. Light Sci. Appl. 10, 107 (2021). doi: 10.1038/s41377-021-00552-3; pmid: 34035215

84. Q. Fan et al., Independent amplitude control of arbitrary orthogonal states of polarization via dielectric metasurfaces. Phys. Rev. Lett. 125, 267402 (2020). doi: 10.1103/PhysRevLett.125.267402; pmid: 33449781

85. Y. Yuan et al., Independent phase modulation for quadruplex polarization channels enabled by chirality-assisted geometric-phase metasurfaces. Nat. Commun. 11, 4186 (2020). doi: 10.1038/s41467-020-17773-6; pmid: 32826879

86. R. Zhao et al., Multichannel vectorial holographic display and encryption. Light Sci. Appl. 7, 95 (2018). doi: 10.1038/s41377-018-0091-0; pmid: 30510691

87. Y. Hu et al., Trichromatic and tripolarization-channel holography with noninterleaved dielectric metasurface. Nano Lett. 20, 994–1002 (2020). doi: 10.1021/acs.nanolett.9b04107; pmid: 31880939

88. R. A. Chipman, W. S. T. Lam, G. Young, Polarized Light and Optical Systems (CRC Press, 2019).

89. N. A. Rubin et al., Matrix Fourier optics enables a compact full-Stokes polarization camera. Science 365, eaax1839 (2019). doi: 10.1126/science.aax1839; pmid: 31273096

90. Z.-L. Deng et al., Diatomic metasurface for vectorial holography. Nano Lett. 18, 2885–2892 (2018). doi: 10.1021/acs.nanolett.8b00047; pmid: 29590530

91. F. Ding et al., Versatile polarization generation and manipulation using dielectric metasurfaces. Laser Photonics Rev. 14, 2000116 (2020). doi: 10.1002/lpor.202000116

92. E. Arbabi, S. M. Kamali, A. Arbabi, A. Faraon, Vectorial holograms with a dielectric metasurface: Ultimate polarization pattern generation. ACS Photonics 6, 2712–2718 (2019). doi: 10.1021/acsphotonics.9b00678

93. Q. Song et al., Ptychography retrieval of fully polarized holograms from geometric-phase metasurfaces. Nat. Commun. 11, 2651 (2020). doi: 10.1038/s41467-020-16437-9; pmid: 32461637

94. Z.-L. Deng et al., Full-color complex-amplitude vectorial holograms based on multi-freedom metasurfaces. Adv. Funct. Mater. 30, 1910610 (2020). doi: 10.1002/adfm.201910610

95. D. Wen, J. J. Cadusch, J. Meng, K. B. Crozier, Vectorial holograms with spatially continuous polarization distributions. Nano Lett. 21, 1735–1741 (2021). doi: 10.1021/acs.nanolett.0c04555; pmid: 33544611

96. A. Pors, M. G. Nielsen, S. I. Bozhevolnyi, Plasmonic metagratings for simultaneous determination of Stokes parameters. Optica 2, 716 (2015). doi: 10.1364/OPTICA.2.000716

97. S. Wei, Z. Yang, M. Zhao, Design of ultracompact polarimeters based on dielectric metasurfaces. Opt. Lett. 42, 1580–1583 (2017). doi: 10.1364/OL.42.001580; pmid: 28409803

98. E. Arbabi, S. M. Kamali, A. Arbabi, A. Faraon, Full-Stokes imaging polarimetry using dielectric metasurfaces. ACS Photonics 5, 3132–3140 (2018). doi: 10.1021/acsphotonics.8b00362

99. A. H. Dorrah, N. A. Rubin, A. Zaidi, M. Tamagnone, F. Capasso, Metasurface optics for on-demand polarization transformations along the optical path. Nat. Photonics 15, 287–296 (2021). doi: 10.1038/s41566-020-00750-2

100. A. H. Dorrah, N. A. Rubin, M. Tamagnone, A. Zaidi, F. Capasso, Structuring total angular momentum of light along the propagation direction with polarization-controlled meta-optics. Nat. Commun. 12, 6249 (2021). doi: 10.1038/s41467-021-26253-4; pmid: 34716326

101. A. H. Dorrah, M. Zamboni-Rached, M. Mojahedi, Experimental demonstration of tunable refractometer based on orbital angular momentum of longitudinally structured light. Light Sci. Appl. 7, 40 (2018). doi: 10.1038/s41377-018-0034-9; pmid: 30839632

102. Y. Qin, Y. Li, D. Deng, Y. Liu, M. Sun, Ultracompact biosensor based on a metalens with a longitudinally structured vector beam. Appl. Opt. 58, 4438–4442 (2019). doi: 10.1364/AO.58.004438; pmid: 31251258

103. Q. Tian et al., The propagation properties of a longitudinal orbital angular momentum multiplexing system in atmospheric turbulence. IEEE Photonics J. 10, 1–16 (2018). doi: 10.1109/JPHOT.2017.2778238

104. R. A. B. Suarez, L. A. Ambrosio, A. A. R. Neves, M. Zamboni-Rached, M. R. R. Gesualdi, Experimental optical trapping with frozen waves. Opt. Lett. 45, 2514–2517 (2020). doi: 10.1364/OL.390909; pmid: 32356804

105. H. Ren et al., Metasurface orbital angular momentum holography. Nat. Commun. 10, 2986 (2019). doi: 10.1038/s41467-019-11030-1; pmid: 31324755

106. H. Ren et al., Complex-amplitude metasurface-based orbital angular momentum holography in momentum space. Nat. Nanotechnol. 15, 948–955 (2020). doi: 10.1038/s41565-020-0768-4; pmid: 32958936

107. X. Fang, H. Ren, M. Gu, Orbital angular momentum holography for high-security encryption. Nat. Photonics 14, 102–108 (2020). doi: 10.1038/s41566-019-0560-x

108. H. Zhou et al., Polarization-encrypted orbital angular momentum multiplexed metasurface holography. ACS Nano 14, 5553–5559 (2020). doi: 10.1021/acsnano.9b09814; pmid: 32348122

109. G. Qu et al., Reprogrammable meta-hologram for optical encryption. Nat. Commun. 11, 5484 (2020). doi: 10.1038/s41467-020-19312-9; pmid: 33127918

110. P. Georgi et al., Optical secret sharing with cascaded metasurface holography. Sci. Adv. 7, eabf9718 (2021). doi: 10.1126/sciadv.abf9718; pmid: 33853788

111. S. W. D. Lim, J.-S. Park, M. L. Meretska, A. H. Dorrah, F. Capasso, Engineering phase and polarization singularity sheets. Nat. Commun. 12, 4190 (2021). doi: 10.1038/s41467-021-24493-v; pmid: 34234140

112. G. H. Yuan, N. I. Zheludev, Detecting nanometric displacements with optical ruler metrology. Science 364, 771–775 (2019). doi: 10.1126/science.aaw7840; pmid: 31072905

113. B. Wang et al., Visible-frequency dielectric metasurfaces for multiwavelength achromatic and highly dispersive holograms.

Nano Lett. 16, 5235–5240 (2016). doi: 10.1021/acs.nanolett.6b02326; pmid: 27398793

114. O. Avayu, E. Almeida, Y. Prior, T. Ellenbogen, Composite functional metasurfaces for multispectral achromatic optics. Nat. Commun. 8, 14992 (2017). doi: 10.1038/ncomms14992; pmid: 28378810

115. Z. Shi et al., Single-layer metasurface with controllable multiwavelength functions. Nano Lett. 18, 2420–2427 (2018). doi: 10.1021/acs.nanolett.7b05458; pmid: 29461838

116. Y. Zhou et al., Multifunctional metaoptics based on bilayer metasurfaces. Light Sci. Appl. 8, 80 (2019). doi: 10.1038/s41377-019-0193-3; pmid: 31666946

117. S. Divitt, W. Zhu, C. Zhang, H. J. Lezec, A. Agrawal, Ultrafast optical pulse shaping using dielectric metasurfaces. Science 364, 890–894 (2019). doi: 10.1126/science.aav9632; pmid: 31048550

118. M. Ossiander et al., Slow light nanocoatings for ultrashort pulse compression. Nat. Commun. 12, 6518 (2021). doi: 10.1038/s41467-021-26920-6; pmid: 34764297

119. G. Li, S. Zhang, T. Zentgraf, Nonlinear photonic metasurfaces. Nat. Rev. Mater. 2, 17010 (2017). doi: 10.1038/natrevmats.2017.10

120. T. Pertsch, Y. Kivshar, Nonlinear optics with resonant metasurfaces. MRS Bull. 45, 210–220 (2020). doi: 10.1557/mrs.2020.65

121. A. Krasnok, M. Tymchenko, A. Alu, Nonlinear metasurfaces: A paradigm shift in nonlinear optics. Mater. Today 21, 8–21 (2018). doi: 10.1016/j.mattod.2017.06.007

122. W. Ye et al., Spin and wavelength multiplexed nonlinear metasurface holography. Nat. Commun. 7, 11930 (2016). doi: 10.1038/ncomms11930; pmid: 27306147

123. E. Almeida, O. Bitton, Y. Prior, Nonlinear metamaterials for holography. Nat. Commun. 7, 12533 (2016). doi: 10.1038/ncomms12533; pmid: 27545581

124. C. Schlickriede et al., Nonlinear imaging with all-dielectric metasurfaces. Nano Lett. 20, 4370–4376 (2020). doi: 10.1021/acs.nanolett.0c01105; pmid: 32374616

125. S. Kruk et al., Asymmetric parametric generation of images with nonlinear dielectric metasurfaces. arXiv:2108.04425 [physics.optics] (2021).

126. J.-H. Song, J. van de Groep, S. J. Kim, M. L. Brongersma, Non-local metasurfaces for spectrally decoupled wavefront manipulation and eye tracking. Nat. Nanotechnol. 16, 1224–1230 (2021). doi: 10.1038/s41565-021-00967-4; pmid: 34594006

127. A. S. Solntsev, G. S. Agarwal, Y. S. Kivshar, Metasurfaces for quantum photonics. Nat. Photonics 15, 327–336 (2021). doi: 10.1038/s41566-021-00793-z

128. H. Sroor et al., High-purity orbital angular momentum states from a visible metasurface laser. Nat. Photonics 14, 498–503 (2020). doi: 10.1038/s41566-020-0623-z

129. C.-W. Qiu, T. Zhang, G. Hu, Y. Kivshar, Quo vadis, metasurfaces? Nano Lett. 21, 5461–5474 (2021). doi: 10.1021/acs.nanolett.1c00828; pmid: 34157842

130. C. M. Lalau-Keraly, S. Bhargava, O. D. Miller, E. Yablonovitch, Adjoint shape optimization applied to electromagnetic design. Opt. Express 21, 21693–21701 (2013). doi: 10.1364/OE.21.021693; pmid: 24104043

131. L. Li et al., Machine-learning reprogrammable metasurface imager. Nat. Commun. 10, 1082 (2019). doi: 10.1038/s41467-019-09103-2; pmid: 30842417

132. M. Mansouree, A. McClung, S. Samudrala, A. Arbabi, Large-scale parametrized metasurface design using adjoint optimization. ACS Photonics 8, 455–463 (2021). doi: 10.1021/acsphotonics.0c01058

133. M. Mansouree et al., Multifunctional 2.5D metastructures enabled by adjoint optimization. Optica 7, 77 (2020). doi: 10.1364/OPTICA.374787

134. H. Ren, W. Shao, Y. Li, F. Salim, M. Gu, Three-dimensional vectorial holography based on machine learning inverse design. Sci. Adv. 6, eaaz4261 (2020). doi: 10.1126/sciadv.aaz4261; pmid: 32494614

## ACKNOWLEDGMENTS

Funding: We acknowledge financial support from the National Science Foundation (grant no. ECCS-2025158), Office of Naval Research (grant no. N00014-20-1-2450), Air Force Office of Scientific Research (grant no. FA95550-19-1-0135), and Natural Sciences and Engineering Research Council of Canada (grant no. PDF-533013-2019). Competing interests: The authors declare that they have no competing interests.

10.1126/science.abi6860
