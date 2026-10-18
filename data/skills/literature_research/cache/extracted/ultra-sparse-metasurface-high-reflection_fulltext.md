# Ultra-sparse metasurface for high reflection of low-frequency sound based on artificial Mie resonances

Y. Cheng $^{1,2}$ , C. Zhou $^{1}$ , B. G. Yuan $^{1}$ , D. J. Wu $^{3}$ , Q. Wei $^{1}$ and X. J. Liu $^{1,2*}$

Acoustic metamaterials offer great flexibility for manipulating sound waves and promise unprecedented functionality, ranging from transformation acoustics, super-resolution imaging to acoustic cloaking. However, the design of acoustic metamaterials with exciting functionality remains challenging with traditional approaches using classic acoustic elements such as Helmholtz resonators and membranes. Here we demonstrate an ultraslow-fluid-like particle with intense artificial Mie resonances for low-frequency airborne sound. Eigenstate analysis and effective parameter retrieval show two individual negative bands in the single-size unit cell, one of which exhibits a negative bulk modulus supported by the monopolar Mie resonance, whereas the other exhibits a negative mass density induced by the dipolar Mie resonance. The unique single-negative nature is used to develop an ultra-sparse subwavelength metasurface with high reflectance for low-frequency sound. We demonstrate a 0.15λ-thick, 15%-filling ratio metasurface with an insertion loss over 93.4%. The designed Mie resonators provide diverse routes to construct novel acoustic devices with versatile applications.

Aoustic metamaterials denote those artificial structures with elastic responses beyond their natural constitutive materials $^{1}$ . In the past decade, great progress in acoustic metamaterials has been made with regard to their novel physical properties and wide range of potential applications $^{2-22}$ , such as acoustic cloaking $^{4,6,9,19,20}$ , negative refraction of sound waves $^{3,7}$ , and ultrasonic imaging with subwavelength resolution $^{10,13,16,22}$ . So far, most acoustic metamaterials have been constructed based on the design principle of LC resonance theory, which enables a phase shift of $\pi$ in material response over the resonance frequency. The use of structured resonant elements on a scale much shorter than their operating wavelength permits their effective parameters to take negative values. For instance, the first locally resonant sonic metamaterials with negative dynamic mass density have been fabricated by means of a high-density solid core and soft coating, acting as mass and spring respectively $^{2}$ . A further class of ultrasonic metamaterials with negative effective dynamic bulk modulus was designed, consisting of subwavelength Helmholtz resonators in which the short neck and cavity, respectively, act as an acoustic mass and capacitance $^{5}$ . Other structured resonant elements, such as tensioned membranes $^{23-25}$ , hybrid elastic unit cells $^{26}$ , and transmission line networks $^{6,21}$ , have also been successfully used to fabricate acoustic metamaterials. These LC-resonance-based acoustic metamaterials usually have considerable absorption loss, fabrication complexity, and an anisotropic response associated with low symmetry, which limit their practical applications.

Recently, distinct from the metallic LC-resonance scheme, Mie resonances of dielectric nanoparticles have provided a novel mechanism for the creation of electric or magnetic resonance, and opened a simple but versatile route for the construction of electromagnetic metamaterials $^{27-37}$ . Adaptation of the concept to acoustics requires acoustically ultraslow fluid particles compared to the background air (resembling a high-index dielectric nanoparticle), which is not practical for audible airborne sound. In addition, this mechanism requires the encapsulation of the soft fluid particle in a dense fluid matrix with a long lifetime, highly stable shape and controllable regular arrangement, which remains a great technical challenge in terms of experimental realization $^{38,39}$ .

## Ultraslow-fluid-like unit cell with Mie resonance

In theory, a common feature of Mie-resonance particles is the high refractive index relative to the background medium. Although the requirement can be naturally fulfilled by a variety of ordinary dielectric materials in electromagnetics, very few natural materials possess a high index relative to the background fluid in acoustics, especially for airborne sound. For example, solid particles and liquid particles only support waves with sound speeds an order of magnitude faster than air, whereas gases in general have sound speeds comparable to air under normal temperature and pressure. Therefore, one must construct an ultraslow artificial structure with a much lower sound speed than that in air to overcome the low-refractive-index limitation in natural acoustic materials. This unique characteristic can enable the artificial spatial concentration of sound in Mie-resonant patterns and produce versatile metamaterials.

We propose a type of two-dimensional (2D) ultraslow-fluid-like unit cell which can generate Mie resonances with monopolar, dipolar and multipolar characteristics. A photograph of the proposed unit cell fabricated with epoxy resin by means of 3D printing is shown in Fig. 1a, and its schematic cross-sectional illustration in the xy-plane is depicted in Fig. 1b. The maze-like structure with a radius R is uniformly divided into eight sections, with each section having a zigzag channel $^{40-42}$ with identical width w=0.08R and a curling number N=8. Here N is defined as how many times the acoustic waves circulate in a zigzag channel. In each section, thin solid resin frames (blue) inserted into the background medium (air, shown in white) form the propagation channel of sound waves isolated from the other seven sections. As scalar waves, sound waves below the cutoff frequency propagate along these zigzag channels instead of in a straight line from the exterior to the central interconnection core of radius $r_{2}$ . This enables the propagation length of sound wave to be multiplied. Consequently, the structure possesses a high relative refractive index $n_{r}$ governed by the ratio between the total path length and the size of the unit cell. The ratio is proportional to the curling number N, and here equals approximately 4.2 for the chosen parameter of R=5 cm. The structure employs the curled channels to achieve a high refractive index, which is similar to the previous reports, even though there are some differences between the two structures $^{43}$ . It should be noted that the important thing here is to find the Mie resonances in the structure and further construct the artificial Mie resonators. The artificial Mie resonators can exhibit distinct monopolar/multipolar behaviour with respect to the previous reports and offer promising novel applications such as sound blocking and tunnelling.

![](images/d8b71cb2359a2f5691a803d20e6ea36198e16c110d1eeb14626f316a17864089.jpg)
Figure 1 | Geometry design and physical model of the unit cell for the metasurface. a, Photograph of a realistic unit cell fabricated with epoxy resin by means of 3D printing. The cover sheet is removed to view the microstructures. b, Schematic cross-sectional illustration of a unit cell with radius R=5 cm. The solid resin frames (blue) inserted into the background medium air (white) have a thickness t=0.02R and a curling number N=8. The zigzag channels with width w=0.08R are the propagation path of sound waves, which is equivalent to a type of artificial ultraslow medium with a low sound speed (relative to the background medium air). c, Equivalent physical model of a multichannel resonant unit cell that can support monopolar, dipolar and multipolar resonances, leading to novel acoustic properties for airborne sound.

A physical model of the structure in Fig. 1b is illustrated in Fig. 1c. The model is composed of an air interconnection region (white), coated by a layer of resin frame material (blue). Eight straight channels filled with the ultraslow medium with high refractive index (yellow) are embedded radially in the frame layer, and connected by the interconnection region. The background medium (white) is chosen to be air for application purposes. Collective in-phase propagation inside the eight channels can enhance the monopolar Mie resonance (for negative bulk modulus), whereas their relative out-of-phase propagation in various combinations can enable the dipolar Mie resonance (for negative mass density) and other multipolar resonances. It should be emphasized that this proposal is essentially different from the Helmholtz resonators, in which the radius of the central air core approaches R and acts as an acoustic capacitance. The monopole frequency is found to be $f_{R} \approx 520$ Hz (refs 44,45; see Methods and Supplementary Notes 1 and 2). Further investigation of the eigenstates presents a clear picture of the Mie resonances and their physical origin. The acoustic Mie-resonance modes of the ultraslow unit cell in Fig. 1b can be calculated with their detailed microstructures, as shown in Fig. 2; there are artificial monopole (518 Hz), dipole (1,080 Hz), quadrupole (1,188 Hz), octupole (1,320 Hz) and second monopole (1,549 Hz) acoustic modes.

Figure 2a shows that the monopole concentrates sound energy in the maze-like structure and radiates sound equally in all directions. Individual channels simply move back and forth about some equilibrium position in a collective in-phase pattern, and the far-field directivity pattern $P(r,\theta)$ is independent of the angle of propagation. We emphasize that this artificial monopole differs from the classic acoustic monopole of a small sphere whose radius alternately expands and contracts. Our fluid-like unit cell is fabricated by solid resin and cannot change its radius, whereas significantly more sound energy can be concentrated owing to its high $n_{r}$ . In addition, note that the normalized frequency $f_{R}R/c_{0}\sim0.075$ , meaning that this subwavelength structure is capable of controlling sound waves with much longer wavelengths. Figure 2b depicts the dipole mode, which cannot distribute equally in all directions. Instead its directivity pattern $P(r,\theta)$ has maxima along the $120^{\circ}$ and $300^{\circ}$ directions, and no sound radiation along the $210^{\circ}$ and $30^{\circ}$ directions (sound cancels in these two regions). Furthermore, the upper-left and lower-right channels have equal high strength but opposite phase (one group inward the other group outward). The wavefronts expanding to the upper-left and lower-right are $180^{\circ}$ out of phase with each other. Compared with classic local resonances using membranes and Helmholtz resonators, it should be emphasized that artificial Mie resonances are much more suitable for manipulating wave propagation owing to the significantly richer resonant modes and the simple geometrically based characteristics. These unique features not only allow the production of individual/simultaneous negative values of $\rho_{eff}$ and $\kappa_{eff}$ , but also large positive and near-zero values. Therefore, although we focus on metasurfaces for high reflection in this paper, the findings can enable flexible control of acoustic waves to realize versatile functional devices with superior performance, such as negative refraction/super-resolution imaging system, invisible cloaking, sound insulators/filters, acoustic tunnelling, enhanced acoustic sensing, sound energy harvesting, and so on. Moreover, the strategy can also be extended far beyond the scope of original acoustics, and versatile metamaterials for electromagnetic waves as well as thermal waves can be constructed using the same principles. Therefore, the investigation of artificial Mie resonance in acoustics has significance in a variety of applications in other fields.

Experiments were further conducted to demonstrate the monopole/dipole Mie resonances with the set-up schematically shown in Fig. 3a (see Methods). As predicted in Fig. 2a, the localized pressure in the interconnection core should increase significantly at the monopolar frequency owing to the high-intensity Mie resonance in the axially symmetrical pattern. This behaviour is clearly confirmed by the pressure magnification in the region of the central interconnection core, as shown in Fig. 3b. The maximum magnification occurs at 518 Hz with a pressure enhancement of 8.8 times in experiments. The inset shows the measured pressure field of the monopole/dipole Mie resonance, whose distribution patterns confirm the numerical results in Fig. 2a,b. These monopole/dipole modes are on a subwavelength scale and can thus induce effective dynamic bulk modulus/mass density values that are negative.

![](images/75b78d26203726770c8d7bef01cf7d82796a0410bd44eefdec619a9f13df4517.jpg)
Figure 2 | Field distributions of acoustic Mie-resonance modes in air supported by the unit cell. a–e, Acoustic modes of the artificial monopole, dipole, quadrupole, octupole and second monopole, respectively. The first and second rows depict the distributions of pressure and phase fields (blue/red for small/large values), respectively. In the third row, the arrows indicate the displacements in the physical model equivalent to the realistic unit cell. In the fourth row, far-field patterns reconfirm the field distributions. Excellent agreement is observed.

Taking loss into consideration (see Methods), the simulation results with loss = 0.0093 (black solid line) are consistent with the experimental results (red open circles) in Fig. 3b. We also show the pressure magnification with different loss values in the inset of Fig. 3b. It is observed that the loss value influences the magnification without introducing an obvious shift in the Mie resonance frequency. Note that, in practice, the loss term can be determined empirically by fitting the calculated response data along the edge of the experimental spectra $^{5}$ . Furthermore, to corroborate the above results, we perform full-wave numerical thermoacoustic simulations to characterize the acoustic response of a practical Mie resonator in a general viscous and thermally conduction fluid. The obtained pressure magnification is shown by the blue triangles in Fig. 3b, which indicate a good agreement with that predicted by simulations using loss = 0.0093 and the experiments. To further illustrate the effect of losses, we investigate the pressure-magnification spectra at the central interconnection core with respect to various N. It is observed that an efficient Mie resonance can still be achieved with a higher N-value, provided that a good compromise is reached between the magnification and dissipation loss (Supplementary Note 3).

## Effective medium description

The physical effects of these artificial Mie resonances can be understood from the point of view of an effective medium. Using the standard procedure $^{46}$ , the effective dynamic parameters $\kappa_{eff}$ and $\rho_{eff}$ normalized to background air are retrieved from the transmission and reflection coefficients for a monolayer of the unit cells. The relative $\kappa_{eff}$ and $\rho_{eff}$ simulated by finite element analysis are shown as the black curves in Fig. 4a and b, respectively. As the monopolar Mie resonance occurs, $\kappa_{eff}$ switches to negative from 532 to 685 Hz. Note that $\rho_{eff}$ retains a positive value in the negative $\kappa_{eff}$ region, thus high reflectance should be expected in this frequency range. A similar result is obtained for $\rho_{eff}$ , here associated with the dipolar Mie resonance. One can observe a negative $\rho_{eff}$ as the frequency increases above 820 Hz, as shown in Fig. 4b. Here, the first peak may correspond to the so-called antiresonant behaviour of $\rho_{eff}$ coupled with the resonant behaviour of $1/\kappa_{eff}$ , concurrently at monopolar Mie resonance. Such a resonance–antiresonance coupling has been discussed previously in acoustic and electromagnetic metamaterials $^{47-49}$ . The features of simulations using loss = 0.0093 accord with the experiments in spite of the limitations of the experimental technique, namely $\kappa_{\mathrm{eff}}(\rho_{\mathrm{eff}})$ is negative over the monopole (dipole) frequency band and is highly dispersive. The sharp corners and extremely high positive/negative values, which usually exist in ideal lossless simulations, should disappear in simulations incorporating loss. This situation will be discussed later in connection with the realization of high-reflectivity metasurfaces. In addition, we also obtain the parameters while twisting the sample to different angles, and find the negative behaviour of $\kappa_{eff}$ and $\rho_{eff}$ remains almost unchanged. Therefore, the proposed unit cells can exhibit near-isotropic negative values owing to their eight-fold symmetry, which is different from the anisotropic acoustic response based on LC-resonance metamaterials.

a
a
![](images/1b742eb7576fe53138b58b9f9f9061ac2ec59b8f8e2d87c6968103404b684b53.jpg)

b
![](images/ab3700899d375d47c7b0b2ab2f0d1364726a6c188201695395376a6985b6511a.jpg)
b
Figure 3 | Experimental demonstration of basic Mie resonance.

a, Schematic of the experimental set-up. The microphone is attached at the upper surface of the waveguide to map the pattern of pressure field.

b, Frequency dependence of pressure magnification in the region of central interconnection core. The red open circles (black solid lines) represent the experimental (simulation) results, and the blue open triangles show the thermoacoustic simulation results. The measured basic monopole mode at the peak of 518 Hz and the dipole mode are shown in the insets. The upper-right inset shows the pressure magnification with different losses.

## Ultra-sparse subwavelength metasurface reflector

Obtaining a high reflectivity for low-frequency airborne sound on a subwavelength scale is a great challenge in science and engineering $^{14}$ . Traditional means of sound reflection use acoustically harder materials with lower refractive indices relative to air (for example, glass, steel, water), which can be easily found in nature and fabricated. The general form of sound intensity transmittance $T_{I}$ for a reflector of thickness L can be expressed as

$$
T _ {I} = 1 / \left[ 1 + \frac {1}{4} \left(\frac {Z}{Z _ {\text { air }}}\right) ^ {2} \sin^ {2} k L \right]
$$

where k denotes the wavenumber in the reflector and Z and $Z_{air}$ represent the acoustic impedance of the reflector and air, respectively. Note that, for low-frequency sound, there is almost total transmission ( $T_{I} \rightarrow 1$ ) through a thin solid reflector ( $kL \ll 1$ ) in spite of its larger impedance ( $Z \gg Z_{air}$ ). Although there is a technique to obtain high reflectivity with a reflector of thickness $L = (2n - 1)\lambda/4$ , efficient shielding of low-frequency sound remains difficult to achieve—for example, the thickness of the solid reflector should be 16.5 cm for 518 Hz, which is heavy, costly and difficult to tune. Recently, acoustic metamaterials have been adopted to achieve high reflectivity. However, they generally result in acoustically thick multilayered structure comparable to the wavelength or subwavelength, but tightly packed in minor intervals, neither of which is optimal for low-frequency sound. Here we present an ultra-sparse acoustic metasurface with subwavelength thickness, comprising the proposed unit cells arranged in large intervals, that aims to totally reflect low-frequency airborne sound at a selected Mie-resonance frequency below 1,000 Hz.

![](images/4c9e16ed319acb19cfe39f72a55a77e04b9c67f7773f169f91916092ebc1338c.jpg)

![](images/3b5bfc0c78c318da2be032e54532dc8ca1baf983c96985103f0b59ee09155965.jpg)
Figure 4 | Effective acoustic parameters obtained from a single-layer metasurface. a, Effective bulk modulus, which has a negative value in the frequency range from 532 to 685 Hz (shaded background) owing to the monopole Mie resonance. b, Effective dynamic mass density, which has a negative value in the frequency range above 820 Hz (shaded background) owing to the dipole Mie resonance. Simulations using loss = 0.0093 accord with the experiments, which separately realized either a negative effective bulk modulus or a negative effective dynamic mass using just one structure. Note that $\kappa_{eff}$ and $\rho_{eff}$ here characterize the structure's dynamic response to the incident sound waves, distinct from the conventional static volume-average parameters.

We first construct a simple metasurface sample as a proof of principle. The inset of Fig. 5a illustrates the schematic of the metasurface, comprising of three unit cells along the y-direction with a period of $d(=10R=50\mathrm{~cm})$ . The metasurface is sandwiched between two parallel plates, and a continuous sinusoidal sound wave of frequency 537 Hz (monopole Mie resonance) radiates normally along the x-direction. The thickness of the metasurface is $2R=0.15\lambda$ and the volume filling ratio is just $\pi R^{2}/2Rd=15\%$ . For comparison, a contrast sample made of a rigid cylinder array in the same arrangement is also investigated, as shown in the inset of Fig. 5d. The simulated normalized sound intensity pattern for the metasurface sample (rigid cylinder array) in the waveguide is shown in Fig. 5a (Fig. 5d), and the simulated and experimental results in the region corresponding to the white solid box behind the sample are shown in Fig. 5b,c (Fig. 5e,f) for clear view. Comparing Fig. 5a–c with

![](images/c538a8785a15d4c03a59ab5f56744aceeffef916890e17df66be18299a71a907.jpg)

b
![](images/75d788a10b1db3d36d75b3360879aed5217960a8274235e1fdb58d196b019daf.jpg)

d
![](images/89ea048cd99227e8e8ba09616d668349f47634a8a8c171fe547d8b6d97e291c3.jpg)

![](images/96ea72fb516147377e491b0ce9962611b4f9312666773bd0f92ddc822a017455.jpg)
c

![](images/0a016ac4fd1af28219f9218799228d20f2abfa8ae2de88c549b591b7802a65af.jpg)

f
![](images/d78df3c488ca8d25d0ff6a54e8fa093828881dbbf0833e417ea866e6296e074e.jpg)
Figure 5 | Comparison between an ultra-sparse thin metasurface sample and a rigid cylinder array of the same size and arrangement. a, Normalized sound intensity pattern for the metasurface sample in the waveguide at the monopole resonant frequency of 537 Hz, which blueshifts a little from the unit cell's monopole frequency at 518 Hz owing to the coupling effect between adjacent cells. The inset depicts the arrangement of the unit cells with d/R=10; the incident sound wave radiates from left to right. The large amplitude of the standing wave in front of the metasurface due to reflection is clearly observed. b,c, Simulated and experimental results in the region corresponding to the white solid box shown in a (behind the metasurface). d-f, Field distributions with the rigid cylinder array for comparison. In this particular sparse case, an average sound intensity reflection efficiency of 91.7% (78.1%) is achieved in simulations (experiments) within the 10 cm × 10 cm region for the metasurface sample, whereas only 3.6% (19.6%) is obtained for the rigid cylinder array. Good agreement clearly demonstrates the high reflectivity of the metasurface.

Fig. 5d–f, two remarks should be made. First, a significantly larger amplitude of the standing wave in front of the metasurface is clearly observed (Fig. 5a), which is induced by the high reflectivity of the metasurface. Second, an average sound intensity reflection efficiency of 91.7% (78.1%) is achieved in simulations (experiments) within the 10 cm × 10 cm region for the metasurface sample, whereas only 3.6% (19.6%) is achieved for the rigid cylinder array. The good agreement demonstrates the high reflectivity of the ultrasparse metasurface at the monopole frequency.

The sound reflectivity of the proposed metasurface is further characterized by the insertion loss (IL), which is defined as the loss of sound power resulting from the insertion of a sample in the transmission channel. Figure 6a shows the simulated/measured IL as a function of frequency for the metasurface and rigid cylinder array in Fig. 5. Close to 537 Hz, the IL of metasurface reaches a maximum of 93.4% in simulations and 83.6% in experiments, whereas the IL of the rigid cylinder array preserves a low value of 9.3% in simulations and 17.2% in experiments. For the parameters of R=5 cm and N=8, the monopole Mie resonance for the single-unit cell would occur at $f_{R}=518$ Hz—thus, the effective resonance for the metasurface is blueshifted relative to the resonance of the single-unit cell. The blueshift originates from diffraction coupling between the adjacent unit cells. As a result of this unique coupling characteristic, a high transmission loss and broadband operation can be easily achieved simply by rearranging the Mie-resonant cells (Supplementary Note 4).

It should be noted that the ultra-sparse metasurface possesses several characteristics which are distinguished from other sound-blocking panels with orifices, such as the pioneering membrane-type metamaterials (MTMs; refs 50,51). Specifically, the velocity fields of Mie cells show a radial inflow distribution (see Fig. 6b) and give rise to a thin near-zero-velocity region in front of the Mie cells, causing the high reflection. This feature is distinct from the vortical distribution around the MTMs, in which the continuous field (through the orifice) and the reversed resonant field (by the MTMs) interfere destructively, cancelling the far-field transmission. The different resonance patterns also lead to a different sound blocking performance. The line shape of the transmission spectra is distinct from the Fano-like asymmetric dip–peak profile that is a characteristic of membrane resonance systems $^{51}$ . In addition, the orifice is on a subwavelength scale (orifice radius of 12 mm $\ll$ operating wavelength of 1 m), whereas the opening aperture (0.4 m) is of the same order as the operating wavelength (0.63 m) in the Mie metasurface.

The distinct nature of Mie-resonant cells in the proposed metasurface can be further understood by using effective parameters. In a conventional acoustic low-frequency resonant system such as a membrane, the vibrational motion of the membrane's first eigenmode shows a predominantly dipolar characteristic, and hence a negative $\rho_{eff}$ region in the frequency regime before the first eigenfrequency. The negative $\rho_{eff}$ means that the acceleration response of the system is opposite to the external driving force, giving rise to a Fano-like asymmetric dip–peak profile. The excitation of monopolar motion (compression and expansion which should result in a change in volume) is impossible in the low-frequency regime owing to the extremely small thickness of the membrane. In this case, we have clearly demonstrated an intense artificial monopolar Mie resonance not reported before and also confirmed a negative $\kappa_{eff}$ region. The negative $\kappa_{eff}$ means that the system can support a volume expansion following an external compression. Thus, neither constructive interference nor a transmission peak is found close to the dip, and the system is robust with respect to variation and artefacts (Supplementary Note 5). In addition, it should be noted that the ultraslow unit cells are also different from Helmholtz resonators.

a

![](images/99a8f4187cb80bc8e50fbe2c1b0f7e1b996ce45a7c6b47fb6f427d4598fdff87.jpg)

![](images/b535164feddde38dfbea1b408ae5434cfacdb5651d4d78ad715a7ed9f2b1fc14.jpg)
Figure 6 | Sound shielding performance and its operation. a, Insertion loss (IL) as a function of frequency. The circles (triangles) show the average experimental results of the metasurface (rigid cylinder array). Error bars correspond to the standard deviation (s.d.) of a collection of four scan sequences. The experimental results are found to be in agreement with the theoretical predictions, showing a significant reflection peak at 537 Hz. b, Pressure field distribution and velocity profile at the IL peak. A plane wave is incident from the left, represented by the large white arrows. The rainbow colour represents the amplitude of acoustic pressure and the black arrows show the direction and amplitude of the air velocity. The velocity shows a radial inflow pattern and gives rise to a thin near-zero-velocity region in front of the Mie cells (see Supplementary Movie 1 for dynamic view).

owing to their distinct intrinsic and inertial nature, as well as the eigenmodes supportable (Supplementary Note 6).

## Operation frequency tunability

Traditional acoustic metasurfaces achieve high reflection over a static frequency region. From the applications point of view, it is highly desirable to realize tunable operating frequency. As shown in the previous sections, such an artificial Mie resonance is strongly dependent on the geometrical microstructure. Hence, we can tune the resonant frequency within a broad band by adjusting the curling number N (Supplementary Fig. 3). Another scheme used to tune the operating frequency of the Mie-resonance metasurface is to alter the ambient temperature inside the unit cell. The strong dependence of $c_{air}$ on the ambient temperature T can be described by $c_{air}=20.047\sqrt{273.15+T}$ . Therefore, the IL peak corresponding to the monopole Mie resonance of N=8 redshifts gradually from 545 to 341 Hz as the temperature is decreased from 50 to $-150^{\circ}C$ . In addition, by combining the two schemes, flexible and wide-band tunability of the operating frequency is expected for the entire spectrum of low-frequency sound. A significantly low operating frequency can be achieved—for example, the IL peak appears around 180 Hz (thickness $2R\approx\lambda/20$ ) for N=22 and $T=-150^{\circ}C$ . Furthermore, although we have focused on the monopolar Mie-resonance mode, sound blocking can also be achieved by using the multipolar Mie-resonance modes (Supplementary Note 7).

## Conclusions

We have reported an ultraslow-fluid-like particle that employs an artificial high refractive index (relative to air, rare in nature) to achieve the intense Mie resonances for low-frequency airborne sound. Its subwavelength scale makes it feasible to construct acoustic metasurfaces based on Mie resonances with either negative bulk modulus or negative mass density, essentially different from the LC-resonance scheme. As a result of the single-negative nature, an ultra-sparse subwavelength metasurface with high reflectance was developed for low-frequency sound. We demonstrate a $0.15\lambda$ -thick, 15%-filling ratio metasurface monolayer with an insertion loss over 93.4%. The proposed ultraslow-fluid-like particle exhibits a flexibly tunable frequency, a highly stable shape and a controllably regular arrangement—factors that significantly extend our ability to control sound waves in air. Promising potential applications of devices based on these principles include sound shielding along highways and high-speed rail, acoustic Veselago lenses with subwavelength resolution, and sound tunnelling (Supplementary Note 8).

## Methods

Methods and any associated references are available in the online version of the paper.

Received 14 August 2014; accepted 22 July 2015; published online 31 August 2015

## References

1. Fok, L., Ambati, M. & Zhang, X. Acoustic metamaterials. MRS Bull. 33, 931–934 (2008).

2. Liu, Z. Y. et al. Locally resonant sonic materials. Science 289, 1734–1736 (2000).

3. Garcia-Chocano, V. M., Christensen, J. & Sanchez-Dehesa, J. Negative refraction and energy funneling by hyperbolic materials: An experimental demonstration in acoustics. Phys. Rev. Lett. 112, 144301 (2014).

4. Sanchis, L. et al. Three-dimensional axisymmetric cloak based on the cancellation of acoustic scattering from a sphere. Phys. Rev. Lett. 110, 124301 (2013).

5. Fang, N. et al. Ultrasonic metamaterials with negative modulus. Nature Mater. 5, 452–456 (2006).

6. Zhang, S., Xia, C. & Fang, N. Broadband acoustic cloak for ultrasound waves. Phys. Rev. Lett. 106, 024301 (2011).

7. Zhang, S., Yin, L. L. & Fang, N. Focusing ultrasound with an acoustic metamaterial network. Phys. Rev. Lett. 102, 194301 (2009).

8. Rupin, M., Lemoult, F., Lerosey, G. & Roux, P. Experimental demonstration of ordered and disordered multiresonant metamaterials for Lamb waves. Phys. Rev. Lett. 112, 234301 (2014).

9. Farhat, M., Guenneau, S. & Enoch, S. Ultrabroadband elastic cloaking in thin plates. Phys. Rev. Lett. 103, 024301 (2009).

10. Zhu, J. et al. A holey-structured metamaterial for acoustic deep-subwavelength imaging. Nature Phys. 7, 52–55 (2011).

11. Christensen, J. et al. Extraordinary absorption of sound in porous lamella-crystals. Sci. Rep. 4, 4674 (2014).

12. Liang, Z., Willatzen, M., Li, J. & Christensen, J. Tunable acoustic double negativity metamaterial. Sci. Rep. 2, 859 (2012).

13. Li, J. S., Fok, L., Yin, X. B., Bartal, G. & Zhang, X. Experimental demonstration of an acoustic magnifying hyperlens. Nature Mater. 8, 931–934 (2009).

14. Mei, J. et al. Dark acoustic metamaterials as super absorbers for low-frequency sound. Nature Commun. 3, 756 (2012).

15. Lee, S. H., Park, C. M., Seo, Y. M., Wang, Z. G. & Kim, C. K. Composite acoustic medium with simultaneously negative density and modulus. Phys. Rev. Lett. 104, 054301 (2010).

16. Park, C. M. et al. Amplification of acoustic evanescent waves using metamaterial slabs. Phys. Rev. Lett. 107, 194301 (2011).

17. Fleury, R. & Alu, A. Extraordinary sound transmission through density-near-zero ultranarrow channels. Phys. Rev. Lett. 111, 055501 (2013).

18. Fleury, R., Sounas, D. L., Sieck, C. F., Haberman, M. R. & Alu, A. Sound isolation and giant linear nonreciprocity in a compact acoustic circulator. Science 343, 516–519 (2014).

19. Farhat, M. et al. Platonic scattering cancellation for bending waves in a thin plate. Sci. Rep. 4, 4644 (2014).

20. Cheng, Y., Yang, F., Xu, J. Y. & Liu, X. J. A multilayer structured acoustic cloak with homogeneous isotropic materials. Appl. Phys. Lett. 92, 151913 (2008).

21. Cheng, Y., Xu, J. Y. & Liu, X. J. One-dimensional structured ultrasonic metamaterials with simultaneously negative dynamic density and modulus. Phys. Rev. B 77, 045134 (2008).

22. Cheng, Y., Zhou, C., Wei, Q., Wu, D. J. & Liu, X. J. Acoustic subwavelength imaging of subsurface objects with acoustic resonant metalens. Appl. Phys. Lett. 103, 224104 (2013).

23. Ma, G., Yang, M., Xiao, S., Yang, Z. & Sheng, P. Acoustic metasurface with hybrid resonances. Nature Mater. 13, 873–878 (2014).

24. Yang, M., Ma, G., Yang, Z. & Sheng, P. Coupled membranes with doubly negative mass density and bulk modulus. Phys. Rev. Lett. 110, 134301 (2013).

25. Yang, Z., Mei, J., Yang, M., Chan, N. H. & Sheng, P. Membrane-type acoustic metamaterial with negative dynamic mass. Phys. Rev. Lett. 101, 204301 (2008).

26. Lai, Y., Wu, Y., Sheng, P. & Zhang, Z. Q. Hybrid elastic solids. Nature Mater. 10, 620–624 (2011).

27. Zhao, Q., Zhou, J., Zhang, F. L. & Lippens, D. Mie resonance-based dielectric metamaterials. Mater. Today 12, 60–69 (December, 2009).

28. Brongersma, M. L., Cui, Y. & Fan, S. H. Light management for photovoltaics using high-index nanostructures. Nature Mater. 13, 451–460 (2014).

29. Shi, L., Tuzer, T. U., Fenollosa, R. & Meseguer, F. A new dielectric metamaterial building block with a strong magnetic response in the sub-1.5-micrometer region: Silicon colloid nanocavities. Adv. Mater. 24, 5934–5938 (2012).

30. Liu, X. M., Zhao, Q., Lan, C. W. & Zhou, J. Isotropic Mie resonance-based metamaterial perfect absorber. Appl. Phys. Lett. 103, 031910 (2013).

31. Moitra, P., Slovick, B. A., Yu, Z. G., Krishnamurthy, S. & Valentine, J. Experimental demonstration of a broadband all-dielectric metamaterial perfect reflector. Appl. Phys. Lett. 104, 171102 (2014).

32. Yahiaoui, R. et al. Towards left-handed metamaterials using single-size dielectric resonators: The case of $TiO_{2}$ -disks at millimeter wavelengths. Appl. Phys. Lett. 101, 042909 (2012).

33. Kallos, E., Chremmos, I. & Yannopapas, V. Resonance properties of optical all-dielectric metamaterials using two-dimensional multipole expansion. Phys. Rev. B 86, 245108 (2012).

34. Kang, L. & Lippens, D. Mie resonance based left-handed metamaterial in the visible frequency range. Phys. Rev. B 83, 195125 (2011).

35. Slovick, B., Yu, Z. G., Berding, M. & Krishnamurthy, S. Perfect dielectric-metamaterial reflector. Phys. Rev. B 88, 165116 (2013).

36. Zhang, F. L., Zhao, Q., Kang, L., Zhou, J. & Lippens, D. Experimental verification of isotropic and polarization properties of high permittivity-based metamaterial. Phys. Rev. B 80, 195119 (2009).

37. Ginn, J. C. et al. Realizing optical magnetism from dielectric metamaterials. Phys. Rev. Lett. 108, 097402 (2012).

38. Bretagne, A., Tourin, A. & Leroy, V. Enhanced and reduced transmission of acoustic waves with bubble meta-screens. Appl. Phys. Lett. 99, 221906 (2011).

39. Leroy, V. et al. Design and characterization of bubble phononic crystals. Appl. Phys. Lett. 95, 171904 (2009).

40. Liang, Z. X. & Li, J. S. Extreme acoustic metamaterial by coiling up space. Phys. Rev. Lett. 108, 114301 (2012).

41. Xie, Y. B., Popa, B. I., Zigoneanu, L. & Cummer, S. A. Measurement of a broadband negative index with space-coiling acoustic metamaterials. Phys. Rev. Lett. 110, 175501 (2013).

42. Liang, Z. X. et al. Space-coiling metamaterials with double negativity and conical dispersion. Sci. Rep. 3, 1614 (2013).

43. Xie, Y., Konneker, A., Popa, B.-I. & Cummer, S. A. Tapered labyrinthine acoustic metamaterials for broadband impedance matching. Appl. Phys. Lett. 103, 201906 (2013).

44. Bohren, C. F. & Huffman, D. R. Absorption and Scattering of Light by Small Particles (Wiley-Interscience, 1983).

45. Hu, X., Ho, K.-M., Chan, C. T. & Zi, J. Homogenization of acoustic metamaterials of Helmholtz resonators in fluid. Phys. Rev. B 77, 172301 (2008).

46. Zigoneanu, L., Popa, B. I., Starr, A. F. & Cummer, S. A. Design and measurements of a broadband two-dimensional acoustic metamaterial with anisotropic effective mass density. J. Appl. Phys. 109, 054906 (2011).

47. Ao, X. & Chan, C. T. Complex band structures and effective medium descriptions of periodic acoustic composite systems. Phys. Rev. B 80, 235118 (2009).

48. Zhang, F. L., Kang, L., Zhao, Q., Zhou, J. & Lippens, D. Magnetic and electric coupling effects of dielectric metamaterial. New J. Phys. 14, 033031 (2012).

49. Koschny, T. et al. Impact of inherent periodic structure on effective medium description of left-handed and related metamaterials. Phys. Rev. B 71, 245105 (2005).

50. Yang, Z., Dai, H. M., Chan, N. H., Ma, G. C. & Sheng, P. Acoustic metamaterial panels for sound attenuation in the 50–1000 Hz regime. Appl. Phys. Lett. 96, 041906 (2010).

51. Ma, G., Yang, M., Yang, Z. & Sheng, P. Low-frequency narrow-band acoustic filter with large orifice. Appl. Phys. Lett. 103, 011903 (2013).

## Acknowledgements

This work was supported by the National Basic Research Program of China (2012CB921504), NSFC (11274171 and 11474162), and SRFDP (20110091120040, 20120091110001 and 20130091130004).

## Author contributions

X.J.L. coordinated and supervised the project. Y.C. developed the device concept and design. Y.C., C.Z. and B.G.Y. constructed the theoretical simulations and experimental set-up. D.J.W. and Q.W. provided insight and interpretation of the Mie-resonance properties. Y.C. and X.J.L. analysed the data and wrote the manuscript.

## Additional information

Supplementary information is available in the online version of the paper. Reprints and permissions information is available online at www.nature.com/reprints.
Correspondence and requests for materials should be addressed to X.J.L.

## Competing financial interests

The authors declare no competing financial interests.

## Methods

Experiments. The unit cells of the metasurface were precision-fabricated using epoxy resin via 3D printing. Measurements of the basic Mie-resonance mode shown in Fig. 3b and the retrieved parameters shown in Fig. 4 were conducted in a modified impedance apparatus which complies with ASTM E1050-12 and ASTM E2611-09. Detailed set-up: A loudspeaker

(ENPILL PD-2121) was mounted on the input surface of the waveguide to generate an incident plane wave. The sample of the unit cell was placed at the centre of the waveguide, a sufficiently large distance away from the loudspeaker to ensure propagation of the plane wave for the entire frequency range measured. Cone-shaped sound absorbing foam was mounted on the output surface of the waveguide to minimize reflections from the waveguide to the open space. Incident burst signals were generated by an arbitrary/function generator (Stanford Research DS-345), enhanced by an audio power amplifier (CPA 2400), and fed to the sample. Local pressure fields were measured by inserting 1/4-inch condensed microphones (Brüel&Kjær type-4939) into the top side of the waveguide at designated positions. The outputs of the microphones were conditioned (Brüel&Kjær type-2690), acquired by a digitizer (NI PXI-5152), and processed by LabVIEW software. A frequency scan was performed. Highly reliable readings of pressure amplitude and phase at each frequency were ensured by multiple measurements, from which we obtained the complex transmission/reflection coefficient by means of the transfer matrix method. A similar set-up was used to map the pressure field pattern of the metasurface (Figs 5 and 6), in which the metasurface was placed in a 2D waveguide constructed using two parallel plates of Plexiglass.

Derivation of Mie-resonant frequency with rigorous scattering theory. Owing to the structural complexity of the physical model, the classical Mie solutions of the diffraction problem which characterize wave scattering by small (relative to the incident wavelength) homogeneous cylindrical particles are not applicable here. Under a long-wavelength limit, we effectively match the rigid resin frame with the deep-subwavelength slits $w < 2\pi R / 8 / 4$ in a period $2\pi R / 8 < \lambda / 4$ to a uniform shell with a thickness $(R - r_2)\eta$ . Here $\eta = 8w / 2\pi R$ is the filling ratio of the narrow channels. Setting $\mathbf{r} = (x,y) = (r,\theta)$ the pressure field $P$ can be decomposed into incident and scattering cylindrical waves represented by Bessel $(J_m)$ and Hankel $(H_m)$ functions, respectively: $P(r) = \sum_{m} G_m J_m (k_0 r) e^{im\theta}$ when $r < r_2$ , $P(r) = \sum_{m} [E_m J_m (k'r) + F_m H_m (k'r)] e^{im\theta}$ when $r_2 < r < r'$ , $P(r) = \sum_{m} [c_m J_m (k_0 r) + D_m H_m (k_0 r)] e^{im\theta}$ when $r > R$ . Here $r' = r_2 + (R - r_2)\eta$ is the radius of the equivalent uniformly coated shell; $D_m \sim G_m$ are the mth-order expansion coefficients for each layer; $k_0$ and $k'$ denote the wavenumbers in air and the equivalent uniform shell, respectively. The virtual layer $r_2 + (R - r_2)\eta < r < R$ , which has the same fields at the two interfaces, is introduced to leave the total radius of the unit cell unchanged. Combining the continuities of $P(r)$ and

$1/\rho \cdot \partial P/\partial \mathbf{r}$ at the interfaces $r = r_{2}$ and $r = r'$ , the coefficients are obtained after solving:

$$
\left[ \begin{array}{c c c c} J _ {m} (k _ {0} r _ {2}) & - J _ {m} (k ^ {\prime} r _ {2}) & - H _ {m} (k ^ {\prime} r _ {2}) & 0 \\ \frac {k _ {0}}{\rho_ {0}} J _ {m} ^ {\prime} (k _ {0} r _ {2}) & - \frac {k ^ {\prime}}{\rho^ {\prime}} J _ {m} ^ {\prime} (k ^ {\prime} r _ {2}) & - \frac {k ^ {\prime}}{\rho^ {\prime}} H _ {m} ^ {\prime} (k ^ {\prime} r _ {2}) & 0 \\ 0 & - J _ {m} (k ^ {\prime} r ^ {\prime}) & - H _ {m} (k ^ {\prime} r ^ {\prime}) & H _ {m} (k _ {0} R) \\ 0 & - \frac {k ^ {\prime}}{\rho^ {\prime}} J _ {m} ^ {\prime} (k ^ {\prime} r ^ {\prime}) & - \frac {k ^ {\prime}}{\rho^ {\prime}} H _ {m} ^ {\prime} (k ^ {\prime} r ^ {\prime}) & \frac {k _ {0}}{\rho_ {0}} H _ {m} ^ {\prime} (k _ {0} R) \end{array} \right]
$$

$$
\times \left[ \begin{array}{c} G _ {m} \\ E _ {m} \\ F _ {m} \\ D _ {m} \end{array} \right] + \left[ \begin{array}{c} 0 \\ 0 \\ J _ {m} (k _ {0} R) \\ \frac {k _ {0}}{\rho_ {0}} J _ {m} ^ {\prime} (k _ {0} R) \end{array} \right] = 0
$$

If we consider only the scattering of the zeroth-order cylindrical wave and use approximations with m=0 under the low frequency $k_{0}R$ and $k_{0}r_{2}\ll1$ , the monopole frequency can be obtained as $f_{R}\approx520$ Hz.

Simulations. Numerical simulations were implemented by using COMSOL Multiphysics v4.3, a finite-element analysis and solver software. The simulations are performed in the Pressure Acoustic module including the detailed microstructures with actual geometric dimensions and loss factors. The materials applied are air and epoxy resin. The standard parameters used for air under an ambient pressure of 1 atm at 20 °C are mass density $\rho_{air} = 1.29 \, kg m^{-3}$ and sound speed $c_{air} = 343 \, m s^{-1}$ . The mass density, Young's modulus and Poisson's ratio for epoxy resin are 1,050 kg m $^{-3}$ , 5.08 GPa and 0.35, respectively. The largest mesh element size was lower than 1/10th of the shortest incident wavelength, and the further refined meshes were applied in the domain of the unit cells of the microstructure. Radiation boundary conditions were imposed on the input and output planes of the air domain to eliminate interference from the reflected waves.

Characterizing the loss effect. When acoustic waves propagate through channels with small dimensions, the viscous and thermal losses (induced by near-wall viscosity and thermal conduction effect) should be significant. To include the influence of thermoviscous losses on the material performance, we add loss explicitly into the wavenumber of internal fluid (air) as $k_{0}=2\pi f/c_{air}-i\alpha$ , where $\alpha$ is the attenuation coefficient. The attenuation coefficient can be set as $\alpha=2\pi f\times loss/c_{air}$ , where loss is independent of frequency. The simulation results are further confirmed by thermoacoustic analysis which takes viscous and thermal conduction in the fluid into consideration.
