PT-Symmetric Acoustics

Xuefeng Zhu, $^{1}$ Hamidreza Ramezani, $^{1}$ Chengzhi Shi, $^{1}$ Jie Zhu, $^{1}$ and Xiang Zhang $^{1,2,*}$

$^{1}$ NSF Nanoscale Science and Engineering Center, University of California,

Berkeley, California 94720, USA

$^{2}$ Materials Sciences Division, Lawrence Berkeley National Laboratory, Berkeley, California 94720, USA (Received 9 May 2014; revised manuscript received 3 July 2014; published 5 September 2014)

We introduce here the concept of acoustic parity-time (PT) symmetry and demonstrate the extraordinary scattering characteristics of the acoustic PT medium. On the basis of exact calculations, we show how an acoustic PT-symmetric medium can become unidirectionally transparent at given frequencies. Combining such a PT-symmetric medium with transformation acoustics, we design two-dimensional symmetric acoustic cloaks that are unidirectionally invisible in a prescribed direction. Our results open new possibilities for designing functional acoustic devices with directional responses.

DOI: 10.1103/PhysRevX.4.031042

Subject Areas: Acoustics

## I. INTRODUCTION

Controlling acoustic and phononic transport at will is a long-sought-after goal in modern physics and applications. Recent advancements in man-made materials (“metamaterials”) have resulted in intriguing achievements in acoustic and phononic transport manipulation $[1]$ . These discoveries include dynamic negative density and a bulk modulus $[2–8]$ , subwavelength imaging $[9–11]$ , acoustic and surface wave cloaking $[12–15]$ , a phononic band gap $[16,17]$ , extraordinary acoustic transmission $[18,19]$ , Anderson localization $[20]$ , and asymmetric transmission $[21–25]$ . These works, so far, are based on the modulation of the real part of the acoustic parameters. However, exploration of acoustic characteristics in a complex domain—as we shall discuss in more detail in the following—offers a new door to non-Hermitian acoustics with rich physics and entirely new functionalities of metamaterials. An asymmetric acoustic response is among them, where ingenious acoustic devices can be designed to distinguish different directions and react in a judicious way, depending on the direction of wave propagation. Such acoustic devices may have a range of useful applications, such as directional noise cancellation, ultrasonic medical imaging, architectural acoustics, acoustic amplification, and thermal insulating materials that can regulate heat flow.

It has been shown that systems with non-Hermitian potentials that violate time-reversal (T) symmetry but retain the combined parity-time (PT) symmetry might have real spectra, proposing a possible generalization of quantum mechanics [26]. Although the concept of PT-symmetric quantum mechanics, as a fundamental theory, is still under a heated debate [27], it has been explored in optics and electronics by means of interleaving balanced gain-and-loss regions. Several interesting physical features have been explored, such as power oscillations [28], unidirectional invisibility [29–32], the reconfigurable Talbot effect [33], and coherent perfect laser absorbers [34,35]. Moreover, in the nonlinear domain, PT symmetry has been used to realize potential optical isolators and circulators [36–40].

We introduce here the concept of PT symmetry in acoustics by judicious designs of acoustic materials with complex parameters featuring carefully balanced loss-and-gain regions. Based on exact analytical expressions, we first show that our design displays unique scattering characteristics such as asymmetric reflection, resulting in the medium being unidirectionally transparent. We further introduce transformation acoustics into PT-symmetric synthetic matter and demonstrate a PT-symmetric cloak that protects its inner information from being detected only along one side and not another—a one-way cloak.

## II. SCATTERING PROPERTIES

In acoustics, we can design a PT-symmetric medium with a complex bulk modulus obeying the condition $\kappa(\mathbf{r}) = \kappa^{*}(-\mathbf{r})$ and the mass density $\rho(\mathbf{r}) = \rho(-\mathbf{r})$ . The designed one-dimensional (1D) PT-symmetric medium is schematically shown in the main panel of Fig. 1. In this system, the acoustic waves $Pe^{i\omega t}$ are propagating in the z direction, which can be described by the Helmholtz equation $(d^{2}/dz^{2})P + \omega^{2}\rho\kappa^{-1}P = 0$ , with P the pressure, $\omega = 2\pi f$ (f is the frequency) the so-called angular frequency, $\rho$ the mass density, and $\kappa$ the complex bulk modulus. It needs to be mentioned that, generally, the modulus of acoustic material should have a positive imaginary part, indicating that the material is lossy with inherent damping.

![](images/19d9b03e7914c6a523984732c3d7f0d36c9187add9be2ab836b1a0d5bb8463ac.jpg)
FIG. 1. Schematic of the acoustic PT-symmetric medium. The scattering domain is composed of 11 sections: three gain regions (red blocks), three loss regions (green blocks), and five passive regions in between. The length of the active region is 0.148 m, while the length of each passive region is 0.015 m. The transmission and reflection of acoustic waves incident from the left (blue curve) and right (red curve) are shown. This PT-symmetric medium is designed to be reflectionless for acoustic waves incident from the left. Insets: The normalized amplitude of the pressure field for the left (left inset) and right (right inset) incidences at the exceptional point.

The acoustic gain material with a negative imaginary part of the modulus has not yet been observed in nature, which, however, can be effectively realized by delicate feedback systems using the active sound-controlling apparatus [24]. Since the setup respects PT symmetry, the mass density and the real part of the bulk modulus are an even function of position z=0, while the imaginary part of the bulk modulus is an odd one. In our case, the material parameters are $\rho_{p}=1.2\ kg/m^{3}$ and $\kappa_{p}=1.42\times10^{5}\ Pa$ for passive regions and are $\rho_{l(g)}=1.49\ kg/m^{3}$ and $\kappa_{l(g)}=1.75\times10^{5}\pm1.72\times10^{4}i\ Pa$ for loss (gain) regions, respectively. As depicted in Fig. 1, the scattering region ( $z<|L|=0.4815\ m$ ) is composed of three loss-and-gain sections (green and red blocks) and five passive sections in between. The length of loss-or-gain sections is 0.148 m, and the length of each passive section is 0.015 m.

The underlying symmetries of the acoustic system, viz. PT symmetry, impose a generalized conservation relation on the acoustic wave-scattering properties of the composite structure. In particular, outside the acoustic PT-symmetric setup $(z > |L|)$ , we can decompose the pressure field into the right- and left-traveling waves $P(z > |L|) = P_{f}^{-(+)} e^{-ikz} + P_{b}^{-(+)} e^{ikz}$ , where $- (+)$ is for the left $z < -L$ (right $z > L$ ), k is the wave number, and $P_{f(b)}$ is the amplitude of the forward- (backward-) traveling acoustic waves. Furthermore, the scattering matrix $S(k)$ describing the relation between the incoming and outgoing acoustic waves in a 1D two-port system can be expressed as

$$
\binom{P _ {f} ^ {+}}{P _ {b} ^ {-}} = S (k) \binom{P _ {f} ^ {-}}{P _ {b} ^ {+}}, \qquad S (k) = \left( \begin{array}{c c} t & r _ {R} \\ r _ {L} & t \end{array} \right),\tag{1}
$$

where $r_{L(R)}$ and t are left- (right-) reflection and transmission coefficients, respectively. The PT-symmetric nature of the acoustic system leads to

$$
\binom{P _ {f} ^ {- *}}{P _ {b} ^ {+ *}} = S (k) \binom{P _ {f} ^ {+ *}}{P _ {b} ^ {- *}}.\tag{2}
$$

A comparison between Eqs. (1) and (2) shows that $S^{*}(k) = S^{-1}(k)$ . From this relation, we can conclude that [41,43]

$$
r _ {L} r _ {R} ^ {*} = 1 - | t | ^ {2}.\tag{3}
$$

Equation (3) or its other form $\sqrt{R_{L}R_{R}}=|T-1|$ , with $R_{L(R)}\equiv|r_{L(R)}|^{2}$ and $T\equiv|t|^{2}$ , is a generalization of the more familiar energy-conservation relation $T+R=1$ associated with the lossless passive systems, where the geometric mean of the two reflections $\sqrt{R_{L}R_{R}}$ replaces the reflection R from one side, noting that the reflection in a 1D lossless passive acoustic medium is the same for opposite directions. In 1D acoustic PT systems, the left reflection $R_{L}$ can, in principle, be different from the right reflection $R_{R}$ . An interesting result occurs when we have perfect transmission, that is, T=1. In this case, the product of two reflections is required to vanish, according to Eq. (3). For example, this constraint can be satisfied by one of the reflections getting to 0 while the other remains nonzero, which is of our interest and recalls the unidirectional transparency, since we have reflectionless perfect transmission in one direction but not the other. Based on the transfer-matrix method in acoustics, we have, respectively, calculated the phase [Fig. 2(a)] and amplitude [Fig. 2(b)] of left- and right-reflected waves and transmitted waves propagating through the well-designed acoustic PT-symmetric system. From Eq. (3), for the case T<1, the phases of the left reflection and right reflection should be equivalent, while for T>1, there is $\pi$ phase difference between the two reflections. Moreover, for both cases, there is a $\pm\pi/2$ phase difference between the transmission and reflection parts. Figure 2(a) shows that our results are in good agreement with these theoretical predictions. The results in Fig. 2(b) show that at the frequency $f\approx6$ kHz, the left reflection is exactly 0 and the right reflection is around $R_{R}\approx0.39$ , as marked by darkened dots. At that frequency, the phase of left-reflected waves experiences an abrupt change of $\pi$ , which is related to the singularity of zero left-side reflection. Because of the step-function behavior of the phase, the delay time $\tau_{r} \equiv (d\phi_{r}/d\omega)$ for the left reflection behaves as a delta function. As a result, the reflected acoustic waves will be trapped for a long time in the waveguide and get absorbed completely by the loss. At that frequency, the transmission is exactly unitary but the phase of transmitted waves (approximately $-0.3\pi$ ) differs from the phase (approximately $-0.33\pi$ ) of the acoustic waves after traversing a homogeneous background medium with the same length of the scattering system. Therefore, the acoustic PT-symmetric setup cannot be regarded as invisible in any case, even though the system is reflectionless for the waves propagating from the left. We also notice that a well-designed lossy medium can have unidirectional reflection when operating at the exceptional points (EP) of this non-Hermitian system [31]. However, unidirectional transparency can only be realized in the PT system with gain included, since the transmission must be unitary and the unidirectionally scattered light is produced by the gain part.

![](images/da5b88176b8f5cbb5312bc9756049f42e4939e9fd4dceca167a91d4603c7e2fb.jpg)

![](images/49ecf28d8de05514671574cc53fa4b7b0a385ed05557056195ab9b7f06d25817.jpg)

![](images/994a0a6faeaa7f634dab0f1f3413ecd46599de1906c5fb3b16f97a0632803257.jpg)

![](images/86b1c01bbb41c4755c510a4599814361176707dc39bfd22af4e6d5b47a56ca3e.jpg)
FIG. 2. Phases and amplitude of left-reflected, right-reflected, and transmitted waves of the acoustic PT-symmetric medium are shown in (a) and (b), respectively. The absolute value of the eigenvalues and the second component of the eigenvectors of the corresponding scattering matrix for this acoustic PT-symmetric medium are shown in (c) and (d). Here, we have normalized the eigenvectors such that the first component is always unitary.

In our acoustic PT-symmetric setup, the change of the transmission from T < 1 to T > 1 can be related to the transition from the PT-symmetric phase to the broken phase, according to Refs. [41,43]. In this respect, we shall demonstrate the frequency at which the transmission becomes unitary and is related to an exceptional point by investigating the spectral properties of the scattering matrix. When the acoustic system is in the PT-symmetric phase, the eigenvalues of the scattering matrix are nondegenerate and unimodular. In other words, unimodular eigenvalues correspond to the PT-symmetric phase, whereas nonunimodular eigenvalues, whenever they appear, signify the PT-broken phase. In the acoustic system, we can derive explicit criteria for the spontaneous PT symmetry-breaking transition in terms of the transmission and reflection coefficients in the scattering matrix. From Eq. (1), the eigenvalues of the scattering matrix are expressed as $\lambda_{1,2}=t\pm\sqrt{r_{L}r_{R}}$ while the eigenvectors are given by $(1\pm\sqrt{(r_{L}/r_{R})})^{T}$ . Using Eq. (3), we have $\lambda_{1,2}=t[1\pm i\sqrt{(1-T)/T}]$ . From this relation, we can conclude that when T<1, eigenvalues are unimodular and nondegenerate, and the acoustic system is in the so-called symmetric phase. However, when T>1, the eigenvalues are nonunimodular and the system is in the broken phase. At the exceptional point, the transition between these two phases happens and we have T=1. In Fig. 2(c), we plot the absolute values of the eigenvalues of a scattering matrix, from which the exceptional point can be observed at $f\approx6$ kHz, as marked by a darkened dot. Below the exceptional point, the eigenvalues are nonunimodular and the system is in the broken phase. While above the exceptional point, the eigenvalues are unimodular and the system is in the PT-symmetric phase. At the exceptional point, the system is unidirectionally reflectionless [see the inset of Fig. 2(b)]. Moreover, in Fig. 2(d), we plot the second components of the eigenvectors of the scattering matrix $\sqrt{(r_{L}/r_{R})}$ . The blue curve corresponds to the PT-symmetric phase, where the eigenvectors are real, and clearly, they are invariant under conjugation [43]. In the broken phase (red curve), the second components of the eigenvectors are purely imaginary, and under conjugation, one of the components transforms into the other. It should be pointed out that the unusual scattering properties of the PT-symmetric medium have also been demonstrated in optics and electronics, both in theories and experiments [41,42,44,45].

We also perform full-wave simulations using a finite-element solver (COMSOL Multiphysics) to verify the unidirectional reflectionless effect of the well-designed 1D PT-symmetric medium, as shown in the inset of Fig. 1. In the numerical simulations, the plane wave is incoming from the left and right sides, respectively, with the frequency located at the exceptional point. The interference pattern due to strong Bragg reflection is visualized for the right incidence, whereas the scattered acoustic waves are barely observed for the left incidence. It is interesting to point out that the pressure field for the left incidence is equally distributed in the loss-and-gain parts with the field distribution symmetric to the center of the scattering region, indicating that the energy produced by the acoustic gain is completely absorbed by the loss in the mirrored position. However, when the waves are incoming from the right, the pressure field is more localized in the gain parts and the extra energy from the gain after being balanced out by the loss provides the strong Bragg reflection.

Previously, asymmetric acoustic transport has also been explored using nonlinear material $[22]$ . Other structures of one-way acoustic devices, such as asymmetric nonlinear bead chains, have been demonstrated thereafter $[23]$ . By introducing nonlinear active electric circuits, with incoherent amplification, into the resonant unit cells in acoustic metamaterials, one can realize a nonreciprocal acoustic metamaterial with a large contrast ratio $[24]$ . It is also noted that the acoustic one-way propagation in linear time-varying media is exploited $[25]$ . These works, however, are mainly based on the modulation of the real part of the material properties and inevitably accompanied with a frequency shift. Our proposed acoustic PT approach offers a new paradigm in acoustic wave manipulations, including one-way transport. Different from the above nonlinear approaches, where harmonics have to be involved, one can exploit a PT-acoustic metamaterial with nonlinearity to achieve nonreciprocal wave propagation using nonlinear resonances. For example, when nonlinearity is incorporated into the PT system, if the acoustic wave comes from the loss side, the wave will be damped and the system responds in an “almost” linear way. However, when wave incidents come from the gain side, the nonlinearity kicks in and bends the “backbone” of resonance, resulting in strong asymmetric wave propagation and nonreciprocity at the same frequency without harmonic generation. Such asymmetric transport has already been demonstrated in electronics and optics $[36–40]$ ; however, acoustics remains an open area to be explored.

## III. PT-SYMMETRIC ACOUSTIC CLOAK

Since the acoustic wave equation is invariant under coordinate transformation [15,46–49], it is possible to combine the PT-symmetric medium with transformation acoustics to design myriad transformation acoustics devices of unidirectional responses. As an example, we will show how to design a 2D PT-symmetric acoustic cloak that is one-way invisible. First, we need to construct a PT-symmetric periodic structure with unidirectional invisibility in the virtual space $(r,\theta)$ , for which the complex modulation of material parameters should take the unique form of $\delta e^{i\beta\cdot r}$ rather than the generalized one exemplified in Fig. 1, where $\delta$ and $2\pi/|\beta|$ are the amplitude and periodicity of modulation, respectively. In our case, the density and complex bulk modulus of the unidirectionally invisible PT-symmetric periodic structure are $\rho_{0}=1.2~kg/m^{3}$ and $\kappa_{0}=1.42\times10^{5}\{1+0.1\exp[i219.7r\cos(\theta)]\}$ Pa $(r<b)$ , respectively, in the virtual space $(r,\theta)$ . In cylindrical coordinates, a mapping between virtual space $(r,\theta)$ and real space $(r',\theta')$ to produce a 2D acoustic cloak of cylindrical geometry can be expressed as $r=f(r')$ and $\theta'=\theta$ , with $f(r')=b(r'-a)/(b-a)$ for $a\leq r'\leq b$ . Here, a and b are the inner and outer radii of the acoustic cloaking shell, which are 0.05 and 0.1 m in our numerical simulations. With the transformation acoustic equations in orthogonal coordinates, we have the material parameters [50]

$$
\begin{array}{l} \rho_ {r} ^ {\prime} = \rho_ {0} \frac {r ^ {\prime}}{r ^ {\prime} - a}, \\ \rho_ {\theta} ^ {\prime} = \rho_ {0} \frac {r ^ {\prime} - a}{r ^ {\prime}}, \\ \kappa^ {\prime} = \left(\frac {b - a}{b}\right) ^ {2} \frac {r ^ {\prime}}{r ^ {\prime} - a} \kappa_ {0} [ f (r ^ {\prime}, \theta^ {\prime}) ] \end{array}\tag{4}
$$

for the PT-symmetric acoustic cloak in real space $(r', \theta')$ . As a matter of fact, the specific modulation in the form of $\delta e^{i\beta \cdot r}$ can be regarded as a complex grating that can provide a one-way wave vector $\beta$ . The interaction between incident plane waves of wave vector $k_{1}$ and the complex grating will produce a diffraction mode with the spatial frequency being $k_{1} + \beta$ . The mode transition between incident plane waves and the diffraction mode can take place only when the phase-matching condition is approximately satisfied, viz. $\delta = k_{1} + \beta - k_{2} \approx 0$ , where $k_{2}$ is the spatial frequency of the excited diffraction mode propagating in the surrounding medium, and therefore, $|k_{1}| = |k_{2}|$ . On the other hand, if

![](images/4daf3047fd4ca139a36ad6d94563b0054d91b4dc85594be343e899db68b199f9.jpg)
FIG. 3. The pressure-field maps in (a),(b) the virtual space and (c),(d) the physical space for a one-way invisible PT-symmetric cloak. In (a) and (c), where the plane wave is incoming from the left, a strong backscattered reflection can be observed due to the satisfied phase matching, and therefore, the observers on the left can sense the existence of the cloak. However, in (b) and (d), where the plane wave is incident from the right, the waves propagate without any scattering due to phase mismatching, and thus, an arbitrary object is perfectly concealed in the white region. The wavelengths of incident waves are $\lambda = 0.0572$ m.

$\delta \neq 0$ , the energy transferred to the diffraction mode is negligible due to the momentum mismatch. The proposed one-way invisibility of the PT-symmetric cloak is validated in Fig. 3. For left incidence in virtual space, strong Bragg reflection is observed in terms of the satisfied phase-matching condition $k_{2} \approx k_{1} + \beta$ , as shown by the inset vector diagram in Fig. 3(a). However, for right incidence in Fig. 3(b), scattered waves can hardly be visualized due to phase mismatch and the waves are propagating through the PT-symmetric medium, as if there is a bulk of surrounding medium. From the vector diagram in Fig. 3(b), a diffraction mode with the spatial frequency $k_{1} + \beta$ will be much larger than $k_{2}$ and thus can hardly be excited since it falls into the evanescent regime. Figures 3(c) and 3(d) are the corresponding simulations of the PT-symmetric acoustic cloak in real space after coordinate transformation, where the pressure-field distributions in both incidences agree well with the results in virtual space. It is clearly shown that an arbitrary object positioned in a rigid cylinder is perfectly concealed and one-way cloaked by the transformed PT-symmetric medium. It is worth mentioning that for left incidence in Fig. 3(c), the concealed rigid cylinder still remains invisible to the observer on the right, which is due to the reason that the designed phase matching is only satisfied for the backward reflection. It is also possible to make the phase-matching condition satisfied for the forward reflection, which requires engineering of the direction and periodicity of the complex modulation of material parameters. However, the reflection parts can never be in the same direction of incident waves under the constraint of Lorentz reciprocity in this linear and static system. The complex modulation of material parameters can also be designed along the radial direction, viz. $e^{i\beta\cdot r}$ , where the anisotropic transformed PT-symmetric medium can be realized by alternatively stacking concentric loss-and-gain layers with isotropic material parameters, as indicated by the reducible cloak designing in Ref. [48].

For the generalized $\mathcal{PT}$ -symmetric structure exemplified in Fig. 1, it is shown that the unidirectionality is very sensitive to frequency. However, for $\mathcal{PT}$ -symmetric potentials taking the unique form of $\delta e^{i\beta \cdot r}$ , it has a broadband unidirectional response from 5 to $8\mathrm{kHz}$ , as demonstrated in Fig. 4(a), since the scattering cross section is close to 0 due to the phase mismatch for right-side-incoming waves of different frequencies, whereas the scattered waves will be visualized for the nonignorable scattering cross section in a relatively broad frequency band when the phase-matching condition is approximately satisfied for left-side incidence [51]. It also should be mentioned that for the $\mathcal{PT}$ -cloaking device, the singularity at the inner boundary of the cloak may limit the operating bandwidth and narrows this band of functionality in practice. Moreover, Fig. 4(b) depicts the scattering cross section versus the angle of incidence at the frequency of 6 kHz, where the unidirectional invisibility also holds for an oblique angle of $|\theta| < \pi/6$ .

![](images/be65772d15a20fc6dd89ff600bfc835c3b78be2e8d0e6bf62454a72c605348ef.jpg)

![](images/5f27ff5428dee1941f17dd03860769a0a9630879f3efa2125f2fdaf1cf354ea3.jpg)
FIG. 4. (a) Scattering cross section versus frequency, where the blue (upper) and red (lower) curves describe the scattering cross section of the $\mathcal{PT}$ -symmetric cloak when the light is incoming from the left and right, respectively. (b) Scattering cross section versus the incident angle at the frequency $6\mathrm{kHz}$ , where $\theta = 0$ corresponds to the parallel propagation along the $z$ direction from left to right.

![](images/db36b1c47f0c2a287c218b1cb22c2a039be99477370ffeb9a0c5530d8f974a48.jpg)
FIG. 5. The pressure-field maps in (a),(b) the virtual space and (c),(d) the physical space for a one-way invisible PT-symmetric acoustic ground cloak. In (a) and (c), where the plane wave is incoming from the left, a strong backreflection can be observed due to the satisfied phase matching, and therefore, the observers on the left can sense the existence of the ground cloak. However, in (b) and (d), where the plane wave is incident from the right, the waves propagate without any scattering due to phase mismatching, and thus, an arbitrary object can be concealed in the bump region. The wavelengths of incident waves are $\lambda = 0.8$ m.

A one-way invisible 2D PT-symmetric acoustic ground cloak [52] is also designed and demonstrated in the full-wave simulation. In virtual space, we use $\rho_{0}=1.2\ kg/m^{3}$ and $\kappa_{0}=1.42\times10^{5}\{1+0.03\exp[i2\pi z/(0.4\sqrt{2})]\}$ Pa (1.4<x<11.4 and $|z|<10$ ) for the PT-symmetric medium. The material parameters of the transformed PT-symmetric medium in real space are obtained by the coordinate transformation used in Ref. [53]. This acoustic ground cloak can be clearly observed for left-side incidence because of the strong Bragg reflection in the PT-symmetric medium [Figs. 5(a) and 5(c)], whereas the cloaked region is invisible for right-side incidence due to phase mismatching [Figs. 5(b) and 5(d)].

Currently, considerable experimental difficulties still get involved in the fields of PT-symmetric acoustics, such as the design and realization of acoustic gain by feedback systems using the active sound-controlling apparatus $[24]$ . Compared to the conventional cloaking proposals, the simultaneous transformations of both the real and imaginary parts of the effective bulk modulus in this work may require much more sophisticated metamaterial design and engineering.

## IV. CONCLUSIONS

We have introduced the concept of PT symmetry acoustics and demonstrated the phenomenon of unidirectional transparency at the exceptional points of non-Hermitian acoustic systems. We have then transformed such a unidirectionally transparent acoustic medium into an acoustic cloak (PT-symmetric acoustic cloak) and demonstrated that it makes an object completely invisible in specified directions. We envision that PT acoustics will open a new route for designing functional acoustic systems with nonreciprocal responses and further enable us to explore fundamental PT physics using acoustic nonlinearities or time-varying parameters.

## ACKNOWLEDGMENTS

This research was supported by the Office of Naval Research (ONR) MURI Program under Grant No. N00014-13-1-0631. The authors thank Dr. Kosmas Tsakmakidis for proofreading and revision of the manuscript.

[1] L. Fok, M. Ambati, and X. Zhang, Acoustic Metamaterials, MRS Bull. 33, 931 (2008).

[2] Z. Y. Liu, X. X. Zhang, Y. W. Mao, Y. Y. Zhu, Z. Y. Yang, C. T. Chan, and P. Sheng, Locally Resonant Sonic Materials, Science 289, 1734 (2000).

[3] N. Fang, D. J. Xi, J. Y. Xu, M. Ambati, W. Srituravanich, C. Sun, and X. Zhang, Ultrasonic Metamaterials with Negative Modulus, Nat. Mater. 5, 452 (2006).

[4] S. H. Lee, C. M. Park, Y. M. Seo, Z. G. Wang, and C. K. Kim, Composite Acoustic Medium with Simultaneously Negative Density and Modulus, Phys. Rev. Lett. 104, 054301 (2010).

[5] F. Lemoult, N. Kaina, M. Fink, and G. Lerosey, Wave Propagation Control at the Deep Subwavelength Scale in Metamaterials, Nat. Phys. 9, 55 (2013).

[6] Z. Yang, J. Mei, M. Yang, N. H. Chan, and P. Sheng, Membrane-Type Acoustic Metamaterial with Negative Dynamic Mass, Phys. Rev. Lett. 101, 204301 (2008).

[7] M. Yang, G. C. Ma, Z. Y. Yang, and P. Sheng, Coupled Membranes with Doubly Negative Mass Density and Bulk Modulus, Phys. Rev. Lett. 110, 134301 (2013).

[8] J. Pierre, B. Dollet, and V. Leroy, Resonant Acoustic Propagation and Negative Density in Liquid Foams, Phys. Rev. Lett. 112, 148307 (2014).

[9] J. de Rosny and M. Fink, Overcoming the Diffraction Limit in Wave Physics Using a Time-Reversal Mirror and a Novel Acoustic Sink, Phys. Rev. Lett. 89, 124301 (2002).

[10] J. Li, L. Fok, X. Yin, G. Bartal, and X. Zhang, Experimental Demonstration of an Acoustic Magnifying Hyperlens, Nat. Mater. 8, 931 (2009).

[11] J. Zhu, J. Christensen, J. Jung, L. Martin-Moreno, X. Yin, L. Fok, X. Zhang, and F.J. Garcia-Vidal, A Holey-Structured Metamaterial for Acoustic Deep-Subwavelength Imaging, Nat. Phys. 7, 52 (2011).

[12] J. B. Pendry and L. Jensen, An Acoustic Metafluid: Realizing a Broadband Acoustic Cloak, New J. Phys. 10, 115032 (2008).

[13] M. Farhat, S. Enoch, S. Guenneau, and A. B. Movchan, Broadband Cylindrical Acoustic Cloak for Linear Surface Waves in a Fluid, Phys. Rev. Lett. 101, 134501 (2008).

[14] S. Zhang, C. Xia, and N. Fang, Broadband Acoustic Cloak for Ultrasound Waves, Phys. Rev. Lett. 106, 024301 (2011).

[15] X. Zhu, B. Liang, W. Kan, X. Zou, and J. Cheng, Acoustic Cloaking by a Superlens with Single-Negative Materials, Phys. Rev. Lett. 106, 014301 (2011).

[16] M. S. Kushwaha, P. Halevi, G. Martínez, L. Dobrzynski, and B. Djafari-Rouhani, Theory of Acoustic Band Structure of Periodic Elastic Composites, Phys. Rev. B 49, 2313 (1994).

[17] R. Martinezsala, J. Sancho, J. V. Sanchez, V. Gomez, J. Llinares, and F. Meseguer, Sound-Attenuation by Sculpture, Nature (London) 378, 241 (1995).

[18] M.-H. Lu, X.-K. Liu, L. Feng, J. Li, C.-P. Huang, Y.-F. Chen, Y.-Y. Zhu, S.-N. Zhu, and N.-B. Ming, Extraordinary Acoustic Transmission through a 1D Grating with Very Narrow Apertures, Phys. Rev. Lett. 99, 174301 (2007).

[19] J. Christensen, L. Martin-Moreno, and F. J. Garcia-Vidal, Theory of Resonant Acoustic Transmission through Subwavelength Apertures, Phys. Rev. Lett. 101, 014301 (2008).

[20] C. A. Condat and T. R. Kirkpatrick, Resonant Scattering and Anderson Localization of Acoustic Waves, Phys. Rev. B 36, 6782 (1987).

[21] B. Liang, B. Yuan, and J.-C. Cheng, Acoustic Diode: Rectification of Acoustic Energy Flux in One-Dimensional Systems, Phys. Rev. Lett. 103, 104301 (2009).

[22] B. Liang, X. S. Guo, J. Tu, D. Zhang, and J. C. Cheng, An Acoustic Rectifier, Nat. Mater. 9, 989 (2010).

[23] N. Boechler, G. Theocharis, and C. Daraio, Bifurcation-Based Acoustic Switching and Rectification, Nat. Mater. 10, 665 (2011).

[24] B.-I. Popa and S. A. Cummer, Non-Reciprocal and Highly Nonlinear Active Acoustic Metamaterials, Nat. Commun. 5, 3398 (2014).

[25] R. Fleury, D. L. Sounas, C. F. Sieck, M. R. Haberman, and A. Alu, Sound Isolation and Giant Linear Nonreciprocity in a Compact Acoustic Circulator, Science 343, 516 (2014).

[26] C. M. Bender and S. Boettcher, Real Spectra in Non-Hermitian Hamiltonians Having PT Symmetry, Phys. Rev. Lett. 80, 5243 (1998).

[27] Y.-C. Lee, M.-H. Hsieh, S. T. Flammia, and R.-K. Lee, Local PT Symmetry Violates the No-Signaling Principle, Phys. Rev. Lett. 112, 130404 (2014).

[28] C. E. Ruter, K. G. Makris, R. El-Ganainy, D. N. Christodoulides, M. Segev, and D. Kip, Observation of Parity-Time Symmetry in Optics, Nat. Phys. 6, 192 (2010).

[29] Z. Lin, H. Ramezani, T. Eichelkraut, T. Kottos, H. Cao, and D. N. Christodoulides, Unidirectional Invisibility Induced by PT-Symmetric Periodic Structures, Phys. Rev. Lett. 106, 213901 (2011).

[30] A. Regensburger, C. Bersch, M. A. Miri, G. Onishchukov, D. N. Christodoulides, and U. Peschel, Parity-Time Synthetic Photonic Lattices, Nature (London) 488, 167 (2012).

[31] L. Feng, X. F. Zhu, S. Yang, H. Y. Zhu, P. Zhang, X. B. Yin, Y. Wang, and X. Zhang, Demonstration of a Large-Scale Optical Exceptional Point Structure, Opt. Express 22, 1760 (2014).

[32] X. Zhu, L. Feng, P. Zhang, X. Yin, and X. Zhang, One-Way Invisible Cloak Using Parity-Time Symmetric Transformation Optics, Opt. Lett. 38, 2821 (2013).

[33] H. Ramezani, D. N. Christodoulides, V. Kovanis, I. Vitebskiy, and T. Kottos, PT-Symmetric Talbot Effects, Phys. Rev. Lett. 109, 033902 (2012).

[34] Y. D. Chong, L. Ge, and A. D. Stone, PT-Symmetry Breaking and Laser-Absorber Modes in Optical Scattering Systems, Phys. Rev. Lett. 106, 093902 (2011).

[35] S. Longhi, PT-Symmetric Laser Absorber, Phys. Rev. A 82, 031801(R) (2010).

[36] N. Bender, S. Factor, J. D. Bodyfelt, H. Ramezani, D. N. Christodoulides, F. M. Ellis, and T. Kottos, Observation of Asymmetric Transport in Structures with Active Nonlinearities, Phys. Rev. Lett. 110, 234101 (2013).

[37] H. Ramezani, T. Kottos, R. El-Ganainy, and D. N. Christodoulides, Unidirectional Nonlinear PT-Symmetric Optical Structures, Phys. Rev. A 82, 043803 (2010).

[38] F. Nazari, N. Bender, H. Ramezani, M. K. Moravvej-Farshi, D. N. Chirstodoulides, and T. Kottos, Optical Isolation via PT-Symmetric Nonlinear Fano Resonances, Opt. Express 22, 9574 (2014).

[39] B. Peng, Ş. K. Özdemir, F. Lei, F. Monifi, M. Gianfreda, G. L. Long, S. Fan, F. Nori, C. M. Bender, and L. Yang, Parity-Time-Symmetric Whispering-Gallery Microcavities, Nat. Phys. 10, 394 (2014).

[40] C. Yidong, Nonlinear Optics: Asymmetry from Symmetry, Nat. Phys. 10, 336 (2014).

[41] L. Ge, Y.D. Chong, and A.D. Stone, Conservation Relations and Anisotropic Transmission Resonances in One-Dimensional PT-Symmetric Photonic Heterostructures, Phys. Rev. A 85, 023802 (2012).

[42] J. Schindler, Z. Lin, J. M. Lee, H. Ramezani, F. M. Ellis, and T. Kottos, PT-Symmetric Electronics, J. Phys. A 45, 444029 (2012).

[43] Z. Lin, Honors thesis, Wesleyan University, 2012.

[44] Y. D. Chong, L. Ge, H. Cao, and A. D. Stone, Coherent Perfect Absorbers: Time-Reversed Laser, Phys. Rev. Lett. 105, 053901 (2010).

[45] Z. Lin, J. Schindler, F. M. Ellis, and T. Kottos, Experimental Observation of the Dual Behavior of PT-Symmetric Scattering, Phys. Rev. A 85, 050101 (2012).

[46] U. Leonhardt, Optical Conformal Mapping, Science 312, 1777 (2006).

[47] J. B. Pendry, D. Schurig, and D. R. Smith, Controlling Electromagnetic Fields, Science 312, 1780 (2006).

[48] Y. Cheng and X. J. Liu, Resonance Effects in Broadband Acoustic Cloak with Multilayered Homogeneous Isotropic Materials, Appl. Phys. Lett. 93, 071903 (2008).

[49] Y. Lai, H. Chen, Z.-Q. Zhang, and C. T. Chan, Complementary Media Invisibility Cloak That Cloaks Objects at a Distance Outside the Cloaking Shell, Phys. Rev. Lett. 102, 093901 (2009).

[50] C. Huanyang and C.T. Chan, Acoustic Cloaking and Transformation Acoustics, J. Phys. D 43, 113001 (2010).

[51] N. Kundtz, D. Gaultney, and D. R. Smith, Scattering Cross-Section of a Transformation Optics-Based Metamaterial Cloak, New J. Phys. 12, 043039 (2010).

[52] B.-I. Popa, L. Zigoneanu, and S. A. Cummer, Experimental Acoustic Ground Cloak in Air, Phys. Rev. Lett. 106, 253901 (2011).

[53] P. Zhang, M. Lobet, and S. He, Carpet Cloaking on a Dielectric Half-Space, Opt. Express 18, 18158 (2010).
