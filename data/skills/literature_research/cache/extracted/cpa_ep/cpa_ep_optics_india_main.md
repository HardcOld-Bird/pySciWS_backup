# Non-Hermitian Control of Topological Scattering Singularities Emerging from Bound States in the Continuum

Zarko Sakotic, $^{*}$ Predrag Stankovic, Vesna Bengin, Alex Krasnok, Andrea Alú, and Nikolina Jankovic

Leveraging topological properties in the response of electromagnetic systems can greatly enhance their potential. Although the investigation of singularity-based electromagnetics and non-Hermitian electronics has considerably increased in recent years in the context of various scattering anomalies, their topological properties have not been fully assessed. In this work, it is theoretically and experimentally demonstrated that non-Hermitian perturbations around bound states in the continuum can lead to singularities of the scattering matrix, which are topologically nontrivial and comply with charge conservation. The associated scattering matrix poles, zeros, and pole-zero pairs delineate extreme scattering events, including lasing, coherent perfect absorption, and absorber-lasers. The presented framework enables a recipe for generation, annihilation, and addition of these singularities in electric circuits, with potential for extreme scattering engineering across a broad range of the electromagnetic spectrum for sensing, wireless power and information transfer, polarization control, and thermal emission devices.

## 1. Introduction

The discovery and utilization of the topological properties of physical systems have shown an increasingly significant impact on science and technology in recent years. $^{[1-3]}$ The emergence of topological phenomena in wave systems has opened new research directions, such as immunity to disorder, $^{[4]}$ one-way transport, $^{[5]}$ and topological lasing. $^{[6]}$ The scope of topological photonics has recently expanded beyond the ideas of

topological photonic insulators, enabling nontrivial radiative and scattering phenomena. $^{[7,8]}$ Within this context, bound states in the continuum (BIC) arising in periodic structures have emerged as a beneficial phenomenon for controlling radiative processes. $^{[9]}$ BICs are resonant modes of open systems especially designed to eliminate radiative losses, achieving theoretically infinite quality factors, with a range of applications stemming from their topological nature, such as polarization control, $^{[10,11]}$ Q-factor enhancements, $^{[12]}$ unidirectional BICs, $^{[13]}$ vortex lasers, $^{[14]}$ and vortex beam generation. $^{[15]}$ Beyond the versatile radiation control, they have also been connected to topological scattering processes in photonic crystals, $^{[16]}$ which enabled arbitrary polarization conversion through harnessing topological properties in the scattering response. $^{[17]}$ More recently, the topological scattering features of BICs were discussed in non-periodic, planar reflective structures using epsilon-near-zero (ENZ) and epsilon-near-pole (ENP) materials, $^{[18]}$ which was also leveraged in acoustic $^{[19]}$ and nonreciprocal electromagnetic systems. $^{[20,21]}$ BICs in planar reflective systems are associated with pairs of perfect absorption singularities and reflection phase vortices, which comply with topological charge conservation. Separately from BIC-related phenomena, the topological nature of perfect absorption and reflection zero singularities was also recently studied in metasurfaces for polarization control $^{[22,23]}$ and optical multilayers for sensing, $^{[24,25]}$ highlighting the importance of topological concepts in scattering phenomena.

Singularities in the scattering response have been investigated far beyond just perfect absorption or critical coupling. For example, it has been shown that perfect absorption is a special case of a more general phenomenon, coherent perfect absorption (CPA), in multiple-port or multi-modal systems. $^{[26-29]}$ CPA is the time-reversed version of lasing at the threshold, assuming an ideally linear system. These phenomena are related to singularities of the scattering-matrix eigenvalues, that is, zeros (CPA) and poles (lasing at threshold). In this context, systems obeying parity-time-symmetry (PT-symmetry) have been shown to enable peculiar solutions of Maxwell's equations supporting exceptional points (EPs), $^{[30]}$ CPA-lasing degeneracies, $^{[31,32]}$ unidirectional invisibility, $^{[33]}$ unidirectional spectral singularity, $^{[34]}$ virtual perfect absorption, $^{[35]}$ passive, $^{[36]}$ and transient PT-symmetry. $^{[37]}$ It has also been shown that in purely lossy systems, it is possible to engineer exceptional points of absorption (CPA EP) with distinct, broadened lineshapes. $^{[38,39]}$ Apart from zeros of the S-matrix operator, special interest has also been devoted to reflectionless scattering modes, $^{[40-42]}$ for which the unusual zero EP behavior can be observed in a chosen set of input channels' reflection.

All these S-matrix singularities form the basis for a plethora of applications, including enhanced sensing, $^{[43-46]}$ unconventional lasing, $^{[47]}$ topological wireless power transfer, $^{[48,49]}$ efficient energy storage, $^{[50]}$ analog differentiation, $^{[51]}$ and nonreciprocal microwave transport. $^{[52]}$ Moreover, the general wave nature of these phenomena enables them in different wave systems, such as elastic $^{[53]}$ and optomechanical systems. $^{[54]}$ Their ubiquity and importance has especially been proven in disordered electromagnetic systems, $^{[55-57]}$ which can be reconfigured and offer more flexibility and precision in designing sensitive singularity-based devices. Although the scattering singularity-based research has considerably proliferated in recent years, their topological features were seldomly discussed and have not been fully assessed. For instance, elucidating the connections between BICs and other singular points may further boost their manipulation and enable novel functionalities for high-performance devices.

Here, we present a new outlook on scattering matrix singularities and BICs, and experimentally demonstrate their topological nature in two-port networks. We discuss a broad range of lossless two-port structures that support BICs, including optical and electronic systems. We then show that a non-Hermitian perturbation destroys the BIC, creating pairs of topologically protected S-matrix singularities whose nature depends on the type of perturbation. In particular, we show that loss, gain, and PT-symmetric perturbations lead to the creation of pairs of CPA, laser, and CPA-laser degeneracies, respectively. Each of these conditions is characterized by a winding number in the phase of the S-matrix eigenvalues, and all of them comply with charge conservation. We theoretically and experimentally demonstrate their topological features using electric circuits, validating the general nature of the presented ideas and proposing a recipe to generate topological singular points. These findings lie at the intersection of topology, non-Hermiticity, BICs and singular responses, and they may facilitate the design of sensors, thermal emitters, wireless power and information transfer systems, polarizers, and other electromagnetic devices.

## 2. BICs and Topological S-Matrix Singularities

The conceptual sketch of topological charge creation from BICs is shown in Figure 1a: a BIC is transformed into a pair of topological charges corresponding to CPA (CPA-laser) states when introducing a defect in the system characterized by loss (PT-symmetry). Due to the ubiquitous wave nature of BICs, which can be realized across the electromagnetic spectrum in different platforms, we analyze three analogous systems where the proposed topological features can be observed. In this way, we aim at highlighting the general nature and wide applicability of the proposed framework. At this point, it is instructive to differentiate between topological aspects of BICs in terms of their radiation and scattering properties. As mentioned in the introduction, BICs have so far mostly been explored for their topological charges in radiation, where they behave as eigenpolarization singularities in the momentum space of periodic structures, $^{[9]}$ and where they can be controlled by varying geometrical parameters and/or breaking spatial symmetries in photonics crystals. In this work, we focus on the topological scattering properties of BICs in real space and use non-Hermitian perturbations as a means to control the emerging singularities, expanding the topological picture and the tools of control.

We start the discussion with the general two-port optical system shown in Figure 1b—a dielectric cavity created between two identical resonant layers, representing a generalized Fabry–Perot-type photonic system. In this example, the permittivity of the resonant layers is assumed to have either an ENZ (plasma) or an ENP (Lorentzian) resonance. Such resonances are commonly found in isotropic, anisotropic and 2D materials, such as InAs, ITO, SiC, $\alpha$ -MoO $_{3}$ , and hBN. However, they can also be tailored at the frequency of interest using metamaterials and metasurfaces $^{[58,59,62]}$ —an example of an analogous metasurface system is sketched in Figure 1c. Neglecting loss for the moment, if one of the cavity resonances spectrally coincides with one of the material or metasurface resonances, a perfectly trapped mode or BIC can be induced. $^{[18,62]}$ This feature can be attributed to waves experiencing a hard-wall boundary (perfect mirror) when the wave impedances of the top and bottom layers go to zero or infinity. $^{[18]}$ The resonant thickness of the dielectric, that is, the BIC condition, depends on the permittivity $\epsilon_{d}$ and incident angle $\theta$ as

$$
d = \frac {n \lambda_ {r}}{2} = \frac {n c}{2 f _ {0} \sqrt {\varepsilon_ {d} - \sin^ {2} \theta}}, n = 0, 1, 2, \ldots ,\tag{1}
$$

where $f_{0}$ represents the material or metasurface resonant frequency. As mentioned, this type of BIC can arise in a broad range of materials at different wavelengths, for example, using SiC, $^{[60,61]}$ plasmonic metasurfaces, $^{[62]}$ photonic crystals, $^{[63]}$ and 2D-materials such as $\alpha$ -MoO $_{3}$ . $^{[64]}$ Two examples of optical systems supporting BICs, using $\alpha$ -MoO $_{3}$ layers and Ag-nanoparticle metasurfaces, are discussed in more detail in the Supporting Information. Although their topological nature in lossy one-port structures has recently been explored $^{[18]}$ and utilized for nonreciprocal thermal emission, $^{[20,21]}$ these features have not yet been confirmed experimentally. Furthermore, their topological nature has not been discussed in two-port networks, where, apart from loss, PT-symmetry can also be studied.

(a)
![](images/e139f6977de04600c010460fee88f57523b6100ec9040bfa1454e2d1b0f96225.jpg)

(b)
![](images/8b192563955af9c15eac621fc3cf4d2d80530f7084972abad07eb8d7d3d4f09b.jpg)

(c)
![](images/529c5af66a480436736c767f177b790268be68c0ea6741673ad193c70a2b3d0c.jpg)

(d)
![](images/d50c30efabc8b0960248ccfa6c9176773a18078ebbdc9afab4364ba98c9913b9.jpg)

(e)
![](images/4a2a567b3bf1c5d393e4254606e1e80f6be678c6f1d70c1ee5b1ff3caa1aee3e.jpg)
(f)

Lossy
![](images/c9e1097738476fd82a56295c404e88f3ad99b80ffeb67a9f903e1243b18a7050.jpg)

(g)
PT-symmetry
![](images/c616437c57cb47346099ba1a185f063fde0e3bd00aa5c6ba11ce92ad157d79e0.jpg)

![](images/75f0b9b4581257de3a0f1a297f360b26a25ce43629e4b057c0072549fc1e77d3.jpg)
$d/\lambda_{0}$

![](images/dfb7f70cd45afd6d4206e947ad0a3d7d891212e711d42808631b5c3a58e031ca.jpg)
$d / \lambda_{0}$

![](images/6ef642edefc527dab8893ce17a0d29db1c9c0fb8b3ba00cfbdd8653be0edec27.jpg)
$d / \lambda_{0}$
Figure 1. a) Pairs of topological charges emerging from BICs in various physical systems. b) Optical 3-layer system for a range of materials having ENZ or ENP resonances such as SiC, hBn, and $\alpha$ -MoO $_3$ . c) Two plasmonic metasurfaces separated by a spacer. d) Analogous electrical circuit supporting BICs. e) S-matrix pole and zero dispersion for the circuit calculated in the $d$ - $\omega_{i}$ - $\omega_{r}$ space (upper panels) and phase of the multiplied eigenvalues in the $d$ - $\omega_{r}$ plane (lower panels) for the lossless case $R = 0$ , f) lossy case $R = 1\Omega$ , and g) PT-symmetric case $R = \pm 5\Omega$ .

Due to feasibility, wide availability and cost-efficiency, an appealing platform to explore the topological features of BICs and related phenomena is formed by radio-frequency electronic circuits. To this end, in our study, we propose the basic circuit, analogous to the proposed optical multilayer and metasurface structures, sketched in Figure 1d. For brevity, the following discussion will focus on this electric circuit, while the same analysis for optical systems is provided in the Supporting Information. The circuit has two ports and consists of two shunt RLC resonant circuits separated by a transmission line of length d. This system is analogous to the optical structures under normal incidence when the resonant layers have an ENP resonance (Lorentzian), $^{[18]}$ that is, the series impedance of the resonant layers has a minimum at resonance. The polarization and incident angle degrees of freedom are not present in the circuit scenario. At the resonant frequency $\omega_{0}=1/\sqrt{LC}$ , the impedance of the shunt lines is equal to R. When the transmission line length is

$$
d = \frac {n \lambda_ {0}}{2} = n c \pi \sqrt {L C}, n = 0, 1, 2 \ldots ,\tag{2}
$$

and when R is ideally zero, the circuit supports a BIC—a Fabry–Perot mode perfectly trapped between two short circuits. In the optical case, this is equivalent to the Fabry–Perot dark mode of a dielectric cavity sandwiched between two perfect conductors. When loss, gain, or a PT-symmetric perturbation is added to the RLC circuits, the BIC is revealed to be a degeneracy of scattering matrix singularities that are topologically protected, as shown in Figure 1e–g and analyzed further.

To analyze the BIC and the emerging charges, we use the complex frequency notation and the scattering matrix formalism. Specifically, we look for singularities (poles and zeros) of the scattering matrix eigenvalues in the complex frequency plane, which can describe the scattering process and emerging anomalies. $^{[65]}$ To visualize the charges appearing at real frequencies, we plot the phase of the multiplied eigenvalues of the S-matrix $\arg(s_{1}s_{2})$ , which contain all undefined phase points (vortices) associated with singular values, that is, poles and zeros of the S-matrix eigenvalues where the phase is ill-defined. The polarity and winding number q of the charges can be found by integrating the phase accumulation counterclockwise in the parameter space around the charge $q=\frac{1}{2\pi}\oint d\phi$ . $^{[18]}$ The eigenvalues of the reciprocal two-port circuit are given by $s_{1/2}=t\pm\sqrt{r_{l}r_{r}}$ , where transmission t and reflection coefficients $r_{l}, r_{r}$ are calculated using the ABCD matrix formalism [Supporting Information].

In the lossless case, the presence of BICs corresponds to a diverging phase resonance in the eigenvalue spectrum, as seen in the bottom panel of Figure 1e. In the complex frequency plane (upper panels), the dispersion of eigenvalue poles and zeros mirror each other due to time-reversal symmetry and Hermiticity, and they are parabolic around the resonant condition. Furthermore, the BIC is formed when these dispersion curves touch on the real-frequency axis, that is, a BIC corresponds to a real-frequency pole-zero degeneration in a passive Hermitian system. $^{[18]}$ When loss is added, Figure 1f, these dispersion curves move down along the imaginary frequency axis, inducing two real-frequency zeros of the scattering matrix, that is, two CPAs. As loss is added, these points move in parameter space, but they cannot disappear unless they are annihilated with an oppositely charged singularity. The time-reversed scenario (not shown) entails adding gain in both resonators, shifting the curves up in the complex frequency plane and inducing two real-frequency poles, that is, lasing conditions. It is worth noting that, by placing a perfect mirror in symmetry points (middle of the spacer) of the system, we obtain a one-port structure and CPA solutions become PA solutions, which were explored in detail in. $^{[18]}$ Lastly, we consider a PT-symmetric perturbation to the system, that is, RLC tanks with positive and negative resistances, Figure 1g. Now, the zero-dispersion translates down, while poles translate up along the imaginary frequency axis. Remarkably, these dispersions intersect at the same real frequency, creating two CPA-laser points. The eigenvalue phase spectrum contains singularities with charges of $\pm2$ in the phase, indicating the presence of both a pole and a zero. If PT-symmetry holds, CPA-laser points are topologically protected and move in parameter space for different $|R|$ values.

Due to the translation of system poles in the upper complex half-plane around the BIC, the parametric region between two CPA-laser charges originating from the same BIC is inherently unstable. As we show later, these charges can disappear only through annihilation. However, their dispersion carries important implications for system stability—an aspect often overlooked in theoretical studies of PT-symmetric systems. We note that a similar finding was previously reported in 1D periodic PT-symmetric systems, $^{[66]}$ where PT-symmetric perturbations create so-called PT-BIC rings, with similar implications for stability, $^{[67]}$ although the CPA-laser aspect was not discussed. The proposed theoretical framework reveals a fundamental connection between BICs and different scattering extrema and the topological nature of the emerging singularities. Furthermore, the same phenomena were shown to arise in different electrical and optical systems, indicating the ubiquity and wide applicability of the proposed theory, which we further discuss in the following and verify experimentally.

## 3. Experimental Demonstration of CPA Topological Charge Conservation in Electric Circuits

CPA points were shown to arise from BICs in pairs after adding loss to the RLC resonators in Figure 1. This finding represents a generalization of the topological theory associated with perfect absorption points and BICs in single port structures proposed in ref. [18] To further generalize the discussion and gain more insight into these CPA states, we analyze the system shown in Figure 2a inset. We simplify the structure by analyzing two complex impedances $Z_R$ separated by a transmission line in a two-port network, where $Z_R$ represents the total complex impedance of the series RLC circuit at a fixed frequency $Z_R = R + j\omega L + 1 / j\omega C$ . Using the ABCD matrix formalism, we find the reflection/transmission coefficients and the analytical solutions for CPAs [Supporting Information]. Namely, for a mirror-symmetric and reciprocal system, the zeros of the two eigenvalues are given by $s_{1/2} = t \pm r = 0$ . The two solutions for $Z_R$ as a function of $d$ give rise to symmetric and antisymmetric CPAs:

(a)
![](images/7076fafbde0fb89d0dffacb02c7f509223650df0b1b3e651b25f97af87de00ce.jpg)

(c)
![](images/e3c02e6e446a53e0a319a156815e7bf08a3043192f549287ebe8223e9dc94ace.jpg)

![](images/97407387fdb915e28ce7156f13d18dc9327148903b6e146414db6531edbe1643.jpg)
(d)

(e)
![](images/fdf442730db5883141870affb4dea7109287a54f2b83ae9fd4fa1d7ea9f0164f.jpg)
(f)

![](images/22b5c6290b8c15945eca8acb4e94d5763195df970e41e4d136236a831ce81372.jpg)

![](images/675394643362f2e6a02783e918aafb203fe83522316ad561c85f5d1c6c48798e.jpg)
Figure 2. Analytical condition for a) symmetric and b) antisymmetric CPAs for the circuit model show in the inset of (a). c) Schematic of the proposed RF-circuit. d) Photograph of the realized circuit on a protoboard. e) Comparison of the theoretical and measured eigenvalue amplitude for $R = 50\Omega$ , $L_{1} = 1000\mathrm{nH}$ , $C_1 = 217\mathrm{pF}$ , and $d = 0$ (no transmission line), showing a zero associated with symmetric CPA. f) Antisymmetric CPA for $R = 50.5\Omega$ , $L_{1} = L_{2} = 1500\mathrm{nH}$ , $C_1 = C_2 = 217\mathrm{pF}$ which corresponds to $d = \lambda /2$ . Network characteristic impedance is $Z_0 = 50\Omega$ .

$$
Z _ {R s} = \frac {Z _ {0}}{2} \left(\cos (k d) + \sqrt {\cos^ {2} (k d) + 2 j \sin (k d)}\right),\tag{3}
$$

$$
Z _ {R a} = - \frac {Z _ {0}}{2} \left(\cos (k d) - \sqrt {\cos^ {2} (k d) + 2 j \sin (k d)}\right),\tag{4}
$$

where $k = 2\pi/\lambda$ is the wavenumber, and $Z_{0} = 50 \Omega$ is the network characteristic impedance. For a fixed frequency $\omega = \omega_{0}$ , Figure 2a,b shows the CPA solutions for different transmission line lengths. When d = 0 (no transmission line), the symmetric CPA exists for $Z_{R} = 50 \Omega$ , while for the antisymmetric case, the solution requires that $Z_{R} = 0$ , that is, the system supports an antisymmetric BIC and cannot absorb waves. As d increases, $Z_{R}$ solutions for CPA become complex, with opposite signs of the imaginary part for symmetric and antisymmetric cases. At $d = \lambda/2$ , the system supports a symmetric BIC ( $Z_{R} = 0$ ) and an antisymmetric CPA ( $Z_{R} = 50 \Omega$ ). These solutions periodically repeat as d increases.

To confirm the existence of these CPAs and demonstrate their topological nature, we built an electric circuit consisting of two shunt RLC resonators separated by a double T-circuit, which plays the role of a transmission line and provides the required phase delay between two identical resonators [Supporting Information], Figure 2c. A photograph of the protoboard realization of the proposed electric circuit is shown in Figure 2d. First, we validate the existence of the symmetric and antisymmetric CPAs and measure the response of the two circuits. To account for parasitic effects dominated by the series resistance of inductors, we add series resistors to each inductor in the circuit, which we evaluated to be $\approx1\ \Omega$ at the measured frequencies. The first circuit has a resonant frequency of 10.8 MHz and no transmission line (d=0). With the total series resistance R=50 $\Omega$ (including parasitics) of the RLC tank, the circuit response is in perfect agreement with theory, Figure 2e. The zero in the eigenvalue magnitude spectrum $|s_{1}|=|t+r|$ confirms the emergence of a symmetric CPA. The second circuit has a T-circuit between the two resonators with R=50.5 $\Omega$ , where the T-circuit response is identical to the one of a transmission line with length $d=0.5\lambda_{0}$ at the resonance frequency $f_{0}$ [Supporting Information]. The response displays an antisymmetric CPA with $|s_{2}|=|t-r|$ going to zero at the resonance frequency 8.82 MHz, Figure 2f, for which experimental and theoretical results are in excellent agreement. The resistance R is slightly larger than the theoretically predicted $R=Z_{0}=50\ \Omega$ due to the parasitic resistance in the LC tanks of the transmission line.

The CPA is observed through the eigenvalues retrieved from the measured S-matrix. An in situ observation of the CPA is also reported in Supporting Information. The CPA wavefront consists of two coherent input waves of the same frequency, amplitude, and phase difference of either 0 or $\pi$ (symmetric or antisymmetric CPA). Interestingly, these CPA solutions (with the same CPA wavefronts) also exist for asymmetric circuits when $R_{1} \neq R_{2}$ , for any combination satisfying $R_{1} // R_{2} = 25\Omega$ . They even persist for different combinations of capacitance and inductance values in the left and right circuits, as long as the same resonant frequency is supported in both circuits. However, the absorption line shape at the CPA can change substantially for different combinations of $R_{1}, R_{2}, L_{1}, L_{2}, C_{1}$ , and $C_{2}$ [Supporting Information]. Although single port realizations exploring scattering singularities are less practically demanding in most cases, the proposed two-port setup can be explored further with asymmetries, unlocking phenomena beyond those available in one-port systems, as we show in the following when considering active and PT-symmetric circuits, as well as in examples shown in the Supporting Information. Furthermore, singular points such as CPAs in two-port structures allow the possibility of coherent and dynamic control of scattering features using multiple coherent input signals, that is, they enable the control of light with light without using nonlinearities, $^{[27-29]}$ such as coherent control of polarization features, $^{[68,69]}$ spin-to-orbital angular momentum conversion, $^{[70]}$ diffraction, $^{[71]}$ and fluorescence. $^{[72]}$

$C_{1}[pF]$
(a)
![](images/01c9faabe583ad0b44c210974559ed288fb5795cb25008ee5fa811e10abc8441.jpg)

![](images/980c518f3f5503316a8b577932eca9bcf98103e8039ee4311418c61166099aca.jpg)

![](images/36098a3a7f64247e42fba954503a000d6e252a3cf8c9eaa0d908e95bc7348155.jpg)

![](images/0b7811e881bc6c8cfe265893bcaf93af49c4fef529f8485c349e252a83725dee.jpg)

![](images/86c83697539b80196cc09bb22d3eac17e2a0b661f17e58808915bd89383516f5.jpg)

(b)
(c)
![](images/e964378675e3603173ac0ce014287f231b527efdb55fb4bd604be18bd446cc2a.jpg)

![](images/ab2bdbd4e5a18331c4ecd5060ef1eb7bac437960d31a4819847b62c4a4fe0086.jpg)

![](images/bd5515b3eff2f088f6fbca61bed6d94ca882f44ef933f75e03d80d2dd53e2cd1.jpg)
Figure 3. a) Eigenvalue magnitude $|s_{2}|$ for five different values of R. Horizontal colored lines represent the parametric points to be measured. b) Numerically obtained CPA condition in the parameter space $f-R-C_{1}$ , where $C_{1}$ is represented by the color. Colored dots represent different values for $R-C_{1}$ to be measured. c) Theoretical and experimental obtained $|s_{2}|$ measured for five different parameter points, where the trajectory and eventual annihilation of the CPA is evident. Reactive components used are $C_{2}=217\ pF$ , $L_{1}=L_{2}=1500\ nH$ ,

In order to demonstrate the topological nature and charge conservation of the CPA states, we experimentally show the annihilation of antisymmetric CPA charges using the same circuit in Figure 2d. To understand the dynamics of the annihilation process, we analyze the 2D parameter space consisting of frequency and capacitance $C_{1}$ , exploring the zeros of $s_{2}$ . In this parameter space, zeros (CPA charges) originating from neighboring BICs move towards each other for increasing values of R, Figure 3a. At the critical value of 50.5 $\Omega$ (middle panel of Figure 3a), the charges meet and annihilate. This is evident when R further increases, and the $s_{2}$ zeros associated with the CPAs disappear, as shown in the 4th and 5th panels of Figure 3a. The calculated trajectory of these states in the 3D parameter space $f-R-C_{1}$ is shown in Figure 3b, where the third dimension is represented by color. By tuning the circuit elements, specifically R and $C_{1}$ , different CPA points can be accessed according to this dispersion.

To trace the dispersion of these charges in a real circuit, we modified the original circuit in Figure 3d by adding identical capacitances in the left and right RLC tanks to yield the required capacitance. Furthermore, using identical potentiometers placed in both RLC tanks, we can tune the resistance to access the desired CPA points. We measure the frequency response of the circuit at five different parameter points (values of $C_{1}$ and R shown on panels in Figure 3a), represented by the colored horizontal lines and dots in Figures 3a and 3b, respectively. The theoretical and experimental results are in excellent agreement, Figure 3c. The zero-magnitude point associated with one antisymmetric CPA charge moves through the frequency space until it annihilates with an opposite charge for $R = 50.5 \Omega$ , demonstrating the dispersion and topological charge conservation of CPA states. The presented recipe for their creation and annihilation can be extended to various wave systems, including the optical and infrared electromagnetic systems, as well as acoustic and mechanical systems, where topology plays an increasingly important role in the design of novel devices, including wireless power transfer, polarization control, and thermal emission manipulation.

## 4. PT-Symmetry and Topological CPA-Laser Charge

We next study a PT-symmetric RF circuit, showing a simple procedure to realize a topological CPA-laser degeneracy. We consider the system shown in Figure 4a, similar to the one analyzed in Figure 2a, but yielding positive and negative shunt resistance on different sides of the system—in such a way to obey PT-symmetry. A negative resistance can be realized using a negative impedance converter, which has already been used to explore PT-symmetry in circuit scenarios. $^{[73-75]}$ Alternatively, a recent demonstration has shown that building fully integrated PT-symmetric circuits is also possible, $^{[52]}$ offering more flexibility in the design. Using the same ABCD matrix formalism as for the previous analysis, we compute the eigenmodes with real eigenfrequency, that is, real-frequency poles of the scattering matrix eigenvalues [Supporting Information]. The real and imaginary parts of the complex impedance $Z_{R}$ are

(a)
![](images/d9204953dd332e2fd13169921796cd637f9284de7ea7f1b4847d15011c69689a.jpg)

(b)
![](images/b8e0aaf45bfb7cd006d06b277ca9620d8cfff29e79a7d40ea0a056f0561c5c4d.jpg)

![](images/c055cadf713a037e7bc393902ba51af02303b6310ba05d10a2e83770136796b0.jpg)

(d)
![](images/0ebe781db42047dda4dd503c08cd353f7424a4afdbb573055aaf71443fd3dc65.jpg)
Figure 4. a) PT-symmetric circuit. b) Analytical condition for CPA-laser states at a fixed frequency. c) Total outgoing power calculated at two CPA-Laser points with different excitation phases (red and blue lines, $A_1 = 1.87$ , $A_2 = 1$ , and $\psi = \pm \pi / 2$ ). The inset shows the phase singularities associated with CPA-laser points. d) $P$ calculated at $R = 50 / \sqrt{2} \approx 35.355 \Omega$ where two CPA-Laser charges merge (phase $\arg |s_1 s_2|$ shown in the inset) for $A_1 = 2.4142$ , $A_2 = 1$ and $\psi = \pm \pi / 2$ . For $R = 37 \Omega$ , charges are destroyed as evident from the stripped line and right inset (no phase singularities, $A_1 = 2.274$ , $A_2 = 1$ and $\psi = \pm \pi / 2$ ). Red-shaded regions in (c) and (d) designate unstable regions.

$$
R = \frac {Z _ {0}}{\sqrt {2}} \sqrt {\sin^ {2} (k d) - \frac {\sin^ {2} (2 k d)}{8}}
$$

(5)

$$
X = - \frac {Z _ {0}}{4} \sin (2 k d)\tag{6}
$$

Due to the PT-symmetry, each real-frequency pole degenerates with a zero, implying the emergence of CPA-laser states. Thus, Equations (5), (6) constitute the conditions for CPA-laser states, Figure 4b. To show their usual features of absorbing and lasing at the same frequency, we plot the total output power for two CPA-laser points, emerging for $Z_{R} = 20 \pm j12.4 \Omega$ at two different values of d for different phase inputs, Figure 4c. We plot the total outgoing or scattered power, defined as

$$
P = \left| a _ {1} r _ {L} + a _ {2} t \right| ^ {2} + \left| a _ {2} r _ {R} + a _ {1} t \right| ^ {2}\tag{7}
$$

where $a_{1}=A_{1}\cos(\omega t)$ and $a_{2}=A_{2}\cos(\omega t+\psi)$ represent input signals with amplitudes $A_{1/2}$ and with relative phase difference $\psi$ . At the CPA-laser point, the output P is infinitely large, as there is a pole on the real frequency axis. However, exciting the system with the eigenvector corresponding to the zero of the S-matrix eliminates all outgoing power, and all the energy is absorbed, as the blue curves show in the graph. Additionally, we plot the phase of the eigenvalues in the inset, which clearly shows two points with charges of +2 and -2 associated with the CPA-laser condition. As the value of $|R|$ increases, these charges move closer to each other until they meet at $d/\lambda_{0}=0.25$ and $R=50/\sqrt{2}$ , and annihilate each other for $R>50/\sqrt{2}$ , Figure 4d. As shown in the inset, the phase singularities disappear after annihilation.

crucial aspect of the active systems that has not yet been discussed is their stability. As shown in Figure 1f and briefly discussed above, PT-symmetric perturbations to the BIC create an unstable parametric region between two emerging CPA-laser charges. The pole dispersion translates above the imaginary-frequency axis in the complex plane. Similarly, in Figure 4c two CPA-laser states imply the emergence of points at which the pole dispersion crosses the real axis and moves to the upper complex frequency half-plane, indicating an unstable region (shaded red). As $|R|$ grows to $50/\sqrt{2}$ and beyond, the CPA-laser charges move to the same position and annihilate, at which point all poles are unstable, Figure 4d and [Supporting Information]. In the present analysis, the circuit response is studied at a fixed frequency, hence without considering the dispersion of the involved active circuitry. Additional discussions on this issue based on complex frequency analysis can be found in the Supporting Information. Interestingly, the case $|R| = 50/\sqrt{2}$ was exploited in a similar circuit layout for enhanced sensing, $^{[76]}$ indicating a possible route to experimentally explore the discussed features, operating close to instability. Thus, the annihilation of CPA-laser charges would be impossible to observe in a real system due to the inherent instability after annihilation. It is also worth noting that, although operating near such singularities can boost the performance of a device, the use of active circuits comes at a price of increased energy consumption, which should be taken into account when comparing performance metrics. Nevertheless, the proposed concept presents an important connection between BICs, CPA-laser degeneracies, and topology. Our findings reveal a simple procedure to generate CPA-laser states, promising for next-generation sensors $^{[46,76]}$ and wireless power and information transfer systems. $^{[48,49,77-78]}$ Additionally, we have formulated design guidelines to address stability, a crucial aspect for active and PT-symmetric systems.

(a)
![](images/c894784661bcaf389c19841c732c01d4e79799b0c634c550d5d2bacaa6c47180.jpg)

(c)
(d)
![](images/c20184ae37a5e72ce94f8bc0987ce48a7d06f9598514179c9cd552a3854465ab.jpg)

![](images/ca342e13fb37f29c8e0c7b62a78b19e59ef1c309c4b95b0b23f28d38ffe4a63d.jpg)

(b)
![](images/f75ff2d06fdc7e3315a9bed8edb339b4c206f5afd4aaaaf8ec9dfd80cc01d3b0.jpg)

(e)
![](images/cc3f021373004c41fb116f39bdd2ec178be7481612ddebdab0c1f1caa44679ee.jpg)
Figure 5. a) Circuit with a negative impedance converter, $L_{1} = L_{2} = 1520 \, nH$ , $C_{1} = 27.8 \, pF$ , $C_{2} = 217 \, pF$ . b) Eigenvalue phase in the complex frequency plane where zero EP and pole EP are visible as $\pm 2$ charges. For a passive circuit $R_{1} = R_{2} = 47 \, \Omega$ , the zero-EP is at the real frequency axis forming a CPA-EP. c) Theoretical reflection and transmission coefficients for lossy structure, same as (b). d) Unidirectional absorber, e) Unidirectional spectral singularity. For simulation purposes, we used $R_{3} = R_{4} = 1000 \, \Omega$ , and $R_{2a} = 49.5 \, \Omega$ to obtain the required negative resistance at the desired frequency. We have also added an additional series inductance in the right circuit $L_{p} = 100 \, nH$ to cancel out the parasitic reactance added by the negative impedance converter and equalize the resonant frequencies of left and right resonant circuits.

## 5. Charge Addition, CPA Exceptional Points and Unidirectional Spectral Singularities

Thus far, we have explored the conservation of topological charges through their annihilation. In this section, we explore the possibility of charge addition, which will entail merging of two CPA solutions, forming an exceptional point (EP) of zeros and enabling unusual physics. Traditionally, EPs emerge when two modes support degenerate eigenvalues and eigenvectors, with far-reaching consequences. $^{[30]}$ An EP is represented in the complex frequency space by two merging poles. $^{[65]}$ In contrast, the time-reversed version of such a condition is represented by merging two zeros, that is, a CPA EP, which was recently theoretically and experimentally shown in optical microcavities. $^{[38,39]}$ Using the introduced topological features of CPAs, we demonstrate here that two CPA charges of the same polarity can merge to form an EP of zeros.

Using the passive version of the circuit in Figure 5a, two charges of the same polarity can be brought together at the same real frequency. Due to the same polarity, CPA charges do not annihilate and have a total of +2 charge in phase space, while the analysis of the same system in the complex frequency plane reveals an EP of both poles and zeros, Figure 5b. Since both eigenvalues tend to zero at the same time, this can be characterized as a non-generic CPA EP, where the quartic absorption line is absent, following the discussion in refs. [38, 39]. The existence of such a CPA EP in our system is not surprising: the T-circuit acts as a low pass filter and suppresses transmission for higher frequencies [Supporting Information]. In this scenario, when the transmission is eliminated, the emerging CPAs will be identical to reflection zeros $s_{1/2} = t \pm \sqrt{r_{l} r_{r}} = 0$ , $t = 0 \rightarrow r_{l} = r_{r} = 0$ , and these zeros are co-located due to the mirror symmetry of the circuit.

Although this CPA EP appears trivial at first glance, a rather interesting feature arises when we analyze the PT-symmetric version of this circuit, which breaks mirror symmetry. By varying $R_{2}$ to 0, and then to $-R_{1}$ in the right side of the circuit, a unidirectional resonant transmission is achieved, which was first theoretically explored in ref. [34]. Remarkably, the circuit is transparent from one side with zero reflection and full transmission, while from the other side, the circuit acts like a laser, Figure 5e. As mentioned earlier, negative resistance can be achieved using a negative-impedance converter configuration with an operational amplifier, as sketched in Figure 5a. To test our theoretical prediction obtained with an ideal negative resistor $R_{2}$ , we simulated the response of the PT-symmetric circuit using a realistic model for the commercially available operational amplifier (Texas Instruments LMH6714) in the negative-impedance-converter configuration, which takes into account dispersion effects [Supporting Information]. This configuration provides an effective negative resistance, and the response is in excellent agreement with our theoretical prediction, Figure 5e.

Although the T-circuit strongly suppresses transmission, the peculiar physics of EPs at hand allows full transmission at the unidirectional spectral singularity, which can be exploited for wireless power and information transfer applications. $^{[77]}$ Namely, the transmission can be fully restored with the active circuit, even if the channel is fully attenuated when the circuit is passive. This is expected since one of the eigenvalues is zero, that is, $s_{1/2} = t \pm \sqrt{r_{l} r_{r}} = 0$ . Due to the singular values of the reflection coefficients ( $r_{l} = 0$ and $r_{r} \to \infty$ ) at the resonant frequency in this ideal PT-symmetric case, that is, $|r_{l}| = 1 / |r_{r}| = 0$ , the amplitude of the transmission coefficient must be equal to 1. Notably, the system is stable, as the only pole near the real-frequency axis is associated with the discussed singularity, and it can be explicitly controlled with $R_{2}$ . These results reveal an essential connection between CPA EP and unidirectional spectral singularity.

## 6. Discussion

The topological CPA points emerging around BICs in passive electronic circuits and their optical counterparts are ideally suited to control the scattered waves' amplitude, phase, and polarization. In reflective, one-port structures, these singular points have already been utilized for polarization control, $^{[16-18]}$ thermal emission engineering, $^{[20,21]}$ phase manipulation, $^{[22,23]}$ and interferometric phase sensing. $^{[18,24,25]}$ However, singular scattering points, such as CPAs, in two-port structures allow the possibility of coherent and dynamic control of scattering features. $^{[27-29,68-72]}$ As we have shown, our general framework can facilitate the creation of topological scattering singularities across different geometries and spectral regions, expanding the applicability of such setups. For example, as shown in our analysis of the optical structure, our framework also extends to the infrared regime, where naturally birefringent and dichroic materials are scarce. This could enable coherent control of infrared polarization features through scattering in metasurfaces or even lithography-free structures, $^{[18,79]}$ as polar dielectrics and 2D materials such as hBN and $\alpha$ -MoO $_{3}$ can provide the discussed singular points in the mid- and long-wave infrared regions.

On the other hand, the new insights on topological singular points in PT-symmetric systems expand the utility of our proposed framework. For example, generating and controlling CPA-laser points and factoring in the associated stability issues may help improve the CPA-laser-based electronic sensors $^{[46]}$ and help look for new physics in more complex systems. $^{[66]}$ Furthermore, the unidirectional spectral singularity associated with the PT-symmetric circuit offers new insights into exceptional point physics and could facilitate robust, unidirectional transport of information through attenuated channels.

## 7. Conclusion

Here we have explored and demonstrated the topological nature of S-matrix singularities and their connection with BICs in two-port structures. Using a basic electric circuit, the trajectory and annihilation of CPA charges were experimentally demonstrated, with excellent agreement with theoretical predictions. Furthermore, the analysis of PT-symmetric circuits shows that topological CPA-laser charges emerge from BICs for PT-symmetric perturbations, revealing an important connection between topology, BICs, and PT-symmetry. Using the proposed topological concepts, charge addition was also demonstrated, where two CPAs form an EP of zeros instead of annihilating each other. Due to the peculiarity of the CPA EP, a unidirectional absorber and unidirectional spectral singularity were also shown. The novel outlook on BICs and related singularities enables a simple procedure to leverage their topological properties, which can be applied to design singularity-based devices. Similar to the documented success of the BIC-related polarization topological charges, we believe that our framework meaningfully extends the topological charge picture to phase singularities emerging around BICs. In future work, we expect this viewpoint to be extended and enriched by combining non-Hermiticity with symmetry breaking and/or varying geometrical parameters. Although demonstrated with electric circuits, our findings relate to materials and systems across the electromagnetic spectrum, as all real materials and circuits are non-Hermitian in practice. We expect them to impact thermal emission, wireless power and information transfer, sensing, and polarization control.

## 8. Experimental Section

For the experimental circuit realization, standard RLC components were soldered onto a circuit protoboard. Surface mount (SMD) 0805 capacitors and resistors were used, and value axial inductors were fixed. For the variable resistor, BI-9549 10 Ohm Multiturn trimmer was used. Input/output ports were created by mounting two SMA connectors on the edges of the circuit board. Circuits were measured in the range 5–30 MHz with Vector Network Analyzer—AGILENT E5071C—, and full scattering-matrix parameters were extracted from two-port measurements. In situ CPA measurements were performed using the signal generator RIGOL DG4062, whose two controllable channels were used as two coherent input signals with variable phase difference. The output voltage was measured with the oscilloscope RHODE and SCHWARZ RTC1002.

## Supporting Information

Supporting Information is available from the Wiley Online Library or from the author.

## Acknowledgements

The work described in this paper is conducted within the project NOCTURNO, which receives funding from the European Union's Horizon 2020 research and innovation programme under Grant No. 777714. This work is also supported through the ANTARES project that has received funding from the European Union's Horizon 2020 research and innovation programme under grant agreement SGA-CSA No. 739570 under FPA No. 664387, https://doi.org/10.3030/739570. The work was also supported by the Department of Defense, the Air Force Office of Scientific Research, the National Science Foundation, and the Simons Foundation.

## Conflict of Interest

The authors declare no conflict of interest.

## Author Contributions

Z.S. conceived the idea for the work and performed the analytical and numerical analysis. Z.S. and P.S conducted the experimental part. Z.S. wrote the manuscript. A.K, N.J., V.B. and A.A. contributed to the manuscript and supervised the project.

## Data Availability Statement

The data that support the findings of this study are available from the corresponding author upon reasonable request.

## Keywords

bound states in the continuum, coherent perfect absorption, PT-symmetry, S-matrix singularities, topological photonics

Received: May 2, 2022
Revised: February 17, 2023
Published online:

[1] M. Z. Hasan, C. L. Kane, Rev. Mod. Phys. 2010, 82, 3045.

[2] B. Q. Lv, H. M. Weng, B. B. Fu, X. P. Wang, H. Miao, J. Ma, P. Richard, X. C. Huang, L. X. Zhao, G. F. Chen, Z. Fang, X. Dai, T. Qian, H. Ding, Phys. Rev. X 2015, 5, 031013.

[3] T. Ozawa, H. M. Price, A. Amo, N. Goldman, M. Hafezi, L. Lu, M. C. Rechtsman, D. Schuster, J. Simon, O. Zilberberg, I. Carusotto, Rev. Mod. Phys. 2019, 91, 15006.

[4] C. L. Kane, E. J. Mele, Phys. Rev. Lett. 2005, 95, 146802.

[5] Z. Wang, Y. Chong, J. D. Joannopoulos, M. Soljačić, Nature 2009, 461, 772.

[6] M. A. Bandres, S. Wittek, G. Harari, M. Parto, J. Ren, M. Segev, D. N. Christodoulides, M. Khajavikhan, Science 2018, 359, eaar4005.

[7] C. H. A. O. Peng, Photon. Res. 2020, 8, B25.

[8] W. Liu, W. Liu, L. Shi, Y. Kivshar, Nanophotonics 2021, 10, 1469.

[9] B. Zhen, C. W. Hsu, L. Lu, A. D. Stone, M. Soljačić, Phys. Rev. Lett. 2014, 113, 257401.

[10] W. Liu, B. Wang, Y. Zhang, J. Wang, M. Zhao, F. Guan, X. Liu, L. Shi, J. Zi, Phys. Rev. Lett. 2019, 123, 116104.

[11] S. Kim, B. H. Woo, S. An, Y. Lim, I. C. Seo, D. Kim, S. Yoo, Q. Park, Y. C. Jun, Nano Lett. 2021, 21, 10076.

[12] J. Jin, X. Yin, L. Ni, M. Soljačić, B. Zhen, C. Peng, Nature 2019, 574, 501.

[13] X. Yin, J. Jin, M. Soljačić, C. Peng, B. Zhen, Nature 2020, 580, 467.

[14] C. Huang, C. Zhang, S. Xiao, Y. Wang, Y. Fan, Y. Liu, N. Zhang, G. Qu, H. Ji, J. Han, L. Ge, Y. Kivshar, Q. Song, Science 2020, 367, 1018.

[15] B. Wang, W. Liu, M. Zhao, J. Wang, Y. Zhang, A. Chen, F. Guan, X. Liu, L. Shi, J. Zi, Nat. Photonics 2020, 14, 623.

[16] Y. Guo, M. Xiao, S. Fan, Phys. Rev. Lett. 2017, 119, 167401.

[17] Y. Guo, M. Xiao, Y. Zhou, S. Fan, Adv. Opt. Mat. 2019, 7, 1801453.

[18] Z. Sakotic, A. Krasnok, A. Alú, N. Jankovic, Photon. Res. 2021, 9, 1310.

[19] L. Huang, B. Jia, Y. K. Chiang, S. Huang, C. Shen, F. Deng, T. Yang, Adv. Science 2022, 2200257.

[20] M. Liu, C. Zhao, Y. Zeng, Y. Chen, C. Zhao, C. Qiu, Phys. Rev. Lett. 2021, 127, 266101.

[21] M. Liu, S. Xia, W. Wan, J. Qin, H. Li, C. Zhao, L. Bi, C. W. Qiu, arXiv preprint arXiv:2203.04488v1 2022.

[22] A. Berkhout, A. F. Koenderink, 2019, 6, 2917.

[23] Q. Song, M. Odeh, J. Zúñiga-Pérez, B. Kanté, P. Genevet, Science 2021, 373, 1133.

[24] K. V. Sreekanth, S. Sreejith, S. Han, A. Mishra, X. Chen, H. Sun, C. T. Lim, R. Singh, Nat. Commun. 2018, 9, 369.

[25] G. Ermolaev, K. Voronin, D. G. Baranov, V. Kravets, G. Tselikov, Y. Stebunov, D. Yakubovsky, S. Novikov, A. Vyshnevyy, A. Mazitov, I. Kruglov, S. Zhukov, R. Romanov, A. M. Markeev, A. Arsenin, K. S. Novoselov, A. N. Grigorenko, V. Volkov, Nat. Commun. 2021, 13, 2049.

[26] Y. D. Chong, L. Ge, H. Cao, A. D. Stone, Phys. Rev. Lett. 2010, 105, 053901.

[27] W. Wan, Y. Chong, L. Ge, H. Noh, A. D. Stone, H. Cao, Science 2011, 331, 889.

[28] J. Zhang, K. F. MacDonald, N. I. Zheludev, Light Sci. Appl. 2012, 1, e18.

[29] D. G. Baranov, A. Krasnok, T. Shegai, A. Alù, Y. Chong, Nat. Rev. Mater. 2017, 2, 17064.

[30] M. A. Miri, A. Alù, Science 2019, 363, eaar7709.

[31] S. Longhi, Phys. Rev. A 2010, 82, 031801.

[32] Y. D. Chong, L. Ge, A. D. Stone, Phys. Rev. Lett. 2011, 106, 093902.

[33] Z. Lin, H. Ramezani, T. Eichelkraut, T. Kottos, H. Cao, D. N. Christodoulides, Phys. Rev. Lett. 2011, 106, 213901.

[34] H. Ramezani, H. K. Li, Y. Wang, X. Zhang, Phys. Rev. Lett. 2014, 113, 263905.

[35] D. G. Baranov, A. Krasnok, A. Alù, Optica 2017, 4, 1457.

[36] Y. Sun, W. Tan, H. Q. Li, J. Li, H. Chen, Phys. Rev. Lett. 2014, 112, 143903.

[37] X. Yang, J. Li, Y. Ding, M. Xu, X. - F. Zhu, J. Zhu, Phys. Rev. Lett. 2022, 128, 65701.

[38] W. R. Sweeney, C. W. Hsu, S. Rotter, A. D. Stone, Phys. Rev. Lett. 2019, 122, 93901.

[39] C. Wang, W. R. Sweeney, A. D. Stone, L. Yang, Science 2021, 373, 1261.

[40] A. S. B. B. Dhia, L. Chesnel, V. Pagneux, Proc. R. Soc. A 2018, 474, 20180050.

[41] W. R. Sweeney, C. W. Hsu, A. D. Stone, Phys. Rev. A 2020, 102, 063511.

[42] C. Ferise, P. Del Hougne, S. Félix, V. Pagneux, M. Davy, Phys. Rev. Lett. 2022, 128, 203904.

[43] S. Wei, Nature 2019, 548, 7666.

[44] Z. Xiao, H. Li, T. Kottos, A. Alù, Phys. Rev. Lett. 2019, 123, 213901.

[45] J. Wiersig, Photon. Res. 2020, 8, 1457.

[46] M. Farhat, M. Yang, Z. Ye, P. Chen, ACS Photonics 2020, 7, 2080.

[47] H. Jing, S. K. Özdemir, X. Lü, J. Zhang, L. Yang, F. Nori, Phys. Rev. Lett. 2014, 113, 053604.

[48] L. Zhang, Y. Yang, Z. Jiang, Q. Chen, Q. Yan, Z. Wu, Sci. Bull. 2021, 66, 974.

[49] M. Song, P. Jayathurathnage, E. Zanganeh, M. Krasikova, P. Smirnov, P. Belov, P. Kapitanova, C. Simovski, S. Tretyakov, A. Krasnok, Nat. Electron. 2021, 4, 707.

[50] Y. Ra'di, A. Krasnok, A. Alu, ACS Photonics 2020, 7, 1468.

[51] J. Sol, D. R. Smith, P. Del Hougne, Nat. Commun. 2022, 13, 1713.

[52] W. Cao, C. Wang, W. Chen, S. Hu, H. Wang, L. Yang, X. Zhang, Nat. Nanotechnol. 2022, 17, 262.

[53] G. Trainiti, Y. Ra'di, M. Ruzzene, A. Alù, Sci. Adv. 2019, 5, eaaw3255.

[54] Y. Liu, Q. Liu, S. Wang, Z. Chen, M. A. Sillanpää, T. Li, Phys. Rev. Lett. 2021, 127, 273603.

[55] M. F. Imani, D. R. Smith, P. del Hougne, Adv. Funct. Mater. 2020, 30, 2005310.

[56] B. W. Frazier, T. M. Antonsen Jr., S. M. Anlage, E. Ott, Phys. Rev. Res. 2020, 2, 043422.

[57] P. del Hougne, K. B. Yeo, P. Besnier, M. Davy, Laser Photonics Rev. 2021, 15, 2000471.

[58] Z. Fusco, M. Taheri, R. Bo, T. Tran-Ohu, H. Chen, X. Guo, Y. Zhu, T. Tsuzuki, T. P. White, A. Tricoli, Nano Lett. 2020, 20, 3970.

[59] M. Decker, I. Staude, M. Falkner, J. Dominguez, D. N. Neshev, I. Brener, T. Pertsch, Y. S. Kivshar, Adv. Opt. Mater. 2015, 3, 813.

[60] Z. Sakotic, A. Krasnok, N. Cselyuszka, N. Jankovic, A. Alú, Phys. Rev. Appl. 2020, 13, 064073.

[61] R. Duggan, Y. Ra'di, A. Alu, ACS Photonics 2019, 6, 2949.

[62] A. Krasnok, A. Alù, J. Opt. 2017, 20, 064002.

[63] L. S. Li, H. Yin, Sci. Rep. 2016, 6, 1.

[64] V. Chistyakov, A. Krasnok, Fifteenth Int. Congress on Artificial Materials for Novel Wave Phenomena (Metamaterials), IEEE, NYC, NY, USA, 2021.

[65] A. Krasnok, D. Baranov, H. Li, M. - A. Miri, F. Monticone, A. Alú, Adv. Opt. Photonics 2019, 11, 892.

[66] Q. Song, J. Hu, S. Dai, C. Zheng, D. Han, J. Zi, Z. Q. Zhang, C. T. Chan, Sci. Adv. 2020, 6, eabc1160.

[67] Q. Song, S. Dai, D. Han, Z. Q. Zhang, C. T. Chan, J. Zi, Chinese Phys. Lett. 2021, 38, 084203.

[68] S. A. Mousavi, E. Plum, J. Shi, N. I. Zheludev, Sci. Rep. 2015, 5, 8977.

[69] M. Kang, Z. Zhang, T. Wu, X. Zhang, Q. Xu, A. Krasnok, A. Alù, Nat. Commun. 2022, 13, 4536.

[70] H. Zhang, H. Kang, M. Zhang, X. Guo, W. C. Lv, Y. Li, W. Zhang, J. Han, Adv. Mat. 2017, 29, 1604252.

[71] Z. Zhang, M. Kang, X. Zhang, X. Feng, Y. Xu, X. Chen, H. Zhang, Q. Xu, Z. Tian, W. Zhang, A. Krasnok, J. Han, A. Alù, Adv. Mat. 2020, 32, 2002341.

[72] G. Pirruccio, M. Ramezani, S. R. K. Rodriguez, J. G. Rivas, Phys. Rev. Lett. 2016, 116, 103002.

[73] J. Schindler, A. Li, M. C. Zheng, F. M. Ellis, T. Kottos, Phys. Rev. A 2011, 84, 040101.

[74] M. Sakhdari, M. Hajizadegan, Q. Zhong, D. N. Christodoulides, R. El-Ganainy, P. Y. Chen, Phys. Rev. Lett. 2019, 123, 193901.

[75] A. Stegmaier, S. Imhof, T. Helbig, T. Hofmann, C. H. Lee, M. Kremer, A. Fritzsche, T. Feichtner, S. Klembt, S. Höfling, I. Boettcher, I. C. Fulga, L. Ma, O. G. Schmidt, M. Greiter, T. Kiessling, A. Szameit, R. Thomale, Phys. Rev. Lett. 2021, 126, 215302.

[76] M. Yang, Z. Ye, M. Farhat, P. Y. Chen, Phys. Rev. Appl. 2021, 15, 014026.

[77] Z. Xiao, Y. Ra'di, S. Tretyakov, A. Alù, Research 2019, 7108494.

[78] J. Song, F. Yang, Z. Guo, X. Wu, K. Zhu, J. Jiang, Y. Sun, Y. Li, H. Jiang, H. Chen, Phys. Rev. Appl. 2021, 15, 014009.

[79] S. A. Dereshgi, T. G. Folland, A. A. Murthy, X. Song, I. Tanriover, V. P. Dravid, J. D. Caldwell, K. Aydin, Nat. Commun. 2020, 11, 5771.
