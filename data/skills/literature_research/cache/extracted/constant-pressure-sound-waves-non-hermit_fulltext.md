# Constant-pressure sound waves in non-Hermitian disordered media

Etienne Rivet $^{1}$ , Andre Brandstötter $^{2}$ , Konstantinos G. Makris $^{3}$ , Hervé Lissek $^{1}$ , Stefan Rotter $^{2}$

and Romain Fleury $^{4,*}$

$^{1}$ Signal Processing Laboratory 2, EPFL, 1015 Lausanne, Switzerland

$^{2}$ Institute for Theoretical Physics, Vienna University of Technology (TU Wien), Vienna 1040, Austria, EU

$^{3}$ Department of Physics, University of Crete, 71003 Heraklion, Greece, EU

$^{4}$ Laboratory of Wave Engineering, EPFL, 1015 Lausanne, Switzerland

\*To whom correspondence should be addressed. Email: romain.fleury@epfl.ch

When waves impinge on a disordered material they are back-scattered and form a highly complex interference pattern $^{1-3}$ . Suppressing any such distortions of a wave's free propagation is a challenging task with many applications in a number of different disciplines $^{4-11}$ . In a recent theoretical proposal, it was pointed out that both perfect transmission through disorder as well as a complete suppression of any variation in a wave's intensity can be achieved by adding a continuous gain-loss distribution to the disorder $^{12-14}$ . Here we show that this abstract concept can be implemented in a realistic acoustic system. Our prototype consists of an acoustic waveguide containing several inclusions that scatter the incoming wave in a passive configuration and provide the gain or loss when being actively controlled. Our measurements on this non-Hermitian acoustic metamaterial demonstrate unambiguously the creation of a reflectionless scattering wave state that features a unique form of discrete constant-amplitude pressure waves. In addition to demonstrating that gain-loss additions can turn localised systems into transparent ones, we expect our proof-of-principle demonstration to trigger interesting new developments not only in sound engineering, but also in other related fields such as in non-Hermitian photonics.

The disordered structure of a material renders many of the objects in our everyday lives opaque (we cannot see through them) or impenetrable for sound (that is our hearing is impaired when we plug them in our ears). In contrast to crystalline structures that have well-defined transmission bands and corresponding gaps between them, a disordered medium leads to multiple scattering that typically prevents wave transmission in very broadband spectral intervals. This aspect constitutes, in fact, a central challenge for many active research areas such as for biomedical imaging $^{4}$ , metamaterial design $^{5,11}$ , wireless data transfer $^{6}$ , and wave control $^{8-10}$ . The strategies available to cope with the adverse effects of disorder scattering in these different domains typically involve either shaping the wavefront that impinges on the disorder $^{4,6,8-10}$ , redesigning the internal structure of the disordered medium $^{7,11}$ , exploiting topological protection $^{15}$ , or cloaking the medium altogether $^{5}$ . A novel and potentially very useful approach to achieve novel functionalities of a system is to imprint on it a suitable distribution of gain and loss in the context of non-Hermitian wave physics $^{16-38}$ , a research area initially driven by theoretical and experimental studies of non-Hermitian photonic systems $^{20-23}$ that respect Parity-Time symmetry $^{16-17}$ . Theoretical calculations $^{12-14}$ show that, in this way, so-called “constant-intensity (CI) waves” can be generated that perfectly penetrate even through disordered media without any backscattering and intensity variations throughout the entire transmission process. In other words, such CI waves have the very unconventional feature that they can propagate even through strongly disordered structures like a plane wave through free space. The only signature they carry from the inhomogeneous medium that they penetrate is in the phase they accumulate, but not in their intensity.

As we demonstrate here, these exotic wave states do not just exist for light, as originally predicted $^{13}$ , but also for airborne sound waves. This property opens up the exciting possibility to realise CI waves in an acoustic wave system, for which, contrary to our everyday experience, sound propagates through a non-uniform medium without any variations in its pressure amplitude. To illustrate this conceptually, we firstly consider a finite one-dimensional scattering geometry of length 2L embedded in a uniform background acoustic medium. In the scattering region between -L and L, the bulk modulus $\kappa(x)$ and the complex mass density are position-dependent such as

$$
\rho (x) = \rho_ {R} (x) + i \rho_ {I} (x).\tag{1}
$$

The acoustic pressure $p(x)$ follows the acoustic Helmholtz equation

$$
[ \Delta + \omega^ {2} \rho (x) \kappa^ {- 1} (x) ] p (x) = 0,\tag{2}
$$

where $\Delta = d^{2}/dx^{2}$ is the Laplacian in 1D and $\omega = 2\pi f$ is the angular frequency with f being the frequency. We now make an ansatz for an acoustic wave $p(x)\exp(i\omega t)$ traveling only in positive x-direction with constant pressure and a position-dependent phase between -L and L

$$
p (x) = A e ^ {i k \int_ {- L} ^ {x} W \left(x ^ {\prime}\right) d x ^ {\prime}},\tag{3}
$$

where A is the constant (spatially independent) amplitude of the pressure, k is the wavenumber, and $W(x)$ is a real auxiliary function, which can be chosen arbitrarily. Inserting Eq. (3) into Eq. (2) provides us with a design principle for the real and imaginary parts of the mass density,

$$
\rho_ {R} (x) = \frac {\kappa (x) k ^ {2} W ^ {2} (x)}{\omega^ {2}}\tag{4}
$$

$$
\rho_ {I} (x) = - \frac {\kappa (x) k W ^ {'} (x)}{\omega^ {2}}\tag{5}
$$

with $W' = dW(x)/dx$ . Note that the wavenumber k appearing in $\rho_{R}(x)$ and $\rho_{I}(x)$ in Eqs. (4,5) is the same as in the ansatz in Eq. (3), meaning that only waves with this specific wavenumber impinging from the left-hand side on this structure lead to a constant pressure wave also inside the structure as illustrated in Fig. 1. This approach can easily be generalized so as to also consider complex distributions of $\kappa(x)$ .

For the trivial case that both the auxiliary function $W(x) = 1$ and the bulk modulus $\kappa(x) = \kappa$ are independent of position, we get $\rho_{I}(x) = 0$ and $\rho_{R}(x) = \kappa k^{2} / \omega^{2}$ . Using $\omega = kv$ , we get $v = \sqrt{\kappa / \rho}$ which is the well-known expression for the speed of sound in a uniform material. In this sense, our approach generalises the trivial plane wave solution $p(x) = A \exp(ikx)$ to the non-trivial case of an inhomogeneous medium, where acoustic gain and loss $[\rho_{I}(x) \neq 0]$ are required to maintain a spatially constant pressure wave.

A major obstacle to the practical implementation of this elegant theoretical concept is the challenge of building continuous gain-loss distributions in realistic systems. To bring this abstract idea to life, we extend the notion of constant-intensity scattering states to systems involving discontinuous and discrete distributions of gain and loss. Consider for this purpose a one-dimensional acoustic metamaterial composed of an air-filled tube, loaded with a set of discrete acoustic inclusions modelled by acoustic impedances $Z_{j}$ . At low frequencies, where only a single mode can be excited, the system can be described by a transmission-line (TL) model, as illustrated in Fig. 2a. According to acoustic TL theory, the line voltage represents the complex acoustic pressure p, and the line current represents the volume flow $q^{39}$ . The black boxes represent the acoustic inclusions, introducing some discontinuities of the acoustic pressure along the line, while preserving the volume flow. The grey subsystems correspond to arbitrary transfer matrices $M_{j} = \left[A_{j}, B_{j}; C_{j}, D_{j}\right]$ that connect these impedances. It is important to note that in this description, non-Hermiticity is represented by the non-zero real part of the impedance $Z_{j}$ (describing positive or negative value of resistance), whereas the imaginary part corresponds to the Hermitian component (describing capacitive or inductive behaviour). Indeed, the imaginary and real parts of these impedances correspond to local modifications of the real and imaginary parts of the acoustic density, respectively $^{35}$ . We assume that the finite system with altogether n inclusions is connected to two semi-infinite half-spaces described by characteristic impedances $Z_{L}$ (left) and $Z_{R}$ (right).

As we demonstrate here explicitly, such an acoustic system can support a powerful, discrete version of the constant-pressure sound wave. As in the continuous case, we start by considering an incoming wave with unit amplitude from the left and assuming that the acoustic pressures $p_{j}$ only differ in their phases, but not in their amplitudes, enforcing $p_{j+1} = e^{ik\varphi_{j}}p_{j}$ at the sites $j \in [1, n]$ . In Methods, we show that this assumption forces the volume flows $q_{j}$ to take the values

$$
q _ {j} = \frac {1}{\Pi_ {l = 1} ^ {j - 1} D _ {l}} q _ {1} - \sum_ {l = 1} ^ {j - 1} \frac {C _ {l}}{\Pi_ {m = l} ^ {j - 1} D _ {m}} p _ {l + 1}.\tag{6}
$$

The required acoustic impedance at the site j is then found to be

$$
Z _ {j} = \frac {p _ {j} - A _ {j} p _ {j + 1} - B _ {j} q _ {j + 1}}{C _ {j} p _ {j + 1} + D _ {j} q _ {j + 1}}.\tag{7}
$$

Equation (7) directly implies that in order to obtain a constant-pressure sound wave at every site $j \in [1, n + 1]$ , the real parts of the corresponding acoustic impedances $Z_{j}$ need to be non-zero in general. Therefore, the non-Hermiticity of a system is, indeed, a basic requirement for the realization of non-trivial constant-amplitude waves.

To illustrate the unique properties of these wave states, we consider the example of a disordered acoustic system that is initially Hermitian and terminated by $Z_{L} = Z_{R} =$

$Z_{0}$ (altogether $n = 8$ inclusions are incorporated). We also assume that they are connected by identical air-filled tube portions whose transfer matrices are defined by $A_{j} = D_{j} = \cos kd$ , $B_{j} = iZ_{0} \sin kd$ , $C = iZ_{0}^{-1} \sin kd$ , where $Z_{0}$ is the characteristic acoustic impedance in the tube. In this case, the corresponding acoustic impedances $Z_{j}$ are first assumed to be purely imaginary, corresponding to lossless elements that are either pure masses or springs. Their impedance values are randomly chosen from a uniform distribution. Figure 2b illustrates the impedance at each inclusion by a point on the complex plane, whose colour is associated with the considered site. Upon unitary excitation from the left, the acoustic pressure illustrated in Fig. 2c is strongly non-uniform at the sites $[[1, n + 1]]$ , and its amplitude is very small at the output (yellow point). This case illustrates how disorder usually prevents efficient wave transmission.

We compare this situation with the case of a non-Hermitian system, whose acoustic impedances $Z_{j}$ have the same imaginary parts as in the previous case, but now their real parts are non-zero, following Eq. (7). As shown in Fig. 2d, and comparing with Fig. 2b, each point has now moved away from the imaginary axis along the horizontal direction, which corresponds to the punctual addition of loss or gain to the system. As can be seen in Fig. 2e, these non-Hermitian modifications have a significant impact on the distribution of the acoustic pressures within the system. Remarkably, all the acoustic pressures at the sites $[1, n + 1]$ now stick to the unit circle, such that they have the same amplitude as the incident wave, and different phases that exactly match the values for the phases $\varphi_{j}$ chosen by design. As a result, the pressure wave gets perfectly transmitted.

In order to better grasp the microscopic field dynamics related to this extraordinary phenomenon, including the behaviour of the pressure in between the non-Hermitian inclusions, we simulated a 3D air-filled cylindrical pipe loaded with eight transverse membranes (n = 8), whose corresponding acoustic impedances are set by applying appropriate internal impedance boundary conditions. Figure 3a shows the real part of the acoustic pressure (at a given moment of time; black line) obtained for excitation from the left for the case of the purely Hermitian system with a large degree of disorder (values given in Fig. 2b). All the sites illustrated in colour are lossless and correspond to purely reactive but mismatched acoustic impedances (with $Z_{n+1} = 0$ , consistent with our convention in Fig. 2a). The grey area represents the values spanned by the pressure at each point over a full period of the harmonic field. As a result of the large disorder present in the system, the acoustic wave is mostly reflected and the absolute value of the pressure decreases rapidly along the system. In a next step this situation is compared to the corresponding non-Hermitian system designed to support a constant-pressure wave. In Fig. 3b, the eight inclusions are tuned to provide the right amount of gain and loss as calculated by our theory (consistent with Fig. 2d). Remarkably, the acoustic pressure now propagates through the system and the wave is perfectly transmitted at site $n + 1$ . In addition, we notice that the grey envelope meets the unity dashed red line right in front of every membrane inclusion corresponding to the location of every pressure site $p_j$ . In contrast to continuous constant-amplitude waves, the discrete design adopted here forces the pressure amplitude to be unitary at each check point $p_j$ , but not in between. Notice, however, that this is not a severe limitation of the concept, since the distance between neighbouring sites $p_j$ can be chosen arbitrarily small and, when being subwavelength, prevents large amplitude fluctuations in between the sites. In addition, since the pressure amplitude at the last site $p_{n+1}$ is fully controlled, despite the fact that no inclusion is present at that location, perfect transmission can be guaranteed by design. We conclude that the present discretisation of the constant amplitude property does not introduce any fundamental limitation or change in functionality when compared to the continuous case, while it brings a significant simplification on the required gain-loss distribution. Our findings therefore suggest that a proper discrete distribution of gain and loss offers sufficient control over the scattering state in a strongly disordered system, thanks to local injection and absorption of energy within the material.

So far, we have only demonstrated a special case where the membrane inclusions were at the same time the source of Hermitian disorder and of the required gain and loss. However, our theory is more general and can accommodate any type of disorder placed at any location, including away from the pressure sites. Indeed, the transfer matrices $M_{j}$ , so far considered constant and identical at all sites, can in general be different and describe the presence of scattering defects of any type. To prove this, we consider the full-wave scattering scenario of Figure 4a. An acoustic pipe is filled with defects of various kinds, including pipe section discontinuities, hard-wall obstacles, porous walls, labyrinthine detours, and even a strongly resonant side Helmholtz resonator. These defects, by themselves, completely prevent efficient wave transmission through the system, as shown in Fig. 4a. Yet, knowing their transfer matrices $M_{j}$ , one can use our theory [Eq. (7)] to determine proper values of gain and loss to add between these defects, in order to turn this opaque system into a transparent one (Fig. 4b). From the figure it is clear that at each pressure site $p_{1}$ to $p_{7}$ the amplitude of the pressure has been restored to unity, providing an interesting way of steering sound seamlessly through an arbitrary combination of obstacles by engineering a special scattering state with discrete constant amplitude.

Based on our discrete theory, we built a non-Hermitian metamaterial prototype as shown in Fig. 5. This 2.75 m-long acoustic metamaterial consists of eight identical loudspeakers placed between identical cylindrical acoustic waveguide sections of length d. In the low-frequency approximation, the electrodynamic loudspeaker can be modelled as a one-degree-of-freedom oscillator (that is a mass-spring-damper system) mechanically driven by a voice coil within a magnetic field. When an appropriate electrical circuit is connected to the transducer terminals, it is possible to control the dynamic mechanical behaviour of the diaphragm $^{40}$ . Here, each loudspeaker is connected to a control unit designed to assign a target acoustic impedance [given in Eq. (7)] to the diaphragm at the design frequency. This control unit regulates the diaphragm velocity of the current-driven loudspeaker at site j by measuring the front and rear sides acoustic pressures $p_{fj}$ and $p_{rj}$ , and returning the proper electrical current according to the target complex gains (see Methods). A sound source is located at the left side of the system and is connected to a tube of cross-section area S. A tube of the same cross-section area is connected at the right side of the system, corresponding to a situation in which $Z_{L} = Z_{R} = Z_{0}$ . Lastly, the rightmost end of the system is terminated by a matched load that guarantees no reflection at the design frequency (121 Hz). Figures 5a and 5b show a photograph and the schematic of the complete set-up, respectively.

Figure 6 reports our measurements (coloured data), comparing it with theory (black data). First, we considered a reference Hermitian system in which the imaginary parts of acoustic impedances where chosen in the interval $[-1.8\ Z_{0}, 0]$ (negative values of reactance are more easily obtained from the loudspeakers at the frequency of operation, due to the bulk of their driver). Statistical considerations based on ensemble averages show that such a system operates in the regime of Anderson localisation, with a localisation length just below one meter, making the total prototype about three times the localisation length (see Methods). The realisation of disorder considered in the experiment is shown in Fig. 6b. The acoustic pressures that were measured directly in front of the inclusions with the help of microphones are displayed in Fig. 6c. Clearly, the localised scattering state has a spatially very inhomogeneous amplitude, and a strongly reduced transmission to the other side (measured at 33%). Conversely, when we turn on the control by adding gain and loss in amounts prescribed by theory (Fig. 6d), the measured pressure amplitudes become identical at each site (Fig. 6e), demonstrating the unique ability of non-Hermitian systems to fight disorder. In this situation, we measured a unitary transmittance (with percent-level accuracy), and near-perfect agreement with theory. The very small discrepancies found did not affect the constant-amplitude property, which was observed to be quite robust to small variations of the impedances. Altogether, our measurements unambiguously demonstrate that a tailor-made distribution of gain and loss opens up a transparency window in a disordered and Anderson-localised Hermitian system in which discrete pressure waves can propagate with constant-amplitude.

In conclusion, we have put forward and experimentally demonstrated the concept of discrete constant-amplitude sound propagation in non-Hermitian systems. Such waves are the generalisation of plane waves to disordered non-Hermitian systems. Our theory of discrete, non-Hermitian acoustic systems with waves of constant pressure is exact, very general, and does not rely on some form of homogenisation procedure or unpractical continuous gain-loss distributions. Our experimental results demonstrate that non-Hermiticity can be used to counteract the detrimental effects of disorder even in Anderson-localised systems. Specifically, we locally control the power generation and absorption at the microscopic scale and thereby engineer the acoustic pressure to have desired values at discrete check points. Our concept offers interesting technological perspectives by enriching the toolkit of acousticians with a solution for acoustic transmission through multiple-scattering media that are conventionally opaque. In addition, our experiment provides a beautiful tabletop platform offering fascinating insights into the physics of non-Hermitian disordered systems.

## Figures

![](images/917b537f9d2838f686132fb9b1cfd519ce53a29cea340e9422c6670c34bb305c.jpg)
Figure 1: Concept of continuous constant-pressure waves. a) Absolute value of the acoustic pressure $p(x)$ (orange line) in a disordered Hermitian system when a wave is incident from the left-hand side with normalized wavenumber $k = 2\pi / 0.27$ . Due to variations of the mass density $\rho_R(x)$ (grey), the acoustic pressure wave is partially reflected and shows strong variations inside the medium. b) Adding a tailored gain-loss distribution (red/green) according to Eqs. (4,5) to the system shown in a), the acoustic pressure field gets perfectly transmitted and shows no pressure variations inside the entire scattering region.

a
![](images/1b6d5d66f862f17b1457429b7ba32fb3566a74f8e8b9f0aaba7e1f945713cda7.jpg)

b
![](images/e76eb775cbab837f613b41b2e92a7611d13d28d765ed20de5d0cc81af2c1dd2d.jpg)

d
![](images/fac07622e1c785b85973dfe2291f368c851ca4789d732c98b013abefe1cb316f.jpg)

C
![](images/fda6ae385f0d1cf691544074f4225debee3f9d0871b9d9bc4cadc3b2ac3174e2.jpg)

e
![](images/311dbd303593ffc3674ee4107b8a28bb9ab117769335abfdc319fb12537a60df.jpg)
Figure 2: Discrete constant-pressure acoustic waves. a) We consider an acoustic transmission-line of characteristic impedances $Z_{L}$ at the left end and $Z_{R}$ at the right end, loaded with $n$ series impedances $Z_{j}$ . Each imaginary part of $Z_{j}$ is used to introduce Hermitian disorder to the system, whereas the real part implements an additional discrete distribution of loss or gain. The inclusions are connected by transfer matrices $M_{j}$ , which may correspond to additional disorder in general, but are assumed to be identical ( $M_{j} = M_{0}$ ) in panels b-e, with $M_{0}$ representing a simple Hermitian portion of transmission line with impedance $Z_{0}$ and length $d$ . We first consider the particular case of a Hermitian system ( $n = 8$ ) with random but purely imaginary impedances as shown in b). c) The acoustic pressures $p_{j}$ obtained from our analytical model for the case in b) have non-uniform amplitudes. d) By adding loss and gain, we create a non-Hermitian system whose non-Hermitian impedances have the same imaginary parts as those in the case of the Hermitian system b), but their real parts are now different from zero. e) Tailoring these non-Hermitian additions results in an acoustic scattering state with constant-amplitude pressure at each site $p_{j}$ .

![](images/f231cab7643c368c68def1b8221ccd31e526ecd4a0177c37aa6577b8613a7532.jpg)
b

![](images/3ada6ca3c0e9bced2c0f7f11cc4bae707b2af880a0f4cdbffd3c248e430a2f21.jpg)
Figure 3: Microscopic field dynamics in a non-Hermitian metamaterial supporting discrete constant-pressure acoustic waves. We performed full-wave 3D finite-element simulations to investigate the microscopic behaviour of the pressure field in the Hermitian case of Fig. 2b-c (panel a) and the corresponding non-Hermitian case of Fig. 2d-e (panel b). The black line is a snapshot in time of the acoustic pressure field, whereas the shadowed area shows the range of its oscillations as time evolves. The vertical coloured lines represent the locations of the impedances $Z_{j}$ . a) The disordered Hermitian system transmits poorly and the amplitude inside fluctuates strongly. b) Adding gain or loss to each inclusion counteracts the effect of disorder and makes the system transparent. The discrete constant amplitude property implies that the amplitude of the pressure reaches unity at each check point $p_{j}$ , located directly in front of each inclusion (consistent with Fig. 2a). In panel b), the shadowed grey area indeed touches the dashed red unity line right before each impedance.

![](images/50af70cd417b5e92f694d5b5fa84130a58b67f9ee89ebb53f3ed8d68e20598d8.jpg)
Figure 4: Adding gain and loss to fight scattering defects of any type. a) We consider a pipe filled with various strongly scattering obstacles that prevent efficient transmission of sound (colour map shows pressure amplitude). We have introduced pipe section discontinuities, a resonant side Helmholtz resonator, a hard wall obstacle, a labyrinthine path and porous walls. When only the defects are considered, the pressure amplitude is fluctuating, and reaches almost zero on the transmission side. b) After adding the discrete distribution of gain and loss as prescribed by our theory, we obtain perfect sound transmission. The pressure amplitudes are constant at discrete locations right in front of each added non-Hermitian inclusion (controlled sites $p_{1}$ to $p_{7}$ ).

b
![](images/c44a3f908ad1927dc6676a472659f7147b17f219732ee4dd4b7dd424c14c1870.jpg)
Figure 5: Non-Hermitian metamaterial prototype supporting discrete constant-amplitude pressure waves. a) We built a one-dimensional acoustic metamaterial consisting of a 3.5 m-long waveguide loaded with eight non-Hermitian acoustic inclusions (8d = 2.75 m). The inclusions are electrodynamic loudspeakers, whose acoustic impedance can be conveniently tailored electrically through a harmonic impedance control method. b) Schematic of the experimental set-up showing the inclusions with the control, the sound source at the left end, and the anechoic termination at the right end.

![](images/96f7003ae540c247091cbf67e62f71fec5aca9aafb0e334ab09cb4386b245af1.jpg)
Figure 6: Experimental proof of discrete constant-amplitude pressure waves. We performed two experiments: a reference one with a purely Hermitian disordered system (panels b-c), and a second one on the same system with added gain and loss according to the theory (panels d-e). Measured values are shown with colour lines and predicted values with black lines. a) Schematic of the setup, serving as a column reference for the other panels. b) Measured normalized acoustic impedances in the Hermitian case represented in the complex plane. All impedances are imaginary, unambiguously demonstrating the Hermitian origin of the disorder. c) Pressures measured at each site in the Hermitian case. Pressures are represented as phasors, with the radius of the circle being proportional to their amplitude, and the line segment showing their phase with respect to that of the incident pressure. d) Measured impedances in the non-Hermitian case, highlighting the addition of gain or loss to each inclusion. e) Pressures measured at each site in the non-Hermitian case, demonstrating the discrete constant amplitude property in excellent agreement with theoretical predictions.

## Methods

Discrete constant-pressure waves. The situation represented in Fig. 2a can be analysed through the transfer matrix formalism $^{41}$ . The pressure $p_{j}$ and volume flow $q_{j}$ at each inclusion $j \in [1, n]$ are related by:

$$
\binom{p _ {j}}{q _ {j}} = \left( \begin{array}{c c} 1 & Z _ {j} \\ 0 & 1 \end{array} \right) \left( \begin{array}{c c} A _ {j} & B _ {j} \\ C _ {j} & D _ {j} \end{array} \right) \binom{p _ {j + 1}}{q _ {j + 1}},\tag{8}
$$

where $M_{j} = \left[A_{j}, B_{j}; C_{j}, D_{j}\right]$ represents the transfer matrices connecting the impedances $Z_{j}$ and $Z_{j+1}$ . In addition, assuming an incident wave from the left with amplitude $p_{inc} = 1$ Pa, we use the boundary conditions

$$
p _ {n + 1} / q _ {n + 1} = Z _ {R}\tag{9}
$$

$$
p _ {1} / p _ {i n c} = 1 + R\tag{10}
$$

$$
q _ {1} / p _ {i n c} = (1 - R) / Z _ {L},\tag{11}
$$

where R is the reflection coefficient. If we assume that the acoustic pressures have the same amplitude at all the sites, enforcing $p_{j+1} = e^{ik \varphi_j} p_j$ , we have $p_{j+1} = \exp\left( ik \sum_{l=1}^j \varphi_l \right) p_1$ . Then, all the pressures $p_j$ are known as a function of $p_1$ , or equivalently, R [see Eq. (10)]. This choice for the pressures $p_j$ also fixes the corresponding volume flows $q_j$ according to Eq. (8). Indeed, solving the recurrence equation $q_j = C_j p_{j+1} + D_j q_{j+1}$ directly yields Eq. (6). Then, with Eqs. (6) and (11), we also know all the volume flows $q_j$ as a function of the reflection coefficient R. To determine the value of R, we need to enforce the boundary condition at the end of the line given in Eq. (9). We obtain

$$
(1 + R) e ^ {i k \sum_ {l = 1} ^ {n} \varphi_ {l}} = Z _ {R} \left(\frac {1}{\prod_ {l = 1} ^ {n} D _ {l}} q _ {1} - \sum_ {l = 1} ^ {n} \frac {C _ {l}}{\prod_ {m = l} ^ {n} D _ {m}} p _ {l + 1}\right)\tag{12}
$$

which yields, $R = (Z_{in} - Z_{L}) / (Z_{in} + Z_{L})$ where $Z_{in}$ is the input acoustic impedance (at site 1) expressed as

$$
Z _ {i n} = Z _ {R} \frac {1}{\prod_ {l = 1} ^ {n} D _ {l}} \frac {1}{e ^ {i k \sum_ {l = 1} ^ {n} \varphi_ {l}} + Z _ {R} \sum_ {l = 1} ^ {n} \frac {C _ {l}}{\prod_ {m = l} ^ {n} D _ {m}} e ^ {i k \sum_ {r = 1} ^ {l} \varphi_ {r}}}.\tag{13}
$$

Since all the acoustic quantities are completely determined, we can calculate the target acoustic impedances $Z_{j}$ from Eq. (8), directly getting Eq. (7). We see that if we set all the pressure phases $\varphi_{j}$ , then both the acoustic impedances $Z_{j}$ and the reflection coefficient R are fixed. Another possibility is to set R = 0 to obtain a system without any reflection, at the cost of relaxing two degrees of freedom in the choice of the phases. For instance, we can impose the sum of the phases to be constant (in order to be able to set the total phase of the transmission coefficient) and tune two values of phase to enforce the reflection coefficient $R = 0$ . Since the impedance $Z_{j}$ at the inclusion $j$ in Eq. (7) does not depend on the pressure and volume flow of previous sites $[[1, j - 1]]$ on the TL, we used in our numerical simulations the first two phases $\varphi_{1}$ and $\varphi_{2}$ to set $R = 0$ without modifying the rest of the design. Using this theory, we can generate many matched systems supporting constant-amplitude pressure waves, even for arbitrarily large values of $n$ . Note that even in the case $R \neq 0$ , our theory leads to a distribution of gain and loss in the system with constant pressure at every site $j \in [[1, n + 1]]$ .

Harmonic acoustic impedance control. In the low-frequency approximation, the electrodynamic loudspeaker can be modelled as a one-degree-of-freedom oscillator mechanically driven by a voice coil within a magnetic field. All forces acting on the transducer, especially those resulting from the sound pressures $p_{f}$ and $p_{r}$ at the front and rear faces of the diaphragm, are assumed small enough so that the governing equations remain linear. The mechanical part is assumed as a simple mass - spring - damper system in the low-frequency range, that is the mass $M_{m}$ , the mechanical compliance (inverse of stiffness) $C_{m}$ accounting for the surround suspension and spider, and the mechanical resistance $R_{m}$ , respectively. If we denote the effective piston area by $S_{d}$ and the force factor of the moving-coil transducer by Bl, the equation of motion of the loudspeaker diaphragm at inclusion j is derived from Newton's second law, which can be written using the Fourier transform as

$$
S _ {d} P _ {f j} (\omega) - S _ {d} P _ {r j} (\omega) = Z _ {m j} (\omega) V _ {j} (\omega) + B l _ {j} I _ {j} (\omega),\tag{14}
$$

where $Z_{mj}(\omega)=i\omega M_{mj}+R_{mj}+1/(i\omega C_{mj})$ is the mechanical impedance of the loudspeaker, $V_{j}(\omega)=Q_{j}(\omega)/S_{d}$ is the diaphragm velocity, and $I_{j}(\omega)$ is the current flowing through the voice coil of the loudspeaker.

Our control strategy uses two microphones giving the signals $\widehat{P_{fj}}(\omega)=\sigma_{fj}P_{fj}(\omega)$ and $\widehat{P_{rj}}(\omega)=\sigma_{rj}P_{rj}(\omega)$ , where $\sigma_{fj}$ and $\sigma_{rj}$ are the sensitivities of the microphones (in V.Pa $^{-1}$ ) at the front and rear sides of the corresponding loudspeaker diaphragm respectively, and return the current $I_{j}(\omega)$ with transfer functions $\Theta_{j}(\omega)^{40}$ such as

$$
I _ {j} (\omega) = \Theta_ {j} \big (\omega , \sigma_ {f j} \big) \widehat {P _ {f J}} (\omega) - \Theta_ {j} \big (\omega , \sigma_ {r j} \big) \widehat {P _ {r J}} (\omega).\tag{15}
$$

Assuming the target acoustic impedance $Z_{tj}(\omega)$ , which is defined in Eq. (7), is realized at the diaphragm, the transfer function $\Theta_{j}(\omega)$ from the microphone signal $\widehat{P_{fJ}}(\omega)$ or $\widehat{P_{rJ}}(\omega)$ to the electrical current $I_{j}(\omega)$ can be derived from Eqs. (14) and (15) as

$$
\Theta_ {j} (\omega , \sigma) = \frac {S _ {d}}{\sigma B l _ {j}} \bigg (1 - \frac {Z _ {m j} (\omega)}{S _ {d} ^ {2} Z _ {t j} (\omega)} \bigg).\tag{16}
$$

It is worth noting that the signal filtering through the transfer function $\Theta_{j}$ should be processed before differentiating the signals in Eq. (15), because of possible loss of significance in the controller. Note also that the situation $\Theta_{j}(\omega,\sigma)=0$ through the voltage-controlled current source corresponds to the case where the loudspeaker is in open circuit [see Eqs. (14) and (15)].

For the inclusions giving gain to the system, that is $\mathrm{Re}(Z_{tj}) < 0$ at the frequency of interest $f_{0}$ , the transfer function $\Theta_{j}(\omega)$ is in this case not causal and therefore, it cannot be implemented in a digital controller. We thus turn the broadband impedance control $^{38}$ into a harmonic control through the formalism of demodulation/modulation $^{40}$ . First, the microphone signals $\widehat{P_{fJ}}(\omega)$ and $\widehat{P_{rJ}}(\omega)$ are demodulated at the angular frequency of interest $\omega_{0} = 2\pi f_{0}$ (translating both signals in the frequency domain from $\omega_{0}$ to 0), whose low-pass filter of cut-off angular frequency $\omega_{c} \ll \omega_{0}$ is equal to 2 for $|\omega| < \omega_{c}$ and 0 otherwise. These complex signals are then multiplied by the target complex gain $\Theta_{j}\big(\omega_{0}, \sigma_{fj}\big)$ or $\Theta_{j}\big(\omega_{0}, \sigma_{rj}\big)$ . The signals are then modulated at the angular frequency $\omega_{0}$ (translating both signals in the frequency domain from 0 to $\omega_{0}$ ). Lastly, the output signal of spectral range $[\omega_{0} - \omega_{c}, \omega_{0} + \omega_{c}]$ delivered by the voltage-controlled current source corresponds to the difference of both modulated signals as given in Eq. (15). This way, the target acoustic impedance $Z_{tj}$ is perfectly assigned at the loudspeaker diaphragm at the frequency of interest.

Experiment. The experimental setup consisted of eight electrodynamic loudspeakers $n = 8$ , separated from each other by a distance d = 34.3 cm in a tube of cylindrical cross-section area $S = \pi r^{2} = 40.7 \, cm^{2}$ (see Figure 5 and Supplementary Figure 1). At the left termination we placed a sound source (closed-box loudspeaker) that delivered a band-limited sweep sine of bandwidth [20 Hz – 400 Hz] with the help of a Brüel & Kjaer 2706 power amplifier. At the other termination we placed a matched load with the help of another closed-box loudspeaker. A mass was added to the diaphragm and a resistive shunt was connected to the electrical terminals $^{43}$ to decrease the resonance frequency and match the acoustic resistance at the diaphragm to $Z_{0}$ respectively, guaranteeing no reflection at the design frequency (121 Hz). We only used Monacor SPX-30M loudspeakers of effective piston area $S_{d} = \pi r_{d}^{2} = 32 \, cm^{2}$ in the experiment. The first site was located at a distance $2d' = 50 \, cm$ from the sound source and the last site at a distance $d'$ from the matched load. The target complex gain was defined for each controlled loudspeaker to get the exact values of acoustic resistance and reactance for each site given in Eq. (7) at the desired frequency. Eight pairs of $\frac{1}{4}''$ PCB 130D20/130F20 microphones were wall-mounted in front and behind every loudspeaker diaphragm. Each pair of microphones measured the acoustic pressures $p_{fj}$ and $p_{rj}$ at each site j. Three additional microphones were also used in the experiment. One microphone was located at around a distance $d'$ from the sound source, another one at the position of the ninth inclusion, and the last one in front of the matched load. The frequency responses between microphone signals at the different sites were processed through a Brüel & Kjaer Pulse 3160 multichannel analyser, so as to estimate the normalized acoustic impedance at every inclusion and the transmission coefficient. The control was implemented onto a real-time National Instrument CompactRIO-9068 platform supporting field-programmable gate array technology. The voltage signals from the eight pairs of microphones were digitally converted thanks to four analog modules NI 9215. The output signals were then delivered by two analog modules NI 9263 to voltage-controlled current sources that drove the loudspeakers.

Estimation of the localisation length. We have checked numerically that our disordered system is indeed operated in the localised regime, and that its size is about three times the localisation length. To verify this, we have computed the ensemble average of the logarithm of the transmission coefficient $< \ln T >$ over 1000 realisations of Hermitian disorder, for systems composed of an increasing number of sites (hence increasing the total length of the system). Importantly, the disorder strength was set to be the same as in the experiment (i.e. the normalized reactances were drawn randomly in the range $[-1.8,0]$ ). We found that the quantity $< \ln T >$ varies linearly with the number of sites (varying from 4 to 20 unit cells, see supplementary Figure 2), which is the hallmark for 1D Anderson localisation. The inverse slope, which gives the Anderson localisation length, is extracted to be 2.7 inclusions (92.6 cm). Thus, our 2.75 m-long prototype is then comparable to three times the localisation length. Our experiment therefore clearly demonstrates unambiguously that the addition of gain and loss can turn a localised system into a transparent one.

Measurement of acoustic impedance of every inclusion. The bulkiness of each loudspeaker driver in the tube locally alters the pressure and volume flow behind the diaphragm. To take this acoustic behaviour into account in the design, we model the geometry of the loudspeaker using the transfer matrix formalism. As illustrated in Supplementary Figure 1 and according to Eq. (8), we choose $Z_{j} = Z_{mj}/S_{d}^{2}$ and

$$
M _ {j} = \left( \begin{array}{c c} \mathrm{A} _ {\mathrm{j}} & \mathrm{B} _ {\mathrm{j}} \\ \mathrm{C} _ {\mathrm{j}} & \mathrm{D} _ {\mathrm{j}} \end{array} \right) = M _ {p} (Z _ {0}, d _ {1}) M _ {p} (Z _ {2}, d _ {2}) M _ {p} (Z _ {0}, d - d _ {1} - d _ {2}),\tag{17}
$$

where

$$
M _ {p} (Z, x) = \left( \begin{array}{c c} \cos (k x) & i Z \sin (k x) \\ i \sin (k x) / Z & \cos (k x) \end{array} \right)\tag{18}
$$

and $d_{1}=2.6\ cm$ , $d_{2}=2.8\ cm$ , $Z_{2}=S/(S-S_{2})Z_{0}$ with $S_{2}=\pi r_{2}^{2}=37.4\ cm^{2}$ .

First, we consider a preliminary setup consisting of only one inclusion as illustrated in Supplementary Figure 1. Regardless of the terminations of this piece of waveguide, the acoustic impedance at the diaphragm may be estimated either from the acoustic pressures $p_{A}$ , $p_{B}$ , and $p_{C}$ (hereafter denoted as method ‘ABC’) such that

$$
\left(Z _ {j}\right) _ {A B C} = i Z _ {0} \sin (k d) \frac {p _ {B} - p _ {C}}{p _ {A} - \cos (k d) p _ {B}}\tag{19}
$$

or from the acoustic pressures $p_{B}$ , $p_{C}$ , and $p_{D}$ (method 'BCD') such that

$$
\left(Z _ {j}\right) _ {B C D} = B _ {j} \frac {p _ {B} - p _ {C}}{D _ {j} p _ {C} + (B _ {j} C _ {j} - A _ {j} D _ {j}) p _ {D}}.\tag{20}
$$

The acoustic impedance $Z_{j}$ can then be measured from Eqs. (19) and (20) with frequency responses $H_{BA} = \widehat{p_{A}}/\widehat{p_{B}}$ , $H_{BC} = \widehat{p_{C}}/\widehat{p_{B}}$ , and $H_{BD} = \widehat{p_{D}}/\widehat{p_{B}}$ . When we performed these measurements for the preliminary setup, we found that the methods 'ABC' and 'BCD' gave consistent results. These methods were validated with a direct impedance measurement, from the microphone signals $\widehat{p_{B}}$ and $\widehat{p_{C}}$ and a Polytec OFV-5000 laser vibrometer. However, when all the loudspeakers are inserted between the tube sections in the acoustic metamaterial shown in Fig. 5, we can only use method 'BCD' (due to geometrical constraints for the inclusions [[2,8]]). With this method, the acoustic impedance of each inclusion was measured in-situ to estimate the model parameters of each loudspeaker through two measurements with a least squares fitting method. The first measurement was performed with the loudspeaker in open circuit to estimate the three parameters of the mechanical impedance $Z_{mj}$ . The second measurement was performed with the loudspeaker in short circuit to estimate the force factor Bl. Then, every inclusion was measured again to verify that we were able to reach the target values of the non-Hermitian and Hermitian acoustic impedances. In Figs. 6b and 6d, we show at the frequency of interest the measured real and imaginary parts of each normalized acoustic impedance $Z_{j}/Z_{0}$ (colour dots) in the complex plane representation, in the Hermitian and non-Hermitian cases, respectively. These values are compared with the theoretical values (grey dots) that fulfil Eq. (7). The measurements confirm that the normalized acoustic impedances $Z_{j}/Z_{0}$ are very close to the theoretical ones.

## Data availability

The data that support the plots within this paper and other findings of this study are available from the corresponding author upon reasonable request.

## Author contributions

A.B., K.G.M., and S.R. developed the concept and theory of continuous constant amplitude waves. E.R. and R.F. developed the discrete theory of constant amplitude waves and performed the numerical simulations. E.R. formulated the acoustic impedance control theory, developed the control technology used in the experiment, and performed the experiment. H.L. supervised the experimental work. S.R. and R.F. initiated and supervised the project. All authors discussed the results and contributed to writing the manuscript.

## Acknowledgments

The authors would like to thank Prof. Mario Paolone and the Distributed Electrical Systems Laboratory at Ecole polytechnique fédérale de Lausanne for lending us the National Instrument CompactRIO-9068 platform for the experiment.

## References

1. Tourin, A., Fink, M. & Derode, A. Multiple scattering of sound. Waves Random Media 10, R31–R60 (2000).

2. Akkermans, É. & Montambaux, G. Mesoscopic Physics of Electrons and Photons. (Cambridge University Press, 2007).

3. Sheng, P. Introduction to Wave Scattering, Localization and Mesoscopic Phenomena. (Springer, 2006).

4. Mudry, K. M., Plonsey, R. & Bronzino, J. D. Biomedical Imaging. (CRC Press, 2003).

5. Pendry, J. B., Schurig, D. & Smith, D. R. Controlling Electromagnetic Fields. Science 312, 1780–1782 (2006).

6. Iniewski, K. Wireless Technologies: Circuits, Systems, and Devices. (CRC Press, 2007).

7. Florescu, M., Torquato, S. & Steinhardt, P. J. Designer disordered materials with large, complete photonic band gaps. Proc. Natl. Acad. Sci. 106, 20658–20663 (2009).

8. Tyson, R. Principles of Adaptive Optics. (CRC Press, 2010).

9. Mosk, A. P., Lagendijk, A., Lerosey, G. & Fink, M. Controlling waves in space and time for imaging and focusing in complex media. Nat. Photonics 6, 283–292 (2012).

10. Rotter, S. & Gigan, S. Light fields in complex media: Mesoscopic scattering meets wave control. Rev. Mod. Phys. 89, 015005 (2017).

11. Leseur, O., Pierrat, R. & Carminati, R. High-density hyperuniform materials can be transparent. Optica 3, 763–767 (2016).

12. Makris, K. G., Musslimani, Z. H., Christodoulides, D. N. & Rotter, S. Constant intensity waves and their modulation instabilities in non-Hermitian potentials. Nat. Commun. 6, 7257 (2015).

13. Makris, K. G., Brandstötter, A., Ambichl, P., Musslimani, Z. H. & Rotter, S. Wave propagation through disordered media without backscattering and intensity variations. Light Sci. Appl. 6, e17035 (2017).

14. Sebbah, P. Scattering media: A channel of perfect transmission. Nat. Photonics 11, 337 (2017).

15. Lu, L., Joannopoulos, J. D. & Soljačić, M. Topological states in photonic systems. Nat. Phys. 12, 626–629 (2016).

16. Bender, C. M. & Boettcher, S. Real Spectra in Non-Hermitian Hamiltonians Having PT Symmetry. Phys. Rev. Lett. 80, 5243–5246 (1998).

17. Bender, C. M., Brody, D. C. & Jones, H. F. Complex Extension of Quantum Mechanics. Phys. Rev. Lett. 89, (2002).

18. Heiss, W. D. Exceptional points of non-Hermitian operators. J. Phys. Math. Gen. 37, 2455 (2004).

19. Lévai, G. & Znojil, M. Systematic search for PT-symmetric potentials with real energy spectra. J. Phys. Math. Gen. 33, 7165–7180 (2000).

20. El-Ganainy, R., Makris, K. G., Christodoulides, D. N. & Musslimani, Z. H. Theory of coupled optical PT-symmetric structures. Opt. Lett. 32, 2632 (2007).

21. Makris, K. G., El-Ganainy, R., Christodoulides, D. N. & Musslimani, Z. H. Beam Dynamics in PT Symmetric Optical Lattices. Phys. Rev. Lett. 100, 103904 (2008).

22. Guo, A. et al. Observation of PT-Symmetry Breaking in Complex Optical Potentials. Phys. Rev. Lett. 103, 93902 (2009).

23. Rüter, C. E. et al. Observation of parity–time symmetry in optics. Nat. Phys. 6, 192–195 (2010).

24. Lin, Z. et al. Unidirectional Invisibility Induced by PT-Symmetric Periodic Structures. Phys. Rev. Lett. 106, 213901 (2011).

25. Schindler, J., Li, A., Zheng, M. C., Ellis, F. M. & Kottos, T. Experimental study of active LRC circuits with PT symmetries. Phys. Rev. A 84, 040101 (2011).

26. Bittner, S. et al. PT Symmetry and Spontaneous Symmetry Breaking in a Microwave Billiard. Phys. Rev. Lett. 108, 024101 (2012).

27. Regensburger, A. et al. Parity-time synthetic photonic lattices. Nature 488, 167–171 (2012).

28. Feng, L. et al. Experimental demonstration of a unidirectional reflectionless parity-time metamaterial at optical frequencies. Nat. Mater. 12, 108–113 (2013).

29. Konotop, V. V. & Zezyulin, D. A. Families of stationary modes in complex potentials. Opt. Lett. 39, 5535 (2014).

30. Peng, B. et al. Loss-induced suppression and revival of lasing. Science 346, 328-332 (2014).

31. Doppler, J. et al. Dynamically encircling an exceptional point for asymmetric mode switching. Nature 537, 76-79 (2016).

32. Xu, H., Mason, D., Jiang, L. & Harris, J. G. E. Topological energy transfer in an optomechanical system with exceptional points. Nature 537, 80-83 (2016).

33. Assawaworrarit, S., Yu, X. & Fan, S. Robust wireless power transfer using a nonlinear parity–time-symmetric circuit. Nature 546, 387-390 (2017).

34. Zhu, X., Ramezani, H., Shi, C., Zhu, J. & Zhang, X. PT-Symmetric Acoustics. Phys. Rev. X 4, 031042 (2014).

35. Fleury, R., Sounas, D. & Alù, A. An invisible acoustic sensor based on parity-time symmetry. Nat. Commun. 6, 5905 (2015).

36. Shi, C., Dubois, M., Chen, Y., Cheng, L., Ramezani, H., Wang, Y. & Zhang, X. Accessing the exceptional points of Parity-Time symmetric acoustics. Nat. Commun. 7, 11110 (2016)

37. Aurégan, Y. & Pagneux, V. PT-symmetric scattering in flow-duct acoustics. Phys. Rev. Lett. 118, 174301 (2017).

38. El-Ganainy, R., Makris, K.G., Khajavikhan, M., Musslimani, Z.H., Rotter, S. & Christodoulides, D.M. Non-Hermitian physics and PT symmetry. Nat. Phys. 14, 11-19 (2018).

39. Fleury, R. & Alù, A. Extraordinary sound transmission through density-near-zero ultranarrow channels. Phys. Rev. Lett. 111, 55501 (2013).

40. Rivet E., Karkar S., and Lissek H. Broadband Low-Frequency Electroacoustic Absorbers Through Hybrid Sensor-/Shunt-Based Impedance Control. IEEE Transactions on Control Systems Technology 25.1 (2017): 63-72.

41. Pozar, D. M. Microwave Engineering. (Wiley, 2011).

42. Prandoni P. & Vetterli M., Signal processing for communications. (Collection le savoir Suisse, 2008).

43. Lissek H., Boulandet R., and Fleury R. Electroacoustic absorbers: bridging the gap between shunt loudspeakers and active sound absorption." The Journal of the Acoustical Society of America 129.5 (2011): 2968-2978.
