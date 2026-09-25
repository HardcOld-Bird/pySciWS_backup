# Acoustic Amplifying Diode Using Nonreciprocal Willis Coupling

Xinhua Wen $^{1}$ , Heung Kit Yip, $^{1}$ Choonlae Cho, $^{2}$ Jensen Li $^{1,*}$ and Namkyoo Park $^{2,\dagger}$

$^{1}$ Department of Physics, The Hong Kong University of Science and Technology, Kowloon, Hong Kong, China $^{2}$ Photonic Systems Laboratory, Department of Electrical and Computer Engineering, Seoul National University, Seoul 08826, South Korea

(Received 31 October 2022; accepted 6 April 2023; published 28 April 2023)

We propose a concept called acoustic amplifying diode combining signal isolation and amplification in a single device. The signal is exponentially amplified in one incident direction with no reflection and is perfectly absorbed in another. The reflection is eliminated from the device in both directions with impedance matching, preventing backscattering to the signal source. Here, we demonstrate the amplifying diode using an active metamaterial with nonreciprocal Willis coupling. We also discuss the situation with the presence of both reciprocal and nonreciprocal Willis couplings for more flexibility in implementation. The coexistence of both amplifier and perfect absorber in opposite incident directions extends the regime of sound isolation and further enables applications in sensing and communication, in which nonreciprocity can play an important role.

DOI: 10.1103/PhysRevLett.130.176101

Bianisotropy in electromagnetism has been used to obtain various phenomena, including asymmetric absorption, topological protection and power-efficient wavefront shaping $[1-4]$ . Its counterpart in acoustics, Willis coupling, has only been recently realized through acoustic metamaterials $[5-7]$ . Asymmetric or unidirectional zero reflections (UZR), non-Hermitian exceptional points were revealed using these coupling $[7-11]$ . Willis coupling can achieve a significant size using asymmetric metamaterial structures with strong local resonances $[5,6]$ . However, an upper bound for the magnitude of such coupling exists for implementations constrained with passivity and reciprocity $[12-15]$ . Breaking the time-reversal symmetry, with moving fluid $[16,17]$ or active (non-Hermitian) components $[18-22]$ can be used to generate nonreciprocity and to overcome these constraints. Specifically, programmable metamaterials can implement tailor-made impulse responses by connecting microphones and speakers through a digital feedback circuit at each meta atom $[23-27]$ . They gain programmable control of the constitutive parameters, and go beyond the passivity bound of Willis coupling $[21]$ . The constitutive matrix can be symmetric, antisymmetric or even arbitrary, allowing access to the nonreciprocal regime. Furthermore, the direct and flexible specifications of the constitutive parameters with programmability free us from the implementation details. It allows an “inverse design” approach from an effective-medium description to formulate one-way acoustic devices, including nonreciprocal lenses and acoustic diodes $[16]$ . Acoustic diodes were demonstrated using nonlinear sonic crystals with frequency conversion $[28-30]$ and diffraction structures with mode conversion $[31]$ . However, confining to a single channel, regardless of frequency or spatial mode, with significant transmission and zero reflection, will benefit ideal acoustic isolation.

Here, we propose a concept called amplifying diode in combining acoustic isolation and amplification. The signal is exponentially amplified in one direction with zero reflection while totally absorbed in another. We investigate the required constitutive parameters using active and nonreciprocal Willis metamaterials and develop a realization scheme through programmable metamaterials. We also discuss implementations when both reciprocal and nonreciprocal Willis coupling terms are present. The integration of amplifying action into an acoustic diode will go beyond the current regime of sound isolation and further enables applications like ultrasensitive sensing and nonreciprocal communication.

We begin with the conceptual diagram of an amplifying diode in Fig. 1(a). We use t and r for the complex transmission and reflection coefficients with subscript "f" and "b" for forward and backward incidence. For a conventional diode, the forward transmittance is one ( $|t_{f}|^{2}=1$ ) while the backward transmittance is completely suppressed ( $|t_{b}|^{2}=0$ ), exhibiting nonreciprocal wave propagation. For the proposed amplifying diode, we have amplification in forward transmission and zero reflection simultaneously. It has to be achieved by some active Huygens (secondary) sources [32] in the middle to generate scattering waves only in the forward direction while adding in-phase to the incident wave ( $|t_{f}|^{2}>1$ ). For a backward incident wave, the same Huygens sources still generate zero reflection ( $r_{b}=0$ ) while adding out-of-phase to the incident waves for suppression, ideally to a value of zero ( $t_{b}=0$ ). These Huygens sources have to be both active and asymmetric and are linked to the active Willis coupling in an acoustic effective-medium setting in one-dimension.

![](images/c9c59eb690bebd13dfa3948e4e5c927834cad72da95a0a8da60f247229cb3bf6.jpg)

![](images/d7d0d1986abe1f496af0fc5b1686ce23edfaf611a74a2bba6194f2b8f4eac418.jpg)

(c)
![](images/abae28491adc4fcbda50a7013c1767ebbb7fd7a17461cdeed79679a24aa675ed.jpg)

![](images/b56cf4ed405e9903d8463a2415e0747386ba2fd529a7731e5d3535137cc2b8a0.jpg)
FIG. 1. (a) Schematics of the conventional diode (left panel) and the amplifying diode (right panel). A conventional diode offers a unit forward but zero backward transmission amplitude. The amplifying diode offers amplified forward transmission ( $|t_{f}| > 1$ ), zero backward transmission, and zero reflections in both directions. (b) A photograph of a PCB board integrating with two microphones and two speakers. (c) A photograph of the amplifying diode with nine meta atoms in a 1D waveguide. Each meta-atom consists of a microcontroller connecting to the PCB board in (b).

Without losing generality, the 1D metamaterial is along x and has a $2 \times 2$ constitutive matrix $\{\{\beta, i\tau\}, \{i\tau', \rho\}\}$ in its effective medium representation. $\beta$ and $\rho$ are the dimensionless compressibility and density relative to air. Here, $\tau$ and $\tau'$ are the required Willis coupling terms. Under an $e^{-i\omega t}$ time convention, the wave equation can be written as

$$
\partial_ {x} \binom{v}{p} = \frac {i \omega}{c} \left( \begin{array}{c c} \beta & i \tau \\ i \tau^ {\prime} & \rho \end{array} \right) \binom{p}{v},\tag{1}
$$

where p is the pressure, v is the velocity multiplied by the acoustic impedance of the air, and c is the sound speed in the air. We first consider the scattering from only a small section of the medium of thickness L in the air to obtain analytic insight. In this case, the scattering matrix of the metamaterial can be obtained by integrating Eq. (1) with Padé's approximation [33]:

$$
\left( \begin{array}{c c} t _ {f} & r _ {b} \\ r _ {f} & t _ {b} \end{array} \right) \cong \left( \begin{array}{c c} 1 + \frac {i \phi_ {0}}{2} (\beta + \rho + 2 i \tau_ {\mathrm{nr}}) & \frac {i \phi_ {0}}{2} (\beta - \rho - 2 i \tau_ {\mathrm{r}}) \\ \frac {i \phi_ {0}}{2} (\beta - \rho + 2 i \tau_ {\mathrm{r}}) & 1 + \frac {i \phi_ {0}}{2} (\beta + \rho - 2 i \tau_ {\mathrm{nr}}) \end{array} \right),\tag{2}
$$

where the subscript “f” (“b”) indicates the forward (backward) direction, $\phi_{0} = \omega L / c$ is the phase that elapses across the air with the same thickness of the metamaterial (as one meta-atom later). We also decompose the Willis couplings into reciprocal $\tau_{r}$ and nonreciprocal components $\tau_{nr}$ by $\tau = \tau_{r} + \tau_{nr}$ , and $\tau' = -\tau_{r} + \tau_{nr}$ [12]. According to Eq. (2), a nonzero $\tau_{nr}$ offers nonreciprocal transmission with $t_{b} - t_{f} = 2\phi_{0}\tau_{nr}$ . To avoid potential disturbance back to the source from the reflected signal, we set $r_{f} = r_{b} = 0$ to get the impedance matching condition:

$$
\tau_ {\mathrm{r}} = 0, \beta = \rho .\tag{3}
$$

Interestingly, previous studies have shown that the presence of a nonzero reciprocal Willis coupling $(\tau_{\mathrm{r}})$ results in different impedances in forward and backward directions. This passive reciprocal Willis coupling leads to asymmetric reflections and can approach the UZR (where either $r_{f}$ or $r_{b}=0$ ) at an exceptional point in the scattering matrix [7–11]. More precisely, a nonzero $\tau_{r}$ means a nonzero $r_{f}-r_{b}$ (asymmetric reflection) while $\beta\neq\rho$ means a nonzero $r_{f}+r_{b}$ . An impedance-matched metamaterial has thus a purely non-reciprocal Willis coupling $\tau=\tau^{\prime}=\tau_{nr}$ , with transmission coefficients

$$
\begin{array}{l} t _ {f} \cong 1 + i \phi_ {0} (i \tau_ {\mathrm{nr}} + \beta), \\ t _ {b} \cong 1 + i \phi_ {0} (- i \tau_ {\mathrm{nr}} + \beta). \end{array}
$$

Since there are no reflections at each small section, cascading identical copies of the section multiplies the transmission coefficients and turns the scattering coefficients into the following form:

$$
t _ {f} e ^ {\tau_ {\mathrm{nr}} \phi_ {0}} = t _ {b} e ^ {- \tau_ {\mathrm{nr}} \phi_ {0}} = e ^ {i \beta \phi_ {0}}, \quad r _ {f} = r _ {b} = 0.\tag{4}
$$

This gives further flexibility in tuning the amplification factor. By requesting a real $\tau_{nr}$ , the waves are exponentially amplifying and decaying in opposite directions with zero reflection. It approaches the amplifying diode by cascading more atoms.

For implementation, we adopt a metamaterial made of discrete meta atoms, in which each meta atom has to respond to both monopolar and dipolar incoming waves so that the monopolar and dipolar scattering waves behave as the described Huygens source. The metamaterial platform is shown in Fig. 1(c), with nine meta atoms and a period of L = 9 cm in a 1D waveguide. Each meta atom consists of a pair of speakers and microphones interconnected by a microcontroller on an integrated unit [Fig. 1(b)]. It can carry out four different channels of time convolution in connecting the two speakers (labeled as $S_{1}$ and $S_{2}$ in the unit of pressure in the waveguide) to the two microphones (labeled as $D_{1}$ and $D_{2}$ in the unit of pressure) as a matrix multiplication in time harmonics:

$$
\binom{S _ {1}}{S _ {2}} = \left( \begin{array}{c c} Y _ {1 1} & Y _ {1 2} \\ Y _ {2 1} & Y _ {2 2} \end{array} \right) \binom{D _ {1}}{D _ {2}}.\tag{5}
$$

Such a representation is also schematically shown in Fig. 2(a), where the distance between the speakers or the microphones is 4.5 cm (2 $\Delta\phi$ in free-space phase distance). In the following, we set $\beta=\rho=1$ for simplification and $\tau_{nr}=-0.09$ . It is proved that we obtain the scattering coefficients [Eq. (4)] by zero $Y_{11}$ and $Y_{22}$ , together with

![](images/eb7bef8cc691d77db697ed495e75b9004f6650b58b4f0960ccba7f557d7ddfb3.jpg)
(b)

![](images/40d924b1c83a450cbcdad50c47efe06cb2c14d3bc3b2233911ce9137a4ee878e.jpg)

![](images/e3db2c480269e678d602ea3a36b9b2e16295b30317d5caff6f31a0b546850835.jpg)

![](images/6236094a518f584aaac0d1584b719db5669e301616d70bc44ec73ebbcccede02.jpg)
FIG. 2. (a) Schematic representation of the meta atom consisting of two speakers (labeled as $S_{1}$ and $S_{2}$ ) and two microphones ( $D_{1}$ and $D_{2}$ ). The phase distance between two speakers (microphones) is $2\Delta \phi$ . Orange arrows connecting microphone $D_{j}$ to speaker $S_{i}$ represent a time convolution labeled by $Y_{ij}$ . (b) The transmittance ( $T_{f / b} = |t_{f / b}|^{2}$ ) and reflectance ( $R_{f / b} = |r_{f / b}|^{2}$ ) for both forward and backward directions. The symbols (lines) show the experimental (analytical) results. (c) The extracted constitutive matrix from the scattering parameters in (b). The solid (open) symbols denote the real (imaginary) part of the experimental results, and the solid (dashed) lines denote the real (imaginary) part of the analytical results.

$$
\begin{array}{r l} Y _ {1 2} e ^ {- \tau_ {\mathrm{nr}} \phi_ {0} / 2} & = - Y _ {2 1} e ^ {\tau_ {\mathrm{nr}} \phi_ {0} / 2} \\ & = i \sinh \left(\frac {\tau_ {\mathrm{nr}} \phi_ {0}}{2}\right) \csc 2 \Delta \phi , \end{array}\tag{6}
$$

which are implemented within the microcontroller through two time-domain convolutions, equivalently a Lorentzian-type resonance in the frequency domain [21,24]:

$$
Y _ {1 2 / 2 1} (f) = \frac {g _ {1 2 / 2 1} f _ {0}}{f _ {0} ^ {2} - (f + i \gamma) ^ {2}}.\tag{7}
$$

Then, $Y_{12}$ and $Y_{21}$ can become almost purely imaginary at the operation frequency chosen at $f_{0}$ , the resonating frequency of 1350 Hz, which is much larger than the resonance linewidth $\gamma = 50$ Hz and the resonance strengths. The resonance strengths are set by $g_{12} \cong -2i\gamma Y_{12}(f_{0}) = -10.1$ Hz and $g_{21} \cong -2i\gamma Y_{21}(f_{0}) = 12.3$ Hz. More details are given in Supplemental Material [34] for the implementation model. We note that the detailed implementation of the ideal quantitative values of the constitutive parameters (obtained through our inverse design approach) using meta atoms with digital feedback may not be easily achievable using a conventional physical atoms approach, particularly in both the active and nonreciprocal regime.

The transmittance and reflectance for a single unit cell in both directions are measured and shown in Fig. 2(b) as symbols, matching well with the analytic results [generated from Eq. (7)] in solid lines. Around the resonance frequency, the forward (backward) transmittance of the meta atom is around 1.5 (0.68), indicating amplification (suppression) with nonreciprocal transmission. The product of both amplitudes is approximately 1, as expected from the model. In addition, the reflection amplitudes for both directions are almost zero. We extract the four constitutive parameters from the complex scattering coefficients [34], as shown in Fig. 2(c). The real and imaginary parts of the experimental results are shown with solid and open symbols, respectively, agreeing with the analytical results (lines). The two measured Willis couplings $\tau$ and $\tau'$ are almost the same, i.e., purely nonreciprocal. At the resonance frequency $1350\mathrm{Hz}$ , the Willis couplings terms $i\tau$ and $i\tau'$ are approximately purely imaginary $(-0.09i)$ , resulting in the maximum contrast between the two transmission amplitudes in the spectrum. The extracted constitutive parameters satisfy the impedance matching condition with $\beta = \rho \cong 1$ and $\tau_{\mathrm{r}} = 0$ . We note that the constitutive parameters extracted from a single unit cell (of thickness $L$ ) is valid for multiple unit cells, as there is negligible near-field coupling between neighboring unit cells. The 1D metamaterial assembled by discrete atoms in the propagation direction can be equivalently represented as an effective medium with the same constitutive parameters.

![](images/7e3aeb58dc2ad15cfb050564103fce6fda6bdf5880ba30d1795f5721d04c1b25.jpg)
(c)

![](images/40ffdf046d8f90275e14a9585ac0e5d5db3412de7a1ace67632675633ffee227.jpg)

![](images/86197ff4d3c44570b90ff7177d7f2fce892769bf9e3dc33939c904ce15fdda1b.jpg)

(d)
![](images/a51c0ef0b52d76ef30e3e017fcb845cf75b406dcc154e0521cc1a4e296dd2950.jpg)
FIG. 3. (a) Forward and (b) backward transmittance for amplifying diode with 3 (black), 6 (blue), and 9 atoms (red). (c) The forward (red) and backward (black) transmittance at the resonance frequency 1350 Hz against the number of atoms. (d) Reflectance at the resonance frequency for both directions. Symbols (lines) represent the experimental (analytical) results.

Now, we cascade the meta atoms of identical configuration. We gradually increase the number of meta atoms by turning them on one by one and measure the scattering parameters. Figure 3(a) shows the forward transmittance ( $|t_{f}|^{2}$ ) for 3 (red), 6 (blue), and 9 meta atoms (black) in symbols for experimental and lines for analytic results. The forward transmission is largely amplified around the resonance frequency (1350 Hz). It increases with the number of atoms and reaches around 35 in the case of 9 atoms. The backward transmittance is largely suppressed with more atoms [Fig. 3(b)]. We also plot the two transmittances against the number of atoms N at the resonance frequency as symbols in Fig. 3(c) (in log scale). The forward amplification and backward suppression follow the exponential dependence $\pm8.686(-\phi_{0}\tau)N$ (lines) in dB from model [Eq. (4)]. In fact, the exponential suppression in the backward direction is important when we have amplification in the forward direction. They cancel each other for any tiny reflection from the two boundaries of the metamaterial in bouncing back and forth. As a result, the reflections from both directions always stay low [around -20 dB in Fig. 3(d)]. The combined effect: amplifying with zero reflection in one direction and perfect absorption in another, demonstrates a significant sound isolation beyond the regime of a conventional diode. It should be noted that we can also use a smaller number of atoms but a larger nonreciprocal Willis coupling to achieve the same transmittance due to the programming flexibility of the metamaterial. The programmability can also be used to other aspects such as a tunable bandwidth (see Supplemental Material for additional results [34]).

Up to now, we have mainly focused on purely nonreciprocal Willis coupling $\tau_{nr}$ with impedance matching: $\beta = \rho$ and $\tau_{r} = 0$ . We consider constructing amplifying diodes for more general Willis coupling. To begin, we flip the sign of resonance strength of the kernel $Y_{12}$ , i.e., $g_{12} = 10$ Hz. The small contrast between $Y_{12}$ and $Y_{21}$ now results in a much smaller magnitude of $\tau_{nr}$ and thus a smaller transmission contrast between the two directions [Fig. 4(a)]. There are some reflections with $r_{f} = r_{b}$ due to the deviation from the impedance matching condition as $\beta \neq \rho$ with $\tau_{r} = 0$ . These can be expressed as

$$
r _ {f} = r _ {b} = \frac {Z ^ {2} - 1}{2 i Z \cot (n _ {\mathrm{r}} \phi_ {0}) + 1 + Z ^ {2}},\tag{8}
$$

where $n_{r} = \sqrt{\rho\beta}$ and $Z = \sqrt{\rho/\beta}$ are the refractive index and the impedance of the medium, being independent of $\tau_{nr}$ [rather due to $\beta \neq \rho$ from Eq. (3)]. This is in contrast to the reciprocal Willis coupling, which affects the refractive index and splits the impedance (and reflection coefficients) into two values depending on the incident direction.

![](images/c9be831d37e512cab5af21c6848138d40e0d8989b5cca688ce50862ccdc8d5c3.jpg)

(b)
|r|
![](images/f35779bf6ba383f9bf0dd71e6eb1732ac0d34e2a88eca6664b9d58fdd7c6dd29.jpg)

(c)
![](images/c1169cc2e933a4c7461a50c6be65b2a8d332816cdd16e156f0b0279eecdbe0d7.jpg)
FIG. 4. (a) The measured amplitude of the scattering coefficients (symbols) for a meta atom with $\tau = \tau'$ but $\beta \neq \rho$ , agreeing to the analytical results (lines). (b) The reflection amplitude $|r| = |r_f| = |r_b|$ and (c) the transmission amplitude contrast $|t_f - t_b|$ for a general Willis metamaterial with thickness $L$ against the parameter $Z = \sqrt{\rho / \beta}$ and the reciprocal Willis parameters $\tau_{\mathrm{r}}$ . The white lines indicate the FP resonance condition and the point at $Z = 1$ and $\tau_{\mathrm{r}} = 0$ indicates the amplifying diode working condition with impedance matching.

Then, we extend our discussion to the case with both reciprocal and nonreciprocal Willis coupling present. We fix $\beta = 1$ and $\tau_{nr} = -0.09$ as in the experimental sample while we scan Z (in fact $\sqrt{\rho}$ ) and $\tau_{r}$ . With all real $\beta, \rho, \tau_{r}$ , and $\tau_{nr}, \tau_{r}$ causes the two reflection coefficients in opposite directions differ in phase but not in magnitude. We plot the reflection amplitude $|r|$ and the transmission contrast $|t_{f} - t_{b}|$ , against Z and $\tau_{r}$ in Figs. 4(b) and 4(c). At $\tau_{r} = 0$ , Z = 1, a dip in $|r|$ with a peak at $|t_{f} - t_{b}|$ indicates the amplifying diode with impedance matching. The denominator in Eq. (8) suggests a possible occurrence of Fabry-Pérot (FP) resonance, which becomes the discrete “parabolic” bands of $|r| = 0$ [white dashed curves in both Figs. 4(b) and 4(c)]. $|t_{f} - t_{b}|$ has a large amplitude at the same positions at FP resonances. The FP resonance approach to get amplifying diodes is useful when we do not have complete control of the Willis couplings so that both reciprocal and nonreciprocal components are present. It only works at discrete thicknesses to satisfy resonance conditions (similar to the wavelength sensitivity for a FP resonance). In contrast, the impedance matching approach allows amplification cascaded as thickness grows. It will be helpful when the gain of individual metamaterial atoms is limited by its power source. We note that the zero reflection accompanying the amplifying diode can be used to construct applications like amplified sensing. A sensor signal can be amplified by enclosing it within the metamaterial, but with the scattering kept invariant (see Supplemental Material [34]).

In summary, we have investigated the concept of an acoustic amplifying diode from either impedance matching or FP resonance using nonreciprocal and active Willis coupling. Adopting an “inverse” approach to achieve the ideal constitutive parameters that combine signal isolation and amplification in one dimension can be done at the fundamental signal frequency, as opposed to using frequency conversion with nonlinear sonic crystals $[28–30]$ . The ideal constitutive parameters, which are characterized by a detailed balance among the two Willis coupling, density, and modulus, are realized using meta atoms constructed with programmable digital feedback. The programmable nature of the metamaterial not only enables the implementation of active and nonreciprocal Willis coupling, which cannot be easily achieved using traditional methods, but also provides tunable bandwidth and can be utilized for demonstrations of practical applications such as amplified sensing (see Supplemental Material $[34]$ ). Looking forward to applications in larger systems, the ability to distribute the amplification (without reflection) into different components will further enable nonreciprocal control of wave propagation in more general settings, such as controlling two-dimensional waves for nonreciprocal communication, sensing and non-Hermitian topology $[35,36]$ . It is also interesting to note that the current implementation using non-Hermitian components is also related to a concept called parity-time symmetric laser absorber $[37–40]$ . An optical medium can act as a laser or a perfect absorber at the same time, depending on the form of coherent excitations. We have demonstrated a similar concept in acoustics: the Willis medium acts as an amplifier or an absorber only depending on the incident direction.

J. L. acknowledges support from Research Grants Council (RGC) of Hong Kong through projects No. 16303019 and AoE/P-502/20. N. P. acknowledges support from National Research Foundation of Korea (NRF) through the Global Frontier Program (No. 2014M3A6B3063708).

[1] I. Lindell, A. Sihvola, S. Tretyakov, and A. J. Viitanen, Electromagnetic Waves in Chiral and Bi-Isotropic Media (Artech House, London, 1994).

[2] M. Yazdi, M. Albooyeh, R. Alaee, V. Asadchy, N. Komjani, C. Rockstuhl, C. R. Simovski, and S. Tretyakov, A bianisotropic metasurface with resonant asymmetric absorption, IEEE Trans. Antennas Propag. 63, 3004 (2015).

[3] A. B. Khanikaev, S. Hossein Mousavi, W. K. Tse, M. Kargarian, A. H. MacDonald, and G. Shvets, Photonic topological insulators, Nat. Mater. 12, 233 (2013).

[4] Y. Ra'di, D. L. Sounas, and A. Alù, Metagratings: Beyond the Limits of Graded Metasurfaces for Wave Front Control, Phys. Rev. Lett. 119, 067404 (2017).

[5] M. B. Muhlestein, C. F. Sieck, P. S. Wilson, and M. R. Haberman, Experimental evidence of Willis coupling in a one-dimensional effective material element, Nat. Commun. 8, 15625 (2017).

[6] S. Koo, C. Cho, J. H. Jeong, and N. Park, Acoustic omni meta-atom for decoupled access to all octants of a wave parameter space, Nat. Commun. 7, 13012 (2016).

[7] Y. Liu, Z. Liang, J. Zhu, L. Xia, O. Mondain-Monval, T. Brunet, A. Alù, and J. Li, Willis Metamaterial on a Structured Beam, Phys. Rev. X 9, 011040 (2019).

[8] Y. Meng, Y. Hao, S. Guenneau, S. Wang, and J. Li, Willis coupling in water waves, New J. Phys. 23, 073004 (2021).

[9] A. Merkel, V. Romero-García, J. P. Groby, J. Li, and J. Christensen, Unidirectional zero sonic reflection in passive PT-symmetric Willis media, Phys. Rev. B 98, 201102(R) (2018).

[10] T. Liu, X. Zhu, F. Chen, S. Liang, and J. Zhu, Unidirectional Wave Vector Manipulation in Two-Dimensional Space with an All Passive Acoustic Parity-Time-Symmetric Metamaterials Crystal, Phys. Rev. Lett. 120, 124502 (2018).

[11] C. Shen, J. Li, X. Peng, and S. A. Cummer, Synthetic exceptional points and unidirectional zero reflection in non-Hermitian acoustic systems, Phys. Rev. Mater. 2, 125203 (2018).

[12] M. B. Muhlestein, C. F. Sieck, A. Alù, and M. R. Haberman, Reciprocity, passivity and causality in Willis materials, Proc. R. Soc. A 472, 20160604 (2016).

[13] C. F. Sieck, A. Alù, and M. R. Haberman, Origins of Willis coupling and acoustic bianisotropy in acoustic metamaterials through source-driven homogenization, Phys. Rev. B 96, 104303 (2017).

[14] L. Quan, Y. Ra'di, D. L. Sounas, and A. Alù, Maximum Willis Coupling in Acoustic Scatterers, Phys. Rev. Lett. 120, 254301 (2018).

[15] A. Melnikov, Y. K. Chiang, L. Quan, S. Oberst, A. Alù, S. Marburg, and D. Powell, Acoustic meta-atom with experimentally verified maximum Willis coupling, Nat. Commun. 10, 3148 (2019).

[16] L. Quan, D. L. Sounas, and A. Alù, Nonreciprocal Willis Coupling in Zero-Index Moving Media, Phys. Rev. Lett. 123, 064301 (2019).

[17] L. Quan, S. Yves, Y. Peng, H. Esfahlani, and A. Alù, Odd Willis coupling induced by broken time-reversal symmetry, Nat. Commun. 12, 2615 (2021).

[18] B. I. Popa, Y. Zhai, and H. S. Kwon, Broadband sound barriers with bianisotropic metasurfaces, Nat. Commun. 9, 5299 (2018).

[19] Y. Zhai, H. S. Kwon, and B. I. Popa, Active Willis metamaterials for ultracompact nonreciprocal linear acoustic devices, Phys. Rev. B 99, 220301(R) (2019).

[20] Y. Chen, X. Li, G. Hu, M. R. Haberman, and G. Huang, An active mechanical Willis meta-layer with asymmetric polarizabilities, Nat. Commun. 11, 3681 (2020).

[21] C. Cho, X. Wen, N. Park, and J. Li, Acoustic Willis meta-atom beyond the bounds of passivity and reciprocity, Commun. Phys. 4, 82 (2021).

[22] W. Cheng and G. Hu, Acoustic skin effect with nonreciprocal Willis materials, Appl. Phys. Lett. 121, 041701 (2022).

[23] B. I. Popa, L. Zigoneanu, and S. A. Cummer, Tunable active acoustic metamaterials, Phys. Rev. B 88, 024303 (2013).

[24] C. Cho, X. Wen, N. Park, and J. Li, Digitally virtualized atoms for acoustic metamaterials, Nat. Commun. 11, 251 (2020).

[25] D. A. Kovacevich and B. I. Popa, Programmable bulk modulus in acoustic metamaterials composed of strongly interacting active cells, Appl. Phys. Lett. 121, 101701 (2022).

[26] S. R. Craig, B. Wang, Xiaoshi Su, Debasish Banerjee, Phoebe J. Welch, Mighten C. Yip, Yuhang Hu, and Chengzhi Shi, Extreme material parameters accessible by

active acoustic metamaterials with Willis coupling, J. Acoust. Soc. Am. 151, 1722 (2022).

[27] Y. Hadad, J. C. Soric, and A. Alu, Breaking temporal symmetries for emission and absorption, Proc. Natl. Acad. Sci. U.S.A. 113, 3471 (2016).

[28] B. Liang, B. Yuan, and J. C. Cheng, Acoustic Diode: Rectification of Acoustic Energy Flux in One-Dimensional Systems, Phys. Rev. Lett. 103, 104301 (2009).

[29] B. Liang, X. S. Guo, J. Tu, D. Zhang, and J. C. Cheng, An acoustic rectifier, Nat. Mater. 9, 989 (2010).

[30] N. Boechler, G. Theocharis, and C. Daraio, Bifurcation-based acoustic switching and rectification, Nat. Mater. 10, 665 (2011).

[31] X. F. Li, X. Ni, L. Feng, M. H. Lu, C. He, and Y. F. Chen, Tunable Unidirectional Sound Propagation through a Sonic-Crystal-Based Acoustic Diode, Phys. Rev. Lett. 106, 084301 (2011).

[32] M. Chen, M. Kim, A. M. Wong, and G. V. Eleftheriades, Huygens' metasurfaces from microwaves to optics: A review, Nanophotonics 7, 1207 (2018).

[33] T. Feng, F. Li, W. Y. Tam, and J. Li, Effective parameters retrieval for complex metamaterials with low symmetries, Europhys. Lett. 102, 18003 (2013).

[34] See Supplemental Material at http://link.aps.org/supplemental/10.1103/PhysRevLett.130.176101 for the derivation of implementation model, the effective medium extraction, stability analysis, and results on broader bandwidth.

[35] D. L. Sounas and A. Alu, Non-reciprocal photonics based on time modulation, Nat. Photonics 11, 774 (2017).

[36] E. J. Bergholtz, J. C. Budich, and F. K. Kunst, Exceptional topology of non-Hermitian systems, Rev. Mod. Phys. 93, 015005 (2021).

[37] S. Longhi, PT-symmetric laser absorber, Phys. Rev. A 82, 031801(R) (2010).

[38] Y. D. Chong, L. Ge, and A. D. Stone, PT-Symmetry Breaking and Laser-Absorber Modes in Optical Scattering Systems, Phys. Rev. Lett. 106, 093902 (2011).

[39] Y. Sun, W. Tan, H. Q. Li, J. Li, and H. Chen, Experimental Demonstration of a Coherent Perfect Absorber with PT Phase Transition, Phys. Rev. Lett. 112, 143903 (2014).

[40] Z. Gu, N. Zhang, Q. Lyu, M. Li, S. Xiao, and Q. Song, Experimental demonstration of PT-symmetric stripe lasers, Laser Photonics Rev. 10, 588 (2016).
