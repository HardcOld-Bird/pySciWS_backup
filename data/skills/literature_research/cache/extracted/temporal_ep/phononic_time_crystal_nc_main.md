Article

# Temporal super-cell engineering and acoustic amplification in dispersive phononic time crystals

https://doi.org/10.1038/s41467-026-73459-5

Received: 17 July 2025

Accepted: 13 May 2026

Published online: 02 June 2026

Check for updates

Ziling Liu $^{1,6}$ , Xinghong Zhu $^{2,3,6}$ , Zhi-Guo Zhang $^{1,6}$ , Wei-Min Zhang $^{1}$ , Xue Chen $^{1}$ , Yong-Qiang Yang $^{1}$ , Ruwen Peng $^{4}$ , Mu Wang $^{4}$ , Jensen Li $^{5}$ & Hong-Wei Wu $^{1,4}$

Floquet time crystals, characterized by momentum band gaps (k-gaps), offer powerful mechanisms for exotic wave control. However, selectively harnessing the Floquet band structure and opening multiple k-gaps remains a significant challenge in experiment. In this work, we construct a phononic time crystal by integrating discrete resonant meta-atoms into a one-dimensional acoustic waveguide, effectively creating a time-varying metamaterial. Through dynamic compressibility modulation, we observe amplified transmission and strong emission enhancement for a compact Floquet slab at the k-gap-associated frequency. Based on this versatile platform, we further extend the Floquet band physics by introducing a temporal-supercell concept that creates multiple k-gaps via momentum band folding. By suitably designing the compressibility in each phase of the supercell, we experimentally observe two clear amplified transmission frequency ranges around half and quarter of the original modulation frequency, for a corresponding compact Floquet slab with a band-folding-induced k-gap. This reconfigurable platform enables tailored parametric processes and unlocks pathways to higher-dimensional time crystals and topological temporal phenomena.

Floquet time crystals constitute a new class of artificial materials, distinct from conventional spatial crystals. While spatial crystals are defined by periodic variations in space, Floquet time crystals feature constitutive parameters that remain uniform in space yet periodically modulated in time $^{1-5}$ . This temporal periodicity introduces discrete-time interfaces, leading to interference between time-reflected and time-refracted waves and giving rise to momentum band structures due to the breaking of discrete-time-translation symmetry. In contrast to the energy band gap in spatial crystals, Floquet time crystals exhibit momentum band gaps (k-gaps), which support two distinct Floquet modes: one grows exponentially and the other decays in time. Many intriguing phenomena have been theoretically predicted in photonic time crystals, including topological temporal edge states $^{6-9}$ , temporal Anderson localization $^{10,11}$ , amplified emission from electrons and dipole atoms $^{2,12}$ , and superluminal momentum-gap solitons $^{13}$ . To experimentally realize these effects, various time-varying photonic platforms have been developed. In the microwave regime, dynamic transmission lines have enabled the observation of k-gaps $^{14}$ , and temporally driven resonator arrays have revealed both Bloch-Floquet and non-Bloch band structures $^{15}$ . Additionally, time-varying metasurfaces have been employed to achieve exponential field growth within k-gaps by transforming volumetric systems into surface-based photonic time crystals $^{16}$ . Recently, the topological temporal boundary and the amplified wave in k-gap have also been observed in microwave $^{17,18}$ . However, realizing time-varying materials at optical frequencies remains a formidable task due to the requirement of ultrafast modulation, at least twice the optical carrier frequency. Promising candidates include all-optically modulated transparent conductive oxides, which offer significant changes in effective refractive index. Yet their application is hindered by the high optical pumping power required, which leads to thermal damage and limits performance $^{19-24}$ . Recent proposals have suggested expanding k-gaps using artificial resonators with time-varying resonance frequencies $^{25,26}$ , opening new directions for novel and more feasible modulation schemes.

As a universal concept, Floquet time crystals have been explored across various physical systems, including microwaves $^{16}$ , elastic waves $^{27-29}$ , water waves $^{30}$ , and acoustics $^{31}$ . However, realizing phononic time crystals for airborne sound remains a challenge, primarily because achieving fast, spatially uniform modulation of material properties is difficult. Previous strategies have included mechanically actuated resonators for nonreciprocal sound transmission $^{32}$ , yet these systems suffer from frictional losses, which limit both the achievable modulation rate and depth. Electroacoustic devices employing digital feedback have also been utilized to induce nonreciprocal mode transition $^{33}$ by temporally switching the acoustic impedance of transducers, although they typically operate at low modulation frequencies. More recently, digitally virtualized meta-atoms have been introduced $^{34}$ , which consist of microphone and speaker pairs interconnected by an external microcontroller implementing a time-varying convolution kernel. Such platforms have enabled investigations into temporal effective medium theory $^{35,36}$ at high modulation frequencies. Erewhile, two coupled cavities loading an external circuit are designed to observe the momentum-band topology in PT-symmetric Floquet lattices by temporally modulating the decay rate in acoustic $^{37}$ . Although recent experiments have sought to open a momentum-band gap in phononic time crystals, harnessing the momentum band structure and opening multiple k-gaps remain significant challenges in experiment, regardless of optical, acoustic, or elastic wave systems.

In this work, we experimentally realize a phononic time crystal by integrating programmable, time-varying resonant meta-atoms into a one-dimensional acoustic waveguide. By temporally modulating the resonant strength of these meta-atoms, we observe significantly amplified acoustic transmission, with a transmittivity exceeding 10, and emission enhancement, with a Purcell factor of 10 to a maximum of 30, for an inlaid monopole source within the k-gap-associated frequencies. Based on the versatile experimental platform, we further design temporal supercells that induce multiple Floquet band folding. The experiment results for amplified transmission indicate that the phononic time crystal with a temporal supercell not only exhibits an amplification at conventional k-gap-associated frequency around half the modulation frequency, but also a new amplification induced by band folding around a quarter of the original modulation frequency. Our well-designed time-varying metamaterials offer remarkable flexibility for engineering tailor-made dynamic responses and provide a powerful route for harnessing the Floquet band structure and enriching the physics of acoustic wave propagation.

## Results

## Constructing dispersive phononic time crystal

We begin with a one-dimensional (1D) phononic time crystal along the x-axis, as shown in Fig. 1a. Its compressibility $\beta(t)$ , normalized by the compressibility of air $\beta_{0}$ , is modulated over time (t) with a constant density $\rho_{0}$ . For simplicity, we consider the temporal modulation with alternating phases A and B, over a modulation period $T_{m}$ , shown as red and blue stripes representing $\beta_{A}(f)$ and $\beta_{B}(f)$ . Each phase occupies $T_{m}/2$ . To experimentally realize the dispersive compressibility, we design 9 meta-atoms loading on a 1D waveguide with subwavelength lattice length l = 0.02 m for constructing a homogenous metamaterial, as shown in Fig. 1b. Each meta-atom labeled by an index i, consists of a detector $D_{i}$ and a speaker $S_{i}$ arranged perpendicular to the waveguide with cross section $3cm \times 3cm$ for eliminating the spatial phase difference along the propagating direction. They are interconnected via a microcontroller that performs a time domain convolution on the detected signal. The resulting signal is feedback to the speaker to generate a resonating scattering response at each atom. The detailed operation in microcontroller for detected signal is presented in Note 1 of Supplementary Information. The meta-atom can mimic a time modulated resonator with a Lorentzian (named as positive Lorentzian)/ Anti-Lorentzian (named as negative Lorentzian) response in time. For example, the static Lorentzian-type compressibility can be described in the frequency domain as $^{35,36}$ :

$$
\beta (f) \approx 1 + \frac {c _ {0}}{i \pi f l} Y (f), Y (f) = \frac {i f g}{f _ {r e s} ^ {2} - f ^ {2} - 2 i \gamma f}\tag{1}
$$

Here, the parameters are selected as: resonant strength g=100 Hz, resonant frequency $f_{res}=4.7$ kHz and linewidth $\gamma=100$ Hz, $c_{0}=343$ m/s corresponds to sound speed in air. By modulating the resonating strength $g(t)$ between -100 Hz and 100 Hz with duty cycle $\xi=0.5$ periodically in time (which can be seen in left-top insert of Fig. 1b), we can realize a temporal metamaterial with the compressibility $\beta(f)$ switching between two static configurations. The static compressibility configurations are shown in Fig. 1c and d, labeled as "Static A" and "Static B", the transmission and reflection of a single static meta-atom with positive and negative responses are given in Supplementary Fig. 2.

Above, we have constructed a time-varying metamaterial that switches between positive and negative Lorentzian responses in time. Next, we discuss the Floquet band structure theoretically for a homogenous phononic time crystal. The airborne acoustic wave equations in waveguide are given as

$$
\partial_ {x} p (x, t) + \rho_ {0} \partial_ {t} v (x, t) = 0,\tag{2}
$$

$$
\partial_ {x} v (x, t) + \beta_ {0} \partial_ {t} (p (x, t) + M (x, t)) = 0,\tag{3}
$$

with $p(x,t)$ , $v(x,t)$ , $M(x,t)$ being pressure field, velocity field and the monopolar polarization, the monopolar response can be governed by a Lorentzian-type model

$$
\partial_ {t} ^ {2} M (x, t) + 2 \Gamma \partial_ {t} M (x, t) + \omega_ {0} ^ {2} M (x, t) = a (t) \omega_ {0} ^ {2} p (x, t),\tag{4}
$$

where $\omega_{0}=2\pi f_{res}$ is the resonating (radial) frequency, $\Gamma=2\pi\gamma$ is the resonating linewidth, $a(t)=c_{0}g(t)/(\pi f_{res}^{2}l)$ being the resonance strength $^{36}$ . Under temporal modulation, the wave number k remains unchanged across time interfaces. By substituting $\partial_{x}$ with ik, Eqs. (2), (3) and (4) can be written in matrix form as $^{36}$

$$
i \partial_ {t} \psi = \hat {\omega} \psi ,   \hat {\omega} = \left( \begin{array}{c c c c} 0 & \frac {k}{\beta_ {0}} & 0 & i \\ \frac {k}{\rho_ {0}} & 0 & 0 & 0 \\ 0 & 0 & 0 & - i \\ - i a (t) \omega_ {0} ^ {2} & 0 & i \omega_ {0} ^ {2} & - i 2 \Gamma \end{array} \right),\tag{5}
$$

with state vector $\psi = (p, v, M, -\partial_t M)^T$ . $\hat{\omega}$ is the propagation matrix. The state vector evolves as $\psi(t) = e^{-i\hat{\omega}t}\psi(0)$ . The Floquet modes satisfy $\psi(t + T_m) = e^{-i\hat{\omega}_B T_m/2}e^{-i\hat{\omega}_A T_m/2}\psi(t) = e^{-i\Omega T_m}\psi(t)$ , with $\hat{\omega}_A/\hat{\omega}_B$ being the propagation matrix in phase A/B and $\Omega=2\pi f$ being the Floquet frequency. This leads to the dispersion relation between $\Omega$ versus k by:

![](images/2c1ff0530865205a46bb90654fc5daf1060522ff53c0151bae2547e826cd3291.jpg)

![](images/06d456184c390a3c8dc11a80b75b0d9231f9e7d6853fba474b4baff24858fdbe.jpg)

![](images/14a6936b3831adcb76f8fafdcf9c3b51edc2216b9c98c6c94f43d2ef123e2977.jpg)
Fig. 1 | Concept of phononic time crystal and Floquet band structure.

a Schematic picture of a phononic time crystal whose compressibility switches between $\beta_{A}$ and $\beta_{B}$ with period $T_{m}$ . b Sketch map of an effective time-varying metamaterial constructed by 9 meta-atoms with a lattice length of 2cm, loading on a waveguide with a cross-section of 3cm × 3cm. Each meta-atom consists of a speaker (S) and a microphone (D) interconnected by a microcontroller (Y) performing a Lorentzian response. The resonating strength $g(t)$ is modulated between

$$
\det \left[ e ^ {- i \Omega T _ {m}} I _ {4} - e ^ {- i \hat {\omega} _ {B} T _ {m} / 2} e ^ {- i \hat {\omega} _ {A} T _ {m} / 2} \right] = 0,\tag{6}
$$

where $I_{4}$ is the 4 by 4 identity matrix. Solving this secular equation in Eq. (6), Fig. 1e presents the band structure of a phononic crystal with dispersive compressibility modulated between $\beta_{A}(f)$ and $\beta_{B}(f)$ at a modulation frequency $1/T_{m}=8.4\ kHz$ . A characteristic wave number associated with the modulation frequency is defined as: $k_{m}=2\pi/(c_{0}T_{m})$ . Then, the band structure is plotted as frequency versus normalized wave number $k/k_{m}$ , with red and blue lines denoting the real and imaginary parts of Floquet frequency, respectively. The k-gap region, highlighted in yellow, near half the modulation frequency ( $f_{m}/2=4.2kHz$ ), corresponds to non-zero imaginary part of eigenfrequency, featuring one amplifying and one decaying Floquet mode. Additionally, four more quasi energy bandgaps $^{38}$ are observed: two originate from the inherent resonance at 4.7 kHz, and the other two arise due to Floquet replica induced by time modulation, the detail is given in Supplementary Fig. 3. It should be pointed out that the material with the Lorentzian resonance here introduces an attenuation mode in the k-gap region. Unlike typical k-gap, the gap is no longer a “full” gap but a “partial” gap. This effect, however, does not suppress the temporal growth of the amplified Floquet mode.

## Experimental observation of amplified transmission in phononic time crystal

To experimentally observe amplification transmission in the k-gap-associated frequency as shown in Fig. 1 for demonstrating the capacity of our proposed platform, we construct an array of 9 meta-atoms in a 1D acoustic waveguide, which is same as Fig. 1b. We choose a chain of 9 meta-atoms with total length L=0.18m so that the amplified transmission strength induced by temporal modulation, which can be observed in the experiment. The details are discussed in Supplementary Fig. 4. The experimental set up is shown in Fig. 2a. As described as Fig. 1b, each meta-atom comprises a detector and a speaker (circled with yellow dashed box), the detector senses the acoustic pressure field in the waveguide and feeds the signal to the microcontroller (circled in red dashed box), which performs the atomic frequency response $Y(f)$ but with possible time varying $g(t)$ (as Note 1 of Supplementary Information). The resulting signal is then sent to the speaker through an amplifier (circled with an orange dashed box) to generate monopolar scattering, thereby realizing an effective compressibility $\beta(f)$ described by Eq. (1). We first evaluate the static compressibility of the two phases to verify $\beta_{A}(f)$ and $\beta_{B}(f)$ in the absence of time modulation. By scanning the incident frequency, we measure the transmission and reflection coefficients and extract the compressibility in the two phases $^{36}$ , as shown in Fig. 2b with red and blue symbols corresponding to the positive and negative Lorentzian-shape compressibility, respectively. Solid and hollow symbols represent the real and imaginary part. The experimental results show excellent agreement with the analytic results by Eq. (1) as the red and blue lines in Fig. 2b.

![](images/baed87b180d7de7e1bd6e1029358d8b14f1a1fdb9b7a8b1ef30fc43197a12cc1.jpg)

![](images/7b5abbf80df6623be968a2b5b490cf7aa325eb43e8a4bf831336de377e5d8494.jpg)
-100Hz to 100 Hz with period $T_{m}$ . c, d Static compressibility for phase A and B, cyan dashed line represents the resonating frequency 4.7 kHz, brown dashed line corresponds to the half modulated frequency 4.2 Hz. e Floquet band structure of the phononic time crystal with time-varying compressibility switching between phase $\beta_{\mathrm{A}}$ and $\beta_{\mathrm{B}}$ in c and d with $k_{\mathrm{m}} = 153.9$ rad/m. The insert gives the imaginary part of the Floquet frequency.

After establishing the static configurations, we modulate between two phases and measure the transmittance T and reflectance R spectra, presented in red and blue hollow dots in Fig. 2c. Since our meta-atoms involves feedback and time-varying resonant strength, the whole system (including all the atoms) is time dependent and frequency conversion occurs, thus we perform the stability analysis to make sure that the system is working in the stable regime as discussed in Note 4 of Supplementary Information. In the k-gap-associated frequency region, the transmittance is larger than 1, with larger value around half of the modulation frequency. The experiment results show great agreement with the simulation results, obtained from full-wave simulation using COSMOL Multiphysics (the simulation details are given in "Method"). Furthermore, two transmission valleys appears in the spectrum, at positions consistent with the expected quasi energy bandgaps of the band structure: one associated with the intrinsic resonating frequency 4.7 kHz and another due to the -1st harmonics 3.7 kHz for a modulation frequency of 8.4 kHz, as shown in Fig. 1e. To experimentally demonstrate that the amplified transmission originates from the interference between the Floquet modes, we also measure the Floquet mode components in the phononic time crystal when activating the temporal modulation. The result indicates that the outputting signals not only include the incident 0th order Floquet mode, but also have -1st order Floquet mode for tuning on the meta-atoms, as seen in Supplementary Fig. 6. Furthermore, we also measure the transmission and reflection spectra for different modulation frequency shown in of Supplementary Fig. 7. It is not difficult to find that the frequency range of amplified transmission shifts away from the original k-gap-associated frequency range marked by yellow region as the modulation frequency increases, and finally the transmission and reflection return to baseline (unity and zero) at higher modulation frequencies.

![](images/7f59fb4e8a0b0b559cc02c21ed861bccd1f9acf708015c2801f3eb1e4f068466.jpg)

![](images/93576e1b6b938e0d287857fef93c691b6e1483a0c7949d2ff22fab578a0209f6.jpg)
Fig. 2 | Experiment setup and amplified transmission. a Experimental platform of phononic time crystal with lattice distance $l = 0.02 \, \text{m}$ . b Static compressibility $\beta_A(f)$ (red line obtained from Eq. (1), red symbol extracted from experiment) in phase A and $\beta_B(f)$ (blue line is analytical result, blue symbol corresponds to experiment) in phase B. The parameters are chosen as same as Fig. 1: $f_{res} = 4.7 \, \text{kHz}$ , $g_A = 100 \, \text{Hz}$ , $g_B = -100 \, \text{Hz}$ , $y = 100 \, \text{Hz}$ . The faint yellow region is our interested frequency range around half modulation frequency $4.2 \, \text{kHz}$ marked by vertical orange dashed line.

The central peak at half the modulation frequency $f_{m} / 2$ arises from coupling between two Floquet components with 0th order and -1st order, depending on the phase difference $\Delta\varphi$ as shown in the insert of Fig. 2c. To investigate this effect, we vary the phase difference $\Delta\varphi$ between the incident signal and the modulation cycle from 0 to $2\pi$ and remeasure the transmittance, plotted in Fig. 2d. The transmittance spectrum is largely phase insensitive except near $f_{m}/2$ . Accordingly, we plot the transmittance at $f_{m}/2$ as a function of phase delay in the insert of Fig. 2d, using cyan dashed lines for experimental results and red lines for simulation. The transmittance goes through a cycle over $2\pi$ and a minimum occurs when the incident wave is out of phase with modulation cycle. Despite the phase variation, the transmittance consistently remains greater than 1, demonstrating the robustness of the amplifying mode within the k-gap-associated frequency. We also note that this anomalous peak (the sharp peak at exactly $f_{\mathrm{m}}/2$ ) only emerges for continuous wave excitation, not pulsed wave excitation. In our phononic time crystal, the sample length is only about two wavelengths, so the bulk k-gap is not expected to be fully resolved in the finite structure. Nevertheless, parametric amplification around $f_{\mathrm{m}}/2$ remains significant in this compact Floquet slab. We further find that the amplified transmission increases with sample length and is well described by the finite-slab transmission formula $t = e^{ik_g L}\sec (\Delta k_g L / 2)$ , where $k_{\mathrm{g}}$ and $\Delta k_{\mathrm{g}}$ denote the centre and width of the k-gap, respectively. Although this expression is derived for a nondispersive Floquet slab surrounded by a constant medium of the same averaged compressibility (the free-space one in our case), it provides an effective description of our dispersive system when these quantities are obtained from the corresponding Floquet band structure.

![](images/657f77aacd89365722083f238671521a76ab814d1afe9a541c528549da9c75d8.jpg)

![](images/b449eee779e8864b3509476b126795b54fb9875360d8e74123576890b552e314.jpg)
c Experimental and simulated transmittance and reflectance for phononic time crystal with modulated compressibility between A/B phases in b for $\Delta\varphi=0$ . It is defined in the inset as the zero phase difference between the incident wave arriving the leftmost metaatom represented by the gray sinusoidal curve and the red square wave modulation, the blue one has a positive phase delay $\Delta\varphi$ comparing to the incident wave. d Transmission amplitude with respect to phase delay between incident wave and modulation cycle.

d

![](images/efaecfb421e84e875017f800a495b858ddd2aa9d6f9b143a8e75af482048f950.jpg)

c
![](images/49150fdb342c1bc97065e85bc51d0b48ff8418d4a40203c351272d3561fbc72c.jpg)

b
![](images/02d98a73c5611980572ab1d0c204f84e7858ce99c028994a74e521dae0b403ce.jpg)

![](images/1478ce3b8a86b2db78f03e3db152085729cebf80ec7781af222e73f7725924f5.jpg)
Fig. 3 | Amplified emission in k-gap-associated frequency. Transmittance a and reflectance b spectra with increased modulation depth $\Delta a$ in experiment. c Schematic picture for radiating source in phononic time crystal. d Purcell factor as a function of incident frequency.

## Emission enhancement in the k-gap-associated frequency

In the previous section, we have demonstrated that our experiment platform can efficiently synthesize a phononic time crystal for temporally modulating between two static phases A and B. In Fig. 3a, we present the experimental transmittance as a function of modulating depth $\Delta a$ and frequency $f$ for $\Delta \varphi = 0$ . We observe that the transmittance increases from 1.7 to 13.4 as the modulating depth $\Delta a = \frac{2c_0g}{\pi f_{res}^2l}$ is raised from $4.8\times 10^{-2}$ to $10.79\times 10^{-2}$ . Similarly, the reflectivity peak increases with increasing modulating depth at the k-gap-associated frequencies. In fact, these peaks are broader and more pronounced in experiment than in simulations, likely due to a small impedance mismatch between the realized system and the free-space and enhance residual backscattering in the finite sample.

It is well known that the emission enhancement is a unique property of Floquet time crystals $^{2,11}$ , whose eigenmodes grow temporally over time, independent of initial phase. To further demonstrate the amplified emission in the phononic time crystal, we place an acoustic source at the center of the phononic time crystal structure, as shown in Fig. 3c. To quantify the enhancement, we first deactivate all meta-atoms, and measure the outgoing sound pressure at both ends of the 1D empty waveguide, denoted as initial state: $p_{i}$ . Next, we activate all meta-atoms to generate a k-gap-associated frequency around 4.2 kHz and record the amplified sound pressure, defining the final state: $p_{f}$ . To characterize the amplified capacity of the phononic time

crystal, we define the acoustic Purcell factor as $PF = \left|\frac{p_{f}}{p_{i}}\right|^{2}$ , which is plotted in Fig. 3d as a function of incident frequency. We find that the PF can reach 10 on the broad peak to a maximum of 30 at 4.2 kHz for a modulating depth of $\Delta a = 10.79 \times 10^{-2}$ under finite spatial length for continuous-wave excitation. In fact, the order of amplified emission can be further enhanced by increasing either modulating depth or the number of meta-atoms, leading to laser-like emission behavior.

## Floquet band folding and dual-band transmission amplifications by introducing temporal supercell

Until now, we have demonstrated the capability of well-designed meta-atoms for realizing the phononic time crystal by the experiments of amplified transmission and emission enhancement. In spatial phononic crystals, it is well known that expanding the unit cell to the supercell crystal causes the energy band structure to fold into the first Brillouin zone and open new bandgaps $^{40,41}$ .

In the same spirit, we propose the Floquet band folding by expanding the unit cell “AB” shown in Fig. 1a to temporal supercell “ABAB” as shown in Fig. 4a. The material and modulating parameters are same as Fig. 1, except the temporal unit cell is expanded to supercell. The previous band structure ( $\Omega$ verse k) relationship as Eq. (6) will be rewritten as: $\det\left[e^{-i\Omega2T_{m}}I_{4}-e^{-i\hat{\omega}_{B}T_{m}/2}e^{-i\hat{\omega}_{A}T_{m}/2}e^{-i\hat{\omega}_{B}T_{m}/2}e^{-i\hat{\omega}_{A}T_{m}/2}\right]=0$ for the case of temporal supercell, and the resulting Floquet band structures are shown in Fig. 4b. Here, for the convenience of discussion, we calculate the Floquet frequency range from $-\pi/T_{m}$ to $\pi/T_{m}$ as vertical coordinates corresponding to the first Floquet Brillouin zone. The red bands correspond to the unfolded case for the temporal unit cell “AB”, with Floquet frequency periodicity $2\pi/T_{m}$ . After expanding the duration to $2T_{m}$ , the red bands will be folded into the first Floquet Brillouin zone (yellow region), forming a blue band structure within the region $\pm\pi/2T_{m}$ marked by yellow dashed lines. We can find that the primary k-gap is still around $k_{m}/2$ corresponding to $f_{m}/2$ , same as the red band structure. However, at $k_{m}/4$ and $3k_{m}/4$ corresponding to the frequencies of $f_{m}/4$ and $3f_{m}/4$ , the folding points are susceptible to perturbations (e.g., changes in material parameters, phase durations) and will open to form extra k-gaps.

To open the folding points at $k_{m}/4$ , here we introduce a perturbation in the material parameters (stack-ABAC) to construct phononic time crystal, as illustrated in Fig. 5a. The static phases A, B, C correspond to dispersive compressibilities $\beta_{A}$ , $\beta_{B}$ , and $\beta_{C}$ as shown in Fig. 5b, respectively. Each phase lasts for $T_{m}/2$ . For a practical experiment, we select the modulation frequency $f_{m}=6.6kHz$ and the corresponding wave number $k_{m}=\frac{2nf_{m}}{c_{0}}=120.8rad/m$ . The compressibility values of the three phases here also differ from those used in Fig. 1.

To experimentally observe the Floquet band folding and dual-band amplified transmissions, we design the meta-atoms with two resonances in $Y(f)$ of the Eq. (1), enabling an approximation of the same compressibility around $f_{m}/4$ and $f_{m}/2$ , shown in top inserts of Fig. 5b. The resulting dispersive compressibility is given by:

$$
\beta_ {n} (f) = 1 + \frac {a _ {1 n} f _ {r e s 1} ^ {2}}{f _ {r e s 1} ^ {2} - f ^ {2} - 2 i \gamma_ {1} f} + \frac {a _ {2 n} f _ {r e s 2} ^ {2}}{f _ {r e s 2} ^ {2} - f ^ {2} - 2 i \gamma_ {2} f},\tag{7}
$$

![](images/b0f1d0e61ffea76f2a6cfd8b608a5c0221fa540820acc445322fdbdfa06f980d.jpg)

![](images/2109f5ca8d9d1cf703fda84bb8af80a8ee7f16d7cca19a1e25c4cdbdac0cd1f6.jpg)
Fig. 4 | Temporal supercell and Floquet band folding. a Schematic diagram of temporal supercell with duration $2\mathrm{T}_{\mathrm{m}}$ . b Floquet band structure of phononic time crystal with unit cell (red solid line) and supercell (blue solid line).

![](images/5240863f5c60dc9fbd0d3ed47252d01f0446fd04fba350b14ff6abc9d7a1a62d.jpg)

![](images/094e665872897c15a87f9a452919358cd037bd3c9557dd1417ce9691f5fb9781.jpg)

![](images/e3bb16edb92b849e5c2df61598cf0fdbf51567d58dec2f2b57463e1439a09a5b.jpg)

![](images/f7d911ce950bfe55b6260e194fb17e5e53b6825641af97de5f72bba162c7de3a.jpg)

![](images/74c7c560704be3ec7570ac2b2e860a0dc0a1fde8e3e4962de19432b4f581a441.jpg)
Fig. 5 | Floquet band folding and multiple transmission amplifications. a Conceptual picture of phononic time crystal with temporal supercell "ABAC". The values of the vertical coordinate indicate the compressibility of static phases at $f = f_{m} / 4$ and $f_{m} / 2$ . b Static compressibility for different configurations A ( $\beta_{\mathrm{A}}$ ), B ( $\beta_{\mathrm{B}}$ ), C ( $\beta_{\mathrm{C}}$ ) with two resonating responses, the vertical cyan and orange dashed lines marked the resonating frequencies $f_{res1}$ and $f_{res2}$ , the top inserts shows the

where the subscript n represents static phases “A”, “B” and “C”, the resonating frequencies are $f_{res1}=1.8\ kHz$ , $f_{res2}=3.6\ kHz$ with decay rates $\gamma_{1}=50Hz$ , $\gamma_{2}=100Hz$ . The resonating strengths are chosen to obtain the desired compressibility values $\beta_{A}=1.3$ , $\beta_{B}=1.25$ , $\beta_{C}=0.15$ at $f_{m}/4$ and $f_{m}/2$ as the schematic diagram in Fig. 5a. For example, to realize $\beta_{A}$ , we set $a_{1A}=3a_{10}$ , $a_{2A}=3a_{20}$ , as shown by the red line in Fig. 5b. The yellow line, representing static case B, corresponds to $a_{1B}=2.5a_{10}$ , $a_{2B}=2.5a_{20}$ , while static case C as shown by the purple line requires $a_{1C}=-8.5a_{10}$ , $a_{2C}=-8.5a_{20}$ , where $a_{10}=0.01345$ , $a_{20}=0.0186$ corresponds to $g_{10}=7.98Hz$ , $g_{20}=44.15Hz$ . To solve the band structure in this case, we can construct a 6 by 6 eigenvalue problem and solve the secular equation:

$$
\det \left[ e ^ {- 2 i \Omega T _ {m}} I _ {6} - e ^ {- i \hat {\omega} _ {C} T _ {m} / 2} e ^ {- i \hat {\omega} _ {A} T _ {m} / 2} e ^ {- i \hat {\omega} _ {B} T _ {m} / 2} e ^ {- i \hat {\omega} _ {A} T _ {m} / 2} \right] = 0,\tag{8}
$$

where $I_{6}$ is the 6 by 6 identity matrix, and the expressions of $\hat{\omega}_{A}, \hat{\omega}_{B}, \hat{\omega}_{C}$ are given in Note 7 of Supplementary Information. Figures 5c and 5d present the real and imaginary parts of the Floquet band structure of the dispersive phononic time crystal under temporal modulation with the supercell of Fig. 5a. We observe that the band structure of the dispersive phononic time crystal firmly presents three k-gaps in $k_{m}/4$ , $k_{m}/2$ and $3k_{m}/4$ , denoted as $\frac{1}{4}$ k-gap, $\frac{1}{2}$ k-gap, and $\frac{3}{4}$ k-gap. The $\frac{1}{4}$ k-gap and $\frac{3}{4}$ k-gap come from the band folding and reopen as plotted in blue region, the $\frac{1}{4}$ k-gap has a gap width from $0.222k_{m}$ to $0.275k_{m}$ around $\frac{k_{m}}{4}$ corresponding to the frequencies from 1476Hz to 1842Hz around $f_{m}/4$ . For the chosen parameters in the designed compressibility, the $\frac{3}{4}$ k-gap at $k = 0.75k_{m}$ is too narrow to have significant effect. In fact, if we design meta-atoms with three resonating responses to ensure same compressibility values around $\frac{f_{m}}{4}$ , $\frac{f_{m}}{2}$ and $\frac{3f_{m}}{4}$ , the $\frac{3}{4}$ k-gap width will be obviously expanded and the amplified transmission can also be measured in experiment. In this work, we mainly focus on the $\frac{1}{4}$ k-gap corresponding frequencies in the experiment due to the limitation of high-frequency in our waveguide. Furthermore, the width of $\frac{1}{4}$ k-gap depends on the compressibility contrast between the static phases B and C, the larger contrast, the wider the corresponding $\frac{1}{4}$ k-gap. The $\frac{1}{2}$ k-gap has a width from $0.446k_{m}$ to $0.548k_{m}$ as a pink region around $\frac{k_{m}}{2}$ relating to the associated frequencies from 2964Hz to 3624Hz around $f_{m}/2$ . Due to the high flexibility of our time-varying metamaterials in tailoring resonating responses as discussion in Note 1 of Supplementary Information, we experimentally implement the modulation for dispersive media with the compressibility shown in Fig. 5b at the phase difference $\Delta\varphi=0$ . Figure 5e shows the transmissivity and reflectivity at the frequency range around $f_{m}/4$ corresponding to the $\frac{1}{4}$ k-gap-associated frequency. Red lines and symbols represent simulated and experimental transmissivity, while blue lines and symbols denote the reflectivity results. The experiment results confirm the amplified transmission at the k-gap-associated frequency, as indicated by a transmission spectrum greater than 1. Furthermore, a similar high transmission spectrum is also observed at the second k-gap near $f_{m}/2$ corresponding to $\frac{1}{2}$ k-gap, as shown in Fig. 5f. The stronger transmission at $f_{m}/2$ arises because the finite spatial length $L=0.18$ m of phononic time crystal covers more modulation periods for higher operational frequencies. Furthermore, the amplified transmission peak of experimental measurement is slightly shifted compared with the simulated result in both k-gaps due to slight desynchronization of nine meta-atoms in the experiment. Comparing with $\frac{1}{4}$ k-gap-associated frequency, more obvious discrepancy for $\frac{1}{2}$ k-gap-associated frequency is that the inconsistency is magnified due to more temporal modulation periods for higher operational frequencies. In brief, our versatile time-varying metamaterials successfully demonstrate the band folding and multiple amplified transmissions through time modulation.

![](images/2be6cba9be3394f15b46eed160c546339e1f6b433857b3ea752cea315eaecc1a.jpg)
compressibility around the $f_{m}/4$ and $f_{m}/2$ . c, d Band structure of phononic time crystal with temporal supercell in a for the real and imaginary part, the blue regions for both 1/4 k-gap and 3/4 k-gap, and the pink region correspond to the 1/2 k-gap. Experimental and simulated results of transmittance and reflectance spectra around the $f_{m}/4$ e and $f_{m}/2$ f region.

## Discussion

To summarize, we have experimentally demonstrated the emergence and control of Floquet band structure in phononic time crystals for airborne sound by implementing time-periodic modulation of compressibility. Utilizing acoustic meta-atoms with programmable, time-varying Lorentzian responses, we observed significant amplification in both acoustic transmission and emission within the k-gap-associated frequency. Building on this foundation, we introduced a temporal supercell with three distinct compressibility phases, enabling the formation of multiple Floquet band folding, as confirmed by transmission spectral measurements. We note that in our experiments, at the center of the k-gaps, an anomalous transmission peak appeared under continuous-wave excitation, whose amplitude exhibits stronger sensitivity than the broad peak to the phase difference between the incident wave and the modulation cycle. This anomalous peak, however, disappears under pulse-wave excitation.

In brief, the temporal supercell, in addition to non-periodic temporal modulation, such as the supercell definition in time quasicrystals $^{42}$ , provides a systematic route to engineer Floquet band structures in Floquet time crystals. By extending the modulation period, it enables predictable band folding and controllable k-gap opening, beyond ad hoc temporal modulation. The resulting k-gaps directly lead to sound-amplifying transmission, demonstrating nontrivial wave phenomena enabled by Floquet band engineering. Particularly, this control arises from hierarchical temporal modulation, where multiple periodicities introduced by the supercell provide structured degrees of freedom for independently tuning band folding, k-gap formation, and amplification. Such hierarchical control also enables more flexible engineering of Floquet band topology, while the temporal implementation allows in situ tuning within a single platform. Our platform with programmable met-atoms makes this experimentally accessible. To our knowledge, temporal supercells have not been demonstrated experimentally, likely due to limited control over temporal modulation. Our system enables programmable spatiotemporal modulation and realization of both fundamental and supercell-induced k-gaps, providing a practical basis for exploring more complex band and topological effects.

## Methods

## Numerical simulation

In this work, the simulation is performed by a pressure acoustic model of COMSOL Multiphysics V6.3 in the time domain. The phononic time crystal is constructed using a lattice chain of meta-atoms in a onedimensional waveguide. Each meta-atom (Fig. 1b) is modeled as a point source in its secondary radiation in the waveguide. The wave equations considering secondary radiation of the point sources are governed by:

$$
\partial_ {x} p (x, t) + \rho_ {0} \partial_ {t} v (x, t) = 0
$$

$$
\partial_ {x} v (x, t) + \beta_ {0} \partial_ {t} p (x, t) = \Sigma_ {i} \delta \left(x - x _ {i}\right) \frac {2}{\rho_ {0} c _ {0}} q _ {i} (t),\tag{9}
$$

where $c_{0}=1/\sqrt{\beta_{0}\rho_{0}}$ is air sound speed, $\rho_{0}$ is the density of air and $\beta_{0}$ is the compressibility of air. The right-hand term of the equation represents the monopolar secondary point sources generated by speakers ( $S_{i}$ ) at positions $x_{i}$ . For each atom, the Lorentzian model is implemented by a second order differential equation (ODE), which responds to the pressure field by

$$
\partial_ {t} ^ {2} q _ {i} (t) + 2 \Gamma \partial_ {t} q _ {i} (t) + \omega_ {0} ^ {2} q _ {i} (t) = - G (t) \partial_ {t} p (x _ {i}, t),\tag{10}
$$

where $\omega_{0}=2\pi f_{res}$ is the resonating (radial) frequency, $\Gamma=2\pi\gamma$ is the resonating linewidth, and $G(t)=2\pi g(t)$ is the resonance strength for each atom. $q_{i}$ has the same unit as the pressure p in our convention. The $p(x_{i},t)$ is the total pressure field picked by the detector $D_{i}$ which includes the incident field and the secondary radiation fields from all speakers. While resonating strength g can be generally time-varying to implement a specific compressibility of the metamaterial. In the harmonic representation of Eq. (10), the microcontroller implements a frequency response Y (common for all atoms) defined by

$$
Y (f) = \frac {q _ {i} (f)}{p (x _ {i} , f)} = \frac {i f g}{f _ {r e s} ^ {2} - f ^ {2} - 2 i \gamma f}\tag{11}
$$

The micro-controller only implements an open-circuit frequency response but in the weak scattering limit, such open-circuit frequency response approximates well the closed-circuit impulse response $q_{i}(f)/p(x_{i},f)$ . Then, in the homogenization limit, $q_{i}$ is dispersed into the lattice constant l of the metamaterial, giving rise to $M=c_{0}q_{i}/(i\pi fl)$ . From the constitutive relationship $p+M=\beta p$ , we can obtain the compressibility

$$
\beta (f) = 1 + \frac {c _ {0} Y}{i \pi f l} = 1 + \frac {a f _ {r e s} ^ {2}}{f _ {r e s} ^ {2} - f ^ {2} - 2 i \gamma f},\tag{12}
$$

where $a=c_{0}g/(\pi f_{res}^{2}l)$ is the overall resonance strength in the macroscopic level in specifying compressibility $\beta(f)$ . Equations (9) and (10) are used to implement a specified Lorentzian resonance for $\beta(f)$ . Based on the above, we can implement a time-varying metamaterial, e.g., in switching between two values of a (or $g(t)$ ) on the microscopic level, with a modulation period $T_{m}$ .

## Experimental detail

All experiments were conducted in a one-dimensional waveguide made of stainless steel tube, which featured nine virtualized atoms placed on top of it with a lattice constant of $l = 0.02 \, \text{m}$ , as depicted in Fig. 2. Each atom was composed of a speaker and microphone pair, with a microcontroller (Stm32f407rc) connected between them. The total delay time for each atom, comprising the delay time for the microphone, speaker, and electronic delay for the microcontroller, was experimentally obtained to be $\delta t = 0.25 \, \mu\text{s}$ around 4.2 kHz (the half modulation frequency). To achieve a Lorentz resonant response with a digital microcontroller, Eq. (10) was transformed into a time-discrete model, but with the frequency response function $Y$ now approximated as the open-circuit frequency response function connecting $D_i$ to $S_i$ to be implemented electronically. The sampling frequency was set at 1.0 MHz, with a sampling period of 1 $\mu\text{s}$ . Additionally, the program for the Lorentz response was designed to finish within one sampling period in preparation for the next sample to be sent to the speaker. The setup for a phononic time crystal composed of 9 atoms is depicted in Fig. 2. The 4-point measurement method is utilized in conjunction with a National Instruments DAQ device to determine the reflection and transmission signals. Through the application of the Fourier transform, we can obtain the transmission and reflection coefficients.

## Data availability

The theoretical and experimental data generated in this study have been deposited in the Baidu Netdisk database under accession code w869 [https://pan.baidu.com/s/1xJ-oljpGAu7kqf1RZdG1NQ]. Other data that support the findings of this study are available from the corresponding authors upon request.

## References

1. Else, D. V., Bauer, B. & Nayak, C. Floquet Time Crystals. Phys. Rev. Lett. 117, 090402 (2016).

2. Lyubarov, M. et al. Amplified emission and lasing in photonic time crystals. Science 377, 425–428 (2022).

3. Galiffi, E. et al. Photonics of time-varying media. Adv. Photonics 4, 014002 (2022).

4. Yin, S., Galiffi, E. & Alù, A. Floquet metamaterials. eLight 2, 8 (2022).

5. Asgari, M. M. et al. Theory and applications of photonic time crystals: a tutorial. Adv. Opt. Photon. 16, 958 (2024).

6. Yang, Y. et al. Topologically Protected edge states in time photonic crystals with chiral symmetry. ACS Photonics 12, 2389 (2025).

7. Oudich, M., Deng, Y., Tao, M. & Jing, Y. Space-time phononic crystals with anomalous topological edge states. Phys. Rev. Res. 1, 033069 (2019).

8. Lustig, E., Sharabi, Y. & Segev, M. Topological aspects of photonic time crystals. Optica 5, 1390–1395 (2018).

9. Wang, B. et al. Observation of photonic topological Floquet time crystals. Laser Photonics Rev. 16, 2100469 (2022).

10. Sharabi, Y., Lustig, E. & Segev, M. Disordered photonic time crystals. Phys. Rev. Lett. 126, 163902 (2021).

11. Carminati, R., Chen, H., Pierrat, R. & Shapiro, B. Universal statistics of waves in a random time-varying medium. Phys. Rev. Lett. 127, 094101 (2021).

12. Dikopoltsev, A. et al. Light emission by free electrons in photonic time-crystals. Proc. Natl. Acad. Sci. USA. 119, e2119705119 (2022).

13. Pan, Y., Cohen, M.-I. & Segev, M. Superluminal k-gap solitons in nonlinear photonic time crystals. Phys. Rev. Lett. 130, 233801 (2023).

14. Reyes-Ayona, J. R. & Halevi, P. Observation of genuine wave vector (k or $\beta$ ) gap in a dynamic transmission line and temporal photonic crystals. Appl. Phys. Lett. 107, 074101 (2015).

15. Park, J. et al. Revealing non-Hermitian band structure of photonic Floquet media. Sci. Adv. 8, eabo6220 (2022).

16. Wang, X. et al. Metasurface-based realization of photonic time crystals. Sci. Adv. 9, eadg7541 (2023).

17. Ren, Y. et al. Observation of momentum-gap topology of light at temporal interfaces in a time-synthetic lattice. Nat Commun 16, 707 (2025).

18. Xiong, J. et al. Observation of wave amplification and temporal topological state in a non-synthetic photonic time crystal. Nat Commun 16, 11182 (2025).

19. Alam, M. Z., De Leon, I. & Boyd, R. W. Large optical nonlinearity of indium tin oxide in its epsilon-near-zero region. Science 352, 795–797 (2016).

20. Bohn, J., Luk, T. S., Horsley, S. & Hendry, E. Spatiotemporal refraction of light in an epsilon-near-zero indium tin oxide layer: frequency shifting effects arising from interfaces. Optica 8, 1532–1537 (2021).

21. Zhou, Y. et al. Broadband frequency translation through time refraction in an epsilon-near-zero material. Nat Commun 11, 2180 (2020).

22. Caspani, L. et al. Enhanced nonlinear refractive index in $\varepsilon$ -near-zero materials. Phys. Rev. Lett. 116, 233901 (2016).

23. Tirole, R. et al. Double-slit time diffraction at optical frequencies. Nat. Phys. 19, 999–1002 (2023).

24. Lustig, E. et al. Time-refraction optics with single cycle modulation. Nanophotonics 12, 2221–2230 (2023).

25. Zhang, S., Dong, J., Li, H., Xu, J. & Shapiro, B. Longitudinal optical phonons in photonic time crystals containing a stationary charge. Phys. Rev. B 110, L100306 (2024).

26. Wang, X. et al. Expanding momentum bandgaps in photonic time crystals through resonances. Nat. Photon. 19, 149–155 (2025).

27. Wang, Y. et al. Observation of nonreciprocal wave propagation in a dynamic phononic lattice. Phys. Rev. Lett. 121, 194301 (2018).

28. Trainiti, G. et al. Time-periodic stiffness modulation in elastic metamaterials for selective wave filtering: Theory and experiment. Phys. Rev. Lett. 122, 124301 (2019).

29. Kim, B. L., Chong, C., Hajarolasvadi, S., Wang, Y. & Daraio, C. Dynamics of time-modulated, nonlinear phononic lattices. Phys. Rev. E 107, 034211 (2023).

30. Apffel, B., Wildeman, S., Eddi, A. & Fort, E. Experimental implementation of wave propagation in disordered time-varying media. Phys. Rev. Lett. 128, 094503 (2022).

31. Cheng, Z. et al. Observation of $\pi/2$ modes in an acoustic Floquet system. Phys. Rev. Lett. 129, 254301 (2022).

32. Shen, C., Zhu, X., Li, J. & Cummer, S. A. Nonreciprocal acoustic transmission in space-time modulated coupled resonators. Phys. Rev. B 100, 054302 (2019).

33. Chen, Z. et al. Efficient nonreciprocal mode transitions in spatiotemporally modulated acoustic metamaterials. Sci. Adv. 7, eabj1198 (2021).

34. Cho, C., Wen, X., Park, N. & Li, J. Digitally virtualized atoms for acoustic metamaterials. Nat Commun 11, 251 (2020).

35. Wen, X., Zhu, X., Wu, H. W. & Li, J. Realizing spatiotemporal effective media for acoustic metamaterials. Phys. Rev. B 104, L060304 (2021).

36. Zhu, X., Wu, H.-W., Zhuo, Y., Liu, Z. & Li, J. Effective medium for time-varying frequency-dispersive acoustic metamaterials. Phys. Rev. B 108, 104303 (2023).

37. Tong, S. et al. Observation of momentum-band topology in PT-symmetric Floquet lattices. Nat Commun 16, 9975 (2025).

38. Feng, F., Wang, N. & Wang, G. P. Temporal transfer matrix method for Lorentzian dispersive time-varying media. Appl. Phys. Lett. 124, 101701 (2024).

39. Landi, M., Zhao, J., Prather, W. E., Wu, Y. & Zhang, L. Acoustic Purcell effect for enhanced emission. Phys. Rev. Lett. 120, 114301 (2018).

40. Zhao, D., Xiao, M., Ling, C. W., Chan, C. T. & Fung, K. H. Topological interface modes in local resonant acoustic systems. Phys. Rev. B 98, 014110 (2018).

41. Wu, L., Zhuang, L. & He, S. Degeneracy analysis for a supercell of a photonic crystal and its application to the creation of band gaps. Phys. Rev. E 67, 026612 (2003).

42. Ni, X., Yin, S., Li, H. & Alù, A. Tolopogical wave phenomena in photonic time quasicrystals. Phys. Rev. B 111, 125421 (2025).

## Acknowledgements

The authors thank Z. Sheng and H. Xu for technical support and discussion, and Simon Horsley of University of Exeter for the discussion on theoretical model. H. -W. W. acknowledgments Center for Fundamental Physics of Anhui University of Science and Technology, and the Hefei Comprehensive National Science Center for providing open access funding.

## Author contributions

H.-W. W. initiated and supervised the project. H.-W. W., Z. L. and X. Z. developed the theory and carried out numerical simulations. Z. L. and Z.-G. Z built the experimental setup and performed the experiments. J. L., R. P., M. W. and H.-W. W. analyzed the data and discussion. H.-W. W., J. L. and X. Z. co-wrote the draft. W.-M. Z., Y.-Q. Y. and X. C. participated in discussions.

## Funding

H.-W.Wu discloses support for the research of this work from the Natural Scientific Research Projects of Anhui Educational Committee (Grants No. 2022AH040114). R. P. and M. W. disclose support for the research of this work from the National Key Research and Development Program of China (Grants No. 2022YFA1404303). R.P. discloses the support from the National Natural Science Foundation of China (Grant No. 12234010). J.L. acknowledges support from the EPSRC via the META4D Programme Grant (Grants No. EP/Y015673/1). Z.L., X.Z., Z.-G.Z., W.-M.Z., Y.-Q.Y., and X.C. declare no relevant funding.

## Competing interests

The authors declare no competing interests.

## Additional information

Supplementary information The online version contains supplementary material available at https://doi.org/10.1038/s41467-026-73459-5.

Correspondence and requests for materials should be addressed to Ruwen Peng, Mu Wang, Jensen Li or Hong-Wei Wu.

Peer review information Nature Communications thanks the anonymous reviewers for their contribution to the peer review of this work. A peer review file is available.

Reprints and permissions information is available at http://www.nature.com/reprints

Publisher's note Springer Nature remains neutral with regard to jurisdictional claims in published maps and institutional affiliations.

Open Access This article is licensed under a Creative Commons Attribution 4.0 International License, which permits use, sharing, adaptation, distribution and reproduction in any medium or format, as long as you give appropriate credit to the original author(s) and the source, provide a link to the Creative Commons licence, and indicate if changes were made. The images or other third party material in this article are included in the article's Creative Commons licence, unless indicated otherwise in a credit line to the material. If material is not included in the article's Creative Commons licence and your intended use is not permitted by statutory regulation or exceeds the permitted use, you will need to obtain permission directly from the copyright holder. To view a copy of this licence, visit http://creativecommons.org/licenses/by/4.0/.

© The Author(s) 2026
