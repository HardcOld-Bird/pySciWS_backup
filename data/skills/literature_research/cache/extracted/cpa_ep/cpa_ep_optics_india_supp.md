# Supporting Information

for Laser Photonics Rev., DOI 10.1002/lpor.202200308

Non-Hermitian Control of Topological Scattering Singularities Emerging from Bound States in the Continuum

Zarko Sakotic\*, Predrag Stankovic, Vesna Bengin, Alex Krasnok, Andrea Alú and Nikolina Jankovic

# Supplementary material: Non-Hermitian Control of Topological Scattering Singularities Emerging from Bound States in the Continuum

Zarko Sakotic $^{1,2*}$ , Predrag Stankovic $^{2}$ , Vesna Bengin $^{2}$ , Alex Krasnok $^{3}$ , Andrea Alú $^{4,5}$ , and Nikolina

Jankovic $^{2}$

$^{1}$ Electrical and Computer Engineering Department, University of Texas, Austin, Texas 78758, USA

$^{2}$ BioSense Institute-Research Institute for Information Technologies in Biosystems, University of Novi Sad, Dr

Zorana Djindjica 1a, 21101, Novi Sad, Serbia

$^{3}$ Department of Electrical and Computer Engineering, Florida International University, Miami, FL 33174, USA

$^{4}$ Advanced Science Research Center, City University of New York, New York, NY 10031, USA

$^{5}$ Physics Program, Graduate Center, City University of New York, New York, NY 10016, USA

## S1. In situ measurement of the CPA

In the main text of our manuscript, we have shown experimental results confirming the existence of CPA states in electronic circuits based on the measured S-matrix. For completeness, we also perform an in situ measurement of a CPA state with coherent inputs.

We analyze the circuit shown in Fig. S1 (a), the same as Fig. 2 in the main text. Since our circuit is mirror-symmetric, the coherent waveform corresponding to the CPA eigenvector is simply two signals of the same frequency, amplitude, and phase difference of 0 or $\pi$ , depending on the transmission line length (symmetric or antisymmetric CPA). The analytic condition for a symmetric CPA is shown in Fig. S1 (b), and we choose d=0 (no transmission line – shown with the yellow dot) as the simplest case, where we expect to observe a CPA with two in-phase input signals.

The experimental setup is shown in Fig. S1 (c). In the circuit scenario, two signal generators are connected at the two ports of the circuit, and an oscilloscope is connected at the node $V_{m}$ which is our measurement point. In our analysis, the connected ports have a characteristic impedance of 50 $\Omega$ . Thus, the generator impedance needs to be set to $R_{g}=50\ \Omega$ . For the circuit, we use $R=50\ \Omega$ (consisting of a 43 $\Omega$ resistor connected in series with a 0-10 $\Omega$ trimmer), L=1500 nF, and C=225 pF in both resonant circuits, with the resulting resonant frequency around 8.66 MHz.

The input waveform is chosen according to the previous discussion - two signals of the same frequency and amplitude are selected on the signal generator $(|V_{g}| = 0.5\mathrm{V})$ , while the second generator's phase is varied. As mentioned, two signals need to be in phase for the excitation of the symmetric CPA in this circuit.

(a)
![](images/d8e46a04a5d206663d379d776cbe5149877b4e1af731ac5dfee4582d6cac8a00.jpg)

(b)
![](images/aa694ef2ec9b2f303d0d95aa592f65ee15759a1d5868ee86b6dbc6fa4ac8e393.jpg)

![](images/f9d9657f9ed3c3c90461354dce3189fd6fb23fabf0f3f7299efcce5e6ebd14b0.jpg)

(d)
![](images/fcce7734e4aca74b1751cbe3305532dc07b41dfd1437da0350bfd0ace508e821.jpg)

(e)
![](images/29ac3847d54953fef7dc29dbc77407120172cf65a21314fcd19c369224de654a.jpg)

![](images/72383f618e8456a0eb4669e802c8176812035fd0c416e7b0e1c1266df4680dae.jpg)

![](images/3ca90af4a27d9b549a5c8fa88235b8a3de2824ccf420bb51474181c4a7d955a4.jpg)
Figure S1. (a) The CPA circuit schematic. (b) Analytic condition for the symmetric CPA. (c) Measurement setup with two signal generators and an oscilloscope. (d) Equivalent circuit at resonance. (e) Measured and simulated voltage $|V_{m}|$ at resonance for input signals $V_{g}=0.5\sin(\omega t)$ V with varied phase difference. (f) Measured and simulated output-to-input power ratio with varied phase difference.

We connect an oscilloscope to the node and measure $V_{m}$ for different phase inputs, as shown in Fig. S1 (e). Since the input/output ports are shorted (one node), $V_{m}$ contains all necessary information. The discrepancy between experimental and simulation results is present due to the imperfection of the measurement setup and equipment. For example, the two generator impedances are most likely not exactly equal to each other and could be slightly detuned from the 50-ohm value. Parasitic inductances and capacitances of the setup can also affect the impedance matching conditions, causing deviations from predicted behavior. Nevertheless, the experimental results are in excellent agreement with simulations.

From the measured $V_{m}$ , we can deduce the CPA using the following rationale based on the dissipated power argument. According to [1], an arbitrary loaded transmission line is perfectly matched to the generator when $Z_{g}=Z_{in}^{*}$ , where $Z_{g}$ is the generator impedance and $Z_{in}$ is the input impedance seen from the generator – this is the conjugate matching condition (maximum power transfer), i.e., the maximum available power is transmitted to the loaded transmission line. On the other hand, a zero-reflection condition is achieved when $Z_{g}=Z_{in}$ , as the reflection coefficient is given by $r=(Z_{in}-Z_{g})/(Z_{in}+Z_{g})$ . In general, these two conditions are different. However, when the generator impedance is real, these two conditions become equivalent [1]. In our case, $R_{g}$ is real, and thus we know that there is no reflected (outgoing) power from the system when all the available power from the generator is dissipated in our circuit, i.e., this will be the signature of the CPA. Since our generator has $R_{g}=50$ Ohm, the maximum power available, determined by the generator impedance, is given by $P_{maxI}=|V_{g}|^{2}/(8R_{g})$ , while that power is fully transferred to the loaded transmission line when $R_{in}=R_{g}=50$ Ohm.

At the resonant frequency, L and C become short circuits, and the equivalent circuit is simplified to Fig. S1 (d). Let us first analyze the circuit when just one generator is active as a source and the other port is closed with $Z_{0}=R_{g}=50\ \Omega$ . The input impedance the generator sees is $Z_{in}=R/3$ , and the reflection coefficient is r=-1/2. With just one input signal, a quarter of the available power is reflected back to the generator due to the impedance mismatch; hence, there is no CPA available. Let us now consider the case with both generators active – the maximum available power (input power) to be delivered is now $P_{max}=|V_{g}|^{2}/(4R_{g})$ .

If all this power is dissipated in our circuit, i.e., at the resistor $R_{eq}=0.5R$ , there will be no outgoing power. The power dissipated at the $R_{eq}$ is $P_{r}=|V_{m}|^{2}/2R_{eq}=|V_{m}|^{2}/R$ . This means that outgoing power is zero when $|V_{m}|=|V_{g}|/2$ , which is confirmed by voltage measurements when two signals are in phase. To put this into the context of CPA, we also plot the output-to-input power ratio as $P_{out}/P_{in}=1-P_{R}/P_{in}$ in Fig. S1 (f).

The CPA can also be confirmed with our measured S-parameters of the same circuit, Fig. S2. Total outgoing scattering is zero, as the scattering coefficients r and t are equal in amplitude with phase difference $\pi$ between them at the CPA frequency. For in-phase excitation at both ports, we have $P_{out}/P_{in} = (|a_{1}r_{L} + a_{2}t|^{2} + |a_{2}r_{R} + a_{1}t|^{2}) / (|a_{1}|^{2} + |a_{2}|^{2}) = 0$ , as $a_{1} = a_{2} = \cos(\omega t)$ and $r_{l} = r_{r}$ .

![](images/f1775aee68a156a860e0bdef928855ca36096b113ada57f86b284ddd5d9688c7.jpg)
Figure S2. (a) Measured S-parameters of the circuit with total outgoing power calculated from the S-parameters.

## S2. Topological charge creation in optical structures

As mentioned in the main text, the effects of BIC splitting into topological CPA and CPA-laser charges can be realized in many different systems. To substantiate the claimed ubiquity of the proposed phenomenon, we analyze two examples of optical systems – a tri-layer structure consisting of two $\alpha$ -MoO $_{3}$ layers separated by a dielectric spacer, and a pair of Ag-nanosphere metasurfaces, also separated by a dielectric spacer. Firstly, we analyze the optical tri-layer – we consider normally incident wave polarized in the y-direction, Fig. S3 (a). The permittivity of the material along y-direction is shown in Fig. S3 (b), according to [2]. The material has a Lorentzian or epsilon-near-pole (ENP) resonance at the transverse optical phonon frequency $\omega_{TO}$ , which is necessary for obtaining the BIC condition [3]. If losses in the material are artificially turned to zero, this structure supports a BIC, Fig. S3 (c, left). The spacer thickness BIC condition is given by equation (1) in the main text. Now, if the real losses are “turned back” on, the BIC is split into two zeros of the eigenvalue i.e., two CPAs, Fig. S3(c, middle and right). Further manipulation and annihilation of these charges is possible with different thicknesses of the $\alpha$ -MoO $_{3}$ layers, similar to the discussions in our previous work [3]. Finally, we analyze the PT-symmetric version of the system, where two CPA-laser charges emerge around the BIC, Fig. S3 (d). We note here that the same phenomena can be obtained with any materials with ENZ or ENP resonances, such as SiC, hBN, or InAs, to name a few.

(a)
![](images/b3d17e258993fe3c9501fd0e43f16efbc6e4f4bf27e247f86899b9daa0f55f07.jpg)

(b)
![](images/8e9769cd87e3b7392cfb1d8f6add5c3a2abb1f68160f56922aaf92164511654c.jpg)

(c)
![](images/e89daf190d7fbe00dceaf13a54baf0d7600f558e8e4a43758ddf591e043fadef.jpg)

(d)
![](images/b7d2fdc17ea244ff7d2b4307595051669b6e50a141b9132c046fcf259cb9d97d.jpg)
Figure S3. (a) Optical tri-layer structure with normally incident, y-polarized monochromatic wave, with dielectric permittivity $\varepsilon_{d}=5$ . (b) Permittivity function $\varepsilon_{y}$ of $\alpha$ -MoO $_{3}$ . (c) Eigenvalues phase for lossless (left) and real (middle) $\alpha$ -MoO $_{3}$ , where BIC and CPAs are visible. Eigenvalue $|s_{l}|=|r-t|$ amplitude (right). (d) PT-symmetric structure with CPA-laser charges emerging from BIC.

(a)
![](images/14cbabfd89a00e09b7c8865c83788fd70e325e9a978689efdbe80ac85ef65261.jpg)
(c)

(b)
![](images/200552edf89d52158516e1ce1ed2ded1c1c6f99cb280f38b4cccf48df493a460.jpg)

![](images/6ab47f85ccdb70e6cef82944d88881f61f09a11b697d1e19ef2711eee26b3a41.jpg)

![](images/ab2f821c434109258e0ce18e18ca43bcd64d37b50cb5c7d980818235c95d28a6.jpg)
(d)

![](images/e2c1e8d0d742d972f40747b331775000f01ce0b894108d8f0ff9f5f02081f930.jpg)

![](images/d696c958ae445124377b34dd07db7d659704e99750562bd35bc09c3b729fcb30.jpg)
Figure S4. (a) Two Ag-nanoparticle metasurfaces separated by a spacer, with obliquely incident, y-polarized monochromatic wave. (b) Effective metasurface conductivity. (c) Eigenvalues phase for lossless (left) and real (middle) Ag metasurface where BIC and CPAs are visible. Eigenvalue $|s_{l}|=|r-t|$ amplitude (right). (d) PT-symmetric structure with CPA-laser charges emerging from BIC.

On the other hand, the Lorentzian resonance can be effectively induced by metallic or dielectric metasurfaces utilizing the electric dipole resonance. To that end, we analyze a pair of silver nanoparticle metasurfaces as shown in Fig. S4 (a), and previously analyzed in [4]. The diameter of the nanospheres considered is $r = 15$ nm, while the pitch is $l = 35$ nm. In this regime, only specular reflection is allowed, while no diffraction channels are open [4]. The effective conductivity induced by the metasurfaces is shown in Fig. S4 (b), which was calculated according to the analytical solution given in [4]. As shown in Fig. S4 (c,d), the previously discussed phenomenon of BIC splitting into two CPA or CPA-laser solutions is shown. We thus note that the discussed topological nature of BICs and S-matrix singularities concerns a wide range of materials and metasurface systems. Furthermore, the same effects are expected to be found in acoustics and mechanics, given the ubiquitous wave nature of the proposed framework.

Although the simplified RLC model we use in the main text is not an exact representation of the optical structures discussed here, these systems behave the same from the topological perspective. This analogy can be extended and made more precise to include loss from random disorder and imperfections in metasurfaces, as these can be well modelled by parasitic resistances in the circuit model. Thus, even more complex behavior of optical structures could be modelled and experimentally verified with RLC circuits.

## S3. CPA condition

Here we derive the analytical solution for CPAs shown in the main text in equations (3) and (4). Using the ABCD matrix formalism [1], the total matrix of the circuit analyzed in Fig. S5 (inset of Fig. 2(a) in the main text) can be written as:

$$
\begin{array}{l} M _ {T} = M _ {R} M _ {T L} M _ {R} = \left[ \begin{array}{c c} 1 & 0 \\ \frac {1}{Z _ {R}} & 1 \end{array} \right] \left[ \begin{array}{c c} \cos (k d) & j Z _ {0} \sin (k d) \\ \frac {j}{Z _ {0}} \sin (k d) & \cos (k d) \end{array} \right] \left[ \begin{array}{c c} 1 & 0 \\ \frac {1}{Z _ {R}} & 1 \end{array} \right] = \\ = \left[ \begin{array}{c c} \cos (k d) + j \frac {Z _ {0}}{Z _ {R}} \sin (k d) & j Z _ {0} \sin (k d) \\ \frac {2}{Z _ {R}} \cos (k d) + \frac {j}{Z _ {0}} \sin (k d) \left(\frac {Z _ {0} ^ {2}}{Z _ {R} ^ {2}} + 1\right) & \cos (k d) + j \frac {Z _ {0}}{Z _ {R}} \sin (k d) \end{array} \right] = \left[ \begin{array}{c c} A _ {T} & B _ {T} \\ C _ {T} & D _ {T} \end{array} \right] \end{array}\tag{s1}
$$

The reflection and transmission coefficients are then calculated as:

$$
r = \frac {A _ {T} + B _ {T} / Z _ {0} - C _ {T} Z _ {0} - D _ {T}}{A _ {T} + B _ {T} / Z _ {0} + C _ {T} Z _ {0} + D _ {T}}\tag{s2}
$$

$$
t = \frac {2}{A _ {T} + B _ {T} / Z _ {0} + C _ {T} Z _ {0} + D _ {T}}.\tag{s3}
$$

We next set the eigenvalues to zero $s_{1/2} = t \pm r = 0$ . For the anti-symmetric case we have $s_{1} = t - r = 0$ , and we solve for $Z_{R}$ :

$$
2 Z _ {R} ^ {2} + 2 Z _ {0} \cos (k d) Z _ {R} + j Z _ {0} ^ {2} \sin (k d) = 0,\tag{s4}
$$

which gives two complex solutions for $Z_{R}$ :

$$
Z _ {R a 1 / 2} = Z _ {0} \frac {- \cos (k d) \pm \sqrt {\cos^ {2} (k d) - 2 j \sin (k d)}}{2}.\tag{s5}
$$

Only one of these solutions gives a positive real part for the impedance $Z_{R}$ , so we choose that solution (plus sign). On the other hand, symmetric case $s_{1} = t + r = 0$ leads to two solutions:

$$
Z _ {R s 1 / 2} = Z _ {0} \frac {\cos (k d) \pm \sqrt {\cos^ {2} (k d) + 2 j \sin (k d)}}{2}\tag{s6}
$$

Similarly, we choose the plus sign to obtain the positive real part solution for $Z_{R}$ . Equations s5 and s6 lead to final equations (3-4) and Fig. 2(a-b) in the main text.

$$
\begin{array}{c} Z _ {0} \quad Z _ {0} \quad Z _ {0} \\ 1 \quad \leftarrow d \quad 2 \\ Z _ {R} \quad Z _ {R} \end{array}
$$

Figure S5. Transmission line circuit separated by two identical complex impedances that supports BIC and CPAs.

## S4. CPA-laser condition

To find CPA-laser solutions of the PT-symmetric circuit from Fig. 4 in the main text, we need to derive the conditions for zeros and poles of the eigenvalues $s_{1/2} = t \pm \sqrt{r_l r_r}$ . Since the system is PT-symmetric, any real-frequency pole will be collocated with a real-frequency zero, thus only finding the pole dispersion is sufficient. The left (loss) and right (gain) resonators can be represented as complex admittances $Y_1 = I / Z_1$ and $Y_2 = I / Z_2$ , such that PT-symmetry holds $Z_1 = -Z_2^*$ , where the complex impedances are given by $Z_1 = R + jX, Z_2 = -R + jX$ , and $R$ and $X$ are real numbers $R, X \in \mathbb{R}$ . Since the reflection coefficients are different from different ports, we calculate the left and right total ABCD matrices, as well as reflection and transmission coefficients, as:

$$
\begin{array}{r l} & M _ {T l} = M _ {l o s s} M _ {T L} M _ {g a i n} = \left[ \begin{array}{c c} 1 & 0 \\ Y _ {1} & 1 \end{array} \right] \left[ \begin{array}{c c} \cos (k d) & j Z _ {0} \sin (k d) \\ \frac {j}{Z _ {0}} \sin (k d) & \cos (k d) \end{array} \right] \left[ \begin{array}{c c} 1 & 0 \\ Y _ {2} & 1 \end{array} \right] = \\ & \qquad = \left[ \begin{array}{c c} \cos (k d) + j Z _ {0} Y _ {2} \sin (k d) & j Z _ {0} \sin (k d) \\ \cos (k d) (Y _ {1} + Y _ {2}) + \frac {j \sin (k d)}{Z _ {0}} (1 + Z _ {0} ^ {2} Y _ {1} Y _ {2}) & \cos (k d) + j Z _ {0} Y _ {1} \sin (k d) \end{array} \right] \\ & \qquad = \left[ \begin{array}{c c} A _ {T l} & B _ {T l} \\ C _ {T l} & D _ {T l} \end{array} \right] \end{array}\tag{s7}
$$

$$
\begin{array}{r l} & M _ {T r} = M _ {g a i n} M _ {T L} M _ {l o s s} = \left[ \begin{array}{c c} 1 & 0 \\ Y _ {2} & 1 \end{array} \right] \left[ \begin{array}{c c} \cos (k d) & j Z _ {0} \sin (k d) \\ \frac {j}{Z _ {0}} \sin (k d) & \cos (k d) \end{array} \right] \left[ \begin{array}{c c} 1 & 0 \\ Y _ {1} & 1 \end{array} \right] = \\ & \qquad = \left[ \begin{array}{c c} \cos (k d) + j Z _ {0} Y _ {1} \sin (k d) & j Z _ {0} \sin (k d) \\ \cos (k d) (Y _ {1} + Y _ {2}) + \frac {j \sin (k d)}{Z _ {0}} (1 + Z _ {0} ^ {2} Y _ {1} Y _ {2}) & \cos (k d) + j Z _ {0} Y _ {2} \sin (k d) \end{array} \right] \\ & \qquad = \left[ \begin{array}{c c} A _ {T r} & B _ {T r} \\ C _ {T r} & D _ {T r} \end{array} \right] \end{array}\tag{s8}
$$

$$
r _ {l} = \frac {A _ {T l} + B _ {T l} / Z _ {0} - C _ {T l} Z _ {0} - D _ {T l}}{A _ {T l} + B _ {T l} / Z _ {0} + C _ {T l} Z _ {0} + D _ {T l}} = \frac {q _ {l}}{p _ {l}} = \frac {q _ {l}}{p}\tag{s9}
$$

$$
r _ {r} = \frac {A _ {T r} + B _ {T r} / Z _ {0} - C _ {T r} Z _ {0} - D _ {T r}}{A _ {T r} + B _ {T r} / Z _ {0} + C _ {T r} Z _ {0} + D _ {T r}} = \frac {q _ {r}}{p _ {r}} = \frac {q _ {r}}{p}\tag{s10}
$$

$$
t _ {r} = \frac {2}{A _ {T l} + B _ {T l} / Z _ {0} + C _ {T l} Z _ {0} + D _ {T l}} = \frac {2}{p}\tag{s11}
$$

As required by reciprocity, the transmission coefficient is equal from both sides, i.e., the denominators are equal in equations (s9-s11) - $p_l = p_r = p$ . For brevity purposes, we write the numerators of reflection coefficients as $q_l$ and $q_r$ , and denominator as $p$ . The eigenvalues are then given by:

$$
s _ {1 / 2} = t \pm \sqrt {r _ {l} r _ {r}} = \frac {2 \pm \sqrt {q _ {l} q _ {r}}}{p}\tag{s12}
$$

The pole condition requires that the denominator p is equal to 0, which gives the following equation:

$$
\big (2 + Z _ {0} (Y _ {1} + Y _ {2}) \big) e ^ {j k d} + j \sin (k d) Z _ {0} ^ {2} Y _ {1} Y _ {2} = 0.\tag{s13}
$$

When $Y_{1}=1/Z_{1}=1/(R+jX)$ and $Y_{2}=1/(-R+jX)$ are inserted in equation (s13), the following equation can be obtained:

$$
R ^ {2} + X ^ {2} + j Z _ {0} X = \frac {Z _ {0} ^ {2}}{2} \frac {1}{1 - j \cot (k d)}.\tag{s14}
$$

As $R$ and $X$ are real numbers, we can equate the real and imaginary parts of the left and right sides of the equations(s14) as:

$$
Z _ {0} X = i m a g \left(\frac {Z _ {0} ^ {2}}{2} \frac {1}{1 - j \cot (k d)}\right),\tag{s15}
$$

$$
R ^ {2} + X ^ {2} = r e a l \left(\frac {Z _ {0} ^ {2}}{2} \frac {1}{1 - j \cot (k d)}\right).\tag{s16}
$$

Since $Z_{0}$ is a real number, we can write the solution for the imaginary part of the complex impedance as:

$$
X = \frac {Z _ {0}}{2} i m a g \left(\frac {1}{1 - j \cot (k d)}\right).\tag{s17}
$$

After some trigonometric manipulation, this is further simplified to:

$$
X = - \frac {Z _ {0}}{4} \sin (2 k d).\tag{s18}
$$

Similarly, equation (s16) is simplified to:

$$
R = \frac {Z _ {0}}{\sqrt {2}} \sqrt {\sin^ {2} (k d) - \frac {\sin^ {2} (2 k d)}{8}}.\tag{s19}
$$

The last two equations represent the complete CPAL solution shown in the main text.

## S5. Asymmetric CPA

In the main text, we discuss CPA in mirror-symmetric circuits. However, the CPA can also be induced in asymmetric circuits which can change the absorption linewidth characteristics substantially. To provide an example of this behavior, we analyze the “annihilation point” CPA, Fig. S6 (circuit from Fig. 3 and Fig. 2 (f) in the main text). This CPA exists for $d=0.5\ \lambda_{0}$ and $R_{1}=R_{2}=50\ \Omega$ . However, if we change the left and right resistor in the resonant circuits such that $R_{1}\|R_{2}=25\ \Omega$ , the CPA remains albeit with different absorption linewidth characteristic. This effect on the CPA is even more pronounced when using different combinations of inductors and capacitors in resonant circuits which preserve the same resonant frequencies. For example, using a coefficient k, we can write $L_{1}=L$ , $C_{1}=C$ , $L_{2}=L/k$ , $C_{2}=kC$ . These two effects can provide a wide range of asymmetric CPA linewidths – three orders of magnitude difference in the example shown in Fig. S6.

![](images/99ac382ec161d4393d6b33c925a11c5b1d77df90010657d6fa8c24a4ef00729c.jpg)
Figure S6. Asymmetry provides three orders of magnitude difference in absorption linewidth at CPA, with $d=0.5\ \lambda_{0}$ and $Z_{0}=50\ \Omega$ .

An optical analog to these asymmetric scenarios would use different thicknesses of the top- and bottom-resonant layers, with similar effects.

## S6. Stability and complex frequency analysis

As mentioned in the main text, stability is an important aspect of the PT-symmetric systems analyzed in this paper. As discussed with Fig. 1 in the main text, PT-symmetric perturbation separates the BIC into a pair of CPA-laser states, which act like topological charges of $\pm2$ . Due to the pole dispersion crossing to the upper complex half-plane, the region between two CPA-laser charges originating from the same BIC is necessarily unstable. We continued the discussion in Fig. 4, where annihilation of CPA-laser states was shown. Here we additionally discuss the same circuit with the imaginary frequency dimension, and we plot the dispersion of S-matrix poles and zeros (red and blue lines), Fig. S7.

![](images/401e9f5f118a7b879515ee5ec61c9973b261e3398967c2cc9245a987470cdd93.jpg)

(b)
![](images/1709ed53652c5ac6da391ba42c6d3f3ad1b2e89512995e5a02dad0a22b273d23.jpg)
(d)

(c)
![](images/0b072ec709ed1a861904f358850d148ae5b5b322776488b9672e95283ce5cbd5.jpg)

![](images/54916c1631938d06748513f814730b3f54f8da1effd6fb5abe88b5be6556cb74.jpg)
Figure S7. S-matrix pole and zero dispersion of the circuit from Fig. 4 in the main text. Solutions were found in the parameter space of $d-\omega_{i}-X$ for four different $|R|$ values (a-d). Red-shaded parts of the graphs represent the unstable regions.

In the lossless case $(R=0)$ , the BICs are located at 0 and $0.5\ d/\lambda_{0}$ transmission line thicknesses, Fig. S7 (a). As CPA-laser charges emerge out of neighboring BICs for R>0, the poles cross to the positive imaginary frequency space, indicating instability. As R increases to $R=50/\sqrt{2}$ and the two charges are collocated (at annihilation point), the entire pole dispersion is in the upper complex half-plane, Fig. S7 (c), meaning that there is no stable solution in the entire parameter space. As R increases further, there are no CPA-laser solutions left, poles move higher up, and the system remains unstable, Fig. S7 (d).

It should be noted that the fixed-frequency framework in which this analysis is done does not consider the dispersion of the negative resistance, which is unavoidable in real circuits due to causality. Although the creation, dispersion, and associated instabilities of CPA-laser degeneracies are thoroughly explained with this analysis, it should be used synergically with any dispersions present in real circuits to fully assess the system stability.

## S7. Analogy between transmission lines used in theoretical analysis and T-circuits used in realistic and experimental circuits

In the theoretical analysis of CPA and CPAL states in the main text, we have used a two-port circuit consisting of two shunt RLC resonators separated by a transmission line, Fig. (1,2,4). For practical reasons and experimental purposes, we demonstrated the same effects using a double T-circuit consisting of multiple LC tanks instead of a transmission line, in Figs. (2,5) in the main text. Due to the large physical length of transmission lines at discussed frequencies ( $\lambda_0/2 \approx 17$ m at $f_0 = 8.8$ MHz), a lumped element LC circuit can replace the transmission line as we show next. To validate this analogy, we plot the phase and amplitude response of the two circuits shown in Fig. S8. The double T-circuit in Fig. S8 (b) acts as a low-pass filter, whose response in the passband is very similar to that of the transmission line with length $d = \lambda_0/2$ , and is most similar around $f = f_0$ , where the topological properties were tested in real circuits.

(a)
![](images/39d66c81a92adb50bba7ff4c917b39415a012ea05dc033f81287da522bcb64ea.jpg)

(b)
![](images/e1c661cd944c475d673c348fc2cf8b3b94754b2d581eb97c942fb17e9cf31f9a.jpg)

(c)
![](images/f1f50ca810bb32c93fc251cacd206358c0cb992ebe7cf82294b76dce94990428.jpg)

(d)
![](images/f90f467c4bddaa92ead995059656952c15ec727539ae0e1e3b394eebaf8c929b.jpg)
Figure S8. Phase and amplitude of the transmission coefficient comparison between the (a) transmission line and (b) double T-circuit.

## S8. Negative resistance dispersion

In our simulations of PT-symmetric circuits, we have used a SPICE model for the operational amplifier LMH6714 by Texas Instruments, operating in the negative impedance converter configuration. The simulated effective negative resistance is shown in the Fig. S9.

![](images/07c03db1e941470235ad524c88f5c269b68f4acb5909fd2f1b2b855ac0431288.jpg)

![](images/9cc3442eb2d8b4c2a0e81433a65e342313f3b78fb78ba6c892f8ce56fb41c7aa.jpg)
Figure S9. Simulated negative impedance dispersion (real and imaginary parts) of the operational amplifier LMH6714 in the negative-impedance-converter configuration. $R_{2}=50\ \Omega$ , $R_{3}=R_{4}=250\ \Omega$

## References

[1] Pozar, D.M. Microwave engineering, 4th Edition. John Wiley & Sons, (2011).

[2] Z. Zheng, N. Xu, S. L. Oscurato, M. Tamagnone, F. Sun, Y. Jiang, Y. Ke, J. Chen, W. Huang, W. L. Wilson, A. Ambrosio, S. Deng, and H. Chen, A Mid-Infrared Biaxial Hyperbolic van Der Waals Crystal, Sci. Adv. 5, 1 (2019).

[3] Z. Sakotic, A. Krasnok, A. Alú, and N. Jankovic, Topological Scattering Singularities and Embedded Eigenstates for Polarization Control and Sensing Applications, Photonics Res. 9, 1310 (2021).

[4] A. Krasnok and A. Alu, Embedded Scattering Eigenstates Using Resonant Metasurfaces, J. Opt. 20, (2018).
