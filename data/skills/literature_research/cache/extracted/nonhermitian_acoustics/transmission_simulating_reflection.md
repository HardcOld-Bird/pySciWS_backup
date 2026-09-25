# Tunable nonlocal purely active nonreciprocal acoustic media

Nathan Geib $^{1,*}$ , Aritra Sasmal $^{1}$ , Zhuzhu Wang, $^{1}$ Yuxin Zhai $^{1}$ , Bogdan-Ioan Popa $^{1}$ , and Karl Grosh $^{1,2}$

$^{1}$ Department of Mechanical Engineering, University of Michigan, Ann Arbor, Michigan 48109, USA

$^{2}$ Department of Biomedical Engineering, University of Michigan, Ann Arbor, Michigan 48109, USA

![](images/748be39924bb51e5dc9c1881f08b806eee46ddf25e4933b05650479033c0256a.jpg)

(Received 21 February 2020; revised 15 January 2021; accepted 30 March 2021; published 27 April 2021)

Previously engineered wave-bearing materials including metamaterials generate their acoustic responses from a combination of passive geometrical structures possibly augmented by active elements. The passive geometrical features typically translate to uncontrolled properties outside the band of interest and are the source of adverse effects such as narrowing the operating band significantly or generating difficult-to-control instabilities due to interactions with the active elements. This paper proposes a nonlocal active metamaterial (NAM) architecture that employs a purely active approach to tailoring the acoustic field utilizing spatially separated sensors and actuators. A significant benefit of this approach is that the acoustic properties are described by analytical closed-form equations exposing the complex interplay between the various components of the material. The method is demonstrated by engineering extremely nonreciprocal wave transmission in an otherwise completely transparent subwavelength NAM. We experimentally demonstrate the remarkable flexibility of the system by electronically tuning the acoustic isolation characteristics and discuss the massive design space uncovered by the NAM approach.

DOI: 10.1103/PhysRevB.103.165427

## I. INTRODUCTION

Engineered materials seek to manipulate the acoustic field in ways that homogeneous, natural materials cannot. Passive examples range from the simple, such as a quarter-wavelength side branch, to the complex, such as certain types of three-dimensional metamaterial cloaks $[1]$ . While these engineered materials may achieve remarkable in-band frequency-response design goals, their response is limited by Kramers-Kronig type of constraints and typically require resonances to produce the prescribed functionality $[2]$ . Active elements (i.e., sensors and sources) added to the metamaterial can in principle correct the scattering characteristics of the passive structure and remove the constraints of passivity by injecting energy into the impinging wave. However, the complex near-field interactions between the active elements and the geometrical features of the metamaterial may create conditions such that it is difficult to achieve a stable system without exquisite control of the geometry and active control parameters. In many scenarios this may lead to a drastic reduction in useful bandwidth $[3]$ . In this paper, we propose a metamaterial architecture in which the effective acoustic properties are determined entirely by active elements and in which geometrical features play no role in setting the acoustic behavior. We illustrate the concept by studying a nonlocal, active material whose critical dimensions are subwavelength. We develop the theory and experimental validation of this theory as applied to design a highly nonreciprocal acoustic waveguide.

Interest in nonreciprocal acoustics has grown substantially in recent years due to their potential applications in the biomedical, noise control, and communication industries $[4,5]$ . Acoustic reciprocity is a fundamental property of passive linear wave-bearing media, requiring that a signal transmitted from a source to a receiver will be unaffected by switching the source and receiver locations $[6]$ . This trait is remarkably robust, holding in systems with a variety of material properties and configurations (e.g., lossy viscoelastic media $[7]$ , in the presence of scatterers, or fluid-structure interactions $[8]$ ). However, acoustic metamaterials have proven capable of breaking reciprocity using a variety of mechanisms through the careful design of subwavelength structural and dynamical properties. For instance, combinations of nonlinearities, biasing of the background media, and phononic band-gap materials $[3,9–14]$ have been used to elicit directionally dependent wave propagation in passive metamaterials. Active metamaterials have been developed with the aim of achieving exceptional control over wave propagation, particularly in the subwavelength regime. These strategies include, but are not restricted to, spatiotemporal modulation of material properties $[15–18]$ , and real-time boundary impedance control using a distributed network of sensors and electrodynamical transducers $[19]$ .

Most of these methods suffer from one or more of the following limitations: the introduction of unwanted harmonic tones that require subsequent demodulation, narrow-band functionality (meaning reciprocity is broken over a very limited range of frequencies), or disruption of mean flow in fluid systems. Here, we propose an approach that overcomes these limitations and breaks reciprocity by relying on the spatial separation of sensors and transducers to create a nonlocal active metamaterial (NAM). We demonstrate experimentally how a single NAM unit cell is capable of generating large, broadband, subwavelength nonreciprocity that outperforms existing techniques. We show how the frequencies at which this behavior is centered can be selected by appropriate tuning of the controlling electronics, providing flexibility in performance design.

![](images/ed9c3eada4ad3aaa07d07414a2500d7c1a266371ec19eb386bb999946348eea6.jpg)
FIG. 1. Schematic of the NAM unit cell in a one-dimensional acoustic domain. The pressure source at location $x_{s}$ emits a pressure equal to $G(\omega)P(x_{p})$ , where $G(\omega)$ is the controller gain and $P(x_{p})$ is the pressure measured by a probe a distance $\delta x$ away from the source at $x_{p}$ . Waves entering the system (black) are either attenuated (dark blue) or amplified (light red) depending on their direction of incidence.

## II. IDEAL SYSTEM MODEL

The NAM unit cell represents a fundamental building block of a promising class of nonreciprocal active metamaterials [20]. As shown in Fig. 1, where we have exemplified the NAM concept using a fluid (or airborne) acoustic medium, the NAM unit cell consists of a noncollocated sensor-source pair. Such an approach can also be applied to a variety of wave-bearing media, including piezoelectric or elastic rods, interdigital surface acoustic wave transducers, and dispersive waveguides (e.g., bending waves in structures). The NAM unit cell and systems used for active noise cancellation (ANC) share similar configurations. However, the goal of the NAM system is to create a stable active unit cell with local material properties that provide significant nonreciprocity without resorting to a centralized control system or the use of an error microphone [21]. In the NAM system in Fig. 1, the acoustic pressure sensed at $x_{p}$ acts as the input to a sound source located at $x_{s}$ , a distance $\delta x$ away from the sensor. The voltage from the pressure sensor is multiplied by an electronically controlled gain so that the effective amplitude of the acoustic point source at $x = x_{s}$ is $G(\omega)P(x_{p})$ , where the acoustic pressure is assumed to be harmonic in time, taking the form $p(x,t) = P(x)e^{j\omega t}$ . Under this model, the pressure $P$ at any location $x$ is governed by a modified version of the Helmholtz equation,

$$
\frac {d ^ {2} P (x)}{d x ^ {2}} + k ^ {2} P (x) = G (\omega) P (x _ {p}) \delta (x - x _ {s}),\tag{1}
$$

where $k = \omega/c$ , $\omega$ is the angular frequency, and c is the acoustic speed in the medium. In the NAM system, the signal sensed at $x_{p}$ is a function of both incoming disturbances as well as the feedback from the source at $x_{s}$ . Since most ANC systems aim to cancel broadband noise rather than to synthesize novel material properties, ANC systems are adaptive in nature and feature a multitude of strategies for preventing source feedback from corrupting the signal at the sensor [21–26].

![](images/a634865ccf24deec46fbe6157133aeb3ed6967f7115997ed62e00dd6d46096e0.jpg)

![](images/7a57a13311bc86f17f38c1558e0346f98bbed3c0f010280e5a0c252a17964851.jpg)
FIG. 2. Transmission behavior of the NAM unit cell. (a) Directional transmission coefficients across the NAM unit cell shown in Fig. 1 for $\widetilde{G}=0.5\pi$ . (b) Isolation factors for three $\widetilde{G}$ values demonstrate the tunable nature of the NAM approach.

We are interested in plane-wave transmission through the NAM unit cell as characterized by the off-diagonal elements of the scattering matrix [27], denoted by $\widetilde{S}$ , which is given by

$$
\widetilde {\boldsymbol {S}} = \left[ \begin{array}{c c} S _ {1 1} & S _ {1 2} \\ S _ {2 1} & S _ {2 2} \end{array} \right] = \frac {1}{\widetilde {\Delta}} \left[ \begin{array}{c c} - \widetilde {G} \frac {e ^ {j k \delta x}}{2 j k \delta x} & 1 \\ (1 - \widetilde {G} \operatorname{sinc} k \delta x) & - \widetilde {G} \frac {e ^ {- j k \delta x}}{2 j k \delta x} \end{array} \right],
$$

where

(2)

$$
\widetilde {\Delta} = 1 + \widetilde {G} \frac {e ^ {- j k \delta x}}{2 j k \delta x},\tag{3}
$$

and $\widetilde{G} = G(\omega)\delta x$ is the dimensionless gain factor. The scattering matrix depends on two nondimensional parameters, the gain factor $G$ and the frequency parameter $k\delta x$ . While the nonreciprocity of this system is evident as $S_{12} \neq S_{21}$ in general, we formally prove that reciprocity is broken in the Supplemental Material [28]. We also show in the Supplemental Material [28] how to use the $S$ -parameter matrix to develop the effective material properties [29] for this unit cell following an approach used in electromagnetics [30]. While these effective material properties reflect the nonreciprocity of the system, such effective properties are only considered accurate at low frequencies ( $k\delta x \ll 1$ ). Hence, we elect to use the full $S$ -parameter expansion to consider system behavior over a larger frequency range. In Fig. 2(a), the transmission coefficients from PORT 1 to PORT 2 [given by $20\log(|S_{21}|)$ ] and from PORT 2 to PORT 1 [ $20\log(|S_{12}|)$ ] are plotted over a range of $0 < k\delta x < \pi$ (or for which the wavelength $\lambda > 2\delta x$ ) using $\widetilde{G} = 0.5\pi$ , a gain factor resulting in stable solutions. For stability in the unbounded acoustic medium, the roots of $\widetilde{\Delta}$ must lie in the upper half plane of the complex frequency space. The stability boundary for a constant gain factor is $0 < \widetilde{G} < \pi$ .

The transmission coefficients are dramatically different over nearly the entire frequency range, converging only at the special frequencies where the system is reciprocal (i.e., when $S_{21} = S_{12}$ ). These frequencies occur when $k\delta x = n\pi$ ( $n \in Z^{+}$ ). At these frequencies, $\delta x$ is an integer multiple of $\lambda/2$ and $\operatorname{sinc}(k\delta x) = 0$ . At high frequencies ( $k\delta x \gg \pi$ ), the system largely reverts to a passive waveguide, where $S_{11} = S_{22} = 0$ and $S_{12} = S_{21} = 1$ . The most dramatic difference in transmission occurs when $S_{21} = 0$ , where a wave incident from PORT 1 is exactly canceled by the pressure emitted from the source. Note that this cannot occur for waves incident from PORT 2, because if $S_{12}$ could be zero, this would imply that the pressure at the probe $[P(x_{p})]$ would also be zero. Since the source strength is proportional to $P(x_{p})$ , it is then impossible for the active source with amplitude zero to cancel the incoming wave from PORT 2.

We quantify the level of nonreciprocity by defining an isolation factor I as the ratio of the magnitudes $|S_{12}|$ and $|S_{21}|$ in the decibel scale as

$$
\mathcal {I} = 2 0 \log_ {1 0} \left(\left| \frac {1}{1 - \widetilde {G} \operatorname{sinc} k \delta x} \right|\right).\tag{4}
$$

The maximum $\mathcal{I}$ occurs when $1 - \widetilde{G}$ sinc $k\delta x = 0$ which is the same condition for which $S_{21} = 0$ [Eq. (2)]. Note that for $\widetilde{G} \leqslant 1$ , perfect cancellation cannot be achieved and $\mathcal{I}$ becomes far less dramatic. Hence, the operating limits for the dimensionless gain are $1 \leqslant \widetilde{G} < \pi$ . As shown in the middle curve in Fig. 2(b), when $\widetilde{G} = 0.5\pi$ , the peak $\mathcal{I}$ occurs when $\delta x = 0.25\lambda$ . The adjacent curves in Fig. 2(b) show how the peak frequency $f_{\mathrm{pk}}$ for constant $\widetilde{G}$ increases with $\widetilde{G}$ . When $\widetilde{G}$ is increased to $0.7\pi$ , $f_{\mathrm{pk}}$ increases ( $\delta x \approx 0.3\lambda$ ), while when $\widetilde{G}$ is decreased to $0.4\pi$ , $f_{\mathrm{pk}}$ decreases ( $\delta x \approx 0.2\lambda$ ), a $50\%$ change in frequency. These changes in $\widetilde{G}$ can be imposed by either modifying the sensor-source separation distance $\delta x$ , which is fixed once the NAM unit cell is fabricated, or by varying $G(\omega)$ , the controller transfer function. The latter approach is advantageous because we have the ability to artificially resize our unit cell just through the controlling electronics of the system.

## III. EXPERIMENTAL METHODS

The previous theoretical results show the promise of the NAM approach for producing nonreciprocal media. We next sought to determine if the nonreciprocal properties of the idealized system are conveyed to a real system. To do so, we fabricated an experimental prototype consisting of a 5 cm × 5 cm acoustic waveguide equipped with a 10-cm NAM segment, as shown in Fig. 3(a). The active, NAM portion of the waveguide consisted of a small loudspeaker [manufactured by Dayton Audio, shown in Fig. 3(b)] and an ADMP401 microelectromechanical systems (MEMS) microphone sensor mounted to a breakout board from SparkFun Electronics [Fig. 3(c)], both of which are mounted in a waveguide wall and separated by a distance of 10 cm, as shown in Fig. 3(a) and in the schematic in Fig. 3(d). To implement the NAM feedforward control strategy, the signal from the microphone is amplified and filtered so that the output voltage applied to the loudspeaker, $V_{e}$ , is given by

$$
V _ {e} = H (\omega) P (x _ {p}) = \left[ \frac {H _ {g}}{1 + j f / f _ {\mathrm{lp}}} \right] P (x _ {p}),\tag{5}
$$

where $H_{g}$ is the gain supplied by an audio amplifier, $f_{lp}$ is the corner frequency of a first-order analog low-pass filter, implemented using discrete circuit components on a breadboard, and $P(x_{p})$ is the pressure sensed by the microphone at $x = x_{p}$ . Early testing revealed instabilities associated with higher-order modes above the cutoff frequency of the waveguide which were not accounted for in the one-dimensional (1D) theory. Hence, the low-pass filtering is used to mitigate these modal spillover effects [31]. A disturbance was induced by an electrodynamic speaker placed either at PORT 1 or PORT 2. Transmission and reflection behavior of the experimental unit cell was isolated from the waveguide end conditions using a robust four-microphone technique [32].

![](images/7ef5dd096a99bb17a4101a84d4981cb2f4e123b349274e703fd93e8991a10a0f.jpg)

![](images/ec9f35cca612a85fe720ceb1b71713613d76891172fa80b4d1be4e58f1399fd6.jpg)
FIG. 3. Experimental setup and model. (a) Unit cell fabricated for experimental testing containing a loudspeaker source (b) actuated with signals measured from a MEMS microphone sensor (c) separated by 10 cm from the source. (d) Coupled electromechanical-acoustic system schematic used for the one-dimensional model.

The idealized NAM system relied on the assumption of perfect control of both the electronics and the sound source. In order to predict and analyze the behavior of the experimental NAM unit cell, our idealized mathematical model was expanded to include the dynamics and fluid-structure interaction associated with the real acoustic source. A schematic of the expanded coupled electromechanical-acoustic system model for the experimental setup is shown in Fig. 3(d). The mechanical components of the model are the speaker cone which is represented as a rigid disk with cross sectional area $S_{d}$ , centered at $x_{s}$ , with mass $M_{m}$ , compliance $C_m$ , damping $R_{m}$ , and a velocity $U$ . The rigid disk moves with the loudspeaker's voice coil, which has an electrical resistance and inductance, $R_e$ and $L_e$ , respectively. The electromechanical system is driven by the voltage $V_e$ , equal to the controller transfer function $H(\omega)$ applied to the signal from the microphone located at $x_p$ [Eq. (5)]. $V_e$ induces a current $I$ through the voice coil, generating a force on the voice coil proportional to the product of its magnetic field $B$ , voice coil wire length $l$ , and the current $I$ . The density of air is given by $\rho_0$ and sound speed by $c$ . The cross-sectional area of the waveguide is $S_0$ . The resulting coupled structural-acoustic equations for the expanded 1D model are

$$
\begin{array}{c} Z _ {e} I = - B l U + V _ {e}, \quad Z _ {m} U = B l I - P (x _ {s}) S _ {d}, \\ \frac {d ^ {2} P}{d x ^ {2}} + k ^ {2} P = \frac {- j \omega \rho_ {0} U S _ {d}}{S _ {0}} \delta (x - x _ {s}), \end{array}\tag{6}
$$

![](images/db7db71375a3be3dc35119231140cc0161d5c1941dde0330dbb07e1578ec86e5.jpg)

![](images/1fafd14f30db0d4ff76ff79138b035a7c1f088358c862be0d1d1321f34214190.jpg)

![](images/2ab2d3d7b379cb5f8cf2c9376fb52a0ce8f5304daa4e58e257ecae1537d25cf8.jpg)
FIG. 4. Experimental results. (a) Transmission coefficients across the experimental unit cell for waves incident at PORT 2 (red) and at PORT 1 (blue). (b) Shifts in peak isolation frequency demonstrating the remarkable tunable nature of the NAM system. There is good agreement between the experimental data (circles), expanded 1D model (solid lines), and FW simulations ( $\times$ 's). (c) The value of $H_{g}$ required to generate a peak above 40 dB (solid blue line) as a function of the corner frequency of the low-pass filter $f_{lp}$ is shown on the left axis. The frequency where the peak occurs, $f_{pk}$ (solid red line), and the upper and lower frequencies (dashed curves) between which I exceeds 10 dB are shown on the right axis.

where the loudspeaker electrical impedance is given by $Z_{e} = j\omega L_{e} + R_{e}$ and mechanical impedance by $Z_{m} = j\omega M_{m} + R_{m} + 1/j\omega C_{m}$ . The electromechanical parameters of the loudspeaker ( $R_{e}, L_{e}, B, l, S_{d}, C_{m}, M_{m}$ , and $R_{m}$ ) were determined experimentally from loudspeaker impedance measurements following techniques as detailed in the Supplemental Material [28]. Implicit in Eq. (6) are the assumptions that only plane waves propagate in the x direction, that wavelengths are sufficiently large compared to the source dimensions, and that $\delta x$ is large enough to neglect near-field effects from the loudspeaker. To study the validity of these assumptions, we developed comparison full-wave (FW) simulations using a three-dimensional finite-element model created in COMSOL MULTIPHYSICS, enabling exploration of the importance of dimensionality and evanescent wave contributions.

As in the idealized case, the scattering matrix for the expanded expanded 1D structural acoustic model, $\bar{S}$ , can be determined analytically. Solving Eq. (6) we find

$$
\bar {S} = \frac {1}{\overline {{\Delta}}} \left[ \begin{array}{c c} - \big (\bar {G} \frac {e ^ {j k \delta x}}{2 j k \delta x} + \kappa \big) & 1 \\ (1 - \bar {G} \operatorname{sinc} k \delta x) & - \big (\bar {G} \frac {e ^ {- j k \delta x}}{2 j k \delta x} + \kappa \big) \end{array} \right],\tag{7}
$$

where

$$
\begin{array}{l} \bar {G} = - j \frac {\rho_ {0} c S _ {d} B l H (\omega) k \delta x}{S _ {0} [ Z _ {m} Z _ {e} + (B l) ^ {2} ]}, \\ \kappa = \frac {\rho_ {0} c S _ {d} ^ {2} Z _ {e}}{2 S _ {0} [ Z _ {m} Z _ {e} + (B l) ^ {2} ]}, \\ \bar {\Delta} = 1 + \kappa + \bar {G} \frac {e ^ {- j k \delta x}}{2 j k \delta x}. \end{array}\tag{8}
$$

Comparing the scattering matrix from the idealized case, $\widetilde{S}$ , from Eq. (2), with that from the coupled system, $\bar{S}$ , from Eq. (7), we note a remarkable similarity with two key differences. The first difference is the addition of $\kappa$ in the diagonal elements and in $\bar{\Delta}$ , a term associated with the fluid-structure interaction between the speaker cone and air in the waveguide. For the system we implemented, we found that $\kappa$ was small and had minimal contribution to the behavior of the system as predicted by $\bar{S}$ . Second, and more significantly, $\bar{G}$ is more complex than $\widetilde{G}$ as $\bar{G}$ includes the frequency-dependent electromechanical dynamics of the loudspeaker itself as well as the effects of the low-pass filter [Eq. (5)]. Despite these differences, we will next show that the desired features observed in the ideal model (large, subwavelength, tunable nonreciprocity) are recapitulated in the real system.

## IV. RESULTS

Experimental results for the NAM unit cell with $H_{g} = 0.143 \, V/Pa$ and $f_{lp} = 830 \, Hz$ are plotted in Fig. 4. The measured transmission coefficients (circles) in Fig. 4(a) show a highly nonreciprocal nature similar to that observed in the idealized NAM model [Eq. (1)]. The coupled electromechanical-acoustic response of the NAM unit cell is well represented by both the expanded 1D model (solid lines) and FW models ( $\times's$ ). This demonstrates that the effects of three-dimensionality and the loading of evanescent modes on the source speaker are not dominant. Hence, the 1D structural acoustic model of Eq. (6) can be used to select design parameters in a computationally inexpensive way. For waves traveling from PORT 1 to PORT 2, the NAM behaves primarily as a loss medium, effectively opaque near 525 Hz ( $\delta x \approx 0.15\lambda$ ), resulting in a remarkable I peak of over 40 dB [middle curve in Fig. 4(b)]. The nonreciprocity of the experimental system is broadband, with I magnitudes above 10 dB across a third of an octave (from 470 to 600 Hz). Furthermore, the ability to electronically shift the frequencies at which the I peaks occur was demonstrated experimentally. The adjacent curves in Fig. 4(b) show isolation factors for the experimental setup with $H_{g}^{+} = 0.140 \, V/Pa$ and $f_{lp}^{+} = 1676 \, Hz$ , shifting $f_{pk}$ up to 586 Hz ( $\delta x \approx 0.17\lambda$ ), and with $H_{g}^{-} = 0.152 \, V/Pa$ and $f_{lp}^{-} = 548 \, Hz$ , shifting $f_{pk}$ down to 460 Hz ( $\delta x \approx 0.13\lambda$ ). For both additional peaks, the NAM unit cell remained subwavelength, and the $\approx 40 \, dB$ I magnitudes and relative bandwidths were maintained.

A careful analysis of $\bar{S}$ reveals that real system behavior is strongly linked to both the controller characteristics ( $H_{g}$ and $f_{lp}$ ) and the electromechanical characteristics of the loudspeaker. Hence, accurate characterization of the real loudspeaker source is required in order to select appropriate controller parameters. The relationship between $H_{g}$ and $f_{lp}$ is shown in Fig. 4(c). The value of $H_{g}$ required to achieve I peaks exceeding 40 dB for a given $f_{lp}$ is plotted in blue on the left axis, and the frequency at which the I peak occurs $(f_{\mathrm{pk}})$ is plotted in red (solid line) on the right axis. Also on the right axis are the upper and lower frequencies (dashed lines) between which the $\mathcal{I}$ exceeds 10 dB. It is found that for values of $f_{\mathrm{lp}}$ below 1000 Hz, there is an inverse relationship between $H_g$ and $f_{\mathrm{lp}}$ , and $H_g$ is more sensitive to changes in $f_{\mathrm{lp}}$ than for values of $f_{\mathrm{lp}}$ above 1000 Hz, where $H_g$ begins to increase monotonically with $f_{\mathrm{lp}}$ . As can be seen in Fig. 4(c), for all values of $f_{\mathrm{lp}}$ shown, the large relative bandwidths are maintained. While we demonstrated experimentally three different frequencies at which the NAM system can generate large, broadband, subwavelength nonreciprocity in Fig. 4(b), Fig. 4(c) suggests that these frequencies could be chosen arbitrarily over a range of 100 Hz or more.

The experimental results demonstrate that the NAM approach can meet or exceed the top performance metrics achieved by other methods of breaking reciprocity. A device that utilized rotating fluid to bias the propagation direction of sound generated isolation factors of over 40 dB, but the effects were very narrow band (isolation factors exceeding 10 dB occurring over a frequency range of less than 10 Hz with peaks near 750 Hz) [33]. Other methods have proven capable of breaking reciprocity over a broad range of frequencies, but those methods rarely saw isolation factor magnitudes exceeding 20 dB [15]. While the highly nonreciprocal, subwavelength, broadband nature of the real NAM system is in itself impressive, the ability to tune the system while preserv-

[1] L. Zigoneanu, B. I. Popa, and S. A. Cummer, Nat. Mater. 13, 352 (2014).

[2] S. A. Cummer, J. Christensen, and A. Alù, Nat. Rev. Mater. 1, 16001 (2016).

[3] B. I. Popa and S. A. Cummer, Nat. Commun. 5, 3398 (2014).

[4] F. Zangeneh-Nejad and R. Fleury, Rev. Phys. 4, 100031 (2019).

[5] M. Maldovan, Nature (London) 503, 209 (2013).

[6] L. E. Kinsler, A. R. Frey, A. B. Coppens, and J. V. Sanders, Fundamentals of Acoustics, 4th ed. (Wiley, New York, 1999).

[7] Y. Fung, Foundations of Solid Mechanics (Prentice-Hall, Englewood Cliffs, NJ, 1965).

[8] L. Lyamshev, Sov. Phys. Dokl. 4, 406 (1959).

[9] N. Boechler, G. Theocharis, and C. Daraio, Nat. Mater. 10, 665 (2011).

[10] F. Zangeneh-Nejad and R. Fleury, Appl. Sci. 8, 1083 (2018).

[11] R. Fleury, A. B. Khanikaev, and A. Alù, Nat. Commun. 7, 11744 (2016).

[12] Y.-G. Peng, Y.-X. Shen, D.-G. Zhao, and X.-F. Zhu, Appl. Phys. Lett. 110, 173505 (2017).

[13] A. B. Khanikaev, R. Fleury, S. H. Mousavi, and A. Alù, Nat. Commun. 6, 8260 (2015).

[14] L. Quan, D. L. Sounas, and A. Alù, Phys. Rev. Lett. 123, 064301 (2019).

[15] Y. Zhai, H. S. Kwon, and B. I. Popa, Phys. Rev. B 99, 220301(R) (2019).

[16] H. Nassar, X. C. Xu, A. N. Norris, and G. L. Huang, J. Mech. Phys. Solids 101, 10 (2017).

[17] A. Rogov and E. Narimanov, ACS Photonics 5, 2868 (2018).

ing these characteristics is what makes the NAM technique particularly compelling.

## V. CONCLUSIONS

In conclusion, we have experimentally demonstrated large, tunable, broadband nonreciprocity for a subwavelength device using our NAM strategy. We speculate that performance can be further improved by cascading NAM unit cells and/or developing more sophisticated control schemes. We showed that a simple one-dimensional coupled acoustic model can adequately predict the performance and stability of both full-wave simulations and experimental data, thus further design will require minimal computational resources. As noted, this concept can be applied in a variety of different settings, from acoustic waves in higher-dimensional domains to structural waves in elastic and piezoelectric domains, leaving a vast design and application space as yet unexplored.

## ACKNOWLEDGMENTS

This work was supported by National Science Foundation Grant No. CMMI-1761300 for authors N.G., Z.W., and K.G., National Institutes of Health Grant No. R01-NIDCD 04084 for A.S., and National Science Foundation Grant No. CMMI-1942901 for Y.Z. and B.-I.P.

[18] Y. Chen, X. Li, H. Nassar, G. Hu, and G. Huang, Smart Mater. Struct. 27, 115011 (2018).

[19] M. Collet, P. David, and M. Berthillier, J. Acoustical Soc. Am. 125, 882 (2009).

[20] A. Sasmal, N. Geib, B.-I. Popa, and K. Grosh, New J. Phys. 22, 063010 (2020).

[21] P. A. Nelson and S. J. Elliott, Active Control of Sound (Academic, New York, 1992).

[22] S. M. Kuo and D. R. Morgan, Proc. IEEE 87, 943 (1999).

[23] J. C. Burgess, J. Acoust. Soc. Am. 70, 715 (1981).

[24] R. F. La Fontaine and I. C. Shepherd, J. Sound Vib. 91, 351 (1983).

[25] M. A. Swinbanks, J. Sound Vib. 27, 411 (1973).

[26] J. Winkler and S. J. Elliott, Acustica 81, 475 (1995).

[27] M. Åbom, Mech. Syst. Signal Process. 5, 89 (1991).

[28] See Supplemental Material at http://link.aps.org/supplemental/10.1103/PhysRevB.103.165427 for a formal proof of nonreciprocity, details of the experimental setup, system characterization, and a discussion of effective material properties.

[29] M. B. Muhlestein, C. F. Sieck, P. S. Wilson, and M. R. Haberman, Nat. Commun. 8, 15625 (2017).

[30] B. I. Popa and S. A. Cummer, Phys. Rev. B 85, 205101 (2012).

[31] C. R. Fuller, S. J. Elliott, and P. A. Nelson, Active Control of Vibration (Academic, New York, 1996).

[32] Z. Tao and A. F. Seybert, SAE Technical Papers (2003).

[33] R. Fleury, D. L. Sounas, C. F. Sieck, M. R. Haberman, and A. Alù, Science 343, 516 (2014).
