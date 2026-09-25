# Negative Conductivity Induced Reconfigurable Gain Metasurfaces and Their Nonlinearity

Xiaoyue Zhu $^{1,2}$ , Chao Qian $^{1,2,*}$ Erping Li, $^{1}$ and Hongsheng Chen $^{1,2,3,\dagger}$

$^{1}$ ZJU-UIUC Institute, Interdisciplinary Center for Quantum Information,

State Key Laboratory of Extreme Photonics and Instrumentation, Zhejiang University, Hangzhou 310027, China

$^{2}$ ZJU-Hangzhou Global Science and Technology Innovation Center,

Key Lab. of Advanced Micro/Nano Electronic Devices & Smart Systems of Zhejiang,

Zhejiang University, Hangzhou 310027, China

$^{3}$ Jinhua Institute of Zhejiang University, Zhejiang University, Jinhua 321099, China

(Received 22 January 2024; revised 30 May 2024; accepted 24 July 2024; published 9 September 2024)

The past decades have witnessed the rapid development of metamaterials and metasurfaces. However, loss is still a challenging problem limiting numerous practical applications, including long-range wireless communications, superscattering, and non-Hermitian physics. Recently, great effort has been made to minimize the loss, however, they are too complicated for practical implementation and still restricted by the theoretical limit. Here, we propose and experimentally realize a tunable gain metasurface induced by negative conductivity, with deep theoretical analysis from scattering theory and equivalent circuits. In the experiment, we create metasurface samples embedded with tunable negative (or positive) conductivity to achieve adjustable gain (or loss). By varying the control bias voltages, the metasurfaces can reflect incident waves with additional controllable gain. Interestingly, we find the gain metasurfaces inherently pose nonlinearities, which are beneficial for nonlinear optics and microwave applications, particularly for the nonlinear activation of wave-based neural networks.

DOI: 10.1103/PhysRevLett.133.113801

Manipulating electromagnetic (EM) waves with artificial materials are of great significance for scientific discoveries and applications. Typical examples include metamaterials and metasurfaces, which have greatly reshaped the landscape of well-documented physical laws and unlocked many unconventional phenomena. The extremely high degree of freedom in geometric patterns and material properties endow them with an elegant and efficient way to manipulate EM waves at subwavelength scale $[1]$ . Beyond early studies of passive metasurfaces, researchers strive to embed active components into metasurfaces to enable a dynamical control of EM waves, termed as tunable, reconfigurable, programmable, and coding metasurfaces $[2–4]$ . Typical active control methods are achieved by phase-change materials $[5]$ , liquid crystals $[6]$ , mechanical actuation $[7]$ , varactor diode, and more, depending on specific working bands $[8]$ . Recently, researchers start to introduce deep learning to bring tunable metasurfaces into the next generation of intelligent metasurfaces $[2,9,10]$ .

After two-decades of development, however, loss is still an insurmountable factor to impede the advancements of all passive, tunable, and intelligent metasurfaces [11]. Maxwell's equations elucidate that, to have a full control of

EM waves, both the real and imaginary components of permittivity and permeability must be considered $[12–14]$ . Correspondingly, it implies the necessity in controlling both phase and amplitude of metasurfaces $[15]$ . However, current metasurfaces based works predominantly focus on the phase control, while ignoring the effect of amplitude suppression and energy loss. Realizing amplitude balance and even amplification is extremely important for numerous applications, such as compensating for signal decay and propagation loss in long-distance wireless communications $[16]$ , breaking diffraction limits $[17]$ , facilitating non-Hermitian physics and other physical researches $[12,18,19]$ . Over the years, great effort has been devoted to mitigate losses by meticulously designing metasurface structures and modifying material properties, such as the adoption of low-loss dielectric materials, optimized metallic structures $[20]$ , liquid crystals and special semiconductor materials. Despite these efforts inching closer to theoretical limit, they do not fundamentally address the intrinsic problem $[11,13,17]$ , i.e., loss does exist.

In this work, we propose and experimentally demonstrate a negative resistance enabled tunable gain metasurface. By incorporating tunable conductivity, we have ability to regulate the imaginary components of effective permittivity, consequently leading to the reconfigurability of real parts in Poynting vectors that are intimately linked with power flows in EM fields $[21]$ . To elucidate underlying physics, we also explore the theory through scattering and equivalent circuit analysis $[22,23]$ , shedding lights on the gain mechanisms from a more universal and macroscopic viewpoint. For validations, we fabricated samples by printing simulation-verified shapes on an ultrathin medium plate. Tunable negative resistances are achieved through the embedding of tunnel diode (TD) $[24]$ into the metasurface, a process grounded in quantum mechanical effects. Moreover, ours are experimentally proven to be capable of intrinsically offering additional stable nonlinearity $[25–27]$ , offering a simple solution for wave-based neural networks $[28–31]$ . Looking forward, our work breaks fundamental limitation in traditional metasurfaces, providing a cost-effective and simple method to achieve gains and nonlinearity for a future “gold rush” in microwave neural networks, long-distance metasurface communications, and metasurface-enabled internet of things $[32]$ .

![](images/9437e28fce2ce5889776139beda4458cd16d26ea67fd196f3eeea9bcabb79f15.jpg)
FIG. 1. Conceptual illustration of gain metasurfaces. When the incident wave impinges on the metasurface, it is reflected with an additional gain that can be modulated by adjusting the bias voltages $(V_{1}-V_{4})$ . The coordinate is settled at the center of metasurfaces. The diagram at the center bottom represents the operational states of the metamaterials. Typically, metasurfaces used to work in a lossy state represented by the purple region where magnitudes of reflection coefficients ( $\Gamma$ ) are less than 1. The bottom left corner of the figure elucidates the underlying microscopic mechanisms, where conductivity $[\sigma(\omega)]$ and imaginary components of the effective permittivity $[\epsilon_{\mathrm{eff}}^{\prime\prime}(\omega)]$ are positive. In contrast, ours operate in gain states denoted by the orange region where magnitudes of $\Gamma$ exceed 1. To achieve this, $\epsilon_{\mathrm{eff}}^{\prime\prime}(\omega)$ and $\sigma(\omega)$ should be negative, which also leads to changes in currents as illustrated by $\bar{J}_{2}$ and $\bar{J}_{1}$ . $\bar{S}_{in}$ and $\bar{S}_{out}$ refer to Poynting vectors of incident and reflected waves. And $\bar{S}'$ indicates the Poynting vector of additional power flow provided by the gain metasurfaces where $\operatorname{Re}[\nabla \cdot \bar{S}] > 0$ .

Theoretical analysis—Figure 1 schematically shows the proposed negative conductivity enabled gain metasurfaces that can amplify the reflected waves at will. And the gain value can be manipulated by regulating bias voltages. These results are ensured by new physical mechanisms as analyzed later.

As per the law of energy conservation, gain obtainment necessitates an additional energy input. In EM fields, the energy is linked with the Poynting vector $\bar{S} = \bar{E} \times \bar{H}^{*}$ , where $\bar{E}$ is the electric field intensity and $\bar{H}^{*}$ is the conjugate of magnetic field $\bar{H}$ intensity [14,21]. According to complex Poynting theorems, we have

$$
\nabla \cdot \bar {S} - i \omega [ \bar {H} ^ {*} \cdot \bar {B} - \bar {E} \cdot \bar {D} ^ {*} ] + \bar {J} ^ {*} \cdot \bar {E} = 0,\tag{1}
$$

where $\omega$ is the angular frequency and $\bar{J}$ is the current density. According to the constitutive relations, $\bar{B} = \mu (\omega)\bar{H}$ , $\bar{D} = \epsilon (\omega)\bar{E}$ . In EM fields, the real part of $\nabla \cdot \bar{S}$ indicates the total active power flowing out of an infinitely small volume, which is $\mathrm{Re}[\nabla \cdot \bar{S}]$ . This total active power is precisely the difference between the active power flowing out of and absorbed by the small volume, serving as the additional power for gain realizations. And the imaginary part is associated with the internal power transforming. The item $i\omega [\bar{H}^{*}\cdot \bar{B} ]$ and $i\omega [\bar{E}\cdot \bar{D}^{*}]$ indicate stored magnetic and electric energy under ideal conditions. While, $\bar{J}^{*}\cdot \bar{E}$ represents the Ohmic losses. More details about energy transformation relations implied by Eq. (1) are listed in Supplemental Material [33]. Referring to Ohm's law, $\bar{J} = \sigma (\omega)\bar{E}$ , where $\sigma (\omega)$ denotes conductivity. In lossy mediums, the conductivity can be incorporated into the imaginary part of permittivity to construct an effective dispersive permittivity $[\epsilon_{\mathrm{eff}}(\omega)]$ , which is $\epsilon_{\mathrm{eff}}(\omega) = \epsilon_{\mathrm{eff}}^{\prime}(\omega) + i\epsilon_{\mathrm{eff}}^{\prime \prime}(\omega) = \epsilon^{\prime}(\omega) + i\{\epsilon^{\prime \prime}(\omega) + [\sigma (\omega)] / \omega\}$ .

In EM theories, the imaginary part of permittivity characterizes loss or gain intensities of the material [12,13]. In practice, low loss materials are usually adopted, whose imaginary part $\epsilon_{\mathrm{eff}}^{\prime \prime}(\omega)$ is pretty small. As the metasurface is induced by a plane transversal EM wave as showcased in Fig. 1, the formula $|\bar{E}| = \eta |\bar{H}|$ holds, where $\eta = \sqrt{\mu(\omega) / \epsilon_{\mathrm{eff}}(\omega)}$ is the spatial wave impedance. Combined with the analysis mentioned above, Eq. (1) can also be rewritten as

$$
\begin{array}{c} \nabla \cdot \bar {S} = - \omega \Bigg \{\left[ \frac {\epsilon_ {\mathrm{eff}} ^ {\prime \prime}}{| \epsilon_ {\mathrm{eff}} ^ {\prime \prime} |} + \frac {\mu^ {\prime \prime}}{| \mu^ {\prime \prime} |} \right] \\ - i \left[ \frac {\epsilon_ {\mathrm{eff}} ^ {\prime}}{| \epsilon_ {\mathrm{eff}} ^ {\prime} |} + \frac {\mu^ {\prime}}{| \mu^ {\prime} |} \right] \Bigg \} \cdot | \epsilon_ {\mathrm{eff}} | \cdot | \bar {E} | ^ {2}. \end{array}\tag{2}
$$

The detailed derivation process of Eq. (2) is provided in the Supplemental Material [33], note 1. Obviously, the real part of $\nabla \cdot \bar{S}$ is associated with the imaginary part $\epsilon_{\mathrm{eff}}^{\prime \prime}(\omega)$ . In traditional lossy mediums and metasurfaces, $\sigma (\omega)$ and $\epsilon_{\mathrm{eff}}^{\prime \prime}(\omega)$ are typically positive [11]. Consequently, the condition $(\mathrm{Re}[\nabla \cdot \bar{S}] < 0)$ exists in lossy medium, indicating that the total active power is absorbed by the medium. However, once the conductivity $\sigma (\omega)$ is sufficiently negative, $\epsilon_{\mathrm{eff}}^{\prime\prime}(\omega)$ could be negative. As a result, formula $\operatorname{Re}[\nabla \cdot \bar{S}] > 0$ can be satisfied, which means additional active power is radiated from the metasurface. In other words, the metasurfaces operate in gain states.

From a macroscopic viewpoint, the micro conductivity $\sigma(\omega)$ is associated with differential resistance $R_{L}$ (real part of complex impedance $Z_{L}$ ) [22,23]. Referring to EM wave and scattering theories [44], scattering coefficients can be utilized to portray the scattering properties of metamaterials, microwave circuits and other microwave systems. Among the coefficients, $S_{11}$ specifically characterizes reflections at a single port. While, in the context of reflective metasurfaces, circuits and systems, reflection coefficient ( $\Gamma$ ) is commonly used as a substitute for $S_{11}$ . As per the associated EM theories [10,14,21,22], $\Gamma$ is defined as $\Gamma = (U_{\mathrm{ref}}/U_{\mathrm{ind}})$ , where $U_{ref}$ and $U_{ind}$ denote voltage magnitudes of reflected and incident waves, respectively. Referring to related theories, the value of the coefficient is mainly decided by port impedance $Z_{L}$ , which can be expressed as $\Gamma = [(Z_{L} - Z_{0})/(Z_{L} + Z_{0})]$ [22]. Herein, $Z_{0}$ represents characteristic impedance, typically 50 $\Omega$ . In conventional metasurfaces, the absolute values of reflection coefficients are consistently constrained to be less than unity (i.e., $|\Gamma| < 1$ ), as illustrated in the blue region of the circle in Fig. 1. In other words, the reflected magnitudes of conventional metasurfaces are less than incident magnitudes. Consequently, they inevitably operate in lossy states. To achieve additional gains (i.e., $|\Gamma| > 1$ ), $Z_{L}$ must be negative as $Z_{0}$ is positive. If the real part of $Z_{L}$ is sufficiently negative, the item $|Z_{L} + Z_{0}|$ will be possibly smaller than the numerator (i.e., $|Z_{L} - Z_{0}|$ ), thereby transitioning the system to gain state ( $|\Gamma| > 1$ ). In simpler terms, sufficiently negative differential resistances, corresponding to the conductivity in field and micro analysis, can introduce gains into the metasurfaces. Given that the equations and concepts discussed in this paragraph also apply in microwave circuits and systems designs, it is reasonable to extend the conclusion to these contexts as well. Further details can be found in the Supplemental Material [33].

Gain metasurface designs and simulations—Illustrated in Fig. 2(a) is structure of the proposed gain metasurfaces working at microwave bands. This structure is constructed by printing T-shaped metal patches, optimized through simulation, onto an ultrathin substrate coated with metal at the bottom. A TD is welded in the gap between the two T-shaped patches to offer tunable negative resistance or conductivity, an effect attributable to quantum mechanics [17,24]. The conductivity can be adjusted between positive and negative by varying bias voltages. The two arms of T contribute complex impedances, and the gap introduces an equivalent capacitor that is paralleled with the diode. Detailed values are provided in Supplemental Material [33]. Utilizing circuit simulation software, the equivalent circuit model are derived as depicted in Fig. 2(b).

(a)
TD
![](images/ed351d297cfac9d3a36613f425045783b2149dfda558006b7a179b39f2e228f6.jpg)

![](images/abd6cee59e6f341903dfe947aba4ebb87ab1eb52702e86905a437670baab83c7.jpg)
(d)

![](images/c7ce067c2657e59e38e4e5ae9251cf1c4edb5daf4508adb2bf7aeb89c697aed9.jpg)

![](images/939a211b5e82a595c26cadd3f863da8c06122f8950e4817138acc53cf777d3e5.jpg)
FIG. 2. Design of gain metasurfaces. (a) Three-dimensional illustration of the gain metasurface. The unit cell structure can be characterized by the structural parameters: $a_1 = 50$ , $b_1 = 60$ , $h = 1$ , $w_2 = 10$ , $w_1 = 19$ , $p_1 = 48$ , $p_2 = 20$ , $g_1 = 2$ (units: mm). The overall 3D size is $a_1 \cdot b_1 \cdot h_1$ . TD indicates the tunnel diode. The symbols $Z_{21}$ , $Z_{22}$ , $Z_{23}$ denote equivalent impedances of these three metallic patch parts, respectively. (b) Equivalent circuit models. Both $\Gamma_L$ and $Z_L$ represent reflection coefficients and the equivalent terminal impedance observed from the reflection wave ports, respectively. $-R_d$ can be continuously manipulated in positive or negative regions. (c) Current-voltage ( $I-V$ ) curves. Tested currents with voltages directly varying. The differential resistance values $R_1$ , $R_2$ , $R_3$ , and $R_4$ are about $-42\Omega$ , $-17.5\Omega$ , $0\Omega$ , and $48\Omega$ , respectively. (d) Simulated spectra of reflection coefficients with the differential resistances of TD being $R_1 \sim R_4$ .

According to official references, a TD model is also showcased in the right region of Fig. 2(b).

To ascertain practical current-voltage $(I - V)$ relationships, we experimentally assess the diode's output current by varying voltages directly imposed on TD. As depicted in Fig. 2(c), the current initially increases with escalating applied voltages, diminishes around $0.09\mathrm{V}$ , and then resumes its upward trend at approximately $0.38\mathrm{V}$ . Evidently, the curve can have both positive and negative differential resistances as highlighted in Fig. 2(c).

As previously analyzed, the presence of positive real conductivity or resistance results in loss states, while a sufficiently negative real conductivity or resistance has the potential to introduce additional gains. To validate this, we initially conducted a simulation test using field-circuit cosimulations, with the value of real resistances $-R_{d}$ varying from $R_{1}$ to $R_{4}$ . And the internal parameters of TD are acquired from official data sheets. As exemplified in Fig. 2(d), with sufficiently negative real resistance, such as $R_{1}$ and $R_{2}$ , the metasurface offers additional gains around 3.1 GHz, with a bandwidth of approximately 100 MHz. If the real resistance is positive, the metasurface operates with substantial loss. Moreover, the metasurface can be adjusted to operating in both loss and gain states, as evidenced by the simulation results.

(a)
![](images/3d941958281fedfbf0512c485e0ec6cd8bf112611d3da7ca247aad394ac3eaca.jpg)

(b)
![](images/126478437927c5efdab9e58a60aeede4dc61a4268139efc7bac56de6852ca015.jpg)

![](images/ad9354b5747a6673f9bbe94d0bdf32ea9c7dfcd8748ac5651998b0d602c41add.jpg)

(d)
![](images/741582f506ede21c32af9630a4992773b217b321114c565d0ed37e09748d5b80.jpg)
FIG. 3. Experimental results. (a) A photograph of experimental setups. T and R represent the transmitter and receiver, respectively. The H-shaped gain metasurfaces can work in a wide range of incident angle. (b) Gain—voltage curves. Measured gains of the two samples are plotted correspondingly with the total bias voltage changing. The two kinds of samples are designed to work in different bands. More details can be found in Supplemental Material [33], notes 2 and 3. (c),(d) The gain evolution processes of samples 1 and 2 with total bias voltage changing.

Experimental demonstrations—To corroborate our prior analysis, we fabricated several samples for experimental validations. Figure 3(a) displays the metasurfaces positioned at the center, with two polarization-matched double-rigged horns placed on the rail. During the experiment, two antennas were situated atop the sample to transmit and receive plane waves. The modulation of biasing voltages could magnify or reduce the reflected waves as required. For illustrative purposes, we tested different spectra at various bias voltages and delineated some representative results in Fig. 3(c) for visual clarity. Noticeably, the reflective properties are highly sensitive around 2.55–2.65 GHz. Figure 3(c) generally illustrates the dynamic evolution process of the gain value as bias voltages increase. The discrepancy between the tested results in Fig. 3(c) and the simulations shown in Fig. 2(d) is attributed to fabrication errors and mismatched internal parameters with official data for TDs. Tiny fluctuations on the spectra are mainly introduced by circumstance noises. However, they do not affect overperformance on gain realizations.

To comprehensively portray the dynamic evolution process of the gain value as bias voltages increase, we plot the gain-voltage relations of sample 1 in Fig. 3(b). Initially, the resistance increases in a positive mode, and received reflected magnitude at the working band is significantly less than 0 dB. Subsequently, the resistance switches to a descending negative mode. The reflection magnitudes then begin to rise, peaking at approximately 0.25 V. Following this, the reflection magnitudes start to drop, with the resistance gradually returning to a positive mode. Intriguingly, the curve exhibits multiple peaks due to the sample's high sensitivity to resistance values. Specifically, the maximum gain is attained only when the resistance of TD is negatively matched as discussed in Supplemental Material [33]. Once the resistance deviates from the point, the gain will begin to decline. Meanwhile, the I-V relation of the tunnel diode does not continuously increase or decrease. Instead, the resistance can switch abruptly, as shown in Fig. 2(c). Hence, there are many points for the resistance to be matched and then deviated, which causes several valleys and side peaks. Actually, it is the sensitivity that leads to tunability. To substantiate the universality of these findings, we also designed and tested other samples as aforementioned. Figure 3(d) illustrates that sample 2 exhibits a gain property during 3.85–3.95 GHz. Likewise, the reflected magnitudes escalate from negative to positive and then recede as the bias voltage continues to rise. For integrity, we also plot the gain-voltage relations of sample 2 in Fig. 3(b). For sample 2, variations in tunnel diodes and structures are bound to cause differences such as maximum values, thresholds, and sensitive resistance values. However, the overall trend of loss and gain remains similar.

Nonlinearity for wave-based neural network—Nonlinear behavior is an essential and pressing property in the applications of metasurfaces. For instance, in wave-based neural networks, each metasurface unit can function as a neuron, as depicted in Fig. 4(a). By manually adjusting these neuron units, the entire network can perform complex tasks $[29–31,45]$ . However, traditional metasurfaces typically operate linearly, thereby constraining the activation functions required in the neurons. In practices, many complex tasks necessitate nonlinear activations such as complex image recognitions and task predictions $[27–31,45]$ . Therefore, the ability to operate in nonlinear states is a critical requirement for neuron units. Furthermore, tunable gains provided by the gain metasurface effectively mitigate the substantial transmission losses that severely limit the expansion of interlayer distances and the number of layers in total. In essence, this enhancement significantly unleashes the potential of wave-based neural networks, allowing for greater depth and extended transmitting distances.

![](images/76c640800e785bb37265a1e878f781c458fda2b30b4101228ebb863d831e1ce9.jpg)

![](images/b2c67e6a9e9d67e4f69181ed335e047d3c663b64ab1ec8d58951824c4b2e5b31.jpg)
FIG. 4. Natural nonlinearity and potential applications in wave-based neural networks. (a) An illustration of wave-based neural network structure, which encompasses matrix multiplication and nonlinear activation. The waves that propagate from one layer to the next carry varying phase and magnitude information, acting as trainable weights between adjacent network layers. Herein, $X_{1}-X_{4}$ , $Y_{1}-Y_{4}$ , b, and f are inputs, outputs, bias, and activation, respectively. (b) With the input power linearly enlarging, the output power can increase nonlinearly.

Our designs inherently possess nonlinear properties. As depicted in Fig. 4(b), the output power, represented by the yellow dotted line, exhibits a nonlinear increase with input power rising. Contrary to the linear one plotted via orange, output power of nonlinearly activated units tends to ascend at a progressively slower rate, predominantly due to free electron saturations and power sensitivity of designed structures. Consequently, these units can be activated in a nonlinear manner. Moreover, this nonlinearity is rather stable in practical tests compared with their counterparts especially with extra lumped amplifier circuits, which is mainly because ours do not need a comparison-feedback loop [45,46].

Discussion—To sum up, we proposed a tunable gain metasurface induced by negative conductivity, covering both gain and lossy states. The feasibility of the proposed design, rooted in its ability to surpass conventional limits through negative conductivity, is also confirmed by macroscopic circuit analysis. The negative conductivity induced method can offer large tunable gain or loss without complex structures, massive elements, and realistic circuits. Hence, ours are easy to be generalized and manufactured. In experiments, we incorporated negative conductivity elements into fabricated metasurfaces and tested them in microwave bands to observe tunable gains. Furthermore, we explored and demonstrated nonlinearity of the metasurfaces, which is an open challenge in optical analogy computing. For future commercial usages, broader bandwidth and fewer deviations are suggested to be attained by using better medium plates and modern manufacturing techniques like surface mount technology. Looking forward, this work presents a novel scheme for gain metasurfaces, providing a versatile, lightweight and self-enhancing platform for future research in optics, communications, holography, and beyond.

Acknowledgments—The work was sponsored by the National Natural Science Foundation of China (NNSFC) under Grants No. 61975176 and No. 62101485, the Fundamental Research Funds for the Central Universities.

[1] A. Tuniz and B.T. Kuhlmey, Subwavelength terahertz imaging via virtual superlensing in the radiating near field, Nat. Commun. 14, 6393 (2023).

[2] O. Tsilipakos, A. C. Tasolamprou, A. Pitilakis, F. Liu, X. Wang, M. S. Mirmoosa, D. C. Tzarouchis, S. Abadal, H. Taghvaee, C. Liaskos, A. Tsioliaridou et al., Toward

intelligent metasurfaces: The progress from globally tunable metasurfaces to software-defined metasurfaces with an embedded network of controllers, Adv. Opt. Mater. 8, 2000783 (2020).

[3] T. Gu, H. J. Kim, C. Rivero-Baleine, and J. Hu, Reconfigurable metasurfaces towards commercial success, Nat. Photonics 17, 48 (2023).

[4] A. H. Dorrah and F. Capasso, Tunable structured light with flat optics, Science 376, eabi6860 (2022).

[5] Y. Zhang, C. Fowler, J. Liang, B. Azhar, M. Y. Shalaginov, S. Deckoff-Jones, S. An, J. B. Chou, C. M. Roberts, V. Liberman et al., Electrically reconfigurable non-volatile metasurface using low-loss optical phase-change material, Nat. Nanotechnol. 16, 661 (2021).

[6] X. Zhuang, W. Zhang, K. Wang, Y. Gu, Y. An, X. Zhang, J. Gu, D. Luo, J. Han, and W. Zhang, Active terahertz beam steering based on mechanical deformation of liquid crystal elastomer metasurface, Light Sci. Appl. 12, 14 (2023).

[7] Z. Fan, C. Qian, Y. Jia, Z. Wang, Y. Ding, D. Wang, L. Tian, E. Li, T. Cai, B. Zheng et al., Homeostatic neuro-metasurfaces for dynamic wireless channel management, Sci. Adv. 8, eabn7905 (2022).

[8] Y. Ota, K. Takata, T. Ozawa, A. Amo, Z. Jia, B. Kante, M. Notomi, Y. Arakawa, and S. Iwamoto, Active topological photonics, Nanophotonics 9, 547 (2020).

[9] W. Ma, F. Cheng, and Y. Liu, Deep-learning-enabled on-demand design of chiral metamaterials, ACS Nano 12, 6326 (2018).

[10] C. Qian, B. Zheng, Y. Shen, L. Jing, E. Li, L. Shen, and H. Chen, Deep-learning-enabled self-adaptive microwave cloak without human intervention, Nat. Photonics 14, 383 (2020).

[11] J. B. Khurgin, How to deal with the loss in plasmonics and metamaterials, Nat. Nanotechnol. 10, 2 (2015).

[12] S. Franca, V. Könye, F. Hassler, J. van den Brink, and C. Fulga, Non-Hermitian physics without gain or loss: The skin effect of reflected waves, Phys. Rev. Lett. 129, 086601 (2022).

[13] I. V. Doronin, E. S. Andrianov, and A. A. Zyablovsky, Overcoming the diffraction limit on the size of dielectric resonators using an amplifying medium, Phys. Rev. Lett. 129, 133901 (2022).

[14] S. Wuestner, A. Pusch, K. L. Tsakmakidis, J. M. Hamm, and O. Hess, Overcoming losses with gain in a negative refractive index metamaterial, Phys. Rev. Lett. 105, 127401 (2010).

[15] R. Phon, M. Lee, C. Lor, and S. Lim, Multifunctional reflective metasurface to independently and simultaneously control amplitude and phase with frequency tunability, Adv. Opt. Mater. 11, 2202943 (2023).

[16] J. Koelemeij, H. Dun, C. Diouf, E. Dierikx, G. Janssen, and C. Tiberius, A hybrid optical–wireless network for decimetre-level terrestrial positioning, Nature (London) 611, 473 (2022).

[17] C. Qian, Y. Yang, Y. Hua, C. Wang, X. Lin, T. Cai, D. Ye, E. Li, I. Kaminer, and H. Chen, Breaking the fundamental scattering limit with gain metasurfaces, Nat. Commun. 13, 4383 (2022).

[18] L. Zhang, Y. Yang, Y. Ge, Y. Guan, Q. Chen, Q. Yan, F. Chen, R. Xi, Y. Li, D. Jia et al., Acoustic non-Hermitian

skin effect from twisted winding topology, Nat. Commun. 12, 6297 (2021).

[19] S. K. Turitsyn, A. E. Bednyakova, and E. V. Podivilov, Nonlinear optical pulses in media with asymmetric gain, Phys. Rev. Lett. 131, 153802 (2023).

[20] X. Li, H. Q. Yang, R. W. Shao, F. Zhai, G. B. Liu, Z. X. Wang, H. F. Gao, G. Fan, J. W. Wu, Q. Cheng et al., Low-cost and high-performance 5-bit programmable phased array at ku-band, Prog. Electromagn. Res. 175, 29 (2022).

[21] J. A. Kong, Electromagnetic Wave Theory (EMW Publishing, Cambridge, Massachusetts, USA, 1986).

[22] K. Asano, T. Nakasha, and H. Wakatsuchi, Simplified equivalent circuit approach for designing time-domain responses of waveform-selective metasurfaces, Appl. Phys. Lett. 116, 171603 (2020).

[23] X. Qin, P. Fu, W. Yan, S. Wang, Q. Lv, and Y. Li, Negative capacitors and inductors enabling wideband waveguide metatronics, Nat. Commun. 14, 7041 (2023).

[24] L. Esaki, New phenomenon in narrow germanium $p - n$ junctions, Phys. Rev. 109, 603 (1958).

[25] K. Koshelev, S. Kruk, E. Melik-Gaykazyan, J. Choi, A. Bogdanov, H. Park, and Y. Kivshar, Subwavelength dielectric resonators for nonlinear nanophotonics, Science 367, 288 (2020).

[26] F. Ding, A. Pors, and S. I. Bozhevolnyi, Subwavelength dielectric resonators for nonlinear nanophotonics, Gradient metasurfaces: A review of fundamentals and applications, Rep. Prog. Phys. 81, 026401 (2018).

[27] Z. Fan, Y. Jia, H. Chen, and C. Qian, Spatial multiplexing encryption with cascaded metasurfaces, J. Opt. 25, 125105 (2023).

[28] H. Lu, J. Zhao, B. Zheng, C. Qian, T. Cai, E. Li, and H. Chen, Eye accommodation-inspired neuro-metasurface focusing, Nat. Commun. 14, 3301 (2023).

[29] C. Qian, Z. Wang, H. Qian, T. Cai, B. Zheng, X. Lin, Y. Shen, I. Kaminer, E. Li, and H. Chen, Dynamic recognition and mirage using neuro-metamaterials, Nat. Commun. 13, 2694 (2022).

[30] X. Lin, Y. Rivenson, N. T. Yardimei, M. Veli, Y. Luo, M. Jarrahi, and A. Ozcan, All-optical machine learning using diffractive deep neural networks, Science 361, 1004 (2018).

[31] C. Qian, X. Lin, X. Lin, J. Xu, Y. Sun, E. Li, B. Zhang, and H. Chen, Performing optical logic operations by a diffractive neural network, Light Sci. Appl. 9, 59 (2020).

[32] L. Portilla, K. Loganathan, H. Faber, A. Eid, J. G. D. Hester, M. M. Tentzeris, M. Fattori, E. Cantatore, C. Jiang, A. Nathan et al., Wirelessly powered large-area electronics for the internet of things, Nat. Electron. 6, 10 (2023).

[33] See Supplemental Material at http://link.aps.org/supplemental/10.1103/PhysRevLett.133.113801, which includes Refs. [34–43], for additional information about the details of theoretical analysis and experiments.

[34] F. Z. Goffi, K. Mnasri, M. Plum, C. Rockstuhl, and A. Khrabustovskyi, Towards more general constitutive relations for metamaterials: A checklist for consistent formulations, Phys. Rev. B 101, 195411 (2020).

[35] D. M. Pozar, Microwave Engineering, 4th ed. (John Wiley & Sons Inc., University of Massachusetts at Amherst, 2011).

[36] F. Han, L. Yin, C. Du, and P. Liu, Robust effective-medium characteristics of bianisotropic reflective metasurfaces based on field-circuit combined analysis, Adv. Theory Simul. 4, 2000246 (2021).

[37] N. H. L. Koster and R. H. Jansen, The equivalent circuit of the asymmetrical series gap in microstrip and suspended substrate lines, IEEE Trans. Microwave Theory Tech. 30, 1273 (1982).

[38] K. Sarabandi and N. Behdad, A frequency selective surface with miniaturized elements, IEEE Trans. Antennas Propag. 55, 1239 (2007).

[39] C. Huang, B. Sun, W. Pan, J. Cui, X. Wu, and X. Luo, Dynamical beam manipulation based on 2-bit digitally-controlled coding metasurface, Sci. Rep. 7, 42302 (2017).

[40] C. H. Joseph, D. Mencarelli, L. Pierantoni, P. Russo, and L. Zappelli, Identification of compact equivalent circuit model for metamaterial structures, IEEE Trans. Antennas Propag. 71, 5850 (2023).

[41] G. E. Bonacchini and F. G. Omenetto, Reconfigurable microwave metadevices based on organic electrochemical transistors, Nat. Electron. 4, 424 (2021).

[42] B. Lv, J. Fu, B. Wu, R. Li, Q. Zeng, X. Yin, Q. Wu, L. Gao, W. Chen, Z. Wang et al., Unidirectional invisibility induced by parity-time symmetric circuit, Sci. Rep. 7, 40575 (2017).

[43] Z. Song, J. Zhu, X. Wang, R. Zhang, P. Min, W. Cao, Y. He, J. Han, T. Wang, J. Zhu et al., Origami metamaterials for ultra-wideband and large-depth reflection modulation, Nat. Commun. 15, 3181 (2024).

[44] Y. D. Chong, L. Ge, and A. D. Stone, PT-symmetry breaking and laser-absorber modes in optical scattering systems, Phys. Rev. Lett. 106, 093902 (2011).

[45] C. Liu, Q. Ma, Z. J. Luo, Q. R. Hong, Q. Xiao, H. C. Zhang, L. Miao, W. M. Yu, Q. Cheng, L. Li et al., A programmable diffractive deep neural network based on a digital-coding metasurface array, Nat. Electron. 5, 113 (2022).

[46] Y. Li, S. Wang, H. Wang, H. Li, and T. Cui, Nonreciprocal control of electromagnetic polarizations applying active metasurfaces, Adv. Opt. Mater. 10, 21021154 (2022).
