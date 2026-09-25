# Observation of Coherent Perfect Acoustic Absorption at an Exceptional Point

Yi-Fei Xia, $^{1,*}$ Zi-Xiang Xu, $^{1,*}$ Yu-Ting Yan $^{ID}$ , $^{1,*}$ An Chen, $^{1}$ Jing Yang, $^{1,\dagger}$ Bin Liang, $^{1,\ddagger}$

Jian-Chun Cheng $^{1,\S}$ , and Johan Christensen $^{2,\parallel}$

$^{1}$ Collaborative Innovation Center of Advanced Microstructures and Key Laboratory of Modern Acoustics, MOE, Institute of Acoustics,

Department of Physics, Nanjing University, Nanjing 210093, People's Republic of China

$^{2}$ IMDEA Materials Institute, Calle Eric Kandel, 2, 289006, Getafe, Madrid, Spain

(Received 29 November 2024; revised 10 June 2025; accepted 18 July 2025; published 6 August 2025)

Non-Hermitian systems have recently shown new possibilities to manipulate wave scattering by exploiting loss, yet coherent perfect absorption at an exceptional point (CPA EP) remains elusive in acoustics. Here, we demonstrate it based on a two-channel waveguide with compact lossy resonators. We realize imbalanced losses crucial for CPA EP by using active components to independently modulate the non-Hermiticity. The CPA EP experimentally manifests as full absorption at a unique real frequency and shows high sensitivity to the incident phase variations. Our findings open an avenue to explore novel non-Hermitian physics for classical waves and develop innovative acoustic singularity-based devices.

DOI: 10.1103/slhy-f76q

Introduction—The Hermiticity of a Hamiltonian ensures the conservation of energy and shapes the physical reality in many systems $[1,2]$ . However, in nonconservative systems, interaction with the environment leads to non-Hermitian dynamics. The past decade has witnessed a surge of research on non-Hermitian physics $[3–5]$ , resulting in unprecedented principles, phenomena, and applications in both quantum and classical systems $[6–8]$ . By tailoring gain and loss, non-Hermitian systems exhibit intriguing phenomena near exceptional points (EPs), which are singularities in the parameter space where multiple eigenvalues and the associated eigenstates coalesce $[9–16]$ . This coalescence is accompanied by a plethora of exotic phenomena, such as the skin effect $[17–20]$ , chiral state transfer $[21,22]$ , and non-Abelian braiding $[23–25]$ . Moreover, the introduction of non-Hermiticity has given rise to various applications related to EPs in open wave systems, including unidirectional invisibility, single-mode lasing, and enhanced sensing $[26–28]$ .

As a typical non-Hermitian effect in wave physics, coherent perfect absorption (CPA) occurs under purely incoming boundary conditions when the zeros of the scattering matrix lie on the real frequency axis $[29]$ , which has attracted rapidly growing attention in the past few years and been extended into diverse fields including acoustics and mechanical waves $[30]$ , microwaves $[31]$ , and optical waves $[32,33]$ . Thanks to the intrinsic stability of CPA ensured by the wave interference that entraps the incident radiation inside the lossy media, it provides the possibility that two purely incoming solutions coalesce at a unique real frequency $[2]$ , referred to as CPA EP. This phenomenon has been recently predicted and observed in optics based on bulky ring resonators with coupled non-Hermitian parameters $[1,34]$ . However, given the macroscopic wavelength of sound waves and the difficulty in independently controlling non-Hermiticity with precision and flexibility, realization of CPA EP in practical acoustic scattering systems remains elusive.

Here, we propose an acoustic non-Hermitian scattering system consisting of two-channel waveguides coupled to imbalanced lossy resonant cavities to achieve CPA EP. Based on the coupled-mode theory, we analytically derive the scattering matrix of the system and predict the critical conditions for CPA EP. Furthermore, we introduce a metamaterial-based implementation that is significantly smaller in size compared to its optical counterparts, with coupling strengths and intrinsic losses independently tunable through active acoustic units. We experimentally observe the occurrence of CPA EP in this system, showing the expected strong absorption at the single resonant frequency and distinctive sensitivity to the phase variations of incoming acoustic waves, which is consistent with theoretical and simulation results. Our Letter provides deeper insight to the non-Hermitian physics in classical wave systems and opens new avenues for precise manipulation of coherent acoustic waves or phonons, especially at high frequencies where coherent control becomes expensive, thereby enabling the design and practical application of novel acoustic absorbers, sensors, and directional devices, etc.

Acoustic non-Hermitian scattering system for realizing CPA EP—To realize CPA EP, we propose an acoustic non-Hermitian scattering system, as illustrated in Fig. 1(a), where two coherent acoustic plane waves normally impinge on both sides of the system along opposite propagation channels. These incoming acoustic waves induce lossy resonance in the double cavities, resulting in destructive interference effects that trap and dissipate the radiation within the acoustic scattering system. By adjusting the non-Hermiticity of the system, the resonant modes in the cavities could become degenerate, leading to the emergence of CPA EP, which manifests as perfect absorption at a real frequency. Specifically, the absorption behavior of such a system is described by a scattering matrix S with reflection and transmission coefficients. Then the amplitudes of incoming and outgoing acoustic waves can be related by S as

![](images/bcb880413de113d391511702507507633414e46bb590d7d71dc4e59106f3558c.jpg)

![](images/9de4bd92305da1a72e42f58b20b6dd25528633506f2ac0da55ee2436e9d5e349.jpg)

![](images/0d4b99b9bc3626603ba40cbaea958ec9a250a0b5d703e4aad54023a117480232.jpg)

![](images/3da6d88321cc5a0c6524822d117ee6bbe05c95de6e52377ec4d88eec250d0f91.jpg)

![](images/34214b80121ef24e55554429497cb7d9c0fb71db823b8d62b1fff6ea6f04c231.jpg)
FIG. 1. (a) A schematic diagram of the acoustic non-Hermitian scattering system composed of two-side acoustic propagating channels connected by two coupled resonant modes. The two gray regions represent two propagating channels, and the pink and red parts describe the acoustic resonant modes with imbalanced losses separately. The two-way arrows depict the coupling between acoustic resonant states or between acoustic resonant state and propagation channel. (b),(c) Phase transition diagrams for the real and imaginary parts of $\omega_{z1,2}$ as functions of normalized coupling strength $\kappa/\kappa_{\mathrm{th}}$ , respectively. (d),(e) Phase transition diagrams for the real and imaginary parts of the two eigenvalues passing through a CPA EP as functions of the normalized intrinsic loss $\gamma_1/\gamma_{1\mathrm{th}}$ . (f) Outgoing acoustic intensity in the two channels varied with frequency. The outgoing acoustic intensity reaches its minimum value of zero when $\omega = \omega_z$ , indicating the occurrence of a CPA EP.

$$
\binom{p _ {\mathrm{o} 1}}{p _ {\mathrm{o} 2}} = S \binom{p _ {\mathrm{i} 1}}{p _ {\mathrm{i} 2}} = \left( \begin{array}{c c} r _ {1} & t _ {2} \\ t _ {1} & r _ {2} \end{array} \right) \binom{p _ {\mathrm{i} 1}}{p _ {\mathrm{i} 2}},\tag{1}
$$

where $p_{i1,2}$ and $p_{o1,2}$ respectively, refer to the acoustic pressure for the incoming and outgoing ones, $r_{1,2}$ and $t_{1,2}$ signify the acoustic reflection and transmission coefficients, respectively, and the subscripts 1 and 2 represent the different acoustic propagating channels for such an acoustic non-Hermitian scattering system.

The scattering characteristics of this system depend on the resonance frequency, coupling strength, and intrinsic losses of the two cavities, which can be described using a second-order non-Hermitian Hamiltonian [35]:

$$
H = \left( \begin{array}{c c} \omega_ {1} - i (\gamma_ {1} + \gamma_ {\mathrm{c} 1}) & \kappa \\ \kappa & \omega_ {2} - i (\gamma_ {2} + \gamma_ {\mathrm{c} 2}) \end{array} \right).\tag{2}
$$

Here, $\omega_{1,2}$ are the inherent resonant frequencies of the cavities in the absence of any coupling or losses. For compact subwavelength cavities, their resonance frequencies are primarily determined by their dimensions and geometries. $\kappa$ denotes the internal coupling strength, reflecting the transfer of acoustic energy between the two cavities. $\gamma_{1}$ and $\gamma_{2}$ represent the intrinsic losses of the cavities resulting from acoustic dissipation mechanisms, which can be adjusted using active acoustic units. $\gamma_{c1}$ and $\gamma_{c2}$ denote the external losses between the cavities and their corresponding external channels, describing how acoustic energy leaks out from each cavity into the channels.

Based on the coupled-mode theory [36], the scattering matrix S can be expressed as follows:

$$
S = 1 - i K ^ {\dagger} \frac {1}{\omega - H} K,\tag{3}
$$

where $K = \text{diag}\left(\sqrt{2\gamma_{c1}}, \sqrt{2\gamma_{c2}}\right)$ is the coupling operator, and $\omega$ is the frequency of incident monochromatic acoustic wave [37]. This expression explicitly describes the relationship between the scattering matrix and the system Hamiltonian. Specifically, when, the scattering matrix and the Hamiltonian share the same eigenvectors, and the corresponding eigenvalues satisfy the relationship as

$$
\lambda_ {S} = 1 - i \frac {2 \gamma_ {\mathrm{c} 1}}{\omega - \lambda_ {H}},\tag{4}
$$

where $\lambda_{S}$ and $\lambda_{H}$ are the eigenvalues of the scattering matrix and the Hamiltonian, respectively.

When the eigenvalue of the scattering matrix is zero, i.e., $\lambda_{S}=0$ , the system eliminates the outgoing acoustic waves and achieves CPA. According to Eq. (4), the frequency at which CPA occurs is closely related to the eigenvalue of the Hamiltonian, given by $\omega_{z}=\lambda_{H}+2i\gamma_{c1}$ . For simplicity while without losing generality, we consider a special case where the two resonant frequencies are equal, i.e., $\omega_{1}=\omega_{2}=\omega_{0}$ . In this case, the frequency at which CPA occurs can be calculated as follows:

$$
\begin{array}{l} \omega_ {\mathrm{z1,2}} = \omega_ {0} + i \frac {\gamma_ {\mathrm{c1}} + \gamma_ {\mathrm{c2}} - \gamma_ {1} - \gamma_ {2}}{2} \\ \pm \frac {1}{2} \sqrt {(i (\gamma_ {\mathrm{c1}} - \gamma_ {1} - \gamma_ {\mathrm{c2}} + \gamma_ {2})) ^ {2} + 4 \kappa^ {2}}. \end{array}\tag{5}
$$

In general, there exist two distinct frequencies $\omega_{z1,2}$ both satisfying the CPA conditions, corresponding to the two zeros of the transfer function [37]. However, what is noteworthy is that when the eigenvalues of the Hamiltonian and the scattering matrix simultaneously degenerate, these two zeros merge at the real frequency, resulting in the occurrence of CPA EP. Equation (5) gives the critical condition for CPA EP, expressed as $\gamma_{1} + \gamma_{2} = \gamma_{c1} + \gamma_{c2}$ and $\kappa = |\gamma_{1} - \gamma_{c1}|$ [37]. The phase transition diagrams for the real and imaginary parts of the two solutions $\omega_{z1,2}$ are separately demonstrated in Figs. 1(b) and 1(c), as functions of the normalized coupling strength $\kappa/\kappa_{th}$ . With the increase of $\kappa$ , the real parts of $\omega_{z1,2}$ keep degenerate until $\kappa = \kappa_{th}$ , and in the regime of $\kappa > \kappa_{th}$ , they become divided and are located on either side of $\mathrm{Re}(\omega) = \omega_{0}$ . Opposite to the dependence of the real parts on the coupling strength $\kappa$ , the imaginary parts of $\omega_{z1,2}$ separate at the beginning and coalesce when $\kappa = \kappa_{th}$ . In the following, we investigate the degenerate behavior of $\omega_{z1,2}$ as $\gamma_{1}$ varies and plot typical results in Figs. 1(d) and 1(e). When gradually increasing the intrinsic loss $\gamma_{1}$ , we can find two EPs at $\omega_{1} \approx \omega_{0}$ and $\omega_{2} \approx \omega_{0}(1 - i1.8)$ . The former is a CPA EP where the zeros are at a real frequency, while the latter is a general EP which cannot be observed experimentally due to being located at a complex frequency. Similarly, the above trend of zero merging can also be observed in phase transition diagrams when tuning other system parameters, such as intrinsic losses $\gamma_{1,2}$ or coupling strengths $\gamma_{c1,2}$ .

In the case of CPA EP, both eigenvalues of the scattering matrix are zero, corresponding to the same eigenvector $v = [1, -i]^{T}$ [37]. This implies that when the incoming acoustic waves matches this eigenvector, the intensity of the outgoing sound waves reaches its minimum value of zero at frequency $\omega = \omega_{z}$ , as shown in Fig. 1(f). At this point, the scattering matrix must take the following form:

$$
S = C \left( \begin{array}{c c} i & 1 \\ 1 & - i \end{array} \right),\tag{6}
$$

where C is a constant. Notably, the scattering matrix is a nilpotent matrix $(S^{2}=0)$ , which implies that complete absorption can be achieved by cascading two such systems. This nilpotent property enables the transformation of any incoming waves into phase-orthogonal outgoing waves. For general CPA, the phases of the outgoing acoustic waves vary with that of the incoming acoustic waves. For CPA EP, however, the phases of the incoming waves only affect the intensity of the outgoing acoustic waves. When the incoming acoustic waves in the two channels have equal amplitudes but a phase difference of $\Delta\varphi$ , the intensities of the outgoing acoustic waves depend on $\Delta\varphi$ through the relationship $|p_{o}|^{2}\propto|i+\exp(i\Delta\varphi)|^{2}$ , and are therefore highly sensitive to phase variations.

Metamaterial-based acoustic implementation—For practical implementation of the acoustic non-Hermitian scattering system, we employ a metamaterial-based approach to construct a four-port acoustic cavity-tube model with subwavelength dimensions. Despite slight differences in scattering matrices from the two-port theoretical model, this four-port configuration does not affect the demonstration of CPA EP because (1) both models exhibit eigenvalue degeneracy at the same real frequency, and (2) their Hamiltonians are related by a similarity transformation. Here, we present the two-port model for brevity, while the full four-port theory is provided in Supplemental Material [37]. The proposed metamaterial model is composed of two-channel waveguides linked to coupled acoustic resonant cavities, as illustrated in Fig. 2(a). This model enables independent modulation of each system parameter in the effective Hamiltonian given in Eq. (2), allowing CPA EP to occur at unique real frequency. Specifically, the resonant frequency $\omega_0$ of the two identical cavities coupled with each other can be modulated by designing the inner geometries of cavities. These two compact resonant cavities are connected by a small tube to implement the intercavity coupling strength $\kappa$ . Two single-mode waveguides for sound propagation are linked to the cavities by two small tubes with the coupling strengths being $\gamma_{\mathrm{c}1}$ and $\gamma_{\mathrm{c}2}$ , respectively. Notice that the coupling strength is inversely proportional to the length of the connecting tube. The intrinsic losses $\gamma_{1,2}$ of the two cavities are modified by controlling the attenuation of sound waves inside the cavities which is realized in simulation by adding a modifiable imaginary part to the speed of sound and experimentally implemented via producing the sound interference effect.

(b)
![](images/2d7f7d4e38f69b35446b33c49af244f9af70c3bdd3098bd769bbd0038890eb70.jpg)

![](images/adedc23ec57ddb0c7a7edef5d0202db08b94bbb16080fe81e2650c97fbee5b61.jpg)
(d)

![](images/d54a117b7770c6cb2b4ae044bc3a72a9f1613bd62cd868a02d7860b846f68950.jpg)

![](images/d4379777aea5cd22e8e7857d9ded3b25b345e97535e51d174ff625ad5194eea6.jpg)
FIG. 2. (a) Schematic diagram of the practical implementation which is realized with coupled acoustic resonant cavities connected with double rectangular waveguides as the input and output channels. (b) Trajectories of the eigenvalues of S as $\gamma_{1}$ varies. (c) Theoretical and simulated reflection spectra $R_{1}=|r_{1}|^{2}$ and transmission spectra $T_{1}=|t_{1}|^{2}$ by exciting port 1 only when $\gamma_{1}=9~Hz,\gamma_{2}=31~Hz,\gamma_{c1}=\gamma_{c2}=20~Hz,\kappa=11~Hz$ , and $\omega_{0}=1621~Hz$ . (d) Theoretical and simulated reflection spectra $R_{2}=|r_{2}|^{2}$ and transmission spectra $T_{2}=|t_{2}|^{2}$ by exciting port 3 only.

Here, we select the system parameters as $\gamma_{2}=31$ Hz, $\gamma_{c1}=\gamma_{c2}=20$ Hz, $\kappa=11$ Hz, and $\omega_{0}=1621$ Hz, and analyze the influence of the variable parameter $\gamma_{1}$ on the degeneracy of the zeros. The trajectories of eigenvalues $\omega_{z1,2}$ under the variation of $\gamma_{1}$ are illustrated in Fig. 2(b). Initially, the two complex eigenvalues are separately distributed at two starting points and then approach each other as $\gamma_{1}$ increases. Under the above derived conditions for CPA EP ( $\gamma_{c1}+\gamma_{c2}=\gamma_{1}+\gamma_{2}$ and $\kappa=|\gamma_{c1}-\gamma_{1}|$ ), which is fulfilled when $\gamma_{1}=9$ Hz, two zeros condense at a real frequency. Based on the above analyses, it is easy to give a feasible configuration of system parameters in the practical design satisfying the critical conditions of CPA EP. Under this configuration, we showcase the simulated reflection spectra $R_{1,2}$ and transmission spectra $T_{1,2}$ in Figs. 2(c) and 2(d) when the incident wave is emitted only from port 1 and port 3, respectively. Simulated and theoretical results agree well with each other, with both showing that $R_{1,2}$ and $T_{1,2}$ are almost equal at the resonant frequency $\omega_{0}$ . Despite some discrepancies between the theoretical and simulated results of $R_{2}$ stemming from backscattering effects, the eigenvalues of the scattering matrix remain unaffected, ensuring the occurrence of CPA EP.

Experimental observation of acoustic CPA EP—Based on the theoretical analysis above, achieving CPA EP necessitates imbalanced internal losses between the two resonator cavities. However, since both resonant cavities have identical structures, their inherent losses are approximately equal. To realize the imbalanced losses which are crucial for the experimental observation of CPA EP, we construct a feedback control system by introducing an active acoustic unit to achieve accurate and decoupled control of non-Hermiticity in a cavity with fixed physical dimension, as shown in Fig. 3(a). The active acoustic unit composed of a transmitting transducer, a receiving transducer, and a feedback circuit is installed at the top of each cavity. The amplitude and phase of the emission are precisely controlled according to the signal measured by the receiving transducer, which drives the transmitting transducer to generate waves that destructively interfere with the original cavity modes, thereby effectively producing the required acoustic loss. This feedback engineering strategy decouples the intrinsic losses within the cavities from the cavity-tube coupling, enabling independent control of the non-Hermicity and ensuring the stability of our proposed non-Hermitian system. Through finite element simulation of the dynamics of such a system $[37]$ , we obtain the optimal experimental settings to meet the predicted conditions for CPA EP. When the plane wave is emitted at ports 1 or 3 only and the porous materials are added at ports 2 and 4 for avoiding the undesirable reflections. The reflection and transmission spectra $R_{1,2}$ and $T_{1,2}$ are measured, which are illustrated in Figs. 3(b) and 3(c). The measured spectra show good consistency with the simulated ones, where only a slight deviation occurs owing to unavoidable fabrication errors. This indicates that the current configuration satisfies the critical conditions of the CPA EP.

![](images/bd9512211d3aa1b36806a405cc4d274657ea75d76451857c7d16b1f8390a0f04.jpg)

(b)
(c)
![](images/48902e4a56c0902555f0a3529b95ecca1da1cd187afadffd9e1d75af79d1512e.jpg)

![](images/8bca7faae9e719c5f57631ba8591364bbde76d2333b47b865acfcf88559b14d3.jpg)
FIG. 3. (a) Photograph of the experimental setup. (b) Experimentally measured reflection and transmission spectra by exciting port 1 only. For accurate estimation of the reflection and transmission coefficients, we normalize the acoustic pressure measured in the system with respect to that in an empty waveguide. (c) Experimentally measured spectra of reflection and transmission by exciting port 3 only.

Considering that the coherent absorption behavior is sensitive to the relative phase $\Delta\varphi$ and amplitude ratio $p_{0}$ of incoming acoustic wave $p_{i2}$ with respect to the other one $p_{i1}$ , we inspect the total output power versus the input relative phase and amplitude ratio in Fig. 4(a). Perfect absorption can be achieved when incoming acoustic waves match the eigenvector $v = [1, -i]^{T}$ , which is expected to be observed at the minimum point on the output surface in Fig. 4(a). In Fig. 4(b), we further depict the relationship between the output spectrum and relative phase under equal incident amplitude. The output exhibits a sinusoidal pattern over the phase range and the expected perfect absorption is obtained with relative phase of $-0.5\pi$ , as verified by the good agreement between simulation and experiment. This demonstrates the extreme phase sensitivity of CPA EP, which is a key characteristic unachievable with conventional EPs and may hold great promise for acoustic sensing and communication that typically rely on phase detection.

(a)
![](images/a3d5257a7ac704b2e1ddcf2132917f947a3486cafc5a66008cb137b62a40565c.jpg)

(b)
![](images/375c0afcdd6daa6abff88dbc87909f97aee2cdbd6926a6986ba842e117a4127f.jpg)

(c)
![](images/a1ad55ea64ef19d17875d1305d074cf82e742362f0ff3b84f815e12dbed89a65.jpg)

(d)
![](images/d9184f233562d61ff8caab170304c750d11167cb5c4ee2aaacd76895972f12f7.jpg)
FIG. 4. (a) Theoretical total output power normalized to the total input power at the zero detuning $(\delta = \omega - \omega_{0} = 0)$ as a function of the input amplitude ratio and relative phase. The total output power $P_{t}$ is the sum of the output $P_{2}$ from port 2 and output $P_{4}$ from port 4. (b) Theoretical, simulated, and experimental output spectra as a function of the relative phase under equal incident amplitude. (c), (d) Simulated and experimental spectra of the output power at the CPA EP.

In view of the dependence analysis on the incident wavefront, we properly tune the incoming waves generated by the transmitters with correct input power and relative phase, and perfect absorption is expected to be achieved due to a combination effect of interference and dissipation. The reflected part of the incident wave from port 1 interferes destructively with the transmitted part of the incident one from port 3, and vice versa, and therefore the radiation is trapped in an interference pattern within the lossy system and lost entirely to dissipation. Theoretical calculations indicate that CPA EP exhibits a quartic line-shape near the degenerate frequency [37], which significantly broadens the absorption bandwidth with the relative bandwidth being more than twice that of traditional CPA, while also enhancing sensitivity to frequency changes. Experimental results displayed in Figs. 4(c) and 4(d), in good agreement with the theoretical and simulated ones, show nearly perfect absorption at the resonant frequency. The negligible errors are attributed to inevitable fluctuations of the incident wave phases and inaccuracies in tuning system non-Hermiticity via the electric circuit.

Conclusions—In summary, we theoretically present an acoustic non-Hermitian scattering system composed of two-channel waveguides coupled to lossy resonant cavities to observe CPA EP. As a practical implementation of this system, a compact metamaterial-based model is proposed which allows independent modulation of the system parameters, where the precise adjustment of non-Hermiticity is enabled by the introduction of active acoustic components. We experimentally validate the occurrence of CPA EP, with measured results closely aligning with theoretical predictions and simulations, demonstrating strong absorption at the expected real frequency and extreme sensitivity to the phase variations of incoming acoustic waves. Our Letter enriches the non-Hermitian physics in classical wave systems and provides an important platform for investigation of intriguing phenomena occurring at EPs, which opens avenue for the design and application of singularity-based devices.

Acknowledgments—This work was supported by the National Key R&D Program of China (Grant No. 2022YFA1404402), the National Natural Science Foundation of China (Grant No. 12174190), High-Performance Computing Center of Collaborative Innovation Center of Advanced Microstructures, and A Project Funded by the Priority Academic Program

Development of Jiangsu Higher Education Institutions. J. C. acknowledges support from the Spanish Ministry of Science and Innovation through a Consolidación Investigadora grant (No. CNS2022-135706).

[1] V. Achilleos, G. Theocharis, O. Richoux, and V. Pagneux, Phys. Rev. B 95, 144303 (2017).

[2] W. R. Sweeney, C. W. Hsu, S. Rotter, and A. D. Stone, Phys. Rev. Lett. 122, 093901 (2019).

[3] H. Cao and J. Wiersig, Rev. Mod. Phys. 87, 61 (2015).

[4] Z. Gu, H. Gao, P.-C. Cao, T. Liu, X.-F. Zhu, and J. Zhu, Phys. Rev. Appl. 16, 057001 (2021).

[5] W. D. Heiss, J. Phys. A 45, 444016 (2012).

[6] R. El-Ganainy, K. G. Makris, M. Khajavikhan, Z. H. Musslimani, S. Rotter, and D. N. Christodoulides, Nat. Phys. 14, 11 (2018).

[7] L. Feng, R. El-Ganainy, and L. Ge, Nat. Photonics 11, 752 (2017).

[8] L. Xiao, X. Zhan, Z. H. Bian, K. K. Wang, X. Zhang, X. P. Wang, J. Li, K. Mochizuki, D. Kim, N. Kawakami, W. Yi, H. Obuse, B. C. Sanders, and P. Xue, Nat. Phys. 13, 1117 (2017).

[9] M. A. Miri and A. Alù, Science 363, eaar7709 (2019).

[10] I. Rotter, J. Phys. A 42, 135302 (2009).

[11] X. Zhang, F. Zangeneh-Nejad, Z.-G. Chen, M.-H. Lu, and J. Christensen, Nature (London) 618, 687 (2023).

[12] C. M. Bender and S. Boettcher, Phys. Rev. Lett. 80, 5243 (1998).

[13] Z. Lin, J. Schindler, F. M. Ellis, and T. Kottos, Phys. Rev. A 85, 050101 (2012).

[14] J. Doppler, A. A. Mailybaev, J. Böhm, U. Kuhl, and S. Rotter, Nature (London) 537, 76 (2016).

[15] Y. Huang, C. Min, and G. Veronis, Opt. Express 24, 22219 (2016).

[16] B. Peng, A. K. Özdemir, M. Liertzer, W. Chen, and L. Yang, Proc. Natl. Acad. Sci. U.S.A. 113, 6845 (2016).

[17] N. Okuma, K. Kawabata, K. Shiozaki, and M. Sato, Phys. Rev. Lett. 124, 086801 (2020).

[18] L. Zhang et al., Nat. Commun. 12, 6297 (2021).

[19] T. Hofmann et al., Phys. Rev. Res. 2, 023265 (2020).

[20] W. Zhu, X. Fang, D. Li, Y. Sun, Y. Li, Y. Jing, and H. Chen, Phys. Rev. Lett. 121, 124501 (2018).

[21] W. Wang, X. Wang, and G. Ma, Nature (London) 608, 50 (2022).

[22] N. Moiseyev, Non-Hermitian Quantum Mechanics (Cambridge University Press, Cambridge, England, 2011).

[23] X.-L. Zhang, F. Yu, Z.-G. Chen, Z.-N. Tian, Q.-D. Chen, H.-B. Sun, and G. Ma, Nat. Photonics 16, 390 (2022).

[24] C. Guria, Q. Zhong, S. K. Ozdemir, Y. S. S. Patil, R. El-Ganainy, and J. G. E. Harris, Nat. Commun. 15, 1369 (2024).

[25] K. Wang, J. L. K. König, K. Yang, L. Xiao, W. Yi, E. J. Bergholtz, and P. Xue, arXiv:2410.08191.

[26] W. Chen, Ş. K. Özdemir, G. Zhao, J. Wiersig, and L. Yang, Nature (London) 548, 192 (2017).

[27] M. P. Hokmabadi, A. Schumer, D. N. Christodoulides, and M. Khajavikhan, Nature (London) 576, 70 (2019).

[28] M. C. Rechtsman, Nature (London) 548, 161 (2017).

[29] Y. D. Chong, L. Ge, H. Cao, and A. D. Stone, Phys. Rev. Lett. 105, 053901 (2010).

[30] G. Ma, M. Yang, S. Xiao, Z. Yang, and P. Sheng, Nat. Mater. 13, 873 (2014).

[31] N. I. Landy, S. Sajuyigbe, J. J. Mock, D. R. Smith, and W. J. Padilla, Phys. Rev. Lett. 100, 207402 (2008).

[32] M. Cai, O. Painter, and K. J. Vahala, Phys. Rev. Lett. 85, 74 (2000).

[33] B. C. Sturmberg, T. K. Chong, D.-Y. Choi, T. P. White, L. C. Botten, K. B. Dossou, C. G. Poulton, K. R. Catchpole, R. C. McPhedran, and C. Martijn de Sterke, Optica 3, 556 (2016).

[34] C. Wang, W. R. Sweeney, A. D. Stone, and L. Yang, Science 373, 1261 (2021).

[35] J.-J. Liu, Z.-W. Li, Z.-G. Chen, W. Tang, A. Chen, B. Liang, G. Ma, and J.-C. Cheng, Phys. Rev. Lett. 129, 084301 (2022).

[36] K. J. Dean, Phys. Bull. 35, 339 (1984).

[37] See Supplemental Material at http://link.aps.org/supplemental/10.1103/slhy-f76q for (1) analytical derivation of the two-channel scattering matrix based on coupled-mode theory; (2) dependence of the scattered output

spectrum on zeros and poles; (3) critical conditions for CPA EP; (4) theoretical calculation of output spectra on eigenanalysis of scattering matrix; (5) acoustic four-port metamaterial-based model of non-Hermitian scattering system; (6) retrieval of system parameters in simulation and experiment; (7) stability analysis of active acoustic metamaterial model; and (8) bandwidth broadening and sensitivity enhancing characteristics of CPA EP, which includes Refs. [38–43].

[38] H. A. Haus, Waves and Fields in Optoelectronics (Prentice-Hall, New York, 1984).

[39] S. Zanotto and A. Tredicucci, Sci. Rep. 6, 24592 (2016).

[40] W. Tang, K. Ding, and G. Ma, Phys. Rev. Lett. 127, 034301 (2021).

[41] K. Ding, G. Ma, M. Xiao, Z. Q. Zhang, and C. T. Chan, Phys. Rev. X 6, 021007 (2016).

[42] W. Tang, X. Jiang, K. Ding, Y.-X. Xiao, Z.-Q. Zhang, C. T. Chan, and G. Ma, Science 370, 1077 (2020).

[43] A. Jenkins, Phys. Rep. 525, 167 (2013).
