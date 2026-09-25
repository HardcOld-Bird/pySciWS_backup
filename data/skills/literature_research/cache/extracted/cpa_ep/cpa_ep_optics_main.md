OPTICS

# Coherent perfect absorption at an exceptional point

Changqing Wang $^{1}$ , William R. Sweeney $^{2,3}$ , A. Douglas Stone $^{2,3}$ , Lan Yang $^{1*}$

Recently, exceptional points, a degeneracy of open wave systems, have been observed in photonics, acoustics, and electronics. They have mainly been realized as a degeneracy of resonances; however, a degeneracy associated with the absorption of waves can exhibit distinct and interesting physical features. Here, we demonstrate such an absorbing exceptional point by engineering degeneracies in the absorption spectrum of dissipative optical microcavities. We experimentally distinguished the conditions to realize an absorbing exceptional point versus a resonant exceptional point. Furthermore, when the optical loss was tuned to achieve perfect absorption at an absorbing exceptional point, we observed its signature, an anomalously broadened line shape in the absorption spectrum. The distinct scattering properties of the absorbing exceptional point create opportunities for both fundamental study and applications of non-Hermitian degeneracies.

The field of non-Hermitian physics provides a fertile ground for studying unconventional physical phenomena in open systems (1, 2). Many of the most intriguing phenomena occur at exceptional points (EPs), singularities in the parameter space of a wave system at which two or more discrete eigenvalues and the associated eigenstates of a non-Hermitian system coalesce (3). The most familiar experimental examples are EPs associated with the degeneracies of the resonance frequencies of wave systems corresponding to the discrete, purely outgoing solutions of the relevant wave equations. Such resonant EPs, which often are described within the temporal coupled-mode theory (TCMT) (4) but exist in exact theories as well (5–7), lead to interesting physical effects such as enhanced sensitivity as a parameter is swept through an EP (8–11), nontrivial topological phenomena (12–16), and excess spontaneous emission noise (6, 17). Moreover, because the onset of laser emission corresponds to a resonance at a real frequency, an EP for two resonances at or very near the real axis leads to unusual lasing behavior, including chiral laser emission (4, 18), laser linewidth broadening (17, 19), single-mode lasing (20, 21), lasing and/or perfect absorption switching (22), etc. Except for the case of lasing, the resonant EPs involved are described by a complex frequency, but they are observed by their effects under quasi-steady-state (real frequency) excitation (23, 24). Because the onset of lasing leads to a linear instability that is stabilized by nonlinear saturation, it is difficult to study the purely linear behavior near such a lasing EP experimentally.

A different class of discrete complex frequency wave solutions arise from purely incoming boundary conditions, which are referred to as scattering zeros. It has been shown that the analog of adding gain to achieve lasing was to add absorption to the same structure to make the incoming solution occur at real frequency (25). Such a structure is known as a coherent perfect absorber (CPA), the time-reversed version of a laser at threshold, and the input wave front must be tuned to be the incoming version of the corresponding lasing mode to be fully absorbed (26, 27). Unlike lasing and resonance, there is no intrinsic instability or divergence associated with a CPA. Following the analogy to resonant EPs, it was recently proposed (28) that the eigenfrequencies of two incoming solutions of a wave equation could become degenerate, leading to an absorbing EP or EP zero (generally complex) or to a CPA EP when this occurs at a real frequency. A number of interesting phenomena distinct from resonant EPs were predicted for such absorbing EPs.

Here, we studied absorbing EPs in a system of coupled optical microcavities associated with perfect absorption, which has no instability at real frequency and presents interesting EP-related behavior not previously observed. This was achieved by distinguishing the roles of absorption loss and nondissipative scattering and coupling loss. Although increasing both types of loss causes resonances to move away from the real frequency axis, scattering loss does this for the zeros and absorption loss does the opposite.

Following (28), we consider a model of coupled standing-wave slab cavities with input and output channels on both sides (Fig. 1A). The reflection $(r_{1,2})$ and transmission $(t_{1,2})$ for such a two-channel scattering system lead to a $2\times 2S$ -matrix $S = \left( \begin{array}{cc}r_1 & t_2\\ t_1 & r_2 \end{array} \right)$ . We constructed an equivalent model with a different geometry through coupled whispering gallery mode (WGM) microcavities, $\mu R_{1,2}$ , with resonant frequencies $\omega_{1,2}$ , respectively (Fig. 1B). This was realized experimentally by coupling two silica microtoroid resonators fabricated on the edges of two silicon chips. The intercavity coupling strength ( $\kappa$ ) was tuned by adjusting the gap between the microtoroids, which determines the overlapping evanescent fields of WGMs (fig. S1). The two cavities are coupled to two single-mode fiber taper waveguides with the coupling strengths $\gamma_{c1}$ and $\gamma_{c2}$ , respectively. The intrinsic losses $\gamma_{1,2}$ are dominated by the absorption loss from the material and surface impurities, whereas the radiation and scattering loss was negligible in comparison. Transmission here is defined as the ratio of light passing through the coupled resonators (1→4, 3→2) and reflection as remaining in the same waveguide as the input wave (1→2, 3→4). Because of the negligible backscattering of this structure, only the clockwise mode in $\mu R_{1}$ and the counterclockwise mode in $\mu R_{2}$ are excited as long as only ports 1 and 3 are used for input. In the experiment, at most two pairs of resonances and zeros will be probed at a time. In this case, the system can be described by TCMT with an effective Hamiltonian (see supplementary text S1)

$$
H _ {\text { eff }} = \left( \begin{array}{c c} \omega_ {1} - i \frac {\gamma_ {1} + \gamma_ {\mathrm{c} 1}}{2} & \kappa \\ \kappa & \omega_ {2} - i \frac {\gamma_ {2} + \gamma_ {\mathrm{c} 2}}{2} \end{array} \right) (1)
$$

The corresponding $2 \times 2$ S-matrix in the frequency domain is given by

$$
S (\omega) = \left( \begin{array}{c c} 1 - i \frac {\gamma_ {\mathrm{cl}} \Delta_ {2}}{\Delta_ {1} \Delta_ {2} - \kappa^ {2}} & - i \frac {\sqrt {\gamma_ {\mathrm{cl}} \gamma_ {\mathrm{c2}}} \kappa}{\Delta_ {1} \Delta_ {2} - \kappa^ {2}} \\ - i \frac {\sqrt {\gamma_ {\mathrm{cl}} \gamma_ {\mathrm{c2}}} \kappa}{\Delta_ {1} \Delta_ {2} - \kappa^ {2}} & 1 - i \frac {\gamma_ {\mathrm{c2}} \Delta_ {1}}{\Delta_ {1} \Delta_ {2} - \kappa^ {2}} \end{array} \right)\tag{2}
$$

where $\Delta_{1,2} = \omega - \omega_{1,2} + i\frac{\gamma_{1,2} + \gamma_{c1,2}}{2}$ . S is a symmetric matrix due to the reciprocity of light transport in such structures. To simplify our analysis, we choose $\omega_{1} = \omega_{2} = \omega_{0}$ in the following discussion.

Generically, at any complex frequency $\omega$ , two complex eigenvalues $\lambda_{1,2}$ can be found for S and the corresponding two-by-one eigenvectors $v_{1,2}$ (satisfying $Sv_{1,2} = \lambda_{1,2}v_{1,2}$ ) define the two eigenchannels, representing specific superpositions of physical channels for which the waveform remains invariant during the scattering processes. Generically, at a resonance of S, one eigenvalue of S diverges, and at a zero of S, one eigenvalue vanishes; in each case, the other eigenvalue is finite. However, at a resonance, each element of the S-matrix diverges, which, as noted, makes experimental study complicated. In Fig. 1, C to G, we illustrate the different kinds of EPs that can arise through parameter tuning in this model. In lossless (no absorption) cavities (Fig. 1C), both the poles and zeros are simultaneously degenerate at an EP with the coupling $\kappa = \left|\frac{\gamma_{c1} - \gamma_{c2}}{4}\right|$ . Once absorption is added, the pole and absorbing EPs. (A and B) Phase transition diagrams for the real parts (A) and imaginary parts (B) of the poles and zeros as a function of normalized coupling strength $\kappa/\kappa_{\mathrm{th}}$ . "Exp"/"Theory" refer to experimental/theoretical results. The second waveguide is decoupled from $\mu R_2$ ( $\gamma_{c2} = 0$ ). The parameters $\gamma_1, \gamma_2, \gamma_{c1}$ , and $\kappa$ were extracted by curve fitting the data to $|r_1|^2$ (see eq. S41 in supplementary text S6), and then used to derive the complex poles (see eq. S10 in supplementary text S2) and zeros (see eq. S17 in supplementary text S3). (C) Normalized output spectrum near an absorbing EP. The blue and red curves are experimental and curve fitting results, respectively. (D) Simulation results of the energy distribution in $\mu R_1(E_1)$ and $\mu R_2(E_2)$ and the total dissipation rate as a function of the frequency detuning at the absorbing EP. Input power: $1\mu W$ .

C Lossless EP
D Absorbing EP
![](images/3933d6564cb8d9507fc608eb62d47045708f8bf9755eec9ceb6f8f2b2e1fe6ca.jpg)

![](images/62c0040338de300aa58d62b687fb9b9739def28bd843608c632eeb1b54546f1a.jpg)

![](images/fd3e34f16538659dec984a5e30d218e8c7af9bd8e55c93960ec429231dd48e8e.jpg)
B
E Resonant EP F Generic CPA EP G Nongeneric CPA EP

![](images/1580692e1f7dcf53a284fa5887d601007f837478aabfdfb030b722d22625e18f.jpg)
Fig. 1. Scattering properties of the coupled microcavity systems. (A) Scattering from two one-dimensional cavities formed by external Bragg mirrors coupled through a central, partially reflecting Bragg mirror, studied in (28). The scattered outputs from the incident waves in the left and right channels determine the S-matrix. (B) Schematic diagram of the analogous model to (A) realized with

![](images/543f87c581d970c17bbaafed9c3a00405ad61ad5dedf3d3916fafd124bd878dc.jpg)

![](images/a1399e28df8a68222752e85d6b63cc86b0b0409c3dab73ef624dde4668a7cdc3.jpg)

![](images/b692f00efc940c04c8fec03a342c2564b0ce7be9dbfcdec669c9f25c6a688eb8.jpg)
coupled WGM microcavities ( $\mu R_{1}$ and $\mu R_{2}$ ), with double fiber taper waveguides as the input and output channels. (C to G) Zeros (circles) and poles (triangles) of the system's S-matrix in the complex frequency plane for different types and configurations of EPs. (C) EP in a system without absorption or gain. (D) Absorbing EP. (E) Resonant EP. (F) Generic CPA EP. (G) Nongeneric CPA EP.

![](images/697b2b0629ab39578b7921a10f30a5cdcf3f9d94b3413ee75e4b04b0b5d03931.jpg)

![](images/eba8c5fd31c47176b806ccc35a3558d554560ab3ebb1a99ecfeed66b68654558.jpg)

and zero frequencies are not complex conjugates; one can realize a resonant EP when $\kappa = \left|\frac{(\gamma_{\mathrm{cl}} - \gamma_{\mathrm{c2}}) + (\gamma_1 - \gamma_2)}{4}\right|$ and an EP zero when $\kappa = \left|\frac{(\gamma_{\mathrm{cl}} - \gamma_{\mathrm{c2}}) - (\gamma_1 - \gamma_2)}{4}\right|$ (supplementary text S2 and

![](images/44c7f22e810819a90231c3e5ab0f7e309851d262fea890597ad45157b50663d1.jpg)

![](images/e0257548c0e65751dbffe398f331c40fb884116981c06402e703ba66be616789.jpg)

S3). This implies that as the coupling varies, the EP zero will occur before the resonant EP if the differential coupling has the same sign as the differential absorption between the two cavities (Fig. 1, D and E) and vice versa if they have opposite signs. The case of special interest is when the two zeros meet on the real axis, corresponding to steady-state perfect absorption at an EP, i.e., a CPA EP (Fig. 1F). This happens, within TCMT, at the critical coupling condition, when the total intrinsic loss $\gamma_{1} + \gamma_{2}$ is balanced by the total radiative coupling loss $\gamma_{c1} + \gamma_{c2}$ , with $\kappa = \left| \frac{\gamma_{c1} - \gamma_{1}}{2} \right|$ (supplementary text S4). In general, this is referred to as a generic CPA EP, which leads to an anomalous quartic absorption line shape (28). However, if the coupling rates are equal ( $\gamma_{c1} = \gamma_{c2}$ ), then both the zeros and poles coalescence at the same value of $\kappa$ within the TCMT model and we realize simultaneous resonant and absorbing EPs (Fig. 1G and supplementary text S5); this is referred to as a nongeneric CPA EP. The resulting singular behavior of the S-matrix leads to a distinct scattering behavior from the generic case to be discussed and demonstrated experimentally below.

![](images/91efb6d3325059ccba6b6883eb62dd227bdbd6c7c3c6bb0228fcb924d8c891e9.jpg)

![](images/0ffa73ce52ca6c1f52c7b1703142200d3c892c8123c01b2b31417af4f51dfce6.jpg)
Fig. 3. CPA EPs. (A) Schematic diagram of the experimental setup for a one-channel CPA EP. The S-matrix is reduced to one element, $S_{1\times1}$ . (B) Experimentally obtained normalized output spectrum (blue) at the one-channel CPA EP and the curve-fitting results using quartic (red dashed) and Lorentzian (black dotted) functions. The inset displays a double-logarithmic plot of the output versus the frequency detuning. (C) Schematic diagram of the experimental setup for a two-channel CPA EP. The magnitudes of the two inputs are controlled by two variable optical attenuators (OA1 and OA2).

![](images/a7e22eb1d692967ad2e9af1d77be53faf4d667810541140caf7143e06bbf03ec.jpg)

![](images/e9ef015835d3378e8c1c3c706c58d611db3bdea09951358acc56ce900280f2cf.jpg)
The relative phase of the two inputs is controlled by an electro-optic phase modulator (EOM). The lengths of the two optical paths are balanced by a variable optical delay line (VDL). (D) Experimentally obtained spectra of the output from port 2 (blue curve), the output from port 4 (red curve), and the total output (black curve) normalized to the total input power at the nongeneric CPA EP. To excite the system by the eigenvector of S, the amplitudes and relative phase of the two input fields are adjusted by OA1, OA2, and EOM.

Here, we experimentally realize and compare the cases illustrated in Fig. 1, D to G, for the resonant and absorbing EPs in the setup shown in Fig. 1B. Initially, for simplicity, we study the case with the bottom waveguide decoupled from $\mu R_{2}$ , which implies, in the $S$ -matrix terminology, that only reflection or absorption of the input is possible. The resonant frequencies are aligned by the temperature control of $\mu R_{2}$ through a thermoelectric cooler. The reflection spectrum is measured to derive the complex poles and the zeros. The resonant EP and absorbing EP are distinguished by the phase transitions of the complex poles and zeros when the intercavity coupling strength $\kappa$ is varied (Fig. 2, A and B). For the parameter scenario here, the zeros meet at a lower coupling, $\kappa = \kappa_{th}$ , and then move apart; the coupling is increased further and then the poles meet at $\kappa \approx 1.32\kappa_{th}$ . For each case, after the transition, the zeros and poles no longer have the same real part of the frequency but instead share approximately the same imaginary part. This clearly confirms that resonant EPs and absorbing EPs generally occur at different coupling strengths (or equivalently, different differential loss rates). In the alternative scenario, the absorbing EP occurs after the appearance of resonant EP as $\kappa$ increases (fig. S2).

It is worth noting that the single dip or peak line shape of the output signal spectrum is often regarded as a signature of resonant EPs. Particularly, in a single cavity with coupled clockwise and counterclockwise modes (8) or parity-time (PT) symmetric coupled microcavities (9), the transmission spectrum evolves from a doublet to a single dip at EPs. This is correct for systems without absorption (or gain); however, for general non-Hermitian systems, EPs may not be accompanied by a singlet in the transmission and reflection spectra. This was confirmed experimentally by the spectra of the scattered signal at the absorbing EP (Fig. 2C) and the resonant EP (fig. S3), which both exhibit an absorbing doublet. This is attributed to the fact that the zeros and poles are not complex conjugate pairs and do not coalesce simultaneously, so that the minimum output is not reached at the frequencies $Re(\omega_{z1,2})$ or $Re(\omega_{p1,2})$ (see supplementary text S7). This observation can also be explained by the simulated field redistribution versus the frequency detuning ( $\delta$ ) at the absorbing EP (Fig. 2D). The total dissipation reaches its maximum at nonzero detuning, where the laser frequency is not equal to $Re(\omega_{z1,2})$ or $Re(\omega_{p1,2})$ .

We next study CPA EPs, where the two degenerate zeros occur at a real frequency. With the scheme in Fig. 3A, we realize a one-channel generic CPA EP (equivalent to critical coupling at an EP) by tuning the cavity-waveguide coupling so that $\gamma_{\mathrm{c1}} = 119.2814\mathrm{MHz}\approx \gamma_{1} + \gamma_{2}$ and adjusting the intercavity gap to achieve an absorbing EP ( $\kappa = \left| \frac{\gamma_{c1} + \gamma_2 - \gamma_1}{4} \right|$ ). As predicted, the measured spectrum is well fit by a quartic function, which verifies the anomalous broadband absorption line shape for CPA EPs (Fig. 3B and supplementary text S8). By comparison, the Lorentzian curve-fitting result based on the measured $\gamma_{c1}$ shows a large deviation from the data (Fig. 3B). The quartic relation is further confirmed by a slope of 4 in the logarithmic plot (inset of Fig. 3B).

![](images/0bb94d6e1e29f76979ae375b599cdd88a9f62d1fabd9370cbb75dbdf29943f67.jpg)

![](images/a44b9eeb0d82d6f360136edd433177ca29ace20e8c638b39eb36c30a317acd1a.jpg)
Fig. 4. Scattering properties and phase modulation at the two-channel CPA EP. (A) Experimentally measured spectra of $R_{1} = |r_{1}|^{2}$ and $T_{1} = |t_{1}|^{2}$ by exciting port 1 only. (B) Experimentally measured spectra of $R_{2} = |r_{2}|^{2}$ and $T_{2} = |t_{2}|^{2}$ by exciting port 3 only. (C) Experimentally obtained output power from port 2 ( $P_{2}$ ) and port 4 ( $P_{4}$ ) at the zero detuning ( $\delta = \omega - \omega_0 = 0$ ) normalized to the total input power as a function of the relative phase. $P_{2}$ and $P_{4}$ are equal in TCMT. The total normalized output power ( $P_{\text{tot}}$ ) is obtained by adding $P_{2}$ and $P_{4}$ . (D) Modulation depth, the difference between the maximum and the minimum power normalized to the total input power under varying relative phase, of the output from port 2 ( $M_{2}$ ) and the output from port 4 ( $M_{4}$ ) versus frequency detuning. The modulation depth of the total output power ( $M_{\text{tot}}$ ) is not the sum of $M_{2}$ and $M_{4}$ because at each frequency detuning, the relative phases that maximize (or minimize) the two outputs can be different.

To study two-channel CPA EPs, we restored the coupling to the second waveguide (Fig. 3C). A nongeneric CPA EP can be more easily found because it occurs simultaneously with a resonant EP (supplementary text S5). Thus, in the experiment, we set $\gamma_{\mathrm{c1}} = \gamma_{\mathrm{c2}} = (\gamma_1 + \gamma_2)/2$ , and then reposition $\mu R_2$ in parallel to the waveguide to find the resonant EP. We confirmed experimentally the prediction of a universal form of $S$ at a nongeneric CPA EP, with identical amplitudes in the four elements (Fig. 4, A and B), consistent with the theory (fig. S4) (28). $|r_1|^2$ and $|t_1|^2$ are found to be almost equal at $\delta = 0$ , whereas $|r_2|^2$ and $|t_2|^2$ approach each other with a small discrepancy because of imperfections such as backscattering effects in the cavity modes. Furthermore, the perfect absorption of the input wave is achieved by coupling the input into the absorbing eigenchannel defined by the eigenstate of $S([1,-i]^{T})$ . With balanced input power and the optimized relative phase (29), strong absorption is observed experimentally (Fig. 3D; for simulation results, see fig. S5); the nonideal perfect absorption is caused by the fluctuation of optical phases and coupling parameters. Moreover, the total absorption line shape for the eigenchannel input is quadratic instead of quartic because the eigenvalues of S are not analytic around the zero detuning (see supplementary text S8).

D
![](images/822964c3a97d98455f4fb83dfb52570feb3081a9d15ce1cde3340568250e6fa2.jpg)

![](images/3823b2d9dd4a8116c2d3ff4910a71cd6b536776dbed9ed93e5946d9458a36491.jpg)

The degree of absorption in the case of two-channel CPA is sensitive to the relative phase of the inputs and should be maximal when the phase difference corresponds to the relevant eigenchannel of S. To characterize the phase response of our system operating at a CPA EP, we measure the output power from port 2 and port 4 around zero detuning normalized to the total input power ( $P_{2}$ and $P_{4}$ ) as a function of the relative phase $\Delta\phi$ (figs. S6 and S7) (29). The two output signals oscillate in phase with $\Delta\phi$ , consistent with the theoretical prediction, and add up to the total output $(P_{\mathrm{tot}})$ with a large oscillation amplitude (Fig. 4C). Furthermore, the modulation depth, defined as the difference between the maximum and minimum output power with varying $\Delta\phi$ normalized to the total input power, is measured versus the detuning. It was observed that the modulation depth of the output from port 2 $(M_{2})$ behaved differently from that from port 4 $(M_{4})$ (Fig. 4D). This is because the input vectors that maximize and minimize the output power at a nonzero detuning are not always the eigenstate of S, and thus the feature of the nongeneric CPA EP is hidden. Furthermore, the spectrum of the total output modulation depth $(P_{\mathrm{tot}})$ displayed a flattened window with $P_{tot} > 0.8$ in a frequency range above 500 MHz (comparable with $\gamma_{c}$ ), which indicates a broadband large phase sensitivity at the CPA EP.

Our study suggests the importance of properly choosing the input to reveal the unique scattering properties of EPs. This is critical for non-Hermitian applications such as EP-enhanced sensing, where the amplified spectral splitting can only be observed with a proper probing scheme. Moreover, the absorbing EPs can be potentially found in various non-Hermitian systems including metamaterials and metasurfaces, acoustic systems, electrical circuits, and, more interestingly, open quantum systems. It has been pointed out that realizing PT-symmetric quantum optics is nontrivial because optical gain will unavoidably introduce additional noise that breaks the PT symmetry (30). This hinders the exploration of resonant EPs through emission features in quantum systems. However, by engineering the quantum dissipation, the absorbing EPs can be realized without the need for gain and therefore can be potentially exploited to engineer scattering properties of nonclassical light.

## REFERENCES AND NOTES

1. L. Feng, R. El-Ganainy, L. Ge, Nat. Photonics 11, 752–762 (2017).

2. R. El-Ganainy et al., Nat. Phys. 14, 11–19 (2018).

3. M.-A. Miri, A. Alù, Science 363, eaar7709 (2019).

4. B. Peng et al., Proc. Natl. Acad. Sci. U.S.A. 113, 6845–6850 (2016).

5. Y. D. Chong, L. Ge, A. D. Stone, Phys. Rev. Lett. 106, 093902 (2011).

6. A. Pick et al., Opt. Express 25, 12325–12348 (2017).

7. R. Fleury, D. Sounas, A. Alù, Nat. Commun. 6, 5905 (2015).

8. W. Chen, Ş. Kaya Özdemir, G. Zhao, J. Wiersig, L. Yang, Nature 548, 192–196 (2017).

9. H. Hodaei et al., Nature 548, 187–191 (2017).

10. Y.-H. Lai, Y.-K. Lu, M.-G. Suh, Z. Yuan, K. Vahala, Nature 576, 65–69 (2019).

11. M. P. Hokmabadi, A. Schumer, D. N. Christodoulides, M. Khajavikhan, Nature 576, 70–74 (2019).

12. H. Xu, D. Mason, L. Jiang, J. G. E. Harris, Nature 537, 80–83 (2016).

13. J. Doppler et al., Nature 537, 76–79 (2016).

14. J. W. Yoon et al., Nature 562, 86–90 (2018).

15. W. Tang et al., Science 370, 1077–1080 (2020).

16. H. Zhou et al., Science 359, 1009–1012 (2018).

17. H. Wang, Y. H. Lai, Z. Yuan, M. G. Suh, K. Vahala, Nat. Commun. 11, 1610 (2020).

18. P. Miao et al., Science 353, 464–467 (2016).

19. J. Zhang et al., Nat. Photonics 12, 479–484 (2018).

20. L. Feng, Z. J. Wong, R. M. Ma, Y. Wang, X. Zhang, Science 346, 972–975 (2014).

21. H. Hodaei, M. A. Miri, M. Heinrich, D. N. Christodoulides, M. Khajavikhan, Science 346, 975–978 (2014).

22. Z. J. Wong et al., Nat. Photonics 10, 796–801 (2016).

23. C. Shi et al., Nat. Commun. 7, 11110 (2016).

24. C. Wang et al., Nat. Phys. 16, 334–340 (2020).

25. Y. D. Chong, L. Ge, H. Cao, A. D. Stone, Phys. Rev. Lett. 105, 053901 (2010).

26. D. G. Baranov, A. Krasnok, T. Shegai, A. Alù, Y. Chong, Nat. Rev. Mater. 2, 17064 (2017).

27. W. Wan et al., Science 331, 889–892 (2011).

28. W. R. Sweeney, C. W. Hsu, S. Rotter, A. D. Stone, Phys. Rev. Lett. 122, 093901 (2019).

29. Materials and methods are available as supplementary materials.
30. S. Scheel, A. Szameit, Europhys. Lett. 122, 34001 (2018).

## ACKNOWLEDGMENTS

Funding: This work was supported by the National Science Foundation (NSF grant no. EFMA1641109). A.D.S. was supported by NSF Condensed Matter and Materials Theory (CMMT) grant DMR-1743235. C.W. acknowledges fellowship support through the McDonnell International Scholars Academy. Author contributions: L.Y. and A.D.S. conceived the joint project. C.W. and L.Y. designed the experiments with help from W.R.S. and A.D.S. C.W. performed the experiments and analyzed experimental data with help from W.R.S. C.W. and W.R.S. provided theoretical background and simulations. All authors discussed the results and wrote the manuscript. L.Y. supervised the project.

Competing interests: The authors declare no competing interests. Data and materials availability: All data are available in the main text or the supplementary materials.

## SUPPLEMENTARY MATERIALS

https://science.org/doi/10.1126/science.abj1028
Materials and Methods
Supplementary Text
Figs. S1 to S7
References (31–73)

25 April 2021; accepted 11 August 2021

10.1126/science.abj1028
