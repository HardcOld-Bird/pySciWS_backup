# Exceptional Bound States in the Continuum

Adrià Canós Valero, $^{1,*}$ Zoltan Sztranyovszky $^{\text{id}}$ , $^{2}$ Egor A. Muljarov, $^{3}$ Andrey Bogdanov $^{\text{id}}$ , $^{4,5}$ and Thomas Weiss $^{\text{id}}$ $^{1,6,\dagger}$

$^{1}$ Institute of Physics, University of Graz, and NAWI Graz, 8010 Graz, Austria

$^{2}$ School of Chemical Engineering, University of Birmingham, Birmingham B15 2TT, United Kingdom

$^{3}$ School of Physics and Astronomy, Cardiff University, Cardiff CF24 3AA, United Kingdom

$^{4}$ Qingdao Innovation and Development Center of Harbin Engineering University, Sansha Road 1777, Qingdao 266404, China

$^{5}$ School of Physics and Engineering, ITMO University, Kronverkskiy Street 49, 197101, St. Petersburg, Russia

$^{6}$ Physics Institute and SCoPE, University of Stuttgart, 70569 Stuttgart, Germany

(Received 15 March 2024; accepted 21 January 2025; published 14 March 2025)

Bound states in the continuum and exceptional points are unique singularities of non-Hermitian systems. In optical implementations, the former demonstrate strong enhancement of the electromagnetic field, while the latter exhibit high sensitivity to small perturbations. Hence, exceptional points are being actively investigated as next-generation optical sensors. However, at the nanoscale, their performance is strongly constrained by parasitic radiative losses. Here, we show that several bound states in the continuum can be merged into one exceptional point, forming a new kind of singularity. The resulting state inherits properties from both, namely, it does not radiate and shows extremely high sensitivity to perturbations, making it prospective for the realization of exceptional sensing at the nanoscale. We validate our theory with numerical simulations and demonstrate the formation of second- and third-order exceptional bound states in the continuum in stacked dielectric metasurfaces.

DOI: 10.1103/PhysRevLett.134.103802

Open systems inherently imply interaction with the surrounding space via the exchange of energy. They can be found in both classical and quantum systems, involving acoustical or optical waves $[1-3]$ . Formally, open systems can be described with non-Hermitian Hamiltonians $[4-7]$ . In the past few years, this general framework has revealed new physical phenomena and functionalities that are currently under intense exploration, for instance, coherent perfect absorption $[8]$ , unidirectional energy transport $[9]$ , single-mode lasing $[10]$ , static nonreciprocity $[11]$ , or superscattering $[12]$ .

A remarkable consequence of non-Hermiticity is the emergence of “exceptional points” (EPs) in the eigenenergy spectra. EPs are spectral singularities where at least two eigenvectors and their associated eigenvalues coalesce $[13]$ . In close proximity to an EP, the eigenvalues exhibit a strongly enhanced sensitivity to perturbations, making them prospective for sensing applications $[14,15]$ , only limited by noise $[16]$ . The coalescence of several eigenvectors results in the formation of higher-order EPs showing even higher sensitivity to perturbations $[17]$ . In optics,

EPs have been primarily investigated in PT-symmetric systems $[13]$ . However, they can also emerge in more practical passive structures, such as plasmonic metasurfaces $[18–22]$ , microresonators $[23]$ , and waveguides $[15,24]$ , or even single nanoparticles $[25]$ . Unfortunately, in order to realize exceptional point sensing with passive resonators, the enhanced eigenvalue splitting must exceed the resonance linewidths to resolve it spectrally. In particular, at the nanoscale, radiation losses play a dominant role in the linewidths, largely precluding the sensitivity enhancements $[18,22,25–27]$ .

Another fascinating type of singularities of open systems are “bound states in the continuum” (BICs) $[28–30]$ . BICs were first predicted in quantum mechanics $[31,32]$ , but today they are actively studied in photonics, acoustics, and hydrodynamics $[28,33,34]$ . They are nonradiating resonances, despite being embedded in the continuum of radiating states. Therefore, their radiative Q factor is infinite, while the total Q factor can be finite due to material losses $[35,36]$ .

The radiation cancellation can arise due to symmetry or by tuning the parameters of the system such that radiation from all open channels is suppressed. As a result, BICs are classified as symmetry protected (S-BICs) or accidental, respectively $[28,37]$ . In photonics, S-BICs have been mainly studied in periodic metasurfaces. Symmetry breaking turns an S-BIC into a quasi-BIC with a finite radiative Q factor $[38]$ , resulting in giant amplification of the incident radiation. Today, BICs present exciting opportunities for the development of compact high-Q platforms for biosensing, polaritonics, and nonlinear nanophotonics [39-41].

Given the unusual properties of both kinds of singularities, an intriguing question is whether two or more BICs can coalesce into an EP, giving rise to a new singularity. Besides their fundamental interest, such “EP-BICs” would not suffer from any radiation losses and may inherit the sensitivity of EPs, which could make them ideal candidates for the realization of EP sensors at the nanoscale. Surprisingly, this problem has not been addressed, despite several reports of systems displaying BICs and EPs in the same parameter space $[27,42–46]$ .

In this study, we find a general recipe for the coalescence of several BICs into an EP of arbitrary order, forming an EP-BIC. After establishing a coupling theory valid for eigenmodes with small radiation losses, we verify our results numerically by designing a bilayer and a trilayer dielectric metasurface operating in the visible range, realizing second- and third-order EP-BICs. Remarkably, the novel states retain the nonradiating behavior of BICs, while simultaneously exhibiting the square and cubic root dispersion of EPs. Furthermore, under small symmetry-breaking perturbations, the losses of the resulting quasi-BICs no longer follow the conventional asymptotics for the Q factor vs the asymmetry parameter $[38,47,48]$ .

We gain initial insight into the formation of an EP-BIC through an effective Hamiltonian formalism, describing the behavior of the eigenmodes of the open system, also referred to as resonant states or quasinormal modes $[49–51]$ . BICs and EPs can only coexist in the same parameter space if at least two modes are present. Therefore, we write the simplest possible model effective Hamiltonian $[52,53]$ , representing two coupled, radiative resonators (sketched in Fig. 1) as follows:

$$
\hat {H} = \left( \begin{array}{c c} \tilde {\omega} _ {1} & \tilde {\kappa} \\ \tilde {\kappa} & \tilde {\omega} _ {2} \end{array} \right).\tag{1}
$$

$\hat{H}$ features complex eigenfrequencies of the form $\tilde{\omega}_{1,2} = \omega_{1,2} - i(\gamma_{1,2}^{\mathrm{r}} + \gamma_{1,2}^{\mathrm{int}})$ , and coupling coefficients $\tilde{\kappa} = \kappa - i\sqrt{\gamma_1^{\mathrm{r}}\gamma_2^{\mathrm{r}}}$ . $\omega_{1,2}, \gamma_{1,2}^{\mathrm{r}}$ , and $\gamma_{1,2}^{\mathrm{int}}$ are, respectively, the resonance frequencies, radiation, and intrinsic losses of the uncoupled modes, while $\kappa$ is the near field coupling coefficient, here assumed to be real. If the mode radiates, the total losses are, approximately, half the linewidth of its resonant peak in the optical response. Equation (1) does not specify the physical nature of the system, which can be photonic, or acoustic. The eigenfrequencies $\tilde{\omega}_{\pm}$ are solutions to the following dispersion relation:

$$
\tilde {\omega} _ {\pm} = \frac {\tilde {\omega} _ {1} + \tilde {\omega} _ {2}}{2} \pm \Lambda ,\tag{2}
$$

where $\Lambda = \sqrt{(\tilde{\omega}_1 - \tilde{\omega}_2)^2 / 4 + \tilde{\kappa}^2}$ . First, let us consider $\gamma_{1,2}^{\mathrm{int}} = 0$ . Then, EPs as well as the two known BIC types, accidental and S-BIC, can arise in this system. When $\Lambda = 0$ , Eq. (2) shows that the modes coalesce at an EP at the eigenfrequency $\tilde{\omega}_{\mathrm{EP}} = (\tilde{\omega}_{1} + \tilde{\omega}_{2})/2$ . Similarly, an accidental BIC may occur when $\kappa(\gamma_{1}^{\mathrm{r}} - \gamma_{2}^{\mathrm{r}}) = \sqrt{\gamma_{1}^{\mathrm{r}}\gamma_{2}^{\mathrm{r}}}(\omega_{1} - \omega_{2})$ [28]. Finally, S-BICs can be found trivially when $\gamma_{1}^{r} = \gamma_{2}^{r} = 0$ , decoupling both modes from radiation. In the third case, $\hat{H}$ is Hermitian and the eigenspectrum hosts no EPs. This last scenario is illustrated in Fig. 1(a), depicting two identical resonators, each supporting a hypothetical S-BIC. Figure 1(a) shows the resonance frequencies of the two BICs calculated as a function of $\kappa$ . When $\kappa = 0$ (uncoupled resonators), the two modes form a diabolic point [54,55]. With increasing $\kappa$ , the resonance frequencies split, with a gap proportional to $|\kappa|$ . This is the conventional behavior of S-BICs, recently experimentally reported in Ref. [54].

![](images/60126d1ddf18dabe124ae2ab875db8f8e1049baa30bd0555c404174e87d2beac.jpg)
FIG. 1. Exceptional bound states in the continuum. (a) Upper panel: illustration of two BICs without intrinsic losses, coupled in the near field by $\kappa$ . The BICs have no radiation losses $\gamma^{r}$ (infinite radiative Q factor), as their coupling with the radiation continuum is zero (crossed-out arrows in the sketch). They have identical eigenfrequencies $\omega_{1} = \omega_{2}$ when $\kappa = 0$ . Lower panel: evolution of the resonance frequencies of the BICs vs $\kappa$ . (b) Same as (a), when BIC 2 has $\gamma^{int} \neq 0$ . When $\kappa = \gamma^{int}/2$ , this results in the formation of an EP-BIC with infinite radiative Q factor.

Despite this, the BIC and EP conditions cannot be simultaneously satisfied. To understand why, consider the trace of $\hat{H}$ , which must remain invariant under any basis change, and reads $\mathrm{tr}(\hat{H}) = \tilde{\omega}_{1} + \tilde{\omega}_{2}$ . At an EP, both eigenmodes must have the same eigenfrequency A, so that $\mathrm{tr}(\hat{H}) = \tilde{\omega}_{1} + \tilde{\omega}_{2} = 2A$ . Since BICs cannot radiate, A must be real. The only solution is $\gamma_{1}^{r} = \gamma_{2}^{r} = 0$ , i.e., two S-BICs at a diabolic point. While this argument is insufficient for a larger number of modes, we show [56] that, regardless of the number of modes, BICs (S-BICs or accidental) in a purely radiative system can only form diabolic points.

The restriction can be lifted by introducing intrinsic losses, $\gamma^{int}$ , into one of the S-BICs, as illustrated in Fig. 1(b). Now, one of the S-BICs has a nonzero imaginary part (yet still remains nonradiative). Setting $\Lambda = 0$ leads to the EP conditions $2\kappa = \pm\gamma^{int}$ . Figure 1(b) shows the resonance frequencies vs $\kappa$ for this case, with $\kappa > 0$ . Instead of a linear dependence in the strong coupling regime, the BICs collapse at an EP-BIC at the onset from weak to strong coupling.

(a)
![](images/13db2e9c5fed34f1640599a5e4fe0fdd1e0e6546599b3df5780dbf1a559778dc.jpg)

(d)
![](images/780ff4aea9893fafd2e7b4bdd6cc36bbebca9e3065d95362f9cd8e2ff3380088.jpg)

![](images/e03e02b9001dc850e7f7dab0a2d8fd5b2f552819c78e72b8364a7d399660744a.jpg)

(b)
![](images/f19bf38989b26766b017be05d799fe0f5bacbc253f45745f39423fddd4727641.jpg)

(c)
![](images/b42b73d0ec9c0f53b135e02022a6e9dce01f17e8849a4ca8fb1816ece5111a02.jpg)

![](images/9f83aee443dc1a0a048eaa5be9a02da1e0c77778019e0ea24bedfc6136a96877.jpg)

![](images/395879c350a567be500bb9c1a49424bf5d11ca80f1f86abfb7988d61dbc7d528.jpg)

(e)
![](images/31fccb328f6764446eb8090b19a26751846c15b8de0a8b006ae2571365c9b1f8.jpg)
FIG. 2. EP-BIC formation in a bilayer dielectric metasurface. (a) Left: design schematic, consisting of two stacked metasurfaces of dielectric disks with radius 150 nm, height 50 nm, and period 400 nm. The refractive index of the top and bottom metasurfaces are, respectively, $n_{t} = 4 + ik$ and $n_{b} = 4$ . Right: symmetric and antisymmetric BICs formed by the hybridization of the MD BICs of each metasurface. (b) Comparison between the numerical and analytical results for the resonance frequencies vs separation, with k = 0. (c) Same as (b), but with k = 0.07. (d) Azimuthal component of the electric field for the two BICs at points A, B, A', and B' indicated in (b) and (c). The red crosses emphasize the absence of propagating waves out of the plane of the structure in all cases. (e) Intrinsic losses vs separation, for the case in (c). See [56] for simulation details.

Next, we demonstrate an EP-BIC in a real system. The structure we investigate consists of two stacked high-index dielectric metasurfaces in vacuum [see Fig. 2(a)], with identical dimensions. Each isolated metasurface is designed to support an S-BIC at normal incidence, at the resonance frequency $\omega_0$ . This particular S-BIC can be pictured as an array of out-of-plane magnetic dipole moments. When placed close to each other, the BICs hybridize, forming a pair of symmetric and antisymmetric modes, depicted in Fig. 2(a). The top metasurface is made lossy, introducing a small imaginary part $\gamma^{\mathrm{int}}$ to the corresponding BIC. Hence, the BIC is still nonradiative, but has a finite $Q$ factor $Q = \omega_0 / 2\gamma^{\mathrm{int}}$ . To obtain the Hamiltonian $\hat{H}_M$ , we derived a coupling mode theory using the BICs of the individual metasurfaces as a basis [56,61]. When $\gamma^{\mathrm{int}} \ll \omega_0$ and assuming small interaction, $\hat{H}_{\mathrm{M}}$ simplifies to the following [56]:

$$
\hat {H} _ {\mathrm{M}} \approx \left( \begin{array}{c c} \omega_ {0} - i \gamma^ {\mathrm{int}} & - g \omega_ {0} \\ - g \omega_ {0} & \omega_ {0} \end{array} \right).\tag{3}
$$

Here, g can be retrieved analytically from the fields of the uncoupled BICs as $g = \int_{V_{\mathrm{t}}} \Delta \epsilon \tilde{\mathbf{E}}_{\mathrm{t}}(\mathbf{r}) \cdot \tilde{\mathbf{E}}_{\mathrm{b}}(\mathbf{r}) \, \mathrm{d}V$ [56], where the subscripts (t,b) denote the BICs from the top and bottom metasurface, respectively, and $\Delta\epsilon$ is the difference between the permittivity of the disks in the top metasurface and vacuum. The integral runs over the volume $V_{t}$ of the top disk. Equation (3) allows predicting the eigenfields and eigenfrequencies of the coupled BIC metasurfaces with the sole knowledge of the eigenfields in the bare constituents. This is confirmed by the good agreement between the numerical and analytical results in Fig. 2. Importantly, $\hat{H}_{M}$ takes the form of the toy model in Eq. (1), with $-g\omega_{0}$ playing the role of $\tilde{\kappa}$ .

Figure 2(b) shows the results without ohmic losses. With decreasing separation d between the metasurfaces, the overlap between the evanescent tails of the BICs is larger, and g increases. The resonance frequencies follow the same qualitative picture from the toy model in Fig. 1(a). Figures 2(c) and 2(e) display, respectively, the resonance frequencies and dissipation losses of the BICs when the refractive index of the top metasurface is made complex. The EP-BIC emerges when $d_{EP} \approx 236$ nm. Near the singularity, the resonance frequencies show a drastic difference with respect to the conventional scenario. For $d < d_{EP}$ , they exhibit the characteristic square-root dispersion of an EP. The same can be observed in the intrinsic losses for $d > d_{EP}$ . The resonance frequencies of the symmetric and antisymmetric BICs are quasidegenerate for $d > d_{EP}$ , a phenomenon known as a bulk Fermi arc [25,62]. Despite the presence of intrinsic losses, the eigenmodes are still

(a)
(b)
![](images/bf1a2e9b5c0e33ffca5fadf78a521e0eb73c25a59f97b44790bd556e435147e5.jpg)
(c)

![](images/7eff369df402a89f86fe050935db82bd1b53df7bd5cf6de3fd96292cc8b313d9.jpg)

![](images/cbfd7ec2ffc60383245f5521e30333face0c9a60b5ad25fa90828ddcb2c42912.jpg)

![](images/1e363196529778a7e4114360274c6caacbbef6159f82325d85a8a909ffa9e64d.jpg)
FIG. 3. (a) Total losses of a single (lossy) metasurface with same parameters as Fig. 2, when etching a hole of surface $S_{\mathrm{h}}$ , offset $40~\mathrm{nm}$ from the center of the disk, (see inset). $x$ axis shown in log scale. (b) Bilayer metasurface with $d = 250~\mathrm{nm} > d_{\mathrm{EP}}$ and $d = 236~\mathrm{nm} \approx d_{\mathrm{EP}}$ . Orange and blue lines correspond to the antisymmetric and symmetric BICs, respectively. (c) Radiative and intrinsic contributions to the total loss. The symmetric mode has been scaled up 10 times for better visualization. (d) Azimuthal component of the electric field for the two quasi-BICs at points A, B indicated in (b). Ranges of $S_{\mathrm{h}}$ are the same in (a)-(c). Refer to [56] for an investigation of the $Q$ factors.

BICs, since their field profiles are incompatible with the electromagnetic continuum. This can be visually confirmed from the absence of propagating waves in the field distributions of the BICs [Fig. 2(d)], which remain confined in the metasurfaces. In addition, owing to the collapse of the eigenspectra, near the EP-BIC the fields of the two modes become almost identical [panels A', B' in Fig. 2(d)] [63].

Since EP-BICs are nonradiative, they cannot be optically probed from the far field. We now study the effect of symmetry-breaking perturbations that introduce radiation losses into the system. As a result, the EP-BIC condition is violated, and the perturbed modes turn into radiative quasi-BICs. We demonstrate that the linewidths of the resonant peaks no longer follow the behavior attributed to S-BICs up to date.

As sketched in the inset of Fig. 3(a), we introduced small off-center periodic holes with cross sectional area $S_{\mathrm{h}}$ in the top metasurface, which breaks the in-plane symmetry and couples the S-BICs to normally incident plane waves. We first analyzed the behavior of the S-BIC in a single lossy metasurface [Fig. 3(a)]. As $S_{\mathrm{h}}$ increases, two regimes can be distinguished. Small holes lead to a weak linear growth of $\gamma$ , in agreement with first-order perturbation theory [64], where intrinsic losses dominate [47,48]. With larger $S_{\mathrm{h}}$ , the rapid growth of the total loss indicates that radiative and intrinsic losses become comparable [38]. This is in striking contrast with the behavior near the EP-BIC shown in Fig. 3(b), where the same range of $S_{\mathrm{h}}$ as in Fig. 3(a) is considered. Here, a square-root dependence with small $S_{\mathrm{h}}$ is observed. Interestingly, while the antisymmetric mode becomes more lossy, the symmetric mode decreases its loss, drastically increasing its total Q factor $[56]$ . These unusual characteristics can be observed in the reflection spectra for y polarized light $[56]$ . We also confirmed a similar qualitative behavior in reciprocal space, when the symmetry is broken by changing the Bloch wave vector $[56]$ .

Next, we show that the observed trends can be explained by taking into account the combined effects of a second-order EP and the BICs in the perturbation series. Consider the discriminant $\Lambda$ in Eq (2). At the EP-BIC, $\Lambda = 0$ . Assume now that we modify the losses $\delta\gamma$ of one of the BICs. This represents the effect of symmetry breaking. To leading order, $\Lambda \approx i\sqrt{\gamma^{\mathrm{int}}\delta\gamma/2}$ [56], confirming the square-root asymptotics near the EP. However, to first order the losses of the BIC on the top metasurface follow the law $\delta\gamma = 2aS_{h}$ , where a is a constant. Inserting this ansatz into the expansion of $\Lambda$ , and plugging the latter in Eq. (2), we arrive at the following dispersion relation near the EP-BIC:

$$
\tilde {\omega} _ {\pm} = \tilde {\omega} _ {\mathrm{EP}} - i a S _ {\mathrm{h}} \pm i (\gamma^ {\mathrm{int}}) ^ {1 / 2} a ^ {1 / 2} S _ {\mathrm{h}} ^ {1 / 2}.\tag{4}
$$

In Eq. (4), $S_{h}$ features a linear and a square-root contribution to $\gamma^{tot}$ . The square-root term is absent in conventional BICs, and appears due to the EP dispersion. The different signs explain the increase and decrease in the losses of the quasi-BICs for small $S_{h}$ . However, the analysis does not reveal if the new features are connected to absorption and/or radiation. These two contributions can always be separated in weakly dispersive media [51,56]. The intrinsic losses can be calculated by taking the ratio between half the absorbed power and the electromagnetic energy in the unit cell volume [56]. The radiation losses can be found as $\gamma^{r} = \gamma^{tot} - \gamma^{int}$ . Figure 3(c) displays the evolution of $\gamma^{int}$ and $\gamma^{r}$ for the two quasi-BICs at $d \approx d_{EP}$ . It can be confirmed that only $\gamma^{int}$ is responsible for the square-root trend, as well as the decrease in the losses of the symmetric mode. Conversely, the radiation losses grow quadratically in the antisymmetric quasi-BIC, but saturate very fast in the symmetric one. This is because increasing $S_{h}$ localizes the symmetric mode on the bottom metasurface, leading to a smaller intrinsic loss and a weaker $S_{h}$ dependence, as can be confirmed in the field distributions of the quasi-BICs [Fig. 3(d)].

The approach can be extended to EP-BICs of arbitrary order, provided that intrinsic losses are properly distributed among the resonators. A third-order EP-BIC can be realized by stacking three BIC metasurfaces, [see inset of Fig. 4(b)], obeying the Hamiltonian

$$
\hat {H} _ {\mathrm{M}} \approx \left( \begin{array}{c c c} \omega_ {0} - 2 i \gamma^ {\text {int}} & - g \omega_ {0} & 0 \\ - g \omega_ {0} & \omega_ {0} - i \gamma^ {\text {int}} & - g \omega_ {0} \\ 0 & - g \omega_ {0} & \omega_ {0} \end{array} \right).\tag{5}
$$

![](images/2edeea73e79d3521dd75e6e30e0b08777d965cccc6136d41c083760f7b67137e.jpg)
FIG. 4. Third-order EP-BICs (EP $_3$ -BIC). (a) Resonance frequencies and (b) intrinsic losses of three coupled S-BICs in a trilayer metasurface (unit cell in the inset). The refractive indices of the disks are $n = 4 + ik$ , ( $k_0 = 0.05$ ). (c) Changes in total losses for conventional S-BICs and EP-BICs of order $m = 2$ and 3. Holes are etched on the top (lossy) metasurface. Disks of height 150 nm, radius 150 nm. Period: 350 nm. Ranges of $S_h$ in (c) are the same as Figs. 3(a)–3(c).

Equation (5) leads to a third-order EP when $\gamma^{int} = \sqrt{2}g\omega_{0}$ . The resonance frequencies and intrinsic losses of this system are shown in Figs. 4(a) and 4(b), demonstrating the coalescence of the three BICs into a third-order EP-BIC [56].

Finally, we investigate the effect of symmetry breaking perturbations for EP-BICs of arbitrary order. Employing a generalized Newton-Puisseux series $[14,17]$ , we derived a general rule for the total losses of the quasi-BICs emerging from an EP-BIC of order m $[56]$ , which reads $\gamma_{h} = \gamma_{EP} + \operatorname{Im}\{\alpha^{1/m}\zeta^{h-1}\}p^{1/m}$ , where $h = 1\ldots m$ is an integer denoting each quasi-BIC spanned by the EP-BIC, $\alpha$ is a complex number, $\zeta = e^{2\pi i/m}$ , p is the asymmetry parameter (e.g., $S_{h}$ ), and $\gamma_{EP}$ is the intrinsic loss at the EP-BIC.

A third-order EP-BIC results in quasi-BICs scaling as $\gamma_{h} \propto p^{1/3}$ , implying larger changes for small p with respect to conventional BICs and second-order EP-BICs. This is shown in Fig. 4(c), where we compare the change in loss $\gamma_{h} - \gamma_{EP}$ for perturbations of a conventional S-BIC and second- and third-order EP-BICs.

In conclusion, in this Letter we confirmed the possibility of coalescing BICs into EPs of arbitrary order, forming an EP-BIC. The key for their realization is the introduction of asymmetric loss into a system supporting two or more BICs. We have confirmed our predictions by simulations of bilayer and trilayer metasurfaces hosting S-BICs coalescing at second- and third-order EP-BICs. This mechanism might not be unique. For instance, EP-BICs might exist in the presence of unidirectional coupling $[65–67]$ . This constitutes a promising research direction, since it would remove the need of intrinsic losses that limit the achievable Q factor.

The novel states inherit the infinite radiative Q factors of BICs, but possess the enhanced eigenvalue sensitivity of EPs $[68]$ . Interestingly, introducing radiation losses to an EP-BIC of order m, mediated by a parameter p, results in a change of the total losses in the order of $p^{1/m}$ , unlike the linear response of conventional S-BICs. Owing to the broad applicability of our formalism, EP-BICs are expected to be a general wave phenomenon beyond the studied optical setup. For instance, we anticipate them to play an important role in the emerging hybrid dielectric-plasmonic metasurfaces $[69,70]$ . Besides their fundamental interest, the absence of radiation loss makes EP-BICs exciting candidates for the realization of EP sensors at the nanoscale.

Acknowledgments—This research was funded in whole, or in part, by the Austrian Science Fund (FWF) [Grant DOI:10.55776/P36864]. A.C.V. acknowledges funding from the Latvian Council of Science, Project No. NEO-NATE, No. 1zp-2022/1–0553.

[1] Y. Ashida, Z. Gong, and M. Ueda, Non-Hermitian physics, Adv. Phys. 69, 249 (2020).

[2] R. El-Ganainy, K. G. Makris, M. Khajavikhan, Z. H. Musslimani, S. Rotter, and D. N. Christodoulides, Non-Hermitian physics and PT symmetry, Nat. Phys. 14, 11 (2018).

[3] R. El-Ganainy, M. Khajavikhan, D. N. Christodoulides, and S. K. Ozdemir, The dawn of non-Hermitian optics, Commun. Phys. 2, 37 (2019).

[4] H. Feshbach, Unified theory of nuclear reactions, Ann. Phys. (N.Y.) 5, 357 (1958).

[5] G. Lindblad, On the generators of quantum dynamical semigroups, Commun. Math. Phys. 48, 119 (1976).

[6] J. Y. Lee, J. Ahn, H. Zhou, and A. Vishwanath, Topological correspondence between Hermitian and non-Hermitian systems: Anomalous dynamics, Phys. Rev. Lett. 123, 206404 (2019).

[7] F. Song, S. Yao, and Z. Wang, Non-Hermitian skin effect and chiral damping in open quantum systems, Phys. Rev. Lett. 123, 170401 (2019).

[8] Y. D. Chong, L. Ge, H. Cao, and A. D. Stone, Coherent perfect absorbers: Time-reversed lasers, Phys. Rev. Lett. 105, 053901 (2010).

[9] A. Regensburger, C. Bersch, M.-A. Miri, G. Onishchukov, D. N. Christodoulides, and U. Peschel, Parity–time synthetic photonic lattices, Nature (London) 488, 167 (2012).

[10] W. Liu, M. Li, R. S. Guzzon, E. J. Norberg, J. S. Parker, M. Lu, L. A. Coldren, and J. Yao, An integrated parity-time symmetric wavelength-tunable single-mode microring laser, Nat. Commun. 8, 15389 (2017).

[11] C. Coulais, D. Sounas, and A. Alu, Static non-reciprocity in mechanical metamaterials, Nature (London) 542, 461 (2017).

[12] A. Canós Valero, H. K. Shamkhi, A. S. Kupriianov, T. Weiss, A. A. Pavlov, D. Redka, V. Bobrovs, Y. Kivshar, and A. S. Shalin, Superscattering emerging from the physics

of bound states in the continuum, Nat. Commun. 14, 4689 (2023).

[13] M.-A. Miri and A. Alù, Exceptional points in optics and photonics, Science 363, eaar7709 (2019).

[14] W. Heiss, The physics of exceptional points, J. Phys. A 45, 444016 (2012).

[15] J. Wiersig, Enhancing the sensitivity of frequency and energy splitting detection by using exceptional points: Application to microcavity sensors for single-particle detection, Phys. Rev. Lett. 112, 203901 (2014).

[16] W. Langbein, No exceptional precision of exceptional-point sensors, Phys. Rev. A 98, 023805 (2018).

[17] H. Hodaei, A. U. Hassan, S. Wittek, H. Garcia-Gracia, R. El-Ganainy, D. N. Christodoulides, and M. Khajavikhan, Enhanced sensitivity at higher-order exceptional points, Nature (London) 548, 187 (2017).

[18] Q. Song, M. Odeh, J. Zúñiga-Pérez, B. Kanté, and P. Genevet, Plasmonic topological metasurface by encircling an exceptional point, Science 373, 1133 (2021).

[19] Y. Xu, L. Li, H. Jeong, S. Kim, I. Kim, J. Rho, and Y. Liu, Subwavelength control of light transport at the exceptional point by non-Hermitian metagratings, Sci. Adv. 9, eadf3510 (2023).

[20] J.-H. Park, A. Ndao, W. Cai, L. Hsu, A. Kodigala, T. Lepetit, Y.-H. Lo, and B. Kanté, Symmetry-breaking-induced plasmonic exceptional points and nanoscale sensing, Nat. Phys. 16, 462 (2020).

[21] Y. Sang, C.-Y. Wang, S. S. Raja, C.-W. Cheng, C.-T. Huang, C.-A. Chen, X.-Q. Zhang, H. Ahn, C.-K. Shih, Y.-H. Lee et al., Tuning of two-dimensional plasmon–exciton coupling in full parameter space: A polaritonic non-Hermitian system, Nano Lett. 21, 2596 (2021).

[22] B. Zhen, C. W. Hsu, Y. Igarashi, L. Lu, I. Kaminer, A. Pick, S.-L. Chua, J. D. Joannopoulos, and M. Soljačić, Spawning rings of exceptional points out of Dirac cones, Nature (London) 525, 354 (2015).

[23] K. S. Netherwood, H. Riley, and E. A. Muljarov, Exceptional points in optical systems: A resonant-state expansion study, Phys. Rev. A 110, 033518 (2024).

[24] A. Guo, G. J. Salamo, D. Duchesne, R. Morandotti, M. Volatier-Ravat, V. Aimez, G. A. Siviloglou, and D. N. Christodoulides, Observation of PT-symmetry breaking in complex optical potentials, Phys. Rev. Lett. 103, 093902 (2009).

[25] A. Canoa Valero, V. Bobrovs, T. Weiss, L. Gao, A. S. Shalin, and Y. Kivshar, Bianisotropic exceptional points in an isolated dielectric nanoparticle, Phys. Rev. Res. 6, 013053 (2024).

[26] H. Jiang, W. Zhang, G. Lu, L. Ye, H. Lin, J. Tang, Z. Xue, Z. Li, H. Xu, and Q. Gong, Exceptional points and enhanced nanoscale sensing with a plasmon-exciton hybrid system, Photonics Res. 10, 557 (2022).

[27] N. S. Solodovchenko, K. B. Samusev, and M. F. Limonov, Quadruplets of exceptional points and bound states in the continuum in dielectric rings, Phys. Rev. B 109, 075131 (2024).

[28] C. W. Hsu, B. Zhen, A. D. Stone, J. D. Joannopoulos, and M. Soljačić, Bound states in the continuum, Nat. Rev. Mater. 1, 16048 (2016).

[29] K. L. Koshelev, Z. F. Sadrieva, A. A. Shcherbakov, Yu. S. Kivshar, and A. A. Bogdanov, Bound states in the

continuum in photonic structures, Phys. Usp. 66, 494 (2023).

[30] B. Zhen, C. W. Hsu, L. Lu, A. D. Stone, and M. Soljačić, Topological nature of optical bound states in the continuum, Phys. Rev. Lett. 113, 257401 (2014).

[31] J. Von Neumann and E. Wigner, On some peculiar discrete eigenvalues, Phys. Z. 30, 465 (1929).

[32] F. H. Stillinger and D. R. Herrick, Bound states in the continuum, Phys. Rev. A 11, 446 (1975).

[33] S. I. Azzam and A. V. Kildishev, Photonic bound states in the continuum: From basics to applications, Adv. Opt. Mater. 9, 2001469 (2021).

[34] Y. Plotnik, O. Peleg, F. Dreisow, M. Heinrich, S. Nolte, A. Szameit, and M. Segev, Experimental observation of optical bound states in the continuum, Phys. Rev. Lett. 107, 183901 (2011).

[35] Y. Liang, K. Koshelev, F. Zhang, H. Lin, S. Lin, J. Wu, B. Jia, and Y. Kivshar, Bound states in the continuum in anisotropic plasmonic metasurfaces, Nano Lett. 20, 6351 (2020).

[36] S. I. Azzam, V. M. Shalaev, A. Boltasseva, and A. V. Kildishev, Formation of bound states in the continuum in hybrid plasmonic-photonic systems, Phys. Rev. Lett. 121, 253901 (2018).

[37] S. Neale and E. A. Muljarov, Accidental and symmetry-protected bound states in the continuum in a photonic-crystal slab: A resonant-state expansion study, Phys. Rev. B 103, 155112 (2021).

[38] K. Koshelev, S. Lepeshov, M. Liu, A. Bogdanov, and Y. Kivshar, Asymmetric metasurfaces with high-Q resonances governed by bound states in the continuum, Phys. Rev. Lett. 121, 193903 (2018).

[39] F. Yesilkoy, E. R. Arvelo, Y. Jahani, M. Liu, A. Tittl, V. Cevher, Y. Kivshar, and H. Altug, Ultrasensitive hyperspectral imaging and biodetection enabled by dielectric metasurfaces, Nat. Photonics 13, 390 (2019).

[40] V. Kravtsov, E. Khestanova, F. A. Benimetskiy, T. Ivanova, A. K. Samusev, I. S. Sinev, D. Pidgayko, A. M. Mozharov, I. S. Mukhin, M. S. Lozhkin, Y. V. Kapitonov, A. S. Brichkin, V. D. Kulakovskii, I. A. Shelykh, A. I. Tartakovskii, P. M. Walker, M. S. Skolnick, D. N. Krizhanovskii, and I. V. Iorsh, Nonlinear polaritons in a monolayer semiconductor coupled to optical bound states in the continuum, Light Sci. Appl. 9, 1 (2020).

[41] E. Maggiolini, L. Polimeno, F. Todisco, A. Di Renzo, B. Han, M. De Giorgi, V. Ardizzone, C. Schneider, R. Mastria, A. Cannavale, M. Pugliese, L. De Marco, A. Rizzo, V. Maiorano, G. Gigli, D. Gerace, D. Sanvitto, and D. Ballarini, Strongly enhanced light–matter coupling of monolayer WS2 from a bound state in the continuum, Nat. Mater. 22, 964 (2023).

[42] Y. Yang, Y.-P. Wang, J. W. Rao, Y. S. Gui, B. M. Yao, W. Lu, and C.-M. Hu, Unconventional singularity in anti-parity-time symmetric cavity magnonics, Phys. Rev. Lett. 125, 147202 (2020).

[43] Z.-L. Deng, F.-J. Li, H. Li, X. Li, and A. Alù, Extreme diffraction control in metagratings leveraging bound states in the continuum and exceptional points, Laser Photonics Rev. 16, 2100617 (2022).

[44] Z. Sakotic, P. Stankovic, V. Bengin, A. Krasnok, A. Alú, and N. Jankovic, Non-Hermitian control of topological

scattering singularities emerging from bound states in the continuum, Laser Photonics Rev. 17, 2200308 (2023).

[45] H. Qin, X. Shi, and H. Ou, Exceptional points at bound states in the continuum in photonic integrated circuits, Nanophotonics 11, 4909 (2022).

[46] R. Kikkawa, M. Nishida, and Y. Kadoya, Bound states in the continuum and exceptional points in dielectric waveguide equipped with a metal grating, New J. Phys. 22, 073029 (2020).

[47] A. Aigner, A. Tittl, J. Wang, T. Weber, Y. Kivshar, S. A. Maier, and H. Ren, Plasmonic bound states in the continuum to tailor light-matter coupling, Sci. Adv. 8, eadd4816 (2022).

[48] Y. Tao, B. Fang, G.-M. Pan, D.-Q. Zhang, Z.-W. Jin, Z. Hong, and F.-Z. Shu, Tunable high-Q resonance based on hybrid phase-change metasurfaces, ACS Appl. Opt. Mater. 1, 1452 (2023).

[49] S. Both and T. Weiss, Resonant states and their role in nanophotonics, Semicond. Sci. Technol. 37, 013002 (2021).

[50] P. T. Kristensen, K. Herrmann, F. Intravaia, and K. Busch, Modeling electromagnetic resonators using quasinormal modes, Adv. Opt. Photonics 12, 612 (2020).

[51] P. Lalanne, W. Yan, K. Vynck, C. Sauvan, and J.-P. Hugonin, Light interaction with photonic and plasmonic resonances, Laser Photonics Rev. 12, 1700113 (2018).

[52] A. F. Sadreev, Interference traps waves in an open system: Bound states in the continuum, Rep. Prog. Phys. 84, 055901 (2021).

[53] S. V. Shabanov, Resonant light scattering and higher harmonic generation by periodic subwavelength arrays, Int. J. Mod. Phys. B 23, 5191 (2009).

[54] C. F. Doiron, I. Brener, and A. Cerjan, Realizing symmetry-guaranteed pairs of bound states in the continuum in metasurfaces, Nat. Commun. 13, 7534 (2022).

[55] E. Teller, The crossing of potential surfaces., J. Phys. Chem. 41, 109 (1937).

[56] See Supplemental Material at http://link.aps.org/supplemental/10.1103/PhysRevLett.134.103802, which includes Refs. [57–59], for (i) non-existence of the EP-BIC for conventional lossless systems, (ii) mode coupling theory of resonant states, (iii) simplified effective Hamiltonian, (iv) asymptotics of an EP-BIC with the introduction of radiative loss, (v) separation of losses into absorptive and radiative, (vi) reflection spectra with different off-center holes, (vii) EP-BIC dispersion in reciprocal space, (viii) Details on the numerical models, (ix) investigation of the Q factors of the quasi BICs, (x) numerical simulations of metasurfaces with dissimilar thicknesses, (xi) field distributions of the BICs near the third order EP-BIC. The raw data for the figures can be found at [60].

[57] M. Hentschel, K. Koshelev, F. Sterl, S. Both, J. Karst, L. Shamsafar, T. Weiss, Y. Kivshar, and H. Giessen, Dielectric mie voids: Confining light in air, Light 12, 3 (2023).

[58] K. Cognee, Hybridization of open photonic resonators, Ph.D. thesis, Université de Bordeaux; Universiteit van Amsterdam, 2020.

[59] W. Suh, Z. Wang, and S. Fan, Temporal coupled-mode theory and the presence of non-orthogonal modes in lossless multimode cavities, IEEE J. Quantum Electron. 40, 1511 (2004).

[60] A. Canós Valero, Raw plots from the paper “Exceptional Bound States in the Continuum” (2025), 10.5281/zenodo .14884499.

[61] C. Tao, J. Zhu, Y. Zhong, and H. Liu, Coupling theory of quasinormal modes for lossy and dispersive plasmonic nanoresonators, Phys. Rev. B 102, 045430 (2020).

[62] H. Zhou, C. Peng, Y. Yoon, C. W. Hsu, K. A. Nelson, L. Fu, J. D. Joannopoulos, M. Soljačić, and B. Zhen, Observation of bulk Fermi arc and polarization half charge from paired exceptional points, Science 359, 1009 (2018).

[63] The reason for the difference between the fields is mainly due to the impossibility of exactly reaching an EP numerically, since it is an isolated point in the spectra. Nevertheless, we are generally interested in the effects occurring in its vicinity.

[64] T. Weiss, M. Mesch, M. Schäferling, H. Giessen, W. Langbein, and E. A. Muljarov, From dark to bright: First-order perturbation theory with analytical mode normalization for plasmonic nanoantenna arrays applied to refractive index sensing, Phys. Rev. Lett. 116, 237401 (2016).

[65] S. Wang, B. Hou, W. Lu, Y. Chen, Z. Zhang, and C. T. Chan, Arbitrary order exceptional point induced by photonic spin-orbit interaction in coupled resonators, Nat. Commun. 10, 832 (2019).

[66] X.-Y. Wang, F.-F. Wang, and X.-Y. Hu, Waveguide-induced coalescence of exceptional points, Phys. Rev. A 101, 053820 (2020).

[67] J. Wiersig, Revisiting the hierarchical construction of higher-order exceptional points, Phys. Rev. A 106, 063526 (2022).

[68] In Sec. X of the Supplemental Material [56], we show that even without precisely engineering the device to be at the EP-BIC, the BICs in its vicinity are also significantly more sensitive to perturbations than conventional BICs.

[69] S. I. Azzam, V. M. Shalaev, A. Boltasseva, and A. V. Kildishev, Formation of bound states in the continuum in hybrid plasmonic-photonic systems, Phys. Rev. Lett. 121, 253901 (2018).

[70] Y. Yang, O. D. Miller, T. Christensen, J. D. Joannopoulos, and M. Soljacic, Low-loss plasmonic dielectric nanoresonators, Nano Lett. 17, 3238 (2017).
