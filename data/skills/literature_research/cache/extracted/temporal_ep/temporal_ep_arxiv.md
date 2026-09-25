# Exceptional Point Dynamics in Photonic Time Crystals for Enhanced Optical Sensing

Saurabh Mani Tripathi, $^{1}$ Shalini Kumari, $^{2}$ Krishnan Kundan, $^{2}$ and Neha Ahlawat $^{1}$

$^{1}$ Optics and Photonics Centre, Indian Institute of Technology Delhi, New Delhi 110016, India

$^{2}$ Department of Physics, Indian Institute of Technology Delhi, New Delhi 110016, India (Dated: December 3, 2025)

Exceptional points (EPs) in non-Hermitian photonics offer singular sensitivity enhancements but have thus far been realized almost exclusively in spatially engineered platforms with fixed geometries and limited tunability. Here we extend EP physics into the temporal domain by introducing balanced gain-loss modulation in a photonic time crystal (PTC). A time-periodic refractive-index modulation $n(t) = n_{0} + \delta n \cos(\Omega t)$ generates an effective non-Hermitian Floquet Hamiltonian that supports coalescence of quasi-eigenmodes in frequency space, constituting a genuine temporal exceptional point. Using a reduced two-mode model for the dominant frequency sidebands, we derive a non-Hermitian dimer Hamiltonian $H_{\mathrm{PT}}(\Delta, \gamma, \kappa)$ that is strictly PT-symmetric for $\Delta = 0$ and identify the exact EP condition. Numerical analysis reveals the associated Riemann-sheet topology, mode exchange and Berry-phase accumulation upon encirclement of the EP, and the characteristic $\sqrt{\varepsilon}$ perturbation response indicative of enhanced sensing. We further construct a non-Hermitian transmission model that is exact within the reduced two-mode description, compute the Cramér–Rao bound (CRB) for temperature estimation under an explicit noise model, and show that EP-enhanced sensitivity persists when compared to a linewidth-matched Hermitian reference under identical resource constraints. Monte Carlo simulations confirm that the CRB is saturable using spectral measurements. These results establish temporal non-Hermiticity as a new paradigm for dynamically reconfigurable, broadband, and geometry-independent exceptional-point photonics.

## I. INTRODUCTION

Non-Hermitian photonics has emerged as a powerful framework for controlling light–matter interactions by engineering gain and loss in optical systems $[1-5]$ . In parity–time $(\mathcal{PT})$ -symmetric structures, balanced gain and loss can yield entirely real eigenvalue spectra below a symmetry-breaking threshold, even though the underlying Hamiltonian is non-Hermitian $[1, 3]$ . At a critical value of the gain–loss parameter, the system encounters an exceptional point (EP), where both eigenvalues and eigenvectors coalesce and the Hamiltonian becomes non-diagonalizable. In the vicinity of such a non-Hermitian degeneracy, observable quantities such as frequency splitting exhibit a non-linear dependence on external perturbations, typically scaling as $\sqrt{\varepsilon}$ rather than $\varepsilon$ . This square-root response amplifies small perturbations and has been proposed and demonstrated as a route to enhanced sensing $[4, 6-8]$ .

Optical EPs have been realized in a variety of spatially engineered platforms, including coupled resonators, photonic-crystal cavities, and microtoroids, as well as in electronic and microwave circuits $[9–15]$ . Hodaei et al. $[16]$ and Chen et al. $[17]$ independently demonstrated EPs in coupled microring and whispering-gallery resonators, achieving strong enhancement of the frequency-splitting response to perturbations $[6, 8, 16, 17]$ . Subsequent work has explored higher-order EPs, cascaded non-Hermitian structures, and EP-based inertial sensors to further enlarge the parameter space for ultra-sensitive measurements $[8, 16, 18, 19]$ . Dynamical encirclement of EPs and associated chiral mode conversion have also been demonstrated in optical and microwave systems $[15, 20]$ . Despite these advances, existing implementations share a common limitation: the non-Hermiticity is embedded in a fixed spatial geometry. Once fabricated, the refractive-index distribution $n(x)$ , coupling coefficients, and gain-loss profiles are largely static, which constrains tunability, bandwidth, and the ability to reconfigure the EP in situ.

A complementary route to non-Hermitian photonics is to shift the locus of control from space to time. In temporally modulated media—often referred to as photonic time crystals (PTCs)—the refractive index varies periodically in time, $n(t) = n_{0} + \delta n \cos(\Omega t)$ , generating a Floquet band structure in frequency rather than in wave vector [21–23]. Temporal modulation couples spectral components separated by integer multiples of the modulation frequency $\Omega$ , leading to bandgaps and mode hybridization in the frequency domain. When combined with gain and loss, these time-periodic systems realize explicitly non-Hermitian time-Floquet Hamiltonians that can host exceptional points and other non-Hermitian singularities in the temporal domain [22, 24–26]. In this setting, EPs correspond to coalescence of Floquet quasi-eigenvalues in frequency space and can be tuned dynamically by varying the modulation amplitude, frequency, or phase [21, 25].

The distinction between spatial and temporal EP platforms is summarized schematically in Fig. 1. Spatial EP devices, such as coupled cavities or gratings, realize non-Hermiticity through engineered distributions $n(x)$ and fixed coupling. Their spectra are defined in $(k,\omega)$ space and are constrained by fabrication tolerances and device footprint [1-3, 8]. By contrast, temporal EPs arise in uniform media subject to externally driven $n(t)$ and gain-loss modulation. Here, the relevant band structure is defined in $(\omega,\Omega)$ space, and the operating point can be tuned in real time by adjusting drive parameters [21-23]. This naturally enables broadband operation, adaptive sensitivity control, and reconfigurability, all of which are highly desirable for integrated photonic sensing.

In this work we develop a minimal, yet fully non-Hermitian, theoretical model for temporal exceptional points in a photonic time crystal and connect it quantitatively to sensing performance. Starting from Maxwell's equations in a time-modulated dielectric, we derive a reduced two-mode Floquet model that captures the dominant coupling between a pair of frequency components. This leads to an effective non-Hermitian dimer Hamiltonian $H_{\mathrm{PT}}(\Delta,\gamma,\kappa)$ , where $\Delta$ is a detuning parameter that can be controlled by temperature or refractive-index changes, $\gamma$ represents balanced gain/loss, and $\kappa$ is the modulation-induced coupling. We obtain closed-form expressions for the eigenvalues, identify the exact EP condition, and map out the resulting Riemann surfaces.

We then compute geometric signatures of the temporal EP. By encircling the EP in the $(\Delta,\gamma)$ parameter plane, we observe mode exchange and accumulation of a biorthogonal Berry phase of magnitude $\pi$ , which serves as a robust topological indicator of the EP. Next, we formulate a non-Hermitian scattering problem and derive the transmission spectrum $|T(\omega)|^{2}$ of a probe field interacting with the PTC. This transmission model is used to define a realistic sensing protocol in which temperature-induced shifts of the detuning are inferred from spectral measurements. We compute the Cramér–Rao bound (CRB) for temperature estimation directly from the Fisher information under a clearly stated noise model, construct a linewidth-matched Hermitian reference that is operated under identical resource constraints, and demonstrate a clear EP-induced CRB reduction. Finally, extensive Monte Carlo simulations confirm that the CRB for both the temperature and the associated eigenvalue splitting can be saturated using least-squares fitting to the exact transmission model.

Our results show that temporal non-Hermiticity provides a flexible and dynamically reconfigurable platform for exceptional-point sensing. Unlike static spatial EP devices, temporal EPs can be tuned on demand by adjusting the modulation parameters, enabling broadband operation and adaptive sensitivity while preserving the underlying EP topology $[21, 22, 25]$ .

## II. THEORETICAL MODEL AND EFFECTIVE HAMILTONIAN

## A. From time-varying permittivity to a non-Hermitian dimer

We consider a homogeneous medium whose permittivity varies periodically in time,

$$
\epsilon (t) = \epsilon_ {0} \big [ 1 + m \cos (\Omega t) \big ],\tag{1}
$$

where $m \ll 1$ is the modulation depth and $\Omega$ is the modulation frequency. In the absence of gain and loss, the electric field $E(z,t)$ in such a medium satisfies the one-dimensional wave equation

![](images/1dff4db2cb56b1a383ad283a0a2508282afe23171ea14adc62e9225f804c18fa.jpg)
FIG. 1. Conceptual comparison between spatial and temporal exceptional-point (EP) systems. (a) Spatial EP devices rely on engineered refractive-index profiles $n(x)$ and fixed coupling to realize non-Hermitian band structures in momentum space. (b) In photonic time crystals (PTCs), a uniform medium with time-periodic refractive-index modulation $n(t) = n_0 + \delta n\cos (\Omega t)$ and dynamic gain-loss $\gamma (t)$ yields an effective non-Hermitian Floquet Hamiltonian in the frequency domain, enabling tunable temporal EPs.

$$
\partial_ {z} ^ {2} E (z, t) - \mu_ {0} \epsilon (t) \partial_ {t} ^ {2} E (z, t) = 0.\tag{2}
$$

For a monochromatic carrier at frequency $\omega_{0}$ near a particular guided or cavity mode with propagation constant $\beta_{0}$ , we expand the field in a truncated Floquet basis as

$$
E (z, t) = \Big [ a _ {1} (t) e ^ {- i \omega_ {0} t} + a _ {2} (t) e ^ {- i (\omega_ {0} + \Omega) t} \Big ] e ^ {i \beta_ {0} z} + \mathrm{H.c.},\tag{3}
$$

where $a_{1,2}(t)$ are slowly varying envelopes associated with the two dominant frequency components. Substituting this ansatz into Maxwell's equations in the time-varying medium and applying the rotating-wave and slowly varying envelope approximations yields a pair of coupled-mode equations for the Floquet amplitudes. Retaining only near-resonant terms and assuming a weak modulation $m \ll 1$ so that higher-order sidebands remain off-resonant, we obtain

$$
\dot {a} _ {1} = - i (\Delta + i \gamma) a _ {1} - i \kappa a _ {2},\tag{4a}
$$

$$
\dot {a} _ {2} = - i \kappa a _ {1}, - i (- \Delta - i \gamma) a _ {2},\tag{4b}
$$

where $\kappa$ is the effective modulation-induced coupling between the two Floquet modes, $\gamma$ is an effective gain/loss rate that can be implemented, for example, via pump-induced amplification in one component and balanced attenuation in the other, and $\Delta$ represents a detuning that can be tuned by temperature or refractive-index changes. The detailed steps of this reduction from the full Maxwell equations, together with estimates of the regime of validity of the two-mode truncation, are provided in the Supplementary Information.

Equations (4) can be written compactly as

$$
\dot {\mathbf {a}} = - i H _ {\mathrm{PT}}   \mathbf {a}, \qquad \mathbf {a} = \binom{a _ {1}}{a _ {2}},\tag{5}
$$

![](images/8b42c528f45aadce666ff3428cc4bc095af14a49ab8ed99106e8db895a0a0950.jpg)

with the effective non-Hermitian dimer Hamiltonian

$$
H _ {\mathrm{PT}} (\Delta , \gamma , \kappa) = \left( \begin{array}{c c} \Delta + i \gamma & \kappa \\ \kappa & - \Delta - i \gamma \end{array} \right).\tag{6}
$$

For $\Delta = 0$ , this Hamiltonian is PT-symmetric in the standard sense, with $P = \sigma_{x}$ and T the complex-conjugation operator, and exhibits an unbroken PT phase for $|\gamma| < |\kappa|$ and a broken phase for $|\gamma| > |\kappa|$ [1, 3]. For $\Delta \neq 0$ , the Hamiltonian is no longer PT-symmetric, though the EP still exists at $\Delta = 0$ , $\gamma = \pm\kappa$ for each fixed $\kappa$ , as discussed below. In all cases, the degrees of freedom correspond to frequency sidebands rather than spatial modes, making Eq. (6) the temporal analogue of the widely studied coupled-resonator PT dimer [2, 4].

## B. Eigenvalues and exceptional-point condition

The eigenvalues of $H_{PT}$ are obtained in closed form as

$$
\lambda_ {\pm} = \pm \sqrt {(\Delta + i \gamma) ^ {2} + \kappa^ {2}}.\tag{7}
$$

We emphasize that Eq. (7) is used throughout our analysis; no approximate eigenvalue formulas or surrogate polynomials are employed in any of the numerics below. The corresponding EP condition follows from the coalescence of the two eigenvalues and eigenvectors, which occurs when the argument of the square root vanishes,

$$
(\Delta + i \gamma) ^ {2} + \kappa^ {2} = 0.\tag{8}
$$

For real-valued $\Delta$ and $\gamma$ , this yields the EP locus

$$
\Delta_ {\mathrm{EP}} = 0, \qquad \gamma_ {\mathrm{EP}} = \pm \kappa .\tag{9}
$$

In the three-dimensional parameter space $(\Delta,\gamma,\kappa)$ this locus forms a line; for fixed coupling $\kappa$ it corresponds to the pair of points $(\Delta,\gamma)=(0,\pm\kappa)$ . At these points $H_{PT}$ becomes nondiagonalizable and the eigenvectors coalesce, as in other second-order EP implementations [3, 6].

## C. Normalization, parameter choices, and temperature-induced detuning

To present the results in a dimensionless and platform-independent way, we normalize all frequencies and rates to a reference coupling scale $g_{0}$ . Unless otherwise stated, we set $\kappa = g_{0}$ so that $\Delta/g_{0}$ , $\gamma/g_{0}$ , and $\lambda_{\pm}/g_{0}$ provide a convenient parametrization of the eigenvalue landscape. This choice does not restrict generality and can be mapped to physical units once a specific implementation platform is specified.

In a realistic implementation, temperature changes modify the effective refractive index $n_{\mathrm{eff}}(T)$ and hence the detuning,

$$
\Delta (T) = \Delta_ {0} + \alpha_ {T} (T - T _ {0}),\tag{10}
$$

![](images/9fb69e10d9569c6d9fecb34d427a642e53149416f5520e4794569acc8ec1406e.jpg)
FIG. 2. Exact eigenvalue landscape of the temporal non-Hermitian dimer. (a) and (b) Real and imaginary parts of $\lambda_{+}$ as functions of detuning $\Delta / g_0$ for several values of $\gamma / g_0$ , illustrating the transition from the unbroken $\mathcal{PT}$ phase ( $\gamma < \kappa$ ) through the EP ( $\gamma = \kappa$ ) into the broken phase ( $\gamma > \kappa$ ) at $\Delta = 0$ . (c) and (d) Two-dimensional maps of the mode splitting $|\lambda_{+} - \lambda_{-}| / g_0$ and the magnitude of the imaginary part $|\operatorname{Im} \lambda_{+}| / g_0$ over the $(\Delta / g_0, \gamma / g_0)$ plane. The EP appears as a pinch point at $(\Delta, \gamma) = (0, \pm \kappa)$ .

where $\alpha_{T}=d\Delta/dT$ is an effective thermo-detuning coefficient. Throughout the sensing analysis below, we treat $\Delta T=T-T_{0}$ as the primary perturbation and examine how the complex eigenvalues and associated observables respond as the system is biased near the EP. For the numerical examples, we choose parameters such that all effective decay rates remain negative (see Sec. IV), ensuring dynamical stability.

## III. EIGENVALUE TOPOLOGY AND EP ENCIRCLEMENT

We begin by characterizing the eigenvalue landscape of Eq. (6) as a function of detuning $\Delta$ and gain/loss $\gamma$ . Figure 2 summarizes the behavior of the real and imaginary parts of $\lambda_{\pm}$ and highlights the location of the EP.

Figure 3 displays the real parts of the two eigenvalue sheets over the $(\Delta,\gamma)$ plane, revealing the characteristic double-sheeted Riemann surface. The two branches meet at the EP points and form a square-root branch cut; a continuous path that crosses the branch cut leads to sheet exchange, in direct analogy with spatial EP platforms [3, 6].

To probe the geometric and topological properties of the EP, we adiabatically encircle the EP in the $(\Delta,\gamma)$ plane while tracking the biorthogonal eigenvectors of $H_{PT}$ . We construct left and right eigenvectors $(v_{L},v_{R})$ that satisfy the biorthogonality condition $v_{L}^{\dagger}v_{R}=I$ and compute the accumulated Berry phase from the biorthogonal Berry connection. In the continuous limit, the geometric phase acquired along a closed loop C in parameter space is

$$
\phi_ {B} = i \oint_ {C} \mathrm{d} \boldsymbol {\lambda} \cdot \left\langle v _ {L} (\boldsymbol {\lambda}) \mid \nabla_ {\boldsymbol {\lambda}} v _ {R} (\boldsymbol {\lambda}) \right\rangle ,\tag{11}
$$

![](images/625cbc5820e1ec126f56bf559c452b680431496908ec1d0ad4db34e129df67b2.jpg)

![](images/ae983f7fbff010b67907f029ed70a536e9d9f96524a4e2f80c49055250bbbc8a.jpg)

FIG. 3. Riemann topology of the exact eigenvalues. Real parts of the eigenvalue sheets for (a) $\lambda_{+}$ and (b) $\lambda_{-}$ , as functions of $(\Delta / g_0, \gamma / g_0)$ . The two sheets coalesce at the EP points along $\Delta = 0$ and $\gamma = \pm \kappa$ , forming a double-sheeted Riemann surface with a square-root branch point.
![](images/673204c1ac975945ba92a2eca05ba367b207d713c18992be1d30444967d4e86f.jpg)

![](images/bd3c76a9c1e9841fea2ac19bcf7c09559453e65f344ab91d3500ba6055a2f50c.jpg)
FIG. 4. Encirclement of the EP and Berry-phase accumulation. (a) Parametric loop in the $(\Delta/g_{0},\gamma/g_{0})$ plane encircling the EP. (b) Biorthogonal Berry phase $\phi_{B}(\theta)$ as a function of the loop angle $\theta$ , obtained from the product of overlaps of left and right eigenvectors along the path according to Eq. (12). A total phase of magnitude $\pi$ is accumulated after one encirclement, confirming the nontrivial EP topology. The calculation is purely parametric in $(\Delta,\gamma)$ and does not assume a specific dynamical encirclement protocol in time.

where $\lambda = (\Delta, \gamma)$ . In the numerics, we discretize the loop into $N$ points labeled by an angle $\theta_{k}$ and evaluate

$$
\phi_ {B} \approx - \mathrm{Im} \ln \prod_ {k = 0} ^ {N - 1} \bigl \langle v _ {L} (\theta_ {k}) \big | v _ {R} (\theta_ {k + 1}) \bigr \rangle ,\tag{12}
$$

with $\theta_{N} \equiv \theta_{0}$ ensuring gauge invariance. The resulting mode exchange and phase accumulation are shown in Fig. 4. Similar EP-encirclement phenomena have been reported in spatial non-Hermitian systems [15, 20].

The appearance of mode exchange and a Berry phase of $\pi$ is a robust indicator of a genuine second-order EP and is insensitive to small deformations of the loop, as further quantified in Supplementary Fig. S2.

## IV. NON-HERMITIAN TRANSMISSION MODEL AND SENSING PROTOCOL

To connect the temporal EP to measurable quantities, we embed the non-Hermitian dimer into a simple scattering geometry in which one of the modes is coupled to an input/output channel. A concrete physical realization is a single-mode ring resonator or guided mode whose effective index is modulated in time according to Eq. (1), with balanced gain and loss implemented via optical pumping or carrier modulation in two spectral components [13, 21]. The effective non-Hermitian Hamiltonian including internal loss $\kappa_0$ can be written as

$$
H _ {\mathrm{eff}} = H _ {\mathrm{PT}} (\Delta , \gamma , \kappa) - i \kappa_ {0} \mathbb {I},\tag{13}
$$

where $\kappa_{0} > 0$ represents additional uniform decay (e.g., intrinsic cavity loss) and ensures overall stability when chosen such that all eigenvalues of $H_{eff}$ have negative imaginary parts.

For a probe field at frequency $\omega$ , the steady-state intracavity amplitudes satisfy the frequency-domain input–output relation

$$
\left[ i \omega \mathbb {I} - H _ {\mathrm{eff}} \right] \mathbf {a} (\omega) = \sqrt {\kappa_ {\mathrm{in}}} \mathbf {s} _ {\mathrm{in}} (\omega),\tag{14}
$$

where $\kappa_{in}$ is an input coupling rate and $\mathbf{s}_{\mathrm{in}} = (s_{\mathrm{in}}, 0)^{\mathsf{T}}$ denotes a source driving only the first Floquet mode. The transmitted field in the same channel is then

$$
s _ {\mathrm{out}} (\omega) = s _ {\mathrm{in}} (\omega) - \sqrt {\kappa_ {\mathrm{in}}} a _ {1} (\omega),\tag{15}
$$

so that the observable transmission spectrum is

$$
T (\omega) = \frac {s _ {\mathrm{out}} (\omega)}{s _ {\mathrm{in}} (\omega)} = 1 - \kappa_ {\mathrm{in}} \big [ (i \omega \mathbb {I} - H _ {\mathrm{eff}}) ^ {- 1} \big ] _ {1 1},\tag{16}
$$

and $|T(\omega)|^{2}$ is the measurable intensity transmission. Equation (16) provides an exact expression for the transmission within the reduced two-mode non-Hermitian model. In all numerical calculations we evaluate Eq. (16) without additional approximations.

In our implementation, we choose $\kappa$ and $\gamma$ such that the system is biased close to the EP while ensuring dynamical stability by enforcing $\operatorname{Im}\lambda_{\pm}-\kappa_{0}<0$ ; a representative choice is $\kappa/g_{0}=1$ , $\gamma/g_{0}\lesssim1$ , and $\kappa_{0}/g_{0}\gtrsim1$ . Temperature enters via $\Delta(T)$ , and we treat $\Delta T$ as the parameter to be inferred from noisy measurements of $|T(\omega)|^{2}$ . This defines a concrete sensing protocol that can be analyzed within the framework of estimation theory, in direct analogy to EP-based sensing strategies discussed in Refs. [6–8, 18].

In this work, we adopt a frequency-independent Gaussian noise model for the measured spectrum $|T(\omega)|^{2}$ , corresponding to the experimentally relevant regime where the dominant fluctuations arise from detector and electronic technical noise rather than photon shot noise. In such configurations the variance $\sigma^{2}$ remains approximately constant across the full scan window, which justifies the likelihood and Fisher–information expressions in Eqs. (19)-(21). Importantly, our sensitivity enhancement is therefore not claimed as a fundamental quantum-limit violation; the “no-go” theorems for EP-based metrology under fixed-photon-flux, shot-noise-limited conditions remain fully valid. Instead, the advantage demonstrated here reflects a practical enhancement that naturally emerges in realistic detector-noise-limited experiments, where steep dispersive features near an EP increase the slope of the transmittance curve without simultaneously amplifying the technical noise floor.

## V. SENSING RESPONSE AND CRB-BASED ENHANCEMENT

## A. Noise model, Fisher information, and Cramér–Rao bound

We model the sensing process as follows. The transmission spectrum is sampled at a discrete set of probe frequencies $\{\omega_{j}\}_{j=1}^{N}$ over a fixed bandwidth, and the measured intensities are

$$
y _ {j} = | T (\omega_ {j}; \Delta T) | ^ {2} + \xi_ {j},\tag{17}
$$

where $\xi_{j}$ represents additive measurement noise. Throughout, we assume independent, zero-mean Gaussian noise with variance $\sigma^{2}$ ,

$$
\xi_ {j} \sim \mathcal {N} (0, \sigma^ {2}),\tag{18}
$$

representing the combined effect of detector noise and technical fluctuations under a fixed input power and integration time. The same noise variance $\sigma^{2}$ and sampling grid $\{\omega_{j}\}$ are used for both the EP configuration and the Hermitian reference, ensuring that the comparison is carried out under identical resource constraints.

Let $\theta \equiv \Delta T$ denote the parameter to be estimated. The noise-free model predictions are $\mu_j(\theta) = |T(\omega_j; \theta)|^2$ , and the likelihood of the data $\mathbf{y} = (y_1, \ldots, y_N)$ is

$$
p (\mathbf {y} | \theta) \propto \exp \left[ - \frac {1}{2 \sigma^ {2}} \sum_ {j = 1} ^ {N} \left(y _ {j} - \mu_ {j} (\theta)\right) ^ {2} \right].\tag{19}
$$

The Fisher information for $\theta$ is then

$$
F (\theta) = \frac {1}{\sigma^ {2}} \sum_ {j = 1} ^ {N} \left(\frac {\partial \mu_ {j} (\theta)}{\partial \theta}\right) ^ {2},\tag{20}
$$

and the Cramér–Rao bound (CRB) states that the variance of any unbiased estimator $\hat{\theta}$ obeys

$$
\operatorname{Var} (\hat {\theta}) \geq \operatorname{CRB} _ {\theta} = \frac {1}{F (\theta)}.\tag{21}
$$

For the eigenvalue splitting $R(\theta) = |\lambda_{+}(\theta) - \lambda_{-}(\theta)|$ , the corresponding CRB is obtained by error propagation,

$$
\mathrm{CRB} _ {R} = \left(\frac {\partial R}{\partial \theta}\right) ^ {2} \mathrm{CRB} _ {\theta}.\tag{22}
$$

In the numerics, we choose N = 201 frequency samples uniformly spanning a window of width $4g_{0}$ around the resonance and a representative noise level $\sigma \sim 10^{-3}$ in normalized intensity units. These values are consistent with realistic photodetector noise levels for moderate optical powers and integration times. All reported CRBs are evaluated using Eqs. (20)-(22) and the exact transmission model (16). This CRB-based analysis directly addresses concerns that EP-enhanced eigenvalue susceptibilities do not automatically translate into improved signal-to-noise ratio or estimation precision [27-31].

![](images/ea8254d16ecd183398b48727b1ab72c12e73bc9454867aba6c6e8e44e0463d20.jpg)

![](images/05c835967b0ed2be0b39e535d4537d4d0504246a11f664325d30290203e68e9a.jpg)

![](images/36167e17a597a9ebec2d08e1724b02ea9016dd9a1a789659c80ce46bc3d0d3f1.jpg)

![](images/975c9c80b187d3d275627ed2dafb55b0cb452f471621aa025ed34b0c9a5bd390.jpg)

![](images/30b2a1274517d3c03b3c8c344a89872276027c87d44d29f0a536b1b0ca02c1de.jpg)

![](images/a08f9fd59419562e7769d6d38009691c828ba9a00e12d97b877e12b5aceb41ba.jpg)
FIG. 5. Exact spectral response and estimator validation. (a) Transmission spectra $|T(\omega)|^{2}$ for several temperature shifts $\Delta T$ , computed from the exact non-Hermitian scattering model in Eq. (16). (b) Eigenvalue splitting $R = |\lambda_{+} - \lambda_{-}|/g_{0}$ versus $\Delta T$ , showing excellent agreement between the true splitting (solid), the Cramér–Rao bound (CRB) prediction obtained from Eqs. (20)–(22), and Monte Carlo (MC) estimates from noisy spectra (error bars). (c1)–(c3) Representative residuals between noisy spectra and best-fit spectra at three values of $\Delta T$ , -3.0 K, 0.0 K, and +3.0 K. (d) Histogram of residuals at $\Delta T \approx 0$ , demonstrating Gaussian noise and negligible model mismatch.

## B. Spectral response, CRB, and Monte Carlo validation

We first examine how the transmission spectrum responds to temperature-induced detuning shifts. Figure 5 shows $|T(\omega)|^{2}$ for several values of $\Delta T$ around the operating point, together with the corresponding eigenvalue splitting and its estimation accuracy.

To validate the CRB analysis, we generate $10^{3}$ synthetic spectra at each $\Delta T$ by sampling the noise according to the Gaussian model and fit them using nonlinear least squares to the exact transmission model with $\Delta T$ as the only free parameter. The variances of the resulting estimates $\Delta T$ and $\hat{R}$ are found to agree with the CRBs within a few percent. The residuals are Gaussian-distributed and have subpercent magnitude, confirming that the model accurately captures the spectral response and that the estimator is effectively optimal (see also

![](images/ececa6d1eadfb12d35672998734ef191861df7f39418bdd834688dc4b436d3d1.jpg)

![](images/1a3194dddf4c508b495c958d7e03871455270d1e693d0734042174dc29def506.jpg)
FIG. 6. Cramér–Rao bound (CRB) based sensitivity enhancement. (a) Enhancement factor $\eta_{CRB} = CRB_{\Delta T,lin}/CRB_{\Delta T,EP}$ as a function of the gain-loss parameter $\gamma/g_{0}$ , showing $\eta_{CRB} > 1$ over a broad range. (b) Comparison of effective linewidths of the EP configuration and the matched Hermitian reference, demonstrating that the enhancement is not a trivial consequence of linewidth narrowing. Both systems are compared under identical noise variance, sampling grid, and input power.

Supplementary Fig. S4).

## C. CRB-based sensitivity enhancement at the EP

To quantify the enhancement provided by the temporal EP, we compare the CRB for temperature estimation at the EP bias to that of a Hermitian reference system with the same effective linewidth and operated under the same noise and sampling conditions. Specifically, we define

$$
\eta_ {\mathrm{CRB}} = \frac {\mathrm{CRB} _ {\Delta T , \mathrm{lin}}}{\mathrm{CRB} _ {\Delta T , \mathrm{EP}}},\tag{23}
$$

where $CRB_{\Delta T,EP}$ is computed at a near-EP operating point and $CRB_{\Delta T,\mathrm{lin}}$ is evaluated for a purely lossy (Hermitian) cavity whose linewidth is matched to that of the EP configuration. The Hermitian reference is described by a single-mode transmission function

$$
T _ {\mathrm{lin}} (\omega) = 1 - \frac {\kappa_ {\mathrm{in}}}{i (\omega - \omega_ {c}) + \kappa_ {\mathrm{tot}}},\tag{24}
$$

with total decay rate $\kappa_{tot}$ chosen such that the full width at half maximum of $|T_{\mathrm{lin}}(\omega)|^{2}$ coincides with that of the EP configuration at the operating point. The same set of probe frequencies $\{\omega_{j}\}$ and noise variance $\sigma^{2}$ is used to compute $CRB_{\Delta T,lin}$ via Eqs. (20) and (21).

Figure 6 summarizes the dependence of $\eta_{CRB}$ on $\gamma/g_{0}$ and confirms that $\eta_{CRB} > 1$ over a broad parameter range.

This comparison addresses a key concern in EP sensing, namely that enhanced sensitivity might arise solely from narrowing spectral features rather than from the non-Hermitian degeneracy itself. By matching the linewidths and enforcing identical noise and sampling conditions while still observing a significant CRB reduction, we demonstrate that the temporal EP provides a genuine sensitivity advantage within the specified noise model.

![](images/c3feb7a6a1c83255893a0b4ba3c5d98e0ebc8e85d746f2be5496247170cb732d.jpg)
FIG. 7. Square-root versus linear scaling of the eigenvalue splitting. (a) Splitting $R/g_{0}$ versus perturbation amplitude p on a logarithmic horizontal axis, comparing the EP response (circles) to a linear reference (squares). (b) log–log plot with fitted slopes, confirming $R_{EP} \propto p^{1/2}$ and $R_{lin} \propto p$ . The splitting R is computed directly from the exact eigenvalues in Eq. (7), without surrogate approximations.

## D. Square-root scaling versus linear response

Finally, we verify the characteristic square-root scaling of the eigenvalue splitting with respect to a small perturbation. Figure 7 compares the splitting $R(p)$ obtained from the exact eigenvalues at the EP to that of a linear reference. A log-log fit yields a slope of 0.5 for the EP and 1.0 for the linear reference, in agreement with the expected $R_{\mathrm{EP}} \propto p^{1/2}$ and $R_{\mathrm{lin}} \propto p$ behavior [6, 7]. Here, the parameter $p$ denotes a small physical perturbation applied to the detuning, i.e., $\Delta \to \Delta + p$ , while $\gamma$ and $\kappa$ are held fixed. This corresponds to the experimentally accessible case where a small change in temperature, index, or cavity dispersion induces a shift in the detuning term of the Hamiltonian. The eigenvalue splitting $R(p) = |\lambda_+(p) - \lambda_-(p)|$ therefore represents the response to this controlled detuning perturbation. Scaling exponents (linear vs. square-root) are extracted for small but finite $p$ , ensuring the system remains in the near-EP regime without evaluating $R$ exactly at the EP itself.

The observation of a clean square-root scaling using the exact eigenvalues, without surrogate approximations, is central to the validity of the EP-enhanced sensing mechanism.

## VI. DISCUSSION AND OUTLOOK

The analysis presented here establishes a self-consistent picture of temporal exceptional points in photonic time crystals and their application to optical sensing. By deriving and employing the exact eigenvalues of the non-Hermitian dimer Hamiltonian, and by using a transmission model that is exact within the reduced two-mode description, we avoid the inconsistencies that can arise when approximate eigenvalue formulas or surrogate lineshapes are used. The eigenvalue landscapes, Riemann surfaces, and Berry-phase calculations (Figs. 2–4) provide clear evidence of the EP topology, while the CRB-based sensitivity analysis and Monte Carlo validation (Figs. 5–7) demonstrate that EP-enhanced sensitivity is both physically meaningful and practically attainable under realistic noise levels [8, 18].

From a device perspective, thin-film lithium niobate on insulator (TFLN) and silicon nitride (SiN) photonic integrated circuits provide complementary platforms for realizing temporal EPs. In a TFLN ring or Mach–Zehnder geometry operating near $\lambda_{0} \approx 1550$ nm, electro-optic modulation at $\Omega/2\pi \sim 5-20$ GHz with index swings $\delta n \sim (1-3) \times 10^{-3}$ is routinely achievable, which corresponds to Floquet coupling rates on the order of $g_{0}/2\pi \sim 50-200$ MHz for practical device lengths [21]. Biasing the system near the temporal EP then requires balanced gain-loss parameters $\gamma/2\pi \lesssim g_{0}$ and total decay rates $\kappa_{0}/2\pi \sim 200-500$ MHz, consistent with reported quality factors in state-of-the-art TFLN resonators. In this regime, a thermo-optic coefficient of order $\alpha_{T}/2\pi \sim 5-20$ MHz/K yields sub-mK-level temperature resolution for the CRB values reported in Fig. 6. By contrast, SiN platforms naturally support low-loss, large-footprint time-crystal sections with thermo-optic modulation at $\Omega/2\pi \sim 1-100$ MHz and $\delta n \sim 10^{-4}$ , favoring slower but broadband operation in which the same temporal EP mechanism can be exploited with reduced RF complexity [21–23]. Together, these estimates indicate that the dimensionless EP physics developed here can

[1] R. El-Ganainy, K. G. Makris, M. Khajavikhan, Z. H. Musslimani, S. Rotter, and D. N. Christodoulides, Nat. Phys. 14, 11 (2018).

[2] L. Feng, R. El-Ganainy, and L. Ge, Nat. Photon. 11, 752 (2017).

[3] Ş. K. Özdemir, S. Rotter, F. Nori, and L. Yang, Nat. Mater. 18, 783 (2019).

[4] M.-A. Miri and A. Alù, Science 363, eaar7709 (2019).

[5] L. Xiao, K. Wang, D. Qu, H. Gao, Q. Lin, Z.-H. Bian, X. Zhan, and P. Xue, Photonics Insights 4, R09 (2025).

[6] J. Wiersig, Phys. Rev. Lett. 112, 203901 (2014).

[7] J. Wiersig, Phys. Rev. A 93, 033809 (2016).

[8] J. Wiersig, Photon. Res. 8, 1457 (2020).

[9] B. Peng, Ş. K. Özdemir, F. Lei, F. Monifi, M. Gianfreda, G. L. Long, S. Fan, F. Nori, C. M. Bender, and L. Yang, Nature Physics 10, 394 (2014).

[10] J. Schindler, A. Li, M. Zheng, F. M. Ellis, and T. Kottos, J. Phys. A: Math. Theor. 45, 444029 (2012).

[11] A. Laha and S. Ghosh, Opt. Lett. 41, 942 (2016).

[12] H. Xu, D. Mason, L. Jiang, and J. G. E. Harris, Nature 537, 80 (2016).

[13] W. Mao, Z. Fu, Y. Li, F. Li, and L. Yang, Science Advances 10, eadl5037 (2024).

[14] K.-H. Kim, M.-S. Hwang, H.-R. Kim, J.-H. Choi, Y.-S. No, and H.-G. Park, Nature Communications 7, 13893 (2016).

[15] B. Peng, Ş. K. Özdemir, S. Rotter, H. Yilmaz, M. Liertzer, F. Monifi, C. M. Bender, F. Nori, and L. Yang, Science 346, 328 (2014).

[16] H. Hodaei, A. U. Hassan, S. Wittek, H. Garcia-Gracia, R. El-Ganainy, D. N. Christodoulides, and M. Kha-

be mapped onto realistic integrated-photonics parameter ranges without requiring exotic material properties or unreasonably high modulation strengths.

Future work may explore experimental realizations using electro-optic modulation in TFLN or thermooptic modulation in SiN platforms, as well as extensions to multimode and multidimensional temporal crystals $[12, 25, 26]$ . Incorporating realistic material dispersion, nonlinearities, and technical noise sources into the model will be essential for optimizing performance in specific application scenarios. At a more fundamental level, the temporal EP framework developed here provides a starting point for investigating non-Hermitian Floquet phases, time-dependent symmetry breaking, and nonreciprocal phenomena in driven photonic systems $[23, 24]$ . It also offers a clean setting in which to further examine the interplay between EP-induced singular response and ultimate noise-limited sensitivity, a topic of ongoing interest in the broader EP-sensing literature $[27–31]$ .

## ACKNOWLEDGMENTS

The authors acknowledge helpful discussions with colleagues at IIT Delhi. S.M.T. acknowledges institutional support for computational resources.

javikhan, Nature 548, 187 (2017).

[17] W. Chen, Ş. K. Özdemir, G. Zhao, J. Wiersig, and L. Yang, Nature 548, 192 (2017).

[18] M. De Carlo et al., Sensors 22, 3977 (2022).

[19] R. Kononchuk, J. Cai, F. Ellis, R. Thevamaran, and T. Kottos, Nature 607, 697 (2022).

[20] B. Zhen, C. W. Hsu, Y. Igarashi, L. Lu, I. Kaminer, A. Pick, S.-L. Chua, J. D. Joannopoulos, and M. Soljačić, Nature 525, 354 (2015).

[21] J. S. Martínez-Romero, O. M. Becerra-Fuentes, and P. Halevi, Phys. Rev. A 93, 063813 (2016).

[22] K. Fang, Z. Yu, and S. Fan, Phys. Rev. Lett. 108, 153901 (2012).

[23] H. He, S. Zhang, J. Qi, F. Bo, and H. Li, Applied Physics Letters 122, 051703 (2023).

[24] T. T. Koutserimpas and R. Fleury, Phys. Rev. A 97, 013839 (2018).

[25] S. Longhi, J. Phys. A: Math. Theor. 50, 505201 (2017).

[26] T. Yoshida, R. Peters, and N. Kawakami, Phys. Rev. B 101, 085108 (2020).

[27] W. Langbein, Phys. Rev. A 98, 023805 (2018).

[28] H.-K. Lau and A. A. Clerk, Nat. Commun. 9, 4320 (2018).

[29] L. Bao, M. Zhang, and A. A. Clerk, Phys. Rev. A 103, 042418 (2021).

[30] R. Duggan, S. A. Mann, and A. Alù, ACS Photon. 9, 1554 (2022).

[31] H. Loughlin et al., Phys. Rev. Lett. 132, 243601 (2024).

## SUPPLEMENTARY INFORMATION

## S1. Derivation of the two-mode Floquet model and imaginary parts of the Riemann sheets

Here we briefly outline the derivation of the coupled-mode equations (4) from the full Maxwell equations and then present additional information on the imaginary parts of the eigenvalue surfaces.

We start from the time-domain wave equation

$$
\partial_ {z} ^ {2} E (z, t) - \mu_ {0} \partial_ {t} ^ {2} [ \epsilon (t) E (z, t) ] = 0.\tag{25}
$$

In the limit $\Omega<<\omega_{0}$ and small modulation depth m<<1, terms $\propto\dot{\epsilon}\dot{E}$ are negligible compared to $\epsilon\ddot{E}$ , so we adopt the following as our effective wave equation

$$
\partial_ {z} ^ {2} E (z, t) - \mu_ {0} \epsilon (t) \partial_ {t} ^ {2} E (z, t) = 0.\tag{26}
$$

With $\epsilon(t)$ given by Eq. (1), we insert the Floquet ansatz of Eq. (3) and collect terms oscillating at $e^{-i\omega_{0}t}$ and $e^{-i(\omega_{0}+\Omega)t}$ . Neglecting second-order time derivatives of the slow envelopes $a_{1,2}(t)$ and discarding non-resonant terms that couple to higher-order sidebands $e^{-i(\omega_{0}+n\Omega)t}$ with $|n|\geq2$ yields a pair of first-order differential equations for $a_{1,2}(t)$ . The effective coupling strength $\kappa$ is proportional to the modulation depth m and depends on the overlap between the unperturbed mode and the time-varying permittivity; explicit expressions can be obtained once a specific spatial mode profile is specified. The gain/loss rate $\gamma$ arises from a small imaginary component of the permittivity, which may itself be modulated in time to realize balanced gain and loss in the two Floquet components. The resulting equations are of the form of Eqs. (4a) and (4b), and the validity of the two-mode truncation requires $m\ll1$ and that the detunings to higher-order sidebands exceed the effective coupling strengths.

Figure S1 plots the imaginary parts of the eigenvalues $\lambda_{\pm}$ over the $(\Delta/g_{0},\gamma/g_{0})$ plane, complementing the real-part surfaces shown in Fig. 3. The white contour marks the locus Im $\lambda=0$ , separating decaying and amplifying regimes. The EP lies at the intersection of the two sheets along this boundary.

## S2. Berry phase versus loop radius

In Fig. S2, we examine how the magnitude of the biorthogonal Berry phase depends on the radius of the loop encircling the EP in the $(\Delta,\gamma)$ plane. For loops that do not enclose the EP, the accumulated phase is zero (mod $2\pi$ ). Once the loop radius exceeds a critical value $r_{c}$ such that the EP lies inside the loop, the Berry phase jumps to $\pi$ in magnitude and remains robust for larger radii. This step-like behavior confirms the topological nature of the EP.

![](images/be9153800bb99f6a5e2465a899eac63b63824cdee77643c5498e79f7ed6e0583.jpg)

![](images/5163d8021e8d4c4d8d331d4486059b354c6bb62ba4f0e83999707f8f7a78cd5a.jpg)
FIG. S1. Imaginary parts of the eigenvalue sheets $\lambda_{\pm}$ as functions of $(\Delta/g_{0},\gamma/g_{0})$ . The white contour indicates Im $\lambda=0$ , and the EP is located at $(\Delta,\gamma)=(0,\kappa)$ .

![](images/4ebdf788eeb7484052c59bc0d82b97e0987aa32a034f991c1ad8a37f980ef3cc.jpg)
FIG. S2. Magnitude of the biorthogonal Berry phase $|\phi_{B}|$ as a function of the loop radius $r/g_{0}$ in the $(\Delta,\gamma)$ plane. A transition from 0 to $\pi$ occurs when the loop begins to enclose the EP, demonstrating topological robustness.

## S3. Exceptional-point coordinates versus coupling strength

Figure S3 verifies the analytic EP condition in Eq. (9) by plotting the EP coordinates for several coupling strengths $\kappa$ . In each case, the EP is found at $(\Delta,\gamma)=(0,\pm\kappa)$ , in agreement with theory.

## S4. Cramér–Rao bound versus Monte Carlo reconstruction

Figure S4 compares the Cramér–Rao bound (CRB) for the temperature shift $\Delta T$ and the eigenvalue splitting R to Monte Carlo (MC) histograms obtained by fitting noisy transmission spectra to the exact non-Hermitian model. The operating point is chosen such that $\partial R/\partial T \neq 0$ , ensuring a finite CRB for R. In both cases, the MC variances agree with the CRB predictions to within a few percent, confirming estimator optimality and validating the Fisher-information analysis used in the main text.

![](images/a05de601572c3b26483112d1dbf55ba2bfaf9f583bd3c82de6a2b252155cbf59.jpg)
FIG. S3. Exceptional-point locations in the $(\Delta/g_{0},\gamma/g_{0})$ plane for several values of the coupling strength $\kappa/g_{0}$ . All EPs occur at $\Delta=0$ and $\gamma=\pm\kappa$ , confirming the analytic condition.

![](images/69fc840ffde112fdfafceeac99527e7b33ef2717176ef185f7e36b05f999294c.jpg)

![](images/ae45ac21d765c61e2b5a9f8dee9c44ce830a85ee620b004492c7ab34bc9ec11b.jpg)
FIG. S4. Comparison of Cramér–Rao bounds (CRB) with Monte Carlo (MC) estimator statistics at a nonzero bias point. Left: histogram of temperature estimates $\hat{\Delta T}$ with CRB indicated. Right: histogram of splitting estimates $\hat{R}/g_{0}$ , again in excellent agreement with the corresponding CRB.

## S5. Fit quality and splitting bias

Finally, Fig. S5 quantifies the goodness-of-fit and relative bias in the reconstructed splitting across a range of temperature shifts. The mean spectral coefficient of determination $R^{2}$ exceeds 0.995 for all $\Delta T$ in the range considered, and the relative error in R remains below $\sim3\%$ . These metrics confirm that the exact non-Hermitian transmission model provides an accurate and robust basis for parameter estimation in the proposed temporal EP sensing scheme.

![](images/8f4ec9e8493bd3cdc670c3fcc772a5a9b277a1c809f92da1085ce34a90db3810.jpg)
FIG. S5. Goodness-of-fit and splitting bias. Left axis: mean spectral $R^{2}$ for fits to noisy transmission spectra as a function of $\Delta T$ . Right axis: relative error in the reconstructed splitting R (in percent).
