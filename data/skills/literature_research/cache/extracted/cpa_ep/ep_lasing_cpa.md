# Exceptional Points, Lasing, and Coherent Perfect Absorption in Floquet Scattering Systems

David Globosits, $^{1,*}$ Puneet Garg, $^{2,*}$ Jakob Hüpfl, $^{1}$ Adrià Canós Valero, $^{3,4}$ Thomas Weiss, $^{3}$ Carsten Rockstuhl, $^{2,5}$ and Stefan Rotter $^{1}$

$^{1}$ Institute for Theoretical Physics, Vienna University of Technology (TU Wien), 1040 Vienna, Austria $^{2}$ Institute of Theoretical Solid State Physics, Karlsruhe Institute of Technology, 76131 Karlsruhe, Germany $^{3}$ Institute of Physics, University of Graz, and NAWI Graz, 8010 Graz, Austria $^{4}$ Riga Technical University, Institute of Telecommunications, 1048 Riga, Latvia $^{5}$ Institute of Nanotechnology, Karlsruhe Institute of Technology, 76131 Karlsruhe, Germany

Periodically time-varying media, known as photonic time crystals (PTCs), provide a promising platform for observing unconventional wave phenomena. We analyze the scattering of electromagnetic waves from spatially finite PTCs using the multispectral Floquet scattering matrix, which naturally incorporates the frequency-mixing processes intrinsic to such systems. For dispersionless, real, and time-periodic permittivities, this matrix is pseudounitary. Here we demonstrate that this property leads to multiple symmetry-breaking transitions: for increasing driving strength, scattering matrix eigenvalues lying on the unit circle (unbroken symmetry regime) meet at exceptional points (EPs), where they break up into inverse complex conjugate pairs (broken symmetry regime). We identify the symmetry operator associated with these transitions and show that, in time-symmetric systems, it corresponds to the time-reversal operator. Remarkably, at the parametric resonance condition, one eigenvalue vanishes while its partner diverges, signifying simultaneous coherent perfect absorption (CPA) and lasing. Since our approach relies solely on the Floquet scattering matrix, it is not restricted to a specific geometry but instead applies to any periodically time-varying scattering system. To illustrate this universality, we apply our method to a variety of periodically time-modulated structures, including slabs, spheres, and metasurfaces. In particular, we show that using quasi-bound states in the continuum resonances sustained by a metasurface, the CPA and lasing conditions can be attained for a minimal modulation strength of the permittivity. Our results pave the way for engineering time-modulated photonic systems with tailored scattering properties, opening new avenues for dynamic control of light in next-generation optical devices.

## I. INTRODUCTION

Electromagnetic waves inside dielectric media with a time-varying permittivity have recently been found to produce a host of unconventional wave phenomena $[1]$ , such as magnetic-free non-reciprocal energy transfer $[2]$ , enhanced photon pair generation $[3]$ , or exotic topological phases $[4]$ , among others. Recently, time-varying media were also employed for amplifying or attenuating $[5]$ incoming radiation. Especially useful in that regard are media with a periodic time variation of their properties, so-called Floquet media or photonic time crystals (PTCs) $[6, 7]$ . Several theoretical studies, in which the amplification of waves inside bulk PTCs has been discussed $[8, 9]$ , were recently accompanied by innovative experiments realizing extreme amplification of waves inside time-varying media $[10]$ . Complementary demonstrations of coherent absorption in such media have likewise been reported $[11]$ .

Capturing these developments calls for a unified theoretical framework that accounts for the distinctive features of time-varying media, including frequency conversion, amplification, and attenuation. In practice, realizations of PTCs are always spatially finite, which requires additional considerations beyond the bulk description. A comprehensive framework must therefore also incorporate the role of finite size and geometry, which govern processes such as boundary-induced scattering and resonance effects. These effects can profoundly modify the response of a finite PTC compared to its bulk counterpart $[12–16]$ .

In this work, we provide such a unifying description using the Floquet scattering matrix $[17]$ . This matrix comprehensively characterizes the periodically time-varying scattering process, expressing how incoming light fields are scattered into outgoing light fields $[18–23]$ . Crucially, energy and therefore frequency are not conserved quantities in a time-varying scattering setup, which induces frequency mixing that is correctly captured by the Floquet scattering matrix. Particularly important are those processes in which positive and negative frequencies are coupled $[24]$ . As was recently demonstrated $[25]$ , these processes are responsible for a special algebraic property of the Floquet scattering matrix: If the system is free of losses and dispersion, the Floquet scattering matrix is a pseudounitary matrix, expressing that the wave action, also known as the number of pseudophotons $[26]$ , is a conserved quantity.

Here, we demonstrate that the pseudounitarity of the Floquet scattering matrix has significant implications for the physics of periodically time-varying media. In particular, we find that the distribution of the eigenvalues and eigenstates reveals a close connection between non-Hermitian, PT-symmetric scattering systems and Floquet scattering systems. We show that for weak temporal modulations, all eigenvalues of the corresponding scattering matrix are unimodular. Quite remarkably, by increasing the modulation strength, the system can undergo multiple symmetry-breaking transition at exceptional points (EPs), where two eigenvalues coincide and the associated eigenvectors become parallel. We show that for time-symmetric modulations of the permittivity, the time-reversal symmetry of the eigenstates is broken. By increasing the modulation strength even further, eigenvalues leave the unit circle in a pairwise fashion.

At the parametric resonance condition, one eigenvalue eventually can become zero while its partner eigenvalue simultaneously diverges to infinity. Such an operational condition marks the point where the system can act as both a coherent perfect absorber (CPA) and a laser. The symmetry-breaking transition at an EP and the fact that the system can operate as a CPA-laser are strongly reminiscent of what has been found for PT-symmetric scattering systems $[27, 28]$ . Notably, for the Floquet scattering systems considered here, the wave fields that get emitted by the Floquet laser or that get perfectly absorbed by the Floquet CPA are given by the associated eigenstates of the Floquet scattering matrix $S_{F}$ . Since $S_{F}$ incorporates the scattering behavior of multiple frequencies, the eigenstates and, therefore, the lasing and CPA states are multi-spectral and, thus, in general, pulsed light fields.

Recently, several studies have revealed important insights into the role of symmetry breaking at EPs and their connection to extreme energy transfer in unbounded (bulk) time-modulated media $[29–32]$ . However, to accurately describe phenomena such as CPA and lasing, a framework is required that accounts for the finite size of any practical realization of a time-varying medium. In particular, the presence of spatial boundaries leads to scattering and interference effects that are absent in unbounded systems. As our framework is based on the Floquet scattering matrix, it not only captures the unique features of periodically time-varying systems but also naturally incorporates the effects of a finite system size. With our approach being very general, it applies to a wide range of time-modulated scattering systems, given that a corresponding Floquet scattering matrix can capture their response. It incorporates an arbitrary number of Floquet channels and holds true for any frequency of the incident light field.

Our manuscript is structured as follows: We first discuss our framework based on a finite slab with oscillating permittivity. Then, we apply it to spatially localized scatterers and finally to a periodic arrangement of scatterers in two dimensions, i.e., a metasurface. With that, we show that all the considered Floquet scattering systems share the same underlying physics.

## II. RESULTS

## A. The Floquet Scattering Matrix $S_{F}$

For the sake of simplicity, we first consider linearly polarized light at normal incidence scattering off a dielectric slab that is infinite in $y$ - and $z$ -direction and has finite thickness $L$ in $x$ -direction [see Fig. 1(a)]. Furthermore, we assume that the slab's material responds instantaneously to the external field. The scattering system can then be described with a real, time-periodic dielectric function $\epsilon(\mathbf{r}, t) = \epsilon(\mathbf{r}, t + T)$ , where $T$ is the oscillation period and $\Omega = 2\pi/T$ is the associated angular frequency. Specifically, we choose a time-harmonic modulation of the permittivity [see Fig. 1(b)]. We further assume that the slab is surrounded by free space. This scattering system constitutes an effective one-dimensional Floquet scattering problem governed by the scalar wave equation for the complex-valued electric field $E(x, t)$ as

$$
\partial_ {x} ^ {2} E (x, t) - \frac {1}{c ^ {2}} \partial_ {t} ^ {2} [ \epsilon (x, t) E (x, t) ] = 0,\tag{1}
$$

where c is the speed of light in free space, and the relative permittivity is given by

$$
\epsilon (x, t) = 1 + \chi_ {0} [ 1 - M _ {\mathrm{s}} \cos (\Omega t) ] \Theta (L / 2 - | x |).\tag{2}
$$

Here, $\Theta$ is the Heaviside step function, L is the thickness of the scatterer, $1 + \chi_{0}$ is the static permittivity of the slab, and $M_{s} \geq 0$ is the modulation strength (modulation amplitude). If not stated otherwise, we assume for the parameters of the slab $\chi_{0} = 4$ , $\Omega = 2\pi \times 151$ THz, and L = 1581 nm.

The wave fields in the asymptotic regions left $(x < -L/2, \sigma = 1)$ and right $(x > L/2, \sigma = r)$ of the slab are assumed to be superpositions of plane waves with frequencies $\omega_{n} = \omega + n\Omega$ with $n \in Z$ . Here, $\omega$ is the quasifrequency (Floquet frequency) which we choose to lie in the first temporal Brillouin zone $0 \leq \omega < \Omega$ . The incoming and outgoing wave fields expressed in a photon-flux normalized basis read [25]

$$
E ^ {\mathrm{in}} (x, t) = \sum_ {\sigma , n} \sqrt {\hbar \mu_ {0} c | \omega_ {n} |} c _ {\sigma , n} ^ {\mathrm{in}} e ^ {i k _ {\sigma , n} (x - L / 2)} e ^ {- i \omega_ {n} t},
$$

$$
E ^ {\mathrm{out}} (x, t) = \sum_ {\sigma , n} \sqrt {\hbar \mu_ {0} c | \omega_ {n} |} c _ {\sigma , n} ^ {\mathrm{out}} e ^ {- i k _ {\sigma , n} (x - L / 2)} e ^ {- i \omega_ {n} t}.\tag{3a}
$$

(3b)

In the asymptotic regions, a linear dispersion relation holds such that $ck_{1,n} = -ck_{r,n} = \omega_{n}$ . Furthermore, we introduce the amplitudes $c_{\sigma,n}^{in}$ and $c_{\sigma,n}^{out}$ for incoming and outgoing plane waves at the discrete frequencies $\omega_{n} = \omega + n\Omega$ . The Floquet scattering matrix connects the incoming with the outgoing wave fields as

$$
\left| c ^ {\mathrm{out}} \right\rangle = S _ {\mathrm{F}} \left| c ^ {\mathrm{in}} \right\rangle .\tag{4}
$$

a
![](images/2a070a35a39bfc9687f29164d1a9936383a0e0d35a9731eddd7543ba20b237df.jpg)

C
![](images/58e0944ec493d9e3c5bb4d98d60635da3dffaaecba59330fd14797eb2da07d33.jpg)

d
e
![](images/49c9972f2ca7d8ba1823a34d9433dbadcab9a9f09ba2e068675ec736ad20c00b.jpg)

![](images/7caa41b5e21f2c1e242a40fd5e64ebcce1ec98a77dc097c43b96a37de8dc41dc.jpg)

b
$\epsilon (t) = \epsilon (t + T)$
f
![](images/d2bc4209c854a5f89a1696129c1c9a6e2c6a99f8670f59885a1552cba2bc8e9d.jpg)
FIG. 1. Behavior of the eigenvalues $\lambda$ of the Floquet scattering matrix $S_{\mathrm{F}}$ for a time-varying slab (see text for parameter values). a, We consider light scattering off a slab with a time-periodic permittivity with period $T = 2\pi /\Omega$ . The frequency of the incident light field can change during scattering such that the output light consists of several frequency components $\omega +n\Omega$ . b, We assume a time-harmonic permittivity of the slab. c-e, We track four eigenvalues of $S_{\mathrm{F}}$ as a function of the modulation strength $M_{\mathrm{s}}$ for different choices of the quasifrequency $\omega$ . In all cases, for small $M_{\mathrm{s}}$ all eigenvalues are unimodular and thus lie on the unit circle (blue). At a critical modulation strength, EPs are formed, where two respective eigenvalues coincide. By further increasing $M_{\mathrm{s}}$ , these two eigenvalues leave the unit circle in a pairwise fashion (red). The colorbar to the right of panel e applies to all panels c-e, schematically indicating the respective EPs by a blue-to-red transition (the actual $M_{\mathrm{s}}$ value at which the EPs occur differs between panels). d, A special situation occurs when the quasifrequency of the wave field obeys the parametric resonance condition, $\omega /\Omega = 0.5$ . Then, one eigenvalue vanishes (represented by a star symbol) while another eigenvalue diverges (not shown), at which point the Floquet system acts as both a CPA and a laser. f, The absolute value of the minimal eigenvalue $\lambda_{\mathrm{min}}$ of $S_{\mathrm{F}}$ is shown here as a function of the modulation strength $M_{\mathrm{s}}$ and the quasifrequency $\omega$ . For small $M_{\mathrm{s}}$ (weak modulation) or if the quasifrequency is far detuned from the parametric resonance condition, the minimal eigenvalues, and therefore all eigenvalues, are unimodular (white). This unbroken regime is separated by an exceptional line (black) from the broken regime with $|\lambda_{\mathrm{min}}| < 1$ (yellow). When calculating the winding number along the path $\gamma_{1}$ , we find wind $(S_{\mathrm{F}}) = 2$ , verifying that this path encloses a CPA-lasing point (star symbol). For reference, we also calculate the winding number for the path $\gamma_{2}$ , which does not enclose a CPA-lasing point, and find wind $(S_{\mathrm{F}}) = 0$ .

Here, we arrange the amplitudes into vectors as $|c^{in}\rangle = (\ldots, c_{1,-1}^{in}, c_{1,0}^{in}, \ldots, c_{r,-1}^{in}, c_{r,0}^{in}, \ldots)^{\mathrm{T}}$ and analogously for the outgoing state $|c^{out}\rangle$ . We refer the reader to [18, 25] for details on how to numerically obtain the respective Floquet scattering matrix for the system at hand. For computational reasons, we truncate the system to $n \in [-N, N-1]$ Floquet channels. Specifically, for the slab system, we choose N = 8. When the Floquet scattering matrix is expressed in a photon-normalized basis, it obeys a pseudounitary relation of the form [25]

$$
S _ {\mathrm{F}} ^ {\dagger} V S _ {\mathrm{F}} = V,
$$

where

(5)

$$
V = \left( \begin{array}{c c c c} - \mathbb {1} & \mathbb {0} & \mathbb {0} & \mathbb {0} \\ \mathbb {0} & \mathbb {1} & \mathbb {0} & \mathbb {0} \\ \mathbb {0} & \mathbb {0} & - \mathbb {1} & \mathbb {0} \\ \mathbb {0} & \mathbb {0} & \mathbb {0} & \mathbb {1} \end{array} \right).\tag{6}
$$

The appearing matrices $(\pm1$ and 0) are of appropriate size to ensure that the matrix V assigns a minus sign to negative frequency channels $(n < 0)$ via the matrices -1 and a plus sign to positive frequency channels $(n \geq 0)$ via the matrices 1 on each side of the slab. Specifically, the upper left (lower right) matrix quadrant corresponds to the left (right) port.

## B. Exceptional Points and the CPA-Lasing Threshold

In this work, we demonstrate that the pseudounitary property of $S_{F}$ has profound consequences for the physics of Floquet scattering systems. In particular, we show that effects like EPs, lasing, and coherent perfect absorption, which are impossible to obtain with wave fields associated with real frequencies in energy-conserving, time-invariant systems, can emerge in Floquet scattering systems. Previously, these effects were observed in non-Hermitian systems that break energy conservation with static gain and loss elements, rendering the system non-unitary, or using complex frequency input waves [33, 34]. Here, we reveal that such phenomena also emerge naturally in spatially finite time-varying systems.

In energy-conserving systems, the scattering matrix is unitary when expressed in an energy-flux normalized basis, and thus all eigenvalues $\lambda$ are unimodular $|\lambda_{n}| = 1$ . This property is altered for pseudounitary Floquet scattering matrices, which we can understand by the following observation [35]: Let $|c^{in}\rangle$ be an eigenstate of $S_{F}$ with corresponding eigenvalue $\lambda$ . Then, from Eq. (5), we see that $V|c^{in}\rangle$ is an eigenstate of $S_{F}^{\dagger}$ with eigenvalue $1/\lambda$ as

$$
S _ {\mathrm{F}} ^ {\dagger} V \left| c ^ {\mathrm{in}} \right\rangle = V S _ {\mathrm{F}} ^ {- 1} \left| c ^ {\mathrm{in}} \right\rangle = \frac {1}{\lambda} V \left| c ^ {\mathrm{in}} \right\rangle .\tag{7}
$$

Since the eigenvalues of $S_{F}^{\dagger}$ are the complex conjugates of the eigenvalues of $S_{F}$ , we arrive at the result that both $\lambda$ and $1/\lambda^{*}$ are eigenvalues of $S_{F}$ . Importantly, we conclude that two different regimes are possible: First, eigenvalues may be unimodular, i.e., $|\lambda| = 1$ , such that $\lambda = 1/\lambda^{*}$ . When such a condition is fulfilled, the corresponding eigenstate is in the so-called unbroken regime. By contrast, in the broken regime, eigenvalues are not unimodular ( $|\lambda| \neq 1$ ), but instead come in inverse complex conjugate pairs ( $\lambda, 1/\lambda^{*}$ ). These two regimes are separated by a spontaneous symmetry-breaking transition occurring at an exceptional point where two eigenvalues coalesce and the respective eigenstates of $S_{F}$ become parallel (we numerically checked that the eigenstates indeed become parallel at an EP for all scattering setups [36]).

Notably, this algebraic property of the eigenvalues of a scattering matrix can also be found for PT-symmetric scattering systems $[37]$ . There, static gain and loss elements are arranged in a spatially symmetric configuration, such that the scattering landscape is invariant under the combined action of the parity operator and the time-reversal operator. The eigenvalues of the corresponding scattering matrix are also either unimodular (unbroken PT-symmetry regime) or come in inverse complex conjugate pairs (broken PT-symmetry regime). Similar to the Floquet case, both regimes are separated by an exceptional point indicating the PT-symmetry-breaking transition. Remarkably, a special situation may also occur in which one eigenvalue vanishes while at the same time one eigenvalue diverges. This marks the situation at which the system can simultaneously act as a CPA (vanishing eigenvalue) and as a laser (diverging eigenvalue) [27, 28, 38, 39].

In the following, we show that similar unconventional wave phenomena appear in Floquet scattering systems, which can be understood based on the eigenvalues of the associated Floquet scattering matrix. To demonstrate this, we track the behavior of the eigenvalues of $S_{F}$ as a function of the modulation strength $M_{s}$ . In Fig. 1(c)-(e), we depict those four eigenvalues out of all the eigenvalues of $S_{F}$ that undergo a symmetry-breaking transition with increasing $M_{s}$ (blue-red colorscale) for the chosen set of parameters for three different choices of the quasifrequency $\omega$ . In all three cases, we observe that if the modulation is weak, the eigenvalues are located on the unit circle (represented by blue color). In fact, all the eigenvalues of $S_{F}$ are unimodular in this regime (not shown).

With increasing $M_{s}$ , the eigenvalues start to shift while remaining on the unit circle (blue colorscale). Specifically, we observe in Fig. 1(c)-(e) that two eigenvalues with positive real parts and two other eigenvalues with negative real parts approach each other, respectively. Once a critical modulation strength is reached, the nearby eigenvalues coalesce, marking the formation of two EPs. We notice that these two EPs form at the same modulation strength $M_{s}$ . By increasing $M_{s}$ even further, the four depicted eigenvalues leave the unit circle and form two pairs, each of which contains partners that are inverse complex conjugate to each other, $(\lambda,1/\lambda^{*})$ (red colorscale). This marks the situation where the associated eigenstates enter the broken regime.

A remarkable situation occurs when the quasifrequency meets the parametric resonance condition, $\omega/\Omega = 1/2$ , as depicted in Fig. 1(d). There, one eigenvalue in the broken regime vanishes while its partner eigenvalue diverges. At this particular driving strength, the slab behaves as a CPA ( $\lambda = 0$ ) and simultaneously as a laser ( $\lambda = \infty$ ).

For comparison, we also show the eigenvalue structure for the off-resonant case $\omega/\Omega < 1/2$ [see Fig. 1 (c)] and $\omega/\Omega > 1/2$ [see Fig. 1 (e)], where an eigenvalue becomes small (large) but does not reach zero (infinity), which highlights the importance of the parametric resonance condition not only for bulk but also for spatially finite PTCs to reach the condition for CPA or lasing, respectively. We note that all the other eigenvalues that are not depicted in Fig. 1(c)-(e) stay on the unit circle throughout the whole interval of modulation strength $M_{s}$ for the chosen set of parameters. In general, this means that some eigenstates of $S_{F}$ can be in the broken phase while other eigenstates for the same set of parameters are in the unbroken phase. It would be interesting to explore special situations, such as when multiple EPs form at the same position in parameter space using, for instance, tools from Krein stability analysis [40].

To provide further insights on the influence of the quasifrequency $\omega$ and the modulation strength $M_{s}$ for the formation of EPs and the CPA-lasing threshold, we plot the absolute value of the smallest eigenvalue of $S_{F}$ in Fig. 1(f). We observe that for a weak driving strength and also for light fields that are far detuned from the parametric resonance condition $\omega/\Omega = 1/2$ , the smallest eigenvalue (and therefore all eigenvalues) are unimodular (white color). On the contrary, if the quasifrequency is close to the parametric resonance condition, by increasing $M_{s}$ , the eigenvalues can reach an exceptional point (black line). Figure 1(f) suggests that a finite modulation strength is necessary to observe an EP. However, by properly adjusting the thickness of the slab, an EP can form for arbitrarily small modulation strengths – a result that may be very relevant for the experimental observation of EPs in spatially finite Floquet media. More specifically, using a two-band model approximation under the assumption of a weak modulation strength and setting $\omega/\Omega = 1/2$ , we derive an approximate expression for the minimal modulation strength necessary to observe an exceptional point $M_{s,EP}^{min}$ . We find that

$$
M _ {\mathrm{s,EP}} ^ {\mathrm{min}} \propto \sqrt {1 - \cos (L _ {\mathrm{n}})},\tag{8}
$$

where we introduced the normalized thickness $L_{n} = L\Omega\sqrt{1 + \epsilon_{s}}/c$ . Interestingly, Eq. (8) tells us that $M_{s,EP}^{min}$ vanishes for $L_{n} = 2\pi m$ with $m \in N$ , which exactly corresponds to the resonances of the static slab. Hence, the resonant response is crucial for reducing the modulation strength required to reach the EP. We verify the above observation using the numerical data of our full-scale simulation, which takes more than two frequency components into account (we use $n \in [-8, 7]$ ) and does not rely on the assumption of a weak driving strength [25]. Especially around $L_{n} = 2\pi m$ , we find excellent agreement between the results derived from the linearized two-band model approximation and the full-scale simulation as the assumption of weak modulation is well satisfied in these cases. For details on the derivation of Eq. (8) including the full expression of $M_{s,EP}^{min}$ and a comparison of the approximate result to the non-perturbative data, we refer the reader to Subsec. IV C1.

Figure 1(f) further shows that by increasing $M_{\mathrm{s}}$ and thus going beyond the EP, the smallest eigenvalue enters the broken regime obeying $|\lambda_{\mathrm{min}}| < 1$ (yellow colorscale). In the specific case when $\omega / \Omega = 1/2$ , the smallest eigenvalue becomes zero (marked by the star symbol), thus reaching the CPA condition. By increasing $M_{\mathrm{s}}$ even further, the magnitude of the minimal eigenvalue increases again. We note that in the regime beyond the CPA-lasing condition, non-linear effects can be expected to arise, which are not described by the linear model we assume here [41, 42]. In Subsec. IV C 2 we provide an analysis of the minimal driving strength necessary for the system to operate as a CPA or as a laser, respectively.

The eigenvector corresponding to the vanishing eigenvalue is a light field that, when injected into the scattering system, gets perfectly absorbed inside the time-varying medium such that no outgoing (reflected or transmitted) wave is produced. In Fig. 2(a)-(c), we demonstrate how the CPA-state manages to be perfectly absorbed by plotting the intensities of the electric field at the left $(x = -L/2)$ and right $(x = L/2)$ interfaces of the slab. Since no evanescent modes are excited in this one-dimensional slab system, the wave field at the border of the slab can already be considered as the far-field.

![](images/69848c4984ccbc90dfa720e66603e86f48723bf59e432216084da82086068a73.jpg)

![](images/5d678d2a306165b98e112096ff2940c49a2511bd066e0c60e8fe36b80189e7a8.jpg)
FIG. 2. Intensities of the incoming and the outgoing CPA and lasing states at the borders of the slab (see text for parameter values). a and d, Temporal variation of the periodic permittivity function of the slab. b, Intensity of the incoming part of the CPA light field at the left border at x = -L/2 (solid red) and at the right border at x = L/2 (dashed black) of the slab. The light field approaches the slab from the left and right with the same temporal intensity profile (the two lines overlap). Destructive interference of these pulses eliminates spatial reflections. The wave field builds up a large intensity maximum at times, when the permittivity is rising $0 \leq t \leq T/4$ (gray shaded region). In this way, all the energy of the light field gets perfectly absorbed by the time-varying medium. c, The corresponding output intensity at the borders of the slab is reduced by a factor $10^{-10}$ . e, Same as b but for the lasing state. This light field builds up large intensity maxima at times, when the permittivity is lowered $-T/4 \leq t \leq 0$ (gray shaded region), receiving energy from the time-modulated slab. f, This leads to a huge amplification of the outgoing intensity by a factor of $10^{10}$ . Furthermore, we observe the time-reversal symmetry of the CPA and lasing states: The incoming CPA state is the time-reversed of the outgoing lasing state.

To be perfectly absorbed, the input field depicted in Fig. 2(b) is optimally shaped in both its spatial and temporal degrees of freedom. The spatial degrees of freedom are adjusted so that all reflections off the spatial interfaces of the slab are eliminated through destructive interference. Such an operation is accomplished by simultaneously approaching the scatterer from left and right, which we observe in Fig. 2(b) by noting that the intensity distributions at the left and right interfaces are identical. On the other hand, the temporal degrees of freedom are fine-tuned in such a way that only minimal intensity is built up at the interfaces during the first half of the period $-T/2 \leq t \leq 0$ . However, precisely during the duration when the permittivity starts to rise $0 < t \leq T/4$ [gray region, see also Fig. 2(a)], this light field hits the scatterer and enters the time-varying slab corresponding to the intensity maximum during this time frame in Fig. 2(b). A large intensity build-up occurs inside the Floquet slab. In this way, the energy of the light field gets completely absorbed by the time-varying medium, and nearly no outgoing wave is produced, as shown in Fig. 2(c).

Conversely, we plot the incoming temporal intensity distribution at the interfaces of the slab for the lasing state in Fig. 2(e). This state exhibits a strong intensity maximum at times when the permittivity is lowered during $-T/4 \leq t \leq 0$ [gray region, see also Fig. 2(d)]. In this way, the light field receives energy from the time-varying medium and thus gets amplified [see Fig. 2(f)]. We note that to avoid numerical instabilities in the close proximity of divergences, we plot an eigenstate corresponding to a small but finite eigenvalue $|\lambda| \approx 10^{-5}$ in Fig. 2(b)-(c). Correspondingly, the lasing state depicted in Fig. 2(e)-(f) corresponds to a large but finite eigenvalue of $|\lambda| \approx 10^5$ . Since we plot the intensity of the wave field, which is the square of the electric field, the scaling factor in Fig. 2(c) is about $10^{-10}$ and in Fig. 2(f) about $10^{10}$ .

To prove unambiguously that in the vicinity of this small but finite eigenvalue there truly exists a vanishing eigenvalue together with a diverging eigenvalue, we calculate the associated winding number wind $S_{\mathrm{F}}$ along two loops $\gamma_{1,2}$ parametrized by $0 \leq \phi < 2\pi$ in parameter space [see Fig. 1(e)]. The winding number vanishes if the loop $\gamma$ does not enclose a zero or diverging eigenvalue. If, however, the loop $\gamma$ encloses a CPA-lasing point, we expect a winding number of $\pm 2$ [43]. While originally introduced for single-frequency scattering matrices [44], we here extend the concept of the winding number to multispectral Floquet scattering matrices. Using the pseudounitarity of $S_{\mathrm{F}}$ , the winding number takes the form

$$
\mathrm{wind} (S _ {\mathrm{F}}) = \frac {1}{2 \pi i} \int_ {0} ^ {2 \pi} \mathrm{d} \phi   \mathrm{Tr} \bigg \{V S _ {\mathrm{F}} ^ {\dagger} V \frac {\mathrm{d} S _ {\mathrm{F}}}{\mathrm{d} \phi} \bigg \}.\tag{9}
$$

We indeed numerically confirm that $\mathrm{wind}(S_{\mathrm{F}}) \approx 2 + 4.4 \times 10^{-5}$ along $\gamma_{1}$ , verifying the existence of both a zero and a pole of $S_{F}$ . For comparison, we also calculate the winding number for a loop $\gamma_{2}$ that does not encircle a zero and pole, where we find $\mathrm{wind}(S_{\mathrm{F}}) \approx 0 + 1.9 \times 10^{-5}$ , as expected.

In this subsection, we showed how the eigenvalues of the pseudounitary Floquet scattering matrix behave as a function of the driving strength $M_{s}$ and the quasifrequency $\omega$ . We revealed the appearance of EPs and identified the special cases of CPA and lasing. Notably, since the Floquet scattering matrix is pseudounitary for any real, dispersionless, and time-periodic permittivity function, our framework holds true not only for time-harmonic driving protocols, but for arbitrary periodic modulations. In analogy to static, PT-symmetric systems, where an EP marks the breaking of the PT symmetry of the corresponding eigenstates, we also expect in the time-varying Floquet case a corresponding symmetry-breaking transition to occur at an EP. In the following, we address the question of which symmetry can be spontaneously broken in a Floquet scattering system and which consequences this entails.

## C. The Symmetry-breaking Transition

Spontaneous symmetry-breaking, in our case, means that the eigenstates of $S_{F}$ in the unbroken regime individually possess a symmetry. In contrast, in the broken regime, this symmetry maps one eigenstate onto its partner and vice versa. As we show below, for a time-symmetric driving protocol $\epsilon(t) = \epsilon(-t)$ , the associated symmetry that is broken at an EP is the time-reversal symmetry.

To make this argument more transparent, we introduce the following notation from Ref. 45 to distinguish between states in the unbroken and the broken scattering regime: We label an eigenstate corresponding to a unimodular eigenvalue as $|c_{\nu_0}^{\mathrm{in}} \rangle$ with $|\lambda_{\nu_0}| = 1$ and the two eigenstates corresponding to a distinct pair of non-unimodular eigenvalues as $|c_{\nu_\pm}^{\mathrm{in}} \rangle$ with $|\lambda_{\nu_+}| > 1$ and $|\lambda_{\nu_-}| < 1$ , respectively. Furthermore, we assume that the eigenstates are non-degenerate (we consider the case of degeneracies in Subsec. IV B). For every diagonalizable pseudounitary matrix, we can construct an anti-linear symmetry operator $X$ that has the above-described property [45]

$$
X \left| c _ {\nu} ^ {\mathrm{in}} \right\rangle = \left\{ \begin{array}{l l} \left| c _ {\nu_ {0}} ^ {\mathrm{in}} \right\rangle , & \quad \nu = \nu_ {0}, \\ \left| c _ {\nu_ {\pm}} ^ {\mathrm{in}} \right\rangle , & \quad \nu = \nu_ {\mp}. \end{array} \right.\tag{10}
$$

Notably, this anti-linear operator X can be constructed for any diagonalizable pseudounitary scattering matrix $S_{F}$ and thus even for time-periodic driving schemes $\epsilon(t)$ that do not possess any additional temporal symmetry beyond periodicity. We refer the reader to Subsec. IV B for the treatment of this general case, including also a discussion on how to deal with degeneracies of $S_{F}$ . Furthermore, our formalism is fully general as it holds for an arbitrary choice of the quasifrequency $\omega$ and for an arbitrary number of Floquet channels considered in the corresponding wave field. In this way, we provide a complete and comprehensive description of spontaneous symmetry breaking in Floquet scattering systems, including CPA and lasing. The results of Ref. 46, obtained for the restricted case of two Floquet channels at a fixed quasifrequency $\omega/\Omega = 0.5$ , thus appear as one particular instance of our general framework.

The operator X has a particularly straightforward interpretation for time-reversal symmetric driving protocols $\epsilon(t) = \epsilon(-t)$ , which we will assume in the following. In this case, the symmetry represented by the operator X is the time-reversal symmetry, and the associated time-reversal operator is the complex conjugation operator X = K, such that when acting on an arbitrary input state $|c^{in}\rangle$ we have $K|c^{in}\rangle = |c^{in}\rangle^{*}$ . To explicitly show that K is the correct symmetry operator, we first note that for time-symmetric driving protocols the Floquet scattering matrix can be chosen to satisfy a generalized reciprocity relation $S_{F} = VS_{F}^{T}V$ or equivalently $S_{F}^{*} = VS_{F}^{\dagger}V$ (see Subsec. IV A for a derivation of this result). Together with the pseudounitarity condition [Eq. (5)], we thus have

$$
S _ {\mathrm{F}} ^ {- 1} = S _ {\mathrm{F}} ^ {*} \equiv K S _ {\mathrm{F}} K.\tag{11}
$$

Using the above equation, we can see that the operator K has the desired properties of a symmetry operator [Eq. (10)] by the following argument: If $|c_{\nu}^{in}\rangle$ is an eigenvector of $S_{F}$ to an eigenvalue $\lambda_{\nu}$ then $K|c_{\nu}^{in}\rangle$ is an eigenvector to the eigenvalue $1/\lambda_{\nu}^{*}$ as

$$
S _ {\mathrm{F}} K \left| c _ {\nu} ^ {\mathrm{in}} \right\rangle = K \left(S _ {\mathrm{F}} ^ {- 1} \left| c _ {\nu} ^ {\mathrm{in}} \right\rangle\right) = \frac {1}{\lambda_ {\nu} ^ {*}} K \left| c _ {\nu} ^ {\mathrm{in}} \right\rangle .\tag{12}
$$

Note that in the unbroken regime where $|\lambda_{\nu_{0}}| = 1$ , we have $1/\lambda_{\nu_{0}}^{*} = \lambda_{\nu_{0}}$ . The above shows that states in the unbroken regime are time-reversal symmetric individually. On the contrary, an eigenstate $|c_{\nu_{+}}^{in}\rangle$ in the broken regime is the time-reversed of the corresponding partner state $|c_{\nu_{-}}^{in}\rangle$ . In particular, this means that the CPA input state ( $\lambda = 0$ ) in Fig. 2(b) is the time-reversed of the output lasing state ( $\lambda = \infty$ ) depicted in Fig. 2(f), respectively.

It may seem counterintuitive that, on the one hand, a periodically time-varying medium can reduce or enhance the intensity of the outgoing light field, including the extreme cases of CPA and lasing. On the other hand, however, the number of pseudophotons flowing into the scattering system equals the number of pseudophotons flowing out of the system for an arbitrary input state $\left|c^{in}\right\rangle$ [26] as expressed by the pseudounitarity of the Floquet scattering matrix [see Eq. (5)] [25]. In the following, we show that there is no contradiction between these two observations. Importantly, when counting the number of pseudophotons, negative-frequency components have to be weighted with an additional minus sign via the matrix V from Eq. (6). For example, the number of pseudophotons of an arbitrary photon-normalized input light field reads $\langle c^{in}|V|c^{in}\rangle = \sum_{n} \text{sign}(\omega_n)|c_n^{in}|^2$ . Contrary, for the same state, the number of photons is given by $\langle c^{in}|c^{in}\rangle = \sum_{n} |c_n^{in}|^2$ . Distinguishing between these two quantities is essential for understanding what happens at EPs and at the CPA and lasing condition that are induced by time modulations.

To discuss this in detail, let us first consider eigenstates of $S_{F}$ in the broken symmetry regime $|c_{\nu_{\pm}}^{in}\rangle$ with eigenvalues $|\lambda_{\nu_{\pm}}| \neq 1$ . Here, the number of photons is not conserved but rather reduced (for $|\lambda_{\nu_{-}}| < 1$ ) or enhanced (for $|\lambda_{\nu_{+}}| > 1$ ) during the scattering process as

$$
\langle c _ {\nu_ {\pm}} ^ {\mathrm{out}} | c _ {\nu_ {\pm}} ^ {\mathrm{out}} \rangle = \langle c _ {\nu_ {\pm}} ^ {\mathrm{in}} | S _ {\mathrm{F}} ^ {\dagger} S _ {\mathrm{F}} | c _ {\nu_ {\pm}} ^ {\mathrm{in}} \rangle = \left| \lambda_ {\nu_ {\pm}} \right| ^ {2} \langle c _ {\nu_ {\pm}} ^ {\mathrm{in}} | c _ {\nu_ {\pm}} ^ {\mathrm{in}} \rangle .\tag{13}
$$

The above reasoning also includes the extreme cases of perfect absorption $|\lambda_{\nu_{-}}| = 0$ and lasing $|\lambda_{\nu_{+}}| = \infty$ , respectively. Interestingly, eigenstates in the broken regime are characterized by a vanishing number of pseudophotons: By considering the pseudounitarity condition [Eq. (5)] for states in the broken regime, we arrive at

$$
\langle c _ {\nu_ {\pm}} ^ {\mathrm{in}} | V | c _ {\nu_ {\pm}} ^ {\mathrm{in}} \rangle = \left| \lambda_ {\nu_ {\pm}} \right| ^ {2} \langle c _ {\nu_ {\pm}} ^ {\mathrm{in}} | V | c _ {\nu_ {\pm}} ^ {\mathrm{in}} \rangle .\tag{14}
$$

As $|\lambda_{\nu_{\pm}}|^{2} \neq 1$ , this entails $\langle c_{\nu_{\pm}}^{in}|V|c_{\nu_{\pm}}^{in}\rangle = 0$ . Such a condition can only be achieved if the eigenstates $|c_{\nu_{\pm}}^{in}\rangle$ have an equal amount of photons associated with their negative- and positive-frequency channels. This result highlights the importance of the interplay between positive- and negative-frequency components for states in the broken regime. Crucially, however, the fact that the number of pseudophotons of the states $|c_{\nu_{\pm}}^{in}\rangle$ vanishes, does not necessarily mean that the photon content also vanishes. Rather, such states represent light fields that can carry a finite amount of energy as $\langle c_{\nu_{\pm}}^{in}|c_{\nu_{\pm}}^{in}\rangle$ can be finite even though $\langle c_{\nu_{+}}^{in}|V|c_{\nu_{+}}^{in}\rangle = 0$ .

On the contrary, there exist states that not only conserve the number of pseudophotons during scattering, but additionally they also conserve the number of photons. One example are eigenstates of $S_{F}$ in the unbroken symmetry regime as

$$
\langle c _ {\nu_ {0}} ^ {\mathrm{out}} | c _ {\nu_ {0}} ^ {\mathrm{out}} \rangle = | \lambda_ {\nu_ {0}} | ^ {2} \langle c _ {\nu_ {0}} ^ {\mathrm{in}} | c _ {\nu_ {0}} ^ {\mathrm{in}} \rangle = \langle c _ {\nu_ {0}} ^ {\mathrm{in}} | c _ {\nu_ {0}} ^ {\mathrm{in}} \rangle .\tag{15}
$$

In general, however, a superposition of states from the unbroken regime does not conserve the number of photons due to the non-orthogonality of the eigenstates of $S_{F}$ .

Another example of states that conserve the number of photons during scattering arises if positive- and negative-frequency components of the light field do not mix during the scattering process (for example, for weak and slow modulations). In this case, the pseudounitarity condition of $S_{F}$ reduces to a unitarity condition for the positive-and negative-frequency components individually. Then, eigenstates of the Floquet scattering matrix are orthogonal, and any superposition of such eigenstates conserves the number of photons during scattering. In such a case, no symmetry breaking and no EPs arise, preventing also the scenarios of CPA and lasing.

## D. Complex Scattering Objects

In this section, we demonstrate that our formalism is not only applicable to the basic one-dimensional slab system but also to much more complex scattering objects. Specifically, we study an isolated time-varying sphere and a metasurface made from these spheres.

First, we consider the scattering properties of an isolated time-varying sphere surrounded by free space, thus constituting a localized scatterer in three-dimensional space [see Fig. 3(a)]. The permittivity $\epsilon (\mathbf{r},t)$ of the system is given by

a
![](images/6f2bdcb82e6edb77b4494eed0bf7e22cd117d19eb23086d31371378aaee805cc.jpg)

b
![](images/5c8b739ee3d433151e1e346e75f6f07b74cf61384fd0b079199a236543534f3b.jpg)
e

c
d
![](images/04161f95a79106e9595aee6515581147fd8922413e65dcc1887bc4d5d59bb745.jpg)

![](images/3cefaaeb41eadcf19f81f910ddbbd55b133c566a9b8323fe179a47b8c94c05c8.jpg)

![](images/469daa83a905927fbd57dc332eb7431e00f41ffd6665b5f68148a7d2a682e1fd.jpg)
FIG. 3. Behavior of the eigenvalues $\lambda$ of the Floquet scattering matrix $S_{\mathrm{F}}$ for a time-varying isolated sphere (see text for parameter values). a, We consider the properties of light scattering off a sphere with a time-periodic permittivity function. b-d, The complex eigenvalues of $S_{\mathrm{F}}$ for varying modulation strengths $M_{\mathrm{s}}$ and for different quasifrequencies $\omega/\Omega = 0.47$ , 0.5, 0.53, respectively. For all three choices of the input quasifrequency, EPs form, and the system undergoes symmetry-breaking transitions. The red arrows indicate the appearance of additional EPs primarily associated with higher Floquet channels $n < -1$ and $n > 0$ . For increasing modulation strength, they enter the broken regime but quickly recombine on the unit circle again. Furthermore, only in c, where the light field fulfills the parametric resonance condition $\omega/\Omega = 0.5$ , some eigenvalues vanish. At this operational condition, the system can act as a CPA (star symbol). Since the eigenvalues come in inverse complex conjugate pairs, there simultaneously exist diverging eigenvalues corresponding to the system acting as a laser (not shown). e, The absolute value of the minimal eigenvalue of $S_{\mathrm{F}}$ is shown here as a function of the quasifrequency $\omega$ and modulation strength $M_{\mathrm{s}}$ . The unbroken regime where all eigenvalues are unimodular (white) is separated from the broken regimes in which $|\lambda_{\min}| < 1$ (yellow) by exceptional lines (black lines). The star symbol indicates the CPA condition. The red arrow in panel e indicates the broken regime associated with the additional EPs visible in panel c.

$$
\epsilon (\mathbf {r}, t) = 1 + \chi_ {0} [ 1 + M _ {\mathrm{s}} \cos (\Omega t) ] \Theta (R - r).\tag{16}
$$

We set $\chi_{0}=11.67$ , which corresponds to the susceptibility of silicon at near-infrared frequencies. Furthermore, we choose the radius of the sphere to be R=568 nm and set the modulation frequency to $\Omega=2\pi\times151$ THz. We compute the $S_{F}$ matrix of the time-varying sphere following the method introduced in Ref. 47 using n=[-8,7] Floquet channels and multipoles up to the octupole order (for details see Sec. S2 of the Supplementary Information).

In Fig. 3(b)-(d), we plot all complex eigenvalues of $S_{\mathrm{F}}$ for this system as a function of the modulation strength $M_{\mathrm{s}}$ and for three choices of the input quasifrequency: $\omega / \Omega = 0.47, 0.5, 0.53$ . Similar to the slab system, we observe in all three cases that for a weak modulation, i.e., $M_{\mathrm{s}} \ll 0.1$ , all eigenvalues reside on the unit circle. As

$M_{s}$ increases, EPs form signaling the onset of the broken regime. In Fig. 3(b) and (d), we find that if the wave field is detuned from the parametric resonance condition, the eigenvalues that enter the broken symmetry regime do not approach 0 or $\infty$ . Only for $\omega/\Omega = 0.5$ we see that there exist eigenvalues for which $|\lambda| \to 0$ [see Fig. 3(c)] and, therefore, also eigenvalues $1/|\lambda^{*}| \to \infty$ for the same $M_{s}$ (not shown). This confirms that, like in the slab setup, a sphere made from a material with a periodically time-varying permittivity can also simultaneously act as a CPA and as a laser. We find that the eigenvalues corresponding to the CPA and lasing points arise primarily due to the magnetic dipolar part of the $S_{F}$ matrix of the sphere. Furthermore, due to the symmetry of the sphere along all three spatial dimensions, these eigenvalues have a three-fold degeneracy. Note that we observe additional EPs [indicated by red arrows in Fig. 3(c) and (e)], which only appear due to the higher-order Floquet channels ( $n < -1$ and $n > 0$ ). However, for the parameter values we consider here, these eigenvalues do not reach the CPA and lasing condition, but rather recombine at the unit circle again for increasing modulation strength.

![](images/b527db543b9a19f3000dec7c39dd62f918a3206d3b6b9c5b8b60e02eeab24557.jpg)

![](images/ec0fa4c42fd879e49bf49622e6e611d4c9ad8dc05b1688bde50464c831eda8ab.jpg)

![](images/13e12aef51bece1888c0d9a4efcc902f5e30dc0067fd4dcd57c0e32dc3c96f1b.jpg)

![](images/16dedfcf11cd930da3180f25beb97845bdbd600e743f4b1225299e831c7e41f9.jpg)
FIG. 4. Space-integrated incoming and outgoing intensities for the CPA and lasing states for the time-varying sphere (see text for parameter values) as a function of time t. We evaluate the electric fields in the far-field at $r = 1000 \times Tc$ and spatially integrate them on an imaginary sphere that encloses the time-varying sphere. a, Incoming and b, outgoing intensity corresponding to the CPA state with an eigenvalue $\lambda = 2 \times 10^{-4}$ . An extreme attenuation of the intensity can be observed, and nearly no outgoing field is produced. c, Incoming and d, outgoing intensity of the lasing state corresponding to an eigenvalue $\lambda = 5 \times 10^{3}$ . Here, we observe an extreme amplification of the incoming light field. Furthermore, we see the time-reversal symmetry of the CPA and lasing states: The incoming CPA state is the time-reversed of the outgoing lasing state.

As for the slab, we investigate the influence of the quasifrequency $\omega$ and of the driving strength $M_{s}$ on the eigenvalues of $S_{F}$ also in the present case. The corresponding plot of the minimal eigenvalue $\lambda_{min}$ of $S_{F}$ as a function of $\omega$ and $M_{s}$ is shown in Fig. 3(e). If the incident light field is spectrally far detuned from the parametric resonance condition $\omega/\Omega = 0.5$ , no exceptional points are formed (white region). However, near $\omega/\Omega = 0.5$ , the system can undergo a symmetry-breaking transition and EPs appear (black curve). By choosing $\omega/\Omega = 0.5$ and $M_{s} = 0.3$ , we observe the CPA-lasing point indicated by a nearly vanishing eigenvalue. Specifically, we numerically find that for this set of parameters, $|\lambda_{min}| = 2 \times 10^{-4}$ . To prove that in the vicinity of this small eigenvalue, there truly exists a vanishing eigenvalue, we again calculate the winding number over a loop enclosing this CPA point. We find $\text{wind}(S_{\text{F}}) \approx 6 - 3.5 \times 10^{-4}$ , as expected due to the three-fold degeneracy of each eigenvalue. Next, we investigate the spatiotemporal behavior of the associated CPA and lasing states. In Fig. 4, we plot the space-integrated incoming and outgoing far-field intensity corresponding to the eigenstates of the minimal (CPA) and maximal (lasing) eigenvalues. For numerical convenience, the intensity is space-integrated on the surface of an imaginary sphere with radius $r = 1000 \times Tc$ ( $T = 2\pi/\Omega$ and c is the speed of light in vacuum), such that $r \gg R$ . For the CPA state, we observe an extreme reduction of the outgoing wave intensity compared to the input [see Fig. 4(a)-(b)]. Conversely, we see that if we choose the lasing state as the input, the outgoing intensity is greatly enhanced, as expected [see Fig. 4(c)-(d)]. Furthermore, we can observe the time-reversal symmetry of both states according to Eq. (12): the input CPA field is the time-reversed of the output lasing field.

We already saw that by exploiting static resonances we can reduce the $M_{s}$ needed to form EPs (see Subsec. IV C 1). Now, using high-quality factor (high-Q) resonances, we show that we can also reduce the modulation strength $M_{s}$ required for CPA and lasing points [15]. In particular, we use the quasi-bound states in the continuum (qBICs) of metasurfaces, which are not present in the slab (and homogeneous sphere) system [15, 48]. The considered metasurfaces are made from a periodic arrangement on a square lattice of spheres made from a time-varying medium [see Fig. 5(a)].

First, we consider the case of a static metasurface without any time-modulation, i.e., $M_{s}=0$ . We optimize the geometry parameters of the metasurface such that it supports a bound state in the continuum (BIC) at normal incidence, i.e., for the Bloch wavevector $\mathbf{k}_{\parallel}=(k_{x},k_{y})^{\mathrm{T}}=0$ [49, 50]. The radius of the spheres in the metasurface is R=225 nm, and the lattice constant is a=3R. The reflectivity R of the metasurface is shown in Fig. 5(b) as a function of $\omega$ and the x-component of the Bloch wavevector $k_{x}$ . For simplicity, we assume $k_{y}=0$ . Here, we observe a sharp resonant behavior of R in the $\omega-k_{x}$ space. As $k_{x}\to0$ , the linewidth of the resonance becomes vanishingly small, and the resonance eventually disappears at $k_{x}=0$ and $\omega=\omega_{BIC}$ , indicating a BIC there, which does not couple to external radiation. In the following, we use a qBIC of the metasurface, which is formed by slightly detuning the parameters of the system from the perfect BIC condition, such that the resulting resonance has a high but finite Q-factor.

For the simulations of the time-varying metasurface, we choose $k_{x} = 0.05\pi/a$ and correspondingly $\omega = \omega_{qBIC} = 0.912855\omega_{BIC}$ . Note that the numerical value of $\omega_{qBIC}$ has to be chosen precisely due to the extremely narrow linewidth of the qBIC [see Fig. 5(b)]. Furthermore, we use $\Omega = 2\omega_{qBIC}$ , which ensures that upon time-modulation, the modes of the static metasurface at $\omega_{0} = \omega_{qBIC}$ and $\omega_{-1} = -\omega_{qBIC}$ are resonantly coupled to each other at $\Omega = 2\pi \times 387$ THz, such that the resulting hybrid modes have minimal radiation losses. We numerically compute the corresponding Floquet scattering matrix of the time-varying metasurface following the methods introduced in Refs. 15 and 51 using n = [-2,1] Floquet channels and multipoles up to the octupolar order (see Sec. S3 in the Supplementary Information for details). The metasurface is subwavelength at the frequencies $\omega_{0}$ and $\omega_{-1}$ , and the first diffraction order is propagating for the frequencies $\omega_{-2}$ and $\omega_{1}$ . We verified that this choice of parameters is sufficient to ensure

![](images/37cd95ca8c20927a5b0b63d3b00b66d86af54bafaa439ad7ba702e08ac44b68a.jpg)
c

b
![](images/82bd321c432c324de9dbc4e034602fbaf25901951b218644a2d20199a0130adc.jpg)

![](images/ca0bd07731154b86592b0172627c070425133f7f04b205db3d6c35bea9ac4d67.jpg)

d
![](images/19d6957e1ddf7396eecbf464c24b8094c5478852140ff69f8b225efa5c2082f3.jpg)
FIG. 5. Behavior of the eigenvalues $\lambda$ of the Floquet scattering matrix $S_{\mathrm{F}}$ for a time-varying metasurface. a, We consider the properties of light scattering off a metasurface that consists of a periodic arrangement (on a square lattice) of spheres made from a time-varying medium. The time-varying medium is characterized by the permittivity $\epsilon (\mathbf{r},t) = 1 + \chi_0[1 + M_{\mathrm{s}}\cos (\Omega t)]f(\mathbf{r})$ . Here, $f(\mathbf{r}) = 1$ for $\mathbf{r}$ at the spatial domains occupied by the spheres and 0 otherwise (see text for the other parameter values). b, The reflectivity $\mathcal{R}$ of the metasurface shown in a as a function of $\omega$ and the $x$ -component of Bloch wavevector $k_{x}$ under static conditions (i.e., $M_{\mathrm{s}} = 0$ ) and $k_{y} = 0$ . We note that at $k_{x} = 0$ , there exists a BIC that has a symmetry compatible with TE-polarized plane waves. c, The complex eigenvalues of $S_{\mathrm{F}}$ for varying modulation strengths $M_{\mathrm{s}}$ and for the quasifrequency $\omega /\Omega = 0.5$ . Here, we observe that at a certain $M_{\mathrm{s}}$ , the light field fulfills the parametric resonance condition, leading to vanishing eigenvalues (indicated by the star symbol), signifying CPA and lasing. d, The absolute value of the minimal eigenvalue of $S_{\mathrm{F}}$ as a function of the quasifrequency $\omega$ and modulation strength $M_{\mathrm{s}}$ . The unbroken regime where all eigenvalues are unimodular (white) is separated from the broken regime in which $|\lambda_{\mathrm{min}}| < 1$ (yellow) by an exceptional line (black). The star symbol indicates the CPA condition. To prove that $|\lambda_{\mathrm{min}}|$ vanishes entirely at the CPA point, we calculated the winding number over a loop enclosing the CPA point. We find that the winding number is $\mathrm{wind}(S_{\mathrm{F}})\approx 2 - 10^{-4}$ , as expected, due to the TE-polarized qBIC mode of the metasurface.

numerical convergence.

To show the existence of EPs, CPA, and lasing points, we plot all the complex eigenvalues $\lambda$ of the corresponding $S_{F}$ as a function of $M_{s}$ for the quasifrequency $\omega/\Omega = 0.5$ in Fig. 5(c). As predicted by our formalism, we observe that as $M_{s}$ is increased, the eigenvalues coalesce on the unit circle forming an EP. On a further increase of $M_{s}$ , these eigenvalues leave the unit circle in a pairwise manner. We also find that at a certain $M_{s}$ , there exists an eigenvalue whose absolute value approaches 0, marking a CPA point. Conversely, for its inverse conjugate pair $1/\lambda^{*}$ , the absolute value diverges for the same driving strength $M_{s}$ (not shown). However, note that contrary to the previous scattering setups, the modulation strengths to form EPs, and to make the system operate as a CPA or laser, is drastically reduced. In particular, to form an

EP, we require $M_{s} = 8.16 \times 10^{-5}$ , and for the CPA/lasing point, we require $M_{s} = 1.84 \times 10^{-4}$ . By moving the operation point closer to the BIC, i.e., $\omega \rightarrow \omega_{BIC}$ and $k_{x} \rightarrow 0$ , the modulation strength required for EPs, CPA, and lasing points can be made arbitrarily small, i.e., $M_{s} \rightarrow 0$ . This demonstrates that low-threshold lasing is possible by leveraging the qBICs of metasurfaces [52]. To highlight the CPA and lasing conditions more clearly, we plot the minimal eigenvalue $\lambda_{min}$ of $S_{F}$ of the metasurface as a function of $\omega$ and $M_{s}$ in Fig. 5(c). Here, we observe that at $\omega/\Omega = 0.5$ and $M_{s} = 1.84 \times 10^{-4}$ , the system behaves as a CPA indicated by $|\lambda_{min}| \rightarrow 0$ . Furthermore, as predicted by our theory, for the same values of $\omega$ and $M_{s}$ , $|\lambda_{max}| \rightarrow \infty$ , marking that the system acts as a laser there. We refer the reader to Sec. S1 of the Supplementary Information for details on the symmetry of the far-field intensities assuming the CPA and lasing eigenstates as input fields.

## III. DISCUSSION

We have shown that in periodically time-varying systems, symmetry-breaking transitions occur at exceptional points. Specifically, we revealed that the eigenvalues of the Floquet scattering matrix are either unimodular, corresponding to the unbroken symmetry regime, or come in inverse complex-conjugate pairs, corresponding to the broken symmetry regime. This behavior of the eigenvalues is explained through the pseudounitarity of the Floquet scattering matrix. We provided an expression for the associated symmetry operator for arbitrary temporal modulations and showed that for a time-symmetric drive, this operator is the time-reversal operator. Furthermore, we demonstrated that if the incident light field has a quasifrequency of half the modulation frequency and thus fulfills the parametric resonance condition, a special situation can occur in which one eigenvalue vanishes while another one diverges. At this point, the system simultaneously acts as a CPA and as a laser. We demonstrated the working principle of our formalism using the example of a time-varying slab, a time-varying isolated sphere, and a time-varying metasurface. Leveraging the qBICs of metasurfaces, the modulation strength needed to achieve CPA and lasing points can be made arbitrarily small.

Our framework applies to objects of arbitrary shape. We expect it to be relevant across different experimental platforms, where the characteristics of individual systems can be appropriately taken into account. This includes, but is not limited to, PTCs in the optical regime based on epsilon-near-zero materials $[11, 53, 54]$ , in the microwave regime based on a split-ring resonator setup $[55]$ , or using water waves $[56, 57]$ . Our work also provides a bridge to the quantum optics domain $[58]$ , where it will be of interest to study the implications of our scattering theory for the dynamical Casimir effect $[59, 60]$ or for the properties of the proposed Floquet laser (photon statistics, coherence properties etc.) $[61]$ .

## IV. MATERIAL AND METHODS

## A. Reciprocity

Here, we provide further insights into the connection between time-reversal symmetry and the Floquet scattering matrix. Specifically, we show that for time-reversal symmetric Floquet scattering systems described by $\epsilon(t) = \epsilon(-t)$ , the Floquet scattering matrix can be chosen to satisfy $VS_{F}^{T}V = S_{F}$ . To achieve this, we first prove that $S_{F}^{*} = S_{F}^{-1}$ holds in the considered case. Then, using the pseudounitarity condition Eq. (5), we can immediately conclude that $VS_{F}^{T}V =$

$S_{\mathrm{F}}$ . To keep the derivation general, we expand the far-field electric field as $\mathbf{E}(\mathbf{r},t) = \sum_{n}\mathbf{E}_{n}(\mathbf{r})e^{-i\omega_{n}t}$ , where $\mathbf{E}_n(\mathbf{r}) = \sum_\alpha c_{n,\alpha}^{\mathrm{in}}\mathbf{Z}_{n,\alpha}(\mathbf{r}) + (S_{\mathrm{F}}\mathbf{c}^{\mathrm{in}})_{n,\alpha}\mathbf{Z}_{n,\alpha}^*\left(\mathbf{r}\right)$ . Here, $\mathbf{Z}_{n,\alpha}(\mathbf{r})$ are the incoming far-field spatial mode functions labeled by $\alpha$ and $\mathbf{Z}_{n,\alpha}^{*}(\mathbf{r})$ are the corresponding outgoing far-field modes. Furthermore, we used that $\mathbf{c}^{\mathrm{out}} = S_{\mathrm{F}}\mathbf{c}^{\mathrm{in}}$ . Next, we complex conjugate the outgoing field coefficients $S_{\mathrm{F}}\mathbf{c}^{\mathrm{in}}$ and reinsert them into the system such that the corresponding far field $\tilde{\mathbf{E}}(r,t) = \sum_n\tilde{\mathbf{E}}_n(\mathbf{r})e^{-i\omega_n t}$ is given by $\tilde{\mathbf{E}}_n(\mathbf{r}) = \sum_\alpha (S_{\mathrm{F}}\mathbf{c}^{\mathrm{in}})_{n,\alpha}^*\mathbf{Z}_{n,\alpha}(\mathbf{r}) + [S_{\mathrm{F}}(S_{\mathrm{F}}\mathbf{c}^{\mathrm{in}})^*]_{n,\alpha}\mathbf{Z}_{n,\alpha}^*\left(\mathbf{r}\right)$ . The crucial step now is to show that $\mathbf{E}(\mathbf{r},t)$ and $\tilde{\mathbf{E}}(\mathbf{r},t)$ represent time-reversed fields of each other. This can be seen by noting that $\mathbf{E}_n^*(\mathbf{r})$ and $\tilde{\mathbf{E}}_n(\mathbf{r})$ consist out of the same incident field coefficients $(S_{\mathrm{F}}\mathbf{c}^{\mathrm{in}})_n^*$ and are solutions to the same wave equation

$$
\nabla \times (\nabla \times \mathbf {E} _ {n} ^ {*}) = k _ {n} \sum_ {m} \epsilon_ {n - m} ^ {*} k _ {m} \mathbf {E} _ {m} ^ {*},\tag{17}
$$

$$
\nabla \times (\nabla \times \tilde {\mathbf {E}} _ {n}) = k _ {n} \sum_ {m} \epsilon_ {n - m} k _ {m} \tilde {\mathbf {E}} _ {m},\tag{18}
$$

since for time-symmetric modulations we have $\epsilon_{n-m}^{*} = \epsilon_{n-m}$ . This implies that $S_{\mathrm{F}}(S_{\mathrm{F}}\mathbf{c}^{\mathrm{in}})^{*} = (\mathbf{c}^{\mathrm{in}})^{*}$ , proving that $S_{F}^{*} = S_{F}^{-1}$ .

## B. Symmetry Operators

In this subsection, we provide details on how the symmetry operator X introduced in Eq. (10) can be constructed for diagonalizable pseudounitary Floquet scattering matrices. The following discussion is general in the sense that we do not assume any spatial or temporal symmetries. We start by noting that, by definition, the right eigenvectors $|c_{\nu}^{in}\rangle$ and a left eigenvector $|\tilde{c}_{\nu}^{in}\rangle$ of the Floquet scattering matrix $S_{F}$ satisfy

$$
S _ {\mathrm{F}} \left| c _ {\nu} ^ {\mathrm{in}} \right\rangle = \lambda_ {\nu} \left| c _ {\nu} ^ {\mathrm{in}} \right\rangle ,\tag{19}
$$

$$
S _ {\mathrm{F}} ^ {\dagger} \left| \tilde {c} _ {\nu} ^ {\mathrm{in}} \right\rangle = \lambda_ {\nu} ^ {*} \left| \tilde {c} _ {\nu} ^ {\mathrm{in}} \right\rangle .\tag{20}
$$

Furthermore, we observe that the matrix V transforms a right eigenvector into a left eigenvector, which is a direct consequence of the pseudounitary relation Eq. (5), as

$$
S _ {\mathrm{F}} ^ {\dagger} V \left| c _ {\nu} ^ {\mathrm{in}} \right\rangle = V S _ {\mathrm{F}} ^ {- 1} \left| c _ {\nu} ^ {\mathrm{in}} \right\rangle = \frac {1}{\lambda_ {\nu}} V \left| c _ {\nu} ^ {\mathrm{in}} \right\rangle ,\tag{21}
$$

such that we can read off $|\tilde{c}_{\mu}^{in}\rangle = V|c_{\nu}^{in}\rangle$ . Furthermore, left and right eigenvectors can be chosen to be biorthonormal

$$
\langle \tilde {c} _ {\mu} ^ {\mathrm{in}} | c _ {\nu} ^ {\mathrm{in}} \rangle = \delta_ {\mu , \nu}.\tag{22}
$$

Here, we closely follow $[45]$ . Using left and right eigenvectors, we can now construct an anti-linear symmetry operator as

$$
X = \sum_ {\nu_ {0}} | c _ {\nu_ {0}} ^ {\mathrm{in}} \rangle \star \langle \tilde {c} _ {\nu_ {0}} ^ {\mathrm{in}} | + \sum_ {\nu_ {\pm}} | c _ {\nu_ {+}} ^ {\mathrm{in}} \rangle \star \langle \tilde {c} _ {\nu_ {-}} ^ {\mathrm{in}} | + | c _ {\nu_ {-}} ^ {\mathrm{in}} \rangle \star \langle \tilde {c} _ {\nu_ {+}} ^ {\mathrm{in}} |.\tag{23}
$$

Here, the symbol $\star$ represents complex conjugation in the following way

$$
\star \langle a | b \rangle = \langle a | b \rangle^ {*} = \langle b | a \rangle .\tag{24}
$$

By definition, the operator X satisfies

$$
X \left| c _ {\nu} ^ {\mathrm{in}} \right\rangle = \left\{ \begin{array}{l l} \left| c _ {\nu_ {0}} ^ {\mathrm{in}} \right\rangle , & \quad \nu = \nu_ {0}, \\ \left| c _ {\nu_ {\pm}} ^ {\mathrm{in}} \right\rangle , & \quad \nu = \nu_ {\mp}, \end{array} \right.\tag{25}
$$

and thus constitutes a symmetry operator that maps states from the symmetry-unbroken regime onto themselves and states in the symmetry-broken regime onto their partner state.

## 1. Degeneracies

Due to the underlying spatial symmetries of the scattering system, the associated Floquet scattering matrix $S_{F}$ may have degenerate eigenvalues. Suppose $S_{F}$ has a J-fold degenerate eigenvalue $\lambda$ to the corresponding normalized eigenvectors $|c_{j}^{in}\rangle$ for $j\in[1,J]$ . This degeneracy implies that any linear combination of the eigenvectors $|c_{j}^{in}\rangle$ is also an eigenvector of $S_{F}$ with the same eigenvalue $\lambda$ . Therefore, there are many sets of normalized eigenvectors of $S_{F}$ , and Eq. (10) cannot be applied to every such set. However, as verified numerically, there always exists at least one set of eigenvectors of $S_{F}$ to which Eq. (10) is applicable.

## C. Two-Band Model

Here, we provide additional details on the formation of EPs and the operational conditions for CPA and lasing for the time-varying slab. We employ the two-band model (TBM) approximation to arrive at an analytic result for the minimal modulation strength required for the formation of an EP. This enables us to investigate the impact of the normalized thickness of the slab on the formation of EPs. We analytically demonstrate that for specific slab thicknesses, even a very weak (infinitesimal) driving strength is sufficient to observe an EP. Furthermore, using a resonant state expansion (RSE), we provide an approximate analytical expression for the minimal modulation strength required for the system to act as a CPA or as a laser, solely based on the scattering parameters of the corresponding static system.

## 1. Minimal $M_{s}$ for an EP

In this subsection, we want to gain further insights into the formation of exceptional points and derive an expression for the minimal driving strength necessary to observe an EP [see Eq. (8)]. For this, we fix the quasifrequency to $\omega/\Omega = 1/2$ since we expect EPs to appear at lower modulation strengths for wave fields obeying the parametric resonance condition. Furthermore, we assume that only two modes $(n = -1,0)$ are necessary to describe the wave field inside the modulated slab (TBM approximation) [62]. The Maxwell equations inside the slab reduce in this case to

![](images/874aadd1df688bd2298e8469347f5bb5eadc8ecc59c208327dd35a21412f01fe.jpg)
FIG. 6. Minimal modulation strength $M_{s,EP}^{min}$ necessary to observe an EP for the time-varying slab as a function of its normalized thickness $L_{n}$ . The red and orange lines represent the results derived from the linearized TBM $M_{s,EP}^{+}$ (red) and $M_{s,EP}^{-}$ (orange). Using the TBM without an additional linearization, we find that EPs form at the same modulation strength (blue line). The black line corresponds to data from a full simulation including multiple frequency components $n = [-8,7]$ . In all cases, we see that for $L_{n} = 2m\pi$ with $m \in N$ only a minimal modulation is needed for an EP to form, as analytically predicted by the linearized TBM.

$$
\frac {\Omega^ {2}}{4} \left( \begin{array}{c c} 1 + \chi_ {0} & \chi_ {0} M _ {\mathrm{s}} / 2 \\ \chi_ {0} M _ {\mathrm{s}} / 2 & 1 + \chi_ {0} \end{array} \right) \binom{e _ {- 1}}{e _ {0}} = q ^ {2} c ^ {2} \binom{e _ {- 1}}{e _ {0}}.\tag{26}
$$

Solving the above eigenvalue problem results in wavevectors $q_{p} = \frac{\Omega}{2c}\sqrt{1 + \chi_{0} \pm \chi_{0} M_{s}/2}$ and eigenvectors $\mathbf{e}_{p} = (\pm1, 1)^{\mathrm{T}}$ with band index $p = \{1, 2\}$ . We thus have an analytical (approximate) solution for the wave field within the PTC-slab. We utilize this solution to establish the corresponding Floquet scattering matrix. To arrive at an analytic model, we linearize the above result and retain only terms up to linear order in $M_{s}$ . The resulting $4 \times 4$ Floquet scattering matrix takes the symmetric form

$$
\tilde {S} _ {\mathrm{F}} = \left( \begin{array}{c c} \tilde {r} _ {\mathrm{F}} & \tilde {t} _ {\mathrm{F}} \\ \tilde {t} _ {\mathrm{F}} & \tilde {r} _ {\mathrm{F}} \end{array} \right),\tag{27}
$$

where expressions with a tilde stand for results derived using the approximations described above. We find the eigenvalues $\tilde{\lambda}$ by making use of

$$
\begin{array}{r l} & 0 = \det \Big (\tilde {S} _ {\mathrm{F}} - \tilde {\lambda} \Big) = \det \left( \begin{array}{c c} \tilde {r} _ {\mathrm{F}} - \tilde {\lambda} & \tilde {t} _ {\mathrm{F}} \\ \tilde {t} _ {\mathrm{F}} & \tilde {r} _ {\mathrm{F}} - \tilde {\lambda} \end{array} \right) \\ & \quad = \det \Big (\tilde {r} _ {\mathrm{F}} + \tilde {t} _ {\mathrm{F}} - \tilde {\lambda} \Big) \det \Big (\tilde {r} _ {\mathrm{F}} - \tilde {t} _ {\mathrm{F}} - \tilde {\lambda} \Big). \end{array}\tag{28}
$$

Such a finding implies that each of the $2 \times 2$ matrices $\tilde{r}_{F} \pm \tilde{t}_{F}$ contributes a pair of eigenvalues, which we label as $\tilde{\lambda}_{1,2}^{\pm}$ . We, therefore, expect two EPs: one originating from the coalescence of $\tilde{\lambda}_{1}^{+}$ and $\tilde{\lambda}_{2}^{+}$ ; and one from the coalescence of $\tilde{\lambda}_{1}^{-}$ and $\tilde{\lambda}_{2}^{-}$ . We analytically find that the exceptional points form at the critical driving strengths

![](images/38079baa4d0a1e96702b108bed1380ff161da74422f0897c8ccd4ef9e49eb093.jpg)
FIG. 7. Minimal modulation strength $M_{s,CPA}^{min}$ necessary such that the time-varying slab operates as a CPA or laser as a function of the normalized thickness $L_{n}$ . The purple line shows the result from the resonant state expansion (RSE) Eq. (35). The green line represents the results derived from an analysis of the eigenvalues of $S_{F}$ using the TBM approximation, showing strong agreement with the RSE approach, as expected. For reference, we also plot $M_{s,EP}^{min}$ using the TBM approximation. We perform a full-scale simulation using n = [-8, 7] Floquet channels (black line).

$$
M _ {\mathrm{s,EP}} ^ {\pm} = \frac {4 (1 + \chi_ {0}) \sqrt {1 - \cos (L _ {\mathrm{n}})}}{\chi_ {0} \sqrt {1 + L _ {\mathrm{n}} ^ {2} / 2 - \cos (L _ {\mathrm{n}}) \pm 2 L _ {\mathrm{n}} \sin (L _ {\mathrm{n}} / 2)}},\tag{29}
$$

with the normalized thickness $L_{n} = L\Omega\sqrt{1 + \chi_{0}}/c$ . Equation (29) is the complete version of Eq. (8).

In Fig. 6, we now compare $M_{s,EP}^{\pm}$ (red and yellow dashed lines) to numerical data to investigate the validity of Eq. (29), where we made use of several approximations. For comparison, we calculate the minimal modulation strength necessary to observe an EP again, considering only the two modes n = -1, 0, but without linearizing with respect to the modulation strength (blue line). We find that, contrary to the prediction of the linearized TBM, both EPs form at the same modulation strength, which is why we only plot a single blue line. Furthermore, we present simulated data involving several frequency harmonics $n \in [-8, 7]$ (black line). In all cases, we observe that, indeed, in the vicinity of $L_{n} = 2m\pi$ with $m \in N$ only a very weak modulation is necessary to realize an EP as predicted by Eq. (29), respectively. One feature, which we only observe using the full simulation, is the occurrence of several dips. They primarily appear when the TBM predicts large modulation strengths for the formation of an EP [around $L_{n} = (2m + 1)\pi$ ]. These dips correspond to EPs, corresponding to a symmetry breaking of modes that are associated with higher Floquet channels n > 0 and n < -1. They are thus only visible in a description that goes beyond the TBM using more than two channels.

## 2. Minimal $M_{s}$ for CPA and Lasing

In the following, we derive an expression for the minimal driving strength necessary to observe CPA and lasing from the time-varying slab. Here, we make use of the Floquet resonant state theory developed in Ref. 14. Corresponding to the TBM, we only take into account two static resonant states: First, we have $E_{\mathrm{s},\alpha}^{\mathrm{RS}}$ with complex eigenfrequency $\bar{\omega}_{\mathrm{s},\alpha} = \omega_{\mathrm{s},\alpha} - i\gamma_{\mathrm{s},\alpha}$ , where $\alpha = 0,1,\ldots$ labels the resonant states. Secondly, we use the $m = -1$ replica of the negative twin with eigenfrequency $\Omega -\bar{\omega}_{\mathrm{s},\alpha}^{*}$ . Here, $\omega_{\mathrm{s},\alpha}\in \mathbb{R}$ is the resonance frequency and $\gamma_{\mathrm{s},\alpha}\in \mathbb{R}$ is the decay rate of the corresponding resonant state of the static system. Explicitly, for a slab of thickness $L$ and time-invariant permittivity $\epsilon = 1 + \chi_0$ we have $\omega_{\mathrm{s},\alpha} = \frac{\alpha\pi}{\sqrt{\epsilon}L}$ and $\gamma_{\mathrm{s},\alpha} = \frac{1}{\sqrt{\epsilon}L}\ln (|1 - \epsilon | / |1 + \epsilon |)$ [63, 64]. The resonant states inside the slab $|x|\leq L / 2$ read

$$
E _ {\mathrm{s}, \alpha} ^ {\mathrm{RS}} (x) = \frac {1}{\sqrt {\epsilon \varepsilon_ {0} L}} \left\{ \begin{array}{l l} \sin (\sqrt {\epsilon} \bar {\omega} _ {\mathrm{s}, \alpha} x / c), & \text {if \alpha is odd}, \\ \cos (\sqrt {\epsilon} \bar {\omega} _ {\mathrm{s}, \alpha} x / c), & \text {if \alpha is even}. \end{array} \right.\tag{30}
$$

We introduce the detuning $\Delta_{\alpha} = |\omega_{s,\alpha} - \Omega/2|$ and define the intensity of the resonant state inside the slab as $I_{\alpha} = \int_{-L/2}^{L/2} dx |E_{s,\alpha}^{\mathrm{RS}}(x)|^{2}$ and furthermore $g_{\alpha} = \chi_{0} M_{s} I_{\alpha}/2$ . In the following, we always choose that resonant state, i.e., that specific value of $\alpha$ , corresponding to the resonant state that has minimal detuning $\Delta_{\alpha}$ . Performing a similar analysis as in Ref. 14, Sec. V, we find the eigenfrequencies of the Floquet resonant states

$$
\tilde {\omega} _ {\pm} \approx \frac {\Omega}{2} - i \gamma_ {\mathrm{s}, \alpha} \pm \sqrt {\frac {\Omega^ {2}}{4} - \frac {\Delta_ {\alpha} ^ {2}}{g _ {\alpha} ^ {2}} + \gamma_ {\mathrm{s} , \alpha} ^ {2}}.\tag{31}
$$

Crucially, a scattering system acts as a laser (and thus, in our case, simultaneously as a CPA) when the eigenfrequency of a Floquet resonant state has a vanishing imaginary part. In our case, the eigenfrequency $\tilde{\omega}_{+}$ becomes real when two conditions are fulfilled: (i) the discriminant in the square root of Eq. (31) is positive, and (ii) when the square root compensates for the decay rate of the original mode, such that

$$
\gamma_ {\mathrm{s}, \alpha} = g _ {\alpha} \sqrt {\frac {\Omega^ {2}}{4} - \frac {\Delta_ {\alpha} ^ {2}}{g _ {\alpha} ^ {2}} + \gamma_ {\mathrm{s} , \alpha} ^ {2}}.\tag{32}
$$

For the modes of interest, it typically holds that the decay rate is much smaller than the resonance frequencies. Then, Eq. (32) becomes

$$
\begin{array}{c} \gamma_ {\mathrm{s}, \alpha} = g _ {\alpha} \sqrt {\frac {\Omega^ {2}}{4} - \frac {\Delta_ {\alpha} ^ {2}}{g _ {\alpha} ^ {2}}} \\ = \frac {\chi_ {0} M _ {\mathrm{s}} I _ {\alpha} \Omega}{4} \sqrt {1 - \left(\frac {4 \Delta_ {\alpha}}{\chi_ {0} M _ {\mathrm{s}} I _ {\alpha} \Omega}\right) ^ {2}}. \end{array}\tag{33}
$$

Condition (i) tells us that $\frac{4\Delta_{\alpha}}{\chi_0M_sI_\alpha\Omega} < 1$ . We therefore expand Eq. (33) according to $\sqrt{1 - x^2} = 1 - x^2 /2 + \mathcal{O}(x^4)$

valid for small $x$ and arrive at

$$
\gamma_ {\mathrm{s}, \alpha} \approx \frac {\chi_ {0} M _ {\mathrm{s}} I _ {\alpha} \Omega}{4} - \frac {2 \Delta_ {\alpha} ^ {2}}{\chi_ {0} M _ {\mathrm{s}} I _ {\alpha} \Omega}.\tag{34}
$$

Solving Eq. (34) for $M_{s}$ , the modulation necessary for the system to operate as a CPA or as a laser is given by

$$
M _ {\mathrm{s,CPA}} ^ {\mathrm{min}} \approx \frac {\gamma_ {\mathrm{s} , \alpha} + \sqrt {\gamma_ {\mathrm{s} , \alpha} ^ {2} + 2 \Delta_ {\alpha} ^ {2}}}{\chi_ {0} I _ {\alpha} \Omega / 2}.\tag{35}
$$

We highlight that Eq. (35) can be evaluated solely based on static resonant states and thus allows us to estimate $M_{s,CPA}^{min}$ only based on parameters of the static scattering system and the known driving frequency $\Omega$ . Furthermore, since we express the above equation in terms of the frequency and intensity of the static resonant states, Eq. (35) is general in the sense that it not only holds for the slab but for arbitrarily shaped objects.

To check the validity of the approximations used in the resonant state expansion, we compare Eq. (35) (purple line) with an eigenvalue analysis of the associated $S_{F}$ using the TBM approximation in Fig. 7 (green line). As expected, both approaches give nearly identical results. For reference, we also show the results of a full-scale simulation (black line), in which we track the eigenvalues of $S_{F}$ using n = [-8,7] Floquet channels. In general, we observe that Eq. (35) gives a reasonable approximation for $M_{s,CPA}^{min}$ , allowing us to adequately estimate the CPA-lasing modulation strength solely based on static information. More specifically, we observe that all three lines (RSE, TBM, full simulation) show a similar behavior: For those normalized thicknesses, where only a weak modulation strength $M_{s,EP}^{min}$ is necessary for the system to form an EP (blue line, same as in Fig. 6), also the modulation strength for the system to act as a CPA or a laser $M_{s,CPA}^{min}$ is reduced. Conversely, when a strong modulation is already necessary to reach an EP, a large amplitude is also needed for the system to operate as a CPA and laser.

## D. Data Availability

The data that support the plots within this paper are available from the corresponding author on reasonable request.

## ACKNOWLEDGMENTS

The authors acknowledge helpful discussions with V. Flynn, J. Gohsrich, P. A. Huidobro, L. Rebholz, and M. Verde. D.G., J.H., and S.R. acknowledge support from the Austrian Science Fund (FWF) under project P32300 (WAVELAND). A.C.V. acknowledges support by the project No 1.1.1.9/LZP/1/24/101: “Non-Hermitian physics of spatiotemporal photonic crystals of arbitrary shape (PROTOTYPE)” and the Visiting Awards for High Potentials from the University of Graz. P.G. and C.R. are part of the Max Planck School of Photonics, supported by the Bundesministerium für Bildung und Forschung, the Max Planck Society, and the Fraunhofer Society. P.G. acknowledges support from the Karlsruhe School of Optics and Photonics (KSOP). P.G. and C.R. acknowledge support by the German Research Foundation within the SFB 1173 (project ID no. 258734477).

d

# Supplementary Information

## S1. FAR-FIELD INTENSITY CALCULATIONS FOR THE CPA AND LASING POINTS OF THE TIME-VARYING METASURFACE

![](images/cd5055c5e15ee562709af1faf0e172f05ce0b9085c898d95e8dcdf07f077888c.jpg)

![](images/ae8f539e36853e6998ce956df7b43805d00815a91c9e65f470cf433ead22456d.jpg)

![](images/3d250d8b30d9552c3b2993664f25b1193de2bc2d202910475c692493c97db2ed.jpg)

![](images/e5d34e33f4be23ffacbab8cde3480d3b37b2e5137e4b9d52fb9d95ebf2d86287.jpg)
FIG. S1. Space-integrated incoming and outgoing intensities for the CPA and lasing states of the time-varying metasurface (see main text for parameter values) as a function of time t. We evaluate the electric fields in the far-field at $z = \pm 1000 \times Tc$ away from the metasurface and spatially integrate them in the x - y plane. a, Incoming intensity above the metasurface at $z = 1000 \times Tc$ (red full line) and below the metasurface at $z = -1000 \times Tc$ (black dashed line) and b, outgoing intensity above (green full line) and below (black dashed line) corresponding the CPA state with an eigenvalue $|\lambda| = 10^{-6}$ . An extreme attenuation of the intensity can be observed and nearly no outgoing field is produced. c and d, Same as a, b but for the lasing state corresponding to an eigenvalue $|\lambda| = 10^{6}$ . Here, we observe an extreme amplification of the incoming light field. We again see the time-reversal symmetry between the incoming CPA and outgoing lasing field and vice versa.

We plot the temporal intensity distribution in the far-field of the eigenstate corresponding to the minimal eigenvalue (CPA state) in Fig. S1(a)-(b). Specifically, we evaluate the electric field at $z = 1000 \times Tc$ above (full lines) and below (dashed lines) the metasurface in the far-field and spatially integrate the intensity in this x - y plane. The associated eigenvalue is $|\lambda| = 10^{-6}$ , resulting in a drastic absorption of the input power such that nearly no outgoing field is produced. Conversely, in Fig. S1(c)-(d), we depict the far-field intensity of the eigenstate to the maximal eigenvalue (lasing state). There, we observe a large amplification of the incoming field corresponding to the eigenvalue $|\lambda| = 10^{6}$ of $S_{F}$ . Again, the time-reversal symmetry between the CPA and the lasing state is apparent (the incoming CPA field is the time-reversed of the outgoing lasing field, and the incoming lasing field is the time-reversed of the outgoing CPA field).

## S2. FLOQUET SCATTERING MATRIX OF THE TIME-VARYING SPHERE

We expand the incident field $\mathbf{E}^{\mathrm{inc}}(\mathbf{r},t)$ and scattered field $\mathbf{E}^{\mathrm{sca}}(\mathbf{r},t)$ for the time-varying sphere in a basis of vector spherical waves (VSWs) as

$$
\mathbf {E} ^ {\mathrm{inc}} (\mathbf {r}, t) = \sum_ {n l m s} \tilde {a} _ {n l m s} ^ {\mathrm{inc}} \mathbf {F} _ {l m s} ^ {(1)} (k _ {n} \mathbf {r}) \mathrm{e} ^ {- i \omega_ {n} t},\tag{S1a}
$$

$$
\mathbf {E} ^ {\mathrm{sca}} (\mathbf {r}, t) = \sum_ {n l m s} \tilde {a} _ {n l m s} ^ {\mathrm{sca}} \mathbf {F} _ {l m s} ^ {(3)} (k _ {n} \mathbf {r}) \mathrm{e} ^ {- i \omega_ {n} t},\tag{S1b}
$$

where $k_{n} = \omega_{n}/c$ . Furthermore, $\mathbf{F}_{lms}^{(1)}(k_{n}\mathbf{r}) \left[ \mathbf{F}_{lms}^{(3)}(k_{n}\mathbf{r}) \right]$ represent the regular (radiating) VSWs with total angular momentum l = 1, 2, 3..., $l_{max}$ ; z-component of angular momentum m = -l, -l + 1, ..., l; and parity s = 0, 1. Here, s = 0 represents the transverse-electric (TE), and s = 1 represents the transverse-magnetic (TM) mode, respectively. Moreover, $l_{max}$ is the maximum multipolar order used in the expansion. Next, using the method introduced in Ref. 47, one can connect $|\tilde{a}^{\mathrm{inc}}\rangle$ and $|\tilde{a}^{\mathrm{sca}}\rangle$ by the Floquet T-matrix $\tilde{T}_{0}$ as

$$
\left| \tilde {a} ^ {\mathrm{sca}} \right\rangle = \tilde {T} _ {0} \left| \tilde {a} ^ {\mathrm{inc}} \right\rangle .\tag{S2}
$$

Note that $\tilde{T}_0$ is a square matrix with the dimension $2l_{\mathrm{max}}(2N)(l_{\mathrm{max}} + 2)$ .

Here, for a given $|\tilde{a}^{sca}\rangle$ , the scattered power $P_{n}^{sca}$ for the frequency $\omega_{n}$ in the far-field is given by [47]

$$
P _ {n} ^ {\mathrm{sca}} = \sum_ {l m s} \frac {c ^ {2} | \tilde {a} _ {n l m s} ^ {\mathrm{sca}} | ^ {2}}{Z _ {0} \omega_ {n} ^ {2}}.\tag{S3}
$$

Therefore, the photon flux $\phi_{n}$ corresponding to $P_{n}^{sca}$ reads

$$
\phi_ {n} = \frac {P _ {n} ^ {\mathrm{sca}}}{\hbar | \omega_ {n} |} = \sum_ {l m s} \frac {c _ {0} ^ {2} | \tilde {a} _ {n l m s} ^ {\mathrm{sca}} | ^ {2}}{Z _ {0} \hbar | \omega_ {n} | ^ {3}},\tag{S4}
$$

where $\hbar$ is the reduced Planck's constant, and $Z_{0}$ is the impedance of free space. From Eq. (S4), we note that the scattered field coefficients $|a^{\mathrm{sca}}\rangle$ in a photon-flux normalized basis can be obtained by the transformation

$$
a _ {n l m s} ^ {\mathrm{sca}} = \sqrt {\frac {c ^ {2}}{Z _ {0} \hbar | \omega_ {n} | ^ {3}}} \tilde {a} _ {n l m s} ^ {\mathrm{sca}}.\tag{S5}
$$

Similarly, for the incident field coefficients $|a^{inc}\rangle$ , we can write

$$
a _ {n ^ {\prime} l ^ {\prime} m ^ {\prime} s ^ {\prime}} ^ {\mathrm{inc}} = \sqrt {\frac {c ^ {2}}{Z _ {0} \hbar | \omega_ {n ^ {\prime}} | ^ {3}}} \tilde {a} _ {n ^ {\prime} l ^ {\prime} m ^ {\prime} s ^ {\prime}} ^ {\mathrm{inc}}.\tag{S6}
$$

Furthermore, combining Eqs. (S2), (S5), and (S6), we get

$$
\left| a ^ {\mathrm{sca}} \right\rangle = T _ {0} \left| a ^ {\mathrm{inc}} \right\rangle .\tag{S7}
$$

Here, $T_{0}$ is the Floquet T-matrix of the sphere in a photon-flux normalized basis. The elements of $T_{0}$ can be calculated from those of $\tilde{T}_{0}$ by using the transformation [65]

$$
T _ {0} ^ {\{n l m s \}, \{n ^ {\prime} l ^ {\prime} m ^ {\prime} s ^ {\prime} \}} = \sqrt {\left| \frac {\omega_ {n ^ {\prime}}}{\omega_ {n}} \right| ^ {3}} \tilde {T} _ {0} ^ {\{n l m s \}, \{n ^ {\prime} l ^ {\prime} m ^ {\prime} s ^ {\prime} \}}.\tag{S8}
$$

Here, the first set of indices enumerates rows, and the second set enumerates columns of the Floquet T-matrices.

Finally, we can compute the Floquet scattering matrix of the time-varying sphere in a photon-flux normalized basis as $[66]$

$$
S _ {\mathrm{F}} = \mathbb {1} + 2 T _ {0}.\tag{S9}
$$

Note that the Floquet scattering matrix $S_{F}$ satisfies Eq. (5). Furthermore, $S_{F}$ also satisfies Eq. (11) for permittivity profiles $\epsilon(t)$ that are symmetric with respect to t = 0.

## S3. FLOQUET SCATTERING MATRIX OF THE TIME-VARYING METASURFACE

We expand the incoming field $\mathbf{E}^{\mathrm{in}}(\mathbf{r},t)$ and the outgoing field $\mathbf{E}^{\mathrm{out}}(\mathbf{r},t)$ from the time-varying metasurface for the Floquet frequency $\omega$ and Bloch wavevector $k_{\parallel}$ in a basis of plane waves as [51, 67]

$$
\mathbf {E} ^ {\mathrm{in}} (\mathbf {r}, t) = \sum_ {n \mathbf {g} \alpha d} \tilde {u} _ {n \mathbf {g} \alpha d} ^ {\mathrm{in}} \mathbf {P} _ {\mathbf {g} \alpha d} (k _ {n} \mathbf {r}) \mathrm{e} ^ {- i \omega_ {n} t},\tag{S10a}
$$

$$
\mathbf {E} ^ {\mathrm{out}} (\mathbf {r}, t) = \sum_ {n \mathbf {g} \alpha d} \tilde {u} _ {n \mathbf {g} \alpha d} ^ {\mathrm{out}} \mathbf {P} _ {\mathbf {g} \alpha d} (k _ {n} \mathbf {r}) \mathrm{e} ^ {- i \omega_ {n} t},\tag{S10b}
$$

where $\mathbf{P}_{\mathbf{g}\alpha d}(k_{n}\mathbf{r})$ corresponds to a plane wave with the wavevector $\mathbf{k}_{ng\alpha d} = (\mathbf{k}_{\parallel} + \mathbf{g}) + (-1)^{d}\sqrt{k_{n}^{2} - (\mathbf{k}_{\parallel} + \mathbf{g})^{2}}\hat{\mathbf{z}}$ , and polarization $\alpha$ . Here, $\alpha$ takes the values 0 and 1 for TE and TM polarized plane waves, respectively. Furthermore, d takes the values 0 and 1 for downward and upward propagating plane waves (with respect to the metasurface), respectively. Moreover, g corresponds to a reciprocal lattice vector. In the expansion of the fields $\mathbf{E}^{\mathrm{in}}(\mathbf{r}, t)$ and $\mathbf{E}^{\mathrm{out}}(\mathbf{r}, t)$ , we assume $G_{n}$ as the total number of diffraction orders for the frequency $\omega_{n}$ . Here, $G_{n}$ is chosen such that $|k_{n}| > |(\mathbf{k}_{\parallel} + \mathbf{g})|$ for all $\omega_{n}$ . This choice ensures that only the contribution of propagating plane waves is taken into account when evaluating $\mathbf{E}^{\mathrm{in}}(\mathbf{r}, t)$ and $\mathbf{E}^{\mathrm{out}}(\mathbf{r}, t)$ . Such a choice of $G_{n}$ is justified as it implies that the sources and the detectors for $\mathbf{E}^{\mathrm{in}}(\mathbf{r}, t)$ and $\mathbf{E}^{\mathrm{out}}(\mathbf{r}, t)$ are placed in the far-field of the metasurface.

We connect the plane wave coefficients $|\tilde{u}^{in}\rangle$ and $|\tilde{u}^{out}\rangle$ using the Floquet scattering matrix $\tilde{S}_{F}$ of the metasurface as [51]

$$
\left| \tilde {u} ^ {\mathrm{out}} \right\rangle = \tilde {S} _ {\mathrm{F}} \left| \tilde {u} ^ {\mathrm{in}} \right\rangle .\tag{S11}
$$

Note that $\tilde{S}_{\mathrm{F}}$ is a square matrix with dimension $\sum_{n = -N}^{N - 1}4G_n$ .

Next, we write the power flux $P_{n}^{out}$ carried by the outgoing fields $\mathbf{E}^{\mathrm{out}}(\mathbf{r},t)$ for the frequency $\omega_{n}$ in the far-field as [51]

$$
P _ {\mathrm{out}} ^ {n} = \frac {c}{2 Z _ {0}} \sum_ {\mathbf {g} \alpha d} \frac {| k _ {z , n \mathbf {g} \alpha d} |}{| \omega_ {n} |} | \tilde {u} _ {n \mathbf {g} \alpha d} ^ {\mathrm{out}} | ^ {2}.\tag{S12}
$$

Here, $k_{z,ng\alpha d}$ refers to the z-component of the wavevector $k_{ng\alpha d}$ . The photon flux $\phi_{n}$ corresponding to $P_{n}^{out}$ is given by

$$
\phi_ {n} = \frac {c}{2 Z _ {0}} \sum_ {\mathbf {g} \alpha d} \frac {| k _ {z , n \mathbf {g} \alpha d} |}{\hbar \omega_ {n} ^ {2}} | \tilde {u} _ {n \mathbf {g} \alpha d} ^ {\mathrm{out}} | ^ {2}.\tag{S13}
$$

From Eq. (S13), we note that the outgoing field coefficient $|u^{out}\rangle$ in a photon-flux normalized basis can be obtained by the transformation

$$
u _ {n \mathbf {g} \alpha d} ^ {\mathrm{out}} = \sqrt {\frac {c | k _ {z , n \mathbf {g} \alpha d} |}{2 Z _ {0} \hbar \omega_ {n} ^ {2}}} \tilde {u} _ {n \mathbf {g} \alpha d} ^ {\mathrm{out}}.\tag{S14}
$$

Similarly, for the incoming field coefficients $|u^{in}\rangle$ , we can write

$$
u _ {n ^ {\prime} \mathbf {g} ^ {\prime} \alpha^ {\prime} d ^ {\prime}} ^ {\mathrm{in}} = \sqrt {\frac {c | k _ {z , n ^ {\prime} \mathbf {g} ^ {\prime} \alpha^ {\prime} d ^ {\prime}} |}{2 Z _ {0} \hbar (\omega_ {n} ^ {\prime}) ^ {2}}} \tilde {u} _ {n ^ {\prime} \mathbf {g} ^ {\prime} \alpha^ {\prime} d ^ {\prime}} ^ {\mathrm{in}}.\tag{S15}
$$

Furthermore, combining Eqs. (S11), (S14), and (S15), we get

$$
\left| u ^ {\mathrm{out}} \right\rangle = S _ {\mathrm{F}} \left| u ^ {\mathrm{in}} \right\rangle .\tag{S16}
$$

Here, $S_{F}$ is the Floquet scattering matrix of the metasurface in a photon flux normalized basis. The elements of $S_{F}$ can be calculated from those of $\tilde{S}_{F}$ using the transformation

$$
\begin{array}{r} S _ {\mathrm{F}} ^ {\{n \mathbf {g} \alpha d \}, \{n ^ {\prime} \mathbf {g} ^ {\prime} \alpha^ {\prime} d ^ {\prime} \}} = \left| \frac {\omega_ {n ^ {\prime}}}{\omega_ {n}} \right| \sqrt {\left| \frac {k _ {z , n \mathbf {g} \alpha d}}{k _ {z , n ^ {\prime} \mathbf {g} ^ {\prime} \alpha^ {\prime} d ^ {\prime}}} \right|} \\ \times \tilde {S} _ {\mathrm{F}} ^ {\{n \mathbf {g} \alpha d \}, \{n ^ {\prime} \mathbf {g} ^ {\prime} \alpha^ {\prime} d ^ {\prime} \}}. \end{array}\tag{S17}
$$

We arrange $|u^{out}\rangle$ and $|u^{in}\rangle$ following the conventions: $|u^{out}\rangle = (\mathbf{u}_{\downarrow}^{\mathrm{out}}, \mathbf{u}_{\uparrow}^{\mathrm{out}})^{\mathrm{T}}$ and $|u^{in}\rangle = (\mathbf{u}_{\uparrow}^{\mathrm{in}}, \mathbf{u}_{\downarrow}^{\mathrm{in}})^{\mathrm{T}}$ , respectively. Here, $\mathbf{u}_{\uparrow}^{\mathrm{out}}(\mathbf{u}_{\uparrow}^{\mathrm{in}})$ correspond to the upward propagating outgoing (incoming) and $\mathbf{u}_{\downarrow}^{\mathrm{out}}(\mathbf{u}_{\downarrow}^{\mathrm{in}})$ correspond to the downward propagating outgoing (incoming) field coefficients. Importantly, we arrange the entries of the $S_{F}$ matrix as,

$$
S _ {\mathrm{F}} = \left( \begin{array}{c c} S _ {\downarrow \uparrow} & S _ {\downarrow \downarrow} \\ S _ {\uparrow \uparrow} & S _ {\uparrow \downarrow} \end{array} \right)\tag{S18}
$$

which is crucial for correctly describing the spectral locations of the EPs as shown in [68]. Note that $S_{F}$ satisfies

[1] E. Galiffi, R. Tirole, S. Yin, H. Li, S. Vezzoli, P. A. Huidobro, M. G. Silveirinha, R. Sapienza, A. Alù, and J. B. Pendry, Photonics of time-varying media, Adv. Photonics 4, 014002 (2022).

[2] D. L. Sounas and A. Alù, Non-reciprocal photonics based on time modulation, Nat. Photonics 11, 774 (2017).

[3] A. Prain, S. Vezzoli, N. Westerberg, T. Roger, and D. Faccio, Spontaneous photon production in time-dependent epsilon-near-zero materials, Phys. Rev. Lett. 118, 133904 (2017).

[4] X. Ni, S. Yin, H. Li, and A. Alù, Topological wave phenomena in photonic time quasicrystals, Phys. Rev. B 111, 125421 (2025).

[5] Z. Hayran and F. Monticone, Beyond the Rozanov bound on electromagnetic absorption via periodic temporal modulations, Phys. Rev. Appl. 21, 044007 (2024).

[6] E. Lustig, O. Segal, S. Saha, C. Fruhling, V. M. Shalaev, A. Boltasseva, and M. Segev, Photonic time-crystals - fundamental concepts, Opt. Express 31, 9165 (2023).

[7] M. M. Asgari, P. Garg, X. Wang, M. S. Mirmoosa, C. Rockstuhl, and V. Asadchy, Theory and applications of photonic time crystals: a tutorial, Adv. Opt. Photon. 16, 958 (2024).

[8] M. Lyubarov, Y. Lumer, A. Dikopoltsev, E. Lustig, Y. Sharabi, and M. Segev, Amplified emission and lasing in photonic time crystals, Science 377, 425 (2022).

[9] J. B. Khurgin, Photonic time crystals and parametric amplification: Similarity and distinction, ACS Photonics 11, 2150 (2024).

[10] X. Wang, M. S. Mirmoosa, V. S. Asadchy, C. Rockstuhl, S. Fan, and S. A. Tretyakov, Metasurface-based realization of photonic time crystals, Sci. Adv. 9, eadg7541 (2023).

[11] E. Galiffi, A. C. Harwood, S. Vezzoli, R. Tirole, A. Alù, and R. Sapienza, Optical coherent perfect absorption and amplification in a time-varying medium (2024), arXiv:2410.16426 [physics.optics].

[12] I. Stefanou, P. A. Pantazopoulos, and N. Stefanou, Light scattering by a spherical particle with a time-periodic refractive index, J. Opt. Soc. Am. B 38, 407 (2021).

[13] X. Wang, P. Garg, M. S. Mirmoosa, A. G. Lamprianidis, C. Rockstuhl, and V. S. Asadchy, Expanding momentum bandgaps in photonic time crystals through resonances, Nat. Photonics 19, 149 (2025).

[14] A. C. Valero, S. Gladyshev, D. Globosits, S. Rotter, E. A. Muljarov, and T. Weiss, Resonant states of structured photonic time crystals (2025), arXiv:2506.01472 [physics.optics].

[15] P. Garg, E. Almpanis, L. Zimmer, J. D. Fischbach, X. Wang, M. S. Mirmoosa, M. Nyman, N. Stefanou, N. Papanikolaou, V. Asadchy, and C. Rockstuhl, Photonic time crystals assisted by quasi-bound states in the continuum (2025), arXiv:2507.15644 [physics.optics].

[16] M. Verde and P. A. Huidobro, Optical response by time-varying plasmonic nanoparticles (2025), arXiv:2508.21009 [cond-mat.mes-hall].

Eq. (5). Furthermore, $S_{F}$ also satisfies Eq. (11) for permittivity profiles $\epsilon(t)$ that are symmetric with respect to t = 0.

[17] W. Li and L. E. Reichl, Floquet scattering through a time-periodic potential, Phys. Rev. B 60, 15732 (1999).

[18] J. R. Zurita-Sánchez, P. Halevi, and J. C. Cervantes-González, Reflection and transmission of a wave incident on a slab with a time-periodic dielectric function $\epsilon(t)$ , Phys. Rev. A 79, 053821 (2009).

[19] J. S. Martínez-Romero, O. M. Becerra-Fuentes, and P. Halevi, Temporal photonic crystals with modulations of both permittivity and permeability, Phys. Rev. A 93, 063813 (2016).

[20] J. S. Martínez-Romero and P. Halevi, Parametric resonances in a temporal photonic crystal slab, Phys. Rev. A 98, 053852 (2018).

[21] P. A. Pantazopoulos and N. Stefanou, Layered optomagnonic structures: Time Floquet scattering-matrix approach, Phys. Rev. B 99, 144415 (2019).

[22] S. Buddhiraju, A. Dutt, M. Minkov, I. A. D. Williamson, and S. Fan, Arbitrary linear transformations for photons in the frequency synthetic dimension, Nat. Commun. 12, 2401 (2021).

[23] L. Fan, Z. Zhao, K. Wang, A. Dutt, J. Wang, S. Buddhiraju, C. C. Wojcik, and S. Fan, Multidimensional convolution operation with synthetic frequency dimensions in photonics, Phys. Rev. Appl. 18, 034088 (2022).

[24] J. C. Serra, E. Galiffi, P. A. Huidobro, J. B. Pendry, and M. G. Silveirinha, Particle-hole instabilities in photonic time-varying systems, Opt. Mater. Express 14, 1459 (2024).

[25] D. Globosits, J. Hüpfl, and S. Rotter, Pseudounitary Floquet scattering matrix for wave-front shaping in time-periodic photonic media, Phys. Rev. A 110, 053515 (2024).

[26] J. B. Pendry, Photon number conservation in time dependent systems, Opt. Express 31, 452 (2023).

[27] S. Longhi, PT-symmetric laser absorber, Phys. Rev. A 82, 031801 (2010).

[28] Y. D. Chong, L. Ge, and A. D. Stone, PT-symmetry breaking and laser-absorber modes in optical scattering systems, Phys. Rev. Lett. 106, 093902 (2011).

[29] H. Kazemi, M. Y. Nada, T. Mealy, A. F. Abdelshafy, and F. Capolino, Exceptional points of degeneracy induced by linear time-periodic variation, Phys. Rev. Appl. 11, 014007 (2019).

[30] T. T. Koutserimpas and R. Fleury, Electromagnetic fields in a time-varying medium: Exceptional points and operator symmetries, IEEE Trans. Antennas. Propag. 68, 6717 (2020).

[31] H. Li, S. Yin, E. Galiffi, and A. Alù, Temporal parity-time symmetry for extreme energy transformations, Phys. Rev. Lett. 127, 153903 (2021).

[32] R.-C. Zhang, S. Yang, Y. Sha, Z. Xie, and Y. Yang, Parity-time symmetry phase transition in photonic time-modulated media (2025), arXiv:2507.03337 [physics.optics].

[33] D. G. Baranov, A. Krasnok, and A. Alù, Coherent virtual absorption based on complex zero excitation for ideal

light capturing, Optica 4, 1457 (2017).

[34] A. Canós Valero, V. Bobrovs, T. Weiss, L. Gao, A. S. Shalin, and Y. Kivshar, Bianisotropic exceptional points in an isolated dielectric nanoparticle, Phys. Rev. Res. 6, 013053 (2024).

[35] A. Mostafazadeh, Pseudounitary operators and pseudounitary quantum dynamics, J. Math. Phys. 45, 932-946 (2004).

[36] H. Schomerus, Eigenvalue sensitivity from eigenstate geometry near and beyond arbitrary-order exceptional points, Phys. Rev. Res. 6, 013044 (2024).

[37] Ş. K. Özdemir, S. Rotter, F. Nori, and L. Yang, Parity–time symmetry and exceptional points in photonics, Nat. Mater. 18, 783 (2019).

[38] P. Ambichl, K. G. Makris, L. Ge, Y. Chong, A. D. Stone, and S. Rotter, Breaking of $\mathcal{PT}$ symmetry in bounded and unbounded scattering systems, Phys. Rev. X 3, 041030 (2013).

[39] Z. J. Wong, Y.-L. Xu, J. Kim, K. O'Brien, Y. Wang, L. Feng, and X. Zhang, Lasing and anti-lasing in a single cavity, Nat. Photonics 10, 796 (2016).

[40] V. P. Flynn, E. Cobanera, and L. Viola, Deconstructing effective non-Hermitian dynamics in quadratic bosonic Hamiltonians, New J. Phys. 22, 083004 (2020).

[41] S. Esterhazy, D. Liu, M. Liertzer, A. Cerjan, L. Ge, K. G. Makris, A. D. Stone, J. M. Melenk, S. G. Johnson, and S. Rotter, Scalable numerical approach for the steady-state ab initio laser theory, Phys. Rev. A 90, 023816 (2014).

[42] H. E. Türeci, A. D. Stone, L. Ge, S. Rotter, and R. J. Tandy, Ab initio self-consistent laser theory and random lasers, Nonlinearity 22, C1 (2008).

[43] Z. Sakotic, P. Stankovic, V. Bengin, A. Krasnok, A. Alú, and N. Jankovic, Non-Hermitian control of topological scattering singularities emerging from bound states in the continuum, Laser Photonics Rev. 17, 2200308 (2023).

[44] C. Guo, J. Li, M. Xiao, and S. Fan, Singular topology of scattering matrices, Phys. Rev. B 108, 155418 (2023).

[45] A. Mostafazadeh, Pseudo-Hermiticity and generalized PT- and CPT-symmetries, J. Math. Phys. 44, 974 (2003).

[46] T. T. Koutserimpas, A. Alù, and R. Fleury, Parametric amplification and bidirectional invisibility in PT-symmetric time-Floquet systems, Phys. Rev. A 97, 013839 (2018).

[47] G. Ptitcyn, A. Lamprianidis, T. Karamanos, V. Asadchy, R. Alaee, M. Müller, M. Albooyeh, M. S. Mirmoosa, S. Fan, S. Tretyakov, and C. Rockstuhl, Floquet–Mie theory for time-varying dispersive spheres, Laser Photonics Rev. 17, 2100683 (2023).

[48] C. W. Hsu, B. Zhen, A. D. Stone, J. D. Joannopoulos, and M. Soljačić, Bound states in the continuum, Nat. Rev. Mater. 1, 16048 (2016).

[49] Z. Sadrieva, K. Frizyuk, M. Petrov, Y. Kivshar, and A. Bogdanov, Multipolar origin of bound states in the continuum, Phys. Rev. B 100, 115303 (2019).

[50] N. Ustimenko, C. Rockstuhl, and A. B. Evlyukhin, Resonances in finite-size all-dielectric metasurfaces for light trapping and propagation control, Phys. Rev. B 109, 115436 (2024).

[51] P. Garg, A. G. Lamprianidis, D. Beutel, T. Karamanos, B. Verfürth, and C. Rockstuhl, Modeling four-dimensional metamaterials: a T-matrix approach to describe time-varying metasurfaces, Opt. Express 30, 45832 (2022).

[52] M. Khajavikhan, A. Simic, M. Katz, J. H. Lee, B. Slutsky, A. Mizrahi, V. Lomakin, and Y. Fainman, Thresholdless nanoscale coaxial lasers, Nature 482, 204 (2012).

[53] R. Tirole, S. Vezzoli, E. Galiffi, I. Robertson, D. Maurice, B. Tilmann, S. A. Maier, J. B. Pendry, and R. Sapienza, Double-slit time diffraction at optical frequencies, Nat. Physics 19, 999 (2023).

[54] E. Lustig, O. Segal, S. Saha, E. Bordo, S. N. Chowdhury, Y. Sharabi, A. Fleischer, A. Boltasseva, O. Cohen, V. M. Shalaev, and M. Segev, Time-refraction optics with single cycle modulation, Nanophotonics 12, 2221 (2023).

[55] I. R. Hooper, D. B. Phillips, and S. A. R. Horsley, Harnessing the frequency eigenchannels of ultrafast time-varying media (2025), arXiv:2508.12753 [physics.optics].

[56] V. Bacot, M. Labousse, A. Eddi, M. Fink, and E. Fort, Time reversal and holography with spacetime transformations, Nat. Physics 12, 972 (2016).

[57] B. Apffel and E. Fort, Frequency conversion cascade by crossing multiple space and time interfaces, Phys. Rev. Lett. 128, 064501 (2022).

[58] J. E. Sustaeta-Osuna, F. J. García-Vidal, and P. A. Huidobro, Quantum theory of photon pair creation in photonic time crystals, ACS Photonics 12, 1873 (2025).

[59] M. F. Maghrebi, R. Golestanian, and M. Kardar, Scattering approach to the dynamical Casimir effect, Phys. Rev. D 87, 025016 (2013).

[60] V. V. Dodonov, Current status of the dynamical Casimir effect, Phys. Scr. 82, 038105 (2010).

[61] H. Haken, Light: Laser light dynamics (North-Holland, Amsterdam, 1985).

[62] V. Asadchy, A. Lamprianidis, G. Ptitcyn, M. Albooyeh, Rituraj, T. Karamanos, R. Alaee, S. Tretyakov, C. Rockstuhl, and S. Fan, Parametric Mie resonances and directional amplification in time-modulated scatterers, Phys. Rev. Appl. 18, 054065 (2022).

[63] P. Lalanne, W. Yan, K. Vynck, C. Sauvan, and J.-P. Hugonin, Light interaction with photonic and plasmonic resonances, Laser Photonics Rev. 12, 1700113 (2018).

[64] T. Weiss and E. A. Muljarov, How to calculate the pole expansion of the optical scattering matrix from the resonant states, Phys. Rev. B 98, 085433 (2018).

[65] A. G. Lamprianidis, Generalized transition matrix methods for the analysis of linear nanophotonic systems, Ph.D. thesis, Karlsruher Institut für Technologie (KIT) (2024).

[66] P. Waterman, T-matrix methods in acoustic scattering, J. Acoust. Soc. Am. 125, 42 (2009).

[67] D. Beutel, A. Groner, C. Rockstuhl, and I. Fernandez-Corbaton, Efficient simulation of biperiodic, layered structures based on the T-matrix method, J. Opt. Soc. Am. B 38, 1782 (2021).

[68] A. Novitsky, D. Lyakhov, D. Michels, A. A. Pavlov, A. S. Shalin, and D. V. Novitsky, Unambiguous scattering matrix for non-Hermitian systems, Phys. Rev. A 101, 043834 (2020).
