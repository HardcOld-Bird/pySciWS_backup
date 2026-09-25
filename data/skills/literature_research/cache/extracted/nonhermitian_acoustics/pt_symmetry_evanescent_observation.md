https://doi.org/10.1038/s42005-024-01816-1

# Observation of parity-time symmetry for evanescent waves

Check for updates

Zhaoxian Chen $^{1,7}$ , Huan He $^{2,7}$ , Huanan Li $^{2}$ $\boxtimes$ , Meijie Li $^{2}$ , Jun-long Kou $^{3,4}$ , Yan-qing Lu $^{1}$ $\boxtimes$ , Jingjun Xu $^{2}$ $\boxtimes$ & Andrea Alù $^{5,6}$ $\boxtimes$

Parity-time (PT) symmetry has enabled the demonstration of fascinating wave phenomena in non-Hermitian systems characterized by precisely balanced gain and loss. Until now, the exploration and observation of PT symmetry in scattering settings have largely been limited to propagating waves. Here, we demonstrate a versatile coupled-resonator acoustic waveguide (CRAW) system that enables the observation of PT-symmetric scattering responses for evanescent waves within a bandgap. By examining the generalized scattering matrix in the evanescent wave regime, we observe hallmark PT-symmetric phenomena—including phase transitions at an exceptional point, anisotropic transmission resonances, and laser-absorber modes—in systems that do not require balanced distributions of gain and loss. Owing to the peculiar energy transfer features of evanescent waves, our results not only demonstrate a distinct pathway for observing PT symmetry, but also enable strategies for exotic energy tunneling mechanisms, paving fresh directions for wave engineering grounded in non-Hermitian physics.

Originally formulated for quantum mechanics, parity-time (PT) symmetry has been extensively explored in classical wave settings over the past two decades, significantly reshaping the paradigm for wave control $^{1-7}$ . In these systems, a canonical approach involves spatially balancing gain (amplification) and loss (dissipation), ensuring that the associated Hamiltonian $H^{(PT)}$ satisfies the commutation relation $\left[H^{(PT)}, PT\right]=0$ , with P and T denoting the parity and time-reversal operators, respectively. Exploiting the unique properties of PT symmetry, along with spontaneous symmetry breaking and real-to-complex spectral phase transitions, an array of interesting applications beyond the capabilities of traditional Hermitian structures has emerged, including single-mode lasing $^{8,9}$ , wireless power transfer $^{10}$ and edge state engineering $^{11}$ , among others. To address the complexity of gain generation, including instabilities and noise, strategies such as passive PT symmetry $^{12-14}$ and anti-PT symmetry $^{15-20}$ have been introduced. The former, by introducing an overall offset of loss, enables PT-symmetric phase transitions in a weak form, while the latter, operating under an anti-PT-symmetric Hamiltonian $H^{(APT)}$ that satisfies $\left\{H^{(APT)}, PT\right\}=0$ , does not require gain, but displays features markedly different from those of conventional PT-symmetric systems.

In unbounded scattering systems, the response can be characterized by a scattering matrix S that relates incoming and outgoing waves $^{21-23}$ . In PTsymmetric systems, S satisfies the fundamental relation $^{21}$ $PTSPT = S^{-1}$ . While in the PT-symmetric phase the S matrix eigenvalues are unimodular, i.e., they reside on the complex unit circle as in a Hermitian system, upon PT symmetry breaking they become inverse-conjugate pairs, signaling a phase transition in the underlying Hamiltonian $H^{(PT)22,24}$ . In scattering contexts, PT symmetry enables novel forms of wave manipulation, associated with intriguing features such as laser-absorber modes $^{25}$ and anisotropic transmission resonances (ATRs) $^{24,26}$ . For instance, PT-symmetric systems operating in the laser-absorber mode can simultaneously act as lasers and coherent perfect absorbers (CPAs) as a function of the excitation conditions, as experimentally demonstrated both in the optical $^{27,28}$ and radio-frequency domains $^{29}$ . Similarly, ATRs and one-way reflectionless responses have been successfully demonstrated in on-chip optical waveguides through the use of passive PT symmetry $^{30}$ . The offset of loss in these structures can be compensated by either adding a linear amplifier, or through virtual gain with complex-frequency excitations $^{31}$ .

So far PT symmetry in scattering phenomena has been explored in the most natural setting, involving propagating waves. Yet, recently we have theoretically explored the concept of PT symmetry for evanescent fields by applying the S-matrix formalism to near-field decaying waves $^{32}$ . The rationale for this extension is that the wave number and impedance of evanescent waves is imaginary-valued, hence their features can be mapped onto non-Hermitian responses. When mapped onto the physics of PT symmetry, their features may enable phase transitions and ATRs without the need for gain elements. In this work, we theoretically propose and experimentally demonstrate a versatile coupled-resonator acoustic waveguide (CRAW) system to observe evanescent wave manipulation and scattering control based on this generalized form of PT symmetry. By utilizing a customizable feedback circuit to introduce and control imaginary coupling coefficients, we create an anti-PT-symmetric defect in the CRAW, and demonstrate that in the evanescent wave regime this structure supports a PT-symmetric scattering response. Specifically, we retrieve the generalized scattering matrix for evanescent wave excitations and observe phase transition and ATR phenomena through an engineered passive scatterer. If we consider the active scenario in which the scatterer supports gain, our findings reveal that the scatterer can also align a zero and a pole of its scattering matrix on the real frequency axis, enabling the evanescent wave counterpart of laser-absorber modes. Given that individual evanescent waves do not carry energy (Methods and Supplementary Note 1), this configuration leads to exotic phenomena based on which single-sided excitations enable energy exchanges due to larger-than-unity reflections, while double-sided excitations at the scattering matrix's zero prevent energy exchanges. Overall, our findings demonstrate that PT-symmetry for evanescent waves enables a powerful platform for non-Hermitian physics, less constrained by energy conservation limitations, and offering innovative approaches for near-field manipulation and control.

b

## Results

## Geometry and anti-PT-symmetric scatterer

The proposed CRAW is composed of a chain of cuboid-shaped acoustic cavities connected through an electric feedback circuit, as shown in Fig. 1. Precisely calibrated to resonate at frequency $\omega_{c}/2\pi = 1606$ Hz, each cavity incorporates an in-phase feedback circuit that introduces gain to offset its intrinsic loss $^{33}$ . The inter-resonator coupling is achieved by detecting the sound in one cavity with a microphone and then feeding the signal to the speaker in the neighboring cavity $^{34-37}$ . Specifically, to enable precise control over the coupling phase and amplitude, the coupling circuits are designed with phase shifters and power amplifiers (see Fig. 1b and the Supplementary Note 2 for details). We set the coupling strength in the uniform waveguide to be $\kappa/2\pi = 9$ Hz, which can be exactly calibrated with the coupled mode theory in the Supplementary Note 3. The waveguide dispersion is described by $\hat{\omega} \equiv (\omega - \omega_{c})/\kappa = -2\cos q$ , where $\hat{\omega}$ is the dimensionless angular frequency and $q$ denotes the normalized wavenumber relative to the unit cell size $^{38}$ . When operating within the bandgap $\hat{\omega} \notin [-2, 2]$ , the wave number $q$ has a non-zero imaginary part, corresponding to evanescent waves with decaying amplitudes (experimental results are provided in the Supplementary Note 4), in contrast with propagating waves for $\hat{\omega} \in [-2, 2]$ (see Fig. 1a).

Similar to propagating waves, evanescent waves are scattered when a defect is introduced into the waveguide. Our acoustic platform provides an ideal environment to examine such scattering phenomena, enabling the demonstration of PT symmetry for evanescent waves. Specifically, by leveraging our setup flexibility, we engineered a defect with Hamiltonian $^{32}$ $H = \begin{bmatrix} \omega_{-} + j\gamma & -j\kappa_{r} \\ -j\kappa_{r} & \omega_{+} + j\gamma \end{bmatrix}$ , which consists of a pair of coupled cavities $\mp$ , with resonant frequencies $\omega_{\mp}$ , equal damping $\gamma > 0$ , and imaginary coupling strength $\kappa_{r} > 0$ . By defining $\omega_{\mp} = \triangle_{0}\mp\triangle$ with $\triangle_{0} \equiv (\omega_{+} + \omega_{-})/2$ as the average offset frequency of the two resonators, the defect Hamiltonian can be expressed as $H = \triangle_{0}I_{2} + H^{(APT)}$ , where $I_{2}$ is the $2 \times 2$ identity matrix and $H^{(APT)}$ is given by $H^{(APT)} = \begin{bmatrix} -\triangle + j\gamma & -j\kappa_{r} \\ -j\kappa_{r} & \triangle + j\gamma \end{bmatrix}$ . Consequently, the defect exhibits anti-PT symmetry $^{16}$ , with its Hamiltonian anticommuting with the joint PT operation, $\triangle_{0}$ being the center frequency, and where parity and time-reversal correspond to resonator exchange and complex conjugation, respectively. The eigenvalues of $H^{(APT)}$ , $\lambda_{1,2}^{(APT)} = j\gamma \pm \sqrt{\Delta^{2} - \kappa_{r}^{2}}$ , transition from purely imaginary to opposite complex conjugate, namely, $\lambda_{1}^{(APT),*} = -\lambda_{2}^{(APT)}$ , as the frequency detuning $\triangle$ exceeds the exceptional point (EP) at $\kappa_{r}$ , where eigenvalues and eigenvectors simultaneously coalesce. The EP demarks a phase transition from symmetric to symmetry-broken responses $^{17}$ , observable in a fully passive anti-PT symmetric defect meeting the passivity condition $\gamma \geq |\kappa_{r}|$ . This condition ensures that the dimer defect is lossless or lossy, by enforcing that the stored energy does not increase in time $^{32}$ . In Fig. 2a, by holding constant the parameters $\gamma = \kappa_{r} = \kappa\sqrt{5}/2$ and $\triangle_{0} = \omega_{c} + 3\kappa/2$ , we experimentally retrieve the eigenvalues of the Hamiltonian $H^{(APT)}$ of the defect and observe this phase transition as $\triangle$ varies, by fitting the measured excitation spectra for each $\triangle$ , i.e., the pressure amplitudes and relative phases versus the excitation frequency. As an example, we present the measured excitation spectra in the two cavities for $\triangle/\kappa_{r} = 1$ in Fig. 2b, c, which closely match the theoretical predictions. By performing these measurements for a wide range of $\triangle/\kappa_{r}$ , we retrieved the eigenvalue spectra of $H^{(APT)}$ in Fig. 2a, clearly demonstrating the phase transition and anti-PT-symmetric response.

![](images/b2153defd751f2c79075f4ca244441ed0280a57a3f0cadb545b4678e70f4a003.jpg)

![](images/697f5c3b2dab9985fcf4732f037fc77abee80253f6aa33b5b4334f2543a1e41e.jpg)
Fig. 1 | Coupled-resonator acoustic waveguide (CRAW) system for observing evanescent-wave scattering. a Schematic of the CRAW with resonators labeled from -3 to 3, each coupled adjacently with $-\kappa$ at the ports. The port's dispersion relation illustrates that selecting a frequency at the bandgap (red star in the pink region) excites an evanescent wave, characterized by the exponentially decaying profile, in contrast to the oscillating profile of a propagating wave excited within the band (black star in the gray area). Embedding a dimer defect into the otherwise uniform waveguide enables the scattering of evanescent waves. b Schematic of the acoustic cavities with feedback circuits, implementing the coupling of resonators as indicated by the blue dashed box in (a). Mutual coupling between cavities is achieved by signal routing through microphones, phase shifters (PSs), amplifiers, and speakers (connected with blue lines), while in-phase feedback circuits (connected with red lines) allow for precise control of the cavities' damping rates.

PT-symmetric phase transition and ATRs for evanescent waves

A PT-symmetric Hamiltonian, $H^{(PT)}$ , can be associated with $H^{(APT)}$ of the anti-PT-symmetric dimer defect by multiplying it with the imaginary unit, yielding $H^{(PT)} = -jH^{(APT)} = \begin{bmatrix} \gamma + j\triangle & -\kappa_r \\ -\kappa_r & \gamma - j\triangle \end{bmatrix}$ . Here, the resonance frequency detuning $\triangle$ between the resonators acts as effective gain and loss. Intriguingly, the required imaginary unit can be accessed through the excitation: by impinging with evanescent waves, thanks to their imaginary-valued wave number and impedance, we can expect the scattering from an anti-PT-symmetric scatterer to be PT-symmetric in nature. To this end, we investigate the scattering phenomena within the CRAW bandgap in the presence of the anti-PT symmetric defect at site 0 [see Fig. 1a]. In steady state, the uniform arrays on the left and right of the defect, i.e., the CRAW ports, enable the excitation and scattering measurement of evanescent waves, which decay towards and away from the scatterer, respectively. Correspondingly, the wave amplitudes $\psi_{\sigma}(n,t)$ at the left ( $\sigma=L$ ) and right ( $\sigma=R$ ) port are $\psi_{\sigma}(n,t)=\sigma^{(i)}e^{j\omega t-jqn}+\sigma^{(o)}e^{j\omega t+jqn}$ , with a symmetric labeling scheme where, at the right port, the cavity sites with $n\geq1$ [see Fig. 1a] are relabeled as -n. By linking the evanescent wave amplitudes $\sigma^{(b)}$ for input (b=i) and output (b=o) ports, we define the generalized scattering matrix S as $\binom{L^{(o)}}{R^{(o)}}=S\binom{L^{(i)}}{R^{(i)}}\equiv\binom{r_{L}}{t_{RL}}\binom{t_{LR}}{r_{R}}\binom{L^{(i)}}{R^{(i)}}$ . This matrix describes the scattering of evanescent waves, providing a valuable tool for investigating near-field physics $^{39}$ . For the anti-PT symmetric scatterer with evanescent wave excitation at $\hat{\omega}>2$ , the generalized $S(\hat{\omega})$ matrix is given by $^{32}$

![](images/e6baeccf56b5cee1e4204cc8cd8c48ca79e0c443f877447bafef221a0ee61a97.jpg)

![](images/ff689985588e64cba924ddcf654f42881937658bfe10d2669b426ae1e5dce06b.jpg)
Fig. 2 | Phase transition in the isolated anti-PT symmetric dimer defect. a Real (red circles) and imaginary (blue circles) parts of the experimentally obtained eigenvalues $\lambda_{1,2}^{(APT)}$ (in units of $\kappa$ ) for the passive anti-PT-symmetric dimer defect, alongside theoretical curves, versus resonant frequency detuning $\triangle$ . The light green region indicates the symmetric phase, while the light blue region represents the

$$
S (\hat {\omega}) = - I _ {2} + 2 j c _ {r} (\hat {\omega}) \Big [ H _ {e f f} (\hat {\omega}) + j c _ {r} (\hat {\omega}) I _ {2} \Big ] ^ {- 1},\tag{1}
$$

where $c_{r}(\hat{\omega}) \equiv \hat{c}^{2}\sqrt{\hat{\omega}^{2}-4}/2$ with $\hat{c} = c/\kappa$ being the normalized coupling between the scatterer and the ports. Here, the scatterer effective Hamiltonian is $H_{eff}(\hat{\omega}) = j\varepsilon_{\omega}(1 - \hat{c}^{2}/2)I_{2} + H^{(PT)}/\kappa$ , with PT-symmetric portion $H^{(PT)}$ , and the frequency detuning $\varepsilon_{\omega} \equiv \hat{\omega} - \hat{\omega}_{0}$ where $\hat{\omega}_{0} = (\Delta_{0} - \omega_{c}) / [\kappa(1 - \hat{c}^{2}/2)]$ . When $\hat{\omega} = \hat{\omega}_{0} > 2$ , namely $\varepsilon_{\omega} = 0$ , the effective Hamiltonian $H_{eff}(\hat{\omega}_{0}) = H^{(PT)}/\kappa$ commutates with the joint PT operation. Consequently, the generalized $S(\hat{\omega}_{0})$ matrix in Eq. (1) fulfills the fundamental relations $PTS(\hat{\omega}_{0})PT = S(\hat{\omega}_{0})^{-1}$ , which obeys PT symmetry. We note that PT symmetry here is obeyed at an isolated frequency, similar to conventional PT symmetry for propagating waves in optical scattering systems, as required due to causality constraints $^{40}$ . Furthermore, this PT-symmetric response driven by evanescent wave excitations can occur at real frequencies for stationary scattering processes based on a purely passive scatterer ( $y \geq |\kappa_{r}|$ ).

Now, we explore the implications of a PT-symmetric response on evanescent wave scattering in our acoustic platform. By measuring the reflection and transmission for evanescent wave excitation, we can readily construct the scattering matrix $S(\hat{\omega}_{0})$ (see details in Supplementary Note 1). Using the same parameters from Fig. 2a for a passive scatterer, Fig. 3a displays the evolution of measured eigenvalues of the $S(\hat{\omega}_{0})$ matrix, going from unimodular to inverse-conjugate pairs through an EP, as the resonant frequency detuning $\triangle$ varies from 0 to $2\kappa_{r}$ . This qualitative shift marks the system transition from PT-symmetric to PT-broken phase. Correspondingly, by rearranging Eq. (1), we can extract the effective Hamiltonian $H_{eff}(\hat{\omega})$ from the $S(\hat{\omega})$ matrix:

c
![](images/692565a13d7cb6a7544544935b5e196e685e83570e6aa6a4a5a2658bbe9f941e.jpg)
symmetry-broken phase. Measured (symbols) and fitted (curves) excitation spectra (b) and relative phase (c) of the cavities ± within the dimer defect. The sound source for the excitation is positioned in cavity +, and the system parameters are $\gamma = \kappa_{r} = \kappa\sqrt{5}/2$ , $\Delta/\kappa_{r} = 1$ and $\Delta_{0} = \omega_{c} + 3\kappa/2$ , with $\kappa/2\pi = 9$ Hz and $\omega_{c}/2\pi = 1606$ Hz.

$$
H _ {e f f} (\hat {\omega}) = - j c _ {r} (\hat {\omega}) I _ {2} + 2 j c _ {r} (\hat {\omega}) \left[ S (\hat {\omega}) + I _ {2} \right] ^ {- 1},\tag{2}
$$

for which the experimentally obtained eigenvalues (circles) when $\hat{\omega} = \hat{\omega}_{0}$ , illustrated in Fig. 3b, closely align with the theoretical prediction (lines) $\lambda_{1,2}^{(PT)} = \left(\gamma \pm \sqrt{\kappa_{r}^{2} - \triangle^{2}}\right)/\kappa$ for an effective PT-symmetric scatterer with Hamiltonian $H^{(PT)}/\kappa$ . In contrast with the anti-PT-symmetric scenario (Fig. 2a), the eigenvalues $\lambda_{1,2}^{(PT)}$ of the effective PT-symmetric scatterer are real when $\triangle < \kappa_{r}$ (PT-unbroken phase), and they become complex conjugate for $\triangle > \kappa_{r}$ (PT broken phase), see Fig. 3b.

The PT-symmetric scattering matrix $S(\hat{\omega}_{0})$ obeys a pseudo-unitary conservation relation $^{24}$ $|r_{L}r_{R}|=||t_{S}|^{2}-1|$ , where the transmission amplitude $t_{S}\equiv t_{LR}=t_{RL}$ . In particular, we may seek a condition for the emergence of ATR, which in PT-symmetric systems implies unimodular transmission in both directions, with zero reflection only from one side. Based on Eq. (1), we achieve ATR at $\hat{\omega}_{0}$ for a passive scatterer under evanescent wave excitations by selecting $\gamma=\kappa_{r}$ and $\Delta=\kappa c_{r}(\hat{\omega}_{0})$ . In Fig. 3c,d, we show the extracted magnitude of reflection and transmission coefficients from 1624 to 1642 Hz, corresponding to the frequency range of $\hat{\omega}\in(2,4)$ for evanescent waves. The observed scattering responses (circles) match theoretical predictions (lines), with lower frequency deviations due to residual port resonator loss in the experiment (see details in Supplementary Note 5). Unlike different reflections observed for excitations from opposite directions, the transmission amplitudes are equal due to reciprocity, albeit with small mismatches at higher frequencies due to the fast evanescent wave decay. At the frequency of 1633 Hz (vertical dashed lines), corresponding to $\hat{\omega}=\hat{\omega}_{0}$ , the transmission nearly reaches unity on both sides of the acoustic lattice, with pronounced reflection on the right and virtually none on the left, validating the hallmark ATR phenomenon. This unidirectional, reflection-free behavior in the presence of evanescent wave excitations is captured in the pressure profiles displayed in Fig. 3e,f, illustrating the lattice response to right and left side excitations, respectively. Indeed, when the incidence is from the right, the field profile at the right port includes both incident and reflected evanescent waves due to the large reflection ( $|r_{R}|\approx2$ ), leading to enhanced total pressure field at the cavity + within the dimer defect compared to the non-reflective case (dashed line) that involves only the impinging wave [Fig. 3e]. In contrast, for the left excitation the pressure profile is identical to the non-reflective case (dashed line) [Fig. 3f], since $|r_L| \approx 0$ .

a
![](images/fda06abac93510ce905b2699614c1e35b7faea4d5369a8cb6a94e7e0310112c4.jpg)

b
d
![](images/6e2cfff872bc36d48785fbd3ffcfc4bf84c012047da9a6c8d1fa505ac0a4f3a6.jpg)
Fig. 3 | PT-symmetric responses for incident evanescent waves. Evolution of the eigenvalues for (a) the generalized scattering matrix $S(\hat{\omega}_0)$ and (b) the corresponding effective PT-symmetric Hamiltonian $H^{(PT)} / \kappa$ at the excitation frequency of $\omega / 2\pi = 1633\mathrm{Hz}$ for evanescent wave excitations in the CRAW system, with the resonant frequency detuning $\triangle$ of the passive dimer defect increasing from 0 to $2\kappa_r$ (marked with the gray arrows). The lines indicate calculated results and symbols denote measured data for both panels. In (a), the red arrow marks the exceptional point (EP) at $\triangle = \kappa_r$ for the PT-symmetric phase transition, and the dashed curve represents the unit circle. In (b), the light green and blue regions indicate the symmetric and symmetry-broken phases, respectively. Experimentally measured (circles) and theoretically calculated (lines) spectra of (c) reflection and (d)

## Laser-absorber mode for evanescent waves

Our demonstration of ATR for evanescent waves based on a passive scatterer reveals that the total scattering strength, measured by transmittance $T_{S} = |t_{S}|^{2}$ and reflectances $R_{L(R)} = |r_{L(R)}|^{2}$ , is not bounded by the power conservation constraints that limit propagating waves in passive systems. This is because evanescent waves individually do not carry energy (see Methods), setting an exotic stage for extreme wave interactions between evanescent waves and scatterers, when the poles of the generalized $S(\hat{\omega})$ matrix lie close or on the real frequency axis. Under this condition, unbounded $T_{S}$ and/or $R_{L(R)}$ can be achieved at evanescent wave spectral singularities $^{41}$ , even in passive scenarios. Moreover, the PT-symmetry of $S(\hat{\omega}_{0})$ for evanescent waves can result in poles and zeros coinciding on the real frequency axis, mirroring the behavior of laser-absorber modes found in conventional PT-symmetric systems for propagating waves. In our setup, we achieve this condition by setting the frequency detuning $\triangle$ and the damping $\gamma$ to $\Delta_{PZ} = \sqrt{\kappa_{r}^{2} + \kappa^{2}c_{r}^{2}(\hat{\omega}_{0})}$ and $\gamma_{PZ} = 0$ , respectively. This behavior is captured in Fig. 4a, b, where, as $\gamma$ varies for $\triangle$ fixed at $\Delta_{PZ}$ , the poles and zeros (solid lines) evolve in the complex frequency plane according to $\det[H_{eff}(\hat{\omega}) \pm jc_{r}(\hat{\omega})I_{2}] = 0$ from Eq. (1). The lack of PT symmetry for $\hat{\omega} \neq \hat{\omega}_{0}$ means that the poles and zeros are not complex conjugate pairs, leading to asymmetric paths. However, at $\hat{\omega} = \hat{\omega}_{0}$ (red stars) when $\gamma$ equals $\gamma_{PZ}$ , they intersect on the real frequency axis simultaneously, demonstrating the PT-symmetric nature of $S(\hat{\omega}_{0})$ .

![](images/852b3dbba91321e4ec82d814786a310e2e6c09e914d2fe95f2a8b888dee79021.jpg)

![](images/db0e86a0666f259a309b4f0e3ef4083c5de30fdcdf1d682efe39f1f2b2565a8b.jpg)

![](images/6a575adae9df459d745cb5cf5c1ee1fe0efd7378458922e918058f7b37342980.jpg)

f
![](images/e08e473452ae099258562b52938d2d8cbd518f415f0b6fc10870aef3ee83d3f6.jpg)
transmission for evanescent wave excitation from the right (green) and left (orange) at $\Delta = \kappa_{r}$ . The vertical dashed lines mark the excitation frequency at $\omega/2\pi = 1633$ Hz. Logarithmic representation of acoustic pressure profiles, both measured and simulated, for evanescent wave excitation from (e) the right and (f) the left, with and without the defect scatterer in the CRAW, demonstrating the ATR phenomenon for evanescent waves associated with the vertical dashed lines in (c, d). The arrows mark the locations of excitation sources, while the shaded areas highlight the scattering regions containing cavities – and +. In this figure, the scatterer-port coupling is normalized to $\hat{c} \equiv c/\kappa = 1$ , and other system parameters are the same as in Fig. 2 for $\hat{\omega}_{0} = 3$ .

To explore the emergence of laser-absorber modes for evanescent waves, we analyze the eigenvalues and eigenvectors of the generalized S matrix, satisfying $S\left|\alpha_{\pm}\right\rangle=\alpha_{\pm}\left|\alpha_{\pm}\right\rangle$ . At $\hat{\omega}=\hat{\omega}_{0}$ with $\Delta=\Delta_{PZ}$ , the eigenvalues $\alpha_{\pm}$ and the associated eigenvectors $\left|\alpha_{\pm}\right\rangle$ can be derived from Eq. (1) as

$$
\alpha_ {\pm} = \left[ \left(- \gamma \pm 2 j \kappa c _ {r} (\hat {\omega} _ {0})\right) / \gamma \right] ^ {\pm 1},\tag{3a}
$$

$$
\left| \alpha_ {\pm} \right\rangle = \left[ 1, \pm j \big (\kappa c _ {r} (\hat {\omega} _ {0}) \pm \Delta_ {P Z} \big) / \kappa_ {r} \right] ^ {T}.\tag{3b}
$$

As the zero and pole of $S(\hat{\omega})$ coincide at real $\hat{\omega}_{0}$ for $\gamma$ approaching $\gamma_{PZ}=0$ , the eigenvalues $\alpha_{-}$ and $\alpha_{+}$ tend towards zero and infinity, respectively, and the corresponding eigenvectors $\left|\alpha_{\pm}\right\rangle$ span the input vector space of the scatterer. Consequently, when an evanescent wave at $\hat{\omega}_{0}$ impinges from either side, the non-zero projection onto $\left|\alpha_{+}\right\rangle$ results in an infinite scattering strength as $\gamma$ moves towards $\gamma_{PZ}$ .

This behavior resembles a scattering event common in laser physics; however, in terms of energy transfer, the real-frequency pole of $S(\hat{\omega})$ for evanescent wave excitations introduces intriguing features. Indeed, in our configuration for evanescent waves incident from the left or the right, the transmitted waves do not carry energy; rather, it is the interference between incident and reflected waves that facilitates energy transfer. Specifically, the normalized energy flux $\hat{J}_{\sigma}$ , directed towards the scatterer from the source at excitation port $\sigma = L$ , $R$ , is determined by $\hat{J}_{\sigma} = 2\sqrt{\hat{\omega}^2 - 4\mathrm{Im}(r_{\sigma})}$ (see Methods). In Fig. 4c, d, we present a color map of $\hat{J}_L$ and $\hat{J}_R$ on a base-10 logarithmic scale, plotted against the detuning parameters $\varepsilon_{\gamma} = \gamma - \gamma_{PZ}$ for damping and $\varepsilon_{\omega} = \hat{\omega} - \hat{\omega}_0$ for the excitation frequency, focused around the real-frequency pole. As $\gamma$ decreases and $\varepsilon_{\gamma}$ is less than $\kappa_{r}$ , the scatterer transitions to becoming active; with $\hat{J}_{R}$ remaining positive to absorb energy, $\hat{J}_{L}$ can vary from positive to negative, enabling output of energy. Moreover, as $\varepsilon_{\gamma}$ approaches zero from above with $\varepsilon_{\omega}=0$ , $\hat{J}_{L}$ diverges to negative infinity, while $\hat{J}_{R}$ diverges to positive infinity, characterizing the scatterer as a left-side laser and a right-side absorber; inversely, when $\varepsilon_{\gamma}$ shifts to just below zero, the operation of the scatterer are reversed.

a
C
![](images/271a48fd43c2c812fb4236d5af2c38d7270cd8c128be4a540b1abe405a693d41.jpg)

b
d
![](images/5be1bbd7c97d85a22846c84b6a3082fbac82d7301b27fb8c19276b575b99c296.jpg)

![](images/7f26aba2926765c6aa6750bd63d961a1edc4f91507a12642e5d08ac851d5db1b.jpg)
Fig. 4 | Occurrence of a laser-absorber mode for evanescent waves. Trajectories of the (a) pole and (b) zero of the generalized $S(\hat{\omega})$ matrix in the complex frequency plane, with the damping $\gamma$ in the anti-PT symmetric dimer defect decreasing from $\gamma / 2\pi = 2\mathrm{Hz}$ to $\gamma / 2\pi = -2\mathrm{Hz}$ , as depicted by the solid curves. The pole and zero coincide on the real frequency axis at $\omega / 2\pi = 1.633\mathrm{kHz}$ (red and blue stars) when $\gamma = \gamma_{PZ} = 0$ , indicating an evanescent wave laser-absorber mode. The pole's location in the upper (lower) plane indicates the system's transition to unstable (stable) states, as distinguished by light pink (light green) backgrounds. Dashed curves correspond to experimental scenarios with residual loss $\gamma_c / 2\pi = 1\mathrm{Hz}$ in the port

![](images/e86b254eb47cc11d9f11e6a85184d3b1099fbcaaa3350a01411f2e604db44be1.jpg)

In our experiments, due to residual damping in the port resonators, where $\gamma_{c}/2\pi=1Hz$ , the energy fluxes $\hat{J}_{L,R}$ shown in Fig. 4e, f demonstrate characteristics similar to the ideal scenarios in Fig. 4c, d. Due to the absence of exact PT symmetry here, the real-frequency pole and zero align at the same excitation frequency $\omega/2\pi=1633Hz$ when the scatterer's damping $\gamma$ diverges slightly between the pole $\gamma_{P}$ and the zero $\gamma_{Z}$ (where $\gamma_{Z,P}\approx\hat{c}^{2}\gamma_{c}\left[1/2\pm\hat{c}^{2}\hat{\omega}_{0}c_{r}^{-1}\left(\hat{\omega}_{0}\right)/4\right]$ ), both set against a common adjusted value $\Delta_{PZ}^{\prime}=\Delta_{PZ}+O(\gamma_{c}^{2}/\kappa^{2})$ (see Supplementary Note 6). The corresponding trajectories for both the pole and zero, as $\gamma$ varies, are shown by the dashed curves in Fig. 4a, b. The small difference between the ideal scenarios (solid lines), featuring precise PT symmetry [see Eq. 3], and practical experiments (dashed lines) become evident in Fig. 5a, which plots $\log_{10}\left|\alpha_{\pm}\right|$ for the eigenvalues $\alpha_{\pm}$ of the generalized S matrix against $\gamma$ at a fixed excitation frequency $\omega/2\pi=1633Hz$ . To experimentally observe the laser-absorber operation for evanescent waves (see Supplementary Note 7), we set $\gamma$ to $\gamma_{exp}/2\pi=1Hz$ [see Fig. 5a], thereby shifting the pole marginally off the real frequency axis and ensuring system stability. The experimental performance can be quantified by the overall output coefficient $\Theta(\omega)=(|L^{(o)}|^{2}+|R^{(o)}|^{2})/(|L^{(i)}|^{2}+|R^{(i)}|^{2})$ , i.e., the squared amplitude ratio of scattered to impinging waves $^{25}$ . When exciting with evanescent waves from the left or right, the measured $\Theta$ is 17.2 or 23.3 dB around the frequency $\omega/2\pi=1633Hz$ [see Fig. 5b], enabling energy exchange between the scatterer and the excitation. Due to intense scattering, the field profile shown in Fig. 5c demonstrates growth at the scatterer under left incidence, in contrast with the ATR shown in Fig. 3f. In the same experimental setup, by employing coherent sources for bilateral excitation as indicated by the input $\left|\alpha_{-}\right\rangle$ in Eq. (3b) to excite the S matrix zero, the scatterer effectively absorbs the impinging evanescent waves [see Fig. 5d], with output coefficient $\Theta$ decreasing now to -35 dB [Fig. 5b]. This scattering behavior mirrors the one of CPAs for propagating waves; but with distinct energy transfer mechanisms given the evanescent wave excitation. Specifically, by engaging the CPA for evanescent waves, we effectively halt energy transfer between the excitation sources and the scatterer, as the impinging evanescent wave alone does not transport energy (Methods). The demonstrated CPA phenomenon for evanescent waves relies on coherent inputs from the two sources, offering a promising strategy for high-performance sensors by utilizing the large output contrast of the laser-absorber mode $^{29}$ .

![](images/170c53313f304d97a83eec36ed6d6336cfad3578cfad946b36102edff77f790e.jpg)

f
![](images/966ec378e1b68b4285d4d47a6b2b05bdfd194ede81803b4ba7100f56511a34d7.jpg)
resonators. Density plot of the base-10 logarithm of the normalized energy flux magnitude for (c) left $(\hat{J}_{L})$ and (d) right $(\hat{J}_{R})$ excitation, plotted against the detunings $\varepsilon_{\gamma}$ and $\varepsilon_{\omega}$ of the damping $\gamma$ and normalized excitation frequency $\hat{\omega}$ , relative to the laser-absorber mode. The shaded regions indicate negative energy flux directed from the scatterer to the excitation source. (e, f) Correspond to (c, d), respectively, but display the experimental scenarios with observed energy flux between the scatterer and the excitation port. Common parameters include $\Delta = \Delta_{PZ}$ , with other unspecified system parameters the same as in Fig. 3.

## Conclusions

In this study, we demonstrated the observation of PT symmetry for evanescent wave excitations within a versatile CRAW system operated within a bandgap. The CRAW system features electrically assisted coupling between acoustic cavities, offering extreme tunability and facilitating experimental investigation into evanescent wave scattering. Our results demonstrate that, when extended to the evanescent wave regimes, a stationary PT-symmetric scattering response can be achieved based on an anti-PT symmetric defect scatterer, without requiring balanced gain and loss. In this configuration, the resonance frequency detuning between dimer defect resonators acts as effective gain and loss, allowing us a precise control over PT-symmetric phase transitions at an EP and the observation of unidirectional invisibility for evanescent waves through ATR-based scattering, using a passive and inherently stable scatterer. When the scatterer enters the active regime, our platform can support evanescent laser-absorber modes by aligning the generalized scattering matrix zero and pole on the real frequency axis, yielding amplified scattering from single-side excitation and zero reflection for coherent inputs, with an output contrast ratio surpassing 58 dB, only limited by residual loss at the port resonators. Thanks to a distinctive energy transfer mechanism unique to evanescent waves, where energy flux stems from interference between counter-decaying waves, this evanescent laser-absorber mode offers a exotic strategy for highly controllable energy exchanges, allowing for either infinite or zero energy flux, with reversible flow direction, determined through coherent excitation. Unlike complex frequency excitations, which can introduce an effective gain to overcome loss in the time domain $^{31,42-46}$ , our experimental platform utilizing evanescent wave excitations enhances the scattering strength in a stationary process, independent of energy considerations. Not only it provides an exotic method for observing PT symmetry, avoiding instabilities and noise typical of gain systems, but also sheds light on the manipulation of evanescent wave scattering through non-Hermitian physics, with implications for functional devices and applications in near-field physics, including sensing, communication technology, scanning near-field optical microscopy and radiative heat transfer.

a
C
b
![](images/376a2e1e3d49b79309549ac748d43295a03620abf783ff541f03a1aa8a19ed0a.jpg)

![](images/416e31ede374f060f5fb6172ca9c60b9a9a9b75522beae5c6c4f7ed2f74b7559.jpg)
d

![](images/ef1c7ba0cc4e4ceeee3c0970a6fe3e525780f092f2a4e444f5f831ff4275f15c.jpg)

![](images/f6fa9528c7ab1eef4902661897eeb40e25f60cd76d460ce3f934adf553dd10ef.jpg)
Fig. 5 | Experimental observation of the evanescent wave laser-absorber mode.

a Logarithmic plot of the eigenvalues $\alpha_{\pm}$ (red and blue solid lines) of the generalized $S(\hat{\omega}_{0})$ matrix versus the damping $\gamma$ , showcasing the evanescent wave laser-absorber mode indicated in Fig. 4. The blue and red dashed lines correspond to the experimental scenarios [see Fig. 4], while the vertical dashed lines indicate the critical damping values $\gamma_{PZ} = 0$ , $\gamma_{P}/2\pi = -0.17$ Hz, $\gamma_{Z}/2\pi = 1.17$ Hz, and $\gamma_{exp}/2\pi = 1$ Hz discussed in the main text.

b Theoretical (curves) and experimental (circles) results for the overall output coefficient $\Theta$ versus excitation frequency, considering left (red), right (orange) incident evanescent waves, and bilateral excitation fulfilling $|\alpha_{-}\rangle$ (blue), under the same experimental scenarios as in (a) with the dimer defect's damping coefficient set to $\gamma = \gamma_{exp}$ . Logarithmic representation of measured and simulated acoustic pressure profiles for (c) left incidence and (d) bilateral excitation fulfilling $|\alpha_{-}\rangle$ , associated with $\Theta$ extremes of 17.2 dB and -36.4 dB, respectively, at frequency 1.633 kHz in (b). The arrows mark the locations of excitation sources, while the shaded areas highlight the scattering regions containing cavities – and +. Other unspecified system parameters are the same as in Fig. 4.

## Methods

## Energy flux in the CRAW ports

Here, we derive an expression for the energy flux into the CRAW port using coupled mode theory, considering the residual damping $\gamma_{c}$ in each port resonator, which yields the complex resonant frequency as $u_{c} = \omega_{c} + j\gamma_{c}$ . With the symmetric labeling scheme [where sites with $n \geq 1$ at the right port (see Fig. 1) are relabeled as $-n$ , we can write the Schrödinger-like coupled-mode equations as

$$
j \frac {d \psi_ {\sigma} (n , t)}{d t} = - u _ {c} \psi_ {\sigma} (n, t) + \kappa \psi_ {\sigma} (n + 1, t) + \kappa \psi_ {\sigma} (n - 1, t),\tag{4}
$$

for the energy normalized time-dependent field amplitudes $\psi_{\sigma}(n,t)$ at sites $n\leq-2$ for both ports (left $\sigma=L$ and right $\sigma=R$ ), where $\kappa$ is the coupling strength within the ports. Derived from Eq. (4), the continuity equation governing energy flow is given by

$$
\frac {d \left| \psi_ {\sigma} (n , t) \right| ^ {2}}{d t} = J _ {n - 1 \rightarrow n} ^ {(\sigma)} - J _ {n \rightarrow n + 1} ^ {(\sigma)} - 2 \gamma_ {c} \left| \psi_ {\sigma} (n, t) \right| ^ {2},\tag{5}
$$

where $J_{n\to n+1}^{(\sigma)} \equiv 2\kappa\mathrm{Im}\left[\psi_{\sigma}(n,t)\psi_{\sigma}^{*}(n+1,t)\right]$ represents the energy flux flowing from site n to $n+1$ in port $\sigma$ . As expected, the rate of change in total energy at site n corresponds to the net inflow of energy, $J_{n-1\to n}^{(\sigma)} - J_{n\to n+1}^{(\sigma)}$ , minus the energy dissipation due to damping loss $\gamma_{c}$ . This damping $\gamma_{c}$ imparts a position-dependent characteristics to the energy flux $J_{n\to n+1}^{(\sigma)}$ in steady state. However, when $\gamma_{c}$ is sufficiently small to be considered negligible, this flux transitions to a position-independent regime, resulting in a uniform steady-state energy flux $J^{(\sigma)}$ .

For the derivation of the uniform flux $J^{(\sigma)}$ , we assume $\gamma_c = 0$ and apply the general steady-state solution of Eq. (4), $\psi_\sigma(n,t) = \sigma^{(i)}e^{j\omega t - jqn} + \sigma^{(o)}e^{j\omega t + jqn}$ [see the main text], into the expression for $J_{n\to n + 1}^{(\sigma)}$ . When the normalized frequency $\hat{\omega} \equiv (\omega - \omega_c)/\kappa \in [-2,2]$ , the wave number $q$ is real. In this case, the general solution $\psi_\sigma(n,t)$ consists of two counterpropagating waves, and the associated energy flux is $J^{(\sigma)} = 2\kappa \sin q(|\sigma^{(i)}|^2 - |\sigma^{(o)}|^2)$ . When $\hat{\omega} > 2$ and thus $\sin q = j\sqrt{\hat{\omega}^2 - 4}/2$ [see the main text], the wave number $q$ attains an nonzero imaginary component, with its real part $\mathrm{Re}(q)$ fixed at $\pi$ . Consequently, the general solution $\psi_\sigma(n,t)$ involves two counter-decaying evanescent waves, and the associated energy flux becomes $J^{(\sigma)} = 2\kappa\sqrt{\hat{\omega}^2 - 4}\mathrm{Im}\big(\sigma^{(i)},*\sigma^{(o)}\big)$ . Unlike propagating waves, the energy flux $J^{(\sigma)}$ for evanescent waves is nonzero only when both forward $\sigma^{(i)}$ and backward $\sigma^{(o)}$ waves are present simultaneously. As a result, for evanescent wave incidence from the port $\sigma = L,R$ , the single transmitted evanescent wave does not carry energy in steady state. However, due to reflection $r_\sigma$ , the incident and reflected wave, with amplitudes $\sigma^{(i)}$ and $\sigma^{(o)} = \sigma^{(i)}r_\sigma$ respectively, coexist at the excitation port, leading to the energy flux $J^{(\sigma)} = \kappa|\sigma^{(i)}|^2\hat{J}_\sigma$ , where $\hat{J}_\sigma = 2\sqrt{\hat{\omega}^2 - 4}\mathrm{Im}(r_\sigma)$ represents the normalized energy flux.

## Sample preparation

The cavity is made from stainless steel through precise machining. The top cover is made from acrylic board and is equipped with a microphone and loudspeakers for detecting the signals and introducing sound, respectively. The effective coupling is realized with a feedback circuit containing the following elements. The power amplifier (TPA3116) has two parallel channels and, thus, can tune the coupling strength of the two mutual channels simultaneously. The phase shifter (model MCP41010) is introduced to adjust the coupling phase.

## Acoustic measurements

For the measurement, a multi-channel audio analyzer RS1284 (from RSTECH Co., Ltd.) is utilized to generate the sound signals with the output generator module and to record the sound signals with the signal acquisition module. The sound signals inside the cavities are detected with the 1/4-inch microphones (RST1200).

## Data availability

The data that support the findings of this study are available from the corresponding authors upon reasonable request.

Received: 19 June 2024; Accepted: 28 September 2024; Published online: 16 October 2024

## References

1. Christodoulides, D. & Yang, J. Parity-Time Symmetry and Its Applications. (Springer, 2018).

2. Zhao, H. & Feng, L. Parity–time symmetric photonics. Natl Sci. Rev. 5, 183–199 (2018).

3. El-Ganainy, R. et al. Non-Hermitian physics and PT symmetry. Nat. Phys. 14, 11–19 (2018).

4. Miri, M.-A. & Alù, A. Exceptional points in optics and photonics. Science 363, eaar7709 (2019).

5. Özdemir, Ş. K., Rotter, S., Nori, F. & Yang, L. Parity–time symmetry and exceptional points in photonics. Nat. Mater. 18, 783–798 (2019).

6. Parto, M., Liu, Y. G. N., Bahari, B., Khajavikhan, M. & Christodoulides, D. N. Non-Hermitian and topological photonics: optics at an exceptional point. Nanophotonics 10, 403–423 (2021).

7. Li, A. et al. Exceptional points and non-Hermitian photonics at the nanoscale. Nat. Nanotechnol. 18, 706–720 (2023).

8. Feng, L., Wong, Z. J., Ma, R.-M., Wang, Y. & Zhang, X. Single-mode laser by parity-time symmetry breaking. Science 346, 972–975 (2014).

9. Hodaei, H., Miri, M.-A., Heinrich, M., Christodoulides, D. N. & Khajavikhan, M. Parity-time-symmetric microring lasers. Science 346, 975–978, (2014).

10. Assawaworrarit, S., Yu, X. & Fan, S. Robust wireless power transfer using a nonlinear parity–time-symmetric circuit. Nature 546, 387–390 (2017).

11. Weimann, S. et al. Topologically protected bound states in photonic parity–time-symmetric crystals. Nat. Mater. 16, 433–438 (2017).

12. Guo, A. et al. Observation of PT-symmetry breaking in complex optical potentials. Phys. Rev. Lett. 103, 093902 (2009).

13. Ornigotti, M. & Szameit, A. Quasi PT-symmetry in passive photonic lattices. J. Opt. 16, 065501 (2014).

14. Joglekar, Y. N. & Harter, A. K. Passive parity-time-symmetry-breaking transitions without exceptional points in dissipative photonic systems [Invited]. Photon. Res. 6, A51–A57 (2018).

15. Ge, L. & Türeci, H. E. Antisymmetric PT-photonic structures with balanced positive- and negative-index materials. Phys. Rev. A 88, 053810 (2013).

16. Peng, P. et al. Anti-parity–time symmetry with flying atoms. Nat. Phys. 12, 1139–1145 (2016).

17. Yang, F., Liu, Y.-C. & You, L. Anti-PT symmetry in dissipatively coupled optical systems. Phys. Rev. A 96, 053845 (2017).

18. Li, Y. et al. Anti-parity-time symmetry in diffusive systems. Science 364, 170–173 (2019).

19. Zhang, F., Feng, Y., Chen, X., Ge, L. & Wan, W. Synthetic anti-PT symmetry in a single microcavity. Phys. Rev. Lett. 124, 053901 (2020)

20. Bergman, A. et al. Observation of anti-parity-time-symmetry, phase transitions, and exceptional points in an optical fibre. Nat. Commun. 12, 486 (2021).

21. Chong, Y. D., Ge, L. & Stone, A. D. PT-symmetry breaking and laser-absorber modes in optical scattering systems. Phys. Rev. Lett. 106, 093902 (2011).

22. Ambichl, P. et al. Breaking of PT symmetry in bounded and unbounded scattering systems. Phys. Rev. X 3, 041030 (2013).

23. Krasnok, A. et al. Anomalies in light scattering. Adv. Opt. Photon. 11, 892–951, (2019).

24. Ge, L., Chong, Y. D. & Stone, A. D. Conservation relations and anisotropic transmission resonances in one-dimensional PT-symmetric photonic heterostructures. Phys. Rev. A 85, 023802 (2012).

25. Longhi, S. PT-symmetric laser absorber. Phys. Rev. A 82, 031801 (2010).

26. Lin, Z. et al. Unidirectional invisibility induced by PT-symmetric periodic structures. Phys. Rev. Lett. 106, 213901 (2011).

27. Wong, Z. J. et al. Lasing and anti-lasing in a single cavity. Nat. Photonics 10, 796–801 (2016).

28. Gu, Z. et al. Experimental demonstration of PT-symmetric stripe lasers. Laser Photon. Rev. 10, 588–594 (2016).

29. Yang, M., Ye, Z., Farhat, M. & Chen, P.-Y. Enhanced radio-frequency sensors based on a self-dual emitter-absorber. Phys. Rev. Appl. 15, 014026 (2021).

30. Feng, L. et al. Experimental demonstration of a unidirectional reflectionless parity-time metamaterial at optical frequencies. Nat. Mater. 12, 108–113 (2013).

31. Li, H., Mekawy, A., Krasnok, A. & Alù, A. Virtual parity-time symmetry. Phys. Rev. Lett. 124, 193901 (2020).

32. Li, H., Mekawy, A. & Alù, A. Gain-free parity-time symmetry for evanescent fields. Phys. Rev. Lett. 127, 014301 (2021).

33. Liu, J.-J. et al. Experimental realization of Weyl exceptional rings in a synthetic three-dimensional non-hermitian phononic crystal. Phys. Rev. Lett. 129, 084301 (2022).

34. Zhang, L. et al. Acoustic non-Hermitian skin effect from twisted winding topology. Nat. Commun. 12, 6297 (2021).

35. Chen, Z. et al. Sound non-reciprocity based on synthetic magnetism. Sci. Bull. 68, 2164–2169 (2023).

36. Zhang, Q. et al. Observation of acoustic non-Hermitian Bloch braids and associated topological phase transitions. Phys. Rev. Lett. 130, 017201 (2023).

37. Chen, Z.-X. et al. Transient logic operations in acoustics through dynamic modulation. Phys. Rev. Appl. 21, L011001 (2024).

38. Yariv, A., Xu, Y., Lee, R. K. & Scherer, A. Coupled-resonator optical waveguide: a proposal and analysis. Opt. Lett. 24, 711–713, (1999).

39. Carminati, R., Sáenz, J. J., Greffet, J. J. & Nieto-Vesperinas, M. Reciprocity, unitarity, and time-reversal symmetry of the S matrix of fields containing evanescent components. Phys. Rev. A 62, 012712 (2000).

40. Zyablovsky, A. A., Vinogradov, A. P., Pukhov, A. A., Dorofeenko, A. V. & Lisyansky, A. A. PT-symmetry in optics. Phys. Usp. 57, 1063 (2014).

41. He, H. et al. Evanescent wave spectral singularities in non-Hermitian photonics. Phys. Rev. B 109, L041405 (2024).

42. Baranov, D. G., Krasnok, A. & Alù, A. Coherent virtual absorption based on complex zero excitation for ideal light capturing. Optica 4, 1457–1461 (2017).

43. Kim, S., Lepeshov, S., Krasnok, A. & Alù, A. Beyond bounds on light scattering with complex frequency excitations. Phys. Rev. Lett. 129, 203601 (2022).

44. Kim, S., Peng, Y.-G., Yves, S. & Alù, A. Loss compensation and superresolution in metamaterials with excitations at complex frequencies. Phys. Rev. X 13, 041024 (2023).

45. Guan, F. et al. Overcoming losses in superlenses with synthetic waves of complex frequency. Science 381, 766–771 (2023).

46. Gu, Z. et al. Transient non-Hermitian skin effect. Nat. Commun. 13, 7668 (2022).

## Acknowledgements

Y.L. was supported by the National Key Research and Development Program of China (No. 2022YFA1405000) and the Natural Science Foundation of Jiangsu Province (No. BK20212004). H.H., H.L., M.L., and J.X. were supported by the National Natural Science Foundation of China (Nos. 12274240, 92250302) and the Fundamental Research Funds for the Central Universities (No. 010-63243083). Z.C. was supported by the China Postdoctoral Science Foundation (Nos. 2023M731609 and 2024T170392), the National Natural Science Foundation of China (No. 12404506), and Jiangsu Funding Program for Excellent Postdoctoral Talent (No. 2023ZB473). A.A. was supported by the National Science Foundation STC NewFoS program, the Air Force Office of Scientific Research MURI program, and the Simons Foundation.

## Author contributions

H.L. and A.A. conceived the idea; Z.C. and J.K. performed the experiments; H.H., H.L., and M.L. conducted theoretical research; All authors discussed the results; Z.C., H.H., H.L., and A.A. contributed to the writing of the manuscript; H.L., Y.L., J.X., and A.A. supervised the project.

## Competing interests

The authors declare no competing interests.

## Additional information

Supplementary information The online version contains supplementary material available at https://doi.org/10.1038/s42005-024-01816-1.

Correspondence and requests for materials should be addressed to Huanan Li, Yan-qing Lu, Jingjun Xu or Andrea Alù.

Peer review information Communications Physics thanks Wenbo Mao and the other, anonymous, reviewer(s) for their contribution to the peer review of this work.

Reprints and permissions information is available at http://www.nature.com/reprints

Publisher's note Springer Nature remains neutral with regard to jurisdictional claims in published maps and institutional affiliations.

Open Access This article is licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License, which permits any non-commercial use, sharing, distribution and reproduction in any medium or format, as long as you give appropriate credit to the original author(s) and the source, provide a link to the Creative Commons licence, and indicate if you modified the licensed material. You do not have permission under this licence to share adapted material derived from this article or parts of it. The images or other third party material in this article are included in the article's Creative Commons licence, unless indicated otherwise in a credit line to the material. If material is not included in the article's Creative Commons licence and your intended use is not permitted by statutory regulation or exceeds the permitted use, you will need to obtain permission directly from the copyright holder. To view a copy of this licence, visit http://creativecommons.org/licenses/by-nc-nd/4.0/.

© The Author(s) 2024

## Terms and Conditions

Springer Nature journal content, brought to you courtesy of Springer Nature Customer Service Center GmbH (“Springer Nature”).

Springer Nature supports a reasonable amount of sharing of research papers by authors, subscribers and authorised users (“Users”), for small-scale personal, non-commercial use provided that all copyright, trade and service marks and other proprietary notices are maintained. By accessing, sharing, receiving or otherwise using the Springer Nature journal content you agree to these terms of use (“Terms”). For these purposes, Springer Nature considers academic use (by researchers and students) to be non-commercial.

These Terms are supplementary and will apply in addition to any applicable website terms and conditions, a relevant site licence or a personal subscription. These Terms will prevail over any conflict or ambiguity with regards to the relevant terms, a site licence or a personal subscription (to the extent of the conflict or ambiguity only). For Creative Commons-licensed articles, the terms of the Creative Commons license used will apply.

We collect and use personal data to provide access to the Springer Nature journal content. We may also use these personal data internally within ResearchGate and Springer Nature and as agreed share it, in an anonymised way, for purposes of tracking, analysis and reporting. We will not otherwise disclose your personal data outside the ResearchGate or the Springer Nature group of companies unless we have your permission as detailed in the Privacy Policy.

While Users may use the Springer Nature journal content for small scale, personal non-commercial use, it is important to note that Users may not:

1. use such content for the purpose of providing other users with access on a regular or large scale basis or as a means to circumvent access control;

2. use such content where to do so would be considered a criminal or statutory offence in any jurisdiction, or gives rise to civil liability, or is otherwise unlawful;

3. falsely or misleadingly imply or suggest endorsement, approval, sponsorship, or association unless explicitly agreed to by Springer Nature in writing;

4. use bots or other automated methods to access the content or redirect messages

5. override any security feature or exclusionary protocol; or

6. share the content in order to create substitute for Springer Nature products or services or a systematic database of Springer Nature journal content.

In line with the restriction against commercial use, Springer Nature does not permit the creation of a product or service that creates revenue, royalties, rent or income from our content or its inclusion as part of a paid for service or for other commercial gain. Springer Nature journal content cannot be used for inter-library loans and librarians may not upload Springer Nature journal content on a large scale into their, or any other, institutional repository.

These terms of use are reviewed regularly and may be amended at any time. Springer Nature is not obligated to publish any information or content on this website and may remove it or features or functionality at our sole discretion, at any time with or without notice. Springer Nature may revoke this licence to you at any time and remove access to any copies of the Springer Nature journal content which have been saved.

To the fullest extent permitted by law, Springer Nature makes no warranties, representations or guarantees to Users, either express or implied with respect to the Springer nature journal content and all parties disclaim and waive any implied warranties or warranties imposed by law, including merchantability or fitness for any particular purpose.

Please note that these rights do not automatically extend to content, data or other material published by Springer Nature that may be licensed from third parties.

If you would like to use or distribute our Springer Nature journal content to a wider audience or on a regular basis or in any other manner not expressly permitted by these Terms, please contact Springer Nature at
