REVIEW SUMMARY

OPTICS

# Complex-frequency excitations in photonics and wave physics

Seunghwi Kim†, Alex Krasnok†, Andrea Alù\*

BACKGROUND: Customizing how a system responds to external stimuli is essential for a wide range of wave-based technologies, such as photonics, acoustics, elastodynamics, radiofrequency engineering, and quantum optics. This response involves the intricate dynamics of wave interactions with matter, leading to reflection, absorption, diffraction, and scattering of waves, altering their momentum and energy flow. In turn, these principles underpin a multitude of wave phenomena and technologies, from the reflection of light and the echo of sound to the intricate behavior observed in photonic systems. In most settings, these phenomena are limited by fundamental system properties such as passivity, linearity, and time-reversal symmetry, imposing constraints on bandwidth, efficiency, and other performance metrics. Recent efforts to overcome these limitations involve the use of material gain; tailored responses in systems characterized by nonconservative interactions with their environment; time modulation, amplification, and lasing; and nonreciprocal materials. Although these strategies show promise, they often introduce unwanted challenges, such as increased complexity, reduced ease of integration, high costs, and footprint requirements.

ADVANCES: In recent years, excitations that oscillate at complex-valued frequency have transitioned from being merely analytical and numerical tools to model photonic systems to becoming a viable route to experimentally access exotic wave responses. By utilizing excitation signals with tailored waveforms whose amplitudes grow or decay exponentially in time, researchers have been able to effectively mimic the effect of gain and loss in passive systems without modifying their material properties. This advancement has led to experimental demonstrations of phenomena previously thought unattainable in passive systems. Notable examples include compensating losses in metamaterials, enhanced sensing, parity-time symmetry transitions without the need for active elements, and generation of optical pulling forces without specially designed spatial field gradients. These breakthroughs have also unlocked new capabilities, such as the manipulation of light for super-resolution imaging and real-time control over light-matter interactions and critical coupling of optical cavities, as well as phenomena that mimic the presence of material gain in passive systems. By bridging theoretical concepts with experimental implementations, these

![](images/24d8ca3b91508748a9f219bb8ced80d405304a9d410014ac843c3ea253f41df4.jpg)

Complex-frequency excitations in various wave physics settings. Exotic wave phenomena available across optics, radiofrequency (RF), elastodynamics, mid-infrared (mid-IR), acoustics, and quantum wave systems, leveraging complex-frequency excitations to enhance modern technologies.

advances demonstrate the feasibility of accessing non-Hermitian responses in passive linear systems. This enables new possibilities for wave-based technologies without the need for complex materials and the downsides of active elements.

OUTLOOK: The recent progress in the use of complex frequencies and their associated wave-matter interactions offers metamaterials and wave control new opportunities, particularly in the context of non-Hermitian wave phenomena. In optics and photonics, these tools offer opportunities to dramatically alter how light interacts with matter in a highly dynamic and tunable fashion, enabling enhanced control over light emission and transport. This paves the way for observing non-Hermitian and topological wave phenomena without relying on complex non-Hermitian materials, which are challenging to realize. By applying this excitation approach to well-established platforms, it becomes possible to exploit the interplay of effective gain and loss encoded in the temporal waveform of the excitation to create new functionalities and enhance the performance of modern technologies. For instance, in optical communications, sensing, and computing, the ability to manipulate waves by using complex-frequency excitations may lead to more efficient and adaptable systems.

Future research directions include developing more efficient methods for generating complex-frequency excitations, integrating these techniques into existing technologies, and exploring their applications across various fields. Emerging platforms such as metasurfaces, polaritonic materials, optomechanical systems, and topological insulators provide unexplored opportunities to investigate the effects of complex-frequency excitations in systems with inherently large nonlinearities, naturally strong light-matter interactions, and intrinsic robustness. Tailored effective gain and loss driven by the excitation waveform in these systems opens the potential for a substantial shift in the study, application, and control of wave-matter interactions across many physical domains. By bridging theoretical advancements with practical implementations, we anticipate that complex-frequency excitations may become crucial for future technological innovations, impacting fields beyond photonics and wave physics, such as quantum computing, biomedical engineering and sensing, imaging, and energy harvesting.

![](images/0612225ae6a91ed70011760889de61b925e5fb7d6c2b1d144cd748a6a81a74ba.jpg)

REVIEW

OPTICS

# Complex-frequency excitations in photonics and wave physics

Seunghwi Kim $^{1+}$ , Alex Krasnok $^{2+}$ , Andrea Alù $^{1,3*}$

Closed, lossless optical cavities are characterized by a Hamiltonian that obeys Hermiticity, resulting in strictly real-valued resonance frequencies. By contrast, non-Hermitian wave systems are characterized by Hamiltonians with poles and zeros at complex frequencies, whose control through precise engineering of material loss and gain can lead to exotic scattering phenomena. Notably, excitation signals that oscillate at complex-valued frequencies can mimic the emergence of gain and loss, facilitating access to these non-Hermitian responses without material modifications. These findings have been advancing the fundamental understanding of wave-matter interactions and are enabling breakthroughs in metamaterials, imaging, sensing, and computing. This Review examines theoretical advances and experimental discoveries in this emerging field, demonstrating how tailored time-domain excitations offer new opportunities for wave manipulation and control.

ave-matter interactions form the cornerstone of wave control across various domains, including photonics, acoustics, elasticity, radiofrequency (RF) engineering, and quantum optics.

They enable control over wave properties such as direction, energy, and momentum, and encompass a broad spectrum of natural phenomena and technological applications, from light reflection and sound echoes to the dynamics of complex photonic and quantum systems. Because of this broad relevance, wave scattering has been a deeply studied topic throughout all aspects of wave physics, including optics, acoustics, and elastic waves (1–4). In photonics, the study of light-matter interactions has evolved from canonical processes, such as reflection, refraction, and absorption, to sophisticated responses facilitated by the advent of metamaterials and nanophotonics (5, 6), uncovering remarkable phenomena that include super-resolution (7–10), nonreciprocal responses (11–15), and cloaking (16–20). Acoustic and elastic metamaterials have enabled unprecedented control over sound waves, offering capabilities such as negative effective mass and modulus (21, 22), acoustic and elastic cloaking (23–25), and subwavelength focusing (26, 27), with a potential impact on ultrasound imaging, architectural acoustics, and sonar systems (28). In RF engineering, reconfigurable metamaterials have been enhancing communication and radar systems by enabling exotic wave-matter interactions (29–31). Quantum optics also benefits from stronger wave-particle interactions, impacting crucial quantum technologies such as quantum communication and computing (32, 33).

Recent theoretical advances in these areas of wave physics have revealed exotic wave phenomena emerging from carefully designed structures. Examples include bound states in the continuum (BICs) (34–36)—where localized states persist despite lying within a continuum of radiation modes—and parity-time (PT) symmetric phases in non-Hermitian systems (37–39). Non-Hermiticity requires non-conservative exchanges of energy with the environment, and when gain and loss are balanced, such systems can support PT-symmetric responses, leading to real-valued eigenvalues and phase transitions (37, 40). These fascinating properties can be found across many classical wave platforms, including optics (41), RF circuits (42), mechanics (43), and acoustics (44).

In linear systems, the wave interactions can be efficiently analyzed through the poles and zeros of the scattering matrix at complex frequencies. The scattering matrix captures how incident waves are transformed into outgoing waves, with poles and zeros representing resonances and antiresonances, respectively (39). The scattering matrix approach can be rigorously derived from the corresponding field equations, e.g., Maxwell's equations in electrodynamics (39). One can realize extreme scattering responses by tailoring the position of these singularities in the complex plane. However, intrinsic limitations such as passivity (inability to generate energy), causality (cause precedes effect), and energy conservation constrain the degree to which these poles and zeros can be manipulated, ultimately restricting device performance. Introducing tailored non-Hermitian elements, such as adding gain or loss, allows for greater manipulation of poles and zeros, potentially aligning them with real frequencies and overcoming these intrinsic barriers.

Remarkably, recent research has shown that one can access non-Hermitian wave phenomena not just through structural or material modifications but through carefully designed temporal excitations. By using complex-frequency excitations—whose real-frequency oscillation is accompanied by exponential growth or decay—one can emulate gain or loss without altering the material properties. This strategy unlocks new opportunities to control wave-matter interactions, as evidenced by recent discoveries of various wave phenomena.

As foundational examples, we start by discussing how virtual loss and virtual gain can be accessed in a linear system by exciting it around a zero or a pole of its scattering matrix in the complex-frequency plane. We assume an $e^{-i\omega t}$ time convention throughout the text. Virtual loss is manifested when the signal waveform matches the complex frequency of a scattering zero (Fig. 1A, upper half-plane). Such a signal, with an exponentially growing amplitude, mimics absorption even though no actual loss mechanism exists. In a lossless open system, because zeros and poles appear as complex conjugates, the required signal is the time-reversed version of one of its eigenmodes (45, 46), and the energy is stored in the cavity as long as the excitation continues to grow. Conversely, virtual gain arises when an exponentially decaying signal (Fig. 1A, lower half-plane) matches a complex pole, replicating the self-oscillatory behavior of a resonant mode that would otherwise require actual material gain. This pole-based effect is linked to quasi-normal modes (QNMs) (47, 48), which are the natural resonances of open systems.

By synthesizing gain and loss in lossless systems through properly crafted excitations, we can broaden the response of linear systems across various scattering platforms and access complex non-Hermitian wave phenomena. Because the scattering parameters—reflection, absorption, transmission, and extinction—are analytically extendable in the complex-frequency plane, we can directly engage their poles and zeros at complex frequencies (49–51). In this Review, we explore the potential of such time-domain complex-frequency excitations in wave physics, examining how these methods can enable fundamental science and application breakthroughs. We begin by examining phenomena associated with scattering systems, such as virtual critical coupling and virtual perfect absorption (52, 53) (Fig. 1, B and C). We then discuss how virtual gain enables the manipulation of scattering and extinction cross sections beyond passive limits $(54, 55)$ (Fig. 1D). Next we explore how virtual gain can compensate loss in resonant systems, enabling superresolution imaging $(56, 57)$ (Fig. 1E). We then investigate the excitation of complex zeros and poles to induce non-Hermitian wave phenomena, such as PT symmetry $(58)$ and the temporal non-Hermitian skin effect $(59)$ (Fig. 1F). By bridging theoretical concepts with practical implementations, we highlight the transformative potential of complex-frequency excitations in wave physics. Lastly, we highlight emerging directions and future advances in this rapidly expanding field of research.

![](images/79ce1148df6a36c40ea2c98a419f460fbd2306ea015f0e16ecbd282d714192be.jpg)
Fig. 1. Overview of complex-frequency excitations. (A) Virtual loss and virtual gain. By exciting a linear system with a complex frequency around a zero in the complex plane, the excitation energy can be temporarily stored in the system, mimicking absorption. In a similar way, virtual gain emerges when the amplitude of the scattered signal is larger than the incident one because the excitation maps the temporal waveform of a natural resonant mode. (B) Singularities in the reflection coefficient under complex-frequency excitations, including virtual critical coupling and virtual perfect absorption. (C) Singularities of the scattering matrix, such as coherent virtual absorption. (D) Extreme scattering responses associated with complex-frequency excitations. (E) Super-resolution with complex frequencies. (F) Engaging zeros and poles, as in virtual PT symmetry and temporal non-Hermitian skin effect.

## Theoretical background Scattering phenomena in linear systems

Wave phenomena across different physical domains share a unifying mathematical foundation, captured by the wave equation. In closed, lossless (Hermitian) systems—such as a one-dimensional (1D) cavity with perfectly reflecting boundaries—waves remain confined, resulting in discrete, real-valued resonance frequencies that correspond to standing wave patterns with no energy loss. By contrast, an open system, where at least one boundary allows outward radiation, creates a pathway for excitation and allows measuring the response to such excitation. In such open configurations, reflection poles typically lie below the real frequency axis, while reflection zeros reside above it. For lossless media, these poles and zeros appear in complex-conjugate pairs, shifting closer to the real axis as the open boundary gradually becomes a perfect reflector (Fig. 2A).

By introducing loss or gain into this picture, the complex-conjugate symmetry between reflection zeros and poles is broken, allowing them to move in the complex frequency plane (Fig. 2, B and C). A zero moves to the real axis when the radiation rate equals the absorption rate in the cavity, leading to perfect absorption. In single-port systems, where the scattering matrix consists of a scalar complex reflection coefficient, $r(\omega)$ , the condition of zero reflection indicates critical coupling, achievable with real-frequency excitations (60, 61). Similarly, a pole moves to the real axis in the presence of gain, whose rate perfectly compensates radiation, corresponding to the lasing threshold (39).

The following discussion extends beyond this single-channel scenario to systems with multiple resonances and multiple ports (Fig. 2D). In multiport settings, perfect absorption can be accompanied by coherent control, leading to coherent perfect absorption (CPA) (49, 62–64) and reflectionless scattering modes (RSMs) (65). Coherent perfect absorption exploits interference within the resonator among multiple excitation channels to fully absorb incident waves, providing dynamic control over absorption and scattering. This platform can also lead to intriguing wave phenomena, such as BICs and exceptional points (EPs). BICs occur when a pole and its corresponding zero converge onto the real-frequency axis (66), resulting in states that remain perfectly localized despite existing within the continuum of radiation modes, with applications in sensing, lasing, and energy harvesting (34, 67–69). By contrast, EPs are singularities where eigenvalues and eigenvectors coalesce, resulting in drastic phase transitions. These features promise opportunities for enhanced sensing and robust mode switching (38, 70).

Although finding complex zeros and poles in realistic systems can be difficult because of the challenges in accurate modeling and measurements, recent advances in numerical and experimental techniques have been enabling the efficient computation of these singularities (71–73). These advances highlight the practical application of complex-frequency inputs, expanding the toolkit for wave manipulation and control. For example, methods such as time-domain spectroscopy and complex-frequency modal analysis have been developed to identify and engage these singularities.

## Complex-frequency excitations

Complex-frequency analysis has long served as a powerful tool for modeling wave-based systems (39, 73–75). More recently, it has been discovered that complex frequencies can be incorporated into time-domain excitation signals themselves, allowing direct interaction with the poles and zeros of a system to realize virtual gain or loss. The exotic wave phenomena discussed in the previous section are associated with zeros and poles of the scattering operators. Traditionally, these singularities have been studied in non-Hermitian systems at real frequencies. Crafting excitation signals at complex frequencies makes these singularities accessible in the complex-frequency plane, considerably broadening the landscape of achievable scattering effects, even in Hermitian or passive systems.

![](images/c9c3b54a54f25d4d207d7b8eeed50e411e024d80b185637e6f5f554b95070a55.jpg)
Fig. 2. Scattering phenomena in complex-frequency plane. (A) In a closed Hermitian system, poles and zeros on the real frequency axis form bound states. In the presence of radiation, they can move to the complex plane as complex conjugates of each other. (B and C) Introducing loss (B) or gain (C) moves a zero (or pole) to the real axis, enabling phenomena such as perfect absorption, CPA, or lasing. (D) Multiple resonances coupling to multiple channels give rise to rich phenomena such as CPA, BICs, or EPs.

Complex-frequency excitations enable experimental access to complex poles and zeros of the scattering matrix, unattainable with monochromatic real-frequency excitations. The imaginary frequencies of poles are related to the resonance quality factor, $Q = \mathrm{Re}(\omega) / \mathrm{Im}(\omega)$ , whereas the imaginary parts of zeros vary on the basis of system losses (Fig. 2B). Here, $\mathrm{Re}(\omega)$ and $\mathrm{Im}(\omega)$ represent the real and imaginary frequencies, respectively. Generally, resonant scattering phenomena under complex-frequency excitations can be efficiently modeled using coupled-mode theory (CMT), a versatile framework that describes multiple resonances and channels in the weak-coupling regime (76, 77). For a single-port system (Fig. 2A), CMT predicts that for a monochromatic excitation at frequency $\omega_{r}$ , the reflection coefficient measured at the port is given by $r = \frac{(\gamma_{\mathrm{ex}} - \gamma_{\mathrm{in}}) - i(\omega_0 - \omega_r)}{(\gamma_{\mathrm{ex}} + \gamma_{\mathrm{in}}) + i(\omega_0 - \omega_r)}$ , where $\gamma_{ex}$ is the external coupling rate, $\gamma_{in}$ is the internal loss rate, and $\omega_0$ is the resonant frequency of the system (76, 78, 79). By analytically extending this expression into the complex-frequency domain, we can investigate the system response at complex frequencies. We replace $\omega_{r}$ with the complex quantity $\omega_{r} + i\omega_{i}$ , leading to Eq. 1 described in (52):

$$
r _ {\mathrm{qs}} = \frac {(\gamma_ {\mathrm{ex}} - \gamma_ {\mathrm{in}}) - \omega_ {i} - \mathbf {i} (\omega_ {0} - \omega_ {r})}{(\gamma_ {\mathrm{ex}} + \gamma_ {\mathrm{in}}) + \omega_ {i} + \mathbf {i} (\omega_ {0} - \omega_ {r})}\tag{1}
$$

A positive imaginary part of the complex frequency ( $\omega_{i} > 0$ ) corresponds to a reflection coefficient that experiences an effective additional loss. Conversely, a negative imaginary part ( $\omega_{i} < 0$ ) implies that the system reflects as if gain were introduced (Fig. 3A). In lossless systems ( $\gamma_{in} = 0$ ), for instance, input waves can experience zero reflection when $\omega_{i} = \gamma_{ex}$ because of virtual loss, enabling critical coupling at the input port. Here, $\gamma_{in}, \gamma_{ex}, \omega_{0}, \omega_{r}$ , and $\omega_{i}$ are real-valued.

Although extending the system response to the complex-frequency domain offers insights into the wave behavior (76, 78), under certain conditions these responses can be accessed by using time-domain waveforms that oscillate at a complex frequency. However, generating such signals poses practical challenges because monochromatic signals with a complex frequency are unrealizable; this is because they are unbounded. Specifically, a temporal signal of the form $e^{-\mathrm{i}(\omega_r + \mathrm{i}\omega_i)t} = e^{\omega_i t}e^{-\mathrm{i}\omega_r t}$ grows or decays exponentially, making it inherently transient. For $\omega_{i} > 0$ , the signal grows exponentially with time $(t\rightarrow +\infty)$ and decays exponentially as $t\rightarrow -\infty$ ; for $\omega_{i} < 0$ , the signal decays as $t\rightarrow +\infty$ and grows as $t \rightarrow -\infty$ . Thus, these excitations must be truncated in time to remain physically realizable. Moreover, because complex exponentials do not constitute an orthogonal basis, a system driven by a complex frequency may not necessarily respond solely at that frequency.

We can define a temporal reflection coefficient for a complex-frequency signal as the ratio of the output field to the input field at any instant in time (80):

$$
r (t) = r _ {\mathrm{qs}} + r _ {\mathrm{ts}} (t)\tag{2}
$$

Here, $r(t)$ consists of two parts: the quasi-steady-state term $r_{qs}$ , given by Eq. 1, which oscillates at the same complex frequency as the input, and the transient response $r_{\mathrm{ts}}(t)$ , which varies over time. The transient term can be expressed as $r_{\mathrm{ts}}(t) = -e^{-Z(\omega)t}2\gamma_{\mathrm{ex}}/Z(\omega)$ , where $Z(\omega) = (\gamma_{\mathrm{ex}} + \gamma_{\mathrm{in}}) + \omega_i + \mathrm{i}(\omega_0 - \omega_r)$ . In contrast to real-frequency excitations, in which the transient necessarily vanishes over time for passive systems and the response settles into its steady state, complex-frequency excitations may produce a dominant or even diverging transient that may mask the quasi-steady-state response oscillating at the same complex frequency as the excitation. However, by properly tuning the system parameters and the excitation waveform, one can ensure that the quasi-steady-state response dominates and thus experimentally access the non-Hermitian features associated with complex-frequency phenomena.

As illustrated in Fig. 3B, the temporal response of the reflection $R(t) = |r(t)|^{2}$ under complex-frequency excitation reveals perfect absorption or critical coupling in an over-coupled resonator (which ordinarily reflects some power at real frequencies). By driving the resonator at the complex zero of its scattering response, the reflection approaches zero after a transient period on the order of the inverse total loss rate, $\tau > 1$ , where $\tau \equiv t(\gamma/2\pi)$ . This demonstrates how virtual loss introduced by shaping the excitation in time can enable impedance matching and efficient excitation of the resonance. Furthermore, Fig. 3C shows that the corresponding spectrum converges to the spectrum of a critically coupled resonator, even if the port is not matched. In the same vein, virtual gain can offset intrinsic resonator losses ( $\gamma_{in}$ ), assisting in achieving critical coupling for overdamped systems (56, 57, 81).

Equation 2 underscores the potential of complex-frequency excitations for a single-port resonator, serving as a simple yet illustrative example. These features can be extended to systems with multiple modes and ports by using the language of quasi-normal mode expansion (47, 48). Poles and zeros featuring large imaginary parts (high $|\omega_i|$ ) are difficult to engage in practice because the required waveforms grow or decay too rapidly. Consequently, complex-frequency excitations are most practical when $|\omega_{i}| \ll \omega_{r}$ , as indicated outside the gray region in Fig. 3A.

A
![](images/e84d7a9ad2eddb6087c7a57a3f380a884ee29f7d1dfb58b9f653ffebce649b13.jpg)

B
![](images/bac1891a47caa07558a16ae57c71f0104424288ca84667cb048f709ab51f5fa8.jpg)

![](images/848262b8a2245fc7a8777385fdbb0d50a227828cdb765dbcb52c11c8d3a75c7b.jpg)
E

C
![](images/277ea54ac4e72c86342482bc142b676d5652c8c534615bc946ca04b531282488.jpg)

![](images/93ba8fdca2120e52bf09b01dad54fa1c7907f4432edd31217416b6cde0fed19f.jpg)

F
![](images/1693573d952115dcd23b7bb964d4ab0e6f54a66056a5928453747cfc056fd75a.jpg)

G
H
![](images/d28a3ecdf9571ba7ace0fc7b7143bd8ef532880de8c1289187b3a7f05fc7c606.jpg)

![](images/6e2f7676b12e1b6e39618c7d521549958de8acec5242fe1dbb30160ae881ae12.jpg)
Fig. 3. Understanding complex-frequency excitations. (A) States in the complex-frequency plane can be accessed within the green area (as shown here) under complex-frequency excitations, which enable virtual loss (or gain) for $\omega_{i} > 0$ (or $\omega_{i} < 0$ ). (B) Temporal evolution of the reflection $R$ under virtual critical coupling. $R$ converges to zero after a transient, indicating the quasi-steady state regime. (C) Frequency spectrum of the measured states in (B) represented at various measured times, converging toward the critical coupling spectrum (dashed line). (D and E) Finite pulses emulating complex
frequencies associated with virtual loss (D) and gain (E). (F and G) Fourier spectrum of the pulse shown in (E). Each harmonic wave can be sampled [red dashed lines in (F)] to reconstruct the original pulse, and the synthesized pulse (black dashed line) is shown in (G). (H) Schematic of complex-frequency excitations through synthesized pulses derived from multiple harmonic waves, characterized by amplitudes $a_{n}^{i}$ and phases $\phi_{n}^{i}$ . Each output with amplitude and phase $(a_{n}^{o}, \phi_{n}^{o})$ is collected and postprocessed to obtain the desired response.

Figure 3D illustrates a finite pulse exponentially growing at rate $\omega_{i}$ , starting from negative time to $\tau = 0$ and then ramped down—a typical waveform to access virtual loss. Similarly, Fig. 3E shows a pulse oscillating at a complex frequency with $\omega_{i} < 0$ for $\tau > 0$ , enabling virtual gain. These waveforms can be readily produced and applied to resonant systems in acoustics, mechanics, and radio frequencies. In these settings, arbitrary waveform generators (AWGs) can generate pulses with an imaginary frequency (growth or decaying rates) up to a few hundred megahertz. At higher frequencies, as in the mid-infrared or visible ranges, temporal pulse shaping becomes more challenging owing to physical limitations, including the time-bandwidth product (TBP), and to practical limitations in laser sources.

Even so, integrated photonic platforms can achieve fine pulse shaping by using electro-optic modulators with bandwidths reaching $\sim 100\mathrm{GHz}$ at near-infrared and visible wavelengths (80). In free space, mode-locked lasers can generate pulses with spectral widths up to a few terahertz, albeit fundamentally limited by TBP and gain bandwidth (82). As the frequency increases and the poles and zeros of resonant systems move further from the real axis, the rapid amplitude modulation required for generating these signals becomes increasingly impractical. This limitation is particularly pronounced for low $Q$ -factor resonators in nanophotonics and other high-frequency systems. For resonators with multiple, closely spaced singularities in the complex-frequency plane, the required excitation waveforms can be more intricate. Perfect absorption in such scenarios demands carefully devised analytical (83) or experimental techniques (53, 84).

An alternative strategy to access complex-frequency excitations is to synthesize them by superimposing monochromatic components in the Fourier domain (57, 81). For instance, the temporal waveform in Fig. 3E has a Fourier spectrum (Fig. 3F). By sampling this spectrum at discrete real frequencies (Fig. 3F, red dashed lines) and combining them with appropriate complex weights, one can reconstruct the desired complex-frequency excitation. Figure 3G shows the reconstructed pulse generated by summing 101 harmonic waves sampled over the frequency range shown in Fig. 3F. The response to each frequency is measured at the output port, and the resulting outputs are combined with proper complex weights to reconstruct the desired response (Fig. 3H). This method leverages the linearity of the system but requires precise control and extensive postprocessing. This approach becomes particularly cumbersome in high-frequency settings, where measuring relative phases over a broad frequency range is challenging. Additionally, this technique is not applicable to nonlinear systems, in which complex-frequency excitations may offer other interesting opportunities.

## Exciting complex-frequency zeros

Early studies on harnessing complex-frequency excitations for boosting coupling to optical and quantum resonators date back to the early 2010s. Researchers showed that using the time-reversed signal of the spontaneous decay of a quantum system enables strong photon-matter interactions, achieving high coupling efficiencies and coherent control over absorption and emission in the quantum regime. By tailoring temporal waveforms, one can effectively direct energy into optical and quantum resonators, paving the way for advanced control and manipulation of both classical and quantum wave transport (85). Other related works explored wave-chaotic sensing techniques in which time-reversed signals were used to detect small perturbations. In lossy environments, these methods use exponential amplification to compensate for dissipation, effectively leveraging complex-frequency excitations (86, 87).

Recent work has underscored that these insights can be applied across multiple photonic and wave platforms to enhance mode coupling (39, 88). This understanding has led to enhanced control over wave interactions with resonators. Perfect absorbers are structures engineered to fully capture incident waves, such as electromagnetic or acoustic waves, by preventing reflections (89, 90). Such devices enable diverse applications in energy harvesting, stealth, enhanced wave-matter interactions, sensing, and noise control. Advances in this area have emphasized the pivotal role of wave interference in creating perfect absorption (63). For example, the Salisbury screen achieves zero reflectance with a resistive sheet placed a quarter wavelength above a reflector, by using constructive interference at the absorber layer. In a similar vein, lossy resonant cavities and impedance matching strategies, often based on metamaterials, achieve perfect absorption by carefully engineering wave interference (91). These concepts can be further extended by considering multiple input waves. The introduction of coherent perfect absorbers marked a turning point in this research field, facilitating perfect absorption of waveforms from different angles or phases (49, 62, 92). A generalization to multi-port networks leads to RSMs (65, 92–95), offering powerful control over directionality and wave manipulation. Because of this flexibility in shaping waveforms, multiport photonic systems benefit from precise scattering and absorption control (63).

All these schemes rely on material losses to convert incoming electromagnetic or acoustic energy into alternate forms (e.g., heat and electricity). Complex-frequency excitations transcend this paradigm by enabling perfect absorption in systems with any level of material loss, even zero. In a single-port resonator (Fig. 4A), an exponentially growing signal that matches a complex zero's temporal profile ensures that, in a quasi-steady state, resonator decay is nullified by destructive interference at the input port (52, 80). Known as “virtual critical coupling,” this effect has been applied in microstrip line technology (96, 97) and microcavity-based high-power plasma generation (53, 84) at radio frequencies and was recently demonstrated at telecom wavelengths in silicon nitride (80). By tuning the imaginary part of the excitation frequency (Fig. 4D), one can seamlessly transition a resonator between overcoupled and under-coupled states without altering its physical structure—an especially promising feature for high-frequency high-Q-factor photonic systems.

Expanding virtual critical coupling to multi-port systems enables coherent virtual absorption, which provides enhanced control over energy storage and release in lossless structures (39, 88, 98). By reaching complex zeros through multiport excitations, one can manage electromagnetic energy storage and release in a low-loss or lossless environment. As illustrated in Fig. 4B, a planar slab with refractive index n = 3 is illuminated from both sides, and an exponentially growing field (dashed red line) matching the complex-frequency zero sustains energy storage in the cavity; once the exponential growth stops at $\omega't = 0$ , the accumulated energy is released. Here, each zero is equally spaced, with the spacing determined by the width of the cavity L (Fig. 4B, top left). The lower panel shows the electric field intensity at that instant, highlighting the efficient excitation of a resonant slab mode.

Coherent virtual absorption has also been demonstrated in elastodynamic systems $(99)$ . Tailoring the relative phases of the excitations on both sides of a lossless elastic cavity controls how stored energy is symmetrically or asymmetrically released after the pump terminates (Fig. 4C). This concept extends to 2D and 3D scattering scenarios: Recent findings $(100)$ show that complex-frequency waveforms can create a “wave sink” in a lossless medium (Fig. 4E). Unlike monochromatic signals, which need absorbing singularities $(101)$ , complex-frequency excitations form a subdiffraction sink without material loss, offering novel ways to direct and concentrate wave energy. This capability enhances imaging and sensing by surpassing the resolution limits of conventional techniques, opening new frontiers in wave focusing and control.

## Exciting complex-frequency poles

Targeting complex-frequency poles in a system's response can mimic material gain through virtual gain, much like targeting complex zeros can mimic loss through virtual absorption. By focusing on poles in the complex-frequency plane, these excitations enable responses that overcome the passivity constraints of linear systems (55). Passivity and causality principles dictate that these poles must lie in the lower half of the complex plane for the $e^{-i\omega t}$ convention (102–104), imposing stringent bounds on a system's response. Wave scattering by small particles provides a useful platform to analyze these limitations and explore the opportunities afforded by virtual gain (105). Scattering strength describes how effectively light is scattered relative to the incident irradiance and is constrained by the passivity (1, 106). These limitations can be overcome by advanced photonic designs that overlap multiple scattering channels at the same frequency (107, 108) or incorporate material gain, breaking the passivity constraint (109). Similarly, passivity forbids making the forward scattering too low because the total power intercepted by a passive scatterer must be proportional to its shadow, resulting in forward scattering (110, 111).

Complex-frequency excitations can induce exotic wave phenomena that push the scattering bounds set by passivity in small scatterers (55). Figure 5A illustrates the scattering pattern from a dielectric cylinder excited at the complex frequency of a scattering pole. Notably, the scattering efficiency in the quasi-steady state surpasses the passive limit for monochromatic excitations by two orders of magnitude. This lasing-like response mimics material gain, revealing clean multipolar scattering patterns (e.g., quadrupolar) typically masked under monochromatic excitation. Furthermore, virtual gain from complex-frequency excitations can suppress the shadow of a passive scatterer, reducing forward scattering to zero. Figure 5B (left) shows the scattering pattern of small dielectric spheres under plane wave excitation at the complex frequency corresponding to a zero of the forward scattering for both transverse-electric and transverse-magnetic polarizations. The forward scattering is fully suppressed in the quasi-steady state (55), as further visualized in Fig. 5B (right), which plots the normalized forward and total scattering cross sections.

These phenomena occur because a resonant scatterer temporarily stores energy during the transient and later releases it in the quasi-steady state, interfering with decaying excitation fields. With an appropriately tailored wave profile, this leakage can enhance a resonant mode (virtual lasing) or suppress forward scattering (virtual zero forward scattering). Notably, these effects emerge naturally without engineered structures because the required waveform is determined by analyzing the transfer function's singularities in the complex-frequency plane.

A
![](images/ff4dcb9a03ea1e827e605f1ab8cf022b5f22ac74ebbf7811d3f417a6e738fc4e.jpg)

B
![](images/cd81864d9a78fb496f3621763fb979ce19f51993df000a87b80a833d4e16df2a.jpg)

![](images/67fd49503250c58ca0c5087a80f8a1e992aac141d2ed361ccba3b94e9eb2a7a6.jpg)

![](images/cca0b1f135ce626690f0a1ff6d33010a3d742dba9a48e59acbcc00359a93eaf2.jpg)

![](images/8134e67600d62a2bf6731cd1c8a4b3a9440517d34f8430c7e78e769a2de9dee7.jpg)

![](images/b0b655c527da6d2cd0ea622452c2dbf06d4ea4a5d912c857f536819e2c085ad4.jpg)

![](images/67edfc62be54e897cebae85cb22718bb76b5c0358dee0f676c6e73d356b18acf.jpg)

![](images/8d32ca3f0a40365d310c3fe32f140679ddb337b96f87588bcc6cecdfa6ea641c.jpg)

C
![](images/f7ea0bc640861c36977426a938d1190ec58019266b24b5cbdf7cb0f008b29ae5.jpg)
Fig. 4. Virtual absorption and its applications. (A) Schematic of a single-port planar resonator excited by an exponentially growing signal, leading to virtual perfect absorption or virtual critical coupling [adapted with permission from (52). Copyright © 2025 American Chemical Society]. (B) Coherent virtual absorption in a planar slab under normal incidence from opposite sides, with the upper panel showing the geometry, the center panel illustrating the predicted scattering response to an exponentially growing field, and the lower panel displaying electric field intensity indicating reflection suppression [adapted with permission from (88). Copyright © 2017 Optica Publishing Group]. (C) Experimental setup for
elastodynamic wave control in a waveguide coupled to a resonator, showcasing energy capture and release through phase adjustment of the incoming signals, highlighting the impact of coherent virtual absorption principles for elastodynamic wave control [adapted with permission from (99)]. (D) Realization of CPA in an integrated photonic system [adapted with permission from (80)]. (E) Generalization of virtual perfect absorption in a 2D elastic wave system, using complex-frequency signals, enabling subwavelength focusing and surpassing traditional diffraction limits in imaging [adapted with permission from (100)].

Virtual gain also enhances the control over wave-induced forces. Light and sound carry momentum, exerting forces on objects $(112)$ , but passive bodies under plane waves cannot experience pulling forces $(113)$ . Conventional approaches involve structured beams $(114)$ , active objects $(113)$ , or engineered scatterers $(115)$ , adding complexity $(116)$ . Complex-frequency excitations, however, enable pulling forces on passive scatterers under plane wave excitation $(54)$ (Fig. 5C), substituting material gain with energy release from a resonant state excited during the transient phase.

Access to balanced gain and loss enables PT-symmetric phenomena, including lasing-absorbing modes and anisotropic transmission resonances (ATRs), which exhibit unity transmission with zero reflection on one side—impossible to achieve in passive systems (44, 117–119). Virtual gain allows these effects in passive setups, such as an RLC (resistor-inductor-capacitor) circuit excited by complex-frequency signals, supporting ATRs and lasing-absorbing modes (58, 120) (Fig. 5D). Here, transmission remains at unity, but reflection is zero from one port and large from the other, violating passivity.

Another landmark advance associated with non-Hermiticity in wave physics is found in the new perspectives that have been gained on topological phases of matter. The non-Hermitian skin effect, characterized by the anomalous localization of eigenmodes, occurs at spatial boundaries (121, 122) and has been demonstrated, typically relying on active platforms (123, 124). Material gain, however, introduces stability and complexity issues. Virtual gain, through complex-frequency excitations, recently enabled the non-Hermitian skin effect in a passive acoustic ring resonator lattice (59) (Fig. 5E). Tailored exponentially decaying pulses amplify waves propagating away from the source while suppressing those moving toward it, showcasing this effect's key features. These findings can extend to mechanics, RF circuits, and photonics.

Virtual gain also offers a pathway to overcoming material losses in metamaterials. Pendry's 2000 proposal of a negative-index superlens (7, 125) has demonstrated subdiffraction imaging (8, 9, 26, 27, 126) but suffers from evanescent wave attenuation because of material losses (127, 128). Although active materials can compensate for absorption (129–131), they introduce instability, noise, nonlinearities, and saturation issues (132–134). Recent theoretical work has shown that superlenses can support enhanced resolution when excited with pulsed or abruptly terminated harmonic signals (135, 136). This approach is analogous to the physics of virtual gain, which utilizes exponentially decaying excitations. As shown in Fig. 6A, the transfer function of a lossy superlens in the image plane can be considerably improved for larger wave numbers—associated with subwavelength resolution—when the imaginary part of the complex excitation frequency $\gamma$ matches the material loss rate $\alpha$ (56), thereby restoring the super-resolving capabilities of the slab.

A
![](images/8ee6746a1dcf7e3b3f0f492576160780cd7e6d8ffaa99885453637fd4f13bda3.jpg)
B

![](images/9628bd3929807bd6a7af5952986cf87a3f7597d9da994a805a95f904b5685b5c.jpg)

![](images/d16624971bd817a0aad67ef6018c17572ddb22f130c3e0918b87dfb9f13b143b.jpg)

C
![](images/68791f3fac2a15a6bcf2f7f330e7e5454e759afb0adde5b234b8673184db0616.jpg)

D
![](images/68449f1e6a961dbb3de37551e729b6f90314bc77243a14007f0920bf339caf0a.jpg)
E

![](images/a4c844256cb18bd02e00d6518f9d046737be7926ff6671b9bfb268bc36ec0eaa.jpg)

![](images/7118630dfc5913662c875b28908ea419942e393207264297b39c869919d16ae4.jpg)

![](images/c8a2ae5adf06d449d3bbd63d077938d62db7135bf1c7d669c3fcf66e3e7bac63.jpg)
Fig. 5. Virtual gain to overcome passivity limitations. (A) Temporal evolution of normalized scattering patterns under complex-frequency excitations from simulations with the normalized time $\tau$ , leading to large quadrupolar response beyond the passivity limit [adapted with permission from (55)]. (B) Scattering patterns computed at $\omega R / c = 0.824 - i0.063$ for both parallel and perpendicular polarization (pol.), illustrating (left) zero forward scattering for a dielectric nanoparticle and (right) normalized forward and total scattering cross-section bounded by passivity with a different refractive index, $m$ (blue area). FDTD, finite difference time domain. Complex-frequency excitations enable scattering features beyond the passivity bound (green line). Here, $R$ is the radius of particles [adapted with permission from (55)]. (C) (Top) Schematic of a
nanoparticle (NP) pulled by an exponentially decreasing signal (I, intensity; F, force) and (bottom) corresponding optical forces [adapted with permission from $(54)$ . Copyright © 2020 Optica Publishing Group]. (D) Virtual PT symmetry in a passive circuit, supporting anisotropic transmission resonances under complex-frequency excitations. Although both panels show transmitted signals (blue) equal to the inputs in quasi-steady states (red), the reflected signals (green) in the upper panel rapidly converge to zero, whereas strong reflection is observed in the lower panel, equal to the input [adapted with permission from $(58)$ ]. (E) Demonstration of transient non-Hermitian skin effect (NHSE) under complex-frequency excitations. Acoustic coupled resonators (top) support selective amplification of left-propagating waves (bottom), a feature of the skin effect [adapted with permission from $(59)$ ].

This effect was experimentally validated using a 3D-printed acoustic metamaterial superlens (26, 137) (Fig. 6B). Under monochromatic excitation, resolution is limited by thermoviscous dissipation and fabrication imperfections. However, excitation with a complex frequency matching the material loss rate ( $\alpha = 180$ Hz) considerably enhances resolution in the quasi-steady-state regime (56). A similar approach was demonstrated in the infrared regime (57) (Fig. 6C), where complex frequencies were synthesized by combining multiple monochromatic images, leveraging Fourier expansion (Fig. 3H). The reconstructed image exhibited significantly higher resolution than any individual monochromatic image.

Although retrieving and combining complex fields is challenging, requiring iterative methods and extensive postprocessing (138, 139), alternative techniques such as ptychography (140, 141) may offer comparable benefits. This proof of concept demonstrates the potential of virtual gain for imaging and loss compensation in metamaterials. Direct complex-frequency excitations in optical platforms (80) could further enable super-resolution imaging in microscopy.

Virtual gain can counteract losses and enhance quality factors in practical systems such as polariton propagation and high-Q sensors. Polaritons, arising from strong photon-matter coupling, are key to condensed matter research and photonic applications but suffer from intrinsic losses that material gain cannot easily compensate (134). Figure 6D shows how virtual gain from complex-frequency excitations mitigates these losses, extending polaritonic field decay lengths (81), with implications for imaging, sensing, lithography, computing, communications, and quantum technologies. In another context, virtual gain has been utilized to enhance the resonance of sensing systems by effectively increasing their Q factors, thereby amplifying sensitivity to perturbations. Figure 6E illustrates its effect in an infrared resonant biosensor (139). Although further study is needed to determine detection limits, virtual gain clearly offers a promising approach to loss compensation in metamaterials.

A
![](images/d8d143533a86c2097a4a8ac5589ea3bac3a606806d1520442e76d7f6849d9051.jpg)
Fig. 6. Wave phenomena enhanced by virtual gain. (A) Transfer function versus input wave number for a superlens with finite material loss, with absorption rate $\alpha$ , and for complex-frequency excitations varying the decay rate $\gamma$ . As $\gamma$ approaches $\alpha$ , a wider range of transverse wave numbers is transmitted, enhancing the superlens resolution [adapted with permission from (56)]. (B) Experimental demonstration of loss compensation for acoustic superlens, enhancing the resolution through virtual gain [adapted with permission from
(56)]. (C) Loss compensation of an infrared (IR) superlens utilizing a synthesized pulse with complex frequencies [adapted with permission from (57)]. (D) The decay length of phonon polaritons can be extended through complex-frequency excitations [adapted with permission from (81)]. hBN, hexagonal boron nitride. (E) Complex-frequency excitations to enhance the sensitivity of nanophotonic biosensors through virtual gain [adapted with permission from (139)]. CFW, complex-frequency wave.

## Conclusions and outlook

Complex-frequency excitations offer a powerful platform for wave control by introducing and tuning non-Hermitian phenomena through nonmonochromatic temporal waveforms. Extending the transfer function of linear systems into the complex-frequency plane allows dynamic manipulation of virtual gain and loss, overcoming Hermitian and passive limitations. Complex-frequency signals are temporally bounded and nonorthogonal. This means that a system under such excitations is not guaranteed to reach a quasi-steady state. However, with proper design, we can access complex zeros and poles, enabling effects such as virtual perfect absorption (88) and lasing (55). When combined with photonic engineering, a plethora of non-Hermitian phenomena can be unveiled, including CPA, RSMs, EPs (142, 143), PT-symmetry, the non-Hermitian skin effect, and other complex scattering phenomena. This technique introduces high tunability, allowing real-time control over gain and loss without altering material properties (80).

Whereas the assumption of system linearity enables analytical continuation into the complex-frequency plane, nonlinear interactions present new opportunities for complex-frequency excitations $(144)$ . For instance, virtual perfect absorption in nonlinear resonators may trigger saturation effects, decoupling the excitation and energy storage from its release once the excitation is stopped. Moreover, a pair of resonators exhibiting single-photon nonlinearities associated with an embedded eigenstate was utilized to demonstrate single-photon memories $(145)$ . By leveraging virtual perfect absorption in such nonlinear resonators, it may be possible to efficiently store signals and trap them within cavities after the excitation ceases. This opens untapped opportunities to combine tailored nonlinearities in nanophotonic systems with complex-frequency excitations.

In addition to classical applications, complex-frequency excitations hold promise for quantum technologies (146), particularly in improving quantum memory efficiency for single-photon capture and release (147, 148). Although demonstrated in acoustics, elastodynamics, and low-frequency electromagnetics, scaling to optical frequencies remains challenging owing to fast temporal modulation requirements. Solutions include all-optical modulation and high-Q resonances to shift complex-frequency singularities closer to the real axis (80). Alternatively, synthesizing complex-frequency excitations (Fig. 3H) through postprocessing (57, 81, 139) offers an alternative—albeit measurement- and computation-intensive—approach.

Looking forward, numerous opportunities are emerging to push devices and functionalities beyond the conventional limitations on bandwidth and efficiency imposed by passivity, for instance in the context of the Rozanov bound for absorbers (89, 149). This fundamental principle restricts how thin an absorber can be while still achieving large absorption across a wide frequency range. The use of non-Hermitian approaches has been explored to overcome this limit; thus excitations at complex frequencies may follow a more relaxed bandwidth-thickness trade-off. Furthermore, the concept of complex-frequency excitations may find analogies in other research domains. For instance, Deschamps introduced interesting radiation waveforms generated by sources whose spatial distribution can be described by emitters localized at a complex spatial coordinate, impacting electromagnetics and microwave engineering (150). This parallel suggests that complex-frequency techniques may be extended beyond the temporal domain and open exciting opportunities to synthesize complex-radiation sources.

Complex-frequency excitations offer a powerful framework for controlling wave-matter interactions. By integrating all-optical excitation schemes, designing optimal waveforms for quasi-steady states, and engineering resonant structures with tailored singularities at complex frequencies, these techniques can push wave phenomena beyond traditional limits in imaging, sensing, communications, computing, wave transport, and energy harvesting. Their impact spans classical and quantum technologies, redefining wave manipulation across scientific and technological domains.

## REFERENCES AND NOTES

1. C. F. Bohren, D. R. Huffman, Absorption and Scattering of Light by Small Particles (Wiley-VCH Verlag, 1998). doi: 10.1002/9783527618156

2. J. D. Jackson, Classical Electrodynamics (John Wiley & Sons, 2021).

3. P. M. Morse, K. U. Ingard, Theoretical Acoustics (Princeton Univ. Press, 1986).

4. K. F. Graff, Wave Motion in Elastic Solids (Dover Publications, 2012).

5. N. Engheta, R. W. Ziolkowski, Eds., Metamaterials: Physics and Engineering Explorations (John Wiley & Sons, 2006). doi: 10.1002/0471784192

6. A. F. Koenderink, A. Alù, A. Polman, Nanophotonics: Shrinking light-based technology. Science 348, 516–521 (2015). doi: 10.1126/science.1261243; pmid: 25931548

7. J. B. Pendry, Negative refraction makes a perfect lens. Phys. Rev. Lett. 85, 3966–3969 (2000). doi: 10.1103/PhysRevLett.85.3966; pmid: 11041972

8. N. Fang, H. Lee, C. Sun, X. Zhang, Sub-diffraction-limited optical imaging with a silver superlens. Science 308, 534–537 (2005). doi: 10.1126/science.1108759; pmid: 15845849

9. T. Taubner, D. Korobkin, Y. Urzhumov, G. Shvets, R. Hillenbrand, Near-field microscopy through a SiC superlens. Science 313, 1595 (2006). doi: 10.1126/science.1131025; pmid: 16973871

10. N. I. Zheludev, G. Yuan, Optical superoscillation technologies beyond the diffraction limit. Nat. Rev. Phys. 4, 16–32 (2021). doi: 10.1038/s42254-021-00382-7

11. Z. Yu, S. Fan, Complete optical isolation created by indirect interband photonic transitions. Nat. Photonics 3, 91–94 (2009). doi: 10.1038/nphoton.2008.273

12. R. Fleury, D. L. Sounas, C. F. Sieck, M. R. Haberman, A. Alù, Sound isolation and giant linear nonreciprocity in a compact acoustic circulator. Science 343, 516–519 (2014). doi: 10.1126/science.1246957; pmid: 24482477

13. N. Reiskarimian, H. Krishnaswamy, Magnetic-free non-reciprocity based on staggered commutation. Nat. Commun. 7, 11217 (2016). doi: 10.1038/ncomms11217; pmid: 27079524

14. J. Kim, S. Kim, G. Bahl, Complete linear optical isolation at the microscale with ultralow loss. Sci. Rep. 7, 1647 (2017). doi: 10.1038/s41598-017-01494-w; pmid: 28484213

15. H. Lira, Z. Yu, S. Fan, M. Lipson, Electrically driven nonreciprocity induced by interband photonic transition on a silicon chip. Phys. Rev. Lett. 109, 033901 (2012). doi: 10.1103/PhysRevLett.109.033901; pmid: 22861851

16. J. B. Pendry, D. Schurig, D. R. Smith, Controlling electromagnetic fields. Science 312, 1780–1782 (2006). doi: 10.1126/science.1125907; pmid: 16728597

17. D. Schurig et al., Metamaterial electromagnetic cloak at microwave frequencies. Science 314, 977–980 (2006). doi: 10.1126/science.1133628; pmid: 17053110

18. R. G. Newton, Scattering Theory of Waves and Particles (Springer, 2013).

19. A. Alù, N. Engheta, Achieving transparency with plasmonic and metamaterial coatings. Phys. Rev. E Stat. Nonlin. Soft Matter Phys. 72, 016623 (2005). doi: 10.1103/PhysRevE.72.016623; pmid: 16090123

20. W. Cai, U. K. Chettiar, A. V. Kildishev, V. M. Shalaev, Optical cloaking with metamaterials. Nat. Photonics 1, 224–227 (2007). doi: 10.1038/nphoton.2007.28

21. N. Fang et al., Ultrasonic metamaterials with negative modulus. Nat. Mater. 5, 452–456 (2006). doi: 10.1038/nmat1644; pmid: 16648856

22. Z. Yang, J. Mei, M. Yang, N. H. Chan, P. Sheng, Membrane-type acoustic metamaterial with negative dynamic mass. Phys. Rev. Lett. 101, 204301 (2008). doi: 10.1103/PhysRevLett.101.204301; pmid: 19113343

23. G. W. Milton, M. Briane, J. R. Willis, On cloaking for elasticity and physical equations with a transformation invariant form. New J. Phys. 8, 248 (2006). doi: 10.1088/1367-2630/8/10/248

24. N. Stenger, M. Wilhelm, M. Wegener, Experiments on elastic cloaking in thin plates. Phys. Rev. Lett. 108, 014301 (2012). doi: 10.1103/PhysRevLett.108.014301; pmid: 22304261

25. L. Zigoneanu, B.-I. Popa, S. A. Cummer, Three-dimensional broadband omnidirectional acoustic ground cloak. Nat. Mater. 13, 352–355 (2014). doi: 10.1038/nmat3901; pmid: 24608143

26. J. Zhu et al., A holey-structured metamaterial for acoustic deep-subwavelength imaging. Nat. Phys. 7, 52–55 (2011). doi: 10.1038/nphys1804

27. N. Kaina, F. Lemoult, M. Fink, G. Lerosey, Negative refractive index and acoustic superlens from multiple scattering in single negative metamaterials. Nature 525, 77–81 (2015). doi: 10.1038/nature14678; pmid: 26333466

28. S. A. Cummer, J. Christensen, A. Alù, Controlling sound with acoustic metamaterials. Nat. Rev. Mater. 1, 16001 (2016). doi: 10.1038/natrevmats.2016.1

29. L. Zhang et al., Space-time-coding digital metasurfaces. Nat. Commun. 9, 4334 (2018). doi: 10.1038/s41467-018-06802-0; pmid: 30337522

30. W. Li, Q. Yu, J. H. Qiu, J. Qi, Intelligent wireless power transfer via a 2-bit compact reconfigurable transmissive-metasurface-based router. Nat. Commun. 15, 2807 (2024). doi: 10.1038/s41467-024-46984-4; pmid: 38561373

31. Q. Wu, S. Zhang, B. Zheng, C. You, R. Zhang, Intelligent Reflecting Surface-Aided Wireless Communications: A

Tutorial. IEEE Trans. Commun. 69, 3313–3351 (2021). doi: 10.1109/TCOMM.2021.3051897

32. O. Astafiev et al., Resonance fluorescence of a single artificial atom. Science 327, 840–843 (2010). doi: 10.1126/science.1181918; pmid: 20150495

33. S. M. Anlage, The physics and applications of superconducting metamaterials. J. Opt. 13, 024001 (2011). doi: 10.1088/2040-8978/13/2/024001

34. C. W. Hsu, B. Zhen, A. D. Stone, J. D. Joannopoulos, M. Soljačić, Bound states in the continuum. Nat. Rev. Mater. 1, 16048 (2016). doi: 10.1038/natrevmats.2016.48

35. S. I. Azzam, A. V. Kildishev, Photonic Bound States in the Continuum: From Basics to Applications. Adv. Opt. Mater. 9, 2001469 (2021). doi: 10.1002/adom.202001469

36. I. Deriy, I. Toftul, M. Petrov, A. Bogdanov, Bound States in the Continuum in Compact Acoustic Resonators. Phys. Rev. Lett. 128, 084301 (2022). doi: 10.1103/PhysRevLett.128.084301; pmid: 35275659

37. N. Moiseyev, Non-Hermitian Quantum Mechanics (Cambridge Univ. Press, 2011). doi: 10.1017/CB09780511976186

38. Ş. K. Özdemir, S. Rotter, F. Nori, L. Yang, Parity-time symmetry and exceptional points in photonics. Nat. Mater. 18, 783–798 (2019). doi: 10.1038/s41563-019-0304-9; pmid: 30962555

39. A. Krasnok et al., Anomalies in light scattering. Adv. Opt. Photonics 11, 892 (2019). doi: 10.1364/AOP.11.000892

40. C. M. Bender, S. Boettcher, Real spectra in non-hermitian hamiltonians having PT symmetry. Phys. Rev. Lett. 80, 5243–5246 (1998). doi: 10.1103/PhysRevLett.80.5243

41. C. E. Rüter et al., Observation of parity-time symmetry in optics. Nat. Phys. 6, 192–195 (2010). doi: 10.1038/nphys1515

42. M. Sakhdari et al., Experimental Observation of PT Symmetry Breaking near Divergent Exceptional Points. Phys. Rev. Lett. 123, 193901 (2019). doi: 10.1103/PhysRevLett.123.193901; pmid: 31765193

43. C. M. Bender, B. K. Berntson, D. Parker, E. Samuel, Observation of PT phase transition in a simple mechanical system. Am. J. Phys. 81, 173–179 (2013). doi: 10.1119/1.4789549

44. X. Zhu, H. Ramezani, C. Shi, J. Zhu, X. Zhang, PT - Symmetric Acoustics. Phys. Rev. X 4, 031042 (2014). doi: 10.1103/PhysRevX.4.031042

45. G. Beck, H. M. Nussenzveig, On the physical interpretation of complex poles of the S-matrix - I. Nuovo Cim. 16, 416–449 (1960). doi: 10.1007/BF02731907

46. H. M. Nussenzveig, On the physical interpretation of complex poles of the S-matrix — II. Nuovo Cim. 20, 694–714 (1961). doi: 10.1007/BF02731560

47. E. S. C. Ching et al., Quasinormal-mode expansion for waves in open systems. Rev. Mod. Phys. 70, 1545–1554 (1998). doi: 10.1103/RevModPhys.70.1545

48. P. Lalanne, W. Yan, K. Vynck, C. Sauvan, J.-P. Hugonin, Light Interaction with Photonic and Plasmonic Resonances. Laser Photonics Rev. 12, 1700113 (2018). doi: 10.1002/lpor.201700113

49. Y. D. Chong, L. Ge, H. Cao, A. D. Stone, Coherent perfect absorbers: Time-reversed lasers. Phys. Rev. Lett. 105, 053901 (2010). doi: 10.1103/PhysRevLett.105.053901; pmid: 20867918

50. G. Gamow, Zur Quantentheorie des Atomkernes. Eur. Phys. J. A 51, 204–212 (1928). doi: 10.1007/BF01343196

51. Y. B. Zel'Dovich, On the Theory of Unstable States. Sov. Phys. JETP 12, 542–545 (1961).

52. Y. Ra'di, A. Krasnok, A. Alù, Virtual Critical Coupling. ACS Photonics 7, 1468–1475 (2020). doi: 10.1021/acsphotonics.0c00165

53. T. Delage, O. Pascal, J. Sokoloff, V. Mazières, Experimental demonstration of virtual critical coupling to a single-mode microwave cavity. J. Appl. Phys. 132, 153105 (2022). doi: 10.1063/5.0107041

54. S. Lepeshov, A. Krasnok, Virtual optical pulling force. Optica 7, 1024 (2020). doi: 10.1364/OPTICA.391569

55. S. Kim, S. Lepeshov, A. Krasnok, A. Alù, Beyond Bounds on Light Scattering with Complex Frequency Excitations. Phys. Rev. Lett. 129, 203601 (2022). doi: 10.1103/PhysRevLett.129.203601; pmid: 36462013

56. S. Kim, Y. G. Peng, S. Yves, A. Alù, Loss Compensation and Superresolution in Metamaterials with Excitations at Complex Frequencies. Phys. Rev. X 13, 041024 (2023). doi: 10.1103/PhysRevX.13.041024

57. F. Guan et al., Overcoming losses in superlenses with synthetic waves of complex frequency. Science 381, 766–771 (2023). doi: 10.1126/science.adi1267; pmid: 37590345

58. H. Li, A. Mekawy, A. Krasnok, A. Alù, Virtual Parity-Time Symmetry. Phys. Rev. Lett. 124, 193901 (2020). doi: 10.1103/PhysRevLett.124.193901; pmid: 32469571

59. Z. Gu et al., Transient non-Hermitian skin effect. Nat. Commun. 13, 7668 (2022). doi: 10.1038/s41467-022-35448-2; pmid: 36509774

60. M. Cai, O. Painter, K. J. Vahala, Observation of critical coupling in a fiber taper to a silica-microsphere whispering-gallery mode system. Phys. Rev. Lett. 85, 74–77 (2000). doi:10.1103/PhysRevLett.85.74; pmid: 10991162

61. S. Thongrattanasiri, F. H. L. Koppens, F. J. García de Abajo, Complete optical absorption in periodically patterned graphene. Phys. Rev. Lett. 108, 047401 (2012). doi: 10.1103/PhysRevLett.108.047401; pmid: 22400887

62. W. Wan et al., Time-reversed lasing and interferometric control of absorption. Science 331, 889–892 (2011). doi: 10.1126/science.1200735; pmid: 21330539

63. D. G. Baranov, A. Krasnok, T. Shegai, A. Alù, Y. Chong, Coherent perfect absorbers: Linear control of light with light. Nat. Rev. Mater. 2, 17064 (2017). doi: 10.1038/natrevmats.2017.64

64. L. Chen, T. Kottos, S. M. Anlage, Perfect absorption in complex scattering systems with or without hidden symmetries. Nat. Commun. 11, 5826 (2020). doi: 10.1038/s41467-020-19645-5; pmid: 33203847

65. W. R. Sweeney, C. W. Hsu, A. D. Stone, Theory of reflectionless scattering modes. Phys. Rev. A 102, 063511 (2020). doi: 10.1103/PhysRevA.102.063511

66. C. W. Hsu et al., Observation of trapped light within the radiation continuum. Nature 499, 188–191 (2013). doi: 10.1038/nature12289; pmid: 23846657

67. A. Kodigala et al., Lasing action from photonic bound states in continuum. Nature 541, 196–199 (2017). doi: 10.1038/nature20799; pmid: 28079064

68. M. Kang, T. Liu, C. T. Chan, M. Xiao, Applications of bound states in the continuum in photonics. Nat. Rev. Phys. 5, 659–678 (2023). doi: 10.1038/s42254-023-00642-8

69. J. Jin et al., Topologically enabled ultrahigh-Q guided resonances robust to out-of-plane scattering. Nature 574, 501–504 (2019). doi: 10.1038/s41586-019-1664-7; pmid: 31645728

70. S. Assawaworrarit, X. Yu, S. Fan, Robust wireless power transfer using a nonlinear parity-time-symmetric circuit. Nature 546, 387–390 (2017). doi: 10.1038/nature22404; pmid: 28617463

71. L. Chen, S. M. Anlage, Use of transmission and reflection complex time delays to reveal scattering matrix poles and zeros: Example of the ring graph. Phys. Rev. E 105, 054210 (2022). doi: 10.1103/PhysRevE.105.054210; pmid: 35706202

72. V. A. Chistyakov, A. Krasnok, Thermal Emission Control via Twist Tuning of Embedded Eigenstates in $\alpha$ -MoO 3 Nanostructures. ACS Appl. Nano Mater. 7, 1519–1525 (2024). doi: 10.1021/acsanm.3c03076

73. F. Binkowski, F. Betz, R. Colom, P. Genevet, S. Burger, Poles and zeros in non-Hermitian systems: Application to photonics. Phys. Rev. B 109, 045414 (2024). doi: 10.1103/PhysRevB.109.045414

74. E. Mikheeva et al., Asymmetric phase modulation of light with parity-symmetry broken metasurfaces. Optica 10, 1287 (2023). doi: 10.1364/OPTICA.495681

75. S. Esterhazy et al., Scalable numerical approach for the steady-state ab initio laser theory. Phys. Rev. A 90, 023816 (2014). doi: 10.1103/PhysRevA.90.023816

76. H. A. Haus, Waves and Fields in Optoelectronics (Prentice-Hall, 1984).

77. H. A. Haus, W. Huang, Coupled-mode theory. Proc. IEEE 79, 1505–1518 (1991). doi: 10.1109/5.104225

78. W. Suh, Z. Wang, S. Fan, Temporal coupled-mode theory and the presence of non-orthogonal modes in lossless multimode cavities. IEEE J. Quantum Electron. 40, 1511–1518 (2004). doi:10.1109/JQE.2004.834773

79. C. Gardiner, P. Zoller, Quantum Noise: A Handbook of Markovian and Non-Markovian Quantum Stochastic Methods with Applications to Quantum Optics (Springer, 2004).

80. J. Hinney et al., Efficient excitation and control of integrated photonic circuits with virtual critical coupling. Nat. Commun. 15, 2741 (2024). doi: 10.1038/s41467-024-46908-2; pmid: 38548757

81. F. Guan et al., Compensating losses in polariton propagation with synthesized complex frequency excitation. Nat. Mater. 23, 506–511 (2024). doi: 10.1038/s41563-023-01787-8; pmid: 38191633

82. A. E. Siegman, Lasers (University Science Books, 1986).

83. C. Ferise, P. del Hougne, M. Davy, Optimal matrix-based spatiotemporal wave control for virtual perfect absorption, energy deposition, and scattering-invariant modes in disordered systems. Phys. Rev. Appl. 20, 054023 (2023). doi: 10.1103/PhysRevApplied.20.054023

84. T. Delage et al., Plasma Ignition via High-Power Virtual Perfect Absorption. ACS Photonics 10, 3781–3788 (2023). doi: 10.1021/acsphotonics.3c01023

85. J. Wenner et al., Catching Time-Reversed Microwave Coherent State Photons with 99.4% Absorption Efficiency. Phys. Rev. Lett. 112, 210501 (2014). doi: 10.1103/PhysRevLett.112.210501

86. B. T. Taddese, T. M. Antonsen, E. Ott, S. M. Anlage, Sensing small changes in a wave chaotic scattering system. J. Appl. Phys. 108, 114911 (2010). doi: 10.1063/1.3518047

87. B. T. Taddese, J. Hart, T. M. Antonsen, E. Ott, S. M. Anlage, Sensor based on extending the concept of fidelity to classical waves. Appl. Phys. Lett. 95, 114103 (2009). doi: 10.1063/1.3232214

88. D. G. Baranov, A. Krasnok, A. Alù, Coherent virtual absorption based on complex zero excitation for ideal light capturing. Optica 4, 1457 (2017). doi: 10.1364/OPTICA.4.001457

89. Y. Ra'di, C. R. Simovski, S. A. Tretyakov, Thin Perfect Absorbers for Electromagnetic Waves: Theory, Design, and Realizations. Phys. Rev. Appl. 3, 037001 (2015). doi: 10.1103/PhysRevApplied.3.037001

90. S. Huang, Y. Li, J. Zhu, D. P. Tsai, Sound-Absorbing Materials. Phys. Rev. Appl. 20, 010501 (2023). doi: 10.1103/PhysRevApplied.20.010501

91. N. I. Landy, S. Sajuyigbe, J. J. Mock, D. R. Smith, W. J. Padilla, Perfect metamaterial absorber. Phys. Rev. Lett. 100, 207402 (2008). doi: 10.1103/PhysRevLett.100.207402; pmid: 18518577

92. Y. Slobodkin et al., Massively degenerate coherent perfect absorber for arbitrary wavefronts. Science 377, 995–998 (2022). doi: 10.1126/science.abq8103; pmid: 36007051

93. A.-S. B.-B. Dhia, L. Chesnel, V. Pagneux, Trapped modes and reflectionless modes as eigenfunctions of the same spectral problem. Proc. R. Soc. London Ser. A 474 20180050 (2018). doi: 10.1098/rspa.2018.0050

94. X. Jiang et al., Coherent control of chaotic optical microcavity with reflectionless scattering modes. Nat. Phys. 20, 109–115 (2024). doi: 10.1038/s41567-023-02242-w

95. J. Sol, A. Alhulaymi, A. D. Stone, P. Del Hougne, Reflectionless programmable signal routers. Sci. Adv. 9, eadf0323 (2023). doi: 10.1126/sciadv.adf0323; pmid: 36696503

96. A. Marini, D. Ramaccia, A. Toscano, F. Bilotti, Metasurface-bounded open cavities supporting virtual absorption: Freespace energy accumulation in lossless systems. Opt. Lett. 45, 3147–3150 (2020). doi: 10.1364/OL.389389; pmid: 32479481

97. A. V. Marini, D. Ramaccia, A. Toscano, F. Bilotti, Perfect Matching of Reactive Loads Through Complex Frequencies: From Circuital Analysis to Experiments. IEEE Trans. Antenn. Propag. 70, 9641–9651 (2022). doi: 10.1109/TAP.2022.3177571

98. Q. Zhong, L. Simonson, T. Kottos, R. El-Ganainy, Coherent virtual absorption of light in microring resonators. Phys. Rev. Res. 2, 013362 (2020). doi: 10.1103/PhysRevResearch.2.013362

99. G. Trainiti, Y. Ra'di, M. Ruzzene, A. Alù, Coherent virtual absorption of elastodynamic waves. Sci. Adv. 5, eaaw3255 (2019). doi: 10.1126/sciadv.aaw3255; pmid: 31497641

100. C. Rasmussen, M. I. N. Rosa, J. Lewton, M. Ruzzene, A Lossless Sink Based on Complex Frequency Excitations. Adv. Sci. 10, e2301811 (2023). doi: 10.1002/advs.202301811; pmid: 37587017

101. G. Ma et al., Towards anti-causal Green's function for three-dimensional sub-diffraction focusing. Nat. Phys. 14, 608–612 (2018). doi: 10.1038/s41567-018-0082-3

102. L. D. Landau, E. M. Lifshitz, Electrodynamics of Continuous Media (Pergamon Press, ed. 2, 1984).

103. A. Welters, Y. Avniel, S. G. Johnson, Speed-of-light limitations in passive linear media. Phys. Rev. A 90, 023847 (2014). doi: 10.1103/PhysRevA.90.023847

104. A. Srivastava, Causality and passivity: From electromagnetism and network theory to metamaterials. Mech. Mater. 154, 103710 (2021). doi: 10.1016/j.mechmat.2020.103710

105. G. Mie, Beiträge zur optik trüber medien speziell kolloidaler metallösungen. Ann. Phys. 330, 377–445 (1908). doi: 10.1002/andp.19083300302

106. H. C. van de Hulst, Light Scattering by Small Particles (Dover Publications, 1981).

107. Z. Ruan, S. Fan, Superscattering of light from subwavelength nanostructures. Phys. Rev. Lett. 105, 013901 (2010). doi: 10.1103/PhysRevLett.105.013901; pmid: 20867445

108. C. Qian et al., Experimental Observation of Superscattering. Phys. Rev. Lett. 122, 063901 (2019). doi: 10.1103/PhysRevLett.122.063901; pmid: 30822094

109. C. Qian et al., Breaking the fundamental scattering limit with gain metasurfaces. Nat. Commun. 13, 4383 (2022). doi: 10.1038/s41467-022-32067-9; pmid: 35902584

110. M. Kerker, D.-S. Wang, C. L. Giles, Electromagnetic scattering by magnetic spheres. J. Opt. Soc. Am. 73, 765–767 (1983). doi: 10.1364/JOSA.73.000765

111. A. Alù, N. Engheta, How does zero forward-scattering in magnetodielectric nanoparticles comply with the optical theorem? J. Nanophotonics 4, 041590 (2010). doi: 10.1117/1.3449103

112. L. Novotny, B. Hecht, Principles of Nano-Optics (Cambridge Univ. Press, ed. 2, 2012).

113. A. Mizrahi, Y. Fainman, Negative radiation pressure on gain medium structures. Opt. Lett. 35, 3405–3407 (2010). doi: 10.1364/OL.35.003405; pmid: 20967081

114. J. Chen, J. Ng, Z. Lin, C. T. Chan, Optical pulling force. Nat. Photonics 5, 531–534 (2011). doi: 10.1038/nphoton.2011.153

115. A. S. Shalin, S. V. Sukhov, A. A. Bogdanov, P. A. Belov, P. Ginzburg, Optical pulling forces in hyperbolic metamaterials. Phys. Rev. A 91, 063830 (2015). doi: 10.1103/PhysRevA.91.063830

116. A. Dogariu, S. Sukhov, J. Sáenz, Optically induced “negative forces”. Nat. Photonics 7, 24–27 (2013). doi: 10.1038/nphoton.2012.315

117. R. Fleury, D. Sounas, A. Alù, An invisible acoustic sensor based on parity-time symmetry. Nat. Commun. 6, 5905 (2015). doi: 10.1038/ncomms6905; prnid: 25562746

118. L. Ge, Y. D. Chong, A. D. Stone, Conservation relations and anisotropic transmission resonances in one-dimensional PT-symmetric photonic heterostructures. Phys. Rev. A 85, 023802 (2012). doi: 10.1103/PhysRevA.85.023802

119. Z. Lin et al., Unidirectional invisibility induced by PT-symmetric periodic structures. Phys. Rev. Lett. 106, 213901 (2011). doi: 10.1103/PhysRevLett.106.213901; pmid: 21699297

120. D. Trivedi, A. Madanayake, A. Krasnok, Anomalies in light scattering: A circuit-model approach. Phys. Rev. Appl. 22, 034061 (2024). doi: 10.1103/PhysRevApplied.22.034061

121. T. E. Lee, Anomalous Edge State in a Non-Hermitian Lattice. Phys. Rev. Lett. 116, 133903 (2016). doi: 10.1103/PhysRevLett.116.133903; pmid: 27081980

122. N. Okuma, K. Kawabata, K. Shiozaki, M. Sato, Topological Origin of Non-Hermitian Skin Effects. Phys. Rev. Lett. 124, 086801 (2020). doi: 10.1103/PhysRevLett.124.086801; pmid: 32167324

123. T. Helbig et al., Generalized bulk-boundary correspondence in non-Hermitian topoelectrical circuits. Nat. Phys. 16, 747–750 (2020). doi: 10.1038/s41567-020-0922-9

124. A. Ghatak, M. Brandenbourger, J. van Wezel, C. Coulais, Observation of non-Hermitian topology and its bulk-edge correspondence in an active mechanical metamaterial. Proc. Natl. Acad. Sci. U.S.A. 117, 29561–29568 (2020). doi: 10.1073/pnas.2010580117; pmid: 33168722

125. V. G. Veselago, The electrodynamics of substances with simultaneously negative values of $\varepsilon$ and $\mu$ . Sov. Phys. Usp. 10, 509–514 (1968). doi: 10.1070/PU1968v010n04ABEH003699

126. R. A. Shelby, D. R. Smith, S. Schultz, Experimental verification of a negative index of refraction. Science 292, 77–79 (2001). doi: 10.1126/science.1058847; pmid: 11292865

127. V. A. Podolskiy, E. E. Narimanov, Near-sighted superlens. Opt. Lett. 30, 75–77 (2005). doi: 10.1364/OL.30.000075; pmid: 15648643

128. I. A. Larkin, M. I. Stockman, Imperfect perfect lens. Nano Lett. 5, 339–343 (2005). doi: 10.1021/nl047957a; pmid: 15794622

129. P. Kinsler, M. W. McCall, Causality-based criteria for a negative refractive index must be used with care. Phys. Rev. Lett. 101, 167401 (2008). doi: 10.1103/PhysRevLett.101.167401; pmid: 18999712

130. B. Nistad, J. Skaar, Causality and electromagnetic properties of active media. Phys. Rev. E Stat. Nonlin. Soft Matter Phys. 78, 036603 (2008). doi: 10.1103/PhysRevE.78.036603; pmid: 18851176

131. S. Xiao et al., Loss-free and active optical negative-index metamaterials. Nature 466, 735–738 (2010). doi: 10.1038/nature09278; pmid: 20686570

132. M. P. H. Andresen, A. V. Skaldebø, M. W. Haakestad, H. E. Krogstad, J. Skaar, Effect of gain saturation in a gain compensated perfect lens. J. Opt. Soc. Am. B 27, 1610 (2010). doi: 10.1364/JOSAB.27.001610

133. M. I. Stockman, Nanoplasmonics: Past, present, and glimpse into future. Opt. Express 19, 22029–22106 (2011). doi: 10.1364/OE.19.022029; pmid: 22109053

134. J. B. Khurgin, A. Boltasseva, Reflecting upon the losses in plasmonics and metamaterials. MRS Bull. 37, 768–779 (2012). doi: 10.1557/mrs.2012.173

135. A. Archambault, M. Besbes, J.-J. Greffet, Superlens in the time domain. Phys. Rev. Lett. 109, 097405 (2012). doi: 10.1103/PhysRevLett.109.097405; pmid: 23002884

136. A. Rogov, E. Narimanov, Space-Time Metamaterials. ACS Photonics 5, 2868–2877 (2018). doi: 10.1021/acsphotonics.8b00233

137. J. Christensen, L. Martin-Moreno, F. J. Garcia-Vidal, Theory of resonant acoustic transmission through subwavelength apertures. Phys. Rev. Lett. 101, 014301 (2008). doi: 10.1103/PhysRevLett.101.014301; pmid: 18764114

138. G. Lerosey, J. de Rosny, A. Tourin, M. Fink, Focusing beyond the diffraction limit with far-field time reversal. Science 315, 1120–1122 (2007). doi: 10.1126/science.1134824; pmid: 17322059

139. K. Zeng et al., Synthesized complex-frequency excitation for ultrasensitive molecular sensing. eLight 4, 1 (2024). doi: 10.1186/s43593-023-00058-y

140. A. M. Maiden, M. J. Humphry, F. Zhang, J. M. Rodenburg, Superresolution imaging via ptychography. J. Opt. Soc. Am. A Opt. Image Sci. Vis. 28, 604–612 (2011). doi: 10.1364/JOSAA.28.000604; pmid: 21478956

141. P. Li, A. M. Maiden, Ten implementations of ptychography. J. Microsc. 269, 187–194 (2018). doi: 10.1111/jmi.12614; pmid: 28758682

142. W. R. Sweeney, C. W. Hsu, S. Rotter, A. D. Stone, Perfectly Absorbing Exceptional Points and Chiral Absorbers. Phys. Rev. Lett. 122, 093901 (2019). doi: 10.1103/PhysRevLett.122.093901; pmid: 30932516

143. C. Wang, W. R. Sweeney, A. D. Stone, L. Yang, Coherent perfect absorption at an exceptional point. Science 373, 1261–1265 (2021). doi: 10.1126/science.abj1028; pmid: 34516794

144. J. Leuthold, C. Koos, W. Freude, Nonlinear silicon photonics Nat. Photonics 4, 535–544 (2010). doi: 10.1038/nphoton.2010.185

145. M. Cotrufo, A. Alù, Excitation of single-photon embedded eigenstates in coupled cavity–atom systems. Optica 6, 799 (2019). doi: 10.1364/OPTICA.6.000799

146. D. V. Novitsky, Tunable virtual gain in resonantly absorbing media. Phys. Rev. A 107, 013516 (2023). doi: 10.1103/PhysRevA.107.013516

147. A. Farhi, W. Dai, S. Kim, A. Alù, D. Stone, Efficient general waveform catching by a cavity at an absorbing exceptional point. Phys. Rev. A 109, L041502 (2024). doi: 10.1103/PhysRevA.109.L041502

148. S. Zhang et al., Coherent control of single-photon absorption and reemission in a two-level atomic ensemble. Phys. Rev. Lett. 109, 263601 (2012). doi: 10.1103/PhysRevLett.109.263601; pmid: 23368560

149. K. N. Rozanov, Ultimate thickness to bandwidth ratio of radar absorbers. IEEE Trans. Antenn. Propag. 48, 1230–1234 (2000). doi: 10.1109/8.884491

150. G. A. Deschamps, Gaussian beam as a bundle of complex rays. Electron. Lett. 7, 684–685 (1971). doi: 10.1049/el:19710467

## ACKNOWLEDGMENTS

Funding: This work was supported by the Science and Technology Center New Frontiers of Sound (NewFoS) through NSF cooperative agreement no. 2242925, the Department of Defense, and the Simons Foundation. Author contributions: All authors contributed to the paper conception, preparation, and revision. Competing interests: The authors declare that they have no competing interests. License information: Copyright © 2025 the authors, some rights reserved; exclusive licensee American Association for the Advancement of Science. No claim to original US government works. https://www.science.org/about/science-licenses-journal-article-reuse

Submitted 13 November 2024; accepted 20 February 2025 10.1126/science.ado4128
