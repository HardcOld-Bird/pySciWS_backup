# A Time Grating Approach to Ultrahigh-Q Guided Mode Resonance

Youxiu Yu $^{1}$ | Xiaofeng Xu $^{2}$ | Yang Long $^{3}$ | Gui-Geng Liu $^{4}$ | Dongliang Gao $^{2}$ | Xiao Lin $^{5}$ | Hao Hu $^{1}$

$^{1}$ National Key Laboratory of Microwave Photonics & College of Electronic and Information Engineering, Nanjing University of Aeronautics and Astronautics, Nanjing, China | $^{2}$ College of Physical Science and Technology, Soochow University, Suzhou, China | $^{3}$ School of Physics Science and Engineering, Tongji University, Shanghai, China | $^{4}$ Department of Electronic and Information Engineering, School of Engineering, Westlake University, Hangzhou, China | $^{5}$ Interdisciplinary Center for Quantum Information, State Key Laboratory of Modern Optical Instrumentation, Zhejiang University, Hangzhou, China

Correspondence: Hao Hu (hao.hu@nuaa.edu.cn)

Received: 5 January 2026 | Revised: 22 March 2026 | Accepted: 2 April 2026

Keywords: Goos-Hänchen shift | guided mode resonance | quality factor | time grating

## ABSTRACT

Guided mode resonance (GMR), the resonant coupling of free-space light into leaky waveguide modes, is traditionally achieved with periodic patterned structures. However, this approach makes its key properties, such as quality factor (Q-factor), fabrication-dependent and non-tunable. Here, we introduce a time grating platform, i.e., a homogeneous waveguide whose refractive index is modulated periodically in time, that allows tunable GMRs through temporal modulation engineering rather than spatial structural redesign. We show that the Q-factors of these GMRs diverge as the modulation depth vanishes. Furthermore, unconstrained by energy conservation, the resonances exhibit near-unity reflection for fundamental harmonics and values exceeding 40 for first-order harmonics. Our findings not only apply to yield a giant Goos–Hänchen shift over $10^{3}$ times wavelength without sacrificing the reflection magnitude, but also open new avenues for related phenomena such as bound states in the continuum, unidirectional GMRs, and beyond.

## 1 | Introduction

Guided mode resonance (GMR) occurs when free-space waves couple into guided modes through periodic structures such as gratings or photonic crystal slabs $[1, 2]$ . Owing to its direct accessibility from free-space, GMR has enabled a broad range of applications, including molecular sensing, spectral filtering, light emission, etc $[1, 3–5]$ . However, the inherently leaky nature of GMR inevitably leads to radiation losses, fundamentally constraining the achievable Q-factor and compromising the performance of GMR-based devices. These losses become even more pronounced in practical implementations, where fabrication imperfections such as structural defects and disorders further exacerbate the mode leakage. Consequently, mitigating radiation losses remains a key challenge in the design of high-Q GMR-based photonic devices.

Subsequent studies have demonstrated that radiation losses of GMRs can be entirely suppressed at specific wavevectors, giving rise to a class of states known as bound states in the continuum (BICs) $[6–8]$ . These states feature theoretically infinite Q-factors due to their complete decoupling from free-space radiation. However, such a perfect confinement also precludes any energy exchange between the guided modes and external waves. To achieve controlled radiation while retaining high Q-factors, a moderate periodic perturbation is typically introduced into the BIC system. Unfortunately, as previously mentioned, fabrication imperfections in these patterned structures would induce unwanted radiation losses, greatly limiting the stability of the Q-factors. Thus, it is interesting to ask whether there is an alternate mechanism to enhance the robustness and environmental tolerance of high-Q GMR systems.

Recently, time-varying media (such as time photonic crystals and time gratings) have been recognized as transformative platforms for wave manipulation $[9–13]$ . Unlike their spatial counterparts, these media feature constitutive parameters (e.g., permittivity and conductivity) that are rapidly modulated in time through optical or electrical signals $[14, 15]$ . Research reveals that time-varying media not only replicate a variety of electromagnetic phenomena observed in static systems (e.g., double-slit diffraction, Anderson localization, topological bulk-edge correspondence) $[16–19]$ , but also unlock unconventional behaviors such as broadband frequency translation, coherent wave control with different incident frequencies, non-resonant amplification, and threshold-free Cherenkov radiation $[20–24]$ . Crucially, because they do not rely on spatially patterned structures, time-varying systems are inherently robust against fabrication imperfections. Their high degree of tunability and temporal reconfigurability offer enhanced environmental adaptability, making them promising candidates for next-generation photonic devices. This naturally raises the question: can time-varying media be harnessed to achieve GMR with enhanced Q-factor control and resilience?

In this work, we propose a time grating approach to achieve ultrahigh-Q GMR. The time grating, i.e., a homogeneous dielectric waveguide with permittivity periodically modulated in time, supports GMR when the energy mismatch between free-space waves and guided modes is compensated by the modulation frequency. Remarkably, the Q-factor of the resulting GMR can be readily controlled by adjusting the modulation depth. For instance, as the modulation depth is reduced to 0.005, we observe an ultrahigh Q-factor exceeding $10^{5}$ . Moreover, because the temporal modulation breaks the energy conservation, the system exhibits highly nontrivial reflection behaviors: a near-unity reflection coefficient is achieved for the fundamental harmonic, while the first-order harmonic reflection coefficient can even exceed 40. The combination of ultrahigh Q-factors and amplified reflection makes time-grating-based GMR an indispensable ingredient for Goos–Hänchen (GH) shift enhancement, overcoming the traditional trade-off between lateral shift magnitude and reflection. These findings open new avenues for dynamically reconfigurable, fabrication-tolerant photonic devices and pave the way toward time-modulated platforms for advanced sensing, signal processing, and active wavefront control.

## 2 | Principle

Without loss of generality, we consider a time grating made of a homogeneous dielectric waveguide with a sinusoidal modulation of permittivity in time. The permittivity of a dielectric waveguide is expressed as $\varepsilon_{r}(t)=\varepsilon_{r0}[1+\delta\cos(\Omega t)]$ , where $\varepsilon_{r0}$ is the unperturbed permittivity, $\delta$ is the modulation depth, and $\Omega$ is the modulation frequency. Note that the modulation frequency is used to form the time grating and has no direct relationship with the incident frequency described below. For clarity and simplicity, the terms “modulation frequency” and “incidence frequency” used here specifically refer to the angular frequency. As plotted in Figure 1b, the surface of the time grating aligns with the y-z plane, while the surface normal is oriented along the x-axis. To achieve platform-independent universality, we present our results using normalized quantities for frequencies and thicknesses. The normalized thickness of the time grating is adopted as $L_{TG} = L \sqrt{\varepsilon_{r0}} \Omega / (2\pi c)$ . Unless otherwise specified, the unperturbed permittivity and modulation frequency are taken as $\varepsilon_{r0} \approx 3.2$ and $\Omega \approx 555.88$ THz, respectively, in the following analytical calculations. We emphasize that such a parameter setup is experimentally feasible by employing indium tin oxide (ITO) as the material platform, which exhibits pronounced nonlinear optical responses. First, the unperturbed permittivity of ITO is $\varepsilon_{r0} \approx 3.2$ in the wavelength band of 0.6–0.66 $\mu$ m (corresponding to the working frequency band of 5.1–5.6 $\Omega$ ). Second, the rise time of ITO has been reported to be only a few femtoseconds, which allows it to instantaneously respond to the external optical pumping with the modulation frequency $\Omega$ considered here [16]. We remark that the specific parameter values adopted here are intended as representative examples rather than strict constraints of the proposed mechanism.

First, we present the excitation condition of GMRs in a time grating. Initially, a free-space incidence with transverse-magnetic (TM) polarization cannot excite GMR in the waveguide without external modulations. This is straightforward because the frequency of the incident wave $\omega_{i}=(k_{y}c)/\sin\theta$ (with $\theta$ being the incident angle) is different from that of the guided mode $\omega_{g}$ at the same tangential wavevector $k_{y}$ . However, by introducing a periodic temporal modulation, the system provides additional frequency compensation as $\Delta\omega=m(2\pi/T)$ to the incident wave. Here, m is an arbitrary integer. Such a frequency compensation couples the TM incidence to the guided mode if

$$
\omega_ {g} (k _ {y}) = \omega_ {i} (k _ {y}) \pm m \frac {2 \pi}{T}\tag{1}
$$

where $T = 2\pi/\Omega$ corresponds to the temporal modulation period (see mode transition diagram in Figure 1d). This excitation mechanism is fundamentally different from that in a conventional space grating. In a typical space grating as shown in Figure 1a, the structural periodicity (with a period denoted as P) introduces an extra momentum compensation, $\Delta k = m(2\pi/P)$ , to incident waves. Then, GMR is excited if the difference between the incident wavevector component $k_{y,i} = k_{0} \sin \theta$ and the guided mode' one $k_{y,g}$ is compensated by $\Delta k$ at a given frequency $\omega$ (Figure 1c). In other words, the excitation condition of GMR in space grating is expressed as $k_{y,g}(\omega) = k_{y,i}(\omega) \pm m(2\pi/P)$ .

## 3 | Result and Discussion

The excitation of GMRs in time grating is spectrally manifested as sharp peaks in the reflection spectrum. To illustrate this, we plot in Figure 2b the reflection coefficient as a function of incident angle and frequency for the fundamental harmonic in time grating (For the analytical calculation of the reflection coefficient, see Section S2). Obviously, the reflection coefficient is significantly enhanced at specific incident angles and frequencies. For example, at the incident angle of $64.5^{\circ}$ , a sharp asymmetric Fano line shape is observed at the frequency of $\omega_{i} = 5.3\Omega$ (see the red dot in Figure 2b and its inset). Such a Fano line shape originates from the interference between high-Q GMR and low-Q Fabry-Pérot resonance. We find that the parameters leading to the enhanced reflection are in excellent agreement with theoretical predictions using Equation (1), namely that the resonance of this structure can be simply predicted through the

![](images/7592b3c47b5ebb47f092e7eccb96c89706a8a6d0cda47a4fca3ba66b3770dd23.jpg)

![](images/998b4f59528f4919c6c3d2642639ad525c27ac55f3d6fb016e95978e34194bb7.jpg)

![](images/626758212d3b6842eec0155c433af4e797ba6773353507a04d7ecce4ca9a7ed8.jpg)
10

![](images/f0bd7747ec8741f923a6e4642594df431319603fd3a971ccc2468972d07ccb9a.jpg)
FIGURE 1 | Comparison between GMRs in space grating and time grating. (a) Schematic of GMR in a space grating. (b) Schematic of GMR in a time grating. In space grating, the effective permittivity is periodically changed along $y$ axis, while in time grating, the permittivity is periodically changed along time axis. The modulation periods of the space and time grating are denoted as $P = 2\pi c / \Omega$ and $T = 2\pi / \Omega$ , respectively. (c) Excitation mechanism of GMR in the space grating. (d) Excitation mechanism of GMR in the time grating. In (c,d), the black solid lines depict the dispersion in the waveguide, the black dashed lines depict the dispersion in the air background, whereas the red and orange dashed lines indicate the dispersion relations for $m = 1, 2$ in the space grating and $m = -1, -2$ in the time granted. In all the panels, the unperturbed permittivity, modulation depth, and normalized thickness of the time grating are $\varepsilon_{r0} \approx 3.2$ , $\delta = 0.2$ , and $L_{TG} = 1/4\pi$ , respectively. The incidence angle is $\theta = 64.5^{\circ}$ .

GMR induced by frequency compensation. For comparison, in a conventional waveguide without time modulation, the above sharp asymmetric Fano line shapes disappear in Figure 2a. These results further show that time modulation offers an alternative route for achieving GMRs.

Next, we demonstrate that the Q-factor of GMR in the time grating could be flexibly controlled by adjusting the modulation depth. The Q-factor is extracted from the reflection spectrum of the fundamental harmonic in Figure 3a, under an incident angle of $\theta = 64.5^{\circ}$ . When the modulation depth is relatively large, e.g., $\delta = 0.5$ , The GMR exhibits a broad linewidth, corresponding to a small Q-factor of 22 (Figure 3b). By contrast, if the modulation depth is sufficiently reduced to, e.g., $\delta = 0.005$ , an extremely narrow resonance linewidth emerges. Then the Q-factor is remarkably enhanced to a value exceeding $10^{5}$ . With a further reduction of the modulation depth, the Q-factor can increase even more and theoretically approach infinity, which can be practically achieved by decreasing the intensity of the external signals. Importantly, this tunable control of Q-factors is realized without any structural reconfiguration (i.e., without requiring special spatial structures), highlighting the unique advantage of our time grating system. It noted that the time grating degenerates into a purely spatial waveguide when the modulation depth is reduced to zero. Under this condition, the Q-factor is infinitely larger in theory due to the waveguide mode lies below the light cone.

A small modulation depth in the time grating not only results in ultrahigh-Q GMR, but also induces strong reflection at the resonant frequency and its higher-order harmonics. As the modulation depth decreases, the resonance linewidth becomes narrower; however, the maximum reflection coefficient at the fundamental harmonic consistently remains unity, as shown in Figure 3a. This phenomenon is caused by constructive interference between the GMR and the Fabry–Pérot background under time modulation, as shown in the inset of Figure 2 and Figure S1. At higher-order harmonic frequencies in the time grating, the reflection coefficient can even exceed unity. This can be learned from Figure 3c, presenting the first-order harmonic reflection coefficient as a function of incident frequency and modulation depth. Remarkably, as the modulation depth decreases, the reflection coefficients become even higher: the calculated reflection coefficients are 3.5, 4.7, 6.6, 10.2, 20.8, and 41.4 for modulation depths of 0.5, 0.4, 0.3, 0.2, 0.1, and 0.05, respectively. This unconventional reflection spectrum arises from time-modulation-induced amplification, where the energy from the modulation signal is constantly fed into the GMR (see Section S3). In sharp contrast, in a conventional grating, the reflection coefficients of high harmonics are generally small and become negligible when the structural perturbation is sufficiently weak.

$10^{6}$
![](images/76ed064cdc81c669a0eb5868d6b8fbc608f1fb63a8e3b76c5b9577cad6fda983.jpg)
FIGURE 2 | Comparison of reflection spectra in waveguides with and without time modulation. (a) Reflection spectra in waveguides without time modulation. (b) Reflection spectra in waveguides with time modulation. In (a, b), the insets correspond to the reflection coefficient as a function of the incident frequency at $\theta = 64.5^{\circ}$ (as indicated by the red dashed line in (a,b)). As marked by the red dot in (b), the peak of the reflection coefficient occurs at $\theta = 64.5^{\circ}$ and $\omega_{i} = 5.3\Omega$ .

![](images/4adf2363d714193cc8385c3e46df92fd32d858422330ce72bc585b8b1e23fa31.jpg)

![](images/33a7f158a121a2cc141fc9f225d3ac772f4c5cef49ae896e47f72e36d1720c70.jpg)

![](images/dcbb5e1aa0aaa2e35713014635f2567fdd8b7b7877c2832105d930d7cbd436eb.jpg)
FIGURE 3 | Influence of modulation depth on reflection spectra of time grating and the Q-factor of GMR. (a) Influence of modulation depth on reflection spectra of the time grating for the fundamental harmonic. (b) Influence of modulation depth on Q-factor of GMR. (c) Influence of modulation depth on reflection spectra of time grating for the first-order harmonic. In all panels, the incident angle is $\theta = 64.5^{\circ}$ .

Last but not least, we reveal that our proposed GMR in time grating offers an indispensable way to engineer GH shift. GH shift is known as the lateral displacement that occurs in geometric space when an incident beam impinges on the interface between two different media (see schematic in Figure 4a) [25, 26]. As mentioned in Ref. [26], the GH shift is quantified as $\Delta GH =$ $-(\lambda/2\pi\cos(\theta))d\varphi/d\theta$ , where $\varphi$ is the phase of the reflection coefficient, and $\lambda$ is the incidence wavelength. Typically, the GH shift is very small (on the order of a wavelength or less). Due to this reason, while GH shift shows great promise for applications in precision measurement, optical switches, and wavelength division multiplexers [27–29], this small magnitude makes it difficult to be detected and exploited in practical systems. Although previous work proposed Brewster effect or resonance-based mechanisms to enhance GH shift [8, 30–33], the corresponding reflection magnitude is typically too small to be detected. Therefore, achieving a balance between the GH shift length and reflection magnitude remains a fundamental challenge.

![](images/e4e8d603022503a2b923b5c64a238715b1ff1612bc7a818b437d6dd3ac615387.jpg)

![](images/d69cdba72b56a2c2a103a0abff0eebbbe3b369f9c3fc61da118e12989144bda2.jpg)

![](images/cd2d84060bbdf20fe9ffa07dcf44118414421b72740e7887119d1189b30dce9c.jpg)

![](images/47a6b2bdbc27fe7e748ef7506ecd81145ea32c370abe50ca6260be1ceadbf495.jpg)

![](images/4da91c2f852540536ec2f38cb4286e0890eeadf165b5263f1a629ff1dc423d61.jpg)

![](images/3a5165fe2c977354848556787afc4eb37834a280c2f86880def26223077492e5.jpg)

![](images/58a21c1b38a498b3bfdfbc082df609546546dae4815c6895d583cdffbd24611f.jpg)
FIGURE 4 | GH shift in the time grating. (a) Schematic of the GH shift enhanced by GMR in the time grating. (b) GH shifts of the first-order harmonic as a function of incidence angle and modulation depth in the time grating. (c) Reflection coefficient of first-order harmonic as a function of incidence angle in the time grating. (d) The phase angle of the reflection coefficient as a function of incident angle in the time grating. (e) GH shift of the first-order harmonic as a function of incidence angle in the time grating. (f) Relation between maximum GH shift and modulation depth. In all panels, the incident frequency is 5.3 Ω.

The above challenge can be effectively addressed through the GMR enabled by our time-grating platform. The GH shift as a function of incident angle and modulation depth is shown in Figure 4b. At a fixed modulation depth, the maximum GH shift is observed at the incident angle of $\theta = 64.5^{\circ}$ , where GMR is excited in time grating. This is because GMR induces a sharp phase variation as the incident angle varies, favorably enhancing the GH shift (see Figure 4d). Such a phase variation becomes even more drastic as the modulation depth goes to zero, and as a result, the GH shift enhancement is further optimized (Figure 4c,e). More quantitatively, the GH shift exhibits a linear dependence on the modulation depth $1 / \delta^{2}$ (Figure 4f). Remarkably, a giant GH shift exceeds over $10^{3}$ times the wavelength when the modulation depth is reduced to $\delta = 0.05$ (Figure 4e). Such a giant GH shift does not sacrifice the magnitude of reflection (e.g., the reflection coefficient of the first-order harmonic is as large as 41.4 if the modulation depth is $\delta = 0.05$ ). These findings suggest that GMR in time grating could simultaneously enhance GH shift strength and reflection magnitude, paving the way for advanced optical manipulation and precision metrology applications.

## 4 | Conclusion

In this work, we successfully extend the concept of GMR from space to time grating. Such an extension is nontrivial, as the Q-factors of GMR, initially tuned by structural reconfiguration, could now be flexibly engineered by adjusting the modulation depth in a flat waveguide. By minimizing the modulation depth, the revealed GMR not only possesses an ultrahigh Q-factor up to $10^{5}$ , but also induces strong reflection coefficients exceeding 40 for first-order harmonics. These exotic properties of GMR in time grating make it a powerful platform to enhance GH shift with strong reflection, overcoming the conventional trade-off between GH shift length and reflection magnitude. Our configuration could be implemented in a variety of physical systems. For example, the time grating can be realized in a waveguide made of nonlinear materials such as indium tin oxide (ITO), whose refractive index is varied periodically in time by employing a temporally modulated signal [16, 34]. Such time grating can also be achieved in time-varying transmission lines or metasurfaces, where the refractive index is temporally controlled by adopting optoelectronic components, such as varactor diodes and photodiodes [35–39].

Our work also inspires future exploration of rich physics related to GMR in time grating. One promising opportunity is that

BIC modes, which are the limiting case of GMRs with vanishing radiation loss, may arise under proper temporal symmetry or parameter conditions. Furthermore, introducing traveling-wave modulations can break mirror symmetry and render the revealed GMR unidirectional. Such BICs, unidirectional GMRs, and beyond in time gratings offer distinct advantages of strong tunability and unconventional spectral features compared to their conventional counterparts, and thus hold great potential for practical applications in the dynamic control of light-matter interactions.

## Acknowledgements

This work was supported by the National Natural Science Foundation of China (Grant Nos. 12404363, 12174281), Natural Science Foundation of Jiangsu Province (Grant No. BK20241374), Distinguished Professor Fund of Jiangsu Province, and Fundamental Research Funds for the Central Universities, NUAA (Grants Nos. NS2024022, NE2024007).

## Conflicts of Interest

The authors declare no conflict of interest.

## Data Availability Statement

The data that supports the findings of this study are available in the supplementary material of this article.

## References

1. S. S. Wang and R. Magnusson, “Theory and Applications of Guided-Mode Resonance Filters,” Applied Optics 32 (1993): 2606–2613, https://doi.org/10.1364/AO.32.002606.

2. G. Quaranta, G. Basset, O. J. F. Martin, and B. Gallinet, “Recent Advances in Resonant Waveguide Gratings,” Laser & Photonics Reviews 12 (2018): 1800017, https://doi.org/10.1002/lpor.201800017.

3. Y. S. Choi, C. Y. Park, S.-C. An, J. H. Pyo, and J. W. Yoon, "Topological Guided-Mode Resonances: Basic Theory, Experiments, and Applications," Nanophotonics 14 (2025): 1069–1082, https://doi.org/10.1515/nanoph-2024-0612.

4. G. Lan, S. Zhang, H. Zhang, et al., “High-Performance Refractive Index Sensor Based on Guided-Mode Resonance in All-Dielectric Nano-Silt Array,” Physics Letters A 383 (2019): 1478–1482, https://doi.org/10.1016/j.physleta.2019.01.057.

5. B. Zhao, L. Lin, and M. Lawrence, “Polarization-Independent High-Q Phase Gradient Metasurfaces,” Nano Letters 25 (2025): 1862–1869, https://doi.org/10.1021/acs.nanolett.4c05260.

6. J. Yu, W. Yao, M. Qiu, and Q. Li, "Free-Space High-Q Nanophotonics," Light: Science & Applications 14 (2025): 174, https://doi.org/10.1038/s41377-025-01825-x.

7. L. Huang, R. Jin, C. Zhou, et al., “Ultrahigh-Q Guided Mode Resonances in an All-Dielectric Metasurface,” Nature Communications 14 (2023): 3433, https://doi.org/10.1038/s41467-023-39227-5.

8. F. Wu, J. Wu, Z. Guo, et al., “Giant Enhancement of the Goos-Hänchen Shift Assisted by Quasibound States in the Continuum,” Physical Review Applied 12 (2019): 014028, https://doi.org/10.1103/PhysRevApplied.12.014028.

9. M. M. Asgari, P. Garg, X. Wang, M. S. Mirmoosa, C. Rockstuhl, and V. Asadchy, “Theory and Applications of Photonic Time Crystals: A Tutorial,” Advances in Optics and Photonics 16 (2024): 958–1063, https://doi.org/10.1364/AOP.525163.

10. E. Galiffi, R. Tirole, S. Yin, et al., “Photonics of Time-Varying Media,” Advanced Photonics 4 (2022): 014002, https://doi.org/10.1117/1.AP.4.1.014002.

11. Z. Gong, R. Chen, H. Chen, and X. Lin, “Anomalous Maxwell-Garnett Theory for Photonic Time Crystals,” Applied Physics Letters 12 (2025): 031414.

12. S. Yin and A. Alù, “Efficient Phase Conjugation in a Space-Time Leaky Waveguide,” ACS Photonics 9 (2022): 979–984.

13. J. Jiang, H. Hu, Y. Long, L. Liu, S. Hou, and D. Liu, “Broadband Temporal Localization and Delocalized Temporal Edge States in Time Photonic Crystals,” arXiv (2026): arXiv:2603.15115.

14. M. Zahirul Alam, I. D. Leon, and R. W. Boyd, “Large Optical Nonlinearity of Indium Tin Oxide in Its Epsilon-Near-Zero Region,” Science 352 (2016): 795–797, https://doi.org/10.1126/science.aae0330.

15. E. Galiffi, G. Xu, S. Yin, H. Moussa, Y. Ra'di, and A. Alù, "Broadband Coherent Wave Control Through Photonic Collisions at Time Interfaces," Nature Physics 19 (2023): 1703–1708, https://doi.org/10.1038/s41567-023-02165-6.

16. R. Tirole, S. Vezzoli, E. Galiffi, et al., “Double-Slit Time Diffraction at Optical Frequencies,” Nature Physics 19 (2023): 999–1002, https://doi.org/10.1038/s41567-023-01993-w.

17. Y. Sharabi, E. Lustig, and M. Segev, “Disordered Photonic Time Crystals,” Physical Review Letters 126 (2021): 163902, https://doi.org/10.1103/PhysRevLett.126.163902.

18. E. Lustig, Y. Sharabi, and M. Segev, “Topological Aspects of Photonic Time Crystals,” Optica 5 (2018): 1390–1395.

19. Y. Yang, H. Hu, L. Liu, et al., “Topologically Protected Edge States in Time Photonic Crystals With Chiral Symmetry,” ACS Photonics 12 (2025): 2389–2396, https://doi.org/10.1021/acsphotonics.4c01785.

20. E. Galiffi, P. A. Huidobro, and J. B. Pendry, “Broadband Nonreciprocal Amplification in Luminal Metamaterials,” Physical Review Letters 123 (2019): 206101, https://doi.org/10.1103/PhysRevLett.123.206101.

21. Y. Zhou, M. Z. Alam, M. Karimi, et al., “Broadband Frequency Translation Through Time Refraction in an Epsilon-Near-Zero Material,” Nature Communications 11 (2020): 2180, https://doi.org/10.1038/s41467-020-15682-2.

22. Y. Yu, D. Gao, Y. Yang, et al., “Generalized Coherent Wave Control at Dynamic Interfaces,” Laser & Photonics Reviews 19 (2024): 2400399, https://doi.org/10.1002/lpor.202400399.

23. J. B. Khurgin, "Photonic Time Crystals and Parametric Amplification: Similarity and Distinction," ACS Photonics 11 (2024): 2150-2159.

24. D. Oue, K. Ding, and J. B. Pendry, “Čerenkov Radiation in Vacuum from a Superluminal Grating,” Physical Review Research 4 (2022): 013064, https://doi.org/10.1103/PhysRevResearch.4.013064.

25. F. Goos and H. Hänchen, “Ein Neuer Und Fundamentaler Versuch Zur Totalreflexion,” Annalen der Physik 436 (1947): 333–346, https://doi.org/10.1002/andp.19474360704.

26. K. Artmann, “Berechnung der Seitenversetzung des Totalreflektierten Strahles,” Annalen der Physik 437 (1948): 87–102, https://doi.org/10.1002/ andp.19484370108.

27. A. Farmani, A. Mir, and Z. Sharifpour, “Broadly Tunable and Bidirectional Terahertz Graphene Plasmonic Switch Based on Enhanced Goos-Hänchen Effect,” Applied Surface Science 453 (2018): 358–364, https://doi.org/10.1016/j.apsusc.2018.05.092.

28. D. Chauvat, O. Emile, F. Bretenaker, and A. L. Floch, “Direct Measurement of the Wigner Delay Associated With the Goos-Hänchen Effect,” Physical Review Letters 84 (2000): 71–74, https://doi.org/10.1103/PhysRevLett.84.71.

29. D. Xu, S. He, J. Zhou, S. Chen, S. Wen, and H. Luo, “Goos-Hänchen Effect Enabled Optical Differential Operation and Image Edge Detection,” Applied Physics Letters 116 (2020): 211103, https://doi.org/10.1063/5.0006483.

30. X. Zhou, S. Liu, Y. Ding, L. Min, and Z. Luo, “Precise Control of Positive and Negative Goos-Hänchen Shifts in Graphene,” Carbon 149 (2019): 604–608, https://doi.org/10.1016/j.carbon.2019.04.064.

31. I. V. Soboleva, V. V. Moskalenko, and A. A. Fedyanin, “Giant Goos-Hänchen Effect and Fano Resonance at Photonic Crystal Surfaces,” Physical Review Letters 108 (2012): 123901, https://doi.org/10.1103/PhysRevLett.108.123901.

32. M. Wei, Y. Long, F. Wu, G.-G. Liu, and B. Zhang, “Abrupt lateral Beam Shifts from Terahertz Quasi-Bound States in the Continuum,” Science Bulletin 70 (2025): 882–888, https://doi.org/10.1016/j.scib.2025.01.006.

33. Y. Huang, G. Tang, J. Chen, Z.-Y. Li, and W. Liang, “Adjustable Enhanced Goos-Hänchen Shift in a Magneto-Optic Photonic Crystal Waveguide,” Optics Express 30 (2022): 36478–36488, https://doi.org/10.1364/OE.470009.

34. R. Tirole, S. Vezzoli, D. Saxena, et al., “Second Harmonic Generation at a Time-Varying Interface,” Nature Communications 15 (2024): 7752, https://doi.org/10.1038/s41467-024-51588-z.

35. T. R. Jones, A. V. Kildishev, M. Segev, and D. Peroulis, “Time-Reflection of Microwaves by a Fast Optically-Controlled Time-Boundary,” Nature Communications 15 (2024): 6786, https://doi.org/10.1038/s41467-024-51171-6.

36. J. Sisler, P. Thureja, M. Y. Grajower, R. Sokhoyan, I. Huang, and H. A. Atwater, “Electrically Tunable Space–Time Metasurfaces at Optical Frequencies,” Nature Nanotechnology 19 (2024): 1491–1498, https://doi.org/10.1038/s41565-024-01728-9.

37. X. Ye, Y. G. Wang, J. F. Yao, Y. Wang, C. X. Yuan, and Z. X. Zhou, “Realization of Spatiotemporal Photonic Crystals Based on Active Metasurface,” Laser & Photonics Reviews 19 (2025): 2401345, https://doi.org/10.1002/lpor.202401345.

38. H. Moussa, G. Xu, S. Yin, E. Galiffi, Y. Ra'di, and A. Alù, "Observation of Temporal Reflection and Broadband Frequency Translation at Photonic Time Interfaces," Nature Physics 19 (2023): 863–868, https://doi.org/10.1038/s41567-023-01975-y.

39. X. Wang, M. S. Mirmoosa, V. S. Asadchy, C. Rockstuhl, S. Fan, and S. A. Tretyakov, “Metasurface-Based Realization of Photonic Time Crystals,” Science Advances 9 (2023): adg7541, https://doi.org/10.1126/sciadv.adg7541.

## Supporting Information

Additional supporting information can be found online in the Supporting Information section.
Supporting File: lpor71179-sup-0001-SuppMat.docx.
