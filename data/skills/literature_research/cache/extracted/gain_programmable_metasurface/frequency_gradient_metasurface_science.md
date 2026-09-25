METASURFACES

# Spatiotemporal light control with frequency-gradient metasurfaces

Amr M. Shaltout $^{1}$ , Konstantinos G. Lagoudakis $^{2,3}$ , Jorik van de Groep $^{1}$ , Soo Jin Kim $^{1,4}$ , Jelena Vučković $^{2}$ , Vladimir M. Shalaev $^{5}$ , Mark L. Brongersma $^{1*}$

The capability of on-chip wavefront modulation has the potential to revolutionize many optical device technologies. However, the realization of power-efficient phase-gradient metasurfaces that offer full-phase modulation (0 to $2\pi$ ) and high operation speeds remains elusive. We present an approach to continuously steer light that is based on creating a virtual frequency-gradient metasurface by combining a passive metasurface with an advanced frequency-comb source. Spatiotemporal redirection of light naturally occurs as optical phase-fronts reorient at a speed controlled by the frequency gradient across the virtual metasurface. An experimental realization of laser beam steering with a continuously changing steering angle is demonstrated with a single metasurface over an angle of $25^{\circ}$ in just 8 picoseconds. This work can support integrated-on-chip solutions for spatiotemporal optical control, directly affecting emerging applications such as solid-state light detection and ranging (LIDAR), three-dimensional imaging, and augmented or virtual systems.

Phase-gradient metasurfaces (1) have demonstrated the ability to shape optical wavefronts in light-bending devices (2), meta-lenses (3-5), meta-holograms (6), wave plates (7-9), as well as devices with chiral

(10–12) responses. These functions are achieved by controlling the scattering of light waves from a dense array of nanoscale optical antennas that make up a metasurface. Most metasurfaces to date are passive, and their functions are fixed during their fabrication. Many optical applications require dynamic manipulation of light waves, and an expansion to spatiotemporally varying metasurfaces could open many new opportunities for the metamaterials field (13–21). However, the development of dynamic metasurfaces has been hampered substantially by the challenge of realizing antenna structures capable of providing large changes in the amplitude and phase (0 to $2\pi$ ) of scattered light with an external stimulus. In addition, there is a major technological challenge in addressing individual nanoscale antennas within a large-area metasurface device in terms of the electrical wiring and power consumption. Our approach toward spatiotemporal optical control completely avoids the challenges of actively controlling a metasurface.

We began by expanding on the basic concepts that underlie the operation of existing phase-gradient metasurfaces and illustrating how spatiotemporal optical patterns can be manipulated by controlling the interference of waves in four-dimensional (4D) space-time. We first replaced the typical continuous wave source used to illuminate a metasurface with a frequency-comb source. Such a source features a set of equally spaced and phase-locked frequency lines. We engineered a passive metasurface to interact with such a frequency-comb source to fashion distinct spatial field profiles for each of the spectral lines in the comb. Because the spectral lines are phase-locked, their individual spatial patterns constructively interfere to generate a 4D optical pattern in which the spatial light intensity distribution naturally evolves in time.

Such spatiotemporal metasurfaces add a valuable new dimension to phase-gradient metasurfaces (22, 23), illustrated through comparison of the simplest implementations of these two optical elements. Shown in Fig. 1A is a basic phase-gradient metasurface that imparts a linearly varying phase shift on an incident plane wave to redirect it into a desired angle (2, 24). The corresponding case of a frequency-gradient metasurface constructed from phase-locked optical sources whose frequency is linearly graded across the metasurface is shown in Fig. 1B. Their radiation constructively interferes to naturally direct light toward different angles as time progresses.

The mathematical formulation for the operation of spatiotemporal metasurfaces combines the descriptions for spatial mode transformation by a passive metasurface and temporal pulse shaping known from ultrafast laser physics (25–27). This behavior is conveniently handled in a Fourier synthesis description. Demonstrated in Fig. 1C is how a pulse composed of a set of $2N + 1$ frequency lines $[a(t) = \sum_{n=-N}^{n=N} a_n e^{-i\omega_n t}; \omega_n = \omega_0 + n\Delta\omega]$ interacts with a passive, nanostructured metasurface. The frequencies are centered around an operation frequency $\omega_0$ , and together, they form a pulse $a(t) = A(t)e^{-i\omega_0 t}$ , for which $A(t) = \sum_{n=-N}^{n=N} a_n e^{-in\Delta\omega t}$ is the envelope.

In our analysis, a uniform spatial dependence for the pulse amplitude was assumed. Upon illumination, the passive metasurface maps each spectral component into a set of distinct spatial modes $[a_{n}\rightarrow b_{n}(\mathbf{r})]$ , generating a spatiotemporal optical pattern $[b(\mathbf{r},t)=\sum_{n=-N}^{n=N}b_{n}(\mathbf{r})e^{-i\omega_{n}t}]$ .

To achieve continuous steering of light, we aimed to synthesize a new, virtual frequency-gradient metasurface with a spatial profile $b_{n}(\mathbf{r})$ that mimics an array of optical sources whose frequencies are linearly graded across the surface. The spatiotemporal interference of these sources will naturally produce a scanning beam. This can be seen by analyzing the radiation from an array of line sources that are equally spaced at a distance d along the y axis. The optical source located at the position $\mathbf{r}_{n} = (0, nd)$ in the array radiates at an angular frequency $\omega_{n}$ and a wave number $k_{n} = \omega_{n}/c = k_{o} + n\Delta k$ , and its optical field distribution is described by

$$
b _ {n} (\mathbf {r}) = a _ {n} G (\mathbf {r} - \mathbf {r} _ {n}) e ^ {i k _ {n} (| \mathbf {r} - \mathbf {r} _ {n} |)}
$$

The quantity $a_{n}$ describes the strength of the source, which is proportional to the amplitude of the corresponding frequency line in the incident pulse. The term $G(\mathbf{r}-\mathbf{r}_{n})$ is a slowly varying envelope that governs the angular divergence of the radiation pattern, and $e^{i\tilde{k}n(|\mathbf{r}-\mathbf{r}_{n}|)}$ is a fast varying phase term. With $2N+1$ sources distributed from y = -Nd to y = Nd, the resulting spatiotemporal pattern in the far field takes the following form

$$
\begin{array}{l} b (\mathbf {r}, t) = \sum_ {n = - N} ^ {n = N} b _ {n} (\mathbf {r}) e ^ {- i \omega_ {n} t} \\ = \sum_ {n = - N} ^ {n = N} a _ {n} G (\mathbf {r} - \mathbf {r} _ {n}) e ^ {i (k _ {n} | \mathbf {r} - \mathbf {r} _ {n} | - \omega_ {n} t)} \end{array}\tag{1}
$$

where the spatial coordinate $\mathbf{r} = (x, y) = (r \cos \theta, -r \sin \theta)$ is represented in terms of the distance r from the center of the array $(0, 0)$ and the clockwise angle $\theta$ (Fig. 1C). As the fields produced by each source evolve in time at different frequencies, the resulting interference pattern will also evolve in time. In the far-field $(r \gg Nd)$ , the interference pattern can be approximated as

$$
b (\mathbf {r}, t) \approx A \left(t - \frac {k _ {0} d}{\Delta \omega} \sin \theta - \frac {r}{c}\right) G (\mathbf {r}) e ^ {i (k _ {0} r - \omega_ {0} t)}\tag{2}
$$

(supplementary text S1). The term $A\left(t-\frac{k_{0}d}{\Delta\omega}\sin\theta-\frac{r}{c}\right)$ captures the dynamic beam steering action and shows how the light is directed toward different angles $\theta$ at different times t. The term r/c accounts for the propagation delay between the frequency-gradient metasurface and the point of detection at a distance r. For a cylindrical surface at a fixed distance r, the time-dependent steering angle can be written in the form

$$
\sin \theta = \left(\frac {\Delta \omega}{k _ {0} d}\right) t\tag{3}
$$

after translating the time reference through replacing $t - \frac{r}{c}$ with t. It follows that the speed of

![](images/3adb7bdeab2d1aae2c3d0481a253172b1d759d26e384bd118fc73bdf5f29b1f2.jpg)

C
![](images/e192a4cde97720ae4c530da8ab6eefad2861298a3d177a7b38b7b9df1d8c0ded.jpg)
angular steering is directly proportional to the frequency gradient $(\Delta\omega/d)$ across the surface. Equations 2 and 3 also demonstrate that the angular beam width is related to the pulse width by the proportionality factor $\left(\frac{\Delta\omega}{k_{o}d}\right)$ . For the simple case of cylindrical sources with an omnidirectional response $\left[G(\mathbf{r}) \sim \frac{1}{\sqrt{r}}\right]$ , and equal amplitudes $(a_{n} = 1)$ , we find

$$
b (\mathbf {r}, t) = \sum_ {n = - N} ^ {N} \frac {e ^ {i (k _ {n} | r - r _ {n} | - \omega_ {n} t)}}{\sqrt {| r - r _ {n} |}}\tag{4}
$$

The calculated intensity $I(\mathbf{r}, t) = |b(\mathbf{r}, t)|^{2}$ is demonstrated in Fig. 1D at different times for an array of 41 cylindrical sources ( $2N + 1 = 41; N = 20$ ), a separation between sources d = 360 nm, a center wavelength of $\lambda_{o} = 720$ nm ( $\omega_{o} = 2\pi \times 416.66$ THz), and $\Delta\omega = 2\pi \times 100$ GHz. When the scanning time is fast enough compared with propagation time of light across the observation areas, the light beam appears to be curved. This is analogous to the apparent bending of a water stream emerging from a rapidly rotating water hose. The simulations are discussed in more detail in supplementary text S2, where we also highlight that the scanning rate is proportional to the frequency gradient ( $\Delta\omega/d$ ).

The setup that is used to experimentally demonstrate beam steering is shown schematically in Fig. 2A. A frequency-gradient virtual metasurface was implemented by using a mode-locked laser with a frequency-comb spectrum and a dielectric, phase-gradient metasurface that focuses different spectral components to produce a series of line sources in a designed focal plane. The mode-locked pulsed laser source generates a

![](images/370fc9080ed47db9a48ac785b26a77125a97ba961debe56d2e035832449019e2.jpg)

![](images/ef15e89380266b6f698edd1ca4fafb71b735be6f1139ae431ea64e9aa3ab9854.jpg)
train of pulses that is centered at wavelength $\lambda_{o} = 720$ nm and features a temporal width of $\sim 2.5$ ps. The required spectral separation of the lines in the comb can be accomplished by directing the laser beam onto the metasurface at an off-normal angle of incidence $\theta_{i}$ . In our case, this angle was chosen as $\theta_{i} = 45^{\circ}$ . When the phase shift induced by the metasurface mimicked a diffraction grating ( $\varnothing_{m1} = -k_{o} \sin \theta_{i}x$ ), the different spectral components were decomposed and transmitted into slightly different directions around the sample normal (supplementary text S3). The required focusing could be accomplished with a second metasurface acting as a cylindrical convex lens with a focal distance ( $f_{c}$ ) and a phase profile [ $\varnothing_{m2} = -k_{o}(f_{c} - \sqrt{f_{c}^{2} + x^{2}})$ ]. However, one of the key benefits of using a phase-gradient metasurface is that these two optical functions can be combined in a single metasurface by engineering a phase profile ( $\varnothing_{m} = \varnothing_{m1} + \varnothing_{m2}$ ). Accordingly, the entire beam-steering system is composed of only a frequency-comb source and a single metasurface.

The designed focal length is $f_{c} = 1 \, cm$ , resulting in a frequency gradient $\frac{\Delta\omega}{d} = \frac{\omega_{0}}{f_{c}\sin\theta_{i}} = 2\pi \times 58.9 \, GHz/\mu m$ (supplementary text S3). Decreasing and increasing the designed focal length would result in a larger and smaller, respectively, frequency gradient and beam steering rate. A silicon (Si) metasurface based on the geometric phase (28–30) is used to create the virtual frequency-gradient metasurface (Fig. 2, B and C). The Si nanoantennas in this device were fabricated on a sapphire substrate by means of electron-beam lithography and reactive-ion etching. They are 500 nm high, feature a rectangular cross section of 180 by 110 nm, and are

## Fig. 1. Theoretical concept behind the proposed frequency-gradient metasurfaces.

(A) Schematic of a phase-gradient metasurface capable of redirecting an incident plane wave toward a new, fixed direction by tilting the optical phase-front. (B) Schematic of a frequency-gradient metasurface that naturally keeps reorienting the optical phase-front as time progresses, facilitating continuous dynamic beam steering. (C) Illustration of the light-matter interaction between a frequency-comb source and a passive metasurface, in which the metasurface maps each spectral line of the frequency components into a spatial optical mode $[a_{n} \rightarrow b_{n}(\mathbf{r})]$ , generating a spatiotemporal optical pattern $b(\mathbf{r}, t)$ . (D) Light generated at different time instants by our designed frequency-gradient metasurface, consisting of 41 cylindrical sources $(2N + 1 = 41; N = 20)$ , a separation between sources d = 360 nm, a center wavelength of $\lambda_{o} = 720$ nm ( $\omega_{o} = 2\pi \times 416.66$ THz), and $\Delta\omega = 2\pi \times 100$ GHz.

spaced by 350 nm. The entire metasurface is large (measures 8 by 4 mm) in order to increase the numerical aperture (NA) required for strong focusing and a larger steering angle. More fabrication details on the design and fabrication can be found in the supplementary materials, materials and methods. The geometric phase was obtained by using an array of Si nanoantennas with space-variant orientations (Fig. 2, B and C) (28–30). The geometric phase emerging from a surface patterned with such antennas is given by $\varnothing_{\mathrm{m}} = 2\sigma\psi(x, y)$ , where $\psi(x, y)$ is the in-plane orientation of the nanoantennas and $\sigma$ quantifies the spin angular momentum for the incident light—right ( $\sigma = 1$ ) or left ( $\sigma = -1$ ) circular polarization. This phase can be designed to span the full $2\pi$ range while maintaining a uniform transmission amplitude across the metasurface. These aspects of geometric phase elements can be used to achieve high diffraction efficiency optical elements for incident beams with circularly polarized light (supplementary text S3).

The spatiotemporal interference from the array of virtual sources creates a small-diameter steering beam. We then used a microscope objective to map the time-varying steering angle $\theta$ to an axial dimension $y_{1}$ at the back focal plane ( $f_{\mathrm{M.O.}}$ ) of the microscope objective. Last, we used a 4f imaging system to scale down the spatial extent of the beam $\left(y_{2} = -\frac{f_{2}}{f_{1}} y_{1}\right)$ to fit with the acceptance slit size of the streak camera we used to measure the time dependence of the steering angle $\theta$ . Parameters used in the experiment are $\theta_{i} = 45^{\circ}$ , $f_{\mathrm{c}} = 1 \mathrm{~cm}$ , $f_{\mathrm{M.O.}} = 0.3 \mathrm{~cm}$ , $f_{1} = 30 \mathrm{~cm}$ , and $f_{2} = 10 \mathrm{~cm}$ .

The streak camera measurements show laser scanning over an angle of $25^{\circ}$ within a time

![](images/63b2c43f7fb999e13ce87dc5dc735e1b8e0ac80cbedd870ffd5b0e2810a5417b.jpg)
C

B
![](images/c62a2cd4f81b1bd83380dc9c455fa93bee05006871f7570384b0a8b90d819d49.jpg)

![](images/ad1bcf92d42f11578da5856295d309291e0f2d15b9d11dfa37e980b9856c0e74.jpg)
Fig. 2. Experimental realization of laser beam steering by using a frequency-gradient metasurface. (A) Experimental setup to demonstrate and quantitatively analyze the beam steering. The beam-steering device consists of a pulsed laser source and a passive, phase-gradient Si metasurface that focuses different spectral components to produce an array of focal lines that makes up a virtual frequency-gradient metasurface. Measurements of steering angle $\theta$ versus time is obtained with a streak camera. Mapping of the angle $\theta$ to the streak camera is done by using a microscope objective that maps to axial dimension $y_{1}$ on the back focal plane and a 4f imaging system to scale down the spatial extent $y_{2} = -\frac{f_{2}}{f_{1}} y_{1}$ in order to fit with the slit size of the streak camera. Experimental parameters were $\theta_{i} = 45^{\circ}$ , $f_{c} = 1\mathrm{cm}$ , $f_{\mathrm{M.O.}} = 0.3\mathrm{cm}$ , $f_{1} = 30\mathrm{cm}$ , and $f_{2} = 10\mathrm{cm}$ . (B and C) Scanning electron microscopy of the geometric-phase Si-based metasurface. The Si nano-antennas on top of sapphire substrate have a 180 by $110\mathrm{nm}$ footprint and a height of $500\mathrm{nm}$ .

period of $\sim8$ ps (Fig. 3). The temporal pulse width is $\sim2$ to 3 ps (similar to that of input pulse), and the angular pulse width is $\sim5^{\circ}$ to $6^{\circ}$ , which corresponds to a temporal pulse width multiplied by the angular scan speed, as discussed earlier. Because of the temporal resolution of the streak camera being $\sim2$ ps, we used an interpolation algorithm to determine the temporal location of the pulse peak for each steering angle (Fig. 3, white dots). The white dashed line in Fig. 3 represents a linear fit to this data, which we used to calculate the angular scan speed to be $3.45^{\circ}/ps$ . The yellow dashed line in Fig. 3 represents the time versus angle dependence according to Eq. 3 that shows an approximately linear curve for the time versus angle relation and demonstrates the angular speed $\sim2.4^{\circ}/ps$ . We attribute the small difference of the experimental slope and the theoretical prediction to the variations in the time delays acquired by the pulses passing through different vertical locations $y_{1}$ or $y_{2}$ in Fig. 2A. This can be induced by different factors, including slight variations in optical path lengths, deviations in dimensions of Si antennas across the metasurface, as well as the dispersive properties of the objective lens and other lenses of our 4f imaging system that can cause different group delays across the lens surface (31). These delays can be on the order of picoseconds but do not affect the observed results in a fundamental way.
![](images/61b87695a7b3ae95a2d70076975d37f3007562128cb60f68fc8a0430a0e087cd.jpg)

The properties of the steered beam are controlled by both the metasurface and the frequency-comb source. Although the angular resolution is determined by the bandwidth of the frequency-comb source, the beam steering angle of view (AOV) is controlled by the dielectric metasurface. Large AOV requires high numeric aperture of the cylindrical lensing action in order to provide narrow focal lines at the virtual frequency-gradient source. This is controlled by the footprint size and quality of the nanostructured metasurface. The dielectric metasurface is also responsible for the power efficiency and steering speed (supplementary text S3).

## Fig. 3. Experimental results of the beam-steering action.

This work illustrates how a spatiotemporally varying metasurface can be created by combining a frequency comb source with a passive metasurface. A special case of such a surface, a virtual frequency gradient metasurface, is implemented and demonstrated experimentally. It demonstrates a conceptually new pathway for nonmechanical, ultrafast dynamic beam steering.

Streak camera measurements after mapping each time with its corresponding beam steering angle. It demonstrates the angular redirection over $25^{\circ}$ in 8 ps. The white dots denote the peak locations for the pulse at each steering angle. The white dashed line shows the linear fit used to calculate the angular speed of beam steering. The yellow dashed line shows the theoretical prediction of the time versus angle relation, assuming dispersion-less imaging optics.

Given the notable advances in on-chip frequency-comb sources, it is within reach to integrate the entire system on a single chip (32, 33). The proposed technology can also be expanded to realize other spatiotemporal patterns that, for example, allow for 2D raster scanning and axial scanning of focusing lenses.

## REFERENCES AND NOTES

1. Y. W. Huang et al., Nano Lett. 16, 5319–5325 (2016).

2. J. Park, J. H. Kang, S. J. Kim, X. Liu, M. L. Brongersma, Nano Lett. 17, 407–413 (2017).

3. S. M. Kamali, E. Arbabi, A. Arbabi, Y. Horie, A. Faraon, Laser Photonics Rev. 10, 1002–1008 (2016).

4. A. Komar et al., Appl. Phys. Lett. 110, 071109 (2017).

5. B. Gholipour, J. Zhang, K. F. MacDonald, D. W. Hewak, N. I. Zheludev, Adv. Mater. 25, 3050–3054 (2013).

6. L. H. Nicholls et al., Nat. Photonics 11, 628–633 (2017).

7. K. Lee et al., Nat. Photonics 12, 765–773 (2018).

8. Z. Zhu, P. G. Evans, R. F. Haglund Jr., J. G. Valentine, Nano Lett. 17, 4881–4885 (2017).

9. M. C. Sherrott et al., Nano Lett. 17, 3027–3034 (2017).

10. N. Meinzer, W. L. Barnes, I. R. Hooper, Nat. Photonics 8, 889–898 (2014).

11. N. Yu et al., Science 334, 333–337 (2011).

12. F. Aieta et al., Nano Lett. 12, 4932–4936 (2012).

13. E. Hasman, V. Kleiner, G. Biener, A. Niv, Appl. Phys. Lett. 82, 328–330 (2003).

14. P. Lalanne, P. Chavel, Laser Photonics Rev. 11, 1600295 (2017).

15. S. Larouche, Y. J. Tsai, T. Tyler, N. M. Jokerst, D. R. Smith, Nat. Mater. 11, 450–454 (2012).

16. A. Pors, S. I. Bozhevolnyi, Opt. Express 21, 2942–2952 (2013).

17. N. Yu et al., Nano Lett. 12, 6328–6333 (2012).

18. Z. H. Zhu et al., Opt. Lett. 37, 698–700 (2012).

19. A. Papakostas et al., Phys. Rev. Lett. 90, 107404 (2003).

20. E. Plum et al., Phys. Rev. Lett. 102, 113902 (2009).

21. A. Shaltout, J. Liu, A. Kildishev, V. Shalaev, Optica 2, 860–863 (2015).

22. A. Shaltout, A. Kildishev, V. Shalaev, Opt. Mater. Express 5, 2459–2467 (2015).

23. Y. Hadad, D. L. Sounas, A. Alu, Phys. Rev. B 92, 100304 (2015).

24. X. Ni, N. K. Emani, A. V. Kildishev, A. Boltasseva, V. M. Shalaev, Science 335, 427 (2012).

25. A. M. Weiner, Rev. Sci. Instrum. 71, 1929–1960 (2000).

26. A. M. Weiner, Opt. Commun. 284, 3669–3692 (2011).

27. D. H. Froula et al., Nat. Photonics 12, 262–265 (2018).

28. Z. Bomzon, V. Kleiner, E. Hasman, Opt. Lett. 26, 1424–1426 (2001).

29. Z. Bomzon, G. Biener, V. Kleiner, E. Hasman, Opt. Lett. 27, 1141–1143 (2002).

30. D. Lin, P. Fan, E. Hasman, M. L. Brongersma, Science 345, 298–302 (2014).

31. A. M. Weiner, Ultrafast Optics (Wiley, 2011).

32. P. Del'Haye et al., Nature 450, 1214–1217 (2007).

33. F. Ferdous et al., Nat. Photonics 5, 770–776 (2011).

## ACKNOWLEDGMENTS

The authors thank A. Weiner for insightful discussions. Funding: This work was supported by the U.S. Air Force Office of Scientific Research (AFOSR) (grant FA9550-14-1-0389). In addition, V.M.S. also acknowledges partial support from by AFOSR grant FA9550-18-1-0002, and J.V. from the AFOSR grant

FA9550-17- 1-0002. J.v.d.G. was supported by a Rubicon Fellowship from the Nederlandse organisatie voor Wetenschappelijk Onderzoek (NWO). Author contributions: A.M.S., V.M.S., and M.L.B., developed the concept. K.G.L. and J.V. carried out the streak camera measurements. J.v.d.G. and S.K. fabricated the Si metasurface. All the authors contributed to and approved the manuscript. Competing interests: A.M.S., V.M.S., and M.L.B. are inventors on U.S. patent application WO2019018035-A1 held and submitted by Purdue University that covers the use of frequency gradient metasurfaces for active wavefront manipulation and control. The authors declare no other competing interests. Data and materials availability: All data are available in the main text or in the supplementary materials.

## SUPPLEMENTARY MATERIALS

science.sciencemag.org/content/365/6451/374/suppl/DC1
Materials and Methods
Supplementary Text S1 to S3
Figs. S1 to S4

5 March 2019; accepted 2 July 2019

10.1126/science.aax2357
