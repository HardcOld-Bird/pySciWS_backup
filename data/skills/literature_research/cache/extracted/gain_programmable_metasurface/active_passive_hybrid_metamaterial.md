# Transmission Loss of a Labyrinthine Acoustic Metamaterial Augmented with Multichannel Feedforward Active Noise Control

Gregory M. Hernandez $^{*}$ and Jordan Cheer $^{\dagger}$

Institute of Sound and Vibration Research, University of Southampton (UK) $^{\ddagger}$

Gianluca Memoli $^{§}$

AURORA Project, School of Engineering and Informatics, University of Sussex (UK) $^{¶}$ (Dated: September 30, 2025)

Acoustic metamaterials and active noise control are two advanced noise control treatments that can typically offer performance that exceeds that of conventional passive noise control treatments. Acoustic metamaterials utilize sub-wavelength structures to realize sound field control, whilst active noise control treatments achieve control via the introduction of additional sources driven to generate a secondary sound field that interferes in a controlled way with the original, primary sound field. This paper presents an investigation into combining these two noise control techniques, to achieve enhanced noise control over a spatial region using a “hybrid” device. In particular, conventional feedforward active noise control is combined with a labyrinthine metasurface and the increase in performance offered by the hybrid solution is demonstrated.

## I. INTRODUCTION

Industrial noise has been a part of human society since the industrial revolution, and it is now well accepted that this pollutant has a significant cost to society, due to its negative health effects $[1]$ . For many centuries, however, acoustical engineers had two principle methods of managing noise: mass-based solutions and acoustic absorbers $[2]$ . Unless resonators are used, however, these two methods require a large spatial footprint or mass to achieve control at the lower frequencies typical of industrial settings and ventilation systems.

Active Noise Control (ANC), whose first commercial applications emerged in the 1980s [3], offers a solution to overcome the low frequency limitations of conventional passive noise control treatments in a variety of applications including headphones [3], road vehicles [4] or aircraft [5]. There are a variety of physical aspects that limit the potential application of active noise control systems, but when the objective is to achieve sound field control over an extended spatial domain the number of secondary sources, or control loudspeakers, rapidly increases. This increases both the complexity and the cost of utilizing an active system and, therefore, their use is often limited to high performance, critical applications.

An alternative solution to overcoming the limitations of conventional passive noise control treatments is offered by acoustic metamaterials $[6–8]$ . Typically subwavelength, these structures have an artificial bulk modulus and mass density that can facilitate sound field manipulation enabling applications such as beamforming $[9]$ or acoustic cloaking [10]. Various acoustic metamaterial designs have been proposed that are able to achieve effective sound attenuation at low frequencies, for example in [7] an array of resonators is used to form a metasurface that achieves 99% energy absorption at 511 Hz with a surface thickness of about $\lambda/20$ th, where $\lambda$ is the acoustic wavelength; in [11] space coiling is used to realize perfect absorption at 125 Hz with a thickness of about $\lambda/223$ ; and in [12] a membrane-type acoustic metamaterial is shown to achieve perfect absorption at 152 Hz with a thickness of around $\lambda/133$ . These various metamaterial devices, however, have two key limitations: they are effective only over a limited bandwidth and can be challenging to mass-manufacture. Overcoming these two limitations has led to significant research effort in both the manufacture and application of metamaterials [13]. However, real-world application has remained somewhat limited, with only a few examples of metamaterials applied in practice [14–16]. Despite extensive development, and emerging commercial exploitation [16]), the utilization of passive metamaterials for dynamic applications remains challenging.

To address the challenge posed by dynamic applications, various researchers have proposed active acoustic metamaterial solutions for wave control $[17–20]$ , however, these are typically rather complex to implement and the cost would be prohibitive for many applications. To reach a balance between performance under dynamic applications and complexity, previous research has also explored the integration of active noise control techniques with a passive Helmholtz resonator based metamaterial $[21]$ . This hybrid passive-active solution demonstrated a 10 dB enhancement in transmission loss compared to either the passive or active systems operating in isolation, and achieving more than a factor of 8 increase in the bandwidth compared to the passive resonator-based metamaterial.

In this work, an alternative hybrid active-passive acoustic metamaterial is investigated, which is realized by combining a feedforward active noise control system with a static acoustic metasurface realized using the labyrinthine unit cells described in $[9]$ . The objective here is to enhance the downstream attenuation performance of the metasurface by using active noise control, and thus demonstrate the benefits of a hybrid approach, providing useful insights for future development of similar hybrid metamaterials. Notably, the hybrid system is evaluated in a 2D waveguide, which goes beyond the 1D waveguide investigations used in much of the literature $[18, 21]$ . The paper is structured as follows: Section II describes the physical system, including the labyrinthine metasurface and the loudspeakers and microphones utilized to implement the active control system; Section III presents the passive performance of the metasurface; Section IV presents the performance of both the active noise control system alone and when combined in various ways with the passive metasurface to realize different hybrid systems; finally, Section V presents conclusions.

![](images/b99287f4021eb61511773402e39aa487cff7c46c6a0bb6f60ca752a1f03c35e1.jpg)
FIG. 1: (a) The 2D rectangular waveguide is 100 cm x 50 cm x 4.5 cm. There are 40 microphones positioned in the upstream and downstream sections (20 each side). The signal generator sends Gaussian white noise to an amplifier that excites a loudspeaker at discrete locations. The data acquisition unit sends the sensor signals to the computer where they are stored, and processed in a computer with MATLAB. MS indicates the metamaterial position. (b) The top-down view of the waveguide.

## II. EXPERIMENTAL CONSTRUCTION

The experimental apparatus is depicted in Fig. 1a and 1b. The 2D waveguide - 100 cm long, 50 cm wide, and 4.5 cm tall - is constructed of 5 mm thick rigid engineering plastic (polyamide 66). The lateral walls of the waveguide are screwed to both the top and bottom plates and further secured with epoxy to the bottom plate.

## A. Labyrinthine Metasurface

The considered metasurface, noted by MS in Figure 1, has been realized using the concept of “metamaterial bricks” introduced by Memoli et al. [9]. This work demonstrated that, once the main frequency of operation has been selected, most narrow-band metasurfaces can be built by reconfiguring 16 labyrinthine pre-defined metamaterial bricks, each encoding a specific phase shift between 0 and $15/8\pi$ . In [9], the metamaterial bricks have a length of $\lambda$ in the direction of propagation, which will be referred to as the “thickness” of the device, and are $\lambda/2$ wide in the other two dimensions. However, these cells were considered too large for the current setup. At 2400 Hz the unit cells would have been 14.3 cm thick and 7.14 cm in the lateral dimensions. Therefore, the metamaterial bricks presented in [22] have been used here, which have a thickness of $\lambda/3$ , a lateral dimension of $\lambda/6$ , and encode a phase delay ranging from 0 to $15/8\pi$ .

![](images/b052dc8169959d1a367ae771438e9cb6b52300afaac2a9fb1495a2968324123d.jpg)
FIG. 2: (a) The 2D top-down perspective of the metamaterial unit cells situated in the waveguide. (b) The 3D-printed metamaterial constructed of PLA plastic. There are a total of 18 unit cells: 9 are open units, the other 9 are labyrinth unit cells.

The metasurface utilized in this experiment contains 18 unit cells, constructed of PLA plastic, giving a total width of 50 cm and a thickness of $\approx 4.76$ cm. The 2D blueprints in [22] were elongated to fit the waveguide, so that the final height of the metasurface was 4.5 cm. The thickness of the metasurface was 5.7 cm (40% or 2 $\lambda$ /5 in air), with a physical wall construction thickness of 2 mm to ensure an effective connection with the loudspeaker array structure (see Fig. 3b-e) and to ensure that the structure is sufficiently rigid to avoid structural effects.

The metasurface has been constructed from two distinct metamaterial bricks, which are arranged in an alternating pattern, as shown in Figure 2a and 2b. This two unit cell architecture (analogous to a dipole pair) is the basis of the metasurface design used here. At the selected frequency of 2400 Hz, one unit cell is an open channel (phase shift: 0), through which the incident wave travels without any change in phase. The other is a meandering, or labyrinthine unit cell, designed to create a phase shift of $\pi$ , so that the wave that it radiates destructively interferes with the output waveform traveling through the open unit cell at the selected frequency. For the purposes of this study, this metasurface can be considered as a passive noise-canceling device operating at a single frequency.

## B. Active Noise Control System

The active noise control system consists of two principle components – the loudspeakers used to implement control and the microphones used to monitor the sound field in the waveguide. The control system, as described in Appendix B, uses the downstream microphone signals to determine the optimal loudspeaker drive signals in order to control the downstream sound field and maximize transmission loss of the system. The following subsections describe the loudspeaker and microphone arrangements.

## 1. Loudspeakers

The metasurface described in the previous section has been augmented using an array of 9 miniature loudspeakers (see Fig. 3b) each rated as a 32 ohm, 500 mW driver with a 23 mm diameter (model: MCABS-231-RC by multicompPRO). These transducers were chosen to meet the height limitation of the waveguide and lateral opening of the metasurface unit cells, and to have a frequency response covering the bandwidth of the designed metasurface. Each transducer was housed in a 3D-printed enclosure, designed to match the channel dimensions and wall thickness of the metasurface (shown in red, in Fig. 3b). The loudspeaker units were designed with an alternating arrangement, as shown in Fig. 3b, to match the unit cell arrangement of the metasurface. Specifically, three loudspeaker configurations have been investigated: firstly, with the loudspeakers upstream, or posterior, of the open unit cells (Fig. 3c); secondly, with the loudspeakers downstream, or anterior, of the open unit cells (Fig. 3d); and finally, with the loudspeakers upstream, or posterior, of the labyrinthine unit cells (Fig. 3e). The loudspeakers could not be placed in front of the labyrinth cells of the metasurface due to their narrow geometry.

## 2. Microphones

The minimum number of microphones to be used for active noise control has been determined using the modal decomposition approach by Zhang et al. [23]. As described in Appendix A, this method allows the propagating modes within a 2D-waveguide to be detected and is therefore much simpler than the raster scanning method employed in other works [7, 18]. White noise is utilized to excite the waveguide, thus allowing for a large bandwidth to be analyzed. The method proposed in [23] also allows the transmission and reflection coefficients for all modes within the considered frequency range of the experiment to be obtained.

As described in Appendix A, a total of 40 sensors are needed to ensure that the modal matrix is overdetermined. Therefore, 20 electret microphones (PCB Piezotronics, Model 130F20) were placed upstream of the metamaterial, and 20 located downstream. The general location of these sensors is depicted in Fig. 1a. The microphones are connected to a National Instruments CompactDAQ cDAQ-9178, depicted above the waveguide in Fig. 1a, which sends the time domain microphone signals to MATLAB. The source located at the end of the waveguide, also shown in Fig. 1 outputs Gaussian white noise with a standard deviation of 1 Volt, driven via a Wondom Class D Audio Amplifier (Sure Electronics AA-AB32155). Note that in Fig. 1a,b MS indicates the metasurface location (or any other device being measured).

![](images/6857a29f3493ce5b00f88525550240c3d7d88a332c36c173d29bd7ebfc131d21.jpg)
FIG. 3: (a) The general top-down view of the waveguide depicting the source location and positions of the metasurface and loudspeaker units for active control. (b) The loudspeaker array utilized in the active control implementation. (c) The hybrid metasurface with the loudspeakers placed in the posterior position of the open unit cells. (d) The hybrid metasurface with the loudspeakers placed in the anterior position of the open unit cells. (e) The hybrid metasurface with the loudspeakers placed in the anterior position of the labyrinth unit cells.

## III. PASSIVE PERFORMANCE

This section presents measurements of the passive attenuation provided by both the undriven or passive loudspeaker array within the waveguide and the hybrid labyrinthine metasurfaces. The measured passive performance is evaluated in terms of the transmission of the $0^{th}$ propagating mode. Fig. 4 shows the transmission coefficient for the empty waveguide, the passive loudspeaker array and the passive metasurface with the loudspeakers located in the possible configurations described in Section II B1. It is important to first note from the results presented in Fig. 4 that a numerical error arises at 1715 Hz; this is due to the modal decomposition method in obtaining the plane wave propagation (see Appendix A), and can be attributed to the conditioning of the modal matrix derived by Zhang et al. [23]. Although the conditioning of this problem may be improved via regularization techniques, since the numerical error occurs at a frequency that is below the range of interest for the designed meta-surface, it will not be considered further.

The results presented in Fig. 4 reveal that the transmission of the empty waveguide is not exactly unitary, which can be attributed to the thermo-viscous losses and the damping within the waveguide itself. It can also be seen from Fig. 4 that the insertion of the passive loudspeaker array introduces a small decrease in the transmission. Specifically, at the frequencies between 1 kHz and 2 kHz, the transmission of the $0^{th}$ propagating mode is also 90% with the loudspeaker array, and above 2 kHz the transmission decreases to 85% as the frequency approaches 3 kHz. This can be related to the absorption being provided by the passive components of the loudspeaker array (i.e. the materials of the loudspeaker and the plastic housing) and, at higher frequencies, to reflection from the loudspeakers as the wavelength decreases.

Fig. 4 also shows the passive transmission performance of the metasurface with the loudspeaker array in the three considered configurations. Within the 1000-2000 Hz bandwidth, the introduction of the metasurface results in nearly 10% more of the incident wave being reflected and/or absorbed compared to the passive transmission of the loudspeaker array alone. A further dip in the transmission can be observed in Fig. 4 for the three passive metasurface-loudspeaker configurations between around 2000 and 2250 Hz, which defines the operating bandwidth of the passive metasurface. For the two configurations with the loudspeakers in the posterior position, the bandwidth of the metasurface is relatively consistent. However, there is a small shift in the resonant peak – from 2150 Hz to 2180 Hz – when comparing the two posterior responses (blue and red) to the anterior response (pink). Additionally, the level and bandwidth of transmission loss with loudspeakers in front of the open cells is not as significant as in the two posterior loudspeaker variations. At frequencies above the operational bandwidth of the passive hybrid metasurface, the transmission loss depends on the relative position of the loudspeakers: when the loudspeakers are upstream of the labyrinth, the transmission decreases with frequency, reaching around 70% at 3 kHz; when the loudspeakers are either upstream or downstream of the open unit cells, the transmission stays constant at around 80%. Note that the resonance for either posterior hybrid metasurface configuration will be referenced to 2150 Hz moving forward.

It is worth discussing the difference between the measured (2150 Hz) and designed (2400 Hz) operational frequency of the passive metasurface. This can be related to the additional path length introduced in the experimental realization due to the loudspeaker array, which decreases the frequency at which effective interference between the open and labyrinth unit cells occurs. To support this hypothesis, 2D and 3D simulations were performed (using the Acoustic Module of COMSOL Multiphysics) to explore the effect of the additional thickness introduced by the inclusion of the loudspeaker array. The simulation results, reported in Appendix C, show a similar decrease in the resonance frequency of around 400 Hz when the loudspeaker array is introduced either in front or behind the metasurface. The difference between the simulated and measured shift in frequency can be attributed to the approximations used in the simulations (e.g. the loudspeakers were modeled as hard objects) and to the higher-order physics interaction between the electroacoustic devices and the metasurface.

![](images/c36f51e15df71c7c072f0034f96fab62c91bf442ef41c7f1d15f236e6761dad0.jpg)
FIG. 4: Passive measurements. Transmission of the $0^{th}$ propagating mode within the waveguide. Passive attenuation for an empty waveguide (green), standalone loudspeaker array (black), metasurface and loudspeaker hybrid system: loudspeakers behind open cells (blue), loudspeakers behind labyrinth cells (red), loudspeakers in front of open cells (pink).

## IV. ACTIVE PERFORMANCE

Having demonstrated the passive performance of the hybrid metasurfaces in the previous section, this section presents an investigation into the active performance of these proposed systems. Since there was no significant difference in the transmission of the $0^{th}$ propagating mode with the loudspeaker array positioned either posterior or anterior to the open unit cells of the metasurface, the hybrid performance is explored here only for the two cases where the loudspeakers are posterior to the open or the labyrinth unit cells. Additionally, to provide context to the additional performance offer by the proposed hybrid metasurface, the active performance provided by the loudspeaker array in isolation is also analyzed.

In all three active cases, control is realized using an optimal multichannel feedforward active control system and simulated offline using the responses measured using the experimental system. This approach allows the physical limits on control performance to be investigated, without introducing the complexities of real-time implementation which are well understood $[24]$ . The assumed feedforward control strategy, which is introduced in Appendix B, calculates the signals required to drive the array of loudspeakers to minimize the sum of the squared pressures measured at the array of downstream microphones. The active control algorithm also includes a constraint on the control effort, which is defined as the sum of the squared magnitudes of the control signals driving the loudspeakers, as given by equation B3. It is necessary to constrain the control effort in practice to avoid over-driving the loudspeakers, but also to make the controller robust to real-world uncertainties. A control effort constraint is imposed here so that the control signals driving the loudspeakers are realizable in practice, but also to ensure that the required control effort is consistent between the three considered active systems.

Fig. 5a,b presents the transmission and transmission loss respectively of the three passive systems previously presented in Section III along with the three active systems, including the two hybrid metasurfaces, and the standalone active loudspeaker array. Transmission loss is defined as

$$
T L = 2 0 \log_ {1 0} (1 / | T |),\tag{1}
$$

where $|T|$ is the magnitude of the transmission coefficient.

Firstly, the active control algorithm was applied to the standalone loudspeaker array, placed in the posterior position relative to the metasurface (although the passive metasurface itself has been removed). The sold black line in Fig. 5a,b shows the performance of this active system and comparing this to the passive loudspeaker array shows that a significant reduction in the transmission is achieved by the active system, particularly at frequencies above 2000 Hz. This slightly curious result from the perspective of active control, which is generally more effective at lower frequencies, may be related to the fact that only the $0^{th}$ order propagating mode is being considered here. Nevertheless, it is clear from these results that the active system using the loudspeaker array in isolation does exhibit a transmission loss that increases between 1000 Hz and 3000 Hz by around 11 dB, compared to a 1 dB increase for the passive loudspeaker array (dashed black line). It is also worth noting that at the 2150 Hz resonance of the hybrid metasurfaces, the transmission loss for the active loudspeaker array is 4 dB, which is an increase of around 3 dB compared to the passive attenuation provided by the loudspeaker array at this frequency.

Fig. 5 also shows the performance of the two active hybrid metasurfaces, with the loudspeaker posterior to either the open or labyrinth unit cells (solid red and blue lines respectively). In both cases, the hybrid metasurfaces significantly outperform both their respective passive responses (dashed red and blue lines) and the purely active system. An important note is that the two hybrid metasurfaces do not achieve the same active performance, even at frequencies where the passive performance is quite similar, which highlights a difference in the interaction between the active and passive components of the system.

The performance of the two hybrid metasurfaces can be effectively discussed considering three distinct frequency bands: below the passive metasurface interference band (f < 2 kHz); within the passive metasurface interference band (2 kHz f < 2.25 kHz); and above the passive metasurface interference band (f > 2.25 kHz). At frequencies below the passive metasurface interference band, the two hybrid metasurfaces achieve quite consistent active performance, with an average level of around 6 dB. The hybrid systems also provide a significant performance advantage compared to the purely active system over this lower frequency bandwidth, with an increase in the transmission loss of up to 3 dB. At frequencies around the passive metasurface interference band, the hybrid system with the loudspeakers posterior to the open unit cells achieves the highest performance, with a transmission loss of 13 dB; while the hybrid metasurface with the loudspeakers positioned posterior to the labyrinth unit cells achieves a maximum transmission loss in this bandwidth of 9 dB. Finally, at frequencies above the passive metasurface interference band, the hybrid system with the loudspeakers posterior to the open unit cells achieves performance largely consistent with the purely active system, with a maximum transmission loss of 12 dB. In the same frequency band, the transmission loss achieved by the metasurface with the loudspeakers posterior to the labyrinth unit cells increases at a greater rate than the other active systems, providing a transmission loss up to 11 dB greater than the other hybrid metasurface configuration.

The results presented for both hybrid metasurface configurations demonstrate a significant performance advantage compared to traditional active noise control, both in the frequency band for which the passive metasurface is designed and beyond. However, the differences in performance between the two hybrid metasurfaces highlight differences in the coupling between the loudspeakers and the passive metasurface and it is insightful to discuss the potential underlying physical mechanisms.

To investigate the nature of the coupling, in this study we analyzed the data using the techniques of optical/impedance spectroscopy [25]:

1. We assumed that the “active control only” is the background performance i.e. that the hybrid system performs at least as that. We then fitted the background TL with a polynomial, finding that a 2nd degree polynomial was sufficient i.e. $TL = 4e - 6 \cdot f^{2} - 0.0105 \cdot f + 8.71$ where f is the frequency in Hz, as shown in Figure 6a.

2. We subtracted the fitted background from the TL curves with the speakers in the two configurations – see e.g. Figure 6b for the case with the loudspeakers behind the labyrinthine cells.

3. We used the Peak Finder function in OriginPro (OriginLab, version 2025b) to fit the obtained curves with multiple Gaussian peaks. The algorithm found 13 peaks, which were examined for relevance: eliminating those with too much uncertainty, we reduced the number of peaks to 7. The resulting fits can be found in Figures 6c,d and the coefficients in I.

As shown in Table I, the spectra with the loudspeakers in the two cases (i.e. behind labyrinthine & behind open cells) show two components:

1. a single peak at $2125 \pm 20$ Hz, with an average FWMH (Full Width at Maximum Height) of 70 Hz, which we attributed to the metasurface.

a)
![](images/8b93a8878fc3fe52cba6c28729f02584e664bfd7815e51b577824a6bcd6ad1ce.jpg)

b)
![](images/aefe7e7b44ef0087c02cacd4eff039492944012fa373285ab00874ce5909d000.jpg)
FIG. 5: Transmission (a) and transmission loss (b) of the $0^{th}$ propagating mode within the waveguide. Three passive cases are presented without active control—loudspeaker array (dashed black), hybrid metasurface with loudspeakers: behind labyrinth cells (dashed red), behind open unit cells (dashed blue). The analogous offline active control applications are also given—loudspeaker array (hard black), hybrid metasurface with loudspeakers: behind labyrinth cells (hard red), behind open unit cells (hard blue). Overlaid in (b) are the 2D geometric views of the two hybrid systems analyzed for active control.

TABLE I: Coefficients of the multi-peak fits in Figure 6. Uncertainty (not reported) were approximately $10\%$ of the values in the table.

<table><tr><td></td><td>Peak 0</td><td>Peak 1</td><td>Peak 2</td><td>Peak 3</td><td>Peak 4</td><td>Peak 5</td><td>Peak 6</td><td>Peak 7</td><td>Peak 8</td></tr><tr><td>Amplitude / dB</td><td>7</td><td>3</td><td>3</td><td>4</td><td>8</td><td>4</td><td>2</td><td>1</td><td>n.a.</td></tr><tr><td>Frequency / Hz</td><td>940</td><td>1242</td><td>1556</td><td>1926</td><td>2139</td><td>2289</td><td>2558</td><td>2778</td><td>n.a.</td></tr><tr><td>FWMH / Hz</td><td>56</td><td>157</td><td>121</td><td>200</td><td>82</td><td>90</td><td>169</td><td>63</td><td>n.a.</td></tr></table>

(a) Fitting values for the “open cells” case.

<table><tr><td></td><td>Peak 0</td><td>Peak 1</td><td>Peak 2</td><td>Peak 3</td><td>Peak 4</td><td>Peak 5</td><td>Peak 6</td><td>Peak 7</td><td>Peak 8</td></tr><tr><td>Amplitude / dB</td><td>7</td><td>2</td><td>3</td><td>4</td><td>2</td><td>1</td><td>5</td><td>4</td><td>6</td></tr><tr><td>Frequency / Hz</td><td>940</td><td>1258</td><td>1578</td><td>1951</td><td>2114</td><td>2335</td><td>2701</td><td>2959</td><td>3164</td></tr><tr><td>FWMH / Hz</td><td>51</td><td>146</td><td>117</td><td>185</td><td>68</td><td>63</td><td>233</td><td>176</td><td>314</td></tr></table>

(b) Fitting values for the “labyrinthine cells” case.

![](images/c8c41289ee00aac85fafafdd99a15b9c922b9df069f44174619455690acc348d.jpg)
(a) Background data, fitted with a 2nd degree polynomial.

![](images/ea13d7d5931e240203e7e68530e9b36f9f2f38aface8955544a0262ea66470f4.jpg)
(b) Signal data: loudspeakers behind the labyrinthine cells.

![](images/d28163531f7e8f73115124bd5014dfec63255f6cff91bd658959987af92331a4.jpg)
(c) Multi-peak fit, open cells.

![](images/249556e3c9596b66efb6abfb77ea3c67d118097d3b29e0edae58994e1977cef3.jpg)
(d) Multi-peak fit, labyrinthine cells.
FIG. 6: A schematic of the spectroscopy analysis: (a) background fit; (b) signal; (c) multi-peak fit in the case with loudspeakers behind the open cells; (d) multi-peak fit in the case with loudspeakers behind the labyrinthine cells.

2. a periodic pattern, with peaks separated by $285 \pm 60$ Hz, which we attributed to resonances in the system (e.g. to the dimensions of the waveguide).

The efficiency of absorbing energy through the peak at 2125 Hz depends on the configuration: transmission loss at this frequency is more effective when the loudspeakers are behind the open cells. When the loudspeakers go behind the labyrinthine cells, in fact, the energy transfer changes: the energy initially transferred to the single peak now gets distributed to peaks at higher frequencies, so much that an additional peak of the periodic system (i.e. at 3164 Hz) was needed to complete the fit. The description of our hybrid metasurface in terms of the linear combination of two systems, however, has some limitations:

1. If the periodic pattern is a feature of the geometry, it should also appear in the “active control only” case, but it was not possible to isolate it there. The energy transfer phenomenon is only visible when the metasurface is present, and this observation justifies the use of the “active control only” data as “background”.

2. In the “open cells” case, the difference between the signal and the background is negative above 2860 Hz. Once again, the periodic pattern should appear, if present.

These two observations suggest a different physics. The coupling between a single-line/narrow-band resonator (i.e. the metasurface) and a system capable of finding solutions at multiple frequencies, almost continuously (i.e. the active noise control system) has been described by Fano in 1961 [26], in the case of two electronic configurations of a He atom. As shown in more recent studies [27], however, this type of interaction can lead to two regimes: one of weak coupling, where the response appears like a single peak over an existing trend, and one of strong coupling, where the single narrowband peak splits and two peaks appear, separated in frequency and introducing a change in the trend. Considering the hybrid metasurface with the loudspeakers posterior to the open cells, the results are characteristic of a weak-coupling regime, with the trend outside the narrowband metasurface region ( $\sim$ 2150 Hz) being similar to the one obtained with the purely active system, only with an enhancement of its periodic response. Instead, in the case where the loudspeakers are posterior to the labyrinthine unit cells, the performance appears to be the one of a strong-coupling regime, with the presence of two peaks. One peak has been shifted to a frequency lower than 2150 Hz and the other above 3000 Hz. In this explanation, the periodic pattern would decrease in intensity as the frequency increases. A deeper understanding of how to pilot these couplings (which in atomic physics are called “polaritons” [27]) will be the subject of future studies.

## V. CONCLUSIONS

In this work, we presented an investigation into the attenuation of wave transmission using “hybrid” metasurfaces, realized by combining a multichannel feedforward active control system with a passive labyrinthine acoustic metasurface. Modal decomposition was utilized to analyze the transmission of the $0^{th}$ mode propagating in a rectangular waveguide, for the cases of the passive hybrid metasurfaces, traditional active noise control and two hybrid metasurface configurations. Both active hybrid metasurfaces outperformed their respective passive responses and the purely active system, demonstrating the advantages offered by hybrid systems for broadband noise management applications. The presented results show evidence that it is possible to pilot the physical coupling between active control and metamaterials simply by mechanically positioning the loudspeakers in different positions relative to a metasurface. Like in passive metamaterials, geometrical and design choices may result in performance changes, opening a new design space for active metamaterials.

This research has shown that the hybrid metasurfaces offer a synergistic performance advantage, with the performance exceeding that offered by the simple linear summation of the passive metasurface and the traditional active control system over different frequency ranges depending on the selected configuration. As well as investigating real-time implementation of the proposed hybrid metasurfaces, it will also be important to further optimize the metasurface to maximise the transmission loss. For example, exploring the potential performance when loudspeakers are located behind both the labyrinth and open unit cells may combine the synergistic performance advantages offered by the two hybrid configurations investigated in this work. Additionally, further work is required to investigate the coupling between the active control system and the metasurface, to further explore the concept of “polaritons” within this acoustic metasurface context, potentially allowing greater leverage of the strong attenuation outside the resonant bandwidth of the active metasurface.

## VI. ACKNOWLEDGEMENTS

Gregory Hernandez would like to thank the US-UK Fulbright Commission for providing him the opportunity to study abroad and propose this research at the ISVR. Greg would also like to thank Ze Zhang for their help in assisting with the modal decomposition analysis. Gianluca Memoli acknowledges funding through his collaboration with Metasonixx Ltd. Jordan Cheer was partially supported by the Department of Science, Innovation and Technology (DSIT) Royal Academy of Engineering under the Research Chairs and Senior Research Fellowships programme.

## Appendix A: Modal Decomposition

This first appendix reviews the theoretical analysis necessary to obtain the modal information of the 2D-waveguide under consideration in this paper. This modal decomposition follows the derivations provided by Zhang et al. [23].

The frequency range of interest in this paper is from 1 kHz to 3 kHz. Based on the dimensions of the waveguide (W: 500 mm, L: 1000 mm, H: 45 mm), there are 8 cut-on modes within the waveguide along the x-dimension as depicted in Fig. A1. The first cut-on mode in the y-direction is at 3.81 kHz (dimension out of the page in Fig. A1). The modes can propagate along the z-axis and the modal matrix derived by Zhang et al. accounts for the surface area impedance mismatch at both ends of the waveguide [23]. Note that the modal matrix contains the pressure pattern of the wave along the cross-section of the waveguide and the propagating wave information based on the wavenumber [23]. The metasurface shown in Fig. A1 splits the waveguide into two sections—an upstream and downstream portion. The pressure amplitudes and direction of the propagating modes are depicted by the red arrows in Fig. A1 where $P_{u}^{\pm}$ and $P_{d}^{\pm}$ are the pressure amplitudes of each mode—± indicates positive or negative direction given by the coordinate system in Fig. A1, and the subscripts u, d refer to either the upstream or downstream section. Equation A1 relates the pressure amplitudes of each upstream and downstream mode to one another through the scattering matrix S

![](images/4ca4360bc5f274bae6a02bb5cc004ffaf50717d2889866b35a1c4b25ba460e1f.jpg)
FIG. A1: A schematic of the scattering matrix variables overlaid on a real image of the waveguide used in the experiment (top-down view). In the center is the metasurface.

$$
\left[ \frac {P _ {d} ^ {+}}{P _ {u} ^ {-}} \right] = \left[ \frac {\underline {{T ^ {+}}}}{\underline {{R ^ {+}}}} \frac {\underline {{R ^ {-}}}}{\underline {{T ^ {-}}}} \right] \left[ \frac {P _ {u} ^ {+}}{P _ {d} ^ {-}} \right] = \underline {{S}} \left[ \frac {P _ {u} ^ {+}}{P _ {d} ^ {-}} \right].\tag{A1}
$$

The matrices $T^{\pm}$ and $R^{\pm}$ represent the transmission and reflection coefficients for the different modes with positive and negative z-direction incidences [23]. The modal matrix derived by Zhang et al. requires s = 2N independent measuring positions (s being the number of microphones) [23]. The scattering matrix (S) is of size 2Nx2N where N is the number of cut-on modes being considered [23]. Since there are 8 cut-on modes being considered, there must be at least 18 microphones on one side of the waveguide to capture all pressure amplitudes (this accounts for the plane wave too), and at least 18 independent measurements to obtain the scattering matrix S. This experiment utilized a total of 40 microphones (20 in the upstream or downstream section of the waveguide) and 22 independent measuring positions (split evenly in the upstream and downstream section of the waveguide) to make the modal and scattering matrices overdetermined.

The scattering matrix is further defined as

$$
\begin{array}{r l} \underline {{S}} = & \left[ \begin{array}{c c c} P _ {d, 0, 1} ^ {+} & \dots & P _ {d, 0, 2 N} ^ {+} \\ \vdots & & \vdots \\ P _ {d, N - 1, 1} ^ {+} & \dots & P _ {d, N - 1, 2 N} ^ {+} \\ P _ {u, 0, 1} ^ {-} & \dots & P _ {u, 0, 2 N} ^ {-} \\ \vdots & & \vdots \\ P _ {u, N - 1, 1} ^ {-} & \dots & P _ {u, N - 1, 2 N} ^ {-} \end{array} \right] \\ & \left[ \begin{array}{c c c} P _ {u, 0, 1} ^ {+} & \dots & P _ {u, 0, 2 N} ^ {+} \\ \vdots & & \vdots \\ P _ {u, N - 1, 1} ^ {+} & \dots & P _ {u, N - 1, 2 N} ^ {+} \\ P _ {d, 0, 1} ^ {-} & \dots & P _ {d, 0, 2 N} ^ {-} \\ \vdots & & \vdots \\ P _ {d, N - 1, 1} ^ {-} & \dots & P _ {d, N - 1, 2 N} ^ {-} \end{array} \right] ^ {- 1}, \end{array}\tag{A2}
$$

where in $P_{*,k,l}^{\pm}$ , k represents the kth mode, and l denotes the lth measurement [23]. The elements of equation A2 depend upon the inversion of the modal matrix defined by Zhang et al. [23]. To ensure the accuracy of the scattering matrix the conditioning number of the modal matrix must be small [23]. The modal matrix depends upon microphone position, which were optimized utilizing fmincon in MATLAB. The conditioning number of the modal matrix was reduced from 2.08 to 0.87 which strongly reduced the cut-on mode information as seen with the fifth cut-on mode (1715 Hz) in Figs. 4 and 5.

From equation A2 the absorption, transmission, and reflection of each mode can be given by

$$
\alpha_ {m} = 1 - \left| \underline {{S}} _ {(1: N, m)} \right| ^ {2} - \left| \underline {{S}} _ {((N + 1): 2 N, m)} \right| ^ {2},\tag{A3}
$$

where $\underline{S}_{(k,l)}$ represents the kth row and lth column of the scattering matrix S [23]. The transmission and reflection coefficients to various modes from the mth mode incidence are given as $\underline{S}_{(1:N,m)}$ and $\underline{S}_{((N+1):2N,m)}$ respectively. Making use of the term $\underline{S}_{(1:N,m)}$ from equation A3, the transmission of the $0^{th}$ propagating mode in the positive direction, or plane wave, within the waveguide can be determined. This is the resulting transmission seen in Figs. 4 and 5.

## Appendix B: Multichannel Feedforward Active Control System

This second section introduces the multichannel feedforward active control algorithm that was applied to the loudspeaker-metasurface combination to create an active metasurface. A feedforward system, unlike a feedback one, senses a disturbance upstream from the control system and this disturbance is utilized as a reference signal for control downstream of the system. The downstream portion of the system contains an error sensor where the signal is controlled. The following equation generalizes a multichannel feedforward system

![](images/6c950e8c05d266a028bef3f07fe71aec64639ec68bbcf12f9896c19e45a95243.jpg)
FIG. A2: COMSOL Multiphysics 3D geometry experimental setup with the metasurface placed in the center of the waveguide.

![](images/1db76dcf324b22169e714c58761a72da6350789421fe6d0bd1c44464f8676dbe.jpg)
FIG. A3: The 3D simulation passive transmission results for the empty waveguide (green), metasurface (red) and loudspeaker-metasurface combination with the loudspeakers behind the open unit cells (blue).

$$
\mathbf {e} = \mathbf {d} + \mathbf {G u},\tag{B1}
$$

where e is the vector of error sensors, d is the vector of disturbance signals, G is the matrix of complex plant responses, and u is the vector of control signals applied to the loudspeakers [28]. The plant response, G, represents the transfer function between the error sensors and individual loudspeakers.

To determine optimal control signals, it is necessary to define a cost function to be minimized, which in active noise control is typically provided by the summation of the modulus squared error signals given as

$$
J = \sum_ {l = 1} ^ {L} | e _ {l} | ^ {2} = \mathbf {e} ^ {\mathrm{H}} \mathbf {e},\tag{B2}
$$

where the power H is the Hermitian transpose, and L is the number of error sensors [28]. In most physical systems, the loudspeakers have a power limiting factor which requires a constraint on the control effort to be introduced. The control effort is given as

$$
P = \sum_ {m = 1} ^ {M} | u _ {m} | ^ {2} = \mathbf {u} ^ {\mathrm{H}} \mathbf {u},\tag{B3}
$$

which is related to the electrical power required to drive the loudspeakers, and M is the number of actuators used for active control $[28]$ . The cost function given by Equation B2 is then modified to include a term proportional to the control effort

$$
J = \mathbf {e} ^ {\mathrm{H}} \mathbf {e} + \beta \mathbf {u} ^ {\mathrm{H}} \mathbf {u},\tag{B4}
$$

where $\beta$ is the regularization factor which is a positive real effort-weighting parameter [28]. The regularization factor is utilized to adjust the constraint on the control effort, which can limit the level of control signals to be within the limitations of the selected loudspeakers and can also improve the robustness of the system to real-world uncertainty.

As mentioned, the objective of the active control system is to minimize the cost function defined by equation B4. The multichannel feedforward active control system considered here is overdetermined, since it has more error sensors than control sources and the optimal vector of control signals is thus given as

$$
\mathbf {u} _ {\mathrm{opt}} = - \left[ \mathbf {G} ^ {\mathrm{H}} \mathbf {G} + \beta \mathbf {I} \right] ^ {- 1} \mathbf {G} ^ {\mathrm{H}} \mathbf {d},\tag{B5}
$$

where I is the identity matrix with size 2Mx2M, and the matrix $G^{H}G$ is assumed to be positive definite [28]. Note that the optimized control vector, and derivation of the aforementioned equations, are calculated based on frequency domain information. Equation B5 is applied to equation B1 to evaluate the error signal at the downstream section of the waveguide. Equation B5 is computed for each frequency (1 kHz to 3 kHz in steps of 5 Hz) for a chosen value of beta based on the control effort defined in equation B3. Since there are multiple independent sources based on the modal decomposition approach, equation B5 is applied to each measurement, for all frequencies, with a different regularization factor.

[1] World Health Organization, Burden of disease from environmental noise (WHO, 2011).

[2] F. Fahy, Foundations of engineering acoustics (Elsevier, London, 2001) Chap. 7, 4th ed.

[3] A. G. Bose and J. Carter, Headphoning, patent no. 4,455,675 (1984).

[4] T. J. Sutton, S. J. Elliott, M. McDonald, and T. J. Saun-

To evaluate a reasonable regularization factor, the control effort of each individual loudspeaker, for a given measurement, was compared to the rms voltage of the signal driving the loudspeakers. The excitation signal was a Gaussian white noise signal with a 1-Volt standard deviation and mean of zero. The control effort for each loudspeaker was compared to a value of 1 and a regularization factor was chosen so that no single loudspeaker out of the entire array would go above this defined threshold.

## Appendix C: 3D Simulation

Fig. A2 presents the geometry of the 3D simulation of the experimental setup including the metasurface located in the center of the waveguide using COMSOL Multiphysics. The maximum element size for the mesh of the domain was chosen to be $\frac{1}{10}$ of the wavelength for the highest frequency (3 kHz) under consideration. The waveguide had a perfectly matched layer (PML) appended to each end that represented the open end termination in the real experiment. All other boundaries were chosen as sound hard boundaries. The excitation source was chosen to be a point monopole source. Simulations were run in steps of 5 Hz from 1000 Hz to 3000 Hz. Pressure responses were recorded using domain point probes and were located based on the location of the microphones determined from Appendix A. Thermoviscous effects were not considered for this experiment.

The 3D simulation transmission results for the metasurface (red) and the metasurface-loudspeaker variation with loudspeakers behind the open unit cells (blue) are presented in Fig. A3. The most important part of this plot is the noticeable shift in the resonant frequency of either device. The difference in frequency is 400 Hz between the metasurface-loudspeaker combination and just metasurface. This is expected since the effective path length of the wave is increased by the addition of the loudspeaker unit cells. Secondly, the transmission loss capabilities increased by 40% between the two metasurface cases with the passive metasurface-loudspeaker response outperforming the standalone metasurface transmission results.

It is also worth noting that the $5^{th}$ cut-on mode and other higher-order mode contributions are seen in both transmission results of Fig. A3. As mentioned in Section III the possible cause for the shift in frequency is due to the passive electroacoustic coupling of the loudspeaker and unit cells of the metasurface.

ders, Active control of road noise inside vehicles, Institute of Noise Control Engineering (1994).

[5] S. J. Elliot, P. A. Nelson, I. M. Stothers, and C. C. Boucher, In-flight experiments on the active control of propeller-induced cabin noise, Journal of Sound and Vibration 140, 219 (1990).

[6] S. A. Cummer, J. Christensen, and A. Alu, Control-

ling sound with acoustic metamaterials, Nature Reviews (2016).

[7] J. Li, W. Wang, Y. Xie, B.-I. Popa, and S. A. Cummer, A sound absorbing metasurface with coupled resonators, Applied Physics Letters 109 (2016).

[8] J. Li, A. Song, and S. A. Cummer, Bianisotropic acoustic metasurface for surface-wave-enhanced wavefront transformation, Physics Review Applied 14 (2020).

[9] G. Memoli, M. Caleap, M. Asakawa, D. R. Sahoo, B. W. Drinkwater, and S. Subramanian, Metamaterial bricks and quantization of meta-surfaces, Nature Communications (2017).

[10] B.-I. Popa and S. A. Cummer, Homogeneous and compact acoustic ground cloaks, Physical Review B 83 (2011).

[11] Y. Li and B. M. Assouar, Acoustic metasurface-based perfect absorber with deep subwavelength thickness, Applied Physics Letters 108 (2016).

[12] G. Ma, M. Yang, S. Xiao, Z. Yang, and P. Sheng, Acoustic metasurface with hybrid resonances, Nature materials 13, 873 (2014).

[13] G. Liao, C. Luan, Z. Wang, J. Liu, X. Yao, and J. Fu, Acoustic metamaterials: A review of theories, structures, fabrication approaches, and applications, Advanced Materials Technologies 6, 2000787 (2021).

[14] L. Sangiuliano, B. Reff, J. Palandri, F. Wolf-Monheim, B. Pluymers, E. Deckers, W. Desmet, and C. Claeys, Low frequency tyre noise mitigation in a vehicle using metal 3d printed resonant metamaterials, Mechanical Systems and Signal Processing 179, 109335 (2022).

[15] F. Pires, R. F. Boukadia, M. Wandel, C. Thomas, E. Deckers, W. Desmet, and C. Claeys, Novel resonator concept for improved performance of locally resonant based metamaterials, Thin-Walled Structures 209, 112866 (2025).

[16] G. Memoli, L. Chisari, L. Bonoldi, and A. A. E. Ouahabi, Bringing acoustic metamaterials to hospital noise,

Acoustic Bulletin 47, 42 (2021).

[17] B.-I. Popa, L. Zigoneanu, and S. A. Cummer, Tunable active acoustic metamaterials, Physical Review B 88 (2013).

[18] B.-I. Popa, Y. Zhai, and H.-S. Kwon, Broadband sound barriers with bianisotropic metasurfaces, Nature Communications (2018).

[19] J. Tan, J. Cheer, and S. Daley, Realisation of nonreciprocal transmission and absorption using wave-based active noise control, JASA Express Letters 2 (2022).

[20] J. Tan, J. Cheer, and C. House, Realisation of broadband two-dimensional nonreciprocal acoustics using an active acoustic metasurface, The Journal of the Acoustical Society of America 156, 1231 (2024).

[21] J. Cheer, S. Daley, and C. McCormick, Feedforward control of sound transmission using an active acoustic metamaterial, Smart Materials and Structures (2017).

[22] G. Memoli, L. Chisari, J. P. Eccles, M. Caleap, B. W. Drinkwater, and S. Subramanian, Vari-sound: A varifocal lens for sound, in Proceedings of the 2019 CHI Conference on Human Factors in Computing Systems, CHI '19 (Association for Computing Machinery, New York, NY, USA, 2019) p. 1–14.

[23] Z. Zhang, H. Denayer, C. Claeys, W. Desmet, and E. Deckers, Angle-dependent reflection, transmission and absorption coefficients measurements using a 2d waveguide, Applied Acoustics 177 (2021).

[24] S. J. Elliott, Signal processing for active control (Academic, 2001) p. 511.

[25] E. Barsoukov and J. Ross Macdonald, Impedance Spectroscopy: Theory, Experiment, and Applications (Wiley, 2005).

[26] U. Fano, Effects of configuration interaction on intensities and phase shifts, Phys. Rev. 124, 1866 (1961).

[27] M. A. Zeb, Fano resonance in the strong-coupling regime, Phys. Rev. B 106, 155134 (2022).

[28] S. Elliott, Signal processing for active control (Academic Press, 2000) Chap. 4, 1st ed.
