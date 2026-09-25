# Supplemental Information

Ruizhi Dong, $^{1,*}$ Yihuan Zhu, $^{1,*}$ Dongxing Mao, $^{1}$ , Xu Wang, $^{1,\dagger}$ and Yong Li, $^{1,\ddagger}$

$^{1}$ Institute of Acoustics, Tongji University, Shanghai 200092, P.R. China

\*R.D. and Y.Z. contributed equally to this work.

$^{\dagger}$ xuwang@tongji.edu.cn

$^{\ddagger}$ yongli@tongji.edu.cn

## Supplementary Section A

## Simulation details

To analyze the eigenmodes and intuitively demonstrate the mode characteristics of the TBIC system, we utilized the commercial software COMSOL Multiphysics to conduct finite-element simulations, leveraging its preset Pressure Acoustics Interfaces. These simulations encompassed both Eigenfrequency and Frequency Domain analyses. Throughout the simulation process, we exclusively concentrated on the material properties of the air within the structure. Therefore, we set the air and environment parameters uniformly in all simulations. The material is set as air, with static density $\rho = 1.21 \, kg/m^{3}$ , sound speed $c_{0} = 343 \, m/s$ , dynamic viscosity $\mu = 1.81 \times 10^{-5} \, N \cdot s/m^{2}$ , and the preset environment temperature $T = 293.15 \, K$ . We performed energy band analysis and supercell mode analysis using the Eigenfrequency study module. For eigenfrequency simulation, the dissipation of the structure is neglected. All boundaries of the model are set as “Sound Hard edge” due to the huge impedance mismatch between air and the walls within the frequency range of interest. It should be noted that in the energy band analysis presented in Fig. 2(a), Floquet periodic edge conditions were applied to the lattice along the periodic direction. We employ “Frequency Domain” simulation to investigate the transmission response of the TBIC system when subjected to plane wave excitation. To achieve this, we introduce a background pressure field representing a plane wave incident on one side of the supercell and utilize perfect matching layers on both sides to absorb reflected waves. The detailed parameters of the simulation are shown in Fig. S1. The lattice constant D is set to 150 mm. The width W of the waveguide is 40 mm, the height C of the cavity is 59.4 mm, and the width U of the cavity is 30 mm. The initial parameter t used to regulate the topological phase transition of the structure is 45 mm. In Fig. 2(a), the parameters t used in the supercell are 20 mm and 70 mm, respectively.

![](images/d25af83f503e606c40200c9aaaa24fbb9dcde6451d749649eecbc3cc53e3399d.jpg)

![](images/b76f145c2b83963ea296d91a1045b2389dda2171a27c0a069815dc30296a40ac.jpg)
FIG. S1. Schematic diagram of structural detailed parameters.

## Topology invariant calculation

The Zak phase corresponding to the band can be calculated by integrating the Berry connection over the entire Brillouin zone as:

$$
\theta^ {Z a k} = \int_ {\mathrm{BZ}} A (k).\tag{S1}
$$

The standard equation for the Berry connection $A(k)$ in a one-dimensional system is given by:

$$
A (k) = <   \psi | i \partial_ {k} | \psi > d k,\tag{S2}
$$

where $\psi$ is the eigenstate of the system and k is the Bloch wavenumber. Zak phase specific to the kind of acoustic system calculations [S1] can be expressed as:

$$
\theta_ {n} ^ {Z a k} = \int_ {- \pi / D} ^ {\pi / D} \mathrm{d} k \left[ i \int_ {\mathrm{lattice}} \frac {1}{2 \rho c _ {0} ^ {2}} d y d x u _ {n, k} ^ {*} (x, y) \partial_ {k} u _ {n, k} (x, y) \right],\tag{S3}
$$

where $u_{n,k}(x,y)$ is the periodic part of the Bloch eigenfunction for states with k in the n-th band. A more intuitive criterion arises from the fact that the acoustic field eigenfrequency analysis mode directly corresponds to the eigenstate of the equivalent H-matrix. When considering a unit cell that exhibits mirror symmetry relative to its central cross-section, the Zak phase is constrained to be either 0 or $\pi$ [S2]. The values of the Zak phase are dictated by the symmetry properties of the band-edge states. Specifically, if both the states at the center and the edge exhibit the same symmetry, be it even or odd, relative to the central cross-section, then the Zak phase of that particular band will be 0. Conversely, if they differ, the Zak phase will be $\pi$ . To explain more insightfully, the corresponding relationships between the energy band, sound field, and Zak phase are presented in Figure S2.

![](images/305867ee382747f763f06212b41a491ca1ade37761ebe39cc5df536e8efc9dac.jpg)

![](images/57abce30a67770e0262c45d17ee6e4b7f35e83194b488218df31e36d6648fc67.jpg)
FIG. S2. Band structures of the shrunk lattice (left panel) and the extended lattice (right panel) are shown, with the Zak phase labeled on each band. Bands corresponding to $\Sigma$ ( $\Pi$ ) modes are colored in blue (red). The acoustic field analysis corresponding to the center and boundaries of each energy band is simultaneously attached to the band diagram.

## Band structure regulation

For this TBIC system, the structural parameters and equivalent Hamilton matrix can be directly related by linear fitting. Take for example the cavity height C studied in the main text. For $\Sigma$ mode, its equivalent Hamilton matrix form is:

$$
H _ {\Sigma} = \left( \begin{array}{c c} f _ {\Sigma} & v _ {\Sigma} + u _ {\Sigma} e ^ {- j k} \\ v _ {\Sigma} + u _ {\Sigma} e ^ {- j k} & f _ {\Sigma} \end{array} \right).\tag{S4}
$$

Under other parameters fixed (as studied in Fig. 3), the cavity height C and the above variables satisfy the following conditions:

$$
\begin{array}{c} {f _ {\Sigma} = 3 8 9 0 - 1 0. 2 9 C,} \\ {v _ {\Sigma} = 1 0 5 0 - 1 2. 1 7 C,} \\ {u _ {\Sigma} = 4 4 0 + 3. 6 5 C.} \end{array}\tag{S5}
$$

For $\Pi$ mode, its equivalent Hamilton matrix form is:

$$
H _ {\Pi} = \left( \begin{array}{c c} f _ {\Pi} & v _ {\Pi} + u _ {\Pi} e ^ {- j k} \\ v _ {\Pi} + u _ {\Pi} e ^ {- j k} & f _ {\Pi} \end{array} \right).\tag{S6}
$$

Under other parameters fixed (as studied in Figure 3), the cavity height C and the above variables satisfy the following conditions:

$$
\begin{array}{c} f _ {\Pi} = 5 8 5 0 - 4 1. 2 0 C, \\ v _ {\Pi} = 3 0 - 0. 3 5 C, \\ u _ {\Pi} = 2 9 0 - 2. 8 9 C. \end{array}\tag{S7}
$$

From the above relationship, it can be seen that there is a good linear relationship between the cavity height $C$ and the variables in the system's equivalent Hamilton matrix. Based on the above relationship, we have drawn the energy band diagram of the structure when $C = 59.4 \mathrm{~mm}$ (the structural parameter used in the main text), and compared it with the simulation results in Fig. S3. The results match well at the center and boundary positions, and the width of the bulk band can be successfully calculated.

![](images/e5c6f1f5285db350219b41750b88b6212755742d4f0517bc898661cf2e09db70.jpg)
FIG. S3. Calculated (lines) and simulated (dots) band structures of the lattice. Bands corresponding to $\Sigma$ (II) modes are colored in blue (red).

## Q factor discussion

The difference between TBIC and eTBIC can also be illustrated from another perspective, which is about the quality factors (Q factors) of these two types of states with different symmetries. On the one hand, symmetric modes ( $\Sigma$ ) inherently have lower Q factors than antisymmetric modes ( $\Pi$ ), as illustrated by the slopes ( $d\omega/dk$ ) of the energy bands of both modes in Fig. 2(a). On the other hand, for a certain symmetry, the edge state naturally has a much higher Q factor than the bulk state. Combining the above two points, we can see that eTBIC and TBIC are clearly different in the Q factors between their respective topological edge states and bulks embedded, as shown in Fig. S4. In particular, note that the edge state of the TBIC has a Q factor of up to $10^{5.7}$ , which means that the relative bandwidth of the transmission spectrum of this edge state is theoretically only $10^{-5.7}$ . Therefore, the transmission peak of TBIC is absolutely unobservable for acoustic systems where intrinsic losses (air thermo-viscous losses) cannot be ignored [as shown in Fig 4(b)]. Note that a high signal - to - noise ratio is crucial for sensing. Since the edge state of TBIC cannot be pinpointed in the transmission spectrum, perturbing it results in almost no observable effect, regardless of either axial or lateral perturbations. A high - transmission peak that pinpoints the eTBIC lays a foundation for our work. The Q values of the $\Pi$ bulks and $\Sigma$ edge states for eTBIC in a finite-size lattice can be relatively close [Fig. S4(a)], this paves a way for the observation of anisotropic sensitivity response of TBICs. In other words, the anisotropy response arises because the bound states comprising the eTBIC do not significantly differ in Q factor from the bulk band at the same frequency, as depicted in Fig. S4(c). Consequently, breaking one symmetry can induce strong hybridization between these two states.

![](images/5098713acff477228b9f050745cbbc7c869873f7c31a373114d037e8d43caabc.jpg)

(b)
![](images/4b6f80e99b6553d60424a26309cd210e506053796fabd2fc093346f37bca8b72.jpg)
FIG. S4. (a) Schematic diagram of Q factors corresponding to bulk states and edge states of eTBIC and TBIC. (b) The fields and Q factors of the bulk and edge state of TBIC under the radiation boundary. (c) The fields and Q factors of the bulk and edge state of eTBIC under the radiation boundary.

## Sound field discussion of the eTBIC

In Fig. S5, we give the energy spectrum and corresponding sound fields of eTBIC as shown in Figure 2. The blue background is used to label the edge state of the $\Sigma$ mode, and the pink background is used to label bulk modes of the $\Pi$ mode. The corresponding sound fields are shown in Fig. S5(b). The edge state of $\Sigma$ mode is a gradually decaying waveguide mode in which the sound energy is concentrated in the center of the system. The $\Pi$ bulk modes acting as the continuum with sound field appears as cavity modes symmetrically decoupled from the waveguide mode.

(a)
![](images/ba381fa09108e1bc5498060b6d7f8c49d6b5885dd6d5fdf972d1be82d66620c4.jpg)

![](images/c417e6e53e728e6bbfd556b2ab960e406151cb885127408580a5be3986d508a5.jpg)
FIG. S5. The energy spectrum (a) and corresponding sound fields (b) of the eTBIC.

## Evolution of topological states

In this section, the effects of cavity width and waveguide width on topological state evolution are discussed. We showcase this tunability by adjusting the waveguide width $(W)$ in Fig. S6. The phase diagram in the parameter space of frequency and W reveals the evolution of band structures for both $\Pi$ and $\Sigma$ modes. The blue (red) bands correspond to the bulks of $\Sigma$ ( $\Pi$ ) states, while the dots in corresponding colors represent the respective TESs. Since $\Pi$ modes are cavity-dominated, varying waveguide width significantly influences their coupling strength. Consequently, increasing W effectively increases the bulk band width of the $\Pi$ mode. Such an effect also occurs in the evolution of $\Sigma$ modes. In general, the tuning of the waveguide width is relatively limited for the energy band manipulation of this system, where only traditional TES and eTBIC can be observed, but not TBIC and TF. We also show the tunability in terms of topological state evolution by adjusting the cavity width $(U)$ , as shown in Fig. S7. Since $\Sigma$ modes are primarily governed by the waveguide, increasing the cavity width has almost no effect on their band structures. However, the increase in cavity width effectively modulates the $\Pi$ modes dominated in the cavity, resulting in rich topological phenomena by inducing a continuous evolution of the four topological states (TES, TBIC, eTBIC, TF) in the parameter range.

![](images/0d59966fd774b512ce4067713ce5090e1d952ac354ad37de6e6217d2d40387cd.jpg)
FIG. S6. The evolution of topological band structures with the width of waveguide (W). Bulks and edges can be simultaneously tuned in each subspace without hybridization, resulting in rich topological features: eTBIC and TES. Those blue (red) bands and dots represent the bulks and edges of $\Sigma$ ( $\Pi$ ) modes, respectively.

![](images/49a81da24077cf5dbc26b761fd0b2ae7ae1696394bf4e5835600d4c8bf609a58.jpg)
FIG. S7. The evolution of topological band structures with the width of cavities $(U)$ . Bulks and edges can be simultaneously tuned in each subspace without hybridization, resulting in rich topological features: TBIC, eTBIC, TF, and TES. Those blue (red) bands and dots represent the bulks and edges of $\Sigma$ ( $\Pi$ ) modes, respectively.

## Supplementary Section B

## The transmission response of different topological states

In the main text, we delve into the study of the energy spectra encompassed by TBIC, topological Fano resonance, and eTBIC. Within this section, we discuss their transmission response characteristics.

In Fig. S8, we present the eigenfrequency analysis of eTBIC and the corresponding transmission spectral lines under a radiation boundary condition for three perturbation conditions (with perturbation strengths of 0, 0.5, and 1, respectively). Visual inspection of the data reveals that the splitting of the transmission spectral lines is attributed to mode hybridization.

![](images/51ef33a5ccbfc545f5da99c248ff1c950ff61d641c9e801819d9cc2aa13b732e.jpg)

![](images/6686bae805363d70215a4d253acf1c931482ea88309ef7b849d7c925b559094e.jpg)

![](images/e64323a29567533edf705352a6f1e0b90c8fbacf2a3d5d0bae07e1695e9e3c6b.jpg)
FIG. S8. The energy spectrum and transmission of the eTBIC under three different intensities of perturbations. The structural parameters utilized here are consistent with those shown in Fig. 4(b) of the manuscript.

In Fig. S9, we present the transmission responses of TBIC, topological Fano resonance, and exotic TBIC under symmetric excitation. From the left panel of Fig. S9(a), it is evident that, in the absence of perturbation, TBIC exhibits wide-band and high-transmission behavior, which is attributed to the $\Sigma$ mode bulk band at the observed position. However, it remains unexcited due to the decoupling between the topological protection edge state supported by the $\Pi$ mode, and the plane-wave mode (symmetric mode). At this point, the $\Pi$ mode-supported topological edge state resides within the $\Sigma$ mode bulk band. Upon applying a perturbation along the lateral periodic direction, the $\Pi$ mode ceases to be decoupled from the plane-wave mode, resulting in a clear single-frequency transmission oscillation with a high quality factor along the transmission spectrum. This phenomenon is consistent with common BICs [S3]. The topological Fano resonance arises from the degeneracy, or coincidence, of the edge states supported by the two different modes on the energy spectrum. Under normal conditions, due to mode decoupling, only the transmission peak originating from the edge state of the $\Sigma$ mode is discernible on the transmission spectrum. However, when a minute perturbation is applied perpendicular to the periodicity, the edge state of the $\Pi$ mode is no longer decoupled from the edge state of the $\Sigma$ mode. Instead, the two edge states have very different $Q$ factor, giving rise to the characteristic Fano resonance shape [S4]. Under plane-wave excitation, eTBIC displays a single transmission peak profile, akin to the topological Fano resonance, both of which feature a transmission peak that stems from the edge state of the $\Sigma$ mode. However, the edge state of the $\Sigma$ mode in eTBIC resides within the $\Pi$ mode bulk, differentiating it from other phenomena. When eTBIC experiences the same perturbation as TBIC, the plane-wave mode couples to and excites the $\Pi$ mode bulk band. Consequently, the transmission peak originating from the $\Sigma$ mode edge state is overwhelmed by the transmission perturbations induced by the $\Pi$ bulk band in the spectrum. This divergence is pronounced compared to the outcomes observed in common BICs. Simultaneously, the hybrid interaction between the $\Sigma$ mode edge state and the $\Pi$ mode bulk results in a splitting into two distinct transmission peaks.

To further demonstrate the difference between eTBIC and TBIC, we compare the following two cases: TBICs excited by antisymmetric excitations and eTBICs excited by symmetric excitations (as illustrated in Fig. S10 below). Figure S10 shows the transmission spectrum for these two cases. The difference between the two is evident from the outset: eTBIC can induce perfect transmission, where the bulk is muted and the edge can be uniquely identified in the transmission spectrum as a peak with theoretically 100% efficiency. However, for the TBIC case, neither the bulk nor the edge can be observed in the spectrum. The disappearance of the bulk band comes from the decoupling of the symmetry of the excitation mode from the Waveguide propagation mode, and the inability to observe the edge state results from the “bandwidth vanishing” caused by its high Q factor. Note that a high signal-to-noise ratio is crucial for sensing, therefore a high transmission pinpointing the eTBIC lays a foundation for our work.

In order to study in greater depth the possible weak variation of this TBIC under perturbation, we have redrawn Fig. 4 using a logarithmic scale. Now, in the unperturbed case, we can see that the edge state of TBIC is also a transmission peak, but it is a very low peak with an amplitude of only $10^{-4}$ , as shown in Fig. S11. The very small transmitted energy results in a poor signal-to-noise ratio when used for sensing. When perturbations are introduced in one dimension, TBIC exhibits a similar robustness behavior as eTBIC, with its faint, imperceptible transmission peaks remaining unchanged. Upon the introduction of lateral perturbations, the antisymmetric excitation mode is coupled with the waveguide transmission mode. Consequently, the bulk response of the TBIC overlaps with the original edge state response, causing the peak value of the latter to diminish into a barely noticeable trough within the transmission spectrum. Another significant aspect that cannot be overlooked is that the edge state of TBIC demonstrates consistent robustness against various symmetry perturbations, as evidenced by the nearly constant frequency of its edge state response.

![](images/90389481d21709c679dc524275f8c9194113b47efde4f1dd841f18956b3fad3f.jpg)

![](images/ae9c7e05efad63ee8268fa9851998af64eae634739ac2d9ee2616a4d07901259.jpg)

![](images/76c82bfa84b85967c3399df1d0c717215349a898e4efca84fedb1677b8cded6f.jpg)

![](images/fb3e43132bb0be8eca348fc625a697d0454455b648951f4a73a3d79794438983.jpg)

![](images/91f9d1f7a18c4c85f2d5d7e48f4431f58d773418fba3e53f9802312ca8759f3d.jpg)

![](images/ee7acae2f84a792f1c41cb99dc47bc5f7e0207b9c86fd379e91fa61236940f91.jpg)
FIG. S9. (a) The left panel is the power transmission of TBIC supercell containing three nontrivial and three trivial lattices under plane wave excitation, and the structural parameters inherit the structural parameters used in Fig. S2. The height of C is changed to 52 mm. The right panel is the power transmission in the case of applied lateral perturbations with strength of 0.5 mm. (b) The left panel is the power transmission of two edge states degenerate supercell containing three nontrivial and three trivial lattices under plane wave excitation, and the structural parameters inherit the structural parameters used in Fig. S2. The height of C is changed to 62.1 mm. The right panel is the power transmission of topological Fano resonance in the case of applied lateral perturbations with strength of 0.01 mm. (c) The left panel is the power transmission of eTBIC supercell containing three nontrivial and three trivial lattices under plane wave excitation, and the structural parameters inherit the structural parameters used in Fig. S2. The height of C is changed to 59.4 mm. The right panel is the power transmission in the case of applied lateral perturbations with strength of 0.5 mm. In these three sets of simulation, the way we apply lateral perturbation is unified as $[+ \delta, -\delta, +\delta, -\delta, +\delta, -\delta, -\delta, +\delta, -\delta, +\delta]$ .

![](images/fec8122df1af5675ccdce8b460961907e102715a8b83b6c5c5825c4cafdeedd6.jpg)
(b)

![](images/dd0c0d1fe662c3a8bf271bfd182c36713adb2a682e1ba03af9182e664f7df245.jpg)
FIG. S10. Transmission response of TBIC and eTBIC under antisymmetric and symmetric excitation. The upper panel is the field distribution, and the lower panel is the transmission response. The structural parameters utilized here are consistent with those shown in Fig. 4 of the manuscript, with the height of C adjusted to 52 mm for TBIC and to 59.4 mm for eTBIC.

In Fig. S12, we further demonstrate the evolution trend of the quality factors of bulk and edge states of TBIC and the two hybrid edge modes of eTBIC with respect to the lateral perturbation strength. For TBIC, there is always a huge difference of order of magnitude in the quality factors of edge state and bulk, while for eTBIC, the quality factors of the two hybrid edge modes are always similar. Therefore, as we have pointed out in the main text, the similar quality factor of the edge and bulk state of eTBIC is crucial for the observation for strong hybridization (leading to the splitting of transmission peaks) found in our work.

(a)
![](images/4c66f1677942c00023b0e981ff6dafb341df35a6ddc3fea58a6812c8ebcee3ce.jpg)

(b)
![](images/f92a6998508cdfbbc46f0823f748f0f24ca7160ad43dd2087efddc36745f1cee.jpg)

![](images/fba1e9dae66959f258f53d476fa8d66e6ac9a407f031340c60760887ad661298.jpg)

![](images/f84f6fdd1df4d0a27c36f231b1fa6c50f83a3bd5c31f1e53fe52605dda5c30d4.jpg)
FIG. S11. The logarithmic version of the results presented in Fig. 4.

![](images/34bf65bfbca50d5d5f3c600a95abdd1683d4c1d3c8d942f5b0610ddcf859b0fe.jpg)
FIG. S12. Evolution trend of eTBIC and TBIC quality factors under lateral perturbation. The structural parameters utilized here are consistent with those shown in Fig. 4 of the manuscript, with the height of C adjusted to 52 mm for TBIC and to 59.4 mm for eTBIC.

## Discussion of mode hybridization in eTBIC

In the previous section we mentioned that eTBIC's bimodal transmission lines arising from the application of perturbations perpendicular to the periodic direction come from mode hybridizations, and in this section we will illustrate this phenomenon in combination with mode analysis.

We give the energy spectrum of same eTBIC supercells as mentioned in Fig. 2 and the energy spectrum of this case after a lateral perturbation with an strength of 1 mm perpendicular to periodic applied. It can be seen that after the perturbation, two modes separate from $\Pi$ mode bulk appear at the upper and lower ends of the bulk in mode $\Pi$ , which correspond to the bimodal line in the transmission spectrum line. After hybridization, the distribution of the two modes of sound field show similar patterns (approximately reversal to each other), and the difference only exists in the radiation pressure intensity near the lattice ends, indicating different Q factors for these hybrid modes. It can be concluded that the appearance of the bimodal pattern comes from the separation of the two hybrid modes from the bulk. Combined with the sound field analysis corresponding to these modes, we can observe that both modes have the characteristic of being bound to the central interface. We believe that this hybrid process is as follows: the edge state supported by $\Sigma$ mode and a bulk state of $\Pi$ mode are split after hybridization. The two new hybrid modes after splitting inherit the edge state property, resulting in two distinct transmission peaks on the transmission spectrum line. When 1 mm lateral perturbations are applied to the TBIC, it is due to the influence of mode symmetry and topological protection that the $\Pi$ mode edge state hardly hybridizes with the $\Sigma$ bulk, so the edge state of the TBIC is still in the bulk. The $\Pi$ edge state in TBIC after perturbation has only weak leakage, and its symmetry is still consistent with that without perturbation.

![](images/ab378215b1c8326999ded15c832773132059a1bb88111010833f6a3989c4ff94.jpg)

(b)
![](images/1c1ba1a943348b762d261b3cd60273f627f5bb2f6ad6002a444d2f54cb6a4dd5.jpg)

![](images/1e1d08baf92edf6749b437d0e92dd7bb4124dd3bbf00738d394170ca6837fa78.jpg)

![](images/28a0bcaa99dfdfa6bfd3e7b5bf116add2768890e97312b865c61850f7e61c852.jpg)

(e)
![](images/bc987cc94724f1f0e5541b14db287be04a851e749ff846effca54195a383a8a0.jpg)

(f)
![](images/24bc06c09418cb84708cd7779d1907bc7dca19a41502d6bb36ac6bfbbe376b73.jpg)
FIG. S13. (a) Energy spectrum of eTBIC, color is used to distinguish mode types, edge states and bulks using size distinctions of dots. A supercell composed of five nontrivial and five trivial lattices is used in the analysis. The corresponding parameters are the same as those used in Fig. 4(b). (b) The energy spectrum of (a) after a lateral perturbation with strength of $1\mathrm{mm}$ is applied to lattices. In these three sets of simulation, the way we apply perturbation perpendicular to the periodic direction is unified as $[+\delta, -\delta, +\delta, -\delta, +\delta, -\delta, +\delta, -\delta, +\delta, -\delta, +\delta, -\delta, +\delta, -\delta, +\delta, -\delta, +\delta, -\delta, +\delta]$ . The purple dots represent hybrid modes resulting from system hybridization. The illustration is a partial enlargement of the lower bulk band corresponding to mode $\Pi$ . (c) The sound field distributions corresponding to the $\Pi$ mode bulk, $\Sigma$ mode edge and two hybrid modes. (d) Energy spectrum of eTBIC. The corresponding parameters are the same as those used in Fig. 4(c). (e) The energy spectrum of (d) after a lateral perturbation with strength of $1\mathrm{mm}$ is applied to lattices. (f) The sound field distributions corresponding to the $\Pi$ edge featured in both (d) and (e).

## Supplementary Section C

## Sensitivity experiment setting

Figure S14 shows a photo of our experimental platform. Our experimental platform involves an acoustic impedance tube with a loudspeaker mounted at one end (to generate plane incidence waves) and absorbing materials attached at the other end (creating an anechoic termination). Placed in the middle of the impedance tube is a sample of the proposed topological supercell, containing three trivial and three nontrivial unit cells that support the eTBIC. The sample shown in Fig. S14 is fabricated via the stereo lithography apparatus (SLA) with a precision of 0.1 mm using photosensitive resin (density $\rho_{m} = 1.16 \times 10^{3}$ kg/m $^{3}$ , Young's modulus $E_{m} = 2450$ MPa, and Poisson's ratio $\nu_{m} = 0.41$ ). The experimental system is transformed from the impedance tube transmission measurement system, and the test waveguide and sample with a wall thickness of 4mm are constructed by 3D printing. The signals (amplitude and phase) are detected by the four microphones (GRAS, type 46BD) fixed on the wall of the waveguide. The measured signals are processed by the PXI Multifunction I/O Device (National Instruments, type PXIe-4497). The experimental test method is referenced to the transfer matrix method [S5]. Considering that the effective precision of 3D printing we used is 0.1 mm, we set the sample perturbation strength range from 0.2 mm to 1.0 mm and the gradient of perturbation strength is set to 0.1 mm. In addition, in order to eliminate the possible uncontrollable errors in sample manufacturing, we manufactured multiple samples for each perturbation gradient.

![](images/cb854529f27165533d6ca273556530a3934587795d501b2772e8cf34856d8f66.jpg)
FIG. S14. The top panel displays detailed information about the experimental platform, while the lower panel presents a photograph of the experimental samples.

Below are some details and discussions regarding sensitivity simulations and experiments. The structural parameters utilized in both the sensitivity simulation and the experiment are presented in Fig. S15. Each side of the structure is supported by three nontrivial (trivial) lattices. To demonstrate the anisotropic sensitivity of our proposed eTBIC, we have also investigated the response of the structure under a single directional perturbation. The perturbations employed in both the simulation and the experiment are depicted in Fig. S16. The global perturbation in our study is defined as the sum of two independent single perturbations applied in different directions.

![](images/2c3869ce3ebdeb416a00f8516dae504159097efe8523b4576b220805d7c89320.jpg)
FIG. S15. Schematic diagram of the air domain in the simulation and experimental structure.

![](images/a2fdecbdcc000789008c41cf743c2167366b32f6e390a7b5c901352945792354.jpg)
FIG. S16. Diagram of perturbation direction in sensitivity simulation and experimental verification. The top panel shows the direction of axial perturbation, the middle panel shows the direction of lateral perturbation, and the lower panel shows the direction of global perturbation.

## 3D model validation

Compared to the simulation in 2D, the geometries of the third direction of the experimental sample remains unchanged, therefore the results from simulation (2D) should be consistent with the experimental results (3D). To confirm this, we carried out eigenfrequency analysis on both 2D and 3D models. As shown in Fig. S17, these two models (2D and 3D) are consistent with each other, which underscores the feasibility of our proposed experimental verification model.

![](images/1b596e0af012c1a8c86695b4a762fdbcc7117413bc54d3aeed2a9d8569c1cb82.jpg)

![](images/dcb59f417a1eb047684bc6c070340f1f0c4cc2f4f2aa8f2a9e8d0b0b850eece2.jpg)
FIG. S17. Energy spectra of eTBIC in 2D and 3D models.

## Discussion of system losses

It should be noted that the intrinsic thermal viscous loss of the structure is considered in the simulation of sensitivity verification. (Thermoviscous Edge Layer Impedance is imposed on all the walls of the structure, and the mechanical and thermal conditions on these boundaries are respectively set as No-slip and Isothermal.) However, the introduction of this term does not affect the modal response of the observed system under perturbation. As shown in Fig. S18, we present the evolution trend of the power transmission coefficient with perturbation strength under different perturbation modes, both without and with considering the structural intrinsic thermal viscous loss. It can be seen that the system retains the predicted evolution even when considering the intrinsic thermal viscous loss of the structure. This also proves that the inherent loss within the system under study does not have a substantial impact on the problem being studied. The experimental scheme designed by us to verify the sensitivity of the system is feasible.

The peak transmission height of the system can be discussed as a second index of detection, whose physical substance actually comes from the Q value of the mode. In Fig.

![](images/408b9012c8cc2d35ad1b640373287fd3112674f7a0da06b630c151999078389a.jpg)

![](images/c97c9ef00bb034d750df728ce4803eaba69d3fe3b63ea51022deb95c425754de.jpg)

(b)
![](images/f8674e886d697fb9997cc03e68bb15ce1b331f33b0286a97bcf30ab18ec38a03.jpg)

![](images/8847d457892b092373a47e0bf3d002dbf436aaaa6dfabd8cd0a2535de5501f7b.jpg)

(c)
Global perturbation
![](images/fabd298e3e015b2e469ee061cd323aabfa1e66a8f55842482665c9b60fb4abd7.jpg)

![](images/8380d4a8ea92ceae812256789b9543b948f727c2c621fb30368b4c3cb59666c0.jpg)
FIG. S18. (a) Evolution of transmission with the strength of perturbation when a lateral perturbation is applied to the system. The left panel shows the result without considering the loss, while the right panel shows the result after considering the loss. (b) Evolution of transmission with the strength of perturbation when an axial perturbation is applied to the system. The left panel presents the result without considering the loss, and the right panel presents the result after considering the loss. (c) Evolution of transmission with the strength of perturbation when a global perturbation is applied to the system. The left panel displays the result without considering the loss, whereas the right panel displays the result after considering the loss.

S19, we showed that these two modes have different Q values. Therefore, introducing losses in this system leads to the difference in the peak values of the two hybridized edge states, which is due to the difference in the quality factors (Q factors) of the two states (Fig. S19). We can understand this by the following way: Different Q factors symbolize that the edge states have different confinement capabilities with different radiation loss. Therefore, for the same structure (the same intrinsic loss or dissipation loss), different radiation loss will differently couple to the same intrinsic loss, resulting in different transmission peaks (it is well known that when radiation loss matches the intrinsic loss will result in critical coupling and perfect absorption, leading to zero transmission, so, in this case, different coupling between radiation loss and intrinsic loss will result in different transmission peaks). This suggests that loss may be an additional degree of freedom for the proposed eTBIC-based sensing. Our sensing approach not only relies on the difference in frequency (the transmission peak frequency, $\Delta\omega$ ) for detecting perturbations in geometries, but it also considers the difference in amplitude (the transmission peak height) for detecting perturbations related to non-Hermitian parameters such as loss and gain. Here, we present one of the most direct methods to determine sensitivity using peak difference, that is, to characterize sensitivity by the height difference between two peaks. As shown in Fig. S20(b), sensitivity simulation results based on peak height difference are given. Compared with the sensitivity results obtained through frequency splitting (i.e. frequency difference) shown in Fig. S20(a), the peak-based sensitivity response demonstrates a capability to effectively differentiate between global and lateral perturbations.

![](images/fa6687a197712fe9bbf506d6fce0e1c4ad11d28da8720098e60a7fdf56114435.jpg)
FIG. S19. Evolution of absorption with the strength of perturbation when a global perturbation is applied to the system. On the right side are the corresponding eigenfields and Q factors.

(a)
![](images/66be0a77b82597c7035aa80e32c16c3de8ae99d5e67242f4cb92a716d602eb3e.jpg)

(b)
![](images/fa4177cac674e82ba3cd422ce43ae3f0b1d7ce1c1296332df483c51f2d46acd4.jpg)
FIG. S20. (a) Sensitivity of the eTBIC, measured by tracking the variation of the frequency difference $(\Delta\omega)$ between the split peaks with increasing perturbation strength. Perturbations are randomly induced in either only axial, only lateral, or both axial and lateral directions. The blue (red) solid line represent the simulated results for the eTBIC with only axial (lateral) perturbations, while the green line represents the simulated results for the eTBIC with global perturbations. (b) Sensitivity of the eTBIC, measured by tracking the variation of the transmission response difference $(\Delta T/T_{max})$ between the split peaks with increasing perturbation strength. Perturbations are randomly induced in either only axial, only lateral, or both axial and lateral directions. The blue (red) solid line represent the simulated results for the eTBIC with only axial (lateral) perturbations, while the green line represents the simulated results for the eTBIC with global perturbations.

## Experimental results

Figure S21 presents the results utilized in the main text for calculating the sensitivity of eTBIC under various perturbations, as well as the experimentally obtained structural normalized power transmission coefficients under varying intensities of gradient perturbation. The experimental results are in good agreement with the simulation data, indicating a high level of experimental confidence. The eTBIC system exhibits strong robustness to axial perturbation, and it can be seen from the experimental transmission spectrum that the system always maintains a single transmission peak. The eTBIC system has a strong response to lateral perturbation, and it can be seen from the experimental transmission spectrum that the splitting peak gradually appears with the increase of perturbation strength. From the experimental results, it can be deduced that the sensitivity characteristic of the system under global perturbation stems from the sensitivity to lateral perturbations. Conversely, for axial perturbations the system exhibits a high degree of robustness.

(a)
![](images/759e4be2bc512c4254602ed1c8f0a4a195d3fad80d01869310fde75b410ff90b.jpg)

(b)
![](images/d4328316f720aac4de5a49020ff84bf3e485f2f32c3759823726a9f401c2a82b.jpg)
(c)

![](images/17ad3381870c88b93d44a5631b093ff0f06bbc3c043e4f07407df1a2edf611e7.jpg)

![](images/265d134581cf9540bc634921e230ba7840054ccb4b212f30a73739e2e438fc87.jpg)

![](images/b60a10b0f3b5f2ee48fc37d21fd6afc91d05d0df6f2215a40cd20a00442e971e.jpg)

![](images/24c29adc36fc2b9d1bc3294f88d79fe3e9878466914591e71b32e422d81439b8.jpg)

![](images/ffe4af3ed2d0d105a5975c05a8fe54ead0c66e9f5e1697300e7d59dd012e24e8.jpg)

![](images/ee09d483cfc3b43bc15d2e8ca51680f63ad56ba811577155e7a20bd5a61d12d0.jpg)

![](images/00123dc168511ab17c7167df44482b140e0fefbec8cbce3e916983dde8b89c63.jpg)

![](images/abc3352f7650f3ced48b7b9d9b1d081e140f3f57c7179fc74bfc20dc738cfd0a.jpg)

![](images/488f50b428e2416790ae02ae40f3d92bdf6ac130da50b90d64a4f8719e2135d6.jpg)

![](images/9ea41373826ecaff00ad071477e599fd0578bb320a6878ef78fa992c820ca962.jpg)

![](images/c2519192d196703ad631e4357019362663274d8722834356190b08a394be643c.jpg)

![](images/f0d191e713f9bf7b7b5375d4f015f4ababe1070ab0b67d513a19a67715b046d5.jpg)

![](images/8d312624041a41b047ebedb1fdbb1e03ca52a29daecc0b5ede12e86ef1aec706.jpg)

![](images/8e320c131a2fc32ac86d2e8531af8c3627d423380d6fc4389355fd0922bb3319.jpg)

![](images/fc7cda706ce90276f39d24b114e3bb5c5564290299d7a399dd7628e70b0838a5.jpg)

![](images/fee5f3e9321ad6a41df06582032f884e7a02b7433c6ee534cdc8e99825bd5503.jpg)

![](images/64528a69fd833bdadcd6ca01a72816cfdc39455efdaee6aea992f7328c704c54.jpg)

![](images/9a192b0335c1ca7ee2877876086040824d11f37b0222c1f8d3cb3dc244c05230.jpg)

![](images/1cfe49cb376f506928a62d76c66be2ab330f92c26d20e2e1d6f502d5bb299895.jpg)

![](images/fb32cf3a93fcde1de02cb6a1101679c709a08d1decd47054634afa13b6b056fe.jpg)

![](images/69f2eab7b2c0852addf11f093babe583bf8836c693498127823a48718c8debe8.jpg)

![](images/185253a2223ead90342ca5d25958d1abc1622ddafd4e63f504f64dd9bd04eeb7.jpg)

![](images/f260034c148b0a7823fc2ac1c6daf6a3d19e6dff7ad0aa01042a69bf16242ad2.jpg)

![](images/2a36a8550a112104960cf5fb96edcdff6e6ed2fc68f196cbe5f812cc4dc3ad7c.jpg)

![](images/38b393a9eb20946eb752037e59f344dcdfc191e5e0124ebc9c73e8e9a046f86c.jpg)
FIG. S21. Transmission of the eTBIC under three different types of perturbations. All the perturbations are with a gradient of 0.1 mm from 0.2 mm to 1.0 mm. (a) Evolution of transmission with the strength of perturbation when axial perturbation is applied to the system. (b) Evolution of transmission with the strength of perturbation when lateral perturbation is applied to the system. (c) Evolution of transmission with the strength of perturbation when a global perturbation is applied to the system. Here, both simulated (solid curves) and measured (dots) unperturbed results are presented in blue, while the perturbed results are in red.

## Simulation results of different perturbations

Considering the practical limitation of the number of experimental samples, we also added some simulation verifications under different perturbations for completeness. Based on the perturbation modes in the main text, we have randomly combined multiple configurations. Under different kinds of lateral perturbations, the transmissions of eTBIC show an obvious trend of peak splitting (as shown in Fig. S22). However, regardless of the various random axial perturbations applied, the eTBIC maintains its robustness and its transmission remains unimodal (as shown in Fig. S23). These simulation results strongly support the anisotropic sensitivity of eTBIC proposed by us.

# # # # # # # # # # #
![](images/3130353c8c6a41ab157062e50cc23b67de0d0021e4dc18096fb4fbfcaa427636.jpg)

![](images/12121d91e71cf9d307b8f3b362a8a2e5d0a0a2996cba0a426637fdb9104285f5.jpg)

![](images/5611ba2717e827039fd4373a2a8f9916c6deb829824a9bbcef765998cb72d8cd.jpg)
# # # # # # # # # #

![](images/ccb4652124db071ce3daa15ac5867e5c54b1d7bfeec10f16afab27d6bd6f381b.jpg)

![](images/cfe553e2c5469254fd828bf9c650f40b2e0cb9dacfd6fc75480506acc272a8b2.jpg)

![](images/01af0331c3f11cb0bc2059c08db8dbc5b37bfef0f8ed09efcff98dd9a27121ca.jpg)
FIG. S22. Evolution of transmission with the strength of perturbation when different lateral perturbation is applied to the system.

- - - - - - - - - - - - -
- - - - - - - - - - - - - - - -
![](images/5f750374d73cc33b408269358f6846f9e15cd33f069adca1663fd49483ac13bb.jpg)

![](images/c0f6f1a21ab2f5835b7ea7b294996fa7935547ce1bcc4cc4577aa4c0a4a7e485.jpg)

- - - - - - - - - - - -
![](images/670914bb8e27992f77e86d59db814e794b00b7430177fce083d1ba8441b0795a.jpg)

- - - - - - - - - - - - - - - -
![](images/b66d1293aa2bcf4c82473cfa89767c53fba230ee2b3069f5ffa59756f7b90a95.jpg)

- - - - - - - - - - - - -
![](images/3d56c1b4fd54433d8aa93a9a6b0e328d2f2054e026649dc8f8f7cba065ae4464.jpg)

→ ← ← ← ← ← ← ← ← ← ←
![](images/e4b15780e976d3e6dd60a3688884f41dc4583b182cdba8834af857975a39422c.jpg)
FIG. S23. Evolution of transmission with the strength of perturbation when different axial perturbation is applied to the system.

The average perturbation, as a particular instance of random perturbation, is addressed separately. Given symmetry considerations, we have focused solely on upward lateral perturbations and rightward axial perturbations (as shown in Fig. S24). The simulation outcomes align with our predictions, confirming that the sensitivity of anisotropy persists within the system. For lateral perturbations, the initial values of the perturbation are $[0.1, -0.1, 0.05, 0.05, -0.05, -0.05, 0.15, -0.15, 0.1, -0.1, 0.1, -0.1]$ . Therefore, the eTBIC has already experienced peak splitting at the zero point of the perturbation. For axial perturbations, the initial values of the perturbation are $[0.1, -0.1, 0.05, -0.05, 0.15, -0.15, 0.15, 0.15, -0.15, -0.15, 0.05, -0.05]$ .

(a)
![](images/7586a7f824981794ab6fc22373bff4588b3107a7d826eee2e04fb976c7a7f78f.jpg)

(b)
![](images/9d779ad1e4a9471c578e6be8ad24b99c86c4ba7241ff52f3b3348a9232238290.jpg)
FIG. S24. (a) Evolution of transmission with respect to the strength of perturbation when an upward lateral average perturbation is applied to the system. (b) Evolution of transmission with respect to the strength of perturbation when a rightward axial average perturbation is applied to the system.

[S1] M. Xiao, G. Ma, Z. Yang, P. Sheng, Z. Q. Zhang, and C. T. Chan, Geometric phase and band inversion in periodic acoustic systems, Nat. Phys. 11, 240 (2015).

[S2] X. Li, Y. Meng, X. Wu, S. Yan, Y. Huang, S. Wang, and W. Wen, Su-Schrieffer-Heeger model inspired acoustic interface states and edge states, Appl. Phys. Lett. 113, 203501 (2018).

[S3] L. Huang, Y. K. Chiang, S. Huang, C. Shen, F. Deng, Y. Cheng, B. Jia, Y. Li, D. A. Powell, and A. E. Miroshnichenko, Sound trapping in an open resonator, Nat. Commun. 12, 4819 (2021).

[S4] F. Zangeneh-Nejad and R. Fleury, Topological fano resonances, Phys. Rev. Lett. 122, 014301 (2019).

[S5] Standard test method for normal incidence determination of porous material acoustical properties based on the transfer matrix method, ASTM E2611-24 (2024).
