# Observation of Anomalous Floquet Non-Abelian Topological Insulators

Huahui Qiu $^{ID}$ , $^{*}$ Shuaishuai Tong $^{ID}$ , $^{*}$ Qicheng Zhang, Kun Zhang, and Chunyin Qiu $^{ID}$ $^{\dagger}$

Key Laboratory of Artificial Micro- and Nano-Structures of Ministry of Education and School of Physics and Technology, Wuhan University, Wuhan 430072, China

(Received 25 March 2025; accepted 17 February 2026; published 20 March 2026)

Non-Abelian topological phases, which go beyond traditional Abelian topological band theory, are garnering increasing attention. This is further spurred by periodic driving, leading to predictions of many novel multigap Floquet topological phases, including anomalous Euler and Dirac string phases induced by non-Abelian Floquet braiding, as well as Floquet non-Abelian topological insulators (FNTIs) that exhibit multifold bulk-edge correspondence. Here, we report the first experimental realization of anomalous FNTIs, which demonstrate topological edge modes in all three gaps despite having a trivial bulk charge. Concretely, we construct an experimentally feasible one-dimensional three-band Floquet model and implement it in acoustics by integrating time-periodic coupling circuits to static acoustic crystals. Furthermore, we observe counterintuitive topological interface modes in the domain wall formed by an anomalous FNTI and its counterpart with swapped driving sequences—modes previously inaccessible in Floquet Abelian systems. Our work paves the way for further experimental exploration of the uncharted nonequilibrium topological physics.

DOI: 10.1103/qn87-bm33

Subject Areas: Acoustics,
Materials Science,
Topological Insulators

## I. INTRODUCTION

The discovery of topological insulators has revolutionized our understanding of material properties through the lens of symmetry and topology $[1-5]$ . The global invariants underlying these phases are often characterized by Abelian charges like Chern numbers or winding numbers. Recently, the concept of non-Abelian charges has been introduced to PT-symmetric systems with multiple intertwined band gaps $[6-10]$ . For instance, a one-dimensional (1D) three-band topological insulator can be described by the non-Abelian quaternion group $Q_{8}=\{+1,\pm i,\pm j,\pm k,-1\}$ , which obeys the fundamental multiplication rules $i^{2}=j^{2}=k^{2}=ijk=-1$ and ij=-ji, ik=-ki, jk=-kj. In this context, the charge $q=+1$ signifies a trivial phase without topological edge modes (TEMs) in any band gap. In contrast to their Abelian counterparts, non-Abelian topological phases display more intricate behaviors, such as trajectory-dependent nodal point collisions in two-dimensional semimetals, admissible nodal line configurations in three-dimensional semimetals, and multigap TEMs governed by nontrivial quaternion charges $[6–10]$ . Experimentally, non-Abelian phases with tangled multigap topology have been realized across a variety of platforms, including photonic systems $[11–14]$ , acoustic crystals $[15–18]$ , and transmission line networks $[19,20]$ .

Floquet systems are out-of-equilibrium quantum states that evolve under periodic driving $[21-30]$ . As a powerful tool for manipulating band structures, Floquet engineering enables novel topological phenomena without static analogs, such as anomalous chiral edge modes even when all bulk bands carry trivial Chern numbers $[30]$ . Very recently, the synergy of Floquet setting and non-Abelian topology has sparked the prediction of numerous unprecedented topological phases with unique non-Abelian multi-gap dynamics—features that break free from the paradigmatic tenfold way and expand the scope of non-Abelian topological physics beyond traditional static frameworks $[31-33]$ . Notable examples include anomalous Euler (Dirac string) phases $[31]$ and Floquet non-Abelian topological insulators (FNTIs) $[33]$ . Intriguingly, the latter exhibit a multifold bulk-edge correspondence governed by the multiplication rule of the quaternion group $Q_{8}$ (see Fig. 1). Mathematically, the bulk charge q of a 1D FNTI equals the ordered product of the (quaternion) charges $\bar{q}_{m}$ of all phase-band singularities (i.e., $q = \prod_{m} \bar{q}_{m}$ ), each resulting in a change in the mass term and the emergence of an in-gap TEM according to Jackiw-Rebbi's argument [33,34]. This enables multifold bulk-edge correspondence, as a bulk charge q can be expressed in different multiplicative forms, each corresponding to a specific TEM configuration. For example, in the absence of any phase-band singularities, the trivial charge $q = +1$ does not contribute any TEM. In contrast, the charge $q = +1$ represented by $\bar{j} \cdot \bar{i} \cdot \bar{k}$ gives rise to TEMs in all band gaps, thereby defining an anomalous FNTI. Here, “anomalous” specifically denotes that the quaternion charge +1, which corresponds to a trivial phase with no TEMs in static (equilibrium) non-Abelian systems, can give rise to TEMs in all three band gaps under Floquet driving.

![](images/018de847abeb9c7c8a32ab3a9d060dc767d2998ba4628a968dc0506b7539f527.jpg)

(b)
![](images/08057bca08636da91da9591f941f496797f21c73ad37324a538268b441e9ee2e.jpg)
FIG. 1. Multifold bulk-edge correspondence of 1D three-band FNTIs. (a) Topological transitions and the resulting diverse TEM configurations, with the bands and band gaps numbered in (b). Starting from the trivial phase ① with $q = +1$ , the system undergoes gap closure and reopening in the second/third/first gap, yielding a nontrivial FNTI ②/⑥/④ with $q = \pm i / \pm j / \pm k$ . Further phase transition leads to the FNTI ③/⑦/⑤, which exhibits distinct TEM configurations despite belonging to the same conjugacy class of ⑥/④/②. The subsequent phase transition culminates in the anomalous FNTI (⑧) that hosts TEMs in all three gaps, despite having a trivial charge of $q = +1$ . The dashed arrows illustrate one of the transition paths from the trivial to the anomalous FNTI, i.e., ①→④→③→⑧, which experiences band inversions across all three gaps. (b) Multifold bulk-edge correspondence interpreted by the phase-band singularities of the time-evolution operator over one driving period: the system's bulk charge ( $q$ ) equals the ordered product of all charges ( $\bar{q}_m$ ) of phase-band singularities, each of which determines a TEM in the corresponding gap. This is exemplified by the anomalous FNTI with $q = \bar{j} \cdot \bar{i} \cdot \bar{k} = +1$ , where the three TEMs result from three phase-band closings. Note that the phase bands at $t = T$ exactly correspond to the system's quasienergy bands.

To date, although Floquet Abelian systems have been implemented across various platforms $[35–44]$ , Floquet non-Abelian topological phenomena have yet to be observed in any experiment. This is primarily due to the substantial challenges in existing experimental platforms: (i) realizing targeted real-time dynamic couplings, and (ii) tracking the complicated, drive-induced multigap interactions and topological responses—both critical for confirming the existence of such non-Abelian Floquet phases. Here, we report the first experimental implementation of the highly elusive FNTIs using our acoustic platform. By adopting an electric circuit-driven dynamic coupling strategy, we successfully realize the anomalous FNTI $q = +1$ that hosts TEMs in all three gaps, alongside a comparative FNTI $q = \pm j$ where the $\pi$ -gap TEM vanishes. Unambiguously, we demonstrate these two FNTIs by characterizing their quasienergy spectra and TEM patterns, through both zero-order and high-order harmonic signals. More intriguingly, we observe a unique interface response at the domain wall between the anomalous FNTI and its counterpart with swapped driving sequences—another hallmark manifestation of non-Abelian dynamics in multi-gap Floquet systems. Collectively, our findings—rooted in the first experimental realization of FNTIs and direct observation of TIMs—close the foundational theory-experiment gap for non-Abelian dynamics in driven multi-gap systems, a breakthrough that advances fundamental understanding of nonequilibrium topological physics.

## II. THEORETICAL MODELS

As depicted in Fig. 2(a), we begin with a simple 1D three-band dynamic model with PT symmetry. Its Floquet-Bloch Hamiltonian in momentum space reads

(a)
![](images/1bc8260dea55fe9b1330454e83356a28fd913e69d14075b37b14bcd3cb36bf07.jpg)

(c)
![](images/3f4b499e76109f658e0986550378ec73d1a0ac95e527c0d15b44ac61552933c3.jpg)

![](images/21f2d67567102885ca34d859db232947426082cae0a0314892205a92921d01bc.jpg)
(d)

![](images/8ebb97b522ee24923637350654dfaa0cebd2474e3b8e467cfe30a960ae3b7c98.jpg)

(e)
(f)
![](images/870e1dac4bdff2344e869c419bdec56c96b0f0382e93874064e6e9cc540fc2a4.jpg)

(g)
![](images/e6aa35b839d52014c0fc455e05026b57152dc03fabe9ec7b3f4809e00bcee28e.jpg)

![](images/b9bf4883328a4fcb7530e941923153bdd0547b0ee16d239c0636f6d2aa8bbcdd.jpg)
FIG. 2. Tight-binding model. (a) Schematic of our 1D three-band Floquet model with PT symmetry. Each unit cell contains three sites (colored spheres) connected by static real couplings (solid rods) and dynamic complex couplings (dashed arrows). (b) PT-symmetric driving protocol. (c) Phase diagram in the parameter space $(s_{12}, s_{13}, T)$ , where red, green, and blue solid lines indicate the closures of the first, second, and third band gaps, respectively. (d) Quaternion charges and TEM configurations of nine representative phases, further concretizing the multifold bulk-edge correspondence. The edge-state configuration shown for phase ⑨ is one representative among several possible configurations. (e) Spectral evolution of a finite-sized lattice along the transition path A → G sketched by the dashed polyline in (c), which resembles the path highlighted by dashed arrows in Fig. 1(a). (f) Quasienergy band structures for phases E and G. (g) Corresponding eigenstate distributions of the finite-sized systems. Notably, the anomalous FNTI (phase G) exhibits TEMs in all band gaps.

$$
H (k _ {x}, t) = \left[ \begin{array}{c c c} h _ {1 1} (k _ {x}) & s _ {1 2} & h _ {1 3} (k _ {x}, t) \\ s _ {1 2} & \omega_ {2} & s _ {2 3} \\ h _ {1 3} ^ {*} (k _ {x}, t) & s _ {2 3} & h _ {3 3} (k _ {x}) \end{array} \right].\tag{1}
$$

More concretely, the static, diagonal matrix elements can be written as $h_{11}(k_x) = \omega_1 + 2v_{11} \cos k_x$ and $h_{33}(k_x) = \omega_3 + 2v_{33} \cos k_x$ , while the dynamic, off-diagonal one takes the form of $h_{13}(k_x, t) = s_{13} + v_{13}(t)e^{-ik_x} + v_{13}^*(t)e^{ik_x}$ . Here, the complex-valued intercell coupling $v_{13}(t)$ is a periodic function of time t. As shown in Fig. 2(b), we adopt an easily implemented, steplike driving scheme for $v_{13}(t)$ : $z_1 = 0.5$ if $t/T \in [0, 1/4] \cup [3/4, 1]$ and $z_2 = 0.5 + 0.5i$ if $t/T \in [1/4, 3/4]$ , with T being the driving period. (Thus, the real part of $v_{13}(t)$ is time invariant, while its imaginary part switches periodically between 0 and 0.5.) The other parameters in $H(k_x, t)$ are real constants: intracell couplings $s_{12} = -0.1$ , $s_{23} = 0.5$ , and $s_{13} = 0.3$ ; intercell couplings $v_{11} = 1$ and $v_{33} = -1$ ; and onsite energies $\omega_{1} = 0.5$ , $\omega_{2} = 0$ , and $\omega_{3} = -0.5$ .

To obtain Floquet quasienergy bands and associated eigenstates of the driven system, we consider the effective Floquet Hamiltonian

$$
H _ {F} = \mathrm{i} \log U (T) / T,\tag{2}
$$

where the Floquet operator $U(T)$ is given by $U(T) = \mathrm{e}^{-\mathrm{i}H_{1}T/4}\mathrm{e}^{-\mathrm{i}H_{2}T/2}\mathrm{e}^{-\mathrm{i}H_{1}T/4}$ , with real-valued, piecewise Hamiltonians $H_{1} = H(t = 0)$ and $H_{2} = H(t = T/2)$ . Solving the eigenvalue problem $H_{F}|u_{n}\rangle = \varepsilon_{n}|u_{n}\rangle$ , we obtain the quasienergies within the first Floquet Brillouin zone (FBZ), $\varepsilon_{n} \in (-\pi/T, \pi/T]$ . Notably, given the symmetric driving protocol in Fig. 2(b), the effective Floquet Hamiltonian $H_{F}$ respects PT symmetry ( $H_{F} = H_{F}^{*}$ ) and

enables real-valued eigenstates $|u_{n}\rangle$ . These eigenstates, in turn, allow for the definition of non-Abelian frame charges that characterize the rotation of the eigenstate frame as the momentum $k_{x}$ varies from $-\pi$ to $\pi$ . The frame charges can be either theoretically calculated by generalized Wilson operators or directly visualized from eigenstate frame spheres [6,19]. As shown in the phase diagram [Fig. 2(c)], our simple Floquet model supports all types of frame charges in the quaternion group and hosts a rich variety of TEM configurations [Fig. 2(d)]. (This minimal model was deliberately crafted to balance physical richness and experimental feasibility—enabling us to meet a critical prerequisite for identifying the highly elusive Floquet non-Abelian topological physics). For conciseness, Figs. 2(c) and 2(d) list the charges for all distinct conjugacy classes without distinguishing the sign differences within $\pm i$ , $\pm j$ , and $\pm k$ . In our detailed computation, the signs of these conjugacy classes can be uniquely determined by fixing a common base point [6,17,19,33]. Further details are provided in Supplemental Material [45] (Figs. S1 and S2). Figure 2(e) shows the spectral evolution of a finite-sized lattice along the dashed polyline path A → G indicated in Fig. 2(c). Starting from the trivial phase A, the three band gaps close and reopen sequentially, during which new TEMs emerge in the corresponding gap. (The edge states, while topologically protected, can be “buried” by the bulk bands for generic system parameters [19].) Accordingly, the bulk quaternion charge evolves from +1 to +k and +j, and finally back to +1. Intriguingly, without any static analog, the final phase G hosts TEMs in all gaps despite a trivial bulk (q = +1). More concretely, Figs. 2(f) and 2(g) showcase comparative quasi-energy bands and finite-system eigenstates for phases E and G. Consistent with the predictions based on phase-band singularities (see Supplemental Material [45], Fig. S3), phase G, the anomalous FNTI, exhibits TEMs in all band gaps, in contrast to phase E, where the π-gap TEM disappears.

![](images/d72e11af22408502ad06be2f1b23c49a74377a824374136f9a2b8b2ce1ae35ef.jpg)
(a)
(c)

## III. ACOUSTIC IMPLEMENTATION OF THREE-BAND FLOQUET LATTICES

Now, we turn to the acoustic realization of the anomalous FNTI. Figure 3(a) shows our experimental setup. It consists of a static acoustic lattice (10 unit cells in total) and active feedback circuits that implement unidirectional, time-periodic acoustic couplings. As displayed in Fig. 3(b), our static sample is realized by an air-filled cavity-tube structure, where the cavities simulate the lattice sites with onsite energies $\omega_{1}=3400$ , $\omega_{2}=3350$ , and $\omega_{3}=3300$ Hz (effectively, $\omega_{1}=50$ and $\omega_{3}=-50$ Hz if shifting $\omega_{2}$ to 0). The narrow tubes mimic the couplings between cavities: $s_{12}=-10$ , $s_{23}=50$ , $s_{13}=30$ , $v_{11}=100$ , $v_{33}=-100$ , and the time-independent $\mathrm{Re}(v_{13})=50\;\mathrm{Hz}$ . The performance of this static lattice has been examined in our acoustic experiments (see Supplemental Material [45], Fig. S4).

![](images/e8d307c23580fc277168ed903634df494655428d6293c133c02ba0ea4044aa77.jpg)

(e)
![](images/ad65d27f0b2e764febf8af2c4dab79a511708d09dfd17d75c4733a7ed67ea08f.jpg)

(b)
![](images/13607d1a83a6d391e3e120d49bce7027cbfb052703ddc39f7e52724562ea5075.jpg)

![](images/a712da2391476b0c06a25fae1b2b0d4a4bfa1a1db944781034b4698bc06eeae9.jpg)

(f)
![](images/82df258c60421670b8378a8492256eb319aec529ba8c7b514fd843156c0e0292.jpg)
FIG. 3. Acoustic emulation of the FNTI model. (a) Experimental setup. The static sample, together with feedback circuits for dynamic couplings, realizes our acoustic Floquet lattice. (b) Geometry structure of the static sample. The air-filled cavities emulate atomic orbitals and the narrow tubes mimic static couplings between them. (c) Circuit realization of a unidirectional, square-wave dynamic coupling between cavities 1 and 3, illustrated by the black arrow. (d) Connections of the active circuits used for realizing the dynamic couplings. Note that the yellow tubes contribute the static couplings $\mathrm{Re}(v_{13}) = \mathrm{Re}(v_{13}^*)$ . (e) Binary cavity-tube structure used to characterize the unidirectional dynamic couplings $\mathrm{Im}(v_{13})$ and $\mathrm{Im}(v_{13}^*)$ , sketched by color arrows like (d). (f) Transmission spectra $|\mathrm{S}_{11}|$ (circles) measured by individually activating the coupling circuits of $\mathrm{Im}(v_{13})$ and $\mathrm{Im}(v_{13}^*)$ . All spectral responses for the 0 and $\pm 1$ orders are plotted against the excitation frequency $f_{\mathrm{in}}$ . The emergence of high-order harmonic signals (amplified by a factor of 20), as predicted by Floquet theory (lines), witnesses the implementation of dynamic couplings.

The unidirectional, steplike dynamic couplings $\mathrm{Im}(v_{13})$ and $\mathrm{Im}(v_{13}^{*})$ are achieved by the external circuits that connect cavities 1 and 3, as sketched in Figs. 3(c) and 3(d). Experimentally, to realize a unidirectional coupling with desired amplitude and phase, the sound signal in cavity 1(3) is picked up by a microphone, modulated by an amplifier and a phase shifter, and then fed back to cavity 3(1) through a loudspeaker [50–55]. Furthermore, a wave generator provides a T-period square-wave voltage to the switcher, ultimately yielding a unidirectional coupling swept between 0 and 50i Hz (or -50i Hz) with a duty cycle of 50% [Fig. 3(c)]. (Here, “i Hz” denotes the unit of imaginary frequency, consistent with ‘i’ used for the tight-binding model.) Together with the time-invariant static coupling $\mathrm{Re}(v_{13})$ , ultimately, we realize the unidirectional, dynamic, complex couplings $v_{13}(t)$ and $v_{13}^{*}(t)$ sketched in Fig. 2(b). To demonstrate our acoustic implementation of dynamic couplings $\mathrm{Im}(v_{13})$ and $\mathrm{Im}(v_{13}^{*})$ , we connect two sets of independent unidirectional coupling circuits to a simple binary cavity-tube structure, with one responsible for $\mathrm{Im}(v_{13})$ and the other for $\mathrm{Im}(v_{13}^{*})$ , as illustrated by the color arrows in Fig. 3(e). Activating one set of circuits while deactivating the other, we measure the transmission response $|S_{11}|$ by exciting cavity 1 and detecting the pressure signal in the same cavity. (The asymmetric double-peak structure for each order results from the synergy between the static detuning and the dynamic nonreciprocal coupling.) Note that under the time-periodic modulation, each single-frequency excitation generates multiple harmonics spaced by 1/T (see Supplemental Material [45], Fig. S5). As an example (with T = 2.4 ms), Fig. 3(f) shows the 0-order and ±1-order harmonic components varying with the excitation frequency $f_{in}$ (rather than with the harmonic frequencies themselves). All experimental data match well with the predictions from temporal coupled-mode theory [56]. It is worth pointing out that the high-order harmonics, originating from the couplings between the high-order Floquet replicas and the 0-order one, are essential for our subsequent experimental characterization of FNTIs.

## IV. BULK AND EDGE RESPONSES OF ANOMALOUS FNTIS

Next, we characterize the quasienergy bands and TEM patterns for phases E and G, which correspond to the scenarios before and after closing $\pi$ -gap. (Note that all acoustic couplings and effective onsite energies are proportional to those used in the tight-binding model.) Conveniently, these two FNTIs can be realized in the same experimental settings, where the phase transition is achieved by simply reconfiguring the driving period T from 1.5 ms to 2.4 ms. To measure the quasienergy spectra, we position a sound source at the middle cell of the sample and scan the sound pressure response across all unit cells. As mentioned above, the presence of high-order harmonic signals serves as a distinctive signature that distinguishes Floquet systems from static systems. To avoid mixing different harmonics, we employ single-frequency excitation and extract the data from the 0-order and $\pm1$ -order harmonics separately. After performing a spatial Fourier transform, we obtain the momentum-space sound energy distribution for each harmonic at given excitation frequency. Finally, sweeping frequency gives the bulk spectra of different harmonics. Furthermore, to characterize the associated TEM configurations, we apply onsite excitation-detection across all cavities in the sample, during which the movable sound source and probe are positioned in the same cavity.

Figures 4(a) and 4(b) present our experimentally measured quasienergy spectra (color scale) for phases E and G, respectively. In both cases, the 0-order harmonic spectra match well the predicted band structures (solid lines). Interestingly, compared to phase E, phase G exhibits a significantly stronger high-order harmonic signal around the $\pi$ -gap. (Note that a much smaller amplification factor is applied to phase G.) This is because the $\pi$ -gap in phase G is caused by band inversion, where neighboring Floquet replicas exhibit strong interactions near the gap. Figures 4(c) and 4(d) display the frequency-dependent sound distributions measured for phases E and G. In both cases, the 0-order data show TEMs inside gaps 1 and 2. The key distinction of phase G (anomalous FNTI) from phase E is the emergence of TEM 3 in the $\pi$ -gap. This is demonstrated more clearly in the high-order results, where phase G exhibits four strong $\pi$ -gap bright spots at both ends of the sample. Overall, our experimental data for quasienergy spectra and TEM patterns align well with coupled-mode theory predictions (see Supplemental Material [45], Figs. S6 and S7).

![](images/c853d6f4f4fdccca67e1022cddf9b4a127dbba69879b8547774e896b89a48b6a.jpg)

![](images/095b5f3c1512e79040d07b03e921012bc8e06640a0d81fc7b6b09856b0f3bced.jpg)

![](images/83c2cb941a9924ba6325e0651da6946b727b16e6ebacae396100deecc32020aa.jpg)

![](images/4a67c8a608fd8dd38e963a331ee00e185cc16b83774bae4dba80d4ea9112391d.jpg)
FIG. 4. Characterizing Floquet quasienergy bands and TEMs. (a),(b) Experimentally measured bulk spectra (color scale) for phases E and G, along with their quasienergy band structures (solid lines) for comparison. The horizontal dashed lines indicate FBZ boundaries. To facilitate comparison with the 0-order harmonic signals, the data superimposed from the ±1-order harmonics are amplified by factors of 1500 for phase E and 150 for phase G. (c),(d) Frequency-resolved sound distributions detected for phases E and G. In contrast to phase E, phase G exhibits additional π-gap TEMs in both the 0-order and ±1-order harmonic signals.

## V. EXOTIC TIMS INDUCED BY SWAPPED DRIVING SEQUENCES

Akin to the multifold bulk-edge correspondence, a domain-wall system formed by two distinct FNTIs follows a multifold bulk-interface correspondence. It is governed by the multiplicative relation $\Delta q = \prod_{m} \Delta \bar{q}_{m}$ , each nontrivial $\Delta \bar{q}_{m}$ enabling topological interface modes (TIMs) inside gap m. As illustrated in Fig. 5(a), here $\Delta q = q^{L}/q^{R}$ and $\Delta \bar{q}_{m} = \bar{q}_{m}^{L}/\bar{q}_{m}^{R}$ respectively characterize charge variations of the quasienergy bands and phase-band singularities between the left and right subsystems. In this work, we are particularly interested in the domain-wall system comprising two FNTIs with swapped driving sequences, which can support exotic TIMs exclusive to Floquet non-Abelian systems. Note that TIMs do not emerge in Floquet Abelian systems [24,25], because the two driving-swapped subsystems must share identical Abelian topological invariants (defined for individual quasienergy band gaps) to match with their edge states [33]. Therefore, the presence of TIMs in the driving-swapped domain-wall systems represents another unique manifestation of the Floquet non-Abelian topology [33].

As depicted in Fig. 5(b), the left and right subsystems follow the driving sequences $H_1 \to H_2 \to H_1$ and $H_2 \to H_1 \to H_2$ within one full period, respectively. Their Floquet operators can be related by a unitary transformation $U^L = V^{-1}U^R V$ , with $V = \mathrm{e}^{-\mathrm{i}H_2T / 4}\mathrm{e}^{-\mathrm{i}H_1T / 4}$ accounting for a $T / 2$ time shift. While this transformation introduces distinct eigenstates, it preserves the bulk spectra and phase-band singularity structures. Notice that for each phase-band singularity, it carries a quaternion charge that is determined up to a sign and belongs to one of the specific conjugacy classes $\bar{q}_1 = \pm \bar{k}, \quad \bar{q}_2 = \pm \bar{i}, \quad \text{and} \quad \bar{q}_3 = \pm \bar{j}$ [6,9]. Consequently, the charges of the left and right subsystems belong to the same conjugacy class, and their ratio, $\Delta \bar{q}_m = \bar{q}_m^L /\bar{q}_m^R$ , must be $\pm 1$ for any given gap $m$ , where the possibility of $\Delta \bar{q}_m = -1$ arises from the distinct global structures of the eigenstates between the two subsystems. [The charge values can be uniquely determined using the fixed-base-point method (see Supplemental Material [45], Fig. S3)]. More concretely, we construct the domain-wall systems $E|E'$ and $G|G'$ based on the existing FNTIs $E$ and $G$ , where the charge variations are outlined in Fig. 5(c) (see Appendix A). As expected, in contrast to the system $E|E'$ that shows no TIM in any gap, the system $G|G'$ displays the emergence of TIM 3 inside the $\pi$ -gap due to the nontrivial charge variation $\Delta \bar{q}_3 = -1$ . To experimentally verify these interface phenomena, we implement the domain-wall systems by simply applying a $T / 2$ time delay between the square-wave voltage signals of the left and right subsystems. Figures 5(d) and 5(e) present the sound distributions measured for these two domain-wall systems. As shown in Fig. 5(d), no TIM signals appear in the domain-wall system $E|E'$ , neither in the 0-order nor the high-order data. In contrast, for the system $G|G'$ in Fig. 5(e), the $\pi$ -gap TIM 3 signal emerges in the 0-order spectrum and is more clearly discernible in the high-order harmonic data, particularly when compared to the single-crystal case in Fig. 4(d). All experimental data closely match the coupled-mode simulations (see Supplemental Material [45], Fig. S8). Essentially, the appearance of the

![](images/496cda54cbcf026e0aa7e4312dd0d37a4ff41e7c6568e7251a0b4f080735f2ad.jpg)
(d)

![](images/3ac83ec7f1936e45a4c9f6ad365392f9e086899db08fd63bf60de450e529cdde.jpg)

![](images/10d2b59c9611c545e8d9297d889a8b25817721abce9783f671b94a4799466fb8.jpg)

(c)
![](images/78eb0160e9a3de299205a7ef98a48cefe3fa71286d483caaf3c06df9eefa0a07.jpg)

(e)
![](images/cad412d27c52af892ee14145b3509973cc297800fb764f8ee3059f6695005442.jpg)
FIG. 5. Observation of counter-intuitive TIMs in domain-wall systems. (a) Multifold bulk-interface correspondence manifested through the multiplicative relation $\Delta q = \prod_{m} \Delta \bar{q}_{m}$ , each nontrivial $\Delta \bar{q}_{m}$ enabling a TIM in gap $m$ . (b) Schematic of the domain-wall system formed by two FNTIs with swapped driving sequences. (c) Eigenfield patterns of the domain-wall systems $E|E'$ and $G|G'$ . The latter exhibits a TIM 3 in the $\pi$ -gap due to the nontrivial $\Delta \bar{q}_{3} = -1$ . (d) Sound distribution characterized for the domain-wall system $E|E'$ . The inset illustrates swapped square-wave sequences in the left and right subsystems. (e) Similar to (d), but for the system $G|G'$ , which clearly shows the emergence of the $\pi$ -gap TIM 3.

TIM reflects the non-commutativity of the two driving sequences, offering another distinctive manifestation of the non-Abelian Floquet dynamics.

## VI. CONCLUSION AND OUTLOOK

This work bridges a critical theory-experiment gap in non-equilibrium topological physics through two key experimental advances in non-Abelian Floquet topology. First, we successfully realized FNTIs—including an “anomalous FNTI” phase—using our acoustic platform, and unambiguously identified their TEMs via zero- and high-order harmonic spectra. Second, we observed counterintuitive TIMs at the domain wall between two FNTIs with swapped driving sequences. These interface modes, as a direct signature of non-commutative driving, are exclusive to non-Abelian Floquet systems and inaccessible in their Abelian or static counterparts. Notably, the harmonic effect—a unique fingerprint of Floquet topological phases—cannot be probed in previous waveguide-based photonic or acoustic setups $[35,37,38,41,43,44]$ , which emulate time with real space and lack access to real-time dynamics. In contrast, our platform captures the full dynamic response and thus offers a solid experimental foundation for exploring more complex non-equilibrium topological phenomena in the future.

Looking forward, the excellent scalability and programmability of our acoustic platform open multiple avenues for further exploration in non-Abelian Floquet physics. On one hand, future work could extend to higher dimensions—such as 2D Floquet Dirac semimetals and 3D nodal-line semimetals—to study driving-induced non-Abelian braiding of band nodes [6,9,15,17,31,57] and momentum-resolved multifold bulk-boundary correspondence (see Appendix B), while further studies may explore four-band systems (governed by the generalized quaternion group $\mathrm{Q}_{16}$ , [20]) to investigate even more complex conjugacy class dynamics and multi-gap topological transitions that are absent in three-band systems. On the other hand, integrating non-Hermiticity [50,51] and/or nonlinearity [58,59] presents a fascinating but completely unexplored frontier for controlling non-Abelian dynamic topological phases. Their synergistic effects, which could profoundly reshape non-equilibrium topological phenomena, may open promising avenues and deserve to be systematically investigated. To complement these explorations, the real-time experimental setup demonstrated here—with exceptional controllability and reconfigurability—can be adapted to explore a broad range of nonequilibrium topological physics, including time-modulated non-Hermitian skin effects and temporal TIMs in momentum gaps [60].

The implications of this work extend beyond non-equilibrium topology, underpinned by shared principles like Floquet engineering and multiband non-Abelian topology. It sets a reference for probing dynamic non-Abelian phases in systems ranging from cold atoms to photonics, phononics, and condensed matter physics, and enables direct tests of scattering theory in the non-Abelian context $[61]$ . In conclusion, by delivering experimental access and systematic methodologies, our work helps foundationalize the emerging subfield of “dynamic non-Abelian topology.” These results are poised to stimulate further progress in both fundamental non-equilibrium physics and the pursuit of topologically protected wave devices.

Note added. Recently, we became aware of Ref. [62], which reports similar work realized in photonic quantum walks.

## ACKNOWLEDGMENTS

We thank Haiping Hu and Robert-Jan Slager for helpful discussions. This project is supported by the National Natural Science Foundation of China (Grants No. 12374418, No. 12404510, and No. 12304495), the National Key R&D Program of China (Grant No. 2023YFA1406900), the Fundamental Research Funds for the Central Universities, the National Postdoctoral Program for Innovative Talents (Grant No. BX20240269), the China Postdoctoral Science Foundation (Grant No. 2024M752454), the Natural Science Foundation of Hubei Province of China (Grant No. 2024AFB064), and the Postdoctoral Project of Hubei Province (Grant No. 2024HBBHCXB056).

## DATA AVAILABILITY

The data that support the findings of this article are not publicly available. The data are available from the authors upon reasonable request.

## APPENDIX A: CHARGE CALCULATIONS IN DOMAIN-WALL SYSTEMS

The domain-wall system follows a multifold bulk-interface correspondence, which is governed by the multiplicative relation $\Delta q = \prod_{m} \Delta \bar{q}_{m}$ , each nontrivial $\Delta \bar{q}_{m}$ enabling a TIM inside gap $m$ . Here $\Delta q = q^{L} / q^{R}$ and $\Delta \bar{q}_{m} = \bar{q}_{m}^{L} / \bar{q}_{m}^{R}$ respectively characterize charge variations of the quasienergy bands and phase-band singularities between the left and right subsystems. Note that $\bar{q}_{m}$ represents the quaternion charge of a phase-band singularity residing within the $m$ -th band gap, as illustrated in the Fig. 1(b). Therefore, the precise TIM patterns are encoded in the phase band pictures. The Floquet operators of the left and right subsystems are $U^{L} = \mathrm{e}^{-\mathrm{i}H_{1}T / 4}\mathrm{e}^{-\mathrm{i}H_{2}T / 2}\mathrm{e}^{-\mathrm{i}H_{1}T / 4}$ and $U^{R} = \mathrm{e}^{-\mathrm{i}H_{2}T / 4}\mathrm{e}^{-\mathrm{i}H_{1}T / 2}\mathrm{e}^{-\mathrm{i}H_{2}T / 4}$ . To derive the relationship between the quaternion charges of the driving-swept subsystems, we analyze the corresponding PT-symmetric time-evolution operators [33],

$$
\tilde {U} ^ {L} (k _ {x}, t) = \left\{ \begin{array}{l l} \mathrm{e} ^ {- \mathrm{i} H _ {1} t}, & t \in \left[ 0, \frac {T}{2} \right], \\ \mathrm{e} ^ {- \frac {\mathrm{i} H _ {1} T}{4}} \mathrm{e} ^ {- \mathrm{i} H _ {2} (t - \frac {T}{2})} \mathrm{e} ^ {- \frac {\mathrm{i} H _ {1} T}{4}}, & t \in \left[ \frac {T}{2}, T \right], \end{array} \right.\tag{A1}
$$

and

$$
\tilde {U} ^ {R} (k _ {x}, t) = \left\{ \begin{array}{l l} \mathrm{e} ^ {- \mathrm{i} H _ {1} t}, & t \in \left[ 0, \frac {T}{2} \right], \\ \mathrm{e} ^ {- \frac {\mathrm{i} H _ {2} (t - \frac {T}{2})}{2}} \mathrm{e} ^ {- \frac {\mathrm{i} H _ {1} T}{2}} \mathrm{e} ^ {- \frac {\mathrm{i} H _ {2} (t - \frac {T}{2})}{2}}, & t \in \left[ \frac {T}{2}, T \right]. \end{array} \right.\tag{A2}
$$

The phases of their eigenvalues define the phase bands $\phi_{n}^{L}(k_{x},t)$ and $\phi_{n}^{R}(k_{x},t)$ . By examining Eqs. (A1) and (A2), one can establish an SO(3) transformation relating $\tilde{U}^L (k_x,t)$ and $\tilde{U}^R (k_x,t)$ at any moment $t$ . This connection enforces identical phase bands, $\phi_n^L (k_x,t) = \phi_n^R (k_x,t) = \phi_n(k_x,t)$ , and thus identical phase singularity structures for the two subsystems. Notably, the topological charge of each phase-band singularity is determined up to a sign and belongs to one of the three conjugate classes: $\bar{q}_1 = \pm \bar{k}$ , $\bar{q}_2 = \pm \bar{i}$ , and $\bar{q}_3 = \pm \bar{j}$ . (This framework directly generalizes the description of band nodes in static systems to the phase-band singularities in our Floquet setting.) Therefore, the charges $\bar{q}_m^L$ and $\bar{q}_m^R$ of the left and right subsystems belong to the same conjugacy class, and their ratio, $\Delta \bar{q}_m = \bar{q}_m^L /\bar{q}_m^R$ , must be $\pm 1$ for any given gap $m$ , where the possible case of $\Delta \bar{q}_m = -1$ arises from the distinct global structures of the eigenstates between the two subsystems. Note that the charge values were uniquely determined using the fixed-base-point method [see Supplementary Material [45], Figs. S3(c) and S3(g)] to resolve the sign ambiguity. As examples, Fig. 6 demonstrates the charge variations calculated for the domain-wall systems $E|E'$ and $G|G'$ . In the case of $E|\mathrm{E}'$ , the corresponding quaternion charge distributions give the trivial variations $\Delta \bar{q}_1 = \Delta \bar{q}_2 = +1$ , which explain why no TIM is observed at the domain-wall $E|E'$ [Figs. 6(a) and 6(b)]; in contrast, for the domain-wall system $G|G'$ , the nontrivial charge variation $\Delta \bar{q}_3 = -1$ accounts for the emergence of TIM 3 within the $\pi$ -gap [Figs. 6(c) and 6(d)].

![](images/48d2beff08e57683cee8f4098395f9a1fc797e57eeb6f5023198de0f3384fdea.jpg)

![](images/16833a2475287eb0695bb2643a3db64b41a44ee055f2697a1de93b2fcac54894.jpg)

![](images/1d0dd057fc3f6597a728abcd47fa0c416df813cfd16b40982e053c60b9b4ee9b.jpg)

![](images/e66042d740ca5087e1f6c4b6ef283a679abcd876c0ff0615750c05b1011a9c79.jpg)
FIG. 6. Charge variations calculated for the domain-wall systems $E|E'$ and $G|G'$ . (a) Quaternion charges of quasienergy bands calculated for phase E (left subsystem) and its counterpart with swapped driving sequence, $E'$ (right subsystem). This gives a trivial charge variation $\Delta q = +1$ . (b) Corresponding quaternion charges for the two phase-band singularities, which give the trivial variations $\Delta \bar{q}_1 = \Delta \bar{q}_2 = +1$ . Therefore, no TIM occurs at the domain-wall, as sketched in (a). (c),(d) Similar to (a),(b), but for the domain-wall system $G|G'$ . The nontrivial charge variation $\Delta \bar{q}_3 = -1$ explains the emergence of TIM 3 in the $\pi$ -gap.

## APPENDIX B: 2D GENERALIZATION

The 2D system provides a valuable playground for observing rich Floquet non-Abelian topological transitions and the consequent topological physics. As illustrated in Fig. 7(a), we construct a 2D model by stacking the 1D layers along the z-direction with a staggered interlayer coupling $v_{z}$ . The resulting 2D Hamiltonian closely resembles that of the 1D case, except that the original coupling $S_{13}$ is replaced by an effective $k_{z}$ -dependent term $S_{13}^{\mathrm{eff}}(k_{z}) = S_{13} + 2v_{z}\cos k_{z}$ . Therefore, each constant- $k_{z}$ subsystem behaves as a 1D system, and by tuning $k_{z}$ , one can drive non-Abelian topological transitions between distinct Floquet phases, e.g., ③ and ⑧ in Fig. 2(c). The arrows illustrate the transition path as $k_{z}$ varies from $-\pi$ to $+\pi$ , which traverse twice the transition point F with $k_{z} = \pm\pi/2$ . In particular, at $k_{z} = \pm\pi$ , the system resides in phase ③, characterized by a charge of +j and hosting edge states in gaps 1 and 2, while at $k_{z} = 0$ it transitions into phase ⑧, which carries a charge of +1 and supports nontrivial edge states in all three gaps. Figures 7(b)–7(d) present the band structures at three typical $k_{z}$ slices, which confirm the topological transition at $k_{z} = \pm0.5\pi$ with $\pi$ -gap closure. Exactly, the transition phase suggests the emergence of Dirac nodes in the 2D Brillouin zone. Alongside the evolution of more system parameters, one can further study the Floquet non-Abelian braiding of Dirac nodes in 2D momentum space and anomalous Euler (Dirac string) phases induced by non-Abelian Floquet braiding.

(a)
![](images/52e03ff5b4ce408bf0f59a76f86b0e89a70cf0768be4bc8a5e5bbefbb6b4afb3.jpg)

(b)
![](images/46d83b0b2f9ef904d9d7f8640e72c51aa4c0b544bc6aaa5330f29b4f7d262e70.jpg)

(e)
![](images/0fb26e6b9577fa172f4c6a75b535013019011306e7ce3277a0ad2d09d6144d8c.jpg)

(f)
(c)
![](images/95fdd1209a3ed08d801b91dfebebf80193a428b29f16d955da753ce6f773361f.jpg)

![](images/4ce1ab77c77943f694ed755eb94ac1216a39765dae3f68e2433c01b911dd0fb7.jpg)

(d)
![](images/87407ff615df889855d7aac14ca3da75083b82717be714ab0a264edc93730980.jpg)
(h)

(g)
![](images/c86d541e1f7e935a4e941ff3faf1238ddd3021c7cc677c51d3e793cef0f70145.jpg)

![](images/fd2c16a2255d5f174b0cc0db8db52f64ff5c04e6cd61c63305bb59824735ef4e.jpg)
FIG. 7. Topological physics in 2D FNTIs. (a) Schematic of the 2D model, constructed by stacking 1D chains along the z-axis with interlayer couplings $v_{z}=0.125$ . Each constant- $k_{z}$ subsystem serves as a 1D FNTI, and the $k_{z}$ -dependent effective coupling drives a non-Abelian topological transition between phases ③ and ⑧, where F represents the phase transition point indicated in Fig. 2(c) $T=0.175$ . (b)–(d) Quasienergy spectra computed for three typical $k_{z}$ values, demonstrating a phase transition of topological charge from $q=+j$ to $q=+1$ . (e) $k_{z}$ -resolved edge spectrum. (f) $k_{z}$ -resolved quasienergy spectrum for the domain-wall system in (d). (g), (h) Corresponding field distributions for $k_{z}=\pm\pi$ and $k_{z}=0$ , demonstrating the absence and presence of TIMs in the third gap.

We further examine the $k_{z}$ -resolved multifold bulk-edge and bulk-interface correspondences. Figures 7(e) and 7(f) demonstrate the quasienergy spectra for the finite-sized edge and domain-wall systems, respectively, as functions of $k_{z}$ . The data consistently reflect the above topological transition. It shows that within $|k_{z}| \in [\pi/2, \pi]$ , TEMs appear only in gaps 1 and 2, whereas within $|k_{z}| \in [0, \pi/2]$ , TEMs emerge in all three gaps. The physics is consistent with their topological charges of phase-band singularities: the former corresponds to two phase-band singularities $\bar{q}_{1}$ and $\bar{q}_{2}$ within $|k_{z}| \in [\pi/2, \pi]$ , while the latter corresponds to three singularities $\bar{q}_{1}, \bar{q}_{2}$ and $\bar{q}_{3}$ . More details are omitted here. In the domain-wall system formed by two 2D phases with swapped driving sequences, TIMs are observed exclusively in the third gap within $|k_{z}| \in [0, \pi/2]$ . In Figs. 7(g) and 7(h), we check the eigenfield distributions at the slices $k_{z} = \pm\pi$ and 0. For the former, $\Delta\bar{q}_{1,2} = +1$ imply the absence of TIM; while for the latter, $\Delta\bar{q}_{1,2} = +1$ and $\Delta\bar{q}_{3} = -1$ indicate the presence of TIM in the third gap.

[1] B. A. Bernevig, T. L. Hughes, and S. C. Zhang, Quantum spin Hall effect and topological phase transition in HgTe quantum wells, Science 314, 1757 (2006).

[2] M. König, S. Wiedmann, C. Brüne, A. Roth, H. Buhmann, L. W. Molenkamp, X.-L. Qi, and S.-C. Zhang, Quantum spin Hall insulator state in HgTe quantum wells, Science 318, 766 (2007).

[3] M. Z. Hasan and C. L. Kane, Colloquium: Topological insulators, Rev. Mod. Phys. 82, 3045 (2010).

[4] J. E. Moore, The birth of topological insulators, Nature (London) 464, 194 (2010).

[5] X.-L. Qi and S.-C. Zhang, Topological insulators and superconductors, Rev. Mod. Phys. 83, 1057 (2011).

[6] Q. Wu, A. A. Soluyanov, and T. Bzdusek, Non-Abelian band topology in noninteracting metals, Science 365, 1273 (2019).

[7] J. Ahn, S. Park, and B.-J. Yang, Failure of Nielsen-Ninomiya Theorem and fragile topology in two-dimensional systems with space-time inversion symmetry: Application to twisted bilayer graphene at magic angle, Phys. Rev. X 9, 021013 (2019).

[8] A. Tiwari and T. Bzdušek, Non-Abelian topology of nodal-line rings in PT-symmetric systems, Phys. Rev. B 101, 195130 (2020).

[9] A. Bouhon, Q. Wu, R.-J. Slager, H. Weng, O. V. Yazyev, and T. Bzdušek, Non-Abelian reciprocal braiding of Weyl points and its manifestation in ZrTe, Nat. Phys. 16, 1137 (2020).

[10] Y. Yang, B. Yang, G. Ma, J. Li, S. Zhang, and C. T. Chan, Non-Abelian physics in light and sound, Science 383, eadf9621 (2024).

[11] E. Yang, B. Yang, O. You, H. C. Chan, P. Mao, Q. Guo, S. Ma, L. Xia, D. Fan, Y. Xiang, and S. Zhang, Observation of

non-Abelian nodal links in photonics, Phys. Rev. Lett. 125, 033901 (2020).

[12] D. Wang, B. Yang, M. Wang, R. Y. Zhang, X. Li, Z. Q. Zhang, S. Zhang, and C. T. Chan, Observation of non-Abelian charged nodes linking nonadjacent gaps, Phys. Rev. Lett. 129, 263604 (2022).

[13] D. Wang, Y. Wu, Z. Q. Zhang, and C. T. Chan, Non-Abelian frame charge flow in photonic media, Phys. Rev. X 13, 021024 (2023).

[14] Y. Hu, M. Tong, T. Jiang, J. H. Jiang, H. Chen, and Y. Yang, Observation of two-dimensional time-reversal broken non-Abelian topological states, Nat. Commun. 15, 10036 (2024).

[15] B. Jiang, A. Bouhon, Z.-K. Lin, X. Zhou, B. Hou, F. Li, R.-J. Slager, and J.-H. Jiang, Experimental observation of non-Abelian topological acoustic semimetals and their phase transitions, Nat. Phys. 17, 1239 (2021).

[16] M. Wang, S. Liu, Q. Ma, R.-Y. Zhang, D. Wang, Q. Guo, B. Yang, M. Ke, Z. Liu, and C. T. Chan, Experimental observation of non-Abelian earring nodal links in phononic crystals, Phys. Rev. Lett. 128, 246601 (2022).

[17] H. Qiu, Q. Zhang, T. Liu, X. Fan, F. Zhang, and C. Qiu, Minimal non-Abelian nodal braiding in ideal metamaterials, Nat. Commun. 14, 1261 (2023).

[18] X.-C. Sun, J.-B. Wang, C. He, and Y.-F. Chen, Non-Abelian topological phases and their quotient relations in acoustic systems, Phys. Rev. Lett. 132, 216602 (2024).

[19] Q. Guo, T. Jiang, R. Y. Zhang, L. Zhang, Z. Q. Zhang, B. Yang, S. Zhang, and C. T. Chan, Experimental observation of non-Abelian topological charges and edge states, Nature (London) 594, 195 (2021).

[20] T. Jiang, Q. Guo, R. Y. Zhang, Z. Q. Zhang, B. Yang, and C. T. Chan, Four-band non-Abelian topological insulator and its experimental realization, Nat. Commun. 12, 6471 (2021).

[21] T. Kitagawa, E. Berg, M. Rudner, and E. Demler, Topological characterization of periodically driven quantum systems, Phys. Rev. B 82, 235114 (2010).

[22] N. H. Lindner, G. Refael, and V. Galitski, Floquet topological insulator in semiconductor quantum wells, Nat. Phys. 7, 490 (2011).

[23] R. Fleury, A. B. Khanikaev, and A. Alù, Floquet topological insulators for sound, Nat. Commun. 7, 11744 (2016).

[24] S. Yao, Z. Yan, and Z. Wang, Topological invariants of Floquet systems: General formulation, special properties, and Floquet topological defects, Phys. Rev. B 96, 195303 (2017).

[25] R. Roy and F. Harper, Periodic table for Floquet topological insulators, Phys. Rev. B 96, 155118 (2017).

[26] H. Hübener, M. A. Sentef, U. De Giovannini, A. F. Kemper, and A. Rubio, Creating stable Floquet–Weyl semimetals by laser-driving of 3D Dirac materials, Nat. Commun. 8, 1 (2017).

[27] A. Eckardt, Colloquium: Atomic quantum gases in periodically driven optical lattices, Rev. Mod. Phys. 89, 011004 (2017).

[28] M. S. Rudner and N. H. Lindner, Band structure engineering and non-equilibrium dynamics in Floquet topological insulators, Nat. Rev. Phys. 2, 229 (2020).

[29] C. Bao, P. Tang, D. Sun, and S. Zhou, Light-induced emergent phenomena in 2D materials and topological materials, Nat. Rev. Phys. 4, 33 (2021).

[30] M. S. Rudner, N. H. Lindner, E. Berg, and M. Levin, Anomalous edge states and the bulk-edge correspondence for periodically driven two-dimensional systems, Phys. Rev. X 3, 031005 (2013).

[31] R.J. Slager, A. Bouhon, and F.N. Unal, Non-Abelian Floquet braiding and anomalous Dirac string phase in periodically driven systems, Nat. Commun. 15, 1144 (2024).

[32] V. Karle, M. Lemeshko, A. Bouhon, R. J. Slager, and F. N. Unal, Anomalous multi-gap topological phases in periodically driven quantum rotors, Phys. Rev. A 113, 012216 (2026).

[33] T. Li and H. Hu, Floquet non-Abelian topological insulator and multifold bulk-edge correspondence, Nat. Commun. 14, 6418 (2023).

[34] F. Nathan and M. S. Rudner, Topological singularities and the general classification of Floquet–Bloch systems, New J. Phys. 17, 125014 (2015).

[35] M. C. Rechtsman, J. M. Zeuner, Y. Plotnik, Y. Lumer, D. Podolsky, F. Dreisow, S. Nolte, M. Segev, and A. Szameit, Photonic Floquet topological insulators, Nature (London) 496, 196 (2013).

[36] Y.-G. Peng, C.-Z. Qin, D.-G. Zhao, Y.-X. Shen, X.-Y. Xu, M. Bao, H. Jia, and X.-F. Zhu, Experimental demonstration of anomalous Floquet topological insulator for sound, Nat. Commun. 7, 13368 (2016).

[37] L. J. Maczewsky, J. M. Zeuner, S. Nolte, and A. Szameit, Observation of photonic anomalous Floquet topological insulators, Nat. Commun. 8, 13756 (2017).

[38] S. Mukherjee, A. Spracklen, M. Valiente, E. Andersson, P. Ohberg, N. Goldman, and R. R. Thomson, Experimental observation of anomalous topological edge modes in a slowly driven photonic lattice, Nat. Commun. 8, 13918 (2017).

[39] S. Stutzer, Y. Plotnik, Y. Lumer, P. Titum, N. H. Lindner, M. Segev, M. C. Rechtsman, and A. Szameit, Photonic topological Anderson insulators, Nature (London) 560, 461 (2018).

[40] K. Wintersperger, C. Braun, F. N. Ünal, A. Eckardt, M. D. Liberto, N. Goldman, I. Bloch, and M. Aidelsburger, Realization of an anomalous Floquet topological system with ultracold atoms, Nat. Phys. 16, 1058 (2020).

[41] S. Mukherjee and M. C. Rechtsman, Observation of Floquet solitons in a topological bandgap, Science 368, 856 (2020).

[42] G. G. Pyrialakos, J. Beck, M. Heinrich, L. J. Maczewsky, N. V. Kantartzis, M. Khajavikhan, A. Szameit, and D. N. Christodoulides, Bimorphic Floquet topological insulators, Nat. Mater. 21, 634 (2022).

[43] Z. Cheng, R. W. Bomantara, H. Xue, W. Zhu, J. Gong, and B. Zhang, Observation of $\pi/2$ modes in an acoustic Floquet system, Phys. Rev. Lett. 129, 254301 (2022).

[44] W. Zhu, H. Xue, J. Gong, Y. Chong, and B. Zhang, Time-periodic corner states from Floquet higher-order topology, Nat. Commun. 13, 11 (2022).

[45] See Supplemental Material at http://link.aps.org/supplemental/10.1103/qn87-bm33 for further theoretical and experimental details, which includes Refs. [46–49].

[46] T. Li, J. Du, Q. Zhang, Y. Li, X. Fan, F. Zhang, and C. Qiu, Acoustic Möbius insulators from projective symmetry, Phys. Rev. Lett. 128, 116803 (2022).

[47] T. Li, L. Liu, Q. Zhang, and C. Qiu, Acoustic realization of projective mirror Chern insulators, Commun. Phys. 6, 268 (2023).

[48] Z. K. Lin, Y. Zhou, B. Jiang, B. Q. Wu, L. M. Chen, X. Y. Liu, L. W. Wang, P. Ye, and J. H. Jiang, Measuring entanglement entropy and its topological signature for phononic systems, Nat. Commun. 15, 1601 (2024).

[49] B. Jiang, A. Bouhon, S.-Q. Wu, Z.-L. Kong, Z.-K. Lin, R.-J. Slager, and J.-H. Jiang, Observation of an acoustic topological Euler insulator with meronic waves, Sci. Bull. 69, 1653 (2024).

[50] L. Zhang, Y. Yang, Y. Ge, Y. Guan, Q. Chen, Q. Yan, F. Chen, R. Xi, Y. Li, D. Jia, S. Yuan, H. Sun, H. Chen, and B. Zhang, Acoustic non-Hermitian skin effect from twisted winding topology, Nat. Commun. 12, 6297 (2021).

[51] Q. Zhang, Y. Li, H. Sun, X. Liu, L. Zhao, X. Feng, X. Fan, and C. Qiu, Observation of acoustic non-Hermitian bloch braids and associated topological phase transitions, Phys. Rev. Lett. 130, 017201 (2023).

[52] Z.-X. Chen, A. Chen, Y.-G. Peng, Z. W. Li, B. Liang, J. Yang, X.-F. Zhu, Y. Q. Lu, and J. C. Cheng, Observation of acoustic Floquet $\pi$ modes in a time-varying lattice, Phys. Rev. B 109, L020302 (2024).

[53] S. Tong, Q. Zhang, L. Qi, G. Li, X. Feng, and C. Qiu, Observation of Floquet-bloch braids in non-Hermitian

spatiotemporal lattices, Phys. Rev. Lett. 134, 126603 (2025).

[54] S. Tong, Q. Zhang, G. Li, K. Zhang, C. Xie, and C. Qiu, Observation of momentum-band topology in PT-symmetric Floquet lattices, Nat. Commun. 16, 9975 (2025).

[55] S. Tong, Q. Zhang, G. Li, K. Zhang, and C. Qiu, Acoustic realization of monoatomic topological space-time crystals, Newton 2, 2100304 (2026).

[56] T. T. Koutserimpas and R. Fleury, Nonreciprocal gain in non-Hermitian time-Floquet systems, Phys. Rev. Lett. 120, 087401 (2018).

[57] F. N. Ünal, A. Bouhon, and R.-J. Slager, Topological Euler class as a dynamical observable in optical lattices, Phys. Rev. Lett. 125, 053601 (2020).

[58] K. Zhang, Q. Zhang, S. Tong, W. Wu, X. Feng, and C. Qiu, Experimental observation of hidden multistability in nonlinear systems, Phys. Rev. Lett. 136, 037201 (2026).

[59] Z.-X. Chen, Y. Ru, G.-C. He, M.-H. Lu, Y.-F. Chen, Y.-Q. Lu, and Z.-G. Chen, Nonlinear coupling induced anomalous state transfer and complete multistate excitation via adiabatic control, Phys. Rev. Lett. 136, 037202 (2026).

[60] E. Lustig, Y. Sharabi, and M. Segev, Topological aspects of photonic time crystals, Optica 5, 1390 (2018).

[61] S. Franca, F. Hassler, and I. C. Fulga, Simulating Floquet topological phases in static systems, SciPost Phys. Core 4, 007 (2021).

[62] Q. Lin, T. Li, H. Hu, W. Yi, and P. Xue, Simulating Floquet non-Abelian topological insulator with photonic quantum walks arXiv:2508.06466.
