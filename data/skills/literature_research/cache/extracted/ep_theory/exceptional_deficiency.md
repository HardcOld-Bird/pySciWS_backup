# Exceptional deficiency of non-Hermitian systems: high-dimensional coalescence and dynamics

Zhen Li $^{1}$ , Xulong Wang $^{1}$ , Rundong Cai $^{1}$ , Kenji Shimomura $^{2}$ ,

Zhesen Yang $^{3*}$ , Masatoshi Sato $^{2*}$ , Guancong Ma $^{1,4*}$

$^{1}$ Department of Physics, Hong Kong Baptist University, Kowloon Tong, Hong Kong, China.

$^{2}$ Center for Gravitational Physics and Quantum Information, Yukawa Institute for Theoretical Physics,

Kyoto University, Kyoto 606-8502, Japan.

$^{3}$ Department of Physics, Xiamen University, Xiamen 361005, Fujian, China.

$^{4}$ Shenzhen Institute for Research and Continuing Education, Hong Kong Baptist University, Shenzhen 518000, China.

## Abstract

Exceptional points (EPs) are non-Hermitian singularities associated with the coalescence of individual eigenvectors accompanied by the degeneracy of their complex energies. Here, we report the discovery of a generalization to the concept of EP called exceptional deficiency (ED), which features the complete coalescence of two eigenspaces with identical but arbitrarily large dimensions and the coincidence of entire spectral continua. The characteristics of the ED are studied using one-way coupled Hermitian and non-Hermitian lattices. The ED can induce an anomalous absence and presence of non-Hermitian skin effect (NHSE) that transcends the topological bulk-edge correspondence of NHSE, resulting in unexpected synergistic skin-propagative dynamics. The conditions of the ED are also explored for unprecedented control of localization and propagation in non-Hermitian systems. These effects are experimentally observed using active mechanical lattices. The discovery of ED opens multiple new frontiers in non-Hermitian physics and can potentially resolve long-standing challenges in related applications.

Exceptional points (EPs) are unique singularities in the complex spectrum of non-Hermitian systems $[1,2]$ , which are open systems governed by both internal degrees of freedom and energy (or particle) exchange with the external. At an EP, the eigenvectors of two states (or more, in cases of higher-order EPs) coalesce, accompanied by the degeneracy of the corresponding eigenvalues. The Hilbert space resultantly “loses” a dimension and becomes defective $[3,4]$ , leading to a Jordan canonical form of the Hamiltonian. The investigations of EPs have led to a multitude of revolutionary physics $[5–8]$ and diverse novel phenomena with rich application potentials $[9–12]$ . Yet, because EP formations only involve an $\mathcal{O}(1)$ number of states isolated in the spectrum, related phenomena and functionalities are intrinsically limited to a very narrow bandwidth and require precise control to access.

In this work, we generalize the concept of EP by considering large non-Hermitian Hamiltonians with a block-triangular form. We discover that two high-dimensional eigenspaces, instead of individual eigenvectors, can completely align. (In this work, “high-dimensional” refers to the dimension of the eigenspace instead of spatial dimensions.) We denote this as “exceptional deficiency (ED).” At ED, the spectra of two coalescing eigenspaces coincide, and the Hilbert space is $\mathcal{O}(N)$ defective, with defective states emerging over a spectral continuum instead of at isolated spectral points. The properties of the ED enable novel non-Hermitian dynamics in systems consisting of two large eigenspaces of equal dimensions, one is Hermitian, the other is non-Hermitian. The latter is realized as a lattice under non-Hermitian skin effect (NHSE), a localization mechanism that turns states in continuum bands to skin modes clinging to an open boundary [13–16]. As a crucial consequence of the ED, the celebrated topological bulk-edge correspondence of NHSE [17] is broken, leading to unprecedented non-Hermitian dynamic effects characterized by the synergy of NHSE and propagation. The condition of ED also unveils a universal and convenient route for the reliable and flexible control of localization and propagation in non-Hermitian systems. Our work heralds the advent of a new era for non-Hermitian physics across many realms.

We begin with a simple two-level Hamiltonian, $\mathbf{H}=\begin{pmatrix}h_{1}&\kappa\\0&h_{2}\end{pmatrix}$ , which is non-Hermitian when $\kappa\neq0$ . Figure 1(a, b) plots its eigenvalues and eigenvectors as functions of $\Delta h=|h_{1}-h_{2}|$ . When $\Delta h$ reduces to zero, the eigenvalues become identical, and the eigenvectors are increasingly skewed and eventually aligned. At this point, H is a Jordan block, and an EP is reached [3,18].

Consider a generalization that replaces all entries in H with $N \times N$ square matrices:

$$
\mathbf {H} = \left( \begin{array}{c c} \mathbf {h} _ {1} & \boldsymbol {\kappa} \\ \mathbf {0} & \mathbf {h} _ {2} \end{array} \right).\tag{1}
$$

In the simple case where $\kappa=0$ and $h_{1,2}$ are Hermitian, $H=h_{1}\oplus h_{2}$ is spanned by the eigenvectors of $h_{1}$ and $h_{2}$ , which form a complete set of orthonormal bases, as graphically depicted in Fig. 1(c). However, when $\kappa\neq0$ and $h_{1}=h_{2}$ , $\mathcal{E}(h_{2})$ coalesces with $\mathcal{E}(h_{1})$ , where $\mathcal{E}(\cdot)$ denotes eigenspace: H is $\mathcal{O}(N)$ defective and half of the span of the Hilbert space is “missing.” We denote this situation as the ED. (Here, $\mathcal{E}(h_{1,2})$ is formed by the eigenvectors of H with eigenvalues in the spectra $\eta(h_{1,2})$ of $h_{1,2}$ . It does not refer to the original Hilbert space of $h_{1,2}$ .)

The ED hinges on the block-triangular form of H, which has an invariant subspace [19]. It can be proved that the necessary condition of the ED is $\eta(\mathbf{h}_{1}) = \eta(\mathbf{h}_{2})$ [20]. In other words, rather surprisingly, ED can appear even when $h_{1}$ and $h_{2}$ are different matrices. An intriguing example is shown in Fig. 1(d), where $h_{1}$ is Hermitian and $h_{2}$ is non-Hermitian. The skewed $\mathcal{E}(\mathbf{h}_{2})$ coalesces with the orthogonal $\mathcal{E}(\mathbf{h}_{1})$ . Remarkably, the highly defective Hilbert space at the ED remains an orthogonal linear space (Fig. 1(d)), which is in stark contrast with conventional EPs. On the contrary, if $h_{1}$ and $h_{2}$ are exchanged, the Hilbert space is skewed at the ED (Fig. 1(e)). The skewness is not controlled by $\kappa$ , it is a characteristic of the non-Hermitian block. These interesting characteristics are highly consequential in the following systems.

Now we investigate the exotic physical phenomena induced by the ED. Two examples, which are one-dimensional double-chain lattices, are shown in Fig. 2(a). In system-I, two Su-Schrieffer-Heeger chains, one Hermitian (denoted chain-A) and the other non-Hermitian with asymmetric intra-cell hopping (denoted chain-B), are one-way coupled by hopping $\kappa_{1}$ that hops only upwards (from chain-B to A). In system-II, everything is the same except the one-way hopping $\kappa_{2}$ hops downwards (from chain-A to B). The Bloch Hamiltonians of the two systems are both block-triangular matrices

$$
\mathbf {H} _ {\mathrm{I}} (k) = \left( \begin{array}{c c} \mathbf {H} _ {A} (k) & \kappa_ {1} \mathbf {I} _ {2} \\ \mathbf {0} & \mathbf {H} _ {B} (k) \end{array} \right), \mathbf {H} _ {\mathrm{II}} (k) = \left( \begin{array}{c c} \mathbf {H} _ {A} (k) & \mathbf {0} \\ \kappa_ {2} \mathbf {I} _ {2} & \mathbf {H} _ {B} (k) \end{array} \right).\tag{2}
$$

Here, $\mathbf{H}_{A}(k)=d_{Ax}\boldsymbol{\sigma}_{x}+d_{Ay}\boldsymbol{\sigma}_{y}$ , with $d_{Ax}=v_{1}+w\cos k$ , $d_{Ay}=w\sin k$ , where $\sigma_{x,y}$ are the Pauli matrices, $I_{2}$ is a $2\times2$ identity matrix; and $\mathbf{H}_{B}(k)=d_{Bx}\boldsymbol{\sigma}_{x}+d_{By}\boldsymbol{\sigma}_{y}$ , with $d_{Bx}=v_{2}+w\cos k$ , $d_{By}=w\sin k+i\delta$ . These two seemingly unassuming lattices have some rather surprising characteristics. First, since the characteristic equations of $H_{I,II}$ can reduce to $H_{A}$ and $H_{B}$ , the systems' spectra under a periodic boundary condition (PBC) are $\eta(\mathbf{H}_{\mathrm{I},\mathrm{II}})=\eta(\mathbf{H}_{A})\cup$ $\eta(\mathbf{H}_{B})$ , which are independent of the unidirectional inter-chain hopping $\kappa_{1,2}$ . Therefore, the two systems have identical PBC spectra, which is the union of two distinct sub-spectra: a pair of loop spectra with identical nontrivial winding and a pair of real-valued spectra (Fig. 2(b)). And because $\eta(\mathbf{H}_{A}) \neq \eta(\mathbf{H}_{B})$ , ED does not appear in the PBC systems.

Let us now examine the open boundary condition (OBC) Hamiltonians, denoted $h_{I}$ and $h_{II}$ , which also have $\eta(\mathbf{h}_{\mathrm{I,II}})=\eta(\mathbf{h}_{A})\cup\eta(\mathbf{h}_{B})$ . Here the OBC spectra of both chains, $\eta(\mathbf{h}_{A})$ and $\eta(\mathbf{h}_{B})$ , are real. When $v_{1}=v_{e}=-\sqrt{(v_{2}-\delta)(v_{2}+\delta)}$ , $\eta(\mathbf{h}_{A})=\eta(\mathbf{h}_{B})$ (Fig. 2(c)) and OBC Hamiltonians reach ED. In system-I, $\mathcal{E}(\mathbf{h}_{B})$ becomes defective. Examining the OBC eigenstates, we see that not a single state is skin mode – all states are fully extended (Fig. 2(e)). For the same reason, $\mathcal{E}(\mathbf{h}_{A})$ is defective in system-II, and the OBC eigenstates thereof are entirely skin modes (Fig. 2(h)). These situations clearly deviate from the prevailing theories about the correspondence between NHSE and nontrivial PBC spectral topology [15,16,21] since system-I and II have the same PBC and OBC spectra but totally different localization properties.

To gain more insights, we analyze the systems using non-Bloch band theories [22,16]. Note that the characteristic polynomials $f_{\mathrm{I,II}}(\beta, E) = \det \left[\mathbf{H}_{\mathrm{I,II}}(\beta) - E\mathbf{I}_4\right] = 0$ have no dependence on $\kappa_{1,2}$ , and is exactly factorizable as $f_{\mathrm{I,II}}(\beta, E) = f_A(\beta, E)f_B(\beta, E)$ , where $\beta := e^{i(k + ik')}$ . Inserting the identical OBC spectra of the two systems, i.e., $E = \eta (\mathbf{h}_{\mathrm{I}}) = \eta (\mathbf{h}_{\mathrm{II}})$ , into the characteristic polynomials, we can obtain the solution consists of two parts: one is a unit circle $\beta_A = e^{ik}$ representing Bloch waves associated with the Hermitian chain-A, the other is the non-Bloch waves given by $\beta_B = \sqrt{(v_2 - \delta) / (v_2 + \delta)}e^{ik}$ , which only relates to the non-Hermitian chain-B (Fig. 2(d)). From the OBC eigenstates shown in Fig. 2(e, h), it is clear that system-I (II) is dominated by sub-GBZ $\beta_A(\beta_B)$ , which corresponds to Bloch (non-Bloch) waves. This anomalous effect is rooted in the unique property of the ED, at which an entire eigenspace spanned by non-Bloch (Bloch) waves becomes defective.

We use the cosine similarity to quantify the behavior of the coalescing eigenspaces

$$
\mathcal {C} \big (\mathcal {E} (\mathbf {h} _ {A}), \mathcal {E} (\mathbf {h} _ {B}) \big) = \frac {\langle \mathcal {E} (\mathbf {h} _ {A}) , \mathcal {E} (\mathbf {h} _ {B}) \rangle_ {F}}{\| \mathcal {E} (\mathbf {h} _ {A}) \| _ {F} \| \mathcal {E} (\mathbf {h} _ {B}) \| _ {F}},\tag{3}
$$

where $\langle\cdot,\cdot\rangle_{F}$ is the Frobenius inner product and $\|\cdot\|_{F}$ is the Frobenius norm [20]. The cosine similarity is an extension of the vector inner product – it essentially projects a multi-dimensional linear space $\mathcal{E}(\mathbf{h}_{A})$ onto another $\mathcal{E}(\mathbf{h}_{B})$ to evaluate their similarity. The results are plotted in Fig. 3(a, d) for system-I (system-II) as functions of $v_{1}$ and $\kappa_{1}$ ( $\kappa_{2}$ ). It is seen that when $v_{1}=v_{e}$ (indicated by the blue planes), ED emerges with an infinitesimally small $\kappa_{1(2)}$ and the two eigenspaces perfectly align, resulting in C = 1. Even when the condition of ED is not exactly reached, i.e., $v_{1}$ deviates from $v_{e}$ , $\mathcal{E}(\mathbf{h}_{A})$ and $\mathcal{E}(\mathbf{h}_{B})$ nearly aligned when $\kappa_{1,2}$ are sufficiently large. (Comparing Fig. 3(a, d), C of system-I drops faster when $v_{1}$ deviates from $v_{e}$ . This is attributed to the profiles of the extended states, which make them more sensitive to $v_{1}$ .) In other words, with sufficient $\kappa_{1,2}$ , the characteristics of the OBC eigenstates are still dominated by the properties of ED even when $\eta(\mathbf{h}_{A})$ and $\eta(\mathbf{h}_{B})$ are slightly different.

To corroborate, we compute for each state a quantity

$$
\mathcal {G} := \frac {1}{2 N} \frac {\sum_ {n = 1} ^ {2 N} n | \phi_ {n} |}{\sum_ {n = 1} ^ {2 N} | \phi_ {n} |}\tag{4}
$$

with $\phi_{n}$ being the n-th entry of an OBC eigenstate, and 2N representing the total site number. G effectively gauges the spatial distribution of an OBC eigenstate. In Fig. 3b [3(e)], we see that $G \cong \frac{1}{2}$ (G is near zero) at non-zero $\kappa_{1(2)}$ , indicating all modes are extended (skin) modes at the ED. In Fig. 3(c, f), where $v_{1} = -0.765$ , the extended (skin) modes dominate system-I (II) when $\kappa_{1(2)}$ exceeds a threshold. This characteristic is highly desirable for realizing ED-related phenomena, because it could be difficult for two high-dimensional eigenspaces to have identical spectra in reality.

We use an active mechanical system, which is capable of realizing sophisticated non-Hermitian parameters $[23–28]$ , to experimentally realize the two double-chain systems. The one-way hopping is realized by electronic control. The natural frequency for each oscillator is 13.09 Hz and the two OBC bands span from $f_{1}=10.5$ Hz to $f_{2}=15.5$ Hz under the experimental parameters. (Note that the experimental parameters are not precisely at the ED, but as discussed, such deviation does not affect the general characteristics.) To obtain the steady-state response, one oscillator is harmonically driven and the angular displacement $\theta_{n}(f)$ is recorded for all oscillators. The integrated response, $A_{n}=\int_{f_{1}}^{f_{2}}|\theta_{n}(f)|^{2}df$ , is plotted in Fig. 2(f) for system-I, where delocalized responses in chain-A are seen. They agree with the theoretically computed responses shown in Fig. 2(g). For system-II, the responses are clearly due to the skin modes, as shown in Fig. 2(i, j).

The ED induces two distinct dynamic effects. We reveal them by preparing two different initial states as a localized wavepacket at the center of chain-A and B, respectively, then solve the time-dependent Schrödinger equation $i \frac{d}{dt} |\psi(t) \rangle = \mathbf{h}_{\mathrm{I},\mathrm{II}} |\psi(t) \rangle$ . The results in Fig. 4(a, b) are the dynamic responses of system-I. When the initial state is at chain-A, the wavepacket propagates symmetrically in two opposite directions and is reflected by the boundaries, which is not different from propagation in Hermitian systems [27]. However, when the initial state is at chain-B, the wavepacket starts to propagate in both directions, but the leftward propagation is amplified, similar to NHSE. Unlike conventional skin modes that remain localized at the boundary, this wave is reflected by the left boundary and counter-propagates across the bulk. Eventually, the wave field is propagative in the bulk and amplified to three orders of magnitudes of the initial state. So we call this skin-effect amplified propagation (SEAP).

The fascinating dynamics of SEAP is caused by the synergy of NHSE and propagation, which only occur in the proximity of the ED. In system-I, $\mathcal{E}(\mathbf{h}_{B})$ , which is spanned by skin modes, is defective. So the system's long-time behavior is dominated by propagative modes belonging to the non-defective $\mathcal{E}(\mathbf{h}_{A})$ . But an injection on chain-B can still trigger transient skin-mode responses, which amplifies the wave toward the left. This is similar to the excitation of a single “missing dimension” at an EP [3]. But this amplified wave refuses to congregate at the boundary because it can “escape” via the propagative channels provided by the non-defective $\mathcal{E}(\mathbf{h}_{A})$ . In comparison, when the injection is on chain-A, the defective $\mathcal{E}(\mathbf{h}_{B})$ is “hidden.”

The SEAP is observed in our mechanical lattice, wherein the wave propagates in both directions with the left-going wave experiencing an amplification and then reflected by the left edge (Fig. 4(c)). Due to system loss, the wave dissipates before reaching the right edge. The measured result sufficiently shows the key features of SEAP and agrees well with the theoretical result with dissipation included (Fig. 4(d)).

The second dynamic effect, propagation-enhanced skin effect (PESE), appears in system-II. When the wavepacket is injected at chain-A, it propagates both ways at first with an amplification to the leftward wave. The wave reflected by the right end (left-going after reflection) experiences considerable amplification upon reaching the left end (Fig. 4(e)). Clearly, the PESE is due to a different type of synergy of NHSE and propagation near the ED, with NHSE dominating the long-time dynamics. In comparison, when the wave packet is injected at the center of chain-B, the overall dynamics are akin to conventional NHSE (Fig. 4(f)). These hallmarks are also experimentally observed, as shown in Fig. 4(g), where the wave propagates in both directions in the beginning, but the branch reflected by the right boundary further grows in amplitude and eventually clings to the left edge [20]. The results agree well with the theoretical calculations (Fig. 4(h)).

Now recall that one necessary condition for ED is $\eta(\mathbf{h}_{A}) = \eta(\mathbf{h}_{B})$ . This condition can be broken by, e.g., offsetting $\eta(\mathbf{h}_{B})$ by a constant $\zeta_{0}$ (Fig. 5(a, b)). The new spectra can be expressed as $\eta(\mathbf{h}_{\mathrm{I,II}}) = [\eta(\mathbf{h}_{A}) \cap \eta(\mathbf{h}_{B} + \zeta_{0}\mathbf{I}_{N})] \cup \Delta_{A} \cup \Delta_{B}$ , where $\Delta_{A,B}$ are the nonoverlapping parts of the spectra that belong to $\eta(\mathbf{h}_{A,B})$ , respectively. (In finite-sized systems, the “overlapping” part of the spectra does not exactly overlap. But the differences are negligible when the lattice is sufficiently large such that the OBC spectra approach continuum.) Apparently, the ED is now the alignment of certain portions of the two eigenspaces and does not involve the states with eigenvalues belonging to $\Delta_{A,B}$ . This is confirmed in Fig. 5(c, d), in which $\Delta_{A(B)}$ are populated by extended (skin) modes, whereas the states belonging to the overlapping parts remain dominated by the ED. In other words, extended and skin modes stably coexist in massive amount $(\mathcal{O}(N))$ , and their ratios are tunable by simply adjusting $\zeta_{0}$ . This functionality uniquely emerges when the system is near ED, where the topological correspondence of NHSE can be violated. It is beyond the capability of any existing non-Hermitian systems, wherein NHSE originates from the non-trivial topology of the PBC spectrum [17,29] so it cannot be “turned off” without fundamentally changing the system’s symmetry class [30]. Competing mechanisms, e.g., magnetic flux that triggers Landau quantization [31,32] and disorders that induce Anderson localization [33], can partially control NHSE, but they rely on the delicate balance of multiple physical effects and, hence, are far more difficult to implement and control.

The effectiveness of this scheme is validated in our experiments. The onsite natural frequency of chain-B is increased from 13.09 Hz to 14.12 Hz, causing a mismatch in the frequency ranges of the eigenvalues of chain-A and chain-B. Figure 5(e) presents the steady-state response for system-I, where the excitation is a mono-frequency signal at 11.1 Hz, which belongs to $\Delta_{A}$ . The response is clearly due to extended states. Shifting the excitation frequency to 15 Hz, which falls in the overlapping part of the spectrum, the response is dominated by the SEAP (Fig. 5(f)). For excitation at 16 Hz, which is in $\Delta_{B}$ , the response indicates conventional NHSE (Fig. 5(g)). In system-II, when the excitation is at 11.1 Hz and 16 Hz, the results are due to extended and skin modes, respectively (Fig. 5(h, j)). For an excitation at 15 Hz, the response indicates skin modes (Fig. 5(i)).

By summarizing the above, we arrive at a definition of the ED: a phenomenon in which the Hilbert space of a $2N \times 2N$ square matrix is $\mathcal{O}(N)$ defective, accompanied by all eigenvalues being two-fold degenerate. Such a square matrix is in a Jordan canonical form composed of $\mathcal{O}(N)$ Jordan cells of size 2. These conditions are distinct from conventional order-N EPs, where all N eigenvectors align, and the eigenvalues are N-fold degenerate. As a multi-dimensional generalization of EPs, ED transcends previous understandings and expectations of non-Hermitian physics and opens diverse horizons repleted with opportunities.

ED is a generic condition hinging on the block-triangular form of the Hamiltonian and is characterized by the $\mathcal{O}(N)$ defectiveness of Hilbert space. There are no restrictions on the specifics of the Hilbert space (such as orthogonality and dimensionality) and the forms of the spectra – they can be continuum, discrete (formed by states with isolated energy), or any combination of them. As such, ED and related phenomena are universal. One surprising property of the ED formed by continua is the robustness against certain types of local perturbations, such as randomness in onsite energy or in-chain coupling. Yet, in the meantime, ED is critically sensitive to specific perturbations that break the mathematical structure of the Hamiltonian, which somewhat resembles the critical NHSE [20,34].

ED is realizable using different metamaterials or even with natural materials by simply stacking them in layers with purely one-way coupling, which is achievable by utilizing spin-orbit coupling in microwaves $[35]$ , electrical coupling in acoustics $[36]$ , and electrical components $[37,38]$ . Systems near ED can also be regarded as a new type of material dominated by exceptional physics.

We believe the skin-propagative dynamics reported here are merely the tip of an iceberg for what new phenomena ED can bring. Perhaps the most anticipated advancement is the broadband exceptional physics and functionalities – a viable solution to the narrowband characteristics inherent to all EPs. Indeed, the ED involves the coincidence of spectra without bandwidth limitation is particularly desirable for sensing $[39]$ , modal control $[40]$ , and lasing $[41]$ , which are famed applications hinging on the properties of EPs.

Future works will explore ED appearing in Hamiltonians with more general forms beyond the Jordan cell structure, and the impact of different coupling structure $\kappa$ . Considering the diverse topological properties of EPs [42–48], ED may also be associated with exotic high-dimensional topology that governs their evolutions in parameter space and possible merger.

## References

[1] C. M. Bender and S. Boettcher, Real Spectra in Non-Hermitian Hamiltonians Having P T Symmetry, Phys. Rev. Lett. 80, 5243 (1998).

[2] W. D. Heiss, The physics of exceptional points, J. Phys. A Math. Theor. 45, 444016 (2012).

[3] H.-Z. Chen et al., Revealing the missing dimension at an exceptional point, Nat. Phys. 16, 571 (2020).

[4] T. Liu, S. An, Z. Gu, S. Liang, H. Gao, G. Ma, and J. Zhu, Chirality-switchable acoustic vortex emission via non-Hermitian selective excitation at an exceptional point, Science Bulletin 67, 1131 (2022).

[5] Y. Ashida, Z. Gong, and M. Ueda, Non-Hermitian physics, Adv. Phys. 69, 249 (2020).

[6] E. J. Bergholtz, J. C. Budich, and F. K. Kunst, Exceptional topology of non-Hermitian systems, Rev. Mod. Phys. 93, 015005 (2021).

[7] K. Ding, C. Fang, and G. Ma, Non-Hermitian topology and exceptional-point geometries, Nat. Rev. Phys. 4, 745 (2022).

[8] C. M. Bender and D. W. Hook, PT -symmetric quantum mechanics, Rev. Mod. Phys. 96, 045002 (2024).

[9] L. Feng, R. El-Ganainy, and L. Ge, Non-Hermitian photonics based on parity–time symmetry, Nat. Photon. 11, 752 (2017).

[10] Ş. K. Özdemir, S. Rotter, F. Nori, and L. Yang, Parity–time symmetry and exceptional points in photonics, Nat. Mater. 18, 783 (2019).

[11] M.-A. Miri and A. Alù, Exceptional points in optics and photonics, Science 363, eaar7709 (2019).

[12] L. Huang et al., Acoustic resonances in non-Hermitian open systems, Nat Rev Phys 6, 11 (2023).

[13] N. Okuma and M. Sato, Non-Hermitian Topological Phenomena: A Review, Annu. Rev. Condens. Matter Phys. 14, 83 (2023).

[14] R. Lin, T. Tai, M. Yang, L. Li, and C. H. Lee, Topological Non-Hermitian skin effect, Front. Phys. 18, 53605 (2023).

[15] S. Yao and Z. Wang, Edge States and Topological Invariants of Non-Hermitian Systems, Phys. Rev. Lett. 121, 086803 (2018).

[16] Z. Yang, K. Zhang, C. Fang, and J. Hu, Non-Hermitian Bulk-Boundary Correspondence and Auxiliary Generalized Brillouin Zone Theory, Phys. Rev. Lett. 125, 226402 (2020).

[17] K. Zhang, Z. Yang, and C. Fang, Correspondence between Winding Numbers and Skin Modes in Non-Hermitian Systems, Phys. Rev. Lett. 125, 126402 (2020).

[18] A. P. Seyranian and A. A. Mailybaev, Multiparameter Stability Theory with Mechanical Applications (World Scientific Pub. Co, Singapore River Edge, N.J, 2003).

[19] H. Radjavi and P. Rosenthal, Invariant Subspaces (Springer Berlin Heidelberg, Berlin, Heidelberg, 1973).

[20] See the supplementary materials for additional information., (n.d.).

[21] L. Zhang et al., Acoustic non-Hermitian skin effect from twisted winding topology, Nat Commun 12, 6297 (2021).

[22] K. Yokomizo and S. Murakami, Non-Bloch Band Theory of Non-Hermitian Systems, Phys. Rev. Lett. 123, 066404 (2019).

[23] W. Wang, X. Wang, and G. Ma, Non-Hermitian morphing of topological modes, Nature 608, 50 (2022).

[24] W. Wang, X. Wang, and G. Ma, Extended State in a Localized Continuum, Phys. Rev. Lett. 129, 264301 (2022).

[25] W. Wang, M. Hu, X. Wang, G. Ma, and K. Ding, Experimental Realization of Geometry-Dependent Skin Effect in a Reciprocal Two-Dimensional Lattice, Phys. Rev. Lett. 131, 207201 (2023).

[26] X. Cui, R.-Y. Zhang, X. Wang, W. Wang, G. Ma, and C. T. Chan, Experimental Realization of Stable Exceptional Chains Protected by Non-Hermitian Latent Symmetries Unique to Mechanical Systems, Phys. Rev. Lett. 131, 237201 (2023).

[27] Z. Li, L.-W. Wang, X. Wang, Z.-K. Lin, G. Ma, and J.-H. Jiang, Observation of dynamic non-Hermitian skin effects, Nat Commun 15, 6544 (2024).

[28] W. Wang, X. Wang, and G. Ma, Anderson Transition at Complex Energies in One-Dimensional Parity-Time-Symmetric Disordered Systems, Phys. Rev. Lett. 134, 066301 (2025).

[29] N. Okuma, K. Kawabata, K. Shiozaki, and M. Sato, Topological Origin of Non-Hermitian Skin Effects, Phys. Rev. Lett. 124, 086801 (2020).

[30] K. Kawabata, K. Shiozaki, M. Ueda, and M. Sato, Symmetry and Topology in Non-Hermitian Physics, Phys. Rev. X 9, 041015 (2019).

[31] H. Gao, W. Zhu, H. Xue, G. Ma, and Z. Su, Controlling acoustic non-Hermitian skin effect via synthetic magnetic fields, Applied Physics Reviews 11, 031410 (2024).

[32] H. T. Teo, S. Mandal, Y. Long, H. Xue, and B. Zhang, Pseudomagnetic suppression of non-Hermitian skin effect, Science Bulletin 69, 1667 (2024).

[33] N. Hatano and D. R. Nelson, Localization Transitions in Non-Hermitian Quantum Mechanics, Phys. Rev. Lett. 77, 570 (1996).

[34] L. Li, C. H. Lee, S. Mu, and J. Gong, Critical non-Hermitian skin effect, Nat Commun 11, 5491 (2020).

[35] S. Wang, B. Hou, W. Lu, Y. Chen, Z. Q. Zhang, and C. T. Chan, Arbitrary order exceptional point induced by photonic spin–orbit interaction in coupled resonators, Nat Commun 10, 832 (2019).

[36] Z. Chen, Z. Li, J. Weng, B. Liang, Y. Lu, J. Cheng, and A. Alù, Sound non-reciprocity based on synthetic magnetism, Science Bulletin 68, 2164 (2023).

[37] L. Su, C.-X. Guo, Y. Wang, L. Li, X. Ruan, Y. Du, S. Chen, and D. Zheng, Observation of size-dependent boundary effects in non-Hermitian electric circuits, Chinese Phys. B 32, 038401 (2023).

[38] C.-X. Guo, L. Su, Y. Wang, L. Li, J. Wang, X. Ruan, Y. Du, D. Zheng, S. Chen, and H. Hu, Scale-tailored localization and its observation in non-Hermitian electrical circuits, Nat Commun 15, 9120 (2024).

[39] W. Chen, Ş. Kaya Özdemir, G. Zhao, J. Wiersig, and L. Yang, Exceptional points enhance sensing in an optical microcavity, Nature 548, 192 (2017).

[40] J. Doppler, A. A. Mailybaev, J. Böhm, U. Kuhl, A. Girschik, F. Libisch, T. J. Milburn, P. Rabl, N. Moiseyev, and S. Rotter, Dynamically encircling an exceptional point for asymmetric mode switching, Nature 537, 76 (2016).

[41] L. Feng, Z. J. Wong, R.-M. Ma, Y. Wang, and X. Zhang, Single-mode laser by parity-time symmetry breaking, Science 346, 972 (2014).

[42] K. Kawabata, T. Bessho, and M. Sato, Classification of Exceptional Points and Non-Hermitian Topological Semimetals, Phys. Rev. Lett. 123, 066405 (2019).

[43] C. C. Wojcik, X.-Q. Sun, T. Bzdušek, and S. Fan, Homotopy characterization of non-Hermitian Hamiltonians, Phys. Rev. B 101, 205417 (2020).

[44] Z. Li and R. S. K. Mong, Homotopical characterization of non-Hermitian band structures, Phys. Rev. B 103, 155129 (2021).

[45] W. Tang, X. Jiang, K. Ding, Y.-X. Xiao, Z.-Q. Zhang, C. T. Chan, and G. Ma, Exceptional nexus with a hybrid topological invariant, Science 370, 1077 (2020).

[46] W. Tang, K. Ding, and G. Ma, Experimental realization of non-Abelian permutations in a three-state non-Hermitian system, Natl. Sci. Rev. 9, nwac010 (2022).

[47] W. Tang, K. Ding, and G. Ma, Realization and topological properties of third-order exceptional lines embedded in exceptional surfaces, Nat. Commun. 14, 6660 (2023).

[48] K. Yang, Z. Li, J. L. K. König, L. Rødland, M. Stålhammar, and E. J. Bergholtz, Homotopy, symmetry, and non-Hermitian band topology, Rep. Prog. Phys. 87, 078002 (2024).

Acknowledgments K.S., M.S., and G.M. thank the hospitality of the Simons Center for Geometry and Physics at Stony Brook University. This work was supported by the National Key R&D Program (2022YFA1404400, 2023YFA1407500), the Hong Kong Research Grants Council (RFS2223-2S01, 12301822), and the Hong Kong Baptist University (RC-RSRG/23-24/SCI/01, RC-SFCRG/23-24/R2/SCI/12). K.S. and M.S. were supported by JST CREST (JPMJCR19T2). M.S. was supported by JSPS KAKANHI (JP24K00569). K.S. was supported by JST SPRING (JPMJSP2110). Z.Y. was also sponsored by the National Natural Science Foundation of China (12322405, 12104450).

![](images/e530c6923ad3a618014d7f26011135ac7f2b83c5d6e09acd1a5f2bfeb7ea8f69.jpg)

![](images/a258f6a0a6b4e806f0b116e906682fb2734bee29f551a5eeac3c5514f2b946a2.jpg)

![](images/0198120aecd286494f1b60a2431a135bd3fda80048d08cb0a59456c94f41ea43.jpg)
(c)

![](images/a0d163d0674b534964d36f8a6be20f22c2cf0714425b07ca63c1854d49469457.jpg)

![](images/ce5c6eada3cf795af7616b0644dfe7e91e5d0e2ba85c1d5a19ef48b0909c1564.jpg)
$\mathcal{E}(\mathbf{h}_{2})$

![](images/4c8f5f8ba5123013f833dc0ce37ed1dd2c465026911a651b72bb2b700524a7e9.jpg)

![](images/4850b58bfcf41bdabdd9cdc4ebecab8fd47e20f82df4df5aa9c2058f903af88f.jpg)

![](images/fcda81249933144306005bc0be03418a0967e9b8362679e16bc00e94a1914c6e.jpg)
ED
Fig. 1. ED as a generalization of EP. (a and b) When a two-level system approaches an EP (marked by the magenta color), the eigenvalues gradually become identical (a) and the eigenvectors coalesce (b). (c to e) Conceptual depictions of the emergence and properties of ED. Here, high-dimensional orthogonal (skewed) eigenspaces are represented by three-dimensional projections of hyper-cubes (hyper-parallelepipeds). (c) ED formed by two identical Hermitian eigenspaces. (d) ED can also be formed by coalescing a skewed non-Hermitian eigenspace with an orthogonal Hermitian one, resulting in a highly defective but orthogonal Hilbert space. (e) Conversely, the Hilbert space at ED can also be defective and skewed.

![](images/520bc068c9706c7b2721efd3993d51b06869d989c7d868e3448df7206eca9d93.jpg)

![](images/9c3e126a355238730b71092632f2c63bb3e2ad7b7f72eabcf2f12c182e99d98a.jpg)

![](images/b311ba9cf1a51c8c4a9b243c5bf25e5320bb019a5b27597c245c0e03a281e7b0.jpg)

![](images/9386d9f415f7dc14c6d7ae47b4b479e203b6761e1cfc6f991d5ecd60a7bae04a.jpg)

![](images/ba9b1fc05aad8b4db9afb3a7cc0faab4b03cf5e21561efdc0484200c0458c09c.jpg)

![](images/f7c782848a06adae00d2243a5aee1e190df60e50973db6c4c3d0ceab486396a7.jpg)

![](images/2c019f2d18ce91a0e0482af7cd2f1036a9a49562ecf5f1e877a05b4d99ef3a6b.jpg)
Fig. 2. ED-induced anomalous NHSE in one-dimensional double-chain lattices. (a) The two double-chain models with block-triangular Hamiltonians (Eq. 2). The two chains are one-way coupled by $\kappa_{1}$ and $\kappa_{2}$ , respectively. The PBC spectra (b) and OBC spectra (c) of the double-chain models under PBC and OBC, respectively. The spectra of system-I and II are identical. (d) The two models have identical GBZs. (e and h) The OBC eigenstates of system-I (e) and system-II (h), ordered in ascending order of the eigenvalues. The OBC bandgap and zero-energy edge modes are irrelevant to our study and are omitted for clarity of presentation. (f and g) Measured and computed steady-state response in system-I. (i and j) Steady-state response results for system-II. Parameters in theoretical calculations are $w = -1.565$ , $v_{2} = -0.885$ , $\delta = -0.453$ , $\kappa_{1} = \kappa_{2} = -1.081$ , $v_{1} = v_{e} = -\sqrt{(v_{2} - \delta)(v_{2} + \delta)}$ . (The values are retrieved from our experimental setup. As a result, $v_{e}$ is an irrational number. We kept 30 digits in the calculations.) In the experiments, the lattices have 8 unit cells. The natural frequency of individual oscillators is $f = 13.09\mathrm{Hz}$ . In the experiments, $v_{1} = -0.765$ , which slightly deviates from $v_{e}$ .

![](images/0b90a86039d5062827ea0c265a3b56be5fd1fcd9d49506619acbbc29dfc7b7f4.jpg)

![](images/84bb1b1216434bceeb19248cdb05c24a2fdff2d1e0956793bd1529b6b55f2241.jpg)

(b)
![](images/915679549d3d7fbb6af2ea540edb3518b8237aef5bc435379f3655217ebe1c7d.jpg)

![](images/8d12c65ca35907410bf52df55ac70760f661b9322d4346b5e155e8f6fa0c7891.jpg)

![](images/de777bd947de2a0b9cd97e3052386d4e625427603a2bf67294359ae6d06df3bd.jpg)

![](images/f97cefb3b49269e417754dd8d2fa183fbeb7f523dcb1edc12dc2cd9d4b779b10.jpg)
Fig. 3. The properties of the Hilbert space and eigenstates of the OBC lattices near the ED. (a and d) Cosine similarities C between the two coalescing eigenspaces in system-I and II. The ED is precisely reached at the blue planes, where the two eigenspaces align with infinitesimal $\kappa_{1,2}$ . The red points in (a) and (d) represent the cosine similarities when $v_{1} = -0.765$ and $\kappa_{1} = \kappa_{2} = -1.081$ , which are the experimental parameters. The two eigenspaces can be nearly aligned when the parameters slightly deviate from the ED condition, provided $\kappa_{1,2}$ is sufficient. (b and c) Spatial characteristics of continuum eigenstates at ED (b) and slightly away from ED (c) for system-I. In (c), extended states dominate with sufficient $\kappa_{1}$ , despite the deviation from the ED (indicated by the pink plane in (a). (e and f) Similar to (b, c) but for system-II. Here, skin modes dominate near the ED. The quantity G is defined in Eq. 4.

![](images/c68900f276b10024b9b67abcf1fe502b2b5e484754196ea2aadb28c1ab4ca35a.jpg)

(b) $^{10}$
![](images/2480b75b54a557239030ea546801f82a9c29ac493f1ffdeec2396110ecd74f4d.jpg)

![](images/bf42c5f9d685d942a958ba836634414a72a41dbbe447b383fcbc20d8eee75815.jpg)

![](images/ea7e070ebfe4029ee8d2e073b0af3d5232ed38dd41db22d51c324009461f4e8d.jpg)

![](images/69ac8d8884c9a4455accd1eb2526c8d16a86dbd13a02cfe7a71d7cf0abeaa809.jpg)

(d)
![](images/a8ed158e096e188c3a6e59dd22cc6da4def7223791504ccc6c29f78c37bedd9d.jpg)

![](images/253cb779dd8b96944a4a83b264e0e2e60bcfc38eeb1c79a91eb183dbe1b442dd.jpg)

![](images/c067619b0c79463b7954b30de1fde856187093f9c0d8004b824da422bfa1cd2a.jpg)
Fig. 4. ED-induced skin-propagative synergized dynamics. (a to d) SEAP in system-I. (a, b) The distribution of the probability density in system-I as functions of time when a wavepacket is injected is at the center of chain-A (marked by the black arrow) (a), and chain-B (marked by the blue arrow) (b). The SEAP is seen in (b). (c) Experimentally measured SEAP in the mechanical lattice. (d) Theoretically computed SEAP dynamics using experimental parameters, with loss considered. (e to h) PESE in system-II. The panels are organized in the same manner as (a to d). The parameters in (a and b) and (e and f) are the same as used in Fig. 2, and $v_{1} = -0.765$ . The onsite term is $13.09 - 0.34i$ in (d) and (h).

![](images/c9a092f6b9f20bb6396b8057b44b2bc2aa4eb61bcffe5c8dc23cf50848f2a56f.jpg)

![](images/d763ea84a4f67f75e426429c11ca643d72a3529bb4390dd4a4d43f3c458c9acd.jpg)

![](images/139fe997834ea923a642ba8e2ad56624fc9a491338a41d4abab879d4d06f0001.jpg)

![](images/2fd446f9d2513eba4a0b012935b93cb41d3b9c449b2d8470bad65eb2d83d45f3.jpg)

![](images/2ac5c7965a9dd7b19773ba1c0dfa5f4148206f70ed490a7f76081edabe3086ac.jpg)

![](images/ec41c04c7ea99c4d0aac14f88e8c8d0e369305f8a5b3208dcce04289bb7766b5.jpg)
Fig. 5. Controlling extended and skin modes using ED. (a and b) The PBC and OBC spectra of system-I and II with an offset in the spectrum of chain-B. (c and d) The eigenstates of the continuous bands of system-I and II. Extended modes and skin modes coexist in massive amounts. Their numbers are arbitrarily tunable by the offset. (e to g) Experimentally measured steady-state response of system-I at 11.1 Hz, 15 Hz, 16 Hz. (h to j) The experimental results for system-II. The excitation position is marked by the arrows. The natural frequencies of chain-A and chain-B are $f_{A} = 13.09$ Hz, $f_{B} = 14.12$ Hz. All other parameters are the same as those in Fig. 2.

# Exceptional deficiency of non-Hermitian systems: high-dimensional coalescence and dynamics

Zhen Li $^{1}$ , Xulong Wang $^{1}$ , Rundong Cai $^{1}$ , Kenji Shimomura $^{2}$ ,

Zhesen Yang $^{3}$ , Masatoshi Sato $^{2}$ , Guancong Ma $^{1,4}$

$^{1}$ Department of Physics, Hong Kong Baptist University, Kowloon Tong, Hong Kong, China.

$^{2}$ Center for Gravitational Physics and Quantum Information, Yukawa Institute for Theoretical Physics,

Kyoto University, Kyoto 606-8502, Japan.

$^{3}$ Department of Physics, Xiamen University, Xiamen 361005, Fujian, China.

$^{4}$ Shenzhen Institute for Research and Continuing Education, Hong Kong Baptist University, Shenzhen 518000, China.

## 1.1 Eigenstates of block-triangular matrices

Here, we take the system-I as an example and demonstrate the exact form of eigenstates and their relationships with the eigenvalues. The OBC Hamiltonian of system-I is $\mathbf{h}_{\mathrm{I}} = \begin{pmatrix} \mathbf{h}_{A} & \mathbf{\kappa}_{1} \\ 0 & \mathbf{h}_{B} \end{pmatrix}$ , where $h_{A}$ and $h_{B}$ represent the OBC Hamiltonian of chain-A and chain-B, respectively (Note that the specific form of $h_{I}$ here requires re-numbering the sites, which simply amounts to a similarity transform and has no effect on the system). $h_{I}$ is also block-triangular and both $h_{A}$ and $h_{B}$ are diagonalizable, so there exists an invariant subspace [1]. The eigenproblem gives three equations

$$
\left( \begin{array}{c c} \mathbf {h} _ {A} & \boldsymbol {\kappa} _ {1} \\ 0 & \mathbf {h} _ {B} \end{array} \right) \binom{| u _ {A} \rangle}{| u _ {B} \rangle} = E _ {\mathrm{I}} \binom{| u _ {A} \rangle}{| u _ {B} \rangle},\tag{1}
$$

$$
\mathbf {h} _ {A} | u _ {A} \rangle + \kappa_ {1} | u _ {B} \rangle = E _ {\mathrm{I}} | u _ {A} \rangle ,\tag{2}
$$

$$
\mathbf {h} _ {B} | u _ {B} \rangle = E _ {\mathrm{I}} | u _ {B} \rangle .\tag{3}
$$

Theorem of invariant subspace suggests that the spectrum of $h_{I}$ is the union of the spectrum of $h_{A}$ and $h_{B}$ , $\eta(\mathbf{h}_{\mathrm{I}})=\eta(\mathbf{h}_{A})\cup\eta(\mathbf{h}_{B})$ . Then for $|\phi\rangle=\binom{|u_{A}\rangle}{|u_{B}\rangle}$ , where $|u_{A}\rangle$ and $|u_{B}\rangle$ have the same dimension, to be right eigenvectors of $h_{I}$ , there are three possibilities for the eigenvalue $E_{I}$ :

When $E_{\mathrm{I}} \in \eta(\mathbf{h}_{A})$ and $E_{\mathrm{I}} \notin \eta(\mathbf{h}_{B})$ , we have $|u_{B}\rangle = 0$ from Eq. (3). Thus, Eq. (2) becomes $h_{A}|u_{A}\rangle = E_{I}|u_{A}\rangle$ . Therefore, $|u_{A}\rangle$ is either a zero vector (a trivial solution) or a right eigenvector of $h_{A}$ . So such eigenvectors become $|\phi\rangle = \begin{pmatrix} |u_{A}\rangle \\ 0 \end{pmatrix}$ .

When $E_{\mathrm{I}} \notin \eta(\mathbf{h}_{A})$ and $E_{\mathrm{I}} \in \eta(\mathbf{h}_{B})$ , $(\mathbf{h}_{A} - E_{\mathrm{I}})^{-1}$ exists, so Eq. (2) implies $|u_{A}\rangle = -(\mathbf{h}_{A} - E_{\mathrm{I}})^{-1}\mathbf{k}_{1}|u_{B}\rangle$ . Thus, if $|u_{B}\rangle = 0$ , then we also have $|u_{A}\rangle = 0$ , which leads to a trivial solution. So, to obtain a non-trivial solution, $|u_{B}\rangle$ must be a non-zero vector. Then, Eq. (3)

suggests that $|u_B\rangle$ is the eigenstates of $\mathbf{h}_B$ , so the eigenvectors are $|\phi \rangle = \begin{pmatrix} - (\mathbf{h}_A - E_I)^{-1}\mathbf{\kappa}_1|u_B\rangle \\ |u_B\rangle \end{pmatrix}$ . Those two cases are exemplified in Fig. 5(b) and 5(c) in the main text.

The third possibility is $\eta(\mathbf{h}_{A})=\eta(\mathbf{h}_{B})$ . So for $E_{I}\in\eta(\mathbf{h}_{A})$ , there is also $E_{I}\in\eta(\mathbf{h}_{B})$ . If $|u_{B}\rangle\neq0$ , then $|u_{B}\rangle$ is the right eigenstate of $h_{B}$ from Eq. (3). Since there is also $E_{I}\in\eta(\mathbf{h}_{A})$ , there exists a left eigenstate $\langle v_{A}|$ of $h_{A}$ satisfying $\langle v_{A}|\mathbf{h}_{A}=E_{I}\langle v_{A}|$ . Thus, Eq. (2) implies

$$
\langle v _ {A} | \mathbf {k} _ {1} | u _ {B} \rangle = 0.\tag{4}
$$

Since Eq. (4) is not generally held, so we must require $|u_{B}\rangle = 0$ . (In certain cases, Eq. (4) holds. But these are non-generic cases that require $K_{1}$ or $|u_{B}\rangle$ to have special structures.) Then, from Eq. (2), $|u_{A}\rangle$ is a right eigenstate of $h_{A}$ , and $|\phi\rangle = \begin{pmatrix} |u_{A}\rangle \\ 0 \end{pmatrix}$ . This is exactly the situation of the exceptional deficiency (ED), as shown in Fig. 2(c) and 2(e) in the main text.

We have shown that Eq. (4) is a necessary condition for the absence of ED so far, the converse can also be proved: for any $E_{\mathrm{I}} \in \eta(\mathbf{h}_{A}) = \eta(\mathbf{h}_{B})$ , ED does not occur if Eq. (4) holds for all the right eigenvectors $|u_{B}, b\rangle$ of $h_{B}$ and all the left eigenvectors $\langle v_{A}, a|$ of $h_{A}$ associated

$$
\langle v _ {A}, E _ {\mathrm{I}}, a | \mathbf {k} _ {1} | u _ {B}, E _ {\mathrm{I}}, b \rangle = 0, \text {for} \forall a, b, E _ {\mathrm{I}}\tag{5}
$$

where a, b index the eigenvectors associated with $\mathcal{E}(\mathbf{h}_{A})$ and $\mathcal{E}(\mathbf{h}_{B})$ . Indeed, under Eq. (5), we can construct all the linearly-independent right eigenvectors of $h_{I}$ with each eigenvalue $E_{I}$ as

$$
| \phi \rangle = \binom {| u _ {A}, E _ {\mathrm{I}}, a \rangle} {0}, \binom {- \sum_ {E \neq E _ {\mathrm{I}}} (E - E _ {\mathrm{I}}) ^ {- 1} \sum_ {a} \langle v _ {A}, E, a | \mathbf {k} _ {1} | u _ {B}, E _ {\mathrm{I}}, b \rangle   | u _ {A}, E, a \rangle} {| u _ {B}, E _ {\mathrm{I}}, b \rangle},\tag{6}
$$

which span the total Hilbert space. Here,

$$
\mathbf {h} _ {A} = \sum_ {E \in \eta (\mathbf {h} _ {A})} \sum_ {a} E | u _ {A}, E, a \rangle \langle v _ {A}, E, a |,\tag{7}
$$

$$
\mathbf {h} _ {B} = \sum_ {E \in \eta (\mathbf {h} _ {B})} \sum_ {b} E | u _ {B}, E, b \rangle \langle v _ {B}, E, b |,\tag{8}
$$

$$
\langle v _ {A}, E, a | u _ {A}, E ^ {\prime}, a ^ {\prime} \rangle = \delta_ {E, E ^ {\prime}} \delta_ {a, a ^ {\prime}}, \langle v _ {B}, E, b | u _ {B}, E ^ {\prime}, b ^ {\prime} \rangle = \delta_ {E, E ^ {\prime}} \delta_ {b, b ^ {\prime}}.\tag{9}
$$

Hence, Eq. (4) is a necessary and sufficient condition for the absence of the ED; equivalently the negation of Eq. (4) is that for the presence of the ED.

## 1.2 Cosine similarity

Cosine similarity evaluates the similarity between two linear spaces. In our work, the two eigenspaces, denoted $\mathcal{E}(\mathbf{h}_{A})$ and $\mathcal{E}(\mathbf{h}_{B})$ , are spanned by the eigenvectors corresponding to $\eta(\mathbf{h}_{A})$ and $\eta(\mathbf{h}_{B})$ , respectively (Each row of $\mathcal{E}\big(\mathbf{h}_{A(B)}\big)$ represents a right eigenvector $\left|\phi_{m_{A(B)}}\right\rangle$ ). The cosine similarity is defined as $\mathcal{C}\big(\mathcal{E}(\mathbf{h}_{A}),\mathcal{E}(\mathbf{h}_{B})\big)=\frac{\langle\mathcal{E}(\mathbf{h}_{A}),\mathcal{E}(\mathbf{h}_{B})\rangle_{F}}{\|\mathcal{E}(\mathbf{h}_{A})\|_{F}\|\mathcal{E}(\mathbf{h}_{B})\|_{F}}$ , where $\langle\mathcal{E}(\mathbf{h}_{A}),\mathcal{E}(\mathbf{h}_{B})\rangle_{F}=\sum_{m_{A(B)}=1}^{M_{A(B)}}\left|\left\langle\phi_{m_{A}}\mid\phi_{m_{B}}\right\rangle\right|$ is the Frobenius inner product (Here, $M_{A}+M_{B}$ gives the total number of bulk states, and in our systems $M_{A}=M_{B}=(2N-4)/2$ , 2N is the lattice size, and there are $(2N-4)$ modes in the continuum bands, 4 modes are in-gap edge modes), $\left\langle\phi_{m_{A}}\right|=\left(|\phi_{m_{A}}\rangle)^{\dagger}$ , $\left\|\mathcal{E}\big(\mathbf{h}_{A(B)}\big)\right\|_{F}=\sqrt{\sum_{m_{A(B)}}^{M_{A(B)}}\sum_{n=1}^{2N}\left|\phi_{m_{A(B)}n}\right|^{2}}$ is the Frobenius norm of $\mathcal{E}\big(\mathbf{h}_{A(B)}\big)$ . We do not differentiate parallel and anti-parallel spaces, with the resulting similarity ranging from 0 to 1, indicating orthogonal or parallel relation between the two eigenspaces.

## 1.3 Experimental setup

The experimental system is based on the active mechanical lattice, a proven technology for realizing non-Hermitian lattices $[2–6]$ . As shown in Fig. S1(a), the lattice has eight unit cells, each with four sites, which are realized by harmonic oscillators with a single rotational mode. A selected unit cell is highlighted by the white box, and its schematic diagram (of system-I) is depicted in Fig. S1(b). Each oscillator is comprised of a rigid arm loaded with weights, anchored with springs, and fixed on a programmable motor. The oscillators are properly connected by springs with designated spring constants to realize reciprocal hopping. Asymmetric hopping is achieved through a programmed feedback control that drives specific motors. For example, for the non-reciprocal hopping in chain-B, a microcontroller receives the real-time rotation angle $\theta_{4}$ of motor 4 (measured by a Hall sensor), then it drives motor 3 with a torque $\tau_{3}=a_{34}\theta_{4}$ , where $a_{34}$ is a controlling parameter. This effectively emulates an additional linear spring for oscillator 3. The two chains are coupled through purely one-way hopping, they are coupled solely through such programmed feedback with no physical spring connecting them. The experimental parameters are obtained using one unit cell by the Green's function, which is described in detail in refs. $[2,4,6]$ .

![](images/5dfe1a4bbdeff3ebdb160e2005f09826975c89c156b7af9120ba0b373301058a.jpg)

(b)
![](images/279f7360dad8cb050acc8d1fa34b48b3cc0c55b96872a71b04dad58428869dae.jpg)
Fig. S1. (a) The experimental lattice. The white box highlights a unit cell. (b) The schematic diagram of the unit cell in system-I. The blue and orange lines represent reciprocal hopping realized by tensioned springs, and non-reciprocal hopping is denoted by the red arrows, which are implemented using feedback-driven programmable motors.

## 2.1 Robustness and fragility of the ED

The ED shows rich behaviors under perturbation of different kinds and variations of parameters. Here, we present a comprehensive discussion.

## 2.1.1 Robustness

One would be tempted to think that ED being such a unique and unorthodox situation, any perturbation would have devastating effects. Surprisingly, this is not necessarily the case. Because ED-induced phenomenon affects a spectral continuum, it is quite robust against local perturbations. For example, Fig. S2(a, b) presents the change of the cosine similarity when onsite random disorders are introduced to a certain percentage of sites in system-I. Therein, $\alpha$ denotes that magnitude of randomness. It is seen that the cosine similarities remain at a level close to unity even in the presence of substantial disorder. The spectral mismatches at the band edges are a major contributing factor to the reduction in the cosine similarities. This result is corroborated by the OBC eigenstates that remain stably extended modes, as shown in Fig. S2(c, d). Figure S2(e-h) shows the corresponding results for system-II.

In the OBC systems, when the density of states is sufficiently large such that the bands become semi-continuum, and the presence of disorder merely shuffles the order of eigenstates, the spectra of the two eigenspaces still overlap. And because the formation of ED does not care about the specifics of the eigenstates, the ED-induced phenomena survive such disorder.

## 2.1.2 Criticality

As mentioned in the main text, the block-triangular form of the Hamiltonian is crucial for the emergence of the ED. Consequently, the ED is fragile against perturbations that break the block-triangular form. Figure S3(a, b) shows the PBC and OBC spectra of system-I when the inter-chain hopping is no longer strictly one-way. As $\kappa_{2}$ increases from 0 to 0.1, the spectra $\eta(\mathbf{h}_{A})$ and $\eta(\mathbf{h}_{B})$ no longer overlap (in fact, it is no longer possible to clearly separate the two eigenspaces into Bloch waves and non-Bloch waves). And the OBC eigenvectors gradually evolve into skin modes (Fig. S3(c-f)). In system-II, when the same perturbation is applied as $\kappa_{1}$ , the PBC and OBC spectra remain identical to those of system-I (Fig. S3(g, h)). The skin modes eventually become identical to the ones of system-I (Fig. S3(i-l)). However, the differences are less pronounced because the modes of system-II are already skin modes at ED to begin with.

![](images/c46d2cb6cf7685682466cc3d6bc6ed81cce546373e1f3bfa3c753d29c66b6855.jpg)

![](images/032456f905a5a6aa80253c5ea09302c359cbda21774d289dc62e21eff8e6ae33.jpg)

![](images/ff9ba5156bc39b2b510d3b75ffa4f5e0ec07d404807cf912745f51d1b76793f7.jpg)
(f)

![](images/1e8b2a6a9136b61b51b5ba79cb10810be231e2fc7ad855c80485ea6a989976df.jpg)

![](images/01b3f4c58efc8ab2a57dd90dd3865225b8c30cce14029604b7a8f8ed2f5bb193.jpg)

![](images/e42e2af849e7134c3111191806709d4bc0441830730b9cdbe183bf1d2ac8b2b9.jpg)

![](images/0e4a9200b57d01015277409a3a631115068e00d2f2f52867966c5abdfba675ed.jpg)

![](images/db081d1afa08ad87d04a37da3b28d4341843c86972d43c12bdd8b679269a3d91.jpg)
Fig. S2. (a) and (b) The cosine similarities between eigenspaces in system-I when different percentages of disorders, with amplitude $\alpha = 0.1$ , 0.2 are introduced to the onsite energies. (c) and (d) Two selected sets of OBC eigenstates under disorder perturbations. (e) to (h) The corresponding results in system-II. In (a, b) and (e, f), the blue points indicate the cosine similarities averaged over 100 configurations, and the red bars represent the range of all data sets. All parameters are the same as Fig. 2 in the main text with onsite terms.

(f)
![](images/b7d49b6e20488991a654ca2e87adb48defc1d018f95dd8d07b70193bd9a92839.jpg)

![](images/e1a779335e2b4bbcad3ddbce05044e4752346b1ddb4aa032cd98255fba0db72c.jpg)

![](images/4e4675b843460a8565f847cc5b9e61a84b372e76b1befdbd3b82b2541fc2c677.jpg)

![](images/3b9a094338625007f9a527d5cfde73b9e777ed3c152dbb541e7513dc4cdd321a.jpg)

![](images/a8703f7fd5950532b0f87f957b1a1942a15e537f595ffa6529ef9fabad571682.jpg)

![](images/4f164edbdd5e8dea20d496520d8b572ef92ac481a7acd2b9357b94101a25eac3.jpg)

![](images/0da9496276dc3d3c8787f37e12eb6f9e53d9f2e7d7b48092d6eeecd2c6b9c2dd.jpg)

![](images/594b54dce1aac4061964c2427518f440439d2e0b42dfb6479b571fce7924cfe0.jpg)

![](images/b64da43ca773edb948257b89bc013afe46eb025e7cedd80e8585507e49500cd9.jpg)

![](images/55a7cf9b7ee1a93f136a0a81acfbf4ee34f2516209b9c58ec211469c6f4b9a17.jpg)

![](images/467271b24bc5f178ed80bbefa6d7d3903b80eaa492ca1eeae35b5519f023eb62.jpg)

![](images/f260b35f8001b4ba1ea6b558c833a5edb10cd75ad70d75cdbb05bb6ca3d34a51.jpg)
Fig. S3. (a) The PBC and (b) OBC spectra of system-I when a perturbation is introduced as the A-to-B inter-chain hopping, denoted $\kappa_{2}$ . (c-f) The corresponding OBC eigenstates. (g) PBC and (h) OBC spectra of system-II when a perturbation is introduced as the B-to-A inter-chain hopping, denoted $\kappa_{1}$ . (i-l) The corresponding OBC eigenstates. The parameters used in the calculations are identical to Fig. 2 in the main text with zero onsite terms.

## 2.2 "ED curves"

Exceptional points (EPs) are known to form continuous curves under suitable conditions. Although the ED is completely different from EPs in that it is not a point in the spectrum, it can appear over a continuous region of parameters. Figure S4 illustrates the similarity as a function of system parameters $v_{1}$ and $\delta$ in our systems. For system-I, the ED exists when the equation $v_{1}^{2} + \delta^{2} = v_{2}^{2}$ is satisfied, which is a circle in the $v_{1}\delta$ -space. A similar ED circle is also found in system-II.

![](images/f12d95bdea56f667c504d94cbd76c4a1bb59f35bb7816ec9d872a4b4465a0319.jpg)

![](images/96206bbc435d217d1316beac7a0ad131c0235f1671b9f7135909b2f2cf9f90f5.jpg)
Fig. S4. The emergence of ED circles (marked by the magenta color) in system-I (a) and II (b). The equation of the ED curve is $v_{1} = -\sqrt{(v_{2} - \delta)(v_{2} + \delta)}$ . The cyan dots represent the positions of the ED in Fig. 2 in the main text. Other parameters are the same as Fig. 2 in the main text with zero onsite terms. Only a quarter of the circle is shown.

## 2.3 Additional models

To demonstrate the generality of the ED, we present similar effects in several other models.

Figure S5(a) depicts systems consisting of double-chain Hatano-Nelson (HN) models. The intra-chain hopping parameters in chain-a and b have the same amplitude but opposite signs, i.e., $\delta_{a} = -\delta_{b}$ . When the two chains are isolated, their PBC and OBC spectra coincide (Fig. S5(b, c)). The windings of the PBC spectra are opposite (Fig. S5(b)), which indicates that the skin modes in chain-a and chain-b are localized at the left edge and right edge, respectively. When the two chains are one-way coupled, it is straightforward to see that the Hamiltonian has a block-triangular form, and the condition for ED is met. As a consequence, system-1 (2) is exclusively populated by leftward (rightward) skin modes, as shown in Fig. S5 (d, e).

When $\delta_{a} \neq -\delta_{b}$ , the PBC and OBC spectra of the two chains no longer coincide, as depicted in Fig. S5(f, g). The OBC eigenstates of the two systems are shown in Fig. S5(h, i). The OBC eigenstates corresponding to the overlapping spectra of system-1 and 2 are dependent on the direction of the inter-chain hopping, whereas the non-overlapping eigenstates retain the profile of leftward skin modes in system-2, which belong to chain-a. In particular, both leftward and rightward skin modes exist in system-2, and the ratio of the two kinds of skin modes is tunable by changing the spectra of chain-a and chain-b. These observations are consistent with the theoretical predictions of the ED.

To further extend the idea, we also explore a model that couples a Hermitian SSH model and a non-Hermitian HN model, denoted as system-3 and system-4 and are shown in Fig. S6(a). The PBC spectra consist of a closed loop with a clockwise winding and a real spectrum, as shown in Fig. S6(b). The OBC spectra are comprised of two spectral lines lying on the real axis with partially overlapping energies, as illustrated in Fig. S6(c). The GBZs of those two systems are also identical (Fig. S6(d)). The eigenstates corresponding to the overlapping energies are extended states in system-3 and skin modes in system-4 (Fig. S6(e, f)).

These results indicate that ED-induced effects can be observed in many different systems, as long as the relevant conditions are met.

![](images/bd772400f57354162717ad530adc1777b76e49ab4e934f149845d5db0094248a.jpg)

![](images/af4724f747bbc44f454061217c97cd4caab05c745b7b5bd887940547b1ec44ce.jpg)

![](images/735c4886e6f7e4d207ef5420b11db93d93e709c729d72c32228b71f585a0db3b.jpg)

![](images/140e8b61e952bc845a2b6580316f123fd4af7c826cc30d6ac57c0e17b4115890.jpg)

![](images/16a893ee8bacc1b013da1141dc748a1efe15f234d7f5564b48a100207de28bf7.jpg)

![](images/1bdd5f57ec1872398547877efdca39965f8b721d00f62538ee844cd31511a914.jpg)

![](images/8ed3661a7da2ce87d9b7894ac8d7deec27d698f95473b7058b5fd1f460ad30b8.jpg)

![](images/7b7aed7a72e27f6460ecb5bf79e4bda5ca1dbe5d8284c3de48c93658ba790e59.jpg)

![](images/7ac3707f28b774a26493e9a7429ddf03b842513ff8d3f6f31cebfd0309676f1e.jpg)

![](images/bcf303843518fef2308f145c684140b3dc6796db02c62c6dd5c57b1b869b6eae.jpg)
Fig. S5. (a) The coupled HN models. (b and c) The PBC and OBC spectra of system-1(2) when the parameters are $t_{a} = t_{b} = 1$ , $\delta_{a} = 0.5$ , $\delta_{b} = -0.5$ , and $\kappa_{1} = \kappa_{2} = 1$ . (d and e) The corresponding OBC eigenstates of system-1 and system-2, respectively. (f and g) The

PBC and OBC spectra of system-1(2) when the parameters are $t_{a}=2$ , $\delta_{a}=0.5$ , $t_{b}=1$ , $\delta_{b}=-0.25$ , and $\kappa_{1}=\kappa_{2}=1$ . (h and i) The corresponding OBC eigenstates of system-1 and system-2, respectively.

![](images/4b9f55fcc17af23ccb0e4ad2129bcf23e883e8f8435ed0869c1a4e93108d9c93.jpg)

![](images/a3638d168c304e4b59936780632254e17920d0fa5ab3f35458d484aafea05680.jpg)

![](images/4b0f02e5d2876950268ba221309769612bd06b3e55f59d81e61e8860b294b9cd.jpg)

![](images/5e30a91843a944cc7cb976760c8c6ada76e2c6853f3dd8cba8460e95ac737518.jpg)

![](images/e439ada93201551e34d58df84b3cda041300cbaa78af9d3e5507b9e9bcad594e.jpg)

![](images/e001a5cac4fd5387443dc0693abb9ccbbb863904abfb550475605ca560b699c2.jpg)
Fig. S6. (a) The double-chain models consisted of a Hermitian SSH chain and a non-Hermitian HN chain. (b and c) PBC and OBC spectra of system-3 (4). (d) The GBZ of system-3 (4). (e and f) The OBC eigenstates of system-3 and system-4. The red and orange colors in (b to d) represent the spectrum and GBZ of chain-A and chain-b, respectively. The parameters are $v_{1} = -0.765$ , w = -1.565, $t_{b} = 1$ , $\delta_{b} = -0.25$ , and $\kappa_{1} = \kappa_{2} = 10$ . The OBC bandgap and zero-energy edge modes are omitted for clarity of presentation.

## 2.4 Exactly solvable case

In the special case of $t_{a} = -\delta_{a}$ , we can exactly solve the eigenproblem of the additional model system-1. Hereafter we consider the following OBC Hamiltonian

$$
\mathbf {h} _ {1} = \left( \begin{array}{c c} \mathbf {h} _ {a} & \boldsymbol {\kappa} _ {1} \\ 0 & \mathbf {h} _ {b} \end{array} \right), \mathbf {h} _ {a} = \left( \begin{array}{c c c c c} V & 0 & \dots & 0 & 0 \\ J & V & \dots & 0 & 0 \\ \vdots & \vdots & \ddots & \vdots & \vdots \\ 0 & 0 & \dots & V & 0 \\ 0 & 0 & \dots & J & V \end{array} \right), \mathbf {h} _ {b} = \left( \begin{array}{c c c c c} 0 & t r & \dots & 0 & 0 \\ t r ^ {- 1} & 0 & \dots & 0 & 0 \\ \vdots & \vdots & \ddots & \vdots & \vdots \\ 0 & 0 & \dots & 0 & t r \\ 0 & 0 & \dots & t r ^ {- 1} & 0 \end{array} \right),\tag{10}
$$

which is just the system-1 (Fig. S5(a)) with specific parameters $J/2 = t_{a} = -\delta_{a}$ , $t^{2} = t_{b}^{2} - \delta_{b}^{2}$ , $r^{2} = (t_{b} + \delta_{b})/(t_{b} - \delta_{b})$ , and additional onsite potential $V \in C$ in chain-a. Let L be the length of each chain-a, b; Both $h_{a}$ and $h_{b}$ are $L \times L$ matrices. While $h_{a}$ is non-diagonalizable and has a single eigenvalue V with an eigenvector $(0, \ldots, 0, 1)^{\mathrm{T}}$ , $h_{b}$ has L distinct eigenvalues $\epsilon_{q} = 2t\cos q$ , $q \in \left\{\frac{\pi}{L+1}, \frac{2\pi}{L+1}, \ldots, \frac{L\pi}{L+1}\right\}$ with eigenvectors $u_{q} = (r^{-1}\sin q, r^{-2}\sin 2q, \ldots, r^{-L}\sin Lq)^{\mathrm{T}}$ . If $V \notin [-2t, 2t]$ , the spectra of $h_{a}$ and $h_{b}$ do not overlap. Thus, the eigenvector $v_{q}$ of $h_{1}$ with the eigenvalue $\epsilon_{q}$ is

$$
v _ {q} = \binom {\psi_ {q}} {u _ {q}}, \quad \psi_ {q} = - \mathbf {\kappa} _ {1} (\mathbf {h} _ {a} - \epsilon_ {q}) ^ {- 1} u _ {q}.\tag{11}
$$

Thanks to the specific form of $h_{a}$ , we know an analytic form of the inverse $(\mathbf{h}_{a}-\epsilon_{q})^{-1}$ as

$$
- (\mathbf {h} _ {a} - \epsilon_ {q}) ^ {- 1} = \frac {1}{J} \left( \begin{array}{c c c c c} \frac {J}{\epsilon_ {q} - V} & 0 & \dots & 0 & 0 \\ \left(\frac {J}{\epsilon_ {q} - V}\right) ^ {2} & \frac {J}{\epsilon_ {q} - V} & \dots & 0 & 0 \\ \vdots & \vdots & \ddots & \vdots & \vdots \\ \left(\frac {J}{\epsilon_ {q} - V}\right) ^ {L - 1} & \left(\frac {J}{\epsilon_ {q} - V}\right) ^ {L - 2} & \dots & \frac {J}{\epsilon_ {q} - V} & 0 \\ \left(\frac {J}{\epsilon_ {q} - V}\right) ^ {L} & \left(\frac {J}{\epsilon_ {q} - V}\right) ^ {L - 1} & \dots & \left(\frac {J}{\epsilon_ {q} - V}\right) ^ {2} & \frac {J}{\epsilon_ {q} - V} \end{array} \right).\tag{12}
$$

From this, we can write the n-th component of $\psi_{q}$ as

$$
(\psi_ {q}) _ {n} = \frac {\kappa_ {1}}{\epsilon_ {q} - V} \frac {r ^ {- n} \left(\sin n q - \frac {r J}{\epsilon_ {q} - V} \sin (n + 1) q\right) + r \left(\frac {J}{\epsilon_ {q} - V}\right) ^ {n + 1} \sin q}{1 - 2 \frac {r J}{\epsilon_ {q} - V} \cos q + \left(\frac {r J}{\epsilon_ {q} - V}\right) ^ {2}}.\tag{13}
$$

It should be emphasized that the eigenvector $\psi_q$ on chain-a has two competing localization terms $r^{-n}$ and $\left(\frac{J}{\epsilon_g - V}\right)^{n + 1}$ .

Next, we normalize the eigenvector $v_{q}$ as

$$
\tilde {v} _ {q} = \frac {v _ {q}}{\parallel v _ {q} \parallel_ {F}} = \frac {1}{\sqrt {\parallel \psi_ {q} \parallel_ {F} ^ {2} + \parallel u _ {q} \parallel_ {F} ^ {2}}} \binom {\psi_ {q}} {u _ {q}}\tag{14}
$$

and thereby we examine the infinite-volume behavior of the normalized eigenvector $\tilde{v}_{q}$ with fixed q. To do this, we evaluate the ration $\|\psi_{q}\|_{F}^{2}/\|u_{q}\|_{F}^{2}$ . After a long calculation, we have

$$
\frac {\| \psi_ {q} \| _ {F} ^ {2}}{\| u _ {q} \| _ {F} ^ {2}} \stackrel {{L \to \infty}} {{\to}} \left\{ \begin{array}{l l} \infty & \text {if} \left\{ \begin{array}{c} \left| \frac {r J}{\epsilon_ {q} - V} \right| \geq 1   \&   0 <   r <   1 \\ \left| \frac {J}{\epsilon_ {q} - V} \right| > 1   \&   r = 1 \\ \left| \frac {J}{\epsilon_ {q} - V} \right| \geq 1   \&   r > 1. \end{array} \right. \\ \text {const.} & \text {otherwise.} \end{array} \right.\tag{15}
$$

Intuitively, this result can be interpreted as competition between two localization behaviors $r^{-n}$ and $\left(\frac{J}{\epsilon_{q}-V}\right)^{n+1}$ . In case of $\left|\frac{J}{\epsilon_{q}-V}\right| \geq 1$ & r > 1, for instance, the term $\left(\frac{J}{\epsilon_{q}-V}\right)^{n+1}$ increases as one moves to the right of the chain while the term $r^{-n}$ is bounded, so $\| \psi_{q} \|_{F}$ containing both terms is larger than $\| u_{q} \|_{F}$ containing only the term $r^{-n}$ . Accordingly, we find the normalized eigenvector exhibits an asymptotic behavior of

$$
\tilde {v} _ {q} \stackrel {L \to \infty} {\rightarrow} \binom {*} {0} \quad \text {if} \left\{\begin{array}{r l}&\left| \frac {r J}{\epsilon_ {q} - V} \right| \geq 1 \&0 <   r <   1\\&\left| \frac {J}{\epsilon_ {q} - V} \right| > 1 \&r = 1\\&\left| \frac {J}{\epsilon_ {q} - V} \right| \geq 1 \&r > 1.\end{array}\right.\tag{16}
$$

In other words, the eigenvectors of $h_{1}$ get deficient in the Hilbert subspace $0 \oplus C^{L}$ of chain-b after the infinite volume limit. It is noteworthy that such an asymptotic emergence of ED occurs only when $J \geq |\epsilon_{q} - J|$ , which implies the eigenvalue $\epsilon_{q}$ is encircled by the PBC spectrum $\{V + Je^{ik} \mid k \in R\}$ of $h_{a}$ .

Lastly, we discuss about general criterion for asymptotic ED apart from the special case above. We just have to explore a condition for $\|\left(\mathbf{h}_{a}-E\right)^{-1}\kappa_{1}u\|_{F}/\|u\|_{F}\stackrel{L\to\infty}{\to}\infty$ , where $h_{b}u=Eu$ . We take the largest singular value $s_{L}(E_{0})=\|\left(\mathbf{h}_{a}-E_{0}\right)^{-1}\|$ and the right-singular vector $w_{L}(E_{0})$ of $(\mathbf{h}_{a}-E_{0})^{-1}$ for each L. Then we have

$$
\frac {\parallel (\mathbf {h} _ {a} - E) ^ {- 1} \pmb {\kappa} _ {1} u \parallel_ {F}}{\parallel u \parallel_ {F}} \geq \frac {s _ {L} (E _ {0}) | w _ {L} (E) ^ {\dagger} \pmb {\kappa} _ {1} u |}{\parallel u \parallel_ {F}},\tag{17}
$$

which gives a sufficient condition for asymptotic ED as

$$
\frac {\parallel (\mathbf {h} _ {a} - E) ^ {- 1} \parallel | w _ {L} (E) ^ {\dagger} \pmb {\kappa} _ {1} u |}{\parallel u \parallel_ {F}} \stackrel {L \to \infty} {\rightarrow} \infty .\tag{18}
$$

Because of $|w_L(E)^\dagger \mathbf{k}_1 u| / \| u \|_F \leq \| \mathbf{k}_1 \|$ , this condition involves the divergence of the matrix norm $\| (\mathbf{h}_a - E)^{-1} \|$ as long as $\| \mathbf{k}_1 \|$ is uniformly bounded with respect to $L$ . If the skin effect occurs for $\mathbf{h}_a$ , $\| (\mathbf{h}_a - E)^{-1} \|$ for OBC Hamiltonian $\mathbf{h}_a$ relates with the corresponding PBC spectrum. To see this, we focus on the pseudospectrum $\sigma_\epsilon(\mathbf{h}_a)$ , defined as the region of $z \in \mathbb{C}$ satisfying $\| (\mathbf{h}_a - z)^{-1} \| > \epsilon^{-1}$ . As known in the context of topological origin of skin effects, a reference energy $E_0$ at which the PBC spectrum has a nonzero winding number belongs to the limiting pseudospectrum $\lim_{\epsilon \to 0} \lim_{L \to \infty} \sigma_\epsilon(\mathbf{h}_a)$ and therefore satisfies $\| (\mathbf{h}_a - E_0)^{-1} \| \overset{L \to \infty}{\to} \infty$ [7]. Thus, a typical situation to realize Eq. (18) is when the PBC spectrum of $\mathbf{h}_a$ encircles an OBC eigenvalue $E$ of $\mathbf{h}_b$ with nonzero winding induced by the skin effect which is consistent with the result in the specific model.

## 2.5 Additional experimental results

In addition to the experimental results shown in the main text, we have comprehensively characterized system-I and II. Figure S7(a) plots the measured steady-state response of system-I when a harmonic source is applied to chain-A. In this scenario, the system's response is identical to a Hermitian SSH chain. For system-II, when chain-B is excited by a harmonic signal, the steady-state response is significantly amplified towards the left edge, which is the typical response of NHSE (Fig. S7(c)). The theoretical results calculated using Green's function agree well with the experimental results, as shown in Fig. S7(b, d).

Figure S8(a) plots the measured dynamics of system-I when a wavepacket is injected to chain-A. It is observed that the wavepacket propagates symmetrically in both directions. This behavior is identical to wavepacket propagation in a Hermitian SSH chain. The theoretical result shown in Fig. S8(b) aligns well with the experimental findings. Figure S8(c) plots the experimental results for system-II when the injection is at chain-B. In this case, the wavepacket predominantly propagates towards the left edge, as shown in Fig. S8(c, d). This is identical to the typical non-Hermitian skin dynamics.

![](images/ef09fad296dbc12924f1059179c99bad7a689575e5cf2ba488254e0e10d0d0dc.jpg)

Figure S9(a-c) depicts the steady-state response of system-I when a single-frequency excitation at different frequencies is applied to chain-A. As illustrated in Fig. S9(a, b), the wavepacket propagates to both edges of the system, demonstrating the properties of bulk modes. When the excitation frequency is outside the eigenfrequency range of chain-A, the wavepacket decays significantly as it moves away from the source, as shown in Fig. S9(c). For system-II, when an excitation with a frequency lower than the eigenfrequency of chain-B is applied to chain-B, it exhibits an asymmetric attenuation trend, as depicted in Fig. S9(d). When the excitation frequency matches the eigenfrequency of chain-B, the skin modes are successfully detected, as shown in Fig. S9(e, f).

![](images/08cea9c87089af6a40ae6692f89bd8c82286b05c5d37db32dfe51c174276ee26.jpg)

![](images/b1e0fb087cec2d55090742cf9ce6e0f48d8aeccf6686640d438f953b436f171a.jpg)
Fig. S7. (a) The experimentally measured steady-state response of system-I when the source is at chain-A. (b) The corresponding theoretical response. (c) The experimental results of system-II when the source is at chain-B. (d) Corresponding theoretical result. The parameters are $v_{1} = -0.765$ , w = -1.565, $v_{2} = -0.885$ , $\delta = -0.453$ , $\kappa_{1} = \kappa_{2} = -1.081$ , $\omega_{0} = 13.09$ . The onsite dissipation is $\gamma = 0.03$ for system-I and $\gamma = 0.34$ for system-II. The position of the source is marked by the small arrow.

![](images/50e622855e540523b762f5e8ea55c7183ef6c2634dfef90624bf0c90bd6ea95d.jpg)

![](images/f74cba09cb293d3043788b7fdb067c696947bd4a0ce60dbf05a753fc5285e31a.jpg)

![](images/b8fc6c7fab358f1036311a7360937a074eec5a3d971d797547b878e8e42cb090.jpg)

![](images/e1d1c97fcd1c0be85a98a5fbc67536e7e10e6d92aaaf443d5211c99fc56754e6.jpg)
Fig. S8. Dynamic behaviors of system-I and system-II. (a and b) Dynamic behaviors of system-I with wavepacket injection at chain-A. (c and d) Dynamic behaviors of system-II with wavepacket injection at chain-B. The onsite parameters are $\omega_{0}=13.09$ , and $\gamma=0.34$ . The hopping parameters are the same as in Fig. S7.

![](images/c2a7bd064814e0266ec7fdf61667a3b834b0e203a60c5c4828b7779afd8638d6.jpg)

![](images/b7068b9b71d57b96f45846f3348bba3d501fc33f4cf11aaaac306a1825426d33.jpg)

![](images/5b66a906464228af077d69052bf11f761b9af2db3cadc0af4204e4bf810fa4a5.jpg)

![](images/7610ea7a1a4321a765a63de6a4e247074383cc56e02b70b2f657ab5d8454f120.jpg)

![](images/7eb74afb8a4d61cb09142b9e2f2d19786f2ff2a5845f96fd072c1f716187c6e4.jpg)

![](images/cc2515080c6047c692b90d23a344b273c98cf976c90c69e3d566117c5d897a6c.jpg)
Fig. S9. (a to c) Steady-state response of system-I when the excitation is applied at site 17 (indicated by the black arrow). (d to f) Experimental results for system-II, with the excitation at site 16 (indicated by the blue arrow). The natural frequencies of chain-A and chain-B are $f_{A} = 13.09$ , $f_{B} = 14.12$ . All other parameters remain the same.

## References

[1] H. Radjavi and P. Rosenthal, Invariant Subspaces (Springer Berlin Heidelberg, Berlin, Heidelberg, 1973).

[2] W. Wang, X. Wang, and G. Ma, Non-Hermitian morphing of topological modes, Nature 608, 50 (2022).

[3] W. Wang, X. Wang, and G. Ma, Extended State in a Localized Continuum, Phys. Rev. Lett. 129, 264301 (2022).

[4] W. Wang, M. Hu, X. Wang, G. Ma, and K. Ding, Experimental Realization of Geometry-Dependent Skin Effect in a Reciprocal Two-Dimensional Lattice, Phys. Rev. Lett. 131, 207201 (2023).

[5] X. Cui, R.-Y. Zhang, X. Wang, W. Wang, G. Ma, and C. T. Chan, Experimental Realization of Stable Exceptional Chains Protected by Non-Hermitian Latent Symmetries Unique to Mechanical Systems, Phys. Rev. Lett. 131, 237201 (2023).

[6] Z. Li, L.-W. Wang, X. Wang, Z.-K. Lin, G. Ma, and J.-H. Jiang, Observation of dynamic non-Hermitian skin effects, Nat Commun 15, 6544 (2024).

[7] N. Okuma, K. Kawabata, K. Shiozaki, and M. Sato, Topological Origin of Non-Hermitian Skin Effects, Phys. Rev. Lett. 124, 086801 (2020).
