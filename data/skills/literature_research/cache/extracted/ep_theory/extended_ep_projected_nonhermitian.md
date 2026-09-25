PAPER • OPEN ACCESS

# Extended exceptional points in projected non-Hermitian systems

To cite this article: Xiao-Ran Wang et al 2024 New J. Phys. 26 033040

View the article online for updates and enhancements.

You may also like

\- Encircling exceptional points of Bloch waves: mode conversion and anomalous scattering
Guy Elbaz, Adi Pick, Nimrod Moiseyev et al.

\- Local high chirality near exceptional points based on asymmetric backscattering
Jingnan Yang, Hancong Li, Sai Yan et al.

\- Non-local and non-Hermitian acoustic metasurfaces
Xu Wang, Ruizhi Dong, Yong Li et al.

# New Journal of Physics

The open access journal at the forefront of physics

Deutsche Physikalische Gesellschaft Φ DPG

IOP Institute of Physics

Published in partnership with: Deutsche Physikalische Gesellschaft and the Institute of Physics

![](images/93db439418dcd10a972435bd2d7d3c9a3b5d85d705881f536d397da63e21540b.jpg)

OPEN ACCESS

RECEIVED
8 December 2023

ACCEPTED FOR PUBLICATION
11 March 2024

REVISED
29 February 2024

PUBLISHED
22 March 2024

Original Content from this work may be used under the terms of the Creative Commons Attribution 4.0 licence.

Any further distribution of this work must maintain attribution to the author(s) and the title of the work, journal citation and DOI.

![](images/382e21078c8c63328df32f27a1e7d0c4f3570d817ad20fc09a3f5cbdd83f9542.jpg)

PAPER

# Extended exceptional points in projected non-Hermitian systems

Xiao-Ran Wang $^{1}$ , Fei Yang $^{2}$ $^{ID}$ , Xian-Qi Tong $^{2}$ , Xiao-Jie Yu $^{2}$ , Kui Cao $^{2}$ and Su-Peng Kou $^{2,*}$

$^{1}$ College of Teacher Education, Hebei Normal University, Shijiazhuang 050024, People's Republic of China

$^{2}$ Center for Advanced Quantum Studies, Department of Physics, Beijing Normal University, Beijing 100875, People's Republic of China
\* Author to whom any correspondence should be addressed.

E-mail: spkou@bnu.edu.cn

Keywords: topological insulator, non-Hermitian physics, exceptional points

## Abstract

Exceptional points are interesting physical phenomena in non-Hermitian physics at which the eigenvalues are degenerate and the eigenvectors coalesce. In this paper, we find that in projected non-Hermitian two-level systems (sub-systems under projecting partial Hilbert space) the singularities of exceptional points (EPs) is due to basis defectiveness rather than energy degeneracy or state coalescence. This leads to the discovery of extended exceptional points (EEPs). For EEPs, more subtle structures (e.g. the so-called Bloch peach), additional classification, and 'hidden' quantum phase transitions are explored. By using the topologically protected sub-space from two edge states in the non-Hermitian Su–Schrieffer–Heeger model as an example, we illustrate the physical properties of different types of EEPs.

## 1. Introduction

Exceptional point (EP) is a mathematical term introduced by Kato over half a century ago $[1]$ . In mathematics, EPs are branch point singularities of a spectrum and eigenfunctions for non-Hermitian matrices $[2, 3]$ . In the form of Jordan block matrix, at EPs the algebra multiplier of a matrix becomes larger than its geometric multiplier. Since the publication of the related paper by Bender and Boettcher $[4]$ , EPs have become one of the most interesting phenomena in non-Hermitian physics $[5, 6]$ . As a particular example, for non-Hermitian Hamiltonians with parity–time (PT) symmetry $[7–26]$ , spontaneous PT-symmetry breaking corresponds to a typical EP, at which the energy levels become degenerate and the eigenvectors coalesce. In experiments, the phenomenon of EPs has been realized and simulated using various approaches $[27–57]$ .

On the other hand, in some quantum many-body models, due to special conditions of symmetry/topology, there may exist protected sub-systems. For example, for topological insulators, there exist topologically protected edge states with gapless energy spectra (or zero modes for one dimensional cases) $[58, 59]$ ; For the many-body systems with spontaneously symmetry breaking there exist symmetry-protected degenerate ground states; for the topological orders with long range entanglement, there exist topologically protected degenerate ground states (on a torus) that make up topological qubits and may be possible to be applied to incorporate intrinsic fault tolerance into a quantum computer $[60–63]$ . For these topologically/symmetry protected sub-systems in different quantum many-body models, the quantum properties will be changed under non-Hermitian perturbations. In this paper, we investigate the influence of non-Hermitian perturbations on the quantum properties of quantum many-body systems through the topologically/symmetry protected sub-system. We find that basis defectiveness plays a key role in EPs and that in projected non-Hermitian systems there may exist EPs without eigenvalue degeneracy, EPs without the coalescence of different eigenvectors, or those without both features. This leads to the discovery of extended EPs based on the basis defectiveness.

The remainder of the paper is organized as follows. In section 2, we review the theory of singularity for usual EPs in a simple two-level systems and show the reason why eigenstates coalesce at an EP. In section 3, we discuss a general theory for projected non-Hermitian sub-spaces that are protected by

topologically/symmetry in different quantum many-body models and show how to derive the effective Hamiltonian and the corresponding (initial) basis. In section 4, we develop a theory of singularity for the extended EPs and show their complete classification. In section 5, an example of a one-dimensional (1D) non-Hermitian topological insulator—nonreciprocal Su–Schrieffer–Heeger (SSH) model is focused on. According to the global phase diagram of the topologically protected sub-systems from the two edge states, we show the occurrence of different types of extended EPs. Finally, conclusions are given in section 6.

## 2. Singularity at an EP in a traditional non-Hermitian two-level PT system

## 2.1. EP in a traditional non-Hermitian two-level PT system

To learn the nature of the singularity at an EP, we study a traditional two-level PT system described by the following Hamiltonian:

$$
\hat {H} _ {\mathrm{NH}} = h _ {x} \sigma_ {x} + i h _ {z} \sigma_ {z}.\tag{1}
$$

For this two-level non-Hermitian system, $\{|\psi_{0}^{R}\rangle\}=\{| \psi_{0}\rangle\}=\{| \uparrow \rangle, |\downarrow \rangle\}$ denotes an initial basis obeying orthogonal and normalization conditions, i.e. $\langle \uparrow | \downarrow \rangle = 0$ and $\langle \uparrow | \uparrow \rangle = \langle \downarrow | \downarrow \rangle = 1$ . At $h_{x} = h_{z}$ , a typical spontaneous PT-symmetry breaking occurs: for the case of $h_{x} > h_{z}$ , the energy levels $|+\rangle$ and $|- \rangle$ are $E_{\pm} = \pm \sqrt{h_{x}^{2} - h_{z}^{2}}$ ; For the case $h_{x} < h_{z}$ , these two energy levels are $E_{\pm} = \pm i \sqrt{h_{z}^{2} - h_{x}^{2}}$ ; For the case of $h_{x} = h_{z}$ , the system is at an EP with eigenstate coalescence and energy degeneracy. To strictly characterize the coalescence of the eigenstates, we define their state similarity, $\Lambda = |\langle \tilde{\psi}_{+}^{R} | \tilde{\psi}_{-}^{R} \rangle|$ , where $|\tilde{\psi}_{+}^{R}\rangle$ and $|\tilde{\psi}_{-}^{R}\rangle$ are two eigenstates of the system which satisfying the self-normalization conditions $|\langle \tilde{\psi}_{+}^{R} | \tilde{\psi}_{+}^{R}\rangle| = 1$ and $|\langle \tilde{\psi}_{-}^{R} | \tilde{\psi}_{-}^{R}\rangle| = 1$ . Figures 1(a) and (b) show the two energy levels ( $E_{+}$ and $E_{-}$ ) and the state similarity $\Lambda$ , respectively. It is obvious that both $E_{+} = E_{-}$ and $\Lambda = 1$ hold at the EP ( $h_{x} = h_{z}$ ).

## 2.2. Basis defectiveness with singular non-Hermitian similarity transformation

In this part, we solve the problem why eigenvectors coalesce at EPs. By performing a non-Hermitian similarity transformation $\widehat{S}_M = e^{-\beta^M\sigma_y}$ , we transform the original non-Hermitian Hamiltonian $\hat{H}_{\mathrm{NH}}$ into a Hermitian/anti-Hermitian one, $\hat{H}$ ( $\hat{H}_0^\dagger = \hat{H}_0$ or $\hat{H}_0^\dagger = -\hat{H}_0$ ), i.e.

$$
\hat {H} _ {0} = \widehat {\mathcal {S}} _ {M} ^ {- 1} \hat {H} _ {\mathrm{NH}} \widehat {\mathcal {S}} _ {M},\tag{2}
$$

where $\beta^{M} = \left|\frac{1}{2}\ln \left|\frac{h_{x} + h_{z}}{h_{x} - h_{z}}\right|\right|$ . The eigenvalue of $\hat{H}_0$ is the same as that of $\hat{H}_{\mathrm{NH}}$ . Under the non-Hermitian similarity transformation $\widehat{\mathcal{S}}_M$ , the basis $\{| \psi^{\mathrm{R}} \rangle\}$ of $\hat{H}_0$ becomes,

$$
\left\{\left| \psi^ {\mathrm{R}} \right\rangle \right\} = \left\{\widehat {\mathcal {S}} _ {M} \left(\sigma_ {y}, \beta^ {M}\right) \left| \psi_ {0} ^ {\mathrm{R}} \right\rangle \right\}.\tag{3}
$$

In the following, we refer to $\left\{|\psi^{R}\rangle\right\}$ as the matrix basis. If we select the initial basis $\left\{|\psi_{0}^{R}\rangle\right\}$ to be the eigenstates of $\sigma_{y}$ , i.e.

$$
\left\{\left| \psi_ {0} ^ {\mathrm{R}} \right\rangle \right\} = \left\{\left| \psi_ {0} \right\rangle \right\} = \left\{\left| 1 \right\rangle , \left| 2 \right\rangle \right\} = \left\{\frac {1}{\sqrt {2}} \left(\left| \uparrow \right\rangle + i | \downarrow \rangle\right), \frac {1}{\sqrt {2}} \left(\left| \uparrow \right\rangle - i | \downarrow \rangle\right) \right\},\tag{4}
$$

then the matrix basis $\{| \psi^{\mathrm{R}} \rangle\}$ becomes

$$
\left\{\left| \psi^ {\mathrm{R}} \right\rangle \right\} = \left\{e ^ {- \beta^ {M} \sigma_ {y}} \left| \psi_ {0} ^ {\mathrm{R}} \right\rangle \right\} = \left\{\left| 1 \right\rangle^ {\mathrm{R}}, \left| 2 \right\rangle^ {\mathrm{R}} \right\},\tag{5}
$$

where $|1\rangle^{\mathrm{R}} = |1\rangle$ and $|2\rangle^{\mathrm{R}} = e^{-\beta^{M}}|2\rangle$ . The matrix basis $\{|1\rangle^{\mathrm{R}},|2\rangle^{\mathrm{R}}\}$ obeys the orthogonal condition $^{\mathrm{R}}\langle 1|2\rangle^{\mathrm{R}} = 0$ but does not obey the normalization conditions, i.e. $^{\mathrm{R}}\langle 1|1\rangle^{\mathrm{R}} = 1$ and $^{\mathrm{R}}\langle 2|2\rangle^{\mathrm{R}} = e^{-2\beta^{M}}$ . Approaching the EP, the non-Hermitian similarity transformation becomes singular, i.e.

$$
\widehat {\mathcal {S}} _ {M} \left(\sigma_ {y}, \beta^ {M}\right) = e ^ {- \beta^ {M} \sigma_ {y}}\tag{6}
$$

with $\beta^M = \left|\frac{1}{2}\ln \left|\frac{h_x + h_z}{h_x - h_z}\right|\right| \to \infty$ . As a result, the matrix basis becomes defective, i.e.

$$
\left\{ \right.\left| \right. \psi^ {\mathrm{R}} \left. \right\rangle\left. \right\} = \left\{ \right.\widehat {\mathcal {S}} _ {M} \left(\sigma_ {y}, \beta^ {M} \rightarrow \infty\right)\left| \right. \psi_ {0} ^ {\mathrm{R}} \left. \right\rangle\left. \right\} = \left\{ \right.\left| \right. 1 \left. \right\rangle^ {\mathrm{R}}, \left| \right. 2 \left. \right\rangle^ {\mathrm{R}} \left. \right\}\rightarrow \left\{ \right.\left| \right. 1 \left. \right\rangle^ {\mathrm{R}}, 0 \left. \right\}.\tag{7}
$$

Here, the base $|2\rangle^{R}$ disappears! To illustrate the defectiveness of the matrix basis, in figure 1(c), we plot $\frac{1}{N}^{R}\langle1|1\rangle^{R}$ and $\frac{1}{N}^{R}\langle2|2\rangle^{R}$ , where $N=^{R}\langle1|1\rangle^{R}+^{R}\langle2|2\rangle^{R}$ is a normalization factor. Near the EP, $\frac{1}{N}^{R}\langle2|2\rangle^{R}$ becomes zero ( $^{R}\langle2|2\rangle^{R} \to 0$ ). As a result, according to the defective matrix basis without $|2\rangle^{R}$ , the two eigenstates $|\tilde{\psi}_{+}^{R}\rangle$ and $|\tilde{\psi}_{-}^{R}\rangle$ must coalesce into $|1\rangle$ , i.e.

![](images/dc36a5f850280665473cdc6760d921966334353bcffda0a9f4bb81d0592050a5.jpg)

![](images/9f9288b28811924b703c80268d40db8a256b44148e7048d470b6e40781d5566b.jpg)

![](images/0b9c01f8945f2dfa6b731f4f4e3545257ef38511fb4c09bedfe5eeaf3567a313.jpg)

![](images/38ee3c4c309cd717e1c990dca169fe0ea8ee9f6976aa5cd2495f5d68c266ddce.jpg)
(d)
Figure 1. Physical properties of the EP ( $h_x = h_z$ ) for the non-Hermitian two-level model $\hat{H}_{\mathrm{NH}} = h_x\sigma_x + ih_z\sigma_z$ . $h_x$ is set to unity. (a) The two energy levels, $E_{\pm}$ . At the EP ( $h_x = h_z$ ), these two energy levels become degenerate; (b) The state similarity $\Lambda = |\langle \tilde{\psi}_+^{\mathrm{R}}|\tilde{\psi}_-^{\mathrm{R}}\rangle|$ for two eigenstates $|\tilde{\psi}_+^{\mathrm{R}}\rangle$ and $|\tilde{\psi}_-^{\mathrm{R}}\rangle$ satisfying the self-normalization conditions $|\langle \tilde{\psi}_+^{\mathrm{R}}|\tilde{\psi}_+^{\mathrm{R}}\rangle| = 1$ and $|\langle \tilde{\psi}_-^{\mathrm{R}}|\tilde{\psi}_-^{\mathrm{R}}\rangle| = 1$ . At the EP ( $h_x = h_z$ ), we have $\Lambda = 1$ ; (c) The self-normalization of the matrix basis, $\frac{1}{\mathcal{N}}^{\mathrm{R}}\langle 1|1\rangle^{\mathrm{R}}$ and $\frac{1}{\mathcal{N}}^{\mathrm{R}}\langle 2|2\rangle^{\mathrm{R}}$ , where $\mathcal{N} = ^{\mathrm{R}}\langle 1|1\rangle^{\mathrm{R}} + ^{\mathrm{R}}\langle 2|2\rangle^{\mathrm{R}}$ . At the EP ( $h_x = h_z$ ), we have $\frac{1}{\mathcal{N}}^{\mathrm{R}}\langle 1|1\rangle^{\mathrm{R}} = 1$ and $\frac{1}{\mathcal{N}}^{\mathrm{R}}\langle 2|2\rangle^{\mathrm{R}} = 0$ ; (d) The Bloch peach for the quantum states $|\psi^{\mathrm{R}}\rangle = \cos \frac{\theta}{2}|1\rangle^{\mathrm{R}} + e^{i\phi}\sin \frac{\theta}{2}|2\rangle^{\mathrm{R}}$ at the EP ( $h_x = h_z$ ). Here, we have ${}^{\mathrm{R}}\langle 1|2\rangle^{\mathrm{R}} = 0$ , ${}^{\mathrm{R}}\langle 1|1\rangle^{\mathrm{R}} = 1$ , and ${}^{\mathrm{R}}\langle 2|2\rangle^{\mathrm{R}} = e^{-2\beta^{M}}$ .

$$
\left| \right. \tilde {\psi} _ {+} ^ {R} \left. \right\rangle = \frac {1}{\sqrt {1 + e ^ {- 2 \beta^ {M}}}} \left( \right.\left| \right. 1 \left. \right\rangle^ {R} + \left| \right. 2 \left. \right\rangle^ {R}\left. \right)\rightarrow \frac {1}{\sqrt {1 + e ^ {- 2 \beta^ {M}}}} | 1 \rangle\tag{8}
$$

and

$$
\left| \right. \tilde {\psi} _ {-} ^ {R} \left. \right\rangle = \frac {1}{\sqrt {1 + e ^ {- 2 \beta^ {M}}}} \left( \right.\left| \right. 1 \left. \right\rangle^ {R} - \left| \right. 2 \left. \right\rangle^ {R}\left. \right)\rightarrow \frac {1}{\sqrt {1 + e ^ {- 2 \beta^ {M}}}} | 1 \rangle .\tag{9}
$$

Now, near the EP, where $\beta^{M} \to \infty$ , the state similarity $\Lambda = |\langle \tilde{\psi}_{+}^{\mathrm{R}}|\tilde{\psi}_{-}^{\mathrm{R}}\rangle| = \frac{1}{1 + e^{-2\beta M}}$ obviously approaches 1, which is consistent with figure 1(b). Thus, one can see that the singularity of the EP arises from the defective matrix basis $\{|1\rangle^{\mathrm{R}},0\}$ due to the singular non-Hermitian similarity transformation $\widehat{\mathcal{S}}_M(\sigma_y,\beta^M\to \infty)$ .

## 2.3. Bloch peach for two-level states under the non-Hermitian similarity transformation

To further illustrate the singularity of EPs, we use a geometric approach to illustrate the deformation of the Bloch sphere under a singular non-Hermitian similarity transformation $\widehat{\mathcal{S}}_M(\sigma_y,\beta^M\to \infty)$ . For the Hermitian case, one may use a point on the Bloch sphere with SU(2) rotation symmetry to represent an arbitrary quantum state of the two-level system, i.e.

$$
| \psi \rangle = \cos \frac {\theta}{2} | 1 \rangle + e ^ {i \varphi} \sin \frac {\theta}{2} | 2 \rangle .\tag{10}
$$

Here, $\theta \in [0,\pi]$ and $\varphi \in [0,2\pi]$ are real numbers, and the radius $r$ of the Bloch sphere can be obtained as $r = \langle \psi |\psi \rangle = 1$ . For a two-level $\mathcal{PT}$ system, a quantum state under the non-Hermitian similarity transformation $\widehat{\mathcal{S}}_M(\sigma_y,\beta^M)$ becomes

$$
| \psi^ {\mathrm{R}} \rangle = \widehat {\mathcal {S}} _ {M} | \psi \rangle = \cos \frac {\theta}{2} | 1 \rangle + e ^ {i \varphi} e ^ {- \beta^ {M}} \sin \frac {\theta}{2} | 2 \rangle\tag{11}
$$

(with $\theta \in [0,\pi]$ and $\varphi \in [0,2\pi]$ ), and the radius $R$ of the Bloch sphere can be obtained as

$$
R = \langle \psi^ {\mathrm{R}} | \psi^ {\mathrm{R}} \rangle = \cos^ {2} \frac {\theta}{2} + e ^ {- 2 \beta^ {M}} \sin^ {2} \frac {\theta}{2}.\tag{12}
$$

Therefore, the original Bloch sphere changes into a peach-like closed surface with residue U(1) symmetry along the y-axis (we call this the Bloch peach). With increasing $\beta$ , one pole of the Bloch peach moves upward. At the EP, this pole touches the origin of the coordinate system. See the illustration of the Bloch peach in the limit of $\beta^{M} \rightarrow \infty$ in figure 1(d).

## 3. Projected non-Hermitian sub-systems

For a many-body system, due to special conditions of symmetry/topology, there may exist projected sub-systems, such as the subsystem formed by edge states in topological insulators and subsystem formed by topologically protected degenerate ground states in intrinsic topological orders. Through these projected sub-systems that are projected by symmetry/topology, one can study the quantum properties of many-body systems more concisely and intuitively. To completely characterize the many-body model and its projected sub-systems, we give their definitions, $\{\hat{H}_{MB}, B_{MB}\}$ and $\{\hat{H}_{S}, B_{S}\}$ , respectively. For the Hermitian case, both $B_{MB} = \{|\Psi_{j}\rangle, j = 1, 2, \ldots, N\}$ and $B_{S} = \{\mathcal{P}|\Psi_{j}\rangle, j = 1, 2, \ldots, N\} = \{|S_{j}\rangle, j = 1, 2, \ldots, K\}$ are normal basis with $B_{S} = P B_{MB} \in B_{MB}$ where P is a projective operator on the basis of the many-body system. It is obvious that K < N. When one adds a non-Hermitian perturbation $i \delta \hat{H}$ on the many-body model, the total Hamiltonian becomes non-Hermitian, i.e.

$$
\hat {H} _ {\mathrm{MB}} \rightarrow \hat {H} _ {\mathrm{NH-MB}} = \hat {H} _ {\mathrm{MB}} + i \delta \hat {H}.\tag{13}
$$

Under the non-Hermitian perturbation, the basis has no changing, the many-body model is denoted by $\{\hat{H}_{NH-MB}, B_{MB}\}$ . In general, for a non-Hermitian system, the biorthogonal set for the basis is defined by $|\Psi_{j}^{R}\rangle$ and $|\Psi_{j}^{L}\rangle$ ( $j = 1, 2, \ldots, N$ ), i.e.

$$
\hat {H} _ {\mathrm{NH-MB}} | \Psi_ {j} ^ {\mathrm{R}} \rangle = E _ {j} | \Psi_ {j} ^ {\mathrm{R}} \rangle ,\tag{14}
$$

and

$$
\hat {H} _ {\mathrm{NH-MB}} ^ {\dagger} | \Psi_ {j} ^ {\mathrm{L}} \rangle = (E _ {j}) ^ {*} | \Psi_ {j} ^ {\mathrm{L}} \rangle ,\tag{15}
$$

and $\langle\Psi_{j}^{L}|\Psi_{j}^{R}\rangle=1$ where j is state index. We then assume the non-Hermitian terms are perturbation and do not change the existence of the projected sub-system and $\{\hat{H}_{NH-S},B_{S}\}=\{\mathcal{P}\hat{H}_{MB}\mathcal{P}^{-1},\mathcal{P}B_{MB}\}$ is used to describe the projected non-Hermitian sub-system. Here, $\hat{H}_{NH-S}=\mathcal{P}\hat{H}_{MB}\mathcal{P}^{-1}$ is the effective Hamiltonian of the protected sub-system. Under the projected operation P, the basis of the projected sub-system is obtained as

$$
B _ {\mathrm{S}} = \mathcal {P} B _ {\mathrm{MB}} = \left\{\mathcal {P} | \Psi_ {j} ^ {\mathrm{R}} \rangle \right\} = \left\{| \psi_ {j} ^ {\mathrm{R}} \rangle \right\}.\tag{16}
$$

The basis of projected sub-system $B_{\mathrm{S}} = \{| \psi_j^{\mathrm{R}} \rangle \}$ is always abnormal under the projection $\mathcal{P}$ of the non-Hermitian system. For example, it does't follow usual normalization $\langle \psi_j^{\mathrm{R}} | \psi_j^{\mathrm{R}} \rangle \neq 1$ . In addition, we show the method to calculate $\hat{H}_{\mathrm{NH-S}}$ . Based on the basis $B_{\mathrm{S}} = \{| \psi_j^{\mathrm{R}} \rangle \}$ , an effective Hamiltonian of projected sub-system is derived as

$$
\hat {H} _ {\mathrm{NH-S}} = \sum_ {i j} h _ {I J}\tag{17}
$$

where

$$
h _ {I J} = \left\langle \psi_ {j} ^ {\mathrm{L}} \right| \hat {H} _ {\mathrm{NH-MB}} \left| \psi_ {j} ^ {\mathrm{R}} \right\rangle , I, J = 1, 2, \dots , K.\tag{18}
$$

In this paper, We consider the projected two-level sub-system of pseudo-Hermitian systems that can perform the similarity transformation $[24]$ . So, we only focus on the case of K=2, the dimension of the sub-system is 2, shown as figure 2.

## 4. Extended EPs—universal feature and classification in projected non-Hermitian systems

We next develop the theory for the projected non-Hermitian systems with a singularity and introduce the concept of extended EPs. This phenomenon always occurs in subsystems of certain non-Hermitian models, for example, the defective edge states of a non-Hermitian topological insulator, the defective degenerate ground states in non-Hermitian systems with spontaneous symmetry breaking, or the topologically

protected degenerate ground states in intrinsic topological orders. An arbitrarily non-Hermitian protected two-level sub-system can be described as $\{\hat{H}_{\mathrm{NH}},\{\left|\psi_{0}^{\mathrm{R}}\right\rangle\}\}$ . The Hamiltonian is

$$
\hat {H} _ {\mathrm{NH}} = h _ {0} + \vec {h} \cdot \vec {\sigma},\tag{19}
$$

where $h_{0}$ is a complex number, $\vec{h} = (h^{x}, h^{y}, h^{z})$ is a complex vector and $\vec{\sigma} = (\hat{\sigma}_{x}, \hat{\sigma}_{y}, \hat{\sigma}_{z})$ is the vector of Pauli matrices. For Hermitian systems, the Hamiltonian itself can fully describe the energy level structure and wave function of the system, but in non-Hermitian systems, the combination of the Hamiltonian and the system basis vector can fully describe the energy level structure and corresponding wave function of the system, so we use terms like $\{\hat{H}_{\mathrm{NH}}, \{|\psi_{0}^{\mathrm{R}}\rangle\}\}$ to describe the non-Hermitian two-level sub-system (the same goes for $\{\hat{H}_{\mathrm{NH-MB}}, B_{\mathrm{MB}}\}$ and $\{\hat{H}_{\mathrm{NH-S}}, B_{S}\}$ mentioned earlier).

## 4.1. Three equivalent basis representations

For non-Hermitian two-level systems mentioned above, we found that there are three representations that can equivalently describe the system.

## 4.1.1. Initial basis representation

Shown as figure 2, the basis $\{|\psi_0^{\mathrm{R}}\}$ of the (projected) sub-system $\hat{H}_{\mathrm{NH}}$ is always abnormal under the projective operator $\mathcal{P}$ from the non-Hermitian system to its sub-system. At the same time, $\hat{H}_{\mathrm{NH}}$ is a non-Hermitian Hamiltonian. For this non-Hermitian system, we can just use the combination of the Hamiltonian $\hat{H}_{\mathrm{NH}}$ and the basis $\{|\psi_0^{\mathrm{R}}\}$ to directly describe this two-level system. We define $\{\hat{H}_{\mathrm{NH}},\{| \psi_0^{\mathrm{R}}\} \}$ as the initial basis representation.

## 4.1.2. Matrix basis representation

In this part, we transform the non-Hermitian Hamiltonian $\hat{H}_{NH}$ into a Hermitian/anti-Hermitian one with the similarity transformation and derive the Matrix basis representation that can equivalently describe the non-Hermitian sub-system. To transform $\hat{H}_{NH}$ into a Hermitian/anti-Hermitian one, Firstly, we have to rewrite $\hat{H}_{NH} = h_{0} + \vec{h} \cdot \vec{\sigma}$ into

$$
\hat {H} _ {\mathrm{NH}} = h _ {0} + (\operatorname{Re} \vec {h}) \vec {\sigma} _ {\mathrm{Re}} + i (\operatorname{Im} \vec {h}) \vec {\sigma} _ {I m},\tag{20}
$$

where $\vec{\sigma}_{\mathrm{Re}}$ and $\vec{\sigma}_{\mathrm{Im}}$ ( $\vec{\sigma}_{\mathrm{Re}} \cdot \vec{\sigma}_{\mathrm{Re}} = 1$ and $\vec{\sigma}_{\mathrm{Im}} \cdot \vec{\sigma}_{\mathrm{Im}} = 1$ ) are the Pauli matrices corresponding to the real and imaginary parts of $\vec{h}$ , respectively. And secondly, we divide $i(\mathrm{Im}\vec{h})\vec{\sigma}_{\mathrm{Im}}$ into two parts, $i(\mathrm{Im}\vec{h})^A\vec{\sigma}_{\mathrm{Im}}^A$ and $i(\mathrm{Im}\vec{h})^C\vec{\sigma}_{\mathrm{Im}}^C$ , with $[\vec{\sigma}_{\mathrm{Im}}^C, \vec{\sigma}_{\mathrm{Re}}] = 0$ and $\{\vec{\sigma}_{\mathrm{Im}}^A, \vec{\sigma}_{\mathrm{Re}}\} = 0$ . Then, the non-Hermitian Hamiltonian becomes

$$
\hat {H} _ {\mathrm{NH}} = h _ {0} + \vec {h} _ {\mathrm{Re}} ^ {\prime} \cdot \vec {\sigma} _ {\mathrm{Re}} + i \vec {h} _ {\mathrm{Im}} ^ {\prime} \cdot \vec {\sigma} _ {\mathrm{Im}} ^ {A},\tag{21}
$$

where $\vec{h}_{\mathrm{Re}}^{\prime} = (\mathrm{Re}\vec{h}) + i(\mathrm{Im}\vec{h})^{C}$ and $\vec{h}_{\mathrm{Im}}^{\prime} = (\mathrm{Im}\vec{h})^{A}$ . Then, we can transform the original NH Hamiltonian $\hat{H}_{\mathrm{NH}}$ into a Hermitian/anti-Hermitian one with a non-Hermitian similarity transformation $\widehat{\mathcal{S}}_M(\vec{\sigma}^M,\beta^M)$ ,

$$
\hat {H} _ {0} = \widehat {\mathcal {S}} _ {M} (\vec {\sigma} ^ {M}, \beta^ {M}) \hat {H} _ {\mathrm{NH}} \widehat {\mathcal {S}} _ {M} ^ {- 1} (\vec {\sigma} ^ {M}, \beta^ {M}) = h _ {0} + \sqrt {\vec {h} ^ {2}} \vec {\sigma} _ {\mathrm{Re}},\tag{22}
$$

where

$$
\widehat {\mathcal {S}} _ {M} \left(\vec {\sigma} ^ {M}, \beta^ {M}\right) = e ^ {- \beta^ {M} \cdot \vec {\sigma} ^ {M}}\tag{23}
$$

with $\vec{\sigma}^M = \frac{1}{2i} [\vec{\sigma}_{\mathrm{Im}}^A, \vec{\sigma}_{\mathrm{Re}}]$ and $\beta^M = \left|\frac{1}{2}\ln \frac{|\vec{h}_{\mathrm{Re}}' + \vec{h}_{\mathrm{Im}}'|}{|\vec{h}_{\mathrm{Re}}' - \vec{h}_{\mathrm{Im}}'|}\right|$ . As a result, the eigenvalue of $\hat{H}_0$ is the same as that of $\hat{H}_{\mathrm{NH}}$ , i.e.

$$
E _ {\pm} = h _ {0} \pm \sqrt {\vec {h} ^ {2}}.\tag{24}
$$

For the case of $\vec{h}^{2}>0$ , $\hat{H}_{0}$ is a Hermitian Hamiltonian, $\hat{H}_{0}=\hat{H}_{0}^{\dagger}$ , whose energy levels $E_{\pm}=h_{0}\pm\left|\vec{h}\right|$ are real; for the case of $\vec{h}^{2}<0$ , $\hat{H}_{0}$ is an anti-Hermitian Hamiltonian, $\hat{H}_{0}=-\hat{H}_{0}^{\dagger}$ , whose energy levels $E_{\pm}=h_{0}\pm i\left|\vec{h}\right|$ are an imaginary pair.

Finally, one can see that under the non-Hermitian similarity transformation, the initial basis $\{|\psi_{0}^{R}\rangle\}$ is correspondingly changed into the unique matrix basis $\{|\psi^{R}\rangle\}$ , i.e.

$$
\left\{\left| \psi^ {\mathrm{R}} \right\rangle \right\} = \left\{\widehat {\mathcal {S}} _ {M} \left(\vec {\sigma} ^ {B}, \beta^ {B}\right) \left| \psi_ {0} ^ {\mathrm{R}} \right\rangle \right\}.\tag{25}
$$

So, we can use the combination of the Hamiltonian $\hat{H}_0$ ( $\hat{H}_0$ is a Hermitian/anti-Hermitian Hamiltonian) and the basis $\{| \psi^R \rangle (\{| \psi_0^R \rangle \text{ is abnormal})\}$ to directly describe this two-level system equivalently as the initial basis representation. We define $\{\hat{H}_0, \{| \psi^R \rangle \}\}$ as the initial basis representation.

![](images/7f292ed5a3753b6718504cf1b0a739da096e468a5ff4c70d2519d08045f10d25.jpg)
Figure 2. Schematic diagram of the relationship among three equivalent representations for a projected non-Hermitian two-level systems: $\{\hat{H}_{\mathrm{NH}},\{| \psi_0^{\mathrm{R}}\} \}$ , under the initial basis; $\{\hat{H}_0,\{| \psi^R\rangle \} \}$ , under the matrix basis; and $\{\hat{H}_{\mathrm{NH}}^{\beta^B},\{| \psi_0\rangle \} \}$ , under the normal basis. Here, $\hat{H}_{\mathrm{NH}}$ is the original non-Hermitian Hamiltonian with initial basis $\{| \psi_0^{\mathrm{R}}\}$ . $\{\hat{H}_0,\{| \psi^R\rangle \} \}$ with $\{| \psi^R\rangle \} = \{\widehat{\mathcal{S}}_M|\psi_0^{\mathrm{R}}\} = \{\widehat{\mathcal{S}}_M\widehat{\mathcal{S}}_B|\psi_0\rangle \}$ classifies different types of EEPs (M-EEPs, B-EEPs, and H-EEPs). Under the normal basis $\{\hat{H}_{\mathrm{NH}}^{\beta^B},\{| \psi_0\rangle \} \}$ , we classifies the subclasses of B-EEPs.

## 4.1.3. Normal basis representation

In addition, to describe the same non-Hermitian two-level system, one can also use another representation,

$$
\hat {H} _ {\mathrm{NH}} ^ {\beta^ {B}} = \widehat {\mathcal {S}} _ {B} ^ {- 1} \hat {H} _ {\mathrm{NH}} \widehat {\mathcal {S}} _ {B}, \{| \psi_ {0} \rangle \} = \left\{\widehat {\mathcal {S}} _ {B} ^ {- 1} | \psi_ {0} ^ {\mathrm{R}} \rangle \right\}\tag{26}
$$

with $\widehat{S}_{B}=e^{-\beta^{B}\cdot\vec{\sigma}^{B}}$ . The Hamiltonian $\hat{H}_{NH}^{\beta^{B}}$ is a non-Hermitian one, and the $|\psi_{0}\rangle$ is a normal basis consisting of eigenstates of $\vec{\sigma}^{B}$ (or $\vec{\sigma}^{B}|\psi_{0}\rangle=\pm|\psi_{0}\rangle$ ) based on the normal basis, so the two-level system can also equivalently describe by the normal basis representation $\{\hat{H}_{NH}^{\beta^{B}},\{| \psi_{0}\rangle \}$ . Besides, we can associate the normal basis representation with the matrix basis representation through similarity transformation,

$$
\left\{\left| \psi^ {\mathrm{R}} \right\rangle \right\} = \left\{\widehat {\mathcal {S}} _ {M} \left(\vec {\sigma} ^ {M}, \beta^ {M}\right) \widehat {\mathcal {S}} _ {B} \left(\vec {\sigma} ^ {B}, \beta^ {B}\right) \left| \psi_ {0} \right\rangle \right\}.\tag{27}
$$

To clearly show the relationship among the three representations for the Hamiltonians of the same non-Hermitian two-level system, $\{\hat{H}_{\mathrm{NH}},\{| \psi_0^R\rangle \} \}$ under the initial basis, $\{\hat{H}_0,\{| \psi^R\rangle \} \} \}$ under the matrix basis, and $\{\hat{H}_{\mathrm{NH}}^{\beta^B},\{| \psi_0\rangle \} \}$ under the normal basis, we plot figure 2. $\widehat{\mathcal{S}}_M,\widehat{\mathcal{S}}_B$ , and $\widehat{\mathcal{S}}_M\widehat{\mathcal{S}}_B$ are different non-Hermitian similarity transformations relating these representations.

## 4.2. Definition of extended EPs

We point out that the universal feature of the projected non-Hermitian system with singularities is defectiveness of the matrix basis $\{|\psi^{R}\rangle\}$ as $\beta^{M}\to\infty$ and/or $\beta^{B}\to\infty$ rather than energy degeneracy or state coalescence. Thus, we define extended EPs as follows: Definition—extended EPs (EEPs): For an arbitrary projected two-level non-Hermitian systems system, EEPs exist if and only if the matrix basis $\{|\psi^{R}\rangle\}=\{|1\rangle^{R},|2\rangle^{R}\}$ becomes defective, i.e. $\frac{1}{N}^{R}\langle1|1\rangle^{R}=1$ and $\mathrm{R}\langle2|2\rangle^{R}=0$ or $\frac{1}{N}^{R}\langle2|2\rangle^{R}=1$ and $\mathrm{R}\langle1|1\rangle^{R}=0$ . Here, $N=^{R}\langle1|1\rangle^{R}+^{R}\langle2|2\rangle^{R}$ is a normalization factor. As a result, at EEPs, the two energy levels may not necessarily be degenerate, i.e. $E_{+}=E_{-}$ and $E_{+}\neq E_{-}$ are both allowed; the two eigenstates $\left|\tilde{\psi}_{+}^{R}\right\rangle$ and $\left|\tilde{\psi}_{-}^{R}\right\rangle$ may not necessarily coalesce, i.e. $\Lambda\equiv1$ and $\Lambda\equiv0$ are both allowed. In addition, in the form of Jordan block matrix, at EEPs the algebra multiplier of a matrix may be same to its geometric multiplier.

## 4.3. Classification of extended EPs

By considering different behaviors of defective bases, we can classify EEPs for an arbitrary projected non-Hermitian two-level system. Depending on the behavior of the defective matrix basis $\{|\psi^{R}\rangle\}=\{\widehat{\mathcal{S}}_{M}(\vec{\sigma}^{M},\beta^{M})\widehat{\mathcal{S}}_{B}(\vec{\sigma}^{B},\beta^{B})|\psi_{0}\rangle\}$ , there are three types of EEPs: matrix-type EEPs (M-EEPs), with $\beta^{M}\to\infty$ and $\beta^{B}\not\to\infty$ ; basis-type EEPs (B-EEPs), with $\beta^{M}\not\to\infty$ and $\beta^{B}\to\infty$ ; and hybrid-type EEPs (H-EEPs), with $\beta^{M}\to\infty$ and $\beta^{B}\to\infty$ . In addition, there are two different classes of B-EEPs, i.e. IB-EEPs and IIB-EEPs. For IB-EEPs, the two eigenstates will never coalesce with each other $\Lambda \equiv 0$ ; for IIB-EEPs, the two eigenstates will always coalesce with each other $\Lambda \equiv 1$ , shown as the table 1.

Table 1. Three types of EEP definitions under normal basis representation.

<table><tr><td rowspan="2"></td><td rowspan="2">M-EEP</td><td colspan="2">B-EEP</td><td rowspan="2">H-EEP</td></tr><tr><td>Class I</td><td>Class II</td></tr><tr><td>Definition</td><td> $\beta^{M} \to \infty, \beta^{B} \not\to \infty$ </td><td> $\beta^{M} \not\to \infty, \beta^{B} \to \infty$ </td><td> $\beta^{M} \not\to \infty, \beta^{B} \to \infty$ </td><td> $\beta^{M} \to \infty, \beta^{B} \to \infty$ </td></tr><tr><td>State similarity</td><td> $\Lambda = 1$ </td><td> $\Lambda = 0$ </td><td> $\Lambda = 1$ </td><td> $\Lambda = 1$ </td></tr><tr><td>Energy degeneracy</td><td> $E_{+} = E_{-}$ </td><td> $E_{+} \neq E_{-}$ </td><td> $E_{+} \neq E_{-}$ </td><td> $E_{+} = E_{-}$ </td></tr></table>

To show the physical properties of these two classes of B-EEPs, we perform a non-Hermitian similarity transformation $\widehat{S}_{B}^{-1}$ on the initial basis $\{|\psi_{0}^{R}\rangle\}$ and obtain a representation under the normal basis, i.e.

$$
\left\{\left| \psi_ {0} \right\rangle \right\} = \left\{\widehat {\mathcal {S}} _ {B} ^ {- 1} \left| \psi_ {0} ^ {\mathrm{R}} \right\rangle \right\}.\tag{28}
$$

Correspondingly, the original Hamiltonian $\hat{H}_{NH}$ is transformed into

$$
\hat {H} _ {\mathrm{NH}} ^ {\beta^ {B}} = \widehat {\mathcal {S}} _ {B} ^ {- 1} \hat {H} _ {\mathrm{NH}} \widehat {\mathcal {S}} _ {B}.\tag{29}
$$

There are two possibilities for $\hat{H}_{\mathrm{NH}}^{\beta^B}$ , which correspond to the two classes of B-EEPs: one possibility is that all elements of $\hat{H}_{\mathrm{NH}}^{\beta^B}$ are finite, in which case the eigenstates do not coalesce (or $\Lambda \equiv 0$ ), and the other is one or more elements diverge, i.e. $(\hat{H}_{\mathrm{NH}}^{\beta^B})_{ij} \to \infty$ , in which case the eigenstates coalesce (or $\Lambda \equiv 1$ ). It is obvious that at IB-EEPs the algebra multiplier of the Hamiltonian $\hat{H}_{\mathrm{NH}}$ is equal to its geometric multiplier. Therefore, there exists a quantum phase transition between IB-EEPs (the region without eigenstate coalescence) and IIB-EEPs (the region with eigenstate coalescence). Let us give a simple explanation of this fact. For IB-EEPs, the Hamiltonian $\hat{H}_{\mathrm{NH}}$ commutes with $\vec{\sigma}^B$ , i.e. $[\hat{H}_{\mathrm{NH}}, \vec{\sigma}^B] = 0$ . As a result, the Hamiltonian $\hat{H}_{\mathrm{NH}}$ must be written as $\lambda \vec{\sigma}^B$ with $\lambda \neq 0$ . Under the non-Hermitian similarity transformation $\widehat{\mathcal{S}}_B^{-1} = e^{\beta^B \vec{\sigma}^B}$ , we have

$$
\hat {H} _ {\mathrm{NH}} ^ {\beta^ {B}} = \widehat {\mathcal {S}} _ {B} ^ {- 1} \hat {H} _ {\mathrm{NH}} \widehat {\mathcal {S}} _ {B} = \lambda \vec {\sigma} ^ {B}.\tag{30}
$$

Now, the basis of $\hat{H}_{\mathrm{NH}}^{\beta^B}$ is normal basis $\{| \psi_0 \rangle = \{\widehat{S}_B^{-1} | \psi_0^{\mathrm{R}} \rangle\}$ . Here, the $|\psi_0\rangle$ are eigenstates of $\vec{\sigma}^B$ , or $\vec{\sigma}^B |\psi_0\rangle = \pm |\psi_0\rangle$ . Because the eigenstates $|\psi_{\pm}^{\mathrm{R}}\rangle$ of $\hat{H}_{\mathrm{NH}} = \lambda \vec{\sigma}^B$ are also those of $|\psi_0\rangle$ , the state similarity of $|\psi_{\pm}^{\mathrm{R}}\rangle$ must be zero, i.e.

$$
\Lambda = | \langle \tilde {\psi} _ {+} ^ {\mathrm{R}} | \tilde {\psi} _ {-} ^ {\mathrm{R}} \rangle | = 0.\tag{31}
$$

On the other hand, for IIB-EEPs, the Hamiltonian $\hat{H}_{NH}$ does not commute with $\vec{\sigma}^{B}$ , i.e. $[\hat{H}_{NH},\vec{\sigma}^{B}]\neq0$ . The Hamiltonian $\hat{H}_{NH}$ must be written as

$$
\hat {H} _ {\mathrm{NH}} = \lambda \vec {\sigma} ^ {B} + \eta (\vec {\sigma} ^ {B}) ^ {\perp}\tag{32}
$$

with $\lambda^2 +\eta^2\neq 0$ and $\{(\vec{\sigma}^B)^\perp ,\vec{\sigma}^B\} = 0$ . Now, one element of $\hat{H}_{\mathrm{NH}}^{\beta^B} = \widehat{\mathcal{S}}_B^{-1}\hat{H}_{\mathrm{NH}}\widehat{\mathcal{S}}_B$ diverges. On the basis of $\{| \psi_0\rangle \}$ , the divergent term is proportional to (0 1) or (0 0). As a result, the Hamiltonian $\hat{H}_{\mathrm{NH}}^{\beta^B} = \widehat{\mathcal{S}}_B^{-1}\hat{H}_{\mathrm{NH}}\widehat{\mathcal{S}}_B$ is dominated by this divergent term, and we can ignore other terms. In this case, the state similarity of $|\psi_{\pm}^{\mathrm{R}}\rangle$ must be 1, i.e.

$$
\Lambda = | \langle \tilde {\psi} _ {+} ^ {\mathrm{R}} | \tilde {\psi} _ {-} ^ {\mathrm{R}} \rangle | = 1.\tag{33}
$$

Without sudden changing the energy levels and the defectiveness of matrix basis, this quantum phase transition is always ‘hidden’.

## 5. Example: 1D nonreciprocal SSH model

## 5.1. The model

In this section, we take the 1D nonreciprocal SSH model as an example to illustrate the different types of EEPs for its topologically protected sub-space of two edge states. The Bloch Hamiltonian for the nonreciprocal SSH model under periodic boundary conditions (PBC) is given by

$$
\hat {H} _ {\mathrm{PBC}} (k) = \sum_ {k} c _ {k} ^ {\dagger} \tau_ {x} \left(t _ {1} + t _ {2} \cos k\right) c _ {k} + \sum_ {k} c _ {k} ^ {\dagger} \tau_ {y} \left(t _ {2} \sin k + i \gamma\right) c _ {k} + i \varepsilon \sum_ {k} c _ {k} ^ {\dagger} \tau_ {z} c _ {k},\tag{34}
$$

where $c_{k}^{\dagger} = (c_{k,\mathrm{A}}^{\dagger}, c_{k,\mathrm{B}}^{\dagger})$ ; the $\tau_{i}$ are the Pauli matrices acting on the (A or B) sublattice subspaces; $t_{1}$ and $t_{2}$ describe the intracell and intercell hopping strengths, respectively; $\gamma$ describes unequal intracell hopping; and $\varepsilon$ denotes the strength of an imaginary staggered potential on the two sublattices. $t_{1}, t_{2}, \gamma$ , and $\varepsilon$ are all real. In this paper, we set $t_{2} = 1$ .

Under a non-Hermitian similarity transformation $\widehat{S}_{\mathrm{NHP}}$ , the physics properties of the 1D nonreciprocal SSH model under open boundary conditions (OBC) are characterized by $\hat{H}_{\mathrm{OBC}}(k)$ rather than $\hat{H}_{\mathrm{PBC}}(k)$ [13]. Here, the non-Hermitian similarity transformation $\widehat{S}_{\mathrm{NHP}}$ given by

$$
c _ {k} ^ {\dagger} \rightarrow \tilde {c} _ {k} ^ {\dagger} = c _ {k - i q _ {0}} ^ {\dagger} = \widehat {\mathcal {S}} _ {\mathrm{NHP}} c _ {k} ^ {\dagger}\tag{35}
$$

or

$$
c _ {n} ^ {\dagger} \rightarrow \tilde {c} _ {n} ^ {\dagger} = e ^ {- q _ {0} (n - 1)} c _ {n} ^ {\dagger} = \widehat {\mathcal {S}} _ {\mathrm{NHP}} c _ {n} ^ {\dagger}\tag{36}
$$

where $e^{q_0} = \sqrt{\frac{t_1 - \gamma}{t_1 + \gamma}}$ . Consequently, the effective hopping parameters become $\bar{t}_1 = \sqrt{(t_1 + \gamma)(t_1 - \gamma)}$ and $\bar{t}_2 = t_2$ .

To characterize the topological properties of the non-Hermitian topological system, the non-Bloch topological invariant $\bar{w}$ of $\hat{H}_{\mathrm{OBC}}(k)$ is introduced, i.e.

$$
\bar {w} = \frac {1}{2 \pi} \int_ {- \pi} ^ {\pi} \partial \bar {\phi} (k) d k\tag{37}
$$

where $\bar{\phi}(k)=\tan^{-1}(\bar{h}_{y}/\bar{h}_{x})$ and $\bar{h}_{x}=\bar{t}_{1}+\bar{t}_{2}\cos k,\bar{h}_{y}=\bar{t}_{2}\sin k$ . In the region of $|\bar{t}_{1}|<|\bar{t}_{2}|$ and $\bar{w}=1$ , the system is a topological insulator (the gray region in figure 3(a)); in the region of $|\bar{t}_{1}|>|\bar{t}_{2}|$ and $\bar{w}=0$ , the system is a normal insulator (the white region in figure 3(a)). A quantum phase transition occurs at $|\bar{t}_{1}|=|\bar{t}_{2}|$ , where the bulk energy gap under OBC is closed.

## 5.2. Two-level systems from two edge states in topological phase

In the topological phase with $\bar{w}=1$ , there exist two edge states $\left|\psi_{1}^{R}\right\rangle$ and $\left|\psi_{2}^{R}\right\rangle$ . These two edge states make up a topologically protected subspace denoted by $\{\hat{H}_{\mathrm{NH}},\left\{\left|\psi_{0}^{\mathrm{R}}\right\rangle\right\}\}$ . Under the biorthogonal set, the initial basis $\left\{\left|\psi_{0}^{\mathrm{R}}\right\rangle\right\}$ is $\left\{\left|\psi_{1}^{\mathrm{R}}\right\rangle,\left|\psi_{2}^{\mathrm{R}}\right\rangle\right\}$ . The effective Hamiltonian $\hat{H}_{NH}$ is written as

$$
\hat {H} _ {\mathrm{NH}} = \left( \begin{array}{c c} h _ {1 1} & h _ {1 2} \\ h _ {2 1} & h _ {2 2} \end{array} \right),\tag{38}
$$

where $h_{ij} = \left\langle \psi_i^{\mathrm{L}}\right|\hat{H}\left|\psi_j^{\mathrm{R}}\right\rangle, i,j = 1,2$ . Through straightforward calculations [18], we can analytically obtain the effective Hamiltonian of the two edge states as

$$
\hat {H} _ {\mathrm{NH}} = \bar {\Delta} \sigma^ {x} + i \varepsilon \sigma^ {z},\tag{39}
$$

where $\bar{\Delta} = \frac{(t_2^2 - t_1^2)}{\bar{t}_2} (\frac{\bar{t}_1}{\bar{t}_2})^N$ . The two energy levels for the two eigenstates $|\psi_{+}^{\mathrm{R}}\rangle$ and $|\psi_{-}^{\mathrm{R}}\rangle$ are $E = \pm \sqrt{\bar{\Delta}^2 - \varepsilon^2}$ .

## 5.3. Basis defectiveness

In this part, we directly observe the phenomenon of basis anomalies through the 1D nonreciprocal SSH model.

As mentioned earlier, through the non-Hermitian similarity transformation $\widehat{S}_{\mathrm{NHP}} = diag\{1, e^{q_0}, e^{q_0}, e^{2q_0}, e^{2q_0}, \ldots, e^{(N-1)q_0}, e^{(N-1)q_0}, e^{Nq_0}\}$ , we can obtain the Hermitian form Hamiltonian $\hat{H}_{\mathrm{OBC}}(k)$

$$
\hat {H} _ {\mathrm{OBC}} (k) = \widehat {\mathcal {S}} _ {\mathrm{NHP}} ^ {- 1} \hat {H} _ {\mathrm{PBC}} (k) \widehat {\mathcal {S}} _ {\mathrm{NHP}} = \sum_ {k} c _ {k} ^ {\dagger} \tau_ {x} (\bar {t} _ {1} + \bar {t} _ {2} \cos k) c _ {k} + \sum_ {k} c _ {k} ^ {\dagger} \tau_ {y} \bar {t} _ {2} \sin k c _ {k},\tag{40}
$$

with $e^{q_0} = \sqrt{\frac{(t_1 - \gamma)}{(t_1 + \gamma)}}$ , so in topological phase where $|\bar{t}_1| < |\bar{t}_2|$ , bases $(\{|\bar{b}^1\rangle \text{ and } |\bar{b}^2\rangle\})$ of $\hat{H}_{\mathrm{OBC}}(k)$ the basis of the edge states is given by [18].

$$
\left| \bar {b} ^ {1} \right\rangle = \frac {1}{\bar {\mathcal {N}}} \sum_ {n = 1} ^ {N} \left(- \frac {\bar {t} _ {1}}{\bar {t} _ {2}}\right) ^ {n - 1} | n \rangle \otimes (1, 0) ^ {T}, \left| \bar {b} ^ {2} \right\rangle = \frac {1}{\bar {\mathcal {N}}} \sum_ {n = 0} ^ {N - 1} \left(- \frac {\bar {t} _ {1}}{\bar {t} _ {2}}\right) ^ {n} | N - n \rangle \otimes (0, 1) ^ {T}.\tag{41}
$$

With $\bar{\mathcal{N}} = \sqrt{(1 - (\frac{\bar{t}_1}{\bar{t}_2})^{2N}) / (1 - (\frac{\bar{t}_1}{\bar{t}_2})^2)}$ , $(1,0)^T$ and $(0,1)^T$ represent sub-lattice degrees of freedom. The basis vector satisfies orthogonal normalization,

$$
\langle \bar {b} ^ {1} | \bar {b} ^ {2} \rangle = 0, \langle \bar {b} ^ {1} | \bar {b} ^ {1} \rangle = 1, \langle \bar {b} ^ {2} | \bar {b} ^ {2} \rangle = 1.
$$

(42)

![](images/004d8ff800cbc36ed259b9cf9a3e0cfcebb19bb9c836f21597402271e0dba329.jpg)

(b)
![](images/69c1ba6fd339f4d7e6774e78b22dc60c645ea8c1a052e111387a520079b55f75.jpg)

(a)
![](images/5ce549131902cda775f8903e00ddae8ab0f88506894bc4c94d34dabc0151095b.jpg)

(b $_{3}$ )
![](images/ea34f2afb1cd9771bdeee935a6b9ab5550ace2b31829dbb5eaa94176a163af5f.jpg)
Figure 3. (a) is the global phase diagram for the 1D nonreciprocal SSH model under open boundary conditions (N = 20 and $\varepsilon = 0.001$ ): four red dots → M-EEPs; solid black lines → H-EEPs; yellow region → IB-EEPs; gray region → IIB-EEPs; dashed lines → hidden quantum phase transition between IB-EEPs and IIB-EEPs. (b $_{1}$ ), (b $_{2}$ ) and (b $_{3}$ ) show examples of the Bloch peaches for M-EEPs, B-EEPs, and H-EEPs, respectively.

As mentioned earlier, the relationship between the wave functions of the two systems before and after similarity transformation satisfies $|\psi_{n}\rangle^{R}=\widehat{S}_{\mathrm{NHP}}|\bar{\psi}_{n}\rangle$ . Here, $|\psi_{n}\rangle^{R}$ are states of $\hat{H}_{\mathrm{PBC}}(k)$ and $|\bar{\psi}_{n}\rangle$ are states of $\hat{H}_{\mathrm{OBC}}(k)$ . So the basis of the $\hat{H}_{\mathrm{PBC}}(k)$ satisfies $\{|\tilde{b}^{1}\rangle,|\tilde{b}^{2}\rangle\}=\widehat{S}_{\mathrm{NHP}}\{||\tilde{b}^{1}\rangle,|\tilde{b}^{2}\rangle\}$ ,

$$
| \tilde {\mathrm{b}} ^ {1} \rangle = \hat {S} | \bar {\mathrm{b}} ^ {1} \rangle = \frac {1}{\bar {\mathcal {N}}} \sum_ {n = 1} ^ {N} \left(- \frac {\bar {t} _ {1}}{\bar {t} _ {2}}\right) ^ {n - 1} e ^ {q _ {0} (n - 1)} | n \rangle \otimes (1, 0) ^ {T},\tag{43}
$$

$$
| \tilde {\mathrm{b}} ^ {2} \rangle = \hat {S} | \bar {\mathrm{b}} ^ {2} \rangle = \frac {1}{\bar {\mathcal {N}}} \sum_ {n = 0} ^ {N - 1} \left(- \frac {\bar {t} _ {1}}{\bar {t} _ {2}}\right) ^ {n} e ^ {q _ {0} (N - n)} | N - n \rangle \otimes (0, 1) ^ {T}.\tag{44}
$$

The basis of the $\hat{H}_{\mathrm{PBC}}(k)$ has the following orthogonality normalization,

$$
\langle \tilde {\mathbf {b}} ^ {1} | \tilde {\mathbf {b}} ^ {2} \rangle = 0,\tag{45}
$$

$$
\langle \tilde {\mathbf {b}} ^ {1} | \tilde {\mathbf {b}} ^ {1} \rangle = \frac {1 - \left[ - \frac {(t _ {1} - \gamma)}{\bar {t} _ {2}} \right] ^ {2 N}}{1 - \left[ - \frac {(t _ {1} - \gamma)}{\bar {t} _ {2}} \right] ^ {2}} \frac {1}{\bar {\mathcal {N}} ^ {2}} = \frac {1 - \left[ - \frac {(t _ {1} - \gamma)}{\bar {t} _ {2}} \right] ^ {2 N}}{1 - \left[ - \frac {(t _ {1} - \gamma)}{\bar {t} _ {2}} \right] ^ {2}} \frac {1 - \left[ - \frac {\sqrt {\bar {t} _ {1} ^ {2} - \gamma^ {2}}}{\bar {t} _ {2}} \right] ^ {2}}{1 - \left[ - \frac {\sqrt {\bar {t} _ {1} ^ {2} - \gamma^ {2}}}{\bar {t} _ {2}} \right] ^ {2 N}} \approx 1,\tag{46}
$$

$$
\langle \tilde {\mathrm{b}} ^ {2} | \tilde {\mathrm{b}} ^ {2} \rangle = e ^ {2 q _ {0} N} \frac {1 - \left[ - \frac {(t _ {1} + \gamma)}{\bar {t} _ {2}} \right] ^ {2 N}}{1 - \left[ - \frac {(t _ {1} + \gamma)}{\bar {t} _ {2}} \right] ^ {2}} \frac {1}{\mathcal {N} ^ {2}} = e ^ {2 q _ {0} N} \frac {1 - \left[ - \frac {(t _ {1} + \gamma)}{\bar {t} _ {2}} \right] ^ {2 N}}{1 - \left[ - \frac {(t _ {1} + \gamma)}{\bar {t} _ {2}} \right] ^ {2}} \frac {1 - \left[ - \frac {\sqrt {t _ {1} ^ {2} - \gamma^ {2}}}{\bar {t} _ {2}} \right] ^ {2}}{1 - \left[ - \frac {\sqrt {t _ {1} ^ {2} - \gamma^ {2}}}{\bar {t} _ {2}} \right] ^ {2 N}} \approx e ^ {2 q _ {0} N}.\tag{47}
$$

This indicates that the basis vectors of the non-Hermitian system satisfy orthogonality, but do not satisfy normalization relationships. $\{|\tilde{b}^{1}\rangle, |\tilde{b}^{2}\rangle\}$ of the non-Hermitian are shown in figure 4(a) and (b) with the distribution of n. Here the basis $\tilde{b}^{2}|\tilde{b}^{2}\rangle$ is approximately equal to 0, this is consistent with equations (46) and (47). And this also indicates that the basis vector of the system is defective.

## 5.4. Extended EPs

Then, based on $\{\hat{H}_{\mathrm{NH}},\left\{|\psi_{0}^{\mathrm{R}}\rangle\right\}$ , we study the EEPs for the two edge states in the 1D nonreciprocal SSH model. In topological phase with $\bar{w}=1$ (or $|\bar{t}_{1}|<|\bar{t}_{2}|$ ), we have EEPs for the two edge states except for the Hermitian/anti-Hermitian cases at $\gamma=0$ or $t_{1}=0$ .

![](images/a8ce22dc6b690495bb7768eab9c8d1aaf2290d3338648120148b9c79433654fa.jpg)
(a)

![](images/29056bd5b4b74159aa7ab051844d63366d1caaf37d872f62002c4a38dcbfb579.jpg)
(b)
Figure 4. With $N = 40, t_{2} = 1, t_{1} = 0.5, \gamma = 0.05$ , the distribution basis (a) $|\tilde{b}^{1}\rangle$ and (b) $|\tilde{b}^{2}\rangle$ via the number of lattices n.

![](images/02844b13b86fd3172de395fdc517e29fcfaf18976ec340ab9b46e5422426362e.jpg)
(a)

![](images/f21cb5fc9b016b459ffdcc41a982497e9ca9b88928ef7cf102364deb1570fa1f.jpg)
(b)
Figure 5. (a) is the energy levels of two edge states; (b) is the state similarity of two edge states, $\Lambda = |\langle \tilde{\psi}_{+}^{\mathrm{R}}|\tilde{\psi}_{-}^{\mathrm{R}}\rangle |$ . (a) and (b) are all for the case of $N = 50$ , $t_2 = 1$ , $\varepsilon = 0.001$ and $t_1 = 0$ . The two energy levels become degenerate and the state similarity becomes 1 at the gray dotted line ( $\gamma = \pm 0.86$ ), where the M-EEP occurs.

Figure 3(a) is an illustration of the global phase diagram for EEPs: the four red dots correspond to M-EEPs, the solid black lines correspond to H-EEPs, the yellow region corresponds to IB-EEPs and the gray region corresponds to IIB-EEPs. The Bloch peaches for different types of EEPs are illustrated in figure 3(b₁) (an example for M-EEPs), figure 3(b₂) (an example for B-EEPs), and figure 3(b₃) (an example for H-EEPs). Then we discuss the different types of EEPs separately.

## 5.4.1. M-EEPs

Firstly, we consider the case of M-EEPs by setting $\varepsilon$ to a purely real value and $t_{1}=0$ . Now, the initial basis becomes normal,

$$
\left\{\left| \psi_ {0} ^ {\mathrm{R}} \right\rangle \right\} = \left\{\left| \psi_ {0} \right\rangle \right\}\tag{48}
$$

with $\widehat{\mathcal{S}}_B(\vec{\sigma}^B,\beta^B) = 1$ . The Hamiltonian of the effective two-level model is reduced to

$$
\hat {H} _ {\mathrm{NH}} = i \varepsilon \sigma^ {z} + \Delta_ {0} \sigma^ {x},\tag{49}
$$

where $\Delta_0 = \frac{(t_2^2 - t_1^2)}{t_2} (-\frac{t_1}{t_2})^N$ . A spontaneous $\mathcal{PT}$ -symmetry-breaking transition occurs at $|\varepsilon| = \Delta_0$ . The energy levels become degenerate, shown as figure 5(a),

$$
E _ {\pm} = \pm \sqrt {\varepsilon^ {2} - \Delta_ {0} ^ {2}} \rightarrow 0,\tag{50}
$$

and the non-Hermitian similarity transformation becomes singular,

$$
\widehat {\mathcal {S}} _ {M} = e ^ {- \beta^ {M} \cdot \sigma^ {y}}\tag{51}
$$

with $\beta^{M} = \left|\frac{1}{2}\ln \left|\frac{\Delta_{0} + \varepsilon}{\Delta_{0} - \varepsilon}\right|\right| \to \infty$ . We have an M-EEP with the following defective matrix basis:

$$
\left\{ \right.\left| \right. \psi^ {\mathrm{R}} \left. \right\rangle\left. \right\} = \left\{ \right.\widehat {\mathcal {S}} _ {M} \left(\sigma^ {\gamma}, \beta^ {M} \rightarrow \infty\right)\left| \right. \psi_ {0} \left. \right\rangle\left. \right\}.\tag{52}
$$

![](images/2ba86663b0bf132ebe0393d8488b80b7b2c03d43b3d7da5ee65ef2054e8c25da.jpg)
(a)

![](images/96c46554bf868dc36c5fe535637e59094eb5ce1f9f234deb6be3fc1d419ec2b9.jpg)
(b)
Figure 6. (a) is the energy levels of two edge states; (b) is the state similarity of two edge states, $\Lambda = |\langle \tilde{\psi}_{+}^{\mathrm{R}}|\tilde{\psi}_{-}^{\mathrm{R}}\rangle |$ . (a) and (b) are all for the case of $N = 50$ , $t_2 = 1$ , $\varepsilon = 0.001$ and $\gamma = 0.5$ . The two energy levels become degenerate ( $E_{+} = E_{-}$ ) at the gray dotted line ( $t_1 = 1.03$ ), where the H-EEP occurs with $\Lambda = 1$ ; The orange dotted line ( $t_1 = 0.38$ ) denoted the phase transition between IB-EEPs ( $E_{+} \neq E_{-}$ , $\Lambda = 0$ ) and IIB-EEPs ( $E_{+} \neq E_{-}$ , $\Lambda = 1$ ).

The state similarity $\Lambda = |\langle \tilde{\psi}_{+}^{\mathrm{R}}|\tilde{\psi}_{-}^{\mathrm{R}}\rangle| = 1$ indicates the coalescence of the two edge states, shown as the figure 5(b). From the illustration of figure 4, we can obtain that at M-EEPs two energy levels become degenerate and two energy states coalesce ( $E_{+} \neq E_{-}, \Lambda = 1$ ), shown as the gray dotted line at $\gamma = \pm 0.86$ . In addition, as shown in figure 3(b\$\_{1}\$), the Bloch peach has a symmetric axis along the $y$ -direction.

## 5.4.2. B-EEPs

Secondly, we consider the case of B-EEPs by setting $\left|\bar{\Delta}\right| \neq |\varepsilon|$ with $t_{1} \neq 0$ and $\gamma \neq 0$ . Now, the initial basis becomes defective, i.e.

$$
\left\{\left| \psi_ {0} ^ {\mathrm{R}} \right\rangle \right\} = \left\{\widehat {\mathcal {S}} _ {B} \left| \psi_ {0} \right\rangle \right\}
$$

with $\widehat{S}_{B}=e^{-\beta^{B}\cdot\sigma_{z}}$ and $\beta^{B}=Nq_{0}$ . Here, $q_{0}=\frac{1}{2}\ln\left(\frac{t_{1}-\gamma}{t_{1}+\gamma}\right)$ is an imaginary wave vector that characterizes the non-Hermitian skin effect. In the thermodynamic limit $N\to\infty$ , we have a B-EEP with the following defective matrix basis:

$$
\left\{\left| \psi^ {\mathrm{R}} \right\rangle \right\} = \left\{\widehat {\mathcal {S}} _ {M} \left(\vec {\sigma} ^ {M}, \beta^ {M}\right) \widehat {\mathcal {S}} _ {B} \left(\vec {\sigma} ^ {B}, \beta^ {B}\right) \left| \psi_ {0} \right\rangle \right\}.\tag{53}
$$

In figure 3(a), except for the black lines and red dots, B-EEPs exist throughout the whole topological insulator region. An interesting fact is that the energy levels are not degenerate, shown as figure 6(a), i.e.

$$
E _ {\pm} = \pm \sqrt {\bar {\Delta} ^ {2} - \varepsilon^ {2}} \neq 0.\tag{54}
$$

As a result, this is an example of EEPs without energy degeneracy. Now, the Bloch peach has a symmetric axis along the z-direction; see the illustration in figure 3(b₂).

Let us discuss the classes of B-EEPs in detail. For simplicity of calculation, we choose to discuss the classification of B-EEPs on a normal basis representation $\{\hat{H}_{\mathrm{NH}}^{\beta^{B}},\{| \psi_{0} \rangle \}\}$ here, with $\{| \psi_{0} \rangle\} = \{|1\rangle, |2\rangle\}$ and,

$$
\hat {H} _ {\mathrm{NH}} ^ {\beta^ {B}} = \left(\widehat {\mathcal {S}} _ {B}\right) ^ {- 1} \hat {H} _ {\mathrm{NH}} \left(\widehat {\mathcal {S}} _ {B}\right) = \bar {\Delta} ^ {+} \sigma^ {+} + \bar {\Delta} ^ {-} \sigma^ {-} + i \varepsilon \sigma^ {z},\tag{55}
$$

where $\bar{\Delta}^{+}=\bar{\Delta}\exp\left(-Nq_{0}\right)$ and $\bar{\Delta}^{-}=\bar{\Delta}\exp\left(Nq_{0}\right)$ . In the thermodynamic limit $N\to\infty$ , there exist three phases: a phase with $\left|\bar{\Delta}^{+}\right|\to\infty$ and $\left|\bar{\Delta}^{-}\right|\to0$ , a phase with $\left|\bar{\Delta}^{+}\right|\to0$ and $\left|\bar{\Delta}^{-}\right|\to\infty$ , and a phase with $\left|\bar{\Delta}^{+}\right|\to0$ and $\left|\bar{\Delta}^{-}\right|\to0$ . At $\left|\bar{\Delta}^{\pm}\right|=1$ or $t_{1}\pm\gamma=\pm1$ , the ‘hidden’ quantum phase transition occurs, with a sudden change between IB-EEPs ( $\Lambda=0$ ) and IIB-EEPs ( $\Lambda=1$ ). Shown as figure 6(b), the orange dotted line at $t_{1}=0.38$ identifies the phase transition between IB-EEPs ( $E_{+}\neq E_{-},\Lambda=0$ ) and IIB-EEPs ( $E_{+}\neq E_{-},\Lambda=1$ ). We find that the situation changes for the state similarity $\Lambda=|\langle\tilde{\psi}_{+}^{\mathrm{R}}|\tilde{\psi}_{-}^{\mathrm{R}}\rangle|$ of two edge states. In figure 7, one can see the ‘hidden’ quantum phase transition from the numerical results for the case of N=20 and $\varepsilon=0.001$ that $\Lambda$ suddenly changes from 0 to 1 at $\left|\bar{\Delta}^{\pm}\right|=1$ .

## 5.4.3. H-EEPs

Thirdly, we consider the case of H-EEPs by setting $\varepsilon$ to a purely real value, $\gamma \neq 0$ , and $|\bar{\Delta}| = |\varepsilon|$ . On the one hand, the initial basis is defective,

$$
\left\{\left| \psi_ {0} ^ {\mathrm{R}} \right\rangle \right\} = \left\{\widehat {\mathcal {S}} _ {B} \mid \psi_ {0} \right\rangle \Bigg \} = \left\{e ^ {- \beta^ {B} \cdot \sigma_ {z}} \mid \psi_ {0} \right\rangle \Bigg \}\tag{56}
$$

![](images/42bf431542a4d3d2d519eb8b09016eed3ba421af81239cbbec6689399b88808f.jpg)
Figure 7. The state similarity of two edge states, $\Lambda = |\langle \tilde{\psi}_{+}^{\mathrm{R}}|\tilde{\psi}_{-}^{\mathrm{R}}\rangle |$ , for the case of $N = 20$ and $\varepsilon = 0.001$ as a function of $\gamma$ and $t_1$ . A hidden phase transition occurs with a sudden change between IB-EEPs ( $\Lambda = 0$ , green regions) and IIB-EEPs ( $\Lambda = 1$ , red regions).

with $\beta^{B}=Nq_{0}\to\infty$ . In the thermodynamic limit $N\to\infty$ , the two-level system may be regarded as a B-EEP. On the other hand, for the case of $\left|\frac{\bar{\Delta}}{\varepsilon}\right|=1$ , the same system can be regarded as an M-EEP with another singular non-Hermitian similarity transformation $\widehat{\mathcal{S}_{B}}(\beta^{M}\to\infty)$ . Therefore, in thermodynamic limit $N\to\infty$ , we have an H-EEP with the following defective matrix basis:

$$
\left\{ \right.\left| \right. \psi^ {R} \left. \right\rangle\left. \right\} = \left\{ \right.\widehat {\mathcal {S}} _ {M} \left(\sigma_ {y}, \beta^ {M} = \left| \frac {1}{2} \ln \left| \frac {\bar {\Delta} + \varepsilon}{\bar {\Delta} - \varepsilon} \right|\right|\rightarrow \infty\right) \cdot \widehat {\mathcal {S}} _ {B} \left(\sigma_ {z}, \beta^ {B} = N q _ {0} \rightarrow \infty\right)\left| \right. \psi_ {0} \left. \right\rangle\left. \right\}.
$$

Shown the gray dotted line in figure 5 at $t_1 = 1.03$ . We can see that the two energy levels become degenerate ( $E_{+} = E_{-}$ ), where the H-EEP occurs with $\Lambda = 1$ .

Now, the quantum state under the non-Hermitian similarity transformations $\widehat{\mathcal{S}}_{M}(\sigma_{y},\beta^{M})$ and $\widehat{\mathcal{S}}_{B}(\sigma_{z},\beta^{B})$ becomes

$$
\left| \psi^ {R} \right\rangle = \frac {1}{\sqrt {2}} \left(\cos \frac {\theta}{2} + e ^ {- \beta^ {B}} \sin \frac {\theta}{2}\right) | 1 \rangle + \frac {i}{\sqrt {2}} e ^ {i \varphi} e ^ {- \beta^ {M}} \left(\cos \frac {\theta}{2} - e ^ {- \beta^ {B}} \sin \frac {\theta}{2}\right) | 2 \rangle\tag{57}
$$

(with $\theta \in [0,\pi]$ and $\varphi \in [0,2\pi]$ ), and the radius of the Bloch sphere $R$ can be obtained as

$$
R = \left\langle \psi^ {\mathrm{R}} \mid \psi^ {\mathrm{R}} \right\rangle = \frac {1}{\sqrt {2}} \left[ \left(\cos \frac {\theta}{2} + e ^ {- \beta^ {B}} \sin \frac {\theta}{2}\right) ^ {2} + e ^ {- 2 \beta^ {M}} \left(\cos \frac {\theta}{2} - e ^ {- \beta^ {B}} \sin \frac {\theta}{2}\right) ^ {2} \right] ^ {1 / 2}.\tag{58}
$$

As shown in figure $6(b_{3})$ , the Bloch peach, with a symmetric axis along the z-direction shrinks. In addition, for the non-Hermitian SSH model, the H-EEP is an unusual spontaneous PT-symmetry breaking accompanied by a transition from real to complex spectra.

Finally, we numerically calculated the similarity between two edge states $(\Lambda = 1)$ in the entire topology region of the SSH model, as shown in figure 7. This is completely consistent with the analytical phase diagram in figure 6(a).

## 6. Conclusion and discussion

It is widely accepted that exceptional points (EPs) are unique and ubiquitous features of non-Hermitian systems, at which both eigenvalues and eigenvectors coalesce. In this paper, we illustrate that the phenomenon of EPs in projected non-Hermitian systems is much more interesting than expected. we found that the essential reason for the coalesce of eigenvectors at the EP point is the basis defectiveness. Based on defective basis vectors, we define the concept of Extended exceptional points (EEPs). For projected non-Hermitian two-level systems, there may exist three types of EEPs—M-EEPs (with a defective matrix basis and a normal initial basis), B-EEPs (with a normal matrix basis and a defective initial basis), and H-EEPs (with a defective matrix basis and a defective initial basis). In addition, there are two classes of B-EEPs—IB-EEPs, without eigenstate coalescence, and IIB-EEPs, with eigenstate coalescence.

By taking the topologically protected edge states in the non-Hermitian SSH model as an example, we demonstrate the phenomenon of basis defectiveness and explore the physical properties of EEPs. In the future, we will study higher-order EEPs in projected non-Hermitian systems and attempt to develop a complete theory of EEPs.

## Data availability statement

All data that support the findings of this study are included within the article (and any supplementary files).

[18] Wang X-R, Guo C-X and Kou S-P 2020 Phys. Rev. B 101 121116

Bender C M, Brody D C, Jones H F and Meister B K 2007 Phys. Rev. Lett. 98 040403

## Acknowledgments

This work is supported by National Key R&D Program of China (Grant No. 2023YFA1406704) and National Natural Science Foundation of China (NSFC) (Grant Nos. 11704186, 11974053 and 12174030).

## ORCID iDs

Fei Yang https://orcid.org/0000-0002-1870-6824

Su-Peng Kou https://orcid.org/0000-0003-2225-3677

## References

[1] Kato T 1966 Perturbation Theory of Linear Operators (Springer)

[2] Heiss W D 1970 Nucl. Phys. A 144 417

[3] Moiseyev N and Friedland S 1980 Phys. Rev. A 22 618

[4] Bender C M and Boettcher S 1998 Phys. Rev. Lett. 80 5243

[5] Miri M-A and Al A 2019 Science 363 7709

[6] Minganti F, Miranowicz A, Chhajlany R W and Nori F 2019 Phys. Rev. A 100 062131

[7] Bender C M, Brody D C and Jones H F 2002 Phys. Rev. Lett. 89 270401

[8] Heiss W D 2012 J. Phys. A: Math. Theor. 45 444016

[9] Mostafazadeh A 2007 Phys. Rev. Lett. 99 130502

[10] Lee Y C, Hsieh M H, Flammia S T and Lee R K 2014 Phys. Rev. Lett. 112 130404

[11] Kawabata K, Ashida Y and Ueda M 2017 Phys. Rev. Lett. 119 190401

[12] El-Ganainy R, Makris K G, Khajavikhan M, Musslimani Z H, Rotter S and Christodoulides D N 2018 Nat. Phys. 14 11

[13] Yao S and Wang Z 2018 Phys. Rev. Lett. 121 086803
Yao S, Song F and Wang Z 2018 Phys. Rev. Lett. 121 136802

[14] Zdemir Ş K, Rotter S, Nori F and Yang L 2019 Nat. Mater. 18 783

[15] Lin S, Jin L and Song Z 2019 Phys. Rev. B 99 165148

[16] Deng T S and Yi W 2019 Phys. Rev. B 100 035102

[17] Liu Y, Jiang X P, Cao J and Chen S 2020 Phys. Rev. B 101 174205

[19] Ashida Y, Gong Z and Ueda M 2020 Adv. Phys. 69 249–435

[20] Guo C-X, Wang X-R, Wang C and Kou S-P 2020 Phys. Rev. B 101 144439

[21] Guo C-X, Wang X-R, Wang C and Kou S-P 2020 Europhys. Lett. 131 27002

[22] Wang C, Wang X-R and Kou S-P 2020 Int. J. Mod. Phys. B 34 2050146

[23] Wang C, Yang M-L, Guo C-X, Zhao X-M and Kou S-P 2019 Europhys. Lett. 128 41001

[24] Moiseyev N 2011 Non-Hermitian Quantum Mechanics (Cambridge University Press)

[25] Ju C-Y, Miranowicz A, Minganti F, Chan C-T, Chen G-Y and Nori F 2022 Phys. Rev. Res. 4 023070

[26] Ju C-Y, Miranowicz A, Chen G-Y and Nori F 2014 Phys. Rev. A 100 062118

[27] Feng L, Wong Z-J, Ma R-M, Wang Y and Zhang X 2014 Science 346 972

[28] Peng B, Zdemir S K, Rotter S, Yilmaz H, Liertzer M, Monifi F, Bender C M, Nori F, Yang L and Fu L 2014 Science 346 328

[29] Hodaei H, Miri M A, Heinrich M, Christodoulides D N and Khajavikhan M 2014 Science 346 975

[30] Guo A, Salamo G J, Duchesne D, Morandotti R, Volatier-Ravat M, Aimez V, Siviloglou G A and Christodoulides D N 2009 Phys. Rev. Lett. 103 093902

[31] Chen W, Zdemir S K, Zhao G, Wiersig J and Yang L 2017 Nature 548 192

[32] Hodaei H, Hassan A U, Wittek S, Garcia-Garcia H, El-Ganainy R, Christodoulides D N and Khajavikhan M 2017 Nature 548 187

[33] Rter C E, Makris K G, El-Ganainy R, Christodoulides D N, Segev M and Kip D 2010 Nat. Phys. 6 192

[34] Schindler J, Li A, Zheng M C and Ellis F M a T 2011 Phys. Rev. A 84 040101(R)

[35] Feng L, Ayache M, Huang J, Xu Y-L, Lu M-H, Chen Y-F, Fainman Y and Scherer A 2011 Science 333 729

[36] Regensburger A, Bersch C, Miri M-A, Onishchukov G, Christodoulides D N and Peschel U 2012 Nature 488 167

[37] Bittner S, Dietz B, Gnther U, Harney H L, Miski-Oglu M, Richter A and Schfer F 2012 Phys. Rev. Lett. 108 024101

[38] Bender C M, Berntson B K, Parker D and Samuel E 2013 Am. J. Phys. 81 173

[39] Hang C, Huang G and Konotop V V 2013 Phys. Rev. Lett. 110 083604

[40] Zhu X, Ramezani H, Shi C, Zhu J and Zhang X 2014 Phys. Rev. X 4 031042

[41] Brandstetter M et al 2014 Nat. Commun. 5 4034

[42] Popa B-I and Cummer S A 2014 Nat. Commun. 5 3398

[43] Fleury R, Sounas D and Al A 2015 Nat. Commun. 6 5905

[44] Zhen B, Hsu C W, Igarashi Y, Lu L, Kaminer I, Pick A, Chua S-L, Joannopoulos J D and Soljačić M 2015 Nature 525 354

[45] Xu H, Mason D, Jiang L and Harris J G E 2016 Nature 537 80

[46] Peng P, Cao W, Shen C, Qu W, Wen J, Jiang L and Xiao Y 2016 Nat. Phys. 12 1139

[47] Zhang Z, Zhang Y, Sheng J, Yang L, Miri M-A, Christodoulides D N, He B, Zhang Y and Xiao M 2016 Phys. Rev. Lett. 117 123601

[48] Assawaworrarit S, Yu X and Fan S 2017 Nature 546 387

[49] Choi Y, Hahn C, Yoon J W and Song S H 2018 Nat. Commun. 9 2182

[50] Scheel S and Szameit A 2018 Europhys. Lett. 122 34001

[51] Wu Y, Liu W, Geng J, Song X, Ye X, Duan C-K, Rong X and Du J 2019 Science 364 878

[52] Xiao L et al 2017 Nat. Phys. 13 1117

[53] Naghiloo M, Abbasi M, Joglekar Y N and Murch K W 2019 Nat. Phys. 15 1232

[54] Xiao L, Wang K, Zhan X, Bian Z, Kawabata K, Ueda M, Yi W and Xue P 2019 Phys. Rev. Lett. 123 230401

[55] Liu W, Wu Y, Duan C-K, Rong X and Du J 2021 Phys. Rev. Lett. 126 170506

[56] Wang W-C et al 2021 Phys. Rev. A 103 L020201

[57] Ding L, Shi K, Zhang Q, Shen D, Zhang X and Zhang W 2021 Phys. Rev. Lett. 126 083604

[58] Hasan M Z and Kane C L 2010 Rev. Mod. Phys. 82 3045

[59] Qi X-L and Zhang S-C 2011 Rev. Mod. Phys. 83 1057

[60] Kitaev A 2003 Ann. Phys., NY 303 2
Kitaev A 2006 Ann. Phys., NY 321 2

[61] Wen X-G 2004 Quantum Field Theory of Many-Body Systems (Oxford University Press) p 520

[62] Wen X-G 1990 Int. J. Mod. Phys. B 4 239

[63] Kou S-P 2009 Phys. Rev. Lett. 102 120402 Yu J and Kou S-P 2009 Phys. Rev. B 80 075107 Kou S-P 2009 Phys. Rev. A 80 052317
