# Projective Hilbert space structures at exceptional points

To cite this article: Uwe Günther et al 2007 J. Phys. A: Math. Theor. 40 8815

View the article online for updates and enhancements.

## You may also like

\- Electrocatalytic Detection of Epinephrine at Polyaniline-Biosynthesized Nickel Oxide Modified Electrode-Supported By Computational Study
Gloria Ebube Uwaya and Krishna Bisetty

\- Application of Clustering Algorithm by Data Mining in the Analysis of Smart Grid from the Perspective of Electric Power Yaru Qi, Junda Ren, Ni Sun et al.

\- Electroporation microchips for in vitro gene transfection
Yu-Cheng Lin and Ming-Yuan Huang

J. Phys. A: Math. Theor. 51 (2018) 459501 (1pp)

Corrigendum

# Corrigendum: Projective Hilbert space structures at exceptional points (2007 J. Phys. A: Math. Theor. 40 8815)

Uwe Günther $^{1}$ , Ingrid Rotter $^{2}$ and Boris F Samsonov $^{3,4}$

$^{1}$ Helmholtz Center Dresden-Rossendorf, D-01328 Dresden, Germany

$^{2}$ Max Planck Institute for physics of complex systems, D-01187 Dresden, Germany

$^{3}$ Physics Department, Tomsk State University, 634050 Tomsk, Russia

E-mail: u.guenther@hzdr.de and rotter@pks.mpg.de

Received 4 September 2018

Accepted for publication 21 September 2018

Published 16 October 2018

![](images/b30196e52911e6cf9dfe316c9651f3d11f54585a169b54806af0e5c1abf99cc4.jpg)

There was an error in the explanations of the monodromy group for encircling an exceptional point (EP) in parameter space. The correct statement concerning equation (42) should read:

These elements $W(\alpha)$ form an Abelian parabolic subgroup $P$ of the special linear group $SL(2, \mathbb{C}) \supset P$ (see, e.g. [62, 63, 66])

$$
W (\alpha + \beta) = W (\alpha) W (\beta) = W (\beta) W (\alpha)\tag{42}
$$

corresponding to the mapping $e^{i\alpha} \in S^{1} \approx U(1) \mapsto P \subset SL(2, \mathbb{C})$ .

Additionally, two text statements in the last paragraphs of the introduction and the conclusion sections should read accordingly:

The corresponding monodromy group is identified as parabolic Abelian subgroup of the special linear group $SL(2,\mathbb{C})$ and evidence is given that vector norm scalings are only due to complex dynamical phases whereas geometrical phases are purely real-valued and norm preserving.

With the help of the Puiseux expanded eigenvectors it has been shown that the geometric phase obtained on circles around EPs of complex symmetric Hamiltonians is purely real-valued and that the corresponding monodromy transformations are induced by an Abelian parabolic subgroup of $SL(2,\mathbb{C})$ .

## ORCID iDs

Uwe Günther https://orcid.org/0000-0002-2272-4021

# Projective Hilbert space structures at exceptional points

Uwe Günther $^{1}$ , Ingrid Rotter $^{2}$ and Boris F Samsonov $^{3}$

$^{1}$ Research Center Dresden-Rossendorf, PO 510119, D-01314 Dresden, Germany

$^{2}$ Max Planck Institute for the Physics of Complex Systems, D-01187 Dresden, Germany

$^{3}$ Physics Department, Tomsk State University, 36 Lenin Avenue, 634050 Tomsk, Russia

E-mail: u.guenther@fzd.de, rotter@mpipks-dresden.mpg.de and samsonov@phys.tsu.ru

Received 23 April 2007, in final form 6 June 2007

Published 12 July 2007

Online at stacks.iop.org/JPhysA/40/8815

## Abstract

A non-Hermitian complex symmetric $2 \times 2$ -matrix toy model is used to study projective Hilbert space structures in the vicinity of exceptional points (EPs). The bi-orthogonal eigenvectors of a diagonalizable matrix are Puiseux-expanded in terms of the root vectors at the EP. It is shown that the apparent contradiction between the two incompatible normalization conditions with finite and singular behaviour in the EP-limit can be resolved by projectively extending the original Hilbert space. The complementary normalization conditions correspond then to two different affine charts of this enlarged projective Hilbert space. Geometric phase and phase-jump behaviour are analysed, and the usefulness of the phase rigidity as measure for the distance to EP configurations is demonstrated. Finally, EP-related aspects of PT-symmetrically extended quantum mechanics are discussed and a conjecture concerning the quantum brachistochrone problem is formulated.

PACS numbers: 03.65.Fd, 03.65.Vf, 03.65.Ca, 02.40.Xx

## 1. Introduction

A generic property of non-Hermitian operators is the possible occurrence of nontrivial Jordan blocks in their spectral decomposition [1]. For an operator $H(\mathbf{X})$ depending on a set of parameters $\mathbf{X} = (X_{1}, \ldots, X_{m}) \in \mathcal{M}$ from a space M this means that, in case of a single Jordan block, two or more spectral branches $\lambda_{1}(\mathbf{X}), \ldots, \lambda_{k}(\mathbf{X})$ may coalesce (degenerate) at certain parameter hypersurfaces $V \subset M$ under simultaneous coalescence of the corresponding eigenvectors $\Phi_{1}(\mathbf{X}), \ldots, \Phi_{k}(\mathbf{X})$ : $\lambda_{1}(\mathbf{X}_{c}) = \cdots = \lambda_{k}(\mathbf{X}_{c}), \Phi_{1}(\mathbf{X}_{c}) = \cdots = \Phi_{k}(\mathbf{X}_{c}) \equiv \Theta_{0}(\mathbf{X}_{c})$ for $X_{c} \in V$ . Spectral points of this type are branch points of the spectral Riemann surface and are called exceptional points (EPs) [1]. At the EPs the set of the originally k linearly independent eigenvectors $\Phi_{1}(\mathbf{X}), \ldots, \Phi_{k}(\mathbf{X})$ is replaced by the single eigenvector $\Theta_{0}(\mathbf{X}_{c})$ and k-1 associated vectors $\Theta_{1}(\mathbf{X}_{c}),\ldots,\Theta_{k-1}(\mathbf{X}_{c})$ which form a Jordan chain. Together they span the so-called k-dimensional algebraic eigenspace (or root space) $\mathfrak{S}_{\lambda}(\mathbf{X}_{c})=\operatorname{span}[\Theta_{0}(\mathbf{X}_{c}),\ldots,\Theta_{k-1}(\mathbf{X}_{c})]$ [1, 2] so that the total space dimension remains preserved. The construction extends straightforwardly to the presence of several Jordan blocks for the same eigenvalue $\lambda(\mathbf{X}_{c})$ . In general, the degeneration hypersurface $V\subset M$ consists of components $V_{a}$ of different co-dimension co dim $V_{a}=a$ . Higher order Jordan blocks require a higher degree of parameter tuning (they have a higher co-dimension) and a correspondingly lower dimension of the component $V_{a}$ . Due to the different dimensions of its components $V_{a}$ the degeneration hypersurface $V=\bigcup_{a}V_{a}$ itself has the structure of a stratified manifold [3].

EPs occur naturally in quantum scattering setups $[4, 5]$ when two or more resonance states coalesce and higher order poles of the S-matrix form. Within the Gamow state approach such S-matrix double poles have been considered in $[6–9]$ , whereas in the Feshbach projection operator formalism (one of the basic approaches to analyse open quantum systems) they naturally occurred in studies of nuclei $[10]$ , atoms $[11, 12]$ and quantum dots $[13, 14]$ . EP-related crossing and avoided crossing scenarios have been studied for bound states in the continuum $[11, 15–17]$ as well as for phase transitions $[18–21]$ . In asymptotic analyses of quasi-stationary systems EPs show up as hidden crossings $[22]$ . EP-related theoretically predicted level and width bifurcation properties have been experimentally verified in a series of microwave resonator cavity experiments. In $[23]$ , the resonance trapping phenomenon (width bifurcation) $[10]$ has directly been proven. The four-fold winding around an EP has been found experimentally $[24]$ in full agreement with the theoretical prediction $[19, 25]$ and related studies $[14, 26]$ . In $[27]$ , two-level coalescences have been associated with chiral system behaviour. The geometric phase at EPs has been discussed in $[14, 26–32]$ .

EPs also play an important role in the recently considered PT-symmetrically extended quantum models $[33–35]$ . There they correspond to the phase-transition points between physical sectors of exact PT symmetry and unphysical sectors of spontaneously broken PT symmetry $[36–40]$ .

Other, non-quantum mechanical examples where EPs play an important role are the optics of bianisotropic crystals $[41]$ , acoustic models $[42]$ , many hydrodynamic setups where EPs have been studied within pseudo-spectral approaches $[43]$ as well as a large number of mechanical models $[44]$ where they are connected with regimes of critical stability $[45]$ . Recent results on magnetohydrodynamic dynamo models indicate on a close connection between nonlinear polarity reversal mechanisms of magnetic fields and EPs $[46]$ .

For completeness we note that the perturbation theory for systems in the vicinity of EPs dates back to 1960 [47] (see also [2]) and that it is closely related to singularity theory, catastrophe theory and versal deformations of Jordan structures [48]. Supersymmetric mappings between EP configurations have been recently considered in [49, 50].

A correct perturbative treatment of models in the vicinity of EPs has to be built on an expansion in terms of root vectors (eigenvectors and associated vectors $\Theta_{i}(\mathbf{X}_{c})$ ) at the corresponding unperturbed eigenvalue $\lambda(\mathbf{X}_{c})$ (see e.g. [2, 44]). For $X \notin V$ (away from the EP at $X_{c}$ and from other EPs), the operator $H(\mathbf{X})$ has a diagonal spectral decomposition with corresponding eigenvectors $\Phi_{i}(\mathbf{X})$ . Choosing the normalization of these eigenvectors away from the EP and without regard to the expansion in terms of root vectors leads to a divergence of the normalization constants in the EP-limit $X \rightarrow X_{c}$ . The diagonalization break-down at $X_{c}$ , the occurrence of the Jordan block structure and the singular behaviour of the eigenvector decomposition are generic and were many times described in various contexts (see e.g. [51, 52]). The natural question connected with the fitting of the root-vector-based normalization and the diagonalizable-configuration normalization (and related controversial discussions on their physical interpretation $[51, 52]$ is whether and how the singular behaviour affects the projective Hilbert space structure of quantum systems.

In the present paper, we answer this question by resolving the singularity with the help of embedding the original Hilbert space $H = C^{n}$ into its projective extension $CP^{n}$ instead of projecting it down to $CP^{n-1}$ as in standard Hermitian quantum mechanics. Diagonal spectral decompositions and decompositions with nontrivial root spaces live then simply in different (and complementary) affine charts of $CP^{n}$ similar like monopole configurations of Hermitian systems have to be covered with two charts (north-pole chart and south-pole chart) of the unit sphere $S^{2}$ [53].

The basic construction is demonstrated on a simple complex symmetric (non-Hermitian) $2 \times 2$ -matrix toy model. The consideration of complex symmetric matrices sets no restriction because by a similarity transformation any complex matrix can be brought to a complex symmetric form (see, e.g. [54, 55]). The Hilbert space notations for the $2 \times 2$ -matrix model are fixed in section 2. In section 3, following [32, 44] we derive the leading-order perturbative expansion in the vicinity of an EP at $\mathbf{X}_c$ in terms of root vectors and fit it then explicitly with expressions of the diagonal spectral decomposition at $\mathbf{X} \neq \mathbf{X}_c$ . Combining geometric phase techniques for non-Hermitian systems [28] with projective Hilbert space techniques from [56], we generalize the projective geometric phase techniques of Hermitian systems to paths around EPs (section 4). The corresponding monodromy group is identified as a parabolic Abelian subgroup of the orthogonal group $O(2, \mathbb{C})$ and evidence is given that vector norm scalings are only due to complex dynamical phases whereas geometrical phases are purely real-valued and norm preserving. In section 5, we consider an instantaneous (stationary type) picture of the system. Within such a picture, we resolve the singular normalization behaviour by projectively embedding the Hilbert space $\mathcal{H} = \mathbb{C}^2 \hookrightarrow \mathbb{CP}^2$ . We discuss the necessity for an affine multichart covering of $\mathbb{CP}^2$ in order to accommodate diagonal-decomposition vectors and root vectors at EPs simultaneously. The usefulness of the phase rigidity as distance measure to EPs is discussed in section 6. In section 7, some EP-related aspects of $\mathcal{PT}$ -symmetric quantum models are discussed and a conjecture concerning the quantum brachistochrone problem [34, 57] is formulated. Conclusions (section 8) are followed by the appendix A where auxiliary results on Jordan structures of complex symmetric matrices are listed.

## 2. Setup

The subject of our consideration is the behaviour of a quantum system near a level crossing point of two resonance states—supposing that for an N-level system the influence of the other N - 2 levels is sufficiently weak. Under this assumption the setup can be modelled by an effective complex symmetric (non-Hermitian) $2 \times 2$ -matrix Hamiltonian

$$
H = \left( \begin{array}{c c} \epsilon_ {1} & \omega \\ \omega & \epsilon_ {2} \end{array} \right), \qquad H = H ^ {T}.\tag{1}
$$

The effective energies $\epsilon_{1,2} \in \mathbb{C}$ and the effective channel coupling $\omega \in \mathbb{C}$ will in general depend on underlying parameters $\mathbf{X} = (X_1, \ldots, X_k) \in \mathcal{M}$ from a space $\mathcal{M}$ .

For nonvanishing coupling $\omega \neq 0$ , the Hamiltonian can be rewritten as

$$
H = E _ {0} \otimes I _ {2} + \omega \left( \begin{array}{c c} Z & 1 \\ 1 & - Z \end{array} \right),\tag{2}
$$

with $I_{2}$ denoting the $2 \times 2$ unit matrix and

$$
E _ {0} := \frac {1}{2} (\epsilon_ {1} + \epsilon_ {2}), \quad Z := \frac {\epsilon_ {1} - \epsilon_ {2}}{2 \omega}.\tag{3}
$$

In this representation the eigenvalues $E_{\pm}$ and eigenvectors $\Phi_{\pm}$ of $H$ take the very simple form

$$
E _ {\pm} = E _ {0} \pm \omega \sqrt {Z ^ {2} + 1}\tag{4}
$$

and

$$
\Phi_ {\pm} = \binom{1}{- Z \pm \sqrt {Z ^ {2} + 1}} c _ {\pm}, \qquad c _ {\pm} \in \mathbb {C} ^ {*} := \mathbb {C} - \{0 \},\tag{5}
$$

which makes the branching behaviour most transparent $^{4}$ . From the overlap

$$
\begin{array}{l} \langle \Phi_ {+} | \Phi_ {-} \rangle \equiv \Phi_ {+} ^ {* T} \Phi_ {-} \\ = c _ {+} ^ {*} c _ {-} [ 1 + | Z | ^ {2} - | Z ^ {2} + 1 | + 2 \operatorname{Im} (Z ^ {*} \sqrt {Z ^ {2} + 1}) ] \end{array}\tag{6}
$$

one reads that $\langle\Phi_{+}|\Phi_{-}\rangle=0$ holds only for Im Z=0 and that for general $Z\in C$ the two states $\Phi_{+}$ and $\Phi_{-}$ are not orthogonal $\langle\Phi_{+}|\Phi_{-}\rangle\neq0$ . Following standard techniques [44] for non-Hermitian operators, we consider a dual (left) basis $\Xi_{\pm}$ bi-orthogonal to $\Phi_{\pm}$

$$
(H ^ {+} - E _ {\pm} ^ {*}) \Xi_ {\pm} = 0, \qquad \langle \Xi_ {k} | \Phi_ {l} \rangle \propto \delta_ {k l}, \qquad k, l = \pm .\tag{7}
$$

For complex symmetric H it holds $\Xi_{\pm} \propto \Phi_{\pm}^{*}$ so that the most general ansatz for the right and left basis vectors $\Phi_{\pm}$ and $\Xi_{\pm}$ can be chosen as

$$
\Phi_ {\pm} = c _ {\pm} \chi_ {\pm}, \qquad \Xi_ {\pm} = d _ {\pm} ^ {*} \chi_ {\pm} ^ {*}, \qquad c _ {\pm}, d _ {\pm} \in \mathbb {C} ^ {*}\tag{8}
$$

$$
\chi_ {\pm} := \binom{1}{- Z \pm \sqrt {Z ^ {2} + 1}}.\tag{9}
$$

The bi-orthogonality

$$
\langle \Xi_ {\pm} | \Phi_ {\mp} \rangle = d _ {\pm} c _ {\mp} \chi_ {\pm} ^ {T} \chi_ {\mp} = 0\tag{10}
$$

is ensured by the structure of $\chi_{\pm}$ and holds for any value of the parameter $Z\in \mathbb{C}$ . A normalization $\langle \Xi_{\pm}|\Phi_{\pm}\rangle = 1$ would set two constraints on the four free scaling parameters $c_{\pm}$ , $d_{\pm}\in \mathbb{C}^*$ :

$$
\langle \Xi_ {\pm} | \Phi_ {\pm} \rangle = d _ {\pm} c _ {\pm} \chi_ {\pm} ^ {T} \chi_ {\pm} = 1,\tag{11}
$$

so that the system would still have two free parameters which should be fixed by additional assumptions. Subsequently, we will mainly work with the bi-orthogonality properties of the vectors $\Phi_{\pm}$ , $\Xi_{\pm}$ and fix their normalization only when explicitly required.

Due to the arbitrary scaling parameters $c_{\pm}, d_{\pm} \in C^{*}$ of the right and left eigenvectors $\Phi_{\pm}, \Xi_{\pm} \in H = C^{2}(8)$ , it is natural to consider equivalence classes of such vectors defined by corresponding lines $\pi(\Phi_{\pm}), \pi(\Xi_{\pm})$ . These lines form the projective Hilbert space $\mathbb{P}(\mathcal{H}) = \mathcal{H}^{*}/\mathbb{C}^{*} = \mathbb{CP}^{1} \ni \pi(\Phi_{\pm}), \pi(\Xi_{\pm})$ [58–60], where $H^{*} := H - \{0\}$ denotes the original Hilbert space with the point at origin $\{0\} = (0, 0)$ deleted to allow for a consistent definition of $\mathbb{P}(\mathcal{H})$ . The space $\mathbb{P}(\mathcal{H})$ is covered by a single chart of homogeneous coordinates $(z_{0}, z_{1})^{T} \in \mathcal{H}$ and two complementary charts of affine coordinates $U_{0} \ni (1, z_{1}/z_{0}), z_{0} \neq 0$ and $U_{1} \ni (z_{0}/z_{1}, 1), z_{1} \neq 0$ . Comparison with the structure of the auxiliary vectors $\chi_{\pm}(9)$ shows that the $\chi_{\pm}$ can be straightforwardly re-interpreted as points of the projective space $CP^{1}$ described by the affine coordinate over $U_{0}: \chi_{\pm} \approx \pi(\Phi_{\pm})$ . The vectors $\{\Phi_{\pm}, \Xi_{\pm}\}$ themselves can be understood as sections of the natural line bundle $L = \{(p, v) \in \mathbb{P}(\mathcal{H}) \times \mathcal{H | v = cp, c \in C^{*}}\}$ [53], i.e. as $\Phi_{\pm} = \pi(\Phi_{\pm}) \otimes c_{\pm}, \Xi_{\pm} = \pi(\Xi_{\pm}) \otimes d_{\pm}^{*}$ , where $\pi$ denotes the projection $\pi : H^{*} \to P(\mathcal{H})$ . The bundle structure is locally trivial $\pi^{-1}(U_{0}) \approx U_{0} \times C^{*} \ni \Phi_{\pm}[61]^{5}$ .

## 3. Jordan structure

At an EP, the two eigenvalues $E_{\pm}$ coalesce $E_{+} = E_{-} = E_{0}$ . According to (4), this happens for $Z^{2} = -1$ and $Z = Z_{c} := \pm i$ and via (8) it is connected with a coalescence of the corresponding lines $\pi(\Phi_{+}) = \pi(\Phi_{-}) =: \pi(\Phi_{0})$ encoded in

$$
\chi_ {+} = \chi_ {-} = \chi_ {0} := \binom{1}{- Z _ {c}} = \binom{1}{\mp \mathrm{i}}.\tag{12}
$$

This means that the eigenvalue $E_0$ has algebraic multiplicity $n_a(E_0) = 2$ and geometric multiplicity $n_g(E_0) = 1$ , and by definition the level crossing point is an EP of the spectrum. The bi-orthogonality (10) of $\Phi_{\pm}$ and $\Xi_{\mp}$ is compatible with the coalescence of the lines due to the vanishing bi-norm $\chi_0^T\chi_0 = 0$ , i.e. the isotropy $^6$ of $\chi_0$ —a generic fact holding for the (geometric) eigenvector at any EP [2, 44]. We note that the coalescence $\pi(\Phi_{+}) = \pi(\Phi_{-}) = \pi(\Phi_0)$ still leaves the freedom for the vectors $\Phi_{+} = c_{+}\chi_{0}$ and $\Phi_{-} = c_{-}\chi_{0}$ of being two different sections $\Phi_{+} \neq \Phi_{-}$ of the same fibre $\pi(\Phi_0) \times \mathbb{C}^*$ over $\pi(\Phi_0)$ .

The right and left eigenvectors $\Phi_0$ , $\Xi_0$ at the EP are supplemented by corresponding associated vectors (algebraic eigenvectors) $\Phi_1$ and $\Xi_1$ defined by the Jordan chain relations [44]

$$
\begin{array}{l l} [ H (Z _ {c}) - E _ {0} I _ {2} ] \Phi_ {0} = 0, & [ H (Z _ {c}) - E _ {0} I _ {2} ] \Phi_ {1} = \Phi_ {0} \\ [ H (Z _ {c}) - E _ {0} I _ {2} ] ^ {+} \Xi_ {0} = 0, & [ H (Z _ {c}) - E _ {0} I _ {2} ] ^ {+} \Xi_ {1} = \Xi_ {0}. \end{array}\tag{13}
$$

From the inhomogeneity of these Jordan chain relations it follows immediately that the root vectors $\Phi_0$ and $\Phi_{1}$ as well as $\Xi_0$ and $\Xi_{1}$ scale simultaneously and in a linked way with the same single scale factor $c_{0}$ and $d_0^*$ , respectively. This is also visible from their explicit representation (A.9) derived in the appendix

$$
\begin{array}{l l} \Phi_ {0} = \sigma q c _ {0} \binom{1}{- Z _ {c}}, & \Phi_ {1} = \sigma q ^ {- 1} c _ {0} \binom{- Z _ {c}}{1} \\ \Xi_ {0} = \sigma q ^ {*} d _ {0} ^ {*} \binom{- Z _ {c}}{1}, & \Xi_ {1} = \sigma q ^ {* - 1} d _ {0} ^ {*} \binom{1}{- Z _ {c}} \end{array}\tag{14}
$$

$$
\sigma := \frac {\mathrm{e} ^ {\mathrm{i} \mu \frac {\pi}{4}}}{\sqrt {2}}, \qquad q := \sqrt {2 \omega}, \qquad Z _ {c} = \pm \mathrm{i} =: \mu \mathrm{i}, \qquad c _ {0}, d _ {0} \in \mathbb {C} ^ {*}.\tag{15}
$$

The simultaneous scaling means that the lines $\pi(\Phi_{0}),\pi(\Xi_{0})$ at the EP should be interpreted as the one-dimensional components (projections) of two-dimensional planes which span the root space $^{7}$ $\mathfrak{S}(E_{0})$ [2] and which scale as a whole with a single scale factor. Such a higher dimensional (complex) plane structure goes clearly beyond the one-dimensional line structure of the projective space $\mathbb{P}(\mathcal{H})$ (mathematically one should extend the natural line bundle of the original projective space to a more general projective flag bundle [62, 63]) $^{8}$ and underlines the fact that the state at an EP itself is not an element of the projective Hilbert space $\mathbb{P}(\mathcal{H})$ in its usual understanding.

The basis sets $\{\Phi_0, \Phi_1\}$ and $\{\Xi_0, \Xi_1\}$ satisfy the well-known bi-orthogonality conditions [44]

$$
\langle \Xi_ {0} | \Phi_ {0} \rangle = \langle \Xi_ {1} | \Phi_ {1} \rangle = 0 \qquad \langle \Xi_ {0} | \Phi_ {1} \rangle = \langle \Xi_ {1} | \Phi_ {0} \rangle = d _ {0} c _ {0} \neq 0.\tag{16}
$$

Again, a normalization $\langle \Xi_0|\Phi_1\rangle = \langle \Xi_1|\Phi_0\rangle = 1$ would only lead to a constraint $d_0c_0 = 1$ on the scale factors, but would not fix them completely. Due to this scaling freedom the single line $\pi (\Phi_0)$ of a given Jordan structure, in general, still allows for different sections $\Phi_{0,a}\neq \Phi_{0,b}$ of the corresponding fibre $\pi (\Phi_0)\times \mathbb{C}^*\ni \Phi_{0,a},\Phi_{0,b},\pi (\Phi_{0,a}) = \pi (\Phi_{0,b}) = \pi (\Phi_0)$ .

Let us now consider in detail what happens when the system approaches one of the critical values $Z_{c} = \pm i$ . For this purpose, we use the well-defined (but completely general and arbitrary) ansatz

$$
Z = Z _ {c} + \varepsilon , \qquad | \varepsilon | \ll 1, \qquad \varepsilon \in \mathbb {C}\tag{17}
$$

and expand the eigenvalues (4) and the line defining vectors $\chi_{\pm}$ (9) in terms of $\varepsilon$ . This gives the leading contributions to their Puiseux series representation [2, 44] in $\varepsilon^{1/2}$ as

$$
E _ {\pm} = E _ {0} \pm \varepsilon^ {1 / 2} \Delta E + o (\varepsilon^ {1 / 2}),\tag{18}
$$

$$
\Delta E := \omega \sqrt {2 Z _ {c}}, \qquad \chi_ {\pm} = \binom{1}{- Z _ {c}} \pm \varepsilon^ {1 / 2} \binom{0}{\sqrt {2 Z _ {c}}} + o (\varepsilon^ {1 / 2}).\tag{19}
$$

Following [32, 44], we expand the eigenvectors $\Phi_{\pm}(Z)$ of the diagonal spectral decomposition in the same local $\varepsilon^{1/2}$ approximation in terms of the Jordan chain (root) vectors $\Phi_{0,1}$ :

$$
\Phi_ {\pm} = \Phi_ {0} + \varepsilon^ {1 / 2} (b _ {0} \Phi_ {0} + b _ {1} \Phi_ {1}) + o (\varepsilon^ {1 / 2}),\tag{20}
$$

$$
b _ {0} = \pm \frac {Z _ {c}}{2 \omega} \Delta E, \qquad b _ {1} = \pm \Delta E.\tag{21}
$$

The coefficients $b_{0,1}$ are obtained by a two-step procedure. Substituting (17), (18), (20) into the eigenvalue equation and making explicit use of the chain relations (13) yields $b_{1}$ and leaves $b_{0}$ still undefined. The coefficient $b_{0}$ is found by comparing the line structures $^{9}$ of $\Phi_{\pm}$ with $\chi_{\pm}$ in (19).

It remains to match the fibre sections—what can be done in two ways. One may assume a single-scaling coefficient $c_{0}$ of the root space given and consider the coefficients $c_{\pm}$ of the sections $\Phi_{\pm}$ as derived objects. This leads to the identification $c_{+}=c_{-}=\sigma qc_{0}$ . Apart from this option, one may assume the scaling coefficients $c_{\pm}$ as primary objects and given so that they may take different values $c_{+}\neq c_{-}$ . Correspondingly, the scaling factor $c_{0}$ of the root space will then be fitted to $c_{\pm}$ so that it will take two different values

$$
c _ {0, \pm} = c _ {\pm} / (\sigma q).\tag{22}
$$

Both constructions are possible and compatible with the smooth fitting of the line structure encoded in the EP-limiting behaviour $\pi(\Phi_{\pm}) \rightarrow \pi(\Phi_{0})$ .

In a way similar to the above two-step procedure with subsequent fibre fitting the left eigenvectors can be obtained as

$$
\Xi_ {\pm} ^ {*} = \Xi_ {0} ^ {*} + \varepsilon^ {1 / 2} (b _ {0} \Xi_ {0} ^ {*} + b _ {1} \Xi_ {1} ^ {*}) + o (\varepsilon^ {1 / 2}),\tag{23}
$$

$$
d _ {\pm} = \sigma^ {*} q Z _ {c} d _ {0, \pm}.\tag{24}
$$

Here, $b_{0}$ and $b_{1}$ are the same as in (21) and full compatibility with the bi-orthogonality conditions (7) as well as with (8) is easily verified by direct calculation. In case of a single scaling factor $d_{0}$ of the dual root space, the coefficients $d_{\pm}$ will coincide $d_{+}=d_{-}=\sigma^{*}qZ_{c}d_{0}$ .

Combining (20) and (23) one finds the limiting behaviour of the inner products as

$$
\begin{array}{r} \langle \Xi_ {\pm} | \Phi_ {\pm} \rangle = 2 b _ {1} d _ {0, \pm} c _ {0, \pm} \varepsilon^ {1 / 2} + o (\varepsilon^ {1 / 2}) \\ = \frac {2 b _ {1}}{\omega Z _ {c}} d _ {\pm} c _ {\pm} \varepsilon^ {1 / 2} + o (\varepsilon^ {1 / 2}). \end{array}\tag{25}
$$

Here, one has to distinguish two normalization schemes. If one assumes the root vector sets $\{\Phi_{0},\Phi_{1}\},\{\Xi_{0},\Xi_{1}\}$ normalized, e.g., with $d_{0,\pm}c_{0,\pm}=1$ or $d_{0}c_{0}=1$ in (16) then the scalar product $\langle\Xi_{\pm}|\Phi_{\pm}\rangle$ of the eigenvectors in the diagonal spectral decomposition (see (25)) vanishes in the EP-limit. Starting, in contrast, from normalized eigenvector pairs $\{\Phi_{\pm},\Xi_{\pm}\}$ of diagonalizable Hamiltonians as in (11), i.e. with $\langle\Xi_{\pm}|\Phi_{\pm}\rangle=1$ , then the scale factor products $d_{\pm}c_{\pm}$ diverge as $d_{\pm}c_{\pm}\propto\varepsilon^{-1/2}$ for $\varepsilon\to0$ . Both normalization schemes are possible and compatible with the smooth limiting behaviour $\pi(\Phi_{\pm})\to\pi(\Phi_{0})$ of the lines encoded in $\chi_{\pm}(\varepsilon\to0)\to\chi_{0}$ [cf (12)]. We see that this special behaviour is only related to the fibre sections and not to the fibres (lines) themselves. The two incompatible normalization schemes simply indicate on the need for two complementary charts to cover the whole physical picture in the vicinity of a $2\times2$ Jordan block $J_{2}$ . One of these charts (we will call it the root vector chart) remains regular in the EP-limit, whereas the other (diagonal representation) chart becomes singular.

The situation is similar to the two affine charts required to cover the Riemann sphere $CP^{1}$ . Starting from homogeneous coordinates $(x_{0}, x_{1}) \in \mathbb{CP}^{1}$ one has the two affine charts $U_{0} \ni (1, x_{1}/x_{0}), x_{0} \neq 0$ and $U_{1} \ni (x_{0}/x_{1}, 1), x_{1} \neq 0$ . The mutually complementary affine coordinates $z := x_{1}/x_{0} \in C^{1}$ and $w := x_{0}/x_{1} \in C^{1}$ are then related to the well-known fractional transformation w = 1/z, so that the singular $|z| \to \infty$ limit in the z chart corresponds simply to the regular $w \to 0$ limit in the w chart. In other words, the two charts cover the north-pole and the south-pole regions of the Riemann sphere—a construction well known, e.g., from complex analysis and the description of magnetic monopoles [53].

Returning to the two-chart picture of the normalization, we see that the original Hilbert space $H = C^{2}$ should be extended by the set of infinite vectors $\Phi_{\pm}$ what can be naturally accomplished by embedding it into a larger projective space $H \hookrightarrow CP^{2}$ . Correspondingly, the fibres $\pi(\Phi_{\pm}) \times C^{*}$ should be extended as $\pi(\Phi_{\pm}) \times C^{*} \hookrightarrow \pi(\Phi_{\pm}) \times CP^{1}$ . A detailed discussion of these structures will be presented in [64]. An explicit embedding construction for simplified setups with coinciding scale factors $d_{\pm} = c_{\pm}$ is given in section 5 below.

## 4. Geometric phase

Following earlier studies $[27–31]$ , geometric phases $[65]$ of eigenvectors of non-Hermitian complex symmetric operators have been recently considered in $[32]$ for paths in parameter space encircling an EP. The results showed full agreement with the phase considerations of $[27]$ . In this section, we combine techniques for non-Hermitian systems $[28, 32]$ with explicit projective space parameterizations for Hermitian systems $[56]$ to provide an explicit projective-space-based derivation of the phase representation for non-Hermitian systems. Such an explicit reshaping of the results of $[56]$ to non-Hermitian setups seems missing up to now.

Following [28-30, 32] we consider an auxiliary system with a general non-Hermitian Hamiltonian $H(t)$ depending on a set of non-stationary parameters $\mathbf{X}(t) = [X_1(t), \ldots, X_m(t)] \in \mathcal{M}$ , $H(t) = H[\mathbf{X}(t)]$ and an EP hypersurface $\mathcal{V} \subset \mathcal{M}$ which is encircled by an appropriate loop $\Gamma$ in parameter space $\mathcal{M} \ni \Gamma = \{\mathbf{X}(t), t \in [0, T] : \mathbf{X}(0) = \mathbf{X}(T)\}$ . The evolution of the quantum system is governed by a usual Schrödinger equation for the right eigenvectors

$$
\mathrm{i} \partial_ {t} \Phi_ {n} (t) = H (t) \Phi_ {n} (t),\tag{26}
$$

and, due to the time invariance of the bi-orthogonal product

$$
\langle \Xi_ {m} (t) | \Phi_ {n} (t) \rangle = \delta_ {m n},\tag{27}
$$

by a complementary evolution law for the left eigenvectors [28]

$$
\mathrm{i} \partial_ {t} \Xi_ {m} (t) = H ^ {+} (t) \Xi_ {m} (t).\tag{28}
$$

For an adiabatic motion cycle $\Gamma\subset M$ with Hamiltonian $H[\mathbf{X}(T)]=H[\mathbf{X}(0)]$ , the resulting eigenvector $\Phi_{n}(t=T)$ of $H[\mathbf{X}(T)]$ must lay on the same line as the initial $\Phi_{n}(t=0)$ , i.e. it can only obtain an additional scaling factor which we parameterize as complex-valued phase

$$
\Phi_ {n} (T) = \mathrm{e} ^ {\mathrm{i} \phi_ {n} (T)} \Phi_ {n} (0), \qquad \phi_ {n} (T) \in \mathbb {C}.\tag{29}
$$

Due to the preserved orthonormality (27) the corresponding left eigenvectors evolve as

$$
\Xi_ {m} (T) = \mathrm{e} ^ {\mathrm{i} \phi_ {m} ^ {*} (T)} \Xi_ {m} (0).\tag{30}
$$

The complex phase $\phi_{n}(T)$ can be split into a dynamical component [28, 56]

$$
\epsilon_ {n} (T) = - \int_ {0} ^ {T} \frac {\langle \Xi_ {n} (t) | H (t) | \Phi_ {n} (t) \rangle}{\langle \Xi_ {n} (t) | \Phi_ {n} (t) \rangle} \mathrm{d} t\tag{31}
$$

and the geometric phase

$$
\gamma_ {n} (T) = \phi_ {n} (T) - \epsilon_ {n} (T).\tag{32}
$$

Adapting the techniques of [56], we calculate $\gamma_{n}(T)$ in terms of explicit projective space coordinates. Setting

$$
\Phi_ {n} (t) = c _ {n} (t) \chi_ {n} (t) =: [ z _ {0} (t), z _ {1} (t) ] ^ {T} = z _ {0} (t) [ 1, w (t) ] ^ {T}
$$

$$
\Xi_ {m} (t) = d _ {m} ^ {*} (t) \chi_ {m} ^ {*} (t) =: [ y _ {0} (t), y _ {1} (t) ] ^ {T} = y _ {0} (t) [ 1, v (t) ] ^ {T}\tag{33}
$$

(omitting in the projective space coordinates the mode indices m, n) one identifies

$$
\phi_ {n} (T) = - \mathrm{i} \ln [ z _ {0} (T) / z _ {0} (0) ]\tag{34}
$$

and obtains from (26), (31) and (32) the differential 1-form of the geometric phase as

$$
\mathrm{d} \gamma = - \mathrm{i} \frac {\mathrm{d} z _ {0}}{z _ {0}} + \mathrm{i} \frac {y _ {0} ^ {*} \mathrm{d} z _ {0} + y _ {1} ^ {*} \mathrm{d} z _ {1}}{y _ {0} ^ {*} z _ {0} + y _ {1} ^ {*} z _ {1}} = \mathrm{i} \frac {v ^ {*} \mathrm{d} w}{1 + v ^ {*} w ^ {*}}.\tag{35}
$$

Due to the symmetry (8) between left and right eigenlines this simplifies to

$$
\mathrm{d} \gamma = \mathrm{i} \frac {\chi^ {T} d \chi}{\chi^ {T} \chi} = \frac {\mathrm{i}}{2} \mathrm{d} \ln (1 + w ^ {2}).\tag{36}
$$

Similar to results on Hermitian systems [56] the differential 1-form (35) is independent of the coordinates $z_0$ and $y_0$ along the fibres and, hence, defines a horizontal connection over the projective Hilbert space of the system. The mere difference in the definitions of the projective structures is in $\mathbb{CP}^1 = S^3 / S^1$ for Hermitian systems, whereas $\mathbb{CP}^1 = \mathcal{H}^* / \mathbb{C}^*$ for non-Hermitian ones [61].

Let us now apply the general technique to the concrete $2 \times 2$ -matrix model (1). Parameterizing the cycle around the EP by (17) with

$$
\varepsilon = r \mathrm{e} ^ {\mathrm{i} \alpha}, \qquad \alpha \in [ 0, 2 \pi ], \qquad 0 <   r \ll 1\tag{37}
$$

one reproduces the 1-forms of the geometric phases of [29]

$$
\mathrm{d} \gamma_ {\pm} = \frac {\mathrm{i}}{4} \mathrm{d} \ln \varepsilon = - \frac {1}{4} \mathrm{d} \alpha + \frac {\mathrm{i}}{4} \mathrm{d} \ln r.\tag{38}
$$

In a similar way one obtains the same 1-forms for the corresponding left eigenvectors $\Xi_{\pm}$ . Upon integration over a full cycle $\alpha(T) = \alpha(0) + 2\pi$ , $r(T) = r(0)$ one finds

$$
\gamma_ {\pm} (T) - \gamma_ {\pm} (0) = - \frac {\pi}{2}.\tag{39}
$$

The relation between geometric phases $\gamma_{\pm}$ and the cycle phase $\alpha$ can be gained also directly from the structure of the sections $\Phi_{\pm}$ . These sections may be arranged as columns of a diagonalizable $2 \times 2$ -matrix

$$
\Phi (\alpha) := [ \Phi_ {+} (\alpha), \Phi_ {-} (\alpha) ].\tag{40}
$$

The evolution along a cycle is then encoded in the transformation matrix $W(\alpha) = \Phi(\alpha) [\Phi(0)]^{-1}$ which for small $\varepsilon$ with $0 \neq |\varepsilon| \ll 1$ can be calculated from the representation (19) as

$$
W (\alpha) = \left[ \begin{array}{c c} \mathrm{e} ^ {- \mathrm{i} \frac {\alpha}{4}} & 0 \\ 2 \mathrm{i} Z _ {c} \sin \left(\frac {\alpha}{4}\right) & \mathrm{e} ^ {\mathrm{i} \frac {\alpha}{4}} \end{array} \right].\tag{41}
$$

The elements $W(\alpha)$ form an Abelian parabolic subgroup P of the complex orthogonal group $O(2, \mathbb{C}) \supset P$ (see, e.g., [62, 63, 66])

$$
W (\alpha + \beta) = W (\alpha) W (\beta) = W (\beta) W (\alpha)\tag{42}
$$

corresponding to the mapping $\mathrm{e}^{\mathrm{i}\alpha}\in S^{1}\approx U(1)\mapsto P\subset O(2,\mathbb{C})$ . For full cycles $\alpha = 2\pi N$ , $N\in \mathbb{Z}$ they yield the monodromy transformations [67]

$$
\begin{array}{l} W _ {0} := W (0) = I _ {2}, \qquad W _ {1} := W (2 \pi) = \left( \begin{array}{c c} - \mathrm{i} & 0 \\ 2 \mathrm{i} Z _ {c} & \mathrm{i} \end{array} \right), \\ W _ {2} := W (4 \pi) = W ^ {2} (2 \pi) = - I _ {2}, \\ W _ {3} := W (6 \pi) = - W (2 \pi), \qquad W (8 \pi) = I _ {2} = W _ {0}. \end{array}\tag{43}
$$

The geometric phase (38) and the monodromy transformations (43) show the typical fourfold covering of the mapping $\alpha \mapsto \gamma$ which was earlier described in [19, 25-27, 32] and experimentally demonstrated in [24]. A cycle around the EP in parameter space $\mathcal{M}$ has to be passed four times in order to produce one full $2\pi$ -cycle in the geometric phase. The (non-oriented) eigenline $\pi (\Phi \neq \Phi_0)\in \mathbb{CP}^1$ is already recovered after two cycles $\pi (W_2\Phi) = \pi (-\Phi) = \pi (\Phi)$ —similar to the eigenvalue $E$ which for the $2\times 2$ -matrix lives on a two-sheeted Riemann surface with the same two branch points $Z_{c} = \pm \mathrm{i}$ as the line bundle. For the isotropic limiting vector $\Phi_0$ at the EP it holds (due to (12))

$$
W (\alpha) \Phi_ {0} = \mathrm{e} ^ {- \mathrm{i} \alpha / 4} \Phi_ {0}\tag{43a}
$$

so that the parabolic subgroup $P \ni W(\alpha)$ can be identified as invariance group of the projective line at the EP

$$
\pi (W (\alpha) \Phi_ {0}) = \pi (\mathrm{e} ^ {- \mathrm{i} \alpha / 4} \Phi_ {0}) = \pi (\Phi_ {0}).\tag{43b}
$$

We note that despite the non-Hermiticity of the Hamiltonian H the geometric phase is purely real—as for Hermitian systems. Relations (38), (39) show that possible imaginary phase contributions (which would result in a re-scaling of the eigenvectors $\Phi_{\pm}$ ) are cancelled by the closed-cycle condition $r(T) = r(0)$ . Hence, the non-preservation of the vector norm in non-Hermitian systems is induced solely by a complex dynamical phase $\epsilon$ and requires the presence of the bi-orthogonal basis where a decaying behaviour of the right eigenvectors $^{10}$

$$
\Phi_ {n} \propto \mathrm{e} ^ {- \mathrm{i} \epsilon_ {n} t - \frac {\Gamma_ {n}}{2} t}, \qquad \langle \Phi_ {n} | \Phi_ {n} \rangle = | | \Phi_ {n} | | ^ {2} \propto \mathrm{e} ^ {- \Gamma_ {n} t}\tag{44}
$$

is necessarily connected with increasing vector norms of the dual left eigenvectors

$$
\Xi_ {m} \propto \mathrm{e} ^ {- \mathrm{i} \epsilon_ {m} t + \frac {\Gamma_ {m}}{2} t}, \qquad | | \Xi_ {m} | | ^ {2} \propto \mathrm{e} ^ {\Gamma_ {m} t}\tag{45}
$$

so that indeed $\langle \Xi_m|\Phi_n\rangle = \delta_{mn}$ . This behaviour is well known from resonances and Gamow vector theory (see, e.g. [7]).

Comparison of (8) and (44), (45) shows that the formal ansatz $\Xi_{m} = \Phi_{m}^{*}$ for the eigenvectors of the complex symmetric Hamiltonian (cf [11, 13, 14]) can be used only for instantaneous eigenvectors at a single fixed $t = t_{0}$ (which formally can be set to $t_{0} = 0$ ) as well as for the subclass of real symmetric matrices (when the system becomes Hermitian and norm-preservation of the eigenvectors holds). In contrast, for explicitly time-dependent non-Hermitian setups it only holds $\Xi_{m}(t) \propto \Phi_{m}^{*}(t)$ , i.e. the dual basis vectors necessarily live on complex conjugate lines (fibres) $\pi[\Xi_{m}(t)] = (\pi[\Phi_{m}(t)])^{*}$ but with $\Xi_{m}(t) \neq \Phi_{m}^{*}(t)$ for $t \neq t_{0}$ .

Aspects of the parameter dependence of the phases and scalings in an instantaneous picture with $\Xi_{m} = \Phi_{m}^{*}$ together with the explicit EP-limit $\varepsilon \rightarrow 0$ are the subject of the following section.

## 5. Instantaneous picture

In modern quantum physics not only the properties of natural systems such as nuclei or atoms are of interest, but rather the design and functionality of artificial quantum-system-based devices plays an essential role. In many cases, for the understanding of the dynamical features of these man-made quantum systems the time dependence is of minor interest. The properties of these systems are mainly governed by the position and number of EPs, i.e. the level crossing points in the complex plane, and their dependence on external control parameters. In this context it appears natural to study the parameter dependence of level energies and widths as well as the corresponding eigenvectors in terms of the instantaneous picture with $\Xi_{m} = \Phi_{m}^{*}$ and $c_{\pm} = d_{\pm}$ . This picture is compatible with the Hermitian limit when $\operatorname{Im} \epsilon_{1,2} = \operatorname{Im} \omega = 0$ in (1) and the condition $\Xi_{m} = \Phi_{m}^{*}$ is fulfilled by definition. $^{11}$

We have to distinguish the two possible normalization schemes—the root-vector-based normalization (16) with $d_{0}c_{0}=1$ or $d_{0,\pm}c_{0,\pm}=1$ and the diagonal-representation-based normalization (11) with $d_{\pm}c_{\pm}\chi^{T}\chi=1$ .

In the root-vector-based normalization scheme the conditions $c_{\pm} = d_{\pm}$ and $d_{0,\pm}c_{0,\pm} = 1$ together with the two relations (22) and (24) imply (in leading-order approximation in $\varepsilon$ ) $c_{0,\pm} = d_{0,\pm}$ and, hence, $c_{0,\pm} = \kappa$ with $\kappa = \pm 1$ (independently of the signs $\pm$ in the index of $c_{0,\pm}$ ) as well as $c_{\pm} = d_{\pm} = \kappa \sigma q$ . We see that in leading-order approximation in $\varepsilon$ the scaling factors $c_{\pm} = d_{\pm}$ are rigidly fixed and independent of $\varepsilon$ . A geometric phase (necessarily induced via an $\varepsilon$ dependence) appears incompatible with this normalization.

Let us now turn to the diagonal-representation-based normalization (11). In the EP-limit $\varepsilon \rightarrow 0$ , the normalization condition (11) for the eigenvectors (5) yields

$$
\begin{array}{r} 1 = \langle \Xi_ {\pm} | \Phi_ {\pm} \rangle = \Phi_ {\pm} ^ {T} \Phi_ {\pm} = [ 1 + (Z \mp \sqrt {Z ^ {2} + 1}) ^ {2} ] c _ {\pm} ^ {2} \\ \approx \mp 2 Z _ {c} \sqrt {2 Z _ {c} \varepsilon} c _ {\pm} ^ {2} \end{array}\tag{46}
$$

and we find the expected divergent scaling factors as

$$
c _ {\pm} ^ {2} \approx \mp 2 ^ {- 3 / 2} Z _ {c} ^ {- 3 / 2} \varepsilon^ {- 1 / 2} \quad \Longrightarrow \quad c _ {\pm} \sim \varepsilon^ {- 1 / 4}.\tag{47}
$$

$^{11}$ When compatibility with the Hermitian limit is not required, then the bi-orthonormalization constraints $d_{0,\pm}$ $c_{0,\pm}=1$ or $d_{\pm}c_{\pm}\chi_{\pm}^{T}\chi_{\pm}=1$ fix only two of the four constants $c_{0,\pm}, d_{0,\pm}$ or $c_{\pm}, d_{\pm}$ and the remaining two can be chosen arbitrarily. For instance, one may set the eigenvector scaling factors as $c_{0,\pm}=C\neq1$ or $c_{\pm}=1$ so that $d_{0,\pm}=C^{-1}$ or $d_{\pm}=(\chi_{\pm}^{T}\chi_{\pm})^{-1}$ what would define instantaneous pictures not compatible with the Hermitian limit.

On the one hand, (47) reproduces the local four-sheeted Riemann surface structure connected with the geometric phase (38), (41), i.e. a four-fold winding around the EP is needed to return to an eigenvector pointing into the same complex direction as a starting vector. (In contrast to the root-vector-normalization scheme full compatibility with the geometric phase setup holds.)

On the other hand, it leads to divergent vector norms

$$
| | \Phi_ {\pm} | | ^ {2} = \langle \Phi_ {\pm} | \Phi_ {\pm} \rangle \approx 2 | c _ {\pm} | ^ {2} \approx | 2 \varepsilon | ^ {- 1 / 2}\tag{48}
$$

for $\varepsilon \to 0$ . As it was indicated in section 3, the corresponding singularity can be naturally resolved by embedding the original Hilbert space $\mathcal{H} \approx \mathbb{C}^2$ into its projective extension $\mathcal{H} \hookrightarrow \mathbb{CP}^2 \ni \phi = (u_0, u_1, u_2)$ so that the set of infinite vectors becomes well defined. Interpreting the two components $z_0$ and $z_1$ of the vector (fibre section)

$$
\Phi = c (1, w) = (z _ {0}, z _ {1}) \in \mathbb {C} ^ {2}\tag{49}
$$

as affine coordinates on the chart $U_{2}\ni \left(\frac{u_{0}}{u_{2}},\frac{u_{1}}{u_{2}},1\right),u_{2}\neq 0,U_{2}\subset \mathbb{CP}^{2}$

$$
\Phi = (c, c w) \hookrightarrow (c, c w, 1),\tag{50}
$$

we can identify $\Phi$ with the point $\phi \in CP^{2}$ with homogeneous coordinates

$$
\phi = (u _ {0}, u _ {1}, u _ {2}) = (1, w, c ^ {- 1}).\tag{51}
$$

The singularity $|c| \to \infty$ at the EP corresponds then simply to the point $\phi_{0} = (1, w, 0) \in \mathbb{CP}^{2}$ with $u_{2} = 0$ and we see that the affine chart $U_{2} \in C P^{2}$ is no longer appropriate for covering $\phi_{0}$ . This is in contrast to the root-vector-normalization scheme where c is fixed and the chart $U_{2}$ remains suitable for the covering. Within the present diagonal-representation normalization, instead, $\phi_{0}$ should be parameterized in terms of affine coordinates corresponding to one of the charts $^{12}$ $U_{0}$ or $U_{1}$ with $u_{0} \neq 0$ or $u_{1} \neq 0$ . Most natural for our representation (49), (51) is the affine chart $U_{0} \ni \left(1, \frac{u_{1}}{u_{0}}, \frac{u_{2}}{u_{0}}\right)$ which can be used for a suitable projective representation of the fibre sections $\Phi$ :

$$
\Phi \approx (1, w, c ^ {- 1}) = (\chi^ {T}, c ^ {- 1}) \approx (\pi (\Phi), c ^ {- 1}).\tag{52}
$$

Interpreting the normalization condition (46) as constraint on the affine coordinates of $\Phi$ in the chart $U_{2}$ ,

$$
0 = \Phi^ {T} \Phi - 1 = \frac {u _ {0} ^ {2}}{u _ {2} ^ {2}} + \frac {u _ {1} ^ {2}}{u _ {2} ^ {2}} - 1,\tag{53}
$$

one immediately sees that it is equivalent to the conic (singular quadric) $^{13}$

$$
u _ {0} ^ {2} + u _ {1} ^ {2} - u _ {2} ^ {2} = 0\tag{54}
$$

in homogeneous coordinates which cover the whole $CP^{2}$ . This conic remains regular at EPs which merely correspond to configurations with $u_{2}=0$ . In terms of $(\chi^{T},c^{-1})$ notations it reads

$$
\chi^ {T} \chi - c ^ {- 2} = 0.\tag{55}
$$

It is clear that the conic construction is straightforwardly extendable to Hilbert space embeddings $H = C^{n} \hookrightarrow CP^{n}$ of any dimension n. We arrive at the conclusion that the appropriate state space for open quantum systems in an instantaneous setting will be related to the projective extension $CP^{n}$ of the original Hilbert space $H = C^{n}$ with states identified with conics $\sum_{k=0}^{n-1}u_{k}^{2}-u_{n}^{2}=0$ . This is in contrast to Hermitian systems where it is sufficient to project the Hilbert space $H^{*}=C^{n}-\{0\}$ down to the base space $CP^{n-1}$ , i.e. $\pi:H^{*}\to\mathbb{P}(\mathcal{H}^{*})\approx\mathbb{CP}^{n-1}$ . In non-Hermitian setups each fibre $\pi(\Phi)\times\mathbb{C}^{*}$ should be supplemented by $\infty$ . This suggests to extend them to $\pi(\Phi)\times\mathbb{CP}^{1}$ . From the above construction, we see that the singular behaviour with regard to the two affine charts is only related to the scale factors $c\in\mathbb{CP}^{1}$ , whereas $\pi(\Phi)$ behaves smoothly and regular. On its turn, this suggests to reconsider the model-dependent physical interpretation of the eigenvector self-orthogonality (isotropy) and the corresponding diverging or non-diverging sensitivity in perturbation expansions like in [51, 52] as a result of divergent or non-divergent normalization constants.

The Hilbert space extension $\mathcal{H} = \mathbb{C}^2\hookrightarrow \mathbb{CP}^2$ together with the observed simultaneous scaling of the whole root space $\mathfrak{S}_{\lambda}$ obtained in section 3, the upper and lower triangular (parabolic subgroup type) structure of the $\mathfrak{S}_{\lambda}$ -related matrices in (A.7), (A.8) as well as the parabolic subgroup structure (41) at EPs provide strong indications that the natural structure at EPs is connected with projective flags [63]. A study of Jordan chain related flag bundles and the mappings between their complementary affine charts will be presented in [64].

Returning to the $\varepsilon \rightarrow 0$ limit in (47) we see that

$$
\begin{array}{r c l} \frac {c _ {+} ^ {2}}{c _ {-} ^ {2}} \to - 1 & \Longrightarrow & \frac {c _ {+}}{c _ {-}} \to \pm \mathrm{i} \\ & \Longrightarrow & \Phi_ {+} \to \pm \mathrm{i} \Phi_ {-}, \end{array}\tag{56}
$$

i.e. the two eigenvectors (fibre sections) $\Phi_{+}$ , $\Phi_{-}$ are phase-shifted one relative to the other by $\pm i$ when tending to their common coalescence line at $\varepsilon = 0$ : $\pi(\Phi_{+}) = \pi(\Phi_{-}) = \pi(\Phi_{0})$ . We note that this relative $\pm i$ phase shift of the vectors $\Phi_{+}$ , $\Phi_{-}$ is generic for models in their instantaneous picture and with $d_{\pm} = c_{\pm}$ and normalization $\langle \Xi_{\pm} | \Phi_{\pm} \rangle = 1$ .

A further result which immediately follows from (47) is the typical distance-dependent phase-jump behaviour in the vicinity of the EP. In a sufficiently close vicinity of an EP ( $|\varepsilon| \ll 1$ ) any sufficiently smooth trajectory in an underlying parameter space can be roughly approximated by a straight line segment with an effective parametrization of the type

$$
\varepsilon = \mathrm{e} ^ {\mathrm{i} \alpha_ {0}} (\rho + \mathrm{i} s), \qquad s \in [ - s _ {0}, s _ {0} ] \subset \mathbb {R},\tag{57}
$$

where $\alpha_{0}=$ const fixes the direction orthogonal to the effective trajectory in the complex $\varepsilon$ plane and $\rho$ is the minimal distance $\rho=|\varepsilon(s=0)|$ of this trajectory to the EP. The parameter along the path is $s\in[-s_{0},s_{0}]\subset R$ . This parametrization gives:

$$
\begin{array}{l} [ \varepsilon (s) ] ^ {- 1 / 4} = \mathrm{e} ^ {- \mathrm{i} \frac {\alpha_ {0}}{4} - \mathrm{i} \theta (s)} | \varepsilon (s) | ^ {- 1 / 4} \\ | \varepsilon (s) | = (\rho^ {2} + s ^ {2}) ^ {1 / 2} \\ \theta (s) = \frac {1}{4} \arctan (s / \rho) \in (- \pi / 8, \pi / 8) \end{array}\tag{58}
$$

and we observe that the minimal distance $\rho$ between the parameter trajectory and the EP defines the smoothness of the phase changes. The closer the path approaches the EP the more it will take the form of a Heaviside step function with jump height $\pi/4$ :

$$
\theta (s; \rho \rightarrow 0) \rightarrow \frac {\pi}{4} \left[ \Theta (s) - \frac {1}{2} \right].\tag{59}
$$

The phase-jump behaviour can be used as an implicit indicator of a possible close location of an EP—a fact especially useful in numerical studies of systems with complicated parameter dependence, but where phases of eigenvectors can be easily extracted. Jumps $\pm\pi/4$ of wavefunction phases have been observed numerically in [69] for the model Hamiltonian (1) and in [14] for the special case of a small quantum billiard. According to these results, the phases of the components change smoothly (as a function of a certain control parameter) in approaching the EP and jump by $\pi/4$ at the smallest distance from this point. Other phase-jump values are possible, but require especially tuned paths.

## 6. Phase rigidity

In numerical studies of man-made open quantum systems depending in a complicated way on several parameters $\mathbf{X} = (X_{1}, \ldots, X_{m}) \in \mathcal{M}$ , it is usually important to know how close a given configuration is located to an EP. EPs dominate the system behaviour also in their vicinities, spectral bands may merge at EPs [51] or the transmission properties of quantum dots (QDs) may become optimal at EPs [70]. A measure for the distance between a given point in parameter space and a closely located EP would provide a convenient tool for adjusting and tuning parameters so that a system may be ‘moved’ in parameter space towards or away from this EP.

In [70], it has been shown numerically that within the instantaneous picture $(\Phi = \Xi^{*})$ an appropriate measure for the detection of EP vicinities is the fraction

$$
r = \frac {\Phi^ {T} \Phi}{\langle \Phi | \Phi \rangle}.\tag{60}
$$

We note that originally similar fractions have been introduced in $[71]$ to describe the transitions between Hamiltonian ensembles with orthogonal and unitary symmetry in Hermitian quantum chaotic systems. There the square modulus $|r|^{2}$ was dubbed ‘phase rigidity’. In our considerations of non-Hermitian systems we use this term in loose analogy for r itself.

Decomposing $\Phi$ into real and complex components $\Phi = \Phi_{r} + i\Phi_{i}$ , we find from the normalization that

$$
\Phi^ {T} \Phi = 1 = \Phi_ {r} ^ {T} \Phi_ {r} - \Phi_ {i} ^ {T} \Phi_ {i}, \qquad \Phi_ {r} ^ {T} \Phi_ {i} = 0\tag{61}
$$

and $^{14}$ hence that the norm is bounded below

$$
\left| \left| \Phi \right| \right| ^ {2} = \langle \Phi | \Phi \rangle = \Phi_ {r} ^ {T} \Phi_ {r} + \Phi_ {i} ^ {T} \Phi_ {i} = 2 \Phi_ {i} ^ {T} \Phi_ {i} + 1 \geqslant 1.\tag{62}
$$

The phase rigidity can be expressed as

$$
r = \frac {1}{| | \Phi | | ^ {2}} \in [ 0, 1 ],\tag{63}
$$

where according to (48) for the EP-limit $\varepsilon \rightarrow 0$ holds

$$
r \approx | 2 \varepsilon | ^ {1 / 2} \to 0.\tag{64}
$$

The opposite limit $r \rightarrow 1$ is reached when the channel coupling $\omega$ in the Hamiltonian (1) vanishes, i.e. when the interaction between the two decaying resonance states tends to zero and any eigenvector can be taken purely real-valued in the instantaneous picture.

Finally, we note that for certain quantum dot systems the phase rigidity r is closely related to the transmission properties of these systems. The capability of corresponding numerical studies (including the visualizations of transmission and phase rigidity ‘landscapes’ over parameter space) has been recently demonstrated in $[70]$ .

## 7. PT-symmetric models

Toy model Hamiltonians of $2 \times 2$ -matrix type have been often used as a test ground in PT-symmetrically extended quantum mechanics (PTSQM) [33–35]. They can be obtained from non-Hermitian complex symmetric $2 \times 2$ -matrix Hamiltonians by imposing a PT-symmetry constraint. In a suitable parametrization they have the form

$$
H = \left( \begin{array}{c c} r   \mathrm{e} ^ {\mathrm{i} \theta} & s \\ s & r   \mathrm{e} ^ {- \mathrm{i} \theta} \end{array} \right), \qquad r, s, \theta \in \mathbb {R}\tag{65}
$$

and commute with the operator PT

$$
[ \mathcal {P T}, H ] = 0, \qquad \mathcal {P} = \left( \begin{array}{c c} 0 & 1 \\ 1 & 0 \end{array} \right).\tag{66}
$$

Here, P is the parity reflection operator and T—the time inversion (acting as complex conjugation). The eigenvalues of H are

$$
E _ {\pm} = r \cos (\theta) \pm \sqrt {s ^ {2} - r ^ {2} \sin^ {2} (\theta)},\tag{67}
$$

and the corresponding eigenvectors can be represented as [35]

$$
\begin{array}{l} | E _ {+} \rangle = \frac {\mathrm{e} ^ {\mathrm{i} \alpha / 2}}{\sqrt {2 \cos (\alpha)}} \binom{1}{\mathrm{e} ^ {- \mathrm{i} \alpha}} =: c _ {+} \chi_ {+} \\ | E _ {-} \rangle = \frac {\mathrm{i} \mathrm{e} ^ {- \mathrm{i} \alpha / 2}}{\sqrt {2 \cos (\alpha)}} \binom{1}{- \mathrm{e} ^ {\mathrm{i} \alpha}} =: c _ {-} \chi_ {-}, \end{array}\tag{68}
$$

where

$$
\sin (\alpha) = \frac {r}{s} \sin (\theta).\tag{69}
$$

With regard to the indefinite (Krein space type [38]) PT inner product $(u, v) = \mathcal{PT} u \cdot v$ the vectors are normalized as

$$
(E _ {\pm}, E _ {\pm}) = \pm 1, \qquad (E _ {\pm}, E _ {\mp}) = 0.\tag{70}
$$

The indefinite $\mathcal{PT}$ inner product is then mapped by the dynamical operator $\mathcal{C}$ with $[\mathcal{C}, H] = 0$ and

$$
\mathcal {C} = \frac {1}{\cos (\alpha)} \left( \begin{array}{c c} \mathrm{i} \sin (\alpha) & 1 \\ 1 & - \mathrm{i} \sin (\alpha) \end{array} \right)\tag{71}
$$

(see, e.g., [35]) into the positive definite (Hilbert space type) $\mathcal{CPT}$ inner product $((u,v)) = \mathcal{CPT}u\cdot v$ which yields

$$
((E _ {\pm}, E _ {\pm})) = 1, \qquad ((E _ {\pm}, E _ {\mp})) = 0.\tag{72}
$$

Let us now reshape the model in terms of the EP-relevant notations of section 2. A simple comparison of (1), (3) with (65), (69) shows that

$$
Z = \mathrm{i} \sin (\alpha)\tag{73}
$$

and hence that

$$
\mathcal {C} = \frac {1}{\cos (\alpha)} \left( \begin{array}{c c} Z & 1 \\ 1 & - Z \end{array} \right)\tag{74}
$$

and that the model is actually one-parametric with essential parameter Z. Together with (2) the parametrization (74) leads to a representation of the Hamiltonian (65) as

$$
H = E _ {0} I _ {2} + s \cos (\alpha) \mathcal {C}, \qquad E _ {0} = r \cos (\theta)\tag{75}
$$

and $[C, H] = 0$ is fulfilled trivially.

The compatibility of the PT and the CPT inner products (70), (72) with the bi-orthogonality relations (7) is ensured by the fact that for an eigenvector $\Phi = c(1, b)^{T}$ exact PT symmetry requires $\mathcal{PT}\Phi \propto \Phi$ and, hence, $c^{*}b^{*}(1, 1/b^{*})^{T} \propto c(1, b)^{T}$ so that $|b|^{2} = 1$ . For such vectors it holds $PT\Phi \propto \Xi^{*}$ and due to the dynamically tuned C also $CPT\Phi \propto \Xi^{*}$ . As a result one finds $CPT\Phi_{k} \cdot \Phi_{l} \propto PT\Phi_{k} \cdot \Phi_{l} \propto \Xi_{k}^{+}\Phi_{l}$ , and full compatibility of the bi-orthogonality with the PT and CPT inner products is established.

From (73), we see that possible EPs are solely defined by the value of $\alpha$ . From $Z_{c} = \pm i$ , we find the corresponding critical $\alpha_{c}$ as

$$
\alpha_ {c} = \pm \pi / 2 + 2 N \pi , \qquad N \in \mathbb {Z}.\tag{76}
$$

Furthermore, it follows from (69) that a purely Hermitian model with $\theta = n\pi, n \in \mathbb{Z}$ corresponds to $\alpha = N\pi, N \in \mathbb{Z}$ . Exact $\mathcal{PT}$ symmetry is preserved for $\alpha \in \mathbb{R} - \{\pi/2 + \pi\mathbb{Z}\}$ , and the corresponding models are parameterized by elements $Z$ belonging to the purely imaginary straight line segment connecting the two EPs, i.e. by $Z \in (-i, i)$ , $\operatorname{Re} Z = 0$ .

According to (68), at the EPs the eigenvectors lie on the same line $\pi(|E_{+}\rangle) = \pi(|E_{-}\rangle) \approx \chi_0 = (1, Z_c)^T$ and their norms diverge for $\alpha \to \alpha_c$ like

$$
| | | E _ {\pm} \rangle | | ^ {2} = \langle E _ {\pm} | E _ {\pm} \rangle \approx \frac {1}{| \cos (\alpha) |} \to \infty .\tag{77}
$$

The operator C in (71) shows the same singular behaviour, i.e. the C-induced mapping between the Krein space and the Hilbert space breaks down at the EPs. In analogy to the singularity resolution presented in section 5, we may map the vectors $|E_{\pm}\rangle \in C^{2}$ into elements from the affine chart $U_{2} \subset CP^{2}$ corresponding to points $e_{\pm} \in CP^{2}$ with homogeneous coordinates

$$
| E _ {\pm} \rangle \mapsto e _ {\pm} = \left(\chi_ {\pm} ^ {T}, c _ {\pm} ^ {- 1}\right).\tag{78}
$$

The original normalization via $\mathcal{PT}$ inner product $\mathcal{PT}|E_{\pm}\rangle \cdot |E_{\pm}\rangle = 1$ acts then as generalized conic

$$
\mathcal {P T} \chi_ {\pm} \cdot \chi_ {\pm} - \left(\mathcal {T} c _ {\pm} ^ {- 1}\right) c _ {\pm} ^ {- 1} = 0\tag{79}
$$

which remains regular in the EP-limit $\alpha\to\alpha_{c}$ , but shows the typical EP-related self-orthogonality (isotropy) of the lines $PT\chi_{\pm}\cdot\chi_{\pm}\to0$ . Again we arrive at the conclusion that the original Hilbert space $H=C^{2}$ should be projectively embedded into $CP^{2}$ in order to accommodate EP-related singularities.

Finally, we note that the recently uncovered solutions of the PT-symmetric brachistochrone problem with vanishing optimal passage time $[34]$ occur for $\alpha = \pi/2$ what according to $(76)$ can be identified as an EP-regime $^{15}$ . This fact appears compatible with the results of $[57]$ where a vanishing passage time was reported for arbitrary non-Hermitian Hamiltonians. In this regard it is natural to conjecture that a vanishing optimal passage time might be a generic EP-related feature of non-Hermitian systems not necessarily restricted to PTSQM models.

## 8. Conclusion

In the present paper, we considered projective Hilbert space structures in the vicinity of EPs. Starting from a leading-order Puiseux-expansion of the bi-orthogonal eigenvectors of a non-Hermitian (complex symmetric) diagonalizable $2 \times 2$ -matrix Hamiltonian in terms of root vectors (algebraic eigenvectors) at an EP the normalization divergence of the eigenvectors in the EP-limit has been parameterized. It has been shown that the natural projective line structure related to the eigenvectors of the diagonal Hamiltonian has to be replaced at an EP by a higher dimensional projective structure in which all the root vectors of a Jordan block scale simultaneously with the same single factor. For a simplified setup with left eigenvectors equated to their complex conjugate right counterparts, the normalization divergence has been resolved by embedding the original Hilbert space $H = C^{2}$ into its projective extension $H \hookrightarrow CP^{2}$ . Eigenvectors normalized according to the diagonalizable Hamiltonian and eigenvectors with a normalization inherited from the root vector normalization live then merely in different (complementary) affine charts of $CP^{2}$ . The states themselves can be interpreted as conics in $CP^{2}$ . The line structure of the states behaves smoothly and independently of these charts and their possibly singular transition functions. This indicates on the possibility of a technically efficient description of the global behaviour of the non-Hermitian system by factoring the eigenvectors in globally smoothly varying non-singular projective line components $\pi(\Phi_{k})$ and possibly diverging scale factors $^{16} c_{k}$ .

With the help of the Puiseux-expanded eigenvectors it has been shown that the geometric phase obtained on circles around EPs of complex symmetric Hamiltonians is purely real-valued and that the corresponding monodromy transformations are induced by an Abelian parabolic subgroup of $O(2, \mathbb{C})$ . Furthermore, the Puiseux expansion has been used to explain phase jumps which in prior work had been numerically observed in the vicinity of EPs. An analytical foundation for the usefulness of the phase rigidity as a distance measure to EPs has been provided. Finally, a PT-symmetric model has been studied. It has been shown that the EP-related singularities show up not only in the normalization conditions of the eigenvectors but also in the dynamical symmetry operator C. The normalization singularity has been resolved via a projective extension of the original Hilbert space. From the singularity structure it has been conjectured that the zero passage time effect in the brachistochrone problem of non-Hermitian Hamiltonians might be a generic EP-related artefact.

## Acknowledgments

We thank Hugh Jones and Andreas Fring for useful comments on $[34, 57]$ . This work has been supported by the German Research Foundation DFG, grant GE 682/12-3 (UG) and by the grants RFBR-06-02-16719, SS-5103.2006.2 (BFS).

## Appendix. Jordan normal forms for complex symmetric $2 \times 2$ matrices

At the EPs with $Z_{c} = \pm i =: \mu i$ the matrix

$$
H (Z _ {c}) - E _ {0} I _ {2} = \omega \left( \begin{array}{c c} Z _ {c c} & 1 \\ 1 & - Z _ {c c} \end{array} \right) =: M\tag{A.1}
$$

is related to its Jordan normal form $J_{2}(0) = \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix}$ by a similarity transformation [64],

$$
M = P R J _ {2} (0) R ^ {- 1} P ^ {- 1}.\tag{A.2}
$$

From the symmetry properties

$$
M = M ^ {T}, \qquad J _ {2} (0) = S _ {2} J _ {2} ^ {T} (0) S _ {2}\tag{A.3}
$$

with $S_{2} = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$ and $P^{2} = S_{2}$ one finds

$$
P = \frac {\mathrm{e} ^ {\mathrm{i} \mu   \frac {\pi}{4}}}{\sqrt {2}} \left( \begin{array}{c c} 1 & - \mathrm{i} \mu \\ - \mathrm{i} \mu & 1 \end{array} \right), \qquad P = P ^ {T} = (P ^ {- 1}) ^ {+}\tag{A.4}
$$

and

$$
R = \left( \begin{array}{c c} q & 0 \\ 0 & q ^ {- 1} \end{array} \right), \qquad q := \sqrt {2 \omega}.\tag{A.5}
$$

The elementary Jordan block $J_{2}(0)$ has right- and left-root vectors $\Theta_{0}, \Theta_{1}$ and $\Psi_{0}, \Psi_{1}$ satisfying

$$
\begin{array}{l l} J _ {2} (0) \Theta_ {0} = 0, & \qquad J _ {2} (0) \Theta_ {1} = \Theta_ {0} \\ J _ {2} ^ {T} (0) \Psi_ {0} = 0, & \qquad J _ {2} ^ {T} (0) \Psi_ {1} = \Psi_ {0}. \end{array}\tag{A.6}
$$

The explicit solutions of these Jordan chains can be arranged as Toeplitz and Hankel matrices

$$
\Theta = [ \Theta_ {0}, \Theta_ {1} ] = \left( \begin{array}{c c} c _ {0} & c _ {1} \\ 0 & c _ {0} \end{array} \right), \qquad \Psi = [ \Psi_ {0}, \Psi_ {1} ] = \left( \begin{array}{c c} 0 & d _ {0} ^ {*} \\ d _ {0} ^ {*} & d _ {1} ^ {*} \end{array} \right)\tag{A.7}
$$

and

$$
\tilde {\Psi} := \Psi S _ {2} = \left( \begin{array}{c c} d _ {0} ^ {*} & 0 \\ d _ {1} ^ {*} & d _ {0} ^ {*} \end{array} \right).\tag{A.8}
$$

From the simplest realization of the bi-orthonormality condition $\Psi^{+}\Theta = S_{2}, \tilde{\Psi}^{+}\Theta = I_{2}$ one finds the parameters $c_{1} = d_{1} = 0, d_{0}c_{0} = 1$ . Via similarity transformations $\Theta_{0,1} \mapsto \Phi_{0,1} = PR\Theta_{0,1}$ and $\Psi_{0,1} \mapsto \Xi_{0,1} = P(R^{-1})^{+}\Psi_{0,1}$ one arrives at the root vectors of M

$$
\begin{array}{l l} \Phi_ {0} = \sigma q c _ {0} \binom{1}{- Z _ {c}}, & \qquad \Phi_ {1} = \sigma q ^ {- 1} c _ {0} \binom{- Z _ {c}}{1} \\ \Xi_ {0} = \sigma q ^ {*} d _ {0} ^ {*} \binom{- Z _ {c}}{1}, & \qquad \Xi_ {1} = \sigma q ^ {* - 1} d _ {0} ^ {*} \binom{1}{- Z _ {c}} \\ \sigma := \frac {\mathrm{e} ^ {\mathrm{i} \mu \frac {\pi}{4}}}{\sqrt {2}}. \end{array}\tag{A.9}
$$

## References

[1] Kato T 1966 Perturbation Theory for Linear Operators (Berlin: Springer)

[2] Baumgärtel H 1984 Analytic Perturbation Theory for Matrices and Operators (Berlin: Akademie-Verlag)
Baumgärtel H 1985 Analytic Perturbation Theory for Matrices and Operators (Operator Theory Series, Advances and Applications vol 15) (Basel: Birkhäuser)

[4] Newton R G 1966 Scattering Theory of Waves and Particles (New York: Springer)

[3] Trotman D and Wilson L C (eds) 1997 Stratification, singularities and differential equations: II. Stratification and topology of singular spaces, Proceedings of the Meeting on Stratifications and Singularities (Luminy, France) (Paris: Hermann)

[5] Mahaux C and Weidenmüller H A 1969 Shell-Model Approach to Nuclear Reactions (Amsterdam: North-Holland)

Dembowski C, Dietz B, Gräf H-D, Harney H L, Heine A, Heiss W D and Richter A 2004 Phys. Rev. E 69 056216
Dietz B, Friedrich T, Metz J, Miski-Oglu M, Richter A, Schäfer F and Stafford C A 2007 Phys. Rev. E 75 027201

[6] Hernández E and Mondragón A 1994 Phys. Lett. B 326 1

[7] Mondragón A and Hernández E 1993 J. Phys. A: Math. Gen. 26 5595

[8] Bohm A, Loewe M, Maxson S, Patuleanu P, Püntmann C and Gadella M 1997 J. Math. Phys. 38 6072 (Preprint quant-ph/9705011)

[9] Mondragón A, Jáuregui A and Hernández E 2003 Phys. Rev. A 67 022721 (Preprint quant-ph/0204084)

[10] Rotter I 1991 Rep. Prog. Phys. 54 635
Okolowicz J, Ploszajczak M and Rotter I 2003 Phys. Rep. 374 271

[11] Magunov A I, Rotter I and Strakhova S I 1999 J. Phys. B: At. Mol. Opt. Phys. 32 1669

[12] Vanroose W 2001 Phys. Rev. A 64 062708

[13] Rotter I and Sadreev A F 2004 Phys. Rev. E 69 066201

[14] Rotter I and Sadreev A F 2005 Phys. Rev. E 71 036227

[15] Rotter I and Sadreev A F 2005 Phys. Rev. E 71 046204

[16] Friedrich H and Wintgen D 1985 Phys. Rev. A 31 3964
Friedrich H and Wintgen D 1985 Phys. Rev. A 32 3231

[17] Bulgakov E N, Rotter I and Sadreev A F 2007 Phys. Rev. A 75 067401

[19] Heiss W D, Müller M and Rotter I 1998 Phys. Rev. E 58 2894

[20] Jung C, Müller M and Rotter I 1999 Phys. Rev. E 60 114

[21] Nazmitdinov R G, Sim H S, Schomerus H and Rotter I 2002 Phys. Rev. B 66 241302(R) Bulgakov E N, Gopar V A, Mello P A and Rotter I 2006 Phys. Rev. B 73 155302

[22] Solov'ev E A 1981 Sov. Phys.—JETP 54 893
Solov'ev E A 2005 J. Phys. B: At. Mol. Opt. Phys. 38 R153
Briggs J S, Savichev V I and Solov'ev E A 2000 J. Phys. B: At. Mol. Opt. Phys. 33 3363

[23] Persson E, Rotter I, Stöckmann H J and Barth M 2000 Phys. Rev. Lett. 85 2478

[24] Dembowski C, Gräf H-D, Harney H L, Heine A, Heiss W D, Rehfeld H and Richter A 2001 Phys. Rev. Lett. 86 787

Dembowski C, Dietz B, Gräf H-D, Harney H L, Heine A, Heiss W D and Richter A 2003 Phys. Rev. Lett. 90 034101

[25] Heiss W D 1999 Eur. Phys. J. D 7 1
Heiss W D 2000 Phys. Rev. E 61 929

[26] Keck F, Korsch H J and Mossmann S 2003 J. Phys. A: Math. Gen. 36 2125–37

[28] Garrison J C and Wright E M 1988 Phys. Lett. A 128 177

[29] Berry M V 1990 ‘Quantum adiabatic anholonomy’ in Anomalies, Phases, Defects ed U M Bregola, G Marmo and G Morandi (Naples: Bibliopolis) pp 125–181

[30] Berry M V 1995 Ann. New York Acad. Sci. 755 303

[31] Mondragón A and Hernández E 1996 J. Phys. A: Math. Gen. 29 2567
Mondragón A and Hernández E 1998 Accidental degeneracy and Berry phase of resonant states Irreversibility and Causality: Semigraps and Rigged Hilbert Spaces eds A Bohm, H-D Doebner and P Kielanowski (Berlin: Springer)

[32] Mailybaev A A, Kirillov O N and Seyranian A P 2005 Phys. Rev. A 72 014104 (Preprint quant-ph/0501040)

[33] Bender C M and Boettcher S 1998 Phys. Rev. Lett. 80 5243 (Preprint physics/9712001)
Bender C M, Brody D C and Jones H F 2002 Phys. Rev. Lett. 89 270401 (Preprint quant-ph/0208076)

[34] Bender C M, Brody D C, Jones H F and Meister B K 2007 Phys. Rev. Lett. 98 040403 (Preprint quant-ph/0609032)

[35] Bender C M 2007 Making sense of non-Hermitian Hamiltonians Preprint hep-th/0703096

[36] Dorey P, Dunning C and Tateo R 2001 J. Phys. A: Math. Gen. 34 L391 (Preprint hep-th/0104119)

57] Mostafazadeh A 2002 J. Math. Phys. 43 6343
Mostafazadeh A 2003 J. Math. Phys. 44 943 (Preprint math-ph/0207009) (erratum)

[38] Günther U, Stefani F and Znojil M 2005 J. Math. Phys. 46 063504 (Preprint math-ph/0501069)

[39] Günther U and Stefani F 2005 Czech. J. Phys. 55 1099 (Preprint math-ph/0506021)

[40] Znojil M 2007 Phys. Lett. B 647 225 (Preprint quant-ph/0701232)

[41] Berry M V and Dennis M R 2003 Proc. R. Soc. A 459 1261
Berry M V 2004 Czech. J. Phys. 54 1040
Berry M V 2005 Proc. R. Soc. A 461 2071

[42] Shuvalov A L and Scott N H 2000 Acta Mech. 140 1

[44] Seyranian A P and Mailybaev A A 2003 Multiparameter Stability Theory with Mechanical Applications (Singapore: World Scientific)

[45] Kirillov O N and Seyranian A P 2004 SIAM J. Appl. Math. 64 1383
Mailybaev A A, Kirillov O N and Seyranian A P 2005 J. Phys. A: Math. Gen. 38 1723 (Preprint math-ph/0411024)

[46] Stefani F and Gerbeth G 2005 Phys. Rev. Lett. 94 184506 (Preprint physics/0411050)
Stefani F, Gerbeth G, Günther U and Xu M 2006 Earth Planet. Sci. Lett. 243 828–40 (Preprint physics/0509118)
Stefani F, Xu M, Sorriso-Valvo L, Gerbeth G and Günther U 2007 Reversals in nature and the nature of reversals Preprint physics/0701026

[47] Vishik I M and Lyusternik L A 1960 Russ. Math. Surv. 15 1

[48] Arnol'd V I, Vasil'ev V A, Goryunov V V and Lyashko O V 1993 Dynamical systems: VI Singularity Theory vol 1 (Berlin: Springer)

Arnol'd V I, Goryunov V V, Lyashko O V and Vasil'ev V A 1993 Dynamical systems: VIII Singularity Theory vol 2 (Berlin: Springer)

[49] Samsonov B F 2005 J. Phys. A: Math. Gen. 38 L397 (Preprint quant-ph/0503075)

[50] Andrianov A A, Cannata F and Sokolov A V 2006 Non-linear Supersymmetry for non-Hermitian, non-diagonalizable Hamiltonians: I. General properties Preprint math-ph/0610024

[51] Narevicius E, Serra P and Moiseyev N 2003 Europhys. Lett. 62 789

[52] Sokolov A V, Andrianov A A and Cannata F 2006 J. Phys. A: Math. Gen. 39 10207 (Preprint quant-ph/0602207)

[53] Nakahara M 1990 Geometry, Topology and Physics (Bristol: Institute of Physics Publishing)

[54] Gantmacher F R 1959 The Theory of Matrices vol 2 (New York: Chelsea)

[55] Reid C E and Brändas E 1989 On a theorem for complex symmetric matrices and its relevance in the study of decay phenomena Resonances ed E Brändas and N Elander (Berlin: Springer)

[56] Page D N 1987 Phys. Rev. A 36 3479

[57] Assis P E G and Fring A 2007 The quantum brachistochrone problem for non-Hermitian Hamiltonians (Preprint quant-ph/0703254)

[58] Kodaira K 1986 Complex Manifolds and Deformation of Complex Structures (New York: Springer)

[59] Mumford D 1976 Algebraic Geometry: I. Complex Projective Varieties (Berlin: Springer)

[60] Naber G L 1997 Topology, Geometry, and Gauge Fields: Foundations (New York: Springer)

[61] Kobayashi S and Nomizu K 1969 Foundations of Differential Geometry vol 2 (New York: Interscience)

[62] Ito K 1993 Encyclopedic Dictionary of Mathematics (Cambridge, MA: MIT Press)

[63] Pragacz P (ed) 2005 Topics in Cohomological Studies of Algebraic Varieties. Impanga Lecture Notes (Basel: Birkhäuser)

[64] Günther U et al Projective resolutions of exceptional point singularities (in preparation)

[65] Berry M V 1984 Proc. R. Soc. A 392 45

[66] Havlicek M, Posta S and Winternitz P 1999 J. Math. Phys. 40 3104

[67] Danilov V I 2002 Monodromy transformation Encyclopaedia of Mathematics (Berlin: Springer) http://eom.springer.de/

[68] Candelas P and de la Ossa X C 1990 Nucl. Phys. B 342 246

[69] Rotter I 2001 Phys. Rev. E 64 036213

[70] Bulgakov E N, Rotter I and Sadreev A F 2006 Phys. Rev. E 74 056204 (Preprint quant-ph/0605056)
Bulgakov E N, Rotter I and Sadreev A F Correlated behavior of conductance and phase rigidity in the transition from the weak-coupling to the strong-coupling regime (in preparation)

[71] van Langen S A, Brouwer P W and Beenakker C W J 1997 Phys. Rev. E 55 R1–R4 (Preprint cond-mat/9609100) Brouwer P W 2003 Phys. Rev. E 68 046205 (Preprint nlin.CD/0302052)

[72] Martin D 2007 Is PT-symmetric quantum mechanics just quantum mechanics in a non-orthogonal basis? Preprint quant-ph/0701223
