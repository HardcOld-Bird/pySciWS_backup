# A Short Course on Topological Insulators

Band Structure and Edge States in One and Two Dimensions

# Lecture Notes in Physics

Volume 919

Founding Editors

W. Beiglböck

J. Ehlers

K. Hepp

H. Weidenmüller

Editorial Board

M. Bartelmann, Heidelberg, Germany

B.-G. Englert, Singapore, Singapore

P. Hänggi, Augsburg, Germany

M. Hjorth-Jensen, Oslo, Norway

R.A.L. Jones, Sheffield, UK

M. Lewenstein, Barcelona, Spain

H. von Löhneysen, Karlsruhe, Germany

J.-M. Raimond, Paris, France

A. Rubio, Donostia, San Sebastian, Spain

M. Salmhofer, Heidelberg, Germany

S. Theisen, Potsdam, Germany

D. Vollhardt, Augsburg, Germany

J.D. Wells, Ann Arbor, USA

G.P. Zank, Huntsville, USA

## The Lecture Notes in Physics

The series Lecture Notes in Physics (LNP), founded in 1969, reports new developments in physics research and teaching-quickly and informally, but with a high quality and the explicit aim to summarize and communicate current knowledge in an accessible way. Books published in this series are conceived as bridging material between advanced graduate textbooks and the forefront of research and to serve three purposes:

\- to be a compact and modern up-to-date source of reference on a well-defined topic

\- to serve as an accessible introduction to the field to postgraduate students and nonspecialist researchers from related areas

\- to be a source of advanced teaching material for specialized seminars, courses and schools

Both monographs and multi-author volumes will be considered for publication. Edited volumes should, however, consist of a very limited number of contributions only. Proceedings will not be considered for LNP.

Volumes published in LNP are disseminated both in print and in electronic formats, the electronic archive being available at springerlink.com. The series content is indexed, abstracted and referenced by many abstracting and information services, bibliographic networks, subscription agencies, library networks, and consortia.

Proposals should be sent to a member of the Editorial Board, or directly to the managing editor at Springer:

Christian Caron

Springer Heidelberg

Physics Editorial Department I

Tiergartenstrasse 17

69121 Heidelberg/Germany

christian.caron@springer.com

# A Short Course on Topological Insulators

Band Structure and Edge States in One and Two Dimensions

János K. Asbóth
Wigner Research Centre for Physics
Hungarian Academy of Sciences
Budapest, Hungary

László Oroszlány
Department of Physics of Complex Systems
Eötvös Loránd University
Budapest, Hungary

András Pályi
Department of Materials Physics
Eötvös Loránd University
Budapest, Hungary

Department of Physics
Budapest University of Technology and Economics
Budapest, Hungary

ISSN 0075-8450
Lecture Notes in Physics
ISBN 978-3-319-25605-4
DOI 10.1007/978-3-319-25607-8

ISSN 1616-6361 (electronic)

ISBN 978-3-319-25607-8 (eBook)

Library of Congress Control Number: 2015960963

Springer Cham Heidelberg New York Dordrecht London
© Springer International Publishing Switzerland 2016

This work is subject to copyright. All rights are reserved by the Publisher, whether the whole or part of the material is concerned, specifically the rights of translation, reprinting, reuse of illustrations, recitation, broadcasting, reproduction on microfilms or in any other physical way, and transmission or information storage and retrieval, electronic adaptation, computer software, or by similar or dissimilar methodology now known or hereafter developed.

The use of general descriptive names, registered names, trademarks, service marks, etc. in this publication does not imply, even in the absence of a specific statement, that such names are exempt from the relevant protective laws and regulations and therefore free for general use.

The publisher, the authors and the editors are safe to assume that the advice and information in this book are believed to be true and accurate at the date of publication. Neither the publisher nor the authors or the editors give a warranty, express or implied, with respect to the material contained herein or for any errors or omissions that may have been made.

Cover designer: eStudio Calamar, Berlin/Figueres

Printed on acid-free paper

Springer International Publishing AG Switzerland is part of Springer Science+Business Media (www.springer.com)

These lecture notes provide an introduction to some of the main concepts of topological insulators, a branch of solid state physics that is developing at a fast pace. They are based on a one-semester course for MSc and PhD students at Eötvös University, Budapest, which the authors have been giving since 2012.

Our aim is to provide an understanding of the core topics of topological insulators—edge states, bulk topological invariants, bulk-boundary correspondence—with as simple mathematical tools as possible. We restricted our attention to one- and two-dimensional band insulators. We use noninteracting lattice models of topological insulators and build these up gradually to arrive from the simplest one-dimensional case (the Su–Schrieffer–Heeger model for polyacetylene) to two-dimensional time-reversal invariant topological insulators (the Bernevig–Hughes–Zhang model for HgTe). In each case we introduce the model first, discuss its properties, and then generalize. The prerequisite for the reader is quantum mechanics and not much else: solid state physics background is provided as we go along.

Since this is an introduction, rather than a broad overview, we try to be self-contained and give citations to the current literature only where it is absolutely necessary. For a broad overview, including pointers to the original papers and current topics, we refer the reader to review articles and books in the Introduction.

Supporting material for these lecture notes in form of Jupyter notebooks are available online at https://github.com/topologicalbudapest/topins.

Despite our efforts, the book inevitably contains typos, errors, and less comprehensible explanations. We would appreciate if you could inform us of any of those; please send your comments to janos.asboth@wigner.mta.hu.

Acknowledgments We are grateful for enlightening discussions on topological insulators with Anton Akhmerov, Andrea Alberti, Carlo Beenakker, and Alberto Cortijo. We thank the feedback we got from participants at the courses at Eötvös University, especially Vilmos Kocsis. A version of this course was given by one of us (J.K.A.) in the PhD program of the University of Geneva, on invitation by Markus Büttiker, which helped shape the course.

The preparation of the first version of these lecture notes was supported by the grant TÁMOP4.1.2.A/1-11/0064. We acknowledge financial support from the Hungarian Scientific Research Fund (OTKA), Grant Nos. PD100373, K108676, NN109651, and from the Marie Curie program of the European Union, Grant No. CIG-293834. J.K.A. and A.P. were supported by the János Bolyai Scholarship of the Hungarian Academy of Sciences.

Budapest, Hungary

September 2015

János K. Asbóth

László Oroszlány

András Pályi

The band theory of electric conduction was one of the early victories of quantum mechanics in the 1920s. It gave a simple explanation of how some crystalline materials are electric insulators, even though electrons in them can hop from one atom to the next. In the bulk of a band insulator, the electrons occupy eigenstates that form energy bands. In a band insulator, there are no partially filled bands: completely filled bands are separated by an energy gap from completely empty bands; the gap represents the energy cost of mobilizing electrons. In contrast, materials with partially filled bands are conductors, where there are plane wave states available to transmit electrons across the bulk at arbitrarily low energy. Although we now know of situations where band theory is inadequate (e.g., for Mott insulators), it remains one of the cornerstones of solid state physics.

The discovery of the quantum Hall effect (1980) has shown that the simple division into band insulators and metals is not the end of the story, not even in band theory. In the quantum Hall effect, a strong magnetic field confines the motion of electrons in the bulk, but the same field forces them into delocalized edge states on the surface. A two-dimensional metal in strong magnetic field is thus an insulator in the bulk, but conducts along the surface, via a discrete number of completely open edge state channels (in the language of the Landauer–Büttiker formalism). The number of edge state channels was linked to the Chern number, a topological invariant of the occupied bands.

Over the last 20 years, theoretical progress over artificial systems has shown that the external magnetic field is not necessary for an insulator to have robust conducting edge states: instead, the nontrivial topology of the occupied bands is the crucial ingredient. The name topological insulator was coined for such systems, and their study became a blossoming branch of solid state physics. Following the theoretical prediction (Bernevig, Hughes, and Zhang [5]), electronic transport measurements confirmed that a thin layer of HgTe is a topological insulator (König et al. [21]). Since that time, a host of materials have been shown to be three-dimensional topological insulators and thin films and quantum wires shown to be two- and one-dimensional topological insulators [2].

The intense theoretical interest in topological insulators has led to signature results, such as the “the periodic table of topological insulators” [29], which shows that similarly to phase transitions in statistical mechanics, it is the dimensionality and the basic symmetries of an insulator that decide whether it can be a topological insulator or not. Although it was derived by different ways of connecting topological insulators of various dimensions and symmetries (the so-called dimensional reduction schemes), the mathematically rigorous proof of the periodic table is still missing.

The field of topological insulators is very active, with many experimental challenges and open theoretical problems, regarding the effects of electron–electron interaction, extra crystalline symmetries, coupling to the environment, etc.

## Literature

To get a quick and broad overview of topological insulators, with citations for relevant research papers, we recommend the review papers $[7, 17, 25]$ . For a more in-depth look, there are already a few textbooks on the subject (by Bernevig and Hughes $[4]$ and by Shen $[30]$ and one edited by Franz and Molenkamp $[10]$ ). To see the link between momentum-space topology and physics in a broader context, we direct the reader to a book by Volovik $[34]$ .

There are also introductory courses on topological insulators with a broad scope. We recommend the lectures by Charles Kane (the video recording of the version given at Veldhoven is freely available online) and the online EdX course on topology in condensed matter by a group of lecturers, with the corresponding material collected at topocondmat.org.

## These Lecture Notes

Our aim with this set of lecture notes is to complement the literature cited above: we wish to provide a close look at some of the core concepts of topological insulators with as simple mathematical tools as possible. Using one- and two-dimensional noninteracting lattice models, we explain what edge states and what bulk topological invariants are, how the two are linked (this is known as the bulk-boundary correspondence), and the meaning and impact of some of the fundamental symmetries.

To keep things as simple as possible, throughout the course we use noninteracting models for solid state systems. These are described using single-particle lattice Hamiltonians, with the zero of the energy corresponding to the Fermi energy. We use natural units, with $\hbar = 1$ and length measured by the lattice constant.

## Contents

The Su-Schrieffer-Heeger (SSH) Model 1
1.1 The SSH Hamiltonian 1
1.1.1 External and Internal Degrees of Freedom 2
1.2 Bulk Hamiltonian 3
1.2.1 Bulk Momentum-Space Hamiltonian 4
1.2.2 Periodicity in Wavenumber 4
1.2.3 The Hopping Is Staggered to Open a Gap 5
1.2.4 Information Beyond the Dispersion Relation 6
1.3 Edge States 7
1.3.1 Fully Dimerized Limits 7
1.3.2 Moving Away from the Fully Dimerized Limit 8
1.4 Chiral Symmetry 10
1.4.1 No Unitary Symmetries 10
1.4.2 A Different Type of Symmetry 10
1.4.3 Consequences of Chiral Symmetry for Energy Eigenstates 11
1.4.4 Sublattice Projectors and Chiral Symmetry of the SSH Model 13
1.4.5 Consequence of Chiral Symmetry: Bulk Winding Number for the SSH Model 13
1.5 Number of Edge States as Topological Invariant 16
1.5.1 Adiabatic Equivalence of Hamiltonians 16
1.5.2 Topological Invariant 17
1.5.3 Number of Edge States as a Topological Invariant 18
1.5.4 Bulk-Boundary Correspondence in the SSH Model 18
1.5.5 Bound States at Domain Walls 19
1.5.6 Exact Calculation of Edge States 20
Problems 22

2 Berry Phase, Chern Number 23
2.1 Discrete Case 23
2.1.1 Relative Phase of Two Nonorthogonal Quantum States 24
2.1.2 Berry Phase 24
2.1.3 Berry Flux 25
2.1.4 Chern Number 27
2.2 Continuum Case 28
2.2.1 Berry Connection 28
2.2.2 Berry Phase 29
2.2.3 Berry Curvature 29
2.2.4 Chern Number 33
2.3 Berry Phase and Adiabatic Dynamics 34
2.4 Berry's Formulas for the Berry Curvature 37
2.5 Example: The Two-Level System 38
2.5.1 No Continuous Global Gauge 38
2.5.2 Calculating the Berry Curvature and the Berry Phase 40
2.5.3 Two-Band Lattice Models and Their Chern Numbers 42
Problems 43
3 Polarization and Berry Phase 45
3.1 The Rice-Mele Model 46
3.2 Wannier States in the Rice-Mele Model 46
3.2.1 Defining Properties of Wannier States 47
3.2.2 Wannier States Are Inverse Fourier Transforms of the Bloch Eigenstates 48
3.2.3 Wannier Centers Can Be Identified with the Berry Phase 49
3.2.4 Wannier States Using the Projected Position Operator 49
3.3 Inversion Symmetry and Polarization 51
3.3.1 Quantization of the Wilson Loop Due to Inversion Symmetry 52
Problems 53
4 Adiabatic Charge Pumping, Rice-Mele Model 55
4.1 Charge Pumping in a Control Freak Way 56
4.1.1 Adiabatic Shifting of Charge on a Dimer 56
4.1.2 Putting Together the Control Freak Sequence 56
4.1.3 Visualizing the Motion of Energy Eigenstates 57
4.1.4 Edge States in the Instantaneous Spectrum 59
4.2 Moving Away from the Control Freak Limit 60
4.2.1 Edge States in the Instantaneous Spectrum 60
4.2.2 The Net Number of Edge States Pumped in Energy Is a Topological Invariant 62

4.3 Tracking the Charges with Wannier States 64
4.3.1 Plot the Wannier Centers 64
4.3.2 Number of Pumped Particles Is the Chern Number 65
4.3.3 Tuning the Pump Using the Average Intracell Hopping Amplitude $\overline{v}$ 67
4.3.4 Robustness Against Disorder 67
Problems 68
5 Current Operator and Particle Pumping 69
5.1 Particle Current at a Cross Section of the Lattice 70
5.1.1 Current Operator in the Rice-Mele Model 71
5.1.2 Current Operator in a Generic One-Dimensional Lattice Model 73
5.1.3 Number of Pumped Particles 75
5.2 Time Evolution Governed by a Quasi-Adiabatic Hamiltonian 76
5.2.1 The Parallel-Transport Time Parametrization 76
5.2.2 Quasi-Adiabatic Evolution 77
5.3 The Pumped Current Is the Berry Curvature 81
Problems 83
6 Two-Dimensional Chern Insulators: The Qi-Wu-Zhang Model 85
6.1 Dimensional Extension: From an Adiabatic Pump to a Chern Insulator 86
6.1.1 From the Rice-Mele Model to the Qi-Wu-Zhang Model 86
6.1.2 Bulk Dispersion Relation 87
6.1.3 Chern Number of the QWZ Model 87
6.1.4 The Real-Space Hamiltonian 90
6.2 Edge States 91
6.2.1 Dispersion Relation of a Strip Shows the Edge States 91
6.2.2 Edge States Conduct Unidirectionally 92
6.2.3 Edge States and Edge Perturbation 93
6.2.4 Higher Chern Numbers by Coupling Layers 94
6.3 Robustness of Edge States 96
6.3.1 Smoothly Removing Disorder 96
6.3.2 Unitarity: Particles Crossing the Clean Part Have to Go Somewhere 97
Problems 98
7 Continuum Model of Localized States at a Domain Wall 99
7.1 One-Dimensional Monatomic Chain in an Electric Potential 99
7.1.1 The Model 100
7.1.2 Envelope-Function Approximation 101
7.1.3 Envelope-Function Approximation: The Proof 103

7.2 The SSH Model and the One-Dimensional Dirac Equation 104
7.2.1 The Metallic Case 105
7.2.2 The Nearly Metallic Case 106
7.2.3 Continuum Description of the Nearly Metallic Case 107
7.2.4 Localized States at a Domain Wall 108
7.3 The QWZ Model and the Two-Dimensional Dirac Equation 111
7.3.1 The Metallic Case 111
7.3.2 The Nearly Metallic Case 112
7.3.3 Continuum Description of the Nearly Metallic Case 112
7.3.4 Chiral States at a Domain Wall 113
Problems 117
Time-Reversal Symmetric Two-Dimensional Topological Insulators: The Bernevig–Hughes–Zhang Model 119
8.1 Time-Reversal Symmetry 120
8.1.1 Time Reversal in Continuous Variable Quantum Mechanics (Without Spin) 120
8.1.2 Lattice Models with an Internal Degree of Freedom 121
8.1.3 Two Types of Time-Reversal 122
8.1.4 Time Reversal of Type $\hat{\mathcal{T}}^{2} = -1$ Gives Kramers' Degeneracy 124
8.1.5 Time-Reversal Symmetry of a Bulk Hamiltonian 124
8.2 Doubling the Hilbert Space for Time-Reversal Symmetry 125
8.2.1 Time Reversal with $\hat{\mathcal{T}}^{2} = -1$ Requires Antisymmetric Coupling Operator $\hat{C}$ 126
8.2.2 Symmetric Coupling Operator $\hat{C}$ Gives Time Reversal with $\hat{\mathcal{T}}^{2} = +1$ 127
8.3 A Concrete Example: The Bernevig-Hughes-Zhang Model 127
8.3.1 Two Time-Reversal Symmetries If There Is No Coupling 128
8.4 Edge States in Two-Dimensional Time-Reversal Invariant Insulators 128
8.4.1 An Example: The BHZ Model with Different Types of Coupling 128
8.4.2 Edge States in $\hat{\mathcal{T}}^{2} = -1$ 129
8.4.3 $\mathbb{Z}_{2}$ Invariant: Parity of Edge State Pairs 131
8.5 Absence of Backscattering 132
8.5.1 The Scattering Matrix 132
8.5.2 A Single Kramers Pair of Edge States 134
8.5.3 An Odd Number of Kramers Pairs of Edge States 137
8.5.4 Robustness Against Disorder 137

9 The $\mathbb{Z}_2$ Invariant of Two-Dimensional Topological Insulators 139
9.1 Tools: Nonabelian Berry Phase, Multiband Wannier States 140
9.1.1 Preparation: Nonabelian Berry Phase 140
9.1.2 Wannier States for Degenerate Multiband One-Dimensional Insulators 142
9.2 Time-Reversal Restrictions on Wannier Centers 145
9.2.1 Eigenstates at $\mathbf{k}$ and $-\mathbf{k}$ Are Related 145
9.2.2 Wilson Loops at $k_y$ and $-k_y$ Have the Same Eigenvalues 146
9.2.3 Wilson Loop Eigenvalues at $k_y = 0$ and $k_y = \pi$ Are Doubly Degenerate 147
9.3 Two Types of Wannier Center Flow 148
9.3.1 Bulk Topological Invariant 149
9.4 The $\mathbb{Z}_2$ Invariant for Systems with Inversion Symmetry 150
9.4.1 Definition of Inversion Symmetry 150
9.4.2 At a Time-Reversal Invariant Momentum, the Kramers Pairs Have the Same Inversion Eigenvalue 150
9.4.3 Example: The BHZ Model 151
Problems 152
10 Electrical Conduction of Edge States 153
10.1 Electrical Conduction in a Clean Quantum Wire 154
10.2 Phase-Coherent Electrical Conduction in the Presence of Scatterers 156
10.3 Electrical Conduction in Two-Dimensional Topological Insulators 157
10.3.1 Chern Insulators 158
10.3.2 Two-Dimensional Time-Reversal Invariant Topological Insulators with $\hat{\mathcal{I}}^2 = -1$ 160
10.4 An Experiment with HgTe Quantum Wells 160
References 165

# Chapter 1 The Su-Schrieffer-Heeger (SSH) Model

We take a hands-on approach and get to know the basic concepts of topological insulators via a concrete system: the Su-Schrieffer-Heeger (SSH) model describes spinless fermions hopping on a one-dimensional lattice with staggered hopping amplitudes. Using the SSH model, we introduce the concepts of the single-particle Hamiltonian, the difference between bulk and boundary, chiral symmetry, adiabatic equivalence, topological invariants, and bulk-boundary correspondence.

## 1.1 The SSH Hamiltonian

The Su-Schrieffer-Heeger (SSH) model describes electrons hopping on a chain (one-dimensional lattice), with staggered hopping amplitudes, as shown in Fig. 1.1. The chain consist of N unit cells, each unit cell hosting two sites, one on sublattice A, and one on sublattice B. Interactions between the electrons are neglected, and so the dynamics of each electron is described by a single-particle Hamiltonian, of the form

$$
\hat {H} = v \sum_ {m = 1} ^ {N} \big (| m, B \rangle \langle m, A | + h. c. \big) + w \sum_ {m = 1} ^ {N - 1} \big (| m + 1, A \rangle \langle m, B | + h. c. \big).\tag{1.1}
$$

Here $|m, A\rangle$ and $|m, B\rangle$ , with $m \in \{1, 2, \ldots, N\}$ , denote the state of the chain where the electron is on unit cell $m$ , in the site on sublattice $A$ , respectively, $B$ , and $h.c.$ stands for Hermitian Conjugate (e.g., $h.c.$ of $\alpha |m, B\rangle \langle m', A|$ is $\alpha^* |m', A\rangle \langle m, B|$ for any $\alpha \in \mathbb{C}$ ).

The spin degree of freedom is completely absent from the SSH model, since no term in the Hamiltonian acts on spin. Thus, the SSH model describes spin-polarized electrons, and when applying the model to a real physical system, e.g., polyacetylene, we have to always take two copies of it. In this chapter we will just consider a single copy, and call the particles fermions, or electrons, or just particles.

![](images/0c6aeca312d786a813854723208505e6bca129dfef6edc57a4118a9ca76a0ec1.jpg)
Fig. 1.1 Geometry of the SSH model. Filled (empty) circles are sites on sublattice A (B), each hosting a single state. They are grouped into unit cells: the n = 6th cell is circled by a dotted line. Hopping amplitudes are staggered: intracell hopping v (thin lines) is different from intercell hopping w (thick lines). The left and right edge regions are indicated by blue and red shaded background

We are interested in the dynamics of fermions in and around the ground state of the SSH model at zero temperature and zero chemical potential, where all negative energy eigenstates of the Hamiltonian are singly occupied (because of the Pauli principle). As we will show later, due to the absence of onsite potential terms, there are N such occupied states. This situation—called half filling—is characteristic of the simplest insulators such as polyacetylene, where each carbon atom brings one conduction electron, and so we find one particle (of each spin type) per unit cell.

For simplicity, we take the hopping amplitudes to be real and nonnegative, $v, w \geq 0$ . If this was not the case, if they carried phases, $v = |v| e^{i\phi_v}$ , and $w = |w| e^{i\phi_w}$ , with $\phi_v, \phi_w \in [0, 2\pi)$ these phases could always be gauged away. This is done by a redefinition of the basis states: $|m, A\rangle \to e^{-i(m-1)(\phi_v + \phi_w)}$ , and $|m, B\rangle \to e^{-i\phi_v} e^{-i(m-1)(\phi_v + \phi_w)}$ .

The matrix for the Hamiltonian of the SSH model, Eq. $(1.1)$ , on a real-space basis, for a chain of N = 4 unit cells, reads

$$
H = \left( \begin{array}{c c c c c c c c} 0 & v & 0 & 0 & 0 & 0 & 0 & 0 \\ v & 0 & w & 0 & 0 & 0 & 0 & 0 \\ 0 & w & 0 & v & 0 & 0 & 0 & 0 \\ 0 & 0 & v & 0 & w & 0 & 0 & 0 \\ 0 & 0 & 0 & w & 0 & v & 0 & 0 \\ 0 & 0 & 0 & 0 & v & 0 & w & 0 \\ 0 & 0 & 0 & 0 & 0 & w & 0 & v \\ 0 & 0 & 0 & 0 & 0 & 0 & v & 0 \end{array} \right)\tag{1.2}
$$

## 1.1.1 External and Internal Degrees of Freedom

There is a practical representation of this Hamiltonian, which emphasizes the separation of the external degrees of freedom (unit cell index m) from the internal degrees of freedom (sublattice index $\alpha$ ). We can use a tensor product basis,

$$
\left| \right. m, \alpha \left. \right\rangle\rightarrow \left| \right. m \left. \right\rangle \otimes \left| \right. \alpha \left. \right\rangle \in \mathscr {H} _ {\text { external }} \otimes \mathscr {H} _ {\text { internal }},\tag{1.3}
$$

## 1.2 Bulk Hamiltonian

with $m = 1, \ldots, N$ , and $\alpha \in \{A, B\}$ . In this basis, with the Pauli matrices,

$$
\sigma_ {0} = \left( \begin{array}{c c} 1 & 0 \\ 0 & 1 \end{array} \right); \quad \sigma_ {x} = \left( \begin{array}{c c} 0 & 1 \\ 1 & 0 \end{array} \right); \quad \sigma_ {y} = \left( \begin{array}{c c} 0 & - i \\ i & 0 \end{array} \right); \quad \sigma_ {z} = \left( \begin{array}{c c} 1 & 0 \\ 0 & - 1 \end{array} \right),\tag{1.4}
$$

the Hamiltonian can be written as

$$
\hat {H} = v \sum_ {m = 1} ^ {N} | m \rangle \langle m | \otimes \hat {\sigma} _ {x} + w \sum_ {m = 1} ^ {N - 1} \left(| m + 1 \rangle \langle m | \otimes \frac {\hat {\sigma} _ {x} + i \hat {\sigma} _ {y}}{2} + h. c.\right).\tag{1.5}
$$

The intracell hopping shows up here as an intracell operator, while the intercell hopping as a hopping operator.

## 1.2 Bulk Hamiltonian

As every solid-state system, the long chain of the SSH model has a bulk and a boundary. The bulk is the long central part of the chain, the boundaries are the two ends, or “edges” of the chain, indicated by shading in Fig. 1.1. We first concentrate on the bulk, since, in the thermodynamic limit of $N \rightarrow \infty$ , it is much larger than the boundaries, and it will determine the most important physical properties of the model. Although the treatment of the bulk using the Fourier transformation might seem like a routine step, we detail it here because different conventions are used in the literature.

The physics in the bulk, the long central part of the system, should not depend on how the edges are defined, and so for simplicity we set periodic (Born-von Karman) boundary conditions. This corresponds to closing the bulk part of the chain into a ring, with the bulk Hamiltonian defined as

$$
\hat {H} _ {\text { bulk }} = \sum_ {m = 1} ^ {N} (v | m, B \rangle \langle m, A | + w | (m \bmod N) + 1, A \rangle \langle m, B |) + h. c..\tag{1.6}
$$

We are looking for eigenstates of this Hamiltonian,

$$
\hat {H} _ {\mathrm{bulk}} \left| \Psi_ {n} (k) \right\rangle = E _ {n} (k) \left| \Psi_ {n} (k) \right\rangle ,\tag{1.7}
$$

with $n\in \{1,\dots ,2N\}$

## 1.2.1 Bulk Momentum-Space Hamiltonian

Due to the translation invariance of the bulk, Bloch's theorem applies, and we look for the eigenstates in a plane wave form. We introduce the plane wave basis states only for the external degree of freedom,

$$
| k \rangle = \frac {1}{\sqrt {N}} \sum_ {m = 1} ^ {N} e ^ {i m k} | m \rangle , \quad \text {   for   } k \in \{\delta_ {k}, 2 \delta_ {k}, \dots , N \delta_ {k} \} \quad \text {   with   } \delta_ {k} = \frac {2 \pi}{N},\tag{1.8}
$$

where the wavenumber $k$ was chosen to take on values from the first Brillouin zone. The Bloch eigenstates read

$$
| \Psi_ {n} (k) \rangle = | k \rangle \otimes | u _ {n} (k) \rangle ; \qquad | u _ {n} (k) \rangle = a _ {n} (k) | A \rangle + b _ {n} (k) | B \rangle .\tag{1.9}
$$

The vectors $|u_n(k)\rangle \in \mathcal{H}_{\mathrm{internal}}$ are eigenstates of the bulk momentum-space Hamiltonian $\hat{H}(k)$ defined as

$$
\hat {H} (k) = \langle k | \hat {H} _ {\mathrm{bulk}} | k \rangle = \sum_ {\alpha , \beta \in \{A, B \}} \langle k, \alpha | H _ {\mathrm{bulk}} | k, \beta \rangle \cdot | \alpha \rangle \langle \beta |;\tag{1.10}
$$

$$
\hat {H} (k) \left| u _ {n} (k) \right\rangle = E _ {n} (k) \left| u _ {n} (k) \right\rangle .\tag{1.11}
$$

## 1.2.2 Periodicity in Wavenumber

Although Eq. (1.9) has a lot to do with the continuous-variable Bloch theorem, $\Psi_{n,k}(x) = e^{ikx} u_{n,k}(x)$ , this correspondence is not direct. In a discretization of the continuous-variable Bloch theorem, the internal degree of freedom would play the role of the coordinate within the unit cell, which is also transformed by the Fourier transform. Thus, the function $u_{n,k}(x)$ is periodic in real space, $u_{n,k}(x+1) = u_{n,k}(x)$ , but not periodic in the Brillouin zone, $u_{n,k+2\pi}(x+1) \neq u_{n,k}(x)$ . Our Fourier transform acts only on the external degree of freedom, and as a result, we have periodicity in the Brillouin zone,

$$
\hat {H} (k + 2 \pi) = \hat {H} (k); \qquad | u _ {n} (k + 2 \pi) \rangle = | u _ {n} (k) \rangle .\tag{1.12}
$$

This convention simplifies the formulas for the topological invariants immensely. Note, however, that the other convention, the discretization of the Bloch theorem, is also widely used in the literature.

(d)

## 1.2 Bulk Hamiltonian

As an example, take the SSH model on a chain of N = 4 unit cells. Then, by inserting Eq. (1.9) into the Schrödinger equation (1.7), the latter translates to the following matrix eigenvalue equation:

$$
\left( \begin{array}{c c c c c c c c} 0 & v & 0 & 0 & 0 & 0 & 0 & w \\ v & 0 & w & 0 & 0 & 0 & 0 & 0 \\ 0 & w & 0 & v & 0 & 0 & 0 & 0 \\ 0 & 0 & v & 0 & w & 0 & 0 & 0 \\ 0 & 0 & 0 & w & 0 & v & 0 & 0 \\ 0 & 0 & 0 & 0 & v & 0 & w & 0 \\ 0 & 0 & 0 & 0 & w & 0 & v \\ w & 0 & 0 & 0 & 0 & v & 0 \end{array} \right)
$$

$$
\left( \begin{array}{c} a (k) e ^ {i k} \\ b (k) e ^ {i k} \\ a (k) e ^ {2 i k} \\ b (k) e ^ {2 i k} \\ a (k) e ^ {3 i k} \\ b (k) e ^ {3 i k} \\ a (k) e ^ {N i k} \\ b (k) e ^ {N i k} \end{array} \right) = E (k) \left( \begin{array}{c} a (k) e ^ {i k} \\ b (k) e ^ {i k} \\ a (k) e ^ {2 i k} \\ b (k) e ^ {2 i k} \\ a (k) e ^ {3 i k} \\ b (k) e ^ {3 i k} \\ a (k) e ^ {N i k} \\ b (k) e ^ {N I k} \end{array} \right)\tag{1.13}
$$

The Schrödinger equation defining the matrix $H(k)$ of the bulk momentum-space Hamiltonian reads

$$
H (k) = \left( \begin{array}{c c} 0 & v + w e ^ {- i k} \\ v + w e ^ {i k} & 0 \end{array} \right); \quad H (k) \binom{a (k)}{b (k)} = E (k) \binom{a (k)}{b (k)}.\tag{1.14}
$$

## 1.2.3 The Hopping Is Staggered to Open a Gap

The dispersion relation of the bulk can be read off from Eq. (1.14), using the fact that $\hat{H}(k)^2 = E(k)^2\hat{\mathbb{I}}_2$ . This gives us

$$
E (k) = \pm \left| v + e ^ {- i k} w \right| = \pm \sqrt {v ^ {2} + w ^ {2} + 2 v w \cos k}\tag{1.15}
$$

We show this dispersion relation for five choices of the parameters in Fig. 1.2.

![](images/6f194b6376ebf987d3224e9cd31e16adabba5eaa500883b24499aaf6a98420c7.jpg)
(f)

![](images/4da54a0b4861d8b2f049b37512aeab1c6e5f139cb8c5d66a9fc74a7e8fbaa749.jpg)

![](images/b874b9d5a4039a6bb610a385a2cf8660b57f0f942ef62fd30d0d5dbdd6859a0e.jpg)

(g)
![](images/ce3ec1f0e7b2f22764552d8a252533e2995a5e13eee8c5ed04325d7ca43070af.jpg)

![](images/7a3c739742ec72f4e6f7f74277836683dafce0d1aeaed801922216fa9d29e583.jpg)

![](images/21118aa364dd55bc28bba22e4c08de8525b6d6cfe51b1e034b2d27e8103e4d36.jpg)

![](images/38c616a60bb07be70daa5dd786fc3e2030f58ae27a7cb0d072225eaa2fdd5617.jpg)

(h)
![](images/63d4a794678bc593fe1134a779d69684eda0d573f0c248d85ee55a8d4df92fdf.jpg)
(i)

(j)
![](images/39bf8a50c71f9dab6b516561bc281a01e579a7414fe43bda792deeb4d3054219.jpg)

![](images/520054d6f97aefbded638d32ad9586b2929b95fa3876c0252c0bbe439cb6e393.jpg)
Fig. 1.2 Dispersion relations of the SSH model, Eq. (1.15), for five settings of the hopping amplitudes: (a): v = 1, w = 0; (b): v = 1, w = 0.6; (c): v = w = 1; (d): v = 0.6, w = 1; (e): v = 0, w = 1. In each case, the path of the endpoints of the vector $\mathbf{d}(k)$ representing the bulk momentum-space Hamiltonian, Eqs. (1.17) and (1.18), are also shown on the $d_{x}$ , $d_{y}$ plane, as the wavenumber is swept across the Brillouin zone, $k = 0 \rightarrow 2\pi$

As long as the hopping amplitudes are staggered, $v \neq w$ , (Fig. 1.2a,b,d,e), there is an energy gap of $2\Delta$ separating the lower, filled band, from the upper, empty band, with

$$
\varDelta = \min _ {k} E (k) = | v - w |.\tag{1.16}
$$

Without the staggering, i.e., if v = w, (Fig. 1.2c), the SSH model describes a conductor. In that case there are plane wave eigenstates of the bulk available with arbitrarily small energy, which can transport electrons from one end of the chain to the other.

The staggering of the hopping amplitudes occurs naturally in many solid state systems, e.g., polyacetylene, by what is known as the Peierls instability. A detailed analysis of this process necessitates a model where the positions of the atoms are also dynamical $[32]$ . Nevertheless, we can understand this process intuitively just from the effects of a slight staggering on the dispersion relation. As the gap due to the staggering of the hopping amplitudes opens, the energy of occupied states is lowered, while unoccupied states move to higher energies. Thus, the staggering is energetically favourable.

## 1.2.4 Information Beyond the Dispersion Relation

Although the dispersion relation is useful to read off a number of physical properties of the bulk of the system (e.g., group velocities), there is also important information about the bulk that it does not reveal. Stationary states do not only have an energy and wavenumber eigenvalue, but also an internal structure, represented by the components of the corresponding vector $|u_{n}(k)\rangle \in \mathcal{H}_{\text{internal}}$ . We now define a compact representation of this information for the SSH model.

The bulk momentum-space Hamiltonian $\dot{H}(k)$ of any two-band model (i.e., a model with 2 internal states per unit cell), reads

$$
H (k) = d _ {0} (k) \hat {\sigma} _ {0} + d _ {x} (k) \hat {\sigma} _ {x} + d _ {y} (k) \hat {\sigma} _ {y} + d _ {z} (k) \hat {\sigma} _ {z} = d _ {0} (k) \hat {\sigma} _ {0} + \mathbf {d} (k) \hat {\boldsymbol {\sigma}}.\tag{1.17}
$$

For the SSH model, $d_{0}(k)=0$ , and the real numbers $d_{x,y,z}\in R$ , the components of the k-dependent 3-dimensional vector $\mathbf{d}(k)$ , read

$$
d _ {x} (k) = v + w \cos k; \quad d _ {y} (k) = w \sin k; \quad d _ {z} (k) = 0.\tag{1.18}
$$

The internal structure of the eigenstates with momentum k is given by the direction in which the vector $\mathbf{d}(k)$ of Eq. (1.18) points (the energy is given by the magnitude of $\mathbf{d}(k)$ ; for details see Sect. 2.5).

As the wavenumber runs through the Brillouin zone, $k = 0 \rightarrow 2\pi$ , the path that the endpoint of the vector $\mathbf{d}(k)$ traces out is a closed circle of radius w on the $d_{x}, d_{y}$ plane, centered at $(v, 0)$ . For more general 2-band insulators, this path need not be a circle, but it needs to be a closed loop due to the periodicity of the bulk momentum-space Hamiltonian, Eq. (1.12), and it needs to avoid the origin, to describe an insulator. The topology of this loop can be characterized by an integer, the bulk winding number $\nu$ . This counts the number of times the loop winds around the origin of the $d_{x}, d_{y}$ plane. For example, in Fig. 1.2f,g, we have $\nu = 0$ , in Fig. 1.2i,j, we have $\nu = 1$ , while in Fig. 1.2h, the winding number $\nu$ is undefined.

## 1.3 Edge States

Like any material, the SSH Hamiltonian does not only have a bulk part, but also boundaries (which we refer to as ends or edges). The distinction between bulk and edge is not sharply defined, it describes the behaviour of energy eigenstates in the thermodynamic limit. In the case we consider in these lecture notes, the bulk is translation invariant, and then the we can distinguish edge states and bulk states by their localized/delocalized behaviour in the thermodynamic limit. We will begin with the fully dimerized limits, where the edge regions can be unambiguously defined. We then move away from these limits, and use a practical definition of edge states.

## 1.3.1 Fully Dimerized Limits

The SSH model becomes particularly simple in the two fully dimerized cases: if the intercell hopping amplitude vanishes and the intracell hopping is set to 1, v = 1, w = 0, or vice versa, v = 0, w = 1. In both cases the SSH chain falls apart to a sequence of disconnected dimers, as shown in Fig. 1.3.

![](images/ee41a7b1b4c18d4754974afe1eb0d21abb791adb8b8c3ea1d8d3d18c9e0ea52f.jpg)
Fig. 1.3 Fully dimerized limits of the SSH model, where the chain has fallen apart to disconnected dimers. In the trivial case (top, only intracell hopping, v = 1, w = 0), every energy eigenstate is an even or an odd superposition of two sites at the same unit cell. In the topological case, (bottom, only intercell hopping, v = 0, w = 1), dimers are between neighboring unit cells, and there is 1 isolated site per edge, that must contain one zero-energy eigenstate each, as there are no onsite potentials

## 1.3.1.1 The Bulk in the Fully Dimerized Limits Has Flat Bands

In the fully dimerized limits, one can choose a set of energy eigenstates which are restricted to one dimer each. These consist of the even (energy $E = +1$ ) and odd (energy $E = -1$ ) superpositions of the two sites forming a dimer.

In the v = 1, w = 0 case, which we call trivial, we have

$$
v = 1, w = 0: \qquad \hat {H} (| m, A \rangle \pm | m, B \rangle) = \pm (| m, A \rangle \pm | m, B \rangle).\tag{1.19}
$$

The bulk momentum-space Hamiltonian is $\hat{H}(k) = \hat{\sigma}_x$ , independent of the wavenumber $k$ .

In the $v = 0, w = 1$ case, which we call topological, each dimer is shared between two neighboring unit cells,

$$
v = 0, w = 1: \hat {H} (| m, B \rangle \pm | m + 1, A \rangle) = \pm (| m, B \rangle \pm | m + 1, A \rangle),\tag{1.20}
$$

for $m = 1, \ldots, N - 1$ . The bulk momentum-space Hamiltonian now is $\hat{H}(k) = \hat{\sigma}_x \cos k + \hat{\sigma}_y \sin k$ .

In both fully dimerized limits, the energy eigenvalues are independent of the wavenumber, $E(k) = \pm1$ . In this so-called flat-band limit, the group velocity is zero, which again shows that as the chain falls apart to dimers, a particle input into the bulk will not move along the chain.

## 1.3.1.2 The Edges in the Fully Dimerized Limit Can Host Zero Energy States

In the trivial case, v = 1, w = 0, all energy eigenstates of the SSH chain are given by the formulas of the bulk, Eq. (1.19). A topological, fully dimerized SSH chain, with v = 0, w = 1, however, has more energy eigenstates than those listed Eq. (1.20). Each end of the chain hosts a single eigenstate at zero energy,

$$
v = 0, w = 1: \quad \hat {H} | 1, A \rangle = \hat {H} | N, B \rangle = 0.\tag{1.21}
$$

These eigenstates have support on one site only. Their energy is zero because onsite potentials are not allowed in the SSH model. These are the simplest examples of edge states.

## 1.3.2 Moving Away from the Fully Dimerized Limit

We now examine what happens to the edge states as we move away from the fully dimerized limit. To be specific, we examine how the spectrum of an open topological chain, v = 0, w = 1, of N = 10 unit cells changes, as we continuously turn on the intracell hopping amplitude v. The spectra, Fig. 1.4, reveal that the energies of the edge states remain very close to zero energy.

![](images/67d39c3ef6a956f4ac18b6d53d9d30ffb14a0c2f0843b6ba94239d91b9af48d4.jpg)
Fig. 1.4 Energy spectrum and wave functions of a finite-sized SSH model. The number of unit cells is N = 10. (a) Energy spectrum of the system for intercell hopping amplitude w = 1 as a function the intracell hopping amplitude v. v < 1 (v > 1) corresponds to the topological (trivial) phases. (b) and (c) shows the wave functions of the hybridized edge states, while (d) shows a generic bulk wave function

The wavefunctions of almost-zero-energy edge states have to be exponentially localized at the left/right edge, because the zero of energy is in the bulk band gap. A plot of the wavefunctions (which have only real components, since the Hamiltonian is real), Fig. 1.4, reveals that the almost-zero-energy eigenstates are odd and even superpositions of states localized exponentially on the left and right edge. This is a result of the exponentially small overlap between the left and the right edge states. We will later show, in Sect. 1.5.6, that the edge-state energies are also controlled by this overlap, and are of the order $E = e^{-N/\xi}$ , with a localization length $\xi = 1/\log(v/w)$ .

There is an important property of the left (right) edge states, which is only revealed by the plot of the wavefunctions, Fig. 1.4. The right edge state has nonvanishing components only on the A sublattice while the left edge state on the B sublattice.

In the following, we show the generality of these properties, and show the link between the bulk winding number and the presence/absence of edge states, known as bulk-boundary correspondence. In the case of the SSH model, all this hinges on a property of the model known as chiral symmetry.

## 1.4 Chiral Symmetry

In quantum mechanics, we say that a Hamiltonian $\hat{H}$ has a symmetry represented by a unitary operator $\hat{U}$ if

$$
\hat {U} \hat {H} \hat {U} ^ {\dagger} = \hat {H}.\tag{1.22}
$$

In case of a symmetry, $\hat{U}$ and $\hat{H}$ can be diagonalized together, and therefore, $\hat{H}$ has no matrix elements between two eigenstates of $\hat{U}$ with different eigenvalues. This can be understood as a superselection rule: if we partition the Hilbert space into different sectors, i.e., eigenspaces of $\hat{U}$ , labeled by the corresponding eigenvalues, then the dynamics as defined by $\hat{H}$ can be regarded separately in each sector.

## 1.4.1 No Unitary Symmetries

A unitary symmetry can be simply made to disappear if we restrict ourselves to one sector of the Hilbert space. This is how we obtained the bulk momentum-space Hamiltonian, in Sect. 1.2, where the symmetry was the lattice translation operator $\hat{U} = |m + 1,A\rangle \langle m,A| + |m + 1,B\rangle \langle m,B|$ , and the labels of the superselection sectors were the quasimomenta $k$ .

## 1.4.2 A Different Type of Symmetry

The word “symmetry” is also used in a different sense in condensed matter physics. One example is chiral symmetry. We say that a system with Hamiltonian $\hat{H}$ has chiral symmetry, if

$$
\hat {\Gamma} \hat {H} \hat {\Gamma} ^ {\dagger} = - \hat {H},\tag{1.23}
$$

with an operator $\hat{\Gamma}$ that is not only unitary, but fulfils some other criteria as well. Notice the extra minus sign on the right hand side. This has important consequences, which we come to later, but first discuss the criteria on the symmetry operator.

First, the chiral symmetry operator has to be unitary and Hermitian, $\hat{\Gamma}^{\dagger} = \hat{\Gamma}$ , which can be written succinctly as

$$
\hat {\Gamma} ^ {\dagger} \hat {\Gamma} = \hat {\Gamma} ^ {2} = 1.\tag{1.24}
$$

The reason for this requirement is that if the operator $\hat{\Gamma}^{2}$ was nontrivial, it would represent a unitary symmetry, since

$$
\hat {\Gamma} \hat {\Gamma} \hat {H} \hat {\Gamma} \hat {\Gamma} = - \hat {\Gamma} \hat {H} \hat {\Gamma} = \hat {H}.\tag{1.25}
$$

This could still leave room for the chiral symmetry operator to square to a state-independent phase, $\hat{\Gamma}^{2}=e^{i\phi}$ . However, this can be got rid of by a redefinition of the chiral symmetry, $\Gamma\to e^{-i\phi/2}\Gamma$ .

Second, it is also required that the sublattice operator $\hat{\Gamma}$ be local. The system is assumed to consist of unit cells, and matrix elements of $\hat{\Gamma}$ between sites from different unit cells should vanish. In the SSH chain, this means that for $m \neq m'$ , we have $\langle m, \alpha | \hat{\Gamma} | m', \alpha' \rangle = 0$ , for any $\alpha, \alpha' \in (A, B)$ . To keep things simple, we can demand that the sublattice operator act in the same way in each unit cell (although this is not strictly necessary), its action represented by a unitary operator $\hat{\gamma}$ acting on the internal Hilbert space of one unit cell, i.e.,

$$
\hat {\Gamma} = \hat {\gamma} \oplus \hat {\gamma} \oplus \ldots \oplus \hat {\gamma} = \bigoplus_ {m = 1} ^ {N} \hat {\gamma},\tag{1.26}
$$

where N is the number of unit cells.

A third requirement, which is often not explicitly stated, is that the chiral symmetry has to be robust. To understand what we mean by that, first note that in solid state physics, we often deal with Hamiltonians with many local parameters that vary in a controlled or uncontrolled way. An example is the SSH model, where the values of the hopping amplitudes could be subject to spatial disorder. We gather all such parameters in a formal vector, and call it $\xi \in E$ . Here E is the set of all realizations of disorder that we investigate. Instead of talking about the symmetries of a Hamiltonian $\hat{H}$ , we should rather refer to symmetries of a set of Hamiltonians $\{\hat{H}(\underline{\xi})\}$ , for all $\xi \in E$ . This set has chiral symmetry represented by $\hat{I}$ if

$$
\forall \underline {{\xi}} \in \mathcal {E}: \quad \hat {\Gamma} \hat {H} (\underline {{\xi}}) \hat {\Gamma} = - \hat {H} (\underline {{\xi}})\tag{1.27}
$$

with the symmetry operator $\hat{\Gamma}$ independent of the parameters $\underline{\xi}$ . This is the robustness of the chiral symmetry.

## 1.4.3 Consequences of Chiral Symmetry for Energy Eigenstates

We now come to the consequences of chiral symmetry, which are very different from those of conventional symmetries, due to the extra minus sign in its definition, Eq. (1.23).

## 1.4.3.1 Sublattice Symmetry

Chiral symmetry is also called sublattice symmetry. Given the chiral symmetry operator $\hat{\Gamma}$ , we can define orthogonal sublattice projectors $\hat{P}_{A}$ and $\hat{P}_{B}$ , as

$$
\hat {P} _ {A} = \frac {1}{2} (\mathbb {I} + \hat {\Gamma}); \quad \hat {P} _ {B} = \frac {1}{2} (\mathbb {I} - \hat {\Gamma}),\tag{1.28}
$$

where $\mathbb{I}$ represents the identity operator on the Hilbert space of the system. Note that $\hat{P}_A + \hat{P}_B = \mathbb{I}$ , and $\hat{P}_A\hat{P}_B = 0$ . The defining relation of sublattice symmetry, Eq. (1.23), can be written in an equivalent form by requiring that the Hamiltonian induces no transitions from any site on one sublattice to any site on the same sublattice,

$$
\hat {P} _ {A} \hat {H} \hat {P} _ {A} = P _ {B} \hat {H} \hat {P} _ {B} = 0; \qquad \hat {H} = \hat {P} _ {A} \hat {H} \hat {P} _ {B} + \hat {P} _ {B} \hat {H} \hat {P} _ {A}.\tag{1.29}
$$

In fact, using the projectors $\hat{P}_{A}$ and $\hat{P}_{B}$ is an alternative and equivalent way of defining chiral symmetry.

## 1.4.3.2 Symmetric Spectrum

The spectrum of a chiral symmetric Hamiltonian is symmetric. For any state with energy E, there is a chiral symmetric partner with energy -E. This is simply seen,

$$
\hat {H} | \psi_ {n} \rangle = E _ {n} | \psi_ {n} \rangle \quad \Longrightarrow \quad \hat {H} \hat {\Gamma} | \psi_ {n} \rangle = - \hat {\Gamma} \hat {H} | \psi_ {n} \rangle = - \hat {\Gamma} E _ {n} | \psi_ {n} \rangle = - E _ {n} \hat {\Gamma} | \psi_ {n} \rangle .\tag{1.30}
$$

This carries different implications for nonzero energy eigenstates and zero energy eigenstates.

For $E_{n} \neq 0$ , the states $|\psi_{n}\rangle$ and $\hat{\Gamma} |\psi_{n}\rangle$ are eigenstates with different energy, and, therefore, have to be orthogonal. This implies that every nonzero energy eigenstate of $\hat{H}$ has equal support on both sublattices,

$$
\mathrm{If} E _ {n} \neq 0: 0 = \langle \psi_ {n} | \hat {\Gamma} | \psi_ {n} \rangle = \langle \psi_ {n} | P _ {A} | \psi_ {n} \rangle - \langle \psi_ {n} | P _ {B} | \psi_ {n} \rangle .\tag{1.31}
$$

For $E_{n} = 0$ , zero energy eigenstates can be chosen to have support on only one sublattice. This is because

$$
\text {   If   } \hat {H} \left| \psi_ {n} \right\rangle = 0: \quad \hat {H} \hat {P} _ {A / B} \left| \psi_ {n} \right\rangle = \hat {H} \left(\left| \psi_ {n} \right\rangle \pm \hat {\Gamma} \left| \psi_ {n} \right\rangle\right) = 0.\tag{1.32}
$$

These projected zero-energy eigenstates are eigenstates of $\hat{\Gamma}$ , and therefore are chiral symmetric partners of themselves.

## 1.4.4 Sublattice Projectors and Chiral Symmetry of the SSH Model

The Hamiltonian of the SSH model, Eq. (1.1), is bipartite: the Hamiltonian includes no transitions between sites with the same sublattice index. The projectors to the sublattices read

$$
\hat {P} _ {A} = \sum_ {m = 1} ^ {N} | m, A \rangle \langle n, A |; \quad \hat {P} _ {B} = \sum_ {m = 1} ^ {N} | m, B \rangle \langle n, B |.\tag{1.33}
$$

Chiral symmetry is represented by the sublattice operator $\hat{\Sigma}_{z}$ , that multiplies all components of a wavefunction on sublattice B by $(-1)$ ,

$$
\hat {\Sigma} _ {z} = \hat {P} _ {A} - \hat {P} _ {B}.\tag{1.34}
$$

Note that this operator has the properties required of the chiral symmetry operator above: it is unitary, Hermitian, and local.

The chiral symmetry of the SSH model is a restatement of the fact that the Hamiltonian is bipartite,

$$
\hat {P} _ {A} \hat {H} \hat {P} _ {A} = \hat {P} _ {B} \hat {H} \hat {P} _ {B} = 0; \qquad \Longrightarrow \qquad \hat {\Sigma} _ {z} \hat {H} \hat {\Sigma} _ {z} = - \hat {H}.\tag{1.35}
$$

This relation holds because $\hat{H}$ only contains terms that are multiples of $|m,A\rangle \langle m',B|$ , or of $|m,B\rangle \langle m',A|$ with $m,m'\in \mathbb{Z}$ . Upon multiplication from the left and the right by $\hat{\Sigma}_z$ , such a term picks up a single factor of $-1$ (because of the multiplication from the left or because of the multiplication from the right). Note that this relation, equivalent to an anticommutation of $\hat{H}$ and $\hat{\Sigma}_z$ , holds whether or not the hopping amplitudes depend on position: therefore, the chiral symmetry represented by $\hat{\Sigma}_z$ has the required property of robustness.

## 1.4.5 Consequence of Chiral Symmetry: Bulk Winding Number for the SSH Model

We now consider the bulk momentum-space Hamiltonian $\hat{H}(k) = \mathbf{d}(k)\hat{\sigma}$ .

The path of the endpoint of $\mathbf{d}(k)$ , as the wavenumber goes through the Brillouin zone, $k = 0 \rightarrow 2\pi$ , is a closed path on the $d_{x}, d_{y}$ plane. This path has to avoid the origin, d = 0: if there was a k at which $\mathbf{d}(k) = 0$ , the gap would close at this k, and we would not be talking about an insulator. Because of chiral symmetry, the vector $\mathbf{d}(k)$ is restricted to lie on the $d_{x}d_{y}$ plane,

$$
\hat {s i g m a} _ {z} \hat {H} (k) \hat {s i g m a} _ {z} = - \hat {H} (k) \quad \Longrightarrow \quad d _ {z} (k) = 0.\tag{1.36}
$$

The endpoint of $\mathbf{d}(k)$ is then a closed, directed loop on the plane, and thus has a well defined integer winding number about the origin.

## 1.4.5.1 Winding Number as the Multiplicity of Solutions

The simplest way to obtain the winding number graphically is counting the number of times $\mathbf{d}(k)$ intersects a curve that goes from the origin of the $d_{x}, d_{y}$ plane to infinity.

1. Since $\mathbf{d}(k)$ is a directed curve, it has a left side and a right side. Paint the left side blue, the right side red, as shown in Fig. 1.5a.

2. Take a directed curve L going from 0 to infinity. We can call this the “line of sight to infinity”, although it need not be a straight line. A simple choice is the half-infinite line, $d_{y} = 0$ , $d_{x} \geq 0$ . Two other choices are shown in Fig. 1.5.

3. Identify the intersections of $\mathbf{d}(k)$ with $\mathcal{L}$ .

4. Each intersection has a signature: this is +1 if the line of sight meets it from the blue side, -1 for the red side.

5. The winding number $\nu$ is the sum of the signatures.

We now consider how the winding number $\nu$ defined above can change under continuous deformations of L or of $\mathbf{d}(k)$ . We only allow for deformations that keep both curves on the plane, maintain L going from the origin to infinity, and do not create points where $\mathbf{d}(k)=0$ . Due to the deformations the intersections of L and $\mathbf{d}(k)$ can move, but this does not change $\nu$ . They can also appear or disappear, at points where L and $\mathbf{d}(k)$ touch. However, they can only appear or disappear pairwise: a red and a blue intersection together, which does not change $\nu$ . As an example, the two choices of the line of sight L in Fig. 1.5a, have 1 or 3 intersections, but the winding number is +1, for either of them.

![](images/aba3db1a82e52aec79c55e5adba55921f5669b3fe8e684eb1bf5106ee3d1e58f.jpg)

![](images/708509344f8e1db39e51be7f3306a4a5c21ae6f05b714f6530cdd12833e2f3fb.jpg)
Fig. 1.5 The endpoints of the vector $\mathbf{d}(k)$ as $k$ goes across the Brillouin zone (red or blue closed circles)

## 1.4.5.2 Winding Number as an Integral

The winding number can also be written as a compact formula using the unit vector $\tilde{d}$ , defined as

$$
\tilde {\mathbf {d}} = \frac {\mathbf {d}}{| \mathbf {d} |}.\tag{1.37}
$$

This is the result of projecting the curve of $\mathbf{d}(k)$ to the unit circle, as shown in Fig. 1.5b. The vector $\bar{\mathbf{d}}(k)$ is well defined for all $k$ because $\mathbf{d}(k) \neq 0$ .

You can check easily that the winding number $\nu$ is given by

$$
\nu = \frac {1}{2 \pi} \int_ {- \pi} ^ {\pi} \left(\tilde {\mathbf {d}} (k) \times \frac {d}{d k} \tilde {\mathbf {d}} (k)\right) _ {z} d k.\tag{1.38}
$$

To calculate $\nu$ directly from the bulk momentum-space Hamiltonian, note that it is off-diagonal (in the basis of eigenstates of the chiral symmetry operator $\hat{\sigma}_{z}$ ),

$$
H (k) = \left( \begin{array}{c c} 0 & h (k) \\ h ^ {*} (k) & 0 \end{array} \right); \qquad \qquad h (k) = d _ {x} (k) - i d _ {y} (k).\tag{1.39}
$$

The winding number of $\mathbf{d}(k)$ can be written as an integral, using the complex logarithm function, $\log (|h|e^{i\arg h}) = \log |h| + i\arg h$ . It is easy to check that

$$
\nu = \frac {1}{2 \pi i} \int_ {- \pi} ^ {\pi} d k \frac {d}{d k} \log h (k).\tag{1.40}
$$

Here during the calculation of the integral, the branch cut for the logarithm is always shifted so that the derivative is always well defined. The above integral is always real, since $|h(k = -\pi)| = |h(k = \pi)|$ .

## 1.4.5.3 Winding Number of the SSH Model

For the SSH model, the winding number is either 0 or 1, depending on the parameters. In the trivial case, when the intracell hopping dominates the intercell hopping, v > w, the winding number is v = 0. In the topological case, when w > v, we have v = 1.

To change the winding number $\nu$ of the SSH model, we need to either (a) pull the path of $\mathbf{d}(k)$ through the origin in the $d_x$ , $d_y$ plane, or (b) lift it out of the plane and put it back on the plane at a different position. This is illustrated in Fig. 1.6. Method (a) requires closing the bulk gap. Method (b) requires breaking chiral symmetry.

![](images/951643ec4f33931981d64404ed7427afdc9d8be4caae82441eb3a83c2787d376.jpg)
Fig. 1.6 The endpoints of the vector $\mathbf{d}(k)$ as $k$ goes across the Brillouin zone (red or blue closed circles), for various parameter settings in the SSH model. In (a), intercell hopping is kept constant at $w = 1$ , while the intracell hopping is increased from $v = 0$ to $v = 2.3$ . In the process, the bulk gap was closed and reopened, as the origin (black point) falls on one of the blue circles. The winding number is changed from 1 to 0. In (b), we again keep $w = 1$ , and increase $v$ from 0 to 2.3, but this time avoid closing the bulk gap by introducing a sublattice potential, $H_{\mathrm{sublattice}} = u\hat{\sigma}_z$ . We do this by tuning a parameter $\theta$ from 0 to $\pi$ , and setting $v = 1.15(1 - \cos \theta)$ , and $u = \sin \theta$ . At the end of the process, $\theta = \pi$ , there is no sublattice potential, so chiral symmetry is restored. The winding number has been changed from 1 to 0

## 1.5 Number of Edge States as Topological Invariant

We now introduce the notion of adiabatic deformation of insulating Hamiltonians. An insulating Hamiltonian is adiabatically deformed if

\- its parameters are changed continuously,

\- the important symmetries of the system are maintained,

• the bulk gap around E = 0 remains open.

The deformation is a fictitious process, and does not take place in time. However, if we do think of it as a process in real time, the adiabatic theorem $[15]$ tells us, that, starting from the many-body ground state (separated from excited states by the energy gap), and performing the deformation slowly enough, we end up in the ground state, at least as far as the bulk of the system is concerned. At the edges of a system, changes can occur, and there is a subtle point to be made about adiabatic deformations being slow, but not too slow, that the edges should still be considered separately. We will come back to this point in Chap. 4.

## 1.5.1 Adiabatic Equivalence of Hamiltonians

Two insulating Hamiltonians are said to be adiabatically equivalent or adiabatically connected if there is an adiabatic deformation connecting them, that respects the important symmetries. For example, in the phase diagram Fig. 1.7 of the SSH model,

Fig. 1.7 Phase diagram of the SSH model. The winding number of the bulk momentum-space Hamiltonian $\hat{H}(k)$ can be $\nu = 0$ , if $v > w$ , or $\nu = 1$ , if $v < w$ . This defines the trivial (gray) and the topological phase (white). The boundary separating these phases (black solid line), corresponds to $v = w$ , where the bulk gap closes at some $k$ . Two Hamiltonians in the same phase are adiabatically connected
![](images/4380d711539d87e596af953d4e1c08a5a97e4707dfd253d63c6646099d638608.jpg)
the two Hamiltonians corresponding to the two black points in the topological phase $(w > v)$ are adiabatically connected, as one can draw a path between them which does not cross the gapless topological-trivial phase boundary w = v.

## 1.5.2 Topological Invariant

We call an integer number characterizing an insulating Hamiltonian a topological invariant, or adiabatic invariant, if it cannot change under adiabatic deformations. Note that the use of adiabatic deformations implies two properties of the topological invariant: (1) it is only well defined in the thermodynamic limit, (2) it depends on the symmetries that need to be respected. An example for a topological invariant is the winding number $\nu$ of the SSH model.

We know that two insulating Hamiltonians are not adiabatically equivalent if their topological invariants differ. Consider as an example two Hamiltonians corresponding to two points on different sides of the phase boundary in Fig. 1.7 of the SSH model. They are not adiabatically connected in the phase diagram. Nevertheless, one might think that continuously modifying the bulk Hamiltonian by the addition of extra terms, while maintaining chiral symmetry, can lead to a connection between them. However, their winding numbers differ, and since winding numbers cannot change under adiabatic deformation, we know that they are not adiabatically equivalent.

## 1.5.3 Number of Edge States as a Topological Invariant

We have seen in Sect. 1.3.2, that the number of edge states at one end of the SSH model was an integer that did not change under a specific type of adiabatic deformation. We now generalize this example.

Consider energy eigenstates at the left end of a gapped chiral symmetric one-dimensional Hamiltonian in the thermodynamic limit, i.e., with length $N \rightarrow \infty$ , in an energy window from $-\varepsilon < E < \varepsilon$ , with $\varepsilon$ in the bulk gap. There can be nonzero-energy edge states in this energy window, and zero-energy edge states as well. Each nonzero-energy state has to have a chiral symmetric partner, with the state and its partner occupying the same unit cells (the chiral symmetry operator is a local unitary). The number of zero-energy states is finite (because of the gap in the bulk), and they can be restricted to a single sublattice each. There are $N_{A}$ zero-energy states on sublattice A, and $N_{B}$ states on sublattice B.

Consider the effect of an adiabatic deformation of the Hamiltonian, indexed by some continuous parameter $d:0\to1$ , on the number $N_{A}-N_{B}$ . The Hamiltonian respects chiral symmetry, and its bulk energy gap exceeds $2\varepsilon$ , for all values of d.

The deformation can create zero-energy states by bringing a nonzero-energy edge state $|\Psi_{0}(d=0)\rangle$ to zero energy, $E_{0}(d)=0$ for $d\geq d'$ but not for $d<d'$ . In that case, the chiral symmetric partner of $|\Psi_{0}\rangle$ , which is $\Gamma|\Psi_{0}(d)\rangle$ up to a phase factor, has to move simultaneously to zero energy. The newly created zero energy edge states are $\hat{P}_{A}|\Psi_{0}(d')\rangle$ and $\hat{P}_{B}|\Psi_{0}(d')\rangle$ , which occupy sublattice A and B, respectively. Thus, the number $N_{A}-N_{B}$ is unchanged.

The deformation can also bring a zero energy state $\left|\Psi_{0}\right\rangle$ to energy E > 0 at some $d = d'$ . However, it must also create a chiral symmetric partner with energy E < 0 at the same $d'$ . This is the time reverse of the process of the previous paragraph: here, both $N_{A}$ and $N_{B}$ must decrease by 1, and, again, $N_{A} - N_{B}$ is unchanged.

The deformation can move nonzero-energy states in or out of the $-\varepsilon < E < \varepsilon$ energy window. This obviously has no effect on the number $N_{A} - N_{B}$ .

Due to the deformation, the wavefunction of a zero-energy eigenstate can change so that it extends deeper and deeper into the bulk. However, because of the gap condition, zero-energy states have to have wavefunctions that decay exponentially towards the bulk, and so this process cannot move them away from the edge. Thus, $N_{A}$ and $N_{B}$ cannot be changed this way.

The arguments above show that $N_{A} - N_{B}$ , the net number of edge states on sublattice A at the left edge, is a topological invariant.

## 1.5.4 Bulk-Boundary Correspondence in the SSH Model

We have introduced two topological invariants for the SSH model: the winding number $\nu$ , of Eq. (1.38), and the net number of edge states, $N_A - N_B$ , of this section. The first one was obtained from the bulk Hamiltonian only, the second by looking at the low energy sector of the left edge. In the trivial case of the SSH model, v > w, both are 0; in the topological case, v < w, both are 1. This shows that we can use the bulk topological invariant (the winding number) to make simple robust predictions about the low-energy physics at the edge. This is a simple example for the bulk-boundary correspondence, a recurrent theme in the theory of topological insulators, which will reappear in various models in the forthcoming chapters.

## 1.5.5 Bound States at Domain Walls

Edge states do not only occur at the ends of an open chain, but also at domain walls between different insulating domains of the same chain. This can be understood via the fully dimerized limit. The example in Fig. 1.8 hosts two types of domain walls: one containing a single isolated site, which hosts a zero-energy state (no onsite potentials are allowed), and one containing a trimer. On a trimer, the odd superposition of the two end sites form a zero-energy eigenstate. In the example of Fig. 1.8, this is

$$
\hat {H} (| 6, B \rangle - | 7, B \rangle) = 0.\tag{1.41}
$$

Note that, just as the edge states at the ends of the chain, these zero-energy states at the domain walls have wavefunctions that take nonzero values on one sublattice only.

From a perfect dimerized phase without domains it is only possible to germinate an even number of domain walls. This means that if one encounters a domain wall with a localized state on one sublattice then there will be another domain wall somewhere in the system—possibly at the system's edge—with a localized state on the opposite sublattice.

Consider a domain wall in an SSH system that is not in the fully dimerized limit. The wavefunctions of the edge states at the domain walls will penetrate to some small depth into the bulk, with exponentially decaying evanescent tails. For two domain walls at a distance of M unit cells, the two edge states on the walls will hybridize, form “bonding” and “anti-bonding” states. At half filling, of these only the negative energy eigenstate will be occupied. This state hosts a single electron, however, its wavefunction is localized with equal weight on the two domain walls.

![](images/1fe10c09756e24cbe2330a66313c773d5b66f51ca8637f900dc484b5b19addd4.jpg)
Fig. 1.8 A long, fully dimerized SSH chain with 3 domains. The boundaries between the domains, the “domain walls”, host zero energy eigenstates (yellow shading). These can be localized on a single site, as for the domain wall at n = 3, or on a superposition of sites, as the odd superposition of the ends of the trimer shared between the n = 6 and n = 7 unit cells

Hence each domain wall, when well separated from other domain walls and the ends of the chain, will carry half an electronic charge. This effect is sometimes referred to as “fractionalization” of the charge.

## 1.5.6 Exact Calculation of Edge States

The zero energy edge states of the SSH model can also be calculated exactly, even in the absence of translational invariance. Take an SSH model on N unit cells, with complex intracell and intercell hopping amplitudes,

$$
\hat {H} = \sum_ {m = 1} ^ {N} \left(v _ {m} | m, B \rangle \langle m, A | + h. c.\right) + \sum_ {m = 1} ^ {N - 1} \left(w _ {m} | m + 1, A \rangle \langle m, B | + h. c.\right).\tag{1.42}
$$

We are looking for a zero energy eigenstate of this Hamiltonian,

$$
\hat {H} \sum_ {m = 1} ^ {N} \left(a _ {m} | m, A \rangle + b _ {m} | m, B \rangle\right) = 0.\tag{1.43}
$$

This gives us 2N equations for the amplitudes $a_{m}$ and $b_{m}$ , which read

$$
m = 1, \ldots , N - 1: \qquad v _ {m} a _ {m} + w _ {m} a _ {m + 1} = 0; \qquad w _ {m} b _ {m} + v _ {m + 1} b _ {m + 1} = 0;\tag{1.44a}
$$

boundaries :

$$
v _ {N} a _ {N} = 0;
$$

$$
v _ {1} b _ {1} = 0.\tag{1.44b}
$$

The first set of equations is solved by

$$
m = 2, \ldots , N:
$$

$$
a _ {m} = \prod_ {j = 1} ^ {m - 1} \frac {- v _ {j}}{w _ {j}} a _ {1};\tag{1.45}
$$

$$
m = 1, \ldots , N - 1:
$$

$$
b _ {m} = \frac {- v _ {N}}{w _ {m}} \prod_ {j = m + 1} ^ {N - 1} \frac {- v _ {j}}{w _ {j}} b _ {N}.\tag{1.46}
$$

However, we also have to fulfil Eq. (1.44b), which give

$$
b _ {1} = a _ {N} = 0.\tag{1.47}
$$

These equations together say that, in the generic case, there is no zero energy eigenstate, $a_{m} = b_{m} = 0$ .

Although there is no exactly zero energy state, Eqs. (1.45), (1.46) and (1.47) admit two approximate solutions in the thermodynamic limit, $N \rightarrow \infty$ , if the average intercell hopping is stronger than the intracell hopping. More precisely, we define the “bulk average values”,

$$
\overline {{\log | v |}} = \frac {1}{N - 1} \sum_ {m = 1} ^ {N - 1} \log | v _ {m} |; \quad \overline {{\log | w |}} = \frac {1}{N - 1} \sum_ {m = 1} ^ {N - 1} \log | w _ {m} |.\tag{1.48}
$$

Equations (1.45) and (1.46) translate to

$$
| a _ {N} | = | a _ {1} | e ^ {- (N - 1) / \xi}; \qquad | b _ {1} | = | b _ {N} | e ^ {- (N - 1) / \xi} \frac {| v _ {N} |}{| v _ {1} |},\tag{1.49}
$$

with the localization length

$$
\xi = \frac {1}{\overline {{\log | w |}} - \overline {{\log | v |}}}.\tag{1.50}
$$

If in the thermodynamic limit, the bulk average values, Eq. (1.48) make sense, and $\xi > 0$ , we have two approximate zero energy solutions,

$$
| L \rangle = \sum_ {m = 1} ^ {N} a _ {m} | m, A \rangle ; \quad | R \rangle = \sum_ {m = 1} ^ {N} b _ {m} | m, B \rangle ,\tag{1.51}
$$

with the coefficients $a_{m}$ and $b_{m}$ chosen according to Eqs. (1.45) and (1.46), and $a_{1}$ , respectively, $b_{N}$ , used to fix the norm of $|L\rangle$ , respectively, $|R\rangle$ .

## 1.5.6.1 Hybridization of Edge States

The two states $|L\rangle$ and $|R\rangle$ hybridize under $\hat{H}$ to an exponentially small amount, and this induces a small energy splitting. We can obtain an estimate for the splitting, and the energy eigenstates, to a good approximation using adiabatic elimination of the other eigenstates. In this approximation, the central quantity is the overlap

$$
\langle R | \hat {H} | L \rangle = \left| a _ {1} e ^ {- (N - 1) / \xi} v _ {N} b _ {N} \right| e ^ {i \phi},\tag{1.52}
$$

with some $\phi\in[0,2\pi)$ . The energy eigenstates are approximated as

$$
\left| 0 + \right\rangle = \frac {e ^ {- i \phi / 2} \left| L \right\rangle + e ^ {i \phi / 2} \left| R \right\rangle}{\sqrt {2}}; \qquad E _ {+} = \left| a _ {1} e ^ {- (N - 1) / \xi} v _ {N} b _ {N} \right|;\tag{1.53}
$$

$$
| 0 - \rangle = \frac {e ^ {- i \phi / 2} | L \rangle - e ^ {i \phi / 2} | R \rangle}{\sqrt {2}}; \qquad E _ {-} = - \left| a _ {1} e ^ {- (N - 1) / \xi} v _ {N} b _ {N} \right|.\tag{1.54}
$$

The energy of the hybridized states thus is exponentially small in the system size.

## Problems

## 1.1 Higher winding numbers

The SSH model is one-dimensional in space, and has a two-dimensional internal Hilbert space. Construct a lattice model that has these properties of the SSH model, but which has a bulk winding number of 2. Generalize the construction for an arbitrary integer bulk winding number.

## 1.2 Complex-valued hopping amplitudes

Generalize the SSH model in the following way. Assume that the hopping amplitudes $v = |v|e^{i\phi_{v}}$ and $w = |w|e^{i\phi_{w}}$ are complex, and include a third complex-valued hopping amplitude $z = |z|e^{i\phi_{z}}$ between the states $|m, A\rangle$ and $|m + 1, B\rangle$ for every m. Provide a specific example where the tuning of one of the phases changes the bulk winding number.

## 1.3 A possible generalization to two dimensions

Consider a two dimensional generalization of the SSH model. Take parallel copies of the SSH chain and couple them without breaking chiral symmetry. What will happen with the edge states?

# Chapter 2 Berry Phase, Chern Number

To describe the theory of topological band insulators we will use the language of adiabatic phases. In this chapter we review the basic concepts: the Berry phase, the Berry curvature, and the Chern number. We further describe the relation between the Berry phase and adiabatic dynamics in quantum mechanics. Finally, we illustrate these concepts using a two-level system as a simple example.

For pedagogical introductions, we refer the reader to Berry's original paper [6], and papers from the American Journal of Physics [14, 18]. For the application to solid state physics, we will mostly build on Resta's lecture note [26], and the review paper [36].

## 2.1 Discrete Case

The subject of adiabatic phases is strongly related to adiabatic quantum dynamics, when a Hamiltonian is slowly changed in time, and the time evolution of the quantum state follows the instantaneous eigenstate of the Hamiltonian. In that context, as time is a continuous variable and the time-dependent Schrödinger equation is a differential equation, the adiabatic phase and the related concepts are expressed using differential operators and integrals. We will arrive to that point later during this chapter; however, we start the discussion using the language of discrete quantum states. Besides the conceptual simplicity, this language also offers an efficient tool for the numerical evaluation of the Chern number, which is an important topological invariant for two-dimensional electron systems.

## 2.1.1 Relative Phase of Two Nonorthogonal Quantum States

In quantum mechanics, the state of a physical system is represented by an equivalence class of vectors in a Hilbert space: a multiplication by a complex phase factor does not change the physical content. A gauge transformation is precisely such a multiplication:

$$
\left| \right. \Psi \left. \right\rangle\rightarrow e ^ {i \alpha} \left| \right. \Psi \left. \right\rangle , \quad \text { with } \alpha \in [ 0, 2 \pi).\tag{2.1}
$$

In that sense, the phase of a vector $|\Psi\rangle$ does not represent physical information. We can try to define the relative phase $\gamma_{12}$ of two nonorthogonal states $|\Psi_{1}\rangle$ and $|\Psi_{2}\rangle$ as

$$
\gamma_ {1 2} = - \arg \left\langle \Psi_ {1} \mid \Psi_ {2} \right\rangle ,\tag{2.2}
$$

where $\arg(z)$ denotes the phase of the complex number z, with the specification that $\arg(z) \in (-\pi, \pi]$ . Clearly, the relative phase $\gamma_{12}$ fulfils

$$
e ^ {- i \gamma_ {1 2}} = \frac {\langle \Psi_ {1} | \Psi_ {2} \rangle}{| \langle \Psi_ {1} | \Psi_ {2} \rangle |}.\tag{2.3}
$$

However, the relative phase is not invariant under a local gauge transformation,

$$
\left| \right. \Psi_ {j} \left. \right\rangle\rightarrow e ^ {i \alpha_ {j}} \left| \right. \Psi_ {j} \left. \right\rangle \quad e ^ {- i \gamma_ {1 2}} \rightarrow e ^ {- i \gamma_ {1 2} + i (\alpha_ {2} - \alpha_ {1})}.\tag{2.4}
$$

## 2.1.2 Berry Phase

Take $N \geq 3$ states in a Hilbert space, order them in a loop, and ask about the phase around the loop. As we show below, the answer—the Berry phase—is gauge invariant. For states $\left|\Psi_{j}\right\rangle$ , with $j = 1, 2, \ldots, N$ , and for the ordered list $L = (1, 2, \ldots, N)$ which define the loop, shown in Fig. 2.1, the Berry phase is defined as

$$
\gamma_ {L} = - \arg e ^ {- i (\gamma_ {1 2} + \gamma_ {2 3} + \dots + \gamma_ {N 1})} = - \arg \left(\langle \Psi_ {1} | \Psi_ {2} \rangle \langle \Psi_ {2} | \Psi_ {3} \rangle \dots \langle \Psi_ {N} | \Psi_ {1} \rangle\right).\tag{2.5}
$$

To show the gauge invariance of the Berry phase, it can be rewritten as

$$
\gamma_ {L} = - \arg \operatorname{Tr} \left(\left| \Psi_ {1} \right\rangle \left\langle \Psi_ {1} \right| \left| \Psi_ {2} \right\rangle \left\langle \Psi_ {2} \right| \dots \left| \Psi_ {N} \right\rangle \left\langle \Psi_ {N} \right|\right).\tag{2.6}
$$

Here, we expressed the Berry phase $\gamma_{L}$ using projectors that are themselves gauge invariant.

Even though the Berry phase is not the expectation value of some operator, it is a gauge invariant quantity, and as such, it can have a direct physical significance. We will find such a significance, but first, we want to gain more intuition about its behaviour.

## 2.1.3 Berry Flux

Consider a Hilbert space of quantum states, and a finite two-dimensional square lattice with points labelled by $n, m \in Z$ , $1 \leq n \leq N$ , and $1 \leq m \leq M$ . Assign a quantum state $|\Psi_{n,m}\rangle$ from the Hilbert space to each lattice site. Say you want to know the Berry phase of the loop L around this set,

$$
\begin{array}{c} \gamma_ {L} = - \arg \exp \Bigg [ - i \left(\sum_ {n = 1} ^ {N - 1} \gamma_ {(n, 1), (n + 1, 1)} + \sum_ {m = 1} ^ {M - 1} \gamma_ {(N, m), (N, m + 1)} \right. \\ \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad + \sum_ {n = 1} ^ {N - 1} \gamma_ {(n + 1, M), (n, M)} + \sum_ {m = 1} ^ {M - 1} \gamma_ {(1, m + 1), (1, m)} \Bigg) \Bigg ] \end{array}\tag{2.7}
$$

as shown in Fig. 2.1. Although the Berry phase is a gauge invariant quantity, calculating it according to the recipe above involves multiplying together many gauge dependent complex numbers. The alternative route, via Eq. (2.6), involves multiplying gauge independent matrices, and then taking the trace.

There is a way to break the calculation of the Berry phase of the loop down to a product of gauge independent complex numbers. To each plaquette (elementary square) on the grid, with n, m indexing the lower left corner, we define the

![](images/2bad46e69389b185922bb3ace00c5336d8db639ad7a7dfab1b6909bbe59c1ce0.jpg)

(b)
![](images/29a422d0c75775a64a27a6a6f14785624aa364c51529c312ca49da8721d852ea.jpg)
Fig. 2.1 Berry phase, Berry flux and Berry curvature for discrete quantum states. (a) The Berry phase $\gamma_{L}$ for the loop L consisting of N = 3 states is defined from the relative phases $\gamma_{12}, \gamma_{23}, \gamma_{31}$ . (b) The Berry phase of a loop defined on a lattice of states can be expressed as the sum of the Berry phases $F_{1,1}$ and $F_{2,1}$ of the plaquettes enclosed by the loop. The plaquette Berry phase $F_{n,m}$ is also called Berry flux

Berry flux $F_{n,m}$ of the plaquette using the sum of the relative phases around its boundary,

$$
\begin{array}{c} F _ {n m} = - \arg \exp \left[ - i \left(\gamma_ {(n, m), (n + 1, m)} + \gamma_ {(n + 1, m), (n + 1, m + 1)} \right. \right. \\ \left. \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad + \gamma_ {(n + 1, m + 1), (n, m + 1)} + \gamma_ {(n, m + 1), (n, m)}) \right], \end{array}\tag{2.8}
$$

for $n = 1, \ldots, N$ and $m = 1, \ldots, M$ . Note that the Berry flux is itself a Berry phase and is therefore gauge invariant. Alternatively, we can also write

$$
\begin{array}{c} F _ {n m} = - \arg \big (\langle \Psi_ {n, m} \mid \Psi_ {n + 1, m} \rangle   \langle \Psi_ {n + 1, m} \mid \Psi_ {n + 1, m + 1} \rangle \\ \qquad \qquad \qquad \qquad \qquad \qquad \qquad \langle \Psi_ {n + 1, m + 1} \mid \Psi_ {n, m + 1} \rangle   \langle \Psi_ {n, m + 1} \mid \Psi_ {n, m} \rangle \big), \end{array}\tag{2.9}
$$

Now consider the product of all plaquette phase factors $e^{-iF_{nm}}$ ,

$$
\begin{array}{l} \prod_ {n = 1} ^ {N - 1} \prod_ {m = 1} ^ {M - 1} e ^ {- i F _ {n m}} = \exp \left[ - i \sum_ {n = 1} ^ {N - 1} \sum_ {m = 1} ^ {M - 1} F _ {n m} \right] = \exp \left[ - i \sum_ {n = 1} ^ {N - 1} \sum_ {m = 1} ^ {M - 1} \left(\gamma_ {(n, m), (n + 1, m)} + \gamma_ {(n + 1, m), (n + 1, m + 1)} + \gamma_ {(n + 1, m + 1), (n, m + 1)} + \gamma_ {(n, m + 1), (n, m)}\right) \right] \end{array}\tag{2.10}
$$

Each internal edge of the lattice is shared between two plaquettes, and therefore occurs twice in the product. However, since we fixed the orientation of the plaquette phases, these two contributions will always be complex conjugates of each other, and cancel each other. Therefore the exponent in the right-hand-side of Eq. (2.10) simplifies to the exponent appearing in Eq. (2.7), implying

$$
\exp \bigg [ - i \sum_ {n = 1} ^ {N - 1} \sum_ {m = 1} ^ {M - 1} F _ {n m} \bigg ] = e ^ {- i \gamma_ {L}}.\tag{2.11}
$$

This result is reminiscent of the Stokes theorem connecting the integral of the curl of a vector field on an open surface and the line integral of the vector field along the boundary of the surface. In Eq. (2.11), the sum of the relative phases, i.e., the Berry phase $\gamma_{L}$ , plays the role of the line integral, whereas the double sum of the Berry fluxes plays the role of the surface integral. There is an important difference with respect to the Stokes theorem, namely, the equality of the total Berry flux and the Berry phase is not guaranteed: Eq. (2.11) only tells that they are either equal or have a difference of $2\pi$ times an integer.

## 2.1.4 Chern Number

Consider states in a Hilbert space arranged on a grid as above, $|\Psi_{n,m}\rangle$ , with $n, m \in Z$ , $1 \leq n \leq N$ , and $1 \leq m \leq M$ , but now imagine this grid to be on the surface of a torus. We use the same definition for the Berry flux per plaquette as in (2.9), but now with $n \mod N + 1$ in place of $n + 1$ and $m \mod M + 1$ in place of $m + 1$ .

The product of the Berry flux phase factors of all plaquettes is now 1,

$$
\prod_ {m = 1} ^ {M} \prod_ {n = 1} ^ {N} e ^ {- i F _ {n m}} = 1.\tag{2.12}
$$

The same derivation can be applied as for Eq. (2.11) above, but now every edge is an internal edge, and so all contributions to the product cancel.

The Chern number Q associated to our structure is defined via the sum of the Berry fluxes of all the plaquettes forming the closed torus surface:

$$
Q = \frac {1}{2 \pi} \sum_ {n m} F _ {n m}.\tag{2.13}
$$

The fact that the Chern number Q is defined via the gauge invariant Berry fluxes ensures that Q itself is gauge invariant. Furthermore, taking the arg of Eq. (2.12) proves that the Chern number Q is an integer.

It is worthwhile to look a little deeper into the discrete formula for the Chern number. We can define modified Berry fluxes $\tilde{F}_{nm}$ as

$$
\tilde {F} _ {n m} = \gamma_ {(n, m), (n + 1, m)} + \gamma_ {(n + 1, m), (n + 1, m + 1)} + \gamma_ {(n + 1, m + 1), (n, m + 1)} + \gamma_ {(n, m + 1), (n, m)}.\tag{2.14}
$$

Since each edge is shared between two neighboring plaquettes, the sum of the modified Berry fluxes over all plaquettes vanishes,

$$
\sum_ {m = 1} ^ {M} \sum_ {n = 1} ^ {N} \tilde {F} _ {n m} = 0.\tag{2.15}
$$

If $-\pi \leq \tilde{F}_{nm} < \pi$ , then we have $\tilde{F}_{nm} = F_{nm}$ . However, $\tilde{F}_{nm}$ can be outside the range $[-\pi, \pi)$ : then as the logarithm is taken in Eq. (2.8), $F_{nm}$ is taken back into $[-\pi, \pi)$ by adding a (positive or negative) integer multiple of $2\pi$ . In that case, we say the plaquette nm contains a number $Q_{nm} \in Z$ of vortices, with

$$
Q _ {n m} = \frac {F _ {n m} - \tilde {F} _ {n m}}{2 \pi} \in \mathbb {Z}.\tag{2.16}
$$

We have found a simple picture for the Chern number: The Chern number Q, that is, the sum of the Berry fluxes of all the plaquettes of a closed surface, is the number of vortices on the surface,

$$
Q = \frac {1}{2 \pi} \sum_ {n m} F _ {n m} = \sum_ {n m} Q _ {n m} \in \mathbb {Z}.\tag{2.17}
$$

Although we proved it here for the special case of a torus, the derivation is easily generalized to all orientable closed surfaces. We focused on the torus, because this construction can be used as a very efficient numerical recipe to discretize and calculate the (continuum) Chern number of a 2-dimensional insulator [12], to be defined in Sect. 2.2.4.

## 2.2 Continuum Case

We now assume that instead of a discrete set of states, $\{| \Psi_j \rangle\}$ , we have a continuum, $|\Psi(\mathbf{R}) \rangle$ , where the $\mathbf{R}$ 's are elements of some $D$ -dimensional parameter space $\mathcal{P}$ .

## 2.2.1 Berry Connection

We take a smooth directed path $\mathcal{C}$ , i.e., a curve in the parameter space $\mathcal{P}$ ,

$$
\mathcal {C}: [ 0, 1) \to \mathcal {P}, \quad t \mapsto \mathbf {R} (t).\tag{2.18}
$$

We assume that all components of $|\Psi(\mathbf{R})\rangle$ are smooth, at least in an open neighborhood of the curve C. The relative phase between two neighbouring states on the curve C, corresponding to the parameters R and $R + dR$ , is

$$
e ^ {- i \Delta \gamma} = \frac {\langle \Psi (\mathbf {R}) | \Psi (\mathbf {R} + d \mathbf {R}) \rangle}{| \langle \Psi (\mathbf {R}) | \Psi (\mathbf {R} + d \mathbf {R}) \rangle |}; \quad \Delta \gamma = i \left\langle \Psi (\mathbf {R}) \right| \nabla_ {\mathbf {R}} | \Psi (\mathbf {R}) \rangle \cdot d \mathbf {R},\tag{2.19}
$$

obtained to first order in $dR \rightarrow 0$ . The quantity multiplying dR on the right-hand side defines the Berry connection,

$$
\mathbf {A} (\mathbf {R}) = i \left\langle \Psi (\mathbf {R}) \mid \nabla_ {\mathbf {R}} \Psi (\mathbf {R}) \right\rangle = - \operatorname{Im} \left\langle \Psi (\mathbf {R}) \mid \nabla_ {\mathbf {R}} \Psi (\mathbf {R}) \right\rangle .\tag{2.20}
$$

Here $\left|\nabla_{\mathbf{R}}\varPsi(\mathbf{R})\right\rangle$ is defined by requiring for every Hilbert space vector $\left|\varPhi\right\rangle$ , that

$$
\left\langle \Phi \mid \nabla_ {\mathbf {R}} \Psi (\mathbf {R}) \right\rangle = \nabla_ {\mathbf {R}} \left\langle \Phi \mid \Psi (\mathbf {R}) \right\rangle .\tag{2.21}
$$

The second equality in Eq. (2.20) follows from the conservation of the norm, $\nabla_{\mathbf{R}}\left\langle \Psi (\mathbf{R})\mid \Psi (\mathbf{R})\right\rangle = 0$

We have seen in the discrete case that the relative phase of two states is not gauge invariant; neither is the Berry connection. Under a gauge transformation, it changes as

$$
| \Psi (\mathbf {R}) \rangle \to e ^ {i \alpha (\mathbf {R})} | \Psi (\mathbf {R}) \rangle : \qquad \mathbf {A} (\mathbf {R}) \to \mathbf {A} (\mathbf {R}) - \nabla_ {\mathbf {R}} \alpha (\mathbf {R}).\tag{2.22}
$$

## 2.2.2 Berry Phase

Consider a closed directed curve C in parameter space. The Berry phase along the curve is defined as

$$
\gamma (\mathcal {C}) = - \arg \exp \left[ - i \oint_ {\mathcal {C}} \mathbf {A} \cdot d \mathbf {R} \right]\tag{2.23}
$$

The Berry phase of a closed directed curve is gauge invariant, since it can be interpreted as a limiting case of the discrete Berry phase, via Eqs. $(2.20)$ , $(2.19)$ , and $(2.5)$ , and the latter has been shown to be gauge invariant.

## 2.2.3 Berry Curvature

As in the discrete case above, we would like to express the gauge invariant Berry phase as a surface integral of a gauge invariant quantity. This quantity is the Berry curvature. Similarly to the discrete case, we consider a two-dimensional parameter space, and for simplicity denote the parameters as x and y. We take a simply connected region F in this two-dimensional parameter space, with the oriented boundary curve of this surface denoted by $\partial F$ , and consider the continuum Berry phase corresponding to the boundary.

## 2.2.3.1 Smoothness of the Manifold of States

Before relating the Berry phase to the Berry curvature, an important note on the manifold $|\Psi(\mathbf{R})\rangle$ of considered states is in order. From now on, we consider a manifold of states, living in our two-dimensional parameter space, that is smooth, in the sense that the map $\mathbf{R} \mapsto |\Psi(\mathbf{R})\rangle \langle\Psi(\mathbf{R})|$ is smooth. Importantly, this condition does not necessarily imply that the function $\mathbf{R} \mapsto |\Psi(\mathbf{R})\rangle$ , also referred to as a gauge describing our manifold, is smooth. (For further discussion and examples, see Sect. 2.5.1.) Nevertheless, even if the gauge $\mathbf{R} \mapsto |\Psi(\mathbf{R})\rangle$ is not smooth in a point $R_{0}$ of the parameter space, one can always find an alternative gauge $|\Psi'(\mathbf{R})\rangle$ which is (i) locally smooth, that is, smooth in the point $R_{0}$ , and (ii) locally generates the same map as $|\Psi(\mathbf{R})\rangle$ , that is, for which $|\Psi'(\mathbf{R})\rangle\langle\Psi'(\mathbf{R})| = |\Psi(\mathbf{R})\rangle\langle\Psi(\mathbf{R})|$ in an infinitesimal neighborhood of $R_{0}$ . Let us formulate an intuitive argument supporting the latter claim using quantum-mechanical perturbation theory. Take the Hamiltonian $\hat{H}(\mathbf{R}) = -|\Psi(\mathbf{R})\rangle\langle\Psi(\mathbf{R})|$ , which can be substituted in the infinitesimal neighborhood of $R_{0}$ with $\hat{H}(\mathbf{R}_{0} + \Delta\mathbf{R}) = \hat{H}(\mathbf{R}_{0}) + \Delta\mathbf{R} \cdot (\nabla\hat{H})(\mathbf{R}_{0})$ . According to first-order perturbation theory, the ground state of the latter is given by

$$
\left| \Psi^ {\prime} (\mathbf {R} _ {0} + \Delta \mathbf {R}) \right\rangle = | \Psi (\mathbf {R} _ {0}) \rangle - \sum_ {n = 2} ^ {D} | \Psi_ {n} (\mathbf {R} _ {0}) \rangle \langle \Psi_ {n} (\mathbf {R} _ {0}) | \Delta \mathbf {R} \cdot (\nabla \hat {H}) (\mathbf {R} _ {0}) | \Psi (\mathbf {R} _ {0}) \rangle ,\tag{2.24}
$$

where the states $|\Psi_{n}(\mathbf{R}_{0})\rangle$ ( $n=2,3,\ldots,D$ ), together with $|\Psi(\mathbf{R}_{0})\rangle$ , form a basis of the Hilbert space. On the one hand, Eq. (2.24) defines a function that is smooth in $R_{0}$ , hence the condition (i) above is satisfied. On the other hand, as $|\Psi'(\mathbf{R}_{0}+\Delta\mathbf{R})\rangle$ is the ground state of $\hat{H}(\mathbf{R}_{0}+\Delta\mathbf{R})$ , condition (ii) is also satisfied.

## 2.2.3.2 Berry Phase and Berry Curvature

Now return to our original goal and try to express the Berry phase as a surface integral of a gauge invariant quantity. We start by relating the Berry phase to its discrete counterpart:

$$
\oint_ {\partial \mathcal {F}} \mathbf {A} \cdot d \mathbf {R} = \lim _ {\Delta x, \Delta y \rightarrow 0} \gamma_ {\partial \mathcal {F}},\tag{2.25}
$$

where we discretize the parameter space using a square grid of steps $\Delta x$ , $\Delta y$ , and express the integral as the discrete Berry phase $\gamma_{\partial F}$ of a loop approximating $\partial F$ , in the limit of an infinitesimally fine grid. Then, from Eq. (2.25) and the Stokes-type theorem in Eq. (2.11), we obtain

$$
\exp \left[ - i \oint_ {\partial \mathscr {F}} \mathbf {A} \cdot d \mathbf {R} \right] = \lim _ {\Delta x, \Delta y \rightarrow 0} e ^ {- i \sum_ {n m} F _ {n m}},\tag{2.26}
$$

where the $nm$ sum goes for the plaquettes forming the open surface $\mathcal{F}$ . Furthermore, let us take a gauge $|\Psi'(\mathbf{R})\rangle$ and the corresponding Berry connection $\mathbf{A}'$ that is smooth in the plaquette $nm$ ; this could be $|\Psi(\mathbf{R})\rangle$ and $\mathbf{A}$ if that was already smooth. Then, due to the gauge invariance of the Berry flux we have

$$
e ^ {- i F _ {n m}} = e ^ {- i F _ {n m} ^ {\prime}},\tag{2.27}
$$

where $F_{nm}^{\prime}$ is the Berry flux corresponding to the locally smooth gauge. Furthermore, in the limit of an infinitely fine grid it holds that

$$
\begin{array}{c} F _ {n m} ^ {\prime} = A _ {x} ^ {\prime} \left(x _ {n} + \frac {\Delta x}{2}, y _ {m}\right) \Delta x + A _ {y} ^ {\prime} \left(x _ {n + 1}, y _ {m} + \frac {\Delta y}{2}\right) \Delta y \\ - A _ {x} ^ {\prime} \left(x _ {n} + \frac {\Delta x}{2}, y _ {m + 1}\right) \Delta x - A _ {y} ^ {\prime} \left(x _ {n}, y _ {m} + \frac {\Delta y}{2}\right) \Delta y. \end{array}\tag{2.28}
$$

Taylor expansion of the Berry connection around $\mathbf{R}_{nm} = \left(x_{n} + \frac{\Delta x}{2},y_{n} + \frac{\Delta y}{2}\right)$ to first order yields

$$
F _ {n m} ^ {\prime} = \left[ \partial_ {x} A _ {y} ^ {\prime} (\mathbf {R} _ {n m}) - \partial_ {y} A _ {x} ^ {\prime} (\mathbf {R} _ {n m}) \right] \Delta x \Delta y.\tag{2.29}
$$

Thereby, with the definition of the Berry curvature as

$$
B = \lim _ {\Delta x, \Delta y \to 0} \frac {F _ {n m} ^ {\prime}}{\Delta x \Delta y},\tag{2.30}
$$

we obtain a quantity that is gauge invariant, as it is defined via the gauge invariant Berry flux, and is related to the Berry connection via

$$
B = \partial_ {x} A _ {y} ^ {\prime} (\mathbf {R} _ {n m}) - \partial_ {y} A _ {x} ^ {\prime} (\mathbf {R} _ {n m}).\tag{2.31}
$$

We can rephrase Eq. $(2.29)$ as follows: the Berry flux for the nm plaquette is expressed as the product of the Berry curvature on the plaquette and the surface area of the plaquette.

Substituting Eqs. (2.27) and (2.29) into Eq. (2.26) yields

$$
\exp \left[ - i \oint_ {\partial \mathcal {F}} \mathbf {A} \cdot d \mathbf {R} \right] = \exp \left[ - i \int_ {\mathcal {F}} B (x, y) d x d y \right],\tag{2.32}
$$

which is the continuum version of the result $(2.11)$ . Equation $(2.32)$ can also be rephrased as

$$
\gamma (\partial \mathcal {F}) = - \arg e ^ {- i \int_ {\mathcal {F}} B (x, y) d x d y}.\tag{2.33}
$$

## 2.2.3.3 A Special Case Where the Usual Stokes Theorem Works

A shortcut towards a stronger result than Eq. $(2.32)$ is offered in the special case when $|\Psi(\mathbf{R})\rangle$ is smooth on the open surface F. Then, a direct application of the two-dimensional Stokes theorem implies

$$
\oint_ {\partial \mathcal {F}} \mathbf {A} \cdot d \mathbf {R} = \int_ {\mathcal {F}} (\partial_ {x} A _ {y} - \partial_ {y} A _ {x}) d x d y = \int_ {\mathcal {F}} B d x d y\tag{2.34}
$$

Summarizing Eqs. $(2.32)$ and $(2.34)$ , we can say that line integral of the Berry connection equals the surface integral of the Berry curvature if the set of states $|\Psi(\mathbf{R})\rangle$ is smooth on F, but they might differ with an integer multiple of $2\pi$ otherwise.

## 2.2.3.4 The Case of the Three-Dimensional Parameter Space

Let us briefly discuss also the case of a three-dimensional parameter space. This will be particularly useful in the context of two-level systems. Starting with the case when the gauge $|\Psi(\mathbf{R})\rangle$ on the two-dimensional open surface F embedded in the three-dimensional parameter space is smooth in the neighborhood of F, we can directly apply the three-dimensional Stokes theorem to convert the line integral of A to the surface integral of the curl of A to obtain

$$
\oint_ {\partial \mathcal {F}} \mathbf {A} \cdot d \mathbf {R} = \int_ {\mathcal {F}} \mathbf {B} \cdot d \mathbf {S},\tag{2.35}
$$

where the Berry curvature is defined as the vector field $\mathbf{B}(\mathbf{R})$ via

$$
\mathbf {B} (\mathbf {R}) = \nabla_ {\mathbf {R}} \times \mathbf {A} (\mathbf {R}),\tag{2.36}
$$

which is gauge invariant as in the two-dimensional case. Even if $|\Psi (\mathbf{R})\rangle$ is not smooth on $\mathcal{F}$ , the relation

$$
\gamma (\partial \mathcal {F}) = - \arg e ^ {- i \oint_ {\partial \mathcal {F}} \mathbf {A} \cdot d \mathbf {R}} = - \arg e ^ {- i \int_ {\mathcal {F}} \mathbf {B} \cdot d \mathbf {S}}\tag{2.37}
$$

holds, similarly to the two-dimensional result Eq. (2.32).

Note furthermore that the Berry phase $\gamma(\partial\mathcal{F})$ for a fixed boundary curve $\partial\mathcal{F}$ is not only gauge invariant, but also invariant against continuous deformations of the two-dimensional surface F embedded in three dimensions, as long as the Berry curvature is smooth everywhere along the way.

We also remark that although we used the three-dimensional notation here, but the above results can be generalized for any dimensionality of the parameter space.

The notation A and B for the Berry connection and Berry curvature suggest that they are much like the vector potential and the magnetic field. This is a useful analogy, for instance, $\nabla_{R}B = 0$ , from the definition (2.36). Nevertheless, it is not true that in every problem where the Berry curvature is nonzero, there is a physical magnetic field.

## 2.2.4 Chern Number

In the discrete case, we defined the Chern number as a sum of Berry fluxes for a square lattice living on a torus (or any other orientable closed surface). Here, we take a continuum parameter space that has the topology of a torus. The motivation is that certain physical parameter spaces in fact have this torus topology, and the corresponding Chern number does have physical significance. One example will be the Brillouin zone of a two-dimensional lattice representing a solid crystalline material, where the momentum vectors $(k_{x}, k_{y})$ , $(k_{x} + 2\pi, ky)$ , and $(k_{x}, k_{y} + 2\pi)$ are equivalent.

Quite naturally, in the continuum definition of the Chern number, the sum of Berry fluxes is replaced by the surface integral of the Berry curvature over the whole of the parameter space P,

$$
Q = - \frac {1}{2 \pi} \int_ {\mathcal {P}} B d x d y.\tag{2.38}
$$

As this can be interpreted as a continuum limit of the discrete Chern number, it inherits the properties of the latter: the continuum Chern number is a gauge invariant integer.

For future reference, let us lay down the notation to be used for calculating the Chern numbers of electronic energy bands in two-dimensional crystals. Consider a square lattice for simplicity, which has a square-shaped Brillouin zone as well. Our parameter space $\mathcal{P}$ is the two-dimensional Brillouin zone now, which has a torus topology as discussed above. The parameters are the Cartesian components $k_{x}, k_{y} \in [-\pi, \pi)$ of the momentum vector $\mathbf{k}$ . The electronic energy bands and the corresponding electron wavefunctions can be obtained from the bulk momentum-space Hamiltonian $\hat{H}(k_x, k_y)$ . The latter defines the Schrödinger equation

$$
\hat {H} (\mathbf {k}) \left| u _ {n} (\mathbf {k}) \right\rangle = E _ {n} (\mathbf {k}) \left| u _ {n} (\mathbf {k}) \right\rangle ,\tag{2.39}
$$

where $n = 1, 2, \ldots$ is the band index, which has as many possible values as the dimension of the Hilbert space of the internal degree of freedom of our lattice model. Note that defining the Berry connection, the Berry curvature and the Chern number for the nth band is possible only if that band is separated from other bands by energy gaps. The Berry connection of the nth band, in line with the general definition (2.20), reads

$$
A _ {j} ^ {(n)} (\mathbf {k}) = i \left\langle u _ {n} (\mathbf {k}) \right| \partial_ {k _ {j}} \left| u _ {n} (\mathbf {k}) \right\rangle , \quad \text { for } \quad j = x, y.\tag{2.40}
$$

The Chern number of the nth band, in correspondence with Eqs. (2.38) and (2.31), reads

$$
Q ^ {(n)} = - \frac {1}{2 \pi} \int_ {B Z} d k _ {x} d k _ {y} \left(\frac {\partial A _ {y} ^ {(n)}}{\partial k _ {x}} - \frac {\partial A _ {x} ^ {(n)}}{\partial k _ {y}}\right).\tag{2.41}
$$

Certain approximations of the band structure theory of electrons provide low-dimensional momentum-space Hamiltonians that can be diagonalized analytically, allowing for an analytical derivation of the Chern numbers of the electronic bands. More often, however, the electronic wave functions are obtained from numerical techniques on a finite-resolution grid of $(k_{x}, k_{y})$ points in the Brillouin zone. In that case, the Chern number of a chosen band can still be effectively evaluated using the discrete version of its definition (2.13).

The Chern number of a band of an insulator is a topological invariant in the following sense. One can imagine that the Hamiltonian describing the electrons on the lattice is deformed adiabatically, that is, continuously and with the energy gaps separating the nth band from the other bands kept open. In this case, the Berry curvature varies continuously, and therefore its integral for the Brillouin zone, which is the Chern number, cannot change as the value of the latter is restricted to integers. If the deformation of the crystal Hamiltonian is such that some energy gaps separating the nth band from a neighboring band is closed and reopened, that is, the deformation of the Hamiltonian is not adiabatic, then the Chern number might change. In this sense, the Chern number is a similar topological invariant for two-dimensional lattice models as the winding number is for the one-dimensional SSH model.

## 2.3 Berry Phase and Adiabatic Dynamics

In most physical situations of interest, the set of states whose geometric features (Berry phases) we are interested in are eigenstates of some Hamiltonian $\hat{H}$ . Take a physical system with D real parameters that are gathered into a formal vector $\mathbf{R} = (R_{1}, R_{2}, \ldots, R_{D})$ . The Hamiltonian is a smooth function $\hat{H}(\mathbf{R})$ of the parameters, at least in the region of interest. We order the eigenstates of the Hamiltonian according to the energies $E_{n}(\mathbf{R})$ ,

$$
\hat {H} (\mathbf {R}) \left| n (\mathbf {R}) \right\rangle = E _ {n} (\mathbf {R}) \left| n (\mathbf {R}) \right\rangle .\tag{2.42}
$$

We call the set of eigenstates $|n(\mathbf{R})\rangle$ the snapshot basis.

The definition of the snapshot basis involves gauge fixing, i.e., specifying the otherwise arbitrary phase prefactor for every $|n(\mathbf{R})\rangle$ . This can be a tricky issue: even in cases where a gauge exists where all elements of the snapshot basis are smooth functions of the parameters, this gauge might be very challenging to construct.

We consider the following problem. We assume that the system is initialized with $\mathbf{R} = \mathbf{R}_0$ and in an eigenstate $|n(\mathbf{R}_0)\rangle$ that is in the discrete part of the spectrum, i.e., $E_{n}(\mathbf{R}) - E_{n - 1}(\mathbf{R})$ and $E_{n + 1}(\mathbf{R}) - E_n(\mathbf{R})$ are nonzero. At time $t = 0$ we thus have

$$
\mathbf {R} (t = 0) = \mathbf {R} _ {0}; \quad | \psi (t = 0) \rangle = | n (\mathbf {R} _ {0}) \rangle .\tag{2.43}
$$

Now assume that during the time $t = 0 \rightarrow T$ the parameter vector R is slowly changed: R becomes $\mathbf{R}(t)$ , and the values of $\mathbf{R}(t)$ define a continuous directed curve C. Also, assume that $|n(\mathbf{R})\rangle$ is smooth along the curve C. The state of the system evolves according to the time-dependent Schrödinger equation:

$$
i \frac {d}{d t} \left| \psi (t) \right\rangle = \hat {H} (\mathbf {R} (t)) \left| \psi (t) \right\rangle .\tag{2.44}
$$

Further, assume that R is varied in such a way that at all times the energy gaps around the state $|n(\mathbf{R}(t))\rangle$ remain finite. We can then choose the rate of variation of $\mathbf{R}(t)$ along the path C to be slow enough compared to the frequencies corresponding to the energy gap, so the adiabatic approximation holds In that case, the system remains in the energy eigenstate $|n(\mathbf{R}(t))\rangle$ , only picking up a phase. We are now going to find this phase.

By virtue of the adiabatic approximation, we take as Ansatz

$$
\left| \psi (t) \right\rangle = e ^ {i \gamma_ {n} (t)} e ^ {- i \int_ {0} ^ {t} E _ {n} (\mathbf {R} (t ^ {\prime})) d t ^ {\prime}} \left| n (\mathbf {R} (t)) \right\rangle .\tag{2.45}
$$

For better readability, in the following we often drop the t argument where this leads to no confusion. The time derivative of Eq. (2.45) reads

$$
i \frac {d}{d t} | \psi (t) \rangle = e ^ {i \gamma_ {n}} e ^ {- i \int_ {0} ^ {t} E _ {n} (\mathbf {R} (t ^ {\prime})) d t ^ {\prime}} \left(- \frac {d \gamma_ {n}}{d t} | n (\mathbf {R}) \rangle + E _ {n} (\mathbf {R}) | n (\mathbf {R}) \rangle + i \left| \frac {d}{d t} n (\mathbf {R}) \right\rangle\right)\tag{2.46}
$$

To show what we mean by $\left|\frac{d}{dt}n(\mathbf{R}(t))\right\rangle$ , we write it out explicitly in terms of a fixed basis, that of the eigenstates at $R = R_{0}$ :

$$
\left| n (\mathbf {R}) \right\rangle = \sum_ {m} c _ {m} (\mathbf {R}) \left| m (\mathbf {R} _ {0}) \right\rangle ;\tag{2.47}
$$

$$
\left| \frac {d}{d t} n (\mathbf {R} (t)) \right\rangle = \frac {d \mathbf {R}}{d t} \cdot | \nabla_ {\mathbf {R}} n (\mathbf {R}) \rangle = \frac {d \mathbf {R}}{d t} \sum_ {m} \nabla_ {\mathbf {R}} c _ {m} (\mathbf {R}) \left| m (\mathbf {R} _ {0}) \right\rangle .\tag{2.48}
$$

We insert the Ansatz $(2.45)$ into the right hand side of the Schrödinger equation $(2.44)$ , use the snapshot eigenvalue relation $(2.42)$ , simplify and reorder the Schrödinger equation, and obtain

$$
- \frac {d \gamma_ {n}}{d t} | n (\mathbf {R}) \rangle + i \left| \frac {d}{d t} n (\mathbf {R}) \right\rangle = 0.\tag{2.49}
$$

Multiplying from the left by $\langle n(\mathbf{R})|$ , and using Eq. (2.48), we obtain

$$
\frac {d}{d t} \gamma_ {n} (t) = i \left\langle n (\mathbf {R} (t)) \mid \frac {d}{d t} n (\mathbf {R} (t)) \right\rangle = \frac {d \mathbf {R}}{d t} i \left\langle n (\mathbf {R}) \mid \nabla_ {\mathbf {R}} n (\mathbf {R}) \right\rangle .\tag{2.50}
$$

We have found that for the directed curve $\mathcal{C}$ in parameter space, traced out by $\mathbf{R}(t)$ , there is an adiabatic phase $\gamma_{n}(\mathcal{C})$ , which reads

$$
\gamma_ {n} (\mathcal {C}) = \int_ {\mathcal {C}} i \left\langle n (\mathbf {R}) \mid \nabla_ {\mathbf {R}} n (\mathbf {R}) \right\rangle d \mathbf {R}.\tag{2.51}
$$

A related result is obtained after a similar derivation, if the parameter space of the R points is omitted and the snapshot basis $|n(t)\rangle$ is parametrized directly by the time variable. Then, the adiabatic phase is

$$
\gamma_ {n} (t) = \int_ {0} ^ {t} i \left\langle n (t ^ {\prime}) \mid \partial_ {t ^ {\prime}} n (t ^ {\prime}) \right\rangle d t ^ {\prime}.\tag{2.52}
$$

Equation (2.51) allows us to formulate the key message of this section as the following. Consider the case of an adiabatic and cyclic change of the Hamiltonian, that is, when the curve C is closed, implying $\mathbf{R}(T) = \mathbf{R}_{0}$ . In this case, the adiabatic phase reads

$$
\gamma_ {n} (\mathcal {C}) = \oint_ {\mathcal {C}} i \left\langle n (\mathbf {R}) \mid \nabla_ {\mathbf {R}} n (\mathbf {R}) \right\rangle d \mathbf {R}.\tag{2.53}
$$

Therefore, the adiabatic phase picked up by the state during a cyclic adiabatic change of the Hamiltonian is equivalent to the Berry phase corresponding to the closed oriented curve representing the Hamiltonian's path in the parameter space.

Two further remarks are in order. First, on the face of it, our derivation seems to do too much. It seems that we have produced an exact solution of the Schrödinger equation. Where did we use the adiabatic approximation? In fact, Eq. (2.50) does not imply Eq. (2.49). For the more complete derivation, showing how the nonadiabatic terms appear, see [15].

The second remark concerns the measurability of the Berry phase. The usual way to experimentally detect phases is by an interferometric setup. This means coherently splitting the wavefunction of the system into two parts, taking them through two adiabatic trips in parameter space, via $\mathbf{R}(t)$ and $\mathbf{R}'(t)$ , and bringing the parts back together. The interference only comes from the overlap between the states: it is maximal if $|n(\mathbf{R}(T))\rangle = |n(\mathbf{R}'(T))\rangle$ , which is typically ensured if $\mathbf{R}(T) = \mathbf{R}'(T)$ . The difference in the adiabatic phases $\gamma_{n}$ and $\gamma_{n}'$ is the adiabatic phase associated with the closed loop C, which is the path obtained by going forward along $t = 0 \rightarrow T : \mathbf{R}(t)$ , then coming back along $t = T \rightarrow 0 : \mathbf{R}'(t)$ .

## 2.4 Berry's Formulas for the Berry Curvature

Berry provided [6] two practical formulas for the Berry curvature. Here we present them in a form corresponding to a three-dimensional parameter space. To obtain the two-dimensional case, where the Berry curvature $B$ is a scalar, one can identify the latter with the component $B_{z}$ of the three-dimensional case treated below; for generalization to higher than 3 dimensions, see the discussion in Berry's paper [6]. First,

$$
B _ {j} = - \mathrm{Im} \epsilon_ {j k l} \partial_ {k} \left\langle n \mid \partial_ {l} n \right\rangle = - \mathrm{Im} \epsilon_ {j k l} \left\langle \partial_ {k} n \mid \partial_ {l} n \right\rangle + 0,\tag{2.54}
$$

where the second term is 0 because $\partial_{k}\partial_{l}=\partial_{l}\partial_{k}$ but $\epsilon_{jkl}=-\epsilon_{jlk}$ .

To obtain Berry's second formula, inserting a resolution of identity in the snapshot basis in the above equation, we obtain

$$
\mathbf {B} ^ {(n)} = - \operatorname{Im} \sum_ {n ^ {\prime} \neq n} \left\langle \nabla n \mid n ^ {\prime} \right\rangle \times \left\langle n ^ {\prime} \mid \nabla n \right\rangle ,\tag{2.55}
$$

where the parameter set R is suppressed for brevity. The term with $n' = n$ is omitted from the sum, as it is zero, since because of the conservation of the norm, $\langle\nabla n \mid n\rangle = -\langle n \mid \nabla n\rangle$ . To calculate $\langle n' \mid \nabla n\rangle$ , start from the definition of the eigenstate $|n\rangle$ , act on both sides with $\nabla$ , and then project unto $|n'\rangle$ :

$$
\hat {H} \left| n \right> = E _ {n} \left| n \right > ;\tag{2.56}
$$

$$
(\nabla \hat {H}) | n \rangle + \hat {H} | \nabla n \rangle = (\nabla E _ {n}) | n \rangle + E _ {n} | \nabla n \rangle ;\tag{2.57}
$$

$$
\left\langle n ^ {\prime} \right| \nabla \hat {H} | n \rangle + \left\langle n ^ {\prime} \right| \hat {H} | \nabla n \rangle = 0 + E _ {n} \left\langle n ^ {\prime} \mid \nabla n \right\rangle .\tag{2.58}
$$

Act with $\hat{H}$ towards the left in Eq. (2.58), rearrange, substitute into (2.55), and you obtain the second form of the Berry curvature, which is manifestly gauge invariant:

$$
\mathbf {B} ^ {(n)} = - \operatorname{Im} \sum_ {n ^ {\prime} \neq n} \frac {\langle n | \nabla \hat {H} | n ^ {\prime} \rangle \times \langle n ^ {\prime} | \nabla \hat {H} | n \rangle}{(E _ {n} - E _ {n ^ {\prime}}) ^ {2}}.\tag{2.59}
$$

This shows that the monopole sources of the Berry curvature, if they exist, are the points of degeneracy.

A direct consequence of Eq. (2.59), is that the sum of the Berry curvatures of all eigenstates of a Hamiltonian is zero. If all the spectrum of $\hat{H} (\mathbf{R})$ is discrete along a closed curve $\mathcal{C}$ , then one can add up the Berry phases of all the energy eigenstates.

$$
\begin{array}{l} \sum_ {n} \mathbf {B} ^ {(n)} = - \operatorname{Im} \sum_ {n} \sum_ {n ^ {\prime} \neq n} \frac {\langle n | \nabla_ {\mathbf {R}} \hat {H} | n ^ {\prime} \rangle \times \langle n ^ {\prime} | \nabla_ {\mathbf {R}} \hat {H} | n \rangle}{(E _ {n} - E _ {n ^ {\prime}}) ^ {2}} \\ = - \operatorname{Im} \sum_ {n} \sum_ {n ^ {\prime} <   n} \frac {1}{(E _ {n} - E _ {n ^ {\prime}}) ^ {2}} \Big (\langle n | \nabla_ {\mathbf {R}} \hat {H} | n ^ {\prime} \rangle \times \langle n ^ {\prime} | \nabla_ {\mathbf {R}} \hat {H} | n \rangle \\ + \left\langle n ^ {\prime} \right| \nabla_ {\mathbf {R}} \hat {H} | n \rangle \times \langle n | \nabla_ {\mathbf {R}} \hat {H} | n ^ {\prime} \rangle \Big) = 0. \end{array}\tag{2.60}
$$

The last equation holds because $a \times b = -b \times a$ for any two vectors a, b.

## 2.5 Example: The Two-Level System

So far, most of the discussion on the Berry phase and the related concepts have been kept rather general. In this section, we illustrate these concepts via the simplest nontrivial example, that is, the two-level system.

## 2.5.1 No Continuous Global Gauge

Consider a Hamiltonian describing a two-level system:

$$
\hat {H} (\mathbf {d}) = d _ {x} \hat {\sigma} _ {x} + d _ {y} \hat {\sigma} _ {y} + d _ {z} \hat {\sigma} _ {z} = \mathbf {d} \cdot \hat {\boldsymbol {\sigma}},\tag{2.61}
$$

with $\mathbf{d} = (d_x, d_y, d_z) \in \mathbb{R}^3 \setminus \{0\}$ . Here, the vector d plays the role of the parameter R in of preceding sections, and the parameter space is the punctured three-dimensional Euclidean space $R^3 \setminus \{0\}$ , to avoid the degenerate case of the energy spectrum. Note the absence of a term proportional to $\sigma_0$ : this would play no role in adiabatic phases. Because of the anticommutation relations of the Pauli matrices, the Hamiltonian above squares to a multiple of the identity operator, $\hat{H}(\mathbf{d})^2 = \mathbf{d}^2 \sigma_0$ . Thus, the eigenvalues of $\hat{H}(\mathbf{d})$ have to have absolute value $|d|$ .

A practical graphical representation of $\hat{H} (\mathbf{d})$ is the Bloch sphere, shown in Fig. 2.2. The spherical angles $\theta \in [0,\pi)$ and $\varphi \in [0,2\pi)$ are defined as

$$
\cos \theta = \frac {d _ {z}}{| \mathbf {d} |};
$$

$$
e ^ {i \varphi} = \frac {d _ {x} + i d _ {y}}{\sqrt {d _ {x} ^ {2} + d _ {y} ^ {2}}}.\tag{2.62}
$$

![](images/8efdce521d9d6298806ab8bc6e4cd8ddd6edbe0116718a30b6c0c26e63718fc2.jpg)
Fig. 2.2 The Bloch sphere. A generic traceless gapped two-level Hamiltonian is a linear combination of Pauli matrices, $\hat{H}(\mathbf{d}) = \mathbf{d} \cdot \hat{\sigma}$ . This can be identified with a point in $\mathbb{R}^3 \backslash \{0\}$ . The eigenenergies are given by the distance of the point from the origin, the eigenstates depend only on the direction of the vector $\mathbf{d}$ , i.e., on the angles $\theta$ and $\varphi$ , as defined in subfigure (a) and in Eq. (2.62) The Berry phase of a closed curve $\mathcal{C}$ is half the area enclosed by the curve when it is projected onto the surface of the Bloch sphere

We denote the two eigenstates of the Hamiltonian $\hat{H} (\mathbf{d})$ by $| + _{\mathbf{d}}\rangle$ and $|-_{\mathbf{d}}\rangle$ , with

$$
\hat {H} (\mathbf {d}) \left| \pm_ {\mathbf {d}} \right\rangle = \pm | \mathbf {d} | \left| \pm_ {\mathbf {d}} \right\rangle .\tag{2.63}
$$

These eigenstates depend on the direction of the 3-dimensional vector d, but not on its length. The eigenstate with $E = +|d|$ of the corresponding Hamiltonian is:

$$
| + _ {\mathbf {d}} \rangle = e ^ {i \alpha (\theta , \varphi)} \binom{e ^ {- i \varphi / 2} \cos \theta / 2}{e ^ {i \varphi / 2} \sin \theta / 2},\tag{2.64}
$$

while the eigenstate with $E = -|d|$ is $|-d\rangle = e^{i\beta(d)} | + _{d}\rangle$ . The choice of the phase factors $\alpha$ and $\beta$ above corresponds to fixing a gauge. We will now review a few gauge choices.

Consider fixing $\alpha(\theta,\varphi)=0$ for all $\theta,\varphi$ . This is a very symmetric choice, in this way in formula (2.64), we find $\theta/2$ and $\varphi/2$ . There is problem, however, as you can see if you consider a full circle in parameter space: at any fixed value of $\theta$ , let $\varphi=0\to2\pi$ . We should come back to the same Hilbert space vector, and we do, but we also pick up a phase of $\pi$ . We can either say that this choice of gauge led to a discontinuity at $\varphi=0$ , or that our representation is not single-valued. We now look at some attempts at fixing these problems, to find a gauge that is both continuous and single valued.

As a first attempt, let us fix $\alpha = \varphi / 2$ ; denoting this gauge by subscript $S$ , we have

$$
| + _ {\mathbf {d}} \rangle_ {S} = \binom{\cos \theta / 2}{e ^ {i \varphi} \sin \theta / 2}.\tag{2.65}
$$

The phase prefactor now gives an additional factor of -1 as we make the circle in $\varphi$ at fixed $\theta$ , and so it seems we have a continuous, single valued representation. There are two tricky points, however: the North Pole, $\theta = 0$ , and the South Pole, $\theta = \pi$ . At the North Pole, $|(0, 0, 1)\rangle_{S} = (1, 0)$ no problems. This gauge is problematic at the South Pole, however (which explains the choice of subscript): there, $|(0, 0, -1)\rangle_{S} = (0, e^{i\varphi})$ , the value of the wavefunction depends on which direction we approach the South Pole from.

We can try to solve the problem at the South Pole by choosing $\alpha = -\varphi / 2$ , which gives us

$$
| + \mathbf {d} \rangle_ {N} = \binom{e ^ {- i \varphi} \cos \theta / 2}{\sin \theta / 2}.\tag{2.66}
$$

As you can probably already see, this representation runs into trouble at the North Pole: $|(0,0,1)\rangle_N = (e^{-i\varphi},0)$ .

We can try to overcome the problems at the poles by taking linear combinations of $|+\mathbf{d}\rangle_{S}$ and $|+\mathbf{d}\rangle_{N}$ , with prefactors that vanish at the South and North Poles, respectively. A family of options is:

$$
\left| + \mathbf {d} \right\rangle_ {\chi} = e ^ {i \chi} \sin \frac {\theta}{2} \left| + \mathbf {d} \right\rangle_ {S} + \cos \frac {\theta}{2} \left| + \mathbf {d} \right\rangle_ {N}\tag{2.67}
$$

$$
= \binom{\cos \frac {\theta}{2} (\cos \frac {\theta}{2} + \sin \frac {\theta}{2} e ^ {i \chi} e ^ {- i \varphi})}{\sin \frac {\theta}{2} e ^ {i \varphi} (\cos \frac {\theta}{2} + \sin \frac {\theta}{2} e ^ {i \chi} e ^ {- i \varphi})}.\tag{2.68}
$$

This is single valued everywhere, solves the problems at the Poles. However, it has its own problems: somewhere on the Equator, at $\theta = \pi/2$ , $\varphi = \chi \pm \pi$ , its norm disappears.

It is not all that surprising that we could not find a well-behaved gauge: there is none. By the end of this chapter, it should be clear, why.

## 2.5.2 Calculating the Berry Curvature and the Berry Phase

Consider the two-level system as defined in the previous section. Take a closed curve $\mathcal{C}$ in the parameter space $\mathbb{R}^3\backslash \{0\}$ . We are going to calculate the Berry phase $\gamma_{-}$ of the $|-\mathbf{d}\rangle$ eigenstate on this curve:

$$
\gamma_ {-} (\mathcal {C}) = \oint_ {\mathcal {C}} \mathbf {A} (\mathbf {d}) d \mathbf {d},\tag{2.69}
$$

with the Berry vector potential defined as

$$
\mathbf {A} (\mathbf {d}) = i \left\langle - _ {\mathbf {d}} \right| \nabla_ {\mathbf {d}} \left| - _ {\mathbf {d}} \right\rangle .\tag{2.70}
$$

The calculation becomes straightforward if we use the Berry curvature,

$$
\mathbf {B} (\mathbf {d}) = \nabla_ {\mathbf {d}} \times \mathbf {A} (\mathbf {d});\tag{2.71}
$$

$$
\gamma_ {-} (\mathcal {C}) = \int_ {\mathcal {S}} \mathbf {B} (\mathbf {d}) d \mathcal {S},\tag{2.72}
$$

where S is any surface whose boundary is the loop C. (Alternatively, it is a worthwhile exercise to calculate the Berry phase directly in a fixed gauge, e.g., one of the three gauges introduced above.)

Specifically, we make use of Berry's gauge invariant formulation (2.59) of the Berry curvature, derived in the last chapter. In the case of the generic two-level Hamiltonian (2.61), Eq. (2.59) gives

$$
\mathbf {B} ^ {\pm} (\mathbf {d}) = - \mathrm{Im} \frac {\langle \pm | \nabla_ {\mathbf {d}} \hat {H} | \mp \rangle \times \langle \mp | \nabla_ {\mathbf {d}} \hat {H} | \pm \rangle}{4 \mathbf {d} ^ {2}},\tag{2.73}
$$

with

$$
\nabla_ {\mathbf {d}} \hat {H} = \hat {\sigma}.\tag{2.74}
$$

To evaluate $(2.73)$ , we choose the quantization axis parallel to d, thus the eigenstates simply read

$$
| + _ {\mathbf {d}} \rangle = \binom{1}{0};
$$

$$
| - \mathbf {d} \rangle = \binom{0}{1}.\tag{2.75}
$$

The matrix elements can now be computed as

$$
\langle - | \hat {\sigma} _ {x} | + \rangle = (0 1) \left( \begin{array}{c c} 0 & 1 \\ 1 & 0 \end{array} \right) \binom{1}{0} = 1,\tag{2.76}
$$

and similarly,

$$
\langle - | \sigma_ {y} | + \rangle = i;\tag{2.77}
$$

$$
\langle - | \sigma_ {z} | + \rangle = 0.\tag{2.78}
$$

So the cross product of the vectors reads

$$
\langle - | \hat {\boldsymbol {\sigma}} | + \rangle \times \langle + | \hat {\boldsymbol {\sigma}} | - \rangle = \left( \begin{array}{c} 1 \\ i \\ 0 \end{array} \right) \times \left( \begin{array}{c} 1 \\ - i \\ 0 \end{array} \right) = \left( \begin{array}{c} 0 \\ 0 \\ 2 i \end{array} \right).\tag{2.79}
$$

This gives us for the Berry curvature,

$$
\mathbf {B} ^ {\pm} (\mathbf {d}) = \pm \frac {\mathbf {d}}{| \mathbf {d} |} \frac {1}{2 \mathbf {d} ^ {2}}.\tag{2.80}
$$

We can recognize in this the field of a pointlike monopole source in the origin. Alluding to the analog between the Berry curvature and the magnetic field of electrodynamics (both are derived from a “vector potential”) we can refer to this field, as a “magnetic monopole”. Note however that this monopole exists in the abstract space of the vectors d and not in real space.

The Berry phase of the closed loop C in parameter space, according to Eq. (2.72), is the flux of the monopole field through a surface S whose boundary is C. It is easy to convince yourself that this is half of the solid angle subtended by the curve,

$$
\gamma_ {-} (\mathcal {C}) = \frac {1}{2} \Omega_ {\mathcal {C}}.\tag{2.81}
$$

In other words, the Berry phase is half of the area enclosed by the image of C, projected onto the surface of the unit sphere, as illustrated in Fig. 2.2.

What about the Berry phase of the other energy eigenstate? From Eq. (2.73), the corresponding Berry curvature $B_{+}$ is obtained by inverting the order of the factors in the cross product: this flips the sign of the cross product. Therefore the Berry phases of the ground and excited state fulfil the relation

$$
\gamma_ {+} (\mathcal {C}) = - \gamma_ {-} (\mathcal {C}).\tag{2.82}
$$

One can see the same result on the Bloch sphere. Since $\langle+|-\rangle=0$ , the point corresponding to $|- \rangle$ is antipodal to the point corresponding to $|+\rangle$ . Therefore, the curve traced by the $|- \rangle$ on the Bloch sphere is the inverted image of the curve traced by $|+\rangle$ . These two curves have the same orientation, therefore the same area, with opposite signs.

## 2.5.3 Two-Band Lattice Models and Their Chern Numbers

The simplest case where a Chern number can arise is a two-band system. Consider a particle with two internal states, hopping on a two-dimensional lattice. The two internal states can be the spin of the conduction electron, but can also be some sublattice index of a spin polarized electron. In the translation invariant bulk, the wave vector $\mathbf{k} = (k_{x}, k_{y})$ is a good quantum number, and the Hamiltonian reads

$$
\hat {H} (\mathbf {k}) = \mathbf {d} (\mathbf {k}) \hat {\boldsymbol {\sigma}},\tag{2.83}
$$

with the function $\mathbf{d}(\mathbf{k})$ mapping each point of the Brillouin Zone to a three-dimensional vector. Since the Brillouin zone is a torus, the endpoints of the vectors $\mathbf{d}(\mathbf{k})$ map out a deformed torus in $R^{3}\backslash\{0\}$ . This torus is a directed surface: its inside can be painted red, its outside, blue.

The Chern number of $\left|-\right\rangle$ (using the notation of Sect. 1.2, of $\left|u_{1}(\mathbf{k})\right\rangle$ ) is the flux of $\mathbf{B}_{-}(\mathbf{d})$ through this torus. We have seen above that $\mathbf{B}_{-}(\mathbf{d})$ is the magnetic field of a monopole at the origin d=0. If the origin is on the inside of the torus, this flux is +1. If it is outside of the torus, it is 0. If the torus is turned inside out, and contains the origin, the flux is -1. The torus can also intersect itself, and therefore contain the origin any number of times.

One way to count the number of times the torus contains the origin is as follows. Take any line from the origin to infinity, and count the number of times it intersects the torus, with a +1 for intersecting from the inside, and a -1 for intersecting from the outside. The sum is independent of the shape of the line, as long as it goes all the way from the origin to infinity.

## Problems

## 2.1 Discrete Berry phase and Bloch vectors

Take an ordered set of three arbitrary, normalized states of a two-level system. Evaluate the corresponding discrete Berry phase. Each state is represented by a vector on the Bloch sphere. Show analytically that if two of the vectors coincide, then the discrete Berry phase vanishes.

## 2.2 Two-level system and the Berry connection

Consider the two-level system defined in Eq. (2.61), and describe the excited energy eigenstates using the gauge $|+\mathbf{d}\rangle_{S}$ defined in Eq. (2.65). Using this gauge, evaluate and visualize the corresponding Berry connection vector field $\mathbf{A}(\mathbf{d})$ . Is it well-defined in every point of the parameter space? Complete the same tasks using the gauge $|+\mathbf{d}\rangle_{N}$ defined in Eq. (2.66).

## 2.3 Massive Dirac Hamiltonian

Consider the two-dimensional massive Dirac Hamiltonian $\hat{H}(k_x, k_y) = m\hat{\sigma}_z + k_x\hat{\sigma}_x + k_y\hat{\sigma}_y$ , where $m \in \mathbb{R}$ is a constant and the parameter space is $\mathbb{R}^2 \ni (k_x, k_y)$ . (a) Take a circular loop with radius $\kappa$ in the parameter space, centered around the origin. Calculate the Berry phase associated to this loop and the ground-state manifold of the Hamiltonian: $\gamma_-(m, \kappa) = ?$ . (b) Calculate the Berry connection $B_-(k_x, k_y)$ for the ground-state manifold. (c) Integrate the Berry connection for the whole parameter space. How does the result depend on $m$ ?

## 2.4 Absence of a continuous global gauge

In Sect. 2.5.1, we have shown example gauges for the two-level system that were not globally smooth on the parameter space. Prove that such globally smooth gauge does not exist.

## 2.5 Chern number of two-band models

Consider a two-band lattice model with the Hamiltonian $\hat{H}(\mathbf{k}) = \mathbf{d}(\mathbf{k}) \cdot \hat{\boldsymbol{\sigma}}$ . Express the Chern number of the lower-energy band in terms of $\mathbf{d}(\mathbf{k}) / |\mathbf{d}(\mathbf{k})|$ .

# Chapter 3 Polarization and Berry Phase

The bulk polarization of a band insulator is a tricky concept. Polarization of a neutral molecule is easily defined using the difference in centers of the negative and positive charges constituting the system. When we try to apply this simple concept to the periodic bulk of a band insulator (assuming for simplicity that the positive atom cores are immobile and localized), we meet complications. The center of the negative charges should be calculated from the electronic charge density in the fully occupied valence bands. However, all energy eigenstates in the valence band are delocalized over the bulk, and so the center of charge of each electron in such a state is ill defined. Nevertheless, insulators are polarizable, and respond to an external electric field by a rearrangement of charges, which corresponds to a (tiny) current in the bulk. Thus, there should be a way to define a bulk polarization.

In this chapter we show how a bulk polarization can be defined for band insulators using the so-called modern theory of polarization $[23, 27, 28]$ . The contribution of the electrons to the polarization is a property of the many-body electron state, a Slater determinant of the energy eigenstates from the fully occupied valence bands. The central idea is to rewrite the same Slater determinant using a different orthonormal basis, one composed of localized states, the so-called Wannier states. The contribution of each electron in a Wannier state to the center of charge can then be easily assessed, and then added up.

We discuss the simplest interesting case, that of a one-dimensional two-band insulator with one occupied and one empty band, and leave the multiband case for later. We show that the center of charge of the Wannier states can be identified with the Berry phase of the occupied band over the Brillouin zone, also known as the Zak phase $[38]$ .

For a more complete and very pedagogical introduction to the Berry phase in electron wavefunctions, we refer the reader to a set of lecture notes by Resta [26].

## 3.1 The Rice-Mele Model

The toy model we use in this chapter is the Rice-Mele model, obtained from the SSH model of Chap. 1 by adding an extra staggered onsite potential. The Hamiltonian for the Rice-Mele model on a chain of N unit cells reads

$$
\begin{array}{l} \hat {H} = v \sum_ {m = 1} ^ {N} \big (| m, B \rangle \langle m, A | + h. c. \big) + w \sum_ {m = 1} ^ {N - 1} \big (| m + 1, A \rangle \langle m, B | + h. c. \big) \\ \qquad \qquad \qquad + u \sum_ {m = 1} ^ {N} \big (| m, A \rangle \langle m, A | - | m, B \rangle \langle m, B | \big), \end{array}\tag{3.1}
$$

with the staggered onsite potential u, the intracell hopping amplitude v, and intercell hopping amplitude w all assumed to be real. The matrix of the Hamiltonian for the Rice-Mele model on a chain of N = 4 sites reads

$$
H = \left( \begin{array}{c c c c c c c c} u & v & 0 & 0 & 0 & 0 & 0 & 0 \\ v & - u & w & 0 & 0 & 0 & 0 & 0 \\ 0 & w & u & v & 0 & 0 & 0 & 0 \\ 0 & 0 & v & - u & w & 0 & 0 & 0 \\ 0 & 0 & 0 & w & u & v & 0 & 0 \\ 0 & 0 & 0 & 0 & v & - u & v & 0 \\ 0 & 0 & 0 & 0 & 0 & w & u & v \\ 0 & 0 & 0 & 0 & 0 & 0 & v & - u \end{array} \right).\tag{3.2}
$$

## 3.2 Wannier States in the Rice-Mele Model

The bulk energy eigenstates of a band insulator are delocalized over the whole system. We use as an example the bulk Hamiltonian of the Rice-Mele model, i.e., the model on a ring of N unit cells. As in the case of the SSH model, Sect. 1.2, the energy eigenstates are the plane wave Bloch states,

$$
\left| \Psi (k) \right\rangle = \left| k \right\rangle \otimes \left| u (k) \right\rangle ,\tag{3.3}
$$

with

$$
| k \rangle = \frac {1}{\sqrt {N}} \sum_ {m = 1} ^ {N} e ^ {i m k} | m \rangle , \quad \text {   for   } k \in \{\delta_ {k}, 2 \delta_ {k}, \dots , N \delta_ {k} \} \quad \text {   with   } \delta_ {k} = \frac {2 \pi}{N}.\tag{3.4}
$$

We omit the index 1 from the eigenstate for simplicity. The $|u(k)\rangle$ are eigenstates of the bulk momentum-space Hamiltonian,

$$
H (k) = \left( \begin{array}{c c} u & v + w e ^ {- i k} \\ v + w e ^ {i k} & - u \end{array} \right),\tag{3.5}
$$

with eigenvalue $E(k)$ .

The Bloch states $|\Psi(k)\rangle$ are spread over the whole chain. They span the occupied subspace, defined by the projector

$$
\hat {P} = \sum_ {k \in B Z} | \Psi (k) \rangle \langle \Psi (k) |.\tag{3.6}
$$

The phase of each Bloch eigenstate $|\Psi(k)\rangle$ can be set at will. A change of these phases, a gauge transformation,

$$
| u (k) \rangle \rightarrow e ^ {i \alpha (k)} | u (k) \rangle ; \qquad | \Psi (k) \rangle \rightarrow e ^ {i \alpha (k)} | \Psi (k) \rangle ,\tag{3.7}
$$

gives an equally good set of Bloch states, with an arbitrary set of phases $\alpha(k) \in \mathbb{R}$ for $k = \delta_{k}, 2\delta_{k}, \ldots, 2\pi$ . Using this freedom it is in principle possible to ensure that in the thermodynamic limit of $N \to \infty$ , the components of $|\Psi(k)\rangle$ are smooth, continuous functions of k. However, this gauge might not be easy to obtain by numerical methods. We therefore prefer, if possible, to work with gauge-independent quantities, like the projector to the occupied subspace defined in Eq. (3.6).

## 3.2.1 Defining Properties of Wannier States

The Wannier states $|w(j)\rangle \in \mathcal{H}_{\mathrm{external}} \otimes \mathcal{H}_{\mathrm{internal}}$ , with $j = 1, \dots, N$ , are defined by the following properties:

$$
\left\langle w (j ^ {\prime}) \mid w (j) \right\rangle = \delta_ {j ^ {\prime} j}
$$

Orthonormal set

$$
\sum_ {j = 1} ^ {N} | w (j) \rangle \langle w (j) | = \hat {P}\tag{3.8a}
$$

Span the occupied subspace

(3.8b)

$$
\langle m + 1 \mid w (j + 1) \rangle = \langle m \mid w (j) \rangle \quad \text { Related   by   translation }\tag{3.8c}
$$

$$
\lim _ {N \rightarrow \infty} \left\langle \right. w (N / 2) \left. \right| (\hat {x} - N / 2) ^ {2} | w (N / 2) \rangle <   \infty \quad \text { Localization }\tag{3.8d}
$$

with the addition in Eq. $(3.8c)$ defined modulo N. Requirement $(3.8d)$ , that of localization, uses the position operator,

$$
\hat {x} = \sum_ {m = 1} ^ {N} m \left(| m, A \rangle \langle m, A | + | m, B \rangle \langle m, B |\right),\tag{3.9}
$$

and refers to a property of $|w(j)\rangle$ in the thermodynamic limit of $N\to \infty$ that is not easy to define precisely. In this one-dimensional case it can be turned into an even stricter requirement of exponential localization, $\langle w(j)|m\rangle \langle m|w(j)\rangle < e^{-|j - m| / \xi}$ for some finite localization length $\xi \in \mathbb{R}$ .

## 3.2.2 Wannier States Are Inverse Fourier Transforms of the Bloch Eigenstates

Because of Bloch's theorem, all energy eigenstates have a plane wave form not only in the canonical basis $|m, \alpha \rangle$ , with $\alpha = A, B$ , but in the Wannier basis as well,

$$
| \Psi (k) \rangle = e ^ {- i \alpha (k)} \frac {1}{\sqrt {N}} \sum_ {j = 1} ^ {N} e ^ {i k j} | w (j) \rangle ,\tag{3.10}
$$

with some phase factors $\alpha(k)$ . To convince yourself of this, consider the components of the right-hand-side in the basis of Bloch eigenstates. The right-hand-side is an eigenstate of the lattice translation operator S, with eigenvalue $e^{-ik}$ , and therefore, orthogonal to all of the Bloch eigenstates $|\Psi(k')\rangle$ with $k'\neq k$ . It is also orthogonal to all positive energy eigenstates, since it is in the occupied subspace. Thus, the only state left is $|\Psi(k)\rangle$ .

From Eq. (3.10), an inverse Fourier transformation gives us a practical Ansatz for Wannier states,

$$
| w (j) \rangle = \frac {1}{\sqrt {N}} \sum_ {k = \delta_ {k}} ^ {N \delta_ {k}} e ^ {- i j k} e ^ {i \alpha (k)} | \Psi (k) \rangle .\tag{3.11}
$$

There is still a large amount of freedom left by this form, since the gauge function $\alpha(k)$ is unconstrained. This freedom can be used to construct Wannier states as localized as possible. If, e.g., a smooth gauge is found, where in the $N \to \infty$ limit, the components of $e^{i\alpha(k)} |\Psi(k)\rangle$ are analytic functions of $k$ , we have exponential localization of the Wannier states due to properties of the Fourier transform. (More generally, if a discontinuity appears first in the $l$ th derivative of $|\Psi(k)\rangle$ , the components of the Wannier state $|w(j)\rangle$ will decay as $\langle m |w(j) \rangle \propto |m - j|^{-l-1}$ .)

## 3.2.3 Wannier Centers Can Be Identified with the Berry Phase

We first assume that we have found a continuous gauge. The center of the Wannier state $|w(0)\rangle$ can be calculated, using

$$
\begin{array}{l} \hat {x} | w (0) \rangle = \frac {1}{2 \pi} \int_ {- \pi} ^ {\pi} d k \sum_ {m} m e ^ {i k m} | m \rangle \otimes | u (k) \rangle \\ = - \frac {i}{2 \pi} \left[ \sum_ {m} e ^ {i k m} | m \rangle \otimes | u (k) \rangle \right] _ {- \pi} ^ {\pi} + \frac {i}{2 \pi} \int_ {- \pi} ^ {\pi} d k \sum_ {m} e ^ {i k m} | m \rangle \otimes | \partial_ {k} u (k) \rangle \\ = \frac {i}{2 \pi} \int_ {- \pi} ^ {\pi} d k \sum_ {m} e ^ {i k m} | m \rangle \otimes | \partial_ {k} u (k) \rangle . \end{array}\tag{3.12}
$$

We find that the center of the Wannier state $|w(j)\rangle$ is

$$
\langle w (j) | \hat {x} | w (j) \rangle = \frac {i}{2 \pi} \int_ {- \pi} ^ {\pi} d k \left\langle u (k) \mid \partial_ {k} u (k) \right\rangle + j.\tag{3.13}
$$

The second term in this equation shows that the centers of the Wannier states are equally spaced, at a distance of one unit cell from each other. The first term, which is the Berry phase (divided by $2\pi$ ) of the occupied band across the Brillouin zone, cf. Eq. (2.53), corresponds to a uniform displacement of each Wannier state by the same amount.

We define the bulk electric polarization to be the Berry phase of the occupied band across the Brillouin zone, the first term in Eq. (3.13),

$$
P _ {\text { e   l   e   c   t   r   i   c }} = \frac {i}{2 \pi} \int_ {- \pi} ^ {\pi} d k \left\langle u (k) \mid \partial_ {k} u (k) \right\rangle .\tag{3.14}
$$

Although the way we derived this above is intuitive, it remains to be shown that this is a consistent definition. From Chap. 2, it is clear that a gauge transformation can only change the bulk electric polarization by an integer. We will show explicitly in Chap. 5 that the change of this polarization in a quasi-adiabatic process correctly reproduces the bulk current.

## 3.2.4 Wannier States Using the Projected Position Operator

A numerically stable, gauge invariant way to find a tightly localized set of Wannier states is using the unitary position operator [28],

$$
\hat {X} = e ^ {i \delta_ {k} \hat {x}}.\tag{3.15}
$$

This operator is useful, because it fully respects the periodic boundary conditions of the ring. The eigensystem of $\hat{X}$ consists of eigenstates localized in cell m with eigenvalue $e^{i\delta_{k}m}$ . Thus, we can associate the expectation value of the position in state $|\Psi\rangle$ with the phase of the expectation value of $\hat{X}$ ,

$$
\langle x \rangle = \frac {N}{2 \pi} \arg \langle \Psi | \hat {X} | \Psi \rangle .\tag{3.16}
$$

The real part of the logarithm carries information about the degree of localization [1, 28].

In order to obtain the Wannier states, we restrict the unitary position operator to the filled bands, defining

$$
\hat {X} _ {P} = \hat {P} \hat {X} \hat {P}.\tag{3.17}
$$

We will show below that in the thermodynamic limit of $N \to \infty$ , the eigenstates of the projected position operator $\hat{X}_{P}$ form Wannier states.

To simplify the operator $\hat{X}_P$ , consider

$$
\begin{array}{l} \left\langle \Psi (k ^ {\prime}) \right| \hat {X} | \Psi (k) \rangle = \frac {1}{N} \sum_ {m ^ {\prime} = 1} ^ {N} e ^ {- i m ^ {\prime} k ^ {\prime}} \left\langle m ^ {\prime} \right| \otimes \left\langle u (k ^ {\prime}) \right| \sum_ {m = 1} ^ {N} e ^ {i \delta_ {k} m} e ^ {i m k} | m \rangle \otimes | u (k) \rangle \\ = \frac {1}{N} \left\langle u (k ^ {\prime}) \mid u (k) \right\rangle \sum_ {m = 0} ^ {N - 1} e ^ {i m (k + \delta_ {k} - k ^ {\prime})} = \delta_ {k + \delta_ {k}, k ^ {\prime}} \left\langle u (k + \delta_ {k}) \mid u (k) \right\rangle \end{array}\tag{3.18}
$$

where $\delta_{k + \delta_k,k'} = 1$ if $k' = k + \delta_k$ , and 0 otherwise. Using this, we have

$$
\begin{array}{c} \hat {X} _ {P} = \sum_ {k ^ {\prime} k} \left| \Psi (k ^ {\prime}) \right\rangle \left\langle \Psi (k ^ {\prime}) \right| \hat {X} | \Psi (k) \rangle \left\langle \Psi (k) \right| \\ = \sum_ {k} \left\langle u (k + \delta_ {k}) \mid u (k) \right\rangle \cdot \left| \Psi (k + \delta_ {k}) \right\rangle \left\langle \Psi (k) \right|. \end{array}\tag{3.19}
$$

We can find the eigenvalues of $\hat{X}_{P}$ using a direct consequence of Eq. (3.19), namely, that raising $\hat{X}_{P}$ to the Nth power gives an operator proportional to the unity in the occupied subspace,

$$
\left(\hat {X} _ {P}\right) ^ {N} = W \hat {P}.\tag{3.20}
$$

We will refer to the constant of proportionality, $W \in C$ , given by

$$
W = \left\langle u (2 \pi) \mid u (2 \pi - \delta_ {k}) \right\rangle \cdot \dots \cdot \left\langle u (2 \delta_ {k}) \mid u (\delta_ {k}) \right\rangle \left\langle u (\delta_ {k}) \mid u (2 \pi) \right\rangle ,\tag{3.21}
$$

as the Wilson loop. Note that W is very similar to a discrete Berry phase, apart from the fact that $|W| \leq 1$ (although $\lim_{N \to \infty} |W| = 1$ ). The spectrum of eigenvalues of $\hat{X}_{P}$ is therefore composed of the Nth roots of W,

$$
\lambda_ {n} = e ^ {i n \delta_ {k} + \log (W) / N}, \quad \text { with } \quad n = 1, \ldots , N; \quad \implies \quad \lambda_ {n} ^ {N} = W.\tag{3.22}
$$

These eigenvalues have the same magnitude $|\lambda_{n}| = \sqrt[N]{|W|} < 1$ , and phases in the interval $[0, 2\pi)$ , spaced by $\delta_{k}$ . Because $\langle w(j)|\hat{X}_{P}|w(j)\rangle = \langle w(j)|\hat{X}|w(j)\rangle$ , the magnitude tells us about the localization properties of the Wannier states, and the phases can be interpreted as position expectation values.

We now check whether eigenstates of $\hat{X}_{P}$ fulfil the properties required of Wannier states, Eq. (3.8). The relation $(\hat{X}_{P})^{N} = W\hat{P}$ above shows that eigenvectors of $\hat{X}_{P}$ span the occupied subspace. Let us introduce the translation operator $\hat{S} = \sum_{m=1}^{N} |(m \bmod N) + 1\rangle \langle m| \otimes I_{internal}$ . The eigenstates are related by translation, since

$$
\hat {S} ^ {\dagger} \hat {X} _ {P} \hat {S} = e ^ {i \delta_ {k}} \hat {X} _ {P};\tag{3.23}
$$

$$
\hat {X} _ {P} \left| \Psi \right> = \left| W \right| ^ {1 / N} e ^ {i \alpha} \left| \Psi \right > ;\tag{3.24}
$$

$$
\hat {X} _ {P} \hat {S} | \Psi \rangle = | W | ^ {1 / N} e ^ {i \alpha + \delta_ {k}} \hat {S} | \Psi \rangle .\tag{3.25}
$$

There is a problem with the orthogonality of the eigenstates though. We leave the proof of localization as an exercise for the reader.

The projected unitary position operator $\hat{X}_{P}$ is a normal operator only in the thermodynamic limit of $N \rightarrow \infty$ . For finite N, it is not normal, i.e., it does not commute with its adjoint, and as a result, its eigenstates do not form an orthonormal basis. This can be seen as a discretization error.

## 3.3 Inversion Symmetry and Polarization

For single-component, continuous-variable wavefunctions $\Psi(r)$ , inversion about the origin (also known as parity) has the effect $\Psi(r) \to \hat{\Pi}\Psi(r) = \Psi(-r)$ . Two important properties of the unitary operator $\hat{\Pi}$ representing inversion follow: $\hat{\Pi}^2 = 1$ , and $\hat{\Pi}e^{ikr} = e^{-ikr}$ . A Hamiltonian has inversion symmetry if $\hat{\Pi}\hat{H}\hat{\Pi}^\dagger = \hat{H}$ .

When generalizing the inversion operator to lattice models of solid state physics with internal degrees of freedom, we have to keep two things in mind.

First, in a finite sample, the edges are bound to break inversion symmetry about the origin (except for very fine-tuned sample preparation). We therefore only care about inversion symmetry in the bulk, and require that it take $|k\rangle \rightarrow |-k\rangle$ .

Second, each unit cell of the lattice models we consider also has its internal Hilbert space, which can be affected by inversion. This includes spin components (untouched by inversion) and orbital type variables (affected by inversion) as well.

In general, we represent the action of inversion on the internal Hilbert space by a unitary operator $\hat{\pi}$ independent of the unit cell.

The inversion operator is represented on the bulk Hamiltonian of a lattice model by an operator $\hat{\Pi}$ , which acts on $H_{internal}$ as $\hat{\pi}$ ,

$$
\hat {\varPi} \left| k \right\rangle \otimes \left| u \right\rangle = \left| - k \right\rangle \otimes \hat {\pi} \left| u \right\rangle ;\tag{3.26}
$$

$$
\hat {\pi} ^ {2} = \hat {\pi} ^ {\dagger} \hat {\pi} = \mathbb {I} _ {\mathrm{internal}}.\tag{3.27}
$$

The action of the inversion operator on the bulk momentum-space Hamiltonian can be read off using its definition,

$$
\hat {\Pi} \hat {H} (k) \hat {\Pi} ^ {- 1} = \hat {\Pi} \langle k | \hat {H} _ {\mathrm{bulk}} | k \rangle \hat {\Pi} ^ {- 1} = \langle - k | \hat {\Pi} \hat {H} _ {\mathrm{bulk}} \hat {\Pi} ^ {- 1} | - k \rangle = \hat {\pi} \hat {H} (- k) \hat {\pi} ^ {\dagger}.\tag{3.28}
$$

A lattice model has inversion symmetry in the bulk, if there exists a unitary and Hermitian $\hat{\pi}$ acting on the internal space, such that

$$
\hat {\pi} \hat {H} (- k) \hat {\pi} = \hat {H} (k).\tag{3.29}
$$

If all occupied bands can be adiabatically separated in energy, so we can focus on one band, with wavefunction $|u(k)\rangle$ , inversion symmetry has a simple consequence. The eigenstates at -k and k are related by

$$
\hat {H} (k) \left| u (k) \right\rangle = E (k) \left| u (k) \right\rangle \quad \Longrightarrow \hat {H} (- k) \hat {\pi} \left| u (k) \right\rangle = E (k) \hat {\pi} \left| u (k) \right\rangle ;\tag{3.30}
$$

$$
\implies | u (- k) \rangle = e ^ {i \phi (k)} \hat {\pi} | u (k) \rangle .\tag{3.31}
$$

For the wavenumbers k = 0 and $k = \pi$ , the so-called time-reversal invariant momenta, this says that they have states with a definite parity,

$$
\left| u (0) \right\rangle = p _ {0} \left| u (0) \right\rangle ; \qquad \left| u (\pi) \right\rangle = p _ {\pi} \left| u (\pi) \right\rangle ,\tag{3.32}
$$

$$
\text { with } \quad p _ {0} = \pm 1; \quad p _ {\pi} = \pm 1.\tag{3.33}
$$

## 3.3.1 Quantization of the Wilson Loop Due to Inversion Symmetry

We now rewrite the Wilson loop W of a band of an inversion-symmetric one-dimensional Hamiltonian, assuming we have a discretization into a number 2M of k-states, labeled by $j = -M + 1, \ldots, M$ , as

$$
| u _ {j} \rangle = \left\{ \begin{array}{l l} | u (2 \pi + j \delta_ {k}) \rangle & \text { if } j \leq 0; \\ | u (j \delta_ {k}) \rangle , & \text { otherwise }. \end{array} \right.\tag{3.34}
$$

We use Eq. (3.31), which takes the form

$$
\left| u _ {- j} \right\rangle = e ^ {i \phi_ {j}} \hat {\pi} \left| u _ {j} \right\rangle .\tag{3.35}
$$

The Wilson loop W of a band of an inversion symmetric one-dimensional insulator can only take on the values $\pm1$ . We show this, using M = 3 as an example,

$$
\begin{array}{r l} & W = \langle u _ {M} | u _ {2} \rangle \langle u _ {2} | u _ {1} \rangle \langle u _ {1} | u _ {0} \rangle \langle u _ {0} | u _ {- 1} \rangle \langle u _ {- 1} | u _ {- 2} \rangle \langle u _ {- 2} | u _ {M} \rangle \\ & \qquad = \langle u _ {M} | u _ {2} \rangle \langle u _ {2} | u _ {1} \rangle \langle u _ {1} | u _ {0} \rangle \langle u _ {0} | e ^ {i \phi_ {1}} \hat {\pi} | u _ {1} \rangle \\ & \qquad \langle u _ {1} | \hat {\pi} e ^ {- i \phi_ {1}} e ^ {i \phi_ {2}} \hat {\pi} | u _ {2} \rangle \langle u _ {2} | \hat {\pi} e ^ {- i \phi_ {2}} | u _ {M} \rangle \\ & \qquad = \langle u _ {1} | u _ {0} \rangle \langle u _ {0} | \hat {\pi} | u _ {1} \rangle \langle u _ {2} | \hat {\pi} | u _ {M} \rangle \langle u _ {M} | u _ {2} \rangle = p _ {0} p _ {\pi} \end{array}\tag{3.36}
$$

$$
\begin{array}{r l} W = \langle u _ {M} | u _ {M - 1} \rangle \dots \langle u _ {1} | u _ {0} \rangle \langle u _ {0} | u _ {- 1} \rangle \dots \langle u _ {- M + 1} | u _ {M} \rangle \\ & = \langle u _ {M} | u _ {M - 1} \rangle \dots \langle u _ {1} | u _ {0} \rangle \langle u _ {0} | e ^ {i \phi_ {1}} \hat {\pi} | u _ {1} \rangle \\ & \langle u _ {1} | \hat {\pi} e ^ {- i \phi_ {1}} e ^ {i \phi_ {2}} \hat {\pi} | u _ {2} \rangle \langle u _ {2} | \hat {\pi} e ^ {- i \phi_ {2}} e ^ {i \phi_ {3}} \hat {\pi} | u _ {3} \rangle \dots \langle u _ {M + 1} | \hat {\pi} e ^ {- i \phi_ {M + 1}} | u _ {M} \rangle \\ & = \langle u _ {1} | u _ {0} \rangle \langle u _ {0} | \hat {\pi} | u _ {1} \rangle \langle u _ {M - 1} | \hat {\pi} | u _ {M} \rangle \langle u _ {M} | u _ {M - 1} \rangle = \pm 1 \end{array}\tag{3.37}
$$

The statement about the Wilson loop can be translated to the bulk polarization, using Eq. $(3.14)$ . Each band of an inversion symmetric one-dimensional insulator contributes to the bulk polarization 0 or 1/2.

## Problems

## 3.1 Inversion symmetry of the SSH model

Does the SSH model have inversion symmetry? If it has, then provide the corresponding local operator acting in the internal Hilbert space.

## 3.2 Eigenstates of the projected position operator are localized

In Sect. 3.2.4, it is claimed that the eigenstates of the projected position operator $\hat{X}_P$ form a Wannier set. One necessary condition for that statement to be true is that the eigenstates are localized, see Eq. (3.8d). Prove this.

# Chapter 4 Adiabatic Charge Pumping, Rice-Mele Model

We now apply the Berry phase and the Chern number to show that by periodically and slowly changing the parameters of a one-dimensional solid, it is possible to pump particles in it. The number of particles (charge) pumped is an integer per cycle, that is given by a Chern number. Along the way we will introduce important concepts of edge state branches of the dispersion relation, and bulk-boundary correspondence. Since we are working towards understanding time-independent topological insulators, this chapter might seem like a detour. However, bulk-boundary correspondence of 2-dimensional Chern insulators, at the heart of the theory of topological insulators, is best understood via a mapping to an adiabatic charge pump.

The concrete system we use in this chapter is the simplest adiabatic charge pump, the time-dependent version of the Rice-Mele model,

$$
\begin{array}{l} \hat {H} (t) = v (t) \sum_ {m = 1} ^ {N} \left(| m, B \rangle \langle m, A | + h. c.\right) + w (t) \sum_ {m = 1} ^ {N - 1} \left(| m + 1, A \rangle \langle m, B | + h. c.\right) \\ \qquad + u (t) \sum_ {m = 1} ^ {N} \left(| m, A \rangle \langle m, A | - | m, B \rangle \langle m, B |\right), \end{array}\tag{4.1}
$$

with the staggered onsite potential u, intracell hopping amplitude v, and intercell hopping amplitude w all assumed to be real and periodic functions of time t. In this chapter, we are going to see how, by properly choosing the time sequences, we can ensure that particles are pumped along the chain.

## 4.1 Charge Pumping in a Control Freak Way

The most straightforward way to operate a charge pump in the Rice-Mele model is to make sure that the system falls apart at all times to disconnected dimers. This will happen if at any time either the intercell hopping amplitude w, or the intracell hopping amplitude v vanishes. We can then use the staggered onsite potential to nudge the lower energy eigenstate to the right. If during the whole cycle we keep a finite energy difference between the two eigenstates, we can do the cycle slowly enough to prevent excitation.

## 4.1.1 Adiabatic Shifting of Charge on a Dimer

As a first step towards the charge pumping protocol, consider a single dimer, i.e., N = 1. Using the adiabatic limit introduced in the last chapter, we can shift charge from one site to the other. The Hamiltonian reads

$$
\hat {H} (t) = u (t) \hat {\sigma} _ {z} + v (t) \hat {\sigma} _ {x},\tag{4.2}
$$

with no hopping allowed at the beginning and end of the cycle, at $t = 0$ , we have $(u,v) = (1,0)$ , we have and at $t = T$ , we have $(u,v) = (-1,0)$ .

We initialize the system in the ground state, which at time t = 0 corresponds to $|A\rangle$ , a particle on site A. Then we switch on the hopping, which allows the particle to spill over to site B, and once it has done that, we switch the hopping off. To ensure that the particle spills over, we raise the onsite potential at A and lower it at B. A practical choice is

$$
u (t) = \cos (\pi t / T);
$$

$$
v (t) = \sin (\pi t / T),\tag{4.3}
$$

whereby the energy gap is at any time 2. According to the adiabatic theorem, if $H(t)$ is varied slowly enough, we will have shifted the charge to $B$ at the end of the cycle.

## 4.1.2 Putting Together the Control Freak Sequence

Once we know how to shift a particle from $|m,A\rangle$ to $|m,B\rangle$ , we can use that to shift the particle further from $|m,B\rangle$ to $|m+1,A\rangle$ . For simplicity, we take a sequence constructed from linear ramps of the amplitudes, using the function $f : [0, 1) \to R$ :

$$
f (x) = \left\{ \begin{array}{l l} 8 x, & \text { if } x \leq 1 / 8 \\ 1, & \text { if } 1 / 8 \leq x <   3 / 8 \\ 1 - 8 (x - 3 / 8), & \text { if } 3 / 8 \leq x <   1 / 2 \\ 0, & \text { otherwise } \end{array} \right.\tag{4.4}
$$

One period of the pump sequence, for $0 \leq t < T$ , reads

$$
u (t) = f (t / T) - f (t / T + \frac {1}{2});\tag{4.5a}
$$

$$
v (t) = 2 f (t / T + \frac {1}{4});\tag{4.5b}
$$

$$
w (t) = f (t / T - \frac {1}{4}).\tag{4.5c}
$$

This period, shown in Fig. 4.1a, is assumed to then be repeated. Note that we shifted the beginning time of the sequence: now at times $t / T = n \in \mathbb{Z}$ , the Hamiltonian is the trivial SSH model, $t / T = n + 1 / 4$ , disconnected monomers, at times $t / T = n + 1 / 2$ , it is the nontrivial SSH model.

The time-dependent bulk momentum-space Hamiltonian reads

$$
\hat {H} (k, t) = \mathbf {d} (k, t) \hat {\sigma} = (v (t) + w (t) \cos k) \hat {\sigma} _ {x} + w (t) \sin k \hat {\sigma} _ {y} + u (t) \hat {\sigma} _ {z},\tag{4.6}
$$

which can be represented graphically as the path of the vector $\mathbf{d}(k,t)$ as the quasimomentum goes through the Brillouin zone, $k:0\to2\pi$ , for various fixed values of time t, as in Fig. 4.1b.

## 4.1.3 Visualizing the Motion of Energy Eigenstates

We can visualize the effects of the control freak pumps sequence in the Rice-Mele model by tracing the trajectories of the energy eigenstates. At any time t, each instantaneous energy eigenstate can be chosen confined to a single dimer: either on a single unit cell, or shared between two cells. In both cases, we can associate a position with the energy eigenstates: the expectation value of the position operator $\hat{x}$ defined as per Eq. (3.9).

The trajectories of energy eigenstates in the position-energy space, Fig. 4.2, show that the charge pump sequence works rather like a conveyor belt for the eigenstates. We engineered the sequence as a unitary operation that pushes all negative energy states in the bulk to the right at the rate of one unit cell per cycle (a current of one particle per cycle). These orthogonal states, one by one, are pushed into the right end region, which has only room for one energy eigenstate. Eigenstates cannot pile up in the right end region: if they did, this would violate unitarity of the time evolution operator $U(t) = \mathbb{T}e^{-i\int_{0}^{t}H(t')dt'}$ , where T stands for time ordering, since initially orthogonal states would acquire finite overlap. So, states pumped to the right edge have to go somewhere, and the only direction they can go is back towards the bulk. This on the other hand is only possible, if they acquire enough energy to be in the upper band, since all states in the lower band in the bulk are pushed towards the right. Moreover, in order to carry these states away from the right edge, and make room for those coming from the bulk, the pump sequence has to push upper band bulk states towards the left.

![](images/46547a8825ead47f8969c553ccf5f8eef1f7ed28f36b9afb32b35d548842d6cb.jpg)

(b)
![](images/219d5bb5e13a144a1b3f8dd2e9357f6abd78b5bf032eefe935d020e67526fa32.jpg)

![](images/5e1576b4711aa21722dc6f773173293288620ca944917cd2bba09a97c8180eef.jpg)

![](images/2c1893068e3b42194fb063a18570e8b91e7f1821b1c73058413986f1b4f5c873.jpg)
Fig. 4.1 The control freak pump sequence in the Rice-Mele model. The sequence is defined via Eqs. (4.5) and (4.6). (a) Time dependence of the hopping amplitudes v, w and the sublattice potential u. (b) The surface formed by the vector $\mathbf{d}(k,t)$ corresponding to the bulk momentum-space Hamiltonian. The topology of the surface is a torus, but its parts corresponding to $t \in [0,0.25]T$ and $t \in [0.75,1]T$ are infinitely thin and appear as a line due to the vanishing value of w in these time intervals. (c) Instantaneous spectrum of the Hamiltonian $\hat{H}(t)$ of an open chain of N = 10 sites. Red (blue) points represent states that are localized in the rightmost (leftmost) unit cells and have energies between -1 and 1

![](images/37b63c5056be5ba3416815fb61f490ba1fcb9a24e9b0202111a80e5732eebc77.jpg)
Fig. 4.2 An instantaneous energy eigenstate of the control freak pump sequence, as it is pumped through the system. At negative (positive) energy, it is pumped towards the right (left), upon reaching the right (left) end, it is pumped into the upper (lower) band

To summarize, the control freak pump sequence is characterized by three statements. The protocol

\- in the bulk, pushes all $E < 0$ eigenstates rightwards, by 1 unit cell per cycle,

\- at the right end, pushes 1 eigenstate per cycle with $E < 0$ to $E > 0$ ,

\- in the bulk, pushes all $E > 0$ eigenstates leftwards, by 1 unit cell per cycle.

If any one of these statements holds, the other two must also hold as a consequence.

## 4.1.4 Edge States in the Instantaneous Spectrum

We can see the charge pump at work indirectly—via its effect on the edge states—using the instantaneous spectrum, the eigenvalues of $\hat{H}(t)$ of the open chain. An example is shown for the control freak pump sequence of the Rice-Mele model on a chain of 10 unit cells (20 sites) in Fig. 4.1. Due to the special choice of the control freak sequence, the bulk consists of N-1-fold degenerate states (the bands are flat). More importantly, there is an energy gap separating the bands, which is open around E=0 at all times. However, there are branches of the spectrum crossing this energy gap, which must represent edge states.

To assign “right” or “left” labels to edge states in the instantaneous spectrum, it is necessary to examine the corresponding wavefunctions. In case of the control freak pump sequence, right (left) edge state wavefunctions are localized on the m = N (m = 1) unit cells, and the corresponding energy values are highlighted in green (red). The edge state branches in the dispersion in Fig. 4.1. clearly show that 1 state per cycle is pushed up in energy at the right edge.

## 4.2 Moving Away from the Control Freak Limit

We will now argue that the number of particles pumped by a cycle of a periodic adiabatic modulation of an insulating chain is an integer, even if the control freak attitude is relaxed. In the generic case, the energy eigenstates are delocalized over the whole bulk, and so we will need new tools to keep track of the charge pumping process. The robust quantization of charge pumping was shown by Thouless, who calculated the bulk current directly: we defer this calculation to the next chapter, and here argue using adiabatic deformations.

As an example for a generic periodically modulated insulator, we take the Rice-Mele model, but we relax the control freak attitude. We consider a smooth modulation sequence,

$$
u (t) = \sin (2 \pi t / T),\tag{4.7a}
$$

$$
v (t) = \overline {{v}} + \cos (2 \pi t / T),\tag{4.7b}
$$

$$
w (t) = 1,\tag{4.7c}
$$

where the sequence is fixed by choosing the average value of the intracell hopping, $\overline{v}$ . With $\overline{v}=1$ , this sequence can be obtained by an adiabatic deformation of the control freak sequence. We show the smooth pump sequence and its representation in the d space for $\overline{v}=1$ in Fig.4.3.

## 4.2.1 Edge States in the Instantaneous Spectrum

Consider the spectrum of the instantaneous energies on an open chain, with an example for N = 20 unit cells shown in Fig. 4.3. Since this charge sequence was obtained by adiabatic deformation of the control freak sequence above, each branch in the dispersion relation is deformed continuously from a branch in Fig. 4.1.

The edge states are no longer confined to a single unit cell, as in the control freak case. However, as long as their energy lies deep in the bulk band gap, they have wavefunctions that decay exponentially towards the bulk, and so they can be unambiguously assigned to the left or the right end. (In case of a degeneracy between edge states at the right and left end, we might find a wavefunction with components on both ends. In that case, however, restriction of that state to the left/right end results in two separate eigenstates, to a precision that is exponentially high in the bulk length). In Fig. 4.3 we used the same simple criterion as in Chap. 1 to define edge states:

$$
| \Psi \rangle \text {   is   on   the   right   edge   } \Leftrightarrow \sum_ {m = N - 1} ^ {N} \left(| \langle \Psi | m, A \rangle | ^ {2} + | \langle \Psi | m, B \rangle | ^ {2}\right) > 0. 6;\tag{4.8}
$$

![](images/3fe90fd3528b3385b5fcf24a65d2d011a8ac45b7349fbddeaf334c6ef30da026.jpg)

(b)
![](images/0ac700d54080042cdcb88adf62811414da491cc8db0c575100f408bba1dc8eff.jpg)

![](images/a8046a871ef00373b2733b19201a232aec6da7aa24f05f438d2bddea7ca03b5f.jpg)

![](images/0eff5a9d5d28f644f99f6253681cf7ef70c44a839f4d0dc2a8211c6333b323af.jpg)
Fig. 4.3 The smooth pump sequence of the Rice-Mele model for $\bar{v} = 1$ . The hopping amplitudes and the sublattice potential (a) are varied smoothly as a function of time. The vector $\mathbf{d}(k,t)$ corresponding to the bulk momentum-space Hamiltonian (b) traces out a torus in the 3-dimensional space. Instantaneous spectrum of the Hamiltonian $\hat{H}(t)$ on an open chain of $N = 10$ sites (c) reveals that during a cycle, one state crosses over to the upper band on the right edge, and one to the lower band on the left edge (dark red/light blue highlights energies of edge states, whose wavefunctions have than $60\%$ weight on the rightmost/leftmost 2 unit cells). The wavefunctions of the edge states (d,e) are exponentially localized to one edge and have support overwhelmingly on one sublattice each. In contrast a typical bulk state (f) has a delocalized wavefunction with support on both sublattices

$$
| \Psi \rangle \text {   is   on   the   left   edge   } \Leftrightarrow \sum_ {m = 1} ^ {2} \left(| \langle \Psi | m, A \rangle | ^ {2} + | \langle \Psi | m, B \rangle | ^ {2}\right) > 0. 6.\tag{4.9}
$$

As in the control freak case, there is a branch of energy eigenstates crossing over from E < 0 to E > 0 at the right edge, and from E > 0 to E < 0, at the left.

We can define the edge spectrum to consist of edge state branches of the dispersion relation, that are clearly assigned to the right end. More precisely, we take two limiting energies, $\varepsilon_{-}$ and $\varepsilon_{+}$ , deep in the bulk gap, and only consider energy eigenstates of the open chain with eigenvalues $E_{n}(t)$ between these limits, $\varepsilon_{-} < E_{n}(t) < \varepsilon_{+}$ , with eigenstates localized at the right edge. Each edge state branch can begin (1) at t = 0, as a continuation of another (or the same) edge state branch ending at t = T, or (2) at $E = \varepsilon_{-}$ , or (3) at $E = \varepsilon_{+}$ . Each edge state branch can end (1) at t = T, to then continue in another (or the same) edge state branch at t = 0, or (2) at $E = \varepsilon_{-}$ , or (3) at $E = \varepsilon_{+}$ . That is 9 possible types of edge state branches. Taking into account that the edge state spectrum, like the total spectrum, has to be periodic in t, the number of edge state branches entering the energy range $\varepsilon_{-} < E < \varepsilon_{+}$ during a cycle is equal to the number of branches leaving it.

## 4.2.2 The Net Number of Edge States Pumped in Energy Is a Topological Invariant

We now define an integer Q, that counts the number of edge states pumped up in energy across at the right edge. Although this quantity is not easily represented by a closed formula, it is straightforward to read it off from the dispersion relation of an open system. We restrict our attention to the neighbourhood of an energy $\varepsilon$ deep in the bulk gap around E = 0, such that at all points where $E_{n} = \varepsilon$ , the derivative $dE_{n}/dt$ does not vanish. Then every edge state energy branch entering this neighborhood crosses $E = \varepsilon$ either towards $E > \varepsilon$ or towards $E < \varepsilon$ . During one cycle, we define for the states at the right edge

$$
N _ {+} = \text { number   of   times } E = \varepsilon \text { is   crossed   from } E <   \varepsilon \text { to } E > \varepsilon ;\tag{4.10}
$$

$$
N _ {-} = \text { number   of   times } E = \varepsilon \text { is   crossed   from } E > \varepsilon \text { to } E <   \varepsilon ;\tag{4.11}
$$

$$
Q = N _ {+} - N _ {-} = \text { net   number   of   edge   states   pumped   up   in   energy }.\tag{4.12}
$$

Note that within the gap, Q is independent of the choice of $\varepsilon$ . If we found a value $Q_{0}$ at $E = \varepsilon_{0}$ , but a different $Q_{1} \neq Q_{0}$ at $E = \varepsilon_{1} > \varepsilon_{0}$ , this would require a net number $Q_{0} - Q_{1}$ of edge state branches at the right edge to enter the energy region $\varepsilon_{0} < E < \varepsilon_{1}$ during a cycle but never exit it. Since both $\varepsilon_{0}$ and $\varepsilon_{1}$ are deep in the bulk gap, away from the bulk bands, this is not possible.

The net number of edge states pumped up inside the gap on the right edge, Q, is a topological invariant: its value cannot change under continuous deformations of the Hamiltonian $H(t)$ that preserve the bulk gap. This so-called topological protection is straightforward to prove, by considering processes that might change this number. We do this using Fig. 4.4.

![](images/d925fc4f994f2509f115fff14fbeaf1eb4965f19e7ee9028775ac0dc2a0d9a79.jpg)

(b)
![](images/8d42e29edcbc980297b506bb57320daad99ceb983ed69b170a4ee307f9deaf3a.jpg)

(c)
![](images/cf8433248789fa6f0ec981e5454ca559b08acdc9b226f2ae9a1c428baf67116d.jpg)

![](images/2faedefa1222b11c6e87ad388ae95afbd9f8d43af76237af2c42c56c05e61fa2.jpg)

![](images/ab41c908c5cc3ed9e880a3f06509053d85511820bb67c2196880d52e4db6ffe7.jpg)

![](images/2ba6bd1cb322ec90dde1d47ac0c4e0faed17476be1c369282accaeccdbce04ce.jpg)

![](images/d3f1efaa8c40cb932c4d01cfab683d045fca80b26842092f5bd32b750e8a5698.jpg)

![](images/bdcd4657da672d2feac10bb575f71167d15fd7dc5b767e64768ab5c5ea40f723.jpg)

![](images/eb1fe29a8e7b8ca08d2a87621a1e2f4064f57062a49b4c5ec42d871e0434bf09.jpg)
Fig. 4.4 Adiabatic deformations of dispersion relations of edge states on one edge, in an energy window that is deep inside the bulk gap. Starting from a system with 3 copropagating edge states (a), an edge state's dispersion relation can develop a "bump", (b)–(c). This can change the number of edge states at a given energy (intersections of the branches with the horizontal line corresponding to the energy), but always by introducing new edge states pairwise, with opposite directions of propagation. Thus the signed sum of edge states remains unchanged. Alternatively, two edge states can develop a crossing, that because of possible coupling between the edge states turns into an avoided crossing (d), (f). This cannot open a gap between branches of the dispersion relation (e), as this would mean that the branches become multivalued functions of the wavenumber $k_{x}$ (indicating a discontinuity in $E(k_{x})$ , which is not possible for a system with short-range hoppings). Therefore, the signed sum of edge states is also unchanged by this process. One might think the signed sum of edge states can change if an edge state's direction of propagation changes under the adiabatic deformation, as in (g). However, this is also not possible, as it would also make a branch of the dispersion relation multivalued. Deformation of the Hamiltonian can also form a new edge state dispersion branch, as in (a)–(h)–(i), but because of periodic boundary conditions along $k_{x}$ , this cannot change the signed sum of the number of edge states

The number of times edge state branches intersect E = 0 can change because new intersection points appear. These can form because an edge state branch is deformed, and as a result, it gradually develops a “bump”, local maximum, and the local maximum gets displaced from E < 0 to E > 0. For a schematic example, see Fig. 4.4a–c. Alternatively, the dispersion relation branch of the edge state can also form a local minimum, gradually displaced from E > 0 to E < 0. In both cases, the number of intersections of the edge band with the E = 0 line grows by 2, but the two new intersections must have opposite pump directions. Therefore, both $N_{+}$ and $N_{-}$ increase by 1, but their difference, $Q = N_{+} - N_{-}$ , stays the same.

New intersection points can also arise because a new edge state branch forms. As long as the bulk gap stays open, though, this new edge state band has to be a deformed version of one of the bulk bands, as shown in Fig. 4.4a–h–i. Because the periodic boundary conditions must hold in the Brillouin zone, the dispersion relation of the new edge state has to come from a bulk band and go back to the same bulk band, or it can be detached from the bulk band, and be entirely inside the gap. In both cases, the above argument applies, and it has to intersect the E = 0 line an even number of times, with no change of Q.

The number of times edge state branches intersect E = 0 can also decrease if two edge state branches develop an energy gap. However, to open an energy gap, the edge states have to be pumped in opposite directions. For states pumped in the same direction, energy crossing between them can become an avoided crossing, but no gap can be opened, as this would violate the single-valuedness of a dispersion relation branch, as illustrated in Fig. 4.4d–f. This same argument shows why it is not possible for an edge state to change its direction of propagation under an adiabatic deformation without developing a local maximum or minimum (which cases we already considered above). As shown in Fig. 4.4g, this would entail that at some stage during the deformation the edge state branch was not single valued.

## 4.3 Tracking the Charges with Wannier States

Electrons in a solid are often described via Bloch states delocalized over the whole lattice. As we have seen in Sect. 3.2 though, one can represent a certain energy band with a set of Wannier states, which inherit the spatial structure (discrete translational invariance) of the lattice, and are well localized. Therefore, it seems possible to visualize the adiabatic pumping process by following the adiabatic motion of the Wannier functions as the parameters of the lattice Hamiltonian are varied in time.

In fact, we will describe the adiabatic evolution of both the position and energy expectation values of the Wannier functions. By this, the toolbox for analyzing the adiabatic pumping procedure for control-freak-type pumping is extended to arbitrary pumping sequences.

## 4.3.1 Plot the Wannier Centers

According to the result (3.13), the Wannier center positions of a certain band, in units of the lattice constant, are given by the Berry phase of that band divided by $2\pi$ . Hence, to follow the motion of the Wannier centers during the pumping procedure, we calculate the Berry phase of the given band for each moment of time.

The numerically computed Wannier-center positions obtained for the smoothly modulated Rice-Mele sequence [defined via Eqs. $(4.6)$ and $(4.7)$ ], with $\bar{v}=1$ , on a finite lattice with periodic boundary conditions, are shown in Fig. 4.5a. Solid (dotted) lines correspond to Wannier states of the valence, i.e., lower (conduction, i.e., upper) band. The results show that during a complete cycle, each Wannier center of the valence (conduction) band moves to the right (left) with a single lattice constant. Figure 4.5b shows time evolution of the position and energy expectation value of a single Wannier center in the valence/conduction band, over a few complete cycles. In complete analogy with control-freak pumping, these results suggest that the considered pumping sequence operates as a conveyor belt: it transports valence-band electrons from left to right, with a speed of one lattice constant per cycle, and would transport conduction-band electrons, if they were present, from right to left, with the same pace.

![](images/732e7c930f1a440e9310a2feb2837e3c6e4834a1244147381fb29e8ccd4e2d4c.jpg)

![](images/b28a017804e745e8164ce5d449e15e55469c1046fc073ffdfd554db643de1482.jpg)
Fig. 4.5 Time evolution of Wannier centers and Wannier energies in a smoothly modulated topological Rice-Mele pumping sequence [defined via Eqs. (4.6) and (4.5)]. The parameter of the sequence is $\bar{v}=1$ , corresponding to a Chern number of 1. In both subfigures, a solid (dotted) line corresponds to the valence (conduction) band. (a) Time evolution of the Wannier centers of the bands. During a cycle, each Wannier center of the valence (conduction) band moves to the right (left) with a single lattice constant. (b) Time evolution of the position and energy expectation value of a single Wannier center in the valence/conduction band, over a few complete cycles

An important result, which is not specific to the considered pumping cycle, arises from the above considerations. As pumping is cyclic, the Berry phase of a given band at t = 0 is equivalent to that at t = T. Therefore, the displacement of the Wannier center during a complete cycle is an integer.

## 4.3.2 Number of Pumped Particles Is the Chern Number

Our above results for the smoothly modulated Rice-Mele cycle suggest the interpretation that during a complete cycle, each electron in the filled valence band is displaced to the right by a single lattice constant. From this interpretation, it follows that the number of particles pumped from left to right, through an arbitrary cross section of the lattice, during a complete cycle, is one. Generalizing this consideration for arbitrary one-dimensional lattice models and pumping cycles, it suggests that the number of pumped particles is an integer.

We now shown that this integer is the Chern number associated to the valence band, that is, to the ground-state manifold of the time-dependent bulk momentum-space Hamiltonian $\hat{H}(k,t)$ . To prove this, we first write the Wannier-center displacement $\Delta x_{0,T}$ for the complete cycle by splitting up the cycle $[0,T]$ to small pieces $\Delta t$ :

$$
\Delta x _ {0, T} = \lim _ {n \rightarrow \infty} \sum_ {i = 0} ^ {n - 1} \Delta x _ {t _ {i}, t _ {i} + \Delta t},\tag{4.13}
$$

where $\Delta t = T/n$ and $t_{i} = i\Delta t$ . Then, we express the infinitesimal displacements with the Berry phases,

$$
\Delta x (t _ {i}, t _ {i} + \Delta t) = \frac {i}{2 \pi} \int_ {- \pi} ^ {\pi} d k \left[ \langle u _ {n} (t _ {i} + \Delta t) | \partial_ {k} u _ {n} (t _ {i} + \Delta t) \rangle - \langle u _ {n} (t _ {i}) | \partial_ {k} u _ {n} (t _ {i}) \rangle \right],\tag{4.14}
$$

where the k argument is suppressed for brevity. Since the $k = -\pi$ and $k = \pi$ values are equivalent, the above integral can be considered as a line integral of the Berry connection to the closed boundary line $\partial R_{i}$ of the infinitesimally narrow rectangle $(k, t) \in R_{i} = [-\pi, \pi) \times [t_{i}, t_{i} + \Delta t]$ ; that is,

$$
\Delta x \left(t _ {i}, t _ {i} + \Delta t\right) = \frac {1}{2 \pi} \oint_ {\partial R _ {i}} \mathbf {A} ^ {(n)} \cdot d \mathbf {R}.\tag{4.15}
$$

Using the fact that we can choose a gauge that is locally smooth on that rectangle $R_{i}$ , we obtain

$$
\Delta x (t _ {i}, t _ {i} + \Delta t) = \frac {1}{2 \pi} \int_ {\partial R _ {i}} B ^ {(n)} d k d t,\tag{4.16}
$$

where $B^{(n)}$ is the Berry curvature associated to the $n$ th eigenstate manifold of $\hat{H}(k,t)$ . Together with Eq. (4.13), this result ensures that the Wannier-center displacement is the Chern number:

$$
\varDelta x _ {0, T} = \frac {1}{2 \pi} \int_ {0} ^ {T} d t \int_ {- \pi} ^ {\pi} d k B ^ {(n)}.\tag{4.17}
$$

Note that even though we have not performed an explicit calculation of the valence-band Chern number of the smooth Rice-Mele pump cycle with $\bar{v} = 1$ , by looking at the motion of the corresponding Wannier centers we can conclude that the Chern number is 1.

## 4.3.3 Tuning the Pump Using the Average Intracell Hopping Amplitude $\overline{v}$

So far, the discussed results were obtained for the special case of the smoothly modulated Rice-Mele pumping cycle with average intracell hopping $\bar{v} = 1$ . Now we ask the question: can the number of pumped particles be changed by tuning the parameter $\bar{v}$ ? To show that the answer is yes, and the pump has such a tunability, on Fig. 4.6a we plot the instantaneous energy spectrum corresponding to $\bar{v} = -1$ . The spectrum reveals that this sequence, similarly to the $\bar{v} = 1$ case, does pump a single particle per cycle. However, the direction of pumping is opposite in the two cases: Fig. 4.6a shows that during a cycle, one edge state on the left (light blue) crosses over from the valence band to the conduction band, revealing that the particles are pumped from right to left in the valence band.

## 4.3.4 Robustness Against Disorder

So far, we have the following picture of an adiabatic pump in a long open chain. If we take a cross section at the middle of the chain, a single particle will be pumped through that, from left to right, during a complete cycle. This implies that at the end of the cycle, the number of particles on the right side of the cross section has grown by one, therefore the final state reached by the electron system is different from the original, ground state. This implies that during the course of the cycle, a valence-band energy eigenstate deformed into a conduction-band state, and that occurred on the right edge of the chain; the opposite happens on the left edge.

![](images/51f873d0cdaaf3bd2cd60b092e5ff46f6a3073481f521eb5e8a97f93fb7cc6b1.jpg)

![](images/25ba86e3bdc92a076591476b4bf4607508cb5e7c44a767a9830271b5c9c1e4d6.jpg)
Fig. 4.6 The smooth pump sequence of the Rice-Mele model for $\bar{\nu} = -1$ , revealing reversed pumping with respect to the $\bar{\nu} = 1$ case. (a) Instantaneous spectrum of the Hamiltonian $\hat{H}(t)$ on an open chain of N = 10 sites. During a cycle, one edge state on the left (light blue) crosses over from the valence to the conduction band, revealing that there is a single particle per cycle is pumped from right to left in the valence band. Wavefunctions of the edge states (b, c) as well as a typical bulk state (d) are also shown

Does that qualitative behavior change if we introduce disorder in the edge regions of the open chain? No: as long as the bulk of the chain remains regular, the pump works at the middle of the chain, and therefore the above conclusion about the exchange of a pair of states between the valence and conduction bands still holds. On the other hand, introducing disorder in the bulk seems to complicate the above-developed description of pumping in terms of Wannier-center motion, and therefore might change number of edge states and the qualitative nature of the instantaneous energy spectrum.

## Problems

## 4.1 Further control-freak pump cycles

Construct a control-freak pump cycle where the spectrum of the bulk does not change during the entire cycle, and the pumped charge is (a) zero (b) one.

## Chapter 5 Current Operator and Particle Pumping

In the previous chapter, we described quantized adiabatic pumping of particles in a one-dimensional lattice in an intuitive and visual fashion, using the concepts of the control-freak pumping cycle and the time evolution of the Wannier centers. Here, we provide a more formal description of the same effect. For simplicity, we consider two-band insulator lattice models with a completely filled lower band, which are described by a periodically time-dependent bulk momentum-space Hamiltonian of the form

$$
\hat {H} (k, t) = \mathbf {d} (k, t) \cdot \hat {\boldsymbol {\sigma}},\tag{5.1}
$$

where $\mathbf{d}(k,t)$ is a dimensionless three-dimensional vector fulfilling $\mathbf{d}(k,t)\geq1$ , and $\hat{\sigma}$ is the vector of Pauli matrices. This Hamiltonian is periodic both in momentum and in time, $\hat{H}(k+2\pi,t)=\hat{H}(k,t+T)=\hat{H}(k,t)$ , where T is the period of the time dependence of the Hamiltonian. The minimal energy gap between the two eigenstates of the Hamiltonian is 2. Furthermore, the frequency characterising the periodicity of the Hamiltonian is $\Omega\equiv2\pi/T$ . We call the periodically time-dependent Hamiltonian quasi-adiabatic, if $\Omega\ll1$ , and the adiabatic limit is defined as $\Omega\to0$ , that is, $T\to\infty$ .

For example, d can be chosen as

$$
\mathbf {d} (k, t) = \left( \begin{array}{c} \bar {v} + \cos \Omega t + \cos k \\ \sin k \\ \sin \Omega t \end{array} \right),\tag{5.2}
$$

corresponding to the smoothly modulated Rice-Mele model, see Eq. $(4.7)$ and Eq. $(4.6)$ .

We will denote the eigenstate of $\hat{H}(k,t)$ with a lower (higher) energy eigenvalue as $|u_{1}(k,t)\rangle$ ( $|u_{2}(k,t)\rangle$ ). With this notation, we can express the central result of this chapter: in adiabatic pumping, the momentum- and time-resolved current carried by the electrons of the filled band is a sum of two terms: a group-velocity term and the Berry curvature of that band. The group-velocity term does not contribute to pumping. As a consequence, the number 2 of particles pumped through an arbitrary cross section of an infinite one-dimensional crystal during a complete adiabatic cycle is the momentum- and time integral of the Berry curvature, that is,

$$
\mathcal {Q} = - i \frac {1}{2 \pi} \int_ {0} ^ {T} d t \int_ {- \pi} ^ {\pi} d k \left(\partial_ {k} \left\langle u _ {1} (k, t) \right| \partial_ {t} u _ {1} (k, t) \right\rangle - \partial_ {t} \left\langle u _ {1} (k, t) \right| \partial_ {k} u _ {1} (k, t) \rangle  .\tag{5.3}
$$

This is the Chern number associated to the ground-state manifold of $\hat{H}(k,t)$ . As the latter is an integer, the number of pumped particles is quantized. This result was discovered by David Thouless [33].

We derive Eq. $(5.3)$ via the following steps. In Sect. 5.1, we consider a generic time-dependent lattice Hamiltonian, and we express the number of particles moving through a cross section of the lattice using the current operator and the time-evolving states of the particles. Then, in Sect. 5.2.2, we provide a description of the time-evolving states in the case of periodic and quasi-adiabatic time dependence of the lattice Hamiltonian. This allows us to express the number of pumped particles for quasi-adiabatic time dependence. Finally, in Sect. 5.3, building on the latter result for quasi-adiabatic pumping, we take the adiabatic limit and thereby establish the connection between the current, the Berry curvature, the number of pumped particles, and the Chern number.

## 5.1 Particle Current at a Cross Section of the Lattice

Our aim here is to express the number of particles pumped through a cross section of the lattice, assuming that the time evolution of the Bloch states due to the time-dependence of the Hamiltonian is known. As intermediate steps toward this end, we derive the real-space current operator and the diagonal matrix elements of the momentum-space current operator, and establish an important relation between those diagonal matrix elements and the momentum-space Hamiltonian. For concreteness, we first discuss these using the example of the Rice-Mele model introduced in the preceding chapter. It is straightforward to generalize the results for lattice models with a generic internal degree of freedom; the generalized results are also given below. Finally, we use the relation between the current and the Hamiltonian to express the number of pumped particles with the time-evolving states and the Hamiltonian.

![](images/5354613d7b37b6e86c5521f8c0e0015e5735f2af1f93b57ddbaf8b02050ba72e.jpg)
Fig. 5.1 A segment enclosed by two cross sections in the one-dimensional SSH model. The segment S is defined as the part of the chain between the pth and $(q + 1)$ th unit cells. The current operators corresponding to the two cross sections can be established by considering the temporal change of the number of particles in the segment. The dashed line represents the periodic boundary condition

## 5.1.1 Current Operator in the Rice-Mele Model

We start things off with an example and consider the Rice-Mele model with $N \gg 1$ unit cells and periodic boundary conditions (Fig. 5.1). The real-space bulk Hamiltonian $\hat{H}_{bulk}$ has almost the same form as Eq. (3.1), with the difference that the sum corresponding to intercell hopping runs up to N, and in accordance with the periodic boundary condition, the unit cell index m should be understood as (m mod N). The bulk momentum-space Hamiltonian $\hat{H}(k)$ of the model is given in Eq. (4.6).

## 5.1.1.1 Influx of Particles into a Segment of the Crystal

We aim at establishing the operator representing the particle current flowing through a cross section of the one-dimensional crystal. We take a cross section between the pth and $(p+1)$ th unit cells, and denote the corresponding current operator as $\hat{j}_{p+1/2}$ . To find the current operator, we first consider a segment S of the crystal, stretching between (and including) the $(p+1)$ th and qth unit cells, where $q \geq p + 1$ . The number of particles in that segment S is represented by the operator

$$
\hat {\mathcal {Q}} _ {S} \equiv \sum_ {m \in S} \sum_ {\alpha \in \{A, B \}} | m, \alpha \rangle \langle m, \alpha |.\tag{5.4}
$$

Now, the time evolution of the number of particles embedded in the segment $S$ follows

$$
\partial_ {t} \langle \hat {\mathcal {Q}} _ {S} \rangle_ {t} = - i \langle [ \hat {\mathcal {Q}} _ {S}, \hat {H} (t) ] \rangle_ {t}\tag{5.5}
$$

Hence, we identify the operator describing the influx of particles into the segment as

$$
\hat {j} _ {S} (t) = - i [ \hat {\mathcal {Q}} _ {S}, \hat {H} (t) ].\tag{5.6}
$$

A straightforward calculation shows that Eq. (5.6) implies

$$
\begin{array}{r l} & {\hat {j} _ {S} (t) = - i w (t) \left(| p + 1, A \rangle \langle p, B | - | p, B \rangle \langle p + 1, A | \right.} \\ & {\quad \left. + | q, B \rangle \langle q + 1, A | - | q + 1, A \rangle \langle q, B |\right)} \end{array}\tag{5.7}
$$

Remarkably, the operator $\hat{j}_{S}(t)$ is time dependent, if the hopping amplitude $w(t)$ is time dependent.

## 5.1.1.2 Current Operator at a Cross Section of the Crystal

Clearly, the terms in Eq. (5.7) can be separated into two groups: the first two terms are hopping operators bridging the cross section $p + 1/2$ (that is, the cross section between unit cell p and unit cell $p + 1$ ), and the last two terms are bridging the cross section $q + 1/2$ . Thereby we define

$$
\hat {j} _ {m + 1 / 2} (t) \equiv - i w (t) \left(| m + 1, A \rangle \langle m, B | - | m, B \rangle \langle m + 1, A |\right),\tag{5.8}
$$

and use this definition to express $\hat{j}_{S}$ as

$$
\hat {j} _ {S} = \hat {j} _ {p + 1 / 2} - \hat {j} _ {q + 1 / 2}.\tag{5.9}
$$

This relation allows us to interpret $\hat{j}_{m + 1 / 2}(t)$ as the current operator describing particle flow, from left to right, across the cross section $m + 1 / 2$ .

## 5.1.1.3 Relation of the Current Operator to the Hamiltonian and to the Group Velocity

Later we will need the momentum-diagonal matrix elements of the current operator, which are defined as

$$
\hat {j} _ {m + 1 / 2} (k, t) = \langle k | \hat {j} _ {m + 1 / 2} (t) | k \rangle .\tag{5.10}
$$

For the Rice-Mele model under consideration, these can be expressed using Eqs. (1.8) and (5.8) as

$$
\hat {j} _ {m + 1 / 2} (k, t) = \frac {1}{N} \left( \begin{array}{c c} 0 & - i w (t) e ^ {- i k} \\ i w (t) e ^ {i k} & 0 \end{array} \right).\tag{5.11}
$$

From a comparison of this result and Eq. $(4.6)$ , we see that the momentum-diagonal matrix elements of the current operator are related to the momentum-space Hamiltonian as

$$
\hat {j} _ {m + 1 / 2} (k, t) = \frac {1}{N} \partial_ {k} \hat {H} (k, t).\tag{5.12}
$$

This is the central result of this section. Even though we have derived it only for the case of the Rice-Mele model, it is a generic result. A generalization is outlined in the next section.

In the case of a lattice without an internal degree of freedom, it is easy to see that Eq.(5.12) establishes the equivalence $j_{m+1/2}(k,t) = v_k(t)/N$ between the current and the (instantaneous) group velocity $v_k(t)$ : in this case, the bulk momentum-space Hamiltonian $\hat{H}(k,t)$ equals the dispersion relation $E(k,t)$ , and its momentum derivative $v_k(t) = \frac{\partial E(k,t)}{\partial k}$ is defined as the group velocity of the energy eigenstates. This correspondence generalizes to lattices with an internal degree of freedom as well. The current carried by an instantaneous energy eigenstate $|\Psi_n(k,t)\rangle = |k\rangle \otimes |u_n(k,t)\rangle$ of such a lattice is

$$
\begin{array}{l} \langle u _ {n} | \hat {j} _ {m + 1 / 2} | u _ {n} \rangle = \frac {1}{N} \langle u _ {n} | [ \partial_ {k} \hat {H} ] | u _ {n} \rangle \\ = \frac {1}{N} \langle u _ {n} | [ \partial_ {k} \sum_ {n ^ {\prime}} E _ {n ^ {\prime}} | u _ {n ^ {\prime}} \rangle \langle u _ {n ^ {\prime}} | ] | u _ {n} \rangle \\ = \frac {1}{N} \{\langle u _ {n} | [ (\partial_ {k} E _ {n}) | u _ {n} \rangle \langle u _ {n} | + E _ {n} | \partial_ {k} u _ {n} \rangle \langle u _ {n} | + E _ {n} | u _ {n} \rangle \langle \partial_ {k} u _ {n} | ] | u _ {n} \rangle \} \\ = \frac {1}{N} (\partial_ {k} E _ {n} + E _ {n} \partial_ {k} \langle u _ {n} | u _ {n} \rangle) = \frac {\partial_ {k} E _ {n}}{N} = \frac {v _ {n , k}}{N}, \end{array} \tag {5.13}
$$

where the arguments $(k, t)$ are suppressed for brevity.

## 5.1.2 Current Operator in a Generic One-Dimensional Lattice Model

Here we prove Eq. $(5.12)$ in a more general setting. Previously, we focused on the Rice-Mele model that has only two bands and hopping only between nearest-neighbor cells. Consider now a general one-dimensional lattice model with $N_{b}$ bands and finite-range hopping with range $1 \leq r \ll N$ ; here, r = 1 corresponds to hopping between neighbouring unit cells only. As before, we take a long chain with $N \gg 1$ unit cells, and assume periodic boundary conditions.

The real-space Hamiltonian has the form

$$
\hat {H} (t) = \sum_ {m, m ^ {\prime} = 1} ^ {N} \sum_ {\alpha , \alpha^ {\prime} = 1} ^ {N _ {b}} H _ {m \alpha , m ^ {\prime} \alpha^ {\prime}} (t) | m, \alpha \rangle \left\langle m ^ {\prime} \alpha^ {\prime} \right|,\tag{5.14}
$$

where $m$ and $m'$ are unit cell indices, and $\alpha, \alpha' \in \{1, 2, \ldots, N_b\}$ correspond to the internal degree of freedom within the unit cell. Due to the finite-range-hopping assumption, the Hamiltonian can also be written as

$$
\hat {H} (t) = \sum_ {m = 1} ^ {N} \sum_ {i = - r} ^ {r} \sum_ {\alpha , \alpha^ {\prime} = 1} ^ {N _ {b}} H _ {m + i, \alpha ; m, \alpha^ {\prime}} (t) | m + i, \alpha \rangle \left\langle m \alpha^ {\prime} \right|.\tag{5.15}
$$

Again, a unit cell index m should be understood as $(m \mod N)$ . Also, due to the discrete translational invariance, we have $H_{m+i,\alpha;m,\alpha'} = H_{i\alpha;0,\alpha'}$ , implying

$$
\hat {H} (t) = \sum_ {m = 1} ^ {N} \sum_ {i = - r} ^ {r} \sum_ {\alpha , \alpha^ {\prime} = 1} ^ {N _ {b}} H _ {i, \alpha ; 0 \alpha^ {\prime}} (t) | m + i, \alpha \rangle \left\langle m \alpha^ {\prime} \right|\tag{5.16}
$$

The bulk momentum-space Hamiltonian then reads

$$
\hat {H} (k, t) \equiv \langle k | \hat {H} (t) | k \rangle = \sum_ {m = - r} ^ {r} H _ {m, \alpha ; 0, \alpha^ {\prime}} (t) e ^ {- i k m} | \alpha \rangle \left\langle \alpha^ {\prime} \right|.\tag{5.17}
$$

Next, we establish the operator representing the particle current flowing through a cross section of the one-dimensional crystal, the same way we did in Sect. 5.1.1. We take a cross section between the pth and $(p + 1)$ th unit cells, and denote the corresponding current operator as $\hat{j}_{p+1/2}$ . To find the current operator, we first consider a segment S of the crystal, stretching between (and including) the $(p + 1)$ th and qth unit cells, where $q - p \geq r$ . The number of particles embedded in that long segment is represented by the operator

$$
\hat {\mathcal {Q}} _ {S} \equiv \sum_ {m \in S} \sum_ {\alpha = 1} ^ {N _ {b}} | m \alpha \rangle \langle m \alpha |.\tag{5.18}
$$

As discussed in the preceding section, we identify the operator describing the influx of particles into the wire segment as

$$
\hat {j} _ {S} (t) = - i [ \hat {\mathcal {Q}} _ {S}, \hat {H} (t) ].\tag{5.19}
$$

From this, a straightforward calculation shows that

$$
\begin{array}{r}\hat {j} _ {S} (t) = - i \sum_ {m \in S} \sum_ {m ^ {\prime} \notin S} \sum_ {\alpha , \alpha^ {\prime} = 1} ^ {N _ {b}} \left[ \right. H _ {m \alpha , m ^ {\prime} \alpha^ {\prime}} (t)   | m \alpha \rangle \left<   m ^ {\prime} \alpha^ {\prime} \left. \right|\left. \right.\\\left. - H _ {m ^ {\prime} \alpha^ {\prime}, m \alpha} (t)   | m ^ {\prime} \alpha^ {\prime} \rangle \langle m \alpha | \right].\end{array}\tag{5.20}
$$

Note that Eq. (5.20) testifies that the operator $\hat{j}_{S}(t)$ is constructed only from those hopping matrix elements of the Hamiltonian that bridge either the $p + 1/2$ or the $q + 1/2$ cross sections of the crystal, i.e., one of the two cross sections that terminate the segment under consideration. This is ensured by the condition that the segment is at least as long as the range r of hopping. A further consequence of this is that the terms in Eq. $(5.20)$ can be separated into two groups: one containing the hopping matrix elements bridging the cross section $p + 1/2$ , and one with those bridging the cross section $q + 1/2$ . The former reads

$$
\begin{array}{r} \hat {j} _ {p + 1 / 2} (t) = - i \sum_ {m = p + 1} ^ {p + r} \sum_ {m ^ {\prime} = p + 1 - r} ^ {p} \sum_ {\alpha , \alpha^ {\prime} = 1} ^ {N _ {b}} \left[ H _ {m \alpha , m ^ {\prime} \alpha^ {\prime}} (t) | m \alpha \rangle \langle m ^ {\prime} \alpha^ {\prime} | - H _ {m ^ {\prime} \alpha^ {\prime}, m \alpha} (t) | m ^ {\prime} \alpha^ {\prime} \rangle \langle m \alpha | \right]. \end{array}\tag{5.21}
$$

Using this as a definition for any cross section $m + 1/2$ , we conclude that Eq. (5.9) holds without any change in this generalized case as well. This conclusion allows us to interpret $\hat{j}_{m+1/2}$ as the current operator describing particle flow, from left to right, across the cross section $m + 1/2$ .

After defining the momentum-diagonal matrix elements of the current operator exactly the same way as in Eq. (5.10), the relation between the current operator and the Hamiltonian has exactly the same form as in Eq. (5.12). This can be proven straightforwardly using Eqs. (5.10), (5.21), and the $k$ -derivative of Eq. (5.17).

## 5.1.3 Number of Pumped Particles

Let us return to particle pumping in insulating two-band models. As the electrons are assumed to be non-interacting, and the time-dependent lattice Hamiltonian has a discrete translational invariance for all times, the many-electron state $\Phi(t)$ is a Slater determinant of Bloch-type single-particle states $\left|\tilde{\Psi}_{1}(k,t)\right\rangle = |k\rangle \otimes |\tilde{u}_{1}(k,t)\rangle$ . Here, we adopted the notation introduced in Sect. 1.2, with n = 1 referring to the filled band. There are also two additions with respect to the notation of Sect. 1.2: first, we explicitly denote the time dependence of the state; second, we added a tilde here to denote that the state $\left|\tilde{\Psi}_{1}(k,t)\right\rangle$ is not an instantaneous lower-band eigenstate $\left|\Psi_{1}(k,t)\right\rangle$ of the Hamiltonian, but is slightly different from that due to the quasi-adiabatic driving.

The number of particles pumped through the cross section $m + 1/2$ within the time interval $t \in [0, T]$ is the time-integrated current, that is,

$$
\mathcal {Q} = \int_ {0} ^ {T} d t \left\langle \Phi (t) \right| j _ {m + 1 / 2} ^ {M} (t) \left| \Phi (t) \right\rangle ,\tag{5.22}
$$

where $j_{m+1/2}^{M}(t)$ is the many-particle generalisation of the current operator defined in Eq. (5.21), or, for the special case of the Rice-Mele model, in Eq. (5.8), and $\Phi(t)$ is the many-electron Slater determinant formed by the filled Bloch-type single-particle states, introduced in the preceding paragraph. Equation (5.22) can be converted to

an expression with single-particle states:

$$
\mathcal {Q} = \int_ {0} ^ {T} d t \sum_ {k \in \mathrm{BZ}} \left\langle \tilde {\Psi} _ {1} (k, t) \right| \hat {j} _ {m + 1 / 2} (t) \left| \tilde {\Psi} _ {1} (k, t) \right\rangle ,\tag{5.23}
$$

which is related to the momentum-diagonal matrix elements of the current operator as

$$
\mathcal {Q} = \int_ {0} ^ {T} d t \sum_ {k \in \mathrm{BZ}} \left\langle \tilde {u} _ {1} (k, t) \right| \hat {j} _ {m + 1 / 2} (k, t) \left| \tilde {u} _ {1} (k, t) \right\rangle .\tag{5.24}
$$

Here, BZ stands for Brillouin Zone. Finally, this is rewritten using the current-Hamiltonian relation Eq. (5.12) as

$$
\mathcal {Q} = \frac {1}{N} \int_ {0} ^ {T} d t \sum_ {k \in \mathrm{BZ}} \left\langle \tilde {u} _ {1} (k, t) \right| \partial_ {k} \hat {H} (k, t) \left| \tilde {u} _ {1} (k, t) \right\rangle .\tag{5.25}
$$

To evaluate this in the case of adiabatic, periodically time-dependent Hamiltonian, we first need to understand how the two-level wave functions $|\tilde{u}_{1}(k,t)\rangle$ evolve in time in the quasi-adiabatic case; then we can insert those in Eq. (5.25), and take the adiabatic limit.

## 5.2 Time Evolution Governed by a Quasi-Adiabatic Hamiltonian

Our goal here is to describe the time evolution of Bloch-type electronic energy eigenstates. Nevertheless, as the pumping dynamics preserves the wavenumber k, the task simplifies to describe the dynamics of distinct two-level systems, labelled by the wavenumber k. Therefore, in this section we discuss the dynamics of a single two-level system, hence the wavenumber k does not appear in the formulas. We will restore k when evaluating the number of pumped particles in the next section. To describe the time evolution of the electronic states subject to quasi-adiabatic driving, it is convenient to use the so-called parallel-transport gauge or parallel-transport time parametrization, which we introduce below. Then, our goal is reached by performing perturbation theory in the small frequency $\Omega \ll 1$ characterizing the quasi-adiabatic driving.

## 5.2.1 The Parallel-Transport Time Parametrization

As mentioned earlier, the instantaneous energy eigenstates of the bulk momentum-space Hamiltonian $\hat{H}(k,t)$ are denoted as $|u_{n}(k,t)\rangle$ . Here, in order to simplify the derivations, we will use a special time parametrization (gauge) for these eigenstates, which is called the parallel-transport time parametrization or parallel-transport gauge. As mentioned above, we suppress the momentum k.

We will call the smooth time parametrization $|u_{n}(t)\rangle$ of the instantaneous nth eigenstate of the Hamiltonian $\hat{H}(t)$ a parallel-transport time parametrization, if for any time point t and any band n, it holds that

$$
\langle u _ {n} (t) | \partial_ {t} | u _ {n} (t) \rangle = 0.\tag{5.26}
$$

Using a time parametrization with this property will simplify the upcoming calculations of this section.

In Sect. 2.3, we used smooth parametrizations that were defined via a parameter space, and therefore were cyclic. For any time parametrization $\left|u_{n}^{\prime}(t)\right\rangle$ having those properties, we can construct a parallel-transport time parametrizaton $\left|u_{n}(t)\right\rangle$ via the definition

$$
\left| u _ {n} (t) \right\rangle = e ^ {i \gamma_ {n} (t)} \left| u _ {n} ^ {\prime} (t) \right\rangle ,\tag{5.27}
$$

where $\gamma_{n}(t)$ is the adiabatic phase associated to the adiabatic time evolution of the initial state $\left|u_n'(t = 0)\right\rangle$ , governed by our adiabatically varying Hamiltonian $\hat{H}(t)$ :

$$
\gamma_ {n} (t) = i \int_ {0} ^ {t} d t ^ {\prime} \left\langle u _ {n} ^ {\prime} (t ^ {\prime}) \right| \partial_ {t ^ {\prime}} \left| u _ {n} ^ {\prime} (t ^ {\prime}) \right\rangle .\tag{5.28}
$$

The fact that $|u_n(t)\rangle$ indeed fulfils Eq. (5.26) can be checked by performing the time derivation and the scalar product on the left hand side of the latter.

As an interpretation of Eq. (5.27), we can say that a parallel-transport time parametrization is an adiabatically time-evolving state divided by the dynamical phase factor. Furthermore, as the Berry phase factor $e^{i\gamma_{n}(T)}$ is, in general, different from 1, the parallel-transport time parametrization (5.27) is, in general, not cyclic.

## 5.2.2 Quasi-Adiabatic Evolution

Here, following Thouless [33], we describe the quasi-adiabatic time evolution using stationary states, also known as Floquet states, that are characteristic of periodically driven quantum systems. Again, we focus on two-level systems as introduced in Eq. (5.1), and suppress the wave number $k$ in our notation. The central result of this section is Eq. (5.42), which expresses how the instantaneous ground state mixes weakly with the instantaneous excited state due to the quasi-adiabatic time dependence of the Hamiltonian. In the next section, this result is used to evaluate the particle current and the number of pumped particles.

As an example, we can consider the state corresponding to the wavenumber k = 0 in the smoothly modulated Rice-Mele model with $\bar{v} = 1$ , see Eq. (5.2):

$$
\mathbf {d} (t) = \left( \begin{array}{c} 2 + \cos (\varOmega t) \\ 0 \\ \sin (\varOmega t) \end{array} \right).\tag{5.29}
$$

## 5.2.2.1 Stationary States of Periodically Driven Dynamics

The stationary states are special solutions of the periodically time-dependent Schrödinger equation, which are essentially periodic with period T; that is, which fulfill $|\psi(t+T)\rangle = e^{-i\phi}|\psi(t)\rangle$ for any t, with $\phi$ being a t-independent real number. The number of such nonequivalent solutions equals the dimension of the Hilbert space of the quantum system, i.e., there are two of them for the case we consider. Here we describe stationary states in the quasi-adiabatic case, when the time evolution of the Hamiltonian is slow compared to the energy gap between the instantaneous energy eigenvalues: $\Omega \ll 1$ . This condition suggests that the deviation from the adiabatic dynamics is small, and therefore each stationary state is in the close vicinity of either the instantaneous ground state or the instantaneous excited state. Thereby, we will label the stationary states with the band index n, and denote them as $|\tilde{u}_{n}(t)\rangle$ .

Since, after all, we wish to describe pumping in a lattice with a filled lower band and an empty upper band, we mostly care about the stationary state corresponding to the lower band, $|\tilde{u}_{1}(t)\rangle$ , and therefore want to solve the time-dependent Schrödinger equation

$$
- i \partial_ {t} | \tilde {u} _ {1} (t) \rangle + \hat {H} (t) | \tilde {u} _ {1} (t) \rangle = 0.\tag{5.30}
$$

## 5.2.2.2 Making Use of the Parallel-Transport Gauge

We characterize the time evolution of the wave function $|\tilde{u}_{1}(t)\rangle$ by a time-dependent linear combination of the instantaneous energy eigenstates:

$$
| \tilde {u} _ {1} (t) \rangle = a _ {1} (t) e ^ {- i \int_ {0} ^ {t} d t ^ {\prime} E _ {1} (t ^ {\prime})} | u _ {1} (t) \rangle + a _ {2} (t) e ^ {- i \int_ {0} ^ {t} d t ^ {\prime} E _ {2} (t ^ {\prime})} | u _ {2} (t) \rangle ,\tag{5.31}
$$

Recall that we are using the parallel-transport time parametrization, having the properties (5.26) and (5.27). Therefore, in the adiabatic limit $\Omega \to 0$ , we already now that $a_1(t) = 1$ and $a_2(t) = 0$ . Here, we are mostly interested in the quasi-adiabatic case defined via $\Omega \ll 1$ , and then it is expected that $a_1(t) \sim 1$ and $a_2(t) \sim \Omega \ll 1$ .

Before making use of that consideration in the form of perturbation theory in $\Omega$ , we convert the time-dependent Schrödinger equation (5.30) to two differential equations for the two unknown functions $a_{1}(t)$ and $a_{2}(t)$ . We insert $|\tilde{u}_{1}(t)\rangle$ of Eq. (5.31) to the time-dependent Schrodinger equation (5.30), yielding

$$
\begin{array}{l} - i \partial_ {t} \left[ a _ {1} (t) e ^ {- i \int_ {0} ^ {t} d t ^ {\prime} E _ {1} (t ^ {\prime})} | u _ {1} (t) \rangle + a _ {2} (t) e ^ {- i \int_ {0} ^ {t} d t ^ {\prime} E _ {2} (t ^ {\prime})} | u _ {2} (t) \rangle \right] \\ + \hat {H} (t) \left[ a _ {1} (t) e ^ {- i \int_ {0} ^ {t} d t ^ {\prime} E _ {1} (t ^ {\prime})} | u _ {1} (t) \rangle + a _ {2} (t) e ^ {- i \int_ {0} ^ {t} d t ^ {\prime} E _ {2} (t ^ {\prime})} | u _ {3} (t) \rangle \right] = 0 \end{array}\tag{5.32}
$$

After evaluating the time derivatives, the left hand side consists of 8 terms. Using the instantaneous eigenvalue relations $\hat{H}(t)|u_n(t)\rangle = E_n(t)|u_n(t)\rangle$ , two pairs of terms annihilate each other, and only 4 terms remain:

$$
\begin{array}{c} \dot {a} _ {1} (t)   | u _ {1} (t) \rangle + a _ {1} (t) \partial_ {t}   | u _ {1} (t) \rangle + \dot {a} _ {2} (t) e ^ {- i \int_ {0} ^ {t} d t ^ {\prime} E (t ^ {\prime})}   | u _ {2} (t) \rangle \\ + a _ {2} (t) e ^ {- i \int_ {0} ^ {t} d t ^ {\prime} E (t ^ {\prime})} \partial_ {t}   | u _ {2} (t) \rangle = 0, \end{array}\tag{5.33}
$$

where $E(t) = E_2(t) - E_1(t) = 2d(t)$ .

Projecting Eq. (5.33) onto $\langle u_{1}(t)|$ and $\langle u_{2}(t)|$ , respectively, and making use of the parallel-transport gauge, yields

$$
\dot {a} _ {1} (t) + a _ {2} (t) e ^ {- i \int_ {0} ^ {t} d t ^ {\prime} E (t ^ {\prime})} \left\langle u _ {1} (t) \right| \partial_ {t} | u _ {2} (t) \rangle = 0,\tag{5.34}
$$

$$
a _ {1} (t) \left\langle u _ {2} (t) \right| \partial_ {t} | u _ {1} (t) \rangle + \dot {a} _ {2} (t) e ^ {- i \int_ {0} ^ {t} d t ^ {\prime} E (t ^ {\prime})} = 0.\tag{5.35}
$$

The latter result can be rewritten as

$$
\dot {a} _ {2} (t) = - a _ {1} (t) \left<   u _ {2} (t) \left. \right| \partial_ {t} \left| \right. u _ {1} (t) \right> e ^ {i \int_ {0} ^ {t} d t ^ {\prime} E (t ^ {\prime})}\tag{5.36}
$$

## 5.2.2.3 Making Use of the Quasi-Adiabatic Condition

Note that the quasi-adiabatic condition has not been invoked so far; this is the next step. As mentioned above, the quasi-adiabatic nature of the Hamiltonian suggests that one of the two stationary states will be close to the instantaneous ground state, suggesting $a_1(t) \sim 1$ and $a_2(t) \sim \Omega$ . Furthermore, we know that $\langle u_1(t)|\partial_t|u_2(t)\rangle \sim \Omega$ , since variations in $|u_n(t)\rangle$ become slower as the adiabatic limit is approached. The latter relation is explicitly demonstrated by the example in Eq. (5.29): if we use $|u_1(t)\rangle = (-\sin (\theta /2),\cos (\theta /2))^T$ and $|u_2(t)\rangle = (\cos (\theta /2),\sin (\theta /2))^T$ with $\theta = \arctan \left(\frac{2 + \cos\Omega t}{\sin\Omega t}\right)$ , fulfilling the parallel-gauge criterion, we find $\langle u_1(t)|\partial_t|u_2(t)\rangle = -\Omega \frac{1 + 2\cos\Omega t}{10 + 8\cos\Omega t}$ .

As we are interested in the quasi-adiabatic case $\Omega \ll 1$ , we drop those terms from (5.34) and (5.36) that are at least second order in $\Omega$ . This results in

$$
\dot {a} _ {1} (t) = 0,\tag{5.37}
$$

$$
\dot {a} _ {2} (t) = - a _ {1} (t) \left\langle u _ {2} (t) \right| \partial_ {t} \left| u _ {1} (t) \right\rangle e ^ {i \int_ {0} ^ {t} d t ^ {\prime} E (t ^ {\prime})}.\tag{5.38}
$$

If we assume $a_{1}(t=0)=1+o(\Omega)$ , then the first equation guarantees that $a_{1}(t)=1+o(\Omega)$ . Then this allows for a further simplification of Eq. (5.38):

$$
\dot {a} _ {2} (t) = - \left<   u _ {2} (t) \left. \right| \partial_ {t} \left| \right. u _ {1} (t) \right> e ^ {i \int_ {0} ^ {t} d t ^ {\prime} E (t ^ {\prime})}.\tag{5.39}
$$

## 5.2.2.4 Solution of the Equation of Motion

The remaining task is to solve Eq. $(5.39)$ for $a_{2}(t)$ . Instead of doing this in a constructive fashion, we give the solution $a_{2}(t)$ and prove that it indeed fulfills Eq. $(5.39)$ up to the desired order. The solution reads

$$
a _ {2} (t) = i \frac {\langle u _ {2} (t) | \partial_ {t} | u _ {1} (t) \rangle}{E (t)} e ^ {i \int_ {0} ^ {t} d t ^ {\prime} E (t ^ {\prime})}.\tag{5.40}
$$

First, let us check if it solves the differential equation (5.39):

$$
\begin{array}{l} \partial_ {t} a _ {2} (t) = i \frac {\left(\partial_ {t} \langle u _ {2} (t) | \partial_ {t} | u _ {1} (t) \rangle\right)}{E (t)} e ^ {i \int_ {0} ^ {t} d t ^ {\prime} E (t ^ {\prime})} - i \frac {\left(\partial_ {t} E (t)\right) \langle u _ {2} (t) | \partial_ {t} | u _ {1} (t) \rangle}{E (t) ^ {2}} e ^ {i \int_ {0} ^ {t} d t ^ {\prime} E (t ^ {\prime})} \\ - \langle u _ {2} (t) | \partial_ {t} | u _ {1} (t) \rangle e ^ {i \int_ {0} ^ {t} d t ^ {\prime} E (t ^ {\prime})}. \end{array} \tag {5.4}\tag{5.41}
$$

The first two terms on the right hand side scale as $\Omega^{2}$ , whereas the third one scales as $\Omega$ . Hence we conclude that in the quasi-adiabatic case, Eq. (5.40) is the solution of Eq. (5.39) we were after. The corresponding solution of the time-dependent Schrödinger equation (5.30) is constructed using Eqs. (5.31), $a_{1}(t)=1$ and (5.38), and reads

$$
| \tilde {u} _ {1} (t) \rangle = e ^ {- i \int_ {0} ^ {t} d t ^ {\prime} E _ {1} (t ^ {\prime})} \left[ | u _ {1} (t) \rangle + i \frac {\langle u _ {2} (t) | \partial_ {t} | u _ {1} (t) \rangle}{E (t)} | u _ {2} (t) \rangle \right].\tag{5.42}
$$

In words, Eq. (5.42) assures that the stationary state has most of its weight in the instantaneous ground state $|u_{1}(t)\rangle$ , with a small, $\sim\Omega\ll1$ admixture of the instantaneous excited state $|u_{2}(t)\rangle$ . Interestingly, even though this small admixture vanishes in the adiabatic limit $\Omega\to0$ , the corresponding contribution to the number of pumped particles can give a finite contribution, as the cycle period T goes to infinity in the adiabatic limit. This will be shown explicitly in the next section.

Finally, we show that this state $|\tilde{u}_{1}(t)\rangle$ is indeed stationary. That is proven if we can prove that $|\tilde{u}_{1}(T)\rangle$ is equal to $|\tilde{u}_{1}(0)\rangle$ up to a phase factor. This arises as the consequence of the following fact. If the Berry phase associated to the state $|u_{1}\rangle$ is $\gamma$ , that is, if $|u_{1}(T)\rangle = e^{i\gamma} |u_{1}(0)\rangle$ , then

$$
\begin{array}{r l}\left[ \partial_ {t} | u _ {1} (t) \rangle \right] _ {T}&= \lim _ {\epsilon \rightarrow 0} \frac {| u _ {1} (T + \epsilon) \rangle - | u _ {1} (T) \rangle}{\epsilon} = \lim _ {\epsilon \rightarrow 0} \frac {e ^ {i \gamma} | u _ {1} (\epsilon) \rangle - e ^ {i \gamma} | u _ {1} (0) \rangle}{\epsilon}\\&= e ^ {i \gamma} \partial_ {t} | u _ {1} (0) \rangle .\end{array}\tag {5}\tag{5.43}
$$

Therefore, the two terms in the square bracket of Eq. (5.42) acquire the same phase factor $e^{i\gamma}$ at the end of the cycle, hence the obtained $|\tilde{u}_{1}(t)\rangle$ solution is stationary.

## 5.3 The Pumped Current Is the Berry Curvature

The number of particles pumped through an arbitrary cross section of the one-dimensional lattice, in the duration T of a quasi-adiabatic cycle, is evaluated combining Eqs. $(5.25)$ and $(5.42)$ . We define the momentum- and time-resolved current of the filled band as

$$
j _ {m + 1 / 2} ^ {(1)} (k, t) = \frac {1}{N} \left\langle \tilde {u} _ {1} (k, t) \right| \partial_ {k} \hat {H} (k, t) \left| \tilde {u} _ {1} (k, t) \right\rangle ,\tag{5.44}
$$

and perform the usual substitution $\frac{1}{N}\sum_{k\in BZ}\cdots=\int_{BZ}\frac{dk}{2\pi}\ldots$ , yielding the following formula for the number of pumped particles:

$$
\mathcal {Q} = \int_ {0} ^ {T} d t \int_ {\mathrm{BZ}} \frac {d k}{2 \pi} j _ {m + 1 / 2} ^ {(1)} (k, t).\tag{5.45}
$$

In the rest of this section, we show that the relevant contribution of the momentum-and time-resolved current is the Berry curvature associated to the filled band, and therefore the number of pumped particles is the Chern number, which in turn is indeed an integer.

To this end, we insert the result (5.42) to the definition (5.44). The contribution that incorporates two lower-band wave functions $|u_{1}(k,t)\rangle$ , is finite; however, its integral over the Brillouin Zone vanishes, and therefore we disregard it as it does not contribute to particle pumping. Hence the leading relevant contribution is the one incorporating one filled-band $|u_{1}(k,t)\rangle$ and one empty-band $|u_{2}(k,t)\rangle$ wavefunction:

$$
j _ {m + 1 / 2} ^ {(1)} (k, t) = i \frac {\langle u _ {1} | [ \partial_ {k} \hat {H} ] | u _ {2} \rangle \langle u _ {2} | \partial_ {t} | u _ {1} \rangle}{E} + c. c.\tag{5.46}
$$

where the k and t arguments are suppressed for brevity.

Now we use

$$
\langle u _ {1} | [ \partial_ {k} \hat {H} ] | u _ {2} \rangle = (E _ {1} - E _ {2}) \langle \partial_ {k} u _ {1} | u _ {2} \rangle = - E \langle \partial_ {k} u _ {1} | u _ {2} \rangle ,\tag{5.47}
$$

which has a straightforward proof using the spectral decomposition $\hat{H} = E_{1}|u_{1}\rangle\langle u_{1}| + E_{2}|u_{2}\rangle\langle u_{2}|$ of the Hamiltonian and the fact that $\partial_{k}\langle u_{1}|u_{2}\rangle = 0$ . Therefore,

$$
j _ {m + 1 / 2} ^ {(1)} = - i \left<   \partial_ {k} u _ {1} \left. \right| u _ {2} \rangle \left<   u _ {2} \left. \right| \partial_ {t} \left| \right. u _ {1} \right> + c. c..\tag{5.48}
$$

Since we use the parallel-transport gauge, we can replace the projector $|u_{2}\rangle\langle u_{2}|$ with unity in the preceding formula, hence the latter can be simplified as

$$
\begin{array}{c} j _ {m + 1 / 2} ^ {(1)} = - i \left\langle \partial_ {k} u _ {1} | \partial_ {t} u _ {1} \right\rangle + c. c. = - i \left(\left\langle \partial_ {k} u _ {1} | \partial_ {t} u _ {1} \right\rangle - \left\langle \partial_ {t} u _ {1} | \partial_ {k} u _ {1} \right\rangle\right) \\ = - i \left(\partial_ {k} \left\langle u _ {1} | \partial_ {t} u _ {1} \right\rangle - \partial_ {t} \left\langle u _ {1} | \partial_ {k} u _ {1} \right\rangle\right). \end{array}\tag{5.49}
$$

This testifies that the momentum- and time-resolved current is indeed the Berry curvature corresponding to the filled band, and thereby confirms the result promised in Eq. (5.3).

As a straightforward application of our result, we calculate the time dependence of the current and the number of pumped particles through an arbitrary cross section in the smoothly modulated Rice-Mele model, see Eq. $(5.2)$ . The results corresponding to four different values of the parameter $\bar{v}$ are shown in Fig. 5.2. The momentum- and time-resolved current $j_{m+1/2}^{(1)}(k,t)$ of the filled band can be obtained analytically from Eq. $(5.49)$ . Then, the time-resolved current j is defined as the integrand of the t integral in Eq. $(5.45)$ . We evaluate j via a numerical k integration, and plot the result in Fig. 5.2a. The number of pumped particles as a function of time is then evaluated numerically via $\mathcal{Q}(t)=\int_{0}^{t}dt^{\prime}j(t^{\prime})$ ; the results are shown in Fig. 5.2b. These results confirm that the number of particles pumped through the cross section during the complete cycle is an integer, and is given by the Chern number associated to the pumping cycle.

In this chapter, we have provided a formal description of adiabatic pumping in one-dimensional lattices. After identifying the current operator describing particle flow at a cross section of the lattice, we discussed the quasi-adiabatic time evolution of the lower-band states in a two-band model, and combined these results to express the number of pumped particles in the limit of adiabatic pumping. The central result is that the relevant part of the momentum- and time-resolved current carried by the lower-band electrons is the Berry curvature associated to their band.

![](images/44532c88a28af8f5298dc86c4de45fa7cf867926990e43543d85b1cd82f78dae.jpg)

(b)
![](images/6b3e94ef191b0c02e58293b877a0c05187cd3955dfb52c57bd28644350910e8d.jpg)
Fig. 5.2 Time dependence of the current and the number of pumped particles in an adiabatic cycle

## Problems

## 5.1 The smooth pump sequence of the Rice-Mele model

For the smoothly modulated Rice-Mele pumping cycle, see $(5.2)$ , evaluate the momentum- and time-dependent current density, and the number of particles pumped through an arbitrary unit cell boundary as the function of time; that is, reproduce Fig. 5.2.

## 5.2 Parallel-transport time parametrization

Specify a parallel-transport time parametrization for the ground state of the two-level Hamiltonian defined by Eqs. (5.1), (5.2), and (a) $k = 0$ (b) $k = \pi$ .

## 5.3 Quasi-adiabatic dynamics with a different boundary condition

In Sect. 5.2.2, we described a stationary state of a quasi-adiabatically driven two-level system, and used the result to express the number of particles pumped during a complete cycle. How does the derivation and the result change, if we describe the dynamics not via the stationary state, but by specifying that the initial state is the instantaneous ground state of the Hamiltonian at t = 0? Is the final result for the number of pumped particles different in this case?

## 5.4 Adiabatic pumping in multiband models

Generalize the central result of this chapter in the following sense. Consider adiabatic charge pumping in a one-dimensional multi-band system $(n = 1, 2, \ldots, N_{b})$ , where the energies of the first $N_{filled}$ bands $(n = 1, 2, \ldots, N_{\text{filled}})$ are below the Fermi energy and the energies of the remaining bands $(n = N_{\text{filled}} + 1, \ldots, N_{b})$ are above the Fermi energy, and the bands do not cross each other. Show that the number of particles adiabatically pumped through an arbitrary cross section of the crystal is the sum of the Chern numbers of the filled bands.

# Chapter 6 Two-Dimensional Chern Insulators: The Qi-Wu-Zhang Model

The unique physical feature of topological insulators is the guaranteed existence of low-energy states at their boundaries. We have seen an example of this for a one-dimensional topological insulator, the SSH model: A finite, open, topologically nontrivial SSH chain hosts zero-energy bound states at both ends. The bulk-boundary correspondence was the way in which the topological invariant of the bulk—in the case of the SSH chain, the winding number of the bulk Hamiltonian—can be used to predict the number of edge states.

We will show that the connection between the Chern number and the number of edge-state channels is valid in general for two-dimensional insulators. This is the statement of bulk-boundary correspondence for Chern insulators. The way we will show this inverts the argument above: taking any two-dimensional insulator, we can map it to an adiabatic pump sequence in a one-dimensional insulator by demoting one of the wavenumbers to time. The connection between the Chern number and the number of edge states in the higher dimensional Hamiltonian is a direct consequence of the connection between Chern number and charge pumping in the lower dimensional system.

Chern insulators (two-dimensional band insulators with nonvanishing Chern number) were first used to explain the Quantum Hall Effect. There an external magnetic field, included in lattice models via a Peierls substitution, is responsible for the nonzero value of the Chern number. Peierls substitution, however, breaks the lattice translation invariance. This necessitates extra care, including the use of magnetic Brillouin zones whose size depends on the magnetic field.

The models we construct in this chapter describe the so-called Quantum Anomalous Hall Effect. Here we have the same connection between edge states and bulk Chern number as in the Quantum Hall Effect, however, there is no external magnetic field, and thus no complications with magnetic Brillouin zones. The Quantum Anomalous Hall Effect has recently been observed in thin films of chromium-doped $(\mathrm{Bi},\mathrm{Sb})_{2}\mathrm{Te}_{3}$ [8].

To illustrate the concepts of Chern insulators, we will use a toy model introduced by Qi, Wu and Zhang $[24]$ , which we call the QWZ model. This model is also important because it forms the basic building block of the Bernevig-Hughes-Zhang model for the Quantum Spin Hall Effect (Chap. 8), and thus it is also sometimes called “half BHZ”.

## 6.1 Dimensional Extension: From an Adiabatic Pump to a Chern Insulator

We want to construct a two-dimensional lattice Hamiltonian $\hat{H}$ with a nonvanishing bulk Chern number. We will do this by first constructing the bulk momentum-space Hamiltonian $\hat{H}(k_x, k_y)$ , from which the real-space Hamiltonian can be obtained by Fourier transformation. For the construction we simply take an adiabatic pump sequence on a one-dimensional insulator, $\hat{H}(k, t)$ , and reinterpret the cyclic time variable $t$ as a new momentum variable $k_y$ . This way of gaining an extra dimension by promotion of a cyclic parameter in a continuous ensemble to a momentum is known as dimensional extension. This, and the reverse process of dimensional reduction, are key tools to construct the general classification of topological insulators [29].

## 6.1.1 From the Rice-Mele Model to the Qi-Wu-Zhang Model

To see how the construction of a Chern insulator works, we take the example of the smooth pump sequence on the Rice-Mele model from the previous chapter, Eq. (4.7). In addition to the promotion of time to an extra wavenumber, $\Omega t \to k_y$ , we also do an extra unitary rotation in the internal Hilbert space, to arrive at the Qi-Wu-Zhang model,

$$
\hat {H} (k) = \sin k _ {x} \hat {\sigma} _ {x} + \sin k _ {y} \hat {\sigma} _ {y} + [ u + \cos k _ {x} + \cos k _ {y} ] \hat {\sigma} _ {z}.\tag{6.1}
$$

The mapping is summarized in Table 6.1.

Table 6.1 Mapping of an adiabatic pump sequence of the Rice-Mele model, $\hat{H}(k,t)$ to the QWZ model for the Anomalous Hall Effect, $\hat{H}(k_{x},k_{y})$

<table><tr><td>Adiabatic pump in the RM model</td><td>QWZ model (Chern insulator)</td></tr><tr><td>Average intracell hopping  $\overline{v}$ </td><td>Staggered onsite potential  $u$ </td></tr><tr><td>Wavenumber  $k \in [0, 2\pi)$ </td><td>Wavenumber  $k_x \in [0, 2\pi)$ </td></tr><tr><td>Time  $t \in [0, T)$ </td><td>Wavenumber  $k_y \in [0, 2\pi)$ </td></tr><tr><td> $\sigma_x, \sigma_y, \sigma_z$ </td><td> $\sigma_y, \sigma_z, \sigma_x$ </td></tr></table>

The corresponding $\mathbf{d}(\mathbf{k})$ vector reads,

$$
\mathbf {d} (k _ {x}, k _ {y}) = \left( \begin{array}{c} \sin k _ {x} \\ \sin k _ {y} \\ u + \cos k _ {x} + \cos k _ {y} \end{array} \right).\tag{6.2}
$$

## 6.1.2 Bulk Dispersion Relation

We can find the dispersion relation of the QWZ model using the algebraic properties of the Pauli matrices, whereby $\hat{H}^{2}=E(\mathbf{k})\mathbb{I}_{2}$ , with $I_{2}$ the unit matrix. Thus, the spectrum of the QWZ model has two bands, the two eigenstates of $\hat{H}(\mathbf{k})$ , with energies

$$
\begin{array}{c} E _ {\pm} (k _ {x}, k _ {y}) = \pm | \mathbf {d} (k _ {x}, k _ {y}) | \\ = \pm \sqrt {\sin^ {2} (k _ {x}) + \sin^ {2} (k _ {y}) + (u + \cos (k _ {x}) + \cos (k _ {y})) ^ {2}}. \end{array}\tag{6.3}
$$

(6.4)

The spectrum of the QWZ model is depicted in Fig. 6.1.

There is an energy gap in the spectrum of the QWZ model, which closes at finetuned values of $u = +2, 0, -2$ . This is simple to show, since the gap closing requires $\mathbf{d}(\mathbf{k}) = 0$ at some $\mathbf{k}$ . From Eq. (6.2), $d_x(\mathbf{k}) = d_y(\mathbf{k}) = 0$ restricts us to four inequivalent points in the Brillouin zone:

\- if $u = -2$ : at $k_x = k_y = 0$ , the so-called $\Gamma$ point;

\- if $u = 0$ : at $k_x = 0, k_y = \pi$ and $k_x = \pi, k_y = 0$ , two inequivalent so-called $X$ points;

\- if $u = +2$ : at $k_x = \pi, k_y = \pi$ , the so-called $M$ point; note that $k_x = \pm \pi$ , $k_y = \pm \pi$ are all equivalent

In the vicinity of a gap closing point, called Dirac point, the dispersion relation has the shape of a Dirac cone, as seen in Fig. 6.1. For all other values of $u \neq -2, 0, 2$ , the spectrum is gapped, and thus it makes sense to investigate the topological properties of the system.

## 6.1.3 Chern Number of the QWZ Model

Although we calculated the Chern number of the corresponding pump sequence in the previous chapter, we show the graphical way to calculate the Chern number of the QWZ model. We simply count how many times the torus of the image of the Brillouin zone in the space of d contains the origin. To get some feeling about the not so trivial geometry of the torus, it is instructive to follow a gradual sweep of the Brillouin zone in Fig. 6.2. The parameter $u$ shifts the whole torus along the $d_z$ direction, thus as we tune it we also control whether the origin is contained inside it or not. For the QWZ model three situations can occur as depicted in Fig. 6.3. It can happen that the torus does not contain the origin, as in (a) and (d), and the Chern number is $Q = 0$ . This is the case for $|u| > 2$ . It can also happen that the origin is in the inside of the torus: a line from the origin to infinity will then inevitably pierce the torus. The first piercing can be from the blue side (outside) of the surface as in (b), with $Q = -1 -$ for $-2 < u < 0 -$ , or from the red side (inside) as in (c), with $Q = 1 -$ for $0 < u < 2$ .

![](images/f8287c86575a0e6c1cce8b16f153dafa0f2138c4e6420eceb4f4517fd438c1c8.jpg)

![](images/86cf0d963faf4719fb542afd3b35fd9b406acf0239e6ac14e3afa0f7ad89033c.jpg)

(c) u=2
![](images/404c847fdffa5b1fac2f8048e68c7a870a26d0f604f47f28434fb9b4ba46c931.jpg)

(d) u = -1.8
![](images/00210b98ba3a6ce2ee2d2471f358d60b629168749deb881aa5ee7524dce47391.jpg)
Fig. 6.1 The bulk dispersion relation of the QWZ model, for various values of u, as indicated in the plots. In (a)–(c), the gapless cases are shown, where the bulk gap closes at so-called Dirac points. In (d), with a generic value u = -1.8, the system is insulating

To summarize, the Chern number Q of the QWZ model is

$$
u <   - 2 \quad : \quad Q = 0;\tag{6.5a}
$$

$$
- 2 <   u <   0 \quad : \quad Q = - 1;\tag{6.5b}
$$

![](images/f01e40f97a0ef66f9e946731a60fbe5bee06c4a1e2c4ad8f2336b2d242e8cb88.jpg)

![](images/b3bfa557b0df792432f2d6d23f5c42c2c430effdc8121f12755087d5aa8d752d.jpg)

![](images/1e925b35bdeed7122483067ddbdac1442a1495d1c7c52a68b9377a711d8b46ee.jpg)

![](images/547d03ee1703e3684ac645a719c07389ef813203d19079edda65cd0310cc359c.jpg)

(e)
![](images/c3b91783e73f7725dfe39f93be2942c30d3775c2ae9ddc358d03557f8719a57b.jpg)

(f)
![](images/714865990c11009332ee32de1cfacce8baac08a9beb8788eb7e73f8dcf2d4241.jpg)

(g)
![](images/b3316e23702bdb7903b14b58ec4fc09b751b7274055de6268060da054e270e11.jpg)

$d_{z}$
(h)
![](images/db85d71411564e101cd943c88623ce405da3eab1ad0acbf72b8d42396e0daf7d.jpg)
Fig. 6.2 The surface $\mathbf{d}(\mathbf{k})$ for the QWZ model as $\mathbf{k}$ sweeps through the whole Brillouin zone. To illustrate how this surface is a torus the sweeping is done gradually with $u = 0$ . In (a) the image of the $k_y = -\pi$ line is depicted. In (b) the image for the region $k_y = -\pi \cdots - 0.5\pi$ , in (c) $k_y = -\pi \cdots - 0.25\pi$ , in (d) $k_y = -\pi \cdots - 0$ , in (e) $k_y = -\pi \cdots 0.25\pi$ , in (f) $k_y = -\pi \cdots 0.5\pi$ , in (g) $k_y = -\pi \cdots 0.75\pi$ and finally in (h) the image of the whole Brillouin zone is depicted and the torus is closed

![](images/f20741547f646de07a98a5c2d82a0e3c77f8b81cc8bb90f3e209534f2f4df00c.jpg)

![](images/fff569b6dd392c5848ee3c995c836ab39784e19c28d5b57f1d6ed9f057f5638b.jpg)

![](images/405bef6e2f48db2e0d47518155e9c37240f79a32c57796556954ce1dcde6db79.jpg)

![](images/b37b05a2da0de6e36040b9eec0b3a1dae08891fb0194bc84c81e79f0cc4a727e.jpg)
Fig. 6.3 The torus $\mathbf{d}(\mathbf{k})$ of the QWZ model for different values of $u$ . For clarity only the image of half of the Brillouin zone is shown. In (a) and (d) $u = \mp 2.2$ and the torus does not contain the origin hence $Q = 0$ . In (b) $u = -1$ , taking an infinite line from the origin along the positive $z$ axis we hit the blue side of the torus once hence $Q = -1$ . In (c) $u = 1$ , taking the infinite line in the negative $z$ direction we hit the red side of the torus thus $Q = 1$

$$
0 <   u <   2 \quad : \quad Q = + 1;\tag{6.5c}
$$

$$
2 <   u \quad : \quad Q = 0.\tag{6.5d}
$$

## 6.1.4 The Real-Space Hamiltonian

We obtain the full Hamiltonian of the Qi-Wu-Zhang model by inverse Fourier transform of the bulk momentum-space Hamiltonian, Eq. (6.1), as

$$
\begin{array}{l} \hat {H} = \sum_ {m _ {x} = 1} ^ {N _ {x} - 1} \sum_ {m _ {y} = 1} ^ {N _ {y}} \left(\left| m _ {x} + 1, m _ {y} \right\rangle \left\langle m _ {x}, m _ {y} \right| \otimes \frac {\hat {\sigma} _ {z} + i \hat {\sigma} _ {x}}{2} + h. c.\right) \\ \qquad + \sum_ {m _ {x} = 1} ^ {N _ {x}} \sum_ {m _ {y} = 1} ^ {N _ {y} - 1} \left(\left| m _ {x}, m _ {y} + 1 \right\rangle \left\langle m _ {x}, m _ {y} \right| \otimes \frac {\hat {\sigma} _ {z} + i \hat {\sigma} _ {y}}{2} + h. c.\right) \\ \qquad + u \sum_ {m _ {x} = 1} ^ {N _ {x}} \sum_ {m _ {y} = 1} ^ {N _ {y}} \left| m _ {x}, m _ {y} \right\rangle \left\langle m _ {x}, m _ {y} \right| \otimes \hat {\sigma} _ {z}. \end{array}\tag{6.6}
$$

As sketched in Fig. 6.4, the model describes a particle with two internal states hopping on a lattice where the nearest neighbour hopping is accompanied by an operation on the internal degree of freedom, and this operation is different for the hoppings along the x and y directions. In addition, there is a staggered onsite potential of strength u (if we envision the internal degree of freedom as representing two sites within the unit cell). Unlike in the case of the SSH model (Chap. 1), the real-space form of the QWZ Hamiltonian is not intuitive.

![](images/1b4dd4088b1cd119d754e8508ecc7223c5316cf3e404c6a590ea6e7e56b68f82.jpg)

![](images/c4b136d2563bdd3e72cd0fa2ffdb3a8920bb5cbbf1ac92050e0c73b7f91cd008.jpg)
Fig. 6.4 Sketch of the QWZ model: a particle with two internal states hopping on a square lattice. (a): The onsite potential and the hopping amplitudes are operators acting on the internal states. (b): A strip, with periodic boundary conditions along y, open boundaries along x (hopping amplitudes set to zero). Light blue/dark red highlights left/right edge region. (c): Upon Fourier transformation along y, the strip falls apart to an ensemble of one-dimensional Hamiltonians, indexed by $k_{y}$

## 6.2 Edge States

We constructed a Chern insulator using an adiabatic charge pump. As we saw in Chap. 4, charge pumps also induce energy eigenstates at the edge regions that cross from negative energy to positive energy bands, or vice versa. What do these energy eigenstates correspond to for the Chern insulator?

## 6.2.1 Dispersion Relation of a Strip Shows the Edge States

To see edge states, consider a strip of a two-dimensional insulator depicted in Fig. 6.4. Along y, we take periodic boundary conditions (close the strip to a cylinder), and go to the limit $N_{y} \rightarrow \infty$ . Along x, the strip is terminated by setting the hopping amplitudes to 0 (open boundary condition), and it consists of N sites. Translation invariance holds along y, so we can partially Fourier transform—only along y. After the Fourier transformation, the original Hamiltonian falls apart to a set of one-dimensional lattice Hamiltonians indexed by a continuous parameter $k_{y}$ , the wavenumber along y. For the QWZ model, Eq. (6.6), the $k_{y}$ -dependent Hamiltonian reads

$$
\begin{array}{l} \hat {H} (k _ {y}) = \sum_ {m _ {x} = 1} ^ {N _ {x} - 1} \left(| m _ {x} + 1 \rangle \langle m _ {x} | \otimes \frac {\hat {\sigma} _ {z} + i \hat {\sigma} _ {x}}{2} + h. c.\right) + \\ \sum_ {m _ {x} = 1} ^ {N _ {x}} | m _ {x} \rangle \langle m _ {x} | \otimes (\cos k _ {y} \hat {\sigma} _ {z} + \sin k _ {y} \hat {\sigma} _ {y} + u \hat {\sigma} _ {z}). \end{array}\tag{6.7}
$$

Note that this is the same dimensional reduction argument as before, but for a system with edges. Energy eigenstates $\left|\Psi(k_{y})\right\rangle$ of the strip fall into the categories of bulk states and edge states, much as in the one-dimensional case. All states are delocalized along y, but bulk states are also delocalized along x, while edge states are exponentially confined to the left (x = 1) or the right (x = N) edge. If we find energy eigenstates with energy deep in the bulk gap, they have to be edge states, and can be assigned to the left or the right edge.

An example for the dispersion relation of a strip is shown in Fig. 6.5, edge states on the left/right edge highlighted using light blue/dark red. We used the QWZ model, strip width N = 10, sublattice potential parameter u = -1.5, and the same practical definition of edge states as in the Rice-Mele model,

$$
\left| \Psi (k _ {y}) \right\rangle \text {   is   on   the   right   edge   } \Leftrightarrow \sum_ {m _ {x} = N - 1} ^ {N} \sum_ {\alpha \in \{A, B \}} \left| \left\langle \Psi (k _ {y}) \mid m _ {x}, \alpha \right\rangle \right| ^ {2} > 0. 6;\tag{6.8}
$$

$$
\left| \Psi (k _ {y}) \right\rangle \text {   is   on   the   left   edge   } \Leftrightarrow \sum_ {m _ {x} = 1} ^ {2} \sum_ {\alpha \in \{A, B \}} \left| \left\langle \Psi (k _ {y}) \mid m _ {x}, \alpha \right\rangle \right| ^ {2} > 0. 6.\tag{6.9}
$$

## 6.2.2 Edge States Conduct Unidirectionally

Notice the edge state branches of the dispersion relation of the QWZ strip, Fig. 6.5, which connect the lower and upper band across the bulk gap. They are the edge states of the pumped Rice-Mele model, Fig. 4.6, but we now look at them with a new eye. For the edge states in the QWZ model, $dE/dk_{y}$ corresponds to the group velocity along the edge. Thus, the dispersion relation tells us that particles in the QWZ model at low energy are confined either to the to left edge and propagate upwards, or to the right edge and propagate downwards.

The presence of one-way conducting edge state branches implies that the QWZ model is no longer, strictly speaking, an insulator. Because of the bulk energy gap, it cannot conduct (at low energies) between the left and right edges. However, it will conduct along each edge.

![](images/d7df97fb6322b1fb8dc54b1cb2beca6eb3e75042860809b74d1e21dd530dfcfc.jpg)

![](images/ecf4dd8d3c71d5652680530ba008706c0adfb636d357f9d21451bac8b5c7945d.jpg)
Fig. 6.5 Dispersion relation of a strip of the QWZ model, of width N = 10, and sublattice potential parameter u = -1.5. Because the strip is translation invariant along the edge, the wavenumber $k_{y}$ is a good quantum number, and the energy eigenvalues can be plotted (a) as a function of $k_{y}$ , forming branches of a dispersion relation. Light blue (dark red) highlights energies of edge states, whose wavefunction has over 60% weight on unit cells with $m_{x} \leq 2$ (with $m_{x} \geq N - 1$ ). These are parts of the Nth and $N + 1$ th branch, which split off from the bulk around $-\pi/4 < k_{y} < \pi/4$ , and have an avoided crossing with an exponentially small gap near $k_{y} = 0$ . We show the marginal position probability distribution of the Nth energy eigenstate, $P_{N}(m_{x}) = \sum_{\alpha \in \{A,B\}} \sum_{m_{y}} |\langle m_{x}, \alpha | \Psi_{N} \rangle|^{2}$ , for three values of $k_{y}$ . Depending on $k_{y}$ , this state can be an edge state on the right edge (b), on the left edge (c), or a bulk state (d)

## 6.2.3 Edge States and Edge Perturbation

We can use dimensional reduction and translate the discussion about the robustness of edge states from Chap. 4 to the edge states of the QWZ model. This treats the case where the Hamiltonian is modified in a way that it only acts in the edge regions, and is translation invariant along the edges. As an example, we introduce an extra, state-independent next-nearest neighbor hopping, and onsite potentials at the left and right edge of the sample. As Fig. 6.6 shows, this can modify the existing edge state branches, as well as create new edge state branches by deforming bulk branches. Including the new local terms the Hamiltonian of Eq. (6.7) is augmented to read

$$
\begin{array}{l} \hat {H} (k _ {y}) = \sum_ {m _ {x} = 1} ^ {N _ {x} - 1} \left(| m _ {x} + 1 \rangle   \langle m _ {x} | \otimes \frac {\hat {\sigma} _ {z} + i \hat {\sigma} _ {x}}{2} + h. c.\right) + \\ \sum_ {m _ {x} = 1} ^ {N _ {x}} | m _ {x} \rangle   \langle m _ {x} | \otimes (\cos k _ {y} \hat {\sigma} _ {z} + \sin k _ {y} \hat {\sigma} _ {y} + u \hat {\sigma} _ {z}) + \\ \sum_ {m _ {x} \in \{1, N \}} | m _ {x} \rangle   \langle m _ {x} | \otimes \hat {\mathbb {I}} _ {2} \left(\mu^ {(m _ {x})} + h _ {2} ^ {(m _ {x})} \cos 2 k _ {y}\right). \end{array}\tag{6.10}
$$

![](images/b1822071c7ed01baea5700372f62e07803ed13c8e288d91f1c4a672cfb0e31ce.jpg)

![](images/bcd8b42ad41f80aac499c4a50e8c88e633219a2564e3dca94f6f76141b9d92f6.jpg)
Fig. 6.6 Dispersion relation of a strip of the QWZ model, with edge states on the left/right edge highlighted in light blue/dark red. Top row: clean system. Bottom row: with extra next nearest neighbor hopping along y, on the left edge, with amplitude $h_{2}^{(1)} = 2$ and an onsite potential $\mu^{(N)}$ on the right edge. The value of the potential in (c) is $\mu^{(N)} = 0.5$ and in (d) it is $\mu^{(N)} = 1.5$ . In (e) the schematics of the considered perturbations is shown. The additional potential terms do not affect the bulk states but distort the edge modes and bring in new edge modes at energies that are in the bulk gap or above/below all bulk energies

Here, $\mu^{(1)/(N)}$ is the onsite potential on the left/right edge and $h_{2}^{(1)/(N)}$ describes a second nearest neighbor hopping on the left/right edge. If not stated otherwise these terms are considered to be zero.

In the top row of Fig. 6.6 the spectrum of strips without edge perturbations are depicted for Chern number Q = -1 (a) and Q = 0 (b) respectively. As we expected, a nonzero Chern number results in edge states, one on each edge. In (c) and (d) switching on perturbations, we see new edge states moving in to the gap. The onsite potential acts as an overall shift in energy on the states around $m_{x} = N$ , the second nearest neighbor hopping adds a considerable warping to the states localized around $m_{x} = 1$ . The deformations can change the number of edge states at a specific energy, but only by adding a pair of edge states with opposite propagation directions. This leaves the topological invariant unchanged.

## 6.2.4 Higher Chern Numbers by Coupling Layers

A systematic way to construct models with higher Chern numbers is to layer sheets of Chern insulators onto each other, as illustrated in Fig. 6.7. The single-particle Hilbert space of the composite system of D layers is a direct sum of the Hilbert spaces of the layers.

$$
\mathcal {H} _ {D} = \mathcal {H} _ {L 1} \oplus \mathcal {H} _ {L 2} \oplus \dots \oplus \mathcal {H} _ {L D}.\tag{6.11}
$$

![](images/3d01cfe38de77d713158bc192c1fc3c252701255d97891311de3a59f8f669ba1.jpg)
Fig. 6.7 Layering sheets of 2-dimensional insulators on top of each other is a way to construct a two-dimensional insulator with higher Chern numbers. For uncoupled layers, the Chern numbers can simply be summed to give the total Chern number of the 3-layer structure, $Q_{1} + Q_{2} + Q_{3}$ . Switching on coupling (hopping terms) between the sheets cannot change the Chern number as long as the bulk gap is not closed

The Hamiltonian, including a state-independent interlayer coupling with amplitude C, is

$$
\hat {H} _ {D} = \sum_ {d = 1} ^ {D} | d \rangle \langle d | \otimes \hat {H} _ {L d} + \sum_ {d = 1} ^ {D - 1} (| d + 1 \rangle \langle d | + | d \rangle \langle d + 1 | \otimes C \hat {\mathbb {I}} _ {2 N _ {x} N _ {y}},\tag{6.12}
$$

with $\hat{I}_{2N_{x}N_{y}}$ the unit operator on the Hilbert space of a single layer. The operators $\hat{H}_{Ld}$ we consider below are of the form of Eq. (6.10) with different values of u, and can have an overall real prefactor. In layer d the strength of the local edge potential is denoted as $\mu^{(1)/(N)_{d}}$ . As an example, the matrix of the Hamiltonian of a system with three coupled layers reads

$$
H _ {3} = \left[ \begin{array}{c c c} H _ {L 1} & C \mathbb {I} & 0 \\ C \mathbb {I} & H _ {L 2} & C \mathbb {I} \\ 0 & C \mathbb {I} & H _ {L 3} \end{array} \right].\tag{6.13}
$$

Numerical results for two and three coupled layers, with different Chern numbers in the layers, are shown in Fig. 6.8. The coupling of copropagating edge modes lifts the degeneracies, but cannot open gaps in the spectrum, except in the case of strong coupling. This is a simple consequence of the fact that an energy eigenstate has to be a single valued function of momentum. To open a gap, counterpropagating edge states have to be coupled. We achieve this by coupling layers of the QWZ model with opposite sign of the Chern number Q. For the case of two layers (Fig. 6.8, second row), this opens a gap in the spectrum. If there are three layers, there is a majority direction for the edge states, and so one edge state survives the coupling.

![](images/b62de48a5c9d3e39ecb218436bebd7495ec98e927e43614cdf99a31a206875ec.jpg)
Fig. 6.8 Dispersion relations of strips of multilayered QWZ model, $N = 10$ unit cells wide. In all cases the bulk Hamiltonian of the first layer $\hat{H}_{L1}$ is the QWZ Hamiltonian with $u = -1.2$ . To elucidate the interplay of the edge states we consider a finite potential acting on the edge $\mu_{L1}^{(1)/(N)} = 0.2 / -0.2$ . In the left column ((a) and (d)), we have two layers, with bulk $\hat{H}_{L2} = -\hat{H}_{L1}$ and edge onsite potential $\mu_{L2}^{(1)/(N)} = \mu_{L1}^{(1)/(N)}$ . In the middle ((b), (e)) and right ((c), (f)) columns, we have three layers. The third layer is characterized in the bulk by $\hat{H}_{L3} = \hat{H}_{L1}$ and on the edge by $\mu_{L3}^{(1)/(N)} = -\mu_{L1}^{(1)/(N)}$ . In (b) and (e) $\hat{H}_{L2} = 2\hat{H}_{L1}$ , while in (d) and (e) $\hat{H}_{L2} = -2\hat{H}_{L1}$ . For all four cases $\mu_{L2}^{(1)/(N)} = 0$ . Coupling in (d), (e) and (f) is uniform with magnitude $C = 0.4$

## 6.3 Robustness of Edge States

Up to now, we have considered clean edges, i.e., two-dimensional Chern insulators that were terminated by an edge (at $m_{x} = 1$ and $m_{x} = N$ ), but translationally invariant along the edge, along y. This translational invariance, and the resulting fact that the wavenumber $k_{y}$ is a good quantum number, was used for the definition of the topological invariant $N_{+} - N_{-}$ , which was the net number of edge bands propagating along the edge, equal to the bulk Chern number, Q. With disorder in the edge region that breaks translational invariance along y, we no longer have a good quantum number $k_{y}$ , and edge state bands are not straightforward to define. However, as we show in this section, the edge states must still be there in the presence of disorder, since disorder at the edges cannot close the bulk gap.

## 6.3.1 Smoothly Removing Disorder

Consider a finite sample of a Chern insulator, with a clean bulk part but disordered edge region, as depicted in Fig. 6.9. The bulk gap of the sample decreases due to disorder, but we suppose that it is not closed completely (just renormalized). Consider now a small part of the sample, containing some of the edge, indicated by the dotted rectangle on the right of Fig. 6.9. Although this is much smaller than the whole sample, it is big enough so that part of it can be considered as translation invariant “bulk”. Now in this small part of the sample, we adiabatically (and smoothly) deform the Hamiltonian in such a way that we set the disorder gradually to 0. This includes straightening the part of the open boundary of the sample that falls into the dotted rectangle, to a straight line. The deformation is adiabatic in the sense that the bulk gap is not closed in the process. Since this small part is a clean Chern insulator, with a bulk Chern number of Q, it can be deformed in such a way that the only edge states it contains are $|Q|$ states propagating counterclockwise (if Q > 0, say).

Fig. 6.9 A disordered sample of Chern insulator. The dotted lines indicate rectangular parts of the sample, where disorder can be turned off adiabatically to reveal edge states (indicated in black). Since particles cannot turn back (unidirectional, or chiral channels), and cannot go into the bulk (in the gap), they have to travel all the way on the perimeter of the disordered sample, coming back to the rectangular, clean part

![](images/75b92c4729f6acf64aed165868e002255a578c12297895ce3a77ffff8d26548d.jpg)

## 6.3.2 Unitarity: Particles Crossing the Clean Part Have to Go Somewhere

Consider a particle in an edge state in the small clean part of the sample, with energy deep inside the bulk gap $E \approx 0$ . What can its future be, as its time evolution follows the Schrödinger equation appropriate for this closed system? Since the edge state is a chiral mode, the particle has to propagate along the edge until it leaves the clean region. Because of unitarity, the particle cannot “stop” at the edge: that would mean that at the “stopping point”, the divergence of the particle current in the energy eigenstate is nonzero. In other words, the particle current that flows in the edge state has to flow somewhere. (Put differently, if the mode describing an edge state particle “stopped at the interface”, two particles, initially orthogonal, following each other in the mode, would after some time arrive to the same final state. This would break unitarity.) After leaving the clean part of the sample, the particle cannot propagate into the bulk, since its energy is deep in the bulk gap. The disorder in the clean part was removed adiabatically, and thus there are no edge states at the interface of the clean part and the disordered part of the sample, along the dashed line. The particle cannot turn back, as there are no edge states running “down” along the edge in the clean part. The only thing the particle can do is propagate along the edge, doing a full loop around the sample until it comes back to the clean part from below again.

The argument of the previous paragraph shows that even though the sample is disordered, there has to be a low energy mode that conducts perfectly (reflectionless) along the edge. Since at zero energy there are Q orthogonal states a particle can be in at the edge of the clean part of the sample, unitarity of the dynamics of the particles requires that all along the edge of the disordered sample there are Q orthogonal modes that conduct counterclockwise. There can be additional low energy states, representing trapped particles, or an equal number of extra edge states for particles propagating counterclockwise and clockwise. However, the total number of counterclockwise propagating edge modes at any part of the edge always has to be larger by Q than the number of clockwise propagating edge modes. Because the Hamiltonian is short range, our conclusions regarding the number of edge states at any point far from the deformed region have to hold independent of the deformation.

To be precise, in the argument above we have shown the existence of Q edge states all along the edge of the sample, except for the small part that was adiabatically cleaned from disorder. One way to finish the argument is by considering another part of the sample. If we now remove the disorder adiabatically only in this part, we obtain the existence of the edge modes in parts of the sample including the original dotted rectangle, which was not covered by the argument of the previous paragraph.

## Problems

## 6.1 Phase diagram of the anisotropic QWZ model

The lattice Hamiltonian of the QWZ model is provided in Eq. $(6.6)$ . Consider the anisotropic modification of the Hamiltonian when the first term of Eq. $(6.6)$ , describing the hopping along the x axis, is multiplied by a real number A. Plot the phase diagram of this model, that is, evaluate the Chern number as a function of two parameters u and A.

## Chapter 7 Continuum Model of Localized States at a Domain Wall

So far, we have discussed edge states in lattice models, in which the states live on discrete lattice sites, and the Hamiltonian governing the physics is a matrix. In this chapter, we argue that in certain cases, it is also possible to describe these states via a continuum model, in which the states live in continuous space, and the Hamiltonian is a differential operator. One benefit of such a continuum description is that it allows one to use the vast available toolkit of differential equations for solid-state problems in general, including the description of topologically protected states in particular. Another interesting aspect of these continuum models is their strong similarity with the Dirac equation describing relativistic fermions. A limitation of the continuum models is that their validity is restricted to narrow windows in momentum and energy; typically they are applied in the vicinities of band edges. Here, we obtain the continuum differential equations for three basic lattice models: the one-dimensional monatomic chain, the one-dimensional SSH model, and the two-dimensional QWZ model. In the cases of the SSH and QWZ models, the resulting equations will be used to analytically characterize the localized states appearing at boundaries between regions with different topological invariants. Even though in the entire chapter we build our discussion on the three specific lattice models, the applicability of the technique introduced here, called envelope-function approximation, is more general and widely used to describe electronic states in various crystalline solids.

## 7.1 One-Dimensional Monatomic Chain in an Electric Potential

We use this minimal lattice model to illustrate the basic concepts of the envelope-function approximation (EFA) [3], the technique that allows us to find the continuum versions of our lattice models. Of course, the one-dimensional monatomic chain does not host topologically protected states.

## 7.1.1 The Model

We take a long lattice with $N \gg 1$ unit cells (or sites) without an internal degree of freedom, with periodic boundary conditions, and a negative hopping amplitude t < 0. We consider the situation when the electrons are subject to an inhomogeneous electric potential $V(x)$ . This setup is pictured in Fig. 7.1. The lattice Hamiltonian describing this inhomogeneous system reads

$$
H _ {\mathrm{i}} = H + V,\tag{7.1}
$$

where

$$
H = t \sum_ {m = 1} ^ {N} | m \rangle \langle m + 1 | + h. c.,\tag{7.2}
$$

$$
V = \sum_ {m = 1} ^ {N} V _ {m} | m \rangle \langle m |,\tag{7.3}
$$

with $V_{m} = V(x = m)$ . Our aim is to construct a continuum model that accurately describes the low-energy eigenstates of this lattice Hamiltonian.

Before discussing the inhomogeneous case incorporating $V(x)$ , focus first on the homogeneous system. The bulk momentum-space Hamiltonian is a scalar ( $1 \times 1$ matrix) in this model, since there is no integral degree of freedom associated to the unit cell; it reads

$$
H (k) = \epsilon (k) | k \rangle \langle k |,\tag{7.4}
$$

where $\epsilon(k)$ is the electronic dispersion relation:

$$
\epsilon (k) = - 2 | t | \cos k.\tag{7.5}
$$

![](images/0470134ed3fe139d90dd9b68a73080b816275edd0b01ee39ff8740d0e4d01ab4.jpg)
Fig. 7.1 One-dimensional monatomic chain in an inhomogeneous potential $V(x)$

The low-energy part of the dispersion relation is located around zero momentum, and is approximated by a parabola:

$$
\epsilon (k) \approx - 2 | t | + | t | k ^ {2} = \epsilon_ {0} + \frac {k ^ {2}}{2 m ^ {*}},\tag{7.6}
$$

where we introduced the minimum energy of the band $\epsilon_{0} = -2|t|$ , and the effective mass $m^{*} = \frac{1}{2|t|}$ characterizing the low-energy part of the dispersion relation. (Using proper physical units, the effective mass would have the form $m^{*} = \hbar^{2}/(2|t|a^{2})$ , with a being the lattice constant.) For simplicity, we suppress $\epsilon_{0}$ in what follows; that is, we measure energies with respect to $\epsilon_{0}$ .

## 7.1.2 Envelope-Function Approximation

Our goal is to find the low-energy eigenstates of the inhomogeneous lattice Hamiltonian $H_{i}$ . The central proposition of the EFA, applied to our specific example of the one-dimensional monatomic chain, says that it is possible to complete this goal by solving the simple continuum Schrödinger equation

$$
H _ {\mathrm{EFA}} \varphi (x) = E \varphi (x),\tag{7.7}
$$

where the envelope-function Hamiltonian $H_{EFA}$ has a very similar form to the free-electron Hamiltonian with the electric potential V:

$$
H _ {\mathrm{EFA}} = \frac {\hat {p} ^ {2}}{2 m ^ {*}} + V (x).\tag{7.8}
$$

Here $\hat{p} = -i\partial_{x}$ is the usual real-space representation of the momentum operator, and the function $\varphi(x)$ is usually called envelope function. Note the very simple relation between the low-energy dispersion in Eq. (7.6) and the kinetic term in the EFA Hamiltonian (7.8): the latter can be obtained from the former by substituting the momentum operator $\hat{p}$ in place of the momentum k.

Before formulating the EFA proposition more precisely, we introduce the concept of a spatially slowly varying envelope function. We say that $\varphi(x)$ is spatially slowly varying, if its Fourier transform

$$
\tilde {\varphi} (q) = \int_ {0} ^ {N} d x \frac {e ^ {- i q x}}{\sqrt {N}} \varphi (x)\tag{7.9}
$$

is localized to the $|q| \ll \pi$ region, i.e., to the vicinity of the center of the Brillouin zone.

With this definition at hand, we can formulate the EFA proposition. Consider the inhomogeneous one-dimensional monatomic chain described by $H_{i}$ . Assume that $\varphi(x)$ is a spatially slowly varying eigenfunction of $H_{EFA}$ with eigenvalue E. Then,

the state $|\psi\rangle$ defined on the lattice via

$$
| \psi \rangle = \sum_ {m = 1} ^ {N} \varphi (x = m) | m \rangle .\tag{7.10}
$$

is approximately an eigenstate of the lattice Hamiltonian $H_{\mathrm{i}}$ with eigenvalue $E$ . The proof follows below in Sect. 7.1.3.

Note that if the envelope function $\varphi(x)$ fulfils the normalization condition

$$
\int_ {0} ^ {N} d x | \varphi (x) | ^ {2} = 1,\tag{7.11}
$$

then, for a long lattice $N \gg 1$ , the lattice state $|\psi\rangle$ will also be normalized to a good accuracy:

$$
\langle \psi | \psi \rangle \approx 1.\tag{7.12}
$$

It is important to point out two possible interpretations of the lattice state $|\psi\rangle$ introduced in Eq. (7.10). (1) The state $|\psi\rangle$ can be interpreted as the zero-momentum band-edge eigenstate $|k=0\rangle = \frac{1}{\sqrt{N}}\sum_{m=1}^{N}|m\rangle$ of the homogeneous system, modulated by the envelope function $\varphi(x)$ restricted to the lattice-site positions x=m. (2) The state $|\psi\rangle$ can also be interpreted as a wave packet, composed of those eigenstates $|k\rangle$ of the homogeneous lattice Hamiltonian H that have wave numbers k close to the band-edge wave number, the latter being zero in this case. To see this, we first Fourier-decompose $\varphi(x)$ :

$$
\varphi (x) = \sum_ {k \in \mathrm{BZ}} \tilde {\varphi} (k) \frac {e ^ {i k x}}{\sqrt {N}} \approx \sum_ {k} ^ {\prime} \tilde {\varphi} (k) \frac {e ^ {i k x}}{\sqrt {N}}.\tag{7.13}
$$

In the approximate equality, we used the fact that $\varphi(x)$ is spatially slowly varying, i.e., its Fourier transform $\tilde{\varphi}(k)$ is localized to the central part of the Brillouin zone, and introduced the notation $\sum_{k}^{\prime}$ for a wave-number sum that goes only for the central part of the Brillouin zone. By inserting Eq. (7.13) to Eq. (7.10), we find

$$
| \psi \rangle = \sum_ {m = 1} ^ {N} \left(\sum_ {k} ^ {\prime} \tilde {\varphi} (k) \frac {e ^ {i k m}}{\sqrt {N}}\right) | m \rangle = \sum_ {k} ^ {\prime} \tilde {\varphi} (k) | k \rangle .\tag{7.14}
$$

That is, $|\psi\rangle$ is indeed a packet of plane waves with small wave numbers.

## 7.1.3 Envelope-Function Approximation: The Proof

To prove the EFA proposition, we calculate $H_{\mathrm{i}}|\psi \rangle$ and utilize the spatially-slowly-varying condition on $\varphi (x)$ . Start with the contribution of the bulk Hamiltonian $\hat{H}$ : utilizing Eq. (7.14), we find

$$
H \left| \psi \right\rangle = \left[ \sum_ {k \in \mathrm{BZ}} \epsilon (k) \left| k \right\rangle \left\langle k \right| \right] \left[ \sum_ {q} ^ {\prime} \tilde {\varphi} (q) \left| q \right\rangle \right].\tag{7.15}
$$

Performing the scalar product yields

$$
H \left| \psi \right\rangle = \sum_ {q} ^ {\prime} \epsilon (q) \tilde {\varphi} (q) \left| q \right\rangle .\tag{7.16}
$$

Using the fact that the $q$ sum goes for the central part of the Brillouin zone, where the dispersion relation $\epsilon(q)$ is well approximated by a parabola, we find

$$
H \left| \psi \right\rangle \approx \sum_ {q} ^ {\prime} \frac {q ^ {2}}{2 m ^ {*}} \tilde {\varphi} (q) \left| q \right\rangle .\tag{7.17}
$$

Utilizing the definition of the plane wave $|q\rangle$ , we obtain

$$
H \left| \psi \right\rangle \approx \sum_ {q} ^ {\prime} \frac {q ^ {2}}{2 m ^ {*}} \tilde {\varphi} (q) \sum_ {m = 1} ^ {N} \frac {e ^ {i q m}}{\sqrt {N}} \left| m \right\rangle ,\tag{7.18}
$$

which can be rewritten as

$$
\begin{array}{l} H \left| \psi \right\rangle \approx \sum_ {m = 1} ^ {N} \left[ \sum_ {q} ^ {\prime} \frac {q ^ {2}}{2 m ^ {*}} \tilde {\varphi} (q) \frac {e ^ {i q x}}{\sqrt {N}} \right] _ {x = m} \left| m \right\rangle \\ = \sum_ {m = 1} ^ {N} \left[ - \frac {1}{2 m ^ {*}} \partial_ {x} ^ {2} \sum_ {q} ^ {\prime} \tilde {\varphi} (q) \frac {e ^ {i q x}}{\sqrt {N}} \right] _ {x = m} \left| m \right\rangle \\ = \sum_ {m = 1} ^ {N} \left[ \frac {\hat {p} ^ {2}}{2 m ^ {*}} \varphi (x) \right] _ {x = m} \left| m \right\rangle . \end{array}\tag{7.19}
$$

(7.20)

(7.21)

Continue with the contribution of the potential V. Using Eqs. $(7.3)$ and $(7.10)$ , we find

$$
\begin{array}{l} V \left| \psi \right\rangle = \left[ \sum_ {m = 1} ^ {N} V _ {m} \left| m \right\rangle \left\langle m \right| \right] \left[ \sum_ {m ^ {\prime} = 1} ^ {N} \varphi (m ^ {\prime}) \left| m ^ {\prime} \right\rangle \right] = \sum_ {m = 1} ^ {N} V _ {m} \varphi (m) \left| m \right\rangle \\ = \sum_ {m = 1} ^ {N} [ V (x) \varphi (x) ] _ {x = m} \left| m \right\rangle . \end{array}\tag{7.22}
$$

Summing up the contribution (7.21) of $H$ and the contribution (7.22) of $V$ , we find

$$
(H + V) | \psi \rangle \approx \sum_ {m = 1} ^ {N} [ H _ {\mathrm{EFA}} \varphi (x) ] _ {x = m} | m \rangle = \sum_ {m = 1} ^ {N} [ E \varphi (x) ] _ {x = m} | m \rangle\tag{7.23}
$$

$$
= E \sum_ {m = 1} ^ {N} \varphi (m) | m \rangle = E | \psi \rangle ,\tag{7.24}
$$

which concludes the proof.

## 7.2 The SSH Model and the One-Dimensional Dirac Equation

To illustrate how the EFA captures topologically protected bound states in one-dimensional, we use the SSH model described in detail in Chap. 1. The model is visualized in Fig. 1.1. The bulk Hamiltonian is characterized by two parameters, the intra-cell and inter-cell hopping amplitudes v, w > 0, respectively. The bulk lattice Hamiltonian of the SSH model is given in Eq. (1.1), whereas the bulk momentum-space Hamiltonian is given in Eq. (1.14). We have seen that the $(v, w)$ parameter space is separated to two adiabatically connected partitions by the v = w line. In Sect. 1.5.5, we have also seen that localized zero-energy states appear at a domain wall between two half-infinite homogeneous regions, if the two regions have different bulk topological invariants; that is, if the sign of v - w is different at the two sides of the domain wall.

This is the phenomenon that we address in this section: we show that an analytical description of such localized states can be given using the EFA. First, we discuss the electronic dispersion relation of the metallic $(v = w)$ and nearly metallic $(|v - w| \ll |v + w|)$ homogeneous SSH model. Second, we obtain the Dirac-type differential equation providing a continuum description for the inhomogeneous SSH model (see Fig. 7.2) for the energy range in the vicinity of the bulk band gap. Finally, we solve that differential equation to find the localized zero-energy states at a domain wall. Remarkably, the analytical treatment remains useful even if the spatial structure of the domain wall is rather irregular.

![](images/90029e4c81769c68ba008922973ae0eedb000bed895a15135d2eaa5ede17e945.jpg)
Fig. 7.2 Inhomogeneous intracell hopping and domain walls in the SSH model. The dashed ellipse denotes the unit cell. The dashed line connecting the edges of the chain denotes the periodic boundary condition

## 7.2.1 The Metallic Case

First, consider the metallic homogeneous SSH model, where v = w. The dispersion relation is shown as the blue solid line in Fig. 1.2c. The filled and empty bands touch at the end of the Brillouin zone, at $k = k_{0} \equiv \pi$ . Figure 1.2c shows that in the vicinity of that touching point, commonly referred to as a Dirac point, the dispersion relations are linear functions of the relative wave vector $q = k - k_{0}$ . The slope of these linear functions, corresponding to the group velocity of the electrons, can be determined, e.g. by Taylor-expanding the bulk momentum-space Hamiltonian $\hat{H}(k) = (v + w \cos k)\hat{\sigma}_{x} + w \sin k\hat{\sigma}_{y}$ , see Eq. (1.10), to first order in q:

$$
\hat {H} (k _ {0} + q) \approx - w q \hat {\sigma} _ {y}, (v = w).\tag{7.25}
$$

which indeed has a linear dispersion relation,

$$
E _ {\pm} (q) = \pm w q.\tag{7.26}
$$

The eigenstates of the linearized Hamiltonian (7.25) are

$$
\psi_ {\pm} (q) = \frac {1}{\sqrt {2}} \binom{1}{\mp i}.\tag{7.27}
$$

Note that the dispersion relation of the Dirac equation of fermions with zero mass is

$$
E _ {\pm} (k) = \pm \hbar k c,\tag{7.28}
$$

where $\hbar$ is the reduced Planck's constant and $c$ is the speed of light. Comparing Eqs. (7.26) and (7.28), we conclude that the dispersion of the metallic SSH model is analogous to that of massless Dirac fermions, and the hopping amplitude of the metallic SSH model plays the role of $\hbar c$ . Because of the similarity of the dispersions (7.26) and (7.28), the linearized Hamiltonian (7.25) is often called a massless Dirac Hamiltonian.

At this point, the linearization of the bulk momentum-space Hamiltonian of the SSH model does not seem to be a particularly fruitful simplification: to obtain the dispersion relation and the eigenstates, a $2 \times 2$ matrix has to be diagonalized, no matter if the linearization has been done or not. However, linearizing the Hamiltonian is the first step towards the EFA, as discussed below.

## 7.2.2 The Nearly Metallic Case

Now consider a homogeneous, insulating SSH model that is nearly metallic; that is, the scale of the energy gap $|v-w|$ opened at $k_{0}$ is significantly smaller than the scale of the band width $v+w$ . An example is are shown in Fig. 1.2b, where the dispersion relation is plotted for the parameter values v=1 and w=0.6.

We wish to describe the states close to the band gap located around zero energy. Hence, again, we can use the approximate bulk momentum-space Hamiltonian obtained via linearization in the relative momentum q:

$$
\hat {H} (k _ {0} + q) \approx M \hat {\sigma} _ {x} - w q \hat {\sigma} _ {y},\tag{7.29}
$$

where we defined M = v - w. The dispersion relation reads

$$
E _ {\pm} (q) = \pm \sqrt {M ^ {2} + w ^ {2} q ^ {2}}.\tag{7.30}
$$

The (unnormalized) eigenstates of the linearized Hamiltonian $(7.29)$ have the form

$$
\psi_ {\pm} (q) = \binom{M + i w q}{E _ {\pm} (q)}.\tag{7.31}
$$

Note that the dispersion relation of the Dirac equation for fermions with finite mass $\mu \neq 0$ reads

$$
E _ {\pm} (k) = \pm \sqrt {\mu^ {2} c ^ {4} + \hbar^ {2} k ^ {2} c ^ {2}}.\tag{7.32}
$$

Therefore, the parameter M = v - w of the SSH model plays the role of the mass-related term $\mu c^{2}$ of the relativistic dispersion relation (7.32), and the linearized Hamiltonian (7.29) is often called a massive Dirac Hamiltonian.

## 7.2.3 Continuum Description of the Nearly Metallic Case

We are mostly interested in a continuum description of the zero-energy localized states formed at a domain wall between two topologically distinct regions. For simplicity and concreteness, consider the case when the domain wall is created so that the intra-cell hopping amplitude v varies in space while the inter-cell one is constant, as shown in Fig. 7.2.

In what follows, we will focus on one of the two domain walls shown in Fig. 7.2. The inhomogeneous Hamiltonian has the form

$$
H _ {\mathrm{i}} = \sum_ {m = 1} ^ {N} v _ {m} (| m, B \rangle \langle m, A | + h. c.) + w \sum_ {m = 1} ^ {N} (| m, B \rangle \langle m + 1, A | + h. c.),\tag{7.33}
$$

where $v_{m} = v(x = m)$ and $v(x) \geq 0$ is a continuously varying function of position, which takes the constant value $v_{-}(v_{+})$ far on the left (right) from the domain wall.

We also assume that the local Hamiltonian is a nearly metallic SSH Hamiltonian everywhere in space. That is, $|v(x) - w| \ll v(x) + w$ . This ensures that the local band gaps $|v_{\pm} - w|$ on the two sides of the domain wall are much smaller than the local band widths $v_{\pm} + w$ .

Based on our experience with the EFA in the inhomogeneous one-dimensional monatomic chain (see Sect. 7.1), now we construct the EFA proposition corresponding to this inhomogeneous, nearly metallic SSH model. Recall that in the former case, we obtained the EFA Hamiltonian by (i) Taylor-expanding the bulk momentum-space Hamiltonian around the wave vector corresponding to the band extremum (that was $k_{0} = 0$ in Sect. 7.1), (ii) replacing the relative wave vector q with the momentum operator $\hat{p} = -i\partial_{x}$ , and (iii) incorporating the inhomogeneity of the respective parameter, which was the on-site potential $V(x)$ in that case. The same procedure, applied now for the SSH model with a first-order Taylor expansion, yields the following EFA Hamiltonian:

$$
H _ {\mathrm{EFA}} = M (x) \hat {\sigma} _ {x} - w \hat {p} \hat {\sigma} _ {y}.\tag{7.34}
$$

The EFA proposition is then formulated as follows. Assume that $\varphi(x) = (\varphi_{A}(x), \varphi_{B}(x))$ is a spatially slowly varying eigenfunction of $H_{EFA}$ in Eq. (7.34), with eigenvalue E. Then, the state $|\psi\rangle$ defined on the lattice via

$$
| \psi \rangle = \sum_ {m = 1} ^ {N} \sum_ {\alpha = A, B} \varphi_ {\alpha} (m) e ^ {i k _ {0} m} | m, \alpha \rangle\tag{7.35}
$$

is approximately an eigenstate of the lattice Hamiltonian $H_{i}$ with energy E.

Note that, in analogy with Eq. (7.14), the lattice state $|\psi\rangle$ can be reformulated as a wave packet formed by plane waves from the vicinity of the band-edge momentum $k_{0}$ :

$$
| \psi \rangle \approx \sum_ {q} ^ {\prime} \sum_ {\alpha = A, B} \tilde {\varphi} _ {\alpha} (q) | k _ {0} + q \rangle \otimes | \alpha \rangle .\tag{7.36}
$$

## 7.2.4 Localized States at a Domain Wall

Having the envelope-function Schrödinger equation

$$
\left[ M (x) \hat {\sigma} _ {x} - w \hat {p} \hat {\sigma} _ {y} \right] \varphi (x) = E \varphi (x)\tag{7.37}
$$

at hand, we can study the domain wall between the two topologically distinct regions. First, we consider a step-type domain wall, defined via

$$
M (x) = \left\{ \begin{array}{l l} M _ {0} & \text { if } x > 0, \\ - M _ {0} & \text { if } x <   0 \end{array} \right.,\tag{7.38}
$$

and $M_{0} > 0$ , as shown in Fig. 7.3a.

We wish to use the EFA Schrödinger equation $(7.37)$ to establish the zero-energy states localized to the domain wall, which were revealed earlier in the lattice SSH model. That is, we look for evanescent solutions of Eq. $(7.37)$ on both sides of the domain wall, and try to match them at the domain wall at x = 0. For the x > 0 region, our evanescent-wave Ansatz reads

$$
\varphi_ {x > 0} (x) = \binom{a}{b} e ^ {- \kappa x}\tag{7.39}
$$

![](images/604353fb97a2796f0c1cfb4acf9f21827551b000b8865ee00c5cb8eb466eb07b.jpg)

![](images/81144e210765e41ceba8c5840f990424f621941d74ff8b16ac7e52d1509a5809.jpg)
Fig. 7.3 (a) Step-like and (b) irregular spatial dependence of the mass parameter $M(x)$ of the one-dimensional Dirac equation

with $\kappa > 0$ . Substituting this to Eq. (7.37) yields a quadratic characteristic equation for the energy E, having two solutions

$$
E _ {\pm} = \pm \sqrt {M _ {0} ^ {2} - w ^ {2} \kappa^ {2}}.\tag{7.40}
$$

The corresponding unnormalized spinors read

$$
\binom{a _ {\pm}}{b _ {\pm}} = \binom{\frac {M _ {0} - w \kappa}{E _ {\pm}}}{1}.\tag{7.41}
$$

An analogous Ansatz for the $x < 0$ region is

$$
\varphi_ {x <   0} (x) = \binom{c}{d} e ^ {\kappa x}\tag{7.42}
$$

with $\kappa > 0$ , yielding the same energies as in Eq. (7.40), and the spinors

$$
\binom{c _ {\pm}}{d _ {\pm}} = \binom{\frac {- M _ {0} + w \kappa}{E _ {\pm}}}{1}.\tag{7.43}
$$

Now consider an energy eigenstate with a given energy $E$ . For clarity, set $M_0 > E \geq 0$ . The (unnormalized) envelope function of the energy eigenstate has the form

$$
\varphi (x) = \varphi_ {x > 0} (x) \Theta (x) + C \varphi_ {x <   0} (x) \Theta (- x),\tag{7.44}
$$

where $\varphi_{x<0}$ and $\varphi_{x>0}$ should be evaluated by replacing $E_{\pm}\mapsto E$ and $\kappa\mapsto\frac{\sqrt{M_{0}^{2}-E^{2}}}{w}$ , and C is a yet unknown parameter to be determined from the boundary conditions at the domain wall. The envelope function (7.44) is an eigenstate of the EFA Hamiltonian with energy E if the boundary condition that the wave function is continuous at x=0, that is,

$$
\varphi_ {x <   0} (0) = C \varphi_ {x > 0} (0),\tag{7.45}
$$

is fulfilled. Note that in our case, the Dirac equation is a first-order differential equation and therefore there is no boundary condition imposed on the derivative of the wave function. From the second component of Eq. (7.45), we have C = 1. From the first component, we have $M_{0} - w\kappa = -M_{0} + w\kappa$ , implying $M_{0} = w\kappa$ and thereby E = 0. The same result is obtained if the range $-M_{0} < E \leq 0$ of negative energies is considered. Hence we conclude that the zero-energy state at the domain wall does appear in the continuum model of the inhomogeneous SSH chain, as expected.

Let us also determine the coefficients a and c describing this localized state:

$$
a = \lim _ {E \to 0} \frac {M _ {0} - \sqrt {M _ {0} ^ {2} - E ^ {2}}}{E} = 0,\tag{7.46}
$$

and similarly, c = 0. These imply that the localized state is completely sublattice-polarized, i.e., it lives on the B sublattice, and therefore it is its own chiral partner. Considering a similar mass profile with negative $M_{0}$ , we would have found that the localized state lives on the A sublattice. These properties are in line with our expectations drawn from the lattice SSH model.

A further characteristic property of the localized state is its localization length; from our continuum model, we have an analytical result for that:

$$
\frac {1}{\kappa} = \frac {w}{M _ {0}}.\tag{7.47}
$$

(In physical units, that is $\frac{1}{\kappa} = \frac{w}{M_{0}}a$ .) Recall that we are constrained to the nearly metallic regime $w \gg M_{0} = v_{+} - w$ ; together with Eq. (7.47), this implies that the localization length is much larger than one (that is, the lattice constant). This is reassuring: it means that the envelope function $\varphi(x)$ is spatially slowly varying, hence is within the range of validity of the EFA.

The result (7.47) can be compared to the corresponding result for the SSH lattice model. Equation (1.50) provides the localization length $\xi$ of an edge state in a disordered SSH model, which corresponds to the localization length $\frac{1}{\kappa}$ obtained above. Taking the disorder-free special case of Eq. (1.50), we have

$$
\xi = \frac {1}{\log \frac {w}{v}} = \frac {1}{\log \frac {w}{w + (v - w)}} \approx \frac {w}{w - v}.\tag{7.48}
$$

As we are making a comparison to the nearly metallic $v \approx w$ case considered in this section, we could approximate $\xi$ in Eq. (7.48) using a leading-order Taylor expansion in the small quantity $(w - v) / w$ . The approximate result (7.48) is in line with Eq. (7.47) obtained from the continuum model.

A further interesting fact is that the existence of the localized state is not constrained to the case of a sharp, step-like domain wall described above. The simple spinor structure found above also generalizes for less regular domain walls. To see this, consider an almost arbitrary one-dimensional spatial dependence $M(x)$ of the mass, illustrated in Fig. 7.3b, with the only condition that M changes sign between the half-planes x < 0 and x > 0, i.e., $M(x \to -\infty) < 0$ and $M(x \to \infty) > 0$ . We claim that there exists a zero-energy solution of the corresponding one-dimensional Dirac equation that is localized to the domain wall and has the

envelope function

$$
\varphi (x) = \binom{0}{1} f (x).\tag{7.49}
$$

To prove this claim, insert this wave function $\varphi(x)$ to the one-dimensional Dirac equation and substitute E = 0 therein. This procedure results in the single differential equation $\partial_{x}f(x) = -\frac{M(x)}{w}$ , implying that Eq. (7.49) is indeed a zero-energy eigenstate of the envelope-function Hamiltonian if the function f has the form

$$
f (x) = \operatorname{const} \times e ^ {- \frac {1}{w} \int_ {0} ^ {x} d x ^ {\prime} M \left(x ^ {\prime}\right)}.\tag{7.50}
$$

Furthermore, the asymptotic conditions of the mass $M(x)$ guarantee that this envelope function decays as $x \to \pm\infty$ , and therefore is localized at the domain wall.

## 7.3 The QWZ Model and the Two-Dimensional Dirac Equation

We have introduced the QWZ model as an example for a two-dimensional Chern insulator in Chap. 6. The lattice Hamiltonian of the model is given in Eq. (6.6), whereas the bulk momentum-space Hamiltonian is given in Eq. (6.1). The dispersion relation is calculated in Eq. (6.4), and examples of it are shown in Fig. 6.1.

Recall that the model has a single parameter u, and the Chern number of the model is determined by the value of u via Eq. $(6.5)$ . Similarly to the case of the SSH model in one dimension, one can consider a domain wall between locally homogeneous regions of the QWZ model that have different Chern numbers. Just as the edge of a strip, such a domain wall can support topologically protected states that propagate along the domain wall but are localized at the domain wall in the transverse direction. The number and propagation direction of those states is determined by the magnitude and sign of the difference of the Chern numbers in the two domains, respectively. In this section, we use the EFA to provide a continuum description of such states.

## 7.3.1 The Metallic Case

First, consider the metallic cases of the QWZ model; that is, when the band structure has no energy gap. In particular, we will focus on the u = -2 case. The corresponding band structure is shown in Fig. 6.1a. The two bands touch at $\mathbf{k} = (0, 0)$ , and form a Dirac cone at that Dirac point.

To describe excitations in the vicinity of the Dirac point of such a metal, it is sufficient to use a linearized approximation of the QWZ Hamiltonian $\hat{H}(\mathbf{k})$ that is obtained via a Taylor expansion of $\hat{H}(\mathbf{k})$ up to first order in the k-space location $q = k - k_{0}$ measured from the Dirac point $k_{0}$ . In the case u = -2, the Dirac point is $\mathbf{k}_{0} = (0, 0)$ , and the linearized Hamiltonian reads

$$
\hat {H} (\mathbf {k} _ {0} + \mathbf {q}) \approx q _ {x} \hat {\sigma} _ {x} + q _ {y} \hat {\sigma} _ {y}.\tag{7.51}
$$

The dispersion relation is $E_{\pm}(\mathbf{q}) = \pm q$ ; again, this is analogous to that of the massless Dirac equation Eq. (7.28).

## 7.3.2 The Nearly Metallic Case

Now consider a QWZ insulator that is nearly metallic: $u \approx -2$ . The dispersion relation for u = -1.8 is shown in Fig.6.1d. In the vicinity of the metallic state, as seen in the figure, a small gap opens in the band structure at the Dirac point $\mathbf{k}_{0} = (0, 0)$ .

The states and the band structure around $k_{0}$ can again be described by a linearized approximation of the QWZ Hamiltonian $\hat{H}(\mathbf{k}_{0} + \mathbf{q})$ in q:

$$
\hat {H} (\mathbf {k} _ {0} + \mathbf {q}) \approx M \hat {\sigma} _ {z} + q _ {x} \hat {\sigma} _ {x} + q _ {y} \hat {\sigma} _ {y},\tag{7.52}
$$

where we defined the parameter $M = u + 2$ . The dispersion relation reads

$$
E _ {\pm} (\mathbf {q}) = \pm \sqrt {M ^ {2} + q ^ {2}}.\tag{7.53}
$$

A comparison with the relativistic dispersion relation (7.32) reveals that the parameter M of the QWZ model plays the role of $\mu c^{2}$ ; hence M can be called the mass parameter.

## 7.3.3 Continuum Description of the Nearly Metallic Case

We have discussed that the QWZ lattice with an inhomogeneous u parameter might support topologically protected states at boundaries separating locally homogeneous regions with different Chern numbers. Similarly to the one-dimensional SSH model treated above, these localized states can be described analytically, using the envelope function approximation (EFA), also in the two-dimensional QWZ model. In the rest of this chapter, we focus on the nearly metallic case where the inhomogeneous $u(x,y)$ is in the vicinity of -2 (i.e., $|M(x,y)| = |u(x,y) + 2| \ll 1$ ), in which case the low-energy excitations are expected to localize in Fourier space around the band extremum point $k_{0} = (0, 0)$ (see Fig. 6.1a and d). Here we obtain the EFA

Schrödinger-type equation, which resembles the two-dimensional Dirac equation, and in the next subsection we provide its localized solutions for simple domain-wall arrangements.

The considered lattice is inhomogeneous due to the spatial dependence of the parameter $u(x,y)$ . In the tight-binding lattice model, we denote the value of u in unit cell $\mathbf{m} = (m_x, m_y)$ as $u_{\mathbf{m}} = u(x = m_x, y = m_y)$ , and correspondingly, we introduce the local mass parameter via $M(x, y) = u(x, y) + 2$ and $M_{\mathbf{m}} = M(x = m_x, y = m_y)$ .

The EFA Hamiltonian can be constructed the same way as in Sects. 7.1 and 7.2.3. The bulk momentum-space Hamiltonian $H(\mathbf{k}_{0} + \mathbf{q})$ is Taylor-expanded around the band-edge wave vector $\mathbf{k}_{0} = (0, 0)$ , the wave-number components $q_{x}$ and $q_{y}$ are replaced by the differential operators $\hat{p}_{x}$ and $\hat{p}_{y}$ , respectively, and the inhomogeneous mass parameter $M(x, y)$ is incorporated. This yields the following result in our present case:

$$
\hat {H} _ {\mathrm{EFA}} = M (x, y) \hat {\sigma} _ {z} + \hat {p} _ {x} \hat {\sigma} _ {x} + \hat {p} _ {y} \hat {\sigma} _ {y}.\tag{7.54}
$$

Then, the familiar EFA proposition is as follows. Assume that the two-component envelope function $\varphi(x,y)$ is a spatially slowly varying solution of the EFA Schrödinger equation

$$
\hat {H} _ {\mathrm{EFA}} \varphi (x, y) = E \varphi (x, y).\tag{7.55}
$$

Then, the lattice state $|\psi\rangle$ associated to the envelope function $\varphi(x,y)$ is defined as

$$
| \psi \rangle = \sum_ {\mathbf {m}, \alpha} \varphi_ {\alpha} (\mathbf {m}) | \mathbf {m}, \alpha \rangle .\tag{7.56}
$$

It is claimed that the lattice state $|\psi\rangle$ is approximately an eigenstate of the inhomogeneous lattice Hamiltonian with the eigenvalue E.

## 7.3.4 Chiral States at a Domain Wall

We can now use the EFA Hamiltonian $(7.54)$ to describe the chiral states at a domain wall between two topologically distinct regions of the QWZ model. The Dirac-type EFA Schrödinger equation reads:

$$
\left[ M (x, y) \hat {\sigma} _ {z} + \hat {p} _ {x} \hat {\sigma} _ {x} + \hat {p} \hat {\sigma} _ {y} \right] \varphi (x, y) = E \varphi (x, y).\tag{7.57}
$$

Consider the homogeneous case first: $M(x, y) = M_{0}$ , where $M_{0}$ might be positive or negative. What is the dispersion relation for propagating waves? What are the energy eigenstates? The answers follow from the plane-wave Ansatz

$$
\varphi (x, y) = \binom{a}{b} e ^ {i q _ {x} x} e ^ {i q _ {y} y}\tag{7.58}
$$

with $q_{x}, q_{y} \in \mathbb{R}$ and $a, b \in \mathbb{C}$ . With this trial wave function, Eq. (7.57) yields two solutions:

$$
E _ {\pm} = \pm \sqrt {M _ {0} ^ {2} + q _ {x} ^ {2} + q _ {y} ^ {2}},\tag{7.59}
$$

and

$$
\frac {a _ {\pm}}{b _ {\pm}} = \frac {q _ {x} - i q _ {y}}{E _ {\pm} - M _ {0}}.\tag{7.60}
$$

Describe now the states at a domain wall between two locally homogeneous regions where the sign of the mass parameter is different. Remember that the sign of the mass parameter in the EFA Hamiltonian is related to the Chern number of the corresponding homogeneous half-BHZ lattice: in our case, a positive (negative) mass implies a Chern number -1 (0).

To be specific, we will consider the case when the two domains are defined as the y < 0 and the y > 0 half-planes, i.e., the mass profile in Eq. (7.57) are

$$
M (x, y) = \left\{ \begin{array}{l l} M _ {0} & \text { if } y > 0, \\ - M _ {0} & \text { if } y <   0 \end{array} \right..\tag{7.61}
$$

Let $M_0$ be positive; the corresponding mass profile is the same as shown in Fig. 7.3a, with $x$ replaced by $y$ .

Now we look for solutions of Eq. (7.57) that reside in the energy range $-M_{0} < E < M_{0}$ , i.e., in the bulk gap of the two domains, and which propagate along, but decay perpendicular to, the domain wall at y = 0. Our wave-function Ansatz for the upper half plane y > 0 is

$$
\varphi_ {u} (x, y) = \binom{a _ {u}}{b _ {u}} e ^ {i q _ {x} x} e ^ {i q _ {y} ^ {(u)} y}\tag{7.62}
$$

with $q_{x} \in \mathbb{R}$ , $q_{y}^{(u)} \in i\mathbb{R}^{+}$ and $a, b \in \mathbb{C}$ . For the lower half plane, $\varphi_{l}(x,y)$ is defined as $\varphi_{u}(x,y)$ but with $u \leftrightarrow l$ interchanged and $q_{y}^{(l)} \in i\mathbb{R}^{-}$ . The wave function $\varphi_{u}(x,y)$ does solve the two-dimensional Dirac equation defined by Eqs. (7.57) and (7.61) in the upper half plane $y > 0$ provided

$$
q _ {y} ^ {(u)} = i \kappa \equiv i \sqrt {M _ {0} ^ {2} + q _ {x} ^ {2} - E ^ {2}}\tag{7.63}
$$

and

$$
\frac {a _ {u}}{b _ {u}} = \frac {q _ {x} + \kappa}{E - M _ {0}}\tag{7.64}
$$

Similar conditions apply for the ansatz $\varphi_{l}(x,y)$ for the lower half plane, with the substitutions $u \mapsto l$ , $\kappa \mapsto -\kappa$ and $M_{0} \mapsto -M_{0}$ . The complete (unnormalized)

envelope function has the form

$$
\varphi (x, y) = \varphi_ {u} (x, y) \Theta (y) + c \varphi_ {l} (x, y) \Theta (- y),\tag{7.65}
$$

where c is a yet unknown complex parameter to be determined from the boundary conditions at the domain wall.

The wave function (7.65) is an eigenstate of the EFA Hamiltonian with energy $E$ if the boundary condition that the wave function is continuous on the line $y = 0$ , that is,

$$
\varphi_ {u} (x, 0) = \varphi_ {l} (x, 0),\tag{7.66}
$$

is fulfilled for every $x$ .

The boundary condition (7.66) determines the value of the parameter c as well as the dispersion relation $E(q_{x})$ of the edge states. First, (7.66) implies

$$
q _ {x} - \kappa = c (q _ {x} + \kappa) \Rightarrow c = \frac {q _ {x} - \kappa}{q _ {x} + \kappa},\tag{7.67}
$$

$$
E + M _ {0} = c (E - M _ {0}) \Rightarrow - q _ {x} M _ {0} = \kappa E.\tag{7.68}
$$

Note that $\kappa$ depends on $E$ according to Eq. (7.63). It is straightforward to find the dispersion relation of the edge states by solving $-q_xM_0 = \kappa(E)E$ for $E$ with the condition $-M_0 < E < M_0$ :

$$
E = - q _ {x}.\tag{7.69}
$$

This simple dispersion relation is shown in Fig. 7.4a. Together with Eq. (7.63), this dispersion implies that the localization length of edge states is governed by $M_{0}$ only, i.e., is independent of $q_{x}$ . The squared wave function of an edge state is shown in Fig. 7.4b.

A remarkable consequence of this simple dispersion relation is that the spinor components of the envelope function also have a simple form:

$$
\binom{a _ {u}}{b _ {u}} = \binom{a _ {l}}{b _ {l}} = \binom{1}{- 1}.\tag{7.70}
$$

Edge states at similar mass domain walls at $u(y) \approx 0$ and $u(y) \approx 2$ can be derived analogously. Note that at $u(y) \approx 0$ , the low-energy states can reside in two different Dirac valleys, around $\mathbf{k}_{0} = (0, \pi)$ or $\mathbf{k}_{0} = (\pi, 0)$ , and there is one edge state in each valley. The number of edge states obtained in the continuum model, as well as their directions of propagation, are in correspondence with those obtained in the lattice model; as we have seen for the latter case, the number and direction are given by the magnitude and the sign of Chern-number difference across the domain wall, respectively.

![](images/d091efe791fdb33570ac0bfbe140f3821847ab1d7be081f47ceaddd4cedd4a4a.jpg)

![](images/e9033af62c03dae4fbee9e26d8e6fc20d7c1e891cc6b739c5e3d230482a1a394.jpg)
Fig. 7.4 Chiral state obtained from the two-dimensional Dirac equation. (a) Dispersion relation and (b) squared wave function of a chiral state confined to, and propagating along, a mass domain wall

An interesting fact is that the existence of the edge state is not constrained to case of a sharp, step-like domain wall described above. Moreover, the simple dispersion relation and spinor structure found above generalize for more irregular domain walls. This generalization is proven in a similar fashion as in the case of the SSH model, see Sect. 7.2.4. To see this, consider an almost arbitrary one-dimensional spatial dependence of the mass, similar to the one in Fig. 7.3b: $M(x,y) = M(y)$ with the only condition that $M$ changes sign between the half-planes $y < 0$ and $y > 0$ , i.e., $M(y \to -\infty) < 0$ and $M(y \to \infty) > 0$ . We claim that there exists a solution of the corresponding two-dimensional Dirac equation that propagates along the domain wall, has the dispersion relation $E = -q_x$ , is confined in the direction perpendicular to the domain wall, and has the wave function

$$
\varphi (x, y) = \binom{1}{- 1} e ^ {i q _ {x} x} f (y).\tag{7.71}
$$

To prove this proposition, insert this wave function $\varphi(x,y)$ to the two-dimensional Dirac equation and substitute E with $-q_{x}$ therein. This procedure results in two equivalent equations that are fulfilled if $\partial_{y}f(y) = -M(y)f(y)$ , implying that Eq. (7.71) is indeed a normalizable solution with $E = -q_{x}$ provided that the function f has the form

$$
f (y) = e ^ {- \int_ {0} ^ {y} d y ^ {\prime} M (y ^ {\prime})}.\tag{7.72}
$$

To summarize: In the preceding chapters, we introduced the topological characterization of lattice models and the corresponding edge states and states bound to domain walls between regions of different topological character. In this chapter, we demonstrated that a low-energy continuum description (the EFA Schrödinger equation) can be derived from a lattice model, and can be used to analyze those electronic states. Besides being a convenient analytical tool to describe inhomogeneous lattices, the envelope-function approximation also demonstrates that the emergence of topologically protected states is not restricted to lattice models.

## Problems

## 7.1 SSH model with spatially dependent intracell hopping

In Sect. 7.2.3, we provide the EFA proposition for the SSH model with spatially dependent intracell hopping. Prove this proposition, following the procedure detailed in Sect. 7.1.3 for the one-dimensional monatomic chain.

## 7.2 SSH model with spatially dependent intercell hopping

Derive the EFA Hamiltonian for an inhomogeneous SSH model, where w varies in space and v is constant. Assume a nearly metallic scenario, $w(x) \approx v$ .

## 7.3 QWZ model

Prove the EFA proposition for the QWZ model. The proposition is outlined in Sect. 7.3.3. The proof is analogous to that used for the SSH model.

## 7.4 QWZ model at $u\approx 2$

The bulk momentum-space Hamiltonian $\hat{H}(k)$ of the QWZ model is given in Eq.(6.1). Starting from this $\hat{H}(k)$ , derive the EFA Hamiltonian describing low-energy excitations in the case of an inhomogeneous $u$ parameter for which $u \approx 2$ .

# Chapter 8 Time-Reversal Symmetric Two-Dimensional Topological Insulators: The Bernevig–Hughes–Zhang Model

In the previous chapters, we have seen how two-dimensional insulators can host one-way propagating (a.k.a. chiral) edge states, which ensures reflectionless transport along the edge. The existence of chiral edge states precludes time-reversal symmetry. Indeed, time-reversed edge states would describe particles propagating backwards along the edge. In Chern insulators (two-dimensional insulators with nonvanishing Chern number), the absence of these counterpropagating states from the spectrum is what ensures the reflectionless propagation of particles along the edges.

What about time-reversal symmetric or time-reversal invariant two dimensional insulators? According to the above, they cannot be Chern insulators. Interestingly though, the same time-reversal symmetry that ensures that for every edge state mode there is a counterpropagating time reversed partner, can also ensure that no scattering between these two modes occurs. This means that it is possible for time-reversal invariant two-dimensional insulators to host edge states with reflectionless propagation, in both directions, at both edges. The details of why and how this happens are discussed in this and the following chapters.

We will find that all two-dimensional time-reversal invariant insulators fall into two classes: the trivial class, with an even number of pairs of edge states at a single edge, and the topological class, with an odd number of pairs of edge states at a single edge. We then subsequently show that disorder that breaks translational invariance along the edge can destroy edge state conduction in the trivial class, but not in the topological class.

The bulk-boundary correspondence for Chern insulators stated that the net number of edge states on the edge is the same as the Chern number of the bulk, $Q$ . We showed this by mapping the two-dimensional system to a periodically, adiabatically pumped one-dimensional chain. After the mapping, the unit of charge pumped through the chain during a period could be identified with the net number of chiral edge states.

Unfortunately, identifying and calculating the bulk topological invariant of a time-reversal invariant two-dimensional insulator is much more cumbersome than for a Chern insulator. We therefore come back to this problem in the next chapter.

## 8.1 Time-Reversal Symmetry

Before we discuss time-reversal symmetric topological insulators, we first need to understand what we mean by time reversal symmetry, and how it leads to Kramers' degeneracy.

## 8.1.1 Time Reversal in Continuous Variable Quantum Mechanics (Without Spin)

Take a single particle with no internal degree of freedom, described by a wavefunction $\Psi(\mathbf{r})$ . Its dynamics is prescribed by a time independent Hamiltonian $\hat{H} = (\hat{p} - e\mathbf{A}(\hat{r}))^{2} + V(\hat{r})$ , where the functions A and V are the vector and scalar potentials, respectively, and e is the charge of the particle. The corresponding Schrödinger equation for the wavefunction $\Psi(r, t)$ reads

$$
i \partial_ {t} \Psi (\mathbf {r}, t) = \left\{(- i \partial_ {\mathbf {r}} - e \mathbf {A} (\mathbf {r})) ^ {2} + V (\mathbf {r}) \right\} \Psi (\mathbf {r}, t).\tag{8.1}
$$

Any solution $\Psi$ of the above equation can be complex conjugated, and gives a solution of the complex conjugate of the Schrödinger equation,

$$
- i \partial_ {t} \Psi (\mathbf {r}, t) ^ {*} = \left\{(- i \partial_ {\mathbf {r}} + e \mathbf {A} (\mathbf {r})) ^ {2} + V (\mathbf {r}) \right\} \Psi (\mathbf {r}, t) ^ {*}.\tag{8.2}
$$

## 8.1.1.1 The Operator of Complex Conjugation in Real Space Basis

We use $K$ to denote the operator that complex conjugates everything to its right in real space basis,

$$
K f (\mathbf {r}) = f (\mathbf {r}) ^ {*} K;
$$

$$
K ^ {2} = 1,\tag{8.3}
$$

for any complex valued function $f(\mathbf{r})$ of position. The Schrödinger equation above can be rewritten using K as

$$
K i \partial_ {t} \Psi = K i \partial_ {t} K K \Psi = - i \partial_ {t} \Psi^ {*} = K \hat {H} K K \Psi = \hat {H} ^ {*} \Psi^ {*}.\tag{8.4}
$$

Complex conjugation in real space basis conforms to intuitive expectations of time reversal: it is local in space, takes $\hat{x} \rightarrow \hat{x}$ , and flips the momenta, $i\partial_{x} \rightarrow -i\partial_{x}$ .

## 8.1.1.2 Time Reversal

The above relation shows that for any closed quantum mechanical system, there is a simple way to implement time reversal. This requires to change both the wavefunction $\Psi$ to $\Psi^{*}$ and the Hamiltonian $\hat{H}$ to $\hat{H}^{*}$ . The change of the Hamiltonian involves $A \rightarrow -A$ , i.e., flipping the sign of the vector potential.

## 8.1.1.3 Time Reversal Symmetry

In the special case where the Hamiltonian in real space basis is real, $\hat{H}^{*} = \hat{H}$ , we can implement time reversal by only acting on the wavefunction. In that case, we say that the system has time reversal symmetry. For the scalar Schrödinger equation above, this happens if there is no vector potential, A = 0. To see this more explicitly, consider time evolution for a time t, then apply the antiunitary operator K, then continue time evolution for time t, then apply K once more:

$$
\hat {U} = K e ^ {- i \hat {H} t} K e ^ {- i \hat {H} t} = e ^ {- K i \hat {H} t K} e ^ {- i \hat {H} t} = e ^ {i \hat {H} ^ {*} t} e ^ {- i \hat {H} t}\tag{8.5}
$$

If $\hat{H}^{*} = \hat{H}$ , then $\hat{U} = 1$ , which means that $K$ acts like time reversal.

## 8.1.2 Lattice Models with an Internal Degree of Freedom

In these notes we deal with models for solids which are lattice Hamiltonians: the position (the external degree of freedom) is discrete, and there can be an internal degree of freedom (sublattice, orbital, spin, or other).

## 8.1.2.1 Definition of the Operator K of Complex Conjugation

For the operator of complex conjugation we need to fix not only the external position basis, $E_{external} = \{|m\rangle\}$ , but also an internal basis, $E_{internal} = \{|\alpha\rangle\}$ . The property defining K then reads

$$
\begin{array}{c} \forall z \in \mathbb {C}, \quad \forall | \mathbf {m} \rangle  ,   | \mathbf {m} ^ {\prime} \rangle \in \mathcal {E} _ {\text { external }}, \quad \forall | \alpha \rangle  ,   | \alpha^ {\prime} \rangle \in \mathcal {E} _ {\text { internal}}: \\ K z   | \mathbf {m}, \alpha \rangle   \langle \mathbf {m} ^ {\prime}, \alpha^ {\prime} | = z ^ {*}   | \mathbf {m}, \alpha \rangle   \langle \mathbf {m} ^ {\prime}, \alpha^ {\prime} |   K, \end{array}\tag{8.6}
$$

where $z^{*}$ is the complex conjugate of z.

We will use the shorthand $|\Psi^{*}\rangle$ and $\hat{A}^*$ to represent $K|\Psi \rangle$ and $K\hat{A}K$ , respectively. The defining equations are

$$
| \Psi \rangle = \sum_ {m} \sum_ {\alpha} \Psi_ {m, \alpha} | m \rangle \otimes | \alpha \rangle ;\tag{8.7}
$$

$$
| \Psi^ {*} \rangle = K | \Psi \rangle = \sum_ {m} \sum_ {\alpha} \Psi_ {m, \alpha} ^ {*} | m \rangle \otimes | \alpha \rangle ;\tag{8.8}
$$

$$
\hat {A} = \sum_ {m ^ {\prime} m} \sum_ {\alpha^ {\prime} \alpha} A _ {m ^ {\prime}, \alpha^ {\prime}, m, \alpha} | m ^ {\prime} \rangle \langle m | \otimes | \alpha^ {\prime} \rangle \langle \alpha |;\tag{8.9}
$$

$$
\hat {A} ^ {*} = K \hat {A} K = \sum_ {m ^ {\prime} m} \sum_ {\alpha^ {\prime} \alpha} A _ {m ^ {\prime}, \alpha^ {\prime}, m, \alpha} ^ {*} \left| m ^ {\prime} \right\rangle \left\langle m \right| \otimes \left| \alpha^ {\prime} \right\rangle \left\langle \alpha \right|.\tag{8.10}
$$

## 8.1.2.2 Time-Reversal Affects External and Internal Degrees of Freedom

We look for a representation of time reversal symmetry $\hat{T}$ in terms of a general antiunitary operator. Apart from complex conjugation, which acts on both external and internal Hilbert space, we allow for an additional unitary operation on the internal degrees of freedom $\hat{\tau}$ , that is independent of position,

$$
\hat {\mathcal {T}} = \hat {\tau} K.\tag{8.11}
$$

We say that a Hamiltonian $\hat{H}$ is time reversal invariant (or time reversal symmetric) with respect to time reversal represented by $\hat{\mathcal{T}}$ if

$$
\hat {\mathcal {T}} \hat {H} \hat {\mathcal {T}} ^ {- 1} = \hat {H}.\tag{8.12}
$$

In the same sense as for the chiral symmetry (cf. Sect. 1.4), when we talk about a Hamiltonian, what we really mean is a set of Hamiltonians $\hat{H} (\underline{\xi})$ , with $\underline{\xi}$ representing parameters that are subject to disorder. Thus, Eq. (8.12) should hold for any of the $\hat{H} (\underline{\xi})$ , with $\hat{\mathcal{T}}$ independent of $\underline{\xi}$ .

## 8.1.3 Two Types of Time-Reversal

We can require that a time reversal operator $\mathcal{T}$ , when squared, should give at most a phase:

$$
\hat {\tau} K \hat {\tau} K = \hat {\tau} \hat {\tau} ^ {*} = e ^ {i \phi} \mathbb {I} _ {\mathrm{internal}}.\tag{8.13}
$$

If that was not the case, if the unitary operator $\hat{\tau}\hat{\tau}^{*}$ was nontrivial, then it would represent a unitary symmetry of a time-reversal invariant Hamiltonian, since

$$
\hat {\tau} \hat {\tau} ^ {*} \hat {H} (\hat {\tau} \hat {\tau} ^ {*}) ^ {\dagger} = \hat {\tau} K \hat {\tau} K \hat {H} K \hat {\tau} ^ {\dagger} K \hat {\tau} ^ {\dagger} = \hat {\tau} K \hat {H} K \hat {\tau} ^ {\dagger} = \hat {H}.\tag{8.14}
$$

As explained in Sect. 1.4, when we want to investigate topological phases, the usual first step is to get rid of unitary symmetries one by one (except for the lattice translation symmetry of the bulk Hamiltonian), by restricting our attention to a single superselection sector of each symmetry. Thus, the only time reversal symmetries that are left are those that fulfil Eq. (8.13).

The phase factor $e^{i\phi} = \hat{T}^{2}$ turns out to have only two possible values: +1 or -1. Multiplying Eq. (8.13) from the left by $\hat{\tau}^{\dagger}$ , we get $\hat{\tau}^{*} = e^{i\phi}\hat{\tau}^{\dagger} = e^{i\phi}(\hat{\tau}^{*})^{T}$ , where the superscript T denotes transposition. Iterating this last relation once more, we obtain $\hat{\tau}^{*} = e^{2i\phi}\hat{\tau}^{*}$ , which means $e^{i\phi} = \pm1$ , wherefore

$$
\hat {\mathcal {T}} ^ {2} = \pm 1.\tag{8.15}
$$

A Hamiltonian with no unitary symmetries can have only one type of time-reversal symmetry: either $\hat{T}^{2}=+1$ , or $\hat{T}^{2}=-1$ , but not both. Assume a Hamiltonian had two different time-reversal symmetries, $\hat{T}$ and $\hat{T}_{1}$ . Along the lines of Eq. (8.14), the product of the two, the unitary operator $\hat{T}_{1}\hat{T}$ would then represent a unitary symmetry. The only exception is if $\hat{T}_{1}=e^{i\chi}\hat{T}$ , when they are not really different symmetries. However, in this case, since $\hat{T}$ is antiunitary, these two symmetries square to the same number, $\hat{T}_{1}^{2}=e^{i\chi}\hat{T}e^{i\chi}\hat{T}=\hat{T}^{2}$ .

An example for a time-reversal operator with $\hat{T}^{2}=+1$ is given by the complex conjugation K. An example for a time-reversal operator with $\hat{T}^{2}=-1$ is time reversal for a spin-1/2 particle. Since time reversal should also flip the spin, it is achieved by $\hat{T}=-i\sigma_{y}K$ , with K defined on the basis of the eigenstates of $\hat{\sigma}_{z}$ . The fact that this works can be checked by $\hat{T}\hat{\sigma}_{j}\hat{T}^{-1}=-\hat{\sigma}_{j}$ for j=x,y,z.

## 8.1.3.1 The Operator $\hat{\tau}$ Is Symmetric or Antisymmetric

Specifying the square of the time-reversal operation constrains the operator $\hat{\tau}$ to be symmetric or antisymmetric. Consider

$$
\hat {\mathcal {T}} ^ {2} = \hat {\tau} K \hat {\tau} K = \hat {\tau} \hat {\tau} ^ {*} = \pm 1; \qquad \hat {\tau} ^ {*} = \pm \hat {\tau} ^ {\dagger} = (\pm \hat {\tau} ^ {T}) ^ {*},\tag{8.16}
$$

where the subscript T denotes transpose in the same basis where the complex conjugate is defined. Therefore,

$$
\hat {\mathcal {T}} ^ {2} = + 1 \quad \Longleftrightarrow \quad \hat {\tau} = \hat {\tau} ^ {T} \quad \mathrm{symmetric};\tag{8.17}
$$

$$
\hat {\mathcal {T}} ^ {2} = - 1 \quad \Longleftrightarrow \quad \hat {\tau} = - \hat {\tau} ^ {T} \quad \text { antisymmetric }.\tag{8.18}
$$

## 8.1.4 Time Reversal of Type $\hat{\mathcal{T}}^2 = -1$ Gives Kramers' Degeneracy

A defining property of an antiunitary operator $\hat{\mathcal{T}}$ is that for any pair of states $|\Psi\rangle$ and $|\Phi \rangle$ , we have

$$
\langle \hat {\mathcal {T}} \Phi | \hat {\mathcal {T}} \Psi \rangle = (\hat {\tau} | \Phi^ {*} \rangle) ^ {\dagger} \hat {\tau} | \Psi^ {*} \rangle . = | \Phi^ {*} \rangle^ {\dagger} \hat {\tau} ^ {\dagger} \hat {\tau} | \Psi^ {*} \rangle = \langle \Phi^ {*} | \Psi^ {*} \rangle = \langle \Phi | \Psi \rangle^ {*}.\tag{8.19}
$$

Consider now this relation with $|\varPhi\rangle = \hat{\mathcal{T}} |\varPsi\rangle$ :

$$
\langle \hat {\mathcal {T}} \Psi | \Psi \rangle^ {*} = \langle \hat {\mathcal {T}} ^ {2} \Psi | \hat {\mathcal {T}} \Psi \rangle = \langle \pm \Psi | \hat {\mathcal {T}} \Psi \rangle = \pm \langle \hat {\mathcal {T}} \Psi | \Psi \rangle^ {*},\tag{8.20}
$$

where the $\pm$ stands for the square of the time reversal operator $\hat{T}$ , which is $\pm1$ . If $\hat{T}^{2}=+1$ , the above line gives no information, but if $\hat{T}^{2}=-1$ , it leads immediately to $\left\langle\hat{T}\Psi\mid\Psi\right\rangle=0$ , which means that for every energy eigenstate, its time-reversed partner, which is also an energy eigenstate with the same energy, is orthogonal. This is known as Kramers degeneracy.

## 8.1.5 Time-Reversal Symmetry of a Bulk Hamiltonian

We now calculate the effect of time-reversal symmetry $\hat{T} = \hat{\tau}K$ on the bulk momentum-space Hamiltonian $\hat{H}(k)$ . This latter is obtained, as in Sect. 1.2, by setting periodic boundary conditions, and defining a plane wave basis in the corresponding external Hilbert space as

$$
| \mathbf {k} \rangle = \frac {1}{\sqrt {N _ {x} N _ {y}}} \sum_ {\mathbf {k}} e ^ {i \mathbf {m k}} | \mathbf {m} \rangle ; \quad \hat {\mathcal {T}} | \mathbf {k} \rangle = | - \mathbf {k} \rangle \hat {\mathcal {T}}.\tag{8.21}
$$

Next, $\hat{H}_{bulk}$ is the part of $\hat{H}$ in the bulk, with periodic boundary conditions, whose components in the plane wave basis define the bulk momentum-space Hamiltonian,

$$
\hat {H} (\mathbf {k}) = \langle \mathbf {k} | \hat {H} _ {\text { bulk }} | \mathbf {k} \rangle ; \quad \hat {H} _ {\text { bulk }} = \sum_ {\mathbf {k}} | \mathbf {k} \rangle \langle \mathbf {k} | \otimes \hat {H} (\mathbf {k}).\tag{8.22}
$$

The effect of time-reversal symmetry follows,

$$
\hat {\mathcal {T}} \hat {H} _ {\mathrm{bulk}} \hat {\mathcal {T}} ^ {- 1} = \sum_ {\mathbf {k}} | - \mathbf {k} \rangle \langle - \mathbf {k} | \otimes \hat {\tau} \hat {H} (\mathbf {k}) ^ {*} \hat {\tau} ^ {\dagger} = \sum_ {\mathbf {k}} | \mathbf {k} \rangle \langle \mathbf {k} | \otimes \hat {\tau} \hat {H} (- \mathbf {k}) ^ {*} \hat {\tau} ^ {\dagger}.\tag{8.23}
$$

We read off the action of $\hat{\mathcal{T}}$ on the bulk momentum-space Hamiltonian, and obtain the necessary requirement of time-reversal symmetry as

$$
\hat {\tau} \hat {H} (- \mathbf {k}) ^ {*} \hat {\tau} ^ {\dagger} = \hat {H} (\mathbf {k}).\tag{8.24}
$$

Note that time-reversal symmetry of the bulk Hamiltonian is necessary, but not sufficient, for time-reversal symmetry of the system: perturbations at the edges can break time reversal.

A direct consequence of time-reversal symmetry is that the dispersion relation of a time-reversal symmetric Hamiltonian has to be symmetric with respect to inversion in the Brillouin zone, $k \rightarrow -k$ . Indeed, take an eigenstate of $\hat{H}(\mathbf{k})$ , with

$$
\hat {H} (\mathbf {k}) \left| u (\mathbf {k}) \right\rangle = E (\mathbf {k}) \left| u (\mathbf {k}) \right\rangle .\tag{8.25}
$$

Using time-reversal symmetry, Eq. (8.24), we obtain

$$
\hat {\tau} \hat {H} (- \mathbf {k}) ^ {*} \hat {\tau} ^ {\dagger} | u (\mathbf {k}) \rangle = E (\mathbf {k}) | u (\mathbf {k}) \rangle .\tag{8.26}
$$

Multiplying from the left by $\hat{\tau}^{\dagger}$ and complex conjugating, we have

$$
\hat {H} (- \mathbf {k}) \hat {\tau} ^ {T} | u (\mathbf {k}) \rangle^ {*} = E \hat {\tau} ^ {T} | u (\mathbf {k}) \rangle^ {*}.\tag{8.27}
$$

This last line tells us that for every eigenstate $|u(\mathbf{k})\rangle$ of $\hat{H}(\mathbf{k})$ , there is a time-reversed partner eigenstate of $\hat{H}(-\mathbf{k})$ at the same energy, $\hat{\tau}^{T}|u(\mathbf{k})\rangle^{*}$ . This implies inversion symmetry of the energies, $E(\mathbf{k}) = E(-\mathbf{k})$ . Note, however, that $E(\mathbf{k}) = E(-\mathbf{k})$ is not enough to guarantee time-reversal symmetry.

It is especially interesting to look at points in the Brillouin zone which map unto themselves under inversion: the Time-reversal invariant momenta (TRIM). In d dimensions there are $2^{d}$ such points, one of which is at the center of the Brillouin zone (so-called $\Gamma$ point), and others at the boundary of the Brillouin zone. At such momenta, Eq. (8.24) implies that

$$
\hat {\tau} \hat {H} (\mathbf {k} _ {\mathrm{TRIM}}) ^ {*} \hat {\tau} ^ {\dagger} = \hat {H} (\mathbf {k} _ {\mathrm{TRIM}}).\tag{8.28}
$$

If $\hat{T}^{2} = -1$ , then because of Kramers degeneracy, at a time-reversal invariant momentum, every eigenvalue of the bulk momentum-space Hamiltonian is (at least) twice degenerate.

## 8.2 Doubling the Hilbert Space for Time-Reversal Symmetry

There is a simple way to construct a system with Time-Reversal Symmetry, $\hat{H}_{TRI}$ , starting from a lattice Hamiltonian $\hat{H}$ . We take two copies of the system, and change the Hamiltonian in one of them to $\hat{H}^{*} = K\hat{H}K$ . We then couple them, much as we did to layer Chern insulators on top of each other in Sect. 6.2.4:

$$
\hat {H} _ {\mathrm{TRI}} = | 0 \rangle \langle 0 | \otimes \hat {H} + | 1 \rangle \langle 1 | \otimes \hat {H} ^ {*} + \left(| 0 \rangle \langle 1 | \otimes \mathbb {I} _ {\text { external }} \otimes \hat {C} + h. c.\right),\tag{8.29}
$$

where the hopping between the copies is accompanied by a position-independent operation $\hat{C}$ on the internal degree of freedom. In a matrix form, in real-space basis (and somewhat simplified notation), this reads

$$
H _ {\mathrm{TRI}} = \left[ \begin{array}{c c} H & C \\ C ^ {\dagger} & H ^ {*} \end{array} \right].\tag{8.30}
$$

We will use $\hat{s}_{x,y,z}$ to denote the Pauli operators acting on the “copy degree of freedom”, defined as

$$
\hat {s} _ {x / y / z} = \hat {\sigma} _ {x / y / z} \otimes \mathbb {I} _ {\text { external }} \otimes \mathbb {I} _ {\text { internal }}.\tag{8.31}
$$

Using these operators, the time-reversal invariant Hamiltonian reads

$$
\begin{array}{l} \hat {H} _ {\mathrm{TRI}} = \frac {1 + \hat {s} _ {z}}{2} \otimes \hat {H} + \frac {1 - \hat {s} _ {z}}{2} \otimes \hat {H} ^ {*} \\ \qquad + \frac {\hat {s} _ {x} + i \hat {s} _ {y}}{2} \otimes \mathbb {I} _ {\text {external}} \otimes \hat {C} + \frac {\hat {s} _ {x} - i \hat {s} _ {y}}{2} \otimes \mathbb {I} _ {\text {external}} \otimes \hat {C} ^ {\dagger}. \end{array}\tag{8.32}
$$

The choice of the coupling operator $\hat{C}$ is important, as it decides which type of time-reversal symmetry $\hat{H}_{TRI}$ will have.

## 8.2.1 Time Reversal with $\hat{\mathcal{T}}^2 = -1$ Requires Antisymmetric Coupling Operator $\hat{C}$

If we want a time-reversal symmetry that squares to -1, we can go for

$$
\hat {\mathcal {T}} = i \hat {s} _ {y} K;
$$

$$
\hat {\mathcal {T}} ^ {- 1} = K (- i) \hat {s} _ {y},\tag{8.33}
$$

with the factor of i is included for convenience, so that the matrix of $i\hat{s}_{y}$ is real. The requirement of time-reversal symmetry can be obtained using

$$
\begin{array}{c} (i \hat {s} _ {y} K) H _ {\mathrm{TRI}} (i \hat {s} _ {y} K) ^ {- 1} = \left[ \begin{array}{c c} 0 & 1 \\ - 1 & 0 \end{array} \right] \cdot \left[ \begin{array}{c c} H ^ {*} & C ^ {*} \\ C ^ {T} & H \end{array} \right] \cdot \left[ \begin{array}{c c} 0 & - 1 \\ 1 & 0 \end{array} \right] \\ = \left[ \begin{array}{c c} C ^ {T} & H \\ - H ^ {*} & - C ^ {*} \end{array} \right] \cdot \left[ \begin{array}{c c} 0 & - 1 \\ 1 & 0 \end{array} \right] = \left[ \begin{array}{c c} H & - C ^ {T} \\ - C ^ {*} & H ^ {*} \end{array} \right]. \end{array}\tag{8.34}
$$

We have time-reversal symmetry represented by $\hat{T}=i\hat{s}_{y}K$ , if

$$
i \hat {s} _ {\mathrm{y}} K \hat {H} _ {\mathrm{TRI}} (i \hat {s} _ {\mathrm{y}} K) ^ {- 1} = \hat {H} _ {\mathrm{TRI}} \quad \Leftrightarrow \quad \hat {C} = - \hat {C} ^ {T},\tag{8.35}
$$

where the subscript T denotes transposition in the same fixed internal basis that is used to define complex conjugation K.

## 8.2.2 Symmetric Coupling Operator $\hat{C}$ Gives Time Reversal with $\hat{\mathcal{T}}^2 = +1$

If the coupling operator is symmetric, $\hat{C} = \hat{C}^{T}$ , then the same derivation as above shows that we have time-reversal symmetry represented by $\hat{T} = \hat{s}_{x}K$ ,

$$
\hat {s} _ {x} K \hat {H} _ {\mathrm{TRI}} K \hat {s} _ {x} = \hat {H} _ {\mathrm{TRI}} \quad \Leftrightarrow \quad \hat {C} = \hat {C} ^ {T}.\tag{8.36}
$$

This time-reversal operator squares to +1.

If all we want is a lattice Hamiltonian with a time-reversal symmetry that squares to $+1$ , we don't even need to double the Hilbert space. We can just take

$$
\hat {\mathcal {T}} = K; \qquad \qquad \hat {H} _ {\mathrm{TRI}} = \frac {\hat {H} + \hat {H} ^ {*}}{2} = K \hat {H} _ {\mathrm{TRI}} K.\tag{8.37}
$$

Colloquially, this construction is referred to as taking the real part of the Hamiltonian.

## 8.3 A Concrete Example: The Bernevig-Hughes-Zhang Model

To have an example, we use the construction above to build a toy model—called Bernevig-Hughes-Zhang (BHZ) model—for a time-reversal invariant topological insulator starting from the QWZ model of Chap.6. We follow the construction through using the bulk momentum-space Hamiltonian, and obtain

$$
\hat {H} _ {\mathrm{BHZ}} (\mathbf {k}) = \hat {s} _ {0} \otimes [ (u + \cos k _ {x} + \cos k _ {y}) \hat {\sigma} _ {z} + \sin k _ {y} \hat {\sigma} _ {y}) ] + \hat {s} _ {z} \otimes \sin k _ {x} \hat {\sigma} _ {x} + \hat {s} _ {x} \otimes \hat {C},\tag{8.38}
$$

where $\hat{C}$ is a Hermitian coupling operator acting on the internal degree of freedom. For $\hat{C}=0$ , the Hamiltonian $\hat{H}_{BHZ}$ reduces to the 4-band toy model for HgTe, introduced by Bernevig, Hughes and Zhang [5].

## 8.3.1 Two Time-Reversal Symmetries If There Is No Coupling

If there is no coupling between the copies, $\hat{C} = 0$ , the BHZ model has two different time-reversal symmetries, $\hat{\mathcal{T}} = i\hat{s}_yK$ and $\hat{\mathcal{T}}_2 = \hat{s}_xK$ , due to its block diagonal structure reflecting a unitary symmetry, $\hat{s}_z\hat{H}_{\mathrm{BHZ}}\hat{s}_z^\dagger = \hat{H}_{\mathrm{BHZ}}$ . In this situation, the type of predictions we can make will depend on which of these symmetries is robust against disorder. We will in the following require the $\hat{\mathcal{T}}^2 = -1$ symmetry. If this symmetry is robust, then everything we do will apply to the $\hat{C} = 0$ case of $\hat{H}_{\mathrm{BHZ}}$ . The extra time reversal symmetry in that case is just a reminder that most features could be calculated in a more simple way, by working in the superselection sectors of $\hat{s}_z$ separately.

## 8.4 Edge States in Two-Dimensional Time-Reversal Invariant Insulators

We now consider the situation of edge states in a two-dimensional lattice Hamiltonian with time-reversal symmetry, much in the same way as we did for Chern insulators in Sect. 6.2.

## 8.4.1 An Example: The BHZ Model with Different Types of Coupling

We start with the concrete example of the BHZ model. We set the sublattice potential parameter $u = -1.2$ , and plot the edge dispersion relation, defined in the same way as for the Chern insulators in Sect. 6.2.

As long as there is no coupling between the two copies, $\hat{C} = 0$ , the system $\hat{H}_{\mathrm{BHZ}}$ is a direct sum of two Chern insulators, with opposite Chern numbers. As Fig. 8.1a shows, on each edge, there is a pair of edge state branches: a branch on the layer with Hamiltonian $\hat{H}$ , and a counterpropagating branch on the layer with $\hat{H}^*$ . Although these two edge state branches cross, this crossing will not turn into an anticrossing: the states cannot scatter into each other since they are on different layers. The two edge state branches are linked by time-reversal: they occupy the same position, but describe propagation in opposite directions. In fact, they are linked by both time-reversal symmetries this system has, by $\hat{s}_xK$ and $i\hat{s}_yK$ .

A coupling between the layers can gap the edge states out, as shown in Fig. 8.1b. We here used $\hat{C} = 0.3\hat{\sigma}_x$ , which respects the $\hat{\mathcal{T}}^2 = +1$ symmetry but breaks the $\hat{\mathcal{T}}^2 = -1$ one. The crossings between counterpropagating edge states on the same edge have turned into anticrossings, as expected, since the coupling allows particles to hop between the counterpropagating edge states (on the same edge, but in different layers).

![](images/3e223c70e543740da21a83a9320b4c784808c8c493be3dc5974b0aa21d0e0803.jpg)
Fig. 8.1 Stripe dispersion relations of the BHZ model with stripe width N = 10, and with sublattice potential parameter u = -1.2. Right/left edge states (more than 60% weight on the last/first two columns of unit cells) marked in dark red/light blue. (a): uncoupled layers, $\hat{C} = 0$ . (b): Symmetric coupling $\hat{C} = 0.3\hat{\sigma}_{x}$ gaps the edge states out. (c): Antisymmetric coupling $\hat{C} = 0.3\hat{\sigma}_{y}$ cannot open a gap in the edge spectrum

We see something different if we couple the layers while respecting the $\hat{T}^{2} = -1$ time reversal symmetry, by, e.g., $\hat{C} = 0.3\sigma_{y}$ . As Fig. 8.1c shows, the crossing at $k_{y} = 0$ between the edge state branches now does not turn into an anticrossing. As long as the coupling is not strong enough to close the bulk gap, the edge states here appear to be protected.

## 8.4.2 Edge States in $\hat{\mathcal{T}}^2 = -1$

The states form one-dimensional edge state bands in the one-dimensional Brillouin zone $k_{x} = -\pi, \ldots, \pi$ , shown schematically in Fig. 8.2. In general, an edge will host edge states propagating in both directions. However, due to time-reversal symmetry, the dispersion relations must be left-right symmetric when plotted against the wavenumber $k_{x}$ along the edge direction. This means that the number $N_{+}$ of right-moving edge states (these are plotted with solid lines in Fig. 8.2), and $N_{-}$ , the number of left-moving edge states (dashed lines) have to be equal at any energy,

$$
N _ {+} (E) = N _ {-} (E).\tag{8.39}
$$

As with Chern insulators, we next consider the effect of adiabatic deformations of the clean Hamiltonian on edge states. We consider terms in the Hamiltonian that conserve translational invariance along the edge, and respect Time Reversal Symmetry. The whole discussion of Sect. 6.3 applies, and therefore adiabatic deformations cannot change the signed sum of the left- and right-propagating edge states in the gap. However, time-reversal symmetry restricts this sum to zero anyway.

Time reversal symmetry that squares to $\hat{T}^{2} = -1$ , however, provides a further restriction: adiabatic deformations can only change the number of edge states by integer multiples of four (pairs of pairs). To understand why, consider the adiabatic deformation corresponding to Fig. 8.2a–d. Degeneracies in the dispersion relation can be lifted by coupling the edge states, as it happens in (b), and this can lead to certain edge states disappearing at certain energies, as in (c). This can be visualized by plotting the $k_{x}$ values at E = 0 of the branches of the edge state dispersion as functions of the deformation parameter (which is some combination of the parameters of the Hamiltonian) a, as in Fig. 8.2e. Due to the deformation, two counterpropagating edge states can “annihilate”, when the corresponding modes form an avoided crossing. If this happens at a generic momentum value k, as in (c), then, due to the time reversal invariance, it also has to happen at -k, and so the number of edge states decreases by 4, not by 2. The special momentum values of $k_{x}$ are the time-reversal invariant momenta, which in this case are $k_{x} = 0, \pm\pi$ . If the edge state momenta meet at a time-reversal invariant momentum, as in (b) at $k_{x} = 0$ , their “annihilation” would change the number of edge states by 2 and not by 4. However, this cannot happen, as it would create a situation that violates the Kramers degeneracy: at the time-reversal invariant momenta, energy eigenstates have to be doubly degenerate. The deformations in Fig. 8.2 can be also read from (d) to (a), and therefore apply to the introduction of new edge states as well.

![](images/1d1b2a29f32472eb499fbb30cc5e4fdb661b4a9f40bf0599837e750c1c86bda1.jpg)

![](images/5703b71958081be93f46af7c5364ea0dbaf9a48c6308554a960a3c167b1cacf6.jpg)

![](images/dbe3f4d2abb24a13d57b56e52461cc097afac15d52689ae6c611bbd17da23e63.jpg)

![](images/2b43308866d2ca11ad0e838bc867c0ef5e6eee4146172c56565b2b767265d2b2.jpg)

(e)
![](images/fe7477e82ba312c766e4e4b24f7dae887d83a8fdda01afa6c1ae7c332974db1b.jpg)
Fig. 8.2 Edge states on a single edge of a two-dimensional time-reversal invariant topological insulator with $\hat{\mathcal{T}}^2 = -1$ , as the edge region undergoes a continuous deformation, parametrized by $a$ , respecting the symmetry and the translational invariance along the edge. In (a)-(d), the edge state dispersions relations are shown, in the full edge Brillouin zone $k_x = -\pi, \ldots, \pi$ , and in the energy window corresponding to the bulk gap. For clarity, right- (left-) propagating edge states are denoted by continuous (dashed) lines. Due to the deformation of the Hamiltonian, the edge state branches can move, bend, and couple, while the bulk remains unchanged. From (a) to (b), the crossing points between counterpropagating edge states become anticrossings, i.e., gaps open in these pairs of dispersion relation branches as a usual consequence of any parameter coupling them. Crossings at $k_x = 0$ and $k_x = \pi$ cannot be gapped, as this would lead to a violation of the Kramers theorem. From (b) to (d), these gaps become so large that at energy $E = 0$ , the number of edge states drops from 6 (3 Kramers pairs) to 2 (1 Kramers pair). The $k_x$ values of the edge states at zero energy are plotted in (e), where this change in the number of edge states shows up as an “annihilation” of right-propagating and left-propagating edge states

## 8.4.3 $\mathbb{Z}_2$ Invariant: Parity of Edge State Pairs

At any energy inside the bulk gap, the parity of the number of edge-state Kramers pairs for a given dispersion relation is well defined. In Fig. 8.2a, there are 3 edge-state Kramers pairs for any energy in the bulk gap, i.e., the parity is odd. In Fig. 8.2c, there are 3 of them for every energy except for energies in the mini-gap of the bands on the left and right for which the number of edge-state Kramers pairs is 1, and for the upper and lower boundaries of the mini-gap [the former depicted by the horizontal line in Fig. 8.2c], where the number of Kramers pairs is 2. The parity is odd at almost every energy, except the two isolated energy values at the mini-gap boundaries.

The general proposition is that the parity of the number of edge-state Kramers pairs at a given edge for a given Hamiltonian at a given energy is independent of the choice of energy, as long as this energy is inside the bulk gap. Since in a time-reversal invariant system, all edge states have counter-propagating partners, we can express this number as

$$
D = \frac {N (E)}{2} \bmod 2 = \frac {N _ {+} (E) + N _ {-} (E)}{2} \bmod 2,\tag{8.40}
$$

where $N(E) = N_{+}(E) + N_{-}(E)$ is the total number of edge states at an edge. A caveat is that there are a few isolated energy values where this quantity is not well defined, e.g., the boundaries of mini-gaps in the above example, but these energies form a set of zero measure.

Since D is a topological invariant, we can classify two-dimensional time-reversal invariant lattice models according to it, i.e., the parity of the number of edge-state

Kramers pairs supported by a single edge of the terminated lattice. Because it can take on two values, this 'label' $D$ is called the $\mathbb{Z}_2$ invariant, and is represented by a bit taking on the value 1 (0) if the parity is odd (even).

As a final step, we should next consider disorder that breaks translational invariance along the edge, in the same way as we did for Chern insulators. Due to the presence of edge states propagating in both directions along the edge, the treatment of disorder is a bit trickier than it was for chiral edge states.

## 8.5 Absence of Backscattering

A remarkable property of Chern insulators is that they support chiral edge states, i.e., edge states that have no counter-propagating counterparts. A simple fact implied by the chiral nature of these edge states is that impurities are unable to backscatter particles. As we argue below, absence of backscattering is also characteristic of disordered two-dimensional time-reversal invariant topological insulators with $\hat{T}^{2} = -1$ , although the robustness is guaranteed only against time-reversal symmetric scatterers.

Here, we introduce the scattering matrix, a concept that allows for a formal analysis of scattering at impurities, and discuss the properties of edge state scattering in two-dimensional time-reversal invariant topological insulators with $\hat{T}^{2} = -1$ . The scattering matrix will also serve as a basic tool in the last chapter, where we give a theoretical description of electronic transport of phase-coherent electrons, and discuss observable consequences of the existence and robustness of edge states.

## 8.5.1 The Scattering Matrix

Consider a phase-coherent two-dimensional conductor with a finite width in the y direction, and discrete translational invariance along the x axis. Think of the system as having periodic boundary conditions in the x direction. As earlier, we describe the system in terms of a simple lattice model, where the unit cells form a square lattice of size $N_{x} \times N_{y}$ , and there might be an internal degree of freedom associated to the unit cells.

As the system has discrete translational invariance along $x$ , we can also think of it as a one-dimensional lattice, whose unit cell incorporates both the internal degree of freedom of the two-dimensional lattice and the real-space structure along the $y$ axis. Using that picture, it is clear that the electronic energy eigenstates propagating along the $x$ axis at energy $E$ have a product structure, as required by the one-dimensional Bloch's theorem:

$$
| l, \pm \rangle = | k _ {l, \pm} \rangle \otimes | \varPhi_ {l, \pm} \rangle ,\tag{8.41}
$$

On the right hand side, the first ket corresponds to a usual momentum eigenstate $|k\rangle = \frac{1}{\sqrt{N_{x}}}\sum_{m_{x}=1}^{N_{x}}e^{ikm_{x}}|m_{x}\rangle$ propagating along x, whereas the second ket incorporates the shape of the transverse standing mode as well as the internal degree of freedom. The states appearing in Eq. (8.41) are normalized to unity. The integer $l = 1, 2, \ldots, N$ labels the propagating modes, also referred to as scattering channels. The + and − signs correspond to right-moving and left-moving states, respectively; the direction of movement is assigned according to the sign of the group velocity:

$$
v _ {l, \pm} = \left. \frac {d E (k)}{d k} \right| _ {k = k _ {l, \pm}},\tag{8.42}
$$

where $E(k)$ is the dispersion relation of the one-dimensional band hosting the state $|l, \pm \rangle$ .

Now we re-normalize the states above, such that different states carry the same particle current through an arbitrary vertical cross section of the system. We will use these current-normalized wave functions in the definition of the scattering matrix below, which guarantees that the latter is a unitary matrix. According to the one-dimensional relation we obtained between the current and the group velocity in Eq. (5.13), the current-normalized states can be defined using the group velocity as

$$
| l, \pm \rangle_ {c} = \frac {1}{\sqrt {| v _ {l , \pm} |}} | l, \pm \rangle .\tag{8.43}
$$

Now consider the situation when the electrons are obstructed by a disordered region in the conductor, as shown in Fig. 8.3. A monoenergetic wave incident on the scattering region is characterized by a vector of coefficients

$$
a ^ {(\mathrm{in})} = \left(a _ {L, 1} ^ {(\mathrm{in})}, a _ {L, 2} ^ {(\mathrm{in})}, \dots , a _ {L, N} ^ {(\mathrm{in})}, a _ {R, 1} ^ {(\mathrm{in})}, a _ {R, 2} ^ {(\mathrm{in})}, \dots , a _ {R, N} ^ {(\mathrm{in})}\right).\tag{8.44}
$$

![](images/a7e696c8730b31b0543da8f9ddbb2f9c8b729f41d87b652e8c42daf17853f211.jpg)
Fig. 8.3 Disordered region (gray) obstructing electrons in a two-dimensional phase-coherent conductor. The scattering matrix S relates the amplitudes $a_{L}^{(\mathrm{in})}$ and $a_{R}^{(\mathrm{in})}$ of incoming waves to the amplitudes $a_{L}^{(\mathrm{out})}$ and $a_{R}^{(\mathrm{out})}$ of outgoing waves

The first (second) set of N coefficients correspond to propagating waves (8.43) in the left (right) lead L (R), that is, the clean regions on the left (right) side of the disordered region. The reflected and transmitted parts of the wave are described by the vector

$$
a ^ {(\text { out })} = \left(a _ {L, 1} ^ {(\text { out })}, a _ {L, 2} ^ {(\text { out })}, \dots , a _ {L, N} ^ {(\text { out })}, a _ {R, 1} ^ {(\text { out })}, a _ {R, 2} ^ {(\text { out })}, \dots , a _ {R, N} ^ {(\text { out })}\right).\tag{8.45}
$$

The corresponding energy eigenstate reads

$$
| \psi \rangle = \sum_ {l = 1} ^ {N} a _ {L, l} ^ {\mathrm{(in)}} | l, +, L \rangle_ {c} + a _ {L, l} ^ {\mathrm{(out)}} | l, -, L \rangle_ {c} + a _ {R, l} ^ {\mathrm{(in)}} | l, -, R \rangle_ {c} + a _ {R, l} ^ {\mathrm{(out)}} | l, +, R \rangle_ {c}.\tag{8.46}
$$

Here, the notation for the current-normalized states introduced in Eq. (8.43) has been expanded by the lead index L/R.

The scattering matrix S relates the two vectors introduced in Eqs. $(8.44)$ and $(8.45)$ :

$$
a ^ {(\mathrm{out})} = S a ^ {(\mathrm{in})}.\tag{8.47}
$$

The size of the scattering matrix is $2N \times 2N$ , and it has the following block structure:

$$
S = \left( \begin{array}{c c} r & t ^ {\prime} \\ t & r ^ {\prime} \end{array} \right)\tag{8.48}
$$

where r and $r'$ are $N \times N$ reflection matrices describing reflection from left to left and from right to right, and t and $t'$ are transmission matrices describing transmission from left to right and right to left.

Particle conservation, together with the current normalization Eq. (8.43), implies the unitarity of the scattering matrix S. In turn, its unitary character implies that the Hermitian matrices $tt^{\dagger}$ , $t^{\prime}t^{\prime\dagger}$ , $1 - rr^{\dagger}$ , and $1 - r^{\prime}r^{\prime\dagger}$ all have the same set of real eigenvalues $T_{1}, T_{2}, \ldots, T_{N}$ , called transmission eigenvalues.

## 8.5.2 A Single Kramers Pair of Edge States

Now we use the scattering matrix S to characterize defect-induced scattering of an electron occupying an edge state of a two-dimensional time-reversal invariant topological insulator. Consider a half-plane of such a homogeneous lattice which supports exactly one Kramers pair of edge states at a given energy E in the bulk gap, as shown in Fig. 8.4. Consider the scattering of the electron incident on the defect from the left side in Fig. 8.4. The scatterer is characterized by the Hamiltonian V. We will show that the impurity cannot backscatter the electron as long as V is time-reversal symmetric.

![](images/1642858901d081416996fdf30ff7e32c2565e70e913f838f7e181a991f9d6024.jpg)
Fig. 8.4 Scattering of an edge state on a time-reversal symmetric defect V. In a two-dimensional time-reversal invariant topological insulator with $\hat{T}^{2} = -1$ , having a single Kramers pair of edge states, the incoming electron is transmitted through such a defect region with unit probability

We start out without specifying the number of propagating edge-state Kramers pairs on the edge. Later we distinguish between the cases when (i) that number is one, and (ii) when that is a higher odd number. We choose our propagating modes such that the incoming and outgoing states are related by time-reversal symmetry, i.e.,

$$
\left| l, -, L \right> _ {c} = \hat {\mathcal {T}} \left| l, +, L \right> _ {c}\tag{8.49a}
$$

$$
| l, +, R \rangle_ {c} = \hat {\mathcal {T}} | l, -, R \rangle_ {c}.\tag{8.49b}
$$

Also, recall that $\hat{T}^{2} = -1$ . In the presence of the perturbation V, the edge states of the disorder-free system are no longer energy eigenstates of the system. A general scattering state $|\psi\rangle$ at energy E is characterized by the vector $a^{(\mathrm{in})}$ of incoming amplitudes. According to Eq. (8.46) and the definition (8.47) of the scattering matrix S, the energy eigenstates outside the scattering region can be expressed as:

$$
\begin{array}{c} | \psi \rangle = \sum_ {l = 1} ^ {N} \Big [ a _ {L, l} ^ {\mathrm{(in)}}   | l, +, L \rangle_ {c} + a _ {R, l} ^ {\mathrm{(in)}}   | l, -, R \rangle_ {c} \\ \qquad + \big (S a ^ {\mathrm{(in)}} \big) _ {L, l}   | l, -, L \rangle_ {c} + \big (S a ^ {\mathrm{(in)}} \big) _ {R, l}   | l, +, R \rangle_ {c} \Big ]. \end{array}\tag{8.50}
$$

Using Eq. (8.49), we find

$$
\begin{array}{r l} & {- \hat {\mathcal {T}} | \psi \rangle = \sum_ {l = 1} ^ {N} \Big [ - a _ {L, l} ^ {(\mathrm{in}) *} | l, -, L \rangle_ {c} - a _ {R, l} ^ {(\mathrm{in}) *} | l, +, R \rangle_ {c}} \\ & {\qquad + (S ^ {*} a ^ {(\mathrm{in}) *}) _ {L, l} | l, +, L \rangle_ {c} + (S ^ {*} a ^ {(\mathrm{in}) *}) _ {R, l} | l, -, R \rangle_ {c} \Big ].} \end{array}\tag{8.51}
$$

Due to time reversal symmetry, this state $-\hat{\mathcal{T}}|\psi\rangle$ is also an energy eigenstate having the same energy as $|\psi\rangle$ . Using the unitary character of the scattering matrix, the state $-\hat{\mathcal{T}}|\psi\rangle$ can be rewritten as

$$
\begin{array}{l} - \hat {\mathcal {T}} | \psi \rangle = \sum_ {l = 1} ^ {N} \Big [ \big (S ^ {*} a ^ {(\mathrm{in}) *} \big) _ {L, l} | l, +, L \rangle_ {c} + \big (S ^ {*} a ^ {(\mathrm{in}) *} \big) _ {R, l} | l, -, R \rangle_ {c} \\ \qquad + \big (- S ^ {T} S ^ {*} a ^ {(\mathrm{in}) *} \big) _ {L, l} | l, -, L \rangle_ {c} + \big (- S ^ {T} S ^ {*} a ^ {(\mathrm{in}) *} \big) _ {R, l} | l, +, R \rangle_ {c} \Big ]. \end{array} \tag {8.5}\tag{8.52}
$$

Here, $S^{T}$ denotes the transpose of S, is also an energy eigenstate having the same energy as $|\psi\rangle$ . Comparing Eqs. (8.50) and (8.52), and knowing that the scattering matrix at a given energy is uniquely defined, we conclude that $S = -S^{T}$ , that is

$$
\left( \begin{array}{c c} r & t ^ {\prime} \\ t & r ^ {\prime} \end{array} \right) = S = - S ^ {T} = \left( \begin{array}{c c} - r & - t \\ - t ^ {\prime} & - r ^ {\prime} \end{array} \right).\tag{8.53}
$$

If we have a single edge-state Kramers pair, then the latter relation implies

$$
r = r ^ {\prime} = 0,\tag{8.54}
$$

and hence perfect transmission of each of the two incoming waves.

If the lattice has the geometry of a ribbon, and the time-reversal symmetric scatterer extends to both edges, then the absence of backscattering is not guaranteed. This is illustrated in Fig. 8.5, where we compare three examples. In (a), the defect is formed as a wide constriction on both edges, with a width much larger than the characteristic length of the penetration of the edge states to the bulk region of the ribbon. Backscattering between states at the same edge is forbidden due to time-reversal symmetry, and backscattering between states at different edges is forbidden due to a large spatial separation of their corresponding wave functions. In (b), a similar but narrower constriction with a width comparable to the penetration length of the edge states does allow for scattering between states on the lower and upper edges. In this case, backscattering from a right-moving state on one edge to a left-moving state at the other edge is not forbidden. In (c), the constriction divides the ribbon to two unconnected parts, resulting in zero transmission through the constriction.

![](images/2b98ca31f8bddf415e1a6fae1c0745f1d9105beb2c647bce1ba3698d4841aabc.jpg)
Fig. 8.5 Backscattering of edge states at a constriction. The states forming the edge-state Kramers pairs are depicted as solid and dashed lines. (a) A time-reversal symmetric defect localized to the edges, such as a small constriction shown here, is unable to backscatter the incoming electron. (b) Backscattering is possible between different edges, if the width of the constriction is of the order of the decay length of the edge states. (c) A finite spatial gap between the left and right part of the wire implies zero transmission

Naturally, backscattering is also allowed if the scatterer is not time-reversal symmetric, or if the scattering process is inelastic. Backscattering is not forbidden for the ‘unprotected’ edge states of topologically trivial $D = 0$ two-dimensional time-reversal invariant insulators.

## 8.5.3 An Odd Number of Kramers Pairs of Edge States

The above statement $(8.54)$ implying unit transmission can be generalized for arbitrary two-dimensional time-reversal invariant topological insulator lattice models, including those where the number of edge-state Kramers pairs N is odd but not one. The proposition is that in such a system, given a time-reversal symmetric scatterer V and an arbitrary energy E in the bulk gap, there exists at least one linear combination of the incoming states of energy E from each side of the defect that is perfectly transmitted through the defect.

The proof follows that in the preceding section, with the difference that the quantities r and t describing reflection and transmission are $N \times N$ matrices, and that the antisymmetric nature of the S-matrix $S = -S^{T}$ implies the antisymmetry of the reflection matrices $r = -r^{T}$ . According to Jacobi's theorem, every odd-dimensional antisymmetric matrix has a vanishing determinant, which is implied by

$$
\det (r) = \det (r ^ {T}) = \det (- r) = (- 1) ^ {N} \det (r) = - \det (r),\tag{8.55}
$$

where we used the antisymmetry of r in the second step and the oddness of N in the last step. As a consequence of Eq. (8.55) we know $\det(r)=0$ , hence $\det(r^{\dagger}r)=\det(r^{\dagger})\det(r)=0$ . Therefore, at least one eigenvalue of $r^{\dagger}r$ is zero, which implies that at least one transmission eigenvalue $T_{l}$ is unity.

## 8.5.4 Robustness Against Disorder

The absence-of-backscattering result (8.54) implies a remarkable statement regarding the existence of (at least) one perfectly transmitting edge state in a finite-size disordered sample of a two-dimensional time-reversal invariant topological insulator. (See also the discussion about Fig. 6.9 in the context of Chern insulators.) Such a sample with an arbitrarily chosen geometry is shown in Fig. 8.6. Assume that the disorder is time-reversal symmetric and localized to the edge of the sample. We claim that any chosen segment of the edge of this disordered sample supports, at any energy that is deep inside the bulk gap, (at least) one counterpropagating Kramers pair of edge states that are delocalized along the edge and able to transmit electrons with unit probability. This is a rather surprising feature in light of the fact that in truly one-dimensional lattices, a small disorder is enough to induce Anderson localization of the energy eigenstates, and hence render the system an electronic insulator.

![](images/bb6ce6bea2d49079ce1a2b38cc1504617c274e529862bf19cb131bc51778c6d1.jpg)
Fig. 8.6 A disordered two-dimensional time-reversal invariant topological insulator contacted with two electrodes. Disorder is ‘switched off’ and the edge is ‘straightened out’ within the dashed box, hence the edge modes there resemble those of the disorder-free lattice

To demonstrate the above statement, let us choose an edge segment of the disordered sample for consideration, e.g., the edge segment running outside the dashed box in Fig. 8.6. Now imagine that we ‘switch off’ disorder in the complementer part of the edge of the sample, and ‘straighten out’ the geometry of that complementer part, the latter being shown within the dashed box of Fig. 8.6. Furthermore, via an appropriate spatial adiabatic deformation of the Hamiltonian of the system in the vicinity of the complementer part of the edge (i.e., within the dashed box in Fig. 8.6), we make sure that only a single edge-state Kramers pair is present within this complementer part. The existence of such an adiabatic deformation is guaranteed by the topologically nontrivial character of the sample, see the discussion of Fig. 8.2. The disordered edge segment, outside the dashed box in Fig. 8.6, now functions as a scattering region for the electrons in the straightened part of the edge. From the result (8.54) we know that such a time-reversal symmetric scatterer is unable to induce backscattering between the edge modes of the straightened part of the edge, hence we must conclude that the disordered segment must indeed support a perfectly transmitting edge state in each of the two propagation directions.

In the last chapter, we show that the electrical conductance of such a disordered sample is finite and ‘quantized’, if it is measured through a source and a drain contact that couple effectively to the edge states.

# Chapter 9 The $\mathbb{Z}_2$ Invariant of Two-Dimensional Topological Insulators

In the previous chapter, we have seen that two-dimensional insulators can host topologically protected edge states even if time-reversal symmetry is not broken, provided it squares to -1, i.e., $\hat{T}^{2} = -1$ . Such systems fall into two categories: no topologically protected edge states (trivial), or one pair of such edge states (topological). This property defines a $Z_{2}$ invariant for these insulators. In the spirit of the bulk–boundary correspondence, we expect that the bulk momentum-space Hamiltonian $\hat{H}(\mathbf{k})$ should have a corresponding topological invariant (generalized winding number).

The bulk $Z_{2}$ invariant is notoriously difficult to calculate. The original definition of the invariant [11, 19] uses a smooth gauge in the whole Brillouin zone, that is hard to construct [31], which makes the invariant difficult to calculate. An altogether different approach, which is robust and calculatable, uses the scattering matrix instead of the Hamiltonian [13].

In this chapter we review a definition of the bulk $Z_{2}$ invariant [37] based on the dimensional reduction to charge pumps. This is equivalent to the originally defined bulk invariants [37], but no smooth gauge is required to calculate it. It can be outlined as follows.

1. Start with a bulk Hamiltonian $\hat{H}(k_x, k_y)$ . Reinterpret $k_y$ as time: $\hat{H}(k_x, k_y)$ can be thought of as a bulk one-dimensional Hamiltonian of an adiabatic pump. This is the dimensional reduction we used for Chern insulators in Chap. 6.

2. Track the motion, with $k_{y}$ playing the role of time, of pumped particles in the bulk using Wannier states. We will call this the Wannier center flow.

3. Time-reversal symmetry restricts the Wannier center flow. As a result, in some cases the particle pump cannot be turned off adiabatically—in those cases the insulator is topological. If it can be turned off, the insulator is trivial.

In order to go through this argument, we will first gather the used mathematical tools, i.e., generalize the Berry phase and the Wannier states. We will then define the Wannier center flow, show that it can be calculated from the Wilson loop. Finally, we will use examples to illustrate the $Z_{2}$ invariant and argue that it gives the number of topologically protected edge states.

## 9.1 Tools: Nonabelian Berry Phase, Multiband Wannier States

To proceed to calculate the topological invariants, we need to generalize the tools of geometric phases, introduced in Chap. 2, and of the Wannier states of Chap. 3, to manifolds consisting of more bands.

## 9.1.1 Preparation: Nonabelian Berry Phase

We defined the Berry phase in Chap. 2, as the relative phase around a loop L of N states $\left|\Psi_{j}\right\rangle$ , with $j = 1, 2, \ldots, N$ . Since the Berry phase is gauge independent, it is really a property of the loop over N one-dimensional projectors $\left|\Psi_{j}\right\rangle\left\langle\Psi_{j}\right|$ . In most physical applications—in our case as well—the elements of the loop are specified as projectors to a eigenstates of some Hamiltonian for N different settings of some parameters.

As a generalization of the Berry phase, we ask about the relative phase around a loop on N projectors, each of which is $N_{F}$ dimensional. The physical motivation is that these are eigenspaces of a Hamiltonian, i.e., projectors to subspaces spanned by degenerate energy eigenstates [35].

## 9.1.1.1 Wilson Loop

Consider a loop over $N \geq 3$ sets of states from the Hilbert space, each set consisting of $N_F \in \mathbb{N}$ orthonormal states, $\{|u_n(k)\rangle | n = 1, \ldots, N_F\}$ , with $k = 1, \ldots, N$ . We quantify the overlap between set $k$ and set $l$ by the $N_F \times N_F$ overlap matrix $M^{(kl)}$ , with elements

$$
M _ {n m} ^ {(k l)} = \langle u _ {n} (k) | u _ {m} (l) \rangle ,\tag{9.1}
$$

with $n, m = 1 \ldots, N_F$ .

The Wilson loop is the product of the overlap matrices along the loop,

$$
W = M ^ {(1 2)} M ^ {(2 3)} \dots M ^ {(N - 1, N)} M ^ {(N 1)}.\tag{9.2}
$$

We will be interested in the eigenvalues $\lambda_{n}$ of the Wilson loop, with $n = 1, \ldots, N_{F}$ ,

$$
W \underline {{v}} _ {n} = \lambda_ {n} \underline {{v}} _ {n},\tag{9.3}
$$

where $\underline{v}_n$ is the $n$ th eigenvector, with $n = 1, \ldots, N_F$ .

Note that we could have started the Wilson loop, Eq. (9.2), at the kth group instead of the first one,

$$
W ^ {(k)} = M ^ {(k, k + 1)} M ^ {(k + 1, k + 2)} \dots M ^ {(N, 1)} M ^ {(1, 2)} \dots M ^ {(k - 1, k)}.\tag{9.4}
$$

Although the elements of the matrix $W^{(k)}$ depend on the starting point $k$ , the eigenvalues $\lambda_{n}$ do not. Multiplying Eq. (9.3) from the left by $M^{(k,k + 1)}\ldots M^{(N,1)}$ , we obtain

$$
W ^ {(k)} \left(M ^ {(k, k + 1)} \dots M ^ {(N, 1)}\right) \underline {{v}} _ {n} = \lambda_ {n} \left(M ^ {(k, k + 1)} \dots M ^ {(N, 1)}\right) \underline {{v}} _ {n}.\tag{9.5}
$$

## 9.1.1.2 $U(N_{F})$ Gauge Invariance of the Wilson Loop

We will now show that the eigenvalues of the Wilson loop over groups of Hilbert space vectors only depend on the linear spaces spanned by the vectors of each group. Each group can undergo an independent unitary operation to redefine the vectors, this cannot affect the eigenvalues of the Wilson loop. This is known as the invariance under a $U(N_{F})$ gauge transformation.

A simple route to prove the $U(N_{F})$ gauge invariance is via the operator $\tilde{W}$ defined by the Wilson loop matrix W in the basis of group 1,

$$
\hat {W} = \sum_ {n = 1} ^ {N _ {F}} \sum_ {m = 1} ^ {N _ {F}} | u _ {n} (1) \rangle W _ {n m} \langle u _ {m} (1) |.\tag{9.6}
$$

This operator has the same eigenvalues as the Wilson loop matrix itself. It can be expressed using the projectors to the subspaces spanned by the groups of states,

$$
\hat {P} _ {k} = \sum_ {n} | u _ {n} (k) \rangle \langle u _ {n} (k) |.\tag{9.7}
$$

The Wilson loop operator reads,

$$
\hat {W} = \hat {P} _ {1} \hat {P} _ {2} \hat {P} _ {3} \dots \hat {P} _ {N} \hat {P} _ {1}.\tag{9.8}
$$

We show this explicitly for $N = 3$ ,

$$
\begin{array}{l} \hat {W} = \sum_ {n = 1} ^ {N _ {F}} \sum_ {n _ {2} = 1} ^ {N _ {F}} \sum_ {n _ {3} = 1} ^ {N _ {F}} \sum_ {m = 1} ^ {N _ {F}} | u _ {n} (1) \rangle \\ \langle u _ {n} (1) | u _ {n _ {2}} (2) \rangle \langle u _ {n _ {2}} (2) | u _ {n _ {3}} (3) \rangle \langle u _ {n _ {N}} (N) | u _ {m} (1) \rangle \langle u _ {m} (1) | = \hat {P} _ {1} \hat {P} _ {2} \hat {P} _ {3} \hat {P} _ {1}. \end{array}\tag{9.9}
$$

The generalization to arbitrary $N \geq 3$ is straightforward.

Equation (9.8) makes it explicit that the Wilson loop operator, and hence, the eigenvalues of the Wilson loop, are $U(N_{F})$ gauge invariant.

## 9.1.2 Wannier States for Degenerate Multiband One-Dimensional Insulators

We now generalize the Wannier states of Sect. 3.2 to a one-dimensional insulator with $N_{F}$ occupied bands. In case of nondegenerate bands, a simple way to go would be to define a set of Wannier states for each band separately. However, time reversal symmetry forces degeneracies in the bands, at least at time reversal invariant momenta, and so this is not possible. Moreover, even in the nondegenerate case it could be advantageous to mix states from different bands to create more tightly localized Wannier states.

To be specific, and to obtain efficient numerical protocols, we take a finite sample of a one-dimensional insulator of N = 2M unit cells, with periodic boundary conditions. An orthonormal set of negative energy bulk eigenstates reads

$$
\left| \Psi_ {n} (k) \right\rangle = | k \rangle \otimes \left| u _ {n} (k) \right\rangle = \frac {1}{\sqrt {N}} \sum_ {m = 1} ^ {N} e ^ {i m k} | m \rangle \otimes \left| u _ {n} (k) \right\rangle\tag{9.10}
$$

with, as before, $k \in \{\delta_k, 2\delta_k, \ldots, N\delta_k\}$ , and $\delta_k = 2\pi / N$ . The index $n$ labels the eigenstates, with $n = 1, \ldots, N_F$ for occupied, negative energy states. The $|u_n(k)\rangle$ are the negative energy eigenstates of the bulk momentum-space Hamiltonian $\hat{H}(k)$ . We will not be interested in the positive energy eigenstates.

Although we took a specific set of energy eigenstates above, because of degeneracies at the time-reversal invariant momenta, we really only care about the projector $\hat{P}$ to the negative energy subspace. This is defined as

$$
\hat {P} = \sum_ {k} \sum_ {n = 1} ^ {N _ {F}} | \Psi_ {n} (k) \rangle \langle \Psi_ {n} (k) | = \sum_ {k} | k \rangle \langle k | \otimes \hat {P} (k);\tag{9.11}
$$

$$
\hat {P} (k) = \sum_ {n = 1} ^ {N _ {F}} | u _ {n} (k) \rangle \langle u _ {n} (k) |.\tag{9.12}
$$

## 9.1.2.1 Defining Properties of Wannier States

We will need a total number $N_{F}N$ of Wannier states to span the occupied subspace, $|w_{n}(j)\rangle$ , with $j = 1, \ldots, N$ , and $n = 1, \ldots, N_{F}$ . These are defined by the usual properties:

$$
\left\langle w _ {n ^ {\prime}} (j ^ {\prime}) \mid w _ {n} (j) \right\rangle = \delta_ {j ^ {\prime} j} \delta_ {n ^ {\prime} n} \quad \text { Orthonormal   set }\tag{9.13a}
$$

$$
\sum_ {j = 1} ^ {N} \sum_ {n = 1} ^ {N _ {F}} | w _ {n} (j) \rangle \langle w _ {n} (j) | = \hat {P}
$$

Span occupied subspace

(9.13b)

$$
\forall m: \langle m + 1 \mid w _ {n} (j + 1) \rangle = \langle m \mid w _ {n} (j) \rangle \quad \text { Related   by   translation }\tag{9.13c}
$$

$$
\lim _ {N \rightarrow \infty} \left\langle \right. w _ {n} (N / 2) \left. \right| (\hat {x} - N / 2) ^ {2} \left| \right. w _ {n} (N / 2) \left. \right\rangle <   \infty \quad \text { Localization }\tag{9.13d}
$$

with the addition in Eq. (9.13c) defined modulo N.

The Ansatz of Sect. 3.2 for the Wannier states, Eq. (3.11), generalizes to the multiband case as

$$
\left| w _ {n} (j) \right\rangle = \frac {1}{\sqrt {N}} \sum_ {k = \delta_ {k}} ^ {N \delta_ {k}} e ^ {- i j k} \sum_ {p = 1} ^ {N _ {F}} U _ {n p} (k) \left| \Psi_ {p} (k) \right\rangle .\tag{9.14}
$$

Thus, each Wannier state can contain contributions from all of the occupied bands, the corresponding weights given by a k-dependent unitary matrix $U(k)$ .

## 9.1.2.2 The Projected Unitary Position Operator

As we did in Sect. 3.2, we will specify the set of Wannier states as the eigenstates of the unitary position operator restricted the occupied bands,

$$
\hat {X} _ {P} = \hat {P} e ^ {i \delta_ {k} \hat {x}} \hat {P}.\tag{9.15}
$$

To obtain the Wannier states, we go through the same steps as in Sect. 3.2, with an extra index n. We outline the derivations and detail some of the steps below. You can then check whether the properties required of Wannier states, Eq. (9.13), are fulfilled, in the same way as in the single-band case.

We note that for finite N, the projected unitary position $\hat{X}_{P}$ is not a normal operator, i.e., it does not commute with its adjoint. As a result, its eigenstates form an orthonormal set only in the thermodynamic limit of $N \to \infty$ . Just as in the single-band case, this can be seen as a discretization error, which disappears in the limit $N \to \infty$ .

The first step is to rewrite the operator $\hat{X}_{P}$ . For this, consider

$$
\left\langle \Psi_ {n ^ {\prime}} (k ^ {\prime}) \right| \hat {X} \left| \Psi_ {n} (k) \right\rangle = \delta_ {k + \delta_ {k}, k ^ {\prime}} \left\langle u _ {n ^ {\prime}} (k + \delta_ {k}) \mid u _ {n} (k) \right\rangle\tag{9.16}
$$

where $\delta_{k+\delta_{k},k'} = 1$ if $k' = k + \delta_{k}$ , and 0 otherwise. Using this, the projected unitary position operator can be rewritten as

$$
\begin{array}{l} \hat {X} _ {P} = \sum_ {k ^ {\prime} k} \sum_ {n ^ {\prime}, n = 1} ^ {N _ {F}} \left| \Psi_ {n ^ {\prime}} (k ^ {\prime}) \right\rangle \left\langle \Psi_ {n ^ {\prime}} (k ^ {\prime}) \right| \hat {X} | \Psi_ {n} (k) \rangle \left\langle \Psi_ {n} (k) \right| \\ = \sum_ {k} \sum_ {n ^ {\prime}, n = 1} ^ {N _ {F}} \left\langle u _ {n ^ {\prime}} (k + \delta_ {k}) \mid u _ {n} (k) \right\rangle \cdot \left| \Psi_ {n ^ {\prime}} (k + \delta_ {k}) \right\rangle \left\langle \Psi_ {n} (k) \right|. \end{array}\tag{9.17}
$$

## 9.1.2.3 Spectrum of the Projected Unitary Position Operator and the Wilson Loop

As in the single-band case, the next step is to consider $\hat{X}_P$ raised to the $N$ th power. This time, it will not be simply proportional to the projector $\hat{P}$ , however. Bearing in mind the orthonormality of the energy eigenstates, $\langle \Psi_n(k) \mid \Psi_{n'}(k') \rangle = \delta_{k'k} \delta_{n'n}$ , we find

$$
\left(\hat {X} _ {P}\right) ^ {N} = \sum_ {k} \sum_ {m n} W _ {m n} ^ {(k)} \left| \Psi_ {m} (k) \right\rangle \left\langle \Psi_ {n} (k) \right|.\tag{9.18}
$$

The Wilson loop matrices $W^{(k)}$ , as per Eq. (9.4), are all unitary equivalent, and have the same set of complex eigenvalues,

$$
\lambda_ {n} = | \lambda_ {n} | e ^ {i \theta_ {n}} \quad \mathrm{with} \quad n = 1, \ldots , N _ {F},\tag{9.19}
$$

$$
\left| \lambda_ {n} \right| \leq 1, \quad \theta_ {n} \in [ - \pi , \pi).\tag{9.20}
$$

The spectrum of eigenvalues of $\hat{X}_P$ is therefore composed of the $N$ th roots of these eigenvalues, for $j = 1,\dots ,N$ , and $n = 1,\dots ,N_F$ ,

$$
\lambda_ {n, j} = e ^ {i \theta_ {n} / N + i j \delta_ {k} + \log (| \lambda_ {n} |) / N}, \quad \Longrightarrow \quad (\lambda_ {n, j}) ^ {N} = \lambda_ {n}.\tag{9.21}
$$

## 9.1.2.4 Wannier Centers Identified Through the Eigenvalues of the Wilson Loop

As in the single-band case, Sect. 3.2, we identify the phases of the eigenvalues $\lambda_{n,j}$ of the projected position operator $\hat{X}_P$ with the centers of the Wannier states. There are $N_F$ sets of Wannier states, each set containing states that are spaced by distances of 1,

$$
\langle x \rangle_ {n, j} = \frac {N}{2 \pi} \arg \lambda_ {n, j} = \langle x \rangle_ {n} + j;\tag{9.22}
$$

$$
\langle x \rangle_ {n} = \frac {\theta_ {n}}{2 \pi}.\tag{9.23}
$$

The phases $\theta_{n}$ of the $N_{F}$ eigenvalues of the Wilson loop W are thus identified with the Wannier centers, more precisely, with the amount by which the $N_{F}$ sets are displaced from the integer positions.

## 9.2 Time-Reversal Restrictions on Wannier Centers

We will now apply the prescription for Wannier states above to the one-dimensional insulators obtained as slices of a two-dimensional $\hat{J}^{2} = -1$ time-reversal invariant insulator at constant $k_{y}$ . We will use the language of dimensional reduction, i.e., talk of the bulk Hamiltonian $\hat{H}(k_{x}, k_{y})$ as describing an adiabatic particle pump with $k_{y}$ playing the role of time. We will use the Wannier center flow, i.e., the quantities $\langle x \rangle_{n} = \theta_{n}(k_{y}) / (2\pi)$ to track the motion of the particles in the bulk during a fictitious pump cycle, $k_{y} = -\pi \rightarrow \pi$ .

Time-reversal symmetry places constraints on the Wannier center flow in two ways: it enforces $k_{y} \leftrightarrow -k_{y}$ symmetry, i.e., $\theta_{n}(k_{y}) = \theta_{n'}(-k_{y})$ , and it ensures that for $k_{y} = 0$ and for $k_{y} = \pi$ , the $\theta_{n}$ are doubly degenerate. In this Section we see how these constraints arise.

## 9.2.1 Eigenstates at k and -k Are Related

A consequence of time-reversal symmetry is that energy eigenstates at $\mathbf{k}$ can be transformed to eigenstates at $-\mathbf{k}$ . One might think that because of time-reversal symmetry, energy eigenstates come in time-reversed pairs, i.e., that $\hat{\tau}\hat{H}(-\mathbf{k})^*\hat{\tau}^\dagger = \hat{H}(\mathbf{k})$ would automatically ensure that $|u_n(-\mathbf{k})\rangle = e^{i\phi (\mathbf{k})}\hat{\tau}|u_n(\mathbf{k})^*\rangle$ . However, because of possible degeneracies, this is not necessarily the case. The most we can say is that the state $|u_n(-\mathbf{k})\rangle$ is some linear combination of time-reversed eigenstates,

$$
\left| u _ {n} (- \mathbf {k}) \right\rangle = \hat {\tau} \sum_ {m = 1} ^ {N _ {F}} \left(B _ {n m} (\mathbf {k}) \left| u _ {m} (\mathbf {k}) \right\rangle\right) ^ {*} = \sum_ {m = 1} ^ {N _ {F}} B _ {n m} (\mathbf {k}) ^ {*} \hat {\tau} \left| u _ {m} (\mathbf {k}) ^ {*} \right\rangle .\tag{9.24}
$$

The coefficients $B_{nm}(\mathbf{k})$ define the unitary sewing matrix. An explicit formula for its matrix elements is obtained by multiplying the above equation from the left by $\langle u_a(\mathbf{k})^*| \hat{\tau}^\dagger$ , with some $a = 1, \dots, N_F$ . This has the effect on the left- and right-hand side of Eq. (9.24) of

$$
\langle u _ {a} (\mathbf {k}) ^ {*} | \hat {\tau} ^ {\dagger} | u _ {n} (- \mathbf {k}) \rangle = \left(\langle u _ {n} (- \mathbf {k}) | \hat {\tau} | u _ {a} (\mathbf {k}) ^ {*} \rangle\right) ^ {*};\tag{9.25}
$$

$$
\langle u _ {a} (\mathbf {k}) ^ {*} | \hat {\tau} ^ {\dagger} \sum_ {m = 1} ^ {N _ {F}} B _ {n m} (\mathbf {k}) ^ {*} \hat {\tau} | u _ {m} (\mathbf {k}) ^ {*} \rangle = B _ {n a} (\mathbf {k}) ^ {*},\tag{9.26}
$$

where for the last equation we used the unitarity of $\hat{\tau}$ and the orthonormality of the set $|u_{m}(\mathbf{k})\rangle$ . Comparing the two lines above (and relabeling $a \to m$ ), we obtain

$$
B _ {n m} (\mathbf {k}) = \left\langle u _ {n} (- \mathbf {k}) \right| \hat {\tau} \left| u _ {m} (\mathbf {k}) ^ {*} \right\rangle .\tag{9.27}
$$

Using this definition it is straightforward to show that the sewing matrix is unitary, and that $B_{mn}(-\mathbf{k}) = -B_{nm}(\mathbf{k})$ .

## 9.2.2 Wilson Loops at $k_{y}$ and $-k_{y}$ Have the Same Eigenvalues

To see the relation between the Wilson loops at $k_{y}$ and $-k_{y}$ , we first relate the projectors to the occupied subspace at these momenta. We use a shorthand,

$$
\hat {P} _ {j} (k _ {y}) = \left\{ \begin{array}{l l} \hat {P} (2 \pi + j \delta_ {k}, k _ {y}) & \text { if } j \leq 0; \\ \hat {P} (j \delta_ {k}, k _ {y}), & \text { if } j > 0. \end{array} \right.\tag{9.28}
$$

Using Eq. (9.24), and the unitarity of the sewing matrix B, we find

$$
\begin{array}{c} \hat {P} _ {- j} (- k _ {y}) = \hat {P} (- {\mathbf k}) = \sum_ {n = 1} ^ {N _ {F}} | u _ {n} (- {\mathbf k}) \rangle   \langle u _ {n} (- {\mathbf k}) | \\ = \sum_ {n = 1} ^ {N _ {F}} \sum_ {m = 1} ^ {N _ {F}} \sum_ {m ^ {\prime} = 1} ^ {N _ {F}} B _ {n m} ({\mathbf k}) ^ {*} \hat {\tau} | u _ {m} ({\mathbf k}) ^ {*} \rangle   B _ {n m ^ {\prime}} ({\mathbf k})   \langle u _ {m ^ {\prime}} ({\mathbf k}) ^ {*} |   \hat {\tau} ^ {\dagger} \\ = \hat {\tau} \hat {P} _ {j} (k _ {y}) ^ {*} \tau^ {\dagger} = \hat {\tau} \hat {P} _ {j} (k _ {y}) ^ {T} \tau^ {\dagger}. \end{array}\tag{9.29}
$$

The consequence of Eq. (9.29) for the Wilson loop is

$$
\hat {W} (- k _ {y}) = \hat {\tau} \hat {W} (k _ {y}) ^ {T} \hat {\tau} ^ {\dagger}.\tag{9.30}
$$

We write down the proof explicitly for N = 6,

$$
\begin{array}{c} \hat {W} (- k _ {y}) = \hat {P} _ {3} (- k _ {y}) \hat {P} _ {2} (- k _ {y}) \hat {P} _ {1} (- k _ {y}) \hat {P} _ {0} (- k _ {y}) \hat {P} _ {- 1} (- k _ {y}) \hat {P} _ {- 2} (- k _ {y}) \hat {P} _ {3} (- k _ {y}) \\ = \hat {\tau} \hat {P} _ {3} (k _ {y}) ^ {T} \hat {P} _ {- 2} (k _ {y}) ^ {T} \hat {P} _ {- 1} (k _ {y}) ^ {T} \hat {P} _ {0} (k _ {y}) ^ {T} \hat {P} _ {1} (k _ {y}) ^ {T} \hat {P} _ {2} (k _ {y}) ^ {T} \hat {P} _ {3} (k _ {y}) ^ {T} \hat {\tau} ^ {\dagger} \\ = \hat {\tau} \hat {W} (k _ {y}) ^ {T} \hat {\tau} ^ {\dagger}, \end{array}\tag{9.31}
$$

the generalization to arbitrary even N follows the same lines. The set of eigenvalues of $\hat{W}$ is the same as that of its transpose $\hat{W}^{T}$ , as this holds for any matrix. Moreover, the unitary transformation of $\hat{W}^{T}$ to $\hat{\tau}\hat{W}^{T}\hat{\tau}^{\dagger}$ does not change the eigenvalues either. To summarize, we find that the eigenvalues of the Wilson loop at $-k_{y}$ are the same as of the Wilson loop at $k_{y}$ ,

$$
\theta_ {n} (k _ {\mathrm{y}}) = \theta_ {n} (- k _ {\mathrm{y}}).\tag{9.32}
$$

From Eq. (9.32), we have that the Wannier center flow is symmetric around $k_{y} = 0$ , and hence, also symmetric around $k_{y} = \pi$ . This means that it is enough to examine the Wannier centers from $k_{y} = 0$ to $k_{y} = \pi$ .

## 9.2.3 Wilson Loop Eigenvalues at $k_{y} = 0$ and $k_{y} = \pi$ Are Doubly Degenerate

We now concentrate on the two special values of the $y$ wavenumber, $k_y = 0$ and $k_y = \pi$ , which are mapped unto themselves by time reversal. The one-dimensional Hamiltonians $\hat{H}_{\mathrm{bulk}}(k_x,0)$ and $\hat{H}_{\mathrm{bulk}}(k_x,\pi)$ are time-reversal invariant. Therefore, due to the Kramers theorem, each of their eigenstates $|\Psi \rangle$ has a time-reversed partner $\hat{\mathcal{T}} |\Psi \rangle$ , the two have the same energy, and are orthogonal, $\langle \Psi |\hat{\mathcal{T}} |\Psi \rangle = 0$ .

The eigenvalues of the Wilson loop $\hat{W}$ at $k_{y} = 0$ and $k_{y} = \pi$ are doubly degenerate. To show this, take an eigenstate of the Wilson loop, $\hat{W} |\Psi\rangle = \lambda |\Psi\rangle$ . Using Eq. (9.31), we find

$$
\lambda | \Psi \rangle = \hat {W} | \Psi \rangle = \tau \tau^ {\dagger} \hat {W} \tau \tau^ {\dagger} | \Psi \rangle = \hat {\tau} \hat {W} ^ {T} \hat {\tau} ^ {\dagger} | \Psi \rangle ;\tag{9.33}
$$

$$
\hat {W} ^ {\dagger} \hat {\tau} | \Psi^ {*} \rangle = \lambda^ {*} \hat {\tau} | \Psi^ {*} \rangle .\tag{9.34}
$$

We obtained line (9.34) by multiplication from the left by $\hat{\tau}$ and complex conjugation, and using the antisymmetry of $\hat{\tau}$ . In the final line, we have obtained that the Wilson loop $\hat{W}$ has a left eigenvector with eigenvalue $\lambda^{*}$ . Since this is orthogonal to $|\Psi\rangle$ , however, the right eigenvalue $\lambda$ must be at least twice degenerate.

## 9.3 Two Types of Wannier Center Flow

We now examine the Wannier center flow, i.e., the functions $\theta_{n}(k_{y})$ , in time-reversal invariant two-dimensional insulators with $\hat{J}^{2} = -1$ . Due to the restrictions of $k_{y} \leftrightarrow -k_{y}$ symmetry and degeneracy at $k_{y} = 0, \pi$ , we will find two classes of Wannier center flow. In the trivial class, the Wannier center flow can be adiabatically (i.e., continuously, while respecting the restrictions) deformed to the trivial case, with $\theta_{n} = 0$ for every n and every $k_{y}$ . The topological class is the set of cases where this is not possible.

To have a concrete example at hand, we examine the Wannier center flow for the BHZ model of the previous chapter, Eq. (8.38), in a trivial (a) and in a topological (b) case, and a third, more general topological (c) model. All three cases are covered by a modified BHZ Hamiltonian,

$$
\begin{array}{r l} \hat {H} (\mathbf {k}) & = \hat {s} _ {0} \otimes [ (u + \cos k _ {x} + \cos k _ {y}) \hat {\sigma} _ {z} + \sin k _ {y} \hat {\sigma} _ {y}) ] + \hat {s} _ {z} \otimes \sin k _ {x} \hat {\sigma} _ {x} + \hat {s} _ {x} \otimes \hat {C} + \\ & \quad g \hat {s} _ {z} \otimes \hat {\sigma} _ {y} (\cos k _ {x} + \cos 7 k _ {y} - 2). \end{array} \tag {9.3}\tag{9.35}
$$

For case (a), we set g = 0, take the sublattice potential parameter u = 2.1, and coupling operator $\hat{C} = 0.02\hat{\sigma}_{y}$ . This is adiabatically connected to the trivial limit of the BHZ model at $u = +\infty$ . In case (b), we set g = 0, take sublattice potential parameter u = 1, and coupling operator $\hat{C} = 0.3\hat{\sigma}_{y}$ , deep in the topological regime. In case (c) we add the extra term to the BHZ model to have a more generic case, with g = 0.1, and use u = 1, coupling $\hat{C} = 0.1\hat{\sigma}_{y}$ . The Wannier center flows for the three cases are shown in Fig. 9.1.

We consider how adiabatic deformations of the Hamiltonian can affect the Wannier center flow. Focusing on $k_{y}=0\to\pi$ , the center flow consists of branches $\theta_{n}(k_{y})$ , that are continuous functions of $k_{y}$ , beginning at $\theta_{n}(0)$ and ending at $\theta_{n}(\pi)$ .

![](images/85aee190b67b083ddafd0bf5a9a4a4bf196652ff9bb3140f68e0210e9eb937a9.jpg)
Fig. 9.1 Wannier center flow examples, corresponding to the four-band model of Eq. (9.35). The number of negative-energy, filled bands is $N_{F} = 2$ , hence the graphs show the evolution of two Wannier centers ( $\theta$ ) as a function of the momentum $k_{y}$ . See text for parameter specifications

Due to an adiabatic deformation,

• A branch can bend while $\theta_{n}(0)$ and $\theta_{n}(\pi)$ are fixed;

\- The endpoint at $k_y = 0$ (or $k_y = \pi$ ) of a branch can shift: in that case, the endpoint of the other branch, the Kramers partner at $k_y = 0$ (or $k_y = \pi$ ) is shifted with it;

\- Branches $\theta_{n}$ and $\theta_{m}$ can recombine: a crossing between them at some $k_{y}$ can turn into an avoided crossing.

Consider the example of Fig. 9.1. Bending of the branches and shifting of the endpoints can bring case (a) to a trivial case, where all branches are vertical, $\theta_{n}(k_{y}) = 0$ for every $n$ and $k_{y}$ . Case (b) can be deformed to case (c). Notice, however, that neither cases (b) nor (c) can be deformed to the trivial case.

## 9.3.1 Bulk Topological Invariant

We define the bulk topological invariant $N_{bulk}$ , by choosing some fixed $\tilde{\theta} \in [-\pi, \pi)$ , and asking for the parity of the number of times the Wannier center flow crosses this $\tilde{\theta}$ , as $k_{y}$ is varied between 0 and $\pi$ . In formulas,

$$
\tilde {\theta} \in [ - \pi , \pi);\tag{9.36}
$$

$$
N _ {n} (\tilde {\theta}) = \text { Number   of   solutions } k _ {y} \text { of } \theta_ {n} (k _ {y}) = \tilde {\theta}, k _ {y} \in [ 0, \pi ];\tag{9.37}
$$

$$
N _ {\text { bulk }} = \left(\sum_ {n = 1} ^ {N _ {F}} N _ {n} (\tilde {\theta})\right) \mod 2 \quad \text {(independent of \tilde {\theta}).}\tag{9.38}
$$

The number $N_{bulk}$ is invariant under adiabatic deformations of the bulk Hamiltonian, as can be shown by considering the possible changes. Bending of a branch $\theta_{n}$ can create or destroy solutions of $\theta_{n}(k_{y}) = \tilde{\theta}$ , but only pairwise. Shifting of the endpoint can create or destroy single solutions of $\theta_{n}(k_{y}) = \tilde{\theta}$ , but in that case, a single solution of $\theta_{m}(k_{y}) = \tilde{\theta}$ , is also created/destroyed, where $\theta_{m}$ is the Kramers partner of $\theta_{n}$ at the endpoint. Finally, recombination of branches cannot change the number of crossings. The number $N_{bulk}$ is also invariant under a shift of $\tilde{\theta}$ , as already announced. A shifting of $\tilde{\theta}$ is equivalent to a shifting of the Wannier center flow, whose effects we already considered above.

## 9.3.1.1 The Bulk Topological Invariant Is the $\mathbb{Z}_2$ Invariant of the Previous Chapter

The full proof that the bulk invariant $N_{bulk}$ is the same as the parity D of the number of edge state pairs, Eq. (8.40), is quite involved [16, 37]. We content ourselves with just pointing out here that both $N_{bulk}$ and D represent obstructions to deform the

Hamiltonian adiabatically to the so-called atomic limit, when the unit cells are completely disconnected from each other. Clearly, switching of a charge pump requires that there are no edge states present, and therefore, $N_{bulk} = 0$ requires D = 0. To show that the converse is true is more complicated, and we do not discuss it here.

## 9.4 The $Z_{2}$ Invariant for Systems with Inversion Symmetry

For two-dimensional time-reversal invariant insulators with inversion (i.e., parity) symmetry, the $Z_{2}$ topological invariant becomes very straightforward. We state the result below, and leave the proof as an exercise for the reader.

## 9.4.1 Definition of Inversion Symmetry

As introduced in Sect. 3.3, the operation of inversion, $\hat{\Pi}$ , acts on the bulk momentum-space Hamiltonian using an operator $\hat{\pi}$ , by

$$
\hat {\varPi} \hat {H} (\mathbf {k}) \hat {\varPi} ^ {- 1} = \hat {\pi} \hat {H} (- \mathbf {k}) \hat {\pi} ^ {\dagger}.\tag{9.39}
$$

We now require the operator $\hat{\pi}$ not only to be independent of the wavenumber k, to be unitary, Hermitian, but also to commute with time reversal, i.e.,

$$
\hat {\Pi} ^ {\dagger} \hat {\Pi} = 1; \qquad \hat {\Pi} ^ {2} = 1; \qquad \hat {\mathcal {T}} \hat {\Pi} = \hat {\Pi} \hat {\mathcal {T}}.\tag{9.40}
$$

## 9.4.2 At a Time-Reversal Invariant Momentum, the Kramers Pairs Have the Same Inversion Eigenvalue

Consider the time-reversal invariant momenta, $\varGamma_{j}$ . In the BHZ model, these are $(k_{x}, k_{y}) = (0, 0), (0, \pi), (\pi, 0), (\pi, \pi)$ . In general there are $2^{d}$ such momenta in a d-dimensional lattice model. Each eigenstate $\left|u(\varGamma_{j})\right\rangle$ of the bulk momentum-space Hamiltonian at these momenta has an orthogonal Kramers pair $\hat{\mathcal{T}}\left|u(\varGamma_{j})\right\rangle$ ,

$$
\hat {H} (\Gamma_ {j}) \left| u (\Gamma_ {j}) \right\rangle = E \left| u (\Gamma_ {j}) \right\rangle \quad \Longrightarrow \quad \hat {H} (\Gamma_ {j}) \hat {\mathcal {T}} \left| u (\Gamma_ {j}) \right\rangle = E \hat {\mathcal {T}} \left| u (\Gamma_ {j}) \right\rangle .\tag{9.41}
$$

If $\hat{H}$ is inversion symmetric, $|u\rangle$ can be chosen to be an eigenstate of $\hat{\pi}$ as well, since

$$
\hat {\pi} \hat {H} (\Gamma_ {j}) \hat {\pi} = \hat {H} (- \Gamma_ {j}) = \hat {H} (\Gamma_ {j}).\tag{9.42}
$$

Therefore,

$$
\hat {\pi} \left| u (\Gamma_ {j}) \right\rangle = \pm \left| u (\Gamma_ {j}) \right\rangle .\tag{9.43}
$$

The Kramers pair of $\left|u(\Gamma_j)\right\rangle$ has to have the same inversion eigenvalue as $\left|u(\Gamma_j)\right\rangle$ ,

$$
\hat {\pi} \hat {\mathcal {T}} \left| u (\Gamma_ {j}) \right\rangle = \hat {\mathcal {T}} \hat {\pi} \left| u (\Gamma_ {j}) \right\rangle = \pm \hat {\mathcal {T}} \left| u (\Gamma_ {j}) \right\rangle .\tag{9.44}
$$

In a system with both time-reversal and inversion symmetry, we get $2^{d}$ topological invariants of the bulk Hamiltonian, one for each time-reversal invariant momentum $\varGamma_{j}$ . These are the products of the parity eigenvalues $\xi_{m}(\varGamma_{j})$ of the occupied Kramers pairs at $\varGamma_{j}$ . However, inversion symmetry is usually broken at the edges, and so these invariants do not give rise to robust edge states.

The product of the inversion eigenvalues of all occupied Kramers pairs at all the time-reversal invariant momenta $\Gamma_{j}$ is the same as the $Z_{2}$ invariant,

$$
(- 1) ^ {N _ {\text { bulk }}} = \prod_ {j} \prod_ {m} \xi_ {m} (\Gamma_ {j}).\tag{9.45}
$$

We leave the proof of this useful result as an exercise for the reader.

## 9.4.3 Example: The BHZ Model

A concrete example for inversion symmetry is given by the BHZ model of Sect. 8.2, with no coupling $\hat{C} = 0$ . It can be checked directly that this has inversion symmetry, with

$$
\hat {\pi} = \hat {s} _ {0} \otimes \hat {\sigma} _ {z}.\tag{9.46}
$$

To calculate the $Z_{2}$ invariant of the BHZ model, we take the four time-reversal invariant momenta, $k_{1}, k_{2}, k_{3}, k_{4}$ , are the combinations of $k_{x}, k_{y}$ with $k_{x} = 0, \pi$ and $k_{y} = 0, \pi$ . The Hamiltonian $\hat{H}_{\mathrm{BHZ}}(k_{x}, k_{y})$ at these momenta is proportional to the inversion operator,

$$
\hat {H} _ {\mathrm{BHZ}} (\mathbf {k} _ {1} = 0, 0) = (u + 2) \hat {\pi}; \quad \hat {H} _ {\mathrm{BHZ}} (\mathbf {k} _ {4} = \pi , \pi) = (u - 2) \hat {\pi};\tag{9.47}
$$

$$
\hat {H} _ {\mathrm{BHZ}} (\mathbf {k} _ {2} = 0, \pi) = u \hat {\pi}; \qquad \hat {H} _ {\mathrm{BHZ}} (\mathbf {k} _ {3} = \pi , 0) = u \hat {\pi}.\tag{9.48}
$$

In these cases the Hamiltonian and the inversion operator obviously have the same eigenstates. At each time-reversal invariant momentum, two of these states form one occupied Kramers pair and the two others one empty Kramers pair. If u > 2, at all four time-reversal invariant momenta, the occupied Kramers pair is the one with inversion eigenvalue (parity) of -1, and so Eq. (9.45) gives $N_{bulk} = 0$ . Likewise, if u < -2, the eigenvalues are all +1, and we again obtain $N_{bulk} = 0$ . For 0 < u < 2, we have P eigenvalues -1, -1, +1, -1 at the four time-reversal invariant momenta $k_{1}, k_{2}, k_{3}, k_{4}$ , respectively, whereas if -2 < u < 0, we have -1, +1, +1, +1. In both cases, Eq. (9.45) gives $N_{bulk} = 1$ . This indeed is the correct result, that we obtained via the Chern number earlier.

## Problems

9.1 Inversion symmetry and interlayer coupling in the BHZ model

Consider the BHZ model with layer coupling $\hat{C} = C\hat{s}_{x} \otimes \hat{\sigma}_{y}$ . This breaks the inversion symmetry $\hat{\pi} = \hat{s}_{0} \otimes \hat{\sigma}_{z}$ . Nevertheless, the Wannier centers of the Kramers pairs at $k_{y} = 0$ and $k_{y} = \pi$ are stuck to $\theta = 0$ or $\theta = \pi$ , and are only shifted by the extra term $\propto \hat{s}_{z}\hat{\sigma}_{y}$ added to the BHZ model in Eq. (9.35). Can you explain why? (hint: extra inversion symmetry)

## 9.2 Proof of the formula for the $\mathbb{Z}_2$ invariant of an inversion-symmetric topological insulator

Show, using the results of Sect. 3.3, that the $Z_{2}$ invariant of a two-dimensional time-reversal invariant and inversion symmetric insulator can be expressed using Eq. (9.45).

# Chapter 10 Electrical Conduction of Edge States

It is well known that the electrical conduction of ordinary metallic samples at room temperature shows the following two characteristics. First, there is a linear relation between the electric current I that flows through the sample and the voltage V that drops between the two ends of the sample: $I/V = G \equiv R^{-1}$ , where $G(R)$ is the conductance (resistance) of the sample. Second, the conductances $G_{i}$ of different samples made of the same metal but with different geometries show the regularity $G_{i}L_{i}/A_{i} = \sigma$ for $\forall i$ , where $L_{i}$ is the length of the sample and $A_{i}$ is the area of its cross section. The material-specific quantity $\sigma$ is called the conductivity. Conductors obeying both of these relations are referred to as Ohmic.

Microscopic theories describing the above behavior (e.g., Drude model, Boltzmann equation) rely on models involving impurities, lattice vibrations, and electron scattering within the material. Electrical conduction in clean (impurity-free) nanostructures at low temperature might therefore qualitatively deviate from the Ohmic case. Here, we demonstrate such deviations on a simple zero-temperature model of a two-dimensional, perfectly clean, constant-cross-section metallic wire, depicted in Fig. 10.1a. Then we describe how scattering at static impurities affects the conduction in general. Finally, we discuss electrical conduction in two-dimensional topological insulators. Even though these materials are band insulators judged from their bulk band structure, in finite-sized samples their edge states do conduct. As a central result in the field of topological insulators, we point out that this conduction shows a strong robustness against impurity scattering.

In previous chapters, and also here, we use two-dimensional models. The real samples used in electronic transport experiments, of course, are three dimensional. However, the electrons participating in transport are often confined to a flat, quasi-two-dimensional spatial region, which allows their description using two-dimensional models. This is the case, for example, in semiconductor quantum wells (see below the example of a HgTe-based quantum well). In such structures, the confinement along the third, say, z, dimension, is very narrow, therefore the excitation energies between the corresponding transverse modes are large, and hence the transitions between these can be disregarded. In that case, one can often focus on a single transverse mode along z, the one closest to the electronic Fermi energy, and disregard all other modes, thereby arriving to a two-dimensional model for the electrons. Alternatively, a few of the transverse modes in the vicinity of the Fermi energy might be relevant; in that case, those modes can still be incorporated in a two-dimensional model in the form of an integral degree of freedom.

![](images/027e2d3d0146db1d1899260a556317f5f3ed55bfd612f0a37b16ee591377e203.jpg)

![](images/7e49d954ca263e6c7703297750d804366830e26e9ad547cb0fa883e40ed9cc50.jpg)
Fig. 10.1 (a) Schematic representation of a clean quantum wire contacted to two electron reservoirs (contacts). (b) Occupations of electronic states in the contacts and the quantum wire in the nonequilibrium situation when a finite voltage V is applied between the left and right reservoirs

## 10.1 Electrical Conduction in a Clean Quantum Wire

As shown in Fig. 10.1a, take a wire that lies along the x axis, with length L and width W. Its width W along y can be defined by, for example, an electric confinement potential or lattice termination. Each electronic energy eigenfunction $|l, k\rangle$ in such a wire is a product of a standing wave along y, labeled by a positive integer l, and a plane wave propagating along x, labeled by a real wave number k (see Eq. (8.41)). A typical set of dispersion relations $E_{lk}$ ('subbands') for three different l indices is shown in Fig. 10.1b.

We also make assumptions on the two metallic contacts that serve as source and drain of electrons. We assume that the electrons in each contact are in thermal equilibrium, but the Fermi energies in the contacts differ by $\mu_{L}-\mu_{R}=|e|V>0$ . (Note that in this chapter, proper physical units are used, hence constants such as the elementary charge $|e|$ , reduced Planck's constant $\hbar$ , lattice constant a are reinstated.) We consider the linear conductance, that is, the case of an infinitesimal voltage $|e|V\to0$ . We further assume that both contacts absorb every incident electron with unit probability, and that the energy distribution of the electrons they emit is the thermal distribution with the respective Fermi energy.

These assumptions guarantee that the right-moving (left-moving) electronic states in the wire are occupied according to the thermal distribution of the left (right) contact, as illustrated in Fig. 10.1b. Now, we work with electron states normalized to the area of the channel. It is a simple fact that with this normalization convention, a single occupied state in the lth channel, with wave number k carries an electric current of $\frac{-|e|v_{lk}}{N_{x}a}$ , where $N_{x}a$ is the length of the wire, and $v_{lk} = \frac{1}{\hbar} \frac{dE_{lk}}{dk}$ is the group velocity of the considered state. Therefore, the current flowing through the wire is

$$
I = - | e | \frac {1}{N _ {x} a} \sum_ {l k} v _ {l k} \left[ f (E _ {l k} - \mu_ {L}) - f (E _ {l k} - \mu_ {R}) \right],\tag{10.1}
$$

where $f(\epsilon) = \left( \exp \frac{\epsilon}{k_{\mathrm{B}}T} + 1 \right)^{-1}$ is the Fermi-Dirac distribution. Converting the $k$ sum to an integral via $\frac{1}{N_x a} \sum_k \ldots \mapsto \int_{-\pi / a}^{\pi / a} \frac{dk}{2\pi} \ldots$ yields

$$
I = - | e | \sum_ {n} \int_ {- \pi / a} ^ {\pi / a} \frac {d k}{2 \pi} \frac {1}{\hbar} \frac {d E _ {l k}}{d k} \left[ f (E _ {l k} - \mu_ {L}) - f (E _ {l k} - \mu_ {R}) \right].\tag{10.2}
$$

The Fermi-Dirac distribution has a sharp edge at zero temperature, implying

$$
I = - \frac {| e |}{h} M \int_ {\mu_ {R}} ^ {\mu_ {L}} d E = - \frac {| e |}{h} (\mu_ {L} - \mu_ {R}) M = M \frac {e ^ {2}}{h} V\tag{10.3}
$$

Note that the first equality in $(10.3)$ holds only if the number of subbands intersected by $\mu_{L}$ and $\mu_{R}$ are the same, which is indeed the case if the voltage V is small enough. The number of these subbands, also called ‘open channels’, is denoted by the integer M. From $(10.3)$ it follows that the conductance of the wire is an integer multiple of $e^{2}/h$ (commonly referred to as ‘quantized conductance’):

$$
G = \frac {e ^ {2}}{h} M.\tag{10.4}
$$

The numerical value of $e^{2}/h$ is approximately 40 $\mu$ S (microsiemens), which corresponds to a resistance of approximately 26 k $\Omega$ . Note that the conductance quantum is defined as $G_{0} = 2e^{2}/h$ , i.e., as the conductance of a single open channel with twofold spin degeneracy.

It is instructive to compare the conduction in our clean quantum wire to the ordinary Ohmic conduction summarized above. According to $(10.3)$ , the proportionality between voltage and current holds for a clean quantum wire as well as for an ordinary metal. However, the dependence of the conductance on the length of the sample differs qualitatively in the two cases: in an ordinary metal, a twofold increase in the length of the wire halves the conductance, whereas the conductance of a clean quantum wire is insensitive to length variations.

Whether the conductance of the clean quantum wire is sensitive to variations of the wire width depends on the nature of the transversal modes. Conventional quantum wires that are created by a transverse confinement potential have a subband dispersion similar to that in Fig. 10.1b. There, the energy separation between the subbands decreases as the width of the wire is increased, therefore the number of subbands available for conduction increases. This leads to an increased conductance for an increased width, similarly to the case of ordinary metals. If, however, we consider a topological insulator, where the current is carried by states localized to the edges of the wire, the conductance of the wire will be insensitive to the width of the wire.

## 10.2 Phase-Coherent Electrical Conduction in the Presence of Scatterers

Having calculated the conductance (10.4) of a clean quantum wire, we now describe how this conductance is changed by the presence of impurities. We analyze the model shown in Fig. 10.2, where the disordered region, described by a scattering matrix S, is connected to the two contacts by two identical clean quantum wires, also called 'leads' in this context.

Fig. 10.2 Simple model of a phase-coherent conductor in the presence of scatterers. The ideal contacts L and R are connected via ideal leads to the disordered region represented by the scattering matrix S
![](images/2a866c0a1555094e9be5c6c991f3d29c2e55da39ef9f3213c974c2341d844080.jpg)
clean quantum wires ('leads')

First, we consider the case when each lead supports a single open channel. The current in the lead connecting contact L and the scattering region consists of a contribution from right-moving states arriving from contact L and partially backscattered with probability $R = |r|^{2}$ , and from left-moving states arriving from contact R and partially transmitted with probability $T' = |t'|^{2}$ :

$$
I = - | e | \frac {1}{N _ {x} a} \sum_ {k} v _ {k} \left[ (1 - R (E _ {k})) f _ {L} (E _ {k}) - T ^ {\prime} (E _ {k}) f _ {R} (E _ {k}) \right]\tag{10.5}
$$

Converting the k sum to an integral, assuming that the transmission and reflection probabilities are independent of energy in the small energy window between $\mu_{R}$ and $\mu_{L}$ , and using $1 - R = T = T'$ , we arrive at

$$
I = - \frac {| e |}{h} T \int_ {\mu_ {R}} ^ {\mu_ {L}} d E [ f _ {L} (E) - f _ {R} (E) ] = \frac {e ^ {2}}{h} T V,\tag{10.6}
$$

which implies that the conductance can be expressed through the transmission coefficient T:

$$
G = \frac {e ^ {2}}{h} T.\tag{10.7}
$$

The result $(10.7)$ can be straightforwardly generalized to the case when the leads support more than one open channel. The generalized result for the conductance, also known as the Landauer formula, reads:

$$
G = \frac {e ^ {2}}{h} \sum_ {n = 1} ^ {M} T _ {n},\tag{10.8}
$$

where $T_{n}$ are the transmission eigenvalues of the scattering matrix i.e., the real eigenvalues of the Hermitian matrix $tt^{\dagger}$ , as defined in the preceding chapter.

## 10.3 Electrical Conduction in Two-Dimensional Topological Insulators

After presenting the Landauer formula as a generic tool to describe electrical conduction of a phase-coherent metal, we will use it know to characterize the conductances of various two-dimensional topological insulator samples.

## 10.3.1 Chern Insulators

In Sect. 6.2, we have seen that an impurity-free straight strip of a topologically nontrivial Chern insulator supports edge states. The relation between the Chern number Q of the Chern insulator and the numbers of edge states at a single edge at a given energy E, propagating ‘clockwise’ $(N_{+}(E))$ and ‘anticlockwise’ $(N_{-}(E))$ , is $Q = N_{+}(E) - N_{-}(E)$ . In addition, in Sect. 6.3 it was shown that any segment of the edge of a disordered Chern insulator with Chern number Q and an arbitrary geometry supports $|Q|$ chiral edge modes. Here we show that existence of these edge modes leads to experimentally detectable effects in the electrical transport through Chern insulator samples.

We consider a transport setup where the Chern insulator is contacted with two metallic electrodes, as shown in Fig. 10.3. In this discussion, we rely on the usual assumptions behind the Landauer formula: phase-coherence of the electrons, good contact between contacts and sample, and large spatial separation of the two electrodes ensuring the absence of tunneling contributions to the conductance.

Fig. 10.3 A disordered sample of Chern insulator, with contacts 1 and 2, that can be used to pass current through the sample in order to detect edge states
![](images/6478b43a8802bce0817df296990c4cddf4eb17a18a28fcd70d2b3601b3a12c3e.jpg)

In the following list, we summarize how the phase-coherent electrical conductance of a Chern insulator varies with the sample geometry, absence or presence of disorder, and the value of the electronic Fermi energy.

## 1. Disorder-free sample with a strip geometry (see Fig. 10.1a)

a. Fermi energy lies in a band. In this case, the sample is a clean quantum wire (see Sect. 10.1) with an integer number of open channels. The corresponding transversal wave functions might or might not be localized to the sample edges, and therefore the number of channels might be different from any combination of $Q$ , $N_{+}$ or $N_{-}$ . According to Eq. (10.4), the conductance of such a clean quantum wire is quantized and insensitive to the length of the sample. Furthermore, the conductance grows in a step-like fashion if the width of the sample is increased.

b. Fermi energy lies in the gap. The sample is a clean quantum wire with open channels that are all localized to the sample edges. The number of those channels is $N_{+}(E) + N_{-}(E)$ , where E is the Fermi energy: in each of the two possible direction of current flow, there are $N_{+}$ channels on one edge and $N_{-}$ on the other edge that contribute to conduction. Conductance is finite and quantized, a behavior rather unexpected from an insulator. The conduction is not Ohmic, as the conductance is insensitive to both the length and the width of the sample.

## 2. Disordered sample with an irregular shape (see Fig. 10.3):

a. Fermi energy lies in a band. Because of the presence of disorder, the electrical conduction of such a sample might be Ohmic. There are no protected edge states at the Fermi energy.

b. Fermi energy lies well within the gap. According to Sect. 6.3, any edge segment of such a sample supports Q reflectionless chiral edge modes at the Fermi energy. Therefore, conductance is typically quantized, $G = |Q|e^{2}/h$ , although, disorder permitting, it might in principle be larger than this value. The quantized conductance is insensitive to changes in the geometry or the disorder configuration. This transport property, unexpected for an insulator, let alone for one with disorder, is a hallmark of Chern insulators.

In the case of two-dimensional samples there is often an experimental possibility of tuning the electronic Fermi energy in situ by controlling the voltage applied between the sample and a nearby metallic plate (gate electrode), as discussed in Sect. 10.4. This allows, in principle, to observe the changes in the electrical conduction of the sample as the Fermi energy is tuned across the gap.

## 10.3.2 Two-Dimensional Time-Reversal Invariant Topological Insulators with $\hat{T}^{2} = -1$

In the following list, we summarize the predictions of the Landauer formalism for the conductance of two-dimensional time-reversal invariant topological insulators with $\hat{\mathcal{T}}^{2} = -1$ ('D = 1 insulators' for short).

## 1. Disorder-free sample with a strip geometry:

a. Fermi energy lies in a band. A simple consequence of the Landauer formula is that phase-coherent conductance of an impurity-free D = 1 topological insulator of the strip geometry shown in Fig. 10.1a is quantized. The conductance grows if the width of the strip is increased, but it is insensitive to changes in the length.

b. Fermi energy lies in the gap. Only edge channels are open in this case. These also provide conductance quantization. As the number of edge-state Kramers pair per edge is odd, the conductance might be $2e^{2}/h$ , $6e^{2}/h$ , $10e^{2}/h$ , etc. Conductance is insensitive to width or length changes of the sample.

2. Disordered sample with an irregular shape and time-reversal symmetric disorder:

a. Fermi energy lies in a band. The electrical conduction might be Ohmic.

b. Fermi energy lies in the gap. We have shown in Chap. 8 that a D = 1 insulator supports one protected edge-state Kramers pair per edge, which allows for reflectionless electronic transmission if only time-reversal symmetric defects are present. The Landauer formula (10.8) implies, for typical cases, $G = 2e^{2}/h$ for such a sample, as one edge state per edge contributes to conduction. The conductance might also be larger, provided that the number of edge-state Kramers pairs is larger than 1 and disorder is ineffective in reducing the transmission of the topologically unprotected pairs.

We note that in real materials with $Z_{2}$ invariant D = 1, various mechanisms might lead to backscattering and, in turn, to $G < 2e^{2}/h$ . Examples include time-reversal-symmetry-breaking impurities, time-reversal symmetric impurities that bridge the spatial distance between the edges (see Chap. 8), hybridization of edge states from opposite edges in narrow samples, and inelastic scattering on phonons or spinful impurities.

## 10.4 An Experiment with HgTe Quantum Wells

Electrical transport measurements $[21]$ on appropriately designed layers of the semiconductor material mercury-telluride (HgTe) show signatures of edge-state conduction in the absence of magnetic field. These measurements are in line with the theoretical prediction that a HgTe layer with a carefully chosen thickness can realize a topologically nontrivial $(D = 1)$ two-dimensional time-reversal invariant insulator with $\hat{J}^{2} = -1$ . In this section, we outline the main findings of this experiment, as well as its relation to the BHZ model introduced and discussed in Chap. 8.

![](images/1834697d8702fc7ad9a55c01b71be1888e4fd093df4da49f78096211200b0bc6.jpg)
Fig. 10.4 Schematic representation of a HgTe quantum well of width d, sandwiched between two $Hg_{x}Cd_{1-x}Te$ layers. Electrons are confined to the HgTe layer, and their Fermi energy can be tuned in situ by adjusting the voltage $V_{gate}$ of the metallic electrode on the top of the sample (black). For a more accurate description of the experimental arrangement, see [20]

The experiments are performed on sandwich-like structures formed by a few-nanometer thick HgTe layer (quantum well) embedded between two similar layers of the alloy $Hg_{x}Cd_{1-x}Te$ , as shown in Fig. 10.4. (In the experiment reported in [21], the alloy composition x = 0.3 was used.) In this structure, the electronic states with energies close the Fermi energy are confined to the HgTe layer that is parallel to the x-y plane in Fig. 10.4. The energy corresponding to the confinement direction z is quantized. The carriers are free to move along the HgTe layer, i.e., parallel to the x-y plane, therefore two-dimensional subbands are formed in the HgTe quantum well. Detailed band structure calculations of [5] show that as the thickness d of the HgTe layer is decreased, the lowermost conduction subband and the uppermost valence subband touch at a critical thickness $d = d_{c}$ , and the gap is reopened for even thinner HgTe layers. (For the alloy composition x = 0.3 used in the experiment, the critical thickness is $d_{c} \approx 6.35$ nm.) This behavior is illustrated schematically in Fig. 10.5, which illustrates the electronic dispersions of the uppermost valence subband and the lowermost conduction subband, at the center of the Brillouin zone, for three different thicknesses of the quantum well.

Band structure calculations have also revealed a connection between the subbands depicted in Fig. 10.5 and the BHZ model introduced and discussed in Chap. 8. The $4 \times 4$ effective Hamiltonian describing the two spinful two-dimensional subbands around their extremum point at the centre of the HgTe Brillouin zone resembles the low-energy continuum Hamiltonian derived from the BHZ lattice model in the vicinity of the $u \approx -2$ value. Changing the thickness d of the HgTe layer corresponds to a change in the parameter u of the BHZ model, and the critical thickness $d = d_{c}$ corresponds to u = -2 and, consequently, a zero mass in the corresponding two-dimensional Dirac equation.

![](images/2eb80fa9add4c721c1bf80ec2e2deffa022614830b38e4f176cd999d1c3e8446.jpg)

![](images/fa4b80ae557cda0ac567b0bdadca8bf2e33b3a68a1448013a2e6d09a409bc651.jpg)

![](images/6fb829d1a9a7b0c66bf5d592e2a3ed6cfd2a9151bd74734769e1502b6eea1f4f.jpg)
Fig. 10.5 Evolution of the two-dimensional band structure of a HgTe quantum well as a function of its thickness d. (a) For a thin quantum well below the critical thickness $d < d_{c}$ , the band structure has a gap and the system is a trivial insulator. (b) At a critical thickness $d = d_{c}$ , the band gap closes and the system is metallic. (c) For a thick quantum well with $d > d_{c}$ , the band gap reopens and the system becomes a two-dimensional topological insulator

As a consequence of the strong analogy of the band structure of the HgTe quantum well and that of the BHZ model, it is expected that either for $d < d_{c}$ or for $d > d_{c}$ the material is a D = 1 insulator with a single Kramers pair of edge states. Arguments presented in [5] suggest that the thick quantum wells with $d > d_{c}$ are topologically nontrivial.

Electrical transport measurements were carried out in HgTe quantum wells patterned in the Hall bar geometry shown in Fig. 10.6. The quantity that has been used in this experiment to reveal edge-state transport is the four-terminal resistance $R_{14,23} = V_{23}/I_{14}$ , where $V_{23}$ is the voltage between contacts 2 and 3, and $I_{14}$ is the current flowing between contacts 1 and 4. This quantity $R_{14,23}$ was measured for various devices with different thicknesses d, below and above the critical thickness $d_{c}$ , of the HgTe layer, and for different values of the Fermi energy. The latter can be tuned in situ by controlling the voltage between the HgTe layer and a metallic ‘gate’ electrode on the top of the layered semiconductor structure, as shown in Fig. 10.4.

To appreciate the experimental result, let us first derive the four-terminal resistance $R_{14,23}$ for such a device. To this end, we express $V_{23}$ with $I_{14}$ . Ohm's law implies $V_{23} = I_{23} / G_{23}$ , where $I_{23}$ is the current flowing through the edge segment between contacts 2 and 3, whereas $G_{23}$ is the conductance of that edge segment. Furthermore, as the current $I_{14}$ flowing through terminals 1 and 4 is equally divided between the upper and lower edges, the relation $I_{23} = I_{14} / 2$ holds, implying the result $R_{14,23} = 1 / (2G_{23})$ .

![](images/5f0e8933df40a9bcfc77adc177e97384ea0657332c0b595b917289d2eb40d10f.jpg)
Fig. 10.6 HgTe quantum well patterned in the Hall-bar geometry (gray area). Numbered terminals lead to metallic contacts. Solid and dashed lines depict counterpropagating edge states

If the Fermi energy lies in the bands neighboring the gap, then irrespective of the topological invariant of the system, the HgTe quantum well behaves as a good conductor with $G_{23} \gg e^{2}/h$ , implying $R_{14,23} \ll h/e^{2}$ . If the Fermi energy is tuned to the gap in the topologically nontrivial case $d > d_{c}$ , then $G_{23} = e^{2}/h$ and therefore $R_{14,23} = h/(2e^{2})$ . This holds, of course, only at a temperature low enough and a sample size small enough such that phase coherence is guaranteed. The presence of static time-reversal invariant defects is included. If the system is topologically trivial ( $d < d_{c}$ ), then there is no edge transport, and the quantum well is a good insulator with $R_{14,23} \gg h/e^{2}$ .

The findings of the experiments are consistent with the above expectations. Furthermore, the four-terminal resistance of topologically nontrivial HgTe layers with different widths were measured, with the resistance found to be an approximately constant function of the width W of the Hall bar. This is a further indication that the current in these samples is carried by edge states.

To wrap up this chapter, we note that InAs/GaSb bilayer quantum wells are an alternative semiconductor material system where two-dimensional topological insulators can be realized $[9, 22]$ . Graphene is believed to be a two-dimensional topological insulator as well $[19]$ , even though its energy gap between the valence and conduction band, induced by spin-orbit interaction and estimated to be of the order of $\mu eV$ , seems to be too small to allow for the detection of edge-state transport even at the lowest available temperatures. The concept of a time-reversal invariant topological insulator can be extended to three-dimensional crystals as well, where the role of the edge states is played by states localized to the two-dimensional surface of the three-dimensional material. The description of such systems is out of the scope of the present course; the interested reader might consult, e.g., $[4, 17]$ .

1. A.A. Aligia, G. Ortiz, Quantum mechanical position operator and localization in extended systems. Phys. Rev. Lett. 82, 2560–2563 (1999)

2. Y. Ando, Topological insulator materials. J. Phys. Soc. Jpn. 82(10), 102001 (2013)

3. G. Bastard, Wave Mechanics Applied to Semiconductor Heterostructures (Les Editions de Physique, Les Ulis, 1988)

4. B.A. Bernevig, Topological Insulators and Topological Superconductors (Princeton University Press, Princeton, 2013)

5. B.A. Bernevig, T.L. Hughes, S.-C. Zhang, Quantum spin hall effect and topological phase transition in HgTe quantum wells. Science 314, 1757 (2006)

6. M.V. Berry, Quantal phase factors accompanying adiabatic changes. Proc. R. Soc. Lond. A 392, 45–57 (1984)

7. J.C. Budich, B. Trauzettel, From the adiabatic theorem of quantum mechanics to topological states of matter. Physica Status Solidi RRL 7(1–2), 109–129 (2013)

8. C.-Z. Chang, J. Zhang, X. Feng, J. Shen, Z. Zhang, M. Guo, K. Li, Y. Ou, P. Wei, L.-L. Wang, et al., Experimental observation of the quantum anomalous hall effect in a magnetic topological insulator. Science 340(6129), 167–170 (2013)

9. L. Du, I. Knez, G. Sullivan, R.-R. Du, Robust helical edge transport in gated InAs/GaSb bilayers. Phys. Rev. Lett. 114, 096802 (2015)

10. M. Franz, L. Molenkamp, Topological Insulators, vol. 6 (Elsevier, Oxford, 2013)

11. L. Fu, C.L. Kane, Time reversal polarization and a $Z_{2}$ adiabatic spin pump. Phys. Rev. B 74, 195312 (2006)

12. T. Fukui, Y. Hatsugai, H. Suzuki, Chern numbers in discretized brillouin zone: efficient method of computing (spin) hall conductances. J. Phys. Soc. Jpn. 74(6), 1674–1677 (2005)

13. I.C. Fulga, F. Hassler, A.R. Akhmerov, Scattering theory of topological insulators and superconductors. Phys. Rev. B 85, 165409 (2012)

14. A. Garg, Berry phases near degeneracies: Beyond the simplest case. Am. J. Phys. 78(7), 661–670 (2010)

15. D.J. Griffiths, Introduction to Quantum Mechanics (Pearson Education Limited, Harlow, 2014)

16. F. Grusdt, D. Abanin, E. Demler, Measuring $Z_{2}$ topological invariants in optical lattices using interferometry. Phys. Rev. A 89, 043621 (2014)

17. M.Z. Hasan, C.L. Kane, Colloquium: topological insulators. Rev. Mod. Phys. 82, 3045 (2010)

18. B.R. Holstein, The adiabatic theorem and Berry's phase. Am. J. Phys. 57(12), 1079-1084 (1989)

19. C.L. Kane, E.J. Mele, $Z_{2}$ topological order and the quantum spin hall effect. Phys. Rev. Lett. 95, 146802 (2005)

20. M. König, H. Buhmann, L.W. Molenkamp, T. Hughes, C.-X. Liu, X.-L. Qi, S.-C. Zhang, The quantum spin hall effect: theory and experiment. J. Phys. Soc. Jpn. 77(3), 031007 (2008)

21. M. König, S. Wiedmann, C. Brüne, A. Roth, H. Buhmann, L.W. Molenkamp, X.-L. Xi, S.-C. Zhang, Quantum spin hall insulator state in HgTe quantum wells. Science 318(6), 766–770 (2007)

22. C. Liu, T.L. Hughes, X.-L. Qi, K. Wang, S.-C. Zhang, Quantum spin hall effect in inverted type-ii semiconductors. Phys. Rev. Lett. 100, 236601 (2008)

23. N. Marzari, A.A. Mostofi, J.R. Yates, I. Souza, D. Vanderbilt, Maximally localized wannier functions: Theory and applications. Rev. Mod. Phys. 84, 1419–1475 (2012)

24. X.-L. Qi, Y.-S. Wu, S.-C. Zhang, Topological quantization of the spin hall effect in two-dimensional paramagnetic semiconductors. Phys. Rev. B 74, 085308 (2006)

25. X.-L. Qi, S.-C. Zhang, Topological insulators and superconductors. Rev. Mod. Phys. 83, 1057–1110 (2011)

26. R. Resta, Berry Phase in Electronic Wavefunctions. Troisieme Cycle de la Physique en Suisse Romande (1996)

27. R. Resta, Macroscopic polarization from electronic wavefunctions. arXiv preprint condmat/9903216 (1999)

28. R. Resta, What makes an insulator different from a metal? arXiv preprint cond-mat/0003014 (2000)

29. S. Ryu, A.P. Schnyder, A. Furusaki, A.W.W. Ludwig, Topological insulators and superconductors: tenfold way and dimensional hierarchy. New J. Phys. 12(6), 065010 (2010)

30. S.-Q. Shen, Topological insulators: Dirac equation in condensed matter. Springer Ser. Solid-State Sci. 174 (2012)

31. A.A. Soluyanov, D. Vanderbilt, Smooth gauge for topological insulators. Phys. Rev. B 85, 115415 (2012)

32. J. Sólyom, Fundamentals of the Physics of Solids: Volume III: Normal, Broken-Symmetry, and Correlated Systems, vol. 3 (Springer Science & Business Media, Berlin, 2008)

33. D.J. Thouless, Quantization of particle transport. Phys. Rev. B 27, 6083–6087 (1983)

34. G.E. Volovik, The Universe in a Helium Droplet (Oxford University Press, New York, 2009)

35. F. Wilczek, A. Zee, Appearance of gauge structure in simple dynamical systems. Phys. Rev. Lett. 52, 2111–2114 (1984)

36. D. Xiao, M.-C. Chang, Q. Niu, Berry phase effects on electronic properties. Rev. Mod. Phys. 82, 1959–2007 (2010)

37. R. Yu, X.L. Qi, A. Bernevig, Z. Fang, X. Dai, Equivalent expression of $Z_{2}$ topological invariant for band insulators using the non-abelian Berry connection. Phys. Rev. B 84, 075119 (2011)

38. J. Zak, Berry's phase for energy bands in solids. Phys. Rev. Lett. 62, 2747-2750 (1989)
