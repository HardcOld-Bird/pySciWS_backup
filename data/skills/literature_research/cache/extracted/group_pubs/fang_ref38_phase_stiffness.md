TOPICAL REVIEW

# A non-Hermitian Hamilton operator and the physics of open quantum systems

To cite this article: Ingrid Rotter 2009 J. Phys. A: Math. Theor. 42 153001

View the article online for updates and enhancements.

## You may also like

\- Landau–Lifshitz–Gilbert micromagnetic simulation on spin transfer torque efficiency of sub-30 nm perpendicular magnetic tunnel junctions with etching damage
Kenchi Ito, Satoshi Ohuchida, Masakazu Muraguchi et al.

\- Preparing topologically ordered states by Hamiltonian interpolation
Xiaotong Ni, Fernando Pastawski, Beni Yoshida et al.

\- Porous-Media Flow Fields for Polymer Electrolyte Fuel Cells: I. Low Humidity Operation
Yun Wang

TOPICAL REVIEW

# A non-Hermitian Hamilton operator and the physics of open quantum systems

Ingrid Rotter

Max-Planck-Institut für Physik Komplexer Systeme, D-01187 Dresden, Germany

E-mail: rotter@mpipks-dresden.mpg.de

Received 18 December 2008, in final form 20 February 2009

Published 20 March 2009

## Abstract

Online at stacks.iop.org/JPhysA/42/153001

The Hamiltonian $H_{eff}$ of an open quantum system consists formally of a first-order interaction term describing the closed (isolated) system with discrete states and a second-order term caused by the interaction of the discrete states via the common continuum of scattering states. Under certain conditions, the last term may be dominant. Due to this term, $H_{eff}$ is non-Hermitian. Using the Feshbach projection operator formalism, the solution $\Psi_{c}^{E}$ of the Schrödinger equation in the whole function space (with discrete as well as scattering states, and the Hermitian Hamilton operator H) can be represented in the interior of the localized part of the system in the set of eigenfunctions $\phi_{\lambda}$ of $H_{eff}$ . Hence, the characteristics of the eigenvalues and eigenfunctions of the non-Hermitian operator $H_{eff}$ are contained in observable quantities. Controlling the characteristics by means of external parameters, quantum systems can be manipulated. This holds, in particular, for small quantum systems coupled to a small number of channels. The paper consists of three parts. In the first part, the eigenvalues and eigenfunctions of non-Hermitian operators are considered. Most important are the true and avoided crossings of the eigenvalue trajectories. In approaching them, the phases of the $\phi_{\lambda}$ lose their rigidity and the values of observables may be enhanced. Here the second-order term of $H_{eff}$ determines decisively the dynamics of the system. The time evolution operator is related to the non-Hermiticity of $H_{eff}$ . In the second part of the paper, the solution $\Psi_{c}^{E}$ and the S matrix are derived by using the Feshbach projection operator formalism. The regime of overlapping resonances is characterized by non-rigid phases of the $\Psi_{c}^{E}$ (expressed quantitatively by the phase rigidity $\rho$ ). They determine the internal impurity of an open quantum system. Here, level repulsion passes into width bifurcation (resonance trapping): a dynamical phase transition takes place which is caused by the feedback between environment and system. In the third part, the internal impurity of open quantum systems is considered by means of concrete examples. Bound states in the continuum appearing at certain parameter values can be used in order to stabilize open quantum systems. Of special interest are the consequences of the non-rigidity of the phases of $\phi_{\lambda}$ not only for the problem of dephasing, but also for the dynamical phase transitions and questions related to them such as phase lapses and enhancement of observables.

PACS number: 03.65.Yz

## 1. Introduction

In order to study the properties of quantum systems, the interaction of the system with the environment has to be taken into account. Mostly, the environment is considered to be a measuring device that provides information on the system after interacting with it. The system is localized in space. However, there is always a natural environment into which the quantum system with discrete states is embedded. This environment consists of the continuum of extended scattering states into which the states of the system are embedded and can decay. The coupling matrix elements between the discrete states of the system and the scattering states of the continuum determine the lifetime of the states, which is usually finite due to this coupling.

The natural environment differs from that of a measuring apparatus because it exists at all times and is completely independent of any observer. It allows us to obtain information on the system without observation it all the time. An example is the radioactive dating used in geologic studies. That means under realistic conditions, a quantum system should be considered as an open system consisting of the system itself and the continuum of scattering states into which it is embedded.

An exact description of open quantum systems meets the mathematical problem to consider simultaneously the wavefunctions of discrete and scattering states. Both types of wavefunctions are completely different from one another. The discrete states k characterize the spectrum of the system and are orthonormalized according to the Kronecker delta $\delta_{kk'}$ while the scattering states are continuous in energy E and can be orthonormalized according to the Dirac delta function $\delta(E - E')$ . The wavefunctions of discrete and scattering states appear in a combined manner in most physical expressions characteristic of open quantum systems. Special mathematical considerations are necessary therefore in order to receive reliable results.

In the N-level Friedrichs model $[1, 2]$ , the total Hamiltonian H is defined by

$$
H = H _ {0} + \mu V,\tag{1}
$$

where $\mu$ is a real number and $H_{0}$ is the so-called free Hamiltonian

$$
H _ {0} = \sum_ {n = 1} ^ {N} \omega_ {n} | n \rangle \langle n | + \int_ {K _ {\omega}} \omega | \omega \rangle \langle \omega | \rho (\omega) d \omega .\tag{2}
$$

Here, $|n\rangle$ and $|\omega\rangle$ satisfy the orthonormality conditions $\langle n|n'\rangle = \delta_{nn'}$ , $\langle\omega|\omega'\rangle = \delta(\omega - \omega')/\rho(\omega)$ , and $\langle n|\omega\rangle = 0$ . $\rho(\omega)$ is a non-negative function interpreted as, e.g., an electromagnetic mode density. The sum runs over the finite number N of discrete basic states $|n\rangle$ and the integral is over the considered energy region with $K_{\omega} = \{\omega|\rho(\omega) \neq 0\}$ , like the energy band allowed by the electromagnetic mode.

The interaction Hamilton operator V describes the coupling between $|n\rangle$ and $|\omega\rangle$ ,

$$
V = \sum_ {n} \int_ {K _ {\omega}} (v _ {n} (\omega) | \omega \rangle \langle n | + v _ {n} ^ {*} (\omega) | n \rangle \langle \omega |) \rho (\omega) \mathrm{d} \omega ,\tag{3}
$$

where $v_{n}(\omega)$ is the interaction matrix element between $|n\rangle$ and $|\omega\rangle$ . In the Friedrichs model, the Schrödinger equation with the Hamiltonian H is directly solved. It is not easy to receive results that are of physical interest in a broad range of parameters. An example for the mathematical troubles appearing in direct solving the equations is the study of bound states in the continuum performed in [2].

In order to receive results for concrete systems, the partitioning technique is introduced in quantum chemistry more than 50 years ago $[3]$ . In this approach, the Hilbert space is splitted into subspaces by means of a set of projection operators. The subspaces have virtually nothing to do with each other. This method is applied, e.g., to the description of infrared spectra of different molecules $[4]$ . At about the same time, another projection operator formalism is introduced in nuclear physics $[5]$ . Here, the whole function space is splitted into two subspaces one of which contains the discrete states while the other one contains the continuum of scattering states. This formalism is applied successfully to the description of nuclear reactions by introducing statistical ensembles for the discrete states and their coupling coefficients to the continuum.

Today, there are many different methods to solve the problem in concrete cases numerically. A few examples are the following. The numerical and functional renormalization group approaches are applied recently to the problem of phase lapses $[6, 7]$ observed in the transmission through quantum dots. A powerful method is the Green function approach that is used in the tight-binding lattice method for the description of electronic transport in mesoscopic systems $[8–10]$ . In the framework of the tight-binding approximation, the electronic dynamics is studied in complex molecular networks $[11]$ . Other approaches are the Keldysh formalism that is used recently for the description of environmentally induced quantum-dynamical phase transitions $[12]$ and the bottleneck model of the transition state theory $[13]$ used in quantum chemistry. The method of complex scaling $[14]$ is applied to the description of different systems $[15]$ . The Feshbach projection operator (FPO) formalism $[5]$ is applied today, without using any statistical assumptions, also to the description of light nuclei, especially on the edge of stability $[16, 17]$ , as well as to laser-induced continuum structures in atoms $[18]$ and to the transmission through small quantum cavities $[10]$ .

Only very few methods are applied successfully to the description of an open quantum system under different conditions. Most interesting is the regime of overlapping resonances where the dynamical transition from one regime to another takes place (from low level density to high level density). Using the FPO formalism, this dynamical transition is traced by considering different nuclei, atoms in a laser field and quantum cavities with one or two attached leads $[16, 19]$ . In the transport through small quantum cavities with large openings $[20]$ , direct processes are supported when special states couple strongly to the leads, and can result in deterministic transport as signified by a striking system-specific suppression of shot noise. Experimental as well as theoretical studies performed by Pastawski et al $[12]$ show similar results: in the presence of an environment, the oscillatory dynamics of a quantum two-level system can undergo a quantum-dynamical phase transition to a non-oscillatory phase.

The different behavior of open quantum systems under different conditions is a challenging feature of open quantum systems that has to be addressed in the theoretical description. It is related to the coupling of the system to the environment and the feedback of this coupling onto the system. In the following, we will highlight the feedback between the quantum system and the environment into which it is embedded. Since most studies of this question are performed on the basis of the FPO formalism without using any statistical assumptions, we will center the discussion on this model and its exact solutions. The results obtained will be compared to those received with other methods.

By using the Feshbach projection operator formalism [5], the basic equations for the wavefunctions of the states $n$ and $\omega$ in (2) are solved separately such that the main problem of the Friedrichs model is avoided. The full function space is divided into two subspaces: the $Q$ subspace contains all wavefunctions that are localized inside the system and vanish outside of it while the wavefunctions of the $P$ subspace are extended up to infinity and vanish inside the system, see [16, 19]. The wavefunctions of the two subspaces can be obtained by standard methods: the $Q$ subspace is described by the Hermitian Hamilton operator $H_B$ that characterizes the closed (localized) system with discrete states, while the $P$ subspace is described by the Hermitian Hamilton operator $H_C$ that contains the continuum of scattering wavefunctions. Thus, $H_0 = H_B + H_C$ in (2). The coupling matrix elements are calculated according to (3) by using the eigenfunctions of $H_B$ instead of the basic wavefunctions $n$ that appear in (3). In other words, the closed system (defined by the Hamilton operator $H_B$ ) will be opened, in the FPO formalism, by coupling the wavefunctions of the $Q$ subspace to those of the $P$ subspace under the assumption $\mathcal{P} + \mathcal{Q} = 1$ where the operators $\mathcal{P}$ and $\mathcal{Q}$ project onto the $P$ and $Q$ subspaces, respectively. Due to this coupling, the discrete states of the closed system that lie above particle decay thresholds become resonance states of the open system. The states below decay thresholds receive, as a rule, some energy shift but remain discrete. The resonance states have, in general, a finite lifetime.

The FPO method was introduced by Feshbach $[5]$ about 50 years ago in order to describe particle-induced reactions on heavy nuclei in the region of high level density with excitation of narrow compound nucleus resonances. It is impossible (and also not meaningful) to calculate all the matrix elements in the two subspaces as well as those for the coupling matrix V. Instead, Feshbach used statistical methods in order to describe the narrow states of the Q subspace (compound nucleus states) and their coupling to the continuum. He treated exactly only the so-called direct (fast) reaction part. In this manner, it was possible to formulate a unified theory of nuclear reactions, i.e. a unified description of the fast direct reaction part and the much slower compound nucleus reaction part.

The situation is another one for particle-induced reactions on light nuclei due to their low level density. In this case, it is possible today to perform the calculations in the Q subspace with the same accuracy as the calculations for the corresponding closed system (Q = 1). Also the individual coupling matrix elements $v_{k}(\omega)$ can be calculated [17, 21]. These calculations represent therefore a unified description of structure and reaction phenomena [16, 17, 19]. They allow us to draw general conclusions on the behavior of open quantum systems under different conditions, i.e. by controlling them in a broad parameter range.

The known properties of the narrow resonance states (compound nuclear states) in heavy nuclei prevented, for many years, the application of the shell model (elaborated for light atoms) to the description of nuclear spectra. The difference between atoms and nuclei, as they were known more than 50 years ago, is the low level density of atoms and the high level density of the compound nucleus states to which the shell model is difficult to apply. Today we know that the difference is not between atoms and nuclei, but between systems with low and high level densities: heavy atoms with high level density and light nuclei with low level density do exist and show, in each case, the features characteristic of systems with the corresponding level density. The recent experimental results on phase lapses $[22]$ in the transmission through quantum dots seem to be a hint at a similar situation in mesoscopic systems. The so-called mesoscopic features appear only at low level density while they are washed out at high level density.

The power of the FPO formalism is its transparency. Most interesting is the direct appearance of the non-Hermitian Hamiltonian $H_{eff}$ that describes the localized part of the system (Q subspace) including its coupling to the extended environment (P subspace). It is possible therefore to relate the obtained numerical results directly to the specific characteristics of a non-Hermitian operator. Vice versa, the role of the characteristics of $H_{eff}$ can be studied by controlling the system by means of external parameters. Using these results the parameter range to obtain desired effects, by means of manipulation, can be extended. An example is the high-order harmonic generation in a driven two-level atom. By using scaling laws, one may extend the parameter range to, for instance, that characteristic of solid-state systems in strong fields [23].

The symmetric non-Hermitian operator $H_{eff}$ has a singularity when two (or more) of its eigenvalues coincide. In the mathematical literature, these points are called exceptional points [24]. Here, also the two corresponding eigenfunctions collapse. In the physical literature, the role of these singular points is another one. Although they determine the dynamics of the system, they are not distinguished from the neighboring points by dramatic effects seen in observable quantities. Correspondingly, they are called only seldom ‘exceptional points’ in the physical literature. They are called often (true) crossing points in order to express their relation to the well-known avoided level crossing phenomenon, or branch points in order to express the different physical branches (such as level repulsion and width bifurcation) that originate at these points, or double poles of the S matrix when the scattering problem is considered. Controlling the physical system by an external parameter, a dramatic reduction of the number of eigenstates does not occur at the singular point (in contrast to the assumptions for the exceptional point). Instead, a so-called associated eigenvector appears due to the Jordan chain relations at the singular point as well as a phase jump [25]. Accordingly, the observable quantities change smoothly when passing the singularity.

In the present paper, the description of open quantum systems by using the FPO formalism will be considered in detail. The main body of the paper consists of three sections. In the first part, the non-Hermitian Hamilton operator $H_{eff}$ characterizing an open quantum system is given. Formally, it contains a first-order term and a second-order one. The eigenvalues $z_{\lambda}$ are complex and provide not only the energies of the resonance states but also their widths (lifetimes). The eigenfunctions $\phi_{\lambda}$ are biorthogonal. The crossing points of eigenvalue trajectories are singular points at which the two corresponding eigenfunctions of $H_{eff}$ are linearly dependent, and associated algebraic eigenvectors appear due to the Jordan chain relations. In approaching the crossing points, the phases of the eigenfunctions of $H_{eff}$ lose their rigidity. The phase rigidity varies between $r_{\lambda} = 1$ in the regime of well isolated resonances and 0 at the crossing point. The variation of the phase rigidity as a function of external parameters is the main difference between the physics based on non-Hermitian and that based on Hermitian Hamilton operators. By encircling the crossing point of eigenvalue trajectories, its topological structure can be studied. The phase related to the encircling differs from the Berry phase by a factor 2. This difference is related to the fact that the Berry phase around the diabolic point is related to the standard first-order interaction term of the Hamiltonian. The phase around the crossing point in the complex plane is however related to the second-order interaction term of $H_{eff}$ that is dominant at the crossing point. Finally, an expression for the time evolution operator is given in this first part of the paper.

The second part of the present paper is devoted to the question whether and to which extent the properties described by $H_{eff}$ survive when the full problem in the total function space with the Hermitian Hamilton operator H is solved. First the solution $\Psi_{c}^{E}$ of the whole problem $(H - E)\Psi_{c}^{E} = 0$ is derived by using the FPO formalism and, by using $\Psi_{c}^{E}$ , an expression for the scattering matrix is obtained. Inside the localized part of the system, the solutions $\Psi_{c}^{E}$ can be represented in the set $\{\phi_{\lambda}\}$ of eigenfunctions of $H_{eff}$ . Due to this representation the solutions $\Psi_{c}^{E}$ , and also the resonance part of the S matrix, contain direct hints at the non-Hermitian Hamiltonian $H_{eff}$ . An expression for the phase rigidity $\rho$ of $\Psi_{c}^{E}$ is derived that varies between

1 (for well isolated resonances) and 0 (in the regime of overlapping resonances). The phase rigidity $\rho$ is directly related to the dynamics of open quantum systems which, therefore, can be manipulated by external parameters. The influence of neighboring resonance states onto one another, expressed quantitatively by $\rho$ , may be interpreted as the internal impurity of the system. This impurity depends, above all, on the degree of overlapping of the resonance states and does not vanish at zero temperature. In the regime of overlapping resonance states, spectroscopic reordering processes (dynamical phase transitions) occur due to width bifurcation and resonance trapping. Finally, the brachistochrone problem is discussed.

In the third part, the internal impurity of open quantum systems in the regime of overlapping resonances is considered by means of concrete examples. The results obtained are compared with those obtained in the framework of other approaches used in some special cases, and with existing experimental data. First, the decay rates at high level density are derived. Since the decay does not occur exponentially when the individual resonance states overlap with neighboring states, the decay rate depends on time, generally. Atomic spectra can be manipulated by means of lasers in a broad parameter range due to the existence of branch points (double poles of the S matrix). The transmission through small quantum dots is enhanced just below the threshold for opening a new channel where the phase rigidity $\rho$ is reduced. The anticorrelation between transmission and phase rigidity $\rho$ can be seen in all considered cases. Further, it is looked at the interesting phenomenon of bound states in the continuum. It is related to the avoided level crossing phenomenon and causes a stabilization of the system at certain parameter values. The phase lapses observed experimentally are discussed from the point of view of resonance trapping. The many experimental data on dephasing can surely be related to the non-rigid phases of the eigenfunctions $\phi_{\lambda}$ of the non-Hermitian Hamilton operator $H_{eff}$ . Finally, some remarks on open quantum systems with a non-symmetric non-Hermitian Hamiltonian are given.

The results are summarized in section 5. The unique strength of the FPO formalism consists of the following. First, the many-body problem can be solved in the same manner as for discrete states, including the computation of the coupling matrix elements between discrete and scattering states. Second, the direct relation of observables to the non-Hermitian Hamilton operator $H_{eff}$ allows, on the one hand, their interpretation from the point of view of the specific properties of non-Hermitian operators. On the other hand, knowing the specific features of non-Hermitian operators, especially the position of their singularities, it becomes possible to manipulate open quantum systems with the aim to receive a system with desired properties.

## 2. Characteristics of open quantum systems

## 2.1. The non-Hermitian Hamilton operator

Characteristic of an open quantum system is the interaction of its states via a common environment. The environment consists, by nature, of a continuum of scattering wavefunctions: the system itself is localized in space while the environment is extended up to infinity. The Hamilton operator of an open quantum system consists therefore of a first-order and a second-order interaction term,

$$
H _ {\mathrm{eff}} = H _ {B} + \sum_ {C} V _ {B C} \frac {1}{E ^ {+} - H _ {C}} V _ {C B}.\tag{4}
$$

Here, $H_{B}$ is the Hamilton operator describing the closed (isolated) system with discrete states,

$$
\left(H _ {B} - E _ {i} ^ {B}\right) \Phi_ {i} ^ {B} = 0,\tag{5}
$$

$G_{P}^{(+)} \equiv (E^{+} - H_{C})^{-1}$ is the Green function in the continuum and $V_{BC}, V_{CB}$ describes the coupling of the closed system to the continuum. Further, $H_{C}$ is the Hamiltonian describing the environment of decay channels. The wavefunctions $\xi_{E}^{c}$ of the scattering states (channel wavefunctions) follow from

$$
(H _ {C} - E) \xi_ {E} ^ {c} = 0.\tag{6}
$$

The solutions to (5) are orthonormalized according to the Kronecker delta $\delta_{ik}$ and those to (6) according to the Dirac $\delta$ function $\delta(E - E')$ in each channel $c$ .

The Hamiltonian (4) expresses formally the embedding of the system with N discrete states into K continua of scattering states. It appears in the description of open quantum systems as will be shown in section 3, where expression (4) will be derived. In the present section, we are interested in the mathematical properties of Hamilton operators of the type (4) as well as in their eigenvalues and eigenfunctions. The knowledge of these properties allows us to receive an understanding for many unexpected and often counterintuitive physical properties of open quantum systems.

The Hamilton operator $H_{eff}$ is symmetric and non-Hermitian with

$$
\operatorname{Re} \left\{\left\langle \Phi_ {i} ^ {B} \mid H _ {\text { eff }} \mid \Phi_ {j} ^ {B} \right\rangle \right\} = \left\langle \Phi_ {i} ^ {B} \mid H _ {B} \mid \Phi_ {j} ^ {B} \right\rangle + \frac {1}{2 \pi} \sum_ {\mathrm{c}} \mathcal {P} \int_ {\epsilon_ {\mathrm{c}}} ^ {\infty} \mathrm{d} E ^ {\prime} \frac {\hat {\gamma} _ {i} ^ {c} \hat {\gamma} _ {j} ^ {c}}{E - E ^ {\prime}}
$$

$$
\equiv \left<   \Phi_ {i} ^ {B} \left. \right| H _ {B} \left| \right. \Phi_ {j} ^ {B} \right> + \mathrm{Re} (\hat {W} _ {i j}),\tag{7}
$$

$$
\mathrm{Im} \bigl \{\bigl \langle \Phi_ {i} ^ {B} \big | H _ {\mathrm{eff}} \big | \Phi_ {j} ^ {B} \bigr \rangle \bigr \} = - \frac {1}{2} \sum_ {\mathrm{c}} \hat {\gamma} _ {i} ^ {c} \hat {\gamma} _ {j} ^ {c} \equiv \mathrm{Im} (\hat {W} _ {i j}),\tag{8}
$$

where

$$
\hat {\gamma} _ {i} ^ {c} = \sqrt {2 \pi} \bigl \langle \xi_ {E} ^ {c} \bigr | V \bigr | \Phi_ {i} ^ {B} \bigr \rangle = \sqrt {2 \pi} \bigl \langle \Phi_ {i} ^ {B} \bigr | V ^ {\dagger} \bigr | \xi_ {E} ^ {c} \bigr \rangle\tag{9}
$$

are the coupling matrix elements of the discrete states i (following from (5)) to the decay channels c (described by (6)). One has $i = 1, \ldots, N$ and $c = 1, \ldots, K$ , where N is the number of discrete states and K is the number of different (elastic and inelastic) decay channels each of which consists of scattering wavefunctions normalized according to the Dirac $\delta$ function. Due to the coupling between the two subspaces, the discrete states of the system become resonance states which have, mostly, a finite lifetime.

The eigenvalues and eigenfunctions of $H_{B}$ contain the interaction u of the discrete states which is contained in $H_{B}$ . This interaction is of standard type in closed (isolated) systems and may be called therefore the internal interaction. It contains, generally, also the contributions from the many-body forces. The eigenvalues and eigenfunctions of $H_{eff}$ contain additionally the interaction v of the resonance states via the common continuum (v is used here instead of the concrete matrix elements of the second term of $H_{eff}$ ). This part of the interaction is, formally, of second order and may be called the external interaction. It plays an important role in the regime of overlapping resonance states.

The standard approximation in quantum mechanics is the assumption that the Hamilton operator $H_{eff}$ is Hermitian. That means, effective forces are introduced by defining an effective Hermitian Hamilton operator $H_{eff}^{h}$ . The matrix elements of $H_{eff}^{h}$ correspond to (7). However in the standard calculations for concrete systems, $H_{eff}^{h}$ is not obtained from (7), but determined phenomenologically. The difference between the effective forces and the original ones calculated from $H_{B}$ may be rather large. Neglecting the non-Hermitian term of $H_{eff}$ corresponds to the assumption that $\operatorname{Im}\left\{\left\langle\Phi_{i}^{B}\right|H_{\mathrm{eff}}\left|\Phi_{j}^{B}\right\rangle\right\}=0$ . This assumption can be justified only for low-lying discrete states that are not directly coupled to the scattering wavefunctions of the environment.

In the present paper, the non-Hermiticity of $H_{eff}$ is taken into account and the complex matrix elements $\left\langle\Phi_{i}^{B}\mid H_{eff}\mid\Phi_{j}^{B}\right\rangle$ are calculated in a straightforward manner. In this way, the nontrivial topological structure of the continuum is taken into account. It causes nonlinear effects in the regime of overlapping resonance states. Under certain conditions, the second-order term of $H_{eff}$ may become the leading term (see section 2.3).

## 2.2. Eigenvalues and eigenfunctions of $H_{eff}$

The operator $H_{eff}$ , equation (4), is symmetric. Its eigenvalues $z_{\lambda}$ and eigenfunctions $\phi_{\lambda}$ are complex,

$$
(H _ {\mathrm{eff}} - z _ {\lambda}) \phi_ {\lambda} = 0.\tag{10}
$$

The eigenvalues provide not only the energies $E_{\lambda}$ of the resonance states but also their widths $\Gamma_{\lambda}$ ,

$$
z _ {\lambda} = E _ {\lambda} - \frac {\mathrm{i}}{2} \Gamma_ {\lambda}.\tag{11}
$$

The eigenvalues of states below particle decay thresholds are real (according to $\Gamma_{\lambda}=0$ ), while those of states above thresholds are complex, generally (with $\Gamma_{\lambda}\neq0$ ). Since the operator $H_{eff}$ depends explicitly on energy E, so do its eigenvalues $z_{\lambda}$ and eigenfunctions $\phi_{\lambda}$ . Far from thresholds, the energy dependence is weak, as a rule, in an energy interval (around $E_{\lambda}$ ) of the order of magnitude of the width of the resonance state.

The eigenfunctions $\phi_{\lambda}$ of $H_{eff}$ are biorthogonal, and $\langle\psi_{\lambda}|=\langle\phi_{\lambda}^{*}|$ because of the symmetry of $H_{eff}$ . The normalization condition $\langle\phi_{\lambda}^{*}|\phi_{\lambda}\rangle=(\phi_{\lambda})^{2}$ fixes only two of the four free parameters [25]. This freedom can be used in order to provide a smooth transition from an open quantum system (with, in general, nonvanishing decay widths $\Gamma_{\lambda}$ of its states and biorthogonal wavefunctions) to the corresponding closed one (with $\Gamma_{\lambda}\to0$ and real wavefunctions that are normalized in the standard manner): $\langle\phi_{\lambda}^{*}|\phi_{\lambda}\rangle\to\langle\phi_{\lambda}|\phi_{\lambda}\rangle=1$ if $V_{BC}, V_{CB}\to0$ in (4). That means, the orthonormality conditions can be chosen as

$$
\langle \phi_ {\lambda} ^ {*} | \phi_ {\lambda^ {\prime}} \rangle = \delta_ {\lambda , \lambda^ {\prime}}\tag{12}
$$

with the consequence that

$$
\langle \phi_ {\lambda} | \phi_ {\lambda} \rangle = \operatorname{Re} (\langle \phi_ {\lambda} | \phi_ {\lambda} \rangle)
$$

$$
A _ {\lambda} \equiv \langle \phi_ {\lambda} | \phi_ {\lambda} \rangle \geqslant 1\tag{13}
$$

$$
\langle \phi_ {\lambda} | \phi_ {\lambda^ {\prime} \neq \lambda} \rangle = \mathrm{i} \operatorname{Im} (\langle \phi_ {\lambda} | \phi_ {\lambda^ {\prime} \neq \lambda} \rangle) = - \langle \phi_ {\lambda^ {\prime} \neq \lambda} | \phi_ {\lambda} \rangle
$$

$$
\left| B _ {\lambda} ^ {\lambda^ {\prime}} \right| \equiv | \langle \phi_ {\lambda} | \phi_ {\lambda^ {\prime} \neq \lambda} | \geqslant 0.\tag{14}
$$

In the regime of overlapping resonances, one has $A_{\lambda} > 1$ and $\left|B_{\lambda}^{\lambda'}\right| \neq 0$ due to the second term of (4), i.e. due to the interaction of the resonance states via the continuum. The strength of this interaction is given by the non-diagonal matrix elements of (4). The relation $\left|B_{\lambda}^{\lambda'}\right| \neq 0$ means that the two eigenstates $\lambda$ and $\lambda'$ are not orthogonal to one another.

Using the eigenfunctions of $H_{eff}$ , the coupling matrix elements of the resonance states to the decay channels read

$$
\gamma_ {\lambda} ^ {c} = \sqrt {2 \pi} \bigl \langle \xi_ {E} ^ {c} \bigr | V | \phi_ {\lambda} \rangle = \sqrt {2 \pi} \langle \phi_ {\lambda} ^ {*} | V ^ {\dagger} \bigr | \xi_ {E} ^ {c} \bigr \rangle .\tag{15}
$$

In contrast to (9), the coupling matrix elements (15) contain the feedback from the continuum of scattering wavefunctions on the system. One has [26]

$$
\Gamma_ {\lambda} = \frac {1}{A _ {\lambda}} \sum_ {c} \left| \gamma_ {\lambda} ^ {c} \right| ^ {2} \leqslant \sum_ {c} \left| \gamma_ {\lambda} ^ {c} \right| ^ {2}.\tag{16}
$$

The relation $\Gamma_{\lambda} = \sum_{\mathrm{c}}\left|\gamma_{\lambda}^{c}\right|^{2}$ can be justified only for isolated resonance states $\lambda$ for which $A_{\lambda}\approx 1$ .

## 2.3. Crossing points in the complex plane (exceptional points)

In contrast to the trajectories of the real eigenvalues $E_{i}^{B}(X)$ of a Hermitian Hamilton operator (where X is a certain parameter), those of the complex eigenvalues $z_{\lambda}(X)$ of a non-Hermitian operator may cross. In order to illustrate this statement, let us consider the Hamilton operator

$$
\hat {H} = \left( \begin{array}{c c} \epsilon_ {1} & \omega \\ \omega & \epsilon_ {2} \end{array} \right), \qquad \hat {H} = \hat {H} ^ {T},\tag{17}
$$

where $\epsilon_{\lambda}$ are complex and stand for the energies and widths of two isolated resonance states, $\lambda = 1, 2$ . The interaction $\omega$ between the two states is complex, generally. Both, $\epsilon_{\lambda}$ and $\omega$ , may depend on a parameter X by means of which the system properties can be controlled. The restriction to the $2 \times 2$ Hamilton operator $\hat{H}$ in (17) is justified by the fact that we are interested here in the case when two eigenvalue trajectories $z_{\lambda}(X)$ and $z_{\lambda'}(X)$ cross (or nearly cross) at a certain critical value $X = X^{cr}$ . At $X^{cr}$ , the distance to all the other eigenvalue trajectories is relatively large such that they do (almost) not influence the crossing scenario of the two states $\lambda$ and $\lambda'$ .

The eigenvalues of (17) are

$$
E _ {\pm} = E _ {0} \pm \frac {1}{2} \sqrt {(\epsilon_ {1} - \epsilon_ {2}) ^ {2} + 4 \omega^ {2}} = E _ {0} \pm \omega \sqrt {Z ^ {2} + 1}\tag{18}
$$

with

$$
E _ {0} = \frac {\epsilon_ {1} + \epsilon_ {2}}{2}, \quad Z = \frac {\epsilon_ {1} - \epsilon_ {2}}{2 \omega}.\tag{19}
$$

The two trajectories $E_{+}(X)$ and $E_{-}(X)$ cross when $Z^2 \equiv Z_c^2 = -1$ , i.e. when $(\epsilon_1 - \epsilon_2) / (2\omega) = \pm i$ .

The eigenfunctions of $\hat{H}$ at the branch (crossing) point are linearly dependent

$$
\phi_ {+} ^ {\mathrm{cr}} \rightarrow \pm \mathrm{i} \phi_ {-} ^ {\mathrm{cr}}, \quad \phi_ {-} ^ {\mathrm{cr}} \rightarrow \mp \mathrm{i} \phi_ {+} ^ {\mathrm{cr}}.\tag{20}
$$

This relation follows from analytical studies [16, 25, 27–29] as well as from numerical studies on a realistic system (laser-induced continuum structures in atoms [30]). At the crossing point, a phase jump of the wavefunction by $\pi/4$ appears. Here, $A_{\lambda} \to \infty$ , $\left|B_{\lambda}^{\lambda'}\right| \to \infty$ . Furthermore, in spite of the fact that the two wavefunctions $\phi_{+}^{\mathrm{cr}}$ and $\phi_{-}^{\mathrm{cr}}$ at the crossing point are linearly dependent on one another, there are two different wavefunctions also at this point. The right and left eigenvectors are supplemented, at the crossing point, by corresponding associated vectors (algebraic eigenvectors) $\phi_{\pm}^{\mathrm{cra}}$ defined by the Jordan chain relations [25],

$$
[ \hat {H} (Z _ {\mathrm{c}}) - E _ {0} ] \phi_ {\pm} ^ {\mathrm{cr}} = 0, \qquad [ \hat {H} (Z _ {\mathrm{c}}) - E _ {0} ] \phi_ {\pm} ^ {\mathrm{cra}} = \phi_ {\pm} ^ {\mathrm{cr}}.\tag{21}
$$

It should be remarked that (10) with the non-Hermitian Hamiltonian (17) can be rewritten into a Schrödinger equation with the Hermitian Hamiltonian $H_{B}$ and a nonlinear source term [29]. The nonlinearities arise from the quantities $A_{\lambda} = |\phi_{\lambda}|^{2}$ defined in (13). They are large when the resonance states strongly overlap. This equation gives, on the one hand, relation (20) between the eigenfunctions at the crossing point [29]. On the other hand, the equation becomes linear in the limit $A_{\lambda} \rightarrow 1$ , i.e. in the standard Hermitian quantum mechanics. Thus, the two limiting cases are correctly described. That means, the non-Hermiticity of the Hamiltonian describing the open quantum system can be simulated partly by a nonlinear Schrödinger equation with Hermitian Hamiltonian.

In the mathematical literature, the crossing points are called mostly exceptional points [24]. In these papers, the control of the open quantum system in approaching the crossing point and the associated vectors (21) are usually not considered. Hence, the relations between the two wavefunctions at the exceptional point may (and do) differ from those given in (20), e.g. [31, 32]. Furthermore, the phase jump of the wavefunction need not to be considered when only the exceptional point is considered. As a consequence, the state at the exceptional point may be interpreted as a chiral state [33] what is not in contradiction to (20). However controlling the system in approaching the crossing point, the phase jump destroys this simple interpretation.

In the physical literature, different notations are used for the crossing points, e.g. branch points $[29]$ , hidden crossings $[34]$ , double poles of the S matrix $[30, 35, 36]$ . The different notations are a hint at the great role they play in physical systems. Indeed, they are responsible for the avoided level crossings appearing in their vicinity and, consequently, for the dynamics of open quantum systems.

An essential difference between Hermitian and non-Hermitian quantum mechanics is that all quantities in $\hat{H}$ , equation (17), are real in Hermitian physics while they are complex in non-Hermitian physics. As a consequence, $(\epsilon_1 - \epsilon_2)^2 + 4\omega^2$ is always larger than zero for Hermitian operators (with nonvanishing $\omega$ ) and $\omega \sqrt{Z^2 + 1}$ is real, leading to level repulsion. The corresponding point of the avoided level crossing is called usually diabolic point.

For non-Hermitian operators, $\omega \sqrt{Z^2 + 1}$ may take any value, including a purely real value, a purely imaginary one and zero. Hence, the avoided level crossing phenomenon in the complex plane is much richer than that in the real plane: it appears not only in $\mathrm{Re}(z_{\lambda})$ (if $\omega \sqrt{Z^2 + 1}$ is real) but also in $\mathrm{Im}(z_{\lambda})$ (if $\omega \sqrt{Z^2 + 1}$ is imaginary). Correspondingly, we have level repulsion in the first case and width bifurcation in the second case, i.e. different physical situations in approaching the crossing (branch) point under different conditions.

In the case of narrow resonance states and (almost) real $\omega$ , the levels repel in energy. This behavior is similar to that what is well known for discrete states: level repulsion characterizes the tendency of the system to avoid clustering. When, however, the imaginary part of $\omega$ is sufficiently large, width bifurcation appears. In this regime, the levels attract each other in energy, what means there is a tendency to form level clusters (when the interaction via the continuum is large). Controlling the system by a second parameter, the crossing (branch) point in between the two scenarios with level repulsion and width bifurcation can always be met.

As a numerical example for the avoided level crossing phenomenon, we refer here to the calculations performed for a double quantum dot (billiard) [27, 28]. Two eigenvalues and the corresponding eigenfunctions of $H_{\mathrm{eff}}$ are shown in figures 1 and 2 as a function of the coupling strength $v$ between the double quantum dot and the two attached leads (corresponding to $K = 2$ channels). The eigenfunctions are given in the representation

$$
| 1 \rangle = \left( \begin{array}{c} a \\ 0 \\ b \end{array} \right), \qquad | 3 \rangle = \left( \begin{array}{c} b \\ 0 \\ - a \end{array} \right).\tag{22}
$$

The eigenvalue pictures show the characteristic features seen in many calculations. When the length L of the wire that connects the two single dots (billiards) is equal to the critical value $L_{c}$ , the two eigenvalues 1 and 2 cross at v = 1. In the two figures, $L \neq L_{c}$ and the two eigenvalues avoid crossing in the complex plane. In figure 1, $L > L_{c}$ . Here, the positions in energy of the resonance states avoid crossing, while their widths cross. In figure 2 with $L < L_{c}$ , we see the opposite behavior: the positions cross while the widths do not cross at v = 1. In both cases, the widths bifurcate at v > 1. At the critical point, the norm of the eigenfunctions diverges and their phases jump by $\pm\pi/4$ .

Similar results for the eigenvalues and eigenfunctions of $H_{\mathrm{eff}}$ are obtained numerically by using the Hamiltonian (17), see [29]. They agree with the results of analytical studies [25].

(a)
![](images/c4a831a5cf6645c688cc33aa2d956711d248fe8cdfa8bd5bf338f1253c1c4c9f.jpg)
(c)

(b)
![](images/96fe34c6ffc17729712666ab68f915fbb01e6b0f3b793c37ec5f3540f928eda0.jpg)

![](images/bf792dcd128e023d1330bc3e5ae072ed00ca3a5ae28b75f84db2102066ff714d.jpg)

![](images/94063cd8eadc074facb5f5ff3838efb45a1c225d1fa637a1eaddc51628e9f446.jpg)
Figure 1. The evolution of the eigenvalues $z_{1}$ (solid lines) and $z_{3}$ (dashed lines) ( $a$ ) and ( $b$ ) and of the components $a = |a| \mathrm{e}^{\mathrm{i}\alpha}$ (dashed lines) and $b = |b| \mathrm{e}^{\mathrm{i}\beta}$ (solid lines) of the eigenfunctions 1 and 3 (c) and (d) of the effective Hamiltonian $H_{\mathrm{eff}}$ for a double quantum dot (billiard) as a function of the coupling strength $v$ to two (identical) attached leads. The eigenvalue trajectories cross at $L_{\mathrm{c}} = 1.4645$ under the conditions chosen in the calculation. In the figure $L = L_{\mathrm{c}} + 0.01$ , $\operatorname{Re}(z_1)$ and $\operatorname{Re}(z_3)$ avoid crossing while $\operatorname{Im}(z_1)$ and $\operatorname{Im}(z_3)$ cross. At the critical value of $v$ , $|a| \gg 1$ , $|b| \gg 1$ and the phases jump by $\pi/4$ . Figure taken from [27].

There is strong evidence therefore to state that the results shown in figures 1 and 2 are generic. They can be seen not only as a function of the coupling strength v, but in any parameter that influences the value of the non-diagonal matrix elements of (17).

## 2.4. Phase rigidity of the eigenfunctions of $H_{\mathrm{eff}}$

The normalization condition (12) entails that the phases of the eigenfunctions of $H_{eff}$ in the overlapping regime are not rigid: the normalization condition $\langle\phi_{\lambda}^{*}|\phi_{\lambda}\rangle=1$ is fulfilled, in this regime, only when $\operatorname{Im}\langle\phi_{\lambda}^{*}|\phi_{\lambda}\rangle\propto\operatorname{Re}\phi_{\lambda}\cdot\operatorname{Im}\phi_{\lambda}=0$ , i.e. by rotating the wavefunction through a certain angle $\beta_{\lambda}$ . The phase rigidity defined by

$$
r _ {\lambda} = \frac {\left\langle \phi_ {\lambda} ^ {*} \mid \phi_ {\lambda} \right\rangle}{\left\langle \phi_ {\lambda} \mid \phi_ {\lambda} \right\rangle} = \frac {1}{(\operatorname{Re} \phi_ {\lambda}) ^ {2} + (\operatorname{Im} \phi_ {\lambda}) ^ {2}} = \frac {1}{A _ {\lambda}}\tag{23}
$$

is a useful measure [37, 38] for the rotation angle $\beta_{\lambda}$ . When the resonance states are distant from one another, it is $r_{\lambda} \approx 1$ due to $\langle \phi_{\lambda}|\phi_{\lambda}\rangle \approx \langle \phi_{\lambda}^{*}|\phi_{\lambda}\rangle$ . In approaching a branch point in the complex energy plane [27, 29], we have $A_{\lambda} \to \infty$ . Therefore $1 \geqslant r_{\lambda} \geqslant 0$ . Thus, the phase rigidity $r_{\lambda}$ of the eigenfunctions of $H_{\mathrm{eff}}$ is reduced in the regime of overlapping resonance states, while it is well preserved for isolated ones at low level density.

(a)
![](images/9eced3c6b15172478b35d84f935551a1c7c09899041dc45030f449bc9ee60f78.jpg)

(b)
![](images/4ea338d848a3674d498e7cce13d743150afac1d94460d8176b774c0397d5f47d.jpg)

(c)
![](images/583d4c87c6aadb6e46f196ff8be24732445e51b4790adb274e473429fa9dac12.jpg)

(d)
![](images/d4ea7e86e39959bd0c6c54844ff9c4437361a37718cf7c296dacddf5f61b5738.jpg)
Figure 2. The same as figure 1 but $L = L_{\mathrm{c}} - 0.01$ . In this case, $\operatorname{Re}(z_1)$ and $\operatorname{Re}(z_3)$ cross while $\operatorname{Im}(z_1)$ and $\operatorname{Im}(z_3)$ do not cross. At the critical value of $v$ , it is $|a| \gg 1$ , $|b| \gg 1$ and the phases jump by $-\pi / 4$ . Figure taken from [27].

The phase rigidity $r_{\lambda}$ is a measure for the degree of alignment of one of the overlapping resonance states with one of the scattering states $\xi_{E}^{c}$ of the environment. This alignment takes place at the cost of at least one other state that decouples, to a certain extent, from the environment. The alignment is enabled by the interaction of the states via the common continuum, i.e. by the second (non-Hermitian) term of the Hamiltonian (4). According to this, it occurs only in the regime of overlapping resonance states. Another notation for alignment of states is width bifurcation or resonance trapping.

The freedom to align some states of the system to the scattering states of the environment does not exist in a closed quantum system described by a Hermitian Hamiltonian with discrete states. Here $A_{\lambda} = r_{\lambda} = 1$ and $B_{\lambda}^{\lambda'} = B_{\lambda'}^{\lambda} = 0$ due to the normalization condition $\langle\phi_{\lambda}|\phi_{\lambda'}\rangle = \delta_{\lambda\lambda'}$ . Hence, the two wavefunctions $\phi_{\lambda}$ and $\phi_{\lambda'\neq\lambda}$ of a closed system are always orthogonal to one another, in contrast to those of an open quantum system.

Formally, the phenomenon of width bifurcation (alignment of states to the environment) is analogous to level repulsion in energy, i.e. to the repulsion of the real parts of two eigenvalues of $H_{\mathrm{eff}}$ at a critical value of the control parameter. According to the eigenvalue equation (18), width bifurcation is caused by $\mathrm{Im}(\omega \sqrt{Z^2 + 1})$ while level repulsion is caused by $\mathrm{Re}(\omega \sqrt{Z^2 + 1})$ . In contrast to level repulsion in energy, width bifurcation occurs however by means of the rotation of the eigenfunctions of $H_{\mathrm{eff}}$ under the influence of the scattering wavefunctions of the environment. In other words, level repulsion is caused by the internal interaction contained in $H_B$ (and effectively also by the principal value integral of the external interaction). Width bifurcation, however, is caused by the external interaction of the states via the continuum in which the feedback of the coupling to the continuum is contained.

## 2.5. Topological structure of the crossing (exceptional) points

The topological structure of the branch points in the complex plane (also called crossing or exceptional points) is nontrivial. Theoretically, it is studied in different papers, e.g. [25, 27, 31, 39–43]. In these papers, the mathematical properties are emphasized, and the singular points are called mostly exceptional points. In all these papers, the geometric phase of the eigenvectors of non-Hermitian complex operators has been considered for paths in parameter space that encircle the branch point. In the case of the symmetric complex $2 \times 2$ matrix Hamiltonian (17), the geometric phase is topological [44].

In the papers [25, 31, 41], the studies are performed on the basis of the symmetrical toy model (17)-(19), see also [45]. In [27, 40], a realistic open system with symmetrical $H_{\mathrm{eff}}$ and with two and three states, respectively, is considered. As a result, a cycle around the branch (exceptional) point has to be performed four times in order to produce one full $2\pi$ circle in the geometric phase. That means, the branch point has to be encircled two times more than a diabolic point in order to restore the wavefunction. For illustration, the results for surrounding a branch point can be represented in the following manner. According to (20)

$$
\begin{array}{l l} 1. & \text {Cycle:} \qquad E _ {\pm} \to E _ {\mp} \qquad \phi_ {\pm} \to \pm \mathrm{i} \phi_ {\mp} \\ 2. & \text {Cycle:} \qquad E _ {\mp} \to E _ {\pm} \qquad \pm \mathrm{i} \phi_ {\mp} \to - \phi_ {\pm} \\ 3. & \text {Cycle:} \qquad E _ {\pm} \to E _ {\mp} \qquad - \phi_ {\pm} \to \mp \mathrm{i} \phi_ {\mp} \\ 4. & \text {Cycle:} \qquad E _ {\mp} \to E _ {\pm} \qquad \mp \mathrm{i} \phi_ {\mp} \to \phi_ {\pm}. \end{array}\tag{24}
$$

The difference between the geometric phases of a diabolic and a branch (exceptional) point does not have any relation to the fact that the branch point is a true crossing point of eigenvalue trajectories in contrast to the diabolic point, since the singular point is encircled at a certain distance, in any case. The difference between the geometric phases in the two cases consists rather in the following. In the case of the diabolic point, the Hamiltonian $H_{B}$ of the system contains only the internal interaction u. In the case of the branch (exceptional) point, however, the Hamiltonian is $H_{eff}$ which contains additionally a second-order term arising from the coupling to the continuum (external interaction v, see (4)). At the branch point, this second-order term becomes the leading term, see section 2.3. Hence, the difference between the geometric phases in the two cases with $H_{B}$ and $H_{eff}$ , respectively, illustrates once more the importance of the interaction via the continuum when the quantum system is open.

The topological structure of the diabolic and branch points has been studied experimentally on microwave cavities. The results show the Berry phase in the case of the diabolic point $[46]$ and the four-fold winding $(24)$ in the case of the branch point $[32]$ , in full agreement with the theoretical studies.

In another experiment [33], the phase difference between the two eigenvectors in approaching the branch point has been studied. As a result, the phase difference between the two modes changes from $\pi$ at large distance between them to $\pi /2$ in approaching the branch (exceptional) point. This result has been explained by the authors [33] as observation of a chiral state by means of the assumptions that there is only one state at the exceptional point (and not two as follows from the Jordan chain relations (21)), that this state is a chiral one (in spite of the phase jump occurring at this point [25]) and that a single point in the continuum can be identified experimentally.

The experimental results [33] can be explained by means of the phase rigidity $r_{\lambda}$ of the complex eigenfunctions $\phi_{\lambda}$ of the non-Hermitian Hamilton operator $H_{\mathrm{eff}}$ [37]. The phase rigidity drops smoothly from its maximum value $r_{\pm} = 1$ far from the branch point (with the phase difference $\pi$ (or $2\pi$ ) between the wavefunctions of isolated resonance states) to its minimum value $r_{\pm} = 0$ at the branch point (with the phase difference $\pm \pi / 2$ according to (20)), see section 2.4. This interpretation explains, in a natural manner, the experimentally observed smooth reduction of the phase difference in a comparably large parameter range. Also the phase jump occurring at the branch point (section 2.3) is not in disagreement with the experimental data.

In this manner, the experimental results $[32]$ prove the topological structure of the branch points. Furthermore, the results $[33]$ can be considered to demonstrate the (parametric) dynamics of open quantum systems which is generated by the branch points. At and in the neighborhood of these points, the phase rigidity $r_{\lambda}$ of the wavefunctions of the system is reduced. In other words, a neighboring resonance state influences the considered one in the regime of overlapping due to the coupling of both states to the common continuum.

## 2.6. The time evolution operator

An old problem of standard quantum mechanics based on a Hermitian Hamilton operator is the absence of a time operator in the theory. The system is described, usually, by $H_{B}$ or $H_{eff}^{h}$ as defined in section 2.1. This theory is able to provide the energies $E_{i}^{B}$ of the states to a high degree of accuracy since the effective forces are usually taken into account in a phenomenological manner. However, the Hermitian Hamilton operator $H_{B}$ (and $H_{eff}^{h}$ , respectively) is an energy operator and provides only the energies of the states in a proper manner.

In the standard theory of the Hermitian quantum mechanics, the states have an infinitely long lifetime. In order to calculate the experimentally well-known finite lifetimes of most states e.g. in nuclei, corrections to this basic assumption are treated by perturbation theory. The finite lifetimes $\tau_{i}^{B}$ are calculated from

$$
\Gamma_ {i} ^ {B} = \sum_ {\mathrm{c}} \left| \hat {\gamma} _ {i} ^ {c} \right| ^ {2} \propto \frac {1}{\tau_ {i} ^ {B}}\tag{25}
$$

where the partial widths $\left|\hat{\gamma}_{i}^{c}\right|$ are defined by (9). This relation follows by assuming that the coupling of the system to the continuum does not have any feedback. By taking into account the feedback between system and environment, relation (25) has to be replaced by (16). That means, relation (25) used in standard quantum mechanics for the calculation of the lifetimes $\tau_{i}^{B}$ holds only for isolated resonance states, i.e. for resonance states that are well separated from neighboring ones.

In contrast to $H_{B}$ , the Hamiltonian $H_{eff}$ is non-Hermitian. Its complex eigenvalues $z_{\lambda}$ provide not only the energies $E_{\lambda}$ of the resonance states, but also their lifetimes $\tau_{\lambda} \propto \Gamma_{\lambda}^{-1}$ . The energy shifts $\Delta E = E_{i}^{B} - E_{\lambda}$ and the finite lifetimes $\tau_{\lambda}$ of the eigenstates of $H_{eff}$ follow from the principal value integral and the residuum, respectively, of the second term of the non-Hermitian operator $H_{eff}$ , equation (4). They arise from the embedding of the system into the continuum of decay channels.

Using the Wigner–Smith time delay function, the relation between $\Gamma_{\lambda}$ and the time the wave spends inside a quantum billiard with one lead attached to it is considered in [47]. As a function of increasing coupling strength between waveguide and billiard, the time delay functions and the decay widths $\Gamma_{\lambda}$ of three resonance states have been calculated. As a result, two of the states become long lived (trapped) due to width bifurcation while the width of the third one continues to increase for all coupling strengths. This behavior of the widths of the states is reflected in the time delay function; see figure 3. Thus, the widths of the resonance states have a physical meaning even in this case in which width bifurcation plays a role.

(a)
![](images/e994697ee65d667de669f5057af11ab151f2f2a94c3c7c9ef62f6a6b6f47234f.jpg)

![](images/182516e5ee04ed346d9e79f6cde9f16160a3dd14c0847e7483f084b9907d4144.jpg)
Figure 3. (a) Contour and surface plot of $\ln (\tau_w)$ where $\tau_{w}$ is the time the wave spends inside the system (Wigner–Smith time delay function). The system is a rectangular billiard with one attached waveguide. The parameter $\lambda$ stands for the coupling strength between billiard and waveguide (increasing $\lambda$ corresponds to decreasing coupling strength). The darker the plot, the larger the time delay. (b) The motion of the corresponding eigenvalues of $H_{\mathrm{eff}}$ with $\lambda$ . The positions of the resonance states for $\lambda = 44$ are denoted by squares, those for $\lambda = 23.5$ and $\lambda = 0$ by large dots. The time delay corresponds, in any case, to the motion of the eigenvalues. The width of the state at about $E_R = 38.7$ first increases and then decreases with $\lambda$ , while the energy shift of this state is, finally, almost zero. Correspondingly, the wavefunction of this state at large $\lambda$ is almost the same as that at small $\alpha$ [47]. The wavefunction of the state moving from about $E_R = 39.6$ to 39.0 with decreasing $\lambda$ , is mixed at small $\alpha$ (large coupling strength) with the wavefunction of the broad state [47]. Figure taken from [47].

Let us now consider the time-dependent Schrödinger equation describing the resonance state $\lambda$ at the energy E without taking into account the interaction of the state $\lambda$ with other states in the neighborhood,

$$
H _ {\text { eff }} \phi_ {\lambda} (t) = \mathrm{i} \hbar \frac {\partial}{\partial t} \phi_ {\lambda} (t).\tag{26}
$$

The right and left solutions $\phi_{\lambda}$ and $\psi_{\lambda}$ may be represented by

$$
| \phi_ {\lambda} (t) \rangle = \mathrm{e} ^ {- \mathrm{i} H _ {\mathrm{eff}} t / \hbar} | \phi_ {\lambda} (t _ {0}) \rangle = \mathrm{e} ^ {- \mathrm{i} z _ {\lambda} t / \hbar} | \phi_ {\lambda} \rangle ,\tag{27}
$$

$$
\langle \psi_ {\lambda} (t) | = \langle \psi (t _ {0}) | \mathrm{e} ^ {\mathrm{i} H _ {\mathrm{eff}} ^ {\dagger} t / \hbar} = \langle \psi_ {\lambda} | \mathrm{e} ^ {\mathrm{i} z _ {\lambda} ^ {*} t / \hbar},\tag{28}
$$

with $\langle \psi_{\lambda}| = \langle \phi_{\lambda}^{*}|$ , see section 2.2. By means of (27) and (28) the population probability of the state $\lambda$ can be obtained. It reads

$$
\langle \phi_ {\lambda} ^ {*} (t) | \phi_ {\lambda} (t) \rangle = \mathrm{e} ^ {- \Gamma_ {\lambda} t / \hbar},\tag{29}
$$

and the decay rate of the state $\lambda$ is

$$
k _ {\lambda} (t) = - \frac {\partial}{\partial t} \ln \left\langle \phi_ {\lambda} ^ {*} (t) \mid \phi_ {\lambda} (t) \right\rangle = \frac {1}{\hbar} \Gamma_ {\lambda}.\tag{30}
$$

Thus, the physical meaning of the decay width $\Gamma_{\lambda}$ of the state $\lambda$ is its relation to the decay rate $k_{\lambda}$ and the lifetime $\tau_{\lambda}$ , respectively. It is independent of time t, as long as the resonance state is not overlapped by another resonance state (as supposed in (26)).

Equations (26)-(30) show the physical meaning of the non-Hermitian part of $H_{\mathrm{eff}}$ : it may be identified with the time evolution operator. The lifetime $\tau_{\lambda}$ of a resonance state follows from the eigenvalue $z_{\lambda}$ of $H_{\mathrm{eff}}$ in the same manner as the energy $E_{\lambda}$ of this state. Both quantities are fundamentally different from the time $t$ and the energy $E$ . They characterize the state $\lambda$ while $t$ and $E$ appear as general parameters. In this theory, the width $\Gamma_{\lambda}$ of the resonance state $\lambda$ appears as the uncertainty of the energy $E_{\lambda}$ of this state.

Mathematically, the existence of the time evolution operator entails the time asymmetry contained in the non-Hermitian Hamilton operator $H_{eff}$ , equation (4), of an open quantum system. Furthermore, the energies $E_{\lambda}$ and lifetimes $\tau_{\lambda}$ of the resonance states $\lambda$ of an open quantum system are bounded from (at least) below, see section 3.6 for the discussion of the brachistochrone problem in open quantum systems.

## 3. Description of the open quantum system and the scattering problem

## 3.1. Feshbach projection operator formalism

In order to solve the scattering problem, the equation

$$
(H - E) \Psi_ {\mathrm{c}} ^ {E} = 0\tag{31}
$$

with the Hermitian Hamilton operator $H$ has to be solved in the whole function space that contains not only the wavefunctions of the localized part of the quantum system but also those of the extended scattering states. The mathematical properties of the two wavefunction sets are different: they are orthonormalized according to $\delta_{ik}$ and $\delta(E - E')$ , respectively. It is therefore very difficult to find the exact solutions $\Psi_{\mathrm{c}}^{E}$ of (31) in a direct manner, see the Friedrichs model [1].

A very powerful method to solve (31) has been proposed by Feshbach about 50 years ago [5]. The basic idea is to define two subspaces $P$ and $Q$ (with $\mathcal{P} + \mathcal{Q} = 1$ for the corresponding projection operators $\mathcal{P}$ and $\mathcal{Q}$ ) such that all wavefunctions of a subspace are orthonormalized in the same manner, either according to $\delta_{ik}$ ( $Q$ subspace) or according to $\delta(E - E')$ ( $P$ subspace). The solutions in the two subspaces can be found by using standard methods, and the solution $\Psi_c^E$ can be obtained by applying $\mathcal{P} + \mathcal{Q} = 1$ to (31), as will be shown below.

Let us define the two projection operators by means of the solutions to (5) and (6),

$$
\mathcal {Q} = \sum_ {i = 1} ^ {N} \left| \Phi_ {i} ^ {B} \right\rangle \left\langle \Phi_ {i} ^ {B} \right|, \quad \mathcal {P} = \sum_ {c = 1} ^ {K} \int_ {\epsilon_ {c}} ^ {\infty} d E \left| \xi_ {E} ^ {c} \right\rangle \left\langle \xi_ {E} ^ {c} \right|\tag{32}
$$

with $Q \cdot \xi_{E}^{c} = 0$ ; $P \cdot \Phi_{i}^{B} = 0$ . We identify $H_{B}$ with QHQ, $H_{C}$ with PHP, and denote QHP by $V_{BC}$ and PHQ by $V_{CB}$ (where V stands for the interaction between the two subspaces). Thus, $H = H_{B} + H_{C} + V_{BC} + V_{CB}$ .

Assuming $Q + P = 1$ , a third wavefunction can be determined by solving the coupled-channel equations with source term [21]

$$
\hat {\omega} _ {i} = G _ {C} ^ {(+)} V _ {C B} \cdot \Phi_ {i} ^ {B},\tag{33}
$$

where $G_{C}^{(+)} = \mathcal{P}(E - H_{C})^{-1}\mathcal{P}$ is the Green function in the P subspace. In contrast to the wavefunctions $\left\{\Phi_{i}^{B}\right\}$ and $\left\{\xi_{E}^{c}\right\}$ , the wavefunctions $\{\hat{\omega}_{i}\}$ contain the information on the coupling between the two subspaces (source term in (33)). Using the representation (32) of the P operator, one gets

$$
\sum_ {c ^ {\prime}} (H _ {C} - E) \bigl \langle \xi_ {E} ^ {c ^ {\prime}} \bigr | \hat {\omega} _ {i} \bigr \rangle = - \frac {1}{\sqrt {2 \pi}} \cdot \hat {\gamma} _ {i} ^ {c} (E)\tag{34}
$$

with $\hat{\gamma}_i^c$ defined in (9).

Using the three function sets $\left\{\Phi_{i}^{B}\right\},\left\{\xi_{E}^{c}\right\}$ and $\{\hat{\omega}_{i}\}$ , the solution $\Psi = Q\Psi + P\Psi$ in the total function space can be obtained in the following manner. From (31) follows

$$
(H _ {C} - E) \cdot \mathcal {P} \Psi_ {\mathrm{c}} ^ {E} = - V _ {C B} \cdot \mathcal {Q} \Psi_ {\mathrm{c}} ^ {E},\tag{35}
$$

$$
(H _ {B} - E) \cdot \mathcal {Q} \Psi_ {\mathrm{c}} ^ {E} = - V _ {B C} \cdot \mathcal {P} \Psi_ {\mathrm{c}} ^ {E}\tag{36}
$$

and

$$
\mathcal {P} \Psi_ {\mathrm{c}} ^ {E} = \xi_ {\mathrm{c}} ^ {E} + G _ {C} ^ {(+)} V _ {C B} \cdot \mathcal {Q} \Psi_ {\mathrm{c}} ^ {E},
$$

$$
\mathcal {Q} \Psi_ {\mathrm{c}} ^ {E} = (E - H _ {\mathrm{eff}}) ^ {- 1} \cdot V _ {B C} \cdot \xi_ {\mathrm{c}} ^ {E}.\tag{37}
$$

(38)

Here $H_{\mathrm{eff}}$ is the effective non-Hermitian Hamilton operator defined in (4). Further,

$$
\begin{array}{r l} & {\Psi_ {\mathrm{c}} ^ {E} = (\mathcal {P} + \mathcal {Q}) \Psi_ {\mathrm{c}} ^ {E}} \\ & {\quad = \xi_ {\mathrm{c}} ^ {E} + (1 + G _ {C} ^ {(+)} V _ {C B}) \cdot \mathcal {Q} \Psi_ {\mathrm{c}} ^ {E}} \\ & {\quad = \xi_ {\mathrm{c}} ^ {E} + (1 + G _ {C} ^ {(+)} V _ {C B}) \cdot (E - H _ {\mathrm{eff}}) ^ {- 1} \cdot V _ {B C} \cdot \xi_ {\mathrm{c}} ^ {E}} \end{array}\tag{39}
$$

and with (33)

$$
\Psi_ {\mathrm{c}} ^ {E} = \xi_ {\mathrm{c}} ^ {E} + \sum_ {i j} \left(\Phi_ {i} ^ {B} + \hat {\omega} _ {i}\right) \left\langle \Phi_ {i} ^ {B} \right| \frac {1}{E - H _ {\text {eff}}} \left| \Phi_ {j} ^ {B} \right\rangle \left\langle \Phi_ {j} ^ {B} \right| V _ {B C} \left| \xi_ {\mathrm{c}} ^ {E} \right\rangle .\tag{40}
$$

This solution is called formal solution of the problem by Feshbach [5]. Corresponding to this statement, the distribution of $\Phi_{i}^{B}$ and of the coupling coefficients $\left\langle\Phi_{j}^{B}\mid V_{BC}\mid\xi_{c}^{E}\right\rangle$ is obtained from statistical assumptions.

In [21], the solutions $\Psi_{c}^{E}$ are calculated, for the first time, without any statistical assumptions. Using the complex eigenfunctions $\phi_{\lambda}$ and eigenvalues $z_{\lambda}$ of $H_{eff}$ , equation (10), the solution $\Psi_{C}^{E}$ of the Schrödinger equation (31) in the total function space of discrete and scattering states reads

$$
\Psi_ {\mathrm{c}} ^ {E} = \xi_ {\mathrm{c}} ^ {E} + \frac {1}{\sqrt {2 \pi}} \sum_ {\lambda = 1} ^ {N} \Omega_ {\lambda} \cdot \frac {\gamma_ {\lambda} ^ {c}}{E - z _ {\lambda}}.\tag{41}
$$

Here

$$
\Omega_ {\lambda} = \phi_ {\lambda} + \omega_ {\lambda} = \big (1 + G _ {C} ^ {(+)} V _ {C B} \big) \phi_ {\lambda}\tag{42}
$$

with $\omega_{\lambda}$ defined by

$$
\omega_ {\lambda} = G _ {C} ^ {(+)} V _ {C B} \cdot \phi_ {\lambda}\tag{43}
$$

in analogy to (33), and $\gamma_{\lambda}^{c}$ defined in (15). The function $\Omega_{\lambda}$ is the wavefunction of the resonance state $\lambda$ . Equation (41) is an exact solution of the scattering problem (31) provided that all interactions are known analytically and $\mathcal{Q} + \mathcal{P} = 1$ holds.

According to (42), $\Omega_{\lambda} \approx \phi_{\lambda}$ in the interior of the system while the asymptotic behavior of the wavefunction of the resonance state is given by $G_C^{(+)} V_{CB} \phi_{\lambda}$ . Therefore, the characteristic feature of the FPO formalism consists, above all, of the fact that the scattering wavefunction $\Psi_c^E$ can be represented, in the interior of the system, in the set of (biorthogonal) wavefunctions $\{\phi_{\lambda}\}$ that describe the open quantum system with the non-Hermitian Hamilton operator $H_{\mathrm{eff}}$ . In other words, the eigenfunctions $\phi_{\lambda}$ of the non-Hermitian Hamilton operator $H_{\mathrm{eff}}$ determine the part $\hat{\Psi}_c^E$ of the scattering wavefunction $\Psi_c^E$ that is localized in the interior of the system,

$$
\begin{array}{l}\left| \right. \Psi_ {\mathrm{c}} ^ {E} \left. \right\rangle\rightarrow \left| \right. \hat {\Psi} _ {\mathrm{c}} ^ {E} \left. \right\rangle = \sum_ {\lambda} c _ {c \lambda} ^ {E} | \phi_ {\lambda} \rangle\\c _ {c \lambda} ^ {E} = \frac {\langle \phi_ {\lambda} ^ {*} | V _ {B C} | \xi_ {\mathrm{c}} ^ {E} \rangle}{E - z _ {\lambda}} = \frac {1}{\sqrt {2 \pi}} \frac {\gamma_ {\lambda} ^ {c}}{E - z _ {\lambda}}.\end{array}\tag{44}
$$

The coefficients $c_{c\lambda}^{E}$ depend strongly on energy. Most important is however the following fact. Due to relation (44), the physical phenomena caused by the true and avoided crossings of the eigenvalue trajectories of $H_{eff}$ survive when the problem (31) in the whole function space is solved.

Feshbach applied the projection operator formalism, about 50 years ago, to heavy nuclei with excitation of neutron resonances (compound nucleus resonances) [5]. Here, the level density is high ( $10^{4}$ to $10^{6}$ states in an energy interval typical for the corresponding particle-particle interaction) and the neutron resonances are well isolated from one another in energy due to their extremely long lifetimes. The resonance states as well as the coupling coefficients to the continuum are treated successfully by means of statistical methods. In these studies, the non-Hermitian Hamilton operator is not considered explicitly. It is rather approximated, phenomenologically, by the assumptions contained in the statistical ensembles used in the study.

The situation is another one in light nuclei where the level density is low and the resonance states keep, to a large extent, their individual features. They cannot be treated by statistical methods. All the coupling matrix elements have to be calculated. This is done first in $[21]$ for nuclei around ${}^{16}O$ . Meanwhile, the FPO formalism is applied also to other nuclei $[16, 17]$ and to small quantum systems such as atoms and quantum dots (quantum billiards) $[16]$ . In all calculations, the solution (41) of the scattering problem (31) is obtained numerically, the non-Hermiticity of $H_{eff}$ is considered explicitly, and its consequences for the dynamics of these systems are studied. The main problem in solving the scattering problem (31) consists, in these calculations, in a meaningful definition of the two subspaces Q (system) and P (environment) such that the eigenvalues $z_{\lambda}$ of $H_{eff}$ can be identified with the energies and widths of the resonance states. The criteria are the following [21]: the system (Q subspace) is localized and contains all resonance-like phenomena while the environment (P subspace) is extended and causes the smooth part of the scattering process in the energy region considered.

The characteristic features of the FPO formalism being contained in the solutions (41) and (44), respectively, of the scattering problem (31) consist in the following.

(i) The spectroscopic information on the resonance states is obtained directly from the complex eigenvalues $z_{\lambda}$ and eigenfunctions $\phi_{\lambda}$ of the non-Hermitian Hamilton operator $H_{eff}$ . $z_{\lambda}$ and $\phi_{\lambda}$ are energy-dependent functions, generally, and contain the influence of neighboring resonance states as well as of decay thresholds onto the considered state $\lambda$ . This energy dependence allows us to describe decay and resonance phenomena also in the very neighborhood of decay thresholds (section 3.2) and in the regime of overlapping resonances (section 3.5).

(ii) The resonance states are directly related to the discrete states of a (many-body) closed system described by standard quantum mechanics (with the Hermitian Hamilton operator $H_{B}$ ). They are generated by opening the system what is achieved by coupling the discrete states to the environment of scattering states by means of the second term of the Hamilton operator $H_{eff}$ . Therefore, they are realistic localized many-particle states of an open quantum system, that have, generally, a finite lifetime when lying above the lowest decay threshold (or inside the window of conductance), and an infinitely long lifetime when lying below this threshold (or outside the window). The transition from resonance states (described by the non-Hermitian $H_{eff}$ ) to discrete states (described by the Hermitian $H_{B}$ ) can be controlled.

(iii) The properties of branch points in the continuum and their vicinity can be studied relatively easy. At these points, two (or more) eigenvalues $z_{\lambda}$ of $H_{\mathrm{eff}}$ coalesce. Since it is not necessary to consider the poles of the $S$ matrix in the FPO formalism, additional mathematical problems at and in the vicinity of branch points are avoided.

(iv) The phases of the eigenfunctions $\phi_{\lambda}$ of $H_{eff}$ are not rigid in the vicinity of a branch point. This fact allows us to describe the spectroscopic reordering processes in the system that take place under the influence of the scattering wavefunctions $\xi_{c}^{E}$ of the environment into which the system is embedded.

## 3.2. The S matrix

By means of the solution $\Psi_{\mathrm{c}}^{E}$ of the scattering problem (31) it is possible to find the $S$ matrix [49],

$$
\begin{array}{l} S _ {c c ^ {\prime}} = \delta_ {c c ^ {\prime}} - \int \frac {\left\langle \chi_ {c ^ {\prime}} ^ {E} \right| V \left| \Psi_ {\mathrm{c}} ^ {E} \right\rangle}{E - E ^ {\prime}} \mathrm{d} E ^ {\prime} \\ = \delta_ {c c ^ {\prime}} - \mathcal {P} \int \frac {\left\langle \chi_ {c ^ {\prime}} ^ {E} \right| V \left| \Psi_ {\mathrm{c}} ^ {E} \right\rangle}{E - E ^ {\prime}} \mathrm{d} E ^ {\prime} - 2 \mathrm{i} \pi \bigl \langle \chi_ {c ^ {\prime}} ^ {E} \big | V \left| \Psi_ {\mathrm{c}} ^ {E} \right\rangle . \end{array}\tag{45}
$$

Here, c and $c'$ belong to the same set of basic (uncoupled) channel wavefunctions $\left\{\chi_{c}^{E}\right\}$ . The principal value integral depends smoothly on energy while the residuum shows a resonance-like behavior on energy. With (41), we can write

$$
S _ {c c ^ {\prime}} = \delta_ {c c ^ {\prime}} - S _ {c c ^ {\prime}} ^ {(1)} - S _ {c c ^ {\prime}} ^ {(2)}\tag{46}
$$

where

$$
S _ {c c ^ {\prime}} ^ {(1)} = \mathcal {P} \int \frac {\left\langle \chi_ {c ^ {\prime}} ^ {E} \right| V \left| \Psi_ {\mathrm{c}} ^ {E} \right\rangle}{E - E ^ {\prime}} \mathrm{d} E ^ {\prime} + 2 \mathrm{i} \pi \left\langle \chi_ {c ^ {\prime}} ^ {E} \right| V \left| \xi_ {\mathrm{c}} ^ {E} \right\rangle\tag{47}
$$

is smoothly dependent on energy while

$$
S _ {c c ^ {\prime}} ^ {(2)} = \mathrm{i} \sqrt {2 \pi} \sum_ {\lambda = 1} ^ {N} \left\langle \chi_ {c ^ {\prime}} ^ {E} \right| V | \Omega_ {\lambda} \rangle \cdot \frac {\gamma_ {\lambda} ^ {c}}{E - z _ {\lambda}}\tag{48}
$$

is the resonance term of the S matrix. It contains the excitation of the resonance state $\Omega_{\lambda}$ , equation (42), from the channel $c'$ with wavefunction $\chi_{c'}^{E}$ (incoming wave) as well as the decay of the eigenstate $\phi_{\lambda}$ of $H_{eff}$ into the channel c (outgoing wave) which is described by $\gamma_{\lambda}^{c}$ , equation (15).

Relation (42) between the wavefunction $\Omega_{\lambda}$ of the resonance state and the eigenfunction $\phi_{\lambda}$ of $H_{\mathrm{eff}}$ is completely analogous to the Lippman-Schwinger equation

$$
\xi_ {\mathrm{c}} ^ {E} = \big (1 + G _ {C} ^ {(+)} \cdot V _ {C} \big) \chi_ {\mathrm{c}} ^ {E}\tag{49}
$$

between the two scattering wavefunctions. One arrives therefore at [49]

$$
\left\langle \chi_ {c ^ {\prime}} ^ {E} \right| V _ {C B} | \Omega_ {\lambda} \rangle = \left\langle \xi_ {c ^ {\prime}} ^ {E} \right| V _ {C B} | \phi_ {\lambda} \rangle .\tag{50}
$$

Using this relation, the resonance part (48) of the S matrix reads

$$
S _ {c c ^ {\prime}} ^ {(2)} = \mathrm{i} \sum_ {\lambda = 1} ^ {N} \frac {\gamma_ {\lambda} ^ {c ^ {\prime}} \gamma_ {\lambda} ^ {c}}{E - z _ {\lambda}} \equiv 2 \mathrm{i} \pi \sum_ {\lambda = 1} ^ {N} \frac {\left\langle \xi_ {c ^ {\prime}} ^ {E} \right| V _ {C B} | \phi_ {\lambda} \rangle \left\langle \phi_ {\lambda} ^ {*} \mid V _ {B C} \mid \xi_ {\mathrm{c}} ^ {E} \right\rangle}{E - z _ {\lambda}}.\tag{51}
$$

Here, $\gamma_{\lambda}^{c'}$ is related to the incoming wave in channel $c'$ while $\gamma_{\lambda}^{c}$ is related to the outgoing wave in channel c. Although (51) contains only the product $\gamma_{\lambda}^{c}\gamma_{\lambda}^{c'}$ , equation (48) shows that one of these factors stands for the excitation of the (extended) resonance state $\Omega_{\lambda}$ and the other one for the decay of the (localized) eigenstate $\phi_{\lambda}$ of $H_{eff}$ .

In (51), excitation and decay of the resonance state $\lambda$ take place each via one of the channels belonging to the set of channel wavefunctions $\{\chi_{E}^{c}\}$ . While the decay occurs, in any case, to one of the channel wavefunctions $\chi_{E}^{c}$ , the excitation may occur in a completely different manner. Examples are photo-nuclear reactions and the excitation of an optically prepared sample of ultra-cold atoms. In the first case, the process may be described by the inverse $(\gamma, n)$ reaction on the target nucleus, and the Schrödinger equation reads [21]

$$
(H - E) \Psi_ {F} ^ {E} = F\tag{52}
$$

instead of (31). The source term F describes the excitation of the state $\Psi_{F}^{E}$ by the interaction of the electromagnetic field with the ground state of the target nucleus. The amplitude of the resonance part of the S matrix we are interested in, is given by [21]

$$
S ^ {\mathrm{res}} = 2 \pi \mathrm{i} \sum_ {\lambda} \frac {\left\langle \xi_ {F} ^ {E} \right| V _ {C B} | \phi_ {\lambda} \rangle \left\langle \phi_ {\lambda} ^ {*} \right| V _ {B C} \left| \xi_ {\mathrm{c}} ^ {E} \right\rangle}{E - z _ {\lambda}}.\tag{53}
$$

This expression contains the excitation of the state $\lambda$ via the source term F, equation (52), and its decay into the channel c. Both equations (31) and (52) are defined in the whole function space $(\mathcal{P} + \mathcal{Q} = 1)$ such that the Hamilton operator H appearing in these equations, is Hermitian. In the present paper, only F = 0 will be considered, correspondingly to the scattering process.

Expression (51) for the resonance part of the S matrix is formally the same as that of the standard theory. In the unified theory of nuclear reactions formulated by Feshbach [5], equation (41) is approximated by using statistical assumptions for the resonance states and their coupling matrix elements to the continuum. Only the direct reaction part $S_{cc'}^{(1)}$ is treated explicitly. In this approach, all the matrix elements involved in $S_{cc'}^{(2)}$ are energy independent. The spectroscopic information is obtained from the poles of the S matrix.

The differences of the exact expression (51) to the standard expression of the S matrix are the following.

(i) According to (48), the resonance phenomena of the $S$ matrix are determined by the decay properties of the resonance states that are described by the complex eigenvalues $z_{\lambda}$ and dual eigenfunctions $\phi_{\lambda}, \phi_{\lambda}^{*}$ of the non-Hermitian Hamilton operator $H_{\mathrm{eff}}$ . Thus, the FPO formalism provides a unified description of resonance and decay phenomena.

(ii) The spectroscopic information is obtained from the eigenvalues and eigenfunctions of the non-Hermitian Hamiltonian $H_{\mathrm{eff}}$ . It is therefore not necessary to consider the poles of the $S$ matrix. For comparison with the results of the standard theory, the poles of the $S$ matrix can be determined approximately by solving the fixed-point equations $E_{\lambda} = \mathrm{Re}(z_{\lambda})_{|E = E_{\lambda}}$ and finding $\Gamma_{\lambda} = -2\mathrm{Im}(z_{\lambda})_{|E = E_{\lambda}}$ [16]. They can be determined also exactly. Examples are given in [30, 36] for laser-induced continuum structures in atoms and in [50] for resonance scattering on Bargmann-type potentials. However, they are not needed when the FPO formalism is used.

(iii) The energy dependence of $z_{\lambda}$ guarantees the unitarity of the $S$ matrix (51) at all (real) energies $E$ . This holds even at the energy of the branch point where the coupling matrix elements (15) between the states $\lambda$ and the scattering states $\xi_{\mathrm{c}}^{E}$ show a resonance-like behavior [48].

(iv) In the physical observables related to the S matrix (51), the eigenvalues $z_{\lambda}$ with their full energy dependence are contained. Due to this fact, information on the vicinity (in energy) of the considered resonance states such as the position of decay thresholds [51] and of neighboring resonance states [37] is contained in the S matrix and can be received. Such an information cannot be obtained from the poles of the S matrix being (energy-independent) numbers.

(v) Deviations from the Breit–Wigner resonance line shape and from the exponential decay law are involved in expression $(51)$ . These deviations become important for isolated resonance states in the long-time scale due to the fact that the decay thresholds lie at a finite energy $[51]$ . For example, a cusp may appear in the cross section instead of a Breit–Wigner resonance, see figure 4. Qualitatively, this result agrees with experimental data $[52, 53]$ . Another example is the well-known influence of evanescent modes onto the cross section, see figure 5 for an example. At high level density, deviations appear even in the short-time scale due to the mutual influence of neighbored resonance states $[37]$ , see the following sections.

(vi) The cross section obtained from the $S$ matrix is independent of the manner the two subspaces of the FPO formalism are defined as long as $\mathcal{P} + \mathcal{Q} = 1$ is fulfilled. However, in order to receive the spectroscopic information, the two subspaces have to be defined in a meaningful manner, see section 3.1.

(vii) The S matrix (51) provides results that are numerically exact. This result is proven by a numerical analysis performed for exactly solvable potentials with a finite number of resonances [50].

## 3.3. Phase rigidity $\rho$ of the scattering wavefunction

One of the most interesting features of the FPO formalism is that the scattering wavefunction $\Psi_{c}^{E}$ in the interior of the system can be represented in the set of eigenfunctions $\phi_{\lambda}$ of $H_{eff}$ , equation (44). According to section 2.4, the phases of the eigenfunctions $\phi_{\lambda}$ are not rigid in the regime of overlapping resonances. As a consequence, also the phases of the scattering states $\Psi_{\mathrm{c}}^{E}\to \hat{\Psi}_{\mathrm{c}}^{E}$ inside the system will not be rigid when the resonance states overlap. This can be seen in the following manner.

![](images/b73e31a5a0d911edfa4bc2f858b9f5846476ca0b1f1fe10485b416348e7a3da0.jpg)
Figure 4. Influence of the energy dependence of $\Gamma_{\lambda}$ onto the line shape of a resonance in $^{15}\mathrm{N} + n$ . Top: width $\Gamma_{\lambda}$ of the resonance state $\lambda$ . Bottom: line shape of the resonance $\lambda$ lying in the very neighborhood of the threshold. The width $\Gamma_{\lambda}$ increases strongly at the position of the threshold $E_{\mathrm{thr}}$ for opening the inelastic channel to which it is strongly coupled. It influences the line shape of the resonance: the shape may change from a Breit-Wigner-like shape below the threshold to a cusp at the threshold. The calculation is performed in the framework of the continuum shell model for nuclear reactions [21]. The position $E_{i} \equiv E_{R}^{SM}$ of the discrete state is varied by hand such that $E_{\lambda}$ varies from $E_{\lambda_1} < E_{\mathrm{thr}} < E_{\lambda_n}$ where $E_{\lambda_k}(k = 1,\dots,n)$ are the different positions of the resonance state obtained by varying $E_{i}$ . All other parameters of the system are fixed. Figure taken from [51].

From (44) follows for the right and left scattering wavefunctions in the interior of the system

$$
\left| \hat {\Psi} _ {c} ^ {E} \right\rangle = \sum_ {\lambda} c _ {c \lambda} ^ {E} | \phi_ {\lambda} \rangle , \quad \left\langle \hat {\Psi} _ {c ^ {\prime}} ^ {E} \right| = \sum_ {\lambda} d _ {c ^ {\prime} \lambda} ^ {E} \left\langle \phi_ {\lambda} ^ {*} \right|\tag{54}
$$

and

$$
\langle \hat {\Psi} _ {c ^ {\prime}} ^ {E} | \hat {\Psi} _ {\mathrm{c}} ^ {E} \rangle = \sum_ {\lambda \lambda^ {\prime}} d _ {c ^ {\prime} \lambda} ^ {E} c _ {c \lambda} ^ {E} \langle \phi_ {\lambda} ^ {*} | \phi_ {\lambda^ {\prime}} \rangle = \sum_ {\lambda} d _ {c ^ {\prime} \lambda} ^ {E} c _ {c \lambda} ^ {E}\tag{55}
$$

due to (12). When $c = c'$ , it follows

$$
\left\langle \hat {\Psi} _ {c} ^ {E} \right| \hat {\Psi} _ {c} ^ {E} \rangle = \sum_ {\lambda} \left(c _ {c \lambda} ^ {E}\right) ^ {2}\tag{56}
$$

with $d_{c\lambda}^{E} = \left\langle \xi_{\mathrm{c}}^{E}\right|V|\phi_{\lambda}\rangle /(E - z_{\lambda}) = c_{c\lambda}^{E}$ according to (15). Since $(c_{\lambda E})^2$ is a complex number, the normalization

$$
\left<   \hat {\Psi} _ {\mathrm{c}} ^ {E} \left. \right| \hat {\Psi} _ {\mathrm{c}} ^ {E} \rangle = 1\tag{57}
$$

corresponds to a rotation with $\mathrm{Re}\big(c_{c\lambda}^{E}\big)\mathrm{Im}\big(c_{c\lambda}^{E}\big) = 0$ . The normalization has to be done separately at every energy $E$ due to the explicit energy dependence of the $c_{c\lambda}^{E}$ . Further,

$$
\left<   \hat {\Psi} _ {c ^ {\prime}} ^ {E *} \left. \right| \hat {\Psi} _ {\mathrm{c}} ^ {E} \rangle = \sum_ {\lambda \lambda^ {\prime}} d _ {c ^ {\prime} \lambda} ^ {E *} c _ {c \lambda^ {\prime}} ^ {E} \langle \phi_ {\lambda} | \phi_ {\lambda^ {\prime}} \rangle .\tag{58}
$$

![](images/16abb714d0532c5bccf041127cbe97ec646b0bda110433c0da23ceee27cd0690.jpg)

![](images/7986d5dcb07609aaef8e2a44ff5f81fd3924857ae1ff441decd01177f77a8164.jpg)

![](images/cba603ff8f24c1ff8ed89a892e255b1638d61db063ec4605821e2d07566accfc.jpg)
Figure 5. Influence of an evanescent mode on the line shape of a resonance. Top: one resonance state (solid line), middle: the 'tail' of a bound state (solid line) lying slightly below the threshold, bottom: interference between the 'tail' of the bound state and the resonance state (solid line). The nuclear reaction cross section $\sigma^{\mathrm{tot}}$ is calculated for $^{15}\mathrm{O} + n$ in the framework of the continuum shell model [21] with one open neutron channel. The dashed lines show the direct reaction part. Because of the neighborhood to the (elastic) threshold, the resonance shape is nonsymmetric with a comparably long tail to larger energies. Figure taken from [54].

For $c' = c$ follows

$$
\begin{array}{r l} & {\left\langle \hat {\Psi} _ {\mathrm{c}} ^ {E *} \right| \hat {\Psi} _ {\mathrm{c}} ^ {E} \rangle = \sum_ {\lambda} \left| c _ {c \lambda} ^ {E} \right| ^ {2} A _ {\lambda} + \sum_ {\lambda \neq \lambda^ {\prime}} c _ {c \lambda} ^ {E *} c _ {c \lambda^ {\prime}} ^ {E} B _ {\lambda} ^ {\lambda^ {\prime}}} \\ & {\qquad = \sum_ {\lambda} \left| c _ {c \lambda} ^ {E} \right| ^ {2} A _ {\lambda} + \sum_ {\lambda <   \lambda^ {\prime}} \left(c _ {c \lambda} ^ {E *} c _ {c \lambda^ {\prime}} ^ {E} - c _ {c \lambda^ {\prime}} ^ {E *} c _ {c \lambda} ^ {E}\right) B _ {\lambda} ^ {\lambda^ {\prime}}} \end{array}\tag{59}
$$

where definitions (13) and (14) for $A_{\lambda}$ and $B_{\lambda}^{\lambda'}$ , respectively, are used.

It is now possible to define the phase rigidity $\rho$ of the wavefunctions $\hat{\Psi}_{c}^{E}$ in analogy to (23),

$$
\rho = \mathrm{e} ^ {2 \mathrm{i} \theta} \frac {\left<   \tilde {\Psi} _ {\mathrm{c}} ^ {E *} \left. \right|\left. \tilde {\Psi} _ {\mathrm{c}} ^ {E} \right > }{\left<   \tilde {\Psi} _ {\mathrm{c}} ^ {E} \right|\left. \right. \tilde {\Psi} _ {\mathrm{c}} ^ {E} \right > }.\tag{60}
$$

The quantity $\rho$ corresponds to a rotation of $\hat{\Psi}_{\mathrm{c}}^{E}$ through $\theta$ being determined by the ratio between its real and imaginary parts. In spite of the complicated structure of $\rho$ , it holds $1 \geqslant \rho \geqslant 0$ (since $1 \leqslant (a^2 - b^2) / (a^2 + b^2) \leqslant 0$ for every summand $(a + \mathrm{i}b)^2$ in (60)). Equations (57) and (59) show that the definition of $\rho$ is meaningful only if the sum of all the overlapping states $\lambda$ at the energy $E$ is considered and, moreover, the average over energy $E$ of the system is performed, $\rho \to \langle \rho \rangle$ .

It should be underlined that the wavefunctions $\hat{\Psi}_{c}^{E}$ are the exact solutions of the scattering problem in the interior of the system, equation (44), and that the phase rigidity $\langle\rho\rangle$ obtained for these wavefunctions is related to the individual $r_{\lambda}$ (i.e. to the corresponding values $A_{\lambda}$ and $B_{\lambda}^{\lambda'}$ ). The quantities $r_{\lambda}$ characterize the changes of the wavefunctions $\phi_{\lambda}$ of the resonance states $\lambda$ under the influence of their interaction with other resonance states via the continuum of scattering wavefunctions $\xi_{c}^{E}$ , see section 2.4. These quantities belong therefore to the internal properties of an open quantum system. They characterize the internal impurity of the open quantum system. This internal impurity depends on the degree of resonance overlapping and does not vanish at zero temperature.

The phase rigidity $\rho$ can be determined experimentally in an open microwave system [55]. It is, indeed, a highly fluctuating quantity and is different for the wavefunctions of the different states. However, the averaged phase rigidity depends on general properties of the system. It may approach $\langle\rho\rangle\approx0$ , in agreement with the discussion above.

Originally, the notation phase rigidity $\rho_{br}$ is introduced in the standard quantum mechanics by means of an arbitrary wavefunction $\tilde{\Psi}$ [56],

$$
\rho_ {\mathrm{br}} = \mathrm{e} ^ {2 \mathrm{i} \Theta} \frac {\int \mathrm{d} r (| \operatorname{Re} \tilde {\Psi} (r) | ^ {2} - | \operatorname{Im} \tilde {\Psi} (r) | ^ {2})}{\int \mathrm{d} r (| \operatorname{Re} \tilde {\Psi} (r) | ^ {2} + | \operatorname{Im} \tilde {\Psi} (r) | ^ {2})}.\tag{61}
$$

This expression is formally analogous to definition (60). However in the case of $\rho_{br}$ , the source for the reduction of the phase rigidity is an external one, e.g. a magnetic impurity. Expression (61) is used in analyzing experimental data, and the contribution from the internal impurity of an open quantum system is, usually, not considered in this analysis.

## 3.4. Width bifurcation and resonance trapping

The phenomenon of resonance trapping is known since more than 20 years. It was found first in numerical studies of nuclear reactions on light nuclei as a function of increasing level density by using the FPO method (continuum shell model) [57]. In molecules, it is studied first in [58]. The phenomenon is counterintuitive since, with increasing overall coupling strength, most states decouple finally from the continuum.

![](images/b06d52ae3b94b91a902872a366077b9f932e3ec6f97b314492ffa3580958974f.jpg)

![](images/b00b9cdad325482dcdb0111674858241cbb937b3de6d14a679322a909f868d32.jpg)
Figure 6. Eigenvalue trajectories of four resonance states coupled to one decay channel as a function of the overall coupling strength $\alpha$ . The calculations are performed with the toy model (62). Left: equidistant level distribution, right: random level distribution. At large coupling strength, three states are trapped by one state. The trapping occurs hierarchically. Figure taken from [64].

Resonance trapping is a direct consequence of the feedback between environment and system. When the resonance states start to overlap, some reordering processes take place in the open quantum system under the influence of the environment of decay channels, see sections 2.3 and 2.4. Finally, the total coupling strength between system and environment (expressed by the sum of the decay widths $\sum_{\lambda}\Gamma_{\lambda}$ ) is concentrated on a few states while the remaining states are more or less decoupled from the continuum of decay channels (width bifurcation).

Many of the following studies, e.g. [59-65], are performed on the basis of toy models with the non-Hermitian Hamilton operator

$$
H _ {\text { eff }} ^ {\text { to }} = H ^ {0} - \mathrm{i} \alpha V V ^ {+}.\tag{62}
$$

Here $H^{0}$ and $VV^{+}$ are Hermitian, $H^{0}$ characterizes the internal structure of the closed system, $VV^{+}$ stands for the interaction between system and environment (continuum of scattering wavefunctions) and the coupling strength $\alpha$ is an overall parameter. The rank of $H^{0}$ is equal to the number N of states of the system, $i = 1, \ldots, N$ . The coupling matrix V is a $K \times N$ matrix where K is the number of channels, $c = 1, \ldots, K$ . Thus, the rank of $VV^{+}$ is K. At low level density, $H^{0}$ is dominant and all N individual resonance states couple to the continuum. At high level density with overlapping resonance states however, $VV^{+}$ is dominant. Here, only K resonance states couple to the continuum and their widths increase with further increasing $\alpha$ . The rest of the states effectively decouples from the environment. Thus, the simple model (62) gives an understanding for the resonance trapping phenomenon. It describes generic features of open quantum systems. The transition from the regime with N individual resonance states to that with K short-lived and N - K trapped states occurs hierarchically [63, 64].

For illustration of the resonance trapping phenomenon, in figure 6 the eigenvalue trajectories of four resonance states coupled to one channel as a function of the overall coupling strength $\alpha$ in the Hamiltonian (62) are shown. First, the widths of all states increase with increasing $\alpha$ . In the critical region of resonance overlapping, the resonance states approach each other in energy and the widths bifurcate. Finally, the width of only one of the states increases further with increasing $\alpha$ while the widths of all the other states decrease. The hierarchical process of resonance trapping can clearly be seen in the figure. The time delay function shows an analogue behavior [47], see figure 3.

Meanwhile, the resonance trapping phenomenon has been verified experimentally in a microwave cavity that is opened by attaching a lead to it $[66]$ . At large opening, the decoupling of resonance states from the continuum of scattering wavefunctions can be seen clearly.

Another interesting result is that the trapped resonance states at high level density have, in general, more chaotic features than the individual resonance states at low level density. This is true even without taking into account the principal value integral (7) in (4) as has been shown in calculations with the toy model (62): when the internal Hamiltonian $H^{0}$ corresponds to an ordered system with Poissonian spectral fluctuations, the trapped resonance states at large coupling strength $\alpha$ tend to show level repulsion similar to that of the Gaussian orthogonal ensemble, typical of chaotic systems [62]. Taking into account an imaginary part of $\alpha$ in the Hamiltonian (62) would amplify the effect. This is due to the well-known fact that any perturbation of a Poissonian distribution by a Hermitian interaction term in the Hamilton operator induces level repulsion in the system [67].

Indeed, the spectra of nuclei at high level density provide features characteristic of chaotic systems $[68]$ . They are described well by the Gaussian orthogonal ensemble. However, also short-lived states exist and induce some regularity in the system: the so-called single-particle resonances with large decay widths are well known in nuclei. Their widths are larger by several orders of magnitude than those of the narrow compound nucleus resonances. An impressive example are whispering gallery modes that may be formed in microwave cavities with chaotic as well as with regular features when they are opened sufficiently strong to the continuum of scattering wavefunctions $[20, 69]$ . Here, direct deterministic processes are supported when the leads are attached in a suitable manner. They exist simultaneously with indeterministic transport processes.

## 3.5. Dynamical phase transitions

The singular points at which (at least) two eigenvalues of $H_{eff}$ cross are decisively for the dynamics of open quantum systems. For illustration, the cross section with N = 2 resonance states and K = 1 channel, calculated with the toy Hamiltonian (62), is shown in figure 7. While the cross section at low coupling strength $\alpha$ can be described by means of two Breit–Wigner resonances, the cross section at large $\alpha$ is completely different. Here, one narrow resonance is superimposed by a background that originates from the broad state at large coupling strength. The transition from one picture to the other one occurs at $\alpha = 1$ where the eigenvalues of the two states coincide (crossing point). Here, the cross section has a double hump structure [48],

$$
S = 1 + 4 \mathrm{i} \frac {\mathrm{Im} (z _ {d})}{E - z _ {d}} - 4 \frac {\mathrm{Im} (z _ {d}) ^ {2}}{(E - z _ {d}) ^ {2}},\tag{63}
$$

where $z_{1} = z_{2} \equiv z_{d}$ . It vanishes at the position $E_{d}$ of the crossing point due to the interference between the two terms.

The dynamics of the inelastic cross section is interesting. Here, the system is coupled to (at least) K = 2 channels, the elastic and the inelastic channel. In order to see clearly the resonance trapping phenomenon, the number of states should be at least N = 3 such that one of the states can be trapped by the two states that align each to one of the channels. An example of an inelastic process is the transmission through a quantum dot (billiard) with the two channels corresponding to the waves in the two attached leads [28]. For illustration, the eigenvalue trajectories as a function of the coupling strength v between quantum billiard and attached leads as well as the transmission probability through the quantum billiard at three different values of v are shown in figure 8. The crossing point of the eigenvalue trajectories is at $v_{c} = 0.84$ in the case considered, as can be seen from the eigenvalue pictures. At $v \ll v_{c}$ , the transmission probability has three peaks at the positions of the resonance states, as expected in the framework of the standard theory. At $v \approx v_{c}$ , however, the number of peaks is reduced to one in accordance with the resonance trapping effect and K = 2. Most interesting is the transmission probability at $v = 0.53 < v_{c} = 0.84$ where the transmission is plateau-like. Here, the quantum billiard is transparent in a comparably large energy range. At v = 0.53, the phase rigidity is zero (compare figure 11). This picture shows that the dynamics of the inelastic processes is determined by the redistribution processes taking place at $v < v_{c}$ under the influence of the crossing points of the eigenvalue trajectories. At $v = v_{c}$ , the spectroscopic redistribution (resonance trapping) is completed.

![](images/b9ecca02233bae813d81e16de054f9c189393d07f3f3d9a40401cfd69b721d7b.jpg)
Figure 7. The quantity $|1 - S|^{2}$ determining the total cross section for N = 2 states coupled to K = 1 channel for three values of the overall coupling strength $\alpha$ of the Hamiltonian (62): $\alpha = 0.08, 1, 4$ (solid lines). The Breit–Wigner curves calculated from the complex eigenvalues of the two resonances for the same $\alpha$ (dashed lines). At small $\alpha$ (bottom), two narrow resonances appear in the cross section that are well separated from one another. At large $\alpha$ (top), one narrow resonance appears as a dip in the background caused by the broad short-lived state. At $\alpha = 1$ (middle), the complex eigenvalues of the two states coincide. Here, the cross section has a double hump structure described by (63). Only at small $\alpha$ , the cross section can be described by two separated Breit–Wigner resonances. At the critical value $\alpha = 1$ , a dynamical phase transition takes place. Figure taken from [64].

In [65], the model (62) has been used in order to investigate analytically and numerically if and under which conditions the transition from the low-level density scenario to the scenario with overlapping resonance states can be understood as a phase transition. The study is performed with $M = 2N + 1$ resonance states and K = 1 channels such that finally one short-lived mode is formed after M - 1 avoided or true crossings with M - 1 resonance states. It is shown analytically that, in the limit $M \to \infty$ , a simultaneous coalescence of all eigenvalues occurs at a finite real value of $\alpha$ , when the distribution of the real eigenvalues $E_{k}$ of $H^{0}$ and the coupling matrix elements $v_{k}$ (i.e. the elements of the vector V) are appropriately chosen. The most illustrative case is a picket-fence model with equal distance between the states and equal coupling strength of all the states to the continuum, $v \equiv v_{k}$ for all k. More generally, an appropriate condition can be achieved when regions with a smaller level density of the unperturbed states are stronger coupled to the decay channel than those with a higher level density. Such a situation appears, for example, when the level distribution $E_{k}^{2} \approx x^{t}$ and the coupling strength $v_{k}^{2} \approx x^{r}$ are related by $2(r + 1) = t$ [65]. Here, $\alpha_{\mathrm{crit}} = (r + 1) / \pi = t / (2\pi)$ . For the picket-fence model, it is $t = 2$ , $r = 0$ and $\alpha_{\mathrm{crit}} = 1 / \pi$ .

E
![](images/336b5b147519696c3fbdf2cc8133a1dd4bd109564754d598fd65ba1c2323e2d5.jpg)

![](images/02347965c97a8f1c152ebd07bd34ba5cfa39c1c2bf2b8dae4403beef0aaa8aa4.jpg)

![](images/169ac0d2ac846871f4ef3c3cd7d0cdde6564c8f16f1204b9eb39c5ee9e07b05a.jpg)
Figure 8. The evolution of $\mathrm{Re}(z_{k})$ (top) and $\mathrm{Im}(z_{k})$ (middle), k = 1, 3 (solid lines), k = 2 (dashed line), of three eigenvalues of the effective Hamiltonian $H_{eff}$ for a double quantum dot (see [28]) as a function of the coupling strength v to two attached (identical) leads. The two eigenvalues coalesce at $v_{c} = 0.8409$ . Bottom: the transmission probability through the double quantum dot for fixed v = 0.2 (dashed line), v = 0.53 (solid line) and v = 0.83 (dot-dashed line). At small v, the transmission has three narrow peaks at the positions of the resonance states, while there is only one resonance peak at large v that is superimposed by a background (arising from the broad state). At $v = 0.53 < v_{c} = 0.84$ , the double quantum dot is transparent in a broad energy range. Here, the transmission is enhanced and a dynamical phase transition takes place. The calculations are performed at E = 0. The subfigures are taken from [28].

The condition $2(r + 1) = t$ is decisive for the formation of a global short-lived state at the finite critical value $\alpha_{\mathrm{crit}}$ [65]. If $2(r + 1) > t$ , $\alpha_{\mathrm{crit}} \to 0$ in the limit $M \to \infty$ . That means, there exists a state with large decay width already at $\alpha = 0$ . If however $2(r + 1) < t$ , it follows $\alpha_{\mathrm{crit}} \to \infty$ , i.e. the reorganization process occurs locally and never finishes. In this case also an eigenvalue with large imaginary part appears, but now via a successive infinite chain of level repulsions.

Although mathematically the limit $M \to \infty$ is required for the simultaneous coalescence of all eigenvalues, the evolution of the system traced by varying $\alpha$ along the real axis resembles nicely all features of a second-order phase transition even for N = 100 states (when $2(r + 1) = t$ ) [65]. Here the coupling strength $\alpha$ acts as a control parameter while the imaginary part of the large eigenvalue plays the role of an order parameter. Furthermore, it could be shown that the relation between the distribution of unperturbed states and the coupling strength, i.e. between r and t, has to be fulfilled only approximately. If either the level density of $H^{0}$ or the coupling matrix $VV^{+}$ (or both) are additionally altered by noisy perturbations, an abrupt transition occurs at $\alpha_{crit}$ numerically even when only a comparably small configuration space is considered [65]. Under the condition $2(r + 1) \approx t$ , all exceptional points of the system accumulate at some finite real value of the parameter $\alpha = \alpha_{crit}$ . In the limit $M \to \infty$ a perfect coalescence of an infinite number of exceptional points is succeeded [31].

It is interesting to remark that, in the case of a phase transition, the short-lived eigenstate $\lambda_{0}$ is collective in the sense that the number of principle components of its eigenfunction jumps abruptly to its maximal value at the critical value $\alpha_{cr}$ , i.e. its wavefunction consists of a (constructive) superposition of components of all eigenstates of $H^{0}$ . The wavefunctions of all the other M-1 eigenstates of $H_{eff}$ , however, stay almost pure in this basis. In this sense, the short-lived eigenmode with a large imaginary part $\Gamma_{0}/2$ of its eigenvalue is an extremely collective state. This is true, although $\Gamma_{0}/2$ is much smaller than the extension of the spectrum.

The studies based on the Hamiltonian (62) show that width bifurcation and resonance trapping are generic features of open quantum systems. According to sections 2.3 and 2.4, the mechanism is the alignment of states to the scattering states of the environment in which the system is embedded. It causes long-range correlations in the system, especially in the neighborhood of branch (crossing) points, as shown above. However, there are still unsolved problems. For example, the question whether or not true multiple exceptional (crossing) points exist in realistic systems (described by a realistic Hamiltonian $H_{\mathrm{eff}}$ ) is not studied up to now. Another open question is the relation between the phase transition discussed above and the position of a threshold for opening a new decay channel.

The results of microscopic calculations for light nuclei show that the open nuclear system is governed by self-organization $[19, 70]$ . The redistribution of the spectroscopic properties takes place, as in other quantum systems, around some critical value of the strength of the coupling between discrete and scattering states. As a result of the redistribution, the effective number of degrees of freedom is reduced. In the critical region of the coupling strength, information entropy in relation to the discrete states of the closed system is created. This result corresponds to the maximum information entropy principle. The long-lived trapped modes have a large information entropy while the short-lived modes have a small part of the information entropy of the whole system $[70]$ .

Similar results are obtained from a shot-noise analysis of the transport in small quantum cavities with large openings $[20]$ . In this case, direct processes are supported when special states couple strongly to the attached leads, and can result in deterministic transport as signified by a striking system-specific suppression of shot noise.

Environmentally induced quantum-dynamical phase transitions are studied experimentally and theoretically by Pastawski et al [12]. The authors present a set of $^{13}\mathrm{C}-^{1}\mathrm{H}$ cross-polarization NMR data, controlled by the ratio between internal interaction and system-environment interaction strengths. These results clearly show that two dynamical regimes for the $^{13}\mathrm{C}$ polarization exist and that the transition between these two regimes is not a smooth crossover. It has rather the characteristic features of a critical phenomenon, i.e. a non-analytical dependence of the quantum dynamics on the control parameter. The oscillation period diverges at the quantum-dynamical phase transition. The authors developed a model (within the Keldysh formalism) that describes the phases as well as the critical region in detail. The phase transition is manifested not only in the observable swapping frequency but also in the decoherence rate.

It is an interesting task to study in detail the experimental results on the dynamical phase transition in the cross-polarization NMR data in relation to the branch (exceptional) points in the complex plane which are responsible for the dynamics of open quantum systems. The divergent oscillation period at the branch point is surely related to the fact that the two eigenfunctions of $H_{eff}$ are linearly dependent at this point, equation (20). The question whether or not the results can be explained also by using the FPO formalism (sections 3.1 and 3.2) or on the basis of the toy model (62) should be investigated. The results of such a study will give a deeper understanding of the system–environment interaction (including the feedback between system and environment) and guidelines for the manipulation of small quantum systems.

## 3.6. The brachistochrone problem

The brachistochrone problem consists of finding the minimal time for the transition from a given initial state to a given final state of the considered system. The recent discussion of the brachistochrone problem in quantum mechanics started with the paper Faster than Hermitian quantum mechanics by Bender et al [71]. It was shown in this paper that the time can be made arbitrarily small when the system is described by a non-Hermitian PT symmetric Hamiltonian (where P and T stand for parity and time, respectively). Such a phenomenon can also be obtained for dissipative systems [72] such that the effect of a tunable passage time can be attributed to the non-Hermitian nature of the time-evolution operator rather than to its PT symmetry.

As a physical example of the quantum brachistochrone problem, let us consider here the inelastic scattering on the localized part of the system, which is studied in the two-channel case in the framework of the FPO formalism $[37]$ . According to standard quantum mechanics with Hermitian Hamiltonian, the inelastic scattering occurs by means of resonance states lying at the energies $E_{i}$ . The time delay is, for each state i, proportional to its lifetime (inverse proportional to the width $\Gamma_{i}$ of the state). Around the energies $E_{i}$ , the transition through the localized part of the system is maximal while it vanishes in between the positions of the resonance states (if the channel wavefunctions differ only in the spatial coordinates).

This picture is true as long as the resonance states do not overlap. It breaks down in the regime of overlapping resonances where the individual resonance states can no longer be identified. Here, the wavefunctions of the system are partly aligned to the channel wavefunctions such that the time for the transition from a given initial state of the scattering system to a given final state of this system may be radically shortened. At the crossing point of the two eigenvalue trajectories $z_{\lambda}$ , $z_{\lambda'}$ (exceptional point), the transition time from state $\lambda$ to state $\lambda'$ vanishes. However, this is a single point in the continuum, and the transition time at this single point cannot be observed.

For illustration, let us consider the transmission through quantum dots (billiards) with two (almost) identical attached leads and one channel wavefunction in each lead. In this case, the two channel wavefunctions are (almost) identical, except for the spatial coordinates. Furthermore, the two states of the system that align each to one of the scattering states, are also very similar to one another such that the system may become transparent.

That this may happen, indeed, can be seen in the following manner. The resonance part (51) of the inelastic scattering may be rewritten,

$$
S _ {c c ^ {\prime}} ^ {(2)} = 2 \mathrm{i} \pi \bigl \langle \xi_ {c ^ {\prime}} ^ {E} \bigr | V \bigr | \hat {\Psi} _ {\mathrm{c}} ^ {E} \bigr \rangle\tag{64}
$$

by using (44). In contrast to (51), this representation of $S_{cc'}^{(2)}$ does not suggest the existence of resonance peaks in the inelastic cross section. Equation (64) shows that $S_{cc'}^{(2)}$ is determined by the degree of alignment of the wavefunction $\hat{\Psi}_{c}^{E}$ in the interior of the system with the propagating mode $\xi_{c'}^{E}$ in the channel, i.e. by the phase rigidity $\rho$ . According to (59), $\rho$ is determined by the contributions from all overlapping states at the considered energy E. It is small when many overlapping states $\lambda$ are almost aligned such that, after averaging over energy, $\langle\rho\rangle\to0$ in a certain finite energy region. This happens at $v<v_{c}$ where $v_{c}$ is the critical coupling strength (at which K short-lived and N-K trapped long-lived states are finally formed, in a hierarchical manner, and $\rho\approx1$ ). In the limiting case, $\rho\approx0$ is correlated with $S_{cc'}^{(2)}\approx1$ . In contrast to the transmission via an isolated resonance state with rigid phase, the transmission is maximal and the time delay is minimal not only at the single point $E=E_{i}$ but in the whole energy region where $\langle\rho\rangle\approx0$ .

A numerical study is performed [38] for the transmission through a system with three states coupled to two identical attached leads, see figures 8 and 11 (left part). At the value $v = v^{\mathrm{pl}}$ of the coupling strength, a plateau in the transmission with $S_{cc'}^{(2)} = 1$ appears. Here $\rho = 0$ . That means, the system becomes transparent in the whole energy window in which the phase rigidity $\rho$ of the scattering wavefunction vanishes. When averaged over energy, the transmission probability is enhanced in this energy window (as compared to the results of the theory with rigid phases of the eigenstates of the Hamiltonian). Furthermore, the time for the transition from the initial scattering state to the final one is shortened as follows from the eigenvalue picture (figure 8). The enhanced and accelerated transmission is described best by a direct process. Correspondingly, the time is bounded from below in full agreement with the fact that the crossing point of the eigenvalue trajectories (exceptional point) is at $v_{\mathrm{c}} > v^{\mathrm{pl}}$ .

In other words, the time for traveling through the cavity does never vanish. It is bounded from below: it cannot be smaller than the time corresponding to the transparency of the system. In the non-Hermitian physics, it is however shorter than the traveling time obtained in Hermitian physics with rigid phases of the wavefunctions. Thus, non-Hermitian physics allows us to describe the quantum-mechanical brachistochrone problem in a convincing manner.

Also in the many-level case, the value $v^{pl}$ is smaller than the critical coupling strength $v_{c}$ beyond which N - K long-lived trapped states coexist with K short-lived aligned ones (corresponding to $\rho \approx 1$ ). Whispering gallery modes may appear in quantum billiards with convex boundary when the leads are attached to the billiard in a suitable manner. These modes are relatively stable. Transmission of the system via these modes is plateau-like enhanced while the phase rigidity $\rho$ is reduced in the corresponding energy window, see figure 11 (right part). The plateau is well expressed when the number of channels is small [69]. Its relation to fast direct processes is shown in a dynamical analysis of the transport through small quantum cavities with large opening [20].

It should be mentioned that near to $v_{c}$ , a new decay channel may open. The broad state ceases to be localized in space due to its alignment with a scattering state. It is problematic therefore to include it, together with the trapped states, into the Q subspace. However, the cross section is independent of the manner the two subspaces are defined (as long as $Q + P = 1$ ). In any case, $\rho \rightarrow 0$ appears at $v^{pl} < v_{c}$ , see also section 3.5. The enhancement and acceleration of the transmission due to the reduced phase rigidity $\rho$ are expected therefore to be a realistic effect that can be observed.

Recently, the quantum-mechanical brachistochrone system with a PT-symmetric Hamiltonian is reinterpreted as a subsystem of a Hermitian system in a higher dimensional Hilbert space $[73]$ . As a result, the compatibility of the vanishing passage time solution of a PT-symmetric brachistochrone with the lower bound for passage times of Hermitian brachistochrones is demonstrated and, furthermore, a way to a direct experimental implementation in an entangled two-spin system is shown.

These results are only at first glance in contradiction to those discussed above. There is surely agreement at a certain point in the continuum which is, however, of measure zero. Measurable effects are expected if the phase rigidity $r_{\lambda}$ of several neighboring states is small. Then the phase rigidity $\rho$ of the scattering wavefunction $\Psi_{c}^{E}$ may vanish in a certain finite interval of the parameter considered. Here, the passage time becomes minimal but different from zero (as discussed above). Although bounded from below, it is generally shorter than the passage time calculated in the standard theory with rigid phases of the eigenfunctions of the effective Hamiltonian. The passage time is an internal characteristic property of the open system. Up to now, the relation of these results to those of the PT-symmetric theory is not investigated in detail, see section 4.4.

There are interesting consequences of the brachistochrone problem. First, the problem to derive convincingly the uncertainty relation between energy and time in conventional quantum theory with Hermitian Hamilton operator is unsolved up to now, see e.g. $[74]$ (only the uncertainty relation between momentum and space is derived usually in quantum mechanics textbooks). The problem with the uncertainty relation between energy and time is expected to become solvable in non-Hermitian quantum mechanics where both, energy and time, are related to operators (see section 2.6). Second, the instantaneous entanglement of quantum states occurring without any time delay is contained in the eigenfunctions of $H_{eff}$ that describe the open quantum system in the framework of the FPO formalism. Further studies are needed in order to clarify the interesting relation between the different approaches.

## 4. Internal impurity of the open quantum system

## 4.1. Decay rates at high level density

The time-dependent Schrödinger equation reads

$$
H \hat {\Psi} _ {\mathrm{c}} ^ {E} (t) = \mathrm{i} \hbar \frac {\partial}{\partial t} \hat {\Psi} _ {\mathrm{c}} ^ {E} (t)\tag{65}
$$

with H defined in (31) and $\hat{\Psi}_{c}^{E}$ in (44). The right solutions $\hat{\Psi}_{c}^{E}$ may be represented, according to (44), by an ensemble of resonance states $\lambda$ that describes the decay of the localized part of the system at the energy E,

$$
\begin{array}{c} \left| \hat {\Psi} _ {\mathrm{c}} ^ {E} (t) \right\rangle = \mathrm{e} ^ {- \mathrm{i} H _ {\mathrm{eff}} t / \hbar} \left| \hat {\Psi} _ {\mathrm{c}} ^ {E} (t _ {0}) \right\rangle \\ = \sum_ {\lambda} \mathrm{e} ^ {- \mathrm{i} z _ {\lambda} t / \hbar} c _ {c \lambda} ^ {E} | \phi_ {\lambda} \rangle \end{array}\tag{66}
$$

with $c_{c\lambda}^{E} = \langle \phi_{\lambda}^{*}|V|\xi_{\mathrm{c}}^{E}\rangle / (E - z_{\lambda})$ . The left solution of (65) reads

$$
\begin{array}{r} \left\langle \hat {\Psi} _ {c} ^ {E} (t) \right| = \left\langle \hat {\Psi} _ {c} ^ {E} (t _ {0}) \right| \mathrm{e} ^ {\mathrm{i} H _ {\mathrm{eff}} ^ {\dagger} t / \hbar} \\ = \sum_ {\lambda} \langle \phi_ {\lambda} ^ {*} | c _ {c \lambda} ^ {E} \mathrm{e} ^ {\mathrm{i} z _ {\lambda} ^ {*} t / \hbar}. \end{array}\tag{67}
$$

The coefficients $c_{c\lambda}^{E}$ are complex and strongly fluctuating with energy. However averaging provides meaningful quantities, $c_{c\lambda}^{E} \rightarrow c_{c\lambda}$ . By means of (66) and (67) the population probability

$$
\langle \tilde {\Psi} _ {c} (t) | \tilde {\Psi} _ {c} (t) \rangle = \sum_ {\lambda} c _ {c \lambda} ^ {2} e ^ {- \Gamma_ {\lambda} t / \hbar}\tag{68}
$$

(averaged over a certain energy region) can be defined, and the decay rate reads [75]

$$
\begin{array}{r} k _ {\mathrm{gr}} (t) = - \frac {\partial}{\partial t} \ln \langle \tilde {\Psi} _ {\mathrm{c}} (t) | \tilde {\Psi} _ {\mathrm{c}} (t) \rangle \\ = \frac {1}{\hbar} \frac {\sum_ {\lambda} \Gamma_ {\lambda} c _ {c \lambda} ^ {2} \mathrm{e} ^ {- \Gamma_ {\lambda} t / \hbar}}{\sum_ {\lambda} c _ {c \lambda} ^ {2} \mathrm{e} ^ {- \Gamma_ {\lambda} t / \hbar}}. \end{array}\tag{69}
$$

The decay properties of the resonance states can be studied best when their excitation takes place in a time interval that is very short as compared to the lifetime $\tau_{\lambda}$ of the resonance states. In such a case, no perturbation of the decay process by the still continuing excitation process will take place.

For an isolated resonance state $\lambda$ , (69) becomes the standard expression

$$
k _ {\mathrm{gr}} (t) \rightarrow k _ {\lambda} = \Gamma_ {\lambda} / \hbar ,\tag{70}
$$

compare equation (30). In this case, the quantity $k_{\lambda}$ is constant in time and corresponds to the standard relation $\tau_{\lambda} = \hbar / \Gamma_{\lambda}$ with $\tau_{\lambda} = 1 / k_{\lambda}$ . It describes the idealized case of the exponential decay law and, according to (53), a Breit–Wigner resonance in the cross section. Generally, $k_{\mathrm{gr}}(t)$ is time dependent and deviations from the exponential decay law and from the Breit–Wigner line shape appear under the influence of neighboring resonance states and (or) of decay thresholds. Also the background term appearing in most reactions may cause deviations from the ideal exponential decay law, see section 3.2.

Equation (69) describes the decay rate in the regime of overlapping resonances. For numerical examples, see [75]. The overlapping and mutual influence of resonance states is maximal at the crossing points in the complex plane where two eigenvalues $z_{\lambda}$ and $z_{\lambda'}$ of the effective Hamilton operator $H_{\mathrm{eff}}$ coalesce. Nevertheless, the decay rate is everywhere smooth as can be seen also directly from (69). This result coincides with the general statement according to which all observable quantities behave smoothly at singular points.

An interesting result is the saturation of the average decay rate $k_{av}$ in the regime of strongly overlapping resonances. According to the bottleneck picture of the transition state theory, it starts at a certain critical value of bound–continuum coupling [13, 76]. This saturation is caused by width bifurcation [77] (formation of long-lived resonance states by resonance trapping, see section 3.4) occurring in the neighborhood of the branch points in the complex plane. The definition of an average lifetime of the resonance states is meaningful therefore only for either the long-lived states or the short-lived ones. The long-lived (trapped) resonance states are almost decoupled from the continuum of decay channels. Their widths $\Gamma_{\lambda}$ are almost the same for all the different states $\lambda$ , i.e. $\Gamma_{av} \approx \Gamma_{\lambda}$ for all long-lived trapped resonance states. It follows therefore

$$
k _ {\mathrm{av}} \approx \Gamma_ {\mathrm{av}} / \hbar\tag{71}
$$

from (69). According to the average width $\Gamma_{av}$ , the average lifetime of the long-lived states can be defined by $\tau_{av} = 1/k_{av}$ . Then (71) is equivalent to $\tau_{av} = \hbar/\Gamma_{av}$ . That means, the basic relation between lifetimes and decay widths of resonance states holds not only for isolated resonance states (see equation (70)), but also for narrow resonance states superimposed by a smooth background (that may originate from a few short-lived states). In the last case, the relation holds for the averaged quantities $\Gamma_{av}$ and $\tau_{av}$ .

Finally we mention here the experimental data that were obtained for proton scattering on ${}^{58,60}Ni$ about 30 years ago [78]. Lifetime measurements using the crystal blocking technique showed that the directly measured average lifetime of compound nucleus states is significantly larger at the bombarding energy E = 6.50 MeV than at E = 5.65 MeV. Furthermore, the corresponding widths are substantially smaller than those of the observed structures in the excitation functions. These results do not agree with the expectations on the basis of the statistical approach used in the FPO formalism for heavy nuclei. According to the discussion above they may be considered, however, as a hint at the saturation of the average decay widths in the regime of overlapping resonances, i.e. at the resonance trapping effect.

Another interesting result is that the narrow resonance states at high level density do not decay according to an exponential decay law $[79]$ . Calculations for the Gaussian orthogonal ensemble have shown that the decay of the states occurs via an algebraic law (although the narrow states do not overlap) and that the deviation from the exponential law is especially large in the one-channel case. This result is a hint at the fact that the trapped resonance states differ, indeed, from the individual resonance states i at low level density.

## 4.2. Laser-induced continuum structures in atoms

The effects arising in the continuum spectrum of an atom in the vicinity of an autoionizing state coupled to another autoionizing state by a strong resonant laser field are studied, some years ago, in the framework of the FPO formalism. An analytical expression for the photoionization cross section is derived in which the interference between the direct and resonant ionization channels caused by the probe field, and between the transitions induced by the strong field are taken into account $[18]$ . The cross section is described by only a few parameters, what is very convenient for the spectral analysis. The parameters contain characteristics of the field-free atom as well as of the coupling field. The formulae of this approach are tested by comparing the results with those obtained in the non-perturbative time-dependent approach for autoionizing states in helium and with the experimental data for magnesium. In both cases, a good quantitative agreement was found $[18]$ .

In a further study $[36]$ , the motion of the complex energies (eigenvalues of the effective non-Hermitian Hamilton operator $H_{eff}$ ) is traced as a function of the field strength for different field frequencies and atomic parameters. Most interesting is the critical region where a true or avoided crossing of the eigenvalue trajectories occurs. At this critical field intensity, the levels repel each other in the complex plane. With further increasing intensity, the complex energies of the states move differently. When the resonance states are coupled mainly via one common continuum, width bifurcation dominates. When, however, the direct coupling dominates, level repulsion along the real axis takes place. All these effects can be seen in the non-trivial variation of the cross section of a laser-driven atom.

Calculations for the effects induced by two strong laser fields in the continuum of the hydrogen atom are given in $[30]$ . The coupling to the 2s, 5s, 5d, 5g states is considered in the photo-absorption spectra of the probe field from the ground 1s state. The avoided crossing of the quasi-levels in the laser-induced continuum structures is traced theoretically as a function of the intensities of the strong laser fields. Under certain conditions, the quasi-levels cross and the S matrix has a double pole.

In figure 9, the non-trivial motion of the trajectories of the complex energies in the neighborhood of the crossing points (double poles of the S matrix) is shown for different situations. The only trajectories which are almost not influenced by the crossing point are those of the 5g states. The corresponding photo-absorption cross sections from the ground state of the hydrogen atom are also very interesting due to their great diversity $[30]$ .

![](images/78ab3672baebd3ec2489a99b1b8f344f27e6401c6579fc209eef1e9992108d1b.jpg)

![](images/c171f13b456db708d9c3afd68d26c98f8ed189621d2784bcf27e7867fbb8d411.jpg)

![](images/7ace04cbd99632e6aad569c6f8f48958a514fba3c8df1e65d91a9de658246773.jpg)

![](images/b64584476bc2bbe814d5e4a85bd60e0f9975aebedd78608080d3c8069dfbb973.jpg)
Figure 9. The trajectories of the complex eigenvalues of $H_{eff}$ which cross at the critical values of the laser intensities $I_{1}$ and $I_{2}$ , respectively. The intensities as well as the complex energies are given in units of $\Delta = \Delta_{2} - \Delta_{5}$ which is the difference of the resonance detunings of the probe field frequency from those of the lasers. (a) $I_{2} = 5.45 \times 10^{-3}$ fixed and $I_{1}$ varied, (b) $I_{1} = 1.39 \times 10^{-2}$ fixed and $I_{2}$ varied, (c) $I_{1} = 2.44 \times 10^{-2}$ fixed and $I_{2}$ varied and (d) $I_{2} = 8.18 \times 10^{-3}$ fixed and $I_{1}$ varied. The crossing points are marked by a circle. The trajectories show a complicated motion with the only exception of the trajectory of the 5g state which moves linearly. Figure taken from [30].

In another paper [80], the line shape of resonances in the regime of overlapping resonances is studied by using the FPO formalism and the $S$ matrix derived by means of it. A generalized expression $\tilde{q}_{\lambda}$ for the Fano parameter [81] of the resonance state $\lambda$ is derived that contains the interaction of the state $\lambda$ with short-lived and long-lived neighboring states $\lambda' \neq \lambda$ via the continuum. It is energy dependent. Under certain conditions, the energy-dependent $\tilde{q}_{\lambda}$ are equivalent to the generalized complex energy-independent Fano parameters that are introduced in analyzing experimental data on electron transport through mesoscopic systems [82]. Narrow resonances appear mostly isolated from one another in the cross section, also when they are overlapped by short-lived states. An analysis of $\tilde{q}_{\lambda}(E)$ of narrow resonances allows us to study the complicated interplay between different timescales in the regime of overlapping resonance states by controlling them as a function of an external parameter.

The cross section in the neighborhood of a crossing point of the eigenvalue trajectories (double pole of the S matrix) is described, in the one-channel case, by $(63)$ when the direct scattering phase $\delta$ is zero. The second term of this equation corresponds to the usual linear term (up to a constant factor) while the third term is quadratic. The interference between these two parts causes the double hump structure at $\delta = 0$ . In figure 10, the cross section at the crossing point is shown for different direct scattering phases $\delta$ . In any case, the cross section shows a completely other structure than expected without taking into account the coupling between the two resonance states via the continuum (dashed lines in figure 10). Due to this result, the interpretation of resonance data in the regime of overlapping resonances has to be done very carefully. Reliable results can be obtained by controlling the resonance structure by external parameters.

![](images/cfc55aaaf7781b403edbb859ae001b8e6241e82984a9f61012ba95383eda3d56.jpg)
Figure 10. The cross section in the neighborhood of a crossing point of two eigenvalue trajectories (double pole of the S matrix). The direct scattering phase $\delta$ is 0 (a), $\pi/4$ (b), $\pi/2$ (c) and $3\pi/4$ (d). The dashed curves correspond to the case of the two resonance states without any interaction between them (Breit–Wigner resonances). At $\delta = \pi/2$ , the cross section shows one more or less isolated narrow resonance in the middle of the spectrum which results from the interferences between the two resonance states. Figure taken from [80].

## 4.3. Transmission through small quantum dots (billiards)

An unsolved problem in standard quantum mechanics is the description of the crossover from the regime with weak coupling between discrete and continuous states to that with strong coupling between them. For example, an interpolation procedure between the limiting cases with isolated resonances at low level density and narrow resonances at high level density (superimposed by a smooth background term) is introduced in [83] for the transmission through a quantum dot. In contrast to such an interpolation procedure, the crossover can be described in the FPO formalism. Important is the non-rigidity of the phases of the eigenfunctions $\phi_{\lambda}$ of the non-Hermitian Hamiltonian $H_{\mathrm{eff}}$ and, consequently, of the scattering wavefunction $\hat{\Psi}_{\mathrm{c}}^{E}$ inside the localized part of the system.

In [10, 37, 84], the amplitude of the transmission through a microwave cavity is considered in the framework of the $S$ matrix theory by using the FPO formalism,

$$
t = - 2 \pi \mathrm{i} \sum_ {\lambda} \frac {\left\langle \xi_ {L} ^ {E} \right| V | \phi_ {\lambda} \rangle \left\langle \phi_ {\lambda} ^ {*} | V \mid \xi_ {R} ^ {E} \right\rangle}{E - z _ {\lambda}}.\tag{72}
$$

Here, $\xi_{L}^{E}$ and $\xi_{R}^{E}$ are the wavefunctions in the left and right, respectively, attached lead, and t corresponds to the inelastic cross section from channel $c' = L$ to channel c = R, see equation (51). In (72), the eigenvalues $z_{\lambda}$ and eigenfunctions $\phi_{\lambda}$ of $H_{eff}$ are contained with their full energy dependence, see section 3.2.

For $\rho = 1$ and well-isolated resonance states, the transmission amplitude (72) repeats the resonance structure of (44) of the wavefunction $\hat{\Psi}_{C}^{E}$ . The transmission peaks appear at the positions $E_{\lambda} \equiv \mathrm{Re}(z_{\lambda})_{|E=E_{\lambda}} \approx E_{\lambda}^{B}$ of the resonance states. An analogous result holds when there is a nonvanishing background term additional to the resonance term (72) of the transmission amplitude. In this case, the resonances appear as Fano-like resonances on the smooth background. Mostly, they are narrow and well separated from one another. The timescale corresponding to the background term (so-called direct part of the transmission) is, generally, well separated from the scale corresponding to the resonance part described by (72). Due to the different timescales of the resonance and direct processes, it is $|\rho| \approx 1$ also in this case.

The situation is another one when the resonance states overlap. In the overlapping regime, the transmission does not show a resonance structure. Instead, it might be nearly plateau-like $[37, 38, 84]$ . Let us rewrite therefore the transmission amplitude (72) by means of the wavefunction (44),

$$
t = - 2 \pi \mathrm{i} \bigl \langle \xi_ {L} ^ {E} \bigr | V \bigr | \hat {\Psi} _ {R} ^ {E} \bigr \rangle\tag{73}
$$

with $\hat{\Psi}_{R}^{E}$ being complex, in general, see equations (64) and (46). The advantage of this representation consists of the fact that it relates the transmission directly to the degree of alignment of the wavefunction $\hat{\Psi}_{R}^{E}$ with the propagating mode $\xi_{L}^{E}$ in the lead, i.e. to the phase rigidity $\rho$ . Nevertheless, expressions (72) and (73) are fully equivalent.

The numerical results $[84]$ obtained by using the tight-binding lattice Green function method $[8]$ for the transmission through microwave cavities of different shapes show exactly the features discussed above (figure 11). In the weak-coupling regime as well as in the strong-coupling regime, the transmission shows a resonance structure as expected from the standard quantum mechanics. The difference between the two cases is the appearance of a smooth background term in the strong-coupling regime which does not exist in the weak-coupling case, and the reduction of the number of resonance peaks by two (corresponding to the alignment of two states each with one channel in each of the two identical attached leads).

In the crossover from the weak-coupling regime to the strong-coupling one, the calculated transmission is plateau-like instead of showing a resonance structure (figures 8 and 11). It is enhanced as compared to the transmission probability in the two borderline cases. Most interesting is the anticorrelation between transmission $\langle t\rangle$ and phase rigidity $\langle\rho\rangle$ (both quantities averaged over energy) which can be seen very clearly in all the numerical results obtained in [38, 84].

Moreover, the transmission in the crossover regime is not only enhanced. It is also faster than the transmission calculated in standard quantum mechanics, see section 3.6 for the brachistochrone problem. However, the time delay in the cavity cannot be smaller than that allowing traveling through it in accordance with traveling through the attached leads, i.e. the system may become transparent at most. This lower bound can be reached in a system with the non-Hermitian Hamilton operator $H_{eff}$ by aligning the wavefunctions of the system with those of the environment.

![](images/8e09427c0143fa1f0c0ceb227a23c0147123e3530e148a3a167ec07b4418999e.jpg)

![](images/6b632f4729a56c0c8c273116d4bfd767e89fcb0ee4cbdf72ed3d62129c9ca022.jpg)

![](images/4bceba1530e3a9d36052c152341f37aa9ab1f5b071af1975c433f5c43c0cc512.jpg)

![](images/bacaf0dd3dbdd6fa6bd55b2cc85047ac4e068ef26c06d77d5eedf136cb0f0956.jpg)
Figure 11. Left: the transmission probability (top) and the landscape of the phase rigidity $\rho$ (bottom, thin lines) for a double quantum billiard with three states over energy $E$ and coupling strength $v$ . The transmission probability varies between 1 (white) and 0 (black). The distance between the contour lines is $1/30$ . The minimal value $\rho = 0$ is surrounded by a high density of contour lines. The highest shown contour line corresponds to $\rho = 1 - 1/30$ . The $\mathrm{Re}(z_{\lambda})$ of the three eigenstates (thick lines in both panels of the figure) are calculated at $E = 0$ . The crossing point is at $v_{\mathrm{c}} = 0.500$ . Around $v = 0.345 < v_{\mathrm{c}}$ , the phase rigidity is minimal and the transmission maximal and plateau-like (compare figure 8). Right: the transmission probability (top) and phase rigidity $\rho$ (bottom) for a Bunimovich billiard as a function of energy $E$ and coupling strength $v$ between billiard and attached leads. The leads are attached in such a manner that transmission via whispering gallery modes is supported. The transmission probability and the phase rigidity vary between 1 (white) and 0 (black). The calculations are performed in the tight-binding lattice model [8]. Around $v \approx 0.9 - 1$ , the phase rigidity is minimal and the transmission is maximal. The eigenvalue trajectories for $v > 1$ are shown for illustration. The subfigures are taken from [38, 84], respectively.

The behavior of the transmission in the crossover regime with overlapping resonance states is a characteristic of an open quantum system with non-Hermitian Hamiltonian. It does not correspond to the expectations of the standard quantum mechanics with rigid phases of the eigenfunctions of a Hermitian Hamilton operator and with decay widths obtained from the poles of the S matrix, see e.g. [83].

## 4.4. Bound states in the continuum

The question whether or not bound states in the continuum (BICs) exist in realistic quantum systems is of principal interest and might be as well of interest for applications. The reason for this interest arises from the fact that the system is stabilized at the energy of a BIC as well as in its vicinity, and that the wavefunction is localized at all times inside the system in spite of embedding it into the continuum of extended wavefunctions.

![](images/96f943d9e8e3cce276deddc81c83b599ac4d9fe90995bc60920e89fd425c3e65.jpg)

![](images/8df444cb64a9ca4819077c92036277ea3ccbb39ea38b7e079ad30c49941f6cb7.jpg)
Figure 12. The trajectories of the complex energies of two atomic states obtained by varying the laser intensity for different Fano parameters $Q = \Omega_R / \Omega_c$ (where $\Omega_R$ is the Rabi frequency for the direct transition and $\Omega_c$ for the transition via the continuum). It is $Q = 0$ (full line), 0.5 (long-dashed line), 1 (short-dashed line), 5 (dotted line) and 25 (dash-dotted line). All values are given in units of the width $\Gamma_2$ of the resonance state 2. The laser frequency detuning $\Delta$ is equal to the width $\Gamma_2$ . The autoionizing width of the state 1 is $\Gamma_1^a = 0$ (top) and $\Gamma_1^a = 0.2\Gamma_2$ (bottom). If $\Gamma_1^a = 0$ (top), the two eigenvalues cross when $Q = 0$ while they do not cross when $\Gamma_1^a \neq 0$ (bottom). The widths bifurcate at small $Q$ (large contribution from the transition via the continuum) while the levels repel in energy at large $Q$ (small contribution from the transition via the continuum). BICs appear only in the symmetrical case with $\Gamma_1^a = 0$ (top) and if width bifurcation is comparable to level repulsion ( $Q \approx 1$ ). Subfigures taken from [36].

Mathematically, the existence of bound states in the continuum is shown already in 1929 by von Neumann and Wigner [85]. In 1985, Friedrich and Wintgen [86] considered the problem by using the FPO formalism. They related the existence of BICs to avoided level crossings, these being another quantum-mechanical phenomenon discussed by von Neumann and Wigner [87] in 1929. As discussed in section 2.3, avoided level crossings are related to true crossings of eigenvalue trajectories (branch points) and appear in their vicinity.

Since BICs are states that do not decay, the population probability of these states is constant in time. This fact is called population trapping in studies on laser-induced continuum structures in atoms $[88]$ . Similar results are obtained $[30, 36]$ in the time-independent approach by using the FPO formalism and demanding a vanishing decay width for the BIC. In these papers, the relation between BICs and the avoided level crossing phenomenon as well as the stabilization of the system in a broad range of parameter values (characteristic of the laser) is shown explicitly. As an example, the eigenvalue trajectories of two resonance states by varying the laser intensity are shown in figure 12 (top). Most important is that BICs may appear at laser intensities that are relatively small and do not destroy the atom. The condition is that the interaction of the resonance states via the continuum is comparable in value with the direct interaction of the states $Q \approx 1$ .

![](images/18e8248ad70a11749ec8b46a10f71661fd3429f741048dde9f2c663fe32d4c6b.jpg)

![](images/4ec5c2df87033bf0ec57a8f7eba0c98adce616bc65cc3d7b17b1bfdd88f9ed74.jpg)

![](images/c16d5ff9b2ea899fa425eba69afe43a1842978ecdf08ba6a1d013874f2123832.jpg)

![](images/51a5e117bd4fa686d8fbc173795ef5630c3d7d0873c08bb03a5756e182fde51b.jpg)
Figure 13. The real parts (left) and imaginary parts (right) of the five eigenvalues $z_{\lambda}$ of the effective Hamiltonian of a double quantum billiard as a function of the length L of the wire that connects the two single billiards. Top: the spectra of the two single quantum billiards are identical. The eigenvalue trajectory of the state in the middle of the spectrum crosses the energy of the transmission zero at $L = L_{cr} = 2.75$ . At this value of L, the imaginary part of this eigenvalue vanishes at all energies. Bottom: the spectra of the two single billiards differ from one another. There are two neighbored transmission zeros in the middle of the spectrum which are, however, not crossed by the middle eigenvalue trajectory. Correspondingly, there is no critical value of L at which the imaginary part of this eigenvalue vanishes. The two figures illustrate the importance of the spatial symmetry for the appearance of a BIC. Subfigures are taken from [89].

A similar study is performed (by using the FPO formalism) for the transmission through a double quantum dot (billiard) where a BIC appears at that energy at which the resonant transmission crosses a transmission zero $[89]$ . The parameter used in these calculations for the control of the system, is the length L of the wire that connects the two single dots. The eigenvalue trajectories are shown in figure 13 (top) as a function of L. The widths of the states depend strongly on L, especially around $L \approx L_{cr} = 2.75$ where the middle eigenvalue crosses a transmission zero. The width vanishes at $L = L_{cr} = 2.75$ and the eigenstate becomes a BIC at this critical parameter value. Further studies are performed for the transmission through quantum billiards of different shapes $[89, 90]$ .

Common to all these studies on the basis of the FPO formalism is the definition of a BIC as a resonance state with vanishing width,

$$
\left. \Gamma_ {\lambda_ {0}} \right| _ {(E = E _ {\lambda_ {0}})} = 0.\tag{74}
$$

A necessary and sufficient condition to fulfil (74) is the decoupling of the state from all channels of the continuum as can be seen in the following manner. The relation between $\Gamma_{\lambda}$ and the coupling matrix elements is, generally, given by relation (16) which holds true at all energies. As a consequence, a state being decoupled from all channels $c$ of the continuum according to

$$
\left\langle \right. \xi_ {C} ^ {E} \left. \right| V | \phi_ {\lambda_ {0}} \rangle \rightarrow 0\tag{75}
$$

is a BIC with $\Gamma_{\lambda_{0}} \equiv -2\operatorname{Im}(z_{\lambda_{0}}) \to 0$ (condition (75) is equivalent to $\left\langle\phi_{\lambda_{0}}^{*}\mid V\mid\xi_{C}^{E}\right\rangle \to 0$ , see (15)). The opposite case follows by considering the S matrix, see (51) for the amplitude of its resonance part. At the position of a BIC, we have $E - z_{\lambda_{0}} \to 0$ and, due to the unitarity of the S matrix, it follows (75) for all c. Thus, (75) is a necessary and sufficient condition for the BIC when defined by (74). The advantage of definition (74) as compared to the N-level Friedrichs model [2] in studying BICs for unstable multilevel systems is discussed in [91].

The appearance of a BIC can be traced, in the FPO formalism, as a function of a certain control parameter X, i.e. by controlling the trajectories $E_{\lambda}(X)$ and $\Gamma_{\lambda}(X)$ , see figures 12 (top) and 13 (top). The BIC appears at the point $X = X_{0}$ where $\Gamma_{\lambda}(X_{0}) = 0$ . It is even possible to consider the vicinity of the BIC including the cases when $\Gamma_{\lambda}(X')$ is always different from zero and $\Gamma_{\lambda}(X_{0}')$ corresponds to the minimum of $\Gamma_{\lambda}(X')$ with a small but nonvanishing value $\Gamma_{\lambda}(X_{0}')$ , see figures 12 (bottom) and 13 (bottom). This feature of the FPO technique is invaluable for applications since the stabilization of the system (caused by the vanishing width $\Gamma_{\lambda}$ ) must be known not only at the single point $X_{0}$ but also in its vicinity (where $\Gamma_{\lambda} > 0$ , but small) in order to estimate the possibility of an experimental observation.

Figures 12 and 13 show very clearly the relation of BICs and BIC-like states to the avoided level crossing phenomenon and to an external constraint, respectively. Due to the width bifurcation, every BIC and BIC-like state appears together with at least one other state whose width is enhanced around $X = X_{0}$ and whose energy is, generally, different from $E_{\lambda_{0}}$ . Thus, the knowledge of the crossing points of eigenvalue trajectories (double poles of the S matrix) and their vicinity allows one to find the conditions for the stabilization of the system.

In the considered cases, the condition for the exact appearance of a BIC is space reflection symmetry of the system as can be seen from figures 12 and 13. Violation of space reflection symmetry leads to $\Gamma_{\lambda}(X_0')$ small but different from zero, i.e. to a violation of time reflection symmetry. The short-lived states appearing together with the exact BICs due to width bifurcation have the same space reflection symmetry as the BICs. However, time symmetry is broken in this case.

In figures 12 and 13, only single points in the parameter space with real eigenvalues are shown. According to the calculations for a double quantum billiard with a small number of states, a surface of crossing points of the eigenvalues of $H_{eff}$ is defined for four parameters of the system (see figure 2 in [27]). One expects therefore that also the BICs appear in families when considered in a parameter space of higher dimension. The relation of these results to those of the PT symmetric theory [92, 93] for non-Hermitian Hamiltonian systems is not investigated up to now.

It should be underlined here that the BICs are eigenstates of a non-Hermitian Hamilton operator, indeed, in spite of their infinite lifetime. This can be seen from the results represented in figure 14. Here modulus and amplitude of one of the peaks of the transmission through a double quantum dot (corresponding to a peak of the inelastic cross section) are shown as a function of E for different values of the parameter L (being the length of the wire that connects the two single dots). In approaching the critical value $L_{cr} = 2.75$ , the peak narrows and vanishes at $L = L_{cr}$ . The phase, however, becomes more steeply in approaching $L_{cr}$ and passes into a jump at $L = L_{cr}$ . That means, the BIC is nothing but a special resonance state.

Resonance states with very long lifetimes are found recently in a two channel quantum wire with an adatom $[94]$ . They occur in a wide region of the parameter space and are therefore of interest for applications. Such a state, called quasi-bound state in the continuum $[94]$ , arises from a bound state which is slightly destabilized by the existence of the second energy band. The nature of these states differs obviously from that of the BIC-like states although both types of states appear in a wide parameter range.

![](images/e4a3b9539c95b2afa3ee25135641b4f0172ed0c2c91cccf3083b6dbf9f163182.jpg)
Figure 14. The energy dependence of the modulus $|t(E)|$ (bottom) and of the phase $\arg[t(E)]/\pi$ (top) of the amplitude t for the transmission through a double quantum billiard (in an energy window with one transmission peak) for different lengths L of the wire connecting the two single billiards. The motion of the corresponding eigenvalue of $H_{eff}$ is shown in figure 13, top, full line. A transmission zero appears at L = 2.75 denoted by a star in $|t(E)|$ . At the transmission zero, the phase jumps by $\pi$ . The ordinate is shifted every time by 0.1 when L is changed by 0.25. All phases are shifted by $\pi$ . Figure taken from [89].

It is interesting to remark that a phenomenon similar to the appearance of BICs in quantum systems occurs in electromagnetic scattering. An electromagnetic wave of a specific frequency can be trapped forever by a structure that is neither a metal cavity nor a defect in a photonic crystal $[95]$ . It represents, in fact, a bound state in the radiation continuum where the electromagnetic field is trapped by the structure for an infinitely long time. Photonic nanostructures with bound states in the radiation continuum are expected to have many applications, similar as the BICs.

## 4.5. Phase lapses

In experiments [22, 96, 97] on Aharonov–Bohm rings containing a quantum dot in one arm, both the phase and the magnitude of the transmission amplitude $T = |T| \mathrm{e}^{\mathrm{i}\beta}$ of the dot can be extracted. The results obtained caused much discussion since they do not fit into the standard understanding of the transmission process. As a function of the plunger gate voltage $V_g$ , a series of well-separated transmission peaks of rather similar width and height has been observed in many-electron dots and, according to expectations, the transmission phases $\beta(V_g)$ increase continuously by $\pi$ across every resonance. In contrast to expectations, however, $\beta$ always jumps sharply downward by $\pi$ in each valley between any two successive peaks. These jumps called phase lapses were observed in a large succession of valleys for every many-electron dot studied. Only in few-electron dots, the expected so-called mesoscopic behavior is observed, i.e. the phases are sensitive to details of the dot configuration. The problem is considered theoretically in many papers $[6, 7, 98–107]$ over many years without solving it convincingly.

In [108], the generic features of phase lapses in the inelastic cross section are studied by using the S matrix (51) together with the toy model Hamiltonian (62). According to the S matrix, the phase lapses are related to the zeros of the inelastic cross section. This is due to the following fact. When (in the two-channel case) the inelastic cross section (that corresponds to the transmission) is zero, the whole process takes place via the elastic channel. A transmission zero means therefore that the beam is completely reflected and the coupling strength $\alpha$ between system and continuum does not play any role at this energy. Thus, abrupt jumps by $\pi$ in the phase of the transmission amplitude are associated with the occurrence of transmission zeros and, further, the zeros of the transmission are characteristic of the isolated dot structure. They do not depend on the strength of the coupling to the leads. A similar result is obtained from an analysis of scattering phases in quantum dots that is performed on the basis of lattice models [98, 99].

That means, number and location of the transmission zeros, and consequently also number and location of the phase lapses, are determined exclusively by the distribution of the unperturbed levels $E_{i}^{0}$ (eigenstates of $H^{0}$ ) and the matrix elements $V_{i}^{c}$ that characterize their coupling to the decay channels c. They do not depend on the overall coupling strength $\alpha$ (as long as the system can be described by the Hamiltonian $H^{0}$ ).

However, number and position of the resonance peaks do depend on $\alpha$ due to the resonance trapping phenomenon. In the regime of trapped resonances ( $\alpha > \alpha_{c}$ ), the resonance states are described by a non-Hermitian Hamiltonian $H_{eff}^{tr}$ that differs from $H_{eff}$ : (i) $H^{0}$ has to be replaced by the Hamiltonian $H_{tr}^{0}$ describing the N - 2 trapped states, and the corresponding coupling vectors $V_{tr}$ describing the coupling of the trapped states to the continuum, have to be calculated, (ii) the S matrix has to be supplemented by the background term arising from the two broad states. Neither $H_{tr}^{0}$ nor the corresponding coupling coefficients can be obtained analytically since the spectroscopic redistribution processes in the regime of overlapping resonance states are related to the singular crossing points of eigenvalue trajectories at which resonance trapping (width bifurcation) originates.

The solution of this problem is given by Feshbach [5] without considering the non-Hermitian Hamilton operator $H_{\mathrm{eff}}$ : (i) the eigenstates of $H_{\mathrm{tr}}^{0}$ and the coupling vectors $V_{\mathrm{tr}}$ are described by using statistical approaches (mostly the Gaussian orthogonal ensemble) and (ii) only the direct reaction part of the $S$ matrix is solved exactly (unified theory of nuclear reactions). The states of a Gaussian orthogonal ensemble differ from the states $i$ at low level density, indeed. They are states of a many-body random ensemble (rather than of a two-body random ensemble) and decay according to an algebraic law when the number of decay channels is small [79].

The trapped resonance states are more uniformly distributed than the original resonance states as calculations on the basis of the toy model (62) have shown [62]. Furthermore, the wavefunctions of these states differ less from one another than the eigenfunctions $\Phi_{i}$ of $H_{\mathrm{eff}}$ : in the critical regime where the spectroscopic redistribution processes take place, information entropy in relation to $\Phi_{i}$ is created [70]. As a consequence, also the coupling coefficients of these states to the continuum differ less from one another than those of the eigenstates of $H_{\mathrm{eff}}$ . Hence the zeros of the inelastic cross section and the corresponding phase lapses may show more regularity at large coupling strength $\alpha$ than at small $\alpha$ .

The redistribution of the spectroscopic properties corresponds to a dynamical phase transition taking place in the critical regime at $v \leqslant v_{c}$ , see sections 3.5 and 3.6. Hence, the universal features observed in the phase lapses at high level density, in contrast to the mesoscopic features at low level density, may be considered to be a hint at the dynamical phase transition occurring by controlling the system from low to high level density.

Recently, the problem of phase lapses is studied theoretically by using the numerical and functional renormalization group approaches for maximal N = 4 states [6]. The results for N = 4 resonance states obtained in this approach and those for N = 10 resonance states in the calculations [108] on the basis of (62) and (51) agree, above all, in the fact that the widths of two states are much larger than those of the rest of N - 2 states at large resonance overlapping. This result is, of course, nothing but the resonance trapping phenomenon.

In order to prove experimentally the results obtained from (51) and (62), the phase lapses should be studied as a function of the degree of opening of the quantum dot, i.e. as a function of the overall coupling strength $\alpha$ . Characteristic of such a study is that the number of states of the closed system (described by $H_{B}$ or $H^{0}$ ) and their coupling matrix elements $V_{i}^{c}$ to the continuum are fixed such that the observed results can clearly be related to the phenomenon of resonance trapping. Under this condition, it should be possible to study the influence of the overlapping of the resonance states onto the spectroscopic properties of an open quantum system in a definite manner.

## 4.6. Dephasing

Comparing the basic ingredients of the theory of open quantum systems with the experimental results on dephasing at very low temperature, it should first be stated that the concept dephasing is used differently in the different papers. Within the framework of Landau's theory of Fermi liquids, dephasing is related to the time an electron can travel in the system before losing its phase coherence and thus its wave-like behavior. In open quantum systems, however, the spectroscopic properties of localized states are considered which are described by the non-Hermitian Hamilton operator $H_{\mathrm{eff}}$ . The phases of the eigenfunctions of $H_{\mathrm{eff}}$ are well defined but not rigid, generally (section 2.4).

In the following, a short discussion of the results obtained experimentally on dephasing will be given from the point of view of an open quantum system. The discussion is qualitatively by using the results obtained in different recent studies. It avoids to comment the many controversial discussions that exist in the literature to this question.

In the proceedings of a recent conference, the experimental progress on the saturation problem in metallic quantum wires is reviewed $[109]$ . As a conclusion of this analysis, based on all presently available measurements of the phase coherence time $\tau_{\phi}$ in very clean metallic wires, it is hard to conceive that the apparent saturation of $\tau_{\phi}$ is solely due to the presence of an extremely small amount of magnetic impurities.

The absolute value of $\tau_{\phi}$ (and not just its temperature dependence) is studied in [110]. It is found that the electron dwell time is the central parameter governing the saturation of phase coherence at low temperature. The condition for the occurrence of saturation is found to be $\tau_{\phi}^{sat} \approx \tau_{d}$ , where $\tau_{\phi}^{sat}$ is the saturated coherence time and $\tau_{d}$ is the dwell time. This simple behavior holds over the three orders of magnitude covered by the available data in the literature. According to the authors, $\tau_{\phi}$ is found to be intrinsic to the physics of the quantum dots, but not due to the coherence time of the electrons themselves. Furthermore, it is found [110] that $\tau_{\phi}$ is strongly influenced by the population of the second electronic subband in the quantum well.

According to $[111]$ , one consensus has been reached by several groups, saying that the responsible electron dephasing processes in highly disordered and weakly disordered metals might be dissimilar. That means, while one mechanism is responsible for dephasing in weakly disordered metals, another mechanism may be relevant for the saturation (or very weak temperature dependence) of $\tau_{\phi}$ found in highly disordered alloys. According to the authors of [111], the intriguing electron dephasing is very unlikely due to magnetic scattering. It may originate from specific dynamical structure defects in the samples.

In [112], experimental data from many different publications for $\tau_{\phi}^{0}$ obtained in metallic samples with different diffusion coefficients are collected. The conclusion is that low-temperature saturation of $\tau_{\phi}$ is universally caused by electron–electron interactions. The authors found seemingly contradicting dependencies of $\tau_{\phi}^{0}$ on the diffusion coefficient D in weakly and strongly disordered conductors. While the trend less disorder–less decoherence for sufficiently clean conductors is quite obvious, the opposite trend more disorder–less decoherence in strongly disordered structures is unexpected.

All these statements obtained from the results of many experimental studies fit qualitatively into the expectations received by considering the quantum dot as an open quantum system. First of all, the saturation of $\tau_{\phi}$ appears in a natural manner since most states of an open quantum dot have a finite lifetime at zero temperature. The value of the lifetime can be obtained from the imaginary part of the complex eigenvalue $z_{\lambda}$ of the non-Hermitian Hamilton operator $H_{eff}$ (i.e. from $\mathrm{Im}(z_{\lambda})$ ). It expresses the time the electron stays in the quantum dot. This time is called usually dwell time. Thus, the result obtained in [110] supports the description of the quantum dot as an open quantum system.

Also the more complicated result of different processes in weakly and strongly disordered systems is by no means in contradiction to the properties known for the eigenstates of open quantum systems. In some cases, $\tau_{\phi}^{0}$ depends only weakly on the electron diffusion constant D: it is somewhat smaller when D is larger. That means, states with a large lifetime give only a small contribution to the diffusion—a result which is very well known. In other cases, the relation between $\tau_{\phi}^{0}$ and the diffusion constant D shows the opposite trend. Also in this case the states with a large lifetime give, of course, a small contribution to the diffusion. In contrast to the foregoing case, however, the main contribution to the diffusion arises obviously from short-lived states. This follows from the resonance trapping phenomenon (width bifurcation) characteristic of the regime of overlapping resonances. In this regime, the widths of the short-lived states increase and the widths of the trapped resonance states decrease with increasing degree of overlapping (section 3.4). Finally, the short-lived states form some background for the long-lived resonance states. The diffusion constant is determined mainly by the contribution of the background states. Therefore, the diffusion constant D increases with increasing $\tau_{\phi}^{0}$ of the (long-lived) resonance states—a result being counterintuitive in the same manner as the resonance trapping effect. The last one is directly proven experimentally [66].

In this respect another experimental result obtained in $[1\ 10]$ is interesting. It shows that, in the systems considered, the quantity $\tau_{\phi}$ is strongly influenced by the population of the second electronic subband in the quantum well. Obviously this means that the degree of overlapping of the states plays an important role for the lifetimes of the states—according to one of the basic properties of the eigenstates of $H_{eff}$ (section 3.4). Further experimental studies related to this question would be very useful.

As a result of this discussion, dephasing shows features that might be related to the non-rigidity of the phases of the wavefunctions of an open quantum system in the regime of overlapping resonances. A quantitative description of the experimental data by using the theory of open quantum systems with a non-Hermitian Hamilton operator, is not performed up to now. It is, however, interesting to remark that a decoherence rate $1/\tau_{\phi}$ appears also in the dynamics of a SWAP gate [12]. According to experimental results, this quantity is a non-trivial function of the system–environment interaction rate $\tau_{SE}$ : one has $1/\tau_{\phi} \propto 1/\tau_{SE}$ at low $\tau_{SE}$ (Fermi golden rule) but $1/\tau_{\phi} \propto \tau_{SE}$ at large $\tau_{SE}$ —in (qualitative) agreement with the results discussed above.

## 4.7. Open quantum systems with non-symmetric $H_{eff}$

In the present paper, open quantum systems described by the symmetric non-Hermitian Hamiltonian (4) are considered. Here, the resonance states interact via the common continuum, while the influence of electric or magnetic fields is not taken into account. The geometrical phase of the branch point (crossing point of eigenvalues of the symmetric Hamiltonian (4)) is always different from the Berry phase, also when the coupling strength between system and environment approaches zero. The reason is that the second-order term in (4) is the leading term at the branch (exceptional) point, see section 2.5, while the Berry phase is related to the first-order term of (4).

The opposite case is an antisymmetric non-Hermitian Hamiltonian which considers the influence of electric or magnetic fields onto the open system, however without account of the interaction of the states (Gamow states) via the common continuum. In this case, the second-order term in $(4)$ is not considered. Correspondingly, the geometrical phase of the exceptional point passes, in this case, into the Berry phase in the limit of vanishing coupling to the continuum $[39, 42]$ .

In the realistic case of an open quantum system under the influence of electric or magnetic fields with inclusion of the interaction of its states via the common continuum of scattering wavefunctions, the non-Hermitian Hamiltonian is neither symmetric nor antisymmetric. As an example, we mention here the study on optical microspiral cavities $[113]$ which allow us to obtain a unidirectional light output. The non-orthogonality of the states is significant, in this case, in a broad parameter regime. It causes an extraordinary high quality factor (similar to that known for whispering gallery modes in circular or spherical cavities) such that microspiral cavities are interesting for experimental studies. In this case, the geometrical phases arising from encircling exceptional points are, generally, geometrical and not topological in nature $[43]$ .

Exceptional points are found also in the Bose–Einstein condensation of gases with attractive 1/r interaction [114]. The authors concentrate on the case of self-trapping, i.e. condensation without external trap. In this case, the bifurcations of the two stationary solutions to the nonlinear Gross–Pitaevskii equation at critical physical parameter values, where collapse of the condensates sets in, exhibit the typical structure of exceptional points. According to the authors of [114], there is good reason to believe that the critical parameter values of attractive Bose–Einstein condensates are associated, quite generally, with exceptional points.

These examples show once more that open quantum systems in the neighborhood of branch (exceptional) points have interesting properties that cannot be described in standard quantum mechanics with Hermitian Hamilton operators. These properties may be important for applications. Examples considered in the present paper are, among others, the enhanced and accelerated transmission through quantum dots just below the branch points (section 4.3), quantum-dynamical phase transitions (section 3.5), bound states in the continuum (section 4.4). They are considered, up to now, only without account of contributions from magnetic or electric fields. A study of these effects under the influence of external fields will surely give valuable results for applications as well as for the general understanding of quantum phenomena.

## 5. Summary and outlook

The FPO formalism, together with the S matrix derived by means of it, is a powerful method for the description of the scattering on a system localized in space. The core of the method is the definition of two subspaces one of which is localized (Q subspace, containing discrete states) while the other one is extended up to infinity (P subspace, containing the continuum of scattering states). The most important consequence of this definition is the appearance of the non-Hermitian symmetric Hamilton operator $H_{eff}$ that describes the localized states in the Q subspace under the influence of their coupling to the P subspace. The eigenvalues $z_{\lambda}$ of $H_{eff}$ are complex and provide not only the energies but also the widths of the states $\lambda$ .

The non-Hermitian operator $H_{eff}$ has some specific features that are related to its singularities. At the singular points, two (or more) eigenvalue trajectories of $H_{eff}$ cross. The singular (crossing) points are branch points in the complex plane: approaching the branch points under different conditions, the system is characterized by different physical phenomena. Level repulsion is the characteristic phenomenon when the non-diagonal matrix elements of $H_{eff}$ are almost real and small, while width bifurcation characterizes the system when the complex non-diagonal matrix elements are large, i.e. when the levels strongly interact via the continuum. In the neighborhood of the true crossing points, the eigenvalue trajectories avoid crossing. As a result of width bifurcation, the width of one of the states may even vanish, corresponding to an infinitely long lifetime of this state. This state is called usually bound state in the continuum. That means, the system can be stabilized at the energy of this state by means of external parameters.

The singular points of $H_{eff}$ are interesting not only because of the avoided level crossing phenomenon occurring in their neighborhood. Also the eigenfunctions $\phi_{\lambda}$ of $H_{eff}$ differ from those of a Hermitian operator. While the phases of the last ones are always rigid, those of the $\phi_{\lambda}$ are not rigid, generally. The value of the phase rigidity may vary, $1 \geqslant r_{\lambda} \geqslant 0$ . Approaching a true crossing point of two eigenvalue trajectories, the phase rigidity of the corresponding eigenfunctions approaches zero, meaning that the wavefunctions of the two states become linearly dependent at the crossing point. At the same time, a so-called associated vector appears due to the Jordan chain relations such that there are two different states also at the singular point. The non-rigidity of the phases of the eigenfunctions $\phi_{\lambda}$ of $H_{eff}$ in the regime of overlapping resonances allows the alignment of some of them to the scattering wavefunctions of the environment and the (accompanying) decoupling of other states from the continuum (resonance trapping due to width bifurcation). That means, it allows spectroscopic redistribution processes (dynamical phase transitions) in the system under the influence of its coupling to the environment. These redistribution processes occur due to the feedback from the environment of scattering states onto the system. They can be seen by varying any parameter.

Numerical and analytical studies have shown that the features caused by the non-Hermiticity of $H_{eff}$ survive when the problem $(H - E)\Psi_{c}^{E} = 0$ in the whole function space $(\mathcal{Q} + \mathcal{P} = 1)$ is solved. The physical observables behave smoothly when the singular point is crossed. However, the avoided level crossings of the eigenvalue trajectories in their neighborhood determine the dynamics of open quantum systems. An example are laser-induced continuum structures in atoms. The eigenvalue trajectories behave nonlinear in the neighborhood of the branch points, and the spectrum of the atom may be manipulated by means of varying intensity and (or) frequency of the laser.

Moreover, the non-rigidity of the phases of $\Psi_{c}^{E}$ in the interior of the localized system may cause an enhancement of observable quantities in this regime with respect to the values obtained in the standard quantum mechanics with Hermitian Hamilton operator. An example is the transmission through quantum billiards: the reduction of the phase rigidity $\rho \rightarrow 0$ is correlated with an enhancement of the transmission up to its maximum value (corresponding to full transparency) and with an acceleration of the process (corresponding to a direct process). The most clear examples are whispering gallery modes in cavities with convex boundary. They exist in the regime of overlapping resonances and are very stable. They will be destroyed only at larger coupling strength between system and environment when further spectroscopic redistribution processes reduce finally the number of short-lived states to the number of open decay channels.

Of great interest is the direct proof of the dynamical phase transition in spite of the many hints at their existence. Although clearly observed experimentally in cross-polarization NMR data $[12]$ , these results are not related, up to now, to the non-Hermitian quantum mechanics. The relation of the phase lapses found experimentally in the transmission through quantum dots $[22]$ , to the resonance trapping phenomenon (i.e. to the physics with a non-Hermitian Hamiltonian) is discussed in section 4.5. It has to be proven by further experimental and theoretical studies.

Another advantage of the FPO formalism (in comparison with other approaches) is that the many-body problem is solved at the very beginning in (5) and (9). That means, it is incorporated into the calculations in the same manner as in the standard calculations for discrete states. Formally, the FPO formalism may be considered as a generalization of the R matrix approach $[115]$ . In both cases, the wavefunctions of the system are localized in coordinate space (Q subspace in the FPO formalism) and coupled to an extended continuum of scattering wavefunctions (P subspace in the FPO formalism). However, the standard spectroscopic parameters of the R matrix approach do not contain any feedback from the continuum of scattering wavefunctions. In the FPO formalism, these quantities are replaced by the energy-dependent functions $E_{\lambda}$ and $\Gamma_{\lambda}$ in which the feedback is contained. It should be underlined that all states of the considered system are eigenstates of $H_{eff}$ , including those lying below the first decay threshold (or beyond the transmission window). The widths of these states are zero, however the principal value integral (7) does not vanish, in general.

The properties of open quantum systems originating from the non-Hermitian Hamilton operator $H_{eff}$ are contained, of course, also in calculations for special systems performed on the basis of other models (if these calculations do not contain serious approximations). For example, the effective Green function derived by Datta [8] for the description of electronic transport in mesoscopic systems, is equivalent to $H_{eff}$ . The advantage of the FPO formalism (in comparison with other methods) consists of the fact that it provides not only numerical results for the cross section of the considered system. Most important is that the formalism is very transparent such that the influence of the peculiarities of the eigenvalues and eigenfunctions of the non-Hermitian Hamilton operator $H_{eff}$ onto observable quantities can be seen immediately. The formalism is an adequate tool for treating the physics of open quantum systems. In many cases, it may be useful to combine a special method for the description of the considered system with the study of the corresponding non-Hermitian Hamilton operator.

In spite of much knowledge received recently on the role of non-Hermitian Hamilton operators in open quantum systems, there are still many open questions that have to be solved. One example is the relation between densely lying (avoided and true) crossing points and the threshold for opening a new decay channel as well as the role of external constraints onto the system (e.g. transmission zeros or edge effects). Another example is the relation of trapped states to quantum chaos. Also dynamical phase transitions have to be studied in more detail. All these questions are of fundamental interest and may be important for applications as well.

## Acknowledgments

The present paper is based on the many results obtained in collaboration with Alexander Magunov, Markus Müller, Emil Persson and Almas Sadreev as well as with Evgeny Bulgakov, Stanislaw Drożdź, Carla Figureia de Morisson Faria, Thomas Gorin, Uwe Günther, Christof Jung, Ulrich Kuhl, Rashid Nazmitdinov, Jacek Okołowicz, Konstantin Pichugin, Marek Płoszajczak, Boris Samsonov, Henning Schomerus, Petr Šeba, Varvara Shamshutdinova, H-S

Sim, Hans-Jürgen Stöckmann, Svetlana Strakhova during last about 10 years. I am grateful to all of them for the collaboration and for many valuable discussions. I am also grateful to Horacio Pastawski for discussions on quantum chemistry.

## References

[1] Friedrichs K O 1948 Commun. Pure Appl. Math. 1 361

[3] Löwdin P O 1968 Int. J. Quantum Chem. 2 867

[4] Primas H and Günthard Hs H 1953 Helv. Phys. Acta 36 1659

[5] Feshbach H 1958 Ann. Phys. (NY) 5 357
Feshbach H 1962 Ann. Phys. (NY) 19 287

[6] Karrasch C, Hecht T, Weichselbaum A, Oreg Y, von Delft J and Meden V 2007 Phys. Rev. Lett. 98 186802

[7] Karrasch C, Hecht T, Weichselbaum A, von Delft J, Oreg Y and Meden V 2007 New J. Phys. 9 123

[8] Datta S 1995 Electronic Transport in Mesoscopic Systems (Cambridge: Cambridge University Press) 2005

[9] Pastawski H M and Medina E 2001 Rev. Mex. Fis. 47s1 1 (arXiv:0103219)

[10] Sadreev A F and Rotter I 2003 J. Phys. A: Math. Gen. 36 11413

[11] Volkovich R, Toroker M C and Peskin U 2008 J. Chem. Phys. 129 034501
Toroker M C and Peskin U 2007 J. Chem. Phys. 127 154706
Peskin U, Huang Z and Kais S 2007 Phys. Rev. A 76 012102

[12] Álvarez G A, Danieli E P, Levstein P R and Pastawski H M 2006 J. Chem. Phys. 124 194507

Dente A D, Bustos-Marún R A and Pastawski H M 2008 Phys. Rev. A 78 062116

[13] Peskin U, Reisler H and Miller W H 1994 J. Chem. Phys. 101 9672

[14] Moiseyev N 1998 Phys. Rep. C 302 211

[15] Sajeev Y and Moiseyev N 2008 Phys. Rev. B 78 075316 Fleischer A and Moiseyev N 2008 Phys. Rev. A 77 010102 Sajeev Y, Sindelka M and Moiseyev N 2008 J. Chem Phys. 1

[16] Okolowicz J, Ploszajczak M and Rotter I 2003 Phys. Rep. 374 271

[17] Volya A and Zelevinsky V G 2006 Phys. Rev. C 74 064314

[18] Magunov A I, Rotter I and Strakhova S I 1999 J. Phys. B: At. Mol. Opt. Phys. 32 1489

[19] Rotter I 1991 Rep. Prog. Phys. 54 635

[20] Nazmitdinov R G, Sim H S, Schomerus H and Rotter I 2002 Phys. Rev. B 66 241302

[21] Barz H W, Rotter I and Höhn J 1977 Nucl. Phys. A 275 111

[22] Avinun-Kalish M, Heiblum M, Zarchin O, Mahalu D and Umansky V 2005 Nature 436 529

[23] Figueira de Morisson Faria C and Rotter I 2002 Phys. Rev. A 66 013402

[24] Kato T 1966 Peturbation Theory for Linear Operators (Berlin: Springer)

[25] Günther U, Rotter I and Samsonov B F 2007 J. Phys. A: Math. Theor. 40 8815 (arXiv:0704.1291)

[26] Okolowicz J, Ploszajczak M and Rotter I 2003 Phys. Rep. 374 271 (appendix A)

[27] Rotter I and Sadreev A F 2005 Phys. Rev. E 71 036227

[28] Rotter I and Sadreev A F 2004 Phys. Rev. E 69 066201

[29] Rotter I 2001 Phys. Rev. E 64 036213

[30] Magunov A I, Rotter I and Strakhova S I 2001 J. Phys. B: At. Mol. Opt. Phys. 34 29

[31] Heiss W D, Müller M and Rotter I 1998 Phys. Rev. E 58 2894

[32] Dembowski C, Gräf H-D, Harney H L, Heine A, Heiss W D, Rehfeld H and Richter A 2001 Phys. Rev. Lett. 86 787

[33] Dembowski C, Dietz B, Gräf H D, Harney H L, Heine A, Heiss W D and Richter A 2003 Phys. Rev. Lett. 90 034101

[34] Solov'ev E A 1981 Sov. Phys. JETP 54 893

[35] Newton R G 1982 Scattering Theory of Waves and Particles (New York: Springer)

[36] Magunov A I, Rotter I and Strakhova S I 1999 J. Phys. B: At. Mol. Opt. Phys. 32 1669

[58] Pavlov-Verevkin V B 1988 Phys. Lett. A 129 168

[59] Sokolov V V and Zelevinsky V G 1988 Phys. Lett. B 202 10

[37] Rotter I 2007 J. Phys. A: Math. Theor. 40 14515

[38] Bulgakov E N, Rotter I and Sadreev A F 2006 Phys. Rev. E 74 056204

[40] Cartarius H, Main J and Wunner G 2007 Phys. Rev. Lett. 99 173003

[41] Keck F, Korsch H J and Mossmann S 2003 J. Phys. A: Math. Gen. 36 2125

[42] Nesterov A I and Ovchinnikov S G 2008 Phys. Rev. E 78 015202
Nesterov A I and Aceves de la Cruz F 2008 arXiv:0806.3720

[43] Mehri-Dehnavi H and Mostafazadeh A 2008 J. Math. Phys. 49 082105

[44] Mostafazadeh A 2009 private communication, see equation (32) in [43]

[45] Müller M and Rotter I 2008 J. Phys. A: Math. Theor. 41 244018

[46] Lauber H M, Weidenhammer P and Dubbers D 1994 Phys. Rev. Lett. 72 1004

[47] Persson E, Pichugin K, Rotter I and Seba P 1998 Phys. Rev. E 58 8001

[48] Rotter I 2003 Phys. Rev. E 68 016211

[49] Rotter I 1981 Ann. Phys. (Lpz.) 493 221 (earlier vol 38)

[50] Shamshutdinova V V, Pichugin K N, Rotter I and Samsonov B F 2008 Phys. Rev. A 78 062712

[51] Rotter I, Barz H W and Höhn J 1978 Nucl. Phys. A 297 237

[52] Malmberg P R 1956 Phys. Rev. 101 114

[52] Maimborg P K 1930 Phys. Rev. 101 114 Brown L, Steiner E, Arnold L G and Seyler R G 1973 Nucl. Phys. A 206 353

[53] Rothe C, Hintschich S I and Monkman A P 2006 Phys. Rev. Lett. 96 163601

[54] Persson E, Müller M and Rotter I 1996 Phys. Rev. C 53 3002

[55] Kuhl U 2007 Eur. Phys. J. Special Topics 145 103

[56] Brouwer P W 2003 Phys. Rev. E 68 046205

[57] Kleinwächter P and Rotter I 1985 Phys. Rev. C 32 1742

Brems V, Desouter-Lecomte M and Lievin J 1996 J. Chem. Phys. 104 2222

Sokolov V V and Zelevinsky V G 1989 Nucl. Phys. A 504 562
Sokolov V V and Zelevinsky V G 1992 Ann. Phys. (NY) 216 323

[60] Dittes F M, Cassing W and Rotter I 1990 Z. Phys. A 337 243

[61] Dittes F M, Harney H L and Rotter I 1991 Phys. Lett. A 153 451

[62] Dittes F M, Rotter I and Seligman T H 1991 Phys. Lett. A 158 14

[63] Iskra W, Rotter I and Dittes F M 1993 Phys. Rev. C 47 1086

[64] Müller M, Dittes F M, Iskra W and Rotter I 1995 Phys. Rev. E 52 5961

[65] Jung C, Müller M and Rotter I 1999 Phys. Rev. E 60 114

[66] Persson E, Rotter I, Stöckmann H J and Barth M 2000 Phys. Rev. Lett. 85 2478

[67] Seligman T H and Nishioka H (ed) 1986 Quantum Chaos and Statistical Nuclear Physics Proceedings,

[68] Haq R U, Pandey A and Bohigas O 1982 Phys. Rev. Lett. 48 1086

[69] Nazmitdinov R G, Pichugin K N, Rotter I and Seba P 2001 Phys. Rev. E 64 056214
Nazmitdinov R G, Pichugin K N, Rotter I and Seba P 2002 Phys. Rev. B 66 085322
Bulgakov E N, Gopar V A, Mello P A and Rotter I 2006 Phys. Rev. B 73 155302
Bulgakov E N and Rotter I 2006 Phys. Rev. E 73 066222

[70] Iskra W, Müller M and Rotter I 1993 J. Phys. G: Nucl. Part. Phys. 19 2045
Iskra W, Müller M and Rotter I 1994 J. Phys. G: Nucl. Part. Phys. 20 775
Iskra W, Müller M and Rotter I 1994 Prog. Theor. Phys. Suppl. 116 385

[71] Bender C M, Brody D C, Jones H F and Meister B K 2007 Phys. Rev. Lett. 98 040403

[72] Assis P E G and Fring A 2008 J. Phys. A: Math. Theor. 41 244002

[73] Günther U and Samsonov B F 2008 Phys. Rev. Lett. 101 230404
Günther U and Samsonov B F 2008 Phys. Rev. A 78 042115

[74] Briggs J S and Rost J M 2000 Eur. Phys. J. D 10 311

[75] Persson E, Gorin T and Rotter I 1996 Phys. Rev. E 54 3339

[76] Peskin U, Reisler H and Miller W H 1997 J. Chem. Phys. 106 4812

[77] Rotter I 1997 J. Chem. Phys. 106 4810

[78] Kanter E P, Kollewe D, Komaki K, Leuca I, Temmer G M and Gibson W M 1978 Nucl. Phys. A 299 230

[79] Dittes F M, Harney H L and Müller A 1992 Phys. Rev. A 45 701

[80] Magunov A I, Rotter I and Strakhova S I 2003 Phys. Rev. B 68 245305

[81] Fano U 1961 Phys. Rev. 124 1866

[82] Kobayashi K, Aikawa H, Katsumoto S and Iye Y 2002 Phys. Rev. Lett. 88 256806 Kobayashi K, Aikawa H, Katsumoto S and Iye Y 2003 Phys. Rev. B 68 235304

[83] Pnini R and Shapiro B 1996 Phys. Rev. E 54 R1032

[84] Bulgakov E N, Rotter I and Sadreev A F 2007 Phys. Rev. B 76 214302

[85] von Neumann J and Wigner E 1929 Phys. Z. 30 465

[86] Friedrich H and Wintgen D 1985 Phys. Rev. A 31 3964
Friedrich H and Wintgen D 1985 Phys. Rev. A 32 3231
Friedrich H 1990 Theoretical Atomic Physics (Berlin: Springer)

[87] von Neumann J and Wigner E 1929 Phys. Z. 30 467

[88] Knight P L, Lauder M A and Dalton B J 1990 Phys. Rep. 190 1 Kylstra N J and Joachain C J 1998 Phys. Rev. A 57 412

[89] Rotter I and Sadreev A F 2005 Phys. Rev. E 71 046204

[91] Bulgakov E N, Rotter I and Sadreev A F 2007 Phys. Rev. A 75 067401

[92] Bender C M and Boettcher S 1998 Phys. Rev. Lett. 80 5243
Bender C M, Brody D C and Jones H F 2002 Phys. Rev. Lett. 89 270401

[93] Fring A, Jones H and Znojil M (ed) 2008 J. Phys. A: Math. Theor. 41

[94] Nakamura H, Hatano N, Garmon S and Petrosky T 2007 Phys. Rev. Lett. 99 210404

[95] Marinica D C, Borisov A G and Shabanov S V 2008 Phys. Rev. Lett. 100 183902

[96] Yacoby A, Heiblum M, Mahalu D and Shtrikman H 1995 Phys. Rev. Lett. 74 4047

[97] Schuster R, Buks E, Heiblum M, Mahalu D, Umansky V and Shtrikman H 1997 Nature 385 417

[98] D'Amato J L, Pastawski H M and Weisz J F 1989 Phys. Rev. 39 3554 Foa Torres L E F, Pastawski H M and Medina E 2006 Europhys. Lett. 73 164

[99] Levy Yeyati A and Büttiker M 2000 Phys. Rev. B 62 7307

[100] Oreg Y and Gefen Y 1997 Phys. Rev. B 55 13726
Silva A, Oreg Y and Gefen Y 2002 Phys. Rev. B 66 195316
König J and Gefen Y 2005 Phys. Rev. B 71 201308
Sindel M, Silva A, Oreg Y and von Delft J 2005 Phys. Rev. B 72 125316
Golosov D I and Gefen Y 2006 Phys. Rev. B 74 205316
Golosov D I and Gefen Y 2007 New J. Phys. 9 120

[101] Silvestrov P G and Imry Y 2000 Phys. Rev. Lett. 85 2565
Silvestrov P G and Imry Y 2001 Phys. Rev. B 65 035309
Silvestrov P G and Imry Y 2007 New J. Phys. 9 125

[102] Hackenbroich G 2001 Phys. Rep. 343 463

[103] Meden V and Marquardt F 2006 Phys. Rev. Lett. 96 146801

[104] Berkovits R, von Oppen F and Kantelhardt J W 2004 Eur. Phys. Lett. 68 699 Goldstein M and Berkovits R 2007 New J. Phys. 9 118

[105] Kashcheyevs V, Schiller A, Aharony A and Entin-Wohlman O 2007 Phys. Rev. B 75 115313

[106] Oreg Y 2007 New J. Phys. 9 122

[107] Gurvitz S A 2008 Phys. Rev. B 77 201302

[108] Müller M and Rotter I 2009 at press

[109] Saminadayar L, Mohanty P, Webb R A, Degiovanni P and Bäuerle C 2007 Physica 40 12

[110] Hackens B, Faniel S, Gustin C, Wallart X, Bollaert S, Cappy A and Bayot V 2005 Phys. Rev. Lett. 94 146802

[111] Lin J J, Lee T C and Wang S W 2007 Physica 40 25

[112] Golubev D S and Zaikin A D 2007 Physica 40 32

[113] Wiersig J, Kim S W and Hentschel M 2008 Phys. Rev. A 78 053809

[114] Cartarius H, Main J and Wunner G 2008 Phys. Rev. A 77 013618

[115] Rotter I 2004 Acta Phys. Pol. B 35 1269
