# Parity-time symmetry and exceptional points in photonics

Ş. K. Özdemir $^{1*}$ , S. Rotter $^{2*}$ , F. Nori $^{3,4}$ and L. Yang $^{5}$

Exploiting the interplay between gain, loss and the coupling strength between different optical components creates a variety of new opportunities in photonics to generate, control and transmit light. Inspired by the discovery of real eigenfrequencies for non-Hermitian Hamiltonians obeying parity-time (PT) symmetry, many counterintuitive aspects are being explored, particularly close to the associated degeneracies also known as 'exceptional points'. This Review explains the underlying physical principles and discusses the progress in the experimental investigation of PT-symmetric photonic systems. We highlight the role of PT symmetry and non-Hermitian dynamics for synthesizing and controlling the flow of light in optical structures and provide a roadmap for future studies and potential applications.

Symmetries are essential for understanding and describing the physical world $^{1}$ ; they give rise to the conservation laws of physics, lead to degeneracies, control the structure of matter, and dictate interactions. Symmetries require the laws of physics to be invariant under changes of redundant degrees of freedom; equivalently, one expects the physics to be equivalent for a particle and its antiparticle (charge conjugation; C symmetry), for a system and its mirror image (parity; P symmetry); and even when time is running backwards (time reversal; T symmetry). Symmetries are perceived as the 'key to nature's secret' $^{2}$ ; however, it is the symmetry breaking that creates nontrivial physics by lifting the degeneracies.

Systems exhibiting PT symmetry are in general non-Hermitian systems, which are invariant under the combined action of the P and T operations. PT symmetry has its roots in quantum field theory and opens a new perspective in studying non-Hermitian Hamiltonians (Box 1). While it is still being debated whether PT symmetry is a fundamental property of nature, it shares common features with naturally occurring symmetries: it can exhibit spontaneous symmetry breaking accompanied by a real-to-complex spectral phase transition, or, conversely, it can undergo a phase transition that restores the broken symmetry. The spectral degeneracies induced by PT symmetry are, however, very different from conventional ones: they are non-Hermitian and known as exceptional points (EPs) $^{3}$ where the real parts and the corresponding imaginary parts of certain eigenvalues coincide, as well as their associated eigenvectors. In essence, the system behaves as if it loses its dimensionality near an EP because the vector space becomes severely skewed. This is strikingly different from the spectral degeneracies of Hermitian systems—so-called diabolic points (DPs) $^{4}$ —where only the eigenvalues coalesce, but the eigenvectors can be chosen to be orthogonal to each other (Box 1). DPs are associated with a geometric Berry phase $^{5}$ , they play an essential role in the understanding of reaction mechanisms in chemistry $^{4,6}$ (where DPs are known as conical intersections), and are exploited in optical sensing methods using whispering gallery mode (WGM) resonators $^{7,8}$ , where they appear as ‘mode splittings’.

Since 1998 when Bender and Boettcher showed that a wide class of non-Hermitian Hamiltonians can exhibit real spectra if they commute with the PT operator $^{9}$ , PT symmetry has become an active research area, because it can provide new design strategies for devices with novel functionalities (Fig. 1a). The realization of PT-symmetric optical structures from the combination of symmetric index guiding with an antisymmetric gain/loss profile $^{10,11}$ , as in a coupled two-component system (Fig. 1b), attracted considerable attention, beyond the initially interested community of mathematical physicists. The progress in understanding and creating PT-symmetric photonic structures has resulted in the demonstration of many nontrivial effects such as power oscillations $^{11-13}$ , loss-induced transparency $^{14}$ and lasing $^{15,16}$ , non-reciprocal light propagation $^{17-19}$ , unidirectional invisibility $^{20-22}$ , PT-symmetric lasers $^{23,24}$ and laser-absorbers $^{25-27}$ , chiral modes and on-demand directional emission $^{28-30}$ , and orbital angular momentum lasers $^{31}$ . In parallel, there has been a growing excitement about exploring PT symmetry in atomic and quantum systems $^{32-34}$ , in optomechanics $^{35-40}$ , acoustics $^{41-43}$ , electronics $^{44}$ , plasmonics $^{45,46}$ , metamaterials $^{47-51}$ and photonic crystals $^{52-54}$ .

In this Review, we focus on PT symmetry and the degeneracies that it gives rise to, with a particular emphasis on implementations in optics. We will review key concepts and experiments that have probed fundamental properties of PT symmetry and demonstrated the ability of PT-symmetric structures to control and manipulate light transport and point out future challenges. PT-symmetric nonlinear optics (not covered here) has been discussed already elsewhere $^{55,56}$ .

## PT symmetry in optics

One of the challenges in photonics is to develop on-chip optical devices that enable better control for information processing and transmission. Artificial materials with functionalities beyond those of natural materials can offer control over light propagation and optical dynamics through engineering the signs and profiles of the permittivity $\varepsilon(r,t)=n^{2}(r,t)$ and the permeability $\mu(r,t)$ (Fig. 1a). PT-symmetric optics is mostly concerned with the refractive index

## Box 1 | PT-symmetric Hamiltonians

In quantum mechanics, the Hamiltonian $H$ is assumed to be Hermitian, $H = H^{\dagger}$ , where the superscript $\dagger$ denotes Hermitian conjugation (that is, transposition plus complex conjugation). This ensures real energy eigenvalues and correspondingly a unitary time evolution for which the probability to find this particle somewhere is conserved. Systems with gain or dissipation, on the other hand, are described by non-Hermitian Hamiltonians—that is, $H \neq H^{\dagger}$ for which the probability is in general not conserved and time-evolution is not unitary. What Bender and Boettcher showed, however, was that Hermiticity is not a necessary condition for real eigenvalues of $H$ , and that a whole class of non-Hermitian Hamiltonians (featuring both gain and loss) can have real eigenvalues without being Hermitian $^{9}$ . This, not necessarily exclusive, class of Hamiltonians has the property of being PT-symmetric in the sense that they commute with the $PT$ operator—that is, $[PT, H] = PTH - HPT = 0$ where $P$ and $T$ operators satisfy $P^{2} = 1$ , $P = P^{\dagger}$ , $T^{2} = 1$ , $T = T^{\dagger}$ and $[P, T] = 0$ . The action of the operators $P$ and $T$ are, respectively, defined as $P: i \to i$ , $\hat{x} \to -\hat{x}$ , $\hat{p} \to -\hat{p}$ , and $T: i \to -i$ , $\hat{x} \to \hat{x}$ , $\hat{p} \to -\hat{p}$ with $\hat{p}$ and $\hat{x}$ satisfying the commutator $[x, p] = xp - px = ih$ . The complex conjugation $(i \to -i)$ when the $T$ operator is applied implies that $T$ is an anti-linear operator. (An antilinear operator K satisfies $\mathrm{K}[c_1\psi_1(x) + c_2\psi_2(x)] = c_1^*\mathrm{K}\psi_1(x) + c_2^*\mathrm{K}\psi_2(x)$ , where \* denotes complex conjugation.) Applying the $PT$ operator to the above commutator $PT[x, p]TP = -ih$ shows that the PT operator is an antilinear operator too.

The $2 \times 2$ Hamiltonian

$$
H = \left[ \begin{array}{c c} {{\omega_ {1} - i \gamma_ {1}}} & {{\kappa}} \\ {{\kappa}} & {{\omega_ {2} - i \gamma_ {2}}} \end{array} \right] \begin{array}{l} {{\mathrm{非厄密系统的} H \mathrm{矩阵}}} \end{array}
$$

describes a non-Hermitian $(H\neq H^{\dagger})$ two-component system with complex frequencies $\omega_{1}-i\gamma_{1}$ and $\omega_{2}-i\gamma_{2}$ . Here $\gamma_{1},\gamma_{2}$ are the corresponding loss/gain coefficients and $\kappa$ is the coupling strength between the components of the system. The eigenvalues of this Hamiltonian are $\omega_{\pm}=\omega_{0}-i\chi\pm\sqrt{\kappa^{2}+\Gamma^{2}}$ where $\omega_{0}=(\omega_{1}+\omega_{2})/2$ , $\chi=(\gamma_{1}+\gamma_{2})/2$ and $\Gamma=\delta+i\beta$ with $\delta=(\omega_{1}-\omega_{2})/2$ and $\beta=(\gamma_{1}-\gamma_{2})/2$ . When varying two of the available parameters, the two eigenvalues can be forced to coincide at a specific point called an 'exceptional point' (EP) where $\kappa^{2}+\Gamma^{2}=0$ , implying the solutions $(\delta=0,\kappa=\mp\beta)$ or $(\beta=0,\kappa=\mp i\delta)$ . At an EP not only the eigenvalues coincide in their real and imaginary parts, but also the eigenvectors become completely parallel, such that the Hamiltonian is defective and not diagonalizable.

The necessary conditions for the two eigenvalues of the Hamiltonian H to be real are found as: (i) $\chi=0$ , which leads to $\gamma_{1}=-\gamma_{2}$ , implying that while one of the components of the system is dissipating, the other component is amplifying; and (ii) $\kappa^{2}+\Gamma^{2}\geq0$ , which can be satisfied if $\delta=0$ , implying $\omega_{1}=\omega_{2}$ . Under the necessary conditions of $\gamma_{1}=-\gamma_{2}=\gamma$ and for the case of $\omega_{1}=\omega_{2}=\omega'$ , the Hamiltonian of the two-component system becomes

特征值为实即存在

PT对称性，非厄密

系统中可以构造

特征值，从而保留

loss=gain来构造实

$$
H ^ {\prime} = \left[ \begin{array}{c c} \omega^ {\prime} - i \gamma & \kappa \\ \kappa & \omega^ {\prime} + i \gamma \end{array} \right]
$$

$n(r, t)=n_{\mathrm{r}}(r, t)+in_{\mathrm{i}}(r, t)$ and the interplay between its real part $n_{p}$ , which determines the phase velocity, and its imaginary part $n_{i}$ , which determines attenuation (loss; $n_{i}>0$ ) or amplification (gain; $n_{i}<0$ ). The crucial question is how to design $n_{r}$ and $n_{i}$ to achieve functionalities beyond optical amplification or damping alone.

with the eigenvalues given as $\omega_{\pm}^{\prime}=\omega^{\prime}\pm\sqrt{\kappa^{2}-\gamma^{2}}$ . Thus, for this system to have real eigenvalues one also has to satisfy $\kappa\geq\gamma$ . The Hamiltonian $H^{\prime}$ is PT-symmetric because it stays invariant after the action of the PT operator PT $H^{\prime}=H^{\prime}$ PT. Since the T operator is anti-linear a PT-symmetric Hamiltonian does not necessarily have eigenvectors that inherit this symmetry; but if they do, then the PT symmetry of the eigenvectors entails that these states experience an equal amount of gain and loss, such that the corresponding eigenvalues are real. This PT-symmetric phase of the eigenvectors occurs for $\kappa/\gamma>1$ and gets broken at the EP (spontaneous PT-symmetry breaking), which is located right at the point $\kappa/\gamma=1$ , where both the eigenstates and eigenvectors become degenerate. The parameter restriction imposed by PT symmetry implies that the variation of a single parameter is sufficient to reach the EP, where an abrupt phase transition from real-to-complex spectra occurs.

Plotting the full dependence of both eigenvalues of the non-Hermitian Hamiltonian on $\kappa$ and $\delta$ yields two intersecting Riemann sheets wrapped around the EP right in the centre (panel a of the figure below). The interesting topology of this plot also nicely illustrates that the EP is, in fact, a branch point singularity $^{3}$ as known from the complex analysis of multivalued functions. For a Hermitian Hamiltonian with loss/gain parameters $\gamma_{1}, \gamma_{2}=0$ , the resultant eigenvalues $\omega_{\pm}^{\prime\prime} = \omega_{0} \pm \sqrt{\kappa^{2} + \delta^{2}}$ form a double cone topology as a function of $\kappa$ , $\delta$ such that the cones touch each other at their apex ( $\kappa=\delta=0$ ), creating a 'diabolic point' (DB) (panel b). A DP differs from an EP in the sense that in the latter both the eigenfrequencies (energy levels) and the corresponding eigenstates coalesce, whereas in the former only the eigenfrequencies coalesce but the eigenstates may be chosen such as to be orthogonal to each other.

![](images/c70d7217f3a17f6a8e9b761bf56d5292069010a0049854862759497d4762d3f8.jpg)
Topology of non-Hermitian and Hermitian degeneracies. a, Varying the coupling strength $\kappa$ and the frequency detuning $\delta$ of two coupled gain-loss elements reveals a complex eigenvalue topology of two intersecting Riemann sheets centred around an exceptional point (EP). The black trajectories show the adiabatic evolution of eigenstates when encircling the EP, and the white lines indicate the square-root parameter dependence right at the EP. b, In a Hermitian system EPs do not occur and conventional degeneracies known as diabolic points (DP) have a double-cone topology around them.

Early works $^{57-59}$ on the simultaneous presence of loss and gain in optical systems did not mention PT symmetry or its breaking; however, their findings can be associated with the implications of PT symmetry. The starting point for the use of optical structures to simulate PT symmetry was the similarity between the scalar paraxial optical wave equation and the time-dependent Schrödinger equation $^{10,11}$ (Box 2). While being of fundamental interest initially, this similarity inspired many PT experiments and follow-up studies in which gain and loss are treated on an equal footing, controlled by the coupling between them (Fig. 1). The most basic features resulting from this interplay can already be seen in the example of two coupled discs (Fig. 1c): in the strong coupling regime $(\kappa/\kappa_{\mathrm{PT}}>1)$ , gain can help to fully compensate the loss, and the system operates in the PT-symmetric phase (exact phase) with real eigenvalues; in the weak coupling regime $(\kappa/\kappa_{\mathrm{PT}}<1)$ , on the other hand, gain is unable to compensate the loss, and thus the system operates in the regime of broken-PT symmetry (broken phase) with complex conjugate eigenvalues. In between these regimes, a real-to-complex spectral phase transition takes place at an EP $(\kappa/\kappa_{\mathrm{PT}}=1)$ . The crucial aspect here is that having a sub-system with gain equal to the amount of loss of a second sub-system is not sufficient to have real eigenvalues: a proper amount of coupling between them is also needed. PT-symmetric systems are open because they have loss and gain, thus in the broken-PT phase they are non-equilibrium systems with exponential amplification and dissipation (that is, complex conjugate eigenvalue pairs), but at the same time they behave like closed systems in the exact PT phase because they have a real energy spectrum (although the power is not conserved during the time evolution $^{11}$ ).

![](images/1d648eee66016b96f3837c729ff69b5af7ed71af7bf96403c461431ed70190f4.jpg)

![](images/05b40bc0230973857033019cbf5fbb42374a9f583e563e89e775ceba816fe56c.jpg)

![](images/b2141a2e46aef904e986c1639df4085e67412ece29540cbd52bbb43f5e6e4215.jpg)

![](images/1f08cfd0740952c193f95f10520119b143666cd74f1d40389a382917a6698ad7.jpg)
Fig. 1 | PT symmetry. a, Controlling the magnetic permeability ( $\mu$ ) and the real part of the dielectric permittivity ( $\varepsilon_{r}$ ) has enabled novel functionalities. PT symmetry and non-Hermitian photonics open new possibilities by controlling the imaginary part of the dielectric permittivity ( $\varepsilon_{i}$ ), and by considering gain and loss on equal footing. b, A system of two coupled structures (with coupling strength $\kappa$ ), one with passive loss (L) and one with gain (G) compensating the loss of the other, forms a PT-symmetric system. The parity operator (P) exchanges the structures spatially, and with the action of the time-reversal operator (T) loss becomes gain and gain becomes loss. The combined action of PT leaves the system unchanged. c, Real and imaginary parts of eigenvalues versus coupling strength $\kappa$ in a PT-symmetric system with gain-loss balance. Red circles and blue squares are obtained by numerical simulation in COMSOL assuming that the system is composed of resonators. $\kappa$ is normalized to the critical coupling strength $\kappa_{PT}$ at which the PT phase transition takes place (EP). Insets feature the intracavity field intensities of the resonators in the PT-symmetric regime ( $\kappa/\kappa_{PT}>1$ ) and in the broken-PT regime ( $\kappa/\kappa_{PT}<1$ ). In the broken-PT regime, one of the supermodes experiences amplification while the other one attenuation.

## Passive PT-symmetry breaking

In the first experimental realizations, the stringent requirement of loss–gain balance was circumvented by generalizing the definition of PT symmetry to include loss-only structures $^{14,60}$ (Box 3). Such systems are identified as passive PT-symmetric systems to emphasize that gain is not involved.

The ability of PT-symmetric systems to confine and guide light is one of their major strengths first highlighted in a passive PT-symmetric experiment performed using coupled waveguides $^{14}$ (Fig. 2a) whose loss-imbalance was tuned by introducing additional loss to one of the waveguides. The concept was subsequently extended to coupled microwave $^{61}$ and optical $^{16}$ (Fig. 3a) resonators. These experiments with two-component systems revealed that when the additional loss was below a threshold value, the eigenmodes of the system were distributed evenly between the components, and experienced similar losses. As a result, the total transmitted power through the waveguides (Fig. 2a) (similarly the total field inside the resonators) decreased with increasing loss. When the loss was increased to the symmetry-breaking threshold and beyond it, the system transited from the strong to the weak coupling regime. Consequently, the modes, which were initially evenly distributed between the components, were rearranged to an asymmetric distribution, with one mode progressively localized in the more lossy component and the other in the less lossy component. As a result, the total transmission through the waveguides increased (similarly the total field in the resonators increased) despite the increased loss. This ‘loss-induced transparency’ is counterintuitive from the conventional view that increasing loss should decrease the transmitted power.

Despite the absence of spatially symmetric gain and loss regions, these experiments are considered to be the first realizations of PT

## Box 2 | PT-symmetric optical potentials

A basic approach to derive a potential that satisfies PT symmetry is to start with the paraxial equation of diffraction used in optics $^{10,11}$

$$
i \frac {\mathrm{d} E (x , z)}{\mathrm{d} z} + \frac {1}{2 k} \frac {\mathrm{d} ^ {2} E (x , z)}{\mathrm{d} x ^ {2}} + k _ {0} [ n _ {r} (x) + i n _ {i} (x) ] E (x, z) = 0
$$

where $E(x,z)$ is the electric field envelope, $n(x)=n_{\mathrm{r}}(x)+in_{i}(x)$ is the complex refractive index distribution decomposed in its real $n_{r}(x)$ and imaginary $n_{i}(x)$ parts, $k=k_{0}n_{0}$ is the wavevector, $k_{0}=2\pi/\lambda$ with $\lambda$ being the wavelength of the field in vacuum and $n_{0}$ the substrate index. This widely used equation is mathematically isomorphic to the Schrödinger equation

$$
i \hbar \frac {\mathrm{d} \psi (x , t)}{\mathrm{d} t} + \frac {\hbar^ {2}}{2 m} \frac {\mathrm{d} ^ {2} \psi (x , t)}{\mathrm{d} x ^ {2}} - V (x) \psi (x, t) = 0
$$

Here, $\psi(x, t)$ is the probability amplitude, $\hbar$ is Planck's constant, $m$ is the mass and $V(x)$ is the complex potential. The paraxial equation of diffraction describes the spatial propagation of a light beam whose amplitude varies very little in the direction of propagation over a distance comparable to its wavelength (that is, all of the light travels nearly parallel to the direction of propagation and the spreading of the wave in transverse direction is very small). The Schrödinger equation, on the other hand, describes the temporal evolution of a quantum particle in a potential. Nonetheless, if the parameters $E(x, z)$ , $z$ , $k$ and $k_0 n(x)$ of the paraxial equation are replaced by $\psi(x, t)$ , $t$ , $m/\hbar$ and $-V(x)/\hbar$ , we obtain the Schrödinger equation with the $z$ coordinate in the paraxial equation playing the role of time $t$ in the Schrödinger equation. The refractive index distribution $n(x)$ translates to the complex potential $V(x)$ , and is thus considered as the optical potential.

We first apply the PT operator to the Schrödinger equation to find the conditions on the complex potential $V(x)$ to be a PT-symmetric potential. The isomorphism suggests that the same condition then can be imposed on $n(x)$ such that it is also PT-symmetric. Applying the T operator performs $i \to -i$ , $\hat{x} \to \hat{x}$ and $\hat{p} \to -\hat{p}$ , yielding

symmetry in optics. The transitions from a spatially symmetric to an asymmetric state at EPs constitute the PT phase transitions. Different from the waveguide experiment $^{14}$ , an optical resonator-based design $^{16}$ allowed the variation of both the loss and the interresonator coupling strength. Thereby, the eigenfrequency surfaces that exhibited a complex square-root-function topology, and a continuous thread of EPs along an ‘exceptional line’ were conveniently obtained. Moreover, it enabled the investigation of nonlinear processes and Raman lasing $^{16}$ in the vicinity of an EP where field localization enhances the intracavity field intensity.

The relation between purely dissipative and balanced gain-loss systems has subsequently spurred other investigations with loss-only optical structures on wave transport in non-Hermitian systems $^{62}$ , achieving effects like the one-sided absence of reflection or unidirectional invisibility $^{63,64}$ . Moreover, going beyond two waveguides, a passive PT-symmetric lattice fabricated with direct-laser writing was used to demonstrate that wave transport in a non-Hermitian system can undergo a sudden transition from ballistic to diffusive transport even in the absence of disorder, and that this transition depends only on the degree of dissipation (or the Hermiticity) rather than on the coupling strength between the

$$
\stackrel {T} {\rightarrow} i \hbar \frac {\mathrm{d} \psi^ {\star} (x , - t)}{\mathrm{d} t} + \frac {\hbar^ {2}}{2 m} \frac {\mathrm{d} ^ {2} \psi^ {\star} (x , - t)}{\mathrm{d} x ^ {2}} - V ^ {\star} (x) \psi^ {\star} (x, - t) = 0
$$

where the superscript $^{*}$ denotes complex conjugation and we replaced $t\to-t$ to have the same sign in front of the first term. The P operator, which performs $i\to i$ , $\hat{x}\to-\hat{x}$ and $\hat{p}\to-\hat{p}$ , transforms the above expression as

$$
P: x - > - x
$$

$$
\xrightarrow {P T} i \hbar \frac {\mathrm{d} \psi^ {\star} (- x , - t)}{\mathrm{d} t} + \frac {\hbar^ {2}}{2 m} \frac {\mathrm{d} ^ {2} \psi^ {\star} (- x , - t)}{\mathrm{d} x ^ {2}} - V ^ {\star} (- x) \psi^ {\star} (- x, - t) = 0
$$

We may thus conclude that if $\psi(x,t)$ satisfies the Schrödinger equation so does $\psi^{*}(-x,-t)$ provided that $V(x)=V^{*}(-x)$ . Rewriting the complex potential as $V(x)=V_{\mathrm{r}}(x)+iV_{\mathrm{i}}(x)$ and $V^{*}(-x)=V_{\mathrm{r}}(-x)-iV_{\mathrm{i}}(-x)$ , respectively, where subscripts $V_{r}$ and $V_{i}$ denote real and imaginary parts, it becomes clear that the PT potential should satisfy $V_{\mathrm{r}}(x)=V_{\mathrm{r}}(-x)$ and $V_{\mathrm{i}}(x)=-V_{\mathrm{i}}(-x)$ . This implies that the complex PT potential should have a real part, which is an even function of x, and an imaginary part, which is an odd function of x. Using the isomorphism between the paraxial equation of diffraction and the Schrödinger equation, we thus conclude that for the optical potential $n(x)=n_{\mathrm{r}}(x)+in_{\mathrm{i}}(x)$ to be PT-symmetric, its real part should be an even function of x and its imaginary part should be an odd function of x—that is, $n_{\mathrm{r}}(x)=n_{\mathrm{r}}(-x)$ and $n_{\mathrm{i}}(x)=-n_{\mathrm{i}}(-x)$ . The real part $n_{\mathrm{r}}(x)$ of $n(x)$ determines the index profile whereas its imaginary part $n_{\mathrm{i}}(x)$ determines the gain-loss profile, where a positive imaginary part implies loss and a negative imaginary part implies gain. Thus, an optical system with PT-symmetric potential (complex refractive index) has a symmetric index profile but an asymmetric gain/loss profile. Such a refractive index profile can be obtained in a system of two coupled optical structures (for example, waveguides, resonators and so on), one having passive loss and the other having gain compensating the loss of the other. While the analogy between PT symmetry in quantum mechanics and in optics is here clearly most apparent, PT-symmetric concepts are not restricted to the paraxial wave equation, but can emerge in a multitude of different scenarios (see main text).

waveguides $^{62}$ . In another example, a passive PT-symmetric structure formed using a Si waveguide, whose permittivity was modulated in its imaginary and real parts by periodically arranged and sinusoidal-shaped Ge/Cr and Si structures on top, was used to demonstrate ‘asymmetric back-reflection’ at an EP $^{65}$ (Fig. 2c), where the destructive interference of the coalescing modes suppressed the reflectance for one input direction but not for the other. The concept of unidirectional invisibility utilizing EPs was later extended to a wafer-scale multilayer structure $^{66}$ (Fig. 2g). Although such systems are unidirectionally reflectionless, the Lorentz reciprocity of light still holds.

## PT-symmetric waveguides and asymmetric power oscillations

The first fully PT-symmetric system with gain and loss was demonstrated $^{13}$ using two coupled waveguides, with constant coupling strength, fabricated from Fe-doped $LiNbO_{3}$ (Fig. 2b). Gain was gradually built up through two-wave mixing by selectively irradiating one of the waveguides with a pulsed laser. The other waveguide had loss arising from the optical excitation of electrons from $Fe^{2+}$ centres to the conduction band. At t=0 (no gain), transmis-

## Box 3 | EPs and passive PT symmetry

The simplest Hamiltonian describing a non-Hermitian system consisting of two lossy components is represented by a $2 \times 2$ matrix, where the off-diagonal elements represent the coupling strength (real $\kappa$ ) between the optical modes. The diagonal elements represent the unperturbed complex energies $\omega_{0}-i\gamma_{1}$ and $\omega_{0}-i\gamma_{2}$ of the modes ( $\gamma_{1}, \gamma_{2}>0$ corresponds to loss):

$$
H = \left[ \begin{array}{c c} {{\omega_ {0} - i \gamma_ {1}}} & {{\kappa}} \\ {{\kappa}} & {{\omega_ {0} - i \gamma_ {2}}} \end{array} \right] \quad \text {两个系统只有loss不同}
$$

The corresponding Schrödinger equation is given as follows (we set $\hbar=1$ without loss of generality):

$$
i \frac {\mathrm{d}}{\mathrm{d} t} \left[ \begin{array}{c} A _ {1} \\ A _ {2} \end{array} \right] = \left[ \begin{array}{c c} \omega_ {0} - i \gamma_ {1} & \kappa \\ \kappa & \omega_ {0} - i \gamma_ {2} \end{array} \right] \left[ \begin{array}{c} A _ {1} \\ A _ {2} \end{array} \right]
$$

Defining $\chi = (\gamma_{1} + \gamma_{2}) / 2$ and $\beta = (\gamma_{2} - \gamma_{1}) / 2$ and applying the gauge transformation $\begin{bmatrix} A_1 \\ A_2 \end{bmatrix} = \mathrm{e}^{-\chi t} \begin{bmatrix} A_1' \\ A_2' \end{bmatrix}$ , we end up with

求导即得

$$
\begin{array}{r l} i \frac {\mathrm{d}}{\mathrm{d} t} \left[ \begin{array}{c} A _ {1} ^ {'} \\ A _ {2} ^ {'} \end{array} \right] = & \left[ \begin{array}{c c} \omega_ {0} - i \gamma_ {1} & \kappa \\ \kappa & \omega_ {0} - i \gamma_ {2} \end{array} \right] \left[ \begin{array}{c} A _ {1} ^ {'} \\ A _ {2} ^ {'} \end{array} \right] + i \chi \left[ \begin{array}{c} A _ {1} ^ {'} \\ A _ {2} ^ {'} \end{array} \right] \\ = & \underbrace {\left[ \begin{array}{c c} \omega_ {0} + i \beta & \kappa \\ \kappa & \omega_ {0} - i \beta \end{array} \right]} _ {H ^ {'}} \left[ \begin{array}{c} A _ {1} ^ {'} \\ A _ {2} ^ {'} \end{array} \right] \end{array}
$$

The new Hamiltonian $H'$ has the same off-diagonal elements $\kappa$ as the original Hamiltonian H; however, the imaginary parts of its diagonal elements are now balanced (that is, the gain in one element has the same magnitude as the loss in the other). For this matrix to be PT-symmetric, $[H', PT] = 0$ should be satisfied. Note that the parity P operator is the Pauli operator $\sigma_{x} = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}$ and the

sion was the same regardless of the waveguide into which light was launched; during $0<t<t_{critical}$ (gain is smaller than a critical value), the system was in the exact PT phase that exhibits asymmetric power oscillations violating right–left symmetry; at $t=t_{critical}$ (gain reaches the critical value), the supermodes coalesced, and spontaneous PT breaking occurred. For $t>t_{critical}$ (gain exceeds the critical value) the system was in the broken-PT phase, and the supermodes were distributed asymmetrically between the waveguides, with one being confined in the active waveguide (with exponential amplification), and the other in the lossy waveguide (with decay). As a result, transmission always appeared at the output of the active waveguide regardless of the input waveguide. This effect is sometimes called non-reciprocal in the sense that power oscillations between the waveguides are asymmetric—this terminology is, however, ambiguous since Lorentz reciprocity still holds as long as no nonlinearity builds up.

## PT-symmetric resonators and non-reciprocal light transmission

In 2014, two works $^{18,67}$ reported PT symmetry and its breaking in coupled silica (passive, lossy) and an erbium-doped silica (active, with gain) WGM microresonators. The power of a pump laser controlled the gain while the distance between the resonators tuned the coupling strength (Fig. 2d). At fixed gain-to-loss time-reversal T operator corresponds to complex conjugation. Thus, starting with the commutator, we find that

$$
\begin{array}{r c l} {H ^ {\prime} P T} & = & {P T H ^ {\prime} \Rightarrow H ^ {\prime} P T (P T) ^ {- 1} = P T H ^ {\prime} (P T) ^ {- 1}} \\ & & {\Rightarrow H ^ {\prime} = P T H ^ {\prime} T P = P H ^ {\star} P,} \end{array}
$$

where we have used $T=T^{-1}$ , $P=P^{-1}$ and $TH^{\prime}T=H^{\prime*}$ . Thus, for this matrix to represent a PT-symmetric Hamiltonian, $H^{\prime}=PH^{\prime}*P$ should hold:

$$
\begin{array}{r l r} {{P H ^ {\prime \star} P}} & {{=}} & {{\left[ \begin{array}{l l} {{0}} & {{1}} \\ {{1}} & {{0}} \end{array} \right] \left[ \begin{array}{c c} {{\omega_ {0} - i \beta}} & {{\kappa}} \\ {{\kappa}} & {{\omega_ {0} + i \beta}} \end{array} \right] \left[ \begin{array}{l l} {{0}} & {{1}} \\ {{1}} & {{0}} \end{array} \right]}} \\ & {{=}} & {{\left[ \begin{array}{l l} {{0}} & {{1}} \\ {{1}} & {{0}} \end{array} \right] \left[ \begin{array}{c c} {{\kappa}} & {{\omega_ {0} - i \beta}} \\ {{\omega_ {0} + i \beta}} & {{\kappa}} \end{array} \right]}} \\ & {{=}} & {{\underbrace {\left[ \begin{array}{c c} {{\omega_ {0} + i \beta}} & {{\kappa}} \\ {{\kappa}} & {{\omega_ {0} - i \beta}} \end{array} \right]} _ {P H ^ {\prime \star} P} = H ^ {\prime}}} \\ & {} & {{\mathrm{通过改变参考点从} 0 \mathrm{到X}, \mathrm{实现loss-}}} \\ & {} & {{\mathrm{loss系统与gain-}}} \\ & {} & {{\mathrm{loss系统类比}, \mathrm{同样具有PT对称性}}} \end{array}
$$

Thus, the gauge transformation reveals the hidden PT symmetry present in the non-Hermitian matrix H by changing the reference point: before the transformation gain and loss are defined with respect to zero (positive $\gamma_{1,2}$ implies loss and negative $\gamma_{1,2}$ implies gain). After the transformation the reference point becomes $\chi$ . Consequently, if $\gamma_{1,2}$ is greater than $\chi$ , the system is considered as lossy. If $\gamma_{1,2}$ is less than $\chi$ , it is considered as a system with gain. Similarly, one can write the general non-Hermitian matrix as the sum of a PT-symmetric part and a lossy part

$$
\left[ \begin{array}{c c} \omega_ {0} - i \gamma_ {1} & \kappa \\ \kappa & \omega_ {0} - i \gamma_ {2} \end{array} \right] = \underbrace {\left[ \begin{array}{c c} \omega_ {0} + i \beta & \kappa \\ \kappa & \omega_ {0} - i \beta \end{array} \right]} _ {\text {PT - symmetric coupled systems}} + \underbrace {\left[ \begin{array}{c c} - i \chi & 0 \\ 0 & - i \chi \end{array} \right]} _ {\text {lossy uncoupled systems}}
$$

In this sense the dynamics of a loss–loss system is equivalent to that of a loss–gain system apart from a global exponential decay/amplification.

ratio, the resonances describing the two supermodes exhibited spontaneous PT-symmetry breaking $^{18}$ as the coupling strength was reduced. In the strong coupling regime (exact phase), two resonances (split modes) with different frequencies but similar linewidths were seen. With decreasing coupling strength, the split modes approached each other and finally coalesced at a critical value. Further decrease pushed the system into the broken phase where the modes bifurcated in their linewidths: one mode became increasingly localized in the active resonator and the other in the lossy resonator. As a result, the field was localized and amplified in the active microresonator regardless of whether it was input into the passive or the active microresonator. Consequently, the intracavity field intensity was resonantly amplified leading to a gain-saturation nonlinearity. Following an earlier theoretical suggestion $^{17}$ , this nonlinearity was used to demonstrate the first nonlinearity-based non-reciprocal optical device utilizing PT-symmetry breaking $^{18}$ .

This work $^{18}$ with microresonators ended the debate on the relation between PT-symmetry breaking and Lorentz reciprocity, by showing that such systems with gain and loss are reciprocal in the linear regime regardless of whether they are operated in the exact or in the broken-PT phase. Magneto-optical, nonlinear or time-dependent materials and structures are needed to break reciprocity $^{68}$ . The strong field localization in the broken-PT phase just helps to obtain the nonlinear response at much lower power with respect to the exact phase. PT-symmetric microresonators thus enable a significant reduction in the required input power and a higher rectification (complete absence of transmission in one direction and resonantly enhanced transmission in the opposite direction) for nonlinearity-based non-reciprocal light transmission.

![](images/096272936a62dde1a503d0a1a71ec403b66932c13b3a3e5b22d3e5e2eba0a5a5.jpg)
b

![](images/cff806a5e195d2922790e72ddbdcddfba52ec9ae0451029aeb9525efec6d4e87.jpg)

![](images/47a68f07bd2ec9cb7efa6fc809530fe02496f9868d8dd5f8ba831f7426eedfef.jpg)

![](images/84d577fd2e2651c5bc976203300748621d058322f1a7de48305a532d5daa7d8c.jpg)

![](images/41e2cc53f0a276da705a143b60e55de0a4e91edcdfbd021efa889bf0907c223a.jpg)

![](images/05b5719abd72a5648f49773ceb999aba798272d0b597a07795654ad9336a00e4.jpg)

![](images/4cfde55c216f3f93c2f3f21f9d8a8a4e19f9487f6fd199b73bfd1b846baab3c8.jpg)

![](images/694dc2235e50d1d389695c1c9abf3f9b483831d9d04715fd45c5209bb2669fa8.jpg)

![](images/0a41ed8c77a32239d956c7c76a247d9864dc861fb67250ec6bef542233840f0d.jpg)
Fig. 2 | PT symmetry and EPs in optical systems. a, Loss-induced transparency due to an EP in coupled passive waveguides $^{14}$ . b, Asymmetric power oscillations in a PT-symmetric system consisting of two coupled waveguides one with gain and the other with loss $^{13}$ . c, Unidirectional reflectionless propagation at an EP in a passive waveguide with a periodically modulated permittivity $^{22}$ . d, Non-reciprocal light transmission in the broken-PT phase in coupled WGM microresonators, one with passive loss and one with gain compensating the loss of the other $^{18}$ . e, Bloch oscillations and unidirectional invisibility in a PT-symmetric temporal lattice formed by two coupled fibre loops periodically switching between gain and loss $^{21}$ . f, Spawning rings of EPs in a 2D photonic crystal $^{53}$ . g, Unidirectional reflectance in forward and backward directions via EPs in a wafer-scale multilayer passive structure $^{66}$ .

In general the degree of non-reciprocity in such nonlinear devices depends on the intensity of the signal. Operating a

PT-symmetric system in the broken-PT phase helps to achieve a higher degree of non-reciprocity at much lower signal intensities. Moreover, although nonlinearity-based non-reciprocal devices may transmit strong signals in one direction and suppress them in the opposite direction, they may also exhibit 'dynamic reciprocity' for a certain class of weak signals that coexist with the strong signal $^{69}$ . Therefore, they cannot be used as isolators for arbitrary signals because an isolator should suppress the transmission in

e
c
a
![](images/081e0db4f5dd6fa7d5ccb089925ed64de6640f978e45bc2e69d3a040e5f467a9.jpg)
b

![](images/1fb979a0de4298ad422f3c189beec2f4b2983be95632a3d13c8c72a334f28d0a.jpg)

![](images/aea0d4ae8c46bafc1f6207788709ceab35dbf35cebc47c117e9e9dc4275c364d.jpg)

![](images/1bb7b76dc0595073a0c7683685ca4ba84ec6953bed92549b06a6e47a73b302a8.jpg)

![](images/0a49383ae290ae7b730ffa9f3777877bf4794222d299cc4e6e254a95bef1d825.jpg)

![](images/c515b2f554619b444bb8f98f9c553f515d0cea67f7b3c388ffdfebeece3da0e9.jpg)

![](images/43ed36457f1d7d5776719ce1308f1ba7eaa80fb793f0d4265091b5b645e48826.jpg)

![](images/5c564ae9b9a54b783c23be4bea4d479cc8f41e9f76da5a51696747f4927412bc.jpg)

![](images/9715635e6cdd0df22445d9aecd4670291b116969699a9b7a821edf6a56c03632.jpg)

d
![](images/f6570b1fcdef4e69851cd9950682d51d7e6d584f74a0177d17529d29b26b1c9a.jpg)

![](images/ce38468947702a2eb444bc0d7b7d747c6367af35999c33d2ff7e87e90e594928.jpg)

![](images/8497b9e04e59a31026ec81c16b2af78a50676c0b585b8f682c2cd443c8c2e9dd.jpg)

![](images/1b97466d7f88c276e850d2305965dcd95cb10160565a58c4d10d46fe2f9dedbd.jpg)

![](images/2ebb0f6e41ad4a791f70920eab6cd015415cecf8c5af9bc009afbb9cb94a805b.jpg)

![](images/6fcb595b7e774721e6cb341e5d2b18fe3dcec66a99bb8b71bea08bd0c696a14d.jpg)

![](images/615d75960c7f2551b3b77a42891645bf40b7c26322db820937ee456487976e71.jpg)
Fig. 3 | EPs and PT symmetry in lasers. a, Loss-induced suppression and revival of Raman lasing in WGM silica microresonators in the vicinity of an EP $^{16}$ . b, Reversal of pump dependence of lasing in quantum cascade lasers close to an EP $^{15}$ . c, Single-mode lasing in a microring resonator with PT-symmetry breaking $^{24}$ . d, Mode suppression and single mode lasing in coupled active-passive microring resonators by PT-symmetry breaking $^{23}$ . e, Partially directional lasing at an EP in a semiconductor microdisk laser with subwavelength-scale boundary perturbations $^{29}$ . f, Controlling emission of lasing in an erbium-doped WGM resonator from bidirectional to a fully unidirectional output in a preferred direction by transiting from one EP to another using Rayleigh scatterers $^{30}$ .

one direction regardless of the presence of a weak signal in the opposite direction, or in both. This is true even in PT-symmetric systems where gain and loss coexist. A PT-symmetric system that can be used as an isolator is yet to be discovered, with its advantages and disadvantages over the conventional ones being still under debate.

## PT-symmetric lattices and photonic crystals

Large-scale temporal PT-symmetric lattices $^{21}$ were realized using two coupled optical fibre loops of different lengths, which were periodically switched between gain (provided by semiconductor optical amplifiers) and loss (induced using acoustic modulators). Optical pulses propagating in the system are delayed or advanced, creating discretized arrival times to the detectors. Each of these 'time slots' corresponded to a node of an analogous spatial lattice (Fig. 2e). In the absence of gain (passive lattice) two bands formed without gap and the power in the lattice was conserved. In the broken-PT phase, parts of the band structure became imaginary and the power grew exponentially in time. At the EP, bandmerging took place and the power increased linearly in time. In the exact PT phase, an entirely real band structure with bandgap formed, and power oscillations took place due to the interference of non-orthogonal Floquet–Bloch modes, leading to power revivals during evolution. At the EP, light input in one direction experienced no reflection and was fully transmitted, even without any phase imprint, whereas the light in the other direction was reflected with a reflection coefficient exceeding one, leading to unidirectional invisibility.

The temporal PT lattice was subsequently used to demonstrate defect states $^{70}$ and optical solitons $^{71}$ . Defect states in PT lattices do not only trap light but also exhibit linearly growing emission if the associated EP is located within the continuum $^{70}$ . The existence of such bound states in the continuum in Hermitian systems is often linked to some symmetry, which completely decouples them from their surroundings. These observations suggest the possibility of controlling light emission by tuning defect parameters in PT lattices $^{72}$ .

In the PT lattice below the PT transition point, an optical input pulse with sufficiently high power to induce nonlinearity leads to stable discrete solitons $^{71}$ . In the broken-PT phase, the wavepacket spreads and is exponentially amplified when the power is low and the system is in the linear regime. In the nonlinear regime, however, the system becomes unstable, nonlinearity cannot prevent wavepacket spreading, and thus no solitary wave is formed. When the symmetry is restored and the band structure becomes real, discrete stable optical solitons appear.

Another related concept is ‘constant-intensity waves’ $^{73}$ supported by certain loss–gain potentials, which give rise to waves without any interference fringes in their intensity in spite of spatial potential variations—a feature that is impossible to realize in Hermitian scattering landscapes. These waves survive the presence of a Kerr nonlinearity and can thus be used to study ‘modulation instability’ in non-uniform potential landscapes. This concept also allows for perfect transmission through disordered media with gain and loss components $^{74}$ —an effect that was meanwhile experimentally realized in an acoustic setup $^{75}$ .

In PT-symmetric photonic crystals (PhCs), increasing the strength of gain and loss may lead to the coalescence of EPs $^{52}$ . One intriguing feature $^{53,54}$ is band-merging, leading to rings and contours of EPs, and to bandgaps opening. The Dirac point where the two conical bands of the material dispersion touch each other in a Hermitian system, gives rise to a ring of EPs in passive two-dimensional PhC slabs due to non-Hermiticity induced radiation losses $^{53}$ (Fig. 2f).

## EPs in resonators and lasers

One of the earliest studies on the role of EPs in lasers is the prediction and the measurement of linewidth broadening beyond the Schawlow–Townes linewidth by a factor, known as the Petermann excess noise factor $^{76}$ . This excess noise originates from non-orthogonal modes formed due to the non-Hermiticity of a laser. While a first generation of experiments $^{77}$ already probed the linewidth-broadening due to non-orthogonality, the maximal broadening at an EP was explicitly explored in a recent work on phonon lasers $^{39}$ . EPs in lasers are also at the origin of many dynamical phenomena, such as fast self-pulsations ('dispersive self Q-switching') in two-section distributed-feedback lasers where mode-beating occurs when two nearly degenerate modes have similar threshold gain $^{78}$ . In addition, an investigation of PT-symmetric resonators in the vicinity of an EP have revealed that non-Hermiticity, if properly engineered and controlled, can enhance the laser performance regarding single-mode operation, wavelength tunability, and stability $^{79}$ .

A striking example for the nontrivial laser physics occurring at an EP is the reduction in light emission despite an increasing pump power $^{15,80}$ . The theoretical prediction $^{80}$ that pumping each subcomponent of a coupled laser system separately steers its complex eigenvalues towards a 'pump-induced EP' that inhibits lasing was first realized in coupled quantum cascade lasers $^{15}$ (Fig. 3b). If the lasers are pumped equally, no EP appears and the system emits in two modes associated with the individual lasers. Driving one of the lasers with a constant pump slightly above its lasing threshold and varying the pump of the other one reverses the dependence of the emitted light intensity on the pump strength: as the pump of the second laser is increased, the system gradually approaches an EP; the output of the existing single lasing line gradually decreases and then completely turns off at the EP ('lasing death at an EP'). Further increase of the pump moves the system away from the EP and hence the lasing re-emerges.

While these experiments demonstrated the influence of an EP on a laser, the EP itself was observed only indirectly, as being pushed below the lasing threshold by the lasing death phenomenon $^{15}$ . A system of coupled resonators $^{16}$ allowed to probe the resonances also below the threshold through their input–output relations and through their Raman lasing above threshold (Fig. 3a). The positions and widths of the resonances in this ‘photonic molecule’ provided the real and imaginary parts of the eigenvalues that were steered to an EP by adjusting the inter-resonator coupling and the loss imbalance of the resonators. This information was then directly matched with the observation of a loss-induced suppression and revival of Raman lasing as the system was moved to and away from the EP. Keeping the pump power above the lasing threshold and introducing loss to one of the resonators annihilated an existing Raman laser. Increasing the loss further pushed the system closer to the EP and enhanced the intracavity field intensity via field localization, thereby reviving the Raman laser, whose intensity increased with increasing loss. Similarly, the laser threshold power first increased but then decreased with increasing loss $^{16}$ . These observations are in stark contrast with the general expectation that the loss is detrimental to laser systems.

## PT-symmetric lasers

PT-symmetry breaking was proposed for single-mode operation in multimode large-area laser amplifiers $^{81}$ where the coupling strength between identical modes in two subsystems (with loss and gain, respectively) forming the PT structure is higher for higher-order modes. Thus, the gain (and the same amount of loss) required for PT breaking is higher for the higher-order modes than for the fundamental mode. If the system is suitably designed and excited, the fundamental mode will be the first to undergo PT breaking and will thus be amplified. The rest of the modes will be bounded in their amplitudes. The modes below the PT-breaking threshold are symmetrically distributed and exhibit oscillations. The mode pair above the PT-breaking threshold has an asymmetric distribution with the mode that is localized in the gain subsystem being shifted across the lasing threshold.

Single-mode lasing in the PT-breaking regime was demonstrated in two coupled InGaAsP microrings $^{23}$ and in an InGaAsP/InP microring $^{24}$ (Fig. 3c,d). These experiments showed that by selectively breaking PT symmetry, stable single-mode operation is possible in laser cavities with multiple modes within the gain spectral bandwidth. Any resonator with a spectrally non-uniform gain profile can sustain single mode operation if the gain compensates losses in only one resonance but not in the others. Single-mode operation can also be achieved in lasers without PT configurations by using a spatially modulated pump, exploiting the Vernier effect or by introducing dispersive elements. The crucial difference between lasers with a PT-symmetric setting and those without lies in the maximum achievable gain (while maintaining single-mode operation), which is defined as the contrast between the principal mode gain $g_{0}$ and the gain $g_{1}$ of the next competing mode. For a laser in the PT configuration one finds

$$
g _ {\mathrm{max}} ^ {\mathrm{PT}} = g _ {\mathrm{max}} \sqrt {\frac {g _ {0} / g _ {1} + 1}{g _ {0} / g _ {1} - 1}}
$$

whereas $g_{max}=g_{0}-g_{1}$ is the maximum achievable gain contrast for a laser without PT configuration $^{23}$ . This square-root behaviour is a consequence of the square-root topology in the vicinity of an EP in non-Hermitian systems, including PT-symmetric systems (Box 1). As $g_{max}$ is very small for broadband gain materials, this square-root behaviour leads to higher gain enhancement and better mode selection.

The concept of PT symmetry and its breaking was later applied to different laser configurations and platforms. EPs and an associated transition to single-mode lasing were observed in PhC nanolasers through controlling the gain contrast of coupled lasers via graphene-induced loss $^{82}$ . PT-symmetry breaking occurred in electrically pumped and coherently coupled vertical-cavity surface-emitting laser (VCSEL) arrays $^{83}$ . Beam steering (different from the conventional method of locking each element to a common master) and hopping of the lasing mode between in- and out-of-phase modes were shown to arise from the non-Hermiticity of the array caused by the gain-loss contrast. The maximum relative phase tuning was reached at or beyond the EP in the PT-broken regime.

## Coherent-perfect absorption and lasing

Resonances appear as poles of the scattering matrix that describes how a resonator scatters incoming modes into the outgoing ones (Box 4). For a resonator without a gain medium, these complex poles have negative imaginary parts. But when gain is added, they move towards the real axis until the first pole reaches the real axis and lasing sets in. The scattering matrix of the resonator also features zeros in the upper part of the complex plane located at the mirror-images of the poles. When adding loss they move downwards until a zero reaches the real axis and the time-reversed process of lasing takes place—a coherent and monochromatic field illuminating the resonator with a suitable wavefront gets perfectly absorbed $^{84}$ . This ‘coherent-perfect absorption’ (CPA) generalizes the concept of ‘critical coupling’ to an arbitrary number of modes. First experimental realizations have recently been reported $^{85}$ .

In a PT-symmetric structure, the conditions for CPA and those for lasing can coincide $^{25}$ . The discrete points in parameter space where CPA-lasing occurs are the points where a pole and a zero of the scattering matrix (Box 4) meet on the real axis $^{26}$ . A first implementation was based on a pair of passive resonators coupled to a microwave transmission line $^{27}$ . Lasing (coherent amplification) and anti-lasing (coherent absorption) in the same optical device was realized $^{86}$ in a waveguide where InGaAsP quantum wells on an InP substrate provided gain, and periodic Cr/Ge structures on top induced loss to provide PT-symmetric gain-loss modulation. The phase difference of the light input to the waveguide in both directions was tuned to selectively excite either the lasing or the

## Box 4 | PT-symmetry breaking in scattering systems

PT-symmetric Hamiltonians exhibit spontaneous symmetry breaking in their eigenvalues, which are either real or come in complex conjugate pairs. Whereas Hamiltonians are convenient to describe problems with specific boundary conditions, unbounded scattering problems are typically characterized by a scattering matrix S, which relates the amplitudes of incoming modes to those of the outgoing modes, $S \overrightarrow{\psi}^{in} = \overrightarrow{\psi}^{out}$ (in a flux-normalized basis). In systems without any gain or loss the scattering matrix is unitary—that is, $S^{\dagger}S=1$ and the eigenvalues of S are uni-modular complex numbers. To obtain a corresponding condition for PT-symmetric systems, we recall that the scattering matrix for scattering at a potential V that is part of the Hamiltonian $H=H_{0}+V$ can be written as the following limit, $S=\lim_{t\to\infty}\exp[iH_{0}t]\exp[-i2Ht]\exp[iH_{0}t]$ . Knowing that both the Hamiltonian of free space $H_{0}$ and the Hamiltonian H including the potential V are PT-symmetric, $(PT)H_{0}(PT)=H_{0}$ and $(PT)H(PT)=H$ , one can easily show that $(PT)\exp[iH_{0}t](PT)=\exp[-iH_{0}t]$ and $(PT)\exp[-i2Ht](PT)=\exp[i2Ht]$ . We thus obtain the desired relation $^{26}$ $(PT)S(PT)=S^{-1}$ , which is reminiscent of the unitarity condition for Hermitian systems $S^{\dagger}=S^{-1}$ . Indeed, both relations result in the relation $|\det S|=1$ . Unitarity, however, is a stronger constraint, in the sense that the eigenvalues $s_{n}$ of unitary scattering matrices always lie on the complex unit circle, whereas the eigenvalues of a PT-symmetric S matrix are either uni-modular or come in pairs with reciprocal moduli—for example, $s_{2}=1/s_{1}^{*}$ for a $2\times2$ scattering matrix. Correspondingly, one finds that also in scattering problems the eigenchannels of the scattering matrix can be in two different phases separated by an exceptional point at which $s_{1}=s_{2}$ : In the PT-symmetric phase the S matrix features uni-modular eigenvalues (with no net amplification between input and output) and PT-symmetric eigenvectors. In the broken-PT phase the S matrix eigenvalues are shifted away from the unit circle and the PT operator transforms one eigenvector (with gain) into the other one (with loss). In this PT-broken phase, if one eigenvalue goes to zero ( $s_{1}\to0$ ) its counterpart goes to infinity ( $s_{2}\to\infty$ ). At these special parameter points, the object described by such a scattering matrix can emit coherent waves (such as laser light) and simultaneously fully absorb incoming waves with appropriate frequency, amplitude and phase $^{25,26}$ . An experimental realization of such a PT-symmetric laser absorber is very challenging, particularly in the presence of noise, but first results have been reported $^{86}$ .

anti-lasing mode. CPA lasers provide the flexibility of using the same device as an absorber, a laser or as a modulator $^{86}$ .

## Asymmetric backscattering and directional emission

Highly symmetrical shapes of WGM microresonators and microlasers prevent directional light output as desired for many applications. To circumvent this problem, the resonators' circular rotational symmetry is intentionally broken either by deforming their shapes or by introducing defects so that the clockwise (CW) and the counterclockwise (CCW) mode components differ $^{87,88}$ . Chiral WGMs with a predominantly CCW or CW component, such as in spiral-shaped resonators $^{88}$ , were linked to the asymmetric scattering between CCW and CW waves $^{89}$ , which forms non-orthogonal co-propagating modes. These properties are strongly enhanced at an EP where the scattering between the CW and CCW components vanishes completely in one of the two directions. This asymmetric backscattering and the consequent chirality, non-orthogonality and co-propagation have been observed in microcavities with intentionally placed scattering centres $^{30,90-93}$ .

Asymmetric backscattering with partially directional emission was reported in a wavelength-scale microdisc with InGaAsP/InGaAs quantum wells $^{92}$ (Fig. 3e). The link between asymmetric backscattering and EPs was unambiguously demonstrated in a passive microresonator $^{30}$ , using two nanotips as Rayleigh scatterers whose relative size and distance within the resonator's mode volume could be finely adjusted to drive the system close to or away from EPs (Fig. 3f). At an EP, the transmission was the same regardless of the input direction. Reflection curves, however, showed a pronounced resonance peak only for one of the input directions. These experiments clarified the difference between directionality (the amplitude difference between the CW and CCW directions without correcting for the bias of the input light direction) and chirality (an intrinsic property of a WGM independent of the input direction), concluding that chirality is the quantifier of asymmetric backscattering. Directionality changes depending on how close the system is to an EP; however, the direction in which the input light is injected remains dominant. Chirality, on the other hand, can take any value in the range [-1,1]: it is zero if backscattering is equal for the CW and CCW inputs (orthogonal eigenstates, away from an EP); it is non-zero if backscattering for the input in one direction is larger than that for the other direction; it approaches 1 or -1 as the system approaches an EP. This concept, when translated to WGM microlasers, revealed that laser emission is bidirectional when chirality is close to zero; it occurs only in one direction when chirality is maximal; and its direction can be reversed by moving from one EP to another (Fig. 3f).

Travelling CW/CCW modes in WGM resonators carry high optical angular momentum (OAM). However, the reduction, if not complete cancellation, of the OAM due to simultaneous excitation of CW and CCW eigenmodes, and the difficulty of extracting the highly confined WGMs into free-space without reducing their OAM have hindered their practical use. It was suggested that the chirality at an EP can be used for developing WGM microlasers that emit OAM or vortex beams $^{30}$ . Experimentally, single-mode OAM lasing into free space was demonstrated $^{94}$ in a microring laser, with an embedded angular grating structure operating at an EP.

## Encircling EPs

There is still considerable potential to explore new physics when encircling an EP rather than approaching it $^{95}$ . Changing system parameters sufficiently slowly and continuously around an EP was originally expected to give rise to an adiabatic state evolution. A closed loop on the self-intersecting Riemann surface at an EP (Box 1) would move from one Riemann sheet to the other one such that an initial state transported around the EP would then return to a different state in terms of a so-called state flip $^{32,95}$ . Only a second loop would return the system to its initial state (apart from a Berry phase of $\pi$ ).

Recent theoretical works have shown, however, that the non-Hermiticity necessary for the observation of an EP actually prevents an adiabatic evolution $^{96-98}$ . The presence of non-adiabatic terms leads to a chiral behaviour, such that the encircling direction of the EP essentially determines in which final state the system will arrive (Fig. 4a,b). This behaviour remained unattainable in earlier studies due to the parametric (not dynamical) encircling consisting of a concatenation of static experiments $^{32,95}$ . To go beyond this limitation, the dynamical variation of parameters along a loop around an EP was mapped onto a boundary-modulated waveguide with two transverse modes $^{99}$ . The corresponding device transmitted waves only into one of the two transverse modes at either one of its two outputs. In another implementation $^{35}$ an optomechanical system was steered to encircle an EP, giving rise to asymmetric energy transfer: in one encircling direction, there is energy transfer between the system's eigenmodes; in the other, non-adiabatic transitions lead to its breakdown. This concept was recently implemented in silicon photonics to span the entire optical communications band $^{100}$ .

## PT optomechanics

The interaction between optical fields and mechanical motion is the central topic of cavity optomechanics (COM) $^{101}$ . Hybridization of non-Hermitian physics and optomechanics to engineer these interactions has initiated the field of PT optomechanics $^{36}$ (or non-Hermitian optomechanics more generally).

The first example $^{36}$ investigated phonon lasing in a PT-symmetric structure (a passive optical resonator with a mechanical mode coupled to an active resonator without mechanical mode), predicting a giant enhancement in the optomechanical coupling strength that enables highly efficient phonon lasing around an EP $^{36}$ . A study of the nonlinear dynamics of this system without resorting to phonon lasing revealed that gain saturation helps to regulate oscillation amplitudes that otherwise would be unbounded $^{40}$ . In coupled mechanical resonators with optically induced loss and gain, a combination of nonlinear saturation and noise leads to preserved or weakly broken PT symmetry, and a transition from a thermal to a lasing state with small amplitude $^{102}$ . The same structure was used to study optomechanically induced transparency $^{101}$ (OMIT), revealing $^{38}$ an inverted spectrum with a transparency dip between two sideband peaks, gain-induced suppression of optical transmission (reversal of gain dependence) and transitions between slow- and fast-light regimes by tuning the gain-to-loss ratio.

The enhancement of optomechanical interactions and associated nonlinearities around an EP has inspired numerous studies. A system consisting of a lossy resonator with weak mechanical nonlinearity coupled to a resonator with mechanical gain but no nonlinearity was proposed for non-reciprocal phonon transport in the broken-PT regime $^{103}$ . Ultralow threshold chaos $^{104}$ was linked with PT-symmetry breaking in an optomechanical system due to enhanced optomechanical nonlinearity. Emergence of a third-order EP (three eigenfrequencies and eigenmodes of the system coalesce) and the consequent enhancement of optical spring and mechanical damping, which enable low-power mechanical cooling, were predicted in a PT-symmetric optomechanical system $^{38}$ . A mechanical analogue of the loss-induced optical lasing $^{16}$ was predicted in optomechanical resonators by tuning the loss of intrinsic two-level system (TLS) defects that exist in amorphous materials used in the fabrication of optomechanical resonators $^{105}$ . Despite many theoretical studies, there are only two reported experiments $^{35,39}$ so far on non-Hermitian optomechanics. In one, encircling an EP resulted in asymmetric energy transfer $^{35}$ and in the other operating a phonon laser at an EP lead to significant linewidth broadening $^{39}$ . These studies provide deeper insights into the interplay between non-Hermiticity and optomechanical coupling and can thus help to explore the behaviour of thermal and quantum fluctuations in the vicinity of EPs, and to construct new devices for controlling photon–phonon interactions.

## Sensing at exceptional points

Lifting DP and/or EP degeneracies by perturbations leads to a splitting of energy levels, which can be quantified to understand the nature of the perturbations. The topology exhibited in the vicinity of a DP or an EP (Box 1) leads to a striking difference in their sensitivity to perturbations $^{106}$ (Box 5). While frequency splitting close to a DP is proportional to the perturbation strength $\epsilon$ , it scales as $\epsilon^{1/N}$ in a system $^{106}$ close to an N-fold EP where N eigenfrequencies and eigenmodes coalesce simultaneously $^{107}$ . Hence, splitting at an EP is larger for a sufficiently small perturbation ( $\epsilon \ll 1$ ). Similarly, the splitting is larger at an EP of higher degree. For stronger perturbations, the response of a system operating at an EP approaches that of a system operating

![](images/9cc0bb63d8b8bb7c7eef7d9482b9aff1e49fd6f3b29a8dc9bac4c166129aae9a.jpg)

![](images/d129abb56a2c0f8b762038094ada6fd28683848cd7d2857c84d2e0d505f9376b.jpg)

![](images/cfb8910ba2200e2643cbbf9388b9de7a31385e7e4c2a5ce59e7351b4abd4654c.jpg)

![](images/f84cbbac404ca7d43142874da97e80f65d8f0f45df88b37e2f6a3912a7adc6d2.jpg)

![](images/c3c39735e77f3049ce205d01767011bbe8fe0c61a94608f122bffcbe27237c33.jpg)

![](images/29629afa73d96cbdca22b600472be1f4ed64c63d17cc8d417cf2736d1aa1732b.jpg)

![](images/55549d5add8868e504e4a0e835a1abb05cd0965c831d16fca4542bb8d680070d.jpg)

![](images/466f1b44b8eb131928b7c8d5ce34d7e7f12c7f1b59af7255c1b799eb033e6bb1.jpg)

![](images/e7ce10cd8d508521de77a3c02c4dee1fb9e3a8c2038d72aa9871a4494ca6a672.jpg)
Fig. 4 | Topological photonics in non-Hermitian systems. a,b, Dynamically encircling an EP with starting points on different Riemann sheets denoted with red (gain) and blue (loss) surfaces shows topological features. A clockwise (CW) loop around an EP ends on the blue surface (a), and a counterclockwise (CCW) loop around the EP ends on the red surface (b). End points depend only on the encircling direction, but not on the starting point. c, Realization of the complex Su–Schrieffer–Heeger (SSH) chain with a lattice composed of coupled dielectric cylindrical resonators $^{72}$ . A defect created by repeating one spacing breaks the sub-lattice symmetry, which in turn isolates the state from losses that respect PT symmetry, and enhances its visibility because losses suppress all other possible states. The panels depicting the time evolution of the field intensity show that without losses, diffraction and interferences drastically degrade the propagation of the interface state. d, Topologically protected bound state in a PT-symmetric array of coupled passive waveguides fabricated from fused silica glass $^{123}$ . The light is confined at the defect site formed by dimerization in a PT-symmetric system in the exact phase (left) and in a Hermitian system (right).

at a DP. When operating the sensor at an EP with PT symmetry, an important advantage, in addition to the $\epsilon^{1/N}$ behaviour, arises: due to the gain–loss balance the resonance modes have narrow linewidths, which enables resolving much smaller frequency splittings. The same Nth root behaviour at the EP occurs also for loss-only systems, leading to an equivalent response to small perturbations. However, the increased losses in such systems produce resonances with larger linewidths, which impose a limit on the smallest amount of frequency splitting that can be resolved. As a result, PT systems can detect much smaller perturbations than loss-only systems due to their gain–loss balance and hence narrower linewidth.

## Box 5 | Enhancement of sensing at EPs

Sensors operating in the vicinity of EPs are asserted to be superior to those operating away from EPs. This sensitivity enhancement has been attributed to the nth root topology of the eigenvalue surfaces formed by varying the parameters of the system (Fig. 1 and the figure in Box 1 depict a square-root behaviour because two eigenvalues are involved). Let us first consider here a system that is either at a conventional Hermitian degeneracy (DP) or at a generic non-Hermitian degeneracy (EP). Adding the same perturbation to both of these two situations individually differentiates them in the way the perturbation lifts the degeneracy in the corresponding eigenvalues. At a DP the unperturbed Hamiltonian can be written as follows

$$
H _ {0} ^ {\mathrm{DP}} = \left( \begin{array}{c c} E _ {0} & 0 \\ 0 & E _ {0} \end{array} \right)
$$

Adding now a non-Hermitian perturbation $\varepsilon H_{1}$ where

$$
H _ {1} = \left( \begin{array}{c c} E _ {1} & A _ {1} \\ B _ {1} & E _ {1} \end{array} \right)
$$

with $E_{1}, A_{1}, B_{1} \in C$ , leads to eigenvalues of the perturbed Hamiltonian $H = H_{0}^{DP} + \epsilon H_{1}$ that are split by $\Delta E_{DP} = 2\epsilon\sqrt{A_{1}B_{1}}$ . The real part of $\Delta E_{DP}$ corresponds, for example, to the splitting between two resonance positions away from the perfect overlap at the DP (the imaginary part corresponds to a splitting in the linewidth). When applying the same perturbation $\varepsilon H_{1}$ to a Hamiltonian that is located at an EP, where

$$
H _ {0} ^ {\mathrm{EP}} = \left( \begin{array}{c c} E _ {0} & A _ {0} \\ 0 & E _ {0} \end{array} \right)
$$

we obtain an eigenvalue splitting of $\Delta E_{EP}=2\sqrt{\epsilon}\sqrt{A_{0}B_{1}+\epsilon A_{1}B_{1}}$ that is not linear in the perturbation strength $\epsilon$ (as for the DP), but that scales with the square root of $\epsilon$ . For a sufficiently weak perturbation (that is, for $\varepsilon\ll1$ or $|A_{0}|\gg|A_{1}|$ ) we thus find that the splitting at the EP is always enhanced as compared to the DP: $\Delta E_{EP}=\Delta E_{DP}\sqrt{\epsilon+A_{0}/A_{1}}/\sqrt{\epsilon}$ . The situation described here is typically encountered in particle sensors based on microcavities $^{7,8,112}$ , where the two degenerate modes may, for example, be the whispering gallery modes in a circular resonator and the perturbation may be a particle that splits their degeneracy when being close to the rim of the resonator (panel a of the figure below).

![](images/15a38c704d51afc5cce675f58012a510890518aa130953032d78af8ff4aba717.jpg)

A similar situation occurs in sensors that detect a small change in the refractive index of a medium. Consider as an illustrative example the case of three nearest-neighbour-coupled sites with balanced gain and loss at the outer two elements and no gain/loss in the middle site $^{113}$ . Taking out the common real part of the refractive index in these sites, we end up with the simple Hamiltonian

$$
H _ {0} ^ {\mathrm{EP}} = \left( \begin{array}{c c c} i g & \kappa & 0 \\ \kappa & 0 & \kappa \\ 0 & \kappa & - i g \end{array} \right)
$$

that can be tuned to a threefold EP (EP3) simply by shifting the gain/loss-strength g to the value $g = \sqrt{2} \kappa$ , where $\kappa$ is the coupling strength between sites. At this parametric position, where all the three eigenvalues and eigenvectors of $H_{0}^{EP}$ coincide, the sensitivity of the system to a perturbation in any of the three elements is strongly enhanced. Consider adding to the gain element a perturbation in the form of a refractive index change $ig \rightarrow ig + \epsilon$ , for which case one can show using perturbation theory around the EP that two of the degenerate eigenvalues split as $\Delta E_{EP} \approx 3\kappa^{2/3} \sqrt[3]{\epsilon}/2$ . At an EP3 we thus obtain an eigenvalue splitting that scales with the third root of the perturbation strength $\varepsilon$ , suggesting that operating a sensor at an EP in general and at a higher-order EP in particular can significantly enhance a system's sensitivity with respect to perturbations (panel b).

In addition to the splitting of the eigenvalues one, however, also needs to take into account their resolvability. Within the context of nanoparticle sensing Özdemir et al. introduced a criterion that states that the amount of splitting should be larger than the total loss of the system (that is, resonator losses plus nanoparticle-induced losses) $^{131}$ . It is here that PT-symmetric sensing setups come in handy, as the corresponding gain-loss balance in them results in narrower linewidth that can resolve much smaller mode splittings and hence much smaller perturbations. A loss-only system operating at an EP, although having the same dependence (that is, $\epsilon^{1/2}$ ) on the perturbation, may not be able to resolve an induced mode splitting because it has higher loss and broader linewidth.

![](images/d732e661e2727373af56777077d4229ea00422fa062e6ef48cf8ccd43b511ff4.jpg)
Enhancement of sensor response at an EP. a, Enhancement of the response of a nanoparticle sensor when operating at a twofold EP compared to operation at a DP $^{112}$ . The square-root dependence at an EP is seen in the inset. With increasing perturbation strength, the response at an EP approaches that at a DP. b, Enhancement of the response of a refractive index sensor when it is operated at an EP3 $^{113}$ .

The sensitivity enhancement at an EP was theoretically investigated for particle detection in resonators $^{106}$ and one-dimensional PhC nanobeams $^{108}$ , chemical detection in PT-symmetric photoexcited graphene metasurfaces $^{109}$ , and refractive index sensing with PT-symmetric microlasers $^{110}$ . A sensitivity enhancement of two orders of magnitude was predicted $^{111}$ close to the transition point from the unbroken- to the broken-PT regime for detecting mechanical motion.

The theoretical concepts developed around EP- and PT-enhanced sensing were recently confirmed $^{112,113}$ in ring resonators for nanoparticle sensing using twofold EPs with a square-root dependence $^{112}$ , and for refractive index sensing using threefold EPs with cubic-root dependence $^{113}$ on the perturbation strength (Box 5). The enhancement in nanoparticle detection at an EP originates from the joint contribution of the symmetric scattering due to the perturbation (as is the case at a DP) and the fully asymmetric internal backscattering, which is much larger than the former for weak perturbations $^{106}$ . Although these proof-of-principle experiments have opened an exciting direction in optical sensing with unparalleled resolution and sensitivity, there are still many remaining technical challenges before such sensors can be put to practical use. For example, since each detection event in these sensors will move the system away from the EP, the sensitivity will gradually decrease with an increasing number of detection events (unless the system is brought back to the EP after each event). This limits the number of events that can be detected with an EP-enabled sensitivity. Therefore, practical systems for bringing the system back to the EP are needed. In this regard, PT-symmetric systems can have advantages over systems with only lossy components as they can be turned back to an EP after each detection event simply by modifying the gain under fixed coupling strength.

A plethora of platforms and techniques have been utilized for imaging and sensing, spanning from nano/microelectromechanical systems (NEMS/MEMS) and plasmonics to optical resonators and electronics, and from the use of conventional classical approaches to inherently quantum or quantum-inspired designs. It is not clear yet whether EP- and PT-enhanced sensors can beat these matured technologies in practical settings and, more importantly, whether they can approach quantum-noise-limited precision and compete with the performance of quantum(-inspired) sensors. EP-enhanced sensing is still in its infancy and we expect more theoretical studies on the connections of this field to the established quantum limits and a development of the reported proof-of-principle experiments into more elaborate and practical sensing platforms.

## Topological effects in PT-symmetric systems

Topological photonics has emerged as one of the most fascinating and rapidly growing fields in physics $^{114}$ . In a broader sense, already the effects related to encircling of EPs are topological since they rely on the specific topology of the Riemann sheets that self-intersect at an EP (Fig. 4a,b). In a stricter sense, topological photonics deals with topologically protected phases, which enable robust and highly confined states of light. These can arise at the interface between two regions that have topologically distinct band structures. Recently, there have been efforts to investigate topological physics in dissipative non-Hermitian systems $^{115-118}$ .

The argument that a topological edge state in a system, described by a PT-symmetric Hamiltonian, cannot be an eigenstate of the PT operator because the action of the PT operator will send the state to the other edge and break PT symmetry, has been used to state the impossibility of PT-symmetric topological states $^{119}$ . The theoretical demonstration $^{118}$ of the existence of topologically protected mid-gap states in the broken-PT regime of a photonic system was supported by the first experimental demonstration $^{116}$ of a topological transition in the bulk of a non-Hermitian system in the broken-PT regime. Photonic Floquet topological insulators with scatter-free edge transport were realized experimentally $^{117}$ , and the existence of a PT-symmetric variant in a non-Hermitian system with periodically modulated potentials was pointed out theoretically $^{120}$ . Recent studies have shown that topologically protected defect states can emerge in non-Hermitian systems at the interface of two regions with different non-Hermiticity by combining PT and chiral symmetries $^{121}$ , and that chiral topological edge modes in a non-Hermitian variant of two-dimensional Dirac equations are related to the EPs of the bulk Hamiltonians $^{122}$ .

Selective control and enhancement of a topologically induced state by localized absorptive losses was demonstrated in a one-dimensional chain of coupled microwave resonators with alternating spacing between them $^{72}$ (Fig. 4c). A defect that broke PT symmetry was formed at an interface state by repeating one of the spacings. The first truly PT-symmetric topological interface state was recently demonstrated $^{123}$ in a photonic waveguide lattice with a (similarly induced) topological defect (Fig. 4d). The waveguides were coupled to their nearest neighbours with alternating coupling strengths determined by their spacing. Dimerization (difference of the coupling strengths) was used to tune the system across an EP, completing the transition from the broken- to the unbroken-PT regime. For zero dimerization, due to the absence of an interface/defect state, an incoming wavepacket spreads regardless of whether the lattice is Hermitian or not. For non-zero dimerization, the interface state creates a topological state that strongly confines light at the defect site. For a non-Hermitian lattice, this takes place only when the system moves into the PT-symmetric regime. Probing topological effects in non-Hermitian systems may further clarify how topological Chern numbers are related with EPs, and how global and local PT symmetry affect light transport.

## Future outlook and discussions

A considerable number of theoretical and experimental breakthroughs have been reported in PT-symmetric photonics. Experimental implementations have already reached a mature stage beyond mere demonstrations of PT symmetry and its breaking, now aiming at creating functionalities unseen in purely dissipative or amplifying systems. For example, the scattering properties of nanoparticles with gain and loss regions or nanostructures composed of such nanoparticles seem suitable for designing structures with tunable scattering and transmission. Similarly, the control of physical processes in composite structures, formed with organic and molecular materials, small molecules, polymers and two-dimensional materials, via PT symmetry concepts can be used for developing novel photonic functionalities. Additional work is needed to clarify the fundamental limits of EP-enhanced sensors and whether they can approach quantum-noise-limited performance. Here we provide a brief account of some of the topics that we expect to attract attention in future PT-physics studies.

Plasmonics. Plasmonics is concerned with the interaction of light with electrons in metals. It allows one to localize light fields in sub-wavelength dimensions and to achieve strong intensity enhancements $^{124}$ . Unfortunately, plasmons suffer from dissipation in metals (Ohmic losses). These losses can be compensated, and dissipative dynamics in plasmonics can be controlled by embedding structures in gain-providing environments or by injecting carriers/plasmons. These efforts provide a platform for PT plasmonics in the sense that the loss in one plasmonic system can be balanced with gain in another system configured to exchange energy with it. Although there have been several theoretical studies $^{45,46}$ on PT-symmetric plasmonics, no experiment has been reported yet. It is also still unclear whether PT plasmonics can lead to strong coupling between an emitter and a plasmonic mode.

PT symmetry and non-Hermiticity in the quantum regime. The majority of experimental and theoretical concepts developed so far in PT-symmetric physics consider the classical regime, but we expect an extension to the quantum regime. Since PT systems operate with gain and loss, which are intrinsically linked to quantum noise through the quantum fluctuation-dissipation theorem, a quantum description should encapsulate such effects $^{102,125}$ . The ultimate goal would be to precisely tailor non-Hermitian photonic systems via coupling strength, loss-gain ratio or loss engineering to address the challenges in quantum photonics. Engineering transitions in quantum dynamical systems, by driving them closer to or further away from EPs or investigating the interaction of a single atom with the field of a PT-symmetric system would be of interest. It is still an open question whether photon number statistics and quantum correlations will be preserved when a quantum state of light is transmitted through a PT-symmetric network in the broken- and unbroken-PT regimes, and whether gain-loss engineering can help to provide longer propagation distances without degrading the quantum nature of light. Similarly, investigating non-Hermitian nonlinear parametric processes $^{126,127}$ for preparing quantum states of light, quantum frequency translation, parametric amplification and frequency combs, as well as for enhancing photon-photon interactions, will be of interest. A recent study $^{128}$ suggests that PT-symmetric qubits may be more robust against decoherence, and thus may be better suited for quantum information processing. A study $^{129}$ on information flow in PT-symmetric non-Hermitian systems has revealed that complete information retrieval is possible from the environment only in the exact PT phase; the PT phase transition (at the EP) marks the boundary between the reversible and irreversible criticality of information flow; and in its vicinity the recurrence times and distinguishability of quantum states exhibit power law behaviour. The information retrieval is then attributed to a hidden entangled partner protected by PT symmetry. The idea of PT symmetry was also extended to strongly correlated many-body systems, revealing a novel quantum phase transition, accompanied by the spontaneous breaking of PT symmetry, which does not have a correspondence in Hermitian quantum many-body systems $^{130}$ . The same study also predicts anomalously enhanced superfluid correlations in the PT-broken quantum critical phase. The experimental verification of these predictions related to quantum phenomena and quantum processes will certainly encourage further studies on PT-symmetric quantum information science.

## Conclusion

Introducing PT symmetry and non-Hermiticity into photonics has established novel ways of using gain, loss and their coupling to control light transport. Although the field has grown exponentially in recent years, there is still ample room for new insights and innovation. In the coming years, we expect the discovery of novel features in different physical systems configured to exploit PT symmetry and non-Hermitian dynamics, new PT-symmetric devices, and an efficient use of PT symmetry as solutions for fields suffering from loss. Exploiting non-Hermiticity and PT symmetry in already established and emerging fields of physics and photonics could dramatically improve the performance and the robustness of devices, and promises breakthroughs in both the fundamental and technological outcomes in photonics and materials science.

Received: 20 March 2017; Accepted: 29 January 2019;

Published online: 8 April 2019

## References

1. Livio, M. Why symmetry matters. Nature 490, 472-473 (2012).

2. Weinberg, S. Symmetry: a 'key to nature's secrets'. The New York Review of Books https://www.nybooks.com/articles/2011/10/27/symmetry-key-naturessecrets/ (2011).

3. Kato, T. Perturbation Theory of Linear Operators (Springer, 1966).

4. Teller, E. J. The crossing of potential surfaces. J. Phys. Chem. 41, 109–116 (1937).

5. Berry, M. V. M. The adiabatic phase and Pancharatnam's phase for polarized light. J. Mod. Opt. 34, 1401-1407 (1987).

6. Yarkony, D. R. Diabolical conical intersections. Rev. Mod. Phys. 68, 985–1013 (1996).

7. Zhu, J. et al. On-chip single nanoparticle detection and sizing by mode splitting in an ultrahigh-Q microresonator. Nat. Photon. 4, 46–49 (2010).

8. Özdemir, Ş. K. et al. Highly sensitive detection of nanoparticles with a self-referenced and self-heterodyned whispering-gallery Raman microlaser. Proc. Natl Acad. Sci. USA 111, E3836–E3844 (2014).

9. Bender, C. M. & Boettcher, S. Real spectra in non-hermitian hamiltonians having PT symmetry. Phys. Rev. Lett. 80, 5243–5246 (1998).

10. El-Ganainy, R., Makris, K. G., Christodoulides, D. N. & Musslimani, Z. H. Theory of coupled optical PT-symmetric structures. Opt. Lett. 32, 2632–2634 (2007).

11. Makris, K. G., El-Ganainy, R., Christodoulides, D. N. & Musslimani, Z. H. Beam dynamics in PT symmetric optical lattices. Phys. Rev. Lett. 100, 103904 (2008).

12. Klaiman, S., Günther, U. & Moiseyev, N. Visualization of branch points in PT-symmetric waveguides. Phys. Rev. Lett. 101, 080402 (2008).

13. Rüter, C. E. et al. Observation of parity–time symmetry in optics. Nat. Phys. 6, 192–195 (2010).

14. Guo, A. et al. Observation of PT-symmetry breaking in complex optical potentials. Phys. Rev. Lett. 103, 093902 (2009).

15. Brandstetter, M. et al. Reversing the pump dependence of a laser at an exceptional point. Nat. Commun. 5, 4034 (2014).

16. Peng, B. et al. Loss-induced suppression and revival of lasing. Science 346, 328–332 (2014).

17. Ramezani, H., Kottos, T., El-Ganainy, R. & Christodoulides, D. N. Unidirectional nonlinear PT-symmetric optical structures. Phys. Rev. A 82, 43803 (2010).

18. Peng, B. et al. Parity-time-symmetric whispering-gallery microcavities. Nat. Phys. 10, 394–398 (2014).

19. Chang, L. et al. Parity-time symmetry and variable optical isolation in active-passive-coupled microresonators. Nat. Photon. 8, 524–529 (2014).

20. Lin, Z. et al. Unidirectional invisibility induced by PT-symmetric periodic structures. Phys. Rev. Lett. 106, 213901 (2011).

21. Regensburger, A. et al. Parity-time synthetic photonic lattices. Nature 488, 167–171 (2012).

22. Feng, L. et al. Experimental demonstration of a unidirectional reflectionless parity-time metamaterial at optical frequencies. Nat. Mater. 12, 108–113 (2013).

23. Hodaei, H., Miri, M.-A., Heinrich, M., Christodoulides, D. N. & Khajavikhan, M. Parity-time-symmetric microring lasers. Science 346, 975–978 (2014).

24. Feng, L., Wong, Z. J., Ma, R.-M., Wang, Y. & Zhang, X. Single-mode laser by parity-time symmetry breaking. Science 346, 972–975 (2014).

25. Longhi, S. PT-symmetric laser absorber. Phys. Rev. A 82, 031801(R) (2010).

26. Chong, Y. D., Ge, L. & Stone, A. D. PT-symmetry breaking and laser-absorber modes in optical scattering systems. Phys. Rev. Lett. 106, 093902 (2011).

27. Sun, Y., Tan, W., Li, H., Li, J. & Chen, H. Experimental demonstration of a coherent perfect absorber with PT phase transition. Phys. Rev. Lett. 112, 143903 (2014).

28. Wiersig, J. Chiral and nonorthogonal eigenstate pairs in open quantum systems with weak backscattering between counterpropagating traveling waves. Phys. Rev. A 89, 012119 (2014).

29. Kim, M., Kwon, K., Shim, J., Jung, Y. & Yu, K. Partially directional microdisk laser with two Rayleigh scatterers. Opt. Lett. 39, 2423–2426 (2014).

30. Peng, B. et al. Chiral modes and directional lasing at exceptional points. Proc. Natl Acad. Sci. USA 113, 6845–6850 (2016).

31. Miao, P. et al. Orbital angular momentum microlaser. Science 353, 464–466 (2016).

32. Gao, T. et al. Observation of non-Hermitian degeneracies in a chaotic exciton-polariton billiard. Nature 526, 554–558 (2015).

33. Peng, P. et al. Anti-parity-time symmetry with flying atoms. Nat. Phys. 12, 1139–1145 (2016).

34. Zhang, Z. et al. Observation of Parity-Time Symmetry in Optically Induced Atomic Lattices. Phys. Rev. Lett. 117, 123601 (2016).

35. Xu, H., Mason, D., Jiang, L. & Harris, J. G. E. Topological energy transfer in an optomechanical system with exceptional points. Nature 537, 80–83 (2016).

36. Jing, H., Ozdemir, S. K., Lu, X.-Y., Zhang, J., Yang, L. & Nori, F. PT-symmetric phonon laser. Phys. Rev. Lett. 13, 053604 (2014).

37. Jing, H. et al. Optomechanically-induced transparency in parity-time-symmetric microresonators. Sci. Rep. 5, 9663 (2015).

38. Jing, H., Ozdemir, S. K., Lü, H. & Nori, F. High-order exceptional points in optomechanics. Sci. Rep. 7, 3386 (2017).

39. Zhang, J. et al. A phonon laser operating at an exceptional point. Nat. Photon. 12, 479 (2018).

40. Schönleber, D. W., Eisfeld, A. & El-Ganainy, R. Optomechanical interactions in non-Hermitian photonic molecules. New J. Phys. 18, 045014 (2016).

41. Zhu, X., Ramezani, H., Shi, C., Zhu, J. & Zhang, X. PT-symmetric acoustics. Phys. Rev. X 4, 031042 (2014).

42. Fleury, R., Sounas, D. L. & Alù, A. An invisible acoustic sensor based on parity-time symmetry. Nat. Commun. 6, 5905 (2015).

43. Ding, K., Ma, G., Xiao, M., Zhang, Z. Q. & Chan, C. T. Emergence, coalescence, and topological properties of multiple exceptional points and their experimental realization. Phys. Rev. X 6, 021007 (2016).

44. Schindler, J., Li, A., Zheng, M. C., Ellis, F. M. & Kottos, T. Experimental study of active LRC circuits with PT symmetries. Phys. Rev. A 84, 040101 (2011).

45. Benisty, H. et al. Implementation of PT symmetric devices using plasmonics: principle and applications. Opt. Express 19, 18004–18019 (2011).

46. Alaeian, H. & Dionne, J. A. Parity-time-symmetric plasmonic metamaterials. Phys. Rev. A 89, 033829 (2014).

47. Kang, M., Liu, F. & Li, J. Effective spontaneous PT-symmetry breaking in hybridized metamaterials. Phys. Rev. A 87, 053824 (2013).

48. Sun, Y., Tan, W., Li, H.-Q., Li, Jensen & Chen, H. Experimental demonstration of a coherent perfect absorber with PT phase transition. Phys. Rev. Lett. 112, 143903 (2014).

49. Kang, M., Chen, J. & Chong, Y. D. Chiral exceptional points in metasurfaces. Phys. Rev. A 94, 033834 (2016).

50. Xiao, S., Gear, J., Rotter, S. & Li, J. Effective PT-symmetric metasurfaces for subwavelength amplified sensing. New J. Phys. 18, 085004 (2016).

51. Fleury, R., Sounas, D. L. & Alù, A. Negative refraction and planar focusing based on parity-time symmetric metasurfaces. Phys. Rev. Lett. 113, 023903 (2014).

52. Ding, K., Zhang, Z. Q. & Chan, C. T. Coalescence of exceptional points and phase diagrams for one-dimensional PT-symmetric photonic crystals. Phys. Rev. B 92, 235310 (2015).

53. Zhen, B. et al. Spawning rings of exceptional points out of Dirac cones. Nature 525, 354–358 (2015).

54. Cerjan, A., Raman, A. & Fan, S. Exceptional contours and band structure design in parity-time symmetric photonic crystals. Phys. Rev. Lett. 116, 203902 (2016).

55. Konotop, V., Yang, J. & Zezyulin, D. A. Nonlinear waves in PT-symmetric systems. Rev. Mod. Phys. 88, 035002 (2016).

56. Suchkov, S. V. et al. Nonlinear switching and solitons in PT-symmetric photonic systems. Laser Photon. Rev. 10, 177–213 (2016).

57. Chen, Y., Snyder, A. W. & Payne, D. N. Twin Core Nonlinear Couplers with Gain and Loss. IEEE J. Quant. Electron. 28, 239–245 (1992).

58. Malomed, B. A., Peng, G. D. & Chu, P. L. Nonlinear-optical amplifier based on a dual-core fiber. Opt. Lett. 21, 330-332 (1996).

59. Visser, T. D., Blok, H. & Lenstra, D. Modal Analysis of a Planar Waveguide with Gain and Losses. IEEE J. Quant. Electron. 31, 1803–1810 (1995).

60. Ornigotti, M. & Szameit, A. Quasi-PT-symmetry in passive photonic lattices. J. Opt. 16, 065501 (2014).

61. Bittner, S. et al. PT symmetry and spontaneous symmetry breaking in a microwave billiard. Phys. Rev. Lett. 108, 024101 (2012).

62. Eichelkraut., T. et al. Mobility transition from ballistic to diffusive transport in non-Hermitian lattices. Nat. Commun. 4, 2533 (2013).

63. Lin, Z. et al. Unidirectional Invisibility Induced by PT-Symmetric Periodic Structures. Phys. Rev. Lett. 106, 213901 (2011).

64. Longhi, S. Invisibility in PT-symmetric complex crystals. J. Phys. A 44, 485302 (2011).

65. Feng, L. et al. Experimental demonstration of a unidirectional reflectionless parity-time metamaterial at optical frequencies. Nat. Mater. 12, 108–113 (2013).

66. Feng, L. et al. Demonstration of a large-scale optical exceptional point structure. Opt. Express 22, 1760-1767 (2014).

67. Chang, L. et al. Parity-time symmetry and variable optical isolation in active-passive-coupled microresonators. Nat. Photon. 8, 524–529 (2014).

68. Jalas, D. et al. What is—and what is not—an optical isolator. Nat. Photon. 7, 579–582 (2013).

69. Shi, Y., Yu, Z. & Fan, S. Limitations of nonlinear optical isolators due to dynamic reciprocity. Nat. Photon. 9, 388–392 (2015).

70. Regensburger, A. et al. Observation of defect states in PT-symmetric optical lattices. Phys. Rev. Lett. 110, 223902 (2013).

71. Wimmer, M. et al. Observation of optical solitons in PT-symmetric lattices. Nat. Commun. 6, 7782 (2015).

72. Poli, C., Bellec, M., Kuhl, U., Mortessagne, F. & Schomerus, H. Selective enhancement of topologically induced interface states in a dielectric resonator chain. Nat. Commun. 6, 6710 (2015).

73. Makris, K. G. et al. Constant-intensity waves and their modulation instability in non-Hermitian potentials. Nat. Commun. 6, 7257 (2015).

74. Makris, K. G., Brandstötter, A., Ambichl, P., Musslimani, Z. H. & Rotter, S. Wave propagation through disordered media without backscattering and intensity variations. Light Sci. Appl. 6, e17035 (2017).

75. Rivet, E. et al. Constant-pressure sound waves in non-Hermitian disordered media. Nat. Phys. 14, 942–947 (2018).

76. Peterman, K. Calculated spontaneous emission factor for double-heterostructure injection lasers with gain-induced waveguiding. IEEE J. Quant. Electron. 15, 566–570 (1979).

77. Hamel, W. A. & Woerdman, J. P. Observation of enhanced fundamental linewidth of a laser due to nonorthogonality of its longitudinal eigenmodes. Phys. Rev. Lett. 64, 1506–1509 (1990).

78. Wenzel, H., Bandelow, U., Wunsche, H. J. & Rehberg, J. Mechanisms of fast self pulsations in two-section DFB lasers. IEEE J. Quant. Electron. 32, 69–78 (1996).

79. Hodaei, H. et al. Parity-time-symmetric coupled microring lasers operating around an exceptional point. Opt. Lett. 40, 4955–4958 (2015).

80. Liertzer, M. et al. Pump-induced exceptional points in lasers. Phys. Rev. Lett. 108, 173901 (2012).

81. Miri, M.-A., LiKamWa, P. & Christodoulides, D. N. Large area single-mode parity-time-symmetric laser amplifiers. Opt. Lett. 37, 764–766 (2012).

82. Kim, K.-H. et al. Direct observation of exceptional points in coupled photonic-crystal lasers with asymmetric optical gains. Nat. Commun. 7, 13893 (2016).

83. Gao, Z. et al. Parity-time symmetry in coherently coupled vertical cavity laser arrays. Optica 4, 323-329 (2017).

84. Chong, Y. D., Ge, L., Cao, H. & Stone, A. D. Coherent perfect absorbers: time-reversed lasers. Phys. Rev. Lett. 105, 053901 (2010).

85. Wan, W. et al. Time-reversed lasing and interferometric control of absorption. Science 331, 889–892 (2011).

86. Wong, Z. J. et al. Lasing and anti-lasing in a single cavity. Nat. Photon. 10, 796–801 (2016).

87. He, L., Ozdemir, S. K. & Yang, L. Whispering gallery microcavity lasers. Laser Photon. Rev. 7, 60–82 (2013).

88. Lee, J. Y., Luo, X. & Poon, A. W. Reciprocal transmissions and asymmetric modal distributions in waveguide-coupled spiral-shaped microdisk resonators. Opt. Express 15, 14650-14666 (2007).

89. Wiersig, J. et al. Nonorthogonal pairs of copropagating optical modes in deformed microdisk cavities. Phys. Rev. A 84, 023845 (2011).

90. Redding, B. et al. Local chirality of optical resonances in ultrasmall resonators. Phys. Rev. Lett. 108, 253902 (2012).

91. Zhu, J., Özdemir, S. K., He, L. & Yang, L. Controlled manipulation of mode splitting in an optical microcavity by two Rayleigh scatterers. Opt. Express 18, 23535–23543 (2010).

92. Kim, M., Kwon, K., Shim, J., Jung, Y. & Yu, K. Partially directional microdisk laser with two Rayleigh scatterers. Opt. Lett. 39, 2423–2426 (2014).

93. Wiersig, J. Structure of whispering-gallery modes in optical microdisks perturbed by nanoparticles. Phys. Rev. A 84, 063828 (2011).

94. Miao, P. et al. Orbital angular momentum microlaser. Science 353, 464–467 (2016).

95. Dembowski, C. et al. Experimental observation of the topological structure of exceptional points. Phys. Rev. Lett. 86, 787–790 (2001).

96. Uzdin, R., Mailybaev, A. & Moiseyev, N. On the observability and asymmetry of adiabatic state flips generated by exceptional points. J. Phys. Math. Theor. 44, 435302 (2011).

97. Graefe, E.-M., Mailybaev, A. A. & Moiseyev, N. Breakdown of adiabatic transfer of light in waveguides in the presence of absorption. Phys. Rev. A 88, 033842 (2013).

98. Milburn, T. J. et al. General description of quasiadiabatic dynamical phenomena near exceptional points. Phys. Rev. A 92, 052124 (2015).

99. Doppler, J. et al. Dynamically encircling exceptional points in a waveguide: asymmetric mode switching from the breakdown of adiabaticity. Nature 537, 76–79 (2016).

100. Yoon, J. W. et al. Time-asymmetric loop around an exceptional point over the full optical communications band. Nature 562, 86–90 (2018).

101. Aspelmeyer, M., Kippenberg, T. J. & Marquardt, F. Cavity optomechanics. Rev. Mod. Phys. 86, 1391–1452 (2014).

102. Kepesidis, K. V. et al. PT-symmetry breaking in the steady state of microscopic gain-loss systems. New J. Phys. 18, 095003 (2016).

103. Zhang, J. et al. Giant nonlinearity via breaking parity-time symmetry: A route to low-threshold phonon diodes. Phys. Rev. B. 92, 115407 (2015).

104. Lü, X.-Y., Jing, H., Ma, J.-Y. & Wu, Y. PT-symmetry-breaking chaos in optomechanics. Phys. Rev. Lett. 114, 253601 (2015).

105. Lü, H., Ozdemir, S. K., Kuang, L.-M., Nori, F. & Jing, H. Exceptional points in random-defect phonon lasers. Phys. Rev. Appl. 8, 044020 (2017).

106. Wiersig, J. Enhancing the sensitivity of frequency and energy splitting detection by using exceptional points: Application to microcavity sensors for single-particle detection. Phys. Rev. Lett. 112, 203901 (2014).

107. Demange, G. & Graefe, E.-M. Signatures of three coalescing eigenfunctions. J. Phys. A 45, 025303 (2012).

108. Zhang, S., Yong, Z., Zhang, Y. & He, S. Parity-time symmetry breaking in coupled nanobeam cavities. Sci. Rep. 6, 24487 (2015).

109. Chen, P.-Y. & Jung, J. PT symmetry and singularity-enhanced sensing based on photoexcited graphene metasurfaces. Phys. Rev. Appl. 5, 064018 (2016).

110. Jouybari, S. N. Refractive index measurement using coupled microresonator laser based on parity-time symmetry breaking. J. Mod. Opt. 63, 798–803 (2016).

111. Liu, Z.-P. et al. Metrology with PT-symmetric cavities: enhanced sensitivity near the PT-phase transition. Phys. Rev. Lett. 117, 110802 (2016).

112. Chen, W., Ozdemir, S. K., Zhao, G., Wiersig, J. & Yang, L. Exceptional points enhance sensing in an optical microcavity. Nature 548, 192–196 (2017).

113. Hodaei, H. et al. Enhanced sensitivity at higher-order exceptional points. Nature 548, 187–191 (2017).

114. Lu, L., Joannopoulos, J. D. & Soljačić, M. Topological photonics. Nat. Photon. 8, 821–829 (2014).

115. Rudner, M. S. & Levitov, L. S. Topological transition in a non-hermitian quantum walk. Phys. Rev. Lett. 102, 065703 (2009).

116. Zeuner, J. M. et al. Observation of a topological transition in the bulk of a non-Hermitian system. Phys. Rev. Lett. 115, 040402 (2015).

117. Rechtsman, M. C. et al. Photonic Floquet topological insulators. Nature 496, 196–200 (2013).

118. Schomerus, H. Topologically protected midgap states in complex photonic lattices. Opt. Lett. 38, 1912–1914 (2013).

119. Hu, Y. C. & Hughes, T. L. Absence of topological insulator phases in non-Hermitian PT-symmetric Hamiltonians. Phys. Rev. B 84, 153101 (2011).

120. Yuce, C. PT symmetric Floquet topological phase. Eur. Phys. J. D 69, 184 (2015).

121. Malzard, S., Poli, C. & Schomerus, H. Topologically protected defect states in open photonic systems with non-Hermitian charge-conjugation and parity-time symmetry. Phys. Rev. Lett. 115, 200402 (2015).

122. Leykam, D., Bliokh, K. Y., Huang, C., Chong, Y. D. & Nori, F. Edge modes, degeneracies, and topological numbers in non-Hermitian systems. Phys. Rev. Lett. 118, 040401 (2017).

123. Weimann, S. et al. Topologically protected bound states in photonic parity-time-symmetric crystals. Nat. Mater. 16, 433–438 (2017).

124. Tame, M. S., McEnery, K. R., Ozdemir, S. K., Maier, S. A. & Kim, M. S. Quantum plasmonics. Nat. Phys. 9, 329–340 (2013).

125. Schomerus, H. Quantum noise and self-sustained radiation of PT-symmetric systems. Phys. Rev. Lett. 104, 233601 (2010).

126. Antonosyan, D. A., Solntsev, A. S. & Sukhorukov, A. A. Parity-time anti-symmetric parametric amplifier. Opt. Lett. 40, 4575–4578 (2015).

127. El-Ganainy, R., Dadap, J. I. & Osgood, R. M. Optical parametric amplification via non-Hermitian phase matching. Opt. Lett. 40, 5086–5089 (2015).

128. Gardas, B., Deffner, S. & Saxena, A. PT-symmetric slowing down of decoherence. Phys. Rev. A 94, 040101(R) (2016).

129. Kawabata, K., Ashida, Y. & Ueda, M. Information retrieval and criticality in parity-time-symmetric systems. Phys. Rev. Lett. 119, 190401 (2017).

130. Ashida, Y., Furukawa, S. & Ueda, M. Parity-time symmetric quantum critical phenomena. Nat. Commun. 8, 15791 (2017).

131. Özdemir, Ş. K., Zhu, J., He, L. & Yang, L. Estimation of Purcell factor from mode-splitting spectra in an optical microcavity. Phys. Rev. A 83, 033817 (2011).

## Acknowledgements

The authors thank A. Brandstötter, M. Horodynski, M. Kühmayer, K. Pichler, A. Schumer and H. Yilmaz, for their help with the figures and illustrations. Ş.K.Ö. is supported by Army Research Office (ARO) grant no. W911NF-16-1-0339, W911NF-18-1-0043, Air Force Office of Scientific Research (AFOSR) award no. FA9550-18-1-0235, National Science Foundation (NSF) (1807485), and by The Pennsylvania State University, Materials Research Institute (MRI). Ş.K.Ö. thanks Jan Mateo for his continuous support. S.R. is supported by the Austrian Science Fund (FWF) through projects no. SFB-NextLite F49-P10 and I 1142-N27, and by the Horizon 2020 programme of the European Union (RISE project NHQWAVE 691209). F.N. is supported in part by the MURI Center for Dynamic Magneto-Optics via AFOSR award no. FA9550-14-1-0040, ARO grant no. W911NF-18-1-0358), Asian Office of Aerospace Research and Development (AOARD) grant no. FA2386-18-1-4045, Japan Science and Technology Agency (JST) (Q-LEAP program, ImpACT program, and CREST grant no. JPMJCR1676), Japan Society for the Promotion of Science (JSPS) (JSPS-RFBR grant no. 17-52-50023, and JSPS-FWO grant no.VS.059.18N), RIKEN-AIST Challenge Research Fund, and the John Templeton Foundation.

## Author contributions

§.K.Ö. and S.R. wrote the manuscript with contributions from F.N. All authors read and agreed with the content and discussions in the manuscript.

## Competing interests

The authors declare no competing interests.

## Additional information

Reprints and permissions information is available at www.nature.com/reprints.

Correspondence should be addressed to Ş.K.Ö. and S.R.

Publisher's note: Springer Nature remains neutral with regard to jurisdictional claims in published maps and institutional affiliations.

© Springer Nature Limited 2019
