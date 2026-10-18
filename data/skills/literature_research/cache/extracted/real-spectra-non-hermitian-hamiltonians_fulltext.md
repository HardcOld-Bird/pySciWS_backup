# Real Spectra in Non-Hermitian Hamiltonians Having PT Symmetry

Carl M. Bender $^{1}$ and Stefan Boettcher $^{2,3}$

$^{1}$ Department of Physics, Washington University, St. Louis, MO 63130, USA

$^{2}$ Center for Nonlinear Studies, Los Alamos National Laboratory, Los Alamos, NM 87545, USA

$^{3}$ CTSPS, Clark Atlanta University, Atlanta, GA 30314, USA

(November 26, 2024)

The condition of self-adjointness ensures that the eigenvalues of a Hamiltonian are real and bounded below. Replacing this condition by the weaker condition of PT symmetry, one obtains new infinite classes of complex Hamiltonians whose spectra are also real and positive. These PT symmetric theories may be viewed as analytic continuations of conventional theories from real to complex phase space. This paper describes the unusual classical and quantum properties of these theories.

PACS number(s): 03.65-w, 03.65.Ge, 11.30.Er, 02.60.Lj

Several years ago, D. Bessis conjectured on the basis of numerical studies that the spectrum of the Hamiltonian $H = p^{2} + x^{2} + ix^{3}$ is real and positive [1]. To date there is no rigorous proof of this conjecture. We claim that the reality of the spectrum of H is due to PT symmetry. Note that H is invariant neither under parity P, whose effect is to make spatial reflections, $p \rightarrow -p$ and $x \rightarrow -x$ , nor under time reversal T, which replaces $p \rightarrow -p$ , $x \rightarrow x$ , and $i \rightarrow -i$ . However, PT symmetry is crucial. For example, the Hamiltonian $p^{2} + ix^{3} + ix$ has PT symmetry and our numerical studies indicate that its entire spectrum is positive definite; the Hamiltonian $p^{2} + ix^{3} + x$ is not PT-symmetric, and the entire spectrum is complex.

The connection between PT symmetry and positivity of spectra is simply illustrated by the harmonic oscillator $H = p^{2} + x^{2}$ , whose energy levels are $E_{n} = 2n + 1$ . Adding ix to H does not break PT symmetry, and the spectrum remains positive definite: $E_{n} = 2n + \frac{5}{4}$ . Adding -x also does not break PT symmetry if we define P as reflection about $x = \frac{1}{2}$ , $x \to 1 - x$ , and again the spectrum remains positive definite: $E_{n} = 2n + \frac{3}{4}$ . By contrast, adding ix - x does break PT symmetry, and the spectrum is now complex: $E_{n} = 2n + 1 + \frac{1}{2}i$ .

The Hamiltonian studied by Bessis is just one example of a huge and remarkable class of non-Hermitian Hamiltonians whose energy levels are real and positive. The purpose of this Letter is to understand the fundamental properties of such a theory by examining the class of quantum-mechanical Hamiltonians

$$
H = p ^ {2} + m ^ {2} x ^ {2} - (i x) ^ {N} \quad (N \text {   real }).\tag{1}
$$

As a function of N and mass $m^{2}$ we find various phases with transition points at which entirely real spectra begin to develop complex eigenvalues.

There are many applications of non-Hermitian PT-invariant Hamiltonians in physics. Hamiltonians rendered non-Hermitian by an imaginary external field have been introduced recently to study delocalization transitions in condensed matter systems such as vortex fluxline depinning in type-II superconductors [2], or even to study population biology [3]. Here, initially real eigenval ues bifurcate into the complex plane due to the increasing external field, indicating the unbinding of vortices or the growth of populations. We believe that one can also induce dynamic delocalization by tuning a physical parameter (here N) in a self-interacting theory.

Furthermore, it was found that quantum field theories analogous to the quantum-mechanical theory in Eq. (1) have astonishing properties. The Lagrangian $L = (\nabla \phi)^{2} + m^{2} \phi^{2} - g(i \phi)^{N}$ (N real) possesses PT invariance, the fundamental symmetry of local self-interacting scalar quantum field theory [4]. Although this theory has a non-Hermitian Hamiltonian, the spectrum of the theory appears to be positive definite. Also, L is explicitly not parity invariant, so the expectation value of the field $\langle \phi \rangle$ is nonzero, even when N = 4 [5]. Thus, one can calculate directly (using the Schwinger-Dyson equations, for example [6]) the (real positive) Higgs mass in a renormalizable theory such as $-g \phi^{4}$ or $ig \phi^{3}$ in which symmetry breaking occurs naturally (without introducing a symmetry-breaking parameter).

Replacing conventional $g\phi^{4}$ or $g\phi^{3}$ theories by $-g\phi^{4}$ or $ig\phi^{3}$ theories has the effect of reversing signs in the beta function. Thus, theories that are not asymptotically free become asymptotically free and theories that lack stable critical points develop such points. For example, PT-symmetric massless electrodynamics has a nontrivial stable critical value of the fine-structure constant $\alpha$ [7].

Supersymmetric non-Hermitian, PT-invariant Lagrangians have been examined [8]. It is found that the breaking of parity symmetry does not induce a breaking of the apparently robust global supersymmetry. The strong-coupling limit of non-Hermitian PT-symmetric quantum field theories has been investigated [9]; the correlated limit in which the bare coupling constants g and $-m^{2}$ both tend to infinity with the renormalized mass M held fixed and finite, is dominated by solitons. (In parity-symmetric theories the corresponding limit, called the Ising limit, is dominated by instantons.)

To elucidate the origin of such novel features we examine the elementary Hamiltonian (1) using extensive numerical and asymptotic studies. As shown in Fig. 1, when m = 0 the spectrum of H exhibits three distinct behaviors as a function of N. When $N \geq 2$ , the spectrum is infinite, discrete, and entirely real and positive. (This region includes the case N = 4 for which $H = p^{2} - x^{4}$ ; the spectrum of this Hamiltonian is positive and discrete and $\langle x \rangle \neq 0$ in the ground state because H breaks parity symmetry!) At the lower bound N = 2 of this region lies the harmonic oscillator. A phase transition occurs at N = 2; when 1 < N < 2, there are only a finite number of real positive eigenvalues and an infinite number of complex conjugate pairs of eigenvalues. In this region PT symmetry is spontaneously broken [10]. As N decreases from 2 to 1, adjacent energy levels merge into complex conjugate pairs beginning at the high end of the spectrum; ultimately, the only remaining real eigenvalue is the ground-state energy, which diverges as $N \to 1^{+}$ [11]. When $N \leq 1$ , there are no real eigenvalues. The massive case $m \neq 0$ is even more elaborate; there is a phase transition at N = 1 in addition to that at N = 2.

![](images/0252f4c397cfffe66db91da3b599d62477cf7cfd08612a9c5d2d73a2dfaefe87.jpg)
FIG. 1. Energy levels of the Hamiltonian $H = p^{2} - (ix)^{N}$ as a function of the parameter N. There are three regions: When $N \geq 2$ the spectrum is real and positive. The lower bound of this region, N = 2, corresponds to the harmonic oscillator, whose energy levels are $E_{n} = 2n + 1$ . When 1 < N < 2, there are a finite number of real positive eigenvalues and an infinite number of complex conjugate pairs of eigenvalues. As N decreases from 2 to 1, the number of real eigenvalues decreases; when $N \leq 1.42207$ , the only real eigenvalue is the ground-state energy. As N approaches $1^{+}$ , the ground-state energy diverges. For $N \leq 1$ there are no real eigenvalues.

The Schrödinger eigenvalue differential equation corresponding to the Hamiltonian (1) with m = 0 is

$$
- \psi^ {\prime \prime} (x) - (i x) ^ {N} \psi (x) = E \psi (x).\tag{2}
$$

Ordinarily, the boundary conditions that give quantized energy levels E are that $\psi(x) \to 0$ as $|x| \to \infty$ on the real axis; this condition suffices when 1 < N < 4. However, for arbitrary real N we must continue the eigenvalue problem for (2) into the complex-x plane. Thus, we replace the real-x axis by a contour in the complex plane along which the differential equation holds and we impose the boundary conditions that lead to quantization at the endpoints of this contour. (Eigenvalue problems on complex contours are discussed in Ref. [12].)

![](images/43637f09cc848076adf2fb4c10b6cc55a3670e2518cb32985dab0a439b44f62b.jpg)
FIG. 2. Wedges in the complex-x plane containing the contour on which the eigenvalue problem for the differential equation (2) for N = 4.2 is posed. In these wedges $\psi(x)$ vanishes exponentially as $|x| \to \infty$ . The wedges are bounded by Stokes lines of the differential equation. The center of the wedge, where $\psi(x)$ vanishes most rapidly, is an anti-Stokes line.

The regions in the cut complex-x plane in which $\psi(x)$ vanishes exponentially as $|x| \to \infty$ are wedges (see Fig. 2); these wedges are bounded by the Stokes lines of the differential equation [13]. The center of the wedge, where $\psi(x)$ vanishes most rapidly, is called an anti-Stokes line.

There are many wedges in which $\psi(x) \to 0$ as $|x| \to \infty$ . Thus, there are many eigenvalue problems associated with a given differential equation [12]. However, we choose to continue the eigenvalue equation (2) away from the conventional harmonic oscillator problem at N = 2. The wave function for N = 2 vanishes in wedges of angular opening $\frac{1}{2}\pi$ centered about the negative- and positive-real x axes. For arbitrary N the anti-Stokes lines at the centers of the left and right wedges lie at the angles

$$
\theta_ {\mathrm{left}} = - \pi + \frac {N - 2}{N + 2} \frac {\pi}{2} \quad \text { and } \quad \theta_ {\mathrm{right}} = - \frac {N - 2}{N + 2} \frac {\pi}{2}.\tag{3}
$$

The opening angle of these wedges is $\Delta = 2\pi/(N + 2)$ . The differential equation (2) may be integrated on any path in the complex-x plane so long as the ends of the path approach complex infinity inside the left wedge and the right wedge [14]. Note that these wedges contain the real-x axis when 1 < N < 4.

As N increases from 2, the left and right wedges rotate downward into the complex-x plane and become thinner. At $N = \infty$ , the differential equation contour runs up and down the negative imaginary axis and thus there is no eigenvalue problem at all. Indeed, Fig. 1 shows that the eigenvalues all diverge as $N \to \infty$ . As N decreases below 2 the wedges become wider and rotate into the upper-half x plane. At N = 1 the angular opening of the wedges is $\frac{2}{3}\pi$ and the wedges are centered at $\frac{5}{6}\pi$ and $\frac{1}{6}\pi$ . Thus, the wedges become contiguous at the positive-imaginary x axis, and the differential equation contour can be pushed off to infinity. Consequently, there is no eigenvalue problem when N = 1 and, as we would expect, the ground-state energy diverges as $N \to 1^{+}$ (see Fig. 1).

To ensure the numerical accuracy of the eigenvalues in Fig. 1, we have solved the differential equation (2) using two independent procedures. The most accurate and direct method is to convert the complex differential equation to a system of coupled, real, second-order equations which we solve using the Runge-Kutta method; the convergence is most rapid when we integrate along anti-Stokes lines. We then patch the two solutions together at the origin. We have verified those results by diagonalizing a truncated matrix representation of the Hamiltonian in Eq. (1) in harmonic oscillator basis functions.

Semiclassical analysis: Several features of Fig. 1 can be verified analytically. When $N \geq 2$ , WKB gives an excellent approximation to the spectrum. The novelty of this WKB calculation is that it must be performed in the complex plane. The turning points $x_{\pm}$ are those roots of $E + (ix)^{N} = 0$ that analytically continue off the real axis as N moves away from N = 2 (the harmonic oscillator):

$$
x _ {-} = E ^ {1 / N} e ^ {i \pi (3 / 2 - 1 / N)}, \quad x _ {+} = E ^ {1 / N} e ^ {- i \pi (1 / 2 - 1 / N)}.\tag{4}
$$

These turning points lie in the lower-half (upper-half) x plane in Fig. 2 when N > 2 (N < 2).

The leading-order WKB phase-integral quantization condition is $(n+1/2)\pi=\int_{x_{-}}^{x_{+}}dx\sqrt{E+(ix)^{N}}$ . It is crucial that this integral follows a path along which the integral is real. When N>2, this path lies entirely in the lower-half x plane and when N=2 the path lies on the real axis. But, when N<2 the path is in the upper-half x plane; it crosses the cut on the positive-imaginary

TABLE I. Comparison of the exact eigenvalues (obtained with Runge-Kutta) and the WKB result in (5).

<table><tr><td>N</td><td>n</td><td> $E_{\text{exact}}$ </td><td> $E_{\text{WKB}}$ </td><td>N</td><td>n</td><td> $E_{\text{exact}}$ </td><td> $E_{\text{WKB}}$ </td></tr><tr><td>3</td><td>0</td><td>1.1562</td><td>1.0942</td><td>4</td><td>0</td><td>1.4771</td><td>1.3765</td></tr><tr><td></td><td>1</td><td>4.1092</td><td>4.0894</td><td></td><td>1</td><td>6.0033</td><td>5.9558</td></tr><tr><td></td><td>2</td><td>7.5621</td><td>7.5489</td><td></td><td>2</td><td>11.8023</td><td>11.7689</td></tr><tr><td></td><td>3</td><td>11.3143</td><td>11.3042</td><td></td><td>3</td><td>18.4590</td><td>18.4321</td></tr><tr><td></td><td>4</td><td>15.2916</td><td>15.2832</td><td></td><td></td><td></td><td></td></tr></table>

axis and thus is not a continuous path joining the turning points. Hence, WKB fails when N < 2.

When $N \geq 2$ , we deform the phase-integral contour so that it follows the rays from $x_{-}$ to 0 and from 0 to $x_{+}$ : $(n + 1/2)\pi = 2\sin(\pi/N)E^{1/N+1/2}\int_{0}^{1} ds\sqrt{1-s^{N}}$ . We then solve for $E_{n}$ :

$$
E _ {n} \sim \left[ \frac {\Gamma (3 / 2 + 1 / N) \sqrt {\pi} (n + 1 / 2)}{\sin (\pi / N) \Gamma (1 + 1 / N)} \right] ^ {\frac {2 N}{N + 2}} \quad (n \to \infty).\tag{5}
$$

We perform a higher-order WKB calculation by replacing the phase integral by a closed contour that encircles the path in Fig. 2 (see Ref. [13,10]). See Table I.

It is interesting that the spectrum of the $|x|^{N}$ potential is like that of the $-(ix)^{N}$ potential. The leading-order WKB quantization condition (accurate for N > 0) is like Eq. (5) except that $\sin(\pi/N)$ is absent. However, as $N \to \infty$ , the spectrum of $|x|^{N}$ approaches that of the square-well potential $[E_{n} = (n + 1)^{2}\pi^{2}/4]$ , while the energies of the $-(ix)^{N}$ potential diverge (see Fig. 1).

Asymptotic study of the ground-state energy near N = 1: When N = 1, the differential equation (2) can be solved exactly in terms of Airy functions. The anti-Stokes lines at N = 1 lie at $30^{\circ}$ and at $150^{\circ}$ . We find the solution that vanishes exponentially along each of these rays and then rotate back to the real-x axis to obtain

$$
\psi_ {\mathrm{left,right}} (x) = C _ {1, 2} \operatorname{Ai} (\mp x e ^ {\pm i \pi / 6} + E e ^ {\pm 2 i \pi / 3}).\tag{6}
$$

We must patch these solutions together at x = 0 according to the patching condition $\frac{d}{dx}|\psi(x)|^{2}\big|_{x=0} = 0$ . But for real E, the Wronskian identity for the Airy function is

$$
\frac {d}{d x} | \mathrm{Ai} (x e ^ {- i \pi / 6} + E e ^ {- 2 i \pi / 3}) | ^ {2} \Big | _ {x = 0} = - \frac {1}{2 \pi}\tag{7}
$$

instead of 0. Hence, there is no real eigenvalue.

Next, we perform an asymptotic analysis for $N = 1 + \epsilon$ , $-\psi''(x) - (ix)^{1 + \epsilon}\psi(x) = E\psi(x)$ , and take $\psi(x) = y_0(x) + \epsilon y_1(x) + \mathrm{O}(\epsilon^2)$ as $\epsilon \to 0+$ . We assume that $E \to \infty$ as $\epsilon \to 0+$ , let $C_2 = 1$ in Eq. (6), and obtain

$$
y _ {0} (0) = \mathrm{Ai} (E e ^ {- 2 i \pi / 3}) \sim e ^ {i \pi / 6} E ^ {- 1 / 4} e ^ {\frac {2}{3} E ^ {3 / 2}} / (2 \sqrt {\pi}).\tag{8}
$$

We set $y_{1}(x) = Q(x)y_{0}(x)$ in the inhomogeneous equation $-y_{1}^{\prime \prime}(x) - ixy_{1}(x) - Ey_{1}(x) = ix\ln (ix)y_{0}(x)$ and get

$$
Q ^ {\prime} (0) = \frac {i}{y _ {0} ^ {2} (0)} \int_ {0} ^ {\infty} d x x \ln (i x) y _ {0} ^ {2} (x).\tag{9}
$$

Choosing $Q(0) = 0$ , we find that the patching condition at $x = 0$ gives $1 = 2\pi \epsilon |y_0(0)|^2 [Q'(0) + Q^{*\prime}(0)]$ , where we have used the zeroth-order result in Eq. (7). Using Eqs. (8) and (9) this equation becomes

$$
1 = \frac {\epsilon}{\sqrt {E}} e ^ {\frac {4}{3} E ^ {3 / 2}} \mathrm{Re} \left[ \frac {i}{y _ {0} ^ {2} (0)} \int_ {0} ^ {\infty} d x x \ln (i x) y _ {0} ^ {2} (x) \right].\tag{10}
$$

Since $y_{0}(x)$ decays rapidly as x increases, the integral in Eq. (10) is dominated by contributions near 0. Asymptotic analysis of this integral gives an implicit equation for E as a function of $\epsilon$ (see Table II):

$$
1 \sim \epsilon e ^ {\frac {4}{3} E ^ {3 / 2}} E ^ {- 3 / 2} [ \sqrt {3} \ln (2 \sqrt {E}) + \pi - (1 - \gamma) \sqrt {3} ] / 8.\tag{11}
$$

Behavior near N = 2: The most interesting aspect of Fig. 1 is the transition that occurs at N = 2. To describe quantitatively the merging of eigenvalues that begins when N < 2 we let $N = 2 - \epsilon$ and study the asymptotic behavior as $\epsilon \to 0+$ . (A Hermitian perturbation causes adjacent energy levels to repel, but in this case the non-Hermitian perturbation of the harmonic oscillator $(ix)^{2-\epsilon} \sim x^{2} - \epsilon x^{2}[\ln(|x| + \frac{1}{2}i\pi\operatorname{sgn}(x)]$ causes the levels to merge.) A complete description of this asymptotic study is given elsewhere [10].

The onset of eigenvalue merging is a phase transition that occurs even at the classical level. Consider the classical equations of motion for a particle of energy E subject to the complex forces described by the Hamiltonian (1). For m = 0 the trajectory $x(t)$ of the particle obeys $\pm dx[E + (ix)^{N}]^{-1/2} = 2dt$ . While E and dt are real, $x(t)$ is a path in the complex plane in Fig. 2; this path terminates at the classical turning points $x_{\pm}$ in (4).

When $N \geq 2$ , the trajectory is an arc joining $x_{\pm}$ in the lower complex plane. The motion is periodic; we have a complex pendulum whose (real) period T is

$$
T = 2 E ^ {\frac {2 - N}{2 N}} \cos \left[ \frac {(N - 2) \pi}{2 N} \right] \frac {\Gamma (1 + 1 / N) \sqrt {\pi}}{\Gamma (1 / 2 + 1 / N)}.\tag{12}
$$

At N = 2 there is a global change. For N < 2 a path starting at one turning point, say $x_{+}$ , moves toward but misses the turning point $x_{-}$ . This path spirals outward crossing from sheet to sheet on the Riemann surface, and eventually veers off to infinity asymptotic to the angle $\frac{N}{2-N}\pi$ . Hence, the period abruptly becomes infinite. The total angular rotation of the spiral is finite for all $N \neq 2$ and as $N \to 2^{+}$ , but becomes infinite as $N \to 2^{-}$ . The path passes many turning points as it spirals anticlockwise from $x_{+}$ . [The nth turning point lies at the angle $\frac{4n-N+2}{2N}\pi$ ( $x_{+}$ corresponds to n = 0).] As N approaches 2 from below, when the classical trajectory passes a new

TABLE II. Comparison of the exact ground-state energy E near N = 1 and the asymptotic results in Eq. (11). The explicit dependence of E on $\epsilon$ is roughly $E \propto (-\ln \epsilon)^{2/3}$ .

<table><tr><td>ε = N - 1</td><td> $E_{\text{exact}}$ </td><td>Eq. (11)</td></tr><tr><td>0.1</td><td>1.6837</td><td>2.0955</td></tr><tr><td>0.01</td><td>2.6797</td><td>2.9624</td></tr><tr><td>0.001</td><td>3.4947</td><td>3.6723</td></tr><tr><td>0.0001</td><td>4.1753</td><td>4.3013</td></tr><tr><td>0.00001</td><td>4.7798</td><td>4.8776</td></tr><tr><td>0.000001</td><td>5.3383</td><td>5.4158</td></tr><tr><td>0.0000001</td><td>5.8943</td><td>5.9244</td></tr></table>

turning point, there corresponds an additional merging of the quantum energy levels as shown in Fig. 1). This correspondence becomes exact in the limit $N \to 2^{-}$ and is a manifestation of Ehrenfest's theorem.

Massive case: The $m \neq 0$ analog of Fig. 1 exhibits a new transition at N = 1 (see Fig. 3). As N approaches 1 from above, the energy levels reemerge from the complex plane in pairs and at N = 1 the spectrum is again entirely real and positive. Below N = 1 the energies once again disappear in pairs, now including the ground state. As $N \rightarrow 0$ the infinite real spectrum reappears again. The massive case is discussed further in Ref. [10].

![](images/4688a378598db53850e0a37de5fd63beb895a5ae96309326ee974768b3223933.jpg)
FIG. 3. The $m \neq 0$ analog of Fig. 1. Note that transitions occur at N = 2 and N = 1.

We thank D. Bessis, H. Jones, P. Meisinger, A. Wightman, and Y. Zarmi for illuminating conversations. CMB thanks the Center for Nonlinear Studies, Los Alamos National Laboratory and STB thanks the Physics Department at Washington University for its hospitality. This work was supported by the U.S. Department of Energy.

[1] D. Bessis, private discussion. This problem originated from discussions between Bessis and J. Zinn-Justin, who was studying Lee-Yang singularities using renormalization group methods. An $i\phi^{3}$ field theory arises if one translates the field in a $\phi^{4}$ theory by an imaginary term.

[2] N. Hatano and D. R. Nelson, Phys. Rev. Lett. 77, 570 (1996), and Phys. Rev. B 56, 8651 (1997).

[3] D. R. Nelson and N. M. Shnerb, condmat/9708071.

[4] R. F. Streater and A. S. Wightman, PCT, Spin & Statistics, and all that (Benjamin, New York, 1964). There is no analog of the $\mathcal{C}$ operator in quantum systems having

one degree of freedom.

[5] C. M. Bender and K. A. Milton, Phys. Rev. D 55, R3255 (1997). There is a factor of 2 error in Eqs. (15) and (16).

[6] C. M. Bender and K. A. Milton (submitted).

[7] C. M. Bender and K. A. Milton (submitted).

[8] C. M. Bender and K. A. Milton, Phys. Rev. D 57, 3595 (1998).

[9] C. M. Bender, S. Boettcher, H. F. Jones, and P. N. Meisinger, Phys. Rev. D (submitted).

[10] C. M. Bender, S. Boettcher, and P. N. Meisinger, Phys. Rev. D (submitted).

[11] It is known that the spectrum of $H = p^2 - ix$ is null. See I. Herbst, Commun. Math. Phys. 64, 279 (1979).

[12] C. M. Bender and A. Turbiner, Phys. Lett. A 173, 442 (1993).

[13] C. M. Bender and S. A. Orszag, Advanced Mathematical Methods for Scientists and Engineers (McGraw-Hill, New York, 1978).

[14] In a Euclidean path integral representation for a quantum field theory, the (multiple) integration contour follows the same anti-Stokes lines. See Ref. [5].
