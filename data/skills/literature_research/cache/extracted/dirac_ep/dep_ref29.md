# Analysis of Dirac exceptional points and their isospectral Hermitian counterparts

Jose H. D. Rivero, $^{1}$ Liang Feng, $^{1}$ and Li Ge $^{2,3,*}$

$^{1}$ Department of Materials Science and Engineering,

University of Pennsylvania, Philadelphia, PA 19104, USA $^{2}$ Department of Physics and Astronomy, College of Staten Island, CUNY, Staten Island, NY 10314, USA $^{3}$ The Graduate Center, CUNY, New York, NY 10016, USA

Recently, a Dirac exceptional point (EP) was reported in a non-Hermitian system. Unlike a Dirac point in Hermitian systems, this Dirac EP has coalesced eigenstates in addition to the degenerate energy. Also different from a typical EP, the two energy levels connected at this Dirac EP remain real in its vicinity and display a linear instead of square root dispersion, forming a tilted Dirac cone in the hybrid space consisting of a momentum dimension and a synthetic dimension for the strength of non-Hermiticity. In this report, we first present simple three-band and two-band matrix models with a Dirac EP, where the linear dispersion of the tilted Dirac cone can be expressed analytically. Importantly, our analysis also reveals that there exist Hermitian and non-Hermitian systems that have the same (real-valued) energy spectrum in their entire parameter space, with the exception that one or more degeneracies in the former are replaced by Dirac EPs in the later. Finally, we show the existence of an imaginary Dirac cone with an EP at its center.

## I. INTRODUCTION

Dirac points, the degeneracies at the center of Dirac cones, give graphene $[1, 2]$ and other Dirac matters $[3]$ their unusual electronic properties. Their signature, i.e., the linear dispersion around a Dirac point in momentum space, signals a massless fermion that differs significantly from a classical object with the quadratic kinetic energy relation. Similar to all other degeneracies in Hermitian systems, a Dirac point corresponds to two different quantum eigenstates at the same energy level and hence is an example of a diabolic point $[4]$ .

An exceptional point (EP) [5, 6], on the other hand, features coalesced eigenstates at a degenerate energy level. Its existence is a unique feature of non-Hermitian systems, which arise when a physical system is represented by a Hamiltonian with partial degrees of freedom or by another operator that describes the exchange of energy or particles with the environment (such as the scattering matrix [7–11]). This approach has been adopted in studies ranging from nuclear decay [12] to photon lifetime in optical microcavities [13], yielding insightful results as well as providing new directions of research [13–16]. For example, the motion of eigenfrequencies or resonances in the complex plane can be quite unusual in the vicinity of an EP, which results in intriguing behaviors such as gain-suppressed lasing [17–20] and loss-induced lasing [21].

While a degeneracy in a Hermitian system cannot be an exceptional point, a degeneracy in a non-Hermitian system is not necessarily an EP. Such a non-EP degeneracy can occur accidentally or by symmetry, similar to the mechanisms in a Hermitian system. Some examples include the zero modes in a non-Hermitian flatband $[22]$ and Dirac points constructed in non-Hermitian lattices [23, 24]. For the degeneracies that are indeed EPs, traditionally one associates their perturbative dependence on a system parameter with a fractional exponent [e.g., a square root for an EP of order 2 (EP2)], which implies a stronger response and hence potentially enhanced sensitivity [25–27]. Furthermore, if an EP has its roots in a non-Hermitian symmetry, such as parity-time (PT) [16, 28–30], anti-PT [31–35], or particle-hole [36–39] symmetries, the non-Hermitian system experiences a corresponding spontaneous symmetry breaking or restoration across the EP. This phenomenon is usually accompanied by dramatic changes in the energy spectrum: It may transition from real to complex or even imaginary and vice-versa, which is used as one experimental signature of the EP itself [32, 40–42].

Therefore, the recent, accidental finding of a Dirac EP [43] in a one-dimensional periodic PT-symmetric system came as a surprise: Two energy bands connected by this Dirac EP display a linear and conical “dispersion” in a two-dimensional hybrid space, as a function of both momentum and the non-Hermitian parameter given by the optical gain and loss strength. In addition, the two energy bands remain real in the vicinity of the Dirac EP, without undergoing a spontaneous symmetry breaking. While a three-band matrix model was introduced in Ref. [43] to capture the latter, the linear and conical dispersion at this Dirac EP remains to be elucidated. The differences between Dirac points, Dirac EPs, and conventional EPs are summarized in Table. I. In the last row, state flip is marked “possible” instead of “yes” when a conventional EP2 is encircled, due to the non-adiabatic transition from the low-gain/high-loss state to the high-gain/low-loss state [6]. Such non-adiabatic behaviors caused by the different modal gain (or loss) are absent in the vicinity of a Dirac EP thanks to its real spectrum.

In this Report, we first introduce a revised three-band model and show explicitly the linear dispersion of the (tilted) Dirac cone centered at the Dirac EP, via a perturbative treatment. Our approach is different from the standard procedure of using the alternating Puiseux series for a conventional EP $[44, 45]$ , which is inapplicable at a Dirac EP as we will show. Because the Dirac EP connects two bands instead of three, we further reduce this three-band model to a two-band non-Hermitian Hamiltonian where the Dirac EP can be found. Using a slightly different approach, we find, interestingly, that while the same linear and conical dispersion is produced in the hybrid dimensions, the Dirac point becomes a diabolic point instead, with two distinct eigenstates. We show that this dilemma can be resolved by revisiting the three-band model. More importantly, this comparison has a far-reaching implication: There exist Hermitian and non-Hermitian systems that have the same (real-valued) energy spectrum in their entire parameter space, with the exception that one or more degeneracies in the former are replaced by Dirac EPs in the later. Finally, we show the existence of an imaginary Dirac cone with an EP at its center.

TABLE I. Comparison of three types of degeneracies.

<table><tr><td></td><td>Dirac Points</td><td>Dirac EPs</td><td>Conventional EP2s</td></tr><tr><td>Algebraic multiplicity</td><td>2</td><td>2</td><td>2</td></tr><tr><td>Geometric multiplicity</td><td>2</td><td>1</td><td>1</td></tr><tr><td>Coalescence of wave functions</td><td>No</td><td>Yes</td><td>Yes</td></tr><tr><td>Locally real spectrum</td><td>Yes</td><td>Yes</td><td>No</td></tr><tr><td>Branch cuts at degeneracy</td><td>No</td><td>No</td><td>Yes</td></tr><tr><td>Node type in energy (real part)</td><td>Point</td><td>Point</td><td>Line or surface</td></tr><tr><td>Energy dispersion</td><td>Conical</td><td>Conical</td><td>Square roots</td></tr><tr><td>State flip when encircled</td><td>No</td><td>No</td><td>Possible</td></tr></table>

## II. THREE-BAND MODEL

The systems exhibiting a Dirac EP we study originate from the following Schrödinger equation

$$
i \frac {d}{d t} \psi (x, t) = [ - \partial_ {x} ^ {2} + V (x) ] \psi (x, t),\tag{1}
$$

where time, position, and potential are in their dimensionless forms and $\hbar = 1$ . $V(x) = V_{0}(\cos x + i\tau \sin x) (\tau \geq 0)$ is a complex potential with the spatial period $a = 2\pi$ . It is PT-symmetric and satisfies $V(x) = V^{*}(-x)$ [13]. The asterisk denotes complex conjugation and represents time reversal, and the imaginary part of the potential represents optical gain and loss with strength $\tau$ [13].

The Dirac EP in this system is found at the point contact between the second and third bands in the first Brillouin zone, where $\tau = 1$ and k = 0. To gain analytical understanding of this Dirac EP, Ref. [43] first expanded the Bloch wave function in the plane-wave basis, i.e.,

$$
\psi_ {n k} (x, t) = e ^ {i k x - i \omega t} \sum_ {m \in \mathbb {Z}} a _ {m} e ^ {i m x},\tag{2}
$$

and derived the Bloch Hamiltonian

$$
\begin{array}{r l} H _ {k} = \sum_ {m \in \mathbb {Z}} (m + k) ^ {2} | m \rangle \langle m | & + t _ {-} | m \rangle \langle m + 1 | \\ & + t _ {+} | m \rangle \langle m - 1 | \end{array}\tag{3}
$$

that satisfies

$$
H _ {k} \Psi_ {n k} (m) = \omega_ {n k} \Psi_ {n k} (m).\tag{4}
$$

Here $n = 1, 2, \ldots$ is the band index, $\omega_{nk}$ is energy of the nth band when the momentum is k, $\Psi_{nk}(m) = [\ldots, a_{-1}, a_0, a_1, \ldots]^T$ is the wave function in the plane-wave basis, and $t_{\pm} = V_0(1 \pm \tau)/2 \in \mathbb{R}$ . Then by truncating $H_k$ given by Eq. (3) and keeping only the m = -1, 0 and 1 block, Ref. [43] showed analytically that indeed the second and third bands remain real-valued when we increase $\tau$ across 1, i.e., the system does not experience a spontaneous PT breaking by going through the Dirac EP. While this analysis was only performed for k = 0 and the truncation turns out to be a crude approximation, it shone light on how a Dirac EP could be constructed.

Below we first use the insight from this truncation to introduce a revised three-band model, where a Dirac EP exists and where the linear dispersion of its (tilted) Dirac cone can be expressed analytically. In this model, we allow k to be a (small) free parameter in addition to the gain and loss strength $\tau$ , and the three-band non-Hermitian Hamiltonian is given by

$$
H ^ {(3)} = \left( \begin{array}{c c c} 1 - 2 k & t _ {-} & 0 \\ t _ {+} & 0 & t _ {-} \\ 0 & t _ {+} & 1 + 2 k \end{array} \right)\tag{5}
$$

with the asymmetric couplings $t_{\pm}$ introduced in Eq. (3). The eigenvalues $\omega_{i}$ of $H^{(3)}$ are the solutions of the characteristic polynomial

$$
\omega_ {i} (1 - \omega_ {i}) ^ {2} + 2 t ^ {2} (1 - \omega_ {i}) - 4 k ^ {2} \omega_ {i} = 0,\tag{6}
$$

where $t^{2} \equiv t_{-}t_{+} = V_{0}^{2}(1 - \tau^{2})/4$ . This cubic equation can be solved analytically, but the resulting expressions for $\omega_{i}$ 's are rather complicated (e.g., with a square root inside a cubic root) and do not help us understand the properties of the Dirac EP. We could perform a Taylor expansion of these expressions for $\omega_{i}$ 's, but a much simpler approach is to expand Eq. (6) directly, which gives the same results. We do note that being a cubic equation with real coefficients, Eq. (6) indicates that the three energy bands are either all real or one real plus a complex conjugate pair. Therefore, it does not exclude the possibility of a spontaneous PT breaking, which nevertheless does not take place at the Dirac EP.

It is straightforward to verify that this EP has energy $\omega = 1 \equiv \omega_{0}$ (again $\hbar = 1$ ) and exists at $\tau = 1, k = 0$ . Its coalesced eigenstate is given by $[a_{-1}, a_{0}, a_{1}]^{T} = [0, 0, 1]^{T} \equiv \Psi_{0}$ . This EP is the point contact of the second and third bands that form a tilted Dirac cone (see Fig. 1), similar to the original Hamiltonian $H_{k}$ shown in Ref. [43].

![](images/8609d77638840c14ae3ca24aeb2c80d96d23089c4008fa8ae7067417ec774c6d.jpg)
FIG. 1. Tilted Dirac cone of $H^{(3)}$ centered at a Dirac EP. k is momentum and $\Delta\tau$ is the change of the non-Hermitian strength from its value (i.e., 1) at the Dirac EP. $V_{0}=1$ is used in the couplings $t_{\pm}$ . The intersection of the Dirac cone and two parallel planes given by $\omega=1-s\Delta\tau\pm d$ are also shown, where $\omega_{0}=1$ is the energy at the Dirac EP, $s=\sqrt{2}-1$ , and d=1/40. The arrow shows the diagonal direction used in Fig. 2(b).

To derive the dispersion relation near this Dirac EP, we write $\omega_{i} \equiv \omega_{0} + \Delta \omega_{i} (|\Delta \omega_{i}| \ll \omega_{0})$ and study how $\Delta \omega_{i}$ depends on the two small parameters $\Delta \tau \equiv \tau - 1$ and $k$ .

Here we take the advantage of knowing that the dispersion is linear and conical, which implies that $\Delta\omega_{i}$ is of the same order as k and $\Delta\tau$ . Consequently, Eq. (6) becomes

$$
\Delta \omega_ {i} ^ {2} - 2 t ^ {2} \Delta \omega_ {i} - 4 k ^ {2} = 0,\tag{7}
$$

to the leading order of $\Delta\omega_{i}$ (i.e., $\Delta\omega_{i}^{2}$ ), where $t^{2} \approx -(V_{0}^{2}/2)\Delta\tau$ is also of the same order as $\Delta\tau$ (with $V_{0}$ chosen to be of order unity). We then find

$$
\Delta \omega_ {i} = t ^ {2} \pm \sqrt {t ^ {4} + 4 k ^ {2}}\tag{8}
$$

or

$$
\Delta \omega_ {i} \approx (- \alpha \pm \sqrt {4 + \alpha^ {2}}) k\tag{9}
$$

in any arbitrary direction $\Delta\tau=(2\alpha/V_{0}^{2})k(\alpha\in\mathbb{R})$ from the origin of the $\Delta\tau-k$ plane.

The square root in Eq. (9) directly captures the linear and conical dispersion of the Dirac cone shown in Fig.1, with the linear term in front of it explaining the tilt of this Dirac cone. We also note that this expression is exact when $\Delta\tau=0$ (i.e., along the k direction from the Dirac EP), which can be checked easily by setting the second term in Eq. (6) to be zero. In Fig. 2, we show the comparison of Eq. (9) and the actual band energies along two other directions, i.e., the $\Delta\tau$ axis and the one with $\alpha=2.5$ (diagonal direction in Fig. 1). We observe that the band closer to $\omega=1$ is better approximated by Eq. (9), which will be explained using the equivalent two-band model below where the analysis becomes easier.

We also note that our approach is different from the perturbative expansion used at a conventional EP, which utilizes alternating Puiseux series $[45]$ with fractional powers of a small parameter. In fact, this standard treatment is inapplicable at a Dirac EP as we show below. This approach calculates the perturbative corrections to the eigenstates and their energies when the Hamiltonian changes from $H_{0}$ to $H = H_{0} + \epsilon H_{1} (\epsilon \ll 1)$ , and when an EP is the result of two coalescing eigenstates (i.e., of multiplicity 2) [44], the alternating Puiseux series assume integer and half-integer powers of $\epsilon$ , i.e.,

$$
\omega_ {\pm} = \omega_ {0} \pm \epsilon^ {1 / 2} \omega_ {1} + \epsilon \omega_ {2} \pm \epsilon^ {3 / 2} \omega_ {3} + \dots\tag{10}
$$

$$
\Psi_ {\pm} = \Psi_ {0} \pm \epsilon^ {1 / 2} \Psi_ {1} + \epsilon \Psi_ {2} \pm \epsilon^ {3 / 2} \Psi_ {3} + \dots\tag{11}
$$

Here $\omega_{\pm}$ are the energies of the two eigenstates $\Psi_{\pm}$ that become coalesced at the EP, with energy $\omega_{0}$ and wave function $\Psi_{0}$ . Even if we ignore the $\epsilon^{1/2}$ term, Eq. (10) indicates clearly that $\omega_{\pm}$ have the same linear dependence on $\epsilon$ . In other words, they stay the same to the linear order and hence cannot form a Dirac cone in this perturbative expansion.

## III. TWO-BAND MODEL

Since the Dirac EP only connects two bands of $H^{(3)}$ [i.e., the second and third bands; represented by the two signs in Eq. (8)], it should be possible to reduce $H^{(3)}$ to a two-band Hamiltonian, which not only simplifies the understanding of the Dirac EP but also provides another instance where it exists. To this end, we first write down the time-dependent Schrödinger equation corresponding to $H^{(3)}$ :

![](images/9f1e82648515745d33d00ff6dc2989055af3aa691e3dc2a55d9752f05cab7617.jpg)

![](images/26e18df1fe67298296767c210111ee8360b371731194cfda53f512036eb0d7dc.jpg)
FIG. 2. Linear dispersion of the tilted Dirac cone along (a) $\Delta\tau$ and (b) the diagonal direction given by $\Delta\tau = 5k$ in Fig. 1. Solid and dashed lines show the actual and approximated band energies, which cannot be distinguished by eye for the band closer to $\omega = 1$ .

$$
i \frac {d}{d t} a _ {- 1} = (1 - 2 k) a _ {- 1} + t _ {-} a _ {0}\tag{12}
$$

$$
i \frac {d}{d t} a _ {0} = t _ {+} a _ {- 1} + t _ {-} a _ {1}\tag{13}
$$

$$
i \frac {d}{d t} a _ {1} = (1 + 2 k) a _ {1} + t _ {+} a _ {0}\tag{14}
$$

Using $a_{0} \propto e^{-i\omega_{i}t}$ in an eigenstate with energy $\omega_{i}$ , we obtain

$$
a _ {0} = (t _ {+} a _ {- 1} + t _ {-} a _ {1}) / \omega_ {i}\tag{15}
$$

from Eq. (13) and use this expression to eliminate $a_{0}$ in Eqs. (12) and (14). The result is a two-band Hamiltonian

$$
H _ {a} = (1 + t ^ {2} / \omega_ {i}) \mathbf {1} + \left( \begin{array}{c c} - 2 k & t _ {-} ^ {2} / \omega_ {i} \\ t _ {+} ^ {2} / \omega_ {i} & 2 k \end{array} \right),\tag{16}
$$

where 1 is the identity matrix. Because no approximations have been used in deriving $H_{a}$ , it has the same eigenvalues $\omega_{i}$ 's as $H^{(3)}$ . Furthermore, we note that the eigenvalue $\omega_{i}$ appears in the Hamiltonian $H_{a}$ itself, and hence this problem can be treated as a nonlinear eigenvalue problem [46]. Nevertheless, if we replace $\omega_{i}$ by $\omega_{0}=1$ in $H_{a}$ itself near the expected EP, i.e.,

$$
H _ {a} \to H _ {a} ^ {\prime} = (1 + t ^ {2}) \mathbf {1} + \left( \begin{array}{c c} - 2 k & t _ {-} ^ {2} \\ t _ {+} ^ {2} & 2 k \end{array} \right),\tag{17}
$$

we immediately find that $H_{a}^{\prime}$ takes the Jordan normal form when k = 0 and $\Delta\tau = 0$ , at which $t_{-}$ vanishes as well that leads to an EP with energy $\omega_{0} = 1$ . It is easy to check that the two eigenvalues of $H_{a}^{\prime}$ are the same as those given by Eq. (8), and hence it gives the same linear and conical dispersion relation (9).

If we have not replaced $\omega_{i}$ by $\omega_{0}=1$ in Eq. (16), we can also express the nonlinear eigenvalues of $H_{a}$ in a self-consistent way, i.e.,

$$
\omega_ {i} = (1 + t ^ {2} / \omega_ {i}) \pm \sqrt {4 k ^ {2} + t ^ {4} / \omega_ {i} ^ {2}},\tag{18}
$$

which is equivalent to Eq. (6). It is then clear that the errors introduced by taking $\omega_{i} \approx \omega_{0}$ in Eq. (17) only originate from the $t^{2}$ , $t^{4}$ terms in Eq. (18). They are partially cancelled (increased) in the solution with the “−” (“+”) sign in Eq. (18), which is also closer to (further from) the energy at the Dirac EP. This is exactly what we have observed in Fig. 2.

## IV. ISOSPECTRAL HERMITIAN AND NON-HERMITIAN SYSTEMS

As mentioned in the introduction, there is more than one way to reduce $H^{(3)}$ to a two-band Hamiltonian. Because the amplitudes $a_{\pm1}$ are coupled indirectly through $a_{0}$ in $H^{(3)}$ , one may seek eigenstates in the forms of symmetric and antisymmetric superpositions of $a_{\pm1}$ , weighted by the couplings $t_{\mp}$ :

$$
a _ {\pm} = t _ {+} a _ {- 1} \pm t _ {-} a _ {1}.\tag{19}
$$

Similar to the first approach above, we also eliminate $a_{0}$ and express it in terms of $a_{\pm}$ . The resulting two-band Hamiltonian for $a_{\pm}$ is then found by multiplying Eqs. (12) and (14) by $t_{+}, t_{-}$ respectively and taking the summation and difference of the results:

$$
H _ {b} = \left( \begin{array}{c c} 1 + 2 t ^ {2} / \omega_ {i} & - 2 k \\ - 2 k & 1 \end{array} \right).\tag{20}
$$

Again, by approximating $\omega_{i}$ in $H_{b}$ by $\omega_{0}=1$ at the Dirac EP, i.e.,

$$
H _ {b} \to H _ {b} ^ {\prime} = \left( \begin{array}{c c} 1 + 2 t ^ {2} & - 2 k \\ - 2 k & 1 \end{array} \right),\tag{21}
$$

it is straightforward to show that the two eigenvalues of $H_{b}^{\prime}$ are given by Eq. (8), and we recover the linear and conical dispersion relation (9). However, $H_{b}^{\prime}$ (and $H_{b}$ ) becomes an identity matrix at where we expect to find the Dirac EP, i.e., $k = \Delta\tau = 0$ , and hence the degeneracy $\omega_{0} = 1$ is a diabolic point instead of an EP, with two distinct eigenstates $[0,1]^{T}$ and $[1,0]^{T}$ . This should not be surprising, however, because $H_{b}^{\prime}$ is a real symmetric matrix, and hence it is Hermitian and cannot have an EP.

This apparent contradiction can be easily resolved by realizing that one step leading to $H_{b}$ [i.e., multiplying Eq. (14) by $t_{-}$ ] fails when $t_{-}=0$ (or equivalently, $\Delta\tau=1$ ) where the degeneracy exists. Therefore, $H_{b}$ is not equivalent to $H_{a}$ at $\Delta\tau=1$ and it is allowed to differ from $H_{a}$ , i.e., having a diabolic point instead of an EP.

More importantly, this comparison reveals a far-reaching implication: There exist Hermitian and non-Hermitian systems that have the same (real) energy spectrum in their entire parameter space, with the exception that one or more degeneracies in the former are replaced by Dirac EPs in the latter.

This observation holds for the pair of linear Hamiltonians $H_{a}^{\prime}, H_{b}^{\prime}$ in the entire $\tau-k$ parameter space. Here $H_{b}^{\prime}$ should be treated as given, and hence the illegitimacy mentioned above from $H^{(3)}$ to $H_{b}$ at the Dirac EP is irrelevant. Although there is only one degeneracy (a diabolic point) replaced by one EP in our examples here, cases with more or even all degeneracies replaced by EPs can be trivially constructed. For example, one can generate a series of two-band Hamiltonians similar to $H_{a}^{\prime}$ but with different energy shifts (i.e., replacing $\omega_{0}=1$ in its diagonal elements by an increasing series $\Delta_{m}$ 's) and then stack them to form a block diagonal non-Hermitian Hamiltonian $H_{A}^{\prime}$ . By following the same process but using $H_{b}^{\prime}$ instead, we end up with another block diagonal Hamiltonian $H_{B}^{\prime}$ which is Hermitian. It is easy to see that they have the same energy eigenvalues throughout the parameter space $\tau-k$ , with $\Delta_{m}$ 's being diabolic points in $H_{B}^{\prime}$ but EPs in $H_{A}^{\prime}$ .

We note that the isospectral property between a Hermitian and a non-Hermitian system we report here is stronger than that found in Ref. [43], where this equivalence was only established in the PT-symmetric regime of a linear non-Hermitian system and away from its EP. The more general claim here is made possible partly by the elimination of the conventional EPs of the system studied in Ref. [43], which occur at the edge of the Brillouin zone when $\tau = 1$ . Near these conventional EPs, the band energies become complex along the $\Delta\tau$ direction and hence lose the equivalence to their Hermitian counterparts. If we restrict our discussion to the one-dimensional parameter space along k with $\tau$ fixed at 1, one may attempt to claim that this system also have the more general isospectral property reported here: Its entire band structure is real valued in the first Brillouin zone and identical to that of a Hermitian system with $V(x) = 0$ ; its EPs, both the conventional ones at the band edge and the unconventional ones at the center of the Brillouin zone (including the Dirac EP), are replaced by degeneracies in the Hermitian system. However, one quickly realizes that with $V(x) = 0$ , this “crystal” is just free space with a single dispersion relation $\omega = k^{2}$ . Therefore, its degeneracies in the band analysis are artifacts of applying the periodic boundary condition to a “unit cell” of an arbitrary length, resulting in the folding of this single energy relation into the first Brillouin zone.

## V. LINEAR AND CONICAL DISPERSION

![](images/15235ae3a23e7b5dfd3f029857e1d5e9b3c414fd0d6f9bb526f0110bde6aaf80.jpg)
FIG. 3. Energy surfaces of $H_{a}^{\prime\prime}$ intersecting at a Dirac exceptional line instead of a Dirac EP. The parameters are the same as in Fig. 1.

While both our three-band model $H^{(3)}$ and linearized two-band model $H_{a}^{\prime}$ host a Dirac EP, there is a noticeable difference between them: The small changes in the former, i.e., $H^{(3)} = H_{0} + \Delta H$ where

$$
H _ {0} = \left( \begin{array}{c c c} 1 & 0 & 0 \\ V _ {0} & 0 & 0 \\ 0 & V _ {0} & 1 \end{array} \right), \quad \Delta H = \left( \begin{array}{c c c} - 2 k & - g & 0 \\ g & 0 & - g \\ 0 & g & 2 k \end{array} \right),\tag{22}
$$

and $g \equiv V_{0}\Delta\tau/2$ , are of the same order and linear in terms of k and $\Delta\tau$ ; there are, however, higher-order terms proportional to $\Delta \tau^2$ in the two-band model $H_{a}^{\prime} = H_{0} + \Delta H + \Delta H^{\prime}$ , where

$$
H _ {0} = \left( \begin{array}{c c} 1 & 0 \\ V _ {0} ^ {2} & 1 \end{array} \right), \quad \Delta H = - \frac {V _ {0} ^ {2}}{2} \Delta \tau \mathbf {1} + \left( \begin{array}{c c} - 2 k & 0 \\ V _ {0} ^ {2} \Delta \tau & 2 k \end{array} \right),
$$

and

$$
\Delta H ^ {\prime} = \frac {V _ {0} ^ {2}}{4} \Delta \tau^ {2} \left( \begin{array}{c c} - 1 & 1 \\ 1 & - 1 \end{array} \right).
$$

Without these higher-order terms, especially the upper right element in $\Delta H'$ , $\omega_{0}=1$ is still an EP of the resulting Hamiltonian $H_{a}^{\prime\prime}\equiv H_{0}+\Delta H$ , and the dispersion at this EP is still linear:

$$
\Delta \omega_ {i} = - \frac {V _ {0} ^ {2}}{2} \Delta \tau \pm 2 k.\tag{23}
$$

However, these two energy surfaces intersect at an EP line instead of a Dirac EP (see Fig. 3). One may refer to this line as a Dirac exceptional line (node) following the terminology of a Dirac line node in condensed matter systems [47, 48].

In fact, a two-band model cannot host a Dirac EP in a two-dimensional parameter space, when the perturbation is just first-order. We note that such two-band models are widely used to generate Dirac cones in Hermitian systems $[49]$ , and hence this finding highlights another difference between Hermitian and non-Hermitian systems in terms of their Dirac points. To show this difference, we note that all two-band models with an EP can be put into the Jordan normal form after a similar transformation. Therefore, we can take

$$
H _ {0} = \left( \begin{array}{c c} 0 & 1 \\ 0 & 0 \end{array} \right)\tag{24}
$$

without loss of generality. We then express the perturbation as

$$
\Delta H = \Delta_ {+} \sigma_ {+} + \Delta_ {-} \sigma_ {-} + \Delta_ {3} \sigma_ {3},\tag{25}
$$

where we have neglected perturbations proportional to the identity matrix because they merely cause a shift of the whole spectrum. Here $\sigma_{\pm} = (\sigma_{1} \pm i\sigma_{2})/2$ and $\sigma_{i} (i = 1, 2, 3)$ are the three Pauli matrices. $\Delta_{\pm}, \Delta_{3} \in C$ are three complex perturbation amplitudes of the same order. The two energy eigenvalues are then given by

$$
\omega_ {\pm} = \pm \sqrt {4 \Delta_ {-} (1 + \Delta_ {+}) + \Delta_ {3} ^ {2}}.\tag{26}
$$

Clearly, due to the leading order term $4\Delta_{-}$ in the randicand, the dispersion of this system cannot be conical; only when $\Delta_{-}$ is second order (i.e., $\Delta_{-} = \delta_{-}^{2} \sim \Delta_{3}^{2}$ ) do we recover a conic dispersion to the leading order:

$$
\omega_ {\pm} \approx \pm \sqrt {4 \delta_ {-} ^ {2} + \Delta_ {3} ^ {2}} (\delta_ {-}, \Delta_ {3} \in \mathbb {R}).\tag{27}
$$

In comparison, $H_{0}$ in a two-band Hermitian model with a Dirac point would simply vanish, and we have

$$
\omega_ {\pm} = \pm \sqrt {4 \Delta_ {-} \Delta_ {+} + \Delta_ {3} ^ {2}}\tag{28}
$$

![](images/d5906dc512dcc906912a16c1158d234f228a93ef37f545da66d88f452d122cb3.jpg)

![](images/e9011dd92df825567e38cffcd85b32e07fddac91346f3c39bff9bdb7b280a318.jpg)
FIG. 4. Imaginary Dirac cone with an EP at its center. (a) and (b) show the real and imaginary parts of the three energies of the Hamiltonian given in Eq. (29).

instead. A Dirac cone is then found, e.g., by letting $\Delta_{3} \in R$ together with either $\Delta_{-} = \Delta_{+} \in R$ or $\Delta_{-} = -\Delta_{+} \in iR$ .

## VI. IMAGINARY DIRAC CONE

If we multiply a non-Hermitian Hamiltonian with a Dirac EP by i, it is clear that the Dirac cone now exists in the imaginary part of the energy, which is uniquely non-Hermitian. Moreover, the general analysis in the last section, particular Eq. (27), indicates that it is unnecessary to change the unperturbed Hamiltonian $H_{0}$ to construct an imaginary Dirac cone with an EP at its center; we just need to change $\delta_{-}, \Delta_{3}$ from real to imaginary in this two-band model.

In a three-band model, we find that the following Hamiltonian hosts an imaginary Dirac cone:

$$
H = \left( \begin{array}{c c c} i k & g & 1 \\ g & 1 & g \\ 0 & g & - i k \end{array} \right) \quad (k, g \in \mathbb {R}).\tag{29}
$$

The unperturbed Hamiltonian still has real eigenvalues 0 (EP) and 1. The real and imaginary parts of the three energy eigenvalues are shown in Fig. 4, where the real parts of the two coalesced eigenvalues at the EP stay the same in the two-dimensional parameter space. In other words, these two bands are complex conjugates, with the other band being real. These observations are consistent with the characteristic equation

$$
\omega^ {3} - \omega^ {2} + (k ^ {2} - 2 g ^ {2}) \omega - (k ^ {2} + g ^ {2}) = 0,\tag{30}
$$

which again has real coefficients. When expanding near the EP, we can drop the higher-order cubic term and solve the remaining quadratic equation. The result is

$$
\omega_ {\pm} = \pm i \sqrt {k ^ {2} + g ^ {2}}\tag{31}
$$

to the leading order, showing the explicit conical and linear dispersion.

## VII. CONCLUSION AND DISCUSSIONS

In summary, we studied a novel type of non-Hermitian degeneracies around which the energy spectrum remains real, while they manifest eigenvector coalescence in accordance with the definition of EPs. These Dirac EPs are characterized by a conical dispersion around them, for which the standard perturbative description using the alternating Puiseux series fails. We also identified its imaginary counterpart, where a Dirac cone is formed in the imaginary parts of the energies with an EP at its center.

The Dirac EPs we presented emerge in a non-Hermitian three-band model, which can be reduced to two-band models in several ways. This reduction may result in either Hermitian or non-Hermitian models which are isospectral, and both host degeneracy points. This observation led to a startling discovery: There are Hermitian and non-Hermitian systems that have the same real-valued energy spectrum in their entire two-dimensional parameter space, and their degeneracy points consist of diabolic points in the former, and Dirac EPs in the latter. While isospectral non-Hermitian and Hermitian systems have been reported before (see, for example, Ref. [43]), one with EPs and existing in an entire two-dimensional parameter space has not. As mentioned, we were able to achieve this isospectral property thanks to the elimination of conventional EPs, whose branch cuts would make the non-Hermitian spectrum complex in that region of the two-dimensional parameter space.

Finally, we showed that for two-band models, it is impossible to generate Dirac EPs with linear dispersion along all directions in a two-dimensional parameter space, if only first-order perturbations are introduced to an underlying non-Hermitian Hamiltonian at the EP; second-order terms are necessary to produce Dirac EPs with a conical dispersion. We note that this result holds for higher-dimensional systems as well, as our derivation based on Eq. (25) is independent of the physical dimensions. This property is in stark contrast to Hermitian systems, where the conical dispersion around a Dirac point is produced with first-order perturbations. This observation represents another intriguing difference between Dirac points in Hermitian and non-Hermitian systems.

The results in this work broaden our understanding of non-Hermitian degeneracies. We break the traditional link between eigenvector coalescence and the characteristic integer root dispersion of the eigenvalue spectrum around the exceptional point in a two-dimensional parameter space. While a similar finding was known in the mathematical literature [50], it was based on a single (complex) parameter turning, where the real and imaginary parts of the perturbed Hamiltonian are collinear. When two independent perturbations are allowed instead as we do here (i.g., either using $\Delta \tau$ , $k$ or $\alpha$ , $k$ ), this result was found to break down in general and no traces of a Dirac EP were found [51]: The dispersion near the EP is no longer linear except for one particular direction [52], consistent with previous findings [50]. Furthermore, although the enhanced sensitivity stemming from the nonlinear dispersion of an EP is useful for applications in sensing, it simultaneously imposes a challenge for tuning and observing exceptional points. This fact, combined with the complex character of the non-Hermitian spectrum, often obscure the physics of eigenvector coalescence at the EP. Our work provides a more tolerant platform towards studying eigenstates coalescence in a two-dimensional parameter

[1] K. S. Novoselov, Rev. Mod. Phys. 83, 837 (2011).

[2] A. K. Geim, Rev. Mod. Phys. 83, 851 (2011).

[3] T. O. Wehling, A. M. Black-Schaffer, and A. V. Balatsky, Adv. Phys. 63, 1 (2014).

[4] M. V. Berry, Cze. J. Phys. 54, 1039 (2004).

[5] W. D. Heiss, J. Phys. A: Math. Gen. 37, 2455 (2004).

[6] M.-A. Miri and A. Alù, Science 363, (2019).

[7] Y. D. Chong, L. Ge, and A. D. Stone, Phys. Rev. Lett. 106, 093902 (2011).

[8] L. Ge, Y. D. Chong, and A. D. Stone, Phys. Rev. A 85, 023802 (2012).

[9] L. Ge, K. G. Makris, D. N. Christodoulides, and L. Feng, Phys. Rev. A 92, 062135 (2015).

[10] Z. Lin, H. Ramezani, T. Eichelkraut, T. Kottos, H. Cao, and D. N. Christodoulides, Phys. Rev. Lett. 106, 213901 (2011).

[11] W. R. Sweeney, C. W. Hsu, and A. D. Stone, Phys. Rev. A 102, 063511 (2020).

[12] G. Gamow, Z. Phys. 51, 204 (1928).

[13] L. Feng, R. El-Ganainy, and L. Ge, Nat. Photon. 11, 752–762 (2017).

[14] R. El-Ganainy, K. G. Makris, M. Khajavikhan, Z. H. Musslimani, S. Rotter, and D. N. Christodoulides, Nat. Phys. 14, 11–19 (2018).

[15] V. V. Konotop, J. Yang, and D. A. Zezyulin, Rev. Mod. Phys. 88, 035002 (2016).

[16] C. M. Bender and S. Boettcher, Phys. Rev. Lett. 80, 5243 (1998).

[17] L. Ge, Y. D. Chong, S. Rotter, H. E. Türeci, and A. D. Stone, Phys. Rev. A 84, 023820 (2011).

[18] M. Liertzer, L. Ge, A. Cerjan, A. D. Stone, H. E. Türeci, and S. Rotter, Phys. Rev. Lett. 108, 173901 (2012).

[19] R. El-Ganainy, M. Khajavikhan, and L. Ge, Phys. Rev. A 90, 013802 (2014).

[20] M. Brandstetter, M. Liertzer, C. Deutsch, P. Klang, J. Schöberl, H. E. Türeci, G. Strasser, K. Unterrainer, and S. Rotter, Nat. Commun. 5, 4034 (2014).

[21] B. Peng, Ş. K. Özdemir, S. Rotter, H. Yilmaz, M. Liertzer, F. Monifi, C. M. Bender, F. Nori, and L. Yang, Science 346, 328 (2014).

[22] L. Ge, Phys. Rev. A 92, 052103 (2015).

[23] L. Ge, Photon. Res. 6, A10 (2018).

[24] H. Xue, Q. Wang, B. Zhang, and Y. D. Chong, Phys. Rev. Lett. 124, 236403 (2020).

[25] J. Wiersig, Phys. Rev. Lett. 112, 203901 (2014).

[26] W. Chen, Ş. K. Özdemir, G. Zhao, J. Wiersig, L. Yang, Nature 548, 192 (2017).

[27] Y.-H. Lai, Y.-K. Lu, M.-G. Suh, Z. Yuan, and K. Vahala, Nature 576, 65 (2019).

[28] K. G. Makris, R. El-Ganainy, D. N. Christodoulides, and Z. H. Musslimani, Phys. Rev. Lett. 100, 103904 (2008).

space, which may be generalized to higher dimensions as well.

This project is supported by NSF under Grant No. PHY-1847240 and and ECCS-1846766.

[29] A. Guo, G. J. Salamo, D. Duchesne, R. Morandotti, M. Volatier-Ravat, V. Aimez, G. A. Siviloglou, and D. N. Christodoulides, Phys. Rev. Lett. 103, 093902 (2009).

[30] C. E. Rüter, K. G. Makris, R. El-Ganainy, D. N. Christodoulides, M. Segev, and D. Kip, Nat. Phys. 6, 192 (2010).

[31] L. Ge and H. E. Türeci, Phys. Rev. A 88, 053810 (2013).

[32] F. Zhang, Y. Feng, X. Chen, L. Ge, and W. Wan, Phys. Rev. Lett. 124, 053901 (2020).

[33] P. Peng, W. Cao, C. Shen, W. Qu, J. Wen, L. Jiang, and Y. Xiao, Nat. Phys. 12, 1139 (2016).

[34] Y. Choi, C. Hahn, J. W. Yoon, and S. H. Song, Nat. Commun. 9, 2182 (2018).

[35] X.-L. Zhang, T. Jiang, and C. T. Chan, Light Sci. Appl. 8, 1 (2019).

[36] S. Malzard, C. Poli, and H. Schomerus, Phys. Rev. Lett. 115, 200402 (2015).

[37] L. Ge, Phys. Rev. A 95, 023812 (2017).

[38] B. Qi, L. Zhang and L. Ge, Phy. Rev. Lett. 120, 093901 (2018).

[39] K. Kawabata, K. Shiozaki, M. Ueda, and M. Sato, Phys. Rev. X 9, 041015 (2019).

[40] S. Bittner, B. Dietz, H. L. Harney, M. Miski-Oglu, A. Richter, and F. Sch afer, Phys. Rev. E 89, 032909 (2014).

[41] K. Ding, G. Ma, M. Xiao, Z. Q. Zhang, and C. T. Chan, Phys. Rev. X 6, 021007 (2016).

[42] J. Schindler, A. Li, M. C. Zheng, F. M. Ellis, and T. Kottos, Phys. Rev. A 84, 040101(R) (2011).

[43] J. H. D. Rivero, L. Feng, and L. Ge, Phys. Rev. Lett. 129, 243901 (2022).

[44] A. Pick, B. Zhen, O. D. Miller, C. W. Hsu, F. Hernandez, A. W. Rodriguez, M. Soljačić, and S. G. Johnson, Optics Exp. 25, 12325 (2017).

[45] A. P. Seyranian and A. A. Mailybaev, Multiparameter Stability Theory With Mechanical Applications (World Scientific, Singapore, 2003), Vol. XIII.

[46] A. Friedman and M. Shinbrot, Acta Math. 121, 77-125 (1968).

[47] S. M. Young and C. L. Kane, Phys. Rev. Lett. 115, 126803 (2015).

[48] G.-H. Hong, C.-W. Wang, J. Jiang, C. Chen, S.-T. Cui, H.-F. Yang, A.-J. Liang, S. Liu, Y.-Y. Lv, J. Zhou et al., Chinese Phys. B 27, 017105 (2018).

[49] B. Zhen, C. W. Hsu, Y. Igarashi, L. Lu, I. Kaminer, A. Pick, S.-L. Chua, J. D. Joannopoulos, and M. Soljačić, Nature 525, 354 (2015).

[50] Y. Ma and A. Edelman, Linear Algebra Appl. 273, 45 (1998).

[51] G. Demange and E.-M. Graefe, J. Phys. A: Math. Theor. 45, 025303 (2012).

[52] H. Shen, B. Zhen, and L. Fu, Phys. Rev. Lett. 120, 146402 (2018).
