# Spectral singularities, biorthonormal systems and a two-parameter family of complex point interactions

To cite this article: Ali Mostafazadeh and Hossein Mehri-Dehnavi 2009 J. Phys. A: Math. Theor. 42125303

View the article online for updates and enhancements.

## You may also like

\- Fundamentals of quantum mechanics in Liouville space
Jerryman A Gyamfi

\- Biorthonormal matrix-product-state analysis for the non-Hermitian transfer-matrix renormalization group in the thermodynamic limit
Yu-Kun Huang

\- Hamiltonian formulation of linear non-Hermitian systems
Qi Zhang

# Spectral singularities, biorthonormal systems and a two-parameter family of complex point interactions

Ali Mostafazadeh $^{1}$ and Hossein Mehri-Dehnavi $^{2}$

$^{1}$ Department of Mathematics, Koç University, Rumelifeneri Yolu, 34450 Sariyer, Istanbul, Turkey

$^{2}$ Department of Physics, Institute for Advanced Studies in Basic Sciences, Zanjan 45195-1159, Iran

E-mail: amostafazadeh@ku.edu.tr and mehrideh@iasbs.ac.ir

Received 15 December 2008, in final form 3 February 2009
Published 27 February 2009
Online at stacks.iop.org/JPhysA/42/125303

## Abstract

A curious feature of complex scattering potentials $v(x)$ is the appearance of spectral singularities. We offer a quantitative description of spectral singularities that identifies them with an obstruction to the existence of a complete biorthonormal system consisting of the eigenfunctions of the Hamiltonian operator, i.e., $-\frac{\mathrm{d}^{2}}{\mathrm{d}x^{2}} + v(x)$ , and its adjoint. We establish the equivalence of this description with the mathematicians' definition of spectral singularities for the potential $v(x) = z_{-}\delta(x+a)+z_{+}\delta(x-a)$ , where $z_{\pm}$ and a are respectively complex and real parameters and $\delta(x)$ is the Dirac delta function. We offer a through analysis of the spectral properties of this potential and determine the regions in the space of the coupling constants $z_{\pm}$ where it admits bound states and spectral singularities. In particular, we find an explicit bound on the size of certain regions in which the Hamiltonian is quasi-Hermitian and examine the consequences of imposing PT-symmetry.

PACS number: 03.65.-w

(Some figures in this article are in colour only in the electronic version)

## 1. Introduction

The use of non-Hermitian Hamiltonians in theoretical physics has a long history. It extends from early attempts to construct divergence-free relativistic quantum field theories $[1]$ to more practical and successful applications in nuclear and atomic physics $[2]$ and particularly quantum optics $[3, 4]$ . During the past ten years there has been a renewed interest in the study of a special class of non-Hermitian Hamiltonians that possess a real spectrum. The best-known examples are the PT-symmetric Hamiltonians [5] such as $p^{2} + ix^{3}$ . These belong to the wider class of pseudo-Hermitian Hamiltonians H whose adjoint $H^{\dagger}$ is given by

$$
H ^ {\dagger} = \eta H \eta^ {- 1},\tag{1}
$$

for some Hermitian invertible operator $\eta$ [6, 7]. What makes pseudo-Hermitian Hamiltonians interesting is that they are Hermitian with respect to a possibly indefinite inner product $^{3}$ , namely

$$
\langle \cdot , \cdot \rangle_ {\eta} := \langle \cdot | \eta \cdot \rangle ,\tag{2}
$$

where $\langle\cdot|\cdot\rangle$ is the inner product of the Hilbert space in which all the relevant operators act. Most of the recent work on the subject is concentrated on a particular class of pseudo-Hermitian Hamiltonians, called quasi-Hermitian [8], that satisfies (1) for some positive-definite (metric) operator $\eta$ . In this case, (2) is a positive-definite inner product, and H becomes Hermitian provided that we define the physical Hilbert space of the system using the inner product $\langle\cdot,\cdot\rangle_{\eta}$ , [9, 10]. This allows for formulating the pseudo-Hermitian representation of quantum mechanics in which PT-symmetric as well as non-PT-symmetric quasi-Hermitian Hamiltonians can be employed to describe unitary quantum systems [10]. The techniques developed in this framework have so far found interesting applications in relativistic quantum mechanics [11], quantum cosmology [12], quantum field theory [13], bound-state scattering [14] and classical electrodynamics [15]. But these developments do not undermine the importance of the requirement that the observables of a unitary quantum system must be Hermitian with respect to the inner product of the physical Hilbert space [10].

Among the properties of Hermitian operators that make them indispensable in quantum mechanics is their diagonalizability. For Hermitian and more generally normal operators (those commuting with their adjoint), diagonalizability is equivalent to the existence of an orthonormal basis consisting of the eigenvectors of the operator. This is more commonly referred to as completeness. For a non-normal (and hence non-Hermitian) operator H, diagonalizability of H means the existence of a basis $B^{\dagger}$ consisting of (scattering and bound-state) eigenfunctions of the adjoint operator $H^{\dagger}$ that is biorthonormal to some basis B consisting of the eigenfunctions of H, i.e., B and $B^{\dagger}$ form a biorthonormal system of the Hilbert space [10]. For brevity we shall call such a biorthonormal system a biorthonormal eigensystem for H.

Diagonalizability is a weaker condition than Hermiticity, but diagonalizable operators with a real and discrete spectrum can be related to Hermitian operators via similarity transformations $[7]$ . This in turn implies that they are quasi-Hermitian $[8]$ . The situation is more complicated when the spectrum is continuous. A serious difficulty is the emergence of spectral singularities that conflict with the diagonalizability of the operator in question $[16]$ . The aim of this paper is to elucidate the mechanism by which spectral singularities obstruct the construction of a biorthonormal eigensystem for the operator. We shall achieve this aim by obtaining a quantitative measure of lack of a biorthonormal eigensystem and comparing the latter with the mathematical condition for the presence of spectral singularities that is based on the behavior of the Jost functions. In order to clarify the meaning and consequences of spectral singularities we shall offer a detailed investigation of the spectral properties of the Hamiltonian operators of the form

$$
H = - \frac {\hbar^ {2}}{2 m} \frac {\mathrm{d} ^ {2}}{\mathrm{dx} ^ {2}} + \zeta_ {+} \delta (\mathrm{x} - \alpha) + \zeta_ {-} \delta (\mathrm{x} + \alpha),\tag{3}
$$

where $\zeta_{\pm}$ are complex coupling constants, $\alpha$ is a real parameter and $\delta(x)$ stands for the Dirac delta function.

An alternative mechanism that can make a non-Hermitian operator non-diagonalizable is the emergence of exceptional points. These correspond to degeneracies where both eigenvalues and eigenvectors coalesce $[17]$ . Exceptional points have various physical implications $[3, 4, 18]$ . But they must not be confused with spectral singularities. Unlike exceptional points that can be present for operators with a discrete spectrum (in particular matrix Hamiltonians), spectral singularities are exclusive features of certain operators having a continuous part in their spectrum. As we will see in sections 2 and 3, for an operator having a spectral singularity we can still define two linearly independent (scattering) eigenfunctions for each eigenvalue, nevertheless it is impossible to construct a biorthonormal eigensystem for the operator. To the best of our knowledge, physical meaning of spectral singularities and their possible applications have not been previously studied. This is the subject of $[19]$ where the results of the present paper have been used to develop a physical interpretation for spectral singularities.

Reference [20] uses the mathematical theory of spectral singularities developed in [21, 22] to emphasize their relevance to the recent attempts at using complex scattering potentials to define unitary quantum systems. The results of [20] are, however, confined to potentials defined on the half-line $x \geqslant 0$ , where the Hamiltonian operator acts in the Hilbert space of square-integrable functions $\psi : [0, \infty) \to \mathbb{C}$ satisfying the boundary condition $\psi(0) = 0$ . Furthermore, due to the nature of the concrete potentials studied in [20], it has not been possible to construct bases of the corresponding Hamiltonian and its adjoint and show by explicit calculation how the presence of a spectral singularity obstructs the existence of a biorthonormal eigensystem. This is quite essential, because for the cases that the spectrum is real, the availability of a biorthonormal eigensystem is a necessary condition for the existence of an associated metric operator and the quasi-Hermiticity of the Hamiltonian [10].

The only thoroughly studied example of a complex scattering potential that is defined on the whole real line and can lead to spectral singularities is the single-delta-function potential with a complex coupling [24]. The Hamiltonian operator is given by (3) with $\alpha = \zeta_{-} = 0$ . It develops a spectral singularity if and only if the coupling constant ( $\zeta_{+}$ ) is imaginary. In particular, for the cases that the real part of $\zeta_{+}$ is positive, the bound states are also lacking and the Hamiltonian is quasi-Hermitian. The complex single-delta-function potentials provide a class of manifestly non- $\mathcal{PT}$ -symmetric Hamiltonians with a continuous spectrum that happen to be quasi-Hermitian. An advantage of considering complex double-delta-function potentials is that their space of coupling constants has a subspace, given by $\zeta_{+} = \zeta_{-}^{*}$ , where the Hamiltonian is $\mathcal{PT}$ -invariant. Therefore, these potentials provide an opportunity to investigate the significance of $\mathcal{PT}$ -symmetry [23].

The spectral properties of the PT-symmetric double- and multiple-delta function potentials have been studied in $[25–28]$ . The results are, however, confined to the determination of the (scattering and bound-state) spectrum of these potentials, and no attempt has been made to decide if these potentials lead to spectral singularities.

In this present paper, we will try to obtain a map of the space $C^{2} = R^{4}$ of the coupling constants $\zeta_{\pm}$ that specifies the regions corresponding to the existence of bound states and spectral singularities. We will in particular investigate the intersection of these regions with the two-dimensional PT-symmetric subspace: $\zeta_{-} = \zeta_{+}^{*}$ . The following is an outline of the results we report in this paper. In section 2, we obtain an explicit quantitative measure of the existence of biorthonormal eigensystems and compare the latter with the known condition of the presence of spectral singularities. Here we also provide a useful characterization of spectral singularities and bound states for complex scattering potentials. Section 3 treats the spectral properties of the double-delta function potentials. It consists of four subsections in which we obtain the regions in the space of coupling constants where spectral singularities and bound states exist, find their location in the spectrum of the operator, and determine a lower bound on the size of certain regions in $C^{2}$ where the operator (3) is quasi-Hermitian. Section 4 presents our concluding remarks.

## 2. Spectral singularities

Consider a complex-valued potential $v : R \to C$ depending on a set of complex coupling constants $z_{1}, z_{2}, \ldots, z_{d}$ such that $v^{*}$ is obtained by complex conjugating the coupling constants in the expression for v. Suppose that v decays rapidly $^{4}$ as $|x| \to \infty$ and that the spectrum of the corresponding Hamiltonian operator,

$$
H = - \frac {\mathrm{d} ^ {2}}{\mathrm{d} x ^ {2}} + v (x), \qquad x \in \mathbb {R},\tag{4}
$$

is the set of nonnegative real numbers $^{5}$ . Let $\psi_{ak}^{\vec{z}}(x)$ denote the (generalized) eigenfunctions of H, i.e., linearly independent bounded solutions of

$$
H \psi_ {\mathfrak {a} k} ^ {\vec {z}} (x) = k ^ {2} \psi_ {\mathfrak {a} k} ^ {\vec {z}} (x),\tag{5}
$$

where $k \in R^{+}$ and $a \in \{1,2\}$ are respectively the spectral and degeneracy $^{6}$ labels and $\vec{z} := (z_{1}, z_{2}, \ldots, z_{d})$ .

By definition, H is diagonalizable, if $\psi_{\mathfrak{a}k}^{\vec{z}}(x)$ together with a set of (generalized) eigenfunctions $\phi_{\mathfrak{a}k}^{\vec{z}}(x)$ of $H^{\dagger}$ form a complete biorthonormal system $\left\{\psi_{\mathfrak{a}k}^{\vec{z}}, \phi_{\mathfrak{a}k}^{\vec{z}}\right\}$ , i.e., they satisfy

$$
\big \langle \phi_ {\mathfrak {a} k} ^ {\vec {z}} \big | \psi_ {\mathfrak {b} q} ^ {\vec {z}} \big \rangle = \delta_ {\mathfrak {a b}} \delta (k - q), \qquad \sum_ {\mathfrak {a} = 1} ^ {2} \int_ {0} ^ {\infty} \mathrm{d} k \big | \psi_ {\mathfrak {a} k} ^ {\vec {z}} \big \rangle \big \langle \phi_ {\mathfrak {a} k} ^ {\vec {z}} \big | = 1,\tag{6}
$$

where $\langle\cdot|\cdot\rangle$ is the usual $L^{2}$ -inner product. The biorthonormality relations (6) imply the spectral representation of H,

$$
H = \sum_ {\mathfrak {a} = 1} ^ {2} \int_ {0} ^ {\infty} \mathrm{d} k   k ^ {2} \big | \psi_ {\mathfrak {a} k} ^ {\vec {z}} \big \rangle \big \langle \phi_ {\mathfrak {a} k} ^ {\vec {z}} \big |,\tag{7}
$$

as well as the eigenfunction expansion:

$$
f (x) = \sum_ {\mathfrak {a} = 1} ^ {2} \int_ {0} ^ {\infty} \mathrm{d} k f _ {\mathfrak {a} k} \psi_ {\mathfrak {a} k} ^ {\vec {z}} (x),\tag{8}
$$

where $f:\mathbb{R}\to \mathbb{C}$ is a test function and

$$
f _ {\mathfrak {a} k} := \left\langle \phi_ {\mathfrak {a} k} ^ {\vec {z}} \right| f \rangle .\tag{9}
$$

Because $H^{\dagger} = -\frac{d^{2}}{\mathrm{d}x^{2}} + v(x)^{*}$ , $\psi_{\mathfrak{a}k}^{\vec{z}^*}$ are the eigenfunctions of $H^{\dagger}$ . This in turn means that $\phi_{\mathfrak{a}k}^{\vec{z}^*}$ must be a linear combination of $\psi_{\mathfrak{a}k}^{\vec{z}^*}$ , i.e., there are $J_{\mathfrak{a}\mathfrak{b}} \in \mathbb{C}$ satisfying

$$
\phi_ {\mathfrak {a} k} ^ {\vec {z}} = \sum_ {\mathfrak {b} = 1} ^ {2} J _ {\mathfrak {a b}} \psi_ {\mathfrak {b} k} ^ {\vec {z} ^ {*}}.\tag{10}
$$

In view of (6), there must exist $K_{\mathfrak{ab}} \in \mathbb{C}$ such that

$$
\left\langle \psi_ {\mathfrak {a} k} ^ {\vec {z} ^ {*}} \right| \psi_ {\mathfrak {b} q} ^ {\vec {z}} \rangle = K _ {\mathfrak {a b}} \delta (k - q).\tag{11}
$$

Furthermore, if we respectively denote by $I, J$ and $K$ the $(2 \times 2)$ identity matrix and the matrices with entries $J_{\mathfrak{ab}}$ and $K_{\mathfrak{ab}}$ , we find $J^{*}K = I$ . In particular, $K$ must be an invertible matrix and $J_{\mathfrak{ab}} = K_{\mathfrak{ab}}^{-1*}$ . We can write this relation in the form

$$
J _ {\mathfrak {a b}} = \frac {\tilde {K} _ {\mathfrak {a b}} ^ {*}}{\det (K) ^ {*}},\tag{12}
$$

where $\tilde{K}$ is the transpose of the matrix of cofactors of K. It satisfies

$$
\left( \begin{array}{c c} \big \langle \psi_ {2 k} ^ {\vec {z} ^ {*}} \big | \psi_ {2 q} ^ {\vec {z}} \big \rangle & - \big \langle \psi_ {1 k} ^ {\vec {z} ^ {*}} \big | \psi_ {2 q} ^ {\vec {z}} \big \rangle \\ - \big \langle \psi_ {2 k} ^ {\vec {z} ^ {*}} \big | \psi_ {1 q} ^ {\vec {z}} \big \rangle & \big \langle \psi_ {1 k} ^ {\vec {z} ^ {*}} \big | \psi_ {1 q} ^ {\vec {z}} \big \rangle \end{array} \right) = \tilde {K}   \delta (k - q).
$$

We can use (12) and (10) to express (9) as

$$
f _ {\mathfrak {a} k} = \frac {1}{\det (K)} \sum_ {\mathfrak {b} = 1} ^ {2} \tilde {K} _ {\mathfrak {a b}} \bigl \langle \psi_ {\mathfrak {b} k} ^ {\vec {z} ^ {*}} \big | f \bigr \rangle .\tag{13}
$$

According to this equation if $\det(K)=0$ , the eigenfunction expansion (8) breaks down the eigenfunctions $\psi_{ak}^{\vec{z}}$ do not form a complete set and H is not diagonalizable. We identify this situation with the presence of spectral singularities:

spectral singularities are points $k^2$ of the continuous spectrum of $H$ where $\det(K) = 0$ .

(14)

Because of (complex) analyticity property of the eigenfunctions $\psi_{ak}^{\vec{z}}$ , $\det(K)$ is an analytic function of k. Therefore, the (real) zeros of $\det(K)$ are isolated points forming a countable subset of the real line. Moreover, because v is a bounded function decaying rapidly away from zero, the eigenfunctions tend to plane waves as k becomes large. This shows that $\det(K)$ does not have arbitrarily large zeros (for fixed $\vec{z}$ ). As a result, the zeros of $\det(K)$ are not only isolated but actually finite in number. In other words, depending on the values of the coupling constants $z_{1}, z_{2}, \ldots, z_{d}$ , $\det(K)$ may have no (real) zeros in which case spectral singularities do not arise and H is diagonalizable, or a finite number of (non-vanishing real) zeros $\kappa_{1}, \kappa_{2}, \ldots, \kappa_{\mu}$ in which case $\kappa_{1}^{2}, \kappa_{2}^{2}, \ldots, \kappa_{\mu}^{2}$ are spectral singularities and H is not diagonalizable.

In general the space of coupling constants can be divided into two regions, namely the singular region where H has spectral singularities and the regular region where it is diagonalizable.

In the mathematics literature a spectral singularity is defined as follows:

Definition 1. An element $E_{\star}$ of the (continuous) spectrum of $H$ is called a spectral singularity if the integral kernel of the resolvent operator: $(H - E)^{-1}$ , i.e., the Green's function $\langle x|(H - E)^{-1}|y\rangle$ , is an unbounded function in every small open neighborhood of $E_{\star}$ , but $E_{\star}$ is not an eigenvalue of $H$ with a square-integrable eigenfunction [30] $^{7}$ .

There is a rather general theory of spectral singularities for the differential operators of the form (4) where the spectral singularities are characterized as the real zeros of certain analytic functions [21, 22, 29, 31-33]. For the case that the operator acts in $L^2(\mathbb{R})$ , this is the Wronskian,

$$
W [ \psi_ {k -}, \psi_ {k +} ] := \psi_ {k -} (x) \psi_ {k +} ^ {\prime} (x) - \psi_ {k -} ^ {\prime} (x) \psi_ {k +} (x) = \psi_ {k -} (0) \psi_ {k +} ^ {\prime} (0) - \psi_ {k -} ^ {\prime} (0) \psi_ {k +} (0),\tag{15}
$$

of the Jost solutions $\psi_{k\pm}$ of the eigenvalue equation $H\psi = k^2\psi$ . These are defined in terms of their asymptotic behavior

$$
\psi_ {k -} (x) \to \mathrm{e} ^ {- \mathrm{i} k x} \quad \text { for } \quad x \to - \infty , \qquad \psi_ {k +} (x) \to \mathrm{e} ^ {\mathrm{i} k x} \quad \text { for } \quad x \to \infty .\tag{16}
$$

More specifically, we have [30]

spectral singularities are the real (non-vanishing) zeros of $W[\psi_{k-}, \psi_{k+}]$ .

(17)

This description of spectral singularities seems to differ from the one given in (14). In section 3, we demonstrate the equivalence of the two descriptions for the double-delta function potential by explicit calculations. The following calculation shows how the description (14) relates to definition 1. Using (6), (7), (10) and (12), we have

$$
\begin{array}{l} \langle x | (H - E) ^ {- 1} | y \rangle = \sum_ {\mathfrak {a} = 1} ^ {2} \int_ {0} ^ {\infty} \mathrm{d} k \frac {\psi_ {\mathfrak {a} k} ^ {\vec {z}} (x) \phi_ {\mathfrak {a} k} ^ {\vec {z}} (y) ^ {*}}{k ^ {2} - E} = \sum_ {\mathfrak {a}, \mathfrak {b} = 1} ^ {2} \int_ {0} ^ {\infty} \mathrm{d} k \frac {J _ {\mathfrak {a b}} ^ {*} \psi_ {\mathfrak {a} k} ^ {\vec {z}} (x) \psi_ {\mathfrak {b} k} ^ {\vec {z} *} (y) ^ {*}}{k ^ {2} - E} \\ = \sum_ {\mathfrak {a}, \mathfrak {b} = 1} ^ {2} \int_ {0} ^ {\infty} \mathrm{d} k \frac {\tilde {K} _ {\mathfrak {a b}} \psi_ {\mathfrak {a} k} ^ {\vec {z}} (x) \psi_ {\mathfrak {b} k} ^ {\vec {z} *} (y) ^ {*}}{\det (K) (k ^ {2} - E)}. \end{array}
$$

In the remainder of this section we provide a useful characterization of the spectral singularities and bound states.

Because $|v(x)|$ decays rapidly as $|x| \to \infty$ , solutions of (5) have the asymptotic behavior:

$$
\psi_ {k \mathfrak {a}} ^ {\vec {z}} (x) \to A _ {\pm} \mathrm{e} ^ {\mathrm{i} k x} + B _ {\pm} \mathrm{e} ^ {- \mathrm{i} k x} \qquad \text {for} \quad x \to \pm \infty ,\tag{18}
$$

where $A_{\pm}$ and $B_{\pm}$ are possibly k-dependent complex coefficients. If we denote the coefficients $A_{\pm}$ and $B_{\pm}$ for the Jost solutions $\psi_{k\pm}$ by $A_{\pm}^{\pm}$ and $B_{\pm}^{\pm}$ , we can express (16) as

$$
A _ {+} ^ {+} = B _ {-} ^ {-} = 1, \qquad A _ {-} ^ {-} = B _ {+} ^ {+} = 0.\tag{19}
$$

Next, we let $M = (M_{\mathfrak{ab}})$ be the possibly $k$ -dependent $(2 \times 2)$ transfer matrix [34] satisfying

$$
\binom{A _ {+}}{B _ {+}} = M \binom{A _ {-}}{B _ {-}},\tag{20}
$$

and use this relation and equations (18) and (19) to obtain

$$
A _ {-} ^ {+} = \frac {M _ {2 2}}{\det M}, \quad B _ {-} ^ {+} = - \frac {M _ {2 1}}{\det M}, \quad A _ {+} ^ {-} = M _ {1 2}, \quad B _ {+} ^ {-} = M _ {2 2}.\tag{21}
$$

Inserting these equations in (18), we find

$$
\psi_ {k -} (x) \to M _ {1 2} \mathrm{e} ^ {\mathrm{i} k x} + M _ {2 2} \mathrm{e} ^ {- \mathrm{i} k x} \qquad \text {for} \quad x \to \infty ,\tag{22}
$$

$$
\psi_ {k +} (x) \rightarrow \frac {M _ {2 2} \mathrm{e} ^ {\mathrm{i} k x} - M _ {2 1} \mathrm{e} ^ {- \mathrm{i} k x}}{\det M} \quad \text {   for   } x \rightarrow - \infty .\tag{23}
$$

Because according to Abel's theorem [35], the Wronskian of solutions of (5) is independent of $x$ , we can use the asymptotic formulae for the Jost solutions to compute their Wronskian. We use equations (16), (22) and (23) to perform this calculation first for $x \to \infty$ and then for $x \to -\infty$ . This gives

$$
W [ \psi_ {k -}, \psi_ {k +} ] = 2 \mathrm{i} k M _ {2 2} (k),\tag{24}
$$

$$
W [ \psi_ {k -}, \psi_ {k +} ] = \frac {2 \mathrm{i} k M _ {2 2} (k)}{\det M (k)},\tag{25}
$$

where we have made the $k$ -dependence of $M_{22}$ and $M$ explicit. A direct consequence of (24) and (25) is

$$
\det M (k) = 1.\tag{26}
$$

More importantly, we have the following characterization of spectral singularities that follows from (17) and (24).

spectral singularities are given by $k^2$ where $k$ is a (non-vanishing) real zero of $M_{22}(k)$ .

(27)

Finally, consider the more general case that the Hamiltonian operator (4) has, in addition to a continuous spectrum corresponding to $k \in R^{+}$ , a possibly complex discrete spectrum. The latter corresponds to the square-integrable solutions of (5) that represent bound states. It is not difficult to show that the spectral labels corresponding to these bound states are also zeros of $M_{22}(k)$ , but unlike the zeros associated with the spectral singularities these must have a positive imaginary part. In other words, we have the following characterization of the bound states.

Bound state energies are given by $k^{2}$ where k is a zero of $M_{22}(k)$ with $\operatorname{Im}(k) > 0$ .

(28)

## 3. The double-delta function potential

## 3.1. Eigenfunctions

Consider the time-independent Schrödinger equation

$$
\left[ - \frac {\hbar^ {2}}{2 m} \frac {\mathrm{d} ^ {2}}{\mathrm{dx} ^ {2}} + \zeta_ {+} \delta (\mathrm{x} - \alpha) + \zeta_ {-} \delta (\mathrm{x} + \alpha) \right] \psi = E \psi .\tag{29}
$$

Let $\ell$ be an arbitrary length scale and introduce the dimensionless quantities

$$
z _ {\pm} := \frac {2 m \ell \zeta_ {\pm}}{\hbar^ {2}}, \qquad x := \frac {\mathrm{x}}{\ell}, \qquad a := \frac {\alpha}{\ell}, \qquad E := \frac {2 m \ell^ {2} \mathrm{E}}{\hbar^ {2}}.\tag{30}
$$

Then (29) takes the form

$$
- \psi^ {\prime \prime} + [ z _ {+} \delta (x - a) + z _ {-} \delta (x + a) ] \psi = E \psi .\tag{31}
$$

We can write solutions of (31) as

$$
\psi (x) = \left\{ \begin{array}{l l} \psi^ {-} (x) & \quad \text { for } \quad x <   - a, \\ \psi^ {0} (x) & \quad \text { for } \quad | x | \leqslant a, \\ \psi^ {+} (x) & \quad \text { for } \quad x > a, \end{array} \right.\tag{32}
$$

$$
\psi^ {\nu} (x) = A _ {\nu} \mathrm{e} ^ {\mathrm{i} k x} + B _ {\nu} \mathrm{e} ^ {- \mathrm{i} k x}, \quad \nu \in \{-, 0, + \},\tag{33}
$$

where $k := \sqrt{E}$ and without loss of generality we require that the principal argument of $k$ belongs to $[0, \pi)$ .

To determine the matching conditions at $x = \pm a$ , we demand that $\psi$ be continuous, i.e.,

$$
\psi^ {-} (- a) = \psi^ {0} (- a), \qquad \psi^ {0} (a) = \psi^ {+} (a).\tag{34}
$$

Furthermore, we integrate both sides of (31) over the intervals $[\mp a - \epsilon, \mp a + \epsilon]$ and take the limit $\epsilon \to 0$ in the resulting formulae to find

$$
\psi^ {- \prime} (- a) - \psi^ {0 ^ {\prime}} (- a) + z _ {-} \psi^ {0} (- a) = 0, \qquad \psi^ {0 ^ {\prime}} (a) - \psi^ {+ ^ {\prime}} (a) + z _ {+} \psi^ {0} (a) = 0.\tag{35}
$$

Introducing

$$
w _ {\pm} := \frac {\mathrm{i} z _ {\pm}}{2 k},\tag{36}
$$

and inserting (32) and (33) into (34) and (35) yield the desired matching conditions that we can write in the form

$$
\binom{A _ {-}}{B _ {-}} = \left( \begin{array}{c c} 1 + w _ {-} & w _ {-} \mathrm{e} ^ {2 \mathrm{i} a k} \\ - w _ {-} \mathrm{e} ^ {- 2 \mathrm{i} a k} & 1 - w _ {-} \end{array} \right) \binom{A _ {0}}{B _ {0}}, \qquad \binom{A _ {+}}{B _ {+}} = \left( \begin{array}{c c} 1 - w _ {+} & - w _ {+} \mathrm{e} ^ {- 2 \mathrm{i} a k} \\ w _ {+} \mathrm{e} ^ {2 \mathrm{i} a k} & 1 + w _ {+} \end{array} \right) \binom{A _ {0}}{B _ {0}}.\tag{37}
$$

In light of these relations, the matrix $M$ satisfying (20) reads

$$
M = \left( \begin{array}{c c} 1 - w _ {-} - w _ {+} + (1 - \mathrm{e} ^ {- 4 \mathrm{i} a k}) w _ {-} w _ {+} & 2 \mathrm{i} w _ {-} w _ {+} \sin (2 a k) - w _ {-} \mathrm{e} ^ {2 \mathrm{i} a k} - w _ {+} \mathrm{e} ^ {- 2 \mathrm{i} a k} \\ - 2 \mathrm{i} w _ {-} w _ {+} \sin (2 a k) + w _ {-} \mathrm{e} ^ {- 2 \mathrm{i} a k} + w _ {+} \mathrm{e} ^ {2 \mathrm{i} a k} & 1 + w _ {-} + w _ {+} + (1 - \mathrm{e} ^ {4 \mathrm{i} a k}) w _ {-} w _ {+} \end{array} \right).\tag{38}
$$

It is easy to check that indeed $\det(M)=1$ .

Next, we let $\vec{z}$ stand for $(z_{-}, z_{+})$ and use $\psi_{1k}^{\vec{z}}$ and $\psi_{2k}^{\vec{z}}$ to denote the eigenfunctions obtained by setting $A_0 = (2\pi)^{-1/2}$ , $B_0 = 0$ and $A_0 = 0$ , $B_0 = (2\pi)^{-1/2}$ , respectively. Then

$$
\psi_ {1 k} ^ {\vec {z}} (x) = (2 \pi) ^ {- 1 / 2} \times \left\{ \begin{array}{c c} (1 + w _ {-}) \mathrm{e} ^ {\mathrm{i} k x} - w _ {-} \mathrm{e} ^ {- \mathrm{i} k (x + 2 a)} & \quad \text { for } \quad x <   - a, \\ \mathrm{e} ^ {\mathrm{i} k x} & \quad \text { for } \quad | x | \leqslant a, \\ (1 - w _ {+}) \mathrm{e} ^ {\mathrm{i} k x} + w _ {+} \mathrm{e} ^ {- \mathrm{i} k (x - 2 a)} & \quad \text { for } \quad x > a, \end{array} \right.\tag{39}
$$

$$
\psi_ {2 k} ^ {\vec {z}} (x) = (2 \pi) ^ {- 1 / 2} \times \left\{ \begin{array}{c c} (1 - w _ {-})   \mathrm{e} ^ {- \mathrm{i} k x} + w _ {-} \mathrm{e} ^ {\mathrm{i} k (x + 2 a)} & \quad \text { for } \quad x <   - a, \\ \mathrm{e} ^ {- \mathrm{i} k x} & \quad \text { for } \quad | x | \leqslant a, \\ (1 + w _ {+})   \mathrm{e} ^ {- \mathrm{i} k x} - w _ {+} \mathrm{e} ^ {\mathrm{i} k (x - 2 a)} & \quad \text { for } \quad x > a. \end{array} \right.\tag{40}
$$

We can construct a set of eigenfunctions of $H^{\dagger}$ by taking $z_{\pm}$ to $z_{\pm}^{*}$ or $w_{\pm}$ to $-w_{\pm}^{*}$ in these relations. They are given by

$$
\psi_ {1 k} ^ {\vec {z} ^ {*}} (x) = (2 \pi) ^ {- 1 / 2} \times \left\{ \begin{array}{l l} (1 - w _ {-} ^ {*})   \mathrm{e} ^ {\mathrm{i} k x} + w _ {-} ^ {*} \mathrm{e} ^ {- \mathrm{i} k (x + 2 a)} & \quad \text {for} \quad x <   - a, \\ \mathrm{e} ^ {\mathrm{i} k x} & \quad \text {for} \quad | x | \leqslant a, \\ (1 + w _ {+} ^ {*})   \mathrm{e} ^ {\mathrm{i} k x} - w _ {+} ^ {*} \mathrm{e} ^ {- \mathrm{i} k (x - 2 a)} & \quad \text {for} \quad x > a, \end{array} \right.\tag{41}
$$

$$
\psi_ {2 k} ^ {\vec {z} ^ {*}} (x) = (2 \pi) ^ {- 1 / 2} \times \left\{ \begin{array}{c c} (1 + w _ {-} ^ {*})   \mathrm{e} ^ {- \mathrm{i} k x} - w _ {-} ^ {*} \mathrm{e} ^ {\mathrm{i} k (x + 2 a)} & \quad \text {for} \quad x <   - a, \\ \mathrm{e} ^ {- \mathrm{i} k x} & \quad \text {for} \quad | x | \leqslant a, \\ (1 - w _ {+} ^ {*})   \mathrm{e} ^ {- \mathrm{i} k x} + w _ {+} ^ {*} \mathrm{e} ^ {\mathrm{i} k (x - 2 a)} & \quad \text {for} \quad x > a. \end{array} \right.\tag{42}
$$

## 3.2. Characterization of spectral singularities

In this subsection, we use (14) to determine the spectral singularities of the double-delta-function potential. This requires computes $\langle\psi_{a,k}^{\bar{z}^{*}}|\psi_{b,q}^{\bar{z}}\rangle$ for all $a, b \in \{1, 2\}$ . Using (39) and (40) and the identities

$$
\int_ {\nu} ^ {\infty} \mathrm{e} ^ {\mathrm{i} \mu x} \mathrm{d} x = \pi \delta (\mu) + \frac {\mathrm{i} \mathrm{e} ^ {\mathrm{i} \mu \nu}}{\mu}, \quad \int_ {- \infty} ^ {\nu} \mathrm{e} ^ {\mathrm{i} \mu x} \mathrm{d} x = \pi \delta (\mu) - \frac {\mathrm{i} \mathrm{e} ^ {\mathrm{i} \mu \nu}}{\mu},
$$

we find

$$
\left( \begin{array}{c c} \big \langle \psi_ {1 k} ^ {\vec {z} ^ {*}} \big | \psi_ {1 q} ^ {\vec {z}} \big \rangle & \big \langle \psi_ {1 k} ^ {\vec {z} ^ {*}} \big | \psi_ {2 q} ^ {\vec {z}} \big \rangle \\ \big \langle \psi_ {2 k} ^ {\vec {z} ^ {*}} \big | \psi_ {1 q} ^ {\vec {z}} \big \rangle & \big \langle \psi_ {2 k} ^ {\vec {z} ^ {*}} \big | \psi_ {2 q} ^ {\vec {z}} \big \rangle \end{array} \right) = \delta (k - q) K,\tag{43}
$$

where $K = (K_{ij})$ is a $(2 \times 2)$ matrix with entries

$$
K _ {1 1} = K _ {2 2} = 1 - w _ {-} ^ {2} - w _ {+} ^ {2} = 1 + \frac {z _ {-} ^ {2} + z _ {+} ^ {2}}{4 k ^ {2}},\tag{44}
$$

$$
\begin{array}{r l} & K _ {1 1} = K _ {2 2} - \omega_ {-} - \omega_ {+} + 4 k ^ {2}, \\ & K _ {1 2} = w _ {-} (1 - w _ {-}) \mathrm{e} ^ {2 \mathrm{i} a k} - w _ {+} (1 + w _ {+}) \mathrm{e} ^ {- 2 \mathrm{i} a k} \end{array}
$$

$$
= (4 k ^ {2}) ^ {- 1} [ \mathrm{i} z _ {-} (2 k - \mathrm{i} z _ {-}) \mathrm{e} ^ {2 \mathrm{i} a k} - \mathrm{i} z _ {+} (2 k + \mathrm{i} z _ {+}) \mathrm{e} ^ {- 2 \mathrm{i} a k} ],\tag{45}
$$

$$
\begin{array}{c} K _ {2 1} = - w _ {-} (1 + w _ {-}) \mathrm{e} ^ {- 2 \mathrm{i} a k} + w _ {+} (1 - w _ {+}) \mathrm{e} ^ {2 \mathrm{i} a k} \\ = (4 k ^ {2}) ^ {- 1} [ - \mathrm{i} z _ {-} (2 k + \mathrm{i} z _ {-}) \mathrm{e} ^ {- 2 \mathrm{i} a k} + \mathrm{i} z _ {+} (2 k - \mathrm{i} z _ {+}) \mathrm{e} ^ {2 \mathrm{i} a k} ]. \end{array}\tag{46}
$$

In the $\mathcal{PT}$ -symmetric case, where $z_{+} = z_{-}^{*} =: z \neq 0$ , $K$ is a real matrix, and

$$
K _ {1 1} = K _ {2 2} = 1 + \frac {\mathrm{Re} (z ^ {2})}{2 k ^ {2}},\tag{47}
$$

$$
K _ {1 2} = (2 k ^ {2}) ^ {- 1} \operatorname{Im} [ z (2 k + \mathrm{i} z) \mathrm{e} ^ {- 2 \mathrm{i} a k} ],\tag{48}
$$

$$
K _ {2 1} = (2 k ^ {2}) ^ {- 1} \operatorname{Im} [ z (- 2 k + \mathrm{i} z) \mathrm{e} ^ {2 \mathrm{i} a k} ].\tag{49}
$$

The fact that $K$ is not generally diagonal shows that $\{\psi_{\mathfrak{a}k}^{\vec{z}},\psi_{\mathfrak{b}q}^{\vec{z}^{*}}\}$ is not a biorthonormal system. To construct the basis biorthonormal to $\{\psi_{\mathfrak{a}k}^{\vec{z}}\}$ we transform $\psi_{\mathfrak{a}k}^{\vec{z}^{*}}$ according to

$$
\psi_ {\mathfrak {a} k} ^ {\vec {z} ^ {*}} \rightarrow \phi_ {\mathfrak {a} k} ^ {\vec {z}} := \sum_ {\mathfrak {b} = 1} ^ {2} J _ {\mathfrak {a b}} \psi_ {\mathfrak {b} k} ^ {\vec {z} ^ {*}},
$$

and fix the coefficients $J_{\mathfrak{ab}}$ by demanding that $\left\{\psi_{\mathfrak{a}k}^{\vec{z}}, \phi_{\mathfrak{a}k}^{\vec{z}}\right\}$ be a biorthonormal system. As we explained in section 2, in terms of $K$ this condition takes the form $\delta_{\mathfrak{ab}} = \sum_{c=1}^{2} J_{\mathfrak{ac}}^{*} K_{\mathfrak{cb}}$ . Therefore, a basis biorthonormal to $\left\{\psi_{\mathfrak{a}k}^{\vec{z}}\right\}$ exists provided that the matrix $K$ is invertible, and the matrix $J$ of coefficients $J_{\mathfrak{ab}}$ has the form $J = K^{-1*}$ .

The nonzero real values of k for which K is a singular matrix give the spectral singularities of H. These are the non-vanishing real zeros of $\det(K)$ that we can obtain using (44)–(46):

$$
\det (K) = 1 + \frac {z _ {-} ^ {2} + z _ {+} ^ {2}}{4 k ^ {2}} + \frac {z _ {-} ^ {2} z _ {+} ^ {2}}{8 k ^ {4}} + \frac {z _ {-} z _ {+}}{2 k ^ {2}} \left[ \left(1 - \frac {z _ {-} z _ {+}}{4 k ^ {2}}\right) \cos (4 a k) + \left(\frac {z _ {-} + z _ {+}}{2 k}\right) \sin (4 a k) \right] = 0.\tag{50}
$$

If either $z_{-}=0$ and $z:=z_{+}$ or $z_{+}=0$ and $z:=z_{-}$ , this equation reduces to

$$
1 + \frac {z ^ {2}}{4 k ^ {2}} = 0.
$$

Therefore, for pure imaginary $z$ there is a spectral singularity located at $k = \pm \mathrm{i}z / 2 = |z| / 2$ . This agrees with the results for the single-delta-function potential [24].

For the $\mathcal{PT}$ -symmetric case $(z_{+} = z_{-}^{*}=:z)$ , we have

$$
\det (K) = 1 + \frac {\operatorname{Re} \left(z ^ {2}\right)}{2 k ^ {2}} + \frac {| z | ^ {4}}{8 k ^ {4}} + \frac {| z | ^ {2}}{2 k ^ {2}} \left[ \left(1 - \frac {| z | ^ {2}}{4 k ^ {2}}\right) \cos (4 a k) + \left(\frac {\operatorname{Re} (z)}{k}\right) \sin (4 a k) \right] = 0.\tag{51}
$$

In particular if z is purely imaginary, i.e., $z = i\sigma$ for some $\sigma \in R$ ,

$$
\det (K) = \cos^ {2} (2 a k) + \left(1 - \frac {\sigma^ {2}}{2 k ^ {2}}\right) ^ {2} \sin^ {2} (2 a k).\tag{52}
$$

Therefore, $\det(K) = 0$ iff $\cos(2ak) = 0$ and $k = |\sigma|/\sqrt{2}$ . This implies that there is a spectral singularity for $k = |\sigma|/\sqrt{2} = |z|/\sqrt{2}$ iff $\sigma$ takes one of the following values

$$
\sigma_ {n} := \frac {\pi (2 n + 1)}{2 \sqrt {2} a}, \qquad n \in \mathbb {Z}.\tag{53}
$$

In summary, for the case that $z_{+} = -z_{-} =: z$ is purely imaginary, H has a single spectral singularity given by

$$
E _ {\star} = \frac {\sigma_ {n} ^ {2}}{2} = \left[ \frac {(2 n + 1) \pi}{4 a} \right] ^ {2},\tag{54}
$$

if $z = \mathrm{i}\sigma_{n}$ for some $n \in \mathbb{Z}$ . Otherwise it does not have any spectral singularities.

Next, consider the general $\mathcal{PT}$ -symmetric case where $z_{+} = z_{-}^{*} =: z$ , and $z$ need not be purely imaginary. In this more general case, we rewrite (51) in the form

$$
\det (K) = | f (z, a, k) | ^ {2},\tag{55}
$$

where

$$
f (z, a, k) := \frac {| z | ^ {2} \sin (2 a k)}{2 k ^ {2}} + \mathrm{e} ^ {- 2 \mathrm{i} a k} \left(\frac {\operatorname{Re} (z)}{k} - \mathrm{i}\right).\tag{56}
$$

It is easy to compute

$$
\operatorname{Re} [ f (z, a, k) ] = \left(\frac {| z | ^ {2}}{2 k ^ {2}} - 1\right) \sin (2 a k) + \frac {\operatorname{Re} (z)}{k} \cos (2 a k),\tag{57}
$$

$$
\operatorname{Im} [ f (z, a, k) ] = - \left[ \cos (2 a k) + \frac {\operatorname{Re} (z)}{k} \sin (2 a k) \right].\tag{58}
$$

In view of (55), $\det(K) = 0$ iff $\operatorname{Re}[f(z, a, k)] = \operatorname{Im}[f(z, a, k)] = 0$ . Imposing $\operatorname{Im}[f(z, a, k)] = 0$ , we have

$$
\cos (2 a k) = - \frac {\operatorname{Re} (z)}{k} \sin (2 a k),\tag{59}
$$

which in particular implies $\sin(2ak) \neq 0$ . Moreover, $\cos(2ak) = 0$ iff $\operatorname{Re}(z) = 0$ . In light of $\sin(2ak) \neq 0$ and (59), $\operatorname{Re}[f(z, a, k)] = 0$ gives

$$
- \operatorname{Re} (z) ^ {2} + \operatorname{Im} (z) ^ {2} = 2 k ^ {2}.\tag{60}
$$

This implies that if $|\operatorname{Re}(z)| \geqslant |\operatorname{Im}(z)|$ , there is no spectral singularity.

Next, we solve (60) for $k$ to obtain

$$
k = \sqrt {\frac {- \mathrm{Re} (z) ^ {2} + \mathrm{Im} (z) ^ {2}}{2}},\tag{61}
$$

and express (59) as

$$
\operatorname{Re} (z) = - k \cot (2 a k).\tag{62}
$$

Inserting (61) into (62) yields a necessary and sufficient condition for the existence of a spectral singularity, namely

$$
2 \operatorname{Re} (z) \tan \left(a \sqrt {2 [ - \operatorname{Re} (z) ^ {2} + \operatorname{Im} (z) ^ {2} ]}\right) + \sqrt {2 [ - \operatorname{Re} (z) ^ {2} + \operatorname{Im} (z) ^ {2} ]} = 0.\tag{63}
$$

Introducing the variables

$$
r := 2 a \operatorname{Re} (z), \qquad s := 2 a \operatorname{Im} (z), \qquad t := a \sqrt {2 [ - \operatorname{Re} (z) ^ {2} + \operatorname{Im} (z) ^ {2} ]},\tag{64}
$$

we can express (63) in the form

$$
r = - t \cot t,\tag{65}
$$

and establish

$$
s = \pm t \sqrt {1 + \csc^ {2} t}.\tag{66}
$$

10

![](images/fad44e4658dce82f9c70d270fb5bd3fcf2b0bf0c24e4e47e0d0c8d20298a763e.jpg)
Figure 1. Curves in the r-s plane giving the location of the spectral singularities for the general PT-symmetric double-delta function potential. The dashed lines are the asymptotes $s = \pm r$ . The intersection of the curves with the s-axis corresponds to the spectral singularities given by equation (54).

Figure 1 shows a plot of the parametric curve defined by (65) and (66). It consists of an infinite set of disjoint open curves with asymptotes $s = \pm r$ in the $r - s$ plane. The points on these curves correspond to the values of the coupling constant $z$ for which a spectral singularity appears.

Next, consider the general not necessarily $\mathcal{PT}$ -symmetric case. Generalizing our treatment of the $\mathcal{PT}$ -symmetric case, we use (50) to factorize $\det(K)$ as

$$
\det (K) = f _ {-} \left(z _ {-}, z _ {+}, a, k\right) f _ {+} \left(z _ {-}, z _ {+}, a, k\right),\tag{67}
$$

where

$$
f _ {\pm} (z _ {-}, z _ {+}, a, k) := \frac {u}{2 k ^ {2}} \sin (2 a k) + \mathrm{e} ^ {\pm 2 \mathrm{i} a k} \left(\frac {v}{k} \pm \mathrm{i}\right), \quad u := z _ {-} z _ {+}, \quad v := \frac {z _ {-} + z _ {+}}{2}.\tag{68}
$$

Therefore, $\det(K)=0$ if and only if at least one of the $f_{\pm}(z_{-},z_{+},a,k)$ vanishes.

Let us abbreviate $f_{-}(z_{-}, z_{+}, a, k)$ as $f(k)$ , i.e., set

$$
f (k) := \frac {u}{2 k ^ {2}} \sin (2 a k) + \mathrm{e} ^ {- 2 \mathrm{i} a k} \left(\frac {v}{k} - \mathrm{i}\right).\tag{69}
$$

Then it is easy to see that $f_{+}(z_{-}, z_{+}, a, k) = -f(-k)$ . Therefore, the positive zeros of $f_{+}(z_{-}, z_{+}, a, k)$ are identical with the absolute value of the negative zeros of $f_{-}(z_{-}, z_{+}, a, k)$ . In other words, the spectral singularities are given by positive and negative real zeros of $f(k)$ . Another interesting property of $f(k)$ is that it satisfies

$$
f (k) = - \mathrm{ie} ^ {- 2 \mathrm{i} a k} [ 1 + w _ {-} + w _ {+} + w _ {-} w _ {+} (1 - \mathrm{e} ^ {4 a \mathrm{i} k}) ] = - \mathrm{ie} ^ {- 2 \mathrm{i} a k} M _ {2 2} (k),\tag{70}
$$

where $M_{22}(k)$ is the entry of the matrix M of (38) with the row and column labels 2. According to (70) the spectral singularities are the non-vanishing real zeros of $M_{22}(k)$ . This establishes the equivalence of (14) and (17) for the double-delta function potentials.

In order to characterize the real zeros of $f(k)$ , we set the real and imaginary parts of the right-hand side of (69) equal to zero. This gives

$$
\left(- 1 + \frac {\operatorname{Re} (u)}{2 k ^ {2}} + \frac {\operatorname{Im} (v)}{k}\right) \sin (2 a k) + \left(\frac {\operatorname{Re} (v)}{k}\right) \cos (2 a k) = 0,\tag{71}
$$

$$
\left(\frac {\operatorname{Im} (u)}{2 k ^ {2}} - \frac {\operatorname{Re} (v)}{k}\right) \sin (2 a k) + \left(\frac {\operatorname{Im} (v)}{k} - 1\right) \cos (2 a k) = 0.\tag{72}
$$

Because $\sin(2ak)$ and $\cos(2ak)$ cannot vanish simultaneously, these equations hold provided that the matrix of coefficients of $\sin(2ak)$ and $\cos(2ak)$ is singular. Equating the determinant of this matrix to zero and simplifying the resulting equation, we find

$$
g (k) := k ^ {3} - 2 \operatorname{Im} (v) k ^ {2} + \left(- \frac {\operatorname{Re} (u)}{2} + | v | ^ {2}\right) k + \frac {1}{2} [ \operatorname{Re} (u) \operatorname{Im} (v) - \operatorname{Re} (v) \operatorname{Im} (u) ] = 0.\tag{73}
$$

Because $g$ is a real cubic polynomial, it always has at least one real root $\kappa$ . If $\kappa \neq 0$ , $E_{\star} = \kappa^{2}$ is a spectral singularity. Expressing $\kappa$ as a function of $u$ and $v$ and inserting it into say (72) we find a sufficient condition on the coupling constants $z_{\pm}$ for the existence of a spectral singularity. Repeating this for all the roots of $g$ (for the cases that (73) has other nonzero real solutions) we obtain a complete characterization of the spectral singularities. They lie on a three-dimensional surface $S$ embedded in the four-dimensional space $(\mathbb{C}^2)$ of the coupling constants $(z_{-}, z_{+})$ . Figure 1 is a graphical demonstration of the intersection of $S$ with the plane $z_{+} = z_{-}^{*}$ that represents the $PT$ -symmetric region of $\mathbb{C}^2$ . In the following we examine some non- $PT$ -symmetric regions of $\mathbb{C}^2$ and their intersection with $S$ .

(1) Consider the plane $\Pi_1$ in $\mathbb{C}^2$ defined by $z_{+} = -z_{-}^{*}=: z$ where $u = -|z|^2$ and $v = \mathrm{i}\mathrm{Im}(z)$ . Then equations (71) and (72) take the form

$$
\left[ 1 + \left(\frac {\operatorname{Re} (z)}{k}\right) ^ {2} + \left(\frac {\operatorname{Im} (z)}{k} - 1\right) ^ {2} \right] \sin (2 a k) = 0 = \left(\frac {\operatorname{Im} (z)}{k} - 1\right) \cos (2 a k).
$$

These are satisfied if and only if $\sin(2ak)=0$ and $k=|\operatorname{Im}(z)|$ . Therefore, we have a spectral singularity located at $E_{\star}=\kappa^{2}=\operatorname{Im}(z)^{2}$ , if and only if

$$
\operatorname{Im} (z) = n \pi , \quad \text {   for   some   } \quad n \in \mathbb {Z} - \{0 \}.\tag{74}
$$

This shows that $\Pi_1$ intersects $S$ along equidistant lines parallel to the $\operatorname{Re}(z)$ -axis in $\Pi_1$ . (2) Consider the case that both $z_+$ and $z_-$ are purely imaginary. This also defines a plane in $\mathbb{C}^2$ that we denote by $\Pi_2$ . In this case, we can express $z_{\pm}$ as

$$
z _ {\pm} =: \frac {\mathrm{i} y _ {\pm}}{a},
$$

where $y_{\pm}$ are nonzero real numbers. In terms of $y_{\pm}$ , equations (71) and (72) take the form

$$
\left(\frac {y _ {+} + y _ {-}}{2 a k} - 1\right) \cos (2 a k) = 0 = \left(\frac {y _ {+} + y _ {-}}{2 a k} - \frac {y _ {+} y _ {-}}{2 a ^ {2} k ^ {2}} - 1\right) \sin (2 a k).
$$

There are two ways to satisfy these equations. Either

$$
\frac {y _ {+} + y _ {-}}{2 a k} - 1 = \sin (2 a k) = 0,\tag{75}
$$

or

$$
2 a ^ {2} k ^ {2} - a k \left(y _ {+} + y _ {-}\right) + y _ {+} y _ {-} = \cos (2 a k) = 0.\tag{76}
$$

We consider these two cases separately.

![](images/a767561381a5c111bbb7c3b628ae2e54facaaf745ff4bed650d55e08cef73391.jpg)
Figure 2. Curves in the $y_{+}-y_{-}$ plane ( $\Pi_{2}$ ) along which one has a spectral singularity for purely imaginary couplings. There are spectral singularities along the $y_{-}$ - and $y_{+}$ -axes. The dashed line ( $y_{+} = -y_{-}$ ) represents the PT-symmetric double-delta-function potential with purely imaginary couplings. The intersection of these lines with the full curves corresponds to the spectral singularities given by equation (54).

If (75) holds, $E_{\star} = \kappa^{2}$ with $\kappa := (y_{+} + y_{-})/2a$ is a spectral singularity provided that

$$
y _ {+} + y _ {-} = \frac {n \pi}{2}, \qquad n \in \mathbb {Z} - \{0 \}.\tag{77}
$$

This defines a set of equidistance parallel lines in $\Pi_2$ along which we have spectral singularities.

If (76) holds, $k = \kappa_{n}$ where $\kappa_{n} := (2n + 1)\pi/(4a)$ for all $n \in Z$ , and

$$
y _ {+} = a \kappa_ {n} \left(\frac {y _ {-} - 2 a \kappa_ {n}}{y _ {-} - a \kappa_ {n}}\right) = \frac {(2 n + 1) \pi}{2} \left[ \frac {2 y _ {-} - (2 n + 1) \pi}{4 y _ {-} - (2 n + 1) \pi} \right].\tag{78}
$$

This equation gives the location of another set of spectral singularities, namely $E_{\star} = \kappa_{n}^{2}$ , in the plane $\Pi_{2}$ .

Figure 2 shows the curves in $\Pi_2$ along which a spectral singularity arises, i.e., $\Pi_2 \cap S$ .

(3) Consider the plane $\Pi_3$ in $\mathbb{C}^2$ corresponding to $z_{+} = -z_{-} =: z$ . Then $v = 0$ and $u = -z^2$ . In particular, $\operatorname{Im}(u) = -2\operatorname{Re}(z)\operatorname{Im}(z)$ . We can confine our attention to the subcase: $\operatorname{Im}(u) \neq 0$ , because for $\operatorname{Im}(u) = 0$ either $\operatorname{Re}(z) = 0$ , in which case $z_{\pm}$ are purely imaginary and the results of the case 2 apply, or $\operatorname{Im}(z) = 0$ , in which case the potential is real and there are no spectral singularities.

In view of $v = 0$ and (73), $g(k) = k(k^2 - \operatorname{Re}(u) / 2)$ . Therefore, $k$ does not have a real zero and there is no spectral singularities, if $\operatorname{Re}(u) \leqslant 0$ . For $\operatorname{Re}(u) > 0$ , there is a spectral singularity at $E_{\star} = \kappa_{\pm}^{2} = \mathrm{Re}(u) / 2$ , where

![](images/6d1180fa1b4433692135b66a80ee24afb3862c73b3094aada0e2cd911933ef22.jpg)
Figure 3. Curves in the r-s plane along which spectral singularities occur for the coupling constants with opposite sign. The origin $s = r = 0$ does not actually lie on these curves. The intersection of the curves with the s-axis corresponds to the spectral singularities given by equation (54). The dashed lines are the lines $s = \pm r$ .

$$
\kappa_ {\pm} := \pm \sqrt {\frac {\operatorname{Re} (u)}{2}}.\tag{79}
$$

Inserting this equation into (72) gives

$$
\operatorname{Im} (u) = \pm \operatorname{Re} (u) \cot (a \sqrt {2 \operatorname{Re} (u)}).\tag{80}
$$

Introducing the parameter $t := a\sqrt{2\operatorname{Re}(u)}$ , we can use (80) to obtain the following parametric equations for the $r := a\operatorname{Re}(z)$ and $s := a\operatorname{Im}(z)$ values that correspond to the spectral singularities:

$$
| r (t) | = \frac {t}{2} \sqrt {| \csc t | - 1}, \quad | s (t) | = \frac {t \cos t}{\sqrt {| \sin t | - \sin^ {2} t}}.\tag{81}
$$

Figure 3 shows the graph of the parametric curves defined by (81). They form the intersection of the plane $\Pi_3$ with the singular region $S$ of $\mathbb{C}^2$ .

(4) Consider the case that $z_{\pm} = (1 + \mathrm{i}s_{\pm}) / a$ with $s_{\pm} \in \mathbb{R}$ arbitrary. This corresponds to another plane in $\mathbb{C}^2$ that we denote by $\Pi_4$ . Introducing

$$
s := \frac {s _ {-} + s _ {+}}{2}, \qquad t := \frac {1 + s _ {-} s _ {+}}{2},\tag{82}
$$

we have

$$
v = \frac {1 + \mathrm{i} s}{a}, \qquad u = \frac {1 - t + \mathrm{i} s}{a ^ {2}}.\tag{83}
$$

Inserting these into (73) yields

$$
a ^ {3} g (k) = (a k - s) \left(a ^ {2} k ^ {2} - a s k + t\right) = 0.\tag{84}
$$

Therefore, we need to consider the following two possibilities.

(a) $k = s / a = (s_{-} + s_{+}) / (2a)$ . In this case (72) is satisfied automatically while (71) yields

$$
t = s \cot (2 s) + 1.\tag{85}
$$

We can use (82) and (85) to express $s_{\pm}$ in terms of s. This gives

$$
s _ {-} = s \mp \sqrt {s ^ {2} + 1 - 2 (s \cot (2 s) + 1)},\tag{86}
$$

$$
s _ {+} = 2 s - s _ {-} = s \pm \sqrt {s ^ {2} + 1 - 2 (s \cot (2 s) + 1)}.\tag{87}
$$

(b) $k \neq s / a$ . Then according to (84),

$$
t = a s k - a ^ {2} k ^ {2}.\tag{88}
$$

Furthermore, both (71) and (72) become

$$
\tan (2 a k) + a k = 0.\tag{89}
$$

This equation has a countably infinite set of real solutions $\kappa_{n}$ that can be easily obtained numerically. Substituting $\kappa_{n}$ for $k$ into (88) and using (82), we find

$$
s _ {-} = s \mp \sqrt {s ^ {2} + 1 - 2 \left(a s \kappa_ {n} - a ^ {2} \kappa_ {n} ^ {2}\right)},\tag{90}
$$

$$
s _ {+} = 2 s - s _ {-} = s \pm \sqrt {s ^ {2} + 1 - 2 \left(a s \kappa_ {n} - a ^ {2} \kappa_ {n} ^ {2}\right)}.\tag{91}
$$

Figure 4 shows the parametric plot of the curves in the $s_{-} - s_{+}$ plane corresponding to the spectral singularities for both cases 4(a) and 4(b) with $a = 1$ . As seen from this figure, there are no spectral singularity in the unit-disc defined by $s_{-}^{2} + s_{+}^{2} \leqslant 1$ .

## 3.3. Location of the spectral singularities and the bound states

As we noted in section 2, the spectral singularities are given by the real zeros of $M_{22}(k)$ while the bound states correspond to the zeros of $M_{22}(k)$ with a positive imaginary part. For the double-delta function potential, we can write $M_{22}(k) = 0$ in the following more compact form:

$$
(\mathfrak {K} - \mathfrak {z} _ {-}) (\mathfrak {K} - \mathfrak {z} _ {+}) = \mathfrak {z} _ {-} \mathfrak {z} _ {+} e ^ {2 \mathfrak {K}},\tag{92}
$$

where we have used (38) and introduced

$$
\mathfrak {z} _ {\pm} := a z _ {\pm} = \frac {2 m \alpha \zeta_ {\pm}}{\hbar^ {2}}, \qquad \mathfrak {K} := 2 \mathrm{i} a k.
$$

In particular, the spectral singularities are given by

$$
E _ {\star} := - \frac {\mathcal {K} ^ {2}}{4 a ^ {2}},\tag{93}
$$

where $\mathfrak{K}$ is a nonzero solution of (92) lying on the imaginary axis in the complex $\mathfrak{K}$ -plane, i.e.,

$$
\mathcal {K} \in \ell := \{w \in \mathbb {C} | \operatorname{Re} (w) = 0 \neq w \},
$$

whereas the bound state ‘energies’ are given by (93) for solutions $\mathcal{K}$ of (92) lying to the left of this axis, i.e.,

$$
\mathcal {K} \in \Pi_ {-} := \{w \in \mathbb {C} | \operatorname{Re} (w) <   0 \}.
$$

![](images/193e9abc76f18238dec445fc9b81a7893be921d6945dfe5ed9d363a09f20f9bb.jpg)
Figure 4. Curves in the $s_{-} - s_{+}$ plane along which the spectral singularities occur for the coupling constants of the form $z_{\pm} = 1 + \mathrm{i}s_{\pm}$ . The solid (red) and dashed (blue) curves correspond to the spectral singularities with $k = (s_{+} + s_{-}) / 2$ (case 4(a)) and $k \neq (s_{+} + s_{-}) / 2$ (case 4(b)), respectively. Also shown (in green) is the unit-disc: $s_{-}^{2} + s_{+}^{2} \leqslant 1$ , where there are no spectral singularities.

For both spectral singularities and bound states, we have $\mathrm{Re}(\mathfrak{K}) \leqslant 0$ which implies $|\mathrm{e}^{2\mathfrak{K}}| \leqslant 1$ . Taking the modulus of both sides of (92), we find

$$
\left| \mathfrak {K} - \mathfrak {z} _ {-} \right| \left| \mathfrak {K} - \mathfrak {z} _ {+} \right| \leqslant \left| \mathfrak {z} _ {+} \right| \left| \mathfrak {z} _ {-} \right|.\tag{94}
$$

This is violated for any $\mathfrak{K}$ fulfilling

$$
\left| \mathfrak {K} - \mathfrak {z} _ {-} \right| > \left| \mathfrak {z} _ {-} \right| \quad \text { and } \quad \left| \mathfrak {K} - \mathfrak {z} _ {+} \right| > \left| \mathfrak {z} _ {+} \right|.\tag{95}
$$

Therefore, the solutions of (92) with $\operatorname{Re}(\mathfrak{K}) \leqslant 0$ must belong to the union of the discs

$$
D _ {\pm} := \{\mathfrak {K} \in \mathbb {C} | | \mathfrak {K} - \mathfrak {z} _ {\pm} | \leqslant | \mathfrak {z} _ {\pm} | \}.
$$

This provides an upper bound on the size of the region in the complex $\mathfrak{K}$ -plane where bound-state energies and spectral singularities are located, namely

$$
\mathcal {R} _ {\vec {\mathfrak {z}}} := (\Pi_ {-} \cup \ell) \cap (D _ {+} \cup D _ {-}).
$$

Here we have used the index $\vec{\mathfrak{z}} := (\mathfrak{z}_{-}, \mathfrak{z}_{+})$ to emphasize the $\mathfrak{z}_{\pm}$ -dependence of $\mathcal{R}_{\vec{\mathfrak{z}}}$ . Figure 5 illustrates the discs $D_{\pm}$ and the region $\mathcal{R}_{\vec{\mathfrak{z}}}$ for a generic choice of $\mathfrak{z}_{\pm}$ and also for the case that $\mathfrak{z}_{\pm}$ are real and positive. It is easy to see that in the latter case $\mathcal{R}_{\vec{\mathfrak{z}}}$ is empty and there are no spectral singularities or bound states.

Let $D_{\sigma}$ and $\mathfrak{D}_{\sigma}$ be the disc and half-disc defined by

$$
D _ {\sigma} := \{\mathfrak {K} \in \mathbb {C} | | \mathfrak {K} | \leqslant \sigma \}, \quad \mathfrak {D} _ {\sigma} := \{\mathfrak {K} \in \mathbb {C} | | \mathfrak {K} | \leqslant \sigma , \operatorname{Re} (\mathfrak {K}) \leqslant 0. \},\tag{96}
$$

where $\sigma$ is the largest of $2|\mathfrak{z}_{\pm}|$ , i.e.,

$$
\sigma := 2 \max (| \mathfrak {z} _ {-} |, | \mathfrak {z} _ {+} |).\tag{97}
$$

![](images/81d9f3c1c2942e0d1503bf76151286724185d49e6dfe0b8dd003af1db51051b1.jpg)
Figure 5. (a) Discs $D_{\pm}$ and $D_{\sigma}$ for generic values of $\mathfrak{z}_{\pm}$ . The gray area with the origin excluded is the region $\mathcal{R}_{\mathfrak{z}}^{-}$ where the bound states and spectral singularities are located (if any). (b) $D_{\pm}$ for $\mathfrak{z}_{\pm} \in \mathbb{R}^{+}$ . In this case $\mathcal{R}_{\mathfrak{z}}^{-}$ is empty.

Then, $\mathfrak{K} \in \mathfrak{D}_{\sigma}$ is a weaker necessary condition for the existence of bound states and spectral singularities. This is simply because $D_{\pm} \subseteq D_{\sigma}$ . See figure 5(a).

Because the spectral singularities and bound states are given by the zeros of

$$
F _ {\vec {z}} (\mathfrak {K}) := (\mathfrak {K} - \mathfrak {z} _ {-}) (\mathfrak {K} - \mathfrak {z} _ {+}) - \mathfrak {z} _ {-} \mathfrak {z} _ {+} e ^ {2 \mathfrak {K}},\tag{98}
$$

which is an entire (everywhere complex-analytic) function, and these zeros are contained in $D_{\sigma}$ which is a compact subset of the complex K-plane, we can determine the location of the spectral singularities and bound states using the following well-known result of complex analysis.

Theorem 1. Let C be a counterclockwise oriented contour bounding a compact and simply-connected region R in complex plane and $h : C \to C$ be a function that is analytic on an open subset containing R. Then h has a finite number of zeros in R. Moreover, if none of these zeros lie on C, the contour integral

$$
n _ {C} := \frac {1}{2 \pi \mathrm{i}} \oint_ {C} \frac {h ^ {\prime} (w)}{h (w)} \mathrm{d} w\tag{99}
$$

gives the sum of orders of zeros of h contained in R. In particular, if all of these zeros are simple (of order 1), $n_{C}$ gives their number [36, section 10].

A proper use of this theorem requires a careful analysis of the order of zeros of $F_{\vec{3}}$ . It is not difficult to show that the zeros of $F_{\vec{3}}$ can at most be of order 3. Moreover, K is a third order zero of $F_{\vec{3}}$ if and only if K = 0 and

$$
\mathfrak {z} _ {-} = \frac {- 1 \pm \mathrm{i}}{2}, \quad \mathfrak {z} _ {+} = \frac {1}{2 \mathfrak {z} _ {-}} = \frac {- 1 \mp \mathrm{i}}{2}.\tag{100}
$$

This does not correspond to a spectral singularity or a bound state. $F_{\tilde{3}}$ has a second-order zero $\mathfrak{K}_2$ if and only if

![](images/3b59366a06d08e583c5adb924eeb4faec750d06077297f1eee058bd2f35969a3.jpg)
Figure 6. (a) $c_{\pm}(\rho_{\pm})$ are rectangular contours of width $\epsilon \ll 1$ and height $\rho_{\pm} \leqslant \sigma$ ; (b) $C(\rho, \theta)$ is the boundary of the region lying between circular arcs of side length $\epsilon \ll 1$ and $\rho \leqslant \sigma$ and opening angle $\theta \in [\epsilon, \pi - \epsilon]$ .

$$
2 _ {\mathfrak {z} - \mathfrak {z} _ {+}} \mathrm{e} ^ {1 + \mathfrak {z} _ {-} + \mathfrak {z} _ {+} \pm \sqrt {1 + (\mathfrak {z} _ {-} - \mathfrak {z} _ {+}) ^ {2}}} = 1 \pm \sqrt {1 + (\mathfrak {z} _ {-} - \mathfrak {z} _ {+}) ^ {2}},\tag{101}
$$

$$
\mathcal {K} _ {2} = \frac {1}{2} \big [ 1 + \mathfrak {z} _ {-} + \mathfrak {z} _ {+} \pm \sqrt {1 + (\mathfrak {z} _ {-} - \mathfrak {z} _ {+}) ^ {2}} \big ].\tag{102}
$$

Requiring that $\mathrm{Re}(\mathfrak{K}_2) \leqslant 0$ , we can use (101) to show that $|1 \pm \sqrt{1 + (\mathfrak{z}_- - \mathfrak{z}_+)^2}| \leqslant 2|\mathfrak{z}_-\mathfrak{z}_+|$ . Therefore, there is no spectral singularity or bound state associated with a second-order zero of $F_{\vec{3}}$ , if for both choices of the sign,

$$
\left| 1 \pm \sqrt {1 + (\mathfrak {z} _ {-} - \mathfrak {z} _ {+}) ^ {2}} \right| > 2 | \mathfrak {z} _ {-} \mathfrak {z} _ {+} |.\tag{103}
$$

This inequality in turn implies the following sufficient condition for the lack of spectral singularities and bound states associated with a second-order zero of $F_{\tilde{3}}$ .

$$
\left| \mathfrak {z} _ {-} \mathfrak {z} _ {+} \right| \left(\left| \mathfrak {z} _ {-} \mathfrak {z} _ {+} \right| - 1\right) <   \frac {\left| \mathfrak {z} _ {-} - \mathfrak {z} _ {+} \right| ^ {2}}{4}.\tag{104}
$$

In particular, such bound states or spectral singularities are forbidden if $|z_{-}z_{+}| \leqslant 1$ .

Next, we return to the idea of using theorem 1 for locating the spectral singularities and bound states. For this purpose we can use the contours $C(\rho, \theta)$ and $c_{\pm}(\rho_{\pm})$ depicted in figure 6 to compute

$$
n _ {\pm} (\rho) := \frac {1}{2 \pi \mathrm{i}} \oint_ {c _ {\pm} (\rho)} \frac {F _ {\vec {3}} ^ {\prime} (\mathfrak {K})}{F _ {\vec {3}} (\mathfrak {K})} \mathrm{d} \mathfrak {K},\tag{105}
$$

$$
n (\rho_ {-}, \rho_ {+}) := n _ {-} (\rho_ {-}) + n _ {+} (\rho_ {+}),\tag{106}
$$

$$
N (\rho , \theta) := \frac {1}{2 \pi \mathrm{i}} \oint_ {C (\rho , \theta)} \frac {F _ {\vec {\mathfrak {z}}} ^ {\prime} (\mathfrak {K})}{F _ {\vec {\mathfrak {z}}} (\mathfrak {K})} \mathrm{d} \mathfrak {K},\tag{107}
$$

where $\rho, \rho_{\pm} \in [\epsilon, \sigma], \epsilon \ll 1$ , and $\theta \in [\epsilon, \pi - \epsilon]$ . In the generic case where $F_{\vec{3}}$ has no second-order zeros, $n_{\pm}(\rho)$ and $N(\rho, \theta)$ give the number of zeros of $F_{\vec{3}}$ enclosed by $c_{\pm}(\rho)$ and $C(\rho, \theta)$ , respectively. Therefore, plotting $n(\rho_{-}, \rho_{+})$ and $N(\rho, \theta)$ as functions of $\rho_{\pm}$ and $(\rho, \theta)$ , we can locate all the spectral singularities and bound states of the double-delta function potential for given coupling constants $z_{\pm}$ . In particular, for $\epsilon \to 0$ , $n_{\mathrm{tot}} := n(\sigma, \sigma)$ and $N_{\mathrm{tot}} := N(\sigma, \pi - \epsilon)$ respectively give the total number of spectral singularities and bound states, except for the cases that for some imaginary K both K and $-K$ are zeros of $F_{\vec{3}}$ . In the latter case, K and $-K$ give rise to the same spectral singularity, and one must account for the corresponding double counting in $n_{tot}$ .

![](images/310bf85c5e89581d6ab13f5c11e247d991539643df012c7a83020fdb65ab9b88.jpg)

![](images/fcf103414fa3ac067c641ddc176d53f3ec0811cc96f50214c0e9be3121f11c43.jpg)
Figure 7. Density plots of $N(\rho, \theta)$ for the $\mathcal{PT}$ -symmetric case $\mathfrak{z}_{\pm} = -8 \pm 3\mathrm{i}$ (on the left) and the non- $\mathcal{PT}$ -symmetric case $\mathfrak{z}_{-} = -8 + 3\mathrm{i}, \mathfrak{z}_{+} = -4 - 2\mathrm{i}$ (on the right) in the complex $\mathfrak{K}$ -plane. $\mathfrak{K}_{r}$ and $\mathfrak{K}_{i}$ mark the real and imaginary axes. As the color changes from the lightest to the darkest $N(\rho, \theta)$ takes values 0,1 and 2, respectively. The critical points marked by black spots are the $\mathfrak{K}$ -values corresponding to bound states. They are symmetric about the $\mathfrak{K}_{r}$ -axis for the $\mathcal{PT}$ -symmetric case.

Note that locating spectral singularities is most conveniently carried out using (73). In the absence of an analogous equation giving the $k$ values for the bound states, we use $N(\rho, \theta)$ to locate the latter. Figure 7 shows the density plots of $N(\rho, \theta)$ for the $\mathcal{PT}$ -symmetric case $\mathfrak{z}_{\pm} = -8 \pm 3\mathrm{i}$ and the non- $\mathcal{PT}$ -symmetric case $\mathfrak{z}_{-} = -8 + 3\mathrm{i}, \mathfrak{z}_{+} = -4 - 2\mathrm{i}$ . These resemble the phase diagrams of statistical mechanics where the critical points correspond to the bound states. As we expect, for the $\mathcal{PT}$ -symmetric case the location of these points is symmetric about the real axis in the complex $\mathfrak{K}$ -plane.

Figure 8 shows the graphs of $N(\rho, \pi - \epsilon)$ for the $\mathcal{PT}$ -symmetric case $\mathfrak{z}_{\pm} = -1 \pm 8\mathrm{i}$ and the non- $\mathcal{PT}$ -symmetric case $\mathfrak{z}_{-} = -2 + 7\mathrm{i}, \mathfrak{z}_{+} = -4 - 5\mathrm{i}$ . These show the distance between the bound states from the origin. For the PT-symmetric case the bound states are created in complex-conjugate pairs with the same distance from the origin. This explains the fact that the number of bound states changes in increments of 2. This is clearly not the case for the non-PT-symmetric case. For both of the above choices of the coupling constants, $\sigma < 17$ . Therefore, the maximum value of each curve gives the total number of bound states for the corresponding system.

![](images/f79d1277c51c63426d280b213b259bb2f86a3ba72432962a30fe98de72b217ed.jpg)

![](images/bb0d705a5f250e5e63d3059bea610ed962f62f665a5fc33ec2667e40417b9dec.jpg)
Figure 8. Plots of $N(\rho, \pi - .01)$ (top figure) and $N(\sigma, \theta)$ (bottom figure) for the $\mathcal{PT}$ -symmetric system defined by $\mathfrak{z}_{\pm} = -1 \pm 8\mathrm{i}$ (the solid curves) and the non- $\mathcal{PT}$ -symmetric system defined by $\mathfrak{z}_{-} = -2 + 7\mathrm{i}$ and $\mathfrak{z}_{+} = -4 - 5\mathrm{i}$ (the dashed curves). For the $\mathcal{PT}$ -symmetric model $N(\rho, \pi - 0.01)$ changes in increments of 2 while $N(\sigma, \theta)$ is symmetric with respect to the $\theta = \pi / 2$ line.

Figure 9 shows a contour plot of $N_{\mathrm{tot}} := N(\sigma, \pi - \epsilon)$ for $z_{\pm} = 1 + i s_{\pm}$ and $\epsilon = 10^{-6}$ as functions of $s_{\pm} \in R$ . Although the real part of the coupling constants are positive and equal, for large enough values of their imaginary part the system develops bound states. This is in contrast to the single-delta-function potential where there are no bound states for coupling constants with a positive real part. We also see that in the PT-symmetric case $s_{+} = -s_{-}$ , which corresponds to the depicted diagonal line, the number of bound states change in increments of 2. This is consistent with the fact that these are produced in complex-conjugate pairs.

## 3.4. Real bound states and quasi-Hermiticity

An important feature of the graphical demonstration of the location of spectral singularities and bound states in the complex $\mathfrak{K}$ -plane is that for the cases that $\mathrm{Re}(\mathfrak{z}_{\pm}) > 0$ and $|\mathrm{Im}(\mathfrak{z}_{\pm})|$ are sufficiently small, the system does not have any spectral singularities or bound states. Figures 4 and 9 provide a clear demonstration of this phenomenon for the case that $\mathrm{Re}(\mathfrak{z}_{\pm}) = 1$ .

The presence of spectral singularities is an obstruction to the quasi-Hermiticity of the Hamiltonian operator. This is also true for the bound states unless they happen to have real energies (eigenvalues). We will refer to these bound states as “real bound states”. It is not difficult to see that generic bound states are not real. In this subsection, we shall first derive analytic expressions for the existence and location of real bound states and then for fixed and positive values of $\mathrm{Re}(\mathfrak{z}_{\pm})$ we establish the existence of a positive lower bound on the size of a region in the $\mathrm{Im}(\mathfrak{z}_{-})-\mathrm{Im}(\mathfrak{z}_{+})$ plane where the system is free of both the spectral singularities and bound states. This is a region where the Hamiltonian operator is quasi-Hermitian. It is in this region that we can employ the machinery of pseudo-Hermitian quantum mechanics [9, 10] to construct an associated positive-definite metric operator and use the Hamiltonian operator to define a unitary quantum system.

![](images/44b116c0fed96b88d4d7ec022a2af16c33e3d49f32b0abdfa54ce1c724e00186.jpg)
Figure 9. Contour plot of the number $N_{\mathrm{tot}}$ of bound states located in the region: $\pi / 2 + \epsilon \leqslant \arg(\mathfrak{K}) \leqslant 3\pi / 2 - \epsilon$ for $\mathfrak{z}_{\pm} = 1 + s_{\pm}\mathrm{i}$ and $\epsilon = 10^{-6}$ . As the color changes from the lightest to the darkest $N_{\mathrm{tot}}$ take values 0, 1, 2, 3, 4, respectively. The diagonal line $s_{+} = -s_{-}$ corresponds to the PT-symmetric region along which the number of bound states changes in increments of 2.

Excluding the case of a single-delta-function potential [24] where $\mathfrak{z}_{+}\mathfrak{z}_{-} = 0$ , we can express (92) as

$$
\left(\frac {\mathfrak {K}}{\mathfrak {z} _ {+}} - 1\right) \left(\frac {\mathfrak {K}}{\mathfrak {z} _ {-}} - 1\right) = \mathrm{e} ^ {2 \mathfrak {K}}.\tag{108}
$$

According to (93), the real bound states are given by real and negative solutions of (108). For these solutions the right-hand side of (108) is real, positive and less than 1. Equating the left-hand side with its complex conjugate yields

$$
\operatorname{Im} \left(\mathfrak {z} _ {-} \mathfrak {z} _ {+}\right) \mathcal {K} = | \mathfrak {z} _ {-} | ^ {2} \operatorname{Im} \left(\mathfrak {z} _ {+}\right) + | \mathfrak {z} _ {+} | ^ {2} \operatorname{Im} \left(\mathfrak {z} _ {-}\right).\tag{109}
$$

In order to explore the consequences of this equation we introduce the notation

$$
r _ {\pm} := \operatorname{Re} (\mathfrak {z} _ {\pm}), \quad s _ {\pm} := \operatorname{Im} (\mathfrak {z} _ {\pm}),\tag{110}
$$

and consider the following cases separately.

(i) $\operatorname{Im}(\mathfrak{z}_{-}\mathfrak{z}_{+}) = 0$ . In this case,

$$
r _ {-} s _ {+} + r _ {+} s _ {-} = 0, \quad \frac {s _ {-}}{| \mathfrak {z} _ {-} | ^ {2}} + \frac {s _ {+}}{| \mathfrak {z} _ {+} | ^ {2}} = 0.\tag{111}
$$

Therefore, either both $s_{\pm}$ vanish and the potential is real or both $s_{\pm}$ are nonzero. In the latter case, (111) implies $\mathfrak{z}_{-} = \mathfrak{z}_{+}^{*}$ . This is the PT-symmetric case for which (92) reduces to

$$
| \mathcal {K} - \mathfrak {z} _ {+} | = | \mathfrak {z} _ {+} | e ^ {\mathcal {K}}.\tag{112}
$$

![](images/656f09f431cb9c5703d2ab69dc933b98f959a151e44ceb73b2ffa9fa46d651c5.jpg)
Figure 10. Curves in the complex $\mathfrak{z}$ -plane along which real bound sates exist for $\mathfrak{z}_{\pm} = \mathfrak{z}e^{\pm iv / 2}$ , $v = \pi (2n - 1) / 20$ and $n \in \{1,2,\dots,10\}$ . The numbers attached to each curve segment is the corresponding value of $n$ . $r$ and $s$ respectively mark the $\operatorname{Re}(\mathfrak{z})$ - and $\operatorname{Im}(\mathfrak{z})$ -axes. Note that all the curves have finite length.

Because $e^{\mathfrak{K}} < 1$ , this equation cannot be satisfied, if $\operatorname{Re}(\mathfrak{z}_{+}) \geqslant 0$ . This is consistent with the results of [28]. Furthermore, for the non-PT cases with real $z_{-}z_{+}$ , such as $z_{-} = -z_{+}^{*}$ or imaginary $z_{\pm}$ with $z_{-} \neq z_{+}^{*}$ , there are no real bound states.

(ii) $\operatorname{Im}(\mathfrak{z}_{-}\mathfrak{z}_{+}) \neq 0$ . In this case, we can write (109) as

$$
\mathfrak {K} = \frac {| \mathfrak {z} _ {-} | ^ {2} \operatorname{Im} (\mathfrak {z} _ {+}) + | \mathfrak {z} _ {+} | ^ {2} \operatorname{Im} (\mathfrak {z} _ {-})}{\operatorname{Im} (\mathfrak {z} _ {-} \mathfrak {z} _ {+})}.\tag{113}
$$

Substituting this equation into (92) gives a rather complicated relation between $z_{-}$ and $z_{+}$ . This relation together with the requirement that the right-hand side of (113) be negative provide the necessary and sufficient condition for the existence of real bound states for non-PT-symmetric cases. We have implemented this condition to address the existence of real bound states for the special cases where $z_{+} = z_{-} e^{iv} := z e^{iv/2}$ with $v \in [0, 2\pi)$ . Figure 10 shows the curves in the complex z-plane along which real bound states exist for various values of v. It is important to note that all these curves are finite in length. Therefore, there are no real bound states for sufficiently large values of $|z|$ .

Next, we wish to show the existence of regions in the space of the coupling constants $z_{\pm}$ where there are no spectral singularities or bound states. Our main tools are the following basic theorems of real and complex analysis.

Theorem 2. Let $n \in \mathbb{Z}^+$ , $D$ be a compact subset of $\mathbb{R}^n$ with its standard topology, and $\varphi : \mathbb{R}^n \to \mathbb{R}$ be a function that is continuous on $D$ . Then $\{\varphi(\vec{x}) | \vec{x} \in D\}$ has both a minimum and a maximum [37, section 8].

Theorem 3 (Maximum modulus theorem). Let C be a contour bounding a compact and simply-connected subset R of the complex plane and $h : C \to C$ be a function that is analytic on an open subset containing R. Then $\{|h(w)||w \in R\}$ attains its maximum on C, [38, section III.1].

First, we use theorem 3 to prove the following preliminary results.

Lemma 1. Let $\mathfrak{D}_{\rho}$ denote the following half-disc of radius $\rho \in \mathbb{R}^{+}$ :

$$
\mathfrak {D} _ {\rho} := \{\mathfrak {K} \in \mathbb {C} | | \mathfrak {K} | \leqslant \rho , \operatorname{Re} (\mathfrak {K}) \leqslant 0 \},
$$

and $L:\mathbb{C}\to \mathbb{C}$ be the function defined by

$$
L (\mathfrak {K}) := \left\{ \begin{array}{c c} \frac {1 - \mathrm{e} ^ {2 \mathfrak {K}}}{\mathfrak {K}} & \quad \text { for } \quad \mathfrak {K} \neq 0, \\ - 2 & \quad \text { for } \quad \mathfrak {K} = 0. \end{array} \right.\tag{114}
$$

Then $|L|$ attains its maximum value on $\mathfrak{D}_{\rho}$ at $\mathfrak{K} = 0$ , i.e., $2 = |L(0)|$ is the maximum of $\mathcal{A}_{\rho} := \{|L(\mathfrak{K})| \mid \mathfrak{K} \in \mathfrak{D}_{\rho}\}$ for all $\rho \in \mathbb{R}^{+}$ .

Proof. First, consider the case $\rho < 1$ . Then $\mathfrak{D}_{\rho} \subsetneq D_{1}$ , which implies $\mathcal{A}_{\rho} \subseteq \mathcal{A}_{1}$ . Therefore, the maximum of $\mathcal{A}_{\rho}$ is less than or equal to that of $\mathcal{A}_{1}$ . This shows that it is sufficient to prove the lemma for the case $\rho \geqslant 1$ . Because $L$ is an entire function and $\mathfrak{D}_{\rho}$ is compact, according to theorem 3, $\mathcal{A}_{\rho}$ has a maximum that is located on the boundary of $\mathfrak{D}_{\rho}$ . This is the union of the closed line segment $\ell_{\rho} := \{\mathrm{i}y|y \in [-\rho, \rho]\}$ and the open semicircle $C_{\rho} := \{\mathrm{i}\rho \mathrm{e}^{\mathrm{i}\varphi}|\varphi \in (0,\pi)\}$ . The maximum of $\mathcal{A}_{\rho}$ is the largest of the values taken by $|L|$ on $\ell_{\rho}$ and $C_{\rho}$ . We will show that these values are bounded from above by 2. Because $0 \in \mathfrak{D}_{\rho}$ and $|L(0)| = 2$ , this is sufficient to prove the lemma. In the following we consider the values of $|L|$ on $\ell_{\rho}$ and $C_{\rho}$ separately.

\- For all $\mathfrak{K} \in \ell_{\rho}$ , we can write $\mathfrak{K} = iy$ for some $y \in [-\rho, \rho]$ . Inserting $\mathfrak{K} = iy$ into (114) and computing the modulus of both sides of the resulting expression yields

$$
| L (\mathfrak {K}) | = \frac {2 \sin y}{y} \leqslant 2.\tag{115}
$$

\- For all $\mathfrak{K} \in C_{\rho}$ , we can write $\mathfrak{K} = \mathrm{i}\rho \mathrm{e}^{\mathrm{i}\varphi}$ for some $\varphi \in (0, \pi)$ . Because $\rho \geqslant 1$ and $\sin \varphi > 0$ , (114) implies

$$
| L (\mathfrak {K}) | = \frac {| 1 - \exp (2 \mathrm{i} \rho \mathrm{e} ^ {\mathrm{i} \varphi}) |}{\rho} \leqslant 1 + | \exp (2 \mathrm{i} \rho \mathrm{e} ^ {\mathrm{i} \varphi}) | = 1 + \mathrm{e} ^ {- 2 \rho \sin \varphi} <   2.\tag{116}
$$

This together with (115) proves the lemma for $\rho \geqslant 1$ . As we explained above this establishes the statement of the lemma also for the case $\rho < 1$ .

Lemma 2. Suppose that $r_{\pm} > 0$ . Then $\mathfrak{K} = 0$ is a first-order zero of the function $F_{\vec{3}}$ defined by (98).

Proof. Recall that the zeros of $F_{\vec{3}}$ are at most of order 3 and $F_{\vec{3}}(0)=0$ . Therefore, it is sufficient to show that K=0 is not a second- or third-order zero of $F_{\vec{3}}$ . Assume (by contradiction) that K=0 is a second-order zero of $F_{\vec{3}}$ . Then $F_{\vec{3}}^{\prime}(0)=0$ , i.e., $z_{-}+z_{+}+2z_{-}z_{+}=0$ . Equivalently, we have

$$
r _ {+} + r _ {-} + 2 (r _ {-} r _ {+} - s _ {-} s _ {+}) = 0, \quad s _ {+} + s _ {-} + 2 (r _ {-} s _ {+} + r _ {+} s _ {-}) = 0.
$$

Solving the second of these for $s_{-}$ and inserting the result into the first, we find

$$
r _ {+} + r _ {-} + 2 r _ {-} r _ {+} + \frac {2 (1 + 2 r _ {-}) s _ {+} ^ {2}}{1 + 2 r _ {+}} = 0.
$$

But this equation cannot be satisfied for $r_{\pm} > 0$ . This shows that the above assumption is false and $\mathfrak{K} = 0$ is not a second-order zero of $F_{\vec{j}}$ . Next, we recall that $\mathfrak{K} = 0$ is a third-order zero of $F_{\vec{j}}$ if and only if (100) hold. But these conflict with the condition $r_{\pm} > 0$ . Hence $\mathfrak{K} = 0$ is not a third-order zero of $F_{\vec{j}}$ .

Next, we use theorem 2 and lemmas 1 and 2 to prove the following desired result.

Theorem 4. Suppose that $r_{\pm} > 0$ and $|s_{\pm}| < r_{\max} := \max(r_{-}, r_{+})$ . Then there is a positive upper bound $B_{\bar{r}}$ on $|s_{\pm}|$ such that for all $s_{\pm}$ satisfying $|s_{\pm}| < B_{\bar{r}}$ , the Hamiltonian (3) does not have any spectral singularities or bound states $^{8}$ .

Proof. Recall that spectral singularities and bound states are zeros $\mathcal{K}$ of $F_{\mathfrak{z}}$ with $\operatorname{Re}(\mathcal{K}) \leqslant 0$ and that they belong to $D_{\sigma}$ , where $\sigma := 2\max(|\mathfrak{z}_{-}|, |\mathfrak{z}_{+}|)$ . The latter is a subset of the half-disc

$$
\mathfrak {D} := \mathfrak {D} _ {\sqrt {8} r _ {\max}} = \{\mathfrak {K} \in \mathbb {C} | | \mathfrak {K} | \leqslant \sqrt {8} r _ {\max}, \operatorname{Re} (\mathfrak {K}) \leqslant 0 \},
$$

because in view of $r_{\pm} \leqslant r_{\max}$ and $|s_{\pm}| < r_{\max}$ , we have $\sigma < \sqrt{8} r_{\max}$ . According to lemma 2, $\mathfrak{K} = 0$ is a first-order zero of $F_{\vec{j}}$ . This implies that the function $G_{\vec{j}}: \mathbb{C} \to \mathbb{C}$ defined by

$$
G _ {\vec {3}} (\mathfrak {K}) := \left\{ \begin{array}{l l} \mathfrak {K} ^ {- 1} F _ {\vec {3}} (\mathfrak {K}) & \quad \text { for } \quad \mathfrak {K} \neq 0, \\ F _ {\vec {3}} ^ {\prime} (0) & \quad \text { for } \quad \mathfrak {K} = 0 \end{array} \right.\tag{117}
$$

is an entire function and $G_{\mathfrak{z}}^{-}(0) \neq 0$ . Furthermore, the spectral singularities and bound states of the Hamiltonian (3) correspond to the zeros $K_{0}$ of $G_{\mathfrak{z}}^{-}$ lying in D. Another important observation is that $G_{\bar{r}}$ has no zeros K with $\operatorname{Re}(\mathfrak{K}) \leqslant 0$ , because if they existed these zeros would have corresponded to the spectral singularities or bound states of the Hamiltonian (3) with real and positive coupling constants ( $j_{\pm} \in R^{+}$ ). But as we argued above this Hamiltonian does not have any spectral singularities or bound states. This observation establishes the fact that

$$
G _ {\vec {r}} (\mathfrak {K}) \neq 0, \quad \text {   for   all   } \quad \mathfrak {K} \in \mathfrak {D}.\tag{118}
$$

Because $G_{\vec{r}}$ is an entire function, $|G_{\vec{r}}|$ is continuous on D which is a compact subset of $C = R^{2}$ . In view of theorem 2, this implies that the set $\{|G_{\vec{r}}(\mathfrak{K})| \mid \mathfrak{K} \in \mathfrak{D}\}$ has a minimum $m_{\vec{r}}$ , i.e., there is $K_{min} \in D$ such $m_{\vec{r}} = |G_{\vec{r}}(\mathfrak{K}_{\min})|$ . Because $K_{0}, K_{min} \in D$ and (118) holds, we have

$$
0 <   | G _ {\vec {r}} (\mathfrak {K} _ {\min}) | = m _ {\vec {r}} \leqslant | G _ {\vec {r}} (\mathfrak {K} _ {0}) |.\tag{119}
$$

Next, we introduce $J_{\vec{3}}:\mathbb{C}\to \mathbb{C}$ as the function defined by

$$
J _ {\vec {3}} (\mathfrak {K}) := G _ {\vec {3}} (\mathfrak {K}) - G _ {\vec {r}} (\mathfrak {K}).\tag{120}
$$

Because $G_{\vec{\mathfrak{z}}}(\mathfrak{K}_0) = 0$ , we have

$$
| J _ {\vec {z}} (\mathfrak {K} _ {0}) | = | G _ {\vec {r}} (\mathfrak {K} _ {0}) |.\tag{121}
$$

Furthermore, in view of (117), (120), (114), and the fact that $\mathfrak{K}_0\neq 0$

$$
J _ {\tilde {3}} (\mathfrak {K} _ {0}) = - i (s _ {-} + s _ {+}) + [ - s _ {-} s _ {+} + i (r _ {-} s _ {+} + r _ {+} s _ {-}) ] L (\mathfrak {K} _ {0}).\tag{122}
$$

This implies

$$
\begin{array}{l} | J _ {\vec {3}} (\mathfrak {K} _ {0}) | \leqslant | s _ {-} | + | s _ {+} | + (| s _ {-} | | s _ {+} | + | r _ {-} | | s _ {+} | + | r _ {+} | | s _ {-} |) | L (\mathfrak {K} _ {0}) | \\ \leqslant 2   (3 r _ {\max} + 1)   s _ {\max}, \end{array}\tag{123}
$$

where $s_{\max} := \max(|s_{-}|, |s_{+}|)$ and we have used the triangular inequality, the condition $|s_{\pm}| \leqslant r_{\max}$ , and $|L(\mathfrak{K}_{0})| \leqslant 2$ that follows from lemma 1.

![](images/5d096c9df1671f25ec47ed12dca70f922c4c06a04693ed5ff662ac606c8f539f.jpg)
Figure 11. Plots of $|G_{\vec{r}}(\gamma_{\vec{r}}(t))|$ (the full curve) and $|L(\gamma_{\vec{r}}(t))|$ (the dashed curve) as a function of $t \in [-1, 1]$ . $t_{-} \approx -0.949$ and $t_{+} \approx -0.051$ are the minimum points of $|G_{\vec{r}}(\gamma_{\vec{r}}(t))|$ corresponding to $\mathfrak{K}_{\pm} \approx \pm 1.795\mathrm{i}$ . These give the minimum value $m_{\vec{r}} \approx 1.906$ . The maximum value of $|L(\gamma_{\vec{r}}(t))|$ is 2 that is attained at $t_0 = -0.5$ corresponding to $\mathfrak{K} = 0$ .

If we combine (123) with (121) and (119), we obtain

$$
0 <   \frac {m _ {\vec {r}}}{2 (3 r _ {\max} + 1)} \leqslant s _ {\max}.\tag{124}
$$

This inequality is violated for the values of $s_{\pm}$ for which

$$
\left| s _ {\pm} \right| <   \frac {m _ {\vec {r}}}{2 \left(3 r _ {\max} + 1\right)} =: B _ {\vec {r}}.\tag{125}
$$

Therefore, for the cases that $|s_{\pm}| < B_{\vec{r}}$ the existence of $\mathfrak{K}_0$ leads to a contradiction; such a $\mathfrak{K}_0$ cannot exist; and there are no spectral singularities or bound states.

The upper bound $B_{\vec{r}}$ given in (125) involves the minimum $m_{\vec{r}}$ of $|G_{\vec{r}}|$ on the half-disc $\mathfrak{D}$ . Because $G_{\vec{r}}$ is a nowhere-zero analytic function on $\mathfrak{D}$ , $1 / G_{\vec{r}}$ is also analytic on $\mathfrak{D}$ . Hence, according to theorem 3, $1 / |G_{\vec{r}}|$ attains its maximum $M_{\vec{r}}$ on the boundary of $\mathfrak{D}$ . It is not difficult to see that $m_{\vec{r}} = 1 / M_{\vec{r}}$ . Therefore, in practice, for given values of $r_{\pm}$ , we can obtain $m_{\vec{r}}$ by exploring the values of $|G_{\vec{r}}|$ on the boundary of $\mathfrak{D}$ .

We can identify the boundary of $\mathfrak{D}$ with the graph $\Gamma_{\vec{r}}$ of the parameterized curve:

$$
\gamma_ {\vec {r}} (t) := 2 \mathrm{i} r _ {\max} [ (2 t + 1) \Theta (- t) + \mathrm{e} ^ {\mathrm{i} \pi t} \Theta (t) ], \qquad t \in [ - 1, 1 ],\tag{126}
$$

where $\Theta$ is the unit step function: $\Theta(0):=1/2$ and $\Theta(t):=(1+t/|t|)/2$ for $t\neq0$ . Figure 11 shows the graphs of $|G_{\vec{r}}(\gamma_{\vec{r}}(t))|$ and $|L(\gamma_{\vec{r}}(t))|$ for the case $r_{\pm}=1$ that is considered in figures 4 and 9. In this case, $r_{max}=1$ and D is the half-disc of radius 2 lying in $\Pi_{-}$ . As seen from the graph of values of $|L|$ , it attains its maximum at t = -0.5 (corresponding to $\Re = 0$ ) and its maximum value is 2. This is consistent with the statement of lemma 1. The minimum points of $|G_{\vec{r}}|$ are located at t = -0.949 and t = -0.051. These correspond to $\Re_{min} \approx \pm 1.795i$ where $|G_{\vec{r}}|$ takes its minimum value: $m_{\vec{r}} \approx 1.906$ . According to (125), this gives $B_{\vec{r}} = m_{\vec{r}} / 8 \approx 0.238$ . Therefore, for $z_{\pm} = 1 \pm is_{\pm}$ with $|s_{\pm}| < 0.238$ there should be no spectral singularities or bound states. This is in complete agreement with the graphical data depicted in figures 4 and 9; the disc with center $s_{\pm} = 0$ and radius 0.238 lies in the region with no spectral singularities or bound states.

## 4. Concluding remarks

In this paper we provided an explicit demonstration of how spectral singularities obstruct the existence of a biorthonormal eigensystem and render the Hamiltonian non-diagonalizable. We achieved this by obtaining a characterization of spectral singularities in terms of the $M_{22}$ entry of the matrix M of equation (20). In particular we showed that while bound states are zeros of $M_{22}(k)$ with $\operatorname{Im}(k) > 0$ , the spectral singularities are the real zeros of $M_{22}(k)$ . It is not difficult to infer from this observation that, similar to the bound states, the spectral singularities are linked with singularities of the scattering matrix [19]. However, unlike the bound states, they lie on the real axis in the complex k-plane. This in turn suggests interpreting spectral singularities as resonances having a vanishing width. Reference [19] provides a thorough description of this interpretation and its physical implications.

We established the utility of our general results by providing a thorough analysis of the spectral properties of a two-parameter family of complex point interactions. We obtained various results on the nature and location of the bound states and spectral singularities for this family and proved the existence of regions in the space of coupling constants where both bound states and spectral singularities are lacking and the Hamiltonian is quasi-Hermitian.

Throughout our study we examined the consequences of imposing PT-symmetry which corresponds to restricting the coupling constants to a complex plane in the space $C^{2}$ of coupling constants. This revealed a previously unnoticed fact that PT-symmetric double-delta function potential can involve spectral singularities.

The results of this paper may be extended to complex point interactions corresponding to three or a larger number of delta-function potentials. Another line of research is to try to compute a metric operator $\eta_{+}$ and the corresponding equivalent Hermitian Hamiltonian h and the pseudo-Hermitian position and momentum operators X and P for the double-delta function potential whenever the Hamiltonian is quasi-Hermitian. Theorem 4 provides the mathematical basis for a perturbative calculation of $\eta_{+}$ , h, X and P. We plan to report the results of this calculation in a forthcoming publication.

## Acknowledgments

This work has been supported by the Scientific and Technological Research Council of Turkey (TÜBİTAK) in the framework of the project no: 108T009, and by the Turkish Academy of Sciences (TÜBA). We wish to express our gratitude to Professor Gusein Guseinov for preparing and sending us a detailed description of spectral singularities [30].

## References

[1] Pauli W 1943 Rev. Mod. Phys. 15 175 Sudarshan E C G 1961 Phys. Rev. 123 2183

[7] Mostafazadeh A 2002 J. Math. Phys. 43 2814
Mostafazadeh A 2002 J. Math. Phys. 43 3944

[2] Dyson F J 1956 Phys. Rev. 102 1217, 1230
Feshbach H 1962 Ann. Phys., NY 19 287
Löwdin P-O 1988 Adv. Quant. Chem. 19 87

[3] Schomerus H, Frahm K M, Patra M and Beenakker C W J 2000 Physica A 278 469

[4] Berry M V 2003 J. Mod. Opt. 50 63
Berry M V 2004 Czech. J. Phys. 54 1039

[5] Bender C M and Boettcher S 1998 Phys. Rev. Lett. 80 5243

[6] Mostafazadeh A 2002 J. Math. Phys. 43 205

[8] Scholtz F G, Geyer H B and Hahne F J W 1992 Ann. Phys., NY 213 74

[9] Mostafazadeh A and Batal A 2004 J. Phys. A: Math. Gen. 37 11645

[10] Mostafazadeh A 2008 Pseudo-Hermitian quantum mechanics arXiv: 0810.5643

[11] Mostafazadeh A 2006 Int. J. Mod. Phys. A 21 2553
Mostafazadeh A and Zamani F 2006 Ann. Phys., NY 321 2183, 2210
Zamani F and Mostafazadeh A 2008 arXiv:0805.1651

[12] Mostafazadeh A 2003 Class. Quantum Grav. 20 155

[13] Bender C M, Brandt S F, Chen J-H and Wang Q 2005 Phys. Rev. D 71 025014 Jones H F 2008 Phys. Rev. D 77 065023

[14] Matzkin A 2006 J. Phys. A: Math. Gen. 39 10859

[15] Mostafazadeh A and Loran F 2008 Europhys. Lett. 81 10007

[17] Heiss W D 2004 J. Phys. A: Math. Gen. 37 2455

[16] Samsonov B F 2005 J. Phys. A: Math. Gen. 38 L397

[18] Heiss W D, Müller M and Rotter I 1998 Phys. Rev. E 58 2894
Stehmann T, Heiss W D and Scholtz F G 2004 J. Phys. A: Math. Gen. 37 7813
Dembowski C, Dietz B, Gräf H-D, Harney H L, Heine A, Heiss W D and Richter A 2004 Phys. Rev. E 69 056216
Mailybaev A A, Kirillov O N and Seyranian A P 2005 Phys. Rev. A 72 014104

[19] Mostafazadeh A 2009 arXiv:0901.4472

[20] Samsonov B F 2005 J. Phys. A: Math. Gen. 38 L571

[21] Naimark M A 1960 Am. Math. Soc. Trans. 16 103

[22] Ljance V É 1967 Am. Math. Soc. Trans. 60 185, 227

[23] Bender C M 2007 Rep. Prog. Phys. 70 947

[24] Mostafazadeh A 2006 J. Phys. A: Math. Gen. 39 13506

[25] Jones H F 1999 Phys. Lett. A 262 242

[26] Ahmed Z 2001 Phys. Lett. A 286 231

[27] Albeverio S, Fei A-M and Kurasov P 2002 Lett. Math. Phys. 59 227

[28] Uncu H and Demiralp E 2006 Phys. Lett. A 359 190

[29] Tunca G B and Bairamov E 1999 Czech. Math. J. 49 689

[30] Guseinov G On the concept of spectral singularities (unpublished)

[31] Naimark M A 1968 Linear Differential Operators: Part II (New York: Ungar)

[32] Kemp R R D 1958 Can. J. Math. 10 447

[33] Schwartz J 1960 Commun. Pure Appl. Math. 13 609

[34] Razavy M 2003 Quantum Theory of Tunneling (Singapore: World Scientific)

[35] Boyce W E and DiPrima R C 2005 Elementary Differential Equations and Boundary Value Problems (New Jersey: Wiley)

[36] Howie J M 2003 Complex Analysis (London: Springer)

[37] Edwards C H 1994 Advanced Calculus of Several Variables (New York: Dover)

[38] Lang S 1999 Complex Analysis (New York: Springer)
