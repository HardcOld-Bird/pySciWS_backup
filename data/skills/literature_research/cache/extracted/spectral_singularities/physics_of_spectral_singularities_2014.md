# Physics of Spectral Singularities

Ali Mostafazadeh

Abstract. Spectral singularities are certain points of the continuous spectrum of generic complex scattering potentials. We review the recent developments leading to the discovery of their physical meaning, consequences, and generalizations. In particular, we give a simple definition of spectral singularities, provide a general introduction to spectral consequences of PT-symmetry (clarifying some of the controversies surrounding this subject), outline the main ideas and constructions used in the pseudo-Hermitian representation of quantum mechanics, and discuss how spectral singularities entered in the physics literature as obstructions to these constructions. We then review the transfer matrix formulation of scattering theory and the application of complex scattering potentials in optics. These allow us to elucidate the physical content of spectral singularities and describe their optical realizations. Finally, we survey some of the most important results obtained in the subject, drawing special attention to the remarkable fact that the condition of the existence of linear and nonlinear optical spectral singularities yield simple mathematical derivations of some of the basic results of laser physics, namely the laser threshold condition and the linear dependence of the laser output intensity on the gain coefficient.

Mathematics Subject Classification (2010). 34L25, 47A40, 78A60.

Keywords. Spectral singularity, complex potential, scattering, zero-width resonance, PT-symmetry, pseudo-Hermitian operator, laser, antilaser.

## 1. Introduction

The term ‘spectral singularity’ entered mathematical literature through the work of Jack Schwartz [1] who coined this name for a mathematical object discovered by Mark Aronovich Naimark in 1954 [2]. Naimark had come across spectral singularities and worked out some of their consequences in his attempts at generalizing the well-known spectral theory of self-adjoint Schrödinger operators,

$$
H = - \frac {d ^ {2}}{d x ^ {2}} + v (x),\tag{1}
$$

defined on the half-line, i.e., for $x \in [0, \infty)$ , to situations where $v(x)$ was a complex scattering potential. This marks the starting point of a comprehensive study of spectral singularities that has attracted the attention of mathematicians for over half a century [3–10]. For further references, see [11].

Following the pioneering work of Naimark, the notion of spectral singularity was generalized in various directions $[3, 5, 6, 8–10]$ . In particular, Kemp $[3]$ considered spectral singularities of the Schrödinger operators (1) defined on the full line, i.e., $x \in R$ . These admit a simple description in terms of certain solutions of the Schrödinger equation $[11]$

$$
- \psi^ {\prime \prime} (x) + v (x) \psi (x) = k ^ {2} \psi (x), \qquad x \in \mathbb {R},\tag{2}
$$

called the Jost solutions.

Let $v(x)$ be a real or complex scattering potential defined on R, and suppose that $|v(x)| \to 0$ as $x \to \pm \infty$ in such a manner that [3]

$$
\int_ {- \infty} ^ {\infty} (1 + | x |) | v (x) | d x <   \infty .\tag{3}
$$

Then for each $k \in R$ , the Schrödinger equation (2) admits a pair of solutions $\psi_{k\pm}$ fulfilling the asymptotic boundary conditions [3]:

$$
\lim _ {x \to \pm \infty} e ^ {\mp i k x} \psi_ {k \pm} (x) = 1, \quad \lim _ {x \to \pm \infty} e ^ {\mp i k x} \psi_ {k \pm} ^ {\prime} (x) = \pm i k.\tag{4}
$$

These are the celebrated Jost solutions of (2).

Definition 1. Let $v: \mathbb{R} \to \mathbb{C}$ be a function satisfying (3) and $H$ be the Schrödinger operator (1) that is defined by $v$ on $\mathbb{R}$ . A real and positive number $k_{\star}^{2}$ is called a spectral singularity of $H$ or $v$ , if the Jost solutions $\psi_{k_{\star}\pm}$ of (2) are linearly dependent.

It is not difficult to see that the Jost solutions correspond to the scattering states of the potential $v(x)$ . But as we shall see below, for real scattering potentials they are always linearly independent. This is why physicists did not pay much attention to spectral singularities throughout the twentieth century.

The situation began to change in 1998 by the discovery of a class of complex potentials that possessed a real spectrum [12]. A well-known example is $v(x) = ix^3$ , whose spectrum is discrete, real, and positive [13]. This unexpected result was initially associated with the fact that the corresponding Schrödinger operator (1) was invariant under the parity-time-reversal transformation, also known as spacetime reflection [12],

$$
\psi (x) \longrightarrow \mathcal {P T} \psi (x) = \psi (- x) ^ {*},\tag{5}
$$

where $\psi$ is an arbitrary square-integrable function, i.e., $\psi \in L^{2}(\mathbb{R})$ , and P and T are respectively the parity and time-reversal operators defined by

$$
\mathcal {P} \psi (x) := \psi (- x), \quad \mathcal {T} \psi (x) := \psi (x) ^ {*}.\tag{6}
$$

We postpone the discussion of the spectral implications of PT-symmetry to Section 2. Here we suffice to mention that during the last 16 years there has been a great interest in the study of PT-symmetric potentials. A substantial amount of the early work on this topic consisted of searching for other examples of complex potentials possessing a real spectrum. A rather straightforward method of constructing such potentials is to generate them from real potentials via non-unitary similarity transformations. The simplest example is a complex translation of the form $v(x) \to v(x + \mathfrak{z})$ , where $\mathfrak{z}$ is a complex parameter [15]. To the best of our knowledge, it was in this context that spectral singularities entered in the physics literature.

In 2005 Boris Samsonov noted that the application of complex translations on a real potential could yield complex scattering potentials supporting spectral singularities [16]. He also proposed means of removing these spectral singularities by performing certain supersymmetry transformations. In Samsonov's words this meant “curing” a “disease”, for he maintained that “Hamiltonians with spectral singularities are ‘bad’.” This is a typical reaction when one encounters a ‘singularity’. However, the history of science teaches us that some of the greatest discoveries of mankind have their root at unwanted ‘singularities’. One of the aims of the present article is to show that the same applies to spectral singularities, as they provide the mathematical basis for one of the most important discoveries of all times, namely lasers.

## 2. PT-Symmetry versus Pseudo-Hermiticity

Samsonov's article [16] failed to provide the necessary incentive for the study of the physical aspects of spectral singularities. But soon after, spectral singularities were to reveal their presence in the study of a delta-function potential with an imaginary coupling constant, i.e., $v(x) = i\alpha \delta(x)$ with $\alpha \in \mathbb{R}$ , [17]. The motivation for this study was provided by attempts at finding a set of necessary and sufficient conditions for the reality of the spectrum of a non-Hermitian linear operator $H$ . A well-advertised claim is that $\mathcal{PT}$ -symmetry provides such a condition [18]. This is certainly not true if we take (6) as the definition of $\mathcal{P}$ and $\mathcal{T}$ , for there are infinity of examples of real potentials, such as $v(x) = x^2 + \sin x$ , that do not commute with $\mathcal{PT}$ but have a real spectrum.

In order to ensure the validity of the above claim, we need to reinterpret what we mean by PT-symmetry or generalize it appropriately. First, we recall the following obvious consequences of (6).

$$
[ \mathcal {P}, \mathcal {T} ] = 0,
$$

$$
\mathcal {P} ^ {2} = \mathcal {T} ^ {2} = (\mathcal {P T}) ^ {2} = I,\tag{7}
$$

where I stands for the identity operator. The following is a precise definition of PT-symmetry.

Definition 2. Let P and T be the linear operators defined on $L^{2}(\mathbb{R})$ by (6). Then a linear operator H acting in $L^{2}(\mathbb{R})$ is said to be PT-symmetric, if it commutes with PT, i.e., $[H, PT] = 0$ . Moreover, suppose that H has a discrete and non-degenerate spectrum, and there is a complete set $^{1}$ of eigenvectors $\psi_{n}$ of H that are also eigenvectors of PT. Then H is said to have an unbroken or exact PT-symmetry.

For the cases where H has an exact PT-symmetry, there are $\epsilon_{n} \in C$ such that $PT\psi_{n} = \epsilon_{n}\psi_{n}$ . Then, in view of (7), $\psi_{n} = (\mathcal{PT})^{2}\psi_{n} = \mathcal{PT}(\epsilon_{n}\psi_{n}) = |\epsilon_{n}|^{2}\psi_{n}$ . This shows that $\epsilon_{n} = e^{i\alpha_{n}}$ for some real number $\alpha_{n}$ . Now, setting $\tilde{\psi}_{n} := e^{i\alpha_{n}/2}\psi_{n}$ , we find $PT\tilde{\psi}_{n} = e^{-i\alpha_{n}/2}PT\psi_{n} = e^{i\alpha_{n}/2}\psi_{n} = \tilde{\psi}_{n}$ . Therefore, exact PT-symmetry of H means the existence of a complete set of PT-invariant eigenvectors of H. This argument relies only on the fact that PT is an antilinear operator $^{2}$ squaring to I. Therefore, it applies to all such operators $X^{3}$ . We use this observation to introduce the notion of exact anti-linear symmetry.

Definition 3. Let H and X be respectively linear and antilinear operators acting in a Hilbert space H, and I be the identity operator on H. H is said to be X-symmetric, if $[X,H]=0$ . Furthermore, suppose that H has a discrete spectrum and $X^{2}=I$ . Then H is said to have an exact X-symmetry if there is a complete set of eigenvectors $\psi_{n}$ of H that are invariant under X, i.e., $X\psi_{n}=\psi_{n}$ .

The following is a useful property of exact antilinear symmetry. Its application to PT-symmetry is the reason for the claim that exact PT-symmetry implies the reality of the spectrum.

Theorem 1. Eigenvalues of every linear operator H that has an exact anti-linear symmetry are real.

Proof. Let $E_{n}$ be an eigenvalue of H and $\psi_{n}$ be a corresponding X-invariant eigenvector. Then, in view of the fact that $X\psi_{n} = \psi_{n}$ and $[X, H] = 0$ ,

$$
\begin{array}{r c l} E _ {n} ^ {*} & = & \frac {\langle \psi_ {n} | E _ {n} ^ {*} \mathcal {X} \psi_ {n} \rangle}{\langle \psi_ {n} | \psi_ {n} \rangle} = \frac {\langle \psi_ {n} | \mathcal {X} E _ {n} \psi_ {n} \rangle}{\langle \psi_ {n} | \psi_ {n} \rangle} = \frac {\langle \psi_ {n} | \mathcal {X} H \psi_ {n} \rangle}{\langle \psi_ {n} | \psi_ {n} \rangle} \\ & = & \frac {\langle \psi_ {n} | H \mathcal {X} \psi_ {n} \rangle}{\langle \psi_ {n} | \psi_ {n} \rangle} = \frac {\langle \psi_ {n} | H \psi_ {n} \rangle}{\langle \psi_ {n} | \psi_ {n} \rangle} = E _ {n}. \end{array}
$$

This theorem suggests generalizing PT-symmetry to the presence of an antilinear symmetry. In order to avoid using the same symbols for different concepts, we use ‘PT-symmetry’ to refer to this generalization.

Definition 4. We say that $H$ is PT-symmetric if it has an exact antilinear symmetry.

In view of this definition, Theorem 1 is equivalent to the statement that PT-symmetry is a sufficient condition for the reality of the spectrum of H. The converse can also be established if H is finite-dimensional or if we impose a further technical condition $^{4}$ on H, [20]. Therefore, introducing the above notion of PT-symmetry secures the validity of the claim that it is a necessary and sufficient condition for the reality of the spectrum of a large class of linear operators. However the price one pays for doing so is a clear distinction between PT-symmetry and the parity-time-reversal (spacetime reflection) symmetry that we label by PT-symmetry. Indeed unlike Hermiticity which is a sufficient condition for the reality of the spectrum, PT-symmetry is both necessary and sufficient. But this PT-symmetry does not mean spacetime reflection (PT) symmetry. Similarly to Hermiticity exact PT-symmetry is a sufficient but not necessary condition for the reality of the spectrum of a linear operator. Non-exact PT-symmetry, which can be immediately checked for a given operator, is neither necessary nor sufficient.

A major difficulty with the above notion of PT-symmetry is that there does not exist a universal choice for the antilinear symmetry appearing in Definition 4; each PT-symmetric linear operator H has its own set of anti-linear operators X which make H exactly X-symmetric. If we denote this set by $S_{H}$ , then PT-symmetry of H is equivalent to the condition that $S_{H}$ is nonempty. In practice, in order to check if this is the case, one must determine an appropriate set of eigenvectors $\psi_{n}$ of H, make sure that they form a complete set, and try to construct an antilinear operator X that leaves $\psi_{n}$ 's invariant and squares to I. This is generally a difficult task.

The question of finding a necessary and sufficient condition for the reality of the spectrum of a non-Hermitian operator has a more illuminating answer.

Theorem 2. Let H be a linear operator with a complete set of eigenvectors that acts in a finite-dimensional inner-product space. Then H has a real spectrum if and only if there is a positive-definite operator $\eta_{+}$ intertwining H and its adjoint $H^{\dagger}$ , i.e.,

$$
H ^ {\dagger} \eta_ {+} = \eta_ {+} H,\tag{8}
$$

alternatively $H^{\dagger}$ is related to H by the similarity transformation:

$$
H ^ {\dagger} = \eta_ {+} H \eta_ {+} ^ {- 1}.\tag{9}
$$

Conditions (8) and (9), that we call ‘ $\eta_{+}$ -pseudo-Hermiticity’ of H, was derived in Refs. [20,22,23] for a more general class of operators. These act in a possibly infinite-dimensional Hilbert space H, have a discrete spectrum, and possess a complete biorthonormal eigensystem $\{(\psi_{n},\phi_{n})\}$ , [19]. The latter means the existence of a sequence of complex numbers $\{E_{n}\}$ and a pair of sequences of vectors, $\{\psi_{n}\}$ and $\{\phi_{n}\}$ , which satisfy

$$
H \psi_ {n} = E _ {n} \psi_ {n}, H ^ {\dagger} \phi_ {n} = E _ {n} ^ {*} \phi_ {n},\tag{10}
$$

$$
\langle \psi_ {m} | \phi_ {n} \rangle = \delta_ {m n}, \quad \sum_ {n} | \psi_ {n} \rangle \langle \phi_ {n} | = I.\tag{11}
$$

Here $\langle\cdot|\cdot\rangle$ stands for the inner product of H. We use the term ‘diagonalizable’ to mean that H admits a complete biorthonormal eigensystem.

Theorem 2 admits an infinite-dimensional generalization provided that we impose further restrictions on H and $\eta_{+}$ [19]. For pedagogical reasons we postpone discussing these to the final three paragraphs of this section. For the moment, we follow the physicists' tradition of assuming that what we know about finite dimensions is essentially valid in infinite dimensions. For example, we take the following condition as the definition of a Hermitian or self-adjoint operator:

$$
\langle \phi | H \psi \rangle = \langle H \phi | \psi \rangle ,
$$

where $\phi$ and $\psi$ are arbitrary elements of H (hence neglecting domain issues).

A key observation made in $[20,22,23]$ is that the reality of the spectrum of H is related to the fact that we can turn it into a Hermitian operator by modifying the inner product of H properly. Using the term ‘Hermitizability’ for the latter property, we can say that a diagonalizable operator with a real spectrum need not be Hermitian, but it is necessarily Hermitizable. Conversely, every Hermitizable operator is diagonalizable and has a real spectrum. Therefore, Hermitizability is a necessary and sufficient condition for the reality of the spectrum of H. $^{5}$

The modified inner product that achieves the Hermitization of H is determined by the operator $\eta_{+}$ according to

$$
\langle \psi , \phi \rangle_ {\eta_ {+}} := \langle \psi | \eta_ {+} \phi \rangle ,\tag{12}
$$

where $\psi,\phi\in\mathscr{H}$ are arbitrary. In other words, if $H_{\eta_{+}}$ labels the Hilbert space obtained by endowing the set of vectors belonging to H with the inner product $\langle\cdot,\cdot\rangle_{\eta_{+}}$ , then $H:H_{\eta_{+}}\to H_{\eta_{+}}$ is Hermitian.

The operator $\eta_{+}$ that defines the modified inner product (12) is usually called a ‘metric operator.’ It is not difficult to show that its positive square root, $\rho := \sqrt{\eta_{+}}$ , defines a unitary operator mapping $H_{\eta_{+}}$ onto H and that $h := \rho H \rho^{-1}$ is a Hermitian operator acting in H, [29]. This provides a direct evidence for the reality of the spectrum of H, for H and h are isospectral. It also makes a connection with an earlier work on quasi-Hermitian operators that is done in the context of nuclear physics [30].

Another useful result of Refs. [20, 22, 23] is the following spectral representation of the metric operator.

$$
\eta_ {+} = \sum_ {n} | \phi_ {n} \rangle \langle \phi_ {n} |.\tag{13}
$$

Because the eigenvectors $\phi_n$ of $H^\dagger$ are not unique, this equation signifies the non-uniqueness of the metric operator [29,31]. Different choices for $\{\phi_n\}$ determine different metric operators for $H$ . Once such a choice is made, we can construct the corresponding Hilbert space $\mathcal{H}_{\eta_+}$ and view $H$ as a Hamiltonian operator acting in $\mathcal{H}_{\eta_+}$ . By construction $H: \mathcal{H}_{\eta_+} \to \mathcal{H}_{\eta_+}$ is a Hermitian operator. Therefore $(\mathcal{H}_{\eta_+}, H)$ determines a unitary quantum system. The pure states of this system are represented by the rays in $\mathcal{H}_{\eta_+}$ , the observables are given by Hermitian operators acting in $\mathcal{H}_{\eta_+}$ , and the dynamics is governed by the time-dependent Schrödinger equation defined by $H$ in $\mathcal{H}_{\eta_+}$ .

Applications of the above method of constructing metric operators and the modified inner products for various toy models have been explored in the literature. A comprehensive list of references published prior to 2010 is given in the review article [19]. Here we confine our attention to a very simple example that was originally considered in [32].

Consider the case that $v(x)=0$ , i.e., H is the second derivative operator acting in $L^{2}(\mathbb{R})$ . It is well-known that H is Hermitian and has a nonnegative real continuous spectrum. We can easily check that it satisfies the $\eta_{+}$ -pseudo-Hermiticity relation (9) for

$$
\eta_ {+} := e ^ {- \kappa \mathcal {P}} = \cosh (\kappa) I - \sinh (\kappa) \mathcal {P},\tag{14}
$$

where $\kappa$ is an arbitrary real number, and P is the parity operator (6). Equation (14) defines a genuine metric operator. Substituting it in (12) yields the following expression for the corresponding modified inner product.

$$
\langle \phi , \psi \rangle_ {\eta_ {+}} = \cosh (\kappa) \int_ {- \infty} ^ {\infty} \phi (x) ^ {*} \psi (x) d x - \sinh (\kappa) \int_ {- \infty} ^ {\infty} \phi (x) ^ {*} \psi (- x) d x.\tag{15}
$$

For the standard position operator X, that is given by $X\psi(x) := x\psi(x)$ , we can use (15) to show that

$$
\langle \phi , X \psi \rangle_ {\eta_ {+}} - \langle X \phi , \psi \rangle_ {\eta_ {+}} = 2 \sinh (\kappa) \int_ {- \infty} ^ {\infty} x \phi (x) ^ {*} \psi (- x) d x.
$$

This quantity differs from zero for $\kappa \neq 0$ and $\psi(x) = x\phi(x) = xe^{-x^2}$ . Therefore, as an operator acting in $\mathcal{H}_{\eta_+}$ , $X$ is not Hermitian, unless if $\kappa = 0$ . The same holds for the standard momentum operator, $P := -i\frac{d}{dx}$ .

Clearly the positive square root of the metric operator (14) has the form $\rho = e^{-\kappa P/2}$ . In view of this relation, it is easy to show that the operators $X_{\eta_{+}}$ and $P_{\eta_{+}}$ defined by

$$
X _ {\eta_ {+}} := \rho^ {- 1} X   \rho = \eta_ {+} ^ {- 1} X = [ \cosh (\kappa) I + \sin (\kappa) \mathcal {P} ] X,
$$

$$
P _ {\eta_ {+}} := \rho^ {- 1} P   \rho = \eta_ {+} ^ {- 1} P = [ \cosh (\kappa) I + \sin (\kappa) \mathcal {P} ] P,
$$

are Hermitian operators acting in $H_{\eta+}$ . Because $[X_{\eta+}, P_{\eta+}] = iI$ , we can take $X_{\eta+}$ and $P_{\eta+}$ to represent the position and momentum observables of the system defined by $(\mathcal{H}_{\eta+}, H)$ . For all real numbers $\kappa$ , this is just a free particle moving on a straight line. But for different choices of $\kappa$ , we have different operators representing the position and momentum of the particle. This has some peculiar consequences. For example, for $\kappa \neq 0$ , the spatially localized states of the particle correspond to a linear combination of two Dirac delta-functions rather than a single delta-function [32]!

When $\mathcal{H}$ is an infinite-dimensional Hilbert space the above constructions are valid provided that we impose some additional technical conditions. Specifically, the complete biorthonormal eigensystem $\{(\psi_n,\phi_n)\}$ should be bounded [19]. This is equivalent to the condition that $\{\psi_n\}$ and $\{\phi_n\}$ are Riesz bases of $\mathcal{H}$ , which means that they can be mapped to an orthonormal basis by a bounded invertible operator [19,35]. The boundedness of $\{(\psi_n,\phi_n)\}$ implies that the metric operator $\eta_{+}$ must be a positive automorphism, i.e., a positive invertible operator that is defined everywhere in $\mathcal{H}$ (which makes it bounded) and has a bounded inverse [19].

It turns out that if we define the Schrödinger operator for the potential $ix^{3}$ as a linear operator (with maximal domain) acting in $\mathcal{H} := L^{2}(\mathbb{R})$ , then we cannot satisfy (8) or (9) using a bounded positive-definite operator $\eta_{+}$ that is inversely bounded. Therefore, strictly speaking, an appropriate metric operator does not exist for this potential [33]. As we explain below this is a mathematical technicality that can be circumvented by paying due attention to the role of the linear operators representing physical observables in quantum mechanics.

Consider redefining the Hilbert space H and the operator H in such a way that the new Hilbert space $H'$ includes the eigenvectors of H and the new operator $H'$ , which acts in $H'$ , shares both the spectrum and eigenvectors of H, [34]. Because we can only prepare state vectors which are superpositions of the eigenvectors of the relevant observables, as far as H is concerned both H and $H'$ include all the preparable state vectors, and H and $H'$ are equivalent as representations of a quantum mechanical observable. As shown in Ref. [34], for a given metric operator $\eta_{+}$ , which may violate the conditions of boundedness or inverse boundedness, it is possible to construct $H'$ and $H'$ in such a way that they have the above-mentioned properties and in addition $H'$ be a Hermitian operator. Therefore although one cannot use $(\mathcal{H}, H)$ to define a unitary quantum system directly, one can construct $H'$ and $H'$ which contain the same physically relevant ingredients and use $(\mathcal{H}', H')$ to define a unitary quantum system.

## 3. Singularities of the Metric Operators

Consider the Hilbert space H obtained by endowing $C^{2}$ with the Euclidean inner product. The elements of H and the linear operators acting in it can be respectively represented by $2 \times 1$ and $2 \times 2$ matrices in the standard basis of $C^{2}$ . Using the same symbol for the matrix representations and the corresponding vectors and operators, we consider constructing the most general metric operator for

$$
H := \left[ \begin{array}{c c} 0 & 1 \\ x ^ {2} & 0 \end{array} \right],\tag{16}
$$

where $x \in R$ . It is easy to show that for this operator,

$$
E _ {n} = (- 1) ^ {n} x, \qquad \psi_ {n} = \frac {N _ {n}}{\sqrt {2}} \left[ \begin{array}{c} (- 1) ^ {n} \\ x \end{array} \right], \qquad \phi_ {n} = \frac {1}{\sqrt {2} N _ {n} ^ {*}} \left[ \begin{array}{c} (- 1) ^ {n} \\ x ^ {- 1} \end{array} \right],\tag{17}
$$

where $n = 1,2$ and $N_{n}$ are arbitrary nonzero complex coefficients possibly depending on $x$ .

Inserting the last of Eqs. (17) in (13) and introducing $a_{\pm} := (|N_2|^{-2} \pm |N_1|^{-2})/2$ , we find

$$
\eta_ {+} = \left[ \begin{array}{c c} a _ {+} & a _ {-} x ^ {- 1} \\ a _ {-} x ^ {- 1} & a _ {+} x ^ {- 2} \end{array} \right].
$$

This relation identifies x = 0 with a singularity of all possible metric operators for H. Note also that H loses the property of being diagonalizable precisely for this value of x. This is an example of what is called an exceptional point [36, 37] or a non-Hermitian degeneracy [38].

The term ‘exceptional point’ is introduced by Kato in his study of the effects of perturbations of a linear operator on its spectral properties [39]. The following is a widely used definition of this concept which differs slightly from Kato’s.

Definition 5. Let V be a vector space, m be a positive integer, $H(x):V\to V$ be a linear operator depending on m real parameters $x_{1},x_{2},\cdots,x_{m}$ . We identify these with local coordinates of a point x of a parameter space (a smooth manifold) M. Suppose that for each $x\in M$ the eigenvalues of $H(x)$ have finite geometric multiplicity and form a countable set of isolated points of C that we denote by $E_{n}(x)$ . Here n is a spectral label taking values in a discrete set N. Let $\mu_{n}(x)$ be the geometric multiplicity of $E_{n}(x)$ , i.e., the dimension of the span of eigenvectors of $H(x)$ that are associated with the eigenvalue $E_{n}(x)$ . A point $x_{0}$ of M is called an exceptional point of $H(x)$ if there are $n\in N$ , $\epsilon\in R^{+}$ , and a parameterized curve in M, i.e., a continuous function, $\gamma:(-\epsilon,\epsilon)\to M$ , such that $\gamma(0)=x_{0}$ and for all $t\neq0$ , $\mu_{n}(\gamma(t))\neq\mu_{n}(x_{0})$ .

For the case that V is endowed with the structure of an inner-product space, we can speak of the adjoint of $H(x)$ and decide whether it is Hermitian. If for all $x \in M$ , $H(x)$ is a Hermitian operator, the geometric multiplicity of the eigenvalues $E_{n}(x)$ do not undergo discontinuous changes and an exceptional point cannot exist. Therefore, non-Hermiticity is a necessary condition for the emergence of an exceptional point.

It turns out that exceptional points have a number of interesting physical realizations. See for example $[36–38, 40–42]$ and references therein. In particular, they lead to certain geometric phases which have been the subject of intensive theoretical $[36–38, 40, 41, 43]$ and experimental studies $[44–46]$ since the early 1990's.

The two-dimensional model (16) can be easily generalized to higher dimensional matrix Hamiltonians $H(x)$ , [37]. If we choose the eigenvectors of

$H(x)$ in such a way that they are nonsingular functions of x, then exceptional points appear as the singularities of the eigenvectors of $H(x)^{\dagger}$ and consequently the corresponding metric operator (13).

Definition 5 introduces exceptional points in terms of a condition on the eigenvalues of $H(x)$ . If this operator acts in a Hilbert space, we can speak of its spectrum. This is a subset of $\mathbb{C}$ that in addition to the eigenvalues may contain numbers that are not eigenvalues of $H(x)$ . The latter constitute two disjoint sets called the continuous and the residual spectra of $H(x)$ [47]. Hermitian operators have an empty residual spectrum. The same is true for a large class of non-Hermitian operators.

A natural question that arises in the study of non-Hermitian operators with a real spectrum is how to generalize the notions of diagonalizability and the metric operator for operators whose spectrum includes a continuous part. The first step in this direction was taken in Ref. [48]. It involved a direct extension of the approach developed for operators with a discrete spectrum to the imaginary PT-symmetric barrier potential,

$$
v (x) = \left\{ \begin{array}{c c c} - i \zeta & \text { for } & - 1 \leq x \leq 0, \\ i \zeta & \text { for } & 0 <   x \leq 1, \\ 0 & \text { for } & | x | \leq 1, \end{array} \right. \qquad \zeta \in \mathbb {R}.\tag{18}
$$

To the best of our knowledge, this provided the first example of a PT-symmetric potential which admitted an optical realization [49]. The next step was to carry out the same analysis for the delta-function potential with a complex coupling constant [17],

$$
v (x) = \mathfrak {z}   \delta (x), \quad \mathfrak {z} \in \mathbb {C}.\tag{19}
$$

The treatment of (18) and (19) that was offered in [48] and [17] is perturbative in nature. But there is an important difference; for imaginary values of $\mathfrak{z}$ regardless of how small $|\mathfrak{z}|$ is, the perturbative calculation of the metric operator for (19) is obstructed by the emergence of a singularity. In the remainder of this section, we provide a general description of this phenomenon and its relation to spectral singularities that was originally noticed in [17] and explored more thoroughly for the double-delta-function potential in [50]:

$$
v (x) = \mathfrak {z} _ {-}   \delta (x + a) + \mathfrak {z} _ {+}   \delta (x - a), \qquad \mathfrak {z} _ {\pm} \in \mathbb {C},   a \in \mathbb {R} ^ {+}.\tag{20}
$$

Let $v_{z}: \mathbb{R} \to \mathbb{C}$ be a scattering potential depending on complex parameters $z_{1}, z_{2}, \cdots, z_{m}$ , that we collectively denote by $z$ , i.e., $z := (z_{1}, z_{2}, \cdots, z_{m})$ . Suppose that the Schrödinger operator $H_{z} := -\frac{d^{2}}{dx^{2}} + v_{z}(x)$ acts in $L^{2}(\mathbb{R})$ and has a real and purely continuous spectrum given by $[0, \infty)$ , i.e., its point and residual spectra are empty. Then the nonzero elements of the spectrum of $H_{z}$ correspond to the numbers $k^{2}$ appearing on the right-hand side of the Schröndinger equation (2). These are associated with a linearly independent pair of solutions of this equation that we denote by $\psi_{k,a}^{(z)}$ with $a = 1, 2$ ;

$$
H _ {z} \psi_ {k, a} ^ {(z)} = k ^ {2} \psi_ {k, a} ^ {(z)}.\tag{21}
$$

Because $\psi_{k,a}^{(z)}$ do not belong to $L^2 (\mathbb{R})$ , they are not eigenvectors of $H_{z}$ . We refer to them as 'generalized eigenfunctions' of $H_{z}$ . Similarly, we can construct generalized eigenfunctions of $H_{z}^{\dagger} := -\frac{d^{2}}{dx^{2}} + v_{z}(x)^{*}$ that we denote by $\phi_{k,a}^{(z)}$ . These satisfy

$$
H _ {z} ^ {\dagger} \phi_ {k, a} ^ {(z)} = k ^ {2} \phi_ {k, a} ^ {(z)}.\tag{22}
$$

We can generalize the notion of ‘diagonalizability’ for $H_{z}$ , by demanding the existence of an eigensystem $\{(\psi_{k,a}^{(z)},\phi_{k,a}^{(z)})\}$ which satisfy the following biorthonormality and completeness relations [48].

$$
\langle \phi_ {k, a} ^ {(z)} | \psi_ {q, b} ^ {(z)} \rangle = \delta_ {a b} \delta (k - q), \quad \sum_ {a = 1} ^ {2} \int_ {0} ^ {\infty} d k | \psi_ {k, a} ^ {(z)} \rangle \langle \phi_ {k, a} ^ {(z)} | = I.\tag{23}
$$

Similarly we generalize the expression (13) for the metric operator:

$$
\eta_ {+} = \sum_ {a = 1} ^ {2} \int_ {0} ^ {\infty} d k | \phi_ {k, a} ^ {(z)} \rangle \langle \phi_ {k, a} ^ {(z)} |.\tag{24}
$$

Now, we demand that $v_{z}(x)^{*} = v_{z^{*}}(x)$ . Then it is easy to see that

$$
H _ {z} ^ {\dagger} \psi_ {k, a} ^ {(z ^ {*})} = \left[ - \frac {d ^ {2}}{d x ^ {2}} + v _ {z} (x) ^ {*} \right] \psi_ {k, a} ^ {(z ^ {*})} = H _ {z ^ {*}} \psi_ {k, a} ^ {(z ^ {*})} = k ^ {2} \psi_ {k, a} ^ {(z ^ {*})}.\tag{25}
$$

Because for each $k \in R^{+}$ , the Schrödinger equation $H_{z}^{\dagger}\psi = k^{2}\psi$ has two linearly independent solutions, Eqs. (22) and (25) imply that $\phi_{k,a}^{(z)}$ are linear combinations of $\psi_{k,a}^{(z^{*})}$ , i.e., there are $J_{ab}(k) \in \mathbb{C}$ such that

$$
\phi_ {k, a} ^ {(z)} = \sum_ {b = 1} ^ {2} J _ {a b} ^ {(z)} (k) \psi_ {k, b} ^ {(z ^ {*})}.\tag{26}
$$

It is also not difficult to show that $\langle \psi_{k,a}^{(z^*)}|\psi_{q,b}^{(z)}\rangle$ is proportional to $\delta (k - q)$ , i.e., there are $K_{ab}^{(z)}(k)\in \mathbb{C}$ such that

$$
\langle \psi_ {k, a} ^ {(z ^ {*})} | \psi_ {q, b} ^ {(z)} \rangle = K _ {a b} ^ {(z)} (k) \delta (k - q).\tag{27}
$$

Inserting (26) in the first equation in (23) and making use of (27), we find [50]

$$
\mathbf {J} (k) ^ {(z) *} \mathbf {K} ^ {(z)} (k) = \mathbf {I},\tag{28}
$$

where $\mathbf{J}^{(z)}(k)$ and $\mathbf{K}^{(z)}(k)$ are $2 \times 2$ matrices having $J_{ab}^{(z)}(k)$ and $K_{ab}^{(z)}(k)$ as their entries, and I is the $2 \times 2$ identity matrix. Similarly, using (24), (26), and (28), we obtain

$$
\eta_ {+} = \sum_ {a, b = 1} ^ {2} \int_ {0} ^ {k} d k \mathcal {E} _ {a b} ^ {(z)} (k) | \psi_ {k, a} ^ {(z ^ {*})} \rangle \langle \psi_ {k, b} ^ {(z ^ {*})} |,\tag{29}
$$

where $\mathcal{E}_{ab}^{(z)}(k)$ are entries of the matrix $[\mathbf{K}^{(z)}(k)\mathbf{K}^{(z)}(k)^{\dagger}]^{-1}$ .

Equation (28) implies that $\det(\mathbf{K}^{(z)}(k)) \neq 0$ . But in general there is no reason why this relation should hold for all k and z. Explicit calculations for the potentials (19) and (20) show that the values of $k^{2}$ for which $\operatorname{det}(\mathbf{K}^{(z)}(k)) = 0$ are precisely the spectral singularities of the Schrödinger operator $H_{z}$ , [50]. We are not aware of a proof of this statement for a general complex scattering potential. The proof for the double-delta-function potential that was given in [50] revealed a useful connection between spectral singularities and the transfer matrix $\mathbf{M}(k)$ of scattering theory [51]. It turned out that $\operatorname{det}(\mathbf{K}^{(z)}(k))$ was proportional to the $M_{22}(k)$ entry of $\mathbf{M}(k)$ with a nonzero proportionality factor. This provided the key observation that immediately led to the explanation of the physical meaning of a spectral singularity [52]. We give a detailed discussion of these developments in the next section.

We close this section by noting that according to (29), spectral singularities are also singularities of the metric operator. In this sense they are generalizations of the phenomenon of exceptional points to the linear operators that possess a nonempty continuous spectrum.

## 4. Scattering Theory and Spectral Singularities

Consider a possibly complex scattering potential $v(x)$ satisfying (3). The left- and right-incident scattering solutions of the Schrödinger equation, that we respectively denote by $\psi_{k}^{l}(x)$ and $\psi_{k}^{r}(x)$ , satisfy the following asymptotic boundary conditions.

$$
\psi_ {k} ^ {l} (x) \to \left\{ \begin{array}{c c c} \mathcal {A} ^ {l} (k) \left[ e ^ {i k x} + R ^ {l} (k) e ^ {- i k x} \right] & \text {as} & x \to - \infty , \\ \mathcal {A} ^ {l} (k) T ^ {l} (k) e ^ {i k x} & \text {as} & x \to \infty , \end{array} \right.\tag{30}
$$

$$
\psi_ {k} ^ {r} (x) \to \left\{ \begin{array}{c c c} \mathcal {A} ^ {r} (k) T ^ {r} (k) e ^ {- i k x} & \text { as } & x \to - \infty , \\ \mathcal {A} ^ {r} (k) \left[ e ^ {- i k x} + R ^ {r} (k) e ^ {i k x} \right] & \text { as } & x \to \infty , \end{array} \right.\tag{31}
$$

where $A^{l/r}$ , $R^{l/r}$ , and $T^{l/r}$ are in general complex-valued functions. Because the Schrödinger equation (2) is linear, the choice of $A^{l/r}$ does not affect the physically measurable quantities. This is not the case for $R^{l/r}$ and $T^{l/r}$ , which are known as the left/right reflection and transmission amplitudes. Their modulus squared, $|R^{l/r}|^{2}$ and $|T^{l/r}|^{2}$ , determine the left/right reflection and transmission coefficients that can be measured in experiments. $^{6}$

A well known consequence of the linearity of the Schrödinger equation (2) is that $T^{l}=T^{r}$ , [52–54]. $^{7}$ We therefore use T for $T^{l/r}$ . It is also easy to see that $\psi_k^{l / r}$ coincide with the Jost solutions $\psi_{k\pm}$ for $\mathcal{A}^{l / r}(k)T(k) = 1$ ;

$$
\psi_ {k +} (x) \to \left\{ \begin{array}{c c c} T (k) ^ {- 1} \left[ e ^ {i k x} + R ^ {l} (k) e ^ {- i k x} \right] & \text {as} & x \to - \infty , \\ e ^ {i k x} & \text {as} & x \to \infty , \end{array} \right.\tag{32}
$$

$$
\psi_ {k -} (x) \to \left\{ \begin{array}{c c c} e ^ {- i k x} & \text {as} & x \to - \infty , \\ T (k) ^ {- 1} \left[ e ^ {- i k x} + R ^ {r} (k) e ^ {i k x} \right] & \text {as} & x \to \infty . \end{array} \right.\tag{33}
$$

The existence of the Jost solutions implies that $T(k) \neq 0$ , i.e., perfectly absorbing potentials [56] do not exist.

The coefficients of the $e^{\pm ikx}$ that appear on the right-hand side of (32) and (33) turn out to coincide with the entries of a $2 \times 2$ complex matrix known as the transfer matrix.

Because $v(x) \to 0$ as $x \pm \infty$ , every solution $\psi(x)$ of the Schrödinger equation (2) satisfies

$$
\psi (x) \to A _ {\pm} (k) e ^ {i k x} + B _ {\pm} (k) e ^ {- i k x} \quad \text { as } \quad x \to \pm \infty ,\tag{34}
$$

where $A_{\pm}(k)$ and $B_{\pm}(k)$ are complex coefficients. The transfer matrix $\mathbf{M}(k)$ is defined by the relation

$$
\left[ \begin{array}{c} A _ {+} (k) \\ B _ {+} (k) \end{array} \right] = \mathbf {M} (k) \left[ \begin{array}{c} A _ {-} (k) \\ B _ {-} (k) \end{array} \right].
$$

In light of (32), (33), and (34), we can relate the entries $M_{ij}(k)$ of the transfer matrix $\mathbf{M}(k)$ with the reflection and transmission amplitudes. This results in [52]

$$
M _ {1 1} = T - \frac {R ^ {l} R ^ {r}}{T}, \quad M _ {1 2} = \frac {R ^ {r}}{T}, \quad M _ {2 1} = - \frac {R ^ {l}}{T}, \quad M _ {2 2} = \frac {1}{T},\tag{35}
$$

which, in particular, imply $\operatorname{det} \mathbf{M}(k) = 1$ . Furthermore, we can use these relations to express (32) and (33) in the form

$$
\psi_ {k +} (x) \to \left\{ \begin{array}{c l} M _ {2 2} (k) e ^ {i k x} - M _ {2 1} (k) e ^ {- i k x} & \text {as} \quad x \to - \infty , \\ e ^ {i k x} & \text {as} \quad x \to \infty , \end{array} \right.\tag{36}
$$

$$
\psi_ {k -} (x) \to \left\{ \begin{array}{c c c} e ^ {- i k x} & \text {as} & x \to - \infty , \\ M _ {2 2} (k) e ^ {- i k x} + M _ {1 2} (k) e ^ {i k x} & \text {as} & x \to \infty . \end{array} \right.\tag{37}
$$

The following characterization of spectral singularities is a direct consequence of these equations.

Theorem 3. Let $v: \mathbb{R} \to \mathbb{C}$ and $H$ be as in Definition 1, $\mathbf{M}(k)$ be the transfer matrix of $v$ , $M_{ij}(k)$ be the entries of $\mathbf{M}(k)$ , and $k_{\star}$ be a positive real number. Then $k_{\star}^{2}$ is a spectral singularity of $H$ (or $v$ ) if and only if $M_{22}(k_{\star}) = 0$ .

Proof. $k_{\star}^{2}$ is a spectral singularity of H whenever $\psi_{k_{\star}-}$ and $\psi_{k_{\star+}}$ are linearly dependent. According to (36) and (37) and the fact that these equations determine $\psi_{k\pm}$ uniquely, this happens if and only if $M_{22}(k_{\star}) = 0$ . ☐

Combining the statement of Theorem 3 with Eqs. (35) yields the physical meaning of spectral singularities, namely that spectral singularities are the real and positive values of the energy $k_{\star}^{2}$ at which reflection and transmission amplitudes diverge [52]. The latter is a characteristic property of resonances, for they satisfy the outgoing boundary conditions [57]. As seen from (36) and (37), for the cases that $k_{\star}$ corresponds to a spectral singularity and $\psi_{k_{\star}\pm}$ become linearly dependent, they also satisfy the outgoing boundary conditions.

The main distinction between the wave function for a resonance and the Jost solutions $\psi_{k_{\star}\pm}$ at a spectral singularity $k_{\star}^{2}$ is that, unlike the latter, the former satisfies the Schrödinger equation for a non-real value of $k^{2}$ . Because the imaginary part of $k^{2}$ for a resonance determines its width, we can identify spectral singularities with the energies of certain zero-width resonances. Note, however, that spectral singularities determine genuine non-decaying scattering states with real and positive energy [52]. This distinguishes them from the bound states in the continuum [58]. Although the latter are also associated with zero-width resonances, their wave function is a square-integrable solutions of the Schrödinger equation. For a discussion of other differences between spectral singularities and bound states in the continuum, see [59].

The fact that the reflection and transmission amplitudes and consequently the reflection and transmission coefficients $|R^{l/r}(k)|^{2}$ and $|T(k)|^{2}$ diverge for a resonance does not conflict with the well-known unitarity condition

$$
| R ^ {l / r} (k) | ^ {2} + | T (k) | ^ {2} = 1,\tag{38}
$$

because the k-value for a resonance is not real. For a spectral singularity, k is real and (38) is violated. This provides a simple proof of the following result.

## Theorem 4. Real potentials cannot support a spectral singularity.

In the standard formulation of quantum mechanics, the Hamiltonian operator H is required to be Hermitian and the potential functions v are necessarily real-valued. Therefore, they do not display spectral singularities. The same applies to the pseudo-Hermitian representation of quantum mechanics $[19]$ where H may not be Hermitian but Hermitizable. This is because the presence of a spectral singularities obstructs the existence of a metric operator that achieves the Hermitization process. However, complex scattering potentials have a number of applications in other areas of physics. The primary example is the optical potentials used in modeling optically active material. This is the arena in which the role and implications of spectral singularities have so far been studied. We devote the next section to a brief description of the optical realizations of spectral singularities.

## 5. Spectral Singularities in Optics

Consider an isotropic charge-free linear medium whose electromagnetic properties changes along one direction, that we take to be the x-axis in a Cartesian coordinate system. We can encode these properties in the definition of the refractive index of the medium $\mathbf{n}(x)$ which is a generally complex quantity. Suppose that we are interested in the propagation of a linearly polarized time-harmonic electromagnetic wave in this medium. If we choose our y-axis along the polarization direction, we can express the electric field in the form $\vec{E}(\vec{r},t)=e^{-i\omega t}\mathcal{E}(\vec{r})\hat{e}_{y}$ , where $\vec{r}:=(x,y,z)$ , $\omega$ is the angular frequency of the wave, $\mathcal{E}(\vec{r})$ is a solution of

$$
\left[ \nabla^ {2} + k ^ {2} \mathbf {n} (x) ^ {2} \right] \mathcal {E} (\vec {r}) = 0,\tag{39}
$$

$\hat{e}_{y}$ is the unit vector along the positive y-axis, $k := \omega/c$ is the wavenumber, and c is the speed of light in vacuum [60].

Equation (39) admits solutions depending only on x; $\mathcal{E}(\vec{r}) = \psi(x)$ . In view of (39), $\psi(x)$ satisfies the Schrödinger equation (2) corresponding to the potential

$$
v (x) := k ^ {2} [ 1 - \mathbf {n} (x) ^ {2} ].\tag{40}
$$

If the medium is confined to a compact region in empty space, $\mathbf{n}(x)=1$ for sufficiently large values of $|x|$ . This together with the fact that n is a complex-valued function imply that $v(x)$ is a (finite-range) complex scattering potential. Therefore, optical potentials (40) provide a fertile ground for the investigation of the physical implications of spectral singularities. Ref. [52], which offers the first such investigation, explores spectral singularities in a medium described by an optical potential of the form (18).

The physical meaning of these spectral singularities is more easily understood for a simpler model that consists of a homogeneous optically active infinite planar slab of length L placed in vacuum $[61,62]$ . This corresponds to a complex barrier potential,

$$
v (x) = \left\{ \begin{array}{l l} \mathfrak {z} & \text { for } \quad | x | \leq L / 2, \\ 0 & \text { for } \quad | x | > L / 2, \end{array} \right. \qquad \mathfrak {z} := k ^ {2} (1 - \mathfrak {n} ^ {2}),\tag{41}
$$

where n stands for the refractive index of the slab.

Inside the slab, where $|x| \leq L/2$ , the Schrödinger equation (2) admits a solution of the form $\psi(x) = \mathcal{E}_{0} e^{ik \mathfrak{n}(x + L/2)}$ , where $E_{0}$ is a constant. This corresponds to a right-going plane wave

$$
\vec {E} (\vec {r}, t) = \mathcal {E} _ {0} e ^ {i [ k \mathfrak {n} (x + L / 2) - \omega t ]} \hat {e} _ {y}, \quad | x | \leq L / 2.
$$

If we use $\eta$ and $\kappa$ to respectively denote the real and imaginary parts of n, so that $n = \eta + i\kappa$ , we find

$$
| \vec {E} (\vec {r}, t) | ^ {2} = | \mathcal {E} _ {0} | ^ {2} e ^ {- k \kappa (2 x + L)}, \quad | x | \leq L / 2.
$$

In particular, as the wave travels through the slab, its intensity changes from $|\mathcal{E}_{0}|^{2}$ to $|\mathcal{E}_{0}|^{2}e^{-2k\kappa L}$ , i.e., it undergoes an exponential loss or gain of intensity by a factor of $e^{-2k\kappa L}$ depending on whether $\kappa > 0$ or $\kappa < 0$ . Because of this a medium that has a positive (respectively negative) value for $\kappa$ is called a lossy (respectively gain) medium. The factor $2k|\kappa|$ that determines the amount of the exponential loss (gain) per unit distance traversed by the wave is called the attenuation (respectively gain) coefficient. In terms of the wavelength, $\lambda := 2\pi/k$ , this quantity takes the form $4\pi|\kappa|/\lambda$ . In particular, the gain coefficient is given by [63]

$$
g := - \frac {4 \pi \kappa}{\lambda}.\tag{42}
$$

Because the complex barrier potential (41) is exactly solvable, we can easily determine its transfer matrix and explore its spectral singularities. This is done in Refs. [61,62]. Here we suffice to state that the relation $M_{22}(k_{\star}) = 0$ , which determines the spectral singularities $k_{\star}^{2}$ whenever $k_{\star} \in \mathbb{R}^{+}$ , reduces to the following complex transcendental equation [62].

$$
e ^ {- 2 i \mathfrak {n} k _ {\star} L} = \left(\frac {\mathfrak {n} - 1}{\mathfrak {n} + 1}\right) ^ {2}.\tag{43}
$$

The right-hand side of this relation is a well-known quantity in optics called the reflectivity R. If we compute the modulus (absolute-value) of both sides of (43) and use (42) in the resulting expression, we obtain [62]

$$
g = \frac {1}{2 L} \ln \frac {1}{| \mathcal {R} | ^ {2}}.\tag{44}
$$

This equation that is a consequence of the existence of a spectral singularity is one of the basic relations of laser physics known as the laser threshold condition $[63]$ . The right-hand side of $(44)$ is the minimum amount of gain necessary for a (mirrorless) slab laser to begin emitting laser light. It is called the threshold gain coefficient.

Every laser amplifies the background noise to sizable intensities and emits it as coherent electromagnetic radiation. This is precisely what a spectral singularity does, because it leads to infinite reflection and transmission coefficients that are capable of amplifying extremely weak background electromagnetic waves to considerable intensities. The fact that the waves emitted from both sides of a slab laser have the same intensity and phase (are coherent) also follows from (43). This is indeed a general property of spectral singularities, because they are invariant under the space reflection (parity) P. Under P the transfer matrix $\mathbf{M}(k)$ of every scattering potential transforms as

$$
\mathbf {M} (k) \stackrel {\mathcal {P}} {\longleftrightarrow} \sigma_ {1} \mathbf {M} (k) ^ {- 1} \sigma_ {1},\tag{45}
$$

where $\sigma_{1}$ is the first Pauli matrix, i.e., the $2\times2$ matrix with zero diagonal and unit off-diagonal entries, [64,65]. According to (45), $M_{22}(k)$ is P-invariant. Therefore, the same holds for the spectral singularities that are given by the real and positive zeros of $M_{22}(k)$ .

In Ref. [66], we develop a nonlinear generalization of spectral singularities that apply to nonlinearities that are confined in space (have compact support.) It turns out that the mathematical relation describing these nonlinear spectral singularities for the above simple slab model supplemented with a weak Kerr nonlinearity yields an equation relating the output intensity I of the slab laser to its gain coefficient [67]. For a typical optical gain medium [63], which satisfies $|\kappa| \ll 1 < \eta$ , this equation takes the following form.

$$
I = \frac {f (\eta) (g - g _ {t h})}{\sigma g _ {t h}},\tag{46}
$$

where f is a real-valued function taking strictly positive values, g is the gain coefficient (42), $g_{th}$ is the threshold gain coefficient that is given by the right-hand side of (44), and $\sigma$ is the Kerr coefficient which, for generic gain media, takes small but positive values.

Because $f(\eta) > 0$ , $\sigma > 0$ , and $I \geq 0$ , Eq. (46) implies that there is no power emitted from a slab laser unless we have $g > g_{th}$ , and for $g > g_{th}$ the intensity of emitted wave increases linearly as a function of $g - g_{th}$ . Both of these statements are among the basic results of the physics of lasers. Here they follow as logical consequences of the purely mathematical condition of the existence of a nonlinear spectral singularity. Let us also mention that (46) has a more general domain of validity. In Ref. [68], we explore the consequences of the emergence of nonlinear spectral singularities for a weakly nonlinear $\mathcal{PT}$ -symmetric bilayer slab. This consists of a pair of adjacent infinite homogeneous planar slabs with complex-conjugate refractive index, $\eta \pm i\kappa$ , so that one's gain (loss) is balanced by the other's loss (gain) [65,69]. The laser output intensity computed using the condition of the appearance of a nonlinear spectral singularity is also given by (46), albeit with a different choice for the function $f$ , [68].

Another interesting development having its root in optical spectral singularities is the discovery of perfect coherent absorbers (CPA) which are also called antilasers $[64,70–73]$ . These are optical devices that function as time-reversed lasers, i.e., they completely absorb coherent electromagnetic waves.

Under the time-reversal transformation (6), scattering potentials $v(x)$ and their transfer matrix $\mathbf{M}(k)$ transform according to

$$
v (x) \stackrel {\mathcal {T}} {\longleftrightarrow} v (x) ^ {*}, \qquad \qquad \mathbf {M} (k) \stackrel {\mathcal {T}} {\longleftrightarrow} \sigma_ {1} \mathbf {M} (k) ^ {*} \sigma_ {1}.\tag{47}
$$

In light of these relations, the time-reversal transformation T converts an optical potential (40) describing a gain media into that of a lossy medium, and induces the transformation:

$$
M _ {1 1} (k) \stackrel {\mathcal {T}} {\longleftrightarrow} M _ {2 2} (k) ^ {*}.
$$

This, in particular, means that the spectral singularities of $v(x)$ correspond to the real values of the wavenumber k at which the $M_{11}(k)$ entry of the transfer matrix of the time-reversed potential, $v(x)^{*}$ , vanishes. At this wavenumber the optical system modeled by $v(x)^{*}$ serves as a CPA. In other words, CPA action is a realization of the spectral singularities of the time-reversed (complex-conjugate) optical potential [64,65].

## 6. Concluding Remarks

Spectral singularities were introduced by Naimark about sixty years ago and have since become a subject of research in operator theory. Given their interesting mathematical implications, it is quite surprising that their relevance to scattering theory and their physical meaning could not be understood earlier than in 2009. It turns out that the optics of gain media offers various physical models in which this concept can be realized. The study of the optical realization of spectral singularities shows that they form a mathematical basis for lasers. This observation could be made much earlier, had the optical physicists knew about spectral singularities. Indeed, the solution of the wave equations with outgoing boundary conditions, which leads to spectral singularities for real wavenumbers, has been employed in laser theory previously $[74]$ .

The discovery of the physical aspects of spectral singularities has boosted interest in their study particularly among physicists. During the past five years there have appeared a number of research publications on the subject. The following is a list of those that we did not elude to above.

\- Refs. [75-78] address some of the formal and conceptual aspects of the subject.

\- Refs. [50,65,79–81] explore specific toy models supporting spectral singularities.

\- Refs. [82, 83] study the application of semiclassical approximation and perturbation theory for determining spectral singularities of non-homo geneous gain media with planar symmetry.

\- Refs. [84, 85] examine the optical spectral singularities in spherical and cylindrical geometries. In particular, [85] offers a detailed and careful treatment of spectral singularities in the whispering gallery modes. These correspond to the cylindrical and spherical lasers.

\- Refs. [86,87] discuss some of the applications of spectral singularities in condensed matter physics.

\- Refs. [88, 89] consider spectral singularities in certain optically active waveguides and elaborate on their regularization due to the presence of nonlinearities.

\- Ref. [90] offers an extension of the analysis of [62] to waves with a non-normal incidence angle.

\- Refs. [91-93] are some other publications that discuss spectral singularities.

The recent development of a nonlinear generalization of spectral singularities $[66]$ has opened the way towards applications of this concept in the vast territory of nonlinear waves. The fact that the simple applications in effectively one-dimensional optical systems yield a mathematical derivation of the known behavior of the laser output intensity provides ample motivation for further study of nonlinear spectral singularities in other areas of physics.

## Acknowledgment

I would like to express my gratitude to the organizers of the XXXIII Workshop on Geometric Methods in Physics, in particular Piotr Kielanowski, for their hospitality during this meeting. I am also indebted to Hamed Ghaemidizicheh for helping me locate and correct the typos in the first draft of the manuscript. This work has been supported by the Scientific and Technological Research Council of Turkey (TÜBİTAK) in the framework of the project no: 112T951, and by the Turkish Academy of Sciences (TÜBA).

## References

[1] J. Schwartz, Comm. Pure Appl. Math. 13, 609 (1960).

[2] M. A. Naimark, Trudy Moscow. Mat. Obsc. 3, 181 (1954) in Russian, English translation: Amer. Math. Soc. Transl. (2), 16, 103 (1960).

[3] R. R. D. Kemp, Canadian J. Math. 10, 447 (1958).

[4] V. E. Lyantse, Mat. Sb. 64 521 (1964) and 65, 47 (1964).

[5] M. G. Gasymov, Soviet Math. Dokl. 9, 390 (1968).

[6] M. G. Gasymov and F. G. Maksudov, Func. Anal. Appl. 6, 185 (1972).

[7] M. G. Gasymov, Func. Anal. Appl. 14, 11 (1980).

[8] H. Langer, Spectral functions of definitizable operators in Krein space, Lecture Notes in Mathematics 948, 1 (1982).

[9] B. Nagy, J. Operator Theory 15, 307 (1986).

[10] E. Bairamov, Ö. Çakar, and A. M. Krall, J. Diff. Eq. 151, 268 (1999).

[11] G. Sh. Guseinov, Pramana J. Phys. 73, 587 (2009).

[12] C. M. Bender and S. Boettcher, Phys. Rev. Lett. 80, 5243 (1998).

[13] P. Dorey, C. Dunning, and R. Tateo, J. Phys. A 34, 5679 (2001).

[14] A. Mostafazadeh, J. Phys. A 41, 055304 (2008).

[15] G. Levai and M. Znojil, J. Phys. A: Math. Gen. 33 7165 (2000).

[16] B. F. Samsonov, J. Phys. A 38, L571 (2005).

[17] A. Mostafazadeh, J. Phys. A 39, 13495 (2006).

[18] C. M. Bender and P. D. Mannheim, Phys. Lett. A 374 1616 (2010).

[19] A. Mostafazadeh, Int. J. Geom. Methods Mod. Phys. 7, 1191 (2010).

[20] A. Mostafazadeh, J. Math. Phys. 43, 3944 (2002).

[21] C. M. Bender, D. C. Brody and H. F. Jones, Am. J. Phys. 71, 1095 (2003).

[22] A. Mostafazadeh, J. Math. Phys. 43, 205 (2002).

[23] A. Mostafazadeh, J. Math. Phys. 43, 2814 (2002).

[24] V. I. Istrătescu, Introduction to Linear Operator Theory, Marcel Dekker, New York, 1981.

[25] J. Dieudonné, in Proceedings of the International Symposium on Linear Spaces, Jerusalem, 1960, Pergamon, Oxford, 1961, pp 115-122.

[26] S. Albeverio and S. Kuzhel, Lett. Math. Phys. 67, 223 (2004).

[27] J. P. Antoine and C. Trapani, J. Phys. A 46, 025204 (2013).

[28] J. P. Antoine and C. Trapani, J. Math. Phys. 55, 013503 (2014).

[29] A. Mostafazadeh, J. Phys. A 36, 7081 (2003).

[30] F. G. Scholtz, H. B. Geyer, and F. J. W. Hahne, Ann. Phys. (NY) 213 74 (1992).

[31] A. Mostafazadeh, J. Math. Phys. 44, 974 (2003).

[32] A. Mostafazadeh, J. Phys. A 39, 10171 (2006).

[33] P. Siegl and D. Krejčiřek, Phys. Rev. D 86, 121702(R) (2012).

[34] A. Mostafazadeh, Phil. Trans. R. Soc. A 371, 20120050 (2013).

[35] F. Bagarello A. Inoue and C. Trapani, J. Math. Phys. 55, 033501 (2014).

[36] W. D. Heiss, Phys. Rep. 242, 443 (1994) and J. Phys. A 37, 2455 (2004).

[37] H. Mehri-Dehnavy and A. Mostafazadeh, J. Math. Phys. 49, 082105 (2008).

[38] M. V. Berry, Czech. J. Phys. 54, 1039 (2004).

[39] T. Kato, Perturbation Theory for Linear Operators, Springer, Berlin, 1995.

[40] W. D. Heiss and A. L. Sannino, J. Phys. A 23, 1167 (1990).

[41] M. Müller and I. Rotter, J. Phys. 41, 244018 (2008).

[42] N. Moiseyev, Non-Hermitian Quantum Mechanics, Cambridge University Press, Cambridge, 2011.

[43] A. A. Mailybaev, O. N. Kirillov, and A. P. Seyranian, Phys. Rev. A 72, 014104 (2005).

[44] C. Dembowski, et al, Phys. Rev. Lett. 86, 787 (2001).

[45] T. Stehmann, W. D. Heiss, and F. G. Scholtz, J. Phys. A 37, 7813 (2004).

[46] B. Dietz et al, Phys. Rev. Lett. 106, 150403 (2011).

[47] M. Reed and B. Simon, Functional Analysis, vol. I, Academic Press, San Diego, 1980.

[48] A. Mostafazadeh, J. Math. Phys. 46, 102108 (2005).

[49] A. Ruschhaupt, F. Delgado, and J. G. Muga, J. Phys. A 38, L171 (2005).

[50] A. Mostafazadeh and H. Mehri-Dehnavy, J. Phys. A 42, 125303 (2009).

[51] L. L. Sánchez-Soto, J. J. Monzóna, A. G. Barriuso, and J. F. Cariñena, Phys. Rep. 513 191 (2012).

[52] A. Mostafazadeh, Phys. Rev. Lett. 102, 220402 (2009).

[53] Z. Ahmed, Phys. Rev. A 64, 042716 (2001).

[54] L. L. Sanchez-Soto and J. J. Monzon, Symmetry 6, 396 (2014).

[55] K. Chadan and P. C. Sabatier, Inverse Problems in Quantum Scattering Theory, Springer, New York, 1989.

[56] J. G. Muga, J. P. Palao, B. Navarro, and I. L. Egusquiza, Phys. Rep. 395, 357 (2004).

[57] A. J. F. Siegert, Phys. Rev. 56, 750 (1939).

[58] J. von Neumann and E. Wigner, Phys. Z. 30, 465 (1929).

[59] A. Mostafazadeh, Acta Polytechnica 53, 306 (2013).

[60] M. Born and E. Wolf, Principles of Optics, Cambridge University Press, Cambridge, 1999.

[61] A. Mostafazadeh, Phys. Rev. A 80, 032711 (2009)

[62] A. Mostafazadeh, Phys. Rev. A 83, 045801 (2011).

[63] W. T. Silfvast, Laser Fundamentals, Cambridge University Press, Cambridge, 1996.

[64] S. Longhi, Phys. Rev. A 82, 031801 (2010).

[65] A. Mostafazadeh, J. Phys. A 45, 444024 (2012).

[66] A. Mostafazadeh, Phys. Rev. Lett. 110, 260402 (2013).

[67] A. Mostafazadeh, Phys. Rev. A 87, 063838 (2013).

[68] A. Mostafazadeh, Stud. Appl. Math. 133, 353 (2014).

[69] A. Mostafazadeh, Phys. Rev. A 87, 012103 (2013).

[70] Y. D. Chong, L. Ge, H. Cao, and A. D. Stone, Phys. Rev. Lett. 105, 053901 (2010)

[71] W. Wan, Y. Chong, L. Ge, H. Noh, A. D. Stone, and H. Cao, Science 331, 889 (2011).

[72] S. Longhi, Phys. Rev. A 83, 055804 (2011) and Phys. Rev. Lett. 107, 033901 (2011).

[73] L. Ge, Y. D. Chong, S. Rotter, H. E. Türeci, and A. D. Stone, Phys. Rev. A 84, 023820 (2011).

[74] H. E. Türeci, A. D. Stone, and B. Collier, Phys. Rev. A 74, 043822 (2006).

[75] A. A. Andrianov, F. Cannata, and A. V. Sokolov, J. Math. Phys. 51, 052104 (2010).

[76] B. F. Samsonov, J. Phys. A 44, 392001 (2011) and Phil. Trans. R. Soc. A 371, 20120044, (2013).

[77] F. Correa and M. S. Plyushchay, Phys. Rev. D 86, 085028 (2012).

[78] L. Chaos-Cador and G. Garcia-Calderon, Phys. Rev. A 87, 042114 (2013).

[79] Z. Ahmed, J. Phys. A 42, 472005 (2009); 45, 032004 (2012); and 47, 385303 (2014).

[80] A. Mostafazadeh, J. Phys. A 44, 375302 (2011).

[81] A. Sinha and R. Roychoudhury, J. Math. Phys. 54, 112106 (2013).

[82] A. Mostafazadeh, Phys. Rev. A 83, 045801 (2011).

[83] A. Mostafazadeh and S. Rostamzadeh, Phys. Rev. A 86, 022103 (2012).

[84] A. Mostafazadeh and M. Sarisaman, Phys. Lett. A 375, 3387 (2011) and Proc. R. Soc. A 468, 3224 (2012).

[85] A. Mostafazadeh and M. Sarisaman, Phys. Rev. A 87, 063834 (2013) and 88, 033810 (2013).

[86] S. Longhi, Phys. Rev. B 80, 165125 (2009) and Phys. Rev. A 81, 022102 (2010).

[87] G. R. Li, X. Z. Zhang, and Z. Song, Ann. Phys. (N.Y.) 349, 288 (2014).

[88] X. Liu, S. D. Gupta, and G. S. Agarwal1, Phys. Rev. A 89, 013824 (2014).

[89] K. N. Reddy and S. D. Gupta, Optics Lett. 39, 4595 (2014).

[90] R. Aalipour, Phys. Rev. A 90, 013820 (2014).

[91] A. Mostafazadeh, Ann. Phys. (N.Y.) 341, 77 (2014).

[92] A. Mostafazadeh, Phys. Rev. A 90, 023833 (2014); Addendum, Phys. Rev. A 90, 055803 (2014).

[93] G. Garcia-Calderon and L. Chaos-Cador, Phys. Rev. A 90, 032109 (2014).

Ali Mostafazadeh

Departments of Mathematics and Physics

Koç University

Sarıyer 34450, Istanbul

Turkey

e-mail: amostafazadeh@ku.edu.tr
