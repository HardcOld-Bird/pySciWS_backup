# Imaginary Gauge Transformation in Momentum Space and Dirac Exceptional Point

Jose H. D. Rivero, $^{1,2}$ Liang Feng, $^{3}$ and Li Ge $^{1,2,*}$

$^{1}$ Department of Physics and Astronomy, College of Staten Island, CUNY, Staten Island, New York 10314, USA $^{2}$ The Graduate Center, CUNY, New York, New York 10016, USA

$^{3}$ Department of Materials Science and Engineering, University of Pennsylvania, Philadelphia, Pennsylvania 19104, USA

(Received 23 September 2022; accepted 22 November 2022; published 7 December 2022)

An imaginary gauge transformation is at the core of the non-Hermitian skin effect. Here, we show that such a transformation can be performed in momentum space as well, which reveals that certain gain- and loss-modulated systems in their parity-time $(\mathcal{PT})$ symmetric phases are equivalent to Hermitian systems with real potentials. Our analysis in momentum space also distinguishes two types of exceptional points (EPs) in the same system. Besides the conventional type that leads to a PT transition upon the continuous increase of gain and loss, we find real-valued energy bands connected at a Dirac EP in hybrid dimensions, consisting of a spatial dimension and a synthetic dimension for the gain and loss strength.

DOI: 10.1103/PhysRevLett.129.243901

From nuclear decay $[1]$ to photon lifetime in optical microcavities $[2]$ , a non-Hermitian description of the openness of a physical system has fascinated the physics community for nearly a century. Different forms of non-Hermiticity have been introduced to modify an otherwise Hermitian Hamiltonian, including, for example, a complex potential and asymmetric hoppings, and the existence of unique non-Hermitian degeneracies known as exceptional points (EPs) has led to many intriguing discoveries $[3]$ .

Among these different non-Hermitian systems, a particularly interesting family is constructed with a complex potential that satisfies two of the most fundamental symmetries in nature, i.e., parity and time-reversal symmetries. When combined, they constitute an approach to realizing non-Hermitian Hamiltonians with real energies and, hence, provide a basis for non-Hermitian extension of quantum mechanics $[4]$ . In the past decade, this notion has inspired a plethora of explorations in photonics and related fields $[2,5–7]$ , studying, for example, the spontaneous symmetry breaking of parity-time (PT) and other non-Hermitian symmetries $[8–13]$ , reflectionless scattering modes $[14–16]$ , generalized conservation relations $[16–18]$ , enhanced sensitivity around an EP $[19–21]$ , and unique roles of non-Hermiticity in topological photonics $[22–29]$ .

Another type of non-Hermitian Hamiltonians that have attracted great interest arises from their off-diagonal non-Hermiticity, i.e., in the form of asymmetric hoppings or nonreciprocal couplings $[30,31]$ . Such systems show extreme sensitivity to the boundary condition: While a one-dimensional lattice with real-valued and asymmetric nearest-neighbor (NN) couplings on a ring displays a complex energy spectrum, opening it up leads to a real spectrum instead. The latter is understood through an imaginary gauge transformation, which establishes its equivalence to a Hermitian system with symmetric NN couplings. Such an imaginary gauge transformation essentially exponentializes all bulk states in the system, leading to the so-called non-Hermitian skin effect, which can be observed in photonic [32], acoustic [33], and condensed matter systems [34].

While these two forms of non-Hermiticity have been studied in the system $[13]$ , a deeper connection between them has not been found. One entertaining question naturally arise in this regard: Can an imaginary gauge transformation also establish an equivalence between a non-Hermitian system with a complex potential and a Hermitian system with a real potential? One may attempt to say no because, while a gauge transformation in position space changes the vector and scalar potentials in Maxwell's equations $[35]$ , it leaves the potential invariant in a Schrödinger-like equation $[13]$ . Furthermore, a non-Hermitian and a Hermitian potential differ fundamentally in many aspects, including their degeneracies. While all degenerate states in a Hermitian system have distinct wave functions, they can coalesce in non-Hermitian systems at EPs. The intriguing topology of the Riemann sheets near an EP, including both the real and imaginary parts of the energy, has enabled state flipping by simply encircling the EP in the parameter space $[36,37]$ .

While a single EP can be extended to a ring $[38]$ or a surface $[39]$ , it is unclear whether the coalescing energies around an EP can stay real in a higher-dimensional parameter space, in a fashion similar to a Dirac or Weyl point in Hermitian systems. If such an EP exists, then the absence of a branch cut near it will have a deep impact on both band topology and encircling topology around it. Furthermore, the linear “dispersion” or the sensitivity to a system parameter will also be distinct from known EPs.

In this Letter, we address both intriguing questions raised above regarding the connections between non-Hermitian

PT-symmetric systems with complex potentials and Hermitian systems. We first show, through an imaginary gauge transformation in momentum space, that certain periodic complex potentials in their PT-symmetric phases are equivalent to real (and Hermitian) potentials. We further show that a Dirac EP can be achieved in hybrid dimensions, consisting of one spatial dimension and one synthetic dimension for the gain and loss strength. This conical band structure stays real in the vicinity of this embedded EP at the center of the Brillouin zone (BZ), which coexists with conventional EPs at the edge of the BZ. Surprisingly, they can be swapped when an additional term is introduced to this potential.

Let us consider the Schrödinger equation

$$
i \frac {d}{d t} \psi (x, t) = [ - \partial_ {x} ^ {2} + V (x) ] \psi (x, t),\tag{1}
$$

where we have used dimensionless time, position, and potential. Optical waves propagating in coupled waveguides satisfy essentially the same equation, i.e., the paraxial equation where t is replaced by the propagation distance z, and they are used routinely to demonstrate various non-Hermitian photonic effects $[40,41]$ . We define $V_{m}(x)=V_{0}(\cos mx+i\tau\sin mx)(\tau\geq0)$ and choose $V(x)=V_{1}(x)$ , which is periodic with the period $a=2\pi$ and PT symmetric, satisfying $V(x)=V^{*}(-x)$ $[2]$ . The asterisk denotes complex conjugation and represents time reversal, and the imaginary part of the potential represents optical gain and loss.

In order to perform the imaginary gauge transformation in momentum space, we first expand the Bloch wave function in the plane-wave basis, i.e.,

$$
\psi_ {n k} (x, t) = e ^ {i k x - i \omega t} \sum_ {m \in \mathbb {Z}} a _ {m} e ^ {i m x},\tag{2}
$$

which leads to the following equation that determines its band structure $\omega_{nk}$ :

$$
H _ {k} \Psi_ {n k} (m) = \omega_ {n k} \Psi_ {n k} (m).\tag{3}
$$

Here, $n=1,2,\ldots$ is the band index, $\Psi_{nk}(m)=[\ldots,a_{-1},a_{0},a_{1},\ldots]^{T}$ , and the Bloch Hamiltonian $H_{k}$ is a tridiagonal matrix given by

$$
\begin{array}{l} H _ {k} = \sum_ {m \in \mathbb {Z}} (m + k) ^ {2} | m \rangle \langle m | + t _ {-} | m \rangle \langle m + 1 | \\ \qquad + t _ {+} | m \rangle \langle m - 1 |, \end{array}\tag{4}
$$

where $t_{\pm}=V_{0}(1\pm\tau)/2\in\mathbb{R}$ . $H_{k}$ resembles a tight-binding Hamiltonian in position space with NN couplings, which are asymmetric when $\tau\neq0$ . If we perform a gauge transformation by multiplying the mth element of the wave function $\Psi_{nk}(m)$ by $e^{im\theta}$ , $H_{k}$ is then transformed to $\tilde{H}_k = GH_k G^{-1}$ without changing its diagonal elements [13], where $G = \mathrm{Diag}[\dots, e^{-2i\theta}, e^{-i\theta}, 1, e^{i\theta}, e^{2i\theta}, \dots]$ is a diagonal matrix. Now, if we let $\theta$ equal

![](images/cdf0a87e4ad546e7ea6b2e7607f46e8ccc046d81b182f1ee8ed3ecbb3066d001.jpg)

![](images/6739b7fdc076e542d7bb893db153c6995634401fe608db70b62d6f97b6f57e8c.jpg)

![](images/09404458071b6ac002ee29af9903d5d025b67e2b33e1d15197fddedabe0d6e34.jpg)

![](images/a1f6bdc6b4d46941465283593e3ff9fb6a1e582dd56ec016293694965db0cce1.jpg)
FIG. 1. Equivalence of a non-Hermitian system in its PT-symmetric phase (a) and a Hermitian system (b) related by an imaginary gauge transformation in momentum space. (c) Their identical band structure showing the first three bands. $V_{0}=1$ , $a=2\pi$ , and $\tau=0.8$ . (d) The thin line and dots show $|\Psi_{1k}|$ and $|G\Psi_{1k}|$ at k=0 in the non-Hermitian system, respectively. Arrows mark the scaling of the wave function due to the imaginary gauge transformation. The thick line shows $|\tilde{\Psi}_{1k}|$ in the Hermitian system, coinciding with the dots.

$$
\theta = i \frac {1}{2} \ln \frac {t _ {+}}{t _ {-}},\tag{5}
$$

it is straightforward to see that the resulting $\tilde{H}_{k}$ features symmetric NN coupling $t = \sqrt{t_{-}t_{+}}$ .

We note that $\theta$ is imaginary when $\tau < 1$ , and, hence, $G$ represents an imaginary gauge transformation [30-32], which performs an $m$ -dependent scaling of the momentum-space wave function. In contrast, $t_{-}$ is negative when $\tau > 1$ , and $\theta$ becomes complex with a real part equal to $\pi/2$ . As a result, the gauge transformation is now a complex one instead of an imaginary one. In both cases, $t$ can be written as $t = V_0 \sqrt{1 - \tau^2}/2$ , and, by comparing $\tilde{H}_k$ with $\tau \neq 0$ and $H_k$ with $\tau = 0$ , we know immediately that $\tilde{H}_k$ is the Bloch Hamiltonian of a system with the potential

$$
\tilde {V} (x) = V _ {0} \sqrt {1 - \tau^ {2}} \cos x.\tag{6}
$$

This finding is quite unusual: Our system with $V(x) = V_{0}(\cos x + i \sin x)$ is in its PT-symmetric phase when $\tau < 1$ [Fig. 1(c)], and it has the same band structure as the Hermitian system with the real potential $\tilde{V}(x) = V_{0}\sqrt{1 - \tau^{2}} \cos x$ [Fig. 1(b)]. This observation holds not only for a one-dimensional “crystal” of infinite length, but also for a finite-sized ring of length $L = 2\pi$ , which is just a special case of our discussions above with k = 0 [42].

![](images/919fbcedf57a984fd8f572d90ab8bf236fccb4dd3f9c8651aa667b5452e72133.jpg)
(c)

![](images/382aed077333a5fc552f41641b90a37564037e11e9d52ecc7ae6723baa17e320.jpg)

![](images/347f8e5bfed4942db673ee19851c7fce2a28711b12df1c55933643a7f8a9a50b.jpg)

(d)
![](images/e9c5bcd8d6aa2c954830daed994862bf6361e304fc5e334a79dcc16b5b975d08.jpg)
FIG. 2. Equivalence of a non-Hermitian system in its PT-broken phase (a) and another with an imaginary potential (b) related by a complex gauge transformation in momentum space. (c) and (d) Real and imaginary parts of their identical band structure, showing the first three bands. $V_{0}=1$ , $a=2\pi$ , and $\tau=1.1$ .

We note that the band structures shown in Fig. 1(c) are found numerically using the finite difference method in position space [43], which has no knowledge of the gauge transformation $G$ we performed in momentum space. Nevertheless, the Fourier transform of the obtained Hermitian Bloch wave functions [denoted by $\tilde{\Psi}_{nk}(m)$ ] are indeed given by those of the non-Hermitian system after the gauge transformation [Fig. 1(d)], which further elucidates their equivalence besides their identical band structure. We also note that the participation ratio $\mathrm{PR} = (\sum_{m}|a_{m}|^{2})^{2} / \sum_{m}|a_{m}|^{4}$ , which measures the localization length here in momentum space, reduced from 1.99 in $\Psi_{nk}(m)$ to 1.29 in $\tilde{\Psi}_{nk}(m)$ , similar to the non-Hermitian skin effect in position space albeit not as drastic [42].

This imaginary gauge transformation also provides a different perspective on the transition of the original non-Hermitian system to its $\mathcal{PT}$ -broken phase when $\tau > 1$ : The equivalent system with the Bloch Hamiltonian $\tilde{H}_k$ no longer has a real potential $\tilde{V}(x)$ when $\tau > 1$ ; instead, it has an imaginary potential $\tilde{V}(x) = iV_0\sqrt{\tau^2 - 1}\cos x$ [Fig. 2(b)], leading to a partially complex band structure [Figs. 2(c) and 2(d)]. In momentum space, this change is reflected by the change of the coupling $t$ from real to imaginary in the equivalent Bloch Hamiltonian $\tilde{H}_k$ . Right at $\tau = 1$ , the angle $\theta$ is undefined, and so is the gauge transformation.

Next, we address the other question raised in the introduction, i.e., whether a non-Hermitian degeneracy (i.e., an EP) can be embedded as a point singularity in higher-dimensional parameter space, similar to Dirac and Weyl points in Hermitian systems. To this end, we first note that, in the system discussed above, all neighboring bands of $H_{k}$ with $\tau = 1$ touch at either the center or the edge of the BZ [black solid lines in Fig. 3(a)]. While the ones at the edge of the BZ are known to be EPs [2], the ones at the center have not been studied in this regard. They may seem to resemble accidental diabolic points [38] or carefully engineered non-Hermitian diabolic points [44], both featuring distinct wave functions, but these degeneracies at k = 0 in Fig. 3(a) are EPs instead, as we exemplify in Fig. 3(b): The second and third bands have identical wave functions at their touching point at k = 0, and so do the fourth and fifth bands [42]. However, unlike the ones at the edge of the BZ that undergo a PT transition when $\tau$ becomes greater than unity [Fig. 3(c)], the EPs at k = 0 do not and the energies stay real in its vicinity. We refer to the one shown in Fig. 3(a) as a Dirac EP, because the energy difference between the second and third bands near it is a linear function of both $\tau$ and k (see also Ref. [42]). Note that it is fundamentally different from previously studied Dirac points in non-Hermitian (and Hermitian) systems, which are still diabolic points and not EPs.

![](images/772fb4ca4adb19bf6c33bd141cf501b9846f5cef7b0d3a36db5ca04abab82c06.jpg)

![](images/94eb7ff909b6716c4ebdcba37797403284c534d0995412b6cc506a0122e7ca6e.jpg)
(d)

![](images/64c7c0672412383bfad6d6a08a79132ee5938ad9b72c858e3dcd04454b1b3c4a.jpg)

![](images/ee52f75a4b262a16c34ab2a0051436eccca9bff939512b5df3fa77e08ba495da.jpg)
FIG. 3. Two types of EPs in the same system with $V(x) = V_{1}(x)$ . (a) Its band structure plotted in the hybrid dimensions of k and $\tau$ . Black solid lines show the band structure at $\tau = 1$ . Black and colored dots show conventional EPs and the Dirac EP, respectively. (b) Coalesced wave functions at the conventional EP (dashed line) and the Dirac EP (solid line). (c) and (d) Changes of the band structure as a function of $\tau$ at k = 0.5 and 0, respectively.

To gain some analytical insights on the contrasting properties of these EPs occurring at $\tau=1$ , we truncate the momentum-space Hamiltonian $H_{k}$ given by Eq. (4) at k=0 and 0.5, respectively. Note that the diagonal elements of $H_{k}$ , given by $(m+k)^{2}$ , are symmetric at both these k values: The ones with $m=-m_{0}$ and $m_{0}(\geq0)$ are the same when k=0, and the ones with $m=-(m_{0}+1)$ and $m_{0}$ are the same when k=0.5. We maintain these symmetries when truncating $H_{k}$ , and we aim to show that the truncated Hamiltonian indeed has an EP at $\tau=1$ and, more importantly, that this EP evolves to two energies that behave differently across $\tau = 1$ , with the ones at $k = 0$ being real and the ones at $k = 0.5$ experiencing a $\mathcal{PT}$ transition.

![](images/bcf2bcd3f627460d1b7ef6d2feb48722fec626ea4d3a4fd1eaf994bec92cac80.jpg)

![](images/cbc4c6691eec8fcd8e147e302004f45400e261b097731713d632203f9e5e8444.jpg)

![](images/9397a9127b283cee60ee62911b06007de82e3f26119ccffe398af213920ae5c2.jpg)

![](images/19b90e01e63730b8c60922b2bfe90437b2b2fcb330f47baa3170f426f7d33e4b.jpg)
FIG. 4. The same as Fig. 3 but with $V(x) = V_{1}(x) + V_{2}(x)$ . The regular and Dirac EPs are now swapped, with the former at $k = 0$ and the latter at $k = 0.5$ .

At k = 0.5, the simplest truncation retaining the aforementioned symmetry keeps the m = -1 and 0 block of the full $H_{k}$ :

$$
H ^ {(2)} = \left( \begin{array}{c c} \omega & t _ {-} \\ t _ {+} & \omega \end{array} \right),\tag{7}
$$

where $\omega = 0.5^{2}$ . Indeed, this truncated Hamiltonian features a conventional EP at $\tau = 1$ : Its two eigenvalues are real (i.e., $\omega_{\pm} = \omega \pm |t|$ ) when $\tau < 1$ and complex conjugates (i.e., $\omega_{\pm} = \omega \pm i|t|$ ) when $\tau > 1$ , hence experiencing a PT transition. Here, $t = \sqrt{t_{-}t_{+}} = V_{0}\sqrt{1 - \tau^{2}}/2$ as before, and these two eigenvalues correspond to the first and second bands that host the conventional EP in the full Hamiltonian.

Similarly, at k=0 where the Dirac EP exists, the simplest yet nontrivial truncation retaining the aforementioned symmetry keeps the m=-1, 0, and 1 block of the full $H_{k}$ :

$$
H ^ {(3)} = \left( \begin{array}{c c c} \omega & t _ {-} & 0 \\ t _ {+} & 0 & t _ {-} \\ 0 & t _ {+} & \omega \end{array} \right),\tag{8}
$$

where $\omega = 1$ . The three eigenvalues of $H^{(3)}$ are given by $\omega, (\omega \pm \sqrt{\omega^{2} + 8t^{2}})/2$ . At $\tau = 1$ , they are $\omega, \omega$ , and 0. The last one gives the energy of the first band at k = 0 in the full $H_{k}$ , and the first two are at the Dirac EP. Note that these two eigenvalues of $H^{(3)}$ are real on both sides of $\tau = 1$ in its vicinity (defined by $|t| < \omega/2\sqrt{2}$ ), which is a prominent property of the Dirac EP as we have mentioned.

Intriguingly, the conventional and Dirac EPs are switched when we choose $V(x) = V_{1}(x) + V_{2}(x)$ and set $V_{0} = 1$ , where $V_{m}(x)$ was introduced below Eq. (1). As Fig. 4(a) shows, its band structure at $\tau = 1$ is identical to that with $V(x) = V_{1}(x)$ shown in Fig. 3(a). However, as $\tau$ becomes greater than unity, the bands undergo a PT transition at the center of the BZ instead of at its edge, i.e., with the conventional and Dirac EPs switched [Figs. 4(c) and 4(d)]. This switching can again be understood using our analysis in momentum space, where $V_{2}(x)$ adds asymmetric next-nearest-neighbor (NNN) couplings to $H_{k}$ [42].

![](images/762231836a878e4dd65f3a96a888d561f83035173a5a58fe20e98324d3e97913.jpg)

![](images/cb5500a0efa26dbd5e06a25304fb42a3eb074ba252def8e35861f8fa61a04255.jpg)

![](images/3951570ea3446c6e9adfe552b7d8772e8533dd2431970dfa1552e2ef66d148f9.jpg)

![](images/abf62a67af3c3d46ee1f65068900378988e7899d15b3c40d2a229ea958b3581b.jpg)
FIG. 5. The same as Fig. 1 but with a $\mathcal{PT}$ -symmetric potential $V(x) = V_{1}(x) + V_{0}[(1 + \tau^{2})\cos 2x + 2\tau i\sin 2x]$ in (a) and a real potential $\tilde{V}(x) = V_0[\sqrt{1 - \tau^2}\cos x + (1 - \tau^2)\cos 2x]$ in (b).

For this more complicated potential, it cannot be transformed to a real (and Hermitian) potential using a gauge transformation in momentum space. However, there are special cases where such equivalence can be established, despite that their Bloch Hamiltonians have both asymmetric NN and NNN couplings [42]. $V(x) = V_{1}(x) + V_{0}[(1 + \tau^{2}) \cos 2x + 2\tau i \sin 2x](0 < \tau < 1)$ is one example. This potential is PT symmetric, and its Bloch Hamiltonian is given by

$$
\begin{array}{l} H _ {k} = \sum_ {m \in \mathbb {Z}} (m + k) ^ {2} | m \rangle \langle m | + t _ {-} | m \rangle \langle m + 1 | + t _ {+} | m \rangle \langle m - 1 | \\ \qquad + t _ {-} ^ {\prime} | m \rangle \langle m + 2 | + t _ {+} ^ {\prime} | m \rangle \langle m - 2 |, \end{array}
$$

where $t'_{\pm}=V_{0}(1\pm\tau)^{2}/2$ . The same imaginary gauge transformation we have used turns it into an $\tilde{H}_{k}$ with symmetric NN coupling $t=\sqrt{t_{-}t_{+}}$ and symmetric NNN coupling $t'=\sqrt{t_{-}'t_{+}'}=V_{0}(1-\tau^{2})/2$ , which corresponds to a Hermitian system with a real potential $\tilde{V}(x)=V_{0}[\sqrt{1-\tau^{2}}\cos x+(1-\tau^{2})\cos 2x]$ (Fig. 5).

In summary, we have first shown that two well-studied forms of non-Hermiticity, i.e., a complex potential and asymmetric hoppings, can be rigorously related in one-dimensional periodic systems by analyzing the former in momentum space. This relation is not limited to the examples we have chosen above $[42]$ , which, however, do allow us to apply the imaginary gauge transformation in momentum space and find their equivalent Hermitian potentials. This transformation should be distinguished from the change to the canonical momentum after a gauge transformation in position space $[45]$ . We have also reported the finding of a Dirac EP in hybrid dimensions, consisting of one spatial dimension and a synthetic dimension for the gain and loss strength. Its implication on topological photonics will be studied in a future work.

This project is supported by National Science Foundation under Grants No. PHY-1847240 and No. ECCS-1846766.

$^{*}$ li.ge@csi.cuny.edu

[1] G. Gamow, Z. Phys. 51, 204 (1928).

[2] L. Feng, R. El-Ganainy, and L. Ge, Nat. Photonics 11, 752 (2017).

[3] M.-A. Miri and A. Alù, Science 363 (2019).

[4] C. M. Bender and S. Boettcher, Phys. Rev. Lett. 80, 5243 (1998).

[5] R. El-Ganainy, K. G. Makris, M. Khajavikhan, Z. H. Musslimani, S. Rotter, and D. N. Christodoulides, Nat. Phys. 14, 11 (2018).

[6] V. V. Konotop, J. Yang, and D. A. Zezyulin, Rev. Mod. Phys. 88, 035002 (2016).

[7] S. Longhi, Europhys. Lett. 120, 64001 (2017).

[8] K. G. Makris, R. El-Ganainy, and D. N. Christodoulides, Phys. Rev. Lett. 100, 103904 (2008).

[9] L. Ge and H. E. Türeci, Phys. Rev. A 88, 053810 (2013).

[10] L. Ge, Phys. Rev. A 95, 023812 (2017).

[11] B. Qi, L. Zhang, and L. Ge, Phys. Rev. Lett. 120, 093901 (2018).

[12] L. Ge, Photonics Res. 6, A10–A17 (2018).

[13] J. D. H. Rivero and L. Ge, Phys. Rev. B 103, 014111 (2021).

[14] Z. Lin, H. Ramezani, T. Eichelkraut, T. Kottos, H. Cao, and D. N. Christodoulides, Phys. Rev. Lett. 106, 213901 (2011).

[15] W. R. Sweeney, C. W. Hsu, and A. D. Stone, Phys. Rev. A 102, 063511 (2020).

[16] L. Ge, Y. D. Chong, and A. D. Stone, Phys. Rev. A 85, 023802 (2012).

[17] J. D. H. Rivero and L. Ge, Phys. Rev. Lett. 125, 083902 (2020).

[18] L. Ge, K. G. Makris, D. N. Christodoulides, and L. Feng, Phys. Rev. A 92, 062135 (2015).

[19] W. Chen, S. K. Ozdemir, G. Zhao, J. Wiersig, and L. Yang, Nature (London) 548, 192 (2017).

[20] Y.-H. Lai, Y.-K. Lu, M.-G. Suh, Z. Yuan, and K. Vahala, Nature (London) 576, 65 (2019).

[21] H. Hodaei, A. U. Hassan, S. Wittek, H. Garcia-Gracia, R. El-Ganainy, D. N. Christodoulides, and M. Khajavikhan, Nature (London) 548, 187 (2017).

[22] K. Kawabata, K. Shiozaki, M. Ueda, and M. Sato, Phys. Rev. X 9, 041015 (2019).

[23] P. St-Jean, V. Goblot, E. Galopin, A. Lemaître, T. Ozawa, L. Le Gratiet, I. Sagnes, J. Bloch, and A. Amo, Nat. Photonics 11, 651 (2017).

[24] B. Bahari, A. Ndao, F. Vallini, A. El Amili, Y. Fainman, and B. Kanté, Science 358, 636 (2017).

[25] M. A. Bandres, S. Wittek, G. Harari, M. Parto, J. Ren, and M. Segev, Science 359, eaar4005 (2018).

[26] H. Zhao, P. Miao, M. H. Teimourpour, S. Malzard, R. El-Ganainy, H. Schomerus, and L. Feng, Nat. Commun. 9, 981 (2018).

[27] M. Pan, H. Zhao, P. Miao, S. Longhi, and L. Feng, Nat. Commun. 9, 1308 (2018).

[28] M. Parto, S. Wittek, H. Hodaei, G. Harari, M. A. Bandres, J. Ren, M. C. Rechtsman, M. Segev, D. N. Christodoulides, and M. Khajavikhan, Phys. Rev. Lett. 120, 113901 (2018).

[29] D. Leykam, K. Y. Bliokh, C. Huang, Y. D. Chong, and F. Nori, Phys. Rev. Lett. 118, 040401 (2017).

[30] N. Hatano and D.R. Nelson, Phys. Rev. Lett. 77, 570 (1996).

[31] N. Hatano and D.R. Nelson, Phys. Rev. B 56, 8651 (1997).

[32] S. Longhi, D. Gatti, and G. D. Valle, Sci. Rep. 5, 13376 (2015).

[33] W. Wang, X. Wang, and G. Ma, Nature (London) 608, 50 (2022).

[34] S. Franca, V. K:onye, F. Hassler, J. van den Brink, and C. Fulga, Phys. Rev. Lett. 129, 086601 (2022).

[35] J. D. Jackson, Classical Electrodynamics, 3rd ed. (John Wiley and Sons, New York, 1999).

[36] J. Doppler, A. A. Mailybaev, J. Böhm, U. Kuhl, A. Girschik, F. Libisch, T. J. Milburn, P. Rabl, N. Moiseyev, and S. Rotter, Nature (London) 537, 76 (2016).

[37] H. Xu, D. Mason, L. Jiang, and J. G. E. Harris, Nature (London) 537, 80 (2016).

[38] B. Zhen, C. W. Hsu, Y. Igarashi, L. Lu, I. Kaminer, A. Pick, S.-L. Chua, J. D. Joannopoulos, and M. Soljačić, Nature (London) 525, 354 (2015).

[39] X. Zhang, K. Ding, X. Zhou, J. Xu, and D. Jin, Phys. Rev. Lett. 123, 237202 (2019).

[40] A. Guo, G. J. Salamo, D. Duchesne, R. Morandotti, M. Volatier-Ravat, V. Aimez, G. A. Siviloglou, and D. N. Christodoulides, Phys. Rev. Lett. 103, 093902 (2009).

[41] C. E. Rüter, K. G. Makris, R. El-Ganainy, D. N. Christodoulides, M. Segev, and D. Kip, Nat. Phys. 6, 192 (2010).

[42] See Supplemental Material at http://link.aps.org/supplemental/10.1103/PhysRevLett.129.243901 for additional discussions.

[43] L. Ge, Photonics Res. 5, B20 (2017).

[44] H. Xue, Q. Wang, B. Zhang, and Y. D. Chong, Phys. Rev. Lett. 124, 236403 (2020).

[45] J. J. Sakurai and J. Napolitano, Modern Quantum Mechanics, 2nd ed. (Cambridge University Press, Cambridge, England, 2017).
