Editors' Suggestion

# Experimental Observation of Dirac Exceptional Points

Yang Wu $^{1,2}$ , Dongfanghao Zhu, $^{1,2}$ Yunhan Wang, $^{1,3,2}$ Xing Rong $^{1,3,2,*}$ and Jiangfeng Du $^{1,3,2,4,\dagger}$

$^{1}$ CAS Key Laboratory of Microscale Magnetic Resonance and School of Physical Sciences,

University of Science and Technology of China, Hefei 230026, China

$^{2}$ Anhui Province Key Laboratory of Scientific Instrument Development and Application,

$^{3}$ Hefei National Laboratory, University of Science and Technology of China, Hefei 230088, China $^{4}$ Institute of Quantum Sensing and School of Physics, Zhejiang University, Hangzhou 310027, China

(Received 2 July 2024; revised 24 February 2025; accepted 11 March 2025; published 14 April 2025)

The energy-level degeneracies, also known as exceptional points (EPs), are crucial for comprehending emerging phenomena in materials and enabling innovative functionalities for devices. Since EPs were proposed over half a century ago, only two types of EPs have been experimentally discovered, revealing intriguing phases of materials such as Dirac and Weyl semimetals. These discoveries have showcased numerous exotic topological properties and novel applications, such as unidirectional energy transfer. Here, we report the observation of a novel type of EP, named the Dirac EP, utilizing a nitrogen-vacancy center in diamond. Two of the eigenvalues are measured to be degenerate at the Dirac EP and remain real in its vicinity. This exotic band topology associated with the Dirac EP enables the preservation of the symmetry when passing through, and makes it possible to achieve adiabatic evolution in non-Hermitian systems. We examined the degeneracy between the two eigenstates by quantum state tomography, confirming that the degenerate point is a Dirac EP rather than a Hermitian degeneracy. Our research of the distinct type of EP contributes a fresh perspective on dynamics in non-Hermitian systems and is potentially valuable for applications in quantum control in non-Hermitian systems and the study of the topological properties of EPs.

DOI: 10.1103/PhysRevLett.134.153601

Introduction—Exceptional points (EPs) were introduced in mathematics to characterize the energy-level degeneracy of a linear operator over half a century ago $[1]$ . Since EPs were proposed, only two types of EPs have been experimentally discovered. One type consists of degeneracies in the Hermitian system, such as diabolical points $[2,3]$ . The eigenvalues are always real and vary linearly with the parameter near the Hermitian degeneracy. At the Hermitian degeneracy, the eigenvalues are degenerate, but the eigenstates are orthogonal [Fig. 1(a)]. This type of EP has been extensively studied, revealing intriguing phases of materials such as Dirac and Weyl semimetals $[4]$ . The other type consists of the typical EPs introduced by non-Hermitian physics $[5–11]$ , where both eigenvalues and eigenstates are degenerate [Fig. 1(b)]. Typical EPs in non-Hermitian systems are usually accompanied by changes in symmetries, like parity-time (PT), anti-PT, or particle-hole symmetry. A non-Hermitian system undergoes a corresponding breaking or restoration of symmetry across the EP. This occurrence is often marked by significant alternations in the energy spectrum, shifting between real and imaginary values. Investigations of typical EPs not only have deepened the understanding of topological physics, such as novel non-Hermitian topological phases $[12–15]$ , exceptional nodal topologies $[16,17]$ , and topological state control $[11]$ , but also have given rise to a variety of potential applications, such as single-mode lasing $[18,19]$ and unidirectional invisibility $[20–23]$ . Moreover, the fractional exponent response of eigenvalues to parameter perturbations, such as a square- and cubic-root relationship near a second-order and third-order EP, implies potentially enhanced sensitivity $[24–27]$ .

Here, we report the experimental observation of a novel type of EPs, named the Dirac EPs $[28]$ , utilizing a nitrogen vacancy (NV) center in diamond. The simultaneous degeneracy of two of the eigenvalues and eigenstates clearly distinguishes the Dirac EP from the Hermitian degeneracy [Fig. 1(c)]. The eigenvalues measured near the EP are all real, which maintains the parity-time symmetry when passing through the Dirac EP. Due to the real eigenvalues around the Dirac EP, nonadiabatic transitions that occur when encircling typical EPs no longer appear. Moreover, the two eigenvalues connected at the Dirac EP exhibit a linear and conical dispersion with the change of the parameters. The multitude of exotic properties exhibited by the Dirac EP, in contrast to the typical EPs, deepens our understanding of mechanism for dynamics in non-Hermitian systems and holds potential for the investigation of mode switching without dissipation and adiabatic evolution in non-Hermitian systems.

![](images/99a8e78e3472ec98ee45e9c571caf6e4ce07b546d0e1a53dce80e62e1a920811.jpg)

![](images/acf172ae22a9b3fbea68154876ab9e82f147d5ffc74a5546bb14c0e9dabdd384.jpg)

![](images/d191106eba255bde0f311174b7286fc75c01782450f63956ba1aa9e97946b8fd.jpg)
FIG. 1. Degeneracy of Hermitian system and typical EP and Dirac EP of non-Hermitian system. (a)–(c) The real (blue lines) and imaginary (red dashed lines) parts of the Hamiltonian hosting different types of degeneracies. (a) Near the Hermitian degeneracy, the eigenvalues are always real and vary linearly with the parameter k. The eigenstates $|\psi_{1}\rangle$ and $|\psi_{2}\rangle$ are orthogonal at the Hermitian degeneracy. (b) The eigenvalues change from real to imaginary at typical EP and exhibit a square-root dependence on the parameters. The eigenstates coalesce at typical EP. (c) In the vicinity of the Dirac EP, the eigenvalues are real and have a linear relationship with the parameters. The eigenstates are degenerate at the Dirac EP.

Theoretical model—To observe the Dirac EP, we construct a PT-symmetric non-Hermitian Hamiltonian in the tight binding model as follows [28]:

$$
\begin{array}{l} H _ {\mathrm{tb}} = \sum_ {m \in \mathbb {Z}} (m + k _ {1}) ^ {2} | m \rangle \langle m | + \frac {1 - k _ {2}}{2} | m \rangle \langle m + 1 | \\ \qquad + \frac {1 + k _ {2}}{2} | m \rangle \langle m - 1 |, \end{array}\tag{1}
$$

where $k_{1}$ represents momentum and $k_{2}$ characterizes the magnitude of the nonreciprocity of the coupling between adjacent lattices. By truncating the block of m = -1, 0, and 1 in the tight binding model [28], the $3 \times 3$ non-Hermitian Hamiltonian can be obtained as follows:

$$
H \left(k _ {1}, k _ {2}\right) = 3 S _ {z} ^ {2} + 2 k _ {1} S _ {z} + \sqrt {2} \left(S _ {x} - i k _ {2} S _ {y}\right),\tag{2}
$$

where $S_{x}$ , $S_{y}$ , and $S_{z}$ are spin-1 operators. The Hamiltonian in Eq. (2) naturally has a Dirac EP in its parameter space, without introducing any high-order terms of parameters, which is a must for constructing Dirac EPs in two-band models [29]. The positions of EPs are determined by the condition that the characteristic polynomial $P(E) \equiv \det[H(k_{1}, k_{2}) - E] = f_{3}E^{3} + f_{2}E^{2} + f_{1}E + f_{0}$ has multiple roots, where $f_{3} = -1$ , $f_{2} = 6$ , $f_{1} = 4k_{1}^{2} - 2k_{2}^{2} - 7$ , $f_{0} = 6(k_{2}^{2} - 1)$ . (See Supplemental Material [30], Sec. 2, for the detailed derivation.) As shown in Fig. 2, an exceptional line consisting entirely of typical EPs emerges as the black line. The Dirac EP is located at $k_{1} = 0$ , $k_{2} = \pm 1$ . There exists a real–imaginary transition of the eigenvalues when crossing a typical EP, whereas the imaginary parts of the three eigenvalues are all zero in the vicinity of the Dirac EP [Fig. 2(c)]. Therefore, the three eigenvalues are all real numbers near the Dirac EP. The real parts of the two eigenvalues are also degenerate at the Dirac

EP [Fig. 2(a)]. Unlike the square-root relationship commonly observed near typical EPs, the eigenvalues exhibit a linear relationship with parameter variations in the vicinity of the Dirac EP. For any direction in the two-dimensional space, the eigenvalues satisfy the relationship $E_{1,2} = 3 + 2(-\sin \theta \pm \sqrt{\sin^2\theta + 9\cos^2\theta})\Delta k / 3 + O[(\Delta k)^2]$ , where $\Delta k_{1} = \cos \theta \Delta k$ , $\Delta k_{2} = \sin \theta \Delta k$ , $\theta \in [0,2\pi]$ , and $\Delta k > 0$ . Therefore, the linear relationship is satisfied in all directions, provided that the parameters are very close to the Dirac EP, presenting the Dirac cone [Fig. 2(b)].

Experimental realizations—The nitrogen-vacancy (NV) center, which is an atomic-scale defect in diamond, was utilized to observe the Dirac EP. (See Supplemental Material [30], Sec. 1, for the experimental setup.) The Dirac EP is characterized by the eigenvalue and eigenstate information of the non-Hermitian Hamiltonian $H(k_{1}, k_{2})$ , which can be obtained by the evolution of the quantum state $|\psi(t)\rangle$ under the non-Hermitian Hamiltonian. Taking the electron spin in the NV center as the system and the nuclear spin as the ancilla qubit, the evolution under the non-Hermitian Hamiltonian $H(k_{1}, k_{2})$ can be realized based on the dilation method [34–37]. The evolution under the non-Hermitian Hamiltonian is achieved in a subspace of the $6 \times 6$ dilated Hermitian Hamiltonian $H_{tot}$ . The detailed procedure is as follows: First, the coupled system is prepared to the initial state $|\psi(0)\rangle_{e}|-\rangle_{n} + [\eta_{0}|\psi(0)\rangle_{e}]|+\rangle_{n}$ , where subscripts e and n label the electron and nuclear spin states, respectively. Here, $\eta_{0}$ is a factor properly chosen for the convenience of experimental realization, and $|\pm\rangle_{n}$ are the eigenstates of the Pauli matrix $\sigma_{y}$ . Second, the dilated Hamiltonian $H_{tot}$ is constructed for the non-Hermitian Hamiltonian $H(k_{1}, k_{2})$ as

![](images/7878639b87d48146ebd64495f6172338209ad673301a4ce7107c0b4bfc414d36.jpg)

![](images/ad7559101b592e0ae2fed72715352b2e9da2b236500dfc75a9a6edc8f69adcb7.jpg)

![](images/dbe49696e1be32fe90263c432046f2468c04c14a68876f6cf23d7b29b2fcfadf.jpg)
FIG. 2. The eigenvalue structure and the Dirac exceptional points of the non-Hermitian Hamiltonian $H(k_{1}, k_{2})$ . The colored surfaces are the eigenvalue sheets of $H(k_{1}, k_{2})$ for the (a) real and (c) imaginary parts. Yellow, green, and purple sheets correspond to eigenvalues $E_{1}$ , $E_{2}$ , and $E_{3}$ , respectively. The Dirac EP marked by the red dot is located at $k_{1} = 0$ and $k_{2} = 1$ , where the eigenvalues $E_{1}$ and $E_{2}$ degenerate. (b) The eigenvalue structure near the Dirac EP of the non-Hermitian Hamiltonian showing the Dirac cone. The imaginary parts of the three eigenvalues near Dirac EP are zero. An exceptional line consisting entirely of typical EPs is represented by the black line.

$$
H _ {\mathrm{tot}} (t) = \Gamma (t) \otimes | 1 \rangle_ {n n} \langle 1 | + \Lambda (t) \otimes | 0 \rangle_ {n n} \langle 0 |.\tag{3}
$$

Here, $\Gamma(t)$ and $\Lambda(t)$ are Hermitian operators in the following form:

$$
\begin{array}{l} \Gamma (t) = \left[ \begin{array}{c c c} d _ {1} (t) & a _ {1} (t) & c _ {1} (t) \\ a _ {1} ^ {*} (t) & d _ {2} (t) & b _ {1} (t) \\ c _ {1} ^ {*} (t) & b _ {1} ^ {*} (t) & d _ {3} (t) \end{array} \right], \\ \Lambda (t) = \left[ \begin{array}{c c c} d _ {4} (t) & a _ {2} (t) & c _ {2} (t) \\ a _ {2} ^ {*} (t) & d _ {5} (t) & b _ {2} (t) \\ c _ {2} ^ {*} (t) & b _ {2} ^ {*} (t) & d _ {6} (t) \end{array} \right], \end{array}\tag{4}
$$

where the parameters $a_{i}(t)$ , $b_{i}(t)$ , $c_{i}(t)$ (i=1,2), and $d_{j}(t)(j=1,\ldots,6)$ are determined by the non-Hermitian Hamiltonian $H(k_{1},k_{2})$ according to the dilation method. (See Supplemental Material [30], Sec. 5, for the derivation and the detailed expressions.) Third, the dilated Hamiltonian $H_{tot}$ can be realized by applying microwave and electric field control pulses in the NV center system. The initial state evolves to $|\psi(t)\rangle_{e}|-\rangle_{n} + [\eta(t)|\psi(t)\rangle_{e}]|+\rangle_{n}$ under the Hamiltonian $H_{tot}$ , where $\eta(t)$ is a properly chosen time-dependent operator. The effective Hamiltonian governing the evolution of $|\psi(t)\rangle_{e}$ in the $|- \rangle_{n}$ subspace corresponds to $H(k_{1},k_{2})$ . The diagonal elements $d_{j}(t)$ of $H_{\mathrm{tot}}(t)$ were realized by choosing an appropriate interaction picture. The off-diagonal elements $a_{1}(t)$ , $b_{1}(t)$ , $a_{2}(t)$ , and $b_{2}(t)$ of $H_{\mathrm{tot}}(t)$ were realized by four selective microwave pulses. The off-diagonal elements $c_{1}(t)$ and $c_{2}(t)$ in $H_{\mathrm{tot}}(t)$ were implemented by alternating current electric field pulses. The time-dependent amplitudes and phases of the microwave and electric field pulses were appropriately set according to the off-diagonal elements in $H_{\mathrm{tot}}(t)$ . Two examples of the amplitudes and phases of the microwave and electric field pulses are shown in Supplemental Material [30], Sec. 5. Finally, the state evolving under $H(k_{1},k_{2})$ is obtained by postselecting the $|- \rangle_{n}$ subspace, leading to the determination of the eigenvalues and eigenstates of $H(k_{1},k_{2})$ .

The characterization of the Dirac EP requires information about the eigenvalues and eigenstates of the non-Hermitian Hamiltonian. The eigenvalues of the non-Hermitian Hamiltonian are encoded in the population of eigenstates depending on the given model. (See Supplemental Material [30], Sec. 6, for the detailed derivation.) The eigenstates of an NH Hamiltonian $H(k_1, k_2)$ can be obtained from the steady states under the evolution of the NH Hamiltonian $g(H)$ [H and $g(H)$ have the same eigenstates], where $g(x)$ is an analytic function of $x$ . It is worth noting that near the Dirac EP, the eigenvalues are all real. By selecting $g(H)$ as $iH(k_1, k_2), -iH(k_1, k_2)$ , and $1/i(H(k_1, k_2) - cI)$ , where the parameter $c$ is chosen to be closest to one of the eigenvalues, the three eigenstates can be achieved by evolving sufficiently long under NH Hamiltonian $g(H)$ . Then, the eigenvalues can be obtained by measuring the populations of the eigenstates under different bases. Furthermore, the eigenstates can be fully characterized through quantum state tomography.

The entire experimental procedure can be divided into three parts: the state preparation, evolution under the dilated Hamiltonian, and the population measurement. First, the initial state is polarized to $|0\rangle_{e}|1\rangle_{n}$ by optical pumping under the static magnetic field of 501 G. After polarization, the initial state of the coupled system is prepared in the form $|0\rangle_{e}(|-\rangle_{n}+\eta_{0}|+\rangle_{n})$ by single-qubit rotation on the nuclear spin. Following the nuclear spin rotation, an appropriate operation on the electron spin prepares the coupled system into the state $|\psi_{\mathrm{ini}}\rangle_{e}(|-\rangle_{n}+\eta_{0}|+\rangle_{n})$ to ensure a sufficiently high population in the subspace during readout. Second, the dilated Hamiltonian $H_{\mathrm{tot}}(t)$ is implemented using microwave and electric field pulses, whose amplitudes, frequencies, and phases are determined based on parameters in Eq. (4). After a sufficiently long evolution time, the state of the coupled system becomes $|\psi_{i}\rangle_{e}|-\rangle_{n} + [\eta(t)|\psi_{i}\rangle_{e}]|+\rangle_{n}$ , where $|\psi_{i}\rangle_{e}$ is the ith eigenstate of the non-Hermitian Hamiltonian, and $\eta(t)$ is an operator. Finally, the measurement basis is transformed by operations on the electron and nuclear spin before readout. (See Supplemental Material [30], Sec. 6, for the detailed derivation.) By renormalization in the subspace $|- \rangle_{n}$ , measurements of populations in different bases and quantum state tomography of the three eigenstates are realized.

(a)
(b)
![](images/21957783b689d639c2f4ae8b16e86caa107be5f77bfff7a47965578ca22c26ce.jpg)

![](images/853ecde409205cfd53c2a1518875e5b055fbb4897b49f26b3c2bd9f5b7f789e3.jpg)
(d)

(c)
![](images/7bb3ed8857eadb7a2a4dee7df0df47a70797605cfc8d5571fdac948665d1c150.jpg)

![](images/3e46274fe4f327f91c19917502edcb5c45a422211d1d2ff41be27b5da966c3c9.jpg)
FIG. 3. Observation of the Dirac EP. (a),(c) The real and (b),(d) imaginary parts of the three eigenvalues of the non-Hermitian Hamiltonian. Black, red, and green points and lines correspond to the eigenvalues $E_1$ , $E_2$ , and $E_3$ , respectively. (a),(b) When $k_2$ is set to 1, the change of eigenvalues with parameter $k_1$ confirms that the Dirac EP is located at $k_1 = 0$ . (c),(d) By fixing $k_1$ to 0, the Dirac EP can also be confirmed by observing the change of the eigenvalues with another parameter $k_2$ . The black, red, and green dots with error bars are the experimental results of the three eigenvalues. The lines of the corresponding colors are theoretical predictions.

Experimental results—Figure 3 depicts the pure real eigenvalues and linear dispersion in the vicinity of the Dirac EP. We demonstrate the eigenvalues varying with respect to two parameters separately. When $k_{2}$ is set to 1, the eigenvalue $E_{3}$ is always zero as $k_{1}$ varies, as shown in Figs. 3(a) and 3(b). The eigenvalues $E_{1}$ and $E_{2}$ degenerate at the Dirac EP point when $k_{1}=0$ . The real parts of the eigenvalues exhibit a linear response to the parameter $k_{1}$ , while the imaginary parts remain zero, indicating real eigenvalues in the vicinity of the Dirac EP [Fig. 3(b)]. A similar situation arises when $k_{1}$ is set to 0 and $k_{2}$ is varied. The imaginary parts of the eigenvalues remain constant at 0 near the Dirac EP, as shown in Fig. 3(d). Such entirely real eigenvalues ensure that the eigenstates of the non-Hermitian Hamiltonian are always eigenstates of the PT operator when passing through the Dirac EP, indicating the absence of the spontaneous PT symmetry breaking [29]. This is in stark contrast to what occurs at typical EPs, where the eigenvalues transit from real to complex when passing through them. Figures 3(a) and 3(c) show that the real parts

$\operatorname {Im}(\rho_{2}^{\exp})$
(a)
![](images/95fe92d9161392300db48c723c23a725731dd13b2062f1e8e7e3cbc1f9a83095.jpg)

![](images/1e707c9a77c8f37038bc497a94909be3b14ce78def3b7f7863ba498ae7209c6a.jpg)
(b)

(c)
$\operatorname{Re}\left(\rho_{2}^{\exp}\right)$
![](images/cd6d56a3a2f02d362c7a9832732d4677ed0f6a26b4ccf4a983b5a85a6c18f867.jpg)

![](images/a960a141fb8354a14676c0a57984ca6dd1dd4be959799936c9c0d783857b70b1.jpg)
(e)
$\operatorname{Re}\left(\rho_{3}^{\exp}\right)$

(d)
![](images/02b8ff5619318372e4fbe97374f0a6c6839c20999f2db13faa0a3defecea4411.jpg)

$\operatorname{Im}(\rho_{3}^{\exp})$
![](images/a239f91299554b3bf003765a04181e3949be8c9f4e876a174eb2449538d84783.jpg)

$$
\mathrm{Re} (\rho_ {2} ^ {\mathrm{theo}})
$$

![](images/eef2333d92ae79854737d94917105a033dde6cd18e735cb2265339c6e1db9432.jpg)
(f)

![](images/7f50ea9ff950736cda9c8b1d67fa5ae1d8f85cc2ceeb11ae81ea86d72f0d77ef.jpg)

![](images/62681f0b45dd2ca9596a0eeeea1697c86a22231949ac823e1a8a58fb271375ca.jpg)

![](images/d86d1305c0837efb559d682f40f7cd6b42a1918e28dce5bf8ac32f93eb37f09f.jpg)

$$
\mathrm{Re} \left(\rho_ {3} ^ {\text { theo }}\right)
$$

$\operatorname{Im}(\rho_{3}^{\text{theo}})$
![](images/1c1ab7ec6ad447ccae81be03bc37080a630b9765f0483c4eaa8e287f0fe08369.jpg)

![](images/71545ee5e1542ebb56e6e95411458d5e6ec1a98650d88f7ae399f14b5f4a8ae2.jpg)
FIG. 4. Eigenstates of the non-Hermitian Hamiltonian at the Dirac EP, where $k_{1} = 0$ and $k_{2} = 1$ . (a),(c),(e) $\rho_1^{\mathrm{exp}}$ , $\rho_2^{\mathrm{exp}}$ , and $\rho_3^{\mathrm{exp}}$ are the measured density matrices of three eigenstates (labeled by 1, 2, and 3) obtained by quantum state tomography. (b),(d),(f) $\rho_1^{\mathrm{theo}}$ , $\rho_2^{\mathrm{theo}}$ , and $\rho_3^{\mathrm{theo}}$ are the density matrices of theoretically predicted eigenstates.

of the eigenvalues degenerate at the Dirac EP. They also illustrate the linear relationship between the eigenvalue and the parameter when the parameter is very close to the Dirac EP. When $k_{2}=1$ , the eigenvalues have the forms $E_{1}=3+2|k_{1}|$ , $E_{2}=3-2|k_{1}|$ , and $E_{3}=0$ , showing a linear response to the parameter [Fig. 3(a)]. When $k_{1}$ is fixed at 0, the eigenvalues vary with $k_{2}$ as $E_{1}=\max[(3+\sqrt{17-8k_{2}^{2}})/2,3]$ , $E_{2}=\min[(3+\sqrt{17-8k_{2}^{2}})/2,3]$ , and $E_{3}=(3-\sqrt{17-8k_{2}^{2}})/2$ . Near the Dirac EP with $k_{2}<1$ , $E_{1}$ exhibits a linear variation with $\Delta k_{2}=k_{2}-1$ as $E_{1}=3-4\Delta k_{2}/3+O[(\Delta k_{2})^{2}]$ , while the value of $E_{2}$ remains constant at 3. For the case of $k_{2}>1$ , $E_{1}=3$ , and $E_{2}=3-4\Delta k_{2}/3+O[(\Delta k_{2})^{2}]$ . It can be observed in Fig. 3(c) that when $\Delta k_{2}$ is small, the variation of $E_{1}$ with $k_{2}$ conforms well to the linear relationship. However, when $\Delta k_{2}$ is large, influenced by higher-order terms of $\Delta k_{2}$ , the variation of $E_{1}$ with $k_{2}$ gradually deviates from linearity. The eigenvalues are linear in two perpendicular directions $k_{1}$ and $k_{2}$ when the parameter is very close to the Dirac EP. These results are consistent with the dispersion relation shown in Fig. 2(b). Thus, the energy spectrum near the Dirac EP forms a structure akin to a Dirac cone.

The degeneracy of two of the eigenstates can distinguish the Dirac EP from the Hermitian degeneracy. The eigenstates of the non-Hermitian Hamiltonian at the Dirac EP are obtained by quantum state tomography, as shown in Fig. 4. In order to avoid a directly obtained state that is unphysical, the maximum likelihood estimation method is used to obtain the density matrix. The overlap between two states $\rho_{i}$ and $\rho_{j}$ is characterized by the fidelity, defined as $[\mathrm{Tr}(\sqrt{\sqrt{\rho_i}\rho_j\sqrt{\rho_i}})]^2$ . The obtained fidelities between the experimental eigenstates $\rho_{i}^{\mathrm{exp}}$ corresponding to $E_{i}$ and the theoretical ones $\rho_{i}^{\mathrm{theo}}$ are 0.99(2), 0.99(2), and 0.99(3), yielding the high-fidelity reconstructions of the eigenstates at the Dirac EP. At the Dirac EP, the experimental results of the fidelity $F_{ij}$ between the experimental eigenstates $\rho_{i}^{\mathrm{exp}}$ and $\rho_{j}^{\mathrm{exp}}$ corresponding to the eigenvalues $E_{i}$ and $E_{j}$ are $F_{12} = 0.98(3)$ , $F_{13} = 0.32(6)$ , and $F_{23} = 0.33(5)$ . These results demonstrate the degeneracy of the eigenstates $\rho_{1}$ and $\rho_{2}$ . Both the degeneracy of the two eigenstates and the corresponding eigenvalues confirm that such a degeneracy is a Dirac EP rather than a Hermitian degeneracy.

Discussion—Our Letter investigates the Dirac EP, a novel type of energy-level degeneracies different from typical EPs. Around the Dirac EP, the eigenvalues are observed to be real. This result reveals that the PT operator always shares common eigenstates with the non-Hermitian Hamiltonian, and spontaneous PT symmetry breaking does not occur when passing through the Dirac EP. The dispersion around the Dirac EP exhibits a conical shape. The measured degeneracy of the eigenstates further confirms that the Dirac EP is entirely different from Hermitian degeneracy. The inner product of the eigenstates satisfies a quadratic relationship as the parameters $k_{1}$ and $k_{2}$ approach the Dirac EP, which is distinctly different from the linear relationship exhibited in the vicinity of a typical EP. (See Supplemental Material [30], Sec. 2, for the details.) The occurrence of the Dirac EP is not limited to the model parameters we used, which indicates that similar non-Hermitian degeneracy can potentially be observed in a wider range of systems. (See Supplemental Material [30], Sec. 2, for the details.)

The discovery of the novel type of EP not only provides an unprecedented mechanism for dynamics in non-Hermitian systems but also holds potential application value in the fields of quantum information processing and topological physics. The Dirac EPs are expected to provide new mechanisms of mode switching that are beyond previous frameworks for typical EPs $[38]$ . Both chiral and nonchiral mode switching can be realized by passing through the Dirac EP in different directions. (See Supplemental Material $[30]$ , Sec. 4, for the details.) This result unveils a novel mechanism for dynamics in non-Hermitian systems, distinct from the gain and loss arising from the imaginary parts of eigenvalues near typical EPs $[38]$ , which warrants deeper investigation in the future.

The Dirac EPs exhibit advantages compared to typical EPs in realizing mode switching. Due to its robustness against variations in the pathway, mode switching offers a robust fashion for controlling quantum states, thereby holding potential applications in quantum information processing. Realizing mode switching by typical EPs suffers from probability loss due to the imaginary parts of eigenvalues, which makes its application in quantum controls very challenging, since the successful postselection probability is almost zero. The real eigenvalues near Dirac EP enable the mode switching to overcome the low efficiency in obtaining the target final state. (See Supplemental Material [30], Sec. 4, for the details.) Thus, the investigation of the Dirac EP provides a novel method of quantum-state control and potentially unlocks applications in quantum information processing. These results also hold potential in mode switching devices in optical and mechanical systems.

An application of Dirac EP in the realm of topological physics lies in its facilitation of the experimental investigation of complex geometric phases $[39–41]$ . Such a complex geometric phase is the generalization of geometric phase, a fundamental concept of topological physics, to the realm of non-Hermitian systems. Unlike recent work that specifically chooses a trajectory with a real spectrum in parameter space $[42]$ , utilizing Dirac EP offers an alternative approach to achieve adiabatic evolution in non-Hermitian systems, circumventing the nonadiabatic transitions associated with typical EPs. This enables the investigation of complex geometric phases. (See Supplemental Material $[30]$ , Sec. 3, for the details.)

Acknowledgments—This work was supported by the Innovation Program for Quantum Science and

Technology (Grant No. 2021ZD0302200), the National Natural Science Foundation of China (Grants No. T2388102, No. 12174373, and No. 12261160569), the Chinese Academy of Sciences (Grants No. XDC07000000 and No. GJJSTD20200001), the Hefei Comprehensive National Science Center, and the Fundamental Research Funds for the Central Universities (Grant No. WK3540000013).

Y. W., D. Z., and Y. W. contributed equally to this work.

[1] T. Kato, Perturbation Theory for Linear Operators (Springer, New York, 1966).

[2] J. Yang, C. Qian, X. Xie et al., Diabolical points in coupled active cavities with quantum emitters, Light Sci. Appl. 9, 6 (2020).

[3] L. Polimeno, G. Lerario, M. De Giorgi et al., Tuning of the Berry curvature in 2D perovskite polaritons, Nat. Nanotechnol. 16, 1349 (2021).

[4] D. R. Yarkony, Diabolical conical intersections, Rev. Mod. Phys. 68, 985 (1996).

[5] C. M. Bender and S. Boettcher, Real spectra in non-Hermitian Hamiltonians having PT symmetry, Phys. Rev. Lett. 80, 5243 (1998).

[6] S. Yao and Z. Wang, Edge states and topological invariants of non-Hermitian systems, Phys. Rev. Lett. 121, 086803 (2018).

[7] K. Kawabata, K. Shiozaki, M. Ueda, and M. Sato, Symmetry and topology in non-Hermitian physics, Phys. Rev. X 9, 041015 (2019).

[8] Z. Gong, Y. Ashida, K. Kawabata, K. Takasan, S. Higashikawa, and M. Ueda, Topological phases of non-Hermitian systems, Phys. Rev. X 8, 031079 (2018).

[9] H. Zhou, P. Chao, Y. Yoon, C. W. Hsu, K. A. Nelson, L. Fu, J. D. Joannopoulos, and M. Soljačić, and B. Zhen, Observation of bulk Fermi arc and polarization half charge from paired exceptional points, Science 359, 1009 (2018).

[10] L. Xiao, T. Deng, K. Wang, G. Zhu, Z. Wang, W. Yi, and P. Xue, Non-Hermitian bulk-boundary correspondence in quantum dynamics, Nat. Phys. 16, 761 (2020).

[11] A. U. Hassan, B. Zhen, M. Soljačić, M. Khajavikhan, and D. N. Christodoulides, Dynamically encircling exceptional points: Exact evolution and polarization state conversion, Phys. Rev. Lett. 118, 093002 (2017).

[12] E. J. Bergholtz, J. C. Budich, and F. K. Kunst, Exceptional topology of non-Hermitian systems, Rev. Mod. Phys. 93, 015005 (2021).

[13] H. Hu and E. Zhao, Knots and non-Hermitian Bloch bands, Phys. Rev. Lett. 126, 010401 (2021).

[14] H. Shen, B. Zhen, and L. Fu, Topological band theory for non-Hermitian Hamiltonians, Phys. Rev. Lett. 120, 146402 (2018).

[15] K. Kawabata, T. Bessho, and M. Sato, Classification of exceptional points and non-Hermitian topological semimetals, Phys. Rev. Lett. 123, 066405 (2019).

[16] M. Stalhammar and E. J. Bergholtz, Classification of exceptional nodal topologies protected by PT symmetry, Phys. Rev. B 104, L201104 (2021).

[17] K. Ding, C. Fang, and G. Ma, Non-Hermitian topology and exceptional-point geometries, Nat. Rev. Phys. 4, 745 (2022).

[18] L. Feng, Z. J. Wong, R.-M. Ma, Y. Wang, and X. Zhang, Single mode laser by parity-time symmetry breaking, Science 346, 972 (2014).

[19] H. Hodaei, M.-A. Miri, M. Heinrich, D. N. Christodoulides, and M. Khajavikhan, Parity–time-symmetric microring lasers, Science 346, 975 (2014).

[20] L. Chang, X. Jiang, S. Hua, C. Yang, J. Wen, L. Jiang, G. Li, G. Wang, and M. Xiao, Parity–time symmetry and variable optical isolation in active-passive-coupled microresonators, Nat. Photonics 8, 524 (2014).

[21] B. Peng, S. K. Özdemir, F. Lei, F. Monifi, M. Gianfreda, G. L. Long, S. Fan, F. Nori, C. M. Bender, and L. Yang, Parity-time symmetric whispering-gallery microcavities, Nat. Phys. 10, 394 (2014).

[22] Z. Lin, H. Ramezani, T. Eichelkraut, T. Kottos, H. Cao, and D. N. Christodoulides, Unidirectional invisibility induced by PT-symmetric periodic structures, Phys. Rev. Lett. 106, 213901 (2011).

[23] H. Xu, D. Mason, L. Jiang, and J. G. E. Harris, Topological energy transfer in an optomechanical system with exceptional points, Nature (London) 537, 80 (2016).

[24] W. Chen, S. K. Özdemir, G. Zhao, J. Wiersig, and L. Yang, Exceptional points enhance sensing in an optical microcavity, Nature (London) 548, 192 (2017).

[25] H. Hodaei, A. U. Hassan, S. Wittek, H. Garcia-Gracia, R. El-Ganainy, D. N. Christodoulides, and M. Khajavikhan, Enhanced sensitivity at higher-order exceptional points, Nature (London) 548, 187 (2017).

[26] J.-H. Park, A. Ndao, W. Cai, L. Hsu, A. Kodigala, T. Lepetit, Y.-H. Lo, and Kanté, Symmetry-breaking-induced plasmonic exceptional points and nanoscale sensing, Nat. Phys. 16, 462 (2020).

[27] W. Mao, Z. Fu, Y. Li, F. Li, and L. Yang, Exceptional–point-enhanced phase sensing, Sci. Adv. 10, eadl5037 (2024).

[28] J. H. D. Rivero, L. Feng, and L. Ge, Imaginary gauge transformation in momentum space and Dirac exceptional point, Phys. Rev. Lett. 129, 243901 (2022).

[29] J. H. D. Rivero, L. Feng, and L. Ge, Analysis of Dirac exceptional points and their isospectral Hermitian counterparts, Phys. Rev. B 107, 104106 (2023).

[30] See Supplemental Material at http://link.aps.org/supplemental/10.1103/PhysRevLett.134.153601 for details of properties of Dirac EP, experimental setup, methods, and pulse sequences, which includes Refs. [31–33].

[31] R. Uzdin, A. Mailybaev, and N. Moiseyev, On the observability and asymmetry of adiabatic state flips generated by exceptional points, J. Phys. A 44, 435302 (2011).

[32] X.-L. Zhang, S. Wang, B. Hou, and C. T. Chan, Dynamically encircling exceptional points: In situ control of encircling loops and the role of the starting point, Phys. Rev. X 8, 021066 (2018).

[33] Q. Liu, S. Li, B. Wang, S. Ke, C. Qin, K. Wang, W. Liu, D. Gao, P. Berini, and P. Lu, Efficient mode transfer on a compact silicon chip by encircling moving exceptional points, Phys. Rev. Lett. 124, 153903 (2020).

[34] U. Günther and B. F. Samsonov, The Naimark-dilated PT-symmetric brachistochrone, Phys. Rev. Lett. 101, 230404 (2008).

[35] Y. Wu, W. Liu, J. Geng, X. Song, X. Ye, C.-K. Duan, X. Rong, and J. Du, Observation of parity-time symmetry breaking in a single-spin system, Science 364, 878 (2019).

[36] W. Liu, Y. Wu, C.-K. Duan, X. Rong, and J. Du, Dynamically encircling an exceptional point in a real quantum system, Phys. Rev. Lett. 126, 170506 (2021).

[37] Y. Wu, Y. Wang, X. Ye, W. Liu, Z. Niu, C.-K. Duan, Y. Wang, X. Rong, and J. Du, Third-order exceptional line in a nitrogen-vacancy spin system, Nat. Nanotechnol. 19, 160 (2024).

[38] P. Kumar, Y. Gefen, and K. Snizhko, General theory of slow non-Hermitian evolution, arXiv:2502.04214.

[39] J. Gong and Q. Wang, Geometric phase in PT-symmetric quantum mechanics, Phys. Rev. A 82, 012103 (2010).

[40] X.-D. Cui and Y. Zheng, Geometric phases in non-Hermitian quantum mechanics, Phys. Rev. A 86, 064104 (2012).

[41] Q. Zhang and B. Wu, Non-Hermitian quantum systems and their geometric phases, Phys. Rev. A 99, 032121 (2019).

[42] I. I. Arkhipov, F. Minganti, A. Miranowicz, S. K. Özdemir, and F. Nori, Restoring adiabatic state transfer in time-modulated non-Hermitian systems, Phys. Rev. Lett. 133, 113802 (2024).
