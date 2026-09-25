# Extreme Wave Manipulation via Non-Hermitian Metagratings on Degenerated States

Xinsheng Fang, $^{1,2,3,\S}$ Nengyin Wang, $^{2,\S}$ Wenwei Wu, $^{1,3,*}$ Weibo Wang, $^{1,3}$ Xuewen Yin, $^{1,3}$ Xu Wang, $^{2,\dagger}$ and Yong Li $^{2,\ddagger}$

$^{1}$ Department of Basic Science and Technology Research, Taihu Laboratory of Deepsea Technological Science, Wuxi 214082, China

$^{2}$ Institute of Acoustics, School of Physics Science and Engineering, Tongji University, Shanghai 200092, China $^{3}$ National Key Laboratory on Ship Vibration and Noise, China Ship Scientific Research Center, Wuxi 214082, China

(Received 7 February 2023; revised 27 March 2023; accepted 6 April 2023; published 1 May 2023)

Metagratings offer promising opportunities for highly efficient yet anomalous wave-diffraction control characterized by scattering matrices. In this work, those degenerated states of scattering matrices, arising from elaborately induced non-Hermiticity and encoding unique scattering behaviors, are of interest. In particular, we theoretically and experimentally investigate the exotic degeneracies, exceptional points (EPs) and diabolic points (DPs) in such scattering systems. We show the distinct response strength induced by EP and DP. The extremely asymmetrical reflection occurs on the EP state in a metagrating, which exhibits a stronger reaction to disturbances and offers opportunities for microvariable detection and external perturbation monitoring. As a contrast, a stable dual-angle absorber is proposed on the DP state, which is almost unaffected by external perturbations. Our work may pave the way for extreme wave manipulation via non-Hermitian metagratings on degenerated states.

DOI: 10.1103/PhysRevApplied.19.054003

## I. INTRODUCTION

The evolution of eigenstates leads to a variety of exciting applications both in natural science and engineering $[1,2]$ . Resultant singularities associated with peculiar physical effects have attracted significant attention from researchers. Among them, exceptional point (EP) is conspicuous for its strong response to external perturbation $[3–5]$ , and it occurs while the eigenvalues and eigenvectors in Hamiltonian (H matrix) or scattering matrix (S matrix) simultaneously coalesce $[6–11]$ . Prompted by this concept, researchers have proposed various opportunities for wave control, such as laser absorbers $[10]$ , coherent perfect wave absorption $[12]$ , and asymmetric reflection and transmission $[13–16]$ . In these works, accompanied by the change of system parameters, eigenvalues would become coupled and decoupled, accounting for the diverse spectacles. However, according to our previous research, the generalized nth-order EP requires the H matrix or S matrix to be similar to the Jordan-block canonical form $[17,18]$ $(n \times n)$ . All the eigenvectors should be linearly correlated, otherwise leading to another, more conventional degenerate state, diabolic point (DP) [19–21]. As a less demanding singularity, eigenvalues merge around DP accompanied by n linearly independent eigenvectors [12,19,20,22–24]. Due to the increasing sensitivity from EP or DP, these singularities play an essential role in sensing applications [3,5,17,23,25]. For example, light-matter interaction is leveraged for constructing high-quality optical microcavities, achieving accurate monitoring for nanoscale objects in a confined volume [3,26]. Nevertheless, there is a lack of research on acoustics, and the discrepant characteristics of EP and DP in acoustic detection remain mysterious.

In the last decade, the two-dimensional (2D) category of functional acoustic microstructures, namely metasurfaces $[27]$ , have offered efficient opportunities to exploit the physical predominance of non-Hermitian systems $[16,17,25,28,29]$ . The traditional acoustic metasurface is intrinsically composed of various subunits, each individually designed so that the locality fundamentally limits the efficiency of anomalous wave manipulation. Notably, recent advances in nonlocality have revealed its crucial role in wave manipulation and unlocked the strength of coupling among subunits $[30,31]$ . Pivoted on this concept, researchers have proposed the metagrating $[32–36]$ , in which the subunits were randomly arranged, and the phase gradient was no longer dominant in wave manipulation.

In this paper, we show that the nonlocal metagrating $[32–36]$ could be leveraged as an ideal platform for theoretically and experimentally exploiting the scattering fields around EP and DP. Empowered by this, this paper intends to demonstrate the evolution of eigenvalues and eigenvectors associated with these two singularities, to quantify the diverse response strengths to external perturbations. To be convenient but without loss of generality, the second-order scattering matrices $(2 \times 2)$ around singularities are taken as examples.

## II. RESULTS AND DISCUSSIONS

## A. Multichannel scattering system based on the metagrating

Following the significance of EP and DP, the scattering matrix on second-order singularities are ruled as $[17,18]$ ,

$$
S _ {\mathrm{EP}} = \left( \begin{array}{c c} E _ {0} & 1 \\ 0 & E _ {0} \end{array} \right),   S _ {\mathrm{DP}} = \left( \begin{array}{c c} E _ {0} & 0 \\ 0 & E _ {0} \end{array} \right).\tag{1}
$$

Here $S_{\mathrm{EP}} \sim J^{(2)}(E_0)$ so that the eigenvalues and eigenvectors coalesce simultaneously, while the identical eigenvalues in $S_{DP}$ are accompanied by the eigenvectors perpendicular to each other. To intuitively show the disturbance in the sound field after introducing the perturbation, both degenerate eigenvalues set $E_0 = 0$ in Eq. (1). The singularities from non-Hermitian scattering matrices [19] are demonstrated in the two-channel metagratings filled with sound-absorbing materials, which is a two-port scattering system creating anomalous reflections at $\pm45^{\circ}$ , as shown in Fig. 1. Without loss of generality, three rectangular-groove-type subunits arranged periodically are adopted to construct the nonlocal metagrating prototype [17,36]. It is apparent that the metagrating acts as an asymmetric unidirectional retroreflector around EP or an all-angle absorber on DP.

According to the generalized diffraction law in the grating, the nth-order reflected wave and incidence should be satisfied $k_{0}(\sin\theta_{r}-\sin\theta_{i})=nG$ . $k_{0},\theta_{r}$ , and $\theta_{i}$ are the wave number in free space, reflection, and incident angles, respectively. $G=2\pi/D$ is the reciprocal lattice vector. We notice that the grating period D determines the number of free-space propagation modes. For the two-channel metagrating, $G=\sqrt{2}k_{0}$ , and $D=\sqrt{2}\lambda/2$ . $\theta_{i}$ and $\theta_{r}$ are set to be $\pm45^{\circ}$ . Then the state vectors of reflection and incident fields could be represented as

$$
\binom{p _ {r} ^ {R}}{p _ {r} ^ {L}} = S \binom{p _ {i} ^ {L}}{p _ {i} ^ {R}} = \left( \begin{array}{c c} r _ {0} ^ {L} & r _ {+ 1} ^ {R} \\ r _ {- 1} ^ {L} & r _ {0} ^ {R} \end{array} \right) \binom{p _ {i} ^ {L}}{p _ {i} ^ {R}}.\tag{2}
$$

p indicates the complex pressure amplitude, where the superscript $L(R)$ represents the left (right) side of the two-channel metagrating. The subscript $r(i)$ marks the output (input) sound waves, i.e., the reflection or incidence.

![](images/1d644223b3e99065b6159b12371fccda8314b2ca0e887f7876c4a18b504f7a70.jpg)
FIG. 1. Diagram of metagratings at EP (a) and DP (b). D is the period of metagrating. $t_{s}$ , $l_{s}$ , and $x_{s}$ refer to the width, depth, and spacing of the sth groove. One or two tracks are filled with sound-absorbing materials in EP or DP scenarios in each period, as shown in white-dashed rectangles. Yellow and green rays represent the incident wave and strong retroreflection, respectively. Black arrows show the propagation waves, while the dashed arrows represent the suppressed reflected waves. p denotes the complex pressure amplitude, the superscript represents the left (L) or right (R) channel, while the subscript marks the incidence (i) or reflection (r).

The scattering matrix $(S)$ relating these vectors is $S = (r_{0}^{L}, r_{+1}^{R}; r_{-1}^{L}, r_{0}^{R})$ , while the elements represent the reflection coefficients of each mode, whose superscripts refer to the direction of incidence, i.e., L (R) indicates to the left (right) incidences with $\theta_{i} = 45^{\circ} (-45^{\circ})$ . The subscripts denote the reflection orders. Notably, the clinodiagonal elements in $S(r_{+1}^{R}, r_{-1}^{L})$ correspond to retroreflection in two channels, while $r_{0}^{L}$ and $r_{0}^{R}$ represent the mirror reflection.

As Eq. (1) reveals, EP is achieved when the $2 \times 2$ scattering matrix coalesces, accompanied by two eigenvalues degenerating and the initially orthogonal eigenvectors converting to parallel. At the same time, the latter maintains orthogonal on DP. Coupled-mode theory considering complex velocities is utilized to theoretically calculate the scattering coefficients in parameter space. Note that the global optimization, genetic algorithm (GA), is a powerful tool, while Eq. (1) is logically defined as the optimization objective in the design. We assume the operating frequency to be 3430 Hz, then the geometrical parameters from two metagratings could be rigorously solved. Relevant details about coupled-mode theory and the geometrical parameters are shown in Appendix A.

## B. Eigenvalues' variation around EP and DP

We start with the eigenvalues' behavior around the singularities. The $S$ matrix on EP associated with the perturbation term should be written as [3,23]

$$
S (\delta) _ {\mathrm{EP}} = S _ {0 \mathrm{EP}} + \delta S _ {1} = \left( \begin{array}{c c} E _ {0} & A _ {0} \\ 0 & E _ {0} \end{array} \right) + \delta \left( \begin{array}{c c} E _ {1} & A _ {1} \\ B _ {1} & E _ {2} \end{array} \right),\tag{3}
$$

$S_{0EP}$ represents the accurate EP matrix $[S_{EP}$ in Eq. (1)]. $\delta S_{1} = \delta \begin{pmatrix} E_{1} & A_{1} \\ B_{1} & E_{2} \end{pmatrix}$ reveals a stochastic matrix referring to the environmental perturbation introduced into the metagrating. In other words, $\delta S_{1}$ represents the deviation in reflection coefficients accounting for complex sound velocity, complex frequency, or even the error in sample sizes, etc.

After simplification, the gap between eigenvalues approaching EP in parameter space characterizes as

$$
\begin{array}{c} \Delta E _ {\mathrm{EP}} = \sqrt {\delta} \sqrt {\delta (E _ {1} - E _ {2}) ^ {2} + 4 A _ {0} B _ {1} + 4 \delta A _ {1} B _ {1}} \\ = \sqrt {\delta} \sqrt {4 A _ {0} B _ {1}} + o (\delta), \end{array}\tag{4}
$$

where $o(\delta)$ is the higher-order terms of $\delta$ , which is negligible when the perturbation is imperceptible. Similarly, the general S matrix approaching DP accompanied by disturbance could write as [3,23]

$$
S (\delta) _ {\mathrm{DP}} = S _ {0 \mathrm{DP}} + \delta S _ {1} = \left( \begin{array}{c c} E _ {0} & 0 \\ 0 & E _ {0} \end{array} \right) + \delta \left( \begin{array}{c c} E _ {1} & A _ {1} \\ B _ {1} & E _ {2} \end{array} \right).\tag{5}
$$

For comparison, the elements in $S_{1}$ here are the same as in the EP case. Then we could obtain the gap between eigenvalues near the DP state as

$$
\Delta E _ {\mathrm{DP}} = \delta \sqrt {(E _ {1} - E _ {2}) ^ {2} + 4 A _ {1} B _ {1}}.\tag{6}
$$

Comparing Eq. (4) with Eq. (6), the linear eigenvalue deviation around DP relates to the perturbation. While in the general matrix around EP, since $A_{0} \neq 0$ , the departure of eigenvalues is exponentially related to $\delta$ . As a multiple of $\sqrt{\delta}$ , $A_{0}$ determines the separation distance between eigenvalues, and it is the crucial element in the scattering system to characterize the response strength to an external perturbation. Note only if $B_{1} \neq 0$ in $S_{1}$ that the more extensive splitting around EP would exist [23]. Otherwise, $\Delta E_{EP} = \Delta E_{DP}$ , which induces by the turmoil associated with $B_{1} = 0$ .

![](images/43e779a08804c2630ae832d84449601117ac5490f03d8b1a3db25f645238d20c.jpg)

In order to elucidate the utterly different trend around two singularities intuitively, the Riemann surface $[11,37]$ is calculated, as shown in Fig. 2. Here, we take the perturbation in complex sound velocity as an example to provide the eigenvalues' variation, which is associated with the real part of the normalized complex velocity $c_{s}^{r}$ and the imaginary parts $c_{s}^{i}$ . The subscript s represents the serial number of grooves. The white domain in the Reimann surface on the z=0 plane refers to the merging state of EP [Fig. 2(a)] and DP [Fig. 2(b)], which is marked on the intersection of green lines (Im( $\lambda$ )=0). It is apparent that the eigenvalues travel on a nonlinear path around EP, while the two eigenvalue surfaces cross each other, accompanied by a flat and smooth variation around DP. Indeed, after the interaction on EP, the eigenvalues could not be distinguished immediately, which is different from DP.

To further contrast the above splitting tendency in eigenvalues, i.e., the response sensitivity around singularities [25], we calculate the real-part deviation in Fig. 3. The abscissa $\delta = c_s^i - c_s^i|_{\mathrm{EP(DP)}}$ represents disturbance induced into the lossy groove. $c_s^i|_{\mathrm{EP(DP)}}$ is the original imaginary part of sound velocity in the $n$ th track on EP (DP). In Fig. 3, the red triangles refer to the eigenvalues' difference solved analytically from the metagrating on EP. In contrast, the fitted curve (red solid line) corroborates that the sensitivity follows a square-root dependence on the $\delta$ ( $|\mathrm{Re}(\Delta E)| \propto \sqrt{\delta}$ ). For comparison, the eigenvalues' division tendency around DP is marked by blue boxes. It fits by the solid blue line, which satisfies $|\mathrm{Re}(\Delta E)| \propto \delta$ .

![](images/6aff2e9f8e6926b8e28defff4ffa74e27296898ab4f30e541d44a8cbe0ab0421.jpg)
FIG. 2. Riemann surfaces around EP (a) and DP (b) for two metagratings. The x (y) axis refers to the real (imaginary) part of the normalized velocity, while the z axis and color bar represent the range of eigenvalues. Bright green lines refer to the contours of $\mathrm{Im}(\lambda)=0$ . The merging states are marked at the intersection of the contours on the z=0 plane.

![](images/d857149e43e5a2b75992af9bc4a2d457569c29f82d2182fcc0d9857b63eb5623.jpg)
FIG. 3. Response sensitivity of the designed metagratings around EP (red data sets) and DP (blue data sets). $|\mathrm{Re}(\Delta E)|$ on the $y$ axis represents the splitting tendency of the eigenvalues' real parts varying with $\delta = c_s^i - c_s^i|_{\mathrm{EP(DP)}}$ .

The above tendency is consistent with the analytical solution in Eqs. (4) and (6). From Fig. 3, when $\delta$ is small enough (for example, on the $e^{-4}$ scale), the division of eigenvalues away from EP is more aggressive than that from DP. This muscular response strength is precisely owing to element $A_{0}$ in Eq. (4).

## C. Eigenvectors' variation around EP and DP

In addition to the evolutionary trajectories of Riemann surfaces, the general mathematical condition on EP [17] and DP [19,20] [Eq. (1)] indicates that the eigenvectors ought to possess different variation tracks accompanied by the disturbance. This paper will then elicit the evolution of eigenvectors around the two singularities. Phase rigidity [11,18,38] has been defined in previous research to describe the deviation of eigenfunctions around singularities. Furthermore, this variable is adopted here to illustrate the evolution of eigenvectors. For convenience, we first define the normalized biorthogonal left eigenvectors $\left\langle\tilde{\phi}^{L}\right|$ and the right one $|\tilde{\phi}^{R}\rangle$ based on Dirac symbol [39], as

$$
\left<   \tilde {\phi} ^ {L} \left. \right| = \frac {\left<   \phi^ {L} \left. \right|}{\sqrt {\left<   \phi^ {L} | \phi^ {R} \right > }}, | \tilde {\phi} ^ {R} \rangle = \frac {| \phi^ {R} \rangle}{\sqrt {\left<   \phi^ {L} | \phi^ {R} \right > }}.\tag{7}
$$

Whenever the eigenvalues are determined, the corresponding eigenvectors could be obtained as $\left\langle\phi^{L}\right|$ and $|\phi^{R}\rangle$ , where $\left\langle\phi^{L}\right|$ is the conjugate vector of $|\phi^{R}\rangle$ and $\left\langle\tilde{\phi}_{i}^{L}|\tilde{\phi}_{j}^{R}\right\rangle=\delta_{ij}$ . Here i and j are the serial numbers of eigenvectors. Then the phase rigidity could be defined as [11,18]

$$
r _ {j} = \frac {\left\langle \tilde {\phi} _ {j} ^ {L} | \tilde {\phi} _ {j} ^ {R} \right\rangle}{\left\langle \tilde {\phi} _ {j} ^ {R} | \tilde {\phi} _ {j} ^ {R} \right\rangle} = \frac {1}{\left\langle \tilde {\phi} _ {j} ^ {R} | \tilde {\phi} _ {j} ^ {R} \right\rangle}.
$$

![](images/222e0cbb78539161011357a8ac88010e0f7171eca29d4893e3ee9135dd1c2b73.jpg)

(8)

Figure 4 describes the variation of $r_{j}$ accompanied by $c_{s}^{i}$ . It is apparent that the phase rigidity evolves dramatically and vanishes near EP, revealing the depravity and division process of eigenvectors [18,38] [Fig. 4(a)]. For comparison, the steady state of $r_{j}$ remains unchanged around DP [Fig. 4(b)], accompanied by the linearly independent eigenvectors. Considering the components of scattering matrices on EP and DP, $A_{0}$ in Eq. (4) is also the critical variable determining whether the eigenvectors degenerate.

The above discussion explains why the scattering systems at two singularities own different sensitivities to perturbations. Of note, the parameter dependence around EP could apply to the high sensitivity detection. At the same time, the insensitivity to perturbations behind DP could be utilized to achieve stable wave manipulation, such as dual-angle absorption, as shown in the following.

## D. Numerical simulation and experimental demonstration for EP and DP

To demonstrate the parameter dependence around EP and DP in theory, numerical simulation and experimental

![](images/a1fe51df3aab050e4ca1d6f090b35d32064e1251837cf2751ee44fbbafe4f5c4.jpg)
FIG. 4. Phase rigidity profile calculated theoretically around EP and DP, where the ordinates are managed with the logarithm. (a) The eigenvectors' evaluation from the EP metagrating varies with $c_2^i$ . (b) The trajectory of eigenvectors calculated from the DP metagrating as functions of $c_1^i$ .

![](images/4097594ae064ea2d760e00576de30afeb818af3b7ab6bd435889f1de595354e8.jpg)

![](images/1f3e922810e36f14e312b069f84749862c229bc62b77513563da89034d6a1425.jpg)

![](images/38d3c8141a7c36339a93085176eb616e2b2618f4e268b639fd6f8a51700efcf0.jpg)

![](images/710b34d3b10626d78606adf9133faafa2f84b7744da3d244c8411e5175843c34.jpg)

![](images/0af0fa654494b83670b86681ab6dea58e8b2da339c765d1e7c577e5085f235fb.jpg)

![](images/048e80ac04818ecb1f6de7ef82f91de7dd1cab7f7212c483b4db708222ee65bb.jpg)

![](images/9f1d0584cba2293cc4be37ba374d72a929a5536a99fd4cfa311cd71ce8908db8.jpg)

![](images/5abf0697c4e75bbe89f533427cf3d8c7fa7056f54af58b847aa3661d7160320c.jpg)
FIG. 5. Numerical simulation and experimental demonstration around EP (top two lines) and DP (bottom two lines). (a)–(d) The reflected field without disturbance introduced in the groove height. (e)–(h) The reflected field when a 2-mm gasket adds into the first (second) groove in the EP (DP) scenario. The scanning area is marked by the solid-line box in the simulation, while the sweep graph displays outside the semicircular field. White arrows indicate the propagation of incidence and reflection waves, and the less-visible ones refer to the near-perfect absorption (no reflection). Lines and dots in (i)–(l) represent the simulation and experiment data, respectively, extracted along the scanning area's central lines. The black-dotted lines in (i)–(l) depict simulation data on the red-dashed lines in (a)–(d). $\lambda$ in lateral axes is the wavelength of the operating frequency. The operating frequency in the simulation is 3430 Hz.

demonstration are then performed. To precisely regulate the perturbation variables, the slight change in the groove's height is taken as $\delta$ , then the disturbance in scattering fields could be calculated and demonstrated intuitively. Numerical simulation is performed with the finite-element analysis in COMSOL Multiphysics. The reflected fields in the simulation and experiment around EP and DP are shown in Fig. 5.

In the EP scenario, the reflected fields under the left and right incidence in the simulation are shown in Figs. 5(a)

and 5(b), respectively. The measured fields outside the semicircular field are normalized by the incident pressure. To verify the response strength of EP status, a 2-mm gasket is added into the first groove to tune the height. The details of the experimental setup and the preparation of measured samples are described in Appendix B. Relevant results are shown in Figs. 5(e) and 5(f). Simulated and measured data extracted from the scanning areas are consistent in both channels, as shown in Figs. 5(i) and 5(j). Through the comparison between solid and dashed lines in (i) and (j), the stray speckle would appear in the original reflected field after the groove depth is reduced by 2 mm ( $\lambda/50$ ). The coincidence between simulation and experiment also exists in the DP situation, as shown in the bottom two lines in Fig. 5. Differently, when the groove depth tunes with the same size (2 mm), the scattering field remains unchanged, as demonstrated in Figs. 5(k) and 5(l). In other words, the efficient dual-angle absorption in this metagrating is not affected by the external perturbation. To demonstrate the weak response strength induced by DP, near-perfect absorption under various incident angles is further investigated, indicating a consistently quasiperfect absorption around the designed incident angles ( $\theta_{i} = \pm45^{\circ}$ ), as shown in Appendix C. These curious phenomena stem from the nonlocal effect manifested by the nonuniform energy flow around the grating surface at EP and DP, as shown in Appendix D.

## III. CONCLUSION

According to the general condition for second-order EP and DP in S matrix, we theoretically investigate the fantastic wave behavior on the two singularities with metagratings. The diverse response strengths around EP and DP are demonstrated based on the variation of eigenvalues and eigenvectors. In addition, through the subtle change in the size of metaunits, we experimentally investigate the evolution of scattering systems in the face of perturbations. This paper provides efficient approaches toward microvariation detection and sound dual-angle absorption, based on the physical significance of the two singularities. Relevant ideas could apply to acoustic impedance engineering or object identification in underwater environments.

## ACKNOWLEDGMENT

This work is supported by National Natural Science Foundation of China (No. U2141244, No. 12074288 and No. 12074286), the Shanghai Science and Technology Committee (Grants No. 21JC1405600, No. 22JC1404100, No. 20ZR1460900, and No. 20ZR1461700), and the Fundamental Research Funds for the Central Universities.

## APPENDIX A: DEVIATION OF THE NUMERICAL METHOD TO SOLVE THE REFLECTION SPECTRA

As shown in Fig. 1, when the plane waves are incident into the reflection-type metagrating, we could describe the total field $p_{t}$ above the surface as

$$
p _ {t} (x, y) = A _ {0} ^ {+} e ^ {j (- k _ {0} \sin \theta_ {i} x + k _ {0} \cos \theta_ {i} y)} + \sum_ {n} A _ {n} ^ {-} e ^ {- j k _ {x, n} x} e ^ {- j k _ {y, n} y},\tag{A1}
$$

where $k_{0}$ is the wave number in free space, $A_{0}^{+}$ is the incident wave amplitude with angle $\theta_{i}$ , and $A_{n}^{-}$ represents the amplitude of the nth-order diffraction wave. $k_{x,n} = k_{0} \sin \theta_{i} + nG$ , while $G = 2\pi / D$ is the reciprocal vector determining the propagation orders in free space. To demonstrate the second-order singularities in scattering systems, we set n = 1 to provide the two-channel metagratings. $k_{y,n} = \sqrt{k_{0}^{2} - k_{x,n}^{2}}$ represents the nth-order wave vector in the y direction. We could describe the pressure distribution inside the closed-end grooves as

$$
p _ {g} ^ {s} (x, y) = \sum_ {k} a _ {k s} \cos \alpha_ {k s} (x - x _ {s}) \left(e ^ {j \beta_ {k s} y} + e ^ {- j \beta_ {k s} (y + 2 l _ {s})}\right).\tag{A2}
$$

$p_{g}^{s}$ refers to the pressure in the sth groove, $a_{ks}$ and $b_{ks}$ mean the amplitudes of the kth-order waveguide mode along $\mp y$ directions inside the sth track. Due to the melamine cotton added into tracks, the complex wave number in the sth lossy groove should represent $k' = k_{0}/(1 + ic_{s}^{i})$ . $\alpha_{ks} = (k\pi)/t_{s}$ and $\beta_{ks} = \sqrt{k'^{2} - \alpha_{ks}^{2}}$ represent the x and y components. $t_{s}, l_{s}$ , and $x_{s}$ are the width, depth, and initial position of the sth groove.

Based on the above equation, particle velocity in the y direction could be expressed as

$$
\begin{array}{l} v _ {t} (x, y) = - \frac {A _ {0} ^ {+}}{\rho \omega} k _ {0} \cos \theta_ {i} e ^ {j (- k _ {0} \sin \theta_ {i} x + k _ {0} \cos \theta_ {i} y)} \\ \qquad + \sum_ {n} \frac {k _ {y , n}}{\rho \omega} A _ {n} ^ {-} e ^ {- j k _ {x, n} x} e ^ {- j k _ {y, n} y}, \end{array}\tag{A3}
$$

$$
\begin{array}{c} v _ {g} ^ {s} (x, y) = - \sum_ {k} \frac {\beta_ {k s}}{\rho \omega} a _ {k s} \mathrm{cos} \alpha_ {k s} (x - x _ {s}) e ^ {j \beta_ {k s} y} \\ + \sum_ {k} \frac {\beta_ {k s}}{\rho \omega} b _ {k s} \mathrm{cos} \alpha_ {k s} (x - x _ {s}) e ^ {- j \beta_ {k s} y}. \end{array}\tag{A4}
$$

According to the continuum distribution for the sound pressure and normal velocity, we could deduce the linear equations about the corresponding vectors as

$$
\binom{P _ {1}}{V _ {1}} A _ {0} ^ {+} = \left( \begin{array}{c c} - P _ {2} & P _ {3} \\ - V _ {2} & V _ {3} \end{array} \right) \binom{A ^ {-}}{A},\tag{A5}
$$

where $A^{-} = (A_{-N}^{-}, \ldots, 0, \ldots, A_{N}^{-})^{T}$ describes the amplitude of $2N + 1$ orders of reflection components. $A = (A_{1}, \ldots, A_{S})^{T}$ and $A_{S} = (A_{1,S}, \ldots, A_{K,S})^{T}$ represent the amplitude of waveguide modes, in which K and S are the total numbers of standing-wave modes and grooves, respectively. The conditional matrix elements in $P_{1} \sim P_{3}$ can be expressed after multiplying by the orthogonal

functions.

$$
P _ {1} ^ {s} (k) = \frac {1}{t _ {s}} \int_ {x _ {s}} ^ {x _ {s} + t _ {s}} e ^ {- j k _ {0} \sin \theta_ {i} x} \cos \alpha_ {k s} (x - x _ {s}) d x,\tag{A6}
$$

$$
P _ {2} ^ {s} (k, n) = \frac {1}{t _ {s}} \int_ {x _ {s}} ^ {x _ {s} + t _ {s}} e ^ {- j k _ {x, n} x} \cos \alpha_ {k s} (x - x _ {s}) d x,\tag{A7}
$$

$$
\begin{array}{l} P _ {3} ^ {s} (k _ {2}, k _ {1}) = \frac {1}{t _ {s}} \left(1 + e ^ {- 2 j \beta_ {k _ {1} s} h _ {s}}\right) \\ \times \int_ {x _ {s}} ^ {x _ {s} + t _ {s}} \cos \alpha_ {k _ {2} s} (x - x _ {s}) \cos \alpha_ {k _ {1} s} (x - x _ {s}) d x, \end{array}\tag{A8}
$$

Similarly, elements in $V_{1} \sim V_{3}$ can be represented as

$$
V _ {1} (m) = - \frac {k _ {0} \cos \theta_ {i}}{D} \int_ {0} ^ {D} e ^ {- j k _ {0} \sin \theta_ {i} x} e ^ {j k _ {x, m} x} d x,\tag{A9}
$$

$$
V _ {2} (m, n) = \frac {k _ {y , n}}{D} \int_ {0} ^ {D} e ^ {- j (k _ {x, n} - k _ {x, m}) x} d x,\tag{A10}
$$

$$
\begin{array}{l} V _ {3} ^ {s} (m, k) = - \frac {\beta_ {k , s}}{D} \left(1 - e ^ {- 2 j \beta_ {k s} l _ {s}}\right) \\ \times \int_ {x _ {s}} ^ {x _ {s} + t _ {s}} \cos \alpha_ {k s} (x - x _ {s}) e ^ {j k _ {x, m} x} d x, \end{array}\tag{A11}
$$

where $m \in (-N, \ldots, N)$ represents the order of the orthogonal term. After the geometric parameters of the metagrating are confirmed, we can obtain numerical solutions for all elements in conditional matrices. Once the complex amplitude of incidence is known, the reflection spectrum and the amplitude of each order standing wave can be solved rigorously.

However, we cannot determine the geometrical parameters immediately according to the target reflection spectrum. The global optimization (genetic algorithm, GA) is adopted in this paper to obtain the geometrical parameters, further realizing the fantastic sound distribution around singularities. Detailed parameters of metagratings on EP and DP are shown in Tables I and II, respectively. dx represents the spacing of rectangular grooves, and $c_{s}^{i}$ is the imaginary parts of the normalized complex sound velocity in the sth groove.

TABLE I. Parameters of two-channel metagrating on EP (in terms of D or $\lambda$ ).

<table><tr><td>s</td><td> $l_s(\lambda)$ </td><td> $t_s(D)$ </td><td> $dx_s(D)$ </td><td> $c_s^i$ </td></tr><tr><td>1</td><td>0.569</td><td>0.227</td><td>0.276</td><td>0</td></tr><tr><td>2</td><td>0.195</td><td>0.115</td><td>0.070</td><td>0.074</td></tr><tr><td>3</td><td>0.232</td><td>0.153</td><td></td><td>0</td></tr></table>

TABLE II. Parameters of two-channel metagrating on DP (in terms of D or $\lambda$ ).

<table><tr><td>s</td><td>hs(λ)</td><td>ts(D)</td><td>dxs(D)</td><td>ciS</td></tr><tr><td>1</td><td>0.211</td><td>0.137</td><td>0.096</td><td>0.240</td></tr><tr><td>2</td><td>0.491</td><td>0.187</td><td>0.081</td><td>0</td></tr><tr><td>3</td><td>0.210</td><td>0.145</td><td></td><td>0.251</td></tr></table>

## APPENDIX B: DETAILS FOR EXPERIMENTAL SETUP AND THE CONSTRUCTION OF MEASURED SAMPLES

As for the experimental setup, we construct the two-dimensional waveguide. The schematic diagram of the experimental design is shown in Fig. 6. Organic glass plates are put on the top and bottom of the grating sample (fabricated with 12 periods). The waveguide height is 40 mm (<λ/2) to guarantee the plane-wave propagation. Sound waves from 3.33 to 3.53 kHz generated by the transducer array, which is constructed with seven loudspeakers (one-inch, Hivi B1S) and placed on the periphery at ±45° to the center of the sample. This generator is placed 600 mm from the sample center to produce continuous sound waves. Melamine cotton around the perimeter and the generator is omitted in Fig. 6. The amplitude and phase signals can be measured by a 1/8-inch microphone (Brüel & Kjær Type 2670) connected with data-acquisition hardware (NI PXI-4461 in NI PXIe-1071). The microphone ties to the automatic scanning stage. In order to mimic the nonreflective boundary, melamine cotton places on the sides of the waveguide. In the experiment, we need to scan without a sample to obtain the incident field. Then the piece is identified and scanned again. We can get the scattering field around EP and DP by subtracting the background from the total sound field.

![](images/44d593153a63d0ab5080fbacbebf4d5a71927783fccaf06ef235b6dd8b2a14e0.jpg)
FIG. 6. Experimental setup diagram. The measured area is labeled by the red-dotted box, which scans by a 1/8-inch microphone connected to the automatic sweep controller. The black cuboid identifies one period of the sample and the position of the melamine cotton, which is placed in the waveguide constructed with two pieces of organic glasses.

TABLE III. Parameters of metagrating on EP in the experiment (in terms of period D and wavelength $\lambda$ ).

<table><tr><td>s</td><td> $l_s(\lambda)$ </td><td> $t_s(D)$ </td><td> $dx_s(D)$ </td><td> $l_s^{\text{cotton}}(\lambda)$ </td></tr><tr><td>1</td><td>0.569</td><td>0.227</td><td>0.276</td><td>0</td></tr><tr><td>2</td><td>0.684</td><td>0.138</td><td>0.070</td><td>0.161</td></tr><tr><td>3</td><td>0.232</td><td>0.153</td><td></td><td>0</td></tr></table>

The metagrating samples on EP and DP fabricated with stereo lithography apparatus are selected in the experiment. Note that, the loss factors $c_{s}^{i}$ in theory could be tuned by adjusting the thickness of melamine cotton added into the lossy grooves. To imitate the loss medium accurately in authentic environments, we slightly tune the groove sizes and the thickness of the sth melamine cotton based on the Johnson-Champoux-Allard model and optimization module in COMSOL Multiphysics. So the metagrating samples covered by melamine cotton can be described by the exact EP or DP scattering matrices. The finely tuned geometries of metagratings and the thickness of melamine cotton are shown in Tables III and IV. The fixed height of covered cotton in lossy grooves is $l_{s}^{cotton}$ .

## APPENDIX C: ABSORPTION BEHAVIOR BY VARYING INCIDENT ANGLES IN A BROAD RANGE

The dual-angle absorption induced by the metagrating on the DP state has been revealed in the main body of this paper. To demonstrate the weak response strength induced by the external perturbation around DP, we further investigate the absorption behavior by varying incident angles with the same design. The absorption coefficient is relevant to the reflection power of each order and is defined as

$$
\alpha = 1 - \sum_ {n} (P _ {n} ^ {r} / P ^ {i}),\tag{C1}
$$

where P represents the sound power, the superscript r and i refers to incidence and reflection, respectively, the subscript n is the reflection order. Considering that the period of the metagrating determines a finite number of propagation modes, the relation between the absorption coefficient and the incident angles could be solved theoretically, as shown in Fig. 7. The perfect sound absorption is realized at $\theta_{i} = \pm45^{\circ}$ , and a remarkable absorption ( $\alpha \geq 0.8$ ) under incident angles in a broad range ( $-60^{\circ} \leq \theta_{i} \leq 60^{\circ}$ ) is presented without altering the design, which demonstrates the robustness against the perturbation in incident angles induced by DP.

TABLE IV. Parameters of metagrating on DP in the experiment (in terms of period D and wavelength $\lambda$ ).

<table><tr><td>s</td><td> $l_s(\lambda)$ </td><td> $t_s(D)$ </td><td> $dx_s(D)$ </td><td> $l_s^{\text{cotton}}(\lambda)$ </td></tr><tr><td>1</td><td>0.213</td><td>0.098</td><td>0.182</td><td>0.213</td></tr><tr><td>2</td><td>0.049</td><td>0.161</td><td>0.051</td><td>0</td></tr><tr><td>3</td><td>0.216</td><td>0.106</td><td></td><td>0.216</td></tr></table>

![](images/97dd2fef5c2fa3fd46feff051000c41f9be7b253a8fb259171e528e52c10c3b5.jpg)
FIG. 7. Absorption coefficients of sound power under various incident angles.

## APPENDIX D: DETAILS OF THE NONLOCAL EFFECT IN THE METAGRATING

To prove the significance of nonlocal effects in these metagratings, the local energy intensity and sound distribution are represented in Fig. 8, which is numerically calculated with the acoustic frequency module in COMSOL Multiphysics. Here the acoustic intensity refers to the product of sound pressure and complex conjugate of the velocity in both x and y directions. Figures 8(a) and 8(b) show the total field and intensity distribution near the surface on EP, while (c) and (d) represent the DP scenario. The white arrows indicate the direction of local intensity, where the amplitude is represented by its length. Lossy grooves are marked by the black dotted lines. When approaching the surface of metagratings, white arrows are strongly distorted, meaning that the lateral energy flow (evanescent wave) along the surface plays a role in sound control, which refers to the nonlocal effect. Total absorption occurs when the energy flow concentrates into the lossy grooves [Figs. 8(a), 8(c), and 8(d)]. Note that the energy intensity in Fig. 8(b) is so weak due to the counteraction between the incident and reflected waves. This reveals that flexible control of multichannel sound field could be achieved with the nonlocal metagrating, so it is an excellent platform to approach the scattering matrix exactly on EP and DP.

![](images/c2cd3f6af7c9f11071d020793cb622a92cb2090f51e496627e29616a4b105cdf.jpg)
FIG. 8. Sound field and nonlocal intensity distribution near the surface of metagrating in one period. (a),(b) refer to EP, (c),(d) represent DP state. The black-dotted lines represent the lossy grooves. The white and orange arrows illustrate the energy flow and the direction of incidence, respectively.

[1] W. R. Hamilton, On a General Method of Expressing the Paths of Light, & of the Planets, by the Coefficients of a Characteristic Function (PD Hardy, Dublin, 1833).

[2] M. Berry, R. Bhandari, and S. Klein, Black plastic sandwiches demonstrating biaxial optical anisotropy, Eur. J. Phys. 20, 1 (1999).

[3] W. J. Chen, S. K. Ozdemir, G. M. Zhao, J. Wiersig, and L. Yang, Exceptional points enhance sensing in an optical microcavity, Nature 548, 192 (2017).

[4] H. Wang, Y. H. Lai, Z. Yuan, M. G. Suh, and K. Vahala, Petermann-factor sensitivity limit near an exceptional point in a Brillouin ring laser gyroscope, Nat. Commun. 11, 1610 (2020).

[5] P. Djorwe, Y. Pennec, and B. Djafari-Rouhani, Exceptional Point Enhances Sensitivity of Optomechanical Mass Sensors, Phys. Rev. Appl. 12, 024002 (2019).

[6] C. M. Bender and S. Boettcher, Real Spectra in Non-Hermitian Hamiltonians having PT Symmetry, Phys. Rev. Lett. 80, 5243 (1998).

[7] C. Dembowski, H. D. Graf, H. L. Harney, A. Heine, W. D. Heiss, H. Rehfeld, and A. Richter, Experimental Observation of the Topological Structure of Exceptional Points, Phys. Rev. Lett. 86, 787 (2001).

[8] W. D. Heiss, Exceptional points of non-Hermitian operators, J. Phys. A: Math. Gen. 37, 2455 (2004).

[9] M. A. Miri and A. Alu, Exceptional points in optics and photonics, Science 363, 42 (2019).

[10] L. Feng, Z. J. Wong, R. M. Ma, Y. Wang, and X. Zhang, Single-mode laser by parity-time symmetry breaking, Science 346, 972 (2014).

[11] W. Tang, X. Jiang, K. Ding, Y.-X. Xiao, Z.-Q. Zhang, C. T. Chan, and G. Ma, Exceptional nexus with a hybrid topological invariant, Science 370, 1077 (2020).

[12] V. Achilleos, G. Theocharis, O. Richoux, and V. Pagneux, Non-hermitian acoustic metamaterials: Role of exceptional points in sound absorption, Phys. Rev. B 95, 144303 (2017).

[13] L. Feng, X. F. Zhu, S. Yang, H. Y. Zhu, P. Zhang, X. B. Yin, Y. Wang, and X. Zhang, Demonstration of a large-scale optical exceptional point structure, Opt. Express 22, 1760 (2014).

[14] Y. Huang, G. Veronis, and C. Min, Unidirectional reflectionless propagation in plasmonic waveguide-cavity systems at exceptional points, Opt. Express 23, 29882 (2015).

[15] Z. Lin, H. Ramezani, T. Eichelkraut, T. Kottos, H. Cao, and D. N. Christodoulides, Unidirectional Invisibility Induced by PT-Symmetric Periodic Structures, Phys. Rev. Lett. 106, 213901 (2011).

[16] T. Liu, X. F. Zhu, F. Chen, S. J. Liang, and J. Zhu, Unidirectional Wave Vector Manipulation in Two-Dimensional Space with an all Passive Acoustic Parity-Time-Symmetric Metamaterials Crystal, Phys. Rev. Lett. 120, 124502 (2018).

[17] X. S. Fang, N. J. R. K. Gerard, Z. L. Zhou, H. Ding, N. Y. Wang, B. Jia, Y. C. Deng, X. Wang, Y. Jing, and Y. Li, Observation of higher-order exceptional points in a non-local acoustic metagrating, Commun. Phys. 4, 271 (2021).

[18] U. Günther, I. Rotter, and B. F. Samsonov, Projective Hilbert space structures at exceptional points, J. Phys. A: Math. Theor. 40, 8815 (2007).

[19] A. P. Seyranian, O. N. Kirillov, and A. A. Mailybaev, Coupling of eigenvalues of complex matrices at diabolic and exceptional points, J. Phys. A: Math. Gen. 38, 1723 (2005).

[20] F. Keck, H. J. Korsch, and S. Mossmann, Unfolding a diabolic point: A generalized crossing scenario, J. Phys. A: Math. Gen. 36, 2125 (2003).

[21] J. Yang, et al., Diabolical points in coupled active cavities with quantum emitters, Light: Sci. Appl. 9, 6 (2020).

[22] C. Dembowski, B. Dietz, H. D. Graf, H. L. Harney, A. Heine, W. D. Heiss, and A. Richter, Encircling an exceptional point, Phys. Rev. E 69, 056216 (2004).

[23] J. Wiersig, Sensors operating at exceptional points: General theory, Phys. Rev. A 93, 033809 (2016).

[24] W. D. Heiss, Phases of wave functions and level repulsion, Eur. Phys. J. D 7, 1 (1999).

[25] H. Hodaei, A. U. Hassan, S. Wittek, H. Garcia-Gracia, R. El-Ganainy, D. N. Christodoulides, and M. Khajavikhan, Enhanced sensitivity at higher-order exceptional points, Nature 548, 187 (2017).

[26] W. Chen, C. Wang, B. Peng, and L. Yang, Ultra-High-Q Optical Microcavities (World Scientific, Singapore, 2021), pp. 269.

[27] N. F. Yu, P. Genevet, M. A. Kats, F. Aieta, J. P. Tetienne, F. Capasso, and Z. Gaburro, Light propagation with phase discontinuities: Generalized laws of reflection and refraction, Science 334, 333 (2011).

[28] T. Liu, G. C. Ma, S. J. Liang, H. Gao, Z. M. Gu, S. W. An, and J. Zhu, Single-sided acoustic beam splitting based on parity-time symmetry, Phys. Rev. B 102, 014306 (2020).

[29] X. Wang, X. S. Fang, D. X. Mao, Y. Jing, and Y. Li, Extremely Asymmetrical Acoustic Metasurface Mirror at the Exceptional Point, Phys. Rev. Lett. 123, 214302 (2019).

[30] T. Liu, F. Chen, S. Liang, H. Gao, and J. Zhu, Subwavelength Sound Focusing and Imaging via Gradient Metasurface-Enabled Spoof Surface Acoustic Wave Modulation, Phys. Rev. Appl. 11, 034061 (2019).

[31] L. Quan and A. Alu, Hyperbolic Sound Propagation over Nonlocal Acoustic Metasurfaces, Phys. Rev. Lett. 123, 244303 (2019).

[32] Y. Ra'di, D. L. Sounas, and A. Alu, Metagratings: Beyond the Limits of Graded Metasurfaces for Wave Front Control, Phys. Rev. Lett. 119, 067404 (2017).

[33] D. Torrent, Acoustic anomalous reflectors based on diffraction grating engineering, Phys. Rev. B 98, 060101 (2018).

[34] Y. B. Jin, X. S. Fang, Y. Li, and D. Torrent, Engineered Diffraction Gratings for Acoustic Cloaking, Phys. Rev. Appl. 11, 011004 (2019).

[35] H. Q. Ni, X. S. Fang, Z. L. Hou, Y. Li, and B. Assouar, High-efficiency anomalous splitter by acoustic meta-grating, Phys. Rev. B 100, 104104 (2019).

[36] Z. L. Hou, X. S. Fang, Y. Li, and B. Assouar, Highly Efficient Acoustic Metagrating with Strongly Coupled Surface Grooves, Phys. Rev. Appl. 12, 034021 (2019).

[37] Q. Song, M. Odeh, J. Zúñiga-Pérez, B. Kanté, and P. Genet, Plasmonic topological metasurface by encircling an exceptional point, Science 373, 1133 (2021).

[38] I. Rotter, A non-Hermitian Hamilton operator and the physics of open quantum systems, J. Phys. A: Math. Theor. 42, 153001 (2009).

[39] J. Y. Zeng, Quantum Mechanics (Science Press, Beijing, 2000), Vol. I.
