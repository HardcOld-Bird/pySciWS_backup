# Bulk-spatiotemporal vortex correspondence in gyromagnetic zero-index media

https://doi.org/10.1038/s41586-025-08948-6

Received: 5 August 2024

Ruo-Yang Zhang $^{1,8}$ , Xiaohan Cui $^{1,8✉}$ , Yuan-Song Zeng $^{2,8}$ , Jin Chen $^{2}$ , Wenzhe Liu $^{1,3}$ , Mudi Wang $^{1,4}$ , Dongyang Wang $^{5}$ , Zhao-Qing Zhang $^{1}$ , Neng Wang $^{6✉}$ , Geng-Bo Wu $^{2✉}$ & C. T. Chan $^{1,7✉}$

Accepted: 26 March 2025

Published online: 14 May 2025

Check for updates

Photonic double-zero-index media, distinguished by concurrently zero-valued permittivity and permeability, exhibit extraordinary properties not found in nature $^{1-8}$ . Notably, the notion of zero index can be substantially expanded by generalizing the constitutive parameters from null scalars to non-reciprocal tensors with non-zero matrix elements but zero determinants $^{9,10}$ . Here we experimentally realize this class of gyromagnetic double-zero-index metamaterials possessing both double-zero-index features and non-reciprocal hallmarks. As an intrinsic property, this metamaterial always emerges at a spin-1/2 Dirac point of a topological phase transition. We discover and demonstrate that a spatiotemporal reflection vortex singularity is always anchored to the Dirac point of the metamaterial, with the vortex charge being determined by the topological invariant leap across the phase transition. This establishes a unique bulk–spatiotemporal vortex correspondence that extends the protected boundary effects into the time domain and characterizes topological phase-transition points, setting it apart from any pre-existing bulk-boundary correspondence. Based on this correspondence, we propose and experimentally demonstrate a mechanism to deterministically generate optical spatiotemporal vortex pulses $^{11,12}$ with firmly fixed central frequency and momentum, hence showing ultrarobustness. Our findings uncover connections between zero-refractive-index photonics, topological photonics and singular optics, which might enable the manipulation of space-time topological light fields using the inherent topology of extreme-parameter metamaterials.

Photonic zero-refractive-index media $^{13-16}$ are an exemplary family of extreme-parameter metamaterials, which exhibit extraordinary optical properties and have wide-ranging applications in wave manipulation $^{17-19}$ and nonlinear optics $^{20,21}$ . In this family of metamaterials, a special member is the media with both zero permittivity ( $\varepsilon$ ) and permeability ( $\mu$ ), known as double-zero-index metamaterials (DZIMs) $^{1-8}$ . DZIMs are unique in that they not only present a zero refractive index but also show universal impedance matching with any surrounding media and possess a conical band intersection at the double-zero-index frequency $^{1,2}$ . These properties unlock several interesting optical characteristics and functionalities, such as perfect transmission and percolation $^{22}$ , super Klein tunnelling $^{8}$ and zero-refractive-index bound states in the continuum $^{23,24}$ .

Although the conventional DZIMs have null scalar $\varepsilon$ and $\mu$ , it has been shown that by incorporating anisotropic and non-reciprocal constitutive tensors with non-zero matrix elements but zero determinants, the realm of zero-refractive-index metamaterials can be substantially expanded $^{9,10,25,26}$ . In this work, we experimentally realize a generalized DZIM, called gyromagnetic double-zero-index metamaterials (GDZIMs) $^{27-29}$ , possessing a null scalar permittivity $\varepsilon_{z}$ and a non-zero gyromagnetic permeability tensor with a determinant of zero det[ $\stackrel{\leftrightarrow}{\mu}_{T}$ ] = 0. We theoretically and experimentally show that the GDZIMs not only maintain the prominent features of conventional DZIMs but also possess non-reciprocal characteristics that allow us to observe unprecedented wave phenomena in zero-refractive-index photonics.

The resemblance and distinction between ordinary DZIMs (Fig. 1a,b) and GDZIMs (Fig. 1c,d) are first encoded in their band structures near the DZIM frequency $\omega_0$ . In a two-dimensional (2D) ordinary DZIM with linear material dispersion near $\omega_0$ , that is, $\varepsilon(\omega) = c_{\varepsilon}(\omega - \omega_0)$ and $\mu(\omega) = c_{\mu}(\omega - \omega_0)$ , the frequency bands exhibit a so-called Dirac-like triple crossing between a cone and an irremovable flat band at the $\Gamma$ point ( $k = 0$ ) of momentum space $^{1,8}$ (Fig. 1b). Likewise, a 2D GDZIM for transverse magnetic (TM) waves can always be characterized by linearly dispersive constitutive parameters around $\omega_0$ through Taylor expansion:

$$
\begin{array}{l} \varepsilon_ {z} (\omega) = c _ {\varepsilon} (\omega - \omega_ {0}), \\ \stackrel {{\leftrightarrow}} {{\mu}} _ {\mathrm{T}} (\omega) = \left( \begin{array}{c c} \alpha_ {0} & - \mathrm{i} \alpha_ {0} \\ \mathrm{i} \alpha_ {0} & \alpha_ {0} \end{array} \right) + \left( \begin{array}{c c} c _ {\mu_ {\mathrm{d}}} & - \mathrm{i} c _ {\alpha} \\ \mathrm{i} c _ {\alpha} & c _ {\mu_ {\mathrm{d}}} \end{array} \right) (\omega - \omega_ {0}), \end{array}\tag{1}
$$

![](images/1b8d6a8fc4c1f03710613e7e850a4d9c71114a9f7fe7da2585aebbce3ead4074.jpg)
Fig. 1 | Comparing the characteristics of gyromagnetic and ordinary DZIMs. a,c, Schematics of plane waves in homogeneous ordinary (left) and gyromagnetic (right) DZIMs. The green arrows in c indicate the gyromagnetic precession of the magnetic moments around an external magnetic field inside the GDZIM. b,d, The band structures near the double-zero-index frequency ( $\omega_{0}$ ) form a spin-1 cone (b) and a spin-1/2 Dirac cone (d) in ordinary and gyromagnetic DZIMs, respectively. The bars and ellipses on the bottom planes represent the eigen polarization of the magnetic field H(k) of the upper band in the two kinds
of DZIMs. In b, an ordinary DZIM, the linearly polarized magnetic fields approach a V-point-type singularity at the $\Gamma$ point (k = 0) of the momentum space, whereas in d, a GDZIM, the eigenfields approach circular polarization $|C\rangle = (\hat{\mathbf{x}} - \mathrm{i}\hat{\mathbf{y}})/\sqrt{2}$ (a C point) at $\Gamma$ . The colours of the polarization ellipses in d represent the phase of the eigenfield projected to the circular basis: $\phi = \arg\langle C|\mathbf{H}(\mathbf{k})\rangle$ . e, Illustration of a 2D Gaussian pulse normally impinging on a GDZIM slab at the Dirac cone frequency and the normal reflection of the pulse creating a spatiotemporal vortex.

which gives rise to a spin-1/2 Dirac cone dispersion pinned at $(\omega, \mathbf{k}) = (\omega_{0}, \mathbf{0})$ (see Fig. 1d and Methods for details). Notwithstanding the analogous conical shape, the Dirac-like cone in ordinary DZIMs has a spin-1 nature and is thus fundamentally different from a true Dirac cone of spin-1/2 (ref. 8). In bosonic systems, time-reversal symmetry inherently prohibits a 2D spin-1/2 Dirac point from existing at time-reversal-invariant momenta (Methods). Therefore, the appearance of the centred spin-1/2 Dirac point in GDZIMs directly signifies time-reversal breaking and implies non-trivial non-reciprocal effects $^{28,29}$ . For instance, the eigen magnetic fields in GDZIMs converge to an in-plane circular polarization at $\Gamma$ , as a result of the gyromagnetic precession of the magnetic moments induced by the external magnetic field. This highlights the specific preference of GDZIMs for a certain out-of-plane polarization of the transverse optical spin $\mathbf{S}_{\mathrm{opt}} = \frac{\mu_{0}}{4\omega_{0}}\mathrm{Im}(\mathbf{H}^{*} \times \mathbf{H})$ for waves propagating in any in-plane direction.

GDZIMs inherit the salient characteristic of DZIMs, namely, the ability to match impedance with any surrounding medium, and hence support perfect transmission (zero reflection) at the Dirac frequency $\omega_{0}$ for normally incident plane waves. However, unlike the bilaterally symmetric reflection by ordinary DZIMs, the non-reciprocity of a GDZIM slab allows for an asymmetric reflection about the normal axis ( $k_{\parallel}=0$ ), resulting in a spatiotemporal reflection phase vortex emerging and firmly anchored at the projection of the Dirac point, $(\omega, k_{\parallel}) = (\omega_{0}, 0)$ . Particularly, we uncover that a GDZIM always resides at the topological transition point of photonic Chern insulator phases, and the quantized charge of the reflection vortex is precisely determined by the variation of bulk Chern number across the phase transition, hence establishing a new bulk-boundary correspondence. This vortical reflection spectrum further ensures that when a Gaussian pulse with the central frequency of $\omega_{0}$ normally impinges on the slab, the reflected wave always forms a spatiotemporal vortex pulse (STVP) (Fig. 1e). Optical spatiotemporal vortices $^{11,12,30-40}$ , which carry transverse orbital angular momenta, have attracted considerable attention because of their potential applications in optical tweezers, superresolution imaging and optical information processing. Compared with other methods of synthesizing STVPs, such as using customized spatial light modulators $^{30-32}$ or resonance-based metasurfaces $^{35-40}$ , the current approach stems from the intrinsic Dirac cone topology of a homogeneous GDZIM and possesses ultrarobustness in that the central frequency and wavevector of the generated STVPs are independent of the thickness of the GDZIM slab, the refractive index of the background medium and even the crystal cutting directions of the realistic metamaterials.

We will now show how to design and realize the GDZIMs, as well as the theory and experiments of ultrarobust generation of STVPs in the microwave regime.

![](images/3dd516424d6b4a46f0b95729b105ff75762bd231dc1e1cdb4487676e29c1063f.jpg)
Fig. 2 | Experimental realization of GDZIM. a, Schematic of the gyromagnetic PC. b, Unit cell of the PC with a lattice constant $a = 17.2 \mathrm{~mm}$ ; a YIG rod (height: $h = 4 \mathrm{~mm}$ , radius: $r_{\mathrm{c}} = 3.1 \mathrm{~mm}$ ) is sandwiched between two perfect electric conductor (PEC) parallel plates. A permanent magnet under the YIG cylinder creates an effective bias magnetic field $H_{0} = 900 \mathrm{Oe}$ in the $z$ direction. c, The top panel shows the band inversion of the two $\Gamma$ -point modes on the second and third TM bands of the PC controlled by the filling ratio ( $r_{\mathrm{c}} / a$ ) of YIG rods ( $\varepsilon_{z} = 13, \mu_{\mathrm{d}} = 0.865, \alpha = -0.54$ ). The bottom panel shows the band structures along high-symmetry lines for three PCs in the trivial phase, at the
phase-transition point and in the Chern insulator phase. d, Profiles of the two $\Gamma$ -point modes ( $\Gamma_{1}^{-}$ : out-of-plane electric dipole and $\Gamma_{4}^{+}$ : in-plane left circularly polarized (LCP) magnetic dipole). e, Retrieved effective parameters and band dispersion (blue circles) of the effective medium near the Dirac frequency $f_{0}=10.14$ GHz. f, Simulation of plane-wave propagation through the PC and effective homogeneous medium (EHM) slab at $f_{0}$ . g, Simulated local density of states and measured projected bands of a PC with 10 layers in the x direction. White dashed lines represent the anticipated bulk Dirac cone along $k_{y}=0$ .

## Design and realization of GDZIMs

To realize the GDZIMs, we devise a 2D gyromagnetic photonic crystal (PC) (Fig. 2a,b) consisting of a square lattice of yttrium iron garnet (YIG) cylinders, each magnetized by a permanent magnet under it. Here we focus on the second and third TM bands in Fig. 2c, on which the two $\Gamma$ -point modes—an out-of-plane electric dipole and an in-plane circularly polarized magnetic dipole (Fig. 2d)—are consistent with the two eigen polarizations ( $\mathbf{E} \propto \hat{\mathbf{z}}$ and $\mathbf{H} \propto (\hat{\mathbf{x}} - \mathrm{i}\hat{\mathbf{y}})/\sqrt{2}$ ) in a homogeneous GDZIM. The gap between the two bands can be continuously tuned by the filling ratio of the PC ( $r_c/a$ ). At the critical point of gap closing, an accidentally degenerate spin-1/2 Dirac point $^{41}$ emerges at the centre of the Brillouin zone (Fig. 2c), signifying the topological transition between a photonic Chern insulator phase $^{42,43}$ with a non-trivial gap Chern number Ch = -1 and a trivial phase with Ch = 0. In this way, both the Dirac cone dispersion and the mode symmetry suggest that the PC behaves as a GDZIM at the topological phase-transition point.

Using the boundary effective medium approach $^{28}$ (Supplementary Note 1), we can retrieve the homogenized effective constitutive parameters of the PC near the Dirac point: $\varepsilon_{z}^{\mathrm{eff}}(\omega)$ and $\stackrel{\leftrightarrow}{\mu}_{\mathrm{T}}^{\mathrm{eff}}(\omega)=\begin{pmatrix}\mu_{\mathrm{d}}^{\mathrm{eff}}&-\mathrm{i}\alpha^{\mathrm{eff}}\\\mathrm{i}\alpha^{\mathrm{eff}}&\mu_{\mathrm{d}}^{\mathrm{eff}}\end{pmatrix}$ , as shown in Fig. 2e. At the Dirac frequency $f_{0}=\omega_{0}/(2\pi)=10.14$ GHz, we observe that both the effective permittivity and the determinant of effective permeability tensor reduce to zero: $\varepsilon_{z}^{\mathrm{eff}}(\omega_{0})=0$ and $\det[\stackrel{\leftrightarrow}{\mu}_{\mathrm{T}}^{\mathrm{eff}}(\omega_{0})]=0$ , whereas the diagonal and gyromagnetic components of $\stackrel{\leftrightarrow}{\mu}_{\mathrm{T}}^{\mathrm{eff}}$ remain finite and exhibit an identical value $\alpha^{\mathrm{eff}}(\omega_{0})=\mu_{\mathrm{d}}^{\mathrm{eff}}(\omega_{0})=\alpha_{0}=-1.45$ , which precisely reproduce the GDZIM properties described in equation (1). In Fig. 2f, we observe no spatial phase change in the field within the GDZIM region, which validates the zero-index characteristic of the PC and the efficacy of the effective medium methodology.

We have conducted experimental detection for the spin-1/2 Dirac cone occurring at the topological transition point of the PC (Methods and Extended Data Fig. 2). Figure 2g shows the measured projected band structure, alongside the local density of states, of a PC with 10 layers (Methods and Supplementary Note 6.B). Their good agreement substantiates that the constructed PC achieves a GDZIM at the anticipated frequency $f_{0} = 10.14$ GHz.

## Ultrarobust reflection phase vortex

The unique constitutive parameters and band structure of GDZIMs provide them with exceptional transport peculiarities. To see this, we consider a TM plane wave incident on a GDZIM slab with thickness d embedded in a background medium with isotropic parameters $\varepsilon_{1}$ and $\mu_{1}$ , as shown in Fig. 3a. For normal incidence ( $k_{y}=0$ ), the reflection coefficient satisfies (Supplementary Note 2.A)

$$
R \propto \left(\frac {e ^ {2 \mathrm{i} k _ {x} ^ {\text { eff }} d} - 1}{k _ {x} ^ {\text { eff }}}\right) \left(\varepsilon_ {1} \widetilde {\mu} ^ {\text { eff }} - \varepsilon_ {z} ^ {\text { eff }} \mu_ {1}\right),\tag{2}
$$

with $\tilde{\mu}^{\mathrm{eff}} = \det [\hat{\mu}_{\mathrm{T}}^{\leftrightarrow \mathrm{eff}}] / \mu_{\mathrm{d}}^{\mathrm{eff}}$ and $k_{x}^{\mathrm{eff}}$ denoting, respectively, the virtual scalar permeability and the $x$ component of wavevector in the GDZIM. Equation (2) shows that perfect transmission $(R = 0)$ occurs if either of the two right-hand-side terms equals zero. As shown in Fig. 3b, the first term reduces to zero at Fabry-Pérot resonances (bent curves) that depend on the slab thickness. By contrast, the vanishing of the second term defines the generalized impedance matching condition for gyromagnetic materials: $\varepsilon_{1}\tilde{\mu}^{\mathrm{eff}} = \varepsilon_{z}^{\mathrm{eff}}\mu_{\mathrm{I}}$ In particular, at the Dirac frequency (white dashed lines), $\varepsilon_{z}^{\mathrm{eff}}(\omega_{0})=\widetilde{\mu}^{\mathrm{eff}}(\omega_{0})=0$ , the impedance matching is universally established for arbitrary values of $\varepsilon_{1}$ and $\mu_{1}$ . Therefore, the GDZIMs inherit the ability as ordinary DZIMs to match impedance with a medium of any parameters and arbitrary thickness $^{1}$ .

b
![](images/8d1cd926d2b4a7ea15781baa2bb38c83d56d50136b9d3a0bd5ffbd9e207dd2ba.jpg)

![](images/74695ecea989b06a3d0b3bb00d1c38689492cb70a39d964b95600a5bef247559.jpg)

![](images/31ad773aa0245bad0497416284f4184f643674bc14344d511c77dcb392d854f4.jpg)
Fig. 3 | Ultrarobust reflection phase vortex and its topological origin.

a, Diagram of plane-wave reflection by a GDZIM slab. b, Normal incidence reflection spectra by GDZIM slabs made of metamaterials (left) and of effective homogeneous media (right), changing with the slab thickness d. The white dashed line indicates the Dirac frequency $f_{0}=10.14$ GHz. c, Reflection amplitude (left) and phase (right) of an ordinary DZIM slab with a spin-1 Dirac-like cone (grey dashed lines). Parameters of DZIM: $\varepsilon_{z}=13$ , $\mu=1$ , $r_{c}=3.4$ mm and a=17.2 mm.

Although GDZIMs and ordinary DZIMs both support universal impedance matching for normal incidence, the off-normal reflection feature of GDZIMs differs markedly from that of conventional DZIMs by non-reciprocity. The presence of reciprocity and mirror-y symmetry of an ordinary DZIM slab imposes two constraints on the scattering matrix (Supplementary Note 3): $S(k_y, \omega)^{\mathrm{T}} = S(-k_y, \omega)$ and $S(k_y, \omega) = S(-k_y, \omega)$ , both of which protect the reflections bilaterally symmetric about the $k_y = 0$ axis: $R(-k_y, \omega) = R(k_y, \omega)$ , as shown in Fig. 3c. Consequently, the winding of reflection phase around the reflection singularity (projection of Dirac-like point) must be zero. However, for GDZIM slabs, both reciprocity and mirror-y symmetry are broken because of the external magnetic field. Thus, the reflections (Fig. 3d,e) from a GDZIM slab are generically asymmetric about the normal axis, which leads to a deterministic formation of a topological reflection phase vortex encircling the isolated reflection singularity anchored at the projected Dirac point, $(k_y, \omega) = (0, \omega_0)$ , in the $(k_y, \omega)$ -plane.

The GDZIM-induced reflection phase vortex is topologically stable as protected by a quantized phase winding number around it, $\nu(R, C_{\mathrm{D}}) = \frac{1}{2\pi} \oint_{C_{\mathrm{D}}} d\arg(R) = \pm 1$ (ref. 44). But beyond that, surpassing the usual stability of spectral phase singularities that merely persist but generally move with varying parameters, the GDZIM-induced vortex exhibits ultrarobustness, as its frequency and wavevector are solidly pinned regardless of the background medium, the slab thickness or even the specific crystal plane along which the metamaterials are truncated.

Beyond theoretical analysis, we experimentally demonstrated that the GDZIM-induced vortex centre remains fixed with varying slab thickness from one to four layers (Methods and Extended Data Fig. 3) and provided numerical evidence for the ultrarobustness to the background medium variations and crystal plane truncations (Supplementary Notes 2.B and 2.C).

![](images/ed6e74d7b86ae3f93218b2d4cbb30048423d89099022ce1b9d49536f63b43fc3.jpg)

![](images/9b8c8833f165d96dd6cce419b63ec43b97944707284fc21b96143d5957e43f5f.jpg)

![](images/276479aafadcfd8d50aa42db2ccc83fa9937f7864c7fab868929468418186280.jpg)
d,e, Reflection spectra of a GDZIM slab (d) and the corresponding effective medium slab (e). The slab thickness of c-e is d = 6a. f, Evolution of the reflection spectra during the topological phase-transition process of a gyromagnetic near-DZIM slab, which happens to be an exact GDZIM at the critical gap-closing point $(\rho, k_{y}, \omega) = (\rho_{0}, 0, \omega_{0})$ . The white cone indicates the edges of the projected bulk bands, and the red line indicates the loci satisfying impedance matching and R = 0.

## Bulk-spatiotemporal vortex correspondence

The remarkable stability of the GDZIM-induced reflection vortex can be further traced to a more profound bulk topological origin, that is, a new type of bulk–spatiotemporal vortex correspondence linking the topological phase transition at the accidental Dirac point and the emergence of the spatiotemporal reflection phase vortex of a 2D homogeneous medium with finite thickness. Figure 3f shows the evolution of the reflection spectrum of a metamaterial slab that undergoes a topological transition, modulated by a differentiable variable $\rho$ of the effective constitutive parameters $\varepsilon_{z}^{\mathrm{eff}}(\omega,\rho)$ and $\stackrel{\leftrightarrow}{\mu}_{\mathrm{T}}^{\mathrm{eff}}(\omega,\rho)$ (see details in Supplementary Note 4.C). As $\rho$ varies, the reflection vortices coalesce into a singularity line in the $(\rho,k_{y},\omega)$ space. In this case, this line coincides with the impedance matching frequencies, denoted by $\omega_{IM}$ , obeying $\varepsilon_{1}\widetilde{\mu}^{\mathrm{eff}}(\omega_{\mathrm{IM}},\rho)=\varepsilon_{z}^{\mathrm{eff}}(\omega_{\mathrm{IM}},\rho)\mu_{1}$ . In particular, we have proved in the Methods that the discontinuous jump in Chern number across the phase transition enforces the line of reflection singularities to pass through the critical gap-closing point $(\rho_{0},0,\omega_{0})$ . Consequently, a spatiotemporal reflection phase vortex arises almost invariably at the surface projection of the accidental Dirac point, and the topological charge carried by the vortex is determined by the change of bulk Chern number across the topological phase transition, together with the slope of the singularity line:

$$
\nu (R, C _ {\mathrm{D}}) = \operatorname{sgn} \left[ \frac {\mathrm{d} \Delta \omega_ {\mathrm{IM}}}{\mathrm{d} \rho} \Bigg | _ {\rho_ {0}} \right] (\mathrm{Ch} _ {\rho <   \rho_ {0}} - \mathrm{Ch} _ {\rho > \rho_ {0}}),\tag{3}
$$

where $\Delta\omega_{IM}=\omega_{IM}-\omega_{mid}$ with $\omega_{\mathrm{mid}}(\rho)$ denoting the mid-gap frequency as a function of $\rho$ . Note that $v(R,C_{D})$ is essentially a $Z_{2}$ topological invariant. Thus, only the parity of the vortex charge is determined by the Chern number jump ( $Ch_{\rho<\rho_{0}}-Ch_{\rho>\rho_{0}}$ ), whereas the sign of the slope can vary with the background medium (Methods and Supplementary Note 4.B).

![](images/01fdc4168750a5060bd2cabd29390ba6f96a45171ed4de9541ae4d2f0e361d7e.jpg)
Fig. 4 | Generation and observation of STVPs. a, Time slices showing the generation of an STVP by reflecting a normally incident Gaussian pulse on a homogeneous GDZIM slab (d = 4a). The blue and yellow arrows represent the centroid worldlines of the reflected STVP and of the transmitted and incident pulses, respectively. $T_{0} = 2\pi/\omega_{0}$ denotes the temporal period corresponding to the Dirac frequency $\omega_{0}$ . b, Schematic of the experimental setup for generating and detecting STVPs. The scan line (grey dashed) is
0.5 m away from the PC samples (size: $N \times 30$ units, N = 1–4 layers). c–e, Envelopes and phases $(A(y, t) = E_{z}(y, t)e^{\mathrm{i}\omega_{0}t})$ on the scan line of input pulse and reflected pulses corresponding to the samples with 1 to 4 layers, which are obtained by theoretical calculation of the effective homogeneous medium slabs (thickness: d = Na) (c), by numerical simulations of the metamaterial slabs (d) and by experimental measurements (e).

Bulk-boundary correspondence is one of the most important effects in topological physics, establishing a bridge connecting abstract topological properties of bulk materials to the boundary transport phenomena. Compared with any known correspondences $^{45-48}$ , the new result bears three fundamental distinctions. First, the bulk topology is not attributed to a single phase; it is decided by the topological transition between two phases. Second, unlike the usual scenarios in which the boundary effects are well-defined only for a semi-infinite or large enough bulk, the current boundary reflection inherently requires that the thickness of the continuous bulk medium is finite and can even take on very small values. Third, as explained later, the new correspondence intrinsically leads to a spatiotemporal boundary effect, that is, the ultrarobust generation of STVPs, setting it apart from any stationary topological boundary effects predicted by conventional ones.

## Intrinsic generation of STVPs

With the vortical reflection phase, GDZIM slabs provide an intrinsic route towards robustly generating STVPs through reflecting a normally incident Gaussian pulse $E_{z}^{\mathrm{in}}(\mathbf{r}, t) = A^{\mathrm{in}}(\mathbf{r}, t)\mathrm{e}^{\mathrm{i}\omega_{0}(x/c-t)}$ with a central frequency at the Dirac frequency, $\omega_{0}$ , of the GDZIM, and pulse widths $\Delta x$ , $\Delta y$ in the two directions. The time slices in Fig. 4a show the generation of the reflected STVP from a homogeneous GDZIM slab with thickness $d$ . The visualization of STVP generation process using both homogeneous media and PCs is provided in Supplementary Video 1. Theoretically, the $(k_y,\omega)$ -plane vortical reflection spectrum yields an approximate expression of the reflected pulse $^{35}$ (Supplementary Note 5)

$$
E _ {z} ^ {\mathrm{r}} (\mathbf {r}, t) \propto d \left[ \frac {c \kappa (x + c t)}{\Delta x ^ {2}} - \mathrm{i} \frac {y}{\Delta y ^ {2}} \right] A ^ {\mathrm{in}} (- x, y, t) \mathrm{e} ^ {- \mathrm{i} \omega_ {0} (x / c + t)},\tag{4}
$$

where $\kappa$ is a parameter dependent on the background medium. The term in the square bracket represents a factor induced by the reflection vortex, which reduces to zero and hence forms phase singularities along the centroid worldline of the reflected pulse, $\mathbf{r}_{\mathrm{c}} = -ct\hat{\mathbf{x}}$ , leading to the emergence of a spatiotemporal vortex carrying the topological charge $v = -\mathrm{sgn}[\kappa] \in \{\pm 1\}$ along this worldline in real space-time. Hence, either on a spatial cross-section ( $t = \text{const.}$ ) or on a space-time cross-section (for example, the plane of $x = \text{const.}$ in Fig. 4c) that traverses the worldline of the pulse, a 2D phase vortex can be observed. From the perspective of topology, the generation of the STVP is the ultimate consequence of the bulk-boundary correspondence in equation (3), thereby intrinsically mirroring the bulk critical Dirac topology of GDZIMs. From the perspective of angular momentum transport, the transverse orbital angular momentum of the reflected STVP originates essentially from the transverse spin angular momentum of the eigenstates in GDZIMs.

Based on this approach, we have successfully generated and observed STVPs using the GDZIM slabs, with an experimental setup schematically shown in Fig. 4b. We first utilized an arbitrary waveform generator to emit a temporal Gaussian pulse with cylindrical wavefronts. After passing through a 3D-printing metamaterial lens (Supplementary Note 6.D), the pulse is reshaped into a Gaussian-like wavepacket that propagates along the x direction and then strikes the GDZIM sample. Using a real-time oscilloscope, we recorded the time series of the incident and reflected pulses along a scan line (the dashed line) in the y direction (Methods and Extended Data Fig. 6). To ensure the synchronization of signals measured at different points on the scan line, a reference signal, which is synchronous with the pulse-input channel, was generated from the arbitrary waveform generator and used as the trigger signal for the oscilloscope. We have detected the STVPs produced by the PC samples with one to four layers, respectively. The experimental measurements, after removing the fast oscillatory dynamical phase $A(\mathbf{r}, t) = E_{z}(\mathbf{r}, t)\mathrm{e}^{\mathrm{i}\omega_{0}t}$ , are compared with the simulated results of both effective media and metamaterials, as shown in Fig. 4c–e. We can see that spatiotemporal phase vortices appear on the $(t, y)$ space-time cross-section for all samples, and they exhibit a consistent geometric phase distribution, $\arg(A(\mathbf{r}, t))$ , around the vortex centres, although the strength of the reflected pulses increases with the slab thickness. This demonstrates that the generation of the STVPs is insensitive to the variation in the thickness of the sample.

## Conclusion and outlook

By constructing topological metamaterials residing at the critical transition point of photonic Chern insulator phases, we have experimentally realized GDZIMs, a conceptual advancement over the conventional DZIMs, that incorporates non-reciprocal and tensor-valued constitutive indices of null determinants. The non-reciprocal nature of GDZIMs, marked by an unpaired spin-1/2 Dirac point at the centre of the momentum space, gives rise to a striking spatiotemporal vortical reflection spectrum securely pinned at the surface projection of the bulk Dirac point. We established a new form of bulk-boundary correspondence, which shows that the reflection phase vortex is essentially determined by the change of the bulk topological invariants across the phase transition and ultimately leads to an intrinsic and ultrarobust pathway to the generation of STVPs through the use of GDZIMs. The realization of GDZIMs not only opens up opportunities for achieving highly functional devices, such as unidirectional zero-refractive-index waveguides $^{25}$ and scalable single-mode chiral emitting lasers $^{49,50}$ , but also builds a bridge that connects zero-refractive-index photonics $^{15,16}$ , topological photonics $^{45}$ and singular optics $^{44}$ , providing insights on each field and implying unexplored interdisciplinary research directions.

## Online content

Any methods, additional references, Nature Portfolio reporting summaries, source data, extended data, supplementary information, acknowledgements, peer review information; details of author contributions and competing interests; and statements of data and code availability are available at https://doi.org/10.1038/s41586-025-08948-6.

1. Huang, X., Lai, Y., Hang, Z. H., Zheng, H. & Chan, C. T. Dirac cones induced by accidental degeneracy in photonic crystals and zero-refractive-index materials. Nat. Mater. 10, 582–586 (2011).

2. Nguyen, V. C., Chen, L. & Halterman, K. Total transmission and total reflection by zero index metamaterials with defects. Phys. Rev. Lett. 105, 233908 (2010).

3. Moitra, P. et al. Realization of an all-dielectric zero-index optical metamaterial. Nat. Photon. 7, 791–795 (2013).

4. Li, Y. et al. On-chip zero-index metamaterials. Nat. Photon. 9, 738–742 (2015).

5. Cui, X., Ding, K., Dong, J.-W. & Chan, C. T. Realization of complex conjugate media using non-PT-symmetric photonic crystals. Nanophotonics 9, 195–203 (2019).

6. Xu, C. et al. Three-dimensional electromagnetic void space. Phys. Rev. Lett. 127, 123902 (2021).

7. Li, Y., Chan, C. T. & Mazur, E. Dirac-like cone-based electromagnetic zero-index metamaterials. Light. Sci. Appl. 10, 203 (2021).

8. Fang, A., Zhang, Z. Q., Louie, S. G. & Chan, C. T. Klein tunneling and supercollimation of pseudospin-1 electromagnetic waves. Phys. Rev. B 93, 035422 (2016).

9. Davoyan, A. R. & Engheta, N. Theory of wave propagation in magnetized near-zero-epsilon metamaterials: evidence for one-way photonic states and magnetically switched transparency and opacity. Phys. Rev. Lett. 111, 257401 (2013).

10. Horsley, Sa. R. & Woolley, M. Zero-refractive-index materials and topological photonics. Nat. Phys. 17, 348–355 (2021).

11. Jhajj, N. et al. Spatiotemporal optical vortices. Phys. Rev. X 6, 031037 (2016).

12. Bliokh, K. Y. Spatiotemporal vortex pulses: angular momenta and spin-orbit interaction. Phys. Rev. Lett. 126, 243601 (2021).

13. Ziolkowski, R. W. Propagation in and scattering from a matched metamaterial having a zero index of refraction. Phys. Rev. E 70, 046608 (2004).

14. Silveirinha, M. & Engheta, N. Tunneling of electromagnetic energy through subwavelength channels and bends using $\varepsilon$ -near-zero materials. Phys. Rev. Lett. 97, 157403 (2006).

15. Liberal, I. & Engheta, N. Near-zero refractive index photonics. Nat. Photon. 11, 149–158 (2017).

16. Kinsey, N., DeVault, C., Boltasseva, A. & Shalaev, V. M. Near-zero-index materials for photonics. Nat. Rev. Mater. 4, 742–760 (2019).

17. Liberal, I., Mahmoud, A. M., Li, Y., Edwards, B. & Engheta, N. Photonic doping of epsilon-near-zero media. Science 355, 1058–1062 (2017).

18. Ciattoni, A., Marini, A. & Rizza, C. Efficient vortex generation in subwavelength epsilon-near-zero slabs. Phys. Rev. Lett. 118, 104301 (2017).

19. Liu, M. et al. Broadband mid-infrared non-reciprocal absorption using magnetized gradient epsilon-near-zero thin films. Nat. Mater. 22, 1196–1202 (2023).

20. Suchowski, H. et al. Phase mismatch-free nonlinear propagation in optical zero-index materials. Science 342, 1223–1226 (2013).

21. Alam, M. Z., Leon, I. D. & Boyd, R. W. Large optical nonlinearity of indium tin oxide in its epsilon-near-zero region. Science 352, 795–797 (2016).

22. Luo, J., Hang, Z. H., Chan, C. T. & Lai, Y. Unusual percolation threshold of electromagnetic waves in double-zero medium embedded with random inclusions. Laser Photonics Rev. 9, 523–529 (2015).

23. Minkov, M., Williamson, I. A. D., Xiao, M. & Fan, S. Zero-index bound states in the continuum. Phys. Rev. Lett. 121, 263901 (2018).

24. Dong, T. et al. Ultra-low-loss on-chip zero-index materials. Light Sci. Appl. 10, 10 (2021).

25. Davoyan, A. R., Mahmoud, A. M. & Engheta, N. Optical isolation with epsilon-near-zero metamaterials. Opt. Express 21, 3279–3286 (2013).

26. Yang, Y. et al. Magnetically tunable zero-index metamaterials. Photon. Res. 11, 1613–1626 (2023).

27. Zhou, X., Leykam, D., Chattopadhyay, U., Khanikaev, A. B. & Chong, Y. D. Realization of a magneto-optical near-zero index medium by an unpaired Dirac point. Phys. Rev. B 98, 205115 (2018).

28. Wang, N., Zhang, R.-Y., Chan, C. T. & Wang, G. P. Effective medium theory for a photonic pseudospin-1/2 system. Phys. Rev. B 102, 094312 (2020).

29. Feng, F., Wang, N. & Wang, G. P. Magneto-optical double zero-index media and their electromagnetic properties in the bulk. New J. Phys. 24, 113023 (2022).

30. Hancock, S. W., Zahedpour, S., Goffin, A. & Milchberg, H. M. Free-space propagation of spatiotemporal optical vortices. Optica 6, 1547–1553 (2019).

31. Chong, A., Wan, C., Chen, J. & Zhan, Q. Generation of spatiotemporal optical vortices with controllable transverse orbital angular momentum. Nat. Photon. 14, 350–354 (2020).

32. Liu, X. et al. Spatiotemporal optical vortices with controllable radial and azimuthal quantum numbers. Nat. Commun. 15, 5435 (2024).

33. Gui, G., Brooks, N. J., Kapteyn, H. C., Murnane, M. M. & Liao, C.-T. Second-harmonic generation and the conservation of spatiotemporal orbital angular momentum of light. Nat. Photon. 15, 608–613 (2021).

34. Hancock, S. W., Zahedpour, S. & Milchberg, H. M. Second-harmonic generation of spatiotemporal optical vortices and conservation of orbital angular momentum. Optica 8, 594–597 (2021).

35. Wang, H., Guo, C., Jin, W., Song, A. Y. & Fan, S. Engineering arbitrarily oriented spatiotemporal optical vortices using transmission nodal lines. Optica 8, 966–971 (2021).

36. Zhang, H. et al. Topologically crafted spatiotemporal vortices in acoustics. Nat. Commun. 14, 6238 (2023).

37. Liu, W. et al. Exploiting topological darkness in photonic crystal slabs for spatiotemporal vortex generation. Nano Lett. 24, 943–949 (2024).

38. Che, Z. et al. Generation of spatiotemporal vortex pulses by resonant diffractive grating. Phys. Rev. Lett. 132, 044001 (2024).

39. Huo, P. et al. Observation of spatiotemporal optical vortices enabled by symmetry-breaking slanted nanograting. Nat. Commun. 15, 3055 (2024).

40. Ni, X. et al. Three-dimensional reconfigurable optical singularities in bilayer photonic crystals. Phys. Rev. Lett. 132, 073804 (2024).

41. Liu, G.-G. et al. Observation of an unpaired photonic Dirac point. Nat. Commun. 11, 1873 (2020).

42. Haldane, F. D. M. & Raghu, S. Possible realization of directional optical waveguides in photonic crystals with broken time-reversal symmetry. Phys. Rev. Lett. 100, 013904 (2008).

43. Wang, Z., Chong, Y., Joannopoulos, J. D. & Soljačić, M. Observation of unidirectional backscattering-immune topological electromagnetic states. Nature 461, 772–775 (2009).

## Article

44. Ni, J. et al. Multidimensional phase singularities in nanophotonics. Science 374, eabj0039 (2021).

45. Ozawa, T. et al. Topological photonics. Rev. Mod. Phys. 91, 015006 (2019).

46. Xiao, M., Zhang, Z. Q. & Chan, C. T. Surface impedance and bulk band geometric phases in one-dimensional systems. Phys. Rev. X 4, 021017 (2014).

47. Hu, W. et al. Measurement of a topological edge invariant in a microwave network. Phys. Rev. X 5, 011012 (2015).

48. Wang, Q., Xiao, M., Liu, H., Zhu, S. & Chan, C. T. Optical interface states protected by synthetic Weyl points. Phys. Rev. X 7, 031032 (2017).

49. Chua, S.-L., Lu, L., Bravo-Abad, J., Joannopoulos, J. D. & Soljačić, M. Larger-area single-mode photonic crystal surface-emitting lasers enabled by an accidental Dirac point. Opt. Lett. 39, 2072–2075 (2014).

50. Contractor, R. et al. Scalable single-mode surface-emitting laser via open-Dirac singularities. Nature 608, 692–698 (2022).

Publisher's note Springer Nature remains neutral with regard to jurisdictional claims in published maps and institutional affiliations.

Springer Nature or its licensor (e.g. a society or other partner) holds exclusive rights to this article under a publishing agreement with the author(s) or other rightsholder(s); author self-archiving of the accepted manuscript version of this article is solely governed by the terms of such publishing agreement and applicable law.

© The Author(s), under exclusive licence to Springer Nature Limited 2025

## Methods

Comparison of GDZIMs and DZIMs from a spin perspective Spin-1/2 Dirac point and GDZIM. A generic 2D spin-n Dirac-type Hamiltonian reads (n can be any positive integer or half-integer)

$$
\hat {H} _ {\mathrm{spin-} n} = v _ {\mathrm{f}} [ S _ {x} ^ {(n)} k _ {x} + S _ {y} ^ {(n)} k _ {y} ],\tag{5}
$$

where $\upsilon_{f}$ is the Fermi velocity, and the spin-n operators $\{S_{x}^{(n)}, S_{y}^{(n)}, S_{z}^{(n)}\}$ , obeying the angular momentum commutation relation $[S_{i}^{(n)}, S_{j}^{(n)}] = i\epsilon_{ijk}S_{k}^{(n)} (i, j, k \in \{x, y, z\})$ , generate the unique $(2n + 1)$ -dimensional irreducible representation of $\mathfrak{su}(2)$ Lie algebra. The spin-1/2 operators are just the Pauli matrices $S_{i}^{\left(\frac{1}{2}\right)} = \sigma_{i}$ , and $\hat{H}_{\mathrm{spin}-\frac{1}{2}} = \hat{H}_{\mathrm{Dirac}}$ recovers the standard 2D massless Dirac Hamiltonian.

As described in equation (1), a homogeneous gyromagnetic dispersive medium reducing to GDZIM at $\omega_{0}$ generically exhibits linear material dispersion near $\omega_{0}$ , obtained by Taylor expansion. And it is shown that the k · p Hamiltonian at the GDZIM frequency exactly matches the spin-1/2 Dirac Hamiltonian $\hat{H}_{Dirac}$ with the effective Fermi velocity linked to the linear coefficients of the constitutive parameters $^{28}$

$$
\nu_ {\mathrm{f}} = \frac {c}{\omega_ {0} \sqrt {2 c _ {\varepsilon} (c _ {\mu_ {\mathrm{d}}} - c _ {\alpha})}},\tag{6}
$$

where c is the vacuum light speed. For any realistic GDZIM, the positive definiteness of electromagnetic energy density at $\omega_{0}$ requires the following matrix to be positive-definite

$$
\frac {\partial}{\partial \omega} \left[ \omega \left( \begin{array}{c c} \varepsilon_ {z} (\omega) & 0 \\ 0 & \stackrel {{\leftrightarrow}} {{\mu}} _ {T} (\omega) \end{array} \right) \right] \Bigg | _ {\omega_ {0}} > 0,\tag{7}
$$

indicating all eigenvalues of this matrix must be positive:

$$
c _ {\varepsilon} > 0, \quad c _ {\mu_ {\mathrm{d}}} - c _ {\alpha} > 0, \quad 2 \alpha_ {0} + (c _ {\mu_ {\mathrm{d}}} + c _ {\alpha}) \omega_ {0} > 0.\tag{8}
$$

This ensures non-zero linear dispersion coefficients of $\varepsilon_{z}$ and $\stackrel{\leftrightarrow}{\mu}_{T}$ at $\omega_{0}$ and a real positive Fermi velocity $v_{f}>0$ . Consequently, the conical band crossing at the zero-index frequency is a universal feature of all GDZIMs, guaranteed by energy-positive definiteness.

Compatibility with time-reversal symmetry. In ordinary DZIMs, the k·p Hamiltonian at the DZIM frequency $\omega_{0}$ is a spin-1 Dirac-like Hamiltonian (n=1 in equation (5)) (refs. 1,8). Although a spin-1 Dirac-like cone shows two conical bands identical with a spin-1/2 Dirac cone, $\hat{H}_{spin-1}$ is inequivalent to the direct sum of $\hat{H}_{Dirac}$ and an independent flat band. Among various proofs of this inequivalence, we especially focus on their fundamental distinction in compatibility with time-reversal symmetry. For spin-1/2 Dirac points, the following theorem holds.

Theorem 1. A 2D massless Dirac Hamiltonian $\hat{H}_{Dirac}$ does not commute with any bosonic time-reversal operator $T_{b}$ :

$$
[ \hat {H} _ {\mathrm{Dirac}}, \mathcal {T} _ {\mathrm{b}} ] \neq 0.\tag{9}
$$

Thus, in any 2D bosonic system with time-reversal symmetry, spin-1/2 Dirac points cannot appear at time-reversal-invariant momenta. Proof. A bosonic time-reversal operator $\mathcal{T}_{\mathrm{b}}$ is antiunitary and satisfies $\mathcal{T}_{\mathrm{b}}^{2} = 1$ . For a $2 \times 2$ Hamiltonian in $\mathbf{k}$ -space, $\mathcal{T}_{\mathrm{b}}$ takes the general form $\mathcal{T}_{\mathrm{b}} = U\mathcal{K}(\mathbf{k} \to -\mathbf{k})$ with $\mathcal{K}$ denoting complex conjugation. $\mathcal{T}_{\mathrm{b}}^{2} = 1$ requires the unitary part $U$ to satisfy $UU^{*} = 1$ , hence $U = U^{\mathrm{T}}$ is symmetric and can be expressed as (a free $U(1)$ phase is omitted)

$$
U = \exp [ i \varphi \mathbf {n} \cdot \boldsymbol {\sigma} ] = \cos \varphi \sigma_ {0} + i \sin \varphi \mathbf {n} \cdot \boldsymbol {\sigma}, \quad \mathbf {n} \cdot \mathbf {e} _ {y} = 0,\tag{10}
$$

where $\sigma=\{\sigma_{x},\sigma_{y},\sigma_{z}\}$ denotes the vector of Pauli matrices, $\sigma_{0}$ is the 2D identity matrix, and n gives an arbitrary unit vector in the x-z plane of the spin space. Then, we have

$$
\begin{array}{r l} & {[ \hat {H} _ {\mathrm{Dirac}}, \mathcal {T} _ {\mathrm{b}} ] = (U \hat {H} _ {\mathrm{Dirac}} (- \mathbf {k}) ^ {*} - \hat {H} _ {\mathrm{Dirac}} (\mathbf {k}) U) \mathcal {K}} \\ & {\qquad = (- \{U, \sigma_ {x} \} k _ {x} + [ U, \sigma_ {y} ] k _ {y}) \mathcal {K}.} \end{array}\tag{11}
$$

$[\hat{H}_{\mathrm{Dirac}}, \mathcal{T}_{\mathrm{b}}] = 0$ requires the anticommutator $\{U, \sigma_{x}\} = 0$ and the commutator $[U, \sigma_{y}] = 0$ to hold simultaneously. However, from equation (10), $[U, \sigma_{y}] = 0$ only if $U = \sigma_{0}$ (that is, $\varphi = 0$ ), which leads to $\{U, \sigma_{x}\} = 2\sigma_{x} \neq 0$ . As a result, $[\hat{H}_{\mathrm{Dirac}}, \mathcal{T}_{\mathrm{b}}] \neq 0$ is always true. Furthermore, in periodic systems, the $\mathbf{k} \cdot \mathbf{p}$ Hamiltonian at a Dirac point can always be expressed as $\hat{H}_{\mathrm{Dirac}}(\mathbf{k})$ in a properly selected basis. The theorem is established.

By contrast, the $\Gamma$ -point spin-1 Dirac-like point in nonmagnetic DZIMs demonstrates that the spin-1 Hamiltonian preserves time-reversal symmetry. Consequently, in the presence of time-reversal symmetry, regardless of any perturbation applied, it is impossible to achieve an isolated spin-1/2 Dirac point at $\Gamma$ by displacing the flat band from a spin-1 cone.

Phase diagram. Using a phase diagram (Extended Data Fig. 1a), we show how a 2D square PC composed of dielectric cylinders evolves from an ordinary DZIM to GDZIMs across a 2D parameter space spanned by the radius of the cylinder $r_{c}$ and the gyromagnetic coupling $\alpha$ in the permeability tensor of the cylinder:

$$
\stackrel {\leftrightarrow} {\mu} _ {T} = \left( \begin{array}{c c} \mu_ {d} & - \mathrm{i} \alpha \\ \mathrm{i} \alpha & \mu_ {\mathrm{d}} \end{array} \right),\tag{12}
$$

which is assumed to be dispersiveless for simplicity. We focus on the three $\Gamma$ -point modes, that is, two in-plane magnetic dipole ( $M$ -dipole) modes and an out-of-plane electric dipole ( $E_z$ -dipole) mode, shown in Extended Data Fig. 1b. For nonmagnetic PCs with $\alpha = 0$ (the top blue edge of the phase diagram), the $D_{4h}$ (4/mmm) point group protects the degeneracy of two $M$ -dipoles. However, when an external magnetic field is applied, the symmetry of the PC is reduced to the magnetic point group $4/mm'm'$ , and the degenerate $M$ -dipoles undergo Zeeman splitting, resulting in two nondegenerate $M$ -dipoles with opposite circular polarizations. Therefore, the three $\Gamma$ -point modes are generically nondegenerate when $\alpha \neq 0$ and follow the one-dimensional irreducible co-representations $\Gamma_1^- \oplus \Gamma_4^+ \oplus \Gamma_3^+$ of $4/mm'm'$ (ref. 51) (Extended Data Fig. 1d). Based on the order of these three modes, the gyromagnetic PCs form three distinct fully gapped phases within the lower-half ( $r_c$ , $\alpha$ ) plane with $\alpha < 0$ . At the triple-phase point, all three modes accidentally degenerate at a spin-1 Dirac-like point and the PC behaves as an ordinary DZIM with $\varepsilon^{\mathrm{eff}} = \mu^{\mathrm{eff}} = 0$ (ref. 1). At the phase-transition boundary between phase I (phase III) and phase II, the $E_z$ -dipole and LCP (RCP) $M$ -dipole become accidentally degenerate, forming a spin-1/2 Dirac point, whereas the RCP (LCP) $M$ -dipole mode remains a singlet state on the third band. On these two phase-transition lines, the PCs behave as GDZIMs with $\varepsilon_z^{\mathrm{eff}} = \det[\stackrel{\leftrightarrow}{\mu}_{\mathrm{T}}^{\mathrm{eff}}] = 0$ at the Dirac point frequency. Extended Data Fig. 1c shows typical band structures for PCs in different regions of the phase diagram. Note that the phase diagram in the half-plane of $\alpha > 0$ is symmetric to that in the $\alpha < 0$ half-plane, which is not shown here for brevity.

The phase diagram indicates that GDZIMs can be achieved by tuning the filling ratio $r_{c}/a$ and the gyromagnetic strength. The grey arrows in Extended Data Fig. 1a provide a roadmap for the evolution from an ordinary DZIM to GDZIMs. Starting with a reciprocal PC exhibiting a spin-1 Dirac-like point, the application of an external magnetic field drives the PC into phase II, creating a magnetic-field-induced single-negative metamaterial studied in ref. 26. Further adjustment of the filling ratio induces a phase transition from phase II to phase I, in which the transition point precisely determines the parameters required for GDZIM realization.

## Bulk-spatiotemporal vortex correspondence

Here we provide the rigorous statement of the bulk-spatiotemporal vortex correspondence (Theorem 4) and sketch the proof steps

## Article

following Extended Data Fig. 4. More detailed proof and discussions are provided in Supplementary Note 4. We restrict our discussions to 2D homogeneous and Hermitian media respecting mirror-z symmetry so that 2D waves can always be separated as decoupled TM and TE components. For conciseness, a medium satisfying these conditions is referred to as a target medium in the following discussion.

Step 1: mid-gap reflection by semi-infinite media. As shown in Extended Data Fig. 4a, we first consider plane wave reflection at the interface between two semi-infinite homogeneous media, in which the background medium (grey) with $\varepsilon_{1},\mu_{1}$ is isotropic, conservative and transparent, whereas the target medium (green) with a constitutive tensor $\hat{M}_{2}(\omega ,\rho)$ (see its general form in Supplementary Note 4.C) can exhibit trivial or non-trivial topology controlled by an internal parameter $\rho$ . For TM plane waves incident from the background medium side, the reflection coefficient by the semi-infinite target medium is $R_{\mathrm{semi}}(k_y,\omega ;\rho)$ .

Suppose at a critical parameter $\rho = \rho_0$ , the bulk band gap of the target medium closes at a single twofold degeneracy, in which the topological phase transition occurs. This gap-closing point forms a synthetic Weyl point at $\widetilde{\mathbf{k}}_0 = (k_{x0}, k_{y0}, \rho_0)$ in the 3D synthetic space spanned by the right-handed frame ( $k_x, k_y, \rho$ ). The charge of the Weyl point is equal to the Chern number change across the phase transition: $\mathrm{Ch}_{\mathrm{Weyl}} = \mathrm{Ch}(\rho > \rho_0) - \mathrm{Ch}(\rho < \rho_0)$ , as shown in Extended Data Fig. 4b. To obtain a well-defined projected band gap around the Weyl point, we require the synthetic Weyl cone to be of type I in the $k_x$ direction. And we can always shift and tilt the band structure according to the mid-gap frequency $\omega_{\mathrm{mid}}(k_y, \rho) = \frac{1}{2} (\omega_{\mathrm{upper}}(k_y, \rho) + \omega_{\mathrm{lower}}(k_y, \rho))$ , where $\omega_{\mathrm{upper}}(k_y, \rho)$ and $\omega_{\mathrm{lower}}(k_y, \rho)$ denote the upper and lower band edges of the projected bands in the $(k_y, \rho)$ plane. Then, the shifted band dispersion $\Delta \omega(k_y, \rho) = \omega(k_y, \rho) - \omega_{\mathrm{mid}}(k_y, \rho)$ always shows as an ideal type-I Weyl cone.

Extended Data Fig. 4b shows that the reflection phase of $R_{\mathrm{semi}}$ within the bulk band gap forms a vortex around the synthetic Weyl point—a well-documented bulk-reflection correspondence phenomenon for semi-infinite Weyl materials[48,52-54,55]. However, a general mathematical proof of this phenomenon for photonic full-wave systems is still lacking. Specifically, focusing on the synthetic Weyl point for 2D TM modes in homogeneous media, we have the following rigorous theorem.

Theorem 2: mid-gap reflection vortex by a semi-infinite medium. Consider the reflection coefficient $R_{\mathrm{semi}}(k_y, \omega; \rho)$ for a semi-infinite homogeneous target medium $\stackrel{\leftrightarrow}{M}_2(\omega; \rho)$ , evolved along an anticlockwise loop $C_{\mathrm{gap}}$ in the 3D $(k_y, \rho, \omega)$ parameter space. Provided that the loop and the material satisfy

1. $C_{gap}$ lies within the mid-gap of $\stackrel{\leftrightarrow}{M}_{2}$ ;

2. $C_{gap}^{gap}$ encircles the phase-transition point (synthetic Weyl point) of $\stackrel{\leftrightarrow}{M}_{2}$ located at $\widetilde{k}_{0}=(k_{xQ},k_{y0},\rho_{0})$ ;

3. The band crossing at $\widetilde{K}_{0}$ is linear and of type I in the $k_{x}$ direction; the reflection phase winding number along $C_{gap}$ is identical to the topological charge of the synthetic Weyl point:

$$
\begin{array}{c} v (R _ {\text {semi}}, C _ {\text {gap}}) = \oint_ {C _ {\text {gap}}} d \arg [ R _ {\text {semi}} (k _ {y}, \omega ; \rho) ] \\ = \mathsf {C h} _ {\text {Weyl}} = \mathsf {C h} _ {\rho > \rho_ {0}} - \mathsf {C h} _ {\rho <   \rho_ {0}}. \end{array}\tag{13}
$$

The complete proof of the theorem is provided in Supplementary Note 4.D, in which we show that this bulk–reflection correspondence is closely related to the well-established bulk–band inversion surface correspondence in the dynamical topological classification theory $^{56-60}$ .

Step 2: mid-gap reflection by a homogeneous slab with finite thickness. Next, we examine the relationship between the reflection from infinite homogeneous media and finitely thick homogeneous slabs (Extended Data Fig. 4c). Direct analysis yields two lemmas (see proofs in Supplementary Note 4.A).

Lemma 1. For a 2D photonic slab made of a homogeneous target medium $\stackrel{\leftrightarrow}{M}_{2}(\omega)$ , zero reflection of plane waves incident on this slab can take place only within the projection of the bulk bands of the medium onto the $(k_{y},\omega)$ plane. In other words,

$$
R (k _ {y}, \omega ; d) \neq 0,\tag{14}
$$

for $d > 0$ and $(k_y, \omega) \in \text{projected bulk band gap of } \stackrel{\leftrightarrow}{M}_2(\omega)$ .

Lemma 2. For a 2D photonic slab made of a homogeneous target medium $\stackrel{\leftrightarrow}{M}_{2}(\omega)$ , the slab reflection coefficient $R(k_{y},\omega;d)$ converges to the semi-infinite reflection coefficient $R_{\mathrm{semi}}(k_y,\omega)$ between the background medium and the target medium composing the slab, as the thickness of the slab, $d$ , tends to infinity:

$$
\lim _ {d \to \infty} R (k _ {y}, \omega ; d) = R _ {\text { semi }} (k _ {y}, \omega),\tag{15}
$$

for any $(k_y, \omega) \in$ projected bulk band gap of $\stackrel{\leftrightarrow}{M}_2(\omega)$ .

According to Lemma 2, the reflection spectrum within the bulk band gap of a slab with sufficiently large but finite thickness should approach that of a semi-infinite medium. Furthermore, by Lemma 1, altering the thickness of the slab will not cause reflection zeros to depart from or enter the bulk band projection. Consequently, the phase winding number around the Weyl cone remains constant regardless of slab thickness variations, and it will be equal to the outcome of the corresponding semi-infinite reflection evaluated along the same loop:

$$
\begin{array}{c} \oint_ {C _ {\text {gap}}} d \arg [ R (k _ {y}, \omega ; \rho , d) ] = \lim _ {d \to \infty} \oint_ {C _ {\text {gap}}} d \arg [ R (k _ {y}, \omega ; \rho , d) ] \\ = \oint_ {C _ {\text {gap}}} d \arg [ R _ {\text {semi}} (k _ {y}, \omega ; \rho) ]. \end{array}
$$

This result indicates that Theorem 2 can be generalized to the finitely thick slab reflection, as depicted in Extended Data Fig. 4d.

Theorem 3: mid-gap reflection vortex by a homogeneous slab. Consider the reflection coefficient $R(k_y, \omega; \rho, d)$ for a homogeneous target medium slab with thickness $d$ evolved along an anticlockwise loop $C_{\text{gap}}$ in the 3D $(k_y, \rho, \omega)$ parameter space. Provided that the three premises in Theorem 2 are still satisfied, the phase winding number of $R(k_y, \omega; \rho, d)$ along $C_{\text{gap}}$ is always identical to the topological charge of the synthetic Weyl point, irrespective of the slab's thickness $d$ :

$$
\begin{array}{c} \nu (R, C _ {\text { gap }}) = \oint_ {C _ {\text { gap }}} d \arg [ R (k _ {y}, \omega ; \rho , d) ] \\ = \mathsf {C h} _ {\text { Weyl }} = \mathsf {C h} _ {\rho > \rho_ {0}} - \mathsf {C h} _ {\rho <   \rho_ {0}}. \end{array}\tag{16}
$$

Step 3: Wick rotation of the reflection vortex by a homogeneous slab. The immediate implication of Theorem 3 is that the surface projection of the topological phase-transition point $(k_{y0}, \omega_{0}, \rho_{0})$ of a homogeneous medium always corresponds to a reflection phase singularity and, consequently, a reflection zero point $(R(k_{y0}, \omega_{0}; \rho_{0}) = 0)$ for slabs of any thickness made from such a medium. We restrict the discussion to elementary topological phase transitions with $v(R, C_{\mathrm{gap}}) = \mathrm{Ch}_{\mathrm{Weyl}} \in \{\pm 1\}$ . As (1) the elementary phase vortices generically form continuous lines in a 3D parameter space and (2) the reflection zeros of a slab are confined within the bulk band projection according to Lemma 1, we can conclude that a zero-reflection line always passes through the synthetic Weyl point from the interior region of the projected bulk bands (that is, the Weyl cone) in the 3D $(k_{y}, \rho, \omega)$ space, as shown by the red line in Extended Data Fig. 4e. Therefore, the mid-gap loop $C_{gap}$ encircling the synthetic Weyl point must also enclose the zero-reflection line.

Next, we perform Wick rotation to the horizontal mid-gap loop such that it continuously deforms to a vertical loop located in the $\rho = \rho_{0}$ plane. As long as the slope of the singularity line of zero reflection is finite at the synthetic Weyl point, that is, $|d\Delta\omega_{ZR}/d\rho|_{\rho_{0}}| < \infty$ , it is always possible to choose a specific chirality for the Wick rotation so that the loop does not intersect the singularity line throughout the Wick rotation process. For instance, if $d\Delta\omega_{ZR}/d\rho|_{\rho_{0}}>0$ as the case shown in Extended Data Fig. 4e, the Wick rotation should be performed in a clockwise direction, then the obtained vertical loop has a handedness opposite to the positive surface normal of the vertical $\rho=\rho_{0}$ plane (Extended Data Fig. 4f). As a result, we obtain the relationship between the phase winding numbers along a mid-gap anticlockwise loop $C_{gap}$ and a vertical anticlockwise loop $C_{D}$ encircling the singularity lines:

$$
\nu (R, C _ {\mathrm{D}}) = - \operatorname{sgn} \left[ \frac {\mathrm{d} \Delta \omega_ {\mathrm{ZR}}}{\mathrm{d} \rho} \Bigg | _ {\rho_ {0}} \right] \nu (R, C _ {\text { gap }}).\tag{17}
$$

Combining this result and Theorem 3, we obtain the main theorem of this work.

Theorem 4: bulk–spatiotemporal vortex correspondence. Consider a homogeneous medium slab with thickness $d < \infty$ . Provided that the medium $\vec{M}_{2}(\omega, \rho_{0})$ comprising the slab satisfies the premises

P1. it is Hermitian and mirror-z symmetric;

P2. its TM band structure has a twofold degeneracy point at $D_{0} = (k_{x0}, k_{y0}, \rho_{0}, \omega_{0})$ ;

P3. the degeneracy $D_{0}$ may arise as the critical point of an elementary topological phase-transition process controlled by a parameter $\rho$ , that is, $(\mathrm{Ch}_{\rho>\rho_{0}}-\mathrm{Ch}_{\rho<\rho_{0}})\in\{\pm1\}$ ;

then the following conclusions about the TM reflection $R(k_{y},\omega ;\rho_{0},d)$ of the slab are always established:

C1. The surface projection of the degeneracy $D = (k_{y0}, \omega_{0})$ must support zero reflection irrespective of the background medium and the thickness of the slab, that is,

$$
R (k _ {y 0}, \omega_ {0}; \rho_ {0}, d) \equiv 0.\tag{18}
$$

C2. For almost arbitrary isotropic, conservative and transparent background medium, the reflection phase around the projected degeneracy point D forms a spatiotemporal vortex on the $(k_{y}, \omega)$ plane, and its vortex charge is a $Z_{2}$ topological invariant associated with the Chern number change across the phase transition:

$$
\nu (R, C _ {\mathrm{D}}) = \left(\operatorname{Ch} _ {\rho > \rho_ {0}} - \operatorname{Ch} _ {\rho <   \rho_ {0}}\right) \bmod 2 = \pm 1,\tag{19}
$$

where $C_{D}$ is any anticlockwise loop on the $(k_{y}, \omega)$ plane that encircles only one singularity at D, and the sign ± depends on the specific background medium.

C3. In the 3D parameter space $(k_{y}, \rho, \omega)$ , there always exists a line of zero reflection, characterized by the one-parameter equation $(k_{y}, \rho, \omega) = (k_{\mathrm{ZR}}(\rho), \rho, \omega_{\mathrm{ZR}}(\rho))$ , traversing the degeneracy point from the interior of the projected bulk bands. Provided that $|\mathrm{d}\Delta\omega_{\mathrm{ZR}}/\mathrm{d}\rho|_{\rho_{0}} \neq \infty$ with $\Delta\omega_{ZR} = \omega_{ZR} - \omega_{mid}$ for a certain background medium, the reflection spatiotemporal vortex charge for this background medium can be exactly determined by

$$
\nu (R, C _ {\mathrm{D}}) = \operatorname{sgn} \left[ \frac {\mathrm{d} \Delta \omega_ {\mathrm{ZR}}}{\mathrm{d} \rho} \Bigg | _ {\rho_ {0}} \right] \left(\mathrm{Ch} _ {\rho <   \rho_ {0}} - \mathrm{Ch} _ {\rho > \rho_ {0}}\right).\tag{20}
$$

Further discussions of this bulk–spatiotemporal vortex correspondence, including its comparison to conventional bulk-boundary correspondences and potential generalizations, are presented in Supplementary Note 4.

Bulk-reflection-edge correspondence. As a direct application of the bulk-spatiotemporal vortex correspondence, the vortical reflection phase explains why a bulk Dirac point nearly always induces a band of edge states that are localized on a hard-wall boundary and connect to its projection point. Specifically, when an insulated boundary encases the left end of a GDZIM, with a free-space reflection coefficient $R_{\mathrm{bdy}}(k_y,\omega) = \mathrm{e}^{\mathrm{i}\phi_{\mathrm{bdy}}}$ , the phase matching condition for edge states subsisting at this boundary is $R_{\mathrm{bdy}}(k_{y}, \omega)R(k_{y}, \omega) = 1$ (refs. 46,61). Therefore, as long as the boundary is time-reversal invariant, say, $\phi_{\mathrm{bdy}}(k_{y}, \omega)$ takes any constant value, the vortical reflection phase of the GDZIM ensures that the phase matching can always be satisfied along a half-curve terminating at the vortex centre, that is, the projected Dirac point. Extended Data Fig. 5 shows the examples of a perfect electric conductor (PEC) boundary. As $R_{bdy} = -1$ for PEC, the edge states on the boundary should disperse along the constant reflection phase curve $\arg(R) = \pi$ (Extended Data Fig. 5a), which precisely predicts the actual edge band obtained through simulations (Extended Data Fig. 5b). The experimental test of the edge states on a PEC boundary (Extended Data Fig. 5c) further verifies the approach.

## Simulation and experimental realization of GDZIMs

For the actual PC in experiments, the magnetized YIG rods have a constant relative permittivity $\varepsilon_{z}=13$ and a dispersive permeability tensor (equation (12)) with

$$
\mu_ {\mathrm{d}} (f) = 1 + \frac {(f _ {\mathrm{l}} - \mathrm{i} \beta f) f _ {\mathrm{m}}}{(f _ {\mathrm{l}} - \mathrm{i} \beta f) ^ {2} - f ^ {2}}, \alpha (f) = \frac {f f _ {\mathrm{m}}}{(f _ {\mathrm{l}} - \mathrm{i} \beta f) ^ {2} - f ^ {2}}.\tag{21}
$$

Here, the Larmor frequency $f_{1}=\mu_{0}\gamma H_{0}/2\pi=2.52$ GHz under the effective external magnetic field $H_{0}=900$ Oe with the gyromagnetic ratio $\gamma=2\pi\times2.8$ MHz G $^{-1}$ , and $f_{m}=\mu_{0}\gamma M_{s}/2\pi=5.18$ GHz with the saturation magnetization of YIG $M_{s}=1,850$ Oe. The value of damping $\beta$ is about $10^{-3}$ in YIG, having negligible influence at our operating frequency and thus omitted from the main text calculations. The intrinsic dispersion in equation (21) has been incorporated into the band structure calculations by treating the magnetic moment of the YIG as an auxiliary field. More discussions of loss and dispersive effects appear in Supplementary Note 6.A.

The experimental setup for measuring the projected bands of the GDZIM is shown in Extended Data Fig. 2. We fabricated samples (Extended Data Fig. 2e) comprising of $N_{x} \times N_{y} = 10 \times 30$ arrays of YIG rods within a PEC parallel-plate waveguide. Through applying uniformly distributed weights (>10 kg) on the top metal plate, we ensured intimate and uniform contact between all YIG pillars and the metallic surfaces during the experimental process. Permanent magnets are placed beneath the metal tape (lower metallic surface of the waveguide), as shown in Extended Data Fig. 2f. Field distributions are extracted from the S-parameters, collected by a vector network analyser Keysight N5234B (Extended Data Fig. 2g) connected to the source and probe antennas. Then, the projected band structures along the $k_{y}$ direction are obtained by Fourier transformation of the probed fields along the scan line (Extended Data Fig. 2h).

To demonstrate the band inversion process (Extended Data Fig. 2a), we varied the lattice constant $a$ across different samples. Extended Data Fig. 2b–d shows the measured projected bulk bands for the gyromagnetic PCs with $a = 16 \, \text{mm}$ , $17.2 \, \text{mm}$ and $18.4 \, \text{mm}$ , respectively. The simulated local density of states along the field scan line ( $x = 0$ ) shows good agreement with the measurements (see details of the LDOS calculation in Supplementary Note 6.B). At $a = 17.2 \, \text{mm}$ , we observe the unpaired Dirac point at the phase-transition critical point. Note that the observed minor gap for $a = 17.2 \, \text{mm}$ originates from the finite-size effect (Supplementary Note 6.B). This finite-size-induced gap does not influence the observed reflection phase or the spatiotemporal vortex generation, as our bulk–spatiotemporal vortex correspondence is strictly established for slabs with finite thicknesses.

## Measurement of reflection phase vortices

We present an efficient method for experimentally detecting the ultrarobust reflection phase vortices, which outperforms the usual angle-resolved spectrometry by easily obtaining reflection spectra at near-zero angles. As shown in Extended Data Fig. 3a, we first probed the total electric field $E_{z}^{\mathrm{prob}}(\mathbf{r}) = E_{z}^{\mathrm{in}}(\mathbf{r}) + E_{z}^{\mathrm{r}}(\mathbf{r})$ spreading in the region

## Article

between the source and the sample. Using discrete Fourier transform (Supplementary Note 6.C), we then obtained the plane-wave components, $\widetilde{E}_z^{\mathrm{prob}}(\mathbf{k})$ , of the probed field distributed on the light cone of free space (Extended Data Fig. 3b). The plane-wave components with opposite signs of $k_x$ have different origins: all the $k_x > 0$ components represent the incident fields from the source, whereas all the $k_x < 0$ components correspond to the fields reflected from the sample slab. Hence, for any given $k_y$ , the reflection coefficient can be extracted as $R(k_y,\omega) = \widetilde{E}_z^{\mathrm{prob}}(-|k_x|,k_y)/\widetilde{E}_z^{\mathrm{prob}}(|k_x|,k_y)$ with $|k_x| = \sqrt{(\omega/c)^2 - k_y^2}$ . Scanning across the frequency range of 8–12 GHz, we finally obtained the reflection spectra exhibited in Extended Data Fig. 3d for samples with one to four layers. The measured spectra of all samples manifest, with sufficient accuracy, that a reflection phase vortex always arises at the projected Dirac point, irrespective of the number of layers (see error analysis in Supplementary Note 7.B). These agree well with the theoretical results presented in Extended Data Fig. 3c and thus demonstrate the ultrarobustness of the reflection vortex against the variation of GDZIM slab thickness.

## Generation and measurement of STVPs

The experimental setup for generating and characterizing the reflected STVPs is shown in Extended Data Fig. 6. A temporal Gaussian microwave pulse, generated by an arbitrary waveform generator (Keysight M9502A) and amplified by a custom-made power amplifier, is launched into a metallic parallel-plate waveguide. On propagation through the PEC horn and metamaterial lens (Supplementary Note 6.D), the pulse is shaped into a planar wavefront with a Gaussian intensity profile along the y axis. A narrow slot cut along the scan line (dashed black line) in the upper waveguide plate allows a probe antenna to measure the real-time electric field within the waveguide. Synchronization of measurements at different points along the scan line is achieved by using a reference signal from the arbitrary waveform generator to trigger the real-time oscilloscope (Keysight MSOV334A). In Extended Data Fig. 6c,d, the recorded temporal waveform along the scan line shows that the incident and reflected pulses are well-separated in time. Removing the fast oscillatory dynamical phases, we can obtain the measured STVPs in Fig. 4e.

## Data availability

The data that support the results in this paper are available at GitHub (https://github.com/xcuiad/GDZIM-data).

## Code availability

The code used to evaluate the conclusions in the paper is available from the corresponding authors upon request.

51. Bilbao Crystallographic Server. Irreducible Corepresentations of the Magnetic Point Group 4/mm'm' (N. 15.6.58) https://www.cryst.ehu.es/cgi-bin/cryst/programs/corepresentations\_point.pl?magnum=15.6.58 (2020).

52. Liu, Y., Yu, Z.-M., Xiao, C. & Yang, S. A. Quantized circulation of anomalous shift in interface reflection. Phys. Rev. Lett. 125, 076801 (2020).

53. Cheng, H. et al. Vortical reflection and spiraling Fermi arcs with Weyl metamaterials. Phys. Rev. Lett. 125, 093904 (2020).

54. Wang, H., Zhou, L. & Chong, Y. D. Floquet Weyl phases in a three-dimensional network model. Phys. Rev. B 93, 144114 (2016).

55. Guo, Q. et al. Three dimensional photonic Dirac points in metamaterials. Phys. Rev. Lett. 119, 213901 (2017).

56. Zhang, L., Zhang, L., Niu, S. & Liu, X.-J. Dynamical classification of topological quantum phases. Sci. Bull. 63, 1385 (2018).

57. Zhang, L., Zhang, L. & Liu, X.-J. Dynamical detection of topological charges. Phys. Rev. A 99, 053606 (2019).

58. Yi, C.-R. et al. Observing topological charges and dynamical bulk-surface correspondence with ultracold atoms. Phys. Rev. Lett. 123, 190603 (2019).

59. Hu, H. & Zhao, E. Topological invariants for quantum quench dynamics from unitary evolution. Phys. Rev. Lett. 124, 160402 (2020).

60. Wang, Z.-Y. et al. Realization of an ideal Weyl semimetal band in a quantum gas with 3D spin-orbit coupling. Science 372, 271–276 (2021).

61. Cui, X., Zhang, R.-Y., Zhang, Z.-Q. & Chan, C. T. Photonic $Z_{2}$ topological Anderson insulators. Phys. Rev. Lett. 129, 043902 (2022).

Acknowledgements We thank J. Pendry, K. Ding, Y. Lai, G. Ma and Y. Wu for discussions. We also thank K. M. Shum for helping with the experiments. This work is supported by the Research Grants Council of Hong Kong (16310422, AoE/P-502/20, JLFS/P-603/24 and CityU 21207824) and by the National Natural Science Foundation of China (12174263).

Author contributions R.-Y.Z., X.C., N.W. and C.T.C. conceived the idea. R.-Y.Z., X.C., W.L. and N.W. developed the theory. X.C. performed numerical simulations. X.C., R.-Y.Z., M.W. and D.W. designed and carried out the static experimental measurements. Y.-S.Z., X.C., G.-B.W. and J.C. designed and carried out the time-domain experiments. R.-Y.Z., X.C., Y.-S.Z. and C.T.C. wrote the paper. C.T.C., G.-B.W. and Z.-Q.Z. supervised the project. All authors contributed to the discussions.

Competing interests The authors declare no competing interests.

## Additional information

Supplementary information The online version contains supplementary material available at https://doi.org/10.1038/s41586-025-08948-6.

Correspondence and requests for materials should be addressed to Xiaohan Cui, Neng Wang, Geng-Bo Wu or C. T. Chan.

Peer review information Nature thanks Lei Bi, Simon Horsley and Alex Krasnok for their contribution to the peer review of this work. Peer reviewer reports are available. Reprints and permissions information is available at http://www.nature.com/reprints.

![](images/834bb2630a80be365ef5482ae7cd64c88769086245c791cb84dc0ff3934f3338.jpg)

b
![](images/b0ebe15d9f457a6f8ffdd48c058803bfc919a8fb42dc18266142ee0045b47333.jpg)

① DZIM
![](images/d0fc50ddb2ae474fe1fd2ab52fbf4fd76244c38631639e6002e10ddaf60409c7.jpg)

② Phase-II
![](images/1df885fe6bfa7fb7fc7de2c9c4fe4b7bd0c45fa289477a0e4912c1fa7db07770.jpg)

③ GDZIM
![](images/8e9c9722ff84ae3f7cb41bec0596381154ec13862ccbcd5375996233beb447df.jpg)

④ Phase-I
![](images/672673516c938f18f9fad9991dc2897ace80301583e1be48c5ea28591325fe15.jpg)

⑤ GDZIM
![](images/9537ae26e07d31a8c2a91b73c3b5c5d7808d3196cbf94db1f677519592db1af0.jpg)

⑥ Phase-III
![](images/e97fcd6979c553300f750a13c23da864bbfa03037c2abb003e2a1dcf44befc2a.jpg)

d

<table><tr><td rowspan="2">Modes</td><td colspan="2">Irreducible co-representations</td><td colspan="8">Unitary symmetries</td></tr><tr><td> $\Gamma$  labels</td><td>Mulliken notation</td><td>1</td><td> $2_{001}$ </td><td> $4^{+}_{001}$ </td><td> $4^{-}_{001}$ </td><td> $\overline{1}$ </td><td> $m_{001}$ </td><td> $\overline{4}^{+}_{001}$ </td><td> $\overline{4}^{-}_{001}$ </td></tr><tr><td> $E_z$ -dipole</td><td> $\Gamma_1^-$ </td><td> $A_u$ </td><td>1</td><td>1</td><td>1</td><td>1</td><td>-1</td><td>-1</td><td>-1</td><td>-1</td></tr><tr><td>RCP M-dipole</td><td> $\Gamma_3^+$ </td><td> $^2E_g$ </td><td>1</td><td>-1</td><td>i</td><td>-i</td><td>1</td><td>-1</td><td>i</td><td>-i</td></tr><tr><td>LCP M-dipole</td><td> $\Gamma_4^+$ </td><td> $^1E_g$ </td><td>1</td><td>-1</td><td>-i</td><td>i</td><td>1</td><td>-1</td><td>-i</td><td>i</td></tr></table>

Extended Data Fig. 1 | Phase diagram depicting the evolution from an ordinary DZIM to a GDZIM. a, Phase diagram of gyromagnetic PCs in the $(r_c, \alpha)$ plane. Fixed PC parameters: $a = 17.2\mathrm{mm}$ , $\varepsilon_z = 13$ , $\mu_d = 1$ . The phase diagram divides into three phases based on the ordering of the $F$ -point modes (marked by red, green and blue dots). At phase transition boundaries, two bands become degenerate, forming a spin- $\frac{1}{2}$ Dirac point. The triple-phase point forms a spin-1 Dirac-like point. b, Mode profiles corresponding to the red, green, and blue dots: an out-of-plane electric dipole, and in-plane right-circularly polarized

(RCP) and left-circularly polarized (LCP) magnetic dipoles. $\Gamma_{1}^{-}, \Gamma_{3}^{+}$ and $\Gamma_{4}^{+}$ denote the corresponding irreducible co-representations of the magnetic point group $4/mm'm'$ . c, Six representative band structures corresponding to different phases and critical points in the phase diagram. d, Character table of the PC's magnetic point group $4/mm'm'$ for the three F-point modes in (b). Only the unitary symmetries of the group are listed. The full representation table including both unitary and antiunitary symmetries can be found in ref. 51.

a
![](images/c3775f6f6cff093a89db3298441f1d8f38c509ca46d31a93d1c0ee71bd5b6007.jpg)

![](images/9e8ad661947d8b6a30b2212783c95a3fc39dc177e041a58e2d571254f45995a5.jpg)

c
![](images/e426fe5a8d63eb91e1466736e4eb06b8dd14f3c4f3615cff7f6ae79b4b80e0f1.jpg)

d
![](images/edc77d44dd6272290ec7ef872c13b854163f6674c0178ce3740a9c60748fbd8e.jpg)

e
![](images/3a031931fb5748bffe4e81ccb4a5e7b60796e27edc6d31ab886ac91bc0c8950c.jpg)

g
![](images/801056f3d8c6554c5a00c8f13584bbed2872977503029c778dfffb458416d851.jpg)
h

![](images/b35dc271809bd7bbb266607bebbd9252d5f5b12ff6085c5f9daf5c8ebf2d3227.jpg)

![](images/2afcf7ac48226d7329b88c0a00b549fc1da78610c1325abd5929658011cd24c1.jpg)
Extended Data Fig. 2 | Experimental realization of GDZIMs. a, The eigenfrequencies of the gyromagnetic PC at the $\Gamma$ point as functions of the lattice constant (a), showing the band inversion process of the second and third bands controlled by the lattice constant. b-d, Projected band structures shown through (left panel) simulated local density of states (LDOS) along $x = 0$ and (right panel) experimentally measured data for the PC ribbons ( $N_x \times N_y = 10 \times 30$ ) with lattice constants (b) $a = 16$ mm, (c) $a = 17.2$ mm and (d) $a = 18.4$ mm, respectively. e, Photo of the YIG pillars (height: 4 mm, radius: 3.1 mm) positioned between two 3-mm-thick parallel PEC plates—an aluminum top plate and an acrylic bottom plate
covered with metallic tape. f, Photo of permanent magnets (radius: 5 mm, height: 3 mm, surface magnetic strength: 3340 G) positioned beneath a whole piece of metallic tape (thickness: 0.2 mm), embedded in drilled holes within an acrylic sheet. g, h, Field scanning setup: a source antenna is inserted into the parallel-plate waveguide, and the excited fields are detected by a probe antenna along a straight slot in the center of the cover plate (along x = 0). The source and probe ports are connected to (g) a vector network analyzer, which can give the measured field by collecting the S-parameters. The scanning spatial step is $d_{y} = a/2$ , and the frequency range is 8–12 GHz.

d
b
a
![](images/b4f8bb62587a5214fecf651fd5ed15a4647f6ac99bbc625ec1dba4a65b16487c.jpg)

![](images/a99dd7fec8ac429e9591c511dd4b5a53918e7fe333e67307396c9efc645f6575.jpg)

![](images/803dc37f36bdf962850a8781bd5ddec27b146af4b5a6178bbe581df0eff13536.jpg)

![](images/6165b0ab9925c37df32a55f5d35e81cc047066f5744242afd8f67b330d2ae85e.jpg)
Extended Data Fig. 3 | Experimental measurement of ultrarobust reflection phase vortices. a, Diagram of the interior view of the measurement setup, with the system sandwiched between two parallel PEC plates. A source antenna is placed on the left of the GDZIM sample and microwave absorbers are tiled around the periphery. A probe antenna, connected to a vector network analyzer, detects the total field (with spatial sampling step $d_{x}=d_{y}=0.5a$ ) in the region ( $10a\times30a$ ) between the source and the sample within the frequency range of 8–10 GHz
with a step of 0.02 GHz. The colormap depicts an example of the total field pattern $E_z(x,y)$ at 10 GHz. b, The Fourier spectra of the incident $E_z^{\mathrm{in}} = \widetilde{E}_z(k_x > 0)$ and reflected field $E_z^{\mathrm{r}} = \widetilde{E}_z(k_x < 0)$ measured at 10 GHz. c,d, Theoretical (c) and experimental (d) results for the phase and amplitude of the reflection spectrum $R(k_y,\omega)$ for GDZIM slabs with different numbers of layers. The reflection vortices are fixed at the Dirac frequency $f_0 = 10.14\mathrm{GHz}$ and ky = 0 (black dashed line) for different slabs.

Step 1. Bulk-isofrequency reflection correspondence for semi-infinite Weyl medium [Theorem 2]

![](images/05cf35276ce3a3f9dbe85331b174d1f0e6acdf6de15cc60879bc27afd1919776.jpg)

$R_{\mathrm{slab}} \neq 0$ inside gap [Lemma 1]

m $\mathsf{R}_{\mathrm{slab}} = R_{\mathrm{semi}}$ inside gap [Lemma 2]

a Semi-infinite reflection

![](images/18e99c2967c85a838697382ac8ddc333a7a8a08ca27b2167141c9427bc0d6201.jpg)

Step 2. Bulk-isofrequency reflection correspondence for a homogeneous slab [Theorem 3]

![](images/61faf5db236a31c02456d16312070c484a5bb6eedf8803328759158564d9a8a9.jpg)

c Slab reflection

![](images/451f9752089bef3c131689665cf0ddb3e3a0ed79052be3682625fb45649e6b6f.jpg)

e Wick rotation of vortex

Step 3. Bulk-spatiotemporal vortex correspondence [Theorem 4]

$$
v (R, C _ {D}) = \operatorname{sgn} \left[ \frac {d \Delta \omega_ {Z R}}{d \rho} \Bigg | _ {\rho_ {0}} \right] \left(\mathrm{Ch} _ {\rho <   0} - \mathrm{Ch} _ {\rho > 0}\right)
$$

![](images/21968aceef2dad6a2af15a815cf4f169a8ce12a99b988e54818eff8cc4d4e177.jpg)

Extended Data Fig. 4 | Proof steps for the bulk-spatiotemporal vortex correspondence [Theorem 4]. a, Schematic of plane wave reflection by a semi-infinite homogeneous target medium (green). b, Reflection amplitude (volume density plot) and isofrequency reflection phase (colormap on the horizontal plane) by the semi-infinite medium. Dashed cone: region of projected bulk band of the target medium. Cone vertex (pink dot): synthetic Weyl point in

b Isofrequency reflection vortex by bulk

![](images/c90d2161d105be29463f301201d23fd8f5811f8e83553a162ff3f9bc1843df92.jpg)

d Isofrequency reflection vortex by slabs

![](images/a4c9dbf3a5a7abb4307516c9952c702588a02e21eb12d2f0259999fe18cddb2d.jpg)

f Spatiotemporal reflection vortex by slabs

![](images/63e090b43ae50155b1adbffd161efb4bda4fd9a4b7965cdc8ec7cc18c635235a.jpg)

the $(k_{x}, k_{y}, \rho)$ -parameter space. c, Schematic of plane wave reflection by a homogeneous medium slab of thickness d. d, Reflection amplitude (volume density plot) and isofrequency reflection phase (colormap on the horizontal plane) from the finite-thickness slab. e, Wick rotation of the loop encircling the line of zero reflection (red). f, Spatiotemporal phase vortex of slab reflection at the critical parameter $(\rho = \rho_{0})$ of gap closing.

![](images/0024e01663909fb5f2c71ab62e8ec1c7f0d405d5228ede4d09bba9ca7c1d7dbf.jpg)

Experiment
a
b
c
![](images/c1c1ace6ac0476afc154e5dab1cb6554e74465e93c1b2e0d65bca2d28d12ed5f.jpg)

![](images/c2935876002ef4ced4866ec090395dd89af7c3ed3a1c7ca8ac8c93c0368b18f8.jpg)

Extended Data Fig. 5 | Emergence of edge states protected by a reflection spatiotemporal vortex. a, Simulated reflection phase of a 10-layer metamaterial slab, where the equi-phase contour $\arg(R) = \pi$ , i.e., $R = -1$ , predicts the band of edge states localized on a PEC boundary. b, Simulated local density of states (LDOS) spectrum along the scan line ( $x = 73$ mm) of a supercell with $N_x = 10$ units in the x-direction. c, Experimentally measured edge band compared with

theoretical prediction (white line). d, Eigen electric (colormap) and magnetic (black vectors) fields of the edge state at $k_{y}=0.138$ (blue dot in c) in the supercell. e, Schematic of the experimental setup for measuring the edge state band (c) of the metamaterial ( $10 \times 30$ unit cells). Red dots indicate the probe positions along the line (x=73 mm) close to the YIG rods near a PEC boundary. Purple star at $(x,y)=(4.5a,-15a)$ marks the position of the monopole antenna source.

![](images/3c1b77cd960adaba2f187bb6046c7214ff629ffa88874c86052495af4d425760.jpg)

![](images/7a4202b0c20ff4a98759384c5d59ad3a6098451488c3ebad0a294a71c6654424.jpg)

![](images/f493f6f034e823217113795cf61d4bda9061cfab2ab9a28434c4cb270d5bf9b7.jpg)

![](images/7fb2f41738edf4157e9adcfe3e04c3a9c826b0e4c2da2503302f0e5b40aa212a.jpg)
Extended Data Fig. 6 | Experimental setup for generating and measuring STVPs. a, An arbitrary waveform generator (AWG) generates Gaussian pulses that serve as both input and reference signals. The input signals are amplified and launched into the parallel-plate waveguide. The reference signals trigger the real-time oscilloscope, ensuring synchronization of the measured signals with the input. A probe antenna, inserted through a slot in the top aluminum plate, moves along the scan line with a 5 mm step to measure the electric field. b, Interior view of the parallel-plate waveguide. Permanent magnets are embedded into the drilled holes in the bottom aluminum plate and are covered with a piece of double-sided conductive copper tape to ensure that the bottom

![](images/e4bf4ec3b2f41d8a4107373124ad4fa885b8b37e8ff777f0f48498b63adbee6f.jpg)
aluminum plate forms a continuous conductive PEC surface. YIG pillars are placed above the copper tape on each magnet. Microwave absorbers are tiled around the periphery of the waveguide. c, Measured time signals of electric field, $E_{z}(x_{\mathrm{scan}}, y, t)$ , on the scan line that is 0.5 m away from the surface of a three-layer sample. d, Measured time signals for a single scan point, y = 0.022 m (blue solid line in c). e, Measured input (time range between the dashed green lines in c) and reflected (time range between the dashed red lines in c) pulses on the scan line after removing the fast oscillatory dynamical phases, $A(y, t) = E_{z}(y, t)e^{i\omega_{0}t}$ .
