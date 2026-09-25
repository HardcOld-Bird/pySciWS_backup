APPLIED PHYSICS

# Nonreciprocal field transformation with active acoustic metasurfaces

Xinhua Wen $^{1}$ , Choonlae Cho $^{2}$ , Xinghong Zhu $^{1}$ , Namkyoo Park $^{2*}$ , Jensen Li $^{1*}$

Field transformation, as an extension of the transformation optics, provides a unique means for nonreciprocal wave manipulation, while the experimental realization remains a substantial challenge as it requires stringent material parameters of the metamaterials, e.g., purely nonreciprocal bianisotropic parameters. Here, we develop and demonstrate a nonreciprocal field transformation in a two-dimensional acoustic system, using an active metasurface that can independently control all constitutive parameters and achieve purely nonreciprocal Willis coupling. The field-transforming metasurface enables tailor-made field distribution manipulation, achieving localized field amplification by a predetermined ratio. The metasurface demonstrates the self-adaptive capability to various excitation conditions and can be extended to other geometric shapes. The metasurface also achieves nonreciprocal wave propagation for internal and external excitations, demonstrating a one-way acoustic device. The nonreciprocal field transformation not only extends the framework of the transformation theory for nonreciprocal wave manipulation but also holds great potential in applications such as ultrasensitive sensors and nonreciprocal communication.

Copyright © 2024 The Authors, some rights reserved; exclusive licensee American Association for the Advancement of Science. No claim to original U.S. Government Works. Distributed under a Creative Commons Attribution License 4.0 (CC BY).

## INTRODUCTION

Over the past two decades, metamaterials with engineered material parameters have provided a remarkable way to achieve wave manipulation for different classical waves like electromagnetic (1–4), acoustic (5–8), and elastic waves (9–12). The combination of metamaterials and the concept of transformation optics allows us to engineer the wave propagation driven by coordinate transformations (13–16), enabling the concepts of cloaking and illusion that appeared in science fiction to become a reality. The concept of transformation optics is later extended to acoustic and elastic waves, and many novel devices designed by the transformation optics approach, including invisibility cloaks (17–20), illusion devices (21, 22), wave concentrators (23, 24), and rotators (25, 26), have been demonstrated. The vast majority of these devices intrinsically conform to the principle of reciprocity, as they are designed on the basis of the transformation that uses the metric tensor independent of direction (13, 14). Nonetheless, controlling nonreciprocal wave propagation using the transformation approach, like nonreciprocal cloaking and sensing (27, 28), remains highly desirable for extending the transformation approach. It has been proposed that nonreciprocal cloaking can be achieved with a coordinate-transformed gyrotropic medium in an external magnetic field (27). However, to attain magnetic-free nonreciprocity, the framework of conventional transformation optics rooted in Lorentz reciprocity is not easily applicable. An extension of transformation optics that combines frame deformation and coordinate transformation has been recently proposed for an electromagnetic continuum with local rotation at each point, to obtain a nonreciprocal response (29), while the implementation of such a structural anisotropy with metamaterials is very challenging.

On the other hand, field transformation (30–33), as another extension of the transformation approach, transforms the fields directly instead of using coordinate transformation, offering a unique means for nonreciprocal wave manipulation. By choosing a scaling-type transformation function, i.e., directly scaling the complex amplitudes of the fields, one can obtain nonreciprocal constitutive relations. Such a particular kind of field transformation can be exploited to realize nonreciprocal field transformation, with a field-transforming metamaterial that has the form of material relations of slowly moving media (30, 34). To satisfy such a material relation, it requests a metamaterial that can flexibly and independently control all constitutive parameters and also generate purely nonreciprocal bianisotropic coupling parameters. This is quite challenging for passive metamaterials with the constraint of reciprocity, even for conventional active metamaterials (35–37), thus the experimental realization of nonreciprocal field transformation has not yet been demonstrated.

In acoustics, active meta-atoms with a software-defined impulse response (38–40), allow us to independently control all constitutive parameters and achieve nonreciprocal acoustic bianisotropic coupling, also known as Willis coupling (41–44), opening the possibility for the realization of nonreciprocal field transformation. In this work, we propose and experimentally demonstrate nonreciprocal field transformation in a two-dimensional (2D) acoustic system. Our field-transforming metasurface constructed by active meta-atoms allows to locally amplify the field within a specific region without phase distortion, while the field outside is not affected because of the elimination of scattering waves of the metasurface. In contrast to conventional active control systems that require a priori knowledge of the incident wave field or sufficient latency time for recalculating the required secondary sources (45, 46), here, the field-transforming metasurface can already adapt to various incident waves with a tailor-made and fixed material response. Furthermore, the metasurface also enables a one-way acoustic device with nonreciprocal transmission in response to internal and external excitations. In contrast to conventional nonreciprocal acoustic devices based on nonlinearity (47, 48), moving media (49), space-time modulation (50, 51), and active electronics (36, 52), our field-transforming metasurface enables simultaneous 2D nonreciprocal control with amplification and the suppression of scattering wave outside. Nonreciprocal field transformation greatly extends the framework of transformation theory for nonreciprocal wave manipulation but also holds great potential for achieving highly sensitive sensing and nonreciprocal communication.

## RESULTS

## Nonreciprocal metamaterial approach for acoustic field transformation

Field-transforming metamaterials allow for desired field distribution within the medium. Here, we demonstrate the acoustic wave manipulation using a field-transforming metasurface, for convenience, with cylindrical symmetry. We start from the desired field distribution using a ring-shaped field-transforming metasurface with inner and outer radii $R_{1}$ and $R_{2}$ , as illustrated in Fig. 1A. For an external incident wave outside the metasurface, the metasurface transforms the pressure field by a transformation function $f(\boldsymbol{r})$ , resulting that the pressure field within the region enclosed by the metasurface (dark blue region) is amplified by a factor of $1 + s$ , i.e., $p(\boldsymbol{r}) = (1 + s)p_{0}(\boldsymbol{r})$ , where $p_{0}(\boldsymbol{r})$ represents the propagating pressure field in the background (without presence of metasurface), and a positive s denotes field amplification. On the other hand, we aim for the pressure field outside the metasurface (cyan region) to remain unaffected, ensuring $p(\boldsymbol{r}) = p_{0}(\boldsymbol{r})$ by eliminating the scattered field outside the metasurface (to prevent backscattering to the source). Such a field manipulation is illustrated by the mapping of the pressure field along the red dashed line, as shown with the red line in the lower panel of Fig. 1A. The pressure field inside exhibits a scaling factor of $1 + s$ , gradually decreasing (from $R_{1}$ to $R_{2}$ ) back to a value of 1 at the outer boundary of the metasurface. The metasurface transforms the pressure field through a transformation function $f(\boldsymbol{r})$ , which is closely associated with the Willis coupling $\tau$ of the metasurface (denoted by the green line). The relationship between them will be discussed later. We note that the transformation function $f(\boldsymbol{r})$ can be flexibly designed depending on the practical applications.

![](images/458d8454466d421e6bfb5e41e0b14009fe32d8d28870e29218f9f7e50fbcfb2b.jpg)

To achieve the desired field manipulation, we use an active metasurface with the necessary material parameters to be deduced from a theory of field transformation. As shown in Fig. 1B, the ring-shaped metasurface comprises 24 active meta-atoms, with a distance of $\Delta l = 0.039\mathrm{m}$ between neighboring atoms (i.e., the center circle of the metasurface has a radius of $0.15\mathrm{m}$ ). Figure 1C presents a sectional view of a specific region surrounding a meta-atom (marked by a yellow dashed rectangle in Fig. 1B) in the 2D waveguide, in which the meta-atom (right panel) is flipped and affixed at the bottom plate (cyan color). The meta-atom comprises two printed circuit boards (PCBs), with the lower board integrating two microphones and two speakers and connecting to the upper board that houses a microcontroller (ItsyBitsy M4). The flipped meta-atom makes the microphones and speakers face upward to sense and reradiate into the waveguide, as shown in Fig. 1D. Specifically, the microphones ( $D_{1}$ and $D_{2}$ ) detect signals propagating in the free space between the bottom and top plates and perform the time convolution with the convolution kernels $Y_{ij}$ (orange arrows). The convoluted result is then fed back to the speakers ( $S_{1}$ and $S_{2}$ ) to reradiate scattered waves into the waveguide. Such a meta-atom, referred to as a digitally virtualized atom (38-40), allows to implement a customized scattering response. To measure the field pattern, four measurement microphones (labeled as $\mathrm{M}_1$ to $\mathrm{M}_4$ ), inserted into the top plate (purple color) that connects to a movable positioning stage, enable the scanning of the pressure field across the entire 2D region. More details about the experiment implementation are provided in Materials and Methods.

![](images/f5b68c71668a4dadd81e3838425dd577ca68c086669156a6fc7d91e1b4bb9a10.jpg)
Fig. 1. Localized field amplification and its implementation setup. (A) Schematic of the localized field amplification by a ring-shaped metasurface with inner and outer radii $R_{1}$ and $R_{2}$ (top). The pressure field inside the metasurface (dark blue region) is amplified by a factor of $1 + s$ , while the pressure field outside (cyan region) remains unaffected. Bottom: The mapping of the pressure field along the red dashed line, in which the transformation function is associated with the Willis coupling $\tau$ (green line). (B) Experiment setup for 2D field pattern scanning. A metasurface consisting of 24 active meta-atoms is affixed to the bottom plate (cyan color) of a 2D waveguide. The top plate of the waveguide (purple color), with four inserted microphones (labeled as $M_{1}$ to $M_{4}$ ), is connected to a movable positioning stage to enable scanning across the entire region. (C) A sectional view of a specific region surrounding the meta-atom [marked by a yellow dashed rectangle in (B)] in the 2D waveguide, in which the meta-atom in the right panel is flipped and assembled at the bottom plate. (D) The flipped meta-atom and schematic representation of the convolution (orange arrows).

To deduce the required material parameters for implementation, we use a theory of field transformation without any coordinate transformation (30). The acoustic wave equations in cylindrical coordinates $(r, \theta)$ , at a harmonic radial frequency $\omega$ (with $e^{-i\omega t}$ convention), can be expressed as

$$
\begin{array}{r l} & {\frac {\partial p}{\partial \theta} = i \omega r (\rho_ {0} v _ {\theta} + d _ {\theta}),} \\ & {\frac {\partial p}{\partial r} = i \omega (\rho_ {0} v _ {\mathrm{r}} + d _ {\mathrm{r}}) \frac {\partial r v _ {\mathrm{r}}}{\partial r} + \frac {\partial v _ {\theta}}{\partial \theta} = i \omega r (\beta_ {0} p + m)} \end{array}\tag{1}
$$

Here, p represents the pressure field, and $v_{r}$ and $v_{\theta}$ are the radial and azimuthal components of the velocity field, respectively. m corresponds to the acoustic monopole density generated by the metasurface, while $d_{r}$ and $d_{\theta}$ represent the radial and azimuthal components of the acoustic dipole moment density. $\beta_{0}$ and $\rho_{0}$ denote the free-space compressibility and density, respectively. To manipulate the local field amplitudes, we request a scaling-type field transformation given by

$$
p = f (r) p _ {0}, \nu_ {\mathrm{r}} = f (r) \nu_ {\mathrm{r0}}, \nu_ {\theta} = f (r) \nu_ {\theta 0}\tag{2}
$$

where $f(r)$ is a transformation function (only in terms) of r to relate the original (without the presence of the field-transforming meta-surface) and the transformed fields. The background pressure field $p_{0}$ and velocity field $v_{r0}$ and $v_{\theta0}$ represent the background fields that satisfy the free-space wave equation, i.e., Eq. 1 with zero acoustic monopole and dipole moments. By substituting Eq. 2 into Eq. 1, we obtain the required monopole and dipole moments

$$
\mathrm{m} = \frac {f ^ {\prime} (r)}{i \omega} \nu_ {\mathrm{r0}}, \mathrm{d} _ {\mathrm{r}} = \frac {f ^ {\prime} (r)}{i \omega} p _ {0}, \mathrm{d} _ {\theta} = 0\tag{3}
$$

This is akin to those active control systems where the induced secondary sources are determined by the incident wave (45). Thus, the secondary sources need to be recalculated when the incident field changes. However, by combining Eq. 2, Eq. 3 can be rewritten as

$$
\mathrm{m} = \frac {f ^ {\prime} (r)}{\imath \omega f (r)} v _ {\mathrm{r}}, \mathrm{d} _ {\mathrm{r}} = \frac {f ^ {\prime} (r)}{\imath \omega f (r)} p, \mathrm{d} _ {\theta} = 0\tag{4}
$$

For our field-transforming metasurface, the secondary source can be directly associated with the local field. As a result, the active metasurface with a fixed effective material response can adapt to the various incident fields without prior knowledge of the incident wavefield. Moreover, the fixed material response can be specified by controlling the convolution kernels in the program, which will be discussed in detail later. Furthermore, Eq. 4 can be expressed as a dimensionless constitutive matrix relating the volume strain $-\epsilon$ and momentum density $\mu$ (quantities in brackets in Eq. 1) to the pressure and velocity fields (40).

$$
\left( \begin{array}{c} - \epsilon / \beta_ {0} \\ c \mu_ {\mathrm{r}} \\ c \mu_ {\theta} \end{array} \right) = \left( \begin{array}{c c c} \beta & i \tau & 0 \\ i \tau^ {\prime} & \rho & 0 \\ 0 & 0 & \rho \end{array} \right) \left( \begin{array}{c} p \\ \eta_ {0} v _ {\mathrm{r}} \\ \eta_ {0} v _ {\theta} \end{array} \right)\tag{5}
$$

with compressibility and density $\beta=\rho=1$ and the Willis couplings $\tau=\tau^{\prime}=-f^{\prime}(r)/[k_{0}f(r)]$ . $k_{0}=\omega/c$ is the free-space wave number, and c represents the sound speed in free space. $\eta_{0}=\rho_{0}c$ corresponds to the wave impedance in free space. The constitutive matrix of the field-transforming metasurface in Eq. 5, within the region with $\tau=\tau^{\prime}\neq0$ , is symmetric (purely nonreciprocal) rather than antisymmetric ( $\tau=-\tau^{\prime}$ , reciprocal) and is simultaneously non-Hermitian (active) (40). In addition, $\beta=\rho=1$ , indicates that the metasurface possesses the same compressibility and density as that of free space. The wave impedance of the metasurface does not change from that of the free space despite the presence of the purely nonreciprocal Willis coupling $\tau$ , because the pressure and velocity fields are transformed in the same way (30). Consequently, the scattered field outside the metasurface is eliminated, leaving the pressure field outside intact.

By choosing an appropriate transformation function $f(r)$ , we can obtain the required constitutive parameters of the ring-shaped metasurface. At the inner and outer boundaries, the pressure fields experience different scaling factors defined as $f(R_{1}) = f_{1}$ and $f(R_{2}) = f_{2}$ . Taking $\ln f$ to be linear in r, the transformation function (for $R_{1} \leq r \leq R_{2}$ ) can be written as

$$
\ln f = \frac {R _ {2} - r}{R _ {2} - R _ {1}} \ln f _ {1} + \frac {r - R _ {1}}{R _ {2} - R _ {1}} \ln f _ {2}\tag{6}
$$

Thus, the nonreciprocal Willis coupling $\tau$ can be obtained as

$$
\tau = - \frac {f ^ {\prime} (r)}{k _ {0} f (r)} = - \frac {\ln f _ {2} - \ln f _ {1}}{k _ {0} (R _ {2} - R _ {1})}\tag{7}
$$

Unlike the conventional transformation optics approach that typically requires a metamaterial with an inhomogeneous material profile to achieve the desired field manipulation, the field-transforming metasurface described here is purposely designed to have homogeneous material parameters. This is achieved by setting the linear property of $\ln f$ with respect to r, making the experimental implementation much simpler. According to Eq. 7, the local field amplification ratio can be equivalently written as

$$
f _ {1} / f _ {2} = e ^ {\tau k _ {0} (R _ {2} - R _ {1})} = 1 + s\tag{8}
$$

Therefore, a positive real nonreciprocal Willis coupling $\tau$ enables filed amplitude amplification without phase distortion. Moreover, the amplification ratio $1 + s$ can be adjusted by controlling the value of $\tau$ .

Before delving into the implementation of the metasurface, we first verify our theoretical framework of field transformation using full-wave simulations at the level of material parameters derived above. As an example, we set the local field amplification ratio to be $1 + s = 1.5$ (i.e., s = 0.5) throughout the main text without further mention. We set the constitutive parameters of the metasurface: $\tau = 0.27$ with $\beta = \rho = 1$ and a thickness of metasurface as 0.24 of the free-space wavelength $\lambda$ (i.e., $R_{2} - R_{1} = 0.24\lambda$ ) to satisfy Eq. 8. By specifying these material parameters for the field-transforming metasurface in COMSOL Multiphysics (general PDE module), we can simulate the local field amplification phenomenon. As shown in Fig. 2B, the field-transforming metasurface enables the amplification of the pressure field inside the metasurface without any phase distortion, compared to the situation without the metasurface present (Fig. 2A). Furthermore, the pressure field outside the metasurface is unaffected. Figure 2C shows the radial mapping of the nonreciprocal Willis coupling $\tau$ (upper) and the pressure field amplitude $|p|$ (lower) along the white lines in Fig. 2 (A and B), verifying that the ring-shaped metasurface with a homogeneous $\tau = 0.27$ (red line) indeed enables the local field amplification inside the metasurface $(x/\lambda \leq R_{1})$ with a ratio of 1.5 (purple line). In addition, the pressure field outside the metasurface $(x/\lambda \geq R_{2})$ equals the background pressure field (purple and green dashed lines), demonstrating the complete suppression of the scattered field outside due to impedance matching. Consequently, such a metasurface indeed provides the desired field distribution in Fig. 1A.

B
![](images/394f4976ab1f953a3c6f41b0a3b4212d299ce28cdb6a63f79f74e8bcece4a1ae.jpg)

![](images/debe8359c555ec61b6b24e16a9c17b127137470513815912ef874d737755ee27.jpg)

![](images/361f1c397e58c395bd0fb8d87fcf154ad60faf53581f64a79d678983ddfa916b.jpg)

C
![](images/bfe8899d464e9a9d150baf6f2071c82ed37539aa0f47e82d8da42e87992555af.jpg)

![](images/8b52654ab0aab2c2d93a787f29bfcdbf1372aa460c100649bdede3bafc65b355.jpg)
F

![](images/41c192e9a80ee631698e23726b9528cf2e6b609b564ed4dacd44076b5d49a8fd.jpg)
Fig. 2. Numerical verification of the field-transforming metasurface with the required material parameters (A and B) Simulated pressure field patterns for an external plane wave excitation when the metasurface is (A) absent (denoted as OFF) and (B) present (denoted as ON). (C) The radial mapping of the Willis coupling $\tau$ (top) and pressure field amplitude (bottom) along the white lines in (A) and (B). (D and E) Similar to (A) and (B) but for an external point source excitation at $(5\lambda, -4\lambda)$ . (F) The simulated pressure field pattern for a bean-shaped metasurface. In (A) to (E), $R_{1} = 3.76\lambda$ , $R_{2} = 4.0\lambda$ with $\lambda$ being the free-space wavelength. $f_{1}(f_{2})$ corresponds to the scaling factor at the inner (outer) boundary of the metasurface.

Because the generated secondary source of the metasurface can be directly associated with the local field (Eq. 4), the field-transforming metasurface can adapt to other types of external incident waves, like the circular wavefront from a point source. As shown in Fig. 2 (D and E), the desired local field amplification also occurs for an external point source excitation at $(5\lambda,-4\lambda)$ . Outside the metasurface, the total field remains the same as the original cylindrical wave (i.e., the metasurface is absent) without scattering. In addition, the shape of the field-transforming metasurface is not limited to the geometric shape with rotational symmetry and can also be extended to other shapes, like the bean shape shown in Fig. 2F. The bean-shaped metasurface with an equal distance between the inner and outer boundaries $(0.24\lambda)$ , has the same compressibility and density $\beta=\rho=1$ and a homogeneous $\tau=0.27$ along the direction perpendicular to the geometric shape, enabling the desired local field amplification. A theoretical framework for designing the metasurface with a general geometric shape is elaborated in note S1.

## Experimental verification

For the experimental realization, we turn to a metasurface constructed with 24 active meta-atoms (Fig. 1B). By manipulating the software-defined convolution kernels $Y_{ij}$ of the meta-atoms, all constitutive parameters of the metasurface, including compressibility, density, and the Willis coupling, can be independently controlled (38–40). Particularly, such an active metasurface enables the realization of purely nonreciprocal Willis coupling, in contrast to those typical Willis metamaterials that can only generate reciprocal Willis coupling (42–44). Therefore, a field-transforming metasurface with required constitutive parameters (Eq. 5) can be experimentally realized, by controlling four time-convolution kernels $Y_{ij}$ of the meta-atoms that connect the detected pressure fields by two microphones to the radiated pressure fields from two speakers (denoted by orange arrows in Fig. 1D). In time harmonic, the time convolution in each meta-atom can be expressed as a matrix multiplication as

$$
\binom{S _ {1}}{S _ {2}} = \left( \begin{array}{c c} Y _ {1 1} & Y _ {1 2} \\ Y _ {2 1} & Y _ {2 2} \end{array} \right) \binom{D _ {1}}{D _ {2}}\tag{9}
$$

For our field-transforming metasurface with $\beta = \rho = 1$ , and non-reciprocal Willis coupling $\tau$ from Eq. 7, the convolution kernels $Y_{ij}$ can be set as (see derivation in note S2)

$$
\begin{array}{l} Y _ {1 1} = Y _ {2 2} = 0, Y _ {1 2} = \frac {1}{2} i (- 1 + e ^ {\tau \phi_ {0}}) \csc (2 \Delta \phi), \\ Y _ {2 1} = \frac {1}{2} i (- 1 + e ^ {- \tau \phi_ {0}}) \csc (2 \Delta \phi) \end{array}\tag{10}
$$

where $\phi_{0}=k_{0}(R_{2}-R_{1})$ is the phase elapse across the meta-atom, and $\Delta\phi$ is the propagation phase distance from the center of the

B

meta-atom to either one of the speakers or microphones (Fig. 1D). These convolution kernels in Eq. 10 allow the suppression of the backscattering fields and the realization of nonreciprocal Willis coupling due to $Y_{12} \neq Y_{21}$ .

In the experiment, the constructed ring-shaped metasurface has a thickness of $R_{2} - R_{1} = 0.06 \, m$ (see Fig. 1, B and D), and the incident wavelength is chosen as $\lambda = 0.25 \, m$ (i.e., working frequency is 1372 Hz), to obtain the amplification ratio 1.5 (satisfying Eq. 8). To achieve the required nonreciprocal Willis coupling $\tau = 0.27$ , the necessary convolution kernels $Y_{12}$ and $Y_{21}$ of all meta-atoms can be obtained from Eq. 10

$$
Y _ {1 2} = 0. 2 7 6 i, Y _ {2 1} = - 0. 1 8 4 i\tag{11}
$$

As resonance is helpful to greatly enhance Willis coupling (53) and ensure our system stability (40), we use Lorentzian-type resonance for the convolution kernels $Y_{12}$ and $Y_{21}$ in the implementation

$$
Y _ {1 2 / 2 1} (f) = \frac {g _ {1 2 / 2 1} f _ {0}}{f _ {0} ^ {2} - (f + i \gamma) ^ {2}}\tag{12}
$$

Here, the resonance linewidth $\gamma$ is 50 Hz, and the resonance frequency $f_{0}$ is chosen as the working frequency of the metasurface (1372 Hz), such that $Y_{12/21}$ can be approximated as $Y_{12/21}(f_{0}) \cong \mathrm{ig}_{12/21}/2\gamma$ , a purely imaginary number at the resonance frequency. Then, we can control the resonance strength $g_{12/21}$ to obtain the specified convolution kernels in Eq. 11 at $f_{0}$ . In the actual implementation, the radiated pressure field from two speakers is associated with the distance between neighboring meta-atoms $\Delta l$ (Fig. 1B); thus, we first implement and test the meta-atom in a 1D waveguide, to verify that the metasurface has the required constitutive parameters (see details in note S3).

D
![](images/6b9af74c6a22956bb3aade5bf4f73411220c32b018a7dc9d9cc4b0fc90685df5.jpg)

![](images/25ccb7e1bb19a38e857499200143776799cdd322a2f6f0a76f72a37b43c8e26f.jpg)

Using the field-transforming metasurface with the required constitutive parameters, we now experimentally validate the localized field amplification. In the experiment, the metasurface features a central circle with a radius of 0.15 m, which is henceforth represented as a black circular line. As depicted in Fig. 3B, when a point source excitation at 1372 Hz is applied at the point (0.29 m, -0.1 m), labeled as point A, the measured pressure field within the region enclosed by the metasurface is significantly amplified compared to the case in Fig. 3A when the metasurface is turned off. The amplification ratio will be discussed later. The metasurface, featuring purely nonreciprocal Willis coupling and impedance matching, only affects the pressure field inside the metasurface. Essentially, the secondary sources of all the meta-atoms in the metasurface interfere to generate a scattered field distributed only inside the metasurface, propagating in phase with the background field (refer to note S4 for further details). Thus, constructive interference between the scattered field and background field occurs, leading to local field amplification inside the metasurface. Meanwhile, the scattered field outside is almost completely suppressed, leaving the background field outside unaffected.

With a fixed material response defined in the program, the active metasurface can adapt to different incident waves automatically without the need for reconfiguration of the meta-atoms. As illustrated in Fig. 3 (C and D), the localized field amplification also occurs when the metasurface is activated (Fig. 3D) for a point source excitation at a different location A' (-0.27 m, 0.18 m). In this case, the scattered field outside is also suppressed (see note S4), although some nonvanishing scattered fields, resulting from the imperfect sound absorber, are present around the boundaries of the experimental domain (Fig. 3, C and D). The localized field amplification and the self-adaptive property can also be numerically verified in COMSOL Multiphysics using a metasurface with 24 discrete meta-atoms, as discussed in note S5. Numerical results also confirm that the field-transforming metasurface can adapt to external point source excitations at different locations, as well as a plane wave excitation with an arbitrary incidence angle (see fig. S9).

![](images/78c4b22ad0f4a7994dbb41b6d5a91781b6231b69c7d8951e34a1a61dd8ff8aab.jpg)
Fig. 3. Experimental demonstration of localized field amplification. (A and B) Measured pressure field pattern for a point source at location A when the metasurface is (A) turned off and (B) turned on. The solid (dashed) black circular line represents the turned-on (turned-off) metasurface with a ring shape. (C and D) Similar to (A) and (B) but for a point source excitation at location A'.

The same field-transforming metasurface can also be used to achieve a one-way acoustic device with nonreciprocal transmission. As schematically shown in Fig. 4A, for an external point source excitation at point A, the transmitted wave detected at point B (0.09 m, 0 m) is amplified by a factor of $1 + s$ (indicated by the red arrow). Conversely, for a point source excitation at point B, within the region enclosed by the metasurface, the transmitted wave detected at point A is scaled by a factor of $1/(1 + s)$ (blue arrow), showcasing nonreciprocal wave propagation. Figure 4B displays the simulated pressure field pattern for a point source excitation at point B, using the metasurface with 24 discrete active meta-atoms (see note S5 for the simulation approach). The pressure field outside the metasurface is significantly reduced because of the destructive interference between the scattered field and the background field (see note S4). However, the pressure field within the region enclosed by the metasurfaces remains unaffected, because of the suppression of the scattering field inside. The measured field pattern in Fig. 4C closely matches the simulation result (Fig. 4B), providing experimental validation of nonreciprocal transmission using our field-transforming metasurface. Therefore, we experimentally demonstrate a one-way device that allows a sensor situated inside to clearly detect the sound from outside (i.e., detect an amplified signal) but not vice versa.

For the nonreciprocal transmission, we experimentally validate the field scaling factors for two opposite propagation directions. As shown in Fig. 4A, we place a point source at point A (B) and measure the pressure at point B (A) and two nearby points denoted by open square and triangular symbols (all three measured points have an equal distance of $2.54~\mathrm{cm}$ ). The scaling factor is defined as the pressure field amplitude ratio with the metasurface turned on and off $(|p / p_0|)$ ; see note S6 for details of the derivation. Figure 4D presents the measured scaling factor in symbols as a function of incident frequency, where the lines represent the model results calculated from Eqs. 8, 10 and 12 (with a small frequency shift). At the working frequency of $1372\mathrm{Hz}$ , the detected pressure field at point B (A) is indeed scaled by a ratio of around 1.5 (0.61) for a point source excitation at point A (B), which aligns with the expected values indicated by the dashed gray lines. The pressure field detected at the two nearby points (open square and triangular symbols) also shows consistent experimental results. Although the spectrum demonstrates a limited working bandwidth (around $87\mathrm{Hz}$ ) due to the resonance nature (Eq. 12), this is helpful for the stability of a system with gain, as it effectively precludes the amplification of white noise beyond the desired frequency range. Moreover, the active metasurface, endowed with programmability, allows for the reconfigurability of the working bandwidth and frequency (38). As previously mentioned, the field-transforming metasurface can extend to other geometry shapes (Fig. 2F), and we also numerically verify nonreciprocal wave propagation using a bean-shaped metasurface comprising 24 discrete meta-atoms (see note S7). In addition, a larger amplification ratio of 1.8 is experimentally achieved by increasing the nonreciprocal Willis coupling, as discussed in note S8. The field-transforming metasurface has potential for achieving significantly larger values of s, or even a value close to -1, which enables to effectively deplete the field inside (see note S9).

A
![](images/42ab42337427da3e37c2a86c1ee473ddf036655481c08382a9df384e03f00751.jpg)

![](images/b6142b636bfb62b09de5d8bad93f6ee8feb0b50f0afe79ff130b94b81bbcf13e.jpg)

![](images/4b239ae44354831ffbfa84498482a69bebcc96b341702ba781fc2fd6c7a939d1.jpg)

![](images/ebec8ecb706c185380634171c6664d30364babb637ecd5b2c4796537a6f9a919.jpg)
Fig. 4. Nonreciprocal wave propagation with the field-transforming metasurface. (A) Schematic of the nonreciprocal transmission for external and internal point source excitations. The transmitted wave is scaled by a factor of $1 + s$ from A to B (external excitation, red arrow), while the transmitted wave is scaled by a factor of $1/(1 + s)$ from B to A (internal excitation, blue arrow). (B) Simulated and (C) measured pressure field patterns for an internal point source at location B. (D) The field scaling factor $\left|p/p_0\right|$ as a function of frequency. Symbols represent the measured results, and lines represent the model results.

## DISCUSSION

We proposed and experimentally demonstrated nonreciprocal field transformation in a 2D acoustic system. Using active meta-atoms with a software-defined material response, we realize a metasurface with the purely nonreciprocal Willis coupling terms and the same compressibility and density as that of free space (Eq. 5). The field-transforming metasurface enables tailor-made field distribution manipulation, achieving the localized field amplification by a predesigned ratio and nonreciprocal wave propagation. Our scale-type intensity manipulation approach stands in stark contrast to those field intensity manipulation approaches that rely on a judiciously designed gain-loss profile (54–57). The latter remains constrained by reciprocity and achieves the desired field manipulation only for a specific incident direction, while our approach breaks reciprocity and demonstrates a self-adaptive capability to various incident conditions, either a plane wave excitation with an arbitrary incident angle (see fig. S9) or a point source excitation. In addition, we present an experimental demonstration of nonreciprocal wave control using the transformation approach, greatly extending the framework of the transformation theory for nonreciprocal wave manipulation and anticipating to trigger more nonreciprocal devices designed by transformation approach like nonreciprocal cloaking devices (28). Furthermore, the metasurface with purely nonreciprocal Willis coupling opens up a new avenue toward real applications based on nonreciprocal amplification, like nonreciprocal amplifiers, ultrasensitive sensors, and nonreciprocal communication with signal amplification in one propagation direction. For example, the field-transforming metasurface substantially enhances sensor sensitivity while suppressing backscattering waves without disturbing the external field. Such a highly sensitive sensor can adapt to various incident waves as needed, even in a dynamic environment.

## MATERIALS AND METHODS

The meta-atom comprises two microphones (ADMP401) and two speakers (SMT-1028-T-2-R) that are integrated into a PCB (with a size of $0.03 \times 0.06$ m), connected to a microcontroller (Adafruit ItsyBitsy M4 Express). The microcontroller performs four different channels of time convolution in connecting the two speakers to the two microphones. In the actual experiment, the efficiency of the speakers that radiate the pressure field converted from the volumetric flow depends on the type of speakers used in the experiment. Therefore, calibration factors in the program are used to obtain the expected convolution kernels when implementing the meta-atom in the 1D waveguide. Once the implemented program for the meta-atom in the 1D waveguide is tested to achieve the required constitutive parameters (see note S3), we upload the same program for all the meta-atoms of the metasurface.

As shown in Fig. 1B, the whole 2D experimental setup consists of a ring-shaped metasurface, a 2D waveguide made of acrylic material, a movable positioning stage (not shown), and the pressure field measurement system with four microphones connecting to the NI DAQ. The meta-atoms of the metasurface are affixed in the bottom plate (cyan color) of the box, thus not blocking the wave propagation within the waveguide. Specifically, there are preserved holes on the bottom plate for microphones and speakers, to sense and reradiate the pressure field into the 2D waveguide. For the metasurface with 24 meta-atoms, each meta-atom couples with one another through the 2D waveguide.

Sound absorbers made of acoustic foams are placed around the four rigid side plates. Four microphones labeled as $M_{1}$ to $M_{4}$ , are inserted in the preserved holes on the top plate (purple color) to measure the pressure field inside the waveguide. For pressure field pattern scanning, the top plate connects to a movable positioning stage, such that a maximum area of 0.693 m by 0.586 m can be scanned by moving the top plate. In addition, we use a LabVIEW system to control the positioning stage and take the time data from the NI DAQ device connecting to four microphones. Using the Fourier transform, we can get the frequency data in every scanned position, obtaining the pressure field pattern. It is worth noting that the height of the 2D waveguide is 0.02 m (Fig. 1C); thus, only the fundamental waveguide mode for the working frequency 1372 Hz is allowed to propagate inside the box, which guarantees an experimental system of a 2D nature.

## Supplementary Materials

Notes S1 to S9
Figs. S1 to S18

## REFERENCES AND NOTES

1. J. B. Pendry, A. J. Holden, D. J. Robbins, W. J. Stewart, Magnetism from conductors and enhanced nonlinear phenomena. IEEE Trans. Microw. Theory Tech. 47, 2075–2084 (1999).

2. D. R. Smith, W. J. Padilla, D. C. Vier, S. C. Nemat-Nasser, S. Schultz, Composite medium with simultaneously negative permeability and permittivity. Phys. Rev. Lett. 84, 4184–4187 (2000).

3. N. Yu, P. Genevet, M. A. Kats, F. Aieta, J. P. Tetienne, F. Capasso, Z. Gaburro, Light propagation with phase discontinuities: generalized laws of reflection and refraction. Science 334, 333–337 (2011).

4. V. M. Shalaev, Optical negative-index metamaterials. Nat. Photonics 1, 41–48 (2007).

5. Z. Liu, X. Zhang, Y. Mao, Y. Y. Zhu, Z. Yang, C. T. Chan, P. Sheng, Locally resonant sonic materials. Science 289, 1734–1736 (2000).

6. N. Fang, D. Xi, J. Xu, M. Ambati, W. Srituravanich, C. Sun, X. Zhang, Ultrasonic metamaterials with negative modulus. Nat. Mater. 5, 452–456 (2006).

7. J. Li, C. T. Chan, Double-negative acoustic metamaterial. Phys. Rev. E 70, 055602 (2004).

8. J. Li, X. Wen, P. Sheng, Acoustic metamaterials. J. Appl. Phys. 129, 171103 (2021).

9. Y. Wu, Y. Lai, Z. Q. Zhang, Elastic metamaterials with simultaneously negative effective shear modulus and mass density. Phys. Rev. Lett. 107, 105506 (2011).

10. R. Zhu, X. N. Liu, G. K. Hu, C. T. Sun, G. L. Huang, Negative refraction of elastic waves at the deep-subwavelength scale in a single-phase metamaterial. Nat. Commun. 5, 5510 (2014).

11. Y. Liu, X. Su, C. T. Sun, Broadband elastic metamaterial with single negativity by mimicking lattice systems. J. Mech. Phys. Solids 74, 158–174 (2015).

12. Y. Liu, Z. Liang, J. Zhu, L. Xia, O. Mondain-Monval, T. Brunet, A. Alù, J. Li, Willis metamaterial on a structured beam. Phys. Rev. X 9, 011040 (2019).

13. J. B. Pendry, D. Schurig, D. R. Smith, Controlling electromagnetic fields. Science 312, 1780–1782 (2006).

14. U. Leonhardt, Optical conformal mapping. Science 312, 1777–1780 (2006).

15. D. Schurig, J. J. Mock, B. J. Justice, S. A. Cummer, J. B. Pendry, A. F. Starr, D. R. Smith, Metamaterial electromagnetic cloak at microwave frequencies. Science 314, 977–980 (2006).

16. W. Cai, U. K. Chettiar, A. V. Kildishev, V. M. Shalaev, Optical cloaking with metamaterials. Nat. Photonics 1, 224–227 (2007).

17. H. Chen, C. T. Chan, Acoustic cloaking in three dimensions using acoustic metamaterials. Appl. Phys. Lett. 91, 183518 (2007).

18. S. A. Cummer, D. Schurig, One path to acoustic cloaking. New J. Phys. 9, 45 (2007).

19. H. Chen, C. T. Chan, Acoustic cloaking and transformation acoustics. J. Phys. D Appl. Phys. 43, 113001 (2010).

20. A. N. Norris, A. L. Shuvalov, Elastic cloaking theory. Wave Motion 48, 525-538 (2011).

21. W. Kan, B. Liang, X. Zhu, R. Li, X. Zou, H. Wu, J. Yang, J. Cheng, Acoustic illusion near boundaries of arbitrary curved geometry. Sci. Rep. 3, 1427 (2013).

22. P. Zhao, L. Luo, Y. Liu, J. Li, Flexural wave illusion on a curved plate. arXiv:2307.14062 [physics.app-ph] (26 July 2023).

23. J. Yang, M. Huang, C. Yang, G. Cai, A metamaterial acoustic concentrator with regular polygonal cross section. J. Vib. Acoust. 133, 061016 (2011).

24. Y. R. Wang, H. Zhang, S. Y. Zhang, L. Fan, H. X. Sun, Broadband acoustic concentrator with multilayered alternative homogeneous materials. J. Acoust. Soc. Am. 131, EL150–EL155 (2012).

25. X. Jiang, B. Liang, X. Y. Zou, L. L. Yin, J. C. Cheng, Broadband field rotator based on acoustic metamaterials. Appl. Phys. Lett. 104, 083510 (2014).

26. H. Chen, B. Hou, S. Chen, X. Ao, W. Wen, C. T. Chan, Design and experimental realization of a broadband transformation media field rotator at microwave frequencies. Phys. Rev. Lett. 102, 183903 (2009).

27. C. He, X. L. Zhang, L. Feng, M. H. Lu, Y. F. Chen, One-way cloak based on nonreciprocal photonic crystal. Appl. Phys. Lett. 99, 151112 (2011).

28. M. Dehmollaian, G. Lavigne, C. Caloz, Transmittable nonreciprocal cloaking. Phys. Rev. Appl. 19, 014051 (2023).

29. Y. Zhang, L. Shi, C. T. Chan, K. H. Fung, K. Chang, Geometrical theory of electromagnetic nonreciprocity. Phys. Rev. Lett. 130, 203801 (2023).

30. S. A. Tretyakov, I. S. Nefedov, P. Alitalo, Generalized field-transforming metamaterials. New J. Phys. 10, 115028 (2008).

31. F. Liu, Z. Liang, J. Li, Manipulating polarization and impedance signature: A reciprocal field transformation approach. Phys. Rev. Lett. 111, 033901 (2013).

32. F. Liu, J. Li, Gauge field optics with anisotropic media. Phys. Rev. Lett. 114, 103902 (2015).

33. Y. Liu, S. Tang, H. Shi, J. Zhao, W. Wang, B. Zhou, Dielectric approximation media to reproduce dispersion for field transformation. Appl. Optics 59, 7613–7620 (2020).

34. J. Vehmas, S. Hrabar, S. Tretyakov, Transmission lines emulating moving media. New J. Phys. 16, 093065 (2014).

35. G. Lavigne, T. Kodera, C. Caloz, Metasurface magnetless specular isolator. Sci. Rep. 12, 5652 (2022).

36. Y. Zhai, H. S. Kwon, B. I. Popa, Active Willis metamaterials for ultracompact nonreciprocal linear acoustic devices. Phys. Rev. B 99, 220301 (2019).

37. Y. Chen, X. Li, G. Hu, M. R. Haberman, G. Huang, An active mechanical Willis meta-layer with asymmetric polarizabilities. Nat. Commun. 11, 3681 (2020).

38. C. Cho, X. Wen, N. Park, J. Li, Digitally virtualized atoms for acoustic metamaterials. Nat. Commun. 11, 251 (2020).

39. C. Cho, X. Wen, N. Park, J. Li, Acoustic Willis meta-atom beyond the bounds of passivity and reciprocity. Commun. Phys. 4, 82 (2021).

40. X. Wen, H. K. Yip, C. Cho, J. Li, N. Park, Acoustic amplifying diode using nonreciprocal willis coupling. Phys. Rev. Lett. 130, 176101 (2023).

41. J. R. Willis, Variational principles for dynamic problems for inhomogeneous elastic media. Wave Motion 3, 1–11 (1981).

42. M. B. Muhlestein, C. F. Sieck, P. S. Wilson, M. R. Haberman, Experimental evidence of Willis coupling in a one-dimensional effective material element. Nat. Commun. 8, 15625 (2017).

43. S. Koo, C. Cho, J. H. Jeong, N. Park, Acoustic omni meta-atom for decoupled access to all octants of a wave parameter space. Nat. Commun. 7, 13012 (2016).

44. M. B. Muhlestein, C. F. Sieck, A. Alù, M. R. Haberman, Reciprocity, passivity and causality in Willis materials. Proc. Math. Phys. Eng. Sci. 472, 20160604 (2016).

45. M. Selvanayagam, G. V. Eleftheriades, Experimental demonstration of active electromagnetic cloaking. Phys. Rev. X 3, 041011 (2013).

46. T. S. Becker, D.-J. Van Manen, T. Haag, C. Bärlocher, X. Li, N. Börsing, A. Curtis, M. Serra-Garcia, J. O. Robertsson, Broadband acoustic invisibility and illusions. Sci. Adv. 7, eabi9627 (2021).

47. B. Liang, B. Yuan, J. C. Cheng, Acoustic diode: Rectification of acoustic energy flux in one-dimensional systems. Phys. Rev. Lett. 103, 104301 (2009).

48. B. Liang, X. S. Guo, J. Tu, D. Zhang, J. C. Cheng, An acoustic rectifier. Nat. Mater. 9, 989–992 (2010).

49. R. Fleury, D. L. Sounas, C. F. Sieck, M. R. Haberman, A. Alù, Sound isolation and giant linear nonreciprocity in a compact acoustic circulator. Science 343, 516–519 (2014).

50. R. Fleury, D. L. Sounas, A. Alù, Subwavelength ultrasonic circulator based on spatiotemporal modulation. Phys. Rev. B 91, 174306 (2015).

51. C. Shen, J. Li, Z. Jia, Y. Xie, S. A. Cummer, Nonreciprocal acoustic transmission in cascaded resonators via spatiotemporal modulation. Phys. Rev. B 99, 134306 (2019).

52. B. I. Popa, S. A. Cummer, Non-reciprocal and highly nonlinear active acoustic metamaterials. Nat. Commun. 5, 3398 (2014).

53. L. Quan, S. Yves, Y. Peng, H. Esfahlani, A. Alù, Odd Willis coupling induced by broken time-reversal symmetry. Nat. Commun. 12, 2615 (2021).

54. S. Yu, X. Piao, N. Park, Bohmian photonics for independent control of the phase and amplitude of waves. Phys. Rev. Lett. 120, 193902 (2018).

55. K. G. Makris, A. Brandstötter, P. Ambichl, Z. H. Musslimani, S. Rotter, Wave propagation through disordered media without backscattering and intensity variations. Light Sci. Appl. 6, e17035 (2017).

56. A. Steinfurth, I. Krešić, S. Weidemann, M. Kremer, K. G. Makris, M. Heinrich, S. Rotter, A. Szameit, Observation of photonic constant-intensity waves and induced transparency in tailored non-Hermitian lattices. Sci. Adv. 8, eabl7412 (2022).

57. B. I. Popa, S. A. Cummer, Complex coordinates in transformation optics. Phys. Rev. A 84, 063837 (2011).

Acknowledgments: We thank S. Zhang for useful discussions. X.W. would like to thank J. Park, H. Noh, and G. Yoon for help when X.W. was in SNU for this project. Funding: This work was supported by Research Grants Council (RGC) of Hong Kong through project nos. 16303019, 16307522, and AoE/P-502/20; the Croucher Foundation (CF23SC01); and Korean National Research Foundation grant RS-2023-00274348. Author contributions: Conceptualization: J.L. and N.P. Methodology: J.L. and X.W. Investigation: X.W., C.C., and X.Z. Visualization: X.W. Supervision: J.L. and N.P. Writing—original draft: X.W. Writing—review and editing: X.W., J.L., N.P., and X.Z. Competing interests: The authors declare that they have no competing interests. Data and materials availability: All data needed to evaluate the conclusions in the paper are present in the paper and/or the Supplementary Materials.

Submitted 22 November 2023
Accepted 30 April 2024
Published 31 May 2024
10.1126/sciadv.adm9673
