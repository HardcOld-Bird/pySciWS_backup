# MIT Open Access Articles

Bound states in the continuum

The MIT Faculty has made this article openly available. Please share how this access benefits you. Your story matters.

Citation: Hsu, Chia Wei et al. “Bound States in the Continuum.” Nature Reviews Materials 1.9 (2016): 16048.

As Published: http://dx.doi.org/10.1038/natrevmats.2016.48

Publisher: Nature Publishing Group

Persistent URL: http://hdl.handle.net/1721.1/108400

Version: Author's final manuscript: final author's manuscript post peer review, without publisher's formatting or copy editing

Terms of use: Creative Commons Attribution-Noncommercial-Share Alike

# Bound States in the Continuum

Chia Wei Hsu $^{1*}$ , Bo Zhen $^{2,3*}$ , A. Douglas Stone $^{1}$ , John D. Joannopoulos $^{2}$ , and Marin Soljačić $^{2}$

$^{1}$ Department of Applied Physics, Yale University, New Haven, Connecticut 06520, USA.

$^{2}$ Research Laboratory of Electronics, Massachusetts Institute of Technology, Cambridge, Massachusetts 02139, USA.

$^{3}$ Physics Department and Solid State Institute, Technion, Haifa 32000, Israel.

\*These authors contributed equally to this work.

## Abstract

Bound states in the continuum are waves that, defying conventional wisdom, remain localized even though they coexist with a continuous spectrum of radiating waves that can carry energy away. Their existence was first proposed in quantum mechanics and, being a general wave phenomenon, later identified in electromagnetic, acoustic, and water waves. They have been studied in a wide variety of material systems such as photonic crystals, optical waveguides and fibers, piezoelectric materials, quantum dots, graphene, and topological insulators. This Review describes recent developments in this field with an emphasis on the physical mechanisms that lead to these unusual states across the seemingly very different platforms. We discuss recent experimental realizations, existing applications, and directions for future work.

## Introduction

The partial or complete confinement of waves is ubiquitous in nature and in wave-based technology. Examples range from electrons bound to atoms and molecules, light confined in optical fibers, to the partial confinement of sound in music instruments. The allowed frequencies of oscillation are known as the wave spectrum. To determine whether a wave can be perfectly confined or not (i.e., whether there exists a “bound state” or not) in an open system, a simple rule of thumb is to look at its frequency: if the frequency is outside the continuous spectral range spanned by the propagating waves, it can exist as a bound state since there is no pathway for it to radiate away. Conversely, a wave state with frequency inside the continuous spectrum can only be a “resonance” that leaks and radiates out to infinity. This is the conventional wisdom as described in many textbooks. A bound state in the continuum (BIC) is an exception to this conventional wisdom: it lies inside the continuum and coexists with extended waves, but it remains perfectly confined without any radiation. As we describe in this Review, BICs can be found in a wide variety of material systems through confinement mechanisms that are fundamentally different from those of the conventional bound states.

Figure 1 illustrates the general picture. Consider waves that oscillate in a sinusoidal way as $e^{-i\omega t}$ in time t at frequency $\omega$ . Extended waves (shown in blue) exist across a continuous range of frequencies. Outside this continuum lie discrete levels of conventional bound states (shown in green) that have no access to radiation channels; this is the case for the bound electrons of an atom (at negative energies), the guided modes of an optical fiber (below the light line), and the defect modes in a bandgap. Inside the continuum one may find resonances (shown in orange) that locally resemble a bound state but in fact couple to the extended waves and leak out; they can be associated with a complex frequency $\omega = \omega_{0} - i\gamma/2$ where the imaginary part $\gamma$ represents their leakage rate. (This complex frequency is defined rigorously as the eigenvalue of the wave equation with outgoing boundary conditions, for example see refs. 1,2.) In addition to these familiar types of wave states, there is the less known possibility of BICs (shown in red) that reside inside the continuum but remain perfectly localized with no leakage. In a scattering experiment, waves coming in from infinity can excite the resonances, causing a rapid variation with spectral line-width $\gamma$ in the scattered waves. But such waves cannot excite BICs at all, as BICs are completely decoupled from the radiating waves and are invisible in this sense. One can therefore consider BIC a resonance with zero leakage and zero line-width $\gamma = 0$ (or, infinite quality factor $Q = \omega_{0}/\gamma$ ). Some authors also refer to BICs as embedded eigenvalues or embedded trapped modes.

The possibility of BICs was first proposed in 1929 by von Neumann and Wigner $^{3}$ . As an example, the authors mathematically constructed a three-dimensional (3D) potential that extends to infinity and oscillates in a way tailored to support an electronic BIC. This type of BIC-supporting systems is rather artificial and has never been realized. Since that initial proposal, many other distinct mechanisms leading to BICs have been identified in different material systems, many of which are realistic and have now been observed in experiments in electromagnetic, acoustic, and water waves. In recent years, photonic structures have emerged as a particularly attractive platform thanks to the ability to custom tailor the material and structure, which is often impossible in quantum systems. The unique properties of BICs have led to a wide range of applications from lasers and sensors to filters and low-loss fibers, with many more potential possibilities to be realized.

Most theoretically proposed and all experimentally observed BICs are realized in extended structures. The reason is that in most wave systems, BICs are forbidden in compact structures, as we explain in Box 1. Among the extended structures that support BICs, many are uniform or periodic in one or more directions (e.g., x and y), and the BIC is localized only in the other directions (e.g., z). In such systems, the concept of BICs needs to be defined carefully: since translational symmetry conserves the wave vector $\mathbf{k}_{//} = (k_x, k_y)$ , a state is considered a BIC when it exists inside the continuous spectrum of modes at the same $k_{//}$ yet remains localized and does not radiate in the z direction; these BICs can be found at isolated wave vectors and may be protected by topology (see Box 2).

The goal of this Review is to present the key concepts and physical mechanisms that unify BICs across different material systems in different types of waves. Some emphasis is placed on experimental realizations and applications. Section I describes BICs protected by symmetry and separability, section II discusses BICs achieved through parameter tuning (with coupled resonances or with a single resonance), and section III is on BICs built with inverse construction (potential, hopping rate, or shape engineering). We conclude with the existing and emerging applications of BICs and outlook.

## I. Bound states protected by symmetry or separability

The simplest places to find BICs are in systems where the couplings of certain resonances to the radiation modes are forbidden by symmetry or separability. We describe them in the following.

## Symmetry mismatch.

When a system exhibits a reflection symmetry or rotational symmetry, modes of different symmetry classes completely decouple. It is common to find a bound state of one symmetry class embedded in the continuous spectrum of modes of another symmetry class, and their coupling is forbidden as long as the symmetry is preserved.

The simplest example concerns sound waves in air, with a plate placed in the centerline of an acoustic waveguide (Fig. 2a). The fluctuation of air pressure p follows the scalar Helmholtz equation with Neumann boundary condition $\partial p/\partial n = 0$ on the surfaces of the walls and of the plate, where n is the direction normal to the surface. The waveguide supports a continuum of waves propagating in x that are either even or odd under mirror reflection in y; the odd ones (shown in orange) require at least one oscillation in y and only exist above a cutoff frequency ( $\pi c_{s}/h$ ), where $c_{s}$ is the sound speed and h is the width of waveguide. Since the plate respects the mirror symmetry, modes localized near it are also either even or odd in y, and an odd mode below the cutoff is guaranteed to be a bound state despite being in the continuum of even modes (Fig. 2a). Parker first measured $^{4}$ and analyzed $^{5}$ such modes in a cascade of parallel plates in a wind tunnel. These modes can be excited from the near field and can be heard audibly with a stethoscope placed near the plate. Many subsequent works studied this plate-in-waveguide system, for example see refs. 6–8, as well as other obstacles with arbitrary symmetric shapes $^{9}$ . Note that obstacles that are infinitesimally thin and parallel to the waveguide are decoupled from the fundamental waveguide mode even without mirror-in-y symmetry $^{10-13}$ .

Similar symmetry-protected bound states exist in canals as surface water waves $^{14-20}$ , in quantum wires $^{21-23}$ where the wave function follows the scalar Helmholtz equation with Dirichlet boundary condition, or for electrons in potential surfaces with anti-symmetric couplings $^{24}$ . A common setup is to create a 1D waveguide or lattice array that supports a continuum of even-in-y extended states, and attach two defects symmetrically above and below this array to create an odd-in-y defect bound state; such configuration has been explored with the attached defects being single-mode optical waveguides $^{25-28}$ , mechanically coupled beads $^{29,30}$ , quantum dots $^{31-41}$ , graphene flakes $^{42,43}$ , ring structures $^{37,44,45}$ , or impurity atoms $^{46,47}$ ; among these, refs. 27,28 are experimental realizations. When the mirror symmetry is broken, the bound state turns into a leaky resonance. In ref. 27, the mirror symmetry is broken by bending the defect waveguides, which allows coupling light into and out of the would-be BIC. In ref. 28 (shown in Fig. 2b), a temperature gradient changes the material's refractive index and breaks the mirror symmetry, which induces radiation in a controllable way.

Symmetry-protected BICs also exist in periodic structures. We provide an example in a photonic crystal (PhC) slab $^{48}$ where a square array of cylindrical holes are etched into a dielectric slab (Fig. 2b). Given the periodicity in the x and y directions, the photonic modes can be labeled by their wave vector $\mathbf{k}_{//} = (k_x, k_y)$ . When the 180-degree rotational symmetry around the z axis ( $C_2$ ) is preserved [for example at $\mathbf{k}_{//} = (0, 0)$ , commonly known as the $\Gamma$ point], even and odd modes with respect to $C_2$ are decoupled. At frequencies below the diffraction limit of $\omega_c = 2\pi c/na$ (where a is periodicity and n is the refractive index of the surrounding medium), the only radiating states are plane waves in the normal direction (z) with the electric and magnetic field vectors being odd under $C_2$ , so any even mode at the $\Gamma$ point is a BIC. Away from the $\Gamma$ point, these states start to couple to radiation as they are no longer protected by $C_{2}$ . Such a disappearance of radiation has been observed in periodic metallic grids by Ulrich $^{49}$ , and was well documented in early theoretical works on PhC slabs $^{50-56}$ . Fig. 2c shows experimental data measured using large-area PhC slabs $^{57}$ . The suppressed radiation has also been characterized in the lasing pattern of 1D periodic gratings $^{58,59}$ . Such photonic BICs are commonly realized in silicon photonics and with III-V semiconductors, and have found applications in lasers, sensors and filters, as we describe more in section IV.1.

In crystal acoustics, symmetry-protected BICs exist as the surface acoustic wave (SAW) in anisotropic solids, such as piezoelectric materials, which can be used to enhance the material performance beyond the typical limit of bulk materials. For example, the phase velocity $V = \omega / |\mathbf{k}_{//}|$ of a SAW is typically limited to the speed of the slowest bulk wave, since otherwise it becomes a leaky resonance. However, along high-symmetry directions, symmetry may decouple the SAW from the bulk waves, turning the resonance into a supersonic but perfectly confined SAW $^{60-64}$ , allowing higher phase velocity than the bulk limit. A related example exists in optics in uniform slabs with anisotropic permittivity and permeability tensors $^{65}$ .

## Decoupling through separability.

One may also exploit separability to construct BICs. For example, consider a two-dimensional (2D) system with Hamiltonian of the form:

$$
H = H _ {x} (x) + H _ {y} (y),\tag{1}
$$

where $H_{x}$ acts only on the x variable, and $H_{y}$ acts only on the y variable. One can separately solve the 1D eigen-problems $H_{x}\psi_{\mathrm{x}}^{(n)}(x)=E_{\mathrm{x}}^{(n)}\psi_{\mathrm{x}}^{(n)}(x)$ and $H_{y}\psi_{\mathrm{y}}^{(m)}(y)=E_{\mathrm{y}}^{(m)}\psi_{\mathrm{y}}^{(m)}(y)$ . If $\psi_{\mathrm{x}}^{(n)}(x)$ and $\psi_{\mathrm{y}}^{(m)}(y)$ are bound states of the 1D problems, their product $\psi_{\mathrm{x}}^{(n)}(x)\psi_{\mathrm{y}}^{(m)}(y)$ is bound in both dimensions and will remain localized even if its eigenvalue $E_{\mathrm{x}}^{(n)}+E_{\mathrm{y}}^{(m)}$ lies within the continuous spectrum of the extended states for the 2D Hamiltonian; coupling to the extended state is forbidden by separability. This type of BICs was first proposed by Robnik $^{66}$ and subsequently studied in other quantum systems $^{67-69}$ and in Maxwell's equation in 2D $^{70-73}$ . To date, separable BICs have not been realized experimentally, but there are promising examples in several material systems including photorefractive medium, optical traps for cold atoms, and certain lattices described by tight-binding models $^{74}$ .

## II. Bound states through parameter tuning

When the number of radiation channels is small, one may completely suppress radiation from all the channels by tuning the parameters of a system. Generally speaking, if radiation is characterized by N degrees of freedom, then at least N parameters need to be tuned to achieve a BIC. Such suppression can be interpreted as an interference effect where two or more radiating components cancel each other. Below we describe three different scenarios.

## Fabry-Pérot BIC (through coupled resonances).

It is well known that a resonant structure coupled to a single radiation channel (e.g., a defect mode coupled to a single-mode waveguide) can lead to unity reflection near the resonance frequency $\omega_{0}$ when there are no other losses; this is because the interference between the direct transmission and the resonant radiation completely cancel each other (for example, see ref. 75).

Therefore, two such resonant structures can act as a pair of perfect mirrors that trap waves in between them. BICs are formed when the resonance frequency or the spacing d between the two structures is tuned to make the round-trip phase shifts add up to an integer multiple of $2\pi$ , as illustrated in Fig. 3a. This structure is then equivalent to a Fabry-Pérot cavity formed between two resonant reflectors.

Temporal coupled-mode theory $^{76}$ provides a simple tool to model such BICs. In the absence of external driving sources, the two resonance amplitudes $A = [A_{1}, A_{2}]^{T}$ evolve in time as $i\partial A/\partial t = HA$ with Hamiltonian $^{77-79}$

$$
H = \left( \begin{array}{c c} \omega_ {0} & \kappa \\ \kappa & \omega_ {0} \end{array} \right) - i \gamma \left( \begin{array}{c c} 1 & e ^ {i \psi} \\ e ^ {i \psi} & 1 \end{array} \right),\tag{2}
$$

where $\kappa$ is the near-field coupling between the two resonators, $\gamma$ is the radiation rate of the individual resonances, and $\psi = kd$ is the propagation phase shift between the two resonators (see Fig. 3a). The two eigenvalues of H are

$$
\omega_ {\pm} = \omega_ {0} \pm \kappa - i \gamma (1 \pm e ^ {i \psi}).\tag{3}
$$

When $\psi$ is an integer multiple of $\pi$ , one of the two eigenmodes becomes a BIC with a purely real eigen-frequency, while the other eigenmode becomes more lossy with twice the original decay rate.

Fabry-Pérot BICs are commonly found in systems with two identical resonances coupled to a single radiation channel. They exist in water waves between two obstacles $^{80-85}$ as first proposed by M. McIver $^{80}$ ; these are sometimes called sloshing trapped modes $^{86}$ . In quantum mechanics, they are found in impurity pairs in a waveguide $^{20,87}$ , time-dependent double-barrier structures $^{88}$ , quantum dot pairs connected to a wire $^{89-93}$ , double metal chains on a metal substrate $^{94}$ , or double waveguide bends $^{95}$ . In photonics, they exist in structures ranging from stacked PhC slabs $^{96-98}$ and double gratings $^{99,100}$ , to off-channel resonant defects connected to a waveguide or waveguide array $^{25,27,45,77,101,102}$ . In ref. 27, light is coupled into and out of a would-be Fabry-Pérot BIC by bending the waveguides. Such BICs have also been studied in acoustic cavities $^{103}$ .

A unique property of Fabry-Pérot BICs is that the two resonators interact strongly via radiation even when they are far apart. Such long-range interactions have been studied in cavities or qubits coupled through a waveguide $^{104-106}$ and for two leaky solitons coupled through free-space radiation $^{107}$ .

The same principle applies when a single resonant structure is next to a perfectly reflecting boundary (e.g. a hard wall, lattice termination, or a PhC with bandgap). For example, Fabry-Pérot BICs exist on the surface of a photonic crystal $^{108}$ , and in a semi-infinite 1D lattice with a side-coupled defect as predicted in ref. 109 and experimentally realized in ref. 110 by Weimann et al using coupled optical waveguides (see Fig. 3b). This principle can also extend to polar or spherical coordinates $^{111,112}$ .

## Friedrich-Wintgen BIC (through coupled resonances).

The intuitive unity-reflection explanation of Fabry-Pérot BICs can be used when the two resonators are far apart. However, from equation (3), one can see that a BIC can arise even with no separation (when d = 0). In other words, two resonances at the same location can lead to a BIC through interference of radiation; unity reflection is not a requirement.

In temporal coupled-mode theory, when two resonances reside in the same cavity and are coupled to the same radiation channel, the resonance amplitudes evolve with the Hamiltonian $^{113,114}$

$$
H = \left( \begin{array}{c c} \omega_ {1} & \kappa \\ \kappa & \omega_ {2} \end{array} \right) - i \left( \begin{array}{c c} \gamma_ {1} & \sqrt {\gamma_ {1} \gamma_ {2}} \\ \sqrt {\gamma_ {1} \gamma_ {2}} & \gamma_ {2} \end{array} \right).\tag{4}
$$

Here, we consider the general scenario where the two resonances can have different resonance frequencies $\omega_{1,2}$ and different radiation rates $\gamma_{1,2}$ . The two resonances radiate into the same channel, so interference of radiation gives rise to the via-the-continuum coupling term $\sqrt{\gamma_{1}\gamma_{2}}$ . One finds that when

$$
\kappa (\gamma_ {1} - \gamma_ {2}) = \sqrt {\gamma_ {1} \gamma_ {2}} (\omega_ {1} - \omega_ {2}),\tag{5}
$$

one of the two eigenvalues becomes purely real and turns into a BIC while the other one becomes more lossy. Equation 5 was first derived by Friedrich and Wintgen $^{115}$ , so we name this type of BIC after them. Note that when $\kappa = 0$ or when $\gamma_{1} = \gamma_{2}$ , the BIC is obtained at $\omega_{1} = \omega_{2}$ ; therefore, when $\kappa \approx 0$ or $\gamma_{1} \approx \gamma_{2}$ , this type of BICs occur near the frequency crossings of the uncoupled resonances. More generally, such BICs are possible when the number of resonances N exceeds the number of radiation channels $M^{116,117}$ , but the required number of tuning parameters also grows with M.

Examples of such BICs were first proposed in atoms and molecules $^{118,119}$ , and its effect has been observed experimentally as a suppressed autoionization in certain doubly excited Rydberg states of barium $^{120}$ . More recently, Friedrich-Wintgen BICs have been studied in continuum shell models $^{121}$ , cold-atom collisions $^{122}$ , 2D topological insulators with defect $^{123}$ , and for quantum graphs $^{124}$ , quantum billiards $^{125}$ , or impurity atoms $^{126,127}$ attached to leads. In acoustics they have been studied in multi-resonant cavities $^{103,128}$ . In optics they have been studied in multi-resonant dielectric objects in microwave waveguides $^{129,130}$ , and the “dark state laser” described in ref. 131 is also a Friedrich-Wintgen BIC if one were to ignore the intrinsic radiation of the micro-ring cavities.

## Single Resonance

The preceding examples concern two (or more) coupled resonances whose radiations cancel to produce BICs. Meanwhile, a single resonance can also evolve into a BIC when enough parameters are tuned. The physical picture is similar to the previous examples; here, the single resonance itself can be thought of as arising from two (or more) sets of waves, and the radiation of the constituting waves can be tuned to cancel each other.

Fig. 3c-e shows an example realized experimentally by Hsu and Zhen et al in a PhC slab $^{132}$ . At wave-vectors away from $\mathbf{k}_{//} = (0, 0)$ , modes above the light line ( $\omega > | \mathbf{k}_{//}|c/n$ ) radiate and form leaky resonances $^{54}$ . However, at a generic k point along the $\Gamma$ -to-X direction, the resonance turns into a bound state, as evidenced by the diverging radiative quality factor $Q_{r}$ (Fig. 3d,e). The quality factor can be determined through the reflectivity spectrum $^{132}$ , or through photocurrent spectrum by embedding a detector in the slab $^{133}$ . Such BICs also exist in a linear periodic array of rectangles $^{134,135}$ , cylinders $^{136}$ , or spheres $^{137}$ , and related BICs have been found in time-periodic systems $^{138}$ . One may analyze them through spatial coupled-wave theory $^{139}$ or mode-expansion techniques $^{140}$ . While these BICs are not guaranteed to exist by symmetry, when they do exist they are robust to small changes in the system parameters, and their generation, evolution, and annihilation follow strict rules that can be understood through the concept of topological charges $^{141}$ , which also govern other types of BICs (see Box 2). One may say that such BICs are topologically protected and exist generically if the system parameters (e.g. the lattice spacing and thickness of the PhC) can be varied over a sufficient range. The topological protection of such BICs in a periodic structure has also been studied in quantum Hall insulators $^{142}$ (see Box 2).

These single-resonance parametric BICs can also exist in non-periodic structures, as shown theoretically in acoustic and water waveguides with an obstacle $^{143-148}$ , in quantum waveguides with impurities $^{149-151}$ or bends $^{95,152,153}$ , for mechanically coupled beads $^{29,30}$ and mechanical resonators $^{154}$ , and in optics for a low-index waveguide on a high-index membrane $^{155}$ .

This type of BICs exist as the SAW of anisotropic solids. For example, it was predicted $^{156,157}$ that on the (001) plane of gallium arsenide (GaAs), the leaky branch of SAW becomes a true surface wave (i.e. no leakage into the bulk) at a propagation direction of $\varphi\approx33^{\circ}$ (where $\varphi$ is the angle from the [100] direction), in addition to the more well-known symmetry-protected SAW at the [110] direction $\varphi=45^{\circ}$ (see Fig. 3f-h). The reduced attenuation near $\varphi\approx33^{\circ}$ was observed experimentally $^{158,159}$ , with data shown in Fig. 3h. Such SAWs exist in many other solids $^{160-168}$ and are sometimes called secluded supersonic SAWs $^{161}$ . With a periodic mass loading on the surface, secluded supersonic SAWs may also be found in isotropic solids $^{169-171}$ . This type of acoustic BICs was first reported in a piezoelectric material lithium niobate (LiNbO $_{3}$ ) by Yamanouchi and Shibayama $^{172}$ , ands have later been used in supersonic SAW devices $^{173-176}$ , as we describe in section IV.2.

## III. Bound states from inverse construction

Instead of looking for the presence of BICs in a given system, one may turn the problem around: start with a desired BIC, and determine what system can support this bound state and the continuous spectrum containing it. Such an inverse construction can be achieved by engineering the potential, the hopping rate, or the boundary shape of the structure.

## Potential engineering.

The first proposal of BICs by von Neumann and Wigner in 1929 was based on potential engineering $^{3}$ . For a desired BIC with wave function $\psi$ and energy E > 0, one can determine the corresponding potential V by rewriting the Schrödinger equation (in reduced units),

$$
- \frac {1}{2} \nabla^ {2} \psi + V \psi = E \psi \rightarrow V = E + \frac {\nabla^ {2} \psi}{2 \psi}.\tag{6}
$$

One must choose $\psi$ and E appropriately so that the resulting V vanishes at infinity (to support the continuum) and is well defined everywhere. There are many possible solutions. The example given by von Neumann and Wigner is $\psi(\mathbf{r}) = f(\mathbf{r}) \sin(kr)/kr$ with $f(r) = [A^{2} + (2kr - \sin(2kr))^{2}]^{-1}$ , which has an energy $E = k^{2}/2$ embedded in the continuum $E \geq 0$ . This bound wave function and the corresponding potential $V(\mathbf{r})$ from Equation 6 is shown in Fig. 4a for A = 25, $k = \sqrt{8}$ , and E = 4 (note that ref. 3 contains an algebraic mistake, as noted in refs. 177,178). More examples can be found in ref. 178, and this procedure has been generalized to non-local potentials $^{179}$ and lattice systems $^{180,181}$ . From a mathematical point of view, this inverse construction is closely related to the inverse spectral theory of the Schrödinger operator $^{182}$ and the Gel'fand-Levitan formalism of the inverse scattering problem, which can also be used to construct potentials supporting finite $^{183-186}$ or even infinite number of BICs $^{182,187}$ .

A related approach uses the Darboux transformation $^{188}$ that is commonly used in supersymmetric (SUSY) quantum mechanics to generate a family of potentials that share the same spectrum. This transformation can be applied to a free-particle extended state to yield a different potential where the corresponding state keeps its positive energy (remaining in the continuum) but becomes spatially localized $^{189-191}$ . In some cases, this SUSY method is equivalent to von Neumann and Wigner's approach and the Gel'fand-Levitan approach $^{192}$ . The SUSY method has been applied to generate BICs in point interaction systems $^{193}$ , periodic Lamé potentials $^{194}$ , and photonic crystals $^{195}$ . The SUSY method has also been extended to non-Hermitian systems with material gain and loss, where BICs are found below, above, and at the exceptional point $^{196-204}$ .

Potential engineering allows for analytic solutions of the BICs. However, the resulting potentials tend to be unrealistic—indeed none of these potentials has been realized experimentally so far. Also, perturbations generally reduce such BICs into ordinary resonances $^{205,206}$ .

## Hopping rate engineering.

A more experimentally relevant construction is to engineer the hopping rate between nearest neighbors in a tight-binding lattice model. Such construction can be carried out through the SUSY transformation $^{199,207}$ and has been realized by Corrielli et al in ref. 208 in a coupled optical waveguide array, where the hopping rate is tuned by the distance between neighboring waveguides. Intuitively, this method can also be understood as “kinetic energy engineering.”

Ref. 208 considers a semi-infinite 1D lattice where the on-site energy is constant and the hopping rate $\kappa_{n}$ between sites n and $n+1$ follows (Fig. 4b)

$$
\kappa_ {n} = \left\{ \begin{array}{l l} \kappa & n \neq l N, \\ \left(\frac {l + 1}{l}\right) ^ {\beta} \kappa & n = l N, (l = 1, 2, 3, \dots), \end{array} \right.\tag{7}
$$

where N > 1 is an integer, $\beta > 1/2$ is an arbitrary real number, and $\kappa$ is the reference hopping rate. This system supports N - 1 BICs localized near the surface (n = 1). The particular case of N = 2, $\beta = 1$ was experimentally realized with 40 evanescently coupled optical waveguides $^{208}$ . Fig. 4b shows the theoretical hopping rates and the BIC mode intensity $|c_{n}|^{2}$ , together with the experimentally measured intensity when light is launched from the first site.

## Boundary shape engineering.

Instead of engineering the wave equation itself, one can also engineer the boundary shape to achieve BICs. M. McIver first proposed such method in the context of water waves $^{80}$ . In ref. 80, two line sources are placed at distance d apart on the water surface such that $kd = \pi$ , where k is the transverse wavenumber. Surface wave radiations from the two sources cancel, resulting in a spatially confined mode profile. Then, the two line sources are replaced with two obstacles whose boundary shapes correspond to streamlines of the mode profile that contain the two sources. In this way, the mode profile in the original driven system is a BIC in the new un-driven system with obstacles, since it satisfies the Neumann boundary condition on the obstacle surface by the definition of the streamline.

BICs constructed with two line sources or a ring of sources typically lead to the Fabry-Pérot type $^{80,82,83,111,112}$ as described earlier. But this procedure can also be extended to more complex shapes $^{209,210}$ and to free-floating instead of fixed obstacles $^{211-213}$ .

## IV. Applications of BICs and quasi-BICs

The unique properties of BICs can enable a wide range of applications. Here we describe the applications that have already been realized, leaving the potential applications to the outlook section.

## Lasing, sensing and filtering.

Structures with BICs are natural high-Q resonators, since the radiative Q is, in the ideal case, infinity. This makes them useful for many optical and photonic applications. In particular, the macroscopic size (on the centimeter scale or larger) and ease of fabrication make BICs in PhC slabs unique for large-area high-power applications such as lasers $^{214-219}$ , sensors $^{220,221}$ , and filters $^{222}$ .

A large number of surface-emitting lasers are based on symmetry-protected BICs at the $\Gamma$ point, similar to those shown in Fig. 2c. Researchers first observed this effect via a suppression of radiation into the normal direction in a surface-emitting distributed feedback laser with one-dimensional periodicity $^{58,59}$ . This led to PhC surface emitting lasers (PCSELs) that lase through BICs with two-dimensional periodicity $^{214,215}$ , followed by the realizations of various lasing patterns $^{216,217}$ , lasing at the blue-violet wavelengths $^{218}$ , and lasing with organic molecules $^{221}$ . The suppressed radiation at the normal direction means that a PCSEL can have a low lasing threshold but also with a limited output power. Therefore, recent designs intentionally break the $C_{2}$ symmetry to allow some radiation into the normal direction. Fig. 5a shows results from ref. 219, where the $C_{2}$ symmetry is broken using triangular air-holes; this work achieves continuous-wave lasing at room temperature with 1.5 watt output power and high beam quality ( $M^{2} \leq 1.1$ ), while the threshold is still relatively low. Also, PCSELs produce vector beams $^{223,224}$ with the order numbers given by the topological charges of the BICs $^{141}$ (see Box 2), which may find applications in super-resolution microscopy and in table-top particle accelerators (see ref. 225 for a review on vector beams).

Another application lies in chemical and biological sensing, particularly in optofluidic setups $^{226}$ . One sensing mechanism uses the shift of resonance frequency to detect the change of refractive index in the surroundings. Resonators with higher Qs enable narrower line-widths and higher sensitivity; researchers were able to directly visualize a single mono-layer of proteins with the naked eye using the high-Q resonances close to a BIC $^{220}$ . Another type of sensing relies on measuring fluorescence signals. It was shown that the spontaneous emission from organic molecules can be strongly enhanced and the angular distribution can be strongly modulated near BICs in a PhC slab, leading to a total enhancement of angular fluorescence intensity by 6300 times $^{221}$ . BICs also enabled large-area narrow-band filters in the infrared regime $^{222}$ due to their high and tunable Q factors.

## Supersonic surface acoustic wave devices.

BICs on the surface of anisotropic solids, such as the example shown in Fig. 3f-h, can enable useful acoustic devices. Fig. 5c shows a schematic setup: an interdigital transducer placed on a piezoelectric substrate converts the input electric signal into acoustic wave, which propagates through a surface acoustic wave (SAW) to the other side and gets converted back to electric signal on the output. While regular SAWs are subsonic whose speed is limited by the speed of the bulk waves, a BIC allows propagation at a much faster supersonic speed (Fig. 5d). The BIC propagates at a fixed direction ( $\varphi = 36^{\circ}$ in Fig. 5g), as other directions are lossy. The spatial periodicity of the interdigital transducer determines the wavenumber $|\mathbf{k}_{//}|$ , while the angular frequency of the SAW is given by $\omega = |\mathbf{k}_{//}|V$ , where $V$ is the phase velocity. Such devices are commonly used as filters; a characteristic filtering spectrum using supersonic SAW filter on Y-X cut LiTaO $_{3}$ is shown in Fig. 5e $^{176}$ . Supersonic SAW filters based on BICs are widely used in cell phones and cordless phones, Bluetooth devices, and delay lines $^{173-175}$ , due to their advantages of low loss, high piezoelectric coupling while reasonable temperature stability, excellent accuracy and repeatability, and compatibility with photolithography; see ref. 227 for more information.

## Guiding photons in gapless PhC fibers.

PhC fibers can guide light in a low-index material through a photonic bandgap $^{48}$ , but the bandwidth is limited by the width of the bandgap. A type of hollow-core Kagome-lattice PhC fiber (shown in Fig. 5f) can provide wave guiding inside the continuum without a bandgap $^{228,229}$ . Its mechanism, referred to as “inhibited coupling” by some, is due to the dissimilar azimuthal dependence of the core and cladding modes: the core mode varies slowly with angle, while the cladding mode is fast oscillating (see Fig. 5f). While such fiber modes are not true BICs because there can be residual radiation (the propagation loss is typically 1 dB/m), they enable broadband guidance in air and have found many applications including multiple-octave frequency comb generations $^{229}$ (shown in Fig. 5d), all-fiber gas cells $^{230,231}$ , particle transportation and levitation $^{232}$ , and Raman sensing $^{233}$ .

## V. Outlook

As a general wave phenomenon, bound states in the continuum arise through a number of distinct mechanisms and exist in a wide variety of material systems. This Review has described the main mechanisms with examples drawn from atomic and molecular systems, quantum dots, electromagnetic waves, acoustic waves in air, water waves, and elastic waves in solids.

We have not covered all possible mechanisms. For example, BICs in systems with chiral symmetry $^{234}$ are distinct from the symmetry-protected BICs. In some two-particle Hubbard models, there are bound states that can move into and out of the continuum continuously $^{235-238}$ ; the confinement requires no parameter tuning and has been credited to integrability $^{236}$ . Systems with a perfectly flat band can support localized states $^{239,240}$ . Localization can be induced with strong gain and/or loss, for example in a defect site with high loss $^{241}$ and in the bulk $^{200,202,242,243}$ or on the surface $^{244}$ of parity-time symmetric systems; ref. 245 realized this in a synthetic photonic lattice. There may also be more constructions not yet discovered.

Even though the very first proposal $^{3}$ and a large body of theoretical works concern BICs in quantum systems, there has not been any conclusive observation of a quantum BIC except for the suppressed line-width in Rydberg atoms $^{120}$ . The system studied in ref. 246 is frequently mistaken as a BIC by others (even though the paper does not make such claim), but it is a conventional bound state as it lies in a band gap instead of a continuum. Ref. 247 claims a BIC in a multiple-quantum-wells structure but with data indicating a finite leakage rate and no evidence for localization. The difficulty arises from the relatively few control parameters and the large number of decay pathways in quantum systems. Therefore, the realization of a quantum BIC remains a challenge.

Optical systems provide a clean and versatile platform to realize different types of BICs $^{27,28,49,57,110,132,208,245}$ , due to the advanced nano-fabrication technologies that enable the creation of custom photonic structures. An optical BIC exhibits an ultrahigh quality factor—its radiative quality factor is technically infinity—which can increase the interaction time between light and matter by orders of magnitude. While this Review has described some applications that utilize the high Q, there are many more opportunities such as in nonlinearity enhancement and in quantum optical applications that have not been explored. The long-range interactions in Fabry-Pérot BICs may be useful for nanophotonic circuits $^{104}$ and for quantum information processing $^{105,106}$ . It has also been proposed that the light intensity may act as another tuning parameter in nonlinear materials, which may enable robust BICs $^{248}$ , tunable channel dropping $^{249}$ , light storage and release $^{250,251}$ , and frequency comb generation $^{252}$ . Finally, it was shown that particle statistics can be used to modify some properties of BICs $^{253}$ .

Given the many types of BICs, a natural question is whether a common concept underlies all of them aside from the vanishing of coupling to radiation via interference. To this end, the topological interpretation of BICs (see Box 2 and refs. 141,142,254) seems promising. The topological arguments may guide the discovery of BICs and new ways to trap waves, which may also exist in quasi-particle systems such as magnons, polaritons, polarons, and anyons. Since BICs defy the conventional wisdom and provide new ways to confine waves, their realization in different material systems are certain to provide even more surprises and advances in both fundamental physics and technological applications.

## Acknowledgement

We thank A. Maznev, N. Rivera, F. Wang, M. Segev, N. Moiseyev, S. Longhi, P. McIver, M. McIver, Fetah Benabid, and S. G. Johnson for discussions. This work was partly supported by the National Science Foundation through grant no. DMR-1307632 and by the Army Research Office through the Institute for Soldier Nanotechnologies under contract no. W911NF-13-D-0001. B.Z., J.D.J. and M.S. were partly supported by S3TEC (analysis and reading of the manuscript), an Energy Frontier Research Center funded by the US Department of Energy under grant no. DE-SC0001299. B.Z. was partially supported by the United States-Israel Binational Science Foundation (BSF) under award no. 2013508.

<table><tr><td>Box 1 | Non-existence of single-particle BICs in compact structures.</td></tr><tr><td>Most BIC-supporting structures extend to infinity in at least one direction. This is because BICs are generally forbidden in compact structures for single-particle-like systems.</td></tr><tr><td>Consider a 3D compact optical structure in air, characterized by its permittivity ε(r) and permeability μ(r), and let R be the radius of a sphere that encloses the structure. For |r| &gt; R, ε(r) = μ(r) = 1, so the electric and magnetic fields (E, H) follow the Helmholtz equation and can be expanded in spherical harmonics and spherical Hankel functions with wavenumber k = ω/c. A bound state must not carry radiating far field, but every term in the expansion carries an outgoing Poynting flux, so all terms must be zero, meaning E and H must both vanish for |r| &gt; R. If ε(r) and μ(r) are neither infinite nor zero anywhere, continuity of the fields requires E and H to be zero everywhere in space, so such a bound state cannot exist. Ref. 255 provides a more rigorous proof. The same argument applies when the structure and the fields depend on two or one coordinates only.</td></tr><tr><td>This non-existence theorem does not exclude compact BICs when the material has ε = ±∞, μ = ±∞, ε = 0, or μ = 0, which can act as hard walls that spatially separate the bound state from the extended ones. Examples with ε = 0 are proposed in refs. 251,255,256 but are difficult to realize since the metal loss Im(ε) is typically significant at the plasmon frequency where Re(ε) = 0.</td></tr><tr><td>The same argument applies to the single-particle Schrödinger equation. For an electron with a non-vanishing effective mass m (the m = 0 case is studied in ref. 257) in a compact and finite potential (V(r) = 0 for |r| &gt; R, and V(r) ≠ ±∞ everywhere), a bound state with positive energy E &gt; 0 cannot exist. Similarly, this non-existence theorem can be applied to acoustic waves in air and to linearized water waves in constant-depth (z-independent) structures, since both systems are described by the Helmholtz equation. However, this theorem does not apply to water waves in structures with z dependence (e.g., ref. 80), which follow the Laplace equation instead.</td></tr></table>

## Box 2 | Topological nature of BICs.

Perturbations typically turn a BIC into a leaky resonance. However, some BICs are protected topologically and cannot be removed except by large variations of the system parameters.

The topological nature of BICs in PhC slabs is described in ref. 141. For a general resonance on a PhC slab, the polarization direction of the far-field radiation is given by a two-dimensional vector $\mathbf{E}_{//} = (E_x, E_y)$ as shown below in Fig. B2a. BICs do not radiate, so they exist at the crossing points between the nodal lines of $E_x = 0$ and those of $E_y = 0$ . In the k space, the polarization vector forms a vortex around each BIC with a corresponding “topological charge” q; a few examples (q = -2, -1, and 1) are shown in Fig. B2b, as well as the case with no BIC (q = 0). Once any crossing (BIC) occurs, large changes in the system parameters are required to remove it. Since the topological charges are conserved quantities protected by the boundary conditions $^{258}$ , a BIC of this type can only be removed when it annihilates with another BIC of the opposite charge.

a) Polarization vector $E_{//}$
![](images/7656ad680f9aadb3b5a80f787c44bb269c6ee1c4ee7951e99de812a9c50fc184.jpg)

b)
Topological charges of BICs
![](images/69a7743b76a40d41f31c1c2a58b51b05f2674dfd8c9bde4fb304829ce121d206.jpg)
For electrons, the topological properties of BICs was studied in the case where a two-dimensional quantum Hall insulator is placed on top of a bulk normal insulator $^{142}$ . At isolated k points, pure surface modes embedded in the continuum of the bulk modes can be found. Their existence is required by the dissimilar topological invariants: the bulk band of the quantum Hall insulator has a non-zero Chern number, while that of the normal insulator has a zero Chern number (see ref. 259 for a review on Chern numbers). In fact, BICs in this electronic system can also be understood as topological defects, whose existence is protected by the Chern number difference $^{254}$ .

Figures
![](images/aa3189a542a3f52db24e51691a1d1fb0546a54a5ce7d815e08a5738fb1a1bcde.jpg)
Fig. 1 | General illustration of a bound state in the continuum (BIC). In an open system, the frequency spectrum consists of a continuum or several continua of spatially extended states (shown in blue) and discrete levels of bound states (shown in green) that carry no outgoing flux. The purple dashed line illustrates the structure that provides confinement. States inside the continuous spectrum typically couple to the extended waves and radiate, becoming leaky resonances (shown in orange). BICs (shown in red) are special states that lie inside the continuum but remain localized with no radiation.

a Acoustic waveguide
![](images/881d181deae2c89691db77eeb0b434bf618fc68099cf807607db79c7b3a9a831.jpg)

b Coupled waveguide array
![](images/6f973789ccb48dff73c1ccdbe670489943964e2903e1dc60b21c86b4277db83e.jpg)

![](images/2684bf9b659e1dce06e3f6ac6e4d5b1b2d5126b9990dd54ff254d5a758d31351.jpg)

c Photonic crystal slab
![](images/5fffbdab6602126a5529ca15b686b1855017d99847ea440c88eed82131faa8f6.jpg)

![](images/68ee9809693ea92dcc09d2a99edf2a319bcdd4d596f3384fa11df6789905dda0.jpg)

![](images/ebe6424d7a98391b64baf4d5609df07ee41cf008ed6380cc0b87a4e802fada25.jpg)

![](images/240e0816872b0bcfd9f286482b8d8d9e5a44f67d8328df66691216165d10f860.jpg)
Fig. 2 | Symmetry-protected bound states. a, An acoustic waveguide with an obstructing plate (shown in black) placed at the center. An odd bound state exists at the same frequency as an even extended state but cannot couple to it. Measuring the sound pressure near the plate reveals the bound state (bottom panel). b, A coupled-waveguide system with two defects placed symmetrically parallel to a linear array, which supports a similar odd bound state. The propagation constant $\beta_{z}$ plays the role of frequency. A temperature gradient can break the mirror symmetry via thermo-optic effect and turn the bound state into a leaky resonance (bottom panel). c, A photonic crystal (PhC) slab with a 180-degree rotational symmetry around the $z$ -axis ( $C_2$ ). At the $\Gamma$ point, modes that are even under $C_2$ cannot radiate because planewaves in the normal direction are odd under $C_2$ . Away from the normal direction, the bound states become leaky with finite Qs, as confirmed by reflectivity measurements (bottom panel). Experimental data obtained with permission from: a, ref. 6, © 1971 Elsevier; b, ref. 28, © 2011 APS; c, ref. 57, © 2012 APS.

## Fabry-Pérot BIC (coupled resonances)

a Mechanism
![](images/b9de4e2649e6b104675c982503e62ba70e045886cc60d79faab43e78d002cba1.jpg)

b Realization
![](images/6f78b26206ee465b1c1dd35d25b3263ab6608acdb536634ec9ef94b02d6db5df.jpg)

![](images/58dfac1fd683bc01d6b48945f71fe9bd55a9f110f61d76f2f6749a90e988d00a.jpg)

![](images/7be21aa2a701b82e1af76c1a3f0e126b4af3f6999b0430f735479f4bbd36f544.jpg)
Single-resonance parametric BIC

c
![](images/7b2b89c6a8f4bb4a8518204abae8c528cd29dbad3eb08f129ee2975359bca703.jpg)

![](images/99b9a7c0e646fd491fb2fb42bb95673dc441bba4ead03dbdeee877d6adba4247.jpg)

![](images/b0eae6a948db5cfd4c52de08ad8c89d25b38c63d0c24dee21a19f8f79abffc95.jpg)

![](images/dbf3fde293060b9ab1b63d7d3fbc614582e2b3d1a61328dd4d84ed61c5fd3e80.jpg)

![](images/b078fdb7a23ca10a6c447aa7bf9855ebe4d79b8adf48fbe6a5cc36d982082c4a.jpg)

![](images/97a6e239cee17fe229b5d78132308fb8764464a5ad2e1be6a6d12a1ebaa58756.jpg)
Fig. 3 | BICs through parameter tuning. a, Schematic illustration of the Fabry-Pérot BIC. Two resonances are coupled to one radiation channel, and each resonance acts as a perfect reflector near the resonance frequency $\omega_0$ , so the two can trap waves in between when the round-trip phase shift is an integer multiple of $2\pi$ . b, Realization of a Fabry-Pérot BIC in a semi-infinite coupled waveguide array, where the defect waveguide (F) and its mirror image with respect to the end play the role of the two resonances. c-e, BIC from a single resonance in a PhC slab. c, Schematic of the system. d, Photonic band structure. The leaky resonance (orange line) turns into two BICs: one at $\mathbf{k}_{//} = (0, 0)$ (due to symmetry) and the other at $\mathbf{k}_{//} \approx (0.27, 0)$ $2\pi/a$ (through tuning) as marked by red circles. e, Radiative quality factor $Q_r$ determined from reflectivity measurements. f-h, BIC from the leaky surface acoustic wave (SAW) of GaAs. f, The (001) surface of GaAs. g, Acoustic band structure. Radiation of the leaky SAW (orange line) vanishes at $\varphi = 45^\circ$ (due to symmetry) and at $\varphi \approx 33^\circ$ (through tuning). h, Theoretical attenuation in log scale and measured resonance line-width in linear scale. Figures adapted with permission from: b, ref. 110, © 2013 APS; c-e, ref. 132, © 2013 NPG; g, ref. 157, © 1976 AIP; h, ref. 159, © 1992 Elsevier.

a)
![](images/b0933d0be12725b2d2999dc055ba483d047496abe2bd3c3f4869cda76dcd4614.jpg)
Potential engineering
b)

![](images/9bb2e7ffc92dcff21961111e56e5a3224a22cd435795e8cd52543643b7cda055.jpg)
Hopping rate engineering

![](images/3d7599fd9ae2549a576be1740f53ac290f1543cc01d2c46e37e5a87ba7289b59.jpg)

![](images/952ad969e4b8153ed10f8c5029988e906c1a8d90a85ee87e69f5650d3365cfe5.jpg)

![](images/7165316cd8b2ca54aa776d24bb1e169995da33f536ab2d62847bfcd4a9bc0b18.jpg)

![](images/9772b4610acc9f47910986db9a0bd4c77744f88ef49d30c10b922388a312fae6.jpg)
z axis (cm)
Fig. 4 | BICs through inverse construction. a, The BIC proposed by von Neumann and Wigner $^{3}$ . A potential (left) is engineered to support a localized electron wave function (right) with its energy embedded in the continuous spectrum of extended states. b, Construction of a BIC by engineering the hopping rates in a semi-infinite lattice system. The hopping rates $\kappa_{n}$ (top left) follow Equation 7 to support a bound state (lower left) at $\beta_{z}=0$ , embedded in the continuum of the extended states ( $-2\kappa\leq\beta_{z}\leq2\kappa$ ). This BIC is experimentally realized in an array of coupled optical waveguides (top right); light launched at one end of the array excites a BIC that propagates along the waveguides (intensity image shown in the bottom right). Figures reproduced with permission from: b, ref. 208, © 2013 APS.

Photonic crystal surface emitting lasers
a) Experimental setup

![](images/533d6b2b46d81aacca3768c6a26acd43f4fd8c2b75d84aa3171cae120abfaa94.jpg)
Supersonic surface acoustic wave devices
c) Setup

![](images/fbe081e553127f30556c8fb2acb18029cffc9278de953aa9633f1031bcc46069.jpg)
Guiding photons in gapless PhC fibers
f) Hollow-core Kagome-lattice fiber

![](images/e0606b3947df60e84e1dc1e867b392d2875d5059e1500d5bec5b81abd1610791.jpg)

![](images/7205592ab256d299bd886b60fea37e6ee03f1e82c7628abd8660576032b139dc.jpg)

![](images/2139991ce80424d3f0cfcf8a52a06822c8bf855b393ca938d3f9f991543d4dde.jpg)

![](images/3a89b40e3304d612b196ead0e81ac9d743b2a04eb8082435ac71006d141433fb.jpg)

![](images/412d7932ca7516d5f38fe38a0e46a421edff5518b8364c567d718a731a2d36df.jpg)

![](images/e73802b03273a801ad360fca20b168d9f289ecdce3e2db72b813983ad0915296.jpg)

![](images/3e53d860636a229a1a6e1642fc70ff57f9dfcf16f63ea965a875c69c0e8a3c22.jpg)
Fig. 5 | Applications of BICs and quasi-BICs. a,b, Photonic crystal surface-emitting lasers (PCSELs). a, Schematic drawing of the setup. The lasing mode here is a quasi-BIC, since the 180-degree rotational symmetry of the PhC is broken by the triangular air-hole shapes. b, The input-output curve of the PCSEL operating under room-temperature continuous-wave condition demonstrating a low threshold and a high output power. c-e, Supersonic surface acoustic wave (SAW) filters. c, Schematic drawing of the setup: two interdigital transducers are placed on a piezo-electric substrate along the direction of the acoustic BIC. d, Comparison between the phase velocities of supersonic and subsonic SAWs on Y-X cut LiTaO₃. e, A characteristic spectrum through a supersonic SAW filter on the surface of LiTaO₃. f,g, Guiding photons without bandgaps. f, Upper panels: scanning electron microscope (SEM) images of a hollow-core Kagome-lattice PhC fiber. Photonic guiding in such fibers uses quasi-BICs relying on the “inhibited coupling” between the core and cladding modes (lower panels). g, Image and spectrum showing the generation and guidance of a three-octave spectral comb using the quasi-BICs in such fibers. Figures reproduced with permission from: a,b, ref. 219, © 2014 NPG; d, ref. 227, © 2007 Academic Press; e, ref. 176, © 2002 IEEE; f,g, ref. 229, © 2007 AAAS.

## References

1. Moiseyev, N. Non-Hermitian Quantum Mechanics. (Cambridge University Press, 2011).

2. Kukulin, V. I., Krasnopol'sky, V. M. & Horáček, J. Theory of Resonances: Principles and Applications. (Springer, 1989).

3. von Neumann, J. & Wigner, E. Über merkwürdige diskrete Eigenwerte. Phys. Z. 30, 465–467 (1929). This paper proposes the possibility of BICs using an engineered quantum potential as an example.

4. Parker, R. Resonance effects in wake shedding from parallel plates: Some experimental observations. J. Sound Vib. 4, 62–72 (1966). This paper reports the observation of symmetry-protected BICs in acoustic waveguides.

5. Parker, R. Resonance effects in wake shedding from parallel plates: Calculation of resonant frequencies. J. Sound Vib. 5, 330–343 (1967).

6. Cumpsty, N. A. & Whitehead, D. S. The excitation of acoustic resonances by vortex shedding.
J. Sound Vib. 18, 353–369 (1971).

7. Koch, W. Resonant acoustic frequencies of flat plate cascades. J. Sound Vib. 88, 233–242 (1983).

8. Parker, R. & Stoneman, S. A. T. The Excitation and Consequences of Acoustic Resonances in Enclosed Fluid Flow Around Solid Bodies. Proc. Inst. Mech. Eng. C J. Mech. Eng. Sci. 203, 9–19 (1989).

9. Evans, D. V., Levitin, M. & Vassiliev, D. Existence theorems for trapped modes. J. Fluid Mech. 261, 21–31 (1994).

10. Evans, D. V., Linton, C. M. & Ursell, F. Trapped mode frequencies embedded in the continuous spectrum. Q. J. Mech. Appl. Math. 46, 253–274 (1993).

11. Groves, M. D. Examples of embedded eigenvalues for problems in acoustic waveguides.
Math. Method. Appl. Sci. 21, 479–488 (1998).

12. Linton, C. & McIver, M. Trapped modes in cylindrical waveguides. Q. J. Mech. Appl. Math. 51, 389–412 (1998).

13. Davies, E. & Parnovski, L. Trapped modes in acoustic waveguides. Q. J. Mech. Appl. Math. 51, 477–492 (1998).

14. Ursell, F. Trapping modes in the theory of surface waves. Math. Proc. Cambridge 47, 347-358 (1951).

15. Jones, D. S. The eigenvalues of $\nabla^{2}u + \lambda u = 0$ when the boundary conditions are given on semi-infinite domains. Math. Proc. Cambridge 49, 668–684 (1953).

16. Callan, M., Linton, C. M. & Evans, D. V. Trapped modes in two-dimensional waveguides. J. Fluid Mech. 229, 51–64 (1991).

17. Retzler, C. H. Trapped modes: an experimental investigation. Appl. Ocean Res. 23, 249–250 (2001).

18. Cobelli, P. J., Pagneux, V., Maurel, A. & Petitjeans, P. Experimental observation of trapped modes in a water wave channel. Euro. Phys. Lett. 88, 20006 (2009).

19. Cobelli, P. J., Pagneux, V., Maurel, A. & Petitjeans, P. Experimental study on water-wave trapped modes. J. Fluid Mech. 666, 445–476 (2011).

20. Pagneux, V. in Dynamic Localization Phenomena in Elasticity, Acoustics and Electromagnetism (eds. Craster, R. & Kaplunov, J.) 547, 181–223 (Springer Vienna, 2013).

21. Schult, R. L., Ravenhall, D. G. & Wyld, H. W. Quantum bound states in a classically unbound system of crossed wires. Phys. Rev. B 39, 5476–5479 (1989).

22. Exner, P., Šeba, P., Tater, M. & Vaněk, D. Bound states and scattering in quantum waveguides coupled laterally through a boundary window. J. Math. Phys. 37, 4867–4887 (1996).

23. Moiseyev, N. Suppression of Feshbach resonance widths in two-dimensional waveguides and quantum dots: a lower bound for the number of bound states in the continuum. Phys. Rev. Lett. 102, 167404 (2009).

24. Cederbaum, L. S., Friedman, R. S., Ryaboy, V. M. & Moiseyev, N. Conical Intersections and Bound Molecular States Embedded in the Continuum. Phys. Rev. Lett. 90, 013001 (2003).

25. Longhi, S. Transfer of light waves in optical waveguides via a continuum. Phys. Rev. A 78, 013815 (2008).

26. Longhi, S. Optical analog of population trapping in the continuum: Classical and quantum interference effects. Phys. Rev. A 79, 023811 (2009).

27. Dreisow, F. et al. Adiabatic transfer of light via a continuum in optical waveguides. Opt. Lett. 34, 2405–2407 (2009). This paper realizes light transfer based on symmetry-protected and Fabry-Pérot BICs in a coupled-waveguide array.

28. Plotnik, Y. et al. Experimental observation of optical bound states in the continuum. Phys. Rev. Lett. 107, 183901 (2011). This paper realizes an optical symmetry-protected BIC in a coupled-waveguide array.

29. Shipman, S. P., Ribbeck, J., Smith, K. H. & Weeks, C. A Discrete Model for Resonance Near Embedded Bound States. IEEE Photon. J. 2, 911–923 (2010).

30. Ptitsyna, N. & Shipman, S. P. A lattice model for resonance in open periodic waveguides. Discrete Contin. Dyn. Syst. Ser. S 5, 989–1020 (2012).

31. Ladrón de Guevara, M. L., Claro, F. & Orellana, P. A. Ghost Fano resonance in a double quantum dot molecule attached to leads. Phys. Rev. B 67, 195335 (2003).

32. Orellana, P. A., Ladrón de Guevara, M. L. & Claro, F. Controlling Fano and Dicke effects via a magnetic flux in a two-site Anderson model. Phys. Rev. B 70, 233315 (2004).

33. Ladrón de Guevara, M. L. & Orellana, P. A. Electronic transport through a parallel-coupled triple quantum dot molecule: Fano resonances and bound states in the continuum. Phys. Rev. B 73, 205303 (2006).

34. Voo, K.-K. & Chu, C. S. Localized states in continuum in low-dimensional systems. Phys. Rev. B 74, 155306 (2006).

35. Solís, B., Ladrón de Guevara, M. L. & Orellan, P. A. Friedel phase discontinuity and bound states in the continuum in quantum dot systems. Phys. Lett. A 372, 4736–4739 (2008).

36. Gong, W., Han, Y. & Wei, G. Antiresonance and bound states in the continuum in electron transport through parallel-coupled quantum-dot structures. J. Phys. Condens. Matter 21, 175801 (2009).

37. Han, Y., Gong, W. & Wei, G. Bound states in the continuum in electronic transport through parallel-coupled quantum-dot structures. Phys. Status Solidi B 246, 1634–1641 (2009).

38. Vallejo, M. L., Ladrón de Guevara, M. L. & Orellana, P. A. Triple Rashba dots as a spin filter: Bound states in the continuum and Fano effect. Phys. Lett. A 374, 4928 - 4932 (2010).

39. Yan, J.-X. & Fu, H.-H. Bound states in the continuum and Fano antiresonance in electronic transport through a four-quantum-dot system. Physica B 410, 197 – 200 (2013).

40. Ramos, J. P. & Orellana, P. A. Bound states in the continuum and spin filter in quantum-dot molecules. Physica B 455, 66 - 70 (2014).

41. Álvarez, C., Domínguez-Adame, F., Orellana, P. A. & E. Díaz. Impact of electron-vibron interaction on the bound states in the continuum. Phys. Lett. A 379, 1062–1066 (2015).

42. González, J. W., Pacheco, M., Rosales, L. & Orellana, P. A. Bound states in the continuum in graphene quantum dot structures. Euro. Phys. Lett. 91, 66001 (2010).

43. Cortés, N., Chico, L., Pacheco, M., Rosales, L. & Orellana, P. A. Bound states in the continuum: Localization of Dirac-like fermions. Euro. Phys. Lett. 108, 46008 (2014).

44. Bulgakov, E. N., Pichugin, K. N., Sadreev, A. F. & Rotter, I. Bound states in the continuum in open Aharonov-Bohm rings. JETP Lett. 84, 430–435 (2006).

45. Voo, K.-K. Trapped electromagnetic modes in forked transmission lines. Wave Motion 45, 795 - 803 (2008).

46. Guessi, L. H. et al. Catching the bound states in the continuum of a phantom atom in graphene. Phys. Rev. B 92, 045409 (2015).

47. Guessi, L. H. et al. Quantum phase transition triggering magnetic bound states in the continuum in graphene. Phys. Rev. B 92, 245107 (2015).

48. Joannopoulos, J. D., Johnson, S. G., Winn, J. N. & Meade, R. D. Photonic Crystals: Molding the Flow of Light. (Princeton University Press, 2008).

49. Ulrich, R. Modes of propagation on an open periodic waveguide for the far infrared. in Symposium on Optical and Acoustical Micro-Electronics (ed. Fox, J.) 359–376 (1975). This work observes a symmetry-protected BIC in a periodic metal grid.

50. Bonnet-Bendhia, A.-S. & Starling, F. Guided waves by electromagnetic gratings and non-uniqueness examples for the diffraction problem. Math. Method Appl. Sci. 17, 305–338 (1994).

51. Paddon, P. & Young, J. F. Two-dimensional vector-coupled-mode theory for textured planar waveguides. Phys. Rev. B 61, 2090–2101 (2000).

52. Pacradouni, V. et al. Photonic band structure of dielectric membranes periodically textured in two dimensions. Phys. Rev. B 62, 4204–4207 (2000).

53. Ochiai, T. & Sakoda, K. Dispersion relation and optical transmittance of a hexagonal photonic crystal slab. Phys. Rev. B 63, 125107 (2001).

54. Fan, S. & Joannopoulos, J. D. Analysis of guided resonances in photonic crystal slabs. Phys. Rev. B 65, 235112 (2002).

55. Tikhodeev, S. G., Yablonskii, A. L., Muljarov, E. A., Gippius, N. A. & Ishihara, T. Quasiguided modes and optical properties of photonic crystal slabs. Phys. Rev. B 66, 045102 (2002).

56. Shipman, S. P. & Venakides, S. Resonant transmission near nonrobust periodic slab modes. Phys. Rev. E 71, 026611 (2005).

57. Lee, J. et al. Observation and differentiation of unique high-Q optical resonances near zero wave vector in macroscopic photonic crystal slabs. Phys. Rev. Lett. 109, 067401 (2012).

58. Henry, C. H., Kazarinov, R. F., Logan, R. A. & Yen, R. Observation of destructive interference in the radiation loss of second-order distributed feedback lasers. IEEE J. Quantum Electron. 21, 151–154 (1985). This paper demonstrates lasing through a symmetry-protected BIC in a distributed feedback laser with one-dimensional periodicity.

59. Kazarinov, R. F. & Henry, C. H. Second-order distributed feedback lasers with mode selection provided by first-order radiation losses. IEEE J. Quantum Electron. 21, 144–150 (1985).

60. Lim, T. C. & Farnell, G. W. Character of Pseudo Surface Waves on Anisotropic Crystals. J. Acoust. Soc. Am. 45, 845–851 (1969).

61. Farnell, G. W. in Physical Acoustics (eds. Mason, W. P. & Thurston, R. N.) 6, 109–166 (Academic Press, 1970).

62. Alshits, V. I. & Lothe, J. Comments on the relation between surface wave theory and the theory of reflection. Wave Motion 3, 297–310 (1981).

63. Chadwick, P. The Behaviour of Elastic Surface Waves Polarized in a Plane of Material Symmetry I. General Analysis. Proc. R. Soc. A 430, 213–240 (1990).

64. Alshits, V. I., Darinskii, A. N. & Shuvalov, A. L. Elastic waves in infinite and semi-infinite anisotropic media. Phys. Scripta 1992, 85 (1992).

65. Shipman, S. P. & Welters, A. Resonance in anisotropic layered media. in Proceedings of the 2012 International Conference on Mathematical Methods in Electromagnetic Theory (MMET) 227–232 (2012).

66. Robnik, M. A simple separable Hamiltonian having bound states in the continuum. J. Phys. A 19, 3845 (1986). This paper proposes BICs in a quantum well based on separability.

67. Nockel, J. U. Resonances in quantum-dot transport. Phys. Rev. B 46, 15348 (1992).

68. Duclos, P., Exner, P. & Meller, B. Open quantum dots: resonances from perturbed symmetry and bound states in strong magnetic fields. Rep. Math. Phys. 47, 253–267 (2001).

69. Prodanović, N., Milanović, V., Ikonić, Z., Indjin, D. & Harrison, P. Bound states in continuum: quantum dots in a quantum well. Phys. Lett. A 377, 2177–2181 (2013).

70. Čtyroký, J. Photonic bandgap structures in planar waveguides. J. Opt. Soc. Am. A 18, 435–441 (2001).

71. Kawakami, S. Analytically solvable model of photonic crystal structures and novel phenomena. J. Lightwave Technol. 20, 1644 (2002).

72. Watts, M. R., Johnson, S. G., Haus, H. A. & Joannopoulos, J. D. Electromagnetic cavity with arbitrary Q and small modal volume without a complete photonic bandgap. Opt. Lett. 27, 1785–1787 (2002).

73. Apalkov, V. M. & Raikh, M. E. Strongly localized mode at the intersection of the phase slips in a photonic crystal without band gap. Phys. Rev. Lett. 90, 253901 (2003).

74. Rivera, N. et al. Controlling Directionality and Dimensionality of Radiation by Perturbing Separable Bound States in the Continuum. arXiv:1507.00923 (2016).

75. Fan, S., Suh, W. & Joannopoulos, J. D. Temporal coupled-mode theory for the Fano resonance in optical resonators. J. Opt. Soc. Am. A 20, 569–572 (2003).

76. Haus, H. A. Waves and Fields in Optoelectronics. (Prentice-Hall, 1984).

77. Fan, S. et al. Theoretical analysis of channel drop tunneling processes. Phys. Rev. B 59, 15882–15892 (1999).

78. Manolatou, C. et al. Coupling of modes analysis of resonant channel add-drop filters. IEEE J. Quantum Electron. 35, 1322–1331 (1999).

79. Wang, Z. & Fan, S. Compact all-pass filters in photonic crystals as the building block for high-capacity optical delay lines. Phys. Rev. E 68, 066616 (2003).

80. McIver, M. An example of non-uniqueness in the two-dimensional linear water wave problem. J. Fluid Mech. 315, 257–266 (1996). This paper proposes a Fabry-Pérot BIC for water waves by engineering the shapes of two obstacles.

81. Linton, C. M. & Kuznetsov, N. G. Non-uniqueness in two-dimensional water wave problems: numerical evidence and geometrical restrictions. Proc. R. Soc. A 453, 2437–2460 (1997).

82. Evans, D. V. & Porter, R. An example of non-uniqueness in the two-dimensional linear water-wave problem involving a submerged body. Proc. R. Soc. A 454, 3145-3165 (1998).

83. McIver, M. Trapped modes supported by submerged obstacles. Proc. R. Soc. A 456, 1851–1860 (2000).

84. Kuznetsov, N., McIver, P. & Linton, C. M. On uniqueness and trapped modes in the water-wave problem for vertical barriers. Wave Motion 33, 283–307 (2001).

85. Porter, R. Trapping of water waves by pairs of submerged cylinders. Proc. R. Soc. A 458, 607–624 (2002).

86. Linton, C. M. & McIver, P. Embedded trapped modes in water waves and acoustics. Wave Motion 45, 16 (2007). This paper reviews theoretical studies of BICs in acoustic and water waves.

87. Shahbazyan, T. V. & Raikh, M. E. Two-channel resonant tunneling. Phys. Rev. B 49, 17123–17129 (1994).

88. Kim, C. S. & Satanin, A. M. Dynamic confinement of electrons in time-dependent quantum structures. Phys. Rev. B 58, 15389–15392 (1998).

89. Rotter, I. & Sadreev, A. F. Zeros in single-channel transmission through double quantum dots. Phys. Rev. E 71, 046204 (2005).

90. Sadreev, A. F., Bulgakov, E. N. & Rotter, I. Trapping of an electron in the transmission through two quantum dots coupled by a wire. JETP Lett. 82, 498–503 (2005).

91. Ordonez, G., Na, K. & Kim, S. Bound states in the continuum in quantum-dot pairs. Phys. Rev. A 73, 022113 (2006).

92. Tanaka, S., Garmon, S., Ordonez, G. & Petrosky, T. Electron trapping in a one-dimensional semiconductor quantum wire with multiple impurities. Phys. Rev. B 76, 153308 (2007).

93. Cattapan, G. & Lotti, P. Bound states in the continuum in two-dimensional serial structures. Eur. Phys. J. B 66, 517-523 (2008).

94. Díaz-Tendero, S., Borisov, A. G. & Gauyacq, J.-P. Extraordinary Electron Propagation Length in a Metallic Double Chain Supported on a Metal Surface. Phys. Rev. Lett. 102, 166807 (2009).

95. Sadreev, A. F., Maksimov, D. N. & Pilipchuk, A. S. Gate controlled resonant widths in double-bend waveguides: bound states in the continuum. J. Phys. Condens. Matter 27, 295303 (2015).

96. Suh, W., Yanik, M. F., Solgaard, O. & Fan, S. Displacement-sensitive photonic crystal structures based on guided resonance in photonic crystal slabs. Appl. Phys. Lett. 82, 1999–2001 (2003).

97. Suh, W., Solgaard, O. & Fan, S. Displacement sensing using evanescent tunneling between guided resonances in photonic crystal slabs. J. Appl. Phys. 98, 033102 (2005).

98. Liu, V., Povinelli, M. & Fan, S. Resonance-enhanced optical forces between coupled photonic crystal slabs. Opt. Express 17, 21897–21909 (2009).

99. Marinica, D. C., Borisov, A. G. & Shabanov, S. V. Bound states in the continuum in photonics. Phys. Rev. Lett. 100, 183902 (2008).

100. Ndangali, R. F. & Shabanov, S. V. Electromagnetic bound states in the radiation continuum for periodic double arrays of subwavelength dielectric cylinders. J. Math. Phys. 51, 102901 (2010).

101. Bulgakov, E. N. & Sadreev, A. F. Bound states in the continuum in photonic waveguides inspired by defects. Phys. Rev. B 78, 075105 (2008).

102. Longhi, S. Optical analogue of coherent population trapping via a continuum in optical waveguide arrays. J. Mod. Opt. 56, 729–737 (2009).

103. Hein, S., Koch, W. & Nannen, L. Trapped modes and Fano resonances in two-dimensional acoustical duct-cavity systems. J. Fluid Mech. 692, 257–287 (2012).

104. Sato, Y. et al. Strong coupling between distant photonic nanocavities and its dynamic control. Nature Photon. 6, 56–61 (2012).

105. Zheng, H. & Baranger, H. U. Persistent Quantum Beats and Long-Distance Entanglement from Waveguide-Mediated Interactions. Phys. Rev. Lett. 110, 113601 (2013).

106. van Loo, A. F. et al. Photon-Mediated Interactions Between Distant Artificial Atoms. Science 342, 1494–1496 (2013).

107. Peleg, O., Plotnik, Y., Moiseyev, N., Cohen, O. & Segev, M. Self-trapped leaky waves and their interactions. Phys. Rev. A 80, 041801 (2009).

108. Hsu, C. W. et al. Bloch surface eigenstates within the radiation continuum. Light: Science & Applications 2, e84 (2013).

109. Longhi, S. Bound states in the continuum in a single-level Fano-Anderson model. Eur. Phys. J. B 57, 45–51 (2007).

110. Weimann, S. et al. Compact Surface Fano States Embedded in the Continuum of Waveguide Arrays. Phys. Rev. Lett. 111, 240403 (2013). This work realizes a Fabry-Pérot BIC in a coupled-waveguide array.

111. McIver, P. & McIver, M. Trapped modes in an axisymmetric water-wave problem. Q. J. Mech. Appl. Math. 50, 165–178 (1997).

112. Kuznetsov, N. & McIver, P. On uniqueness and trapped modes in the water-wave problem for a surface-piercing axisymmetric body. Q. J. Mech. Appl. Math. 50, 565-580 (1997).

113. Suh, W., Wang, Z. & Fan, S. Temporal coupled-mode theory and the presence of non-orthogonal modes in lossless multimode cavities. IEEE J. Quantum Electron. 40, 1511–1518 (2004).

114. Devdariani, A. Z., Ostrovskii, V. N. & Sebyakin, Y. N. Crossing of quasistationary levels.
Sov. Phys. JETP 44, 477 (1976).

115. Friedrich, H. & Wintgen, D. Interfering resonances and bound states in the continuum. Phys. Rev. A 32, 3231–3242 (1985). This paper proposes that a BIC can arise from two interfering resonances.

116. Remacle, F., Munster, M., Pavlov-Verevkin, V. B. & Desouter-Lecomte, M. Trapping in competitive decay of degenerate states. Phys. Lett. A 145, 265–268 (1990).

117. Berkovits, R., Oppen, F. von & Kantelhardt, J. W. Discrete charging of a quantum dot strongly coupled to external leads. Euro. Phys. Lett. 68, 699 (2004).

118. Fonda, L. & Newton, R. G. Theory of resonance reactions. Ann. Phys. 10, 490-515 (1960).

119. Friedrich, H. & Wintgen, D. Physical realization of bound states in the continuum. Phys. Rev. A 31, 3964–3966 (1985).

120. Neukammer, J. et al. Autoionization Inhibited by Internal Interferences. Phys. Rev. Lett. 55, 1979–1982 (1985).

121. Volya, A. & Zelevinsky, V. Non-Hermitian effective Hamiltonian and continuum shell model. Phys. Rev. C 67, 054322 (2003).

122. Deb, B. & Agarwal, G. S. Creation and manipulation of bound states in the continuum with lasers: Applications to cold atoms and molecules. Phys. Rev. A 90, 063417 (2014).

123. Sablikov, V. A. & Sukhanov, A. A. Helical bound states in the continuum of the edge states in two dimensional topological insulators. Phys. Lett. A 379, 1775-1779 (2015).

124. Texier, C. Scattering theory on graphs: II. The Friedel sum rule. J. Phys. A 35, 3389 (2002).

125. Sadreev, A. F., Bulgakov, E. N. & Rotter, I. Bound states in the continuum in open quantum billiards with a variable shape. Phys. Rev. B 73, 235342 (2006).

126. Sadreev, A. F. & Babushkina, T. V. Two-electron bound states in a continuum in quantum dots. JETP Lett. 88, 312–317 (2008).

127. Boretz, Y., Ordonez, G., Tanaka, S. & Petrosky, T. Optically tunable bound states in the continuum. Phys. Rev. A 90, 023853 (2014).

128. Lyapina, A. A., Maksimov, D. N., Pilipchuk, A. S. & Sadreev, A. F. Bound states in the continuum in open acoustic resonators. J. Fluid Mech. 780, 370–387 (2015).

129. Lepetit, T., Akmansoy, E., Ganne, J.-P. & Lourtioz, J.-M. Resonance continuum coupling in high-permittivity dielectric metamaterials. Phys. Rev. B 82, 195307 (2010).

130. Lepetit, T. & Kanté, B. Controlling multipolar radiation with symmetries for electromagnetic bound states in the continuum. Phys. Rev. B 90, 241103 (2014).

131. Gentry, C. M. & Popović, M. A. Dark state lasers. Opt. Lett. 39, 4136-4139 (2014).

132. Hsu, C. W. et al. Observation of trapped light within the radiation continuum. Nature 499, 188–191 (2013). This work realizes a single-resonance parametric BIC in a photonic crystal slab.

133. Gansch, R. et al. Measurement of bound states in the continuum by a detector embedded in a photonic crystal. Light: Science & Applications accepted article; doi:10.1038/lsa.2016.147 (2016).

134. Evans, D. V. & Porter, R. On the Existence of Embedded Surface Waves Along Arrays of Parallel Plates. Q. J. Mech. Appl. Math. 55, 481-494 (2002).

135. Porter, R. & Evans, D. V. Embedded Rayleigh-Bloch surface waves along periodic rectangular arrays. Wave Motion 43, 29–50 (2005).

136. Bulgakov, E. N. & Sadreev, A. F. Bloch bound states in the radiation continuum in a periodic array of dielectric rods. Phys. Rev. A 90, 053801 (2014).

137. Bulgakov, E. N. & Sadreev, A. F. Light trapping above the light cone in a one-dimensional array of dielectric spheres. Phys. Rev. A 92, 023816 (2015).

138. Longhi, S. & Della Valle, G. Floquet bound states in the continuum. Sci. Rep. 3, 2219 (2013).

139. Yang, Y., Peng, C., Liang, Y., Li, Z. & Noda, S. Analytical Perspective for Bound States in the Continuum in Photonic Crystal Slabs. Phys. Rev. Lett. 113, 037401 (2014).

140. Gao, X. et al. Formation Mechanism of Guided Resonances and Bound States in the Continuum in Photonic Crystal Slabs. arXiv: 1603.02815 (2016).

141. Zhen, B., Hsu, C. W., Lu, L., Stone, A. D. & Soljačić, M. Topological Nature of Optical Bound States in the Continuum. Phys. Rev. Lett. 113, 257401 (2014). This paper explains the topological nature of BICs in photonic crystal slabs.

142. Yang, B.-J., Bahramy, M. S. & Nagaosa, N. Topological protection of bound states against the hybridization. Nat. Commun. 4, 1524 (2013).

143. Linton, C. M., McIver, M., McIver, P., Ratcliffe, K. & Zhang, J. Trapped modes for off-centre structures in guides. Wave Motion 36, 67–85 (2002).

144. Evans, D. & Porter, R. Trapped modes embedded in the continuous spectrum. Q. J. Mech. Appl. Math. 51, 263-274 (1998).

145. McIver, M., Linton, C. M., McIver, P., Zhang, J. & Porter, R. Embedded Trapped Modes for Obstacles in Two-Dimensional Waveguides. Q. J. Mech. Appl. Math. 54, 273–293 (2001).

146. McIver, M., Linton, C. M. & Zhang, J. The Branch Structure of Embedded Trapped Modes in Two-Dimensional Waveguides. Q. J. Mech. Appl. Math. 55, 313-326 (2002).

147. Koch, W. Acoustic Resonances in Rectangular Open Cavities. AIAA Journal 43, 2342-2349 (2005).

148. Duan, Y., Koch, W., Linton, C. M. & McIver, M. Complex resonances and trapped modes in ducted domains. J. Fluid Mech. 571, 119–147 (2007).

149. Kim, C. S., Satanin, A. M., Joe, Y. S. & Cosby, R. M. Resonant tunneling in a quantum waveguide: Effect of a finite-size attractive impurity. Phys. Rev. B 60, 10962–10970 (1999).

150. Linton, C. M. & Ratcliffe, K. Bound states in coupled guides. I. Two dimensions. J. Math. Phys. 45, 1359–1379 (2004).

151. Cattapan, G. & Lotti, P. Fano resonances in stubbed quantum waveguides with impurities. Eur. Phys. J. B 60, 51–60 (2007).

152. Olendski, O. & Mikhailovska, L. Bound-state evolution in curved waveguides and quantum wires. Phys. Rev. B 66, 035331 (2002).

153. Olendski, O. & Mikhailovska, L. Fano resonances of a curved waveguide with an embedded quantum dot. Phys. Rev. B 67, 035310 (2003).

154. Chen, Y. et al. Mechanical bound state in the continuum for optomechanical microresonators. arXiv: 1601.04061 (2016).

155. Zou, C.-L. et al. Guiding light through optical bound states in the continuum for ultrahigh-Q microresonators. Laser Photon. Rev. 9, 114–119 (2015).

156. Penunuri, D. & Lakin, K. M. Leaky Surface Wave Propagation on Si, GaAs, GaP, $Al_{2}O_{3}$ and Quartz. in 1975 IEEE Ultrasonics Symposium 478–483 (1975).

157. Stegeman, G. I. Normal-mode surface waves in the pseudobranch on the (001) plane of gallium arsenide. J. Appl. Phys. 47, 1712–1713 (1976).

158. Aleksandrov, V. V. et al. New data concerning surface Mandelstamm-Brillouin light scattering from the basal plane of germanium crystal. Phys. Lett. A 162, 418–422 (1992).

159. Aleksandrov, V. V., Velichkina, T. S., Potapova, J. B. & Yakovlev, I. A. Mandelstamm-Brillouin studies of peculiarities of the phonon frequency distribution at cubic crystal (001) surfaces. Phys. Lett. A 171, 103–106 (1992).

160. Taylor, D. B. Surface Waves in Anisotropic Media: The Secular Equation and its Numerical Solution. Proc. R. Soc. A 376, 265–300 (1981).

161. Gundersen, S. A., Wang, L. & Lothe, J. Secluded supersonic elastic surface waves. Wave Motion 14, 129–143 (1991).

162. Barnett, D. M., Chadwick, P. & Lothe, J. The Behaviour of Elastic Surface Waves Polarized in a Plane of Material Symmetry. I. Addendum. Proc. R. Soc. A 433, 699–710 (1991).

163. Maznev, A. A. & Every, A. G. Secluded supersonic surface waves in germanium. Phys. Lett. A 197, 423–427 (1995).

164. Darinskii, A. N., Alshits, V. I., Lothe, J., Lyubimov, V. N. & Shuvalov, A. L. An existence criterion for the branch of two-component surface waves in anisotropic elastic media. Wave Motion 28, 241–257 (1998).

165. Xu, Y. & Aizawa, T. Pseudo surface wave on the (1012) plane of sapphire. J. Appl. Phys. 86, 6507–6511 (1999).

166. Maznev, A. A., Lomonosov, A. M., Hess, P. & Kolomenskii, A. Anisotropic effects in surface acoustic wave propagation from a point source in a crystal. Eur. Phys. J. B 35, 429–439 (2003).

167. Trzupek, D. & Zieliński, P. Isolated True Surface Wave in a Radiative Band on a Surface of a Stressed Auxetic. Phys. Rev. Lett. 103, 075504 (2009).

168. Every, A. G. Supersonic surface acoustic waves on the 001 and 110 surfaces of cubic crystals. J. Acoust. Soc. Am. 138, 2937–2944 (2015).

169. Every, A. G. Guided elastic waves at a periodic array of thin coplanar cavities in a solid.
Phys. Rev. B 78, 174104 (2008).

170. Maznev, A. A. & Every, A. G. Surface acoustic waves in a periodically patterned layered structure. J. Appl. Phys. 106, 113531 (2009).

171. Every, A. G. & Maznev, A. A. Elastic waves at periodically-structured surfaces and interfaces of solids. AIP Advances 4, 124401 (2014).

172. Yamanouchi, K. & Shibayama, K. Propagation and Amplification of Rayleigh Waves and Piezoelectric Leaky Surface Waves in $LiNbO_{3}$ . J. Appl. Phys. 43, 856–862 (1972). This

work predicts and measures an acoustic single-resonance parametric BIC on the surface of a piezoelectric solid.

173. Lewis, M. F. Acoustic wave devices employing surface skimming bulk waves (1979). US Patent 4,159,435.

174. Ueda, M. et al. Surface acoustic wave device using a leaky surface acoustic wave with an optimized cut angle of a piezoelectric substrate (2000). US Patent 6,037,847.

175. Kawachi, O. et al. Optimal cut for leaky SAW on $LiTaO_{3}$ for high performance resonators and filters. IEEE Trans. Ultrason., Ferroelect., Freq. Control 48, 1442–1448 (2001).

176. Naumenko, N. & Abbot, B. Optimized cut of $LiTaO_{3}$ for resonator filters with improved performance. in Proceedings of 2002 IEEE Ultrasonics Symposium 1, 385–390 (IEEE, 2002).

177. Simon, B. On positive eigenvalues of one-body Schrödinger operators. Communications on Pure and Applied Mathematics 22, 531–538 (1969).

178. Stillinger, F. H. & Herrick, D. R. Bound states in the continuum. Phys. Rev. A 11, 446–454 (1975).

179. Jain, A. & Shastry, C. Bound states in the continuum for separable nonlocal potentials. Physical Review A 12, 2237 (1975).

180. Molina, M. I., Miroshnichenko, A. E. & Kivshar, Y. S. Surface bound states in the continuum. Phys. Rev. Lett. 108, 070401 (2012).

181. Gallo, N. & Molina, M. Bulk and surface bound states in the continuum. Journal of Physics A: Mathematical and Theoretical 48, 045302 (2015).

182. Simon, B. Some Schrödinger operators with dense point spectrum. Proc. Amer. Math. Soc. 125, 203–208 (1997).

183. Moses, H. E. & Tuan, S. Potentials with zero scattering phase. Il Nuovo Cimento 13, 197–206 (1959).

184. Gazdy, B. On the bound states in the continuum. Physics Letters A 61, 89-90 (1977).

185. Meyer-Vernet, N. Strange bound states in the Schrödinger wave equation: When usual tunneling does not occur. Am. J. Phys. 50, 354–356 (1982).

186. Pivovarchik, V. N., Suzko, A. A. & Zakhariev, B. N. New Exactly Solved Models with Bound States above the Scattering Threshold. Phys. Scripta 34, 101 (1986).

187. Naboko, S. N. Dense point spectra of Schrödinger and Dirac operators. Theor. Math. Phys. 68, 646–653 (1986).

188. Darboux, M. G. Sur une proposition relative aux équations liéaires. C. R. Acad. Sci. 94, 1456 (1882).

189. Svirsky, R. An application of double commutation to the addition of bound states to the spectrum of a Schrodinger operator. Inverse Probl. 8, 483 (1992).

190. Pappademos, J., Sukhatme, U. & Pagnamenta, A. Bound states in the continuum from supersymmetric quantum mechanics. Physical Review A 48, 3525 (1993).

191. Stahlhofen, A. A. Completely transparent potentials for the Schrödinger equation. Phys. Rev. A 51, 934–943 (1995).

192. Weber, T. A. & Pursey, D. L. Continuum bound states. Phys. Rev. A 50, 4478-4487 (1994).

193. Kočinac, S. L. S. & Milanović, V. Bound States in Continuum Generated by Point Interaction and Supersymmetric Quantum Mechanics. Modern Physics Letters B 26, 1250177 (2012).

194. Ranjani, S. S., Kapoor, A. & Panigrahi, P. Normalizable states through deformation of Lamé and the associated Lamé potentials. Journal of Physics A: Mathematical and Theoretical 41, 285302 (2008).

195. Prodanović, N., Milanović, V. & Radovanović, J. Photonic crystals with bound states in continuum and their realization by an advanced digital grading method. J. Phys. A 42, 415304 (2009).

196. Petrović, J. S., Milanović, V. & Ikonić, Z. Bound states in continuum of complex potentials generated by supersymmetric quantum mechanics. Physics Letters A 300, 595–602 (2002).

197. Andrianov, A. A., Sokolov, A. V. & others. Resolutions of identity for some non-Hermitian Hamiltonians. I. Exceptional point in continuous spectrum. SIGMA 7, 19 (2011).

198. Sokolov, A. V. & others. Resolutions of identity for some non-Hermitian Hamiltonians. II. Proofs. SIGMA 7, 16 (2011).

199. Longhi, S. & Della Valle, G. Optical lattices with exceptional points in the continuum. Physical Review A 89, 052132 (2014).

200. Longhi, S. Bound states in the continuum in PT-symmetric optical lattices. Optics Lett. 39, 1697–1700 (2014). This paper classifies and compares two types of BICs in parity-time symmetric systems.

201. Fernández-García, N., Hernández, E., Jáuregui, A. & Mondragón, A. Bound states at exceptional points in the continuum. in Journal of Physics: Conference Series 512, 012023 (IOP Publishing, 2014).

202. Garmon, S., Gianfreda, M. & Hatano, N. Bound states, scattering states, and resonant states in PT-symmetric open quantum systems. Phys. Rev. A 92, 022125 (2015).

203. Demić, A., Milanović, V. & Radovanović, J. Bound states in the continuum generated by supersymmetric quantum mechanics and phase rigidity of the corresponding wavefunctions. Physics Letters A (2015).

204. Correa, F., Jakubský, V. & Plyushchay, M. S. PT-symmetric invisible defects and confluent Darboux-Crum transformations. Physical Review A 92, 023839 (2015).

205. Pursey, D. & Weber, T. Scattering from a shifted von Neumann-Wigner potential. Physical Review A 52, 3932 (1995).

206. Weber, T. A. & Pursey, D. L. Scattering from a truncated von Neumann-Wigner potential. Phys. Rev. A 57, 3534-3545 (1998).

207. Longhi, S. Non-Hermitian tight-binding network engineering. Phys. Rev. A 93, 022102 (2016).

208. Corrielli, G., Della Valle, G., Crespi, A., Osellame, R. & Longhi, S. Observation of Surface States with Algebraic Localization. Phys. Rev. Lett. 111, 220403 (2013). This work demonstrates a BIC in a tight-binding lattice with engineered hopping rates.

209. McIver, M. & Porter, R. Trapping of waves by a submerged elliptical torus. J. Fluid Mech. 456, 277–293 (2002).

210. McIver, P. & Newman, J. N. Trapping structures in the three-dimensional water-wave problem. J. Fluid Mech. 484, 283–301 (2003).

211. McIver, P. & McIver, M. Trapped modes in the water-wave problem for a freely floating structure. J. Fluid Mech. 558, 53–67 (2006).

212. McIver, P. & McIver, M. Motion trapping structures in the three-dimensional water-wave problem. J. Eng. Math. 58, 67–75 (2007).

213. Porter, R. & Evans, D. V. Water-wave trapping by floating circular cylinders. J. Fluid Mech. 633, 311–325 (2009).

214. Meier, M. et al. Laser action from two-dimensional distributed feedback in photonic crystals. Appl. Phys. Lett. 74, 7–9 (1999).

215. Imada, M. et al. Coherent two-dimensional lasing action in surface-emitting laser with triangular-lattice photonic crystal structure. Applied physics letters 75, 316–318 (1999).

216. Noda, S., Yokoyama, M., Imada, M., Chutinan, A. & Mochizuki, M. Polarization Mode Control of Two-Dimensional Photonic Crystal Laser by Unit Cell Structure Design. Science 293, 1123–1125 (2001).

217. Miyai, E. et al. Photonics: lasers producing tailored beams. Nature 441, 946–946 (2006).

218. Matsubara, H. et al. GaN Photonic-Crystal Surface-Emitting Laser at Blue-Violet Wavelengths. Science 319, 445–447 (2008).

219. Hirose, K. et al. Watt-class high-power, high-beam-quality photonic-crystal lasers. Nature Photon. 8, 406–411 (2014). This paper demonstrated continuous-wave lasing through a quasi-BIC with a high output-power and a low threshold at room temperature.

220. Yanik, A. A. et al. Seeing protein monolayers with naked eye through plasmonic Fano resonances. Proc. Natl. Acad. Sci. U.S.A. 108, 11784–11789 (2011).

221. Zhen, B. et al. Enabling enhanced emission and low-threshold lasing of organic molecules using special Fano resonances of macroscopic photonic crystals. Proc. Natl. Acad. Sci. U.S.A. 110, 13711–13716 (2013).

222. Foley, J. M., Young, S. M. & Phillips, J. D. Symmetry-protected mode coupling near normal incidence for narrow-band transmission filtering in a dielectric grating. Physical Review B 89, 165111 (2014).

223. Iwahashi, S. et al. Higher-order vector beams produced by photonic-crystal lasers. Opt. Express 19, 11963–11968 (2011).

224. Kitamura, K., Sakai, K., Takayama, N., Nishimoto, M. & Noda, S. Focusing properties of vector vortex beams emitted by photonic-crystal lasers. Opt. Lett. 37, 2421–2423 (2012).

225. Zhan, Q. Cylindrical vector beams: from mathematical concepts to applications. Adv. Opt. Photon. 1, 1–57 (2009).

226. Fan, X. & White, I. M. Optofluidic microsystems for chemical and biological analysis. Nature photonics 5, 591–597 (2011).

227. Morgan, D. Surface acoustic wave filters: With applications to electronic communications and signal processing. (Academic Press, 2007). Chapter 11.

228. Benabid, F., Knight, J. C., Antonopoulos, G. & Russell, P. S. J. Stimulated Raman Scattering in Hydrogen-Filled Hollow-Core Photonic Crystal Fiber. Science 298, 399–402 (2002).

229. Couny, F., Benabid, F., Roberts, P. J., Light, P. S. & Raymer, M. G. Generation and Photonic Guidance of Multi-Octave Optical-Frequency Combs. Science 318, 1118–1121 (2007).

230. Benabid, F., Couny, F., Knight, J., Birks, T. & Russell, P. S. J. Compact, stable and efficient all-fibre gas cells using hollow-core photonic crystal fibres. Nature 434, 488–491 (2005).

231. Dudley, J. M. & Taylor, J. R. Ten years of nonlinear optics in photonic crystal fibre. Nature Photonics 3, 85–90 (2009).

232. Benabid, F., Knight, J. & Russell, P. Particle levitation and guidance in hollow-core photonic crystal fiber. Optics express 10, 1195–1203 (2002).

233. Benoît, A. et al. Over-five octaves wide Raman combs in high-power picosecond-laser pumped $H_{2}$ -filled inhibited coupling Kagome fiber. Optics Express 23, 14002–14009 (2015).

234. Mur-Petit, J. & Molina, R. A. Chiral bound states in the continuum. Phys. Rev. B 90, 035434 (2014).

235. Zhang, J. M., Braak, D. & Kollar, M. Bound states in the continuum realized in the one-dimensional two-particle Hubbard model with an impurity. Phys. Rev. Lett. 109, 116405 (2012).

236. Zhang, J. M., Braak, D. & Kollar, M. Bound states in the one-dimensional two-particle Hubbard model with an impurity. Phys. Rev. A 87, 023613 (2013).

237. Longhi, S. & Della Valle, G. Tamm-Hubbard surface states in the continuum. J. Phys. Condens. Matter 25, 235601 (2013).

238. Della Valle, G. & Longhi, S. Floquet-Hubbard bound states in the continuum. Phys. Rev. B 89, 115118 (2014).

239. Vicencio, R. A. et al. Observation of Localized States in Lieb Photonic Lattices. Phys. Rev. Lett. 114, 245503 (2015).

240. Mukherjee, S. et al. Observation of a Localized Flat-Band State in a Photonic Lieb Lattice. Phys. Rev. Lett. 114, 245504 (2015).

241. Koirala, M. et al. Critical states embedded in the continuum. New Journal of Physics 17, 013003 (2015).

242. Joglekar, Y. N., Scott, D. D. & Saxena, A. PT-symmetry breaking with divergent potentials: Lattice and continuum cases. Physical Review A 90, 032108 (2014).

243. Molina, M. I. & Kivshar, Y. S. Embedded States in the Continuum for PT-Symmetric Systems. Studies in Applied Mathematics 133, 337-350 (2014).

244. Longhi, S. Invisible surface defects in a tight-binding lattice. Eur. Phys. J. B 87, 189 (2014).

245. Regensburger, A. et al. Observation of Defect States in PT-Symmetric Optical Lattices. Phys. Rev. Lett. 110, 223902 (2013). This paper demonstrates a BIC in a pair of parity-time symmetric coupled fiber loops.

246. Capasso, F. et al. Observation of an electronic bound state above a potential well. Nature 358, 565–567 (1992).

247. Albo, A., Fekete, D. & Bahir, G. Electronic bound states in the continuum above (Ga, In)(As, N)/(Al, Ga) As quantum wells. Phys. Rev. B 85, 115307 (2012).

248. Bulgakov, E. N. & Sadreev, A. F. Robust bound state in the continuum in a nonlinear microcavity embedded in a photonic crystal waveguide. Opt. Lett. 39, 5212-5215 (2014).

249. Bulgakov, E., Pichugin, K. & Sadreev, A. Channel dropping via bound states in the continuum in a system of two nonlinear cavities between two linear waveguides. J. Phys. Condens. Matter 25, 395304 (2013).

250. Bulgakov, E. N., Pichugin, K. N. & Sadreev, A. F. All-optical light storage in bound states in the continuum and release by demand. Opt. Express 23, 22520–22531 (2015).

251. Lannebere, S. & Silveirinha, M. G. Optical meta-atom for localization of light with quantized energy. Nat. Commun. 6, 8766 (2015).

252. Pichugin, K. N. & Sadreev, A. F. Frequency comb generation by symmetry-protected bound state in the continuum. J. Opt. Soc. Am. B 32, 1630–1636 (2015).

253. Crespi, A. et al. Particle Statistics Affects Quantum Decay and Fano Interference. Phys. Rev. Lett. 114, 090201 (2015).

254. Zhou, H. et. al. General theorem of disallowed couplings protected by Chern numbers.
Manuscript in preparation.

255. Silveirinha, M. G. Trapping light in open plasmonic nanostructures. Phys. Rev. A 89, 023813 (2014). This paper proposes three-dimensional confinement of light using an $\varepsilon = 0$ material and provides an non-existence theorem when $\varepsilon \neq 0$ (see Box 1).

256. Monticone, F. & Alù, A. Embedded Photonic Eigenvalues in 3D Nanostructures. Phys. Rev. Lett. 112, 213903 (2014).

257. Hrebikova, I., Jelinek, L. & Silveirinha, M. G. Embedded energy state in an open semiconductor heterostructure. Phys. Rev. B 92, 155303 (2015).

258. Mermin, N. D. The topological theory of defects in ordered media. Rev. Mod. Phys. 51, 591 (1979).

259. Hasan, M. Z. & Kane, C. L. Colloquium: topological insulators. Rev. Mod. Phys. 82, 3045 (2010).
