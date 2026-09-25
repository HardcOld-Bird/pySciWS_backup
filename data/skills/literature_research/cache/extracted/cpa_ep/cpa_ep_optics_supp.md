# Supplementary Materials for

# Coherent perfect absorption at an exceptional point

Changqing Wang et al.

Corresponding author: Lan Yang, yang@seas.wustl.edu

Science 373, 1261 (2021)
DOI: 10.1126/science.abj1028

The PDF file includes:

Materials and Methods

Supplementary Text

Figs. S1 to S7

References

## Materials and Methods

Balancing optical power in the two input channels

The CPA operation is sensitive to both the amplitudes and phases of the input signals (31–40). In the case of a non-generic CPA EP, the eigenvector of the S-matrix is given by

$$
v = \binom{\pm i}{1},\tag{S1}
$$

where we take the plus sign if $\gamma_{1} < \gamma_{c1}$ , and the minus sign otherwise (see supplementary text S8). As a result, the optical fields injected from port 1 and port 3 should have equal amplitudes and a phase difference of $\pm\pi/2$ .

In our experimental setup, since silica fiber tapers are used as the input and output channels, the optical loss is not negligible along the guided wave propagation. The losses in the two input channels could be unequal due to the unbalanced optical path lengths or different loss in the tapered fibers. Thus, the optical power at the two taper-cavity coupling points could be different. To overcome the loss imbalance and achieve equal input amplitudes, we utilize thermo-optic effects as a reference of the input power at the taper-cavity coupling points.

Based on the setup shown in Fig. S1, we remove $\mu R_{2}$ and use $\mu R_{1}$ only to calibrate the loss difference in the two optical paths. We first tune the variable optical attenuator in the first optical path (OA1) to adjust the optical input power to the level of $100\mu W$ so that the cavity enters the nonlinear regime with strong thermal effects. We critically couple $\mu R_{1}$ to the first fiber taper, and a thermal triangle lineshape appears in the transmission spectrum when the laser is scanned from shorter to longer wavelengths due to the thermo-optic effect (41–45). Under the fixed cavity-taper coupling condition, the width of the triangle has a deterministic relation with the input power. Then we decouple $\mu R_{1}$ from the first taper and critically couple $\mu R_{1}$ to the second taper, where we can also observe a thermal triangle in the transmission spectrum. We then tune the variable optical attenuator in the second optical path (OA2) so that the triangle has the same width as measured in the first one. After that, we use a photodetector to measure the output power coming out from OA1 and OA2, respectively. Assuming linear optical loss in the optical paths, the ratio between the two output powers will remain invariant with arbitrary input power from the laser. During the CPA EP measurement, we maintain the ratio between the optical power coming out of the two attenuators and reduce the laser power dramatically to the $\mu W$ level so that the cavity enters the linear regime with a negligible influence of thermo-optic effects. Thereby, the balance of the input field amplitudes is achieved.

## Measurement of scattered output signals versus the relative phase

In the measurement of CPA EP, the balance between the optical path lengths is critical for obtaining a clear spectrum of the output signal. To see this, we investigate the case where the two optical path lengths are different by $\Delta l$ , and the laser frequency is scanned at the rate of v [Hz/s]. The frequency difference between the two beams arriving at the two taper-cavity coupling points is given by

$$
\Delta f = v \frac {n \Delta l}{c},\tag{S2}
$$

where n is the average refractive index of the media in the optical paths. The different frequencies for the light in the cavities will lead to a beat note in the output signals, with a period of $\frac{2}{\Delta f}$ . Consequently, the relative phase between the two input signals will also vary as $\Delta\phi = 2\pi\Delta ft + \phi_{0}$ , which leads to an oscillation in the spectrum of the scattered output signals. In the measurement for Fig. 3D in the main text, we need to keep $\Delta l = 0$ by tuning the variable optical delay line (VDL) (Fig. S1) so that the relative optical phase between the two optical paths is fixed as the laser frequency is scanned.

However, the oscillation in the spectrum induced by unbalanced optical paths could be useful for investigating the system's response to the relative phase, as shown in Fig. 4 in the main text. If $\Delta l$ is sufficiently large, the oscillation period is much smaller than the time it takes to scan a linewidth of the cavity mode ( $\Delta v$ ), i.e.,

$$
\frac {2}{\Delta f} \ll \frac {\Delta v}{v},\tag{S3}
$$

which yields

$$
\Delta l \gg \frac {2 c}{n \Delta \nu}.\tag{S4}
$$

Then within a small range of frequency scanning, the relative phase could be modulated over multiple periods. Thus, the detuning is approximately constant within one period of relative phase oscillation. The response of the scattered signal to the relative phase can then be measured at any frequency within the scanning range. In our experiment, $v \approx 2400GHz/s$ , $n \approx 1.45$ , $\Delta v > 40MHz$ . Therefore the condition works well if $\Delta l \gg 10m$ , which can be fulfilled by inserting long fibers into one optical path as delay lines. Meanwhile, to avoid the effect of the frequency difference between the two input fields, we need to make sure that $\Delta f \ll \Delta v$ , i.e., $\Delta l \ll \frac{c\Delta v}{nv} = 3448.3m$ , which is not hard to satisfy in the experiments.

In the measurement for Fig. 4C in the main text, we adopt this method, that is, to intentionally break the balance of the lengths of the two optical channels by inserting fibers as delay lines (Fig. S1), and to measure the output power spectra from port 2 and 4 within a small frequency range near the zero detuning, which shows a response as a function of the relative phase between the two inputs. The result can be considered as the signal at zero detuning against the relative phase, to a good approximation. The data can also be used to derive Fig. 4D in the main text, by extracting the maximum and minimum output power at different frequency detuning.

In principle, an alternative method is to apply a sinusoidal phase modulation via EOM and record the spectra of the output signals at each relative phase, followed by extracting the maximum and minimum at each detuning. However, several factors may cause inaccuracy to this method. First, in order to quantify the exact phase value at each frame, one need to precisely set the voltage amplitude of EOM to be $V_{pi}$ , which is the voltage to generate a $\pi$ phase shift. Second, due to the finite number of frames of data collection and limited measurement speed, the maximum and minimum output power cannot be precisely captured.

## Supplementary Text

Theoretical analysis of various kinds of exceptional points (EPs) in the coupled microcavity system

## S1. Theoretical framework

We build a theoretical framework based on the temporal coupled-mode theory (TCMT) (46) to study the scattering behavior of the coupled microcavity system. The Hamiltonian describing the two directly coupled microcavities with resonant frequencies $\omega_{1,2}$ and intrinsic loss rates $\gamma_{1,2}$ can be written as

$$
H _ {0} = \left( \begin{array}{c c} \omega_ {1} - i \frac {\gamma_ {1}}{2} & \kappa \\ \kappa & \omega_ {2} - i \frac {\gamma_ {2}}{2} \end{array} \right),\tag{S5}
$$

where $\kappa$ is the coupling strength between the two microcavities. The coupling channels to the waveguides can be described by a diagonal coupling matrix

$$
D = d i a g \bigl (\sqrt {\gamma_ {c 1}}, \sqrt {\gamma_ {c 2}} \bigr).\tag{S6}
$$

Based on the TCMT, the scattering (S) matrix is then given by

$$
S = 1 - i D ^ {\dagger} \frac {1}{\omega - \left[ H _ {0} - \frac {i D ^ {\dagger} D}{2} \right]} D,\tag{S7}
$$

where the term $H_{0}-iD^{\dagger}D/2$ gives the definition of the effective Hamiltonian that involves the optical dissipation into the coupling channels

$$
H _ {e f f} = H _ {0} - \frac {i D ^ {\dagger} D}{2} = \left( \begin{array}{c c} \omega_ {1} - i \frac {\gamma_ {1} + \gamma_ {c 1}}{2} & \kappa \\ \kappa & \omega_ {2} - i \frac {\gamma_ {2} + \gamma_ {c 2}}{2} \end{array} \right).\tag{S8}
$$

It is noted that our model here is purely linear. In presence of nonlinear effect, the S-matrix may not preserve symmetry, and it is possible to observe various nonlinear optical phenomena at EPs (47–50). In general, we can derive the poles and zeros of S as well as the conditions of resonant and absorbing EPs based on Eq. (S8). We note that for either type of EPs, multiple solutions can be obtained based on the conditions for the parameters. For simplicity, we consider a special case that the two cavity resonances match with each other, i.e., $\omega_{1} = \omega_{2}$ . Other cases which break this restriction may be of interest for studying EPs in other schemes, for example anti-PT symmetry (51–60).

## S2. Resonant EP

We first calculate the eigenvalues of $H_{eff}$ , which governs the resonances of the system,

$$
\begin{array}{c} \lambda_ {r \pm} = \frac {\omega_ {1} + \omega_ {2}}{2} - i \frac {\gamma_ {1} + \gamma_ {c 1} + \gamma_ {2} + \gamma_ {c 2}}{4} \\ \pm \frac {1}{2} \sqrt {\left(\omega_ {1} + \omega_ {2} - i \frac {\gamma_ {1} + \gamma_ {c 1} + \gamma_ {2} + \gamma_ {c 2}}{2}\right) ^ {2} - 4 \left(\omega_ {1} - i \frac {\gamma_ {1} + \gamma_ {c 1}}{2}\right) \left(\omega_ {2} - i \frac {\gamma_ {2} + \gamma_ {c 2}}{2}\right) + 4 \kappa^ {2}} \\ = \frac {\omega_ {1} + \omega_ {2}}{2} - i \frac {\gamma_ {1} + \gamma_ {c 1} + \gamma_ {2} + \gamma_ {c 2}}{4} \pm \frac {1}{2} \sqrt {\left(\omega_ {1} - \omega_ {2} - i \frac {\gamma_ {1} + \gamma_ {c 1} - \gamma_ {2} - \gamma_ {c 2}}{2}\right) ^ {2} + 4 \kappa^ {2}}, \end{array}\tag{S9}
$$

which are also the poles of S. When $\omega_{1} = \omega_{2} = \omega_{0}$ , the resonances are given by

$$
\lambda_ {r 1, 2} = \omega_ {0} - i \frac {\gamma_ {1} + \gamma_ {c 1} + \gamma_ {2} + \gamma_ {c 2}}{4} \pm \sqrt {\kappa^ {2} - \left(\frac {\gamma_ {1} + \gamma_ {c 1} - \gamma_ {2} - \gamma_ {c 2}}{4}\right) ^ {2}},\tag{S10}
$$

which are equal to the poles of $S(\omega_{p1,2})$ . Then we derive the conditions for the resonant EPs

$$
\kappa = \frac {| \gamma_ {1} + \gamma_ {c 1} - \gamma_ {2} - \gamma_ {c 2} |}{4}.\tag{S11}
$$

## S3. Absorbing EP

Now we turn to the calculation of zeros of S. For simplicity, we define $\Omega_{1,2} = \omega_{1,2} - i \frac{\gamma_{1,2} + \gamma_{c1,2}}{2}$ . It follows that

$$
\frac {1}{\omega - H _ {e f f}} = \left( \begin{array}{c c} \omega - \Omega_ {1} & - \kappa \\ - \kappa & \omega - \Omega_ {2} \end{array} \right) ^ {- 1} = \frac {1}{\det (\omega - H _ {e f f})} \left( \begin{array}{c c} \omega - \Omega_ {2} & \kappa \\ \kappa & \omega - \Omega_ {1} \end{array} \right).\tag{S12}
$$

Thus, we can obtain the S-matrix

$$
\begin{array}{c} S = 1 - i \left( \begin{array}{c c} \sqrt {\gamma_ {c 1}} & 0 \\ 0 & \sqrt {\gamma_ {c 2}} \end{array} \right) \left( \begin{array}{c c} \Delta_ {2} & \kappa \\ \kappa & \Delta_ {1} \end{array} \right) \left( \begin{array}{c c} \sqrt {\gamma_ {c 1}} & 0 \\ 0 & \sqrt {\gamma_ {c 2}} \end{array} \right) \frac {1}{\Delta_ {1} \Delta_ {2} - \kappa^ {2}} \\ = \left( \begin{array}{c c} 1 - i \frac {\gamma_ {c 1} \Delta_ {2}}{\Delta_ {1} \Delta_ {2} - \kappa^ {2}} & - i \frac {\sqrt {\gamma_ {c 1} \gamma_ {c 2}} \kappa}{\Delta_ {1} \Delta_ {2} - \kappa^ {2}} \\ - i \frac {\sqrt {\gamma_ {c 1} \gamma_ {c 2}} \kappa}{\Delta_ {1} \Delta_ {2} - \kappa^ {2}} & 1 - i \frac {\gamma_ {c 2} \Delta_ {1}}{\Delta_ {1} \Delta_ {2} - \kappa^ {2}} \end{array} \right). \end{array}\tag{S13}
$$

where $\Delta_{1,2} = \omega - \Omega_{1,2}$ . The S-matrix is symmetric in our study which indicates the reciprocal wave transport. We can then calculate the eigenvalues of S

$$
\begin{array}{r l r} & & {\sigma_ {1, 2} = 1 - i \frac {\gamma_ {c 1} (\delta_ {2} + i \frac {\gamma_ {2} + \gamma_ {c 2}}{2}) + \gamma_ {c 2} (\delta_ {1} + i \frac {\gamma_ {1} + \gamma_ {c 1}}{2})}{2 (\Delta_ {1} \Delta_ {2} - \kappa^ {2})}} \\ & & {\pm \frac {1}{2 (\Delta_ {1} \Delta_ {2} - \kappa^ {2})} \sqrt {- \left(\gamma_ {c 1} (\delta_ {2} + i \frac {\gamma_ {2} + \gamma_ {c 2}}{2}) - \gamma_ {c 2} (\delta_ {1} + i \frac {\gamma_ {1} + \gamma_ {c 1}}{2})\right) ^ {2} - 4 \gamma_ {c 1} \gamma_ {c 2} \kappa^ {2}}.} \end{array}\tag{S14}
$$

If $\omega_{1}=\omega_{2}=\omega_{0}$ , the eigenvalues of S are given by

$$
\begin{array}{r} \sigma_ {1, 2} = \frac {\left(\left(\delta + i \frac {\gamma_ {1}}{2}\right) \left(\delta + i \frac {\gamma_ {2}}{2}\right) + \left(\frac {\gamma_ {c 1}}{2}\right) \left(\frac {\gamma_ {c 2}}{2}\right) - \kappa^ {2}\right)}{\left(\Delta_ {1} \Delta_ {2} - \kappa^ {2}\right)} \\ \pm \frac {1}{\left(\Delta_ {1} \Delta_ {2} - \kappa^ {2}\right)} \sqrt {- \left(\frac {\gamma_ {c 1}}{2} \left(\delta + i \frac {\gamma_ {2} + \gamma_ {c 2}}{2}\right) - \frac {\gamma_ {c 2}}{2} \left(\delta + i \frac {\gamma_ {1} + \gamma_ {c 1}}{2}\right)\right) ^ {2} - \gamma_ {c 1} \gamma_ {c 2} \kappa^ {2}}, \end{array}\tag{S15}
$$

where $\delta = \omega -\omega_0$

For zeros, we have $\sigma_{1,2} = 0$ , that is

$$
\frac {\left(\left(\delta + i \frac {\gamma_ {1}}{2}\right) \left(\delta + i \frac {\gamma_ {2}}{2}\right) + \left(\frac {\gamma_ {c 1}}{2}\right) \left(\frac {\gamma_ {c 2}}{2}\right) - \kappa^ {2}\right)}{\pm \sqrt {- \left(\frac {\gamma_ {c 1}}{2} \left(\delta + i \frac {\gamma_ {2}}{2}\right) - \frac {\gamma_ {c 2}}{2} \left(\delta + i \frac {\gamma_ {1}}{2}\right)\right) ^ {2} - \gamma_ {c 1} \gamma_ {c 2} \kappa^ {2}} = 0,}\tag{S16}
$$

which yields two solutions of zeros

$$
\begin{array}{r} \omega_ {z 1, z 2} = \frac {\Omega_ {1} + \Omega_ {2} + i \gamma_ {c 1} + i \gamma_ {c 2}}{2} \pm \frac {1}{2} \sqrt {(\Omega_ {1} + i \gamma_ {c 1} - \Omega_ {2} - i \gamma_ {c 2}) ^ {2} + 4 \kappa^ {2}} \\ = \frac {\omega_ {1} + \omega_ {2}}{2} + i \frac {\gamma_ {c 1} + \gamma_ {c 2} - \gamma_ {1} - \gamma_ {2}}{4} \pm \frac {1}{2} \sqrt {\left(\omega_ {1} + i \frac {\gamma_ {c 1} - \gamma_ {1}}{2} - \omega_ {2} - i \frac {\gamma_ {c 2} - \gamma_ {2}}{2}\right) ^ {2} + 4 \kappa^ {2}}. \end{array}\tag{S17}
$$

The EP of zeros requires that

$$
(\Omega_ {1} + i \gamma_ {c 1} - \Omega_ {2} - i \gamma_ {c 2}) ^ {2} + 4 \kappa^ {2} = 0,\tag{S18}
$$

which follows that

$$
\big (2 \kappa + i (\varOmega_ {1} + i \gamma_ {c 1} - \varOmega_ {2} - i \gamma_ {c 2}) \big) \big (2 \kappa - i (\varOmega_ {1} + i \gamma_ {c 1} - \varOmega_ {2} - i \gamma_ {c 2}) \big) = 0.\tag{S19}
$$

Thus, two solutions are derived

$$
\kappa_ {1} = \frac {i (\omega_ {1} - \omega_ {2})}{2} + \frac {\gamma_ {1} + \gamma_ {c 2} - \gamma_ {2} - \gamma_ {c 1}}{4},\tag{S20}
$$

or

$$
\kappa_ {2} = \frac {i (\omega_ {2} - \omega_ {1})}{2} + \frac {\gamma_ {c 1} + \gamma_ {2} - \gamma_ {1} - \gamma_ {c 2}}{4}.\tag{S21}
$$

For $\omega_{1} = \omega_{2}$ , and $\kappa > 0$ , we have

$$
\kappa = \frac {| \gamma_ {1} + \gamma_ {c 2} - \gamma_ {2} - \gamma_ {c 1} |}{4}.\tag{S22}
$$

It is worth noting that the critical value of $\kappa$ for an absorbing EP to occur, which is given by Eq. (S22), can be larger, smaller or identical compared to that for a resonant EP to occur, which is given by Eq. (S11). In the phase transition when we gradually increase $\kappa$ , the absorbing EP is reached before the resonant EP if $\gamma_{1} > \gamma_{2}$ , as we observe in Fig. 2 of the main text; but the resonant EP is reached before the absorbing EP if $\gamma_{1} < \gamma_{2}$ , as shown by the experimental and simulation results presented in Fig. S2.

## S4. CPA EP

Besides the conditions for the absorbing EPs, to further satisfy the requirement of a CPA EP, i.e., the absorbing EP occurs for real frequency, we need

$$
0 = I m \frac {(\Omega_ {1} + \Omega_ {2} + i \gamma_ {c 1} + i \gamma_ {c 2})}{2} = \frac {\gamma_ {c 1} + \gamma_ {c 2} - \gamma_ {1} - \gamma_ {2}}{4},\tag{S23}
$$

which yields

$$
\gamma_ {c 1} + \gamma_ {c 2} = \gamma_ {1} + \gamma_ {2}.\tag{S24}
$$

Therefore, we have

$$
\kappa = \frac {| \gamma_ {1} - \gamma_ {c 1} |}{2}.\tag{S25}
$$

We can finally summarize the conditions for CPA EP at $\omega_{1} = \omega_{2}$

$$
\gamma_ {c 1} + \gamma_ {c 2} = \gamma_ {1} + \gamma_ {2},\tag{S26a}
$$

$$
\kappa = \frac {| \gamma_ {1} - \gamma_ {c 1} |}{2}.\tag{S26b}
$$

Under these conditions, we can evaluate the S-matrix

$$
S = A \left( \begin{array}{c c} S _ {1 1} & \pm i \sqrt {\gamma_ {c 1} \gamma_ {c 2}} \\ \pm i \sqrt {\gamma_ {c 1} \gamma_ {c 2}} & S _ {2 2} \end{array} \right),\tag{S27}
$$

where $A = \frac{1}{\left(\delta + i\frac{\gamma_{1} + \gamma_{c1}}{2}\right)\left(\delta + i\frac{\gamma_{2} + \gamma_{c2}}{2}\right) - \kappa^{2}} \frac{\gamma_{1} - \gamma_{c1}}{2}$ , $S_{11} = \left(\frac{\delta}{\frac{\gamma_{1} - \gamma_{c1}}{2}} - i\right)\left(\delta - i\frac{\gamma_{2} + \gamma_{c2}}{2}\right) - \frac{\gamma_{1} - \gamma_{c1}}{2}$ , and $S_{22} = \left(\delta - i\frac{\gamma_{1} + \gamma_{c1}}{2}\right)\left(\frac{\delta}{\frac{\gamma_{1} - \gamma_{c1}}{2}} + i\right) - \frac{\gamma_{1} - \gamma_{c1}}{2}$ . We take the positive sign if $\gamma_{1} < \gamma_{c1}$ , and the negative sign otherwise.

At zero detuning, $S$ becomes

$$
S = A \left( \begin{array}{c c} - \gamma_ {c 2} & \pm i \sqrt {\gamma_ {c 1} \gamma_ {c 2}} \\ \pm i \sqrt {\gamma_ {c 1} \gamma_ {c 2}} & \gamma_ {c 1} \end{array} \right),\tag{S28}
$$

where A becomes a constant. The eigenvector of S associated with the eigenvalue 0 is given by $v_{1}=\left(\frac{\pm i\sqrt{\gamma_{c1}}}{\sqrt{\gamma_{c2}}}\right)$ ; the eigenvector of S associated with the eigenvalue $A(\gamma_{c1}-\gamma_{c2})$

is given by $v_{2}=\left(\frac{\pm i\sqrt{\gamma_{c2}}}{\sqrt{\gamma_{c1}}}\right)$ . We take a positive sign if $\gamma_{1}<\gamma_{c1}$ , and a negative sign otherwise. One sees from this result that for a generic CPA EP ( $\gamma_{c1}\neq\gamma_{c2}$ ) the input amplitudes from the two input ports are not balanced (unequal power), whereas for the non-generic case ( $\gamma_{c1}=\gamma_{c2}$ ), they are balanced, making it easier to find the non-generic CPA EP input eigenvector, as we do in the experiment. Further, for the generic case the S-matrix has two distinct eigenvectors and is not at an EP, even though the wave-operator for incoming boundary conditions is at an EP. However, when $\gamma_{c1}=\gamma_{c2}$ , one sees that the two eigenvalues each becomes zero and the two eigenvectors coalesce, implying that balanced input coupling is a sufficient condition to have an EP of the scattering matrix occur simultaneously with an EP of the wave operator. We note that this coincidence of the two kind of EPs is only found within the TCMT approximation and will not hold exactly when the exact S-matrix of a physical system is calculated due to the presence of other resonances (28).

## S5. Scattering EP

A Scattering EP happens at certain points in the parameter space and at certain frequencies, where the S-matrix becomes defective, the two eigenvalues are degenerate, and the two eigenvectors coalesce (61). The eigenvalue at the degeneracy is not in general zero. The general condition for this is:

$$
\left(\frac {\gamma_ {c 1}}{2} \Big (\delta + i \frac {\gamma_ {2}}{2} \Big) - \frac {\gamma_ {c 2}}{2} \Big (\delta + i \frac {\gamma_ {1}}{2} \Big)\right) ^ {2} + \gamma_ {c 1} \gamma_ {c 2} \kappa^ {2} = 0,\tag{S29}
$$

which leads to

$$
\delta \left(\frac {\gamma_ {c 1}}{2} - \frac {\gamma_ {c 2}}{2}\right) + i \frac {\gamma_ {2} \gamma_ {c 1} - \gamma_ {1} \gamma_ {c 2}}{4} = \pm i \sqrt {\gamma_ {c 1} \gamma_ {c 2}} \kappa .\tag{S30}
$$

The condition for a scattering EP within this TMCT model is thus

$$
\delta (\gamma_ {c 1} - \gamma_ {c 2}) = 0,\tag{S31}
$$

and

$$
\kappa = \pm \frac {\gamma_ {2} \gamma_ {c 1} - \gamma_ {1} \gamma_ {c 2}}{4 \sqrt {\gamma_ {c 1} \gamma_ {c 2}}}.\tag{S32}
$$

To prove consistency with our above results, we put in the condition for CPA EPs. Plugging Eq. (S24) into Eq. (S32), we have

$$
\kappa = \pm \frac {(\gamma_ {c 1} + \gamma_ {c 2}) (\gamma_ {c 1} - \gamma_ {1})}{4 \sqrt {\gamma_ {c 1} \gamma_ {c 2}}}.\tag{S33}
$$

Combining Eq. (S33) with Eq. (S25), we get

$$
\pm \frac {(\gamma_ {c 1} + \gamma_ {c 2}) (\gamma_ {c 1} - \gamma_ {1})}{4 \sqrt {\gamma_ {c 1} \gamma_ {c 2}}} = \frac {| \gamma_ {1} - \gamma_ {c 1} |}{2}.\tag{S34}
$$

To find a solution, we can only choose the sign to make the left-hand side positive. Then we have

$$
\gamma_ {c 1} + \gamma_ {c 2} = 2 \sqrt {\gamma_ {c 1} \gamma_ {c 2}},\tag{S35}
$$

which yields

$$
\gamma_ {c 1} = \gamma_ {c 2}.\tag{S36}
$$

As argued above, this result shows that for the CPA EP case, the S-matrix can be reduced to a defective form (within TCMT approximation) only if we have symmetric coupling strengths in the two channels. For $\gamma_{c1} = \gamma_{c2} = \gamma_{c}$ , the S-matrix takes a defective form

$$
S = A \left( \begin{array}{c c} \frac {\delta^ {2}}{\frac {\gamma_ {1} - \gamma_ {c}}{2}} - \frac {i \gamma_ {c} \delta}{\frac {\gamma_ {1} - \gamma_ {c}}{2}} - \gamma_ {c} & \pm i \gamma_ {c} \\ \pm i \gamma_ {c} & \frac {\delta^ {2}}{\frac {\gamma_ {1} - \gamma_ {c}}{2}} - \frac {i \gamma_ {c} \delta}{\frac {\gamma_ {1} - \gamma_ {c}}{2}} + \gamma_ {c} \end{array} \right),\tag{S37}
$$

where we take the positive sign if $\gamma_{1} < \gamma_{c1}$ , and the minus sign otherwise. The eigenvalue of $S$ is $2A\frac{\delta^{2} - i\gamma_{c}\delta}{\gamma_{1} - \gamma_{c}}$ , and the corresponding eigenvector is

$$
v = \sqrt {\gamma_ {c}} \binom{\mp i}{1}.\tag{S38}
$$

The eigenvector is always the same for arbitrary detuning at the non-generic CPA EP; hence within TCMT we remain at an EP (just not CPA EP, with zero eigenvalue) as the frequency is varied near the CPA EP frequency. This prediction is not valid outside of the TCMT approximation, just as the S-matrix EP and the CPA EP do not exactly coincide in an exact scattering calculation $28$ . The symmetric coupling is also a necessary condition for an absorbing EP and a resonant EP to occur at the same time, again within the TCMT approximation, but not in general (the two EPs will be close in parameter space, but will not exactly coincide).

Theory on the lineshape of the spectrum

We also study the underlying factors that influence the lineshape of the output spectrum. To offer a general description, we discuss the coupled microcavities with two waveguide channels, in which we can choose to inject probe light in various forms and measure both the reflection and transmission spectra.

## S6. The reflection and transmission spectra

When we probe the system from port 1, for example, the reflection spectrum takes the form

$$
r _ {1} = 1 - i \frac {\gamma_ {c 1} \Delta_ {2}}{\Delta_ {1} \Delta_ {2} - \kappa^ {2}}.\tag{S39}
$$

For $\omega_{1} = \omega_{2} = \omega_{0}$ , it becomes

$$
r _ {1} = \frac {\left(\delta + i \frac {\gamma_ {1} - \gamma_ {c 1}}{2}\right) \left(\delta + i \frac {\gamma_ {2} + \gamma_ {c 2}}{2}\right) - \kappa^ {2}}{\left(\delta + i \frac {\gamma_ {1} + \gamma_ {c 1}}{2}\right) \left(\delta + i \frac {\gamma_ {2} + \gamma_ {c 2}}{2}\right) - \kappa^ {2}}.\tag{S40}
$$

We thus obtain the absorption spectrum

$$
R = | r _ {1} | ^ {2} = \frac {\left(\delta^ {2} - \frac {(\gamma_ {1} - \gamma_ {c 1}) (\gamma_ {2} + \gamma_ {c 2})}{4} - \kappa^ {2}\right) ^ {2} + \delta^ {2} \left(\frac {\gamma_ {1} - \gamma_ {c 1} + \gamma_ {2} + \gamma_ {c 2}}{2}\right) ^ {2}}{\left(\delta^ {2} - \frac {(\gamma_ {1} + \gamma_ {c 1}) (\gamma_ {2} + \gamma_ {c 2})}{4} - \kappa^ {2}\right) ^ {2} + \delta^ {2} \left(\frac {\gamma_ {1} + \gamma_ {c 1} + \gamma_ {2} + \gamma_ {c 2}}{2}\right) ^ {2}}.\tag{S41}
$$

The denominator vanishes at the poles, while the numerator vanishes at other complex frequencies (zeros if $\gamma_{c2}=0$ ). In general cases without PT symmetry, the poles and zeros are not complex conjugate pairs. As a result, when the system is probed by laser light with real frequencies, the reflection spectrum is influenced by both the poles and the zeros, and consequently the dips may not occur at the real part of poles or zeros. This analysis can be readily applied to the situation in which the microcavities are probed by only one waveguide channel, since we can treat the coupling to the second channel as an additional loss to the second cavity and combine $\gamma_{c2}$ and $\gamma_{2}$ into a total loss rate $\gamma_{2}^{\prime}$ . In this way, we can write $r_{1}\propto\frac{(\delta-\delta_{z1})(\delta-\delta_{z2})}{(\delta-\delta_{p1})(\delta-\delta_{p2})}$ , where $\delta_{z1,2}=\omega_{z1,2}-\omega_{0}$ . Thereby, the reflection spectrum can be interpreted as the ratio between two sets of geometric distances in the complex plane: the distances between the point $(\omega,0)$ and the zeros $(Re(\omega_{z1,2}),Im(\omega_{z1,2}))$ , and the distance between the point $(\omega,0)$ and the poles $(Re(\omega_{p1,2}),Im(\omega_{p1,2}))$ . Typically, one can find its local minimum when the probe frequency is close to (not exactly at) $Re(\omega_{z1,2})$ and the local maximum when close to (not exactly at) $Re(\omega_{p1,2})$ . Therefore, the split zeros can lead to a doublet in the spectrum. However, it is noted that the poles are further away from the real axis than the zeros $(|Im(\omega_{p1,2})|>|Im(\omega_{Z1,2})|$ when $\kappa>\kappa_{th}$ as shown by Fig. 2B in the main text), so that the influence of the zeros on the spectrum is larger than that of the poles. As a result, when $\kappa$ exceeds the $\kappa_{th}$ for the resonant EP, the central transparency window in the spectrum cannot be split by the separated $Re(\omega_{p1,2})$ .

On the other hand, when we inject light from port 1 and collect the signal from port 3, the transmission spectrum is

$$
\left| t _ {1, 2} \right| ^ {2} = \left| - i \frac {\sqrt {\gamma_ {c 1} \gamma_ {c 2}} \kappa}{\Delta_ {1} \Delta_ {2} - \kappa^ {2}} \right| ^ {2},\tag{S42}
$$

which has a constant numerator and reaches maximum when the laser frequency is equal to the real part of the poles. Thus, the transmission spectrum can be utilized to infer the presence of resonant EPs. In experiments, we found the resonant EP by continuously decreasing the intercavity coupling strength $\kappa$ until two peaks in the transmission spectrum $\left|t_{1,2}\right|^{2}$ coalesce.

Now we turn to the scattered output spectrum when the system is probed by the eigenvector of the S-matrix. One eigenvalue of $S$ has the form $\sigma_{1} = C_{1}\frac{(\omega - \omega_{z1})(\omega - \omega_{z2})}{(\omega - \omega_{p1})(\omega - \omega_{p2})}$ with an eigenvector $v_{1}$ , where $C_{1}$ is a constant, $\omega_{z1,2}$ are the zeros and $\omega_{p1,2}$ are the poles. When probed by $v_{1}$ , the output signal is $v_{out} = Sv_{1} = C_{1}\frac{(\omega - \omega_{z1})(\omega - \omega_{z2})}{(\omega - \omega_{p1})(\omega - \omega_{p2})} v_{1}$ , which also vanishes at the complex zeros and diverges at the complex poles. However, under real frequency probe, the output signal does not completely vanish at $Re(\omega_{z1,2})$ , and the poles which have different real parts than zeros can shift the location of the local minimum. Thus, the scattered signal spectrum for this eigenvector probe with real frequencies is not able to directly reveal the locations of the zeros and poles. For the other eigenvalue $\lambda_{S2}$ which does not have zeros, the extraction of poles is possible. It is also noted that due to the fact that the poles are associated with each element of the S-matrix, there will be no input vector that can avoid the scattering effect of poles. Therefore, the direct detection of zeros will not be straightforward. In our experiments, the locations of zeros are retrieved by curve fitting all the parameters of the system.

## S7. The lineshape of the spectrum at resonant and absorbing EPs

In the Fig. 2 in the main text, we have shown the lineshape of the output spectrum of the coupled microcavities at an absorbing EP displays a doublet instead of a single dip, even though the zeros coalesce. As noted, this can be understood in two ways: first, from the field distribution among the two cavities, and second, from the locations of poles and zeros. Here we discuss more on the explanations.

First, in Fig. 2D, we have shown the simulated distribution of the optical field within the two cavities as a function of frequency detuning. At zero detuning, the optical field tends to localize in $\mu R_{2}$ (since the field distribution is not only influenced by the eigenstate of the Hamiltonian, but also by how the system is probed (62)). Away from the zero detuning, the intracavity energy $E_{2}$ reduces sharply due to the narrow linewidth of $\mu R_{2}$ , while $\mu R_{1}$ , being the only source to couple energy into $E_{2}$ , restores more energy in itself. The total dissipation thereby increases because the redistributed field experiences larger loss in $\mu R_{1}$ ( $\gamma_{1} > \gamma_{2}$ ). Therefore, we find a reduction of the output power (Fig. 2C) when the detuning deviates from zero. This behavior occurs because neither EP occurs on the real axis, and will disappear for the next case, CPA EP.

Second, this could be understood as the result of the existence of two poles with different imaginary parts when zeros become degenerate. As a result, the output reaches the minimum not at $Re(\omega_{z1,2})$ or $Re(\omega_{p1,2})$ , as noted in the main text. Besides what has been found for the absorbing EP, here in Fig. S3, we show the output spectrum at a resonant EP, where the intercavity coupling strength is equal to the critical value, i.e., $\kappa = \kappa_{th}$ . While two dips can still be observed from the transmission spectrum, the transparency window around the zero detuning is slightly broader than that for the absorbing EP. Note that in a more general situation without the restriction of the EPs, such lineshapes have been previously studied in differentiating electromagnetically induced transparency from Autler–Townes splitting (63). Such transparency effect can also arise from the polarization mismatch between two indirectly coupled resonators (64). Here, the results indicate that the spectrum of the scattered output signal may not be considered reliable for judging EPs, since the peaks or dips may not reflect the exact locations of the resonances (poles) or zeros in the complex plane. It is noted that this is true even in the one-waveguide probing case, where we have chosen the correct eigenvector of the S-matrix as the input optical signal.

It is of interest to explore the sufficient and necessary condition that the absorbing EPs and resonant EPs occur at the same conditions, i.e., the zeros and poles become degenerate simultaneously with the same real part of the frequency. We generalize our discussion to a generic case of two coupled optical modes with two incoming and two outcoming channels, modeled under TCMT. The coupling from mode 1 to mode 2 (vice versa) is described by the coupling rates $\kappa_{12}$ ( $\kappa_{21}$ ). The condition for simultaneous resonant and absorbing EPs can be given by

$$
\begin{array}{l} 0 = \sqrt {\left(\omega_ {1} + i \frac {\gamma_ {c 1} - \gamma_ {1}}{2} - \omega_ {2} - i \frac {\gamma_ {c 2} - \gamma_ {2}}{2}\right) ^ {2} + 4 \kappa_ {1 2} \kappa_ {2 1}} \\ = \sqrt {\left(\omega_ {1} - \omega_ {2} - i \frac {\gamma_ {1} + \gamma_ {c 1} - \gamma_ {2} - \gamma_ {c 2}}{2}\right) ^ {2} + 4 \kappa_ {1 2} \kappa_ {2 1}}, \end{array}\tag{S43}
$$

which leads to

$$
\left(\omega_ {1} - \omega_ {2} + i \frac {\gamma_ {c 1} + \gamma_ {2} - \gamma_ {1} - \gamma_ {c 2}}{2}\right) = \pm \left(\omega_ {1} - \omega_ {2} - i \frac {\gamma_ {1} + \gamma_ {c 1} - \gamma_ {2} - \gamma_ {c 2}}{2}\right),\tag{S44a}
$$

$$
0 = \left(\omega_ {1} + i \frac {\gamma_ {c 1} - \gamma_ {1}}{2} - \omega_ {2} - i \frac {\gamma_ {c 2} - \gamma_ {2}}{2}\right) ^ {2} + 4 \kappa_ {1 2} \kappa_ {2 1}.\tag{S44b}
$$

It follows that

$$
\gamma_ {c 1} = \gamma_ {c 2},\tag{S45a}
$$

$$
0 = \left(\omega_ {1} - \omega_ {2} + i \frac {\gamma_ {2} - \gamma_ {1}}{2}\right) ^ {2} + 4 \kappa_ {1 2} \kappa_ {2 1}.\tag{S45b}
$$

Or

$$
\gamma_ {1} = \gamma_ {2},\tag{S46a}
$$

$$
\omega_ {1} = \omega_ {2},\tag{S46b}
$$

$$
\left(\frac {\gamma_ {c 1} - \gamma_ {c 2}}{2}\right) ^ {2} = 4 \kappa_ {1 2} \kappa_ {2 1},\tag{S46c}
$$

Conditions described by Eq. (S45b) or (S46c) can be satisfied by tuning the coupling strengths between the optical modes. Furthermore, we summarize the sufficient and necessary conditions for a resonant EP and an absorbing EP to occur simultaneously: (1) $\gamma_{c1} = \gamma_{c2}$ or (2) $\gamma_{1} = \gamma_{2}, \omega_{1} = \omega_{2}$ . In another word, the coalescence of resonant and absorbing EPs happens only at the situation where we have either symmetric coupling channels or two identical optical modes. This aligns with the theoretical prediction on PT-symmetric systems (65) or gauged PT-symmetric systems (5, 66, 67) except that here we treat the system by TCMT. A gauged PT-symmetric system refers to a physical system that preserves PT-symmetry if we apply a gauge transformation to its complex potential (by for example, factoring out the average loss). In optics, such a gauged PT-symmetric system is often realized by coupling two optical units (for example, resonators or waveguides) with different amount of loss. Unlike the PT-symmetric system which has real eigenvalues in the unbroken regime, the eigenvalues of a gauged PT-symmetric system are shifted away from the real axis due to the overall loss, whereas the PT phase transition feature is preserved. Now we turn to a detailed discussion of each of these cases.

In the first case, we find that the zeros and poles become degenerate simultaneously as long as the coupling channels are symmetric and the intercavity coupling strength is properly tuned, regardless of the other conditions, which enable an absorbing EP and a resonant EP to occur at the same time. This case also represents a gauged PT-symmetric scheme (or PT-symmetric scheme if there is no net gain or loss) where, under TCMT, the scattering EP occurs simultaneously with resonant and absorbing EPs. A special example has been shown by Fig. 3 in the main text, where we have $\omega_{1} = \omega_{2}$ as well, and a coalescence of three types of EPs (resonant, absorbing and scattering EPs) is achieved within the framework of TCMT. In general, the relaxation on the coresonant condition ( $\omega_{1} = \omega_{2}$ ) offers more degrees of freedom for non-Hermitian engineering. For example, with different resonant frequencies for two optical modes and an imaginary coupling strength $\kappa(10)$ , anti-PT symmetry (68) can be readily achieved, for which the EPs must be associated with a single-dip lineshape by probing from symmetric coupling channels.

The second case refers to a situation that there are two optical modes with the same resonant frequencies and loss rates. The asymmetric coupling breaks the gauged PT-symmetry so that the scattering EP does not occur simultaneously. This case could be found, for example, by coupling two identical microcavities, with a judicious design of the asymmetric coupling channels to achieve EPs. Experimentally it will be not straightforward to engineer such identical modes while maintaining asymmetric coupling.

However, a special case belongs to the simultaneous realization of conditions (1) and (2), bringing complete symmetry to the system. In this case, the EP can only be achieved if either $\kappa_{12}$ or $\kappa_{21}$ vanish. One example can be found in a single microcavity consisting of coupled CW and CCW modes with the suppression of one-side backscattering (4, 8, 24, 69–73), just as shown in the chiral absorber (28). In these cases, the real parts of the resonances and zeros can be derived from the peaks in the absorption spectrum.

Apart from the cases discussed above, the peaks or dips in the reflection spectrum may not reflect the correct positions of the resonances or the zeros. The presence of EPs, especially absorbing EPs, is revealed from the curve fitting of parameters instead of a signature lineshape of the transmission/absorption spectra.

## S8. Lineshape of the spectrum at a CPA EP

CPA EPs are special cases at which the total output signal vanishes at zero detuning due to the degenerate and purely real zeros. To investigate the lineshape of the spectra, we assume $\omega_{1}=\omega_{2}=\omega_{0}$ , and investigate the scattered output signal $v_{out}$ under the probe of an eigenvector $v_{in}$ of the S-matrix associated with the eigenvalue $\sigma$

$$
v _ {o u t} = S v _ {i n} = \sigma v _ {i n},\tag{S47}
$$

where the eigenvalues of S-matrix are given by

$$
\begin{array}{r} \sigma_ {1, 2} = \frac {\left(\left(\delta + i \frac {\gamma_ {1}}{2}\right) \left(\delta + i \frac {\gamma_ {2}}{2}\right) + \left(\frac {\gamma_ {c 1}}{2}\right) \left(\frac {\gamma_ {c 2}}{2}\right) - \kappa^ {2}\right)}{\left(\Delta_ {1} \Delta_ {2} - \kappa^ {2}\right)} \\ \pm i \frac {1}{\left(\Delta_ {1} \Delta_ {2} - \kappa^ {2}\right)} \sqrt {\left(\frac {\gamma_ {c 1}}{2} \left(\delta + i \frac {\gamma_ {2} + \gamma_ {c 2}}{2}\right) - \frac {\gamma_ {c 2}}{2} \left(\delta + i \frac {\gamma_ {1} + \gamma_ {c 1}}{2}\right)\right) ^ {2} + \gamma_ {c 1} \gamma_ {c 2} \kappa^ {2}}, \end{array}\tag{S48}
$$

where $\delta = \omega -\omega_0$

For a one-channel CPA EP, S is reduced to a reflection coefficient

$$
r _ {1} = \frac {1}{1 + \frac {i (\gamma_ {1} + \gamma_ {2})}{\delta} - \frac {(\gamma_ {1} + \gamma_ {2}) \gamma_ {2}}{2 \delta^ {2}}}.\tag{S49}
$$

It follows that

$$
R _ {1} = | r _ {1} | ^ {2} = \frac {1}{1 + \frac {(\gamma_ {1} + \gamma_ {2}) \gamma_ {1}}{\delta^ {2}} + \frac {(\gamma_ {1} + \gamma_ {2}) ^ {2} \gamma_ {2} ^ {2}}{4 \delta^ {4}}} = \frac {\delta^ {4}}{\delta^ {4} + (\gamma_ {1} + \gamma_ {2}) \gamma_ {1} \delta^ {2} + \frac {(\gamma_ {1} + \gamma_ {2}) ^ {2} \gamma_ {2} ^ {2}}{4}},\tag{S50}
$$

which takes the form of a quartic lineshape, as $R_{1} \sim \delta^{4}$ when $\delta \to 0$ .

For a two-channel CPA EP, the quartic lineshape happens only for the generic case, where the waveguide-cavity coupling rates are asymmetric. In this case, the eigenvalues of S are given by

$$
\begin{array}{r} \sigma_ {1, 2} = \frac {\delta^ {2} + i \frac {\gamma_ {1} + \gamma_ {2}}{2} \delta - \frac {(\gamma_ {c 2} - \gamma_ {c 1}) (\gamma_ {1} - \gamma_ {c 1})}{4}}{\delta^ {2} + i (\gamma_ {1} + \gamma_ {2}) \delta - \frac {\gamma_ {1} \gamma_ {c 2} + \gamma_ {2} \gamma_ {c 1}}{2}} \\ \pm \frac {i \sqrt {\delta^ {2} \left(\frac {\gamma_ {c 1} - \gamma_ {c 2}}{2}\right) ^ {2} - i \frac {(\gamma_ {c 1} + \gamma_ {c 2}) (\gamma_ {1} - \gamma_ {c 1}) (\gamma_ {c 1} - \gamma_ {c 2})}{4} \delta - \frac {(\gamma_ {c 1} - \gamma_ {c 2}) ^ {2} (\gamma_ {1} - \gamma_ {c 1}) ^ {2}}{1 6}}}{\delta^ {2} + i (\gamma_ {1} + \gamma_ {2}) \delta - \frac {\gamma_ {1} \gamma_ {c 2} + \gamma_ {2} \gamma_ {c 1}}{2}}. \end{array}\tag{S51}
$$

At $\delta = 0$ , one of the eigenvalues approaches zero, which is associated with the perfect absorption channel for the CPA. For $\delta \ll \left| \frac{(\gamma_{1}-\gamma_{c1})(\gamma_{c1}-\gamma_{c2})}{4(\gamma_{c1}+\gamma_{c2})} \right|$ , we can write this eigenvalue in Taylor expansion. In the case $(\gamma_{c2}-\gamma_{c1})(\gamma_{1}-\gamma_{c1}) > 0$ ,

$$
\sigma_ {1} \approx \frac {\delta^ {2}}{\delta^ {2} + i (\gamma_ {1} + \gamma_ {2}) \delta - \frac {\gamma_ {1} \gamma_ {c 2} + \gamma_ {2} \gamma_ {c 1}}{2}},\tag{S52}
$$

for which we have taken the plus sign in Eq. (S49). It is obvious that $|\sigma_{1}|^{2}\sim\delta^{4}$ when $\delta\to0$ , corresponding to a quartic lineshape in the output spectrum when the eigenvector is chosen as the input. For the case $(\gamma_{c2}-\gamma_{c1})(\gamma_{1}-\gamma_{c1})<0$ , $\sigma_{2}$ will take the same form as Eq. (S50), and similar results will be found.

For the non-generic CPA EP, we cannot do a similar Taylor expansion for the eigenvalue, since $\delta \ll \left|\frac{(\gamma_{1}-\gamma_{c1})(\gamma_{c1}-\gamma_{c2})}{4(\gamma_{c1}+\gamma_{c2})}\right|$ is no longer satisfied. Instead, both eigenvalues take the form $\sigma_{1,2} = \frac{\delta^{2} + i\frac{\gamma_{1} + \gamma_{2}}{2}\delta}{\left(\delta + i\frac{\gamma_{1} + \gamma_{2}}{2}\right)^{2}} = \frac{\delta}{\left(\delta + i\frac{\gamma_{1} + \gamma_{2}}{2}\right)}$ . Without loss of generality, we consider the example $\gamma_{1} > \gamma_{c1}$ . The S-matrix is given by

$$
S = A \left( \begin{array}{c c} S _ {1 1} & - i \gamma_ {c 1} \\ - i \gamma_ {c 1} & S _ {2 2} \end{array} \right),\tag{S53a}
$$

where

$$
S _ {1 1} = \left(\frac {\delta}{\frac {\gamma_ {1} - \gamma_ {c}}{2}} + i\right) \left(\delta + i \frac {\gamma_ {2} + \gamma_ {c}}{2}\right) - \frac {\gamma_ {1} - \gamma_ {c}}{2},\tag{S53b}
$$

$$
S _ {2 2} = \left(\frac {\delta}{\frac {\gamma_ {1} - \gamma_ {c}}{2}} - i\right) \left(\delta + i \frac {\gamma_ {1} + \gamma_ {c}}{2}\right) - \frac {\gamma_ {1} - \gamma_ {c}}{2},\tag{S53c}
$$

$$
A = \frac {\gamma_ {1} - \gamma_ {c}}{2 (\Delta_ {1} \Delta_ {2} - \kappa^ {2})} = \frac {\gamma_ {1} - \gamma_ {c}}{2 \left(\delta + i \frac {\gamma_ {1} + \gamma_ {2}}{2}\right) ^ {2}}.\tag{S53d}
$$

Furthermore, the eigenchannel now becomes

$$
v _ {i n} = \sqrt {\gamma_ {c}} \binom {- i} {1}.\tag{S54}
$$

Therefore, the output vector is given by

$$
v _ {o u t} = S v _ {i n} = \frac {\delta}{\left(\delta + i \frac {\gamma_ {1} + \gamma_ {2}}{2}\right)} \binom {- 1} {- i}.\tag{S55}
$$

We get the spectrum of the total output power

$$
| v _ {o u t} | ^ {2} = \frac {2 \delta^ {2}}{\delta^ {2} + \frac {(\gamma_ {1} + \gamma_ {2}) ^ {2}}{4}}.\tag{S56}
$$

We can find that $|v_{out}|^{2}\sim\delta^{2}$ when $\delta\to0$ , which shows the feature of the quadratic lineshape.

In Fig. S4, we show the simulation results of $|r_{1}|^{2}$ , $|t_{1}|^{2}$ , $|r_{2}|^{2}$ and $|t_{2}|^{2}$ at a non-generic CPA EP, which match the experimental results in Figs. 4A and 4B in the main text. Only a single peak is observed in the spectrum of $|t_{1}|^{2}$ (or $|t_{2}|^{2}$ ) as a result of the coalescence of the resonant EP and the CPA EP. $|r_{2}|^{2}$ still displays two dips due to the influences from both the poles and the zeros. The four elements of the S-matrix become equal at the zero detuning in the case of non-generic CPA EP, as seen in Fig. S4.

In Fig. S5(a), we show the experimental results of the spectra of the output signal under eigenvector probe at the non-generic CPA EP. The output spectra display single absorption dips. As shown in Fig. S5(c), away from the CPA EP, one of the spectra shows a doublet. These are verified by numerical results in Figs. S5(b)(d).

In Fig. S6 and Fig. S7, we show the spectra of the output signal with different relative phases between the two inputs when the system is close to the non-generic CPA EP. It is obvious that the relative phase change can easily split the single dip in one of the output spectra, verifying the importance of the correct relative phase.

## Supplementary figures

![](images/e77de8c8abe07e77ad8f3c90f06cb32f5767576c432908318ca97823b0b5234c.jpg)

(b)
![](images/9562ae974e115462e9b370a860c4c5e9d037bbee32035342c901ba09834fc1ce.jpg)

![](images/ef71867ab8681c73ecfbb0b63e9f317f42643f144e3cbe8b23096a55c4099cb8.jpg)
Fig. S1: Experimental setup for CPA EP measurement with two variable input fields. (a) Schematic diagram of the setup. BS: beam splitter; OA: optical attenuator; VDL: variable delay line; EOM: electric-optical modulator; PC: polarization controller; PD: photodetector. (b)(c) Top view (b) and side view (c) of the optical microscope image of the microtoroid resonators coupled to two fiber taper waveguides. The microtoroids used in the experiments have major diameters in the range between $100 \mu m$ and $120 \mu m$ , and minor diameters in the range between $4 \mu m$ and $9 \mu m$ .

![](images/7d066ba854fa8653a25b43741d836f7288413856fb0cc4467902dc9c47d66891.jpg)

![](images/bc610a985f149ac985fe6184a0e3b53cedcb51647a34d55289325e2aa4b1668f.jpg)

$$
\begin{array}{l l} \square \text {Pole} \omega_ {p 1} (\text {Exp}) & \circ \text {Zero} \omega_ {z 1} (\text {Exp}) - - - \text {Poles} \omega_ {p 1, 2} (\text {Theory}) \\ \times \text {Pole} \omega_ {p 2} (\text {Exp}) & + \text {Zero} \omega_ {z 2} (\text {Exp}) - - - \text {Zeros} \omega_ {z 1, 2} (\text {Theory}) \end{array}
$$

Fig. S2: Phase transitions of zeros and poles. Experimentally and theoretically obtained phase transition diagrams for the real parts (a) and imaginary parts (b) of the poles and zeros as a function of normalized coupling strength $\kappa/\kappa_{th}$ . In this experiment and simulation, $\gamma_{1}<\gamma_{2}$ , which is different from the parameter condition for Fig. 2 in the main text.

![](images/735fefe5fa0e7cb5c8467cf022b1c906e1dc3e1a77987144806530694c062e11.jpg)
Frequency detuning (MHz)
Fig. S3: Reflection spectrum at a resonant EP. Reflection spectrum ( $|r_{1}|^{2}$ ) of the coupled microcavities at a resonant EP. The lineshape takes a similar form to that of electromagnetically induce transparency, instead of a single broad dip. The blue and red curves are experimental and curve fitting results, respectively.

![](images/f3d0c89e8016fc27da2296fa16eacd19e1257141a79d099587b4b77df107ece6.jpg)

![](images/062d3a6dcfd7602d97bda956ff507fd897cb120cfdbdbee5a659eda86017cc1c.jpg)
Fig. S4: Scattering properties of the two-channel non-generic CPA EP. (a) Spectra of $R_{1} = |r_{1}|^{2}$ and $T_{1} = |t_{1}|^{2}$ obtained by TCMT. (b) Simulated spectra of $R_{2} = |r_{2}|^{2}$ and $T_{2} = |t_{2}|^{2}$ obtained by TCMT.

![](images/1d29f634ae6907865160c42f5e52f547e35d16cea95eb1e1d87da9e77cbd1ef9.jpg)

![](images/64dc533c7df637e6b4f56d8307d8db6ff14f263ed03587042c6ed0d85c4f10f0.jpg)

(c)
![](images/549145e791715e605a8569daeeb5c5458e5988edd722e671274b4267270e58b2.jpg)

(d)
![](images/f181467b312c9570349501ce6b48170196314b11e16a593257169de4e9025dbd.jpg)
Fig. S5: Extended data at a CPA EP and a CPA away from the EP. (a) Experimentally obtained spectra of the output 1 from port 2 (blue curve), output 2 from port 4 (red curve), and the total absorption (black curve) at the non-generic CPA EP. The amplitudes of the two input signals at the waveguide-cavity coupling points are tuned to be equal, and phases of the two input beams are properly adjusted by the EOM. (b) Simulation result in comparison to the experimental result in (a). Parameters: $\gamma_{1} = 64.782MHz$ , $\gamma_{2} = 242.93MHz$ , $\gamma_{c1} = 153.86MHz$ , $\gamma_{c2} = 153.86MHz$ , $\kappa = 44.538MHz$ . (c) Experimentally obtained spectra when the system is away from the CPA EP. The amplitudes of the two input beams are equal. (d) Simulation result in comparison to the experimental result in (c). Parameters: $\gamma_{1} = 64.782MHz$ , $\gamma_{2} = 242.93MHz$ , $\gamma_{c1} = 184.63MHz$ , $\gamma_{c2} = 123.09MHz$ , $\kappa = 95.877MHz$ .

![](images/c843b66d6b482f47541349a3e5fb08a8f6835c97e3c0f416b4dcb92da83403d9.jpg)
1

![](images/af33b464e539886da1b74330b4af257aecdbe5c3315205c622efeb20da4d15c2.jpg)
2

![](images/deecaaf1f3b3ecdbf8d8b6f6162281d43ca545be6f45aee27f47b06cede4f378.jpg)
3

![](images/e89b3c130a971b88911b331efb9006e0f1ab1cce9c784baa36c6dc9836382e58.jpg)
4

![](images/f208826f9e7abb7b8a3b1542216a5f51e272985438828f1db8287f558e289165.jpg)
5

![](images/265ee6a1c638af1a5b2d0ef85eb1bf215666ef7e87c4db51685ca0dc8142de92.jpg)
6
Fig. S6: Experimental results of the output signal spectra for different relative phases between the two input fields. Experimentally obtained spectra of the output 1 from port 2 (blue curve), output 2 from port 4 (red curve), and the total output (black curve) with different relative phases between the two input beams. The amplitudes of the two input beams are equal. The results numbered from 1 to 6 are given in sequence with the change of the relative phase.

![](images/6bd273f8540309b559d1ff6bd7915d79709fd4689d546f29b27927f6a8ca5208.jpg)
1

![](images/f8e8518f180165bd4a90b84fec34fe501cac8591610f356a1432c24480ec05ce.jpg)
2

![](images/33bdd854d6350afec8b99e03dd5c25d4a6565faeb968e40d05d3f975fc0bc82d.jpg)
3

![](images/2b2831d8f3703bcf389980f2356adf223f629b730bd12b1897f1b7c9c72bcbc5.jpg)
4

![](images/047bd3c7ca7ff5426734f3f5231b3985446e6bfedf6a33af5bd9ecda199e1997.jpg)
5

![](images/375c92c900c3e82c4a5be4843101f9f6402bc74581f70322a1964103d8d3d05d.jpg)
6
Fig. S7: Simulation results of the output signal spectra for different relative phases between the two input fields. Simulated spectra of the output 1 from port 2 (blue curve), output 2 from port 4 (red curve), and the total output (black curve) with various phase difference between the two input beams. The amplitudes of the two input beams are equal. The results numbered from 1 to 6 correspond to the relative phases equal to $\frac{1}{2}\pi$ , $\frac{1}{6}\pi$ , $-\frac{1}{6}\pi$ , $-\frac{1}{2}\pi$ , $-\frac{5}{6}\pi$ , $-\frac{7}{6}\pi$ , respectively.

## References and Notes

1. L. Feng, R. El-Ganainy, L. Ge, Non-Hermitian photonics based on parity–time symmetry. Nat. Photonics 11, 752–762 (2017). doi:10.1038/s41566-017-0031-1

2. R. El-Ganainy, K. G. Makris, M. Khajavikhan, Z. H. Musslimani, S. Rotter, D. N. Christodoulides, Non-Hermitian physics and PT symmetry. Nat. Phys. 14, 11–19 (2018). doi:10.1038/nphys4323

3. M.-A. Miri, A. Alù, Exceptional points in optics and photonics. Science 363, eaar7709 (2019). doi:10.1126/science.aar7709 Medline

4. B. Peng, Ş. K. Özdemir, M. Liertzer, W. Chen, J. Kramer, H. Yılmaz, J. Wiersig, S. Rotter, L. Yang, Chiral modes and directional lasing at exceptional points. Proc. Natl. Acad. Sci. U.S.A. 113, 6845–6850 (2016). doi:10.1073/pnas.1603318113 Medline

5. Y. D. Chong, L. Ge, A. D. Stone, PT-symmetry breaking and laser-absorber modes in optical scattering systems. Phys. Rev. Lett. 106, 093902 (2011). doi:10.1103/PhysRevLett.106.093902 Medline

6. A. Pick, B. Zhen, O. D. Miller, C. W. Hsu, F. Hernandez, A. W. Rodriguez, M. Soljačić, S. G. Johnson, General theory of spontaneous emission near exceptional points. Opt. Express 25, 12325–12348 (2017). doi:10.1364/OE.25.012325 Medline

7. R. Fleury, D. Sounas, A. Alù, An invisible acoustic sensor based on parity-time symmetry. Nat. Commun. 6, 5905 (2015). doi:10.1038/ncomms6905 Medline

8. W. Chen, Ş. Kaya Özdemir, G. Zhao, J. Wiersig, L. Yang, Exceptional points enhance sensing in an optical microcavity. Nature 548, 192–196 (2017). doi:10.1038/nature23281Medline

9. H. Hodaei, A. U. Hassan, S. Wittek, H. Garcia-Gracia, R. El-Ganainy, D. N. Christodoulides, M. Khajavikhan, Enhanced sensitivity at higher-order exceptional points. Nature 548, 187–191 (2017). doi:10.1038/nature23280 Medline

10. Y.-H. Lai, Y.-K. Lu, M.-G. Suh, Z. Yuan, K. Vahala, Observation of the exceptional-point-enhanced Sagnac effect. Nature 576, 65–69 (2019). doi:10.1038/s41586-019-1777-z Medline

11. M. P. Hokmabadi, A. Schumer, D. N. Christodoulides, M. Khajavikhan, Non-Hermitian ring laser gyroscopes with enhanced Sagnac sensitivity. Nature 576, 70–74 (2019). doi:10.1038/s41586-019-1780-4 Medline

12. H. Xu, D. Mason, L. Jiang, J. G. E. Harris, Topological energy transfer in an optomechanical system with exceptional points. Nature 537, 80–83 (2016). doi:10.1038/nature18604 Medline

13. J. Doppler, A. A. Mailybaev, J. Böhm, U. Kuhl, A. Girschik, F. Libisch, T. J. Milburn, P. Rabl, N. Moiseyev, S. Rotter, Dynamically encircling an exceptional point for asymmetric mode switching. Nature 537, 76–79 (2016). doi:10.1038/nature18605 Medline

14. J. W. Yoon, Y. Choi, C. Hahn, G. Kim, S. H. Song, K. Y. Yang, J. Y. Lee, Y. Kim, C. S. Lee, J. K. Shin, H. S. Lee, P. Berini, Time-asymmetric loop around an exceptional point

over the full optical communications band. Nature 562, 86–90 (2018). doi:10.1038/s41586-018-0523-2 Medline

15. W. Tang, X. Jiang, K. Ding, Y.-X. Xiao, Z.-Q. Zhang, C. T. Chan, G. Ma, Exceptional nexus with a hybrid topological invariant. Science 370, 1077–1080 (2020). doi:10.1126/science.abd8872 Medline

16. H. Zhou, C. Peng, Y. Yoon, C. W. Hsu, K. A. Nelson, L. Fu, J. D. Joannopoulos, M. Soljačić, B. Zhen, Observation of bulk Fermi arc and polarization half charge from paired exceptional points. Science 359, 1009–1012 (2018). doi:10.1126/science.aap9859 Medline

17. H. Wang, Y. H. Lai, Z. Yuan, M. G. Suh, K. Vahala, Petermann-factor sensitivity limit near an exceptional point in a Brillouin ring laser gyroscope. Nat. Commun. 11, 1610 (2020). doi:10.1038/s41467-020-15341-6 Medline

18. P. Miao, Z. Zhang, J. Sun, W. Walasik, S. Longhi, N. M. Litchinitser, L. Feng, Orbital angular momentum microlaser. Science 353, 464–467 (2016). doi:10.1126/science.aaf8533 Medline

19. J. Zhang, B. Peng, Ş. K. Özdemir, K. Pichler, D. O. Krimer, G. Zhao, F. Nori, Y. Liu, S. Rotter, L. Yang, A phonon laser operating at an exceptional point. Nat. Photonics 12, 479–484 (2018). doi:10.1038/s41566-018-0213-5

20. L. Feng, Z. J. Wong, R. M. Ma, Y. Wang, X. Zhang, Single-mode laser by parity-time symmetry breaking. Science 346, 972–975 (2014). doi:10.1126/science.1258479 Medline

21. H. Hodaei, M. A. Miri, M. Heinrich, D. N. Christodoulides, M. Khajavikhan, Parity-time-symmetric microring lasers. Science 346, 975–978 (2014). doi:10.1126/science.1258480 Medline

22. Z. J. Wong, Y. L. Xu, J. Kim, K. O'Brien, Y. Wang, L. Feng, X. Zhang, Lasing and anti-lasing in a single cavity. Nat. Photonics 10, 796–801 (2016). doi:10.1038/nphoton.2016.216

23. C. Shi, M. Dubois, Y. Chen, L. Cheng, H. Ramezani, Y. Wang, X. Zhang, Accessing the exceptional points of parity-time symmetric acoustics. Nat. Commun. 7, 11110 (2016). doi:10.1038/ncomms11110 Medline

24. C. Wang, X. Jiang, G. Zhao, M. Zhang, C. W. Hsu, B. Peng, A. D. Stone, L. Jiang, L. Yang, Electromagnetically induced transparency at a chiral exceptional point. Nat. Phys. 16, 334–340 (2020). doi:10.1038/s41567-019-0746-7

25. Y. D. Chong, L. Ge, H. Cao, A. D. Stone, Coherent perfect absorbers: Time-reversed lasers. Phys. Rev. Lett. 105, 053901 (2010). doi:10.1103/PhysRevLett.105.053901 Medline

26. D. G. Baranov, A. Krasnok, T. Shegai, A. Alù, Y. Chong, Coherent perfect absorbers: Linear control of light with light. Nat. Rev. Mater. 2, 17064 (2017). doi:10.1038/natrevmats.2017.64

27. W. Wan, Y. Chong, L. Ge, H. Noh, A. D. Stone, H. Cao, Time-reversed lasing and interferometric control of absorption. Science 331, 889–892 (2011). doi:10.1126/science.1200735 Medline

28. W. R. Sweeney, C. W. Hsu, S. Rotter, A. D. Stone, Perfectly absorbing exceptional points and chiral absorbers. Phys. Rev. Lett. 122, 093901 (2019). doi:10.1103/PhysRevLett.122.093901 Medline

29. Materials and methods are available as supplementary materials.

30. S. Scheel, A. Szameit, PT-symmetric photonic quantum systems with gain and loss do not exist. Europhys. Lett. 122, 34001 (2018). doi:10.1209/0295-5075/122/34001

31. P. Malara, C. E. Campanella, A. Giorgini, S. Avino, P. De Natale, G. Gagliardi, Super-Resonant Intracavity Coherent Absorption. Sci. Rep. 6, 28947 (2016). doi:10.1038/srep28947 Medline

32. S. M. Rao, J. J. F. Heitz, T. Roger, N. Westerberg, D. Faccio, Coherent control of light interaction with graphene. Opt. Lett. 39, 5345–5347 (2014). doi:10.1364/OL.39.005345 Medline

33. X. Fang, M. Lun Tseng, J.-Y. Ou, K. F. MacDonald, D. Ping Tsai, N. I. Zheludev, Ultrafast all-optical switching via coherent modulation of metamaterial absorption. Appl. Phys. Lett. 104, 141102 (2014). doi:10.1063/1.4870635

34. C. Altuzarra, S. Vezzoli, J. Valente, W. Gao, C. Soci, D. Faccio, C. Couteau, Coherent Perfect Absorption in Metamaterials with Entangled Photons. ACS Photonics 4, 2124–2128 (2017). doi:10.1021/acsphotonics.7b00514

35. L. N. Pye, M. L. Villinger, S. Shabahang, W. D. Larson, L. Martin, A. F. Abouraddy, Octave-spanning coherent perfect absorption in a thin silicon film. Opt. Lett. 42, 151–154 (2017). doi:10.1364/OL.42.000151 Medline

36. T. Roger, S. Vezzoli, E. Bolduc, J. Valente, J. J. F. Heitz, J. Jeffers, C. Soci, J. Leach, C. Couteau, N. I. Zheludev, D. Faccio, Coherent perfect absorption in deeply subwavelength films in the single-photon regime. Nat. Commun. 6, 7031 (2015). doi:10.1038/ncomms8031 Medline

37. J. Zhang, K. F. MacDonald, N. I. Zheludev, Controlling light-with-light without nonlinearity. Light Sci. Appl. 1, e18 (2012). doi:10.1038/lsa.2012.18

38. J. M. Rothenberg, C. P. Chen, J. J. Ackert, J. I. Dadap, A. P. Knights, K. Bergman, R. M. Osgood, R. R. Grote, Experimental demonstration of coherent perfect absorption in a silicon photonic racetrack resonator. Opt. Lett. 41, 2537–2540 (2016). doi:10.1364/OL.41.002537 Medline

39. A. Krasnok, A. Alú, Coherent control of light scattering. arXiv:1904.11384 [physics.optics] (25 April 2019).

40. H. Noh, Y. Chong, A. D. Stone, H. Cao, Perfect coupling of light to surface plasmons by coherent absorption. Phys. Rev. Lett. 108, 186805 (2012). doi:10.1103/PhysRevLett.108.186805 Medline

41. T. Carmon, L. Yang, K. Vahala, Dynamical thermal behavior and thermal self-stability of microcavities. Opt. Express 12, 4742–4750 (2004). doi:10.1364/OPEX.12.004742 Medline

42. X. Jiang, L. Yang, Optothermal dynamics in whispering-gallery microresonators. Light Sci. Appl. 9, 24 (2020). doi:10.1038/s41377-019-0239-6 Medline

43. Y. Li, X. Jiang, G. Zhao, L. Yang, Whispering gallery mode microresonator for nonlinear optics. arXiv:1809.04878 [physics.optics] (25 April 2019).

44. L. He, Y.-F. Xiao, J. Zhu, S. K. Özdemir, L. Yang, Oscillatory thermal dynamics in high-Q PDMS-coated silica toroidal microresonators. Opt. Express 17, 9571–9581 (2009). doi:10.1364/OE.17.009571 Medline

45. Y. Liu, X. Jiang, C. Wang, L. Yang, Optothermally induced mechanical oscillation in a silk fibroin coated high-Q microsphere. Appl. Phys. Lett. 116, 201104 (2020). doi:10.1063/1.5142649

46. H. A. Haus, Waves and Fields in Optoelectronics (Prentice-Hall, 1984).

47. J. Zhang, B. Peng, Ş. K. Özdemir, Y. X. Liu, H. Jing, X. Y. Lü, Y. L. Liu, L. Yang, F. Nori, Giant nonlinearity via breaking parity-time symmetry: A route to low-threshold phonon diodes. Phys. Rev. B Condens. Matter Mater. Phys. 92, 115407 (2015). doi:10.1103/PhysRevB.92.115407

48. A. Pick, Z. Lin, W. Jin, A. W. Rodriguez, Enhanced nonlinear frequency conversion and Purcell enhancement at exceptional points. Phys. Rev. B 96, 224303 (2017). doi:10.1103/PhysRevB.96.224303

49. H. Lü, C. Wang, L. Yang, H. Jing, Optomechanically Induced Transparency at Exceptional Points. Phys. Rev. Appl. 10, 014006 (2018). doi:10.1103/PhysRevApplied.10.014006

50. H. Jing, Ş. K. Özdemir, Z. Geng, J. Zhang, X. Y. Lü, B. Peng, L. Yang, F. Nori, Optomechanically-induced transparency in parity-time-symmetric microresonators. Sci. Rep. 5, 9663 (2015). doi:10.1038/srep09663 Medline

51. S. Fan, R. Baets, A. Petrov, Z. Yu, J. D. Joannopoulos, W. Freude, A. Melloni, M. Popović, M. Vanwolleghem, D. Jalas, M. Eich, M. Krause, H. Renner, E. Brinkmeyer, C. R. Doerr, Comment on “Nonreciprocal light propagation in a silicon photonic circuit”. Science 335, 38.2 (2012). doi:10.1126/science.1216682 Medline

52. Y. Choi, C. Hahn, J. W. Yoon, S. H. Song, Observation of an anti-PT-symmetric exceptional point and energy-difference conserving dynamics in electrical circuit resonators. Nat. Commun. 9, 2182 (2018). doi:10.1038/s41467-018-04690-y Medline

53. H. Wang, W. Kong, P. Zhang, Z. Li, D. Zhong, Coherent perfect absorption laser points in one-dimensional anti-parity–time-symmetric photonic crystals. Appl. Sci. MDPI 9, 2738 (2019). doi:10.3390/app9132738

54. F. Yang, Y. C. Liu, L. You, Anti-PT symmetry in dissipatively coupled optical systems. Phys. Rev. A 96, 053845 (2017). doi:10.1103/PhysRevA.96.053845

55. H. Zhang, R. Huang, S.-D. Zhang, Y. Li, C.-W. Qiu, F. Nori, H. Jing, Anti-PT symmetry by spinning a resonator. arXiv:2003.04246 [physics.optics] (9 March 2020).

56. P. Peng, W. Cao, C. Shen, W. Qu, J. Wen, L. Jiang, Y. Xiao, Anti-parity-time symmetry with flying atoms. Nat. Phys. 12, 1139–1145 (2016). doi:10.1038/nphys3842

57. Y. Li, Y. G. Peng, L. Han, M. A. Miri, W. Li, M. Xiao, X. F. Zhu, J. Zhao, A. Alù, S. Fan, C. W. Qiu, Anti-parity-time symmetry in diffusive systems. Science 364, 170–173 (2019). Medline

58. W. Li, H. Zhang, P. Han, X. Chang, S. Jiang, Y. Zhou, A. Huang, Z. Xiao, Real frequency splitting indirectly coupled anti-parity-time symmetric nanoparticle sensor. J. Appl. Phys. 128, 134503 (2020). doi:10.1063/5.0020944

59. H. Zhang, R. Huang, S. D. Zhang, Y. Li, C. W. Qiu, F. Nori, H. Jing, Breaking anti-PT symmetry by spinning a resonator. Nano Lett. 20, 7594–7599 (2020). doi:10.1021/acs.nanolett.0c03119 Medline

60. J. Zhao, Y. Liu, L. Wu, C.-K. Duan, Y.-X. Liu, J. Du, Observation of anti-PT-symmetry phase transition in the magnon-cavity-magnon coupled system. Phys. Rev. Appl. 13, 014053 (2020). doi:10.1103/PhysRevApplied.13.014053

61. A. Krasnok, D. Baranov, H. Li, M. A. Miri, F. Monticone, A. Alú, Anomalies in light scattering. Adv. Opt. Photonics 11, 892–951 (2019). doi:10.1364/AOP.11.000892

62. H. Z. Chen, T. Liu, H. Y. Luan, R. J. Liu, X. Y. Wang, X. F. Zhu, Y. B. Li, Z. M. Gu, S. J. Liang, H. Gao, L. Lu, L. Ge, S. Zhang, J. Zhu, R. M. Ma, Revealing the missing dimension at an exceptional point. Nat. Phys. 16, 571–578 (2020). doi:10.1038/s41567-020-0807-y

63. B. Peng, Ş. K. Özdemir, W. Chen, F. Nori, L. Yang, What is and what is not electromagnetically induced transparency in whispering-gallery microcavities. Nat. Commun. 5, 5082 (2014). doi:10.1038/ncomms6082 Medline

64. C. Wang, X. Jiang, W. R. Sweeney, C. W. Hsu, Y. Liu, G. Zhao, B. Peng, M. Zhang, L. Jiang, A. D. Stone, L. Yang, Induced transparency by interference or polarization. Proc. Natl. Acad. Sci. U.S.A. 118, e2012982118 (2021). doi:10.1073/pnas.2012982118 Medline

65. B. Peng, Ş. K. Özdemir, F. Lei, F. Monifi, M. Gianfreda, G. L. Long, S. Fan, F. Nori, C. M. Bender, L. Yang, Parity–time-symmetric whispering-gallery microcavities. Nat. Phys. 10, 394–398 (2014). doi:10.1038/nphys2927

66. A. Guo, G. J. Salamo, D. Duchesne, R. Morandotti, M. Volatier-Ravat, V. Aimez, G. A. Siviloglou, D. N. Christodoulides, Observation of PT-symmetry breaking in complex optical potentials. Phys. Rev. Lett. 103, 093902 (2009). doi:10.1103/PhysRevLett.103.093902 Medline

67. J. H. Park, A. Ndao, W. Cai, L. Hsu, A. Kodigala, T. Lepetit, Y. H. Lo, B. Kanté, Symmetry-breaking-induced plasmonic exceptional points and nanoscale sensing. Nat. Phys. 16, 462–468 (2020). doi:10.1038/s41567-020-0796-x

68. F. Zhang, Y. Feng, X. Chen, L. Ge, W. Wan, Synthetic Anti-PT Symmetry in a Single Microcavity. Phys. Rev. Lett. 124, 053901 (2020). doi:10.1103/PhysRevLett.124.053901Medline

69. J. Wiersig, Enhancing the sensitivity of frequency and energy splitting detection by using exceptional points: Application to microcavity sensors for single-particle detection. Phys. Rev. Lett. 112, 203901 (2014). doi:10.1103/PhysRevLett.112.203901

70. J. Wiersig, Sensors operating at exceptional points: General theory. Phys. Rev. A 93, 033809 (2016). doi:10.1103/PhysRevA.93.033809

71. W. Chen, C. Wang, B. Peng, L. Yang, “Non-Hermitian physics and exceptional points in high-quality optical microresonators,” in Ultra-High-Q Optical Microcavities, Y.-F. Xiao, Ed. (World Scientific, 2020), pp. 269–313.

72. C. Wang, Z. Fu, L. Yang, “Non-Hermitian physics and engineering in silicon photonics,” in Silicon Photonics IV, D. J. Lockwood, L. Pavesi, Eds. (Springer, 2021), pp. 323–364.

73. S. K. Özdemir, S. Rotter, F. Nori, L. Yang, Parity-time symmetry and exceptional points in photonics. Nat. Mater. 18, 783–798 (2019). doi:10.1038/s41563-019-0304-9 Medline
