## A nonreciprocal and tunable active acoustic scatterer $^{a)}$ FREE

Anis Maddi $^{ID}$ ; Gaelle Poignand $^{ID}$ ; Vassos Achilleos $^{ID}$ ; Vincent Pagneux $^{ID}$ ; Guillaume Penelet $^{ID}$

![](images/55f55e0fa67b8bff246f114e18380c09759f1692ad78abb2385cf26149de60b3.jpg)

Check for updates

![](images/e00412baee481412e87a15daf402115b8c0f86649e370a30b339f666b99cfd67.jpg)

J. Acoust. Soc. Am. 157, 3814–3823 (2025)

https://doi.org/10.1121/10.0036695

![](images/b2ba765740f03c9f1c7059592f2df2ff3cf7f0ec7999df15e98c52c303e1253e.jpg)

# A nonreciprocal and tunable active acoustic scatterer $^{a)}$

Anis Maddi, $^{b)}$ Gaelle Poignand, $id$ Vassos Achilleos, $id$ Vincent Pagneux, $id$ and Guillaume Penelet $id$ Laboratoire d'Acoustique de l'Université du Mans (LAUM), UMR 6613, Institut d'Acoustique—Graduate School (IA-GS), CNRS, Le Mans Université, France

## ABSTRACT:

A passive loudspeaker mounted in a duct acts as a reciprocal scatterer for plane waves impinging on either of its sides. However, the reciprocity can be broken by means of an asymmetric electroacoustic feedback which supplies to the loudspeaker a signal picked-up from a microphone facing only one of its sides. This simple modification offers new opportunities for the control and manipulation of sound waves. In this paper, we investigate the scattering features of a pair of such actively controlled loudspeakers connected by means of a short and narrow duct. The theoretical and experimental results demonstrate that by tuning the feedback loops, the system exhibits several exotic effects, which include an asymmetric reflectionless configuration with one-way transmission or absorption, a directional amplifier with an isolation of 42 dB, and a quasi CPA-lasing configuration. All of these effects were achieved using a single setup in the subwavelength regime, highlighting the versatility of such an asymmetrically active scatterer. © 2025 Acoustical Society of America. https://doi.org/10.1121/10.0036695

(Received 14 December 2024; revised 11 April 2025; accepted 28 April 2025; published online 19 May 2025)

Pages: 3814–3823

## I. INTRODUCTION

The control of sound waves is an active field of research, particularly driven by recent developments in acoustic metamaterials, which have widely extended the capacity to manipulate acoustic waves. Such materials can have various practical applications, $^{1,2}$ including the absorption of sound waves. $^{3-8}$ To date, most developed systems retain reciprocity. However, a growing interest has emerged lately in the development of nonreciprocal acoustic materials. In contrast to conventional systems, these materials allow the unidirectional control of sound waves characterized by an asymmetric wave transmission.

In acoustics, the principle of reciprocity $^{9,10}$ can be broken by various methods, for instance by using large-amplitude waves to trigger nonlinear effects, $^{11-15}$ by imposing a temperature difference across a porous material, $^{16-19}$ by modulating the medium properties in both space and time, or by using active control techniques. $^{20-30}$ Previous applications of these methods have led to the design of systems featuring transmission asymmetries, such as acoustic isolators and circulators. $^{27,31-35}$ However, further studies are needed to cover all the potential applications, while also improving the design of non-reciprocal systems, especially considering their complexity for experimental implementations. Among the methods mentioned, active control allows for easier manipulation of the acoustic waves, $^{36-44}$ but it can also lead to undesired instabilities (audio feedback).

In this paper, we introduce a tunable, subwavelength, and nonreciprocal acoustic system that makes use of actively controlled loudspeakers. Non-reciprocity of a single scatterer is achieved by using a feedback loop, which consists of a microphone placed close to one side of a loudspeaker, and which feeds the latter with an amplified signal proportional to the pressure measured. The experimental system described in the following consists of two identical scatterers, each made up of a loudspeaker enclosed in a cavity and controlled with a feedback loop, while the two cavities are connected via a narrower duct. A single scatterer can be considered a building block of a non-Hermitian topological system, in which the emergence of the non-Hermitian skin effect and topological properties have been previously demonstrated. $^{45}$ Nonhermitian topological systems, $^{46}$ as discussed by Ghaemi and Schomerus, $^{47}$ can exhibit compelling scattering properties by controlling the nonreciprocal coupling in a periodic network, including coherent perfect absorption (CPA), lasing, and reflectionless propagation. In this study, we demonstrate experimentally the possibility to build a versatile nonreciprocal acoustic scatterer with given geometry and components, by only tuning the gain in each independent feedback loop.

A theoretical description of a simplified version of the system is presented in Sec. II, with an objective that is twofold. The first objective is to show that for one cell composed of a single actively controlled loudspeaker, it is possible to achieve broadband nonreciprocity by providing a gain in the feedback loop. It is also shown that by tuning two parameters, namely, the feedback gain and a change in cross-sectional area, the cell can act as a nonreciprocal asymmetric reflectionless two-port and an isolator. The second objective is to show that by using two coupled cells, the resulting two-port can keep the same nonreciprocal properties as for a unique cell, while the tuning becomes easier as it only requires adjusting the gain of each scatterer separately rather than a gain and the geometry. It is notably shown that the system can work as a directional amplifier, $^{48}$ meaning that it not only acts as an isolator but also provides a transmission gain. Moreover, an additional effect of CPA-Laser $^{49-53}$ is enlightened, where the system can achieve either a strong amplification or a coherent perfect absorption of the input power at the same frequency, depending on the amplitude and phasing of the input waves. A global picture of the effects explored in this work is presented in Fig. 1, and an experimental investigation of such exotic scattering properties is presented in Sec. III.

## II. THEORETICAL DESCRIPTION

In this section, we first derive the governing equation of an active loudspeaker where we show the possibility of breaking acoustic reciprocity by supplying to the speaker a current proportional to the acoustic pressure on one side of the membrane. Then, we investigate an electroacoustic cell comprising a non-reciprocal active speaker placed inside a cavity, and we show that the introduction of a cross-sectional area change in addition to the adequate tuning of the gain allows for control of the scattering coefficients. Finally, we explore the scattering features of a system comprising two electroacoustic cells, where the numerical results show the possibility to suppress the reflection in a nonreciprocal system as well as to build a directional amplifier and a CPA-laser.

## A. Breakdown of reciprocity

A loudspeaker mounted inside a duct can be described as a mass-spring-damper system that is submitted to two external forces, namely, one caused by a pressure difference $p_l - p_r$ between the left and right sides of the speaker's membrane, and another one due to an (active) electrodynamic force $F$ , as illustrated in Fig. 2. For a traditional moving coil loudspeaker, this force F stems from the current i that passes through a coil of length $\ell$ in the presence of a magnetic field B. When an acoustic wave propagating along the duct axis arrives at a passive speaker (i.e., F = 0), the latter just acts as a simple oscillator described as a rigid membrane with a mass $M_{m}$ and a surface area $S_{m}$ , and a stiffness $K_{m}$ and mechanical resistance $R_{m}$ . The transmission/reflection/absorption of incident waves by the passive loudspeaker can be easily derived in terms of its scattering matrix. Now, if the speaker is active $F = B\ell i$ , the electrodynamical force can be used to alter the acoustic field. The velocity v of the loudspeaker's membrane is obtained by applying Newton's second law and is written as $^{25,54}$

![](images/7a74f7b57c353c269c5eae5fcb162a1fea36b33128799b192b6a472a68f812ed.jpg)
FIG. 1. Overview of the effects explored in the present work. A single scatterer can achieve different effects by only tuning the electroacoustic feedback loop. These effects include a nonreciprocal unidirectional reflectionless propagation for left or right incident waves, a directional amplifier, and a CPA-laser configuration.

$$
Z _ {l} v = (p _ {l} - p _ {r}) - \frac {B \ell}{S _ {m}} i,\tag{1}
$$

where $Z_{l}=(1/S_{m})[R_{m}+j\omega M_{m}+(K_{m}/j\omega)]$ is the impedance of the loudspeaker, $j^{2}=-1$ , and $B\ell$ is the electrodynamic force factor.

One way to control the speaker is to use an electro-acoustic feedback loop, such that the loudspeaker is supplied with an electric current i proportional to a pressure measured at one of its sides, creating an apparent asymmetry. Herein, the microphone measures a pressure $p_{l}$ , which corresponds to the pressure on the left-hand side of the loudspeaker (see Fig. 2). The signal detected by the microphone then passes through an amplifier with an adjustable gain G, such that the amplifier powers the loudspeaker with the current $i = Gp_{l}$ . Therefore, the resulting electrodynamic force F is also proportional to the pressure facing the left side, giving rise to a nonreciprocal scattering. The transmission through this scatterer indeed depends on the response of the loudspeaker which will differ for left or right incident waves (as will be shown in the following).

Hence, Eq. (1) can be reformulated only in terms of the acoustic variables $(p,v)$ . Moreover, by taking into account the continuity of the velocity v between the left and right sides of the membrane, i.e., $v = v_{\ell} = v_{r}$ , a transfer matrix $M_{0}$ can be derived as follows:

![](images/3418aab89421b95a29be3579506fa5e92d822880dceb8b9cc58fef01270324c5.jpg)
FIG. 2. Sketch of the nonreciprocal electroacoustic device. An active loudspeaker is placed inside a duct and is driven by a signal proportional to the pressure $p_{l}$ facing the left side of the membrane. This results in an electrodynamic force F that depends on the propagation direction.

$$
\binom{p _ {r}}{v _ {r}} = \underbrace {\left( \begin{array}{c c} t & - Z _ {l} \\ 0 & 1 \end{array} \right)} _ {\mathbf {M _ {0}}} \binom{p _ {l}}{v _ {l}},\tag{2}
$$

with

$$
t = 1 - \frac {G B \ell}{S _ {m}}.\tag{3}
$$

It is worth noting that if the amplifier is active, $G \neq 0$ , then the system becomes nonreciprocal, $\det(\mathbf{M}_{\mathbf{0}}) \neq 1$ . Moreover, as the feedback loop includes only a static gain, the parameter t is a real-valued and frequency-independent coefficient. On the other hand, the loudspeaker operates as a passive resonator in the absence of a feedback loop (G = 0).

## B. Scattering of one cell

In the following, a single cell is described, based on the feedback control mentioned previously, but also includes a change in the cross-sectional area on both sides of the loudspeaker, as described in Fig. 3. The system considered consists of a speaker placed in a compact cavity of section $S_{m}$ that is connected to two narrower ducts of cross-section $S_{d}$ . As the system is considered to be compact, the continuity of both pressure and volume velocity applies at the interface between the duct and the cavity, such that

$$
S _ {d} v _ {1} = S _ {m} v _ {\ell}, \quad S _ {d} v _ {2} = S _ {m} v _ {r},\tag{4a}
$$

$$
p _ {1} = p _ {\ell}, \quad p _ {2} = p _ {r}.\tag{4b}
$$

As a result, by taking into account the continuity of velocity and the pressure jump [see Eq. (1)] at the loudspeaker interface, the following equation can be obtained:

$$
\left(j \omega M _ {m} + R _ {m} + \frac {K _ {m}}{j \omega}\right) \frac {S _ {d}}{S _ {m} ^ {2}} v _ {1} = (t p _ {1} - p _ {2}).\tag{5}
$$

The pressures $p_{1,2}$ and velocities $v_{1,2}$ on both sides of the two-port can be decomposed in terms of forward $p^{+}$ and backward $p^{-}$ traveling waves as

$$
p = p ^ {+} + p ^ {-} \quad \& \quad \rho c v = p ^ {+} - p ^ {-},\tag{6}
$$

![](images/b277071b52109dd3ef0e97cf936e77879c58a7dacbc4be6799d86af7c5fa587b.jpg)
FIG. 3. Schematic of the active electroacoustic element placed inside a cavity (of cross-sectional area $S_{m}$ ) that is connected to two narrower ducts (of cross-sectional area $S_{d}$ ).

where c and $\rho$ are the speed of sound and mean density of the fluid, respectively.

This decomposition is used to find the scattering matrix S, which relates the ingoing waves $(p_{1}^{+}, p_{2}^{-})$ to the outgoing waves $(p_{1}^{-}, p_{2}^{+})$ in terms of the scattering coefficients. The scattering problem is written as

$$
\binom{p _ {2} ^ {+}}{p _ {1} ^ {-}} = \left( \begin{array}{c c} \mathbf {T} ^ {+} & \mathbf {R} ^ {-} \\ \mathbf {R} ^ {+} & \mathbf {T} ^ {-} \end{array} \right) \binom{p _ {1} ^ {+}}{p _ {2} ^ {-}},\tag{7}
$$

where the elements of the scattering matrix $T^{+}$ and $R^{+}$ ( $T^{-}$ & $R^{-}$ ) represent the transmission and reflection coefficients of left (right) impinging waves, respectively. Using the previous equations, these coefficients are expressed as follows:

$$
\mathbf {T} ^ {+} = \frac {\frac {2 t}{\alpha} \frac {j \omega}{\omega_ {0}}}{1 - \frac {\omega^ {2}}{\omega_ {0} ^ {2}} + \left[ \frac {1}{Q} + \frac {t + 1}{\alpha} \right] \frac {j \omega}{\omega_ {0}}},\tag{8a}
$$

$$
\mathbf {T} ^ {-} = \frac {\frac {2}{\alpha} \frac {j \omega}{\omega_ {0}}}{1 - \frac {\omega^ {2}}{\omega_ {0} ^ {2}} + \left[ \frac {1}{Q} + \frac {t + 1}{\alpha} \right] \frac {j \omega}{\omega_ {0}}},\tag{8b}
$$

$$
\mathbf {R} ^ {+} = \frac {1 - \frac {\omega^ {2}}{\omega_ {0} ^ {2}} + \left[ \frac {1}{Q} - \frac {t - 1}{\alpha} \right] \frac {j \omega}{\omega_ {0}}}{1 - \frac {\omega^ {2}}{\omega_ {0} ^ {2}} + \left[ \frac {1}{Q} + \frac {t + 1}{\alpha} \right] \frac {j \omega}{\omega_ {0}}},\tag{8c}
$$

$$
\mathbf {R} ^ {-} = \frac {1 - \frac {\omega^ {2}}{\omega_ {0} ^ {2}} + \left[ \frac {1}{Q} + \frac {t - 1}{\alpha} \right] \frac {j \omega}{\omega_ {0}}}{1 - \frac {\omega^ {2}}{\omega_ {0} ^ {2}} + \left[ \frac {1}{Q} + \frac {t + 1}{\alpha} \right] \frac {j \omega}{\omega_ {0}}},\tag{8d}
$$

where $\omega_{0}=\sqrt{K_{m}/M_{m}}$ , $Q=K_{m}/\omega_{0}R_{m}$ are the natural angular frequency and the quality factor of the mechanical resonator, and where $\alpha=(K_{m}/\omega_{0}\rho cS_{m})(S_{d}/S_{m})$ is a geometrical coupling parameter which accounts for the ratio of cross-sectional areas $S_{m}/S_{d}$ .

Unsurprisingly, the expressions of the scattering coefficients show that as far as $t \neq 1$ , the reciprocity is broken and $T^{+} \neq T^{-}$ . Moreover, as all the coefficients depend on the gain t and the change in cross-section through the coupling parameter $\alpha$ , this opens the way for adjusting these two parameters to obtain interesting scattering effects.

Following Fig. 1, a first objective can be, for instance, to make the two-port reflectionless from one side, which can be achieved by setting the numerator to zero in Eq. (8c) such that $R^{+}=0$ , or in Eq. (8d) day such that $R^{-}=0$ . Hence, by setting $R^{\pm}=0$ , we obtain the following complex valued equation:

$$
\left[ \frac {1}{Q} \pm \frac {t - 1}{\alpha} \right] \frac {j \omega}{\omega_ {0}} + 1 - \frac {\omega^ {2}}{\omega_ {0} ^ {2}} = 0.\tag{9}
$$

The solutions of Eq. (9) are found by solving the real and imaginary parts separately, leading to

$$
t = 1 \pm \frac {\alpha}{Q},\tag{10a}
$$

$$
\omega = \omega_ {0}.\tag{10b}
$$

As a result, the two-port can be made one-sided reflectionless at the angular frequency $\omega_{0}$ by adjusting either the gain t or the coupling parameter $\alpha$ to satisfy Eq. (10a).

Another objective can be to make the system transmissionless from one side (see Fig. 1), and Eq. (8a) shows the possibility of blocking the transmission of left incident waves by setting t = 0 such that $T^{+} = 0$ . Interestingly, the latter broadband suppression of the transmitted waves is achieved together with a broadband unitary reflection $R^{+} = 1$ , as can be shown by setting t = 0 in Eq. (8). These broadband effects are obtained no matter the choice of the coupling parameter $\alpha$ , such that adjusting this second parameter can help controlling reflection and transmission from the other side. In particular, $\alpha$ can be chosen such that the scatterer becomes reflectionless for right incident waves at the angular frequency $\omega_{0}$ , leading to the following scattering matrix:

$$
\mathbf {S} (\omega_ {0}) = \left( \begin{array}{c c} 0 & 0 \\ 1 & 1 \end{array} \right),\tag{11}
$$

with $t = 0$ and $\alpha = Q$ .

This scattering matrix describes an acoustic isolator, in which the two-port transmits only right incident waves, whereas it is fully reflective from the opposite side.

The two examples shown previously illustrate that such a system can provide some interesting nonreciprocal scattering effects, obtained by adjusting the gain t and the parameter $\alpha$ . However, there exist limitations to the practical implementation of this system. A first limitation is the fact that, contrarily to the gain t, the parameter $\alpha$ is a geometrical parameter that depends on the ratio $S_{d}/S_{m}$ , and it cannot be tuned for a single device. Moreover, the model presented previously is a simplified model, which notably ignores the presence of a cavity on both sides of the membrane. Actually, those cavities mostly act as additional compliances and inertances that impact the design rules for achieving both reflectionless or transmissionless configurations, as highlighted in the Appendix where a more accurate model is presented.

In order to have a versatile acoustic scatterer with given geometry and components, we decided to introduce a second electroacoustic cell which is connected to the first one. This additional cell serves as an extra degree of freedom to substitute the parameter $\alpha$ which is typically fixed. Therefore, if one fixes a geometry and a loudspeaker (i.e., $\alpha, Q, \omega_{0}$ are set), the new parameters to tune are the independent gains $t_{1}$ and $t_{2}$ of the two electroacoustic cells.

## C. Scattering of two cells

## 1. Simplified model

As mentioned previously, while a single cell can be used (in principle) to break the reciprocity and to control either the reflection or the transmission by tuning the gain t and choosing a parameter $\alpha$ , it is preferable to consider a system consisting of two cells, so that the additional gain offers a tunable degree of freedom. In the following, we first consider a simplified description of a system consisting of two unit cells separated by a duct. This two cell configuration is the one that will be used in experiments, as shown in Fig. 4, which gives a sketch of the experimental setup. The two active loudspeakers can be controlled independently, thus offering greater flexibility to adjust the scattering coefficients.

Similarly to the case of a single cell, we can derive simplified expressions of the scattering coefficients by omitting the impact of the cavities. Here, if one adjusts the length $L_{d}$ of the connecting duct such that it corresponds to a multiple of a wavelength at the resonance frequency of the loudspeakers, then the scattering matrix at $\omega = \omega_{0}$ is given by

$$
\mathbf {S} (\omega_ {0}) = \frac {1}{\frac {\alpha}{Q} + \frac {t _ {1} t _ {2} + 1}{t _ {2} + 1}} \left( \begin{array}{c c} \frac {2 t _ {1} t _ {2}}{t _ {2} + 1} & \frac {\alpha}{Q} + \frac {t _ {1} t _ {2} - 1}{t _ {2} + 1} \\ \frac {\alpha}{Q} - \frac {t _ {1} t _ {2} - 1}{t _ {2} + 1} & \frac {2}{t _ {2} + 1} \end{array} \right).\tag{12}
$$

All the scattering coefficients are explicitly dependent on the gains $t_{1}$ and $t_{2}$ of each speaker. This means that once a geometry and a loudspeaker are selected (i.e., $\omega_{0}$ , Q, and $\alpha$ are fixed), the scattering properties of the system can still be adjusted with the two parameters $t_{1}$ and $t_{2}$ , which can be easily tuned through the gains of the feedback loops.

![](images/f905f0a78949e83bcf33227ce145c8dd2166789c19faebd2aafda0dad44914af.jpg)
J. Acoust. Soc. Am. 157 (5), May 2025
FIG. 4. Sketch of the system comprising two electroacoustic cells. This setup corresponds to the one used experimentally in this study. Each cell includes an independent electroacoustic feedback loop that makes use of a microphone, a current amplifier, and a loudspeaker.

This simplified expression of the scattering matrix coefficients indicates that some effects similar to the ones discussed for one cell can be obtained. This is achieved by adjusting $t_{1}$ and $t_{2}$ , as opposed to t and $\alpha$ in the single cell case. For instance, if $t_{1}=0$ and $1+t_{2}=Q/\alpha$ then $T^{+}=0$ , $R^{+}=1$ , $R^{-}=0$ and $T^{-}=1$ . Yet, this analytical approach is limited since it does not account for the impact of the cavities. Therefore, a numerical approach was adopted for the analysis of the scattering, using the transfer matrix method presented in the Appendix.

## 2. Numerical results

We now proceed to a numerical study of the scattering properties of the two-port as functions of the frequency and for different values of the gains $t_{1}$ and $t_{2}$ . Calculations are performed from a more accurate model which is described in the Appendix, and accounts for the impact of the cavities which were omitted in the simplified model noted previously. Herein, the aim is to explore the effects discussed previously with values taken from the experimental apparatus which is described in Sec. III. More precisely, three distinct effects are investigated and a mapping of their occurrence is given in Fig. 5, depending on the choices of $t_{1,2}$ . The different configurations and their corresponding effects are categorized as follows:

\- One-sided Reflectionless. In this configuration, at least one of the reflection coefficients approaches zero, $\mathbf{R}^{\pm} \to 0$ . The numerical results are highlighted with blue or green surfaces on the map, and correspond either to a vanishing $\mathbf{R}^{+}$ or a vanishing $\mathbf{R}^{-}$ , respectively. In practice, the system cannot be perfectly reflectionless, so the map is generated from the definition of a criterion. Herein, the system is referred to as reflectionless if the magnitude of a reflection coefficient is less than $\mathbf{R}^{\pm} < 0.05$ .

\- Directional amplifier. This particular system acts as an isolator coupled to an amplifier, in which the transmissive port has a gain $A > 1$ . In the ideal case, the scattering matrix looks as follows:

$$
\mathbf {S} = \left( \begin{array}{c c} 0 & 0 \\ 1 & A \end{array} \right).\tag{13}
$$

The configurations that allow us to obtain such a system are mapped with the pink surface in Fig. 5. It was obtained numerically by imposing conditions on the scattering coefficients such that the right incident waves are transmitted with a gain $T^{-} > 1$ and minimal reflection $R^{-} < 0.1$ . On the opposite propagation direction, the magnitude of the transmission needs to be minimized while the reflection should be around unity. These conditions can be satisfied by imposing a restriction on the determinant of the transfer matrix of the system M such that $|\det(\mathbf{M})| < 0.1$ . The resulting minimal isolation factor is $20 \log(|\mathbf{T}^{-}/\mathbf{T}^{+}|) > 20$ dB.

![](images/a0c87edbea136092c3fc93ffef8008bb34f90c9014f46b76a9bc85b17ecc7424.jpg)
FIG. 5. Map of the effects as a function of the parameters $t_{1}$ and $t_{2}$ . Blue, green, pink, and yellow colors represent the configuration that yields a reflectionless configuration, directional amplifier, and CPA-laser, respectively. Gray color highlights the region for which the scatterer is intrinsically unstable. The markers (star, square, dot, and polygon) represent the experimental configurations presented in Sec. III.

\- CPA-laser. Another effect is the CPA-lasing which can be investigated from the singular values $\sigma_{\pm}$ of the scattering matrix. The singular value decomposition satisfies

$$
\mathbf {S} = \mathbf {U} \boldsymbol {\Sigma} \mathbf {V} ^ {\dagger},\tag{14}
$$

where $\dagger$ stands for the conjugate transpose, and where U and V are the orthonormal left and right singular vectors, respectively. Two vectors $U_{\pm}$ and $V_{\pm}$ , are associated with each singular value $\sigma_{+}$ and $\sigma_{-}$ , such that $\Sigma = \text{diag}(\sigma_{+}, \sigma_{-})$ . The singular value decomposition also satisfies

$$
\mathbf {S V} _ {\pm} = \sigma_ {\pm} \mathbf {U} _ {\pm}.\tag{15}
$$

Hence, the singular values decomposition (SVD) allows linearly mapping of the input waves $V_{\pm}$ to the output waves $U_{\pm}$ with a scaling factor $\sigma_{\pm}$ .⁵⁵ The SVD is useful for quantifying the dissipation or generation of acoustic power P,⁵⁶,⁵⁷ as it allows us to find the bounds of the power ratio between the input and output, such that

$$
\sigma_ {-} ^ {2} \leq \frac {\mathcal {P} _ {o u t}}{\mathcal {P} _ {i n}} = \frac {| p _ {2} ^ {+} | ^ {2} + | p _ {1} ^ {-} | ^ {2}}{| p _ {1} ^ {+} | ^ {2} + | p _ {2} ^ {-} | ^ {2}} \leq \sigma_ {+} ^ {2}.\tag{16}
$$

As a result, if the scatterer has a vanishing singular value $\sigma_{-}=0$ and if the incidents waves are adjusted according to the right singular vector $V_{-}$ (i.e., if the input vector $[p_{1}^{+},p_{2}^{-}]^{T}\propto V_{-}$ ), then the system operates as a CPA, meaning that the incident power is fully absorbed by the system. Inversely, lasing happens when the singular value $\sigma_{+}$ is greater than unity, which means that the scatterer can amplify the incident acoustic power with a maximum amplification of $\sigma_{+}^{2}$ if the input waves are tuned according to the right singular vector $V_{+}$ . Finally, a CPA-laser configuration is observed if both coherent perfection absorption and lasing can be achieved at the same frequency. In Fig. 5, the configurations providing CPA-lasing are highlighted with the yellow color on the map. It was considered here that CPA-laser is observed if two conditions are satisfied, namely, an upper bound for the CPA $\sigma_{-}<0.1$ and a lower bound for the lasing $\sigma_{+}>5$ .

Overall, the mapping provided in Fig. 5 gives information on the different settings possible by tuning the parameters $t_{1}$ and $t_{2}$ . It notably shows that the one-sided reflectionless state tends to occur when the nonreciprocal parameters $t_{1,2}$ are distinct. This is because it enhances the mirror asymmetry. Meanwhile, the results also confirm that the directional amplifier configuration is located near the vertical line $t_{1}=0$ as it allows us to fully block left incident waves. Finally, CPA-laser configurations are likely to occur at higher transmission asymmetries $t_{1}t_{2}\gg1$ . Additionally, the gray region shows the configuration for which the system is intrinsically unstable, i.e., instability regardless of where the system is placed. This is determined by analyzing the complex roots of the common denominator of the S matrix obtained using a numerical solver (see the supplementary material). The numerical results indicate that the main region of instability is when the gains are negative $t_{1,2}<0$ .

In Sec. III, an experimental demonstration of each effect is provided, with the selected experimental configurations illustrated in this map with the black markers.

## III. EXPERIMENTAL RESULTS

In Sec. III, the experimental scattering coefficients of the system are measured as functions of the frequency and compared with numerical results. The scattering matrix coefficients are measured using an impedance sensor method (see Refs. 58 and 59 for a description of the method). Note that the system remains stable over a wide range of values for $t_{1,2}$ , which allowed us to explore several configurations. However, for large values of the gains, i.e., $t_{1,2} > 8$ , or for negative values $t_{1,2} < 0$ , undesired self-sustained oscillations are triggered in the system.

The experimental system consists of two unit cells as illustrated in Fig. 4, each with its electroacoustic feedback loop composed of a loudspeaker (model Aura NSW2) with a resonance frequency estimated at 268 Hz (the mechanical parameters were measured using $^{60}$ a Bruel & Kjaer model 4938 microphone; Bruel & Kjaer, Naerum, Denmark, and a current amplifier). $^{61}$ The latter makes it possible to power the loudspeaker with a current directly proportional to the measured pressure and to avoid the impact of the back electromotive force (EMF) on the feedback loop. Each cell consists of a cavity of length $L_{c}=1.8~cm$ and cross-section $S_{c}=15~cm^{2}$ , connected on both sides to a duct of length $L_{d}=9~cm$ and a cross-section $S_{d}=0.5~cm^{2}$ . The total length of the system is $L_{tot}\approx0.22~m$ . Most of the effects discussed in the following are observed at a frequency of about 200 Hz, corresponding to a wavelength of about 1.7 m. Hence, the typical length of the scatterer is much lower than the wavelength.

## A. One-sided reflectionless

## 1. One-way amplification

First, the one-sided reflectionless configuration is investigated, which is highlighted in Fig. 5 by the star and square markers. The scattering coefficients and transfer matrix determinant are plotted in Fig. 6 as functions of the frequency for $t_1 = 1$ and $t_2 = 5.4$ (star marker). The blue and red lines represent the experimental and the theoretical results, respectively. In this configuration, the experimental reflection coefficient for a left-incident wave reaches $|\mathbf{R}^+| = 0.04$ at a frequency of $f = 163\mathrm{Hz}$ , which corresponds to wavelength $\lambda = 9L_{tot}$ . Meanwhile, the transmission coefficient for a wave incident from the left reaches $|\mathbf{T}^+| \approx 1.64$ , indicating a unidirectional amplification of the left incident waves. In contrast, for a right impinging wave, the scatterer acts essentially as an absorber, with a reflection of $|\mathbf{R}^-| \approx 0.47$ and a transmission of $|\mathbf{T}^-| \approx 0.3$ . Consequently, the absorption coefficient is $\alpha^- = 1 - |\mathbf{T}^-|^2 - |\mathbf{R}^-|^2 = 0.69$ . Moreover, the determinant of the transfer matrix is almost constant over the whole frequency range $\det(\mathbf{M}) \approx 5.4$ , indicating a constant asymmetry between the transmission coefficients.

## 2. One-way absorption

Having achieved a reflectionless configuration for left-hand impinging waves, the configuration yielding a reflectionless state for wave incident on the right side is next investigated. Figure 7 depicts the magnitudes of the scattering coefficients as functions of the frequency for $t_{1}=6.4$ and $t_{2}=1$ . For this experimental setting, a wave incident from the right-hand side is mostly absorbed, resulting in a reflection coefficient of $R^{-}\approx0.06$ and a transmission coefficient of $T^{-}\approx0.21$ at f=252 Hz. Meanwhile, for a rightward incident wave, the system has transmission higher than unity and a high reflection coefficient. In this configuration, the system achieves a high subwavelength absorption of the right incident waves with $\alpha^{-}\approx0.96$ and $L_{tot}<\lambda/6$ .

![](images/acfd4bc7366c6a6e26c7b299ce515d319ec08b24c805400022730f18b93ad86b.jpg)
FIG. 6. Magnitude of the scattering coefficients and determinant of the transfer matrix as functions of the frequency for $t_{1}=1$ and $t_{2}=5.4$ . Blue and red colors represent the experimental and theoretical results, respectively.

## B. Directional amplifier

Next, a directional amplifier configuration is investigated experimentally, which is denoted in the map of Fig. 5 with the dot marker. In Fig. 8, the magnitude of the scattering coefficients and the determinant of the transfer matrix are shown as functions of the frequency for $t_1 \approx 0$ and $t_2 = 1$ . In this configuration, the loudspeaker gain $t_1$ is adjusted to achieve a broadband zero determinant, $\det(\mathbf{M}) \approx 0$ , which results in a negligible transmission coefficient for a leftward incident wave, $\mathbf{T}^+ \approx 0$ , as well as a high and broadband isolation factor of 42 dB. Moreover, the left-sided reflection is close to unity, indicating that the device acts as a rigid wall for left incident waves. Meanwhile, for a right-sided incident wave, the system allows a high transmission at low frequency, particularly around $f = 170\mathrm{Hz}$ with an amplification of $A = 1.62$ . At this frequency, the reflection coefficient goes down to less than $10\%$ . Hence, at $f = 170\mathrm{Hz}$ , the system acts as an amplifying diode by transmitting only waves incoming from the right side.

![](images/7ebca6f06d97d2d9740c973812950dc1f6e461f95cf64a01d29415291617eb4a.jpg)
FIG. 7. Magnitude of the scattering coefficients and absorption coefficient $\alpha^{-}$ as a function of the frequency for $t_{1}=6.4$ and $t_{2}=1$ . Blue and red colors represent respectively the experimental and theoretical results.

![](images/c5a69e7fe40cb4b0f05e45a89e7cb6f2b2546f0e1eadc559ad79d8700b780b0a.jpg)
FIG. 8. Magnitude of the Scattering coefficients and determinant of the transfer matrix as a function of the frequency for $t_{1}=0$ and $t_{2}=1$ . Blue and red colors represent respectively the experimental and theoretical results.

## C. Coherent-perfect absorption and lasing

Finally, the CPA-laser mode is investigated experimentally, with a configuration marked by the polygon in Fig. 5. The magnitude of the scattering coefficients and the singular values are plotted in Fig. 9 as functions of the frequency. Here, the gain of both cells was adjusted to the largest possible value before triggering an instability, corresponding to $t_1 = t_2 \approx 8$ . In such a configuration, the system is characterized by a high and broadband isolation factor $20\log (\mathbf{T}^{+} / \mathbf{T}^{-}) \approx 36$ , with a strong amplification of left incident waves $\mathbf{T}^{+} \approx 4.5$ and a low transmission from the opposite side. Furthermore, the system is mirror-symmetrical in this case, and the reflection coefficients reach a minimal value of 0.2 at $f = 205\mathrm{Hz}$ .

The squares of the singular values $\sigma_{\pm}^{2}$ , which quantify the scattered acoustic power, are also plotted as a function of the frequency. It is notably found that around f = 205 Hz, the singular values reach two extreme values, such that the singular value associated with the coherent perfect absorption is $\sigma_{-}^{2} \approx 0.0025$ , while the one for lasing $\sigma_{+}^{2} \approx 20.25$ . These two states of large absorption or large amplification can be achieved by tuning the input waves accordingly with the singular vectors $V_{\pm}$ . Due to the strong nonreciprocity of the system, these eigenvectors correspond primarily to waves incident on only one side, as depicted by the blue vectors in the polar plots. Lasing occurs when the input waves satisfy $|p_{1}^{+}| = 13.2|p_{2}^{-}|$ , denoting a predominance of left-incident waves, while CPA is observed when the input waves satisfy $|p_{1}^{+}| = (13.2)^{-1}|p_{2}^{-}|$ , thus corresponding mostly to a right-incident wave. As with the input, the output waves (see the yellow vectors) are predominantly one-sided, such that the lasing is predominantly to the right $|p_{2}^{+}| = 13.9|p_{1}^{-}|$ , meanwhile, the remaining unabsorbed power for the CPA is to the left. Note that these results are based on the singular value decomposition of the S-matrix measured experimentally, and the system was not excited on either side by these vectors.

![](images/f06758127f948f23ef3d67b348fe8c1a54f6a1b97726f92f9c5318f8c26b980c.jpg)
FIG. 9. (a) Magnitude of the scattering coefficients as a function of the frequency for $t_{1} = t_{2} = 8$ . Blue and red colors represent the experimental and theoretical results, respectively. (b) Singular values $\sigma_{\pm}^{2}$ and the associated singular vectors at the CPA-Lasing configuration.

## IV. CONCLUSION

Using a simple nonreciprocal acoustic device, composed of two actively controlled loudspeakers with asymmetrical feedback loops, we have demonstrated the possibility of exhibiting various scattering effects by simply tuning the feedback gain of each loudspeaker. The experimental and numerical results show good agreement as well as a large range of scattering effects, including a nonreciprocal reflectionless propagation with a one-way transmission gain of 1.64 or an absorption of 96%, a directional amplifier, and nonreciprocal CPA-laser configurations. The presented device yields broadband and high isolation up to 42 dB and operates in the subwavelength regime, typically with $L_{tot} \sim \lambda/6$ or less for the effects highlighted here in experiments. The electroacoustic scatterer studied has the advantage of being simple to implement and to tune, as it only requires us to adjust the gains of the amplifiers. In future works, it could also be interesting to use digital controllers, as they offer more advanced possibilities in terms of electroacoustic feedback loops.

## SUPPLEMENTARY MATERIAL

See supplementary material for further details on the experimental apparatus.

## ACKNOWLEDGMENTS

V.A. acknowledges financial support from the NoHeNA project funded under the program Etoiles Montantes of the Region Pays de la Loire. V.A. Is supported by the EU H2020 ERC StG “NASA” Grant No. 101077954.

## AUTHOR DECLARATIONS

## Conflict of Interest

The authors have no conflicts of interest to declare.

## DATA AVAILABILITY

The data that support the findings of this study are available from the corresponding author upon reasonable request.

## APPENDIX: TRANSFER MATRIX

## 1. Full model for one cell

The transfer matrix of a straight duct of length $L_{c}$ is given by

$$
\mathbf {M} _ {c} = \left( \begin{array}{c c} \cos (k L _ {c}) & - i \rho c \sin (k L _ {c}) \\ - i \sin (k L _ {c}) / (\rho c) & \cos (k L _ {c}) \end{array} \right).\tag{A1}
$$

At the interface between two ducts with a different cross-section, the continuity of pressure and volume velocity allows us to write the following transfer matrix:

$$
\mathbf {M} _ {S _ {1} - S _ {2}} = \left( \begin{array}{c c} 1 & 0 \\ 0 & \frac {S _ {1}}{S _ {2}} \end{array} \right)\tag{A2}
$$

for a transition from a cross-section $S_{1}$ to $S_{2}$ , a speaker with a cross-section $S_{m}$ and a transfer matrix $M_{0}$ is enclosed in a cavity of cross-section $S_{c}$ connected on both sides to ducts with a cross-section $S_{d}$ . The transfer matrix $M_{cell}$ that describes this cell is obtained by using the continuity of pressure and velocity at each interface, which results in the following matrix:

$$
\mathbf {M} _ {\text { cell }} = \mathbf {M} _ {S _ {c} - S _ {d}} \mathbf {M} _ {\mathrm{c}} \mathbf {M} _ {S _ {m} - S _ {c}} \mathbf {M} _ {\mathbf {0}} \mathbf {M} _ {S _ {c} - S _ {m}} \mathbf {M} _ {\mathrm{c}} \mathbf {M} _ {S _ {d} - S _ {c}}.
$$

(A3)

## 2. Simplified version

Equation (A3) can be simplified by assuming that the speaker and the cavity have the same cross-section, $S_{m} = S_{c}$ . Additionally, when the cavity is short compared to the typical wavelength, we can further simplify the problem by taking into account the acoustic compliance $C = L_{c}S_{c}/(\rho c^{2})$ and inductance $I = \rho L_{c}/S_{c}$ of the cavity, such that the transfer matrix writes as

$$
\mathbf {M} _ {c} = \left( \begin{array}{c c} 1 & - j \omega I \\ - j \omega C & 1 \end{array} \right).\tag{A4}
$$

After some algebra, one finds the following condition on the frequency to achieve a vanishing reflection coefficient $R^{\pm}$ ,

$$
\omega = \frac {\omega_ {1}}{\sqrt {1 - \omega_ {1} ^ {2} \frac {C (t + 1)}{\omega_ {0} \alpha} \frac {\rho c S _ {m}}{S _ {d}}}},\tag{A5}
$$

where $\omega_{1} = \sqrt{K / M + IS_{m}}$ .

Equation (A5) shows that the angular frequency at which the reflection can be suppressed depends on the compliance and inductance of the cavity and that it might not have real solutions for all configurations.

## 3. Transfer matrix of two cell

Starting with Eq. (A3) which allows us to write the transfer matrix of one cell, we can write the transfer matrix $M_{sys}$ of a system composed of two cells connected by a duct of transfer matrix $M_{c}$ . The resulting transfer matrix is given by the following product:

$$
\mathbf {M} _ {\text { sys }} = \mathbf {M} _ {\text { cell }, 2} \mathbf {M} _ {\text { c }} \mathbf {M} _ {\text { cell }, 1},\tag{A6}
$$

where $M_{cell,1}$ and $M_{cell,2}$ stand for the transfer matrix of one cell with a gain $t_{1}$ or $t_{2}$ .

$^{1}$ S. A. Cummer, J. Christensen, and A. Alù, “Controlling sound with acoustic metamaterials,” Nat. Rev. Mater. 1(3), 16001 (2016).

$^{2}$ M. R. Haberman and M. D. Guild, “Acoustic metamaterials,” Phys. Today 69(6), 42–48 (2016).

$^{3}$ G. Ma and P. Sheng, “Acoustic metamaterials: From local resonances to broad horizons,” Sci. Adv. 2(2), e1501595 (2016).

$^{4}$ X. Jiang, B. Liang, R.-Q. Li, X.-Y. Zou, L.-L. Yin, and J.-C. Cheng, “Ultra-broadband absorption by acoustic metamaterials,” Appl. Phys. Lett. 105(24), 243505 (2014).

$^{5}$ M. Yang and P. Sheng, “Acoustic metamaterial absorbers: The path to commercialization,” Appl. Phys. Lett. 122(26), 260504 (2023).

$^{6}$ Z. Zhou, S. Huang, D. Li, J. Zhu, and Y. Li, “Broadband impedance modulation via non-local acoustic metamaterials,” Natl. Sci. Rev. 9(8), nwab171 (2022).

$^{7}$ S. Huang, Y. Li, J. Zhu, and D. P. Tsai, “Sound-absorbing materials,” Phys. Rev. Appl 20(1), 010501 (2023).

$^{8}$ Y. Li and B. M. Assouar, “Acoustic metasurface-based perfect absorber with deep subwavelength thickness,” Appl. Phys. Lett. 108(6), 063502 (2016).

$^{9}$ H. Nassar, B. Yousefzadeh, R. Fleury, M. Ruzzene, A. Alù, C. Daraio, A. N. Norris, G. Huang, and M. R. Haberman, “Nonreciprocity in acoustic and elastic materials,” Nat. Rev. Mater. 5(9), 667–685 (2020).

$^{10}$ C. Rasmussen, L. Quan, and A. Alù, “Acoustic nonreciprocity,” J. Appl. Phys. 129(21), 210903 (2021).

$^{11}$ B.-I. Popa and S. A. Cummer, “Non-reciprocal and highly nonlinear active acoustic metamaterials,” Nat. Commun. 5(1), 3398 (2014).

$^{12}$ A. Mojahed, J. Bunyan, S. Tawfick, and A. F. Vakakis, “Tunable acoustic nonreciprocity in strongly nonlinear waveguides with asymmetry,” Phys. Rev. Appl. 12(3), 034033 (2019).

$^{13}$ C. Fu, B. Wang, T. Zhao, and C. Chen, “High efficiency and broadband acoustic diodes,” Appl. Phys. Lett. 112(5), 051902 (2018).

$^{14}$ T. Devaux, A. Cebrecos, O. Richoux, V. Pagneux, and V. Tournat, “Acoustic radiation pressure for nonreciprocal transmission and switch effects,” Nat. Commun. 10(1), 3292 (2019).

$^{15}$ T. Devaux, V. Tournat, O. Richoux, and V. Pagneux, “Asymmetric acoustic propagation of wave packets via the self-demodulation effect,” Phys. Rev. Lett. 115(23), 234301 (2015).

$^{16}$ A. Maddi, C. Olivier, G. Poignand, G. Penelet, V. Pagneux, and Y. Aurégan, “Frozen sound: An ultra-low frequency and ultra-broadband non-reciprocal acoustic absorber,” Nat. Commun. 14(1), 4028 (2023).

$^{17}$ C. Olivier, G. Poignand, M. Malléjac, V. Romero-García, G. Penelet, A. Merkel, D. Torrent, J. Li, J. Christensen, and J.-P. Groby, “Nonreciprocal and even willis couplings in periodic thermoacoustic amplifiers,” Phys. Rev. B 104(18), 184109 (2021).

$^{18}$ T. Biwa, H. Nakamura, and H. Hyodo, “Experimental demonstration of a thermoacoustic diode,” Phys. Rev. Appl. 5(6), 064012 (2016).

$^{19}$ C. Olivier, A. Maddi, G. Poignand, and G. Penelet, “Asymmetric transmission and coherent perfect absorption in a periodic array of thermoacoustic cells,” J. Appl. Phys. 131(24), 244701 (2022).

$^{20}$ C. Shen, J. Li, Z. Jia, Y. Xie, and S. A. Cummer, “Nonreciprocal acoustic transmission in cascaded resonators via spatiotemporal modulation,” Phys. Rev. B 99(13), 134306 (2019).

$^{21}$ X. Zhu, J. Li, C. Shen, X. Peng, A. Song, L. Li, and S. A. Cummer, “Non-reciprocal acoustic transmission via space-time modulated membranes,” Appl. Phys. Lett. 116(3), 034101 (2020).

$^{22}$ J. Li, C. Shen, X. Zhu, Y. Xie, and S. A. Cummer, “Nonreciprocal sound propagation in space-time modulated media,” Phys. Rev. B 99(14), 144311 (2019).

$^{23}$ Y. Chen, X. Li, H. Nassar, A. N. Norris, C. Daraio, and G. Huang, “Nonreciprocal wave propagation in a continuum-based metamaterial with space-time modulated resonators,” Phys. Rev. Appl. 11(6), 064052 (2019).

$^{24}$ N. Geib, A. Sasmal, Z. Wang, Y. Zhai, B.-I. Popa, and K. Grosh, “Tunable nonlocal purely active nonreciprocal acoustic media,” Phys. Rev. B 103(16), 165427 (2021).

$^{25}$ G. Penelet, V. Pagneux, G. Poignand, C. Olivier, and Y. Aurégan, “Broadband nonreciprocal acoustic scattering using a loudspeaker with asymmetric feedback,” Phys. Rev. Appl. 16(6), 064012 (2021).

$^{26}$ A. Sasmal, N. Geib, B.-I. Popa, and K. Grosh, “Broadband nonreciprocal linear acoustics through a non-local active metamaterial,” New J. Phys. 22(6), 063010 (2020).

$^{27}$ X. Wen, H. K. Yip, C. Cho, J. Li, and N. Park, “Acoustic amplifying diode using nonreciprocal willis coupling,” Phys. Rev. Lett. 130(17), 176101 (2023).

$^{28}$ X. Guo, H. Lissek, and R. Fleury, “Observation of non-reciprocal harmonic conversion in real sounds,” Commun. Phys. 6(1), 93 (2023).

$^{29}$ Y. Zhai, H.-S. Kwon, and B.-I. Popa, “Active willis metamaterials for ultracompact nonreciprocal linear acoustic devices,” Phys. Rev. B 99(22), 220301 (2019).

$^{30}$ M. Padlewski, R. Fleury, and H. Lissek, “Amplitude-driven nonreciprocity for energy guiding,” arXiv:2409.20032 (2024).

$^{31}$ R. Fleury, D. L. Sounas, C. F. Sieck, M. R. Haberman, and A. Alù, “Sound isolation and giant linear nonreciprocity in a compact acoustic circulator,” Science 343(6170), 516–519 (2014).

$^{32}$ T. Pedergnana, A. Faure-Beaulieu, R. Fleury, and N. Noiray, “Loss-compensated non-reciprocal scattering based on synchronization,” Nat. Commun. 15(1), 7436 (2024).

$^{33}$ L. Zhang, Y. Ge, Y.-J. Guan, F. Chen, N. Han, Q. Chen, Y. Pan, D. Jia, S.-Q. Yuan, H.-X. Sun, J. Christensen, H. Chen, and Y. Yang, “Nonreciprocal acoustic devices with asymmetric peierls phases,” Phys. Rev. Lett. 133(13), 136601 (2024).

$^{34}$ M. Malléjac and P. R. Fleury, “Experimental realization of an active time-modulated acoustic circulator,” available at SSRN 4958315 (2024).

$^{35}$ Y. Zhu, L. Cao, A. Merkel, S.-W. Fan, B. Vincent, and B. Assouar, “Janus acoustic metascreen with nonreciprocal and reconfigurable phase modulations,” Nat. Commun. 12(1), 7089 (2021).

$^{36}$ R. Fleury, D. Sounas, and A. Alu, “An invisible acoustic sensor based on parity-time symmetry,” Nat. Commun. 6(1), 5905 (2015).

$^{37}$ S. Chen, Y. Fan, Q. Fu, H. Wu, Y. Jin, J. Zheng, and F. Zhang, “A review of tunable acoustic metamaterials,” Appl. Sci. 8(9), 1480 (2018).

$^{38}$ H. Lissek, E. Rivet, T. Laurence, and R. Fleury, “Toward wideband steerable acoustic metasurfaces with arrays of active electroacoustic resonators,” J. Appl. Phys. 123(9), 091714 (2018).

$^{39}$ E. De Bono, M. Collet, G. Matten, S. Karkar, H. Lissek, M. Ouisse, K. Billon, T. Laurence, and M. Volery, “Effect of time delay on the

impedance control of a pressure-based, current-driven electroacoustic absorber," J. Sound Vib. 537, 117201 (2022).

$^{40}$ H. Lissek, R. Boulandet, and R. Fleury, “Electroacoustic absorbers: Bridging the gap between shunt loudspeakers and active sound absorption,” J. Acoust. Soc. Am. 129(5), 2968–2978 (2011).

$^{41}$ K. Wang, L. Shi, H. Zou, S. Zhao, C. Shen, and J. Lu, “A broadband active sound absorber with adjustable absorption coefficient and bandwidth,” J. Acoust. Soc. Am. 156(2), 1048–1057 (2024).

$^{42}$ T. T. Koutserimpas, E. Rivet, H. Lissek, and R. Fleury, “Active acoustic resonators with reconfigurable resonance frequency, absorption, and bandwidth,” Phys. Rev. Appl. 12(5), 054064 (2019).

$^{43}$ S. Sergeev, R. Fleury, and H. Lissek, “Ultrabroadband sound control with deep-subwavelength plasmacoustic metalayers,” Nat. Commun. 14(1), 2874 (2023).

$^{44}$ M. Padlewski, M. Volery, R. Fleury, H. Lissek, and X. Guo, “Active acoustic Su-Schrieffer-Heeger-like metamaterial,” Phys. Rev. Appl. 20(1), 014022 (2023).

$^{45}$ A. Maddi, Y. Auregan, G. Penelet, V. Pagneux, and V. Achilleos, “Exact analog of the hatano-nelson model in one-dimensional continuous nonreciprocal systems,” Phys. Rev. Res. 6(1), L012061 (2024).

$^{46}$ N. Okuma and M. Sato, “Non-Hermitian topological phenomena: A review,” Annu. Rev. Condens. Matter Phys. 14(1), 83–107 (2023).

$^{47}$ H. Ghaemi-Dizicheh and H. Schomerus, “Compatibility of transport effects in non-Hermitian nonreciprocal systems,” Phys. Rev. A 104(2), 023515 (2021).

$^{48}$ D. Malz, L. D. Tóth, N. R. Bernier, A. K. Feofanov, T. J. Kippenberg, and A. Nunnenkamp, “Quantum-limited directional amplifiers with optomechanics,” Phys. Rev. Lett. 120(2), 023601 (2018).

$^{49}$ Y. Aurégan and V. Pagneux, “Pt-symmetric scattering in flow duct acoustics,” Phys. Rev. Lett. 118(17), 174301 (2017).

$^{50}$ G. Poignand, C. Olivier, and G. Penelet, “Parity-time symmetric system based on the thermoacoustic effect,” J. Acoust. Soc. Am. 149(3), 1913–1922 (2021).

$^{51}$ S. Longhi, “PT-symmetric laser absorber,” Phys. Rev. A 82(3), 031801 (2010).

$^{52}$ Y. Chong, L. Ge, and A. D. Stone, “PT-symmetry breaking and laser-absorber modes in optical scattering systems,” Phys. Rev. Lett. 106(9), 093902 (2011).

$^{53}$ M. Yang, Q. Zhong, Z. Ye, S. K. Özdemir, M. Farhat, R. El-Ganainy, and P.-Y. Chen, “Experimental observation of coherent-perfect-absorber and laser points in anti-PT symmetry,” Phys. Rev. A 110(3), 033504 (2024).

$^{54}$ X. Guo, H. Lissek, and R. Fleury, “Improving sound absorption through nonlinear active electroacoustic resonators,” Phys. Rev. Appl. 13(1), 014018 (2020).

$^{55}$ C. Guo, J. Li, M. Xiao, and S. Fan, “Singular topology of scattering matrices,” Phys. Rev. B 108(15), 155418 (2023).

$^{56}$ Y. Aurégan and R. Starobinski, “Determination of acoustical energy dissipation/production potentiality from the acoustical transfer functions of a multiport,” Acta Acust. 85(6), 788–792 (1999).

$^{57}$ T. Holzinger, T. Emmert, and W. Polifke, “Optimizing thermoacoustic regenerators for maximum amplification of acoustic power,” J. Acoust. Soc. Am. 136(5), 2432–2440 (2014).

$^{58}$ C. A. Macaluso and J.-P. Dalmont, “Trumpet with near-perfect harmonicity: Design and acoustic results,” J. Acoust. Soc. Am. 129(1), 404–414 (2011).

$^{59}$ F. C. Bannwart, G. Penelet, P. Lotton, and J.-P. Dalmont, “Measurements of the impedance matrix of a thermoacoustic core: Applications to the design of thermoacoustic engines,” J. Acoust. Soc. Am. 133(5), 2650–2660 (2013).

$^{60}$ A. Novak, Measurement of Loudspeaker Parameters: A Pedagogical Approach (Universitätsbibliothek der RWTH Aachen, Aachen, Germany, 2019).

$^{61}$ A. P. Malvino, D. J. Bates, and P. E. Hoppe, Electronic Principles (Glencoe, Riverside, NJ, 1993).
