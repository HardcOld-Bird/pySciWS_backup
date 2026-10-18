# Supplementary Materials for \\ Coherent Perfect Absorption of Arbitrary Wavefronts\\ at an Exceptional Point

**Authors:** Helmut Hörner, Lena Wild, Yevgeny Slobodkin, Gil Weinberg, Ori Katz, Stefan Rotter



## S1. Deriving the reflection matrix for an MAD-EP-CPA with two cavities

 Here, we derive an expression for the reflection matrix $**R**_\text{L}$ for an MAD-EP-CPA with two cavities (Eq.~(1) in the main text). To do so, we employ a scalar optics model for polarized light, considering a large, yet finite, number of spatial modes. Let $**R**_1$, $**R**_2$, and $**R**_c$ represent the reflection matrices, and $**T**_1$, $**T**_2$, $**T**_\text{c}$, and $**T**_\text{a}$  the transmission matrices for the left, right, central mirror, and the absorber, respectively. Additionally, let $**T**_\text{4f}$ represent the transmission matrix for propagation through a $4f$-system, comprising two lenses in a telescopic configuration. $**T**_\text{4f}$ acts as a double Fourier transform, organizing the development coefficients of the $**k**_\text{xy}$ vectors to produce an image that is flipped in the spatial domain. Moreover, it applies a uniform phase $e^{i \phi}$ to all modes. From these individual transmission matrices, scattering matrices for each optical component and optical propagation can be constructed. These scattering matrices define the relationship between the left-and-right outgoing field amplitude vectors $\mathbf{a}_\mathrm{out}$ in relation to the left-and-right incident amplitude vectors, $\mathbf{a}_\mathrm{in}$. For this particular derivation, we adopt the following convention for any scattering matrix  $\mathbf{S}$:

$$
    \left( \begin{array}{ll} \mathbf{a}^{\mathrm{left}}_{\mathrm{out}} \\
    \mathbf{a}^{\mathrm{right}}_{\mathrm{out}}  \end{array} \right) =
    \mathbf{S}
    \left( \begin{array}{ll} \mathbf{a}^{\mathrm{left}}_{\mathrm{in}} \\
    \mathbf{a}^{\mathrm{right}}_{\mathrm{in}}  \end{array} \right) =
    \left( \begin{array}{cc} \mathbf{R}_\mathrm{L} & \mathbf{T}_\mathrm{RL} \\ \mathbf{T}_\mathrm{LR} & \mathbf{R}_\mathrm{R} \end{array} \right)
    \left( \begin{array}{ll} \mathbf{a}^{\mathrm{left}}_{\mathrm{in}} \\
    \mathbf{a}^{\mathrm{right}}_{\mathrm{in}}  \end{array} \right)

$$

The scattering matrix $\mathbf{S}$ is a block matrix composed of four sub-matrices: $\mathbf{R}_\mathrm{L}$  and $\mathbf{R}_\mathrm{R}$ are the reflection matrices for light incident from the left and right, respectively, while $\mathbf{T}_\mathrm{LR}$ and $\mathbf{T}_\mathrm{RL}$ are the transmission matrices for light passing from left-to-right and right-to-left, respectively. Although these matrices inherently contain complex values to account for phase shifts, the analysis can be simplified. As detailed, for example, in [1,~p.~406], it is customary in single-mode problems (without loss of generality) to represent a scattering matrix using only real numbers, $r$ and $t$, by adopting the following convention:
\begin{eqnarray}
\mathbf{S}=\begin{pmatrix}
r_\mathrm{L} & t_\mathrm{RL} \\
 t_\mathrm{LR} & r_\mathrm{R}
\end{pmatrix} =
\begin{pmatrix}
r & t \\ t & -r
\end{pmatrix} \;\;\; r,t \in \left[ 0, 1\right]
\end{eqnarray}
This convention ensures the energy-conserving unitarity of the scattering matrix, so that $\mathbf{S}\mathbf{S}^\dagger=\openone$, where $\openone$ represents the identity matrix.  Assuming that the same reflection and transmission coefficients apply to all modes for the mirrors and the absorber, the corresponding reflection and transmission matrices $\mathbf{R}$ and $\mathbf{T}$ become diagonal matrices. Hence, convention (§eq:defS) can readily be generalized to our block-matrix-style scattering matrices:
\begin{eqnarray}
\mathbf{S}=\begin{pmatrix}
\mathbf{R}_\mathrm{L} & \mathbf{T}_\mathrm{RL} \\
 \mathbf{T}_\mathrm{LR} & \mathbf{R}_\mathrm{R}
\end{pmatrix} =
\begin{pmatrix}
\mathbf{R} & \mathbf{T} \\ \mathbf{T} & -\mathbf{R}
\end{pmatrix} \;\;\; \mathbf{R}, \mathbf{T} \in \mathbb{R}^{n \times n}
\end{eqnarray}
Given this convention, the scattering matrices for the left, central, and right mirrors can be expressed in the following manner:

$$
\mathbf{S}_1 =   \begin{pmatrix} \mathbf{R}_1 & \mathbf{T}_1 \\ \mathbf{T}_1 & -\mathbf{R}_1 \end{pmatrix} \;\;\;\;\;
\mathbf{S}_\mathrm{c} =   \begin{pmatrix} \mathbf{R}_\mathrm{c} & \mathbf{T}_\mathrm{c} \\ \mathbf{T}_\mathrm{c} & -\mathbf{R}_\mathrm{c} \end{pmatrix}  \;\;\;\;\;
\mathbf{S}_2 =   \begin{pmatrix} \mathbf{R}_2 & \mathbf{T}_2 \\ \mathbf{T}_2 & -\mathbf{R}_2 \end{pmatrix}

$$

 The single-trip propagation in either direction between the mirrors over the distance $4f$ and the absorption in the absorber is represented by the following scattering matrices:

$$
\mathbf{S}_\mathrm{4f} =   \begin{pmatrix} \mathbb{0} & **T**_\text{4f} \\ **T**_\text{4f} & \mathbb{0} \end{pmatrix}  \;\;\;\;\;
\mathbf{S}_\mathrm{a} =   \begin{pmatrix} \mathbb{0} & \mathbf{T}_\mathrm{a} \\ \mathbf{T}_\mathrm{a} & \mathbb{0} \end{pmatrix}

$$

The next step is to convert all scattering matrices into transfer matrices. These transfer matrices relate the incident and outgoing amplitude vectors on the left side to their respective vectors on the right side. For this conversion, we adhere to the following convention for the transfer matrices:

$$
    \left( \begin{array}{ll} \mathbf{a}^{\mathrm{left}}_{\mathrm{out}} \\
    \mathbf{a}^{\mathrm{left}}_{\mathrm{in}}  \end{array} \right) =
    \mathbf{M}
    \left( \begin{array}{ll} \mathbf{a}^{\mathrm{right}}_{\mathrm{in}} \\
    \mathbf{a}^{\mathrm{right}}_{\mathrm{out}}  \end{array} \right)

$$

In order to transform the above-defined scattering matrices into their corresponding transfer matrices, and subsequently back-convert the final transfer matrix $\mathbf{M}_\mathrm{CPA}^\text{(A)}$ for the entire CPA (Design~A) into the scattering matrix $\mathbf{S}_\mathrm{CPA}^\text{(A)}$, we employ the generalized conversions outlined in [2]:
\begin{eqnarray}
\mathbf{M}(\mathbf{S})=
\begin{pmatrix}
\mathbf{S}_{12}-\mathbf{S}_{11}  \mathbf{S}_{21}^{-1} \mathbf{S}_{22} \;\;&
\mathbf{S}_{11} \mathbf{S}_{21}^{-1} \\
-\mathbf{S}_{21}^{-1} \mathbf{S}_{22} &
 \mathbf{S}_{21}^{-1}
\end{pmatrix} \;\;
\mathbf{S}(\mathbf{M})=
\begin{pmatrix}
\mathbf{M}_{12} \mathbf{M}_{22}^{-1} \;\; &
\mathbf{M}_{11}-\mathbf{M}_{12}  \mathbf{M}_{22}^{-1} \mathbf{M}_{21} \\
 \mathbf{M}_{22}^{-1} &
-\mathbf{M}_{22}^{-1} \mathbf{M}_{21}
\end{pmatrix}

\end{eqnarray}
In (§eq:defSMconversion2), $\mathbf{S}_{11}$ is the top-left-quadrant sub-matrix, $\mathbf{S}_{12}$ is the top-right-quadrant sub-matrix, $\mathbf{S}_{21}$ is the bottom-left-quadrant sub-matrix, and $\mathbf{S}_{22}$ is the bottom-right-quadrant sub-matrix of the scattering matrix $\mathbf{S}$. Analogously, $\mathbf{M}_{11}$, $\mathbf{M}_{12}$, $\mathbf{M}_{21}$, and $\mathbf{M}_{22}$ represent the respective quadrants of the transfer matrix $\mathbf{M}$. By employing the $\mathbf{M}(\mathbf{S})$ conversion, we are equipped to transform all the scattering matrices from Eq.~(§eq:defS1ScS2) and Eq.~(§eq:defSpropSa) into their respective transfer matrices. Subsequently, this facilitates the straightforward calculation of the overall transfer matrix for the CPA, which can be expressed as:

$$
    \mathbf{M}_\mathrm{CPA}^\text{(A)} = \mathbf{M}_1 \mathbf{M}_\mathrm{4f} \mathbf{M}_\mathrm{c}  \mathbf{M}_\mathrm{4f} \mathbf{M}_\mathrm{a}
    \mathbf{M}_2 =
    \mathbf{M}(\mathbf{S}_1) \,
    \mathbf{M}(\mathbf{S}_\mathrm{4f}) \,
    \mathbf{M}(\mathbf{S}_\mathrm{c}) \,
    \mathbf{M}(\mathbf{S}_\mathrm{4f}) \,
    \mathbf{M}(\mathbf{S}_\mathrm{a}) \,
    \mathbf{M}(\mathbf{S}_2)


$$

Considering that all the $\mathbf{M}$-matrices mentioned are $2 \times 2$ block matrices, with each quadrant being a matrix in its own right, the following multiplication rule for $2 \times 2$ block matrices $\mathbf{X}$ and $\mathbf{Y}$ must be applied for calculating expression (§eq:Mchain) and any subsequent equivalent expressions:

$$
    \mathbf{X} \mathbf{Y} =
    \begin{pmatrix}
    \mathbf{X}_{11} \mathbf{Y}_{11} +
    \mathbf{X}_{12} \mathbf{Y}_{21} \;\;\;&
    \mathbf{X}_{11} \mathbf{Y}_{12} +
    \mathbf{X}_{12} \mathbf{Y}_{22} \\
    \mathbf{X}_{21} \mathbf{Y}_{11} +
    \mathbf{X}_{22} \mathbf{Y}_{21} \;\;\;&
    \mathbf{X}_{21} \mathbf{Y}_{12} +
    \mathbf{X}_{22} \mathbf{Y}_{22}
    \end{pmatrix}


$$

Using the reverse conversion $\mathbf{S}(\mathbf{M})$ from Eq. (§eq:defSMconversion2), we can subsequently obtain the scattering matrix representing the entirety of the CPA system:

$$
    \mathbf{S}_\mathrm{CPA}^\text{(A)} = \mathbf{S}(\mathbf{M}_\mathrm{CPA}^\text{(A)})

$$

Our main interest is the top-left quadrant of the scattering matrix $\mathbf{S}_\mathrm{CPA}^\text{(A)}$, which corresponds to the CPA's reflection matrix for Design~A, denoted as $ \mathbf{R}_\text{CPA}^\text{(A)}$:

$$
    **R**_\text{CPA}^\text{(A)} = \frac{(**R**_1  **A**
    + **B** **E** ) \, **D** +
    ( **B** **C** +
    **R**_1 **A** \, **R**_\text{c}
    ) \, **G**}{( **F** **E** +
    **A**)\, **D** + (**A** **R**_\text{c} +
    **F** **C** ) \, **G** }

$$

 with the following substitutions:

$$
\begin{aligned}
    **A** &= \left( **T**_\text{c} **T**_\text{4f} **T**_1\right)^{-1} \\
    **B** &= **T**_1 + **R**_1 **T**_1^{-1} **R**_1 \\
    **C** &= **T**_\text{4f} (**T**_\text{c} + **R**_\text{c} **T**_**c**^{-1} **R**_\text{c}) \\
    **D** &= ( **T**_\text{a} **T**_\text{4f} )^{-1} \\
    **E** &= **T**_\text{4f} \, **R**_\text{c} **T**_\text{c}^{-1} \\
    **F** &= **T**_1^{-1} **R**_1 \\
    **G** &= **T**_\text{4f} \, **T**_\text{a} **R**_\text{2}
\end{aligned}
$$

Assuming that the same reflection and transmission coefficients apply to all modes at the mirrors and the absorber, all matrices except $ **T**_\text{4f}$ become diagonal matrices:

$$
\begin{aligned}
    **R**_1 = r_1 \mathbb{1}, \;\;
    **R**_2 = r_2 \mathbb{1}, \;\;
    **R**_\text{c} = r_\text{c} \mathbb{1}, \;\;
    **T**_1 = t_1 \mathbb{1}, \;\;
    **T**_2 = t_2 \mathbb{1}, \;\;
    **T**_\text{c} = t_\text{c} \mathbb{1}, \;\;
    **T**_\text{a} = t_\text{a} \mathbb{1}

\end{aligned}
$$

Thus, Eq. (§eq:RCPA) can be significantly simplified to

$$
\begin{aligned}
    **R**_\text{CPA}^\text{(A)} = \frac{t_\text{a}^2  (r_1^2 + t_1^2) (r_\text{c}^2 + t_\text{c}^2) r_2
    \, **T**_\text{4f}^4 +r_\text{c}   (r_1^2 + t_1^2 + r_1 r_2 t_\text{a}^2)  \,**T**_\text{4f}^2   + r_1 \mathbb{1}}{r_1 r_2 t_\text{a}^2 (r_\text{c}^2 + t_\text{c}^2) \, **T**_\text{4f}^4 + r_\text{c}(r_1+ r_2 t_\text{a}^2)  \, **T**_\text{4f}^2 + \mathbb{1}}
\end{aligned}
$$

Considering the energy conservation relations $ r_1^2 + t_1^2 = 1$ and $r_c^2 + t_c^2 = 1$, and assuming $r_2 = 1$ (indicating a totally reflective mirror on the right side), the equation simplifies to:

$$
\begin{aligned}
    **R**_\text{CPA}^\text{(A)} = \frac{t_\text{a} ^2 \, **T**_\text{4f}^4 +r_\text{c}   (1+r_1 t_\text{a}^2)  \,**T**_\text{4f}^2   + r_1 \mathbb{1}}{r_1 t_\text{a}^2 \, **T**_\text{4f}^4 + r_\text{c}(r_1+t_\text{a}^2)  \, **T**_\text{4f}^2 + \mathbb{1}}
\end{aligned}
$$

This corresponds to Eq.~(1) in the main paper. The matrix $**T**_\text{4f}^2$ results in a quadruple Fourier transformation with a uniform phase shift $e^{2i\phi}$ for all modes and can thus be represented as a simple diagonal matrix $**T**_\text{4f}^2=e^{2i\phi} \mathbb{1}$. Analogously, $**T**_\text{4f}^4=e^{4i\phi} \mathbb{1}$. Consequently, $**R**_\text{CPA}^\text{(A)}$ becomes a diagonal matrix with identical entries $r_\text{CPA}^\text{(A)}(\omega)$ along the diagonal and can be reduced to a simple scalar equation, which proves the spatial degeneracy of the system.

$$
    r_\mathrm{CPA}^\text{(A)} =
    \frac{t_\mathrm{a}^2 e^{4 i \phi} +
    r_\mathrm{c} \left(1 + r_1 t_\mathrm{a}^2\right) e^{2 i \phi} + r_1}
    {r_1 t_\mathrm{a}^2   e^{4 i \phi}   +
    r_\mathrm{c}  \left(r_1+ t_\mathrm{a}^2\right) e^{2 i \phi}
    + 1}.

$$

In Eq.~(§eq:rLEPCPA), $r_1$ is the reflection-coefficient of the left mirror, $r_\text{c}$ is the reflection-coefficient of the central mirror, and $t_\mathrm{a}$ is the one-way transmission coefficient of the weak absorber. For an EP-CPA, we first need to set the expression in Eq.~(§eq:rLEPCPA) to zero for total absorption. Subsequently, we  solve for $\phi$, to find the parameters where the two distinct solutions merge to an Exceptional Point.

## S2. Deriving the CPA conditions for an Exceptional Point CPA with two cavities

 In this section,  the derivation of the CPA conditions for an EP-CPA with two cavities are explained in detail. For total absorption, we set the expression in Eq.~(§eq:rLEPCPA) to zero, and then solve for $\phi$:
\begin{eqnarray}
&&\phi=\pi m - \frac{i}{2}\log{\left(
-\frac{r_\mathrm{c}  +
r_1 r_\mathrm{c} t_\mathrm{a}^2}
{2 t_\mathrm{a}^2}
 \pm \frac{\sqrt{\left( r_\mathrm{c} + r_1 r_\mathrm{c} t_\mathrm{a}^2 \right)^2 -
4 r_1 t_\mathrm{a}^2}}
{2 t_\mathrm{a}^2} \right)}

\;\;\; \mathrm{with}\;\;\; m \in \mathbb{N}
\end{eqnarray}
At the exceptional point, the two distinct solutions for $\phi$ in Eq.~(§eq:three) coincide. Consequently, the expression beneath the square root must become zero.
\begin{eqnarray}
 \left( r_\mathrm{c} + r_1 r_\mathrm{c} t_\mathrm{a}^2 \right)^2 -
4 r_1 t_\mathrm{a}^2 = 0

\end{eqnarray}
for $r_\mathrm{c}$, we find
\begin{eqnarray}
r_\mathrm{c} = \frac{2 \sqrt{r_1} t_\mathrm{a}}{1 + r_1 t_\mathrm{a}^2}

\end{eqnarray}
By substituting this result into Eq.~(§eq:rLEPCPA), and setting the equation to zero again, we can solve for the transmission coefficient $t_\mathrm{a}$. This leads to the derivation of the relationship $t_\mathrm{a} = -  e^{-2 i \phi} \sqrt{r_1}$. We require $t_\mathrm{a} \in \mathbb{R}_+$. This is the case when $e^{-2 i \phi} = -1$, which is equivalent to $\phi_m = \frac{\pi +2 \pi  m}{2}$ ($m \in \mathbb{Z}$). These particular values of $\phi_m$ correspond to the resonance wave numbers $k_m = \frac{\phi_m}{4f}$ that yield perfect absorption. Assuming this condition,  we proceed to write
\begin{eqnarray}
t_\mathrm{a} = \sqrt{r_1}

\end{eqnarray}
Plugging this into Eq.~(§eq:five) yields
\begin{eqnarray}
r_\mathrm{c} = \frac{2r_1}{1+r_1^2}

\end{eqnarray}
By squaring both sides of Eq.~(§eq:six) and substituting $t_\mathrm{a}^2 \rightarrow T_\mathrm{a}$ and $r_1 \rightarrow \sqrt{R_1}$, we get
\begin{eqnarray}
T_\mathrm{a} = \sqrt{R_1}
\;\;\Leftrightarrow\;\;
R_1 = T_\mathrm{a}^2
\end{eqnarray}
Finally, squaring both sides of Eq.~(§eq:rcEPCPA), and substituting $r_\mathrm{c}^2 \rightarrow R_\mathrm{c}$, $r_1^2 \rightarrow R_1$ leads to
\begin{eqnarray}
R_\mathrm{c} = \frac{4 R_1}{\left(1+R_1 \right)^2}

\end{eqnarray}
This proves the EP-conditions in Eq. (2) of the main text.\\

## S3. Alternative Derivation of CPA Conditions  for 2-Cavity-EP-CPA and Quartic Behavior Proof

 In this section we demonstrate an alternative approach for deriving the CPA conditions for an EP-CPA with 2 cavities. In contrast to the previous method, this approach can be generalized for CPAs with more than two cavities. For perfect absorption, the condition $R_\text{CPA}^\text{(A)}(\phi) = |r_\mathrm{CPA}^\text{(A)}(\phi)|^2  = 0$ must be satisfied. Calculating $R_\mathrm{CPA}^\text{(A)}$ yields

$$
    R_\mathrm{CPA}^\text{(A)}(\phi) =
    \frac{r_1^2 r_\mathrm{c}^2 t_\mathrm{a}^4+r_1^2+2 r_1 r_\mathrm{c}^2 t_\mathrm{a}^2+2 r_\mathrm{c} \left(r_1+t_\mathrm{a}^2\right) \left(r_1 t_\mathrm{a}^2+1\right) \cos (2 \phi )+2 r_1 t_\mathrm{a}^2 \cos (4 \phi )+r_\mathrm{c}^2+t_\mathrm{a}^4}{t_\mathrm{a}^4 \left(r_1^2+r_\mathrm{c}^2\right)+r_1^2 r_\mathrm{c}^2+2 r_1 r_\mathrm{c}^2 t_\mathrm{a}^2+2 r_\mathrm{c} \left(r_1+t_\mathrm{a}^2\right) \left(r_1 t_\mathrm{a}^2+1\right) \cos (2 \phi )+2 r_1 t_\mathrm{a}^2 \cos (4 \phi )+1}


$$

 Plotting $R_\mathrm{CPA}^\text{(A)}(\phi)$ with a fixed value for $r_1$ and different arbitrary values for $r_\mathrm{c}$ and $ t_\mathrm{a}$, as depicted in Fig.~§fig:deriv_EP_CPA2a,  it becomes evident that perfect absorption can be anticipated at $\phi=\frac{\pi}{2}$, provided we can find the appropriate values for $r_\mathrm{c}$ and $ t_\mathrm{a}$.
However, our objective is not solely to ensure that $R_\mathrm{CPA}^\text{(A)}(\phi)$ equates to zero at $\phi=\frac{\pi}{2}$; we also want to observe the broadening of the absorption dip caused by the exceptional point. This is the case when $R_\mathrm{CPA}^\text{(A)}(\phi)$ exhibits a saddle point at $\phi=\frac{\pi}{2}$. Mathematically, this is equivalent to asserting that the second derivative of $R_\mathrm{CPA}^\text{(A)}(\phi)$ with respect to $\phi$ also vanishes at $\phi=\frac{\pi}{2}$. This leads to the following system of equations:
\begin{eqnarray}
\left. R_\mathrm{CPA}^\text{(A)}\left(\phi\right)\right\vert_{\phi=\frac{\pi}{2}} &= 0 \\
tial_\phi^2 \left. R_\mathrm{CPA}^\text{(A)}\left(\phi\right)\right\vert_{\phi=\frac{\pi}{2}} &= 0
\end{eqnarray}
Solving this system of equations yields the solutions $t_\mathrm{a}=\sqrt{r_1}$ and $r_\mathrm{c} = \frac{2r_1}{1+r_1^2}$. These solutions are consistent with the conditions (§eq:six) and (§eq:rcEPCPA) derived previously. By incorporating  (§eq:six) and (§eq:rcEPCPA) into Eq. (§eq:RLofPhi), and substituting $r_1^2 = R_1$, we obtain the following expression:

$$
    R_\mathrm{CPA}^\text{(A)}(\phi) =
    \frac{16\left(1 + R_1 \right)^2 \cos^4(\phi)}{1 + 2 R_1 + 18 R_1^2 + 2 R_1^3 + R_1^4 + 8 R_1 \left(1+R_1 \right)^2 \cos(2 \phi) + 2 R_1 \left(1 + R_1 \right)^2 \cos(4 \phi)}

$$

Developing this expression around the resonance points $\phi_m = \frac{2m+1}{2} \pi$ reveals the anticipated quartic behavior of  $R_\mathrm{CPA}^\text{(A)}$:

$$
    R_\mathrm{CPA}^\text{(A)}(\phi) = \frac{16 R_1(R_1+1)^2}{(R_1-1)^4} \left(\phi - \frac{2m+1}{2} \pi \right)^4 + \mathcal{O}^6\left(\phi - \frac{2m+1}{2} \pi \right)^6

$$

[h]

_Figure/Table caption: ^\text{(A)}(\phi)$ with $r_1=0.7$ and some arbitary values for $r_\mathrm{c}$ and $ t_\mathrm{a}$ shows that we can expect to find perfect absorption at $\phi=\frac{\pi}{2}$, provided we find the right values for $r_\text{c}$ and $t_\text{a}$.}



## S4. Critical Parameters

 For practical implementations, it is insightful to investigate the impact of minor alterations in the system parameters on the efficacy of the MAD-EP-CPA depicted in Fig.~1(d) of the main paper. Some deviations~--~such as the influence of unintended lens reflections or a slight tilt in one of the mirrors -- necessitate a numerical simulation of the actual system for accurate assessment. Nevertheless, to analyze the effects stemming from deviations in the reflectivities of the mirrors or the transmissivity of the absorber, the straightforward analytical solution provided by Eq.~(§eq:RLofPhi) proves to be adequately sufficient. As is evident from Fig.~§fig:parameters-insensitive, these calculations reveal that even deviations in the magnitude of a few percent in the mirror reflectivity $R_1$, or in the right absorber's transmissivity $T_\mathrm{a}$ do not substantially compromise the performance of the Exceptional Point CPA. Contrastingly, deviations in the reflectivity $R_\mathrm{c}$ of the central coupling mirror yield significantly different results. As illustrated in Fig.~§fig:parameters-sensitive(a), even a mere one percent deviation in the central mirror’s reflectivity $R_\mathrm{c}$ can substantially impair the performance of the EP-CPA. The precise alignment of the two cavity lengths is even more crucial. Fig.~§fig:parameters-sensitive(b) demonstrates that deviations in the cavity lengths equivalent to less than one percent of the wavelength are already consequential.

_Figure/Table caption: $ of the absorber. (Assumed wavelength $\lambda_0=633 \mathrm{nm}$)}

_Figure/Table caption: $  of the central coupling mirror can significantly impact the effectiveness of the EP-CPA. (b) This is even more evident for minor deviations in the length of the second cavity. (Calculation performed for a reference wavelength $\lambda_0=633 \mathrm{nm}$.)}


 In reference [3], it was demonstrated that in an actual experiment involving a single-cavity MAD-CPA, the weak spurious reflections originating from the facets of the cavity lenses (with a reflectivity $R=0.13 \%$ per surface) played a significant role in diminishing the CPA effect. Given this, it is anticipated that these spurious reflections would have an even more pronounced impact on the MAD-EP-CPA, as it incorporates four lenses. To accurately model and understand the effects of these spurious lens reflections,  the simple analytical approach utilized thus far is deemed inadequate. Consequently, we resort to the numerical Scalar Fourier Optics computer simulation, a method that has previously demonstrated excellent concordance with experimental measurements, as evidenced in [3]. Fig.~§fig:lens-refl presents the outcome of such a simulation. In this simulation, a random speckle field consisting of 160 modes of equal amplitude and random phase in $k$-space serves as input to compute the power-reflection spectrum of the MAD-EP-CPA in comparison to a simple, single-cavity MAD-CPA. It illustrates the detrimental impact of the assumed spurious lens reflections, with a reflectivity $R=0.13 \%$ per lens-facet, on the performance of the MAD-EP-CPA. As a result of these spurious reflections, the power reflection over the whole field-of-view does not dip below approximately $2 \%$, and although the EP broadening-effect remains observable, it is notably less prominent compared to the simulation that omits spurious lens reflections.

[ht]

_Figure/Table caption: $ and lenses with $f=75\mathrm{mm}$ focal length.}


## S5. Higher-Order Exceptional Point CPAs

 It is possible to extend the concept depicted in Fig.~1(b) and Fig.~1(d) of the main paper by critically coupling more than two sub-cavities. For instance, Fig.~§fig:EPEP-CPA illustrates a single-mode EP-CPA composed of three cavities, designed to absorb single-mode light waves incident from the left. This configuration can be mathematically represented using the following transfer matrix:

$$
    \mathbf{M}_\mathrm{CPA}^\text{(3)} = \mathbf{M}_1 \mathbf{M}_\mathrm{4f} \mathbf{M}_\mathrm{c} \mathbf{M}_\mathrm{4f} \mathbf{M}_\mathrm{c}  \mathbf{M}_\mathrm{4f} \mathbf{M}_\mathrm{a} \mathbf{M}_2

$$

[ht]

_Figure/Table caption:

 Observe that in Fig.~§fig:EPEP-CPA the two coupling mirrors share the same reflectivity, denoted as $R_\mathrm{c}$. By applying the back-conversion $\mathbf{S}(\mathbf{M}_\mathrm{CPA}^\text{(3)})$ from Eq. (§eq:defSMconversion2), we can derive the scattering matrix $\mathbf{S}_\mathrm{CPA}^\text{(3)}$ for the entire CPA system. As before, our main focus remains on the top-left quadrant of $\mathbf{S}_\mathrm{CPA}^\text{(3)}$, which represents the CPA's reflection matrix $\mathbf{R}_\mathrm{CPA}^\text{(3)}$ from the left side.

 After applying the substitutions (§eq:diagsubst), replacing $**T**_\text{4f}^2=e^{2i\phi} \mathbb{1}$, and further employing the energy conservation relations $r_1^2 + t_1^2 = 1$, $r_2^2 + t_2^2 = 1$, $r_\mathrm{c}^2 + t_\mathrm{c}^2 = 1$, $t_1^2 = 1 - r_1^2$, and $t_\mathrm{c}^2 = 1 - r_\mathrm{c}^2$, and setting $r_2=1$, the matrix expression for $\mathbf{S}_\mathrm{CPA}^\text{(3)}$ can be reformulated as the following scalar equation:
\begin{eqnarray}
    r_\mathrm{CPA}^\text{(3)}=
    \frac{r_\mathrm{1}
    + r_\mathrm{c}  \left[r_\mathrm{1} \left(r_\mathrm{c}+t_\mathrm{a}^2\right)+1\right] e^{2 i \phi }
    + r_\mathrm{c}  \left[t_\mathrm{a}^2 (r_\mathrm{1}+r_\mathrm{c})+1\right] e^{4 i \phi }
    + t_\mathrm{a}^2 e^{6 i \phi }}
    {1
    + r_\mathrm{c}  \left(r_\mathrm{1}+r_\mathrm{c}+t_\mathrm{a}^2\right) e^{2 i \phi }
    + r_\mathrm{c} \left(r_\mathrm{1} r_\mathrm{c} t_\mathrm{a}^2+r_\mathrm{1}+t_\mathrm{a}^2\right) e^{4 i \phi }
    +r_\mathrm{1} t_\mathrm{a}^2 e^{6 i \phi }}
\end{eqnarray}

 Similarly to Section S3, we can deduce the conditions for the CPA by solving the following system of equations:
\begin{eqnarray}
\left. \vert r_\mathrm{CPA}^\text{(3)}  \vert^2 \right\vert_{\phi=\frac{\pi}{2}} &= 0 \\
tial_\phi^2 \left. \vert r_\mathrm{CPA}^\text{(3)}  \vert^2 \right\vert_{\phi=\frac{\pi}{2}} &= 0
\end{eqnarray}
After solving this system of equations, we obtain:
\begin{eqnarray}
&&t_\mathrm{a} = \sqrt{r_1}  \\
&&r_\mathrm{c} =
\frac{\sqrt{1+14 r_1^2 + r_1^4}-r_1^2 - 1}{2 r_1}
\end{eqnarray}
which can be expressed equivalently as
\begin{eqnarray}
&&T_\mathrm{a}  = \sqrt{R_1} \\
&&R_\mathrm{c}=
\frac{\big(\sqrt{1+14R_1+R_1^2}-R_1-1\big)^2}{4 R_1}

\end{eqnarray}

 Similarly, analytical expressions can be derived to describe the conditions for an EP-CPA consisting of four cavities, as illustrated in Fig.~§fig:EPEPEP-CPA.
[ht]

_Figure/Table caption: $, while the central coupling mirror has a distinct reflectivity $R_\mathrm{c2}$.}

 In this case, since the leftmost and rightmost coupling mirrors share the same reflectivity, denoted as $R_\mathrm{c1}$, while the central coupling mirror has a different reflectivity  $R_\mathrm{c2}$, we must define two distinct scattering matrices $\mathbf{S}_\mathrm{c1}$ and $\mathbf{S}_\mathrm{c2}$, from which we can derive the two corresponding transfer matrices $\mathbf{M}_\mathrm{c1}$ and $\mathbf{M}_\mathrm{c2}$. These will be used to compute the total transfer matrix of the system.

$$
    \mathbf{M}_\mathrm{CPA}^{(4)} = \mathbf{M}_1 \mathbf{M}_\mathrm{4f}
    \mathbf{M}_\mathrm{c1} \mathbf{M}_\mathrm{4f} \mathbf{M}_\mathrm{c2} \mathbf{M}_\mathrm{4f} \mathbf{M}_\mathrm{c1} \mathbf{M}_\mathrm{4f}
    \mathbf{M}_\mathrm{a} \mathbf{M}_2

$$

The scattering matrix $\mathbf{S}_\mathrm{CPA}^{(4)}$ for the entire CPA system can be obtained by using the back-conversion $\mathbf{S}(\mathbf{M}^{(4)})$ from Eq. (§eq:defSMconversion2). This matrix contains the CPA’s reflection matrix $\mathbf{R}_\mathrm{CPA}^{(4)}$ in the top-left quadrant. After making analogous substitutions as previously, we arrive at:

\begin{tiny}
\begin{eqnarray}
    r_\text{CPA}^{(4)} =
    \frac{r_\mathrm{1}
    +r_\mathrm{c1}  \left[r_\mathrm{1} \left(2 r_\mathrm{c2}+t_\mathrm{a}^2\right)+1\right]e^{2 i \phi }
    + \left[t_\mathrm{a}^2 \left(r_\mathrm{1} \left(r_\mathrm{c1}^2+1\right) r_\mathrm{c2}+r_\mathrm{c1}^2\right)+r_\mathrm{c1}^2 (r_\mathrm{1}+r_\mathrm{c2})+r_\mathrm{c2}\right] e^{4 i \phi }
    +r_\mathrm{c1}  \left[t_\mathrm{a}^2 (r_\mathrm{1}+2 r_\mathrm{c2})+1\right] e^{6 i \phi }
    +t_\mathrm{a}^2 e^{8 i \phi }}
    {1
    +r_\mathrm{c1}  \left(r_\mathrm{1}+2 r_\mathrm{c2}+t_\mathrm{a}^2\right) e^{2 i \phi }
    + \left[r_\mathrm{c1}^2 \left(t_\mathrm{a}^2 (r_\mathrm{1}+r_\mathrm{c2})+r_\mathrm{1} r_\mathrm{c2}+1\right)+r_\mathrm{c2} \left(r_\mathrm{1}+t_\mathrm{a}^2\right)\right] e^{4 i \phi }
    +r_\mathrm{c1}  \left(2 r_\mathrm{1} r_\mathrm{c2} t_\mathrm{a}^2+r_\mathrm{1}+t_\mathrm{a}^2\right) e^{6 i \phi }
    +r_\mathrm{1} t_\mathrm{a}^2 e^{8 i \phi }}
    \;\;\;\;
\end{eqnarray}
\end{tiny}

 With the addition of another cavity, the system of equations necessary to determine the CPA conditions now incorporates an additional equation:
\begin{eqnarray}
\left. \vert r_\mathrm{CPA}^{(4)}\vert^2 \right\vert_{\phi=\frac{\pi}{2}} &= 0 \\
tial_\phi^2 \left. \vert r_\mathrm{CPA}^{(4)}\vert^2 \right\vert_{\phi=\frac{\pi}{2}} &= 0 \\
tial_\phi^4 \left. \vert r_\mathrm{CPA}^{(4)}\vert^2 \right\vert_{\phi=\frac{\pi}{2}} &= 0
\end{eqnarray}

 While it is possible to express the exact solution analytically, doing so requires some effort:
\begin{eqnarray}
&t_\mathrm{a} = \sqrt{r_1} \\
&r_\mathrm{c1} =
    \frac{D}{2^{1/3} A}-
    \frac{2^{1/3} B}{A D}+\frac{4 \left(r_\mathrm{1}^5-r_\mathrm{1}\right)}{A}
\end{eqnarray}
with
\begin{footnotesize}
\begin{eqnarray}
A &=& 3 \left(r_\mathrm{1}^6-3 r_\mathrm{1}^4+3 r_\mathrm{1}^2-1  \right) \nonumber  \\
B &=& 3 r_\mathrm{1}^{12}+14 r_\mathrm{1}^{10}-147 r_\mathrm{1}^8+260 r_\mathrm{1}^6-147 r_\mathrm{1}^4+14 r_\mathrm{1}^2+3 \nonumber \\
C &=& 72 r_\mathrm{1}^{17}-880 r_\mathrm{1}^{15}+3312 r_\mathrm{1}^{13}-4272 r_\mathrm{1}^{11}+4272 r_\mathrm{1}^7-3312 r_\mathrm{1}^5+880 \nonumber
r_\mathrm{1}^3-72 r_\mathrm{1} \\
D &=& \left(C+\sqrt{4B^3+C^2}\right)^{1/3}\nonumber
\end{eqnarray}
\end{footnotesize}

 And, finally:
\begin{eqnarray}
&r_\mathrm{c2} =
\frac{
D \left(C^5 I + J K\right)
+8 C^6 F r_\mathrm{1}  \left(r_\mathrm{1}^2+1\right)
-2 D^{2} H \left[3 J+4 C^3 G r_\mathrm{1} \left(r_\mathrm{1}^2+1\right) \right]
-\sqrt{3} \sqrt{B} C^7 E
}
{6 D^{5} r_\mathrm{1} \left(r_\mathrm{1}^4-1\right)}
\end{eqnarray}
with
\begin{footnotesize}
\begin{eqnarray}
A &=& 36 r_1^{17}-440 r_1^{15}+1656 r_1^{13}-2136 r_1^{11}+2136 r_1^7-1656 r_1^5+440 r_1^3-36 r_1 \nonumber \\
B &=& r_1^8+76 r_1^6-282 r_1^4+76 r_1^2+1 \nonumber \\
C &=& r_1^2-1 \nonumber \\
D &=& \left(A-3 \sqrt{3} \sqrt{B} C^7\right)^{1/3} \nonumber \\
E &=& \left( 9 r_1^{16}-168 r_1^{14}+2108 r_1^{12}-6040 r_1^{10}+4086 r_1^8-6040 r_1^6+2108 r_1^4-168 r_1^2+9\right) (r_\mathrm{1}-1) (r_\mathrm{1}+1) \nonumber  \\
F &=& 9 r_1^{20}-462 r_1^{18}+5045 r_1^{16}-32232 r_1^{14}+115266 r_1^{12}-191636 r_1^{10}+115266 r_1^8-32232 r_1^6+5045 r_1^4-462 r_1^2+9 \nonumber \\
G &=& 9 r_1^8-92 r_1^6+230 r_1^4-92 r_1^2+9 \nonumber  \\
H &=& r_1^6-5 r_1^4+5 r_1^2-1 \nonumber \\
I &=& 9 r_1^{20}+1206 r_1^{18}-13547 r_1^{16}+49032 r_1^{14}-56862 r_1^{12}+7556 r_1^{10}-56862 r_1^8+49032 r_1^6-13547 r_1^4+1206 r_1^2+9 \nonumber \\
J &=& \sqrt{3} \sqrt{B C^{14}} \nonumber \\
K &=&   84 r_\mathrm{1}^{11} -572  r_\mathrm{1}^9 +744  r_\mathrm{1}^7 +744  r_\mathrm{1}^5 -572 r_\mathrm{1}^3 +84  r_\mathrm{1}   \nonumber
\end{eqnarray}
\end{footnotesize}

 By performing a series expansion around $r_1=1$, we can approximate the expression for $r_\mathrm{c1}$ with the following simplified formula, which maintains good fidelity for all values\linebreak $0.5 < r_1 < 1$:
\begin{eqnarray}
r_\mathrm{c1} \approx 1 + \frac{1-\sqrt{2}}{2}\left( 2 - 5 r_1 + 4 r_1^2 - r_1^3\right) + \frac{5-4\sqrt{2}}{2} (r_1 - 1)^4 -\frac{4-3\sqrt{2}}{2} (r_1 - 1)^5
\end{eqnarray}

 Furthermore, by employing numerical parameter fitting of a fit-function to the exact solution, we can derive an approximate formula for $r_\mathrm{c2}$ that is suitable for all values $0.5 < r_1 < 1$:
\begin{eqnarray}
r_\mathrm{c2} \approx 1 - 0.0984341 (1 - r_1)^2 + 0.0197372 (1 - r_1)^3 - 0.327913 (1 - r_1)^4
\end{eqnarray}
For an EP-CPA that incorporates five or more cavities, it is no longer possible to represent the exact solutions analytically.

[h]

_Figure/Table caption:

 When generalizing the concept illustrated in Fig.~§fig:EPEPEP-CPA by coupling $n>4$ cavities, $m=n-1$ coupling mirrors with reflection coefficients $R_\mathrm{c1}, R_\mathrm{c2}, ..., R_\mathrm{cm}$ are required. Importantly, for spectral broadening to occur, symmetrical pairs of internal coupling mirrors must have identical reflection coefficients, such that $R_\mathrm{c1} = R_\mathrm{cm}, R_\mathrm{c2} = R_\mathrm{c(m-1)}, $, and so on. However, as the number of coupled cavities in an EP-CPA increases, the reflection spectrum around the resonance frequencies starts to resemble a Butterworth bandpass filter, as known from electrical filter theory. This observation can be supported by plotting the reflectance spectra $R(\Delta \lambda)$ in Fig.~§fig:MCCPAplot as a function of a phase $\phi$ by substituting $\Delta \lambda \rightarrow 2 \pi L / \phi$, and subsequently comparing $R(\phi)$ with $R = 1-\mathrm{b}(\phi)$, where
\begin{eqnarray}
&&\mathrm{b}(\phi)  = \frac{1}{1+\left(\frac{\phi}{\phi_{NN}}\right)^{2n}}
\end{eqnarray}
denotes the conventional Butterworth equation, as recognized in electrical theory, $\phi_{NN}$ is the half-maximum-reflection phase-shift of the corresponding WLC-CPA, and $n$ is the number of coupled cavities. This is consistent with the known fact, as mentioned in [4], that the Butterworth-function can also describe the transmission behavior of a multimirror Fabry-Perot interferometer without a dissipating element, which is a configuration related to the EP-CPAs under discussion. The crucial takeaway from this analysis is that, unless the objective is to construct a Butterworth-like band-absorption-filter, augmenting the number of cavities beyond three does not substantially enhance the broadband absorption behavior

## S6. Deriving the CPA conditions for the alternative EP-CPA design and Quartic Behavior Proof

 In this section, we will outline the procedure to derive CPA conditions for the alternative EP-CPA Design~B illustrated in Fig.~3(a) of the main text and prove the quartic beavior. The calculation will be based on the beam-path depicted in this figure. In our previous calculations, to determine the effect of an optical element (e.g., a mirror), it was sufficient to consider the amplitude vectors of the left-incident light field $\mathbf{a}^{\mathrm{left}}_{\mathrm{in}}$ and the right-incident light field $\mathbf{a}^{\mathrm{right}}_{\mathrm{in}}$. We then calculated the corresponding outgoing amplitude vectors $\mathbf{a}^{\mathrm{left}}_{\mathrm{out}}$ and $\mathbf{a}^{\mathrm{right}}_{\mathrm{out}}$ using a $4 \times 4$ scattering block-matrix, as shown in Eq.~(§eq:smatrix). However, the scenario  becomes more intricate for the case at hand. It necessitates considering for each optical element *two* distinct fields of light incident from the left and right, and calculating *two* separate outgoing fields for the left and right (i.e., their corresponding amplitude vectors). This is accomplished by expanding the amplitude vectors $\mathbf{a}^{\mathrm{left}}_{\mathrm{in}}$, $\mathbf{a}^{\mathrm{right}}_{\mathrm{in}}$, $\mathbf{a}^{\mathrm{left}}_{\mathrm{out}}$, and $\mathbf{a}^{\mathrm{right}}_{\mathrm{out}}$ to to have two components each, where each component is itself an amplitude vector. These components  will represent the amplitudes of the light field above the optical axis (first vector element with index $1$), and the amplitudes of the light field below the optical axis (second vector element with index $2$):

$$
    \mathbf{a}^{\mathrm{left}}_{\mathrm{in}} = \begin{pmatrix} {\mathbf{a}_1}^{\mathrm{left}}_{\mathrm{in}} \\ {\mathbf{a}_2}^{\mathrm{left}}_{\mathrm{in}} \end{pmatrix} \;\;\;
    \mathbf{a}^{\mathrm{right}}_{\mathrm{in}} = \begin{pmatrix} {\mathbf{a}_1}^{\mathrm{right}}_{\mathrm{in}} \\ {\mathbf{a}_2}^{\mathrm{right}}_{\mathrm{in}}
    \end{pmatrix} \;\;\;
     \mathbf{a}^{\mathrm{left}}_{\mathrm{out}} = \begin{pmatrix} {\mathbf{a}_1}^{\mathrm{left}}_{\mathrm{out}} \\ {\mathbf{a}_2}^{\mathrm{left}}_{\mathrm{out}} \end{pmatrix} \;\;\;
    \mathbf{a}^{\mathrm{right}}_{\mathrm{out}} = \begin{pmatrix} {\mathbf{a}_1}^{\mathrm{right}}_{\mathrm{out}} \\ {\mathbf{a}_2}^{\mathrm{right}}_{\mathrm{out}}
    \end{pmatrix}


$$

Consequently, the scattering matrices we will be working with have evolved into $2\times 2$ block matrices, where each element is itself a $2\times 2$ block matrix, transforming them into $4\times 4$  block matrices:

$$
    \left( \begin{array}{ll} \mathbf{a}^{\mathrm{left}}_{\mathrm{out}} \\
    \mathbf{a}^{\mathrm{right}}_{\mathrm{out}}  \end{array} \right) =
    \mathbf{S}
    \left( \begin{array}{ll} \mathbf{a}^{\mathrm{left}}_{\mathrm{in}} \\
    \mathbf{a}^{\mathrm{right}}_{\mathrm{in}}  \end{array} \right) =
    \left( \begin{array}{cc} \mathbf{R}_\mathrm{L} & \mathbf{T}_\mathrm{RL} \\ \mathbf{T}_\mathrm{LR} & \mathbf{R}_\mathrm{R} \end{array} \right)
    \left( \begin{array}{ll} \mathbf{a}^{\mathrm{left}}_{\mathrm{in}} \\
    \mathbf{a}^{\mathrm{right}}_{\mathrm{in}}  \end{array} \right)=
    \left( \begin{array}{cccc}
    \mathbf{R}_\mathrm{L}^{11} & \mathbf{R}_\mathrm{L}^{12} & \mathbf{T}_\mathrm{RL}^{11} & \mathbf{T}_\mathrm{RL}^{12} \\
    \mathbf{R}_\mathrm{L}^{21} & \mathbf{R}_\mathrm{L}^{22} & \mathbf{T}_\mathrm{RL}^{21} & \mathbf{T}_\mathrm{RL}^{22} \\
    \mathbf{T}_\mathrm{LR}^{11} & \mathbf{T}_\mathrm{LR}^{12} & \mathbf{R}_\mathrm{R}^{11} & \mathbf{R}_\mathrm{R}^{12} \\
   \mathbf{T}_\mathrm{LR}^{21} & \mathbf{T}_\mathrm{LR}^{22} & \mathbf{R}_\mathrm{R}^{21} & \mathbf{R}_\mathrm{R}^{22}
    \end{array} \right)
    \left( \begin{array}{ll} \mathbf{a}^{\mathrm{left}}_{\mathrm{in}} \\
    \mathbf{a}^{\mathrm{right}}_{\mathrm{in}}  \end{array} \right)


$$

Although we aim to maintain the convention established in Eq.~(§eq:defS2), it is now necessary to reformulate it for the $4 \times 4$ scattering block matrix:
\begin{eqnarray}
\mathbf{S}=\begin{pmatrix}
\mathbf{R}_\mathrm{L} & \mathbf{T}_\mathrm{RL} \\
 \mathbf{T}_\mathrm{LR} & \mathbf{R}_\mathrm{R}
\end{pmatrix} =
\begin{pmatrix}
\mathbf{R} & \mathbf{T} \\ \mathbf{T} & -\mathbf{R}
\end{pmatrix} =
\begin{pmatrix}
\mathbf{R}_{11} & \mathbf{R}_{12} & \mathbf{T}_{11} & \mathbf{T}_{12} \\
\mathbf{R}_{21} & \mathbf{R}_{22} & \mathbf{T}_{21} & \mathbf{T}_{22} \\
\mathbf{T}_{11} & \mathbf{T}_{12} & -\mathbf{R}_{11} & -\mathbf{R}_{12} \\
\mathbf{T}_{21} & \mathbf{T}_{22} & -\mathbf{R}_{21} & -\mathbf{R}_{22}
\end{pmatrix}
\;\;\;  \mathbf{R}_{ij}, \mathbf{T}_{ij} \in \mathbb{R}^{n \times n}
\end{eqnarray}
Given these modifications, we are now in a position to write down the $4 \times 4$ scattering block matrix $\mathbf{S}_\mathrm{ML}$ for the left ``split'' mirror:
\begin{eqnarray}
\mathbf{S}_\mathrm{ML}=
\begin{pmatrix}
\mathbf{R}_\mathrm{ML1} & \mathbb{0} & \mathbf{T}_\mathrm{ML1} & \mathbb{0} \\
\mathbb{0} & \mathbf{R}_\mathrm{ML2} & \mathbb{0} & \mathbf{T}_\mathrm{ML2} \\
\mathbf{T}_\mathrm{ML1} & \mathbb{0} & -\mathbf{R}_\mathrm{ML1} & \mathbb{0} \\
\mathbb{0} & \mathbf{T}_\mathrm{ML2} & \mathbb{0} & -\mathbf{R}_\mathrm{ML2}
\end{pmatrix}
\end{eqnarray}
It is important to note that $\mathbf{S}_\mathrm{ML}$ includes separate reflection- and transmission matrices for the ``upper'' and the ``lower'' light fields. In contrast, we do not need to distinguish between these two spatially separated light fields in the scattering matrix for the central mirror $\mathbf{S}_\mathrm{MC}$ and the right mirror $\mathbf{S}_\mathrm{MR}$, since each of these mirrors
 reflects and transmits in the same way above and below the optical axis:
\begin{eqnarray}
\mathbf{S}_\mathrm{MC}=
\begin{pmatrix}
\mathbf{R}_\mathrm{MC} & \mathbb{0} & \mathbf{T}_\mathrm{MC} & \mathbb{0} \\
\mathbb{0} & \mathbf{R}_\mathrm{MC} & \mathbb{0} & \mathbf{T}_\mathrm{MC} \\
\mathbf{T}_\mathrm{MC} & \mathbb{0} & -\mathbf{R}_\mathrm{MC} & \mathbb{0} \\
\mathbb{0} & \mathbf{T}_\mathrm{MC} & \mathbb{0} & -\mathbf{R}_\mathrm{MC}
\end{pmatrix}
\;\;\;\;\;
\mathbf{S}_\mathrm{MR}=
\begin{pmatrix}
\mathbf{R}_\mathrm{MR} & \mathbb{0} & \mathbf{T}_\mathrm{MR} & \mathbb{0} \\
\mathbb{0} & \mathbf{R}_\mathrm{MR} & \mathbb{0} & \mathbf{T}_\mathrm{MR} \\
\mathbf{T}_\mathrm{MR} & \mathbb{0} & -\mathbf{R}_\mathrm{MR} & \mathbb{0} \\
\mathbb{0} & \mathbf{T}_\mathrm{MR} & \mathbb{0} & -\mathbf{R}_\mathrm{MR}
\end{pmatrix}
\end{eqnarray}
Given that the absorber only affects the amplitude of the light field below the optical axis, characterized by a transmission matrix $\mathbf{T}_\mathrm{a}$, we can express the corresponding $4 \times 4$ scattering block matrix as follows:
\begin{eqnarray}
\mathbf{S}_\mathrm{a}=
\begin{pmatrix}
\mathbb{0} & \mathbb{0} & \mathbb{1} & \mathbb{0} \\
\mathbb{0} & \mathbb{0} & \mathbb{0} & \mathbf{T}_\mathrm{a} \\
\mathbb{1} & \mathbb{0} & \mathbb{0} & \mathbb{0} \\
\mathbb{0} & \mathbf{T}_\mathrm{a} & \mathbb{0} & \mathbb{0}
\end{pmatrix}
\end{eqnarray}
 Let $**T**_\text{2f}$ represent the transmission matrix for propagation between the focal points of a single lens (hence, representing a Fourier transformation which also applies a uniform phase $e^{i \phi}$ to all modes). Then, the free-space propagation between each optical element, spanning a distance of one focal length $f$, can be represented by the following  $4 \times 4$ scattering block matrix.
\begin{eqnarray}
\mathbf{S}_\mathrm{2f}=
\begin{pmatrix}
\mathbb{0} & \mathbb{0} & \mathbf{T}_\mathrm{2f} & \mathbb{0} \\
\mathbb{0} & \mathbb{0} & \mathbb{0} & \mathbf{T}_\mathrm{2f} \\
\mathbf{T}_\mathrm{2f} & \mathbb{0} & \mathbb{0} & \mathbb{0} \\
\mathbb{0} & \mathbf{T}_\mathrm{2f} & \mathbb{0} & \mathbb{0}
\end{pmatrix}
\end{eqnarray}
The $4 \times 4$ scattering block matrix $\mathbf{S}_\mathrm{LL}$ representing the left lens must represent the behavior demonstrated by the exemplary beam path in  Fig.~4(a) of the main paper, indicating that the left incident light field shifts from above to below the optical axis upon reaching the central mirror, and vice versa. Consequently, the lower-left $2 \times 2$ sub-block-matrix, which represents the left-to-right-transmission, is a block matrix with identity matrices in the counter-diagonal elements. In contrast, the right-to-left propagating light field does not change sides with respect to the optical axis. Therefore, the top-right $2 \times 2$ sub-block-matrix, representing the right-to-left transmission, is a block matrix with identity matrices in the main-diagonal elements, making it an identity matrix:
\begin{eqnarray}
\mathbf{S}_\mathrm{LL}=
\begin{pmatrix}
\mathbb{0} & \mathbb{0} & \mathbb{1} & \mathbb{0} \\
\mathbb{0} & \mathbb{0} & \mathbb{0} & \mathbb{1} \\
\mathbb{0} & \mathbb{1} & \mathbb{0} & \mathbb{0} \\
\mathbb{1} & \mathbb{0} & \mathbb{0} & \mathbb{0}
\end{pmatrix}
\end{eqnarray}
In the case of the right lens, the situation is reversed: the left-to-right propagating light field does not change sides, while the right-to-left propagating light field shifts from above to below the optical axis upon reaching the central mirror (and vice versa). As a result, the $4 \times 4$ scattering block matrix $\mathbf{S}_\mathrm{LR}$ for the right lens is structured accordingly:
\begin{eqnarray}
\mathbf{S}_\mathrm{LR}=
\begin{pmatrix}
\mathbb{0} & \mathbb{0} & \mathbb{0} & \mathbb{1} \\
\mathbb{0} & \mathbb{0} & \mathbb{1} & \mathbb{0} \\
\mathbb{1} & \mathbb{0} & \mathbb{0} & \mathbb{0} \\
\mathbb{0} & \mathbb{1} & \mathbb{0} & \mathbb{0}
\end{pmatrix}
\end{eqnarray}
To convert the previously defined scattering matrices into their corresponding transfer matrices with the goal to compute the total transfer matrix $\mathbf{M}_\mathrm{CPA}$ for the entire CPA, the conversion function $\mathbf{M}(\mathbf{S})$, as defined in  (§eq:defSMconversion2), can be utilized once more. This time, however, the multiplication rule for $2\times 2$ block-matrices, as detailed in (§eq:BMmult2), is required to apply the conversion functions $\mathbf{M}(\mathbf{S})$ and $\mathbf{S}(\mathbf{M})$ from Eqs. (§eq:defSMconversion2). Additionally, calculating the inverse of $2 \times 2$ block matrices becomes essential. Following [5,~p.~184] and [6], we do this as follows: Let $\mathbf{M}$ be a $2 \times 2$ block matrix with the components $\mathbf{M}_{11}=\mathbf{A}$, $\mathbf{M}_{12}=\mathbf{B}$, $\mathbf{M}_{21}=\mathbf{C}$, and $\mathbf{M}_{22}=\mathbf{D}$, each being matrices themselves. Then

$$
\begin{aligned}
    \mathbf{M}^{-1}=
    \begin{cases}
    \begin{pmatrix}
    \mathbb{0} & \mathbf{C}^{-1} \\
     \mathbf{B}^{-1} & \mathbb{0}
    \end{pmatrix}, &\text{for} \; \mathbf{A}=\mathbb{0} \wedge  \mathbf{D}=\mathbb{0} \\
    \noalign{\vskip9pt}
    \begin{pmatrix}
    \mathbf{A}^{-1} & \mathbb{0} \\
    \mathbb{0} &  \mathbf{D}^{-1}
    \end{pmatrix}, &\text{for} \; \mathbf{B}=\mathbb{0} \wedge  \mathbf{C}=\mathbb{0} \\
    \noalign{\vskip9pt}
    \begin{pmatrix}
    (\mathbf{A}-\mathbf{B} \mathbf{D}^{-1} \mathbf{C})^{-1} &
    -\mathbf{A}^{-1} \mathbf{B} ( \mathbf{D} - \mathbf{C} \mathbf{A}^{-1} \mathbf{B})^{-1} \\
    -\mathbf{D}^{-1} \mathbf{C}
    (\mathbf{A}-\mathbf{B}\mathbf{D}^{-1} \mathbf{C})^{-1}&
    (\mathbf{D} - \mathbf{C} \mathbf{A}^{-1}
    \mathbf{B})^{-1}
    \end{pmatrix}, &\text{for} \;
    \mathbf{A}, \mathbf{D} \; \text{nonsingular}
    \end{cases}

\end{aligned}
$$

 After converting all $4 \times 4$ scattering block matrices into the corresponding $4 \times 4$ transfer block matrices using the $\mathbf{M}(\mathbf{S})$ function, we can succinctly express the transfer matrix $\mathbf{M}_\mathrm{CPA}^\text{(B)}$ for the entire CPA (Design~B):
\begin{eqnarray}
\mathbf{M}_\mathrm{CPA}^\text{(B)}=
\mathbf{M}_\mathrm{ML} \mathbf{M}_\mathrm{a} \mathbf{M}_\mathrm{LL}
\mathbf{M}_\mathrm{2f} \mathbf{M}_\mathrm{MC}  \mathbf{M}_\mathrm{LR}  \mathbf{M}_\mathrm{2f} \mathbf{M}_\mathrm{MR}

\end{eqnarray}
Given that all the $\mathbf{M}$-matrices in Eq.~(§eq:MCPA2) are now $4 \times 4$ block matrices, where each element is a distinct matrix in its own right, the following multiplication rule for $4 \times 4$ block matrices, $\mathbf{X}$ and $\mathbf{Y}$, must be applied during each multiplication for every matrix element in the calculation of expression (§eq:MCPA2):

$$
    \left( \mathbf{X} \mathbf{Y}\right)_{ij} = \sum_{k=1}^4{\mathbf{X}_{ik} \mathbf{Y}_{kj}}


$$

The corresponding scattering matrix can then be obtained from $\mathbf{M}_\mathrm{CPA}^\text{(B)}$ by utilizing the $\mathbf{S}(\mathbf{M})$ back-conversion function, again taking into account the $2 \times 2$ block matrix multiplication rule (§eq:BMmult2) and the the $2 \times 2$ block matrix inversion method (§eq:block_matrix_inverse) when calculating
\begin{eqnarray}
\mathbf{S}_\mathrm{CPA}^\text{(B)} = \mathbf{S}(\mathbf{M}_\mathrm{CPA}^\text{(B)}).

\end{eqnarray}
Our interest lies in examining the CPA's behavior with respect to reflections on its left side. Hence, we focus on the top-left quadrant $2 \times 2$ block matrix of  $\mathbf{S}_\mathrm{CPA}^\text{(B)}$. In particular, we are concerned with the CPA’s reflection behavior for *incident light above the optical axis*. Consequently, we must extract the top-left entry of this sub-matrix, which also coincides with the top-left entry of the entire scattering matrix $\mathbf{S}_\mathrm{CPA}^\text{(B)}$. This entry contains the reflection matrix $\mathbf{R}_\mathrm{CPA}^\text{(B)}$ that we are interested in. Assuming again that the same reflection and transmission coefficients apply to all modes at the mirrors and the absorber, then all matrices except $**T**_\text{f}$ become diagonal matrices:

$$
\begin{aligned}
    **R**_\text{ML1} = r_{1a} \mathbb{1}, \;\;
    **R**_\text{ML2} = r_{1b} \mathbb{1}, \;\;
    &**T**_\text{ML1} = t_{1a} \mathbb{1}, \;\;
    **T**_\text{ML2} = r_{1b} \mathbb{1} \\
    **R**_\text{MC} = r_\text{c} \mathbb{1}, \;\;
    **T**_\text{MC} = t_\text{c} \mathbb{1}, \;\;
    **R**_\text{MR} &= r_2 \mathbb{1}, \;\;
    **T**_\text{MR} = t_2 \mathbb{1}, \;\;
    **T**_\text{a} = t_\text{a} \mathbb{1}
    \end{aligned}


$$

 After implementing the aforementioned substitutions, and further applying the energy conservation substitutions $r_\mathrm{1a}^2+t_\mathrm{1a}^2 = 1$, $r_\mathrm{1b}^2+t_\mathrm{1b}^2 = 1$, $r_\mathrm{c}^2+t_\mathrm{c}^2 = 1$, and $r_2^2+t_2^2 = 1$, then setting $r_\text{1b}=1$ and $r_2=1$, and finally substituting $t_\text{c}^2 = 1 - r_\text{c}^2$ and $t_\text{1a}^2 = 1 - r_\text{1a}^2$, a  simplified matrix equation for $**R**_\text{CPA}^\text{(B)}$ can be derived:

$$
\begin{aligned}
    **R**_\text{CPA}^\text{(B)} = \frac{t_\text{a}^2 \, **T**_\text{2f}^{8} + [ 1 + r_\text{1a} t_\text{a}^2  - r_\text{c}^2(t_\text{a}^2+1)(r_\text{1a}^2+1) ] \, **T**_\text{2f}^4  + r_\text{1a} \mathbb{1} }{r_\text{1a} t_\text{a}^2 \, **T**_\text{2f}^{8} + [r_\text{1a} + t_\text{a}^2  - r_\text{c}^2(t_\text{a}^2+1)(r_\text{1a}^2+1) ] \, **T**_\text{2f}^{4} + \mathbb{1}}

\end{aligned}
$$

This correponds to Eq.~(3) in the main paper. Observe how the structure of Eq.~(§eq:RLa) resembles the structure of Eq.~(§eq:RCPA_simplified2). Because $**T**_\text{2f}^4 = **T**_\text{4f}^2$, the matrix $**T**_\text{2f}^4$ in Eq. (§eq:RLa) results in a quadruple Fourier transformation with a uniform phase shift $e^{2i\phi}$ for all modes, and can thus be represented as a simple diagonal matrix $**T**_\text{4f}^2=e^{2i\phi} \mathbb{1}$. Analogously, $**T**_\text{2f}^{8}=e^{4i\phi} \mathbb{1}$. Consequently, $ **R**_\text{CPA}^\text{(B)}$ becomes a diagonal matrix $ **R**_\text{CPA}^\text{(B)} = r_\text{CPA}^\text{(B)} \mathbb{1}$, and can therefore be reduced to a simple scalar equation, which proves the spatial degeneracy of Design~B.
\begin{eqnarray}
r_\mathrm{CPA}^\text{(B)}=
\frac{t_\text{a}^2 \, e^{4 i \phi} + [ 1 + r_\text{1a} t_\text{a}^2  - r_\text{c}^2(t_\text{a}^2+1)(r_\text{1a}^2+1) ] \, e^{2 i \phi}  + r_\text{1a} }{r_\text{1a} t_\text{a}^2 \, e^{4 i \phi} + [r_\text{1a} + t_\text{a}^2  - r_\text{c}^2(t_\text{a}^2+1)(r_\text{1a}^2+1) ] \, e^{2 i \phi} + 1}
\end{eqnarray}

[!ht]

_Figure/Table caption: ^\text{(B)}$ with $r_1=0.7$ and some arbitary values for $r_\mathrm{c}$ and $ t_\mathrm{a}$ shows that we can expect to find perfect absorption at $\phi=\frac{\pi}{2}$, provided we find the right values for $r_\text{c}$ and $t_\text{a}$.}

 Plotting $R_\mathrm{CPA}^\text{(B)}(\phi)= \vert r_\mathrm{CPA}^\text{(B)} \vert^2$ (Fig.~§fig:deriv_48_CPA), we observe that perfect absorption at $\phi=\frac{\pi}{2}$ seems to be possible, contingent on our ability to identify the correct parameter values.
 Following the same reasoning as in section S3, we establish the subsequent system of equations:

$$
\begin{aligned}
\left. R_\mathrm{CPA}^\text{(B)}\left(\phi\right)\right\vert_{\phi=\frac{\pi}{2}} &= 0 \\
tial_\phi^2 \left. R_\mathrm{CPA}^\text{(B)}\left(\phi\right)\right\vert_{\phi=\frac{\pi}{2}} &= 0
\end{aligned}
$$

Solving this set of equations yields the following results:

$$
\begin{aligned}
t_\mathrm{a} &=\sqrt{r_\mathrm{ML1}} \\
r_\mathrm{c} &= \frac{1-r_\mathrm{ML1}}{1+r_\mathrm{ML1}}
\end{aligned}
$$

Squaring both sides of Eqs. (§eq:ta48) and (§eq:rc48), and substituting $t_\mathrm{a}^2 = T_\mathrm{a}$, $r_\mathrm{c}^2 = R_\mathrm{c}$, $r_\mathrm{ML1} = \sqrt{R_\mathrm{1a}}$ results in the CPA conditions for the alternative design as presented in the main paper:

$$
\begin{aligned}
T_\mathrm{a} = \sqrt{R_\mathrm{1a}}
\;\; &\Leftrightarrow \;\;
R_\mathrm{1a} = T_\mathrm{a}^2
\\
R_\mathrm{c} = \left( \frac{1-\sqrt{R_\mathrm{1a}}}{1+\sqrt{R_\mathrm{1a}}} \right)^2
\;\; &\Leftrightarrow \;\;
R_\mathrm{c} = \left( \frac{1-T_\mathrm{a}}{1+T_\mathrm{a}} \right)^2

\\
T_\mathrm{c} = 1- R_\mathrm{c}
\;\; &\Rightarrow \;\;
T_\mathrm{c} =
\frac{4 \sqrt{R_\mathrm{1a}}}{(1+\sqrt{R_\mathrm{1a}})^2} =
\frac{4 T_\mathrm{a}}{(1+T_\mathrm{a})^2}

\end{aligned}
$$

Note, how equation (§eq:Tc48) resembles equation (§eq:Rc8), and hence how the *transmissivity* of the central mirror in the alternative design is calculated similarly to the *reflectivity* of the central mirror in the original design.\\

 By inserting conditions (§eq:ta48) and  (§eq:rc48) into Eq.~(§eq:rcpaB) and substituting $r_\text{1a} = \sqrt{R_\text{1a}}$, we obtain the following expression:
\begin{footnotesize}

$$
    R_\text{CPA}^\text{(B)} =
    \frac{8 R_\text{1a} \cos^4(\phi)}{1 - 4 R_\text{1a}^{1/2} + 9 R_\text{1a} - 4 R_\text{1a}^{3/2} + R_\text{1a}^2 + \left( 1 - 4 R_\text{1a}^{1/2} + 2 R_\text{1a} - 4 R_\text{1a}^{3/2} + R_\text{1a}^2\right) \cos(2 \phi) + R_\text{1a}\cos(4 \phi) }

$$

\end{footnotesize}

 Note the similarity in the structure between expressions (§eq:RCPABopt) and (§eq:RCPAAopt). Expanding Eq.~(§eq:RCPABopt) around the resonance points $\phi_m = \frac{2m+1}{2} \pi$ reveals the anticipated quartic behavior of $R_\text{CPA}^\text{(B)}$:

$$
    R_\mathrm{CPA}^\text{(B)}(\phi) = \frac{4 R_\text{1a}}{(\sqrt{R_\text{1a}}-1)^4} \left(\phi - \frac{2m+1}{2} \pi \right)^4 + \mathcal{O}^6\left(\phi - \frac{2m+1}{2} \pi \right)^6

$$

## S7. Comparison of Design A and Design B

[!ht]

_Figure/Table caption:

 To approach the functionality of Design~B intuitively, it is instructive to compare its round-trip-paths with those in Design~A, and to account for each lossy reflection or transmission. This method allows us to exclude reflections at perfect mirrors from our consideration. In Design~A, light can take self-imaging round-trips in three ways: (i)~In the right sub-cavity, between the central mirror $R_{\mathrm{c}}$ and the right mirror $R_2$. Each round-trip involves one reflection at $R_{\mathrm{c}}$ and two passes through the absorber $T_\text{a}$. The equivalent in Design~B (where there is also an interaction with the central mirror and the absorber) is the circular path between bottom-left (mirror $R_{1\mathrm{b}}$) and top-right (mirror $R_2$), where light passes $T_\text{a}$ twice (as before), but also passes the central mirror twice. (ii)~Design~A also allows for round-trips in the left sub-cavity between $R_1$ and $R_\text{c}$. Here, each round-trip includes one reflection at $R_1$ and one reflection at $R_\text{c}$. The corresponding path in Design~B (where there is also an interaction with the input-coupling mirror and the central mirror) is the circular path between top-left (mirror $R_{1\mathrm{a}}$) and bottom-right (mirror $R_2$), featuring one reflection at $R_{1\text{a}}$ (as before), and again two passes through the central mirror per round-trip. Both round-trip-paths paths (i) and (ii) have in common that one single reflection at the central mirror in Design~A corresponds to two transmission through the central mirror in Design~B (see Fig.~§fig:comparison_1)

[!ht]

_Figure/Table caption:  through the central mirror in Design~A correspond to *two reflections and two transmissions* at the central mirror in Design~B.}

 (iii) In Design~A, light also propagates between the farthest mirrors $R_1$ and $R_2$, getting reflected once at $R_1$, passing twice through the central mirror and twice through $T_\text{a}$. Design~B replicates this with a path following $R_{1\text{a}}$ -- $R_{\text{c}}$~-- $R_2$ (bottom) -- $R_\text{c}$ -- $R_2$ (top) -- $R_\text{c}$ -- $T_\text{a}$~-- $R_{1\text{b}}$~-- $T_\text{a}$ -- $R_{\text{c}}$ -- $R_{1\text{a}}$ (see Fig.~§fig:comparison_2). In this path (where the light also interacts with the input-coupling mirror, the central mirror, and the absorber) each round-trip involves one single reflection at  $R_{1\text{a}}$, two passes through $T_\text{a}$, two transmissions through the central mirror (in the bottom part), and two reflections at the central mirror (at the top part). In cases (i) and (ii), a *single* reflection at the central mirror in Design~A is replaced by two transmissions through the central mirror in Design~B. This suggests that setting the transmissivity $T_\text{c}$ of the central mirror in Design~B equal to the square-root of the reflectivity $R_\text{c}$ in Design~A would establish an equivalence between the two designs. However, case (iii) complicates the matter: Here, *two* transmissions in Design~A correspond to *two transmissions and two reflections* in Design~B. At first glance, this appears to render Designs~A and B inequivalent. However, it is crucial to recognize that such comparisons between Design~A and Design~B offer, at best, a qualitative analogy. A comprehensive comparison requires the sound mathematical analysis from section S6, considering the infinite sum of all possible round trips and the effects of coupling between the respective sub-cavities. Nonetheless, this qualitative comparison indicates that the condition for the *transmissivity* $T_\text{c}$ of the central mirror in Design~B may resemble the condition for the *reflectivity* $R_\text{c}$ of the central mirror in Design~A, which is perfectly in accordance with the  result of the detailed mathematical analysis in section S6.

 Finally, it is important to note that positioning the absorber next to mirror $R_\mathrm{1b}$ in Design~B is not the only viable option. Effective EP configurations can also be achieved by placing the absorber in the right sub-cavity, adjacent to the central coupling mirror  $R_\mathrm{c}$ or next to to mirror $R_2$, either above or below the optical axis.

## S8. Compensation of refraction aberrations

 This section delineates the methodology for compensating the refraction aberrations induced by the weak absorber placed between the rightmost lens and the rightmost mirror $R_2$ in Design~A, as illustrated  in Fig.~1(d) of the main paper, or adjacent to Mirror $R_{1\text{b}}$ in Design~B, as illustrated  in Fig.~4(a) of the main paper. We postulate that this absorber possesses a specific thickness, denoted as $d$, and  its refractive index's real component is represented by $n_r$. In the ensuing section, we detail the refraction compensation based on Design A. However, the findings are equally applicable to Design B.\\

 When the absorber is introduced without additional adjustments, it effectively shifts the focal point of the rightmost lens by a distance $\Delta f$ towards the right, a phenomenon graphically represented in  Fig.~§fig:coverslip_correction. Neglecting this effect leads to the spatial degeneracy of the MAD-EP-CPA being compromized. \\

 The preliminary step in formulating a compensation strategy for this diffractive aberration is to determine the precise value of $\Delta f$. This is a simple ray-optics textbook problem, and can easily be achieved with some trigonometry. In the first step, we observe that the angle $\beta$ is functionally dependent on the incidence angle $\alpha$, as deduced from the fundamental principles of the law of refraction:

$$
\begin{aligned}
    n_\text{r} = \frac{\sin(\alpha)}{\sin(\beta)} \;\; \Longrightarrow \;\;
    \beta = \text{arcsin}\left( \frac{\sin(\alpha)}{n_\text{r}}\right)
\end{aligned}
$$

 Subsequently, by employing the geometrical properties of triangle $ABC$, the distance $b$ can be mathematically represented as a function of the parameters $d$ and $\beta$:

$$
\begin{aligned}
    \tan(\beta) = \frac{b}{d}
    \;\; \Longrightarrow \;\;
    b = d \, \tan \left( \beta \right)
\end{aligned}
$$

Inserting (§eq:step1) into (§eq:step2a) leads to:

$$
\begin{aligned}
     b = d \, \tan \left(  \text{arcsin}\left( \frac{\sin(\alpha)}{n_\text{r}}\right)  \right)
     \;\; \Longrightarrow \;\;
     b = d \, \frac{\sin(\alpha)}{\sqrt{n_{\text{r}}^2-\sin^2(\alpha)}}

\end{aligned}
$$

In the third step, the framework of triangle $ABD$ is utilized to formulate the sum of distances $b$ and $c$ as a function of $\alpha$ and $d$. Subsequently, this equation is transformed to isolate and express the variable $c$:

$$
\begin{aligned}
    \tan(\alpha) = \frac{b+c}{d}
    \;\; \Longrightarrow \;\;
    b+c = d \, \tan(\alpha)
    \;\; \Longrightarrow \;\;
    c = d \, \tan(\alpha) - b
\end{aligned}
$$

[ht]

_Figure/Table caption:

 Inserting (§eq:step2b) into this expression, allows us to express $c$ as a function of $\alpha$:

$$
\begin{aligned}
    c = d \, \left[ \tan(\alpha) - \frac{\sin(\alpha)}{\sqrt{n_\text{r}^2-\sin^2(\alpha)}}\right]
\end{aligned}
$$

In the final step, through the application of the properties of triangle $CDE$, a relationship is established between the variables $\Delta f$, $c$, and $\alpha$:

$$
\begin{aligned}
    \tan(\alpha)=\frac{c}{\Delta f}
    \;\; \Longrightarrow \;\;
    \Delta f = \frac{1}{\tan(\alpha)} \, c
\end{aligned}
$$

Inserting (§eq:step5) into this expression finally leads to:

$$
\begin{aligned}
    \Delta f =  d\,\left[ 1 - \frac{\cos(\alpha)}{\sqrt{n_\text{r}^2-\sin^2(\alpha)}}\right]
\end{aligned}
$$

For small incident angles $\alpha$ this can be approximated as

$$
\begin{aligned}
    \Delta f \approx d \left( 1-  \frac{1}{n_\text{r}}\right)
\end{aligned}
$$

The issue with this result lies in the practical limitations associated with adjusting the position of the rightmost mirror in the MAD-EP-CPA setup, as illustrated in Fig.~1(d) of the main text. Altering the mirror's position by a distance $\Delta f$, while maintaining the focal length $f$ of the adjacent lens constant, results in different optical lengths of the left and the right sub-cavity. While this modification corrects spatial degeneracy, it also changes the right sub-cavity's length, introducing several challenges: (i)~For effective operation as an MAD-EP-CPA, a specific longitudinal resonance mode of the left sub-cavity must be chosen. It is then essential to ensure that $\Delta f$ is an integer multiple of the corresponding wave-length. (ii) Unlike a symmetric design where both cavities are of equal optical length and the EP-CPA exhibits EP behavior across all longitudinal resonance frequencies, this asymmetric design is limited to a single wavelength. Calculations reveal that (keeping all other parameters equal) even a $\Delta f$  as small as $1 \text{mm}$  can result in a minimum absorption of about $1\%$ at the longitudinal mode adjacent to the resonant wavelength, with minimum absorption deteriorating rapidly for all further neighboring modes.  \\

 Therefore, it is more advantageous to establish a new focal length, referred to as $f'$,  for the rightmost lens, such that the total optical path length between the two focal points of this lens remains unchanged. Let us define $L$ as the optical path length between the two focal points of the rightmost lens prior to the modification, and $L'$ as the optical path length between these points post-modification. The optical path length before the modification can be succinctly expressed as follows:

$$
\begin{aligned}
    L = 2f
\end{aligned}
$$

The optical path length $L'$ subsequent to the modification  can be represented as a function of the physical length $L_\text{phys}$. The physical length needs to be adjusted to account for the different optical path length in the absorber (compared to air), which can be expressed as $d(n_\text{r}-1)$. Consequently, we can express the optical length after the modification as:

$$
\begin{aligned}
    L' = L^{'}_\text{phys} + d(n_\text{r}-1)
\end{aligned}
$$

The physical length after the modification can be quantified as

$$
\begin{aligned}
    L^{'}_\text{phys} =2f' + \Delta f
\end{aligned}
$$

Consequently, by inserting Eq.~(§eq:Lsphys) into Eq.~(§eq:Lsopt1a), we can express the optical length after the modification as:

$$
\begin{aligned}
    L' = 2f' + \Delta f + d(n_\text{r}-1)
\end{aligned}
$$

Inserting Eq. (§eq:Deltaf) into Eq. (§eq:Lsopt1) leads to:

$$
\begin{aligned}
    L' = 2f' + d\left( n_\text{r} - \frac{1}{n_\text{r}}\right)
\end{aligned}
$$

We want the optical path length to remain unchanged after the modification. Therefore we stipulate:

$$
\begin{aligned}
    L' = L
\end{aligned}
$$

Inserting (§eq:Lsopt2) and (§eq:Lopt) into Eq. (§eq:L=L) finally leads to:

$$
\begin{aligned}
    f' = f - \frac{d}{2}\left(n_\text{r} - \frac{1}{n_\text{r}} \right)
\end{aligned}
$$

Therefore, adjusting the focal length $f$ of the rightmost lens in the MAD-EP-CPA configuration, as illustrated in Fig.~1(d) of the main text, to a new value $f'$ in accordance with Eq.~(§eq:fs), effectively compensates for refraction aberrations, reinstates spatial degeneracy, and preserves the exceptional point.\\

 Note that Eq.~(§eq:fs) is also applicable for preventing refraction aberrations in the alternative design (``Design B''), as depicted in Fig.~4 of the main paper.
\pagebreak

\rule{\textwidth}{0.5pt}

~~\\

 [1] A. E. Siegman, Lasers *(Revised)* (UNIVERSITY SCIENCE BOOKS, 1986).

 [2] J. Frei, X.-D. Cai, and S. Muller, Multiport s-parameter and t-parameter conversion

with symmetry extension, IEEE Transactions on Microwave Theory and Techniques **56**,

2493 (2008).

 [3] Y. Slobodkin, G.Weinberg, H. H\"orner, K. Pichler, S. Rotter, and O. Katz, Massively

 degenerate coherent perfect absorber for arbitrary wavefronts, Science **377**, 995 (2022).

 [4] H. van de Stadt and J. M. Muller, Multimirror fabry–perot interferometers, Journal of

the
Optical Society of America A **2**, 1363 (1985).

 [5] F. A. Graybill, *Matrices with applications in statistics* (Belmont, Calif. : Wadsworth

International Group, 1983).

 [6] T.-T. Lu and S.-H. Shiou, Inverses of 2 × 2 block matrices, Computers \& Mathematics

with Applications 43, 119 (2002).
