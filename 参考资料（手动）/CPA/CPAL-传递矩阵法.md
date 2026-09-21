# CPA-laser condition

To find CPA-laser solutions of the PT-symmetric circuit from Fig. 4 in the main text, we need to derive the conditions for zeros and poles of the eigenvalues $$s_{1/2} = t \pm \sqrt{r_l r_r}$$. Since the system is PT-symmetric, any real-frequency pole will be collocated with a real-frequency zero, thus only finding the pole dispersion is sufficient. The left (loss) and right (gain) resonators can be represented as complex admittances $$Y_1=1/Z_1$$ and $$Y_2=1/Z_2$$, such that PT-symmetry holds $$Z_1 = -Z_2^*$$, where the complex impedances are given by $$Z_1 = R + jX$$, $$Z_2 = -R + jX$$, and $$R$$ and $$X$$ are real numbers $$R, X \in \mathbb{R}$$. Since the reflection coefficients are different from different ports, we calculate the left and right total ABCD matrices, as well as reflection and transmission coefficients, as:

$$
\begin{aligned}
M_{Tl} &= M_{\text{loss}} M_{TL} M_{\text{gain}} = \begin{bmatrix} 1 & 0 \\ Y_1 & 1 \end{bmatrix} \begin{bmatrix} \cos(kd) & jZ_0 \sin(kd) \\ \frac{j}{Z_0}\sin(kd) & \cos(kd) \end{bmatrix} \begin{bmatrix} 1 & 0 \\ Y_2 & 1 \end{bmatrix} \\
&= \begin{bmatrix} \cos(kd) + jZ_0 Y_2 \sin(kd) & jZ_0 \sin(kd) \\ \cos(kd)(Y_1+Y_2) + \frac{j}{Z_0}\sin(kd)(1+Z_0^2 Y_1 Y_2) & \cos(kd) + jZ_0 Y_1 \sin(kd) \end{bmatrix} \\
&= \begin{bmatrix} A_{Tl} & B_{Tl} \\ C_{Tl} & D_{Tl} \end{bmatrix}
\end{aligned}
\tag{s7}
$$

$$
\begin{aligned}
M_{Tr} &= M_{\text{gain}} M_{TL} M_{\text{loss}} = \begin{bmatrix} 1 & 0 \\ Y_2 & 1 \end{bmatrix} \begin{bmatrix} \cos(kd) & jZ_0 \sin(kd) \\ \frac{j}{Z_0}\sin(kd) & \cos(kd) \end{bmatrix} \begin{bmatrix} 1 & 0 \\ Y_1 & 1 \end{bmatrix} \\
&= \begin{bmatrix} \cos(kd) + jZ_0 Y_1 \sin(kd) & jZ_0 \sin(kd) \\ \cos(kd)(Y_1+Y_2) + \frac{j}{Z_0}\sin(kd)(1+Z_0^2 Y_1 Y_2) & \cos(kd) + jZ_0 Y_2 \sin(kd) \end{bmatrix} \\
&= \begin{bmatrix} A_{Tr} & B_{Tr} \\ C_{Tr} & D_{Tr} \end{bmatrix}
\end{aligned}
\tag{s8}
$$

$$
r_l = \frac{A_{Tl} + B_{Tl}/Z_0 - C_{Tl}Z_0 - D_{Tl}}{A_{Tl} + B_{Tl}/Z_0 + C_{Tl}Z_0 + D_{Tl}} = \frac{q_l}{p_l}
\tag{s9}
$$

$$
r_r = \frac{A_{Tr} + B_{Tr}/Z_0 - C_{Tr}Z_0 - D_{Tr}}{A_{Tr} + B_{Tr}/Z_0 + C_{Tr}Z_0 + D_{Tr}} = \frac{q_r}{p_r}
\tag{s10}
$$

$$
t_r = \frac{2}{A_{Tl} + B_{Tl}/Z_0 + C_{Tl}Z_0 + D_{Tl}} = \frac{2}{p}
\tag{s11}
$$

As required by reciprocity, the transmission coefficient is equal from both sides, i.e., the denominators are equal in equations (s9-s11) $$p=p_l=p_r$$. For brevity purposes, we write the numerators of reflection coefficients as $$q_l$$ and $$q_r$$, and denominator as $$p$$. The eigenvalues are then given by:

$$
s_{1/2} = t \pm \sqrt{r_l r_r} = \frac{2 \pm \sqrt{q_l q_r}}{p}
\tag{s12}
$$

The pole condition requires that the denominator $$p$$ is equal to 0, which gives the following equation:
$$
(2 + Z_0(Y_1 + Y_2))e^{jkd} + j \sin(kd) Z_0^2 Y_1 Y_2 = 0.
\tag{s13}
$$

When $$Y_1 = 1/Z_1 = 1/(R + jX)$$ and $$Y_2 = 1/(-R + jX)$$ are inserted in equation (s13), the following equation can be obtained:
$$
R^2 + X^2 + jZ_0X = \frac{Z_0^2}{2(1-j\cot(kd))}.
\tag{s14}
$$

As $$R$$ and $$X$$ are real numbers, we can equate the real and imaginary parts of the left and right sides of the equations(s14) as:
$$
Z_0 X = \text{imag}\left(\frac{Z_0^2}{2(1-j\cot(kd))}\right)
\tag{s15}
$$

$$
R^2 + X^2 = \text{real}\left(\frac{Z_0^2}{2(1-j\cot(kd))}\right)
\tag{s16}
$$

Since $$Z_0$$ is a real number, we can write the solution for the imaginary part of the complex impedance as:
$$
X = \frac{Z_0}{2}\text{imag}\left(\frac{1}{1 - j\cot(kd)}\right)
\tag{s17}
$$

After some trigonometric manipulation, this is further simplified to:
$$
X = -\frac{Z_0}{4}\sin(2kd).
\tag{s18}
$$

Similarly, equation (s16) is simplified to:
$$
R = \frac{Z_0}{\sqrt{2}} \sqrt{\sin^2(kd) - \frac{\sin^2(2kd)}{8}}
\tag{s19}
$$

The last two equations represent the complete CPAL solution shown in the main text.
