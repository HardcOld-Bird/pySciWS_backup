# Negative conductivity induced reconfigurable gain metasurfaces and their nonlinearity

Xiaoyue Zhu $^{1,2}$ , Chao Qian $^{1,2,*}$ , Erping Li $^{1}$ and Hongsheng Chen $^{1,2,3,*}$

$^{1}$ ZJU-UIUC Institute, Interdisciplinary Center for Quantum Information, State Key Laboratory of Extreme

Photonics and Instrumentation, Zhejiang University, Hangzhou 310027, China.

$^{2}$ ZJU-Hangzhou Global Science and Technology Innovation Center, Key Lab. of Advanced Micro/Nano

Electronic Devices & Smart Systems of Zhejiang, Zhejiang University, Hangzhou 310027, China.

$^{3}$ Jinhua Institute of Zhejiang University, Zhejiang University, Jinhua 321099, China.

\*Corresponding authors: chaoq@intl.zju.edu.cn (C. Qian); hansomchen@zju.edu.cn (H. Chen)

Supplementary Note 1: Equation derivations and theoretical analysis from electromagnetics

Referring to Poynting theorem [21], we have:

$$
\nabla \cdot \bar {S} - i \omega [ \bar {H} ^ {*} \cdot \bar {B} - \bar {E} \cdot \bar {D} ^ {*} ] + \bar {J} ^ {*} \cdot \bar {E} = 0\tag{S1}
$$

where $\bar{S}$ is Poynting vector, defined as $\bar{S} = \bar{E} \times \bar{H}^{*}$ . The term $\nabla \cdot \bar{S}$ is the divergence of the Poynting vector. The symbols $\bar{E}, \bar{D}, \bar{H}, \bar{B}, \bar{J}$ and $\omega$ refer to the electric field intensity, electric displacement vector, magnetic field intensity, magnetic flux density, current density, and angular frequency, respectively [21].

According to the constitutive conditions [21,37-39], we have:

$$
\bar {B} = \mu (\omega) \bar {H}\tag{S2}
$$

$$
\overline {{D}} = \epsilon (\omega) \bar {E}\tag{S3}
$$

where $\epsilon(\omega)$ and $\mu(\omega)$ denote permittivity and permeability, respectively.

According to ohm's law [21, 38], we know that

$$
\bar {J} = \sigma (\omega) \bar {E}\tag{S4}
$$

where $\sigma(\omega)$ denotes conductivity as mentioned in the main text.

By incorporating the above equations (Eqs. (S2) - (S4)) into Eq. (S1), we can easily simplify Eq. (S1) into the following form:

$$
\nabla \cdot \bar {S} + i \omega \epsilon^ {*} (\omega) | \bar {E} | ^ {2} - i \omega \mu (\omega) | \bar {H} | ^ {2} + \sigma (\omega) | \bar {E} | ^ {2} = 0\tag{S5}
$$

where $\epsilon^{*}(\omega)$ is the conjugate complex number of $\epsilon(\omega)$ . In EM theories [21, 38], this equation describes the energy transformation relationships in an EM field. To be specific, the imaginary part of $\nabla\cdot\bar{S}$ (i.e., $Im[\nabla\cdot\bar{S}]$ ) is associated with the inactive power unceasingly transformed between $\bar{E}$ and $\bar{H}$ [21]. Meanwhile, the real part $Re[\nabla\cdot\bar{S}]$ represents the total active power losses.

In an idea medium, the permittivity $\epsilon(\omega)$ and permeability $\mu(\omega)$ are always real. For ideal conductors, the conductivity $\sigma(\omega)$ is also always real. In this context, we can further simplify Eq. (S5) into the following form:

$$
\mathrm{Re} [ \nabla \cdot \bar {S} ] + \sigma (\omega) | \bar {E} | ^ {2} = 0\tag{S6}
$$

$$
\mathrm{Im} [ \nabla \cdot \bar {S} ] + \omega \epsilon | \bar {E} | ^ {2} - \omega \mu | \bar {H} | ^ {2} = 0\tag{S7}
$$

In EM fields, the terms $\frac{1}{2}\omega\epsilon|\bar{E}|^{2}$ , $\frac{1}{2}\omega\mu(\omega)|\bar{H}|^{2}$ and $\frac{1}{2}\sigma(\omega)|\bar{E}|^{2}$ indicate the stored electric energy density, stored magnetic energy density, and the energy consumed by the current $\bar{J}$ , respectively [11]. Obviously, the active power in an ideal medium is primarily consumed by the current losses $\sigma(\omega)|\bar{E}|^{2}$ . Consequently, in past decades, multitudes of sophisticated methods were proposed and tried to reduce $\sigma(\omega)$ for low-loss designs. However, they did not break the physical limit to realize gain materials.

While, in practices, the permittivity $\epsilon(\omega)$ and permeability $\mu(\omega)$ are usually complex, i.e., $\epsilon(\omega)=\epsilon'(\omega)+i\epsilon''(\omega)$ and $\mu(\omega)=\mu'(\omega)+i\mu''(\omega)$ as mentioned in the main text. Therefore, Eq. (S5) can be rewritten into the following form:

$$
\mathrm{Re} [ \nabla \cdot \bar {S} ] + \sigma (\omega) | \bar {E} | ^ {2} + \omega \epsilon^ {\prime \prime} (\omega) | \bar {E} | ^ {2} + \omega \mu^ {\prime \prime} (\omega) | \bar {H} | ^ {2} = 0\tag{S8}
$$

$$
\mathrm{Im} [ \nabla \cdot \bar {S} ] + \omega \epsilon^ {\prime} (\omega) | \bar {E} | ^ {2} - \omega \mu^ {\prime} (\omega) | \bar {H} | ^ {2} = 0\tag{S9}
$$

As implied by Eq. (S8), the total active power losses are predominantly caused by current losses and the imaginary parts of permittivity and permeability. In artificial lossy materials, researchers usually incorporate $\sigma(\omega)$ into $\epsilon''(\omega)$ to form an equivalent permittivity [17, 38]:

$$
\epsilon_ {e f f} (\omega) = \epsilon^ {\prime} (\omega) + i \left[ \epsilon^ {\prime \prime} (\omega) + \frac {\sigma (\omega)}{\omega} \right]\tag{S10}
$$

When induced by plane transverse electromagnetic (TEM) waves, $|\bar{E}| = \eta |\bar{H}|$ , where $\eta = \sqrt{\mu/\epsilon_{eff}}$ . Based on the above analysis and equations, we can thus sperate the Eq. (S5) into the following form:

$$
\mathrm{Re} [ \nabla \cdot \bar {S} ] + \omega \epsilon_ {e f f} ^ {\prime \prime} (\omega) | \bar {E} | ^ {2} + \omega \frac {\mu^ {\prime \prime} (\omega)}{| \mu (\omega) |} | \epsilon_ {e f f} (\omega) | | \bar {E} | ^ {2} = 0\tag{S11}
$$

$$
\mathrm{Im} [ \nabla \cdot \bar {S} ] + \omega \epsilon^ {\prime} (\omega) | \bar {E} | ^ {2} - \omega \frac {\mu^ {\prime} (\omega)}{| \mu (\omega) |} | \epsilon_ {e f f} (\omega) | | \bar {E} | ^ {2} = 0\tag{S12}
$$

Eq. (S11) indicates that the total active power in the artificial lossy medium is primarily consumed by the imaginary parts of permeability and equivalent permittivity. Eq. (S12) describes the internal inactive power transformation conditions which do not directly result in losses. Moreover, Eq. (S11) and Eq. (S12) correspond to the real and imaginary parts of Eq. (2) in the main text. By combining Eq. (S11) and (S12), we can easily obtain the following equation, which is identical to Eq. (2) in the main text:

$$
\nabla \cdot \bar {S} = - \omega \left\{\left[ \frac {\epsilon_ {e f f} ^ {\prime \prime}}{| \epsilon_ {e f f} |} + \frac {\mu^ {\prime \prime}}{| \mu |} \right] - i \left[ \frac {\epsilon_ {e f f} ^ {\prime}}{| \epsilon_ {e f f} |} + \frac {\mu^ {\prime}}{| \mu |} \right] \right\} | \epsilon_ {e f f} | | E | ^ {2}\tag{S13}
$$

As aforementioned, there are still more intriguing phenomena and buried principles to be revealed. Traditionally, the condition $\epsilon''_{eff}(\omega) \geq 0$ holds true for nearly all natural materials. As a result, the formula $\mathrm{Re}[\nabla \cdot \bar{S}] < 0$ holds true for nearly all of the traditional natural materials, indicating that the total active power $\mathrm{Re}[\nabla \cdot \bar{S}]$ is flowing into and absorbed by the materials. However, if $\sigma(\omega)$ is sufficiently negative, the term $\mathrm{Re}[\nabla \cdot \bar{S}]$ will become positive (i.e., $\mathrm{Re}[\nabla \cdot \bar{S}] > 0$ ) as implied by Eq. (S8) and Eq. (S10), which means the total active power flowing into the material is less than the power flowing out. In other words, the material operates in a gain state. The gain value can be controlled by adjusting the conductivity $\sigma(\omega)$ . It is worth noting that these additional gains do not violate energy conservation laws, as the negative conductivity requires additional direct current voltages to induce.

## Supplementary Note 2: Structural designs and equivalent circuits.

As discussed in the main text, we have fabricated various samples operating in different frequency bands. The specifics of the first type of sample are detailed in the main text, while the other is depicted in Fig. S1a. Both of the two kind of cell structures are evidently symmetric. The parameters shown in Fig. S1a are: $a_{2} = 37.5$ , $b_{2} = 45$ , h = 1, $w_{3} = 11.5$ , $w_{4} = 20$ , $p_{3} = 35.5$ , $p_{4} = 8.75$ , g = 2 (unit: mm) respectively. Obviously, these two types of samples primarily differ in their specific structural parameters while maintaining a similar overall structure. The second sample is smaller in size, resulting in a higher resonant frequency.

![](images/3566e839e1bd885345a6aa4030fce2c3ee9fbf55c015f979c94bac4410bbe83e.jpg)
Fig. S1| Equivalent circuit models. a, The unit cell and equivalent model. The blue arrow indicates incident wave ports which can be deemed as an observing terminal of circuits. b, Extruded equivalent circuits of the whole model for circuit analysis.

According to microwave circuit theories [38, 40-41], microstrips can be modeled as complex impedances. For instance, the three parts of the left T patch shown in Fig. S1a can be modeled as two parallel complex impedances (such as $Z_{21}$ and $Z_{22}$ ) cascaded with another complex impedance (e.g., $Z_{23}$ ). In practice, these complex impedances can be also deemed as equivalent inductors, for that the high frequency current flowing through the metallic patches will produce quasi static magnetic field [10, 41-42]. Specifically, $Z_{21} = i\omega L_{21}$ , $Z_{22} = i\omega L_{22}$ , $Z_{23} = i\omega L_{23}$ . Among, the inductance values of these complex impedances are determined by the length (P) and width (W) of the metallic strip. And the inductance values of these components can be computed using the

equation [10, 41, 43]:

$$
L = \frac {\mu P}{2 \pi} \log \left(\frac {1}{\sin \left(\frac {\pi W}{2 P}\right)}\right)\tag{S14}
$$

To acquire the practical inductance values, we can substitute the parameters in the equation with practical structural parameters. For example, considering the sample depicted in Fig. S1a, the length

(P) and width (W) of $L_{21}$ are $\frac{p_{3}}{2}$ and $\frac{w_{3}}{2}$ , respectively. Consequently, $L_{21} = \frac{\mu p_{3}}{4\pi} \log \left( \frac{1}{\sin \left( \frac{\pi w_{3}}{2p_{3}} \right)} \right)$ .

Besides, the gap between the two metallic T patches, where TD is welded, can be modeled as capacitor $C_{g}$ [10, 41-42, 44] whose value is mainly decided by the patch length $p_{4}$ and gap width $g_{2}$ [40-42, 45]. The value of $C_{g}$ can be calculated via simulation software like TXLine and ADS. As for the ground plane, it can be modeled either as an inductance $Z_{b}$ [42], as shown in Fig. S1b, or as transmission lines linked with grounds [10]. Since the above empirical formulas may be not sufficiently accurate, the parameters are usually further calculated and optimized with the help of simulation software such as CST studio, HFSS and ADS. And the equivalent circuit of TD are shown in the right margin of Fig. S1b [45]. With the help of this equivalent circuit models, we can thus acquire the terminal impedance $Z_{A}$ by integrating the equivalent elements shown in Fig. S1b. To be specific, $Z_{A} = (Z_{1} + ((L_{s} + (C_{d} // -R_{d}) + R_{s}) // C_{g}) + Z_{2}) // Z_{b}$ , where “//” is an operator widely used in circuit analysis (defined as: $A // B = \frac{A \cdot B}{A + B}$ ). Sometimes there may also be an equivalent transmission line between this port and the final receiving port [42]. To be rigorous, the final terminal impedance $Z_{L}$ can be calculated via the equation [21, 38]:

$$
Z _ {L} = Z _ {0} \frac {Z _ {A} + i Z _ {0} t a n \beta l}{Z _ {0} + i Z _ {A} t a n \beta l}\tag{S15}
$$

The electrical length $\beta l$ is defined as $\beta l = 2\pi l/\lambda_{g}$ , where l is the length of transmission lines, $Z_{0}$ is the characteristic impedance, typically a real constant and $\lambda_{g}$ is the guide wavelength [38]. As implied by Eq. (S15), whether $Z_{L}$ is negative mainly depends on whether the real part of $Z_{A}$ is sufficiently negative.

In reflective metasurfaces, circuits, and systems, the reflection coefficient $\Gamma_{L}$ is commonly used to depict the reflection properties as mentioned in the main text [13, 42]. And the reflection coefficient $\Gamma_{L}$ can be calculated via the final terminal impedance $Z_{L}$ [38, 42]:

$$
\Gamma_ {L} = \frac {Z _ {L} - Z _ {0}}{Z _ {L} + Z _ {0}}\tag{S16}
$$

As suggested by Eq. (S16) and mentioned in the main text, the magnitude of reflection coefficient $\left(\left|\Gamma_{L}\right|\right)$ are determined by absolute value $\left|\frac{Z_{L}-Z_{0}}{Z_{L}+Z_{0}}\right|$ . To achieve gains, we have to let $\left|\Gamma_{L}\right|>1$ . In other words, $\left|Z_{L}-Z_{0}\right|$ should be bigger than $\left|Z_{L}+Z_{0}\right|$ . As $Z_{0}$ is a real constant, only when the real part of $Z_{L}$ is sufficiently negative can $\left|Z_{L}-Z_{0}\right|>\left|Z_{L}+Z_{0}\right|$ . Hence, negative conductivity is necessary for reflective gain realizations. According to Eq. (S15), we can also derive that the magnitude of the reflection coefficient reaches its maximum only when the real part of $Z_{L}$ is negatively matched with $Z_{0}$ (i.e., $Re[Z_{L}] = -Z_{0}$ ). When the real part of $Z_{L}$ deviates from the point, the magnitude of the reflection coefficient will begin to decline.

## Supplementary Note 3: Reflection spectra.

As mentioned in the main text we have fabricated different samples in microwave band for experimental verifications. Detailed parameters of these different samples are listed in the main text and Supplementary Note 2. To be more intuitive, we put photographs of the two kinds of samples in Fig. S2. To be rigorous, we tested the normalized reflection properties in advance. The reflection properties showcased in Fig. S2 are normalized with a same shaped metal plate. As illustrated, the reflection spectra are not perfectly flat but exhibit small fluctuations due to the resonating properties of the samples and the noise introduced by the testing equipment and environment. However, these fluctuations do not significantly affect performance. As shown in Fig. S2, the reflected spectra display an obvious valley at the working bands around 2.5 GHz and 4.1 GHz, indicating that the metasurfaces operate in a lossy state without typical positive conductivities.

![](images/63ce15f40aa2c98132f9ecd932c6b345a5b4e06b9821cbdcdb453618ad2728a3.jpg)

![](images/55c2a834ba137bc42f304b4332d1851e6896559dec59bc99f9bd5260f99ec439.jpg)
Fig. S2 | Normalized spectra of the used samples with no bias voltage imposed. These results are normalized with the same sized metal plates. a, Normalized spectra of Sample 1. The photograph of a cell in the sample is shown at the top right, where $a_{1} = 50$ , $b_{1} = 60$ . b, Normalized spectra of Sample 2. The photograph of a cell in the sample is shown at the top right, $a_{2} = 37.5$ , $b_{2} = 45$ .
