ARTICLE

Received 14 Sep 2015 | Accepted 18 Feb 2016 | Published 30 Mar 2016

DOI: 10.1038/ncomms11110

# Accessing the exceptional points of parity-time symmetric acoustics

Chengzhi Shi $^{1,*}$ , Marc Dubois $^{1,*}$ , Yun Chen $^{2}$ , Lei Cheng $^{2}$ , Hamidreza Ramezani $^{1}$ , Yuan Wang $^{1}$ & Xiang Zhang $^{1,3,4}$

Parity-time (PT) symmetric systems experience phase transition between PT exact and broken phases at exceptional point. These PT phase transitions contribute significantly to the design of single mode lasers, coherent perfect absorbers, isolators, and diodes. However, such exceptional points are extremely difficult to access in practice because of the dispersive behaviour of most loss and gain materials required in PT symmetric systems. Here we introduce a method to systematically tame these exceptional points and control PT phases. Our experimental demonstration hinges on an active acoustic element that realizes a complex-valued potential and simultaneously controls the multiple interference in the structure. The manipulation of exceptional points offers new routes to broaden applications for PT symmetric physics in acoustics, optics, microwaves and electronics, which are essential for sensing, communication and imaging.

Non-reciprocal wave transport is important for key applications such as isolators $^{1-3}$ , diodes $^{4,5}$ , and rectifiers $^{6,7}$ . However, these schemes rely on high-order harmonic generation, mode conversion or angular-dependent bandgap in order to realize asymmetric transmission. Recent progress on parity-time (PT) symmetric systems provides a new paradigm to realize asymmetric transport without frequency or mode conversion $^{8-20}$ . Only at specific frequencies called exceptional points, complex interference in PT symmetric systems results in unidirectional reflection that is crucial for applications $^{9,10,21}$ . PT symmetric systems are invariant under parity and time-reversal operations. Therefore, they obey the following energy conservation relation:

$$
r _ {\mathrm{l}} r _ {\mathrm{g}} ^ {*} = 1 - | t | ^ {2},\tag{1}
$$

where t is the transmission coefficient, $r_{1}$ is the reflection coefficient from the loss side and $r_{g}$ is the reflection coefficient from the gain side of the system $^{16,21}$ . In PT symmetric systems, phase transition, where the system Hamiltonian switches between real and complex spectrum $^{22,23}$ , provides a novel approach to realize asymmetric transport. This phase transition has been intensely studied in electronics $^{24-26}$ , optics $^{8-14,18-20,27-29}$ , and acoustics $^{16,17}$ . The right-hand side sign of equation (1) determines the PT phase of the system: positive for PT exact phase and negative for PT broken phase $^{16,21}$ . When the system undergoes phase transition, transmission is unity, $|t|=1$ , corresponding to the existence of exceptional points. Therefore, one of the reflections must vanish at these exceptional points, typically resulting in unidirectional transparency. The PT symmetric condition requires the distribution of complex refractive index to be a Hermitian function in the transverse direction such that $n(x)=n^{*}(-x)$ (refs 8,10). This distribution is achieved by precisely balancing loss and gain materials. However, PT symmetric systems require an exact balance of loss and gain in the medium, and the dispersive behaviour of most loss and gain materials prevents this condition to be fulfilled over a broad frequency range. Hence, it is challenging to access the exceptional points in practice.

Here, we present a general method to approach the exceptional points within a large frequency range by tuning an acoustic gain medium that satisfies the PT symmetric condition and simultaneously controls the scattering process. We experimentally demonstrate the ability to reverse the direction of the acoustic transparency at any given frequency. Such control of directional transparency opens new routes for wave physics including acoustics and electromagnetics, and important applications in sensing, communication and imaging.

## Results

Tailoring the exceptional points of PT symmetric structure. Figure 1 illustrates the control of exceptional points for typical dispersive loss and gain materials, the imaginary parts of their indices are balanced at a single frequency $f_{b}$ . When there is no spacing between the loss and gain materials (Fig. 1d), one can observe reflections from both sides, but cannot access exceptional points. Exceptional points and unidirectional transparency are the results of complex interference inside the PT symmetric structure. This scattering process can be controlled by introducing a gap corresponding to a phase delay between the gain and loss cells (Fig. 1b). Analytic calculations based on transfer-matrix formalism are used to scan the response of the structure as a function of spacing and frequency. The reflection from the loss side (Fig. 1c) presents a sharp dip for a specific spacing $L_{0}$ at frequency $f_{b}$ , where the multi-reflection interfere destructively. In this case, one can access an exceptional point where the reflection from the loss side vanishes while achieving total transmission and nonzero reflection from the gain side, making the system to be unidirectional transparent (Fig. 1e).

![](images/91215a36cfa1802c5de1becfda4b20e22770107a05412231fa70b5742d457c71.jpg)
b

![](images/93d42a69e3e52a6bd84c8098290e3b91de46a78138d1fcee0ca82fe996fce1d4.jpg)

![](images/7b00afd5982d4ef473d89e1797b900079de532e3c344325618eb603a3096991d.jpg)

![](images/21cd49f856464ae95120a6b15cc71435d22fe44ff12881ff27ace2c6ad180934.jpg)

![](images/5c28177b6b2e2c1f12bfef4861c6e4f86ecaf12b32fff3b15b646a459144be54.jpg)
Figure 1 | Accessing the exceptional point of acoustic PT system by tuning the spacing between loss and gain materials. (a) Imaginary parts of the refractive indices of loss and gain materials are typically dispersive, thus the PT symmetric condition could only be satisfied at single frequency $f_{b}$ where the loss and gain materials are exactly balanced. (b) Loss and gain materials assembled with spacing. (c) Amplitude of the reflection from the loss side as a function of frequency and the spacing between the loss and gain materials. Exceptional point (EP) occurs at frequency $f_{b}$ and a specific spacing $L_{0}$ when the reflection vanishes. (d) The normalized transmissions (blue, i), reflections from the loss (green, ii), and from the gain (red, iii) in logarithmic scale without spacing, no exceptional point observed. (e) Similar representation to d with spacing $L_{0}$ , an exceptional point observed at $f_{b}$ .

a

Experimental demonstration of unidirectional transparency. Recent development in PT symmetric acoustics promises new applications such as asymmetric cloaking and sensing $^{16,17}$ . The control of exceptional points is crucial for the design of such devices. The realization of PT symmetric acoustics relies on the combination of acoustic loss and gain materials. It has been shown that active elements can be tuned to obtain artificial materials with desired effective acoustic parameters. This was demonstrated in the modulation of the real part of the acoustic parameters $^{30}$ . In this work, we explore the acoustic parameters in the complex domain to exploit non-Hermitian physics and the control of exceptional points. In our experiments, coherent acoustic sources are used to generate a gain response that balances a passive loss material and satisfies the PT symmetric condition. Our acoustic prototype consists of one loss cell, where multiple slits are carved in a waveguide to form a leaky region, and one gain unit, formed by two acoustic source arrays paired with upstream and downstream highly directive sensors (Fig. 2a). We first obtain the scattering matrix of the loss part at 5.3 kHz by measuring the amplitude and phase of transmission and reflection without the gain unit. This frequency corresponds to a local maximum of the loss parameters, resulting in high contrast between the reflections of the PT symmetric system (Fig. 3a). A parameter extraction algorithm is used to retrieve the effective parameters of the loss material from the measured scattering matrix at 5.3 kHz (ref. 31). The calculated mass density is $\rho = 1.13 \exp(-i0.68) \text{ kg m}^{-3}$ and bulk modulus is $\kappa = 0.16 \exp(-i0.87) \text{ MPa}$ (refractive index is $n = c_{0} \sqrt{\rho/\kappa} = 0.9 + 0.09i$ with $c_{0} = 343 \text{ ms}^{-1}$ ). Effective parameters of the gain medium are obtained by complex conjugation, and the same algorithm can be reversed to calculate the gain scattering matrix. By tuning the two acoustic source arrays to match this scattering matrix, the PT symmetric condition can be satisfied at 5.3 kHz. As discussed previously, exceptional point occurrences are dictated by the multiple interferences within the structure. In our experiment, these interferences are controlled by the spacing between the loss and gain materials. Analytical calculation at 5.3 kHz reveals that 1.24-cm spacing results in unidirectional transparency from the loss side. The loss region and the balanced gain unit are assembled in a waveguide with the correct spacing. The scattering matrix of the entire system is measured by two calibrated microphone pairs

$$
| S | = \left( \begin{array}{c c} | t _ {1} | & | r _ {\mathrm{g}} | \\ | r _ {1} | & | t _ {\mathrm{g}} | \end{array} \right) = \left( \begin{array}{c c} 0. 9 9 & 3. 3 4 \\ 0. 0 2 & 1. 0 2 \end{array} \right).\tag{2}
$$

The scattering matrices of the loss and balanced gain units are given in Supplementary Note 1. The experimental results confirm the unidirectional transparency from the loss side, which agree well with the analytical predictions (Fig. 2b–e).

Accessing exceptional points at multiple frequencies. One major benefit of using active gain material is that the PT symmetric condition can be satisfied over a large frequency band, regardless of the loss material dispersion. We demonstrate that unidirectional transparency is achieved by controlling exceptional points for different frequencies within the range of 5–6 kHz. The scattering matrices of the leaky waveguide are measured at four frequencies and the retrieved refractive indices confirm the dispersive behaviour of our loss part (Fig. 3a). The complex conjugates of these indices are used to calculate the scattering matrices of the gain material at each frequency. In order to observe unidirectional transparency at these frequencies, different spacing are required in the system. The calculated spacing distances and the measured scattering matrices are listed in Fig. 3b. By applying opposite phase shifts to the two acoustic source arrays, we introduce an effective spatial offset between the gain medium and these emitting arrays. The phase shifts allow us to set the effective spacing between the loss and gain materials in a fixed structure for each frequency (Supplementary Note 2). For all measurements, total transmissions through the PT symmetric system can be observed within 2% error, the reflections from the loss side vanish (<3%) and the reflections from the gain side are nonzero. It is important to point out that both the loss and gain units are working in linear conditions. Thus, the system will preserve the unidirectional reflection when input signals are linear combinations of the demonstrated frequencies, promising time-dependent signal applications.

![](images/3ee686225aa9376bb4d6cecc99b3bd6818e02fb4ed8f7ca968afaea5750e813f.jpg)

![](images/86cefc05ffc889c9e58e3b5b5e2644089f5b8a1c89fa9b2c616922826f6fa726.jpg)

![](images/b2179bba5b0547377047488b66a18160dea4613f6b045ed7a228c039e35131a5.jpg)

![](images/a2def7210048397f041171a8f2fb59ba1d51e3c4b43f15af9b0560ef6171f288.jpg)

![](images/efd155ced34e9b86ccb1c444162f155ee1c8638c260c50d6fad97c3146a67fb2.jpg)

![](images/67bb9d998058951f0e825ef7865c7a160fc70f588a0fe7ad4b421a7433d12104.jpg)

![](images/869100802571aef45f410d413ab6695b03acdf92182512211f418d955be34fc2.jpg)
Figure 2 | Experiment demonstration of unidirectional transparency of the PT symmetric system at 5.3 kHz. (a) The photo of an experimental sample including loss and gain units. (b,c) The calculated (solid curves) and measured (marked dots) transmissions and reflections when the incident wave is coming from the loss side. (d,e) Similar representation to b and c) when the incident wave is from the gain side. Black, green, red, blue and grey colours denote the incidence, reflection from the loss side, the reflection from the gain side, the transmission and the reflection from the end of the waveguide, respectively. All results have been normalized with the amplitude of incidence. The two calibrated unidirectional microphones are mounted at $I_{1}=15.5$ cm and $I_{2}=13$ cm away from the boundaries of our PT symmetric materials. The spacing between the loss and gain materials is L=1.24 cm. No reflection is observed from the loss side (green curve and dots in b), $\sim330\%$ reflection is observed from the gain side (red curve and dots in d), and total transmissions ( $|t|=1$ ) have been observed on both sides, resulting in unidirectional transparency from the loss side.

![](images/20177d70b91708936125e17c72b853c64152cb86f5158943b876d0b16480c88b.jpg)
Figure 3 | Taming exceptional points at different frequencies. (a) The measured complex-valued refractive indices of the loss material formed by the leaky waveguide at 5, 5.3, 5.7 and 6 kHz (open marks). The active gain material is tuned to have the refractive indices that are complex conjugate of those of the loss material at the corresponding frequencies (filled marks). (b) The measured amplitudes of the scattering matrices for the controlled exceptional points at these four frequencies with the appropriately tuned spacing between the loss and gain materials. The reflections from the loss side almost vanish and total transmissions are observed for all of the frequencies. The reflections from the gain side are nonzero.

Reversing the orientation of unidirectional transparency. Previous studies in optics and acoustics only demonstrate unidirectional transparency from the loss side $^{10,17}$ , whereas equation (1) reveals that two types of exceptional points exist depending on the side where reflection is cancelled. Here we demonstrate that our method can control the orientation of unidirectional transparency at the same working frequency, which is useful for applications such as logical acoustics. Figure 4a presents the transmission and reflection coefficients of the PT symmetric system as functions of spacing. These coefficients are periodic functions with period $\lambda/2$ . Within this scope, two exceptional points are observed. The same as shown in Fig. 2, reflection from the loss side vanishes with 1.24-cm spacing. The other exceptional point occurs at 1.71-cm spacing, where the reflection from the gain side is cancelled. Figure 4b–e shows the transmissions and reflections normalized with their inputs for these two different gaps. In both cases, we can observe total transmissions of waves. Once we tune the spacing to be 1.71 cm, 70% reflection from the loss side is observed while the reflection from the gain side is $<8\%$ . The experimental results agree well with the analytical predictions, and prove that the orientation of the unidirectional transparency can be controlled by tuning the spacing between the loss and gain materials within half wavelength.

## Discussion

We propose and demonstrate a method to control the exceptional points of PT symmetric systems by tuning the spacing between the loss and gain materials. This enables us to shift the exceptional points to desired frequencies and control the orientation of unidirectional transparency. Our control method can be applied to different fields such as electronics, optics, microwaves and so on. Numerically, this control method can also handle PT symmetric periodic structures $^{9}$ or non-Hermitian systems without PT symmetry $^{11}$ (Supplementary Discussion). Thanks to the active acoustic gain medium, this control can be realized over a large frequency band. Unidirectional transparency for pulsed signals can be perceived if the transmitted phases and the non-vanishing reflections are adjusted to be identical at different frequencies using additional degrees of freedom such as number of unit cells, lengths of loss and gain units, and so on. This control ability over the exceptional points in PT symmetric system opens a new avenue for developing asymmetric wave transport devices, important for directional imaging and sensing, logic acoustics and other applications.

![](images/3e7bf210fd02ea3e0342a81270645b5026c52c2848c49c3f3abbe522b770da25.jpg)

![](images/89a165aa0904b0a0138e2b3da2c92060a1af6ab994c365799f32382b28da9a24.jpg)

![](images/1e9a841713777edd2e552c275e1022874a9a182c9f70f2a45f18d2df62cc34a6.jpg)

![](images/64c61c7101074853b4e5ee28bf5cd48f2d3b6ea345895810d470a732f5ca6fd4.jpg)

![](images/bd7ea939fde75a695a53625f0831b8b9f2ff2c33f23a084a938c574b1291af9a.jpg)
Figure 4 | Reversing the unidirectional transparency at 5.3 kHz. (a) Calculated transmission (blue) and reflections from the gain side (orange) and the loss side (red) with varying spacing between the loss and gain materials from zero to half wavelength. Two exceptional points are observed when the spacing is 1.24 and 1.71 cm, respectively. (b,c) Transmission and reflections from the gain and loss sides associated with 1.24-cm spacing. Black, blue, red and green colours denote the incidence, transmission, reflection from the gain side and reflection from the loss side, respectively. Solid curves are calculated values, and marked dots are from measurements. The same as in Fig. 2, unidirectional transparency is observed from the loss side. (d,e) Similar representation to b and c, but with 1.71-cm spacing. The direction of unidirectional transparency is reversed. Nearly zero reflection from gain side and 70% reflection from the loss side are observed.

## Methods

Loss material design. In this article, the acoustic refractive index is defined as n=343/c, where c is the complex-valued sound speed in the medium. We use a 0.148-m-long leaky waveguide consisting of nine slits with filling ratio 0.19 on the wall of the waveguide to form our loss part as shown in Fig. 2a. These slits leak the acoustic energy out of the waveguide. The height of the waveguide we use is 0.024 m. Both simulations and experiments confirm that this loss material is dispersive.

Active gain design. An active unit containing two arrays of Kobnite 254-PS600-RO speakers facing opposite directions is used to reproduce an artificial acoustic gain material because there is no passive acoustic gain material that exists in nature. Each speaker array possesses 15 speakers with spacing 0.03 m in between the centres of each speaker. The 0.03-m spacing is chosen because it is about half of the wavelength of acoustic wave at 6 kHz, the highest frequency we used in our experiment, and hence generate plane wave in the waveguide. The two speaker arrays are used to control the transmission and reflection of the artificial gain material to generate the scattering matrix, an acoustic gain material using a linear operation described in Supplementary Note 2. A third speaker array is used to generate the incident plane wave in our experiment.

Measurement and calibration. These three speaker arrays are controlled by a LabVIEW programme through NI-USB 6259 DAQ system. Two pairs of PUI Audio PUM-5250L-R microphones are assembled on each side of the media to measure the scattering matrix (that is, transmission and reflection coefficients). Each pair of microphones contains two microphones facing opposite directions. The measured signals are amplified and filtered using home-made amplification and filtering circuits. A calibration operation (Supplementary Note 3) is applied to the two microphone pairs so that they can detect the left and right going waves simultaneously.

## References

1. Yu, Z. & Fan, S. Complete optical isolation created by indirect interband photonic transitions. Nat. Photon. 3, 91–94 (2008).

2. Manipatruni, S., Robinson, J. T. & Lipson, M. Optical nonreciprocity in optomechanical structures. Phys. Rev. Lett. 102, 213903 (2009).

3. Kang, M. S., Butsch, A. & Russell, P. St. J. Reconfigurable light-driven opto-acoustic isolators in photonic crystal fiber. Nat. Photon. 5, 549–553 (2011).

4. Li, X. F. et al. Tunable unidirectional sound propagation through a sonic-crystal-based acoustic diode. Phys. Rev. Lett. 106, 084301 (2011).

5. Popa, B. I. & Cummer, S. A. Non-reciprocal and highly nonlinear active acoustic metamaterials. Nat. Commun. 5, 3398 (2014).

6. Liang, B., Guo, X. S., Tu, J., Zhang, D. & Cheng, J. D. An acoustic rectifier. Nat. Mater. 9, 989–992 (2010).

7. Boechler, N., Theocharis, G. & Daraio, C. Bifurcation-based acoustic switching and rectification. Nat. Mater. 10, 665–668 (2011).

8. Ruter, C. E. et al. Observation of parity-time symmetry in optics. Nat. Phys. 6, 192–195 (2010).

9. Lin, Z. et al. Unidirectional invisibility induced by PT-symmetric periodic structures. Phys. Rev. Lett. 106, 213901 (2011).

10. Regensburger, A. et al. Parity-time synthetic photonic lattices. Nature 488, 167–171 (2012).

11. Feng, L. et al. Experimental demonstration of a unidirectional reflectionless parity-time metamaterial at optical frequencies. Nat. Mater. 12, 108–113 (2013).

12. Chang, L. et al. Parity-time symmetry and variable optical isolation in active-passive-coupled resonators. Nat. Photon. 8, 524–529 (2014).

13. Peng, B. et al. Parity-time-symmetric whispering-gallery microcavities. Nat. Phys. 10, 394–398 (2014).

14. Nazari, F. et al. Optical isolation via PT-symmetric nonlinear Fano Resonances. Opt. Express 22, 9574 (2014).

15. Ramezani, H., Li, H. K., Wang, Y. & Zhang, X. Unidirectional spectral singularities. Phys. Rev. Lett. 113, 263905 (2014).

16. Zhu, X., Ramezani, H., Shi, C., Zhu, J. & Zhang, X. PT-symmetric acoustics. Phys. Rev. X 4, 031042 (2014).

17. Fleury, R., Sounas, D. & Alu, A. An invisible acoustic sensor based on parity-time symmetry. Nat. Commun. 6, 5905 (2014).

18. Peng, B. et al. Loss-induced suppression and revival of lasing. Science 346, 328–332 (2014).

19. Feng, L., Wong, Z. L., Ma, R. M., Wang, Y. & Zhang, X. Single-mode laser by parity-time symmetry breaking. Science 346, 972–975 (2014).

20. Hodaei, H., Miri, M. A., Heinrich, M., Christodoulides, D. N. & Khajavikhan, M. Parity-time-symmetric microring lasers. Science 346, 975–978 (2014).

21. Ge, L., Chong, Y. D. & Stone, A. D. Conservation relations and anisotropic transmission resonances in one-dimensional PT-symmetric photonic heterostructures. Phys. Rev. A 85, 023802 (2012).

22. Bender, C. M. & Boettcher, S. Real spectra in non-Hermitian Hamiltonians having PT symmetry. Phys. Rev. Lett. 80, 5243-5246 (1998).

23. Bender, C. M., Boettcher, S. & Meisinger, P. N. PT-symmetric quantum mechanics. J. Math. Phys. 40, 2201–2229 (1999).

24. Schindler, J., Li, A., Zheng, M. C., Ellis, F. M. & Kottos, T. Experimental study of active LRC circuits with PT symmetries. Phys. Rev. A 84, 040101 (2011).

25. Ramezani, H., Christodoulides, D. N., Kovanis, V., Vitebskiy, I. & Kottos, T. PT-symmetric Talbot effect. Phys. Rev. Lett. 109, 033902 (2012).

26. Ramezani, H., Schindler, J., Ellis, F. M., Gunther, U. & Kottos, T. Bypassing the bandwidth theorem with PT-symmetry. Phys. Rev. A 85, 062122 (2012).

27. Longhi, S. PT-symmetric laser absorber. Phys. Rev. A 82, 031801(R) (2010).

28. Chong, Y. D., Ge, L. & Stone, A. D. PT-symmetry breaking and laser absorber modes in optical scattering systems. Phys. Rev. Lett. 106, 093902 (2011).

29. Wan, W. et al. Time-reversed lasing and interferometric control of absorption. Science 331, 889–892 (2011).

30. Popa, B. I., Zigoneanu, L. & Cummer, S. A. Tunable active acoustic metamaterials. Phys. Rev. B 88, 024303 (2013).

31. Fokin, V., Ambati, M., Sun, C. & Zhang, X. Method for retrieving effective properties of locally resonant acoustic metamaterials. Phys. Rev. B 76, 14432 (2007).

## Acknowledgements

This research was supported by the Office of Naval Research (ONR) MURI program under Grant No. N00014-13-1-0631.

## Author contributions

C.S., M.D. and H.R. performed the theoretical simulations. C.S. designed and built the experimental setup. C.S., Y.C. and L.C. designed and fabricated the control circuits. C.S. and M.D. conducted the acoustic measurement. X.Z. and Y.W. supervised the research. All authors contributed to data analysis and discussions.

## Additional information

Supplementary Information accompanies this paper at http://www.nature.com/naturecommunications

Competing financial interests: The authors declare no competing financial interest.

Reprints and permission information is available online at http://npg.nature.com/reprintsandpermissions/

How to cite this article: Shi, C. et al. Accessing the exceptional points of parity-time symmetric acoustics. Nat. Commun. 7:11110 doi: 10.1038/ncomms11110 (2016).

![](images/fb78187c8bdb6530683d68311199c20b23f8cea9c5bfec2c7543529925bb034b.jpg)

This work is licensed under a Creative Commons Attribution 4.0 International License. The images or other third party material in this article are included in the article's Creative Commons license, unless indicated otherwise in the credit line; if the material is not included under the Creative Commons license, users will need to obtain permission from the license holder to reproduce the material. To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/
