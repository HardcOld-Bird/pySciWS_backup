# Deep-Learning-Enabled On-Demand Design of Chiral Metamaterials

Wei Ma, $^{\dagger}$ Feng Cheng, $^{\ddagger}$ and Yongmin Liu $^{*,\dagger,\ddagger}$

$^{\dagger}$ Department of Mechanical and Industrial Engineering and $^{\ddagger}$ Department of Electrical and Computer Engineering, Northeastern University, Boston, Massachusetts 02115, United States

Supporting Information

ABSTRACT: Deep-learning framework has significantly impelled the development of modern machine learning technology by continuously pushing the limit of traditional recognition and processing of images, speech, and videos. In the meantime, it starts to penetrate other disciplines, such as biology, genetics, materials science, and physics. Here, we report a deep-learning-based model, comprising two bidirectional neural networks assembled by a partial stacking strategy, to automatically design and optimize three-dimensional chiral metamaterials with strong chiroptical responses

![](images/c8cdbaf582d28860ec3eb2a41590f65a05f1a262d735ea2c6bd44e4d59ff6b42.jpg)

at predesignated wavelengths. The model can help to discover the intricate, nonintuitive relationship between a metamaterial structure and its optical responses from a number of training examples, which circumvents the time-consuming, case-by-case numerical simulations in conventional metamaterial designs. This approach not only realizes the forward prediction of optical performance much more accurately and efficiently but also enables one to inversely retrieve designs from given requirements. Our results demonstrate that such a data-driven model can be applied as a very powerful tool in studying complicated light–matter interactions and accelerating the on-demand design of nanophotonic devices, systems, and architectures for real world applications.

KEYWORDS: deep learning, neural network, chirality, metamaterial, on-demand design

Chirality refers to the structural property of an object that cannot be superposed onto its mirror image. Due to its universal existence in nature, ranging from molecules at the nanoscale to gastropod shells at the macroscale, chirality has attracted immense research interest with important applications in spectroscopy, $^{1,2}$ sensing, $^{3,4}$ imaging, $^{5,6}$ and pharmaceutical synthesis. $^{7}$ Limited by the small electromagnetic interaction volume, the chiroptical response of natural materials is usually very weak and thus difficult to be detected with high sensitivity. $^{8}$ The advent of metamaterials, which are composed of artificial meta-atoms with elaborately engineered optical properties, $^{9,10}$ offers an elegant and effective solution to this problem. $^{11}$ So far, various intrinsic and extrinsic chiral metamaterials, both two-dimensional and three-dimensional ones, have been demonstrated. $^{12-19}$ Although a set of symmetry requirements derived from Jones matrices can guide the design of chiral metamaterials, $^{20,21}$ these guidelines are insufficient when we want to quantitatively design a metamaterial structure given a desired chiral response or even to simply predict the trend in chiral response as the structure transforms. The difficulties arise from the nonintuitive relationship between geometric chirality and chiroptical responses, which is attributed to the complicated interactions between different components in the sophisticated meta-atom under illuminations of either lefthanded circularly polarized (LCP) or right-handed circularly polarized (RCP) light. $^{22}$

An efficient and comprehensive design scheme for chiral and many other metamaterials should contain two major functions: forward prediction that outputs the full optical responses given the geometric parameters and inverse retrieval that outputs the geometric parameters from the required optical responses. Currently, the prediction task heavily relies on iterative, time-consuming numerical simulations to solve Maxwell's equations on a case-by-case basis, whereas the retrieval task remains extremely challenging with no general close-form solutions. Common approaches to the inverse design problem include genetic algorithm, $^{23}$ level set methods, $^{24}$ and topology optimization. $^{25}$ Nevertheless, the capability of such stochastic algorithms is severely limited by their nature of random search, and hence they are insufficient as the scale and complexity of the problem grow.

Different from numerical optimization approaches, data-driven methods based on machine learning (ML) can represent and generalize complex functions or data, to uncover unknown relations among a huge number of variables. Deep learning (DL) is a type of representation learning, allowing computational models to learn multiple levels of abstract representations of data layer by layer. $^{26}$ It has dramatically improved the state of the art in the domain of speech recognition, $^{27}$ visual object recognition, $^{28}$ as well as decision making, $^{29}$ with superior advantages to discover intricate structures in large data sets by using the back-propagation algorithm in training. Recently, propelled by its success in computer vision and natural language processing, DL has emerged as a revolutionary and powerful methodology in many other research fields such as materials science, $^{30}$ chemistry, $^{31}$ particle physics, $^{32}$ quantum mechanics, $^{33,34}$ and microscopy. $^{35}$ As the most widely used component in a DL architecture, neural networks have been applied to solve some design and prediction problems of electromagnetism $^{36-39}$ but with limited success largely because of the shallow structure and thus poor representation capability. Some very recent works have proposed deep neural networks to model nanophotonic structures, $^{40-42}$ which, however, are mainly constructed by stacking several fully connected layers and, therefore, can only deal with simple structure designs with limited optical responses.

![](images/52b6d78797315df71a11d04c1741bedbcf8054743f721676bc69a66a9828e2ec.jpg)
Figure 1. Schematic of the designed chiral metamaterial. The inset is the zoomed-in structure of a single meta-atom.

In this paper, we propose a purpose-designed deep-learning architecture to automatically model and optimize three-dimensional chiral metamaterials. The deep-learning scheme is composed of two bidirectional neural networks aiming to solve three basic tasks simultaneously, which are mutually connected through an ensemble learning strategy of partial stacking. In the forward modeling, it is a fast prototyping tool with high accuracy comparable to numerical simulations, to predict the full optical responses of a chiral structure under different polarization conditions. On the other hand, given the full optical responses, the network can be used to retrieve the geometric parameters of the chiral meta-atom to solve the inverse problem. Moreover, starting from some basic requirements on the frequency, amplitude, and polarity of the CD resonance with no specifications in full spectra, the deep-learning model can realize the design-on-demand function and produce suitable geometric parameters of the meta-atom to fulfill the given requirements. We envision that our work will develop a new paradigm for the design of optical metamaterials and nanophotonic structures, in general, in order to fully control the amplitude, phase, polarization, and trajectory of light on demand.

## RESULTS AND DISCUSSION

Chiral Metamaterial Structure. The chiral metamaterial under investigation is schematically shown in Figure 1. The unit cell consists of two stacked gold split ring resonators (SRRs) twisted at a certain angle and separated by two spacing dielectric layers with a continuous gold reflector at the bottom. Depending on the geometry, this metamaterial can function as a chiral meta-mirror, enabling selective reflection of designated circularly polarized light without reversing its handedness yet high absorption of the other polarization state at a certain wavelength. $^{21,43,44}$ The thickness and width of the gold SRRs are 50 and 200 nm, respectively, whereas the period of the unit cell is fixed at 2.5 $\mu$ m. Therefore, the structure of the chiral meta-atom is determined by five design parameters, that is, top SRR size $l_{1}$ , bottom SRR size $l_{2}$ , top spacer thickness $t_{1}$ , bottom spacer thickness $t_{2}$ , and the twisted angle $\alpha$ . As the continuous back reflector eliminates transmission, the chiroptical response of the metamaterial is completely described by its reflection coefficients. Considering two different polarization conditions and optical reciprocity, we will focus on three characteristic reflection spectra, that is, LCP-input-LCP-output (LL), RCP-input-RCP-output (RR), and the cross-polarization term LCP-input-RCP-output (RL) that is identical with RCP-input-LCP-output (LR). Here RCP (LCP) is defined if the electric vector rotates clockwise (counterclockwise) when an observer looks along the wave propagation direction. The reflection spectra of interest are set in the mid-infrared region from 30 to 80 THz and discretized into 201 data points. The details about data set collection can be found in the Methods section.

Deep-Learning Model Construction. The proposed deep-learning model for designing chiral metamaterials is schematically depicted in Figure 2. As indicated by the blue dashed boundaries, the model contains a primary network (PN) and an auxiliary network (AN), where both networks have a bidirectional configuration allowing data to flow in both a forward path and an inverse path. The two networks are assembled by a stacking strategy, $^{45}$ which not only improves the accuracy of prediction but also expands the model functions at the same time. The PN deals with the regression problem between design parameters with the dimension of $1 \times 5$ and the full reflection spectra with the dimension of $3 \times 201$ . The chiroptical response of the metamaterial can be characterized by circular dichroism (CD), which is defined as the absorption difference between LCP incidence and RCP incidence. We explicitly calculate CD from the three full reflection spectra for each chiral metamaterial structure and model it separately using the AN. The dataflow in the deep neural network is denoted by the yellow arrows in Figure 2, where all the internal nodes (design parameters, reflection spectra, and CD spectra, highlighted by the purple box) can be treated as either input or output nodes due to the nature of bidirectional mapping. PN and AN are connected at two ends through a forward combiner and an inverse combiner, as output ports for reflection spectra or design parameters with improved accuracy (green ellipses).

![](images/8dd5a8667347f12a7cc56e23f294b9ef594961afbbe6602d2b1c6a4b70ba714d.jpg)
Figure 2. Structure of the deep-learning model for designing chiral metamaterials. The model is composed of two bidirectional neural networks (PN and AN) assembled by a forward combiner and an inverse combiner. Reflection spectra, CD spectra, and design parameters are interconnected in the model (yellow arrows) and can be treated as either input or output at specific ports (fc, fully connected layer; conv, convolutional layer; tconv, transposed convolutional layer).

In the forward path of PN, design parameters with the dimension of $1 \times 5$ are transformed to reflection spectra with the dimension of $3 \times 201$ , indicating a very low input dimension compared with output dimension for a regression task. This huge mismatch makes it difficult for a network to converge and generalize well, especially when the output spectra have strong variation around resonant frequencies. Previous research has tried to avoid this problem by introducing binary input design parameters and using less data points in the output spectra. $^{40}$ In our approach, we adopt a heuristic two-stage structure containing a tensor module followed by an upsampling module, as indicated by the top red dashed boundary in Figure 2. Unlike fully connected layers that simply take linear combinations of the output from previous neurons, tensor layers can model a second-order relationship between a variable couple, which was first proposed for knowledge base completion $^{46}$ and was recently applied in chemistry. $^{47}$ Here, in our case, the tensor layer is used in a self-contained manner in order to describe the interdependency among the five design parameters. The output of the tensor layer is given by

$$
\mathrm{output} _ {\mathrm{tensor}} = f \big (D ^ {T} W _ {k} D + V _ {k} D + B \big)\tag{1}
$$

where f is the rectified linear unit (RELU) activation function, D is the row vector of five design parameters, k is the output vector dimension, $W_{k}$ is a $k \times 5 \times 5$ tensor, $V_{k}$ is a $k \times 5$ weight matrix, and B is a $k \times 1$ bias vector. The output dimension k is chosen to be 50 in our design, and the two following hidden layers both contain 500 neurons.

The tensor module is first trained in a supervised way, where the full $3 \times 201$ reflection data points are uniformly downsampled to three $1 \times 26$ vectors, equivalently increasing the frequency step from 0.25 to 4 THz. These subspectra, each represented by a $1 \times 26$ vector, are used as the ground truth for the pretraining of three parallel tensor units comprising a tensor layer followed by two fully connected layers. After the three parallel tensor units are trained, their outputs are merged into $3 \times 26$ subspectra and fed into the upsampling module to be converted to the full reflection spectra with the dimension increased from $3 \times 26$ to $3 \times 201$ . In the inverse path of PN, a typical convolutional neural network structure is employed with two consecutive convolutional layers followed by a fully connected layer. The inverse network is trained in the full loop of PN with the aid of the well-trained forward path, similar to the tandem training strategy recently reported, $^{42}$ but with the inverse error as a penalty term to enhance its robustness (see Supporting Information for details).

Model Evaluation. After the training of PN is finished, we use the test data that are unseen during training to evaluate the model. Figure 3a,d plot two typical examples from the test set for forward prediction of three reflection spectra, whereas the histograms in Figure 3c,f show the corresponding inverse retrieval of the five geometric parameters. Both forward and inverse paths of PN show reasonably good consistency compared with the numerical simulations, evidently establishing a bidirectional mapping between the geometric parameters and optical responses of the chiral metamaterial. Despite of the success of the purpose-designed PN, we notice that for the forward prediction, the model performance (dashed lines) degrades around resonant frequencies (i.e., 60 THz in Figure 3a and 65 THz in Figure 3d) with large output error. From a probabilistic viewpoint, the multilayer perceptron-based neural network assumes conditional independence among neurons in the same layer. For each neuron in the output layer, the probability distribution is centered at its off-resonance value, whereas its resonant value, highly deviated from the mean but with very few samples during training, can hardly be captured and modeled with high accuracy. On the other hand, as we use the typical mean square error loss function in this regression task that averages the error over the entire output, the contribution of the narrow-banded resonant dips can be easily diluted in the full spectra, leading the network to be trapped in a local optimum that only fits the off-resonance part of the reflection spectra very precisely.

![](images/4e095430e6917d580b6d1315da157a8d50da0aae92206c97494465c35af3ec42.jpg)

(c)
(f)
![](images/648a6a9296fa4d9dade3200a01c2ae0e85c9ba40d4b21904daf25df228d08328.jpg)
Figure 3. Evaluation of the deep-learning model. (a,d) Forward prediction results from PN, in comparison with AN-modified forward predictions. (b,e) CD spectra predicted by the forward path of AN. (c,f) Inverse prediction results from PN, in comparison with AN-modified inverse predictions.

In order to capture the fine spectra features around resonant frequencies and to substantially expand the functionality of the deep-leaning model at the same time, we explicitly create an AN to directly associate design parameters with CD spectra. The CD spectra are calculated from the three reflection spectra and down-sampled to 26 data points. As enclosed in the lower blue dashed boundary in Figure 2, AN also allows data to flow bidirectionally between the five-dimensional design parameters and the 26-dimensional CD spectra, but with simpler architecture mainly due to the less complexed data structure. The forward path has the same network structure as one of the tensor units in PN, with one tensor layer followed by two fully connected layers, whereas the inverse path consists of two fully connected hidden layers each containing 400 neurons. The prediction results of CD spectra by AN are shown in Figure 3b,e as blue dots, exhibiting very high consistency with numerical simulations.

To complete the entire deep-learning model after AN is fully trained, we use an ensemble learning approach of partial stacking strategy to combine PN and AN. In the forward path, the input design parameters simultaneously flow through PN and AN with two prediction outputs of full reflections and CD.

![](images/78384cf6161cfe01f6f16dcb36ccbbba0b6d59e8156d1ab830c3771f2a02faf5.jpg)

![](images/e81a13a9d3d4e2d500d022b488d827a86ddcec7cf25b86c22f620a3285f22f33.jpg)

![](images/ffaba7a8e99e353c9fe64191d48767a3fd7f690b75b5e3fe0cf53c35f6af8695.jpg)

![](images/4aa45d3b82850d11b5abb9ac70075ecce65b6029a645b53425e1d987c8e4a205.jpg)
Figure 4. Forward prediction results (a) and numerical simulation results (b) of the chiral metamaterial when changing the rotation angle from 0 to 180°. (c) CD value at the frequency of 72.25 THz when changing the rotation angle. (d) Normalized electrical field distribution in top and bottom SRRs at the frequency of 72.25 THz for two different rotation angles.

Then a forward combiner is created to locally modify the full reflection at a small spectra piece centered at the resonant frequency covering a range of 4 THz (17 data points), with the information provided by AN (see Supporting Information for details). The performance of the forward prediction given by the combiner is shown by the scattering points in Figure 3a,d, which agrees with numerical simulation much better than the original output from PN. The total test mean square error drops from 0.00087 to 0.00080 after the modification module is applied. Considering that the modified reflection differs from the original PN prediction only by 17 points, such improvement is significant when we check the spectra around resonant frequencies. For the inverse path, the combiner takes both PN and AN retrieved parameters as inputs and outputs a weighted sum of them, where the two weights are obtained by training on the given training data. Shown as the green bar in Figure 3c,f, the retrieval is improved overall when AN and the inverse combiner are introduced.

Predicting Chiroptical Responses. The completely established deep-learning model can be used as a fast prototyping tool to study the optical response of the chiral metamaterial. One example is to investigate the dependence of chirality on the rotation angle $\alpha$ between the top and bottom SRRs. Figures 4a,b demonstrate the evolution of CD spectra when varying the rotation angle from 0 to 180°, whereas the top SRR size, bottom SRR size, top spacer thickness, and bottom spacer thickness are fixed at 1000, 1300, 300, and 500 nm, respectively. The predicted CD spectra (Figure 4a) from the deep-learning model is calculated directly from the modified forward output, showing reasonable consistency with the numerical simulation (Figure 4b). The step of the rotation angle in the two density plots is 1°, which costs 9 h of numerical calculation using an i7-CPU personal computer but, in contrast, only takes less than 1 s for deep-learning prediction. Even though the training process takes some time, this is a one-time cost and will become insignificant if the model is repeatedly used.

Another very interesting finding is that the deep-learning model can give a fairly accurate prediction when trained only by a small number of data. $^{48}$ As we have five-dimensional design parameters, 25000 training data are equivalent to a sparsely sample 7.6 points in the continuously varying rotation angle between 0 and 180°. However, the model successfully predicts the exact frequency of 72.25 THz, where CD can reach maximum and minimum. The CD values at 72.25 THz (indicated by white dashed line in the density plot) as a function of rotation angle are plotted in Figure 4c, clearly showing excellent agreement between simulation and prediction, including the location and value of the extrema. This proves that the proposed deep-learning model does not simply perform averaging or interpolation but instead learns the highly nonlinear relationship between the rotation angle and CD value. To unveil the underlying physics of such high CD contrast originated from a relatively small change in rotation angle, the electric field distribution under incidence with different polarization conditions is explored (Figure 4d). For the chiral metamaterial with an 81° rotation angle, RCP incidence induces highly confined local fields around the SRRs, leading to high absorption. In contrast, LCP incidence is mostly reflected with no obvious field concentration. The net effect is a strongly negative CD value. On the contrary, the metamaterial structure with a 128° rotation angle confines and thus absorbs LCP light much more than RCP light, which results in a strongly positive CD.

![](images/2bbd5ea8c9941d9b3a740c4ec675c08f3197b2918de8069d7a30ab27e354855b.jpg)

![](images/530f1402bfa76055709c21a919fa8eaf38a15b79bc7c933fadb55c9bcae559c6.jpg)

Figure 5. Inverse design using the proposed deep learning model. (a,c) Desired, predicted, and simulated CD spectra. The insets list the retrieved geometric parameters. (b,d) Predicted full reflection spectra along with the full-wave simulation results.
![](images/efc3ddf055f596c6e7d8050025521ee533ec07a5f00f366958935b6378b96908.jpg)
Figure 6. Evolution of CD at 60 THz by varying the size of top and bottom SRRs at selected rotation angles. The white contour lines in the plot correspond to the CD of 0.5 or -0.5.

On-Demand Inverse Design of Chiral Metamaterials. In addition to retrieving the geometric parameters from the full reflection spectra, a practical inverse design scheme should also work when only a few requirements of chiral performance are provided. Such a function is highly desired if we intend to employ chiral metamaterials in sensing, imaging, and photodetection applications, $^{11,44,49-51}$ whereas the chiral response is only specified by a few figures of merit. In a CD spectrum, we can use resonant frequency $\omega_{0}$ , bandwidth $\Delta$ , and amplitude A to define one single resonant feature approximated by a Lorentz line shape. The complete CD response is given by summing up all resonances, as described below:

$$
\mathrm{CD} = \sum_ {i} \frac {A _ {i}}{1 + [ (\omega - \omega_ {0 , i}) / (\Delta_ {i} / 2) ] ^ {2}}\tag{2}
$$

The red solid curves in Figure 5a,c plot CD spectra with desired features defined by eq 2, which merely specify the location, intensity, and bandwidth of the resonances. Figure 5a shows a single CD resonance with A = -0.7, $\omega_{0} = 60$ THz, and $\Delta = 4$ THz, whereas Figure 5c represents the case of dual CD resonances with $A_{1} = 0.6$ , $\omega_{0,1} = 42$ THz, $\Delta_{1} = 3$ THz, $A_{2} = -0.6$ , $\omega_{0,2} = 66$ THz, and $\Delta_{2} = 3$ THz. As long as the prescribed requirements in the CD spectra are realizable for the metamaterial structure based on twisted SRRs, the proposed deep-learning model can retrieve the geometric parameters (insets in Figure 5a,c) that best approximate the requirements. The retrieved geometric parameters from AN pass through the entire deep-learning model, exporting the predicted CD spectra and full reflection spectra (Figure 5b,d) at the same time. The overall good agreement among the desired spectra, prediction, and full-wave simulation results manifests that the model can indeed solve the design-on-demand inverse problem accurately and efficiently.

Interestingly from Figure 5a, we notice that the retrieved rotation angle is as large as $172^{\circ}$ for a strong CD resonance at 60 THz. Because a metamaterial with mirror symmetry has no chiroptical response, $^{21}$ a natural logic is that a high degree of geometric chirality will yield strong optical chirality, as the case shown in Figure 4, where CD resonance happens at two rotation angles near $90^{\circ}$ . However, this intuition is not always correct as Figure 5a implies. To further explore the dependence of chiral response on the rotation angle, we choose to fix the thickness of the two spacer layers both at 400 nm and the frequency of interest at 60 THz. The density plot in Figure 6 shows the CD obtained from the DL model at different rotation angles when changing the size of top and bottom SRRs. We observe a fairly complementary CD distribution at two supplementary angles (i.e., 10 and $170^{\circ}$ , 30 and $150^{\circ}$ , as well as 60 and $120^{\circ}$ ). Notably, when top SRR size and bottom SRR size are around 1500 and 1050 nm, respectively, the metamaterial exhibits very strong CD at rotation angles near 0 and $180^{\circ}$ that correspond to the mirror symmetry in our structure. This result arises from the highly nonlinear relationship between geometric chirality and chiroptical responses. Equipped with the deep-learning model, we can conveniently and efficiently search the entire design space based on the prescribed requirements to uncover the complex evolution of chiral response as the geometric parameters change. The model can precisely predict the design with drastic change in chiral response when mirror symmetry is slightly broken. This counterintuitive result is more clearly demonstrated in Figure S1a, where the CD is plotted as a function of rotation angle (see Supporting Information for details).

## CONCLUSIONS

To conclude, we propose a purpose-designed deep-learning model to comprehensively study chiral metamaterials. The model has a two-level architecture and is trained heuristically with multiple functions of fast prototyping, optimization, and inverse design. On one hand, this model circumvents the computational burden required to numerically solve the differential equations that governs the underlying physics of chirality, basically shifting the methodology from the rule-based approach to data-driven approach. On the other hand, the model allows one to retrieve geometric parameters of the metamaterial from specific requirements on its optical responses, solving the inverse problem that even has no universally applicable solutions. The high efficiency and accuracy of the model make it a promising candidate in the research field of nanophotonics, where the complex light–matter interaction is often a barrier to model straightforwardly based on physical laws. More prominently, the proposed method could be readily extended to other research domains of optics and materials science, enabling on-demand designs and analyses with a broad range of applications such as sensing, imaging, and optical communications.

## METHODS

Numerical simulation package CST Microwave Studio is employed to generate the reflection spectra data by Monte Carlo sampling of the five design parameters uniformly distributed within certain ranges. The reflection spectra of interest are set in the mid-infrared region from 30 to 80 THz and discretized into 201 data points with a step of 0.25 THz. In the simulation, the spacer is modeled as a lossless dielectric with permittivity of 2, and gold is treated by the Drude model. $^{52}$ With the above parametrization, the design of the chiral metamaterial can be converted into a multivariable regression problem, aiming to link the $1 \times 5$ design vector with $3 \times 201$ reflection matrix. From the numerical simulations, we have collected 30000 samples and used 25000 of them for training and the remaining 5000 for testing. The model is constructed under the open-source machine learning framework of TensorFlow.

## ASSOCIATED CONTENT

## Supporting Information

The Supporting Information is available free of charge on the ACS Publications website at DOI: 10.1021/acsnano.8b03569.

Details about the structure and training of the deep-learning model; modeling of strong chiral metamaterial with slight breaking in symmetry (PDF)

## AUTHOR INFORMATION

## Corresponding Author

\*E-mail: y.liu@northeastern.edu.

## ORCID

Wei Ma: 0000-0002-5665-7840

Yongmin Liu: 0000-0003-1084-6651

## Notes

The authors declare no competing financial interest.

## ACKNOWLEDGMENTS

We acknowledge the financial support from the Office of Naval Research (N00014-16-1-2409). We thank Prof. Raymond Fu, Dr. Jun Li, and Dr. Yu Kong for helpful discussions.

## REFERENCES

(1) Greenfield, N. J. Using Circular Dichroism Spectra to Estimate Protein Secondary Structure. Nat. Protoc. 2007, 1, 2876.

(2) Micsonai, A.; Wien, F.; Kernya, L.; Lee, Y.-H.; Goto, Y.; Réfrégiers, M.; Kardos, J. Accurate Secondary Structure Prediction and Fold Recognition for Circular Dichroism Spectroscopy. Proc. Natl. Acad. Sci. U. S. A. 2015, 112, E3095–E3103.

(3) James, T. D.; Samankumara Sandanayake, K. R.; Shinkai, S. Chiral Discrimination of Monosaccharides Using a Fluorescent Molecular Sensor. Nature 1995, 374, 345.

(4) Torsi, L.; Farinola, G. M.; Marinelli, F.; Tanese, M. C.; Omar, O. H.; Valli, L.; Babudri, F.; Palmisano, F.; Zambonin, P. G.; Naso, F. A Sensitivity-Enhanced Field-Effect Chiral Sensor. Nat. Mater. 2008, 7, 412–417.

(5) Kang, L.; Rodrigues, S. P.; Taghinejad, M.; Lan, S.; Lee, K.-T.; Liu, Y.; Werner, D. H.; Urbas, A.; Cai, W. Preserving Spin States upon Reflection: Linear and Nonlinear Responses of a Chiral Meta-Mirror. Nano Lett. 2017, 17, 7102–7109.

(6) Ji, N.; Zhang, K.; Yang, H.; Shen, Y.-R. Three-Dimensional Chiral Imaging by Sum-Frequency Generation. J. Am. Chem. Soc. 2006, 128, 3482–3483.

(7) Savile, C. K.; Janey, J. M.; Mundorff, E. C.; Moore, J. C.; Tam, S.; Jarvis, W. R.; Colbeck, J. C.; Krebber, A.; Fleitz, F. J.; Brands, J.; et al. Biocatalytic Asymmetric Synthesis of Chiral Amines from Ketones Applied to Sitagliptin Manufacture. Science 2010, 329, 305–309.

(8) Wang, Z.; Cheng, F.; Winsor, T.; Liu, Y. Optical Chiral Metamaterials: A Review of the Fundamentals, Fabrication Methods and Applications. Nanotechnology 2016, 27, 412001.

(9) Shelby, R. A.; Smith, D. R.; Schultz, S. Experimental Verification of a Negative Index of Refraction. Science 2001, 292, 77–79.

(10) Liu, Y.; Zhang, X. Metamaterials: A New Frontier of Science and Technology. Chem. Soc. Rev. 2011, 40, 2494–2507.

(11) Hendry, E.; Carpy, T.; Johnston, J.; Popland, M.; Mikhaylovskiy, R.; Lapthorn, A.; Kelly, S.; Barron, L.; Gadegaard, N.; Kadodwala, M. Ultrasensitive Detection and Characterization of Biomolecules Using Superchiral Fields. Nat. Nanotechnol. 2010, 5, 783–787.

(12) Fedotov, V.; Mladyonov, P.; Prosvirnin, S.; Rogacheva, A.; Chen, Y.; Zheludev, N. Asymmetric Propagation of Electromagnetic Waves Through a Planar Chiral Structure. Phys. Rev. Lett. 2006, 97, 167401.

(13) Gansel, J. K.; Thiel, M.; Rill, M. S.; Decker, M.; Bade, K.; Saile, V.; von Freymann, G.; Linden, S.; Wegener, M. Gold Helix Photonic Metamaterial as Broadband Circular Polarizer. Science 2009, 325, 1513–1515.

(14) Zhang, S.; Park, Y.-S.; Li, J.; Lu, X.; Zhang, W.; Zhang, X. Negative Refractive Index in Chiral Metamaterials. Phys. Rev. Lett. 2009, 102, 023901.

(15) Zhao, Y.; Alù, A. Tailoring the Dispersion of Plasmonic Nanorods to Realize Broadband Optical Meta-Waveplates. Nano Lett. 2013, 13, 1086–1091.

(16) Cui, Y.; Kang, L.; Lan, S.; Rodrigues, S.; Cai, W. Giant Chiral Optical Response from a Twisted-Arc Metamaterial. Nano Lett. 2014, 14, 1021–1025.

(17) Kuzyk, A.; Schreiber, R.; Fan, Z.; Pardatscher, G.; Roller, E.-M.; Högele, A.; Simmel, F. C.; Govorov, A. O.; Liedl, T. DNA-Based Self-Assembly of Chiral Plasmonic Nanostructures with Tailored Optical Response. Nature 2012, 483, 311.

(18) Hentschel, M.; Schäferling, M.; Weiss, T.; Liu, N.; Giessen, H. Three-Dimensional Chiral Plasmonic Oligomers. Nano Lett. 2012, 12, 2542–2547.

(19) Kuzyk, A.; Schreiber, R.; Zhang, H.; Govorov, A. O.; Liedl, T.; Liu, N. Reconfigurable 3D Plasmonic Metamolecules. Nat. Mater. 2014, 13, 862.

(20) Menzel, C.; Rockstuhl, C.; Lederer, F. Advanced Jones Calculus for the Classification of Periodic Metamaterials. Phys. Rev. A: At., Mol., Opt. Phys. 2010, 82, 053811.

(21) Wang, Z.; Jia, H.; Yao, K.; Cai, W.; Chen, H.; Liu, Y. Circular Dichroism Metamirrors with Near-Perfect Extinction. ACS Photonics 2016, 3, 2096–2101.

(22) Li, Z.; Mutlu, M.; Ozbay, E. Chiral Metamaterials: From Optical Activity and Negative Refractive Index to Asymmetric Transmission. J. Opt. 2013, 15, 023001.

(23) Huntington, M. D.; Lauhon, L. J.; Odom, T. W. Subwavelength Lattice Optics by Evolutionary Design. Nano Lett. 2014, 14, 7195–7200.

(24) Kao, C. Y.; Osher, S.; Yablonovitch, E. Maximizing Band Gaps in Two-Dimensional Photonic Crystals by Using Level Set Methods. Appl. Phys. B: Lasers Opt. 2005, 81, 235–244.

(25) Andkjær, J.; Nishiwaki, S.; Nomura, T.; Sigmund, O. Topology Optimization of Grating Couplers for the Efficient Excitation of Surface Plasmons. J. Opt. Soc. Am. B 2010, 27, 1828–1832.

(26) LeCun, Y.; Bengio, Y.; Hinton, G. Deep Learning. Nature 2015, 521, 436–444.

(27) Hinton, G.; Deng, L.; Yu, D.; Dahl, G. E.; Mohamed, A.-r.; Jaitly, N.; Senior, A.; Vanhoucke, V.; Nguyen, P.; Sainath, T. N.; et al. Deep Neural Networks for Acoustic Modeling in Speech Recognition: The Shared Views of Four Research Groups. IEEE Signal Processing Mag. 2012, 29, 82–97.

(28) Krizhevsky, A.; Sutskever, I.; Hinton, G. E. ImageNet Classification with Deep Convolutional Neural Networks. Advances in Neural Information Processing SystemsNIPS'12, 2012; pp 1097–1105.

(29) Silver, D.; Huang, A.; Maddison, C. J.; Guez, A.; Sifre, L.; Van Den Driessche, G.; Schrittwieser, J.; Antonoglou, I.; Panneershelvam,

V.; Lanctot, M.; et al. Mastering the Game of Go with Deep Neural Networks and Tree Search. Nature 2016, 529, 484–489.

(30) Ramprasad, R.; Batra, R.; Pilania, G.; Mannodi-Kanakkithodi, A.; Kim, C. Machine Learning in Materials Informatics: Recent Applications and Prospects. npj Comput. Mater. 2017, 3, 54.

(31) Goh, G. B.; Hodas, N. O.; Vishnu, A. Deep Learning for Computational Chemistry. J. Comput. Chem. 2017, 38, 1291–1307.

(32) Baldi, P.; Sadowski, P.; Whiteson, D. Searching for Exotic Particles in High-Energy Physics with Deep Learning. Nat. Commun. 2014, 5, 4308.

(33) Carleo, G.; Troyer, M. Solving the Quantum Many-Body Problem with Artificial Neural Networks. Science 2017, 355, 602–606.

(34) Carrasquilla, J.; Melko, R. G. Machine Learning Phases of Matter. Nat. Phys. 2017, 13, 431.

(35) Rivenson, Y.; Göröcs, Z.; Günaydin, H.; Zhang, Y.; Wang, H.; Ozcan, A. Deep Learning Microscopy. Optica 2017, 4, 1437–1443.

(36) Kabir, H.; Wang, Y.; Yu, M.; Zhang, Q.-J. Neural Network Inverse Modeling and Applications to Microwave Filter Design. IEEE Trans. Microwave Theory Tech. 2008, 56, 867–879.

(37) Freitas, G.; Rego, S.; Vasconcelos, C. Design of Metamaterials Using Artificial Neural Networks. Proceedings of the Microwave & Optoelectronics Conference (IMOC); SBMO/IEEE MTT-S International; IEEE, 2011; pp 541–545.

(38) Vasconcelos, C. F.; Rêgo, S. L.; Cruz, R. M. The Use of Artificial Neural Network in the Design of Metamaterials. International Conference on Intelligent Data Engineering and Automated Learning; Springer, 2012; pp 532–539.

(39) Capizzi, G.; Lo Sciuto, G.; Napoli, C.; Tramontana, E. A Multithread Nested Neural Network Architecture to Model Surface Plasmon Polaritons Propagation. Micromachines 2016, 7, 110.

(40) Malkiel, I.; Nagler, A.; Mrejen, M.; Arieli, U.; Wolf, L.; Suchowski, H. Deep Learning for Design and Retrieval of Nanophotonic Structures. arXiv:1702.07949 2017.

(41) Peurifoy, J.; Shen, Y.; Jing, L.; Yang, Y.; Cano-Renteria, F.; Delacy, B.; Tegmark, M.; Joannopoulos, J. D.; Soljačić, M. Nanophotonic Particle Simulation and Inverse Design Using Artificial Neural Networks. Physics and Simulation of Optoelectronic Devices XXVI; International Society for Optics and Photonics, 2018; p 1052607.

(42) Liu, D.; Tan, Y.; Khoram, E.; Yu, Z. Training Deep Neural Networks for the Inverse Design of Nanophotonic Structures. ACS Photonics 2018, 5, 1365–1369.

(43) Plum, E.; Zheludev, N. I. Chiral Mirrors. Appl. Phys. Lett. 2015, 106, 221901.

(44) Li, W.; Coppens, Z. J.; Besteiro, L. V.; Wang, W.; Govorov, A. O.; Valentine, J. Circularly Polarized Light Detection with Hot Electrons in Chiral Plasmonic Metamaterials. Nat. Commun. 2015, 6, 8379.

(45) Wolpert, D. H. Stacked Generalization. Neural Networks 1992, 5, 241–259.

(46) Socher, R.; Chen, D.; Manning, C. D.; Ng, A. Reasoning with Neural Tensor Networks for Knowledge Base Completion. Advances in Neural Information Processing Systems 26 (NIPS 2013) 2013; pp 926–934.

(47) Schütt, K. T.; Arbabzadah, F.; Chmiela, S.; Müller, K. R.; Tkatchenko, A. Quantum-Chemical Insights from Deep Tensor Neural Networks. Nat. Commun. 2017, 8, 13890.

(48) Peurifoy, J.; Shen, Y.; Jing, L.; Yang, Y.; Cano-Renteria, F.; Tegmark, M.; Joannopoulos, J. D.; Soljacic, M. Nanophotonic Particle Simulation and Inverse Design Using Artificial Neural Networks. arXiv:1712.03222 2017, DOI: .

(49) Schäferling, M.; Dregely, D.; Hentschel, M.; Giessen, H. Tailoring Enhanced Optical Chirality: Design Principles for Chiral Plasmonic Nanostructures. Phys. Rev. X 2012, 2, 031010.

(50) Tkachenko, G.; Brasselet, E. Optofluidic Sorting of Material Chirality by Chiral Light. Nat. Commun. 2014, 5, 3577.

(51) Zhao, Y.; Saleh, A. A.; van de Haar, M. A.; Baum, B.; Briggs, J. A.; Lay, A.; Reyes-Becerra, O. A.; Dionne, J. A. Nanoscopic Control

and Quantification of Enantioselective Optical Forces. Nat. Nanotechnol. 2017, 12, 1055.

(52) Ordal, M. A.; Long, L. L.; Bell, R. J.; Bell, S. E.; Bell, R. R.; Alexander, J. R. W.; Ward, C. A. Optical Properties of the Metals Al, Co, Cu, Au, Fe, Pb, Ni, Pd, Pt, Ag, Ti, and W in the Infrared and Far Infrared. Appl. Opt. 1983, 22, 1099–1119.
