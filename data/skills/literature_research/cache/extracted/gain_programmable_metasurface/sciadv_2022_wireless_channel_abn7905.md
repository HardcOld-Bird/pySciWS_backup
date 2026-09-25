OPTICS

# Homeostatic neuro-metasurfaces for dynamic wireless channel management

Zhixiang Fan $^{1,2,3}$ , Chao Qian $^{1,2,3,*}$ , Yuetian Jia $^{1,2,3}$ , Zhedong Wang $^{1,2,3}$ , Yinzhang Ding $^{4}$ , Dengpan Wang $^{5}$ , Longwei Tian $^{6}$ , Erping Li $^{1,2,3}$ , Tong Cai $^{1,2,3,5}$ , Bin Zheng $^{1,2,3,*}$ , Ido Kaminer $^{7}$ , Hongsheng Chen $^{1,2,3,*}$

Copyright © 2022
The Authors, some rights reserved; exclusive licensee American Association for the Advancement of Science. No claim to original U.S. Government Works. Distributed under a Creative Commons Attribution License 4.0 (CC BY).

The physical basis of a smart city, the wireless channel, plays an important role in coordinating functions across a variety of systems and disordered environments, with numerous applications in wireless communication. However, conventional wireless channel typically necessitates high-complexity and energy-consuming hardware, and it is hindered by lengthy and iterative optimization strategies. Here, we introduce the concept of homeostatic neurometasurfaces to automatically and monolithically manage wireless channel in dynamics. These neuro-metasurfaces relieve the heavy reliance on traditional radio frequency components and embrace two iconic traits: They require no iterative computation and no human participation. In doing so, we develop a flexible deep learning paradigm for the global inverse design of large-scale metasurfaces, reaching an accuracy greater than 90%. In a full perception-decision-action experiment, our concept is demonstrated through a preliminary proof-of-concept verification and an on-demand wireless channel management. Our work provides a key advance for the next generation of electromagnetic smart cities.

## INTRODUCTION

Smart city is a generic term used to describe an urban area that leverages information and communication technologies to optimize transportation systems, social sustainability, resource allocation, and other community services (1). In particular, recent decades have witnessed an unprecedented promotion because of the big advances in the Internet of Things (IoT) and artificial intelligence (2). Much effort has been inaugurated to pursue intellectualization at the data link level and network level (3–4). However, the physical level, the wireless channel, an electromagnetic (EM) link between the transmitter and the receiver with complex propagations inside, also plays an important role (5). As conceptualized in Fig. 1, if we can manage the wireless channel as desired, then a radically new EM smart city/infrastructure could be created (6). In such a vision, many intriguing applications will be facilitated. For example, one can physically cancel off the signal sent from the base station at the location of an eavesdropper, thus effectively reducing information leakage.

A fundamental backbone to create the aforementioned vision is to physically manage the wireless channel and modify the EM environment even in a disordered surrounding (7). Conventionally, the engineering of wireless channel typically necessitates high-complexity and energy-consuming hardware at base stations, and it is limited by lengthy and iterative optimization strategies (8–9). In this context, densely deploying wireless relay techniques with signal regeneration and retransmission will entail intensifying hardware expenditures and maintenance costs. In the past years, metasurfaces, an artificial wave-functional interface composed of arrays of subwavelength resonators, have attracted extensive attention for manipulating wireless channels in a green and cost-effective manner (10–11). By introducing spatiotemporally varying optical response into metasurfaces, we can arbitrarily shape wavefront and adjust polarization state of the impinging waves. Thus far, metasurfaces are being anticipated to bring a new twist in fifth-generation (5G) wireless communication (12), intelligent reflection surface (13), imaging recognition (14), and beyond (15–17).

However, the related metasurface-based works share a common limitation: either they are static in nature (set in stone after fabrication) or work in a trial-and-error mode to satisfy user demands (dependent and iterative) $(18)$ . If one wants to overcome this, then a necessary step is to quickly unlock and streamline the intricate interactions among metasurfaces, dynamic environment, and user demands. Deep learning, as a powerful data-driven method, has recently been welcomed to expedite the on-demand design of metamaterials $(19–27)$ and photonic crystals $(28–30)$ . The state-of-the-art works can be divided into two categories: accurately encapsulate optical responses for a given structure (forward prediction) and inversely design physical structures for a given optical response (inverse design) $(31–32)$ . Compared with full-wave numerical simulations and physics-based approaches, deep learning has found to be efficient, time-saving, and reusable because it is able to unearth obscure optical characteristics and latent physics from a suitable amount of data $(33)$ . For wireless channel management, however, the related deep learning works may become inefficient and even invalid because of the following reasons. First, these works are mostly limited to sub-wavelength and individual elements (local design, ignoring the coupling effects among adjacent elements), while for large-scale metasurfaces, the streamline design is in high demand but still elusive (global design). Second, a majority of works only focus on the analytical design of passive elements, while the reconfigurable explorations in experiment are scarce, let alone to an unmanned experiment (34).

![](images/df26948ee5d464192fb5817407aad610b29b890fd2893d3ee7ffed8101d765de.jpg)
Fig. 1. Illustrative scenario of homeostatic/self-acting neuro-metasurfaces in an EM smart city and infrastructures. A homeostatic neuro-metasurface mounted on the bus stop monolithically manages the wireless channel in a disordered and dynamic environment. A myriad of scenarios can thus be envisioned, such as compensating for signal loss, localizing a moving target, and extending signal coverage in a dead zone. Various homeostatic neuro-metasurfaces can also team up for collaborative work or communicate with third parties via physical layer connection. This scheme reduces the energy consumption and hardware cost over conventional setup, such as oscillators, mixers, and amplifiers, and thus enables a large advance toward flexible and efficient IoT devices.

To overcome these limitations, we propose the concept of homeostatic/self-acting neuro-metasurfaces to globally manage wireless channel during the propagation process and automatically cater to user demands in dynamic environment. Homeostatic neurometasurfaces are a deep learning-driven planar array consisting of a large number of active elements; each active element imparts an independent amplitude/phase/polarization modulation to the incident waves. In this work, we consider a mechanical-actuating neurometasurface, and for each neuro-element, the reflection phase is separately tuned by mechanical rotation. With a moderate number of data and associated data augmentation technique, a generative model for global inverse design is developed. Compared with conventional adaptive strategies, the homeostatic neuro-metasurfaces involve only one single-forward computation, thus saving a notable amount of computational time (35). In experiments, we build up a complete perception-decision-action system to mimic real-world scenarios and carry out two progressive experiments: A proof-of-concept experiment to verify the global inverse design model and an on-demand wireless channel management experiment. Our work opens a new avenue for the next generation of EM smart infrastructure and, more generally, pushes metasurfaces to a new horizon, empowering inanimate metasurfaces with human wisdom (9, 36).

## RESULTS

## Neuro-metasurfaces enabled EM smart cities and infrastructures

The applications of neuro-metasurfaces are abundant with the merits of low profile, light weight, and conformal geometry. As an epitome of EM smart cities, Fig. 1 depicts a vision of future bus stop, where homeostatic neuro-metasurfaces are mounted on a shed to holistically humanize the wireless channel. Several typical scenarios are illustrated. For example, in a dead zone (the direct link between users and the base station is blocked), neuro-metasurfaces can be leveraged as transfer stations to create a virtual line-of-sight link. For directional signal propagation, neuro-metasurfaces can adaptively adjust each element to strengthen the desired signal or suppress the undesired signal. For wireless communication, one can encode information into spatiotemporal sequences to tune neuro-metasurfaces in both time and space (space- and frequency-division multiplexing) (12). Furthermore, multiple neuro-metasurfaces and even third-party devices can be interfaced through physical layer connection for collaborative works. In these examples, neuro-metasurfaces can be flexibly deployed in wireless networks and relieve the heavy reliance on traditional radio frequency components, making a big step toward low-cost and green IoT devices.

To reach this goal, the neuro-metasurface architecture should include three key components: perception, decision-making, and action. We mimic them by EM detector, deep learning algorithm, and reconfigurable metasurfaces, respectively. For most of the metasurface-related devices, they typically work for a specific EM mode that is assumed to be a priori known (37–38). However, in practice, these factors may vary all the time, making a predesigned metasurface device inefficient. We treat these factors (acquired by EM detector) as input parameters of a deep learning algorithm, together with a customer-defined wireless channel. According to the two sets of inputs, the deep learning algorithm will quickly generate candidate metasurface patterns. Although programmable and reconfigurable metasurfaces have been extensively studied and underpin the reconfigurability of some established technologies, they also need to work in tandem with outside helps and in a trial-and-error mode for a customer-specific requirement.

## Global inverse design of homeostatic neuro-metasurfaces

For the core of the neuro-metasurface architecture, we consider deep learning algorithm to bridge wireless channel to large-scale neurometasurfaces. This remains out of reach for the existing inverse design works because they are mostly limited to individual subwavelength elements/unit cells (39). These works buildup a bidirectional channel between physical structures and their local optical properties. However, they inherently neglect the nonlinear interactions and elusive couplings between adjacent structures because of the imperfect theoretical assumption (40–41), and for large-scale metasurfaces, the number of degrees of freedom increases. The seemingly simple increase will make the whole solution space expand exponentially, leading to a burdensome task on dataset collection and algorithm modeling. This thorny issue also widely exists in other numerous applications, such as multidimensional physical crystals and multipixel holograms (42–43).

To tackle the above challenges, the pipeline of data preprocessing and the generative network structure are proposed in Fig. 2. First, we directly consider full-wave simulation data or experimental measured data of neuro-metasurfaces as training data; this way, the complicated interaction between adjacent structures is involved. Then, we decompose each radiation pattern (expressed in polar coordinates) into three feature extraction pipes (pattern, upmask, and submask pipes) and set the frequency, incidence, and polarization of incident waves as the other set, each of which has a dimension of $200 \times 200$ (Fig. 2A). The pattern pipe is the raw radiation pattern from a simulation or experiment. The upmask and submask pipes represent the 3- and 15-dB down boundaries of the pattern pipe, which represent the half power beam width of the radiation pattern and the radiation pattern lobes and side lobe levels, respectively. Compared with traditional methods that take a data matrix as input directly, the representation of polar coordinates and three feature extraction pipes not only preserves the inherent spatial association information but also preprocesses the EM features effectively. In this manner, the characteristics of the radiation pattern can be easily extracted with fast convergence (44).

Regarding the network architecture, a typical encoder-decoder structure comprising convolutional layers, pooling layers, and batch normalization layers is designed. Owing to its hierarchical structure, such a network structure promises rapid inference, strong generalization, translation, and scale invariance (45), making it a superior candidate. For a specific task, the performance of accuracy and the decline of loss are comprehensively considered to determine the encoder-decoder architecture. To facilitate the training process, we identify two evaluation indices, i.e., the cross-entropy loss and accuracy rate (46), to minimize the inconsistencies of the $20 \times 20$ neurometasurface profiles in the ground truth and predicted cases with a labeled state 0/1/2/3; see note S3 for the details of the neural network.

![](images/9f31dbec7a7818e485b354520645713435ad9a5854e42c52f8787596355f769f.jpg)
Fig. 2. Global inverse design of homeostatic neuro-metasurfaces. (A) Data preprocessing. The input wireless channel is decomposed into two sets of inputs including input 1 (frequency, incident angle, and polarization pipes) and input 2 (upmask, submask, and pattern pipes) to effectively extract signal characteristics. All pipes are expressed in polar coordinates. The upmask and submask pipes represent the 3- and 15-dB down boundaries of the pattern channel, respectively. (B) Global inverse design model. The input is a user-favored wireless channel, and the output is the reconfiguration of homeostatic neuro-metasurface. The encoder-decoder structure is composed of convolutional layers, transposed convolutional layers, pooling layers, batch normalization layers, and so on; see note S3 for a detailed description.

Experimental buildup of homeostatic neuro-metasurfaces

In experiments, we build up a full set of intelligent systems that integrate the functionalities of perception, decision, and action, as shown in Fig. 3. For the perception component (Fig. 3A), we deploy an eight-port polarization and incident angle-sensitive antenna array to directly extract the full parameters of the incoming wave, assisted by a generalized regression neural network; see note S7 and Methods (47). For the action component (Fig. 3B), we introduce a mechanical neuro-metasurface, and each element provides an independent local reflection response via a micromotor (with a rated speed of 2000 rpm and rated power of 250 mW). Compared with conventional phased antenna, the power consumption of the neuro-metasurfaces is \~25%; see details in note S6. Compared with the lumped element tuning approach [e.g., an SMV2019-079LF commercial varactor diode has a power dissipation of 250 mW (34)], although their power consumptions are similar, mechanical neuro-metasurfaces execute geometric actuation only in one step without continuous energy supply (nonvolatile advantage). This way, the heat dissipation issue could be relieved to some extent, and the anti-jamming capability could be lifted in volatile environments (48).

Without loss of generality, we design a mechanical neuro-metasurface consisting of double C-shape elements (49); see the schematic diagram in Fig. 4A and the geometrical parameters in fig. S1. The dimensions of the whole neuro-metasurfaces are 160 mm by 160 mm by 2 mm (20 × 20 unit cells), working within 13.1 to 13.5 GHz. For each neuro-element, two identical circular metallic patches and a central axis are etched on a 2-mm-thick F4B substrate (the relative permittivity is $\varepsilon_{r} = 2.65$ and the loss tangent is $\tan\delta = 0.009$ ). In Fig. 4A, the simulated results show that the reflected phase of the mechanical neuro-metasurfaces can be tuned in a wide range by adjusting the rotation angle of the circular metallic patch for cross-polarized wave. For the varied reflected amplitude, it only slightly affects the radiation gain, whereas the shape of the radiation remains almost identical (fig. S6). To facilitate the training of deep learning, we consider four discrete states with rotation angles of $0^{\circ}, 20^{\circ}, 40^{\circ}$ , and $60^{\circ}$ , corresponding to the phase shifts of $159^{\circ}$ , $128^{\circ}$ , $89^{\circ}$ , and $-37^{\circ}$ . The radiation pattern with different phase quantization levels is also analyzed in note S5. We find that the radiation pattern with the four discrete states is close to that with the ideal continuous phase level. The computing time of neural network with different phase discretization level is similar ( $\sim20$ ms).

A
EM detector
B
Neuro-metasurfaces
![](images/33e07691f19ba90c83e0337d4bf73972a613ef0db9265225e8e7796648099981.jpg)
Fig. 3. Perception-decision-action experimental setup. (A) Fabricated prototype of a home-made intelligent EM detector. The operating mechanism of the intelligent detector is shown on the right. An induced voltage sequence is input into the general regression neural network (GRNN) algorithm to directly obtain a complete list information of incoming wave, including frequency, incident angle, and polarization state (47). (B) Fabricated prototype of the mechanical neuro-metasurfaces. Each constituent neuro-element is independently controlled by a micromotor. (C) A rendered photograph of the homeostatic neuro-metasurface architecture in experiment. It mainly includes an intelligent EM detector (perception), a deep learning-driven computing core (decision), and mechanical neuro-metasurfaces (action). The three parts are teamed up to automatically cater to user demands and offset surrounding dynamics. RF, radio frequency; SDR, software-defined radio.

![](images/3c01a08b702102668d207be0bce086d6f331686677745e5d25e1f7832483f0de.jpg)
Ground truth (experiment) Deep learning prediction (experiment)

Fig. 4. Experimental verification of homeostatic neuro-metasurfaces. (A) Reflected response of the neuro-metasurfaces at 13.4 GHz. The result is obtained for cross-polarized wave when the neuro-metasurfaces are illuminated by the x-polarized plane wave. Each element imparts a local phase (blue line) and amplitude (red line) change to the input wave as a function of the rotation angle $\alpha$ . The inset shows a schematic diagram of the neuro-element; see note S1 for details. a.u., arbitrary units. (B) Training results over the epochs. The accuracies of both the training and test sets exceed 90%, indicating that the trained network is without much overfitting. (C) Confusion matrices of the three randomly selected testing instances. The accuracies are 94, 97.5, and 98.25%, which is defined by the number of correctly predicted labels divided by the total number of labels. (D) Experimental results in the xoz plane for the three pairs of neuro-metasurfaces.

## On-demand wireless channel management with homeostatic neuro-metasurfaces

By using Computer Simulation Technology (CST) Microwave Studio software and mixed-sample data augmentation (see Methods), 84,400 simulated far-field/wireless channel data at different frequencies are collected and then separated into training (80%), validation (10%), and test sets (10%). The wireless channel data are normalized and shuffled before being fed into the neural network. A classic VGG-16 architecture is identified as the base architecture to implement our encoder-decoder structure (Fig. 2B). The accuracy rates on both the training and test sets exceed 90%, indicating that the pretrained encoder-decoder is reliable without much overfitting (Fig. 4B). To test our global inverse network experimentally, we blindly choose three wireless channel characteristics from the test set (the corresponding frequencies are 13.2, 13.3, and 13.5 GHz). The selected far fields are inversely designed to generate candidate neuro-metasurface pattern with the accuracies of 94, 97.5, and 98.25%, respectively, as shown in Fig. 4C. We then measured their radar cross section (RCS), $\sigma = 2\pi\rho_{0} |E_{t} - E_{in}|^{2}/|E_{in}|^{2}$ , where $E_{t}$ and $E_{in}$ are the observed electric field and incident electric field and $\rho_{0}$ is set to 1.3 m for the far-field approximation. The high consistency between the RCS curves of the ground-truth and the neuro-metasurfaces in Fig. 4D lays a foundation for the following on-site experiment. The experimental efficiency (about 75%) of neuro-metasurfaces is calculated in note S11.

We then progress to an on-demand wireless channel management with homeostatic neuro-metasurfaces. In addition to the incoming signal obtained by the EM detector, we also use a camera to sense the movement of pedestrians. We consider a real-world scenario near a shopping mall in the city of Hangzhou (movie S1) and randomly extract three dynamics at $t = 3^{\prime\prime}29$ , $11^{\prime\prime}06$ , and $17^{\prime\prime}04$ , as schematically shown in Fig. 5A. In these scenarios, the homeostatic neuro-metasurfaces are assumed to be arranged on a lateral wall, with the aim of dynamically managing the wireless channel without human intervention. The wireless channel is humanized according to both the locations of pedestrians and the EM waves impinging onto the neuro-metasurface. Figure 5B shows the neuro-metasurface patterns predicted by the global inverse design algorithm, and Fig. 5C shows the measured wireless channel (xoy plane) together with the customized wireless channel. The high consistency in Fig. 5C strongly validates the generality of the global inverse design algorithm and the robustness of mechanical metasurfaces. Furthermore, we carry out an experiment for an off-the-shelf scenario, real-time localizing the public bus in Fig. 1 when it turns left. The homeostatic neuro-metasurfaces also exhibit accurate and agile tracking ability; see movie S2. Other complex situations are left in note S10, such as the power management for different user distances and the simultaneous changes of radiation pattern and frequency, and we show that the reflection spectra of the neuro-metasurfaces are relatively robust when the incident angle varies between $-30^{\circ}$ and $30^{\circ}$ .

A

![](images/e015b3a3740b5232e942e7b3a442a58d97457a952738b7dcb51252cd5632a6a2.jpg)
Fig. 5. On-demand wireless channel management with homeostatic neuro-metasurfaces. (A) Dynamics of an on-site video at $t = 3^{\prime \prime}29$ , $11^{\prime \prime}06$ , and $17^{\prime \prime}04$ ; see movie S1 for the full video. The homeostatic/self-acting neuro-metasurfaces, decorated on a lateral wall, aims to enhance outdoor signals at the desired receivers or destructively at the non-intended receivers. The bottom-right inset shows the crucial signal directivities according to the locations and numbers of pedestrians. (B) Homeostatic neuro-metasurface reconfigurations output by the global inverse design algorithm. The homeostatic neuro-metasurfaces are characterized into four states with the rotation angles of $0^{\circ}, 20^{\circ}, 40^{\circ}$ , and $60^{\circ}$ . (C) Experimental wireless channel results (xoy plane) of the homeostatic neuro-metasurfaces and the customized situation.

## DISCUSSION

In conclusion, we have synergized mechanical-actuating metasurfaces with deep learning to usher in an era of EM smart cities and infrastructures and for indoor applications with a high density of users, such as in stadiums and airports (6). No human intervention and iterative computation are involved in the experiment. On a fundamental level, we bestow the neuro-metasurfaces with the generic ability to independently analyze and solve problems, rather than fixed functionalities for fixed environments and incoming waves. This self-learning ability is of paramount importance for numerous applications, such as invisibility cloaks and biological imaging in random media (50). A similar concept can be readily extended to higher frequencies with the proposed global inverse metasurface design and mature micro/nano fabrication technology, such as microelectromechanical systems technique and phase-change materials. We believe that the global inverse design strategy is a unique advantage, which, in combination with optical active metasurfaces, may be key to making various intelligent metadevices (51).

Looking forward, it would be nothing short of astonishing to migrate scenarios, e.g., from smart cities to smart offices, by sharing common experiences and parameters in transfer learning (52). Another meaningful improvement would be the use of semisupervised and unsupervised learning, which would largely relax the high reliance on massive data collection for even larger-scale neuro-metasurfaces. For a more general open-loop operation system, the on-site learning working mode can be applied to regulate wireless channel, providing robustness to unexpected stimuli (53–54). In turn, we also anticipate that the homeostatic neuro-metasurfaces will accelerate deep learning algorithm in optics by harnessing the advantages of parallel computing and speed-of-light operation (15).

## METHODS

## Data generation

The training data are obtained with the commercial software CST Microwave Studio. For the numerical simulation, the actual structures for the designed metasurfaces are adopted. A total of 84,400 sets of metasurfaces are generated in the multiparadigm numerical computing tool MATLAB and then transferred into the commercial software package CST Microwave Studio for continuous automatic full-wave simulations via the MATLAB-CST cosimulation method.

## Augmentation technique

The dataset is expanded by the mixed-sample data augmentation method and random variation data augmentation method. The core idea of the mixed-sample data augmentation method is to randomly mix two training labels (metasurface arrangements) at a certain rate to generate new data, while the random variation data augmentation method involves changing the training labels randomly at a certain variation rate. Starting with 25 representative training labels, the dataset is expanded to 84,400 items by using the method described above with a mixing rate and variation rate that vary from 0 to 100%, thus effectively increasing the diversity of the samples and improving the robustness of the model. During the training period, Gaussian noise and random rotation are randomly applied to enhance the generalization capability of the model.

## Intelligent EM detector

A homemade intelligent EM detector is composed of an eight-port antenna array for simultaneously attaining frequency, directional-of-arrival, and polarization data. The eight-port metasurface antenna array is connected to two radio frequency switches (HMC641ALC4) and used to collect the amplitude-only sequence from ports 1 to 8 on a microsecond time scale. The received signal is amplified by a broadband amplifier and down-converted to 0.2 to 4 GHz. Then, we use an AD9361 as the radio frequency processor, which contains a low-noise amplifier, mixer, and other electric components, and use a Xilinx Zynq for data processing with a calculation accelerator assisted by Field Programmable Gate Array (FPGA). On the basis of the collected data, a generalized regression neural network outputs the frequency, directional of arrival, and polarization of the received signal; see note S7. The entire detection takes about 60 ms, including 35 ms for frequency sweeping, 10 ms for the machine learning calculation, and 15 ms consumed by other data-processing algorithms, such as fast Fourier transform and median filter.

## Experimental measurement

The experiment is carried out in an anechoic chamber, which mainly includes a transmitting horn antenna, a receiving horn antenna, and an intelligent EM detector. In far-field measurements, both the transmitting and receiving horn antennas are fixed on an arch-shaped bracket with a radius of 1.3 m and digitally controlled to rotate within 0- $\pi$ . The receiving horn antenna is connected to a vector network analyzer to detect the scattered field, including the amplitude and phase information. The far-field experimental setup is shown in note S8.

## Running time

The total consuming time of the neuro-metasurfaces includes three parts: detection time ( $\sim$ 60 ms), calculation time ( $\sim$ 20 ms), and execution time ( $\sim$ 5 ms), when the complexity of the input/environment complexity increase does not affect the action time greatly. For example, when the dimension of input channel increases from $200 \times 200$ to $400 \times 400$ , the action time only increases from $\sim$ 20 to $\sim$ 25 ms (with an additional convolutional layer).

## SUPPLEMENTARY MATERIALS

Supplementary material for this article is available at https://science.org/doi/10.1126/sciadv.abn7905

## REFERENCES AND NOTES

1. A. Zanella, N. Bui, A. Castellani, L. Vangelista, M. Zorzi, Internet of things for smart cities. IEEE Internet Things J. 1, 22–32 (2014).

2. C. Kaspar, B. J. Ravoo, W. G. van der Wiel, S. V. Wegner, W. H. P. Pernice, The rise of intelligent matter. Nature 594, 345–355 (2021).

3. S. Mohanty, U. Choppali, E. Kougianos, Everything you wanted to know about smart cities: The Internet of things is the backbone. IEEE Consum. Electron. Mag. 5, 60–70 (2016).

4. E. Basar, Reconfigurable intelligent surface-based index modulation: A new beyond MIMO paradigm for 6G. IEEE Trans. Wirel. Commun. 68, 3187–3196 (2020).

5. H. Yang, Z. Xiong, J. Zhao, D. Niyato, L. Xiao, Q. Wu, Deep reinforcement learning-based intelligent reflecting surface for secure wireless communications. IEEE Trans Wirel Commun. 20, 375–388 (2021).

6. R. Napolitano, W. Reinhart, J. P. Gevaudan, Smart cities built with smart materials. Science 371, 1200–1201 (2021).

7. P. Del Hougne, M. Fink, G. Lerosey, Optimally diverse communication channels in disordered environments with tuned randomness. Nat. Electron. 2, 36–41 (2019).

8. Q. Ma, G. D. Bai, H. B. Jing, C. Yang, L. Li, T. J. Cui, Smart metasurface with self-adaptively reprogrammable functions. Light Sci. Appl. 8, 98 (2019).

9. O. Tsilipakos, A. C. Tasolamprou, A. Pitilakis, F. Liu, X. Wang, M. S. Mirmoosa, D. C. Tzarouchis, S. Abadal, H. Taghvaee, C. Liaskos, A. Tsiolaridou, J. Georgiou, A. Cabellos-Aparicio, E. Alarcón, S. Ioannidis, A. Pitsillides, I. F. Akyildiz, N. V. Kantartzis, E. N. Economou, C. M. Soukoulis, M. Kafesaki, S. Tretyakov, Toward intelligent metasurfaces: The progress from globally tunable metasurfaces to software-defined metasurfaces with an embedded network of controllers. Adv. Opt. Mater. 8, 2000783 (2020).

10. N. Yu, F. Capasso, Flat optics with designer metasurfaces. Nat. Mater. 13, 139–150 (2014).

11. T. Cai, G. M. Wang, S. W. Tang, H. X. Xu, J. W. Duan, H. J. Guo, F. X. Guan, S. L. Sun, Q. He, L. Zhou, High-efficiency and full-space manipulation of electromagnetic wave fronts with metasurfaces. Phys. Rev. Appl. 8, 034033 (2017).

12. L. Zhang, M. Z. Chen, W. Tang, J. Y. Dai, L. Miao, X. Y. Zhou, S. Jin, Q. Cheng, T. J. Cui, A wireless communication scheme based on space- and frequency-division multiplexing using digital metasurfaces. Nat. Electron. 4, 218–227 (2021).

13. Q. Wu, R. Zhang, Towards smart and reconfigurable environment: Intelligent reflecting surface aided wireless network. IEEE Commun Mag. 58, 106–112 (2020).

14. H. Ren, X. Fang, J. Jang, J. Bürger, J. Rho, S. A. Maier, Complex-amplitude metasurface-based orbital angular momentum holography in momentum space. Nat. Nanotechnol. 15, 948–955 (2020).

15. C. Qian, X. Lin, X. Lin, J. Xu, Y. Sun, E. Li, B. Zhang, H. Chen, Performing optical logic operations by a diffractive neural network. Light Sci. Appl. 9, 59 (2020).

16. G. Qu, W. Yang, Q. Song, Y. Liu, C. W. Qiu, J. Han, D. P. Tsai, S. Xiao, Reprogrammable meta-hologram for optical encryption. Nat. Commun. 11, 5484 (2020).

17. N. M. Estakhri, B. Edwards, N. Engheta, Inverse-designed metastructures that solve equations. Science 363, 1333–1338 (2019).

18. J. Jiang, M. Chen, J. A. Fan, Deep neural networks for the evaluation and design of photonic devices. Nat. Rev. Mater. 6, 679–700 (2021).

19. Z. Liu, D. Zhu, K. T. Lee, A. S. Kim, L. Raju, W. Cai, Compounding meta-atoms into metamolecules with hybrid artificial intelligence techniques. Adv. Mater. 32, 1904790 (2020).

20. W. Ma, Y. Liu, A data-efficient self-supervised deep learning model for design and characterization of nanophotonic structures. Sci. China Phys. Mech. 63, 284212 (2020).

21. H. Ren, W. Shao, Y. Li, F. Salim, M. Gu, Three-dimensional vectorial holography based on machine learning inverse design. Sci. Adv. 6, eaaz4261 (2020).

22. P. R. Wiecha, O. L. Muskens, Deep learning meets nanophotonics: A generalized accurate predictor for near fields and far fields of arbitrary 3D nanostructures. Nano Lett. 20, 329–338 (2020).

23. W. Ma, F. Cheng, Y. Xu, Q. Wen, Y. Liu, Probabilistic representation and inverse design of metamaterials based on a deep generative model with semi-supervised learning strategy. Adv. Mater. 31, 1901111 (2019).

24. W. Ma, F. Cheng, Y. Liu, Deep-learning enabled on-demand design of chiral metamaterials. ACS Nano 12, 6326–6334 (2018).

25. W. Ma, Y. Xu, B. Xiong, L. Deng, R. W. Peng, M. Wang, Y. Liu, Pushing the limits of functionality—Multiplexing capability in metasurface design based on statistical machine learning. Adv. Mater. 34, 2110022 (2022).

26. L. Raju, K. T. Lee, Z. Liu, D. Zhu, M. Zhu, E. Poutrina, A. Urbas, W. Cai, Maximized frequency doubling through the inverse design of nonlinear metamaterials. ACS Nano 16, 3926–3933 (2022).

27. X. Chen, Z. Wei, M. Li, P. Rocca, A review of deep learning approaches for inverse scattering problems (Invited review). Prog. Electromagn. Res. 167, 67–81 (2020).

28. Z. Liu, D. Zhu, S. P. Rodrigues, K. T. Lee, W. Cai, Generative model for the inverse design of metasurfaces. Nano Lett. 18, 6570–6576 (2018).

29. L. Gao, X. Li, D. Liu, L. Wang, Z. Yu, A bidirectional deep neural network for accurate silicon color design. Adv. Mater. 31, 1905467 (2019).

30. G. Wetzstein, A. Ozcan, S. Gigan, S. Fan, D. Englund, M. Soljačić, C. Denz, D. A. B. Miller, D. Psaltis, Inference in artificial intelligence with deep optics and photonics. Nature 588, 39–47 (2020).

31. Y. LeCun, Y. Bengio, G. Hinton, Deep learning. Nature 521, 436–444 (2015).

32. R. Iten, T. Metger, H. Wilming, L. del Rio, R. Renner, Discovering physical concepts with neural networks. Phys. Rev. Lett. 124, 010508 (2020).

33. W. Ma, Z. Liu, Z. A. Kudyshev, A. Boltasseva, W. Cai, Y. Liu, Deep learning for the design of photonic structures. Nat. Photon 15, 77–90 (2021).

34. C. Qian, B. Zheng, Y. Shen, L. Jing, E. Li, L. Shen, H. Chen, Deep-learning-enabled self-adaptive microwave cloak without human intervention. Nat. Photon 14, 383–390 (2020).

35. T. Yin, Z. Wei, X. Chen, Non-iterative methods based on singular value decomposition for inverse scattering problems. IEEE Trans. Antennas Propag. 68, 4764–4773 (2020).

36. L. Mohjazi, A. Zoha, L. Bariah, S. Muhaidat, P. C. Sofotasios, M. A. Imran, O. A. Dobre, An outlook on the interplay of artificial intelligence and software-defined metasurfaces: An overview of opportunities and limitations. IEEE Veh. Technol. Mag. 15, 62–73 (2020).

37. T. Cai, S. W. Tang, G. M. Wang, H. X. Xu, S. L. Sun, Q. He, L. Zhou, High-performance bifunctional metasurfaces in transmission and reflection geometries. Adv. Opt. Mater. 5, 1600506 (2017).

38. M. Jia, Z. Wang, H. Li, X. Wang, W. Luo, S. Sun, Y. Zhang, Q. He, L. Zhou, Efficient manipulations of circularly polarized terahertz waves with transmissive metasurfaces. Light Sci. Appl. 8, 16 (2019).

39. J. Peurifoy, Y. Shen, L. Jing, Y. Yang, F. C.-Renteria, B. G. De Lacy, J. D. Joannopoulos, M. Tegmark, M. Soljačić, Nanophotonic particle simulation and inverse design using artificial neural networks. Sci. Adv. 4, eaar4206 (2018).

40. J. Jiang, J. A. Fan, Global optimization of dielectric metasurfaces using a physics-driven neural network. Nano Lett. 19, 5366–5372 (2019).

41. A. Zhan, R. Gibson, J. Whitehead, E. Smith, J. R. Hendrickson, A. Majumdar, Controlling three-dimensional optical fields via inverse Mie scattering. Sci. Adv. 5, eaax4769 (2019).

42. S. Molesky, Z. Lin, A. Y. Piggott, W. Jin, J. Vucković, A. W. Rodriguez, Inverse design in nanophotonics. Nat. Photon. 12, 659–670 (2018).

43. T. Li, A. Chen, L. Fan, M. Zheng, J. Wang, G. Lu, M. Zhao, X. Cheng, W. Li, X. Liu, H. Yin, L. Shi, J. Zi, Photonic-dispersion neural networks for inverse scattering problems. Light Sci. Appl. 10, 154 (2021).

44. T. Shan, X. Pan, M. Li, S. Xu, F. Yang, Coding programmable metasurfaces based on deep learning techniques. IEEE J. Emerg. Sel. Top Circuits Syst. 10, 114–125 (2020).

45. K. Simonyan A. Zisserman, Very deep convolutional networks for large-scale image recognition. arXiv:1409.1556 [cs.CV] (2015).

46. Z. Zhang, M. R. Sabuncu, Generalized cross entropy loss for training deep neural networks with noisy labels. arXiv:1805.07836 [cs.LG] (2018).

47. Z. Wang, C. Qian, T. Cai, L. Tian, Z. Fan, J. Liu, Y. Shen, L. Jing, J. Jin, E. P. Li, B. Zheng, H. Chen, Demonstration of spider-eyes-like intelligent antennas for dynamically perceiving incoming waves. Adv. Intell. Syst. 3, 2100066–2100068 (2021).

48. J. Chen, X. Chen, C. J. Liu, K. Huang, X. B. Xu, Analysis of temperature effect on p-i-n diode circuits by a multiphysics and circuit cosimulation algorithm. IEEE Trans Electron Devices 59, 3069–3077 (2012).

49. Q. Wang, E. Plum, Q. Yang, X. Zhang, Q. Xu, Y. Xu, J. Han, W. Zhang, Reflective chiral meta-holography: Multiplexing holograms for circularly polarized waves. Light Sci. Appl. 7, 25 (2018).

50. Z. Li, X. Tian, C. W. Qiu, J. S. Ho, Metasurfaces for bioelectronics and healthcare. Nat. Electron. 4, 382–391 (2021).

51. Q. Wang, E. T. F. Rogers, B. Gholipour, C. M. Wang, G. Yuan, J. Teng, N. I. Zheludev, Optically reconfigurable metasurfaces and photonic devices based on phase change materials. Nat. Photon 10, 60–65 (2016).

52. R. Zhu, T. Qiu, J. Wang, S. Sui, C. Hao, T. Liu, Y. Li, M. Feng, A. Zhang, C. W. Qiu, S. Qu, Phase-to-pattern inverse design paradigm for fast realization of functional metasurfaces via transfer learning. Nat. Commun. 12, 2974 (2021).

53. K. Chaudhuri, R. Salakhutdinov, Online meta-learning, paper presented at the Proceedings of the 36th International Conference on Machine Learning, California, 9 to 15 June 2019.

54. Y. Jia, C. Qian, Z. Fan, Y. Ding, Z. Wang, D. Wang, E.-P. Li, B. Zheng, T. Cai, H. Chen, In situ customized illusion enabled by global metasurface reconstruction. Adv. Funct. Mater. 32, 2109331 (2022).

55. C. Balanis, Antenna Theory (Wiley, 2016).

56. H. Taghvaee, A. Cabellos-Aparicio, J. Georgiou, S. Abadal, Error analysis of programmable metasurfaces for beam steering. IEEE J. Emerg. Sel. Topics Power Electron. 10, 62–74 (2020).

57. J. Long, E. Shelhamer, T. Darrell, Fully convolutional networks for semantic segmentation. arXiv:1411.4038 [cs.CV] (2015).

58. D. Kingma, J. Ba, Adam: A method for stochastic optimization. arXiv:1412.6980 [cs.LG] (2014).

59. X. Glorot, Y. Bengio, Understanding the difficulty of training deep feedforward neural networks. J. Mach. Learn Res. 9, 249–256 (2010).

60. K. Roth, H. Pirzadeh, A. L. Swindlehurst, J. A. Nossek, A comparison of hybrid beamforming and digital beamforming with low-resolution ADCs for multiple users and imperfect CSI. IEEE J. Sel. Top. Signal Process. 12, 484–498 (2018).

61. A. M. Niknejad, D. Chowdhury, J. Chen, Design of CMOS power amplifiers. IEEE Trans. Microw. Theory Tech. 60, 1784–1796 (2012).

62. W. Luo, S. Sun, H. Xu, Q. He, L. Zhou, Transmissive ultrathin Pancharatnam-Berry metasurfaces with nearly 100% efficiency. Phys. Rev. Applied 7, 044033 (2017).

63. B. Xiong, L. Deng, R. Peng, Y. Liu, Controlling the degrees of freedom in metasurface designs for multi-functional optical devices. Nanoscale. Adv. 1, 3786–3806 (2019).

64. S. Sun, Q. He, S. Xiao, Q. Xu, X. Li, L. Zhou, Gradient-index meta-surfaces as a bridge linking propagating waves and surface waves. Nat. Mat. 11, 426–431 (2012).

Acknowledgments: We thank Y. Q. Chen for fruitful discussions. Funding: This work at Zhejiang University was sponsored by the National Natural Science Foundation of China (NNSFC) under grant nos. 61625502, 11961141010, 61975176, 62071424, 62101485, and 62027805; the Top-Notch Young Talents Program of China; and the Fundamental Research Funds for the Central Universities. Author contributions: C.Q., Z.F., and H.C. conceived the idea of this research. Z.F. and Y.J. performed the simulation and experiment. Z.F. and C.Q. wrote the paper. Z.W. designed the intelligent EM detection system. All authors shared their insights and contributed to discussions on the results. C.Q., B.Z., and H.C. supervised the project. Competing interests: The authors declare that they have no competing interests. Data and materials availability: All data needed to evaluate the conclusions in the paper are present in the paper and/or the Supplementary Materials.

Submitted 20 December 2021
Accepted 23 May 2022
Published 6 July 2022
10.1126/sciadv.abn7905
