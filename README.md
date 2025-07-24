# Cerebellar and Subcortical Contributions to Working Memory Manipulation
This repository contains the code used in my project *"Cerebellar and Subcortical Contributions to Working Memory Manipulation"* (https://doi.org/10.1038/s42003-025-08467-0). 

**Code** was written using a combination of MATLAB and Python scripts.

**Data** was downloaded from *OpenNeuro* [ds002105](https://openneuro.org/datasets/ds002105/versions/1.1.0) and originally collected from this [paper](https://www.nature.com/articles/s41593-019-0436-x)
## code
The code folder contains the code required to run the analyses and produce the figures. The following scripts will be described in an order that fits with the manuscript and analyses. Note that these scripts are not functions and should be viewed similar to a notebook.
- [dataprep_behavioural.m](https://github.com/JoshuaBTan/WM_Manipulation/blob/main/Code/dataprep_behavioural.m) includes code for processing the data after pre-processing. Reads in the behavioural data (.tsv) and timeseries data (.mat), and removes missing values, as well as separates correct and incorrect trials. Statistical analysis of behavioural differences are also calculated here (refer to manuscript for more details). **The outputs from this script are used for all analyses described below.** This script contains code to generate Figures 1c-e.
- [FIR_analysis.m](https://github.com/JoshuaBTan/WM_Manipulation/blob/main/Code/FIR_analysis.m) creates the design matrix to model the BOLD timeseries using a *Finite Impulse Response (FIR) model*. All FIR models are included in this script. This script also has code to produce Figures 2a.

- [lda.m](https://github.com/JoshuaBTan/WM_Manipulation/blob/main/Code/lda.m) runs the *Linear Discriminant Classifiers*. All parameter sweeps/iterations and performance evaluations are present in this script. This code also includes all main analyses conducted on the LDA including: identifying overlapping regions between the two LDA axes, calculating the *Net BOLD Response (Area under the curve)*, and cross-correlation analyses. This scrips produces Figures 2e, 2f, 3a, 3c-f, 4c/d and Supplementary Figures 5 & 7.

- [nrgCalc_WM.m](https://github.com/JoshuaBTan/WM_Manipulation/blob/main/Code/nrgCalc_WM.m) runs *energy landscape analysis* ([Munn et al., 2021](https://www-nature-com.ezproxy.library.sydney.edu.au/articles/s41467-021-26268-x)). Original energy landscape code can be found [here](https://github.com/ShineLabUSYD/Brainstem_DTI_Attractor_Paper). This script also produces Figures 4f-h.
- [workingmemory_1000.m](https://github.com/JoshuaBTan/WM_Manipulation/blob/main/Code/workingmemory_1000.m), [FIR_analysis1000.m](https://github.com/JoshuaBTan/WM_Manipulation/blob/main/Code/FIR_analysis1000.m), and [ldaMATLAB1000.m](https://github.com/JoshuaBTan/WM_Manipulation/blob/main/Code/ldaMATLAB1000.m) contains similar code to the main analyses but replicated using 1000 Schaefer cortical regions, 54 Tian Subcortical Regions, and 28 SUIT Cerebellar Regions. The code was used to produce the results in Supplementary Material Figure 6


## Visualisation
The visualisation folder contains the code required to create the brain visualisations. Details regarding dependencies are described in the README file.

