# CoBDock-2: Enhancing Blind Docking Performance through Hybrid Feature Selection Combining Ensemble and Multimodel Feature Selection Approaches

# Reference Implementation of CobDock-2 algorithm
This readme file documents all of the required steps to run CobDock-2.

Note that the code was implemented and tested on a Linux operating system only.

## How to set up the environment
We have provided an Anaconda environment file for easy setup.
If you do not have Anaconda installed, you can get Miniconda from [here](https://docs.conda.io/en/latest/miniconda.html).
Create the `cobdock_2` environment using the following command:
```bash
conda env create -n cobdock_2 -f environment.yml
conda activate cobdock_2
```

# In order to install requirement packages
```bash
pip install -r requirements.txt
```

# CobDock-2: Step-by-Step Pipeline

This repository contains a step-by-step pipeline for structure-based binding site prediction, feature selection, and local docking.

## Pipeline Overview

1. **cobdock_2_step_1_Fpocket_run_selected**  
   Runs Fpocket to detect and extract selected binding pockets from protein structures.

2. **cobdock_2_step_2_Prepare_Fpocket_features**  
   Extracts physicochemical features from Fpocket output for each pocket.

3. **cobdock_2_step_3_fix_pockets_selected**  
   Fixes and filters pocket selections for consistent downstream analysis.

4. **cobdock_2_step_4_Prepare_amino_acid_features**  
   Computes amino acid-level features around selected pockets.

5. **cobdock_2_step_5_Boruta_feature_selection**  
   Applies Boruta algorithm to identify the most relevant features for classification.

6. **cobdock_2_step_6_autogluon**  
   Trains machine learning models using AutoGluon based on selected features.

7. **cobdock_2_step_7_Make_prediction_binding_site_performance**  
   Evaluates model predictions and performance metrics on test data.

8. **cobdock_2_step_8_local_docking_with_PLANTS**  
   Performs local molecular docking using the PLANTS docking software.

9. **cobdock_2_step_9_RMSD_validation**  
   Validates docking results by calculating RMSD between predicted and reference poses.

# Note: For the Local docking make sure, PLANTS exist in the directory

## License

This project is licensed for **academic and research purposes only**. For commercial usage, please connect with s.yavuz.ugurlu@gmail.com
