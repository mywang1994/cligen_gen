# ClickGen: Directed Exploration of Synthesizable Chemical Space Leading to the Rapid Synthesis of Novel and Active Lead Compounds via Modular Reactions and Reinforcement Learning

![overview of the architecture of ClickGen](/images/figure.png)

## Overview
This repository contains the source of ClickGen, a deep learning model that utilizes modular reactions like click chemistry to assemble molecules and incorporates reinforcement learning to ensure that the proposed molecules display high diversity, novelty and strong binding tendency. 

## Abstract

Despite the vast potential of generative models, the severe challenge in low synthesizability of many generated molecules has restricted their potential impacts in real-world scenarios. In response to this issue, we develop ClickGen, a deep learning model that utilizes modular reactions like click chemistry to assemble molecules and incorporates reinforcement learning along with inpainting technique to ensure that the proposed molecules display high diversity, novelty and strong binding tendency. We then further conducted wet-lab validation on ClickGen’s proposed molecules for PARP1. Due to the guaranteed high synthesizability and model-generated synthetic routes for reference, we successfully produced and tested the bioactivity of these novel compounds in just 20 days, much faster than typically expected time frame when handling sufficiently novel molecules. In bioactivity assays, two lead compounds demonstrated superior anti-proliferative efficacy against cancer cell lines, low toxicity, and nanomolar-level inhibitory activity to PARP1. We anticipate that ClickGen and related models could potentially signify a new paradigm in molecular generation, advancing the future of AI and automated experimentation-driven closed-loop molecular design closer to reality.


## System Requirements

### Hardware requirements

`ClickGen` software is recommended for use on computers with more than 10GB of VRAM.

### OS Requirements
This package is supported for *Linux* and *Windows*. The package has been tested on the following systems:
+ Windows: Windows 11 23H2
+ Linux: Ubuntu 22.04

### Software requirements

- Python == 3.7
- pytorch >= 1.1.0
- openbabel == 2.4.1
- RDKit == 2020.09.5
- autodock vina (for python) [README](https://autodock-vina.readthedocs.io/en/latest/docking_python.html)


if utilizing GPU accelerated model training 
- CUDA==10.2 & cudnn==7.5 

## Creat a new environment in conda 

 ` `



### Install from Github
```
git clone https://github.com/mywang1994/cligen_gen
cd cligen_gen
conda env create -f environment.yml
```









## Running ClickGen
### 1.Prepare synthons dataset
The ClickGen model requires labeled reactants, stored in data files in `.csv` format, as well as protein structures that have been energy-minimized and repaired, saved in `.pdb` format. Finally, a standardized SMILES format for the initial synthons fragment is essential, with the annotation method detailed in `./data/prepare_data.py` or as described in the data preparation section of the article.

 `python ./data/prepare_data.py --input_path 'the path of SMILES csv files'
                                --output_path ./data/synthons` 

### 2.Run the ClickGen




`python run.py    --input [3*]NC1CCCC(N[3*])CC1 # initial synthon fragment 
                  --syn_p  ./data/synthons/synthons.csv #labeled reactants
                  --protein ./data/parp1.pdb #protein
                  --num_sims 100000 # simulation steps`













Or you can use our online [ClickGen server](https://carbonsilico.com/)
