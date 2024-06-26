# ClickGen: Directed Exploration of Synthesizable Chemical Space Leading to the Rapid Synthesis of Novel and Active Lead Compounds via Modular Reactions and Reinforcement Learning

![overview of the architecture of ClickGen](/images/figure.png)

## Overview
This repository contains the source of ClickGen, a deep learning model that utilizes modular reactions like click chemistry to assemble molecules and incorporates reinforcement learning to ensure that the proposed molecules display high diversity, novelty and strong binding tendency. 

## Abstract

Despite the vast potential of generative models, the severe challenge in low synthesizability of many generated molecules has restricted their potential impacts in real-world scenarios. In response to this issue, we develop ClickGen, a deep learning model that utilizes modular reactions like click chemistry to assemble molecules and incorporates reinforcement learning along with inpainting technique to ensure that the proposed molecules display high diversity, novelty and strong binding tendency. We then further conducted wet-lab validation on ClickGen’s proposed molecules for PARP1. Due to the guaranteed high synthesizability and model-generated synthetic routes for reference, we successfully produced and tested the bioactivity of these novel compounds in just 20 days, much faster than typically expected time frame when handling sufficiently novel molecules. In bioactivity assays, two lead compounds demonstrated superior anti-proliferative efficacy against cancer cell lines, low toxicity, and nanomolar-level inhibitory activity to PARP1. We anticipate that ClickGen and related models could potentially signify a new paradigm in molecular generation, advancing the future of AI and automated experimentation-driven closed-loop molecular design closer to reality.


## System Requirements

### Hardware requirements

`ClickGen` software is recommended for use on computers with more than 20GB of VRAM or RAM

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
- openbabel >= 3.1.1


if utilizing GPU accelerated model training 
- CUDA==10.2 & cudnn==7.5 

## Install from Github & Creat a new environment in conda 
```
git clone https://github.com/mywang1994/cligen_gen
cd cligen_gen
conda env create -f environment.yaml

```



## Running ClickGen
### 1.Prepare synthons dataset
The ClickGen model requires labeled reactants, stored in data files in `.csv` format, as well as protein structures that have been energy-minimized and repaired, saved in `.pdbqt` format. Finally, a standardized SMILES format for the initial synthons fragment is essential, with the annotation method detailed in `./data/prepare_data.py` or as described in the data preparation section of the article.

```
python ./data/prepare_data.py   --input_path  SMILES.csv       # the path of SMILES .csv files
                                --output_path output.csv       # the path of save synthons .csv files  
```

### 2.Train the Reaction-based combiner

To train the Reaction-based combiner, it is necessary to utilize a SMILES dataset along with the synthon dataset obtained in step 1. It is essential to define the number of positive and negative samples, as well as some fundamental model parameters, such as the learning rate. 



```
python train_combiner.py   --mol_p  SMILES.csv       # the path of SMILES .csv files
                            --syn_path synthons.csv       # the path of labeled synthons .sv files
                            --num_p 100                   # the number of positive_samples
                            --num_n 1000                  # the number of negative_samples
                            --lr 1e-4                     # the learning rate
                            --epoch 80                    # the training epoches
```
Ultimately, the model file will be stored in the  `./data/model/  `directory.




### 3.Train the Inpainting-based generator

Training the Inpainting-based generator model requires the dataset created in step 1, along with the input model parameters such as embedding and hidden dimensions. Users can also configure the model's skip connections and attention mechanisms flexibly via the command line, allowing the model to be adjusted according to different needs. Additionally, hardware requirements include at least 20GB of GPU memory or CPU memory (not recommended due to slower training speed).

```
python train_inpainting.py  --mol_p  SMILES.csv          # the path of SMILES .csv files
                            --embed_dim 64                 # the embedding size
                            --hid_dim 100                  # the hidden dimension
                            --skip_connection 1000         # skip connection
                            --attention 1e-4               # attention mechanism
                            --lr 1e-4                      # the learning rate
                            --epoch 80                     # the training epoches

```
Ultimately, the model file will be stored in the  `./data/model/  `directory.




### 4.Run the ClickGen


To run the ClickGen model, you need to use the dataset obtained in step 1, as well as the Inpainting-based generator and Reaction-based combiner trained in steps 2 and 3. You also need the starting synthons (which can be omitted in inpainting mode), the corresponding protein target pdb structure, and the input parameters for the model, such as the number of molecules to be generated and the parameters for the Inpainting-based generator and Reaction-based combiner.

```
python run.py     --inpainting Trur/False                      # 是否需要使用inpainting模块
                  --input [3*]NC1CCCC(N[3*])CC1                # initial synthon fragment,如果使用inpainting模式则无需输入起始反应子
                  --syn_p        synthons.csv                  # the path of labeled synthons
                  --protein ./data/parp1.pdb                   # protein
                  --num_sims 10000                            # simulation steps
```

Based on our tests, generating 10,000 molecules with ClickGen takes between 0.5 to 1.5 hours, depending on the system and hardware configuration.

## License

This project is covered under the MIT License.
