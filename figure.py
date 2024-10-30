
import os
import re
import random
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import make_blobs
from sklearn.manifold import TSNE
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import AllChem
from scscore.standalone_model_numpy import SCScorer
from syba.syba import Syba
from gasa import GASA
import RAscore as ra
from vina import Vina



sns.set(palette='tab20_r')
pdb_folder = "./data/gen_smi/"  


data = []


def pdb_to_smiles(pdb_path):
    mol = Chem.MolFromPDBFile(pdb_path, removeHs=True)
    return Chem.MolToSmiles(mol) if mol else None

patterns = {
    'STEP': r'STEP:\s*([\d.]+)',
    'SCORE': r'SCORE:\s*([\d.]+)',
    'LE': r'LE:\s*([\d.]+)',
    'sim': r'LE:\s*([\d.]+)',
    'syn1': r'syn1:\s*([\d.]+)',
    'syn2': r'syn2:\s*([\d.]+)',
    'syn3': r'syn3:\s*([\d.]+)',
    'syn4': r'syn4:\s*([\d.]+)',
    'syn5': r'syn5:\s*([\d.]+)',
    'resdidue1': r'resdidue1:\s*([\d.]+)',
    'resdidue2': r'resdidue1:\s*([\d.]+)',
}


for filename in os.listdir(pdb_folder):
    if filename.endswith('.pdb'):
        pdb_path = os.path.join(pdb_folder, filename)
        

        with open(pdb_path, 'r') as file:
            file_content = file.read()

            tag_values = {}
            for tag, pattern in patterns.items():
                match = re.search(pattern, file_content)
                tag_values[tag] = float(match.group(1)) if match else None
        

        smiles = pdb_to_smiles(pdb_path)
        if smiles:
            tag_values['SMILES'] = smiles
            data.append(tag_values)

# Convert the data to a DataFrame
df = pd.DataFrame(data)


#################################################fig03###################

df_0=pd.read_csv('./data/gen_smi/syn_rock.csv')
df_1=pd.read_csv('./data/gen_smi/bbar_rock.csv')
smi_ckg=df['SMILES'][0:9999]
smi_ckgi=df['SMILES'][10000:19999]
rock1=pd.read_csv('./data/gen_smi/rock1.csv')




df1_0=pd.read_csv('./data/gen_smi/syn_sars.csv')
df1_1=pd.read_csv('./data/gen_smi/bbar_sars.csv')
smi_ckg_1=df['SMILES'][20000:29999]
smi_ckgi_1=df['SMILES'][30000:39999]
sars=pd.read_csv('./data/gen_smi/sars.csv')



df2_0=pd.read_csv('./data/gen_smi/syn_aa2ar.csv')
df2_1=pd.read_csv('./data/gen_smi/bbar_aa2arcsv')
smi_ckg_2=df['SMILES'][40000:49999]
smi_ckgi_2=df['SMILES'][50000:59999]
aa2ar=pd.read_csv('./data/gen_smi/aa2ar.csv')



sc_scorer = SCScorer()
sc_scorer.restore()


syba_model = Syba()
syba_model.fit_default()


ra_model = joblib.load("RAscore_model.pkl") 


smiles_lists = {
    "synnet": df1_0['SMILES'],  
    "bbar": df1_1['SMILES'],
    "click_gen": smi_ckg_1,
    "click_gen_inpainting": smi_ckgi_1
}


results = []


for list_name, smiles_list in smiles_lists.items():
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        

        sc_score = sc_scorer.get_score_from_smi(smiles) if mol else np.nan
        

        ra_score = ra_model.predict([mol]) if mol else np.nan
        

        syba_score = syba_model.predict(mol) if mol else np.nan
        

        try:
            gasa_score = GASA(smiles)[0]  
        except Exception as e:
            print(f"GASA : {smiles},error: {e}")
            gasa_score = np.nan
        

        results.append({
            "List": list_name,
            "SMILES": smiles,
            "SC-SCORE": sc_score,
            "RA-SCORE": ra_score,
            "SYBA": syba_score,
            "GASA": gasa_score
        })

df = pd.DataFrame(results)



plt.subplot(2, 2, 1)
sns.kdeplot(data=df, x="SC-SCORE", hue="List")
plt.title("SC-SCORE Distribution (KDE)")
plt.xlabel("SC-SCORE")
plt.ylabel("Density")

# RA-SCORE - Percentage Distribution
plt.subplot(2, 2, 2)
ra_data = df.groupby("List")["RA-SCORE"].value_counts(normalize=True).unstack().fillna(0) * 100
ra_data.plot(kind="bar", stacked=True, ax=plt.gca())
plt.title("RA-SCORE Percentage Distribution")
plt.xlabel("List")
plt.ylabel("Percentage (%)")

# GASA - Percentage Distribution
plt.subplot(2, 2, 3)
gasa_data = df.groupby("List")["GASA"].value_counts(normalize=True).unstack().fillna(0) * 100
gasa_data.plot(kind="bar", stacked=True, ax=plt.gca())
plt.title("GASA Percentage Distribution")
plt.xlabel("List")
plt.ylabel("Percentage (%)")

# SYBA - KDE Plot
plt.subplot(2, 2, 4)
sns.kdeplot(data=df, x="SYBA", hue="List")
plt.title("SYBA Distribution (KDE)")
plt.xlabel("SYBA")
plt.ylabel("Density")


plt.tight_layout()
plt.savefig('figure3.pdf')
plt.show()




#################################################fig04###################


def smiles_to_ecfp6(smiles_list):
    ecfp6_fingerprints = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=1024)
            ecfp6_fingerprints.append(fp)
    return ecfp6_fingerprints











df_0 = pd.read_csv('./data/gen_smi/syn_rock.csv')
df_1 = pd.read_csv('./data/gen_smi/bbar_rock.csv')
smi_ckg = df['SMILES'][0:9999]
smi_ckgi = df['SMILES'][10000:19999]

rock1 = pd.read_csv('./data/gen_smi/rock1.csv')

df1_0 = pd.read_csv('./data/gen_smi/syn_sars.csv')
df1_1 = pd.read_csv('./data/gen_smi/bbar_sars.csv')
smi_ckg_1 = df['SMILES'][20000:29999]
smi_ckgi_1 = df['SMILES'][30000:39999]
sars = pd.read_csv('./data/gen_smi/sars.csv')

df2_0 = pd.read_csv('./data/gen_smi/syn_aa2ar.csv')
df2_1 = pd.read_csv('./data/gen_smi/bbar_aa2ar.csv')
smi_ckg_2 = df['SMILES'][40000:49999]
smi_ckgi_2 = df['SMILES'][50000:59999]
aa2ar = pd.read_csv('./data/gen_smi/aa2ar.csv')



rock1_ecfp6 = smiles_to_ecfp6(rock1['smiles'])
sars_ecfp6 = smiles_to_ecfp6(sars['smiles'])
aa2ar_ecfp6 = smiles_to_ecfp6(aa2ar['smiles'])

df_0_ecfp6 = smiles_to_ecfp6(df_0['smiles'])
df_1_ecfp6 = smiles_to_ecfp6(df_1['smiles'])
smi_ckg_ecfp6 = smiles_to_ecfp6(smi_ckg)
smi_ckgi_ecfp6 = smiles_to_ecfp6(smi_ckgi)



def tsne_transform(fingerprints):
    tsne = TSNE(n_components=2, random_state=42)
    return tsne.fit_transform(fingerprints)



datasets = [
    ("rock1", rock1_ecfp6, df_0_ecfp6, df_1_ecfp6, smi_ckg_ecfp6, smi_ckgi_ecfp6),
    ("sars", sars_ecfp6, df_0_ecfp6, df_1_ecfp6, smi_ckg_ecfp6, smi_ckgi_ecfp6),
    ("aa2ar", aa2ar_ecfp6, df_0_ecfp6, df_1_ecfp6, smi_ckg_ecfp6, smi_ckgi_ecfp6)
]

# Set up subplots
fig, axes = plt.subplots(3, 4, figsize=(20, 15))
fig.suptitle("t-SNE Distribution of Different Datasets", fontsize=16)



for i, (target_name, target_data, df_0_data, df_1_data, smi_ckg_data, smi_ckgi_data) in enumerate(datasets):


    combined_data = target_data + df_0_data + df_1_data + smi_ckg_data + smi_ckgi_data
    tsne_result = tsne_transform(combined_data)
    


    num_target = len(target_data)
    num_df_0 = len(df_0_data)
    num_df_1 = len(df_1_data)
    num_smi_ckg = len(smi_ckg_data)
    num_smi_ckgi = len(smi_ckgi_data)
    


    axes[i, 0].scatter(tsne_result[:num_target, 0], tsne_result[:num_target, 1], label=target_name, alpha=0.6)
    axes[i, 0].scatter(tsne_result[num_target:num_target + num_df_0, 0], tsne_result[num_target:num_target + num_df_0, 1],  alpha=0.6)
    axes[i, 1].scatter(tsne_result[num_target:num_target + num_df_1, 0], tsne_result[num_target:num_target + num_df_1, 1],  alpha=0.6)
    axes[i, 2].scatter(tsne_result[num_target:num_target + num_smi_ckg, 0], tsne_result[num_target:num_target + num_smi_ckg, 1],  alpha=0.6)
    axes[i, 3].scatter(tsne_result[num_target:num_target + num_smi_ckgi, 0], tsne_result[num_target:num_target + num_smi_ckgi, 1], alpha=0.6)
    
    for j in range(4):
        axes[i, j].set_title(f"{target_name} with Collection {j+1}")
        axes[i, j].legend()

plt.savefig('figure04.pdf')
plt.tight_layout()
plt.show()







#################################################fig05###################





# Function to calculate molecular weight from syn1-syn5
def calculate_molecular_weight(row):
    return row['syn1'] + row['syn2'] + row['syn3'] + row['syn4'] + row['syn5']

# Calculate molecular weight for each entry
df['MolecularWeight'] = df.apply(calculate_molecular_weight, axis=1)

# Split the data for each target
targets = {
    "ROCK": df.iloc[:20000],
    "SARS": df.iloc[20000:40000],
    "AA2AR": df.iloc[40000:]
}

# Plotting
for target_name, target_data in targets.items():
    # Split target data into ckg and ckg_i
    ckg = target_data.iloc[:10000]
    ckg_i = target_data.iloc[10000:]
    
    # First plot: STEP distribution for ckg and ckg_i
    plt.figure(figsize=(12, 6))
    sns.kdeplot(ckg['STEP'], label=f'{target_name} - ckg STEP', fill=True)
    sns.kdeplot(ckg_i['STEP'], label=f'{target_name} - ckg_i STEP', fill=True)
    plt.title(f'STEP Distribution for {target_name}')
    plt.xlabel('STEP')
    plt.ylabel('Density')
    plt.legend()
    plt.show()
    
    # Second plot: Molecular weight distribution for syn1 to syn5
    plt.figure(figsize=(12, 6))
    sns.kdeplot(ckg['MolecularWeight'], label=f'{target_name} - ckg Molecular Weight', fill=True)
    sns.kdeplot(ckg_i['MolecularWeight'], label=f'{target_name} - ckg_i Molecular Weight', fill=True)
    plt.title(f'Molecular Weight Distribution (syn1-syn5) for {target_name}')
    plt.xlabel('Molecular Weight')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig('figure05.pdf')
    plt.show()

#################################################fig06###################

def smiles_to_ecfp6(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=1024) if mol else None

# Generate ECFP6 fingerprints for all SMILES entries
df['Fingerprint'] = df['SMILES'].apply(smiles_to_ecfp6)

# Split data into three targets: ROCK, SARS, and AA2AR
targets = {
    "ROCK": df.iloc[:20000],
    "SARS": df.iloc[20000:40000],
    "AA2AR": df.iloc[40000:]
}

# Function to apply t-SNE on fingerprints
def apply_tsne(fingerprints):
    tsne = TSNE(n_components=2, random_state=42)
    return tsne.fit_transform(fingerprints)

# Plot for each target
for target_name, target_data in targets.items():
    # Split target data into ckg and ckg_i
    ckg = target_data.iloc[:10000]
    ckg_i = target_data.iloc[10000:]
    
    # First Plot: t-SNE Distribution colored by SCORE
    fingerprints_combined = ckg['Fingerprint'].tolist() + ckg_i['Fingerprint'].tolist()
    tsne_result = apply_tsne([list(fp) for fp in fingerprints_combined])

    plt.figure(figsize=(12, 6))
    plt.scatter(tsne_result[:10000, 0], tsne_result[:10000, 1], c=ckg['SCORE'], cmap='viridis', label='ckg', alpha=0.7)
    plt.scatter(tsne_result[10000:, 0], tsne_result[10000:, 1], c=ckg_i['SCORE'], cmap='plasma', label='ckg_i', alpha=0.7)
    plt.colorbar(label='SCORE')
    plt.title(f't-SNE ECFP6 Distribution of {target_name} with Score Gradient')
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend()
    plt.show()
    
    # Second Plot: LE distribution for ckg and ckg_i
    plt.figure(figsize=(12, 6))
    sns.kdeplot(ckg['LE'], label='ckg LE', fill=True)
    sns.kdeplot(ckg_i['LE'], label='ckg_i LE', fill=True)
    plt.title(f'LE Distribution for {target_name}')
    plt.xlabel('LE')
    plt.ylabel('Density')
    plt.legend()
    plt.show()
    
    # Third Plot: Jointplot of SCORE vs SIM
    plt.figure(figsize=(8, 8))
    sns.jointplot(data=target_data, x='SCORE', y='SIM', kind='scatter',  palette='viridis', alpha=0.7)
    plt.suptitle(f'SCORE vs SIM Jointplot for {target_name}', y=1.02)
    plt.xlabel('SCORE')
    plt.ylabel('SIM')
    plt.show()

#################################################fig07###################

sars_df = df.iloc[20000:40000]
smiles_list = sars_df['SMILES'].tolist()


v = Vina(sf_name='vina')


receptor_path = './data/protein.pdbqt'
v.set_receptor(receptor_path)

center = [, , ]  
box_size = [80, 80, 80]

# Prepare output directories
os.makedirs('docked_conformations', exist_ok=True)
os.makedirs('original_conformations', exist_ok=True)

# Function to convert SMILES to 3D conformations and prepare for docking
def smiles_to_3d_pdbqt(smiles, name):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.UFFOptimizeMolecule(mol)
        pdb_path = f'original_conformations/{name}.pdb'
        pdbqt_path = f'docked_conformations/{name}.pdbqt'
        Chem.MolToPDBFile(mol, pdb_path)
        Chem.MolToMolFile(mol, pdbqt_path)
        return pdb_path, pdbqt_path
    return None, None


rmsd_values_model1 = []
rmsd_values_model2 = []

for i, smiles in enumerate(smiles_list):
    pdb_path, pdbqt_path = smiles_to_3d_pdbqt(smiles, f'ligand_{i}')
    if pdb_path and pdbqt_path:

        v.set_ligand_from_file(pdbqt_path)
        

        v.compute_vina_maps(center=center, box_size=box_size)
        

        v.dock(exhaustiveness=8, n_poses=1)
        v.write_poses(f'docked_conformations/ligand_{i}_docked.pdbqt', n_poses=1, overwrite=True)
        

        original_mol = Chem.MolFromPDBFile(pdb_path)
        docked_mol = Chem.MolFromPDBFile(f'docked_conformations/ligand_{i}_docked.pdbqt')
        

        if original_mol and docked_mol:
            rmsd = AllChem.GetBestRMS(original_mol, docked_mol)
            if i < 10000:
                rmsd_values_model1.append(rmsd)
            else:
                rmsd_values_model2.append(rmsd)

# Plot RMSD Distributions for two models
plt.figure(figsize=(12, 6))
sns.kdeplot(rmsd_values_model1, label='ckg RMSD', fill=True)
sns.kdeplot(rmsd_values_model2, label='ckgi RMSD', fill=True)
plt.title('RMSD Distribution for Two Models in SARS Target')
plt.xlabel('RMSD')
plt.ylabel('Density')

plt.savefig('figure07_1.pdf')

plt.legend()
plt.show()









sars_df = df.iloc[20000:40000]

# Separate data into two models
ckg = sars_df.iloc[:10000]
ckg_i = sars_df.iloc[10000:]

# Calculate the proportion of '1's for residue1 and residue2 in each model
proportion_residue1_ckg = ckg['residue1'].mean()  # Mean of 1s gives the proportion in binary data
proportion_residue2_ckg = ckg['residue2'].mean()
proportion_residue1_ckg_i = ckg_i['residue1'].mean()
proportion_residue2_ckg_i = ckg_i['residue2'].mean()

# Create a DataFrame for easy plotting
proportions_df = pd.DataFrame({
    'Model': ['ckg', 'ckg', 'ckg_i', 'ckg_i'],
    'Residue': ['residue1', 'residue2', 'residue1', 'residue2'],
    'Proportion': [proportion_residue1_ckg, proportion_residue2_ckg,
                         proportion_residue1_ckg_i, proportion_residue2_ckg_i]
})

# Plot the proportions
plt.figure(figsize=(10, 6))
sns.barplot(data=proportions_df, x='Residue', y='Proportion', hue='Model')
plt.title('Proportion of his41 and cys145 for SARS Target Models')
plt.xlabel('Residue')
plt.ylabel('Proportion')
plt.legend(title='Model')
plt.savefig('figure07_2.pdf')
plt.show()