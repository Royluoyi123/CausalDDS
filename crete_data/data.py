from argparse import Namespace
from typing import List, Tuple, Union

import torch
import torch.nn.functional as F
torch.set_num_threads(2)

# To fix the random seed
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

import random
random.seed(0)

import pandas as pd
import numpy as np
np.random.seed(0)

from torch_geometric.data import Data
from torch.utils.data import Dataset

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as rdDesc

from tqdm import tqdm

from utils import one_of_k_encoding_unk, one_of_k_encoding


# Atom feature sizes
MAX_ATOMIC_NUM = 100
ATOM_FEATURES = {
    'atomic_num': list(range(MAX_ATOMIC_NUM)),
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-1, -2, 1, 2, 0],
    'chiral_tag': [0, 1, 2, 3],
    'num_Hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
}

# Distance feature sizes
PATH_DISTANCE_BINS = list(range(10))
THREE_D_DISTANCE_MAX = 20
THREE_D_DISTANCE_STEP = 1
THREE_D_DISTANCE_BINS = list(range(0, THREE_D_DISTANCE_MAX + 1, THREE_D_DISTANCE_STEP))

# len(choices) + 1 to include room for uncommon values; + 2 at end for IsAromatic and mass
ATOM_FDIM = sum(len(choices) + 1 for choices in ATOM_FEATURES.values()) + 2
BOND_FDIM = 14

# Memoization
SMILES_TO_GRAPH = {}


def onek_encoding_unk(value: int, choices: List[int]) -> List[int]:
    """
    Creates a one-hot encoding.

    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the value in a list of length len(choices) + 1.
    If value is not in the list of choices, then the final element in the encoding is 1.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding


def atom_features(atom: Chem.rdchem.Atom, functional_groups: List[int] = None) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for an atom.

    :param atom: An RDKit atom.
    :param functional_groups: A k-hot vector indicating the functional groups the atom belongs to.
    :return: A list containing the atom features.
    """
    features = onek_encoding_unk(atom.GetAtomicNum() - 1, ATOM_FEATURES['atomic_num']) + \
            onek_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES['degree']) + \
            onek_encoding_unk(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']) + \
            onek_encoding_unk(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag']) + \
            onek_encoding_unk(int(atom.GetTotalNumHs()), ATOM_FEATURES['num_Hs']) + \
            onek_encoding_unk(int(atom.GetHybridization()), ATOM_FEATURES['hybridization']) + \
           [1 if atom.GetIsAromatic() else 0] + \
           [atom.GetMass() * 0.01]  # scaled to about the same range as other features
    if functional_groups is not None:
        features += functional_groups
    return features


def bond_features(bond: Chem.rdchem.Bond) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for a bond.

    :param bond: A RDKit bond.
    :return: A list containing the bond features.
    """
    if bond is None:
        fbond = [1] + [0] * (BOND_FDIM - 1)
    else:
        bt = bond.GetBondType()
        fbond = [
            0,  # bond is not None
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            (bond.GetIsConjugated() if bt is not None else 0),
            (bond.IsInRing() if bt is not None else 0)
        ]
        fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))
    return fbond


def get_graph_from_smile(molecule_smile, idx):
    """
    Method that constructs a molecular graph with nodes being the atoms
    and bonds being the edges.
    :param molecule_smile: SMILE sequence
    :return: DGL graph object, Node features and Edge features
    """

    molecule = Chem.MolFromSmiles(molecule_smile)
    features = rdDesc.GetFeatureInvariants(molecule)

    stereo = Chem.FindMolChiralCenters(molecule)
    chiral_centers = [0] * molecule.GetNumAtoms()
    for i in stereo:
        chiral_centers[i[0]] = i[1]

    node_features = []
    edge_features = []
    bonds = []
    for i in range(molecule.GetNumAtoms()):

        atom_i = molecule.GetAtomWithIdx(i)

        atom_i_features = atom_features(atom_i)
        node_features.append(atom_i_features)

        for j in range(molecule.GetNumAtoms()):
            bond_ij = molecule.GetBondBetweenAtoms(i, j)
            if bond_ij is not None:
                bonds.append([i, j])
                bond_features_ij = bond_features(bond_ij)
                edge_features.append(bond_features_ij)

    atom_feats = torch.tensor(node_features, dtype = torch.float)
    edge_index = torch.tensor(bonds, dtype = torch.long).T
    edge_feats = torch.tensor(edge_features, dtype = torch.float)

    return Data(x = atom_feats, edge_index = edge_index, edge_attr = edge_feats, idx = idx)


def build_dataset(dataset, drug_a, drug_b, cell_line, target,  smiles_dict):

    processed = list()

    count = 0

    for idx in tqdm(range(len(dataset))):

        solute = smiles_dict[dataset.loc[idx][drug_a]]
        mol = Chem.MolFromSmiles(solute)
        mol = Chem.AddHs(mol)
        solute = Chem.MolToSmiles(mol)
        solute_graph = get_graph_from_smile(solute, idx)

        solvent = smiles_dict[dataset.loc[idx][drug_b]]
        mol = Chem.MolFromSmiles(solvent)
        mol = Chem.AddHs(mol)
        solvent = Chem.MolToSmiles(mol)
        solvent_graph = get_graph_from_smile(solvent, idx)
        cell= dataset.loc[idx][cell_line]
        
        label = torch.tensor(dataset.loc[idx][target], dtype = torch.long)
        #label = torch.tensor(dataset.loc[idx][target], dtype = torch.float)

        processed.append([solute_graph, solvent_graph, cell,label])

    return processed, dataset


class Dataclass(Dataset):
    def __init__(self, dataset, cell_line_dict):
        self.dataset = dataset
        self.cell_line_dict = cell_line_dict

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        
        return [data[0],data[1],self.cell_line_dict[data[2]],data[3]]


if __name__ == "__main__":
    
    for i in range(1,2):

        smiles_df = pd.read_csv("DDS/ddsdata/ONEIL-COSMIC640/drug_smiles_opti.csv",header=0)
        #smiles_dict = dict(zip(smiles_df['Drug_name'],smiles_df['Smiles']))
        smiles_dict = dict(zip(smiles_df['pubchemid'],smiles_df['isosmiles']))
        #df = pd.read_csv(f'DDS/ddsdata/ONEIL-COSMIC640/raw/regression/Maxdis/train.csv', sep=",")
        #processed_data, dataset = build_dataset(df, "drug_a", "drug_b","cell_line","synergy", smiles_dict)
        #torch.save(processed_data, "DDS/ddsdata/ONEIL-COSMIC640/processed/regression/Maxdis/train_"+str(i)+".pt")

        #df = pd.read_csv(f'CMRL-main/DDI/data/raw/ChChMiner_valid.csv', sep=",")
        #processed_data, dataset = build_dataset(df, "smiles_1", "smiles_2", "label")
        #torch.save(processed_data, "CMRL-main/DDI/data/processed/ChChMiner_valid.pt")

        df = pd.read_csv(f'DDS/ddsdata/ONEIL-COSMIC640/raw/regression/warm_start/opti_2.csv', sep=",")
        processed_data, dataset = build_dataset(df, "drug_a", "drug_b","cell_line","synergy", smiles_dict)
        torch.save(processed_data, "DDS/ddsdata/ONEIL-COSMIC640/processed/regression/warm_start/opti_"+str(2)+".pt")

        print(i)
