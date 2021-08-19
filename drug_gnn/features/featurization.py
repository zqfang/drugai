
from typing import List, Tuple, Union


import torch
import numpy as np
import pandas as pd

from itertools import chain, accumulate
from rdkit import Chem



# Atom feature sizes
MAX_ATOMIC_NUM = 100 # FIXME: MAX_ATOMIC_NUM
CIP_CHIRALITY = ['R', 'S']
ATOM_FEATURES = {
    'atomic_num': list(range(MAX_ATOMIC_NUM)),
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-1, -2, 1, 2, 0],
    'chiral_tag': [0, 1, 2, 3],
    #'global_chiral_tag': ['R', 'S'],
    'num_Hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
}

CHIRALTAG_PARITY = {
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW: +1,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW: -1,
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED: 0,
    Chem.rdchem.ChiralType.CHI_OTHER: 0, # default
}

# Distance feature sizes
PATH_DISTANCE_BINS = list(range(10))
THREE_D_DISTANCE_MAX = 20
THREE_D_DISTANCE_STEP = 1
THREE_D_DISTANCE_BINS = list(range(0, THREE_D_DISTANCE_MAX + 1, THREE_D_DISTANCE_STEP))

# len(choices) + 1 to include room for uncommon values; + 2 at end for IsAromatic and mass
ATOM_FDIM = sum(len(choices) + 1 for choices in ATOM_FEATURES.values()) + 2
EXTRA_ATOM_FDIM = 0
BOND_FDIM = 14  # FIXME
EXTRA_BOND_FDIM = 0
REACTION_MODE = None
EXPLICIT_H = False
REACTION = False

def make_mol(s: str, keep_h: bool):
    """
    Builds an RDKit molecule from a SMILES string.
    
    :param s: SMILES string.
    :param keep_h: Boolean whether to keep hydrogens in the input smiles. This does not add hydrogens, it only keeps them if they are specified.
    :return: RDKit molecule.
    """
    if keep_h:
        mol = Chem.MolFromSmiles(s, sanitize = False)
        Chem.SanitizeMol(mol, sanitizeOps = Chem.SanitizeFlags.SANITIZE_ALL^Chem.SanitizeFlags.SANITIZE_ADJUSTHS)
    else:
        mol = Chem.MolFromSmiles(s)
    return mol

def get_atom_fdim(overwrite_default_atom: bool = False) -> int:
    """
    Gets the dimensionality of the atom feature vector.

    :param overwrite_default_atom: Whether to overwrite the default atom descriptors
    :return: The dimensionality of the atom feature vector.
    """
    return (not overwrite_default_atom) * ATOM_FDIM + EXTRA_ATOM_FDIM


def set_explicit_h(explicit_h: bool) -> None:
    """
    Sets whether RDKit molecules will be constructed with explicit Hs.

    :param explicit_h: Boolean whether to keep explicit Hs from input.
    """
    global EXPLICIT_H
    EXPLICIT_H = explicit_h


def set_reaction(reaction: bool, mode: str) -> None:
    """
    Sets whether to use a reaction or molecule as input and adapts feature dimensions.
 
    :param reaction: Boolean whether to except reactions as input.
    :param mode: Reaction mode to construct atom and bond feature vectors.

    """
    global REACTION
    REACTION = reaction
    if reaction:
        global REACTION_MODE
        global EXTRA_BOND_FDIM
        global EXTRA_ATOM_FDIM
    
        EXTRA_ATOM_FDIM = ATOM_FDIM - MAX_ATOMIC_NUM -1
        EXTRA_BOND_FDIM = BOND_FDIM
        REACTION_MODE = mode        

        
def is_explicit_h() -> bool:
    r"""Returns whether to use retain explicit Hs"""
    return EXPLICIT_H


def is_reaction() -> bool:
    r"""Returns whether to use reactions as input"""
    return REACTION


def reaction_mode() -> str:
    r"""Returns the reaction mode"""
    return REACTION_MODE


def set_extra_atom_fdim(extra):
    """Change the dimensionality of the atom feature vector."""
    global EXTRA_ATOM_FDIM
    EXTRA_ATOM_FDIM = extra


def get_bond_fdim(atom_messages: bool = False,
                  overwrite_default_bond: bool = False,
                  overwrite_default_atom: bool = False) -> int:
    """
    Gets the dimensionality of the bond feature vector.

    :param atom_messages: Whether atom messages are being used. If atom messages are used,
                          then the bond feature vector only contains bond features.
                          Otherwise it contains both atom and bond features.
    :param overwrite_default_bond: Whether to overwrite the default bond descriptors
    :param overwrite_default_atom: Whether to overwrite the default atom descriptors
    :return: The dimensionality of the bond feature vector.
    """

    return (not overwrite_default_bond) * BOND_FDIM + EXTRA_BOND_FDIM + \
           (not atom_messages) * get_atom_fdim(overwrite_default_atom=overwrite_default_atom)


def set_extra_bond_fdim(extra):
    """Change the dimensionality of the bond feature vector."""
    global EXTRA_BOND_FDIM
    EXTRA_BOND_FDIM = extra


def onek_encoding_unk(value: int, choices: List[int]) -> List[int]:
    """
    Creates a one-hot encoding with an extra category for uncommon values.

    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the :code:`value` in a list of length :code:`len(choices) + 1`.
             If :code:`value` is not in :code:`choices`, then the final element in the encoding is 1.
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
    if atom is None:
        features = [0] * ATOM_FDIM
    else:
        features = onek_encoding_unk(atom.GetAtomicNum() - 1, ATOM_FEATURES['atomic_num']) + \
            onek_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES['degree']) + \
            onek_encoding_unk(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']) + \
            onek_encoding_unk(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag']) + \
            onek_encoding_unk(int(atom.GetTotalNumHs()), ATOM_FEATURES['num_Hs']) + \
            onek_encoding_unk(int(atom.GetHybridization()), ATOM_FEATURES['hybridization']) + \
            [1 if atom.GetIsAromatic() else 0] + \
            [atom.GetMass() * 0.01]  # scaled to about the same range as other features
        # if args.chiral_features:
        #     features += onek_encoding_unk(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag'])
        # if args.global_chiral_features:
        #     if atom.HasProp('_CIPCode'):
        #         features += onek_encoding_unk(atom.GetProp('_CIPCode'), ATOM_FEATURES['global_chiral_tag'])
        #     else:
        #         features += onek_encoding_unk(None, ATOM_FEATURES['global_chiral_tag'])

        if functional_groups is not None:
            features += functional_groups
    return features


def bond_features(bond: Chem.rdchem.Bond) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for a bond.

    :param bond: An RDKit bond.
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

def parity_features(atom: Chem.rdchem.Atom) -> int:
    """
    Returns the parity of an atom if it is a tetrahedral center.
    +1 if CW, -1 if CCW, and 0 if undefined/unknown

    :param atom: An RDKit atom.
    """
    return CHIRALTAG_PARITY[atom.GetChiralTag()]
    

class MolGraph:
    """
    A :class:`MolGraph` represents the graph structure and featurization of a single molecule.

    A MolGraph computes the following attributes:

    * :code:`n_atoms`: The number of atoms in the molecule.
    * :code:`n_bonds`: The number of bonds in the molecule.
    * :code:`f_atoms`: A mapping from an atom index to a list of atom features.
    * :code:`f_bonds`: A mapping from a bond index to a list of bond features.
    * :code:`a2b`: A mapping from an atom index to a list of incoming bond indices.
    * :code:`b2a`: A mapping from a bond index to the index of the atom the bond originates from.
    * :code:`b2revb`: A mapping from a bond index to the index of the reverse bond.
    * :code:`overwrite_default_atom_features`: A boolean to overwrite default atom descriptors.
    * :code:`overwrite_default_bond_features`: A boolean to overwrite default bond descriptors.
    """

    def __init__(self, mol: Union[str, Chem.Mol, Tuple[Chem.Mol, Chem.Mol]],
                 atom_features_extra: np.ndarray = None,
                 bond_features_extra: np.ndarray = None,
                 overwrite_default_atom_features: bool = False,
                 overwrite_default_bond_features: bool = False):
        """
        :param mol: A SMILES or an RDKit molecule.
        :param atom_features_extra: A list of 2D numpy array containing additional atom features to featurize the molecule
        :param bond_features_extra: A list of 2D numpy array containing additional bond features to featurize the molecule
        :param overwrite_default_atom_features: Boolean to overwrite default atom features by atom_features instead of concatenating
        :param overwrite_default_bond_features: Boolean to overwrite default bond features by bond_features instead of concatenating
        """
        self.is_reaction = is_reaction()
        self.is_explicit_h = is_explicit_h()
        self.reaction_mode = reaction_mode()
        
        # Convert SMILES to RDKit molecule if necessary
        if type(mol) == str:
            if self.is_reaction:
                mol = (make_mol(mol.split(">")[0], self.is_explicit_h), make_mol(mol.split(">")[-1], self.is_explicit_h)) 
            else:
                mol = make_mol(mol, self.is_explicit_h)

        self.smiles = Chem.MolToSmiles(mol)
        self.n_atoms = 0  # number of atoms
        self.n_bonds = 0  # number of bonds
        self.f_atoms = []  # mapping from atom index to atom features
        self.f_bonds = []  # mapping from bond index to concat(in_atom, bond) features
        self.a2b = []  # mapping from atom index to incoming bond indices
        self.b2a = []  # mapping from bond index to the index of the atom the bond is coming from
        self.b2revb = []  # mapping from bond index to the index of the reverse bond
        self.edge_index = []
        self.overwrite_default_atom_features = overwrite_default_atom_features
        self.overwrite_default_bond_features = overwrite_default_bond_features
        self.f_bonds_dim = 0
        # Convert smiles to molecule
        # mol = Chem.MolFromSmiles(mol)

        if not self.is_reaction:
            # Get atom features
            self.f_atoms = [atom_features(atom) for atom in mol.GetAtoms()]
            if atom_features_extra is not None:
                if overwrite_default_atom_features:
                    self.f_atoms = [descs.tolist() for descs in atom_features_extra]
                else:
                    self.f_atoms = [f_atoms + descs.tolist() for f_atoms, descs in zip(self.f_atoms, atom_features_extra)]

            self.n_atoms = len(self.f_atoms)
            if atom_features_extra is not None and len(atom_features_extra) != self.n_atoms:
                raise ValueError(f'The number of atoms in {Chem.MolToSmiles(mol)} is different from the length of '
                                 f'the extra atom features')

            # Initialize atom to bond mapping for each atom
            for _ in range(self.n_atoms):
                self.a2b.append([]) # nested list,

            # Get bond features
            for a1 in range(self.n_atoms): # init to num_nodes
                for a2 in range(a1 + 1, self.n_atoms):
                    bond = mol.GetBondBetweenAtoms(a1, a2) # return bond index ( bond indices start at 0)

                    if bond is None:
                        continue
                    # edge_index 
                    self.edge_index.extend([(a1, a2), (a2, a1)])

                    f_bond = bond_features(bond)
                    if bond_features_extra is not None:
                        descr = bond_features_extra[bond.GetIdx()].tolist()
                        if overwrite_default_bond_features:
                            f_bond = descr
                        else:
                            f_bond += descr

                    # append twice for fwd, rev bond, atom feature only append once
                    #self.f_bonds.append(self.f_atoms[a1] + f_bond) # fwd:  a1--> a2, cat atom and bond features
                    #self.f_bonds.append(self.f_atoms[a2] + f_bond) # rev:  a2--> a1

                    # maybe this is more useful for neural network, instead concat directly above
                    self.f_bonds.append(f_bond)
                    self.f_bonds.append(f_bond)

                    # Update index mappings
                    b1 = self.n_bonds # init 0, bond index
                    b2 = b1 + 1 # rev bond, index + 1

                    self.a2b[a2].append(b1)  # b1 = a1 --> a2, a2 stores incoming bond index (fwd_message), nested list (2d), e,g [[1,3,5,7],[0],[2],[4,9,11]...]
                    self.b2a.append(a1) # fwd_bond index map to a1, list (1d)
                    self.a2b[a1].append(b2)  # b2 = a2 --> a1, a1 stores incoming bond index (rev_message)
                    self.b2a.append(a2) # rev_bond index map to a2
                    self.b2revb.append(b2) # record rev_message index, list (1d)
                    self.b2revb.append(b1) # record rev_mesage index
                    self.n_bonds += 2 # update in paired (fwd, rev)

            if bond_features_extra is not None and len(bond_features_extra) != self.n_bonds / 2:
                raise ValueError(f'The number of bonds in {Chem.MolToSmiles(mol)} is different from the length of '
                                 f'the extra bond features')
           
        else: # Reaction mode
            if atom_features_extra is not None:
                raise NotImplementedError('Extra atom features are currently not supported for reactions')
            if bond_features_extra is not None:
                raise NotImplementedError('Extra bond features are currently not supported for reactions')

            mol_reac = mol[0]
            mol_prod = mol[1]
            ri2pi, pio, rio = map_reac_to_prod(mol_reac, mol_prod)
           
            # Get atom features
            f_atoms_reac = [atom_features(atom) for atom in mol_reac.GetAtoms()] + [atom_features(None) for index in pio]
            f_atoms_prod = [atom_features(mol_prod.GetAtomWithIdx(ri2pi[atom.GetIdx()])) if atom.GetIdx() not in rio else
                            atom_features(None) for atom in mol_reac.GetAtoms()] + [atom_features(mol_prod.GetAtomWithIdx(index)) for index in pio]
            
            if self.reaction_mode in ['reac_diff','prod_diff']:
                f_atoms_diff = [list(map(lambda x, y: x - y, ii, jj)) for ii, jj in zip(f_atoms_prod, f_atoms_reac)]
            if self.reaction_mode == 'reac_prod':
                self.f_atoms = [x+y[MAX_ATOMIC_NUM+1:] for x,y in zip(f_atoms_reac, f_atoms_prod)]
            elif self.reaction_mode == 'reac_diff':
                self.f_atoms = [x+y[MAX_ATOMIC_NUM+1:] for x,y in zip(f_atoms_reac, f_atoms_diff)]
            elif self.reaction_mode == 'prod_diff':
                self.f_atoms = [x+y[MAX_ATOMIC_NUM+1:] for x,y in zip(f_atoms_prod, f_atoms_diff)]
            self.n_atoms = len(self.f_atoms)
            n_atoms_reac = mol_reac.GetNumAtoms()

            # Initialize atom to bond mapping for each atom
            for _ in range(self.n_atoms):
                self.a2b.append([])

            # Get bond features
            for a1 in range(self.n_atoms): #n_atoms init 0
                for a2 in range(a1 + 1, self.n_atoms):
                    if a1 >= n_atoms_reac and a2 >= n_atoms_reac: # Both atoms only in product
                        bond_reac = None
                        bond_prod = mol_prod.GetBondBetweenAtoms(pio[a1 - n_atoms_reac], pio[a2 - n_atoms_reac])
                    elif a1 < n_atoms_reac and a2 >= n_atoms_reac: # One atom only in product
                        bond_reac = None
                        if a1 in ri2pi.keys():
                            bond_prod = mol_prod.GetBondBetweenAtoms(ri2pi[a1], pio[a2 - n_atoms_reac])
                        else:
                            bond_prod = None # Atom atom only in reactant, the other only in product
                    else:
                        bond_reac = mol_reac.GetBondBetweenAtoms(a1, a2)
                        if a1 in ri2pi.keys() and a2 in ri2pi.keys():
                            bond_prod = mol_prod.GetBondBetweenAtoms(ri2pi[a1], ri2pi[a2]) #Both atoms in both reactant and product
                        else:
                            bond_prod = None # One or both atoms only in reactant

                    if bond_reac is None and bond_prod is None:
                        continue

                    f_bond_reac = bond_features(bond_reac)
                    f_bond_prod = bond_features(bond_prod)
                    if self.reaction_mode in ['reac_diff', 'prod_diff']:
                        f_bond_diff = [y - x for x, y in zip(f_bond_reac, f_bond_prod)]
                    if self.reaction_mode == 'reac_prod':
                        f_bond = f_bond_reac + f_bond_prod
                    elif self.reaction_mode == 'reac_diff':
                        f_bond = f_bond_reac + f_bond_diff
                    elif self.reaction_mode == 'prod_diff':
                        f_bond = f_bond_prod + f_bond_diff
                    self.f_bonds.append(self.f_atoms[a1] + f_bond)
                    self.f_bonds.append(self.f_atoms[a2] + f_bond)

                    # Update index mappings
                    b1 = self.n_bonds # init 0
                    b2 = b1 + 1
                    self.a2b[a2].append(b1)  # b1 = a1 --> a2
                    self.b2a.append(a1)
                    self.a2b[a1].append(b2)  # b2 = a2 --> a1
                    self.b2a.append(a2)
                    self.b2revb.append(b2)
                    self.b2revb.append(b1)
                    self.n_bonds += 2  

    def node2linegraph(self) -> Tuple[List[int], List[int]]:
        """
        Inspired by torch_geometric.transforms.line_graph.

        To do edge message passing, we need to convert to line_graph (edge become node, edge_attr become node_attr), 
        so we can use pyg's message passing framework (only works for node_graph)
        
        """
        # force to directed graph
        bond_index = list(range(len(self.b2a))) # b2a is directed, 1d list, num_bonds X 2
        count = [len(b) for b in self.a2b] # a2b is undirected graph, adjacency list
        cumsum = [0] + list(accumulate(count)) # atoms' increment index
        
        # adjacency list, construct target `node`
        edge_index_linegraph_1 = [ bond_index[ cumsum[self.b2a[j]] : cumsum[self.b2a[j] + 1] ] for j in range(self.b2a) ]

        # adjacency list, get source `node`
        edge_index_linegraph_0 = [ [j]*len(b) for j, b in enumerate(edge_index_linegraph_1) ]
        # flatten, concate list
        edge_index_linegraph_1 = chain(*edge_index_linegraph_1)
        edge_index_linegraph_0 = chain(*edge_index_linegraph_0)
        # line_graph's edge_index
        self.edge_index_linegraph = [edge_index_linegraph_0, edge_index_linegraph_1]
        return self.edge_index_linegraph

    def get_components(self, atom_messages: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor,
                                                                   torch.LongTensor, torch.LongTensor, torch.LongTensor,
                                                                   List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Returns the components of the :class:`BatchMolGraph`.

        The returned components are, in order:

        * :code:`f_atoms`
        * :code:`f_bonds`
        * :code:`a2b`
        * :code:`b2a`
        * :code:`b2revb`
        * :code:`a_scope`
        * :code:`b_scope`

        :param atom_messages: Whether to use atom messages instead of bond messages. This changes the bond feature
                              vector to contain only bond features rather than both atom and bond features.
        :return: A tuple containing PyTorch tensors with the atom features, bond features, graph structure,
                 and scope of the atoms and bonds (i.e., the indices of the molecules they belong to).
        """
        if atom_messages:
            f_bonds = self.f_bonds[:, -get_bond_fdim(atom_messages=atom_messages,
                                                     overwrite_default_atom=self.overwrite_default_atom_features,
                                                     overwrite_default_bond=self.overwrite_default_bond_features):]
        else:
            f_bonds = self.f_bonds

        return self.f_atoms, f_bonds, self.a2b, self.b2a, self.b2revb #, self.a_scope, self.b_scope

    def get_b2b(self) -> torch.LongTensor:
        """
        Computes (if necessary) and returns a mapping from each bond index to all the incoming bond indices.

        :return: A PyTorch tensor containing the mapping from each bond index to all the incoming bond indices.
        """
        if self.b2b is None:
            b2b = self.a2b[self.b2a]  # num_bonds x max_num_bonds
            # b2b includes reverse edge for each bond so need to mask out
            revmask = (b2b != self.b2revb.unsqueeze(1).repeat(1, b2b.size(1))).long()  # num_bonds x max_num_bonds
            self.b2b = b2b * revmask

        return self.b2b

    def get_a2a(self) -> torch.LongTensor:
        """
        Computes (if necessary) and returns a mapping from each atom index to all neighboring atom indices.

        :return: A PyTorch tensor containing the mapping from each bond index to all the incoming bond indices.
        """
        if self.a2a is None:
            # b = a1 --> a2
            # a2b maps a2 to all incoming bonds b
            # b2a maps each bond b to the atom it comes from a1
            # thus b2a[a2b] maps atom a2 to neighboring atoms a1
            self.a2a = self.b2a[self.a2b]  # num_atoms x max_num_bonds

        return self.a2a