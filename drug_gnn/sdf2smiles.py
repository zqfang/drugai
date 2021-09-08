from rdkit import Chem
from rdkit.Chem import PandasTools
import re, sys, os
# from rdkit.Chem.Draw import rdMolDraw2D
# from IPython.display import SVG

# molecules = []
# # For reading Smiles or SDF files with large number of records concurrently, MultithreadedMolSuppliers can be used 
# with Chem.MultithreadedSDMolSupplier('Compounds/hmdb_structures.sdf') as suppl:
#     for mol in suppl:
#         if mol is None: continue
#         smile = Chem.MolToSmiles(mol)
#         molecules.append(smile)



## Draw molecules
# d2d = rdMolDraw2D.MolDraw2DSVG(300,300)
# d2d.drawOptions().addAtomIndices=True
# d2d.DrawMolecule(mol)
# d2d.FinishDrawing()
# SVG(d2d.GetDrawingText())


if __name__ == "__main__":
    sdf = sys.argv[1]
    pout = sys.argv[2]
    df = PandasTools.LoadSDF(sdf)
    #pout = re.sub("sdf","SMILES", str(pc), flags=re.IGNORECASE)
    #pout = pout.replace(".gz","")
    #if os.path.exists(pout): continue
    #tmp = PandasTools.LoadSDF(str(pc))
    smiles = df['PUBCHEM_OPENEYE_CAN_SMILES'].drop_duplicates()
    smiles.to_csv(pout, index=False, header=False, sep="\t")