# DrugAI

Compound libraries

Note: SDF file could be parsed by RDKit


1. [Drugbank](https://go.drugbank.com/releases/latest)
    - [API](https://dev.drugbank.com/guides/api)
    - Need to signup with your academic email, then you could download the molecules.
    - Clinical trial molecules (11,294) from DrugBank were used with the NASH stage III markers
2. [PubChem](https://pubchemdocs.ncbi.nlm.nih.gov/downloads)
    - [API](https://pubchemdocs.ncbi.nlm.nih.gov/pug-rest)
    - PubChemPy
    - SMILES representation for virtual screen (Bulk download): https://ftp.ncbi.nlm.nih.gov/pubchem
        - wget -r ftp://ftp.ncbi.nlm.nih.gov/pubchem/Compound/CURRENT-Full/SDF/
    - 3D structrue for virtual screen (Bulk download):
        - wet -r ftp://ftp.ncbi.nlm.nih.gov/pubchem/Compound_3D/01_conf_per_cmpd/SDF/
        - select 1 compound 1 3D structruture

3. [ChEMBL](https://chembl.gitbook.io/chembl-interface-documentation/downloads) :
ChEMBL is a manually curated database of bioactive molecules with drug-like properties. It brings together chemical, bioactivity and genomic data to aid the translation of genomic information into effective new drugs.

    - [chembl_29_chemreps.txt.gz](https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/)
    - [chembl_29.sdf.gz](https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/)

4. [HMDB](https://hmdb.ca/downloads) :
    - [Metabolite Structures](https://hmdb.ca/system/downloads/current/structures.zip)
    - [All metabolites](https://hmdb.ca/system/downloads/current/hmdb_metabolites.zip)

5. [LIPID MAPS](https://www.lipidmaps.org/resources/databases/index.php)
    - [LMSD](https://www.lipidmaps.org/files/?file=LMSD&ext=sdf.zip)
    - [API](https://www.lipidmaps.org/data/structure/programmaticaccess.html) access
    - As of 09/03/2021, LMSD contains 46150 unique lipid structures, making it the largest public lipid-only database in the world.
6. Metlin
    - 960,000 compounds
    - not available 


7. TargetMol
    - Catalog No. L6000: [Natural Compound Library for HTS](https://www.targetmol.com/compound-library/Natural%20Compound%20Library%20for%20HTS)
        - a unique collection of 2960 natural products with known bioactivity, wide source, and high cost effectiveness
    - Catalog No. L4200: [FDA-approved Drug Library](https://www.targetmol.com/compound-library/FDA-approved%20Drug%20Library)
        - 470 FDA approved drugs

8. others:
   - all kinds compounds are curated here, include targetmol
   - http://www2.sibcb.ac.cn/cp13-5_3.asp