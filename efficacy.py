import pandas as pd
from cmapPy.pandasGEXpress import parse
import numpy as np
import sys, os, ast
from typing import List, Tuple, Dict, Union


class ConnectivityScore:
    # This file consists of useful functions that are related to cmap
    def __call__(self, qup: List[str], qdown: List[str], expression: pd.DataFrame):
        '''
        This function takes qup & qdown, which are lists of gene
        names, and  expression, a panda data frame of the expressions
        of genes as input, and output the connectivity score vector
        '''
        # This function takes a panda data frame of gene names and expressions
        # as an input, and output a data frame of gene names and ranks
        ranks = expression.rank(ascending=False, method="first")
        if qup and qdown:
            esup = self.enrichment_score(qup, ranks)
            esdown = self.enrichment_score(qdown, ranks)
            w = []
            for i in range(len(esup)):
                if esup[i]*esdown[i] <= 0:
                    w.append(esup[i]-esdown[i])
                else:
                    w.append(0)
            return pd.DataFrame(w, expression.columns)
        elif qup and qdown==None:
            esup = self.enrichment_score(qup, ranks)
            return pd.DataFrame(esup, expression.columns)
        elif qup == None and qdown:
            esdown = self.enrichment_score(qdown, ranks)
            return pd.DataFrame(esdown, expression.columns)
        else:
            return None

    def enrichment_score(self, q: List[str], r1: pd.DataFrame):
        '''
        This function takes q, a list of gene names, and r1, a panda data
        frame as the input, and output the enrichment score vector
        '''
        if not isinstance(r1, pd.DataFrame):
            raise Exception("r1 must be a pd.Seires")
        if len(q) == 0:
            ks = 0
        elif len(q) == 1:
            ks = r1.loc[q,:]
            ks.index = [0]
            ks = ks.T
        else:
            n = r1.shape[0]
            sub = r1.loc[q,:]
            J = sub.rank()
            a_vect = J/len(q)-sub/n
            b_vect = (sub-1)/n-(J-1)/len(q)
            a = a_vect.max()
            b = b_vect.max()
            ks = []
            for i in range(len(a)):
                if a[i] > b[i]:
                    ks.append(a[i])
                else:
                    ks.append(-b[i])
        return ks


class EfficacyPred:
    def __init__(self, up: List[str], down: List[str], weight_path: str, weight_meta: str):
        """
        Args:

        up:  entrz ids of up-regulated genes
        down: entrz ids of down-regulated genes
        preds: filepath to GNN predition output (978 landmark genes)
        weight_path: filepath to DS_GEO_OLS_WEIGHTS_n979x21290.gctx
        weight_meta: filepath to DS_GEO_OLS_WEIGHTS_n979x21290.gctx's metadata
        
        """
        self.weight_path = weight_path
        self.weight_meta = weight_meta
        self.W, self.bias, self.genes = self._get_weight()
        self.affy2entrz = 
        if not isinstance(up, list):
            up = self._get_genes(up)
        if not isinstance(down, list):
            down = self._get_genes(down)
        # drop genes not in gene list
        self.up = [str(u) for u in up if str(u) in list(self.genes["pr_gene_id"])] 
        self.down = [str(d) for d in down if str(d) in list(self.genes["pr_gene_id"])]
        self._cs = ConnectivityScore()

    def _get_weight(self):
        """
        map 978 genes to 12328 genes

        DS_GEO_OLS_WEIGHTS_n979x21290.gctx	
        The matrix of weights learned by training the L1000 inference algorithm, ordinary least squares (OLS) linear regression, on DSGEO.

        rows: 21,290 inferred features
        columns: 978 landmark genes + intercept = 979

        file found in: GSE92742_Broad_LINCS_auxiliary_datasets.tar.gz
        """
        weight = parse(self.weight_path).data_df
        meta = pd.read_table(self.weight_meta, index_col=0, dtype=str)
        genes = meta.loc[weight.index]
        # 21290 x 978, note: affy_id could without gene symbol
        non_entrze_mask = genes['pr_gene_symbol'].str.startswith("-")
        W = weight.iloc[~non_entrze_mask, 1:] # 18815 X 978
        bias = weight.iloc[~non_entrze_mask, 0] # intercept (987,)
        # FIXME: duplicated gene names, while affy_id is unique
        return W, bias, genes

    def _get_conectivity(self, preds):
        """
        The expression levels for 978 genes
        """
        tmp = pd.read_csv(preds)['prediction'].apply(lambda x: ast.literal_eval(x)).to_list()
        preds = pd.DataFrame(tmp)
        # FIXME
        #self.smiles = 
        # A = 
        return preds


    def _get_genes(self, fl_name):
        """get the gene signatures
           one id per row, entrze id
        """
        up = pd.read_table(fl_name, header=None)
        ups= up.values.astype(int)
        print(ups.shape)
        ups = list(np.squeeze(ups))
        ups_new = [i for i in ups if i in list(self.genes["pr_gene_id"])]
        return ups_new


    def compute_cs(self, exprs: Union[str, pd.DataFrame]):
        """compute the enrichment score"""
        if not isinstance(exprs, pd.DataFrame):
            exprs = self._get_conectivity(exprs)
    
        # get predicted 12328 genes' expression of compounds
        # FIXME: match landmark genes
        self.exprs = self.W.values @ exprs.T.values  + self.bias.values.reshape(-1,1) # L12K
        # (genes, rows)
        L12k_df = pd.DataFrame(self.exprs, index=self.genes["pr_gene_id"], columns=exprs.index) 
        # FIXME: average genes
        L12k_df = L12k_df.groupby(level=0).mean() # 11806 x 978
        cs = self._cs(self.up, self.down, L12k_df)
        cs.columns = ['efficacy']
        self.efficacy_scores = cs
        return cs


if __name__ == '__main__':
    preds = sys.argv[1]
    up = sys.argv[2]
    down = sys.argv[3]

    weight_meta="CMAP_LINCS_2020/GSE92742_Broad_LINCS_auxiliary_datasets/Affymetrix_row_meta.txt",
    weight_path="CMAP_LINCS_2020/GSE92742_Broad_LINCS_auxiliary_datasets/DS_GEO_OLS_WEIGHTS_n979x21290.gctx"
    efficacy = EfficacyPred(up, down, weight_path, weight_meta)
    scores = efficacy.compute_cs(preds)
    scores.to_csv("efficacy.csv")