from argparse import ArgumentParser
import pandas as pd
import numpy as np
import sys, os
from typing import List, Tuple, Dict, Union


def main(args):
    parser = ArgumentParser()
    parser.add_argument('--predicts',  type=str, required=True,
                        help='predicted landmark genes expression file')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to GSE92743_Broad_OLS_WEIGHTS_n979x11350.gctx')
    parser.add_argument('--up', type=str, default=None,
                        help='Path to up-regulated genes. One EntrezID per row')
    parser.add_argument('--down', type=str, default=None,
                        help='Path to up-regulated genes. One EntrezID per row')
    parser.add_argument('--output', type=str, default="efficacy.csv",
                        help='Path to output file')
    args = parser.parse_args()
    efficacy = EfficacyPred(args.weights, args.up, args.down)
    scores = efficacy.compute_score(args.predicts)
    scores.columns = ['ES']
    scores.to_csv(args.output)

class EfficacyPred:
    def __init__(self, weight: Union[str, pd.DataFrame], up: List[str] = None, down: List[str] = None):
        """
        Args:

        up:  entrz ids of up-regulated genes
        down: entrz ids of down-regulated genes
        preds: filepath to GNN predition output (978 landmark genes)
        weight: filepath to GSE92743_Broad_OLS_WEIGHTS_n979x11350.gctx or a dataframe
        """
        self.W, self.bias = self._get_weight(weight) 
        self.genes = self.W.index.append(self.W.columns[1:]).to_list() # 11350 + 979 = 12328 genes

        self.up = self._get_genes(up) if up else None
        self.down = self._get_genes(down) if down else None

    def _get_weight(self, weight):
        """
        map 978 genes to 12328 genes
        Download the weight matrix from GSE92743. This file has correct Entrez id
        GSE92743_Broad_OLS_WEIGHTS_n979x11350.gctx.gz
        
        ## Need to figure out what DS_GEO_OLS_WEIGHTS_n979x21290.gctx use for ?
        ## This file are all affymatrix ids, not Entrez id !!!
        ## DS_GEO_OLS_WEIGHTS_n979x21290.gctx: file found in GSE92742_Broad_LINCS_auxiliary_datasets.tar.gz	
        ## The matrix of weights learned by training the L1000 inference algorithm, ordinary least squares (OLS) linear regression, on DSGEO.

        ## rows: 21,290 inferred features
        ## columns: 978 landmark genes + intercept = 979
        """
        if isinstance(weight, pd.DataFrame):
            pass
        elif isinstance(weight, str):
            if weight.endswith("gctx"):
                from cmapPy.pandasGEXpress.parse import parse
                weight = parse(weight).data_df
            elif weight.endswith("csv"):
                weight = pd.read_csv(weight)
            else:
                weight = pd.read_table(weight)
        else:
            raise Exception("Unsupported file format")

        assert weight.shape[1] == 978
        # weights 
        W = weights.iloc[:, 1:] # 18815 X 978
        bias = weight.iloc[:, 0] # intercept (987,)
        # FIXME: duplicated gene names, while affy_id is unique
        return W, bias

    def get_conectivity(self, preds: Union[str, pd.DataFrame]) -> pd.DataFrame:
        """
        The expression levels for 978 genes
        """
        if isinstance(preds, pd.DataFrame):
            return preds
        preds = pd.read_csv(preds, index_col=0) 
        # index are SMILE strings
        # column are entrz ids of landmark genes
        # FIXME: landmark genes orders, better to dobule check !

        return preds

    def _get_genes(self, genes: Union[str, List[str]]) -> List[str]:
        """get the gene signatures
           one id per row, entrze id
        """
        if isinstance(genes, str):
            up = pd.read_table(genes, header=None, comment="#")
            ups= up.values.astype(str)
            ups = list(np.squeeze(ups))
        elif isinstance(genes, (list, tuple)):
            ups = genes
        else:
            raise Exception("genes must be filepath, list or tuple")
        # filter genes
        ups_new = [str(i) for i in ups if str(i) in self.genes]

        if len(ups_new) < 1: 
            raise Exception("No genes found. Please input proper Entrez id")
        return ups_new


    def infer_expression(self, exprs: Union[str, pd.DataFrame]):
        """compute the enrichment/efficacy score
           exprs: shape (num_smiles, 978)
        """
        # get predicted 12328 genes' expression of compounds
        # match the order of landmark genes
        W = self.W.loc[:, exprs.columns] # 11350 x 978
        L11350 = W.values @ exprs.T.values  + self.bias.values.reshape(-1,1) # linear transform, infer 11350 genes
        L11350_df = pd.DataFrame(L11350, index=W.index, columns=exprs.index) 
        L12328_df = pd.concat([exprs.T, L11350_df]) # note, columns aligned by index automatically
        return L12328_df

    def compute_score(self, preds: Union[str, pd.DataFrame], up: List[str] = None, down: List[str]=None) -> pd.DataFrame:
        """
        preds: (num_smiles x 978) prediction output from GNN model
               row index: SMILE Strings
               column index: Entrez IDs
        
        up: list of entrezids, or a txt file 
        down: list of entrzids, or a txt file

        """
        exprs = self.get_conectivity(preds) # num_smiles x 978
        L12328_df = self.infer_expression(exprs) # 12328 x num_smiles 
        # handle genes
        up = self._get_genes(up) if up else self.up 
        down = self._get_genes(down) if down else self.down
        # compute score
        cs = self._connectivity_socre(up, down, expression=L12328_df)
        return cs


    # This file consists of useful functions that are related to cmap
    def _connectivity_socre(self, qup: List[str], qdown: List[str], expression: pd.DataFrame):
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

        Kolmogorov–Smirnov test
        '''
        if not isinstance(r1, pd.DataFrame):
            raise Exception("r1 must be a pd.DataFrame")
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


if __name__ == '__main__':
    main()