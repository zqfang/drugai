from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import numpy as np
import sys, os
from typing import List, Tuple, Dict, Union


def main():
    parser = ArgumentParser()
    parser.add_argument('--predicts',  type=str, required=True,
                        help='predicted landmark genes expression file')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to GSE92743_Broad_OLS_WEIGHTS_n979x11350.gctx')
    parser.add_argument('--up', type=str, default=None,
                        help='Path to up-regulated genes. One EntrezID per row')
    parser.add_argument('--down', type=str, default=None,
                        help='Path to up-regulated genes. One EntrezID per row')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to output file')
    args = parser.parse_args()
    outfile = Path(args.predicts).stem + ".efficacy.csv"
    if args.output is not None: outfile = args.output
    
    efficacy = EfficacyPred(args.weights, args.up, args.down)
    preds = pd.read_csv(args.predicts, index_col=0) 
    efficacy_socres = []
    # overwrite outputfile
    if os.path.exists(outfile): os.remove(outfile)
    # append mode
    output = open(outfile, 'a')
    step = 5000
    for i in range(0, len(preds), step):
        scores = efficacy.compute_score(preds.iloc[i:i+step])
        #scores.columns = ['ES']
        scores.to_csv(output, mode='a', header=False)
        efficacy_socres.append(scores)
    ## 
    #efficacy_socres = pd.concat(efficacy_socres)
    #efficacy_socres.to_csv(args.output)
    output.close()

class EfficacyPred:
    def __init__(self, weight: Union[str, pd.DataFrame], up: List[str] = None, down: List[str] = None):
        """
        Args:

        up:  entrz ids of up-regulated genes
        down: entrz ids of down-regulated genes
        preds: filepath to GNN predition output (978 landmark genes)
        weight: filepath to GSE92743_Broad_OLS_WEIGHTS_n979x11350.gctx or a dataframe
        """
        self.W = self._get_weight(weight) 
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

        assert weight.shape[1] == 979 # first column is offset
        # weights 
        #W = weight.iloc[:, 1:] # 18815 X 978
        #bias = weight.iloc[:, 0] # W_0 (978,) 
        
        return weight

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
            up = pd.read_table(genes, header=None, comment="#", dtype=str)
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


    def infer_expression(self, landmarks: Union[str, pd.DataFrame]):
        """compute the enrichment/efficacy score
           exprs: shape (num_smiles, 978)
        """
        # insert offset (bias)
        exprs = landmarks.copy()
        exprs.insert(loc=0, column="OFFSET", value=1) # inplace insert
        # get predicted 12328 genes' expression of compounds
        # match the order of landmark genes
        W = self.W.loc[:, exprs.columns] # 11350 x 979
        # [D x S] = [D x L+1] * [L+1 x S]
        L11350 = W @ exprs.T # linear transform, infer 11350 genes
        # non-negative !
        L11350.clip(lower=0, inplace=True)
        L12328_df = pd.concat([landmarks.T, L11350]) # note, columns aligned by index automatically
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
        print(f"up genes: {len(up)}; down genes: {len(down)}")
        # FIXME: it looks like the z-socre,normalize,... are very critical for the es score representation
        # do more trick here to test best method
        L12328_df = L12328_df - L12328_df.mean(axis=1).values.reshape((-1,1))
        # compute score
        cs = self._connectivity_score(up, down, expression=L12328_df)
        return cs


    # This file consists of useful functions that are related to cmap
    def _connectivity_score(self, qup: List[str], qdown: List[str], expression: pd.DataFrame):
        '''
        This function takes qup & qdown, which are lists of gene
        names, and  expression, a panda data frame of the expressions
        of genes as input, and output the connectivity score vector
        '''
        # This function takes a panda data frame of gene names and expressions
        # as an input, and output a data frame of gene names and ranks
        ranks = expression.rank(ascending=False, method="first")
        #ranks = 10000*ranks / ranks.shape[0]
        if qup and qdown:
            esup = self.ks_score(qup, ranks)
            esdown = self.ks_score(qdown, ranks)
            w = []
            for i in range(len(esup)):
                if esup[i]*esdown[i] <= 0:
                    w.append(esup[i]-esdown[i])
                else:
                    w.append(0)
            return pd.DataFrame({'es': w, 'esup': esup, 'esdown':esdown}, expression.columns)
        elif qup and qdown==None:
            esup = self.ks_score(qup, ranks)
            return pd.DataFrame(esup, expression.columns)
        elif qup == None and qdown:
            esdown = self.ks_score(qdown, ranks)
            return pd.DataFrame(esdown, expression.columns)
        else:
            return None

    def ks_score(self, q: List[str], r1: pd.DataFrame):
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
            sub = r1.loc[q,:] # genes X samples
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

    def gsea_score(self, expression: pd.DataFrame, gene_set, weighted_score_type=1, single=False):
        """This is the most important function of GSEApy. It has the same algorithm with GSEA and ssGSEA.
        :param gene_set: up and down gene list (concated)
        :param weighted_score_type:  It's the same with gsea's weighted_score method. Weighting by the correlation
                                is a very reasonable choice that allows significant gene sets with less than perfect coherence.
                                options: 0(classic),1,1.5,2. default:1. if one is interested in penalizing sets for lack of
                                coherence or to discover sets with any type of nonrandom distribution of tags, a value p < 1
                                might be appropriate. On the other hand, if one uses sets with large number of genes and only
                                a small subset of those is expected to be coherent, then one could consider using p > 1.
                                Our recommendation is to use p = 1 and use other settings only if you are very experienced
                                with the method and its behavior.
        :return:
        ES: Enrichment score (real number between -1 and +1)
        """
        N = len(expression)
        gene_list = expression.index.values
        # Test whether each element of a 1-D array is also present in a second array
        # It's more intuitive here than original enrichment_score source code.
        # use .astype to covert bool to integer
        tag_indicator = np.in1d(gene_list, gene_set, assume_unique=True).astype(int)  # notice that the sign is 0 (no tag) or 1 (tag)
        correl_mat = expression.values
        if weighted_score_type == 0 :
            correl_mat = np.ones((expression.shape))
        else:
            correl_mat = np.abs(correl_mat)**weighted_score_type

        # get indices of tag_indicator
        hit_ind = np.flatnonzero(tag_indicator).tolist()
        # set axis to 1, because we have 2D array
        axis = 1
        tag_indicator = np.tile(tag_indicator, (expression.shape[0],1))

        Nhint = tag_indicator.sum(axis=axis, keepdims=True)
        sum_correl_tag = np.sum(correl_mat*tag_indicator, axis=axis, keepdims=True)
        # compute ES score, the code below is identical to gsea enrichment_score method.
        no_tag_indicator = 1 - tag_indicator
        Nmiss =  N - Nhint
        norm_tag =  1.0/sum_correl_tag
        norm_no_tag = 1.0/Nmiss

        RES = np.cumsum(tag_indicator * correl_mat * norm_tag - no_tag_indicator * norm_no_tag, axis=axis)

        if single:
            es_vec = RES.sum(axis=axis)
        else:
            max_ES, min_ES =  RES.max(axis=axis), RES.min(axis=axis)
            es_vec = np.where(np.abs(max_ES) > np.abs(min_ES), max_ES, min_ES)
        # extract values
        es, esnull, RES = es_vec[-1], es_vec[:-1], RES[-1,:]

        return es, esnull, hit_ind, RES


if __name__ == '__main__':
    main()