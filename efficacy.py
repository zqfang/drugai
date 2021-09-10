
import sys, os
import pandas as pd
import numpy as np

from typing import List, Tuple, Dict, Union
from argparse import ArgumentParser
from pathlib import Path



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
    parser.add_argument('--baseline',  type=str, default=None,
                        help='Path to level3_beta_ctrl_n188708x12323.gctx. Control data for baseline correction')
    parser.add_argument('--statistic',  type=str, default='kolmogorov–smirnov', choices={'kolmogorov–smirnov', 'gsea'},
                        help="Efficacy score type, choose from {'kolmogorov–smirnov', 'gsea'}. Default: kolmogorov–smirnov")
    args = parser.parse_args()
    outfile = Path(args.predicts).stem + ".efficacy.csv"
    if args.output is not None: outfile = args.output
    
    efficacy = EfficacyPred(args.weights, args.up, args.down, baseline=args.baseline)
    preds = pd.read_csv(args.predicts, index_col=0) 
    # overwrite outputfile
    if os.path.exists(outfile): os.remove(outfile)
    # append mode
    output = open(outfile, 'a')
    output.write("SMILES,es,esup,esdown\n")
    step = 5000
    for i in range(0, len(preds), step):
        scores = efficacy.compute_score(preds.iloc[i:i+step], statistic=args.stat)
        scores.to_csv(output, mode='a', header=False)
    ## 
    #efficacy_socres = pd.concat(efficacy_socres)
    #efficacy_socres.to_csv(args.output)
    output.close()

class EfficacyPred:
    def __init__(self, weight: Union[str, pd.DataFrame], up: List[str] = None, down: List[str] = None, 
                 baseline: Union[str, pd.DataFrame]=None):
        """
        Args:

        up:  entrz ids of up-regulated genes
        down: entrz ids of down-regulated genes
        preds: filepath to GNN predition output (978 landmark genes)
        weight: filepath to GSE92743_Broad_OLS_WEIGHTS_n979x11350.gctx or a dataframe
        """
        self.W = self._get_weight(weight) 
        self.genes = self.W.index.append(self.W.columns[1:]).to_list() # 11350 + 978 = 12328 genes

        self.up = self._get_genes(up) if up else None
        self.down = self._get_genes(down) if down else None

        if baseline is not None:
            self.benchmark = self.get_baseline(baseline)

    def _get_weight(self, weight):
        """
        map 978 genes to 12328 genes
        Download the weight matrix from GSE92743. This file has correct Entrez id
        GSE92743_Broad_OLS_WEIGHTS_n979x11350.gctx.gz
        q
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

        return weight

    def get_baseline(self, cp_ctrl: Union[pd.DataFrame, str]):
        if isinstance(cp_ctrl, str):
            if cp_ctrl.endswith("gctx"):
                from cmapPy.pandasGEXpress.parse import parse
                cp_gctoo_ctrl = parse(cp_ctrl) # used for benchmark means
                cp =  cp_gctoo_ctrl.data_df
            elif cp_ctrl.endswith("csv"):
                cp = pd.read_csv(cp_ctrl)
            else:
                cp = pd.read_table(cp_ctrl)

            benchmark = cp.mean(axis=1)
            return benchmark

        elif isinstance(cp_ctrl, pd.DataFrame):
            return cp_ctrl
        else:
            raise Exception("Could not understand cp_ctrl, control data are expected as the basal expresion level")

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
           landmarks: shape (num_smiles, 978), predicted landmark expression
        """
        # insert offset (bias)
        exprs = landmarks.copy()
        # inplace insert ones in the first column
        exprs.insert(loc=0, column="OFFSET", value=1) 
        # match the order of landmark genes
        W = self.W.loc[:, exprs.columns] # 11350 x 979
        # get predicted 11350 genes' expression of compounds
        # [D x SIMELS ] = [D x L+1] * [L+1 x SMILES]
        L11350 = W @ exprs.T # linear transform, infer 11350 genes
        # non-negative !
        # L11350.clip(lower=0, inplace=True)
        L12328_df = pd.concat([landmarks.T, L11350]) # note, columns aligned by index automatically
        return L12328_df

    def compute_score(self, preds: Union[str, pd.DataFrame], 
                            up: List[str] = None, 
                            down: List[str] = None, 
                            baseline: Union[pd.Series, str] = None,
                            statistic: str = "kolmogorov–smirnov") -> pd.DataFrame:
        """

        Args::
            preds: (num_smiles x 978) prediction output from GNN model
                row index: SMILE Strings
                column index: Entrez IDs
            baseline: shape (12328,). Average expression level of controls. (Level3 data)        
            up: list of entrezids, or a txt file 
            down: list of entrzids, or a txt file
            statistic: the efficacy scoring method. kolmogorov–smirnov or gsea.

        """
        exprs = self.get_conectivity(preds) # num_smiles x 978
        L12328_df = self.infer_expression(exprs) # 12328 x num_smiles 
        # handle genes
        up = self._get_genes(up) if up else self.up 
        down = self._get_genes(down) if down else self.down
        print(f"up genes: {len(up)}; down genes: {len(down)}")
        # FIXME: it looks like the z-socre,normalize,... are very critical for the es score representation
        # trick 1: minus the averge expression of controls
        if (baseline is None) and (not hasattr(self, 'benchmark')):
            # L12328_df = L12328_df - L12328_df.mean(axis=1).values.reshape((-1,1))
            L12328_df = L12328_df.subtract(L12328_df.mean(axis=1), axis=0)
        elif (baseline is None) and hasattr(self, 'benchmark'):
            basal = self.benchmark.loc[L12328_df.index] # note: benchmark must be shape(12328,)
            L12328_df = L12328_df.subtract(basal, axis=0) 
        elif isinstance(baseline, pd.Series):
            L12328_df = L12328_df.subtract(baseline.loc[L12328_df.index], axis=0) 
        elif isinstance(baseline, str):
            basal = self.get_baseline(baseline)
            L12328_df = L12328_df.subtract(basal.loc[L12328_df.index], axis=0)
        else:
            raise Exception("Could not understand input data: baseline")
        self._data = L12328_df.copy()
        # compute score
        if score_type.lower() == 'kolmogorov–smirnov':
            cs = self.ks_score(L12328_df, up, down)
        elif score_type.lower() =='gsea':
            cs = self.gsea_score(L12328_df, up, down, 1.1)
        else:
            raise Exception("Unsupported score_type ! only select from {gsea, kolmogorov–smirnov}")
        return cs


    # This file consists of useful functions that are related to cmap
    def ks_score(self,  expression: pd.DataFrame, qup: List[str], qdown: List[str]):
        '''
        kolmogorov–smirnov score

        This function takes qup & qdown, which are lists of gene
        names, and  expression, a panda data frame of the expressions
        of genes as input, and output the connectivity score vector
        '''
        # This function takes a panda data frame of gene names and expressions
        # as an input, and output a data frame of gene names and ranks
        ranks = expression.rank(ascending=False, method="first")
        #ranks = 10000*ranks / ranks.shape[0]
        if qup and qdown:
            esup = self._ks_score(qup, ranks)
            esdown = self._ks_score(qdown, ranks)
            w = []
            for i in range(len(esup)):
                if esup[i]*esdown[i] <= 0:
                    w.append(esup[i]-esdown[i])
                else:
                    w.append(0)
            return pd.DataFrame({'es': w, 'esup': esup, 'esdown':esdown}, expression.columns)
        elif qup and qdown==None:
            esup = self._ks_score(qup, ranks)
            return pd.DataFrame(esup, expression.columns)
        elif qup == None and qdown:
            esdown = self._ks_score(qdown, ranks)
            return pd.DataFrame(esdown, expression.columns)
        else:
            return None

    def _ks_score(self, q: List[str], r1: pd.DataFrame):
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

    def gsea_score(self, expression, up, down, weighted_score):
        """
        gsea scoring method
        """
        # gene_set = up + down
        results = []
        for _, ser in expression.iteritems(): # iter columns
            ser = ser.sort_values(ascending=False) # FIXME: the order
            esup = self._gsea_score(ser.index.to_list(), ser.values, up, weighted_score)
            esdown = self._gsea_score(ser.index.to_list(), ser.values, down, weighted_score)
            if esup * esdown <= 0:
                es = esup - esdown
            else:
                es = 0
            results.append((es, esup, esdown))
        return pd.DataFrame(results, index= expression.columns, columns=['es', 'esup','esdown'])


    def _gsea_score(self, gene_list, correl_vector, gene_set, weighted_score_type=1, 
                    single=False, scale=False):
        """It has the same algorithm with GSEA and ssGSEA.

        :param expression: shape [num_smiles, num_genes]
        :param gene_set: up and down gene list (concated)
        :param weighted_score_type float: power of floats

        :return:
        ES: Enrichment score (real number between -1 and +1)
        """
        N = len(gene_list)
        tag_indicator = np.in1d(gene_list, gene_set, assume_unique=True).astype(int)

        if weighted_score_type == 0 :
            correl_vector = np.repeat(1, N)
        else:
            correl_vector = np.abs(correl_vector)**weighted_score_type

        # GSEA Enrichment Score
        Nhint = tag_indicator.sum()
        sum_correl_tag = np.sum(correl_vector*tag_indicator)

        no_tag_indicator = 1 - tag_indicator
        Nmiss =  N - Nhint
        norm_tag =  1.0/sum_correl_tag
        norm_no_tag = 1.0/Nmiss
        RES = np.cumsum(tag_indicator * correl_vector * norm_tag - no_tag_indicator * norm_no_tag)

        if scale: RES = RES / N
        if single: # ssGSEA
            es = RES.sum()
        else:
            max_ES, min_ES =  RES.max(), RES.min()
            es =  max_ES if np.abs(max_ES) > np.abs(min_ES) else min_ES 
            # extract values
        return es


    if __name__ == '__main__':
        main()