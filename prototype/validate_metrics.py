"""
validate_metrics.py - 20241031 - JC

WHAT IT DOES:
    - Loads embeddings from different models/datasets/tasks
    - Computes Dataeval metrics
    - Compares the differences of those metrics between models/datasets/tasks
    
SAMPLE COMMAND:
    python validate_metrics.py --model resnet18 --data MNIST --task classify --metric ks
    
Here's the different uses cases we want to look at:
Autoencoder with reconstruction
[ ]Clusterer         
[ ]BER   
[ ]Coverage   
[ ]Divergence   
[ ]Drift   

Autoencoder with task specific encoding
[ ]Clusterer   
[ ]BER   
[ ]Coverage   
[ ]Divergence   
[ ]Drift   

Embedding from state of the art model
[ ]Clusterer   
[ ]BER   
[ ]Coverage   
[ ]Divergence   
[ ]Drift   

PCA
[ ]Clusterer   
[ ]BER   
[ ]Divergence   
[ ]Drift

"""

import argparse
import os
import numpy as np
import json
from tqdm import tqdm
import copy
import pickle
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from domain_classifier import MVDC
from dataeval.detectors.drift import DriftCVM, DriftKS, DriftMMD
from dataeval.detectors.linters import Clusterer
from dataeval.metrics.bias import coverage
from dataeval.metrics.estimators import divergence
from dataeval.metrics.estimators import ber


def get_input_args():
    parser = argparse.ArgumentParser(conflict_handler='resolve', allow_abbrev=False)
    
    # PATHS
    parser.add_argument('--resdir',         type=str, 
                                            # default = '/mnt/nas_device_1/embeddings',
                                            default = '/home/jchristian/2033/metric_analysis',
                                            help="root directory where the timestamped results folders are")
    

    # RUN metadata
    parser.add_argument('--model',          type = str, 
                                            nargs="+",
                                            default = None,
                                            choices = ['resnet18', 'resnet34', 'segformer', 'pca'],
                                            help="model name of the run you want to retreive")
    
    parser.add_argument('--data',           type = str, 
                                            nargs="+",
                                            default = None,
                                            choices = ['MNIST','fmow'],
                                            help="dataset name of the run you want to retreive")
    
    parser.add_argument('--task',           type = str, 
                                            nargs="+",
                                            default = None,
                                            choices = ['ae','dae', 'classify'],
                                            help="dataset name of the run you want to retreive")
    
    parser.add_argument('--metric',         type = str, 
                                            default = 'mmd',
                                            choices = ['cvm', 'ks', 'mmd', 'mvdc',
                                                       'clusterer',
                                                       'ber', 'coverage', 'divergence'],
                                            help="metric to analyze")
    
    # parser.add_argument('--limit',          type=int, 
    #                                         default = None,
    #                                         help="truncates embeddings in BOTH nsamples, nfeats")
    
    
    # VISUALIZATION
    # parser.add_argument('--showme',         action='store_true',
    #                                         help='If true, show plots')
    
    opt = parser.parse_args()    
    return opt


def load_legend(resdir, fn='results.json'):
    # Load the 'legend' that maps the model/dataset/task to a run name
    legfn = os.path.join(resdir, fn)
    with open(legfn, 'r') as f:
        resdict = json.load(f)
    return resdict


def get_embeddings_labels(runfp, suffix='embed'):
    embfiles = os.listdir(os.path.join(runfp, 'embeddings'))
    trnfn = [item for item in embfiles if item.find(f'TRN_{suffix}') >=0][0]
    tstfn = [item for item in embfiles if item.find(f'TST_{suffix}') >=0][0]
    trn = np.load(os.path.join(runfp, 'embeddings', trnfn))
    tst = np.load(os.path.join(runfp, 'embeddings', tstfn))
    return trn, tst 



class Loader:
    def __init__(self, rootdir, runlegend):
        
        # Root directory where embeddings are stored
        self.rootdir = rootdir
        
        # We're going to evaluate metrics over embeddings from these combinations that we've specifically ran
        self.models = ['resnet18', 'resnet34', 'segformer', 'pca']
        self.data = ['MNIST','fmow']
        self.tasks = ['ae','dae', 'classify']
        
        # This dictionary tells us what run is mapped to what combination
        self.legend = runlegend
        
        # Copy over the dictionary structure but clear it out
        # All these are accessible via self.{whatever}[m][d][t] for m: model, d: data, t: task
        self.embeddings = self.clear_dict_values(runlegend)
        self.metrics = self.clear_dict_values(runlegend)
        self.labels = self.clear_dict_values(runlegend)
        self.outliers = self.clear_dict_values(runlegend)
        self.duplicates = self.clear_dict_values(runlegend)
        
        # TODO, TEMP limit on (n_samples, n_feats)
        # self.limit = (100,4)
        self.limit = None

        self.load()
        
        
    def clear_dict_values(self, origdict):
        newdict = copy.deepcopy(origdict) 
        for m in self.models:
            for d in self.data:
                for t in self.tasks:
                    newdict[m][d][t] = None
        return newdict
                    
        
    def load(self):
        # Load the cached embeddings into the dictionary structure
        for m in self.models:
            for d in self.data:
                for t in self.tasks:
                    try:
                        runfp = os.path.join(self.rootdir, self.legend[m][d][t])
                    except:
                        # Some elements may be empty, ex. pca only did task=ae
                        continue
                    embTRN, embTST = get_embeddings_labels(runfp, suffix='embed')
                    self.embeddings[m][d][t] = (embTRN, embTST)
                    # TODO: labels only make sense for classify...should have them for MNIST, fmow regardless
                    lblTRN, lblTST = get_embeddings_labels(runfp, suffix='label')
                    self.labels[m][d][t] = (lblTRN, lblTST)
                    
                    
    def get(self, mymodel, mydata, mytask):
        # Pull a set of training/test embeddings from the dict
        try:
            embTRN, embTST = self.embeddings[mymodel][mydata][mytask]
            lblTRN, lblTST = self.labels[mymodel][mydata][mytask]
            # Truncate for speed up if specified
            if self.limit is not None:
                nsamp, nfeats = self.limit[0], self.limit[1]
                embTRN = embTRN[:nsamp, :nfeats] 
                embTST = embTST[:nsamp, :nfeats]
                lblTRN = lblTRN[:nsamp]
                lblTST = lblTST[:nsamp]
            return embTRN, embTST, lblTRN, lblTST
        except:
            # Some elements may be empty, ex. pca only did task=ae
            return None, None, None, None


       


def drift(ld, mymetric):
    # Run drift metric over embeddings from all models/data/tasks
    niter = len(ld.models)+len(ld.data)+len(ld.tasks)
    pbar = tqdm(total=niter, desc=f'DRIFT[{mymetric}]')
    savefp = os.path.join(ld.rootdir, f'analysis_{mymetric}.pkl')
    for m in ld.models:
        for d in ld.data:
            for t in ld.tasks:
                # Just get the embeddings, don't need the labels
                embTRN, embTST, _, _ = ld.get(m,d,t)
                
                # Some elements may be empty, ex. pca only did task=ae
                if embTRN is None:
                    pbar.update(1)
                    continue

                # Get the method-dependent 'score' - the meaning of which varies
                if mymetric.lower() == 'mvdc':
                    dc = MVDC(embTRN, embTST, n_folds=5, chunk_sz=128, threshold=(0.4,0.6))
                    # TEMP, for speed up
                    # dc = MVDC(embTRN, embTST, n_folds=3, chunk_sz=10, threshold=(0.4,0.6))
                    dc.fit()
                    dc.predict()
                    # Store metrics, classification label (drift/no drift) over all chunks in the TEST set
                    tstdf = dc.resdf[dc.resdf['chunk']['period']=='analysis']
                    score = tstdf['domain_classifier_auroc']['value'].values
                elif mymetric.lower() =='mmd':
                    dc = DriftMMD(embTRN)
                    pval, score, thr = dc.score(embTST)
                elif mymetric.lower() == 'cvm':
                    dc = DriftCVM(embTRN)
                    pval, score = dc.score(embTST)
                elif mymetric.lower() == 'ks':
                    dc = DriftKS(embTRN)
                    pval, score = dc.score(embTST)

                # Get the drift classification 
                if mymetric.lower() == 'mvdc':
                    classification = tstdf['domain_classifier_auroc']['alert'].values
                else:
                    classification = dc.predict(embTST).is_drift
                    
                # The output format varies atm, so unify it to be all np.arrays
                if type(score) != np.array:
                    score = np.atleast_1d(np.array(score))
                if type(classification) != np.array:
                    classification = np.atleast_1d(np.array(classification))

                # Store metrics, classification label (drift/no drift)
                # NOTE:
                # - ths size of both may vary depending on the method, just needs to be consistent within the method
                ld.metrics[m][d][t] = score
                ld.labels[m][d][t]  = classification
                
                pbar.update(1)
                
                # Save each iteration
                with open(savefp, 'wb') as f:
                    pickle.dump(ld, f)
                
    pbar.close()
    return ld
    

def outlier(ld, mymetric):
    # Run outlier metric over all models/data/tasks
    niter = len(ld.models)+len(ld.data)+len(ld.tasks)
    pbar = tqdm(total=niter, desc=f'OUTLIER[{mymetric}]')
    savefp = os.path.join(ld.rootdir, f'analysis_{mymetric}.pkl')
    for m in ld.models:
        for d in ld.data:
            for t in ld.tasks:
                embTRN, embTST, _, _ = ld.get(m,d,t)
                
                # Some elements may be empty, ex. pca only did task=ae
                if embTRN is None:
                    pbar.update(1)
                    continue

                # Outlier identification in the TRAIN set
                if mymetric.lower() == 'clusterer':
                    # TODO - Ryan working on updated clusterer
                    # cluster = Clusterer(embTRN)
                    # outputs = cluster.evaluate()
                    # ld.outliers[m][d][t] = outputs.outliers
                    pbar.update(1)
                    continue
                elif mymetric.lower() == 'coverage':
                    res = coverage(embTRN)
                    ld.metrics[m][d][t] = res.indices
                    ld.outliers[m][d][t]  = res.critical_value

                pbar.update(1)
                
                # Save each iteration
                with open(savefp, 'wb') as f:
                    pickle.dump(ld, f)
                
    pbar.close()
    return ld


def estimator(ld, mymetric):
    # Run estimator metric over all models/data/tasks
    niter = len(ld.models)+len(ld.data)+len(ld.tasks)
    pbar = tqdm(total=niter, desc=f'ESTIMATOR[{mymetric}]')
    savefp = os.path.join(ld.rootdir, f'analysis_{mymetric}.pkl')
    for m in ld.models:
        for d in ld.data:
            for t in ld.tasks:
                embTRN, embTST, lblTRN, lblTST = ld.get(m,d,t)
                
                # Some elements may be empty, ex. pca only did task=ae
                if embTRN is None or embTST is None:
                    pbar.update(1)
                    continue

                if mymetric.lower() == 'ber':
                    # Estimate performance metrics on the TEST test
                    res = ber(embTST, lblTST, method="MST")
                    ld.metrics[m][d][t] = res.ber
                elif mymetric.lower() == 'divergence':
                    # Estimate distance between the TRN/TST sets
                    res = divergence(embTRN, embTST)
                    ld.metrics[m][d][t] = res.divergence                    

                pbar.update(1)
                
                # Save each iteration
                with open(savefp, 'wb') as f:
                    pickle.dump(ld, f)
    pbar.close()
    return ld 


def compute_metrics(ld, mymetric=None):
    # Currently enabled metrics (matches argparse options)
    drift_metrics = ['cvm', 'ks', 'mmd', 'mvdc']
    outlier_metrics = ['clusterer','coverage']
    estimator_metrics = ['ber', 'divergence']
    
    # All metrics are here, but for run time + parallelization, just do one at a time
    if mymetric is not None:
        if np.isin(mymetric, drift_metrics):
            ld = drift(ld, mymetric)
        elif np.isin(opt.metric, outlier_metrics):
            ld = outlier(ld, mymetric)
        elif np.isin(opt.metric, estimator_metrics):
            ld = estimator(ld, mymetric)
        else:
            warnings.warn(f'Cannot find {opt.metric} within dataeval metrics.')
    else:
        all_metrics = drift_metrics + outlier_metrics + estimator_metrics
        for mymetric in all_metrics:
            if np.isin(mymetric, drift_metrics):
                ld = drift(ld, mymetric)
            elif np.isin(opt.metric, outlier_metrics):
                ld = outlier(ld, mymetric)
            elif np.isin(opt.metric, estimator_metrics):
                ld = estimator(ld, mymetric)
            else:
                warnings.warn(f'Cannot find {opt.metric} within dataeval metrics.')


def read_results(fp):
    results = {}
    for fn in os.listdir(fp):
        ffn = os.path.join(fp, fn)
        mymetric = ffn.split('analysis_')[-1].split('.pkl')[0]
        with open(ffn, 'rb') as myfile:
            results[mymetric] = pickle.load(myfile)
    return results


def compile_df(ld):
    rows = []
    doswarm = False
    for m in ld.models:
        for d in ld.data:
            for t in ld.tasks:
                vals = ld.metrics[m][d][t]
                if vals is not None: 
                    if type(vals)!=np.ndarray or len(vals)==1:
                        vals = np.atleast_1d(np.array(vals))
                        doswarm = True
                    for v in vals:
                        rows.append({
                            'Model': m,
                            'Data': d,
                            'Task': t,
                            'Value': v
                        })

    df = pd.DataFrame(rows)
    
    # Create a catplot 
    if doswarm:
        g = sns.catplot(x="Task", y="Value", 
                        hue="Data", col="Model", 
                        data=df, 
                        kind='swarm',
                        size=8,
                        dodge=True,
                        height=4, aspect=0.7) 
    else:
        g = sns.catplot(x="Task", y="Value", 
                        hue="Data", col="Model", 
                        data=df, 
                        kind="box", 
                        height=4, aspect=0.7) 
            
    g.set_titles("{col_name}") 
    g.set_axis_labels("Task", "Values")
    plt.suptitle(metric)
    plt.show()

    return df


if __name__ == "__main__":
    
    opt = get_input_args()
    
    # Read prior results
    if len(os.listdir(opt.resdir)) != 0:
        results = read_results(opt.resdir)       
        for metric in list(results.keys()):
            df = compile_df(results[metric])
    
    # Compute metrics
    else:
        runlegend = load_legend(opt.resdir)
        
        # Loader class is needed for each metric
        ld = Loader(opt.resdir, runlegend)
        
        # Run the metric and save the analysis results as a .pkl file
        compute_metrics(ld, opt.metric)
                
    
    print('--COMPLETE--')

 
    
    
