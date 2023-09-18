import os
import networkx as nx
import numpy as np
import pandas as pd
import time
from mosadata import MOSADATA
from networkx import Graph
from pandas.api.types import CategoricalDtype
import logging

logging.basicConfig(level=logging.INFO)

class Interactions:
    """
    Estimator to quantify interaction strength between different species in the sample.
    """
    VALID_PREDICTION_TYPES = ['pvalue', 'observation', 'diff']

    def __init__(self, mosadata:MOSADATA, sample: str, attr: str = 'meta_id', mode: str = 'classic', 
                 n_permutations: int = 500,
                 random_seed=None, alpha: float = .01, graph_key: str = 'knn'):
        """Estimator to quantify interaction strength between different species in the sample.
        """

        self.data = mosadata
        self.spl: str = sample
        self.graph_key = graph_key
        self.g: Graph = mosadata.G[sample][graph_key]
        self.attr: str = attr
        self.data: pd.Series = mosadata.obs[sample][attr]
        self.mode: str = mode
        self.n_perm: int = int(n_permutations)
        self.random_seed = random_seed if random_seed else mosadata.random_seed
        self.rng = np.random.default_rng(random_seed)
        self.alpha: float = alpha
        self.fitted: bool = False

        # set dtype categories of data to attributes that are in the data
        self.data = self.data.astype(CategoricalDtype(categories=self.data.unique(), ordered=False))


    def fit(self, prediction_type:str, try_load: bool = True) -> None:
        if prediction_type not in self.VALID_PREDICTION_TYPES:
            raise ValueError(
                f'invalid `prediction_type` {prediction_type}. Available modes are {self.VALID_PREDICTION_TYPES}')

        self.prediction_type = prediction_type
        self.mode == 'proportion'
        relative_freq, observed = True, False
        node_interactions = get_node_interactions(self.g, self.data)
        obs_interaction = get_interaction_score(node_interactions, relative_freq=relative_freq, observed=observed)
        self.obs_interaction = obs_interaction.set_index(['source_ct', 'target_ct'])

        if not self.prediction_type == 'observation':
            if try_load:
                if os.path.isdir(self.path) and self.h0_file in os.listdir(self.path):
                    logging.info(
                        f'loading h0 for {self.spl}, graph type {self.graph_key} and mode {self.mode}')
                    self.h0 = pd.read_pickle(os.path.join(self.path, self.h0_file))
            # if try_load was not successful
            if self.h0 is None:
                logging.info(
                    f'generate h0 for {self.spl}, graph type {self.graph_key} and mode {self.mode} and attribute {self.attr}')
                self.generate_h0(relative_freq=relative_freq, observed=observed, save=True)

        self.fitted = True

    def predict(self) -> pd.DataFrame:
        """Predict interactions strengths of observations.

        Returns: A dataframe with the interaction results.

        """
        if self.prediction_type == 'observation':
            return self.obs_interaction
        elif self.prediction_type == 'pvalue':
            # TODO: Check p-value computation
            data_perm = pd.concat((self.obs_interaction, self.h0), axis=1)
            data_perm.fillna(0, inplace=True)
            data_pval = pd.DataFrame(index=data_perm.index)

            # see h0_models_analysis.py for alterantive p-value computation
            data_pval['score'] = self.obs_interaction.score
            data_pval['perm_mean'] = data_perm.apply(lambda x: np.mean(x[1:]), axis=1, raw=True)
            data_pval['perm_std'] = data_perm.apply(lambda x: np.std(x[1:]), axis=1, raw=True)
            data_pval['perm_median'] = data_perm.apply(lambda x: np.median(x[1:]), axis=1, raw=True)

            data_pval['p_gt'] = data_perm.apply(lambda x: np.sum(x[1:] >= x[0]) / self.n_perm, axis=1, raw=True)
            data_pval['p_lt'] = data_perm.apply(lambda x: np.sum(x[1:] <= x[0]) / self.n_perm, axis=1, raw=True)
            data_pval['perm_n'] = data_perm.apply(lambda x: self.n_perm, axis=1, raw=True)

            data_pval['p'] = data_pval.apply(lambda x: x.p_gt if x.p_gt <= x.p_lt else x.p_lt, axis=1)
            data_pval['sig'] = data_pval.apply(lambda x: x.p < self.alpha, axis=1)
            data_pval['attraction'] = data_pval.apply(lambda x: x.p_gt <= x.p_lt, axis=1)
            data_pval['sigval'] = data_pval.apply(lambda x: np.sign((x.attraction - .5) * x.sig), axis=1)
            return data_pval
        elif self.prediction_type == 'diff':
            data_perm = pd.concat((self.obs_interaction, self.h0), axis=1)
            data_perm.fillna(0, inplace=True)
            data_pval = pd.DataFrame(index=data_perm.index)

            # see h0_models_analysis.py for alterantive p-value computation
            data_pval['score'] = self.obs_interaction.score
            data_pval['perm_mean'] = data_perm.apply(lambda x: np.mean(x[1:]), axis=1, raw=True)
            data_pval['perm_std'] = data_perm.apply(lambda x: np.std(x[1:]), axis=1, raw=True)
            data_pval['perm_median'] = data_perm.apply(lambda x: np.median(x[1:]), axis=1, raw=True)

            data_pval['diff'] = (data_pval['score'] - data_pval['perm_mean'])
            return data_pval

        else:
            raise ValueError(
                f'invalid `prediction_type` {self.prediction_type}. Available modes are {self.VALID_PREDICTION_TYPES}')

    def generate_h0(self, relative_freq, observed, save=True):
        # usually get called if the paramater is prediction_type=observation
        # This function work similar to squidpy permutation but improved with the source cell type and target cell type 
        connectivity = get_node_interactions(self.g).reset_index(drop=True)

        res_perm, durations = [], []
        for i in range(self.n_perm):
            tic = time.time()

            data = permute_labels(self.data, self.rng)
            source_ct = data.loc[connectivity.source].values.ravel()
            target_ct = data.loc[connectivity.target].values.ravel()

            # create pd.Series and node_interaction pd.DataFrame
            source_ct = pd.Series(source_ct, name='source_ct', dtype=self.data.dtype)
            target_ct = pd.Series(target_ct, name='target_ct', dtype=self.data.dtype)
            df = pd.concat((connectivity, source_ct, target_ct), axis=1)

            # get interaction count
            perm = get_interaction_score(df, relative_freq=relative_freq, observed=observed)
            perm['permutation_id'] = i

            # save result
            res_perm.append(perm)

            # stats
            toc = time.time()
            durations.append(toc - tic)

            if (i + 1) % 10 == 0:
                print(f'{time.asctime()}: {i + 1}/{self.n_perm}, duration: {np.mean(durations):.2f}) sec')
        print(
            f'{time.asctime()}: Finished, duration: {np.sum(durations) / 60:.2f} min ({np.mean(durations):.2f}sec/it)')

        h0 = pd.concat(res_perm)
        self.h0 = pd.pivot(h0, index=['source_ct', 'target_ct'], columns='permutation_id', values='score')


def permute_labels(data, rng: np.random.Generator):
    return pd.Series(rng.permutation(data), index=data.index)

def get_node_interactions(g: nx.Graph, data: pd.Series = None):
    ### count the interaction by unique cell type in the grph
    ###
    source, neighs = [], []
    for i in g.nodes:
        if len(g[i]) > 0:  # some nodes might have no neighbors
            source.append(i)
            neighs.append(list(g[i]))
    # convert neighbourhood network to pandas dataframe
    node_interactions = pd.DataFrame({'source': source, 'target': neighs}).explode('target')
    if data is not None:
        # iterate through the network and get the count of edges connected to source node and label node
        node_interactions['source_ct'] = data.loc[node_interactions.source].values
        node_interactions['target_ct'] = data.loc[node_interactions.target].values

    return node_interactions


def get_interaction_score(interactions, observed=False):
    # function to calculate the interaction score by counting the edges and grouped cell type
    # this function takes the paramater interactions (dataframe) as the input, should have 4 columns
    # source, target, source_ct (count of source ct), target_ct (count of target neighbour cell type)
    source_label = interactions[['source', 'source_ct']].drop_duplicates().set_index('source')
    source_label = source_label.squeeze()

    source2target_label = interactions.groupby(['source', 'target_ct'], observed=observed,
                                               as_index=False).size().rename({'size': 'counts'}, axis=1)
    source2target_label.loc[:, 'source_ct'] = source_label[source2target_label.source].values
    totals = source2target_label.groupby('source')['counts'].agg('sum')
    source2target_label['n_neigh'] = totals.loc[source2target_label.source].values
    source2target_label['freq'] = source2target_label['counts'] / source2target_label['n_neigh']
    res = source2target_label.groupby(['source_ct', 'target_ct'], observed=observed)
    res = res['freq'].agg('mean').rename('score').fillna(0).reset_index()
    
    return res