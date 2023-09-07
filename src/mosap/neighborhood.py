import networkx as nx
import numpy as np
import pandas as pd

def get_node_interactions(g: nx.Graph, data: pd.Series = None):
    source, neighs = [], []
    for i in g.nodes:
        if len(g[i]) > 0:  # some nodes might have no neighbors
            source.append(i)
            neighs.append(list(g[i]))
    # convert neighbourhood network to pandas dataframe
    node_interactions = pd.DataFrame({'source': source, 'target': neighs}).explode('target')
    if data is not None:
        # iterate through the netwok and get the count of edges connected to source node and label node
        node_interactions['source_ct'] = data.loc[node_interactions.source].values
        node_interactions['target_ct'] = data.loc[node_interactions.target].values

    return node_interactions


def get_interaction_score(interactions, relative_freq=False, observed=False):
    # function to calculate the interaction score by counting the edges and grouped cell type
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