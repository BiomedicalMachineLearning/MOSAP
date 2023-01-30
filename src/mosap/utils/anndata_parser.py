#!/usr/bin/env python

import numpy as np
import pandas as pd
import anndata as ad
import argparse
import os
import sys
import numpy as np
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description='Create AnnData object from counts matrix and metadata')
    parser.add_argument("counts",
        help="cell x gene counts matrix in MatrixMarket format")
    parser.add_argument("metadata",
        help="cell metadata file in csv format")
    parser.add_argument("genes",
        help="gene names in csv format")
    parser.add_argument("--coords",
        help="spatial coords file in csv format",
        default=None)
    parser.add_argument("--umap",
        help="umap dims file in csv format",
        default=None)
    parser.add_argument("-o", "--outputdir",
        help="Where to write h5ad file",
        default=".")
    parser.add_argument("--filename",
        help="Name for h5ad file",
        default="adata.h5ad")
    parser.add_argument("-n", "--name",
        help="Name of anndata object",
        default="CosMx study")
    args = parser.parse_args()

    adata = ad.read_mtx(args.counts, dtype=np.uint16)
    adata.obs = pd.read_csv(args.metadata, index_col=0)
    adata.var = pd.read_csv(args.genes, index_col=0)
    if args.coords is not None:
        adata.obsm['spatial'] = pd.read_csv(args.coords).to_numpy()
    if args.umap is not None:
        adata.obsm['umap'] = pd.read_csv(args.umap).to_numpy()
    adata.uns['name'] = args.name
    adata.write(os.path.join(args.outputdir, args.filename), compression="gzip")

if __name__ == '__main__':
    sys.exit(main())