from .gcn import GCN, GraphConvolution, GCN_Reg
import warnings
try:
    from .graphsage import GraphSAGE
    from .gat import GAT

except ImportError as e:
    print(e)
    warnings.warn("Please install pytorch geometric if you " +
                  "would like to use the datasets from pytorch " +
                  "geometric. See details in https://pytorch-geom" +
                  "etric.readthedocs.io/en/latest/notes/installation.html")

__all__ = ['GCN', 'GraphConvolution', 'GAT', 'GraphSAGE']
