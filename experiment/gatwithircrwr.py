"""
Reference: GRAPH ATTENTION NETWORKS (2018).

https://github.com/PetarV-/GAT
https://github.com/gordicaleksa/pytorch-GAT
"""

import torch.nn as nn

from layer import GATLayerWithIRCRWR

class GATWithIRCRWR(nn.Module):
    def __init__(self, num_of_additional_layer, num_in_features, num_classes, random_walk_with_restart=True, add_residual_connection=True, bias=True, dropout=0.6):
        super().__init__()

        additional_layers = []

        for _ in range(num_of_additional_layer):
            additional_layers.append(
                    GATLayerWithIRCRWR(
                    num_in_features=8*8,  # consequence of concatenation
                    num_out_features=8,
                    num_of_heads=8,
                    concat=True,
                    activation=nn.ELU(),
                    dropout_prob=dropout,
                    random_walk_with_restart=random_walk_with_restart,
                    add_residual_connection=add_residual_connection,
                    bias=bias
                ),
            )

        self.gat_net = nn.Sequential(
            GATLayerWithIRCRWR(
                num_in_features=num_in_features,  # consequence of concatenation
                num_out_features=8,
                num_of_heads=8,
                concat=True,
                activation=nn.ELU(),
                dropout_prob=dropout,
                random_walk_with_restart=random_walk_with_restart,
                add_residual_connection=add_residual_connection,
                bias=bias
            ),
            *additional_layers,
            GATLayerWithIRCRWR(
                num_in_features=8 * 8,  # consequence of concatenation
                num_out_features=num_classes,
                num_of_heads=1,
                concat=False,  # last GAT layer does mean avg, the others do concat
                activation=None,  # last layer just outputs raw scores
                dropout_prob=dropout,
                random_walk_with_restart=False,
                add_residual_connection=False,
                bias=bias
            )
        )

    # data is just a (in_nodes_features, edge_index) tuple, I had to do it like this because of the nn.Sequential:
    # https://discuss.pytorch.org/t/forward-takes-2-positional-arguments-but-3-were-given-for-nn-sqeuential-with-linear-layers/65698
    def forward(self, data):
        data = data + (None,)
        return self.gat_net(data)