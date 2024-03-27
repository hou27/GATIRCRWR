# GAT with Initial Residual Connection and Random Walk with Restart

This is the implementation of the GAT with Initial Residual Connection and Random Walk with Restart (GAT-IRC-RWR) model. 

## Datasets
The datasets used in the experiments are available at the following links:
- [Cora](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Planetoid.html#torch_geometric.datasets.Planetoid)
- [Citeseer](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Planetoid.html#torch_geometric.datasets.Planetoid)
- [PubMed](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Planetoid.html#torch_geometric.datasets.Planetoid)

## Code References
- [GAT](https://github.com/PetarV-/GAT)
- [pytorch-GAT](https://github.com/gordicaleksa/pytorch-GAT)

## Experiment Results

### Cora Dataset:

| Model      | 2 Layers | 3 Layers | 4 Layers | 5 Layers | 6 Layers | 7 Layers | 8 Layers |
|------------|----------|----------|----------|----------|----------|----------|----------|
| GAT        | 0.786    | 0.802    | 0.800    | 0.796    | 0.767    | 0.627    | 0.354    |
| GATRWR     | 0.797    | 0.804    | 0.811    | 0.807    | 0.813    | 0.811    | 0.811    |
| GATIRC     | 0.757    | 0.810    | 0.801    | 0.798    | 0.412    | 0.381    | 0.466    |
| GATIRCRWR  | 0.747    | 0.802    | 0.808    | 0.803    | 0.807    | 0.809    | 0.809    |

### Citeseer Dataset:

| Model      | 2 Layers | 3 Layers | 4 Layers | 5 Layers | 6 Layers | 7 Layers | 8 Layers |
|------------|----------|----------|----------|----------|----------|----------|----------|
| GAT        | 0.688    | 0.684    | 0.650    | 0.624    | 0.629    | 0.554    | 0.371    |
| GATRWR     | 0.667    | 0.703    | 0.700    | 0.696    | 0.698    | 0.699    | 0.703    |
| GATIRC     | 0.687    | 0.688    | 0.681    | 0.612    | 0.378    | 0.422    | 0.380    |
| GATIRCRWR  | 0.680    | 0.704    | 0.706    | 0.702    | 0.711    | 0.711    | 0.707    |

### Pubmed Dataset:

| Model      | 2 Layers | 3 Layers | 4 Layers | 5 Layers | 6 Layers | 7 Layers | 8 Layers |
|------------|----------|----------|----------|----------|----------|----------|----------|
| GAT        | 0.781    | 0.781    | 0.778    | 0.780    | 0.774    | 0.771    | 0.768    |
| GATRWR     | 0.772    | 0.781    | 0.778    | 0.781    | 0.784    | 0.783    | 0.785    |
| GATIRC     | 0.786    | 0.769    | 0.780    | 0.772    | 0.744    | 0.719    | 0.685    |
| GATIRCRWR  | 0.779    | 0.769    | 0.774    | 0.777    | 0.781    | 0.783    | 0.777    |