# InterpretableGCN

## Usage
### Setup
The whole implementation is built upon [PyTorch](https://pytorch.org) and [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/)

**conda**

See the `environment.yml` for environment configuration. 
```bash
conda env create -f environment.yml
```
**PYG**

To install pyg library, [please refer to the document](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

### Dataset 
**ADNI**

We download this dataset from [here](https://adni.loni.usc.edu/data-samples/access-data/).
We treat multi-modal imaging scans as a brain graph.

### How to run classification?
The BrainGNN framework is integrated in file `main_braingnn.py`. To run
```
python main_braingnn.py 
```
The Sparse Interpretable GNN framework is integrated in file `main_sgcn.py`. To run
```
python main_sgcn.py 
```
You can also specify the learning hyperparameters to run
```
python main_sgcn.py --epochs 200 --lr 0.0001 --search --cuda 0
```
`main_sgcn.py`: tunning hyperparameters

`kernel/train_eval_sgcn.py`: training framework for SGCN

`kernel/train_eval_braingnn.py`: training framework for BrainGNN

`kernel/sgcn.py`: training model for SGCN

`kernel/braingnn.py`: training framework for BrainGNN
