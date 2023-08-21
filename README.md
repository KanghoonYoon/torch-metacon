# torch-metacon
Pytorch Implmentation of Debiased Graph Poisoning Attack via Contrastive Surrogate Objective

We refer to the <ins>DeepRobust<\ins> library to build our implementation. [https://github.com/DSE-MSU/DeepRobust](https://github.com/DSE-MSU/DeepRobust).

## Dataset
We use four datasets: Cora, Cora ML, Citeseer, and Polblogs in Datasets of DeepRobust library.
Note that these datasets are the largest connected component (LCC) of the graph datasets.

## Implementation  

You can conduct the experiment of **Metacon** by running the file in **shell/** directory.

* Train / Test of Metacon
``` python  

bash shell/test_metacon.sh  

```  

To run the experiment of MetaAttack and GraD, 

* Train and Test of Meta-gradeint-based Attacks
``` python  

bash shell/test_meta.sh

bash shell/test_grad.sh

```  

