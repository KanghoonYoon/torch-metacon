# Debiased Graph Poisoning Attack via Contrastive Surrogate Objective
Pytorch Implmentation of Debiased Graph Poisoning Attack via Contrastive Surrogate Objective

We refer to the <ins>DeepRobust</ins> library to build our implementation **src/**. [https://github.com/DSE-MSU/DeepRobust](https://github.com/DSE-MSU/DeepRobust).

## Dataset
We use four datasets: Cora, Cora ML, Citeseer, and Polblogs in Datasets of DeepRobust library.
Note that these datasets are the largest connected component (LCC) of the graph datasets. 

Please make the directory for the dataset first.

``` python  

mkdir datasets

```  

## Implementation  

You can conduct the experiment of **Metacon_S** and **Metacon_D** by running the file in **shell/** directory.

* Train / Test of Metacon
``` python  

bash shell/test_metacons.sh  
bash shell/test_metacond.sh  

```  



