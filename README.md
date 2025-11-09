# CausalDDS

This repository is the official implementation of CausalDDS

Disentangling Causal Substructures for Interpretable and Generalizable Drug Synergy Prediction


## How to run the code
### split the raw data
```
python DDS/ddsdata/process_data.py 
```
### create the .pt file 
```
python DDS/crete_data/data.py
```

### run the main function
```
python DDS/main.py
```



## Requirements

Python version : 3.7.10
Pytorch version: 1.8.1
torch-geometric version: 1.7.0
