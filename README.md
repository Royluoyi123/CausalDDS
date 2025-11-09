# CausalDDS

This repository is the official implementation of CausalDDS

Disentangling Causal Substructures for Interpretable and Generalizable Drug Synergy Prediction


## How to run the code
### Create a directory under the ddsdata directory, example:
DDS/ddsdata/DrugCombDB640/raw/classification10/warm_start/

DDS/ddsdata/DrugCombDB640/processed/classification10/warm_start/

DDS/ddsdata/ONEIL-COSMIC640/raw/regression/drug_pairs/

DDS/ddsdata/ONEIL-COSMIC640/processed/regression/drug_pairs/
### split the raw data
```
python DDS/ddsdata/process_data.py
python DDS/scaffold_split.py
python DDS/simpd_split.py
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
