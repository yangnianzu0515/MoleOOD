<h1 align="center"><b>MoleOOD</b></h1>
<p align="center">
    <a href="https://openreview.net/forum?id=2nWUNTnFijm"> <img alt="License" src="https://img.shields.io/static/v1?label=Pub&message=NeurIPS%2722&color=blue"></a>
    <a href="https://github.com/yangnianzu0515/MoleOOD/blob/master/LICENSE"> <img alt="License" src="https://img.shields.io/github/license/yangnianzu0515/MoleOOD?color=green"></a>
    <a href="https://yangnianzu0515.github.io/slides/paper4-slides-moleood.pdf"> <img src="https://img.shields.io/badge/Slides-grey?&logo=MicrosoftPowerPoint&logoColor=red" alt="Slides"></a>
    <a href="https://github.com/yangnianzu0515/MoleOOD/stargazers"><img src="https://img.shields.io/github/stars/yangnianzu0515/MoleOOD?color=yellow&label=Star" alt="Stars"></a>
</p>

Official implementation for our paper:

Learning Substructure Invariance for Out-of-Distribution Molecular Representations

Nianzu Yang, Kaipeng Zeng, Qitian Wu, Xiaosong Jia, Junchi Yan* (* denotes correspondence)

*Advances in Neural Information Processing Systems* (**NeurIPS 2022**)


## Dataset

We use four datasets from [OGB](https://ogb.stanford.edu/) benchmark and six datasets from [DrugOOD](https://github.com/tencent-ailab/DrugOOD) benchmark.

**OGB**: BACE, BBBP, SIDER, HIV

**DrugOOD**: IC50/EC50-size/scaffold/assay



## Codes for OGB Dataset

### Folder Specification

- ```config/```: configurations for backbone (GCN, GIN, GraphSAGE)
- ```saved_model/```: three trained model on OGB-BACE datasets
- ```modules/```: preprocessing scripts for data and model definition
- ```baseline_ogb.py```: train the baselines on OGB benchmark
- ```main.py```: train or evaluate our model on OGB benchmark

### Package Dependency

```
torch: 1.9.0
numpy: 1.21.2
ogb: 1.3.4
rdkit: 2021.9.4
scikit-learn: 1.0.2
pyg: 2.0.3
```

### Run the Code

Train the baselines on OGB benchmark:

``` 
python baseline_ogb.py --dataset ogbg-molbace --gnn gcn --device ${device} --seed ${seed}
```

Before training our model, we should obtain the substructures from the raw data (here we use BRICS molecular segmentation method as default):

```
python modules/PreProcess.py --dataset ogbg-molbace --method ${decomposition_method}
```

The preprocess results are already uploaded to the folder ```OGB/preprocess/```. 

Then, we can train our model, e.g.:

```
python main.py --base_backend ./config/GCN_base_dp0.1.json --sub_backend ./config/GIN_sub_dp0.1.json --domain_backend ./config/GIN_domain_dp0.1.json --conditional_backend ./config/GIN_cond_dp0.1.json  --dataset ogbg-molbace --lambda_loss ${lambda_loss} --device ${device} --lr ${lr} --num_domain ${num_domain} --epoch_main ${epoch to train main model} --epoch_ast ${epoch to train env inference model} --batch_size ${batch_size} --drop_ratio ${drop_ratio} --seed ${seed} --decomp_method ${decomposition_method} --prior ${uniform/gaussian}
```

or evaluate our model using following commands:

**BACE+GCN:**

```
python evaluate.py --base_backend ./config/GCN_base_dp0.1.json --sub_backend ./config/GIN_sub_dp0.1.json   --dataset ogbg-molbace  --model_path ./saved_model/GCN.pth --decomp_method brics --drop_ratio 0.1 --device ${device} 
```

**BACE+GIN:**

```
python evaluate.py --base_backend ./config/GIN_base_dp0.1.json --sub_backend ./config/GIN_sub_dp0.1.json   --dataset ogbg-molbace  --model_path ./saved_model/GIN.pth --decomp_method brics --drop_ratio 0.1 --device ${device} 
```

**BACE+SAGE:**

```
python evaluate.py --base_backend ./config/SAGE_base_dp0.1.json --sub_backend ./config/GIN_sub_dp0.1.json   --dataset ogbg-molbace  --model_path ./saved_model/SAGE.pth --decomp_method brics --drop_ratio 0.1 --device ${device} 
```



## Codes for DrugOOD Dataset

### Folder Specification

- ```data/```: containing the data for training, including the preprocess result and substructure information merging scripts.
- ```config/```: configuration for model and model training.
- ```main.py```: the script to train our algorithm.
- ```models/```: containing the loss definition, backbone definition for our method.
- ```saved_modesl/```: six trained model on DrugOOD datasets

### Package Dependency

```
torch: 1.11
pyg: 2.0.3
drugood: 0.0.1
rdkit: 2022.3.1
numpy: 1.12.2
```

To install package ```drugood```, please refer to [DrugOOD](https://github.com/tencent-ailab/DrugOOD)  repository. 

### Data Generation

- The first step is to generate the original dataset from CHEMBL database. As for the detailed process or operation, please refer to the  [DrugOOD](https://github.com/tencent-ailab/DrugOOD)  repository. The generated ```json```  files should be put into folder ```DrugOOD/data/ic50``` or ```DrugOOD/data/ec50``` respectively.

- The second step is to generate the substructure information for each molecule and merge it into original dataset. two operations should be run in this step as follows:

  ```
  python PreProcess.py --start ${start_index} --num ${num of molecule to process} --dataset ${ec50/ic50} --method ${decomposition method} --timeout ${maximum time to process a single molecule}
  ```

  After all the substructures are generated, change the working directory into ```DrugOOD/data/ic50``` or ```DrugOOD/data/ec50``` and run the script

  ```
  python merge_data.py
  ```

- All the processed results are uploaded to the folder ```DrugOOD/data/```. 

### Run the Code

To train and evaluate the baseline on DrugOOD dataset, please refer to  [DrugOOD](https://github.com/tencent-ailab/DrugOOD)  repository. 

Our model can be trained like:

```
python main.py --data_config configs/data_assay_ec50.py --model_config configs/GIN_0.5_mean.py --lambda_loss ${lambda loss} --lr ${lr} --num_domain ${num domain} --seed ${seed} --epoch_ast ${epoch to train env inference model} --epoch_main ${epoch to train main model} --dist ${gaussian/uniform} --device ${device}
```

Also the well-trained models can be evaluated by:

**ic50 assay:**

```
python evaluate.py --data_config configs/data_assay_ic50.py --model_config configs/GIN_0.5_mean.py --model_path saved_models/ic50_assay.pth --device ${device}
```

**ic50 scaffold:**

```
python evaluate.py --data_config configs/data_scaffold_ic50.py --model_config configs/GIN_0.5_mean.py --model_path saved_models/ic50_scaffold.pth --device ${device}
```

**ic50 size:**

```
python evaluate.py --data_config configs/data_size_ic50.py --model_config configs/GIN_0.5_mean.py --model_path saved_models/ic50_size.pth --device ${device}
```

**ec50 assay:**

```
python evaluate.py --data_config configs/data_assay_ec50.py --model_config configs/GIN_0.5_mean.py --model_path saved_models/ec50_assay.pth --device ${device}
```

**ec50 scaffold:**

```
python evaluate.py --data_config configs/data_scaffold_ec50.py --model_config configs/GIN_0.5_mean.py --model_path saved_models/ec50_scaffold.pth --device ${device}
```

**ec50 size:**

```
python evaluate.py --data_config configs/data_size_ec50.py --model_config configs/GIN_0.5_mean.py --model_path saved_models/ec50_size.pth --device ${device}
```



## Citation

```bibtex
@inproceedings{yang2022learning,
  title={Learning Substructure Invariance for Out-of-Distribution Molecular Representations},
  author={Nianzu Yang and Kaipeng Zeng and Qitian Wu and Xiaosong Jia and Junchi Yan},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2022},
}
```


Welcome to contact us [yangnianzu@sjtu.edu.cn](mailto:yangnianzu@sjtu.edu.cn) or [zengkaipeng@sjtu.edu.cn](mailto:zengkaipeng@sjtu.edu.cn) for any question.
