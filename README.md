# MoleOOD
Official implementation for our paper:

Learning Substructure Invariance for Out-of-Distribution Molecular Representations

Nianzu Yang, Kaipeng Zeng, Qitian Wu, Xiaosong Jia, Junchi Yan* (* denotes correspondence)

*Advances in Neural Information Processing Systems* (**NeurIPS 2022**)



## Folder Specification

- ```config/```: configurations for backbone (GCN, GIN, GraphSAGE)
- ```saved_model/```: three trained model on OGB-BACE datasets
- ```modules/```: preprocessing scripts for data and model definition
- ```baseline_ogb.py```: train the baselines on OGB benchmark
- ```main.py```: train or evaluate our model on OGB benchmark



## Package Dependency

```
torch: 1.9.0
numpy: 1.21.2
ogb: 1.3.4
rdkit: 2021.9.4
scikit-learn: 1.0.2
pyg: 2.0.3
```



## Run the Code

Train the baselines on OGB benchmark:

``` 
python baseline_ogb.py --dataset ogbg-molbace --gnn gcn --device ${device} --seed ${seed}
```



Before training our model, we should obtain the substructures from the raw data (here we use BRICS molecular segmentation method):

```
python modules/PreProcess.py --dataset ogbg-molbace --method ${decomposition_method}
```



Then, we can train our model, e.g.:

```
python main.py --base_backend ./config/GCN_base_dp0.1.json --sub_backend ./config/GIN_sub_dp0.1.json --domain_backend ./config/GIN_domain_dp0.1.json --conditional_backend ./config/GIN_cond_dp0.1.json  --dataset ogbg-molbace --lambda_loss ${lambda_loss} --device ${device} --lr ${lr} --num_domain ${num_domain} --epoch_main ${epoch_main} --epoch_ast ${epoch_ast} --batch_size ${batch_size} --drop_ratio ${drop_ratio} --seed ${seed} --decomp_method ${decomposition_method} --prior ${uniform/gaussian}
```

or evaluate our model, e.g.:

```
python main.py --base_backend ./config/GCN_base_dp0.1.json --sub_backend ./config/GIN_sub_dp0.1.json --domain_backend ./config/GIN_domain_dp0.1.json --conditional_backend ./config/GIN_cond_dp0.1.json  --dataset ogbg-molbace --lambda_loss ${lambda_loss} --device ${device} --lr ${lr} --num_domain ${num_domain} --epoch_main ${epoch_main} --epoch_ast ${epoch_ast} --batch_size ${batch_size} --drop_ratio ${drop_ratio} --seed ${seed} --test --model_path ./saved_model/GCN.pth --decomp_method ${decomposition_method} --prior ${uniform/gaussian}
```



## Dataset

We use four datasets from [OGB](https://ogb.stanford.edu/) benchmark and six datasets from [DrugOOD](https://github.com/tencent-ailab/DrugOOD) benchmark.

**OGB**: BACE, BBBP, SIDER, HIV

**DrugOOD**: IC50/EC50-size/scaffold/assay



## Citation

```bibtex
@inproceedings{yang2022learning,
  title={Learning Substructure Invariance for Out-of-Distribution Molecular Representations},
  author={Yang, Nianzu and Zeng, Kaipeng and Wu, Qitian and Jia, Xiaosong and Yan, Junchi},
  booktitle={Advances in neural information processing systems},
  year={2022},
}
```

**The codes for DrugOOD benchmark will be released soon.**


Welcome to contact us [yangnianzu@sjtu.edu.cn](mailto:yangnianzu@sjtu.edu.cn) or [zengkaipeng@sjtu.edu.cn](mailto:zengkaipeng@sjtu.edu.cn) for any question.
