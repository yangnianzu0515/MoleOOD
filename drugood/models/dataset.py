from drugood.datasets import DATASETS, DrugOODDataset
__all__ = ['LBAPDatasetWithSub', 'LBAPDatasetWithChem']


@DATASETS.register_module()
class LBAPDatasetWithSub(DrugOODDataset):
    def __init__(self, **kwargs):
        super(LBAPDatasetWithSub, self).__init__(**kwargs)

    def prepare_data(self, idx):
        case = self.data_infos[idx]
        input = case["smiles"]
        results = {
            'input': input,
            'gt_label': int(case[self.label_key]),
            'group': case['domain_id'],
            'subs': case['substructure']
        }
        return self.pipeline(results)

@DATASETS.register_module()
class LBAPDatasetWithChem(DrugOODDataset):
    def __init__(self, **kwargs):
        super(LBAPDatasetWithChem, self).__init__(**kwargs)

    def prepare_data(self, idx):
        case = self.data_infos[idx]
        input = case['smiles']
        results = {
            'input': input, 'smiles': input,
            'gt_label': int(case[self.label_key]),
            'group': case['domain_id'],
            'subs': case['substructure']
        }
        return self.pipeline(results)