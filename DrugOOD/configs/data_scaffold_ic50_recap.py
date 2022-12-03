dataset_type = 'LBAPDatasetWithSub'
ann_file = 'data/ic50/lbap_core_ic50_scaffold_recap.json'

train_pipeline = [
    dict(
        type="SmileToGraph",
        keys=["input"]
    ),
    dict(
        type='Collect',
        keys=['input', 'gt_label', 'group', 'subs']
    )
]
test_pipeline = [
    dict(
        type="SmileToGraph",
        keys=["input"]
    ),
    dict(
        type='Collect',
        keys=['input', 'gt_label', 'group', 'subs']
    )
]


data = dict(
    samples_per_gpu=128,
    workers_per_gpu=4,
    train=dict(
        split="train",
        type=dataset_type,
        ann_file=ann_file,
        pipeline=train_pipeline
    ),
    ood_val=dict(
        split="ood_val",
        type=dataset_type,
        ann_file=ann_file,
        pipeline=test_pipeline
    ),
    ood_test=dict(
        split="ood_test",
        type=dataset_type,
        ann_file=ann_file,
        pipeline=test_pipeline,
    ),
    num_class=2
)
