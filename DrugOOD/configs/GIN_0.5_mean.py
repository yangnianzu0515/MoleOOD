model = dict(
    main=dict(
        type='MyGIN',
        num_node_emb_list=[39],
        num_edge_emb_list=[10],
        num_layers=4,
        emb_dim=128,
        readout='mean',
        JK='last',
        dropout=0.5,
    ),
    sub=dict(
        type='MyGIN',
        num_node_emb_list=[39],
        num_edge_emb_list=[10],
        num_layers=4,
        emb_dim=128,
        readout='mean',
        JK='last',
        dropout=0.5,
    ),
    conditional=dict(
        type='MyGIN',
        num_node_emb_list=[39],
        num_edge_emb_list=[10],
        num_layers=4,
        emb_dim=128,
        readout='mean',
        JK='last',
        dropout=0.5,
    ),
    domain=dict(
        type='MyGIN',
        num_node_emb_list=[39],
        num_edge_emb_list=[10],
        num_layers=4,
        emb_dim=128,
        readout='mean',
        JK='last',
        dropout=0.5,
    )
)
dropout=0.5
