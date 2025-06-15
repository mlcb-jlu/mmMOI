config = {
    'BRCA': {
        'num_class': 4,
        'num_edge': 4,
        'lr': 0.0008,
        'split': ['1_3', '2_6', '5_7', '7_7', '9_9'],
        'dropout': 0.1,
        'nlayer':3,
        'n_hidden':20,
        'n_head':4,
        'nmodal':3
    },
    'GBM': {
        'num_class': 4,
        'num_edge': 2,
        'lr': 0.0008,
        'split': ['0_0', '1_3', '2_6', '3_7', '4_9'],
        'dropout': 0.1,
        'nlayer': 3,
        'n_hidden': 20,
        'n_head': 4,
        'nmodal': 3
    },
    'KIPAN': {
        'num_class': 3,
        'num_edge': 3,
        'lr': 0.0005,
        'split': ['0_0', '1_3', '2_6', '3_7', '4_9'],
        'dropout': 0.1,
        'nlayer': 3,
        'n_hidden': 20,
        'n_head': 4,
        'nmodal': 3
    },
    'OV': {
        'num_class': 4,
        'num_edge': 2,
        'lr': 0.0003,
        'split': ['0_0', '1_3', '2_6', '3_7', '4_9'],
        'dropout': 0.1,
        'nlayer': 3,
        'n_hidden': 20,
        'n_head': 4,
        'nmodal': 3
    }
}
