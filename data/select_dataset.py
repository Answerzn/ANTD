

'''
# --------------------------------------------
# select dataset
# --------------------------------------------
# Kai Zhang (github: https://github.com/cszn)
# --------------------------------------------
'''


def define_Dataset(dataset_opt):
    dataset_type = dataset_opt['dataset_type'].lower()
    # -----------------------------------------
    # denoising
    # -----------------------------------------
    if dataset_type in ['antd_blind_denoising']:
        from data.dataset_antd_blind_denoising import DatasetANTD as D

    elif dataset_type in ['antd_sidd']:
        from data.dataset_antd_sidd import DatasetANTD as D

    else:
        raise NotImplementedError('Dataset [{:s}] is not found.'.format(dataset_type))

    dataset = D(dataset_opt)
    print('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__, dataset_opt['name']))
    return dataset
