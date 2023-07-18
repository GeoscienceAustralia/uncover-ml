import yaml

from pathlib import Path
from sklearn.feature_selection import RFE

from scripts.uncoverml import _load_data
from uncoverml import config, models


# Covariate location and processing information
base_covariates_path = Path('/g/data/jl14/80m_covarite_thumblnails')
themes_path_dict = {
    'climate': 'climate',
    'gamma': 'geophysics/gamma',
    'gravity': 'geophysics/gravity',
    'magnetics': 'geophysics/magnetics',
    'proximity': 'proximity',
    'terrain': 'terrain',
    'cover': 'satellite/cover',
    'fractional_cover': 'satellite/fractional_cover',
    'mineral': 'satellite/mineral',
    'aster': 'satellite/aster',
    'maps': 'maps'
}
exclude_file_names = {
    'maps': ['mask_80m_albers_cogs.tif', 'mask_80m_albers_new_june_2023.tif']
}

# Model information
base_learning_dict = {
    'algorithm': 'transformedrandomforest',
    'arguments': {'n_estimators': 10,
                  'target_transform': 'identity'}
}
base_features_dict = {
    'name': 'my_features',
    'type': 'ordinal',
    'files': [],
    'transforms': ['centre', 'standardise'],
    'imputation': 'mean'
}
base_targets_dict = {
    'file': None,
    'property': None
}


def generate_yaml(theme, target_file, target_label, out_dir):
    feature_config = base_features_dict.copy()
    covariate_list_file = Path(out_dir) / f'{theme}_features.txt'
    feature_config['files'] = [{'list': str(covariate_list_file)}]

    targets_config = base_targets_dict.copy()
    targets_config['file'] = target_file
    targets_config['property'] = target_label

    final_yaml_dict = {
        'learning': base_learning_dict,
        'features': [feature_config],
        'targets': targets_config,
        'output': {'directory': '/g/data/jl14/feature_selection'}
    }
    yaml_save_path = Path(out_dir) / f'{theme}.yaml'
    with open(yaml_save_path, 'w') as yaml_file:
        yaml.dump(final_yaml_dict, yaml_file, default_flow_style=False)

    return str(yaml_save_path)


def generate_covariate_list(theme, out_dir):
    theme_full_path = base_covariates_path / themes_path_dict[theme]
    exclude_files = None
    if theme in exclude_file_names:
        exclude_files = [str(theme_full_path / file_name) for file_name in exclude_file_names[theme]]

    file_list = []
    for file in theme_full_path.iterdir():
        string_path = str(file)
        if exclude_files is not None and string_path in exclude_files:
            continue
        else:
            file_list.append(string_path)

    out_file = Path(out_dir) / f'{theme}_features.txt'
    with open(out_file, 'w') as f:
        for file in file_list:
            f.write(f'{file}\n')


def feature_rank_recursive(theme, config_yaml, partitions=1, n_features=None, step=1):
    current_config = config.Config(config_yaml)
    # We'll start off assuming that partitioning of data is not needed
    targets_all, x_all = _load_data(current_config, partitions)
    current_model = models.modelmaps[current_config.algorithm](**current_config.algorithm_args)
    selector = RFE(current_model, n_features, step)
    selector.fit(x_all, targets_all.observations)
    # Just return for now, we'll save down the results later
    return selector.ranking_


if __name__ == '__main__':
    target_file_path = './GeoChem_test.shp'
    target_file_label = 'CaO_pct'
    out_dir = './theme_configs/'
    result_list = []
    for theme in list(themes_path_dict.keys()):
        print(theme)
        generate_covariate_list(theme, out_dir)
        current_yaml = generate_yaml(theme, target_file_path, target_file_label, out_dir)
        current_result = feature_rank_recursive(theme, current_yaml)
        result_list.append(current_result)

    [print(result) for result in result_list]
