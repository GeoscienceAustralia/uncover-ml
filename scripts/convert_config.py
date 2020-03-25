"""
This script will convert old (<2.0.0) uncover-ml YAML configs to be 
compatible with the current versions (0.3.1 as of writing). The older
'pipeline' configs need to be rewritten in YAML.

To use, run this script and provide it with a config file to convert.
The result will be placed in the same directory as the provided config,
with 'converted_' prefixed to the filename:

.. example:: 

    python convert_config.py /path/to/old_config.yaml
"""
import yaml
import sys
import os

def convert(path):
    print("Converting UncoverML config...")
    with open(path) as f:
        old_config = yaml.load(f, Loader=yaml.FullLoader)

    if 'features' in old_config:
        for index, feat in enumerate(old_config['features']):
            if feat['type'] == 'pickle':
                old_config['pickling'] = {}
                old_config['pickling']['covariates'] = feat['files'].get('covariates')
                old_config['pickling']['targets'] = feat['files'].get('targets')
                old_config['pickling']['featurevec'] = feat['files'].get('featurevec')
            del old_config['features'][index]
            break

    if 'validation' in old_config:
        if 'parallel' in old_config['validation']:
            parallel = True
            del old_config['validation'][old_config['validation'].index('parallel')]
        else:
            parallel = False 
        temp = old_config['validation']
        old_config['validation'] = {}
        for i, e in enumerate(temp):
            if isinstance(e, dict) and 'k-fold' in e:
                old_config['validation'].update(e)
                old_config['validation']['k-fold']['parallel'] = parallel
            else:
                old_config['validation'][e] = True
            break

    if 'output' in old_config:
        old_config['output']['plot_feature_ranks'] = True
        old_config['output']['plot_intersection'] = True
        old_config['output']['plot_real_vs_pred'] = True
        old_config['output']['plot_correlation'] = True
        old_config['output']['plot_target_scaling'] = True

    base, name = os.path.split(path)
    new_path = os.path.join(base, 'converted_' + name)
    print(f"Done! Saving as '{new_path}'")
    with open(new_path, 'w') as f:
        yaml.dump(old_config, f)

if __name__ == '__main__':
    convert(sys.argv[1])
    
                
        
