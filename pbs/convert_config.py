import yaml
import sys

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

    if 'optimisation' in old_config:
        del old_config['optimisation']['algorithm']

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

    print(f"Done! Saving as 'converted_{path}'")
    with open('converted_' + path, 'w') as f:
        yaml.dump(old_config, f)

if __name__ == '__main__':
    convert(sys.argv[1])
    
                
        
