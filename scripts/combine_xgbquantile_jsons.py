import json

all_xgb_params = {
    'mean_model_params' : 'ref_xgb/reference_xgboost_scores.json',
    'upper_quantile_params':  'ref_xgb/reference_xgboost_oss_validation_scores.json',
    'lower_quantile_params': 'ref_xgb/reference_xgboost_optimised_params.json'
}

d = {}
for k, v in all_xgb_params.items():
    with open(v, 'r') as f:
        d[k] = json.load(f)
        if k == 'upper_quantile_params':
            d[k]['alpha'] = 0.95
        if k == 'lower_quantile_params':
            d[k]['alpha'] = 0.05

with open("xgbquantile_combined_params.json", "w") as outfile:
    json.dump(d, outfile)
