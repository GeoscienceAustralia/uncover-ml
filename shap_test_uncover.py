from uncoverml.scripts import uncoverml as uncli

filepath = 'configs/reference_xgboost.yaml'
partitions = 4
uncli.learn(filepath, [], partitions)

