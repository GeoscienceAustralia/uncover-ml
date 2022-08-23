from uncoverml.scripts import uncoverml as uncli

model_file = './gbquantile/gbquantiles.model'
partitions = 200

if __name__ == '__main__':
    print('prediction started')
    uncli.predict(model_file, partitions)
    print('prediction done')
