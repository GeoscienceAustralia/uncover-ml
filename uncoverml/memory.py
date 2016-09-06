import numpy as np

from uncoverml import geoio
from uncoverml import models


def estimate(config, partitions, subsample_fraction, overhead=2):
    targets = geoio.load_targets(shapefile=config.target_file,
                                 targetfield=config.target_property)
    n_targets = targets.observations.shape[0]
    chunksets = geoio.image_resolutions(config)
    res = np.array(next(iter(chunksets[0].values()))[0:2], dtype=float)
    band_pixels = np.product(res)
    n_input_bands = 0
    max_input_bands = 0
    for c in chunksets:
        n_input_bands += np.sum([float(v[2]) for v in c.values()])
        max_input_bands = max(np.amax([v[2] for v in c.values()]),
                              max_input_bands)

    model = models.modelmaps[config.algorithm]()
    n_output_bands = len(model.get_predict_tags())
    bytes_per_pixel = 8 + 1  # float64 values + boolean mask

    # learning (extraction stage and learning stage)
    nbytes_l1 = max_input_bands * band_pixels * bytes_per_pixel
    nbytes_l2 = (n_input_bands + n_output_bands) * n_targets * bytes_per_pixel
    nbytes_l = nbytes_l1 + nbytes_l2
    ngigs_l = nbytes_l * overhead / 1e9 / partitions
    # prediction
    nbytes_p = (n_input_bands + n_output_bands) * band_pixels * bytes_per_pixel
    ngigs_p = nbytes_p * overhead / 1e9 / partitions
    # clustering
    nbytes_c1 = max_input_bands * band_pixels * bytes_per_pixel
    nbytes_c2 = (n_input_bands + 1) * band_pixels * bytes_per_pixel
    ngigs_c1 = nbytes_c1 * 2.5 / 1e9  # 2.5 is rough overhead here
    ngigs_c2 = nbytes_c2 * overhead / 1e9 * subsample_fraction

    result = {'learning': ngigs_l, 'prediction': ngigs_p,
              'clustering-extraction': ngigs_c1,
              'clustering-iteration': ngigs_c2}
    return result
