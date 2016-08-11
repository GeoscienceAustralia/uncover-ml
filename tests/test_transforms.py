import pytest
import numpy as np
from uncoverml import transforms


@pytest.fixture(params=list(transforms.transforms.keys()))
def get_transform_names(request):
    return request.param


def test_transform(get_transform_names):

    y = np.concatenate((np.random.randn(100), np.random.randn(100) + 5))
    transformer = transforms.transforms[get_transform_names]()

    if hasattr(transformer, 'offset'):
        y -= (y.min() - 1e-5)

    transformer.fit(y)
    yt = transformer.transform(y)
    yr = transformer.itransform(yt)

    assert np.allclose(yr, y)
