from __future__ import division

import random
import string
import os.path
import os

import pytest
import numpy as np
import shapefile as shp

from uncoverml import mpiops

UNCOVER = os.path.dirname(os.path.dirname(__file__))
os.environ['UNCOVERML_SRC_DIR'] = UNCOVER
timg = np.reshape(np.arange(1, 17), (4, 4, 1))
SEED = 665
RND = np.random.RandomState(SEED)
random.seed(SEED)

def pytest_addoption(parser):
    parser.addoption('--num_procs', type=int)
    parser.addoption('--num_parts', type=int)

@pytest.fixture(scope='session')
def num_procs(request):
    np = request.config.getoption('--num_procs', default=1)
    np = np if np else 1
    return np

@pytest.fixture(scope='session')
def num_parts(request):
    np = request.config.getoption('--num_parts', default=1)
    np = np if np else 1
    return np

@pytest.fixture(scope='session')
def uncover():
    return UNCOVER

@pytest.fixture
def random_filename(tmpdir_factory):
    def make_random_filename(ext=''):
        dir = str(tmpdir_factory.mktemp('uncoverml').realpath())
        fname = ''.join(random.choice(string.ascii_lowercase)
                        for _ in range(10))
        return os.path.join(dir, fname + ext)
    return make_random_filename

@pytest.fixture
def masked_array():
    x = np.ma.masked_array(data=[[0., 1.], [2., 3.], [4., 5.]],
                           mask=[[True, False], [False, True], [False, False]],
                           fill_value=1e23)
    return x

@pytest.fixture
def masked_data():
    yt, Xt, ys, Xs = make_linear_data()
    mt = RND.choice([True, False], size=Xt.shape, p=[0.05, 0.95])
    ms = RND.choice([True, False], size=Xs.shape, p=[0.05, 0.95])
    Xtm = np.ma.masked_array(Xt, mask=mt)
    Xsm = np.ma.masked_array(Xs, mask=ms)
    return yt, Xtm, ys, Xsm

@pytest.fixture
def int_masked_array():
    x = np.ma.masked_array(data=[[1, 1], [2, 3], [4, 5], [3, 3]],
                           mask=[[True, False], [False, True], [False, False],
                                 [False, False]],
                           fill_value=-9999, dtype=int)
    return x

@pytest.fixture
def make_patch_31():
    pwidth = 1

    # Test output patches, patch centres
    tpatch = np.array([[[1, 2, 3],
                        [5, 6, 7],
                        [9, 10, 11]],
                       [[2, 3, 4],
                        [6, 7, 8],
                        [10, 11, 12]],
                       [[5, 6, 7],
                        [9, 10, 11],
                        [13, 14, 15]],
                       [[6, 7, 8],
                        [10, 11, 12],
                        [14, 15, 16]]])[:, :, :, np.newaxis]

    tx = np.array([1, 1, 2, 2])
    ty = np.array([1, 2, 1, 2])

    return timg, pwidth, tpatch, tx, ty

@pytest.fixture
def make_patch_11():
    pwidth = 0

    # Test output patches, patch centres
    tpatch = np.array([[[timg.flatten()]]]).T

    tx, ty = [g.flatten() for g in np.meshgrid(np.arange(3), np.arange(3))]

    return timg, pwidth, tpatch, tx, ty

@pytest.fixture
def make_points():
    pwidth = 1
    points = np.array([[1, 1], [2, 1], [2, 2]])

    tpatch = np.array([[[1, 2, 3],
                        [5, 6, 7],
                        [9, 10, 11]],
                       [[5, 6, 7],
                        [9, 10, 11],
                        [13, 14, 15]],
                       [[6, 7, 8],
                        [10, 11, 12],
                        [14, 15, 16]]])[:, :, :, np.newaxis]

    return timg, pwidth, points, tpatch

@pytest.fixture
def make_multi_patch(request):
    return request.getfixturevalue(request.param)

@pytest.fixture
def shapefile(random_filename, request):
    # File names for test shapefile and test geotiff
    filename = random_filename(ext=".shp")
    lons = np.arange(0, 20, 2)
    lats = np.arange(-10, 30, 2)

    # Generate data for shapefile
    nsamples = 100
    ntargets = 10
    dlon = lons[RND.randint(0, high=len(lons), size=nsamples)]
    dlat = lats[RND.randint(0, high=len(lats), size=nsamples)]
    fields = [str(i) for i in range(ntargets)] + ["lon", "lat"]
    vals = np.ones((nsamples, ntargets)) * np.arange(ntargets)
    lonlats = np.array([dlon, dlat]).T
    vals = np.hstack((vals, lonlats))

    # write shapefile
    w = shp.Writer(filename)
    w.shapeType = shp.POINT

    # fields
    for f in fields:
        w.field(f, 'N', 16, 6)

    # points
    for p in zip(dlon, dlat):
        w.point(*p)

    # records
    for v in vals:
        vdict = dict(zip(fields, v))
        w.record(**vdict)

    w.close()

    return lonlats, filename

def make_linear_data(seed=SEED):
    rnd = np.random.RandomState(seed)
    Nt = 100
    Ns = 50
    x = np.linspace(-2, 2, Nt + Ns)
    y = 3 + 2 * x + rnd.randn(Nt + Ns) * 1e-3
    X = x[:, np.newaxis]

    trind = rnd.choice(Nt + Ns, Nt, replace=False)
    tsind = np.zeros_like(x, dtype=bool)
    tsind[trind] = True
    tsind = np.where(~tsind)[0]
    return y[trind], X[trind], y[tsind], X[tsind]

@pytest.fixture
def linear_data():
    return make_linear_data

# Make sure all MPI tests use this fixure
@pytest.fixture()
def mpisync(request):
    mpiops.comm_world.barrier()

    def fin():
        mpiops.comm_world.barrier()
    request.addfinalizer(fin)
    return mpiops.comm_world

# Test data
@pytest.fixture(scope='session')
def data_dir(uncover):
    """
    Path to test data directory.
    """
    return os.path.join(uncover, 'tests', 'test_data')

# Sir Samuel 
@pytest.fixture(scope='session')
def data_sirsam(data_dir):
    """
    Path to SirSam test data.
    """
    return os.path.join(data_dir, 'sirsam')

@pytest.fixture(scope='session')
def sirsam_covariate_paths(data_sirsam):
    """
    Paths to SirSam covariate files.
    """
    names = [
        'Clim_Prescott_LindaGregory.tif',
        'U_15v1.tif',
        'U_TH_15.tif',
        'dem_foc2.tif',
        'er_depg.tif',
        'gg_clip.tif',
        'k_15v5.tif',
        'tpi_300.tif'
    ]
    paths = [os.path.join(data_sirsam, 'covariates', n) for n in names]
    return paths

@pytest.fixture(scope='session')
def sirsam_target_path(data_sirsam):
    """
    Path to SirSam target file.
    """
    return os.path.join(data_sirsam, 'targets', 'geochem_sites_log.shp')

# Bootstrapping
@pytest.fixture(scope='session')
def sirsam_bootstrap(data_sirsam):
    """
    Path to SirSam bootstrap files.
    """
    return os.path.join(data_sirsam, 'bootstrapping')

@pytest.fixture(scope='session')
def sirsam_bs_out(sirsam_bootstrap, sirsam_target_path):
    """
    Path to bootstrap output directory.
    """
    return os.path.join(sirsam_bootstrap, 'out')

@pytest.fixture(scope='session')
def sirsam_bs_conf(sirsam_bootstrap):
    """
    Path to bootstrap config.
    """
    return os.path.join(sirsam_bootstrap, 'bootstrapping.yaml')

# Resampling
@pytest.fixture(scope='session')
def sirsam_resample(data_sirsam):
    """
    Path to SirSam resampling outputs.
    """
    return os.path.join(data_sirsam, 'resampling')

@pytest.fixture(scope='session')
def sirsam_rs_out(sirsam_resample, sirsam_target_path):
    """
    Path to ss resample output directory.
    """
    return os.path.join(sirsam_resample, 'out', 
                        os.path.splitext(os.path.basename(sirsam_target_path))[0] + '_resampled')

@pytest.fixture(scope='session')
def sirsam_rs_precomp(sirsam_resample, sirsam_target_path):
    """
    Path to precompued resample outputs.
    """
    return os.path.join(sirsam_resample, 'precomputed',
                        os.path.splitext(os.path.basename(sirsam_target_path))[0] + '_resampled')

@pytest.fixture(scope='session')
def sirsam_rs_conf(sirsam_resample):
    """
    Path to resampling config.
    """
    return os.path.join(sirsam_resample, 'resampling.yaml')

# Random Forest 
@pytest.fixture(scope='session')
def sirsam_rf(data_sirsam):
    """
    Path to SirSam random forest outputs.
    """
    return os.path.join(data_sirsam, 'random_forest')

@pytest.fixture(scope='session')
def sirsam_rf_out(sirsam_rf):
    """
    Path to SirSam random forest output directory.
    """
    return os.path.join(sirsam_rf, 'out')

@pytest.fixture(scope='session')
def sirsam_rf_conf(sirsam_rf):
    """
    Path to SirSam random forest test config.
    """
    return os.path.join(sirsam_rf, 'sirsam_Na_randomforest.yaml')

@pytest.fixture(scope='session')
def sirsam_rf_precomp(sirsam_rf):
    """
    Path to SirSam random forest precomputed outputs.
    """
    return os.path.join(sirsam_rf, 'precomputed')

@pytest.fixture(scope='session')
def sirsam_rf_precomp_learn(sirsam_rf_precomp):
    """
    Path to SirSam random forest precomputed learn outputs.
    """
    return os.path.join(sirsam_rf_precomp, 'learn')

@pytest.fixture(scope='session')
def sirsam_rf_precomp_predict(sirsam_rf_precomp):
    """
    Path to SirSam random forest precomputed predict outputs.
    """
    return os.path.join(sirsam_rf_precomp, 'predict')

@pytest.fixture(scope='session')
def sirsam_rf_precomp_shiftmap(sirsam_rf_precomp):
    """
    Path to SirSam random forest precomputed shiftmap outputs.
    """
    return os.path.join(sirsam_rf_precomp, 'shiftmap')

