from uncoverml import parallel


def test_direct_view(make_ipcluster4):
    cluster = parallel.direct_view(profile=None)
    cluster.execute("import sys; mods = set(sys.modules.keys())")
    for i in range(len(cluster)):
        assert 'uncoverml.geoio' in cluster['mods'][i]
        assert 'uncoverml.patch' in cluster['mods'][i]
        assert 'uncoverml.parallel' in cluster['mods'][i]
        assert 'uncoverml.stats' in cluster['mods'][i]
        assert 'numpy' in cluster['mods'][i]
    # check chunks
    assert cluster["chunk_index"] == list(range(4))
