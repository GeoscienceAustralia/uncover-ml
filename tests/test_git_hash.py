from uncoverml.git_hash import git_hash

def test_git_hash():
    value = git_hash
    assert isinstance(value, str)
    assert len(value) == 40
