import scatterem


def test_public_package_imports():
    assert scatterem.__version__ == "0.1.0"
    assert scatterem.PUBLIC_STATUS == "pre-release scaffold"
