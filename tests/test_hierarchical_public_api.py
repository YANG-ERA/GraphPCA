def test_hierarchical_public_api_imports():
    from GraphPCA import Run_GPCA, Run_Hierarchical_Multi_GPCA
    assert callable(Run_GPCA)
    assert callable(Run_Hierarchical_Multi_GPCA)
