from mini_tpo.data_loading import load_config, read_raw_data


def test_raw_data_loads():
    df = read_raw_data(load_config())
    assert df.shape[0] > 0
    assert "uplift_real" in df.columns

