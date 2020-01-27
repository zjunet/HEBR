# -*- coding: utf-8 -*-


class BasicParam(object):
    save_path = './repo/model/'
    data_name = 'to2019'
    his_len = 180
    n_event = 2
    n_jobs = 40
    gpu = '1'

    input_dim = 6
    dims = {'user': 3,
            'area': 1,
            'climate': 2,
            'user_hidden': 16,
            'area_hidden': 4,
            'climate_hidden': 8,
            'user_area': 32,
            'user_climate': 32,
            'user_area_hidden': 64,
            'user_climate_hidden': 64,
            'user_area_climate': 128,
            'user_area_climate_hidden': 256}
    learning_rate = 0.001
    batch_size = 1500

