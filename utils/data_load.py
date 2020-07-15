# encoding utf-8
import numpy as np

from utils.utils import Z_Score
from utils.utils import generate_dataset


def Data_load(config, timesteps_input, timesteps_output):
    W_nodes = np.load(config['W_nodes']).astype(np.float32)
    X = np.load(config['V_nodes']).astype(np.float32)
    Weather = np.load(config['Weather']).astype(np.float32)

    X = np.reshape(X, (X.shape[0], X.shape[1], 1)).transpose((1, 2, 0))

    X, X_mean, X_std = Z_Score(X)
    Weather, _, _ = Z_Score(Weather)

    index_1 = int(X.shape[2] * 0.6)
    index_2 = int(X.shape[2] * 0.8)

    train_original_data = X[:, :, :index_1]
    val_original_data = X[:, :, index_1:index_2]

    index_1 = int(Weather.shape[0] * 0.6)
    index_2 = int(Weather.shape[0] * 0.8)
    train_weather = Weather[:index_1]
    val_weather = Weather[index_1:index_2]

    train_input, train_target = generate_dataset(train_original_data,
                                                 num_timesteps_input=timesteps_input,
                                                 num_timesteps_output=timesteps_output)
    evaluate_input, evaluate_target = generate_dataset(val_original_data,
                                                       num_timesteps_input=timesteps_input,
                                                       num_timesteps_output=timesteps_output)

    data_set = {}
    data_set['train_input'], data_set['train_target'], data_set['eval_input'], data_set[
        'eval_target'], data_set['train_weather'], data_set['eval_weather'], data_set['X_mean'], data_set['X_std'], \
        = train_input, train_target, evaluate_input, evaluate_target, train_weather, val_weather, X_mean, X_std

    return W_nodes, data_set

