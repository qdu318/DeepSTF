# encoding utf-8
import torch

from utils.utils import Un_Z_Score


def Train(model, optimizer, loss_meathod, W_nodes, data_set, batch_size):
    permutation = torch.randperm(data_set['train_input'].shape[0])
    epoch_training_losses = []
    loss_mean = 0.0
    for i in range(0, data_set['train_input'].shape[0], batch_size):
        model.train()
        optimizer.zero_grad()

        indices = permutation[i:i + batch_size]
        X_batch, y_batch, weather_batch = data_set['train_input'][indices], data_set['train_target'][indices], data_set['train_weather'][indices]

        if torch.cuda.is_available():
            X_batch = X_batch.cuda()
            y_batch = y_batch.cuda()
            weather_batch = torch.tensor(weather_batch).cuda()
            std = torch.tensor(data_set['X_std']).cuda()
            mean = torch.tensor(data_set['X_mean']).cuda()

        perd = model(W_nodes, X_batch, weather_batch)
        perd, y_batch = Un_Z_Score(perd, mean, std), Un_Z_Score(y_batch, mean, std)
        loss = loss_meathod(perd, y_batch)

        loss.backward()
        optimizer.step()
        epoch_training_losses.append(loss.detach().cpu().numpy())
        loss_mean = sum(epoch_training_losses)/len(epoch_training_losses)
    return loss_mean