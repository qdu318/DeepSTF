import os
import json
import argparse
import torch
import torch.nn as nn

from utils.utils import get_normalized_adj
from utils.data_load import Data_load
from model.DeepSTF import DeepSTF
from process.train import Train
from process.evaluate import Evaluate

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = json.load(open('./config.json', 'r'))

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--weight_file', type=str, default='./saved_weights/')
parser.add_argument('--timesteps_input', type=int, default=12)
parser.add_argument('--timesteps_output', type=int, default=9)
parser.add_argument('--out_channels', type=int, default=64)
parser.add_argument('--spatial_channels', type=int, default=16)
parser.add_argument('--N', type=int, default=25)
parser.add_argument('--features', type=int, default=1)
parser.add_argument('--time_slice', type=list, default=[3, 6, 9])
args = parser.parse_args()


if __name__ == '__main__':
    torch.manual_seed(7)
    W_nodes, data_set = Data_load(config, args.timesteps_input, args.timesteps_output)
    model = DeepSTF(
                num_nodes=args.N,
                out_channels=args.out_channels,
                spatial_channels=args.spatial_channels,
                features=args.features,
                timesteps_input=args.timesteps_input,
                timesteps_output=args.timesteps_output
            )
    W_nodes = get_normalized_adj(W_nodes)
    if torch.cuda.is_available():
        model.cuda()
        W_nodes = W_nodes.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    L2 = nn.MSELoss()
    for epoch in range(args.epochs):
        train_loss = Train(
                        model=model,
                        optimizer=optimizer,
                        loss_meathod=L2,
                        W_nodes=W_nodes,
                        data_set=data_set,
                        batch_size=args.batch_size
                    )
        torch.cuda.empty_cache()
        with torch.no_grad():
            eval_loss, eval_index = Evaluate(
                                        model=model,
                                        loss_meathod=L2,
                                        W_nodes=W_nodes,
                                        time_slice=args.time_slice,
                                        data_set=data_set
                                    )
        print("--------------------------------------------------------------------------------------------------")
        print("epoch: {}/{}".format(epoch, args.epochs))
        print("Training loss: {}".format(train_loss))
        for i in range(len(args.time_slice)):
            print("time:{}, Evaluation loss:{}, MAE:{}, RMSE:{}"
                  .format(args.time_slice[i] * 5, eval_loss[-(len(args.time_slice) - i)], eval_index['MAE'][-(len(args.time_slice) - i)],
                          eval_index['RMSE'][-(len(args.time_slice) - i)],))
        print("---------------------------------------------------------------------------------------------------")

        if not os.path.exists(args.weight_file):
            os.makedirs(args.weight_file)

        if (epoch % 50 == 0) & (epoch != 0):
            torch.save(model, args.weight_file + 'model_' + str(epoch))

