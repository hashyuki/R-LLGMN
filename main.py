import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch
from model.RLLGMN import RLLGMN
from train import train_model

def main():
    # fix seed and device
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # set each parameter 
    in_features = 10
    n_class = 2
    n_state = 2
    n_component = 3
    n_epoch = 10
    batch_size = 32

    # generate datas
    x_train = torch.randn(100, 100, in_features)  # (nun_data, depth, features)
    x_train[0:50, :, :] += 0.5
    y_train = torch.ones((100), dtype=int)
    y_train[0:50] -= 1

    # make dataset
    train_dataset = TensorDataset(x_train, y_train)

    # make dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # make a list
    net = RLLGMN(in_features, n_class, n_state, n_component)
    if torch.cuda.is_available():
        net = net.cuda()
    criterion = nn.NLLLoss()    # loss function. Don't use CrossEntropyLoss with R-LLGMN.
    optimizer = optim.SGD(net.parameters(), lr=0.001, weight_decay=1e-6, momentum=0.9, nesterov=True)  # optimizer

    # train model
    train_model(net, train_dataloader, criterion, optimizer, n_epoch)

if __name__ == '__main__':
    main()