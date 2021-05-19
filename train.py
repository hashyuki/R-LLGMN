import torch
from tqdm import tqdm


def train_model(net, dataloader, criterion, optimizer, num_epochs):
    net.train()  # set mode
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-------------')
        epoch_loss = 0.0  # sum of epoch loss
        epoch_corrects = 0  # num of epoch corrects
        for (inputs, labels) in tqdm(dataloader):
            # send to device
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
            # with torch.cuda.amp.autocast():
            outputs = net(inputs)
            outputs += 1e-16    # to avoid nan
            loss = criterion(outputs.log(), labels)  # calc loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            _, preds = torch.max(outputs, 1)  # predict label

            # calc epoch loss & acc
            epoch_loss += loss.item() * len(labels)
            epoch_corrects += torch.sum(preds == labels.data)

        epoch_loss = epoch_loss / len(dataloader.dataset)
        epoch_acc = epoch_corrects / len(dataloader.dataset)
        epoch_acc = epoch_acc.to('cpu').detach().numpy().copy()
        print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
