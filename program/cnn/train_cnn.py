import torch.nn
import torch
import torch.optim
from torch.utils.data.dataloader import DataLoader

LR=0.001
DROPOUT=0.2
EPOCH=100
BATCH_SIZE=50


def train(model:torch.nn.Module,train_data):
    optimizer=torch.optim.Adam(model.parameters(),lr=LR)
    loss_func=torch.nn.CrossEntropyLoss()

    train_loader=DataLoader(train_data,batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(EPOCH):
        for step,(x,y) in enumerate(train_loader):
            output=model(x)
            loss=loss_func(output,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 50 == 0:
                print('Epoch: ', epoch,
                      '| train loss: %.4f' % loss.data.numpy(),
                      # '| test accuracy: %.2f' % accuracy
                      )

def load_dataset():
    pass

def save_model():
    pass

def load_model():
    pass