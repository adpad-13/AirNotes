from text_recognizer import CNN
import torch
import torch.nn as nn
from dataset import get_dataloaders


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN().to(device)
train_loader , test_loader = get_dataloaders()
loss_fn = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(model.parameters(),lr=0.001)

def train(model,num_epochs,train_dl):
    loss_hist_train = [0]*num_epochs
    accuracy_hist_train = [0]*num_epochs
    for epochs in range(num_epochs):
        model.train()
        for x_batch,y_batch in train_dl:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimiser.zero_grad()
            pred = model(x_batch)
            loss = loss_fn(pred,y_batch)
            loss.backward()
            optimiser.step()
            loss_hist_train[epochs] += loss.item() * y_batch.size(0)
            is_correct = (torch.argmax(pred,dim=1)==y_batch).float()
            accuracy_hist_train[epochs] += is_correct.sum()

        loss_hist_train[epochs] /= len(train_dl.dataset)
        accuracy_hist_train[epochs] /= len(train_dl.dataset)

        print(f'Epoch {epochs+1} accuracy: {accuracy_hist_train[epochs]:.4f}')

    return loss_hist_train , accuracy_hist_train


if __name__=='__main__':
    torch.manual_seed(1)
    num_epochs = 20
    print("starting training on:",device)
    train_losses,train_accuracies = train(model,num_epochs,train_loader)

    torch.save(model.state_dict(),"emnist_cnn_weights.pth")
    print("training complete")



