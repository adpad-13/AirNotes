import torch
import torch.nn as nn
from text_recognizer import CNN

from dataset import get_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def test(model,test_dl,loss_fn):
    model.eval()
    test_loss = 0
    test_accuracy = 0

    with torch.no_grad():
        for x_batch, y_batch in test_dl:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            pred  = model(x_batch)
            loss = loss_fn(pred,y_batch)

            test_loss += loss.item() * y_batch.size(0)

            is_correct = (torch.argmax(pred,dim=1)==y_batch).float()
            test_accuracy += is_correct.sum()

        test_loss /= len(test_dl.dataset)
        test_accuracy /= len(test_dl.dataset)

        print(f'Test Loss: {test_loss:.4f} | Test accuracy: {test_accuracy:.4f}')

        return test_loss,test_accuracy
    
if __name__ == "__main__":
    _ , test_loader = get_dataloaders()
    model = CNN().to(device)
    model.load_state_dict(torch.load("emnist_cnn_weights.pth"))

    loss_fn = nn.CrossEntropyLoss()

    test(model,test_loader,loss_fn)

