import torch
from text_recognizer import CNN
import cv2

def load_model(weights,device):
    model = CNN().to(device)
    model.load_state_dict(torch.load(weights,map_location=device))
    model.eval()
    return model

def predict_characters(model,cv_numpy_array,device):
    img = cv2.rotate(cv_numpy_array, cv2.ROTATE_90_CLOCKWISE)
    img = cv2.flip(img, 1)
    img_norm = (img / 255.0 - 0.5) / 0.5
    img_tensor = torch.tensor(img_norm,dtype=torch.float32)
    
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(img_tensor)
        predicted_class = torch.argmax(prediction,dim=1).item()

    return predicted_class