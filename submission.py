
import base64
import pickle
import torch


with open("./model/actor.txt", "rt") as f:
    model_text = f.read()

model = base64.b64decode(model_text)
for name, param in model.items():
    model[name] = torch.tensor(param)


