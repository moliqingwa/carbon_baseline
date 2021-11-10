import base64
import pickle
import torch

from utils import storage


if __name__ == "__main__":
    actor_state_dict = torch.load(str('./model/actor_best.pth'), map_location = torch.device("cpu"))

    for name, param in actor_state_dict.items():
        actor_state_dict[name] = param.numpy()

    model_byte = base64.b64encode(pickle.dumps(actor_state_dict))
    with open("./model/actor.txt", 'wb') as f:
        f.write(model_byte)
    pass
