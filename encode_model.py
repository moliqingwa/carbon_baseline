import base64
import pickle
import torch

from utils import storage


if __name__ == "__main__":
    model_dir = storage.get_model_dir("carbon")
    status = storage.get_status(model_dir, torch.device("cpu"))
    actor_model = status['actor_model']

    for name, param in actor_model.items():
        actor_model[name] = param.numpy()

    model_byte = base64.b64encode(pickle.dumps(actor_model))
    with open("./model/actor.txt", 'wb') as f:
        f.write(model_byte)
    pass
