import base64
import pickle
import torch


if __name__ == "__main__":
    model_path = "model.pth"  # your model file path

    model_state_dict = torch.load(str(model_path), map_location=torch.device("cpu"))
    actor_model = model_state_dict['actor']

    for name, param in actor_model.items():
        actor_model[name] = param.numpy()

    model_byte = base64.b64encode(pickle.dumps(actor_model))
    with open("actor.txt", 'wb') as f:
        f.write(model_byte)
    pass
