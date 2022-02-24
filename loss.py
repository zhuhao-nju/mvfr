import torch.nn as nn

def select_loss_func(name):

    if name == "L1":
        return nn.L1Loss()
    elif name == "MSE":
        return nn.MSELoss()
    elif name == "BCE":
        return nn.BCELoss()
    else:
        raise Exception("Undefined loss function")