""" make dirs for the dataset folder """
from os import mkdir
from os.path import exists

dataroot = "/media/xyz/RED31/mvfr_released/dev"

data_structure = {
    "images": "", 
    "fit_mesh": "", 
    "maps": {
        "train": {
            "norm_map": "",
            "mv_map": "",
            "pos_map": ""
        }, 
        "eval": {
            "norm_map": "",
            "mv_map": "",
            "pos_map": ""
        }
    }, 
    "volume": {
        "train": "", 
        "eval": ""
    }, 
    "pred": {
        "reg_map": "",
        "reg_mesh": "",
        "texture": "",
        "texture_relocated": "",
        "dp_map": "",
    }  
}

def create_folder(path, structure):
    if not isinstance(structure, dict):
        return 
    
    for key in structure.keys():
        if not exists(path + f"/{key}"):
            mkdir(path + f"/{key}")
        create_folder(path + f"/{key}", structure[key])
    
    return


create_folder(dataroot, data_structure)
print("Data folders have been created")