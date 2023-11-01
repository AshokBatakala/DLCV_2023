import shutil
import random
import os

def make_val_set(train_path, val_path, num):
    """ move n images from train_path folder to val_path folder
    Args: 
    num (int): number of images to move
    
    val_path (str): path to validation folder : it must exist
    """
    
    if not os.path.exists(val_path):
        # throw error: 
        raise Exception("val_path does not exist")
    

    files = os.listdir(train_path)
    random.shuffle(files)
    for file in files[:num]:
        shutil.move(os.path.join(train_path, file), os.path.join(val_path, file))

def num_images_in_folder(path):
    """ return number of images in folder
    """ 
    return len(os.listdir(path))

