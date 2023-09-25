import random

glb_seed = 0
def gener_seed():
    global glb_seed
    glb_seed = random.sample(range(1, 100), 1)[0]
    return glb_seed

def get_value():
    global glb_seed
    return glb_seed