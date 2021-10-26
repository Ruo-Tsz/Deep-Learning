import numpy as np

file_path ="/home/ee904/tsz/Deep Learning/hw1/1/fig/4_layer_leaky/100/epoch_5_0.5/w/5_weights.npz"

wee = np.load(file_path)
print(type(wee))
print(len(wee))
print(wee)
for w in wee:
    print(w)
    print(type(w))
    # print(wee[w])