
import os

path = "./Dataset/left/wrist_l/"

files = os.listdir(path)
print(type(files))  # <class 'list'>

# with open("wrist_l.txt", "w") as new_file:
#     for name in files:
#         name = "/home/sobits/catkin_ws/src/Deep-Point/Dataset/left/wrist_l/" + name
#         with open(name) as f:
#             for line in f:
#                 new_file.write(line)
#             new_file.write("\n")


########ここから下はいらない##################

# import numpy as np

# data = np.loadtxt("/home/sobits/catkin_ws/src/Deep-Point/Dataset/left/elbow_l.txt")

# print(type(data))
# print(data.shape)