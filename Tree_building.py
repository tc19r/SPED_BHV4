# 2/15/26：
#Utilizing dataset from https://www.nature.com/articles/s41586-018-0030-5
#This scripts map the yeast trees to points on the SPED.
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.style as m

#Preprocessing metadata, - Get all the data and store it in a data frame?
#We need the index to search in the distance matrix. Coding in Colab.
#2/16
#Everything has actually been done in colab, and the datas are store in dictionaries.
#All that's left is to translate and display.
# Define vertices.
A = 0.0
B = 0.5773502691896257
C = 0.7946544722917661
D = 0.18759247408507987
E = 0.9822469463768461
FF = 0.6070619982066863
G = 0.9341723589627158
H = 0.3568220897730899
I = 0.49112347318842303
J = 0.30353099910334314

flag_Rotate = 0


# Define a rotation matrix K
def rotation_matrix_y_axis(theta):
    """
    Creates a 3x3 rotation matrix for rotation around the y-axis.
    Args:
        theta (float): The rotation angle in radians.
    Returns:
        numpy.ndarray: The 3x3 rotation matrix.
    """
    R = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    return R


if flag_Rotate:
    K = rotation_matrix_y_axis(np.pi / 30)  # 40 np.pi/20
else:
    K = R = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

X14 = K @ np.asarray([-H, I, C]).reshape(-1, 1)  # 14
X15 = K @ np.asarray([H, I, C]).reshape(-1, 1)  # 15
X11 = K @ np.asarray([B, -D, C]).reshape(-1, 1)  # 11
X12 = K @ np.asarray([A, -FF, C]).reshape(-1, 1)  # 12
X13 = K @ np.asarray([-B, -D, C]).reshape(-1, 1)  # 13
X9 = K @ np.asarray([-B, -C, -D]).reshape(-1, 1)  # 7
X8 = K @ np.asarray([-G, J, -D]).reshape(-1, 1)  # 7
X10 = K @ np.asarray([B, -C, -D]).reshape(-1, 1)
X19 = K @ np.asarray([-B, C, D]).reshape(-1, 1)  # 19
X20 = K @ np.asarray([B, C, D]).reshape(-1, 1)  # 20
X16 = K @ np.asarray([G, -J, D]).reshape(-1, 1)  # 16
X17 = K @ np.asarray([A, -E, D]).reshape(-1, 1)  # 17
X18 = K @ np.asarray([-G, -J, D]).reshape(-1, 1)  # 18

edge_translation={
    '1': (X15,X11),
    '2': (X14,X15),
    '3': (X14,X13),
    '4': (X13,X12),
    '5': (X11,X12),
    '6': (X20,X15),
    '7': (X17,X9),
    '8': (X17,X10),
    '9': (X14,X19),
    '10': (X11,X16),
    '11': (X8,X19),
    '12': (X18,X8),
    '13': (X18,X13),
    '14': (X18,X9),
    '15': (X17,X12)
}

interior_edges = {
    '1': ('e2', 'e1'),
    '2': ('e3', 'e1'),
    '3': ('e3', 'e4'),
    '4': ('e4', 'e5'),
    '5': ('e2', 'e5'),
    '6': ('e6', 'e1'),
    '14': ('e10', 'e6'),
    '13': ('e10', 'e4'),
    '15': ('e7', 'e5'),
    '7': ('e7', 'e6'),
    '8': ('e7', 'e8'),
    '9': ('e3', 'e8'),
    '10': ('e2', 'e9'),
    '11': ('e9', 'e8'),
    '12': ('e10', 'e9')
}
with open('data.json', 'r') as f:
    loaded_data = json.load(f)

z_t=[]
for dict in loaded_data:
    K=dict['K']
    keys=interior_edges[str(K)]
    l1=dict[keys[0]]
    l2=dict[keys[1]]
    x = l1
    y = l2
    coords=np.vstack([x,y])
    e1=edge_translation[str(K)][0].flatten()
    e2=edge_translation[str(K)][1].flatten()
    f=np.matrix([e1, e2 ,np.cross(e1,e2)]).T
    coords=np.vstack([coords,0])
    if len(z_t)==0:
        z_t=np.matmul(f,coords)
    else:
        z_t=np.hstack([z_t, np.matmul(f,coords)])
    # print('working on data on surface' + str(K))
    # print('edge lengths are'+ keys[0] + str(x)+ 'and'+ keys[1] + str(y))
