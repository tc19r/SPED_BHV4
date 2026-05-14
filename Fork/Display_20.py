import numpy as np
import matplotlib.pyplot as plt

#from main import edge_points
from itertools import combinations
import os

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

#Set up a new plotting function, that reuse a modified parametrization function, with positional inputs.
def Surface_triangle(X1, X2, t1, t2):
    # This function take two 3d points X1, X2,
    # and two graded parameters t1, t2,
    # to produce a relevant point cloud in 6d, i.e. the space of the symmetric matrices.
    # The length of t2 and t2 must match.
    m1 = X1
    m2 = X2

    c1 = np.array(t1*(1-t2))
    c2 = np.array(t1*t2)



    surface1 = c1[:, :, np.newaxis] * m1.flatten() + c2[:, :, np.newaxis] * m2.flatten()


    return [surface1]  # ,s1,s2,s3

def Surface(X1, X2, t1, t2, pos1=None, pos2=None,pos3=None):
    # This function take two 3d points X1, X2,
    # and two graded parameters t1, t2,
    # to produce a relevant point cloud in 3d space.
    # Consistent to the second parametrization.
    m1 = X1
    m2 = X2

    c1 = np.array(t1)
    c2 = np.array(t2)

    surface1 = c1[:, :, np.newaxis] * m1.flatten() + c2[:, :, np.newaxis] * m2.flatten()
    # surface1 = surface1[:, :, [pos1[0], pos2[0], pos3[0]], [pos1[1], pos2[1], pos3[1]]]
    return [surface1]

def Surface_sym(X1, X2, t1,t2, pos1, pos2,pos3):
    # This function take two 3d points X1, X2,
    # and two graded parameters t1, t2,
    # to produce a relevant point cloud in 6d, i.e. the space of the symmetric matrices.
    # The length of t2 and t2 must match.
    m1 = X1 @ X1.T
    m2 = X2 @ X2.T
    m3 = X1 @ X2.T + X2 @ X1.T
    c = (X1 + X2)

    c1 = np.array((t1 ** 2) * (1 - t2) ** 2)
    c2 = np.array((t1 ** 2) * (t2 ** 2))
    c3 = np.array((t1 ** 2) * (1 - t2) * t2)

    surface1 = c1[:, :, np.newaxis, np.newaxis] * m1 + c2[:, :, np.newaxis, np.newaxis] * m2 + c3[
        :, :, np.newaxis, np.newaxis] * m3
    surface1 = surface1[:, :, [pos1[0], pos2[0], pos3[0]], [pos1[1], pos2[1], pos3[1]]]

    return [surface1]  # ,s1,s2,s3



def Surface_sym_re(X1, X2, t1, t2, pos1, pos2,pos3):
    # This function take two 3d points X1, X2,
    # and two graded parameters t1, t2,
    # to produce a relevant point cloud in 6d, i.e. the space of the symmetric matrices.
    # The length of t2 and t2 must match.
    # A Second parametrization
    m1 = X1 @ X1.T
    m2 = X2 @ X2.T
    m3 = X1 @ X2.T + X2 @ X1.T

    c1 = np.array(t1 ** 2)
    c2 = np.array(t2 ** 2)
    c3 = np.array(t1 * t2)

    surface1 = c1[:, :, np.newaxis, np.newaxis] * m1 + c2[:, :, np.newaxis, np.newaxis] * m2 + c3[
        :, :, np.newaxis, np.newaxis] * m3
    surface1 = surface1[:, :, [pos1[0], pos2[0], pos3[0]], [pos1[1], pos2[1], pos3[1]]]

    return [surface1]  # ,s1,s2,s3

def data_embedding(data):
    #This function takes a n*d data and perform embedding XX&T
    #And returns a length n list of d by d matrices.
    X=np.asarray(data)
    d=X.shape[0]
    n=X.shape[1]
    datas=[]
    for i in range(n):
        v=X[:,i].reshape(-1,1)
        datas.append(v @ v.T)
    return datas

plot_title='sub-coordinates representation'
edge_points = [
    (X14, X15),
    (X15, X11),
    (X11, X12),
    (X12, X13),
    (X13, X14),
    (X11, X16),
    (X12, X17),
    (X13, X18),
    (X14, X19),
    (X15, X20),
    (X19, X8),
    (X18, X8),
    (X18, X9),
    (X17, X9),
    (X17, X10),
]

#For combo in combination, plot_surface(edge_points,combo,plot_title='')
u = np.linspace(0, 1, 20)
v = np.linspace(0, 1, 20)
t1, t2 = np.meshgrid(u, v)
positions = [(0,0), (0,1), (0,2), (1,1), (1,2), (2,2)]
all_combinations = list(combinations(range(6), 3))

# for combo in all_combinations:
#     pos1,pos2,pos3 = [positions[j] for j in combo]
#     surfaces_data =     [
#         (Surface_sym_re(X14, X15, t1, t2, pos1, pos2, pos3), 'b', 'Surface 1'),
#         (Surface_sym_re(X15,X11,t1,t2,pos1, pos2, pos3), 'green', 'Surface 2'),
#         (Surface_sym_re(X11,X12,t1,t2,pos1, pos2, pos3), 'red', 'Surface 3'),
#         (Surface_sym_re(X12,X13,t1,t2,pos1, pos2, pos3), 'cyan', 'Surface 4'),
#         (Surface_sym_re(X13,X14,t1,t2,pos1, pos2, pos3), 'yellow', 'Surface 5'),
#         (Surface_sym_re(X11,X16,t1,t2,pos1, pos2, pos3), 'magenta', 'Surface 6'),
#         (Surface_sym_re(X12,X17,t1,t2,pos1, pos2, pos3), 'brown', 'Surface 7'),
#         (Surface_sym_re(X13,X18,t1,t2,pos1, pos2, pos3), 'olive', 'Surface 8'),
#         (Surface_sym_re(X14,X19,t1,t2,pos1, pos2, pos3), 'lime', 'Surface 9'),
#         (Surface_sym_re(X15,X20,t1,t2,pos1, pos2, pos3), 'yellow', 'Surface 10'),
#         (Surface_sym_re(X19,X8,t1,t2,pos1, pos2, pos3), 'gray', 'Surface 11'),
#         (Surface_sym_re(X18,X8,t1,t2,pos1, pos2, pos3), 'orange', 'Surface 12'),
#         (Surface_sym_re(X18,X9,t1,t2,pos1, pos2, pos3), 'teal', 'Surface 15'),
#         (Surface_sym_re(X17,X9,t1,t2,pos1, pos2, pos3), 'burlywood', 'Surface 13'),
#         (Surface_sym_re(X17,X10,t1,t2,pos1, pos2, pos3), 'pink', 'Surface 14'),
#
#     ]
#
#     fig2 = plt.figure(figsize=(10, 8))
#     ax_combined = fig2.add_subplot(111, projection='3d')
#
#     # Plot all surfaces in one plot
#     for surfs, color, title in surfaces_data:
#         # 2_9: this code looks a bit strange here.
#         for idx, surf in enumerate(surfs):
#             ax_combined.plot_surface(surf[:, :, 0], surf[:, :, 1], surf[:, :, 2],
#                                      color=color, edgecolor='lightgray', linewidth=0.3, alpha=0.3)
#
#     # Add origin point
#     ax_combined.scatter(0, 0, 0, color='red', s=20)
#
#     # Set labels and title
#     ax_combined.set_xlabel('X')
#     ax_combined.set_ylabel('Y')
#     ax_combined.set_zlabel('Z')
#     ax_combined.set_title(plot_title)  # All Surfaces Combined
#     output_dir = r'C:\Users\tinga\Pictures\Temp\20chose3'
#     filename_str = f"{pos1[0]}{pos1[1]}_{pos2[0]}{pos2[1]}_{pos3[0]}{pos3[1]}"
#     plt.savefig(os.path.join(output_dir, f'surface_{filename_str}.png'))
#     #plt.savefig(f'surface_{filename_str}.png')
#     #plt.show()


#Selected Coordinate systems:
# Working plot for the 15 surfaces.
# pos1=(0,1)
# pos2=(1,1)
# pos3=(2,2)
# surfaces_data_sle =     [
#     (Surface_sym_re(X14, X15, t1, t2, pos1, pos2, pos3), 'b', 'Surface 1'),
#     (Surface_sym_re(X15,X11,t1,t2,pos1, pos2, pos3), 'green', 'Surface 2'),
#     (Surface_sym_re(X11,X12,t1,t2,pos1, pos2, pos3), 'red', 'Surface 3'),
#     (Surface_sym_re(X12,X13,t1,t2,pos1, pos2, pos3), 'cyan', 'Surface 4'),
#     (Surface_sym_re(X13,X14,t1,t2,pos1, pos2, pos3), 'yellow', 'Surface 5'),
#     (Surface_sym_re(X11,X16,t1,t2,pos1, pos2, pos3), 'magenta', 'Surface 6'),
#     (Surface_sym_re(X12,X17,t1,t2,pos1, pos2, pos3), 'brown', 'Surface 7'),
#     (Surface_sym_re(X13,X18,t1,t2,pos1, pos2, pos3), 'olive', 'Surface 8'),
#     (Surface_sym_re(X14,X19,t1,t2,pos1, pos2, pos3), 'lime', 'Surface 9'),
#     (Surface_sym_re(X15,X20,t1,t2,pos1, pos2, pos3), 'yellow', 'Surface 10'),
#     (Surface_sym_re(X19,X8,t1,t2,pos1, pos2, pos3), 'gray', 'Surface 11'),
#     (Surface_sym_re(X18,X8,t1,t2,pos1, pos2, pos3), 'orange', 'Surface 12'),
#     (Surface_sym_re(X18,X9,t1,t2,pos1, pos2, pos3), 'teal', 'Surface 15'),
#     (Surface_sym_re(X17,X9,t1,t2,pos1, pos2, pos3), 'burlywood', 'Surface 13'),
#     (Surface_sym_re(X17,X10,t1,t2,pos1, pos2, pos3), 'pink', 'Surface 14'),
#
# ]
#
# fig2 = plt.figure(figsize=(10, 8))
# ax_combined = fig2.add_subplot(111, projection='3d')
#
# # Plot all surfaces in one plot
# for surfs, color, title in surfaces_data_sle:
#     # 2_9: this code looks a bit strange here.
#     for idx, surf in enumerate(surfs):
#         ax_combined.plot_surface(surf[:, :, 0], surf[:, :, 1], surf[:, :, 2],
#                                  color=color, edgecolor='lightgray', linewidth=0.3, alpha=0.3)
#
# # Add origin point
# ax_combined.scatter(0, 0, 0, color='red', s=20)
#
# # Set labels and title
# ax_combined.set_xlabel('X')
# ax_combined.set_ylabel('Y')
# ax_combined.set_zlabel('Z')
# ax_combined.set_title(plot_title)  # All Surfaces Combined
# # output_dir = r'C:\Users\tinga\Pictures\Temp\20chose3'
# # filename_str = f"{pos1[0]}{pos1[1]}_{pos2[0]}{pos2[1]}_{pos3[0]}{pos3[1]}"
# # plt.savefig(os.path.join(output_dir, f'surface_{filename_str}.png'))
# plt.show()

#Candidates:
# pos1=(0,1)
# pos2=(1,1)
# pos3=(2,2)

#3/26/26：
#Creating plots
n_samples = 100
x = np.random.uniform(0, 1, n_samples)
y = np.random.uniform(0, 1, n_samples)
coords=np.vstack([x,y])
e1=X14.flatten()
e2=X15.flatten()
f=np.matrix([e1, e2 ,np.cross(e1,e2)]).T
coords=np.vstack([coords,np.zeros(n_samples)])
z_t= np.matmul(f,coords)
book={}
datas=data_embedding(z_t)
embedded_coords=[datas[i][[0,1,1],[0,0,1]] for i in range(len(datas))]
book['data']=np.asarray(embedded_coords)

pos1=(0,0)
pos2=(1,0)
pos3=(1,1)
surfaces_data_sle =     [
    (Surface_sym_re(X14, X15, t1, t2,pos1, pos2, pos3), 'b', 'Surface 1'),
    (Surface_sym_re(X15,X11,t1,t2,pos1, pos2, pos3), 'green', 'Surface 2'),
    (Surface_sym_re(X11,X12,t1,t2,pos1, pos2, pos3), 'red', 'Surface 3'),
    (Surface_sym_re(X12,X13,t1,t2,pos1, pos2, pos3), 'cyan', 'Surface 4'),
    (Surface_sym_re(X13,X14,t1,t2,pos1, pos2, pos3), 'yellow', 'Surface 5'),
    (Surface_sym_re(X11,X16,t1,t2,pos1, pos2, pos3), 'magenta', 'Surface 6'),
    (Surface_sym_re(X12,X17,t1,t2,pos1, pos2, pos3), 'brown', 'Surface 7'),
    (Surface_sym_re(X13,X18,t1,t2,pos1, pos2, pos3), 'olive', 'Surface 8'),
    (Surface_sym_re(X14,X19,t1,t2,pos1, pos2, pos3), 'lime', 'Surface 9'),
    (Surface_sym_re(X15,X20,t1,t2,pos1, pos2, pos3), 'yellow', 'Surface 10'),
    (Surface_sym_re(X19,X8,t1,t2,pos1, pos2, pos3), 'gray', 'Surface 11'),
    (Surface_sym_re(X18,X8,t1,t2,pos1, pos2, pos3), 'orange', 'Surface 12'),
    (Surface_sym_re(X18,X9,t1,t2,pos1, pos2, pos3), 'teal', 'Surface 15'),
    (Surface_sym_re(X17,X9,t1,t2,pos1, pos2, pos3), 'burlywood', 'Surface 13'),
    (Surface_sym_re(X17,X10,t1,t2,pos1, pos2, pos3), 'pink', 'Surface 14'),

]

fig2 = plt.figure(figsize=(10, 8))
ax_combined = fig2.add_subplot(111, projection='3d')

# Plot all surfaces in one plot
for surfs, color, title in surfaces_data_sle:
    # 2_9: this code looks a bit strange here.
    for idx, surf in enumerate(surfs):
        ax_combined.plot_surface(surf[:, :, 0], surf[:, :, 1], surf[:, :, 2],
                                 color=color, edgecolor='lightgray', linewidth=0.3, alpha=0.3)

# Add origin point
ax_combined.scatter(0, 0, 0, color='red', s=20)

data = book['data']
# for i in range(len(data)):
#     ax_combined.scatter(data[i][0], data[i][1], data[i][2])  # Only support list, changed to arrays.
for i in range(data.shape[0]):
    ax_combined.scatter(data[i, 0], data[i, 1], data[i, 2], color='grey')

# Set labels and title
ax_combined.set_xlabel('X')
ax_combined.set_ylabel('Y')
ax_combined.set_zlabel('Z')
ax_combined.set_title(plot_title)  # All Surfaces Combined
# output_dir = r'C:\Users\tinga\Pictures\Temp\20chose3'
# filename_str = f"{pos1[0]}{pos1[1]}_{pos2[0]}{pos2[1]}_{pos3[0]}{pos3[1]}"
# plt.savefig(os.path.join(output_dir, f'surface_{filename_str}.png'))
plt.show()

# Candidates:
# pos1=(0,1)
# pos2=(1,1)
# pos3=(2,2)