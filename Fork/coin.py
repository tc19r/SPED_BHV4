#Take a surface, build a tangent, using the differential. Take a look at it.
#def tangent(t1,t2,book)
# S= D[t1,t2]+x_0
# Plot surface,
# Plot tangents.
#return surface
#2/20： Cleaned code.
#2/21: Working on tangent plane visualization.

import numpy as np
import matplotlib.pyplot as plt
import json

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

def Surface(X1, X2, t1, t2):
    # This function take two 3d points X1, X2,
    # and two graded parameters t1, t2,
    # to produce a relevant point mesh in 3d space.
    # Consistent to the second parametrization.
    m1 = X1
    m2 = X2

    c1 = np.array(t1)
    c2 = np.array(t2)

    surface1 = c1[:, :, np.newaxis] * m1.flatten() + c2[:, :, np.newaxis] * m2.flatten()

    return surface1
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



def Surface_sym_re(X1, X2, t1, t2):
    # This function take two 3d points X1, X2,
    # and two graded parameters t1, t2,
    # to produce a relevant point mesh in 9d, i.e. the space of the symmetric matrices.
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
    surface1 = surface1[:, :, [1, 2, 2], [1, 1, 2]]

    return surface1  # ,s1,s2,s3

def PGD_init(X1, X2):

    M1 = X1 @ X1.T
    M2 = X2 @ X2.T
    M3 = X1 @ X2.T + X2 @ X1.T
    book = {
        'X1':X1,
        'X2':X2,
        'M1': M1,
        'M2': M2,
        'M3': M3,
        't0s': [0.2, 0.5]
    }
    return book

def get_parametrized_surface(t1, t2, book ,flag_embed):
    #Should be better named as: get parametrized point.
    # rho_val = get_rho(t2, M1, M2, M3)
    X1=book['X1']
    X2=book['X2']
    M1=book['M1'].reshape(-1,1)
    M2 = book['M2'].reshape(-1, 1)
    M3 = book['M3'].reshape(-1, 1)
    if flag_embed==0:
        surface= t1 * X1 + t2 * X2
    else:
        surface = t1 ** 2 * M1 + t2 ** 2 * M2 + t1 * t2 * M3
    return surface

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

def PGD(X1, X2, E0,K=100):
    # This function solve the Projected gradient descent, and return a trace of t_1 and t_2 as T by 2 arrays.
    # The trace of loss will also be returned.

    lr = 0.1
    book = PGD_init(X1, X2)
    M1 = book['M1'].reshape(-1, 1)
    M2 = book['M2'].reshape(-1, 1)
    M3 = book['M3'].reshape(-1, 1)
    t1, t2 = book['t0s']
    trace = []

    for k in range(K):
        surface = get_parametrized_surface(t1,t2,book,flag_embed=1)
        diff = surface - E0
        loss = np.linalg.norm(diff) ** 2
        grad_1 = (4 * t1 * M1 + 2 * t2 * M3).T @ diff
        grad_2 = (4 * t2 * M2 + 2 * t1 * M3).T @ diff

        t1 = t1 - lr * grad_1.item()
        t2 = t2 - lr * grad_2.item()

        if t1 < 0:
            t1 = 0
        if t1 > 1:
            t1 = 1
        if t2 < 0:
            t2 = 0
        if t2 > 1:
            t2 = 1

        if k % 10 == 0:
            trace.append([t1, t2, loss])
    return trace

def plot_surface(surfaces_data=None, books=None,flag_x_bar=0, flag_pgd_trace=0, M1=None, M2=None, M3=None,traces=None,flag_surf_one=1,flag_multi_traces=0,flag_embed=0,flag_plot_data=1,data_title='Uniform sampling in the \'blue\' Surface',flag_axis_equal=1,zrange=None,xrange=None,yrange=None,quiver=None,e_mean=None,conf_region=None):
    # Create a combined plot with all surfaces
    # The  data should be a dictionary. Let's say, it has data, extrinsic mean, and projection [10 traces].
    # The surf should probably be done inside of this function
    # optimize the parameter passing later
    # To enable multiple traces, books and traces are list of surface configurations and time traces. Note that this is independent of the surface data, to
    # help create independence of defining/passing the background surfaces and the traces.
    def surface_parametrization(t1, t2,book,flag_embed):  #Note that this function is only used for trace plot on surface, but not the surface itself.
        surface = get_parametrized_surface(t1, t2, book, flag_embed)
        if flag_embed==1:
            c1 = surface[4]
            c2 = surface[7]
            c3 = surface[8]
        else:
            c1 = surface[0]
            c2 = surface[1]
            c3 = surface[2]
        return c1, c2, c3

    fig2 = plt.figure(figsize=(10, 8))
    ax_combined = fig2.add_subplot(111, projection='3d')

    # Plot all surfaces in one plot
    for surf, color, title in surfaces_data:
            if flag_embed == 0:
                ax_combined.plot_surface(surf[:, :, 0], surf[:, :, 1], surf[:, :, 2],
                                         color=color, edgecolor='lightgray', linewidth=0.3, alpha=0.3)
            elif flag_embed ==1 and surf.ndim == 4:
                surf=surf.reshape((*surf.shape[:2], 9))
                ax_combined.plot_surface(surf[:, :, 4], surf[:, :, 7], surf[:, :, 8],
                                         color=color, edgecolor='lightgray', linewidth=0.3, alpha=0.3)
            elif flag_embed ==1 and surf.ndim == 3 and surf.shape[2]>3:
                ax_combined.plot_surface(surf[:, :, 4], surf[:, :, 7], surf[:, :, 8],
                                         color=color, edgecolor='lightgray', linewidth=0.3, alpha=0.0)
            else:
                ax_combined.plot_surface(surf[:, :, 0], surf[:, :, 1], surf[:, :, 2],
                                         color=color, edgecolor='lightgray', linewidth=0.3, alpha=0.3)

    # Add origin point
    ax_combined.scatter(0, 0, 0, color='red', s=20)

    # Set labels and title
    ax_combined.set_xlabel('X')
    ax_combined.set_ylabel('Y')
    ax_combined.set_zlabel('Z')
    ax_combined.set_title(data_title)  # All Surfaces Combined

    # Set equal aspect ratio
    ax_combined.set_box_aspect([1, 1, 1])
    if book and flag_plot_data==1:  #Note that this book here shouldn't be global - 2_2_26
        data = book['data']
        # for i in range(len(data)):
        #     ax_combined.scatter(data[i][0], data[i][1], data[i][2])  # Only support list, changed to arrays.

        if flag_embed==0:
            for i in range(len(data)):
                ax_combined.scatter(data[i,0], data[i,1], data[i,2],color='gray')
        else:
            for i in range(len(data)):
                ax_combined.scatter(data[i].reshape(-1,1)[4], data[i].reshape(-1,1)[7], data[i].reshape(-1,1)[8],color='gray')
    if flag_x_bar:
        tmp = np.asarray(data)
        x_bar = np.mean(tmp, axis=0)
        ax_combined.scatter(x_bar[0], x_bar[1], x_bar[2], s=200, c='red', label='x_bar', alpha=1, depthshade=True)
    # Show the plot

    if flag_pgd_trace:
        for id, trace in enumerate(traces):
            # 2. Map the 2D Trace to 3D Path
            if flag_multi_traces==0 and id>=1:
                break
            path_3d = []
            for t1, t2, loss in trace:
                x, y, z = surface_parametrization(t1, t2, books[id], flag_embed)
                path_3d.append([x, y, z])

            path_3d = np.array(path_3d)

            # 3. Plot the Path on the Surface
            # We use a bright color and higher zorder to make it pop
            ax_combined.plot(path_3d[:, 0], path_3d[:, 1], path_3d[:, 2],
                             color='black', linewidth=3, label='Projection Path', zorder=10)

            # Mark the start and final projection point
            ax_combined.scatter(path_3d[0, 0], path_3d[0, 1], path_3d[0, 2], color='yellow', s=50)
            ax_combined.scatter(path_3d[-1, 0], path_3d[-1, 1], path_3d[-1, 2], color='cyan', s=100, marker='*')

        # Add origin and labels
        ax_combined.scatter(0, 0, 0, color='red', s=20)
        ax_combined.set_xlabel('X')
        ax_combined.set_ylabel('Y')
        ax_combined.set_zlabel('Z')

    if quiver:
       ax_combined.quiver(e_mean.reshape(-1,1)[4],e_mean.reshape(-1,1)[7],e_mean.reshape(-1,1)[8],quiver[0].reshape(-1,1)[4], quiver[0].reshape(-1,1)[7], quiver[0].reshape(-1,1)[8],color='r')
       ax_combined.quiver(e_mean.reshape(-1,1)[4],e_mean.reshape(-1,1)[7],e_mean.reshape(-1,1)[8],quiver[1].reshape(-1,1)[4], quiver[1].reshape(-1,1)[7], quiver[1].reshape(-1,1)[8],color='b')
    if flag_axis_equal==1:
        plt.axis('equal')
    if zrange:
        ax_combined.set_zlim(zrange)
    if xrange:
        ax_combined.set_xlim(xrange)
    if yrange:
        ax_combined.set_ylim(yrange)

    if conf_region:
        ax_combined.plot(conf_region[:, 0], conf_region[:, 1], conf_region[:, 2])
        pass
    plt.show()

def get_min_surface(traces=None,mode='joint'):
    #This is used to compare multiple trace and extract the final coordinates. Note that this can also work for one trace, just to unpack.
    min_loss = 1e3
    min_K = -1
    if mode=='single':
        res = np.array(traces)
        if res[-1,2] < min_loss:
            min_loss = res[-1,2]
            min_K = None
        return min_K, [traces[-1][0], traces[-1][1]]
    else:
        for i, trace in enumerate(traces):
            res = np.array(trace)
            if res[-1, 2] < min_loss:
                min_loss = res[-1, 2]
                min_K = i
        return min_K, [traces[min_K][-1][0],traces[min_K][-1][1]]
def gram_schmidt(quiver=None):
    basis=[]
    v1=quiver[0].flatten()
    v1=v1/np.linalg.norm(v1)
    v2=quiver[1].flatten()
    v2=v2/np.linalg.norm(v2)
    v2=v2-np.dot(v2,v1)*v1
    basis=[v1/np.linalg.norm(v1), v2/np.linalg.norm(v2)]
    return basis
def project(Xa=None,Xb=None,E0=None):

    basis = gram_schmidt([Xa + Xb, Xa])
    basis = gram_schmidt([basis[0] + basis[1], Xa + Xb])
    [e1, e2] = basis
    E0 = E0.reshape(3, 3)
    eigenvalues, eigenvectors = np.linalg.eigh(E0)
    [v1, v2, v3] = [eigenvectors[:, 0], eigenvectors[:, 1], eigenvectors[:, 2]]
    [w1, w2, w3] = eigenvalues[0:3]

    r1 = w1 * np.dot(e1, v1) ** 2 + w2 * np.dot(e1, v2) ** 2 + w3 * np.dot(e1, v3) ** 2
    r2 = w1 * np.dot(e2, v1) ** 2 + w2 * np.dot(e2, v2) ** 2 + w3 * np.dot(e2, v3) ** 2
    r3 = w1 * np.dot(e1, v1) * np.dot(e2, v1) + w2 * np.dot(e1, v2) * np.dot(e2, v2) + w3 * np.dot(e1, v3) * np.dot(e2,v3)

    if r3<=0:
        edge1 = Xa.flatten()
        d1 = w1 * np.dot(edge1, v1) ** 2 + w2 * np.dot(edge1, v2) ** 2 + w3 * np.dot(edge1, v3) ** 2
        c1 = np.sqrt(d1)

        edge2 = Xb.flatten()
        d2 = w1 * np.dot(edge2, v1) ** 2 + w2 * np.dot(edge2, v2) ** 2 + w3 * np.dot(edge2, v3) ** 2
        c2 = np.sqrt(d2)

        if d1>d2:
            return [c1,0,d1]
        else:
            return [0,c2,d2]

    if r3>0:
        beta = ((r2 - r1) / r3) ** 2
        if r2 - r1 > 0:
            estar2 = (1 / 2) * (1 - np.sqrt(beta / (beta + 4)))
        else:
            estar2 = (1 / 2) * (1 + np.sqrt(beta / (beta + 4)))

        y = np.sqrt(estar2) * e1 + np.sqrt(1 - estar2) * e2

        if  y.T @ (Xa+Xb) > Xa.T @ (Xa+Xb):
            a = estar2
            b = 1 - estar2
            d_max = a * r1 + b * r2 + 2 * np.sqrt(a * b) * r3
            c = np.sqrt(d_max)
            y = c * y
            Ap = np.hstack([Xa,Xb])
            tts, _, _, _ = np.linalg.lstsq(Ap, y, rcond=None)#Sketchy
            tts=tts.tolist()
            tts.append(d_max)
            return tts

        elif y.T @ Xa > y.T @ Xb:
            edge1 = Xa.flatten()
            d1 = w1 * np.dot(edge1, v1) ** 2 + w2 * np.dot(edge1, v2) ** 2 + w3 * np.dot(edge1, v3) ** 2
            c1 = np.sqrt(d1)
            return [c1,0,d1]
        else:
            edge2 = Xb.flatten()
            d2 = w1 * np.dot(edge2, v1) ** 2 + w2 * np.dot(edge2, v2) ** 2 + w3 * np.dot(edge2, v3) ** 2
            c2 = np.sqrt(d2)
            return [0,c2,d2]

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



#
# u = np.linspace(0, 1, 20)
# v = np.linspace(0, 1, 20)
# t1, t2 = np.meshgrid(u, v)
# surfaces_data=[
#      (Surface_sym_re(X12,X17,t1,t2), 'brown', 'Surface 7'),
#     #     (Surface_sym_re(X14, X15, t1, t2), 'b', 'Surface 1'),
#     #     (Surface_sym_re(X15,X11,t1,t2), 'green', 'Surface 2'),
#     #     (Surface_sym_re(X11,X12,t1,t2), 'red', 'Surface 3'),
#     #     (Surface_sym_re(X12,X13,t1,t2), 'cyan', 'Surface 4'),
#     #     (Surface_sym_re(X13,X14,t1,t2), 'yellow', 'Surface 5'),
# ]

# plot_surface(surfaces_data,book,traces=None, flag_x_bar=0,flag_multi_traces=0,flag_embed=1,flag_pgd_trace=0,flag_plot_data=1,flag_axis_equal=0,data_title='e_mean(s)')

#On second thought, I could have used get_parametrized surface to create the data directly, not just parametrization.


#
def gram_schmidt(quiver=None):
    basis=[]
    v1=quiver[0].flatten()
    v1=v1/np.linalg.norm(v1)
    v2=quiver[1].flatten()
    v2=v2/np.linalg.norm(v2)
    v2=v2-np.dot(v2,v1)*v1
    basis=[v1/np.linalg.norm(v1), v2/np.linalg.norm(v2)]
    return basis

def get_tangent_surface(X1=None, X2=None, t1=None, t2=None,e_mean=None,ta=None,tb=None,z1=None,z2=None):
    # This function take two 3d points X1, X2,
    # and two graded parameters t1, t2,
    # to produce a relevant point cloud in 6d, i.e. the space of the symmetric matrices.
    # The length of t2 and t2 must match.
    # A Second parametrization

    # An e_mean is passed in , this the true mean. Its parameter values is also passed.
    # Another set of graded parameters z1, z2, is passed, for traversal of the tangent plane.
    m1 = X1 @ X1.T
    m2 = X2 @ X2.T
    m3 = X1 @ X2.T + X2 @ X1.T

    c1 = np.array(t1 ** 2)
    c2 = np.array(t2 ** 2)
    c3 = np.array(t1 * t2)

    surface1 = c1[:, :, np.newaxis, np.newaxis] * m1 + c2[:, :, np.newaxis, np.newaxis] * m2 + c3[
        :, :, np.newaxis, np.newaxis] * m3

    d1 = np.array(z1)
    d2 = np.array(z2)
    d_d1=2* ta * m1 + tb * m3
    d_d2=2* tb * m2 + ta* m3
    surface_tangent= e_mean.reshape(3,3) + d1[:, :, np.newaxis, np.newaxis] * d_d1 + d2[:, :, np.newaxis, np.newaxis] * d_d2
    quiver =[d_d1/np.linalg.norm(d_d1),d_d2/np.linalg.norm(d_d2)]

    return surface1,surface_tangent,quiver  # ,s1,s2,s3


# plot_surface_plotly(surfaces_data=surfaces_data,book=book,traces=None, flag_x_bar=0,flag_multi_traces=0,flag_embed=1,flag_pgd_trace=0,flag_plot_data=1,flag_axis_equal=1,data_title='tagent',quiver=gram_schmidt(quiver),e_mean=e_mean)

# #Projection and eclipse estimation.
# proj_boot_data=[]
# for data in boot_data:
#     b1=gram_schmidt(quiver)[0]
#     b2=gram_schmidt(quiver)[1]
#     proj=np.dot(data.flatten(),b1)*b1 + np.dot(data.flatten(),b2)*b2
#     proj_boot_data.append(proj)
#
# surfaces_data=[
#     (adapted_frame,'blue','tangent')
# ]
# book['data']=proj_boot_data
# plot_surface(surfaces_data=surfaces_data,books=book,traces=None, flag_x_bar=0,flag_multi_traces=0,flag_embed=1,flag_pgd_trace=0,flag_plot_data=1,flag_axis_equal=1,data_title='tagent',quiver=gram_schmidt(quiver),e_mean=e_mean)




#Get all other e_min
##Locate the ones where the means are on the probable surface

with open('../data.json', 'r') as f:
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
mag=1
z_t=z_t*mag
#Get e_min location
datas=data_embedding(z_t)
embedded_coords=[datas[i][[0,1,1],[0,0,1]] for i in range(len(datas))]  # This is a list, should not be passed.
vec_datas=[datas[i].reshape(-1,1) for i in range(len(datas))]
vec_datas=np.asarray(vec_datas) # Use this
E0=np.mean(vec_datas,axis=0)
book=PGD_init(X12,X17)
# e_trace=PGD(X12,X17,E0,K=3000)
[ta,tb,_]=project(X12,X17,E0)
e_mean=get_parametrized_surface(ta,tb,book,flag_embed=1)

with open('../boot_sol.json', 'r') as f:
    data = json.load(f)
    Ts=data['b']
    data_replicates=data['c']




#
u = np.linspace(0, 1, 20)
v = np.linspace(0, 1, 20)
t1, t2 = np.meshgrid(u, v)

u = np.linspace(start=-0.1, stop=0.1, num=20)
v = np.linspace(start=-0.1, stop=0.1, num=20)
z1, z2 = np.meshgrid(u, v)
surface1,surface_tangent,quiver = get_tangent_surface(X12,X17,t1,t2,e_mean,ta,tb,z1,z2)
# surfaces_data=[
#     (Surface_sym_re(X12,X17,t1,t2), 'brown', 'Surface 7'),
#     # (surface_tangent,'blue','tangent'),
# ]
from plotting import plot_surface_plotly

# plot_surface_plotly(surfaces_data=surfaces_data,book=book,traces=None, flag_x_bar=0,flag_multi_traces=0,flag_embed=1,flag_pgd_trace=0,flag_plot_data=1,flag_axis_equal=1,data_title='tagent',quiver=None,e_mean=e_mean,zrange=[-0.5,0.5],yrange=[0,0.6])
#
d1 = np.array(z1)
d2 = np.array(z2)
adapted_frame=e_mean.flatten() + d1[:, :, np.newaxis] * gram_schmidt(quiver)[0] + d2[:, :, np.newaxis] * gram_schmidt(quiver)[1]
surfaces_data=[
    (surface1, 'brown', 'Surface 7'),
    (adapted_frame,'blue','tangent'),
]


boot_emeans=[]
replicates=[]
# boot_data.append(e_mean)  #mu_e(Qn)
book=PGD_init(X12, X17)
for i in range(len(Ts)):
    [t1,t2] = Ts[i]
    replicate = data_replicates[i]
    if data['a'][i]==15:
        boot_emeans.append(get_parametrized_surface(t1,t2,book,flag_embed=1))
        replicates.append(replicate)
#Bootstrapped mu_e_B(Qn)s

book=PGD_init(X12, X17)
tan_vecs=[]
t_e=[] #Tangent elements, 999*2
basis = gram_schmidt(quiver)

#Extract tangent elements of the bootsrapped extrinsic means, vecs returns the 3d projected vectors.
for b_e_mean in boot_emeans:
    vec=b_e_mean - e_mean
    tan_vecs.append(e_mean.flatten() + (vec.T @ basis[0]) * basis[0]+ (vec.T @ basis[1]) * basis[1])
    t_e.append([vec.T @ basis[0],vec.T @ basis[1]])
book['data']=np.array(tan_vecs)
#plot_surface_plotly(surfaces_data=surfaces_data,book=book,traces=None, flag_x_bar=0,flag_multi_traces=0,flag_embed=1,flag_pgd_trace=0,flag_plot_data=1,flag_axis_equal=1,data_title='tagent',quiver=gram_schmidt(quiver),e_mean=e_mean)
#asserting plot 1 above

# Compute nonpivotal bootstrap statistics
np_stats=[]
n=20
for i in range(len(boot_emeans)):
    np_stats.append(n*np.linalg.norm(t_e[i])**2)
plt.hist(np_stats,bins=20)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Non-pivotal Bootstrap Statistics')
plt.show()
lower = 0
upper = np.percentile(np_stats, 97.5)

print(f"95% CI: [{lower:.4f}, {upper:.4f}]")


def draw_circle_on_plane(center, v1, v2, radius=1.0, n_points=100):
    """
    center : array-like, shape (3,)
    v1, v2 : array-like, shape (3,) - orthonormal basis of the plane
    radius : float
    returns : array of shape (n_points, 3)
    """
    theta = np.linspace(0, 2 * np.pi, n_points)  # (n_points,)

    # Each of shape (n_points, 3)
    points = (center
              + radius * np.cos(theta)[:, None] * v1
              + radius * np.sin(theta)[:, None] * v2)

    return points
adapted_circle=draw_circle_on_plane(e_mean.flatten(),basis[0],basis[1],radius=np.sqrt(upper/20))
# plot_surface_plotly(surfaces_data=surfaces_data,book=book,traces=None, flag_x_bar=0,flag_multi_traces=0,flag_embed=1,flag_pgd_trace=0,flag_plot_data=1,flag_axis_equal=1,data_title='tagent',quiver=gram_schmidt(quiver),e_mean=e_mean,conf_region=adapted_circle)
#asserting plot 1 above
#0.0718
#
# def solve_eclipse():
#     #Project all the bootsrapped extrinsic sample means on to the tangent plane centered at the extrinsic sample mean.
#     #solve for the 95 quantile of the hotelling statistics.
#     pass

#3/18 saving stuffs for calculating gradient on colab, due to difficulty installing Pytorch.
# Xa=X12
# Xb=X17
# basis = gram_schmidt([Xa + Xb, Xa])
# basis = gram_schmidt([basis[0] + basis[1], Xa + Xb])
# [e1, e2] = basis
# E0 = E0.reshape(3, 3)
# eigenvalues, eigenvectors = np.linalg.eigh(E0)
# [v1, v2, v3] = [eigenvectors[:, 0], eigenvectors[:, 1], eigenvectors[:, 2]]
# [w1, w2, w3] = eigenvalues[0:3]
#
# r1 = w1 * np.dot(e1, v1) ** 2 + w2 * np.dot(e1, v2) ** 2 + w3 * np.dot(e1, v3) ** 2
# r2 = w1 * np.dot(e2, v1) ** 2 + w2 * np.dot(e2, v2) ** 2 + w3 * np.dot(e2, v3) ** 2
# r3 = w1 * np.dot(e1, v1) * np.dot(e2, v1) + w2 * np.dot(e1, v2) * np.dot(e2, v2) + w3 * np.dot(e1, v3) * np.dot(e2, v3)
# print([w1, w2, w3])


# np.savez('my_arrays.npz', E0=E0, basis=np.asarray(basis), rs=[r1,r2,r3])
# a


#3/19: pivotal bootsrap
J_loaded = np.load('jacobian.npy')


b1=basis[0].reshape(9,-1)
b2=basis[1].reshape(9,-1)
bs=np.hstack([b1,b2])
temp=J_loaded @ bs
l=temp.T
sigma_0=np.cov(vec_datas.squeeze().T)
s_0=l @ sigma_0 @ l.T
from scipy.stats import chi2
# c=np.sqrt(chi2.ppf(0.95, df=2))
c=4
eigenvalues, eigenvectors = np.linalg.eigh(s_0)
[v1, v2] = [eigenvectors[:, 0], eigenvectors[:, 1]]
[w1, w2] = eigenvalues[0:2]

n=20

extension1=(np.sqrt(w1)*c/np.sqrt(n))*(v1[0]*b1.squeeze()+v1[1]*b2.squeeze())
extension2=(np.sqrt(w2)*c/np.sqrt(n))*(v2[0]*b1.squeeze()+v2[1]*b2.squeeze())
t = np.linspace(0, 2 * np.pi, 300)
ellipse_9d1 = e_mean.flatten() + (np.sqrt(w1)*c/np.sqrt(n)) * np.cos(t)[:, None] * (v1[0]*b1.squeeze()+v1[1]*b2.squeeze()) + (np.sqrt(w2)*c/np.sqrt(n)) * np.sin(t)[:, None] * (v2[0]*b1.squeeze()+v2[1]*b2.squeeze())

c=np.sqrt(chi2.ppf(0.95, df=2))
eigenvalues, eigenvectors = np.linalg.eigh(s_0)
[v1, v2] = [eigenvectors[:, 0], eigenvectors[:, 1]]
[w1, w2] = eigenvalues[0:2]

n=20

extension1=(np.sqrt(w1)*c/np.sqrt(n))*(v1[0]*b1.squeeze()+v1[1]*b2.squeeze())
extension2=(np.sqrt(w2)*c/np.sqrt(n))*(v2[0]*b1.squeeze()+v2[1]*b2.squeeze())
t = np.linspace(0, 2 * np.pi, 300)
ellipse_9d2 = e_mean.flatten() + (np.sqrt(w1)*c/np.sqrt(n)) * np.cos(t)[:, None] * (v1[0]*b1.squeeze()+v1[1]*b2.squeeze()) + (np.sqrt(w2)*c/np.sqrt(n)) * np.sin(t)[:, None] * (v2[0]*b1.squeeze()+v2[1]*b2.squeeze())

plot_surface_plotly(surfaces_data=surfaces_data,book=book,traces=None, flag_x_bar=0,flag_multi_traces=0,flag_embed=1,flag_pgd_trace=0,flag_plot_data=1,flag_axis_equal=1,data_title='tagent',quiver=basis,e_mean=e_mean,conf_region=[adapted_circle,ellipse_9d1,ellipse_9d2])
np.save('emean.npy', e_mean)
#Need to do an eigen-decomposition for s_0

# Hotel_stats=[]
