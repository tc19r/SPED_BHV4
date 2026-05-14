#Designed to test whether the Euclidean Mean is focal, on one embedded surface
#A numerical test selecting 50 different initialization points.
#2/25/26:
#Testing on yeast dataset.
#3/2/26：
#Merging focality and project.
# traces=[PGD(X14,X19,E0,inits=inits) for inits in initpoints ]
# plot_trace(traces, mode='joint', labels=None)
#3/5/26:
#Turning this into a specialized solver.

import numpy as np
import matplotlib.pyplot as plt
import json
#Edge points
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

#Data generator
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

n_samples = 100

def simu_data():
    # x = np.random.uniform(0, 1, n_samples)
    # y = np.random.uniform(0, 1, n_samples)
    # coords=np.vstack([x,y])
    # e1=X14.flatten()
    # e2=X15.flatten()
    # f=np.matrix([e1, e2 ,np.cross(e1,e2)]).T
    # coords=np.vstack([coords,np.zeros(n_samples)])
    # z_t= np.matmul(f,coords)
    #
    # x = np.random.uniform(0, 1, n_samples)
    # y = np.random.uniform(0, 1, n_samples)
    # coords=np.vstack([x,y])
    # e1=X13.flatten()
    # e2=X14.flatten()
    # f=np.matrix([e1, e2 ,np.cross(e1,e2)]).T
    # coords=np.vstack([coords,np.zeros(n_samples)])
    # z_t= np.hstack([z_t, np.matmul(f,coords)])
    #
    # x = np.random.uniform(0, 1, n_samples)
    # y = np.random.uniform(0, 1, n_samples)
    # coords=np.vstack([x,y])
    # e1=X15.flatten()
    # e2=X11.flatten()
    # f=np.matrix([e1, e2 ,np.cross(e1,e2)]).T
    # coords=np.vstack([coords,np.zeros(n_samples)])
    # z_t= np.hstack([z_t, np.matmul(f,coords)])
    # # # # # #
    # x = np.random.uniform(0, 1, n_samples)
    # y = np.random.uniform(0, 1, n_samples)
    # coords=np.vstack([x,y])
    # e1=X13.flatten()
    # e2=X12.flatten()
    # f=np.matrix([e1, e2 ,np.cross(e1,e2)]).T
    # coords=np.vstack([coords,np.zeros(n_samples)])
    # z_t= np.hstack([z_t, np.matmul(f,coords)])
    #
    # x = np.random.uniform(0, 1, n_samples)
    # y = np.random.uniform(0, 1, n_samples)
    # coords=np.vstack([x,y])
    # e1=X12.flatten()
    # e2=X11.flatten()
    # f=np.matrix([e1, e2 ,np.cross(e1,e2)]).T
    # coords=np.vstack([coords,np.zeros(n_samples)])
    # z_t= np.hstack([z_t, np.matmul(f,coords)])
    #
    pass

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


#2/25/26: Testing on yeast data
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

datas=data_embedding(z_t)
embedded_coords=[datas[i][[0,1,1],[0,0,1]] for i in range(len(datas))]  # This is a list, should not be passed.
vec_datas=[datas[i].reshape(-1,1) for i in range(len(datas))]
vec_datas=np.asarray(vec_datas) # Use this
E0=np.mean(vec_datas,axis=0)
book={}
book['data']=z_t.T    # D


def Surface_sym_re(X1, X2, t1, t2):
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
    surface1 = surface1[:, :, [0, 1, 1], [0, 0, 1]]

    return [surface1]


#Modifying PDG with initalization

def get_parametrized_surface(t1, t2, book ,flag_embed):
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



def PGD_init(X1, X2,inits=None):

    M1 = X1 @ X1.T
    M2 = X2 @ X2.T
    M3 = X1 @ X2.T + X2 @ X1.T
    book = {
        'X1':X1,
        'X2':X2,
        'M1': M1,
        'M2': M2,
        'M3': M3,
        't0s': inits
    }
    return book


def PGD(X1, X2, E0,inits=None):
    # This function solve the PGD, and return a trace of t_1 and t_2 as T by 2 arrays.
    # T is the number of samples, here we set it to 10
    # The trace of loss will also be returned.
    max_iter = 3000 #200 times the iteration number needed for the magnified dataset
    grad_tol = 1e-8
    param_tol = 1e-10

    lr = 0.03
    book = PGD_init(X1, X2,inits)
    # lr = book['lr']
    # K = book['K']
    M1 = book['M1'].reshape(-1, 1)
    M2 = book['M2'].reshape(-1, 1)
    M3 = book['M3'].reshape(-1, 1)
    t1, t2 = book['t0s']
    trace = []

    for k in range(max_iter):
        surface = get_parametrized_surface(t1,t2,book,flag_embed=1)
        diff = surface - E0
        loss = np.linalg.norm(diff) ** 2
        grad_1 = (4 * t1 * M1 + 2 * t2 * M3).T @ diff
        grad_2 = (4 * t2 * M2 + 2 * t1 * M3).T @ diff

        # Store current position
        t1_old, t2_old = t1, t2

        t1 = t1 - lr * grad_1.item()
        t2 = t2 - lr * grad_2.item()

        # Compute gradient norm
        grad_norm = np.sqrt(grad_1.item() ** 2 + grad_2.item() ** 2)


        if t1 < 0:
            t1 = 0
        if t1 > 1:
            t1 = 1
        if t2 < 0:
            t2 = 0
        if t2 > 1:
            t2 = 1

        # Compute parameter change
        param_change = np.sqrt((t1 - t1_old)**2 + (t2 - t2_old)**2)

        if k % 10 == 0:
            trace.append([t1, t2, loss])
        # Check convergence
        if grad_norm < grad_tol and param_change < param_tol:
            break
        trace.insert(0, [inits[0],inits[1],trace[0][2]])
    return trace

def Newton(X1, X2, E0,inits=[0.5,0.5]):
    max_iter = 10
    param_tol = 1e-10

    book = PGD_init(X1, X2,inits)
    M1 = book['M1'].reshape(-1, 1)
    M2 = book['M2'].reshape(-1, 1)
    M3 = book['M3'].reshape(-1, 1)
    t1, t2 = book['t0s']
    trace = []

    for k in range(max_iter):
        surface = get_parametrized_surface(t1,t2,book,flag_embed=1)
        d = surface - E0
        loss = np.linalg.norm(d) ** 2
        df_1=2 * t1 * M1 +   t2 * M3
        df_2=2 * t2 * M2 +   t1 * M3


        df11= 2* ( 2*M1.T @ d   + np.linalg.norm(df_1)**2 ).item()
        df12= 2* ( M3.T @ d + df_1.T @ df_2 ).item()
        df21= df12
        df22= 2* ( 2*M2.T @ d + np.linalg.norm(df_2)**2 ).item()

        hessian=np.array([[df11,df12],
                [df21,df22]])

        # grad_1 = (4 * t1 * M1 + 2 * t2 * M3).T @ d
        # grad_2 = (4 * t2 * M2 + 2 * t1 * M3).T @ d

        # Store current position
        t1_old, t2_old = t1, t2

        grad_1 = (4 * t1 * M1 + 2 * t2 * M3).T @ d
        grad_2 = (4 * t2 * M2 + 2 * t1 * M3).T @ d
        delta= np.array([[grad_1.item()],
                [grad_2.item()]])
        increments=np.linalg.inv(hessian) @ delta

        t1 = t1 - increments[0,0]
        t2 = t2 - increments[1,0]

        # Compute gradient norm


        if t1 < 0:
            t1 = 0
        if t1 > 1:
            t1 = 1
        if t2 < 0:
            t2 = 0
        if t2 > 1:
            t2 = 1

        # Compute parameter change
        param_change = np.sqrt((t1 - t1_old)**2 + (t2 - t2_old)**2)

        if k % 10 != 0 :
            trace.append([t1, t2, loss, param_change])
        # Check convergence
        if  param_change < param_tol:
            trace.append([t1, t2, loss, param_change])
            break
        # trace.append([t1, t2, loss, k, grad_norm, param_change])
    trace.insert(0, [inits[0], inits[1], trace[0][2], trace[0][3]])
    print('loss at large'+ str(trace[-1][2]))
    surface = get_parametrized_surface(0, 0, book, flag_embed=1)
    d = surface - E0
    loss0 = np.linalg.norm(d) ** 2
    print('loss at origin'+ str(loss0))
    return trace

def create_grid(n_points=10):
    """
    Create a uniform grid in [0,1] x [0,1]

    Parameters:
    - n_points: number of points along each dimension

    Returns:
    - grid_points: array of shape (n_points^2, 2)
    """
    x = np.linspace(0.001, 0.5, n_points)
    y = np.linspace(0.001, 0.5, n_points)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])
    return grid_points
initpoints=create_grid(n_points=5)
traces=[PGD(X20,X15,E0,inits=inits) for inits in initpoints ]
#traces=[Newton5X12,X17,E0,inits=inits) for inits in initpoints ]
#Plot traces.

def plot_trace(traces, mode='single', labels=None,num_aux=0,save_plot=False,title_add=None):
    """
    Parameters:
    -----------
    traces : list or single trace
        - If mode='single': a single trace (list of [t1, t2, loss])
        - If mode='joint': a list of traces
    mode : str
        - 'single': original behavior, plots parameter path + loss for one trace
        - 'joint': plots only the loss curves for all traces together
    labels : list of str (optional)
        Labels for each trace in joint mode. Defaults to ['Trace 1', 'Trace 2', ...]
    """

    # Soft, muted colors for joint mode
    joint_colors = [
    '#4a90d9',      # soft blue (Surface 1)
    '#5cb85c',      # soft green (Surface 2)
    '#d9534f',      # soft red (Surface 3)
    '#5bc0de',      # soft cyan (Surface 4)
    '#f0ad4e',      # soft yellow (Surface 5)
    '#b98ec4',      # soft magenta (Surface 6)
    '#a67c52',      # soft brown (Surface 7)
    '#9c9c5c',      # soft olive (Surface 8)
    '#a3d977',      # soft lime (Surface 9)
    '#d4a190',      # soft peach (Surface 10)
    '#95a5a6',      # soft gray (Surface 11)
    '#e88c5d',      # soft orange (Surface 12)
    '#5dab9c',      # soft teal (Surface 15)
    '#d4a574',      # soft burlywood (Surface 13)
    '#e89cbb',      # soft pink (Surface 14)
]

    if mode == 'single':
        res = np.array(traces)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # --- Subplot 1: Parameter Trajectory ---
        ax1.plot(res[:, 0], res[:, 1], color='royalblue', marker='o', markersize=3, alpha=0.6,
                 label='Optimization Path')
        ax1.scatter(res[0, 0], res[0, 1], color='green', s=100, label='Start', zorder=5)
        ax1.scatter(res[-1, 0], res[-1, 1], color='red', s=100, label='End', zorder=5)

        ax1.set_xlim(-0.05, 1.05)
        ax1.set_ylim(-0.05, 1.05)
        ax1.set_xlabel('$t_1$')
        ax1.set_ylabel('$t_2$')
        ax1.set_title('Parameter Path (Projected onto [0, 1])')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7)

        # --- Subplot 2: Loss Convergence ---
        ax2.plot(res[:, 2], color='red', linewidth=2)
        ax2.set_yscale('log')
        ax2.set_xlabel('Checkpoint (Every 10th Iteration)')
        ax2.set_ylabel('Loss (Log Scale)')
        ax2.set_title('Log-scale Loss Reduction on each surface')
        ax2.grid(True, which="both", linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.show()

    elif mode == 'joint':
        if labels is None:
            labels = [f'Trace {i + 1}' for i in range(len(traces))]

        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))

        for i, trace in enumerate(traces):
            if i<= len(traces)-1-num_aux:
                res = np.array(trace)
                color = 'yellow' #joint_colors[i % len(joint_colors)]
                #ax2.plot(res[:, 2], color=color, linewidth=2, label=None)
                ax1.plot(res[:, 0], res[:, 1], 'b-', alpha=0.2, linewidth=0.5)
                ax1.scatter(res[0, 0], res[0, 1], color='green', s=50, alpha=0.6, zorder=5)
                ax1.scatter(res[-1, 0 ], res[-1, 1], color='red', s=50, alpha=0.6, zorder=5)
            else:
                res = np.array(trace)
                ax1.scatter(res[-0], res[1], color='blue', s=30, alpha=0.6, zorder=5)
        ax1.set_xlim(-0.05, 1.05)
        ax1.set_ylim(-0.05, 1.05)
        ax1.set_xlabel('$t_1$')
        ax1.set_ylabel('$t_2$')
        # ax1.set_title('Parameter Paths for All Initial Points')
        ax1.set_title(title_add)
        ax1.grid(True, linestyle='--', alpha=0.7)
        # ax2.set_yscale('linear')
        # ax2.set_xlabel('Checkpoint (Every 10th Iteration)')
        # ax2.set_ylabel('Loss (Log Scale)')
        # ax2.set_title('Log-scale Loss Reduction')
        # ax2.legend()
        # ax2.grid(True, which="both", linestyle='--', alpha=0.5)

        plt.tight_layout()
        if save_plot:
            plt.savefig('./Figures/(mag=1) Parameter Paths for All Initial Points'+title_add+'.png')
        # plt.show()
plot_trace(traces, mode='joint', labels=None, num_aux=0, save_plot=False, title_add=None)

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
    #Returns two pairs of times if r3<0. Return 1 otherwise.
    # basis=gram_schmidt([Xa,Xb])

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
        # basis=gram_schmidt([Xa,Xb])
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

def load_data():
    #This function load the bootstrapped data and compute standard deviation and extrinsic means.
    pass

edge_points = [
    (X15, X11),
    (X14, X15),
    (X14, X13),
    (X13, X12),
    (X11, X12),
    (X20, X15),
    (X17, X9),
    (X17, X10),
    (X14, X19),
    (X11, X16),
    (X8, X19),
    (X18, X8),
    (X18, X13),
    (X18, X9),
    (X17, X12),
]

for id,edge in enumerate(edge_points):
    traces = [PGD(edge[0], edge[1], E0, inits=inits) for inits in initpoints]
    projs = project(edge[0], edge[1], E0)
    num_aux = 1
    traces.append([projs[0], projs[1], 0, 0])
    title_add=' Surface' +  str(id+1) +','+'kmax='+str(projs[2])
    plot_trace(traces, mode='joint', labels=None, num_aux=num_aux, save_plot=True, title_add=title_add)
