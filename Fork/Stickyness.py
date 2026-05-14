#A simulation study on an open book section
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import json

mplstyle.use('fast')

#Basic plotting utility complete. Working on including all 15 surfaces.

# 11/22
# 3/5/26:
# Corrected the edge ordering.
def Surface(X1, X2, t1, t2):
    # This function take two 3d points X1, X2,
    # and two graded parameters t1, t2,
    # to produce a relevant point cloud in 3d space.
    # Consistent to the second parametrization.
    m1 = X1
    m2 = X2

    c1 = np.array(t1)
    c2 = np.array(t2)

    surface1 = c1[:, :, np.newaxis] * m1.flatten() + c2[:, :, np.newaxis] * m2.flatten()

    return [surface1]

def Surface_sym(X1, X2, t1, t2):
    # This function take two 3d points X1, X2,
    # and two graded parameters t1, t2,
    # to produce a relevant point cloud in 6d, i.e. the space of the symmetric matrices.
    # The length of t2 and t2 must match.
    m1 = X1 @ X1.T
    m2 = X2 @ X2.T
    m3 = X1 @ X2.T + X2 @ X1.T
    c = (X1 + X2)
    m4 = -(X1 @ c.T + c @ X1.T)
    m5 = -(X2 @ c.T + c @ X2.T)

    c1 = np.array((t1 ** 2) * (1 - t2) ** 2)
    c2 = np.array((t1 ** 2) * (t2 ** 2))
    c3 = np.array((t1 ** 2) * (1 - t2) * t2)
    c4 = np.array(t1 * (1 - t2))
    c5 = np.array(t1 * t2)

    surface1 = c1[:, :, np.newaxis, np.newaxis] * m1 + c2[:, :, np.newaxis, np.newaxis] * m2 + c3[
        :, :, np.newaxis, np.newaxis] * m3
    surface2 = c @ c.T + c4[:, :, np.newaxis, np.newaxis] * m4 + c5[:, :, np.newaxis, np.newaxis] * m5 + surface1
    surface1 = surface1[:, :, [0, 1, 1], [0, 0, 1]]
    surface2 = surface2[:, :, [0, 1, 1], [0, 0, 1]]

    return [surface1, surface2]  # ,s1,s2,s3


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

    return [surface1]  # ,s1,s2,s3


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


def plot_surface(surf, books=None,flag_x_bar=0, flag_pgd_trace=0, M1=None, M2=None, M3=None,traces=None,flag_surf_one=1,flag_multi_traces=0,flag_embed=0,flag_plot_data=1,data_title='Uniform sampling in the \'blue\' Surface'):
    # Create a combined plot with all surfaces
    # The  data should be a dictionary. Let's say, it has data, extrinsic mean, and projection [10 traces].
    # The surf should probably be done inside of this function
    # optimize the parameter passing later
    # To enable multiple traces, books and traces are list of surface configurations and time traces. Note that this is independent of the surface data, to
    # help create independence of defining/passing the background surfaces and the traces.
    def surface_parametrization(t1, t2,book,flag_embed):
        surface = get_parametrized_surface(t1, t2, book, flag_embed)
        if flag_embed==1:
            c1 = surface[0]
            c2 = surface[3]
            c3 = surface[4]
        else:
            c1 = surface[0]
            c2 = surface[1]
            c3 = surface[2]
        return c1, c2, c3

    fig2 = plt.figure(figsize=(10, 8))
    ax_combined = fig2.add_subplot(111, projection='3d')

    # Plot all surfaces in one plot
    for surfs, color, title in surfaces_data:
        for idx, surf in enumerate(surfs):
            if idx == 1 and flag_surf_one==0:
                continue
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
        for i in range(data.shape[0]):
            ax_combined.scatter(data[i,0], data[i,1], data[i,2],color='grey')

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
            for t1, t2, loss,_ in trace:
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
    plt.axis('equal')
    plt.show()


def get_rho(t2, M1, M2, M3):
    rho_val = (1 - t2 )**2 * M1 + t2 ** 2 * M2 + (1 - t2) * t2 * M3
    return rho_val


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


# These are the parametrizing functions, can be used both by optimization and visualization.
# For convenience, the functions assume everything has been vectorized.
def PGD_init(X1, X2,inits=[0.5,0.5]):

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


def PGD(X1, X2, E0,K=100,inits=[0.5,0.5]):
    # This function solve the PGD, and return a trace of t_1 and t_2 as T by 2 arrays.
    # T is the number of samples, here we set it to 10
    # The trace of loss will also be returned.
    max_iter = 6000
    grad_tol = 1e-8
    param_tol = 1e-10

    lr = 0.01
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

        if k % 10 != 0:
            trace.append([t1, t2, loss, param_change])
        # Check convergence
        # if grad_norm < grad_tol and param_change < param_tol:
        #     trace.append([t1, t2, loss, param_change])
        #     break
        # trace.append([t1, t2, loss, k, grad_norm, param_change])
    return trace


def Newton(X1, X2, E0,inits=[0.5,0.5]):
    max_iter = 20
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


def plot_trace(traces, mode='joint', labels=None):
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

        fig, ax = plt.subplots(figsize=(9, 5))
        min_loss=1e3
        min_K=-1
        for i, trace in enumerate(traces):
            res = np.array(trace)
            color = joint_colors[i % len(joint_colors)]
            ax.plot(np.log(res[:, 2]), color=color, linewidth=2, label=labels[i])

        ax.set_yscale('linear')
        ax.set_xlabel('Checkpoint (Every 10th Iteration)')
        ax.set_ylabel('Loss (Log Scale)')
        ax.set_title('Log-scale Loss Reduction')
        ax.legend()
        ax.grid(True, which="both", linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()



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



#Task 1 : Display
#plot_surface(surfaces_data)

#Task 2: Find and display extrinsic mean for uniform data on one leaf.

#Sampling in an orthant

# n_samples = 100
# x = np.random.uniform(0, 1, n_samples)
# y = np.random.uniform(0, 1, n_samples)
#
# fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(6, 6))
#
# # Plotting the samples
# axs[0].scatter(x, y, alpha=0.5, s=10, edgecolor='k')
#
# # Set up the plot to show the unit square
# axs[0].set_title('Samples from Uniform[0,1] × Uniform[0,1]')
# axs[0].set_aspect('equal', adjustable='box')  # Make it square
#
# f=np.matrix([[0.9128, 0.4083], [0.4083, 0.9128]])
# coords=np.vstack([x,y])
# z_t= np.matmul(f,coords)
# # Plotting the samples
# axs[1].scatter(z_t[0,:].tolist(), z_t[1,:].tolist(), alpha=0.5, s=10, edgecolor='k')
#
# # Set up the plot to show the unit square
#
# axs[1].set_title('Samples in 2D with z = 0')
# axs[1].set_aspect('equal', adjustable='box')
#
# plt.show()

def data_gene():
    pass
    return 1


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


dlow=0
dhigh=1
low=0
high=1
#Exp 2 uni on 3 neighbors - openbook
n_samples = 100
x = np.random.uniform(dlow, dhigh, n_samples)
y = np.random.uniform(low, high, n_samples)
coords=np.vstack([x,y])
e1=X17.flatten()
e2=X12.flatten()
f=np.matrix([e1, e2 ,np.cross(e1,e2)]).T
coords=np.vstack([coords,np.zeros(n_samples)])
z_t= np.matmul(f,coords)

# # # #
x = np.random.uniform(0.5*dlow, 0.5*dhigh, n_samples)
y = np.random.uniform(low, high, n_samples)
coords=np.vstack([x,y])
e1=X13.flatten()
e2=X12.flatten()
f=np.matrix([e1, e2 ,np.cross(e1,e2)]).T
coords=np.vstack([coords,np.zeros(n_samples)])
z_t= np.hstack([z_t, np.matmul(f,coords)])

x = np.random.uniform(0.6*dlow, 0.6*dhigh, n_samples)
y = np.random.uniform(low, high, n_samples)
coords=np.vstack([x,y])
e1=X11.flatten()
e2=X12.flatten()
f=np.matrix([e1, e2 ,np.cross(e1,e2)]).T
coords=np.vstack([coords,np.zeros(n_samples)])
z_t= np.hstack([z_t, np.matmul(f,coords)])

#Exp3 - real data from yeast

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


datas=data_embedding(z_t)
embedded_coords=[datas[i][[0,1,1],[0,0,1]] for i in range(len(datas))]  # This is a list, should not be passed.
vec_datas=[datas[i].reshape(-1,1) for i in range(len(datas))]
vec_datas=np.asarray(vec_datas) # Use this
E0=np.mean(vec_datas,axis=0)
book={}
book['data']=z_t.T    # Due

#
u = np.linspace(0, 1, 20)
v = np.linspace(0, 1, 20)
t1, t2 = np.meshgrid(u, v)
surfaces_data=[
    (Surface(X11,X12,t1,t2), 'red', 'Surface 3'),
    (Surface(X12,X13,t1,t2), 'cyan', 'Surface 4'),
    (Surface(X12,X17,t1,t2), 'brown', 'Surface 7'),

]


#plot_surface(surfaces_data,book,flag_x_bar=0,data_title='Independent uniform samples on 5 neighbors')
K=3000
edge_points = [
    (X13, X12),
    (X11, X12),
    (X17, X12),
]

traces=[PGD(edges[0],edges[1],E0) for edges in edge_points]
books=[PGD_init(edges[0],edges[1]) for edges in edge_points]
data=[]
projs = [project(edges[0], edges[1], E0) for edges in edge_points]
# data.append(projs[0][0]*X13+projs[0][1]*X12)
# data.append(projs[1][0]*X11+projs[1][1]*X12)
# data.append(projs[2][0]*X17+projs[2][1]*X12)
data=np.asarray(data)
book={}
#book['data']=data.T
book['data']=z_t.T
plot_surface(surfaces_data,books,traces=traces, flag_x_bar=0,flag_multi_traces=1,flag_embed=0,flag_pgd_trace=0,flag_plot_data=1,data_title='Sample trees from yeast data')

# plot_trace(traces,mode='joint',labels = [
#     'blue surface',
#     'green surface',
#     'red surface',
#     'cyan surface',
#     'yellow surface',
#     'magenta surface',
#     'brown surface',
#     'olive surface',
#     'lime surface',
#     'gold surface',
#     'gray surface',
#     'orange surface',
#     'teal surface',
#     'burlywood surface',
#     'pink surface',
# ])

print(projs)
E0=E0.reshape(3,3)
eigenvalues, eigenvectors = np.linalg.eigh(E0)
print(eigenvalues)

#Bootstrapping projections. See how much extrinsic means falls on the 'longer' surface.
def get_min_surface(traces=None,lambdas=None):
    if traces:
        min_loss = 1e3
        min_K = -1
        for i, trace in enumerate(traces):
            res = np.array(trace)
            if res[-1, 2] < min_loss:
                min_loss = res[-1, 2]
                min_K = i
        return min_K, [traces[min_K][-1][0],traces[min_K][-1][1],traces[min_K][-1][3]]
    else:
        pass


dices=[]
paras=[]
data_replicates=[]


Monte_carlo=1000
for i in range(Monte_carlo):
    indices = np.random.choice(300, size=300, replace=True)
    X_boot = z_t[:, indices]
    datas = data_embedding(X_boot)
    vec_datas = [datas[i].reshape(-1, 1) for i in range(len(datas))]
    vec_datas = np.asarray(vec_datas)  # Use this
    E0 = np.mean(vec_datas, axis=0)
    k=0
    lam_max=0
    t_m1=0
    t_m2=0
    for id, edge in enumerate(edge_points):
        projs = project(edge[0], edge[1], E0)
        [t1,t2,lam]=projs
        if lam>lam_max:
           lam_max=lam
           k=id
           t_m1=t1
           t_m2=t2
    k=k+1
    para=[t_m1,t_m2]
    dices.append(k)
    paras.append(para)
    data_replicates.append(vec_datas.tolist())
from collections import Counter
counter = Counter(dices)
print(counter)

booted_means=[]
for i in range(len(paras)):
    booted_means.append(paras[i][0]*X17+paras[i][1]*X12)
book['data']=np.asarray(booted_means)
plot_surface(surfaces_data,books,traces=traces, flag_x_bar=0,flag_multi_traces=1,flag_embed=0,flag_pgd_trace=0,flag_plot_data=1,data_title='Sample trees from yeast data')
