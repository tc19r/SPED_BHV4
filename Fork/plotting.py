import numpy as np
import plotly.graph_objects as go

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

def plot_surface_plotly(
    surfaces_data=None,
    book=None,
    flag_x_bar=0,
    flag_pgd_trace=0,
    M1=None, M2=None, M3=None,
    traces=None,
    flag_surf_one=1,
    flag_multi_traces=0,
    flag_embed=0,
    flag_plot_data=1,
    data_title="Uniform sampling in the 'blue' Surface",
    flag_axis_equal=1,
    zrange=None,
    xrange=None,
    yrange=None,
    quiver=None,
    e_mean=None,
    conf_region=None,
):
    """
    Plotly refactor of the matplotlib plot_surface function.
    Supports embedded and non-embedded surfaces, data scatter, PGD traces, quiver arrows.
    """

    def surface_parametrization(t1, t2, book, flag_embed):
        surface = get_parametrized_surface(t1, t2, book, flag_embed)
        if flag_embed == 1:
            return surface[4], surface[7], surface[8]
        return surface[0], surface[1], surface[2]

    def get_xyz(surf, flag_embed):
        """Extract X, Y, Z arrays from a surface array depending on embed flag."""
        if flag_embed == 0:
            return surf[:, :, 0], surf[:, :, 1], surf[:, :, 2]
        if surf.ndim == 4:
            surf = surf.reshape((*surf.shape[:2], 9))
            return surf[:, :, 4], surf[:, :, 7], surf[:, :, 8]
        if surf.ndim == 3 and surf.shape[2] > 3:
            # alpha=0.0 case in original — still add but fully transparent
            return surf[:, :, 4], surf[:, :, 7], surf[:, :, 8]
        return surf[:, :, 0], surf[:, :, 1], surf[:, :, 2]

    fig = go.Figure()

    # ── Surfaces ────────────────────────────────────────────────────────────────
    for surf, color, title in surfaces_data:
        x, y, z = get_xyz(surf, flag_embed)

        # alpha=0.0 edge case: make invisible but keep geometry
        opacity = 0.35 if (flag_embed == 1 and surf.ndim == 3 and surf.shape[2] > 3) else 0.35

        fig.add_trace(
            go.Surface(
                x=x, y=y, z=z,
                colorscale=[[0, color], [1, color]],
                showscale=False,
                opacity=opacity,
                contours=dict(
                    x=dict(show=True, color="lightgray", width=1),
                    y=dict(show=True, color="lightgray", width=1),
                ),
                name=title,
                hovertemplate="x: %{x:.3f}<br>y: %{y:.3f}<br>z: %{z:.3f}<extra>" + title + "</extra>",
            )
        )

    # ── Origin ───────────────────────────────────────────────────────────────────
    fig.add_trace(
        go.Scatter3d(
            x=[0], y=[0], z=[0],
            mode="markers",
            marker=dict(size=5, color="red"),
            name="Origin",
            showlegend=False,
        )
    )

    # ── Data points ──────────────────────────────────────────────────────────────
    if book and flag_plot_data == 1:
        data = book["data"]
        if flag_embed == 0:
            px = [data[i, 0] for i in range(len(data))]
            py = [data[i, 1] for i in range(len(data))]
            pz = [data[i, 2] for i in range(len(data))]
        else:
            px = [data[i].reshape(-1, 1)[4].item() for i in range(len(data))]
            py = [data[i].reshape(-1, 1)[7].item() for i in range(len(data))]
            pz = [data[i].reshape(-1, 1)[8].item() for i in range(len(data))]

        fig.add_trace(
            go.Scatter3d(
                x=px, y=py, z=pz,
                mode="markers",
                marker=dict(size=3, color="gray", opacity=0.7),
                name="Data",
            )
        )

    # ── Euclidean mean (x_bar) ───────────────────────────────────────────────────
    if flag_x_bar:
        tmp = np.asarray(book["data"])
        x_bar = np.mean(tmp, axis=0)
        fig.add_trace(
            go.Scatter3d(
                x=[x_bar[0]], y=[x_bar[1]], z=[x_bar[2]],
                mode="markers",
                marker=dict(size=10, color="red", symbol="circle"),
                name="x_bar",
            )
        )

    # ── PGD traces ───────────────────────────────────────────────────────────────
    if flag_pgd_trace and traces:
        for idx, trace in enumerate(traces):
            if flag_multi_traces == 0 and idx >= 1:
                break

            path_3d = []
            for t1, t2, _loss in trace:
                x, y, z = surface_parametrization(t1, t2, book[idx], flag_embed)
                path_3d.append([x, y, z])
            path_3d = np.array(path_3d)

            fig.add_trace(
                go.Scatter3d(
                    x=path_3d[:, 0], y=path_3d[:, 1], z=path_3d[:, 2],
                    mode="lines",
                    line=dict(color="black", width=4),
                    name=f"PGD path {idx}",
                )
            )
            # Start point
            fig.add_trace(
                go.Scatter3d(
                    x=[path_3d[0, 0]], y=[path_3d[0, 1]], z=[path_3d[0, 2]],
                    mode="markers",
                    marker=dict(size=6, color="yellow"),
                    name=f"Start {idx}",
                    showlegend=False,
                )
            )
            # End point
            fig.add_trace(
                go.Scatter3d(
                    x=[path_3d[-1, 0]], y=[path_3d[-1, 1]], z=[path_3d[-1, 2]],
                    mode="markers",
                    marker=dict(size=9, color="cyan", symbol="diamond"),
                    name=f"End {idx}",
                    showlegend=False,
                )
            )

    # ── Quiver (tangent vectors) ──────────────────────────────────────────────────
    if quiver and e_mean is not None:
        ox = e_mean.reshape(-1, 1)[4].item()
        oy = e_mean.reshape(-1, 1)[7].item()
        oz = e_mean.reshape(-1, 1)[8].item()

        for vec, color, label in zip(quiver, ["red", "blue"], ["v1", "v2"]):
            dx = vec.reshape(-1, 1)[4].item()
            dy = vec.reshape(-1, 1)[7].item()
            dz = vec.reshape(-1, 1)[8].item()

            # Plotly has no native quiver in 3D; draw as a cone + line segment
            fig.add_trace(
                go.Scatter3d(
                    x=[ox, ox + dx], y=[oy, oy + dy], z=[oz, oz + dz],
                    mode="lines",
                    line=dict(color=color, width=4),
                    name=label,
                )
            )
            # fig.add_trace(
            #     go.Cone(
            #         x=[ox + dx], y=[oy + dy], z=[oz + dz],
            #         u=[dx], v=[dy], w=[dz],
            #         colorscale=[[0, color], [1, color]],
            #         showscale=False,
            #         sizemode="absolute",
            #         sizeref=0.1,
            #         name=f"{label} tip",
            #         showlegend=False,
            #     )
            # )
    # ── Conf_region (confidence region) ──────────────────────────────────────────────────
    if conf_region is not None :
        colors=['#E8A838', '#E05C2A','#C4273A']
        for id,conf in enumerate(conf_region):
            fig.add_trace(go.Scatter3d(
                x=conf[:, 4],
                y=conf[:, 7],
                z=conf[:, 8],
                mode='lines',
                line=dict(
                    color=colors[id],  # Named color
                    # color='#FF5733',   # Hex color
                    # color='rgb(255, 87, 51)',  # RGB
                    # color='rgba(255, 87, 51, 0.8)',  # RGBA with opacity
                    width=3  # Optional: line thickness
                )
            ))

    # ── Layout ───────────────────────────────────────────────────────────────────
    scene = dict(
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Z",
        aspectmode="cube" if flag_axis_equal else "auto",
    )
    if xrange:
        scene["xaxis"] = dict(range=xrange)
    if yrange:
        scene["yaxis"] = dict(range=yrange)
    if zrange:
        scene["zaxis"] = dict(range=zrange)

    fig.update_layout(
        title=data_title,
        scene=scene,
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    fig.show()
    return fig