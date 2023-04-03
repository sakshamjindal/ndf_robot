import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from PIL import Image
from io import BytesIO

msz = 1.5
op = 1.0
cscale = 'Viridis'
black_marker = {
        'size': msz,
        'color': 'black',
        'colorscale': cscale, 
        'opacity': op 
        }
blue_marker = {
        'size': msz,
        'color': 'blue',
        'colorscale': cscale, 
        'opacity': op
        }
red_marker = {
        'size': msz,
        'color': 'red',
        'colorscale': cscale, 
        'opacity': op
        }
purple_marker = {
        'size': msz,
        'color': 'purple',
        'colorscale': cscale,
        'opacity': op
        }
green_marker = {
        'size': msz,
        'color': 'green',
        'colorscale': cscale,
        'opacity': op
        }
orange_marker = {
        'size': msz,
        'color': 'orange',
        'colorscale': cscale,
        'opacity': op
        }

marker_dict = {
        'black': black_marker,
        'blue': blue_marker,
        'red': red_marker,
        'purple': purple_marker,
        'orange': orange_marker,
        'green': green_marker
        }

def plot3d(pts_list, colors=['black'], fname='default_3d.html', 
           auto_scene=False, scene_dict=None, z_plane=True, write=True,
           extra_data=None):
    '''
    Function to create a 3D scatter plot in plotly

    Args:
        pts_list (list): list of numpy arrays, each containing a separate point cloud
        colors (list): list of color names corresponding to each point cloud in pts. If this is
            not a list, or there's only one element in the list, we will assume to use the 
            specified colors for each point cloud
        fname (str): name of file to save
        auto_scene (bool): If true, let plotly autoconfigure the scene camera / boundaries / etc.
        scene_dict (dict): If we include this, this contains the scene parameters we want. If this
            is left as None, we have a default scene setting used within the function. Expects
            keys '
        z_plane (bool): If True, then a gray horizontal plane will be drawn below all the point clouds
        write (bool): If True, then html file with plot will be saved
        extra_data (list): Additional plotly data that we might want to plot, which is created externally
    '''
    fig_data = []
    if not isinstance(pts_list, list):
        pts_list = [pts_list]
    if not isinstance(colors, list):
        colors = [colors]
    if len(colors) == 1:
        colors = colors * len(pts_list)

    all_pts = np.concatenate(pts_list, axis=0)

    for i, pts in enumerate(pts_list):
        pcd_data = {
                'type': 'scatter3d',
                'x': pts[:, 0],
                'y': pts[:, 1],
                'z': pts[:, 2],
                'mode': 'markers',
                'marker': marker_dict[colors[i]]}
        fig_data.append(pcd_data)

    z_height = min(all_pts[:, 2])
    plane_data = {
       'type': 'mesh3d',
       'x': [-1, 1, 1, -1],
       'y': [-1, -1, 1, 1],
       'z': [z_height]*4,
       'color': 'gray',
       'opacity': 0.5,
       'delaunayaxis': 'z'}
    
    if z_plane:
        fig_data.append(plane_data)

    if extra_data is not None:
        fig_data = fig_data + extra_data
    fig = go.Figure(data=fig_data)

    default_camera = {
        'up': {'x': 0, 'y': 0,'z': 1},
        'center': {'x': 0.45, 'y': 0, 'z': 0.0},
        'eye': {'x': -1.0, 'y': 0.0, 'z': 0.01}
    }
    default_scene = {
        'xaxis': {'nticks': 10, 'range': [-0.1, 0.9]},
        'yaxis': {'nticks': 16, 'range': [-0.5, 0.5]},
        'zaxis': {'nticks': 8, 'range': [-0.01, 1.5]}
    }
    default_width = 1100
    default_margin = {'r': 20, 'l': 10, 'b': 10, 't': 10}
    default_scene_dict = dict(
        scene=default_scene,
        camera=default_camera,
        width=default_width,
        margin=default_margin
    )

    if scene_dict is None:
        scene_dict = default_scene_dict
    else:
        for key in default_scene_dict.keys():
            if key not in scene_dict.keys():
                scene_dict[key] = default_scene_dict[key]

    if not auto_scene:
        fig.update_layout(
            scene=scene_dict['scene'],
            scene_camera=scene_dict['camera'],
            width=scene_dict['width'],
            margin=scene_dict['margin']
        )

    #png_renderer = pio.renderers['png']
    #png_renderer.width = 500
    #png_renderer.height = 500
    #pio.renderers.default = 'png'

    if write:
        #fig.show()
        if fname.endswith('html'):
            fig.write_html(fname)
        else:
            fig.write_image(fname)
    return fig

def plot_point_clouds(points, colors, extra_data=None, size = 1, opacity = 0.8, transform=None, show=False):

    """
    points: list of Nx3 array
    colors: list of colors (strings)
    extra_data: list of plotly data
    """

    if not isinstance(points, list):
        points = [points]
    if not isinstance(colors, list):
        colors = [colors]

    import plotly.graph_objects as go

    data = []
    for point, color in zip(points, colors):
        data.append(go.Scatter3d(
            x=point[:, 0], y=point[:, 1], z=point[:, 2],
            mode='markers',
            marker=dict(
                size=1,
                color=color,                
                opacity=0.8
            )
        ))

    if extra_data is not None:
        data += extra_data

    if transform is not None:
        local_frame =  PlotlyVisualizer().plotly_create_local_frame(transform)
        data += local_frame
        
    fig = go.Figure(
        data=data,
        layout=dict(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False)
            )
        )
    )

    if show:
        fig.show()

    return fig

class PlotlyVisualizer():
    def __init__(self):
        pass

    def _cam_frame_scene_dict(self):
        self.cam_frame_scene_dict = {}
        cam_up_vec = [0, 1, 0]
        plotly_camera = {
            'up': {'x': cam_up_vec[0], 'y': cam_up_vec[1],'z': cam_up_vec[2]},
            'center': {'x': 0, 'y': 0, 'z': 0},
            'eye': {'x': -0.6, 'y': -0.6, 'z': 0.4},
        }

        plotly_scene = {
            'xaxis': 
                {
                    'backgroundcolor': 'rgb(255, 255, 255)',
                    'gridcolor': 'white',
                    'zerolinecolor': 'white',
                    'tickcolor': 'rgb(255, 255, 255)',
                    'showticklabels': False,
                    'showbackground': False,
                    'showaxeslabels': False,
                    'visible': False,
                    'range': [-0.5, 0.5]},
            'yaxis': 
                {
                    'backgroundcolor': 'rgb(255, 255, 255)',
                    'gridcolor': 'white',
                    'zerolinecolor': 'white',
                    'tickcolor': 'rgb(255, 255, 255)',
                    'showticklabels': False,
                    'showbackground': False,
                    'showaxeslabels': False,
                    'visible': False,
                    'range': [-0.5, 0.5]},
            'zaxis': 
                {
                    'backgroundcolor': 'rgb(255, 255, 255)',
                    'gridcolor': 'white',
                    'zerolinecolor': 'white',
                    'tickcolor': 'rgb(255, 255, 255)',
                    'showticklabels': False,
                    'showbackground': False,
                    'showaxeslabels': False,
                    'visible': False,
                    'range': [-0.5, 0.5]},
        }
        self.cam_frame_scene_dict['camera'] = plotly_camera
        self.cam_frame_scene_dict['scene'] = plotly_scene

    @staticmethod
    def plotly_create_local_frame(transform=None, length=0.03, show = False):
        import plotly.graph_objects as go

        if transform is None:
            transform = np.eye(4)

        x_vec = transform[:-1, 0] * length
        y_vec = transform[:-1, 1] * length
        z_vec = transform[:-1, 2] * length

        origin = transform[:-1, -1]

        lw = 8
        x_data = go.Scatter3d(
            x=[origin[0], x_vec[0] + origin[0]], y=[origin[1], x_vec[1] + origin[1]], z=[origin[2], x_vec[2] + origin[2]],
            line=dict(
                color='red',
                width=lw
            ),
            marker=dict(
                size=0.0
            )
        )
        y_data = go.Scatter3d(
            x=[origin[0], y_vec[0] + origin[0]], y=[origin[1], y_vec[1] + origin[1]], z=[origin[2], y_vec[2] + origin[2]],
            line=dict(
                color='green',
                width=lw
            ),
            marker=dict(
                size=0.0
            )
        )
        z_data = go.Scatter3d(
            x=[origin[0], z_vec[0] + origin[0]], y=[origin[1], z_vec[1] + origin[1]], z=[origin[2], z_vec[2] + origin[2]],
            line=dict(
                color='blue',
                width=lw
            ),
            marker=dict(
                size=0.0
            )
        )
        if show:
            fig = go.Figure(data=[x_data, y_data, z_data])
            fig.show()

        data = [x_data, y_data, z_data]
        return data
    
    @staticmethod
    def plot3d(*args, **kwargs):
        return plot3d(*args, **kwargs)
    
    @staticmethod
    def plot_point_clouds(*args, **kwargs):
        return plot_point_clouds(*args, **kwargs)

