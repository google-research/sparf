"""
 Copyright 2022 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

import numpy as np
import torch
import json 
from typing import List, Any, Tuple, Dict


from easydict import EasyDict as edict
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import matplotlib as mpl
from matplotlib import cm

import source.utils.camera as camera


HUGE_NUMBER = 1e10
TINY_NUMBER = 1e-6 # float32 only has 7 decimal digits precision

img_HWC2CHW = lambda x: x.transpose(-1, -3).transpose(-1, -2)
gray2rgb = lambda x: x.unsqueeze(2).repeat(1, 1, 3)
to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)
mse2psnr = lambda x: -10. * np.log(x+TINY_NUMBER) / np.log(10.)


# -------------- visualization of depth maps in color ----------------
def get_vertical_colorbar(h, vmin, vmax, cmap_name='jet', label=None, cbar_precision=2):
    """Get vertical colorbar

    Args:
        h (int): size
        vmin (float): Min value to represent
        vmax (float)): Max value to represent
        cmap_name (str, optional): Defaults to 'jet'.
        label (_type_, optional):  Defaults to None.
        cbar_precision (int, optional): Defaults to 2.
    """

    fig = Figure(figsize=(2, 8), dpi=100)
    fig.subplots_adjust(right=1.5)
    canvas = FigureCanvasAgg(fig)

    # Do some plotting.
    ax = fig.add_subplot(111)
    cmap = cm.get_cmap(cmap_name)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    tick_cnt = 6
    tick_loc = np.linspace(vmin, vmax, tick_cnt)
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                    norm=norm,
                                    ticks=tick_loc,
                                    orientation='vertical')

    tick_label = [str(np.round(x, cbar_precision)) for x in tick_loc]
    if cbar_precision == 0:
        tick_label = [x[:-2] for x in tick_label]

    cb1.set_ticklabels(tick_label)

    cb1.ax.tick_params(labelsize=18, rotation=0)

    if label is not None:
        cb1.set_label(label)

    fig.tight_layout()

    canvas.draw()
    s, (width, height) = canvas.print_to_buffer()

    im = np.frombuffer(s, np.uint8).reshape((height, width, 4))

    im = im[:, :, :3].astype(np.float32) / 255.
    if h != im.shape[0]:
        w = int(im.shape[1] / im.shape[0] * h)
        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)

    return im


def colorize_np(x: np.ndarray, cmap_name='jet', mask: np.ndarray=None, 
                range: List[float]=None, append_cbar: bool=True, 
                cbar_in_image: bool=True, cbar_precision: int=2):
    """turn a grayscale image (numpy array) into a color image

    Args:
        x (np.array): input grayscale, [H, W]
        cmap_name (str, optional):the colorization method. Defaults to 'jet'.
        mask (np.array, optional): the mask image, [H, W]. Defaults to None.
        range (list, optional): the range for scaling, automatic if None, [min, max]. Defaults to None.
        append_cbar (bool, optional): append the color bar to the image?. Defaults to True.
        cbar_in_image (bool, optional): put the color bar inside the image to keep 
                                        the output image the same size as the input image? 
                                        Defaults to True.
        cbar_precision (int, optional): Defaults to 2.
    """
    if range is not None:
        vmin, vmax = range
    elif mask is not None:
        # vmin, vmax = np.percentile(x[mask], (2, 100))
        vmin = np.min(x[mask][np.nonzero(x[mask])])
        vmax = np.max(x[mask])
        # vmin = vmin - np.abs(vmin) * 0.01
        x[np.logical_not(mask)] = vmin
        # print(vmin, vmax)
    else:
        vmin, vmax = np.percentile(x, (1, 100))
        vmax += TINY_NUMBER

    x = np.clip(x, vmin, vmax)
    x = (x - vmin) / (vmax - vmin)
    # x = np.clip(x, 0., 1.)

    cmap = cm.get_cmap(cmap_name)
    x_new = cmap(x)[:, :, :3]

    if mask is not None:
        mask = np.float32(mask[:, :, np.newaxis])
        x_new = x_new * mask + np.ones_like(x_new) * (1. - mask)

    cbar = get_vertical_colorbar(h=x.shape[0], vmin=vmin, vmax=vmax, cmap_name=cmap_name, cbar_precision=cbar_precision)

    if append_cbar:
        if cbar_in_image:
            x_new[:, -cbar.shape[1]:, :] = cbar
        else:
            x_new = np.concatenate((x_new, np.zeros_like(x_new[:, :5, :]), cbar), axis=1)
        return x_new
    else:
        return x_new


# tensor
def colorize(x: torch.Tensor, cmap_name='jet', mask: torch.Tensor=None, range: List[float]=None, 
             append_cbar: bool=False, cbar_in_image: bool=False, channel_first: bool=False):
    """turn a grayscale image (torch tensor) into a color image

    Args:
        x (torch.Tensor): input grayscale, [H, W]
        cmap_name (str, optional):the colorization method. Defaults to 'jet'.
        mask (torch.Tensor, optional): the mask image, [H, W]. Defaults to None.
        range (list, optional): the range for scaling, automatic if None, [min, max]. Defaults to None.
        append_cbar (bool, optional): append the color bar to the image?. Defaults to True.
        cbar_in_image (bool, optional): put the color bar inside the image to keep 
                                        the output image the same size as the input image? 
                                        Defaults to True.
        cbar_precision (int, optional): Defaults to 2.
    """    
    device = x.device
    x = x.detach().cpu().numpy()
    if mask is not None:
        mask = mask.cpu().numpy() > 0.99
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)

    x = colorize_np(x, cmap_name, mask, range, append_cbar, cbar_in_image)
    x = torch.from_numpy(x).to(device)
    if channel_first:
        x = x.permute(2, 0, 1)
    return x

def print_args(args):
    '''
    Print Namespace object in json format
    '''
    dictionary_view = vars(args)
    json_view = json.dumps(dictionary_view, indent=4, sort_keys=True)
    print(json_view)

def convert_to_uint8(rgb):
    """
    Args:
        rgb: torch float tensor
    returns: 
        torch uint8 tensor
    """
    min_val = torch.min(rgb)
    max_val = torch.max(rgb)
    rgb = (rgb  - min_val) / (max_val - min_val)
    rgb = 255 * rgb
    return rgb.astype(np.uint8)


# ---------------Camera visualization -------------------
@torch.no_grad()
def vis_cameras(opt,vis,step,poses=[],colors=["blue","magenta"],plot_dist=True):
    win_name = "{}/{}".format(opt.group,opt.name)
    data = []
    # set up plots
    centers = []
    for pose,color in zip(poses,colors):
        pose = pose.detach().cpu()
        vertices,faces,wireframe = get_camera_mesh(pose,depth=opt.visdom.cam_depth)
        center = vertices[:,-1]
        centers.append(center)
        # camera centers
        data.append(dict(
            type="scatter3d",
            x=[float(n) for n in center[:,0]],
            y=[float(n) for n in center[:,1]],
            z=[float(n) for n in center[:,2]],
            mode="markers",
            marker=dict(color=color,size=3),
        ))
        # colored camera mesh
        vertices_merged,faces_merged = merge_meshes(vertices,faces)
        data.append(dict(
            type="mesh3d",
            x=[float(n) for n in vertices_merged[:,0]],
            y=[float(n) for n in vertices_merged[:,1]],
            z=[float(n) for n in vertices_merged[:,2]],
            i=[int(n) for n in faces_merged[:,0]],
            j=[int(n) for n in faces_merged[:,1]],
            k=[int(n) for n in faces_merged[:,2]],
            flatshading=True,
            color=color,
            opacity=0.05,
        ))
        # camera wireframe
        wireframe_merged = merge_wireframes(wireframe)
        data.append(dict(
            type="scatter3d",
            x=wireframe_merged[0],
            y=wireframe_merged[1],
            z=wireframe_merged[2],
            mode="lines",
            line=dict(color=color,),
            opacity=0.3,
        ))
    if plot_dist:
        # distance between two poses (camera centers)
        center_merged = merge_centers(centers[:2])
        data.append(dict(
            type="scatter3d",
            x=center_merged[0],
            y=center_merged[1],
            z=center_merged[2],
            mode="lines",
            line=dict(color="red",width=4,),
        ))
        if len(centers)==4:
            center_merged = merge_centers(centers[2:4])
            data.append(dict(
                type="scatter3d",
                x=center_merged[0],
                y=center_merged[1],
                z=center_merged[2],
                mode="lines",
                line=dict(color="red",width=4,),
            ))
    # send data to visdom
    vis._send(dict(
        data=data,
        win="poses",
        eid=win_name,
        layout=dict(
            title="({})".format(step),
            autosize=True,
            margin=dict(l=30,r=30,b=30,t=30,),
            showlegend=False,
            yaxis=dict(
                scaleanchor="x",
                scaleratio=1,
            )
        ),
        opts=dict(title="{} poses ({})".format(win_name,step),),
    ))

def get_camera_mesh(pose,depth=1):
    vertices = torch.tensor([[-0.5,-0.5,1],
                             [0.5,-0.5,1],
                             [0.5,0.5,1],
                             [-0.5,0.5,1],
                             [0,0,0]])*depth
    faces = torch.tensor([[0,1,2],
                          [0,2,3],
                          [0,1,4],
                          [1,2,4],
                          [2,3,4],
                          [3,0,4]])

    vertices = camera.cam2world(vertices[None],pose)
    wireframe = vertices[:,[0,1,2,3,0,4,1,2,4,3]]
    return vertices,faces,wireframe

def merge_wireframes(wireframe):
    wireframe_merged = [[],[],[]]
    for w in wireframe:
        wireframe_merged[0] += [float(n) for n in w[:,0]]+[None]
        wireframe_merged[1] += [float(n) for n in w[:,1]]+[None]
        wireframe_merged[2] += [float(n) for n in w[:,2]]+[None]
    return wireframe_merged
    
def merge_meshes(vertices,faces):
    mesh_N,vertex_N = vertices.shape[:2]
    faces_merged = torch.cat([faces+i*vertex_N for i in range(mesh_N)],dim=0)
    vertices_merged = vertices.view(-1,vertices.shape[-1])
    return vertices_merged,faces_merged

def merge_centers(centers):
    center_merged = [[],[],[]]
    for c1,c2 in zip(*centers):
        center_merged[0] += [float(c1[0]),float(c2[0]),None]
        center_merged[1] += [float(c1[1]),float(c2[1]),None]
        center_merged[2] += [float(c1[2]),float(c2[2]),None]
    return center_merged

def plot_save_poses(opt,fig,pose_w2c: torch.Tensor,pose_ref_w2c: torch.Tensor=None,
                    path: str=None,ep: int=None):
    # get the camera meshes
    cam_depth = 0.2  
    _,_,cam = get_camera_mesh(pose_w2c,depth=cam_depth)
    cam = cam.numpy()
    if pose_ref_w2c is not None:
        N_ref = len(pose_ref_w2c)
        _,_,cam_ref = get_camera_mesh(pose_ref_w2c,depth=cam_depth)
        cam_ref = cam_ref.numpy()
    # set up plot window(s)
    plt.title("Iteration {}".format(ep))
    ax1 = fig.add_subplot(121,projection="3d")
    ax2 = fig.add_subplot(122,projection="3d")
    setup_3D_plot(ax1,elev=-90,azim=-90,lim=edict(x=(-1,1),y=(-1,1),z=(-1,1)))
    setup_3D_plot(ax2,elev=0,azim=-90,lim=edict(x=(-1,1),y=(-1,1),z=(-1,1)))
    ax1.set_title("forward-facing view",pad=0)
    ax2.set_title("top-down view",pad=0)
    plt.subplots_adjust(left=0,right=1,bottom=0,top=0.95,wspace=0,hspace=0)
    plt.margins(tight=True,x=0,y=0)
    # plot the cameras
    N = len(cam)
    color = plt.get_cmap("gist_rainbow")
    for i in range(N):
        if pose_ref_w2c is not None:
            ax1.plot(cam_ref[i,:,0],cam_ref[i,:,1],cam_ref[i,:,2],color=(0.3,0.3,0.3),linewidth=1)
            ax2.plot(cam_ref[i,:,0],cam_ref[i,:,1],cam_ref[i,:,2],color=(0.3,0.3,0.3),linewidth=1)
            ax1.scatter(cam_ref[i,5,0],cam_ref[i,5,1],cam_ref[i,5,2],color=(0.3,0.3,0.3),s=40)
            ax2.scatter(cam_ref[i,5,0],cam_ref[i,5,1],cam_ref[i,5,2],color=(0.3,0.3,0.3),s=40)
        c = np.array(color(float(i)/N))*0.8
        ax1.plot(cam[i,:,0],cam[i,:,1],cam[i,:,2],color=c)
        ax2.plot(cam[i,:,0],cam[i,:,1],cam[i,:,2],color=c)
        ax1.scatter(cam[i,5,0],cam[i,5,1],cam[i,5,2],color=c,s=40)
        ax2.scatter(cam[i,5,0],cam[i,5,1],cam[i,5,2],color=c,s=40)

    # the line between them
    if pose_ref_w2c is not None and N == N_ref:
        for i in range(N):
            ax1.plot([cam[i,5,0],cam_ref[i,5,0]],
                    [cam[i,5,1],cam_ref[i,5,1]],
                    [cam[i,5,2],cam_ref[i,5,2]],color=(1,0,0),linewidth=3)
            ax2.plot([cam[i,5,0],cam_ref[i,5,0]],
                    [cam[i,5,1],cam_ref[i,5,1]],
                    [cam[i,5,2],cam_ref[i,5,2]],color=(1,0,0),linewidth=3)

    if path is not None:
        png_fname = "{}/{}.png".format(path,ep)
        plt.savefig(png_fname,dpi=75)
    
    fig.tight_layout(pad=0)
    canvas = FigureCanvas(fig)
    canvas.draw()      
    # draw the canvas, cache the renderer
    width, height = canvas.get_width_height() #fig.get_size_inches() * fig.get_dpi()
    image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    plt.close()
    # clean up
    plt.clf()
    return image


def plot_save_poses_blender(opt,fig,pose_w2c: torch.Tensor,pose_ref_w2c: torch.Tensor=None,
                            path: str=None,ep: int=None):

    # get the camera meshes
    cam_depth = 0.5
    # expresses camera position and frame in world coordinates 
    ver,_,cam = get_camera_mesh(pose_w2c,depth=cam_depth)
    cam = cam.numpy()
    if pose_ref_w2c is not None:
        N_ref = len(pose_ref_w2c)
        ver_ref,_,cam_ref = get_camera_mesh(pose_ref_w2c,depth=cam_depth)
        cam_ref = cam_ref.numpy()

    # set up plot window(s)
    plt.title("Iteration {}".format(ep))
    ax = fig.add_subplot(121,projection="3d")
    ax2 = fig.add_subplot(122,projection="3d")
    ax.set_title("azimuth 35",pad=0)
    ax2.set_title('azimuth 110',pad=0)
    setup_3D_plot(ax,elev=45,azim=35,lim=edict(x=(-3,3),y=(-3,3),z=(-3,2.4)))
    setup_3D_plot(ax2,elev=45,azim=110,lim=edict(x=(-3,3),y=(-3,3),z=(-3,2.4)))
    # elevation is degrees above the x-y plane) and an azimuth is rotated x degrees counter-clockwise about the z-axis
    plt.subplots_adjust(left=0,right=1,bottom=0,top=0.95,wspace=0,hspace=0)
    plt.margins(tight=True,x=0,y=0)
    # plot the cameras
    N = len(cam)
    ref_color = (0.7,0.2,0.7)
    pred_color = (0,0.6,0.7)

    # plot the reference camera
    if pose_ref_w2c is not None:
        ax.add_collection3d(Poly3DCollection([v[:4] for v in cam_ref],alpha=0.2,facecolor=ref_color))
        ax2.add_collection3d(Poly3DCollection([v[:4] for v in cam_ref],alpha=0.2,facecolor=ref_color))
        for i in range(N_ref):
            ax.plot(cam_ref[i,:,0],cam_ref[i,:,1],cam_ref[i,:,2],color=ref_color,linewidth=0.5)
            ax.scatter(cam_ref[i,5,0],cam_ref[i,5,1],cam_ref[i,5,2],color=ref_color,s=20)
            ax2.plot(cam_ref[i,:,0],cam_ref[i,:,1],cam_ref[i,:,2],color=ref_color,linewidth=0.5)
            ax2.scatter(cam_ref[i,5,0],cam_ref[i,5,1],cam_ref[i,5,2],color=ref_color,s=20)
    '''
    if ep==0:
        png_fname = "{}/GT.png".format(path)
        plt.savefig(png_fname,dpi=75)
    '''
    ax.add_collection3d(Poly3DCollection([v[:4] for v in cam],alpha=0.2,facecolor=pred_color))
    ax2.add_collection3d(Poly3DCollection([v[:4] for v in cam],alpha=0.2,facecolor=pred_color))
    for i in range(N):
        ax.plot(cam[i,:,0],cam[i,:,1],cam[i,:,2],color=pred_color,linewidth=1)
        ax.scatter(cam[i,5,0],cam[i,5,1],cam[i,5,2],color=pred_color,s=20)
        ax.text(cam[i,5,0],cam[i,5,1],cam[i,5,2],  f'{i}', size=10, zorder=1, color='k') 

        ax2.plot(cam[i,:,0],cam[i,:,1],cam[i,:,2],color=pred_color,linewidth=1)
        ax2.scatter(cam[i,5,0],cam[i,5,1],cam[i,5,2],color=pred_color,s=20)
        ax2.text(cam[i,5,0],cam[i,5,1],cam[i,5,2],  f'{i}', size=10, zorder=1, color='k') 
    
    # the line between them
    if pose_ref_w2c is not None and N == N_ref:
        for i in range(N):
            ax.plot([cam[i,5,0],cam_ref[i,5,0]],
                    [cam[i,5,1],cam_ref[i,5,1]],
                    [cam[i,5,2],cam_ref[i,5,2]],color=(1,0,0),linewidth=3)
            ax2.plot([cam[i,5,0],cam_ref[i,5,0]],
                    [cam[i,5,1],cam_ref[i,5,1]],
                    [cam[i,5,2],cam_ref[i,5,2]],color=(1,0,0),linewidth=3)
            
    if path is not None:
        png_fname = "{}/{}.png".format(path,ep)
        plt.savefig(png_fname,dpi=75)
    
    fig.tight_layout(pad=0)
    canvas = FigureCanvas(fig)
    canvas.draw()      
    # draw the canvas, cache the renderer
    width, height = canvas.get_width_height() #fig.get_size_inches() * fig.get_dpi()
    image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    plt.close()
    # clean up
    plt.clf()
    return image

def setup_3D_plot(ax,elev,azim,lim=None):
    ax.xaxis.set_pane_color((1.0,1.0,1.0,0.0))
    ax.yaxis.set_pane_color((1.0,1.0,1.0,0.0))
    ax.zaxis.set_pane_color((1.0,1.0,1.0,0.0))
    ax.xaxis._axinfo["grid"]["color"] = (0.9,0.9,0.9,1)
    ax.yaxis._axinfo["grid"]["color"] = (0.9,0.9,0.9,1)
    ax.zaxis._axinfo["grid"]["color"] = (0.9,0.9,0.9,1)
    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.zaxis.set_tick_params(labelsize=8)
    ax.set_xlabel("X",fontsize=16)
    ax.set_ylabel("Y",fontsize=16)
    ax.set_zlabel("Z",fontsize=16)
    ax.set_xlim(lim.x[0],lim.x[1])
    ax.set_ylim(lim.y[0],lim.y[1])
    ax.set_zlim(lim.z[0],lim.z[1])
    ax.view_init(elev=elev,azim=azim)

# --- VISUALIZATION OF KEYPOINT MATCHEs --- 

def error_colormap(x):
    return np.clip(
        np.stack([2-x*2, x*2, np.zeros_like(x), np.ones_like(x)], -1), 0, 1) * 255


def make_matching_plot_fast(image0, image1, kpts0, kpts1, color=None, 
                            text=[], path=None,
                            show_keypoints=True, margin=10,
                            opencv_display=False, opencv_title='',
                            small_text=[]):
    H0, W0 = image0.shape[:2]
    H1, W1 = image1.shape[:2]
    H, W = max(H0, H1), W0 + W1 + margin

    out = 255*np.ones((H, W, 3), np.uint8)
    out[:H0, :W0] = image0
    out[:H1, W0+margin:] = image1

    kpts0, kpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
    valid_pts = (kpts0[:, 0] >=0) &  (kpts0[:, 1] >=1) & (kpts1[:, 0] >=0) & (kpts0[:, 1] >=0) & \
        (kpts0[:, 0] <= W0-1) & (kpts0[:, 1] <= H0-1) & (kpts1[:, 0] <= W1-1) & (kpts1[:, 1] <= H1-1)
    kpts0 = kpts0[valid_pts]
    kpts1 = kpts1[valid_pts] 

    if show_keypoints:
        white = (255, 255, 255)
        black = (0, 0, 0)
        for x, y in kpts0:
            cv2.circle(out, (x, y), 2, black, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x, y), 1, white, -1, lineType=cv2.LINE_AA)
        for x, y in kpts1:
            cv2.circle(out, (x + margin + W0, y), 2, black, -1,
                       lineType=cv2.LINE_AA)
            cv2.circle(out, (x + margin + W0, y), 1, white, -1,
                       lineType=cv2.LINE_AA)


    mkpts0, mkpts1 = kpts0, kpts1
    if color is None:
        color = error_colormap(np.arange(mkpts0.shape[0]) / mkpts0.shape[0])
    color = (np.array(color[:, :3])*255).astype(int)[:, ::-1]
    for (x0, y0), (x1, y1), c in zip(mkpts0, mkpts1, color):
        c = c.tolist()
        cv2.line(out, (x0, y0), (x1 + margin + W0, y1),
                 color=c, thickness=1, lineType=cv2.LINE_AA)
        # display line end-points as circles
        cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + W0, y1), 2, c, -1,
                   lineType=cv2.LINE_AA)

    # Scale factor for consistent visualization across scales.
    sc = min(H / 640., 2.0)

    # Big text.
    Ht = int(30 * sc)  # text height
    txt_color_fg = (255, 255, 255)
    txt_color_bg = (0, 0, 0)
    for i, t in enumerate(text):
        cv2.putText(out, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0*sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0*sc, txt_color_fg, 1, cv2.LINE_AA)

    # Small text.
    Ht = int(18 * sc)  # text height
    for i, t in enumerate(reversed(small_text)):
        cv2.putText(out, t, (int(8*sc), int(H-Ht*(i+.6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5*sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8*sc), int(H-Ht*(i+.6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5*sc, txt_color_fg, 1, cv2.LINE_AA)

    if path is not None:
        print(path)
        cv2.imwrite(str(path), out)

    if opencv_display:
        cv2.imshow(opencv_title, out)
        cv2.waitKey(1)

    return out

