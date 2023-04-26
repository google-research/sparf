## Our dataloaders and data-format

All our dataloaders inherit from [source/datasets/base.py](https://github.com/google-research/sparf/blob/main/source/datasets/base.py). Each dataloader reads images and
pose informations, stored in format specific to each dataset. The `__get_item__` method creates 
a dictionary for each image index containing the following elements: 
* `idx`: the index of the image
* `rgb_path`: the path to the RGB image. Will be used to save the renderings with a name. 
* `image`: the corresponding image, a torch Tensor of shape [3, H, W]. The RGB values are 
              normalized to [0, 1] (not [0, 255]). 
* `intr`: intrinsics parameters, numpy array of shape [3, 3]
* `pose`:  **world-to-camera** transformation matrix, numpy array of shaoe [3, 4]
* `depth_range`: depth_range, numpy array of shape [1, 2]
* `scene`: string, scene name

Optionally, when the depth or a foreground mask are available:
* `depth_gt`: ground-truth depth map, numpy array of shape [H, W]
* `valid_depth_gt`: mask indicating where the depth map is valid, bool numpy array of shape [H, W]
* `fg_mask`: foreground segmentation mask, bool numpy array of shape [1, H, W]


In the code base, after creating the dataloader, we call `prefetch_all_data`, which will batch 
all data, and store the corresponding batch in train_data.all. 


**Images**

`image` = [N, 3, height, width] torch.Tensor of RGB images. Currently we
require all images to have the same resolution.
<br /><br />

**Extrinsic camera poses**

`pose` = [N, 3, 4] numpy array of extrinsic pose matrices. Should be in **world-to-camera** format. 
`pose[i]` should be in **world-to-camera**  format, such that we can run

```
pose_w2c = pose[i]
x_camera = pose_w2c[:3, :3] @ x_world + pose_w2c[:3, 3:4]
```

to convert a 3D world space point `x_world` into a camera space point `x_camera`.

These matrices must be stored in the **OpenCV** coordinate system convention for camera rotation:
x-axis to the right, y-axis down, and z-axis forwards. 


The most common conventions are

-   `[right, up, backwards]`: OpenGL, NeRF, most graphics code.
-   `[right, down, forwards]`: OpenCV, COLMAP, most computer vision code.

Fortunately switching from OpenCV/COLMAP to NeRF and inversely is
[simple](https://github.com/google-research/multinerf/blob/main/internal/datasets.py#L108):
you just need to right-multiply the OpenCV pose matrices by `np.diag([1, -1, -1, 1])`,
which will flip the sign of the y-axis (from down to up) and z-axis (from
forwards to backwards):
```
camtoworlds_opengl = camtoworlds_opencv @ np.diag([1, -1, -1, 1])
```


Importantly, you want your scene 3D points to be more or less centered around 0 in the world coordinate space. 
In 360 degrees inwards scene, this is ensured by centering the camera-to-world poses around 0. 
In scenes like Replica where the cameras are outwards, this is done by substracting the center point of the 3D scene to the camera poses. Check [the dataloader](https://github.com/google-research/sparf/blob/main/source/datasets/rgbd_datasets.py) for more details. 
<br /><br />


**Intrinsic camera poses**

`intr`= [N, 3, 3] numpy array of intrinsic matrices. These should be in
**OpenCV** format, e.g.

```
intr = np.array([
  [focal,     0,  width/2],
  [    0, focal, height/2],
  [    0,     0,        1],
])
```
<br />





## Existing data loaders

We have implemented different dataloaders:

-   [LLFF](https://github.com/google-research/sparf/blob/main/source/datasets/llff.py)
-   [DTU](https://github.com/google-research/sparf/blob/main/source/datasets/dtu.py)
-   [Replica](https://github.com/google-research/sparf/blob/main/source/datasets/rgbd_datasets.py)




