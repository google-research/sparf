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
from easydict import EasyDict as edict
from typing import Callable, Sequence, List, Mapping, MutableMapping, Tuple, Union, Dict
from typing import Any, Optional


def pad_poses(p: torch.Tensor) -> torch.Tensor:
  """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
  bottom = torch.broadcast_to(torch.tensor([0, 0, 0, 1.0], device=p.device, 
                              dtype=p.dtype), 
                              p[..., :1, :4].shape)
  return torch.cat((p[..., :3, :4], bottom), dim=-2)


def unpad_poses(p: torch.Tensor) -> torch.Tensor:
  """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
  return p[..., :3, :4]


def pose_inverse_4x4(mat: torch.Tensor, use_inverse: bool=False) -> torch.Tensor:
    """
    Transforms world2cam into cam2world or vice-versa, without computing the inverse.
    Args:
        mat (torch.Tensor): pose matrix (B, 4, 4) or (4, 4)
    """
    # invert a camera pose
    out_mat = torch.zeros_like(mat)

    if len(out_mat.shape) == 3:
        # must be (B, 4, 4)
        out_mat[:, 3, 3] = 1
        R,t = mat[:, :3, :3],mat[:,:3, 3:]
        R_inv = R.inverse() if use_inverse else R.transpose(-1,-2)
        t_inv = (-R_inv@t)[..., 0]

        pose_inv = torch.cat([R_inv,t_inv[...,None]],dim=-1) # [...,3,4]

        out_mat[:, :3] = pose_inv
    else:
        out_mat[3, 3] = 1
        R,t = mat[:3, :3], mat[:3, 3:]
        R_inv = R.inverse() if use_inverse else R.transpose(-1,-2)
        t_inv = (-R_inv@t)[..., 0]
        pose_inv = torch.cat([R_inv,t_inv[...,None]],dim=-1) # [3,4]
        out_mat[:3] = pose_inv
    # assert torch.equal(out_mat, torch.inverse(mat))
    return out_mat


class Pose():
    """
    A class of operations on camera poses (PyTorch tensors with shape [...,3,4])
    each [3,4] camera pose takes the form of [R|t]
    """

    def __call__(self,R=None,t=None):
        # construct a camera pose from the given R and/or t
        assert(R is not None or t is not None)
        if R is None:
            if not isinstance(t,torch.Tensor): t = torch.tensor(t)
            R = torch.eye(3,device=t.device).repeat(*t.shape[:-1],1,1)
        elif t is None:
            if not isinstance(R,torch.Tensor): R = torch.tensor(R)
            t = torch.zeros(R.shape[:-1],device=R.device)
        else:
            if not isinstance(R,torch.Tensor): R = torch.tensor(R)
            if not isinstance(t,torch.Tensor): t = torch.tensor(t)
        assert(R.shape[:-1]==t.shape and R.shape[-2:]==(3,3))
        R = R.float()
        t = t.float()
        pose = torch.cat([R,t[...,None]],dim=-1) # [...,3,4]
        assert(pose.shape[-2:]==(3,4))
        return pose

    def invert(self,pose: torch.Tensor,use_inverse=False) -> torch.Tensor:
        # invert a camera pose
        R,t = pose[...,:3],pose[...,3:]
        R_inv = R.inverse() if use_inverse else R.transpose(-1,-2)
        t_inv = (-R_inv@t)[...,0]
        pose_inv = self(R=R_inv,t=t_inv)
        return pose_inv

    def compose(self,pose_list: List[torch.Tensor]) -> torch.Tensor:
        # compose a sequence of poses together
        # pose_new(x) = poseN o ... o pose2 o pose1(x)
        pose_new = pose_list[0]
        for pose in pose_list[1:]:
            pose_new = self.compose_pair_b_at_a(pose_a=pose_new,pose_b=pose)
        return pose_new

    def compose_pair_b_at_a(self,pose_a: torch.Tensor,pose_b: torch.Tensor) -> torch.Tensor:
        # pose_new(x) = pose_b o pose_a(x)
        R_a,t_a = pose_a[...,:3],pose_a[...,3:]
        R_b,t_b = pose_b[...,:3],pose_b[...,3:]
        R_new = R_b@R_a
        t_new = (R_b@t_a+t_b)[...,0]
        pose_new = self(R=R_new,t=t_new)
        return pose_new

class Lie():
    """
    Lie algebra for SO(3) and SE(3) operations in PyTorch
    """

    def so3_to_SO3(self,w: torch.Tensor) -> torch.Tensor: # [...,3]
        # from lie algebra to rotation matrix
        # rodrigues formula 
        wx = self.skew_symmetric(w)
        theta = w.norm(dim=-1)[...,None,None]
        I = torch.eye(3,device=w.device,dtype=torch.float32)
        A = self.taylor_A(theta)
        B = self.taylor_B(theta)
        R = I+A*wx+B*wx@wx
        return R

    def SO3_to_so3(self,R: torch.Tensor,eps=1e-7) -> torch.Tensor: # [...,3,3]
        # rotation matrix to lie algebra formulation, as rotation around axis of amount equal to norm of axis
        trace = R[...,0,0]+R[...,1,1]+R[...,2,2]
        theta = ((trace-1)/2).clamp(-1+eps,1-eps).acos_()[...,None,None]%np.pi # ln(R) will explode if theta==pi
        lnR = 1/(2*self.taylor_A(theta)+1e-8)*(R-R.transpose(-2,-1)) # FIXME: wei-chiu finds it weird
        w0,w1,w2 = lnR[...,2,1],lnR[...,0,2],lnR[...,1,0]
        w = torch.stack([w0,w1,w2],dim=-1)
        return w

    def se3_to_SE3(self,wu: torch.Tensor) -> torch.Tensor: # [...,3]
        # se3 is lie algebra and translation vector 
        w,u = wu.split([3,3],dim=-1)

        # so3 to SO3
        wx = self.skew_symmetric(w)
        theta = w.norm(dim=-1)[...,None,None]
        I = torch.eye(3,device=w.device,dtype=torch.float32)
        A = self.taylor_A(theta)
        B = self.taylor_B(theta)
        C = self.taylor_C(theta)
        R = I+A*wx+B*wx@wx

        V = I+B*wx+C*wx@wx
        Rt = torch.cat([R,(V@u[...,None])],dim=-1)
        return Rt

    def SE3_to_se3(self,Rt: torch.Tensor,eps=1e-8) -> torch.Tensor: # [...,3,4]
        R,t = Rt.split([3,1],dim=-1)
        w = self.SO3_to_so3(R)
        wx = self.skew_symmetric(w)
        theta = w.norm(dim=-1)[...,None,None]
        I = torch.eye(3,device=w.device,dtype=torch.float32)
        A = self.taylor_A(theta)
        B = self.taylor_B(theta)
        invV = I-0.5*wx+(1-A/(2*B))/(theta**2+eps)*wx@wx
        u = (invV@t)[...,0]
        wu = torch.cat([w,u],dim=-1)
        return wu    

    def skew_symmetric(self,w: torch.Tensor) -> torch.Tensor:
        w0,w1,w2 = w.unbind(dim=-1)
        O = torch.zeros_like(w0)
        wx = torch.stack([torch.stack([O,-w2,w1],dim=-1),
                          torch.stack([w2,O,-w0],dim=-1),
                          torch.stack([-w1,w0,O],dim=-1)],dim=-2)
        return wx

    def taylor_A(self,x: torch.Tensor,nth=10) -> torch.Tensor:
        # Taylor expansion of sin(x)/x
        ans = torch.zeros_like(x)
        denom = 1.
        for i in range(nth+1):
            if i>0: denom *= (2*i)*(2*i+1)
            ans = ans+(-1)**i*x**(2*i)/denom
        return ans

    def taylor_B(self,x: torch.Tensor,nth=10) -> torch.Tensor:
        # Taylor expansion of (1-cos(x))/x**2
        ans = torch.zeros_like(x)
        denom = 1.
        for i in range(nth+1):
            denom *= (2*i+1)*(2*i+2)
            ans = ans+(-1)**i*x**(2*i)/denom
        return ans

    def taylor_C(self,x: torch.Tensor,nth=10) -> torch.Tensor:
        # Taylor expansion of (x-sin(x))/x**3
        ans = torch.zeros_like(x)
        denom = 1.
        for i in range(nth+1):
            denom *= (2*i+2)*(2*i+3)
            ans = ans+(-1)**i*x**(2*i)/denom
        return ans

class Quaternion():

    def q_to_R(self,q: torch.Tensor) -> torch.Tensor:
        """
        Convert quaternion to rotation in batch. Since all operation in pytorch, support gradient passing.
        Args:
            quad (tensor, batch_size*4): quaternion.
        Returns:
            rot_mat (tensor, batch_size*3*3): rotation.
        """
        q = torch.nn.functional.normalize(q, dim=-1)  # normalize 
        # https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
        qa,qb,qc,qd = q.unbind(dim=-1)
        R = torch.stack([torch.stack([1-2*(qc**2+qd**2),2*(qb*qc-qa*qd),2*(qa*qc+qb*qd)],dim=-1),
                         torch.stack([2*(qb*qc+qa*qd),1-2*(qb**2+qd**2),2*(qc*qd-qa*qb)],dim=-1),
                         torch.stack([2*(qb*qd-qa*qc),2*(qa*qb+qc*qd),1-2*(qb**2+qc**2)],dim=-1)],dim=-2)
        
        '''
        bs = quad.shape[0]
        qr, qi, qj, qk = quad[:, 0], quad[:, 1], quad[:, 2], quad[:, 3]
        two_s = 2.0 / (quad * quad).sum(-1)
        rot_mat = torch.zeros(bs, 3, 3).to(quad.get_device())
        rot_mat[:, 0, 0] = 1 - two_s * (qj ** 2 + qk ** 2)
        rot_mat[:, 0, 1] = two_s * (qi * qj - qk * qr)
        rot_mat[:, 0, 2] = two_s * (qi * qk + qj * qr)
        rot_mat[:, 1, 0] = two_s * (qi * qj + qk * qr)
        rot_mat[:, 1, 1] = 1 - two_s * (qi ** 2 + qk ** 2)
        rot_mat[:, 1, 2] = two_s * (qj * qk - qi * qr)
        rot_mat[:, 2, 0] = two_s * (qi * qk - qj * qr)
        rot_mat[:, 2, 1] = two_s * (qj * qk + qi * qr)
        rot_mat[:, 2, 2] = 1 - two_s * (qi ** 2 + qj ** 2)
        return rot_mat
        '''
        return R

    def R_to_q(self,R: torch.Tensor,eps=1e-8) -> torch.Tensor: # [B,3,3]
        def rotmat2qvec(R):
            # the quaternion will be [w, x, y, z]
            Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
            K = np.array([
                [Rxx - Ryy - Rzz, 0, 0, 0],
                [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
            eigvals, eigvecs = np.linalg.eigh(K)
            qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
            if qvec[0] < 0:
                qvec *= -1
            return qvec

        is_torch = False
        if isinstance(R, torch.Tensor):
            device = R.device
            R = R.cpu().detach().numpy()
            is_torch = True
        if len(R.shape) == 3:
            # batch
            batch = R.shape[0]
            quat=[]
            for b in range(batch):
                q = rotmat2qvec(R[b])
                quat.append(q)
            q = np.stack(quat, axis=0)
        else:
            q = quaternion_from_matrix_single(R)
        
        if is_torch:
            q = torch.from_numpy(q).to(device).float()
        return q

    def invert(self,q: torch.Tensor) -> torch.Tensor:
        qa,qb,qc,qd = q.unbind(dim=-1)
        norm = q.norm(dim=-1,keepdim=True)
        q_inv = torch.stack([qa,-qb,-qc,-qd],dim=-1)/norm**2
        return q_inv

    def product(self,q1: torch.Tensor,q2: torch.Tensor) -> torch.Tensor: # [B,4]
        q1a,q1b,q1c,q1d = q1.unbind(dim=-1)
        q2a,q2b,q2c,q2d = q2.unbind(dim=-1)
        hamil_prod = torch.stack([q1a*q2a-q1b*q2b-q1c*q2c-q1d*q2d,
                                  q1a*q2b+q1b*q2a+q1c*q2d-q1d*q2c,
                                  q1a*q2c-q1b*q2d+q1c*q2a+q1d*q2b,
                                  q1a*q2d+q1b*q2c-q1c*q2b+q1d*q2a],dim=-1)
        return hamil_prod

pose = Pose()
lie = Lie()
quaternion = Quaternion()

def to_hom(X: torch.Tensor) -> torch.Tensor:
    # get homogeneous coordinates of the input
    X_hom = torch.cat([X,torch.ones_like(X[...,:1])],dim=-1)
    return X_hom

# basic operations of transforming 3D points between world/camera/image coordinates
def world2cam(X_world: torch.Tensor,pose_w2c: torch.Tensor) -> torch.Tensor: # [B,N,3]
    """
    Transform 3D points X_world in world coordinate system to camera coordinate sytem.
    Args:
        X_world (torch.Tensor): (.., N, 3)
        pose_w2c (torch.Tensor): (..., 3, 4)

    Returns:
        torch.Tensor: (..., N, 3)
    """
    X_hom = to_hom(X_world)
    return X_hom@pose_w2c.transpose(-1,-2)

def cam2img(X: torch.Tensor,cam_intr: torch.Tensor) -> torch.Tensor:
    return X@cam_intr.transpose(-1,-2)

def img2cam(X: torch.Tensor,cam_intr: torch.Tensor) -> torch.Tensor:
    return X@cam_intr.inverse().transpose(-1,-2)

def cam2world(X_cam: torch.Tensor,pose_w2c: torch.Tensor) -> torch.Tensor:
    """
    Transform 3D point X_cam from camera coordinate to world coordinate system
    Args:
        X_cam (torch.Tensor): (.., N, 3)
        pose_w2c (torch.Tensor): (..., 3, 4)

    Returns:
        torch.Tensor: (..., N, 3)
    """
    X_hom = to_hom(X_cam)
    pose_c2w = Pose().invert(pose_w2c)
    # usually X_w = Pctow @ Xc when Xc is (3, N)
    return X_hom@pose_c2w.transpose(-1,-2)  # here P is (3, 4) and not (4, 4)

def angle_to_rotation_matrix(a: torch.Tensor,axis) -> torch.Tensor:
    # get the rotation matrix from Euler angle around specific axis
    roll = dict(X=1,Y=2,Z=0)[axis]
    O = torch.zeros_like(a)
    I = torch.ones_like(a)
    M = torch.stack([torch.stack([a.cos(),-a.sin(),O],dim=-1),
                     torch.stack([a.sin(),a.cos(),O],dim=-1),
                     torch.stack([O,O,I],dim=-1)],dim=-2)
    M = M.roll((roll,roll),dims=(-2,-1))
    return M

def get_center_and_ray(pose_w2c: torch.Tensor, H: int, W: int, 
                       intr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]: # [HW,2]
    """given the intrinsic/extrinsic matrices, get the camera center and ray directions
    at all pixels of an image of size (H, W)

    Args:
        pose_w2c (torch.Tensor): (B, 3, 4)
        H (int): size of image
        W (int): size of image
        intr (torch.Tensor): (B, 3, 3)

    Returns:
        center_3D, ray: torch.Tensor of sixe (B, HW, 3)
    """
    assert H is not None and W is not None
    # assert(opt.camera.model=="perspective")
    with torch.no_grad():
        # compute image coordinate grid
        y_range = torch.arange(H,dtype=torch.float32,device=pose_w2c.device).add_(0.5)
        x_range = torch.arange(W,dtype=torch.float32,device=pose_w2c.device).add_(0.5)
        Y,X = torch.meshgrid(y_range,x_range) # [H,W]
        xy_grid = torch.stack([X,Y],dim=-1).view(-1,2) # [HW,2]
    # compute center and ray
    batch_size = len(pose_w2c)
    xy_grid = xy_grid.repeat(batch_size,1,1) # [B,HW,2]
    grid_3D = img2cam(to_hom(xy_grid),intr) # [B,HW,3]
    center_3D = torch.zeros_like(grid_3D) # [B,HW,3]
    # transform from camera to world coordinates
    grid_3D = cam2world(grid_3D,pose_w2c) # [B,HW,3]
    center_3D = cam2world(center_3D,pose_w2c) # [B,HW,3]  
    # this is equivalent to just posecam2world[:3, 3], ie the position of the camera in the world coordinate frame

    ray = grid_3D-center_3D # [B,HW,3]  
    # this is equivalent to Rctow @ K-1 @ u, because here multiplied by the whole P and not just R, 
    # so need to substract the translation vector
    return center_3D,ray

def get_center_and_ray_at_pixels(pose_w2c: torch.Tensor, pixels: torch.Tensor, 
                                 intr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]: # [HW,2]
    """ 
    given the intrinsic/extrinsic matrices, get the camera center and ray directions
    of a specific set of pixles

    Args:
        pose_w2c (torch.Tensor): (B, 3, 4)
        pixels: (N, 2) or (B, N, 2) where the pixels can be different for each image in batch
        intr (torch.Tensor): (B, 3, 3)

    Returns:
        center_3D, ray: torch.Tensor of sixe (N, 3) or (B, N, 3)
    """
    # HW is N here 
    batch_size = len(pose_w2c)
    if len(pixels.shape) == 2:
        # (N, 2)
        xy_grid = pixels.unsqueeze(0).repeat(batch_size, 1, 1)
    else:
        xy_grid = pixels # [B,N,2]
        
    grid_3D = img2cam(to_hom(xy_grid),intr) # [B,HW,3]
    center_3D = torch.zeros_like(grid_3D) # [B,HW,3]
    # transform from camera to world coordinates
    grid_3D = cam2world(grid_3D,pose_w2c) # [B,HW,3]
    center_3D = cam2world(center_3D,pose_w2c) # [B,HW,3]  
    # this is equivalent to just pose_cam2world[:3, 3], ie the position of the camera in the world coordinate frame

    ray = grid_3D-center_3D # [B,HW,3]  
    # this is equivalent to Rctow @ K-1 @ u, because here multiplied by the whole P and not just R, 
    # so need to substract the translation vector
    return center_3D,ray

def get_3D_points_from_depth(center: torch.Tensor,ray: torch.Tensor,
                             depth: torch.Tensor,
                             multi_samples: bool=False) -> torch.Tensor:
    """Given the camera center, ray directions and depth samples, compute the corresponding
    3D points in the world coordinate system. 

    Args:
        center (torch.Tensor): Centers of cameras in the world coordinate system (B, N, 3)
        ray (torch.Tensor): Ray directions in the world coordinate system (B, N, 3)
        depth (torch.Tensor):  shape (B, N, N_samples, 1) Depth sampled along the ray
        multi_samples (bool, optional): Defaults to False.

    Returns:
        points_3D (torch.Tensor): shape (B, N, N_samples, 3)
    """
    if multi_samples: center,ray = center[:,:,None],ray[:,:,None]
    # x = c+dv
    points_3D = center+ray*depth # [B,HW,3]+ [N_samples,3]*[B,HW,N_samples,3]
    # equivalent to direction Rctow @ K-1 @ u + tcinw, where tcinw = Pctow[:3, 3]
    return points_3D

def convert_NDC(center: torch.Tensor,ray: torch.Tensor,
                intr: torch.Tensor,near=1) -> Tuple[torch.Tensor, torch.Tensor]:
    """shift camera center (ray origins) to near plane (z=1)
    (unlike conventional NDC, we assume the cameras are facing towards the +z direction)

    Args:
        center (torch.Tensor): Centers of cameras in the world coordinate system (B, N, 3)
        ray (torch.Tensor): Ray directions in the world coordinate system (B, N, 3)
        intr (torch.Tensor): (B, 3, 3)
        near (int, optional): Near plane value. Defaults to 1.
    """
    center = center+(near-center[...,2:])/ray[...,2:]*ray
    # projection
    cx,cy,cz = center.unbind(dim=-1) # [B,HW]
    rx,ry,rz = ray.unbind(dim=-1) # [B,HW]
    scale_x = intr[:,0,0]/intr[:,0,2] # [B]
    scale_y = intr[:,1,1]/intr[:,1,2] # [B]
    cnx = scale_x[:,None]*(cx/cz)
    cny = scale_y[:,None]*(cy/cz)
    cnz = 1-2*near/cz
    rnx = scale_x[:,None]*(rx/rz-cx/cz)
    rny = scale_y[:,None]*(ry/rz-cy/cz)
    rnz = 2*near/cz
    center_ndc = torch.stack([cnx,cny,cnz],dim=-1) # [B,HW,3]
    ray_ndc = torch.stack([rnx,rny,rnz],dim=-1) # [B,HW,3]
    return center_ndc,ray_ndc

def rotation_distance(R1: torch.Tensor,R2: torch.Tensor,eps=1e-7) -> torch.Tensor:
    # http://www.boris-belousov.net/2016/12/01/quat-dist/
    R_diff = R1@R2.transpose(-2,-1)
    trace = R_diff[...,0,0]+R_diff[...,1,1]+R_diff[...,2,2]
    angle = ((trace-1)/2).clamp(-1+eps,1-eps).acos_() # numerical stability near -1/+1
    return angle

def procrustes_analysis(X0: torch.Tensor,X1: torch.Tensor) -> Dict[str, Any]: # [N,3]
    """Computes 3D similarity between two coordinate systems. 

    Args:
        X0 (torch.Tensor): Center of cameras in coordinate system 0, (B, 3)
        X1 (torch.Tensor_type_): Center of cameras in coordinate system 1, (B, 3)
    """
    # translation
    t0 = X0.mean(dim=0,keepdim=True)
    t1 = X1.mean(dim=0,keepdim=True)
    X0c = X0-t0
    X1c = X1-t1
    # scale
    s0 = (X0c**2).sum(dim=-1).mean().sqrt()
    s1 = (X1c**2).sum(dim=-1).mean().sqrt()
    X0cs = X0c/s0
    X1cs = X1c/s1
    # rotation (use double for SVD, float loses precision)
    U, S, Vh = torch.linalg.svd((X0cs.t()@X1cs).cpu().double(), full_matrices=False)
    V = Vh.transpose(-2, -1).conj()
    R = (U@V.t()).float().to(X0.device)
    if R.det()<0: R[2] *= -1
    # align X1 to X0: X1to0 = (X1-t1)/s1@R.t()*s0+t0
    sim3 = edict(t0=t0[0],t1=t1[0],s0=s0,s1=s1,R=R)
    return sim3

def get_novel_view_poses(opt: Dict[str, Any],pose_anchor: torch.Tensor,N: int=60,scale: float=1) -> torch.Tensor:
    """create circular viewpoints (small oscillations)

    Args:
        opt (edict): settings
        pose_anchor (torch.Tensor): w2c input or optimized poses. (B, 3, 4)
        N (int, optional): Number of output poses. Defaults to 60.
        scale (int, optional): Defaults to 1.

    Returns:
        pose_novel (torch.Tensor): w2c of shape (B, 3, 4)
    """
    theta = torch.arange(N)/N*2*np.pi
    R_x = angle_to_rotation_matrix((theta.sin()*0.1).asin(),"X")
    R_y = angle_to_rotation_matrix((theta.cos()*0.1).asin(),"Y")
    pose_rot = pose(R=R_y@R_x)
    pose_shift = pose(t=[0,0,-4*scale])
    pose_shift2 = pose(t=[0,0,3.8*scale])
    pose_oscil = pose.compose([pose_shift,pose_rot,pose_shift2])
    pose_novel = pose.compose([pose_oscil,pose_anchor.cpu()[None]])
    return pose_novel
