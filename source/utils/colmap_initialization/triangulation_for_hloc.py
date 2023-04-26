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

"""Functions for triangulation to use in HLOC. """


from typing import Optional, List, Dict, Any
from pathlib import Path
import pycolmap
import torch
from pathlib import Path
import sys 
from typing import Callable, Sequence, List, Mapping, MutableMapping, Tuple, Union, Dict, Any, Optional


from source.utils.colmap_initialization.reconstruction_know_intrinsics_for_hloc import import_matches_modified
sys.path.append(str(Path(__file__).parent / '../../../third_party/Hierarchical-Localization'))

from hloc import logger
from hloc.utils.database import COLMAPDatabase
from hloc.utils.read_write_model import rotmat2qvec, write_model, Image, Camera
from hloc.triangulation import (create_db_from_model, run_triangulation, 
                                estimation_and_geometric_verification, 
                                import_features, geometric_verification)
                                

def create_empty_db(database_path: Path):
    if database_path.exists():
        logger.warning('The database already exists, deleting it.')
        database_path.unlink()
    logger.info('Creating an empty database...')
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()
    db.commit()
    db.close()
    return 


def triangulation_from_given_camera_param(sfm_dir: Path,
                                          image_list: List, 
                                          pose_w2c: torch.Tensor, 
                                          cam: torch.Tensor, 
                                          image_dir: Path,
                                          pairs: Path,
                                          features: Path,
                                          matches: Path,
                                          skip_geometric_verification: bool = False,
                                          estimate_two_view_geometries: bool = False,
                                          min_match_score: Optional[float] = None,
                                          verbose: bool = False,
                                          mapper_options: Optional[Dict[str, Any]] = None,
                                          ) -> pycolmap.Reconstruction:

    assert features.exists(), features
    assert pairs.exists(), pairs
    assert matches.exists(), matches

    assert features.exists(), features
    assert pairs.exists(), pairs
    assert matches.exists(), matches

    sfm_dir.mkdir(parents=True, exist_ok=True)
    database = sfm_dir / 'database.db'

    # create the reference model
    reference_model = sfm_dir / 'ref_w_known_poses'
    reference_model.mkdir(parents=True, exist_ok=True)
    # import_images(image_dir, database, camera_mode, image_list, image_options)

    # fixing the intrinsics if they are known
    # An initial value for the focal length is obtained from the EXIF data when importing the images. 
    # It is then refined during the reconstruction. It is possible to disable the refinement of intrinsics 
    # during BA (like in the triangulation), but you then need to first update the database 
    # database.db with the know intrinsics. This needs to be done for each camera 
    # (or a single one if shared intrinsics) via SQL queries, after import_images but 
    # before run_reconstruction.
    '''
    db = COLMAPDatabase.connect(reference_model)
    cam_id = 0
    cam_tuple = (cam.model_id, cam.width, cam.height, tuple(cam.params))
    # ATTENTIONL image_list must be in the same order than pose_w2c
    db.add_camera(*cam_tuple, camera_id=cam_id)
    for i, name in enumerate(image_list):
        pose_w2c_im = pose_w2c[i]
        qvec = rotmat2qvec(pose_w2c_im[:3, :3])
        tvec = pose_w2c_im[:3, -1].reshape(-1)
        # (image_id, name, camera_id, prior_q[0], prior_q[1], prior_q[2], 
        # prior_q[3], prior_t[0], prior_t[1], prior_t[2]))
        db.add_image(image_id=i+1, camera_id=cam_id, name=name, prior_q=qvec, prior_t=tvec)
    db.commit()
    db.close()
    '''
    images = {}
    cameras={}
    points = {}
    
    # create a file with known intrinsics and extrinsics
    cam_id = 1
    cameras[cam_id] = Camera(id=cam_id, model='SIMPLE_PINHOLE', width=cam.width, height=cam.height, 
                             params=cam.params)
    # ATTENTIONL image_list must be in the same order than pose_w2c

    for i, name in enumerate(image_list):
        pose_w2c_im = pose_w2c[i]
        qvec = rotmat2qvec(pose_w2c_im[:3, :3])
        tvec = pose_w2c_im[:3, -1].reshape(-1)
        images[i+1] = Image(id=i+1, qvec=qvec, tvec=tvec,
                            camera_id=cam_id, name=name, xys=[], point3D_ids=[])
    write_model(cameras, images, points, path=reference_model, ext=".txt")


    # triangulation from the created reference model

    database = sfm_dir / 'database.db'
    reference = pycolmap.Reconstruction(reference_model)

    image_ids = create_db_from_model(reference, database)
    import_features(image_ids, database, features)
    import_matches_modified(image_ids, database, pairs, matches,
                            min_match_score, skip_geometric_verification)
    if not skip_geometric_verification:
        if estimate_two_view_geometries:
            estimation_and_geometric_verification(database, pairs, verbose)
        else:
            geometric_verification(
                image_ids, reference, database, features, pairs, matches)
    reconstruction = run_triangulation(sfm_dir, database, image_dir, reference,
                                       verbose, mapper_options)
    logger.info('Finished the triangulation with statistics:\n%s',
                reconstruction.summary())
    return reconstruction