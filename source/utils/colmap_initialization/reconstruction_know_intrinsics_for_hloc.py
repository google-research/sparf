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

"""Functions to use in HLOC, to obtain a COLMAP reconstruction, considering
known intrinsics"""

from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import pycolmap
import os
import numpy as np
from tqdm import tqdm
import h5py
import torch
from pathlib import Path
import sys 

sys.path.append(str(Path(__file__).parent / '../../../third_party/Hierarchical-Localization'))


from hloc import logger
from hloc.utils.database import COLMAPDatabase
from hloc.triangulation import (
    import_features, estimation_and_geometric_verification)
from hloc.reconstruction import (create_empty_db, get_image_ids, run_reconstruction)
from hloc.utils.io import find_pair

def reconstruction_w_known_intrinsics(sfm_dir: Path,
                                      image_dir: Path,
                                      pairs: Path,
                                      features: Path,
                                      matches: Path,
                                      cam: torch.Tensor, 
                                      image_list: List[str],
                                      camera_mode: pycolmap.CameraMode = pycolmap.CameraMode.AUTO,
                                      verbose_features: bool = False, 
                                      verbose: bool = False,
                                      skip_geometric_verification: bool = False,
                                      min_match_score: Optional[float] = None,
                                      image_options: Optional[Dict[str, Any]] = None,
                                      mapper_options: Optional[Dict[str, Any]] = None,
                                      ) -> pycolmap.Reconstruction:
    # https://github.com/colmap/pycolmap/issues/61
    assert features.exists(), features
    assert pairs.exists(), pairs
    assert matches.exists(), matches

    sfm_dir.mkdir(parents=True, exist_ok=True)
    database = sfm_dir / 'database.db'

    create_empty_db(database)
    # import_images(image_dir, database, camera_mode, image_list, image_options)

    # fixing the intrinsics if they are known
    # An initial value for the focal length is obtained from the EXIF data when importing the images. 
    # It is then refined during the reconstruction. It is possible to disable the refinement of intrinsics 
    # during BA (like in the triangulation), but you then need to first update the database 
    # database.db with the know intrinsics. This needs to be done for each camera 
    # (or a single one if shared intrinsics) via SQL queries, after import_images but 
    # before run_reconstruction.
    db = COLMAPDatabase.connect(database)
    cam_id = 1
    cam_tuple = (cam.model_id, cam.width, cam.height, tuple(cam.params))
    db.add_camera(*cam_tuple, camera_id=cam_id)
    for i, name in enumerate(image_list):
        db.add_image(name, cam_id, image_id=i+1)
    db.commit()
    db.close()

    image_ids = get_image_ids(database)
    import_features(image_ids, database, features)

    import_matches_modified(image_ids, database, pairs, matches,
                            min_match_score, skip_geometric_verification)
    # to fit pdcnet matches 
    if not skip_geometric_verification:
        estimation_and_geometric_verification(database, pairs, verbose)
    reconstruction = run_reconstruction(
        sfm_dir, database, image_dir, verbose, mapper_options)
    if reconstruction is not None:
        logger.info(f'Reconstruction statistics:\n{reconstruction.summary()}'
                    + f'\n\tnum_input_images = {len(image_ids)}')
    return reconstruction


def get_matches_modified(path: Path, name0: str, name1: str) -> Tuple[np.ndarray]:
    # modified from hloc/utils/io.py
    # modified to fit pdcnet and accept 'matches' 
    with h5py.File(str(path), 'r', libver='latest') as hfile:
        pair, reverse = find_pair(hfile, name0, name1)
        if 'matches' in hfile[pair].keys():
            matches = hfile[pair]['matches'].__array__()  # already the indices of keypoints in both images
            scores = np.ones_like(matches).astype(np.float32)
        else:
            matches = hfile[pair]['matches0'].__array__()
            scores = hfile[pair]['matching_scores0'].__array__()
            idx = np.where(matches != -1)[0]
            matches = np.stack([idx, matches[idx]], -1)
            scores = scores[idx]
    if reverse:
        matches = np.flip(matches, -1)
    return matches, scores


def import_matches_modified(image_ids: Dict[str, int],
                            database_path: Path,
                            pairs_path: Path,
                            matches_path: Path,
                            min_match_score: Optional[float] = None,
                            skip_geometric_verification: bool = False):
    # modified from  hloc/triangulation.py
    # here, modifies get_matches to fit pdncet
    logger.info('Importing matches into the database...')

    with open(str(pairs_path), 'r') as f:
        pairs = [p.split() for p in f.readlines()]

    db = COLMAPDatabase.connect(database_path)

    matched = set()
    for name0, name1 in tqdm(pairs):
        id0, id1 = image_ids[name0], image_ids[name1]
        if len({(id0, id1), (id1, id0)} & matched) > 0:
            continue
        matches, scores = get_matches_modified(matches_path, name0, name1)
        if min_match_score:
            matches = matches[scores > min_match_score]
        db.add_matches(id0, id1, matches)
        matched |= {(id0, id1), (id1, id0)}

        if skip_geometric_verification:
            db.add_two_view_geometry(id0, id1, matches)

    db.commit()
    db.close()
    return 