#!/usr/bin/env python3
"""
Task 1 evaluation script: CBCT tooth segmentation.

Usage
-----
    python score.py <input_dir> <output_dir>

Expected Codabench-style input layout
-------------------------------------
    <input_dir>/
      ref/
        STS24_Train_Labeled_0001.nii.gz
        ...
      res/
        STS24_Train_Labeled_0001.nii.gz
        ...

Reference and prediction files are matched by normalized case id. Common
prediction suffixes such as _Masks, _mask, _seg, _pred, _prediction and _0000
are ignored during matching.

Output
------
    <output_dir>/scores.json

The output JSON contains Dice, mIoU, NSD and IA metrics, plus case counts.

Artifact weighting
------------------
Cases with artifacts are weighted more heavily in the final aggregation:

    normal case weight   = 1.0
    artifact case weight = 2.0

Artifact cases are identified directly from the case filename. Any reference case
whose filename contains `_with-artifacts` (case-insensitive) is treated as an
artifact case and receives the higher aggregation weight.
"""
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import SimpleITK as sitk
import scipy.ndimage


NSD_TOLERANCE_MM = 2.0
ARTIFACT_CASE_WEIGHT = 2.0
DEFAULT_CASE_WEIGHT = 1.0

# neighbour_code_to_normals is a lookup table.
# For every binary neighbour code 
# (2x2x2 neighbourhood = 8 neighbours = 8 bits = 256 codes) 
# it contains the surface normals of the triangles (called "surfel" for 
# "surface element" in the following). The length of the normal 
# vector encodes the surfel area.
#
# created by compute_surface_area_lookup_table.ipynb using the 
# marching_cube algorithm, see e.g. https://en.wikipedia.org/wiki/Marching_cubes
# credit to: http://medicaldecathlon.com/files/Surface_distance_based_measures.ipynb
neighbour_code_to_normals = [
  [[0,0,0]],
  [[0.125,0.125,0.125]],
  [[-0.125,-0.125,0.125]],
  [[-0.25,-0.25,0.0],[0.25,0.25,-0.0]],
  [[0.125,-0.125,0.125]],
  [[-0.25,-0.0,-0.25],[0.25,0.0,0.25]],
  [[0.125,-0.125,0.125],[-0.125,-0.125,0.125]],
  [[0.5,0.0,-0.0],[0.25,0.25,0.25],[0.125,0.125,0.125]],
  [[-0.125,0.125,0.125]],
  [[0.125,0.125,0.125],[-0.125,0.125,0.125]],
  [[-0.25,0.0,0.25],[-0.25,0.0,0.25]],
  [[0.5,0.0,0.0],[-0.25,-0.25,0.25],[-0.125,-0.125,0.125]],
  [[0.25,-0.25,0.0],[0.25,-0.25,0.0]],
  [[0.5,0.0,0.0],[0.25,-0.25,0.25],[-0.125,0.125,-0.125]],
  [[-0.5,0.0,0.0],[-0.25,0.25,0.25],[-0.125,0.125,0.125]],
  [[0.5,0.0,0.0],[0.5,0.0,0.0]],
  [[0.125,-0.125,-0.125]],
  [[0.0,-0.25,-0.25],[0.0,0.25,0.25]],
  [[-0.125,-0.125,0.125],[0.125,-0.125,-0.125]],
  [[0.0,-0.5,0.0],[0.25,0.25,0.25],[0.125,0.125,0.125]],
  [[0.125,-0.125,0.125],[0.125,-0.125,-0.125]],
  [[0.0,0.0,-0.5],[0.25,0.25,0.25],[-0.125,-0.125,-0.125]],
  [[-0.125,-0.125,0.125],[0.125,-0.125,0.125],[0.125,-0.125,-0.125]],
  [[-0.125,-0.125,-0.125],[-0.25,-0.25,-0.25],[0.25,0.25,0.25],[0.125,0.125,0.125]],
  [[-0.125,0.125,0.125],[0.125,-0.125,-0.125]],
  [[0.0,-0.25,-0.25],[0.0,0.25,0.25],[-0.125,0.125,0.125]],
  [[-0.25,0.0,0.25],[-0.25,0.0,0.25],[0.125,-0.125,-0.125]],
  [[0.125,0.125,0.125],[0.375,0.375,0.375],[0.0,-0.25,0.25],[-0.25,0.0,0.25]],
  [[0.125,-0.125,-0.125],[0.25,-0.25,0.0],[0.25,-0.25,0.0]],
  [[0.375,0.375,0.375],[0.0,0.25,-0.25],[-0.125,-0.125,-0.125],[-0.25,0.25,0.0]],
  [[-0.5,0.0,0.0],[-0.125,-0.125,-0.125],[-0.25,-0.25,-0.25],[0.125,0.125,0.125]],
  [[-0.5,0.0,0.0],[-0.125,-0.125,-0.125],[-0.25,-0.25,-0.25]],
  [[0.125,-0.125,0.125]],
  [[0.125,0.125,0.125],[0.125,-0.125,0.125]],
  [[0.0,-0.25,0.25],[0.0,0.25,-0.25]],
  [[0.0,-0.5,0.0],[0.125,0.125,-0.125],[0.25,0.25,-0.25]],
  [[0.125,-0.125,0.125],[0.125,-0.125,0.125]],
  [[0.125,-0.125,0.125],[-0.25,-0.0,-0.25],[0.25,0.0,0.25]],
  [[0.0,-0.25,0.25],[0.0,0.25,-0.25],[0.125,-0.125,0.125]],
  [[-0.375,-0.375,0.375],[-0.0,0.25,0.25],[0.125,0.125,-0.125],[-0.25,-0.0,-0.25]],
  [[-0.125,0.125,0.125],[0.125,-0.125,0.125]],
  [[0.125,0.125,0.125],[0.125,-0.125,0.125],[-0.125,0.125,0.125]],
  [[-0.0,0.0,0.5],[-0.25,-0.25,0.25],[-0.125,-0.125,0.125]],
  [[0.25,0.25,-0.25],[0.25,0.25,-0.25],[0.125,0.125,-0.125],[-0.125,-0.125,0.125]],
  [[0.125,-0.125,0.125],[0.25,-0.25,0.0],[0.25,-0.25,0.0]],
  [[0.5,0.0,0.0],[0.25,-0.25,0.25],[-0.125,0.125,-0.125],[0.125,-0.125,0.125]],
  [[0.0,0.25,-0.25],[0.375,-0.375,-0.375],[-0.125,0.125,0.125],[0.25,0.25,0.0]],
  [[-0.5,0.0,0.0],[-0.25,-0.25,0.25],[-0.125,-0.125,0.125]],
  [[0.25,-0.25,0.0],[-0.25,0.25,0.0]],
  [[0.0,0.5,0.0],[-0.25,0.25,0.25],[0.125,-0.125,-0.125]],
  [[0.0,0.5,0.0],[0.125,-0.125,0.125],[-0.25,0.25,-0.25]],
  [[0.0,0.5,0.0],[0.0,-0.5,0.0]],
  [[0.25,-0.25,0.0],[-0.25,0.25,0.0],[0.125,-0.125,0.125]],
  [[-0.375,-0.375,-0.375],[-0.25,0.0,0.25],[-0.125,-0.125,-0.125],[-0.25,0.25,0.0]],
  [[0.125,0.125,0.125],[0.0,-0.5,0.0],[-0.25,-0.25,-0.25],[-0.125,-0.125,-0.125]],
  [[0.0,-0.5,0.0],[-0.25,-0.25,-0.25],[-0.125,-0.125,-0.125]],
  [[-0.125,0.125,0.125],[0.25,-0.25,0.0],[-0.25,0.25,0.0]],
  [[0.0,0.5,0.0],[0.25,0.25,-0.25],[-0.125,-0.125,0.125],[-0.125,-0.125,0.125]],
  [[-0.375,0.375,-0.375],[-0.25,-0.25,0.0],[-0.125,0.125,-0.125],[-0.25,0.0,0.25]],
  [[0.0,0.5,0.0],[0.25,0.25,-0.25],[-0.125,-0.125,0.125]],
  [[0.25,-0.25,0.0],[-0.25,0.25,0.0],[0.25,-0.25,0.0],[0.25,-0.25,0.0]],
  [[-0.25,-0.25,0.0],[-0.25,-0.25,0.0],[-0.125,-0.125,0.125]],
  [[0.125,0.125,0.125],[-0.25,-0.25,0.0],[-0.25,-0.25,0.0]],
  [[-0.25,-0.25,0.0],[-0.25,-0.25,0.0]],
  [[-0.125,-0.125,0.125]],
  [[0.125,0.125,0.125],[-0.125,-0.125,0.125]],
  [[-0.125,-0.125,0.125],[-0.125,-0.125,0.125]],
  [[-0.125,-0.125,0.125],[-0.25,-0.25,0.0],[0.25,0.25,-0.0]],
  [[0.0,-0.25,0.25],[0.0,-0.25,0.25]],
  [[0.0,0.0,0.5],[0.25,-0.25,0.25],[0.125,-0.125,0.125]],
  [[0.0,-0.25,0.25],[0.0,-0.25,0.25],[-0.125,-0.125,0.125]],
  [[0.375,-0.375,0.375],[0.0,-0.25,-0.25],[-0.125,0.125,-0.125],[0.25,0.25,0.0]],
  [[-0.125,-0.125,0.125],[-0.125,0.125,0.125]],
  [[0.125,0.125,0.125],[-0.125,-0.125,0.125],[-0.125,0.125,0.125]],
  [[-0.125,-0.125,0.125],[-0.25,0.0,0.25],[-0.25,0.0,0.25]],
  [[0.5,0.0,0.0],[-0.25,-0.25,0.25],[-0.125,-0.125,0.125],[-0.125,-0.125,0.125]],
  [[-0.0,0.5,0.0],[-0.25,0.25,-0.25],[0.125,-0.125,0.125]],
  [[-0.25,0.25,-0.25],[-0.25,0.25,-0.25],[-0.125,0.125,-0.125],[-0.125,0.125,-0.125]],
  [[-0.25,0.0,-0.25],[0.375,-0.375,-0.375],[0.0,0.25,-0.25],[-0.125,0.125,0.125]],
  [[0.5,0.0,0.0],[-0.25,0.25,-0.25],[0.125,-0.125,0.125]],
  [[-0.25,0.0,0.25],[0.25,0.0,-0.25]],
  [[-0.0,0.0,0.5],[-0.25,0.25,0.25],[-0.125,0.125,0.125]],
  [[-0.125,-0.125,0.125],[-0.25,0.0,0.25],[0.25,0.0,-0.25]],
  [[-0.25,-0.0,-0.25],[-0.375,0.375,0.375],[-0.25,-0.25,0.0],[-0.125,0.125,0.125]],
  [[0.0,0.0,-0.5],[0.25,0.25,-0.25],[-0.125,-0.125,0.125]],
  [[-0.0,0.0,0.5],[0.0,0.0,0.5]],
  [[0.125,0.125,0.125],[0.125,0.125,0.125],[0.25,0.25,0.25],[0.0,0.0,0.5]],
  [[0.125,0.125,0.125],[0.25,0.25,0.25],[0.0,0.0,0.5]],
  [[-0.25,0.0,0.25],[0.25,0.0,-0.25],[-0.125,0.125,0.125]],
  [[-0.0,0.0,0.5],[0.25,-0.25,0.25],[0.125,-0.125,0.125],[0.125,-0.125,0.125]],
  [[-0.25,0.0,0.25],[-0.25,0.0,0.25],[-0.25,0.0,0.25],[0.25,0.0,-0.25]],
  [[0.125,-0.125,0.125],[0.25,0.0,0.25],[0.25,0.0,0.25]],
  [[0.25,0.0,0.25],[-0.375,-0.375,0.375],[-0.25,0.25,0.0],[-0.125,-0.125,0.125]],
  [[-0.0,0.0,0.5],[0.25,-0.25,0.25],[0.125,-0.125,0.125]],
  [[0.125,0.125,0.125],[0.25,0.0,0.25],[0.25,0.0,0.25]],
  [[0.25,0.0,0.25],[0.25,0.0,0.25]],
  [[-0.125,-0.125,0.125],[0.125,-0.125,0.125]],
  [[0.125,0.125,0.125],[-0.125,-0.125,0.125],[0.125,-0.125,0.125]],
  [[-0.125,-0.125,0.125],[0.0,-0.25,0.25],[0.0,0.25,-0.25]],
  [[0.0,-0.5,0.0],[0.125,0.125,-0.125],[0.25,0.25,-0.25],[-0.125,-0.125,0.125]],
  [[0.0,-0.25,0.25],[0.0,-0.25,0.25],[0.125,-0.125,0.125]],
  [[0.0,0.0,0.5],[0.25,-0.25,0.25],[0.125,-0.125,0.125],[0.125,-0.125,0.125]],
  [[0.0,-0.25,0.25],[0.0,-0.25,0.25],[0.0,-0.25,0.25],[0.0,0.25,-0.25]],
  [[0.0,0.25,0.25],[0.0,0.25,0.25],[0.125,-0.125,-0.125]],
  [[-0.125,0.125,0.125],[0.125,-0.125,0.125],[-0.125,-0.125,0.125]],
  [[-0.125,0.125,0.125],[0.125,-0.125,0.125],[-0.125,-0.125,0.125],[0.125,0.125,0.125]],
  [[-0.0,0.0,0.5],[-0.25,-0.25,0.25],[-0.125,-0.125,0.125],[-0.125,-0.125,0.125]],
  [[0.125,0.125,0.125],[0.125,-0.125,0.125],[0.125,-0.125,-0.125]],
  [[-0.0,0.5,0.0],[-0.25,0.25,-0.25],[0.125,-0.125,0.125],[0.125,-0.125,0.125]],
  [[0.125,0.125,0.125],[-0.125,-0.125,0.125],[0.125,-0.125,-0.125]],
  [[0.0,-0.25,-0.25],[0.0,0.25,0.25],[0.125,0.125,0.125]],
  [[0.125,0.125,0.125],[0.125,-0.125,-0.125]],
  [[0.5,0.0,-0.0],[0.25,-0.25,-0.25],[0.125,-0.125,-0.125]],
  [[-0.25,0.25,0.25],[-0.125,0.125,0.125],[-0.25,0.25,0.25],[0.125,-0.125,-0.125]],
  [[0.375,-0.375,0.375],[0.0,0.25,0.25],[-0.125,0.125,-0.125],[-0.25,0.0,0.25]],
  [[0.0,-0.5,0.0],[-0.25,0.25,0.25],[-0.125,0.125,0.125]],
  [[-0.375,-0.375,0.375],[0.25,-0.25,0.0],[0.0,0.25,0.25],[-0.125,-0.125,0.125]],
  [[-0.125,0.125,0.125],[-0.25,0.25,0.25],[0.0,0.0,0.5]],
  [[0.125,0.125,0.125],[0.0,0.25,0.25],[0.0,0.25,0.25]],
  [[0.0,0.25,0.25],[0.0,0.25,0.25]],
  [[0.5,0.0,-0.0],[0.25,0.25,0.25],[0.125,0.125,0.125],[0.125,0.125,0.125]],
  [[0.125,-0.125,0.125],[-0.125,-0.125,0.125],[0.125,0.125,0.125]],
  [[-0.25,-0.0,-0.25],[0.25,0.0,0.25],[0.125,0.125,0.125]],
  [[0.125,0.125,0.125],[0.125,-0.125,0.125]],
  [[-0.25,-0.25,0.0],[0.25,0.25,-0.0],[0.125,0.125,0.125]],
  [[0.125,0.125,0.125],[-0.125,-0.125,0.125]],
  [[0.125,0.125,0.125],[0.125,0.125,0.125]],
  [[0.125,0.125,0.125]],
  [[0.125,0.125,0.125]],
  [[0.125,0.125,0.125],[0.125,0.125,0.125]],
  [[0.125,0.125,0.125],[-0.125,-0.125,0.125]],
  [[-0.25,-0.25,0.0],[0.25,0.25,-0.0],[0.125,0.125,0.125]],
  [[0.125,0.125,0.125],[0.125,-0.125,0.125]],
  [[-0.25,-0.0,-0.25],[0.25,0.0,0.25],[0.125,0.125,0.125]],
  [[0.125,-0.125,0.125],[-0.125,-0.125,0.125],[0.125,0.125,0.125]],
  [[0.5,0.0,-0.0],[0.25,0.25,0.25],[0.125,0.125,0.125],[0.125,0.125,0.125]],
  [[0.0,0.25,0.25],[0.0,0.25,0.25]],
  [[0.125,0.125,0.125],[0.0,0.25,0.25],[0.0,0.25,0.25]],
  [[-0.125,0.125,0.125],[-0.25,0.25,0.25],[0.0,0.0,0.5]],
  [[-0.375,-0.375,0.375],[0.25,-0.25,0.0],[0.0,0.25,0.25],[-0.125,-0.125,0.125]],
  [[0.0,-0.5,0.0],[-0.25,0.25,0.25],[-0.125,0.125,0.125]],
  [[0.375,-0.375,0.375],[0.0,0.25,0.25],[-0.125,0.125,-0.125],[-0.25,0.0,0.25]],
  [[-0.25,0.25,0.25],[-0.125,0.125,0.125],[-0.25,0.25,0.25],[0.125,-0.125,-0.125]],
  [[0.5,0.0,-0.0],[0.25,-0.25,-0.25],[0.125,-0.125,-0.125]],
  [[0.125,0.125,0.125],[0.125,-0.125,-0.125]],
  [[0.0,-0.25,-0.25],[0.0,0.25,0.25],[0.125,0.125,0.125]],
  [[0.125,0.125,0.125],[-0.125,-0.125,0.125],[0.125,-0.125,-0.125]],
  [[-0.0,0.5,0.0],[-0.25,0.25,-0.25],[0.125,-0.125,0.125],[0.125,-0.125,0.125]],
  [[0.125,0.125,0.125],[0.125,-0.125,0.125],[0.125,-0.125,-0.125]],
  [[-0.0,0.0,0.5],[-0.25,-0.25,0.25],[-0.125,-0.125,0.125],[-0.125,-0.125,0.125]],
  [[-0.125,0.125,0.125],[0.125,-0.125,0.125],[-0.125,-0.125,0.125],[0.125,0.125,0.125]],
  [[-0.125,0.125,0.125],[0.125,-0.125,0.125],[-0.125,-0.125,0.125]],
  [[0.0,0.25,0.25],[0.0,0.25,0.25],[0.125,-0.125,-0.125]],
  [[0.0,-0.25,-0.25],[0.0,0.25,0.25],[0.0,0.25,0.25],[0.0,0.25,0.25]],
  [[0.0,0.0,0.5],[0.25,-0.25,0.25],[0.125,-0.125,0.125],[0.125,-0.125,0.125]],
  [[0.0,-0.25,0.25],[0.0,-0.25,0.25],[0.125,-0.125,0.125]],
  [[0.0,-0.5,0.0],[0.125,0.125,-0.125],[0.25,0.25,-0.25],[-0.125,-0.125,0.125]],
  [[-0.125,-0.125,0.125],[0.0,-0.25,0.25],[0.0,0.25,-0.25]],
  [[0.125,0.125,0.125],[-0.125,-0.125,0.125],[0.125,-0.125,0.125]],
  [[-0.125,-0.125,0.125],[0.125,-0.125,0.125]],
  [[0.25,0.0,0.25],[0.25,0.0,0.25]],
  [[0.125,0.125,0.125],[0.25,0.0,0.25],[0.25,0.0,0.25]],
  [[-0.0,0.0,0.5],[0.25,-0.25,0.25],[0.125,-0.125,0.125]],
  [[0.25,0.0,0.25],[-0.375,-0.375,0.375],[-0.25,0.25,0.0],[-0.125,-0.125,0.125]],
  [[0.125,-0.125,0.125],[0.25,0.0,0.25],[0.25,0.0,0.25]],
  [[-0.25,-0.0,-0.25],[0.25,0.0,0.25],[0.25,0.0,0.25],[0.25,0.0,0.25]],
  [[-0.0,0.0,0.5],[0.25,-0.25,0.25],[0.125,-0.125,0.125],[0.125,-0.125,0.125]],
  [[-0.25,0.0,0.25],[0.25,0.0,-0.25],[-0.125,0.125,0.125]],
  [[0.125,0.125,0.125],[0.25,0.25,0.25],[0.0,0.0,0.5]],
  [[0.125,0.125,0.125],[0.125,0.125,0.125],[0.25,0.25,0.25],[0.0,0.0,0.5]],
  [[-0.0,0.0,0.5],[0.0,0.0,0.5]],
  [[0.0,0.0,-0.5],[0.25,0.25,-0.25],[-0.125,-0.125,0.125]],
  [[-0.25,-0.0,-0.25],[-0.375,0.375,0.375],[-0.25,-0.25,0.0],[-0.125,0.125,0.125]],
  [[-0.125,-0.125,0.125],[-0.25,0.0,0.25],[0.25,0.0,-0.25]],
  [[-0.0,0.0,0.5],[-0.25,0.25,0.25],[-0.125,0.125,0.125]],
  [[-0.25,0.0,0.25],[0.25,0.0,-0.25]],
  [[0.5,0.0,0.0],[-0.25,0.25,-0.25],[0.125,-0.125,0.125]],
  [[-0.25,0.0,-0.25],[0.375,-0.375,-0.375],[0.0,0.25,-0.25],[-0.125,0.125,0.125]],
  [[-0.25,0.25,-0.25],[-0.25,0.25,-0.25],[-0.125,0.125,-0.125],[-0.125,0.125,-0.125]],
  [[-0.0,0.5,0.0],[-0.25,0.25,-0.25],[0.125,-0.125,0.125]],
  [[0.5,0.0,0.0],[-0.25,-0.25,0.25],[-0.125,-0.125,0.125],[-0.125,-0.125,0.125]],
  [[-0.125,-0.125,0.125],[-0.25,0.0,0.25],[-0.25,0.0,0.25]],
  [[0.125,0.125,0.125],[-0.125,-0.125,0.125],[-0.125,0.125,0.125]],
  [[-0.125,-0.125,0.125],[-0.125,0.125,0.125]],
  [[0.375,-0.375,0.375],[0.0,-0.25,-0.25],[-0.125,0.125,-0.125],[0.25,0.25,0.0]],
  [[0.0,-0.25,0.25],[0.0,-0.25,0.25],[-0.125,-0.125,0.125]],
  [[0.0,0.0,0.5],[0.25,-0.25,0.25],[0.125,-0.125,0.125]],
  [[0.0,-0.25,0.25],[0.0,-0.25,0.25]],
  [[-0.125,-0.125,0.125],[-0.25,-0.25,0.0],[0.25,0.25,-0.0]],
  [[-0.125,-0.125,0.125],[-0.125,-0.125,0.125]],
  [[0.125,0.125,0.125],[-0.125,-0.125,0.125]],
  [[-0.125,-0.125,0.125]],
  [[-0.25,-0.25,0.0],[-0.25,-0.25,0.0]],
  [[0.125,0.125,0.125],[-0.25,-0.25,0.0],[-0.25,-0.25,0.0]],
  [[-0.25,-0.25,0.0],[-0.25,-0.25,0.0],[-0.125,-0.125,0.125]],
  [[-0.25,-0.25,0.0],[-0.25,-0.25,0.0],[-0.25,-0.25,0.0],[0.25,0.25,-0.0]],
  [[0.0,0.5,0.0],[0.25,0.25,-0.25],[-0.125,-0.125,0.125]],
  [[-0.375,0.375,-0.375],[-0.25,-0.25,0.0],[-0.125,0.125,-0.125],[-0.25,0.0,0.25]],
  [[0.0,0.5,0.0],[0.25,0.25,-0.25],[-0.125,-0.125,0.125],[-0.125,-0.125,0.125]],
  [[-0.125,0.125,0.125],[0.25,-0.25,0.0],[-0.25,0.25,0.0]],
  [[0.0,-0.5,0.0],[-0.25,-0.25,-0.25],[-0.125,-0.125,-0.125]],
  [[0.125,0.125,0.125],[0.0,-0.5,0.0],[-0.25,-0.25,-0.25],[-0.125,-0.125,-0.125]],
  [[-0.375,-0.375,-0.375],[-0.25,0.0,0.25],[-0.125,-0.125,-0.125],[-0.25,0.25,0.0]],
  [[0.25,-0.25,0.0],[-0.25,0.25,0.0],[0.125,-0.125,0.125]],
  [[0.0,0.5,0.0],[0.0,-0.5,0.0]],
  [[0.0,0.5,0.0],[0.125,-0.125,0.125],[-0.25,0.25,-0.25]],
  [[0.0,0.5,0.0],[-0.25,0.25,0.25],[0.125,-0.125,-0.125]],
  [[0.25,-0.25,0.0],[-0.25,0.25,0.0]],
  [[-0.5,0.0,0.0],[-0.25,-0.25,0.25],[-0.125,-0.125,0.125]],
  [[0.0,0.25,-0.25],[0.375,-0.375,-0.375],[-0.125,0.125,0.125],[0.25,0.25,0.0]],
  [[0.5,0.0,0.0],[0.25,-0.25,0.25],[-0.125,0.125,-0.125],[0.125,-0.125,0.125]],
  [[0.125,-0.125,0.125],[0.25,-0.25,0.0],[0.25,-0.25,0.0]],
  [[0.25,0.25,-0.25],[0.25,0.25,-0.25],[0.125,0.125,-0.125],[-0.125,-0.125,0.125]],
  [[-0.0,0.0,0.5],[-0.25,-0.25,0.25],[-0.125,-0.125,0.125]],
  [[0.125,0.125,0.125],[0.125,-0.125,0.125],[-0.125,0.125,0.125]],
  [[-0.125,0.125,0.125],[0.125,-0.125,0.125]],
  [[-0.375,-0.375,0.375],[-0.0,0.25,0.25],[0.125,0.125,-0.125],[-0.25,-0.0,-0.25]],
  [[0.0,-0.25,0.25],[0.0,0.25,-0.25],[0.125,-0.125,0.125]],
  [[0.125,-0.125,0.125],[-0.25,-0.0,-0.25],[0.25,0.0,0.25]],
  [[0.125,-0.125,0.125],[0.125,-0.125,0.125]],
  [[0.0,-0.5,0.0],[0.125,0.125,-0.125],[0.25,0.25,-0.25]],
  [[0.0,-0.25,0.25],[0.0,0.25,-0.25]],
  [[0.125,0.125,0.125],[0.125,-0.125,0.125]],
  [[0.125,-0.125,0.125]],
  [[-0.5,0.0,0.0],[-0.125,-0.125,-0.125],[-0.25,-0.25,-0.25]],
  [[-0.5,0.0,0.0],[-0.125,-0.125,-0.125],[-0.25,-0.25,-0.25],[0.125,0.125,0.125]],
  [[0.375,0.375,0.375],[0.0,0.25,-0.25],[-0.125,-0.125,-0.125],[-0.25,0.25,0.0]],
  [[0.125,-0.125,-0.125],[0.25,-0.25,0.0],[0.25,-0.25,0.0]],
  [[0.125,0.125,0.125],[0.375,0.375,0.375],[0.0,-0.25,0.25],[-0.25,0.0,0.25]],
  [[-0.25,0.0,0.25],[-0.25,0.0,0.25],[0.125,-0.125,-0.125]],
  [[0.0,-0.25,-0.25],[0.0,0.25,0.25],[-0.125,0.125,0.125]],
  [[-0.125,0.125,0.125],[0.125,-0.125,-0.125]],
  [[-0.125,-0.125,-0.125],[-0.25,-0.25,-0.25],[0.25,0.25,0.25],[0.125,0.125,0.125]],
  [[-0.125,-0.125,0.125],[0.125,-0.125,0.125],[0.125,-0.125,-0.125]],
  [[0.0,0.0,-0.5],[0.25,0.25,0.25],[-0.125,-0.125,-0.125]],
  [[0.125,-0.125,0.125],[0.125,-0.125,-0.125]],
  [[0.0,-0.5,0.0],[0.25,0.25,0.25],[0.125,0.125,0.125]],
  [[-0.125,-0.125,0.125],[0.125,-0.125,-0.125]],
  [[0.0,-0.25,-0.25],[0.0,0.25,0.25]],
  [[0.125,-0.125,-0.125]],
  [[0.5,0.0,0.0],[0.5,0.0,0.0]],
  [[-0.5,0.0,0.0],[-0.25,0.25,0.25],[-0.125,0.125,0.125]],
  [[0.5,0.0,0.0],[0.25,-0.25,0.25],[-0.125,0.125,-0.125]],
  [[0.25,-0.25,0.0],[0.25,-0.25,0.0]],
  [[0.5,0.0,0.0],[-0.25,-0.25,0.25],[-0.125,-0.125,0.125]],
  [[-0.25,0.0,0.25],[-0.25,0.0,0.25]],
  [[0.125,0.125,0.125],[-0.125,0.125,0.125]],
  [[-0.125,0.125,0.125]],
  [[0.5,0.0,-0.0],[0.25,0.25,0.25],[0.125,0.125,0.125]],
  [[0.125,-0.125,0.125],[-0.125,-0.125,0.125]],
  [[-0.25,-0.0,-0.25],[0.25,0.0,0.25]],
  [[0.125,-0.125,0.125]],
  [[-0.25,-0.25,0.0],[0.25,0.25,-0.0]],
  [[-0.125,-0.125,0.125]],
  [[0.125,0.125,0.125]],
  [[0,0,0]]]


def compute_surface_distances(mask_gt, mask_pred, spacing_mm):
  """Compute closest distances from all surface points to the other surface.

  Finds all surface elements "surfels" in the ground truth mask `mask_gt` and
  the predicted mask `mask_pred`, computes their area in mm^2 and the distance
  to the closest point on the other surface. It returns two sorted lists of
  distances together with the corresponding surfel areas. If one of the masks
  is empty, the corresponding lists are empty and all distances in the other
  list are `inf` 
  
  Args:
    mask_gt: 3-dim Numpy array of type bool. The ground truth mask.
    mask_pred: 3-dim Numpy array of type bool. The predicted mask.
    spacing_mm: 3-element list-like structure. Voxel spacing in x0, x1 and x2
        direction 

  Returns:
    A dict with 
    "distances_gt_to_pred": 1-dim numpy array of type float. The distances in mm
        from all ground truth surface elements to the predicted surface, 
        sorted from smallest to largest
    "distances_pred_to_gt": 1-dim numpy array of type float. The distances in mm
        from all predicted surface elements to the ground truth surface, 
        sorted from smallest to largest 
    "surfel_areas_gt": 1-dim numpy array of type float. The area in mm^2 of 
        the ground truth surface elements in the same order as 
        distances_gt_to_pred
    "surfel_areas_pred": 1-dim numpy array of type float. The area in mm^2 of 
        the predicted surface elements in the same order as 
        distances_pred_to_gt
       
  """
  
  # compute the area for all 256 possible surface elements 
  # (given a 2x2x2 neighbourhood) according to the spacing_mm
  neighbour_code_to_surface_area = np.zeros([256])
  for code in range(256):
    normals = np.array(neighbour_code_to_normals[code])
    sum_area = 0
    for normal_idx in range(normals.shape[0]):
      # normal vector
      n = np.zeros([3])
      n[0] = normals[normal_idx,0] * spacing_mm[1] * spacing_mm[2]
      n[1] = normals[normal_idx,1] * spacing_mm[0] * spacing_mm[2]
      n[2] = normals[normal_idx,2] * spacing_mm[0] * spacing_mm[1]
      area = np.linalg.norm(n)
      sum_area += area
    neighbour_code_to_surface_area[code] = sum_area

  # compute the bounding box of the masks to trim
  # the volume to the smallest possible processing subvolume
  mask_all = mask_gt | mask_pred
  bbox_min = np.zeros(3, np.int64)
  bbox_max = np.zeros(3, np.int64)

  # max projection to the x0-axis
  proj_0 = np.max(np.max(mask_all, axis=2), axis=1)
  idx_nonzero_0 = np.nonzero(proj_0)[0]
  if len(idx_nonzero_0) == 0:
    return {"distances_gt_to_pred":  np.array([]), 
            "distances_pred_to_gt":  np.array([]), 
            "surfel_areas_gt":       np.array([]), 
            "surfel_areas_pred":     np.array([])}
    
  bbox_min[0] = np.min(idx_nonzero_0)
  bbox_max[0] = np.max(idx_nonzero_0)

  # max projection to the x1-axis
  proj_1 = np.max(np.max(mask_all, axis=2), axis=0)
  idx_nonzero_1 = np.nonzero(proj_1)[0]
  bbox_min[1] = np.min(idx_nonzero_1)
  bbox_max[1] = np.max(idx_nonzero_1)

  # max projection to the x2-axis
  proj_2 = np.max(np.max(mask_all, axis=1), axis=0)
  idx_nonzero_2 = np.nonzero(proj_2)[0]
  bbox_min[2] = np.min(idx_nonzero_2)
  bbox_max[2] = np.max(idx_nonzero_2)

#  print("bounding box min = {}".format(bbox_min))
#  print("bounding box max = {}".format(bbox_max))

  # crop the processing subvolume.
  # we need to zeropad the cropped region with 1 voxel at the lower, 
  # the right and the back side. This is required to obtain the "full" 
  # convolution result with the 2x2x2 kernel
  cropmask_gt = np.zeros((bbox_max - bbox_min)+2, np.uint8)
  cropmask_pred = np.zeros((bbox_max - bbox_min)+2, np.uint8)

  cropmask_gt[0:-1, 0:-1, 0:-1] = mask_gt[bbox_min[0]:bbox_max[0]+1,
                                          bbox_min[1]:bbox_max[1]+1,
                                          bbox_min[2]:bbox_max[2]+1]

  cropmask_pred[0:-1, 0:-1, 0:-1] = mask_pred[bbox_min[0]:bbox_max[0]+1,
                                              bbox_min[1]:bbox_max[1]+1,
                                              bbox_min[2]:bbox_max[2]+1]

  # compute the neighbour code (local binary pattern) for each voxel
  # the resultsing arrays are spacially shifted by minus half a voxel in each axis.
  # i.e. the points are located at the corners of the original voxels
  kernel = np.array([[[128,64],
                      [32,16]],
                     [[8,4],
                      [2,1]]])
  neighbour_code_map_gt = scipy.ndimage.correlate(cropmask_gt.astype(np.uint8), kernel, mode="constant", cval=0) 
  neighbour_code_map_pred = scipy.ndimage.correlate(cropmask_pred.astype(np.uint8), kernel, mode="constant", cval=0) 

  # create masks with the surface voxels
  borders_gt   = ((neighbour_code_map_gt != 0) & (neighbour_code_map_gt != 255))
  borders_pred = ((neighbour_code_map_pred != 0) & (neighbour_code_map_pred != 255))

  # compute the distance transform (closest distance of each voxel to the surface voxels)
  if borders_gt.any():
    distmap_gt = scipy.ndimage.distance_transform_edt(~borders_gt, sampling=spacing_mm)
  else:
    distmap_gt = np.inf * np.ones(borders_gt.shape)

  if borders_pred.any():  
    distmap_pred = scipy.ndimage.distance_transform_edt(~borders_pred, sampling=spacing_mm)
  else:
    distmap_pred = np.inf * np.ones(borders_pred.shape)

  # compute the area of each surface element
  surface_area_map_gt = neighbour_code_to_surface_area[neighbour_code_map_gt]
  surface_area_map_pred = neighbour_code_to_surface_area[neighbour_code_map_pred]

  # create a list of all surface elements with distance and area
  distances_gt_to_pred = distmap_pred[borders_gt]
  distances_pred_to_gt = distmap_gt[borders_pred]
  surfel_areas_gt   = surface_area_map_gt[borders_gt]
  surfel_areas_pred = surface_area_map_pred[borders_pred]

  # sort them by distance
  if distances_gt_to_pred.shape != (0,):
    sorted_surfels_gt = np.array(sorted(zip(distances_gt_to_pred, surfel_areas_gt)))
    distances_gt_to_pred = sorted_surfels_gt[:,0]
    surfel_areas_gt      = sorted_surfels_gt[:,1]

  if distances_pred_to_gt.shape != (0,):
    sorted_surfels_pred = np.array(sorted(zip(distances_pred_to_gt, surfel_areas_pred)))
    distances_pred_to_gt = sorted_surfels_pred[:,0]
    surfel_areas_pred    = sorted_surfels_pred[:,1]


  return {"distances_gt_to_pred":  distances_gt_to_pred, 
          "distances_pred_to_gt":  distances_pred_to_gt, 
          "surfel_areas_gt":       surfel_areas_gt, 
          "surfel_areas_pred":     surfel_areas_pred}


def compute_average_surface_distance(surface_distances):
  distances_gt_to_pred = surface_distances["distances_gt_to_pred"]
  distances_pred_to_gt = surface_distances["distances_pred_to_gt"]
  surfel_areas_gt      = surface_distances["surfel_areas_gt"]
  surfel_areas_pred    = surface_distances["surfel_areas_pred"]
  average_distance_gt_to_pred = np.sum( distances_gt_to_pred * surfel_areas_gt) / np.sum(surfel_areas_gt)
  average_distance_pred_to_gt = np.sum( distances_pred_to_gt * surfel_areas_pred) / np.sum(surfel_areas_pred)
  return (average_distance_gt_to_pred, average_distance_pred_to_gt)

def compute_robust_hausdorff(surface_distances, percent):
  distances_gt_to_pred = surface_distances["distances_gt_to_pred"]
  distances_pred_to_gt = surface_distances["distances_pred_to_gt"]
  surfel_areas_gt      = surface_distances["surfel_areas_gt"]
  surfel_areas_pred    = surface_distances["surfel_areas_pred"]
  if len(distances_gt_to_pred) > 0:
    surfel_areas_cum_gt   = np.cumsum(surfel_areas_gt) / np.sum(surfel_areas_gt)
    idx = np.searchsorted(surfel_areas_cum_gt, percent/100.0)
    perc_distance_gt_to_pred = distances_gt_to_pred[min(idx, len(distances_gt_to_pred)-1)]
  else:
    perc_distance_gt_to_pred = np.inf
    
  if len(distances_pred_to_gt) > 0:
    surfel_areas_cum_pred = np.cumsum(surfel_areas_pred) / np.sum(surfel_areas_pred)
    idx = np.searchsorted(surfel_areas_cum_pred, percent/100.0)
    perc_distance_pred_to_gt = distances_pred_to_gt[min(idx, len(distances_pred_to_gt)-1)]
  else:
    perc_distance_pred_to_gt = np.inf
    
  return max( perc_distance_gt_to_pred, perc_distance_pred_to_gt)

def compute_surface_overlap_at_tolerance(surface_distances, tolerance_mm):
  distances_gt_to_pred = surface_distances["distances_gt_to_pred"]
  distances_pred_to_gt = surface_distances["distances_pred_to_gt"]
  surfel_areas_gt      = surface_distances["surfel_areas_gt"]
  surfel_areas_pred    = surface_distances["surfel_areas_pred"]
  rel_overlap_gt   = np.sum(surfel_areas_gt[distances_gt_to_pred <= tolerance_mm]) / np.sum(surfel_areas_gt)
  rel_overlap_pred = np.sum(surfel_areas_pred[distances_pred_to_gt <= tolerance_mm]) / np.sum(surfel_areas_pred)
  return (rel_overlap_gt, rel_overlap_pred)

def compute_surface_dice_at_tolerance(surface_distances, tolerance_mm):
  distances_gt_to_pred = surface_distances["distances_gt_to_pred"]
  distances_pred_to_gt = surface_distances["distances_pred_to_gt"]
  surfel_areas_gt      = surface_distances["surfel_areas_gt"]
  surfel_areas_pred    = surface_distances["surfel_areas_pred"]
  overlap_gt   = np.sum(surfel_areas_gt[distances_gt_to_pred <= tolerance_mm])
  overlap_pred = np.sum(surfel_areas_pred[distances_pred_to_gt <= tolerance_mm])
  surface_dice = (overlap_gt + overlap_pred) / (
      np.sum(surfel_areas_gt) + np.sum(surfel_areas_pred))
  return surface_dice


def compute_dice_coefficient(mask_gt, mask_pred):
  """Compute soerensen-dice coefficient.

  compute the soerensen-dice coefficient between the ground truth mask `mask_gt`
  and the predicted mask `mask_pred`. 
  
  Args:
    mask_gt: 3-dim Numpy array of type bool. The ground truth mask.
    mask_pred: 3-dim Numpy array of type bool. The predicted mask.

  Returns:
    the dice coeffcient as float. If both masks are empty, the result is NaN
  """
  volume_sum = mask_gt.sum() + mask_pred.sum()
  if volume_sum == 0:
    return np.nan
  volume_intersect = (mask_gt & mask_pred).sum()
  return 2*volume_intersect / volume_sum


def robust_mean(values: Iterable[float]) -> float:
    valid = [float(v) for v in values if not np.isnan(v)]
    return float(np.mean(valid)) if valid else 0.0


def weighted_mean(values: Iterable[float], weights: Iterable[float]) -> float:
    pairs = [(float(v), float(w)) for v, w in zip(values, weights) if not np.isnan(v) and w > 0]
    if not pairs:
        return 0.0
    numerator = sum(value * weight for value, weight in pairs)
    denominator = sum(weight for _, weight in pairs)
    return float(numerator / denominator) if denominator else 0.0


def normalize_case_id(path: Path) -> str:
    name = path.name
    for suffix in (".nii.gz", ".nii", ".mha", ".mhd"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break
    name = re.sub(r"(_0000|_seg|_mask|_masks|_label|_labels|_pred|_prediction)$", "", name, flags=re.I)
    match = re.search(r"(\d+)(?:_with-artifacts)?$", name, flags=re.I)
    return match.group(1).lstrip("0") or "0" if match else name.lower()


def normalize_case_name(value) -> str:
    text = "" if value is None else str(value).strip()
    for suffix in (".nii.gz", ".nii", ".mha", ".mhd"):
        if text.lower().endswith(suffix):
            text = text[: -len(suffix)]
            break
    text = re.sub(r"(_0000|_seg|_mask|_masks|_label|_labels|_pred|_prediction)$", "", text, flags=re.I)
    return text.lower()


def is_artifact_case(path: Path) -> bool:
    normalized_name = normalize_case_name(path.name)
    return "_with-artifacts" in normalized_name or "_with_artifacts" in normalized_name


def case_weight(path: Path) -> float:
    return ARTIFACT_CASE_WEIGHT if is_artifact_case(path) else DEFAULT_CASE_WEIGHT


def build_file_map(directory: Path) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    if not directory.is_dir():
        return mapping
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.name.lower().endswith((".nii.gz", ".nii", ".mha", ".mhd")):
            case_id = normalize_case_id(path)
            mapping.setdefault(case_id, path)
    return mapping


def read_image(path: Path) -> sitk.Image:
    return sitk.ReadImage(str(path))


def resample_to_reference(image: sitk.Image, reference: sitk.Image) -> sitk.Image:
    same_grid = (
        image.GetDimension() == reference.GetDimension()
        and image.GetSize() == reference.GetSize()
        and np.allclose(image.GetSpacing(), reference.GetSpacing())
        and np.allclose(image.GetOrigin(), reference.GetOrigin())
        and np.allclose(image.GetDirection(), reference.GetDirection())
    )
    if same_grid:
        return image
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(image)


def calculate_iou(gt: np.ndarray, pred: np.ndarray) -> float:
    if not gt.any() and not pred.any():
        return 1.0
    union = np.logical_or(gt, pred).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(gt, pred).sum() / union)


def calculate_nsd(gt: np.ndarray, pred: np.ndarray, spacing_zyx: List[float]) -> float:
    if not gt.any() and not pred.any():
        return 1.0
    surface_distances = compute_surface_distances(gt, pred, spacing_mm=spacing_zyx)
    gt_area = np.sum(surface_distances["surfel_areas_gt"]) if len(surface_distances["surfel_areas_gt"]) else 0
    pred_area = np.sum(surface_distances["surfel_areas_pred"]) if len(surface_distances["surfel_areas_pred"]) else 0
    if gt_area == 0 and pred_area == 0:
        return 0.0 if gt.any() or pred.any() else 1.0
    value = compute_surface_dice_at_tolerance(surface_distances, NSD_TOLERANCE_MM)
    return 0.0 if np.isnan(value) else float(value)


def add_missing_case_scores(acc: Dict[str, List[float]], ref_path: Path, weight: float) -> None:
    acc["dice_image"].append(0.0)
    acc["miou_image"].append(0.0)
    acc["nsd_image"].append(0.0)
    acc["ia"].append(0.0)
    acc["image_weights"].append(weight)
    try:
        ref_np = sitk.GetArrayFromImage(read_image(ref_path))
        gt_labels = np.unique(ref_np[ref_np > 0])
        acc["dice_instance"].extend([0.0] * len(gt_labels))
        acc["miou_instance"].extend([0.0] * len(gt_labels))
        acc["nsd_instance"].extend([0.0] * len(gt_labels))
        acc["instance_weights"].extend([weight] * len(gt_labels))
    except Exception:
        pass


def evaluate_case(ref_path: Path, pred_path: Path, acc: Dict[str, List[float]], weight: float) -> bool:
    try:
        ref_img = read_image(ref_path)
        pred_img = read_image(pred_path)
        if ref_img.GetDimension() != 3 or pred_img.GetDimension() != 3:
            raise ValueError("reference and prediction must both be 3D images")
        pred_img = resample_to_reference(pred_img, ref_img)
        ref_np = sitk.GetArrayFromImage(ref_img)
        pred_np = sitk.GetArrayFromImage(pred_img)
        if ref_np.shape != pred_np.shape:
            raise ValueError(f"shape mismatch after resampling: ref={ref_np.shape}, pred={pred_np.shape}")
    except Exception as exc:
        print(f"Case failed ({ref_path.name} vs {pred_path.name}): {exc}", file=sys.stderr)
        add_missing_case_scores(acc, ref_path, weight)
        return False

    spacing_xyz = list(ref_img.GetSpacing())
    spacing_zyx = [spacing_xyz[2], spacing_xyz[1], spacing_xyz[0]]
    gt_binary = ref_np > 0
    pred_binary = pred_np > 0

    dice_img = compute_dice_coefficient(gt_binary, pred_binary)
    acc["dice_image"].append(0.0 if np.isnan(dice_img) else float(dice_img))
    acc["miou_image"].append(calculate_iou(gt_binary, pred_binary))
    acc["nsd_image"].append(calculate_nsd(gt_binary, pred_binary, spacing_zyx))
    acc["image_weights"].append(weight)

    gt_labels = np.unique(ref_np[ref_np > 0])
    pred_labels = np.unique(pred_np[pred_np > 0])
    matched_iou_count = 0

    for label in gt_labels:
        gt_instance = ref_np == label
        pred_instance = pred_np == label
        dice_instance = compute_dice_coefficient(gt_instance, pred_instance)
        iou_instance = calculate_iou(gt_instance, pred_instance)
        nsd_instance = calculate_nsd(gt_instance, pred_instance, spacing_zyx)

        acc["dice_instance"].append(0.0 if np.isnan(dice_instance) else float(dice_instance))
        acc["miou_instance"].append(iou_instance)
        acc["nsd_instance"].append(nsd_instance)
        acc["instance_weights"].append(weight)
        if iou_instance >= 0.5:
            matched_iou_count += 1

    union_instance_count = len(set(gt_labels.tolist()).union(set(pred_labels.tolist())))
    if union_instance_count == 0:
        acc["ia"].append(1.0)
    else:
        acc["ia"].append(float(matched_iou_count) / float(union_instance_count))
    return True


def write_scores(output_dir: Path, scores: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "scores.json").open("w", encoding="utf-8") as f:
        json.dump(scores, f, indent=4)


def run_evaluation(input_dir: Path, output_dir: Path) -> None:
    ref_dir = input_dir / "ref"
    res_dir = input_dir / "res"
    acc: Dict[str, List[float]] = {
        "dice_instance": [],
        "miou_instance": [],
        "nsd_instance": [],
        "dice_image": [],
        "miou_image": [],
        "nsd_image": [],
        "ia": [],
        "image_weights": [],
        "instance_weights": [],
    }

    if not ref_dir.is_dir():
        write_scores(output_dir, {"error": f"Reference directory not found: {ref_dir}"})
        return

    ref_files = build_file_map(ref_dir)
    pred_files = build_file_map(res_dir)
    if not ref_files:
        write_scores(output_dir, {"error": "No reference NIfTI files found."})
        return

    successful_cases = 0
    failed_cases = 0
    artifact_cases = 0
    for case_id, ref_path in sorted(ref_files.items()):
        weight = case_weight(ref_path)
        if weight > DEFAULT_CASE_WEIGHT:
            artifact_cases += 1
        pred_path: Optional[Path] = pred_files.get(case_id)
        if pred_path is None:
            print(f"Missing prediction for case {case_id}: {ref_path.name}", file=sys.stderr)
            add_missing_case_scores(acc, ref_path, weight)
            failed_cases += 1
            continue
        if evaluate_case(ref_path, pred_path, acc, weight):
            successful_cases += 1
        else:
            failed_cases += 1

    scores = {
        "Dice_Instance": weighted_mean(acc["dice_instance"], acc["instance_weights"]),
        "mIoU_Instance": weighted_mean(acc["miou_instance"], acc["instance_weights"]),
        "NSD_Instance": weighted_mean(acc["nsd_instance"], acc["instance_weights"]),
        "Dice_Image": weighted_mean(acc["dice_image"], acc["image_weights"]),
        "mIoU_Image": weighted_mean(acc["miou_image"], acc["image_weights"]),
        "NSD_Image": weighted_mean(acc["nsd_image"], acc["image_weights"]),
        "IA": weighted_mean(acc["ia"], acc["image_weights"]),
        "Num_Evaluated_Cases": len(ref_files),
        "Num_Successful_Cases": successful_cases,
        "Num_Failed_Cases": failed_cases,
        "Num_Artifact_Cases": artifact_cases,
        "Artifact_Case_Weight": ARTIFACT_CASE_WEIGHT,
        "Default_Case_Weight": DEFAULT_CASE_WEIGHT,
    }
    metric_values = [
        scores["Dice_Instance"],
        scores["mIoU_Instance"],
        scores["NSD_Instance"],
        scores["Dice_Image"],
        scores["mIoU_Image"],
        scores["NSD_Image"],
        scores["IA"],
    ]
    scores["All_Average"] = float(sum(metric_values) / len(metric_values))
    write_scores(output_dir, scores)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python score.py <input_dir> <output_dir>", file=sys.stderr)
        sys.exit(1)
    start = time.time()
    run_evaluation(Path(sys.argv[1]), Path(sys.argv[2]))
    print(f"Evaluation completed in {time.time() - start:.2f} seconds.")

