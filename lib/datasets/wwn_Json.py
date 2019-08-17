# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Functions for common roidb manipulations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import six
import logging
import numpy as np

import utils.boxes as box_utils
import utils.keypoints as keypoint_utils
import utils.segms as segm_utils
import utils.blob as blob_utils
from core.config import cfg
from .json_dataset import JsonDataset

logger = logging.getLogger(__name__)


def Json_result(dataset_name, Is_debuglearn=False):
    """Load and concatenate roidbs for one or more datasets, along with optional
    object proposals. The roidb entries are then prepared for use in training,
    which involves caching certain types of metadata for each roidb entry.
    """
    ds = JsonDataset(dataset_name, Is_debuglearn)

    return ds





