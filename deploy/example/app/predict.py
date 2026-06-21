# Copyright 2018 Airbus. All rights reserved.
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

"""Object detection processing.
"""

# utilities
import json
import base64
import io
import collections.abc

# image processing
import PIL

# matrix processing
import numpy as np

# Local imports
from inference import InferenceEngine


class PredictError(Exception):
    """Predict processing error.
    
    Attributes:
        message (str): Description
    """

    def __init__(self, message):
        """Constructor.
        
        Args:
            message (str): error message
        """
        super(PredictError, self).__init__(message)


class Predict(object):
    """Object detection processing.
    
    Attributes:
        logger (TYPE): the logger
    """

    def __init__(self, logger):
        """Constructor.
        
        Args:
            logger (TYPE): the logger
        """
        self.logger = logger
        self.engine = InferenceEngine()

    def process(self, resolution, tiles):
        """Process tile.
        
        Args:
            resolution (float): size on the ground of a pixel in meters
            tiles (bytearray): array of tiles

        
        Returns:
            object: GeoJSON FeatureCollection
        """
        if self.logger:
            self.logger.debug('Generate prediction')

        try:
            # convert first tile from byte array to PIL image
            img = PIL.Image.open(io.BytesIO(tiles[0]))
            #print(img.size)

            # TODO: Optionnaly, get rid of potential alpha channel (if your model only supports RGB)
            img = img.convert('RGB')

            # TODO: Optionnaly, convert to numpy (if your model needs np arrays rather than PIL)
            img = np.asarray(img, dtype=np.uint8)
            self.logger.info(f"Image shape is {img.shape}")

            # run machine learning algorithm on tile image
            results = self._predict(img, resolution)

            if self.logger:
                #self.logger.info(results)
                self.logger.info('%d objects found', len(results['features']))

        except Exception as error:
            if self.logger:
                self.logger.exception(error)
            raise PredictError(error)

        return results


    def _predict(self, img, resolution):
        """Process tile.
        
        Args:
            img (PIL Image): single image or array of images
            resolution (float): resolution in meters per pixel
        
        Returns:
            array: array of GeoJSON Features
        """
        if isinstance(img, collections.abc.Sequence):
            # batch
            return self.engine.predict(img, resolution)
        elif isinstance(img, PIL.Image.Image) or isinstance(img, np.ndarray):
            # single image
            return self.engine.predict_single(img, resolution)
        else:
            message = "_predict: input shoud be batch or single RGB image in PIL or numpy array format."
            if self.logger:
                self.logger.exception(message)
            raise PredictError(message)


