"""
@author: Kevin Barkevich
"""
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..','..','pupil_src','shared_modules','pupil_detector_plugins'))
from visualizer_2d import draw_pupil_outline
from pupil_detectors import DetectorBase
from pupil_detector_plugins.detector_base_plugin import PupilDetectorPlugin
from pupil_detector_plugins.pye3d_plugin import Pye3DPlugin

import logging
from methods import normalize
import torch
import numpy as np
from pyglui import ui

from ritnet.image import get_mask_from_PIL_image, init_model, get_pupil_ellipse_from_PIL_image
from ritnet.models import model_dict, model_channel_dict
from ritnet.Ellseg.args import parse_precision
from ritnet.Ellseg.pytorchtools import load_from_file

from ritnet_plugin_settings import ritnet_labels, ritnet_ids, default_id

ritnet_directory = os.path.join(os.path.dirname(__file__), 'ritnet\\')
filename = "ellseg_allvsone" # best_model.pkl, ritnet_pupil.pkl, ritnet_400400.pkl, ellseg_allvsone
MODEL_DICT_STR, CHANNELS, IS_ELLSEG, ELLSEG_MODEL = model_channel_dict[filename]
ELLSEG_FOLDER = 'Ellseg'
ELLSEG_FILEPATH = ritnet_directory+ELLSEG_FOLDER
ELLSEG_PRECISION = 32  # precision. 16, 32, 64
ELLSEG_PRECISION = parse_precision(ELLSEG_PRECISION)
USEGPU = True
KEEP_BIGGEST_PUPIL_BLOB_ONLY = True

if USEGPU:
    device=torch.device("cuda")
else:
    device=torch.device("cpu")

logger = logging.getLogger(__name__)

class RITPupilDetector(DetectorBase):
    def __init__(self, model, model_channels):
        self._model = model
        self._model_channels = model_channels

    #def get_circle_coord(self):
    #    # retreive a new point from the circle on every frame refresh
    #    coords = self.pnts[self.ind]
    #    if self.ind < self.stop_ind:
    #        self.ind += 1
    #    else:
    #        self.ind = 0
    #    return coords

    def detect(self, img):
        # here we override the detect method with our own custom detector
        # this is a random artificial pupil center
        # center = (90, 90)
        # we move the center around the circle here
        # center = tuple(sum(x) for x in zip(center, self.get_circle_coord()))
        # return center
        return get_pupil_ellipse_from_PIL_image(img, self._model, isEllseg=IS_ELLSEG, ellsegPrecision=ELLSEG_PRECISION, ellsegEllipse=False)

class Detector2DRITnetEllsegAllvonePlugin(PupilDetectorPlugin):
    uniqueness = "by_class"
    icon_chr = "RE"

    label = "RITnet ellseg_allvsone 2d detector"
    identifier = "ritnet-ellsegav1-2d"
    method = "2d c++"
    order = 0.08
    pupil_detection_plugin = "2d c++"
    
    @property
    def pretty_class_name(self):
        return "RITnet Detector (ellseg_allvsone)"
        
    @property
    def pupil_detector(self) -> RITPupilDetector:
        return self.detector_2d
    
    def update_chosen_detector(self, ritnet_unique_id):
        self.order = 0.08
        
    def __init__(
        self,
        g_pool=None,
        # properties=None,
        # detector_2d: Detector2D = None,
    ):
        super().__init__(g_pool=g_pool)
        #  Set up RITnet
        if torch.cuda.device_count() > 1:
            print('Moving to a multiGPU setup for Ellseg.')
            useMultiGPU = True
        else:
            useMultiGPU = False
        model = model_dict[MODEL_DICT_STR]
    
        if not IS_ELLSEG:
            model  = model.to(device)
            model.load_state_dict(torch.load(ritnet_directory+filename))
            model = model.to(device)
            model.eval()
        else:
            LOGDIR = os.path.join(ELLSEG_FILEPATH, 'ExpData', 'logs',\
                              'ritnet_v2', ELLSEG_MODEL)
            path2checkpoint = os.path.join(LOGDIR, 'checkpoints')
            checkpointfile = os.path.join(path2checkpoint, 'checkpoint.pt')
            print(checkpointfile)
            netDict = load_from_file([checkpointfile, ''])
            #print(checkpointfile)
            #print(netDict)
            model.load_state_dict(netDict['state_dict'])
            #print('Parameters: {}'.format(get_nparams(model)))
            model = model if not useMultiGPU else torch.nn.DataParallel(model)
            model = model.to(device)
            model = model.to(ELLSEG_PRECISION)
            model.eval()
        
        #  Initialize model
        
        self.isAlone = False
        self.model = model
        self.detector_2d = RITPupilDetector(model, 4)

    def _stop_other_pupil_detectors(self):
        plugin_list = self.g_pool.plugins

        # Deactivate other PupilDetectorPlugin instances
        for plugin in plugin_list:
            if plugin.alive is True and isinstance(plugin, PupilDetectorPlugin) and plugin is not self and not isinstance(plugin, Pye3DPlugin):
                plugin.alive = False

        # Force Plugin_List to remove deactivated plugins
        plugin_list.clean()
    
    def detect(self, frame, **kwargs):
        if not self.isAlone:
            self._stop_other_pupil_detectors()
            self.isAlone = True
        result = {}
        ellipse = {}
        eye_id = self.g_pool.eye_id
        result["id"] = eye_id
        result["topic"] = f"pupil.{eye_id}.{self.identifier}"
        ellipse["center"] = (0.0, 0.0)
        ellipse["axes"] = (0.0, 0.0)
        ellipse["angle"] = 0.0
        result["ellipse"] = ellipse
        result["diameter"] = 0.0
        result["location"] = ellipse["center"]
        result["confidence"] = 0.0
        result["timestamp"] = frame.timestamp
        result["method"] = self.method
            
        img = frame.gray
        
        # pred_img, predict = get_mask_from_PIL_image(frame, self.model, USEGPU, False, True, CHANNELS, KEEP_BIGGEST_PUPIL_BLOB_ONLY, isEllseg=IS_ELLSEG, ellsegPrecision=ELLSEG_PRECISION)
        # ellipsedata = get_pupil_ellipse_from_PIL_image(img, self.model)
        # img = np.uint8(get_mask_from_PIL_image(img, self.model) * 255)
        
        ellipsedata = self.detector_2d.detect(img)

        if ellipsedata is not None:
            eye_id = self.g_pool.eye_id
            result["id"] = eye_id
            result["topic"] = f"pupil.{eye_id}.{self.identifier}"
            ellipse["center"] = (ellipsedata[0], ellipsedata[1])
            ellipse["axes"] = (ellipsedata[2]*2, ellipsedata[3]*2)
            ellipse["angle"] = ellipsedata[4]
            result["ellipse"] = ellipse
            result["diameter"] = ellipsedata[2]*2
            result["location"] = ellipse["center"]
            result["confidence"] = 0.99
            result["timestamp"] = frame.timestamp
        else:
            return result

        #eye_id = self.g_pool.eye_id
        location = result["location"]
        norm_pos = normalize(
            location, (frame.width, frame.height), flip_y=True
        )
        result["norm_pos"] = norm_pos
        #result["timestamp"] = frame.timestamp
        #result["topic"] = f"pupil.{eye_id}.{self.identifier}"
        #result["id"] = eye_id
        #result["previous_detection_results"] = result.copy()
        try:
            self.g_pool.Detector2DRITnetEllsegAllvonePlugin[str(self.g_pool.eye_id)] = result
        except:
            self.g_pool.Detector2DRITnetEllsegAllvonePlugin = {str(self.g_pool.eye_id): result}
        return result
        
    def gl_display(self):
        if self._recent_detection_result:
            draw_pupil_outline(self._recent_detection_result, color_rgb=(1, 0, 1))
    
    def init_ui(self):
        super().init_ui()
        self.menu.label = self.pretty_class_name
        self.menu_icon.label_font = "pupil_icons"
        info = ui.Info_Text(
            "(PURPLE) Model using EllSeg, the \"allvsone\" model."
        )
        self.menu.append(info)
        """
        self.menu.append(
            ui.Slider(
                "2d.intensity_range",
                self.proxy,
                label="Pupil intensity range",
                min=0,
                max=60,
                step=1,
            )
        )
        self.menu.append(
            ui.Slider(
                "2d.pupil_size_min",
                self.proxy,
                label="Pupil min",
                min=1,
                max=250,
                step=1,
            )
        )
        self.menu.append(
            ui.Slider(
                "2d.pupil_size_max",
                self.proxy,
                label="Pupil max",
                min=50,
                max=400,
                step=1,
            )
        )
        self.menu.append(
            ui.Switch(
                "ritnet_2d",
                self.g_pool,
                label="Enable RITnet"
            )
        )
        """
