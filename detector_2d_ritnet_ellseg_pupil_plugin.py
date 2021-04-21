"""
@author: Kevin Barkevich
"""
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..','..','pupil_src','shared_modules','pupil_detector_plugins'))
#sys.path.append(os.path.join(os.path.dirname(__file__), 'ritnet', 'Ellseg'))
from visualizer_2d import draw_pupil_outline
from pupil_detectors import DetectorBase
from pupil_detector_plugins.detector_base_plugin import PupilDetectorPlugin
from pupil_detector_plugins.pye3d_plugin import Pye3DPlugin
from pupil_detector_plugins.detector_2d_plugin import Detector2DPlugin

import logging
from methods import normalize
import torch
import numpy as np
from pyglui import ui
import cv2

# from ritnet.image import get_mask_from_PIL_image, init_model, get_pupil_ellipse_from_PIL_image

# OLD ELLSEG
#from ritnet.Ellseg.args import parse_precision
#from ritnet.Ellseg.pytorchtools import load_from_file
# NEW ELLSEG
from ritnet.Ellseg.evaluate_ellseg import parse_args, evaluate_ellseg_on_image_GD, preprocess_frame,rescale_to_original
from ritnet.Ellseg.modelSummary import model_dict as ellseg_model_dict
MODEL_DICT_KEY = 'ritnet_v2'
WEIGHT_LOCATIONS = os.path.join(os.path.dirname(__file__), 'ritnet', 'Ellseg', 'weights', 'all.git_ok')

ritnet_directory = os.path.join(os.path.dirname(__file__), 'ritnet\\')
filename = "ellseg_allvsone" # best_model.pkl, ritnet_pupil.pkl, ritnet_400400.pkl, ellseg_allvsone
IS_ELLSEG = True
USEGPU = True
support_pixel_ratio_exponent = 2.0  ###default PL-> changed
ellipse_true_support_min_dist = 25

CHANNELS = 3  # POSSIBLY INCORRECT

if USEGPU:
    device=torch.device("cuda")
else:
    device=torch.device("cpu")

logger = logging.getLogger(__name__)

class RITPupilDetector(DetectorBase):
    def __init__(self, model, model_channels):
        self._model = model
        self._model_channels = model_channels

    def detect(self, img):
        frame_scaled_shifted, scale_shift = preprocess_frame(img, (240, 320), align_width=True)
        input_tensor = frame_scaled_shifted.unsqueeze(0).to(device)
        values = evaluate_ellseg_on_image_GD(input_tensor, self._model)
        
        if(values):
            # Return ellipse predictions back to original dimensions
            seg_map, pupil_ellipse, iris_ellipse = rescale_to_original(values[0], values[1], values[2],
                                                                       scale_shift, img.shape)

            # Ellseg returns pupil angle in pupil_ellipse[4] as the major axis orientation in radians, clockwise
            return [seg_map, pupil_ellipse, iris_ellipse]
        else:
            # evaluate_ellseg_on_image_GD will return false on a failure to fit the ellipse
            return None

class Detector2DRITnetEllsegAllvonePlugin(Detector2DPlugin):
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
        return self.detector_ritnet_2d
    
    def update_chosen_detector(self, ritnet_unique_id):
        self.order = 0.08
        
    def ellipse_distance_calculator(self, ellipse, points):
        major_radius = (ellipse[1][0])
        minor_radius = (ellipse[1][1])
        center = ellipse[0]
        angle = ellipse[2] * np.pi / 180
        ## define a rotation matrix
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        ## translate points and ellipse to origin
        points = points - center
        ## align points so that ellipse axis is aligned with coordinate system
        points = np.dot(points, rotation_matrix)
        ## normalize to a unit circle
        points /= np.array((major_radius, minor_radius))
        ## compute the Pythagorean distance
        norm_mag = np.sqrt((points * points).sum(axis=1))
        ## compute the difference with unit circle
        norm_dist = abs(norm_mag - 1)
        ## scale factor to make the points represent their distance to ellipse
        ratio = (norm_dist) / norm_mag
        ## vector scalar multiplication: makeing sure that boradcasting is done right
        scaled_error = np.transpose(points.T * ratio)
        real_error = scaled_error * np.array((major_radius, minor_radius))
        error_mag = np.sqrt((real_error * real_error).sum(axis=1))
        return error_mag

    def ellipse_true_support(self, ellipse, points):  ###define points in X and Y
        # points=np.array(np.where((edges)!=0)).T ##points are stored in (Y,X)
        ##ellipse parameters are given in (X,Y) for ellipse center
        distance = (self.ellipse_distance_calculator(ellipse, points))
        support_pixels = points[distance <= self.ellipse_true_support_min_dist]
        return support_pixels

    def ellipse_circumference(self, major_radius, minor_radius):
        return (np.pi * np.abs(3.0 * (major_radius + minor_radius)
                               - np.sqrt(10.0 * major_radius * minor_radius +
                                         3.0 * (pow(major_radius, 2) + pow(minor_radius, 2)))))

    def ellipse_on_image(self, ellipse, imagee, x, y, z, width):
        image = np.copy(imagee)
        cv2.ellipse(image,
                    ((int(ellipse[0][0]), int(ellipse[0][1])), (int(ellipse[1][0]), int(ellipse[1][1])), ellipse[2]),
                    (x, y, z), width)
        return image

    def ellipse_true_support(self, ellipse, points):  ###define points in X and Y
        # points=np.array(np.where((edges)!=0)).T ##points are stored in (Y,X)
        ##ellipse parameters are given in (X,Y) for ellipse center
        distance = (self.ellipse_distance_calculator(ellipse, points))
        support_pixels = points[distance <= ellipse_true_support_min_dist]
        return support_pixels

    ###AAYUSH 12APril:Changed this function
    def resolve_contour(self, contour, edges):
        support_mask = np.zeros(edges.shape)
        # cv2.polylines(support_mask, contour, True, 255, 2)
        cv_ellipse = cv2.fitEllipse(contour)
        support_mask = self.ellipse_on_image(cv_ellipse, support_mask, 255, 225, 225, 2)

        new_edges = cv2.min(np.uint8(edges), np.uint8(support_mask))
        new_contours = np.flip(np.transpose(np.nonzero(new_edges)), axis=1)
        return new_contours

    def calcConfidence(self, pupil_ellipse, seg_map):

        maskEdges = np.uint8(cv2.Canny(np.uint8(seg_map), 1, 2))
        _, contours, _ = cv2.findContours(np.uint8(seg_map), 1, 2)
        contours = max(contours, key=cv2.contourArea)
        final_edges = self.resolve_contour(contours, maskEdges)

        result = pupil_ellipse

        # GD:  Changed indices to major/minor axes.  Previously, both pointed to the major axis.
        ellipse_circum = self.ellipse_circumference(result[2], result[3])  # radii

        r2 = ((pupil_ellipse[0], pupil_ellipse[1]), (pupil_ellipse[2], pupil_ellipse[3]), pupil_ellipse[4])
        support_pixels = self.ellipse_true_support(r2, final_edges)
        support_ratio = ((len(support_pixels) / ellipse_circum))
        goodness = np.minimum(0.99, support_ratio) * np.power(len(support_pixels) / len(final_edges),
                                                              support_pixel_ratio_exponent)

        return goodness

    def saveMaskAsImage(self, img, seg_map, pupil_ellipse, fileName, eye_id, alpha=0.86):

        imOutDir = os.path.join(os.path.dirname(__file__), "../pupilSegMasks")
        os.makedirs(imOutDir, exist_ok=True)

        if eye_id == 0:

            seg_map = cv2.flip(seg_map, 0)
            pupil_ellipse[1] = img.shape[1] - pupil_ellipse[1]
            pupil_ellipse[4] = -pupil_ellipse[4]

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        colorMask = np.zeros(img.shape, img.dtype)
        colorMask[:, :] = (0, 255, 255)
        colorMask = cv2.bitwise_and(colorMask, colorMask, mask=seg_map)

        #(thresh, new_seg) = cv2.threshold(seg_map, 1, 100, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        new_img = cv2.addWeighted(img, alpha, np.uint8(colorMask), 1 - alpha, 0)

        new_img_ellipse = cv2.ellipse(new_img, ((int(pupil_ellipse[0]), int(pupil_ellipse[1])),
                                                (
                                                int(pupil_ellipse[2] * 2), int(pupil_ellipse[3] * 2)),
                                                pupil_ellipse[4]), (255, 0, 0), 1)

        cv2.imwrite("{}/{}".format(imOutDir, fileName), cv2.flip(new_img_ellipse,0))

        #display(Image.fromarray(new_img_ellipse))
        
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
        
        model = ellseg_model_dict[MODEL_DICT_KEY]
        netDict = torch.load(WEIGHT_LOCATIONS)
        model.load_state_dict(netDict['state_dict'], strict=True)
        model.cuda()
        
        #  Initialize model
        self.isAlone = False
        self.model = model
        self.g_pool.save_ellseg_masks = False
        self.g_pool.ellseg_customellipse = False
        self.g_pool.ellseg_reverse = False
        self.g_pool.ellseg_debug = False
        self.detector_ritnet_2d = RITPupilDetector(model, 4)

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
        debugOutputWindowName = None
        if self.g_pool.ellseg_reverse:
            img = np.flip(np.flip(img, axis=1), axis=0)
        if self.g_pool.ellseg_debug:
            cv2.imshow('EYE'+str(eye_id)+' INPUT', img)
            debugOutputWindowName = 'EYE'+str(eye_id)+' OUTPUT'
        
        customEllipse = self.g_pool.ellseg_customellipse
        values = self.detector_ritnet_2d.detect(img)
        if not values:
            return result
        
        # Ellseg results are obtained - begin obtaining or returning final ellipse
        seg_map = values[0]
        
        pupil_ellipse = values[1]
        iris_ellipse = values[2]
        
        # OPTION 1: If custom ellipse setting is NOT toggled on
        if not customEllipse:
            # background, iris, pupil
            seg_map[np.where(seg_map == 0)] = 255
            seg_map[np.where(seg_map == 1)] = 128
            seg_map[np.where(seg_map == 2)] = 0
            seg_map = np.array(seg_map, dtype=np.uint8)
            framedup = lambda: None
            setattr(framedup, 'gray', seg_map)
            setattr(framedup, 'bgr', frame.bgr)
            setattr(framedup, 'width', frame.width)
            setattr(framedup, 'height', frame.height)
            setattr(framedup, 'timestamp', frame.timestamp)
            if self.g_pool.ellseg_debug:
                cv2.imshow(debugOutputWindowName, seg_map)
            return super().detect(framedup)
        
        # OPTION 2: If custom ellipse setting is toggled on
        #########################################
        ### Ellipse data transformations
        
        # background, iris, pupil
        seg_map[np.where(seg_map == 0)] = 0
        seg_map[np.where(seg_map == 1)] = 0
        seg_map[np.where(seg_map == 2)] = 255
        seg_map = np.array(seg_map, dtype=np.uint8)
        
        openCVformatPupil = np.copy(pupil_ellipse)

        if (pupil_ellipse[4]) > np.pi / 2.0:
            pupil_ellipse[4] = pupil_ellipse[4] - np.pi / 2.0

        if (pupil_ellipse[4]) < -np.pi / 2.0:
            pupil_ellipse[4] = pupil_ellipse[4] + np.pi / 2.0

        pupil_ellipse[4] = np.rad2deg(pupil_ellipse[4])

        #########################################
        confidence = self.calcConfidence(pupil_ellipse, seg_map)

        if self.g_pool.save_ellseg_masks == True:

            fname = "eye-{}_{:0.3f}.png".format(eye_id, confidence)
            self.saveMaskAsImage(img,seg_map,openCVformatPupil,fname,eye_id)


        eye_id = self.g_pool.eye_id
        result["id"] = eye_id
        result["topic"] = f"pupil.{eye_id}.{self.identifier}"

        ellipse["center"] = (pupil_ellipse[0], pupil_ellipse[1])
        ellipse["axes"] = (pupil_ellipse[2]*2, pupil_ellipse[3]*2)
        ellipse["angle"] = pupil_ellipse[4]

        result["ellipse"] = ellipse
        result["diameter"] = pupil_ellipse[2]*2
        result["location"] = ellipse["center"]
        result["confidence"] = confidence
        result["timestamp"] = frame.timestamp
        #logger.debug(result)

        location = result["location"]

        norm_pos = normalize(location, (frame.width, frame.height), flip_y= True)

        result["norm_pos"] = norm_pos

        try:
            self.g_pool.ellSegDetector[str(self.g_pool.eye_id)] = result
        except:
            self.g_pool.ellSegDetector = {str(self.g_pool.eye_id): result}

        return result
        
    def gl_display(self):
        if self._recent_detection_result:
            draw_pupil_outline(self._recent_detection_result, color_rgb=(1, 0, 1))
    
    def init_ui(self):
        super(Detector2DPlugin, self).init_ui()
        self.menu.label = self.pretty_class_name
        self.menu_icon.label_font = "pupil_icons"
        info = ui.Info_Text(
            "(PURPLE) Model using EllSeg, the \"allvsone\" model."
        )
        self.menu.append(info)
        self.menu.append(
            ui.Switch(
                "ellseg_reverse",
                self.g_pool,
                label="Flip image horizontally and vertically before processing"
            )
        )
        self.menu.append(
            ui.Switch(
                "ellseg_customellipse",
                self.g_pool,
                label="Use Custom Ellipse Finding Algorithm"
            )
        )
        self.menu.append(
            ui.Switch(
                "ellseg_debug",
                self.g_pool,
                label="Enable Debug Mode"
            )
        )
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
