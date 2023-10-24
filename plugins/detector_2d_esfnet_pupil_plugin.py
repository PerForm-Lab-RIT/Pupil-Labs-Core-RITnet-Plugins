"""
@author: Kevin Barkevich
"""
import sys
import os
import math
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..','..','pupil_src','shared_modules','pupil_detector_plugins'))
#sys.path.append(os.path.join(os.path.dirname(__file__), 'ritnet', 'Ellseg_v2_v2'))
from visualizer_2d import draw_pupil_outline
from pupil_detectors import DetectorBase
from pupil_detector_plugins.detector_base_plugin import PupilDetectorPlugin
from pupil_detector_plugins.pye3d_plugin import Pye3DPlugin
from pupil_detector_plugins.detector_2d_plugin import Detector2DPlugin

from enum import Enum
import logging
from methods import normalize
import torch
import numpy as np
from pyglui import ui
import cv2
from scipy.stats import entropy
from scipy.ndimage import binary_closing
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
# from ritnet.image import get_mask_from_PIL_image, init_model, get_pupil_ellipse_from_PIL_image

# OLD ELLSEG
#from ritnet.Ellseg.args import parse_precision
#from ritnet.Ellseg.pytorchtools import load_from_file
# NEW ELLSEG
#from ritnet.Ellseg_v2_v2.evaluate_ellseg import parse_args, evaluate_ellseg_on_image, preprocess_frame,rescale_to_original
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ESFnet"))
from ESFnet.bdcn_new import BDCN
from ESFnet.evaluate import preprocess_frame, evaluate_ellseg_on_image, rescale_to_original, get_config
from ESFnet.models.RITnet_v2 import DenseNet2D
from ESFnet.pytorchtools import load_from_file

#from ritnet.Ellseg_v2_v2.models_mux import model_dict as ellseg_model_dict
from ritnet.Ellseg_v2_v2.args_maker import make_args
model_file = 'baseline_edge_16.pkl'
setting_file = 'baseline_edge.yaml'
edgemodel_file = 'gen_00000016.pt'
WEIGHT_LOCATIONS = os.path.join(os.path.dirname(__file__), '..', 'ESFnet', 
    model_file)
SETTINGS_LOCATION = os.path.join(os.path.dirname(__file__), '..', 'ESFnet', 
    'configs', setting_file)
EDGEMODEL_LOCATION = os.path.join(os.path.dirname(__file__), '..', 'ESFnet', 
    edgemodel_file)

ritnet_directory = os.path.join(os.path.dirname(__file__), '..', 'ritnet\\')
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


class EntropyConfidence(Enum):
    ALWAYS_ON = 1
    ALL = 2
    SIMPLE = 3
    MIN_5 = 4


class RITPupilDetector(DetectorBase):
    def __init__(self, model, BDCN_network, model_channels, custom_ellipse=False):
        self._model = model
        self._BDCN_network = BDCN_network
        self._model_channels = model_channels
        self._custom_ellipse = custom_ellipse
        self._clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))

    def detect(self, img):
        if True:
            # CLAHE contrast enhancement a-la RITnet
            frame = self._clahe.apply(img)
            frame_scaled_shifted, scale_shift = preprocess_frame(frame, (240, 320), align_width=True)
        else:
            frame_scaled_shifted, scale_shift = preprocess_frame(img, (240, 320), align_width=True)
        input_tensor = frame_scaled_shifted.unsqueeze(0).to(device)
        out_tuple = evaluate_ellseg_on_image(input_tensor, self._model, self._BDCN_network, device=torch.device("cuda"), get_ellipses=self._custom_ellipse)
        if(out_tuple):
            edge_map, seg_map, pupil_ellipse, iris_ellipse = out_tuple
            # Return ellipse predictions back to original dimensions
            out_dict = rescale_to_original(edge_map, seg_map,
                                           pupil_ellipse, iris_ellipse,
                                               scale_shift,
                                               img.shape)
            edge_map, seg_map, pupil_ellipse, iris_ellipse = out_dict
            # Ellseg returns pupil angle in pupil_ellipse[4] as the major axis orientation in radians, clockwise
            return [edge_map, seg_map, pupil_ellipse, iris_ellipse]
        else:
            # evaluate_ellseg_on_image_GD will return false on a failure to fit the ellipse
            return None


class Detector2DESFnetPlugin(Detector2DPlugin):
    uniqueness = "by_class"
    icon_chr = "RE"

    label = "RITnet ESF-net 2d detector"
    identifier = "ritnet-esfnet-2d"
    method = "2d c++"
    order = 0.08
    pupil_detection_plugin = "2d c++"

    @property
    def pretty_class_name(self):
        return "RITnet Detector (ESF-net)"
        
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
        support_pixels = points[distance <= self.ellipse_true_support_min_dist]
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

    def calcConfidence(self, pupil_ellipse, seg_map, debug_confidence_timestamp=None, final_edges=None):
        
        if final_edges is None:
            maskEdges = np.uint8(cv2.Canny(np.uint8(seg_map), 1, 2))
            _, contours, _ = cv2.findContours(np.uint8(seg_map), 1, 2)
            contours = max(contours, key=cv2.contourArea)
            final_edges = self.resolve_contour(contours, maskEdges)
        
        result = pupil_ellipse
        #temp = np.zeros(seg_map.shape)
        #cv2.drawContours(temp, contours, -1, 255, 1)
        #cv2.imshow('Contours' + str(self.g_pool.eye_id), temp)
        #cv2.imshow('maskEdges' + str(self.g_pool.eye_id), maskEdges)
        
        # GD:  Changed indices to major/minor axes.  Previously, both pointed to the major axis.
        ellipse_circum = self.ellipse_circumference(result[2], result[3])  # radii

        r2 = ((pupil_ellipse[0], pupil_ellipse[1]), (pupil_ellipse[2], pupil_ellipse[3]), pupil_ellipse[4])
        support_pixels = self.ellipse_true_support(r2, final_edges)
        support_ratio = ((len(support_pixels) / ellipse_circum))
        goodness = np.minimum(0.99, support_ratio) * np.power(len(support_pixels) / len(final_edges),
                                                              support_pixel_ratio_exponent)
        
        if debug_confidence_timestamp is not None:
            imOutDir = os.path.join(self.g_pool.capture.source_path[0:self.g_pool.capture.source_path.rindex("\\")+1], "eye"+str(self.g_pool.eye_id)+"_confidence_supportpixels")
            os.makedirs(imOutDir, exist_ok=True)
            im = np.zeros(seg_map.shape)
            for i in range(0, len(support_pixels)):
                im[support_pixels[i, 0], support_pixels[i, 1]] = 255
            im = np.uint8(im)
            fileName = "eye-{}_{:0.3f}_{}.png".format(self.g_pool.eye_id, goodness, debug_confidence_timestamp)
            cv2.imwrite("{}/{}".format(imOutDir, fileName), im)
        
        return goodness

    def saveMaskAsImage(self, img, seg_map, pupil_ellipse, fileName, flipImage = False, alpha=0.86):

        if flipImage:

            seg_map = cv2.flip(seg_map, 0)
            pupil_ellipse[1] = img.shape[1] - pupil_ellipse[1]
            pupil_ellipse[4] = -pupil_ellipse[4]

        imOutDir = os.path.join(self.g_pool.capture.source_path[0:self.g_pool.capture.source_path.rindex("\\")+1], "eye"+str(self.g_pool.eye_id)+"_masks")
        imOutDir2 = os.path.join(self.g_pool.capture.source_path[0:self.g_pool.capture.source_path.rindex("\\")+1], "eye"+str(self.g_pool.eye_id)+"_masks2")
        os.makedirs(imOutDir, exist_ok=True)
        os.makedirs(imOutDir2, exist_ok=True)

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        colorMask = np.zeros(img.shape, img.dtype)
        colorMask[:, :] = (0, 255, 255)
        colorMask = cv2.bitwise_and(colorMask, colorMask, mask=seg_map)
        
        new_seg_map = cv2.ellipse(np.zeros(img.shape, img.dtype), ((int(pupil_ellipse[0]), int(pupil_ellipse[1])),
                                                (int(pupil_ellipse[2]*2.0), int(pupil_ellipse[3]*2.0)),
                                                pupil_ellipse[4]), (0, 0, 255), -1)
        new_seg_map[seg_map == 254, 0] = 255
        logical_combo = np.all(
            new_seg_map == (255, 0, 255),
            axis=-1
        )
        new_seg_map[logical_combo, :] = (0, 255, 0)
        
        new_seg_map = cv2.putText(new_seg_map, "mask", (5, 25), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, cv2.LINE_AA)
        new_seg_map = cv2.putText(new_seg_map, "mask", (5, 25), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 1, cv2.LINE_AA)
        new_seg_map = cv2.putText(new_seg_map, "ellipse", (5, 55), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, cv2.LINE_AA)
        new_seg_map = cv2.putText(new_seg_map, "ellipse", (5, 55), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 1, cv2.LINE_AA)
        new_seg_map = cv2.putText(new_seg_map, "overlap", (5, 85), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, cv2.LINE_AA)
        new_seg_map = cv2.putText(new_seg_map, "overlap", (5, 85), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 1, cv2.LINE_AA)
        
        #(thresh, new_seg) = cv2.threshold(seg_map, 1, 100, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        new_img = cv2.addWeighted(img, alpha, np.uint8(colorMask), 1 - alpha, 0)


        new_img_ellipse = cv2.ellipse(new_img, ((int(pupil_ellipse[0]), int(pupil_ellipse[1])),
                                                (int(pupil_ellipse[2]*2.0), int(pupil_ellipse[3]*2.0)),
                                                pupil_ellipse[4]), (255, 0, 0), 1)

        cv2.imwrite("{}/{}".format(imOutDir, fileName), new_img_ellipse)
        cv2.imwrite("{}/{}".format(imOutDir2, fileName), new_seg_map)
        
    def calcFinalEntropyEdges(self, pupil_entropy_mask):
        #final_result['confidence'] = test_conf
        #print(test_conf)
        
        thresh = np.max(pupil_entropy_mask)
        pupil_entropy_mask = (pupil_entropy_mask - np.min(pupil_entropy_mask)) / (np.max(pupil_entropy_mask) - np.min(pupil_entropy_mask))
        #hist, bins = np.histogram(pupil_entropy_mask[pupil_entropy_mask > 0].flatten(), np.linspace(0,1,20))
        #print("---------------")
        #print(hist)
        #print(bins)
        #print("---------------")
        thresh = np.mean(pupil_entropy_mask[pupil_entropy_mask > 0])
        #print("THRESH: ",thresh)
        #print("ZEROS:  ",len(pupil_entropy_mask[pupil_entropy_mask == 0]))
        entropy_edges = pupil_entropy_mask
        entropy_edges[pupil_entropy_mask >= thresh] = 1
        entropy_edges[pupil_entropy_mask < thresh] = 0
        entropy_edges = np.uint8(entropy_edges)
        entropy_edges = binary_closing(entropy_edges, structure=np.ones((10,10)))
        entropy_edges = np.uint8(entropy_edges)
        entropy_edges[entropy_edges != 0] = 255
        return np.flip(np.transpose(np.nonzero(entropy_edges)), axis=1)

        
    def __init__(
        self,
        g_pool=None,
        properties=None,
        # detector_2d: Detector2D = None,
        help_thresh=0.85
    ):
        super().__init__(g_pool=g_pool, properties=properties)
        #  Set up RITnet
        if torch.cuda.device_count() > 1:
            print('Moving to a multiGPU setup for Ellseg.')
            useMultiGPU = True
        else:
            useMultiGPU = False
        
        # NETWORK 1 - EDGE NETWORK
        BDCN_network = BDCN()
        state_dict = torch.load(EDGEMODEL_LOCATION)
        BDCN_network.load_state_dict(state_dict['a'])
        BDCN_network = BDCN_network.cuda()
        BDCN_network.eval()

        # NETWORK 2 - ESF NETWORK
        setting = get_config(SETTINGS_LOCATION)
        model = DenseNet2D(setting)
        print('Setting : {} {}'.format(model_file, setting_file))
        netDict = load_from_file([WEIGHT_LOCATIONS])
        model.load_state_dict(netDict['state_dict'])
        model.cuda()
        
        #  Initialize model
        self.isAlone = False
        self.BDCN_network = BDCN_network
        self.model = model
        self.ellipse_true_support_min_dist = ellipse_true_support_min_dist
        
        def condition(x): return "--min_pupil_size=" in x
        output = [idx for idx, element in enumerate(sys.argv) if condition(element)]
        min_pupil_size = int(sys.argv[output[0]][sys.argv[output[0]].rfind('=')+1:]) if len(output) > 0 else 1
        self.g_pool.ellseg_pupil_size_min = min_pupil_size
        
        self.g_pool.ellseg_customellipse = True if "--custom-ellipse" in sys.argv else False
        self.g_pool.ellseg_reverse = True if self.g_pool.eye_id==1 else False
        self.g_pool.ellseg_debug = False
        self.g_pool.save_masks = True if ("--save-masks=0" in sys.argv and self.g_pool.eye_id==0) or ("--save-masks=1" in sys.argv and self.g_pool.eye_id==1) or "--save-masks=both" in sys.argv else False
        self.g_pool.calcCustomConfidence = True if "--custom-confidence" in sys.argv else False
        
        if "--entropy-confidence=always_on" in sys.argv:
            self.g_pool.entropy_confidence = EntropyConfidence.ALWAYS_ON
        elif "--entropy-confidence=simple" in sys.argv:
            self.g_pool.entropy_confidence = EntropyConfidence.SIMPLE
        elif "--entropy-confidence=all" in sys.argv:
            self.g_pool.entropy_confidence = EntropyConfidence.ALL
        elif "--entropy-confidence=min_5" in sys.argv:
            self.g_pool.entropy_confidence = EntropyConfidence.MIN_5
        else:
            self.g_pool.entropy_confidence = None

        self.standard_count = 0
        self.simple_count = 0
        self.entropy_count = 0
        
        self.detector_ritnet_2d = RITPupilDetector(model, BDCN_network, 4, custom_ellipse=self.g_pool.ellseg_customellipse)
        self.help_thresh = help_thresh

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
        
        # ---------- Abandon early if PL Vanilla does good ----------
        #initial_result = super().detect(frame)
        #if initial_result['confidence'] >= self.help_thresh:
        #    return initial_result

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
        result["norm_pos"] = [0.0, 0.0]#[np.nan,np.nan]

        img = frame.gray
        debugOutputWindowName = None

        if self.g_pool.ellseg_reverse:
            img = np.flip(img, axis=0)

        if self.g_pool.ellseg_debug:
            cv2.imshow('EYE'+str(eye_id)+' INPUT', img)
            debugOutputWindowName = 'EYE'+str(eye_id)+' OUTPUT'

        else:
            try:
                cv2.destroyWindow('EYE'+str(eye_id)+' INPUT')
            except Exception as _:
                pass

        customEllipse = self.g_pool.ellseg_customellipse
        
        values = self.detector_ritnet_2d.detect(img)

        if not values:
            return result
        
        # Ellseg results are obtained - begin obtaining or returning final ellipse
        seg_map = values[1]
        origSeg_map = np.copy(seg_map)
        yellow_mask = np.zeros((origSeg_map.shape[0], origSeg_map.shape[1], 3))
        yellow_mask[:,:,2][origSeg_map == 2] = 128
        yellow_mask[:,:,1][origSeg_map == 2] = 128
        if self.g_pool.eye_id == 1:
            yellow_mask = np.flipud(yellow_mask)
        alpha = 0.25
        beta = 1.0 - alpha
        final_bgr = cv2.addWeighted(yellow_mask.astype(np.uint8), alpha, frame.bgr.astype(np.uint8), beta, 0.0)
        frame.bgr = final_bgr
        
        ellseg_pupil_ellipse = values[2]
        #iris_ellipse = values[2]

        if self.g_pool.ellseg_reverse:
            seg_map = np.flip(seg_map, axis=0)
            
            # Change format of ellseg ellipse to meet PL conventions
            height, width = seg_map.shape
            ellseg_pupil_ellipse[1] = (-ellseg_pupil_ellipse[1]+(2*height/2))
            ellseg_pupil_ellipse[4] = ellseg_pupil_ellipse[4]*-1

        # OPTION 1: If custom ellipse setting is NOT toggled on
        if not customEllipse:

            ## Prepare pupil mask for pupil labs ellipse fit
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

            ## Apply pupil labs ellipse fit to mask
            final_result = super().detect(framedup)
            if self.g_pool.ellseg_debug:
                final_result_ellipse = final_result["ellipse"]
                elcenter = final_result_ellipse["center"]
                elaxes = final_result_ellipse["axes"] # axis diameters

                seg_map_debug = np.stack((np.copy(seg_map),)*3, axis=-1)

                cv2.ellipse(seg_map_debug,
                    (round(elcenter[0]), round(elcenter[1])),
                    (round(elaxes[0]/2), round(elaxes[1]/2)), # convert diameters to radii
                    final_result_ellipse["angle"], 0, 360, (255, 0, 0), 1)

                cv2.imshow(debugOutputWindowName, seg_map_debug)
                
            pl_pupil_ellipse = [final_result["ellipse"]["center"][0], final_result["ellipse"]["center"][1],
                                final_result["ellipse"]["axes"][0]/2.0, final_result["ellipse"]["axes"][1]/2.0,
                                final_result["ellipse"]["angle"]]

            if self.g_pool.calcCustomConfidence:

                # origSeg_map[np.where(origSeg_map == 0)] = 0
                # origSeg_map[np.where(origSeg_map == 1)] = 0
                # origSeg_map[np.where(origSeg_map == 2)] = 255
                # origSeg_map = np.array(origSeg_map, dtype=np.uint8)

                seg_map[np.where(seg_map == 0)] = 254
                seg_map[np.where(seg_map == 255)] = 0
                seg_map[np.where(seg_map == 128)] = 0

                seg_map = np.array(seg_map, dtype=np.uint8)
                
                if self.g_pool.save_masks:
                    final_result['confidence'] = self.calcConfidence(pl_pupil_ellipse, seg_map, debug_confidence_timestamp=frame.timestamp)
                else:
                    final_result['confidence'] = self.calcConfidence(pl_pupil_ellipse, seg_map, debug_confidence_timestamp=None)
                if np.isnan(final_result['confidence']):
                    final_result['confidence'] = 0.0
                elif self.g_pool.entropy_confidence is not None:
                    self.ellipse_true_support_min_dist = 5  # be a LOT more strict since we're working with tight edges
                    # Modify confidence based on entropy
                    if self.g_pool.entropy_confidence == EntropyConfidence.ALWAYS_ON:
                        final_result['confidence'] = 0.99
                    elif self.g_pool.entropy_confidence == EntropyConfidence.ALL:
                        # Standard Pupil Labs Confidence
                        standard_conf = final_result['confidence']
                        
                        # Simple (mean-based) Entropy Confidence
                        simple_entropy_conf = np.power(np.mean(pupil_entropy_mask) - 1, 100)
                        
                        # Pupil Labs w/ Entropy Modification Confidence
                        entropy_edges = self.calcFinalEntropyEdges(pupil_entropy_mask)
                        entropy_conf = self.calcConfidence(pl_pupil_ellipse, seg_map, debug_confidence_timestamp=None, final_edges=entropy_edges) if len(entropy_edges) else 0.0
                        
                        # Final Confidence
                        final_result['confidence'] = np.min([standard_conf, simple_entropy_conf, entropy_conf])
                        argmin = np.argmin([standard_conf, simple_entropy_conf, entropy_conf])
                        
                        histogram_imOutDir = os.path.join(self.g_pool.capture.source_path[0:self.g_pool.capture.source_path.rindex("\\")+1], "graph_outs")
                        if os.path.isfile(histogram_imOutDir+"/"+str(eye_id)+"conf_data.json"):
                            with open(histogram_imOutDir+"/"+str(eye_id)+"conf_data.json") as infile:
                                conf_data = json.load(infile)
                        else:
                            os.makedirs(histogram_imOutDir, exist_ok=True)
                            conf_data = {'standard': {}, 'simple': {}, 'entropy': {}}
                        
                        conf_data['standard'][frame.timestamp] = standard_conf
                        conf_data['simple'][frame.timestamp] = simple_entropy_conf
                        conf_data['entropy'][frame.timestamp] = entropy_conf
                        
                        with open(histogram_imOutDir+"/"+str(eye_id)+"conf_data.json", "w") as outfile:
                            json.dump(conf_data, outfile)
                        
                        print(final_result['confidence'])
                    elif self.g_pool.entropy_confidence == EntropyConfidence.SIMPLE:
                        #test_conf = (-np.tan(np.mean(pupil_entropy_mask))/np.tan(1)) + 1
                        test_conf = np.power(np.mean(pupil_entropy_mask) - 1, 100)
                        final_result['confidence'] = test_conf
                    elif self.g_pool.entropy_confidence == EntropyConfidence.MIN_5:
                        thresh = np.max(pupil_entropy_mask)
                        pupil_entropy_mask = (pupil_entropy_mask - np.min(pupil_entropy_mask)) / (np.max(pupil_entropy_mask) - np.min(pupil_entropy_mask))
                        thresh = np.mean(pupil_entropy_mask[pupil_entropy_mask > 0])
                        entropy_edges = pupil_entropy_mask
                        entropy_edges[pupil_entropy_mask >= thresh] = 1
                        entropy_edges[pupil_entropy_mask < thresh] = 0
                        entropy_edges = np.uint8(entropy_edges)
                        
                        entropy_edges_temp = entropy_edges
                        
                        entropy_edges = binary_closing(entropy_edges, structure=np.ones((10,10)))
                        
                        entropy_edges_temp = np.uint8(np.logical_xor(entropy_edges, entropy_edges_temp))
                        entropy_edges_temp[entropy_edges_temp != 0] = 255
                        cv2.imshow('EYE'+str(eye_id)+' ENTROPY DIFF', entropy_edges_temp)

                        entropy_edges = np.uint8(entropy_edges)
                        entropy_edges[entropy_edges != 0] = 255
                        
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        orgPP = (10, 15)
                        orgPPDiff = (10, 35)
                        orgIouDiff = (10, 55)
                        fontScale = 0.5
                        color = 255
                        thickness = 2
                    
                        if self.g_pool.save_masks:
                            final_edges = np.flip(np.transpose(np.nonzero(entropy_edges)), axis=1)
                            final_result['confidence'] = self.calcConfidence(pl_pupil_ellipse, seg_map, debug_confidence_timestamp=frame.timestamp, final_edges=final_edges) if len(final_edges) else 0.0
                            entropy_edges = cv2.putText(entropy_edges, "CONF:  "+"{:.4f}".format(final_result['confidence']), orgPP, font, fontScale,
                                        color, thickness, cv2.LINE_AA)
                            cv2.imshow('EYE'+str(eye_id)+' ENTROPY', entropy_edges)  # This "edge detector" is elliptical in all good frames and not elliptical in all bad frames
                        else:
                            final_edges = np.flip(np.transpose(np.nonzero(entropy_edges)), axis=1)
                            final_result['confidence'] = self.calcConfidence(pl_pupil_ellipse, seg_map, debug_confidence_timestamp=None, final_edges=final_edges) if len(final_edges) else 0.0
                            entropy_edges = cv2.putText(entropy_edges, "CONF:  "+"{:.4f}".format(final_result['confidence']), orgPP, font, fontScale,
                                        color, thickness, cv2.LINE_AA)
                            cv2.imshow('EYE'+str(eye_id)+' ENTROPY', entropy_edges)  # This "edge detector" is elliptical in all good frames and not elliptical in all bad frames
                        
                        if self.g_pool.save_masks:
                            conf_rounded = int(math.ceil(final_result['confidence']*100 / 10.0))*10/100
                            fname = "{}.png".format(frame.timestamp)
                            imOutDir = os.path.join(self.g_pool.capture.source_path[0:self.g_pool.capture.source_path.rindex("\\")+1], "eye"+str(self.g_pool.eye_id)+"_entropy/{:0.2f}".format(conf_rounded))
                            os.makedirs(imOutDir, exist_ok=True)
                            
                            final_result_ellipse = final_result["ellipse"]
                            elcenter = (final_result_ellipse["center"][0], frame.height - final_result_ellipse["center"][1]) if self.g_pool.ellseg_reverse else final_result_ellipse["center"]
                            elaxes = final_result_ellipse["axes"] # axis diameters
                            elangle = 180 - final_result_ellipse["angle"] if self.g_pool.ellseg_reverse else final_result_ellipse["angle"]
                            
                            img_with_ellipse = np.stack((np.copy(img),)*3, axis=-1)

                            cv2.ellipse(img_with_ellipse,
                                (round(elcenter[0]), round(elcenter[1])),
                                (round(elaxes[0]/2), round(elaxes[1]/2)), # convert diameters to radii
                                final_result_ellipse["angle"], 0, 360, (255, 0, 0), 1)
                            
                            final_entropy_out = cv2.hconcat([img_with_ellipse, np.stack((np.copy(entropy_edges),)*3, axis=-1)])
                            cv2.imwrite('{}/{}'.format(imOutDir, fname), final_entropy_out)
                
            if self.g_pool.save_masks:
                # save mask
                fname = "eye-{}_{:0.3f}_{}.png".format(eye_id, final_result['confidence'], frame.timestamp)
                self.saveMaskAsImage(img, seg_map, pl_pupil_ellipse, fileName=fname, flipImage=self.g_pool.ellseg_reverse)
                
                # save ellipse overlay
                fname = "eye-{}_{:0.3f}_{}.png".format(eye_id, final_result['confidence'], frame.timestamp)
                hstack_mask_imOutDir = os.path.join(self.g_pool.capture.source_path[0:self.g_pool.capture.source_path.rindex("\\")+1], "eye"+str(self.g_pool.eye_id)+"_mask_with_frame")
                os.makedirs(hstack_mask_imOutDir, exist_ok=True)
                
                hstack_mask_out_left = np.stack((np.copy(img),)*3, axis=-1)
                
                hstack_seg_map_right = np.stack((np.copy(img),)*3, axis=-1)
                final_result_ellipse = final_result["ellipse"]
                elcenter = final_result_ellipse["center"]
                elaxes = final_result_ellipse["axes"] # axis diameters
                cv2.ellipse(hstack_seg_map_right,
                    (round(elcenter[0]), round(elcenter[1])),
                    (round(elaxes[0]/2), round(elaxes[1]/2)), # convert diameters to radii
                    final_result_ellipse["angle"], 0, 360, (255, 0, 0), 1)
                    
                hstack_mask_out = cv2.hconcat([hstack_mask_out_left, hstack_seg_map_right])
                
                cv2.imwrite('{}/{}'.format(hstack_mask_imOutDir, fname), hstack_mask_out)
                
            if final_result['diameter'] < self.g_pool.ellseg_pupil_size_min:
                if self.g_pool.save_masks:
                    # write out image
                    imOutDir = os.path.join(self.g_pool.capture.source_path[0:self.g_pool.capture.source_path.rindex("\\")+1], "eye"+str(self.g_pool.eye_id)+"_eliminated_frame")
                    os.makedirs(imOutDir, exist_ok=True)
                    im = np.zeros((frame.height, frame.width, 3))
                    im[:, :, 0] = img
                    im[:, :, 1] = img
                    im[:, :, 2] = img
                    final_result_ellipse = final_result["ellipse"]
                    elcenter = final_result_ellipse["center"]
                    elaxes = final_result_ellipse["axes"] # axis diameters
                    cv2.ellipse(im,
                        (round(elcenter[0]), round(elcenter[1])),
                        (round(elaxes[0]/2), round(elaxes[1]/2)), # convert diameters to radii
                        final_result_ellipse["angle"], 0, 360, (255, 0, 0), 1)
                    fileName = "eye-{}_{:0.3f}_{}.png".format(self.g_pool.eye_id, final_result['confidence'], frame.timestamp)
                    cv2.imwrite("{}/{}".format(imOutDir, fileName), im)
                    # end write out image
                final_result["ellipse"] = {"center": (0.0, 0.0), "axes": (0.0, 0.0), "angle": 0.0}
                final_result["diameter"] = 0.0
                final_result["location"] = (0.0, 0.0)
                final_result['confidence'] = 0.0
            
            return final_result

        elif customEllipse:

            # OPTION 2: If custom ellipse setting is toggled on
            #########################################
            ### Ellipse data transformations

            # background, iris, pupil
            seg_map[np.where(seg_map == 0)] = 0
            seg_map[np.where(seg_map == 1)] = 0
            seg_map[np.where(seg_map == 2)] = 255
            seg_map = np.array(seg_map, dtype=np.uint8)

            openCVformatPupil = np.copy(ellseg_pupil_ellipse)

            if (ellseg_pupil_ellipse[4]) > np.pi / 2.0:
                ellseg_pupil_ellipse[4] = ellseg_pupil_ellipse[4] - np.pi / 2.0

            if (ellseg_pupil_ellipse[4]) < -np.pi / 2.0:
                ellseg_pupil_ellipse[4] = ellseg_pupil_ellipse[4] + np.pi / 2.0

            ellseg_pupil_ellipse[4] = np.rad2deg(ellseg_pupil_ellipse[4])

            #########################################

            if self.g_pool.ellseg_debug:
                seg_map_debug = np.stack((np.copy(seg_map),)*3, axis=-1)
                cv2.ellipse(seg_map_debug,
                    (round(ellseg_pupil_ellipse[0]), round(ellseg_pupil_ellipse[1])),
                    (round(ellseg_pupil_ellipse[2]), round(ellseg_pupil_ellipse[3])),
                    ellseg_pupil_ellipse[4], 0, 360, (255, 0, 0), 1)
                cv2.imshow(debugOutputWindowName, seg_map_debug)
                


            confidence = self.calcConfidence(ellseg_pupil_ellipse, seg_map)

            if self.g_pool.save_masks == True:
                fname = "eye-{}_{:0.3f}.png".format(eye_id, confidence)
                self.saveMaskAsImage(img,seg_map,openCVformatPupil,fname,eye_id)


            eye_id = self.g_pool.eye_id

            result["id"] = eye_id
            result["topic"] = f"pupil.{eye_id}.{self.identifier}"

            ellipse["center"] = (ellseg_pupil_ellipse[0], ellseg_pupil_ellipse[1])
            ellipse["axes"] = (ellseg_pupil_ellipse[2]*2, ellseg_pupil_ellipse[3]*2)
            ellipse["angle"] = ellseg_pupil_ellipse[4]

            result["ellipse"] = ellipse
            result["diameter"] = ellseg_pupil_ellipse[2]*2
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
            
            if result['diameter'] < self.g_pool.ellseg_pupil_size_min:
                if self.g_pool.save_masks:
                    # write out image
                    imOutDir = os.path.join(self.g_pool.capture.source_path[0:self.g_pool.capture.source_path.rindex("\\")+1], "eye"+str(self.g_pool.eye_id)+"_eliminated_frame")
                    os.makedirs(imOutDir, exist_ok=True)
                    im = np.zeros((frame.height, frame.width, 3))
                    im[:, :, 0] = img
                    im[:, :, 1] = img
                    im[:, :, 2] = img
                    final_result_ellipse = result["ellipse"]
                    elcenter = final_result_ellipse["center"]
                    elaxes = final_result_ellipse["axes"] # axis diameters
                    cv2.ellipse(im,
                        (round(elcenter[0]), round(elcenter[1])),
                        (round(elaxes[0]/2), round(elaxes[1]/2)), # convert diameters to radii
                        final_result_ellipse["angle"], 0, 360, (255, 0, 0), 1)
                    fileName = "eye-{}_{:0.3f}_{}.png".format(self.g_pool.eye_id, result['confidence'], frame.timestamp)
                    cv2.imwrite("{}/{}".format(imOutDir, fileName), im)
                    # end write out image
                result["ellipse"] = {"center": (0.0, 0.0), "axes": (0.0, 0.0), "angle": 0.0}
                result["diameter"] = 0.0
                result["location"] = (0.0, 0.0)
                result['confidence'] = 0.0
                
            if self.g_pool.save_masks:
                # save ellipse overlay
                fname = "eye-{}_{:0.3f}_{}.png".format(eye_id, final_result['confidence'], frame.timestamp)
                hstack_mask_imOutDir = os.path.join(self.g_pool.capture.source_path[0:self.g_pool.capture.source_path.rindex("\\")+1], "eye"+str(self.g_pool.eye_id)+"_mask_with_frame")
                os.makedirs(hstack_mask_imOutDir, exist_ok=True)
                
                hstack_mask_out_left = np.stack((np.copy(img),)*3, axis=-1)
                
                hstack_seg_map_right = np.stack((np.copy(img),)*3, axis=-1)
                cv2.ellipse(hstack_seg_map_right,
                    (round(ellseg_pupil_ellipse[0]), round(ellseg_pupil_ellipse[1])),
                    (round(ellseg_pupil_ellipse[2]), round(ellseg_pupil_ellipse[3])),
                    ellseg_pupil_ellipse[4], 0, 360, (255, 0, 0), 1)
                
                hstack_mask_out = cv2.hconcat([hstack_mask_out_left, hstack_seg_map_right])
                
                cv2.imwrite('{}/{}'.format(hstack_mask_imOutDir, fname), hstack_mask_out)
            
            return result

    def gl_display(self):
        if self._recent_detection_result:
            draw_pupil_outline(self._recent_detection_result, color_rgb=(1, 0, 1))
    
    def init_ui(self):
        super(Detector2DPlugin, self).init_ui()
        self.menu.label = self.pretty_class_name
        self.menu_icon.label_font = "pupil_icons"
        info = ui.Info_Text(
            "(PURPLE) Model using ESF-Net, the \"baseline_edge_16\" model."
        )
        self.menu.append(info)
        self.menu.append(
            ui.Slider(
                "ellseg_pupil_size_min",
                self.g_pool,
                label="Pupil min diameter (pixels)",
                min=1,
                max=250,
                step=1,
            )
        )
        self.menu.append(
            ui.Switch(
                "ellseg_reverse",
                self.g_pool,
                label="Flip image vertically before processing"
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

        self.menu.append(
            ui.Switch(
                "save_masks",
                self.g_pool,
                label="Save segmentation masks to disk"
            )
        )

        self.menu.append(
            ui.Switch(
                "calcCustomConfidence",
                self.g_pool,
                label="Use custom confidence metric"
            )
        )
