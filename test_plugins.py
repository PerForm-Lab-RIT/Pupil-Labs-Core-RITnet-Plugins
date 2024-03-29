import sys
import os

import cv2
import torch
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pupil', 'pupil_src','shared_modules'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pupil', 'pupil_src','shared_modules','pupil_detector_plugins'))
import file_methods as fm
from roi import Roi
from pupil_detector_plugins.detector_2d_plugin import Detector2DPlugin as Native
from plugins.detector_2d_ritnet_pupil_plugin import Detector2DRITnetPupilPlugin as RITnetPupil
from plugins.detector_2d_ritnet_ellsegv2_pupil_plugin import Detector2DRITnetEllsegV2AllvonePlugin as EllSegGen
from plugins.detector_2d_esfnet_pupil_plugin import Detector2DESFnetPlugin as ESFnet
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pupil-core-pipeline', 'src'))
from core.pipeline import fake_gpool, load_intrinsics

def main():
    print(torch.cuda.device_count())
    eye0_intrinsics_loc = "./eye0.intrinsics"
    eye1_intrinsics_loc = "./eye1.intrinsics"
    scene_cam_intrinsics_loc = "./world.intrinsics"
    eye0_intrinsics = load_intrinsics(eye0_intrinsics_loc)
    eye1_intrinsics = load_intrinsics(eye1_intrinsics_loc)
    scene_cam_intrinsics = load_intrinsics(scene_cam_intrinsics_loc, resolution=(640,480))
    intrinsics = [eye0_intrinsics, eye1_intrinsics, scene_cam_intrinsics]
    gpool = fake_gpool(intrinsics[0])
    gpool.eye_id = 0
    gpool.display_mode = "n/a"

    def correct_ellpse(p_ellipse):
        if (p_ellipse[4]) > np.pi / 2.0:
            p_ellipse[4] = p_ellipse[4] - np.pi / 2.0

        if (p_ellipse[4]) < -np.pi / 2.0:
            p_ellipse[4] = p_ellipse[4] + np.pi / 2.0

        p_ellipse[4] = np.rad2deg(p_ellipse[4])
        return p_ellipse

    results = {
        'im': [],
        'ritnet': [],
        'ellseg': [],
        'esfnet': []
    }
    
    def annotate_native_pupil(native, frame, timestamp):
        bgr = frame#[:, :]#, 0]
        bgr = bgr.copy(order="C")
        gray = frame[:, :, 0]
        gray = gray.copy(order="C")
        height, width = gray.shape
        roi = Roi(
            g_pool=gpool, frame_size=(width, height), bounds=(0, 0, width, height)
        )
        pupil_frame = lambda: None
        setattr(pupil_frame, "gray", gray)
        setattr(pupil_frame, "bgr", bgr)
        setattr(pupil_frame, "width", width)
        setattr(pupil_frame, "height", height)
        setattr(pupil_frame, "timestamp", timestamp)
        pupil_datum = native.detect(pupil_frame)
        pupil_datum = fm.Serialized_Dict(python_dict=pupil_datum)
        return cv2.ellipse(frame, (int(pupil_datum["ellipse"]["center"][0]), int(pupil_datum["ellipse"]["center"][1])),
                            (int(pupil_datum["ellipse"]["axes"][0] / 2), int(pupil_datum["ellipse"]["axes"][1] / 2)),
                            pupil_datum["ellipse"]["angle"], 0., 360., [0, 128, 255], 1)
    
    def colorize(frame, categories=(0, 1, 2)):
        cr = np.zeros((frame.shape[0], frame.shape[1], 3))
        cr[:,:,0] = (frame == categories[0])*128
        cr[:,:,1] = (frame == categories[1])*128
        cr[:,:,2] = (frame == categories[2])*128
        return cr
    
    BLACKLIST = ['759.png']
    DELTA = 0.3
    native = Native(g_pool=gpool, properties={
                            # eye 0 params
                            "intensity_range": 23,
                            "pupil_size_min": 10,
                            "pupil_size_max": 100
                        })
    detector = RITnetPupil(g_pool=gpool,properties=None)
    for i, image in enumerate(os.listdir('./detect')):
        if image in BLACKLIST:
            continue
        im = cv2.imread(f'./detect/{image}')
        results['im'].append(annotate_native_pupil(native, im.copy(), i))
        res = detector.detector_ritnet_2d.detect(im[:, :, 0], False)
        print(np.unique(res))

        res = (colorize(res, (191, 63, 0)) * DELTA) + (im * (1-DELTA))
        results['ritnet'].append(res)
    detector = EllSegGen(g_pool=gpool,properties=None)
    for image in os.listdir('./detect'):
        if image in BLACKLIST:
            continue
        im = cv2.imread(f'./detect/{image}')
        res, p_ellipse, i_ellipse, _ = detector.detector_ritnet_2d.detect(im[:, :, 0])
        #res = ((res * 127) - 254) * -1
        res = colorize(res)
        p_ellipse = correct_ellpse(p_ellipse)
        i_ellipse = correct_ellpse(i_ellipse)
        res = (res * DELTA) + (im * (1-DELTA))
        res = cv2.ellipse(res, (round(p_ellipse[0]), round(p_ellipse[1])),
                        (round(p_ellipse[2]), round(p_ellipse[3])),
                        p_ellipse[4], 0., 360., [0, 128, 255], 1)
        res = cv2.ellipse(res, (round(i_ellipse[0]), round(i_ellipse[1])),
                        (round(i_ellipse[2]), round(i_ellipse[3])),
                        i_ellipse[4], 0., 360., [0, 255, 0], 1)
        results['ellseg'].append(res)
    detector = ESFnet(g_pool=gpool,properties=None)
    detector.detector_ritnet_2d._custom_ellipse = True
    for image in os.listdir('./detect'):
        if image in BLACKLIST:
            continue
        im = cv2.imread(f'./detect/{image}')
        _, res, p_ellipse, i_ellipse = detector.detector_ritnet_2d.detect(im[:, :, 0])
        #res = ((res * 127) - 254) * -1
        res = colorize(res)
        p_ellipse = correct_ellpse(p_ellipse)
        i_ellipse = correct_ellpse(i_ellipse)
        res = (res * DELTA) + (im * (1-DELTA))
        res = cv2.ellipse(res, (round(p_ellipse[0]), round(p_ellipse[1])),
                        (round(p_ellipse[2]), round(p_ellipse[3])),
                        p_ellipse[4], 0., 360., [0, 128, 255], 1)
        res = cv2.ellipse(res, (round(i_ellipse[0]), round(i_ellipse[1])),
                        (round(i_ellipse[2]), round(i_ellipse[3])),
                        i_ellipse[4], 0., 360., [0, 255, 0], 1)
        results['esfnet'].append(res)
    
    originalim = np.concatenate(results['im'], axis=1)
    ritnetim = np.concatenate(results['ritnet'], axis=1)
    ellsegim = np.concatenate(results['ellseg'], axis=1)
    esfnetim = np.concatenate(results['esfnet'], axis=1)
    
    finalim = np.concatenate((originalim, ritnetim, ellsegim, esfnetim), axis=0)
    cv2.imwrite("out.png", finalim)
    

if __name__ == '__main__':
    main()
