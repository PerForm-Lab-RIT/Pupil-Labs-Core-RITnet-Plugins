import typing as T
import numpy as np


from gaze_mapping.gazer_base import (
    Model,
    NotEnoughDataError,
)
from gaze_mapping.gazer_3d.gazer_hmd import (
    Gazer3D,
    ModelHMD3D_Binocular,
    ModelHMD3D_Monocular,
)


class PosthocGazerHMD3D(Gazer3D):
    label = "Post-hoc HMD 3D"

    @classmethod
    def _gazer_description_text(cls) -> str:
        return "Gaze mapping built specifically for HMD-Eyes."

    def __init__(self, g_pool, *, eye_translations=None, **kwargs):
        # Eye translations, relative to Unity main camera, in mm.
        # hmd-eyes calculates these dynamically here:
        # https://github.com/pupil-labs/hmd-eyes/blob/master/plugin/Scripts/Calibration.cs#L144-L151
        # Should correspond more or less to the subject's IPD
        # Format: relative right eye x/y/z, relative left eye x/y/z
        eye_translations_dummy_values = [33.35, 0, 0], [-33.35, 0, 0]
        # Use eye_translations if set, fallback to dummy values otherwise
        self.__eye_translations = eye_translations or eye_translations_dummy_values
        super().__init__(g_pool, **kwargs)

    @property
    def _gpool_capture_intrinsics_if_available(self) -> T.Optional[T.Any]:
        if hasattr(self.g_pool, "capture"):
            return self.g_pool.capture.intrinsics
        else:
            return None

    def _init_binocular_model(self) -> Model:
        return ModelHMD3D_Binocular(
            intrinsics=self._gpool_capture_intrinsics_if_available,
            eye_translations=self.__eye_translations,
        )

    def _init_left_model(self) -> Model:
        return ModelHMD3D_Monocular(
            intrinsics=self._gpool_capture_intrinsics_if_available
        )

    def _init_right_model(self) -> Model:
        return ModelHMD3D_Monocular(
            intrinsics=self._gpool_capture_intrinsics_if_available
        )

    def fit_on_calib_data(self, calib_data):
        # extract reference data
        ref_data = calib_data["ref_list"]
        # extract and filter pupil data
        pupil_data = calib_data["pupil_list"]
        pupil_data = self.filter_pupil_data(
            pupil_data, self.g_pool.min_calibration_confidence
        )
        # match pupil to reference data (left, right, and binocular)
        matches = self.match_pupil_to_ref(pupil_data, ref_data)
        if matches.binocular[0]:
            self._fit_binocular_model(self.binocular_model, matches.binocular)
            params = self.binocular_model.get_params()
            self.left_model.set_params(
                eye_camera_to_world_matrix=params["eye_camera_to_world_matrix1"],
                gaze_distance=self.binocular_model.last_gaze_distance,
            )
            self.right_model.set_params(
                eye_camera_to_world_matrix=params["eye_camera_to_world_matrix0"],
                gaze_distance=self.binocular_model.last_gaze_distance,
            )
            self.left_model.binocular_model = self.binocular_model
            self.right_model.binocular_model = self.binocular_model
        else:
            raise NotEnoughDataError

    def _extract_reference_features(self, ref_data) -> np.ndarray:
        # HMD-eyes data not available during post-hoc calib:
        # ref_3d = np.array([ref["mm_pos"] for ref in ref_data])
        # assert ref_3d.shape == (len(ref_data), 3), ref_3d
        # return ref_3d

        # Fall back to 2d reference data unprojection
        ref_3d = super()._extract_reference_features(ref_data)
        # calibrate_hmd() expects these to be in Unity coordinates, which has a flipped
        # y-axis compared to Capture's 3d coodinate system.
        
        ref_3d[:, 1] *= -1
        return ref_3d

    def get_init_dict(self):
        return {
            **super().get_init_dict(),
            "eye_translations": self.__eye_translations,
        }
