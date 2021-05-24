import sys
if "--plugin=ellseg" in sys.argv:                                   # ELLSEG
    from plugins.detector_2d_ritnet_ellseg_pupil_plugin import *
elif "--plugin=bestmodel" in sys.argv:                              # BESTMODEL
    from plugins.detector_2d_ritnet_bestmodel_plugin import *
elif "--plugin=ritnetpupil" in sys.argv:                            # RITNET PUPIL
    from plugins.detector_2d_ritnet_pupil_plugin import *
else:                                                               # DEFAULT
    from pupil_detector_plugins.detector_2d_plugin import *