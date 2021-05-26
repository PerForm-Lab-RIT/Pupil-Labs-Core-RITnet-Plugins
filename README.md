# Pupil Labs Core RITnet Plugins
Pupil detector plugins for Pupil Labs Core that integrate various RITnet and EllSeg models.

## Autorun Instructions
1) Install Pupil Labs Core
2) Make sure you are in an environment with all the neccessary python plugins to run RITnet and Pupil Labs (through Anaconda, for example)
3) Move the contents of this repository into *player_settings/plugins/*
3) Fill out a .csv file, with one line per run you want the program to do, in the format shown below in the **Example Autorun CSV** section
4) Enter the *player_settings/plugins/autorun/* folder and use the following command:
```
  run.sh [-v] <in_file> <pupil_folder>
    [-v]: Verbose
```
  where <in_file> is the location of your instructions .csv, and <pupil_folder> is your Pupil Labs Core install location (the folder containing the *pupil_src/* folder).

## **Example Autorun CSV**
| RUN_NAME      | VIDEO_PATH | PLUGIN | SAVE_EYE0_MASKS | SAVE_EYE1_MASKS | CUSTOM_ELLIPSE |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| Subject 1 Vanilla      | C:\\...\P_201218121321_sub1\S001\PupilData\000 | vanilla | false | false | false |
| Subject 1 Ellseg   | C:\\...\P_201218121321_sub1\S001\PupilData\000 | ellseg | true | true | false |
| Subject 2 Vanilla   | C:\\...\P_201219105516_sub2\S001\PupilData\000 | vanilla | false | false | false |
| Subject 2 Ellseg   | C:\\...\P_201219105516_sub2\S001\PupilData\000 | ellseg | true | true | true |
| Subject 2 Bestmodel   | C:\\...\P_201219105516_sub2\S001\PupilData\000 | bestmodel | false | false | false |
| Subject 2 RITnet Pupil   | C:\\...\P_201219105516_sub2\S001\PupilData\000 | ritnetpupil | false | false | false |

- Note: **SAVE_EYE0_MASKS**, **SAVE_EYE1_MASKS**, and **CUSTOM_ELLIPSE** presently only work with **PLUGIN**: *ellseg*
