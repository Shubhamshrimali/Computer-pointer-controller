# Computer Pointer Controller
In this project we will controls the computer pointer with the use of Face direction. It supports input from video file and camera video stream. 

## Project Set Up and Installation
**First step**
Make sure you have the OpenVINO toolkit installed on your system. This project is based on [Intel OpenVINO 2020.2.117](https://docs.openvinotoolkit.org/2020.2/index.html) toolkit, so if you don't have it, make sure to install it first before continue with the next steps.

**Second step**
You have to install the pretrained models needed for this project. The following instructions are for mac.
First you have to initialize the openVINO environment
```bash
source /opt/intel/openvino/bin/setupvars.sh
```
You have to run the above command every time you open an new terminal window.
We need the following models
- [Face Detection Model](https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html)

To download them run the following commands after you have created a folder with name `model` and got into it.
**Face Detection Model**
```bash
$ python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "face-detection-adas-binary-0001"
```
**Third step**
Install the requirements:
```bash
$ pip3 install -r requirements.txt
```


## Demo
To run the application use the following command
```bash
$ python3 main.py -f model/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml -fl -i demo.mp4
```

## Documentation
This application supports the following command line parameters
```bash
$ python3 main.py --help
usage: main.py [-h] -f FACEDETECTIONMODEL -fl FACIALLANDMARKMODEL -hp
               HEADPOSEMODEL -g GAZEESTIMATIONMODEL -i INPUT
               [-flags PREVIEWFLAGS [PREVIEWFLAGS ...]] [-l CPU_EXTENSION]
               [-prob PROB_THRESHOLD] [-d DEVICE]

optional arguments:
  -h, --help            show this help message and exit
  -f FACEDETECTIONMODEL, --facedetectionmodel FACEDETECTIONMODEL
                        Path to .xml file of Face Detection model.
  -i INPUT, --input INPUT
                        Path to video file or enter cam for webcam
  -flags PREVIEWFLAGS [PREVIEWFLAGS ...], --previewFlags PREVIEWFLAGS [PREVIEWFLAGS ...]
                        Specify the flags from fd, fld, hp, ge like --flags fd
                        hp fld (Seperated by space)for see the visualization
                        of different model outputs of each frame,fd for Face
                        Detection, fld for Facial Landmark Detectionhp for
                        Head Pose Estimation, ge for Gaze Estimation.
  -l CPU_EXTENSION, --cpu_extension CPU_EXTENSION
                        path of extensions if any layers is incompatible with
                        hardware
  -prob PROB_THRESHOLD, --prob_threshold PROB_THRESHOLD
                        Probability threshold for model to identify the face .
  -d DEVICE, --device DEVICE
                        Specify the target device to run on: CPU, GPU, FPGA or
                        MYRIAD is acceptable. Sample will look for a suitable
                        plugin for device (CPU by default)
```

## Benchmarks
I ran the benchmarks, using different model precisions, on my macbook pro 2015 model which is equipped with
- Intel Core i7 4980HQ 2.5ghz
- 16 GB Ram

The results are the following
- FP32:
  - The total model loading time is : 0.854129sec
  - The total inference time is : 1.259327sec
  - The total FPS is : 9.445078fps
  

## Results
As we can see the results, models with lower precision gives us better inference time but they loose in accuracy. This happens because lower precision model uses less memory and they are less computationally expensive.

## Stand Out Suggestions
I tried different precision to models to improve inference time and accuracy. Also i am allowing user to select video or camera as input file to the application.

### Edge Cases
There are several edge cases that can be experienced while running inference on video file or camera stream.

1. Multiple people in the frame

   In this happens, application selects one person to work with. This solution works in most cases but may introduce flickering effect between two heads.

2. No head detected in the frame

   In this case application skips the frame and informs the user.
