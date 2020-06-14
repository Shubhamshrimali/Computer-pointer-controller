# Computer Pointer Controller
This project is an application that controls the computer pointer with the use of human eye gaze direction. It supports input from video file and camera video stream. 

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
- [Facial Landmarks Detection Model](https://docs.openvinotoolkit.org/latest/_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html)
- [Head Pose Estimation Model](https://docs.openvinotoolkit.org/latest/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html)
- [Gaze Estimation Model](https://docs.openvinotoolkit.org/latest/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html)

To download them run the following commands after you have created a folder with name `model` and got into it.
**Face Detection Model**
```bash
$ python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "face-detection-adas-binary-0001"
```
**Facial Landmarks Detection**
```bash
$ python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "landmarks-regression-retail-0009"
```
**Head Pose Estimation**
```bash
$ python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "head-pose-estimation-adas-0001"
```
**Gaze Estimation Model**
```bash
$ python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "gaze-estimation-adas-0002"
```

**Third step**
Install the requirements:
```bash
$ pip3 install -r requirements.txt
```

**Project structure**
```bash
|
|--demo.mp4
|--model
    |--intel
        |--face-detection-adas-binary-0001
        |--gaze-estimation-adas-0002
        |--head-pose-estimation-adas-0001
        |--landmarks-regression-retail-0009
|--face_detection.py
|--facial_landmarks_detection.py
|--gaze_estimation.py
|--input_feeder.py
|--main.py
|--mouse_controller.py
|--README.md
|--requirements.txt
```

## Demo
To run the application use the following command
```bash
$ python3 main.py -f model/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml -fl model/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml -hp model/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml -g model/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002.xml -i demo.mp4
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
  -fl FACIALLANDMARKMODEL, --faciallandmarkmodel FACIALLANDMARKMODEL
                        Path to .xml file of Facial Landmark Detection model.
  -hp HEADPOSEMODEL, --headposemodel HEADPOSEMODEL
                        Path to .xml file of Head Pose Estimation model.
  -g GAZEESTIMATIONMODEL, --gazeestimationmodel GAZEESTIMATIONMODEL
                        Path to .xml file of Gaze Estimation model.
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
  
- FP16:
  - The total model loading time is : 0.83298sec
  - The total inference time is : 1.086544sec
  - The total FPS is : 9.923655fps
  
- INT8:
  - The total model loading time is : 0,870566sec
  - The total inference time is : 1.023654sec
  - The total FPS is : 9.954624fps

I also ran the model on an INtel HD Graphics 620 and the results are the following
- FP32:
  - The total model loading time is : 15.65994sec
  - The total inference time is : 12.032688sec
  - The total FPS is : 2.47852fps
  
- FP16:
  - The total model loading time is : 16.233607sec
  - The total inference time is : 12.24576sec
  - The total FPS is : 2.755961fps

## Results
As we can see from the above results, models with lower precision gives us better inference time but they loose in accuracy. This happens because lower precision model uses less memory and they are less computationally expensive.

## Stand Out Suggestions
I tried different precision to models to improve inference time and accuracy. Also i am allowing user to select video or camera as input file to the application.

### Edge Cases
There are several edge cases that can be experienced while running inference on video file or camera stream.

1. Multiple people in the frame

   In this happens, application selects one person to work with. This solution works in most cases but may introduce flickering effect between two heads.

2. No head detected in the frame

   In this case application skips the frame and informs the user.


## License:
MIT License

Copyright (c) 2020 George Koukouris

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
