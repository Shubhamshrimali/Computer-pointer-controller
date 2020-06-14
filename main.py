import cv2
import os
import logging
import numpy as np
import time
from argparse import ArgumentParser
from input_feeder import InputFeeder
from face_detection import Model_Face_Detection
from mouse_controller import MouseController


def build_argparser():
    parser = ArgumentParser()

    parser.add_argument("-f", "--facedetectionmodel", required=True, type=str)
    parser.add_argument("-i", "--input", required=True, type=str)
    parser.add_argument("-flags", "--previewFlags", required=False, nargs='+',
                        default=[] )
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None)
    parser.add_argument("-prob", "--prob_threshold", required=False, type=float,
                        default=0.6)
    parser.add_argument("-d", "--device", type=str, default="CPU")
    
    return parser


def main():
    args = build_argparser().parse_args()
    previewFlags = args.previewFlags
    
    logger = logging.getLogger()
    inputFile = args.input
    inputFeeder = None

    if inputFile.lower() == "cam":
        inputFeeder = InputFeeder("cam")
    else:
        if not os.path.isfile(inputFile):
            logger.error("Unable to find input file")
            exit(1)
        
        inputFeeder = InputFeeder("video",inputFile)

    start_loading = time.time()

    mfd = Model_Face_Detection(args.facedetectionmodel, args.device, args.cpu_extension)
    
    mc = MouseController('medium','fast')

    inputFeeder.load_data()

    mfd.load_model()

    model_loading_time = time.time() - start_loading
    
    counter = 0
    frame_count = 0
    inference_time = 0
    start_inf_time = time.time()
    for ret, frame in inputFeeder.next_batch():
        if not ret:
            break;

        if frame is not None:
            frame_count += 1
            if frame_count%5 == 0:
                cv2.imshow('video', cv2.resize(frame, (500, 500)))
        
            key = cv2.waitKey(60)
            start_inference = time.time()

            croppedFace, face_coords = mfd.predict(frame.copy(), args.prob_threshold)
            if type(croppedFace) == int:
                logger.error("No face detected.")
                if key == 27:
                    break

                continue
           
            
            stop_inference = time.time()
            inference_time = inference_time + stop_inference - start_inference
            counter = counter + 1
            if (not len(previewFlags) == 0):
                preview_window = frame.copy()
                
                if 'fd' in previewFlags:
                    if len(previewFlags) != 1:
                        preview_window = croppedFace
                    else:
                        cv2.rectangle(preview_window, (face_coords[0], face_coords[1]), (face_coords[2], face_coords[3]), (0, 150, 0), 3)

            if len(previewFlags) != 0:
                img_hor = np.hstack((cv2.resize(frame, (500, 500)), cv2.resize(preview_window, (500, 500))))
            else:
                img_hor = cv2.resize(frame, (500, 500))

            cv2.imshow('Visualization', img_hor)

            if frame_count%5 == 0:
                mc.move(new_mouse_coord[0], new_mouse_coord[1])    
            
            if key == 27:
                break

    fps = frame_count / inference_time

    logger.error("Total loading time of the models: " + str(model_loading_time) + " s")
    logger.error("total inference time {} seconds".format(inference_time))
    logger.error("Average inference time: " + str(inference_time/frame_count) + " s")
    logger.error("fps {} frame/second".format(fps/5))

    cv2.destroyAllWindows()
    inputFeeder.close()
     
    
if __name__ == '__main__':
    main()
