import cv2
import os
import logging
import numpy as np
import time
from argparse import ArgumentParser
from input_feeder import InputFeeder
from face_detection import Model_Face_Detection
from facial_landmarks_detection import Model_Facial_Landmarks_Detection
from gaze_estimation import Model_Gaze_Estimation
from head_pose_estimation import Model_Head_Pose_Estimation
from mouse_controller import MouseController


def build_argparser():
    parser = ArgumentParser()

    parser.add_argument("-f", "--facedetectionmodel", required=True, type=str)
    parser.add_argument("-fl", "--faciallandmarkmodel", required=True, type=str)
    parser.add_argument("-hp", "--headposemodel", required=True, type=str)
    parser.add_argument("-g", "--gazeestimationmodel", required=True, type=str)
    parser.add_argument("-i", "--input", required=True, type=str)
    parser.add_argument("-flags", "--previewFlags", required=False, nargs='+',
                        default=[])
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None)
    parser.add_argument("-prob", "--prob_threshold", required=False, type=float,
                        default=0.6)
    parser.add_argument("-d", "--device", type=str, default="CPU")
    
    return parser


def main():
    args = build_argparser().parse_args()
    preFlags = args.previewFlags
    
    log = logging.getLogger()
    input_File = args.input
    input_Feeder = None

    if input_File.lower() == "cam":
        input_Feeder = InputFeeder("cam")
    else:
        if not os.path.isfile(input_File):
            log.error("Unable to find input file")
            exit(1)
        
        input_Feeder = InputFeeder("video",input_File)

    loading_time = time.time()

    face = Model_Face_Detection(args.facedetectionmodel, args.device, args.cpu_extension)
    facial = Model_Facial_Landmarks_Detection(args.faciallandmarkmodel, args.device, args.cpu_extension)
    gaze = Model_Gaze_Estimation(args.gazeestimationmodel, args.device, args.cpu_extension)
    head = Model_Head_Pose_Estimation(args.headposemodel, args.device, args.cpu_extension)
    
    mouse = MouseController('medium','fast')

    input_Feeder.load_data()

    face.load_model()
    facial.load_model()
    gaze.load_model()
    head.load_model()

    model_loading_time = time.time() - loading_time
    
    counter = 0
    total_frame_count = 0
    inference_time = 0
    start_inf_time = time.time()
    for res, f in inputFeeder.next_batch():
        if not res:
            break;

        if f is not None:
            total_frame_count += 1
            if total_frame_count%5 == 0:
                cv2.imshow('video', cv2.resize(f, (500, 500)))
        
            key = cv2.waitKey(60)
            start_inference = time.time()

            crop_Face, face_coor = face.predict(f.copy(), args.prob_threshold)
            if type(crop_Face) == int:
                log.error("No face detected.")
                if key == 27:
                    break

                continue
            
            head_out = head.predict(crop_Face.copy())
            
            left_eye, right_eye, eye_coords = facial.predict(crop_Face.copy())
            
            mouse_coord, gaze_vec = gaze.predict(left_eye, right_eye, head_out)
            
            stop_inference = time.time()
            inference_time = inference_time + stop_inference - start_inference
            counter = counter + 1
            if (not len(preFlags) == 0):
                pre_window = f.copy()
                
                if 'fd' in preFlags:
                    if len(preFlags) != 1:
                        pre_window = crop_Face
                    else:
                        cv2.rectangle(pre_window, (face_coor[0], face_coor[1]), (face_coor[2], face_coor[3]), (0, 150, 0), 3)

                if 'fld' in preFlags:
                    if not 'fd' in preFlags:
                        pre_window = crop_Face.copy()

                    cv2.rectangle(pre_window, (eye_coords[0][0] - 10, eye_coords[0][1] - 10), (eye_coords[0][2] + 10, eye_coords[0][3] + 10), (0,255,0), 3)
                    cv2.rectangle(pre_window, (eye_coords[1][0] - 10, eye_coords[1][1] - 10), (eye_coords[1][2] + 10, eye_coords[1][3] + 10), (0,255,0), 3)
                    
                if 'hp' in preFlags:
                    cv2.putText(
                        pre_window, 
                        "Pose Angles: yaw:{:.2f} | pitch:{:.2f} | roll:{:.2f}".format(head_out[0], head_out[1], head_out[2]), (50, 50), 
                        cv2.FONT_HERSHEY_COMPLEX, 
                        1, 
                        (0, 255, 0), 
                        1, 
                        cv2.LINE_AA
                    )

                if 'ge' in preFlags:
                    if not 'fd' in preFlags:
                        pre_window = crop_Face.copy()

                    x, y, w = int(gaze_vec[0] * 12), int(gaze_vec[1] * 12), 160
                    
                    le = cv2.line(left_eye.copy(), (x - w, y - w), (x + w, y + w), (255, 0, 255), 2)
                    cv2.line(le, (x - w, y + w), (x + w, y - w), (255, 0, 255), 2)
                    
                    re = cv2.line(right_eye.copy(), (x - w, y - w), (x + w, y + w), (255, 0, 255), 2)
                    cv2.line(re, (x - w, y + w), (x + w, y - w), (255, 0, 255), 2)
                    
                    pre_window[eye_coords[0][1]:eye_coords[0][3], eye_coords[0][0]:eye_coords[0][2]] = le
                    pre_window[eye_coords[1][1]:eye_coords[1][3], eye_coords[1][0]:eye_coords[1][2]] = re
            
            if len(preFlags) != 0:
                img_hor = np.hstack((cv2.resize(f, (500, 500)), cv2.resize(pre_window, (500, 500))))
            else:
                img_hor = cv2.resize(f, (500, 500))

            cv2.imshow('Visualization', img_hor)

            if total_frame_count%5 == 0:
                mouse.move(mouse_coord[0], mouse_coord[1])    
            
            if key == 27:
                break

    fps = total_frame_count / inference_time

    log.error("video ended...")
    log.error("Total loading time of the models: " + str(model_loading_time) + " s")
    log.error("total inference time {} seconds".format(inference_time))
    log.error("Average inference time: " + str(inference_time/total_frame_count) + " s")
    log.error("fps {} frame/second".format(fps/5))

    cv2.destroyAllWindows()
    input_Feeder.close()
     
    
if __name__ == '__main__':
    main()
