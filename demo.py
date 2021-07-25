# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2 
import tqdm
import numpy as np
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo
from adet.config import get_cfg
import pdb
# constants
WINDOW_NAME = "COCO detections"


# line_dict = {
#     0:"double white solid",
#     1:"double white dashed" ,
#     2:"double yellow solid" ,
#     3:"double yellow dashed" ,
#     4:"double other solid" ,
#     5:"double other dashed" ,
#     6:"single white solid" ,
#     7:"single white dashed" ,
#     8:"single yellow solid" ,
#     9:"single yellow dashed" ,
#     10:"single other solid" ,
#     11:"single other dashed" ,
# }

line_dict = {
    0:"dws",
    1:"dwd" ,
    2:"dys" ,
    3:"dyd" ,
    4:"dos" ,
    5:"dod" ,
    6:"sws" ,
    7:"swd" ,
    8:"sys" ,
    9:"syd" ,
    10:"sos" ,
    11:"sod" ,
}

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.MEInst.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Demo")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/e2e_mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument("--input", nargs="+", help="A list of space separated input images")
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.3,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    # pdb.set_trace()
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)
    # pdb.set_trace()
    if args.input:
        if os.path.isdir(args.input[0]):
            args.input = [os.path.join(args.input[0], fname) for fname in os.listdir(args.input[0])]
        elif len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            start_time = time.time()
            predictions, visualized_output,classes_temp,mask_temp = demo.run_on_image(img)
            print(classes_temp)
            
            instances_num=len(classes_temp)
            points=[]
            for n in range(instances_num):
                for i in range(len(mask_temp[n])):
                    for j in range(len(mask_temp[n][0])):
                        if(mask_temp[n][i][j]==True):
                            point=[i,j]
                            points.append(point)
                            break
                    if(len(points)==n+1):
                        break
            # print(points)
            logger.info(
                "{}: detected {} instances in {:.2f}s".format(
                    path, len(predictions["instances"]), time.time() - start_time
                )
            )

            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                visualized_output.save(out_filename)
                # visualized_output.save("out_filename.jpg")
                # 读入图片
                
                src = cv2.imread(out_filename)
                # 调用cv.putText()添加文字
                for i in range(len(classes_temp)):
                    # text = line_dict[classes_temp[i]]
                    text=str(classes_temp[i])
                    # AddText = src.copy()
                    img=cv2.putText(src, text, (points[i][1],points[i][0]), cv2.FONT_HERSHEY_COMPLEX, 1.0, (100, 200, 200), 5)
 
                    # 将原图片和添加文字后的图片拼接起来
                cv2.imwrite(out_filename,img)
                
            else:
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit
    elif args.webcam:
        assert args.input is None, "Cannot have both --input and --webcam!"
        cam = cv2.VideoCapture(0)
        for vis in tqdm.tqdm(demo.run_on_video(cam)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cv2.destroyAllWindows()
    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)

        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + ".mkv"
            else:
                output_fname = args.output
            assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*"x264"),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.video_input)
        for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
            if args.output:
                output_file.write(vis_frame)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()
