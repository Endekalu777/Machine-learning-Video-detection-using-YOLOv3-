import os
import sys
import argparse
import pandas as pd
import numpy as np
from PIL import Image
from timeit import default_timer as timer
import sys
sys.path.append(r"C:\Users\Ashis\Downloads\Compressed\Train your YOLO\c\TrainYourOwnYOLO-master\2_Training\src")
from keras_yolo3.yolo import YOLO, detect_video, detect_webcam
from utils import load_extractor_model, load_features, parse_input, detect_object
from Get_File_Paths import GetFileList
from Train_Utils import get_anchors
import tensorflow as tf

# Set environment variable to turn off oneDNN custom operations
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

def get_parent_dir(n=1):
    """Returns the n-th parent directory of the current working directory"""
    current_path = os.path.dirname(os.path.abspath(__file__))
    for _ in range(n):
        current_path = os.path.dirname(current_path)
    return current_path

src_path = os.path.join(get_parent_dir(1), "2_Training", "src")
utils_path = os.path.join(get_parent_dir(1), "Utils")
sys.path.append(src_path)
sys.path.append(utils_path)

# Other imports...
import test
import utils

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Set up folder names for default values
data_folder = os.path.join(get_parent_dir(n=1), "Data")
image_folder = os.path.join(data_folder, "Source_Images")
image_test_folder = os.path.join(image_folder, "Test_Images")
detection_results_folder = os.path.join(image_folder, "Test_Image_Detection_Results")
detection_results_file = os.path.join(detection_results_folder, "Detection_Results.csv")
model_folder = os.path.join(data_folder, "Model_Weights")
model_weights = os.path.join(model_folder, "trained_weights_final.h5")
model_classes = os.path.join(model_folder, "data_classes.txt")
anchors_path = os.path.join(src_path, "keras_yolo3", "model_data", "yolo_anchors.txt")

FLAGS = None

if __name__ == "__main__":
    # Delete all default flags
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    """
    Command line options
    """

    # Add argument definitions...

    FLAGS = parser.parse_args()

    save_img = not FLAGS.no_save_img
    file_types = FLAGS.file_types
    webcam_active = FLAGS.webcam

    if file_types:
        input_paths = GetFileList(FLAGS.input_path, endings=file_types)
    else:
        input_paths = GetFileList(FLAGS.input_path)

    input_image_paths = []
    input_video_paths = []
    for item in input_paths:
        if item.endswith(img_endings):
            input_image_paths.append(item)
        elif item.endswith(vid_endings):
            input_video_paths.append(item)

    output_path = FLAGS.output
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if FLAGS.is_tiny and FLAGS.anchors_path == anchors_path:
        anchors_path = os.path.join(
            os.path.dirname(FLAGS.anchors_path), "yolo-tiny_anchors.txt"
        )

    anchors = get_anchors(anchors_path)

    yolo = YOLO(
        **{
            "model_path": FLAGS.model_path,
            "anchors_path": anchors_path,
            "classes_path": FLAGS.classes_path,
            "score": FLAGS.score,
            "gpu_num": FLAGS.gpu_num,
            "model_image_size": (416, 416),
        }
    )

    out_df = pd.DataFrame(
        columns=[
            "image",
            "image_path",
            "xmin",
            "ymin",
            "xmax",
            "ymax",
            "label",
            "confidence",
            "x_size",
            "y_size",
        ]
    )

    class_file = open(FLAGS.classes_path, "r")
    input_labels = [line.rstrip("\n") for line in class_file.readlines()]

    if input_image_paths and not webcam_active:
        start = timer()
        for i, img_path in enumerate(input_image_paths):
            prediction, image = detect_object(
                yolo,
                img_path,
                save_img=save_img,
                save_img_path=FLAGS.output,
                postfix=FLAGS.postfix,
            )
            y_size, x_size, _ = np.array(image).shape
            for single_prediction in prediction:
                out_df = out_df.append(
                    pd.DataFrame(
                        [
                            [
                                os.path.basename(img_path.rstrip("\n")),
                                img_path.rstrip("\n"),
                            ]
                            + single_prediction
                            + [x_size, y_size]
                        ],
                        columns=[
                            "image",
                            "image_path",
                            "xmin",
                            "ymin",
                            "xmax",
                            "ymax",
                            "label",
                            "confidence",
                            "x_size",
                            "y_size",
                        ],
                    )
                )
        end = timer()
        print(
            "Processed {} images in {:.1f}sec - {:.1f}FPS".format(
                len(input_image_paths),
                end - start,
                len(input_image_paths) / (end - start),
            )
        )
        out_df.to_csv(FLAGS.box, index=False)

    if input_video_paths and not webcam_active:
        start = timer()
        for i, vid_path in enumerate(input_video_paths):
            output_path = os.path.join(
                FLAGS.output,
                os.path.basename(vid_path).replace(".", FLAGS.postfix + "."),
            )
            detect_video(yolo, vid_path, output_path=output_path)

        end = timer()
        print(
            "Processed {} videos in {:.1f}sec".format(
                len(input_video_paths), end - start
            )
        )

    if webcam_active:
        start = timer()
        detect_webcam(yolo)
        end = timer()
        print("Processed from webcam for {:.1f}sec".format(end - start))

    yolo.close_session()
