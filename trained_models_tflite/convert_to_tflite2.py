import tensorflow as tf
import numpy as np
import cv2
import os

localpb = '/usr/local/anaconda3/envs/tensorflow1/models/research/object_detection/all_trains/inception_v2_75K_tflite_graph/tflite_graph.pb'
tflite_file = '/usr/local/anaconda3/envs/tensorflow1/models/research/object_detection/all_trains/inception_v2_75K_tflite_graph/secondo.tflite'

converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
    localpb,
    ['raw_outputs/box_encodings','raw_outputs/class_predictions'],
    ['TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3'],
    input_shapes={'raw_outputs/box_encodings': [1, 1917, 4],'raw_outputs/class_predictions': [1, 1917, 44] }
)
converter.allow_custom_ops = True;
tflite_model = converter.convert()
open(tflite_file,'wb').write(tflite_model)