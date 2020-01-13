import tensorflow as tf
import numpy as np
import cv2
import os

localpb = '/usr/local/anaconda3/envs/tensorflow1/models/research/object_detection/all_trains/inception_v2_75K_tflite_graph/tflite_graph.pb'
tflite_file = '/usr/local/anaconda3/envs/tensorflow1/models/research/object_detection/all_trains/inception_v2_75K_tflite_graph/notquantized.tflite'

converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
    localpb,
    ['normalized_input_image_tensor'],
    ['TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3'],
    input_shapes={"normalized_input_image_tensor" : [1, 300, 300, 3]}
)
converter.allow_custom_ops = True;
tflite_model = converter.convert()
open(tflite_file,'wb').write(tflite_model)