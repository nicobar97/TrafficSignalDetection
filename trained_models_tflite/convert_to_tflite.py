import tensorflow as tf
import numpy as np
import cv2
import os

localpb = '/usr/local/anaconda3/envs/tensorflow1/models/research/object_detection/all_trains/inception_v2_75K_tflite_graph/tflite_graph.pb'
tflite_file = '/usr/local/anaconda3/envs/tensorflow1/models/research/object_detection/all_trains/inception_v2_75K_tflite_graph/primo.tflite'
img_dir = '/Users/nicolobartelucci/Desktop/dataset_crucco/train_png/'
path = os.listdir(img_dir)
def rep_data_gen():
    a = []
    for file_name in path:
        if int(file_name[2:-4]) <= 160:
            #print(file_name[2:-4])
            img = cv2.imread(img_dir + file_name)
            img = cv2.resize(img, (300, 300))
            img = img / 255
            img = img.astype(np.float32)
            a.append(img)
    a = np.array(a)
    #print(a.shape) # a is np array of 160 3D images
    img = tf.data.Dataset.from_tensor_slices(a).batch(1)
    for i in img.take(4):
        #print(i)
        yield [i]


converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
    localpb,
    ['normalized_input_image_tensor'],
    ['raw_outputs/box_encodings','raw_outputs/class_predictions'],
    input_shapes={"normalized_input_image_tensor" : [1, 300, 300, 3]}
)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
converter.inference_input_type = tf.uint8
converter.representative_dataset=rep_data_gen
tflite_model = converter.convert()
open(tflite_file,'wb').write(tflite_model)