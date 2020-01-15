# TrafficSignalDetection
A Traffic Signal Detection Android/iOS App made by using Object Detection API from TensorFlow<br>Sistemi Digitali M at UNIBO
# Downloads
Dataset: https://drive.google.com/file/d/1TNdAzz5U5sh1AtGJNtSHtfqfHDR_FYzZ/view?usp=sharing<br>
Models trained on COCO: https://drive.google.com/file/d/1SY9x8EA3pxPa2J7ntXLbt7Qb_uTQ_8_H/view?usp=sharing
# Setup
0. Set up Anaconda Environment For Python and install TensorFlow 1.15 and other dependencies needed.
1. Clone this repo.
2. Clone TensorFlow Model Repo (https://github.com/tensorflow/models).
3. Copy and overwrite TrafficSignalDetection content to research/object_detection folder.
4. Copy the content of Dataset .zip archive into research/object_detection folder.
5. Copy the content of Models .zip archive into research/object_detection folder.
# Train
1. In order to train your model you need to download the dataset we used or generate your own tfrecord files of your own dataset.
You can use generate_tfrecord.py script (in Dataset.zip) after modifying class bindings:
```
python generate_tfrecord.py --csv_input=train_labels.csv --image_dir=train --output_path=train.record
```
You need to provide a label file in the right format in .csv. 
Few example lines of annotation are below (train_labels.csv from Dataset.zip):
``` 
filename,width,height,class,xmin,ymin,xmax,ymax,
00000.png,1360,800,id11,774,411,815,446, 
....
....
....
00XXX.png,1360,800,idX,AAA,BBB,CCC,DDD,
```
Put the generated or the downloaded train.record and test.record in research/object_detection.
2. Pick one model's pipeline.config from Models .zip (SSD_Mobilenet_v2 gave us best results), open it and modify all PATH_TO_BE_CONFIGURED to match your system. 
3. Copy the label43signals.pbtxt (you can find it into Dataset.zip) and pipeline.config in research/object_detection/training folder.
4. Run train.py script to train:
```
python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/pipeline.config
```
5. When train is done you can export ssd_tflite_graph and convert to .tflite.
Exporting ssd_tflite_graph:
```
python export_tflite_ssd_graph.py 
--pipeline_config_path training/pipeline.config 
--trained_checkpoint_prefix training/model.ckpt-XXXXX --output_directory exported_inference_graph_tflite 
```
Converting to .tflite by using convert_to_tflite3.py inside research/object_detection/trained_models_tflite:
```
python trained_models_tflite/convert_to_tflite3.py
```
6. Now you can use .tflite model in Android or iOS app.<br>
Note: you can't fully quantize the model as custom operation aren't supported yet.<br>
Note2: use TensorFlow 2.0.0 to run this script.<br>
Note3: remember to change PATH_TO_BE_CONFIGURED inside pipeline.config file to match your system.
# Apps Usage 
Android and iOS apps projects are inside mobile_app_projects folder. Open them to Android Studio or xCode to sideload the app on your phone.<br>
Note: You need a real phone with a working mobile camera.<br>
Note2: You need to copy a .tflite in Model folder for xCode or Assets Folder on Android Studio to make the app work.
# Contributors
Nicolò Bartelucci @nicobargit<br>Milo Marchetti @imRaazy
