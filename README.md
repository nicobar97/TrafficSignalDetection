# TrafficSignalDetection
A Traffic Signal Detection Android/iOS App made by using Object Detection API from TensorFlow
Sistemi Digitali M at UNIBO
# Downloads
Dataset: https://drive.google.com/file/d/1TNdAzz5U5sh1AtGJNtSHtfqfHDR_FYzZ/view?usp=sharing<br>
Models trained on COCO: https://drive.google.com/file/d/1SY9x8EA3pxPa2J7ntXLbt7Qb_uTQ_8_H/view?usp=sharing
# Setup
1. Clone this repo
2. Clone TensorFlow Model Repo (https://github.com/tensorflow/models) 
3. Copy and overwrite TrafficSignalDetection content to research/object_detection folder.
4. Copy the content of Dataset .zip archive into research/object_detection folder.
# Train
If you want to train your model you need to download the dataset we used or generate the tfrecord files from your own dataset.
You can use generate_tfrecord.py script (in Dataset.zip) after modifying class bindings.
You need to use a label file in the right format. 
Few example lines of annotation are below:
``` 
filename,width,height,class,xmin,ymin,xmax,ymax,
00000.png,1360,800,id11,774,411,815,446, 
....
....
....
00XXX.png,1360,800,idX,AAA,BBB,CCC,DDD,
```
# Apps Usage 
Android and iOS apps projects are inside mobile_app_projects folder. Open them to Android Studio or xCode to sideload the app on your phone. 
Note: You need a real phone with a working mobile camera.
Note2: You need to copy a .tflite in Model folder for xCode or Assets Folder on Android Studio to make the app work
# Contributors
Nicolò Bartelucci @nicobargit
Milo Marchetti @imRaazy
