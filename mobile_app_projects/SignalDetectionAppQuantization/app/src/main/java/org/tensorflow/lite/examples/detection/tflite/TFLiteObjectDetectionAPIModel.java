/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package org.tensorflow.lite.examples.detection.tflite;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.Trace;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Vector;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.examples.detection.env.Logger;

/**
 * Wrapper for frozen detection models trained using the Tensorflow Object Detection API:
 * github.com/tensorflow/models/tree/master/research/object_detection
 */
public class TFLiteObjectDetectionAPIModel implements Classifier {
  private static final Logger LOGGER = new Logger();

  // Only return this many results.
  private static final int NUM_DETECTIONS = 10;
  private static final int NUM_CLASSES = 43;
  // Float model
  private static final float IMAGE_MEAN = 128.0f;
  private static final float IMAGE_STD = 128.0f;
  // Number of threads in the java app
  private static final int NUM_THREADS = 4;
  private boolean isModelQuantized;
  // Config values.
  private int inputSize;
  // Pre-allocated buffers.
  private Vector<String> labels = new Vector<String>();
  private int[] intValues;
  // outputLocations: array of shape [Batchsize, NUM_DETECTIONS,4]
  // contains the location of detected boxes
  private float[][][] outputLocations;
  // outputClasses: array of shape [Batchsize, NUM_DETECTIONS]
  // contains the classes of detected boxes
  private float[][] outputClasses;
  // outputScores: array of shape [Batchsize, NUM_DETECTIONS]
  // contains the scores of detected boxes
  private float[][] outputScores;
  // numDetections: array of shape [Batchsize]
  // contains the number of detected boxes
  private float[] numDetections;

  private float[][][] boxEncodings;
  private float[][][] classPredictions;
  private static float[][] anchors;
  private ByteBuffer imgData;

  private Interpreter tfLite;
  private Interpreter tfLite2;

  private TFLiteObjectDetectionAPIModel() {
  }

  /**
   * Memory-map the model file in Assets.
   */
  private static MappedByteBuffer loadModelFile(AssetManager assets, String modelFilename)
          throws IOException {
    AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
    FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
    FileChannel fileChannel = inputStream.getChannel();
    long startOffset = fileDescriptor.getStartOffset();
    long declaredLength = fileDescriptor.getDeclaredLength();
    return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
  }

  /**
   * Initializes a native TensorFlow session for classifying images.
   *
   * @param assetManager  The asset manager to be used to load assets.
   * @param modelFilename The filepath of the model GraphDef protocol buffer.
   * @param labelFilename The filepath of label file for classes.
   * @param inputSize     The size of image input
   * @param isQuantized   Boolean representing model is quantized or not
   */
  public static Classifier create(
          final AssetManager assetManager,
          final String modelFilename,
          final String modelFilename2,
          final String labelFilename,
          final int inputSize,
          final boolean isQuantized)
          throws IOException {
    final TFLiteObjectDetectionAPIModel d = new TFLiteObjectDetectionAPIModel();

    InputStream labelsInput = null;
    String actualFilename = labelFilename.split("file:///android_asset/")[1];
    labelsInput = assetManager.open(actualFilename);
    BufferedReader br = null;
    br = new BufferedReader(new InputStreamReader(labelsInput));
    String line;
    while ((line = br.readLine()) != null) {
      LOGGER.w(line);
      d.labels.add(line);
    }
    br.close();
    //anchors = initAnchors("anchors");
    d.inputSize = inputSize;

    try {
      d.tfLite = new Interpreter(loadModelFile(assetManager, modelFilename));
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
    try {
      d.tfLite2 = new Interpreter(loadModelFile(assetManager, modelFilename2));
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
    d.isModelQuantized = isQuantized;
    // Pre-allocate buffers.
    int numBytesPerChannel;
    if (isQuantized) {
      numBytesPerChannel = 1; // Quantized
    } else {
      numBytesPerChannel = 4; // Floating point
    }
    d.imgData = ByteBuffer.allocateDirect(1 * d.inputSize * d.inputSize * 3 * numBytesPerChannel);
    d.imgData.order(ByteOrder.nativeOrder());
    d.intValues = new int[d.inputSize * d.inputSize];

    d.tfLite.setNumThreads(NUM_THREADS);
    d.outputLocations = new float[1][NUM_DETECTIONS][4];
    d.outputClasses = new float[1][NUM_DETECTIONS];
    d.outputScores = new float[1][NUM_DETECTIONS];
    d.numDetections = new float[1];
    return d;
  }

  @Override
  public List<Recognition> recognizeImage(final Bitmap bitmap) {
    // Log this method so that it can be analyzed with systrace.
    Trace.beginSection("recognizeImage");

    Trace.beginSection("preprocessBitmap");
    // Preprocess the image data from 0-255 int to normalized float based
    // on the provided parameters.
    bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

    imgData.rewind();
    for (int i = 0; i < inputSize; ++i) {
      for (int j = 0; j < inputSize; ++j) {
        int pixelValue = intValues[i * inputSize + j];
        if (isModelQuantized) {
          // Quantized model
          imgData.put((byte) ((pixelValue >> 16) & 0xFF));
          imgData.put((byte) ((pixelValue >> 8) & 0xFF));
          imgData.put((byte) (pixelValue & 0xFF));
        } else { // Float model
          imgData.putFloat((((pixelValue >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
          imgData.putFloat((((pixelValue >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
          imgData.putFloat(((pixelValue & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
        }
      }
    }
    Trace.endSection(); // preprocessBitmap

    // Copy the input data into TensorFlow.
    Trace.beginSection("feed");
    int NUM_ANCHORS = NUM_DETECTIONS;
    boxEncodings = new float[1][2034][4];
    classPredictions = new float[1][2034][44];
    outputLocations = new float[1][NUM_DETECTIONS][4];
    outputClasses = new float[1][NUM_DETECTIONS];
    outputScores = new float[1][NUM_DETECTIONS];
    numDetections = new float[1];

    Object[] inputArray = {imgData};
    Map<Integer, Object> outputMap = new HashMap<>();
    Map<Integer, Object> outputMap2 = new HashMap<>();
    outputMap.put(0, outputLocations);
    outputMap.put(1, outputClasses);
    outputMap.put(2, outputScores);
    outputMap.put(3, numDetections);
    outputMap2.put(0, boxEncodings);
    outputMap2.put(1, classPredictions);
    Trace.endSection();

    // Run the inference call.
    Trace.beginSection("run");
    tfLite.runForMultipleInputsOutputs(inputArray, outputMap2);
    Trace.endSection();
    Trace.beginSection("run2");
    Object[] inputArray2 = {outputMap2.get(0), outputMap2.get(1)};
    tfLite2.runForMultipleInputsOutputs(inputArray2, outputMap);
    Trace.endSection();
    //outputLocations = decodeBoxEncodings((float[][][])outputMap2.get(0), anchors, NUM_DETECTIONS);
    // Show the best detections.
    // after scaling them back to the input size.
    final ArrayList<Recognition> recognitions = new ArrayList<>(NUM_DETECTIONS);
    for (int i = 0; i < NUM_DETECTIONS; ++i) {
      final RectF detection =
              new RectF(
                      outputLocations[0][i][1] * inputSize,
                      outputLocations[0][i][0] * inputSize,
                      outputLocations[0][i][3] * inputSize,
                      outputLocations[0][i][2] * inputSize);

      // SSD Mobilenet V1 Model assumes class 0 is background class
      // in label file and class labels start from 1 to number_of_classes+1,
      // while outputClasses correspond to class index from 0 to number_of_classes
      int labelOffset = 1;
      recognitions.add(
              new Recognition(
                      "" + i,
                      labels.get((int) outputClasses[0][i] + labelOffset),
                      outputScores[0][i],
                      detection));
      /*recognitions.add(
              new Recognition(
                      "" + i,
                      "empty",
                      0.9f,
                      detection));*/
    }
    Trace.endSection(); // "recognizeImage"
    return recognitions;
  }

  @Override
  public void enableStatLogging(final boolean logStats) {
  }

  @Override
  public String getStatString() {
    return "";
  }

  @Override
  public void close() {
  }

  public void setNumThreads(int num_threads) {
    if (tfLite != null) tfLite.setNumThreads(num_threads);
    if (tfLite2 != null) tfLite2.setNumThreads(num_threads);
  }

  @Override
  public void setUseNNAPI(boolean isChecked) {
    if (tfLite != null) tfLite.setUseNNAPI(isChecked);
    if (tfLite2 != null) tfLite2.setUseNNAPI(isChecked);
  }

  /*private float[][][] decodeBoxEncodings(final float[][][] boxEncoding, final float[][] anchor, final int numBoxes) {
    final float[][][] decodedBoxes = new float[1][numBoxes][4];
    float y_scale = 10.0f;
    float x_scale = 10.0f;
    float h_scale = 5.0f;
    float w_scale = 5.0f;
    for (int i = 0; i < numBoxes; ++i) {

      final double ycenter = boxEncoding[0][i][0] / y_scale * anchor[i][2] + anchor[i][0];
      final double xcenter = boxEncoding[0][i][1] / x_scale * anchor[i][3] + anchor[i][1];
      final double half_h = 0.5 * Math.exp((boxEncoding[0][i][2] / h_scale)) * anchor[i][2];
      final double half_w = 0.5 * Math.exp((boxEncoding[0][i][3] / w_scale)) * anchor[i][3];

      decodedBoxes[0][i][0] = (float) (ycenter - half_h);   //ymin
      decodedBoxes[0][i][1] = (float) (xcenter - half_w);   //xmin
      decodedBoxes[0][i][2] = (float) (ycenter + half_h);   //ymax
      decodedBoxes[0][i][3] = (float) (xcenter + half_w);   //xmax
    }
    return decodedBoxes;
  }

  private static float[][] initAnchors(String anchorFilename) {
    float[][] anchors = new float[1917][4];
    try {
      BufferedReader br = new BufferedReader(new FileReader(new File(anchorFilename)));
      String line = null;
      int count = 0;
      int anchorCount = 0;
      while((line = br.readLine()) != null) {
          if(line.trim().equals("[") || line.trim().equals("]") || line.trim().equals("[,")) {
            //do nothing
          }
          else {
            if(count == 3) {
              count = 0;
              anchors[anchorCount][count] = Float.parseFloat(line);
            }
            else {
              line = line.substring(0,line.length()-1);
              anchors[anchorCount][count] = Float.parseFloat(line);
            }
            count++;
            anchorCount++;
          }
      }
    }
    catch(Exception e) {

    }
    return anchors;
  }*/
}