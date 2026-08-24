package com.kanyaraasi;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.Rect;
import android.util.Log;

import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.List;

/**
 * FaceRecognitionEngine
 *
 * Combines Google ML Kit Face Detection with MobileFaceNet to detect faces
 * in camera frames and extract 192-dimensional facial feature embeddings.
 */
public class FaceRecognitionEngine {
    private static final String TAG = "FaceRecognitionEngine";
    private static final String MODEL_FILE = "mobile_facenet.tflite";

    private static final float IMAGE_MEAN = 127.5f;
    private static final float IMAGE_STD = 128.0f;

    private int inputImageWidth = 112;
    private int inputImageHeight = 112;
    private int embeddingSize = 192;

    private final Context context;
    private FaceDetector faceDetector;
    private Interpreter tfliteInterpreter;
    private boolean isReady = false;
    private String initError = null;

    public interface OnFaceProcessedListener {
        void onFaceProcessed(FaceData faceData);
        void onNoFaceDetected();
        void onError(String error);
    }

    public static class FaceData {
        public final Bitmap faceCrop;
        public final float[] embedding;
        public final Rect boundingBox;

        public FaceData(Bitmap faceCrop, float[] embedding, Rect boundingBox) {
            this.faceCrop = faceCrop;
            this.embedding = embedding;
            this.boundingBox = boundingBox;
        }
    }

    public FaceRecognitionEngine(Context context) {
        this.context = context;
        initialize();
    }

    private void initialize() {
        try {
            // 1. Setup ML Kit Face Detector (Fast mode for real-time mobile processing)
            FaceDetectorOptions detectorOptions = new FaceDetectorOptions.Builder()
                    .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
                    .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_NONE)
                    .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_NONE)
                    .setMinFaceSize(0.15f)
                    .build();
            faceDetector = FaceDetection.getClient(detectorOptions);

            // 2. Setup MobileFaceNet TFLite Interpreter using direct ByteBuffer
            ByteBuffer modelBuffer = loadModelFile(context, MODEL_FILE);
            Interpreter.Options tfliteOptions = new Interpreter.Options();
            tfliteInterpreter = new Interpreter(modelBuffer, tfliteOptions);

            int[] inShape = tfliteInterpreter.getInputTensor(0).shape();
            if (inShape != null && inShape.length >= 3) {
                inputImageHeight = inShape[1];
                inputImageWidth = inShape[2];
            }
            int[] outShape = tfliteInterpreter.getOutputTensor(0).shape();
            if (outShape != null && outShape.length >= 2) {
                embeddingSize = outShape[1];
            }

            isReady = true;
            initError = null;
            Log.d(TAG, "FaceRecognitionEngine initialized successfully: input=" + inputImageWidth + "x" + inputImageHeight + ", embeddingSize=" + embeddingSize);
        } catch (Exception e) {
            Log.e(TAG, "Failed to initialize FaceRecognitionEngine", e);
            initError = e.getMessage() != null ? e.getMessage() : e.getClass().getSimpleName();
            isReady = false;
        }
    }

    public boolean isReady() {
        return isReady;
    }

    public String getInitError() {
        return initError;
    }

    /**
     * Processes a camera frame: detects face, crops it, and extracts the embedding.
     */
    public void processFrame(Bitmap frame, OnFaceProcessedListener listener) {
        if (!isReady || faceDetector == null || tfliteInterpreter == null) {
            if (listener != null) listener.onError("Face recognition engine not ready");
            return;
        }

        if (frame == null || frame.isRecycled()) {
            if (listener != null) listener.onError("Invalid camera frame");
            return;
        }

        try {
            InputImage inputImage = InputImage.fromBitmap(frame, 0);

            faceDetector.process(inputImage)
                    .addOnSuccessListener(faces -> {
                        if (faces == null || faces.isEmpty()) {
                            if (listener != null) listener.onNoFaceDetected();
                            return;
                        }

                        // Pick the largest/most prominent face in the frame
                        Face primaryFace = getLargestFace(faces);
                        Rect bounds = primaryFace.getBoundingBox();

                        // Add 15% margin around the face for better feature extraction
                        Rect expandedBounds = getExpandedBounds(bounds, frame.getWidth(), frame.getHeight(), 0.15f);

                        // Crop face
                        Bitmap croppedFace = Bitmap.createBitmap(
                                frame,
                                expandedBounds.left,
                                expandedBounds.top,
                                expandedBounds.width(),
                                expandedBounds.height()
                        );

                        // Resize to model input size
                        Bitmap scaledFace = Bitmap.createScaledBitmap(croppedFace, inputImageWidth, inputImageHeight, true);

                        // Extract embedding vector
                        float[] embedding = extractEmbedding(scaledFace);

                        if (listener != null) {
                            listener.onFaceProcessed(new FaceData(scaledFace, embedding, bounds));
                        }
                    })
                    .addOnFailureListener(e -> {
                        Log.e(TAG, "Face detection failed", e);
                        if (listener != null) listener.onError("Face detection error: " + e.getMessage());
                    });

        } catch (Exception e) {
            Log.e(TAG, "Error in processFrame", e);
            if (listener != null) listener.onError("Processing exception: " + e.getMessage());
        }
    }

    /**
     * Runs MobileFaceNet on a face bitmap and returns an L2-normalized embedding.
     */
    public synchronized float[] extractEmbedding(Bitmap faceBitmap) {
        if (tfliteInterpreter == null || faceBitmap == null) {
            return new float[embeddingSize];
        }

        ByteBuffer inputBuffer = convertBitmapToByteBuffer(faceBitmap);
        float[][] output = new float[1][embeddingSize];

        tfliteInterpreter.run(inputBuffer, output);

        // L2-normalize the output embedding vector
        return normalizeL2(output[0]);
    }

    /**
     * Converts a Bitmap to a float ByteBuffer normalized to [-1, 1].
     */
    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(1 * inputImageWidth * inputImageHeight * 3 * 4);
        byteBuffer.order(ByteOrder.nativeOrder());

        int[] intValues = new int[inputImageWidth * inputImageHeight];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        int pixelIndex = 0;
        for (int i = 0; i < inputImageHeight; i++) {
            for (int j = 0; j < inputImageWidth; j++) {
                int val = intValues[pixelIndex++];
                // Normalization: (pixel - 127.5) / 128.0
                float r = (((val >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD;
                float g = (((val >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD;
                float b = ((val & 0xFF) - IMAGE_MEAN) / IMAGE_STD;

                byteBuffer.putFloat(r);
                byteBuffer.putFloat(g);
                byteBuffer.putFloat(b);
            }
        }
        return byteBuffer;
    }

    /**
     * Normalizes a vector to have unit L2 length (||v|| = 1.0).
     */
    private float[] normalizeL2(float[] vector) {
        float sumSquare = 0.0f;
        for (float v : vector) {
            sumSquare += v * v;
        }
        float norm = (float) Math.sqrt(sumSquare);
        if (norm == 0.0f) return vector;

        float[] normalized = new float[vector.length];
        for (int i = 0; i < vector.length; i++) {
            normalized[i] = vector[i] / norm;
        }
        return normalized;
    }

    private Face getLargestFace(List<Face> faces) {
        Face largest = faces.get(0);
        int maxArea = largest.getBoundingBox().width() * largest.getBoundingBox().height();
        for (int i = 1; i < faces.size(); i++) {
            Face f = faces.get(i);
            int area = f.getBoundingBox().width() * f.getBoundingBox().height();
            if (area > maxArea) {
                maxArea = area;
                largest = f;
            }
        }
        return largest;
    }

    private Rect getExpandedBounds(Rect bounds, int imageWidth, int imageHeight, float marginPercent) {
        int marginX = (int) (bounds.width() * marginPercent);
        int marginY = (int) (bounds.height() * marginPercent);

        int left = Math.max(0, bounds.left - marginX);
        int top = Math.max(0, bounds.top - marginY);
        int right = Math.min(imageWidth, bounds.right + marginX);
        int bottom = Math.min(imageHeight, bounds.bottom + marginY);

        return new Rect(left, top, right, bottom);
    }

    private ByteBuffer loadModelFile(Context context, String modelPath) throws IOException {
        try (java.io.InputStream is = context.getAssets().open(modelPath)) {
            java.io.ByteArrayOutputStream baos = new java.io.ByteArrayOutputStream();
            byte[] buffer = new byte[8192];
            int read;
            while ((read = is.read(buffer)) != -1) {
                baos.write(buffer, 0, read);
            }
            byte[] modelBytes = baos.toByteArray();
            ByteBuffer directBuffer = ByteBuffer.allocateDirect(modelBytes.length);
            directBuffer.order(ByteOrder.nativeOrder());
            directBuffer.put(modelBytes);
            directBuffer.rewind();
            return directBuffer;
        }
    }

    public void close() {
        if (faceDetector != null) {
            faceDetector.close();
            faceDetector = null;
        }
        if (tfliteInterpreter != null) {
            tfliteInterpreter.close();
            tfliteInterpreter = null;
        }
    }
}
