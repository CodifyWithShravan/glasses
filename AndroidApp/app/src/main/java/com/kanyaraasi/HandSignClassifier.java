package com.kanyaraasi;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;

import com.google.mediapipe.framework.image.BitmapImageBuilder;
import com.google.mediapipe.framework.image.MPImage;
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark;
import com.google.mediapipe.tasks.core.BaseOptions;
import com.google.mediapipe.tasks.vision.core.RunningMode;
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker;
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/**
 * HandSignClassifier
 *
 * Stage 1: MediaPipe HandLandmarker detects 21 hand landmarks.
 * Stage 2: Embedded pure-Java Neural Network classifies landmarks into signs.
 */
public class HandSignClassifier {
    private static final String TAG = "HandSignClassifier";
    private static final String HAND_LANDMARKER_MODEL = "hand_landmarker.task";
    private static final String SIGN_WEIGHTS_FILE = "sign_weights.json";
    private static final float MIN_CONFIDENCE = 0.85f; // High confidence threshold for instant clean triggers
    private static final long COOLDOWN_MS = 2500; // Cooldown before repeating same spoken sign

    private HandLandmarker handLandmarker;
    private List<String> labels = new ArrayList<>();

    // Neural Network weights
    private float[][] W1; // 63 x 64
    private float[] b1;   // 64
    private float[][] W2; // 64 x 32
    private float[] b2;   // 32
    private float[][] W3; // 32 x numClasses
    private float[] b3;   // numClasses
    private boolean modelLoaded = false;

    // Fast Instant Detection & Debouncing
    private String lastConfirmedSign = null;
    private long lastSignTimestamp = 0;
    private int framesScanned = 0;

    /**
     * Initializes the classifier, loading models and weights.
     *
     * @param context the application context
     */
    public HandSignClassifier(Context context) {
        try {
            Log.d(TAG, "=== Initializing HandSignClassifier ===");
            setupHandLandmarker(context);
            loadModelWeights(context);
            Log.d(TAG, "HandLandmarker initialized: " + (handLandmarker != null));
            Log.d(TAG, "Model weights loaded: " + modelLoaded + " | Labels: " + labels);
            Log.d(TAG, "=== HandSignClassifier ready ===");
        } catch (Exception e) {
            Log.e(TAG, "Error initializing HandSignClassifier", e);
        }
    }

    private void setupHandLandmarker(Context context) {
        try {
            BaseOptions baseOptions = BaseOptions.builder()
                    .setModelAssetPath(HAND_LANDMARKER_MODEL)
                    .build();

            HandLandmarker.HandLandmarkerOptions options = HandLandmarker.HandLandmarkerOptions.builder()
                    .setBaseOptions(baseOptions)
                    // VIDEO mode tracks landmarks across frames. IMAGE mode starts a
                    // full hand search every time, which is needlessly slow for a stream.
                    .setRunningMode(RunningMode.VIDEO)
                    .setNumHands(1)
                    .setMinHandDetectionConfidence(0.3f)
                    .build();

            handLandmarker = HandLandmarker.createFromOptions(context, options);
            Log.d(TAG, "HandLandmarker created successfully");
        } catch (Exception e) {
            Log.e(TAG, "FAILED to create HandLandmarker", e);
            handLandmarker = null;
        }
    }

    private void loadModelWeights(Context context) {
        try (InputStream is = context.getAssets().open(SIGN_WEIGHTS_FILE)) {
            int size = is.available();
            byte[] buffer = new byte[size];
            is.read(buffer);
            String jsonStr = new String(buffer, StandardCharsets.UTF_8);

            JSONObject json = new JSONObject(jsonStr);

            // Labels
            JSONArray labelsArr = json.getJSONArray("labels");
            labels = new ArrayList<>();
            for (int i = 0; i < labelsArr.length(); i++) {
                labels.add(labelsArr.getString(i));
            }

            // W1 (63 x 64)
            JSONArray w1Arr = json.getJSONArray("W1");
            W1 = new float[w1Arr.length()][w1Arr.getJSONArray(0).length()];
            for (int i = 0; i < W1.length; i++) {
                JSONArray row = w1Arr.getJSONArray(i);
                for (int j = 0; j < W1[0].length; j++) {
                    W1[i][j] = (float) row.getDouble(j);
                }
            }

            // b1 (64)
            JSONArray b1Arr = json.getJSONArray("b1");
            b1 = new float[b1Arr.length()];
            for (int i = 0; i < b1.length; i++) {
                b1[i] = (float) b1Arr.getDouble(i);
            }

            // W2 (64 x 32)
            JSONArray w2Arr = json.getJSONArray("W2");
            W2 = new float[w2Arr.length()][w2Arr.getJSONArray(0).length()];
            for (int i = 0; i < W2.length; i++) {
                JSONArray row = w2Arr.getJSONArray(i);
                for (int j = 0; j < W2[0].length; j++) {
                    W2[i][j] = (float) row.getDouble(j);
                }
            }

            // b2 (32)
            JSONArray b2Arr = json.getJSONArray("b2");
            b2 = new float[b2Arr.length()];
            for (int i = 0; i < b2.length; i++) {
                b2[i] = (float) b2Arr.getDouble(i);
            }

            // W3 (32 x numClasses)
            JSONArray w3Arr = json.getJSONArray("W3");
            W3 = new float[w3Arr.length()][w3Arr.getJSONArray(0).length()];
            for (int i = 0; i < W3.length; i++) {
                JSONArray row = w3Arr.getJSONArray(i);
                for (int j = 0; j < W3[0].length; j++) {
                    W3[i][j] = (float) row.getDouble(j);
                }
            }

            // b3 (numClasses)
            JSONArray b3Arr = json.getJSONArray("b3");
            b3 = new float[b3Arr.length()];
            for (int i = 0; i < b3.length; i++) {
                b3[i] = (float) b3Arr.getDouble(i);
            }

            modelLoaded = true;
            Log.d(TAG, "Neural network weights loaded successfully: " + labels);
        } catch (Exception e) {
            Log.e(TAG, "Failed to load model weights from JSON", e);
            modelLoaded = false;
        }
    }

    /**
     * Pure Java Forward Pass (Inference)
     * Input: float[63] -> Dense(64, ReLU) -> Dense(32, ReLU) -> Dense(numClasses, Softmax)
     */
    private float[] forward(float[] x) {
        if (!modelLoaded || W1 == null) return new float[0];

        // Layer 1: Dense(64, ReLU)
        float[] a1 = new float[64];
        for (int j = 0; j < 64; j++) {
            float sum = b1[j];
            for (int i = 0; i < 63; i++) {
                sum += x[i] * W1[i][j];
            }
            a1[j] = Math.max(0f, sum);
        }

        // Layer 2: Dense(32, ReLU)
        float[] a2 = new float[32];
        for (int j = 0; j < 32; j++) {
            float sum = b2[j];
            for (int i = 0; i < 64; i++) {
                sum += a1[i] * W2[i][j];
            }
            a2[j] = Math.max(0f, sum);
        }

        // Layer 3: Dense(numClasses, Softmax)
        int numClasses = b3.length;
        float[] z3 = new float[numClasses];
        float maxZ = Float.NEGATIVE_INFINITY;
        for (int j = 0; j < numClasses; j++) {
            float sum = b3[j];
            for (int i = 0; i < 32; i++) {
                sum += a2[i] * W3[i][j];
            }
            z3[j] = sum;
            if (sum > maxZ) maxZ = sum;
        }

        // Softmax
        float expSum = 0f;
        float[] probs = new float[numClasses];
        for (int j = 0; j < numClasses; j++) {
            probs[j] = (float) Math.exp(z3[j] - maxZ);
            expSum += probs[j];
        }
        for (int j = 0; j < numClasses; j++) {
            probs[j] /= (expSum > 0 ? expSum : 1f);
        }

        return probs;
    }

    /**
     * Runs hand landmark detection and subsequent sign classification on the provided bitmap.
     *
     * @param bitmap      The input image
     * @param timestampMs The timestamp of the frame in milliseconds
     * @return AIResult containing classification and diagnostic info
     */
    public AIResult classify(Bitmap bitmap, long timestampMs) {
        framesScanned++;

        if (handLandmarker == null) {
            return new AIResult(null, 0f, false, 0, "AI Error: MediaPipe model failed to load");
        }
        if (!modelLoaded || labels.isEmpty()) {
            return new AIResult(null, 0f, false, 0, "AI Error: Sign weights failed to load");
        }

        try {
            Bitmap argbBitmap = bitmap;
            if (bitmap.getConfig() != Bitmap.Config.ARGB_8888) {
                argbBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, false);
            }
            
            MPImage mpImage = new BitmapImageBuilder(argbBitmap).build();

            // Run MediaPipe hand detection
            HandLandmarkerResult result = handLandmarker.detectForVideo(
                    mpImage, Math.max(1L, timestampMs));

            if (result == null || result.landmarks().isEmpty()) {
                return new AIResult(null, 0f, false, 0, "AI: Scanning... (Frame #" + framesScanned + ", No hand)");
            }

            List<NormalizedLandmark> landmarks = result.landmarks().get(0);
            if (landmarks.size() != 21) {
                return new AIResult(null, 0f, false, 1, "AI: Hand detected (" + landmarks.size() + "/21 points)");
            }

            // Normalize landmarks relative to wrist (Landmark 0)
            NormalizedLandmark wrist = landmarks.get(0);
            float[] features = new float[63];
            for (int i = 0; i < 21; i++) {
                NormalizedLandmark landmark = landmarks.get(i);
                features[i * 3] = landmark.x() - wrist.x();
                features[i * 3 + 1] = landmark.y() - wrist.y();
                features[i * 3 + 2] = landmark.z() - wrist.z();
            }

            // Run fast Neural Network forward pass
            float[] output = forward(features);
            int numClasses = labels.size();

            // Find best prediction
            float maxConfidence = -1f;
            int maxIndex = -1;
            for (int i = 0; i < numClasses; i++) {
                if (output[i] > maxConfidence) {
                    maxConfidence = output[i];
                    maxIndex = i;
                }
            }

            if (maxIndex != -1 && maxConfidence >= MIN_CONFIDENCE) {
                String detectedLabel = labels.get(maxIndex);
                boolean isNew = true;

                if (detectedLabel.equals(lastConfirmedSign)) {
                    if (timestampMs - lastSignTimestamp < COOLDOWN_MS) {
                        isNew = false;
                    }
                }

                if (isNew) {
                    lastConfirmedSign = detectedLabel;
                    lastSignTimestamp = timestampMs;
                }

                String msg = "🎯 Recognized: " + detectedLabel + " (" + (int)(maxConfidence * 100) + "%)";
                return new AIResult(detectedLabel, maxConfidence, isNew, 1, msg);
            } else {
                // Hand is in frame, but posture doesn't strongly match any sign
                String msg = "AI: Hand detected (Uncertain / " + (int)(maxConfidence * 100) + "%)";
                return new AIResult(null, maxConfidence, false, 1, msg);
            }

        } catch (Exception e) {
            Log.e(TAG, "Exception during classification", e);
            return new AIResult(null, 0f, false, 0, "AI Error: " + e.getMessage());
        }
    }

    /**
     * Closes the classifier and releases resources.
     */
    public void close() {
        if (handLandmarker != null) {
            handLandmarker.close();
            handLandmarker = null;
        }
    }

    /**
     * Represents the result of sign classification with diagnostics.
     */
    public static class AIResult {
        public final String label;
        public final float confidence;
        public final boolean isNew;
        public final int handsCount;
        public final String statusMessage;

        public AIResult(String label, float confidence, boolean isNew, int handsCount, String statusMessage) {
            this.label = label;
            this.confidence = confidence;
            this.isNew = isNew;
            this.handsCount = handsCount;
            this.statusMessage = statusMessage;
        }
    }
}
