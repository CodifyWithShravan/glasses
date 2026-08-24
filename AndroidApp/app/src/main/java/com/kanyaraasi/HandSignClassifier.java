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
import java.util.Collections;
import java.util.List;

/**
 * HandSignClassifier
 *
 * Stage 1: MediaPipe HandLandmarker detects up to TWO hands (Single-Hand & Dual-Hand support).
 * Stage 2: Embedded Pure-Java Neural Network classifies landmarks into signs in < 2ms.
 */
public class HandSignClassifier {
    private static final String TAG = "HandSignClassifier";
    private static final String HAND_LANDMARKER_MODEL = "hand_landmarker.task";
    private static final String SIGN_WEIGHTS_FILE = "sign_weights.json";
    private static final float MIN_CONFIDENCE = 0.85f; // Instant clean trigger confidence threshold
    private static final long COOLDOWN_MS = 2500; // Cooldown before repeating same spoken sign

    private HandLandmarker handLandmarker;
    private List<String> labels = new ArrayList<>();

    // Neural Network weights
    private float[][] W1; // inputDim x hidden1
    private float[] b1;   // hidden1
    private float[][] W2; // hidden1 x hidden2
    private float[] b2;   // hidden2
    private float[][] W3; // hidden2 x numClasses
    private float[] b3;   // numClasses
    private boolean modelLoaded = false;
    private int inputDimension = 126; // Default to 126 (dual hand), adapts dynamically to model

    // Fast Instant Detection & Debouncing
    private String lastConfirmedSign = null;
    private long lastSignTimestamp = 0;
    private int framesScanned = 0;

    public HandSignClassifier(Context context) {
        try {
            Log.d(TAG, "=== Initializing Dual-Hand HandSignClassifier ===");
            setupHandLandmarker(context);
            loadModelWeights(context);
            Log.d(TAG, "HandLandmarker initialized: " + (handLandmarker != null));
            Log.d(TAG, "Model weights loaded: " + modelLoaded + " | Input Dim: " + inputDimension + " | Labels: " + labels);
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

            // Configure MediaPipe to detect up to 2 hands simultaneously
            HandLandmarker.HandLandmarkerOptions options = HandLandmarker.HandLandmarkerOptions.builder()
                    .setBaseOptions(baseOptions)
                    .setRunningMode(RunningMode.IMAGE)
                    .setNumHands(2)
                    .setMinHandDetectionConfidence(0.35f)
                    .build();

            handLandmarker = HandLandmarker.createFromOptions(context, options);
            Log.d(TAG, "Dual-Hand Landmarker created successfully");
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

            // W1 (inputDim x hidden1)
            JSONArray w1Arr = json.getJSONArray("W1");
            inputDimension = w1Arr.length();
            int hidden1 = w1Arr.getJSONArray(0).length();
            W1 = new float[inputDimension][hidden1];
            for (int i = 0; i < inputDimension; i++) {
                JSONArray row = w1Arr.getJSONArray(i);
                for (int j = 0; j < hidden1; j++) {
                    W1[i][j] = (float) row.getDouble(j);
                }
            }

            // b1 (hidden1)
            JSONArray b1Arr = json.getJSONArray("b1");
            b1 = new float[b1Arr.length()];
            for (int i = 0; i < b1.length; i++) {
                b1[i] = (float) b1Arr.getDouble(i);
            }

            // W2 (hidden1 x hidden2)
            JSONArray w2Arr = json.getJSONArray("W2");
            int hidden2 = w2Arr.getJSONArray(0).length();
            W2 = new float[w2Arr.length()][hidden2];
            for (int i = 0; i < W2.length; i++) {
                JSONArray row = w2Arr.getJSONArray(i);
                for (int j = 0; j < hidden2; j++) {
                    W2[i][j] = (float) row.getDouble(j);
                }
            }

            // b2 (hidden2)
            JSONArray b2Arr = json.getJSONArray("b2");
            b2 = new float[b2Arr.length()];
            for (int i = 0; i < b2.length; i++) {
                b2[i] = (float) b2Arr.getDouble(i);
            }

            // W3 (hidden2 x numClasses)
            JSONArray w3Arr = json.getJSONArray("W3");
            int numClasses = w3Arr.getJSONArray(0).length();
            W3 = new float[w3Arr.length()][numClasses];
            for (int i = 0; i < W3.length; i++) {
                JSONArray row = w3Arr.getJSONArray(i);
                for (int j = 0; j < numClasses; j++) {
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
            Log.d(TAG, "Neural network weights loaded: inputDim=" + inputDimension + ", classes=" + labels);
        } catch (Exception e) {
            Log.e(TAG, "Failed to load model weights from JSON", e);
            modelLoaded = false;
        }
    }

    /**
     * Pure Java Forward Pass (Inference in < 2ms)
     */
    private float[] forward(float[] x) {
        if (!modelLoaded || W1 == null) return new float[0];

        int inputLen = Math.min(x.length, inputDimension);
        int hidden1 = b1.length;
        int hidden2 = b2.length;
        int numClasses = b3.length;

        // Layer 1: Dense(hidden1, ReLU)
        float[] a1 = new float[hidden1];
        for (int j = 0; j < hidden1; j++) {
            float sum = b1[j];
            for (int i = 0; i < inputLen; i++) {
                sum += x[i] * W1[i][j];
            }
            a1[j] = Math.max(0f, sum);
        }

        // Layer 2: Dense(hidden2, ReLU)
        float[] a2 = new float[hidden2];
        for (int j = 0; j < hidden2; j++) {
            float sum = b2[j];
            for (int i = 0; i < hidden1; i++) {
                sum += a1[i] * W2[i][j];
            }
            a2[j] = Math.max(0f, sum);
        }

        // Layer 3: Dense(numClasses, Softmax)
        float[] z3 = new float[numClasses];
        float maxZ = Float.NEGATIVE_INFINITY;
        for (int j = 0; j < numClasses; j++) {
            float sum = b3[j];
            for (int i = 0; i < hidden2; i++) {
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
     * Extracts a normalized 126-feature vector from up to 2 detected hands.
     * Sorts hands from left to right for orientation consistency.
     */
    private float[] extractFeatures(List<List<NormalizedLandmark>> allHands) {
        float[] features = new float[inputDimension];

        if (allHands == null || allHands.isEmpty()) {
            return features;
        }

        // Sort hands left-to-right by wrist X coordinate
        List<List<NormalizedLandmark>> sortedHands = new ArrayList<>(allHands);
        Collections.sort(sortedHands, (h1, h2) -> Float.compare(h1.get(0).x(), h2.get(0).x()));

        // Process Hand 1 (First 63 values)
        if (!sortedHands.isEmpty()) {
            List<NormalizedLandmark> hand1 = sortedHands.get(0);
            NormalizedLandmark wrist1 = hand1.get(0);
            for (int i = 0; i < Math.min(21, hand1.size()); i++) {
                NormalizedLandmark lm = hand1.get(i);
                int base = i * 3;
                if (base + 2 < features.length) {
                    features[base] = lm.x() - wrist1.x();
                    features[base + 1] = lm.y() - wrist1.y();
                    features[base + 2] = lm.z() - wrist1.z();
                }
            }
        }

        // Process Hand 2 (Second 63 values, if present and model supports 126 features)
        if (sortedHands.size() > 1 && inputDimension >= 126) {
            List<NormalizedLandmark> hand2 = sortedHands.get(1);
            NormalizedLandmark wrist2 = hand2.get(0);
            for (int i = 0; i < Math.min(21, hand2.size()); i++) {
                NormalizedLandmark lm = hand2.get(i);
                int base = 63 + (i * 3);
                if (base + 2 < features.length) {
                    features[base] = lm.x() - wrist2.x();
                    features[base + 1] = lm.y() - wrist2.y();
                    features[base + 2] = lm.z() - wrist2.z();
                }
            }
        }

        return features;
    }

    /**
     * Runs dual-hand landmark detection and sign classification on the provided bitmap.
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

            // Run MediaPipe dual-hand detection
            HandLandmarkerResult result = handLandmarker.detect(mpImage);

            if (result == null || result.landmarks().isEmpty()) {
                return new AIResult(null, 0f, false, 0, "AI: Scanning... (Frame #" + framesScanned + ", No hands)");
            }

            int handsDetected = result.landmarks().size();
            float[] features = extractFeatures(result.landmarks());

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

                String msg = "🎯 Recognized: " + detectedLabel + " (" + (int)(maxConfidence * 100) + "% | " + handsDetected + " hand" + (handsDetected > 1 ? "s" : "") + ")";
                return new AIResult(detectedLabel, maxConfidence, isNew, handsDetected, msg);
            } else {
                String msg = "AI: " + handsDetected + " hand" + (handsDetected > 1 ? "s" : "") + " detected (" + (int)(maxConfidence * 100) + "%)";
                return new AIResult(null, maxConfidence, false, handsDetected, msg);
            }

        } catch (Exception e) {
            Log.e(TAG, "Exception during classification", e);
            return new AIResult(null, 0f, false, 0, "AI Error: " + e.getMessage());
        }
    }

    public void close() {
        if (handLandmarker != null) {
            handLandmarker.close();
            handLandmarker = null;
        }
    }

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
