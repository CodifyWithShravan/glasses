package com.kanyaraasi;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.util.Log;

import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * ObjectDetectionEngine
 * 
 * 100% Offline Real-Time Object & Obstacle Detection for Blind Assistance.
 * Powered by SSD-MobileNet TFLite trained on 80 COCO everyday objects
 * (chairs, tables, bottles, doors, people, cups, laptops, backpacks, etc.).
 * 
 * Computes spatial direction (Left/Center/Right) and proximity (Very Close/Near/Far)
 * and generates clean, non-intrusive voice alerts for visually impaired users.
 */
public class ObjectDetectionEngine {
    private static final String TAG = "ObjectDetectionEngine";
    private static final String MODEL_FILE = "ssd_mobilenet.tflite";
    private static final String LABELS_FILE = "coco_labels.txt";

    private static final int INPUT_SIZE = 300;
    private static final int NUM_DETECTIONS = 10;
    private static final float MIN_CONFIDENCE = 0.45f;
    private static final long COOLDOWN_MS = 3500; // 3.5s cooldown between repeat alerts

    private Interpreter tfliteInterpreter;
    private final List<String> labels = new ArrayList<>();
    private boolean isReady = false;
    private String initError = null;
    private int framesScanned = 0;

    // Debouncing state
    private String lastSpokenAlert = null;
    private long lastAlertTimestamp = 0;

    public interface OnObjectsDetectedListener {
        void onObjectsDetected(List<DetectedObstacle> obstacles, String spokenAlert, int frameCount);
        void onNoObjectsDetected(int frameCount);
        void onError(String error);
    }

    public static class DetectedObstacle {
        public final String name;
        public final String direction;   // "on your left", "directly in front", "on your right"
        public final String proximity;   // "very close", "near", "far"
        public final float areaPercent;  // percentage of frame (0.0 - 100.0)
        public final RectF boundingBox;
        public final float confidence;

        public DetectedObstacle(String name, String direction, String proximity, float areaPercent, RectF boundingBox, float confidence) {
            this.name = name;
            this.direction = direction;
            this.proximity = proximity;
            this.areaPercent = areaPercent;
            this.boundingBox = boundingBox;
            this.confidence = confidence;
        }

        public String toSummaryString() {
            if ("very close".equals(proximity)) {
                return "⚠️ " + name + " " + direction + " (VERY CLOSE)";
            }
            return name + " " + direction + " (" + proximity + ")";
        }
    }

    public ObjectDetectionEngine(Context context) {
        initialize(context);
    }

    private void initialize(Context context) {
        try {
            // 1. Load labels
            loadLabels(context, LABELS_FILE);

            // 2. Load TFLite Model
            ByteBuffer modelBuffer = loadModelFile(context, MODEL_FILE);
            Interpreter.Options options = new Interpreter.Options();
            options.setNumThreads(4);
            tfliteInterpreter = new Interpreter(modelBuffer, options);

            isReady = true;
            initError = null;
            Log.d(TAG, "ObjectDetectionEngine initialized successfully with " + labels.size() + " labels");
        } catch (Exception e) {
            Log.e(TAG, "Failed to initialize ObjectDetectionEngine", e);
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
     * Synchronously processes a camera frame: runs SSD MobileNet, calculates spatial directions,
     * proximity levels, and returns structured obstacle list & debounced speech alert.
     */
    public synchronized void processFrame(Bitmap frame, OnObjectsDetectedListener listener) {
        if (!isReady || tfliteInterpreter == null) {
            if (listener != null) listener.onError("Engine not ready: " + initError);
            return;
        }

        if (frame == null || frame.isRecycled()) {
            if (listener != null) listener.onError("Invalid camera frame");
            return;
        }

        framesScanned++;
        final int currentFrame = framesScanned;

        try {
            // 1. Preprocess: Resize to 300x300 and convert to uint8 ByteBuffer
            Bitmap resized = Bitmap.createScaledBitmap(frame, INPUT_SIZE, INPUT_SIZE, true);
            ByteBuffer inputBuffer = convertBitmapToByteBuffer(resized);
            if (resized != frame && !resized.isRecycled()) {
                resized.recycle();
            }

            // 2. Prepare Output Tensors
            float[][][] outputLocations = new float[1][NUM_DETECTIONS][4];
            float[][] outputClasses = new float[1][NUM_DETECTIONS];
            float[][] outputScores = new float[1][NUM_DETECTIONS];
            float[] numDetections = new float[1];

            Object[] inputArray = {inputBuffer};
            Map<Integer, Object> outputMap = new HashMap<>();
            outputMap.put(0, outputLocations);
            outputMap.put(1, outputClasses);
            outputMap.put(2, outputScores);
            outputMap.put(3, numDetections);

            // 3. Run Inference
            tfliteInterpreter.runForMultipleInputsOutputs(inputArray, outputMap);

            int count = Math.min((int) numDetections[0], NUM_DETECTIONS);
            List<DetectedObstacle> obstacles = new ArrayList<>();

            for (int i = 0; i < count; i++) {
                float score = outputScores[0][i];
                if (score < MIN_CONFIDENCE) {
                    continue;
                }

                int classIndex = (int) outputClasses[0][i];
                String labelName = (classIndex >= 0 && classIndex < labels.size()) ? labels.get(classIndex) : "Obstacle";
                if ("???".equals(labelName) || labelName.trim().isEmpty()) {
                    labelName = "Obstacle";
                }
                labelName = cleanLabelName(labelName);

                // Coordinates: [top, left, bottom, right] normalized to [0, 1]
                float top = outputLocations[0][i][0];
                float left = outputLocations[0][i][1];
                float bottom = outputLocations[0][i][2];
                float right = outputLocations[0][i][3];

                float width = Math.max(0.01f, right - left);
                float height = Math.max(0.01f, bottom - top);
                float areaPercent = (width * height) * 100f;

                // Determine Spatial Direction (Left / Center / Right)
                float centerX = (left + right) / 2.0f;
                String direction;
                if (centerX < 0.35f) {
                    direction = "on your left";
                } else if (centerX > 0.65f) {
                    direction = "on your right";
                } else {
                    direction = "directly in front";
                }

                // Determine Proximity / Urgency
                String proximity;
                if (areaPercent >= 28.0f) {
                    proximity = "very close";
                } else if (areaPercent >= 8.0f) {
                    proximity = "near";
                } else {
                    proximity = "far";
                }

                RectF box = new RectF(left, top, right, bottom);
                obstacles.add(new DetectedObstacle(labelName, direction, proximity, areaPercent, box, score));
            }

            if (obstacles.isEmpty()) {
                if (listener != null) listener.onNoObjectsDetected(currentFrame);
                return;
            }

            // Sort by largest / closest obstacle first
            Collections.sort(obstacles, (o1, o2) -> Float.compare(o2.areaPercent, o1.areaPercent));

            // Generate debounced spoken alert
            String spokenAlert = generateSpokenAlert(obstacles);

            if (listener != null) {
                listener.onObjectsDetected(obstacles, spokenAlert, currentFrame);
            }

        } catch (Exception e) {
            Log.e(TAG, "Error in processFrame", e);
            if (listener != null) listener.onError("Detection exception: " + e.getMessage());
        }
    }

    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(1 * INPUT_SIZE * INPUT_SIZE * 3);
        byteBuffer.order(ByteOrder.nativeOrder());

        int[] intValues = new int[INPUT_SIZE * INPUT_SIZE];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        int pixelIndex = 0;
        for (int i = 0; i < INPUT_SIZE; i++) {
            for (int j = 0; j < INPUT_SIZE; j++) {
                int val = intValues[pixelIndex++];
                // SSD-MobileNet uint8 input: 0..255 (RGB)
                byteBuffer.put((byte) ((val >> 16) & 0xFF));
                byteBuffer.put((byte) ((val >> 8) & 0xFF));
                byteBuffer.put((byte) (val & 0xFF));
            }
        }
        return byteBuffer;
    }

    private String cleanLabelName(String label) {
        if (label == null) return "Obstacle";
        switch (label.toLowerCase()) {
            case "dining table":
                return "Table";
            case "cell phone":
                return "Phone";
            case "potted plant":
                return "Plant";
            case "traffic light":
                return "Signal";
            case "stop sign":
                return "Stop sign";
            default:
                // Capitalize first letter
                if (!label.isEmpty()) {
                    return Character.toUpperCase(label.charAt(0)) + label.substring(1);
                }
                return label;
        }
    }

    public synchronized String generateSpokenAlert(List<DetectedObstacle> obstacles) {
        if (obstacles == null || obstacles.isEmpty()) {
            return null;
        }

        DetectedObstacle primary = obstacles.get(0);
        String candidateAlert;

        if ("very close".equals(primary.proximity)) {
            candidateAlert = "Warning: " + primary.name + " " + primary.direction + ", very close";
        } else {
            candidateAlert = primary.name + " " + primary.direction;
        }

        long now = System.currentTimeMillis();

        if (candidateAlert.equalsIgnoreCase(lastSpokenAlert)) {
            if (now - lastAlertTimestamp < COOLDOWN_MS) {
                return null; // Suppress repeat alert
            }
        }

        lastSpokenAlert = candidateAlert;
        lastAlertTimestamp = now;
        return candidateAlert;
    }

    public String generateSceneSummary(List<DetectedObstacle> obstacles) {
        if (obstacles == null || obstacles.isEmpty()) {
            return "Path is clear. No obstacles detected in front of you.";
        }

        if (obstacles.size() == 1) {
            DetectedObstacle o = obstacles.get(0);
            return o.name + " " + o.direction + ", " + o.proximity;
        }

        DetectedObstacle first = obstacles.get(0);
        DetectedObstacle second = obstacles.get(1);
        return first.name + " " + first.direction + ", and " + second.name + " " + second.direction;
    }

    private void loadLabels(Context context, String filename) throws IOException {
        labels.clear();
        try (InputStream is = context.getAssets().open(filename);
             BufferedReader reader = new BufferedReader(new InputStreamReader(is))) {
            String line;
            while ((line = reader.readLine()) != null) {
                labels.add(line.trim());
            }
        }
    }

    private ByteBuffer loadModelFile(Context context, String modelPath) throws IOException {
        try (InputStream is = context.getAssets().open(modelPath)) {
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
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
        if (tfliteInterpreter != null) {
            tfliteInterpreter.close();
            tfliteInterpreter = null;
        }
    }
}
