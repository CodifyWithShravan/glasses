package com.kanyaraasi;
import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.net.ConnectivityManager;
import android.net.Network;
import android.net.NetworkCapabilities;
import android.net.NetworkRequest;
import android.net.wifi.WifiNetworkSpecifier;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.text.Editable;
import android.text.TextWatcher;
import android.util.Log;
import android.view.View;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.QuickContactBadge;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.kanyaraasi.glassescontroller.R;


public class MainActivity extends AppCompatActivity{

    private static final String TAG = "MainActivity";
    private static final String STREAM_URL = "http://192.168.4.1:81/stream";

    TextView mainText, connStatus;
    Button submitButton, retryButton, toggleCameraButton, toggleAiButton, faceIdButton;
    TextView faceStatusText;
    WebView camView;
    ImageView aiCameraPreview;

    // AI Recognition UI
    LinearLayout aiStatusPanel;
    TextView aiSignLabel, aiConfidence, aiStatus;

    Context context = this;
    boolean camIsActive = false, justNowConnected = true;
    boolean aiEnabled = false;

    private SerialScanner serialScanner = new SerialScanner();
    ConnectivityManager connManager;
    ConnectivityManager.NetworkCallback networkCallback;

    // AI Pipeline Components
    private HandSignClassifier handSignClassifier;
    private MjpegStreamParser mjpegStreamParser;
    private AudioStreamer audioStreamer;

    // Face Recognition & Voice Enrollment
    private FaceRecognitionEngine faceRecognitionEngine;
    private FaceDatabase faceDatabase;
    private VoiceEnrollmentManager voiceEnrollmentManager;
    private volatile Bitmap latestCameraFrame = null;

    // Object & Obstacle Detection (Blind Assistance)
    private ObjectDetectionEngine objectDetectionEngine;
    private Button toggleObjectDetectionButton;
    private TextView objectStatusText;
    private volatile boolean objectDetectionEnabled = false;

    private Handler handler = new Handler(Looper.getMainLooper());
    private Runnable runnable = new Runnable() {
        @Override
        public void run() {
            connStatus.setText("Checking...");
            connStatus.setTextColor(Color.YELLOW);
            try{
                if(serialScanner.isConnected()){
                    submitButton.setEnabled(true);
                    retryButton.setEnabled(false);
                    connStatus.setText("CONNECTED");
                    connStatus.setTextColor(Color.GREEN);
                } else {
                    submitButton.setEnabled(false);
                    connStatus.setText("DISCONNECTED");
                    connStatus.setTextColor(Color.RED);
                    retryButton.setEnabled(true);
                    camIsActive = false;
                }
            } finally {
                handler.postDelayed(this, 2000);
            }
        }
    };

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main_text);
        checkPermissionsAndConnect();

        submitButton = findViewById(R.id.button);
        mainText = findViewById(R.id.textView);
        connStatus = findViewById(R.id.connStatus);
        retryButton = findViewById(R.id.retry_button);
        toggleCameraButton = findViewById(R.id.toggle_cam_button);
        toggleAiButton = findViewById(R.id.toggle_ai_button);
        toggleObjectDetectionButton = findViewById(R.id.toggle_object_detection_button);
        objectStatusText = findViewById(R.id.objectStatusText);
        camView = findViewById(R.id.cam_view);
        aiCameraPreview = findViewById(R.id.ai_camera_preview);

        toggleObjectDetectionButton.setOnClickListener(v -> toggleObjectDetection());

        // AI Status Panel
        aiStatusPanel = findViewById(R.id.aiStatusPanel);
        aiSignLabel = findViewById(R.id.aiSignLabel);
        aiConfidence = findViewById(R.id.aiConfidence);
        aiStatus = findViewById(R.id.aiStatus);

        connManager = (ConnectivityManager) getSystemService(Context.CONNECTIVITY_SERVICE);

        WifiNetworkSpecifier specifier = new WifiNetworkSpecifier.Builder()
                .setSsid("SampleESPNetwork")
                .build();

        NetworkRequest request = new NetworkRequest.Builder()
                .addTransportType(NetworkCapabilities.TRANSPORT_WIFI)
                .removeCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET)
                .setNetworkSpecifier(specifier)
                .build();

        networkCallback = new ConnectivityManager.NetworkCallback() {
            @Override
            public void onAvailable(@NonNull Network network){
                //connManager.bindProcessToNetwork(network);
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        submitButton.setEnabled(true);
                        retryButton.setEnabled(false);
                        connStatus.setText("CONNECTED");
                        connStatus.setTextColor(Color.GREEN);
                        if (!camIsActive) {
                            toggleCamera();
                        }
                    }
                });
            }

            @Override
            public void onLost(@NonNull Network network) {
                //connManager.bindProcessToNetwork(null);
                // Called when network disconnects
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        submitButton.setEnabled(false);
                        connStatus.setText("DISCONNECTED");
                        connStatus.setTextColor(Color.RED);
                        retryButton.setEnabled(true);
                        camIsActive = false;

                        // Stop AI if running when network drops
                        if (aiEnabled) {
                            toggleAI();
                        }
                    }
                });
            }
        };

        connManager.registerDefaultNetworkCallback(networkCallback);

        connStatus.addTextChangedListener(new TextWatcher() {
            @Override
            public void afterTextChanged(Editable s) {

            }

            @Override
            public void beforeTextChanged(CharSequence s, int start, int count, int after) {

            }

            @Override
            public void onTextChanged(CharSequence s, int start, int before, int count) {
                /*new Handler(Looper.getMainLooper()).postDelayed(() -> {
                    camView.reload();
                }, 500);*/
            }
        });

        submitButton.setOnClickListener(new View.OnClickListener() {

            @Override
            public void onClick(View v) {
                serialScanner.sendTcpCommand("STATUS", mainText);
            }
        });

        retryButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                serialScanner.connectToEsp32(context);
            }
        });

        toggleCameraButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (!aiEnabled) {
                    toggleCamera();
                }
            }
        });

        toggleAiButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                toggleAI();
            }
        });

        camView.setOnLongClickListener(new View.OnLongClickListener() {
            @Override
            public boolean onLongClick(View v) {
                System.out.println("Reloading URL...");
                camView.reload();
                camView.loadUrl(STREAM_URL);
                return true;
            }
        });

        WebSettings camWebSettings = camView.getSettings();
        camWebSettings.setJavaScriptEnabled(true);
        camWebSettings.setLoadWithOverviewMode(true);
        camWebSettings.setUseWideViewPort(true);

        // Face ID UI & components
        faceIdButton = findViewById(R.id.face_id_button);
        faceStatusText = findViewById(R.id.faceStatusText);
        faceDatabase = new FaceDatabase(context);
        voiceEnrollmentManager = new VoiceEnrollmentManager(context);
        updateFaceStatusDisplay();

        faceIdButton.setOnClickListener(v -> onGlassesButtonSingleTap());
        faceIdButton.setOnLongClickListener(v -> {
            onGlassesButtonLongPress();
            return true;
        });

        // Hide WebView permanently, we now exclusively use MjpegStreamParser
        camView.setVisibility(View.GONE);
        aiCameraPreview.setVisibility(View.VISIBLE);

        // Don't auto-start camera here since the network callback will handle it
        camIsActive = false;

        new Handler(Looper.getMainLooper()).postDelayed(() -> {
            runnable.run();
        }, 2000);

        // Initialize AI components on a background thread
        initializeAI();

        // Start listening for physical button events from the Smart Glasses (UDP Port 8888)
        startUdpButtonListener();
    }

    private void updateFaceStatusDisplay() {
        if (faceDatabase != null && faceStatusText != null) {
            int count = faceDatabase.getEnrolledCount();
            faceStatusText.setText("Faces Enrolled: " + count + " | Tap: Identify | Hold: Register");
        }
    }

    @Override
    protected void onDestroy(){
        super.onDestroy();
        handler.removeCallbacks(runnable);
        cleanupAI();
    }

    /**
     * Initialize AI pipeline components (classifier + audio streamer + face engine).
     * Model loading happens on a background thread to avoid blocking the UI.
     */
    private void initializeAI() {
        new Thread(() -> {
            try {
                Log.d(TAG, "Initializing AI & Face components...");

                // Initialize the hand sign classifier
                handSignClassifier = new HandSignClassifier(context);

                // Initialize Face Recognition Engine (ML Kit + MobileFaceNet)
                faceRecognitionEngine = new FaceRecognitionEngine(context);

                // Initialize Object Detection Engine (ML Kit Object Detection)
                objectDetectionEngine = new ObjectDetectionEngine(context);

                // Initialize the audio streamer (TTS engine)
                audioStreamer = new AudioStreamer(context);
                boolean ttsReady = audioStreamer.initialize();

                runOnUiThread(() -> {
                    if (ttsReady) {
                        aiStatus.setText("AI: Ready (tap button to enable)");
                        toggleAiButton.setEnabled(true);
                        Log.d(TAG, "AI, Face & Object components initialized successfully");
                    } else {
                        aiStatus.setText("AI: TTS failed to initialize");
                        toggleAiButton.setEnabled(true);
                        Log.w(TAG, "TTS initialization failed, audio output disabled");
                    }
                    updateFaceStatusDisplay();
                });
            } catch (Exception e) {
                Log.e(TAG, "Failed to initialize AI components", e);
                runOnUiThread(() -> {
                    aiStatus.setText("AI: Init failed - " + e.getMessage());
                    toggleAiButton.setEnabled(false);
                });
            }
        }).start();
    }

    /**
     * Toggle AI recognition on/off.
     * When enabled, starts the MJPEG stream parser to extract frames and
     * runs them through the hand sign classifier.
     */
    private void toggleAI() {
        if (aiEnabled) {
            // --- DISABLE AI ---
            aiEnabled = false;
            toggleAiButton.setText("\uD83E\uDD16 Enable AI Recognition");
            aiStatusPanel.setVisibility(View.GONE);

            // Stop the AI-enabled stream parser
            if (mjpegStreamParser != null) {
                mjpegStreamParser.stop();
                mjpegStreamParser = null;
            }
            camIsActive = false;
            
            // Restart normal camera feed
            toggleCamera();

            aiSignLabel.setText("Sign: ---");
            aiConfidence.setText("Confidence: ---%");
            aiStatus.setText("AI: Disabled");
            Log.d(TAG, "AI recognition disabled");

        } else {
            // --- ENABLE AI ---
            if (handSignClassifier == null) {
                aiStatus.setText("AI: Classifier not ready");
                return;
            }

            aiEnabled = true;
            toggleAiButton.setText("\uD83D\uDED1 Disable AI Recognition");
            aiStatusPanel.setVisibility(View.VISIBLE);
            aiStatus.setText("AI: Starting stream parser...");

            // Stop normal camera feed first
            if (mjpegStreamParser != null) {
                mjpegStreamParser.stop();
                mjpegStreamParser = null;
            }
            camIsActive = false;

            // Create and start the MJPEG stream parser, passing the specific ESP32 network
            mjpegStreamParser = new MjpegStreamParser(STREAM_URL, serialScanner.getEspNetwork());
            
            // Set listener to update the ImageView on every frame
            mjpegStreamParser.setDisplayListener(new MjpegStreamParser.OnFrameDisplayListener() {
                @Override
                public void onFrameForDisplay(Bitmap frame) {
                    latestCameraFrame = frame;
                    runOnUiThread(() -> {
                        if (aiEnabled && frame != null) {
                            aiCameraPreview.setImageBitmap(frame);
                        }
                    });
                }
            });

            // Set listener to run AI processing on every Nth frame
            mjpegStreamParser.setProcessListener(new MjpegStreamParser.OnFrameProcessListener() {
                @Override
                public void onFrameForProcessing(Bitmap frame) {
                    processFrameForAI(frame);
                }
            });
            mjpegStreamParser.setProcessEveryN(3); // Decouple AI inference from camera display for smooth 30 FPS
            
            mjpegStreamParser.start();

            aiStatus.setText("AI: Scanning for hand signs...");
            Log.d(TAG, "AI recognition enabled");
        }
    }

    // Dedicated asynchronous worker for AI inference to prevent video stream lag
    private final java.util.concurrent.ExecutorService aiExecutor = java.util.concurrent.Executors.newSingleThreadExecutor();
    private final java.util.concurrent.atomic.AtomicBoolean isAiProcessing = new java.util.concurrent.atomic.AtomicBoolean(false);

    /**
     * Process a single video frame through the AI pipeline.
     * Runs asynchronously on a dedicated background worker thread with frame-skipping
     * so that the camera stream parser NEVER blocks and has zero latency.
     */
    private void processFrameForAI(Bitmap frame) {
        if ((!aiEnabled && !objectDetectionEnabled) || frame == null || frame.isRecycled()) {
            if (frame != null && !frame.isRecycled()) {
                frame.recycle();
            }
            return;
        }

        // If AI is already busy with a previous frame, DROP this frame immediately.
        // This ensures the camera stream continues at full FPS with ZERO lag.
        if (!isAiProcessing.compareAndSet(false, true)) {
            frame.recycle();
            return;
        }

        aiExecutor.execute(() -> {
            try {
                long timestampMs = System.currentTimeMillis();

                // 1. Hand Sign Recognition (if enabled)
                if (aiEnabled && handSignClassifier != null) {
                    HandSignClassifier.AIResult result =
                            handSignClassifier.classify(frame, timestampMs);

                    if (result != null) {
                        runOnUiThread(() -> {
                            if (result.label != null) {
                                aiSignLabel.setText("Sign: " + result.label);
                                aiConfidence.setText(String.format(java.util.Locale.US, "Confidence: %.0f%%", result.confidence * 100));
                            } else {
                                aiSignLabel.setText("Sign: ---");
                                aiConfidence.setText("Confidence: ---%");
                            }
                            aiStatus.setText(result.statusMessage);
                        });

                        if (result.isNew && result.label != null && audioStreamer != null && audioStreamer.isReady()) {
                            Log.d(TAG, "New sign detected: " + result.label +
                                    " (confidence: " + (result.confidence * 100) + "%)");
                            audioStreamer.speakToGlasses(result.label, serialScanner.getEspNetwork());
                        }
                    }
                }

                // 2. Continuous Object & Obstacle Detection (if enabled)
                if (objectDetectionEnabled && objectDetectionEngine != null && objectDetectionEngine.isReady()) {
                    objectDetectionEngine.processFrame(frame, new ObjectDetectionEngine.OnObjectsDetectedListener() {
                        @Override
                        public void onObjectsDetected(java.util.List<ObjectDetectionEngine.DetectedObstacle> obstacles, String spokenAlert, int frameCount) {
                            if (!obstacles.isEmpty()) {
                                ObjectDetectionEngine.DetectedObstacle primary = obstacles.get(0);
                                runOnUiThread(() -> objectStatusText.setText("👁️ " + primary.toSummaryString() + " [#" + frameCount + "]"));
                            }
                            if (spokenAlert != null && audioStreamer != null && audioStreamer.isReady()) {
                                Log.d(TAG, "Obstacle alert: " + spokenAlert);
                                audioStreamer.speakToGlasses(spokenAlert, serialScanner.getEspNetwork());
                            }
                        }

                        @Override
                        public void onNoObjectsDetected(int frameCount) {
                            runOnUiThread(() -> objectStatusText.setText("👁️ Obstacle Alerts: ACTIVE (Scanning... #" + frameCount + ", Path clear)"));
                        }

                        @Override
                        public void onError(String error) {
                            Log.w(TAG, "Obstacle detection error: " + error);
                            runOnUiThread(() -> objectStatusText.setText("⚠️ Obstacle: " + error));
                        }
                    });
                }

            } catch (Exception e) {
                Log.e(TAG, "Error processing frame for AI", e);
                runOnUiThread(() -> aiStatus.setText("AI Error: " + e.getMessage()));
            } finally {
                // Recycle bitmap and release lock for the next frame
                if (frame != null && !frame.isRecycled()) {
                    frame.recycle();
                }
                isAiProcessing.set(false);
            }
        });
    }

    /**
     * Single-tap action (or ESP32 hardware button single-tap):
     * Identifies the person standing in front of the camera and announces their name.
     */
    public void onGlassesButtonSingleTap() {
        if (latestCameraFrame == null || latestCameraFrame.isRecycled()) {
            runOnUiThread(() -> {
                faceStatusText.setText("👤 Waiting for camera stream...");
                android.widget.Toast.makeText(context, "Camera stream not ready", android.widget.Toast.LENGTH_SHORT).show();
            });
            if (audioStreamer != null && audioStreamer.isReady()) {
                audioStreamer.speakToGlasses("Camera stream not ready", serialScanner.getEspNetwork());
            }
            return;
        }

        if (faceRecognitionEngine == null || !faceRecognitionEngine.isReady()) {
            String err = faceRecognitionEngine != null ? faceRecognitionEngine.getInitError() : "not initialized";
            runOnUiThread(() -> faceStatusText.setText("👤 Face Engine: " + err));
            return;
        }

        runOnUiThread(() -> faceStatusText.setText("👤 Scanning face..."));

        Bitmap frameCopy = latestCameraFrame.copy(Bitmap.Config.ARGB_8888, false);
        faceRecognitionEngine.processFrame(frameCopy, new FaceRecognitionEngine.OnFaceProcessedListener() {
            @Override
            public void onFaceProcessed(FaceRecognitionEngine.FaceData faceData) {
                FaceDatabase.MatchResult match = faceDatabase.findBestMatch(
                        faceData.embedding,
                        FaceDatabase.DEFAULT_SIMILARITY_THRESHOLD
                );

                runOnUiThread(() -> {
                    if (match != null && match.isMatch) {
                        int percent = (int) (match.similarity * 100);
                        faceStatusText.setText("🎯 Recognized: " + match.name + " (" + percent + "%)");
                        if (audioStreamer != null && audioStreamer.isReady()) {
                            audioStreamer.speakToGlasses(match.name + " is in front of you", serialScanner.getEspNetwork());
                        }
                    } else {
                        faceStatusText.setText("👤 Unknown person (Hold button to register)");
                        if (audioStreamer != null && audioStreamer.isReady()) {
                            audioStreamer.speakToGlasses("Unknown person. Hold the button to register them.", serialScanner.getEspNetwork());
                        }
                    }
                });
            }

            @Override
            public void onNoFaceDetected() {
                runOnUiThread(() -> {
                    faceStatusText.setText("👤 No face detected");
                    if (audioStreamer != null && audioStreamer.isReady()) {
                        audioStreamer.speakToGlasses("No face detected", serialScanner.getEspNetwork());
                    }
                });
            }

            @Override
            public void onError(String error) {
                runOnUiThread(() -> faceStatusText.setText("⚠️ " + error));
            }
        });
    }

    /**
     * Long-press action (or ESP32 hardware button long-press):
     * Captures the face in front of the camera, prompts for the person's name via voice,
     * and enrolls them into the face database.
     */
    public void onGlassesButtonLongPress() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.RECORD_AUDIO}, 102);
            return;
        }

        if (latestCameraFrame == null || latestCameraFrame.isRecycled()) {
            runOnUiThread(() -> faceStatusText.setText("👤 Camera stream not ready"));
            return;
        }

        if (faceRecognitionEngine == null || !faceRecognitionEngine.isReady()) {
            String err = faceRecognitionEngine != null ? faceRecognitionEngine.getInitError() : "not initialized";
            runOnUiThread(() -> faceStatusText.setText("👤 Face Engine: " + err));
            return;
        }

        runOnUiThread(() -> faceStatusText.setText("👤 Capturing face for enrollment..."));

        Bitmap frameCopy = latestCameraFrame.copy(Bitmap.Config.ARGB_8888, false);
        faceRecognitionEngine.processFrame(frameCopy, new FaceRecognitionEngine.OnFaceProcessedListener() {
            @Override
            public void onFaceProcessed(FaceRecognitionEngine.FaceData faceData) {
                runOnUiThread(() -> faceStatusText.setText("🎤 Asking name..."));

                // 1. Speak prompt via TTS
                if (audioStreamer != null && audioStreamer.isReady()) {
                    audioStreamer.speakToGlasses("What is this person's name?", serialScanner.getEspNetwork());
                }

                // 2. Wait 1.8 seconds for TTS speech to finish before opening microphone
                handler.postDelayed(() -> {
                    voiceEnrollmentManager.startListening(new VoiceEnrollmentManager.VoiceEnrollmentCallback() {
                        @Override
                        public void onListeningStarted() {
                            runOnUiThread(() -> faceStatusText.setText("🎤 Listening... Speak the name now"));
                        }

                        @Override
                        public void onNameReceived(String spokenName) {
                            // Enroll into database
                            faceDatabase.enrollFace(spokenName, faceData.embedding);

                            runOnUiThread(() -> {
                                updateFaceStatusDisplay();
                                faceStatusText.setText("✅ Saved: " + spokenName + " (" + faceDatabase.getEnrolledCount() + " enrolled)");
                                android.widget.Toast.makeText(context, "Saved " + spokenName, android.widget.Toast.LENGTH_SHORT).show();
                            });

                            if (audioStreamer != null && audioStreamer.isReady()) {
                                audioStreamer.speakToGlasses("Saved " + spokenName + " successfully", serialScanner.getEspNetwork());
                            }
                        }

                        @Override
                        public void onError(String errorMessage) {
                            runOnUiThread(() -> faceStatusText.setText("⚠️ " + errorMessage));
                            if (audioStreamer != null && audioStreamer.isReady()) {
                                audioStreamer.speakToGlasses("Could not hear name. Try again.", serialScanner.getEspNetwork());
                            }
                        }
                    });
                }, 1800);
            }

            @Override
            public void onNoFaceDetected() {
                runOnUiThread(() -> {
                    faceStatusText.setText("👤 Look directly at camera to register");
                    if (audioStreamer != null && audioStreamer.isReady()) {
                        audioStreamer.speakToGlasses("Please look directly at the camera to register", serialScanner.getEspNetwork());
                    }
                });
            }

            @Override
            public void onError(String error) {
                runOnUiThread(() -> faceStatusText.setText("⚠️ " + error));
            }
        });
    }

    /**
     * Toggles continuous real-time object & obstacle detection for blind assistance.
     */
    private void toggleObjectDetection() {
        objectDetectionEnabled = !objectDetectionEnabled;
        if (objectDetectionEnabled) {
            toggleObjectDetectionButton.setText("🛑 Stop Obstacle Alerts");
            objectStatusText.setText("👁️ Obstacle Alerts: ACTIVE (Scanning path...)");
            if (audioStreamer != null && audioStreamer.isReady()) {
                audioStreamer.speakToGlasses("Obstacle detection enabled", serialScanner.getEspNetwork());
            }
            ensureStreamRunning();
        } else {
            toggleObjectDetectionButton.setText("👁️ Object & Obstacle Detection");
            objectStatusText.setText("Obstacle Alerts: OFF (Tap button or Double-tap glasses)");
            if (audioStreamer != null && audioStreamer.isReady()) {
                audioStreamer.speakToGlasses("Obstacle detection disabled", serialScanner.getEspNetwork());
            }
        }
    }

    /**
     * Double-tap action (or ESP32 hardware button double-tap):
     * Scans the entire scene in front of the blind user and speaks a complete summary of obstacles.
     */
    public void onGlassesButtonDoubleTap() {
        if (latestCameraFrame == null || latestCameraFrame.isRecycled()) {
            runOnUiThread(() -> objectStatusText.setText("👁️ Camera stream not ready"));
            if (audioStreamer != null && audioStreamer.isReady()) {
                audioStreamer.speakToGlasses("Camera stream not ready", serialScanner.getEspNetwork());
            }
            return;
        }

        if (objectDetectionEngine == null || !objectDetectionEngine.isReady()) {
            String err = objectDetectionEngine != null ? objectDetectionEngine.getInitError() : "not initialized";
            runOnUiThread(() -> objectStatusText.setText("👁️ Object Engine: " + err));
            return;
        }

        runOnUiThread(() -> objectStatusText.setText("👁️ Scanning scene for obstacles..."));

        Bitmap frameCopy = latestCameraFrame.copy(Bitmap.Config.ARGB_8888, false);
        objectDetectionEngine.processFrame(frameCopy, new ObjectDetectionEngine.OnObjectsDetectedListener() {
            @Override
            public void onObjectsDetected(java.util.List<ObjectDetectionEngine.DetectedObstacle> obstacles, String spokenAlert, int frameCount) {
                String summary = objectDetectionEngine.generateSceneSummary(obstacles);
                runOnUiThread(() -> objectStatusText.setText("👁️ " + summary));
                if (audioStreamer != null && audioStreamer.isReady()) {
                    audioStreamer.speakToGlasses(summary, serialScanner.getEspNetwork());
                }
            }

            @Override
            public void onNoObjectsDetected(int frameCount) {
                runOnUiThread(() -> objectStatusText.setText("👁️ Path is clear (No obstacles detected)"));
                if (audioStreamer != null && audioStreamer.isReady()) {
                    audioStreamer.speakToGlasses("Path is clear. No obstacles in front of you.", serialScanner.getEspNetwork());
                }
            }

            @Override
            public void onError(String error) {
                runOnUiThread(() -> objectStatusText.setText("⚠️ " + error));
            }
        });
    }

    /**
     * Clean up AI resources when the activity is destroyed.
     */
    private void cleanupAI() {
        aiEnabled = false;
        objectDetectionEnabled = false;

        if (mjpegStreamParser != null) {
            mjpegStreamParser.stop();
            mjpegStreamParser = null;
        }
        if (handSignClassifier != null) {
            handSignClassifier.close();
            handSignClassifier = null;
        }
        if (faceRecognitionEngine != null) {
            faceRecognitionEngine.close();
            faceRecognitionEngine = null;
        }
        if (objectDetectionEngine != null) {
            objectDetectionEngine.close();
            objectDetectionEngine = null;
        }
        if (voiceEnrollmentManager != null) {
            voiceEnrollmentManager.stopListening();
            voiceEnrollmentManager = null;
        }
        if (audioStreamer != null) {
            audioStreamer.shutdown();
            audioStreamer = null;
        }
    }

    private void checkPermissionsAndConnect() {
        java.util.List<String> neededPermissions = new java.util.ArrayList<>();
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION)
                != PackageManager.PERMISSION_GRANTED) {
            neededPermissions.add(Manifest.permission.ACCESS_FINE_LOCATION);
        }
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
                != PackageManager.PERMISSION_GRANTED) {
            neededPermissions.add(Manifest.permission.RECORD_AUDIO);
        }

        if (!neededPermissions.isEmpty()) {
            ActivityCompat.requestPermissions(this,
                    neededPermissions.toArray(new String[0]), 101);
        } else {
            serialScanner.connectToEsp32(this);
        }
    }

    private synchronized void ensureStreamRunning() {
        if (mjpegStreamParser != null && mjpegStreamParser.isRunning()) {
            return;
        }

        if (serialScanner.getEspNetwork() == null) {
            Log.w(TAG, "Cannot start camera: ESP32 network not ready");
            return;
        }

        mjpegStreamParser = new MjpegStreamParser(STREAM_URL, serialScanner.getEspNetwork());
        mjpegStreamParser.setDisplayListener(frame -> {
            latestCameraFrame = frame;
            runOnUiThread(() -> {
                if (frame != null && !frame.isRecycled()) {
                    aiCameraPreview.setImageBitmap(frame);
                }
            });
        });
        mjpegStreamParser.setProcessListener(this::processFrameForAI);
        mjpegStreamParser.setProcessEveryN(3); // Decouple AI inference from camera display for smooth 30 FPS
        mjpegStreamParser.start();
        camIsActive = true;
    }

    private void toggleCamera(){
        if(camIsActive){
            // Stop stream
            if (mjpegStreamParser != null) {
                mjpegStreamParser.stop();
                mjpegStreamParser = null;
            }
            aiCameraPreview.setImageBitmap(null);
            camIsActive = false;
        } else {
            ensureStreamRunning();
        }
    }

    /**
     * Listens in background for instant UDP events broadcasted by the ESP32 physical button (Port 8888).
     *   - "TAP"        -> onGlassesButtonSingleTap() (Identify Face)
     *   - "DOUBLE_TAP" -> onGlassesButtonDoubleTap() (Scan Scene & Obstacles)
     *   - "HOLD"       -> onGlassesButtonLongPress() (Voice Register Face)
     */
    private java.net.DatagramSocket udpButtonSocket = null;

    private void startUdpButtonListener() {
        new Thread(() -> {
            try {
                udpButtonSocket = new java.net.DatagramSocket(8888);
                byte[] buffer = new byte[64];
                Log.d(TAG, "UDP Hardware Button listener started on port 8888 (Single/Double/Hold)");

                while (!isFinishing() && udpButtonSocket != null && !udpButtonSocket.isClosed()) {
                    java.net.DatagramPacket packet = new java.net.DatagramPacket(buffer, buffer.length);
                    udpButtonSocket.receive(packet);
                    String message = new String(packet.getData(), 0, packet.getLength()).trim();
                    Log.d(TAG, "Received hardware button UDP event: " + message);

                    if ("TAP".equalsIgnoreCase(message)) {
                        runOnUiThread(this::onGlassesButtonSingleTap);
                    } else if ("DOUBLE_TAP".equalsIgnoreCase(message)) {
                        runOnUiThread(this::onGlassesButtonDoubleTap);
                    } else if ("HOLD".equalsIgnoreCase(message)) {
                        runOnUiThread(this::onGlassesButtonLongPress);
                    }
                }
            } catch (Exception e) {
                Log.w(TAG, "UDP Button Listener ended: " + e.getMessage());
            } finally {
                if (udpButtonSocket != null && !udpButtonSocket.isClosed()) {
                    udpButtonSocket.close();
                    udpButtonSocket = null;
                }
            }
        }).start();
    }
}
