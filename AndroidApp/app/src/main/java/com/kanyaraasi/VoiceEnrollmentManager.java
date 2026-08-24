package com.kanyaraasi;

import android.content.Context;
import android.content.Intent;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.speech.RecognitionListener;
import android.speech.RecognizerIntent;
import android.speech.SpeechRecognizer;
import android.util.Log;

import java.util.ArrayList;
import java.util.Locale;

/**
 * VoiceEnrollmentManager
 *
 * Uses Android SpeechRecognizer to capture spoken names from the phone's
 * microphone for hands-free face registration.
 */
public class VoiceEnrollmentManager {
    private static final String TAG = "VoiceEnrollmentManager";

    private final Context context;
    private SpeechRecognizer speechRecognizer;
    private boolean isListening = false;
    private final Handler mainHandler = new Handler(Looper.getMainLooper());

    public interface VoiceEnrollmentCallback {
        void onListeningStarted();
        void onNameReceived(String spokenName);
        void onError(String errorMessage);
    }

    public VoiceEnrollmentManager(Context context) {
        this.context = context;
    }

    /**
     * Starts listening to the microphone for a spoken person's name.
     */
    public void startListening(VoiceEnrollmentCallback callback) {
        mainHandler.post(() -> {
            if (isListening) {
                stopListening();
            }

            if (!SpeechRecognizer.isRecognitionAvailable(context)) {
                if (callback != null) {
                    callback.onError("Speech recognition is not available on this device");
                }
                return;
            }

            try {
                speechRecognizer = SpeechRecognizer.createSpeechRecognizer(context);
                speechRecognizer.setRecognitionListener(new RecognitionListener() {
                    @Override
                    public void onReadyForSpeech(Bundle params) {
                        isListening = true;
                        Log.d(TAG, "Microphone open and ready for speech...");
                        if (callback != null) callback.onListeningStarted();
                    }

                    @Override
                    public void onBeginningOfSpeech() {
                        Log.d(TAG, "User started speaking...");
                    }

                    @Override
                    public void onRmsChanged(float rmsdB) {}

                    @Override
                    public void onBufferReceived(byte[] buffer) {}

                    @Override
                    public void onEndOfSpeech() {
                        Log.d(TAG, "User finished speaking, processing audio...");
                        isListening = false;
                    }

                    @Override
                    public void onError(int error) {
                        isListening = false;
                        String errorMsg = getErrorMessage(error);
                        Log.e(TAG, "Speech recognition error: " + errorMsg + " (code: " + error + ")");
                        if (callback != null) callback.onError(errorMsg);
                        cleanup();
                    }

                    @Override
                    public void onResults(Bundle results) {
                        isListening = false;
                        ArrayList<String> matches = results.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION);
                        if (matches != null && !matches.isEmpty()) {
                            String rawName = matches.get(0);
                            String cleanName = sanitizeName(rawName);
                            Log.d(TAG, "Recognized spoken name: " + cleanName + " (raw: " + rawName + ")");
                            if (callback != null) callback.onNameReceived(cleanName);
                        } else {
                            if (callback != null) callback.onError("No name could be heard. Please try again.");
                        }
                        cleanup();
                    }

                    @Override
                    public void onPartialResults(Bundle partialResults) {}

                    @Override
                    public void onEvent(int eventType, Bundle params) {}
                });

                Intent intent = new Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH);
                intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM);
                intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault());
                intent.putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1);
                intent.putExtra(RecognizerIntent.EXTRA_PROMPT, "Say the person's name...");

                speechRecognizer.startListening(intent);

            } catch (Exception e) {
                Log.e(TAG, "Failed to start speech recognizer", e);
                isListening = false;
                if (callback != null) callback.onError("Microphone error: " + e.getMessage());
                cleanup();
            }
        });
    }

    public void stopListening() {
        mainHandler.post(() -> {
            if (speechRecognizer != null) {
                try {
                    speechRecognizer.stopListening();
                    speechRecognizer.cancel();
                } catch (Exception ignored) {}
            }
            isListening = false;
            cleanup();
        });
    }

    private void cleanup() {
        if (speechRecognizer != null) {
            try {
                speechRecognizer.destroy();
            } catch (Exception ignored) {}
            speechRecognizer = null;
        }
    }

    private String sanitizeName(String name) {
        if (name == null || name.trim().isEmpty()) return "Person";
        String clean = name.trim().replaceAll("[^a-zA-Z0-9 ]", "");
        if (clean.isEmpty()) return "Person";
        // Capitalize first letter of each word
        String[] words = clean.split("\\s+");
        StringBuilder sb = new StringBuilder();
        for (String w : words) {
            if (!w.isEmpty()) {
                sb.append(Character.toUpperCase(w.charAt(0)));
                if (w.length() > 1) {
                    sb.append(w.substring(1).toLowerCase());
                }
                sb.append(" ");
            }
        }
        return sb.toString().trim();
    }

    private String getErrorMessage(int errorCode) {
        switch (errorCode) {
            case SpeechRecognizer.ERROR_AUDIO:
                return "Audio recording error";
            case SpeechRecognizer.ERROR_CLIENT:
                return "Client side error";
            case SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS:
                return "Microphone permission required";
            case SpeechRecognizer.ERROR_NETWORK:
            case SpeechRecognizer.ERROR_NETWORK_TIMEOUT:
                return "Network connection needed for speech recognition";
            case SpeechRecognizer.ERROR_NO_MATCH:
                return "No name detected, please speak clearly";
            case SpeechRecognizer.ERROR_RECOGNIZER_BUSY:
                return "Speech service is busy, retry";
            case SpeechRecognizer.ERROR_SPEECH_TIMEOUT:
                return "No speech heard (timeout)";
            default:
                return "Speech error code " + errorCode;
        }
    }
}
