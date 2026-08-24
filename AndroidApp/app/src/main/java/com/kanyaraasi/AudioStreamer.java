package com.kanyaraasi;

import android.content.Context;
import android.speech.tts.TextToSpeech;
import android.speech.tts.UtteranceProgressListener;
import android.util.Log;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.net.Socket;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Locale;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Handles TTS synthesis and streaming raw PCM audio to the ESP32.
 */
public class AudioStreamer {
    private static final String TAG = "AudioStreamer";
    private static final String ESP_IP = "192.168.4.1";
    private static final int ESP_PORT = 8080;

    private final Context context;
    private TextToSpeech tts;
    private final ExecutorService executorService;
    private final AtomicBoolean isSpeaking = new AtomicBoolean(false);
    private boolean isReady = false;

    public AudioStreamer(Context context) {
        this.context = context;
        this.executorService = Executors.newSingleThreadExecutor();
    }

    /**
     * Initializes the TTS engine.
     *
     * @return true if initialization was successful.
     */
    public boolean initialize() {
        CountDownLatch latch = new CountDownLatch(1);
        tts = new TextToSpeech(context, status -> {
            if (status == TextToSpeech.SUCCESS) {
                int result = tts.setLanguage(Locale.US);
                if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                    Log.e(TAG, "TTS Language not supported");
                } else {
                    isReady = true;
                }
            } else {
                Log.e(TAG, "TTS Initialization failed");
            }
            latch.countDown();
        });

        try {
            latch.await(5, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            Log.e(TAG, "Interrupted while waiting for TTS init", e);
        }

        return isReady;
    }

    public boolean isReady() {
        return isReady;
    }

    public void shutdown() {
        if (tts != null) {
            tts.stop();
            tts.shutdown();
        }
        executorService.shutdown();
    }

    /**
     * Synthesizes text to a WAV file and streams it over TCP.
     *
     * @param text The text to speak.
     */
    public void speakToGlasses(String text, android.net.Network network) {
        if (!isReady) {
            Log.e(TAG, "AudioStreamer is not ready");
            return;
        }

        if (!isSpeaking.compareAndSet(false, true)) {
            Log.w(TAG, "Already speaking, ignoring request: " + text);
            return;
        }

        executorService.execute(() -> processSpeech(text, network));
    }

    private void processSpeech(String text, android.net.Network network) {
        File cacheDir = context.getCacheDir();
        File tempWav = new File(cacheDir, "temp_tts.wav");
        String utteranceId = "tts_" + System.currentTimeMillis();
        CountDownLatch synthLatch = new CountDownLatch(1);
        AtomicBoolean synthSuccess = new AtomicBoolean(false);

        tts.setOnUtteranceProgressListener(new UtteranceProgressListener() {
            @Override
            public void onStart(String utteranceId) {
                // Started
            }

            @Override
            public void onDone(String utteranceId) {
                synthSuccess.set(true);
                synthLatch.countDown();
            }

            @Override
            public void onError(String utteranceId) {
                Log.e(TAG, "TTS synthesis failed");
                synthLatch.countDown();
            }
        });

        int result = tts.synthesizeToFile(text, null, tempWav, utteranceId);
        if (result != TextToSpeech.SUCCESS) {
            Log.e(TAG, "Failed to queue synthesize request");
            isSpeaking.set(false);
            return;
        }

        try {
            synthLatch.await(10, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            Log.e(TAG, "Interrupted waiting for synthesis", e);
        }

        if (synthSuccess.get() && tempWav.exists()) {
            streamAudioFile(tempWav, network);
        }

        if (tempWav.exists()) {
            tempWav.delete();
        }
        
        isSpeaking.set(false);
    }

    private void streamAudioFile(File wavFile, android.net.Network network) {
        try (FileInputStream fis = new FileInputStream(wavFile)) {
            byte[] header = new byte[44];
            if (fis.read(header) != 44) {
                Log.e(TAG, "Failed to read WAV header");
                return;
            }

            ByteBuffer buffer = ByteBuffer.wrap(header);
            buffer.order(ByteOrder.LITTLE_ENDIAN);

            short channels = buffer.getShort(22);
            int sampleRate = buffer.getInt(24);
            short bitsPerSample = buffer.getShort(34);

            Log.d(TAG, "WAV Info - Channels: " + channels + ", Sample Rate: " + sampleRate + ", Bits/Sample: " + bitsPerSample);

            long pcmLength = wavFile.length() - 44;

            Socket socket = null;
            try {
                if (network != null) {
                    socket = network.getSocketFactory().createSocket(ESP_IP, ESP_PORT);
                } else {
                    socket = new Socket(ESP_IP, ESP_PORT);
                }

                try (OutputStream out = socket.getOutputStream()) {
                     
                    String headerStr = String.format(Locale.US, "AUDIO:%d:%d\n", pcmLength, sampleRate);
                    out.write(headerStr.getBytes());

                    byte[] data = new byte[4096];
                    int read;
                    while ((read = fis.read(data)) != -1) {
                        out.write(data, 0, read);
                    }
                    out.flush();
                    Log.d(TAG, "Audio stream completed successfully");
                } catch (IOException e) {
                    Log.e(TAG, "Error streaming to ESP32", e);
                }
            } catch (IOException e) {
                Log.e(TAG, "Error creating socket to ESP32", e);
            } finally {
                if (socket != null) {
                    try {
                        socket.close();
                    } catch (IOException e) {
                        Log.e(TAG, "Failed to close socket", e);
                    }
                }
            }

        } catch (IOException e) {
            Log.e(TAG, "Error reading WAV file", e);
        }
    }
}
