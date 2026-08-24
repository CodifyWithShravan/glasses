package com.kanyaraasi;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.SystemClock;
import android.util.Log;

import java.io.BufferedInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Parses an MJPEG stream from the ESP32 and extracts individual JPEG frames as Bitmaps.
 * Supports two listeners:
 *   - OnFrameDisplayListener: receives EVERY frame (for live camera preview)
 *   - OnFrameProcessListener: receives every Nth frame (for AI classification)
 */
public class MjpegStreamParser {
    private static final String TAG = "MjpegStreamParser";
    private static final int CONNECT_TIMEOUT_MS = 5000;
    private static final int READ_TIMEOUT_MS = 5000;

    private final String streamUrl;
    private final android.net.Network network;
    private OnFrameDisplayListener displayListener;
    private OnFrameProcessListener processListener;
    private int processEveryN = 1; // Process frames for AI immediately with zero delay

    private ExecutorService executorService;
    private volatile boolean isRunning = false;

    /**
     * Listener that receives EVERY frame for display purposes.
     */
    public interface OnFrameDisplayListener {
        void onFrameForDisplay(Bitmap frame);
    }

    /**
     * Listener that receives every Nth frame for AI processing.
     */
    public interface OnFrameProcessListener {
        /**
         * Called before making the expensive ARGB copy needed by the AI pipeline.
         * Return false when the previous inference is still running so this frame can
         * be discarded without allocating another full-size bitmap.
         */
        boolean canAcceptFrame();

        void onFrameForProcessing(Bitmap frame);
    }

    /**
     * Constructor for MjpegStreamParser.
     *
     * @param streamUrl The URL of the MJPEG stream.
     * @param network   The specific ESP32 network to route traffic through (can be null).
     */
    public MjpegStreamParser(String streamUrl, android.net.Network network) {
        this.streamUrl = streamUrl;
        this.network = network;
    }

    /**
     * Set the display listener (receives every frame for camera preview).
     */
    public void setDisplayListener(OnFrameDisplayListener listener) {
        this.displayListener = listener;
    }

    /**
     * Set the process listener (receives every Nth frame for AI).
     */
    public void setProcessListener(OnFrameProcessListener listener) {
        this.processListener = listener;
    }

    /**
     * Sets how often frames are sent to the process listener.
     * E.g. processEveryN=3 means every 3rd frame goes to AI.
     */
    public void setProcessEveryN(int n) {
        this.processEveryN = Math.max(1, n);
    }

    /**
     * Starts the MJPEG stream parsing on a background thread.
     */
    public void start() {
        if (isRunning) {
            Log.w(TAG, "Parser already running");
            return;
        }
        isRunning = true;
        executorService = Executors.newSingleThreadExecutor();
        executorService.execute(this::parseStream);
        Log.d(TAG, "Stream parser started for: " + streamUrl);
    }

    /**
     * Stops the stream parsing.
     */
    public void stop() {
        Log.d(TAG, "Stopping stream parser...");
        isRunning = false;
        if (executorService != null) {
            executorService.shutdownNow();
            executorService = null;
        }
    }

    /**
     * Returns whether the parser is currently running.
     */
    public boolean isRunning() {
        return isRunning;
    }

    private void parseStream() {
        byte[] readBuffer = new byte[65536];
        ByteArrayOutputStream headerBuffer = new ByteArrayOutputStream(256);
        BitmapFactory.Options decodeOptions = new BitmapFactory.Options();
        decodeOptions.inPreferredConfig = Bitmap.Config.RGB_565;
        decodeOptions.inSampleSize = 1;

        while (isRunning) {
            HttpURLConnection connection = null;
            InputStream inputStream = null;
            try {
                Log.d(TAG, "Connecting to MJPEG stream: " + streamUrl);
                URL url = new URL(streamUrl);
                if (network != null) {
                    connection = (HttpURLConnection) network.openConnection(url);
                } else {
                    connection = (HttpURLConnection) url.openConnection();
                }
                connection.setConnectTimeout(CONNECT_TIMEOUT_MS);
                connection.setReadTimeout(READ_TIMEOUT_MS);
                connection.setUseCaches(false);
                connection.setRequestProperty("Connection", "Keep-Alive");
                connection.connect();

                int responseCode = connection.getResponseCode();
                if (responseCode != HttpURLConnection.HTTP_OK) {
                    Log.e(TAG, "Connection failed with response code: " + responseCode + ". Retrying...");
                    Thread.sleep(1000);
                    continue; // retry
                }

                Log.d(TAG, "Connected! Content-Type: " + connection.getContentType());
                inputStream = new BufferedInputStream(connection.getInputStream(), 65536);
                int frameCount = 0;
                int framesSinceLastReport = 0;
                long lastFpsReportMs = SystemClock.elapsedRealtime();

                while (isRunning) {
                    // Read the multipart header until \r\n\r\n
                    String header = readHeader(inputStream, headerBuffer);
                    if (header == null) {
                        Log.w(TAG, "Stream ended (null header)");
                        break;
                    }

                    int contentLength = parseContentLength(header);

                    if (contentLength > 0) {
                        // Ensure readBuffer is large enough
                        if (readBuffer.length < contentLength) {
                            readBuffer = new byte[contentLength + 8192];
                        }

                        int bytesRead = 0;
                        while (bytesRead < contentLength && isRunning) {
                            int read = inputStream.read(readBuffer, bytesRead, contentLength - bytesRead);
                            if (read == -1) {
                                break;
                            }
                            bytesRead += read;
                        }

                        if (bytesRead == contentLength) {
                            frameCount++;
                            framesSinceLastReport++;
                            Bitmap bitmap = BitmapFactory.decodeByteArray(readBuffer, 0, contentLength, decodeOptions);
                            
                            if (bitmap != null) {
                                // Send EVERY frame for display (camera preview)
                                if (displayListener != null) {
                                    displayListener.onFrameForDisplay(bitmap);
                                }

                                // Send every Nth frame for AI processing
                                if (processListener != null
                                        && frameCount % processEveryN == 0
                                        && processListener.canAcceptFrame()) {
                                    Bitmap processCopy = bitmap.copy(Bitmap.Config.ARGB_8888, false);
                                    if (processCopy != null) {
                                        processListener.onFrameForProcessing(processCopy);
                                    }
                                }

                                long nowMs = SystemClock.elapsedRealtime();
                                if (nowMs - lastFpsReportMs >= 1000) {
                                    float fps = framesSinceLastReport * 1000f / (nowMs - lastFpsReportMs);
                                    Log.i(TAG, String.format(java.util.Locale.US,
                                            "Decoded %.1f FPS (%d frames, %dx%d)", fps,
                                            framesSinceLastReport, bitmap.getWidth(), bitmap.getHeight()));
                                    framesSinceLastReport = 0;
                                    lastFpsReportMs = nowMs;
                                }
                            } else {
                                Log.w(TAG, "Failed to decode JPEG frame #" + frameCount);
                            }
                        } else {
                            Log.e(TAG, "Incomplete frame read: " + bytesRead + "/" + contentLength);
                            break;
                        }
                    } else if (contentLength == 0) {
                        continue;
                    }
                }

            } catch (Exception e) {
                if (isRunning) {
                    Log.e(TAG, "Error in stream parsing, retrying in 1 second...", e);
                    try {
                        Thread.sleep(1000);
                    } catch (InterruptedException ie) {
                        break;
                    }
                }
            } finally {
                if (inputStream != null) {
                    try {
                        inputStream.close();
                    } catch (IOException e) {
                        Log.e(TAG, "Failed to close input stream", e);
                    }
                }
                if (connection != null) {
                    connection.disconnect();
                }
            }
        }
        Log.d(TAG, "Stream parser stopped.");
    }

    /**
     * Parse Content-Length from the multipart header.
     */
    private int parseContentLength(String header) {
        int idx = header.toLowerCase().indexOf("content-length:");
        if (idx != -1) {
            int start = idx + 15;
            int end = header.indexOf("\r\n", start);
            if (end == -1) end = header.indexOf("\n", start);
            if (end == -1) end = header.length();
            try {
                return Integer.parseInt(header.substring(start, end).trim());
            } catch (NumberFormatException e) {
                Log.e(TAG, "Failed to parse content length from: " + header, e);
            }
        }
        return -1;
    }

    /**
     * Reads bytes from the stream until a double CRLF (\r\n\r\n) is found,
     * which marks the end of a multipart header section.
     */
    private String readHeader(InputStream inputStream, ByteArrayOutputStream headerBuffer) throws IOException {
        headerBuffer.reset();
        int prevChar = -1;
        int currChar;
        int crlfCount = 0;

        while ((currChar = inputStream.read()) != -1) {
            headerBuffer.write(currChar);

            if (prevChar == '\r' && currChar == '\n') {
                crlfCount++;
                if (crlfCount == 2) {
                    return headerBuffer.toString("UTF-8");
                }
            } else if (currChar != '\r') {
                crlfCount = 0;
            }
            prevChar = currChar;
        }
        return null;
    }
}
