package com.kanyaraasi;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
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
                connection.connect();

                int responseCode = connection.getResponseCode();
                if (responseCode != HttpURLConnection.HTTP_OK) {
                    Log.e(TAG, "Connection failed with response code: " + responseCode + ". Retrying...");
                    Thread.sleep(1000);
                    continue; // retry
                }

                Log.d(TAG, "Connected! Content-Type: " + connection.getContentType());
                inputStream = new BufferedInputStream(connection.getInputStream(), 16384);
                int frameCount = 0;

                while (isRunning) {
                    // Read the multipart header until \r\n\r\n
                    String header = readHeader(inputStream);
                    if (header == null) {
                        Log.w(TAG, "Stream ended (null header)");
                        break;
                    }

                    int contentLength = parseContentLength(header);

                    if (contentLength > 0) {
                        // Read the JPEG payload
                        byte[] jpegData = new byte[contentLength];
                        int bytesRead = 0;
                        while (bytesRead < contentLength) {
                            int read = inputStream.read(jpegData, bytesRead, contentLength - bytesRead);
                            if (read == -1) {
                                break;
                            }
                            bytesRead += read;
                        }

                        if (bytesRead == contentLength) {
                            frameCount++;
                            BitmapFactory.Options options = new BitmapFactory.Options();
                            options.inPreferredConfig = Bitmap.Config.ARGB_8888;
                            Bitmap bitmap = BitmapFactory.decodeByteArray(jpegData, 0, contentLength, options);
                            if (bitmap != null) {
                                // Send EVERY frame for display (camera preview)
                                if (displayListener != null) {
                                    displayListener.onFrameForDisplay(bitmap);
                                }

                                // Send every Nth frame for AI processing
                                if (processListener != null && frameCount % processEveryN == 0) {
                                    // Create an ARGB_8888 copy for AI processing
                                    Bitmap processCopy = bitmap.copy(Bitmap.Config.ARGB_8888, false);
                                    if (processCopy != null) {
                                        processListener.onFrameForProcessing(processCopy);
                                    } else {
                                        Log.w(TAG, "Failed to copy bitmap for processing");
                                    }
                                }
                            } else {
                                Log.w(TAG, "Failed to decode JPEG frame #" + frameCount);
                            }
                        } else {
                            Log.e(TAG, "Incomplete frame read: " + bytesRead + "/" + contentLength);
                            break;
                        }
                    } else if (contentLength == 0) {
                        // Skip empty frames
                        continue;
                    }
                    // If contentLength is -1 (not found in header), try reading until next boundary
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
        String[] lines = header.split("\\r?\\n");
        for (String line : lines) {
            if (line.toLowerCase().startsWith("content-length:")) {
                try {
                    return Integer.parseInt(line.substring(line.indexOf(":") + 1).trim());
                } catch (NumberFormatException e) {
                    Log.e(TAG, "Failed to parse content length from: " + line, e);
                }
            }
        }
        return -1;
    }

    /**
     * Reads bytes from the stream until a double CRLF (\r\n\r\n) is found,
     * which marks the end of a multipart header section.
     */
    private String readHeader(InputStream inputStream) throws IOException {
        ByteArrayOutputStream headerBuffer = new ByteArrayOutputStream();
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

