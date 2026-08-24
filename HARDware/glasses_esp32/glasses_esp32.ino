#include <WiFi.h>
#include <WebServer.h>
#include <WiFiUdp.h>
#include "esp_camera.h"
#include "esp_http_server.h"
#include "esp_wifi.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include <lwip/sockets.h>
#include <lwip/netdb.h>
#include "driver/i2s.h"

// Set your desired network name and password (password must be at least 8 characters)
const char* ssid = "SampleESPNetwork";
const char* password = "samplepassword"; 

// Create a web server listening on port 80
WebServer server(80);
WiFiServer tcpServer(8080);
WiFiUDP udp;
const int UDP_BUTTON_PORT = 8888;

unsigned long previousBlinkMillis = 0;
const unsigned long WAIT_TIME = 90000;
unsigned long BLINK_INTERVAL = 350;

bool ledState = false, overheatIndicator = false;

// Real-time diagnostics tracker
static volatile uint32_t streamFrameCount = 0;
static portMUX_TYPE streamClientMux = portMUX_INITIALIZER_UNLOCKED;
static bool streamClientActive = false;

#define PWDN_GPIO_NUM     -1
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM     15
#define SIOD_GPIO_NUM     4
#define SIOC_GPIO_NUM     5
#define Y9_GPIO_NUM       16
#define Y8_GPIO_NUM       17
#define Y7_GPIO_NUM       18
#define Y6_GPIO_NUM       12
#define Y5_GPIO_NUM       10
#define Y4_GPIO_NUM       8
#define Y3_GPIO_NUM       9
#define Y2_GPIO_NUM       11
#define VSYNC_GPIO_NUM    6
#define HREF_GPIO_NUM     7
#define PCLK_GPIO_NUM     13

#define LED_PIN 2

// ─── Hardware Button Configuration (Temple Arm Button) ─────
// Connect momentary button between GPIO 1 and GND.
#define BUTTON_PIN 1

unsigned long buttonPressStartTime = 0;
bool buttonIsPressed = false;
bool longPressTriggered = false;

// ─── I2S Audio Configuration (MAX98357A) ───────────────────
// NOTE: These pins must NOT conflict with the camera bus.
#define I2S_BCLK_PIN    14   // Bit Clock
#define I2S_LRC_PIN     21   // Left/Right Clock (Word Select)
#define I2S_DOUT_PIN    47   // Data Out to MAX98357A DIN
#define I2S_PORT        I2S_NUM_0
#define AUDIO_SAMPLE_RATE  16000
#define AUDIO_BUF_SIZE     512   // Bytes per I2S write chunk

// Keep the MJPEG application task away from the Wi-Fi/camera-driver core.
#define STREAM_HTTPD_CORE 1

httpd_handle_t stream_httpd = NULL;

// Handler to stream camera frames continuously as MJPEG
static esp_err_t stream_handler(httpd_req_t *req) {
  camera_fb_t * fb = NULL;
  esp_err_t res = ESP_OK;
  char part_buf[64];

  // Each client has its own capture loop. Two viewers therefore split the finite
  // camera/PSRAM/Wi-Fi budget and commonly make both previews run near 10 FPS.
  // The phone app is the only intended viewer, so reject extra stream clients.
  bool streamAlreadyActive;
  portENTER_CRITICAL(&streamClientMux);
  streamAlreadyActive = streamClientActive;
  if (!streamAlreadyActive) {
    streamClientActive = true;
  }
  portEXIT_CRITICAL(&streamClientMux);
  if (streamAlreadyActive) {
    httpd_resp_send_err(req, HTTPD_503_SERVICE_UNAVAILABLE,
                        "Camera stream is already in use by another client");
    return ESP_FAIL;
  }

  uint32_t windowFrames = 0;
  uint64_t windowCaptureUs = 0;
  uint64_t windowSendUs = 0;
  uint64_t windowBytes = 0;
  uint32_t windowCaptureFailures = 0;
  uint32_t lastMetricsMs = millis();

  // Disable TCP Nagle's algorithm & enlarge send buffer on stream socket to eliminate packet delay
  int sockfd = httpd_req_to_sockfd(req);
  if (sockfd >= 0) {
    int nodelay = 1;
    setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));
    int sndbuf = 32768;
    setsockopt(sockfd, SOL_SOCKET, SO_SNDBUF, &sndbuf, sizeof(sndbuf));
  }

  // 1. WAKE UP the camera sensor when a viewer connects
  digitalWrite(LED_PIN, HIGH);
  if (PWDN_GPIO_NUM != -1) {
    pinMode(PWDN_GPIO_NUM, OUTPUT);
    digitalWrite(PWDN_GPIO_NUM, LOW);
  }
  sensor_t * s = esp_camera_sensor_get();
  if (s) {
    s->set_reg(s, 0x09, 0x01, 0x00); // Clear standby bit (Wake up)
  }

  res = httpd_resp_set_type(req, "multipart/x-mixed-replace; boundary=frame");
  if (res != ESP_OK) {
    portENTER_CRITICAL(&streamClientMux);
    streamClientActive = false;
    portEXIT_CRITICAL(&streamClientMux);
    digitalWrite(LED_PIN, LOW);
    return res;
  }

  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  httpd_resp_set_hdr(req, "X-Framerate", "30");

  while (true) {
    int64_t captureStartedUs = esp_timer_get_time();
    fb = esp_camera_fb_get();
    int64_t captureFinishedUs = esp_timer_get_time();
    if (!fb) {
      windowCaptureFailures++;
      uint32_t nowMs = millis();
      if (nowMs - lastMetricsMs >= 1000) {
        Serial.printf("MJPEG [one client]: 0 FPS | camera returned no frames | dropped %lu\n",
                      (unsigned long)windowCaptureFailures);
        windowCaptureFailures = 0;
        lastMetricsMs = nowMs;
      }
      // Don't kill the stream connection on a single transient frame drop!
      vTaskDelay(pdMS_TO_TICKS(5));
      continue;
    }

    if (fb->format != PIXFORMAT_JPEG) {
      esp_camera_fb_return(fb);
      vTaskDelay(pdMS_TO_TICKS(5));
      continue;
    }

    size_t hlen = snprintf(part_buf, 64, "\r\n--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n", fb->len);
    int64_t sendStartedUs = esp_timer_get_time();
    res = httpd_resp_send_chunk(req, part_buf, hlen);
    if (res == ESP_OK) {
      res = httpd_resp_send_chunk(req, (const char *)fb->buf, fb->len);
      if (res == ESP_OK) {
        streamFrameCount++;
        windowFrames++;
        windowCaptureUs += captureFinishedUs - captureStartedUs;
        windowSendUs += esp_timer_get_time() - sendStartedUs;
        windowBytes += fb->len;
      }
    }
    esp_camera_fb_return(fb);

    uint32_t nowMs = millis();
    if (nowMs - lastMetricsMs >= 1000) {
      const float elapsedSeconds = (nowMs - lastMetricsMs) / 1000.0f;
      const float captureMs = windowFrames == 0 ? 0.0f
          : windowCaptureUs / (1000.0f * windowFrames);
      const float sendMs = windowFrames == 0 ? 0.0f
          : windowSendUs / (1000.0f * windowFrames);
      const float averageJpegKb = windowFrames == 0 ? 0.0f
          : windowBytes / (1024.0f * windowFrames);
      Serial.printf("MJPEG [one client]: %.1f FPS | capture %.1f ms | send %.1f ms | JPEG %.1f KB | dropped %lu\n",
                    windowFrames / elapsedSeconds, captureMs, sendMs, averageJpegKb,
                    (unsigned long)windowCaptureFailures);
      windowFrames = 0;
      windowCaptureUs = 0;
      windowSendUs = 0;
      windowBytes = 0;
      windowCaptureFailures = 0;
      lastMetricsMs = nowMs;
    }

    if (res != ESP_OK) {
      // Client disconnected
      break;
    }
  }

  // 2. PUT SENSOR TO SLEEP when the connection drops or the browser closes
  if (s) {
    s->set_reg(s, 0x09, 0x01, 0x10); // Set standby bit (Power down internal die)
  }

  digitalWrite(LED_PIN, LOW);
  if (PWDN_GPIO_NUM != -1) {
    pinMode(PWDN_GPIO_NUM, OUTPUT);
    digitalWrite(PWDN_GPIO_NUM, HIGH);
  }

  portENTER_CRITICAL(&streamClientMux);
  streamClientActive = false;
  portEXIT_CRITICAL(&streamClientMux);

  return res;
}

void startCameraServer() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 81;                     // Running stream on Port 81
  config.ctrl_port = 32768;
  config.max_open_sockets = 4;
  config.lru_purge_enable = true;
  config.task_priority = tskIDLE_PRIORITY + 5; // High priority stream task
  config.stack_size = 8192;                    // 8KB stack for robust networking
  config.core_id = STREAM_HTTPD_CORE;          // Core 0 is busy with Wi-Fi/camera work

  httpd_uri_t stream_uri = {
    .uri       = "/stream",
    .method    = HTTP_GET,
    .handler   = stream_handler,
    .user_ctx  = NULL
  };

  if (httpd_start(&stream_httpd, &config) == ESP_OK) {
    httpd_register_uri_handler(stream_httpd, &stream_uri);
    Serial.println("Camera Stream Server active on port 81 (/stream)");
  }
}

void setupI2SAudio() {
  i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX),
    .sample_rate = AUDIO_SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = 0,
    .dma_buf_count = 8,
    .dma_buf_len = 256,
    .use_apll = false,
    .tx_desc_auto_clear = true
  };

  i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_BCLK_PIN,
    .ws_io_num = I2S_LRC_PIN,
    .data_out_num = I2S_DOUT_PIN,
    .data_in_num = I2S_PIN_NO_CHANGE
  };

  esp_err_t err = i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
  if (err != ESP_OK) {
    Serial.printf("I2S driver install failed: 0x%x\n", err);
    return;
  }

  err = i2s_set_pin(I2S_PORT, &pin_config);
  if (err != ESP_OK) {
    Serial.printf("I2S set pin failed: 0x%x\n", err);
    return;
  }

  i2s_zero_dma_buffer(I2S_PORT);
  Serial.println("I2S Audio initialized successfully.");
}

void handleAudioData(WiFiClient& client, long pcmLength, int sampleRate) {
  if (pcmLength <= 0 || sampleRate <= 0) {
    client.println("ERROR: INVALID AUDIO PARAMS");
    return;
  }
  
  i2s_set_sample_rates(I2S_PORT, sampleRate);
  Serial.printf("Audio stream: %ld bytes @ %d Hz\n", pcmLength, sampleRate);
  client.println("OK");
  
  uint8_t buffer[AUDIO_BUF_SIZE];
  long bytesRemaining = pcmLength;
  size_t bytesWritten;
  
  while (bytesRemaining > 0 && client.connected()) {
    int toRead = min((long)AUDIO_BUF_SIZE, bytesRemaining);
    int bytesRead = 0;
    unsigned long readStart = millis();
    while (bytesRead < toRead && (millis() - readStart) < 5000) {
      if (client.available()) {
        int chunk = client.read(buffer + bytesRead, toRead - bytesRead);
        if (chunk > 0) bytesRead += chunk;
      }
    }
    if (bytesRead > 0) {
      i2s_write(I2S_PORT, buffer, bytesRead, &bytesWritten, portMAX_DELAY);
      bytesRemaining -= bytesRead;
    } else {
      break;
    }
  }
  
  i2s_zero_dma_buffer(I2S_PORT);
  Serial.printf("Audio complete. %ld bytes remaining.\n", bytesRemaining);
}

unsigned long lastReleaseTime = 0;
int tapCount = 0;
const unsigned long DOUBLE_TAP_GAP_MS = 350;

void setupButton() {
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  udp.begin(UDP_BUTTON_PORT);
  Serial.println("Hardware button listener initialized on GPIO 1 (Single/Double/Hold)");
}

void sendButtonEvent(const char* event) {
  IPAddress broadcastIp(192, 168, 4, 255);
  udp.beginPacket(broadcastIp, UDP_BUTTON_PORT);
  udp.write((const uint8_t*)event, strlen(event));
  udp.endPacket();
  Serial.printf("Hardware Button Event Broadcasted: %s\n", event);
}

void handleButton() {
  int reading = digitalRead(BUTTON_PIN);
  unsigned long now = millis();

  if (reading == LOW) { // Button is pressed (active low)
    if (!buttonIsPressed) {
      buttonIsPressed = true;
      buttonPressStartTime = now;
      longPressTriggered = false;
    } else {
      // Long press detection (> 1000ms)
      if (!longPressTriggered && (now - buttonPressStartTime >= 1000)) {
        longPressTriggered = true;
        tapCount = 0; // reset tap counter on long press
        sendButtonEvent("HOLD");
      }
    }
  } else { // Button is released
    if (buttonIsPressed) {
      unsigned long pressDuration = now - buttonPressStartTime;
      if (!longPressTriggered && pressDuration >= 50 && pressDuration < 1000) {
        tapCount++;
        lastReleaseTime = now;
      }
      buttonIsPressed = false;
      longPressTriggered = false;
    }
  }

  // Evaluate single vs double tap after timeout
  if (tapCount > 0 && !buttonIsPressed) {
    if (tapCount == 1 && (now - lastReleaseTime > DOUBLE_TAP_GAP_MS)) {
      sendButtonEvent("TAP");
      tapCount = 0;
    } else if (tapCount >= 2) {
      sendButtonEvent("DOUBLE_TAP");
      tapCount = 0;
    }
  }
}

void setup() {
  Serial.begin(115200);
  delay(500);

  Serial.println("\n==========================================");
  Serial.println("   ESP32-S3 Smart Glasses Firmware Boot   ");
  Serial.println("==========================================");
  Serial.printf("CPU Frequency: %d MHz\n", getCpuFreqMHz());
  Serial.printf("Internal Free Heap: %d bytes\n", ESP.getFreeHeap());

  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000; // 20 MHz for full 25-30 FPS streaming
  config.pixel_format = PIXFORMAT_JPEG;

  // Utilize PSRAM if available for triple buffering & high FPS
  if (psramFound()) {
    Serial.printf("PSRAM: ENABLED (Free PSRAM: %d bytes)\n", ESP.getFreePsram());
    // Hand landmarks are fed to a small model, so VGA only increases JPEG encode,
    // Wi-Fi, decode, and GC work. QVGA is fast enough for clear close-up signs and
    // is much better suited to a 30 FPS ESP32 camera pipeline.
    config.frame_size = FRAMESIZE_QVGA; // 320x240
    config.jpeg_quality = 15;           // Smaller JPEGs leave bandwidth for 30 FPS
    config.fb_count = 3;               // Triple buffer for seamless DMA pipelining
    config.grab_mode = CAMERA_GRAB_LATEST; // Always fetch freshest frame
  } else {
    Serial.println("WARNING: PSRAM NOT FOUND! Falling back to SRAM mode.");
    Serial.println("To fix: in Arduino IDE, select Tools -> PSRAM: 'OPI PSRAM'");
    config.frame_size = FRAMESIZE_QVGA; // 320x240
    config.jpeg_quality = 15;
    config.fb_count = 1;
    config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  }

  neopixelWrite(RGB_BUILTIN, 0, 0, 0);

  // Camera Init
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x\n", err);
    return;
  }

  // Sensor optimizations for low latency, high FPS, and fast shutter speed
  sensor_t * s = esp_camera_sensor_get();
  if (s) {
    Serial.printf("Camera sensor detected: PID 0x%04X\n", s->id.PID);
    s->set_vflip(s, 0);
    s->set_hmirror(s, 0);
    s->set_brightness(s, 0);
    s->set_contrast(s, 0);
    s->set_saturation(s, 0);
    s->set_special_effect(s, 0);
    s->set_whitebal(s, 1);       // Enable Auto White Balance
    s->set_awb_gain(s, 1);       // Auto White Balance Gain
    s->set_wb_mode(s, 0);
    s->set_exposure_ctrl(s, 1);  // Auto Exposure
    s->set_aec2(s, 0);           // Disable DSP AEC2 (DSP AEC2 causes slow FPS fluctuations)
    s->set_ae_level(s, 0);
    s->set_gain_ctrl(s, 1);      // Auto Gain Control
    s->set_gainceiling(s, (gainceiling_t)GAINCEILING_8X); // 8x Gain Ceiling allows 25-30 FPS shutter speed indoors
    s->set_bpc(s, 0);
    s->set_wpc(s, 1);
    s->set_raw_gma(s, 1);
    s->set_lenc(s, 0);
  }

  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  // Configure Wi-Fi Access Point with Zero Power-Save & Max RF Power
  Serial.println("Configuring Wi-Fi Access Point (Low-Latency Mode)...");
  WiFi.mode(WIFI_AP);
  WiFi.softAP(ssid, password, 1, 0, 4); // Channel 1, max 4 clients
  WiFi.setSleep(false);                 // Disable modem-sleep to eliminate 100-300ms latency spikes
  esp_wifi_set_ps(WIFI_PS_NONE);       // Enforce no power saving at hardware driver level
  WiFi.setTxPower(WIFI_POWER_19_5dBm);  // Maximum RF power for strong RSSI

  setupI2SAudio();
  setupButton();
  tcpServer.begin();

  // Retrieve and print the ESP's IP address (Usually 192.168.4.1)
  IPAddress IP = WiFi.softAPIP();
  Serial.print("AP IP address: ");
  Serial.println(IP);

  startCameraServer();

  // Define what happens when a client loads the root webpage ("/")
  server.on("/", []() {
    String html = "<html>"
                  "<head><title>ESP32 Camera</title></head>"
                  "<body style='text-align:center; background-color:#222; color:#fff; font-family:sans-serif;'>"
                  "  <h1>ESP32-S3 Live Feed</h1>"
                  "  <img src='http://192.168.4.1:81/stream' style='width:90%; max-width:640px; border-radius:10px;'>"
                  "</body>"
                  "</html>";
    server.send(200, "text/html", html);
  });

  // Start the web server
  server.begin();
  Serial.println("Web Server started.");
}

void loop() {
  unsigned long currentMillis = millis();
  
  // 1. Process hardware button presses (Single Tap / Long Press)
  handleButton();

  // 2. Continuously listen for incoming client requests
  server.handleClient();

  // 3. Non-blocking TCP Command handling
  WiFiClient client = tcpServer.available();
  if (client) {
    client.setNoDelay(true); // Disable Nagle's algorithm for instant response
    client.setTimeout(50);   // Max 50ms block, prevents 1-second default timeout freeze

    if (client.available()) {
      String command = client.readStringUntil('\n');
      command.trim();
      
      if (command.length() > 0) {
        Serial.println("Received TCP command: " + command);
        
        if (command == "PING") {
          client.println("PONG");
        } else if (command == "STATUS") {
          client.println("ESP32 SYSTEM OK");
        } else if (command == "SENDIMG") {
          printToSerialSize(200, client);
        } else if (command.startsWith("AUDIO:")) {
          String audioParams = command.substring(6);
          int colonIdx = audioParams.indexOf(':');
          if (colonIdx > 0) {
            long pcmLength = audioParams.substring(0, colonIdx).toInt();
            int sampleRate = audioParams.substring(colonIdx + 1).toInt();
            handleAudioData(client, pcmLength, sampleRate);
          } else {
            client.println("ERROR: INVALID AUDIO FORMAT");
          }
        } else if (command == "STOP_AUDIO") {
          i2s_zero_dma_buffer(I2S_PORT);
          client.println("AUDIO STOPPED");
        } else {
          client.println("ERROR: UNKNOWN COMMAND");
        }
      }
    }
    client.flush();
    client.stop(); // Explicitly close the short-lived command socket
  }

  if (currentMillis >= WAIT_TIME) {
    // Non-blocking blink timer
    if (currentMillis - previousBlinkMillis >= BLINK_INTERVAL) {
      previousBlinkMillis = currentMillis;
      ledState = !ledState; // Toggle ON/OFF state

      if (ledState) {
        neopixelWrite(RGB_BUILTIN, 16, 0, 0); // Mild indicator
      } else {
        neopixelWrite(RGB_BUILTIN, 0, 0, 0);  // OFF
      }
    }
  }

  // 5. Periodic live diagnostic statistics
  printSystemStats();
}

void printSystemStats() {
  static unsigned long lastStatsTime = 0;
  static uint32_t lastFrameCount = 0;
  unsigned long now = millis();

  // Print diagnostics every 5 seconds
  if (now - lastStatsTime >= 5000) {
    float elapsedSec = (now - lastStatsTime) / 1000.0;
    uint32_t currentFrames = streamFrameCount;
    float fps = (currentFrames - lastFrameCount) / elapsedSec;
    lastFrameCount = currentFrames;
    lastStatsTime = now;

    Serial.println("\n========= ESP32 LIVE DIAGNOSTICS =========");
    Serial.printf("Live Stream Frame Rate: %.1f FPS\n", fps);
    Serial.printf("Free Internal Heap:     %u bytes\n", ESP.getFreeHeap());
    Serial.printf("Free PSRAM:             %u bytes\n", ESP.getFreePsram());
    Serial.printf("Largest Free Block:     %u bytes\n", heap_caps_get_largest_free_block(MALLOC_CAP_8BIT));
    Serial.printf("CPU Frequency:          %d MHz\n", getCpuFreqMHz());

    Serial.println("==================================================================");
  }
}

void printToSerialSize(int sizeKB, WiFiClient client){
  uint32_t bytesSent = 0, TARGET_BYTES = sizeKB * 1024;
    while(bytesSent<TARGET_BYTES){
    char c;
    int roll = random(0, 100);

    // Create realistic-looking text structure
    if (roll < 15) {
      c = ' ';        // 15% chance of space
    } else if (roll < 18) {
      c = '\n';       // 3% chance of newline
    } else {
      c = random('a', 'z' + 1); // Lowercase letters
    }

    client.print(c);
    bytesSent++;
    }
    client.print("\n\n=== ");
    client.print(sizeKB);
    client.println("KB GENERATION COMPLETE ===");
    bytesSent++; // Stop execution loop
}
