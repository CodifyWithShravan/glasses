#include <WiFi.h>
#include <WebServer.h>
#include <WiFiUdp.h>
#include "esp_camera.h"
#include "esp_http_server.h"
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
const unsigned long WAIT_TIME = 180000;
unsigned long BLINK_INTERVAL = 350;

bool ledState = false, overheatIndicator = false;

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
// Connect momentary 4-pin switch between GPIO 19 and GND.
#define BUTTON_PIN 19

unsigned long buttonPressStartTime = 0;
bool buttonIsPressed = false;
bool longPressTriggered = false;

// ─── I2S Audio Configuration (MAX98357A) ───────────────────
// GPIO 41 used for BCLK (GPIO 48 is onboard RGB LED and caused blinding light + lag)
#define I2S_BCLK_PIN    41   // Bit Clock (Connect to Pin 41 on right header)
#define I2S_LRC_PIN     21   // Left/Right Clock (Word Select)
#define I2S_DOUT_PIN    47   // Data Out to MAX98357A DIN
#define I2S_PORT        I2S_NUM_0
#define AUDIO_SAMPLE_RATE  16000
#define AUDIO_BUF_SIZE     512   // Bytes per I2S write chunk

httpd_handle_t stream_httpd = NULL;

// Handler to stream camera frames continuously as MJPEG
static esp_err_t stream_handler(httpd_req_t *req) {
  camera_fb_t * fb = NULL;
  esp_err_t res = ESP_OK;
  char part_buf[64];

  res = httpd_resp_set_type(req, "multipart/x-mixed-replace; boundary=frame");
  if (res != ESP_OK) {
    return res;
  }

  while (true) {
    fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println("Camera capture failed, retrying...");
      vTaskDelay(pdMS_TO_TICKS(10));
      continue;
    }

    if (fb->format != PIXFORMAT_JPEG) {
      esp_camera_fb_return(fb);
      vTaskDelay(pdMS_TO_TICKS(10));
      continue;
    }

    size_t hlen = snprintf(part_buf, 64, "\r\n--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n", fb->len);
    res = httpd_resp_send_chunk(req, part_buf, hlen);
    if (res == ESP_OK) {
      res = httpd_resp_send_chunk(req, (const char *)fb->buf, fb->len);
    }

    esp_camera_fb_return(fb);
    if (res != ESP_OK) {
      break; // Client disconnected
    }
  }

  return res;
}

void startCameraServer() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 81; // Running stream on Port 81
  config.ctrl_port = 32768;
  config.max_open_sockets = 4;
  config.lru_purge_enable = true;
  config.task_priority = tskIDLE_PRIORITY + 5; // High priority task for stream handling
  config.stack_size = 8192;                    // 8 KB stack for robust JPEG packet handling

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

// ─── Continuous Idle Sine Wave Generator ────────────────────
#define ENABLE_IDLE_SINE_WAVE  false // Set false to eliminate Class-D electrical switching noise on camera 3.3V rail
#define SINE_FREQ_HZ           440   // 440 Hz (Standard A note)
#define SINE_AMPLITUDE         2500  // Clean, comfortable tone (out of 32767)

// ─── Speaker Protection & Volume Scaling ────────────────────
// MAX98357A on 3.3V into 8Ω delivers max ~0.56W at 0 dBFS.
// 0.70 scaling limits max output power to ~0.28W, keeping the 0.5W
// voice coil cool, safe, and distortion-free.
#define AUDIO_VOLUME_SCALE 0.70f

int16_t sineWaveCycle[64];
int sineCycleLength = 0;
int sinePhaseIndex = 0;
volatile bool isPlayingTTS = false;

void initSineWave() {
#if ENABLE_IDLE_SINE_WAVE
  sineCycleLength = AUDIO_SAMPLE_RATE / SINE_FREQ_HZ;
  if (sineCycleLength > 64) sineCycleLength = 64;
  for (int i = 0; i < sineCycleLength; i++) {
    sineWaveCycle[i] = (int16_t)(sin(2.0 * PI * i / sineCycleLength) * SINE_AMPLITUDE);
  }
  Serial.printf("Sine wave generator initialized (%d Hz @ %d Hz sample rate, Safe Power Mode)\n", SINE_FREQ_HZ, AUDIO_SAMPLE_RATE);
#endif
}

void playIdleSineWave() {
#if ENABLE_IDLE_SINE_WAVE
  if (isPlayingTTS) return;

  // Generate a small chunk of 64 samples (128 bytes)
  int16_t buffer[64];
  for (int i = 0; i < 64; i++) {
    buffer[i] = sineWaveCycle[sinePhaseIndex];
    sinePhaseIndex = (sinePhaseIndex + 1) % sineCycleLength;
  }

  size_t bytesWritten = 0;
  // Non-blocking quick DMA write (10ms timeout)
  i2s_write(I2S_PORT, (const char*)buffer, sizeof(buffer), &bytesWritten, 10);
#endif
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
  initSineWave();
  Serial.println("I2S Audio initialized successfully.");
}

void handleAudioData(WiFiClient& client, long pcmLength, int sampleRate) {
  if (pcmLength <= 0 || sampleRate <= 0) {
    client.println("ERROR: INVALID AUDIO PARAMS");
    return;
  }
  
  // 1. Immediately pause the idle sine wave
  isPlayingTTS = true;
  i2s_zero_dma_buffer(I2S_PORT);

  // 2. Set sample rate for incoming TTS
  i2s_set_sample_rates(I2S_PORT, sampleRate);
  Serial.printf("Audio stream (TTS): %ld bytes @ %d Hz (Scaled 0.70x for 8Ω 0.5W speaker)\n", pcmLength, sampleRate);
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
      // Scale 16-bit PCM samples to safe output power for 8Ω 0.5W speaker
      int16_t* samples = (int16_t*)buffer;
      int sampleCount = bytesRead / 2;
      for (int i = 0; i < sampleCount; i++) {
        samples[i] = (int16_t)(samples[i] * AUDIO_VOLUME_SCALE);
      }

      i2s_write(I2S_PORT, buffer, bytesRead, &bytesWritten, portMAX_DELAY);
      bytesRemaining -= bytesRead;
    } else {
      break;
    }
  }
  
  // 3. Clear DMA buffer & restore default sample rate for idle sine wave
  i2s_zero_dma_buffer(I2S_PORT);
  i2s_set_sample_rates(I2S_PORT, AUDIO_SAMPLE_RATE);
  Serial.printf("TTS complete. Resuming idle sine wave.\n");

  // 4. Resume idle sine wave
  isPlayingTTS = false;
}

unsigned long lastReleaseTime = 0;
int tapCount = 0;
const unsigned long DOUBLE_TAP_GAP_MS = 350;

void setupButton() {
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  udp.begin(UDP_BUTTON_PORT);
  Serial.printf("Hardware button listener initialized on GPIO %d (Single/Double/Hold)\n", BUTTON_PIN);
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
  config.xclk_freq_hz = 20000000; // 20 MHz clean pixel clock (eliminates horizontal scan lines)
  config.pixel_format = PIXFORMAT_JPEG;

  // Utilize PSRAM if available for smooth double buffering
  if (psramFound()) {
    config.frame_size = FRAMESIZE_QVGA; // 320x240 (ultra smooth, high FPS, zero Wi-Fi drops)
    config.jpeg_quality = 12;          // High clarity
    config.fb_count = 2;
    config.grab_mode = CAMERA_GRAB_LATEST;
  } else {
    config.frame_size = FRAMESIZE_QVGA;
    config.jpeg_quality = 12;
    config.fb_count = 1;
  }

  neopixelWrite(RGB_BUILTIN, 0, 0, 0);

  // Camera Init
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x\n", err);
    return;
  }

  // Sensor Register Tuning: Eliminate horizontal scan lines and color artifacts
  sensor_t * s = esp_camera_sensor_get();
  if (s) {
    s->set_brightness(s, 0);
    s->set_contrast(s, 0);
    s->set_saturation(s, 0);
    s->set_whitebal(s, 1);       // Auto White Balance
    s->set_awb_gain(s, 1);
    s->set_wb_mode(s, 0);
    s->set_exposure_ctrl(s, 1);  // Auto Exposure
    s->set_aec2(s, 1);           // Enable DSP Auto Exposure algorithm
    s->set_ae_level(s, -1);      // Lower exposure level to prevent blown-out highlights
    s->set_gain_ctrl(s, 1);      // Auto Gain
    s->set_gainceiling(s, (gainceiling_t)0); // Clamp gain ceiling to 2x (prevents amplifying high-frequency noise lines)
    s->set_bpc(s, 1);            // Black Pixel Correction (cleans sensor lines)
    s->set_wpc(s, 1);            // White Pixel Correction (cleans sensor lines)
    s->set_raw_gma(s, 1);        // Gamma Correction
    s->set_lenc(s, 1);           // Lens Correction
    s->set_dcw(s, 1);            // Downsize Compensation (cleans horizontal banding)
  }

  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  // Set the ESP to act as a Wi-Fi Access Point with maximum RF power
  Serial.println("Starting Access Point...");
  WiFi.mode(WIFI_AP);
  WiFi.softAP(ssid, password, 1, 0, 4);
  WiFi.setTxPower(WIFI_POWER_19_5dBm);

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
  
  // 1. Process hardware button presses (Single Tap / Double Tap / Long Press)
  handleButton();

  // 2. Play continuous idle sine wave through speaker when speaker is free
  playIdleSineWave();

  // 3. Continuously listen for incoming client requests
  server.handleClient();

  WiFiClient client = tcpServer.available();
  if (client) {
    Serial.println("New TCP Client Connected!");
    
    // Keep the connection alive as long as the client is connected
    while (client.connected()) {
      handleButton();
      playIdleSineWave();

      if (client.available()) {
        // Read the incoming TCP packet until a newline character
        String command = client.readStringUntil('\n'); 
        command.trim(); // Clean up trailing \r or \n
        
        Serial.println("Received command: " + command);
        
        // Simple command handling logic
        if (command == "PING") {
          client.println("PONG");
        } else if (command == "STATUS") {
          client.println("ESP32 SYSTEM OK");
        } else if(command == "SENDIMG"){
          printToSerialSize(200, client);
        } else if (command.startsWith("AUDIO:")) {
          // Audio streaming protocol: AUDIO:<pcm_length>:<sample_rate>
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
    // Close the connection when the client disconnects
    client.stop();
    Serial.println("Client Disconnected.");
  }

  if (currentMillis >= WAIT_TIME) {

    // 2. Non-blocking blink timer
    if (currentMillis - previousBlinkMillis >= BLINK_INTERVAL) {
      previousBlinkMillis = currentMillis;
      ledState = !ledState; // Toggle ON/OFF state

      if (ledState) {
        neopixelWrite(RGB_BUILTIN, RGB_BRIGHTNESS, 0, 0); // GREEN (R, G, B)
      } else {
        neopixelWrite(RGB_BUILTIN, 0, 0, 0);             // OFF
      }
    }
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