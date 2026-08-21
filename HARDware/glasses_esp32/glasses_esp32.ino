#include <WiFi.h>
#include <WebServer.h>
#include "esp_camera.h"
#include "esp_http_server.h"

// Set your desired network name and password (password must be at least 8 characters)
const char* ssid = "SampleESPNetwork";
const char* password = "samplepassword"; 

// Create a web server listening on port 80
WebServer server(80);
WiFiServer tcpServer(8080);

unsigned long previousBlinkMillis = 0;
const unsigned long WAIT_TIME = 90000;
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

httpd_handle_t stream_httpd = NULL;

// Handler to stream camera frames continuously as MJPEG
static esp_err_t stream_handler(httpd_req_t *req) {
  camera_fb_t * fb = NULL;
  esp_err_t res = ESP_OK;
  char part_buf[64];

  // 1. WAKE UP the camera sensor when a viewer connects
  digitalWrite(LED_PIN, HIGH);
  pinMode(PWDN_GPIO_NUM, OUTPUT);
  digitalWrite(PWDN_GPIO_NUM, LOW);
  //esp_camera_init(&config);
  sensor_t * s = esp_camera_sensor_get();
  if (s) {
    s->set_reg(s, 0x09, 0x01, 0x00); // Clear standby bit (Wake up)
  }

  res = httpd_resp_set_type(req, "multipart/x-mixed-replace; boundary=frame");
  if(res != ESP_OK) return res;

  while(true) {
    fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println("Camera capture failed");
      res = ESP_FAIL;
    } else {
      if(fb->format != PIXFORMAT_JPEG) {
        res = ESP_FAIL;
      } else {
        size_t hlen = snprintf(part_buf, 64, "\r\n--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n", fb->len);
        res = httpd_resp_send_chunk(req, part_buf, hlen);
        if(res == ESP_OK) {
          res = httpd_resp_send_chunk(req, (const char *)fb->buf, fb->len);
        }
      }
      esp_camera_fb_return(fb);
      if(res != ESP_OK) break; // Client disconnected!
    }
  }

  // 2. PUT SENSOR TO SLEEP when the connection drops or the browser closes
  if (s) {
    s->set_reg(s, 0x09, 0x01, 0x10); // Set standby bit (Power down internal die)
  }

  digitalWrite(LED_PIN, LOW);
  pinMode(PWDN_GPIO_NUM, OUTPUT);
  digitalWrite(PWDN_GPIO_NUM, HIGH);

  return res;
}

void startCameraServer() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 81; // Running stream on Port 81

  httpd_uri_t stream_uri = {
    .uri       = "/stream",
    .method    = HTTP_GET,
    .handler   = stream_handler,
    .user_ctx  = NULL
  };

  if (httpd_start(&stream_httpd, &config) == ESP_OK) {
    httpd_register_uri_handler(stream_httpd, &stream_uri);
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
  config.xclk_freq_hz = 10000000;
  config.pixel_format = PIXFORMAT_JPEG;

  // Utilize PSRAM if available for double buffering
  if (psramFound()) {
    config.frame_size = FRAMESIZE_VGA; // 640x480
    config.jpeg_quality = 15;          // 1-63 (lower means higher quality)
    config.fb_count = 2;
    config.grab_mode = CAMERA_GRAB_LATEST;
  } else {
    config.frame_size = FRAMESIZE_QVGA; // 320x240
    config.jpeg_quality = 15;
    config.fb_count = 1;
  }

  neopixelWrite(RGB_BUILTIN, 0, 0, 0);

  // Camera Init
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x\n", err);
    return;
  }

  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  // Set the ESP to act as a Wi-Fi Access Point
  Serial.println("Starting Access Point...");
  WiFi.softAP(ssid, password);

  //tcpServer.begin();

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
  // Continuously listen for incoming client requests
  server.handleClient();

  WiFiClient client = tcpServer.available();
  if (client) {
    Serial.println("New TCP Client Connected!");
    
    // Keep the connection alive as long as the client is connected
    while (client.connected()) {
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