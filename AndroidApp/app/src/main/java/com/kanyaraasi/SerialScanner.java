package com.kanyaraasi;
import android.content.Context;
import android.net.ConnectivityManager;
import android.net.Network;
import android.net.NetworkCapabilities;
import android.net.NetworkRequest;
import android.net.wifi.WifiNetworkSpecifier;
import android.os.Build;
import android.widget.TextView;

import org.w3c.dom.Text;

import java.io.InputStream;
import java.io.OutputStream;
import java.io.Serial;
import java.net.Socket;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class SerialScanner {

    SerialScanner(){

    }

    boolean isWifiConnected = false;

    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    // Method to initiate the dual-network state
    public void connectToEsp32(Context context) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            ConnectivityManager connManager = (ConnectivityManager) context.getSystemService(Context.CONNECTIVITY_SERVICE);

            // Define the specific ESP32 network credentials
            WifiNetworkSpecifier specifier = new WifiNetworkSpecifier.Builder()
                    .setSsid("SampleESPNetwork")
                    .setWpa2Passphrase("samplepassword")
                    .build();

            // Request a Wi-Fi connection explicitly without internet capabilities
            NetworkRequest request = new NetworkRequest.Builder()
                    .addTransportType(NetworkCapabilities.TRANSPORT_WIFI)
                    .removeCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET)
                    .setNetworkSpecifier(specifier)
                    .build();

            connManager.requestNetwork(request, new ConnectivityManager.NetworkCallback() {
                @Override
                public void onAvailable(Network network) {
                    super.onAvailable(network);

                    // CRITICAL STEP: Bind this app's process to the ESP32 network
                    connManager.bindProcessToNetwork(network);
                    isWifiConnected = true;
                }

                @Override
                public void onLost(Network network) {
                    super.onLost(network);
                    // Unbind when disconnected to restore default OS routing
                    connManager.bindProcessToNetwork(null);
                    isWifiConnected = false;
                }
            });
        }
    }

    public void sendTcpCommand(String command, TextView mainText) {
        executor.execute(() -> {
            try {
                // Connect to the ESP32 IP and Port (matching your Arduino code)
                Socket socket = new Socket("192.168.4.1", 8080);

                // Send the command string
                OutputStream output = socket.getOutputStream();
                output.write((command + "\n").getBytes());
                output.flush();

                // Read the incoming response
                InputStream input = socket.getInputStream();
                byte[] buffer = new byte[1024];
                int bytesRead = input.read(buffer);

                if (bytesRead > 0) {
                    String response = new String(buffer, 0, bytesRead);
                    System.out.println("ESP32 Response: " + response);
                    mainText.setText(response);
                }

                // Close the socket to free resources
                socket.close();

            } catch (Exception e) {
                e.printStackTrace();
            }
        });
    }

    boolean isConnected(){
        return isWifiConnected;
    }
}
