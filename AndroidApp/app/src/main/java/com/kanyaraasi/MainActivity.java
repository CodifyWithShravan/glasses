package com.kanyaraasi;
import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
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
import android.view.View;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.widget.Button;
import android.widget.QuickContactBadge;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.kanyaraasi.glassescontroller.R;


public class MainActivity extends AppCompatActivity{

    TextView mainText, connStatus;
    Button submitButton, retryButton, toggleCameraButton;
    WebView camView;
    Context context = this;
    boolean camIsActive = false, justNowConnected = true;

    private SerialScanner serialScanner = new SerialScanner();
    ConnectivityManager connManager;
    ConnectivityManager.NetworkCallback networkCallback;

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
        camView = findViewById(R.id.cam_view);

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
                        camView.reload();
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
                toggleCamera();
            }
        });

        camView.setOnLongClickListener(new View.OnLongClickListener() {
            @Override
            public boolean onLongClick(View v) {

                System.out.println("Reloading URL...");
                camView.reload();
                camView.loadUrl("http://192.168.4.1:81/stream");
                return true;
            }
        });

        WebSettings camWebSettings = camView.getSettings();
        camWebSettings.setJavaScriptEnabled(true);
        camWebSettings.setLoadWithOverviewMode(true);
        camWebSettings.setUseWideViewPort(true);

        /*String html = "<html><head><style>" +
                "* { margin: 0; padding: 0; box-sizing: border-box; }" +
                "html, body { width: 100%; height: 100%; background-color: #000000; display: flex; justify-content: center; align-items: center; overflow: hidden; }" +
                "img { width: 100%; height: 100%; object-fit: contain; }" + // Use 'cover' to crop and fill completely
                "</style></head><body>" +
                "<img src='http://192.168.4.1:81/stream' />" +
                "</body></html>";*/

        //camView.loadDataWithBaseURL(null, html, "text/html", "UTF-8", null);
        camView.loadUrl("http://192.168.4.1:81/stream");

        new Handler(Looper.getMainLooper()).postDelayed(() -> {
            runnable.run();
        }, 2000);
    }

    @Override
    protected void onDestroy(){
        super.onDestroy();
        handler.removeCallbacks(runnable);
    }


    private void checkPermissionsAndConnect() {
        // Android requires FINE_LOCATION to scan for Wi-Fi SSIDs
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION)
                != PackageManager.PERMISSION_GRANTED) {

            // Request the permission from the user
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.ACCESS_FINE_LOCATION},
                    101);
        } else {
            // Permission already granted, trigger the connection
            // Pass 'this' as the Context to your standard class
            serialScanner.connectToEsp32(this);
        }
    }

    private void toggleCamera(){
        if(camIsActive){
            camView.loadUrl("about:blank");
            camIsActive = false;
        } else {
            camView.loadUrl("http://192.168.4.1:81/stream");
            camIsActive = true;
        }
    }
}
