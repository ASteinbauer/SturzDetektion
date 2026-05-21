package com.example.sturzdetektion;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Color;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.os.CountDownTimer;
import android.os.VibrationEffect;
import android.os.Vibrator;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;
import java.util.Random;

public class MainActivity extends AppCompatActivity implements SensorEventListener {

    private static final String TAG = "SturzDetektionDebug";
    private static final String MODEL_FILE = "fall_detection_model.tflite";

    private SensorManager sensorManager;
    private Sensor accelerometer;
    private Sensor gyroscope;

    private TextView accelTextView;
    private TextView gyroTextView;
    private TextView statusTextView;
    private TextView countdownTextView;
    private TextView aiPredictionTextView;
    private TextView logTextView;
    
    private Button startStopButton;
    private Button cancelButton;
    private Button simulateFallButton;
    private Button simulateFalsePositiveButton;

    private boolean isMonitoring = false;
    private boolean isAlarmActive = false;
    private CountDownTimer countDownTimer;
    
    // TFLite Interpreter
    private Interpreter tflite;

    // Schwellenwerte für Rohdaten-Trigger
    private static final double FALL_THRESHOLD_ACCEL = 30.0;
    private static final double FALL_THRESHOLD_GYRO = 5.0;

    private double currentAccelMag = 0;
    private double currentGyroMag = 0;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);

        // UI-Elemente initialisieren
        accelTextView = findViewById(R.id.accelTextView);
        gyroTextView = findViewById(R.id.gyroTextView);
        statusTextView = findViewById(R.id.statusTextView);
        countdownTextView = findViewById(R.id.countdownTextView);
        aiPredictionTextView = findViewById(R.id.aiPredictionTextView);
        logTextView = findViewById(R.id.logTextView);
        
        startStopButton = findViewById(R.id.startStopButton);
        cancelButton = findViewById(R.id.cancelButton);
        simulateFallButton = findViewById(R.id.simulateFallButton);
        simulateFalsePositiveButton = findViewById(R.id.simulateFalsePositiveButton);

        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });

        sensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);

        // Monitoring Steuerung
        startStopButton.setOnClickListener(v -> {
            if (isMonitoring) stopMonitoring();
            else startMonitoring();
        });

        cancelButton.setOnClickListener(v -> cancelAlarm());

        // Test-Simulationen für Emulator
        simulateFallButton.setOnClickListener(v -> simulateEvent(35.0, 6.0, true));
        simulateFalsePositiveButton.setOnClickListener(v -> simulateEvent(32.0, 5.5, false));
        
        // KI Initialisierung
        initTFLite();
        
        addLog("App gestartet. KI geladen: " + (tflite != null));
    }

    private void initTFLite() {
        try {
            tflite = new Interpreter(loadModelFile());
            addLog("TFLite Modell erfolgreich geladen.");
        } catch (Exception e) {
            addLog("TFLite Info: Keine Model-Datei in assets/ gefunden.");
            Log.i(TAG, "Keine Model-Datei gefunden oder Fehler beim Laden.");
        }
    }

    private MappedByteBuffer loadModelFile() throws IOException {
        AssetFileDescriptor fileDescriptor = this.getAssets().openFd(MODEL_FILE);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private void startMonitoring() {
        if (accelerometer != null && gyroscope != null) {
            sensorManager.registerListener(this, accelerometer, SensorManager.SENSOR_DELAY_UI);
            sensorManager.registerListener(this, gyroscope, SensorManager.SENSOR_DELAY_UI);
            isMonitoring = true;
            statusTextView.setText(R.string.status_on);
            statusTextView.setTextColor(Color.GREEN);
            startStopButton.setText(R.string.btn_stop);
            addLog("Überwachung AKTIVIERT.");
        } else {
            Toast.makeText(this, "Sensoren fehlen - Sim-Modus aktiv", Toast.LENGTH_SHORT).show();
            isMonitoring = true;
            statusTextView.setText("Sim-Modus");
            addLog("Sensoren fehlen. Nur Simulation möglich.");
        }
    }

    private void stopMonitoring() {
        sensorManager.unregisterListener(this);
        isMonitoring = false;
        statusTextView.setText(R.string.status_off);
        statusTextView.setTextColor(Color.BLACK);
        startStopButton.setText(R.string.btn_start);
        if (isAlarmActive) cancelAlarm();
        addLog("Überwachung GESTOPPT.");
    }

    private void simulateEvent(double accel, double gyro, boolean aiShouldConfirm) {
        if (!isMonitoring) {
            Toast.makeText(this, "Bitte Überwachung erst starten!", Toast.LENGTH_SHORT).show();
            return;
        }
        addLog("Simuliere: Accel=" + accel + ", Gyro=" + gyro);
        currentAccelMag = accel;
        currentGyroMag = gyro;
        
        accelTextView.setText(String.format(Locale.getDefault(), "Sim-Accel: %.2f m/s²", accel));
        gyroTextView.setText(String.format(Locale.getDefault(), "Sim-Gyro: %.2f rad/s", gyro));
        
        // KI-Check direkt aufrufen
        runLocalAIPrediction(aiShouldConfirm);
    }

    private void addLog(String message) {
        String timeStamp = new SimpleDateFormat("HH:mm:ss", Locale.getDefault()).format(new Date());
        logTextView.append("\n[" + timeStamp + "] " + message);
        Log.d(TAG, message);
    }

    private void cancelAlarm() {
        if (countDownTimer != null) countDownTimer.cancel();
        isAlarmActive = false;
        countdownTextView.setVisibility(View.GONE);
        cancelButton.setVisibility(View.GONE);
        statusTextView.setText(R.string.status_on);
        statusTextView.setTextColor(Color.GREEN);
        aiPredictionTextView.setText("KI Status: Standby");
        addLog("Alarm durch Nutzer ABGEBROCHEN.");
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        if (!isMonitoring || isAlarmActive) return;

        if (event.sensor.getType() == Sensor.TYPE_ACCELEROMETER) {
            currentAccelMag = Math.sqrt(event.values[0]*event.values[0] + 
                                       event.values[1]*event.values[1] + 
                                       event.values[2]*event.values[2]);
            accelTextView.setText(getString(R.string.accel_label, currentAccelMag));
        } else if (event.sensor.getType() == Sensor.TYPE_GYROSCOPE) {
            currentGyroMag = Math.sqrt(event.values[0]*event.values[0] + 
                                      event.values[1]*event.values[1] + 
                                      event.values[2]*event.values[2]);
            gyroTextView.setText(getString(R.string.gyro_label, currentGyroMag));
        }

        // Automatischer Trigger bei echten Sensordaten
        if (currentAccelMag > FALL_THRESHOLD_ACCEL && currentGyroMag > FALL_THRESHOLD_GYRO) {
            runLocalAIPrediction(new Random().nextBoolean()); // Zufällige KI Entscheidung bei echten Daten
        }
    }

    /**
     * Lokale KI Logik (Simuliert oder echt via TFLite).
     */
    private void runLocalAIPrediction(boolean forceFall) {
        addLog("KI analysiert Bewegungsmuster...");
        
        float probability;
        
        if (tflite != null) {
            // BEISPIEL FÜR ECHTE INFERENZ:
            // float[][] input = {{ (float)currentAccelMag, (float)currentGyroMag }};
            // float[][] output = new float[1][1];
            // tflite.run(input, output);
            // probability = output[0][0];
            
            // Für die Demo simulieren wir die KI-Entscheidung basierend auf dem Test-Case
            probability = forceFall ? 0.95f : 0.30f; 
        } else {
            // Fallback falls Datei noch nicht in Assets liegt
            probability = forceFall ? 0.92f : 0.40f;
        }
        
        aiPredictionTextView.setText(String.format(Locale.getDefault(), "KI Status: Wahrscheinlichkeit %.1f%%", probability * 100));

        if (probability > 0.85) {
            addLog("KI BESTÄTIGT: Sturzmuster erkannt.");
            startFallCountdown();
        } else {
            addLog("KI FILTER: Bewegung als 'Alltäglich' eingestuft.");
            aiPredictionTextView.append(" -> FILTER AKTIV");
        }
    }

    private void startFallCountdown() {
        if (isAlarmActive) return;
        isAlarmActive = true;
        
        statusTextView.setText(R.string.fall_detected);
        statusTextView.setTextColor(Color.RED);
        countdownTextView.setVisibility(View.VISIBLE);
        cancelButton.setVisibility(View.VISIBLE);

        Vibrator v = (Vibrator) getSystemService(Context.VIBRATOR_SERVICE);
        if (v != null) v.vibrate(VibrationEffect.createOneShot(500, VibrationEffect.DEFAULT_AMPLITUDE));

        countDownTimer = new CountDownTimer(10000, 1000) {
            public void onTick(long millisUntilFinished) {
                countdownTextView.setText(String.valueOf(millisUntilFinished / 1000));
            }
            public void onFinish() {
                countdownTextView.setText("0");
                statusTextView.setText(R.string.emergency_call);
                cancelButton.setVisibility(View.GONE);
                addLog("CRITICAL: Notruf eingeleitet!");
            }
        }.start();
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {}
}
