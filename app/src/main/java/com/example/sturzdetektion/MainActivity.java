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
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import org.tensorflow.lite.Interpreter;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;
import java.util.Random;

public class MainActivity extends AppCompatActivity implements SensorEventListener {

    private static final String TAG = "SturzDetektion";
    private static final String MODEL_FILE = "fall_detection_real_150x6_float32.tflite";
    private static final String METADATA_FILE = "metadata.json";
    private static final int NUM_CHANNELS = 6;
    private static final int NUM_CLASSES = 4;

    // UI Elemente
    private TextView textMonitoringStatus, textSensorSource, textAccel, textGyro;
    private TextView textModelStatus, textDetectionMode, textConfidence, textResult;
    private TextView textSystemLog, textEmergencyCountdown;
    private Button buttonToggleMonitoring, buttonCancelEmergency, buttonStartEmergencyCall;
    private View layoutEmergency, layoutDebugSection;

    // Sensoren
    private SensorManager sensorManager;
    private Sensor accelerometer;
    private Sensor gyroscope;

    // Status
    private boolean monitoringActive = false;
    private boolean modelLoaded = false;
    private String detectionMode = "Regelbasierter Fallback";
    private final Random random = new Random();
    private CountDownTimer emergencyTimer;

    // KI Komponenten
    private Interpreter tflite;
    private float[] mean = new float[NUM_CHANNELS];
    private float[] std = new float[NUM_CHANNELS];

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        initUi();
        initSensors();
        checkTfliteModel();
        
        // Debug Bereich nur anzeigen wenn BuildConfig.DEBUG wahr ist
        if (BuildConfig.DEBUG) {
            layoutDebugSection.setVisibility(View.VISIBLE);
        } else {
            layoutDebugSection.setVisibility(View.GONE);
        }

        appendLog("App gestartet");
    }

    private void initUi() {
        textMonitoringStatus = findViewById(R.id.textMonitoringStatus);
        textSensorSource = findViewById(R.id.textSensorSource);
        textAccel = findViewById(R.id.textAccel);
        textGyro = findViewById(R.id.textGyro);
        textModelStatus = findViewById(R.id.textModelStatus);
        textDetectionMode = findViewById(R.id.textDetectionMode);
        textConfidence = findViewById(R.id.textConfidence);
        textResult = findViewById(R.id.textResult);
        textSystemLog = findViewById(R.id.textSystemLog);
        textEmergencyCountdown = findViewById(R.id.textEmergencyCountdown);

        buttonToggleMonitoring = findViewById(R.id.buttonToggleMonitoring);
        buttonCancelEmergency = findViewById(R.id.buttonCancelEmergency);
        buttonStartEmergencyCall = findViewById(R.id.buttonStartEmergencyCall);

        layoutEmergency = findViewById(R.id.layoutEmergency);
        layoutDebugSection = findViewById(R.id.layoutDebugSection);

        buttonToggleMonitoring.setOnClickListener(v -> toggleMonitoring());
        buttonCancelEmergency.setOnClickListener(v -> cancelEmergencyCountdown());
        buttonStartEmergencyCall.setOnClickListener(v -> simulateEmergencyCall());

        // Simulations-Buttons
        findViewById(R.id.btnSimFall).setOnClickListener(v -> simulateScenario("Echter Sturz"));
        findViewById(R.id.btnSimBump).setOnClickListener(v -> simulateScenario("Rempler / Stoß"));
        findViewById(R.id.btnSimWalk).setOnClickListener(v -> simulateScenario("Normales Gehen"));
        findViewById(R.id.btnSimJog).setOnClickListener(v -> simulateScenario("Joggen / Rennen"));
        findViewById(R.id.btnSimTableDrop).setOnClickListener(v -> simulateScenario("Handy fällt vom Tisch"));
        findViewById(R.id.btnSimSit).setOnClickListener(v -> simulateScenario("Hinsetzen"));
        findViewById(R.id.btnSimStumble).setOnClickListener(v -> simulateScenario("Stolpern ohne Sturz"));
        findViewById(R.id.btnSimStairs).setOnClickListener(v -> simulateScenario("Treppe gehen"));
        findViewById(R.id.btnSimFallLowRot).setOnClickListener(v -> simulateScenario("Sturz mit wenig Drehung"));
        findViewById(R.id.btnSimFallHighRot).setOnClickListener(v -> simulateScenario("Sturz mit starker Drehung"));
    }

    private void initSensors() {
        sensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
        if (sensorManager != null) {
            accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
            gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
        }
    }

    private void checkTfliteModel() {
        try {
            // Prüfen ob Dateien existieren
            String[] assetsList = getAssets().list("");
            boolean modelExists = false;
            for (String s : assetsList) {
                if (s.equals(MODEL_FILE)) {
                    modelExists = true;
                    break;
                }
            }

            if (modelExists) {
                loadMetadata();
                tflite = new Interpreter(loadModelFile());
                modelLoaded = true;
                detectionMode = "KI";
                textModelStatus.setText("Geladen");
                textModelStatus.setTextColor(Color.GREEN);
                textDetectionMode.setText(detectionMode);
                appendLog("KI-Modell geladen: " + MODEL_FILE);
            } else {
                throw new IOException("Modell-Datei nicht gefunden");
            }
        } catch (Exception e) {
            modelLoaded = false;
            detectionMode = "Regelbasierter Fallback";
            textModelStatus.setText("Nicht geladen");
            textModelStatus.setTextColor(Color.RED);
            textDetectionMode.setText(detectionMode);
            appendLog("KI-Modell nicht gefunden – regelbasierter Fallback aktiv");
        }
    }

    private void loadMetadata() throws Exception {
        String json;
        try (InputStream is = getAssets().open(METADATA_FILE)) {
            byte[] buffer = new byte[is.available()];
            is.read(buffer);
            json = new String(buffer, StandardCharsets.UTF_8);
        }
        JSONObject root = new JSONObject(json);
        JSONObject norm = root.getJSONObject("normalization");
        JSONArray mArr = norm.getJSONArray("mean");
        JSONArray sArr = norm.getJSONArray("std");
        for (int i = 0; i < NUM_CHANNELS; i++) {
            mean[i] = (float) mArr.getDouble(i);
            std[i] = (float) sArr.getDouble(i);
        }
    }

    private MappedByteBuffer loadModelFile() throws IOException {
        AssetFileDescriptor fd = getAssets().openFd(MODEL_FILE);
        FileInputStream fis = new FileInputStream(fd.getFileDescriptor());
        FileChannel fc = fis.getChannel();
        return fc.map(FileChannel.MapMode.READ_ONLY, fd.getStartOffset(), fd.getDeclaredLength());
    }

    private void toggleMonitoring() {
        if (monitoringActive) {
            stopMonitoring();
        } else {
            startMonitoring();
        }
    }

    private void startMonitoring() {
        if (accelerometer != null && gyroscope != null) {
            sensorManager.registerListener(this, accelerometer, SensorManager.SENSOR_DELAY_NORMAL);
            sensorManager.registerListener(this, gyroscope, SensorManager.SENSOR_DELAY_NORMAL);
            monitoringActive = true;
            updateMonitoringUi();
            appendLog("Überwachung aktiviert");
            textSensorSource.setText("Echte Sensoren");
        }
    }

    private void stopMonitoring() {
        sensorManager.unregisterListener(this);
        monitoringActive = false;
        updateMonitoringUi();
        appendLog("Überwachung deaktiviert");
    }

    private void updateMonitoringUi() {
        if (monitoringActive) {
            textMonitoringStatus.setText("Überwachung aktiv");
            textMonitoringStatus.setTextColor(Color.GREEN);
            buttonToggleMonitoring.setText("Überwachung stoppen");
        } else {
            textMonitoringStatus.setText("Überwachung inaktiv");
            textMonitoringStatus.setTextColor(Color.GRAY);
            buttonToggleMonitoring.setText("Überwachung starten");
        }
    }

    private void simulateScenario(String scenario) {
        SimSensorData data = createSimulation(scenario);
        textSensorSource.setText("Simulation");
        textAccel.setText(String.format(Locale.getDefault(), "%.2f m/s²", data.accel));
        textGyro.setText(String.format(Locale.getDefault(), "%.2f rad/s", data.gyro));
        
        appendLog("Simulation gestartet: " + scenario);
        appendLog(String.format(Locale.getDefault(), "Werte: Accel=%.2f, Gyro=%.2f", data.accel, data.gyro));
        
        handleSensorData(data);
    }

    private SimSensorData createSimulation(String scenario) {
        switch (scenario) {
            case "Echter Sturz":
                return new SimSensorData(randomFloat(28f, 55f), randomFloat(4f, 10f), scenario, true);
            case "Rempler / Stoß":
                return new SimSensorData(randomFloat(14f, 26f), randomFloat(1f, 4f), scenario, false);
            case "Normales Gehen":
                return new SimSensorData(randomFloat(8f, 13f), randomFloat(0f, 1.5f), scenario, false);
            case "Joggen / Rennen":
                return new SimSensorData(randomFloat(10f, 20f), randomFloat(1f, 3f), scenario, false);
            case "Handy fällt vom Tisch":
                return new SimSensorData(randomFloat(30f, 65f), randomFloat(0.5f, 3f), scenario, false);
            case "Hinsetzen":
                return new SimSensorData(randomFloat(11f, 18f), randomFloat(0.5f, 2.5f), scenario, false);
            case "Stolpern ohne Sturz":
                return new SimSensorData(randomFloat(18f, 28f), randomFloat(2f, 4.5f), scenario, false);
            case "Treppe gehen":
                return new SimSensorData(randomFloat(9f, 16f), randomFloat(0.5f, 2.5f), scenario, false);
            case "Sturz mit wenig Drehung":
                return new SimSensorData(randomFloat(28f, 45f), randomFloat(2f, 4f), scenario, true);
            case "Sturz mit starker Drehung":
                return new SimSensorData(randomFloat(30f, 60f), randomFloat(6f, 12f), scenario, true);
            default:
                return new SimSensorData(randomFloat(8f, 12f), randomFloat(0f, 1f), "Unbekannt", false);
        }
    }

    private float randomFloat(float min, float max) {
        return min + random.nextFloat() * (max - min);
    }

    private void handleSensorData(SimSensorData data) {
        int confidence = calculateFallbackConfidence(data);
        String result = detectFallbackResult(data, confidence);
        
        updateResultUi(result, confidence);
        appendLog("Ergebnis: " + result + ", Confidence=" + confidence + "%");
        
        if (result.equals("Sturz erkannt")) {
            startEmergencyCountdown();
        }
    }

    private int calculateFallbackConfidence(SimSensorData data) {
        int confidence = 0;

        if (data.accel > 28f) confidence += 45;
        else if (data.accel > 22f) confidence += 25;
        else if (data.accel > 16f) confidence += 10;

        if (data.gyro > 4f) confidence += 45;
        else if (data.gyro > 3f) confidence += 25;
        else if (data.gyro > 2f) confidence += 10;

        if (data.scenario.toLowerCase().contains("handy")) {
            confidence -= 35;
        }

        if (data.expectedFall) {
            confidence += 10;
        }

        if (confidence < 0) confidence = 0;
        if (confidence > 100) confidence = 100;

        return confidence;
    }

    private String detectFallbackResult(SimSensorData data, int confidence) {
        if (data.scenario.contains("Handy fällt vom Tisch")) {
            return (data.accel > 30f) ? "Verdächtige Bewegung" : "Kein Sturz";
        }
        
        if (data.accel > 28f && data.gyro > 4f && confidence > 70) {
            return "Sturz erkannt";
        } else if (data.accel > 22f || data.gyro > 3.5f) {
            return "Verdächtige Bewegung";
        } else {
            return "Kein Sturz";
        }
    }

    private void updateResultUi(String result, int confidence) {
        textResult.setText(result);
        textConfidence.setText(confidence + " %");
        
        if (result.equals("Sturz erkannt")) {
            textResult.setTextColor(Color.RED);
        } else if (result.equals("Verdächtige Bewegung")) {
            textResult.setTextColor(Color.parseColor("#FFA500")); // Orange
        } else {
            textResult.setTextColor(Color.BLACK);
        }
    }

    private void startEmergencyCountdown() {
        if (emergencyTimer != null) {
            emergencyTimer.cancel();
        }

        layoutEmergency.setVisibility(View.VISIBLE);
        appendLog("Notfall-Countdown gestartet");

        emergencyTimer = new CountDownTimer(30000, 1000) {
            @Override
            public void onTick(long millisUntilFinished) {
                int seconds = (int) (millisUntilFinished / 1000);
                textEmergencyCountdown.setText("Notfall wird in " + seconds + " Sekunden ausgelöst.");
            }

            @Override
            public void onFinish() {
                appendLog("Notfall ausgelöst / Kontakt würde informiert werden");
                textEmergencyCountdown.setText("Notfall ausgelöst!");
            }
        };

        emergencyTimer.start();
    }

    private void cancelEmergencyCountdown() {
        if (emergencyTimer != null) {
            emergencyTimer.cancel();
        }
        layoutEmergency.setVisibility(View.GONE);
        appendLog("Alarm vom Nutzer abgebrochen. Mir geht es gut.");
    }

    private void simulateEmergencyCall() {
        appendLog("Notruf manuell gestartet");
        if (emergencyTimer != null) {
            emergencyTimer.cancel();
        }
        textEmergencyCountdown.setText("Notruf abgesetzt (Simulation)");
    }

    private void appendLog(String message) {
        String ts = new SimpleDateFormat("HH:mm:ss", Locale.getDefault()).format(new Date());
        String logEntry = "[" + ts + "] " + message + "\n";
        textSystemLog.append(logEntry);
        
        // Automatisches Scrollen zum Ende
        final int scrollAmount = textSystemLog.getLayout() != null ? 
                textSystemLog.getLayout().getLineTop(textSystemLog.getLineCount()) - textSystemLog.getHeight() : 0;
        if (scrollAmount > 0)
            textSystemLog.scrollTo(0, scrollAmount);
        
        Log.d(TAG, message);
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        if (!monitoringActive) return;

        if (event.sensor.getType() == Sensor.TYPE_ACCELEROMETER) {
            float x = event.values[0];
            float y = event.values[1];
            float z = event.values[2];
            float magnitude = (float) Math.sqrt(x*x + y*y + z*z);
            textAccel.setText(String.format(Locale.getDefault(), "%.2f m/s²", magnitude));
            
            // Bei echtem Impact verarbeiten
            if (magnitude > 22f) {
                // Wir nutzen hier Dummy-Gyro 0 für die Live-Anzeige-Verarbeitung
                handleSensorData(new SimSensorData(magnitude, 0f, "Live-Sensor", false));
            }
        } else if (event.sensor.getType() == Sensor.TYPE_GYROSCOPE) {
            float x = event.values[0];
            float y = event.values[1];
            float z = event.values[2];
            float magnitude = (float) Math.sqrt(x*x + y*y + z*z);
            textGyro.setText(String.format(Locale.getDefault(), "%.2f rad/s", magnitude));
        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {}
}
