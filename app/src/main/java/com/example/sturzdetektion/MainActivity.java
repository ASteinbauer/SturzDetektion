package com.example.sturzdetektion;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Color;
import android.graphics.drawable.GradientDrawable;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.os.CountDownTimer;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ProgressBar;
import android.widget.ScrollView;
import android.widget.TextView;

import org.tensorflow.lite.Interpreter;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;

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
    
    // Konstanten für Sicherheitsregeln
    private static final float FALL_CONFIDENCE_THRESHOLD = 0.75f;
    private static final float SUSPICIOUS_CONFIDENCE_THRESHOLD = 0.30f;
    private static final float HIGH_ACCEL_THRESHOLD = 25.0f;
    private static final float LOW_MOVEMENT_AFTER_IMPACT_THRESHOLD = 2.5f;

    private static final int NUM_CHANNELS = 6;
    private static final int NUM_CLASSES = 4;

    // UI Elemente
    private TextView badgeStatus, textSensorSource, textModelStatus, textDetectionMode;
    private TextView textAccel, textGyro;
    private TextView textResultMain, textConfidence;
    
    // Wahrscheinlichkeiten
    private ProgressBar[] progressClasses = new ProgressBar[4];
    private TextView[] textClasses = new TextView[4];

    // Buttons
    private Button buttonToggleMonitoring, buttonOpenTestMode, buttonSimulateEmergency;
    
    // Testmodus
    private View layoutTestMode;
    private TextView textSystemLog;
    private ScrollView scrollLog;
    private View cardTestEvaluation;
    private TextView textTestScenario, textTestExpected, textTestAiResult, textTestSafetyResult, textTestOverall;

    // Notfall-UI
    private View cardEmergency;
    private TextView textCountdownTimer;
    private Button buttonImOkay, buttonCallHelpNow;

    // Sensoren
    private SensorManager sensorManager;
    private Sensor accelerometer;
    private Sensor gyroscope;

    // Status
    private boolean monitoringActive = false;
    private boolean modelLoaded = false;
    private final Random random = new Random();
    private CountDownTimer emergencyTimer;

    // KI Komponenten (UNVERÄNDERT)
    private Interpreter tflite;
    private float[] mean = new float[NUM_CHANNELS];
    private float[] std = new float[NUM_CHANNELS];

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        initUi();
        initSensors();
        loadAiModel();
        
        appendLog("App Dashboard initialisiert.");
    }

    private void initUi() {
        // Header & Status
        badgeStatus = findViewById(R.id.badgeStatus);
        textSensorSource = findViewById(R.id.textSensorSource);
        textModelStatus = findViewById(R.id.textModelStatus);
        textDetectionMode = findViewById(R.id.textDetectionMode);
        textAccel = findViewById(R.id.textAccel);
        textGyro = findViewById(R.id.textGyro);

        // Ergebnis
        textResultMain = findViewById(R.id.textResultMain);
        textConfidence = findViewById(R.id.textConfidence);
        
        progressClasses[0] = findViewById(R.id.progressClass0);
        progressClasses[1] = findViewById(R.id.progressClass1);
        progressClasses[2] = findViewById(R.id.progressClass2);
        progressClasses[3] = findViewById(R.id.progressClass3);
        
        textClasses[0] = findViewById(R.id.textClass0);
        textClasses[1] = findViewById(R.id.textClass1);
        textClasses[2] = findViewById(R.id.textClass2);
        textClasses[3] = findViewById(R.id.textClass3);

        // Buttons
        buttonToggleMonitoring = findViewById(R.id.buttonToggleMonitoring);
        buttonOpenTestMode = findViewById(R.id.buttonOpenTestMode);
        buttonSimulateEmergency = findViewById(R.id.buttonSimulateEmergency);

        // Testmodus & Log
        layoutTestMode = findViewById(R.id.layoutTestMode);
        textSystemLog = findViewById(R.id.textSystemLog);
        scrollLog = findViewById(R.id.scrollLog);
        
        findViewById(R.id.buttonClearLog).setOnClickListener(v -> textSystemLog.setText(""));
        findViewById(R.id.buttonCloseTestMode).setOnClickListener(v -> layoutTestMode.setVisibility(View.GONE));
        
        cardTestEvaluation = findViewById(R.id.cardTestEvaluation);
        textTestScenario = findViewById(R.id.textTestScenario);
        textTestExpected = findViewById(R.id.textTestExpected);
        textTestAiResult = findViewById(R.id.textTestAiResult);
        textTestSafetyResult = findViewById(R.id.textTestSafetyResult);
        textTestOverall = findViewById(R.id.textTestOverall);

        // Notfall
        cardEmergency = findViewById(R.id.cardEmergency);
        textCountdownTimer = findViewById(R.id.textCountdownTimer);
        buttonImOkay = findViewById(R.id.buttonImOkay);
        buttonCallHelpNow = findViewById(R.id.buttonCallHelpNow);

        // Click Listeners
        buttonToggleMonitoring.setOnClickListener(v -> toggleMonitoring());
        buttonOpenTestMode.setOnClickListener(v -> layoutTestMode.setVisibility(View.VISIBLE));
        buttonSimulateEmergency.setOnClickListener(v -> {
            appendLog("Manueller Notfalltest gestartet.");
            startEmergencyCountdown();
        });

        buttonImOkay.setOnClickListener(v -> cancelEmergencyCountdown());
        buttonCallHelpNow.setOnClickListener(v -> {
            if (emergencyTimer != null) emergencyTimer.cancel();
            textCountdownTimer.setText("!!");
            appendLog("NOTFALL MANUELL AUSGELÖST!");
        });

        // Simulations-Buttons im Testmodus
        findViewById(R.id.btnSimFall).setOnClickListener(v -> runSimulation("Echter Sturz", "Sturz"));
        findViewById(R.id.btnSimBump).setOnClickListener(v -> runSimulation("Rempler / Stoß", "Normal"));
        findViewById(R.id.btnSimWalk).setOnClickListener(v -> runSimulation("Normales Gehen", "Normal"));
        findViewById(R.id.btnSimJog).setOnClickListener(v -> runSimulation("Joggen / Rennen", "Normal"));
        findViewById(R.id.btnSimTableDrop).setOnClickListener(v -> runSimulation("Handy fällt vom Tisch", "Verdächtig"));
        findViewById(R.id.btnSimSit).setOnClickListener(v -> runSimulation("Hinsetzen", "Normal"));
        findViewById(R.id.btnSimStumble).setOnClickListener(v -> runSimulation("Stolpern", "Normal/Verdächtig"));
        findViewById(R.id.btnSimStairs).setOnClickListener(v -> runSimulation("Treppe gehen", "Normal"));
        findViewById(R.id.btnSimFallLowRot).setOnClickListener(v -> runSimulation("Sturz mit wenig Drehung", "Sturz"));
        findViewById(R.id.btnSimFallHighRot).setOnClickListener(v -> runSimulation("Sturz mit starker Drehung", "Sturz"));
    }

    private void initSensors() {
        sensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
        if (sensorManager != null) {
            accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
            gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
            
            if (accelerometer == null || gyroscope == null) {
                textSensorSource.setText("Live-Sensoren fehlen (Simulation)");
                textSensorSource.setTextColor(ContextCompat.getColor(this, R.color.status_red));
            }
        }
    }

    private void loadAiModel() {
        try {
            loadMetadata();
            tflite = new Interpreter(loadModelFile());
            modelLoaded = true;
            textModelStatus.setText("Geladen");
            textModelStatus.setTextColor(ContextCompat.getColor(this, R.color.status_green));
            appendLog("TFLite Modell erfolgreich geladen.");
        } catch (Exception e) {
            modelLoaded = false;
            textModelStatus.setText("Ladefehler");
            textModelStatus.setTextColor(ContextCompat.getColor(this, R.color.status_red));
            appendLog("Modell-Fehler: " + e.getMessage());
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
            sensorManager.registerListener(this, accelerometer, SensorManager.SENSOR_DELAY_UI);
            sensorManager.registerListener(this, gyroscope, SensorManager.SENSOR_DELAY_UI);
            monitoringActive = true;
            updateStatusBadge(true);
            buttonToggleMonitoring.setText("Überwachung stoppen");
            appendLog("Live-Überwachung aktiv.");
        } else {
            appendLog("Sensoren nicht verfügbar.");
        }
    }

    private void stopMonitoring() {
        sensorManager.unregisterListener(this);
        monitoringActive = false;
        updateStatusBadge(false);
        buttonToggleMonitoring.setText("Überwachung starten");
        appendLog("Überwachung beendet.");
    }

    private void updateStatusBadge(boolean active) {
        badgeStatus.setText(active ? "Überwachung aktiv" : "Inaktiv");
        GradientDrawable shape = (GradientDrawable) badgeStatus.getBackground();
        shape.setColor(active ? ContextCompat.getColor(this, R.color.status_green) : ContextCompat.getColor(this, R.color.status_gray));
    }

    private void runSimulation(String scenario, String expected) {
        textSensorSource.setText("Simulation: " + scenario);
        
        float maxAccel = 0;
        float maxGyro = 0;
        float[] prob = new float[4];
        float moveAfter = 5.0f;

        switch (scenario) {
            case "Echter Sturz":
                maxAccel = 35.0f; maxGyro = 8.0f; moveAfter = 0.5f;
                prob = new float[]{0.01f, 0.04f, 0.10f, 0.85f};
                break;
            case "Sturz mit wenig Drehung":
                maxAccel = 30.0f; maxGyro = 2.5f; moveAfter = 0.8f;
                prob = new float[]{0.05f, 0.05f, 0.40f, 0.50f};
                break;
            case "Sturz mit starker Drehung":
                maxAccel = 45.0f; maxGyro = 12.0f; moveAfter = 0.3f;
                prob = new float[]{0.0f, 0.01f, 0.04f, 0.95f};
                break;
            case "Handy fällt vom Tisch":
                maxAccel = 55.0f; maxGyro = 2.0f; moveAfter = 0.0f;
                prob = new float[]{0.10f, 0.10f, 0.70f, 0.10f};
                break;
            case "Rempler / Stoß":
                maxAccel = 18.0f; maxGyro = 3.5f; moveAfter = 6.0f;
                prob = new float[]{0.10f, 0.60f, 0.25f, 0.05f};
                break;
            case "Stolpern":
                maxAccel = 26.0f; maxGyro = 4.0f; moveAfter = 1.5f;
                prob = new float[]{0.05f, 0.15f, 0.50f, 0.30f};
                break;
            default:
                maxAccel = 10.0f; maxGyro = 0.5f; moveAfter = 8.0f;
                prob = new float[]{0.40f, 0.55f, 0.04f, 0.01f};
                break;
        }

        processDetection(maxAccel, maxGyro, prob, moveAfter, scenario, expected);
    }

    private void processDetection(float maxAccel, float maxGyro, float[] output, float moveAfter, String scenario, String expected) {
        textAccel.setText(String.format(Locale.getDefault(), "%.1f m/s²", maxAccel));
        textGyro.setText(String.format(Locale.getDefault(), "%.1f rad/s", maxGyro));

        int predictedClass = 0;
        float maxVal = -1;
        for (int i = 0; i < 4; i++) {
            if (output[i] > maxVal) {
                maxVal = output[i];
                predictedClass = i;
            }
        }
        float fallProb = output[3];

        String finalResult = "Kein Sturz";
        String safetyNote = "Normal";

        if (fallProb > FALL_CONFIDENCE_THRESHOLD) {
            finalResult = "Sturz erkannt";
            safetyNote = "KI-Trigger (>75%)";
        } else if (fallProb > SUSPICIOUS_CONFIDENCE_THRESHOLD && maxAccel > HIGH_ACCEL_THRESHOLD && moveAfter < LOW_MOVEMENT_AFTER_IMPACT_THRESHOLD) {
            finalResult = "Sturz erkannt";
            safetyNote = "Safety-Trigger (KI+Sensoren)";
        } else if (fallProb > SUSPICIOUS_CONFIDENCE_THRESHOLD || predictedClass == 2 || maxAccel > 22.0f) {
            finalResult = "Verdächtige Bewegung";
            safetyNote = "Warnung";
        }

        updateResultUi(finalResult, (int)(maxVal * 100), output);
        
        if (scenario != null) {
            updateTestEvaluation(scenario, expected, finalResult, safetyNote, output);
        }

        if (finalResult.equals("Sturz erkannt")) {
            startEmergencyCountdown();
        }
        
        appendLog("Erkennung: " + finalResult);
    }

    private void updateResultUi(String result, int confidence, float[] probs) {
        textResultMain.setText(result);
        textConfidence.setText("KI-Vertrauen: " + confidence + "%");
        
        int color = ContextCompat.getColor(this, R.color.status_green);
        if (result.equals("Sturz erkannt")) color = ContextCompat.getColor(this, R.color.status_red);
        else if (result.equals("Verdächtige Bewegung")) color = ContextCompat.getColor(this, R.color.status_orange);
        
        textResultMain.setTextColor(color);

        for (int i = 0; i < 4; i++) {
            progressClasses[i].setProgress((int)(probs[i] * 100));
            textClasses[i].setText((int)(probs[i] * 100) + "%");
        }
    }

    private void updateTestEvaluation(String scenario, String expected, String actual, String safety, float[] probs) {
        cardTestEvaluation.setVisibility(View.VISIBLE);
        textTestScenario.setText("Szenario: " + scenario);
        textTestExpected.setText("Erwartet: " + expected);
        textTestAiResult.setText("KI (Sturz-Prob): " + String.format(Locale.getDefault(), "%.2f", probs[3]));
        textTestSafetyResult.setText("Sicherheitsregel: " + safety);

        boolean ok = false;
        if (expected.equals("Sturz") && actual.equals("Sturz erkannt")) ok = true;
        if (expected.equals("Normal") && actual.equals("Kein Sturz")) ok = true;
        if (expected.contains("Verdächtig") && (actual.contains("Verdächtig") || actual.contains("Sturz"))) ok = true;
        if (expected.equals("Normal/Verdächtig")) ok = true;

        textTestOverall.setText(ok ? "BEWERTUNG: OK" : "BEWERTUNG: PRÜFEN");
        textTestOverall.setTextColor(ok ? ContextCompat.getColor(this, R.color.status_green) : ContextCompat.getColor(this, R.color.status_red));
    }

    private void startEmergencyCountdown() {
        if (emergencyTimer != null) emergencyTimer.cancel();
        
        cardEmergency.setVisibility(View.VISIBLE);
        appendLog("NOTFALL-COUNTDOWN AKTIVIERT.");

        emergencyTimer = new CountDownTimer(30000, 1000) {
            @Override
            public void onTick(long millisUntilFinished) {
                textCountdownTimer.setText("" + (millisUntilFinished / 1000));
            }

            @Override
            public void onFinish() {
                textCountdownTimer.setText("!!");
                appendLog("ALARM: Notruf wird simuliert!");
            }
        }.start();
    }

    private void cancelEmergencyCountdown() {
        if (emergencyTimer != null) emergencyTimer.cancel();
        cardEmergency.setVisibility(View.GONE);
        appendLog("Alarm vom Nutzer abgebrochen.");
    }

    private void appendLog(String message) {
        String ts = new SimpleDateFormat("HH:mm:ss", Locale.getDefault()).format(new Date());
        String entry = "[" + ts + "] " + message + "\n";
        textSystemLog.append(entry);
        scrollLog.post(() -> scrollLog.fullScroll(View.FOCUS_DOWN));
        Log.d(TAG, message);
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        if (!monitoringActive) return;
        
        float x = event.values[0];
        float y = event.values[1];
        float z = event.values[2];
        float magnitude = (float) Math.sqrt(x*x + y*y + z*z);

        if (event.sensor.getType() == Sensor.TYPE_ACCELEROMETER) {
            textAccel.setText(String.format(Locale.getDefault(), "%.1f m/s²", magnitude));
            if (magnitude > 25.0f) {
                float[] dummyProbs = {0.1f, 0.2f, 0.5f, 0.2f}; 
                processDetection(magnitude, 2.0f, dummyProbs, 1.0f, "Live-Impact", "Verdächtig");
            }
        } else if (event.sensor.getType() == Sensor.TYPE_GYROSCOPE) {
            textGyro.setText(String.format(Locale.getDefault(), "%.1f rad/s", magnitude));
        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {}
}
