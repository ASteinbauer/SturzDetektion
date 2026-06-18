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
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.Random;

public class MainActivity extends AppCompatActivity implements SensorEventListener {

    private static final String TAG = "SturzDetektion";
    private static final String MODEL_FILE = "fall_detection_real_150x6_float32.tflite";
    private static final String METADATA_FILE = "metadata.json";
    
    // Konstanten für Sicherheitsregeln (Feingetunt für echten Einsatz)
    private static final float FALL_CONFIDENCE_THRESHOLD = 0.88f;      // Leicht gesenkt für bessere Erkennung
    private static final float SUSPICIOUS_CONFIDENCE_THRESHOLD = 0.40f; // Etwas sensibler für Vor-Warnung
    private static final float HIGH_ACCEL_THRESHOLD = 34.0f;           // Höherer Puffer gegen Joggen/Rempler
    private static final float LOW_MOVEMENT_AFTER_IMPACT_THRESHOLD = 2.0f;

    private static final float CLASS2_STRICT_THRESHOLD = 0.80f;        // Höhere Hürde für Klasse 2 Warnungen
    private static final float CLASS2_WARNING_ACCEL_THRESHOLD = 28.0f; 
    private static final int REQUIRED_FALL_WINDOWS = 2;

    private int consecutiveFallWindows = 0;

    private static final int NUM_CHANNELS = 7; // Erhöht auf 7 Kanäle
    private static final int NUM_CLASSES = 4;
    private static final int WINDOW_SIZE = 150;
    private static final long INFERENCE_INTERVAL_MS = 400;

    // Ringbuffer & Sensor-Daten
    private float[][] sensorWindow = new float[WINDOW_SIZE][NUM_CHANNELS];
    private int windowIndex = 0;
    private int samplesInWindow = 0;
    private long lastInferenceTime = 0;
    private long lastSampleTime = 0; // Neu: Für fixes 50Hz Sampling
    private static final long SAMPLE_PERIOD_MS = 20; // 50Hz Zielrate
    private float[] latestAccel = new float[3];
    private float[] latestGyro = new float[3];
    private boolean accelReceived = false;
    private boolean gyroReceived = false;

    // UI Elemente
    private TextView badgeStatus, textSensorSource, textModelStatus, textDetectionMode;
    private TextView textAccel, textGyro, textPressure;
    private TextView textResultMain, textConfidence, textStatusIcon, textStatusDescription;
    private com.google.android.material.card.MaterialCardView cardResult;
    
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
    private Sensor pressureSensor; // Neu: Barometer

    // Veto-Variablen
    private float startPressure = -1;
    private float lastPressure = -1;
    private float initialTiltAngle = -1;
    private boolean hasBarometer = false;

    // Status
    private boolean monitoringActive = false;
    private boolean modelLoaded = false;
    private final Random random = new Random();
    private CountDownTimer emergencyTimer;

    // Echte Testdaten (Replay-Modus)
    private List<JSONObject> testSequences = null;
    private boolean isReplayMode = false;
    private float[] lastInferenceProbs = new float[4];
    private String lastInferenceResult = "Kein Sturz";
    private String lastInferenceSafetyNote = "";

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
        textPressure = findViewById(R.id.textPressure);

        // Ergebnis
        cardResult = findViewById(R.id.cardResult);
        textResultMain = findViewById(R.id.textResultMain);
        textConfidence = findViewById(R.id.textConfidence);
        textStatusIcon = findViewById(R.id.textStatusIcon);
        textStatusDescription = findViewById(R.id.textStatusDescription);
        
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
        buttonOpenTestMode.setOnClickListener(v -> {
            layoutTestMode.setVisibility(View.VISIBLE);
            loadRealTestSequences();
        });
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

        // Echte Testdaten (Replay)
        findViewById(R.id.btnRealRuhig).setOnClickListener(v ->
                replayRealSequence(findFirstSeqOfClass(0, "upfall")));
        findViewById(R.id.btnRealGehen).setOnClickListener(v ->
                replayRealSequence(findFirstSeqOfClass(1, "sisfall")));
        findViewById(R.id.btnRealFallAehnlich).setOnClickListener(v ->
                replayRealSequence(findFirstSeqOfClass(2, "sisfall")));
        findViewById(R.id.btnRealSturz).setOnClickListener(v ->
                replayRealSequence(findFirstSeqOfClass(3, "sisfall")));

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
            pressureSensor = sensorManager.getDefaultSensor(Sensor.TYPE_PRESSURE);
            
            if (accelerometer == null || gyroscope == null) {
                textSensorSource.setText("Live-Sensoren fehlen (Simulation)");
                textSensorSource.setTextColor(ContextCompat.getColor(this, R.color.status_red));
            }

            if (pressureSensor != null) {
                hasBarometer = true;
                appendLog("Barometer erkannt (Höhen-Check aktiv).");
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
            sensorManager.registerListener(this, accelerometer, SensorManager.SENSOR_DELAY_GAME);
            sensorManager.registerListener(this, gyroscope, SensorManager.SENSOR_DELAY_GAME);
            if (hasBarometer) {
                sensorManager.registerListener(this, pressureSensor, SensorManager.SENSOR_DELAY_NORMAL);
            }
            
            // Buffer zurücksetzen
            windowIndex = 0;
            samplesInWindow = 0;
            accelReceived = false;
            gyroReceived = false;
            lastInferenceTime = 0;
            consecutiveFallWindows = 0;
            startPressure = -1;
            initialTiltAngle = -1;
            
            monitoringActive = true;
            updateStatusBadge(true);
            buttonToggleMonitoring.setText("Überwachung stoppen");
            appendLog("Live-Überwachung aktiv (50Hz).");
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

    // -----------------------------------------------------------------------
    // Echte Testdaten (Replay aus SisFall/UP-Fall/UniMiB)
    // -----------------------------------------------------------------------

    private void loadRealTestSequences() {
        if (testSequences != null) return;
        testSequences = new ArrayList<>();
        try (InputStream is = getAssets().open("test_sequences.json")) {
            byte[] buffer = new byte[is.available()];
            is.read(buffer);
            JSONObject root = new JSONObject(new String(buffer, StandardCharsets.UTF_8));
            JSONArray arr = root.getJSONArray("sequences");
            for (int i = 0; i < arr.length(); i++) {
                testSequences.add(arr.getJSONObject(i));
            }
            appendLog("Echte Testdaten geladen: " + testSequences.size() + " Sequenzen (SisFall/UP-Fall/UniMiB).");
        } catch (Exception e) {
            appendLog("Testdaten-Fehler: " + e.getMessage());
        }
    }

    private JSONObject findFirstSeqOfClass(int cls, String preferDataset) {
        if (testSequences == null) return null;
        // Bevorzugtes Dataset zuerst
        for (JSONObject seq : testSequences) {
            try {
                if (seq.getInt("class") == cls && seq.getString("dataset").equals(preferDataset))
                    return seq;
            } catch (Exception ignored) {}
        }
        // Fallback: beliebiges Dataset
        for (JSONObject seq : testSequences) {
            try {
                if (seq.getInt("class") == cls) return seq;
            } catch (Exception ignored) {}
        }
        return null;
    }

    private void replayRealSequence(JSONObject seq) {
        if (seq == null) { appendLog("Keine Testsequenz gefunden."); return; }
        if (!modelLoaded) { appendLog("Modell nicht geladen."); return; }
        try {
            String label    = seq.getString("label");
            String expected = seq.getString("expected");
            String dataset  = seq.getString("dataset");
            double maxAcc   = seq.optDouble("max_accel_ms2", 0);
            JSONArray values = seq.getJSONArray("values");

            // Sensorwindow mit echten Rohdaten füllen (m/s² / rad/s)
            for (int i = 0; i < WINDOW_SIZE && i < values.length(); i++) {
                JSONArray row = values.getJSONArray(i);
                sensorWindow[i][0] = (float) row.getDouble(0);
                sensorWindow[i][1] = (float) row.getDouble(1);
                sensorWindow[i][2] = (float) row.getDouble(2);
                sensorWindow[i][3] = (float) row.getDouble(3);
                sensorWindow[i][4] = (float) row.getDouble(4);
                sensorWindow[i][5] = (float) row.getDouble(5);
            }
            windowIndex = 0;
            samplesInWindow = WINDOW_SIZE;
            consecutiveFallWindows = 0;

            textSensorSource.setText("Replay: " + label + " [" + dataset + "]");
            appendLog(String.format(Locale.getDefault(),
                    "--- Replay: %s [%s] | max_acc=%.1f m/s² ---", label, dataset, maxAcc));

            isReplayMode = true;
            performLiveInference();
            isReplayMode = false;

            // Auswertung mit gespeichertem KI-Ergebnis anzeigen
            updateTestEvaluation(label + " [" + dataset + "]", expected,
                    lastInferenceResult, lastInferenceSafetyNote, lastInferenceProbs);

        } catch (Exception e) {
            appendLog("Replay-Fehler: " + e.getMessage());
        }
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

        // --- PROF-FILTER (VETO LOGIK) ---
        
        // 1. Winkel-Check: Hat sich die Orientierung geändert?
        boolean orientationChanged = true; // Default true falls kein Initialwert
        if (initialTiltAngle != -1) {
            float currentTilt = calculateCurrentTilt();
            float diff = Math.abs(currentTilt - initialTiltAngle);
            orientationChanged = (diff > 15.0f); // Mindestens 15 Grad Änderung
        }

        // 2. Barometer-Check: Gab es einen Höhenunterschied? (Nur wenn Hardware vorhanden)
        boolean heightConfirmed = true; 
        if (hasBarometer && startPressure != -1 && lastPressure != -1) {
            float pressureDiff = lastPressure - startPressure; 
            // Ein Sturz (1m) erhöht den Druck um ca. 0.12 hPa
            heightConfirmed = (pressureDiff > 0.05f); 
        }

        // 3. Inaktivitäts-Check (moveAfter)
        boolean isLyingStill = (moveAfter < LOW_MOVEMENT_AFTER_IMPACT_THRESHOLD);

        // Kombinierte Entscheidung
        boolean confirmedFallCandidate = (predictedClass == 3 && fallProb > FALL_CONFIDENCE_THRESHOLD);
        
        // Veto-System: KI sagt zwar Sturz, aber die Physik passt nicht
        boolean vetoTriggered = false;
        String vetoReason = "";
        if (confirmedFallCandidate) {
            if (!orientationChanged) { vetoTriggered = true; vetoReason = "Keine Lageänderung"; }
            else if (!heightConfirmed) { vetoTriggered = true; vetoReason = "Kein Höhenunterschied"; }
            else if (!isLyingStill) { vetoTriggered = true; vetoReason = "Bewegung nach Aufprall"; }
        }

        if (confirmedFallCandidate && !vetoTriggered) {
            consecutiveFallWindows++;
        } else {
            consecutiveFallWindows = 0;
        }

        String finalResult = "Kein Sturz";
        String safetyNote = vetoTriggered ? "Veto: " + vetoReason : "Normal";

        if (consecutiveFallWindows >= REQUIRED_FALL_WINDOWS) {
            finalResult = "Sturz erkannt";
            safetyNote = "Bestätigt durch Physik-Check";
        } else if (confirmedFallCandidate && vetoTriggered) {
            finalResult = "Auffällige Bewegung";
            safetyNote = "KI-Verdacht, aber " + vetoReason;
        } else if (fallProb > SUSPICIOUS_CONFIDENCE_THRESHOLD) {
            finalResult = "Verdächtige Bewegung";
            safetyNote = "KI beobachtet...";
        }

        updateResultUi(finalResult, (int)(maxVal * 100), output);
        
        if (scenario != null) {
            updateTestEvaluation(scenario, expected, finalResult, safetyNote, output);
        }

        if (finalResult.equals("Sturz erkannt")) {
            startEmergencyCountdown();
        }
        
        appendLog(String.format(Locale.getDefault(), "Check: Res=%s, Veto=%s, TiltOK=%b, PressOK=%b", 
                finalResult, vetoReason, orientationChanged, heightConfirmed));
    }

    private float calculateCurrentTilt() {
        // Berechnet den Neigungswinkel der Z-Achse zur Schwerkraft
        float ax = latestAccel[0];
        float ay = latestAccel[1];
        float az = latestAccel[2];
        float magnitude = (float) Math.sqrt(ax*ax + ay*ay + az*az);
        if (magnitude < 0.1f) return 0;
        return (float) Math.toDegrees(Math.acos(az / magnitude));
    }

    private void updateResultUi(String result, int confidence, float[] probs) {
        textResultMain.setText(result);
        textConfidence.setText("KI-Vertrauen: " + confidence + "%");
        
        int color = ContextCompat.getColor(this, R.color.status_green);
        int bgColor = ContextCompat.getColor(this, R.color.status_green_light);
        int cardBg = ContextCompat.getColor(this, R.color.white);
        String icon = "✓";
        String description = "Die App überwacht Ihre Bewegungen im Hintergrund.";

        if (result.equals("Sturz erkannt")) {
            color = ContextCompat.getColor(this, R.color.status_red);
            bgColor = ContextCompat.getColor(this, R.color.status_red_light);
            cardBg = ContextCompat.getColor(this, R.color.status_red_light);
            icon = "!!";
            description = "Ein Sturz wurde erkannt! Bitte reagieren Sie auf den Alarm.";
        } else if (result.equals("Verdächtige Bewegung")) {
            color = ContextCompat.getColor(this, R.color.status_orange);
            bgColor = ContextCompat.getColor(this, R.color.status_orange_light);
            cardBg = ContextCompat.getColor(this, R.color.white);
            icon = "!";
            description = "Eine ungewöhnliche Bewegung wurde bemerkt. Wir beobachten weiter.";
        } else if (result.equals("Auffällige Bewegung – kein Sturz")) {
            color = ContextCompat.getColor(this, R.color.status_orange);
            bgColor = ContextCompat.getColor(this, R.color.status_orange_light);
            cardBg = ContextCompat.getColor(this, R.color.status_orange_light);
            icon = "!";
            description = "Die Bewegung war sturzähnlich, aber die Sicherheitsprüfung sieht keinen Notfall.";
        }
        
        textResultMain.setTextColor(color);
        textStatusIcon.setText(icon);
        textStatusIcon.setTextColor(color);
        textStatusDescription.setText(description);
        cardResult.setCardBackgroundColor(cardBg);
        
        // Hintergrund der Icon-Pill anpassen
        GradientDrawable iconShape = (GradientDrawable) textStatusIcon.getBackground();
        if (iconShape != null) iconShape.setColor(bgColor);

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

    private void performLiveInference() {
        if (!modelLoaded || tflite == null) return;

        float[][][] input = new float[1][WINDOW_SIZE][NUM_CHANNELS];
        float maxAccel = 0;
        float totalMovementInWindow = 0;
        float minAccelMagnitude = 100f; // Für Free-Fall Check
        float preImpactActivity = 0;    // Aktivität am Anfang des Fensters
        float preGyroActivity = 0;      // Gyro-Aktivität vor dem Ereignis
        float maxGyroMag = 0;           // Maximale Gyro-Stärke im Fenster

        // Fenster chronologisch aufbauen
        for (int i = 0; i < WINDOW_SIZE; i++) {
            int actualIndex = (windowIndex + i) % WINDOW_SIZE;
            
            float ax = sensorWindow[actualIndex][0];
            float ay = sensorWindow[actualIndex][1];
            float az = sensorWindow[actualIndex][2];
            float gx = sensorWindow[actualIndex][3];
            float gy = sensorWindow[actualIndex][4];
            float gz = sensorWindow[actualIndex][5];
            
            // Berechne den 7. Kanal (Gyro Magnitude) live für die KI
            float gMag = (float) Math.sqrt(gx*gx + gy*gy + gz*gz);

            float aMag = (float) Math.sqrt(ax * ax + ay * ay + az * az);
            if (aMag > maxAccel) maxAccel = aMag;
            if (aMag < minAccelMagnitude) minAccelMagnitude = aMag;
            if (gMag > maxGyroMag) maxGyroMag = gMag;

            // Abweichung von der Erdschwerkraft summieren (Bewegungsenergie)
            totalMovementInWindow += Math.abs(aMag - 9.81f);

            // Die ersten 40 Samples (ca. 0.8s) als "Vorgeschichte" prüfen
            if (i < 40) {
                preImpactActivity += Math.abs(aMag - 9.81f);
                preGyroActivity += gMag;
            }

            input[0][i][0] = (ax - mean[0]) / std[0];
            input[0][i][1] = (ay - mean[1]) / std[1];
            input[0][i][2] = (az - mean[2]) / std[2];
            input[0][i][3] = (gx - mean[3]) / std[3];
            input[0][i][4] = (gy - mean[4]) / std[4];
            input[0][i][5] = (gz - mean[5]) / std[5];
            input[0][i][6] = (gMag - mean[6]) / std[6]; // 7. Kanal normalisieren
        }

        // Durchschnittliche Bewegung im Fenster (Schüttel-Metrik)
        float avgMovement = totalMovementInWindow / WINDOW_SIZE;
        float avgPreActivity = preImpactActivity / 40f;
        float avgPreGyroActivity = preGyroActivity / 40f;

        // moveAfter: Nur die allerletzten 0.5 Sekunden prüfen (Liegt er danach still?)
        float moveAfter = 0;
        int lastSamples = 25; 
        for (int i = WINDOW_SIZE - lastSamples; i < WINDOW_SIZE; i++) {
            int actualIndex = (windowIndex + i) % WINDOW_SIZE;
            float ax = sensorWindow[actualIndex][0];
            float ay = sensorWindow[actualIndex][1];
            float az = sensorWindow[actualIndex][2];
            moveAfter += Math.abs((float)Math.sqrt(ax*ax + ay*ay + az*az) - 9.81f);
        }
        moveAfter /= lastSamples;

        // Orientierung am Anfang vs Ende des Fensters
        float tiltStart = calculateTiltAtIndex(windowIndex);
        float tiltEnd = calculateTiltAtIndex((windowIndex + WINDOW_SIZE - 1) % WINDOW_SIZE);
        float tiltDiff = Math.abs(tiltEnd - tiltStart);

        // TFLite Inferenz
        float[][] output = new float[1][NUM_CLASSES];
        tflite.run(input, output);
        
        processAdvancedDetection(maxAccel, avgMovement, avgPreActivity, minAccelMagnitude, moveAfter, tiltDiff, avgPreGyroActivity, maxGyroMag, output[0]);
    }

    private float calculateTiltAtIndex(int index) {
        float ax = sensorWindow[index][0];
        float ay = sensorWindow[index][1];
        float az = sensorWindow[index][2];
        float mag = (float) Math.sqrt(ax*ax + ay*ay + az*az);
        if (mag < 0.1f) return 0;
        return (float) Math.toDegrees(Math.acos(az / mag));
    }

    private void processAdvancedDetection(float maxAccel, float avgMovement, float avgPreActivity,
            float minAccel, float moveAfter, float tiltDiff,
            float avgPreGyroActivity, float maxGyroMag, float[] probs) {

        float fallProb = probs[3];
        int predictedClass = 0;
        float maxProb = -1;
        for (int i = 0; i < 4; i++) { if (probs[i] > maxProb) { maxProb = probs[i]; predictedClass = i; } }

        // --- ANTI-FALSE-POSITIVE-LOGIK (V4) ---

        boolean wasShakingBefore = (avgPreActivity > 6.0f);
        boolean hasFreeFall = (minAccel < 3.0f);
        boolean isLyingStill = (moveAfter < 1.8f);
        boolean rotated = (tiltDiff > 22.0f);

        // NEU: Handy-Drop-Erkennung
        // Wenn VOR dem Ereignis KEINE Körperbewegung messbar ist (weder Beschleunigung noch Rotation),
        // handelt es sich wahrscheinlich um ein vom Tisch fallendes Handy, nicht um einen Personensturz.
        boolean preCompletelyStill = (avgPreActivity < 0.5f && avgPreGyroActivity < 0.10f);
        boolean noBodyRotation = (maxGyroMag < 1.5f);
        boolean isPhoneDrop = (preCompletelyStill && noBodyRotation);

        String vetoReason = "";
        if (fallProb > 0.80f) {
            if (wasShakingBefore)                  vetoReason = "Vorgeschichte unruhig (Schütteln)";
            else if (!hasFreeFall && maxAccel < 40f) vetoReason = "Kein freier Fall";
            else if (isPhoneDrop)                  vetoReason = "Kein Körpersturz (Handy-Sturz)";
            else if (!isLyingStill)                vetoReason = "Bewegung nach Stoß";
            else if (!rotated)                     vetoReason = "Orientierung stabil";
        }

        boolean finalDecision = (fallProb > FALL_CONFIDENCE_THRESHOLD && vetoReason.isEmpty());

        if (finalDecision) {
            consecutiveFallWindows++;
        } else {
            consecutiveFallWindows = 0;
        }

        String resultText = "Kein Sturz";
        String statusDesc = "Alles im grünen Bereich.";

        if (consecutiveFallWindows >= REQUIRED_FALL_WINDOWS) {
            resultText = "Sturz erkannt";
            statusDesc = "Notfall bestätigt (KI+Physik+Körpersignal)";
        } else if (fallProb > 0.60f) {
            resultText = "Verdächtige Bewegung";
            statusDesc = vetoReason.isEmpty() ? "KI prüft auf Bestätigung..." : "Abgelehnt: " + vetoReason;
        } else if (probs[2] > 0.5f || maxAccel > 25.0f) {
            resultText = "Verdächtige Bewegung";
            statusDesc = "Erhöhte Aktivität erkannt.";
        }

        // Ergebnis für Replay-Modus zwischenspeichern
        lastInferenceResult = resultText;
        lastInferenceSafetyNote = statusDesc;
        System.arraycopy(probs, 0, lastInferenceProbs, 0, probs.length);

        updateResultUi(resultText, (int) (maxProb * 100), probs);
        textStatusDescription.setText(statusDesc);

        if (!isReplayMode && finalDecision && consecutiveFallWindows >= REQUIRED_FALL_WINDOWS) {
            startEmergencyCountdown();
        }

        Log.d(TAG, String.format(Locale.getDefault(),
                "Advanced: AI=%.2f pre_acc=%.2f pre_gyro=%.3f maxGyro=%.2f phoneDrop=%b veto=%s result=%s",
                fallProb, avgPreActivity, avgPreGyroActivity, maxGyroMag, isPhoneDrop, vetoReason, resultText));
        appendLog(String.format(Locale.getDefault(), "KI: %.1f%% | preGyro=%.2f | %s%s",
                fallProb * 100, avgPreGyroActivity, resultText,
                vetoReason.isEmpty() ? "" : " [" + vetoReason + "]"));
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        if (!monitoringActive) return;

        if (event.sensor.getType() == Sensor.TYPE_ACCELEROMETER) {
            latestAccel[0] = event.values[0];
            latestAccel[1] = event.values[1];
            latestAccel[2] = event.values[2];
            accelReceived = true;
            
            float magnitude = (float) Math.sqrt(latestAccel[0] * latestAccel[0] + latestAccel[1] * latestAccel[1] + latestAccel[2] * latestAccel[2]);
            textAccel.setText(String.format(Locale.getDefault(), "%.1f m/s²", magnitude));
        } else if (event.sensor.getType() == Sensor.TYPE_GYROSCOPE) {
            latestGyro[0] = event.values[0];
            latestGyro[1] = event.values[1];
            latestGyro[2] = event.values[2];
            gyroReceived = true;

            float magnitude = (float) Math.sqrt(latestGyro[0] * latestGyro[0] + latestGyro[1] * latestGyro[1] + latestGyro[2] * latestGyro[2]);
            textGyro.setText(String.format(Locale.getDefault(), "%.1f rad/s", magnitude));
        } else if (event.sensor.getType() == Sensor.TYPE_PRESSURE) {
            lastPressure = event.values[0];
            if (startPressure == -1) startPressure = lastPressure;
            
            // Update UI
            runOnUiThread(() -> textPressure.setText(String.format(Locale.getDefault(), "%.1f hPa", lastPressure)));
        }

        if (accelReceived && gyroReceived) {
            long currentTime = System.currentTimeMillis();

            // Fixes 50Hz Sampling: Nur alle 20ms ein Sample in den Puffer aufnehmen
            if (currentTime - lastSampleTime >= SAMPLE_PERIOD_MS) {
                lastSampleTime = currentTime;
                
                // Initialen Winkel für Veto-Check speichern
                if (initialTiltAngle == -1) initialTiltAngle = calculateCurrentTilt();

                // Sample in den Ringbuffer einfügen
                sensorWindow[windowIndex][0] = latestAccel[0];
                sensorWindow[windowIndex][1] = latestAccel[1];
                sensorWindow[windowIndex][2] = latestAccel[2];
                sensorWindow[windowIndex][3] = latestGyro[0];
                sensorWindow[windowIndex][4] = latestGyro[1];
                sensorWindow[windowIndex][5] = latestGyro[2];

                windowIndex = (windowIndex + 1) % WINDOW_SIZE;
                if (samplesInWindow < WINDOW_SIZE) samplesInWindow++;

                // Inferenz-Timing prüfen
                if (samplesInWindow >= WINDOW_SIZE && (currentTime - lastInferenceTime) > INFERENCE_INTERVAL_MS) {
                    performLiveInference();
                    lastInferenceTime = currentTime;
                }
            }

            accelReceived = false;
            gyroReceived = false;
        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {}
}
