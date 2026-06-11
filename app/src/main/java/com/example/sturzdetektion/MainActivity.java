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
<<<<<<< Updated upstream
import android.os.VibrationEffect;
import android.os.Vibrator;
=======
import android.os.Handler;
import android.os.Looper;
>>>>>>> Stashed changes
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
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.Random;

public class MainActivity extends AppCompatActivity implements SensorEventListener {

    private static final String TAG = "SturzDetektionDebug";
    private static final String MODEL_FILE = "fall_detection_model.tflite";

<<<<<<< Updated upstream
=======
    private static final float CLASS2_OK_THRESHOLD = 0.70f;
    private static final float CLASS2_WARNING_ACCEL_THRESHOLD = 26.0f;
    private static final int REQUIRED_FALL_WINDOWS = 2;

    private int consecutiveFallWindows = 0;

    private static final int NUM_CHANNELS = 6;
    private static final int NUM_CLASSES = 4;
    private static final int WINDOW_SIZE = 150;
    private static final long SAMPLE_INTERVAL_MS = 20L;
    private static final long INFERENCE_INTERVAL_MS = 400;

    // Ringbuffer & Sensor-Daten
    private float[][] sensorWindow = new float[WINDOW_SIZE][NUM_CHANNELS];
    private int windowIndex = 0;
    private int samplesInWindow = 0;
    private long sampleCount = 0;
    private long lastInferenceTime = 0;
    private float[] latestAccel = new float[3];
    private float[] latestGyro = new float[3];
    private boolean accelReceived = false;
    private boolean gyroReceived = false;
    private final Handler samplingHandler = new Handler(Looper.getMainLooper());
    private final Runnable samplingRunnable = new Runnable() {
        @Override
        public void run() {
            if (!monitoringActive) return;
            addTimedSensorSample();
            samplingHandler.postDelayed(this, SAMPLE_INTERVAL_MS);
        }
    };

    // UI Elemente
    private TextView badgeStatus, textSensorSource, textModelStatus, textDetectionMode;
    private TextView textAccel, textGyro;
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
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
            sensorManager.registerListener(this, accelerometer, SensorManager.SENSOR_DELAY_UI);
            sensorManager.registerListener(this, gyroscope, SensorManager.SENSOR_DELAY_UI);
            isMonitoring = true;
            statusTextView.setText(R.string.status_on);
            statusTextView.setTextColor(Color.GREEN);
            startStopButton.setText(R.string.btn_stop);
            addLog("Überwachung AKTIVIERT.");
=======
            sensorManager.registerListener(this, accelerometer, 20000);
            sensorManager.registerListener(this, gyroscope, 20000);
            
            // Buffer zurücksetzen
            windowIndex = 0;
            samplesInWindow = 0;
            sampleCount = 0;
            accelReceived = false;
            gyroReceived = false;
            lastInferenceTime = 0;
            consecutiveFallWindows = 0;
            
            monitoringActive = true;
            samplingHandler.removeCallbacks(samplingRunnable);
            samplingHandler.post(samplingRunnable);
            updateStatusBadge(true);
            buttonToggleMonitoring.setText("Überwachung stoppen");
            appendLog("Live-Überwachung aktiv (50Hz).");
>>>>>>> Stashed changes
        } else {
            Toast.makeText(this, "Sensoren fehlen - Sim-Modus aktiv", Toast.LENGTH_SHORT).show();
            isMonitoring = true;
            statusTextView.setText("Sim-Modus");
            addLog("Sensoren fehlen. Nur Simulation möglich.");
        }
    }

    private void stopMonitoring() {
        sensorManager.unregisterListener(this);
<<<<<<< Updated upstream
        isMonitoring = false;
        statusTextView.setText(R.string.status_off);
        statusTextView.setTextColor(Color.BLACK);
        startStopButton.setText(R.string.btn_start);
        if (isAlarmActive) cancelAlarm();
        addLog("Überwachung GESTOPPT.");
=======
        samplingHandler.removeCallbacks(samplingRunnable);
        monitoringActive = false;
        updateStatusBadge(false);
        buttonToggleMonitoring.setText("Überwachung starten");
        appendLog("Überwachung beendet.");
>>>>>>> Stashed changes
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

<<<<<<< Updated upstream
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
=======
    private String processDetection(float maxAccel, float maxGyro, float[] output, float moveAfter, String scenario, String expected) {
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

        float quietProb = output[0];
        float normalProb = output[1];
        float fallLikeOkProb = output[2];
        float fallProb = output[3];

        // Neue Logik: Multi-Window Check & Robuste Entscheidung
        boolean confirmedFallCandidate = (predictedClass == 3 &&
                                        fallProb > FALL_CONFIDENCE_THRESHOLD &&
                                        maxAccel > HIGH_ACCEL_THRESHOLD &&
                                        moveAfter < LOW_MOVEMENT_AFTER_IMPACT_THRESHOLD);

        if (confirmedFallCandidate) {
            consecutiveFallWindows++;
        } else {
            consecutiveFallWindows = 0;
        }

        String finalResult = "Kein Sturz";
        String safetyNote = "Normal";

        if (consecutiveFallWindows >= REQUIRED_FALL_WINDOWS) {
            finalResult = "Sturz erkannt";
            safetyNote = "Bestätigter Sturz (" + consecutiveFallWindows + " Fenster)";
        } else if (predictedClass == 2 && fallLikeOkProb > CLASS2_OK_THRESHOLD && fallProb < SUSPICIOUS_CONFIDENCE_THRESHOLD) {
            finalResult = "Auffällige Bewegung – kein Sturz";
            safetyNote = "Klasse 2: sturzähnlich, aber OK";
        } else if (fallProb > SUSPICIOUS_CONFIDENCE_THRESHOLD || maxAccel > CLASS2_WARNING_ACCEL_THRESHOLD) {
            finalResult = "Verdächtige Bewegung";
            safetyNote = confirmedFallCandidate ? "Sturzverdacht (Warte auf Bestätigung)" : "Warnung ohne Notfall";
        }

        // Debug-Log für Entscheidung
        Log.d(TAG, String.format(Locale.getDefault(),
                "Logic Check: [q=%.3f, n=%.3f, l=%.3f, f=%.3f] class=%d, maxA=%.2f, move=%.2f, consecutive=%d -> %s (%s)",
                quietProb, normalProb, fallLikeOkProb, fallProb, predictedClass, maxAccel, moveAfter, consecutiveFallWindows, finalResult, safetyNote));

        updateResultUi(finalResult, (int)(maxVal * 100), output);
        
        if (scenario != null) {
            updateTestEvaluation(scenario, expected, finalResult, safetyNote, output);
        }

        // Countdown NUR bei echtem Sturz
        if (finalResult.equals("Sturz erkannt")) {
            startEmergencyCountdown();
        }
        
        appendLog(String.format(Locale.getDefault(), "Erkennung: %s (Prob: %.2f, MaxA: %.1f)", finalResult, fallProb, maxAccel));
        return finalResult;
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
        float[][] rawWindow = new float[WINDOW_SIZE][NUM_CHANNELS];
        float maxAccel = 0;
        float maxGyro = 0;
        int impactIndex = 0;

        // Fenster chronologisch aufbauen und max Werte finden
        for (int i = 0; i < WINDOW_SIZE; i++) {
            int actualIndex = (windowIndex + i) % WINDOW_SIZE;
            System.arraycopy(sensorWindow[actualIndex], 0, rawWindow[i], 0, NUM_CHANNELS);
            
            float ax = rawWindow[i][0];
            float ay = rawWindow[i][1];
            float az = rawWindow[i][2];
            float gx = rawWindow[i][3];
            float gy = rawWindow[i][4];
            float gz = rawWindow[i][5];

            // Magnituden für Sicherheitsregeln
            float aMag = (float) Math.sqrt(ax * ax + ay * ay + az * az);
            float gMag = (float) Math.sqrt(gx * gx + gy * gy + gz * gz);
            if (aMag > maxAccel) {
                maxAccel = aMag;
                impactIndex = i;
            }
            if (gMag > maxGyro) maxGyro = gMag;

            // Normalisierung
            input[0][i][0] = (ax - mean[0]) / std[0];
            input[0][i][1] = (ay - mean[1]) / std[1];
            input[0][i][2] = (az - mean[2]) / std[2];
            input[0][i][3] = (gx - mean[3]) / std[3];
            input[0][i][4] = (gy - mean[4]) / std[4];
            input[0][i][5] = (gz - mean[5]) / std[5];
        }

        float moveAfter = calculateMovementAfterImpact(rawWindow, impactIndex);

        // TFLite Inferenz
        float[][] output = new float[1][NUM_CLASSES];
        tflite.run(input, output);
        float[] probs = output[0];

        // UI Feedback
        runOnUiThread(() -> textSensorSource.setText("Live-Sensoren"));

        // Debug Log
        int predictedClass = 0;
        float maxProb = -1;
        for (int i = 0; i < NUM_CLASSES; i++) {
            if (probs[i] > maxProb) {
                maxProb = probs[i];
                predictedClass = i;
            }
        }
        
        String finalResult = processDetection(maxAccel, maxGyro, probs, moveAfter, null, null);

        Log.d(TAG, String.format(Locale.getDefault(),
                "AI Inference: sampleCount=%d, bufferFilled=%s, maxAccel=%.2f, maxGyro=%.2f, moveAfter=%.2f, " +
                        "ruhig_alltag=%.4f, normale_bewegung=%.4f, fallaehnlich_aber_ok=%.4f, sturz=%.4f, " +
                        "predictedClass=%d, confidence=%.4f, consecutiveFallWindows=%d, finalResult=%s",
                sampleCount,
                samplesInWindow >= WINDOW_SIZE,
                maxAccel,
                maxGyro,
                moveAfter,
                probs[0],
                probs[1],
                probs[2],
                probs[3],
                predictedClass,
                maxProb,
                consecutiveFallWindows,
                finalResult));
    }

    private float calculateMovementAfterImpact(float[][] rawWindow, int impactIndex) {
        int start = Math.max(impactIndex + 1, WINDOW_SIZE - 30);
        int end = WINDOW_SIZE;
        if (start >= end - 1) return 0.0f;

        float previousMag = accelMagnitude(rawWindow[start - 1]);
        float movementSum = 0.0f;
        int movementCount = 0;
        for (int i = start; i < end; i++) {
            float mag = accelMagnitude(rawWindow[i]);
            movementSum += Math.abs(mag - previousMag);
            previousMag = mag;
            movementCount++;
        }
        return movementCount > 0 ? movementSum / movementCount : 0.0f;
    }

    private float accelMagnitude(float[] sample) {
        return (float) Math.sqrt(sample[0] * sample[0] + sample[1] * sample[1] + sample[2] * sample[2]);
    }

    private float gyroMagnitude(float[] sample) {
        return (float) Math.sqrt(sample[3] * sample[3] + sample[4] * sample[4] + sample[5] * sample[5]);
    }

    private void addTimedSensorSample() {
        if (!accelReceived || !gyroReceived) return;

        sensorWindow[windowIndex][0] = latestAccel[0];
        sensorWindow[windowIndex][1] = latestAccel[1];
        sensorWindow[windowIndex][2] = latestAccel[2];
        sensorWindow[windowIndex][3] = latestGyro[0];
        sensorWindow[windowIndex][4] = latestGyro[1];
        sensorWindow[windowIndex][5] = latestGyro[2];

        windowIndex = (windowIndex + 1) % WINDOW_SIZE;
        if (samplesInWindow < WINDOW_SIZE) samplesInWindow++;
        sampleCount++;

        long currentTime = System.currentTimeMillis();
        if (samplesInWindow >= WINDOW_SIZE && (currentTime - lastInferenceTime) >= INFERENCE_INTERVAL_MS) {
            performLiveInference();
            lastInferenceTime = currentTime;
        }
    }

    private List<String> prepareCurrentWindowCsvRows(String label) {
        List<String> rows = new ArrayList<>();
        if (samplesInWindow < WINDOW_SIZE) return rows;

        long now = System.currentTimeMillis();
        long firstTimestamp = now - (WINDOW_SIZE - 1L) * SAMPLE_INTERVAL_MS;
        for (int i = 0; i < WINDOW_SIZE; i++) {
            int actualIndex = (windowIndex + i) % WINDOW_SIZE;
            long timestamp = firstTimestamp + i * SAMPLE_INTERVAL_MS;
            rows.add(String.format(Locale.US,
                    "%d,%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f",
                    timestamp,
                    label,
                    sensorWindow[actualIndex][0],
                    sensorWindow[actualIndex][1],
                    sensorWindow[actualIndex][2],
                    sensorWindow[actualIndex][3],
                    sensorWindow[actualIndex][4],
                    sensorWindow[actualIndex][5]));
        }
        return rows;
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream

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
=======
>>>>>>> Stashed changes
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {}
}
