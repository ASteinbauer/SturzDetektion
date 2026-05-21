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

import org.json.JSONArray;
import org.json.JSONObject;
import org.tensorflow.lite.Interpreter;

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
    private static final String MODEL_FILE = "fall_detection_1dcnn_float32.tflite";
    private static final String METADATA_FILE = "metadata.json";

    private SensorManager sensorManager;
    private Sensor accelerometer;
    private Sensor gyroscope;

    private TextView accelTextView;
    private TextView gyroTextView;
    private TextView statusTextView;
    private TextView countdownTextView;
    private TextView aiPredictionTextView;
    private TextView aiResultTextView;
    private TextView logTextView;

    private Button startStopButton;
    private Button cancelButton;
    private Button simulateFallButton;
    private Button simulateFalsePositiveButton;
    private Button btnRandomTest;

    private boolean isMonitoring = false;
    private boolean isAlarmActive = false;
    private CountDownTimer countDownTimer;

    // TFLite / LiteRT
    private Interpreter tflite;
    private double[] mean;
    private double[] std;
    private String[] classNames;

    // Schwellenwerte für Rohdaten-Trigger (Echtzeit)
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
        aiResultTextView = findViewById(R.id.aiResultTextView);
        logTextView = findViewById(R.id.logTextView);

        startStopButton = findViewById(R.id.startStopButton);
        cancelButton = findViewById(R.id.cancelButton);
        simulateFallButton = findViewById(R.id.simulateFallButton);
        simulateFalsePositiveButton = findViewById(R.id.simulateFalsePositiveButton);
        btnRandomTest = findViewById(R.id.btnRandomTest);

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

        // Test-Simulationen
        simulateFallButton.setOnClickListener(v -> simulateTriggerEvent(35.0, 6.0));
        simulateFalsePositiveButton.setOnClickListener(v -> simulateTriggerEvent(32.0, 5.5));

        // KI Integrations-Test Button
        btnRandomTest.setOnClickListener(v -> runRandomAiTest());

        // KI Initialisierung
        initAiComponents();
    }

    private void initAiComponents() {
        try {
            loadMetadata();
            tflite = new Interpreter(loadModelFile());
            addLog("KI-Modell & Metadaten erfolgreich geladen.");
        } catch (Exception e) {
            String error = "Fehler beim Laden der KI: " + e.getMessage();
            addLog(error);
            aiResultTextView.setText(error);
            Log.e(TAG, error, e);
        }
    }

    private void loadMetadata() throws Exception {
        String json;
        try (InputStream is = getAssets().open(METADATA_FILE)) {
            int size = is.available();
            byte[] buffer = new byte[size];
            is.read(buffer);
            json = new String(buffer, StandardCharsets.UTF_8);
        }

        JSONObject root = new JSONObject(json);
        JSONObject norm = root.getJSONObject("normalization");
        JSONArray meanArray = norm.getJSONArray("mean");
        JSONArray stdArray = norm.getJSONArray("std");
        JSONArray classesArray = root.getJSONArray("classes");

        mean = new double[6];
        std = new double[6];
        classNames = new String[classesArray.length()];

        for (int i = 0; i < 6; i++) {
            mean[i] = meanArray.getDouble(i);
            std[i] = stdArray.getDouble(i);
        }
        for (int i = 0; i < classNames.length; i++) {
            classNames[i] = classesArray.getString(i);
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

    /**
     * Führt einen Test mit synthetischen Daten durch.
     */
    private void runRandomAiTest() {
        if (tflite == null) {
            Toast.makeText(this, "Modell nicht geladen!", Toast.LENGTH_SHORT).show();
            return;
        }

        int expectedClass = new Random().nextInt(5);
        float[][] rawData = generateSyntheticData(expectedClass);
        
        // Normalisierung: [100][6] -> [1][100][6]
        float[][][] input = new float[1][100][6];
        for (int i = 0; i < 100; i++) {
            for (int j = 0; j < 6; j++) {
                input[0][i][j] = (float) ((rawData[i][j] - mean[j]) / std[j]);
            }
        }

        float[][] output = new float[1][5];
        tflite.run(input, output);

        // Ergebnis auswerten
        int predictedClass = 0;
        float maxVal = -1;
        StringBuilder probs = new StringBuilder("Wahrscheinlichkeiten:\n");
        for (int i = 0; i < 5; i++) {
            probs.append(String.format(Locale.getDefault(), "%s: %.2f%%\n", classNames[i], output[0][i] * 100));
            if (output[0][i] > maxVal) {
                maxVal = output[0][i];
                predictedClass = i;
            }
        }

        boolean success = (expectedClass == predictedClass);
        String resultText = "ERGEBNIS: " + (success ? "RICHTIG ✅" : "FALSCH ❌") +
                "\nErwartet: " + classNames[expectedClass] +
                "\nErkannt: " + classNames[predictedClass] +
                "\n\n" + probs.toString();

        aiResultTextView.setText(resultText);
        addLog("Test ausgeführt: Erwartet=" + classNames[expectedClass] + ", Erkannt=" + classNames[predictedClass]);
    }

    /**
     * Erzeugt synthetische Daten für die 5 Klassen.
     */
    private float[][] generateSyntheticData(int classIndex) {
        float[][] data = new float[100][6];
        Random r = new Random();

        for (int i = 0; i < 100; i++) {
            switch (classIndex) {
                case 0: // alltag_ruhend: ax~0, ay~0, az~9.81
                    data[i][0] = (float) (r.nextGaussian() * 0.1);
                    data[i][1] = (float) (r.nextGaussian() * 0.1);
                    data[i][2] = (float) (9.81 + r.nextGaussian() * 0.1);
                    break;
                case 1: // gehen: Sinus-Wellen auf Accel
                    data[i][2] = (float) (9.81 + Math.sin(i * 0.5) * 2.0);
                    data[i][0] = (float) (Math.cos(i * 0.5) * 0.5);
                    break;
                case 2: // hinlegen: Langsame Drehung
                    data[i][2] = (float) (9.81 - (i / 100.0) * 9.81);
                    data[i][0] = (float) ((i / 100.0) * 9.81);
                    data[i][4] = 1.0f; // Etwas Gyro y
                    break;
                case 3: // handy_faellt: 0g Phase, dann Peak
                    if (i > 30 && i < 40) { // Freier Fall
                        data[i][0] = data[i][1] = data[i][2] = 0.1f;
                    } else if (i >= 40 && i < 45) { // Aufprall
                        data[i][2] = 40.0f;
                    } else {
                        data[i][2] = 9.81f;
                    }
                    break;
                case 4: // sturz: Viel Bewegung, Impact, Ruhe
                    if (i < 20) data[i][5] = 5.0f; // Rotation vor Sturz
                    else if (i < 30) data[i][2] = 0.5f; // Fall
                    else if (i < 35) data[i][2] = 50.0f; // Heftiger Impact
                    else data[i][2] = 0.0f; // Ruhe nachher (liegt am Boden)
                    break;
            }
        }
        return data;
    }

    private void startMonitoring() {
        if (accelerometer != null && gyroscope != null) {
            sensorManager.registerListener(this, accelerometer, SensorManager.SENSOR_DELAY_UI);
            sensorManager.registerListener(this, gyroscope, SensorManager.SENSOR_DELAY_UI);
            isMonitoring = true;
            statusTextView.setText("Status: Aktiv");
            statusTextView.setTextColor(Color.GREEN);
            startStopButton.setText("Stopp");
            addLog("Überwachung gestartet.");
        } else {
            Toast.makeText(this, "Sensoren nicht verfügbar!", Toast.LENGTH_SHORT).show();
        }
    }

    private void stopMonitoring() {
        sensorManager.unregisterListener(this);
        isMonitoring = false;
        statusTextView.setText("Status: Inaktiv");
        statusTextView.setTextColor(Color.BLACK);
        startStopButton.setText("Start");
        if (isAlarmActive) cancelAlarm();
        addLog("Überwachung gestoppt.");
    }

    private void simulateTriggerEvent(double accel, double gyro) {
        if (!isMonitoring) {
            Toast.makeText(this, "Bitte Überwachung erst starten!", Toast.LENGTH_SHORT).show();
            return;
        }
        addLog("Simuliere Trigger: Accel=" + accel + ", Gyro=" + gyro);
        currentAccelMag = accel;
        currentGyroMag = gyro;
        
        accelTextView.setText(String.format(Locale.getDefault(), "Sim-Accel: %.2f m/s²", accel));
        gyroTextView.setText(String.format(Locale.getDefault(), "Sim-Gyro: %.2f rad/s", gyro));
        
        // In einem echten Szenario würde hier das letzte 100er Fenster an die KI gehen.
        // Für den Demo-Zweck triggern wir hier den KI-Check mit einem "Sturz"-Test
        runRandomAiTest();
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
        statusTextView.setText("Status: Aktiv");
        statusTextView.setTextColor(Color.GREEN);
        aiPredictionTextView.setText("KI Echtzeit Status: Standby");
        addLog("Alarm abgebrochen.");
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        if (!isMonitoring || isAlarmActive) return;

        if (event.sensor.getType() == Sensor.TYPE_ACCELEROMETER) {
            currentAccelMag = Math.sqrt(event.values[0]*event.values[0] + 
                                       event.values[1]*event.values[1] + 
                                       event.values[2]*event.values[2]);
            accelTextView.setText(String.format(Locale.getDefault(), "Beschleunigung: %.2f m/s²", currentAccelMag));
        } else if (event.sensor.getType() == Sensor.TYPE_GYROSCOPE) {
            currentGyroMag = Math.sqrt(event.values[0]*event.values[0] + 
                                      event.values[1]*event.values[1] + 
                                      event.values[2]*event.values[2]);
            gyroTextView.setText(String.format(Locale.getDefault(), "Drehung: %.2f rad/s", currentGyroMag));
        }

        // Automatischer Trigger bei Schwellenwert-Überschreitung
        if (currentAccelMag > FALL_THRESHOLD_ACCEL && currentGyroMag > FALL_THRESHOLD_GYRO) {
            addLog("Trigger erreicht! Starte KI-Analyse...");
            runRandomAiTest(); // Testweise wird hier ein KI-Lauf gestartet
        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {}
}
