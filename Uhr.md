package com.example.myapplication;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

/**
 * Finale Version für Galaxy Watch 4 Classic:
 * 2-Faktor-Sturzerkennung (Bewegung + Herzfrequenz)
 */
public class MainActivity extends AppCompatActivity implements SensorEventListener {

    private SensorManager sensorManager;
    private Sensor accelerometer, gyroscope, heartRateSensor;
    private TextView statusText;

    // --- SCHWELLENWERTE ---
    private static final float SHOCK_THRESHOLD = 30.0f;    
    private static final float ROTATION_THRESHOLD = 10.0f; 
    private static final float PULSE_HIGH = 115.0f; 
    private static final float PULSE_LOW = 45.0f;   

    private float currentHeartRate = 0;
    private boolean impactDetected = false;
    private static final int PERM_CODE = 100;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        
        statusText = findViewById(R.id.statusText);

        sensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
        if (sensorManager != null) {
            accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
            gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
            heartRateSensor = sensorManager.getDefaultSensor(Sensor.TYPE_HEART_RATE);
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.BODY_SENSORS) 
            != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.BODY_SENSORS}, PERM_CODE);
        }
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        if (event.sensor.getType() == Sensor.TYPE_HEART_RATE) {
            currentHeartRate = event.values[0];
            updateDisplay();
        }

        if (event.sensor.getType() == Sensor.TYPE_ACCELEROMETER) {
            double gForce = Math.sqrt(event.values[0] * event.values[0] + 
                                     event.values[1] * event.values[1] + 
                                     event.values[2] * event.values[2]);
            if (gForce > SHOCK_THRESHOLD) {
                impactDetected = true;
            }
        }

        if (event.sensor.getType() == Sensor.TYPE_GYROSCOPE && impactDetected) {
            double rotation = Math.sqrt(event.values[0] * event.values[0] + 
                                       event.values[1] * event.values[1] + 
                                       event.values[2] * event.values[2]);
            if (rotation > ROTATION_THRESHOLD) {
                verifiziereSturzMitVitaldaten();
                impactDetected = false; 
            }
        }
    }

    private void verifiziereSturzMitVitaldaten() {
        // currentHeartRate == 0 ist redundant, da 0 <= PULSE_LOW (45.0)
        if (currentHeartRate >= PULSE_HIGH || currentHeartRate <= PULSE_LOW) {
            ausloeseNotfallAlarm();
        } else {
            runOnUiThread(() -> Toast.makeText(this, getString(R.string.status_ignored), Toast.LENGTH_SHORT).show());
        }
    }

    private void ausloeseNotfallAlarm() {
        runOnUiThread(() -> {
            String msg = getString(R.string.status_emergency, (int)currentHeartRate);
            Toast.makeText(this, msg, Toast.LENGTH_LONG).show();
            if (statusText != null) {
                statusText.setText(msg);
                statusText.setTextColor(ContextCompat.getColor(this, android.R.color.holo_red_dark));
            }
        });
    }

    private void updateDisplay() {
        runOnUiThread(() -> {
            if (statusText != null && currentHeartRate > 0) {
                statusText.setText(getString(R.string.status_monitoring, (int)currentHeartRate));
                statusText.setTextColor(ContextCompat.getColor(this, android.R.color.white));
            }
        });
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (accelerometer != null) {
            sensorManager.registerListener(this, accelerometer, SensorManager.SENSOR_DELAY_GAME);
        }
        if (gyroscope != null) {
            sensorManager.registerListener(this, gyroscope, SensorManager.SENSOR_DELAY_GAME);
        }
        if (heartRateSensor != null && ContextCompat.checkSelfPermission(this, Manifest.permission.BODY_SENSORS) == PackageManager.PERMISSION_GRANTED) {
            sensorManager.registerListener(this, heartRateSensor, SensorManager.SENSOR_DELAY_NORMAL);
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (sensorManager != null) {
            sensorManager.unregisterListener(this);
        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {}

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == PERM_CODE && grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            if (heartRateSensor != null) {
                sensorManager.registerListener(this, heartRateSensor, SensorManager.SENSOR_DELAY_NORMAL);
            }
        } else {
            Toast.makeText(this, getString(R.string.perm_needed), Toast.LENGTH_LONG).show();
        }
    }
}
