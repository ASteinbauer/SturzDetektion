package com.example.sturzdetektion;

public class SimSensorData {
    public final float accel;
    public final float gyro;
    public final String scenario;
    public final boolean expectedFall;

    public SimSensorData(float accel, float gyro, String scenario, boolean expectedFall) {
        this.accel = accel;
        this.gyro = gyro;
        this.scenario = scenario;
        this.expectedFall = expectedFall;
    }
}
