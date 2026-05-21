plugins {
    alias(libs.plugins.android.application)
}

android {
    namespace = "com.example.sturzdetektion"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.example.sturzdetektion"
        minSdk = 29
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    
    // TFLite Optimierung: Verhindert das Komprimieren der Model-Datei
    androidResources {
        noCompress += "tflite"
    }
}

dependencies {
    implementation(libs.appcompat)
    implementation(libs.material)
    implementation(libs.activity)
    implementation(libs.constraintlayout)

    // LiteRT / TensorFlow Lite Runtime für .tflite Interpreter
    implementation("com.google.ai.edge.litert:litert:2.1.0")

    testImplementation(libs.junit)
    androidTestImplementation(libs.ext.junit)
    androidTestImplementation(libs.espresso.core)
}