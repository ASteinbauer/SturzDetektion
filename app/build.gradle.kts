plugins {
    alias(libs.plugins.android.application)
}

android {
    namespace = "com.example.sturzdetektion"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.example.sturzdetektion"
        minSdk = 26
        targetSdk = 36
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
    
    // TensorFlow Lite Runtime
    implementation("org.tensorflow:tensorflow-lite:2.16.1")

    testImplementation(libs.junit)
    androidTestImplementation(libs.ext.junit)
    androidTestImplementation(libs.espresso.core)
}
