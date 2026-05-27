plugins {
    alias(libs.plugins.android.application)
}

android {
    namespace = "com.example.sturzdetektion"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.example.sturzdetektion"
        minSdk = 29
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
            buildConfigField("boolean", "IS_DEBUG", "false")
        }
        debug {
            buildConfigField("boolean", "IS_DEBUG", "true")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }

    buildFeatures {
        viewBinding = false
        buildConfig = true
    }

    androidResources {
        noCompress += "tflite"
    }
}

dependencies {
    implementation(libs.appcompat)
    implementation(libs.material)
    implementation(libs.activity)
    implementation(libs.constraintlayout)

    // TensorFlow Lite (LiteRT wurde entfernt, um Konflikte zu vermeiden)
    implementation(libs.tensorflow.lite)

    testImplementation(libs.junit)
    androidTestImplementation(libs.ext.junit)
    androidTestImplementation(libs.espresso.core)
}
