plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "dev.fbt.camprobe"
    compileSdk = 36
    defaultConfig {
        applicationId = "dev.fbt.camprobe"
        minSdk = 28
        targetSdk = 36
        versionCode = 1
        versionName = "0.1"
    }
    // Same app under different package names: some vendors only reveal
    // auxiliary (ultra-wide) cameras to allow-listed packages.
    flavorDimensions += "pkg"
    productFlavors {
        create("plain") { dimension = "pkg" }
        create("scan3d") { dimension = "pkg"; applicationId = "com.samsung.android.scan3d" }
        create("ruler") { dimension = "pkg"; applicationId = "com.samsung.android.ruler" }
        create("xiaomi") { dimension = "pkg"; applicationId = "com.xiaomi.cameratest" }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

kotlin { jvmToolchain(17) }
