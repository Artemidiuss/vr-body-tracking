plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "dev.fbt.cam"
    compileSdk = 36
    defaultConfig {
        applicationId = "dev.fbt.cam"
        minSdk = 28
        targetSdk = 36
        versionCode = 3
        versionName = "0.3"
    }
    // Vendors reveal ultra-wide lenses only to allow-listed package names
    // (verified 2026-11-14). For personal sideloading only, never publish
    // these flavors.
    flavorDimensions += "pkg"
    productFlavors {
        create("plain") { dimension = "pkg" }
        create("samsung") { dimension = "pkg"; applicationId = "com.samsung.android.scan3d" }
        create("xiaomi") { dimension = "pkg"; applicationId = "com.xiaomi.cameratest" }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

kotlin { jvmToolchain(17) }
