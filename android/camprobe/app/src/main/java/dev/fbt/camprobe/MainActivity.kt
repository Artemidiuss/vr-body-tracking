package dev.fbt.camprobe

import android.Manifest
import android.app.Activity
import android.content.pm.PackageManager
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.widget.ScrollView
import android.widget.TextView
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import kotlin.math.atan
import kotlin.math.max

/**
 * Lists every camera the OS reveals to this package, plus probes IDs that
 * are NOT in the public list (vendors hide ultra-wide lenses), trying to
 * read their characteristics and actually open them.
 * Output goes to the screen and to logcat with tag CAMPROBE.
 */
class MainActivity : Activity() {
    private lateinit var out: TextView
    private val sb = StringBuilder()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        out = TextView(this).apply { textSize = 11f; setPadding(24, 24, 24, 24) }
        setContentView(ScrollView(this).apply { addView(out) })
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.CAMERA), 1)
        } else probe()
    }

    override fun onRequestPermissionsResult(rc: Int, p: Array<out String>, r: IntArray) {
        probe()
    }

    private fun line(s: String) {
        Log.i("CAMPROBE", s)
        sb.append(s).append('\n')
        runOnUiThread { out.text = sb }
    }

    private fun probe() {
        Thread {
            val cm = getSystemService(CameraManager::class.java)
            line("package=$packageName")
            val public = cm.cameraIdList.toList()
            line("public ids=$public")
            for (id in public) describe(cm, id, "PUBLIC")
            // Probe hidden ids by brute force.
            for (n in 0..99) {
                val id = n.toString()
                if (id in public) continue
                try {
                    cm.getCameraCharacteristics(id)
                    describe(cm, id, "HIDDEN")
                } catch (e: Exception) {
                    // not exposed to this package
                }
            }
            line("done")
        }.start()
    }

    private fun describe(cm: CameraManager, id: String, kind: String) {
        val c = cm.getCameraCharacteristics(id)
        val facing = when (c.get(CameraCharacteristics.LENS_FACING)) {
            CameraCharacteristics.LENS_FACING_BACK -> "back"
            CameraCharacteristics.LENS_FACING_FRONT -> "front"
            else -> "ext"
        }
        val f = c.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)?.firstOrNull() ?: 0f
        val size = c.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)
        val long = size?.let { max(it.width, it.height) } ?: 0f
        val vfov = if (f > 0) Math.toDegrees(2 * atan(long / 2.0 / f)) else 0.0
        val fps = c.get(CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES)?.joinToString()
        val caps = c.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES)?.joinToString()
        val phys = c.physicalCameraIds.joinToString()
        line("[$kind] id=$id $facing f=${"%.2f".format(f)}mm vfov(portrait16:9)=${"%.0f".format(vfov)}° fps=$fps caps=$caps phys=[$phys]")
        line("   open: ${tryOpen(cm, id)}")
    }

    private fun tryOpen(cm: CameraManager, id: String): String {
        val ht = HandlerThread("cam$id").apply { start() }
        val latch = CountDownLatch(1)
        var result = "timeout"
        try {
            cm.openCamera(id, object : CameraDevice.StateCallback() {
                override fun onOpened(d: CameraDevice) { result = "OK"; d.close(); latch.countDown() }
                override fun onDisconnected(d: CameraDevice) { result = "disconnected"; d.close(); latch.countDown() }
                override fun onError(d: CameraDevice, e: Int) { result = "error $e"; d.close(); latch.countDown() }
            }, Handler(ht.looper))
            latch.await(3, TimeUnit.SECONDS)
        } catch (e: Exception) {
            result = "exception ${e.javaClass.simpleName}: ${e.message}"
        } finally {
            ht.quitSafely()
        }
        return result
    }
}
