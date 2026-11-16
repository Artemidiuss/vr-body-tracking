package dev.fbt.cam

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.SystemClock
import org.json.JSONArray
import org.json.JSONObject
import kotlin.math.acos
import kotlin.math.sqrt

/**
 * Where "up" is for this phone, from the accelerometer averaged over the last 2 s.
 * The phone stands still on its mount, so the reading is gravity (it points up, as
 * Android reports it: +9.81 on Z for a phone lying face up). Device axes: X right,
 * Y up, Z out of the screen, in the natural (portrait) orientation. The Mac turns it
 * into the camera's axes and uses it as the vertical of the calibration.
 * The spread of the samples tells the Mac whether the phone was really still.
 */
class GravityMeter(ctx: Context) : SensorEventListener {
    private val sm = ctx.getSystemService(SensorManager::class.java)
    private val sensor = sm?.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
    private val samples = ArrayDeque<FloatArray>()      // x, y, z, time ms
    private val windowMs = 2000L

    fun start() { sensor?.let { sm?.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME) } }
    fun stop() { sm?.unregisterListener(this) }

    override fun onSensorChanged(e: SensorEvent) {
        val now = SystemClock.elapsedRealtime()
        synchronized(samples) {
            samples.addLast(floatArrayOf(e.values[0], e.values[1], e.values[2], now.toFloat()))
            while (samples.isNotEmpty() && now - samples.first()[3] > windowMs) samples.removeFirst()
        }
    }

    override fun onAccuracyChanged(s: Sensor?, accuracy: Int) {}

    /** {"v": [x, y, z] mean m/s², "sd_deg": spread of directions, "n": samples}, or null. */
    fun json(): JSONObject? {
        val s = synchronized(samples) { samples.toList() }
        if (s.size < 10) return null
        var x = 0.0; var y = 0.0; var z = 0.0
        for (v in s) { x += v[0]; y += v[1]; z += v[2] }
        x /= s.size; y /= s.size; z /= s.size
        val m = sqrt(x * x + y * y + z * z)
        if (m < 1e-3) return null
        var sum = 0.0
        for (v in s) {
            val n = sqrt((v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).toDouble()).coerceAtLeast(1e-6)
            val c = ((v[0] * x + v[1] * y + v[2] * z) / (n * m)).coerceIn(-1.0, 1.0)
            val a = Math.toDegrees(acos(c))
            sum += a * a
        }
        return JSONObject().apply {
            put("v", JSONArray(listOf(x, y, z)))
            put("sd_deg", sqrt(sum / s.size))
            put("n", s.size)
        }
    }
}
