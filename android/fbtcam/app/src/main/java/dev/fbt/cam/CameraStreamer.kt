package dev.fbt.cam

import android.annotation.SuppressLint
import android.content.Context
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraConstrainedHighSpeedCaptureSession
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.CaptureResult
import android.hardware.camera2.TotalCaptureResult
import android.hardware.camera2.params.OutputConfiguration
import android.hardware.camera2.params.SessionConfiguration
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaFormat
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.os.SystemClock
import android.util.Range
import android.view.Surface
import java.util.concurrent.Executor
import org.json.JSONObject

data class StreamConfig(
    val cam: CamInfo,
    val width: Int,
    val height: Int,
    val fps: Int,
    val highSpeed: Boolean,
    val bitrate: Int,
    val previewTemplate: Boolean = true,   // TEMPLATE_PREVIEW + fast NR/edge (else RECORD)
    val maxOperatingRate: Boolean = true,  // encoder KEY_OPERATING_RATE = max + CBR
    val exposureCompSteps: Int = 0,        // AE compensation (steps); <0 = shorter exposure
)

/**
 * Camera2 -> hardware H.264 encoder (input Surface, zero copy) -> callback.
 * Encoder output timestamps are the sensor capture timestamps, so the Mac
 * knows exactly when each frame was taken.
 */
class CameraStreamer(
    private val ctx: Context,
    private val cfg: StreamConfig,
    private val preview: Surface?,
    private val onConfig: (ByteArray) -> Unit,
    private val onFrame: (ByteArray, Long, Boolean) -> Unit,
    private val log: (String) -> Unit,
) {
    private val thread = HandlerThread("camstream").apply { start() }
    private val handler = Handler(thread.looper)
    private val executor = Executor { handler.post(it) }
    private var encoder: MediaCodec? = null
    private var encoderSurface: Surface? = null
    private var device: CameraDevice? = null
    private var session: CameraCaptureSession? = null
    @Volatile var encodedFrames = 0L
    private val latencies = ArrayDeque<Double>()
    // ---- exposure: metered automatically for a few seconds, then held (see lockExposure)
    private var surfaces: List<Surface> = emptyList()
    private var meterSince = 0L
    private var manualExp: Pair<Long, Int>? = null   // exposure ns, ISO while held manually
    private var aeLock = false                       // held by AE lock (no manual sensor control)
    @Volatile private var expNs = 0L
    @Volatile private var iso = 0
    @Volatile private var held = false
    private val metered = ArrayDeque<Pair<Long, Double>>()   // (time ms, exposure ns x ISO) while metering
    /** Gravity direction (unit, device axes) and its spread over 2 s (deg), from MainActivity. */
    @Volatile var gravity: () -> Pair<DoubleArray, Double>? = { null }
    private var meterDir: DoubleArray? = null        // which way the phone pointed when the light was metered
    private var meterWhileMoving = false             // metered while being handled: meter again once still
    private var checkAt = 0L                         // when to check that the phone obeys the manual exposure
    private val chars by lazy { ctx.getSystemService(CameraManager::class.java).getCameraCharacteristics(cfg.cam.id) }
    private val captureCb = object : CameraCaptureSession.CaptureCallback() {
        override fun onCaptureCompleted(s: CameraCaptureSession, r: CaptureRequest, res: TotalCaptureResult) {
            res.get(CaptureResult.SENSOR_EXPOSURE_TIME)?.let { expNs = it }
            res.get(CaptureResult.SENSOR_SENSITIVITY)?.let { iso = it }
            val now = SystemClock.elapsedRealtime()
            val m = manualExp
            if (held && m != null && checkAt in 1..now) {
                checkAt = 0
                // Some HALs take the keys but keep auto exposure: then hold it with AE lock instead.
                if (kotlin.math.abs(expNs - m.first) > m.first / 5) {
                    log("manual exposure ignored (asked ${m.first / 1_000_000} ms, got ${"%.1f".format(expNs / 1e6)} ms): AE lock")
                    manualExp = null; aeLock = true
                    try { applyRequest(device ?: return, session ?: return) } catch (_: Exception) {}
                }
            }
            if (held || meterSince == 0L) return
            if (expNs > 0 && iso > 0) {
                metered.addLast(now to expNs.toDouble() * iso)
                while (metered.isNotEmpty() && now - metered.first().first > 1500) metered.removeFirst()
            }
            val dt = now - meterSince
            val ae = res.get(CaptureResult.CONTROL_AE_STATE)
            if ((dt > 2500 && ae == CaptureResult.CONTROL_AE_STATE_CONVERGED) || dt > 6000) lockExposure()
        }
    }

    /** Clock the sensor stamps frames with (detected by StreamServer). */
    @Volatile var clockNs: () -> Long = { System.nanoTime() }

    fun medianEncodeLatencyMs(): Double? = synchronized(latencies) {
        if (latencies.isEmpty()) null else latencies.sorted()[latencies.size / 2]
    }

    @SuppressLint("MissingPermission")
    fun start() {
        val fmt = MediaFormat.createVideoFormat(MediaFormat.MIMETYPE_VIDEO_AVC, cfg.width, cfg.height).apply {
            setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatSurface)
            setInteger(MediaFormat.KEY_BIT_RATE, cfg.bitrate)
            setInteger(MediaFormat.KEY_FRAME_RATE, cfg.fps)
            setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1)
            setInteger(MediaFormat.KEY_PROFILE, MediaCodecInfo.CodecProfileLevel.AVCProfileBaseline)
            setInteger(MediaFormat.KEY_LEVEL, MediaCodecInfo.CodecProfileLevel.AVCLevel42)
            setInteger(MediaFormat.KEY_PRIORITY, 0)          // realtime
            setInteger(MediaFormat.KEY_MAX_B_FRAMES, 0)
            setInteger(MediaFormat.KEY_LATENCY, 1)
            // Run the encoder as fast as it can instead of pacing it to the
            // frame rate (common low-latency streaming setting), constant
            // bitrate (no rate-control look-ahead).
            if (cfg.maxOperatingRate) {
                setInteger(MediaFormat.KEY_OPERATING_RATE, Short.MAX_VALUE.toInt())
                setInteger(MediaFormat.KEY_BITRATE_MODE, MediaCodecInfo.EncoderCapabilities.BITRATE_MODE_CBR)
            } else if (cfg.highSpeed) setInteger(MediaFormat.KEY_OPERATING_RATE, cfg.fps)
        }
        val enc = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_VIDEO_AVC)
        enc.setCallback(object : MediaCodec.Callback() {
            override fun onInputBufferAvailable(c: MediaCodec, i: Int) {}
            override fun onOutputBufferAvailable(c: MediaCodec, i: Int, info: MediaCodec.BufferInfo) {
                val buf = try { c.getOutputBuffer(i) } catch (e: Exception) { null } ?: return
                val data = ByteArray(info.size)
                buf.position(info.offset)
                buf.get(data)
                try { c.releaseOutputBuffer(i, false) } catch (_: Exception) {}
                if (info.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG != 0) {
                    onConfig(data)
                } else if (info.size > 0) {
                    encodedFrames++
                    // Capture -> encoded latency measured on the phone itself
                    // (sensor timestamp vs the same clock now).
                    val nowUs = clockNs() / 1000
                    val lat = (nowUs - info.presentationTimeUs) / 1000.0
                    if (lat in 0.0..2000.0) synchronized(latencies) {
                        latencies.addLast(lat); while (latencies.size > 60) latencies.removeFirst()
                    }
                    onFrame(data, info.presentationTimeUs,
                        info.flags and MediaCodec.BUFFER_FLAG_KEY_FRAME != 0)
                }
            }
            override fun onError(c: MediaCodec, e: MediaCodec.CodecException) { log("encoder error: ${e.diagnosticInfo}") }
            override fun onOutputFormatChanged(c: MediaCodec, f: MediaFormat) {}
        }, handler)
        enc.configure(fmt, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
        encoderSurface = enc.createInputSurface()
        enc.start()
        encoder = enc

        val cm = ctx.getSystemService(CameraManager::class.java)
        cm.openCamera(cfg.cam.id, executor, object : CameraDevice.StateCallback() {
            override fun onOpened(d: CameraDevice) { device = d; createSession(d) }
            override fun onDisconnected(d: CameraDevice) { log("camera disconnected"); d.close() }
            override fun onError(d: CameraDevice, e: Int) { log("camera error $e"); d.close() }
        })
    }

    private fun createSession(d: CameraDevice) {
        val surfaces = listOfNotNull(encoderSurface, preview)
        val cb = object : CameraCaptureSession.StateCallback() {
            override fun onConfigured(s: CameraCaptureSession) {
                session = s
                try { startRepeating(d, s, surfaces) } catch (e: Exception) { log("request failed: $e") }
            }
            override fun onConfigureFailed(s: CameraCaptureSession) { log("session configure failed") }
        }
        val type = if (cfg.highSpeed) SessionConfiguration.SESSION_HIGH_SPEED else SessionConfiguration.SESSION_REGULAR
        d.createCaptureSession(SessionConfiguration(type, surfaces.map { OutputConfiguration(it) }, executor, cb))
    }

    private fun startRepeating(d: CameraDevice, s: CameraCaptureSession, surfaces: List<Surface>) {
        this.surfaces = surfaces
        manualExp = null; aeLock = false; held = false
        metered.clear()
        meterSince = SystemClock.elapsedRealtime()
        applyRequest(d, s)
        log("streaming ${cfg.width}x${cfg.height} @${cfg.fps}${if (cfg.highSpeed) " high-speed" else ""}, cam ${cfg.cam.id}")
    }

    /**
     * Auto exposure keeps re-metering: when you walk up to a camera or a light wall fills the
     * frame, the whole room goes darker or brighter, and at 30 fps indoors it picks long
     * exposures that smear fast legs. So it meters for ~3 s, then the exposure is held:
     * the same brightness (exposure x ISO) but at most 10 / 20 / 30 ms (multiples of the
     * 10 ms flicker of 50 Hz lights, so lamps do not band), and white balance held too.
     * It meters again when the phone was turned (the view changed, onGravityTick) and when the
     * Mac asks, at the start of a calibration or a game (remeter()). Someone walking around does
     * not change the room's light, so nothing else does (ARCHITECTURE.md §30).
     */
    private fun lockExposure() {
        val d = device ?: return
        val s = session ?: return
        val t = expNs; val i = iso
        if (t <= 0 || i <= 0 || metered.isEmpty()) return
        // the median light of the last 1.5 s, not one frame: a passing shadow or flash does not stick
        val light = metered.map { it.second }.sorted()[metered.size / 2]
        metered.clear()
        held = true
        val g = gravity()
        meterDir = g?.first
        meterWhileMoving = g == null || g.second >= STILL_DEG
        val caps = chars.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES) ?: IntArray(0)
        val aeModes = chars.get(CameraCharacteristics.CONTROL_AE_AVAILABLE_MODES) ?: IntArray(0)
        val keys = chars.availableCaptureRequestKeys
        // Not every camera that takes manual exposure says MANUAL_SENSOR (the Galaxy A32's
        // camera 50 does not): the AE-off mode and the two keys are enough to try.
        val manual = !cfg.highSpeed && (CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_MANUAL_SENSOR in caps ||
            (CameraCharacteristics.CONTROL_AE_MODE_OFF in aeModes && CaptureRequest.SENSOR_EXPOSURE_TIME in keys &&
             CaptureRequest.SENSOR_SENSITIVITY in keys))
        if (manual) { manualExp = chooseExposure(t, i, light); checkAt = SystemClock.elapsedRealtime() + 1500 }
        else aeLock = true
        try { applyRequest(d, s) } catch (e: Exception) { log("exposure hold failed: $e") }
        val m = manualExp
        log(if (m != null) "exposure held: ${"%.1f".format(m.first / 1e6)} ms ISO ${m.second} (auto was ${"%.1f".format(t / 1e6)} ms ISO $i)"
            else "exposure held (AE lock): ${"%.1f".format(t / 1e6)} ms ISO $i")
    }

    private fun chooseExposure(t: Long, i: Int, light: Double): Pair<Long, Int> {
        val isoRange = chars.get(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE) ?: Range(i, i)
        val expRange = chars.get(CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE) ?: Range(t, t)
        val isoCap = minOf(isoRange.upper, 1600)
        for (ms in longArrayOf(10, 20, 30)) {
            val c = ms * 1_000_000L
            if (c >= t || c > 1_000_000_000L / cfg.fps || c !in expRange) break
            val need = (light / c).toInt()
            if (need <= isoCap) return c to need.coerceIn(isoRange.lower, isoRange.upper)
        }
        return t to (light / t).toInt().coerceIn(isoRange.lower, isoRange.upper)   // bright room or no headroom
    }

    /** Meter the light again (the Mac sends EXPOSURE when a calibration or a game starts). */
    fun remeter() {
        handler.post { meterAgain() }
    }

    private fun meterAgain() {
        val d = device ?: return
        val s = session ?: return
        manualExp = null; aeLock = false; held = false
        metered.clear()
        meterSince = SystemClock.elapsedRealtime()
        try { applyRequest(d, s) } catch (e: Exception) { log("remeter failed: $e") }
    }

    /**
     * Once a second: was the phone turned since the light was metered? Then the camera sees
     * another view (put on its stand after starting in the hand: the ceiling lamps it saw made
     * the picture 1.5x too dark), so it meters again once the phone stands still. Still phones
     * wander < 0.1 deg (measured), so 2 deg is a real turn, not someone walking by.
     */
    fun onGravityTick() {
        handler.post {
            if (!held) return@post
            val (dir, sd) = gravity() ?: return@post
            if (sd >= STILL_DEG) return@post                      // still being handled: wait
            val ref = meterDir
            val turned = ref == null || Math.toDegrees(kotlin.math.acos(
                (dir[0] * ref[0] + dir[1] * ref[1] + dir[2] * ref[2]).coerceIn(-1.0, 1.0))) > TURNED_DEG
            if (meterWhileMoving || turned) {
                log("camera moved: metering the light again")
                meterAgain()
            }
        }
    }

    fun exposureJson(): JSONObject? = if (expNs <= 0) null else JSONObject().apply {
        put("ms", Math.round(expNs / 1e5) / 10.0); put("iso", iso); put("held", held)
    }

    private fun applyRequest(d: CameraDevice, s: CameraCaptureSession) {
        // PREVIEW template + fast processing: the RECORD template asks the ISP for
        // high-quality noise reduction/edge enhancement, which costs latency.
        val b = d.createCaptureRequest(
            if (cfg.previewTemplate) CameraDevice.TEMPLATE_PREVIEW else CameraDevice.TEMPLATE_RECORD)
        if (cfg.previewTemplate) {
            b.set(CaptureRequest.NOISE_REDUCTION_MODE, CaptureRequest.NOISE_REDUCTION_MODE_FAST)
            b.set(CaptureRequest.EDGE_MODE, CaptureRequest.EDGE_MODE_FAST)
        }
        surfaces.forEach { b.addTarget(it) }
        b.set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, Range(cfg.fps, cfg.fps))
        if (cfg.exposureCompSteps != 0) b.set(CaptureRequest.CONTROL_AE_EXPOSURE_COMPENSATION, cfg.exposureCompSteps)
        val m = manualExp
        if (m != null) {
            b.set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_OFF)
            b.set(CaptureRequest.SENSOR_EXPOSURE_TIME, m.first)
            b.set(CaptureRequest.SENSOR_SENSITIVITY, m.second)
            if (CaptureRequest.SENSOR_FRAME_DURATION in chars.availableCaptureRequestKeys)
                b.set(CaptureRequest.SENSOR_FRAME_DURATION, 1_000_000_000L / cfg.fps)
        } else if (aeLock) b.set(CaptureRequest.CONTROL_AE_LOCK, true)
        if (held) b.set(CaptureRequest.CONTROL_AWB_LOCK, true)
        if (cfg.highSpeed) {
            val hs = s as CameraConstrainedHighSpeedCaptureSession
            hs.setRepeatingBurst(hs.createHighSpeedRequestList(b.build()), captureCb, handler)
        } else {
            b.set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_VIDEO)
            b.set(CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE, CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE_OFF)
            s.setRepeatingRequest(b.build(), captureCb, handler)
        }
    }

    fun requestKeyframe() {
        try {
            encoder?.setParameters(Bundle().apply { putInt(MediaCodec.PARAMETER_KEY_REQUEST_SYNC_FRAME, 0) })
        } catch (_: Exception) {}
    }

    fun stop() {
        handler.post {
            try { session?.close() } catch (_: Exception) {}
            try { device?.close() } catch (_: Exception) {}
            try { encoder?.stop() } catch (_: Exception) {}
            try { encoder?.release() } catch (_: Exception) {}
            try { encoderSurface?.release() } catch (_: Exception) {}
            thread.quitSafely()
        }
    }

    private companion object {
        const val STILL_DEG = 0.5       // spread of gravity over 2 s below this: the phone stands still
        const val TURNED_DEG = 2.0      // pointing this much elsewhere than when metered: a new view
    }
}
