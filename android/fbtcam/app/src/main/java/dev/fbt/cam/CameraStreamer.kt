package dev.fbt.cam

import android.annotation.SuppressLint
import android.content.Context
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraConstrainedHighSpeedCaptureSession
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.params.OutputConfiguration
import android.hardware.camera2.params.SessionConfiguration
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaFormat
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.util.Range
import android.view.Surface
import java.util.concurrent.Executor

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
        if (cfg.highSpeed) {
            val hs = s as CameraConstrainedHighSpeedCaptureSession
            hs.setRepeatingBurst(hs.createHighSpeedRequestList(b.build()), null, handler)
            log("streaming ${cfg.width}x${cfg.height} @${cfg.fps} high-speed, cam ${cfg.cam.id}")
        } else {
            b.set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_VIDEO)
            b.set(CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE, CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE_OFF)
            s.setRepeatingRequest(b.build(), null, handler)
            log("streaming ${cfg.width}x${cfg.height} @${cfg.fps}, cam ${cfg.cam.id}")
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
}
