package dev.fbt.cam

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.os.BatteryManager
import android.os.PowerManager
import android.graphics.Color
import android.graphics.SurfaceTexture
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.Gravity
import android.view.Surface
import android.view.TextureView
import android.view.View
import android.view.WindowManager
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.LinearLayout
import android.widget.Spinner
import android.widget.TextView
import org.json.JSONObject
import java.net.NetworkInterface

/**
 * Minimal UI: pick camera + mode, the stream starts automatically and keeps
 * running while the app is open (screen stays on). The Mac connects to
 * <phone IP>:5000 or via `adb forward tcp:5000 tcp:5000`.
 */
class MainActivity : Activity() {
    private val port = 5000
    private lateinit var status: TextView
    private lateinit var logView: TextView
    private lateinit var camSpinner: Spinner
    private lateinit var modeSpinner: Spinner
    private lateinit var texture: TextureView
    private lateinit var previewBtn: Button
    private var showPreview = false
    private var cams: List<CamInfo> = emptyList()
    private var modes: List<StreamConfig> = emptyList()
    private var streamer: CameraStreamer? = null
    private var server: StreamServer? = null
    private val ui = Handler(Looper.getMainLooper())
    private val logLines = ArrayDeque<String>()
    private val prefs by lazy { getSharedPreferences("fbtcam", MODE_PRIVATE) }
    private val gravity by lazy { GravityMeter(this) }
    private var lastFrames = 0L
    private var lastBytes = 0L
    private var lastEnc = 0L

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        gravity.start()
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(Color.BLACK)
            setPadding(24, 48, 24, 24)
        }
        status = TextView(this).apply { textSize = 15f; setTextColor(Color.WHITE) }
        camSpinner = Spinner(this)
        modeSpinner = Spinner(this)
        texture = TextureView(this).apply { visibility = View.INVISIBLE }
        val dim = Button(this).apply {
            text = "Затемнити екран"
            setOnClickListener { toggleDim() }
        }
        // Preview costs the camera an extra output surface (and GPU work). It is
        // only needed while aiming the phone, so it is off by default.
        previewBtn = Button(this).apply {
            text = "Показати відео"
            setOnClickListener {
                showPreview = !showPreview
                text = if (showPreview) "Сховати відео" else "Показати відео"
                texture.visibility = if (showPreview) View.VISIBLE else View.INVISIBLE
                restart()
            }
        }
        logView = TextView(this).apply { textSize = 11f; setTextColor(Color.LTGRAY) }
        root.addView(status)
        root.addView(camSpinner)
        root.addView(modeSpinner)
        root.addView(texture, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f))
        root.addView(previewBtn)
        root.addView(dim)
        root.addView(logView)
        setContentView(root)

        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.CAMERA), 1)
        } else init()
    }

    override fun onRequestPermissionsResult(rc: Int, p: Array<out String>, r: IntArray) {
        if (r.firstOrNull() == PackageManager.PERMISSION_GRANTED) init() else log("потрібен дозвіл на камеру")
    }

    private fun init() {
        cams = CameraCatalog.list(this)
        if (cams.isEmpty()) { log("камер не знайдено"); return }
        camSpinner.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, cams.map { it.label })
        val savedCam = intent?.getStringExtra("cam") ?: prefs.getString("cam", null)
        // Default: the widest back camera.
        val defIdx = cams.indexOfFirst { it.id == savedCam }.takeIf { it >= 0 }
            ?: cams.indices.maxBy { cams[it].vfovPortrait }
        camSpinner.setSelection(defIdx)
        camSpinner.onItemSelectedListener = listener { fillModes(); restart() }
        modeSpinner.onItemSelectedListener = listener { restart() }
        texture.surfaceTextureListener = object : TextureView.SurfaceTextureListener {
            override fun onSurfaceTextureAvailable(s: SurfaceTexture, w: Int, h: Int) = restart()
            override fun onSurfaceTextureSizeChanged(s: SurfaceTexture, w: Int, h: Int) {}
            override fun onSurfaceTextureDestroyed(s: SurfaceTexture) = true
            override fun onSurfaceTextureUpdated(s: SurfaceTexture) {}
        }
        fillModes()
        startServer()
        ui.post(statsTick)
        restart()  // preview is off by default: start the camera without waiting for it
    }

    private fun listener(f: () -> Unit) = object : AdapterView.OnItemSelectedListener {
        private var first = true
        override fun onItemSelected(p: AdapterView<*>?, v: View?, pos: Int, id: Long) {
            if (first) { first = false; return }
            f()
        }
        override fun onNothingSelected(p: AdapterView<*>?) {}
    }

    private fun fillModes() {
        val cam = cams[camSpinner.selectedItemPosition]
        val list = mutableListOf<StreamConfig>()
        for (s in cam.sizes) list += StreamConfig(cam, s.width, s.height, 30, false, bitrateFor(s.width, s.height, 30))
        for ((s, r) in cam.highSpeed) list += StreamConfig(cam, s.width, s.height, r.upper, true, bitrateFor(s.width, s.height, r.upper))
        modes = list
        modeSpinner.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item,
            list.map { "${it.width}x${it.height} @${it.fps}" + if (it.highSpeed) " (high-speed)" else "" })
        // `adb shell am start ... --es mode 1280x720@120hs` overrides the saved mode.
        val saved = intent?.getStringExtra("mode") ?: prefs.getString("mode_${cam.id}", null)
        modeSpinner.setSelection(list.indexOfFirst { modeKey(it) == saved }.coerceAtLeast(
            list.indexOfFirst { it.width == 1280 && !it.highSpeed }.coerceAtLeast(0)))
    }

    private fun bitrateFor(w: Int, h: Int, fps: Int) = (w * h * fps * 0.12).toInt().coerceIn(2_000_000, 16_000_000)
    private fun modeKey(m: StreamConfig) = "${m.width}x${m.height}@${m.fps}${if (m.highSpeed) "hs" else ""}"

    private fun startServer() {
        server = StreamServer(port,
            clockRealtime = { streamer != null && modes.getOrNull(modeSpinner.selectedItemPosition)?.cam?.timestampRealtime == true },
            onKeyframeNeeded = { streamer?.requestKeyframe() },
            log = ::log,
            status = ::statusJson).also { it.start() }
    }

    /** Battery and heat for the Mac's status screen, and gravity for its calibration (STATUS packet). */
    private fun statusJson(): ByteArray {
        val b = applicationContext.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
            ?: return ByteArray(0)
        val level = b.getIntExtra(BatteryManager.EXTRA_LEVEL, -1)
        val scale = b.getIntExtra(BatteryManager.EXTRA_SCALE, 100).coerceAtLeast(1)
        val st = b.getIntExtra(BatteryManager.EXTRA_STATUS, -1)
        val charging = st == BatteryManager.BATTERY_STATUS_CHARGING || st == BatteryManager.BATTERY_STATUS_FULL
        val tempC = b.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, 0) / 10.0
        val thermal = if (Build.VERSION.SDK_INT >= 29)
            (getSystemService(POWER_SERVICE) as PowerManager).currentThermalStatus else -1
        return JSONObject().apply {
            put("battery", if (level >= 0) level * 100 / scale else -1)
            put("charging", charging)
            put("plugged", b.getIntExtra(BatteryManager.EXTRA_PLUGGED, 0) != 0)
            put("temp_c", tempC)
            put("thermal", thermal)
            gravity.json()?.let { put("gravity", it) }
        }.toString().toByteArray()
    }

    private fun restart() {
        if (modes.isEmpty()) return
        if (showPreview && !texture.isAvailable) return  // wait for the preview surface
        streamer?.stop()
        val base = modes[modeSpinner.selectedItemPosition]
        // Latency experiments via `am start --ez template_preview false --ez oprate_max false`
        val cfg = base.copy(
            previewTemplate = intent?.getBooleanExtra("template_preview", prefs.getBoolean("template_preview", true)) ?: true,
            maxOperatingRate = intent?.getBooleanExtra("oprate_max", prefs.getBoolean("oprate_max", true)) ?: true,
            exposureCompSteps = intent?.getIntExtra("ev", prefs.getInt("ev", 0)) ?: 0)
        prefs.edit().putInt("ev", cfg.exposureCompSteps).apply()
        prefs.edit().putBoolean("template_preview", cfg.previewTemplate).putBoolean("oprate_max", cfg.maxOperatingRate).apply()
        prefs.edit().putString("cam", cfg.cam.id).putString("mode_${cfg.cam.id}", modeKey(cfg)).apply()
        val st = if (showPreview) texture.surfaceTexture?.apply { setDefaultBufferSize(cfg.width, cfg.height) } else null
        server?.config = null
        server?.resetClockDetection()
        server?.hello = helloJson(cfg).toByteArray()
        server?.dropViewer()
        val noPreview = !showPreview || intent?.getBooleanExtra("nopreview", false) == true
        streamer = CameraStreamer(this, cfg, if (noPreview || st == null) null else Surface(st),
            onConfig = { server?.config = it },
            onFrame = { d, pts, key -> server?.sendFrame(d, pts, key) },
            log = ::log).also {
            it.clockNs = { server?.phoneClockNs() ?: System.nanoTime() }
            try { it.start() } catch (e: Exception) { log("не вдалося запустити: $e") }
        }
    }

    private fun helloJson(cfg: StreamConfig) = JSONObject().apply {
        put("protocol", 1)
        put("device", "${Build.MANUFACTURER} ${Build.MODEL}")
        put("camera_id", cfg.cam.id)
        put("width", cfg.width); put("height", cfg.height); put("fps", cfg.fps)
        put("high_speed", cfg.highSpeed)
        put("sensor_orientation", cfg.cam.sensorOrientation)
        put("focal_mm", cfg.cam.focalMm.toDouble())
        put("sensor_mm", org.json.JSONArray(listOf(cfg.cam.sensorW.toDouble(), cfg.cam.sensorH.toDouble())))
        put("active_px", org.json.JSONArray(listOf(cfg.cam.activeW, cfg.cam.activeH)))
        put("timestamp_source", if (cfg.cam.timestampRealtime) "realtime" else "monotonic")
    }.toString()

    private val statsTick = object : Runnable {
        override fun run() {
            val s = server
            val st = streamer
            if (s != null) {
                val fps = s.sentFrames - lastFrames
                val kbps = (s.sentBytes - lastBytes) * 8 / 1000
                lastFrames = s.sentFrames; lastBytes = s.sentBytes
                val enc = st?.encodedFrames ?: 0
                android.util.Log.i("FBTCAM", "stats encoded/s=${enc - lastEnc} sent/s=$fps kbps=$kbps dropped=${s.dropped} capture->encoded=${st?.medianEncodeLatencyMs()?.let { "%.0f".format(it) } ?: "-"}ms")
                lastEnc = enc
                status.text = "${ips().joinToString { "$it:$port" }}\n" +
                    (if (s.connected) "● Mac підключено · $fps к/с · $kbps кбіт/с · пропущено ${s.dropped}"
                     else "○ чекаю на Mac · закодовано ${st?.encodedFrames ?: 0}") +
                    (st?.medianEncodeLatencyMs()?.let { "\nзнімання→кодер: ${"%.0f".format(it)} мс" } ?: "")
            }
            ui.postDelayed(this, 1000)
        }
    }

    private fun ips(): List<String> = try {
        NetworkInterface.getNetworkInterfaces().toList().filter { it.isUp && !it.isLoopback }
            .flatMap { it.inetAddresses.toList() }
            .filter { it is java.net.Inet4Address }.map { it.hostAddress!! }
    } catch (_: Exception) { emptyList() }

    private var dimmed = false
    private fun toggleDim() {
        dimmed = !dimmed
        window.attributes = window.attributes.apply { screenBrightness = if (dimmed) 0.01f else -1f }
    }

    private fun log(s: String) {
        android.util.Log.i("FBTCAM", s)
        ui.post {
            logLines.addLast(s)
            while (logLines.size > 6) logLines.removeFirst()
            logView.text = logLines.joinToString("\n")
        }
    }

    override fun onDestroy() {
        gravity.stop()
        streamer?.stop()
        server?.stop()
        ui.removeCallbacks(statsTick)
        super.onDestroy()
    }
}
