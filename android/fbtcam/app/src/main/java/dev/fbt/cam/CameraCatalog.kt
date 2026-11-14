package dev.fbt.cam

import android.content.Context
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.util.Range
import android.util.Size
import kotlin.math.atan
import kotlin.math.max

/** One camera as this package sees it, with what we need for streaming. */
data class CamInfo(
    val id: String,
    val facing: String,
    val focalMm: Float,
    val sensorW: Float,
    val sensorH: Float,
    val vfovPortrait: Double,
    val sensorOrientation: Int,
    val activeW: Int,
    val activeH: Int,
    val timestampRealtime: Boolean,
    val sizes: List<Size>,                 // encoder-compatible sizes
    val highSpeed: List<Pair<Size, Range<Int>>>,
) {
    val label: String
        get() = "id $id · $facing · ${"%.0f".format(vfovPortrait)}°" +
            if (vfovPortrait > 85) " · ширококутна" else ""
}

object CameraCatalog {
    private val WANTED = listOf(Size(1280, 720), Size(1920, 1080), Size(640, 480))

    fun list(ctx: Context): List<CamInfo> {
        val cm = ctx.getSystemService(CameraManager::class.java)
        val ids = cm.cameraIdList.toMutableList()
        // Hidden ids answer getCameraCharacteristics() only for allow-listed packages.
        for (n in 0..99) {
            val id = n.toString()
            if (id in ids) continue
            try { cm.getCameraCharacteristics(id); ids += id } catch (_: Exception) {}
        }
        return ids.mapNotNull { id ->
            try { describe(cm.getCameraCharacteristics(id), id) } catch (_: Exception) { null }
        }.filter { it.facing == "back" && it.focalMm > 0 }
            .distinctBy { "${it.focalMm}-${it.sensorW}" } // drop logical duplicates
    }

    private fun describe(c: CameraCharacteristics, id: String): CamInfo {
        val facing = when (c.get(CameraCharacteristics.LENS_FACING)) {
            CameraCharacteristics.LENS_FACING_BACK -> "back"
            CameraCharacteristics.LENS_FACING_FRONT -> "front"
            else -> "ext"
        }
        val f = c.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)?.firstOrNull() ?: 0f
        val s = c.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)
        val long = s?.let { max(it.width, it.height) } ?: 0f
        val active = c.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE)
        val map = c.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
        val surfaceSizes = map?.getOutputSizes(android.media.MediaRecorder::class.java)?.toSet() ?: emptySet()
        val hs = mutableListOf<Pair<Size, Range<Int>>>()
        val caps = c.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES) ?: IntArray(0)
        if (CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_CONSTRAINED_HIGH_SPEED_VIDEO in caps && map != null) {
            for (size in map.highSpeedVideoSizes) {
                for (r in map.getHighSpeedVideoFpsRangesFor(size)) {
                    if (r.lower == r.upper && r.upper >= 60) hs += size to r
                }
            }
        }
        return CamInfo(
            id = id, facing = facing, focalMm = f,
            sensorW = s?.width ?: 0f, sensorH = s?.height ?: 0f,
            vfovPortrait = if (f > 0) Math.toDegrees(2 * atan(long / 2.0 / f)) else 0.0,
            sensorOrientation = c.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 90,
            activeW = active?.width() ?: 0, activeH = active?.height() ?: 0,
            timestampRealtime = c.get(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE) ==
                CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME,
            sizes = WANTED.filter { it in surfaceSizes },
            highSpeed = hs.distinctBy { "${it.first}-${it.second}" },
        )
    }
}
