package dev.fbt.cam

import android.os.SystemClock
import android.util.Log
import java.io.DataInputStream
import java.io.DataOutputStream
import java.net.ServerSocket
import java.net.Socket
import java.nio.ByteBuffer
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference

/** Wire format: see PROTOCOL.md. */
object Proto {
    const val HELLO = 0
    const val CONFIG = 1
    const val FRAME = 2
    const val PING = 3
    const val PONG = 4
    const val KEYFRAME_REQ = 5
    const val STATUS = 6
    val MAGIC = byteArrayOf('F'.code.toByte(), 'B'.code.toByte(), 'T'.code.toByte(), '1'.code.toByte())
}

class Packet(val type: Int, val flags: Int, val seq: Int, val ts: Long, val payload: ByteArray)

/**
 * TCP server, one viewer at a time (a new connection replaces the old one).
 * Frames go through a short queue; if the viewer can't keep up, the queue is
 * flushed and frames are skipped until the next keyframe, so latency stays
 * bounded instead of growing.
 */
class StreamServer(
    private val port: Int,
    private val clockRealtime: () -> Boolean,
    private val onKeyframeNeeded: () -> Unit,
    private val log: (String) -> Unit,
    private val status: () -> ByteArray = { ByteArray(0) },
) {
    @Volatile var hello: ByteArray = ByteArray(0)
    @Volatile var config: ByteArray? = null
    private val client = AtomicReference<Client?>(null)
    private var acceptThread: Thread? = null
    private var server: ServerSocket? = null
    @Volatile var sentFrames = 0L
    @Volatile var sentBytes = 0L
    @Volatile var dropped = 0L
    val connected: Boolean get() = client.get() != null

    fun start() {
        server = ServerSocket(port).apply { reuseAddress = true }
        acceptThread = Thread({
            while (!Thread.currentThread().isInterrupted) {
                val s = try { server!!.accept() } catch (e: Exception) { break }
                s.tcpNoDelay = true
                s.sendBufferSize = 256 * 1024
                client.getAndSet(Client(s))?.close()
                log("viewer connected: ${s.inetAddress.hostAddress}")
            }
        }, "accept").apply { start() }
    }

    fun stop() {
        try { server?.close() } catch (_: Exception) {}
        acceptThread?.interrupt()
        client.getAndSet(null)?.close()
    }

    /** The stream format changed (another camera or mode): the viewer must reconnect to get
     *  the new HELLO and CONFIG, otherwise it decodes the new frames with the old SPS/PPS. */
    fun dropViewer() { client.getAndSet(null)?.close() }

    fun sendFrame(data: ByteArray, ptsUs: Long, key: Boolean) {
        detectClock(ptsUs)
        client.get()?.offerFrame(Packet(Proto.FRAME, if (key) 1 else 0, 0, ptsUs, data))
    }

    // Some HALs (e.g. Galaxy A32) report timestamp source UNKNOWN but actually
    // stamp frames with elapsedRealtime (or vice versa). Pick whichever clock
    // the frame timestamps are actually close to.
    @Volatile private var useRealtime: Boolean? = null

    private fun detectClock(ptsUs: Long) {
        if (useRealtime != null) return
        val pts = ptsUs * 1000
        val dReal = kotlin.math.abs(SystemClock.elapsedRealtimeNanos() - pts)
        val dMono = kotlin.math.abs(System.nanoTime() - pts)
        useRealtime = dReal < dMono
        log("frame clock: ${if (useRealtime == true) "elapsedRealtime" else "nanoTime"} (HAL says ${if (clockRealtime()) "realtime" else "unknown"})")
    }

    fun resetClockDetection() { useRealtime = null }

    fun phoneClockNs(): Long =
        if (useRealtime ?: clockRealtime()) SystemClock.elapsedRealtimeNanos() else System.nanoTime()

    private inner class Client(val sock: Socket) {
        private val out = DataOutputStream(sock.getOutputStream().buffered(64 * 1024))
        private val q = ArrayBlockingQueue<Packet>(6)
        @Volatile private var waitKey = true
        @Volatile private var alive = true
        private var seq = 0
        private var lastStatus = 0L

        private val writer = Thread({
            try {
                write(Packet(Proto.HELLO, 0, 0, 0, hello))
                config?.let { write(Packet(Proto.CONFIG, 0, 0, 0, it)) }
                onKeyframeNeeded()
                while (alive) {
                    val p = q.poll(500, TimeUnit.MILLISECONDS) ?: continue
                    write(p)
                    if (p.type == Proto.FRAME) { sentFrames++; sentBytes += p.payload.size }
                }
            } catch (e: Exception) {
                Log.i("FBTCAM", "writer ended: $e")
            }
            close()
        }, "writer").apply { start() }

        private val reader = Thread({
            try {
                val inp = DataInputStream(sock.getInputStream().buffered())
                val hdr = ByteArray(24)
                while (alive) {
                    inp.readFully(hdr)
                    val bb = ByteBuffer.wrap(hdr)
                    bb.position(4)
                    val type = bb.get().toInt()
                    bb.position(12)
                    val ts = bb.long
                    val len = bb.int
                    if (len > 0) inp.skipBytes(len)
                    when (type) {
                        Proto.PING -> {
                            val now = phoneClockNs()
                            val pl = ByteBuffer.allocate(8).putLong(now).array()
                            // pongs bypass the frame queue
                            synchronized(out) { writeRaw(Packet(Proto.PONG, 0, 0, ts, pl)) }
                            // Battery, heat and gravity: on every ping (once a second), a few dozen bytes.
                            val t = SystemClock.elapsedRealtime()
                            if (t - lastStatus > 900) {
                                lastStatus = t
                                val st = try { status() } catch (e: Exception) { ByteArray(0) }
                                if (st.isNotEmpty()) synchronized(out) { writeRaw(Packet(Proto.STATUS, 0, 0, 0, st)) }
                            }
                        }
                        Proto.KEYFRAME_REQ -> onKeyframeNeeded()
                    }
                }
            } catch (e: Exception) {
                Log.i("FBTCAM", "reader ended: $e")
            }
            close()
        }, "reader").apply { start() }

        fun offerFrame(p: Packet) {
            if (waitKey) {
                if (p.flags and 1 == 0) { dropped++; return }
                waitKey = false
            }
            if (!q.offer(p)) {
                // Viewer is behind: flush and resync on the next keyframe.
                dropped += q.size + 1
                q.clear()
                waitKey = true
                onKeyframeNeeded()
            }
        }

        private fun write(p: Packet) = synchronized(out) { writeRaw(p) }

        private fun writeRaw(p: Packet) {
            out.write(Proto.MAGIC)
            out.writeByte(p.type)
            out.writeByte(p.flags)
            out.writeShort(0)
            out.writeInt(if (p.type == Proto.FRAME) seq++ else p.seq)
            out.writeLong(p.ts)
            out.writeInt(p.payload.size)
            out.write(p.payload)
            out.flush()
        }

        fun close() {
            if (!alive) return
            alive = false
            try { sock.close() } catch (_: Exception) {}
            client.compareAndSet(this, null)
            log("viewer disconnected")
        }
    }
}
