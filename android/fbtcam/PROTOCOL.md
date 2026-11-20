# FBT Cam stream protocol v1

One TCP connection per viewer, phone is the server (default port **5000**).
Works over Wi-Fi and over USB (`adb forward tcp:5000 tcp:5000`), because
everything — video, metadata and clock sync — goes through the same socket.

## Packet framing (both directions)

24-byte header, big-endian, followed by `len` payload bytes:

| offset | size | field |
|---|---|---|
| 0 | 4 | magic `FBT1` |
| 4 | 1 | type |
| 5 | 1 | flags (bit0 = keyframe) |
| 6 | 2 | reserved |
| 8 | 4 | seq (per type, wraps) |
| 12 | 8 | ts — meaning depends on type |
| 20 | 4 | len |

## Types

| type | dir | ts | payload |
|---|---|---|---|
| 0 HELLO | phone→Mac | 0 | UTF-8 JSON: device, camera id, width/height, fps, sensor orientation, focal length mm, sensor size mm, active array px, timestamp source |
| 1 CONFIG | phone→Mac | 0 | H.264 SPS+PPS, Annex-B |
| 2 FRAME | phone→Mac | sensor capture time, µs, phone clock | H.264 access unit, Annex-B |
| 3 PING | Mac→phone | Mac time, ns | empty |
| 4 PONG | phone→Mac | Mac time echoed, ns | 8 bytes: phone clock, ns (same clock as FRAME ts) |
| 5 KEYFRAME_REQ | Mac→phone | 0 | empty |
| 6 STATUS | phone→Mac | 0 | UTF-8 JSON: battery (%), charging, plugged, temp_c (battery), thermal (Android thermal status 0–6, −1 unknown), gravity {v: [x, y, z] m/s² mean of the last 2 s in device axes (X right, Y up, Z out of the screen; points up), sd_deg: spread of the samples, n}. exposure {ms, iso, held}: the current exposure and whether it is held. stream {enc_fps, sent_fps, enc_ms, dropped}: frames out of the encoder / onto the socket in the last second, median capture -> encoded latency (ms, phone clock), frames skipped so far; the hub's capture -> received latency minus enc_ms is the Wi-Fi. Sent on every PING (once a second). Older receivers ignore it |
| 7 EXPOSURE | Mac→phone | 0 | empty. Meter the light again (auto exposure for ~3 s), then hold it: the same brightness (median exposure × ISO of the last 1.5 s) at 10/20/30 ms, white balance held too. The phone also does this by itself when the camera starts and when the phone was turned (gravity direction > 2° from where it pointed when metered, then still). Older phones ignore it |

## Clock sync

Mac sends PING every second. offset = mac_mid − phone_time, taking the
sample with the smallest round trip. Then `mac_time = frame_ts + offset`,
which gives the true capture time of every frame on the Mac clock: it is
used to synchronise cameras and to measure latency continuously.

## Latency rules

- Encoder: H.264 baseline (no B-frames), keyframe every 1 s, realtime priority.
- Phone never queues more than a few frames per viewer; if the viewer falls
  behind it drops frames until the next keyframe and forces one.
- Frames are sent landscape as the sensor produces them; rotation to portrait
  is done on the Mac (`sensor_orientation` in HELLO).

## Standby

The camera runs only while a viewer is connected, or while the video is shown on the phone (aiming it). With no viewer for 10 s the camera and the
encoder stop and the screen goes almost dark (the app stays in front, screen on). The next
connection gets HELLO at once; the camera starts (~1-2 s) and the new encoder's CONFIG goes out
before its first FRAME. Viewers keep reconnecting as before (the hub retries every ~2 s and drops
a connection with no frames for 5 s, which the restart stays well within).
