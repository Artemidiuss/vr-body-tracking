# FBT: VRChat full-body tracking from two phones

**English** · [Українська](README.uk.md)

Foot, hip, knee and chest trackers for VRChat on **Quest 3 (standalone)** without real trackers. All you need is two old Android phones and a MacBook.

- The phones film you from two sides and stream the video to the Mac.
- The Mac finds body keypoints in every frame with a neural network, builds a 3D skeleton from the two views and sends trackers to VRChat over OSC.
- The MacBook's own camera helps as a third view.
- When VRChat sends the headset pose, the trackers are anchored to the real headset instead of the head seen by the cameras.

The full history of decisions, measurements and rejected options is in [ARCHITECTURE.md](ARCHITECTURE.md) (in Ukrainian). This file covers how to run it and how to adapt it to your own setup.

> The app's interface is in Ukrainian. Button and setting names below are given in English with the Ukrainian label in brackets. All UI text lives in `fbtapp/ui/` if you want to translate it.

---

## What you need

| | Used here | Can be replaced by |
|---|---|---|
| Computer | MacBook Air M3, 8 GB | Any Apple Silicon Mac. Other OSes: see [Not a Mac](#not-a-mac) |
| Cameras | Redmi 12 and Galaxy A32 (ultra-wide rear cameras) | Any two Android 9+ phones with Camera2 and hardware H.264 |
| Third camera (optional) | the MacBook's built-in camera | any camera macOS can see |
| Headset | Quest 3, VRChat standalone | PC VR should work too (same OSC), but is untested |
| Network | all devices on one Wi-Fi | 5 GHz preferred; phones can also use a cable (`adb forward`) |

Software: Python 3.11+ (3.12 here), Xcode Command Line Tools (`swiftc`), Android SDK + JDK 17 (only to build the phone app).

---

## How it works

```
 Phone A ───┐  H.264 + capture time of every frame (TCP :5000)
 Phone B ───┼──────────────► Mac ──────────────────────────────► Quest / VRChat
 Mac camera ┘                 │  1. 2D pose: YOLOX-tiny + RTMPose-m        OSC :9000
                              │     (ONNX Runtime on the GPU via CoreML)   /tracking/trackers/...
                              │  2. frames matched by capture time
                              │  3. triangulation (DLT) with sanity checks
                              │  4. stabiliser: outlier gate, bone lengths,
                              │     One-Euro filter
                              │  5. trackers (position + rotation)
                              ◄─────────────── headset pose (OSCQuery, UDP :9101)
```

**Phone app (FBT Cam, `android/fbtcam`).**
- Camera2 → hardware H.264 → a TCP server on port 5000.
- Every frame carries its sensor capture time. Once a second the Mac syncs its clock with the phone (ping-pong), so it knows the true moment of every frame. That is how the two phones' frames are matched without hardware sync.
- Every 5 s the phone also sends its battery level and temperature.
- The protocol is described in [PROTOCOL.md](android/fbtcam/PROTOCOL.md).

**Mac (`fbt/`, `fbtapp/`, `macapp/`).**
- `macapp/` is the native FBT.app window (WKWebView). It starts the `fbtapp` Python server and shows its interface. A separate app is needed so that macOS grants camera and local network permission to FBT itself.
- `fbtapp/server.py` is a local server on `127.0.0.1:8765`: the page, camera video with the skeleton, state, settings.
- `fbt/engine.py` is the core: self-reconnecting cameras, calibration, the tracking loop, output to VRChat, logs.

**Calibration** (30 s, the Calibrate button).
- The phones' positions are found from your body, no checkerboard: the same body points in two views → essential matrix → bundle adjustment.
- The scale comes from your height.
- The Mac camera is calibrated separately (PnP), together with its focal length and clock offset.
- Your bone lengths are measured too; they then keep the skeleton steady.

**Headset anchor.**
- VRChat can send the headset pose over the network (OSCQuery).
- FBT learns the transform between its own space and VRChat's, and sends the trackers already in VRChat space. The head is the real headset.
- A T-pose sets the rotation at once; after that the link is refined slowly in the background.
- Without headset data everything works from the head seen by the cameras.

---

## Installation

### 1. Python side

```bash
git clone <this repository> fbt && cd fbt
python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest -q
```

- `rtmlib` downloads the models (RTMPose-m Halpe26 and YOLOX-tiny) on first run into `~/.cache/rtmlib`.
- FBT makes fixed-shape copies of them in `models/`, because otherwise CoreML silently runs on the CPU, 2–5× slower.
- The first start takes about a minute; after that everything comes from the cache.

### 2. The FBT app for the Mac

```bash
macapp/build.sh
```

- Builds `FBT.app` and installs it to `~/Applications`, so it is available from Spotlight, Launchpad and the Dock.
- The repository path is baked into the app, so don't move the folder afterwards (or rebuild).
- On first launch macOS asks for camera and local network permission. Allow both.
- Without the app: `.venv/bin/python -m fbtapp --browser` opens the same interface in your browser. In that case, give the camera permission to Terminal.

### 3. FBT Cam on the phones

Enable Developer options → USB debugging on the phones and connect them with a cable:

```bash
cd android/fbtcam
./gradlew assembleDebug
~/Library/Android/sdk/platform-tools/adb install -r app/build/outputs/apk/plain/debug/app-plain-debug.apk
```

There are three build variants (flavors):

| Variant | Package name | Use for |
|---|---|---|
| `plain` | `dev.fbt.cam` | any phone, start with this one |
| `samsung` | `com.samsung.android.scan3d` | Samsung |
| `xiaomi` | `com.xiaomi.cameratest` | Xiaomi / Redmi |

Why `samsung` and `xiaomi` exist:
- Some vendors expose the ultra-wide camera only to apps with "their own" package names. Under any other name the app sees only the main camera.
- The ultra-wide camera sees your whole body from closer, so it is the better choice.
- These builds are for your own phones only. Don't publish them in any store: the package name belongs to someone else.

Install the APK you need (`app/build/outputs/apk/<variant>/debug/`), open FBT Cam and pick:
- **camera:** the widest one, marked «ширококутна» (ultra-wide);
- **mode:** 640×480, 30 fps. A higher resolution adds no quality and heats the phone more (measurements in [ARCHITECTURE.md §25](ARCHITECTURE.md)).

The app starts streaming right away. You can turn the phone's screen off; just don't close the app.

### 4. Network

- The Mac, both phones and the Quest on one network.
- Reserve fixed IPs for the phones and the Quest in your router (DHCP reservation); otherwise you'll have to change them in the settings after a reboot.
- FBT Cam shows the phone's IP. Quest IP: Settings → Wi-Fi → your network.

### 5. VRChat

In VRChat (Action Menu → Options → OSC):
- **Enabled**: turn on.
- **Allow Sending Head and Wrist VR Tracking OSC Data**: turn on for the headset anchor.

Things to know:
- VRChat sends the headset pose only to a service it discovered **after** this option was turned on. So turn the option on first, then start FBT.
- If FBT says headset data isn't coming («дані з шолома не йдуть»), restart VRChat.
- Without this option the trackers still work, just anchored to the head seen by the cameras.

---

## Camera placement

```
             table
   [Phone A]   [MacBook]   [Phone B]
        \          |          /
         \         |         /
          \    play area    /
           \     (you)     /
```

- **Phones:** front left and front right, ~2 m apart, 60–90° between their directions. About 1.1 m high, upright.
- **Framing:** each phone must see all of you, head to feet, everywhere you play.
- **MacBook:** in the middle, lid tilted so the camera sees you. It helps most when sitting: 9× fewer lost leg points.
- **Light:** the brighter the room, the less motion blur on your legs.
- **Don't move the cameras after calibrating.** If you did, calibrate again.

---

## Usage

1. Open FBT Cam on both phones and FBT on the Mac. The top bar shows which cameras are connected and their battery.
2. **First time:** ⚙︎ → enter your height, the phones' IPs and the Quest IP.
3. **Calibrate** («Калібрувати»): once, and again whenever you move a camera. For 30 seconds walk calmly around the whole play area:
   - closer and further, left and right;
   - crouch, raise your arms up and to the sides;
   - stand straight for a few seconds.

   Both phones must see all of you the whole time. Don't freeze, but don't rush either.
4. **Start game** («Почати гру»): a big status screen, readable from across the room.
5. **In VRChat** stand in a T-pose and press Calibrate FBT. At the same moment FBT aligns the trackers from that T-pose.
6. Play. If the trackers drift, do T-pose + Calibrate FBT again.

### Settings (⚙︎)

Everything applies immediately, even mid-game. Saved in `data/app_settings.json`.

| Setting | What it does |
|---|---|
| Chest / Knees / Elbows | extra trackers. Hip and feet are always on |
| Feet always flat | steadier, but no tiptoes |
| Body smoothness | lower is smoother, but lags more |
| Body motion prediction | hides latency but can overshoot. Default 0 |
| Trackers forward / back | if the tracker cubes in game sit off your body |
| Head prediction / smoothness | for the no-headset mode: keeps the legs from lagging when you lean |
| Rotation trim | if your body in game is slightly turned |
| MacBook camera rate | 10 fps is enough; more just heats the Mac with no visible gain |
| Anchor to the headset | use the headset pose when VRChat sends it |
| MacBook camera | the third view on or off |
| Voice alerts | says in game when a phone disconnects or runs low |

---

## Adapting it to your setup

### Other phones

- Build `plain` and see which cameras FBT Cam lists.
- If no ultra-wide camera is listed, your vendor hides it. Then either use the main camera (and place the phones further away), or add a flavor with a package name your vendor lets use that camera (see `android/fbtcam/app/build.gradle.kts`). `android/camprobe` probes all hidden camera ids, which helps here.
- Focal length and sensor size come from the phone; calibration refines them.
- To check latency and frame rate: `.venv/bin/python -m tools.fbtcam_check <ip1> <ip2>`. Here the Redmi gives ~72 ms from capture to the Mac, the Samsung ~111 ms.

### Another camera instead of a phone

- The Mac accepts sources given as strings in `cameras` (`data/app_settings.json`):
  - a phone's IP running FBT Cam;
  - `mac` or `mac:<device number>`: a camera macOS can see (`mac@1920x1080` to set the size);
  - `replay:<recording folder>/camN`: a recording instead of a live camera, for tests.
- New camera type: write a source class modelled on `fbt/maccam.py` or `fbt/fbtcam.py`. The key thing is the **true capture time** of every frame on the `time.perf_counter` clock; without it the sync falls apart.
- Source strings are parsed in `open_source()` (`fbt/maccam.py`).

### Number of cameras

- The interface is built for 2 phones + an optional Mac camera.
- The phones set the frame moments; the third camera only helps (its pose is interpolated to the same moment).
- The triangulation core handles more cameras (`fbt/triangulate.py`), but calibration and the interface would need extending.

### Another pose model

- `fbt/pose.py`, `make_backend("rtmpose-m:gpu")`. Devices: `cpu`, `gpu`, `ane`, `all`.
- Also available: `rtmpose-x` (more accurate, slower) and `mediapipe-full` (weaker on our recordings).
- Heavier models on the M3 gave fewer frames and no better result (comparison in [ARCHITECTURE.md §21](ARCHITECTURE.md)).
- Keypoints use the Halpe26 format (it has heels and toes, which matters for foot rotation).

### Different height / avatar

- Height is in the settings; it sets the scale.
- If your avatar's feet are closer together or wider apart than yours (a chibi, for example), that's the avatar's proportions: do Calibrate FBT in VRChat standing the way the avatar stands.

### Not a Mac

- The core (`fbt/`) is plain Python: OpenCV, ONNX Runtime, SciPy. On Linux/Windows it should work with `rtmpose-m:cpu` and `python -m fbtapp --browser`. Untested.
- macOS-specific parts:
  - `macapp/` (the app window);
  - `fbt/maccam.py` (camera via AVFoundation);
  - Mac battery and heat (`pmset`), `caffeinate`, voice (`say`);
  - CoreML acceleration.
- On the CPU the pose runs slower (~17 ms instead of ~5 ms on the M3), so you need a fast processor.

### Another game / headset

- The output is standard VRChat OSC tracking (`fbt/osc_out.py`, port 9000): `/tracking/trackers/N/position|rotation` plus the head for alignment.
- For PC VR set `quest_ip` to `127.0.0.1` (or the PC's IP). The headset pose should come through OSCQuery there too, but this is untested.

---

## Layout

```
fbt/            core
  engine.py       cameras, calibration, tracking, output, logs (used by FBT.app)
  fbtcam.py       receiving video from FBT Cam, clock sync
  maccam.py       Mac camera (AVFoundation via PyAV) and source-string parsing
  sources.py      base frame source, recording playback
  pose.py         2D pose models, the CoreML fix
  calibrate.py    calibration from the body (phone pair + third camera)
  triangulate.py  triangulation with checks (left/right mix-ups, bone lengths)
  filters.py      skeleton stabiliser, One-Euro, bone-length learning
  trackers.py     tracker positions and rotations, T-pose
  hmd_align.py    headset anchor
  oscquery.py     OSCQuery service (headset pose from VRChat)
  osc_out.py      OSC to VRChat
fbtapp/         local server and interface (ui/)
macapp/         native FBT.app window, build.sh
android/fbtcam  FBT Cam phone app, PROTOCOL.md
android/camprobe  diagnostics: which cameras a phone exposes, and to whom
tools/          recording, offline calibration, measurements, checks
tests/          pytest, no cameras or headset needed
data/           (not in git) calibration, settings, recordings, logs
models/         (not in git) models and the CoreML cache
```

### Useful tools

```bash
.venv/bin/python -m tools.fbtcam_check 192.168.1.20 192.168.1.21                   # phone frame rate and latency
.venv/bin/python -m tools.record 192.168.1.20 192.168.1.21 mac --seconds 30        # synced recording (from Terminal.app if mac is included)
.venv/bin/python -m tools.calibrate --rec data/rec_X --height 1.75 --out calib.json # offline calibration of a recording
.venv/bin/python -m tools.eval_recording --rec data/rec_X --calib calib.json        # 3D quality on a recording
.venv/bin/python -m tools.hmd_probe                                                # is VRChat sending the headset pose
.venv/bin/python -m tools.fake_vrchat                                              # a "VRChat" on the Mac: what exactly we send
.venv/bin/python -m fbtapp --port 8766 --data <copy of data/> \
    --cameras replay:data/rec_X/cam0,replay:data/rec_X/cam1,replay:data/rec_X/cam2  # FBT on a recording instead of cameras
```

Each tool has a description at the top of its file (`tools/*.py`).

---

## Troubleshooting

| Symptom | What to do |
|---|---|
| A phone is offline («не в мережі») | Is FBT Cam open? Same IP? Phone and Mac on the same network? |
| Garbled video after changing the mode in FBT Cam | old FBT Cam version: update it, or restart FBT |
| "Resolution changed, calibrate again" («Змінилась роздільність») | the FBT Cam mode differs from the one you calibrated with: switch it back or recalibrate |
| "Headset data isn't coming" («дані з шолома не йдуть») | Is the Head and Wrist Tracking option on in VRChat? Restart VRChat (it sticks to the first service it found) |
| "Cameras seem to have moved" («камери зрушили») | calibrate again |
| Legs lag | a Mac on low battery slows down a lot: plug it in |
| Tracker cubes sit in front of / behind you | the Trackers forward / back slider |
| Trackers are turned | T-pose + Calibrate FBT in VRChat |
| Mac camera doesn't work | System Settings → Privacy & Security → Camera: allow FBT. After rebuilding FBT.app macOS may ask again |

Game logs are in `data/logs/` (last 5 sessions). Calibration recordings are in `data/rec_*`.

---

## Credits

- [rtmlib](https://github.com/Tau-J/rtmlib) and the [RTMPose](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose) / YOLOX models by OpenMMLab (Apache-2.0).
- [Mediapipe-VR-Fullbody-Tracking](https://github.com/ju1ce/Mediapipe-VR-Fullbody-Tracking): axis conversions for VRChat OSC.
- VRChat's documentation on OSC trackers and OSCQuery.

---

## License

[PolyForm Noncommercial 1.0.0](LICENSE.md). You may use, change and share this code for any noncommercial purpose: personal use, hobby projects, study, research. Commercial use is not allowed. If you share it or a changed version, keep the license and the `Required Notice` line.

The pose models and libraries this project downloads and uses (rtmlib, RTMPose, YOLOX, ONNX Runtime and others) keep their own licenses.
