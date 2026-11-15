"""OSCQuery service so VRChat sends us the headset pose.

VRChat (2023.3.1+) sends head and wrist tracking from the VR system only to
apps it discovers via OSCQuery whose address tree contains /tracking/vrsystem
(the user must enable "Allow Sending Head and Wrist VR Tracking OSC Data" in
Main Menu -> Tracking & IK). This is how SlimeVR gets the HMD on Quest.

We advertise over mDNS:
  _osc._udp     -> our OSC UDP port (VRChat sends packets here)
  _oscjson._tcp -> our HTTP port serving the OSCQuery JSON (Vidvox spec)
"""

from __future__ import annotations

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from zeroconf import ServiceInfo, Zeroconf


def local_ip(towards: str = "192.168.0.1") -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect((towards, 9))
        return s.getsockname()[0]
    finally:
        s.close()


VRSYSTEM_ENDPOINTS = ("/tracking/vrsystem/head/pose", "/tracking/vrsystem/leftwrist/pose",
                      "/tracking/vrsystem/rightwrist/pose")


def build_tree(endpoints=VRSYSTEM_ENDPOINTS, osc_type="ffffff"):
    """OSCQuery tree shaped like VRChat's own C# library serialises it
    (vrc-oscquery-lib, HeadAndWristsReceiver sample):
      - every node on the way is reachable through CONTENTS,
      - leaves carry TYPE "ffffff" and ACCESS 2 (write-only),
      - null fields are omitted (the C# parser is strict).
    VRChat looks endpoints up by EXACT FULL_PATH before asking ?HOST_INFO."""
    root = {"FULL_PATH": "/", "ACCESS": 0, "CONTENTS": {}}
    lookup = {"/": root}
    for ep in endpoints:
        node, path = root, ""
        parts = ep.strip("/").split("/")
        for i, part in enumerate(parts):
            path += "/" + part
            if path not in lookup:
                child = {"FULL_PATH": path, "ACCESS": 0}
                if i == len(parts) - 1:
                    child.update({"TYPE": osc_type, "ACCESS": 2})
                node.setdefault("CONTENTS", {})[part] = child
                lookup[path] = child
            node = lookup[path]
    return root, lookup


class OSCQueryReceiver:
    """Advertises an OSCQuery service wanting /tracking/vrsystem and receives
    whatever VRChat sends. `on_message(address, args, t_received)`."""

    def __init__(self, name: str = "FBT-Hub", on_message=None, ip: str | None = None,
                 towards: str = "192.168.31.1", extra_ports=(9001,)):
        self.name = name
        self.ip = ip or local_ip(towards)
        self.on_message = on_message or (lambda *a: None)
        self.counts: dict[str, int] = {}
        self.last: dict[str, tuple] = {}
        self.senders: set[str] = set()
        self.raw: dict[tuple, int] = {}      # (src ip, local port) -> datagrams
        self.parse_errors: list[str] = []

        # Raw UDP sockets (not python-osc's server) so nothing is dropped
        # silently: our advertised port, plus VRChat's default out port 9001.
        self.socks = []
        main = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        main.bind(("0.0.0.0", 0))
        self.osc_port = main.getsockname()[1]
        self.socks.append(main)
        for extra in extra_ports:
            try:
                sk = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sk.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sk.bind(("0.0.0.0", extra))
                self.socks.append(sk)
            except OSError as e:
                self.parse_errors.append(f"cannot bind {extra}: {e}")
        for sk in self.socks:
            threading.Thread(target=self._recv_loop, args=(sk,), daemon=True).start()

        tree, nodes = build_tree()
        host_info = {"NAME": name,
                     "EXTENSIONS": {"ACCESS": True, "CLIPMODE": False, "RANGE": True,
                                    "TYPE": True, "VALUE": True},
                     "OSC_IP": self.ip, "OSC_PORT": self.osc_port, "OSC_TRANSPORT": "UDP"}
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                path, _, query = self.path.partition("?")
                path = "/" + path.strip("/")
                body = host_info if "HOST_INFO" in self.path else nodes.get(path)
                if body is None:
                    data = b"OSC Path not found"
                    self.send_response(404)
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                data = json.dumps(body).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                outer.http_hits.append((time.time(), self.client_address[0], self.path,
                                        "local " + self.connection.getsockname()[0]))

            def log_message(self, *a):
                pass

        self.http_hits: list = []
        self.http = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
        self.http_port = self.http.server_address[1]
        threading.Thread(target=self.http.serve_forever, daemon=True).start()
        try:  # some OSCQuery clients reuse the HTTP port number for UDP
            sk = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sk.bind(("0.0.0.0", self.http_port))
            self.socks.append(sk)
            threading.Thread(target=self._recv_loop, args=(sk,), daemon=True).start()
        except OSError as e:
            self.parse_errors.append(f"cannot bind udp {self.http_port}: {e}")

        self.zc = Zeroconf(interfaces=[self.ip])
        addr = [socket.inet_aton(self.ip)]
        inst = f"{name}-{self.http_port}"
        self.services = [
            ServiceInfo("_oscjson._tcp.local.", f"{inst}._oscjson._tcp.local.", addresses=addr,
                        port=self.http_port, properties={"help": ""},
                        server=f"{inst}.local."),
            ServiceInfo("_osc._udp.local.", f"{inst}._osc._udp.local.", addresses=addr,
                        port=self.osc_port, properties={"help": ""},
                        server=f"{inst}.local."),
        ]
        for s in self.services:
            self.zc.register_service(s)

    def _recv_loop(self, sk):
        from pythonosc.osc_packet import OscPacket
        port = sk.getsockname()[1]
        while True:
            try:
                data, src = sk.recvfrom(65536)
            except OSError:
                return
            t = time.perf_counter()
            key = (src[0], port)
            self.raw[key] = self.raw.get(key, 0) + 1
            self.senders.add(src[0])
            try:
                for tm in OscPacket(data).messages:
                    m = tm.message
                    self.counts[m.address] = self.counts.get(m.address, 0) + 1
                    self.last[m.address] = tuple(m.params)
                    self.on_message(m.address, tuple(m.params), t)
            except Exception as e:  # keep the first few undecodable packets
                if len(self.parse_errors) < 5:
                    self.parse_errors.append(f"{e}: {data[:64]!r}")

    def close(self):
        for s in self.services:
            try:
                self.zc.unregister_service(s)
            except Exception:
                pass
        self.zc.close()
        for sk in self.socks:
            sk.close()
        self.http.shutdown()
