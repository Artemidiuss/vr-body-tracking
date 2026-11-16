// FBT.app: a native window (WKWebView) over the local FBT server (python -m fbtapp).
// The app starts the server as its child process, so macOS asks for the camera and
// local-network permissions on behalf of "FBT" (Info.plist), and quitting the
// window stops the hub. Built by macapp/build.sh.
import Cocoa
import WebKit

let port = 8765
let base = URL(string: "http://127.0.0.1:\(port)/")!

final class AppDelegate: NSObject, NSApplicationDelegate, NSWindowDelegate, WKNavigationDelegate {
    var window: NSWindow!
    var web: WKWebView!
    var proc: Process?
    var quitting = false
    var thermalTimer: Timer?
    let repo = (Bundle.main.object(forInfoDictionaryKey: "FBTRepo") as? String) ?? FileManager.default.currentDirectoryPath

    func applicationDidFinishLaunching(_ n: Notification) {
        buildMenu()
        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 1360, height: 860),
                          styleMask: [.titled, .closable, .miniaturizable, .resizable],
                          backing: .buffered, defer: false)
        window.title = "FBT"
        window.minSize = NSSize(width: 900, height: 600)
        window.backgroundColor = NSColor(red: 0.055, green: 0.063, blue: 0.075, alpha: 1)
        window.setFrameAutosaveName("FBTMain")
        if !window.setFrameUsingName("FBTMain") { window.center() }
        window.delegate = self
        web = WKWebView(frame: window.contentView!.bounds, configuration: WKWebViewConfiguration())
        web.autoresizingMask = [.width, .height]
        web.navigationDelegate = self
        web.setValue(false, forKey: "drawsBackground")
        window.contentView!.addSubview(web)
        showMessage("Запускаю FBT…", "Завантажую моделі, кілька секунд")
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        healthy { up in
            if up { self.web.load(URLRequest(url: base)) } else { self.startServer() }
        }
        thermalTimer = Timer.scheduledTimer(withTimeInterval: 5, repeats: true) { _ in self.postThermal() }
    }

    func buildMenu() {
        let main = NSMenu()
        let appItem = NSMenuItem(); main.addItem(appItem)
        let appMenu = NSMenu()
        appMenu.addItem(withTitle: "Сховати FBT", action: #selector(NSApplication.hide(_:)), keyEquivalent: "h")
        appMenu.addItem(.separator())
        appMenu.addItem(withTitle: "Вийти з FBT", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        appItem.submenu = appMenu
        let editItem = NSMenuItem(); main.addItem(editItem)
        let edit = NSMenu(title: "Редагування")
        edit.addItem(withTitle: "Відмінити", action: Selector(("undo:")), keyEquivalent: "z")
        edit.addItem(withTitle: "Вирізати", action: #selector(NSText.cut(_:)), keyEquivalent: "x")
        edit.addItem(withTitle: "Копіювати", action: #selector(NSText.copy(_:)), keyEquivalent: "c")
        edit.addItem(withTitle: "Вставити", action: #selector(NSText.paste(_:)), keyEquivalent: "v")
        edit.addItem(withTitle: "Виділити все", action: #selector(NSText.selectAll(_:)), keyEquivalent: "a")
        editItem.submenu = edit
        let viewItem = NSMenuItem(); main.addItem(viewItem)
        let view = NSMenu(title: "Вигляд")
        view.addItem(withTitle: "Оновити", action: #selector(reload), keyEquivalent: "r")
        view.addItem(withTitle: "На весь екран", action: #selector(NSWindow.toggleFullScreen(_:)), keyEquivalent: "f")
        viewItem.submenu = view
        NSApp.mainMenu = main
    }

    @objc func reload() { web.load(URLRequest(url: base)) }

    func startServer() {
        let logDir = repo + "/data/logs"
        try? FileManager.default.createDirectory(atPath: logDir, withIntermediateDirectories: true)
        let logPath = logDir + "/app.log"
        FileManager.default.createFile(atPath: logPath, contents: nil)
        let log = FileHandle(forWritingAtPath: logPath)
        let p = Process()
        p.executableURL = URL(fileURLWithPath: repo + "/.venv/bin/python")
        p.arguments = ["-u", "-m", "fbtapp", "--port", String(port), "--parent-watch"]
        p.currentDirectoryURL = URL(fileURLWithPath: repo)
        var env = ProcessInfo.processInfo.environment
        env["PYTHONUNBUFFERED"] = "1"
        p.environment = env
        p.standardOutput = log
        p.standardError = log
        p.terminationHandler = { pr in
            DispatchQueue.main.async {
                if !self.quitting {
                    let tail = (try? String(contentsOfFile: logPath, encoding: .utf8))?
                        .split(separator: "\n").suffix(12).joined(separator: "\n") ?? ""
                    self.showMessage("FBT зупинився", "Код \(pr.terminationStatus). Останні рядки журналу:\n\n" + tail)
                }
            }
        }
        do { try p.run() } catch {
            showMessage("Не вдалося запустити FBT", "\(error)\n\nОчікую Python тут: \(repo)/.venv/bin/python")
            return
        }
        proc = p
        waitUntilUp(deadline: Date().addingTimeInterval(90))
    }

    func waitUntilUp(deadline: Date) {
        healthy { up in
            if up { self.web.load(URLRequest(url: base)); return }
            if Date() > deadline || self.proc?.isRunning == false { return }
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) { self.waitUntilUp(deadline: deadline) }
        }
    }

    func healthy(_ done: @escaping (Bool) -> Void) {
        var req = URLRequest(url: base.appendingPathComponent("health"))
        req.timeoutInterval = 0.8
        URLSession.shared.dataTask(with: req) { data, resp, _ in
            let ok = (resp as? HTTPURLResponse)?.statusCode == 200
            DispatchQueue.main.async { done(ok) }
        }.resume()
    }

    func postThermal() {
        var req = URLRequest(url: base.appendingPathComponent("api/thermal"))
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = "{\"state\": \(ProcessInfo.processInfo.thermalState.rawValue)}".data(using: .utf8)
        req.timeoutInterval = 1
        URLSession.shared.dataTask(with: req).resume()
    }

    func showMessage(_ title: String, _ text: String) {
        let esc = { (s: String) in s.replacingOccurrences(of: "&", with: "&amp;").replacingOccurrences(of: "<", with: "&lt;") }
        let html = """
        <html><body style="margin:0;height:100vh;display:flex;align-items:center;justify-content:center;background:#0e1013;color:#e9ecf1;
        font:16px -apple-system,sans-serif"><div style="max-width:720px;padding:30px;text-align:center">
        <div style="font-size:30px;font-weight:700;margin-bottom:12px">\(esc(title))</div>
        <pre style="white-space:pre-wrap;color:#8b93a1;font:14px ui-monospace,monospace;text-align:left">\(esc(text))</pre></div></body></html>
        """
        web.loadHTMLString(html, baseURL: nil)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ s: NSApplication) -> Bool { true }

    func applicationWillTerminate(_ n: Notification) {
        quitting = true
        guard let p = proc, p.isRunning else { return }
        p.interrupt()                                  // SIGINT: the hub stops cleanly
        let until = Date().addingTimeInterval(3)
        while p.isRunning && Date() < until { usleep(50_000) }
        if p.isRunning { p.terminate() }
    }
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.setActivationPolicy(.regular)
app.run()
