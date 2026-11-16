#!/bin/zsh
# Builds FBT.app and installs it to ~/Applications (Spotlight, Launchpad, Dock).
set -e
cd "$(dirname "$0")"
REPO="$(cd .. && pwd)"
BUILD=build/FBT.app
rm -rf build && mkdir -p "$BUILD/Contents/MacOS" "$BUILD/Contents/Resources"
swiftc -O main.swift -o "$BUILD/Contents/MacOS/FBT" -framework Cocoa -framework WebKit
sed "s|__REPO__|$REPO|" Info.plist > "$BUILD/Contents/Info.plist"
"$REPO/.venv/bin/python" make_icon.py build/AppIcon.iconset
iconutil -c icns build/AppIcon.iconset -o "$BUILD/Contents/Resources/AppIcon.icns"
codesign --force --deep -s - "$BUILD"
mkdir -p "$HOME/Applications"
rm -rf "$HOME/Applications/FBT.app"
cp -R "$BUILD" "$HOME/Applications/FBT.app"
echo "installed $HOME/Applications/FBT.app"
