#!/bin/bash
set -euo pipefail

# Install engram systemd user units and enable them.
# Run from the engram project directory: ./systemd/install.sh

UNIT_DIR="$HOME/.config/systemd/user"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Installing engram systemd units to $UNIT_DIR"
mkdir -p "$UNIT_DIR"

# Copy unit files
for unit in engram-watch.path engram-debounce.timer engram-reconcile.service engram-settle.timer engram-settle.service; do
    cp "$SCRIPT_DIR/$unit" "$UNIT_DIR/$unit"
    echo "  Installed $unit"
done

# Install the binary
echo "Installing engram binary..."
cargo install --path "$(dirname "$SCRIPT_DIR")" --locked 2>/dev/null || {
    echo "  cargo install failed, copying release binary directly"
    cp "$(dirname "$SCRIPT_DIR")/target/release/engram" "$HOME/.cargo/bin/engram"
}
echo "  Binary: $(which engram || echo "$HOME/.cargo/bin/engram")"

# Reload and enable
systemctl --user daemon-reload

# File watcher (triggers reconcile via debounce timer)
systemctl --user enable --now engram-watch.path
echo "  Enabled engram-watch.path"

# Daily settle timer
systemctl --user enable --now engram-settle.timer
echo "  Enabled engram-settle.timer"

echo ""
echo "Done. Verify with:"
echo "  systemctl --user status engram-watch.path"
echo "  systemctl --user list-timers --all | grep engram"
echo ""
echo "To test reconcile manually:"
echo "  systemctl --user start engram-reconcile.service"
echo "  journalctl --user -u engram-reconcile.service -f"
echo ""
echo "To uninstall:"
echo "  systemctl --user disable --now engram-watch.path engram-settle.timer"
echo "  rm ~/.config/systemd/user/engram-*.{path,timer,service}"
echo "  systemctl --user daemon-reload"
