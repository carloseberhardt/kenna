#!/bin/bash
set -euo pipefail

# Install engram systemd user units and enable them.
# Run from the engram project directory: ./systemd/install.sh

UNIT_DIR="$HOME/.config/systemd/user"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Installing engram systemd units to $UNIT_DIR"
mkdir -p "$UNIT_DIR"

# Clean up old/deprecated units that may still be installed
for old_unit in engram-watch.path engram-debounce.timer; do
    if [ -f "$UNIT_DIR/$old_unit" ]; then
        systemctl --user disable --now "$old_unit" 2>/dev/null || true
        rm "$UNIT_DIR/$old_unit"
        echo "  Removed deprecated $old_unit"
    fi
done

# Copy unit files
for unit in engram-reconcile.timer engram-reconcile.service engram-settle.timer engram-settle.service; do
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

# Periodic reconcile timer (every 2 hours)
systemctl --user enable --now engram-reconcile.timer
echo "  Enabled engram-reconcile.timer"

# Daily settle timer
systemctl --user enable --now engram-settle.timer
echo "  Enabled engram-settle.timer"

echo ""
echo "Done. Verify with:"
echo "  systemctl --user list-timers --all | grep engram"
echo "  journalctl --user -u engram-reconcile.service -n 20"
echo ""
echo "To test reconcile manually:"
echo "  systemctl --user start engram-reconcile.service"
echo "  journalctl --user -u engram-reconcile.service -f"
echo ""
echo "To uninstall:"
echo "  systemctl --user disable --now engram-reconcile.timer engram-settle.timer"
echo "  rm ~/.config/systemd/user/engram-*.{timer,service}"
echo "  systemctl --user daemon-reload"
