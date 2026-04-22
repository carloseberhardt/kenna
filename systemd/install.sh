#!/bin/bash
set -euo pipefail

# Install kenna systemd user units and enable them.
# Run from the kenna project directory: ./systemd/install.sh

UNIT_DIR="$HOME/.config/systemd/user"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Installing kenna systemd units to $UNIT_DIR"
mkdir -p "$UNIT_DIR"

# Clean up old/deprecated units that may still be installed
for old_unit in kenna-watch.path kenna-debounce.timer; do
    if [ -f "$UNIT_DIR/$old_unit" ]; then
        systemctl --user disable --now "$old_unit" 2>/dev/null || true
        rm "$UNIT_DIR/$old_unit"
        echo "  Removed deprecated $old_unit"
    fi
done

# Copy unit files
for unit in kenna-reconcile.timer kenna-reconcile.service kenna-settle.timer kenna-settle.service; do
    cp "$SCRIPT_DIR/$unit" "$UNIT_DIR/$unit"
    echo "  Installed $unit"
done

# Install the binary
echo "Installing kenna binary..."
cargo install --path "$(dirname "$SCRIPT_DIR")" --locked 2>/dev/null || {
    echo "  cargo install failed, copying release binary directly"
    cp "$(dirname "$SCRIPT_DIR")/target/release/kenna" "$HOME/.cargo/bin/kenna"
}
echo "  Binary: $(which kenna || echo "$HOME/.cargo/bin/kenna")"

# Reload and enable
systemctl --user daemon-reload

# Periodic reconcile timer (every 2 hours)
systemctl --user enable --now kenna-reconcile.timer
echo "  Enabled kenna-reconcile.timer"

# Daily settle timer
systemctl --user enable --now kenna-settle.timer
echo "  Enabled kenna-settle.timer"

echo ""
echo "Done. Verify with:"
echo "  systemctl --user list-timers --all | grep kenna"
echo "  journalctl --user -u kenna-reconcile.service -n 20"
echo ""
echo "To test reconcile manually:"
echo "  systemctl --user start kenna-reconcile.service"
echo "  journalctl --user -u kenna-reconcile.service -f"
echo ""
echo "To uninstall:"
echo "  systemctl --user disable --now kenna-reconcile.timer kenna-settle.timer"
echo "  rm ~/.config/systemd/user/kenna-*.{timer,service}"
echo "  systemctl --user daemon-reload"
