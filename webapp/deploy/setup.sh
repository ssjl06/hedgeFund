#!/bin/bash
# ============================================================
# CVaR Portfolio Optimizer — WSL2 Production Deployment Script
# ============================================================
#
# This script sets up the full production stack inside WSL2:
#   1. Install system packages (nginx)
#   2. Install Python dependencies
#   3. Create log directories
#   4. Deploy Nginx config
#   5. Deploy systemd service
#   6. Start everything
#
# Usage:
#   cd /home/user/hedgeFund/webapp
#   sudo bash deploy/setup.sh
#
# After this script, run the PowerShell script on Windows side:
#   .\deploy\windows-port-forward.ps1
# ============================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m'

APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DEPLOY_DIR="$APP_DIR/deploy"

info()  { echo -e "${CYAN}[*]${NC} $1"; }
ok()    { echo -e "${GREEN}[OK]${NC} $1"; }
warn()  { echo -e "${YELLOW}[!]${NC} $1"; }
fail()  { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# Must be root
[[ $EUID -eq 0 ]] || fail "Run with sudo: sudo bash deploy/setup.sh"

echo ""
echo "=========================================="
echo " CVaR Portfolio Optimizer — Deployment"
echo "=========================================="
echo ""

# ----------------------------------------------------------
# 1. System packages
# ----------------------------------------------------------
info "Installing system packages..."
apt-get update -qq
apt-get install -y -qq nginx > /dev/null 2>&1
ok "nginx installed"

# ----------------------------------------------------------
# 2. Python dependencies
# ----------------------------------------------------------
info "Installing Python dependencies..."
pip install -q flask yfinance pandas numpy scipy plotly gunicorn 2>/dev/null || {
    warn "pip install failed — make sure your Python env is activated"
    warn "Continuing anyway..."
}
ok "Python dependencies checked"

# ----------------------------------------------------------
# 3. Log directory
# ----------------------------------------------------------
info "Creating log directory..."
mkdir -p /var/log/cvar-optimizer
chown "$(logname 2>/dev/null || echo user)":"$(logname 2>/dev/null || echo user)" /var/log/cvar-optimizer
ok "Log directory: /var/log/cvar-optimizer"

# ----------------------------------------------------------
# 4. Nginx configuration
# ----------------------------------------------------------
info "Deploying Nginx config..."

# Update the static files path in nginx config to match actual APP_DIR
sed "s|/home/user/hedgeFund/webapp|${APP_DIR}|g" "$DEPLOY_DIR/nginx.conf" \
    > /etc/nginx/sites-available/cvar-optimizer

# Disable default site, enable ours
rm -f /etc/nginx/sites-enabled/default
ln -sf /etc/nginx/sites-available/cvar-optimizer /etc/nginx/sites-enabled/cvar-optimizer

nginx -t 2>&1 || fail "Nginx config test failed!"
ok "Nginx configured"

# ----------------------------------------------------------
# 5. Systemd service
# ----------------------------------------------------------
info "Deploying systemd service..."

# Update paths in service file
sed -e "s|/home/user/hedgeFund/webapp|${APP_DIR}|g" \
    -e "s|User=user|User=$(logname 2>/dev/null || echo user)|g" \
    -e "s|Group=user|Group=$(logname 2>/dev/null || echo user)|g" \
    "$DEPLOY_DIR/cvar-optimizer.service" \
    > /etc/systemd/system/cvar-optimizer.service

systemctl daemon-reload
ok "Systemd service installed"

# ----------------------------------------------------------
# 6. Start services
# ----------------------------------------------------------
info "Starting services..."

systemctl enable --now cvar-optimizer 2>/dev/null || {
    warn "systemd might not be fully available in WSL2"
    warn "Falling back to direct process launch..."

    # Kill any existing gunicorn
    pkill -f "gunicorn.*cvar-optimizer" 2>/dev/null || true
    sleep 1

    # Start gunicorn in background
    cd "$APP_DIR"
    sudo -u "$(logname 2>/dev/null || echo user)" \
        gunicorn -c deploy/gunicorn.conf.py app:app --daemon 2>/dev/null || {
        warn "Gunicorn daemon start failed. Try manually:"
        warn "  cd $APP_DIR && gunicorn -c deploy/gunicorn.conf.py app:app"
    }
}

# Start/restart nginx
systemctl restart nginx 2>/dev/null || service nginx restart 2>/dev/null || nginx -s reload 2>/dev/null || {
    warn "Could not auto-start nginx. Start manually: sudo service nginx start"
}

ok "Services started"

# ----------------------------------------------------------
# Done
# ----------------------------------------------------------
WSL_IP=$(hostname -I | awk '{print $1}')

echo ""
echo -e "${GREEN}=========================================="
echo " Deployment complete!"
echo "==========================================${NC}"
echo ""
echo "  WSL internal:   http://localhost:80"
echo "  WSL IP:         http://${WSL_IP}:80"
echo ""
echo -e "${YELLOW}  NEXT STEP (on Windows PowerShell as Admin):${NC}"
echo "  cd $(wslpath -w "$DEPLOY_DIR" 2>/dev/null || echo "$DEPLOY_DIR")"
echo "  .\\windows-port-forward.ps1"
echo ""
echo "  Then other people on your LAN can access:"
echo "  http://<your-windows-ip>:80"
echo ""
echo -e "${CYAN}  Useful commands:${NC}"
echo "  sudo service nginx status"
echo "  sudo service nginx restart"
echo "  ps aux | grep gunicorn"
echo "  tail -f /var/log/cvar-optimizer/access.log"
echo "  tail -f /var/log/cvar-optimizer/error.log"
echo ""
