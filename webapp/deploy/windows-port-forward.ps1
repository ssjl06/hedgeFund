#Requires -RunAsAdministrator
<#
.SYNOPSIS
    Windows port forwarding & firewall setup for WSL2 CVaR Optimizer.

.DESCRIPTION
    WSL2 uses a NAT virtual network — external machines cannot reach it directly.
    This script:
      1. Finds the current WSL2 IP address
      2. Creates a port proxy  (Windows:80 → WSL:80)
      3. Opens Windows Firewall for inbound TCP 80

    Re-run this script every time WSL restarts (the WSL IP changes).

.USAGE
    Open PowerShell as Administrator, then:
      .\windows-port-forward.ps1           # setup
      .\windows-port-forward.ps1 -Remove   # teardown
#>

param(
    [switch]$Remove,
    [int]$Port = 80
)

$ErrorActionPreference = "Stop"
$ruleName = "CVaR-Optimizer-WSL2"

if ($Remove) {
    Write-Host "[*] Removing port proxy and firewall rule..." -ForegroundColor Yellow
    netsh interface portproxy delete v4tov4 listenport=$Port listenaddress=0.0.0.0 2>$null
    Remove-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue
    Write-Host "[OK] Cleaned up." -ForegroundColor Green
    exit 0
}

# Get WSL2 IP
$wslIp = (wsl hostname -I).Trim().Split(" ")[0]
if (-not $wslIp) {
    Write-Error "Could not determine WSL2 IP. Is WSL running?"
    exit 1
}
Write-Host "[*] WSL2 IP: $wslIp" -ForegroundColor Cyan

# Port proxy: Windows 0.0.0.0:80 → WSL:80
Write-Host "[*] Setting port proxy  0.0.0.0:$Port -> ${wslIp}:$Port" -ForegroundColor Cyan
netsh interface portproxy delete v4tov4 listenport=$Port listenaddress=0.0.0.0 2>$null
netsh interface portproxy add    v4tov4 listenport=$Port listenaddress=0.0.0.0 connectport=$Port connectaddress=$wslIp
netsh interface portproxy show   v4tov4

# Firewall rule
$existing = Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "[*] Firewall rule already exists, updating..." -ForegroundColor Yellow
    Set-NetFirewallRule -DisplayName $ruleName -LocalPort $Port -Protocol TCP -Action Allow -Direction Inbound
} else {
    Write-Host "[*] Creating firewall rule..." -ForegroundColor Cyan
    New-NetFirewallRule -DisplayName $ruleName -Direction Inbound -Protocol TCP -LocalPort $Port -Action Allow -Profile Any
}

# Show result
$winIp = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.InterfaceAlias -notmatch "Loopback|vEthernet" -and $_.IPAddress -ne "127.0.0.1" } | Select-Object -First 1).IPAddress
Write-Host ""
Write-Host "=============================================" -ForegroundColor Green
Write-Host " Setup complete!" -ForegroundColor Green
Write-Host " LAN access:  http://${winIp}:${Port}" -ForegroundColor Green
Write-Host " Local:       http://localhost:${Port}" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Green
Write-Host ""
Write-Host "NOTE: Re-run this script if you restart WSL." -ForegroundColor Yellow
