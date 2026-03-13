#!/usr/bin/env bash
# Setup script for raspberry-claudio: install Akida driver + SDK
# Run this FROM raspberry-paolo (or any machine that can SSH to both).
#
# Prerequisites:
#   - raspberry-paolo has the Akida driver loaded and SDK installed
#   - Both Pis connected via direct Ethernet (10.0.0.1 / 10.0.0.2)
#   - SSH key-based auth configured between the machines
#
# Usage:
#   ssh admin@10.0.0.2 'bash -s' < setup_claudio.sh
# Or run directly on paolo:
#   bash setup_claudio.sh

set -euo pipefail

CLAUDIO_IP="10.0.0.1"
CLAUDIO_USER="admin"
KERNEL="6.12.47+rpt-rpi-2712"
DRIVER_SRC="/lib/modules/${KERNEL}/kernel/drivers/akida-pcie.ko"

echo "=== Step 1: Copy Akida driver from paolo to claudio ==="
scp "${DRIVER_SRC}" "${CLAUDIO_USER}@${CLAUDIO_IP}:/tmp/akida-pcie.ko"

echo "=== Step 2: Install driver on claudio ==="
ssh "${CLAUDIO_USER}@${CLAUDIO_IP}" bash <<'REMOTE_EOF'
set -euo pipefail

KERNEL="6.12.47+rpt-rpi-2712"
DRIVER_DEST="/lib/modules/${KERNEL}/kernel/drivers/akida-pcie.ko"

echo "Copying driver module ..."
sudo cp /tmp/akida-pcie.ko "${DRIVER_DEST}"

echo "Running depmod ..."
sudo depmod -a

echo "Loading akida_pcie module ..."
sudo modprobe akida_pcie

echo "Verifying /dev/akida0 ..."
if [ -c /dev/akida0 ]; then
    echo "SUCCESS: /dev/akida0 is present"
else
    echo "WARNING: /dev/akida0 not found — checking dmesg ..."
    dmesg | tail -20
    exit 1
fi

echo "Adding akida_pcie to /etc/modules for persistence ..."
if ! grep -q akida_pcie /etc/modules; then
    echo "akida_pcie" | sudo tee -a /etc/modules
fi

echo "=== Step 3: Create Python venv and install Akida SDK ==="
sudo apt-get update -qq
sudo apt-get install -y -qq python3.12 python3.12-venv python3.12-dev

if [ ! -d ~/akida-env ]; then
    python3.12 -m venv ~/akida-env
fi
source ~/akida-env/bin/activate

pip install --upgrade pip
pip install akida==2.19.1 akida-models==1.13.1 numpy scipy

echo "=== Step 4: Verify Akida device detection ==="
python3 -c "import akida; devs = akida.devices(); print(f'Akida devices: {devs}'); assert len(devs) > 0, 'No Akida devices found!'"

echo "=== Setup complete on claudio ==="
REMOTE_EOF

echo "=== All done! raspberry-claudio is ready. ==="
