#!/bin/bash
set -ex

echo "=== STEP 1: Dependencies & Keys ==="
apt-get update
apt-get install -y ca-certificates curl gnupg socat

install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc

echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" > /etc/apt/sources.list.d/docker.list

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' > /etc/apt/sources.list.d/nvidia-container-toolkit.list

echo "=== STEP 2: Installing Docker and NVIDIA Container Toolkit ==="
apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin nvidia-container-toolkit

echo "=== STEP 3: Configuring Docker & NVIDIA runtime ==="
nvidia-ctk runtime configure --runtime=docker
systemctl daemon-reload || true
systemctl restart docker || service docker restart

TARGET_USER="${SUDO_USER:-$USER}"
TARGET_HOME=$(getent passwd "$TARGET_USER" | cut -d: -f6)
TARGET_HOME="${TARGET_HOME:-$HOME}"

usermod -aG docker "$TARGET_USER" || true

echo "=== STEP 4: Verifying Docker and GPU access ==="
docker info | grep -i "Runtimes" || true

echo "=== STEP 5: Installing PipeGen 0.5.1 ==="
if [ -f "/opt/PipeGen/PipeGenApp" ]; then
    echo "PipeGen already installed at /opt/PipeGen"
elif [ -f "$1" ]; then
    dpkg -i "$1"
fi
chown -R "$TARGET_USER:$TARGET_USER" /opt/PipeGen || true
mkdir -p "$TARGET_HOME/.pipegen"
chown -R "$TARGET_USER:$TARGET_USER" "$TARGET_HOME/.pipegen" || true

echo "=== Setup completed successfully! ==="
