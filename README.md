# SeniorSoftwareASL
# Senior Design: ASL Recognition System (Docker Edition)

This project uses a Dockerized environment to ensure MediaPipe, OpenCV, and Python dependencies are identical for all team members.

## 🛠 Prerequisites

Before running the code, you must install these three tools on your Windows host:

1. **Docker Desktop**: [Download here](https://www.docker.com/products/docker-desktop/)
2. **VcXsrv (X Server)**: [Download here](https://sourceforge.net/projects/vcxsrv/) (Required for the camera window to pop up).
3. **usbipd-win**: [Download here](https://github.com/dorssel/usbipd-win/releases) (Required to pass your webcam into Docker).

---

## 🚀 Getting Started

### 1. Launch the X-Server (VcXsrv)
You must do this **every time** you reboot.
* Open **XLaunch** from your Start Menu.
* Select **Multiple Windows** -> Next.
* Select **Start no client** -> Next.
* **CRITICAL:** * Uncheck **Native OpenGL**.
    * Check **Disable access control**.
* Click **Finish**. (You will see a black 'X' in your system tray).

### 2. Connect Your Webcam to WSL
Open **Windows PowerShell** as Administrator and run:
```powershell
# List your USB devices
usbipd list

# Find your webcam's Bus ID (e.g., 1-3) and attach it
usbipd attach --wsl --busid <BUS_ID>

window attempt using this if you could have wsl and camera attached before this may not work: usbipd attach --wsl --busid 1-3 --auto-attach