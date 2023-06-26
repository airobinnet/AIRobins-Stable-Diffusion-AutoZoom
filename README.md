# AIRobin's SD AutoZoom 🎥



https://github.com/airobinnet/AIRobins-Stable-Diffusion-AutoZoom/assets/126980386/cd73af2d-6c67-4410-9d54-4895a9212016



https://github.com/airobinnet/AIRobins-Stable-Diffusion-AutoZoom/assets/126980386/f32ea539-7cd0-4498-bef5-e876923ce7e1



https://github.com/airobinnet/AIRobins-Stable-Diffusion-AutoZoom/assets/126980386/d95738f8-3907-458e-8933-ed98902dbd36



This project is a web application that allows users to create zoom animations from images using AI.

## Project Structure

The project consists of two main Python files and an HTML file:

- `main_local.py`: This is the main backend file that runs the server and handles the requests. It uses the `StableDiffusionInpaintPipeline` model to perform the inpainting process and create the zoom animations.

- `main_api.py`: This file uses the `replicate.com` API to perform the inpainting process for users who can't run the model on their local machine.

- `templates/index.html`: This is the main frontend file that provides the user interface for uploading images, setting the parameters, and viewing the results.

## How to Run the Project

1. Make sure you have Python 3.9 or higher installed.

2. Install the required Python packages using pip:

```bash
pip install -r requirements.txt
```

3. Run the `download-models.py` to download the model the cache folder

4. Run the `main_local.py` file to start the server:

```bash
python main_local.py
```

5. Open your web browser and go to `http://localhost:5002` to access the application.

## How to Use the Application

1. Click on the "Upload Image" button to upload an image.

2. Set the parameters for the zoom animation such as the resolution, prompt, quality, and amount.

3. Click on the "Generate" button to start the inpainting process and generate the zoom animation.

4. Once the process is finished, you can view the generated images and download the zoom animation as a video file.

## Using the API Version

If you can't run the model on your local machine, you can use the `main_api.py` file which uses the `replicate.com` API to perform the inpainting process. However, please note that the API version only supports a resolution of 512x512.



# How to Install on Windows
## Prerequisites
Before beginning installation, make sure you have:
- Windows 10/11 with Windows Subsystem for Linux and Virtual Machine Platform capabilities.
- NVIDIA GPU.
  - RTX 1080TI/2000/3000 series
  - Kesler/Tesla/Volta/Ampere series
- Other configurations are not guaranteed to work.

## 1. Install the GPU driver
Per NVIDIA, the first order of business is to install the latest Game Ready drivers for you NVIDIA GPU. [Download here](https://www.nvidia.com/download/index.aspx)

Restart your computer once the driver has finished installation.

## 2. Unlocking features
Open Windows Terminal as an administrator.

Use start to search for "Terminal"
Right click -> Run as administrator...
Run the following powershell command to enable the Windows Subsystem for Linux and Virtual Machine Platform capabilities.

### 2.1. Unlock WSL2
```powershell
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
```
If you see an error about permissions, make sure the terminal you are using is run as an administrator and that you have an account with administrator-level privileges.

### 2.2. Unlock virtualization
```powershell
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
```
If this command fails, make sure to enable virtualization capabilities in your computer's BIOS/UEFI. A successful output will print The operation completed successfully.

Output from running the above commands successfully. Should read "The operation completed successfully"

### 2.3. Reboot
Before moving forward, make sure you reboot your computer so that Windows 11 will have WSL2 and virtualization available to it.

## 3. Update MS Linux kernel
Download and run the WSL2 Linux kernel update package for x64 machines msi installer. When prompted for elevated permissions, click 'yes' to approve the installation.

To ensure you are using the correct WSL kernel, open Windows Terminal as an adminstrator and enter:
```bash
wsl cat /proc/version
```
This will return a complicated string such as:

`Linux version 5.10.102.1-microsoft-standard-WSL2 (oe-user@oe-host) (x86_64-msft-linux-gcc (GCC) 9.3.0, GNU ld (GNU Binutils) 2.34.0.20200220)`

The version we are interested in is `Linux version 5.10.102.1`. At this point, you should have updated your kernel to be at least `Linux version 5.10.43.3`.

If you can't get the correct kernel version to show:
Open Settings → Windows Update → Advanced options and ensure Receive updates for other Microsoft products is enabled. Then go to Windows Update again and click Check for updates.

## 4. Configure WSL 2
First, configure Windows to use the virtualization-based version of WSL (version 2) by default. In a Windows Terminal with adminstrator priveleges, type the following:
```powershell
wsl --set-default-version 2
```
Now, you will need to go to the Microsoft Store and Download Ubuntu 18.04

Launch the "Ubuntu" app available in your Start Menu. Linux will require its own user account and password, which you will need to enter now.

## 5. Configure CUDA WSL-Ubuntu Toolkit
By default, a shimmed version of the CUDA tooling is provided by your Windows GPU drivers.

Important: you should never use instructions for installing CUDA-toolkit in a generic linux fashion. in WSL 2, you always want to use the provided CUDA Toolkit using WSL-Ubuntu Package.

First, open PowerShell or Windows Command Prompt in administrator mode by right-clicking and selecting "Run as administrator". Then enter the following command:
```powershell
wsl.exe
```
This should drop you into your running linux VM. Now you can run the following bash commands to install the correct version of cuda-toolkit for WSL-Ubuntu. Note that the version of CUDA used below may not be the version of CUDA your GPU supports.
```bash
sudo apt-key del 7fa2af80 # if this line fails, you may remove it.
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda-repo-wsl-ubuntu-11-7-local_11.7.0-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-11-7-local_11.7.0-1_amd64.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-11-7
```

execute the below commands to update packages.

```bash
sudo apt update && sudo apt upgrade 
```

Then install the required packages for the compilation of Python source code.

```bash
sudo apt install wget build-essential libreadline-gplv2-dev libncursesw5-dev \
     libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev libffi-dev zlib1g-dev  
```

## 6. Installing Python 3.10 with Source

Download the latest Python version source code from the official websites. Then compile the source code for your system and install it.

Follow the below steps to install Python 3.10:

You can directly download Python 3.10 source archive from its official site or use the below command.

```bash
wget https://www.python.org/ftp/python/3.10.8/Python-3.10.8.tgz 
```

Once the download is completed, extract the archive file on your system.

```bash
tar xzf Python-3.10.8.tgz 
```

Change to the extracted directory with cd command, then prepare the Python source code for the compilation on your system.

```bash
cd Python-3.10.8 
./configure --enable-optimizations 
```

Finally, run the following command to complete the Python installation on the Debian system. The altinstall prevents the compiler to override default Python versions.

```bash
make altinstall 
```

Wait for the Python installation to complete on your system.

## 7. Check Python Version

At this step, you have successfully installed Python 3.10 on Ubuntu or Debian system. You need to type python3.10 to use this version. For example, to check the Python version, execute:

```bash
python3.10 -V 
```

This will output:

```
Python 3.10.8
```

This will also install pip for Python 3.10.

```bash
pip3.10 -V 
```

This will output:

```
pip 21.2.3 from /usr/local/lib/python3.10/site-packages/pip (python 3.10)
```

That’s it, You have successfully installed everything.
```
