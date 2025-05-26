# Road Surface Real-time Recognition Project

This repository contains code and resources for road surface defect detection using deep learning.

## Getting Started

**Please use the [`Realtime_Road_Surface_Recognition`](Realtime_Road_Surface_Recognition/) folder for all main features, including the web server and inference scripts.**


## Folders and Files

- [`Realtime_Road_Surface_Recognition/`](Realtime_Road_Surface_Recognition/): Main project code (Flask server, inference, web UI, etc.)
- `archive/`, `OLD_FILES/`: Legacy files and datasets (not required for main usage)
- `ckpt/`, `ckpt1/`: Model checkpoints
- `*.ipynb`: Notebooks for training and inference

## Notes

- For HTTPS support, see the self-signed certificate instructions in [`Realtime_Road_Surface_Recognition/README.md`](Realtime_Road_Surface_Recognition/README.md).
- For any issues, please check the documentation in the main project folder.

---
For more details, see [`Realtime_Road_Surface_Recognition/README.md`](Realtime_Road_Surface_Recognition/README.md).

## Usage
```bash
# environments and  dependencies, assume you've installed anaconda.
conda env create -f environment.yml

# and activate this env (with the name torch312)
conda activate torch312
```
```bash
# open a web-server on localhost:5000
cd Realtime_Road_Surface_Recognition
python app.py
```
**And now, you can use another device to connect the `https://<server-ip>:5000`, and enjoy the congnition functionality!**
> **Note:**  
> Remember to replace `<server-ip>` with your actual server's IP address.  
> You must use **https** (not http) to allow your phone's camera to be accessible in the web interface.