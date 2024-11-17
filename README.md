# FCS

Main idea of this project is to determine riped and unriped fruit and get the coordenents to send to our robotic arm to catch the fruit if its riped

## Installation

### Debian

Updating repos
```bash
apt update
```


Installing python and nessesery tools
```bash
apt install git
apt install python3
apt install python3-pip
apt install python3.11-venv
```

Cloning the repo
``` bash
git clone https://github.com/qulxizer/FSC
```

Cd into the directory and create venv
```bash
cd FSC

python3 -m venv venv
source ./venv/bin/activate
```

Installing requirements
```bash
pip install -r requirements.txt
```

After that fell free to run all the provided scripts
```bash
# To run the program run
make run

# To unit test the app run
make test

# To capture images run
make getImages

# Finally to calibrate the cameras use
make calibrate

```



## Used Datasets

1.  **Tomato fruits dataset** - **Description**:

    - **Source**: [Tomato fruits dataset](https://www.kaggle.com/datasets/nexuswho/tomatofruits/data) 
    - **Citation**:
    ```
        MPUTU, HASSAN; Abdel-Mawgood, Ahmed; Shimada, Atsushi; Sayed, Mohammed S. (2023), “Tomato fruits dataset for binary and multiclass classification”, Mendeley Data, V1, doi: 10.17632/x4s2jz55dx.1
    ```
2. **Backpack-perfect**
    - **Source**: [Backpack-perfect]
    (https://vision.middlebury.edu/stereo/data/scenes2014/)

3. **Tsukuba**
    - **Source**: [Backpack-perfect]
    (https://vision.middlebury.edu/stereo/data/scenes2001/)

## Credits

- YOLOv11 model: Developed by [Ultralytics](https://github.com/ultralytics/ultralytics)
- Datasets: As listed above with their respective sources and citations.