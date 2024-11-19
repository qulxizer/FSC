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
apt install python3-venv
apt install libxcb-cursor0
apt install qt6-base-dev
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

Downloading the dataset for the detector:
1. Get Apikey from [universe.roboflow](https://universe.roboflow.com)
2. Run ```make download_dataset ARGS="<<APIKEY_HERE>> dataset/tomato_checker"```
3. Change ```dataset/tomato_checker/data.yaml``` paths to full path like this: ```

train: /home/$USER/PATH-TO-REPO/dataset/tomato_checker/valid
val: /home/$USER/PATH-TO-REPO/dataset/tomato_checker/valid/images
test: /home/$USER/PATH-TO-REPO/dataset/tomato_checker/test/images

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

1.  **Tomato Checker** - **Description**:

    - **Source**: [Tomato Checker](https://universe.roboflow.com/money-detection-xez0r/tomato-checker/dataset/1) 

2. **Backpack-perfect**
    - **Source**: [Backpack-perfect]
    (https://vision.middlebury.edu/stereo/data/scenes2014/)

3. **Tsukuba**
    - **Source**: [Backpack-perfect]
    (https://vision.middlebury.edu/stereo/data/scenes2001/)

## Credits

- YOLOv11 model: Developed by [Ultralytics](https://github.com/ultralytics/ultralytics)
- Datasets: As listed above with their respective sources and citations.
