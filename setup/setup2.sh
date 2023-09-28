conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=10.2 -c pytorch
pip install filterpy==1.4.5 scikit-image==0.17.2 lap==0.4.0 ipdb opencv-python numba
pip install scikit-learn==0.20.2
wget https://pjreddie.com/media/files/yolov3.weights -P /root/tracking-with-sort/config
wget https://motchallenge.net/sequenceVideos/MOT16-04-raw.webm -P /root/tracking-with-sort