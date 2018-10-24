# Script to install everything needed for donkeycar except the donkeycar library

#standard updates (5 min)
sudo apt update -y
sudo apt upgrade -y
sudo rpi-update -y

#helpful libraries (2 min)
sudo apt install build-essential python3-dev python3-distlib python3-setuptools  python3-pip python3-wheel -y

sudo apt-get install git cmake pkg-config -y
sudo apt-get install libjpeg-dev libtiff5-dev libjasper-dev libpng12-dev -y
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev -y
sudo apt-get install libxvidcore-dev libx264-dev -y
sudo apt-get install libatlas-base-dev gfortran -y

sudo apt install libzmq-dev -y
sudo apt install xsel xclip -y
sudo apt install python3-h5py -y

#install numpy and pandas (3 min)
sudo apt install libxml2-dev python3-lxml -y
sudo apt install libxslt-dev -y

#remove python2 (1 min)
sudo apt-get remove python2.7 -y
sudo apt-get autoremove -y

#install redis-server (1 min)
sudo apt install redis-server -y

sudo bash make_virtual_env.sh


#create a python virtualenv (2 min)
sudo apt install virtualenv -y
virtualenv ~/env --system-site-packages --python python3
echo '#start env' >> ~/.bashrc
echo 'source ~/env/bin/activate' >> ~/.bashrc
source ~/env/bin/activate


#make sure the virtual environment is active
source ~/env/bin/activate

# install pandas and numpy
pip install pandas #also installs numpy


#install tensorflow (5 min)
echo "Installing MxNet"
wget https://github.com/vlamai/donkey/releases/download/v2.2.1/mxnet-1.3.0-py2.py3-none-any.whl
virtualenv -p /home/pi/env/bin/python3 mxnet_py3
source mxnet_py3/bin/activate
pip install mxnet-1.3.0-py2.py3-none-any.whl
rm mxnet-1.3.0-py2.py3-none-any.whl

echo "installation complete, do 'sudo reboot' to complete installation"