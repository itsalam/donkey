virtualenv -p /home/pi/env/python3 mxnet_py3
source mxnet_py3/bin/activate
wget https://github.com/vlamai/donkey/releases/download/v2.2.1/mxnet-1.3.0-py2.py3-none-any.whl
pip install mxnet-1.3.0-py2.py3-none-any.whl
rm mxnet-1.3.0-py2.py3-none-any.whl
