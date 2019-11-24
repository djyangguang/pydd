##CPU 环境配置，支持linux\macOs
conda create -n chineseocr python=3.6 pip scipy numpy jupyter ipython ##运用conda 创建python环境
source activate chineseocr
cd darknet/ && make && cd ..
pip install easydict opencv-contrib-python==4.0.0.21 Cython h5py lmdb mahotas pandas requests bs4 matplotlib lxml -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install -U pillow -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install web.py==0.40.dev0 redis
pip install keras==2.1.5 tensorflow==1.8
pip tensorflow==1.8
## mac
conda install pytorch torchvision -c pytorch
## linux
## conda install pytorch-cpu torchvision-cpu -c pytorch

pip3 install torch==1.2.0+cu92 torchvision==0.4.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html -i https://pypi.tuna.tsinghua.edu.cn/simple some-package

————————————————
版权声明：本文为CSDN博主「Rosinante.」的原创文章，遵循 CC 4.0 BY 版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/qq_21420941/article/details/100200030
conda install web.py==0.40.dev0 redis

