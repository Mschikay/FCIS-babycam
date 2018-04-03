# FCIS-babycam
This repository modified by Ziqi Tang is mainly based on [FCIS](https://github.com/msracver/FCIS).
###环境配置
##### 1. python***2.7***

	pip install Cython
	pip install opencv-python==3.2.0.6
	pip install easydict==1.6
	pip install hickle
	
##### 2. git此项目后，在FCIS-babycam文件中：

For Linux: run 
	
	sh ./init.sh
	
For Windows: run

	cmd .\init.bat
##### 3. 安装[MXNet@(commit 998378a)](https://github.com/apache/incubator-mxnet/tree/998378a)

Clone MXNet and checkout to MXNet@(commit 998378a) by
	
	git clone --recursive https://github.com/dmlc/mxnet.git
	git checkout 998378a
	git submodule update
	
 Copy channel operators in $(FCIS_ROOT)/fcis/operator_cxx to $(YOUR_MXNET_FOLDER)/src/operator/contrib by
 	
 	cp -r $(FCIS_ROOT)/fcis/operator_cxx/channel_operator* $(MXNET_ROOT)/src/operator/contrib/

Compile MXNet

	cd ${MXNET_ROOT}
	make -j $(nproc) USE_OPENCV=1 USE_BLAS=openblas USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1
	
Install the MXNet Python binding by

	cd ${MXNET_ROOT}python
	sudo python setup.py install
	
*For advanced users, you may put your Python packge into ./external/mxnet/$(YOUR_MXNET_PACKAGE), and modify MXNET_VERSION in ./experiments/fcis/cfgs/*.yaml to $(YOUR_MXNET_PACKAGE). Thus you can switch among different versions of MXNet quickly.*

### 运行程序
##### 1. 下载模型
 在[BaiduYun](https://pan.baidu.com/s/1geOHioV)下载pretrained model。提取密码：tmd4

下载后的模型目录：

	./model/fcis_coco-0000.params

##### 2. Run
	python ./babycam/segbaby.py
