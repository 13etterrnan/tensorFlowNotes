# 安装
1. 首先，为了在多个 GPU 卡上运行 TensorFlow，首先需要确保 GPU 卡具有 NVidia 计算能力
2. 其次，下载并安装相应版本的 CUDA 和 cuDNN 库（TensorFlow 使用 CUDA 和 cuDNN 来控制 GPU 卡并加速计算）
    可以使用nvidia-smi查看CUDA是否正确安装
3. 最后，必须安装支持GPU的tensorflow(conda install tensorflow-gpu)
4. 检测tensorflow是否正确检测并使用CUDA和cuDNN：
    > \>>>import tensorflow as tf <br/>
        I [...]/dso_loader.cc:108] successfully opened CUDA library libcublas.so locally <br/>
        I [...]/dso_loader.cc:108] successfully opened CUDA library libcudnn.so locally <br/>
        I [...]/dso_loader.cc:108] successfully opened CUDA library libcufft.so locally <br/>
        I [...]/dso_loader.cc:108] successfully opened CUDA library libcuda.so.1 locally <br/>
        I [...]/dso_loader.cc:108] successfully opened CUDA library libcurand.so locally <br/>
    > \>>>sess = tf.Sesstion()<br/>
        [...]<br/>
    I [...]/gpu_init.cc:102] Found device 0 with properties:<br/>
    name: GRID K520<br/>
    major: 3 minor: 0 memoryClockRate (GHz) 0.797<br/>
    pciBusID 0000:00:03.0<br/>
    Total memory: 4.00GiB<br/>
    Free memory: 3.95GiB<br/>
    I [...]/gpu_init.cc:126] DMA: 0<br/>
    I [...]/gpu_init.cc:136] 0: Y<br/>
    I [...]/gpu_device.cc:839] Creating TensorFlow device<br/>
    (/gpu:0) -> (device: 0, name: GRID K520, pci bus id: <br/>0000:00:03.0)<br/>

    # 管理GPU
    * 在不同GPU上运行每个进程：<br/>
        设置CUDA_VISIBLE_DEVICES 环境变量，以便每个进程只能看到对应GPU卡
        > $ CUDA_VISIBLE_DEVICES=0,1 python3 program_1.py<br/>
        #and in another terminal<br/>
        $ CUDA_VISIBLE_DEVICES=3,2 python3 program_2.py

        tensorflow只使用一小部分显存。例如只占用每个GPU显存的40%。<br/>
        通过创建一个ConfigProto对象，将其gpu_options.per_process_gpu_memory_fraction选项设置为0.4并使用一下配置创建sesstion：
        > config = tf.ConfigProto()<br/>
        config.gpu_options.per_process_gpu_memory_fraction = 0.4<br/>
        session = tf.Session(config=config)
        
