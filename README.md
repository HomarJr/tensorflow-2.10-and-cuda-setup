# tensorflow-2.10-and-cuda-setup

Guide on how to set up TensorFlow 2.10 with CUDA in windows 10/11

## Why this exists?

Setting up TensorFlow with CUDA was not as straightforward as I expected so I decided to make this for future reference.

## Requirements

### You will need the following

- [Python 3.7.x-3.10.x](https://www.python.org/downloads/)
- [CUDA 11.2.0](https://developer.nvidia.com/cuda-11.2.0-download-archive)
- [cuDNN 8.1.0 for CUDA 11.0,11.1 and 11.2](https://developer.nvidia.com/rdp/cudnn-archive)
- A CUDA-compatible graphics card

### Make sure to have a clean environment

When installing CUDA and cuDNN it is important to not have other versions installed as this can cause the installer to fail. The easiest way to make sure you do not run into problems with the installer is to uninstall every app that contains 'NVIDIA' in its name except for the following:

- NVIDIA Control Panel
- NVIDIA GeForce Experience
- NVIDIA Graphics Driver
- NVIDIA HD Audio Driver

NOTE: If you need to keep a specific version of CUDA (alongside PhysX, FrameView and/or Nsight) for any reason, you could try running the CUDA 11.2.0 installer and hope that it does not fail. If it fails, the version you want to keep may not be compatible with this version. If that is the case, you should check the official [tested configurations](https://www.tensorflow.org/install/source?hl=es-419#gpu), adjust the versions of Python, TensorFlow and cuDNN to match your current installation of CUDA and continue with the guide.

As for Python, there are no problems if you have other versions installed. Just make sure to stick with one and only one of the compatible versions when installing packages and running your code.

## Python setup

### If you do not have Python previously installed

Run the Python installer you downloaded and follow the instructions. Make sure to tick the checkbox that says "Add Python to PATH" before installing.

Then install the TensorFlow package with the following command:

    pip install tensorflow==2.10

NOTE: If you are installing a different version of TensorFlow change the command accordingly.

You can skip to the CUDA and cuDNN setup section

### If you have Python 3.7.x-3.10.x already installed

If you already have a compatible version of python installed, first check if you also have the TensorFlow package with the following command:

    pip list

This will give a list of all the packages you have installed and their version. Depending on which TensorFlow version you have installed, if any, do one of the following:

#### If TensorFlow 2.10.x (or the version you are installing) is on the list

This means you already have the correct version of TensorFlow installed.

You can skip to the CUDA and cuDNN setup section

#### If TensorFlow is not on the list

Install TensorFlow with the following command:

    pip install tensorflow==2.10

NOTE: If you are installing a different version of TensorFlow change the command accordingly.

You can skip to the CUDA and cuDNN setup section

#### If TensorFlow is on the list but with a different version

You will need to uninstall your current version along with its dependencies before installing the correct version. You can uninstall the TensorFlow package and its dependencies with the following command:

    pip uninstall keras Keras-Preprocessing tensorboard tensorboard-data-server tensorboard-plugin-wit tensorflow tensorflow-estimator tensorflow-io-gcs-filesystem

You will be asked if you want to delete the files.

Now you can install the correct version of TensorFlow with the following command:

    pip install tensorflow==2.10

NOTE: If you are installing a different version of TensorFlow change the command accordingly.

You can skip to the CUDA and cuDNN setup section

### If you have a different version of Python installed

Run the Python installer you downloaded and follow the instructions.

NOTE: When running pip commands you will need to specify the version of python to make sure everything is installed in that version and not in the one you already have. To do so, you will need to write the path of the python.exe for the correct version before the pip command. The python.exe file is usually in the AppData folder or the ProgramFiles folder.

Install TensorFlow with the following command:

    C:/PATH/TO/YOUR/PYTHON/INSTALL/python.exe pip install tensorflow==2.10

NOTE: If you are installing a different version of TensorFlow change the command accordingly.

You can skip to the CUDA and cuDNN setup section

## CUDA and cuDNN setup

### CUDA installation

If you have not run the CUDA installer yet, run it and follow the instructions. You can choose not to install the Visual Studio tools if you choose a custom installation.

### Adding cuDNN libraries to CUDA

Extract the zip file of the cuDNN libraries and move everything inside the `cuda` folder to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2`.

### PATH variables

The last step is to set PATH variables for CUDA. If this is not done, TensorFlow will not find all the CUDA binaries and libraries.

You can set all the necessary PATH variables with the following commands:

    SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\bin;%PATH%
    SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\extras\CUPTI\lib64;%PATH%
    SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\include;%PATH%

Do not forget to reset your PC/laptop to see the changes.

## Make sure that it works

To make sure everything is now set up correctly you can run the following Python script:

```python
import tensorflow as tf


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
tf.keras.layers.Flatten(input_shape=(28, 28)),
tf.keras.layers.Dense(128, activation='relu'),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=8)

model.evaluate(x_test, y_test)
```

You should see a load on your GPU and it should not take a long time. You will also see your GPU information as a log near the top.

## Thanks

I hope this guide helped you. If you have any feedback and/or suggestions to improve it please let me know.
