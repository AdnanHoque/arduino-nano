# Arduino BLE

This repository is for my own learning and reference and also future tutorial material. 

If you have a machine with a GPU and would like to be able to leverage its resources for this project please execute the following, ensuring you have docker installed - although it should be noted the models being trained here are going to be extremely small, however this is just my preferred workflow for training models since it also keeps workspaces seperate and containerized


~~~
docker pull tensorflow/tensorflow:latest-gpu
~~~

~~~
sudo docker run -it --rm --gpus all -v $(pwd):/workspace/ tensorflow/tensorflow:nightly-gpu bash
~~~

~~~
cd nano-mcu
~~~

~~~
pip install -r requirements.txt
~~~

Away we go! 

# Motivation 

TinyML is an exciting field that I'd like to keep exploring, hopefully anyone reading this can follow along and explore with me - and moreover find exciting use cases for ml in the embedded world.

# Hardware - Software 

My Dev Machine - OS: Ubuntu 20.04, GPU: RTX 2060 Super 

Hardware Required: Arduino Nano 33 BLE - Processor: 32-bit ARM Cortex-M4 CPU @ 64 MHz


# Workflow

The quantization folder is a walkthrough of converting from fp32 -> int8 using the TensoflowLite Converter - Once this process is completed one more step is required to convert it to a c file which our mcu can make use of. 

At this point the workflow should be fairly clear: train model -> tflite converter -> source.c file -> mcu  

Will come back to this when I have time to implement use case.
