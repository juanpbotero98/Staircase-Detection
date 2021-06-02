# Object detection model quantization for staircase detection
In this repopository you will find the implementation of a FasterRCNN object detection model, trained on a custom staircase dataset. Additionally, this model was quantized used both quantization aware trainning (QAT) and post tranning static quantization (SQ).

In main.py you can find the demostration and test of QAT model. In order to run this script follow these steps:
* Softlink to the dataset and trained models (Only for people who have access to bcv002 server):
  ```
    $ ln -s /media/disk0/Datasets_FP/Botero/Dataset/
    $ ln -s /media/disk0/Datasets_FP/Botero/trained_models/
  ```

* Run command for main.py:
  ```
  $ main.py --mode <'demo' or 'test'> --img <Image name to run demo>
  ```
If the mode selected is demo a image name must be provided, this name must be from the test dataset. To run demo on a custom image one must add the image to (Dataset/test/images) and it's name must be passed in the img argument.


## Code References 
* [TorchVision Object Detection Finetuning Tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)
* [PyTorch Quantization](https://pytorch.org/docs/stable/quantization.html)
* [Static Quantization with Eager Mode in PyTorch](https://tutorials.pytorch.kr/advanced/static_quantization_tutorial.html)
* [Leimao PyTorch Static Quantization](https://leimao.github.io/blog/PyTorch-Static-Quantization/)
* [Leimao PyTorch Quantization Aware trainning](https://leimao.github.io/blog/PyTorch-Quantization-Aware-Training/)