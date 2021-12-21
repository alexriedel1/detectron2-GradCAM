# detectron2-GradCAM
This repo helps you to perform [GradCAM](https://arxiv.org/abs/1610.02391) and [GradCAM++](https://arxiv.org/abs/1710.11063) on the [detectron2](https://github.com/facebookresearch/detectron2) model zoo. It follows other GradCAM implementations but also handles the detectron2 API specific model details. Be sure to have the latest detectron2 version installed. 

There is also [this](https://github.com/yizt/Grad-CAM.pytorch) repo to do GradCAM in detectron2. It advises you to make changes to the detectron2 build which I think is not a good idea..

| Original        | GradCAM (horse)           | GradCAM++  (horse)           |
| ------------- |:-------------:| :-------------:|
| <img src="https://github.com/alexriedel1/detectron2-GradCAM/blob/main/img/input.jpg" alt="drawing" width="400"/>| <img src="https://github.com/alexriedel1/detectron2-GradCAM/blob/main/img/grad_cam.png" alt="drawing" width="400"/> | <img src="https://github.com/alexriedel1/detectron2-GradCAM/blob/main/img/grad_cam++.png" alt="drawing" width="400"/> |


For doing this, check the `main.py` and change the `img_path`, the `config_file` and the `model_file` according to your needs. 

For ResNet50 models the layer `backbone.bottom_up.res5.2.conv3` will be a good choice for the classification explanation. For larger or smaller models, change the layer accordingly via `layer_name`.


For your custom models, either write your own config.yaml or edit [`cfg_list`](https://github.com/alexriedel1/detectron2-GradCAM/blob/main/main.py#L15)


There's also a Colab with everything set up: [GradCam Detecteron2](https://colab.research.google.com/drive/15GN0juUurMPCDHA3tGp6nJ4mxUiSHknZ)
