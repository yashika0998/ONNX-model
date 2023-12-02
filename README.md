# ONNX-model (Open Neural Network Exchange) 

**It is an open standard format for representing machine learning model**

This repository is intended to contain all AI/ML trained model in .onnx form to be deployed on different hardware.

To learn about the process how to export your pytorch model in .ONNX format ([refer code file](https://github.com/yashika0998/ONNX-model/blob/main/Export_Save_PyTorch_model_into_ONNX_format_.ipynb)).

To learn about the process how to use .ONNX extension file for future inference ([refer code file](https://github.com/yashika0998/ONNX-model/blob/main/mnist_onnx_MacOS.py)).

**Model-1:** mnist-1.py - This model predicts handwritten digits using a convolutional neural network (CNN).

mnist_onnx_infrence.py file code can be used for Infrence on different hardware. Code has been tested on linux and MacOS and works correctly giving same input. 

**Model-2:** IoT-23-BERT-Network-Logs-Classification

Model has infrence to see its working to detect network logs as malicious or bening published at Hugging Face Spaces ([refer](https://huggingface.co/spaces/yashika0998/IoT-23-BERT-Network-Logs-Classification)). For more details of model trained for this classsification problem, refer the code and readme published on Hugging Face ([refer](https://huggingface.co/yashika0998/IoT-23-BERT-Network-Logs-Classification)). To refer the dataset used click ([here](https://huggingface.co/datasets/yashika0998/iot-23-preprocessed-allcolumns))

Notes !!

How Inference Works:
* Input Data: In the inference phase, the model receives new, unseen input data. This data should be preprocessed and formatted in a way that matches the input format expected by the model. {Input tensor has shape (1x1x28x28), with type of float32. One image at a time. This model doesn't support mini-batch.}
* Forward Pass: The input data is passed through the model in a forward pass. The model applies the learned transformations, weights, and biases to the input data, resulting in an output.
* Output: The model produces an output, which depends on the specific task. For example, in a classification task, the output might be a probability distribution over classes, while in regression, the output is a numerical prediction.
* Decision-Making: Based on the output, a decision is made. In classification, the class with the highest probability might be selected. In regression, the predicted value is used.

You can use ([refer](https://netron.app)) to get input and output node name of model if missing in documentation of model. Other way is mentioned in .py code.

Sometimes latest versions of python are not compatible with .onnx and are in beta. Always recommended to try older python versions like 3.8 or 3.9 in that case.

This repository is in beta and more related work with ONNX runtime will be posted time to time!!  I hope it will help :)








