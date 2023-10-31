# ONNX-model (Open Neural Network Exchange) 

**It is an open standard format for representing machine learning model**

This repository is intended to contain all ML trained model in .onnx form to be deployed on different hardware.

mnist-1.py - This model predicts handwritten digits using a convolutional neural network (CNN).

mnist_onnx_infrence.py file code can be used for Infrence on different hardware. Code has been tested on linux and MacOS and works correctly giving same input. 

Notes !!

How Inference Works:
* Input Data: In the inference phase, the model receives new, unseen input data. This data should be preprocessed and formatted in a way that matches the input format expected by the model. {Input tensor has shape (1x1x28x28), with type of float32. One image at a time. This model doesn't support mini-batch.}
* Forward Pass: The input data is passed through the model in a forward pass. The model applies the learned transformations, weights, and biases to the input data, resulting in an output.
* Output: The model produces an output, which depends on the specific task. For example, in a classification task, the output might be a probability distribution over classes, while in regression, the output is a numerical prediction.
* Decision-Making: Based on the output, a decision is made. In classification, the class with the highest probability might be selected. In regression, the predicted value is used.

You can use https://netron.app to get input and output node name of model if missing in documentation of model. Other way is mentioned in .py code.

Sometimes latest versions of python are not compatible with .onnx and are in beta. Always recommended to try older python versions like 3.8 or 3.9 in that case.







