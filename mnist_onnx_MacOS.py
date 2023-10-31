import onnxruntime as rt
import onnx
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

model = onnx.load("/Users/yashika.mittal/Desktop/mnist-7.onnx")
session = rt.InferenceSession(model.SerializeToString())

# load an image and prepare it as input for an ONNX model for digit recognition:
img = cv2.imread("/Users/yashika.mittal/Downloads/4.png")

#Preprocess image:
'''
1)Convert to grayscale
2)Resize to 28x28
3)Invert pixel values (black background, white digit)
'''
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, (28, 28))
#Convert image to NumPy array with data type float32:
img = img.astype('float32')
# Normalize pixel values to between 0-1:
img /= 255
input1 = img.reshape(1, 1, 28, 28)

# Get input and output node names for the ONNX model
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Pass image as input to ONNX model:
output = session.run([output_name], {input_name: input1})
#print(output)

# Post-process the output from raw logits for each class (0-9)
'''Apply softmax on the logits to convert them into probabilities'''
# Post-process the output from raw logits for each class (0-9)
output = np.array(output[0])

'''
probs = torch.softmax(torch.tensor(output), dim=1)
pred = torch.argmax(probs)
digit = str(pred.item())'''

# Apply softmax on the logits to convert them into probabilities
probs = np.exp(output) / np.exp(output).sum()

pred = np.argmax(probs)

digit = str(pred)

#print("\n Predicted Digit:", digit, "\n")

print("\nPredicted digit:", digit,"\n")

