# Hand Written Digit Recognition
**It is Divide in two part:**\
**i)The MNIST dataset where digits are normal without any rotation**
*Filename : MNIST.py*\
Dataset contains 50000 training example and 10000 test examples\
Used Simple CNN structure to classify the images into respective label\
Accuracy obtained 98% on Test data.\
\
**ii)The MNIST-rot dataset where digits are rotated at a random angle**
*Filename : MNIST_rot.py*\
Dataset contains 12000 training examples and 10000 test example\
Tried the model used with MNIST dataset but, the accuracy was about 80%, then tried hypertuning on the same model the acuuracy touches maximum upto 83%.So used a differernt model (CNN).\
Getting an accuracy of about 93%


**NOTE** I have not used any validation dataset in the code. The validation dataset was used initially to check if model is not over-fitting.
Also the loss mentioned in the graph is per epochs. To calculate the average loss on the data once can easily divide the loss by total batch size.
