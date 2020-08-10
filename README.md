# Hand Written Digit Recognition
**Its Divide in two part:**\
**i)The MNIST dataset where digits are normal with rotation**
Used Simple CNN structure to classify the images into respective label\
Accuracy obtained 98.84% on Test data.\
**ii)The MNIST-rot dataset where digits are rotated at a random angle**


**NOTE** I have not used any validation dataset in the code. The validation dataset was used initially to check if model is not over-fitting.
Also the loss mentioned in the graph is per epochs. To calculate the average loss on the data once can easily divide the loss by total batch size.
