# cats-vs-dogs-classifier

In this repository, I created an artificial intelligence to classify 250x250 photos of cats and dogs. All models were trained using TensorFlow 2.4.0 and 15 epochs.

My current model produces an accuracy of ~84%.

The dataset I used is the Kaggle Cats vs Dogs Redux: Kernel Edition dataset. https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/
It consists of 25000 training images, 1/2 of them cats & 1/2 of them dogs.

plot-1.png shows some of the photos that were in the dataset.

plot-2.png has been removed due to the data it in being incorrect from a bug in my code.

plot-3.png is a chart showing the training accuracy vs validation accuracy as well as training loss vs validation loss. This data came from my original model.
The way training accuarcy increases exponentially whereas validation accuracy plateaus indicates that the model is overfitting. This is result it the model having
a difficult time generalizing on a new dataset.

My original model consisted of these layers:
Conv2D with 16 filters
MaxPooling2D with 2x2 pool size and 2 stride
Conv2D with 32 filters
MaxPooling2D with 2x2 pool size and 2 stride
Conv2D with 64 filters
MaxPooling2D with 2x2 pool size and 2 stride

plot-final.png shows my current models training accuracy vs validation accuracy & traing loss vs validation loss.
677a42d model layers:
Conv2D with 16 filters
MaxPooling2D with 2x2 pool size and 2 stride
Conv2D with 32 filters
MaxPooling2D with 2x2 pool size and 2 stride
Conv2D with 64 filters
MaxPooling2D with 2x2 pool size and 2 stride
Dropout 20%
