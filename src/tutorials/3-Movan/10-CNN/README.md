# Practice 10 - CNN in TensroFlow

This practice is refer to the following resources credited to [Morvan](https://github.com/MorvanZhou).
* [莫烦PYTHON - TensorFlow: 什么是卷积神经网络 CNN (Convolutional Neural Network)](https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/5-03-A-CNN/)
* [莫烦PYTHON - TensorFlow: CNN 卷积神经网络 1](https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/5-03-CNN1/)
* [莫烦PYTHON - TensorFlow: CNN 卷积神经网络 2](https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/5-04-CNN2/)
* [莫烦PYTHON - TensorFlow: CNN 卷积神经网络 3](https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/5-05-CNN3/)

For more detail about CNN, you can watch the following video on YouTube.
* CNN (Convolutional Neural Network)

    [![CNN (Convolutional Neural Network)](http://img.youtube.com/vi/jajksuQW4mc/0.jpg)](https://www.youtube.com/watch?v=jajksuQW4mc)

---
## Execution

1. Run `main.py`
    ```bash
    # Make sure your current directory is in this folder
    $ python3 main.py
    ```
2. If succeed, you will get the following result (take few minutes)
    ```bash
    # If you run the program first time, you may download the datasets first (optional)
    Downloading data from https://s3.amazonaws.com/img-datasets/mnist.npz
    11493376/11490434 [==============================] - 12s 1us/step
    
    # If you have already run the pregram before, you may see the following information (optional)
    Extracting MNIST_data/train-images-idx3-ubyte.gz
    Extracting MNIST_data/train-labels-idx1-ubyte.gz
    Extracting MNIST_data/t10k-images-idx3-ubyte.gz
    Extracting MNIST_data/t10k-labels-idx1-ubyte.gz

    # The probability of prediction (the result is not unique)
    ('Step   0', 0.055)
    ('Step  50', 0.72)
    ('Step 100', 0.832)
    ('Step 150', 0.871)
    ('Step 200', 0.89)
    ('Step 250', 0.914)
    ('Step 300', 0.917)
    ('Step 350', 0.924)
    ('Step 400', 0.931)
    ('Step 450', 0.938)
    ('Step 500', 0.939)
    ('Step 550', 0.937)
    ('Step 600', 0.948)
    ('Step 650', 0.946)
    ('Step 700', 0.95)
    ('Step 750', 0.951)
    ('Step 800', 0.956)
    ('Step 850', 0.958)
    ('Step 900', 0.958)
    ('Step 950', 0.961)
    ```

---
## References

* [TensorFlow Official - Tutorial](https://www.tensorflow.org/tutorials/)
* [GitHub - tensorFlow/tensorflow](https://github.com/tensorflow/tensorflow)
* [莫烦PYTHON - TensorFlow](https://morvanzhou.github.io/tutorials/machine-learning/tensorflow)