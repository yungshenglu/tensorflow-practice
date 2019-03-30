# Practice 9 - Classification with TensorFLow

This practice is refer to [莫烦PYTHON - TensorFlow: Classification 分类学习](https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/5-01-classifier/). Credits to [Morvan](https://github.com/MorvanZhou).

---
## Execution

1. Run `main.py`
    ```bash
    # Make sure your current directory is in this folder
    $ python3 main.py
    ```
2. If succeed, you will get the following result
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
    Step   0: 0.1184
    Step  50: 0.6477
    Step 100: 0.7448
    Step 150: 0.7807
    Step 200: 0.8077
    Step 250: 0.8186
    Step 300: 0.8275
    Step 350: 0.83
    Step 400: 0.8451
    Step 450: 0.8517
    Step 500: 0.857
    Step 550: 0.854
    Step 600: 0.8588
    Step 650: 0.8625
    Step 700: 0.864
    Step 750: 0.8695
    Step 800: 0.87
    Step 850: 0.8721
    Step 900: 0.8728
    Step 950: 0.8729
    ```

---
## References

* [TensorFlow Official - Tutorial](https://www.tensorflow.org/tutorials/)
* [GitHub - tensorFlow/tensorflow](https://github.com/tensorflow/tensorflow)
* [莫烦PYTHON - TensorFlow](https://morvanzhou.github.io/tutorials/machine-learning/tensorflow)