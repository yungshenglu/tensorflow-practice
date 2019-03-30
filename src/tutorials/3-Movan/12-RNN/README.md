# Practice 12 - RNN in TensroFlow

This practice is refer to the following resources credited to [Morvan](https://github.com/MorvanZhou).
* [莫烦PYTHON - TensorFlow: 什么是循环神经网络 RNN (Recurrent Neural Network)](https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/5-07-A-RNN/)
* [莫烦PYTHON - TensorFlow: 什么是 LSTM 循环神经网络](https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/5-07-B-LSTM//)
* [莫烦PYTHON - TensorFlow: RNN 循环神经网络](https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/5-07-RNN1/)
* [莫烦PYTHON - TensorFlow: RNN LSTM 循环神经网络 (分类例子)](https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/5-08-RNN2/)
* [莫烦PYTHON - TensorFlow: RNN LSTM (回归例子)](https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/5-09-RNN3/)
* [莫烦PYTHON - TensorFlow: RNN LSTM (回归例子可视化)](https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/5-10-RNN4/)

For more detail about RNN and LSTM, you can watch the following video on YouTube.
* [Recurrent Neural Network (RNN)](http://img.youtube.com/vi/H3ciJF2eCJI/0.jpg)

    [![Recurrent Neural Network (RNN)](http://img.youtube.com/vi/H3ciJF2eCJI/0.jpg)](https://www.youtube.com/watch?v=H3ciJF2eCJI)

* [Long-Short Term Memory (LSTM)](http://img.youtube.com/vi/V3D5ovKE9Og/0.jpg)

    [![Long-Short Term Memory (LSTM)](http://img.youtube.com/vi/V3D5ovKE9Og/0.jpg)](https://www.youtube.com/watch?v=V3D5ovKE9Og)

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