# Practice 11 - TensroFlow Saver

This practice is refer to the following resources credited to [Morvan](https://github.com/MorvanZhou).
* [莫烦PYTHON - TensorFlow: Saver 保存读取](https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/5-06-save/)

---
## Execution

### Save the varibles to file

1. Run `save.py`
    ```bash
    # Make sure your current directory is in this folder
    $ python3 save.py
    ```
2. If succeed, you will get the following result
    ```bash
    [INFO] Save to the file: ./out/model.ckpt
    ```

### Restore the varibles to file

1. Run `restore.py`
    ```bash
    # Make sure your current directory is in this folder
    $ python3 restore.py
    ```
2. If succeed, you will get the following result
    ```bash
    [INFO] Restore from the file:  ./out/model.ckpt
    weights:
     [[1. 2. 3.]
     [3. 4. 5.]]
    biases:
     [[1. 2. 3.]]
    ```

---
## References

* [TensorFlow Official - Tutorial](https://www.tensorflow.org/tutorials/)
* [GitHub - tensorFlow/tensorflow](https://github.com/tensorflow/tensorflow)
* [莫烦PYTHON - TensorFlow](https://morvanzhou.github.io/tutorials/machine-learning/tensorflow)