# Practice 8 - Using TensorBoard 2

This practice is refer to the following resources credited to [Morvan](https://github.com/MorvanZhou).
* [莫烦PYTHON - TensorFlow: Tensorboard 可视化好帮手 2](https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/4-2-tensorboard2/)

---
## Execution

1. Run `main.py`
    ```bash
    # Make sure your current directory is in this folder
    $ python3 main.py
    ```
2. If succeed, you can check whether the file is generated in `logs/`
    ```bash
    # Make sure your current directory is in this folder
    $ ls ./logs/
    events.out.tfevents.1553929511.yungshenglu
    # The above filename is for example and won't be same with yours!
    ```
3. Run the log with TensorBoard
    ```bash
    $ tensorboard --logdir='./logs/'
    TensorBoard 1.13.0 at http://yungshenglu:6006 (Press CTRL+C to quit)
    # The above link is just an example. Please use your link!
    ```
4. Open the browser and navigate to the website show in your terminal (`http://0.0.0.0:6006`) (the result is not unique)
    ![](../../../../res/img/movan/8-tensorboard.png)

---
## References

* [TensorFlow Official - Tutorial](https://www.tensorflow.org/tutorials/)
* [GitHub - tensorFlow/tensorflow](https://github.com/tensorflow/tensorflow)
* [莫烦PYTHON - TensorFlow](https://morvanzhou.github.io/tutorials/machine-learning/tensorflow)