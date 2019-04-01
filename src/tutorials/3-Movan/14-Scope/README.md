# Practice 14 - TensorFlow Scope

This practice is refer to the following resources credited to [Morvan](https://github.com/MorvanZhou).
* [莫烦PYTHON - TensorFlow: scope 命名方法](https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/5-12-scope/)

---
## Execution

### Name Scope

1. Run `main1.py`
    ```bash
    # Make sure your current directory is in this folder
    $ python3 main1.py
    ```
2. If succeed, you will get the following result
    ```bash
    var1:0
    [1.]
    a_name_scope/var2:0
    [2.]
    a_name_scope/var2_1:0
    [2.1]
    a_name_scope/var2_2:0
    [2.2]
    ```
    * The name scope will be no effect if you use `tf.get_variable` to define the variable
    * The name of variable will add an identifier (i.e., `_1`, `_2`, etc.) to distinguish different variables with same name if you use `tf.Variable` to define the variable

### Variable Scope

1. Run `main2.py`
    ```bash
    # Make sure your current directory is in this folder
    $ python3 main2.py
    ```
2. If succeed, you will get the following result
    ```bash
    a_variable_scope/var3:0
    [3.]
    a_variable_scope/var4:0
    [4.]
    a_variable_scope/var4_1:0
    [4.]
    a_variable_scope/var3:0
    [3.]
    ```
    * The variable scope will be affect if you use `tf.get_variable` to define the variable
    * `tf.Variable` will define new variable every time
    * To reuse the variable, you need to add `scope.reuse_variables()` before reusing; otherwise, it may have error

### Scope in RNN

1. Run `main3.py`
    ```bash
    # Make sure your current directory is in this folder
    $ python3 main3.py
    ```
2. If succeed, you will get the following result
    ```bash
    input_layer/Weights:0
    Traceback (most recent call last):
    File "main3.py", line 161, in <module>
        main()
    File "main3.py", line 141, in main
        wrongReuseParam(train_config, test_config)
    File "main3.py", line 120, in wrongReuseParam
        test_rnn1 = RNN(test_config)
    File "main3.py", line 33, in __init__
        self.buildRNN()
    File "main3.py", line 48, in buildRNN
        Weights_in = self.weightVar([self.input_size, self.cell_size])
    File "main3.py", line 103, in weightVar
        return tf.get_variable(shape=shape, initializer=init, name=name)
    ...
    ...
    ValueError: Variable input_layer/Weights already exists, disallowed. Did you mean to set reuse=True or reuse=tf.AUTO_REUSE in VarScope? Originally defined at:

    File "main3.py", line 103, in weightVar
        return tf.get_variable(shape=shape, initializer=init, name=name)
    File "main3.py", line 48, in buildRNN
        Weights_in = self.weightVar([self.input_size, self.cell_size])
    File "main3.py", line 33, in __init__
        self.buildRNN()
    ```
    * In line 145 (`main3.py`), we call the method `wrongReuseParam` but it causes an error that the variable `input_layer/Weights` already exists
3. Comment out the **line 145** and remove the comment symbol `#` in **line 148** as follow
    ```python
    # WRONG to reuse parameters in train RNN
    #wrongReuseParam(train_config, test_config)

    # NO reuse parameters in train RNN 
    noReuseParam(train_config, test_config)
    ```
4. Run `main3.py` again
    ```bash
    # Make sure your current directory is in this folder
    $ python3 main3.py
    ```
5. If succeed, you will get the following result
    ```bash
    train_rnn/input_layer/Weights:0
    test_rnn/input_layer/Weights:0
    ```
    * In line 148 (`main3.py`), we call the method `noReuseParam` that avoid the error mentioned above; however, it just create two different variable which **doesn't reuse the variables in train RNN**
6. Comment out the **line 148** and remove the comment symbol `#` in **line 151** as follow
    ```python
    # NO reuse parameters in train RNN
    #noReuseParam(train_config, test_config)
    
    # CORRECT to reuse parameters in train RNN
    correctReuseParam(train_config, test_config)
    ```
7. Run `main3.py` again
    ```bash
    # Make sure your current directory is in this folder
    $ python3 main3.py
    ```
8. If succeed, you will get the following result
    ```bash
    rnn/input_layer/Weights:0
    rnn/input_layer/Weights:0
    ```
    * In line 151 (`main3.py`), we call the method `correctReuseParam` that reuse the variables in train RNN

---
## References

* [TensorFlow Official - Tutorial](https://www.tensorflow.org/tutorials/)
* [GitHub - tensorFlow/tensorflow](https://github.com/tensorflow/tensorflow)
* [莫烦PYTHON - TensorFlow](https://morvanzhou.github.io/tutorials/machine-learning/tensorflow)