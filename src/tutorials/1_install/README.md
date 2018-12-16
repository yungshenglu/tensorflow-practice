# Tutorial 1 - Getting Started

We are going to install Tensorflow in this tutorial.

* **NOTICE:** TensorFlow is tested and supported on the following 64-bit systems:
    * Ubuntu 16.04 or later
    * Windows 7 or later
    * macOS 10.12.6 (Sierra) or later (no GPU support)
    * Raspbian 9.0 or later

---
## 1.1 Installation 

* **NOTICE:** 
    * The following instructions are for **Python 3 (Python 3.4, 3.5, 3.6)**.
    * For *Python 2.7*, you can follow [here](https://www.tensorflow.org/install/pip?lang=python2).

1. Prerequisite (for **Python 3 (Python 3.4, 3.5, 3.6)**)
    ```bash
    # Check if your Python environment is already configured
    $ python3 --version
    $ pip3 --version
    $ virtualenv --version
    # If the above packages are already installed, skip to the next step
    $ sudo apt-get update
    $ sudo apt-get install python3-dev python3-pip
    # Install for system-wide
    $ sudo pip3 install -U virtualenv
    ```
2. Create a virtual environment (recommended)
    ```bash
    # Create a new vitrual environment by choosing a Python interpreter and making a ./env directory to hold it
    $ virtualenv --system-site-packages -p python3 ./venv
3. Activate / Deactivate the virtual environement
    ```bash
    # Activate the virtual environement using a shell-specific command (e.g., sg, bash, etc.)
    $ source ./venv/bin/activate
    # When virtualenv is active, your shell prompt is prefixed with (venv).
    (venv) $
    # To exit virtualenv later
    (venv) $ deactivate
    ```
    * Install packages within a virtual environment without affecting the host system setup
4. Upgrade `pip` within a virtual environment
    ```bash
    # Start by upgrading pip:
    (venv) $ pip install --upgrade pip
    # Show packages installed within the virtual environment
    (venv) $ pip list
    ```
5. Install TensorFlow with `pip`
     ```bash
     # Current release for GPU-only (Python 2.7)
     (venv) $ pip install --upgrade tensorflow
     # GPU package for CUDA-enabled GPU cards (Python 2.7)
     (venv) $ pip install --upgrade tensorflow-gpu
     ```

---
## 1.2 Getting started!

1. Run the example program `./src/hello.py`
    ```bash
    # Make sure your current directory is "./src/"
    (venv) $ python hello.py
    b'Hello world, TensorFlow!'
    ```

---
## References

* [TensorFlow Offical](https://www.tensorflow.org/)
* [TensorFlow Official - Tutorial](https://www.tensorflow.org/tutorials/)
* [GitHub - tensorFlow/tensorflow](https://github.com/tensorflow/tensorflow)
* [GitHub - tensorflow/models](https://github.com/tensorflow/models)
* [GitHub - tendorflow/datasets](https://github.com/tensorflow/datasets)

---
## Contributor

* [David Lu](https://github.com/yungshenglu)

---
## License

Apache License 2.0