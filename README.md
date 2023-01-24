## CuCounter: A Python *k*mer frequency counter object based on a massively parallel CUDA hash table

## Installation
CuStats requires NumPy and CuPy. It also currently only supports Nvidia GPUs with CUDA.
CuStats is currently not on PyPI, so in order to install CuStats:
* clone the CuStats repository
* use pip to install all necessary dependencies as well as CuStats from inside the cloned repository
```Bash
git clone https://github.com/jorgenwh/custats.git
cd custats
pip install -r requirements.txt
pip install .
```

## Usage
All of CuStats' functions will accept either NumPy or CuPy arrays.
CuPy arrays are preferred as it circumvents having to copy memory back and fourth between the host and device.
NumPy is used in the example below, but the same code would work if NumPy had been replaced with CuPy.
```Python
import numpy as np
import custats

...
```
