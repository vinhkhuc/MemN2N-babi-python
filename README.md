# End-To-End Memory Networks for bAbI tasks
This is the implementation of MemN2N model in Python for the [bAbI tasks](http://fb.ai/babi) as shown in the 
Section 4 of the paper "[End-To-End Memory Networks](http://arxiv.org/abs/1503.08895)". This code is based on 
the original [Matlab code](https://github.com/facebook/MemNN/tree/master/MemN2N-babi-matlab).

<p align="center">
<img src="https://raw.githubusercontent.com/vinhkhuc/MemN2N-babi-python/master/demo/web/static/memn2n-babi.gif" 
style="width:700px;">
<a href="https://github.com/vinhkhuc/MemN2N-babi-python#question-answering-demo">Web-based Demo</a>
</p>

## Requirements
* Python 2.7
* Numpy, Flask (optional, only for web-based demo) which can be installed using pip:
```
$ sudo pip install -r requirements.txt
```
* [bAbI dataset](http://fb.ai/babi) which can be downloaded by running:
```
$ wget -qO- http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz | tar xvz -C data
```

## Usage
* To run on a single task:
```
python babi_runner.py -t 1
```
The command will start training and then test on task 1. The output will look like:
```
Using data from data/tasks_1-20_v1-2/en
Train and test for task 1 ...
|=========================================         | 82% 0.3s
```

* To run on all 20 tasks:
```
python babi_runner.py -a
```

* To run using all training data from 20 tasks, use the joint mode:
```
python babi_runner.py -j
```

## Question Answering Demo
* In order to run the Web-based demo using the pretrained model `memn2n_model.pklz` in `trained_model/`, use:
```
python -m demo.qa
```
Note that the Web-based demo requires Flask. 

* Alternatively, there is a console-based demo which can started using:
```
python -m demo.qa -console
```

* The pretrained model `memn2n_model.pklz` was created using the command:
```
python -m demo.qa -train
```

* To show all options, run the script with `-h`.

## Benchmark
See the results [here](https://github.com/vinhkhuc/MemN2N-babi-python/tree/master/bechmarks).

### Author
Vinh Khuc

### License
BSD

### Future Plan
* Port to TensorFlow/Keras
* Support Python 3.
