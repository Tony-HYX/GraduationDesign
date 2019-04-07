# NLM-Reproduce


1. Install Swipl
	[http://www.swi-prolog.org/build/unix.html](http://www.swi-prolog.org/build/unix.html)


2. Set environment variables(Should change file path according to your situation)
```
# cd to GraduationDesign
export NLM_HOME=$PWD
cp /usr/local/lib/swipl/lib/x86_64-linux/libswipl.so.7 $NLM_HOME/src/logic/lib/  
export LD_LIBRARY_PATH=$NLM_HOME/src/lib:/usr/local/cuda:$LD_LIBRARY_PATH  
# for GPU user
# /usr/local/cuda:$LD_LIBRARY_PATH  
export SWI_HOME_DIR=/usr/local/lib/swipl/  
```

3. Install required package
```
pip3 install numpy
```

4. First change the `swipl_include_dir` and `swipl_lib_dir` in `setup.py` to your own SWI-Prolog path.
```
cd src/prolog_cpp_interface
python3 setup.py install
```



```
install numpy tensorflow keras
pip3 install numpy
pip3 install tensorflow
pip3 intall keras
```


5. Run equaiton generator
```
python3 equation_generator.py
```

6. Change arguments in test_keras_mnist4.py and run
```
python3 test_keras_mnist4.py
```

