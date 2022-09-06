# sklearned

Surrogate experiments 


### Experimental choices 

1. Choose a skater from [timemachines](https://github.com/microprediction/timemachines)
2. Choose an optimizer from [humpday](https://github.com/microprediction/humpday)
3. Choose an embedding from the cube into the space of keras models, such as [keras_mostly_linear](https://github.com/microprediction/sklearned/blob/main/sklearned/embeddings/kerasmodels.py)
4. Choose a search strategy and data augmentation strategy
5. Choose a forecast horizon k, such as k=1
6. Choose an input vector length, such as 80 or 160
7. Choose a subset of live data streams from [stream listing](https://www.microprediction.org/browse_streams.html) 
8. Run the skater to generate training data
9. Challenge the existing out-of-sample champion model 

The names of experiment files indicate choices made. 

### M1
Maybe..
    

    brew install cmake protobuf
    pip3 install --pre -i https://pypi.anaconda.org/scipy-wheels-nightly/simple numpy
    conda install -c apple tensorflow-deps 
    python3 -m pip install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.12.0-py3-none-any.whl
    pip3 install onnx
   

