# CNN3D Sample Codes

- The model will ingest 3D inputs and output pin-point prediction.
- I tested the codes on Python3.7 & Tensorflow2.4, EC2 g4dn.xlarge.
- BTW, it might throw an error depends on the machine you run these codes.

### Anyway, to begin with, clone the repository (through https)
```
git clone https://github.com/luo-jason/CNN3D_Sample.git
```
### make sure that you have tensorflow installed. 
**(Better to use a Python virtaul environment to run the codes)**
```
cd CNN3D_Sample
pip install -r requirements.txt
```

### Run the `train.py` to run the training. It will save the model at the end of the training
```
python train.py
```

### After you run the train.py, you can run `prediction.py` to do the prediction.
```
python prediction.py
```

### You can find details of the model architecture in `cnn3d_model.py`
