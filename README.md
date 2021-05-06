# Fish classification
## Download Dataset
1. Go to your kaggle account, scroll to API section and click on Create New API Token - It will download kaggle.json file on your machine.

2. Save that file into google drive and give the path to the folder, where you have saved the kaggle.json file in below command.

```
import os
os.getcwd()
os.environ['KAGGLE_CONFIG_DIR'] = '/content/drive/MyDrive/workspace/kaggle'
```

3. That's it. Now you can access your kaggle account from colab. To download the dataset from the kaggle run this command. Link to the dataset: [A Large Scale Fish Dataset | Kaggle](https://www.kaggle.com/crowww/a-large-scale-fish-dataset)


```
!pip install kaggle
!kaggle datasets download -d crowww/a-large-scale-fish-dataset
```

## Train the model
To train the model, go to the `./src` folder and run the following command. or if you don't want to train the model from scratch, you can download the trained weights from [here](https://drive.google.com/drive/folders/1AQlKd3QJXSDXU_NjA_KyuQbf63lHs2p9?usp=sharing) and directly test the model. 
```
python main.py --data_path="/path-to-dataset" \
                --weigths_path="/path-to-model-weights" \
                --test_img="/path-to-test-images" \
                --train
```
## Test the model
To test the model, go to the `./src` folder and run the following command.
```
python main.py --data_path="/path-to-dataset" \
                --weigths_path="/path-to-model-weights" \
                --test_img="/path-to-test-images" \
                --test
```

## Results

### Accuracy plot for training and validation:
![acc image](https://github.com/shruti-bt/fish-classification/blob/master/outputs/images/loss.png?raw=true)

### Loss plot for training and validation:
![acc image](outputs\images\loss.png)

### Classification results for test images:
![acc image](outputs\images\results.png)
