# Text Detection with EAST

EAST is an "Efficient and Accurate Scene Text Detector" which was [proposed](https://arxiv.org/abs/1704.03155)
 in 2017. 

##

This is a TensorFlow and Keras based implementation of EAST with few modifications.

 - Uses ResNet-50 instead of PVANet
 - Dice loss instead of binary cross-entropy loss.
 - [AdamW](https://github.com/shaoanlu/AdamW-and-SGDW) instead of Adam optimizer.

## How to run the code?

### Install required modules

    python3 -m venv venv
    source venv/bin/activate
    cd project_directory
    pip install -r requirements.txt
    
### DATASET

It is trainable on any dataset, given they follow the annotation formats like below - 

**JSON**

    [
	    {
	        "line": [87, 212, 137, 212, 137, 230, 87, 230],
	        "text": "NAME"
	    },
	    {
	        "line": [87, 243, 147, 243, 147, 263, 87, 263],
	        "text": "TITLE"
	    },
	]

**TXT**

    87, 212, 137, 212, 137, 230, 87, 230, NAME
    87, 243, 147, 243, 147, 263, 87, 263, TITLE

Image and annotation file names should be same and in the same directory.

### Train

The script supports a single CPU and GPU configuration.		

    python train.py --training_data_path=path/to/training_data --validation_data_path=path/to/validation_data --max_epochs=200
   
The script accepts several other arguments which can be referred to from the script. 

### Predictions

    python predict.py --test_data_path=path/to/test_data --model_path=path/to/model.h5

A text file is generated for each image consisting of bounding box represented as lines. 

    x_left, y_top, x_right, y_top, x_right, y_bottom, x_left, y_bottom
