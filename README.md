# EVT-MIL: Deep Multi-Instance Volumetric Image Classification with Extreme Value Distributions

## Running the code
The code was tested in _Keras_ with _Tensorflow_ backend. 
The packages needed are listed in the `requirements.txt` (not all packages in file are used. Just did a pip freeze for my virtual environment.)

### Installing python virtual environment and requirements
 ```
 pip install virtualenv
 virtualenv --no-site-packages vkeras
 source vkeras/bin/activate
 pip install -r path/to/requirements.txt

 ```


### Prepare data
For each data-set:

- Convert each instance into a numpy array with dimensions `[height, width, channels]` and save in a pre specified folder.
- Create a csv file containing the following columns: `['instance_file_name', 'bag_name', 'bag_label', <additional information,>, 'Cross-validation_split']` for each instance. instance_file_name can be bag_name + a unique number,  Cross-validation_split should be a number starting from zero.


### Training the network and obtaining validation results
- Adjust hyper parameters and paths defined in the file `hyperparameters.py`
- Run `train_3d_evt_mil_cv.py`

The AUC values for validation data, at each epoch, will be writen to a .csv file in ./outputs folder 


## Publication

If you find this work useful in your research, please consider citing:

	@ARTICLE{8805413,
	author={R. {Tennakoon} and G. {Bortsova} and S. {Ørting} and A. K. {Gostar} and M. M. W. {Wille} and Z. {Saghir} and R. {Hoseinnezhad} and M. {de Bruijne} and A. {Bab-Hadiashar}},
	journal={IEEE Transactions on Medical Imaging},
	title={Classification of Volumetric Images Using Multi-Instance Learning and Extreme Value Theorem},
	year={2019},
	volume={},
	number={},
	pages={1-1},
	doi={10.1109/TMI.2019.2936244},}
    
