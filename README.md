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

    @INPROCEEDINGS{tennakoon_2018accv, 
		author={R. Tennakoon and A. K. Gostar and  R. Hoseinnezhad and M. De-Bruijne and A. Bab-Haidashar}, 
		booktitle={14th Asian Conference on Computer Vision (ACCV)}, 
		title={Deep Multi-Instance Volumetric Image Classification with Extreme Value Distributions}, 
		year={2018}, 
		number={}, 
		pages={}, 
		ISSN={}, 
		month={Accepted for publication, December},}
    
