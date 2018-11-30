pathology = 'EMC'
instance_info_csv = './data/instance_' + pathology + '_all_100.csv' #Path to csv file with image naes and bag labels
image_root = '/home/truwan/SSD/DATA/erasmus_data/Ruwan/patches_all_100/*' #Path to image patches
weight_save_root = './weights/'  # folder to save trained weights

CROSS_VALID_FOLDS = 4
IMAGE_MEAN = 0.
IMAGE_SD = 1.

BATCH_SIZE = 16
CNN_INPUT_SHAPE = (21, 41, 41, 1)
FEATURE_LEN = 64
NUM_EPOCH = 60

MR_TAIL_SIZE = 50
MR_THRESHOLD = 0.95
MR_K = 3
