import libmr
from itertools import compress

import keras.backend as K
import numpy as np
import os
import scipy.ndimage.interpolation as interp
import skimage.transform as skt
from utils.custom_losses import siamese_loss
from keras.layers import GlobalAveragePooling2D, Dropout, Dense, GlobalAveragePooling3D
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import rmsprop, adam, SGD
from nutsflow import *
from nutsml import *
from scipy.spatial.distance import mahalanobis
from sklearn.metrics import roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt

from hyperparameters import *
from utils.custom_networks import squeeze_net

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def rotate_nd_image(image, angle):
    image_ = image[:, :, :, 0]
    rot_axis = tuple(np.random.choice([0, 1, 2], 2, replace=False))
    image_ = interp.rotate(image_, angle, axes=rot_axis, reshape=False, order=0)
    image_ = np.expand_dims(image_, axis=-1)
    return image_


def flip_lr_nd_image(image):
    return np.flip(image, np.random.choice([2, 3], 1, replace=False).tolist()[0])


def resize_nd_image(image):
    image_ = skt.resize(image, CNN_INPUT_SHAPE, order=0)
    return image_


@nut_processor
def make_serial(iterable):
    for sample in iterable:
        for i in range(0, sample[0].shape[0]):
            yield tuple([sample[j][i, :] for j in range(0, len(sample))])


@nut_processor
def flat_tuple(iterable):
    for sample in iterable:
        yield tuple([sample[i][j] for i in range(0, len(sample)) for j in range(0, len(sample[i]))])


@nut_processor
def repeat_elements(iterator, num_repeats=3):
    for sample in iterator:
        for i in range(0, num_repeats):
            yield sample


@nut_function
def ReadImage_3D(sample, columns, pathfunc=None, as_grey=False):
    """

    :param tuple|list sample: ('nut_color', 1)
    :param None|int|tuple columns: Indices of columns in sample to be replaced
                              by image (based on image id in that column)
                              If None then a flat samples is assumed and
                              a tuple with the image is returned.
    :param string|function|None pathfunc: Filepath with wildcard '*',
      which is replaced by the imageid provided in the sample, e.g.
      'tests/data/img_formats/*.jpg' for sample ('nut_grayscale', 2)
      will become 'tests/data/img_formats/nut_grayscale.jpg'
      or
      Function to compute path to image file from sample, e.g.
      lambda sample: 'tests/data/img_formats/{1}.jpg'.format(*sample)
      or
      None, in this case the image id is take as filepath.
    :param as_grey: If true, load as grayscale image.
    :return: Sample with image ids replaced by image (=ndarray)
            of shape (h, w, c) or (h, w)
    :rtype: tuple
    """

    def load(fileid):
        """Load image for given fileid"""
        if isinstance(pathfunc, str):
            filepath = pathfunc.replace('*', fileid)
        elif hasattr(pathfunc, '__call__'):
            filepath = pathfunc(sample)
        else:
            filepath = fileid
        return np.load(filepath)

    if columns is None:
        return (load(sample),)  # image as tuple with one element

    colset = as_set(columns)
    elems = enumerate(sample)
    return tuple(load(e) if i in colset else e for i, e in elems)


if __name__ == "__main__":

    assert os.path.isfile(instance_info_csv)

    data = ReadCSV(instance_info_csv, skipheader=1)
    data = data >> Collect()

    k_fold = 0
    while k_fold < CROSS_VALID_FOLDS:
        log_cols_test = LogCols('./outputs/evt_mil_3d_' + pathology + '_k_' + str(k_fold) + 'new_100_nimg1.csv', cols=None,
                                colnames=('epoch', 'auc_k', 'auc_m'))

        # Get a new set of train and test data
        is_train = lambda s: int(s[-1]) != k_fold
        train_data = data >> Filter(is_train) >> Collect()
        test_data = data >> FilterFalse(is_train) >> Collect()
        print len(train_data >> GetCols(1) >> Collect(set)), len(test_data >> GetCols(1) >> Collect(set))

        train_bags = train_data >> GetCols(1) >> Collect(set)
        test_bags = test_data >> GetCols(1) >> Collect(set)
        print train_bags.intersection(test_bags)

        assert len(train_bags.intersection(test_bags)) == 0

        train_data_ = train_data >> Shuffle(5000) >> GetCols(0, 2) >> Collect()

        img_reader = ReadImage_3D(0, image_root, as_grey=False)

        # image augmentation and normalization, custom defined to accommodate 3D instances
        TransformImage.register('rotate_nd_image', rotate_nd_image)
        TransformImage.register('flip_lr_nd_image', flip_lr_nd_image)

        TransformImage.register('resize_nd_image', resize_nd_image)

        augment_img = (AugmentImage(0)
                       .by('identical', 0.5)
                       .by('flip_lr_nd_image', .5))

        rotate_image = (AugmentImage(0)
                        .by('identical', 0.5)
                        .by('rotate_nd_image', 0.5, [-10, 10]))

        resize_image = (AugmentImage(0)
                        .by('resize_nd_image', 1.0))

        normalize_image = lambda s: (s - IMAGE_MEAN) / IMAGE_SD
        add_batch_dim = lambda s: np.expand_dims(s, 0)

        # Batching
        build_batch_train = (BuildBatch(BATCH_SIZE, prefetch=0)
                             .by(0, 'vector', 'float32')
                             .by(1, 'one_hot', 'uint8', 2))

        # Define CNN model
        base_model = squeeze_net(input_shape=CNN_INPUT_SHAPE, regularize_weight=0.000001)
        x = base_model.output
        x = GlobalAveragePooling3D()(x)
        x = Dropout(0.25)(x)
        x_f = Dense(FEATURE_LEN, activation='relu', name='final_features')(x)
        predictions = Dense(2, activation='softmax', name='softmax_')(x_f)

        model = Model(inputs=base_model.input, outputs=[predictions, x_f])
        model.summary()
        opti = adam(clipnorm=1.)

        loss_w_1 = K.variable(K.cast_to_floatx(1.0))
        loss_w_2 = K.variable(K.cast_to_floatx(0.0))
        model.compile(optimizer=opti, loss=['categorical_crossentropy', siamese_loss],
                      loss_weights=[loss_w_1, loss_w_2])  # original 0.001
        model_features = Model(inputs=model.input, outputs=model.get_layer('final_features').output)

        def train_model(sample):
            """
            :param sample: (image, true_label)
            :return: batch loss
            """
            outp = model.train_on_batch(sample[0], [sample[1], sample[1]])
            return outp


        def extract_model_features(sample):
            """
            :param sample: (image, true label)
            :return: (features [BATCH_SIZE, FEATURE_DIM], class predictions [BATCH_SIZE, 2], true labels [BATCH_SIZE, 2])
            """
            outp, _ = model.predict(sample[0])
            out_features = model_features.predict(sample[0])
            return out_features, outp, sample[1]


        def predict_test_instance(sample):
            """
            :param sample: (image, true label)
            :return: (positive probability)
            """
            instance = np.expand_dims(sample[0], axis=0)
            outp, f = model.predict(instance)

            return outp[0, 1], f


        check_batch_shape = lambda s: s[0].shape[0] == BATCH_SIZE
        get_class = lambda s: np.int8(np.argmax(s, axis=-1))
        is_true_negative = lambda s: (int(s[1]) == 0 and int(s[2]) == 0)
        is_negative = lambda s: int(s[2]) == 0

        for e in range(0, NUM_EPOCH):
            
            if not e == 0:

                train_data_ >> Stratify(1, 'up') >> Shuffle(2000) >> img_reader >> Shuffle(
                    100) >> rotate_image >> MapCol(0,
                                                   add_batch_dim) >> augment_img >> MapCol(
                    0, normalize_image) >> Shuffle(
                    100) >> build_batch_train >> Filter(check_batch_shape) >> Map(
                    train_model) >> Consume()

            model.save_weights(
                weight_save_root + 'evt_mil_3d_weights_' + pathology + '_k_' + str(k_fold) + 'new_100_nimg1.h5')
            if e % 20 == 0 and e != 0:
                model.save_weights(
                    weight_save_root + 'evt_mil_3d_weights_' + pathology + '_k_' + str(k_fold) + '_e' + str(
                        e) + 'new_100_nimg1.h5')

            # extract F-features for all training instances
            out_features = train_data >> GetCols(0, 2) >> img_reader >> NOP(resize_image) >> MapCol(0,
                                                                                                    add_batch_dim) >> MapCol(
                0, normalize_image) >> build_batch_train >> Filter(check_batch_shape) >> Map(
                extract_model_features) >> make_serial() >> Collect()

            # This step added to make sure the feature length is divisible by batch size
            num_images = len(out_features)
            bag_names = train_data >> GetCols(1) >> Take(num_images) >> Collect()
            instance_names = train_data >> GetCols(0) >> Take(num_images) >> Collect()

            # Collect F-feature of true negative instances
            true_neg_features = out_features >> MapCol(1, get_class) >> MapCol(2, get_class) >> Filter(
                is_true_negative) >> GetCols(0) >> Collect()

            # if not enough instances get gt negatives
            if np.asarray(true_neg_features).shape[0] < MR_TAIL_SIZE:
                print "Running out of true negatives ..."
                true_neg_features = out_features >> MapCol(1, get_class) >> MapCol(2, get_class) >> Filter(
                    is_negative) >> GetCols(0) >> Collect()

            true_neg_features = np.asarray(true_neg_features)[:, 0, :]
            # TODO: Cluster true negative features to get more than one center
            feature_mean = np.mean(true_neg_features, axis=0, keepdims=False)
            feature_cov_inv = np.linalg.pinv(np.cov(true_neg_features, rowvar=False))
            weight_save_root + 'evt_mil_3d_mean_' + pathology + '_k_' + str(k_fold) + '.npy'
            np.save(weight_save_root + 'evt_mil_3d_mean_' + pathology + '_k_' + str(k_fold) + '.npy', feature_mean)
            np.save(weight_save_root + 'evt_mil_3d_covi_' + pathology + '_k_' + str(k_fold) + '.npy', feature_cov_inv)

            train_data_ = list()

            # get image wise neg features for EVT fitting
            l2_dist = lambda s: mahalanobis(s, feature_mean, feature_cov_inv)
            neg_dist = out_features >> MapCol(2, get_class) >> Zip(instance_names) >> flat_tuple() >> Filter(
                is_negative) >> GetCols(0) >> MapCol(0, l2_dist) >> Collect()
            neg_dist = np.asarray(neg_dist, dtype=np.double).flatten()

            neg_pred_class = out_features >> MapCol(2, get_class) >> Zip(instance_names) >> flat_tuple() >> Filter(
                is_negative) >> GetCols(1) >> MapCol(0, get_class) >> Collect()
            neg_pred_class = np.asarray(neg_pred_class, dtype=np.double).flatten()

            neg_instance_names = out_features >> MapCol(2, get_class) >> Zip(instance_names) >> flat_tuple() >> Filter(
                is_negative) >> GetCols(3) >> Flatten() >> Collect()

            neg_bag_names = out_features >> MapCol(2, get_class) >> Zip(bag_names) >> flat_tuple() >> Filter(
                is_negative) >> GetCols(3) >> Flatten() >> Collect(set)

            neg_ext_dist = list()
            for neg_image in neg_bag_names:
                image_indices = [neg_image in pn for pn in neg_instance_names]
                instance_dist = np.asarray(list(compress(neg_dist, image_indices)))
                instance_class = np.asarray(list(compress(neg_pred_class, image_indices)))

                # True negatives
                instance_dist_ = [d for d, c in zip(instance_dist, instance_class) if c < 0.5]

                if len(instance_dist_) > 2:
                    neg_ext_dist.append(np.max(instance_dist_))

            # not enough True negative samples
            if len(neg_ext_dist) < 10:
                print "Not enough true negatives 2"
                neg_ext_dist = list()
                for neg_image in neg_bag_names:
                    image_indices = [neg_image in pn for pn in neg_instance_names]
                    instance_dist = np.asarray(list(compress(neg_dist, image_indices)))
                    instance_class = np.asarray(list(compress(neg_pred_class, image_indices)))

                    neg_ext_dist.append(np.max(instance_dist))

            neg_ext_dist = np.asarray(neg_ext_dist, dtype=np.double)

            # fit EVT
            mr = libmr.MR()
            fit_tail_size = len(neg_ext_dist)
            mr.fit_high(neg_ext_dist, fit_tail_size)

            # sample from negative bags
            # Calculate EVT prob for Negatives
            if mr.is_valid:
                probs = mr.w_score_vector(neg_dist)
            else:
                print "evt not valid"
                probs = np.ones(neg_dist.shape, dtype=np.double)
            probs = probs + np.finfo(np.double).eps
            probs = probs / (np.sum(probs))

            for neg_image in neg_bag_names:
                image_indices = [neg_image in pn for pn in neg_instance_names]
                instance_prob_ = np.asarray(list(compress(probs, image_indices)))
                instance_names_ = list(compress(neg_instance_names, image_indices))

                instance_prob_ = instance_prob_ / (np.sum(instance_prob_))

                n_neg_samples = int(np.floor(instance_prob_.shape[0] * .75))
                neg_samples = np.random.choice(range(0, len(instance_names_)), n_neg_samples, replace=False,
                                               p=instance_prob_)

                for i in range(0, n_neg_samples):
                    train_data_.append((instance_names_[neg_samples[i]], 0))

            # sample from positive bags
            pos_dist = out_features >> MapCol(2, get_class) >> Zip(instance_names) >> flat_tuple() >> FilterFalse(
                is_negative) >> GetCols(0) >> MapCol(0, l2_dist) >> Collect()
            pos_dist = np.asarray(pos_dist, dtype=np.double).flatten()

            pos_instance_names = out_features >> MapCol(2, get_class) >> Zip(
                instance_names) >> flat_tuple() >> FilterFalse(
                is_negative) >> GetCols(3) >> Flatten() >> Collect()

            pos_bag_names = out_features >> MapCol(2, get_class) >> Zip(bag_names) >> flat_tuple() >> FilterFalse(
                is_negative) >> GetCols(3) >> Flatten() >> Collect(set)

            # Calcuate EVT prob for positives
            if mr.is_valid:
                probs = mr.w_score_vector(pos_dist)
            else:
                probs = np.ones(pos_dist.shape, dtype=np.double)

            count = {'high_pos': 0, 'ok_pos': 0, 'low_pos': 0}
            for pos_image in pos_bag_names:
                image_indices = [pos_image in pn for pn in pos_instance_names]
                instance_names = list(compress(pos_instance_names, image_indices))
                instance_probs = np.asarray(list(compress(probs, image_indices)))
                instance_dist = np.asarray(list(compress(pos_dist, image_indices)))

                n_pos = instance_probs > MR_THRESHOLD
                MR_K_MAX = np.int(np.floor(len(instance_names) * .95))
                if np.sum(n_pos) > MR_K:
                    if np.sum(n_pos) > MR_K_MAX:
                        count['high_pos'] = count['high_pos'] + 1
                        sampled_instances = [instance_names[i] for i in np.argsort(instance_dist)[-MR_K_MAX:]]
                        for s in sampled_instances:
                            train_data_.append((s, 1))

                        sampled_instances = [instance_names[i] for i in np.argsort(instance_dist)[:-MR_K_MAX]]
                        for s in sampled_instances:
                            train_data_.append((s, 0))

                    else:
                        count['ok_pos'] = count['ok_pos'] + 1
                        sampled_instances = list(compress(instance_names, n_pos))
                        for s in sampled_instances:
                            train_data_.append((s, 1))

                        n_neg = np.int(np.sum(instance_probs < MR_THRESHOLD))

                        n_neg = np.minimum(n_neg, MR_K_MAX)
                        sampled_instances = [instance_names[i] for i in np.argsort(instance_dist)[:n_neg]]
                        for s in sampled_instances:
                            train_data_.append((s, 0))

                else:
                    count['low_pos'] = count['low_pos'] + 1
                    sampled_instances = [instance_names[i] for i in np.argsort(instance_dist)[-MR_K:]]

                    for s in sampled_instances:
                        train_data_.append((s, 1))

            print count

            train_data_ = train_data_ >> Shuffle(np.floor(len(train_data_) * .9)) >> Collect()

            # Testing image wise predictions
            test_bag_names = test_data >> GetCols(1) >> Flatten() >> Collect(set)

            test_y = list()
            test_y_hat_k = list()
            test_y_hat_m = list()
            for test_image in test_bag_names:
                matching_image = lambda v: test_image in v[0]
                instances_names = test_data >> Filter(matching_image) >> GetCols(0) >> Collect()
                image_class = test_data >> Filter(matching_image) >> GetCols(2) >> Flatten() >> Collect(set)

                assert len(image_class) == 1
                image_class = int(image_class.pop())

                test_out = instances_names >> img_reader >> NOP(resize_image) >> MapCol(0, normalize_image) >> Map(
                    predict_test_instance) >> Collect()

                preds = list()
                f_dist = list()
                for p, d in test_out:
                    preds.append(p)

                preds = np.asarray(preds)

                test_y.append(image_class)
                test_y_hat_k.append(np.sort(preds)[-MR_K])
                test_y_hat_m.append(np.max(preds))

            auc_k = roc_auc_score(test_y, test_y_hat_k)
            auc_m = roc_auc_score(test_y, test_y_hat_m)
            confm = confusion_matrix(test_y, [1 if ii > 0.5 else 0 for ii in test_y_hat_k])
            print 'test auc: ', auc_k, auc_m, confm[0, 0], confm[0, 1], confm[1, 0], confm[1, 1]

            [(e, auc_k, auc_m, confm[0, 0], confm[0, 1], confm[1, 0], confm[1, 1]), ] >> log_cols_test >> Consume()

        K.clear_session()
        k_fold = k_fold + 1
