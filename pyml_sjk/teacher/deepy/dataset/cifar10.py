import os
import pickle
import numpy as np

def unpickle_batch(filepath):
    with open(filepath, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')

    data = batch[b'data'].reshape(10000, 32, 32, 3, order='F')
    data = np.transpose(data, axes=[0, 2, 1, 3])
    labels = batch[b'labels']
    return data, labels

def load(root):
    batch_files = [os.path.join(root, 'data_batch_{}'.format(i)) for i in range(1, 6)]
    batches = [unpickle_batch(filepath) for filepath in batch_files]
    train_data = []
    train_labels = []
    for data, labels in batches:
        train_data.append(data)
        train_labels.extend(labels)

    X_train = np.vstack(train_data)
    y_train = np.array(train_labels)

    X_test, y_test = unpickle_batch(os.path.join(root, 'test_batch'))

    return (X_train, y_train), (X_test, y_test)

def get_class_names(filepath):
    """
    filepath: cifar-10-batches-py/batches.meta
    """
    with open(filepath, 'rb') as f:
        meta = pickle.load(f)
    return meta['label_names']
