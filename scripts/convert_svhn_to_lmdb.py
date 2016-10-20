import numpy as np
import caffe
import lmdb
import scipy.io as sio
import random
from caffe.proto import caffe_pb2

def main():
	""" convert the svhn data to lmdb """
    train=sio.loadmat('train_32x32.mat')
    test=sio.loadmat('test_32x32.mat')

    train_data=train['X']
    train_label=train['y']
    test_data=test['X']
    test_label=test['y']

    train_data = np.swapaxes(train_data, 0, 3)
    train_data = np.swapaxes(train_data, 1, 2)
    train_data = np.swapaxes(train_data, 2, 3)

    test_data = np.swapaxes(test_data, 0, 3)
    test_data = np.swapaxes(test_data, 1, 2)
    test_data = np.swapaxes(test_data, 2, 3)

    N=train_label.shape[0]
    map_size=train_data.nbytes*10
    env=lmdb.open('svhn_train_lmdb',map_size=map_size)
    txn=env.begin(write=True)

    #shuffle the training data
    r=list(range(N))
    random.shuffle(r)

    count=0
    for i in r:
        datum=caffe_pb2.Datum()
        label=int(train_label[i][0])
        if label==10:
            label=0
        datum=caffe.io.array_to_datum(train_data[i],label)
        str_id='{:08}'.format(count)
        txn.put(str_id,datum.SerializeToString())

        count += 1
        if count % 1000 == 0:
            print('already handled with {} pictures'.format(count))
            txn.commit()
            txn = env.begin(write=True)

    txn.commit()
    env.close()

    map_size = test_data.nbytes * 10
    env = lmdb.open('svhn_test_lmdb', map_size=map_size)
    txn = env.begin(write=True)
    count = 0
    for i in range(test_label.shape[0]):
        datum = caffe_pb2.Datum()
        label = int(test_label[i][0])
        if label == 10:
            label = 0
        datum = caffe.io.array_to_datum(test_data[i], label)
        str_id = '{:08}'.format(count)
        txn.put(str_id, datum.SerializeToString())

        count += 1
        if count % 1000 == 0:
            print('already handled with {} pictures'.format(count))
            txn.commit()
            txn = env.begin(write=True)

    txn.commit()
    env.close()

if __name__=='__main__':
    main()
