import lmdb
import os
import cv2
import cPickle
import caffe
from caffe.proto import caffe_pb2

def unpickle(file):
    """ unpickle the data """
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

if __name__=='__main__':
    meta=unpickle(os.path.join('cifar-100-python', 'meta'))
    fine_label_names=meta['fine_label_names']

    env=lmdb.open('cifar100_train_lmdb')
    txn=env.begin()
    cursor=txn.cursor()
    datum=caffe_pb2.Datum()

    i=0
    for key,value in cursor:
        datum.ParseFromString(value)
        if i<10:
            data=caffe.io.datum_to_array(datum)
            label=datum.label
            img=data.transpose(1,2,0)
            cv2.imwrite('{}.png'.format(fine_label_names[label]),img)
        i+=1

    env.close()
    print('there are totally {} pictures'.format(i))