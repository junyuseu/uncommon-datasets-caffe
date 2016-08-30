import scipy.io as sio
import matplotlib.pyplot as plt

print 'Loading Matlab data.'
mat=sio.loadmat('train_32x32.mat')
data=mat['X']
label=mat['y']
for i in range(100):
    plt.subplot(10,10,i+1)
    plt.title(label[i][0])
    plt.imshow(data[...,i])
    plt.axis('off')
plt.show()
