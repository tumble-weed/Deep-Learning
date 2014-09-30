import matplotlib
matplotlib.use('Agg')
import numpy
np=numpy
import theano
from theano import tensor as T

import traceback,pdb,os
import skimage
from skimage import io
from collections import OrderedDict
from theano.tensor.shared_randomstreams import RandomStreams

from custom_io import *
from custom_io.utils import tile_raster_images as show_row_vectors
from custom_io.imsave2 import imsave2,savefig2
from custom_io.utils import plot_together
from custom_io.load_data import load_data
#pdb.set_trace()
save_dir='sparse_autoencoder/mnist'
print save_dir
dataset_gz='../../Datasets/mnist.pkl.gz'
img_shape=(28,28)
n_vis=np.prod(img_shape)

datasets=load_data(dataset_gz)
train,test,valid=datasets
train_x,train_y=train
valid_x,valid_y=valid

n_hid=1000
rho=0.05
minibatch_size=100
n_epochs=500
non_lin=T.nnet.sigmoid
#non_lin=lambda inp:1/(1+inp**4)

n_batches=len(train_x.get_value()+minibatch_size-1)/minibatch_size



lim=np.sqrt(6./(n_hid+n_vis+1))
W=theano.shared(value=2*lim*np.random.random((n_hid,n_vis))-lim)
#W=theano.shared(value=np.random.random((n_hid,n_vis))-0.5)
b_v=theano.shared(value=np.zeros((n_vis,1)),broadcastable=[False,True])
b_h=theano.shared(value=np.zeros((n_hid,1)),broadcastable=[False,True])


g_W=theano.shared(value=np.zeros((n_hid,n_vis)))
g_b_v=theano.shared(value=np.zeros((n_vis,1)),broadcastable=[False,True])
g_b_h=theano.shared(value=np.zeros((n_hid,1)),broadcastable=[False,True])



params=[W,b_v,b_h]
gradients=[g_W,g_b_v,g_b_h]

tile_shape=(int(np.sqrt(n_hid)),int(np.sqrt(n_hid))+1)

tile_spacing=(1,1)


x_row=T.matrix('x_row')
x_col=T.matrix('x_col')

x_col=x_row.T

a1=T.dot(W,x_col)+b_h
h=non_lin(a1)
mean_activity=T.mean(h,axis=1)

a2=T.dot(W.T,h)+b_v
x_col_r=non_lin(a2)

cost_rec=T.sum((x_col-x_col_r)**2,axis=0)
#cost_rec=T.sum(np.abs(x_col-x_col_r),axis=0)

cross_entropy = lambda a,b:T.sum((a*T.log(a/b)+(1-a)*T.log((1-a)/(1-b))),axis=0)
cost_sparsity=cross_entropy(rho,mean_activity)


cost=T.mean(cost_rec)+T.mean(cost_sparsity)#+T.sum(W**2,axis=[0,1])

g_params=T.grad(cost,params)

updates=OrderedDict({})
for p,g_p,g in zip(params,g_params,gradients):
	updates[p]=p-0.1*g_p
	updates[g]=g_p
# define functions
index=T.scalar('index',dtype='int64')
train_fn=theano.function([index],cost,updates=updates,givens={x_row:train_x[index*minibatch_size:(index+1)*minibatch_size]})
validation_fn=theano.function([],cost,givens={x_row:valid_x})
nsamples=100
reconstruct_validation=theano.function([],x_col_r,givens={x_row:valid_x[:nsamples]})




n_batches=20
batch_tile_shape=(int(np.sqrt(minibatch_size)),int(np.sqrt(minibatch_size))+1)
validation_freq=5
train_cost_so_far=[]
validation_cost_so_far=[]

for epoch in range(n_epochs):
        this_train_cost=0
	for batch in range(n_batches):
		print batch
                this_train_cost+=train_fn(batch)
        this_train_cost/=np.float64(n_batches)
     	
	if epoch==0:
		out_im1=show_row_vectors(valid_x.get_value()[:nsamples],tile_shape=batch_tile_shape,tile_spacing=tile_spacing,img_shape=img_shape)
		imsave2(os.path.join(save_dir,'originals.png'),out_im1)
        if (epoch+1)%validation_freq==0:
                print 'validating'
                this_validation_cost=validation_fn()
                train_cost_so_far+=[this_train_cost]
                validation_cost_so_far+=[this_validation_cost]

                F=plot_together(([train_cost_so_far,range(len(train_cost_so_far))],[validation_cost_so_far,range(len(validation_cost_so_far))]),legends=['train','validation'])
                savefig2(os.path.join(save_dir,'errors/%d.png'%epoch))
                savefig2(os.path.join(save_dir,'errors.png'))

                out_im=show_row_vectors(W.get_value(),tile_shape=tile_shape,tile_spacing=tile_spacing,img_shape=img_shape)
                imsave2(os.path.join(save_dir,'filters/%d.png'%(epoch)),out_im)
                reconstructed=reconstruct_validation()
                out_im2=show_row_vectors(reconstructed.T,tile_shape=batch_tile_shape,tile_spacing=tile_spacing,img_shape=img_shape)
                imsave2(os.path.join(save_dir,'reconstructed/%d.png'%(epoch)),out_im2)

                # out_im1=show_row_vectors(check_g_W[:16,...],tile_shape=tile_shape,tile_spacing=tile_spacing,img_shape=img_shape)
                # imsave2(os.path.join(save_dir,'gradient_%d.png'%(epoch)),out_im1)
print 'saving after training Autoencoder'
write_model(save_dir+'/auto_enc.model',params)
