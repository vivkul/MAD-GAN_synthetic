#!/usr/bin/env python
# Tensorflow impl. of InfoGAN (The 2nd alt.)

from tensorflow.examples.tutorials.mnist import input_data
from common import *
from datasets import data_celeba, data_mnist
from models.celeba_models import *
from models.mnist_models import *
from eval_funcs import *

def sample_c(m, n_categorical=5):
    return np.random.multinomial(1, n_categorical*[1.0/n_categorical], size=m)

def train_infogan(data, g_net, d_net, name='InfoGAN',
                 dim_z=128, n_iters=1e5, lr=1e-4, batch_size=128,
                 sampler=sample_z, eval_funcs=[],
                 n_categorical=5):

    #TODOi test with larger network as well
    ### 0. Common preparation
    hyperparams = {'LR': lr}
    base_dir, out_dir, log_dir = create_dirs(name, g_net.name, d_net.name, hyperparams)

    tf.reset_default_graph()

    global_step = tf.Variable(0, trainable=False)
    increment_step = tf.assign_add(global_step, 1)
    lr = tf.constant(lr)

    ### 1. Define network structure
    x_shape = data.train.images[0].shape
    z0 = tf.placeholder(tf.float32, shape=[None, dim_z])            # Latent var.
    c0 = tf.placeholder(tf.float32, shape=[None, n_categorical])
    x0 = tf.placeholder(tf.float32, shape=(None,) + x_shape)        # Generated images
    
    g_inp = tf.concat([z0, c0],1)
    G = g_net(g_inp, 'InfoGAN_G')

    feat1 = d_net.former1(x0, 'InfoGAN_DQ_base')
    feat2 = d_net.former2(feat1, 'InfoGAN_DQ_base')
    D_real = d_net.latter(feat2, 'InfoGAN_D_head')
    feat3 = d_net.former1(G, 'InfoGAN_DQ_base', reuse=True)
    feat4 = d_net.former2(feat3, 'InfoGAN_DQ_base', reuse=True)
    D_fake = d_net.latter(feat4, 'InfoGAN_D_head', reuse=True)

    feat5 = d_net.former1(G, 'InfoGAN_DQ_base', reuse=True)
    feat6 = d_net.former2(feat5, 'InfoGAN_DQ_base', reuse=True)
    Q = d_net.latter(feat6, 'InfoGAN_Q_head', keep_same_output=False, new_output_dim=n_categorical, keep_same_act=False, new_act=tf.nn.softmax)

    D_loss = tf.reduce_mean(-tf.log(D_real)-tf.log(1-D_fake))
    G_loss = tf.reduce_mean(-tf.log(D_fake))
    Q_loss = tf.reduce_mean(-tf.log(Q + 1e-8) * c0)
    
    D_solver = (tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5))  \
            .minimize(D_loss, var_list=get_trainable_params('InfoGAN_DQ_base')
                    +get_trainable_params('InfoGAN_D_head'))
    G_solver = (tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5))  \
            .minimize(G_loss, var_list=get_trainable_params('InfoGAN_G'))
    Q_solver = (tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5))  \
            .minimize(Q_loss, var_list=get_trainable_params('InfoGAN_G')
                    +get_trainable_params('InfoGAN_DQ_base')
                    +get_trainable_params('InfoGAN_Q_head'))

    #### 2. Operations for log/state back-up
    tf.summary.scalar('InfoGAN_D_loss', D_loss)
    tf.summary.scalar('InfoGAN_G_loss', G_loss)
    tf.summary.scalar('InfoGAN_Q_loss', Q_loss)

    if check_dataset_type(x_shape) != 'synthetic':
        tf.summary.image('InfoGAN', G, max_outputs=4)        # for images only

    summaries = tf.summary.merge_all()

    saver = tf.train.Saver(max_to_keep=4)

    # Initial setup for visualization
    outputs = [G]
    figs = [None] * len(outputs)
    fig_names = ['fig_gen_{:04d}_InfoGAN.png']
    data_samples = ['data_samples_{:04d}_InfoGAN.out']
    gen_samples = ['gen_samples_{:04d}_InfoGAN.out']

    #plt.ion() #Interactive mode on

    ### 3. Run a session
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter(log_dir, sess.graph)

    print('{:>10}, {:>7}, {:>7}, {:>7}, {:>7}') \
        .format('Iters', 'cur_LR', 'InfoGAN_D', 'InfoGAN_G', 'InfoGAN_Q')


    for it in range(int(n_iters)):
        batch_xs, batch_ys = data.train.next_batch(batch_size)
        
        _, loss_D = sess.run(           #TODOi try outputing G and using it in next two operations instead of forward pass through G again and again
            [D_solver, D_loss],
            feed_dict={x0: batch_xs, z0: sampler(batch_size, dim_z), c0: sample_c(batch_size, n_categorical)}
        )

        _, loss_G = sess.run(
            [G_solver, G_loss],
            feed_dict={z0: sampler(batch_size, dim_z), c0: sample_c(batch_size, n_categorical)}
        )

        _, loss_Q = sess.run(
            [Q_solver, Q_loss],
            feed_dict={z0: sampler(batch_size, dim_z), c0: sample_c(batch_size, n_categorical)}
        )


        _, cur_lr = sess.run([increment_step, lr])

        if it % PRNT_INTERVAL == 0:
            print('{:10d}, {:1.4f}, {: 1.4f}, {: 1.4f}, {: 1.4f}') \
                    .format(it, cur_lr, loss_D, loss_G, loss_Q)

            # Tensorboard
            cur_summary = sess.run(summaries, feed_dict={x0: batch_xs, z0: sampler(batch_size, dim_z), c0: sample_c(batch_size, n_categorical)})
            writer.add_summary(cur_summary, it)

        if it % EVAL_INTERVAL == 0:
            # FIXME
            img_generator = lambda n: sess.run(output, feed_dict={z0: sampler(n, dim_z), c0: sample_c(n, n_categorical)})

            for i, output in enumerate(outputs):
                figs[i] = data.plot(img_generator, gen_S = out_dir + gen_samples[i].format(it / 1000), data_S = out_dir + data_samples[i].format(it / 1000), fig_id=i, batch_size = batch_size)
                figs[i].canvas.draw()
	        
                plt.savefig(out_dir + fig_names[i].format(it / 1000), bbox_inches='tight')
                if PLT_CLOSE == 1:
                    plt.close()

            # Run evaluation functions
            for func in eval_funcs:
                func(it, img_generator)

        if it % SAVE_INTERVAL == 0:
            saver.save(sess, out_dir + 'InfoGAN', global_step=it, write_meta_graph=False)

    sess.close()
