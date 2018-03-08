import numpy as np
import tensorflow as tf
import time

from data_loader import FingerDataBatch


class ResNet:
    def __init__(self, blocks, batchsize, epoch_num, dropout=.5, lr=.001, augment=True, debug=False):
        """
        Defines the hyperparameters, creates the datasets, and Tensorflow placeholders
        """
        self.start = time.time()

        tf.reset_default_graph()
        # Initialize the hyperparameters
        self.batch_size = batchsize
        self.learning_rate = lr
        self.epochs = epoch_num
        self.debug = debug
        self.do_augment = augment
        self.blocks = blocks

        # Batch normalization parameters
        self.norm_beta = 0.0
        self.norm_gamma = 1.0
        self.norm_epsilon = 0.001
        self.means = tf.Variable(tf.zeros([1,1,1,4]), dtype=tf.float32, trainable=False, name='means')
        self.varis = tf.Variable(tf.zeros([1,1,1,4]), dtype=tf.float32, trainable=False, name='variances')

        # Create the data class
        if self.debug: print("Loading data...")
        self.data = FingerDataBatch(self.batch_size)

        # Create data sets and a session that can be accessed throughout the class
        self.session = tf.Session()
        # self.training = self.data.get_training_batch(self.batch_size)
        # self.validation = self.data.get_validation_batch(-1)
        # self.test = self.data.get_test_batch(-1)

        # Defines placeholders and overall network variables
        self.images = tf.placeholder(tf.float32, [None, 300, 300, 3], name='images')
        self.labels = tf.placeholder(tf.int32, [None], name='labels')
        self.is_training = tf.placeholder_with_default(1, [], name='is_training')
        self.dropout_rate = tf.placeholder(tf.float32, shape=[], name="dropout_rate")

    def one_hot(self, lbls):
        """
        Transforms labels into one hot vectors via TF node.
        Args:
            lbls: Labels to be converted.

        Returns:
            one_hots: One hot encoded labels.
        """
        one_hots = tf.one_hot(lbls, 5)
        return one_hots

    def augment(self, inpt):
        """
        Augments the data through random brightness, contrast, and flipping changes.
        Args:
            inpt: Images to be augmented.

        Returns:
            flip: Augmented images to be passed on.
        """
        bright = tf.map_fn(lambda img: tf.image.random_brightness(img, .25), inpt, name='brightness')
        # contrast = tf.map_fn(lambda img: tf.image.random_contrast(img, 0, .5), bright, name='contrast')
        flip = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), bright, name='flip') # CHANGED INPUT
        return flip

    def flatten(self, inpt):
        """
        Flattens a tensor into 1 dimension.  Used after the convolutional layers are done.
        Args:
            inpt: Tensor to be flattened.

        Returns:
            flat: Flattened tensor.
        """
        shape = inpt.get_shape().as_list()
        flat = tf.reshape(inpt, [-1, shape[1] * shape[2] * shape[3]], name='flatten')
        return flat

    def batch_normalize(self, inpt):
        """
        Normalizes the batches.  If in training, uses batch norm, if in test, uses global mean and variance found
        over the course of training.
        Args:
            inpt: Images to be batch normalized.

        Returns:
            Batch normalized images.
        """
        def training(inpt):
            mean, var = tf.nn.moments(inpt, [1, 2, 3], keep_dims=True)
            tf.assign(self.means, self.means + mean, name='update_mean')
            tf.assign(self.varis, self.varis + var, name='update_varis')
            return tf.nn.batch_normalization(inpt, mean, var, self.norm_beta, self.norm_gamma, self.norm_epsilon, name='batchnorm')
        def not_training(inpt):
            mean = tf.reduce_mean(self.means, name='global_mean')
            var = tf.reduce_mean(self.varis, name='global_var')
            return tf.nn.batch_normalization(inpt, mean, var, self.norm_beta, self.norm_gamma, self.norm_epsilon, name='batchnorm_global')
        return tf.cond(tf.equal(self.is_training, 1), lambda: training(inpt), lambda: not_training(inpt), name='batchnorm_cond')

    def max_pool(self, inpt):
        return tf.nn.max_pool(inpt, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME", name='maxpool')

    def convolution_layer(self, inpt, filters, in_depth=1):
        """
        3x3 Convolutional layer.
        Args:
            inpt: Images to be convolved.
            filters: Number of filter kernels to create
            in_depth: The number of incoming filter maps/channels

        Returns:
            Convolved images.
        """
        conv_weight = tf.get_variable("conv_weights", [3, 3, in_depth, filters],
                                      initializer=tf.random_normal_initializer(stddev=.001))
        conv_bias = tf.get_variable("conv_biases", filters, initializer=tf.constant_initializer(0.0))
        # Not relu'd because the last layer is relu'd after concatenation with the residual
        return tf.nn.conv2d(inpt, conv_weight, strides=[1, 1, 1, 1], padding="SAME") + conv_bias

    def convolution_layer_1x1(self, inpt):
        """
        1x1 Convolutional layer.  Changes filter map depth to 1.
        Args:
            inpt: Images to be convolved.

        Returns:
            Convolved images with 1 filter map.
        """
        conv_weight = tf.get_variable("1x1_conv_weights", [1, 1, inpt.shape[3], 1],
                                      initializer=tf.random_normal_initializer(stddev=.001))
        conv_bias = tf.get_variable("1x1_conv_biases", 1, initializer=tf.constant_initializer(0.0))
        return tf.nn.relu((tf.nn.conv2d(inpt, conv_weight, strides=[1, 1, 1, 1], padding="SAME") + conv_bias))

    def convolution_block(self, inpt, name):
        """
        The ResNet part: creates two convolutional blocks and adds the input of the block to the output.
        If the channels of the input is larger than 1, it is "flattened" by doing a 1x1 convolution before adding it.
        Args:
            inpt: Images to be convolved.

        Returns:
            Convolved images.
        """
        # Incoming input will always have multiple channels
        with tf.variable_scope("residual_conv"+name):
            residual = self.convolution_layer_1x1(inpt)
            residual = self.batch_normalize(residual)
        with tf.variable_scope("conv{}-1".format(name)):
            first_layer = self.convolution_layer(inpt, 16, inpt.shape[3])
            first_layer = tf.nn.relu(first_layer)
            first_layer = self.batch_normalize(first_layer)
        with tf.variable_scope("conv{}-2".format(name)):
            second_layer = self.convolution_layer(first_layer, 8, 16)  # 32->8 due to OOM
            second_layer = self.batch_normalize(second_layer)
        res_connect = tf.add(second_layer, residual)
        if self.debug: print("CONV BLOCK CREATED")
        return tf.nn.relu(res_connect)

    def fully_connected(self, inpt, neurons):
        """
        A fully connected layer with dropout.
        Args:
            inpt: Input to be fed forward.
            neurons: How many neurons to create in the layer

        Returns:
            Output of layer.
        """
        def keep(a):  # Helper function to return activation for the tf.cond
            return a
        fc_weight = tf.get_variable("fc_weights", [inpt.shape[1], neurons],
                                    initializer=tf.random_normal_initializer(stddev=0.001))
        fc_bias = tf.get_variable("fc_biases", [neurons], initializer=tf.constant_initializer(0.0))
        activation = tf.nn.relu((tf.matmul(inpt, fc_weight) + fc_bias))
        # Only dropout when training
        dropped = tf.cond(tf.equal(self.is_training, 1), 
            lambda: tf.nn.dropout(activation, self.dropout_rate), 
            lambda: keep(activation), name='dropout_cond')
        return dropped

    def output_layer(self, inpt):
        """
        The output layer which returns the 5 classes as a vector for input into the softmax cross entropy function.
        Args:
            inpt: Input to be fed forward.

        Returns:
            Output of network in logits form.
        """
        out_weight = tf.get_variable("out_weights", [inpt.shape[1], 5],
                                    initializer=tf.random_normal_initializer(stddev=0.001))
        out_bias = tf.get_variable("out_biases", [5], initializer=tf.constant_initializer(0.0))
        activation = tf.matmul(inpt, out_weight) + out_bias
        return activation

    def inference(self):
        """
        Feeds forward the images through the network.
        Returns:
            Output of network in logits form.
        """
        # Alter brightness and contrast randomly
        if self.do_augment:
            images = self.augment(self.images)
        else:
            images = self.images # for compatibility with the above code

        # with tf.variable_scope("conv_block1"):
        #     conv_block1 = self.convolution_block(images)
        # with tf.variable_scope("conv_block2"):
        #     conv_block2 = self.convolution_block(conv_block1)
        # with tf.variable_scope("fc"):
        #     flattened = self.flatten(conv_block2)
        #     fc = self.fully_connected(flattened, 100)

        with tf.variable_scope("conv"):
            first_conv = self.convolution_layer(images, 32, 3)
            throughput = self.max_pool(first_conv)
        for i, block in enumerate(range(self.blocks)):
            with tf.variable_scope("conv_block{}".format(i + 1)):
                throughput = self.convolution_block(throughput, str(i))
        with tf.variable_scope("fc"):
            flattened = self.flatten(throughput)
            fc = self.fully_connected(flattened, 100)
        with tf.variable_scope("output"):
            output = self.output_layer(fc)
        return output

    def loss_function(self, logits):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.one_hot_labels, logits=logits))

    def backprop(self, output):
        """
        The backpropagation step with Adam optimizer.  Also computes metrics and creates summary statistics.
        """
        self.one_hot_labels = self.one_hot(self.labels)
        self.loss = self.loss_function(output)
        tf.summary.scalar("Loss", self.loss)
        with tf.variable_scope("optimizer"):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.minimize_loss = optimizer.minimize(self.loss)
        with tf.variable_scope("accuracy"):
            # Accuracy calculation: see if the position of the maximum value of the label is the same as the maximum value of the output
            correct_bools = tf.equal(tf.argmax(output, 1), tf.argmax(self.one_hot_labels, 1))
            # If so, it is a one, if not, it is a zero.  Take the average of those 1s and 0s to get the accuracy
            self.accuracy = tf.reduce_mean(tf.cast(correct_bools, tf.float32))
        tf.summary.scalar("Accuracy", self.accuracy)
        self.merged_summaries = tf.summary.merge_all()

    def train(self, continue_training=False):
        """
        Train the network with the training data.
        Can be used to pick up training at a later date with continue_training.
        Args:
            continue_training: Set to true to load previous weights and then train
        """
        print("Generating dataflow graph...")
        output = self.inference()
        self.backprop(output)
        print("Dataflow graph complete.")

        # TODO: More description runs and save files
        run = 'lr'+str(self.learning_rate)[2:] # name of the run, e.g. learning rate .001 = lr001
        train_writer = tf.summary.FileWriter("./summaries/"+run+"train", tf.get_default_graph())
        validation_writer = tf.summary.FileWriter("./summaries/"+run+"validation", tf.get_default_graph())
        print("Data writers created.")

        saver = tf.train.Saver(max_to_keep=5)
        if continue_training:
            print("Continuing training...")
            saver.restore(self.session, tf.train.latest_checkpoint('./checkpoints/train'))
            print("Weights restored...")
        else:
            self.session.run(tf.global_variables_initializer())
        step = 0
        prev_acc = 0
        t_size, v_size, tt_size = self.data.get_sizes()
        print("Ready for training...")
        
        for e in range(self.epochs):
            print("Epoch:", e+1)
            for b in range(t_size // self.batch_size):
                x, y = self.data.get_training_batch()
                summary, _acc, _loss, _ = self.session.run([self.merged_summaries, self.accuracy, self.loss,
                                                            self.minimize_loss], feed_dict={self.images: x,
                                                                                            self.labels: y,
                                                                                            self.is_training: 1,
                                                                                            self.dropout_rate: .5})
                train_writer.add_summary(summary, step)
                if self.debug: print(_acc)

                # Validate every 50 steps
                # if step % 50 == 0:
                #     for bv in range(v_size // self.batch_size):
                #         v_x, v_y = self.data.get_validation_batch()
                #         # is_training is left on to allow us to use all available data to better find the real moments
                #         summary, v_acc, _loss = self.session.run([self.merged_summaries, self.accuracy, self.loss],
                #                                                  feed_dict={self.images: v_x, self.labels: v_y,
                #                                                             self.is_training: 1, self.dropout_rate: 1})
                #         validation_writer.add_summary(summary, step)
                #         print("Current STEP:", step)
                #         print("Validation accuracy:", v_acc)
                #         print("Validation loss:", _loss)


                #         if v_acc > prev_acc:  #and step WHAT IS AND STEP?
                #             saver.save(self.session, "./checkpoints/fingers-{:.2f}-step".format(v_acc), global_step=step)
                #         prev_acc = v_acc

                step += 1

        saver.save(self.session, "./checkpoints/fingers-{:.2f}-end-step".format(v_acc), global_step=step)
        self.data.delete_temp_files()
        end = time.time()
        print("Finished.")
        print('\nTotal time elapsed: {:.2f} minutes'.format(end - self.start))
        
    def test(self):
        """
        Test on the test data.  Only use when training is complete.
        Returns:
            test_acc: Test accuracy.
        """
        raise NotImplementedError
        # saver = tf.train.Saver()
        # saver.restore(self.session, tf.train.latest_checkpoint('./checkpoints/train'))
        # saver = tf.train.import_meta_graph(checkpoints_file_name + '.meta')
        # saver.restore(sess, checkpoints_file_name)


if __name__ == "__main__":
    model = ResNet(2,50,5,debug=True)
    model.train()