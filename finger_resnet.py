import numpy as np
import tensorflow as tf

from data_loader import FingerData


class ResNet:
    def __init__(self, batchsize, epoch_num, dropout=.5, lr=.001, debug=False):
        """
        Defines the hyperparameters, creates the datasets, and Tensorflow placeholders
        """
        # Initialize the hyperparameters
        self.batch_size = batchsize
        self.learning_rate = lr
        self.epochs = epoch_num
        # self.is_training = False
        self.debug = debug

        # Batch normalization parameters
        self.norm_beta = 0.0
        self.norm_gamma = 1.0
        self.norm_epsilon = 0.001
        self.means = []
        self.varis = []

        # Create the data class
        if self.debug: print("Loading data...")
        self.data = FingerData()

        # Create data sets and a session that can be accessed throughout the class
        self.session = tf.Session()
        self.training = self.data.get_training_batch(self.batch_size)
        self.validation = self.data.get_validation_batch(-1)
        self.test = self.data.get_test_batch(-1)

        # Defines placeholders and overall network variables
        self.images = tf.placeholder(tf.float32, [None, 300, 300, 3])
        self.labels = tf.placeholder(tf.float32, [None])
        # self.is_training = tf.placeholder_with_default(1, [])
        # self.do_augment = tf.placeholder_with_default(1, [])
        # self.dropout_rate = tf.placeholder_with_default(dropout, [])
        self.is_training = tf.placeholder(shape=[], dtype=tf.int16)
        self.do_augment = tf.placeholder(shape=[], dtype=tf.int16)
        self.dropout_rate = tf.placeholder(shape=[], dtype=tf.int16)

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
        bright = tf.image.random_brightness(inpt, .25)
        contrast = tf.image.random_contrast(bright, 0, .5)
        flip = tf.image.random_flip_left_right(contrast)
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
        flat = tf.reshape(inpt, [-1, shape[1] * shape[2] * shape[3]])
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
        if self.is_training == 1:
            mean, var = tf.nn.moments(inpt, [1, 2, 3], keep_dims=True)
            self.means.append(mean)
            self.varis.append(var)
            return tf.cast(tf.nn.batch_normalization(inpt, mean, var, self.norm_beta, self.norm_gamma, self.norm_epsilon), tf.float32)
        else:
            print('shouldnt be here', self.is_training)
            print(inpt.shape)
            mean = np.mean(self.means)
            var = np.mean(self.varis)
            return tf.cast(tf.nn.batch_normalization(inpt, mean, var, self.norm_beta, self.norm_gamma, self.norm_epsilon), tf.float32)

    def convolution_layer(self, inpt, filters, in_depth=3):
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
        conv_bias = tf.get_variable("conv_biases", in_depth, initializer=tf.constant_initializer(0.0))
        return tf.nn.relu((tf.nn.conv2d(inpt, conv_weight, strides=[1, 1, 1, 1], padding="SAME") + conv_bias))

    def convolution_layer_1x1(self, inpt, in_depth):
        """
        1x1 Convolutional layer.  Changes filter map depth to 1.
        Args:
            inpt: Images to be convolved.
            in_depth: The number of incoming filters/channels

        Returns:
            Convolved images with 1 filter map.
        """
        conv_weight = tf.get_variable("1x1_conv_weights", [1, 1, in_depth, 1],
                                      initializer=tf.random_normal_initializer(stddev=.001))
        conv_bias = tf.get_variable("1x1_conv_biases", in_depth, initializer=tf.constant_initializer(0.0))
        return tf.nn.relu((tf.nn.conv2d(inpt, conv_weight, strides=[1, 1, 1, 1], padding="SAME") + conv_bias))

    def convolution_block(self, inpt):
        """
        The ResNet part: creates two convolutional blocks and adds the input of the block to the output.
        If the channels of the input is larger than 1, it is "flattened" by doing a 1x1 convolution before adding it.
        Args:
            inpt: Images to be convolved.

        Returns:
            Convolved images.
        """
        residual = self.batch_normalize(inpt)
        if inpt.shape[2] != 1:  # only do a 1x1 if there are more than 1 channel
            residual = self.convolution_layer_1x1(inpt, inpt.shape[2])
        first_layer = self.convolution_layer(residual, 16)
        first_layer = tf.nn.relu(first_layer)
        first_layer = self.batch_normalize(first_layer)
        second_layer = self.convolution_layer(first_layer, 32, 16)
        res_connect = tf.add(second_layer, residual)
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
        fc_weight = tf.get_variable("fc_weights", [inpt.shape[1], neurons],
                                    initializer=tf.random_normal_initializer(stddev=0.001))
        fc_bias = tf.get_variable("fc_biases", [neurons], initializer=tf.constant_initializer(0.0))
        activation = tf.nn.relu((tf.matmul(inpt, fc_weight) + fc_bias))
        if self.is_training: # only do dropout when training
            dropped = tf.nn.dropout(activation, self.dropout_rate)
        else:
            dropped = activation
        return dropped

    def output_layer(self, inpt):
        """
        The output layer which returns the 5 classes as a vector for input into the softmax cross entropy function.
        Args:
            inpt: Input to be fed forward.

        Returns:
            Output of network in logits form.
        """
        fc_weight = tf.get_variable("fc_weights", [inpt.shape[1], 10],
                                    initializer=tf.random_normal_initializer(stddev=0.001))
        fc_bias = tf.get_variable("fc_biases", [5], initializer=tf.constant_initializer(0.0))
        activation = tf.matmul(inpt, fc_weight) + fc_bias
        return activation

    def inference(self):
        """
        Feeds forward the images through the network.
        Returns:
            Output of network in logits form.
        """
        # Alter brightness and contrast randomly
        if self.do_augment == 1:
            images = self.augment(self.images)
        else:
            images = self.images # for compatibility with the above code

        with tf.variable_scope("conv_block1"):
            conv_block1 = self.convolution_block(images)
        with tf.variable_scope("conv_block2"):
            conv_block2 = self.convolution_block(conv_block1)
        with tf.variable_scope("fc_1"):
            flattened = self.flatten(conv_block2)
            fc_1 = self.fully_connected(flattened, 100)
        with tf.variable_scope("output"):
            output = self.output_layer(fc_1)
        return output

    def loss_function(self, logits):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.one_hot_labels, logits=logits))

    def optimize(self):
        """
        The backpropagation step with Adam optimizer.  Also computes metrics and creates summary statistics.
        """
        self.one_hot_labels = self.one_hot(self.labels)
        output = self.inference()
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
        self.inference()
        self.optimize()

        run = 'lr001' # name of the run, e.g. learning rate .001
        train_writer = tf.summary.FileWriter("./summaries/"+run+"train", tf.get_default_graph())
        validation_writer = tf.summary.FileWriter("./summaries/"+run+"validation", tf.get_default_graph())

        saver = tf.train.Saver()
        if continue_training:
            saver.restore(self.session, tf.train.latest_checkpoint('./checkpoints/train'))
        else:
            self.session.run(tf.global_variables_initializer())
        step = 0
        prev_acc = 0

        for e in range(self.epochs):
            for x, y in self.training:
                summary, _acc, _loss, _ = self.session.run([self.merged_summaries, self.accuracy, self.loss,
                                                            self.minimize_loss], feed_dict={self.images: x,
                                                                                            self.labels: y,
                                                                                            self.is_training: 1,
                                                                                            self.dropout_rate: .5,
                                                                                            self.do_augment: 1})
                train_writer.add_summary(summary, step)

                # Validate every 50 steps
                if step % 50 == 0:
                    for v_x, v_y in self.validation:
                        # is_training is left on to allow us to use all available data to better find the real moments
                        summary, v_acc, _loss = self.session.run([self.merged_summaries, self.accuracy, self.loss],
                                                                 feed_dict={self.images: v_x, self.labels: v_y,
                                                                            self.is_training: 1, self.do_augment: 0,
                                                                            self.dropout_rate: 1})
                        validation_writer.add_summary(summary, step)
                        print("Current STEP:", step)
                        print("Validation accuracy:", v_acc)
                        print("Validation loss:", _loss)


                        if v_acc > prev_acc and step:
                            saver.save(self.session, "./checkpoints/fingers-{}.ckpt".format(v_acc, max_to_keep=10))
                        prev_acc = v_acc

                    # Refill the generator because it was exhausted
                    self.validation = self.data.get_validation_batch(-1)

            # Refill the generator because it was exhausted
            self.training = self.data.get_training_batch(self.batch_size)

        saver.save(self.session, "./checkpoints/fingers-{}-end.ckpt".format(v_acc), max_to_keep=20)

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
    model = ResNet(2,2,debug=True)
    model.train()