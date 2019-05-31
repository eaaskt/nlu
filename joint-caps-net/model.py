import tensorflow as tf
import numpy as np


class capsnet():
    """
        Using bidirectional LSTM to learn sentence embedding
        for users' questions
    """

    def __init__(self, FLAGS, initializer=tf.contrib.layers.xavier_initializer()):
        """
            lstm class initialization
        """
        # configurations
        self.hidden_size = FLAGS.hidden_size
        self.vocab_size = FLAGS.vocab_size
        self.word_emb_size = FLAGS.word_emb_size
        # self.batch_size = FLAGS.batch_size
        self.learning_rate = FLAGS.learning_rate
        self.initializer = initializer
        self.intents_nr = FLAGS.intents_nr
        self.slots_nr = FLAGS.slots_nr
        self.margin = FLAGS.margin
        self.keep_prob = FLAGS.keep_prob
        self.slot_routing_num = FLAGS.slot_routing_num
        self.intent_routing_num = FLAGS.intent_routing_num
        self.re_routing_num = FLAGS.re_routing_num
        self.slot_output_dim = FLAGS.slot_output_dim
        self.intent_output_dim = FLAGS.intent_output_dim

        # parameters for self attention
        self.max_sentence_length = FLAGS.max_sentence_length

        # input data
        self.input_x = tf.placeholder("int64", [None, self.max_sentence_length])
        self.batch_size = tf.shape(self.input_x)[0]
        self.sentences_length = tf.placeholder("int64", [None])
        self.encoded_intents = tf.placeholder(tf.float32, shape=[None, self.intents_nr])
        self.encoded_slots = tf.placeholder(tf.float32, shape=[None, self.max_sentence_length, self.slots_nr])

        # graph
        self.instantiate_weights()
        self.H = self.word_caps()
        # capsule
        self.slot_output_vectors, self.slot_weights_c, self.slot_predictions, self.slot_weights_b = self.slot_capsule()
        self.intent_output_vectors, self.intent_weights_c, self.intent_predictions, self.intent_weights_b = self.intent_capsule()
        # = self.rerouting()
        self.loss_val = self.loss()

        self.train_op = self.train()

    def instantiate_weights(self):
        """
            Initializer variable weights
        """
        with tf.name_scope("embedding"):  # embedding matrix
            self.Embedding = tf.get_variable("Embedding",
                                             shape=[self.vocab_size, self.word_emb_size],
                                             initializer=self.initializer, trainable=False)
        with tf.name_scope("slot_capsule_weights"):
            slot_capsule_weights_init = tf.get_variable("slot_capsule_weights_init",
                                                        shape=[1, self.max_sentence_length, self.slots_nr,
                                                               self.slot_output_dim, self.hidden_size * 2],
                                                        initializer=self.initializer)
            self.slot_capsule_weights = tf.tile(slot_capsule_weights_init, [self.batch_size, 1, 1, 1, 1])

        with tf.name_scope("slot_capsule_biases"):
            slot_capsule_biases_init = tf.get_variable("slot_capsule_biases_init",
                                                       shape=[1, self.max_sentence_length, self.slots_nr,
                                                              self.slot_output_dim, 1],
                                                       initializer=self.initializer)
            self.slot_capsule_biases = tf.tile(slot_capsule_biases_init, [self.batch_size, 1, 1, 1, 1])

        with tf.name_scope("intent_capsule_weights"):
            intent_capsule_weights_init = tf.get_variable("intent_capsule_weights_init",
                                                          shape=[1, self.slots_nr, self.intents_nr,
                                                                 self.intent_output_dim, self.slot_output_dim],
                                                          initializer=self.initializer)
            self.intent_capsule_weights = tf.tile(intent_capsule_weights_init, [self.batch_size, 1, 1, 1, 1])

        with tf.name_scope("intent_capsule_biases"):
            intent_capsule_biases_init = tf.get_variable("intent_capsule_biases_init",
                                                         shape=[1, self.slots_nr, self.intents_nr,
                                                                self.intent_output_dim, 1],
                                                         initializer=self.initializer)
            self.intent_capsule_biases = tf.tile(intent_capsule_biases_init, [self.batch_size, 1, 1, 1, 1])

        # with tf.name_scope("rerouting_capsule_weights"):
        #     self.rerouting_capsule_weights = tf.get_variable("rerouting_capsule_weights",
        #                                                      shape=[None, self.max_sentence_length, self.slots_nr,
        #                                                             self.slot_output_dim, self.intent_output_dim],
        #                                                      initializer=self.initializer)

    def word_caps(self):
        # shape:[None, sentence_length, embed_size]
        input_embed = tf.nn.embedding_lookup(self.Embedding, self.input_x, max_norm=1)

        cell_fw = tf.contrib.rnn.LSTMCell(self.hidden_size)
        cell_bw = tf.contrib.rnn.LSTMCell(self.hidden_size)

        H, _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw,
            cell_bw,
            input_embed,
            self.sentences_length,
            dtype=tf.float32)
        H = tf.concat([H[0], H[1]], axis=2)
        return H

    def _squash(self, input_tensor, axis=-1, epsilon=1e-7, name=None):
        with tf.name_scope(name, default_name="squash"):
            squared_norm = tf.reduce_sum(tf.square(input_tensor), axis=axis,
                                         keep_dims=True)
            safe_norm = tf.sqrt(squared_norm + epsilon)
            squash_factor = squared_norm / (1. + squared_norm)
            unit_vector = input_tensor / safe_norm
            return squash_factor * unit_vector

    def _update_routing(self, caps1_n_caps, caps2_n_caps, caps2_predicted,
                        num_iter):
        """Sums over scaled votes and applies squash to compute the activations.
        Iteratively updates routing logits (scales) based on the similarity between
        the activation of this layer and the votes of the layer below.
        Args:
          votes: tensor, The transformed outputs of the layer below.
          biases: tensor, Bias variable.
          logit_shape: tensor, shape of the logit to be initialized.
          num_dims: scalar, number of dimmensions in votes. For fully connected
          capsule it is 4, for convolutional 6.
          input_dim: scalar, number of capsules in the input layer.
          output_dim: scalar, number of capsules in the output layer.
          num_routing: scalar, Number of routing iterations.
          leaky: boolean, if set use leaky routing.
        Returns:
          The activation tensor of the output layer after num_routing iterations.
        """

        def _body(i, raw_weights, output_vectors, routing_weights):
            routing_weights = tf.nn.softmax(raw_weights, dim=2)

            weighted_predictions = tf.multiply(routing_weights, caps2_predicted)
            weighted_sum = tf.reduce_sum(weighted_predictions, axis=1, keep_dims=True)

            caps2_output_round_1 = self._squash(weighted_sum, axis=-2)

            output_vectors = output_vectors.write(i, caps2_output_round_1)
            caps2_output_round_1_tiled = tf.tile(
                caps2_output_round_1, [1, caps1_n_caps, 1, 1, 1])

            agreement = tf.matmul(caps2_predicted, caps2_output_round_1_tiled,
                                  transpose_a=True)

            raw_weights_round_2 = tf.add(raw_weights, agreement)

            return i + 1, raw_weights_round_2, output_vectors, routing_weights

        output_vectors = tf.TensorArray(
            dtype=tf.float32, size=num_iter, clear_after_read=False)
        raw_weights = tf.zeros([self.batch_size, caps1_n_caps, caps2_n_caps, 1, 1], dtype=np.float32)
        routing_weights = tf.nn.softmax(raw_weights, dim=2)
        i = tf.constant(0, dtype=tf.int32)
        _, raw_weights, output_vectors, routing_weights = tf.while_loop(
            lambda i, raw_weights, output_vectors, routing_weights: i < num_iter,
            _body,
            loop_vars=[i, raw_weights, output_vectors, routing_weights],
            swap_memory=True)

        return output_vectors.read(num_iter - 1), raw_weights, routing_weights

    def slot_capsule(self):
        word_caps_output = tf.nn.dropout(self.H, self.keep_prob)

        word_caps_output_expanded = tf.expand_dims(word_caps_output, axis=-1, name="word_caps_output_expanded")
        word_caps_output_tile = tf.expand_dims(word_caps_output_expanded, axis=2, name="word_caps_output_tile")
        word_caps_output_tiled = tf.tile(word_caps_output_tile, [1, 1, self.slots_nr, 1, 1],
                                         name="word_caps_output_tiled")

        slot_caps_predicted_matmul = tf.matmul(self.slot_capsule_weights, word_caps_output_tiled, name="slot_caps_predicted_matmul")
        slot_caps_predicted = tf.tanh(tf.add(slot_caps_predicted_matmul, self.slot_capsule_biases))

        output_vector, weights_b, weights_c = self._update_routing(
            caps1_n_caps=self.max_sentence_length,
            caps2_n_caps=self.slots_nr,
            caps2_predicted=slot_caps_predicted,
            num_iter=self.slot_routing_num,
        )
        return output_vector, weights_c, slot_caps_predicted, weights_b

    def intent_capsule(self):
        slot_caps_output = tf.nn.dropout(self.slot_output_vectors, self.keep_prob)

        slots_output_tshape = [0, 2, 1, 3, 4]
        slot_caps_output_transposed = tf.transpose(slot_caps_output, slots_output_tshape)

        slot_caps_output_tiled = tf.tile(slot_caps_output_transposed, [1, 1, self.intents_nr, 1, 1],
                                         name="slot_caps_output_tiled")

        intent_caps_predicted_matmul = tf.matmul(self.intent_capsule_weights, slot_caps_output_tiled, name="intent_caps_predicted")
        intent_caps_predicted = tf.tanh(tf.add(intent_caps_predicted_matmul, self.intent_capsule_biases))

        output_vector, weights_b, weights_c = self._update_routing(
            caps1_n_caps=self.slots_nr,
            caps2_n_caps=self.intents_nr,
            caps2_predicted=intent_caps_predicted,
            num_iter=self.intent_routing_num,
        )
        return output_vector, weights_c, intent_caps_predicted, weights_b

    def cross_entropy_loss(self):
        reduced_dim_slot_weights_c = tf.squeeze(self.slot_weights_c, axis=[3, 4])
        epsilon = 1e-7
        log_weights_c = tf.log(tf.maximum(reduced_dim_slot_weights_c, epsilon))
        loss_matrix = tf.multiply(self.encoded_slots, log_weights_c)
        loss_vectors = tf.reduce_sum(loss_matrix, axis=2)
        loss = tf.reduce_sum(loss_vectors, axis=1)
        final_loss = tf.reduce_mean(loss)
        negative_loss = tf.negative(final_loss)
        self.cross_entropy_tr_summary = tf.summary.scalar('cross_entropy_loss_training', negative_loss)
        self.cross_entropy_val_summary = tf.summary.scalar('cross_entropy_loss_validation', negative_loss)
        return negative_loss

    def _margin_loss(self, labels, raw_logits, margin=0.4, downweight=0.5):
        """Penalizes deviations from margin for each logit.
        Each wrong logit costs its distance to margin. For negative logits margin is
        0.1 and for positives it is 0.9. First subtract 0.5 from all logits. Now
        margin is 0.4 from each side.
        Args:
            labels: tensor, one hot encoding of ground truth.
            raw_logits: tensor, model predictions in range [0, 1]
            margin: scalar, the margin after subtracting 0.5 from raw_logits.
            downweight: scalar, the factor for negative cost.
        Returns:
            A tensor with cost for each data point of shape [batch_size].
        """
        logits = raw_logits - 0.5
        positive_cost = labels * tf.cast(tf.less(logits, margin),
                                         tf.float32) * tf.pow(logits - margin, 2)
        negative_cost = (1 - labels) * tf.cast(
            tf.greater(logits, -margin), tf.float32) * tf.pow(logits + margin, 2)
        return 0.5 * positive_cost + downweight * 0.5 * negative_cost

    def safe_norm(self, s, axis=-1, epsilon=1e-7, keep_dims=False, name=None):
        with tf.name_scope(name, default_name="safe_norm"):
            squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                         keep_dims=keep_dims)
            return tf.sqrt(squared_norm + epsilon)

    def margin_loss(self):
        intent_vectors = tf.squeeze(self.intent_output_vectors, axis=[1, 4])
        # intent_output_norm = tf.norm(intent_vectors, axis=-1)
        intent_output_norm = self.safe_norm(intent_vectors)
        loss_val = self._margin_loss(self.encoded_intents, intent_output_norm)
        loss_val = tf.reduce_mean(loss_val)
        margin_loss = 1000 * loss_val
        self.margin_loss_tr_summary = tf.summary.scalar('margin_loss_training', margin_loss)
        self.margin_loss_val_summary = tf.summary.scalar('margin_loss_validation', margin_loss)
        return margin_loss

    def loss(self):
        total_loss = tf.reduce_mean(self.margin_loss() + self.cross_entropy_loss()) 
        self.loss_tr_summary = tf.summary.scalar('total_loss_training', total_loss)
        self.loss_val_summary = tf.summary.scalar('total_loss_validation', total_loss)
        return total_loss

    def train(self):
        train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_val)
        return train_op
