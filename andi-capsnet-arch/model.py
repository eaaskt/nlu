import tensorflow as tf
import numpy as np
import util


class CapsNet:
    """
        CapsNetS2I model: capsule neural network model that performs joint intent detection and slot filling
        4 types of capsules: WordCaps, SemanticCaps, SlotCaps, IntentCaps
    """

    def __init__(self, FLAGS, initializer=tf.contrib.layers.xavier_initializer()):
        """
            initialize CapsNet
        """

        # hyperparams
        self.hidden_size = FLAGS.hidden_size
        self.vocab_size = FLAGS.vocab_size
        self.word_emb_size = FLAGS.word_emb_size
        self.learning_rate = FLAGS.learning_rate
        self.initializer = initializer
        self.intents_nr = FLAGS.intents_nr
        self.slots_nr = FLAGS.slots_nr
        self.margin = FLAGS.margin
        self.slot_routing_num = FLAGS.slot_routing_num
        self.intent_routing_num = FLAGS.intent_routing_num
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.slot_output_dim = FLAGS.hidden_size * 2  # same as SemanticCaps output dim
        self.intent_output_dim = FLAGS.intent_output_dim

        self.d_a = FLAGS.d_a
        self.r = FLAGS.r
        self.alpha = FLAGS.alpha
        self.max_sentence_length = FLAGS.max_sentence_length

        # input data
        self.input_x = tf.placeholder('int64', [None, self.max_sentence_length])
        self.batch_size = tf.shape(self.input_x)[0]
        self.sentences_length = tf.placeholder('int64', [None])
        self.encoded_intents = tf.placeholder(tf.float32, shape=[None, self.intents_nr])
        self.encoded_slots = tf.placeholder(tf.float32, shape=[None, self.max_sentence_length, self.slots_nr])

        # CapsNetS2I model
        self.instantiate_weights()
        self.H = self.word_caps()
        self.attention, self.M = self.semantic_caps()
        self.slot_output_vectors, self.slot_weights_c, self.slot_predictions, self.slot_weights_b = self.slot_caps()
        self.intent_output_vectors, self.intent_weights_c, self.intent_predictions, self.intent_weights_b = self.intent_caps()
        self.loss_val = self.loss()

        self.train_op = self.train()

    def instantiate_weights(self):
        """
            Initialize the trainable weights
        """
        with tf.name_scope('embedding'):  # embedding matrix
            self.Embedding = tf.get_variable('Embedding',
                                             shape=[self.vocab_size, self.word_emb_size],
                                             initializer=self.initializer, trainable=False)
        with tf.name_scope('slot_capsule_weights'):
            # Wl
            slot_capsule_weights_init = tf.get_variable('slot_capsule_weights_init',
                                                        shape=[1, self.max_sentence_length, self.slots_nr,
                                                               self.slot_output_dim, self.hidden_size * 2],
                                                        initializer=self.initializer)
            self.slot_capsule_weights = tf.tile(slot_capsule_weights_init, [self.batch_size, 1, 1, 1, 1])

        with tf.name_scope('slot_capsule_biases'):
            # bl
            slot_capsule_biases_init = tf.get_variable('slot_capsule_biases_init',
                                                       shape=[1, self.max_sentence_length, self.slots_nr,
                                                              self.slot_output_dim, 1],
                                                       initializer=self.initializer)
            self.slot_capsule_biases = tf.tile(slot_capsule_biases_init, [self.batch_size, 1, 1, 1, 1])

        with tf.name_scope('intent_capsule_weights'):
            # Wk
            intent_capsule_weights_init = tf.get_variable('intent_capsule_weights_init',
                                                          shape=[1, self.slots_nr + self.r, self.intents_nr,
                                                                 self.intent_output_dim, self.slot_output_dim],
                                                          initializer=self.initializer)
            self.intent_capsule_weights = tf.tile(intent_capsule_weights_init, [self.batch_size, 1, 1, 1, 1])

        with tf.name_scope('intent_capsule_biases'):
            # bk
            intent_capsule_biases_init = tf.get_variable('intent_capsule_biases_init',
                                                         shape=[1, self.slots_nr + self.r, self.intents_nr,
                                                                self.intent_output_dim, 1],
                                                         initializer=self.initializer)
            self.intent_capsule_biases = tf.tile(intent_capsule_biases_init, [self.batch_size, 1, 1, 1, 1])

        # Declare trainable variables for self attention
        with tf.name_scope('self_attention_weights'):
            self.W_s1 = tf.get_variable('W_s1', shape=[self.d_a, 2 * self.hidden_size],
                                        initializer=self.initializer)
            self.W_s2 = tf.get_variable('W_s2', shape=[self.r, self.d_a],
                                        initializer=self.initializer)

    def word_caps(self):
        """
            Build WordCaps subgraph
            Returns:
                H: hidden state matrix
        """
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

    def semantic_caps(self):
        """
            Build SemanticCaps subgraph
            Returns:
                A: Self-attention matrix
                M: Semantic matrix
        """
        # Use the hidden states H from WordCaps
        A = tf.nn.softmax(
            tf.map_fn(
                lambda x: tf.matmul(self.W_s2, x),
                tf.tanh(
                    tf.map_fn(
                        lambda x: tf.matmul(self.W_s1, tf.transpose(x)),
                        self.H))))

        M = tf.matmul(A, self.H)
        return A, M

    def _squash(self, input_tensor, axis=-1, epsilon=1e-7, name=None):
        """
            'Squashes' the input vector - keeps the orientation, but the length is reduced to be smaller than 1
            Args:
                input_tensor: input tensor to be squashed
                axis: axis of tensor on which to compute norm
                epsilon: small value that ensures the norm is different from 0
                name: name of operation
            Returns:
                squashed tensor
        """
        with tf.name_scope(name, default_name='squash'):
            squared_norm = tf.reduce_sum(tf.square(input_tensor), axis=axis,
                                         keep_dims=True)
            safe_norm = tf.sqrt(squared_norm + epsilon)
            squash_factor = squared_norm / (1. + squared_norm)
            unit_vector = input_tensor / safe_norm
            return squash_factor * unit_vector

    def _update_routing(self, caps1_n_caps, caps2_n_caps, caps2_predicted,
                        num_iter):
        """Dynamic routing-by-agreement algorithm
        Args:
          caps1_n_caps: number of capsules in layer l
          caps2_n_caps: number of capsules in layer l+1
          caps2_predicted: prediction vectors for layer l+1 capsules
          num_iter: number of iterations
        Returns:
          output_vector: output vectors of l+1 capsules
          raw_weights: raw weights (before softmax) - bij
          routing_weights: agreement weights (after softmax) - cij
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

    def slot_caps(self):
        """
            Defines the SlotCaps subgraph
            Returns:
                output_vector: SlotCaps output vectors vl
                weights_c: agreement weights cij
                slot_caps_predicted: SlotCaps prediction vectors pl|t
                weights_b: raw weights bij
        """
        word_caps_output = tf.nn.dropout(self.H, self.keep_prob)

        word_caps_output_expanded = tf.expand_dims(
            word_caps_output, axis=-1, name='word_caps_output_expanded')
        word_caps_output_tile = tf.expand_dims(word_caps_output_expanded, axis=2, name='word_caps_output_tile')
        word_caps_output_tiled = tf.tile(word_caps_output_tile, [1, 1, self.slots_nr, 1, 1],
                                         name='word_caps_output_tiled')

        slot_caps_predicted_matmul = tf.matmul(self.slot_capsule_weights, word_caps_output_tiled,
                                               name='slot_caps_predicted_matmul')
        slot_caps_predicted = tf.tanh(
            tf.add(slot_caps_predicted_matmul,
                   self.slot_capsule_biases))

        output_vector, weights_b, weights_c = self._update_routing(
            caps1_n_caps=self.max_sentence_length,
            caps2_n_caps=self.slots_nr,
            caps2_predicted=slot_caps_predicted,
            num_iter=self.slot_routing_num)
        return output_vector, weights_c, slot_caps_predicted, weights_b

    def intent_caps(self):
        """
            Defines the IntentCaps subgraph
            Returns:
                output_vector: IntentCaps output vectors uk
                weights_c: agreement weights cij
                intent_caps_predicted: IntentCaps prediction vectors qk|g
                weights_b: raw weights bij
        """
        # Expand dims for M
        semantic_caps_expanded = tf.expand_dims(self.M, axis=-1, name='semantic_caps_expanded_partial')
        semantic_caps_tile = tf.expand_dims(semantic_caps_expanded, axis=1, name='semantic_caps_expanded')

        semantic_slots_output_vecs = tf.concat([semantic_caps_tile, self.slot_output_vectors], axis=2)
        semantic_slot_caps_output = tf.nn.dropout(semantic_slots_output_vecs, self.keep_prob)

        semantic_slots_output_tshape = [0, 2, 1, 3, 4]
        semantic_slot_caps_output_transposed = tf.transpose(semantic_slot_caps_output, semantic_slots_output_tshape)

        semantic_slot_caps_output_tiled = tf.tile(semantic_slot_caps_output_transposed, [1, 1, self.intents_nr, 1, 1],
                                                  name='slot_caps_output_tiled')

        intent_caps_predicted_matmul = tf.matmul(self.intent_capsule_weights, semantic_slot_caps_output_tiled,
                                                 name='intent_caps_predicted')
        intent_caps_predicted = tf.tanh(tf.add(intent_caps_predicted_matmul, self.intent_capsule_biases))

        output_vector, weights_b, weights_c = self._update_routing(
            caps1_n_caps=self.slots_nr + self.r,
            caps2_n_caps=self.intents_nr,
            caps2_predicted=intent_caps_predicted,
            num_iter=self.intent_routing_num,
        )
        return output_vector, weights_c, intent_caps_predicted, weights_b

    def cross_entropy_loss(self):
        """
            Computes cross-entropy loss
            Returns:
                negative_loss: value of loss
        """
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
        positive_cost = (labels *
                         tf.cast(tf.less(logits, margin),
                                 tf.float32) *
                         tf.pow(logits - margin, 2))
        negative_cost = ((1 - labels) *
                         tf.cast(tf.greater(logits, -margin),
                                 tf.float32) *
                         tf.pow(logits + margin, 2))
        return (0.5 * positive_cost +
                downweight * 0.5 * negative_cost)

    def margin_loss(self):
        """ Compute total margin loss, with the added self-attention penalization term self_atten_loss
            Returns:
                margin_loss: value of margin loss
        """
        intent_vectors = tf.squeeze(self.intent_output_vectors, axis=[1, 4])
        intent_output_norm = util.safe_norm(intent_vectors)
        loss_val = self._margin_loss(self.encoded_intents, intent_output_norm)
        loss_val = tf.reduce_mean(loss_val)
        self_atten_mul = tf.matmul(self.attention, tf.transpose(self.attention, perm=[0, 2, 1]))
        sample_num, att_matrix_size, _ = self_atten_mul.get_shape()
        self_atten_loss = tf.square(tf.norm(
            self_atten_mul - np.identity(att_matrix_size.value)))
        margin_loss = 1000 * loss_val + self.alpha * tf.reduce_mean(self_atten_loss)

        # TensorBoard summaries
        self.margin_loss_tr_summary = tf.summary.scalar('margin_loss_training', margin_loss)
        self.margin_loss_val_summary = tf.summary.scalar('margin_loss_validation', margin_loss)
        return margin_loss

    def loss(self):
        """ Total loss (cross-entropy + margin loss)
            Returns:
                total_loss: total loss value
        """
        total_loss = tf.reduce_mean(
            self.margin_loss() + self.cross_entropy_loss())

        # TensorBoard summaries
        self.loss_tr_summary = tf.summary.scalar('total_loss_training', total_loss)
        self.loss_val_summary = tf.summary.scalar('total_loss_validation', total_loss)
        return total_loss

    def train(self):
        """ Defines train operation. Adam optimizer is used
            Returns:
                train_op: train operation that minimizes the total loss
        """
        train_op = tf.train.AdamOptimizer(self.learning_rate) \
            .minimize(self.loss_val)
        return train_op
