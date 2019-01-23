import tensorflow as tf
from tensorflow.contrib import autograph
import prgrph_ending_classifier

K = tf.keras.backend


class Sent_encoder(tf.keras.Model):
    def __init__(self, name=None):
        if name is None:
            name = 'sent_encoder'
        super().__init__(name=name)

    def call(self, inputs):
        """
        Description:
            encode given sentences with bag of words algorithm
        Args:
            input: sents shape: [current_prgrphs_num,max_sent_len,embedding_dim]
            output: encoded sentences of shape [current_prgrphs_num,encoding_dim] , here encoding_dim is equal to embedding_dim
        """
        ' I assume word embedding for indexes greater that sentnece length is zero vector, so it does not effect sentence encoding '

        return tf.reduce_sum(inputs, 1)


class EntityCell(tf.keras.layers.Layer):
    """
    Entity Cell.
    call with inputs and keys
    """

    def __init__(self, max_entity_num, entity_embedding_dim, name=None, initializer=None, **kwargs):
        self.max_entity_num = max_entity_num
        self.entity_embedding_dim = entity_embedding_dim

    def build(self, input_shape):
        shape = [self.entity_embedding_dim, self.entity_embedding_dim]
        self.U = K.variable(self.initializer(shape), name='U')
        self.V = K.variable(self.initializer(shape), name='V')
        self.W = K.variable(self.initializer(shape), name='W')
        self.built = True

    # def initialize_hidden(self, hiddens):
    #     self.batch_size = hiddens.shape[0]
    #     self.hiddens = hiddens

    # def assign_keys(self, entity_keys):
    #     self.keys = entity_keys

    def get_gate(self, encoded_sents, current_hiddens, current_keys):
        """
        Description:
            calculate the gate g_i for all hiddens of given paragraphs
        Args:
            inputs: encoded_sents of shape: [current_prgrphs_num, encoding_dim]
                    current_hiddens: [current_prgrphs_num, entity_num, entity_embedding_dim]
                    current_keys: [current_prgrphs_num, entity_num, entity_embedding_dim]

            output: gates of shape : [curr_prgrphs_num, entity_num]
        """

        print('enocded_sents dtype:', encoded_sents.dtype)
        print('current_hiddens dtype:', current_hiddens.dtype)
        return tf.sigmoid(tf.reduce_sum(tf.multiply(tf.expand_dims(encoded_sents, 1), current_hiddens) +
                                        tf.multiply(tf.expand_dims(encoded_sents, 1), current_keys), axis=2))

    def update_hidden(self, gates, current_hiddens, current_keys, encoded_sents):
        """
        Description:
            updates hidden_index for all prgrphs
        Args:
            inputs: gates shape: [current_prgrphs_num, entity_num]
                    encoded_sents of shape: [current_prgrphs_num, encoding_dim]
                    current_hiddens: [current_prgrphs_num, entity_num, entity_embedding_dim]
                    current_keys: [current_prgrphs_num, entity_num, entity_embedding_dim]
        """
        curr_prgrphs_num = current_hiddens.shape[0]
        h_tilda = self.activation(
            tf.reshape(tf.matmul(tf.reshape(current_hiddens, [-1, self.entity_embedding_dim]), self.U) +
                       tf.matmul(tf.reshape(current_keys, [-1, self.entity_embedding_dim]), self.V) +
                       tf.matmul(tf.reshape(tf.tile(tf.expand_dims(encoded_sents, 1), [1, self.entity_num, 1]),
                                            shape=[-1, self.entity_embedding_dim]), self.W),
                       shape=[curr_prgrphs_num, self.entity_num, self.entity_embedding_dim]))
        'h_tilda shape: [current_prgrphs_num, entity_num, entity_embedding_dim]'
        # tf.multiply(gates,h_tilda)
        updated_hiddens = current_hiddens + tf.multiply(
            tf.tile(tf.expand_dims(gates, axis=2), [1, 1, self.entity_embedding_dim]), h_tilda)

        return updated_hiddens

    def normalize(self, hiddens):
        return tf.nn.l2_normalize(hiddens, axis=2)

    def call(self, inputs, prev_states, keys, use_shared_keys=False, **kwargs):
        """

        Args:
            inputs: encoded_sents of shape [batch_size, encoding_dim] , batch_size is equal to current paragraphs num
            prev_states: tensor of shape [batch_size, key_num, dim]
            keys: tensor of shape [batch_size, key_num, dim] if use_shared_keys is False and
                  [key_num, dim] if use_shared_keys is True
            use_shared_keys: if it is True, it use shared keys for all samples.

        Returns:
            next_state: tensor of shape [batch_size, key_num, dim]
        """

        # assert isinstance(inputs, list)

        # current_hiddens = tf.gather(self.hiddens, indices)
        # print('ENCODE')
        # print(self.keys.shape)
        # print(indices)
        # current_keys = tf.gather(self.keys, indices)

        # if current_hiddens.shape != current_keys.shape:
        #     raise AttributeError('hiddens and kes must have same shape')

        encoded_sents = inputs
        gates = self.get_gate(encoded_sents, prev_states, keys)
        updated_hiddens = self.update_hidden(gates, prev_states, keys, encoded_sents)
        return self.normalize(updated_hiddens)

    def get_intial_state(self):
        return tf.zeros([self.max_entity_num, self.entity_embedding_dim], dtype=tf.int32)

    def __call__(self, inputs, prev_state, keys, use_shared_keys=False, **kwargs):
        """
        Do not fill this one
        """
        return super().__call__(inputs=inputs, prev_state=prev_state, keys=keys,
                                use_shared_keys=use_shared_keys, **kwargs)


@autograph.convert()
def simple_entity_network(entity_cell, inputs, keys,
                          initial_entity_hidden_state=None,
                          use_shared_keys=False, return_last=True):
    """
    Args:
        entity_cell: the EntityCell
        inputs: a list containing a tensor of shape [batch_size, seq_length, dim] and its mask of shape [batch_size, seq_length]
                batch_size=current paragraphs num, seq_length=max number of senteces
        keys: tensor of shape [batch_size, key_num, dim] if use_shared_keys is False and
                  [key_num, dim] if use_shared_keys is True
        use_shared_keys: if it is True, it use shared keys for all samples.
        mask_inputs: tensor of shape [batch_size, seq_length] and type tf.bool
        initial_entity_hidden_state: a tensor of shape [batch_size, key_num, dim]
        return_last: if it is True, it returns the last state, else returns all states

    Returns:
        if return_last = True then a tensor of shape [batch_size, key_num, dim] else shape of
                         [batch_size, seq_length, key_num, dim]
    """
    encoded_sents, mask = inputs
    seq_length = encoded_sents.shape[1]
    batch_size = encoded_sents.shape[0]
    key_num = keys.shape[1]
    entity_embedding_dim = keys.shape[2]

    if initial_entity_hidden_state is None:
        initial_entity_hidden_state = tf.tile(tf.expand_dims(entity_cell.get_initial_state(), axis=0),
                                              [batch_size, 1, 1])
    if return_last:
        entity_hiddens = initial_entity_hidden_state
    else:
        all_entity_hiddens = tf.expand_dims(initial_entity_hidden_state, axis=1)
    for i in range(seq_length):
        ''' to see which sentences are available '''
        indices = tf.where(mask[:, i])
        indices = tf.cast(tf.squeeze(indices, axis=1), tf.int32)
        curr_encoded_sents = tf.gather(encoded_sents, indices)
        curr_keys = tf.gather(keys, indices)
        if return_last:
            prev_states = tf.gather(entity_hiddens, indices)
            updated_hiddens = entity_cell(curr_encoded_sents, prev_states, curr_keys)
            entity_hiddens = entity_hiddens + tf.scatter_nd(tf.expand_dims(indices, 1), updated_hiddens - prev_states,
                                                            keys.shape)
        else:
            prev_states = tf.gather(all_entity_hiddens[:, -1, :, :], indices)
            updated_hiddens = tf.expand_dims(entity_cell(curr_encoded_sents, prev_states, curr_keys), axis=1)
            all_entity_hiddens = tf.concat([all_entity_hiddens,
                                            tf.scatter_nd(tf.expand_dims(indices, 1), updated_hiddens,
                                                          [batch_size, 1, key_num, entity_embedding_dim])], axis=1)

    if return_last:
        return entity_hiddens
    else:
        return all_entity_hiddens


@autograph.convert()
def rnn_entity_network_encoder(entity_cell, rnn_cell, inputs, keys, mask_inputs=None,
                               initial_hidden_state=None,
                               initial_entity_hidden_state=None, update_positions=None, use_shared_keys=False,
                               return_last=True, self_attention=False):
    """


    """
    raise NotImplementedError


@autograph.convert()
def rnn_entity_network_decoder(entity_cell, rnn_cell, softmax_layer, embedding_layer, keys, training,
                               initial_hidden_state=None, initial_entity_hidden_state=None,
                               labels=None,
                               num_inputs=None, num_keys=None, encoder_hidden_states=None,
                               update_positions=None, use_shared_keys=False, return_last=True,
                               attenton=False, self_attention=False):
    """

    Args:
        entity_cell: EntityCell
        rnn_cell: RNNCell
        softmax_layer: softmax layer
        embedding_layer: embedding layer
        keys: either a tensor of shape [batch_size, key_num, dim] or [key_num, dim] depending on use_shared_keys
        training: boolean ...
    :param initial_hidden_state:
    :param initial_entity_hidden_state:
    :param labels:
    :param num_inputs:
    :param num_keys:
    :param encoder_hidden_states:
    :param update_positions:
    :param use_shared_keys:
    :param return_last:
    :param attenton:
    :param self_attention:
    :return:
    """
    raise NotImplementedError


"""

"""


# def get_embeddings(self,embedding_matrix,input):
#     return tf.nn.embedding_lookup(embedding_matrix, input)

class BasicRecurrentEntityEncoder(tf.keras.Model):
    def __init__(self, embedding_layer, max_entity_num=None, entity_embedding_dim=None, entity_cell=None, name=None,
                 **kwargs):
        if name is None:
            name = 'BasicRecurrentEntityEncoder'
        self.name = name
        if entity_cell is None:
            if entity_embedding_dim is None:
                raise AttributeError('entity_embedding_dim should be given')
            if max_entity_num is None:
                raise AttributeError('max_entity_num should be given')
            entity_cell = EntityCell(max_entity_num=max_entity_num, entity_embedding_dim=entity_embedding_dim,
                                     name='entity_cell')
        self.entity_cell = entity_cell
        self.embedding_layer = embedding_layer

    def call(self, inputs, keys, num_inputs=None, initial_entity_hidden_state=None,
             use_shared_keys=False, return_last=True, **kwargs):
        """
        Args:
            inputs: paragraph, paragraph mask in a list , paragraph of shape:[batch_size, max_sents_num, max_sents_len,
            keys: entity keys of shape : [batch_size, max_entity_num, entity_embedding_dim]
            num_inputs: ??? mask for keys??? is it needed in encoder?
            initial_entity_hidden_state
            use_shared_keys: bool
            return_last: if true, returns last state of entity hiddens, else returns all states
        """

        if len(inputs) != 2:
            raise AttributeError('expected 2 inputs but', len(inputs), 'were given')
        prgrph, prgrph_mask = inputs
        batch_size = prgrph.shape[0]
        max_sent_num = prgrph.shape[1]
        prgrph_embeddings = tf.nn.embedding_lookup(self.embedding_layer, prgrph)
        'prgrph_embeddings shape: [batch_size, max_sent_num, max_sent_len, embedding_dim]'
        encoded_sents = tf.zeros([batch_size, 1, prgrph.shape[3]])
        for i in range(max_sent_num):
            ''' to see which sentences are available '''
            indices = tf.where(prgrph_mask[:, i, 0])
            # print('indices shape encode:, indices.shape)
            indices = tf.cast(tf.squeeze(indices, axis=1), tf.int32)
            current_sents = tf.gather(prgrph_embeddings[:, i, :, :], indices)
            # print('current_sents_call shape:', current_sents.shape)
            curr_encoded_sents = tf.expand_dims(self.sent_encoder_module(current_sents), axis=1)
            encoded_sents = tf.concat([encoded_sents, curr_encoded_sents], axis=1)

        encoded_sents = encoded_sents[:, 1:, :]
        sents_mask = prgrph_mask[:, :, 0]
        return simple_entity_network(entity_cell=self.entity_cell, inputs=[encoded_sents, sents_mask], keys=keys,
                                     initial_entity_hidden_state=initial_entity_hidden_state,
                                     use_shared_keys=use_shared_keys,
                                     return_last=return_last)


class RNNRecurrentEntityEncoder(tf.keras.Model):
    def __init__(self, num_units=None, entity_cell=None, rnn_cell=None, name=None, **kwargs):
        raise NotImplementedError

    def call(self, inputs, keys, num_inputs=None, num_keys=None,
             initial_hidden_state=None,
             initial_entity_hidden_state=None, update_positions=None, use_shared_keys=False,
             return_last=True, self_attention=False, **kwargs):
        raise NotImplementedError


class RNNRecurrentEntitiyDecoder(tf.keras.Model):
    def __init__(self, embedding_matrix, rnn_hidden_size, embedding_dim, entity_cell=None, entity_embedding_dim=None,
                 max_entity_num=None,
                 rnn_cell=None, vocab_size=None, prgrph_ending_Classifier=None, max_sent_num=None,
                 num_units=None, entity_dense=None, start_hidden_dense=None,
                 softmax_layer=None, name=None, **kwargs):
        if name is None:
            name = 'RNNRecurrentEntitiyDecoder'
        self.name = name
        self.embedding_matrix = embedding_matrix
        self.rnn_hidden_size = rnn_hidden_size
        self.embedding_dim = embedding_dim
        if vocab_size is not None:
            self.vocab_size = vocab_size

        if entity_cell in None:
            if entity_embedding_dim is None:
                raise AttributeError("entity_embedding_dim and entity_cell can't be both None")
            if max_entity_num is None:
                raise AttributeError("max_entity_num and entity_cell can't be both None")
            entity_cell = EntityCell(max_entity_num=max_entity_num, entity_embedding_dim=entity_embedding_dim,
                                     name='entity_cell')
        self.entity_cell = entity_cell

        if rnn_cell is None:
            # if rnn_hidden_size is None:
            #     raise AttributeError("rnn_hidden_size and rnn_cell can't be both None")
            rnn_cell = tf.keras.layers.LSTM(rnn_hidden_size, return_state=True)
            self.rnn_cell = rnn_cell

        if softmax_layer is None:
            if vocab_size is None:
                raise AttributeError("softmax_layer and vocab_size can't be both None")
            self.softmax_layer = tf.keras.layers.Dense(vocab_size, activation='softmax')

        if prgrph_ending_classifier is None:
            if max_sent_num is None:
                raise AttributeError("prgrph_ending_Classifier and max_sent_num can't be both None")
            # if rnn_hidden_size is None:
            #     raise AttributeError("prgrph_ending_Classifier and rnn_hidden_size can't be both None")
            if entity_embedding_dim is None:
                raise AttributeError("prgrph_ending_Classifier and entity_embedding_dim can't be both None")
            prgrph_ending_Classifier = prgrph_ending_classifier.Prgrph_ending_classifier(
                max_sent_num=self.max_sent_num,
                encoding_dim=rnn_hidden_size,
                entity_embedding_dim=entity_embedding_dim)
        self.prgrph_ending_classifier = prgrph_ending_Classifier

        'entity_dense: from entity_embedding_dim to hidden_size, which is used after attending on entities'
        if entity_dense is None:
            entity_dense = tf.keras.layers.Dense(rnn_hidden_size)
        self.entity_dense = entity_dense

        if start_hidden_dense is None:
            start_hidden_dense = tf.keras.layers.Dense(rnn_hidden_size)
        self.start_hidden_dense = start_hidden_dense

    def build(self, input_shape):
        self.entity_attn_matrix = K.random_normal_variable(shape=[self.rnn_hidden_size, self.embedding_dim],
                                                           mean=0, scale=0.05, name='entity_attn_matrix')

    def decode_train(self, inputs, labels, entity_cell, keys, keys_mask, encoder_hidden_states=None,
                     initial_hidden_state=None, use_shared_keys=False,
                     return_last=True, attention=False, self_attention=False):
        '''
                   Language model on given paragraph (second prgrph)
        '''

        """
        
        inputs: [test_mode_bool, entity_hiddens, vocab_size]
        labels: [prgrph, prgrph_mask]
        
        :return: guessed words as final_output and their mask as output_mask and ground truth targets
        """

        if len(inputs) != 3:
            raise AttributeError('expected 3 inputs but', len(inputs), 'were given')
        if len(labels) != 2:
            raise AttributeError('expected 2 labels but', len(labels), 'were given')

        print('IN DECODE_TRAIN')

        test_mode_bool, entity_hiddens, vocab_size = inputs
        prgrph, prgrph_mask = labels
        batch_size = prgrph.shape[0]
        max_sent_num = prgrph.shape[1]
        max_sent_len = prgrph.shape[2]

        prgrph_embeddings = self.get_embeddings(prgrph)

        self.update_entity_module.initialize_hidden(entity_hiddens)

        final_output = tf.zeros(shape=[batch_size, max_sent_num, max_sent_len, vocab_size], dtype=tf.float32)
        final_targets = tf.zeros(shape=[batch_size, max_sent_num, max_sent_len], dtype=tf.int32)
        output_mask = tf.zeros(shape=[batch_size, max_sent_num, max_sent_len], dtype=tf.int32)

        ' stores previous hidden_states of the lstm for the prgrph '
        hidden_states = tf.zeros([batch_size, max_sent_num * max_sent_len, self.hidden_Size])
        hiddens_mask = tf.reshape(prgrph_mask, [batch_size, -1])

        cell_states = tf.zeros([1])

        for i in range(max_sent_num):
            # print('p2_mask',p2_mask)
            current_sents_indices = tf.where(prgrph_mask[:, i, 0])
            current_sents_indices = tf.squeeze(current_sents_indices, axis=1)
            for j in range(max_sent_len):
                print('current word indeX:', i, j)
                ' indices of available paragraphs'
                # print('prgrph_mask.shape',tf.where(prgrph_mask).shape)
                # print('prgprh_mask[:,i,j]',prgrph_mask[:,i,j])
                indices = tf.cast(tf.where(prgrph_mask[:, i, j]), dtype=tf.int32)
                print('indices shape:', indices.shape)
                indices = tf.squeeze(indices, axis=1)
                if indices.shape[0] > 0:
                    # print('indices_p2_mask:',indices)   #indices_p2_mask: tf.Tensor([[0]], shape=(1, 1), dtype=int64)
                    if j == 0:
                        # print('start token',self.embedding_matrix[self.start_token].shape)
                        lstm_inputs = tf.tile(tf.expand_dims(self.embedding_matrix[self.start_token], axis=0),
                                              [batch_size, 1])
                        print('lstm_first_inputs:', lstm_inputs.shape)
                    else:
                        # print('prgrph_embedding_shape:',prgrph_embeddings[:, i, j - 1, :].shape)
                        lstm_inputs = tf.gather(prgrph_embeddings[:, i, j - 1, :], indices)

                    # print(tf.gather(second_prgrph[:, i, j], indices).shape)
                    lstm_targets = tf.gather(prgrph[:, i, j], indices)
                    if i * max_sent_len + j == 0:
                        curr_sents_curr_hidden = self.start_hidden_dense(tf.reduce_sum(entity_hiddens, axis=1))
                        cell_states = tf.zeros([batch_size, self.hidden_Size], tf.float32)
                        print('curr_sents_cell_state.shape', cell_states.shape)
                    else:
                        curr_sents_prev_hiddens = tf.gather(hidden_states[:, :i * max_sent_len + j, :], indices)
                        curr_sents_prev_hiddens_mask = tf.gather(hiddens_mask[:, :i * max_sent_len + j], indices)
                        curr_sents_entities = tf.gather(self.update_entity_module.hiddens, indices)
                        curr_sents_curr_hidden = self.calculate_hidden(curr_sents_prev_hiddens, curr_sents_entities,
                                                                       mask=curr_sents_prev_hiddens_mask)
                    curr_sents_cell_state = tf.gather(cell_states, indices)
                    print('lstm_inputs shape:', lstm_inputs.shape)
                    output, next_hidden, next_cell_state = self.lstm(tf.expand_dims(lstm_inputs, axis=1),
                                                                     initial_state=[
                                                                         curr_sents_curr_hidden,
                                                                         curr_sents_cell_state])

                    'updating cell_states'
                    curr_cells_prev_state = tf.gather(cell_states, indices)
                    cell_states = cell_states + tf.scatter_nd(tf.expand_dims(indices, axis=1),
                                                              next_cell_state - curr_cells_prev_state,
                                                              [batch_size, self.hidden_Size])
                    print('next_hidden shape:', next_hidden.shape)
                    'output shape:[len(indices), hidden_size] here, output is equal to next_hidden'
                    index_vector = tf.ones([indices.shape[0], 1], tf.int32) * (i * max_sent_len + j)
                    print('indices type:', indices.dtype)
                    new_indices = tf.concat(values=[tf.expand_dims(indices, 1), index_vector], axis=1)
                    print('new_indices:', new_indices)
                    hidden_states = hidden_states + tf.scatter_nd(new_indices, next_hidden,
                                                                  shape=[batch_size, hidden_states.shape[1],
                                                                         self.hidden_Size])
                    print('hidden_state.shape', hidden_states.shape)
                    # print('hidden_states:',hidden_states)
                    output = self.decoder_dense(output)
                    print('ouput_shape', output.shape)

                    second_dim_ind = tf.ones([indices.shape[0], 1], tf.int32) * i
                    third_dim_ind = tf.ones([indices.shape[0], 1], tf.int32) * j
                    new_indices_output = tf.keras.layers.concatenate(
                        inputs=[tf.expand_dims(indices, 1), second_dim_ind, third_dim_ind], axis=1)
                    final_output = final_output + tf.scatter_nd(new_indices_output, output,
                                                                shape=[batch_size, max_sent_num, max_sent_len,
                                                                       vocab_size])

                    print('final_targets dtype', final_targets.dtype)
                    print(tf.scatter_nd(new_indices_output, lstm_targets,
                                        shape=[batch_size, max_sent_num, max_sent_len]).dtype)
                    final_targets = final_targets + tf.cast(
                        tf.scatter_nd(new_indices_output, lstm_targets, shape=[batch_size, max_sent_num, max_sent_len]),
                        dtype=tf.int32)

                    t = tf.ones([indices.shape[0]], dtype=tf.int32)
                    output_mask = output_mask + tf.scatter_nd(new_indices_output, t,
                                                              shape=[batch_size, max_sent_num, max_sent_len])

            if test_mode_bool == True:
                return tf.zeros([1])
            # print('current_sents_indices',current_sents_indices)
            current_sents = tf.gather(prgrph_embeddings[:, i, :, :], current_sents_indices)
            print('decode_train, currents_sents shape', current_sents.shape)
            encoded_sents = self.sent_encoder_module([current_sents])
            self.update_entity_module([encoded_sents, current_sents_indices])
            # print('updated_hiddens', self.update_entity_module.hiddens)
        final_output = tf.reshape(final_output, [batch_size, -1, vocab_size])
        final_targets = tf.reshape(final_targets, [batch_size, -1])
        output_mask = tf.reshape(output_mask, [batch_size, -1])
        return final_output, final_targets, output_mask

    def decode_test(self, inputs, entity_cell, keys, keys_mask, encoder_hidden_states=None,
                    initial_hidden_state=None, use_shared_keys=False,
                    return_last=True, attention=False, self_attention=False):
        '''
            TASK 3 : predicting second paragraph
        '''

        ''' 
        inputs: entity_hiddens last state
        # yields predicted output of shape [batch_size, vocab_size] each step
        '''

        # if entity_hiddens is None:
        #     raise AttributeError('entity_hiddens is None')
        # if max_sent_len is None:
        #     raise AttributeError('max_sent_len is None')
        # if eos_ind is None:
        #     raise AttributeError('eos_ind is None')

        if len(inputs) != 3:
            raise AttributeError('expected 3 inputs but', len(inputs), 'were given')

        entity_hiddens, max_sent_len, eos_ind = inputs

        self.update_entity_module.initialize_hidden(entity_hiddens)

        batch_size = entity_hiddens.shape[0]

        ' stores previous hidden_states of the lstm for the prgrph '
        hidden_states = tf.zeros([batch_size, self.max_sent_num * max_sent_len, self.hidden_Size])

        'new masks will be added each step'
        hiddens_mask = tf.equal(tf.ones([batch_size, 1]), 1)

        'indices of generated words'
        generated_prgrphs = tf.zeros([batch_size, self.max_sent_num, max_sent_len], dtype=tf.int32)
        generated_prgrphs_embeddings = self.get_embeddings(generated_prgrphs)

        last_noneos_output = tf.zeros([1], dtype=tf.float32)
        ' indices of available paragraphs'
        unfinished_prgrphs_indices = tf.range(start=0, limit=batch_size, dtype=tf.int32)
        ending_hidden_indices = tf.zeros([batch_size, self.max_sent_num], dtype=tf.int32)

        cell_states = tf.zeros([1])

        for i in range(self.max_sent_num):
            indices = tf.identity(unfinished_prgrphs_indices)
            if indices.shape[0] > 0:
                for j in range(max_sent_len):
                    print('current word indeX:', i, j)
                    print("GENERATED PARAGRAPHS")
                    print(generated_prgrphs)
                    if j == 0:
                        # print('start token',self.embedding_matrix[self.start_token].shape)
                        lstm_inputs = tf.tile(tf.expand_dims(self.embedding_matrix[self.start_token], axis=0),
                                              [batch_size, 1])
                    else:
                        lstm_inputs = tf.gather(generated_prgrphs_embeddings[:, i, j - 1, :], indices)

                    if i * max_sent_len + j == 0:
                        # print(tf.reduce_sum(entity_hiddens,axis=1))
                        curr_sents_curr_hidden = self.start_hidden_dense(tf.reduce_sum(entity_hiddens, axis=1))
                        cell_states = tf.zeros([batch_size, self.hidden_Size], tf.float32)
                    else:
                        print('hidden_states shape:', hidden_states.shape)
                        curr_sents_prev_hiddens = tf.gather(hidden_states[:, :i * max_sent_len + j, :], indices)
                        print('curr_sents_prev_hiddens shape:', curr_sents_prev_hiddens.shape, indices.shape)
                        curr_sents_prev_hiddens_mask = tf.gather(hiddens_mask[:, 1:], indices)
                        curr_sents_entities = tf.gather(self.update_entity_module.hiddens, indices)
                        curr_sents_curr_hidden = self.calculate_hidden(curr_sents_prev_hiddens,
                                                                       curr_sents_entities,
                                                                       mask=curr_sents_prev_hiddens_mask)

                    curr_sents_cell_state = tf.gather(cell_states, indices)
                    lstm_output, next_hidden, next_cell_state = self.lstm(tf.expand_dims(lstm_inputs, axis=1),
                                                                          initial_state=[curr_sents_curr_hidden,
                                                                                         curr_sents_cell_state])

                    'to handle the case where at least one sentence has generated <oes> as the first word!'
                    # if j == 0:
                    #     count=0
                    #     lstm_output1 = self.decoder_dense(lstm_output)
                    #     last_output = tf.cast(tf.argmax(lstm_output1, dimension=1), tf.int32)
                    #     all_eos_vector = tf.cast(tf.ones([last_output.shape[0]]) * eos_ind,tf.int32)
                    #     # print('last_output-all_eos_vector',last_output-all_eos_vector)
                    #     # print(np.count_nonzero(last_output - all_eos_vector),last_output.shape[0])
                    #     # print(np.count_nonzero(last_output - all_eos_vector) != last_output.shape[0])
                    #     while np.count_nonzero(last_output - all_eos_vector) != last_output.shape[0] and count<20:
                    #         lstm_output, next_hidden, next_cell_state = self.lstm(tf.expand_dims(lstm_inputs, axis=1),
                    #                                                               initial_state=[curr_sents_curr_hidden,
                    #                                                                              curr_sents_cell_state])
                    #         lstm_output1 = self.decoder_dense(lstm_output)
                    #         last_output = tf.cast(tf.argmax(lstm_output1, dimension=1), tf.int32)
                    #         count=count+1

                    'updating cell_states'
                    curr_cells_prev_state = tf.gather(cell_states, indices)
                    cell_states = cell_states + tf.scatter_nd(tf.expand_dims(indices, axis=1),
                                                              next_cell_state - curr_cells_prev_state,
                                                              [batch_size, self.hidden_Size])

                    print('next_hidden shape:', next_hidden.shape)
                    'output shape:[available_prgrphs_num, hidden_size] here, output is equal to next_hidden'
                    index_vector = tf.ones([indices.shape[0], 1], tf.int32) * (i * max_sent_len + j)
                    new_indices = tf.keras.layers.concatenate(inputs=[tf.expand_dims(indices, 1), index_vector],
                                                              axis=1)
                    print('new_indices:', new_indices)
                    hidden_states = hidden_states + tf.scatter_nd(new_indices, next_hidden,
                                                                  shape=[batch_size, hidden_states.shape[1],
                                                                         self.hidden_Size])
                    'constructing next_hidden mask and concatenating to hiddens_mask'
                    boolean_vec = tf.ones(indices.shape[0], dtype=tf.bool)
                    next_hidden_mask = tf.scatter_nd(tf.expand_dims(indices, axis=1),
                                                     tf.expand_dims(boolean_vec, axis=1),
                                                     shape=[batch_size, 1])
                    hiddens_mask = tf.keras.layers.concatenate(inputs=[hiddens_mask, next_hidden_mask], axis=1)

                    # print('hidden_states:',hidden_states)
                    lstm_output = self.decoder_dense(lstm_output)
                    last_output = tf.cast(tf.argmax(lstm_output, dimension=1), tf.int32)

                    if j == max_sent_len - 1:
                        last_output = tf.cast(tf.ones([indices.shape[0]]) * eos_ind, dtype=tf.int32)
                    'last_output is a one_dimensional vector'
                    print('last_output shape and dtype:', last_output.shape, last_output.dtype)
                    # a=tf.ones([last_output.shape[0]]) * i
                    # print(a)
                    generated_words_indices = tf.transpose(tf.stack([indices,
                                                                     tf.cast(tf.ones([indices.shape[0]]) * i, tf.int32),
                                                                     tf.cast(tf.ones([indices.shape[0]]) * j,
                                                                             tf.int32)]))
                    # print(tf.scatter_nd(generated_words_indices, last_output,[batch_size, self.max_sent_num, max_sent_len]).dtype)
                    generated_prgrphs = generated_prgrphs + tf.cast(tf.scatter_nd(generated_words_indices, last_output,
                                                                                  [batch_size, self.max_sent_num,
                                                                                   max_sent_len]), dtype=tf.int32)

                    generated_prgrphs_embeddings = generated_prgrphs_embeddings + \
                                                   tf.scatter_nd(generated_words_indices,
                                                                 self.get_embeddings(last_output),
                                                                 [batch_size, self.max_sent_num, max_sent_len,
                                                                  self.embedding_dim])

                    'updating indices by eliminating indices which eos was generated in them'
                    indices = tf.boolean_mask(indices, tf.logical_not(tf.equal(last_output, eos_ind)))
                    eos_indices = tf.cast(tf.where(tf.equal(last_output, eos_ind)), tf.int32)
                    # print('last_output',last_output)
                    # print('eos_indices',eos_indices)
                    if (eos_indices.shape[0] > 0):
                        hidden_index_vec = tf.ones([eos_indices.shape[0]]) * (i * max_sent_len + j)
                        index_vec2 = tf.ones([eos_indices.shape[0], 1], tf.int32) * i
                        new_indices2 = tf.keras.layers.concatenate(inputs=[eos_indices, index_vec2], axis=1)
                        # print(tf.scatter_nd(new_indices2, hidden_index_vec,[batch_size, self.max_sent_num]).dtype)
                        ending_hidden_indices = ending_hidden_indices + tf.cast(
                            tf.scatter_nd(new_indices2, hidden_index_vec,
                                          [batch_size, self.max_sent_num]), tf.int32)

                # last_noneos_output = tf.gather(last_output, indices)

                'updating unfinished_sents_indices'
                ending_hidden_indices_upto_i = tf.gather(ending_hidden_indices, unfinished_prgrphs_indices)[:, :i + 1]
                print('ending_hiddens_upto_i', ending_hidden_indices_upto_i.shape)
                a = tf.tile(tf.expand_dims(unfinished_prgrphs_indices, 1), multiples=[1, i + 1])
                ending_indices = tf.keras.layers.concatenate(
                    [tf.expand_dims(a, axis=2), tf.expand_dims(ending_hidden_indices_upto_i, axis=2)], axis=2)
                'ending_indices shape: [available_prgrphs_num, generated_sents_num, 2]'
                curr_prgrphs_last_hiddens = tf.gather_nd(hidden_states, ending_indices)

                curr_sents = tf.gather(generated_prgrphs_embeddings, unfinished_prgrphs_indices)[:, i, :, :]
                encoded_sents = self.sent_encoder_module([curr_sents])
                self.update_entity_module([encoded_sents, unfinished_prgrphs_indices])
                classifier_results = self.prgrph_ending_classifier([curr_prgrphs_last_hiddens,
                                                                    self.update_entity_module.hiddens])
                'classifier_results : probabilities'
                print('classifier_results', classifier_results)
                bool_results = tf.squeeze(tf.less(classifier_results, 0.5), axis=1)
                print('bool_results', bool_results)
                not_ended_prgrphs_indices = tf.squeeze(tf.where(bool_results), axis=1)
                print('unfinished_prgrph_indices', unfinished_prgrphs_indices)
                print('not_ended_prgrphs_indices', not_ended_prgrphs_indices)
                unfinished_prgrphs_indices = tf.gather(unfinished_prgrphs_indices, not_ended_prgrphs_indices)
                # print('unfinished_prgrph_indices2', unfinished_prgrphs_indices)

        return generated_prgrphs

    def call(self, inputs, keys, keys_mask, training, initial_hidden_state=None,
             encoder_hidden_states=None, labels=None,
             num_inputs=None,
             update_positions=None, use_shared_keys=False,
             return_last=True,
             attention=False, self_attention=False):
        """

        inputs: [test_mode_bool, entity_hiddens, vocab_size ] in training mode
                [entity_hiddens, max_sent_len, eos_ind] in test mode
        keys: entity keys
        training: bool
        initial_hidden_state: for rnn
        encoder_hidden_states
        labels: [prgrph, prgrph_mask] in train mode
        num_inputs: ????
        update_positions: ???
        keys_mask : mask for entity keys, which is used while attending on entities
        return_last: if true, returns last state of entity hiddens, else returns all states

        :return: generated paragraph in test mode
        """
        if training:
            if labels is None:
                raise AttributeError('labels are None')
            return self.decode_train(inputs[1:], labels, self.entity_cell,
                                     keys, keys_mask, encoder_hidden_states, initial_hidden_state, use_shared_keys,
                                     return_last, attention, self_attention)

        else:
            return self.decode_test(inputs[1:], keys, keys_mask, encoder_hidden_states, initial_hidden_state,
                                    use_shared_keys,
                                    return_last, attention, self_attention)
