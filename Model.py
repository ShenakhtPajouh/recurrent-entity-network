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

    def __init__(self, max_entity_num, entity_embedding_dim, activation=tf.nn.relu, name=None
                 , **kwargs):
        if name is None:
            name = 'Entity_cell'
        super().__init__(name=name)
        self.max_entity_num = max_entity_num
        self.entity_embedding_dim = entity_embedding_dim
        self.activation = activation
        # self.initializer=initializer

        # if initializer is None:
        #     self.initializer = tf.keras.initializers.random_normal()

        self.U = None
        self.V = None
        self.W = None
        self.built = False

    def build(self, input_shape):
        print("in build")
        shape = [self.entity_embedding_dim, self.entity_embedding_dim]
        self.U = self.add_weight(shape=shape, name='U')
        self.V = self.add_weight(shape=shape, name='V')
        self.W = self.add_weight(shape=shape, name='W')

        self.built = True

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

        # print('enocded_sents dtype:', tf.shape(encoded_sents))
        # print('current_hiddens dtype:', current_hiddens.dtype)
        # print('enocded_sents shape:', tf.shape(encoded_sents))
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
        curr_prgrphs_num = tf.shape(current_hiddens)[0]
        h_tilda = self.activation(
            tf.reshape(tf.matmul(tf.reshape(current_hiddens, [-1, self.entity_embedding_dim]), self.U) +
                       tf.matmul(tf.reshape(current_keys, [-1, self.entity_embedding_dim]), self.V) +
                       tf.matmul(tf.reshape(tf.tile(tf.expand_dims(encoded_sents, 1), [1, self.max_entity_num, 1]),
                                            shape=[-1, self.entity_embedding_dim]), self.W),
                       shape=[curr_prgrphs_num, self.max_entity_num, self.entity_embedding_dim]))
        'h_tilda shape: [current_prgrphs_num, entity_num, entity_embedding_dim]'
        # tf.multiply(gates,h_tilda)
        # print("gates shape:", tf.shape(gates))
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

        encoded_sents = inputs
        gates = self.get_gate(encoded_sents, prev_states, keys)
        updated_hiddens = self.update_hidden(gates, prev_states, keys, encoded_sents)
        return self.normalize(updated_hiddens)

    def get_initial_state(self):
        return tf.zeros([self.max_entity_num, self.entity_embedding_dim], dtype=tf.float32)

    # def __call__(self, inputs, prev_state, keys, use_shared_keys=False, **kwargs):
    #     """
    #     Do not fill this one
    #     """
    #     return super().__call__(inputs=inputs, prev_state=prev_state, keys=keys,
    #                             use_shared_keys=use_shared_keys, **kwargs)


@autograph.convert()
def update_entities_True_rl(encoded_sents, seq_len, mask, keys, entity_cell, entity_hiddens):
    '''
    update entitites with return_last=True
    '''
    # seq_length = encoded_sents.shape[1]
    # batch_size = tf.shape(encoded_sents)[0]
    # key_num = tf.shape(keys)[1]
    # entity_embedding_dim = tf.shape(keys)[2]
    indices = tf.zeros([1])
    for i in range(seq_len):
        ''' to see which sentences are available '''
        indices = tf.where(mask[:, i])
        # print(encoded_sents.shape)
        indices = tf.cast(tf.squeeze(indices, axis=1), tf.int32)
        curr_encoded_sents = tf.gather(encoded_sents[:, i, :], indices)
        # print("curr_encoded_sents shape:",curr_encoded_sents.shape)
        curr_keys = tf.gather(keys, indices)
        prev_states = tf.gather(entity_hiddens, indices)
        updated_hiddens = entity_cell(curr_encoded_sents, prev_states, curr_keys)
        entity_hiddens = entity_hiddens + tf.scatter_nd(tf.expand_dims(indices, 1), updated_hiddens - prev_states,
                                                        tf.shape(keys))
    return entity_hiddens


@autograph.convert()
def update_entities_False_rl(encoded_sents, seq_len, mask, keys, entity_cell, all_entity_hiddens):
    '''
    update entitites with return_last=False
    '''
    # seq_length = tf.shape(encoded_sents)[1]
    batch_size = tf.shape(encoded_sents)[0]
    key_num = tf.shape(keys)[1]
    entity_embedding_dim = tf.shape(keys)[2]
    indices = tf.zeros([1])
    for i in range(seq_len):
        ''' to see which sentences are available '''
        indices = tf.where(mask[:, i])
        indices = tf.cast(tf.squeeze(indices, axis=1), tf.int32)
        curr_encoded_sents = tf.gather(encoded_sents[:, i, :], indices)
        # print("curr_encoded_sents shape:",curr_encoded_sents.shape)
        curr_keys = tf.gather(keys, indices)
        prev_states = tf.gather(all_entity_hiddens[:, -1, :, :], indices)
        updated_hiddens = tf.expand_dims(entity_cell(curr_encoded_sents, prev_states, curr_keys), axis=1)
        all_entity_hiddens = tf.concat([all_entity_hiddens,
                                        tf.scatter_nd(tf.expand_dims(indices, 1), updated_hiddens,
                                                      [batch_size, 1, key_num, entity_embedding_dim])], axis=1)
    return all_entity_hiddens


def simple_entity_network(inputs, keys, entity_cell=None,
                          initial_entity_hidden_state=None,
                          use_shared_keys=False, return_last=True):
    """
    Description:
        acts as a recurrent network for updating entities for a group of successive sentences given their encoding and initialized keys
    Args:
        entity_cell: the EntityCell
        inputs: a list containing a tensor of shape [batch_size, seq_length, dim] and
                its mask of shape [batch_size, seq_length] and max_sent_num
                batch_size=current paragraphs num, seq_length=max number of senteces
        keys: tensor of shape [batch_size, key_num, dim] if use_shared_keys is False and
                  [key_num, dim] if use_shared_keys is True
        use_shared_keys: if it is True, it use shared keys for all samples.
        mask_inputs: tensor of shape [batch_size, seq_length] and type tf.bool
        initial_entity_hidden_state: a tensor of shape [batch_size, key_num, dim]
        return_last: if it is True, it returns the last state, else returns all states

    Returns:
        if return_last = True then a tensor of shape [batch_size, key_num, dim] (entity_hiddens)
        else of shape [batch_size, seq_length+1 , key_num, dim] it includes initial hidden states as well as states for each step ,total would be seq_len+1
    """

    encoded_sents, mask, seq_len = inputs
    # print("encoded_sents shape:", encoded_sents.shape)

    batch_size = tf.shape(encoded_sents)[0]
    key_num = tf.shape(keys)[1]
    entity_embedding_dim = tf.shape(keys)[2]

    if entity_cell is None:
        entity_cell = EntityCell(max_entity_num=key_num, entity_embedding_dim=entity_embedding_dim,
                                 name='entity_cell')

    if initial_entity_hidden_state is None:
        initial_entity_hidden_state = tf.tile(tf.expand_dims(entity_cell.get_initial_state(), axis=0),
                                              [batch_size, 1, 1])
    if return_last:
        entity_hiddens = initial_entity_hidden_state
        return update_entities_True_rl(encoded_sents=encoded_sents, seq_len=seq_len, mask=mask, keys=keys,
                                       entity_cell=entity_cell,
                                       entity_hiddens=entity_hiddens)
    else:
        # print("return_lastttttttttt:", return_last)
        all_entity_hiddens = tf.expand_dims(initial_entity_hidden_state, axis=1)
        return update_entities_False_rl(encoded_sents=encoded_sents, seq_len=seq_len, mask=mask, keys=keys,
                                        entity_cell=entity_cell,
                                        all_entity_hiddens=all_entity_hiddens)


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


class BasicRecurrentEntityEncoder(tf.keras.Model):
    def __init__(self, embedding_matrix, max_sent_num, max_entity_num=None, entity_embedding_dim=None, entity_cell=None,
                 name=None,
                 **kwargs):

        if name is None:
            name = 'BasicRecurrentEntityEncoder'
        super().__init__(name=name)
        self.max_sent_num = max_sent_num

        if entity_cell is None:
            if entity_embedding_dim is None:
                raise AttributeError('entity_embedding_dim should be given')
            if max_entity_num is None:
                raise AttributeError('max_entity_num should be given')
            entity_cell = EntityCell(max_entity_num=max_entity_num, entity_embedding_dim=entity_embedding_dim,
                                     name='entity_cell')
        self.entity_cell = entity_cell
        self.embedding_matrix = embedding_matrix
        self.sent_encoder_module = Sent_encoder()

    # @property
    # def variables(self):
    #     return self.trainable_variables+self.entity_cell.variables

    @autograph.convert()
    def encode_sents(self, prgrph_embeddings, prgrph_mask, max_sent_num):
        batch_size = tf.shape(prgrph_mask)[0]
        # max_sent_num = prgrph_mask.shape[1]
        encoded_sents = tf.zeros([batch_size, 1, tf.shape(prgrph_embeddings)[3]])
        indices = tf.zeros([1])
        current_sents = tf.zeros([1])
        curr_encoded_sents = tf.zeros([1])
        for i in range(max_sent_num):
            ''' to see which sentences are available '''
            indices = tf.where(prgrph_mask[:, i, 0])
            # print('indices shape encode:, indices.shape)
            indices = tf.cast(tf.squeeze(indices, axis=1), tf.int32)
            current_sents = tf.gather(prgrph_embeddings[:, i, :, :], indices)
            # print('current_sents_call shape:', current_sents.shape)
            curr_encoded_sents = tf.expand_dims(self.sent_encoder_module(current_sents), axis=1)
            encoded_sents = tf.concat([encoded_sents, curr_encoded_sents], axis=1)
        return encoded_sents

    def call(self, inputs, keys, num_inputs=None, initial_entity_hidden_state=None,
             use_shared_keys=False, return_last=True, **kwargs):
        """
        Description:
            encodes sentences of the given paragraph
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
        prgrph_embeddings = tf.nn.embedding_lookup(self.embedding_matrix, prgrph)
        # print("prgrph shape",prgrph.shape)
        'prgrph_embeddings shape: [batch_size, max_sent_num, max_sent_len, embedding_dim]'
        encoded_sents = self.encode_sents(prgrph_embeddings, prgrph_mask, self.max_sent_num)
        encoded_sents = encoded_sents[:, 1:, :]
        sents_mask = prgrph_mask[:, :, 0]
        # print("return_last_encoder", return_last)
        return self.entity_cell, simple_entity_network(entity_cell=self.entity_cell,
                                                       inputs=[encoded_sents, sents_mask, self.max_sent_num],
                                                       keys=keys,
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
    def __init__(self, embedding_matrix, rnn_hidden_size, max_sent_num, max_sent_len, entity_cell=None,
                 entity_embedding_dim=None,
                 max_entity_num=None,
                 rnn_cell=None, vocab_size=None, prgrph_ending_Classifier=None,
                 num_units=None, entity_dense=None, start_hidden_dense=None,
                 softmax_layer=None, name=None, **kwargs):
        if name is None:
            name = 'RNNRecurrentEntitiyDecoder'
        super().__init__(name=name)
        self.embedding_matrix = embedding_matrix
        self.rnn_hidden_size = rnn_hidden_size
        self.max_sent_num = max_sent_num
        self.max_sent_len = max_sent_len
        self.embedding_dim = tf.shape(embedding_matrix)[1]
        if vocab_size is not None:
            self.vocab_size = vocab_size

        if entity_cell is None:
            if entity_embedding_dim is None:
                raise AttributeError("entity_embedding_dim and entity_cell can't be both None")
            if max_entity_num is None:
                raise AttributeError("max_entity_num and entity_cell can't be both None")
            entity_cell = EntityCell(max_entity_num=max_entity_num, entity_embedding_dim=entity_embedding_dim,
                                     name='entity_cell')
        self.entity_cell = entity_cell

        if rnn_cell is None:
            # print("rnn_cell is None")
            # if rnn_hidden_size is None:
            #     raise AttributeError("rnn_hidden_size and rnn_cell can't be both None")
            rnn_cell = tf.keras.layers.LSTM(rnn_hidden_size, return_state=True)
            self.rnn_cell = rnn_cell

        if softmax_layer is None:
            if vocab_size is None:
                raise AttributeError("softmax_layer and vocab_size can't be both None")
            self.softmax_layer = tf.keras.layers.Dense(vocab_size, activation='softmax')

        if prgrph_ending_Classifier is None:
            # if max_sent_num is None:
            #     raise AttributeError("prgrph_ending_Classifier and max_sent_num can't be both None")
            # if rnn_hidden_size is None:
            #     raise AttributeError("prgrph_ending_Classifier and rnn_hidden_size can't be both None")
            if entity_embedding_dim is None:
                raise AttributeError("prgrph_ending_Classifier and entity_embedding_dim can't be both None")
            prgrph_ending_Classifier = prgrph_ending_classifier.Prgrph_ending_classifier(
                max_sent_num=max_sent_num,
                encoding_dim=rnn_hidden_size,
                entity_embedding_dim=entity_embedding_dim)
        self.prgrph_ending_classifier = prgrph_ending_Classifier
        # print("self.prgrph_ending_classifier",self.prgrph_ending_classifier)

        'entity_dense: from entity_embedding_dim to hidden_size, which is used after attending on entities'
        if entity_dense is None:
            entity_dense = tf.keras.layers.Dense(rnn_hidden_size)
        self.entity_dense = entity_dense

        if start_hidden_dense is None:
            start_hidden_dense = tf.keras.layers.Dense(rnn_hidden_size)
        self.start_hidden_dense = start_hidden_dense

        self.sent_encoder_module = Sent_encoder()

    def build(self, input_shape):
        # print("in build decoder")
        self.entity_attn_matrix = K.random_normal_variable(shape=[self.rnn_hidden_size, self.embedding_dim],
                                                           mean=0, scale=0.05, name='entity_attn_matrix')

    def attention_hiddens(self, query, keys, memory_mask):
        '''
        Description:
            attention on keys with given quey, value is equal to keys

        Args:tf.shape(prgrph)[0]
            inputs: query shape: [curr_prgrphs_num, hiddens_size]
                    keys shape: [curr_prgrphs_num, prev_hiddens_num, hidden_size]
                    memory_mask: [curr_prgrphs_num, prev_hiddens_num]
            output shape: [curr_prgrphs_num, hidden_size]
        '''
        # print('in attention_hiddens')
        # print('keys shape:', keys.shape)
        # print('query shape:', query.shape)
        # print('mask shape', memory_mask.shape)
        values = tf.identity(keys)
        query_shape = tf.shape(query)
        keys_shape = tf.shape(keys)
        batch_size = query_shape[0]
        seq_length = keys_shape[1]
        indices = tf.where(memory_mask)
        queries = tf.gather(query, indices[:, 0])
        keys = tf.boolean_mask(keys, memory_mask)
        attention_logits = tf.reduce_sum(tf.multiply(queries, keys), axis=-1)
        # print('attention logits:',attention_logits)
        # print('tf.where(memory_mask):',tf.where(memory_mask))
        attention_logits = tf.scatter_nd(tf.where(memory_mask), attention_logits, [batch_size, seq_length])
        attention_logits = tf.where(memory_mask, attention_logits, tf.fill([batch_size, seq_length], -float("Inf")))
        attention_coefficients = tf.nn.softmax(attention_logits, axis=1)
        attention = tf.expand_dims(attention_coefficients, -1) * values

        # print(tf.reduce_sum(attention,1))

        return tf.reduce_sum(attention, 1)

    def attention_entities(self, query, entities, keys_mask):
        '''
        Description:
            attention on entities

        Arges:
            inputs: query shape: [curr_prgrphs_num, rnn_hidden_size]
                    entities shape: [curr_prgrphs_num, entities_num, entitiy_embedding_dim]
                    keys_mask shape: [curr_prgrphs_num, entities_num]
            output shape: [curr_prgrphs_num, entity_embedding_dim]
        '''

        # print("attention_entities, entities shape: ", entities.shape)

        values = tf.identity(entities)
        query_shape = tf.shape(query)
        entities_shape = tf.shape(entities)
        batch_size = query_shape[0]
        seq_length = entities_shape[1]
        indices = tf.where(keys_mask)
        queries = tf.gather(query, indices[:, 0])
        entities = tf.boolean_mask(entities, keys_mask)
        # print(queries.shape)
        # print(self.entity_attn_matrix.shape)
        attention_logits = tf.reduce_sum(tf.multiply(tf.matmul(queries, self.entity_attn_matrix), entities), axis=-1)
        # print('attention logits:',attention_logits)
        # print('tf.where(memory_mask):',tf.where(memory_mask))
        attention_logits = tf.scatter_nd(tf.where(keys_mask), attention_logits, [batch_size, seq_length])
        attention_logits = tf.where(keys_mask, attention_logits, tf.fill([batch_size, seq_length], -float("Inf")))
        attention_coefficients = tf.nn.softmax(attention_logits, axis=1)
        attention = tf.expand_dims(attention_coefficients, -1) * values

        return tf.reduce_sum(tf.multiply(tf.expand_dims(tf.matmul(query, self.entity_attn_matrix), axis=1), entities),
                             axis=1)

    # @autograph.convert()
    def calculate_hidden(self, curr_sents_prev_hiddens, entities, hiddens_mask, keys_mask):
        """
        Description:
            calculates current hidden state that should be fed to lstm for predicting the next word, with attention on previous hidden states, THEN entities

        Args:
            inputs: curr_sents_prev_hiddens shape: [curr_prgrphs_num, prev_hiddens_num, hidden_size]
                    entities: [curr_prgrphs_num, entities_num, entity_embedding_dim]
                    mask: [curr_prgrphs_num, prev_hiddens_num]
            output shape: [curr_prgrphs_num, hidden_size]

        """

        """
        attention on hidden states:
            query: last column (last hidden_state)
            key and value: prev_columns
        """
        attn_hiddens_output = self.attention_hiddens(
            curr_sents_prev_hiddens[:, tf.shape(curr_sents_prev_hiddens)[1] - 1, :],
            curr_sents_prev_hiddens[:, :tf.shape(curr_sents_prev_hiddens)[1], :], hiddens_mask)
        attn_entities_output = self.attention_entities(attn_hiddens_output, entities, keys_mask)
        return self.entity_dense(attn_entities_output)

    def get_embeddings(self, input):
        return tf.nn.embedding_lookup(self.embedding_matrix, input)

    @autograph.convert()
    def language_model_True_rl(self, prgrph, prgrph_mask, prgrph_embeddings, entity_hiddens, start_token, vocab_size,
                               keys,
                               keys_mask, use_shared_keys):
        # print("in language_mtf.equal(j, 0)odel_True_rl")
        batch_size = tf.shape(prgrph)[0]
        # max_sent_num = tf.shape(prgrph)[1]
        # max_sent_len = tf.shape(prgrph)[2]
        # prgrph_embeddings = self.get_embeddings(prgrph)

        final_output = tf.zeros(shape=[batch_size, self.max_sent_num, self.max_sent_len, vocab_size], dtype=tf.float32)
        final_targets = tf.zeros(shape=[batch_size, self.max_sent_num, self.max_sent_len], dtype=tf.int32)
        output_mask = tf.zeros(shape=[batch_size, self.max_sent_num, self.max_sent_len], dtype=tf.int32)

        ' stores previous hidden_states of the lstm for the prgrph '
        hidden_states = tf.zeros([batch_size, self.max_sent_num * self.max_sent_len, self.rnn_hidden_size])
        hiddens_mask = tf.reshape(prgrph_mask, [batch_size, -1])
        indices = tf.zeros([1])
        # lstm_inputs = tf.zeros([1])
        lstm_targets = tf.zeros([1])
        # curr_sents_curr_hidden = tf.zeros([1])
        curr_sents_prev_hiddens = tf.zeros([1])
        curr_sents_prev_hiddens_mask = tf.zeros([1])
        prev_states = tf.zeros([1])
        curr_sents_cell_state = tf.zeros([1])
        # output = tf.zeros([1])
        # next_hidden = tf.zeros([1])
        # next_cell_state = tf.zeros([1])
        curr_cells_prev_state = tf.zeros([1])
        index_vector = tf.zeros([1])
        new_indices = tf.zeros([1])
        new_indices_output = tf.zeros([1])
        second_dim_ind = tf.zeros([1])
        indices_2=tf.zeros([1])
        third_dim_ind = tf.zeros([1])
        t = tf.zeros([1])
        encoded_sents = tf.zeros([1])
        sents_mask = tf.zeros([1])

        # cell_states = tf.zeros([1])

        lstm_inputs = tf.expand_dims(tf.tile(tf.expand_dims(self.embedding_matrix[start_token], axis=0),
                                             [batch_size, 1]), 1)
        curr_sents_curr_hidden = tf.zeros([batch_size, self.rnn_hidden_size], tf.float32)
        cell_states = tf.zeros([batch_size, self.rnn_hidden_size], tf.float32)
        output, next_hidden, next_cell_state = self.rnn_cell(lstm_inputs, initial_state=[curr_sents_curr_hidden,
                                                                                         cell_states])

        for i in range(self.max_sent_num):
            # print('p2_mask',p2_mask)
            current_sents_indices = tf.where(prgrph_mask[:, i, 0])
            current_sents_indices = tf.squeeze(current_sents_indices, axis=1)
            for j in range(self.max_sent_len):
                # print('current word indeX:', i, j)
                ' indices of available paragraphs'
                # print('prgrph_mask.shape',tf.where(prgrph_mask).shape)
                # print('prgprh_mask[:,i,j]',prgrph_mask[:,i,j])
                indices = tf.cast(tf.where(prgrph_mask[:, i, j]), dtype=tf.int32)
                # print('indices shape:', indices.shape)
                indices = tf.squeeze(indices, axis=1)
                # e=tf.shape(indices)[0]
                # print("e",e)
                if tf.shape(indices)[0] > 0:
                    # print("tf.shape(indices)>0")
                    # lstm_inputs=tf.zeros([1])
                    # curr_sents_curr_hidden=tf.zeros([1])
                    # curr_sents_prev_hiddens=tf.zeros([1])
                    # print('indices_p2_mask:',indices)   #indices_p2_mask: tf.Tensor([[0]], shape=(1, 1), dtype=int64)
                    if j == 0:
                        # print('start token',self.embedding_matrix[self.start_token].shape)
                        lstm_inputs = tf.tile(tf.expand_dims(self.embedding_matrix[start_token], axis=0),
                                              [batch_size, 1])
                        # print('lstm_first_inputs:', lstm_inputs.shape)
                    else:
                        # print('prgrph_embedding_shape:',prgrph_embeddings[:, i, j - 1, :].shape)
                        lstm_inputs = tf.gather(prgrph_embeddings[:, i, j - 1, :], indices)

                    # print(tf.gather(second_prgrph[:, i, j], indices).shape)
                    lstm_targets = tf.gather(prgrph[:, i, j], indices)
                    # print("i*max_sent_len+j",i*max_sent_len+j)
                    t = i * self.max_sent_len + j
                    # "if t==0 didn't work! t is a tensor with shape zero"
                    if t == 0:
                        # print("hey!")
                        curr_sents_curr_hidden = self.start_hidden_dense(tf.reduce_sum(entity_hiddens, axis=1))
                        cell_states = tf.zeros([batch_size, self.rnn_hidden_size], tf.float32)
                        # print('curr_sents_cell_state.shape', cell_states.shape)
                    else:
                        curr_sents_prev_hiddens = tf.gather(hidden_states[:, :i * self.max_sent_len + j, :], indices)
                        curr_sents_prev_hiddens_mask = tf.gather(hiddens_mask[:, :i * self.max_sent_len + j], indices)
                        # curr_sents_entities = tf.gather(self.update_entity_module.hiddens, indices)
                        # if return_last:
                        prev_states = tf.gather(entity_hiddens, indices)
                        # print("entity_hiddens shape:", entity_hiddens.shape)
                        # print('return last=', return_last, 'prev_states shape:', prev_states.shape)
                        # else:
                        #     prev_states = tf.gather(all_entity_hiddens[:, -1, :, :], indices)
                        # print("i:", i, "j:", j)
                        curr_sents_curr_hidden = self.calculate_hidden(curr_sents_prev_hiddens, prev_states,
                                                                       hiddens_mask=curr_sents_prev_hiddens_mask,
                                                                       keys_mask=tf.gather(keys_mask, indices))
                    curr_sents_cell_state = tf.gather(cell_states, indices)
                    # print('lstm_inputs shape:', lstm_inputs.shape)
                    # tf.print("curr_sents_curr_hidden",curr_sents_curr_hidden.eval())
                    output, next_hidden, next_cell_state = self.rnn_cell(tf.expand_dims(lstm_inputs, axis=1),
                                                                         initial_state=[curr_sents_curr_hidden,
                                                                                        curr_sents_cell_state])

                    'updating cell_states'
                    curr_cells_prev_state = tf.gather(cell_states, indices)
                    cell_states = cell_states + tf.scatter_nd(tf.expand_dims(indices, axis=1),
                                                              next_cell_state - curr_cells_prev_state,
                                                              [batch_size, self.rnn_hidden_size])
                    # print('next_hidden shape:', next_hidden.shape)
                    'output shape:[len(indices), hidden_size] here, output is equal to next_hidden'
                    index_vector = tf.ones([tf.shape(indices)[0], 1], tf.int32) * (i * self.max_sent_len + j)
                    # print('indices type:', indices.dtype)
                    new_indices = tf.concat(values=[tf.expand_dims(indices, 1), index_vector], axis=1)
                    # print('new_indices:', new_indices)
                    hidden_states = tf.add(hidden_states, tf.scatter_nd(new_indices, next_hidden,
                                                                        shape=[batch_size, tf.shape(hidden_states)[1],
                                                                               self.rnn_hidden_size]))
                    # print('hidden_state.shape', hidden_states.shape)
                    # print('hidden_states:',hidden_states)
                    output = self.softmax_layer(output)
                    # print('ouput_shape', output.shape)

                    second_dim_ind = tf.ones([tf.shape(indices)[0], 1], tf.int32) * i
                    third_dim_ind = tf.ones([tf.shape(indices)[0], 1], tf.int32) * j

                    indices_2=tf.multiply(tf.ones([tf.shape(indices)[0],1],tf.int32),tf.expand_dims(indices,1))
                    new_indices_output = tf.keras.layers.concatenate(
                        inputs=[indices_2, second_dim_ind, third_dim_ind], axis=1)
                    final_output = tf.add(final_output, tf.scatter_nd(new_indices_output, output,
                                                                      shape=[batch_size, self.max_sent_num,
                                                                             self.max_sent_len,
                                                                             vocab_size]))

                    # print('final_targets dtype', final_targets.dtype)
                    # print(tf.scatter_nd(new_indices_output, lstm_targets,
                    #                     shape=[batch_size, max_sent_num, max_sent_len]).dtype)
                    final_targets = tf.add(final_targets, tf.cast(
                        tf.scatter_nd(new_indices_output, lstm_targets,
                                      shape=[batch_size, self.max_sent_num, self.max_sent_len]),
                        dtype=tf.int32))

                    t = tf.ones([tf.shape(indices)[0]], dtype=tf.int32)
                    output_mask = tf.add(output_mask, tf.scatter_nd(new_indices_output, t,
                                                                    shape=[batch_size, self.max_sent_num,
                                                                           self.max_sent_len]))

            encoded_sents = tf.expand_dims(self.sent_encoder_module(prgrph_embeddings[:, i, :, :]), axis=1)
            sents_mask = tf.expand_dims(prgrph_mask[:, i, 0], axis=1)
            entity_hiddens = simple_entity_network(entity_cell=self.entity_cell, inputs=[encoded_sents, sents_mask, 1],
                                                   keys=keys, initial_entity_hidden_state=entity_hiddens,
                                                   use_shared_keys=use_shared_keys, return_last=True)
        return final_targets, final_output, output_mask, entity_hiddens

    @autograph.convert()
    def language_model_False_rl(self, prgrph, prgrph_mask, prgrph_embeddings, entity_hiddens, start_token, vocab_size,
                                keys,
                                keys_mask, use_shared_keys):
        batch_size = tf.shape(prgrph)[0]
        max_sent_num = tf.shape(prgrph)[1]
        max_sent_len = tf.shape(prgrph)[2]
        # prgrph_embeddings = self.get_embeddings(prgrph)

        final_output = tf.zeros(shape=[batch_size, max_sent_num, max_sent_len, vocab_size], dtype=tf.float32)
        final_targets = tf.zeros(shape=[batch_size, max_sent_num, max_sent_len], dtype=tf.int32)
        output_mask = tf.zeros(shape=[batch_size, max_sent_num, max_sent_len], dtype=tf.int32)

        ' stores previous hidden_states of the lstm for the prgrph '
        hidden_states = tf.zeros([batch_size, max_sent_num * max_sent_len, self.rnn_hidden_size])
        hiddens_mask = tf.reshape(prgrph_mask, [batch_size, -1])

        all_entity_hiddens = tf.expand_dims(entity_hiddens, axis=1)

        cell_states = tf.zeros([1])

        indices = tf.zeros([1])
        lstm_inputs = tf.zeros([1])
        lstm_targets = tf.zeros([1])
        curr_sents_curr_hidden = tf.zeros([1])
        curr_sents_prev_hiddens = tf.zeros([1])
        curr_sents_prev_hiddens_mask = tf.zeros([1])
        prev_states = tf.zeros([1])
        curr_sents_cell_state = tf.zeros([1])
        output = tf.zeros([1])
        next_hidden = tf.zeros([1])
        next_cell_state = tf.zeros([1])
        curr_cells_prev_state = tf.zeros([1])
        index_vector = tf.zeros([1])
        new_indices = tf.zeros([1])
        new_indices_output = tf.zeros([1])
        second_dim_ind = tf.zeros([1])
        third_dim_ind = tf.zeros([1])
        t = tf.zeros([1])
        encoded_sents = tf.zeros([1])
        sents_mask = tf.zeros([1])

        for i in range(max_sent_num):
            # print('p2_mask',p2_mask)
            current_sents_indices = tf.where(prgrph_mask[:, i, 0])
            current_sents_indices = tf.squeeze(current_sents_indices, axis=1)
            for j in range(max_sent_len):
                # print('current word indeX:', i, j)
                ' indices of available paragraphs'
                # print('prgrph_mask.shape',tf.where(prgrph_mask).shape)
                # print('prgprh_mask[:,i,j]',prgrph_mask[:,i,j])
                indices = tf.cast(tf.where(prgrph_mask[:, i, j]), dtype=tf.int32)
                # print('indices shape:', indices.shape)
                indices = tf.squeeze(indices, axis=1)
                if tf.shape(indices)[0] > 0:
                    # print('indices_p2_mask:',indices)   #indices_p2_mask: tf.Tensor([[0]], shape=(1, 1), dtype=int64)
                    if j == 0:
                        # print('start token',self.embedding_matrix[self.start_token].shape)
                        lstm_inputs = tf.tile(tf.expand_dims(self.embedding_matrix[start_token], axis=0),
                                              [batch_size, 1])
                        # print('lstm_first_inputs:', lstm_inputs.shape)
                    else:
                        # print('prgrph_embedding_shape:',prgrph_embeddings[:, i, j - 1, :].shape)
                        lstm_inputs = tf.gather(prgrph_embeddings[:, i, j - 1, :], indices)

                    # print(tf.gather(second_prgrph[:, i, j], indices).shape)
                    lstm_targets = tf.gather(prgrph[:, i, j], indices)
                    # print("i*max_sent_len+j",i*max_sent_len+j)
                    t = i * max_sent_len + j
                    "if t==0 didn't work! t is a tensor with shape zero"
                    if tf.equal(t, 0):
                        # print("hey!")
                        curr_sents_curr_hidden = self.start_hidden_dense(tf.reduce_sum(entity_hiddens, axis=1))
                        cell_states = tf.zeros([batch_size, self.rnn_hidden_size], tf.float32)
                        # print('curr_sents_cell_state.shape', cell_states.shape)
                    else:
                        curr_sents_prev_hiddens = tf.gather(hidden_states[:, :i * max_sent_len + j, :], indices)
                        curr_sents_prev_hiddens_mask = tf.gather(hiddens_mask[:, :i * max_sent_len + j], indices)
                        # curr_sents_entities = tf.gather(self.update_entity_module.hiddens, indices)
                        # if return_last:
                        #     prev_states = tf.gather(entity_hiddens, indices)
                        #     print("entity_hiddens shape:", entity_hiddens.shape)
                        #     print('return last=', return_last, 'prev_states shape:', prev_states.shape)
                        # else:
                        prev_states = tf.gather(all_entity_hiddens[:, -1, :, :], indices)
                        # print("i:", i, "j:", j)
                        curr_sents_curr_hidden = self.calculate_hidden(curr_sents_prev_hiddens, prev_states,
                                                                       hiddens_mask=curr_sents_prev_hiddens_mask,
                                                                       keys_mask=tf.gather(keys_mask, indices))
                    curr_sents_cell_state = tf.gather(cell_states, indices)
                    # print('lstm_inputs shape:', lstm_inputs.shape)
                    output, next_hidden, next_cell_state = self.rnn_cell(tf.expand_dims(lstm_inputs, axis=1),
                                                                         initial_state=[
                                                                             curr_sents_curr_hidden,
                                                                             curr_sents_cell_state])

                    'updating cell_states'
                    curr_cells_prev_state = tf.gather(cell_states, indices)
                    cell_states = tf.add(cell_states, tf.scatter_nd(tf.expand_dims(indices, axis=1),
                                                                    next_cell_state - curr_cells_prev_state,
                                                                    [batch_size, self.rnn_hidden_size]))
                    # print('next_hidden shape:', next_hidden.shape)
                    'output shape:[len(indices), hidden_size] here, output is equal to next_hidden'
                    index_vector = tf.ones([tf.shape(indices)[0], 1], tf.int32) * (i * max_sent_len + j)
                    # print('indices type:', indices.dtype)
                    new_indices = tf.concat(values=[tf.expand_dims(indices, 1), index_vector], axis=1)
                    # print('new_indices:', new_indices)
                    hidden_states = tf.add(hidden_states, tf.scatter_nd(new_indices, next_hidden,
                                                                        shape=[batch_size, tf.shape(hidden_states)[1],
                                                                               self.rnn_hidden_size]))
                    # print('hidden_state.shape', hidden_states.shape)
                    # print('hidden_states:',hidden_states)
                    output = self.softmax_layer(output)
                    # print('ouput_shape', output.shape)

                    second_dim_ind = tf.ones([tf.shape(indices)[0], 1], tf.int32) * i
                    third_dim_ind = tf.ones([tf.shape(indices)[0], 1], tf.int32) * j
                    new_indices_output = tf.keras.layers.concatenate(
                        inputs=[tf.expand_dims(indices, 1), second_dim_ind, third_dim_ind], axis=1)
                    final_output = tf.add(final_output, tf.scatter_nd(new_indices_output, output,
                                                                      shape=[batch_size, max_sent_num, max_sent_len,
                                                                             vocab_size]))

                    # print('final_targets dtype', final_targets.dtype)
                    t = tf.scatter_nd(new_indices_output, lstm_targets,
                                      shape=[batch_size, max_sent_num, max_sent_len]).dtype
                    # print(t)
                    final_targets = final_targets + tf.cast(
                        tf.scatter_nd(new_indices_output, lstm_targets, shape=[batch_size, max_sent_num, max_sent_len]),
                        dtype=tf.int32)

                    t = tf.ones([tf.shape(indices)[0]], dtype=tf.int32)
                    output_mask = output_mask + tf.scatter_nd(new_indices_output, t,
                                                              shape=[batch_size, max_sent_num, max_sent_len])

            encoded_sents = tf.expand_dims(self.sent_encoder_module(prgrph_embeddings[:, i, :, :]), axis=1)
            sents_mask = tf.expand_dims(prgrph_mask[:, i, 0], axis=1)
            new_entity_hiddens = simple_entity_network(entity_cell=self.entity_cell,
                                                       inputs=[encoded_sents, sents_mask, self.max_sent_num],
                                                       keys=keys,
                                                       initial_entity_hidden_state=all_entity_hiddens[:, -1, :, :],
                                                       use_shared_keys=use_shared_keys, return_last=True)
            all_entity_hiddens = tf.concat([all_entity_hiddens, tf.expand_dims(new_entity_hiddens, axis=1)], axis=1)

        return final_targets, final_output, output_mask, all_entity_hiddens

    @autograph.convert()
    def train_one_step(self, prgrph, prgrph_mask, entity_hiddens, return_last, start_token, vocab_size,
                       keys_mask, use_shared_keys):
        batch_size = tf.shape(prgrph)[0]
        max_sent_num = tf.shape(prgrph)[1]
        max_sent_len = tf.shape(prgrph)[2]
        prgrph_embeddings = self.get_embeddings(prgrph)

        final_output = tf.zeros(shape=[batch_size, max_sent_num, max_sent_len, vocab_size], dtype=tf.float32)
        final_targets = tf.zeros(shape=[batch_size, max_sent_num, max_sent_len], dtype=tf.int32)
        output_mask = tf.zeros(shape=[batch_size, max_sent_num, max_sent_len], dtype=tf.int32)

        ' stores previous hidden_states of the lstm for the prgrph '
        hidden_states = tf.zeros([batch_size, max_sent_num * max_sent_len, self.rnn_hidden_size])
        hiddens_mask = tf.reshape(prgrph_mask, [batch_size, -1])

        cell_states = tf.zeros([1])
        for i in range(1):
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
                if tf.shape(indices)[0] > 0:
                    # print('indices_p2_mask:',indices)   #indices_p2_mask: tf.Tensor([[0]], shape=(1, 1), dtype=int64)
                    if j == 0:
                        # print('start token',self.embedding_matrix[self.start_token].shape)
                        lstm_inputs = tf.tile(tf.expand_dims(self.embedding_matrix[start_token], axis=0),
                                              [batch_size, 1])
                        print('lstm_first_inputs:', lstm_inputs.shape)
                    else:
                        # print('prgrph_embedding_shape:',prgrph_embeddings[:, i, j - 1, :].shape)
                        lstm_inputs = tf.gather(prgrph_embeddings[:, i, j - 1, :], indices)

                    # print(tf.gather(second_prgrph[:, i, j], indices).shape)
                    lstm_targets = tf.gather(prgrph[:, i, j], indices)
                    # print("i*max_sent_len+j",i*max_sent_len+j)
                    t = i * max_sent_len + j
                    "if t==0 didn't work! t is a tensor with shape zero"
                    if tf.equal(t, 0):
                        # print("hey!")
                        curr_sents_curr_hidden = self.start_hidden_dense(tf.reduce_sum(entity_hiddens, axis=1))
                        cell_states = tf.zeros([batch_size, self.rnn_hidden_size], tf.float32)
                        print('curr_sents_cell_state.shape', cell_states.shape)
                    else:
                        curr_sents_prev_hiddens = tf.gather(hidden_states[:, :i * max_sent_len + j, :], indices)
                        curr_sents_prev_hiddens_mask = tf.gather(hiddens_mask[:, :i * max_sent_len + j], indices)
                        # curr_sents_entities = tf.gather(self.update_entity_module.hiddens, indices)
                        # if return_last:
                        prev_states = tf.gather(entity_hiddens, indices)
                        print("entity_hiddens shape:", entity_hiddens.shape)
                        print('return last=', return_last, 'prev_states shape:', prev_states.shape)
                        # else:
                        #     prev_states = tf.gather(all_entity_hiddens[:, -1, :, :], indices)
                        print("i:", i, "j:", j)
                        curr_sents_curr_hidden = self.calculate_hidden(curr_sents_prev_hiddens, prev_states,
                                                                       hiddens_mask=curr_sents_prev_hiddens_mask,
                                                                       keys_mask=tf.gather(keys_mask, indices))
                    curr_sents_cell_state = tf.gather(cell_states, indices)
                    print('lstm_inputs shape:', lstm_inputs.shape)
                    output, next_hidden, next_cell_state = self.rnn_cell(tf.expand_dims(lstm_inputs, axis=1),
                                                                         initial_state=[
                                                                             curr_sents_curr_hidden,
                                                                             curr_sents_cell_state])

                    'updating cell_states'
                    curr_cells_prev_state = tf.gather(cell_states, indices)
                    cell_states = cell_states + tf.scatter_nd(tf.expand_dims(indices, axis=1),
                                                              next_cell_state - curr_cells_prev_state,
                                                              [batch_size, self.rnn_hidden_size])
                    print('next_hidden shape:', next_hidden.shape)
                    'output shape:[len(indices), hidden_size] here, output is equal to next_hidden'
                    index_vector = tf.ones([tf.shape(indices)[0], 1], tf.int32) * (i * max_sent_len + j)
                    print('indices type:', indices.dtype)
                    new_indices = tf.concat(values=[tf.expand_dims(indices, 1), index_vector], axis=1)
                    print('new_indices:', new_indices)
                    hidden_states = hidden_states + tf.scatter_nd(new_indices, next_hidden,
                                                                  shape=[batch_size, tf.shape(hidden_states)[1],
                                                                         self.rnn_hidden_size])
                    print('hidden_state.shape', hidden_states.shape)
                    # print('hidden_states:',hidden_states)
                    output = self.softmax_layer(output)
                    print('ouput_shape', output.shape)

                    second_dim_ind = tf.ones([tf.shape(indices)[0], 1], tf.int32) * i
                    third_dim_ind = tf.ones([tf.shape(indices)[0], 1], tf.int32) * j
                    new_indices_output = tf.keras.layers.concatenate(
                        inputs=[tf.expand_dims(indices, 1), second_dim_ind, third_dim_ind], axis=1)
                    final_output = final_output + tf.scatter_nd(new_indices_output, output,
                                                                shape=[batch_size, max_sent_num, max_sent_len,
                                                                       vocab_size])

                    print('final_targets dtype', final_targets.dtype)
                    print(tf.scatter_nd(new_indices_output, lstm_targets,
                                        shape=[batch_size, max_sent_num, max_sent_len]).dtype)
                    final_targets = final_targets + tf.cast(
                        tf.scatter_nd(new_indices_output, lstm_targets,
                                      shape=[batch_size, max_sent_num, max_sent_len]),
                        dtype=tf.int32)

                    t = tf.ones([tf.shape(indices)[0]], dtype=tf.int32)
                    output_mask = output_mask + tf.scatter_nd(new_indices_output, t,
                                                              shape=[batch_size, max_sent_num, max_sent_len])

    def decode_train(self, inputs, labels, keys, keys_mask, encoder_hidden_states=None,
                     initial_hidden_state=None, use_shared_keys=False,
                     return_last=True, attention=False, self_attention=False):
        '''
                   Language model on given paragraph (second prgrph)
        '''

        """
        inputs: [test_mode_bool, entity_hiddens, vocab_size, start_token]
        labels: [prgrph, prgrph_mask]
        
        return: guessed words as final_output and their mask as output_mask and ground truth targets
        """

        if len(inputs) != 4:
            raise AttributeError('expected 4 inputs but', len(inputs), 'were given')
        if len(labels) != 2:
            raise AttributeError('expected 2 labels but', len(labels), 'were given')

        print('IN DECODE_TRAIN')

        test_mode_bool, entity_hiddens, vocab_size, start_token = inputs
        # print(entity_hiddens.shape)
        prgrph, prgrph_mask = labels
        batch_size = tf.shape(prgrph)[0]
        prgrph_embeddings = self.get_embeddings(prgrph)
        # max_sent_num = tf.shape(prgrph)[1]
        # max_sent_len = tf.shape(prgrph)[2]

        if test_mode_bool:
            self.train_one_step(prgrph=prgrph, prgrph_mask=prgrph_mask, entity_hiddens=entity_hiddens,
                                return_last=return_last, start_token=start_token, vocab_size=vocab_size,
                                keys_mask=keys_mask,
                                use_shared_keys=use_shared_keys)
            return tf.zeros([1])
        else:
            if return_last:
                final_targets, final_output, output_mask, p2_entity_hiddens = self.language_model_True_rl(prgrph=prgrph,
                                                                                                          prgrph_mask=prgrph_mask,
                                                                                                          prgrph_embeddings=prgrph_embeddings,
                                                                                                          entity_hiddens=entity_hiddens,
                                                                                                          start_token=start_token,
                                                                                                          vocab_size=vocab_size,
                                                                                                          keys=keys,
                                                                                                          keys_mask=keys_mask,
                                                                                                          use_shared_keys=use_shared_keys)
                # print("succeed!")
            else:
                final_targets, final_output, output_mask, p2_entity_hiddens = self.language_model_False_rl(
                    prgrph=prgrph,
                    prgrph_mask=prgrph_mask,
                    prgrph_embeddings=prgrph_embeddings,
                    entity_hiddens=entity_hiddens,
                    start_token=start_token,
                    vocab_size=vocab_size, keys=keys,
                    keys_mask=keys_mask,
                    use_shared_keys=use_shared_keys)

            final_output = tf.reshape(final_output, [batch_size, -1, vocab_size])
            final_targets = tf.reshape(final_targets, [batch_size, -1])
            output_mask = tf.reshape(output_mask, [batch_size, -1])
            return final_output, final_targets, output_mask, p2_entity_hiddens

    # @autograph.convert()
    'convert to tensorflow graph code'

    def decode_test(self, inputs, keys, keys_mask, encoder_hidden_states=None,
                    initial_hidden_state=None, use_shared_keys=False,
                    return_last=True, attention=False, self_attention=False):
        '''
            predicting second paragraph
        '''

        ''' 
        inputs: entity_hiddens last state
        # yields predicted output of shape [batch_size, vocab_size] each step
        '''

        if len(inputs) != 5:
            raise AttributeError('expected 5 inputs but', len(inputs), 'were given')

        entity_hiddens, max_sent_num, max_sent_len, eos_ind, start_token = inputs

        batch_size = tf.shape(entity_hiddens)[0]

        ' stores previous hidden_states of the lstm for the prgrph '
        hidden_states = tf.zeros([batch_size, max_sent_num * max_sent_len, self.rnn_hidden_size])

        'new masks will be added each step'
        hiddens_mask = tf.equal(tf.ones([batch_size, 1]), 1)

        'indices of generated words'
        generated_prgrphs = tf.zeros([batch_size, max_sent_num, max_sent_len], dtype=tf.int32)
        generated_prgrphs_embeddings = self.get_embeddings(generated_prgrphs)

        last_noneos_output = tf.zeros([1], dtype=tf.float32)
        ' indices of available paragraphs'
        unfinished_prgrphs_indices = tf.range(start=0, limit=batch_size, dtype=tf.int32)
        ending_hidden_indices = tf.zeros([batch_size, max_sent_num], dtype=tf.int32)

        if return_last == False:
            all_entity_hiddens = tf.expand_dims(entity_hiddens, axis=1)

        cell_states = tf.zeros([1])

        for i in range(max_sent_num):
            indices = tf.identity(unfinished_prgrphs_indices)
            if tf.shape(indices)[0] > 0:
                for j in range(max_sent_len):
                    print('current word indeX:', i, j)
                    print("GENERATED PARAGRAPHS")
                    print(generated_prgrphs)
                    if j == 0:
                        # print('start token',self.embedding_matrix[self.start_token].shape)
                        lstm_inputs = tf.tile(tf.expand_dims(self.embedding_matrix[start_token], axis=0),
                                              [batch_size, 1])
                    else:
                        lstm_inputs = tf.gather(generated_prgrphs_embeddings[:, i, j - 1, :], indices)
                    t = i * max_sent_len + j
                    if tf.equal(t, 0):
                        # print(tf.reduce_sum(entity_hiddens,axis=1))

                        if initial_hidden_state is None:
                            curr_sents_curr_hidden = self.start_hidden_dense(
                                tf.reduce_sum(tf.multiply(entity_hiddens,
                                                          tf.expand_dims(tf.cast(keys_mask, dtype=tf.float32), axis=2)),
                                              axis=1))
                        else:
                            curr_sents_curr_hidden = initial_hidden_state
                        cell_states = tf.zeros([batch_size, self.rnn_hidden_size], tf.float32)
                    else:
                        print('hidden_states shape:', tf.shape(hidden_states))
                        curr_sents_prev_hiddens = tf.gather(hidden_states[:, :i * max_sent_len + j, :], indices)
                        print('curr_sents_prev_hiddens shape:', tf.shape(curr_sents_prev_hiddens), tf.shape(indices))
                        curr_sents_prev_hiddens_mask = tf.gather(hiddens_mask[:, 1:], indices)
                        # curr_sents_entities = tf.gather(self.update_entity_module.hiddens, indices)
                        if return_last:
                            prev_states = tf.gather(entity_hiddens, indices)
                        else:
                            prev_states = tf.gather(all_entity_hiddens[:, -1, :, :], indices)
                        curr_sents_curr_hidden = self.calculate_hidden(curr_sents_prev_hiddens,
                                                                       prev_states,
                                                                       hiddens_mask=curr_sents_prev_hiddens_mask,
                                                                       keys_mask=tf.gather(keys_mask, indices))

                    curr_sents_cell_state = tf.gather(cell_states, indices)
                    lstm_output, next_hidden, next_cell_state = self.rnn_cell(tf.expand_dims(lstm_inputs, axis=1),
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
                                                              [batch_size, self.rnn_hidden_size])

                    print('next_hidden shape:', tf.shape(next_hidden))
                    'output shape:[available_prgrphs_num, hidden_size] here, output is equal to next_hidden'
                    index_vector = tf.ones([tf.shape(indices)[0], 1], tf.int32) * (i * max_sent_len + j)
                    new_indices = tf.keras.layers.concatenate(inputs=[tf.expand_dims(indices, 1), index_vector],
                                                              axis=1)
                    print('new_indices:', new_indices)
                    hidden_states = hidden_states + tf.scatter_nd(new_indices, next_hidden,
                                                                  shape=[batch_size, tf.shape(hidden_states)[1],
                                                                         self.rnn_hidden_size])
                    'constructing next_hidden mask and concatenating to hiddens_mask'
                    boolean_vec = tf.ones(tf.shape(indices)[0], dtype=tf.bool)
                    next_hidden_mask = tf.scatter_nd(tf.expand_dims(indices, axis=1),
                                                     tf.expand_dims(boolean_vec, axis=1),
                                                     shape=[batch_size, 1])
                    hiddens_mask = tf.keras.layers.concatenate(inputs=[hiddens_mask, next_hidden_mask], axis=1)

                    # print('hidden_states:',hidden_states)
                    lstm_output = self.softmax_layer(lstm_output)
                    last_output = tf.cast(tf.argmax(lstm_output, dimension=1), tf.int32)

                    if tf.equal(max_sent_len - 1, j):
                        last_output = tf.cast(tf.ones([tf.shape(indices)[0]]) * eos_ind, dtype=tf.int32)
                    'last_output is a one_dimensional vector'
                    print('last_output shape and dtype:', last_output.shape, last_output.dtype)
                    generated_words_indices = tf.transpose(tf.stack([indices,
                                                                     tf.cast(tf.ones([tf.shape(indices)[0]]) * i,
                                                                             tf.int32),
                                                                     tf.cast(tf.ones([tf.shape(indices)[0]]) * j,
                                                                             tf.int32)]))
                    generated_prgrphs = generated_prgrphs + tf.cast(tf.scatter_nd(generated_words_indices, last_output,
                                                                                  [batch_size, max_sent_num,
                                                                                   max_sent_len]), dtype=tf.int32)

                    generated_prgrphs_embeddings = generated_prgrphs_embeddings + \
                                                   tf.scatter_nd(generated_words_indices,
                                                                 self.get_embeddings(last_output),
                                                                 [batch_size, max_sent_num, max_sent_len,
                                                                  self.embedding_dim])

                    'updating indices by eliminating indices which eos was generated in them'
                    indices = tf.boolean_mask(indices, tf.logical_not(tf.equal(last_output, eos_ind)))
                    eos_indices = tf.cast(tf.where(tf.equal(last_output, eos_ind)), tf.int32)
                    # print('last_output',last_output)
                    # print('eos_indices',eos_indices)
                    if (tf.shape(eos_indices)[0] > 0):
                        hidden_index_vec = tf.ones([tf.shape(eos_indices)[0]], tf.int32) * (i * max_sent_len + j)
                        index_vec2 = tf.ones([tf.shape(eos_indices)[0], 1], tf.int32) * i
                        new_indices2 = tf.keras.layers.concatenate(inputs=[eos_indices, index_vec2], axis=1)
                        ending_hidden_indices = ending_hidden_indices + tf.cast(
                            tf.scatter_nd(new_indices2, hidden_index_vec,
                                          [batch_size, max_sent_num]), tf.int32)

                # last_noneos_output = tf.gather(last_output, indices)

                'updating unfinished_sents_indices'
                ending_hidden_indices_upto_i = tf.gather(ending_hidden_indices, unfinished_prgrphs_indices)[:, :i + 1]
                print('ending_hiddens_upto_i', tf.shape(ending_hidden_indices_upto_i))
                a = tf.tile(tf.expand_dims(unfinished_prgrphs_indices, 1), multiples=[1, i + 1])
                ending_indices = tf.keras.layers.concatenate(
                    [tf.expand_dims(a, axis=2), tf.expand_dims(ending_hidden_indices_upto_i, axis=2)], axis=2)
                'ending_indices shape: [available_prgrphs_num, generated_sents_num, 2]'
                curr_prgrphs_last_hiddens = tf.gather_nd(hidden_states, ending_indices)

                # curr_sents = tf.gather(generated_prgrphs_embeddings, unfinished_prgrphs_indices)[:, i, :, :]
                # encoded_sents = self.sent_encoder_module([curr_sents])
                # self.update_entity_module([encoded_sents, unfinished_prgrphs_indices])

                'updating entities'
                encoded_sents = tf.expand_dims(self.sent_encoder_module(generated_prgrphs_embeddings[:, i, :, :]),
                                               axis=1)
                # print("unfinished_prgrphs_indices",unfinished_prgrphs_indices)
                sents_mask = tf.scatter_nd(tf.expand_dims(unfinished_prgrphs_indices, axis=1),
                                           tf.expand_dims(
                                               tf.ones([tf.shape(unfinished_prgrphs_indices)[0]], dtype=tf.bool),
                                               axis=1),
                                           [batch_size, 1])
                if return_last:
                    entity_hiddens = simple_entity_network(entity_cell=self.entity_cell,
                                                           inputs=[encoded_sents, sents_mask],
                                                           keys=keys, initial_entity_hidden_state=entity_hiddens,
                                                           use_shared_keys=use_shared_keys, return_last=True)
                else:
                    new_entity_hiddens = simple_entity_network(entity_cell=self.entity_cell,
                                                               inputs=[encoded_sents, sents_mask],
                                                               keys=keys,
                                                               initial_entity_hidden_state=all_entity_hiddens[:, -1, :,
                                                                                           :],
                                                               use_shared_keys=use_shared_keys, return_last=True)
                    all_entity_hiddens = tf.concat([all_entity_hiddens, tf.expand_dims(new_entity_hiddens, axis=1)],
                                                   axis=1)

                if return_last:
                    # print("show me!: ", entity_hiddens.shape)
                    classifier_results = self.prgrph_ending_classifier([curr_prgrphs_last_hiddens,
                                                                        tf.gather(entity_hiddens,
                                                                                  unfinished_prgrphs_indices),
                                                                        tf.gather(keys_mask,
                                                                                  unfinished_prgrphs_indices)])
                else:
                    classifier_results = self.prgrph_ending_classifier(
                        [curr_prgrphs_last_hiddens,
                         tf.gather(all_entity_hiddens[:, -1, :, :], unfinished_prgrphs_indices),
                         tf.gather(keys_mask, unfinished_prgrphs_indices)])
                'classifier_results : probabilities'
                print('classifier_results', classifier_results)
                bool_results = tf.squeeze(tf.less(classifier_results, 0.5), axis=1)
                print('bool_results', bool_results)
                not_ended_prgrphs_indices = tf.squeeze(tf.where(bool_results), axis=1)
                print('unfinished_prgrph_indices', unfinished_prgrphs_indices)
                print('not_ended_prgrphs_indices', not_ended_prgrphs_indices)
                unfinished_prgrphs_indices = tf.gather(unfinished_prgrphs_indices, not_ended_prgrphs_indices)
                # print('unfinished_prgrph_indices2', unfinished_prgrphs_indices)

        if return_last:
            return generated_prgrphs, entity_hiddens
        else:
            return generated_prgrphs, all_entity_hiddens

    def call(self, inputs, keys, keys_mask, training, initial_hidden_state=None,
             encoder_hidden_states=None, labels=None,
             num_inputs=None, update_positions=None, use_shared_keys=False,
             return_last=True, attention=False, self_attention=False):
        print("in call decoder!")
        """
    
        inputs: [test_mode_bool, entity_hiddens, vocab_size, start_token ] in training mode
                [entity_hiddens, max_sent_num, max_sent_len, eos_ind, start_token ] in test mode
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
            return self.decode_train(inputs=inputs, labels=labels,
                                     keys=keys, keys_mask=keys_mask,
                                     encoder_hidden_states=encoder_hidden_states,
                                     initial_hidden_state=initial_hidden_state, use_shared_keys=use_shared_keys,
                                     return_last=return_last, attention=attention, self_attention=self_attention)

        else:
            return self.decode_test(inputs=inputs, keys=keys, keys_mask=keys_mask,
                                    encoder_hidden_states=encoder_hidden_states,
                                    initial_hidden_state=initial_hidden_state, use_shared_keys=use_shared_keys,
                                    return_last=return_last, attention=attention, self_attention=self_attention)
