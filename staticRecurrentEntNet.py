import tensorflow as tf
import numpy as np
import prgrph_ending_classifier
import train

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
        # I assume word embedding for indexes greater that sentnece length is zero vector, so it does not effect sentence encoding '
        assert isinstance(inputs, list)
        sents = inputs[0]

        return tf.reduce_sum(sents, 1)


class Update_entity(tf.keras.Model):
    def __init__(self, entity_num, entity_embedding_dim, activation=tf.nn.relu, initializer=None, name=None):
        if name is None:
            name = 'update_entity'

        super().__init__(name=name)
        self.entity_num = entity_num
        self.entity_embedding_dim = entity_embedding_dim
        self.activation = activation
        if initializer is None:
            self.initializer = tf.keras.initializers.random_normal()
        else:
            self.initializer = initializer
        # defining Variables
        self.U = None
        # self._variables.append(self.U)
        self.V = None
        # self._variables.append(self.V)
        self.W = None
        # self._variables.append(self.W)
        self.built = False

    def build(self, input_shape):
        shape = [self.entity_embedding_dim, self.entity_embedding_dim]
        self.U = K.variable(self.initializer(shape), name='U')
        self.V = K.variable(self.initializer(shape), name='V')
        self.W = K.variable(self.initializer(shape), name='W')
        self.built = True

    def initialize_hidden(self, hiddens):
        self.batch_size = hiddens.shape[0]
        self.hiddens = hiddens

    def assign_keys(self, entity_keys):
        self.keys = entity_keys

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
        # expanded=tf.expand_dims(encoded_sents,axis=1)
        # print('expanded shape:', expanded.shape)
        # print('tile shape:', tf.tile(tf.expand_dims(encoded_sents,axis=1),[1,self.entity_num,1]).shape)
        # print('curent hiddens shape:', current_hiddens.shape)
        #
        # print(tf.reduce_sum(tf.multiply(tf.tile(tf.expand_dims(encoded_sents,axis=1),[1,self.entity_num,1]),current_hiddens)\
        #        +tf.multiply(tf.tile(tf.expand_dims(encoded_sents,axis=1),[1,self.entity_num,1]),current_keys),axis=2).shape)
        # return tf.sigmoid(tf.reduce_sum(tf.multiply(tf.tile(tf.expand_dims(encoded_sents,axis=1),[1,self.entity_num,1]),current_hiddens)\
        #        +tf.multiply(tf.tile(tf.expand_dims(encoded_sents,axis=1),[1,self.entity_num,1]),current_keys),axis=2))

        # break complex formulas to simpler to be trackable!!
        print('enocded_sents dtype:',encoded_sents.dtype)
        print('current_hiddens dtype:',current_hiddens.dtype)
        return tf.sigmoid(tf.reduce_sum(tf.multiply(tf.expand_dims(encoded_sents, 1), current_hiddens) +
                                        tf.multiply(tf.expand_dims(encoded_sents, 1), current_keys), axis=2))

    def update_hidden(self, gates, current_hiddens, current_keys, encoded_sents, indices):
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
                       tf.matmul(tf.reshape(current_hiddens, [-1, self.entity_embedding_dim]), self.V) +
                       tf.matmul(tf.reshape(tf.tile(tf.expand_dims(encoded_sents, 1), [1, self.entity_num, 1]),
                                            shape=[-1, self.entity_embedding_dim]), self.W),
                       shape=[curr_prgrphs_num, self.entity_num, self.entity_embedding_dim]))
        'h_tilda shape: [current_prgrphs_num, entity_num, entity_embedding_dim]'
        # tf.multiply(gates,h_tilda)
        self.hiddens = self.hiddens + tf.scatter_nd(tf.expand_dims(indices, axis=1), tf.multiply(
            tf.tile(tf.expand_dims(gates, axis=2), [1, 1, self.entity_embedding_dim]), h_tilda),
                                                    shape=[self.batch_size, self.entity_num, self.entity_embedding_dim])

    def normalize(self):
        self.hiddens = tf.nn.l2_normalize(self.hiddens, axis=2)

    def call(self, inputs, training=None):
        """
        Description:
            Updates related etities
        Args:
            inputs: encoded_sents shape : [current_prgrphs_num,encoding_dim] , here encoding_dim is equal to embedding_dim
        """
        assert isinstance(inputs, list)

        encoded_sents, indices = inputs
        current_hiddens = tf.gather(self.hiddens, indices)
        print('ENCODE')
        print(self.keys.shape)
        print(indices)
        current_keys = tf.gather(self.keys, indices)

        if current_hiddens.shape != current_keys.shape:
            raise AttributeError('hiddens and kes must have same shape')

        gates = self.get_gate(encoded_sents, current_hiddens, current_keys)
        self.update_hidden(gates, current_hiddens, current_keys, encoded_sents, indices)
        self.normalize()
        return self.hiddens


class StaticRecurrentEntNet(tf.keras.Model):
    def __init__(self, embedding_matrix, entity_num, entity_embedding_dim, rnn_hidden_size, vocab_size, start_token,
                 max_sent_num, name=None):

        if name is None:
            name = 'staticRecurrentEntNet'
        super().__init__(name=name)
        self.embedding_matrix = embedding_matrix
        # embedding_matrix shape: [vocab_size, embedding_dim]
        # I assume the last row is an all zero vector for fake words with index embedding_matrix.shape[0]
        self.embedding_dim = self.embedding_matrix.shape[1]
        # self.add_zero_vector()
        self.entity_num = entity_num
        self.entity_embedding_dim = entity_embedding_dim
        self.hidden_Size = rnn_hidden_size
        self.vocab_size = vocab_size
        self.start_token = start_token
        self.max_sent_num = max_sent_num
        'start_token shape:[1,enbedding_dim]'

        ' defining submodules '
        self.sent_encoder_module = Sent_encoder()
        self.update_entity_module = Update_entity(self.entity_num, self.entity_embedding_dim)
        self.prgrph_ending_classifier = prgrph_ending_classifier.Prgrph_ending_classifier(
            max_sent_num=self.max_sent_num,
            encoding_dim=rnn_hidden_size,
            entity_embedding_dim=entity_embedding_dim)
        self.lstm = tf.keras.layers.LSTM(self.hidden_Size, return_state=True)
        self.decoder_dense = tf.keras.layers.Dense(self.vocab_size, activation='softmax')
        self.entity_dense = tf.keras.layers.Dense(self.hidden_Size)
        self.start_hidden_dense=tf.keras.layers.Dense(self.hidden_Size)


    def build(self, input_shape):
        self.entity_attn_matrix = K.random_normal_variable(shape=[self.hidden_Size, self.embedding_dim],
                                                           mean=0, scale=0.05, name='entity_attn_matrix')

    # @property
    # def trainable(self):
    #     return self._trainable

    def attention_hiddens(self, query, keys, memory_mask):
        '''
        Description:
            attention on keys with given quey, value is equal to keys

        Args:
            inputs: query shape: [curr_prgrphs_num, hiddens_size]
                    keys shape: [curr_prgrphs_num, prev_hiddens_num, hidden_size]
                    memory_mask: [curr_prgrphs_num, prev_hiddens_num]
            output shape: [curr_prgrphs_num, hidden_size]
        '''
        print('in attention_hiddens')
        print('keys shape:',keys.shape)
        print('query shape:',query.shape)
        print('mask shape',memory_mask.shape)
        values = tf.identity(keys)
        query_shape = tf.shape(query)
        keys_shape = tf.shape(keys)
        values_shape = tf.shape(values)
        batch_size = query_shape[0]
        seq_length = keys_shape[1]
        query_dim = query_shape[1]
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

    def attention_entities(self, query, entities):
        '''
        Description:
            attention on entities

        Arges:
            inputs: query shape: [curr_prgrphs_num, hiddens_size]
                    entities shape: [curr_prgrphs_num, entities_num, entitiy_embedding_dim]
            output shape: [curr_prgrphs_num, entity_embedding_dim]
        '''

        return tf.reduce_sum(tf.multiply(tf.expand_dims(tf.matmul(query, self.entity_attn_matrix), axis=1), entities),
                             axis=1)

    def calculate_hidden(self, curr_sents_prev_hiddens, entities, mask):
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
            curr_sents_prev_hiddens[:, curr_sents_prev_hiddens.shape[1] - 1, :],
            curr_sents_prev_hiddens[:, :curr_sents_prev_hiddens.shape[1], :], mask)
        attn_entities_output = self.attention_entities(attn_hiddens_output, entities)
        return self.entity_dense(attn_entities_output)

    # def add_zero_vector(self):
    #     embedding_dim=self.embedding_matrix.shape[1]
    #     self.embedding_matrix=tf.concat([self.embedding_matrix,tf.zeros([1,embedding_dim])],axis=0)

    def get_embeddings(self, input):
        return tf.nn.embedding_lookup(self.embedding_matrix, input)

    def encode_prgrph(self, inputs):
        '''
            TASK 1
            ENCODING given paragraph
        '''

        ''' 
        inputs: entity_keys, prgrph, prgrph_mask
        output: entity_hiddens last state
        '''

        # if prgrph is None:
        #     raise AttributeError('prgrph is None')
        # if prgrph_mask is None:
        #     raise AttributeError('prgrph_mask is None')
        # if entity_keys is None:
        #     raise AttributeError('entity_keys is None')
        if len(inputs) != 3:
            raise AttributeError('expected 3 inputs but', len(inputs), 'were given')
        entity_keys, prgrph, prgrph_mask = inputs
        batch_size = prgrph.shape[0]
        max_sent_num = prgrph.shape[1]

        prgrph_embeddings = self.get_embeddings(prgrph)
        # print('first_prgrph_embedding shape:',first_prgrph_embeddings.shape)
        'prgrph_embeddings shape: [batch_size, max_sent_num, max_sent_len, embedding_dim]'

        self.update_entity_module.initialize_hidden(
            tf.zeros([batch_size, self.entity_num, self.entity_embedding_dim], tf.float32))
        self.update_entity_module.assign_keys(entity_keys)

        for i in range(max_sent_num):
            ''' to see which sentences are available '''
            indices = tf.where(prgrph_mask[:, i, 0])
            print('indices shape encode:',indices.shape)
            indices = tf.cast(tf.squeeze(indices, axis=1),tf.int32)
            print('indices_p1_mask', indices)
            # print('first_prgrph_embedding shape:',first_prgrph_embeddings.shape)
            # print('first_prgrph_embeddings[:,i,:,:] shape:',first_prgrph_embeddings[:,i,:,:].shape)
            current_sents = tf.gather(prgrph_embeddings[:, i, :, :], indices)
            print('current_sents_call shape:', current_sents.shape)
            encoded_sents = self.sent_encoder_module([current_sents])
            self.update_entity_module([encoded_sents, indices])

        return self.update_entity_module.hiddens

    def decode_train(self, inputs):
        '''
            TASK 2 :
            Language model on given paragraph (second prgrph)
        '''

        ''' 
        input: prgrph, prgrph mask, entity_hiddens last state
        returns: final_output of shape : [batch_size, max_sents_num*max_sents_len, vocab_size]
                 final_targets of shape : [[batch_size, max_sents_num*max_sents_len]
                 output_mask of shape : [batch_size, max_sents_num*max_sents_len]
        '''

        # if prgrph is None:
        #     raise AttributeError('prgrph is None')
        # if prgrph_mask is None:
        #     raise AttributeError('prgrph_mask is None')
        # if entity_hiddens is None:
        #     raise AttributeError('entity_hiddens is None')
        if len(inputs) != 5:
            raise AttributeError('expected 5 inputs but', len(inputs), 'were given')

        print('IN DECODE_TRAIN')

        test_mode_bool, entity_hiddens, prgrph, prgrph_mask, vocab_size = inputs
        batch_size = prgrph.shape[0]
        max_sent_num = prgrph.shape[1]
        max_sent_len = prgrph.shape[2]

        prgrph_embeddings = self.get_embeddings(prgrph)

        self.update_entity_module.initialize_hidden(entity_hiddens)

        final_output=tf.zeros(shape=[batch_size,max_sent_num,max_sent_len,vocab_size],dtype=tf.float32)
        final_targets=tf.zeros(shape=[batch_size,max_sent_num,max_sent_len],dtype=tf.int32)
        output_mask=tf.zeros(shape=[batch_size,max_sent_num,max_sent_len],dtype=tf.int32)

        ' stores previous hidden_states of the lstm for the prgrph '
        hidden_states = tf.zeros([batch_size, max_sent_num * max_sent_len, self.hidden_Size])
        hiddens_mask = tf.reshape(prgrph_mask, [batch_size, -1])

        cell_states=tf.zeros([1])

        for i in range(max_sent_num):
            # print('p2_mask',p2_mask)
            current_sents_indices = tf.where(prgrph_mask[:, i, 0])
            current_sents_indices=tf.squeeze(current_sents_indices,axis=1)
            for j in range(max_sent_len):
                print('current word indeX:', i,j)
                ' indices of available paragraphs'
                # print('prgrph_mask.shape',tf.where(prgrph_mask).shape)
                # print('prgprh_mask[:,i,j]',prgrph_mask[:,i,j])
                indices = tf.cast(tf.where(prgrph_mask[:, i, j]),dtype=tf.int32)
                print('indices shape:',indices.shape)
                indices = tf.squeeze(indices, axis=1)
                if indices.shape[0]>0:
                # print('indices_p2_mask:',indices)   #indices_p2_mask: tf.Tensor([[0]], shape=(1, 1), dtype=int64)
                    if j == 0:
                        # print('start token',self.embedding_matrix[self.start_token].shape)
                        lstm_inputs = tf.tile(tf.expand_dims(self.embedding_matrix[self.start_token], axis=0),
                                              [batch_size, 1])
                        print('lstm_first_inputs:',lstm_inputs.shape)
                    else:
                        # print('prgrph_embedding_shape:',prgrph_embeddings[:, i, j - 1, :].shape)
                        lstm_inputs = tf.gather(prgrph_embeddings[:, i, j - 1, :], indices)

                    # print(tf.gather(second_prgrph[:, i, j], indices).shape)
                    lstm_targets = tf.gather(prgrph[:, i, j], indices)
                    if i * max_sent_len + j == 0:
                        curr_sents_curr_hidden = self.start_hidden_dense(tf.reduce_sum(entity_hiddens,axis=1))
                        cell_states = tf.zeros([batch_size, self.hidden_Size], tf.float32)
                        print('curr_sents_cell_state.shape',cell_states.shape)
                    else:
                        curr_sents_prev_hiddens = tf.gather(hidden_states[:, :i * max_sent_len + j, :], indices)
                        curr_sents_prev_hiddens_mask = tf.gather(hiddens_mask[:, :i * max_sent_len + j], indices)
                        curr_sents_entities = tf.gather(self.update_entity_module.hiddens, indices)
                        curr_sents_curr_hidden = self.calculate_hidden(curr_sents_prev_hiddens, curr_sents_entities,
                                                                       mask=curr_sents_prev_hiddens_mask)
                    curr_sents_cell_state=tf.gather(cell_states,indices)
                    print('lstm_inputs shape:',lstm_inputs.shape)
                    output, next_hidden, next_cell_state = self.lstm(tf.expand_dims(lstm_inputs, axis=1),
                                                                           initial_state=[
                                                                               curr_sents_curr_hidden,
                                                                               curr_sents_cell_state])

                    'updating cell_states'
                    curr_cells_prev_state=tf.gather(cell_states,indices)
                    cell_states=cell_states+tf.scatter_nd(tf.expand_dims(indices,axis=1),next_cell_state-curr_cells_prev_state,[batch_size,self.hidden_Size])
                    print('next_hidden shape:', next_hidden.shape)
                    'output shape:[len(indices), hidden_size] here, output is equal to next_hidden'
                    index_vector = tf.ones([indices.shape[0], 1], tf.int32) * (i * max_sent_len + j)
                    print('indices type:',indices.dtype)
                    new_indices = tf.concat(values=[tf.expand_dims(indices, 1), index_vector], axis=1)
                    print('new_indices:', new_indices)
                    hidden_states = hidden_states + tf.scatter_nd(new_indices, next_hidden,
                                                                  shape=[batch_size, hidden_states.shape[1],
                                                                         self.hidden_Size])
                    print('hidden_state.shape',hidden_states.shape)
                    # print('hidden_states:',hidden_states)
                    output = self.decoder_dense(output)
                    print('ouput_shape', output.shape)

                    second_dim_ind=tf.ones([indices.shape[0],1],tf.int32)*i
                    third_dim_ind=tf.ones([indices.shape[0],1],tf.int32)*j
                    new_indices_output=tf.keras.layers.concatenate(inputs=[tf.expand_dims(indices,1),second_dim_ind,third_dim_ind],axis=1)
                    final_output=final_output+tf.scatter_nd(new_indices_output,output,shape=[batch_size,max_sent_num,max_sent_len,vocab_size])

                    print('final_targets dtype',final_targets.dtype)
                    print(tf.scatter_nd(new_indices_output,lstm_targets,shape=[batch_size,max_sent_num,max_sent_len]).dtype)
                    final_targets=final_targets+tf.cast(tf.scatter_nd(new_indices_output,lstm_targets,shape=[batch_size,max_sent_num,max_sent_len]),dtype=tf.int32)

                    t=tf.ones([indices.shape[0]],dtype=tf.int32)
                    output_mask=output_mask+tf.scatter_nd(new_indices_output,t,shape=[batch_size,max_sent_num,max_sent_len])

            if test_mode_bool==True:
                return tf.zeros([1])
            # print('current_sents_indices',current_sents_indices)
            current_sents = tf.gather(prgrph_embeddings[:, i, :, :], current_sents_indices)
            print('decode_train, currents_sents shape',current_sents.shape)
            encoded_sents = self.sent_encoder_module([current_sents])
            self.update_entity_module([encoded_sents, current_sents_indices])
            # print('updated_hiddens', self.update_entity_module.hiddens)
        final_output=tf.reshape(final_output,[batch_size,-1,vocab_size])
        final_targets=tf.reshape(final_targets,[batch_size,-1])
        output_mask=tf.reshape(output_mask,[batch_size,-1])
        return final_output, final_targets,output_mask

    def decode_test(self, inputs):
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
        generated_prgrphs_embeddings=self.get_embeddings(generated_prgrphs)

        last_noneos_output = tf.zeros([1], dtype=tf.float32)
        ' indices of available paragraphs'
        unfinished_prgrphs_indices = tf.range(start=0, limit=batch_size, dtype=tf.int32)
        ending_hidden_indices = tf.zeros([batch_size, self.max_sent_num], dtype=tf.int32)

        cell_states=tf.zeros([1])

        for i in range(self.max_sent_num):
            indices = tf.identity(unfinished_prgrphs_indices)
            if indices.shape[0]>0:
                for j in range(max_sent_len):
                    print('current word indeX:', i,j)
                    print("GENERATED PARAGRAPHS")
                    print(generated_prgrphs)
                    if j == 0:
                        # print('start token',self.embedding_matrix[self.start_token].shape)
                        lstm_inputs = tf.tile(tf.expand_dims(self.embedding_matrix[self.start_token], axis=0),
                                              [batch_size, 1])
                    else:
                        lstm_inputs = tf.gather(generated_prgrphs_embeddings[:,i,j-1,:],indices)

                    if i * max_sent_len + j == 0:
                        # print(tf.reduce_sum(entity_hiddens,axis=1))
                        curr_sents_curr_hidden = self.start_hidden_dense(tf.reduce_sum(entity_hiddens, axis=1))
                        cell_states = tf.zeros([batch_size, self.hidden_Size], tf.float32)
                    else:
                        print('hidden_states shape:',hidden_states.shape)
                        curr_sents_prev_hiddens = tf.gather(hidden_states[:, :i * max_sent_len + j, :], indices)
                        print('curr_sents_prev_hiddens shape:',curr_sents_prev_hiddens.shape,indices.shape)
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
                    next_hidden_mask = tf.scatter_nd(tf.expand_dims(indices, axis=1), tf.expand_dims(boolean_vec, axis=1),
                                                     shape=[batch_size, 1])
                    hiddens_mask = tf.keras.layers.concatenate(inputs=[hiddens_mask, next_hidden_mask], axis=1)

                    # print('hidden_states:',hidden_states)
                    lstm_output = self.decoder_dense(lstm_output)
                    last_output = tf.cast(tf.argmax(lstm_output, dimension=1),tf.int32)

                    if j == max_sent_len - 1:
                        last_output = tf.cast(tf.ones([indices.shape[0]]) * eos_ind,dtype=tf.int32)
                    'last_output is a one_dimensional vector'
                    print('last_output shape and dtype:',last_output.shape,last_output.dtype)
                    # a=tf.ones([last_output.shape[0]]) * i
                    # print(a)
                    generated_words_indices = tf.transpose(tf.stack([indices,
                                                                     tf.cast(tf.ones([indices.shape[0]]) * i,tf.int32),
                                                                     tf.cast(tf.ones([indices.shape[0]]) * j,tf.int32)]))
                    # print(tf.scatter_nd(generated_words_indices, last_output,[batch_size, self.max_sent_num, max_sent_len]).dtype)
                    generated_prgrphs = generated_prgrphs + tf.cast(tf.scatter_nd(generated_words_indices, last_output,
                                                                          [batch_size, self.max_sent_num, max_sent_len]),dtype=tf.int32)

                    generated_prgrphs_embeddings=generated_prgrphs_embeddings+\
                                                 tf.scatter_nd(generated_words_indices,self.get_embeddings(last_output),[batch_size,self.max_sent_num,max_sent_len,self.embedding_dim])

                    'updating indices by eliminating indices which eos was generated in them'
                    indices = tf.boolean_mask(indices, tf.logical_not(tf.equal(last_output, eos_ind)))
                    eos_indices = tf.cast(tf.where(tf.equal(last_output, eos_ind)),tf.int32)
                    # print('last_output',last_output)
                    # print('eos_indices',eos_indices)
                    if (eos_indices.shape[0] > 0):
                        hidden_index_vec = tf.ones([eos_indices.shape[0]]) * (i * max_sent_len + j)
                        index_vec2 = tf.ones([eos_indices.shape[0], 1], tf.int32) * i
                        new_indices2 = tf.keras.layers.concatenate(inputs=[eos_indices, index_vec2], axis=1)
                        # print(tf.scatter_nd(new_indices2, hidden_index_vec,[batch_size, self.max_sent_num]).dtype)
                        ending_hidden_indices = ending_hidden_indices + tf.cast(tf.scatter_nd(new_indices2, hidden_index_vec,
                                                                                  [batch_size, self.max_sent_num]),tf.int32)

                # last_noneos_output = tf.gather(last_output, indices)

                'updating unfinished_sents_indices'
                ending_hidden_indices_upto_i = tf.gather(ending_hidden_indices, unfinished_prgrphs_indices)[:, :i + 1]
                print('ending_hiddens_upto_i',ending_hidden_indices_upto_i.shape)
                a = tf.tile(tf.expand_dims(unfinished_prgrphs_indices,1), multiples=[1, i + 1])
                ending_indices = tf.keras.layers.concatenate([tf.expand_dims(a, axis=2), tf.expand_dims(ending_hidden_indices_upto_i, axis=2)], axis=2)
                'ending_indices shape: [available_prgrphs_num, generated_sents_num, 2]'
                curr_prgrphs_last_hiddens = tf.gather_nd(hidden_states, ending_indices)

                curr_sents = tf.gather(generated_prgrphs_embeddings, unfinished_prgrphs_indices)[:, i, :, :]
                encoded_sents = self.sent_encoder_module([curr_sents])
                self.update_entity_module([encoded_sents, unfinished_prgrphs_indices])
                classifier_results = self.prgrph_ending_classifier([curr_prgrphs_last_hiddens,
                                                                   self.update_entity_module.hiddens])
                'classifier_results : probabilities'
                print('classifier_results',classifier_results)
                bool_results = tf.squeeze(tf.less(classifier_results, 0.5),axis=1)
                print('bool_results',bool_results)
                not_ended_prgrphs_indices = tf.squeeze(tf.where(bool_results),axis=1)
                print('unfinished_prgrph_indices',unfinished_prgrphs_indices)
                print('not_ended_prgrphs_indices',not_ended_prgrphs_indices)
                unfinished_prgrphs_indices = tf.gather(unfinished_prgrphs_indices, not_ended_prgrphs_indices)
                # print('unfinished_prgrph_indices2', unfinished_prgrphs_indices)

        return generated_prgrphs

    def call(self, inputs, training=None):
        '''
        args:
            inputs: mode: encode, decode_train, decode_test
                    prgrph shape : [batch_size, max_sent_num, max_sent_len]
                    * I assume that fake words have index equal to embedding_matrix.shape[0]
                    entity_keys : initialized entity keys of shape : [batch_size, entity_num, entity_embedding_dim] , entity_embedding_dim=embedding_dim for now
                    prgrph_mask : mask for given prgrph, shape=[batch_size, max_sent_num, max_sent_len]
        '''

        print('inputs type:', type(inputs))
        assert isinstance(inputs, list)
        # what is inputs?
        mode = inputs[0]
        if mode == 'encode':
            return self.encode_prgrph(inputs[1:])

        else:
            if mode == 'decode_train':
                return self.decode_train(inputs[1:])


            else:
                if mode == 'decode_test':
                    return self.decode_test(inputs[1:])

