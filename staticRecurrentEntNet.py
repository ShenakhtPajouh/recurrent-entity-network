import tensorflow as tf
import numpy as np


class Sent_encoder(tf.keras.Model):
    def __init__(self, name=None):
        if name is None:
            name = 'sent_encoder'
        super().__init__(name=name)

    def call(self, sents):
        '''
        Description:
            encode given sentences with bag of words algorithm
        Args:
            input: sents shape: [current_prgrphs_num,max_sent_len,embedding_dim]
            output: encoded sentences of shape [current_prgrphs_num,encoding_dim] , here encoding_dim is equal to embedding_dim
        '''

        ' I assume word embedding for indexes greater that sentnece length is zero vector, so it does not effect sentence encoding '

        print('sents shape:', sents.shape)
        return tf.reduce_sum(sents, 1)


class Update_entity(tf.keras.Model):
    def __init__(self, entity_num, entity_embedding_dim, activation=tf.nn.relu, name=None):
        if name is None:
            name = 'update_entity'

        super().__init__(name=name)
        self.entity_num = entity_num
        self.entity_embedding_dim = entity_embedding_dim
        self.activation = activation
        # self._variables = []
        self._trainable_weights=[]
        ' defining Variables '
        self.U = self.add_weight(shape=[self.entity_embedding_dim, self.entity_embedding_dim],initializer='normal',
                                                  name='entityVariable_U',trainable=True)
        # self._variables.append(self.U)
        self.V = self.add_weight(shape=[self.entity_embedding_dim, self.entity_embedding_dim],initializer='normal',
                                                  name='entityVariable_V',trainable=True)
        # self._variables.append(self.V)
        self.W = self.add_weight(shape=[self.entity_embedding_dim, self.entity_embedding_dim],initializer='normal',
                                                  name='entityVariable_W',trainable=True)
        # self._variables.append(self.W)


    @property
    def variables(self):
        return self.trainable_weights

    @property
    def trainable_weights(self):
        return self._trainable_weights

    def initialize_hidden(self, hiddens):
        self.batch_size=hiddens.shape[0]
        self.hiddens = hiddens

    def assign_keys(self, entity_keys):
        self.keys = entity_keys

    def get_gate(self, encoded_sents,current_hiddens,current_keys):
        '''
        Description:
            calculate the gate g_i for all hiddens of given paragraphs
        Args:
            inputs: encoded_sents of shape: [current_prgrphs_num, encoding_dim]
                    current_hiddens: [current_prgrphs_num, entity_num, entity_embedding_dim]
                    current_keys: [current_prgrphs_num, entity_num, entity_embedding_dim]

            output: gates of shape : [curr_prgrphs_num, entity_num]
        '''
        # expanded=tf.expand_dims(encoded_sents,axis=1)
        # print('expanded shape:', expanded.shape)
        # print('tile shape:', tf.tile(tf.expand_dims(encoded_sents,axis=1),[1,self.entity_num,1]).shape)
        # print('curent hiddens shape:', current_hiddens.shape)
        #
        # print(tf.reduce_sum(tf.multiply(tf.tile(tf.expand_dims(encoded_sents,axis=1),[1,self.entity_num,1]),current_hiddens)\
        #        +tf.multiply(tf.tile(tf.expand_dims(encoded_sents,axis=1),[1,self.entity_num,1]),current_keys),axis=2).shape)
        # return tf.sigmoid(tf.reduce_sum(tf.multiply(tf.tile(tf.expand_dims(encoded_sents,axis=1),[1,self.entity_num,1]),current_hiddens)\
        #        +tf.multiply(tf.tile(tf.expand_dims(encoded_sents,axis=1),[1,self.entity_num,1]),current_keys),axis=2))

        return tf.sigmoid(tf.reduce_sum(tf.multiply(tf.expand_dims(encoded_sents,1),current_hiddens)+
                                        tf.multiply(tf.expand_dims(encoded_sents,1),current_keys),axis=2))

    def update_hidden(self, gates, current_hiddens, current_keys, encoded_sents, indices):
        '''
        Description:
            updates hidden_index for all prgrphs
        Args:
            inputs: gates shape: [current_prgrphs_num, entity_num]
                    encoded_sents of shape: [current_prgrphs_num, encoding_dim]
                    current_hiddens: [current_prgrphs_num, entity_num, entity_embedding_dim]
                    current_keys: [current_prgrphs_num, entity_num, entity_embedding_dim]
        '''
        print(current_hiddens.shape , self.U.shape)
        curr_prgrphs_num=current_hiddens.shape[0]
        h_tilda = self.activation(tf.reshape(tf.matmul(tf.reshape(current_hiddens,[-1,self.entity_embedding_dim]),self.U)+
                                             tf.matmul(tf.reshape(current_hiddens,[-1,self.entity_embedding_dim]),self.V)+
                                             tf.matmul(tf.reshape(tf.tile(tf.expand_dims(encoded_sents,0),[1,self.entity_num,1]),
                                                                  shape=[-1,self.entity_embedding_dim]),self.W),
                                             shape=[curr_prgrphs_num,self.entity_num,self.entity_embedding_dim]))
        'h_tilda shape: [current_prgrphs_num, entity_num, entity_embedding_dim]'
        print('h_tilda shape',h_tilda.shape)
        print('gates shape:',gates.shape)
        # tf.multiply(gates,h_tilda)
        self.hiddens=self.hiddens+tf.scatter_nd(tf.expand_dims(indices,axis=1),tf.multiply(tf.tile(tf.expand_dims(gates,axis=2),[1,1,self.entity_embedding_dim]),h_tilda),
                                                shape=[self.batch_size,self.entity_num,self.entity_embedding_dim])

    def normalize(self):
        self.hiddens = tf.nn.l2_normalize(self.hiddens, axis=2)


    def call(self, encoded_sents, indices):
        '''
        Description:
            Updates related etities
        Args:
            inputs: encoded_sents shape : [current_prgrphs_num,encoding_dim] , here encoding_dim is equal to embedding_dim
        '''

        print('self.hiddens shape:', self.hiddens.shape)
        print('indices:', indices)
        current_hiddens = tf.gather(self.hiddens, indices)
        print('current_hidden_call shape:',current_hiddens.shape)
        current_keys = tf.gather(self.keys, indices)

        if current_hiddens.shape!=current_keys.shape:
            raise AttributeError('hiddens and kes must have same shape')

        gates=self.get_gate(encoded_sents,current_hiddens,current_keys)
        self.update_hidden(gates,current_hiddens,current_keys,encoded_sents,indices)
        self.normalize()
        return self.hiddens


class StaticRecurrentEntNet(tf.keras.Model):
    def __init__(self, embedding_matrix, entity_num, entity_embedding_dim, rnn_hidden_size, vocab_size, start_token,
                 name=None):

        if name is None:
            name = 'staticRecurrentEntNet'
        super().__init__(name=name)
        self.embedding_matrix = embedding_matrix
        'embedding_matrix shape: [vocab_size, embedding_dim]'
        'I assume the last row is an all zero vector for fake words with index embedding_matrix.shape[0]'
        self.embedding_dim = self.embedding_matrix.shape[1]
        # self.add_zero_vector()
        self.entity_num = entity_num
        self.entity_embedding_dim = entity_embedding_dim
        self.hidden_Size = rnn_hidden_size
        self.vocab_size = vocab_size
        self.start_token = start_token
        'start_token shape:[1,enbedding_dim]'

        self.total_loss = 0
        self.optimizer = tf.train.AdamOptimizer()

        self._trainable_weights=[]

        ' defining submodules '
        self.sent_encoder_module = Sent_encoder()
        self.update_entity_module = Update_entity(self.entity_num, self.entity_embedding_dim)
        self.lstm = tf.keras.layers.LSTM(self.hidden_Size, return_state=True)
        self.decoder_dense = tf.keras.layers.Dense(self.vocab_size, activation='softmax')
        self.entity_dense = tf.keras.layers.Dense(self.hidden_Size)

        self.entity_attn_matrix =self.add_weight(shape=[self.hidden_Size, self.embedding_dim],initializer='normal',
                                                 name='entity_attn_matrix',trainable=True)


    # @property
    # def trainable(self):
    #     return self._trainable

    @property
    def variables(self):
        return [self.trainable_weights, self.lstm.variables, self.decoder_dense.variables, self.entity_dense.variables,
                self.update_entity_module.variables]

    @property
    def trainable_weights(self):
        return self._trainable_weights

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
        values=tf.identity(keys)
        query_shape = tf.shape(query)
        keys_shape = tf.shape(keys)
        values_shape = tf.shape(values)
        batch_size = query_shape[0]
        seq_length = keys_shape[1]
        query_dim = query_shape[1]
        indices = tf.where(memory_mask)
        queries = tf.gather(query, indices[:, 0])
        keys = tf.boolean_mask(keys, memory_mask)
        attention_logits = tf.reduce_sum(tf.multiply(tf.expand_dims(queries,1), keys),axis=2)
        attention_logits = tf.scatter_nd(tf.where(memory_mask), attention_logits, [batch_size, seq_length])
        attention_logits = tf.where(memory_mask, attention_logits, tf.fill([batch_size, seq_length], -float("Inf")))
        attention_coefficients = tf.nn.softmax(attention_logits,axis=1)
        attention = tf.expand_dims(attention_coefficients, -1) * values

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

        return tf.reduce_sum(tf.multiply(tf.expand_dims(tf.matmul(query,self.entity_attn_matrix),axis=1),entities),axis=1)



    def calculate_hidden(self, curr_sents_prev_hiddens, entities,mask):
        '''
        Description:
            calculates current hidden state that should be fed to lstm for predicting the next word, with attention on previous hidden states, THEN entities

        Args:
            inputs: curr_sents_prev_hiddens shape: [curr_prgrphs_num, prev_hiddens]
                    entities: [curr_prgrphs_num, entities_num, entity_embedding_dim]
            output shape: [curr_prgrphs_num, hidden_size]

        '''

        '''
        attention on hidden states:
            query: last column (last hidden_state)
            key and value: prev_columns
        '''
        print('last hidden shape:',curr_sents_prev_hiddens[:, curr_sents_prev_hiddens.shape[1] - 1, :].shape)
        attn_hiddens_output = self.attention_hiddens(
            curr_sents_prev_hiddens[:, curr_sents_prev_hiddens.shape[1] - 1, :],
            curr_sents_prev_hiddens[:, :curr_sents_prev_hiddens.shape[1], :],mask)
        attn_entities_output = self.attention_entites(attn_hiddens_output, entities)
        return self.entity_dense(attn_entities_output)

    def calculate_loss(self, outputs, lstm_targets):

        '''
        Args:
            inputs: outputs shape : [curr_prgrphs_num, vocab_size]
                    lstm_targets shape : [urr_prgrphs_num]
        '''
        one_hot_labels=tf.one_hot(lstm_targets,outputs.shape[1])
        print('outpus shape:',outputs.shape,outputs)
        print('one_hot_labels shape:',one_hot_labels.shape)
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot_labels, logits=outputs)
        print('loss',loss)
        return tf.reduce_mean(loss)

    # def add_zero_vector(self):
    #     embedding_dim=self.embedding_matrix.shape[1]
    #     self.embedding_matrix=tf.concat([self.embedding_matrix,tf.zeros([1,embedding_dim])],axis=0)

    def get_embeddings(self, prgrph):
        return tf.nn.embedding_lookup(self.embedding_matrix, prgrph)

    def call(self,mode, entity_keys=None,entity_hiddens=None, prgrph=None, prgrph_mask=None, max_sent_num=None, max_sent_len=None,eos_ind=None):
        '''
        args:
            inputs: mode: encode, decode_train, decode_test
                    prgrph shape : [batch_size, max_sent_num, max_sent_len]
                    * I assume that fake words have index equal to embedding_matrix.shape[0]
                    entity_keys : initialized entity keys of shape : [batch_size, entity_num, entity_embedding_dim] , entity_embedding_dim=embedding_dim for now
                    prgrph_mask : mask for given prgrph, shape=[batch_size, max_sent_num, max_sent_len]
        '''


        if mode=='encode':

            '''
            TASK 1 
            ENCODING given paragraph
            '''

            ''' 
            inputs: entity_keys, prgrph, prgrph_mask
            output: entity_hiddens last state
            '''

            if prgrph is None:
                raise AttributeError('prgrph is None')
            if prgrph_mask is None:
                raise AttributeError('prgrph_mask is None')
            if entity_keys is None:
                raise AttributeError('entity_keys is None')

            batch_size = prgrph.shape[0]
            max_sent_num = prgrph.shape[1]

            prgrph_embeddings = self.get_embeddings(prgrph)
            # print('first_prgrph_embedding shape:',first_prgrph_embeddings.shape)
            'prgrph_embeddings shape: [batch_size, max_sent_num, max_sent_len, embedding_dim]'

            self.update_entity_module.initialize_hidden(tf.zeros([batch_size, self.entity_num, self.entity_embedding_dim], tf.float32))
            self.update_entity_module.assign_keys(entity_keys)

            for i in range(max_sent_num):
                ''' to see which sentences are available '''
                indices = tf.where(prgrph_mask[:, i, 0])
                indices=tf.squeeze(indices,axis=0)
                print('indices_p1_mask',indices)
                # print('first_prgrph_embedding shape:',first_prgrph_embeddings.shape)
                # print('first_prgrph_embeddings[:,i,:,:] shape:',first_prgrph_embeddings[:,i,:,:].shape)
                current_sents = tf.gather(prgrph_embeddings[:,i,:,:], indices)
                print('current_sents_call shape:',current_sents.shape)
                encoded_sents = self.sent_encoder_module(current_sents)
                self.update_entity_module(encoded_sents, indices)

            return self.update_entity_module.hiddens

        else:
            if mode=='decode_train':

                '''
                TASK 2 : language model on given paragraph
                '''

                ''' 
                input: prgrph, prgrph mask, entity_hiddens last state
                yields the output predicted of shape [batch_size, vocab_size] and actual labels by predicting each word
                '''

                if prgrph is None:
                    raise AttributeError('prgrph is None')
                if prgrph_mask is None:
                    raise AttributeError('prgrph_mask is None')

                batch_size = prgrph.shape[0]
                max_sent_num = prgrph.shape[1]
                max_sent_len = prgrph.shape[2]

                prgrph_embeddings = self.get_embeddings(prgrph)

                self.update_entity_module.initialize_hidden(entity_hiddens)


                ' stores previous hidden_states of the lstm for the prgrph '
                hidden_states = tf.zeros([batch_size, max_sent_num * max_sent_len, self.hidden_Size])
                hiddens_mask=tf.reshape(prgrph_mask,[batch_size,-1])

                for i in range(max_sent_num):
                    # print('p2_mask',p2_mask)
                    current_sents_indices = tf.where(prgrph_mask[:, i, 0])
                    for j in range(max_sent_len):
                        print('current word indeX:',i * max_sent_len + j)
                        ' indices of available paragraphs'
                        indices = tf.where(prgrph_mask[:, i, j])
                        indices=tf.squeeze(indices,axis=0)
                        # print('indices_p2_mask:',indices)   #indices_p2_mask: tf.Tensor([[0]], shape=(1, 1), dtype=int64)
                        if j == 0:
                            # print('start token',self.embedding_matrix[self.start_token].shape)
                            lstm_inputs = tf.tile(tf.expand_dims(self.embedding_matrix[self.start_token],axis=0),[batch_size,1])
                        else:
                            lstm_inputs = tf.squeeze(tf.gather(prgrph_embeddings[:, i, j - 1, :], indices))

                        # print(tf.gather(second_prgrph[:, i, j], indices).shape)
                        lstm_targets = tf.gather(prgrph[:, i, j], indices)
                        if i * max_sent_len + j == 0:
                            curr_sents_curr_hidden = tf.zeros([batch_size, self.hidden_Size], tf.float32)
                            curr_sents_cell_state = tf.zeros([batch_size, self.hidden_Size], tf.float32)
                        else:
                            curr_sents_prev_hiddens = tf.gather(hidden_states[:, :i * max_sent_len + j, :], indices)
                            curr_sents_prev_hiddens_mask = tf.gather(hiddens_mask[:, :i * max_sent_len + j, :], indices)
                            curr_sents_entities = tf.gather(self.update_entity_module.hiddens, indices)
                            curr_sents_curr_hidden = self.calculate_hidden(curr_sents_prev_hiddens, curr_sents_entities,mask=curr_sents_prev_hiddens_mask)
                        output, next_hidden, curr_sents_cell_state = self.lstm(tf.expand_dims(lstm_inputs,axis=1), initial_state=[
                            curr_sents_curr_hidden, curr_sents_cell_state])
                        print('next_hidden shape:',next_hidden.shape)
                        'output shape:[batch_size, hidden_size] here, output is equal to next_hidden'
                        index_vector=tf.ones([indices.shape[0],1],tf.int64)*(i * max_sent_len + j)
                        new_indices=tf.keras.layers.concatenate(inputs=[tf.expand_dims(indices,1),index_vector],axis=1)
                        print('new_indices:',new_indices)
                        hidden_states=hidden_states+tf.scatter_nd(new_indices, next_hidden, shape=[batch_size,hidden_states.shape[1],self.hidden_Size])
                        # print('hidden_states:',hidden_states)
                        output = self.decoder_dense(output)
                        yield output,lstm_targets
                        # loss = self.calculate_loss(tf.squeeze(output,axis=1), lstm_targets)
                        # self.total_loss += loss
                        # gradients = tape.gradient(loss, self.variables)
                        # self.optimizer.apply_gradients(zip(gradients, self.variables))

                    current_sents = tf.gather(prgrph, current_sents_indices)[:, i, :, :]
                    encoded_sents = self.sent_encoder_module(current_sents)
                    self.update_entity_module(encoded_sents, current_sents_indices)
                    print('updated_hiddens',self.update_entity_module.hiddens)

            else:
                if mode=='decode_test':

                    ''' 
                    TASK 3 : predicting second paragraph
                    '''

                    ''' 
                    inputs: entity_hiddens last state
                    yields predicted output of shape [batch_size, vocab_size] each step
                    '''

                    if entity_hiddens is None:
                        raise AttributeError('entity_hiddens is None')
                    if max_sent_num is None:
                        raise AttributeError('max_sent_num is None')
                    if max_sent_len is None:
                        raise AttributeError('max_sent_len is None')
                    if eos_ind is None:
                        raise AttributeError('eos_ind is None')

                    self.update_entity_module.initialize_hidden(entity_hiddens)

                    batch_size=entity_hiddens.shape[0]

                    ' stores previous hidden_states of the lstm for the prgrph '
                    hidden_states = tf.zeros([batch_size, max_sent_num * max_sent_len, self.hidden_Size])
                    hiddens_mask = tf.equal(tf.ones([batch_size,1]),1)

                    generated_prgrphs=tf.zeros([batch_size,max_sent_num,max_sent_len],dtype=tf.int32)

                    last_output=tf.zeros([1],dtype=tf.float32)
                    indices=tf.zeros([1],dtype=tf.float32)
                    for i in range(max_sent_num):
                        for j in range(max_sent_len):
                            print('current word indeX:', i * max_sent_len + j)
                            ' indices of available paragraphs'
                            unfinished_sents_indices=tf.range(start=0,limit=batch_size,dtype=tf.int32)
                            if j == 0:
                                # print('start token',self.embedding_matrix[self.start_token].shape)
                                lstm_inputs = tf.tile(tf.expand_dims(self.embedding_matrix[self.start_token], axis=0),
                                                      [batch_size, 1])
                            else:
                                lstm_inputs = last_output


                            if i * max_sent_len + j == 0:
                                curr_sents_curr_hidden = tf.zeros([batch_size, self.hidden_Size], tf.float32)
                                curr_sents_cell_state = tf.zeros([batch_size, self.hidden_Size], tf.float32)
                            else:

                                curr_sents_prev_hiddens = tf.gather(hidden_states[:, :i * max_sent_len + j, :], indices)
                                curr_sents_prev_hiddens_mask = tf.gather(hiddens_mask[:, 1:, :],indices)
                                curr_sents_entities = tf.gather(self.update_entity_module.hiddens, indices)
                                curr_sents_curr_hidden = self.calculate_hidden(curr_sents_prev_hiddens,
                                                                               curr_sents_entities,
                                                                               mask=curr_sents_prev_hiddens_mask)
                            lstm_output, next_hidden, curr_sents_cell_state = self.lstm(tf.expand_dims(lstm_inputs, axis=1),
                                                                                   initial_state=[curr_sents_curr_hidden,curr_sents_cell_state])
                            print('next_hidden shape:', next_hidden.shape)
                            'output shape:[batch_size, hidden_size] here, output is equal to next_hidden'
                            index_vector = tf.ones([indices.shape[0], 1], tf.int64) * (i * max_sent_len + j)
                            new_indices = tf.keras.layers.concatenate(inputs=[tf.expand_dims(indices, 1), index_vector],
                                                                      axis=1)
                            print('new_indices:', new_indices)
                            hidden_states = hidden_states + tf.scatter_nd(new_indices, next_hidden,
                                                                          shape=[batch_size, hidden_states.shape[1],
                                                                                 self.hidden_Size])
                            tf.scatter_nd(indices,)
                            # print('hidden_states:',hidden_states)
                            lstm_output = self.decoder_dense(lstm_output)
                            last_output=tf.arg_max(lstm_output,dimension=1)
                            'last_output is a one_dimensional vector'
                            generated_words_indices=tf.transpose(tf.stack([last_output,tf.ones([last_output.shape[0]])*i,
                                                                           tf.ones([last_output.shape[0]])*j]))
                            generated_prgrphs=generated_prgrphs+tf.scatter_nd(generated_words_indices,last_output,[batch_size,max_sent_num,max_sent_len])

                            indices=tf.boolean_mask(indices,tf.logical_not(tf.equal(last_output,eos_ind)))

                            # loss = self.calculate_loss(tf.squeeze(output,axis=1), lstm_targets)
                            # self.total_loss += loss
                            # gradients = tape.gradient(loss, self.variables)
                            # self.optimizer.apply_gradients(zip(gradients, self.variables))

                        current_sents = tf.gather(prgrph, current_sents_indices)[:, i, :, :]
                        encoded_sents = self.sent_encoder_module(current_sents)
                        self.update_entity_module(encoded_sents, current_sents_indices)
                        print('updated_hiddens', self.update_entity_module.hiddens)




if __name__ == '__main__':
    tf.enable_eager_execution()
    embedding = tf.random_normal([10, 20])
    p1 = np.asarray([[[0,1,9,9], [2,3,4,9], [1,2,3,4]]])
    print(p1.shape)
    p1_mask=np.asarray([[[True,True,False,False],[True,True,True,False],[True,True,True,True]]])
    p2 = np.asarray([[[3,1,5,9], [2,3,9,9], [7,2,3,5]]])
    p2_mask = np.asarray([[[True, True, True, False], [True,True,False, False], [True,True,True,True]]])
    entity_keys=tf.random_normal([1,10,20])
    static_recur_entNet=StaticRecurrentEntNet(embedding_matrix=embedding,entity_num=10,entity_embedding_dim=20
                                              ,rnn_hidden_size=15,vocab_size=10,start_token=6,name='static_recur_entNet')
    static_recur_entNet(p1,p2,p1_mask,p2_mask,entity_keys)
    tf.keras.Model().__call__()

    print('hi!')
