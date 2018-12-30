import tensorflow as tf
tf.enable_eager_execution()

class Sent_encoder(tf.keras.Model):
    def __init__(self,name=None):
        if name is None:
            name='sent_encoder'


    def call(self, sents):
        '''
        Description:
            encode given sentences with bag of words algorithm
        Args:
            input: sents shape: [current_prgrphs_num,1,max_sent_len,embedding_dim]
            output: encoded sentences of shape [current_prgrphs_num,encoding_dim] , here encoding_dim is equal to embedding_dim
        '''

        ' I assume word embedding for indexes greater that sentnece length is zero vector, so it does not effect sentence encoding '

        return tf.squeeze(tf.reduce_sum(sents,2),axis=1)



class Update_entity(tf.keras.Model):
    def __init__(self,entity_num,entity_embedding_dim,activation=tf.nn.relu,name=None):
        if name is None:
            name='update_entity'

        self.entity_num=entity_num
        self.entity_embedding_dim=entity_embedding_dim
        self.activation=activation
        self._variables=[]

        ' defining Variables '
        self.U=tf.Variable(tf.random_normal([self.entity_embedding_dim,self.entity_embedding_dim]),name='entityVariable_U')
        self._variables.append(self.U)
        self.V = tf.Variable(tf.random_normal([self.entity_embedding_dim, self.entity_embedding_dim]),name='entityVariable_V')
        self._variables.append(self.V)
        self.W = tf.Variable(tf.random_normal([self.entity_embedding_dim, self.entity_embedding_dim]),name='entityVariable_W')
        self._variables.append(self.W)

    def initialize_hidden(self,batch_size):
        self.hiddens=tf.zeros([batch_size,self.entity_num,self.entity_embedding_dim],tf.float32)

    def assign_keys(self,entity_keys):
        self.keys=entity_keys

    def get_gate(self,encoded_sents,index):
        '''
        Description:
            calculate the gate g_i for index i, equation2 in EntNet article
        Args:
            input: encoded_sents of shape: [current_prgrphs_num, encoding_dim]
                   index: index of entity to be updated
            output: gate g_i of shape: [current_prgrphs_num, 1]
        '''
        return tf.sigmoid(tf.reduce_sum(tf.multiply(encoded_sents,tf.squeeze(self.hiddens[:,index,:],1)),axis=1),tf.reduce_sum(tf.multiply(encoded_sents,tf.squeeze(self.keys[:,index,:],1)),axis=1))

    def update_hidden(self, g_i, index, encoded_sents):
        '''
        Description:
            updates hidden_index for all prgrphs
        Args:
            inputs:
                g_i : gate for corresponded hidden
        '''
        h_i_tilda=self.activation(tf.add(tf.squeeze(self.hiddens[:,index,:]),tf.transpose(self.U),
                                         tf.matmul(tf.squeeze(self.keys[:, index, :]), tf.transpose(self.V)),
                                         tf.matmul(encoded_sents,tf.transpose(self.W))))
        self.hiddens[:,index,:]+=tf.expand_dims(tf.transpose(tf.multiply(g_i,tf.transpose(h_i_tilda))),axis=1)


    def normalize(self,index):
        self.hiddens[:,index,:]=tf.math.l2_normalize(self.hiddens[:,index,:],axis=2)

    @property
    def variables(self):
        return self._variables

    def call(self, encoded_sents, indices):
        current_hiddens=tf.gather(self.hiddens,indices)
        current_keys=tf.gather(self.keys,indices)

        if current_hiddens.shape is not current_keys.shape:
            raise AttributeError('hiddens and kes must have same shape')

        for index in indices:
            g_i=self.get_gate(encoded_sents,index)
            self.update_hidden(g_i,index,encoded_sents)
            self.normalize_hidden(index)




class StaticRecurrentEntNet(tf.keras.Model):
    def __init__(self,embedding_matrix, entity_num, entity_embedding_dim, rnn_hidden_size,vocab_size,start_token, name=None):

        if name is None:
            name='staticRecurrentEntNet'
        super().__init__(name=name)
        self.embedding_matrix=embedding_matrix
        'embedding_matrix shape: [vocab_size, embedding_dim]'
        'I assume the last row is an all zero vector for fake words with index embedding_dim.shape[0]'
        self.embedding_dim=self.embedding_matrix.shape[1]
        # self.add_zero_vector()
        self.entity_num=entity_num
        self.entity_embedding_dim=entity_embedding_dim
        self.hidden_Size=rnn_hidden_size
        self.vocab_size=vocab_size
        self.start_token=start_token
        'start_token shape:[1,enbedding_dim]'
        self.total_loss=0
        self.optimizer=tf.train.AdamOptimizer()


        ' defining submodules '
        self.sent_encoder_module=Sent_encoder()
        self.update_entity_module=Update_entity(self.entity_num,self.entity_embedding_dim)
        self.lstm=tf.keras.layers.LSTM(self.hidden_Size,return_sequences=True,return_state=True)
        self.decoder_dense=tf.keras.layers.Dense(self.vocab_size,activation='softmax')
        self.entity_dense=tf.keras.layers.Dense(self.hidden_Size)

        self.entity_attn_matrix=tf.Variable(tf.random_normal([self.hidden_Size,self.embedding_dim]),name='entity_attn_matrix')


    @property
    def variables(self):
        return self.variables+self.entity_attn_matrix+self.lstm.variables+self.decoder_dense.variables+self.entity_dense.variables+self.update_entity_module.variables


    def attention_hiddens(self,query,keys):
        '''
        Description:
            attention on keys with given quey, value is equal to keys

        Args:
            inputs: query shape: [curr_prgrphs_num, 1, hiddens_size]
                    keys shape: [curr_prgrphs_num, prev_hiddens_num, hidden_size]
            output shape: [curr_prgrphs_num, 1, hidden_size]
        '''

        coefs=tf.nn.softmax(tf.reduce_sum(tf.multiply(query, keys),axis=2),axis=2)
        ''' coef shape : [curr_prgrphs_num, prev_hiddens_num, 1]'''
        return tf.reduce_sum(tf.multiply(coefs, keys),axis=1)

    def attention_entities(self,query,entities):
        '''
        Description:
            attention on entities

        Arges:
            inputs: query shape: [curr_prgrphs_num, 1, hiddens_size]
                    entities shape: [curr_prgrphs_num, entities_num, entitiy_embedding_dim]
            output shape: [curr_prgrphs_num, 1, hidden_size]
        '''
        curr_prgrph_nums=query.shape[0]
        tiles_matrix=tf.tile(tf.expand_dims(tf.matmul(tf.reshape(query,[-1,query.shpae[2]]),self.entity_attn_matrix),axis=0)
                             ,[1,entities.shape[1],1])
        ' tiles_matrix shape: [curr_prgrphs_num, entities_num, entity_embedding_dim]'
        return tf.reduce_sum(tf.multiply(tiles_matrix,entities),axis=1)


    def calculate_hidden(self,curr_sents_prev_hiddens,entities):
        '''
        Description:
            calculates current hidden state that should be fed to lstm for predicting the next word, with attention on previous hidden states, THEN entities

        Args:
            inputs: curr_sents_prev_hiddens shape: [curr_prgrphs_num, prev_hiddens]
                    entities: [curr_prgrphs_num, entities_num, entity_embedding_dim]
            output shape: [curr_prgrphs_num, 1, hidden_size]

        '''

        '''
        attention on hidden states:
            query: last column (last hidden_state)
            key and value: prev_columns
        '''
        attn_hiddens_output=self.attention_hiddens(curr_sents_prev_hiddens[:,curr_sents_prev_hiddens.shape[1]-1,:], curr_sents_prev_hiddens[:,:curr_sents_prev_hiddens.shape[1],:])
        attn_entities_output=self.attention_entites(attn_hiddens_output,entities)
        return self.entity_dense(attn_entities_output)




    def calculate_loss(self,outputs,lstm_targets):
        loss=tf.nn.softmax_cross_entropy_with_logits(labels=lstm_targets, logits=outputs)
        return tf.reduce_mean(loss)

    # def add_zero_vector(self):
    #     embedding_dim=self.embedding_matrix.shape[1]
    #     self.embedding_matrix=tf.concat([self.embedding_matrix,tf.zeros([1,embedding_dim])],axis=0)

    def get_embeddings(self, first_prgrph, second_prgrph):
        return tf.nn.embedding_lookup(self.embedding_matrix,first_prgrph), tf.nn.embedding_lookup(self.embedding_matrix,second_prgrph)

    def call(self,first_prgrph,second_prgrph,entity_keys,p1_mask,p2_mask):
        '''
        args:
            inputs: first_prgrph, second_prgrph shape: [batch_size, max_sent_num, max_sent_len]
                    * I assume that fake words have index equal to embedding_matrix.shape[0]
                    entity_keys : initialized entity keys of shape : [batch_size, entity_num, entity_embedding_dim] , entity_embedding_dim=embedding_dim for now
                    p1_mask : mask for first_prgrph, shape=[batch_size, max_sent_num, max_sent_len]
        '''

        batch_size = first_prgrph.shape[0]
        max_sent_num = first_prgrph.shape[1]
        max_sent_len = first_prgrph.shape[2]

        first_prgrph_embeddings, second_prgrph_embeddings=self.get_embeddings(first_prgrph,second_prgrph)
        'first_prgrph_embeddings and second_prgrph_embeddings shape: [batch_size, max_sent_num, max_sent_len, embedding_dim]'

        '''
        TASK 1 : encoding first_prgrph to update entities
        '''

        self.update_entity_module.initialize_hidden(batch_size)
        self.update_entity_module.assign_keys(entity_keys)

        for i in range(max_sent_num):
            ''' to see which sentences are available '''
            indices=tf.where(p1_mask[:,i,0])
            current_sents=tf.gather(first_prgrph_embeddings,indices)[:,i,:,:]
            encoded_sents=self.sent_encoder_module(current_sents)
            self.update_entity_module(encoded_sents,indices)


        '''
        TASK 2 : language model on second prgrph
        '''

        ' stores previous hidden_states of the lstm for each prgrph '
        hidden_states=tf.zeros([batch_size,max_sent_num*max_sent_len,self.hidden_Size])
        with tf.GradientTape() as tape:
            for i in range(max_sent_num):
                current_sents_indices=tf.where(p2_mask[:,i,0])
                for j in range(max_sent_len):
                    ' indices of available paragraphs'
                    indices=tf.where(p2_mask[:,i,j])
                    if j==0:
                        lstm_inputs=tf.multiply(self.start_token,tf.zeros([batch_size,self.embedding_dim],dtype=tf.int32))
                    else:
                        lstm_inputs=tf.squeeze(tf.gather(second_prgrph_embeddings[:,i,j-1,:],indices))

                    lstm_targets=tf.squeeze(tf.gather(second_prgrph[:,i,j],indices))
                    if i*max_sent_len+j==0:
                        curr_sents_curr_hidden=tf.zeros([batch_size,self.hidden_Size],tf.float32)
                        curr_sents_cell_state=tf.zeros([batch_size,self.hidden_Size],tf.float32)
                    else:
                        curr_sents_prev_hiddens=tf.gather(hidden_states[:,:i*max_sent_len+j,:],indices)
                        curr_sents_entities=tf.gather(self.update_entity_module.hiddens, indices)
                        curr_sents_curr_hidden=self.calculate_hidden(curr_sents_prev_hiddens,curr_sents_entities)
                    output,next_hidden,curr_sents_cell_state=self.lstm(lstm_inputs,initial_state=[tf.squeeze(curr_sents_curr_hidden,axis=1),curr_sents_cell_state])
                    'output shape:[batch_size, hidden_size] here, output is equal to next_hidden'
                    hidden_states[:,i*max_sent_len+j,:]=tf.expand_dims(tf.scatter_nd(indices,next_hidden,shape=[batch_size,self.hidden_Size]),axis=1)
                    output=self.decoder_dense(output)
                    loss=self.calculate_loss(output,lstm_targets)
                    self.total_loss+=loss
                    gradients=tape.gradient(loss,self.variables)
                    self.optimizer.apply_gradients(zip(gradients, self.variables))

                current_sents = tf.gather(second_prgrph, current_sents_indices)[:, i, :, :]
                encoded_sents = self.sent_encoder_module(current_sents)
                self.update_entity_module(encoded_sents, current_sents_indices)


# if __name__=='__main__':
#     embedding
#     print('hi!')
