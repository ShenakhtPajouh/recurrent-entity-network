import tensorflow as tf

K = tf.keras.backend


class Prgrph_ending_classifier(tf.keras.layers.Layer):

    def __init__(self, max_sent_num, entity_embedding_dim, encoding_dim, name=None):
        if name is None:
            name = 'prgrph_encoding_classifier'

        super().__init__(name)
        self.max_sent_num = max_sent_num
        self.entity_embedding_dim = entity_embedding_dim

        ''' equal to rnn_hidden size'''
        self.encoding_dim = encoding_dim
        p_vec = tf.range(tf.cast(self.max_sent_num, tf.float64), dtype=tf.float64)
        p_vec_tiled = tf.tile(tf.expand_dims(p_vec, axis=1), [1, encoding_dim])
        index_vec = tf.range(self.encoding_dim)
        index_vec_tiled = tf.tile(tf.divide(tf.expand_dims(index_vec, axis=0), self.encoding_dim),
                                  [self.max_sent_num, 1])
        index_vec_tiled = tf.cast(index_vec_tiled, tf.float64)
        # print('pow type:',type(tf.pow(200,index_vec_tiled)[0][0]))
        # print(tf.pow(200.0,index_vec_tiled).dtype,p_vec_tiled.dtype)

        self.position_embeddings = tf.cast(
            tf.sin(tf.divide(p_vec_tiled, tf.pow(tf.cast(200.0, tf.float64), index_vec_tiled))), tf.float32)
        'position_embeddings shape: [max_sent_num, encoding_dim]'

        self.dense = None
        self.entity_attn_matrix = None

        self.built = False

    def build(self, input_shape):
        self.entity_attn_matrix = self.add_weight(name="entity_attn_matrix",
                                                  shape=[self.encoding_dim, self.entity_embedding_dim],
                                                  dtype=tf.float32, trainable=True,
                                                  initializer=tf.keras.initializers.TruncatedNormal())
        self.dense = tf.layers.Dense(1)
        self.built = True

    def attention_prev_sents(self, query, keys):
        '''
        Description:
            attention on keys with given quey, value is equal to keys

        Args:
            inputs: query shape: [curr_prgrphs_num, hiddens_size]
                    keys shape: [curr_prgrphs_num, prev_hiddens_num, hidden_size]
            output shape: [curr_prgrphs_num, hidden_size]
        '''
        values = tf.identity(keys)
        attention_logits = tf.reduce_sum(tf.multiply(tf.expand_dims(query, 1), keys), axis=2)
        attention_coefficients = tf.nn.softmax(attention_logits, axis=1)
        attention = tf.expand_dims(attention_coefficients, -1) * values
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

        print("attention_entities, entities shape: ", tf.shape(entities))

        values = tf.identity(entities)
        query_shape = tf.shape(query)
        entities_shape = tf.shape(entities)
        batch_size = query_shape[0]
        seq_length = entities_shape[1]
        indices = tf.where(keys_mask)
        queries = tf.gather(query, indices[:, 0])
        entities = tf.boolean_mask(entities, keys_mask)
        attention_logits = tf.reduce_sum(tf.multiply(tf.matmul(queries, self.entity_attn_matrix), entities), axis=-1)
        attention_logits = tf.scatter_nd(tf.where(keys_mask), attention_logits, [batch_size, seq_length])
        attention_logits = tf.where(keys_mask, attention_logits, tf.fill([batch_size, seq_length], -20.0))
        attention_coefficients = tf.nn.softmax(attention_logits, axis=1)
        attention = tf.expand_dims(attention_coefficients, -1) * values

        return tf.reduce_sum(attention,axis=1)

    def __call__(self, inputs, *training):
        self.build([0])
        return self.call(inputs, training)

    def call(self, inputs, training=None, mask=None):
        '''
        Description:
            given hidden_states and entities determines whether last hidden_state is for the last sentence of the paragraph or not
            hiddens_states are last hidden states of sentences generated so far, they will be added with position_embeddings

            first, it attends on previous sentences' hidden states with last sentence's hidden state as key, and then attends on entities.
            by applying a dense layer to the result of attention, we will get a tensor of size 1 for paragraph, if it is greater that 0.5,
            we conclude that last hidden state is for the last sentence of the generated paragraph.

        inputs:
            inputs: last_hiddens, shape : [curr_prgrphs_num, sents_num, encoding_dim]
                    entities, shape : [curr_prgrphs_num, entities_num, entity_embedding_dim]

            output: outputs a number in [0,1] for each prgrph, indicating whether it has ended or not
        '''

        if len(inputs) != 3:
            raise AttributeError('expected 3 inputs but ', len(inputs), ' were given')
        last_hiddens, entities, keys_mask = inputs
        curr_prgrphs_num = tf.shape(last_hiddens)[0]
        sents_num = tf.shape(last_hiddens)[1]
        last_hiddenss = last_hiddens + tf.tile(tf.expand_dims(self.position_embeddings[:sents_num, :], axis=0),
                                                [curr_prgrphs_num, 1, 1])
        attn_hiddens_output = self.attention_prev_sents(last_hiddenss[:, tf.shape(last_hiddenss)[1] - 1, :],
                                                        last_hiddenss[:, :tf.shape(last_hiddenss)[1], :])
        attn_entities_output = self.attention_entities(attn_hiddens_output, entities, keys_mask)
        return tf.sigmoid(self.dense(attn_entities_output))
