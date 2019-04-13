import tensorflow as tf
import numpy as np
import tensorflow.contrib.seq2seq as seq2seq
import os
import Model
import tensorflow.contrib.eager as tfe
import time
import elmo.ELMo.keras_model as keras_model
import elmo.HP as HP
import elmo.encoder as elmo_encoder
from elmo.ELMo.data import Batcher, TokenBatcher


def calculate_loss(outputs, lstm_targets, mask):
    """
    Args:
        inputs: outputs shape : [batch_size,max_sents_num*max_sents_len, vocab_size]
                lstm_targets shape : [batch_size, max_sents_num*max_sents_len]
                mask : [batch_size, max_sents_num*max_sents_len]

    """
    # one_hot_labels = tf.one_hot(lstm_targets, outputs.shape[1])
    # print('outpus shape:', outputs.shape, outputs)
    # print('one_hot_labels shape:', one_hot_labels.shape)
    # loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot_labels, logits=outputs)
    # print('loss', loss)
    print('IN LOSS FUNCTION')
    print('outputs shape:', outputs.shape)
    print('output dtype:', outputs.dtype)
    print('targets dtype:', lstm_targets.dtype)
    print('mask dtype:', mask.dtype)
    loss = seq2seq.sequence_loss(logits=outputs, targets=lstm_targets, weights=tf.cast(mask, tf.float32),
                                 average_across_batch=True, average_across_timesteps=True)
    return loss


def train(entity_num, entity_embedding_dim, rnn_hidden_size, start_token,
          max_sent_num, vocab_size, p1, p2, label_mask, entity_keys, keys_mask, encoder_save_path, decoder_save_path,
          learning_rate):

    batcher = TokenBatcher(HP.vocab_file)
    inputs, specifier, rnn_mask, indices, sents_mask, end_of_sentences = elmo_encoder.input_provider(
        p1, batcher, max_sent_num=max_sent_num, use_char_input=False)

    print(type(inputs))
    print(type(specifier))
    # print(rnn_mask.shape)
    print(type(indices))
    print(type(sents_mask))
    print(type(end_of_sentences))
    print(type(entity_keys))

    encoder = Model.RNNRecurrentEntityEncoder(max_entity_num=entity_num, entity_embedding_dim=entity_embedding_dim)
    entity_keys_p = tf.placeholder(shape=[None, entity_num, entity_embedding_dim], dtype=tf.float32)

    inputs_p=tf.placeholder(shape=[None, None], dtype=tf.int64)
    specifier_p= tf.placeholder(shape=[None, None, 2], dtype=tf.int64)
    indices_p=tf.placeholder(shape=[None, 2], dtype=tf.int64)
    sents_mask_p=tf.placeholder(shape=[None,None],dtype=tf.int64)
    end_of_sentences_p=tf.placeholder(shape=[None, 2], dtype=tf.int64)
    entity_cell, first_prgrph_entitiess = encoder(
        [inputs_p, specifier_p, indices_p, sents_mask_p, end_of_sentences_p, max_sent_num], entity_keys_p)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        first_prgrph_entities = sess.run(first_prgrph_entitiess,
                                         feed_dict={inputs_p:inputs, specifier_p:specifier, indices_p:indices, sents_mask_p:sents_mask,
                                                    end_of_sentences_p:end_of_sentences, entity_keys_p:entity_keys})

        print("first_prgrph_entities shape", first_prgrph_entities.shape)

        decoder = Model.RNNRecurrentEntitiyDecoder_sent_gen(rnn_hidden_size=rnn_hidden_size,
                                                            entity_cell=entity_cell,
                                                            entity_embedding_dim=entity_embedding_dim,
                                                            vocab_size=vocab_size)
        if len(first_prgrph_entities.shape) == 3:
            decoder_inputs_train = [False, first_prgrph_entities,vocab_size, start_token]
        else:
            'return_last in encoder has been false'
            decoder_inputs_train = [False, first_prgrph_entities[:, -1, :, :], vocab_size, start_token]

        l_inputs_p = tf.placeholder(shape=[None, None], dtype=tf.int64)
        l_specifier_p = tf.placeholder(shape=[None, None, 2], dtype=tf.int64)
        l_indices_p = tf.placeholder(shape=[None, 2], dtype=tf.int64)
        l_end_of_sentences_p = tf.placeholder(shape=[None, 2], dtype=tf.int64)
        _, encoded_label = tf.expand_dims(Model.rnn_encoder([l_inputs_p, l_specifier_p, l_end_of_sentences_p, l_indices_p, 1]),axis=1)
        labels = [encoded_label, label_mask]
        print(encoded_label.shape, label_mask.shape)

        final_output_tt = decoder(inputs=decoder_inputs_train, keys=entity_keys_p,
                                  keys_mask=keys_mask, training=True, labels=labels)

        l_inputs, l_specifier, l_rnn_mask, l_indices, l_sents_mask, l_end_of_sentences = elmo_encoder.input_provider(
            p2, batcher, max_sent_num=max_sent_num, use_char_input=False,get_embedding=True)



        with tf.Session() as sess:
            print("inside decode session")
            sess.run(tf.global_variables_initializer())
            final_output_tt = sess.run(final_output_tt, feed_dict={l_inputs_p:l_inputs, l_specifier_p:l_specifier,
                                                                   l_indices_p: l_indices, l_end_of_sentences_p: l_end_of_sentences,
                                                                   entity_keys_p: entity_keys})
            print(final_output_tt[0][0])
            print("decode_train worked!")

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        variables = decoder.variables + encoder.variables

        # # print('in for,GD')
        # with tf.GradientTape() as tape:
        #     print('in tf.GradientTape')
        #     start = time.time()
        #     output, lstm_targets, mask = decoder(inputs=decoder_inputs_train, keys=entity_keys,
        #                                          keys_mask=keys_mask, training=True, labels=labels)
        #     end = time.time()
        #     training_time = end - start
        #     print('training_time:', training_time)
        #     loss = calculate_loss(output, lstm_targets, mask)
        #     gradients = tape.gradient(loss, variables)
        #     # print('trainable_Variable:',static_recur_entNet.trainable_variables)
        #     # print('gradients:',gradients)
        #     # print('gradinets[0]',gradients[0])
        #     print("decoder variables")
        #     print(len(decoder.variables))
        #     print("encoder variables")
        #     print(len(encoder.variables))
        #     print("variables")
        #     # print(variables)
        # optimizer.apply_gradients(zip(gradients, variables), global_step=tf.train.get_or_create_global_step())
        #
        # 'saving the encoder'
        # checkpoint_dir_encoder = encoder_save_path
        # os.makedirs(checkpoint_dir_encoder, exist_ok=True)
        # checkpoint_prefix_encoder = os.path.join(checkpoint_dir_encoder, 'ckpt')
        # tfe.Saver(encoder.variables).save(checkpoint_prefix_encoder)
        #
        # 'saving the decoder'
        # checkpoint_dir_decoder = decoder_save_path
        # os.makedirs(checkpoint_dir_decoder, exist_ok=True)
        # checkpoint_prefix_decoder = os.path.join(checkpoint_dir_decoder, 'ckpt')
        # tfe.Saver(decoder.variables).save(checkpoint_prefix_decoder)
        #
        # return loss

if __name__ == '__main__':
        # tf.enable_eager_execution()
        # keras_model.dump_token_embeddings(
        #     HP.vocab_file, HP.option_file, HP.weight_file, HP.token_embedding_file
        # )
        # tf.reset_default_graph()
        embedding = tf.random_normal([10, 20])
        # p1 = np.asarray([[[0, 1, 9, 9], [2, 3, 4, 9], [1, 2, 3, 4]],
        #                  [[0, 1, 9, 9], [2, 3, 4, 7], [1, 2, 3, 4]]])
        # print("P1 shape", p1.shape)
        # p1_mask = np.asarray([[[True, True, False, False], [True, True, True, False], [True, True, True, True]],
        #                       [[True, True, False, False], [True, True, True, True], [True, True, True, True]]])
        # p2 = np.asarray([[[3, 1, 5, 9], [2, 3, 9, 9], [7, 2, 3, 5]],
        #                  [[3, 1, 5, 9], [2, 3, 6, 9], [7, 2, 3, 9]]])
        # p2_mask = np.asarray([[[True, True, True, False], [True, True, False, False], [True, True, True, True]],
        #                       [[True, True, True, False], [True, True, True, False], [True, True, True, False]]])
        entity_keys = np.random.normal(size=[2, 10, 512])
        keys_mask = tf.expand_dims(tf.scatter_nd([[0], [1], [2]], [True, True, True], [10]), axis=0)
        keys_mask = tf.concat([keys_mask, tf.expand_dims(
            tf.scatter_nd([[0], [1], [2], [3], [4]], [True, True, True, True, True], [10]), axis=0)], axis=0)

        # static_recur_entNet=StaticRecurrentEntNet(embedding_matrix=embedding,entity_num=10,entity_embedding_dim=20
        #                                           ,rnn_hidden_size=15,vocab_size=10,start_token=6,name='static_recur_entNet',max_sent_num=3)

        # encode_inputs=['encode',entity_keys,p1,p1_mask]
        # entities=static_recur_entNet(inputs=encode_inputs)
        # print(entities)

        p1 = ['Pretrained biLMs compute representations useful for NLP tasks . it\'s amazing .',
                         'They give state of the art performance for many tasks .']

        p2 = ['outside is raining.', 'what a wonderful day.']
        label_mask= np.expand_dims(np.asarray([[True,True,True,True,False],[True,True,True,True,True]]),axis=1)

        total_loss = train(entity_num=10,
                           entity_embedding_dim=512,
                           rnn_hidden_size=15, vocab_size=10, start_token=0, max_sent_num=3, p1=p1,
                           p2=p2, label_mask=label_mask, entity_keys=entity_keys, keys_mask=keys_mask,
                           encoder_save_path='./encoder', decoder_save_path='./decoder',
                           learning_rate=0.01)

        print('hi!')
