import tensorflow as tf
import tensorflow.contrib.eager as tfe
import os
import Model
import numpy as np


def test(embedding_matrix, entity_num, entity_embedding_dim, rnn_hidden_size, vocab_size, start_token,
         max_sent_num, p1, p1_mask, entity_keys, keys_mask, encoder_path, decoder_path, eos_ind):
    encoder = Model.BasicRecurrentEntityEncoder(embedding_matrix=embedding_matrix, max_entity_num=entity_num,
                                                entity_embedding_dim=entity_embedding_dim)

    temp_entity_cell, temp_entities = encoder([p1, p1_mask], entity_keys)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        temp_entitiess=sess.run(temp_entities)

    print("temp_entities shape", temp_entitiess.shape)

    decoder = Model.RNNRecurrentEntitiyDecoder(embedding_matrix=embedding_matrix, rnn_hidden_size=rnn_hidden_size,
                                               entity_cell=temp_entity_cell,
                                               vocab_size=vocab_size, max_sent_num=max_sent_num,
                                               entity_embedding_dim=entity_embedding_dim)

    ' training the model for one step just to initialize all variables '
    decoder_inputs_train = [True, temp_entitiess, vocab_size, start_token]
    labels = [p2, p2_mask]
    final_outputt=decoder(inputs=decoder_inputs_train, keys=entity_keys,
            keys_mask=keys_mask, training=True, labels=labels)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        final_outputtt=sess.run(final_outputt)
        # print("final output train once:", final_outputtt[0])
    max_sent_num = tf.shape(p1)[1]
    max_sent_len = tf.shape(p1)[2]

    ' restoring saved models '
    checkpoint_dir_encoder = encoder_path
    os.makedirs(checkpoint_dir_encoder, exist_ok=True)
    checkpoint_prefix_encoder = os.path.join(checkpoint_dir_encoder, 'ckpt')
    tfe.Saver(encoder.variables).restore(checkpoint_prefix_encoder)

    checkpoint_dir_decoder = decoder_path
    os.makedirs(checkpoint_dir_decoder, exist_ok=True)
    checkpoint_prefix_decoder = os.path.join(checkpoint_dir_decoder, 'ckpt')
    tfe.Saver(decoder.variables).restore(checkpoint_prefix_decoder)

    entity_cell, entity_hiddens = encoder([p1, p1_mask], entity_keys)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        entity_hiddenss=sess.run(entity_hiddens)
    # print("entity_hiddens shape:", entity_hiddens)
    decoder_inputs_test = [entity_hiddenss, max_sent_num, max_sent_len, eos_ind, start_token]
    generated_prgrph, second_prgrph_entities = decoder(inputs=decoder_inputs_test, keys=entity_keys, keys_mask=keys_mask,
                               training=False, return_last=False)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        generated_prgrphh, second_prgrph_entitiess = sess.run([generated_prgrph, second_prgrph_entities])
    print(generated_prgrphh)
    print(second_prgrph_entitiess.shape)


if __name__ == '__main__':
    # tf.enable_eager_execution()
    embedding = tf.random_normal([10, 20])
    p1 = np.asarray([[[0, 1, 9, 9], [2, 3, 4, 9], [1, 2, 3, 4]],
                     [[0, 1, 9, 9], [2, 3, 4, 7], [1, 2, 3, 4]]])
    # print(p1.shape)
    p1_mask = np.asarray([[[True, True, False, False], [True, True, True, False], [True, True, True, True]],
                          [[True, True, False, False], [True, True, True, True], [True, True, True, True]]])
    p2 = np.asarray([[[3, 1, 5, 9], [2, 3, 9, 9], [7, 2, 3, 5]],
                     [[3, 1, 5, 9], [2, 3, 6, 9], [7, 2, 3, 9]]])
    p2_mask = np.asarray([[[True, True, True, False], [True, True, False, False], [True, True, True, True]],
                          [[True, True, True, False], [True, True, True, False], [True, True, True, False]]])
    entity_keys = tf.random_normal([2, 10, 20])
    keys_mask = tf.expand_dims(tf.scatter_nd([[0], [1], [2]], [True, True, True], [10]), axis=0)
    keys_mask = tf.concat([keys_mask, tf.expand_dims(
        tf.scatter_nd([[0], [1], [2], [3], [4]], [True, True, True, True, True], [10]), axis=0)], axis=0)

    test(embedding_matrix=embedding, entity_num=10,
         entity_embedding_dim=20,
         rnn_hidden_size=15, vocab_size=10, start_token=6, max_sent_num=tf.shape(p1)[1], p1=p1, p1_mask=p1_mask,
         entity_keys=entity_keys, keys_mask=keys_mask,
         encoder_path='./encoder', decoder_path='./decoder', eos_ind=9)

