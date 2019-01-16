import tensorflow as tf
import tensorflow.contrib.eager as tfe
import os
import staticRecurrentEntNet
import numpy as np

def test(model_name, embedding_matrix, entity_num, entity_embedding_dim, rnn_hidden_size, vocab_size, start_token, max_sent_num,
          p1, p1_mask, entity_keys,model_path,eos_ind):
    static_recur_entNet = staticRecurrentEntNet.StaticRecurrentEntNet(embedding_matrix=embedding_matrix,
                                                                      entity_num=entity_num,
                                                                      entity_embedding_dim=entity_embedding_dim
                                                                      , rnn_hidden_size=rnn_hidden_size,
                                                                      vocab_size=vocab_size, start_token=start_token,
                                                                      name=model_name, max_sent_num=max_sent_num)
    max_sent_len=p1.shape[2]
    ' training the model for one step just to initialize all variables '
    encode_inputs = ['encode', entity_keys, p1, p1_mask]
    entities = static_recur_entNet(inputs=encode_inputs)
    # print(entities)
    decode_train_inputs = ['decode_train', True, entities, p1, p1_mask, vocab_size]
    static_recur_entNet(decode_train_inputs)
    # print(static_recur_entNet.variables[0])

    ' restoring saved model '
    checkpoint_dir = model_path
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
    tfe.Saver(static_recur_entNet.variables).restore(checkpoint_prefix)


    entity_hiddens=static_recur_entNet(inputs=encode_inputs)
    print(entity_hiddens)
    decode_test_input = ['decode_test', entity_hiddens, max_sent_len, eos_ind]
    generated_prgrph=static_recur_entNet(decode_test_input)
    print(generated_prgrph)



if __name__=='__main__':
    tf.enable_eager_execution()
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
    test(model_name='static_recur_entNet', embedding_matrix=embedding, entity_num=entity_keys.shape[1],
                             entity_embedding_dim=entity_keys.shape[2],
                             rnn_hidden_size=15, vocab_size=10, start_token=6, max_sent_num=p1.shape[1], p1=p1, p1_mask=p1_mask,
                             entity_keys=entity_keys,
                             model_path='./static_ent_net_model',eos_ind=9)

    # print (variables[0])

