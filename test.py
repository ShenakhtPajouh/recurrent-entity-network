import tensorflow as tf
import tensorflow.contrib.eager as tfe
import os
import staticRecurrentEntNet
import numpy as np

def test(model_name, embedding_matrix, entity_num, entity_embedding_dim, rnn_hidden_size, vocab_size, start_token, max_sent_num,
          p1, p1_mask, p2, p2_mask, entity_keys, save_path, learning_rate):
    static_recur_entNet = staticRecurrentEntNet.StaticRecurrentEntNet(embedding_matrix=embedding_matrix,
                                                                      entity_num=entity_num,
                                                                      entity_embedding_dim=entity_embedding_dim
                                                                      , rnn_hidden_size=rnn_hidden_size,
                                                                      vocab_size=vocab_size, start_token=start_token,
                                                                      name=model_name, max_sent_num=max_sent_num)
    encode_inputs = ['encode', entity_keys, p1, p1_mask]
    entities = static_recur_entNet(inputs=encode_inputs)
    # print(entities)
    decode_train_inputs = ['decode_train', True, entities, p2, p2_mask, vocab_size]
    static_recur_entNet(decode_train_inputs)
    print(static_recur_entNet.variables[0])
    checkpoint_dir = './static_ent_net_model'
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
    # root = tf.train.Checkpoint(model=static_recur_entNet)
    # root.restore(tf.train.latest_checkpoint(checkpoint_dir))
    tfe.Saver(static_recur_entNet.variables).restore(checkpoint_prefix)

    return static_recur_entNet.variables

if __name__=='__main__':
    tf.enable_eager_execution()
    embedding = tf.random_normal([10, 20])
    p1 = np.asarray([[[0, 1, 9, 9], [2, 3, 4, 9], [1, 2, 3, 4]],
                     [[0, 1, 9, 9], [2, 3, 4, 7], [1, 2, 3, 4]]])
    print(p1.shape)
    p1_mask = np.asarray([[[True, True, False, False], [True, True, True, False], [True, True, True, True]],
                          [[True, True, False, False], [True, True, True, True], [True, True, True, True]]])
    p2 = np.asarray([[[3, 1, 5, 9], [2, 3, 9, 9], [7, 2, 3, 5]],
                     [[3, 1, 5, 9], [2, 3, 6, 9], [7, 2, 3, 9]]])
    p2_mask = np.asarray([[[True, True, True, False], [True, True, False, False], [True, True, True, True]],
                          [[True, True, True, False], [True, True, True, False], [True, True, True, False]]])
    entity_keys = tf.random_normal([2, 10, 20])
    variables = test(model_name='static_recur_entNet', embedding_matrix=embedding, entity_num=10,
                             entity_embedding_dim=20,
                             rnn_hidden_size=15, vocab_size=10, start_token=6, max_sent_num=3, p1=p1, p1_mask=p1_mask,
                             p2=p2, p2_mask=p2_mask, entity_keys=entity_keys,
                             save_path='./static_entnet_model.ckpt', learning_rate=0.01)

    print (variables[0])

