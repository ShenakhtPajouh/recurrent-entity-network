import tensorflow as tf
import numpy as np
import tensorflow.contrib.seq2seq as seq2seq
import os
import staticRecurrentEntNet
import tensorflow.contrib.eager as tfe
import time



def calculate_loss(outputs, lstm_targets,mask):
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
    print('outputs shape:',outputs.shape)
    print('output dtype:',outputs.dtype)
    print('targets dtype:',lstm_targets.dtype)
    print('mask dtype:',mask.dtype)
    loss=seq2seq.sequence_loss(logits=outputs,targets=lstm_targets,weights=tf.cast(mask,tf.float32),average_across_batch=True,average_across_timesteps=True)
    return loss

def train(model_name, embedding_matrix, entity_num, entity_embedding_dim, rnn_hidden_size, vocab_size, start_token, max_sent_num,
          p1, p1_mask, p2, p2_mask, entity_keys, save_path, learning_rate):
    static_recur_entNet = staticRecurrentEntNet.StaticRecurrentEntNet(embedding_matrix=embedding_matrix, entity_num=entity_num,
                                                entity_embedding_dim=entity_embedding_dim
                                                , rnn_hidden_size=rnn_hidden_size, vocab_size=vocab_size, start_token=start_token,
                                                name=model_name, max_sent_num=max_sent_num)

    encode_inputs = ['encode', entity_keys, p1, p1_mask]
    entities = static_recur_entNet(inputs=encode_inputs)
    # print(entities)
    decode_train_inputs = ['decode_train', False , entities, p2, p2_mask, vocab_size]

    # total_loss = 0
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        # print('in for,GD')
    with tf.GradientTape() as tape:
        print('in tf.GradientTape')
        start=time.time()
        output,lstm_targets,mask =static_recur_entNet(decode_train_inputs)
        end=time.time()
        training_time=end-start
        print('training_time:',training_time)
        loss = calculate_loss(output, lstm_targets,mask)
        gradients = tape.gradient(loss, static_recur_entNet.variables)
        # print('trainable_Variable:',static_recur_entNet.trainable_variables)
        # print('gradients:',gradients)
        # print('gradinets[0]',gradients[0])
        print(static_recur_entNet.variables)
    optimizer.apply_gradients(zip(gradients, static_recur_entNet.variables),global_step=tf.train.get_or_create_global_step())

    'saving the model'
    checkpoint_dir=save_path
    os.makedirs(checkpoint_dir,exist_ok=True)
    checkpoint_prefix=os.path.join(checkpoint_dir,'ckpt')
    # root=tf.train.Checkpoint(model=static_recur_entNet)
    # root.save(checkpoint_prefix)
    tfe.Saver(static_recur_entNet.variables).save(checkpoint_prefix)

    return loss


if __name__ == '__main__':
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
    # static_recur_entNet=StaticRecurrentEntNet(embedding_matrix=embedding,entity_num=10,entity_embedding_dim=20
    #                                           ,rnn_hidden_size=15,vocab_size=10,start_token=6,name='static_recur_entNet',max_sent_num=3)

    # encode_inputs=['encode',entity_keys,p1,p1_mask]
    # entities=static_recur_entNet(inputs=encode_inputs)
    # print(entities)

    total_loss = train(model_name='static_recur_entNet', embedding_matrix=embedding, entity_num=entity_keys.shape[1],
                             entity_embedding_dim=entity_keys.shape[2],
                             rnn_hidden_size=15, vocab_size=10, start_token=6, max_sent_num=p1.shape[1], p1=p1, p1_mask=p1_mask,
                             p2=p2, p2_mask=p2_mask, entity_keys=entity_keys,
                             save_path='./static_ent_net_model', learning_rate=0.01)

    print('hi!')








