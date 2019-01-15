import tensorflow as tf
import numpy as np
import tensorflow.contrib.seq2seq as seq2seq
import staticRecurrentEntNet



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
    decode_train_inputs = ['decode_train', entities, p2, p2_mask, vocab_size]

    # total_loss = 0
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        # print('in for,GD')
    with tf.GradientTape() as tape:
        print('in tf.GradientTape')
        output,lstm_targets,mask =static_recur_entNet(decode_train_inputs)
        loss = calculate_loss(output, lstm_targets,mask)
        gradients = tape.gradient(loss, static_recur_entNet.variables)
        # print('trainable_Variable:',static_recur_entNet.trainable_variables)
        # print('gradients:',gradients)
        print('gradinets[0]',gradients[0])
    optimizer.apply_gradients(zip(gradients, static_recur_entNet.variables),global_step=tf.train.get_or_create_global_step())


    return loss







