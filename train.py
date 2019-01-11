import tensorflow as tf
import numpy as np
import staticRecurrentEntNet


def calculate_loss(outputs, lstm_targets):
    """
    Args:
        inputs: outputs shape : [curr_prgrphs_num, vocab_size]
                lstm_targets shape : [curr_prgrphs_num]
    """
    one_hot_labels = tf.one_hot(lstm_targets, outputs.shape[1])
    print('outpus shape:', outputs.shape, outputs)
    print('one_hot_labels shape:', one_hot_labels.shape)
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot_labels, logits=outputs)
    print('loss', loss)
    return tf.reduce_mean(loss)

def train(model_name, embedding_matrix, entity_num, entity_embedding_dim, rnn_hidden_size, vocab_size, start_token, max_sent_num,
          p1, p1_mask, p2, p2_mask, entity_keys, save_path, learning_rate):
    static_recur_entNet = staticRecurrentEntNet.StaticRecurrentEntNet(embedding_matrix=embedding_matrix, entity_num=entity_num,
                                                entity_embedding_dim=entity_embedding_dim
                                                , rnn_hidden_size=rnn_hidden_size, vocab_size=vocab_size, start_token=start_token,
                                                name=model_name, max_sent_num=max_sent_num)

    encode_inputs = ['encode', entity_keys, p1, p1_mask]
    entities = static_recur_entNet(inputs=encode_inputs)
    # print(entities)
    decode_train_inputs = ['decode_train', entities, p2, p2_mask]

    total_loss = 0
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    for output,lstm_targets in static_recur_entNet(decode_train_inputs):
        print('in for,GD')
        with tf.GradientTape() as tape:
            loss = calculate_loss(output, lstm_targets)
            total_loss += loss
            gradients = tape.gradient(loss, static_recur_entNet.variables)
        # print('trainable_Variable:',static_recur_entNet.trainable_variables)
        print('gradients:',gradients)
        optimizer.apply_gradients(zip(gradients, static_recur_entNet.variables),global_step=tf.train.get_or_create_global_step())


    return total_loss







