import tensorflow as tf
from tensorflow.contrib import autograph


class EntityCell(tf.keras.layers.Layer):
    """
    Entity Cell.
    call with inputs and keys
    """
    def __init__(self, num_units, name=None, initializer=None,**kwargs):
        """
        Args:
            num_units: state_dim
        """
        raise NotImplementedError

    def build(self, input_shape):
        """
        Defining variables with self.add_weight
        """
        raise NotImplementedError

    def call(self, inputs, prev_state, keys, use_shared_keys=False, **kwargs):
        """

        Args:
            inputs: tensor of shape [batch_size, dim]
            prev_state: tensor of shape [batch_size, key_num, dim]
            keys: tensor of shape [batch_size, key_num, dim] if use_shared_keys is False and
                  [key_num, dim] if use_shared_keys is True
            use_shared_keys: if it is True, it use shared keys for all samples.

        Returns:
            next_state: tensor of shape [batch_size, key_num, dim]

        """
        raise NotImplementedError


    def get_intial_state(self):
        raise NotImplementedError

    def __call__(self, inputs, prev_state, keys, use_shared_keys=False, **kwargs):
        """
        Do not fill this one
        """
        return super().__call__(inputs=inputs, prev_state=prev_state, keys=keys,
                                use_shared_keys=use_shared_keys, **kwargs)

@autograph.convert()
def simple_entity_network(entity_cell, inputs, keys, mask_inputs=None,
                          initial_entity_hidden_state=None,
                          use_shared_keys=False, return_last=True):
    """
    Args:
        entity_cell: the EntityCell
        inputs: a tensor of shape [batch_size, seq_length, dim]
        keys: tensor of shape [batch_size, key_num, dim] if use_shared_keys is False and
                  [key_num, dim] if use_shared_keys is True
        use_shared_keys: if it is True, it use shared keys for all samples.
        mask_inputs: tensor of shape [batch_size, seq_length] and type tf.bool
        initial_entity_hidden_state: a tensor of shape [batch_size, key_num, dim]
        return_last: if it is True, it returns the last state, else returns all states

    Returns:
        if return_last = True then a tensor of shape [batch_size, key_num, dim] else shape of
                         [batch_size, seq_length, key_num, dim]
    """
    raise NotImplementedError


@autograph.convert()
def rnn_entity_network_encoder(entity_cell, rnn_cell, inputs, keys, mask_inputs=None,
                                 initial_hidden_state=None ,
                                 initial_entity_hidden_state=None, update_positions=None, use_shared_keys=False,
                                 return_last=True, self_attention=False):
    """


    """
    raise NotImplementedError


@autograph.convert()
def rnn_entity_network_decoder(entity_cell, rnn_cell, softmax_layer, embedding_layer, keys, training,
                               initial_hidden_state=None, initial_entity_hidden_state=None,
                               labels=None,
                               num_inputs=None, num_keys=None, encoder_hidden_states=None,
                               update_positions=None, use_shared_keys=False, return_last=True,
                               attenton=False, self_attention=False):
    """

    Args:
        entity_cell: EntityCell
        rnn_cell: RNNCell
        softmax_layer: softmax layer
        embedding_layer: embedding layer
        keys: either a tensor of shape [batch_size, key_num, dim] or [key_num, dim] depending on use_shared_keys
        training: boolean ...
    :param initial_hidden_state:
    :param initial_entity_hidden_state:
    :param labels:
    :param num_inputs:
    :param num_keys:
    :param encoder_hidden_states:
    :param update_positions:
    :param use_shared_keys:
    :param return_last:
    :param attenton:
    :param self_attention:
    :return:
    """
    raise NotImplementedError

"""

"""


class BasicRecurrentEntityEncoder(tf.keras.Model):
    def __init__(self, num_units=None, entity_cell=None, name=None, **kwargs):
        raise NotImplementedError

    def call(self, inputs, keys, num_inputs=None, initial_entity_hidden_state=None,
                          use_shared_keys=False, return_last=True , **kwargs):
        raise NotImplementedError

class RNNRecurrentEntityEncoder(tf.keras.Model):
    def __init__(self, num_units=None, entity_cell=None, rnn_cell=None, name=None, **kwargs):
        raise NotImplementedError

    def call(self, inputs, keys, num_inputs=None, num_keys=None,
                                 initial_hidden_state=None ,
                                 initial_entity_hidden_state=None, update_positions=None, use_shared_keys=False,
                                 return_last=True, self_attention=False, **kwargs):
        raise NotImplementedError


class RNNRecurrentEntitiyDecoder(tf.keras.Model):
    def __init__(self, embedding_layer, vocab_size=None, num_units=None, entity_cell=None, rnn_cell=None,
                 softmax_layer=None, name=None, **kwargs):
        raise NotImplementedError

    def build(self, input_shape):
        raise NotImplementedError

    def call(self, inputs, keys, training, initial_hidden_state=None,
                               encoder_hidden_states=None,
                               labels=None,
                               num_inputs=None, num_keys=None,
                               update_positions=None, use_shared_keys=False,
                               return_last=True,
                               attention=False, self_attention=False):
        raise NotImplementedError













