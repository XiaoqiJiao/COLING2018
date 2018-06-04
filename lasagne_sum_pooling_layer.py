import lasagne
import theano.tensor as T

class lasagne_sum_pooling_layer(lasagne.layers.Layer):

    def __init__(self, incoming, **kwargs):
        super(lasagne_sum_pooling_layer, self).__init__(incoming, **kwargs)

    def get_output_shape_for(self, input_shape):
        
        return (input_shape[0], input_shape[1], input_shape[3])

    def get_output_for(self, input, **kwargs):
        
        return T.sum(input, axis=2, keepdims=False)