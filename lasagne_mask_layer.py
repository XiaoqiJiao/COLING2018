import lasagne

class lasagne_mask_layer(lasagne.layers.MergeLayer):

    def __init__(self, incoming, **kwargs):
        super(lasagne_mask_layer, self).__init__(incoming, **kwargs)

    def get_output_shape_for(self, input_shapes):
        
        return input_shapes[0]

    def get_output_for(self, inputs, **kwargs):
        emb = inputs[0]
        mask = inputs[1]
        
        return emb*mask.dimshuffle(0, 1, 'x')
