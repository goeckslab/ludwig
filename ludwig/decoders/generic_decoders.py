#! /usr/bin/env python
# coding=utf-8
# Copyright (c) 2019 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import logging
from functools import partial

# import tensorflow as tf
# from tensorflow.keras.layers import Activation
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.layers import Layer

import torch
from torch.nn import Linear, Module
from ludwig.utils.torch_utils import LudwigModule

from ludwig.utils.torch_utils import (LudwigModule, initializers, activations, reg_loss)

from ludwig.constants import LOSS, TYPE


logger = logging.getLogger(__name__)


class Regressor(LudwigModule):

    def __init__(
            self,
            input_size,
            use_bias=True,
            #kernel_initializer='glorot_uniform',
            #bias_initializer='zeros',
            weights_initializer='xavier_uniform',
            bias_initializer='zeros',
            #kernel_regularizer=None,
            weights_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            activation=None,
            **kwargs
    ):
        super().__init__()
        self.name = "Regressor"
        logger.debug(' {}'.format(self.name))

        #logger.debug('  Dense')
        logger.debug('  Linear')

        '''
        self.dense = Dense(
            1,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            activation=activation
        )
        '''
        self.dense = Linear(
                in_features=input_size,
                out_features=1,
                bias=use_bias
        )
        weights_initializer = initializers[weights_initializer]
        weights_initializer(self.dense.weight)

        bias_initializer = initializers[bias_initializer]
        bias_initializer(self.dense.bias)

        if weights_regularizer:
            self.add_loss(lambda: reg_loss(fc.weight, weights_regularizer))
        if bias_regularizer:
            self.add_loss(lambda: reg_loss(fc.bias, bias_regularizer))
        if activity_regularizer:
            # Handle in forward call
            self.add_loss(lambda: self.activation_loss)

        self.activity_regularizer = activity_regularizer

    def forward(self, inputs, **kwargs):
        batch_size = inputs.shape[0]
        output = torch.squeeze(self.dense(inputs), axis=-1)
        if self.activity_regularizer:
            self.activation_loss = reg_loss(output, self.activity_regularizer)/batch_size

        return output


class Projector(Module):

    def __init__(
            self,
            vector_size,
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            activation=None,
            clip=None,
            **kwargs
    ):
        super().__init__()
        logger.debug(' {}'.format(self.name))

        logger.debug('  Dense')
        self.dense = Dense(
            vector_size,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer
        )

        self.activation = Activation(activation)

        if clip is not None:
            if isinstance(clip, (list, tuple)) and len(clip) == 2:
                self.clip = partial(
                    tf.clip_by_value,
                    clip_value_min=clip[0],
                    clip_value_max=clip[1]
                )
            else:
                raise ValueError(
                    'The clip parameter of {} is {}. '
                    'It must be a list or a tuple of length 2.'.format(
                        self.feature_name,
                        self.clip
                    )
                )
        else:
            self.clip = None

    def call(self, inputs, **kwargs):
        values = self.activation(self.dense(inputs))
        if self.clip:
            values = self.clip(values)
        return values


class Classifier(Module):

    def __init__(
            self,
            num_classes,
            use_bias=True,
            weights_initializer='glorot_uniform',
            bias_initializer='zeros',
            weights_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            **kwargs
    ):
        super().__init__()
        logger.debug(' {}'.format(self.name))

        logger.debug('  Dense')
        self.dense = Dense(
            num_classes,
            use_bias=use_bias,
            kernel_initializer=weights_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=weights_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer
        )

        self.sampled_loss = False
        if LOSS in kwargs and TYPE in kwargs[LOSS] and kwargs[LOSS][TYPE] is not None:
            self.sampled_loss = kwargs[LOSS][TYPE].startswith('sampled')

        # this is needed because TF2 initialzies the weights at the first call
        # so the first time we need to compute the full dense,
        # otherwise the weights of the Dense layer would not be initialized
        self.first_call = True

    def call(self, inputs, training=None, **kwargs):
        if training and self.sampled_loss and not self.first_call:
            # this is needed because at training time is the loss is sampled
            # we should not compute the last dense projection,
            # otherwise we defet the purpose of the samples loss
            # which is not to compute the full final projection
            # returning empty tensor to pass graph execution validation test
            return tf.zeros(0)
        else:
            self.first_call = False
            return self.dense(inputs)
