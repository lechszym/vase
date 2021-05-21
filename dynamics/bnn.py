from __future__ import print_function
import numpy as np
import theano.tensor as T
import lasagne
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.serializable import Serializable
from rllab.misc import ext
import theano

# ----------------
BNN_LAYER_TAG = 'BNNLayer'
USE_REPARAMETRIZATION_TRICK = True
# ----------------


class BNNLayer(lasagne.layers.Layer):
    """Probabilistic layer that uses Gaussian weights.

    Each weight has two parameters: mean and standard deviation (std).
    """

    def __init__(self,
                 incoming,
                 num_units,
                 nonlinearity=lasagne.nonlinearities.rectify,
                 prior_sd=None,
                 **kwargs):
        super(BNNLayer, self).__init__(incoming, **kwargs)

        self._srng = RandomStreams()

        # Set vars.
        self.nonlinearity = nonlinearity
        self.num_inputs = int(np.prod(self.input_shape[1:]))
        self.num_units = num_units
        self.prior_sd = prior_sd

        prior_rho = self.std_to_log(self.prior_sd)

        self.W = np.random.normal(0., prior_sd,
                                  (self.num_inputs, self.num_units))  # @UndefinedVariable
        self.b = np.zeros(
            (self.num_units,),
            dtype=theano.config.floatX)  # @UndefinedVariable

        # Here we set the priors.
        # -----------------------
        self.mu = self.add_param(
            lasagne.init.Normal(1., 0.),
            (self.num_inputs, self.num_units),
            name='mu'
        )
        self.rho = self.add_param(
            lasagne.init.Constant(prior_rho),
            (self.num_inputs, self.num_units),
            name='rho'
        )
        # Bias priors.
        self.b_mu = self.add_param(
            lasagne.init.Normal(1., 0.),
            (self.num_units,),
            name="b_mu",
            regularizable=False
        )
        self.b_rho = self.add_param(
            lasagne.init.Constant(prior_rho),
            (self.num_units,),
            name="b_rho",
            regularizable=False
        )

    def log_to_std(self, rho):
        """Transformation for allowing rho in \mathbb{R}, rather than \mathbb{R}_+

        This makes sure that we don't get negative stds. However, a downside might be
        that we have little gradient on close to 0 std (= -inf using this transformation).
        """
        return T.log(1 + T.exp(rho))

    def std_to_log(self, sigma):
        """Reverse log_to_std transformation."""
        return np.log(np.exp(sigma) - 1)

    def get_W(self):
        # Here we generate random epsilon values from a normal distribution
        epsilon = self._srng.normal(size=(self.num_inputs, self.num_units), avg=0., std=1.,
                                    dtype=theano.config.floatX)  # @UndefinedVariable
        # Here we calculate weights based on shifting and rescaling according
        # to mean and variance (paper step 2)
        W = self.mu + self.log_to_std(self.rho) * epsilon
        self.W = W
        return W

    def get_b(self):
        # Here we generate random epsilon values from a normal distribution
        epsilon = self._srng.normal(size=(self.num_units,), avg=0., std=1.,
                                    dtype=theano.config.floatX)  # @UndefinedVariable
        b = self.b_mu + self.log_to_std(self.b_rho) * epsilon
        self.b = b
        return b

    def get_output_for_reparametrization(self, input, **kwargs):
        """Implementation of the local reparametrization trick.

        This essentially leads to a speedup compared to the naive implementation case.
        Furthermore, it leads to gradients with less variance.

        References
        ----------
        Kingma et al., "Variational Dropout and the Local Reparametrization Trick", 2015
        """
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)

        gamma = T.dot(input, self.mu) + self.b_mu.dimshuffle('x', 0)
        delta = T.dot(T.square(input), T.square(self.log_to_std(
            self.rho))) + T.square(self.log_to_std(self.b_rho)).dimshuffle('x', 0)
        epsilon = self._srng.normal(size=(self.num_units,), avg=0., std=1.,
                                    dtype=theano.config.floatX)  # @UndefinedVariable

        activation = gamma + T.sqrt(delta) * epsilon

        return self.nonlinearity(activation)

    def kl_div_p_q(self, p_mean, p_std, q_mean, q_std):
        """KL divergence D_{KL}[p(x)||q(x)] for a fully factorized Gaussian"""
        numerator = T.square(p_mean - q_mean) + \
            T.square(p_std) - T.square(q_std)
        denominator = 2 * T.square(q_std) + 1e-8
        return T.sum(
            numerator / denominator + T.log(q_std) - T.log(p_std))

    def kl_div_new_prior(self):
        kl_div = self.kl_div_p_q(
            self.mu, self.log_to_std(self.rho), 0., self.prior_sd)
        kl_div += self.kl_div_p_q(self.b_mu,
                                  self.log_to_std(self.b_rho), 0., self.prior_sd)
        return kl_div

    def get_output_for(self, input, **kwargs):
        if USE_REPARAMETRIZATION_TRICK:
            return self.get_output_for_reparametrization(input, **kwargs)
        else:
            return self.get_output_for_default(input, **kwargs)

    def get_output_for_default(self, input, **kwargs):
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)

        activation = T.dot(input, self.get_W()) + \
            self.get_b().dimshuffle('x', 0)

        return self.nonlinearity(activation)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)


class BNN(LasagnePowered, Serializable):
    """Bayesian neural network (BNN) based on Blundell2016."""

    def __init__(self, n_in,
                 n_hidden,
                 n_out,
                 layers_type,
                 n_batches,
                 trans_func=lasagne.nonlinearities.rectify,
                 out_func=lasagne.nonlinearities.linear,
                 batch_size=100,
                 n_samples=10,
                 prior_sd=0.5,
                 use_reverse_kl_reg=False,
                 reverse_kl_reg_factor=0.1,
                 learning_rate=0.0001,
                 ):

        Serializable.quick_init(self, locals())
        assert len(layers_type) == len(n_hidden) + 1

        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.batch_size = batch_size
        self.transf = trans_func
        self.outf = out_func
        self.n_samples = n_samples
        self.prior_sd = prior_sd
        self.layers_type = layers_type
        self.n_batches = n_batches
        self.use_reverse_kl_reg = use_reverse_kl_reg
        self.reverse_kl_reg_factor = reverse_kl_reg_factor
        self.learning_rate = learning_rate

        # Build network architecture.
        self.build_network()

        # Build model might depend on this.
        LasagnePowered.__init__(self, [self.network])

        # Compile theano functions.
        self.build_model()

    def log_p_w_q_w_kl(self):
        """KL divergence KL[q_\phi(w)||p(w)]"""
        layers = filter(lambda l: l.name == BNN_LAYER_TAG,
                        lasagne.layers.get_all_layers(self.network)[1:])
        return sum(l.kl_div_new_prior() for l in layers)

    # P(s'|s,a,delta)
    def _log_prob_normal(self, input, mu=0.):
        log_normal = - np.pi*T.sum(T.square(input - mu), axis=1)
        return log_normal

    def pred_sym(self, input):
        return lasagne.layers.get_output(self.network, input)

    def get_log_p_D_given_w(self, input, target):
        # MC samples.
        _log_p_D_given_w = []
        for _ in range(self.n_samples):
            # Make prediction.
            prediction = self.pred_sym(input)
            # Calculate model likelihood log(P(D|w)).
            _log_p_D_given_w.append(self._log_prob_normal(
                target, prediction))
        log_p_D_given_w = sum(_log_p_D_given_w)
        return log_p_D_given_w

    def loss(self, log_p_D_given_w):

        # MC samples.
        # Calculate variational posterior log(q(w)) and prior log(p(w)).
        kl = self.log_p_w_q_w_kl()
        if self.use_reverse_kl_reg:
            kl += self.reverse_kl_reg_factor * \
                self.reverse_log_p_w_q_w_kl()

        # Calculate loss function.
        return kl / self.n_batches - T.sum(log_p_D_given_w) / self.n_samples

    def surprise(self, log_p_D_given_w):
        return - log_p_D_given_w / self.n_samples

    def build_network(self):

        # Input layer
        network = lasagne.layers.InputLayer(shape=(1, self.n_in))

        # Hidden layers
        for i in range(len(self.n_hidden)):
            # Probabilistic layer (1) or deterministic layer (0).
            if self.layers_type[i] == 1:
                network = BNNLayer(
                    network, self.n_hidden[i], nonlinearity=self.transf, prior_sd=self.prior_sd, name=BNN_LAYER_TAG)
            else:
                network = lasagne.layers.DenseLayer(
                    network, self.n_hidden[i], nonlinearity=self.transf)

        # Output layer
        if self.layers_type[len(self.n_hidden)] == 1:
            # Probabilistic layer (1) or deterministic layer (0).
            network = BNNLayer(
                network, self.n_out, nonlinearity=self.outf, prior_sd=self.prior_sd, name=BNN_LAYER_TAG)
        else:
            network = lasagne.layers.DenseLayer(
                network, self.n_out, nonlinearity=self.outf)

        self.network = network

    def build_model(self):

        # Prepare Theano variables for inputs and targets
        # Same input for classification as regression.
        input_var = T.matrix('inputs',
                             dtype=theano.config.floatX)  # @UndefinedVariable
        target_var = T.matrix('targets',
                              dtype=theano.config.floatX)  # @UndefinedVariable

        log_p_D_given_w = self.get_log_p_D_given_w(input_var, target_var)

        # Loss function.
        loss = self.loss(log_p_D_given_w)

        surprise = self.surprise(log_p_D_given_w)

        # Create update methods.
        params = lasagne.layers.get_all_params(self.network, trainable=True)
        updates = lasagne.updates.adam(
            loss, params, learning_rate=self.learning_rate)

        # Train/val fn.
        self.pred_fn = ext.compile_function(
            [input_var], self.pred_sym(input_var), log_name='pred_fn')

        self.train_fn = ext.compile_function(
            [input_var, target_var], loss, updates=updates, log_name='train_fn')

        # Surprise fn.
        self.surprise_fn = ext.compile_function(
            [input_var, target_var], surprise, log_name='surprise_fn')


if __name__ == '__main__':
    pass
