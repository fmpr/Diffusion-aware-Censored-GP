from jax import vmap
from bayesnewton.likelihoods import MultiLatentLikelihood, GaussNewtonMixin, Likelihood, GeneralisedGaussNewtonMixin
from bayesnewton.cubature import variational_expectation_cubature
from jax.scipy.stats import norm
from jax.scipy.special import gammaln
import jax.numpy as jnp
import numpy as np
import objax
from bayesnewton.utils import softplus, sigmoid, sigmoid_diff, softplus_inv
from jax.scipy.stats.poisson import cdf as poisson_cdf


# ------------------------------------------------ Diffused Censored GP - spatio-temporal version (univariate)

class MyGaussian(Likelihood, GaussNewtonMixin):
    """
    The Gaussian likelihood:
        p(yₙ|fₙ) = 𝓝(yₙ|fₙ,σ²)
    TODO: implement multivariate version
    """
    def __init__(self,
                 variance=0.1,
                 fix_variance=False):
        """
        :param variance: The observation noise variance, σ²
        """
        if fix_variance:
            self.transformed_variance = objax.StateVar(jnp.array(softplus_inv(variance)))
        else:
            self.transformed_variance = objax.TrainVar(jnp.array(softplus_inv(variance)))
        super().__init__()
        self.name = 'Gaussian'
        self.link_fn = lambda f: f

    @property
    def variance(self):
        return softplus(self.transformed_variance.value)

    def evaluate_log_likelihood(self, y, f, extra):
        """
        Evaluate the log-Gaussian function log𝓝(yₙ|fₙ,σ²).
        Can be used to evaluate Q cubature points.
        :param y: observed data yₙ [scalar]
        :param f: mean, i.e. the latent function value fₙ [Q, 1]
        :return:
            log𝓝(yₙ|fₙ,σ²), where σ² is the observation noise variance [Q, 1]
        """
        
        log_likelihood = norm.logpdf(y, loc=f, scale=jnp.sqrt(self.variance))
        return jnp.squeeze(log_likelihood)

    def predict(self, mean_f, var_f, cubature=None):
        return mean_f, var_f + self.variance
    
    
class MyCensoredGaussian(Likelihood, GaussNewtonMixin):
    """
    The Gaussian likelihood:
        p(yₙ|fₙ) = 𝓝(yₙ|fₙ,σ²)
    TODO: implement multivariate version
    """
    def __init__(self,
                 variance=0.1,
                 fix_variance=False):
        """
        :param variance: The observation noise variance, σ²
        """
        if fix_variance:
            self.transformed_variance = objax.StateVar(jnp.array(softplus_inv(variance)))
        else:
            self.transformed_variance = objax.TrainVar(jnp.array(softplus_inv(variance)))
        super().__init__()
        self.name = 'Censored Gaussian'
        self.link_fn = lambda f: f

    @property
    def variance(self):
        return softplus(self.transformed_variance.value)

    def evaluate_log_likelihood(self, y_cens, f, extra):
        """
        Evaluate the log-Gaussian function log𝓝(yₙ|fₙ,σ²).
        Can be used to evaluate Q cubature points.
        :param y: observed data yₙ [scalar]
        :param f: mean, i.e. the latent function value fₙ [Q, 1]
        :return:
            log𝓝(yₙ|fₙ,σ²), where σ² is the observation noise variance [Q, 1]
        """
        
        threshold = extra    
        log_likelihood_observed = norm.logpdf(y_cens, loc=f, scale=jnp.sqrt(self.variance))
    
        # Log likelihood of censored values
        log_likelihood_censored = jnp.log(1-norm.cdf(y_cens, loc=f, scale=jnp.sqrt(self.variance)) + 0.0001)
        
        # Combine log likelihoods based on the censored indicators
        censoring_indicator = y_cens == threshold
        log_likelihood_combined = censoring_indicator * log_likelihood_censored + (1 - censoring_indicator) * log_likelihood_observed
        
        log_likelihood = jnp.squeeze(log_likelihood_combined)
        
        return log_likelihood

    def predict(self, mean_f, var_f, cubature=None):
        return mean_f, var_f + self.variance
    

class MyDiffusedCensoredGaussian(Likelihood, GaussNewtonMixin):
    """
    The Gaussian likelihood:
        p(yₙ|fₙ) = 𝓝(yₙ|fₙ,σ²)
    TODO: implement multivariate version
    """
    def __init__(self,
                 X_diffusion,
                 variance=0.1,
                 diffusion_steps=1,
                 diffusion_lengthscale=1.,
                 diffusion_variance=1.,
                 sink_prob=None,
                 fix_diffusion=False,
                 fix_variance=False):
        """
        :param variance: The observation noise variance, σ²
        """
        self.X_diffusion = X_diffusion
        self.diffusion_steps = diffusion_steps
        if sink_prob is not None:
            assert sink_prob >= 0 and sink_prob <= 1
        self.sink_prob = sink_prob
        if fix_diffusion:
            self.diffusion_lengthscale = objax.StateVar(jnp.array(diffusion_lengthscale))
            self.diffusion_variance = objax.StateVar(jnp.array(diffusion_variance))
        else:
            self.diffusion_lengthscale = objax.TrainVar(jnp.array(diffusion_lengthscale))
            self.diffusion_variance = objax.TrainVar(jnp.array(diffusion_variance))
        if fix_variance:
            self.transformed_variance = objax.StateVar(jnp.array(softplus_inv(variance)))
        else:
            self.transformed_variance = objax.TrainVar(jnp.array(softplus_inv(variance)))
        super().__init__()
        self.name = 'Diffused Censored Gaussian'
        self.link_fn = lambda f: f

    @property
    def variance(self):
        return softplus(self.transformed_variance.value)

    def evaluate_log_likelihood(self, y_cens, f, extra):
        """
        Evaluate the log-Gaussian function log𝓝(yₙ|fₙ,σ²).
        Can be used to evaluate Q cubature points.
        :param y: observed data yₙ [scalar]
        :param f: mean, i.e. the latent function value fₙ [Q, 1]
        :return:
            log𝓝(yₙ|fₙ,σ²), where σ² is the observation noise variance [Q, 1]
        """
        
        threshold = extra
        log_likelihood_observed = norm.logpdf(y_cens, loc=f, scale=jnp.sqrt(self.variance))
    
        # Log likelihood of censored values
        log_likelihood_censored = jnp.log(1-norm.cdf(y_cens, loc=f, scale=jnp.sqrt(self.variance)) + 0.0001)
        
        # Combine log likelihoods based on the censored indicators
        censoring_indicator = y_cens == threshold
        log_likelihood_combined = censoring_indicator * log_likelihood_censored + (1 - censoring_indicator) * log_likelihood_observed
        
        log_likelihood = jnp.squeeze(log_likelihood_combined)
        
        return log_likelihood

    def variational_expectation_(self, y, m, v, extra, cubature=None):
        """
        If no custom variational expectation method is provided, we use cubature.
        """
        return variational_expectation_cubature(self, y, m, v, extra, cubature)

    def variational_expectation(self, y, m, v, extra, cubature=None):
        """
        Most likelihoods factorise across data points. For multi-latent models, a custom method must be implemented.
        """

        # align shapes and compute mask
        y = y.reshape(-1, 1, 1)
        thre = extra.reshape(-1, 1, 1)
        m = m.reshape(-1, 1, 1)
        v = jnp.diag(v).reshape(-1, 1, 1)
        mask = jnp.isnan(y)
        y = jnp.where(mask, m, y)
        
        ### Diffuse excess demand based on transition matrix for k diffusion steps
        transition_matrix = compute_transition_matrix(self.X_diffusion, 
                                                      diffusion_lengthscale=self.diffusion_lengthscale, 
                                                      diffusion_variance=self.diffusion_variance,
                                                      sink_prob=self.sink_prob)
        for step in range(self.diffusion_steps):
            excess_demand = jnp.maximum(m - thre, jnp.zeros(m.shape))
            diffused_excess = excess_demand[:,0,0] @ transition_matrix[:len(m),:]
            m = m[:,0,0] + diffused_excess[:len(m)]
            m = m.reshape(-1, 1, 1)
        
        # compute variational expectations and their derivatives
        var_exp, dE_dm, d2E_dm2 = vmap(self.variational_expectation_, (0, 0, 0, 0, None))(y, m, v, extra, cubature)

        # apply mask
        var_exp = jnp.where(jnp.squeeze(mask), 0., jnp.squeeze(var_exp))
        dE_dm = jnp.where(mask, jnp.nan, dE_dm)
        d2E_dm2 = jnp.where(mask, jnp.nan, d2E_dm2)

        return var_exp, jnp.squeeze(dE_dm, axis=2), jnp.diag(jnp.squeeze(d2E_dm2, axis=(1, 2)))

    def predict(self, mean_f, var_f, cubature=None):
        return mean_f, var_f + self.variance
    

class MyDiffusedCensoredGaussianB(Likelihood, GaussNewtonMixin):
    """
    The Gaussian likelihood:
        p(yₙ|fₙ) = 𝓝(yₙ|fₙ,σ²)
    TODO: implement multivariate version
    """
    def __init__(self,
                 X_diffusion,
                 variance=0.1,
                 diffusion_steps=1,
                 diffusion_lengthscale=1.,
                 diffusion_variance=1.,
                 sink_prob=None,
                 fix_diffusion=False,
                 fix_variance=False):
        """
        :param variance: The observation noise variance, σ²
        """
        self.X_diffusion = X_diffusion
        self.diffusion_steps = diffusion_steps
        if sink_prob is not None:
            assert sink_prob >= 0 and sink_prob <= 1
        self.sink_prob = sink_prob
        if fix_diffusion:
            self.diffusion_lengthscale = objax.StateVar(jnp.array(diffusion_lengthscale))
            self.diffusion_variance = objax.StateVar(jnp.array(diffusion_variance))
        else:
            self.diffusion_lengthscale = objax.TrainVar(jnp.array(diffusion_lengthscale))
            self.diffusion_variance = objax.TrainVar(jnp.array(diffusion_variance))
        if fix_variance:
            self.transformed_variance = objax.StateVar(jnp.array(softplus_inv(variance)))
        else:
            self.transformed_variance = objax.TrainVar(jnp.array(softplus_inv(variance)))
        super().__init__()
        self.name = 'Diffused Censored Gaussian'
        self.link_fn = lambda f: f

    @property
    def variance(self):
        return softplus(self.transformed_variance.value)

    def evaluate_log_likelihood(self, y_cens, f, extra):
        """
        Evaluate the log-Gaussian function log𝓝(yₙ|fₙ,σ²).
        Can be used to evaluate Q cubature points.
        :param y: observed data yₙ [scalar]
        :param f: mean, i.e. the latent function value fₙ [Q, 1]
        :return:
            log𝓝(yₙ|fₙ,σ²), where σ² is the observation noise variance [Q, 1]
        """
        
        threshold = extra
        log_likelihood_observed = norm.logpdf(y_cens, loc=f, scale=jnp.sqrt(self.variance))
    
        # Log likelihood of censored values
        log_likelihood_censored = jnp.log(1-norm.cdf(y_cens, loc=f, scale=jnp.sqrt(self.variance)) + 0.0001)
        
        # Combine log likelihoods based on the censored indicators
        censoring_indicator = y_cens >= threshold
        log_likelihood_combined = censoring_indicator * log_likelihood_censored + (1 - censoring_indicator) * log_likelihood_observed
        
        log_likelihood = jnp.squeeze(log_likelihood_combined)
        
        return log_likelihood

    def variational_expectation_(self, y, m, v, extra, cubature=None):
        """
        If no custom variational expectation method is provided, we use cubature.
        """
        return variational_expectation_cubature(self, y, m, v, extra, cubature)

    def variational_expectation(self, y, m, v, extra, cubature=None):
        """
        Most likelihoods factorise across data points. For multi-latent models, a custom method must be implemented.
        """

        # align shapes and compute mask
        y = y.reshape(-1, 1, 1)
        thre = extra.reshape(-1, 1, 1)
        m = m.reshape(-1, 1, 1)
        v = jnp.diag(v).reshape(-1, 1, 1)
        mask = jnp.isnan(y)
        y = jnp.where(mask, m, y)
        
        ### Diffuse excess demand based on transition matrix for k diffusion steps
        transition_matrix = compute_transition_matrix(self.X_diffusion, 
                                                      diffusion_lengthscale=self.diffusion_lengthscale, 
                                                      diffusion_variance=self.diffusion_variance,
                                                      sink_prob=self.sink_prob)
        for step in range(self.diffusion_steps):
            excess_demand = jnp.maximum(m - thre, jnp.zeros(m.shape))
            diffused_excess = excess_demand[:,0,0] @ transition_matrix[:len(m),:]
            m = m[:,0,0] + diffused_excess[:len(m)]
            m = m.reshape(-1, 1, 1)
        
        # compute variational expectations and their derivatives
        var_exp, dE_dm, d2E_dm2 = vmap(self.variational_expectation_, (0, 0, 0, 0, None))(y, m, v, extra, cubature)

        # apply mask
        var_exp = jnp.where(jnp.squeeze(mask), 0., jnp.squeeze(var_exp))
        dE_dm = jnp.where(mask, jnp.nan, dE_dm)
        d2E_dm2 = jnp.where(mask, jnp.nan, d2E_dm2)

        return var_exp, jnp.squeeze(dE_dm, axis=2), jnp.diag(jnp.squeeze(d2E_dm2, axis=(1, 2)))

    def predict(self, mean_f, var_f, cubature=None):
        return mean_f, var_f + self.variance
    

class MyDiffusedCensoredGaussianC(Likelihood, GaussNewtonMixin):
    """
    The Gaussian likelihood:
        p(yₙ|fₙ) = 𝓝(yₙ|fₙ,σ²)
    TODO: implement multivariate version
    """
    def __init__(self,
                 X_diffusion,
                 variance=0.1,
                 diffusion_steps=1,
                 diffusion_lengthscale=1.,
                 diffusion_variance=1.,
                 sink_prob=None,
                 fix_diffusion=False,
                 fix_variance=False):
        """
        :param variance: The observation noise variance, σ²
        """
        self.X_diffusion = X_diffusion
        self.diffusion_steps = diffusion_steps
        if sink_prob is not None:
            assert sink_prob >= 0 and sink_prob <= 1
        self.sink_prob = sink_prob
        if fix_diffusion:
            self.diffusion_lengthscale = objax.StateVar(jnp.array(diffusion_lengthscale))
            self.diffusion_variance = objax.StateVar(jnp.array(diffusion_variance))
        else:
            self.diffusion_lengthscale = objax.TrainVar(jnp.array(diffusion_lengthscale))
            self.diffusion_variance = objax.TrainVar(jnp.array(diffusion_variance))
        if fix_variance:
            self.transformed_variance = objax.StateVar(jnp.array(softplus_inv(variance)))
        else:
            self.transformed_variance = objax.TrainVar(jnp.array(softplus_inv(variance)))
        super().__init__()
        self.name = 'Diffused Censored Gaussian'
        self.link_fn = lambda f: f

    @property
    def variance(self):
        return softplus(self.transformed_variance.value)

    def evaluate_log_likelihood(self, y_cens, f, extra):
        """
        Evaluate the log-Gaussian function log𝓝(yₙ|fₙ,σ²).
        Can be used to evaluate Q cubature points.
        :param y: observed data yₙ [scalar]
        :param f: mean, i.e. the latent function value fₙ [Q, 1]
        :return:
            log𝓝(yₙ|fₙ,σ²), where σ² is the observation noise variance [Q, 1]
        """
        
        threshold = extra
        log_likelihood_observed = norm.logpdf(y_cens, loc=f, scale=jnp.sqrt(self.variance))
    
        # Log likelihood of censored values
        log_likelihood_censored = jnp.log(1-norm.cdf(y_cens, loc=f, scale=jnp.sqrt(self.variance)) + 0.0001)
        
        # Combine log likelihoods based on the censored indicators
        censoring_indicator = y_cens >= threshold
        log_likelihood_combined = censoring_indicator * log_likelihood_censored + (1 - censoring_indicator) * log_likelihood_observed
        
        log_likelihood = jnp.squeeze(log_likelihood_combined)
        
        return log_likelihood

    def variational_expectation_(self, y, m, v, extra, cubature=None):
        """
        If no custom variational expectation method is provided, we use cubature.
        """
        return variational_expectation_cubature(self, y, m, v, extra, cubature)

    def variational_expectation(self, y, m, v, extra, cubature=None):
        """
        Most likelihoods factorise across data points. For multi-latent models, a custom method must be implemented.
        """

        # align shapes and compute mask
        y = y.reshape(-1, 1, 1)
        thre = extra.reshape(-1, 1, 1)
        m = m.reshape(-1, 1, 1)
        v = jnp.diag(v).reshape(-1, 1, 1)
        mask = jnp.isnan(y)
        y = jnp.where(mask, m, y)
        
        ### Diffuse excess demand based on transition matrix for k diffusion steps
        transition_matrix = compute_transition_matrix(self.X_diffusion, 
                                                      diffusion_lengthscale=self.diffusion_lengthscale, 
                                                      diffusion_variance=self.diffusion_variance,
                                                      sink_prob=self.sink_prob)
        m0 = m[:]
        for step in range(self.diffusion_steps):
            excess_demand = jnp.maximum(m - thre, jnp.zeros(m.shape))
            diffused_excess = excess_demand[:,0,0] @ (transition_matrix-jnp.eye(len(transition_matrix)))[:len(m),:]
            m = m[:,0,0] + diffused_excess[:len(m)]
            m = m.reshape(-1, 1, 1)
        
        m = m0 + jnp.maximum(m - m0, jnp.zeros(m.shape))
        
        # compute variational expectations and their derivatives
        var_exp, dE_dm, d2E_dm2 = vmap(self.variational_expectation_, (0, 0, 0, 0, None))(y, m, v, extra, cubature)

        # apply mask
        var_exp = jnp.where(jnp.squeeze(mask), 0., jnp.squeeze(var_exp))
        dE_dm = jnp.where(mask, jnp.nan, dE_dm)
        d2E_dm2 = jnp.where(mask, jnp.nan, d2E_dm2)

        return var_exp, jnp.squeeze(dE_dm, axis=2), jnp.diag(jnp.squeeze(d2E_dm2, axis=(1, 2)))

    def predict(self, mean_f, var_f, cubature=None):
        return mean_f, var_f + self.variance
    

class MyDiffusedCensoredGaussianD(Likelihood, GaussNewtonMixin):
    """
    The Gaussian likelihood:
        p(yₙ|fₙ) = 𝓝(yₙ|fₙ,σ²)
    TODO: implement multivariate version
    """
    def __init__(self,
                 X_diffusion,
                 variance=0.1,
                 diffusion_steps=1,
                 diffusion_lengthscale=1.,
                 diffusion_variance=1.,
                 sink_prob=None,
                 fix_diffusion=False,
                 fix_variance=False):
        """
        :param variance: The observation noise variance, σ²
        """
        self.X_diffusion = X_diffusion
        self.diffusion_steps = diffusion_steps
        if sink_prob is not None:
            assert sink_prob >= 0 and sink_prob <= 1
        self.sink_prob = sink_prob
        if fix_diffusion:
            self.diffusion_lengthscale = objax.StateVar(jnp.array(diffusion_lengthscale))
            self.diffusion_variance = objax.StateVar(jnp.array(diffusion_variance))
        else:
            self.diffusion_lengthscale = objax.TrainVar(jnp.array(diffusion_lengthscale))
            self.diffusion_variance = objax.TrainVar(jnp.array(diffusion_variance))
        if fix_variance:
            self.transformed_variance = objax.StateVar(jnp.array(softplus_inv(variance)))
        else:
            self.transformed_variance = objax.TrainVar(jnp.array(softplus_inv(variance)))
        super().__init__()
        self.name = 'Diffused Censored Gaussian'
        self.link_fn = lambda f: f

    @property
    def variance(self):
        return softplus(self.transformed_variance.value)

    def evaluate_log_likelihood(self, y_cens, f, extra):
        """
        Evaluate the log-Gaussian function log𝓝(yₙ|fₙ,σ²).
        Can be used to evaluate Q cubature points.
        :param y: observed data yₙ [scalar]
        :param f: mean, i.e. the latent function value fₙ [Q, 1]
        :return:
            log𝓝(yₙ|fₙ,σ²), where σ² is the observation noise variance [Q, 1]
        """
        
        threshold = extra
        log_likelihood_observed = norm.logpdf(y_cens, loc=f, scale=jnp.sqrt(self.variance))
    
        # Log likelihood of censored values
        log_likelihood_censored = jnp.log(1-norm.cdf(y_cens, loc=f, scale=jnp.sqrt(self.variance)) + 0.0001)
        
        # Combine log likelihoods based on the censored indicators
        censoring_indicator = y_cens == threshold
        log_likelihood_combined = censoring_indicator * log_likelihood_censored + (1 - censoring_indicator) * log_likelihood_observed
        
        log_likelihood = jnp.squeeze(log_likelihood_combined)
        
        return log_likelihood

    def variational_expectation_(self, y, m, v, extra, cubature=None):
        """
        If no custom variational expectation method is provided, we use cubature.
        """
        return variational_expectation_cubature(self, y, m, v, extra, cubature)

    def variational_expectation(self, y, m, v, extra, cubature=None):
        """
        Most likelihoods factorise across data points. For multi-latent models, a custom method must be implemented.
        """

        # align shapes and compute mask
        y = y.reshape(-1, 1, 1)
        thre = extra.reshape(-1, 1, 1)
        m = m.reshape(-1, 1, 1)
        v = jnp.diag(v).reshape(-1, 1, 1)
        mask = jnp.isnan(y)
        y = jnp.where(mask, m, y)
        
        ### Diffuse excess demand based on transition matrix for k diffusion steps
        transition_matrix = compute_transition_matrix(self.X_diffusion, 
                                                      diffusion_lengthscale=self.diffusion_lengthscale, 
                                                      diffusion_variance=self.diffusion_variance,
                                                      sink_prob=self.sink_prob)
        m0 = m[:]
        for step in range(self.diffusion_steps):
            excess_demand = jnp.maximum(m - thre, jnp.zeros(m.shape))
            diffused_excess = excess_demand[:,0,0] @ (transition_matrix-jnp.eye(len(transition_matrix)))[:len(m),:]
            m = m[:,0,0] + diffused_excess[:len(m)]
            m = m.reshape(-1, 1, 1)
        
        m = m0 + jnp.maximum(m - m0, jnp.zeros(m.shape))
        
        # compute variational expectations and their derivatives
        var_exp, dE_dm, d2E_dm2 = vmap(self.variational_expectation_, (0, 0, 0, 0, None))(y, m, v, extra, cubature)

        # apply mask
        var_exp = jnp.where(jnp.squeeze(mask), 0., jnp.squeeze(var_exp))
        dE_dm = jnp.where(mask, jnp.nan, dE_dm)
        d2E_dm2 = jnp.where(mask, jnp.nan, d2E_dm2)

        return var_exp, jnp.squeeze(dE_dm, axis=2), jnp.diag(jnp.squeeze(d2E_dm2, axis=(1, 2)))

    def predict(self, mean_f, var_f, cubature=None):
        return mean_f, var_f + self.variance


class MyPoisson(Likelihood, GeneralisedGaussNewtonMixin):
    """
    TODO: tidy docstring
    The Poisson likelihood:
        p(yₙ|fₙ) = Poisson(fₙ) = μʸ exp(-μ) / yₙ!
    where μ = g(fₙ) = mean = variance is the Poisson intensity.
    yₙ is non-negative integer count data.
    No closed form moment matching is available, se we default to using cubature.

    Letting Zy = gamma(yₙ+1) = yₙ!, we get log p(yₙ|fₙ) = log(g(fₙ))yₙ - g(fₙ) - log(Zy)
    The larger the intensity μ, the stronger the likelihood resembles a Gaussian
    since skewness = 1/sqrt(μ) and kurtosis = 1/μ.
    Two possible link functions:
    'exp':      link(fₙ) = exp(fₙ),         we have p(yₙ|fₙ) = exp(fₙyₙ-exp(fₙ))           / Zy.
    'logistic': link(fₙ) = log(1+exp(fₙ))), we have p(yₙ|fₙ) = logʸ(1+exp(fₙ)))(1+exp(fₙ)) / Zy.
    """
    def __init__(self,
                 binsize=1,
                 link='exp'):
        """
        :param link: link function, either 'exp' or 'logistic'
        """
        super().__init__()
        if link == 'exp':
            self.link_fn = jnp.exp
            self.dlink_fn = jnp.exp
            self.d2link_fn = jnp.exp
        elif link == 'logistic':
            self.link_fn = softplus
            self.dlink_fn = sigmoid
            self.d2link_fn = sigmoid_diff
        else:
            raise NotImplementedError('link function not implemented')
        self.binsize = jnp.array(binsize)
        self.name = 'Poisson'

    def evaluate_log_likelihood(self, y, f, extra):
        """
        Evaluate the Poisson log-likelihood:
            log p(yₙ|fₙ) = log Poisson(fₙ) = log(μʸ exp(-μ) / yₙ!)
        for μ = g(fₙ), where g() is the link function (exponential or logistic).
        We use the gamma function to evaluate yₙ! = gamma(yₙ + 1).
        Can be used to evaluate Q cubature points when performing moment matching.
        :param y: observed data (yₙ) [scalar]
        :param f: latent function value (fₙ) [Q, 1]
        :return:
            log Poisson(fₙ) = log(μʸ exp(-μ) / yₙ!) [Q, 1]
        """
        mu = self.link_fn(f) * self.binsize
        return jnp.squeeze(y * jnp.log(mu) - mu - gammaln(y + 1))

    def conditional_moments(self, f):
        """
        The first two conditional moments of a Poisson distribution are equal to the intensity:
            E[yₙ|fₙ] = link(fₙ)
            Var[yₙ|fₙ] = link(fₙ)
        """
        # TODO: multi-dim case
        return self.link_fn(f) * self.binsize, self.link_fn(f) * self.binsize
        # return self.link_fn(f) * self.binsize, vmap(jnp.diag, 1, 2)(self.link_fn(f) * self.binsize)


class MyCensoredPoisson(Likelihood, GeneralisedGaussNewtonMixin):
    """
    TODO: tidy docstring
    The Poisson likelihood:
        p(yₙ|fₙ) = Poisson(fₙ) = μʸ exp(-μ) / yₙ!
    where μ = g(fₙ) = mean = variance is the Poisson intensity.
    yₙ is non-negative integer count data.
    No closed form moment matching is available, se we default to using cubature.

    Letting Zy = gamma(yₙ+1) = yₙ!, we get log p(yₙ|fₙ) = log(g(fₙ))yₙ - g(fₙ) - log(Zy)
    The larger the intensity μ, the stronger the likelihood resembles a Gaussian
    since skewness = 1/sqrt(μ) and kurtosis = 1/μ.
    Two possible link functions:
    'exp':      link(fₙ) = exp(fₙ),         we have p(yₙ|fₙ) = exp(fₙyₙ-exp(fₙ))           / Zy.
    'logistic': link(fₙ) = log(1+exp(fₙ))), we have p(yₙ|fₙ) = logʸ(1+exp(fₙ)))(1+exp(fₙ)) / Zy.
    """
    def __init__(self,
                 binsize=1,
                 link='exp'):
        """
        :param link: link function, either 'exp' or 'logistic'
        """
        super().__init__()
        if link == 'exp':
            self.link_fn = jnp.exp
            self.dlink_fn = jnp.exp
            self.d2link_fn = jnp.exp
        elif link == 'logistic':
            self.link_fn = softplus
            self.dlink_fn = sigmoid
            self.d2link_fn = sigmoid_diff
        else:
            raise NotImplementedError('link function not implemented')
        self.binsize = jnp.array(binsize)
        self.name = 'Poisson'
    
    def evaluate_log_likelihood(self, y_cens, f, extra):
        """
        Evaluate the Poisson log-likelihood:
            log p(yₙ|fₙ) = log Poisson(fₙ) = log(μʸ exp(-μ) / yₙ!)
        for μ = g(fₙ), where g() is the link function (exponential or logistic).
        We use the gamma function to evaluate yₙ! = gamma(yₙ + 1).
        Can be used to evaluate Q cubature points when performing moment matching.
        :param y: observed data (yₙ) [scalar]
        :param f: latent function value (fₙ) [Q, 1]
        :return:
            log Poisson(fₙ) = log(μʸ exp(-μ) / yₙ!) [Q, 1]
        """
        #mu = self.link_fn(f) * self.binsize
        #return jnp.squeeze(y * jnp.log(mu) - mu - gammaln(y + 1))
        threshold = extra
        mu = self.link_fn(f) * self.binsize
        log_likelihood_observed = y_cens * jnp.log(mu) - mu - gammaln(y_cens + 1)
        #print("log_likelihood_observed:", log_likelihood_observed.shape)
        
        # Log likelihood of censored values
        #cdf_value = jnp.sum(jnp.exp(jnp.arange(0, y_cens + 1) * jnp.log(mu) - mu - gammaln(jnp.arange(1, y_cens + 2))))
        cdf_value = poisson_cdf(y_cens, mu)
        #print("cdf_value:", cdf_value.shape)
        #log_likelihood_censored = jnp.log(1-cdf_value + 0.01)
        log_likelihood_censored = jnp.log(1-cdf_value + 0.0001)
        
        # Combine log likelihoods based on the censored indicators
        censoring_indicator = y_cens == threshold
        log_likelihood_combined = censoring_indicator * log_likelihood_censored + (1 - censoring_indicator) * log_likelihood_observed
        
        #log_likelihood = jnp.squeeze(log_likelihood_observed)
        log_likelihood = jnp.squeeze(log_likelihood_combined)
        
        #print(log_likelihood_observed.shape, log_likelihood_censored.shape, censoring_indicator.shape, log_likelihood_combined.shape, log_likelihood.shape)
        return log_likelihood

    def conditional_moments(self, f):
        """
        The first two conditional moments of a Poisson distribution are equal to the intensity:
            E[yₙ|fₙ] = link(fₙ)
            Var[yₙ|fₙ] = link(fₙ)
        """
        # TODO: multi-dim case
        return self.link_fn(f) * self.binsize, self.link_fn(f) * self.binsize
        # return self.link_fn(f) * self.binsize, vmap(jnp.diag, 1, 2)(self.link_fn(f) * self.binsize)


class MyDiffusedCensoredPoisson(Likelihood, GeneralisedGaussNewtonMixin):
    """
    TODO: tidy docstring
    The Poisson likelihood:
        p(yₙ|fₙ) = Poisson(fₙ) = μʸ exp(-μ) / yₙ!
    where μ = g(fₙ) = mean = variance is the Poisson intensity.
    yₙ is non-negative integer count data.
    No closed form moment matching is available, se we default to using cubature.

    Letting Zy = gamma(yₙ+1) = yₙ!, we get log p(yₙ|fₙ) = log(g(fₙ))yₙ - g(fₙ) - log(Zy)
    The larger the intensity μ, the stronger the likelihood resembles a Gaussian
    since skewness = 1/sqrt(μ) and kurtosis = 1/μ.
    Two possible link functions:
    'exp':      link(fₙ) = exp(fₙ),         we have p(yₙ|fₙ) = exp(fₙyₙ-exp(fₙ))           / Zy.
    'logistic': link(fₙ) = log(1+exp(fₙ))), we have p(yₙ|fₙ) = logʸ(1+exp(fₙ)))(1+exp(fₙ)) / Zy.
    """
    def __init__(self,
                 X_diffusion,
                 binsize=1,
                 link='exp',
                 diffusion_steps=1,
                 diffusion_lengthscale=1.,
                 diffusion_variance=1.,
                 sink_prob=None,
                 fix_diffusion=False):
        """
        :param link: link function, either 'exp' or 'logistic'
        """
        super().__init__()
        self.X_diffusion = X_diffusion
        self.diffusion_steps = diffusion_steps
        if sink_prob is not None:
            assert sink_prob >= 0 and sink_prob <= 1
        self.sink_prob = sink_prob
        if fix_diffusion:
            self.diffusion_lengthscale = objax.StateVar(jnp.array(diffusion_lengthscale))
            self.diffusion_variance = objax.StateVar(jnp.array(diffusion_variance))
        else:
            self.diffusion_lengthscale = objax.TrainVar(jnp.array(diffusion_lengthscale))
            self.diffusion_variance = objax.TrainVar(jnp.array(diffusion_variance))
        if link == 'exp':
            self.link_fn = jnp.exp
            self.dlink_fn = jnp.exp
            self.d2link_fn = jnp.exp
        elif link == 'logistic':
            self.link_fn = softplus
            self.dlink_fn = sigmoid
            self.d2link_fn = sigmoid_diff
        else:
            raise NotImplementedError('link function not implemented')
        self.binsize = jnp.array(binsize)
        self.name = 'Poisson'
    
    def evaluate_log_likelihood(self, y_cens, f, extra):
        """
        Evaluate the Poisson log-likelihood:
            log p(yₙ|fₙ) = log Poisson(fₙ) = log(μʸ exp(-μ) / yₙ!)
        for μ = g(fₙ), where g() is the link function (exponential or logistic).
        We use the gamma function to evaluate yₙ! = gamma(yₙ + 1).
        Can be used to evaluate Q cubature points when performing moment matching.
        :param y: observed data (yₙ) [scalar]
        :param f: latent function value (fₙ) [Q, 1]
        :return:
            log Poisson(fₙ) = log(μʸ exp(-μ) / yₙ!) [Q, 1]
        """
        #mu = self.link_fn(f) * self.binsize
        #return jnp.squeeze(y * jnp.log(mu) - mu - gammaln(y + 1))
        threshold = extra
        mu = self.link_fn(f) * self.binsize
        log_likelihood_observed = y_cens * jnp.log(mu) - mu - gammaln(y_cens + 1)
        #print("log_likelihood_observed:", log_likelihood_observed.shape)
        
        # Log likelihood of censored values
        #cdf_value = jnp.sum(jnp.exp(jnp.arange(0, y_cens + 1) * jnp.log(mu) - mu - gammaln(jnp.arange(1, y_cens + 2))))
        cdf_value = poisson_cdf(y_cens, mu)
        #print("cdf_value:", cdf_value.shape)
        log_likelihood_censored = jnp.log(1-cdf_value + 0.0001)
    
        
        # Combine log likelihoods based on the censored indicators
        censoring_indicator = y_cens == threshold
        log_likelihood_combined = censoring_indicator * log_likelihood_censored + (1 - censoring_indicator) * log_likelihood_observed
        
        #log_likelihood = jnp.squeeze(log_likelihood_observed)
        log_likelihood = jnp.squeeze(log_likelihood_combined)
        
        #print(log_likelihood_observed.shape, log_likelihood_censored.shape, censoring_indicator.shape, log_likelihood_combined.shape, log_likelihood.shape)
        return log_likelihood

    def variational_expectation_(self, y, m, v, extra, cubature=None):
        """
        If no custom variational expectation method is provided, we use cubature.
        """
        return variational_expectation_cubature(self, y, m, v, extra, cubature)

    def variational_expectation(self, y, m, v, extra, cubature=None):
        """
        Most likelihoods factorise across data points. For multi-latent models, a custom method must be implemented.
        """

        # align shapes and compute mask
        y = y.reshape(-1, 1, 1)
        thre = extra.reshape(-1, 1, 1)
        m = m.reshape(-1, 1, 1)
        v = jnp.diag(v).reshape(-1, 1, 1)
        mask = jnp.isnan(y)
        y = jnp.where(mask, m, y)
        
        ### Diffuse excess demand based on transition matrix for k diffusion steps
        transition_matrix = compute_transition_matrix(self.X_diffusion, 
                                                      diffusion_lengthscale=self.diffusion_lengthscale, 
                                                      diffusion_variance=self.diffusion_variance,
                                                      sink_prob=self.sink_prob)
        for step in range(self.diffusion_steps):
            excess_demand = jnp.maximum(m - thre, jnp.zeros(m.shape))
            diffused_excess = excess_demand[:,0,0] @ transition_matrix[:len(m),:]
            m = m[:,0,0] + diffused_excess[:len(m)]
            m = m.reshape(-1, 1, 1)
        
        # compute variational expectations and their derivatives
        var_exp, dE_dm, d2E_dm2 = vmap(self.variational_expectation_, (0, 0, 0, 0, None))(y, m, v, extra, cubature)

        # apply mask
        var_exp = jnp.where(jnp.squeeze(mask), 0., jnp.squeeze(var_exp))
        dE_dm = jnp.where(mask, jnp.nan, dE_dm)
        d2E_dm2 = jnp.where(mask, jnp.nan, d2E_dm2)

        return var_exp, jnp.squeeze(dE_dm, axis=2), jnp.diag(jnp.squeeze(d2E_dm2, axis=(1, 2)))

    def conditional_moments(self, f):
        """
        The first two conditional moments of a Poisson distribution are equal to the intensity:
            E[yₙ|fₙ] = link(fₙ)
            Var[yₙ|fₙ] = link(fₙ)
        """
        # TODO: multi-dim case
        return self.link_fn(f) * self.binsize, self.link_fn(f) * self.binsize
        # return self.link_fn(f) * self.binsize, vmap(jnp.diag, 1, 2)(self.link_fn(f) * self.binsize)
        

class MyDiffusedCensoredPoissonB(Likelihood, GeneralisedGaussNewtonMixin):
    """
    TODO: tidy docstring
    The Poisson likelihood:
        p(yₙ|fₙ) = Poisson(fₙ) = μʸ exp(-μ) / yₙ!
    where μ = g(fₙ) = mean = variance is the Poisson intensity.
    yₙ is non-negative integer count data.
    No closed form moment matching is available, se we default to using cubature.

    Letting Zy = gamma(yₙ+1) = yₙ!, we get log p(yₙ|fₙ) = log(g(fₙ))yₙ - g(fₙ) - log(Zy)
    The larger the intensity μ, the stronger the likelihood resembles a Gaussian
    since skewness = 1/sqrt(μ) and kurtosis = 1/μ.
    Two possible link functions:
    'exp':      link(fₙ) = exp(fₙ),         we have p(yₙ|fₙ) = exp(fₙyₙ-exp(fₙ))           / Zy.
    'logistic': link(fₙ) = log(1+exp(fₙ))), we have p(yₙ|fₙ) = logʸ(1+exp(fₙ)))(1+exp(fₙ)) / Zy.
    """
    def __init__(self,
                 X_diffusion,
                 binsize=1,
                 link='exp',
                 diffusion_steps=1,
                 diffusion_lengthscale=1.,
                 diffusion_variance=1.,
                 sink_prob=None,
                 fix_diffusion=False):
        """
        :param link: link function, either 'exp' or 'logistic'
        """
        super().__init__()
        self.X_diffusion = X_diffusion
        self.diffusion_steps = diffusion_steps
        if sink_prob is not None:
            assert sink_prob >= 0 and sink_prob <= 1
        self.sink_prob = sink_prob
        if fix_diffusion:
            self.diffusion_lengthscale = objax.StateVar(jnp.array(diffusion_lengthscale))
            self.diffusion_variance = objax.StateVar(jnp.array(diffusion_variance))
        else:
            self.diffusion_lengthscale = objax.TrainVar(jnp.array(diffusion_lengthscale))
            self.diffusion_variance = objax.TrainVar(jnp.array(diffusion_variance))
        if link == 'exp':
            self.link_fn = jnp.exp
            self.dlink_fn = jnp.exp
            self.d2link_fn = jnp.exp
        elif link == 'logistic':
            self.link_fn = softplus
            self.dlink_fn = sigmoid
            self.d2link_fn = sigmoid_diff
        else:
            raise NotImplementedError('link function not implemented')
        self.binsize = jnp.array(binsize)
        self.name = 'Poisson'
    
    def evaluate_log_likelihood(self, y_cens, f, extra):
        """
        Evaluate the Poisson log-likelihood:
            log p(yₙ|fₙ) = log Poisson(fₙ) = log(μʸ exp(-μ) / yₙ!)
        for μ = g(fₙ), where g() is the link function (exponential or logistic).
        We use the gamma function to evaluate yₙ! = gamma(yₙ + 1).
        Can be used to evaluate Q cubature points when performing moment matching.
        :param y: observed data (yₙ) [scalar]
        :param f: latent function value (fₙ) [Q, 1]
        :return:
            log Poisson(fₙ) = log(μʸ exp(-μ) / yₙ!) [Q, 1]
        """
        #mu = self.link_fn(f) * self.binsize
        #return jnp.squeeze(y * jnp.log(mu) - mu - gammaln(y + 1))
        threshold = extra
        mu = self.link_fn(f) * self.binsize
        log_likelihood_observed = y_cens * jnp.log(mu) - mu - gammaln(y_cens + 1)
        #print("log_likelihood_observed:", log_likelihood_observed.shape)
        
        # Log likelihood of censored values
        #cdf_value = jnp.sum(jnp.exp(jnp.arange(0, y_cens + 1) * jnp.log(mu) - mu - gammaln(jnp.arange(1, y_cens + 2))))
        cdf_value = poisson_cdf(y_cens, mu)
        #print("cdf_value:", cdf_value.shape)
        log_likelihood_censored = jnp.log(1-cdf_value + 0.0001)
    
        
        # Combine log likelihoods based on the censored indicators
        censoring_indicator = y_cens >= threshold
        log_likelihood_combined = censoring_indicator * log_likelihood_censored + (1 - censoring_indicator) * log_likelihood_observed
        
        #log_likelihood = jnp.squeeze(log_likelihood_observed)
        log_likelihood = jnp.squeeze(log_likelihood_combined)
        
        #print(log_likelihood_observed.shape, log_likelihood_censored.shape, censoring_indicator.shape, log_likelihood_combined.shape, log_likelihood.shape)
        return log_likelihood

    def variational_expectation_(self, y, m, v, extra, cubature=None):
        """
        If no custom variational expectation method is provided, we use cubature.
        """
        return variational_expectation_cubature(self, y, m, v, extra, cubature)

    def variational_expectation(self, y, m, v, extra, cubature=None):
        """
        Most likelihoods factorise across data points. For multi-latent models, a custom method must be implemented.
        """

        # align shapes and compute mask
        y = y.reshape(-1, 1, 1)
        thre = extra.reshape(-1, 1, 1)
        m = m.reshape(-1, 1, 1)
        v = jnp.diag(v).reshape(-1, 1, 1)
        mask = jnp.isnan(y)
        y = jnp.where(mask, m, y)
        
        ### Diffuse excess demand based on transition matrix for k diffusion steps
        transition_matrix = compute_transition_matrix(self.X_diffusion, 
                                                      diffusion_lengthscale=self.diffusion_lengthscale, 
                                                      diffusion_variance=self.diffusion_variance,
                                                      sink_prob=self.sink_prob)
        for step in range(self.diffusion_steps):
            excess_demand = jnp.maximum(m - thre, jnp.zeros(m.shape))
            diffused_excess = excess_demand[:,0,0] @ transition_matrix[:len(m),:]
            m = m[:,0,0] + diffused_excess[:len(m)]
            m = m.reshape(-1, 1, 1)
        
        # compute variational expectations and their derivatives
        var_exp, dE_dm, d2E_dm2 = vmap(self.variational_expectation_, (0, 0, 0, 0, None))(y, m, v, extra, cubature)

        # apply mask
        var_exp = jnp.where(jnp.squeeze(mask), 0., jnp.squeeze(var_exp))
        dE_dm = jnp.where(mask, jnp.nan, dE_dm)
        d2E_dm2 = jnp.where(mask, jnp.nan, d2E_dm2)

        return var_exp, jnp.squeeze(dE_dm, axis=2), jnp.diag(jnp.squeeze(d2E_dm2, axis=(1, 2)))

    def conditional_moments(self, f):
        """
        The first two conditional moments of a Poisson distribution are equal to the intensity:
            E[yₙ|fₙ] = link(fₙ)
            Var[yₙ|fₙ] = link(fₙ)
        """
        # TODO: multi-dim case
        return self.link_fn(f) * self.binsize, self.link_fn(f) * self.binsize
        # return self.link_fn(f) * self.binsize, vmap(jnp.diag, 1, 2)(self.link_fn(f) * self.binsize)


class MyDiffusedCensoredPoissonD(Likelihood, GeneralisedGaussNewtonMixin):
    """
    TODO: tidy docstring
    The Poisson likelihood:
        p(yₙ|fₙ) = Poisson(fₙ) = μʸ exp(-μ) / yₙ!
    where μ = g(fₙ) = mean = variance is the Poisson intensity.
    yₙ is non-negative integer count data.
    No closed form moment matching is available, se we default to using cubature.

    Letting Zy = gamma(yₙ+1) = yₙ!, we get log p(yₙ|fₙ) = log(g(fₙ))yₙ - g(fₙ) - log(Zy)
    The larger the intensity μ, the stronger the likelihood resembles a Gaussian
    since skewness = 1/sqrt(μ) and kurtosis = 1/μ.
    Two possible link functions:
    'exp':      link(fₙ) = exp(fₙ),         we have p(yₙ|fₙ) = exp(fₙyₙ-exp(fₙ))           / Zy.
    'logistic': link(fₙ) = log(1+exp(fₙ))), we have p(yₙ|fₙ) = logʸ(1+exp(fₙ)))(1+exp(fₙ)) / Zy.
    """
    def __init__(self,
                 X_diffusion,
                 binsize=1,
                 link='exp',
                 diffusion_steps=1,
                 diffusion_lengthscale=1.,
                 diffusion_variance=1.,
                 sink_prob=None,
                 fix_diffusion=False):
        """
        :param link: link function, either 'exp' or 'logistic'
        """
        super().__init__()
        self.X_diffusion = X_diffusion
        self.diffusion_steps = diffusion_steps
        if sink_prob is not None:
            assert sink_prob >= 0 and sink_prob <= 1
        self.sink_prob = sink_prob
        if fix_diffusion:
            self.diffusion_lengthscale = objax.StateVar(jnp.array(diffusion_lengthscale))
            self.diffusion_variance = objax.StateVar(jnp.array(diffusion_variance))
        else:
            self.diffusion_lengthscale = objax.TrainVar(jnp.array(diffusion_lengthscale))
            self.diffusion_variance = objax.TrainVar(jnp.array(diffusion_variance))
        if link == 'exp':
            self.link_fn = jnp.exp
            self.dlink_fn = jnp.exp
            self.d2link_fn = jnp.exp
        elif link == 'logistic':
            self.link_fn = softplus
            self.dlink_fn = sigmoid
            self.d2link_fn = sigmoid_diff
        else:
            raise NotImplementedError('link function not implemented')
        self.binsize = jnp.array(binsize)
        self.name = 'Poisson'
    
    def evaluate_log_likelihood(self, y_cens, f, extra):
        """
        Evaluate the Poisson log-likelihood:
            log p(yₙ|fₙ) = log Poisson(fₙ) = log(μʸ exp(-μ) / yₙ!)
        for μ = g(fₙ), where g() is the link function (exponential or logistic).
        We use the gamma function to evaluate yₙ! = gamma(yₙ + 1).
        Can be used to evaluate Q cubature points when performing moment matching.
        :param y: observed data (yₙ) [scalar]
        :param f: latent function value (fₙ) [Q, 1]
        :return:
            log Poisson(fₙ) = log(μʸ exp(-μ) / yₙ!) [Q, 1]
        """
        #mu = self.link_fn(f) * self.binsize
        #return jnp.squeeze(y * jnp.log(mu) - mu - gammaln(y + 1))
        threshold = extra
        mu = self.link_fn(f) * self.binsize
        log_likelihood_observed = y_cens * jnp.log(mu) - mu - gammaln(y_cens + 1)
        #print("log_likelihood_observed:", log_likelihood_observed.shape)
        
        # Log likelihood of censored values
        #cdf_value = jnp.sum(jnp.exp(jnp.arange(0, y_cens + 1) * jnp.log(mu) - mu - gammaln(jnp.arange(1, y_cens + 2))))
        cdf_value = poisson_cdf(y_cens, mu)
        #print("cdf_value:", cdf_value.shape)
        log_likelihood_censored = jnp.log(1-cdf_value + 0.0001)
    
        
        # Combine log likelihoods based on the censored indicators
        censoring_indicator = y_cens == threshold
        log_likelihood_combined = censoring_indicator * log_likelihood_censored + (1 - censoring_indicator) * log_likelihood_observed
        
        #log_likelihood = jnp.squeeze(log_likelihood_observed)
        log_likelihood = jnp.squeeze(log_likelihood_combined)
        
        #print(log_likelihood_observed.shape, log_likelihood_censored.shape, censoring_indicator.shape, log_likelihood_combined.shape, log_likelihood.shape)
        return log_likelihood

    def variational_expectation_(self, y, m, v, extra, cubature=None):
        """
        If no custom variational expectation method is provided, we use cubature.
        """
        return variational_expectation_cubature(self, y, m, v, extra, cubature)

    def variational_expectation(self, y, m, v, extra, cubature=None):
        """
        Most likelihoods factorise across data points. For multi-latent models, a custom method must be implemented.
        """

        # align shapes and compute mask
        y = y.reshape(-1, 1, 1)
        thre = extra.reshape(-1, 1, 1)
        m = m.reshape(-1, 1, 1)
        v = jnp.diag(v).reshape(-1, 1, 1)
        mask = jnp.isnan(y)
        y = jnp.where(mask, m, y)
        
        ### Diffuse excess demand based on transition matrix for k diffusion steps
        transition_matrix = compute_transition_matrix(self.X_diffusion, 
                                                      diffusion_lengthscale=self.diffusion_lengthscale, 
                                                      diffusion_variance=self.diffusion_variance,
                                                      sink_prob=self.sink_prob)
        m0 = m[:]
        for step in range(self.diffusion_steps):
            excess_demand = jnp.maximum(m - thre, jnp.zeros(m.shape))
            diffused_excess = excess_demand[:,0,0] @ (transition_matrix-jnp.eye(len(transition_matrix)))[:len(m),:]
            m = m[:,0,0] + diffused_excess[:len(m)]
            m = m.reshape(-1, 1, 1)
        
        m = m0 + jnp.maximum(m - m0, jnp.zeros(m.shape))
        
        # compute variational expectations and their derivatives
        var_exp, dE_dm, d2E_dm2 = vmap(self.variational_expectation_, (0, 0, 0, 0, None))(y, m, v, extra, cubature)

        # apply mask
        var_exp = jnp.where(jnp.squeeze(mask), 0., jnp.squeeze(var_exp))
        dE_dm = jnp.where(mask, jnp.nan, dE_dm)
        d2E_dm2 = jnp.where(mask, jnp.nan, d2E_dm2)

        return var_exp, jnp.squeeze(dE_dm, axis=2), jnp.diag(jnp.squeeze(d2E_dm2, axis=(1, 2)))

    def conditional_moments(self, f):
        """
        The first two conditional moments of a Poisson distribution are equal to the intensity:
            E[yₙ|fₙ] = link(fₙ)
            Var[yₙ|fₙ] = link(fₙ)
        """
        # TODO: multi-dim case
        return self.link_fn(f) * self.binsize, self.link_fn(f) * self.binsize
        # return self.link_fn(f) * self.binsize, vmap(jnp.diag, 1, 2)(self.link_fn(f) * self.binsize)


def negative_binomial_log_pmf(m, y, alpha):
    """
    Compute the log-likelihood of the Negative Binomial distribution.

    Parameters:
    y (array-like): Observed counts.
    m (array-like): Mean parameter of the distribution.
    alpha (float): alpha = 1/k where k is the dispersion parameter of the distribution.

    Returns:
    float: Log-likelihood of the Negative Binomial distribution.
    """
    k = 1 / alpha
    return (
        gammaln(k + y)
        - gammaln(y + 1)
        - gammaln(k)
        + y * jnp.log(m / (m + k))
        - k * jnp.log(1 + m * alpha)
    )


class MyNegativeBinomial(Likelihood, GeneralisedGaussNewtonMixin):
    
    def __init__(self,
                 alpha=1.0,
                 link='exp',
                 scale=1.0):
        """
        :param link: link function, either 'exp' or 'logistic'
        """
        super().__init__()
        if link == 'exp':
            self.link_fn = lambda mu: jnp.exp(mu)
            self.dlink_fn = lambda mu: jnp.exp(mu)
        elif link == 'logistic':
            self.link_fn = lambda mu: softplus(mu)
            self.dlink_fn = lambda mu: sigmoid(mu)
        else:
            raise NotImplementedError('link function not implemented')
        self.transformed_alpha = objax.TrainVar(jnp.array(softplus_inv(alpha)))
        self.scale = jnp.array(scale)
        self.name = 'Negative Binomial'

    @property
    def alpha(self):
        return softplus(self.transformed_alpha.value)

    def evaluate_log_likelihood(self, y, f, extra):
        """
        """
        return negative_binomial_log_pmf(self.link_fn(f) * self.scale, y, self.alpha)

    def conditional_moments(self, f):
        """
        """
        conditional_expectation = self.link_fn(f) * self.scale
        conditional_covariance = conditional_expectation + conditional_expectation ** 2 * self.alpha
        return conditional_expectation, conditional_covariance
    

# ------------------------------------------------ Diffused Censored GP - multivariate version

class MyGaussianMulti(MultiLatentLikelihood, GaussNewtonMixin):
    """
    The multivariate Gaussian likelihood:
        p(Yₙ|Fₙ) = 𝓝(Yₙ|Fₙ,Σ)
    """
    def __init__(self,
                 ndim=1,
                 variance=None,
                 fix_variance=False):
        """
        :param covariance: The observation noise covariance, Σ
        """
        stdev = jnp.sqrt(variance)
        if fix_variance:
            self.transformed_var = objax.StateVar(jnp.array(stdev))
        else:
            self.transformed_var = objax.TrainVar(jnp.array(stdev))
        assert len(variance) == ndim
        self.ndim = ndim
        super().__init__()
        self.name = 'Multiple Gaussians (independent)'
        self.link_fn = lambda f: f

    @property
    def variance(self):
        stdev = self.transformed_var.value
        return stdev ** 2

    def evaluate_log_likelihood(self, y, f, extra):
        """
        Evaluate the log-Gaussian function log𝓝(Yₙ|Fₙ,Σ).
        :param y: observed data Yₙ [D]
        :param f: mean, i.e. the latent function value Fₙ [D, 1]
        :return:
            log𝓝(Yₙ|Fₙ,Σ), where Σ is the observation noise covariance [Q, 1]
        """
        mask = jnp.isnan(y).reshape(-1)
        log_likelihood = 0
        for dim in range(self.ndim):
            # Log likelihood of observed values
            #return mvn_logpdf(y, f, self.covariance, mask=mask)
            #log_likelihood_observed = jnp.squeeze(-0.5 * jnp.log(2 * jnp.pi * self.variance[0]) - 0.5 * (y[0] - f[0]) ** 2 / self.variance[0]) \
            #             + jnp.squeeze(-0.5 * jnp.log(2 * jnp.pi * self.variance[1]) - 0.5 * (y[1] - f[1]) ** 2 / self.variance[1])      
            log_likelihood += jnp.sum(norm.logpdf(y[dim], loc=f[dim], scale=self.transformed_var.value[dim]))

        # Sum log likelihoods across all observations
        return log_likelihood

    def predict(self, mean_f, var_f, cubature=None):
        return mean_f, var_f + self.variance
    
    
class MyCensoredGaussianMulti(MultiLatentLikelihood, GaussNewtonMixin):
    """
    The multivariate Gaussian likelihood:
        p(Yₙ|Fₙ) = 𝓝(Yₙ|Fₙ,Σ)
    """
    def __init__(self,
                 ndim=1,
                 variance=None,
                 fix_variance=False):
        """
        :param covariance: The observation noise covariance, Σ
        """
        stdev = jnp.sqrt(variance)
        if fix_variance:
            self.transformed_var = objax.StateVar(jnp.array(stdev))
        else:
            self.transformed_var = objax.TrainVar(jnp.array(stdev))
        assert len(variance) == ndim
        self.ndim = ndim
        super().__init__()
        self.name = 'Multiple Censored Gaussians (independent)'
        self.link_fn = lambda f: f

    @property
    def variance(self):
        stdev = self.transformed_var.value
        return stdev ** 2

    def evaluate_log_likelihood(self, y_cens, f, extra):
        """
        Evaluate the log-Gaussian function log𝓝(Yₙ|Fₙ,Σ).
        :param y: observed data Yₙ [D]
        :param f: mean, i.e. the latent function value Fₙ [D, 1]
        :return:
            log𝓝(Yₙ|Fₙ,Σ), where Σ is the observation noise covariance [Q, 1]
        """
        mask = jnp.isnan(y_cens).reshape(-1)
        log_likelihood = 0
        for dim in range(self.ndim):
            # Log likelihood of observed values
            #return mvn_logpdf(y, f, self.covariance, mask=mask)
            #log_likelihood_observed = jnp.squeeze(-0.5 * jnp.log(2 * jnp.pi * self.variance[0]) - 0.5 * (y[0] - f[0]) ** 2 / self.variance[0]) \
            #             + jnp.squeeze(-0.5 * jnp.log(2 * jnp.pi * self.variance[1]) - 0.5 * (y[1] - f[1]) ** 2 / self.variance[1])      
            log_likelihood_observed = norm.logpdf(y_cens[dim], loc=f[dim], scale=self.transformed_var.value[dim])
        
            # Log likelihood of censored values
            #log_likelihood_censored = norm.logcdf(y[dim], loc=f[dim], scale=self.transformed_var.value[dim]) 
            log_likelihood_censored = jnp.log(1-norm.cdf(y_cens[dim], loc=f[dim], scale=self.transformed_var.value[dim]) + 0.01)
            
            # Combine log likelihoods based on the censored indicators
            censoring_indicator = y_cens[dim] == y_cens[dim+self.ndim]
            log_likelihood_combined = censoring_indicator * log_likelihood_censored + (1 - censoring_indicator) * log_likelihood_observed
            
            log_likelihood += jnp.sum(log_likelihood_combined)

        # Sum log likelihoods across all observations
        return log_likelihood

    def predict(self, mean_f, var_f, cubature=None):
        return mean_f, var_f + self.variance
    
    
class MyDiffusedCensoredGaussianMulti(MultiLatentLikelihood, GaussNewtonMixin):
    """
    The multivariate Gaussian likelihood:
        p(Yₙ|Fₙ) = 𝓝(Yₙ|Fₙ,Σ)
    """
    def __init__(self,
                 X_diffusion,
                 ndim=1,
                 variance=None,
                 diffusion_lengthscale=1.,
                 diffusion_variance=1.,
                 diffusion_steps=1,
                 fix_diffusion=False,
                 fix_variance=False):
        """
        :param covariance: The observation noise covariance, Σ
        """
        self.X_diffusion = X_diffusion
        self.diffusion_steps = diffusion_steps
        if fix_diffusion:
            self.diffusion_lengthscale = objax.StateVar(jnp.array(diffusion_lengthscale))
            self.diffusion_variance = objax.StateVar(jnp.array(diffusion_variance))
        else:
            self.diffusion_lengthscale = objax.TrainVar(jnp.array(diffusion_lengthscale))
            self.diffusion_variance = objax.TrainVar(jnp.array(diffusion_variance))
        stdev = jnp.sqrt(variance)
        if fix_variance:
            self.transformed_var = objax.StateVar(jnp.array(stdev))
        else:
            self.transformed_var = objax.TrainVar(jnp.array(stdev))
        assert len(variance) == ndim
        self.ndim = ndim
        super().__init__()
        self.name = 'Multiple Censored Gaussians (independent)'
        self.link_fn = lambda f: f
        

    @property
    def variance(self):
        stdev = self.transformed_var.value
        return stdev ** 2

    def evaluate_log_likelihood(self, y_cens, f, extra):
        """
        Evaluate the log-Gaussian function log𝓝(Yₙ|Fₙ,Σ).
        :param y: observed data Yₙ [D]
        :param f: mean, i.e. the latent function value Fₙ [D, 1]
        :return:
            log𝓝(Yₙ|Fₙ,Σ), where Σ is the observation noise covariance [Q, 1]
        """
        mask = jnp.isnan(y_cens).reshape(-1)
        #print("y_cens:", y_cens.shape)
        #print("f:", f.shape)
        
        ### Diffuse excess demand based on transition matrix for k diffusion steps
        transition_matrix = compute_transition_matrix(self.X_diffusion, 
                                                      diffusion_lengthscale=self.diffusion_lengthscale, 
                                                      diffusion_variance=self.diffusion_variance)
        #jax.debug.print("🤯 {x} 🤯", x=transition_matrix)
        for step in range(self.diffusion_steps):
            censoring_threshold = y_cens[self.ndim:,0]
            #jax.debug.print("y_cens: {x} f: {y} censoring_threshold: {z} 🤯", x=y_cens.shape, y=f.shape, z=censoring_threshold.shape)
            excess_demand = jnp.maximum(f - censoring_threshold, jnp.zeros(f.shape))
            #excess_demand = jnp.maximum(f - 0.5*np.ones(f.shape), jnp.zeros(f.shape))
            #excess_demand = jnp.maximum(f - 0.5, jnp.zeros(f.shape))
            #excess_demand = jnp.zeros(f.shape)
            #jax.debug.print("excess_demand: {x} 🤯", x=excess_demand)
            f = f + excess_demand @ transition_matrix
        
        ### Compute loglikelihood under censored Gaussian
        log_likelihood = 0
        for dim in range(self.ndim):
            # Log likelihood of observed values   
            log_likelihood_observed = norm.logpdf(y_cens[dim], loc=f[dim], scale=self.transformed_var.value[dim])
        
            # Log likelihood of censored values
            log_likelihood_censored = jnp.log(1-norm.cdf(y_cens[dim], loc=f[dim], scale=self.transformed_var.value[dim]) + 0.01)
            
            # Combine log likelihoods based on the censored indicators
            censoring_indicator = y_cens[dim] == y_cens[dim+self.ndim]
            log_likelihood_combined = censoring_indicator * log_likelihood_censored + (1 - censoring_indicator) * log_likelihood_observed
            
            log_likelihood += jnp.sum(log_likelihood_combined)

        # Sum log likelihoods across all observations
        return log_likelihood

    def predict(self, mean_f, var_f, cubature=None):
        return mean_f, var_f + self.variance
    
    
class MyDiffusedCensoredGaussianMultiB(MultiLatentLikelihood, GaussNewtonMixin):
    """
    The multivariate Gaussian likelihood:
        p(Yₙ|Fₙ) = 𝓝(Yₙ|Fₙ,Σ)
    """
    def __init__(self,
                 X_diffusion,
                 ndim=1,
                 variance=None,
                 diffusion_lengthscale=1.,
                 diffusion_variance=1.,
                 diffusion_steps=1,
                 fix_diffusion=False,
                 fix_variance=False):
        """
        :param covariance: The observation noise covariance, Σ
        """
        self.X_diffusion = X_diffusion
        self.diffusion_steps = diffusion_steps
        if fix_diffusion:
            self.diffusion_lengthscale = objax.StateVar(jnp.array(diffusion_lengthscale))
            self.diffusion_variance = objax.StateVar(jnp.array(diffusion_variance))
        else:
            self.diffusion_lengthscale = objax.TrainVar(jnp.array(diffusion_lengthscale))
            self.diffusion_variance = objax.TrainVar(jnp.array(diffusion_variance))
        stdev = jnp.sqrt(variance)
        if fix_variance:
            self.transformed_var = objax.StateVar(jnp.array(stdev))
        else:
            self.transformed_var = objax.TrainVar(jnp.array(stdev))
        assert len(variance) == ndim
        self.ndim = ndim
        super().__init__()
        self.name = 'Multiple Censored Gaussians (independent)'
        self.link_fn = lambda f: f
        

    @property
    def variance(self):
        stdev = self.transformed_var.value
        return stdev ** 2

    def evaluate_log_likelihood(self, y_cens, f, extra):
        """
        Evaluate the log-Gaussian function log𝓝(Yₙ|Fₙ,Σ).
        :param y: observed data Yₙ [D]
        :param f: mean, i.e. the latent function value Fₙ [D, 1]
        :return:
            log𝓝(Yₙ|Fₙ,Σ), where Σ is the observation noise covariance [Q, 1]
        """
        mask = jnp.isnan(y_cens).reshape(-1)
        #print("y_cens:", y_cens.shape)
        #print("f:", f.shape)
        
        ### Diffuse excess demand based on transition matrix for k diffusion steps
        transition_matrix = compute_transition_matrix(self.X_diffusion, 
                                                      diffusion_lengthscale=self.diffusion_lengthscale, 
                                                      diffusion_variance=self.diffusion_variance)
        #jax.debug.print("🤯 {x} 🤯", x=transition_matrix)
        for step in range(self.diffusion_steps):
            censoring_threshold = y_cens[self.ndim:,0]
            #jax.debug.print("y_cens: {x} f: {y} censoring_threshold: {z} 🤯", x=y_cens.shape, y=f.shape, z=censoring_threshold.shape)
            excess_demand = jnp.maximum(f - censoring_threshold, jnp.zeros(f.shape))
            #excess_demand = jnp.maximum(f - 0.5*np.ones(f.shape), jnp.zeros(f.shape))
            #excess_demand = jnp.maximum(f - 0.5, jnp.zeros(f.shape))
            #excess_demand = jnp.zeros(f.shape)
            #jax.debug.print("excess_demand: {x} 🤯", x=excess_demand)
            f = f + excess_demand @ transition_matrix
    
        
        ### Compute loglikelihood under censored Gaussian
        log_likelihood = 0
        for dim in range(self.ndim):
            # Log likelihood of observed values   
            log_likelihood_observed = norm.logpdf(y_cens[dim], loc=f[dim], scale=self.transformed_var.value[dim])
        
            # Log likelihood of censored values
            log_likelihood_censored = jnp.log(1-norm.cdf(y_cens[dim], loc=f[dim], scale=self.transformed_var.value[dim]) + 0.01)
            
            # Combine log likelihoods based on the censored indicators
            censoring_indicator = y_cens[dim] >= y_cens[dim+self.ndim]
            log_likelihood_combined = censoring_indicator * log_likelihood_censored + (1 - censoring_indicator) * log_likelihood_observed
            
            log_likelihood += jnp.sum(log_likelihood_combined)

        # Sum log likelihoods across all observations
        return log_likelihood

    def predict(self, mean_f, var_f, cubature=None):
        return mean_f, var_f + self.variance
    
    
class MyDiffusedCensoredGaussianMultiC(MultiLatentLikelihood, GaussNewtonMixin):
    """
    The multivariate Gaussian likelihood:
        p(Yₙ|Fₙ) = 𝓝(Yₙ|Fₙ,Σ)
    """
    def __init__(self,
                 X_diffusion,
                 ndim=1,
                 variance=None,
                 diffusion_lengthscale=1.,
                 diffusion_variance=1.,
                 diffusion_steps=1,
                 fix_diffusion=False,
                 fix_variance=False):
        """
        :param covariance: The observation noise covariance, Σ
        """
        self.X_diffusion = X_diffusion
        self.diffusion_steps = diffusion_steps
        if fix_diffusion:
            self.diffusion_lengthscale = objax.StateVar(jnp.array(diffusion_lengthscale))
            self.diffusion_variance = objax.StateVar(jnp.array(diffusion_variance))
        else:
            self.diffusion_lengthscale = objax.TrainVar(jnp.array(diffusion_lengthscale))
            self.diffusion_variance = objax.TrainVar(jnp.array(diffusion_variance))
        stdev = jnp.sqrt(variance)
        if fix_variance:
            self.transformed_var = objax.StateVar(jnp.array(stdev))
        else:
            self.transformed_var = objax.TrainVar(jnp.array(stdev))
        assert len(variance) == ndim
        self.ndim = ndim
        super().__init__()
        self.name = 'Multiple Censored Gaussians (independent)'
        self.link_fn = lambda f: f
        

    @property
    def variance(self):
        stdev = self.transformed_var.value
        return stdev ** 2

    def evaluate_log_likelihood(self, y_cens, f, extra):
        """
        Evaluate the log-Gaussian function log𝓝(Yₙ|Fₙ,Σ).
        :param y: observed data Yₙ [D]
        :param f: mean, i.e. the latent function value Fₙ [D, 1]
        :return:
            log𝓝(Yₙ|Fₙ,Σ), where Σ is the observation noise covariance [Q, 1]
        """
        mask = jnp.isnan(y_cens).reshape(-1)
        #print("y_cens:", y_cens.shape)
        #print("f:", f.shape)
        
        ### Diffuse excess demand based on transition matrix for k diffusion steps
        transition_matrix = compute_transition_matrix(self.X_diffusion, 
                                                      diffusion_lengthscale=self.diffusion_lengthscale, 
                                                      diffusion_variance=self.diffusion_variance)
        #jax.debug.print("🤯 {x} 🤯", x=transition_matrix)
        f0 = f[:]
        for step in range(self.diffusion_steps):
            censoring_threshold = y_cens[self.ndim:,0]
            #jax.debug.print("y_cens: {x} f: {y} censoring_threshold: {z} 🤯", x=y_cens.shape, y=f.shape, z=censoring_threshold.shape)
            excess_demand = jnp.maximum(f - censoring_threshold, jnp.zeros(f.shape))
            #excess_demand = jnp.maximum(f - 0.5*np.ones(f.shape), jnp.zeros(f.shape))
            #excess_demand = jnp.maximum(f - 0.5, jnp.zeros(f.shape))
            #excess_demand = jnp.zeros(f.shape)
            #jax.debug.print("excess_demand: {x} 🤯", x=excess_demand)
            #f = f + excess_demand @ transition_matrix
            f = f + excess_demand @ (transition_matrix-jnp.eye(len(transition_matrix)))
        
        f = f0 + jnp.maximum(f - f0, jnp.zeros(f0.shape))
        
        ### Compute loglikelihood under censored Gaussian
        log_likelihood = 0
        for dim in range(self.ndim):
            # Log likelihood of observed values   
            log_likelihood_observed = norm.logpdf(y_cens[dim], loc=f[dim], scale=self.transformed_var.value[dim])
        
            # Log likelihood of censored values
            log_likelihood_censored = jnp.log(1-norm.cdf(y_cens[dim], loc=f[dim], scale=self.transformed_var.value[dim]) + 0.01)
            
            # Combine log likelihoods based on the censored indicators
            censoring_indicator = y_cens[dim] >= y_cens[dim+self.ndim]
            log_likelihood_combined = censoring_indicator * log_likelihood_censored + (1 - censoring_indicator) * log_likelihood_observed
            
            log_likelihood += jnp.sum(log_likelihood_combined)

        # Sum log likelihoods across all observations
        return log_likelihood

    def predict(self, mean_f, var_f, cubature=None):
        return mean_f, var_f + self.variance
    
    
class MyDiffusedCensoredGaussianMultiD(MultiLatentLikelihood, GaussNewtonMixin):
    """
    The multivariate Gaussian likelihood:
        p(Yₙ|Fₙ) = 𝓝(Yₙ|Fₙ,Σ)
    """
    def __init__(self,
                 X_diffusion,
                 ndim=1,
                 variance=None,
                 diffusion_lengthscale=1.,
                 diffusion_variance=1.,
                 diffusion_steps=1,
                 fix_diffusion=False,
                 fix_variance=False):
        """
        :param covariance: The observation noise covariance, Σ
        """
        self.X_diffusion = X_diffusion
        self.diffusion_steps = diffusion_steps
        if fix_diffusion:
            self.diffusion_lengthscale = objax.StateVar(jnp.array(diffusion_lengthscale))
            self.diffusion_variance = objax.StateVar(jnp.array(diffusion_variance))
        else:
            self.diffusion_lengthscale = objax.TrainVar(jnp.array(diffusion_lengthscale))
            self.diffusion_variance = objax.TrainVar(jnp.array(diffusion_variance))
        stdev = jnp.sqrt(variance)
        if fix_variance:
            self.transformed_var = objax.StateVar(jnp.array(stdev))
        else:
            self.transformed_var = objax.TrainVar(jnp.array(stdev))
        assert len(variance) == ndim
        self.ndim = ndim
        super().__init__()
        self.name = 'Multiple Censored Gaussians (independent)'
        self.link_fn = lambda f: f
        

    @property
    def variance(self):
        stdev = self.transformed_var.value
        return stdev ** 2

    def evaluate_log_likelihood(self, y_cens, f, extra):
        """
        Evaluate the log-Gaussian function log𝓝(Yₙ|Fₙ,Σ).
        :param y: observed data Yₙ [D]
        :param f: mean, i.e. the latent function value Fₙ [D, 1]
        :return:
            log𝓝(Yₙ|Fₙ,Σ), where Σ is the observation noise covariance [Q, 1]
        """
        mask = jnp.isnan(y_cens).reshape(-1)
        #print("y_cens:", y_cens.shape)
        #print("f:", f.shape)
        
        ### Diffuse excess demand based on transition matrix for k diffusion steps
        transition_matrix = compute_transition_matrix(self.X_diffusion, 
                                                      diffusion_lengthscale=self.diffusion_lengthscale, 
                                                      diffusion_variance=self.diffusion_variance)
        #jax.debug.print("🤯 {x} 🤯", x=transition_matrix)
        f0 = f[:]
        for step in range(self.diffusion_steps):
            censoring_threshold = y_cens[self.ndim:,0]
            #jax.debug.print("y_cens: {x} f: {y} censoring_threshold: {z} 🤯", x=y_cens.shape, y=f.shape, z=censoring_threshold.shape)
            excess_demand = jnp.maximum(f - censoring_threshold, jnp.zeros(f.shape))
            #excess_demand = jnp.maximum(f - 0.5*np.ones(f.shape), jnp.zeros(f.shape))
            #excess_demand = jnp.maximum(f - 0.5, jnp.zeros(f.shape))
            #excess_demand = jnp.zeros(f.shape)
            #jax.debug.print("excess_demand: {x} 🤯", x=excess_demand)
            #f = f + excess_demand @ transition_matrix
            f = f + excess_demand @ (transition_matrix-jnp.eye(len(transition_matrix)))
        
        f = f0 + jnp.maximum(f - f0, jnp.zeros(f0.shape))
        
        ### Compute loglikelihood under censored Gaussian
        log_likelihood = 0
        for dim in range(self.ndim):
            # Log likelihood of observed values   
            log_likelihood_observed = norm.logpdf(y_cens[dim], loc=f[dim], scale=self.transformed_var.value[dim])
        
            # Log likelihood of censored values
            log_likelihood_censored = jnp.log(1-norm.cdf(y_cens[dim], loc=f[dim], scale=self.transformed_var.value[dim]) + 0.01)
            
            # Combine log likelihoods based on the censored indicators
            censoring_indicator = y_cens[dim] == y_cens[dim+self.ndim]
            log_likelihood_combined = censoring_indicator * log_likelihood_censored + (1 - censoring_indicator) * log_likelihood_observed
            
            log_likelihood += jnp.sum(log_likelihood_combined)

        # Sum log likelihoods across all observations
        return log_likelihood

    def predict(self, mean_f, var_f, cubature=None):
        return mean_f, var_f + self.variance
    
    
# ------------------------------------------------ Covariance functions and helper functions

def _square_scaled_dist(X, lengthscale, Z=None):
    r"""
    Returns :math:`\|\frac{X-Z}{l}\|^2`.
    """
    if Z is None:
        Z = X
    if X.shape[1] != Z.shape[1]:
        raise ValueError("Inputs must have the same number of features.")

    scaled_X = X / lengthscale
    scaled_Z = Z / lengthscale
    X2 = (scaled_X**2).sum(1, keepdims=True)
    Z2 = (scaled_Z**2).sum(1, keepdims=True)
    XZ = jnp.matmul(scaled_X,scaled_Z.transpose())
    r2 = X2 - 2 * XZ + Z2.transpose()
    return jnp.clip(r2, a_min=0)

def adjaciency_matrix(X, lengthscale):
    r"""
    Computes adjaciency matrix for an input matrix X.
    """
    # compute adjacency matrix
    W = jnp.exp(-_square_scaled_dist(X, lengthscale=lengthscale))
    return W
    
def laplacian(X, lengthscale):
    r"""
    Computes Laplacian matrix for an input matrix X.
    """
    # compute adjacency matrix
    W = adjaciency_matrix(X, lengthscale=lengthscale)
    print("W:", W)
    
    # compute degree matrix
    D = np.diag(np.dot(W, np.ones(X.shape[0])))
    #print("D:", D)
    
    # compute Laplacian matrix
    Lap = D - W
    print("Lap:", Lap)
    return Lap

def norm_adjaciency_matrix(X, lengthscale):
    r"""
    Computes normalized adjaciency matrix for an input matrix X.
    """
    # compute adjacency matrix
    W = adjaciency_matrix(X, lengthscale=lengthscale)
    print("W:", W)
    
    # normalize adjaciency matrix
    W_norm = W / W.sum(axis=1)[:,None]
    print("W normalized:", W_norm)
    return  W_norm

def k_norm_adj(X, lengthscale):
    r"""
    Computes custom kernel based adjaciency matrix for an input matrix X.
    """
    return norm_adjaciency_matrix(X, lengthscale=lengthscale)

def k_RL(X, lengthscale, beta=0.5):
    r"""
    Computes regularized Laplacian kernel matrix for an input matrix X.
    """
    Lap = laplacian(X, lengthscale=lengthscale)
    return np.linalg.pinv(np.eye(X.shape[0]) + beta*Lap)

def k_Diff(X, lengthscale, beta=0.5):
    r"""
    Computes diffusion kernel matrix for an input matrix X. 
    """
    # compute similarity matrix based on kernel
    Lap = laplacian(X, lengthscale)
    return np.exp(-beta*Lap)

def compute_transition_matrix(X_diffusion, diffusion_lengthscale, diffusion_variance=1, beta=0.5, sink_prob=None):
    r"""
    Compute transition matrix based on RBF kernel to be used in the diffusion process. 
    """
    # compute similarity matrix based on kernel
    r2 = _square_scaled_dist(X_diffusion, diffusion_lengthscale)
    similarity_matrix = diffusion_variance * jnp.exp(-beta*r2)
    #print("similarity_matrix:", similarity_matrix)
            
    # compute transition matrix based on similarity matrix
    num_nodes = X_diffusion.shape[0]
    if sink_prob is None:
        transition_matrix = similarity_matrix - jnp.eye(num_nodes)
        transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True) # row normalize
    else:
        transition_matrix = jnp.zeros((num_nodes+1,num_nodes+1)) # initialize transition probs with zeros
        transition_matrix = transition_matrix.at[:num_nodes,:num_nodes].set(similarity_matrix - jnp.eye(num_nodes)) 
        # set last column such that, when the matrix is row-normalized, it becomes equal to sink_prob
        transition_matrix = transition_matrix.at[:num_nodes,num_nodes].set((1/(1-sink_prob)-1)*transition_matrix[:num_nodes,:num_nodes].sum(axis=1))
        transition_matrix = transition_matrix.at[num_nodes,num_nodes].set(1) # once in the sink node, never leaves (transition prob to other nodes is 0)
        transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True) # row normalize
        
    return transition_matrix