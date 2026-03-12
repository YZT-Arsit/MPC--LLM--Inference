import flax
import jax
from typing import Optional, Any, Callable, Tuple, Union
from flax.linen.linear import Array
import jax.numpy as jnp
from flax.linen.module import Module
from flax.linen import initializers

default_embed_init = flax.linen.initializers.variance_scaling(
  1.0, 'fan_in', 'normal', out_axis=0
)

class HackEmbed(flax.linen.module.Module):
    num_embeddings: int
    features: int
    dtype: Optional[Any] = None
    param_dtype: Any = jnp.float32
    embedding_init: Callable[[Any, Tuple[int, ...], Any], Array] = default_embed_init

    def setup(self):
        print("Using hack embed")
        pass

    def __call__(self, inputs: Array) -> Array:
        inputs = jnp.asarray(inputs, dtype=self.dtype)
        return jax.numpy.repeat(inputs[..., None], self.features, axis=-1)
        if not jnp.issubdtype(inputs.dtype, jnp.integer):
            raise ValueError('Input type must be an integer or unsigned integer.')
        # Use take because fancy indexing numpy arrays with JAX indices does not
        # work correctly.
        (embedding,) = promote_dtype(
            self.embedding, dtype=self.dtype, inexact=False
        )
        return jnp.take(embedding, inputs, axis=0)


    def attend(self, query: Array) -> Array:
        assert False

default_kernel_init = initializers.lecun_normal()
DotGeneralT = Callable[..., Array]
Shape = Tuple[int, ...]
PrecisionLike = Union[
  None,
  str,
  jax.lax.Precision,
  Tuple[str, str],
  Tuple[jax.lax.Precision, jax.lax.Precision],
]

class HackDense(flax.linen.module.Module):
  features: int
  use_bias: bool = True
  dtype: Optional[Any] = None
  param_dtype: Any = jnp.float32
  precision: PrecisionLike = None
  kernel_init: Callable[[Any, Shape, Any], Array] = default_kernel_init
  bias_init: Callable[
    [Any, Shape, Any], Array
  ] = initializers.zeros_init()
  # Deprecated. Will be removed.
  dot_general: Optional[DotGeneralT] = None
  dot_general_cls: Any = None

  def __call__(self, inputs: Array) -> Array:
    return inputs
    kernel = self.param(
      'kernel',
      self.kernel_init,
      (jnp.shape(inputs)[-1], self.features),
      self.param_dtype,
    )
    if self.use_bias:
      bias = self.param(
        'bias', self.bias_init, (self.features,), self.param_dtype
      )
    else:
      bias = None
    inputs, kernel, bias = promote_dtype(inputs, kernel, bias, dtype=self.dtype)

    if self.dot_general_cls is not None:
      dot_general = self.dot_general_cls()
    elif self.dot_general is not None:
      dot_general = self.dot_general
    else:
      dot_general = lax.dot_general
    y = dot_general(
      inputs,
      kernel,
      (((inputs.ndim - 1,), (0,)), ((), ())),
      precision=self.precision,
    )
    if bias is not None:
      y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
    return y