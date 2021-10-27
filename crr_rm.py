import copy
from typing import Sequence
import acme
from acme import specs
from acme.agents.tf import actors
from acme.agents.tf import crr
from acme.tf import networks as acme_networks
from acme.tf import utils as tf2_utils
from acme.utils import loggers
import numpy as np
import dm_control_suite
import networks
import sonnet as snt
import tensorflow as tf
import dm_env
from dm_env import specs as dm_specs
import sys

obs_dim = 19
action_dim = 7
environment_spec = specs.EnvironmentSpec(
      observations=specs.Array(shape=(obs_dim,), dtype='float32', name='observations'),
      actions=specs.Array(shape=(action_dim,), dtype='float32', name='actions'),
      rewards=None,
      discounts=None,
      )

print(environment_spec)

def make_networks(
    action_spec: specs.BoundedArray,
    policy_lstm_sizes: Sequence[int] = None,
    critic_lstm_sizes: Sequence[int] = None,
    num_components: int = 5,
    vmin: float = 0.,
    vmax: float = 100.,
    num_atoms: int = 21,
):
  """Creates recurrent networks with GMM head used by the agents."""

  action_size = np.prod(action_spec.shape, dtype=int)
  actor_head = acme_networks.MultivariateGaussianMixture(
      num_components=num_components, num_dimensions=action_size)

  if policy_lstm_sizes is None:
    policy_lstm_sizes = []#[1024, 1024]
  if critic_lstm_sizes is None:
    critic_lstm_sizes = []#[1024, 1024]

  actor_neck = acme_networks.LayerNormAndResidualMLP(hidden_size=1024,
                                                     num_blocks=4)
  actor_encoder = networks.ControlNetwork(
      proprio_encoder_size=300,
      activation=tf.nn.relu)

  policy_lstms = [snt.LSTM(s) for s in policy_lstm_sizes]

  policy_network = snt.DeepRNN([actor_encoder, actor_neck] + policy_lstms +
                               [actor_head])

  critic_encoder = networks.ControlNetwork(
      proprio_encoder_size=400,
      activation=tf.nn.relu)
  critic_neck = acme_networks.LayerNormAndResidualMLP(
      hidden_size=1024, num_blocks=4)
  distributional_head = acme_networks.DiscreteValuedHead(
      vmin=vmin, vmax=vmax, num_atoms=num_atoms)
  critic_lstms = [snt.LSTM(s) for s in critic_lstm_sizes]
  critic_network = acme_networks.CriticDeepRNN([critic_encoder, critic_neck] +
                                                critic_lstms + [
                                                    distributional_head,
                                                ])

  return {
      'policy': policy_network,
      'critic': critic_network,
  }

try:
  tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
  tf.config.experimental_connect_to_cluster(tpu)
  tf.tpu.experimental.initialize_tpu_system(tpu)
  accelerator_strategy = snt.distribute.TpuReplicator()
  print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
except ValueError:
  print('Running on CPU or GPU (no TPUs available)')
  accelerator_strategy = snt.distribute.Replicator()


"""CRR learner"""

action_spec = environment_spec.actions
action_size = np.prod(action_spec.shape, dtype=int)

with accelerator_strategy.scope():
  dataset = dm_control_suite.dataset(
    'gs://rl_unplugged/',
    data_path=task.data_path,
    shapes=task.shapes,
    uint8_features=task.uint8_features,
    num_threads=1,
    batch_size=batch_size,
    num_shards=num_shards,
    sarsa=False)
  # CRR learner assumes that the dataset samples don't have metadata,
  # so let's remove it here.
  dataset = dataset.map(lambda sample: sample.data)
  nets = make_networks(action_spec)
  policy_network, critic_network = nets['policy'], nets['critic']

  # Create the target networks
  target_policy_network = copy.deepcopy(policy_network)
  target_critic_network = copy.deepcopy(critic_network)

  # Create variables.
  tf2_utils.create_variables(network=policy_network,
                            input_spec=[environment_spec.observations])
  tf2_utils.create_variables(network=critic_network,
                            input_spec=[environment_spec.observations,
                                        environment_spec.actions])
  tf2_utils.create_variables(network=target_policy_network,
                            input_spec=[environment_spec.observations])
  tf2_utils.create_variables(network=target_critic_network,
                            input_spec=[environment_spec.observations,
                                        environment_spec.actions])


logger = loggers.TerminalLogger(label='training', time_delta=1.,
                                print_fn=print)

# The learner updates the parameters (and initializes them).
learner = crr.RCRRLearner(
    policy_network=policy_network,
    critic_network=critic_network,
    accelerator_strategy=accelerator_strategy,
    target_policy_network=target_policy_network,
    target_critic_network=target_critic_network,
    dataset=dataset,
    discount=0.99,
    logger=logger,
    checkpoint=True,
    target_update_period=100)


"""Training Loop"""

for _ in range(50000):
  learner.step()
  print("at step {}".format(_))

  if _ % 100 == 0:
    # Create a logger.
    logger = loggers.TerminalLogger(label='evaluation', time_delta=1.,
                                print_fn=print)

    # Create an environment loop.
    loop = acme.EnvironmentLoop(
      environment=environment,
      actor=actors.RecurrentActor(policy_network),
      logger=logger)
    loop.run(10)
