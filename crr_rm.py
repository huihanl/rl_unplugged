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

from rlkit.data_management.obs_dict_replay_buffer import ObsDictReplayBuffer

import robomimic
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
from robomimic.envs.env_base import EnvBase
import robomimic.envs.env_rlkit as ER

from rlkit.misc.roboverse_utils import add_data_to_buffer, \
    add_data_to_buffer_new, VideoSaveFunctionBullet, get_buffer_size

def create_env_from_dataset(dataset_path):
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
    env_meta['env_kwargs']['use_camera_obs'] = True
    env_meta['env_kwargs']['render_camera'] = 'agentview'
    print(env_meta)
    env = EnvUtils.create_env_from_metadata(env_meta=env_meta,
                                            render=False,
                                            render_offscreen=False,#True,
                                            use_image_obs=False)
    return env

import sys
dataset_path_env = sys.argv[1]
obs_keys = ["state"]
img_size = 84
buffer = sys.argv[2]

eval_env = create_env_from_dataset(dataset_path_env)

eval_env = ER.EnvRLkitWrapper(eval_env,
                              obs_img_dim='img_size',  # rendered image size
                              transpose_image=True,  # transpose for pytorch by default
                              camera_names=['frontview'],
                              observation_mode='states',
                              )

expl_env = eval_env
action_dim = eval_env.action_space.low.size

image_modalities = ["image"]
obs_modality_specs = {
    "obs": {
        "low_dim": [],  # technically unused, so we don't have to specify all of them
        "image": image_modalities,
    }
}
ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs)

observation_keys = obs_keys # type is list
print(observation_keys)

if 'state' in observation_keys:
    state_observation_dim = eval_env.observation_space.spaces['state'].low.size
else:
    state_observation_dim = 0

with open(buffer, 'rb') as fl:
    data = np.load(fl, allow_pickle=True)
num_transitions = get_buffer_size(data)
max_replay_buffer_size = num_transitions + 10
replay_buffer = ObsDictReplayBuffer(
    max_replay_buffer_size,
    expl_env,
    observation_keys=observation_keys
)

obs_dim = 19
action_dim = 7
environment_spec = specs.EnvironmentSpec(
      observations=specs.Array(shape=(obs_dim,), dtype='float32', name='observations'),
      actions=specs.Array(shape=(action_dim,), dtype='float32', name='actions'),
      rewards=None,
      discounts=None,
      )

print(environment_spec)

add_data_to_buffer(data, replay_buffer, observation_keys)

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
    replay_buffer=replay_buffer,
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
    loop = acme.EnvironmentLoopRM(
      environment=eval_env,
      actor=actors.RecurrentActor(policy_network),
      logger=logger)
    loop.run(num_trials=10, max_steps=100)

