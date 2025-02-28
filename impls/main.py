import json
import os
import random
import time
from collections import defaultdict

import jax
import numpy as np
import tqdm
from absl import app, flags
from agents import agents
from ml_collections import config_flags
from utils.datasets import Dataset, GCDataset, HGCDataset
from utils.env_utils import make_env_and_datasets
from utils.evaluation import evaluate_no_render
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import get_flag_dict
from tensorboardX import SummaryWriter


FLAGS = flags.FLAGS

flags.DEFINE_string('exp_name', 'exp', 'Experiment name.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'antmaze-large-navigate-v0', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'ckpts/', 'Save directory.')
flags.DEFINE_string('tf_log_dir', 'tf-logs/', 'Tensorboard log directory.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')

flags.DEFINE_integer('train_steps', 1000000, 'Number of training steps.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 100000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 1000000, 'Saving interval.')

flags.DEFINE_integer('eval_tasks', None, 'Number of tasks to evaluate (None for all).')
flags.DEFINE_integer('eval_episodes', 20, 'Number of episodes for each task.')
flags.DEFINE_float('eval_temperature', 0, 'Actor temperature for evaluation.')
flags.DEFINE_float('eval_gaussian', None, 'Action Gaussian noise for evaluation.')
flags.DEFINE_integer('eval_on_cpu', 1, 'Whether to evaluate on CPU.')

config_flags.DEFINE_config_file('agent', 'agents/gciql.py', lock_config=False)


def main(_):
    # Check if the agent name is provided
    config = FLAGS.agent
    exp_name = config['agent_name'] if FLAGS.exp_name == 'exp' else FLAGS.exp_name

    # Set up logger.
    exp_name = f"{exp_name}__{FLAGS.env_name}__{int(time.time())}__{FLAGS.seed}"
    writer = SummaryWriter(f"{FLAGS.tf_log_dir}/{exp_name}")

    agent_dir = os.path.join(FLAGS.save_dir, f"{exp_name}__agent")
    os.makedirs(agent_dir, exist_ok=True)
    flag_dict = get_flag_dict()
    with open(os.path.join(agent_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    # Set up environment and dataset.
    env, train_dataset, val_dataset = make_env_and_datasets(FLAGS.env_name, frame_stack=config['frame_stack'])

    dataset_class = {
        'GCDataset': GCDataset,
        'HGCDataset': HGCDataset,
    }[config['dataset_class']]
    train_dataset = dataset_class(Dataset.create(**train_dataset), config)
    if val_dataset is not None:
        val_dataset = dataset_class(Dataset.create(**val_dataset), config)

    # Initialize agent.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    example_batch = train_dataset.sample(1)
    if config['discrete']:
        # Fill with the maximum action to let the agent know the action space size.
        example_batch['actions'] = np.full_like(example_batch['actions'], env.action_space.n - 1)

    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )

    # Restore agent.
    if FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    # Train agent.
    first_time = time.time()
    last_time = time.time()
    for i in tqdm.tqdm(range(1, FLAGS.train_steps + 1), smoothing=0.1, dynamic_ncols=True):
        # Update agent.
        batch = train_dataset.sample(config['batch_size'])
        agent, update_info = agent.update(batch)

        # Log metrics.
        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            if val_dataset is not None:
                val_batch = val_dataset.sample(config['batch_size'])
                _, val_info = agent.total_loss(val_batch, grad_params=None)
                train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})
            train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            last_time = time.time()
            for k, v in train_metrics.items():
                writer.add_scalar(k, v, i)

        # Evaluate agent.
        if i == 1 or i % FLAGS.eval_interval == 0:
            if FLAGS.eval_on_cpu:
                eval_agent = jax.device_put(agent, device=jax.devices('cpu')[0])
            else:
                eval_agent = agent
            eval_metrics = {}
            overall_metrics = defaultdict(list)
            task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, 'task_infos') else env.task_infos
            num_tasks = FLAGS.eval_tasks if FLAGS.eval_tasks is not None else len(task_infos)
            for task_id in tqdm.trange(1, num_tasks + 1):
                task_name = task_infos[task_id - 1]['task_name']
                eval_info, trajs = evaluate_no_render(
                    agent=eval_agent,
                    env=env,
                    task_id=task_id,
                    config=config,
                    num_eval_episodes=FLAGS.eval_episodes,
                    eval_temperature=FLAGS.eval_temperature,
                    eval_gaussian=FLAGS.eval_gaussian,
                )
                metric_names = ['success']
                eval_metrics.update(
                    {f'evaluation/{task_name}_{k}': v for k, v in eval_info.items() if k in metric_names}
                )
                for k, v in eval_info.items():
                    if k in metric_names:
                        overall_metrics[k].append(v)
            for k, v in overall_metrics.items():
                eval_metrics[f'evaluation/overall_{k}'] = np.mean(v)

            for k, v in eval_metrics.items():
                writer.add_scalar(k, v, i)

        # Save agent.
        if i % FLAGS.save_interval == 0:
            save_agent(agent, agent_dir, i)

    writer.close()


if __name__ == '__main__':
    app.run(main)
