# Script must be run inside kuberlab task. 
#
import os
from mlboardclient.api import client


mlboard = client.Client()
data = {
    'lr.adjust_learning_rate_by': 'train',
    'lr.keep_checkpoint_max': 2,
    'lr.log_step_count_steps': 5,
    'lr.max_input_seq_length': 3510,
    'lr.max_target_seq_length': 600,
    'lr.provider': 'local',
    'lr.save_checkpoints_secs': 600,
    'lr.save_checkpoints_steps': 20,
    'lr.save_summary_steps': 10,
    'without_main_int': 125,
    'without_main_str': 'param',
    'name': 'my-super-job-%s' % os.environ.get('BUILD_ID', '1'),
    'main.batch_size': 8,
    'main.beam_search_decoder': False,
    'main.checkpoint_path': '/notebooks/training/EXP126.5x1024_B8',
    'main.grad_clip': 1,
    'main.hidden_size': 1024,
    'main.input_keep_prob': 0.8,
    'main.input_layer': 'CNN',
    'main.learning_rate': 0.0001,
    'main.lr_decay_op_count': 6,
    'main.lstm_direction_type': 'bidirectional',
    'main.name': '126x5_B1024',
    'main.num_layers': 5,
    'main.output_keep_prob': 0,
    'main.rnn_state_reset_ratio': 1,
    'main.rnn_type': 'CudnnLSTM',
    'main.vq_feature_size': 161,
}

mlboard.update_task_info(data)
