import argparse
import codecs
import json
import logging
import os
import re

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.models import save_model

logging.getLogger().setLevel(logging.INFO)

os.system('mkdir /opt/ml/model/code')
os.system('cp inference.py /opt/ml/model/code')
os.system('cp requirements.txt /opt/ml/model/code')

def process_input(epochs, batch_size, channel, channel_name):
    data = pd.read_csv(os.path.join(channel, channel_name) + '.csv')
    data = data.sample(frac=1)
    data_rep = data.iloc[:,:]
    for i in range(epochs-1):
        data_rep = data_rep.append(data)
    features = data_rep.iloc[:,:-1].to_numpy()
    labels = data_rep.iloc[:,-1].to_numpy()
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.batch(batch_size)
    return (iter(dataset),len(data_rep))

def main(args):
    if 'sourcedir.tar.gz' in args.tensorboard_dir:
        tensorboard_dir = re.sub('source/sourcedir.tar.gz', 'model', args.tensorboard_dir)
    else:
        tensorboard_dir = args.tensorboard_dir
    logging.info('Writing TensorBoard logs to {}'.format(tensorboard_dir))
    
    logging.info('getting data')
    train_data, len_train = process_input(args.epochs, args.batch_size, args.train, 'train')
    eval_data, len_eval = process_input(args.epochs, 1000, args.evaluation, 'evaluation')
    validation_data, len_validation = process_input(args.epochs, args.batch_size, args.validation, 'validation')
    
    logging.info('configuring model')

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(16, activation = 'relu'),
        tf.keras.layers.Dense(1, activation = 'sigmoid')
    ])

    model.compile(
        loss = 'binary_crossentropy',
        optimizer = tf.keras.optimizers.Adam(lr=np.sqrt(10)*1e-4),
        metrics = ['accuracy']
    )

    lr = (1e-5,1e-3)
    lr_scheduler = LearningRateScheduler(
        lambda epoch: lr[0] * 10**(np.log10(lr[1]/lr[0])*(epoch-1)/(args.epochs-1))
    )

    early_stopping = EarlyStopping(
        monitor = 'val_loss',
        patience = 10,
        mode = 'min',
        restore_best_weights = True
    )

    # save the model with the maximum validation accuracy 
    checkpoint = ModelCheckpoint(
        os.path.join(args.sm_model_dir,'best_model'),
        monitor = 'val_accuracy',
        verbose = 1,
        mode = 'max', 
        save_best_only = True
    )
    
    tensorboard_callback = TensorBoard(log_dir = tensorboard_dir)
    
    callbacks = [early_stopping,checkpoint,tensorboard_callback] #lr_scheduler,

    td = next(train_data)
    vd = next(validation_data)
    #for i in range(len_train // args.batch_size):
    history = model.fit(
        x = td[0].numpy(),
        y = td[1].numpy(),
        epochs = 200,
        validation_data = (vd[0].numpy(),vd[1].numpy())
        #callbacks = callbacks
    )
    #    if i == 0:
    h = pd.DataFrame.from_dict(history.history)
    #    else:
    #        h = h.append(pd.DataFrame.from_dict(history.history))
    h = h.reset_index(drop=True)
    h.columns = [re.sub('_[0-9]{1}.*','',c) for c in h.columns]
    h.to_csv(args.sm_model_dir+'/history.csv',index=False)

    logging.info('Best Model written to: ' + args.sm_model_dir + '/best_model')
    logging.info('Model History written to: ' + args.sm_model_dir + '/history.csv')
    
    ed = next(eval_data)
    score = model.evaluate(
        ed[0].numpy(),
        ed[1].numpy(),
        steps = len_eval // 50,
        verbose = 0
    )

    logging.info('test loss:{}'.format(score[0]))
    logging.info('test accuracy:{}'.format(score[1]))

    model.save(os.path.join(args.sm_model_dir,'00001'))

def num_examples_per_epoch(subset='train'):
    if subset == 'train':
        return 9000
    elif subset == 'validation':
        return 1000
    elif subset == 'eval':
        return 1000
    else:
        raise ValueError('Invalid data subset "%s"' % subset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, required = False, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--validation', type=str, required = False, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    parser.add_argument('--evaluation', type=str, required = False, default=os.environ.get('SM_CHANNEL_EVAL'))
    parser.add_argument('--model_dir', type=str)
    # required = True, help='The directory where the model will be stored.'
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--output-dir', type=str, default=os.environ.get('SM_OUTPUT_DIR'))
    parser.add_argument('--tensorboard-dir', type=str, default=os.environ.get('SM_MODULE_DIR'))
    parser.add_argument('--weight-decay', type=float, default=2e-4, help='Weight decay for convolutions.')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--train-batch-size', type=int, default=128)
    parser.add_argument('--validation-batch-size', type=int, default=128)
    parser.add_argument('--data-config', type=json.loads, default=os.environ.get('SM_INPUT_DATA_CONFIG'))
    parser.add_argument('--fw-params', type=json.loads, default=os.environ.get('SM_FRAMEWORK_PARAMS'))
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--momentum', type=float, default='0.9')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--run-validation', type=int, default=1)
    args = parser.parse_args()
    main(args)