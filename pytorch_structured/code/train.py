import argparse
import codecs
import json
import logging
import os
import re

import pandas as pd
import numpy as np
import torch
#from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
#from tensorflow.keras.models import save_model

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
    labels = data_rep.iloc[:,-1]
    labels = np.array([[l] for l in labels]) #.to_numpy()
    #dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    #dataset = dataset.batch(batch_size)
    return (torch.from_numpy(features).float(),torch.from_numpy(labels).float()),len(data_rep)

def main(args):  
    logging.info('getting data')
    train_data, len_train = process_input(args.epochs, args.batch_size, args.train, 'train')
    eval_data, len_eval = process_input(args.epochs, 1000, args.evaluation, 'evaluation')
    validation_data, len_validation = process_input(args.epochs, args.batch_size, args.validation, 'validation')
    
    logging.info('configuring model')

    input_dimension = train_data[0].shape[1]
    print(input_dimension)
    
    class BinaryClassification(torch.nn.Module):
        def __init__(self, input_dimension):
            super().__init__()
            self.linear = torch.nn.Linear(input_dimension, 1)

        def forward(self, input_dimension):
            return self.linear(input_dimension)

    model = torch.nn.Linear(input_dimension, 1)

    def full_gd(model, loss, optimizer, train_data, validation_data, n_epochs=6000):
        train_losses = np.zeros(n_epochs)
        test_losses = np.zeros(n_epochs)
        train_accs = np.zeros(n_epochs)
        test_accs = np.zeros(n_epochs)

        for it in range(n_epochs): 
            outputs = model(train_data[0])
            loss_train = loss(outputs, train_data[1])
            
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            outputs_test = model(validation_data[0])
            loss_test = loss(outputs_test, validation_data[1])

            train_acc = np.mean(train_data[1].numpy() == (outputs.detach().numpy() > 0)) #.numpy()
            test_acc = np.mean(validation_data[1].numpy() == (outputs_test.detach().numpy() > 0))
            
            train_losses[it] = loss_test.item()
            test_losses[it] = loss_test.item()
            train_accs[it] = train_acc
            test_accs[it] = test_acc

            if (it + 1) % 50 == 0:
                print(f'In this epoch {it+1}/{n_epochs}, training loss: {loss_train.item():.4f}, validation loss: {loss_test.item():.4f}, training accuracy: {train_acc:.3f}, validation accuracy: {test_acc:.4f}')

        train_losses = np.expand_dims(train_losses,axis=0)
        test_losses = np.expand_dims(test_losses,axis=0)
        train_accs = np.expand_dims(train_accs,axis=0)
        test_accs = np.expand_dims(test_accs,axis=0)
        hist = pd.DataFrame(np.concatenate(
            (train_losses, test_losses, train_accs, test_accs),axis=0).transpose(),columns=['train_loss','test_loss','train_acc','test_acc']
        )
 
        return hist

    loss = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters())
    h = full_gd(model, loss, optimizer, train_data, validation_data)

    h.to_csv(args.sm_model_dir+'/history.csv',index = False)
    logging.info('Model History written to: ' + args.sm_model_dir + '/history.csv')

    save_path = os.path.join(args.sm_model_dir,'model.pth')
    torch.save(model.state_dict(), save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, required = False, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--validation', type=str, required = False, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    parser.add_argument('--evaluation', type=str, required = False, default=os.environ.get('SM_CHANNEL_EVAL'))
    parser.add_argument('--model_dir', type=str) # required = True, help='The directory where the model will be stored.'
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