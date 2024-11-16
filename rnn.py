import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os
import time
from tqdm import tqdm
import json
import string
from argparse import ArgumentParser
import pickle
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Import TensorBoard SummaryWriter
from torch.utils.tensorboard import SummaryWriter

unk = '<UNK>'

class RNN(nn.Module):
    def __init__(self, input_dim, h):
        super(RNN, self).__init__()
        self.h = h
        self.numOfLayer = 1
        self.rnn = nn.RNN(input_dim, h, self.numOfLayer, nonlinearity='tanh')
        self.W = nn.Linear(h, 5)
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, inputs):
        # Obtain hidden layer representation
        output, hidden = self.rnn(inputs)
        
        # Obtain output layer representations
        z = self.W(output)
        
        # Sum over outputs
        sum_z = torch.sum(z, dim=0)
        
        # Obtain probability distribution
        predicted_vector = self.softmax(sum_z)
        
        return predicted_vector

def load_data(train_data, val_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)

    tra = []
    val = []
    for elt in training:
        tra.append((elt["text"].split(), int(elt["stars"] - 1)))
    for elt in validation:
        val.append((elt["text"].split(), int(elt["stars"] - 1)))
    return tra, val

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required=True, help="hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required=True, help="num of epochs to train")
    parser.add_argument("--train_data", required=True, help="path to training data")
    parser.add_argument("--val_data", required=True, help="path to validation data")
    parser.add_argument("--test_data", default=None, help="path to test data")
    parser.add_argument('--do_train', action='store_true')
    args = parser.parse_args()

    print("========== Loading data ==========")
    train_data, valid_data = load_data(args.train_data, args.val_data)

    print("========== Vectorizing data ==========")
    model = RNN(50, args.hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    word_embedding = pickle.load(open('Data_Embedding/word_embedding.pkl', 'rb'))

    # Initialize TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir='runs/rnn_experiment')

    stopping_condition = False
    epoch = 0

    last_train_accuracy = 0
    last_validation_accuracy = 0
    best_validation_accuracy = 0

    # Lists to store metrics for plotting
    train_losses = []
    train_accuracies = []
    val_accuracies = []

    while not stopping_condition and epoch < args.epochs:
        random.shuffle(train_data)
        model.train()
        print("Training started for epoch {}".format(epoch + 1))
        correct = 0
        total = 0
        minibatch_size = 16
        N = len(train_data)

        loss_total = 0
        loss_count = 0

        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                data_index = minibatch_index * minibatch_size + example_index
                if data_index >= N:
                    break

                input_words, gold_label = train_data[data_index]
                input_words = " ".join(input_words)

                # Remove punctuation
                input_words = input_words.translate(str.maketrans("", "", string.punctuation)).split()

                # Look up word embedding dictionary
                vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i in input_words]

                # Transform the input into required shape
                vectors = torch.tensor(vectors).view(len(vectors), 1, -1)
                output = model(vectors)

                # Get loss
                example_loss = model.compute_Loss(output.view(1, -1), torch.tensor([gold_label]))

                # Get predicted label
                predicted_label = torch.argmax(output)

                correct += int(predicted_label == gold_label)
                total += 1

                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss

            loss = loss / minibatch_size
            loss_total += loss.item()
            loss_count += 1
            loss.backward()
            optimizer.step()

        avg_train_loss = loss_total / loss_count
        train_accuracy = correct / total
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)

        print("Training completed for epoch {}".format(epoch + 1))
        print("Training loss for epoch {}: {:.4f}".format(epoch + 1, avg_train_loss))
        print("Training accuracy for epoch {}: {:.4f}".format(epoch + 1, train_accuracy))

        # Log training metrics to TensorBoard
        writer.add_scalar('Loss/Train', avg_train_loss, epoch + 1)
        writer.add_scalar('Accuracy/Train', train_accuracy, epoch + 1)

        # Validation
        model.eval()
        correct = 0
        total = 0
        true_labels = []
        predicted_labels = []
        print("Validation started for epoch {}".format(epoch + 1))

        for input_words, gold_label in tqdm(valid_data):
            input_words = " ".join(input_words)
            input_words = input_words.translate(str.maketrans("", "", string.punctuation)).split()
            vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i in input_words]

            vectors = torch.tensor(vectors).view(len(vectors), 1, -1)
            output = model(vectors)
            predicted_label = torch.argmax(output).item()
            correct += int(predicted_label == gold_label)
            total += 1

            true_labels.append(gold_label)
            predicted_labels.append(predicted_label)

        validation_accuracy = correct / total
        val_accuracies.append(validation_accuracy)

        print("Validation completed for epoch {}".format(epoch + 1))
        print("Validation accuracy for epoch {}: {:.4f}".format(epoch + 1, validation_accuracy))

        # Log validation accuracy to TensorBoard
        writer.add_scalar('Accuracy/Validation', validation_accuracy, epoch + 1)

         # Generate classification report & Confusion matrix
        labels = [0, 1, 2, 3, 4]
        target_names = ['1-star', '2-star', '3-star', '4-star', '5-star']
        report = classification_report(true_labels, predicted_labels, labels=labels, target_names=target_names, output_dict=True, zero_division=0)
        conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=labels)


        # Log metrics to TensorBoard
        writer.add_scalars('Precision', {f'{k}': report[k]['precision'] for k in target_names}, epoch + 1)
        writer.add_scalars('Recall', {f'{k}': report[k]['recall'] for k in target_names}, epoch + 1)
        writer.add_scalars('F1-Score', {f'{k}': report[k]['f1-score'] for k in target_names}, epoch + 1)

        # Plot and log confusion matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=target_names, yticklabels=target_names)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title(f'Confusion Matrix for Epoch {epoch + 1}')

        # Add the plot to TensorBoard
        writer.add_figure('Confusion Matrix', fig, global_step=epoch + 1)
        plt.close(fig)  # Close the figure to prevent display

        # Early stopping condition
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            # Save the best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            if validation_accuracy < last_validation_accuracy and train_accuracy > last_train_accuracy:
                stopping_condition = True
                print("Training stopped to avoid overfitting!")
                print("Best validation accuracy is: {:.4f}".format(best_validation_accuracy))

        last_validation_accuracy = validation_accuracy
        last_train_accuracy = train_accuracy

        epoch += 1

    writer.close()

    # Plot training and validation accuracy
    epochs = range(1, epoch + 1)
    plt.figure()
    plt.plot(epochs, train_accuracies, 'bo-', label='Training accuracy')
    plt.plot(epochs, val_accuracies, 'ro-', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
