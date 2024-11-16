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
from argparse import ArgumentParser

# Import additional libraries for evaluation metrics and visualization
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter

unk = '<UNK>'

class FFNN(nn.Module):
    def __init__(self, input_dim, h):
        super(FFNN, self).__init__()
        self.h = h
        self.W1 = nn.Linear(input_dim, h)
        self.activation = nn.ReLU()
        self.output_dim = 5
        self.W2 = nn.Linear(h, self.output_dim)

        self.softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, input_vector):
        # Obtain first hidden layer representation
        h = self.activation(self.W1(input_vector))
        
        # Obtain output layer representation
        z = self.W2(h)
        
        # Ensure input to softmax has shape [batch_size, num_classes]
        predicted_vector = self.softmax(z.view(1, -1))
        
        return predicted_vector

def make_vocab(data):
    vocab = set()
    for document, _ in data:
        for word in document:
            vocab.add(word)
    return vocab 

def make_indices(vocab):
    vocab_list = sorted(vocab)
    vocab_list.append(unk)
    word2index = {}
    index2word = {}
    for index, word in enumerate(vocab_list):
        word2index[word] = index 
        index2word[index] = word 
    vocab.add(unk)
    return vocab, word2index, index2word 

def convert_to_vector_representation(data, word2index):
    vectorized_data = []
    for document, y in data:
        vector = torch.zeros(len(word2index)) 
        for word in document:
            index = word2index.get(word, word2index[unk])
            vector[index] += 1
        vectorized_data.append((vector, y))
    return vectorized_data

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

    # Fix random seeds
    random.seed(42)
    torch.manual_seed(42)

    # Load data
    print("========== Loading data ==========")
    train_data, valid_data = load_data(args.train_data, args.val_data)
    vocab = make_vocab(train_data)
    vocab, word2index, index2word = make_indices(vocab)

    print("========== Vectorizing data ==========")
    train_data = convert_to_vector_representation(train_data, word2index)
    valid_data = convert_to_vector_representation(valid_data, word2index)
    
    model = FFNN(input_dim=len(vocab), h=args.hidden_dim)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # Initialize TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir='runs/ffnn_experiment')

    print("========== Training for {} epochs ==========".format(args.epochs))
    
    best_validation_accuracy = 0
    last_train_accuracy = 0
    last_validation_accuracy = 0
    
    # Lists to store metrics
    train_losses = []
    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        correct = 0
        total = 0
        start_time = time.time()
        print("Training started for epoch {}".format(epoch + 1))
        random.shuffle(train_data)
        minibatch_size = 16
        N = len(train_data)
        
        total_loss = 0
        total_batches = 0

        for minibatch_index in tqdm(range((N + minibatch_size - 1) // minibatch_size)):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                data_index = minibatch_index * minibatch_size + example_index
                if data_index >= N:
                    break
                input_vector, gold_label = train_data[data_index]
                predicted_vector = model(input_vector)
                predicted_label = torch.argmax(predicted_vector)
                correct += int(predicted_label == gold_label)
                total += 1
                example_loss = model.compute_Loss(predicted_vector, torch.tensor([gold_label]))
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss
            if loss is not None:
                loss = loss / minibatch_size
                total_loss += loss.item()
                total_batches += 1
                loss.backward()
                optimizer.step()
        avg_train_loss = total_loss / total_batches if total_batches > 0 else 0.0
        train_accuracy = correct / total if total > 0 else 0.0
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)
        
        print("Training completed for epoch {}".format(epoch + 1))
        print("Training loss for epoch {}: {:.4f}".format(epoch + 1, avg_train_loss))
        print("Training accuracy for epoch {}: {:.4f}".format(epoch + 1, train_accuracy))
        print("Training time for this epoch: {:.2f} seconds".format(time.time() - start_time))
        
        # Log training metrics to TensorBoard
        writer.add_scalar('Loss/Train', avg_train_loss, epoch + 1)
        writer.add_scalar('Accuracy/Train', train_accuracy, epoch + 1)
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        start_time = time.time()
        print("Validation started for epoch {}".format(epoch + 1))
        minibatch_size = 16
        N = len(valid_data)
        true_labels = []
        predicted_labels = []

        for minibatch_index in tqdm(range((N + minibatch_size - 1) // minibatch_size)):
            loss = None
            for example_index in range(minibatch_size):
                data_index = minibatch_index * minibatch_size + example_index
                if data_index >= N:
                    break
                input_vector, gold_label = valid_data[data_index]
                predicted_vector = model(input_vector)
                predicted_label = torch.argmax(predicted_vector).item()
                correct += int(predicted_label == gold_label)
                total += 1
                true_labels.append(gold_label)
                predicted_labels.append(predicted_label)
                example_loss = model.compute_Loss(predicted_vector, torch.tensor([gold_label]))
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss
            if loss is not None:
                loss = loss / minibatch_size
        validation_accuracy = correct / total if total > 0 else 0.0
        val_accuracies.append(validation_accuracy)
        
        print("Validation completed for epoch {}".format(epoch + 1))
        print("Validation accuracy for epoch {}: {:.4f}".format(epoch + 1, validation_accuracy))
        print("Validation time for this epoch: {:.2f} seconds".format(time.time() - start_time))
        
        # Log validation accuracy to TensorBoard
        writer.add_scalar('Accuracy/Validation', validation_accuracy, epoch + 1)
        
        # Generate classification report and confusion matrix
        unique_labels = sorted(set(true_labels))
        labels = [0, 1, 2, 3, 4]
        target_names = ['1-star', '2-star', '3-star', '4-star', '5-star']
        
        report = classification_report(true_labels, predicted_labels, labels=labels, target_names=target_names, output_dict=True, zero_division=0)
        conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=labels)
        
        # Log per-class metrics to TensorBoard
        for metric in ['precision', 'recall', 'f1-score']:
            class_metrics = {f'{target_names[i]}': report[target_names[i]][metric] for i in range(len(target_names))}
            writer.add_scalars(metric.capitalize(), class_metrics, epoch + 1)
        
        # Plot and log confusion matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=target_names, yticklabels=target_names)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title(f'Confusion Matrix for Epoch {epoch + 1}')

        # Add the plot to TensorBoard
        writer.add_figure('Confusion Matrix', fig, global_step=epoch + 1)
        plt.close(fig)
        
        # Early stopping or model saving
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            # Save the best model
            torch.save(model.state_dict(), 'best_ffnn_model.pth')
        else:
            if validation_accuracy < last_validation_accuracy and train_accuracy > last_train_accuracy:
                print("Training stopped to avoid overfitting!")
                print("Best validation accuracy is: {:.4f}".format(best_validation_accuracy))
                break  # Early stopping

        last_validation_accuracy = validation_accuracy
        last_train_accuracy = train_accuracy

    writer.close()
    
    # Plot training and validation accuracy
    epochs_range = range(1, len(train_accuracies) + 1)
    plt.figure()
    plt.plot(epochs_range, train_accuracies, 'bo-', label='Training accuracy')
    plt.plot(epochs_range, val_accuracies, 'ro-', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
