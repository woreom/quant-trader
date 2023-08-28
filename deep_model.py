import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
import optuna

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Module, ModuleList
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import joblib

class FeedForward(Module):
    def __init__(self,
                 d_model: int,
                 d_hidden: int = 512):
        super(FeedForward, self).__init__()

        self.linear_1 = torch.nn.Linear(d_model, d_hidden)
        self.linear_2 = torch.nn.Linear(d_hidden, d_model)

    def forward(self, x):

        x = self.linear_1(x)
        x = F.relu(x)
        x = self.linear_2(x)

        return x


class MultiHeadAttention(Module):
    def __init__(self,
                 d_model: int,
                 q: int,
                 v: int,
                 h: int,
                 device: str,
                 mask: bool=False,
                 dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()

        self.W_q = torch.nn.Linear(d_model, q * h)
        self.W_k = torch.nn.Linear(d_model, q * h)
        self.W_v = torch.nn.Linear(d_model, v * h)

        self.W_o = torch.nn.Linear(v * h, d_model)

        self.device = device
        self._h = h
        self._q = q

        self.mask = mask
        self.dropout = torch.nn.Dropout(p=dropout)
        self.score = None

    def forward(self, x, stage):
        Q = torch.cat(self.W_q(x).chunk(self._h, dim=-1), dim=0)
        K = torch.cat(self.W_k(x).chunk(self._h, dim=-1), dim=0)
        V = torch.cat(self.W_v(x).chunk(self._h, dim=-1), dim=0)

        score = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self._q)
        self.score = score

        if self.mask and stage == 'train':
            mask = torch.ones_like(score[0])
            mask = torch.tril(mask, diagonal=0)
            score = torch.where(mask > 0, score, torch.Tensor([-2**32+1]).expand_as(score[0]).to(self.device))

        score = F.softmax(score, dim=-1)

        attention = torch.matmul(score, V)

        attention_heads = torch.cat(attention.chunk(self._h, dim=0), dim=-1)

        self_attention = self.W_o(attention_heads)

        return self_attention, self.score


class Encoder(Module):
    def __init__(self,
                 d_model: int,
                 d_hidden: int,
                 q: int,
                 v: int,
                 h: int,
                 device: str,
                 mask: bool = False,
                 dropout: float = 0.1):
        super(Encoder, self).__init__()

        self.MHA = MultiHeadAttention(d_model=d_model, q=q, v=v, h=h, mask=mask, device=device, dropout=dropout)
        self.feedforward = FeedForward(d_model=d_model, d_hidden=d_hidden)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.layerNormal_1 = torch.nn.LayerNorm(d_model)
        self.layerNormal_2 = torch.nn.LayerNorm(d_model)

    def forward(self, x, stage):

        residual = x
        x, score = self.MHA(x, stage)
        x = self.dropout(x)
        x = self.layerNormal_1(x + residual)

        residual = x
        x = self.feedforward(x)
        x = self.dropout(x)
        x = self.layerNormal_2(x + residual)

        return x, score



class Transformer(Module):
    def __init__(self,
                 d_model: int,
                 d_input: int,
                 d_channel: int,
                 d_output: int,
                 d_hidden: int,
                 q: int,
                 v: int,
                 h: int,
                 N: int,
                 d_encode: int,
                 device: str,
                 dropout: float = 0.1,
                 pe: bool = False,
                 mask: bool = False):
        """
        A transformer model for processing time series data with both channel and input attention.

        Args:
            d_model (int): The dimension of the model.
            d_input (int): The dimension of the input time series.
            d_channel (int): The dimension of the channel data.
            d_output (int): The dimension of the output.
            d_hidden (int): The number of units in the hidden layer.
            q (int): The dimension of the query vectors in the attention mechanism.
            v (int): The dimension of the value vectors in the attention mechanism.
            h (int): The number of attention heads.
            N (int): The number of stacked encoder layers.
            d_encode (int): The dimension of the last encoding layer.
            device (str): The device to use for computations (e.g. "cpu" or "cuda").
            dropout (float, optional): The dropout probability to use. Defaults to 0.1.
            pe (bool, optional): Whether to add positional encodings to the input embeddings. Defaults to False.
            mask (bool, optional): Whether to use a causal mask in the encoder layers. Defaults to False.
        """
        super(Transformer, self).__init__()

        # Initialize two lists of encoder layers, one for the channel data and one for the input time series.
        self.encoder_list_1 = ModuleList([Encoder(d_model=d_model,
                                                  d_hidden=d_hidden,
                                                  q=q,
                                                  v=v,
                                                  h=h,
                                                  mask=mask,
                                                  dropout=dropout,
                                                  device=device) for _ in range(N)])

        self.encoder_list_2 = ModuleList([Encoder(d_model=d_model,
                                                  d_hidden=d_hidden,
                                                  q=q,
                                                  v=v,
                                                  h=h,
                                                  dropout=dropout,
                                                  device=device) for _ in range(N)])

        # Initialize linear layers for embedding the channel data and input time series data.
        self.embedding_channel = torch.nn.Linear(d_channel, d_model)
        self.embedding_input = torch.nn.Linear(d_input, d_model)

        # Initialize a linear layer for the gating mechanism and a linear layer for the output.
        self.gate = torch.nn.Linear(d_model * d_input + d_model * d_channel, 2)
        self.temp_linear= torch.nn.Linear(d_model * d_input + d_model * d_channel, d_encode)
        self.output_linear = torch.nn.Linear(d_encode, d_output)

        self.pe = pe
        self._d_input = d_input
        self._d_model = d_model
  


    def forward(self, x, stage):
        """
        Apply the transformer to the input data.

        Args:
            x (torch.Tensor): The input data of shape (batch_size, sequence_length, channel_size, input_size).
            stage (int): The current training stage.

        Returns:
            torch.Tensor: The output of the transformer of shape (batch_size, d_output).
            torch.Tensor: The encoded input data of shape (batch_size, sequence_length, d_model).
            torch.Tensor: The scores for the input attention of shape (batch_size, h, sequence_length, sequence_length).
            torch.Tensor: The scores for the channel attention of shape (batch_size, h, channel_size, channel_size).
            torch.Tensor: The input data after being transformed by the channel embedding layer of shape (batch_size, sequence_length, d_model).
            torch.Tensor: The channel data after being transformed by the input embedding layer of shape (batch_size, channel_size, d_model).
        """
        # Embed the channel data.
        encoding_1 = self.embedding_channel(x)
        input_to_gather = encoding_1

        # Add positional encodings if specified.
        if self.pe:
            pe = torch.ones_like(encoding_1[0])
            position = torch.arange(0, self._d_input).unsqueeze(-1)
            temp = torch.Tensor(range(0, self._d_model, 2))
            temp = temp * -(math.log(10000) / self._d_model)
            temp = torch.exp(temp).unsqueeze(0)
            temp = torch.matmul(position.float(), temp)  # shape:[input, d_model/2]
            pe[:, 0::2] = torch.sin(temp)
            pe[:, 1::2] = torch.cos(temp)

            encoding_1 = encoding_1 + pe

        # Apply the channel attention layers.
        for encoder in self.encoder_list_1:
            encoding_1, score_input = encoder(encoding_1, stage)

        # Embed the input time series data.
        encoding_2 = self.embedding_input(x.transpose(-1, -2))
        channel_to_gather = encoding_2

        # Apply the input time series attention layers.
        for encoder in self.encoder_list_2:
            encoding_2, score_channel = encoder(encoding_2, stage)

        # Reshape the encoded data for gating.
        encoding_1 = encoding_1.reshape(encoding_1.shape[0], -1)
        encoding_2 = encoding_2.reshape(encoding_2.shape[0], -1)

        # Apply the gating mechanism.
        gate = F.softmax(self.gate(torch.cat([encoding_1, encoding_2], dim=-1)), dim=-1)
        encoding = torch.cat([encoding_1 * gate[:, 0:1], encoding_2 * gate[:, 1:2]], dim=-1)

        # Apply the output layer.
        temp_output = self.temp_linear(encoding)
        output = self.output_linear(temp_output)
        return output, temp_output, score_input, score_channel, input_to_gather, channel_to_gather, gate

def checkpoint(model, filename):
    torch.save(model.state_dict(), filename)
    
def resume(model, filename):
    model.load_state_dict(torch.load(filename))


def train(Inputs, Targets, param, save_name, use_pre_train=True,plotResults=True):
    """
    Trains a transformer network using the preprocessed input features and target labels.

    Parameters:
    -----------
    Inputs : array-like, shape (n_samples, n_features)
        The preprocessed input features, with lags and VMD decomposition applied.
    Targets : array-like, shape (n_samples,)
        The target labels corresponding to the preprocessed input features.
    param : dict
        The dictionary of hyperparameters for the transformer network.
    save_name: str
        path name to save trained model or load that
    use_pre_train : bool, optional (default=True)
        Whether to use pre trained model for forecast or not
    plotResults : bool, optional (default=True)
        Whether to plot the training and validation losses and accuracies, and confusion matrices for the train, validation, and test sets.

    Returns:
    --------
    net : Transformer object
        The trained transformer network.
    Acc: dict
       The Train, Validation, Test Acc of Model
    encoding: array
       The Encodinge features

    """

    # Define the hyperparameters
    num_epochs =100
    batch_size =32
    
    learning_rate = param['learning_rate']
    dropout = param['dropout']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device :', device)

    # Define the network architecture and other settings
    d_model = param['d_model']
    d_hidden = param['d_hidden']
    d_encode = param['d_encode']
    q = param['q']
    v = param['v']
    h = param['h']
    N = param['N']

    d_input = np.shape(Inputs)[1]
    d_channel = np.shape(Inputs)[2]
    d_output = len(np.unique(Targets))

    # Define the transformer network
    net = Transformer(d_model=d_model, d_input=d_input, d_channel=d_channel, d_output=d_output, d_hidden=d_hidden,
                  q=q, v=v, h=h, N=N, d_encode= d_encode, dropout=dropout, pe=True, mask=True, device=device).to(device)

    # Define the optimizer and loss function
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Split the data into training, validation, and test sets
    train_ratio = 0.8
    validation_ratio = 0.1
    test_ratio =0.1

    x_train, x_test, y_train, y_test = train_test_split(Inputs, Targets, test_size=1 - train_ratio, random_state=42, shuffle=False)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio), random_state=42, shuffle=False)

    # Convert the data into PyTorch tensors and create DataLoader objects for each set
    train_data = torch.tensor(x_train, dtype=torch.float32)
    train_labels = torch.tensor(y_train, dtype=torch.float32)

    test_data = torch.tensor(x_test, dtype=torch.float32)
    test_labels = torch.tensor(y_test, dtype=torch.float32)

    val_data = torch.tensor(x_val, dtype=torch.float32)
    val_labels = torch.tensor(y_val, dtype=torch.float32)

    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = torch.utils.data.TensorDataset(test_data, test_labels)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    val_dataset = torch.utils.data.TensorDataset(val_data, val_labels)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()

    # Initialize lists to store the training and validation losses and accuracies
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    # Set early stopping threshold and initialize best accuracy and epoch variables
    early_stop_thresh = 15
    best_accuracy = -1
    best_epoch = -1
    
    
    if use_pre_train==False:
        # Train the network for the specified number ofepochs
        for epoch in range(num_epochs):

            net.to(device)
            net.train()
            t_loss_list, v_loss_list = [], []
            for batch_idx, data in enumerate(train_loader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                # Forward pass through the network and compute the training loss
                outputs, _, _, _, _, _, _  = net(inputs,'train')
                train_loss = criterion(outputs, labels.long().to(device))
                train_loss.backward()
                optimizer.step()
                t_loss_list.append(train_loss.item())

            # Evaluate the performance on the validation set
            net.eval()
            for batch_idx, (inputs, labels) in enumerate(val_loader, 0):
                with torch.no_grad():
                    inputs, labels = inputs.to(device), labels.to(device)
                    prediction, _, _, _, _, _, _ = net(inputs, 'test')
                    val_loss=criterion(prediction, labels.long().to(device))
                    v_loss_list.append(val_loss.item())

            # Append the training and validation losses and accuracies to their respective lists
            train_losses.append(np.mean(t_loss_list))
            val_losses.append(np.mean(v_loss_list))

            train_acc,_,_,_ = predict(net, train_loader, device)
            val_acc,_ ,_,_ = predict(net, val_loader, device)

            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            print(
            f"Epoch {epoch+1:02d}/{num_epochs:02d}"
            f" | Train accuracy: {train_acc:.2f}"
            f" | Val accuracy: {val_acc:.2f}"
                )
        
            # Save the model if the validation accuracy improves
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                best_epoch = epoch
                checkpoint(net, "params/transformer_checkpoint.pth")
            # If the validation accuracy has not improved for a certain number of epochs, stop training early
            elif epoch - best_epoch > early_stop_thresh:
                print("Early stopped training at epoch %d" % epoch)
                break  # terminate the training loop
         
        # Load the best model
        resume(net, "params/transformer_checkpoint.pth")
        torch.save(net, f'params/transformer_trained_for_{save_name}.pth')
    else:
        net = torch.load(f'params/transformer_trained_for_{save_name}.pth') 

    
    # Compute the accuracy and predicted outputs on the training, validation, and test sets
    TrainAcc, TrainOutputs, _ , train_encoding= predict(net, train_loader, device)
    ValAcc, ValOutputs, _ , val_encoding= predict(net, val_loader, device)
    TestAcc, TestOutputs, _ , test_encoding= predict(net, test_loader, device)

    # Print the model's performance on each set
    print('------Model Performance------')
    print('Train Acc:', TrainAcc, '| Validation Acc:', ValAcc, '| Test Acc:', TestAcc)
    
    Acc={'Train': TrainAcc , 'Validation': ValAcc , 'Test': TestAcc}
    
    try:
        encoding=np.concatenate((train_encoding,val_encoding,test_encoding))
    except ValueError:
        a = np.concatenate(train_encoding)
        b = np.concatenate(val_encoding)
        c = np.concatenate(test_encoding)
        
        encoding=np.concatenate((a,b,c))
    
    
    # If plotResults is True, plot the training and validation losses and accuracies, andconfusion matrices for the train, validation, and test sets

    if plotResults:
        plot_results(train_losses, val_losses, train_accs, val_accs)
        plot_confusion_matrix(y_train, TrainOutputs, list(np.unique(Targets)), normalize=True,title='Train')
        plot_confusion_matrix(y_val, ValOutputs, list(np.unique(Targets)), normalize=True,title='Validation')
        plot_confusion_matrix(y_test, TestOutputs, list(np.unique(Targets)), normalize=True,title='Test')

    return net, Acc, encoding


def forecast(net, Forecast_Inputs, device='cpu'):
    """
    Generates a forecast using the trained transformer network.

    Parameters:
    -----------
    net : Transformer object
        The trained transformer network.
    Forecast_Inputs : array-like, shape (n_samples, n_lags, n_decompose)
        The input features for which to generate a forecast.
    device : str, optional (default='cpu')
        The device on which to perform the forecast.

    Returns:
    --------
    forecast : int
        The predicted label for the input features.
    prob: float
        The predicted probibilty for prediction.
    encoding: array-like , shape(n_samples, n_features)
        The encoding features     

    """

    # Convert the test data to a PyTorch tensor and move it to the device
    Forecast_Inputs = torch.tensor(Forecast_Inputs, dtype=torch.float32).to(device)

    # Move the model to the same device as the test data
    net.to(device)

    # Make predictions using the trained network
    net.eval()
    with torch.no_grad():
        predicted, encoding, _, _, _, _, _ = net(Forecast_Inputs,'test')
        prob, predicted = torch.max(predicted, 1)

    # Convert the predicted label to its original class label and return it
    forecast = predicted.cpu().detach().numpy()[-1]
        
    prob=prob.cpu().detach().numpy()[-1]
    encoding=encoding.cpu().detach().numpy()[-1,:]
 

    
    return forecast, prob, encoding

    
def predict(net, data_loader, device):
    """
    Predict Labels and Computes the accuracy of the trained transformer network on a given data loader.

    Parameters:
    -----------
    net : Transformer object
        The trained transformer network.
    data_loader : torch.utils.data.DataLoader
        The data loader containing the test data.
    device : str
        The device on which to perform the computation.

    Returns:
    --------
    accuracy : float
        The accuracy of the network on the test data, as a percentage.
    label_indices : numpy array
        The predicted label indices for all inputs, as a numpy array.
    prob: float
        The predicted probibilty for prediction.
        
    encoding: array-like , shape(n_samples, n_features)
        The encoding features 
    
    """

    correct = 0
    total = 0
    label_indices = []
    
    probs=[]
    encoding=[]
    # Evaluate the network on the test data
    with torch.no_grad():
        net.eval()
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            prediction, encod, _, _, _, _, _ = net(inputs, 'test')
            prob, batch_label_indices = torch.max(prediction.data, dim=-1)
            probs.append(prob.cpu().detach().numpy())
            encoding.append(encod.cpu().detach().numpy())
            label_indices.append(batch_label_indices.cpu().numpy())
            total += batch_label_indices.shape[0]
            correct += (batch_label_indices == labels.long()).sum().item()

    # Compute the accuracy of the network on the test data
    accuracy = 100 * correct / total

    # Concatenate the predicted label indices for all batches into a single numpy array
    label_indices = np.concatenate(label_indices)

    # Return the accuracy and the predicted label indices for all inputs
    return accuracy, label_indices, probs, encoding


def plot_results(train_loss, val_loss, train_acc, val_acc):
    
    """
    This function takes lists of values and creates side-by-side graphs to show training and validation performance
    """
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(
        train_loss, label="train", color="red", linestyle="--", linewidth=2, alpha=0.5
    )
    ax[0].plot(
        val_loss, label="val", color="blue", linestyle="--", linewidth=2, alpha=0.5
    )
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].legend()
    ax[1].plot(
        train_acc, label="train", color="red", linestyle="--", linewidth=2, alpha=0.5
    )
    ax[1].plot(
        val_acc, label="val", color="blue", linestyle="--", linewidth=2, alpha=0.5
    )
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")
    ax[1].legend()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure()
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()



def hyper_tune_model(inputs, labels, n_trials=100):
    """
    Hyperparameter tuning function using Optuna.

    Parameters:
    -----------
    inputs : array-like, shape (n_samples, n_features)
        The input features.
    labels : array-like, shape (n_samples,)
        The target labels.
    n_trials : int, optional (default=100)
        The number of trials to run for hyperparameter optimization.

    Returns:
    --------
    best_param : dict
        The best set of hyperparameters found by Optuna.
    """

    def objective(trial):
        """
        Define the objective function for Optuna.
        """
        # Define the hyperparameters to optimize
        param = {
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-3),
            'dropout': trial.suggest_uniform('dropout', 0, 1),
            'd_model': trial.suggest_int('d_model', 16, 256, log=True),
            'd_hidden': trial.suggest_int('d_hidden', 16, 256, log=True),
            'd_encode':  512,
            'q': trial.suggest_int('q', 1, 8),
            'v': trial.suggest_int('v', 1, 8),
            'h': trial.suggest_int('h', 1, 8),
            'N': trial.suggest_int('N', 1, 8)
        }
        
        try:
            net, acc,_ = train(inputs, labels, param, plotResults=False)
            cost = acc['Test']  # Use test accuracy as the objective value
        except RuntimeError:
            # If the model fails to converge, return a cost of infinity
            cost = 0
        
        return cost

        
    
    # Define the study to optimize the objective function
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    # Get the best set of hyperparameters found by Optuna
    best_param = study.best_params
    print('Best set of hyperparameters:', best_param)
    
    return best_param
    
def transformer_model(inputs, labels, inputs_forecast, save_name, use_pre_train=True, hyper_tune=False, plotResults=True, n_trials=100):
    """
    Trains a transformer model on the preprocessed inputs and labels for time series forecasting.

    Parameters:
    -----------
    inputs : numpy.ndarray
        The lagged input features as a 2D array of shape (num_samples - nlags, num_features * Numimf * nlags).
    labels : numpy.ndarray
        The labels as a 1D array of shape (num_samples - nlags,).
    inputs_forecast : numpy.ndarray
        The lagged input features for the forecast period as a 2D array of shape (nlags, num_features * Numimf * nlags).
    save_name: str
        path name to save trained model or load that
    use_pre_train : bool, optional (default=True)
        Whether to use pre trained model for forecast or not
    hyper_tune : bool, optional
        Whether to perform hyperparameter tuning or use the optimal hyperparameters from a previous run. Default is False.
    plotResults : bool, optional (default=True)
        Whether to plot the training and validation losses and accuracies, and confusion matrices for the train, validation, and test sets.

    n_trials : int, optional
        The number of trials to run during hyperparameter tuning. Default is 100.

    Returns:
    --------
    forecast : numpy.ndarray
        The forecasted labels for the forecast period as a 1D array of shape (nlags,).
    prob : numpy.ndarray
        The predicted probabilities for the forecasted labels as a 2D array of shape (nlags, num_classes).
    net : transformer.TransformerEncoder
        The trained transformer model.
    acc : float
        The accuracy of the trained model.
    """

    if hyper_tune:
        # Perform hyperparameter tuning and store the optimal hyperparameters
        params = hyper_tune_model(inputs, labels, n_trials=n_trials)
        params['d_encode']=512
        joblib.dump(params, 'params/transformer_hyper_params.pkl')
    else:
        try:
            # Load the optimal hyperparameters from a previous run
            params = joblib.load('params/transformer_hyper_params.pkl')
        except FileNotFoundError:
            params = {
                'learning_rate': 0.0001,
                'dropout': 0.5,
                'd_model': 32,
                'd_hidden': 32,
                'd_encode': 512,
                'q': 1,
                'v': 1,
                'h': 1,
                'N': 1
            }

    # Train the model on the preprocessed inputs and labels

    net, acc, encoding = train(inputs, labels, params, save_name, use_pre_train=use_pre_train, plotResults=plotResults)

    # Generate forecast and probability using the trained model
    pred, prob, encoding_f = forecast(net, inputs_forecast)
    
    try:
        encoding=np.concatenate((np.concatenate(encoding, axis=0),encoding_f.reshape(1,-1)))
    except ValueError:
        encoding=np.concatenate((encoding,encoding_f.reshape(1,-1)))
        
    dataset = torch.utils.data.TensorDataset(torch.tensor(inputs, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    accuracy, outputs, probs, encoding= predict(net, data_loader, "cpu")
    
    probs=np.concatenate(probs)
    scaler=preprocessing.MinMaxScaler()
    scaler.fit(probs.reshape(-1, 1))
    prob=scaler.transform(np.reshape(prob,(1,-1)))[0][0]
    
    encoding=np.concatenate(encoding)

    return pred, prob, net, acc, encoding, outputs

    




    

    

    
