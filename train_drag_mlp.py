"""
MLP to calculate drag forces and torque given CUREE velocity vector.
Input: x, y, z velocity components
Output: Fx, Fy, Fz (Force components), Mx, My, Mz (Torque Components)

Data: Expects .txt or .csv files

Author: Steven Roche (rochesh@mit.edu)
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import joblib
import os


class DragMLPLinear():
    def __init__(self, data_path, input_size, output_size, batch_size, epochs, learning_rate, val_size, test_size):
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.val_size = val_size
        self.test_size = test_size
        self.data_path = data_path

    def load_data(self, fit_scalers: bool = True):
        """
        Parameters:
            data_path (str): Path to the CSV data file.
            input_size (int): Number of input features.
            output_size (int): Number of output targets.
            batch_size (int): Batch size for training.
            epochs (int): Number of training epochs.
            learning_rate (float): Learning rate for the optimizer.
            val_size (float): Fraction of data to use for validation (from the original dataset).
            test_size (float): Fraction of data to use for test (from the remaining data after validation split).

        Behavior:
            Loads and preprocesses the data (including dropping specified columns/rows).
            Splits the data into train, validation, and test sets.
            Standardizes features and targets based on the training set.
        """

        DATA_PATH = self.data_path
        INPUT_SIZE = self.input_size
        OUTPUT_SIZE = self.output_size
        VAL_SIZE = self.val_size
        TEST_SIZE = self.test_size
        BATCH_SIZE = self.batch_size

        dataframes = []
        # I assume the header is the same for all dataframes
        # header format: v, x, y, z, Fx, Fy, Fz, Mx, My, Mz
        for index, path in enumerate(DATA_PATH):
            dataframe = pd.read_csv(path)
            # if index == 0:
            #     # 1, 709, 440, 1995
            #     dataframe.drop([1995], axis=0, inplace=True)
            # if index == 1:
            #     print(dataframe.iloc[[27,2070]])
            #     dataframe.drop([27, 2070], axis=0, inplace=True)
            # if index == 2:
            #     print(dataframe.iloc[[1177]])
            #     dataframe.drop([1177], axis=0, inplace=True)
            # remove velocity magnitude
            dataframe.drop(columns=['v'], inplace=True)
            dataframes.append(dataframe)

        X = np.concatenate([df.iloc[:, :INPUT_SIZE].values.astype(np.float32) for df in dataframes])
        print(X.shape)
        y = np.concatenate([df.iloc[:, INPUT_SIZE:INPUT_SIZE+OUTPUT_SIZE].values.astype(np.float32) for df in dataframes])

        # Train/val/test split (70/15/15)
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=VAL_SIZE, shuffle=True, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=TEST_SIZE, shuffle=True, random_state=42)

        # Standardize features
        if fit_scalers:
            scaler_X = StandardScaler().fit(X_train)
            scaler_y = StandardScaler().fit(y_train)
            # persist to instance for saving later
            self.scaler_X = scaler_X
            self.scaler_y = scaler_y
        else:
            # use pre-fitted scalers stored on instance
            scaler_X = getattr(self, 'scaler_X', None)
            scaler_y = getattr(self, 'scaler_y', None)
            if scaler_X is None or scaler_y is None:
                raise ValueError("Scalers not set. Load or fit scalers before calling load_data(fit_scalers=False).")

        X_train = scaler_X.transform(X_train)
        X_val = scaler_X.transform(X_val)
        X_test = scaler_X.transform(X_test)
        y_train = scaler_y.transform(y_train)
        y_val = scaler_y.transform(y_val)
        y_test = scaler_y.transform(y_test)

        # Plot the entire training dataset by feature (6 plots, one for each output feature)
        
        # features_list = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']
        # input_features = ['x', 'y', 'z']
        # fig, axs = plt.subplots(3, 6, figsize=(18, 10))
        # axs = axs.flatten()
        # for i in range(3):
        #     for j in range(6):
        #         index = i * 6 + j
        #         axs[index].scatter(X_train[:, i], y_train[:, j], alpha=0.3, label=f'{input_features[i]} vs {features_list[j]}')
        #         axs[index].set_xlabel('Input Feature Value (standardized)')
        #         axs[index].set_ylabel(f'{features_list[j]} (standardized)')
        #         axs[index].set_title(f'Training Data: {features_list[j]} vs {input_features[i]}')
        #         axs[index].legend()
        # plt.tight_layout()
        # plt.show()

        # PyTorch Dataset
        class DragDataset(Dataset):
            def __init__(self, X, y):
                self.X = torch.tensor(X, dtype=torch.float32)
                self.y = torch.tensor(y, dtype=torch.float32)
            def __len__(self):
                return len(self.X)
            def __getitem__(self, idx):
                return self.X[idx], self.y[idx]

        train_ds = DragDataset(X_train, y_train)
        val_ds = DragDataset(X_val, y_val)
        test_ds = DragDataset(X_test, y_test)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

        return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader, scaler_y

    def train_mlp(self):
        """
        Train a simple MLP regressor on the provided dataset using a traditional train/validation/test split.

        Parameters:
            - input_size (int): Number of input features.
            - output_size (int): Number of output targets.
            - epochs (int): Number of training epochs.
            - learning_rate (float): Learning rate for the optimizer.

        Behavior:
            - Loads and preprocesses the data (including dropping specified columns/rows).
            - Splits the data into train, validation, and test sets.
            - Standardizes features and targets based on the training set.
            - Trains an MLP regressor, monitoring training and validation loss.
            - Plots training and validation loss curves using Plotly.
            - Optionally evaluates and prints test set loss after training.
        """

        train_ds, val_ds, _, train_loader, val_loader, _, scaler_y = self.load_data(fit_scalers=True)
        INPUT_SIZE = self.input_size 
        OUTPUT_SIZE = self.output_size 
        EPOCHS = self.epochs 
        LEARNING_RATE = self.learning_rate 

        # Simple MLP Model 
        def make_mlp(input_size, output_size):
            """
            Construct a simple feedforward neural network (MLP) with one hidden layer.

            Parameters:
                - input_size (int): Number of input features.
                - output_size (int): Number of output targets.

            Returns:
                - nn.Sequential: The constructed MLP model.
            """
            return nn.Sequential(
                nn.Linear(input_size, 1024),
                nn.ReLU(),
                # nn.Dropout(p=0.05),
                nn.Linear(1024, 512),
                nn.ReLU(),
                # nn.Dropout(p=0.05),
                # nn.Linear(1024, 512),
                # nn.ReLU(),
                nn.Linear(512, output_size)
            )

        model = make_mlp(INPUT_SIZE, OUTPUT_SIZE)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        train_losses = []
        val_losses = []

        best_val_loss = float('inf')

        # 6. Training loop with validation
        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0
            for xb, yb in train_loader:
                optimizer.zero_grad()
                preds = model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * xb.size(0)
            epoch_loss = running_loss / len(train_ds)
            train_losses.append(epoch_loss)

            # Validation loss
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for xb1, yb1 in val_loader:
                    preds = model(xb1)
                    loss = criterion(preds, yb1)
                    val_loss += loss.item() * xb1.size(0)
                val_loss /= len(val_ds)
            val_losses.append(val_loss)
            print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                os.makedirs('./temp_best_models', exist_ok=True)
                torch.save(model, './temp_best_models/best_model_linear.pt')
                # save the fitted scalers alongside the model
                if hasattr(self, 'scaler_X') and hasattr(self, 'scaler_y'):
                    joblib.dump(self.scaler_X, './temp_best_models/scaler_X_linear.pkl')
                    joblib.dump(self.scaler_y, './temp_best_models/scaler_y_linear.pkl')
                print(f"Best linear model saved at epoch {epoch+1} with val loss {val_loss:.4f}")

        # 7. Plot training and validation loss
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=train_losses, mode='lines', name='Train Loss'))
        fig.add_trace(go.Scatter(y=val_losses, mode='lines', name='Validation Loss'))
        fig.update_layout(title='Training and Validation Loss', xaxis_title='Epoch', yaxis_title='MSE Loss')
        fig.show()

    def test_mlp(self):
        """
        Test the MLP model on the test set.

        Returns:
            - float: The test set loss, and plot of predicted vs actual values
        """
        # load persisted model and scalers
        model = torch.load('./temp_best_models/best_model_linear.pt')
        self.scaler_X = joblib.load('./temp_best_models/scaler_X_linear.pkl')
        self.scaler_y = joblib.load('./temp_best_models/scaler_y_linear.pkl')
        # reload data using the persisted scalers to ensure consistent transforms
        _, _, _, _, _, test_loader, _ = self.load_data(fit_scalers=False)
        model.eval() 
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for xb, yb in test_loader:
                preds = model(xb)
                all_preds.append(preds.detach().numpy())
                all_targets.append(yb.detach().numpy())
        
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets) 

        # Inverse transform to original scale
        all_preds_orig = self.scaler_y.inverse_transform(all_preds)
        all_targets_orig = self.scaler_y.inverse_transform(all_targets)

        # Compute errors for each feature
        # shape: (num_samples, num_outputs)

        # multiply by 1000 to convert to Newtons (water density is 1000 kg/m^3)
        errors = (all_preds_orig - all_targets_orig) * 1000

        mse_newtons = np.mean(errors ** 2)
        print(f"Total Mean Squared Error (MSE) on test set (in Newtons): {mse_newtons:.4f}")

        # Plot error distribution for each feature
        fig = go.Figure()
        features_list = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']
        for i in range(errors.shape[1]):
            fig.add_trace(go.Box(y=errors[:, i], name=f'{features_list[i]}'))
        fig.update_layout(
            title='Error Distribution by Output Feature (Newtons)',
            yaxis_title='Prediction Error'
        )
        fig.show()

class DragMLPAngular():
    def __init__(self, data_path, input_size, output_size, batch_size, epochs, learning_rate, val_size, test_size):
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.val_size = val_size
        self.test_size = test_size
        self.data_path = data_path

    def load_data(self, fit_scalers: bool = True):
        """
        Parameters:
            data_path (str): Path to the CSV data file.
            input_size (int): Number of input features.
            output_size (int): Number of output targets.
            batch_size (int): Batch size for training.
            epochs (int): Number of training epochs.
            learning_rate (float): Learning rate for the optimizer.
            val_size (float): Fraction of data to use for validation (from the original dataset).
            test_size (float): Fraction of data to use for test (from the remaining data after validation split).

        Behavior:
            Loads and preprocesses the data (including dropping specified columns/rows).
            Splits the data into train, validation, and test sets.
            Standardizes features and targets based on the training set.
        """

        DATA_PATH = self.data_path
        INPUT_SIZE = self.input_size
        OUTPUT_SIZE = self.output_size
        VAL_SIZE = self.val_size
        TEST_SIZE = self.test_size
        BATCH_SIZE = self.batch_size

        dataframes = []
        # I assume the header is the same for all dataframes
        # header format: v, x, y, z, Fx, Fy, Fz, Mx, My, Mz
        for index, path in enumerate(DATA_PATH):
            dataframe = pd.read_csv(path)
            # if index == 0:
            #     # 1, 709, 440, 1995
            #     dataframe.drop([1995], axis=0, inplace=True)
            # if index == 1:
            #     print(dataframe.iloc[[27,2070]])
            #     dataframe.drop([27, 2070], axis=0, inplace=True)
            # if index == 2:
            #     print(dataframe.iloc[[1177]])
            #     dataframe.drop([1177], axis=0, inplace=True)
            # remove velocity magnitude
            dataframe.drop(columns=['v'], inplace=True)
            dataframes.append(dataframe)

        X = np.concatenate([df.iloc[:, :INPUT_SIZE].values.astype(np.float32) for df in dataframes])
        y = np.concatenate([df.iloc[:, INPUT_SIZE:INPUT_SIZE+OUTPUT_SIZE].values.astype(np.float32) for df in dataframes])

        # Train/val/test split (70/15/15)
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=VAL_SIZE, shuffle=True, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=TEST_SIZE, shuffle=True, random_state=42)

        # Standardize features
        if fit_scalers:
            scaler_X = StandardScaler().fit(X_train)
            scaler_y = StandardScaler().fit(y_train)
            self.scaler_X = scaler_X
            self.scaler_y = scaler_y
        else:
            scaler_X = getattr(self, 'scaler_X', None)
            scaler_y = getattr(self, 'scaler_y', None)
            if scaler_X is None or scaler_y is None:
                raise ValueError("Scalers not set. Load or fit scalers before calling load_data(fit_scalers=False).")

        X_train = scaler_X.transform(X_train)
        X_val = scaler_X.transform(X_val)
        X_test = scaler_X.transform(X_test)
        y_train = scaler_y.transform(y_train)
        y_val = scaler_y.transform(y_val)
        y_test = scaler_y.transform(y_test)

        # Plot the entire training dataset by feature (6 plots, one for each output feature)
        # import matplotlib.pyplot as plt
        # features_list = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']
        # input_features = ['x', 'y', 'z']
        # fig, axs = plt.subplots(3, 6, figsize=(18, 10))
        # axs = axs.flatten()
        # for i in range(3):
        #     for j in range(6):
        #         index = i * 6 + j
        #         axs[index].scatter(X_train[:, i], y_train[:, j], alpha=0.3, label=f'{input_features[i]} vs {features_list[j]}')
        #         axs[index].set_xlabel('Input Feature Value (standardized)')
        #         axs[index].set_ylabel(f'{features_list[j]} (standardized)')
        #         axs[index].set_title(f'Training Data: {features_list[j]} vs {input_features[i]}')
        #         axs[index].legend()
        # plt.tight_layout()
        # plt.show()

        # PyTorch Dataset
        class DragDataset(Dataset):
            def __init__(self, X, y):
                self.X = torch.tensor(X, dtype=torch.float32)
                self.y = torch.tensor(y, dtype=torch.float32)
            def __len__(self):
                return len(self.X)
            def __getitem__(self, idx):
                return self.X[idx], self.y[idx]

        train_ds = DragDataset(X_train, y_train)
        val_ds = DragDataset(X_val, y_val)
        test_ds = DragDataset(X_test, y_test)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

        return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader, scaler_y

    def train_mlp(self):
        """
        Train a simple MLP regressor on the provided dataset using a traditional train/validation/test split.

        Parameters:
            - input_size (int): Number of input features.
            - output_size (int): Number of output targets.
            - epochs (int): Number of training epochs.
            - learning_rate (float): Learning rate for the optimizer.

        Behavior:
            - Loads and preprocesses the data (including dropping specified columns/rows).
            - Splits the data into train, validation, and test sets.
            - Standardizes features and targets based on the training set.
            - Trains an MLP regressor, monitoring training and validation loss.
            - Plots training and validation loss curves using Plotly.
            - Optionally evaluates and prints test set loss after training.
        """

        train_ds, val_ds, _, train_loader, val_loader, _, scaler_y = self.load_data(fit_scalers=True)
        INPUT_SIZE = self.input_size 
        OUTPUT_SIZE = self.output_size 
        EPOCHS = self.epochs 
        LEARNING_RATE = self.learning_rate 


        # Simple MLP Model 
        def make_mlp(input_size, output_size):
            """
            Construct a simple feedforward neural network (MLP) with one hidden layer.

            Parameters:
                - input_size (int): Number of input features.
                - output_size (int): Number of output targets.

            Returns:
                - nn.Sequential: The constructed MLP model.
            """
            return nn.Sequential(
                nn.Linear(input_size, 1024),
                nn.ReLU(),
                # nn.Linear(256, 128),
                # nn.ReLU(),
                nn.Linear(1024, output_size)
            )

        model = make_mlp(INPUT_SIZE, OUTPUT_SIZE)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

        train_losses = []
        val_losses = []

        best_val_loss = float('inf')

        # 6. Training loop with validation
        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0
            for xb, yb in train_loader:
                optimizer.zero_grad()
                preds = model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * xb.size(0)
            epoch_loss = running_loss / len(train_ds)
            train_losses.append(epoch_loss)

            # Validation loss
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for xb1, yb1 in val_loader:
                    preds = model(xb1)
                    loss = criterion(preds, yb1)
                    val_loss += loss.item() * xb1.size(0)
                val_loss /= len(val_ds)
            val_losses.append(val_loss)
            print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                os.makedirs('./temp_best_models', exist_ok=True)
                torch.save(model, './temp_best_models/best_model_angular.pt')
                if hasattr(self, 'scaler_X') and hasattr(self, 'scaler_y'):
                    joblib.dump(self.scaler_X, './temp_best_models/scaler_X_angular.pkl')
                    joblib.dump(self.scaler_y, './temp_best_models/scaler_y_angular.pkl')
                print(f"Best angular model saved at epoch {epoch+1} with val loss {val_loss:.4f}")

        # 7. Plot training and validation loss
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=train_losses, mode='lines', name='Train Loss'))
        fig.add_trace(go.Scatter(y=val_losses, mode='lines', name='Validation Loss'))
        fig.update_layout(title='Training and Validation Loss', xaxis_title='Epoch', yaxis_title='MSE Loss')
        fig.show()

    def test_mlp(self):
        """
        Test the MLP model on the test set.

        Returns:
            - float: The test set loss, and plot of predicted vs actual values
        """
        model = torch.load('./temp_best_models/best_model_angular.pt')
        self.scaler_X = joblib.load('./temp_best_models/scaler_X_angular.pkl')
        self.scaler_y = joblib.load('./temp_best_models/scaler_y_angular.pkl')
        _, _, _, _, _, test_loader, _ = self.load_data(fit_scalers=False)
        model.eval() 
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for xb, yb in test_loader:
                preds = model(xb)
                all_preds.append(preds.detach().numpy())
                all_targets.append(yb.detach().numpy())
        
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets) 

        # Inverse transform to original scale
        all_preds_orig = self.scaler_y.inverse_transform(all_preds)
        all_targets_orig = self.scaler_y.inverse_transform(all_targets)

        # Compute errors for each feature
        # shape: (num_samples, num_outputs)

        # multiply by 1000 to convert to Newtons (water density is 1000 kg/m^3)
        errors = (all_preds_orig - all_targets_orig) * 1000

        mse_newtons = np.mean(errors ** 2)
        print(f"Total Mean Squared Error (MSE) on test set (in Newtons): {mse_newtons:.4f}")

        # Plot error distribution for each feature
        fig = go.Figure()
        features_list = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']
        for i in range(errors.shape[1]):
            fig.add_trace(go.Box(y=errors[:, i], name=f'{features_list[i]}'))
        fig.update_layout(
            title='Error Distribution by Output Feature (Newtons)',
            yaxis_title='Prediction Error'
        )
        fig.show()

def main(linear=False, train=False, test=False):
    """
    Main entry point for the script. Sets default parameters and calls the training function.
    """
    # Parameters
    DATA_PATH_LINEAR = ['curee_full_model/7-15LinData.csv', 'curee_full_model/7-16lindata.csv', 'curee_full_model/7-17lindata.csv', 'curee_full_model/7-22lindata.csv', 'curee_full_model/8-11lindata.csv']
    DATA_PATH_ANGULAR = ['curee_full_model/NewAngularResults.csv', 'curee_full_model/8-11angdata.csv']  # CSV/TXT files
    INPUT_SIZE = 3
    OUTPUT_SIZE = 6
    BATCH_SIZE_LINEAR = 32
    EPOCHS_LINEAR = 400
    LEARNING_RATE_LINEAR = 2e-3

    BATCH_SIZE_ANGULAR = 32
    EPOCHS_ANGULAR = 400
    LEARNING_RATE_ANGULAR = 2e-4
    VAL_SIZE = 0.10
    TEST_SIZE = .10/0.90 # 10% of the remaining data after validation split

    if linear:
        print(f"Setting Applied: Linear Model")
        drag_mlp_linear = DragMLPLinear(DATA_PATH_LINEAR, INPUT_SIZE, OUTPUT_SIZE, BATCH_SIZE_LINEAR, 
                                        EPOCHS_LINEAR, LEARNING_RATE_LINEAR, VAL_SIZE, TEST_SIZE)
        if train:
            drag_mlp_linear.train_mlp()

        if test:
            drag_mlp_linear.test_mlp()
    else:
        print(f"Setting Applied: Angular Model")
        drag_mlp_angular = DragMLPAngular(DATA_PATH_ANGULAR, INPUT_SIZE, OUTPUT_SIZE, 
                                            BATCH_SIZE_ANGULAR, EPOCHS_ANGULAR, LEARNING_RATE_ANGULAR, VAL_SIZE, TEST_SIZE)
        if train:
            drag_mlp_angular.train_mlp()
        
        if test:
            drag_mlp_angular.test_mlp()


if __name__ == "__main__":
    main(linear=True, train=True, test=True) 