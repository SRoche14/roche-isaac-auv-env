"""
KNN to calculate drag forces and torque given CUREE velocity vector.
Input: x, y, z velocity components
Output: Fx, Fy, Fz (Force components), Mx, My, Mz (Torque Components)

Data: Expects .txt or .csv files

Author: Steven Roche (rochesh@mit.edu)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
import plotly.graph_objects as go

class DragKNNLinear:
    def __init__(self, data_path, input_size, output_size, n_neighbors, val_size, test_size):
        self.input_size = input_size
        self.output_size = output_size
        self.n_neighbors = n_neighbors
        self.val_size = val_size
        self.test_size = test_size
        self.data_path = data_path

    def load_data(self):
        DATA_PATH = self.data_path
        INPUT_SIZE = self.input_size
        OUTPUT_SIZE = self.output_size
        VAL_SIZE = self.val_size
        TEST_SIZE = self.test_size

        dataframes = []
        for index, path in enumerate(DATA_PATH):
            dataframe = pd.read_csv(path)
            if index == 0:
                dataframe.drop([1995], axis=0, inplace=True)
            if index == 1:
                dataframe.drop([27, 2070], axis=0, inplace=True)
            if index == 2:
                dataframe.drop([1177], axis=0, inplace=True)
            dataframe.drop(columns=['v'], inplace=True)
            dataframes.append(dataframe)

        X = np.concatenate([df.iloc[:, :INPUT_SIZE].values.astype(np.float32) for df in dataframes])
        y = np.concatenate([df.iloc[:, INPUT_SIZE:INPUT_SIZE+OUTPUT_SIZE].values.astype(np.float32) for df in dataframes])

        # Train/val/test split (70/15/15)
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=VAL_SIZE, shuffle=True, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=TEST_SIZE, shuffle=True, random_state=42)

        scaler_X = StandardScaler().fit(X_train)
        scaler_y = StandardScaler().fit(y_train)
        X_train = scaler_X.transform(X_train)
        X_val = scaler_X.transform(X_val)
        X_test = scaler_X.transform(X_test)
        y_train = scaler_y.transform(y_train)
        y_val = scaler_y.transform(y_val)
        y_test = scaler_y.transform(y_test)

        return X_train, X_val, X_test, y_train, y_val, y_test, scaler_X, scaler_y

    def train_knn(self):
        X_train, X_val, X_test, y_train, y_val, y_test, scaler_X, scaler_y = self.load_data()
        OUTPUT_SIZE = self.output_size
        N_NEIGHBORS = self.n_neighbors

        # Fit a separate KNN for each output feature
        self.knn_models = []
        for i in range(OUTPUT_SIZE):
            knn = KNeighborsRegressor(n_neighbors=N_NEIGHBORS)
            if i <= 2:
                # explititly get one feature
                single_feature = X_train[:, i].reshape(-1, 1)
                knn.fit(single_feature, y_train[:, i])
            else:
                # torques seem more coupled; keeping the entire velocity vector upon fitting the KNN
                knn.fit(X_train, y_train[:, i])
            self.knn_models.append(knn)
        print(f"Trained {OUTPUT_SIZE} KNN regressors.")


    def validate_knn(self):
        X_train, X_val, X_test, y_train, y_val, y_test, scaler_X, scaler_y = self.load_data()
        # Predict each feature separately
        all_preds = []
        for i, knn in enumerate(self.knn_models):
            if i <= 2:
                single_feature = X_val[:, i].reshape(-1, 1)
                preds = knn.predict(single_feature)
            else:
                preds = knn.predict(X_val)
            all_preds.append(preds.reshape(-1, 1))
        all_preds = np.concatenate(all_preds, axis=1)
        # Inverse transform to original scale
        all_preds_orig = scaler_y.inverse_transform(all_preds)
        y_val_orig = scaler_y.inverse_transform(y_val)
        # Compute errors for each feature
        errors = (all_preds_orig - y_val_orig) * 1000
        mse_newtons = np.mean(errors ** 2)
        print(f"Total Mean Squared Error (MSE) on validation set (in Newtons): {mse_newtons:.4f}")
        # # Plot error distribution for each feature
        # fig = go.Figure()
        # features_list = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']
        # for i in range(errors.shape[1]):
        #     fig.add_trace(go.Box(y=errors[:, i], name=f'{features_list[i]}'))
        # fig.update_layout(
        #     title='Error Distribution by Output Feature (Newtons)',
        #     yaxis_title='Prediction Error'
        # )
        # fig.show()

    def test_knn(self):
        X_train, X_val, X_test, y_train, y_val, y_test, scaler_X, scaler_y = self.load_data()
        # Predict each feature separately
        all_preds = []
        for i, knn in enumerate(self.knn_models):
            if i <= 2:
                single_feature = X_test[:, i].reshape(-1, 1)
                preds = knn.predict(single_feature)
            else:
                preds = knn.predict(X_test)
            all_preds.append(preds.reshape(-1, 1))
        all_preds = np.concatenate(all_preds, axis=1)
        # Inverse transform to original scale
        all_preds_orig = scaler_y.inverse_transform(all_preds)
        y_test_orig = scaler_y.inverse_transform(y_test)
        # Compute errors for each feature
        errors = (all_preds_orig - y_test_orig) * 1000
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

class DragKNNAngular:
    def __init__(self, data_path, input_size, output_size, n_neighbors, val_size, test_size):
        self.input_size = input_size
        self.output_size = output_size
        self.n_neighbors = n_neighbors
        self.val_size = val_size
        self.test_size = test_size
        self.data_path = data_path

    def load_data(self):
        DATA_PATH = self.data_path
        INPUT_SIZE = self.input_size
        OUTPUT_SIZE = self.output_size
        VAL_SIZE = self.val_size
        TEST_SIZE = self.test_size

        dataframes = []
        for index, path in enumerate(DATA_PATH):
            dataframe = pd.read_csv(path)
            # if index == 0:
            #     dataframe.drop([1995], axis=0, inplace=True)
            # if index == 1:
            #     dataframe.drop([27, 2070], axis=0, inplace=True)
            # if index == 2:
            #     dataframe.drop([1177], axis=0, inplace=True)
            dataframe.drop(columns=['v'], inplace=True)
            dataframes.append(dataframe)

        X = np.concatenate([df.iloc[:, :INPUT_SIZE].values.astype(np.float32) for df in dataframes])
        y = np.concatenate([df.iloc[:, INPUT_SIZE:INPUT_SIZE+OUTPUT_SIZE].values.astype(np.float32) for df in dataframes])

         # Additional plots: torque vs angular velocity for each axis
        # X torque vs velocity around x
        # wx = X[:, 0]  # angular velocity around x
        # mx_true = y[:, 3]  # true Mx
        # fig_x = go.Figure()
        # fig_x.add_trace(go.Scatter(x=wx, y=mx_true, mode='markers', name='True Mx'))
        # fig_x.update_layout(title='Mx vs wx!!!!!', xaxis_title='Angular Velocity wx', yaxis_title='Torque Mx')
        # fig_x.show()

        # # Y torque vs velocity around y
        # wy = X[:, 1]
        # my_true = y[:, 4]
        # fig_y = go.Figure()
        # fig_y.add_trace(go.Scatter(x=wy, y=my_true, mode='markers', name='True My'))
        # fig_y.update_layout(title='My vs wy', xaxis_title='Angular Velocity wy', yaxis_title='Torque My')
        # fig_y.show()

        # # Z torque vs velocity around z
        # wz = X[:, 2]
        # mz_true = y[:, 5]
        # fig_z = go.Figure()
        # fig_z.add_trace(go.Scatter(x=wz, y=mz_true, mode='markers', name='True Mz'))
        # fig_z.update_layout(title='Mz vs wz', xaxis_title='Angular Velocity wz', yaxis_title='Torque Mz')
        # fig_z.show()

        # Train/val/test split (70/15/15)
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=VAL_SIZE, shuffle=True, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=TEST_SIZE, shuffle=True, random_state=42)

        scaler_X = StandardScaler().fit(X_train)
        scaler_y = StandardScaler().fit(y_train)
        X_train = scaler_X.transform(X_train)
        X_val = scaler_X.transform(X_val)
        X_test = scaler_X.transform(X_test)
        y_train = scaler_y.transform(y_train)
        y_val = scaler_y.transform(y_val)
        y_test = scaler_y.transform(y_test)

        return X_train, X_val, X_test, y_train, y_val, y_test, scaler_X, scaler_y

    def train_knn(self):
        X_train, X_val, X_test, y_train, y_val, y_test, scaler_X, scaler_y = self.load_data()
        OUTPUT_SIZE = self.output_size
        N_NEIGHBORS = self.n_neighbors

        # Fit a separate KNN for each output feature
        self.knn_models = []
        for i in range(OUTPUT_SIZE):
            knn = KNeighborsRegressor(n_neighbors=N_NEIGHBORS)
            if i <= 2:
                # forces seem more coupled; keeping the entire velocity vector upon fitting the KNN
                knn.fit(X_train, y_train[:, i])
            else:
                # explicitly get one feature. i is in the set {4, 5, 6}, so index mod 3
                # the assumption here is angular velocity in the _ direction corresponds highly
                # with torque in the _ direction
                single_feature = X_train[:, (i % 3)].reshape(-1, 1)
                knn.fit(single_feature, y_train[:, i])
                
            self.knn_models.append(knn)
        print(f"Trained {OUTPUT_SIZE} KNN regressors.")


    def validate_knn(self):
        X_train, X_val, X_test, y_train, y_val, y_test, scaler_X, scaler_y = self.load_data()
        # Predict each feature separately
        all_preds = []
        for i, knn in enumerate(self.knn_models):
            if i <= 2:
                preds = knn.predict(X_val)
            else:
                single_feature = X_val[:, (i % 3)].reshape(-1, 1)
                preds = knn.predict(single_feature)
            all_preds.append(preds.reshape(-1, 1))
        all_preds = np.concatenate(all_preds, axis=1)
        # Inverse transform to original scale
        all_preds_orig = scaler_y.inverse_transform(all_preds)
        y_val_orig = scaler_y.inverse_transform(y_val)
        # Compute errors for each feature
        errors = (all_preds_orig - y_val_orig) * 1000
        mse_newtons = np.mean(errors ** 2)
        print(f"Total Mean Squared Error (MSE) on validation set (in Newtons): {mse_newtons:.4f}")
        # # Plot error distribution for each feature
        # fig = go.Figure()
        # features_list = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']
        # for i in range(errors.shape[1]):
        #     fig.add_trace(go.Box(y=errors[:, i], name=f'{features_list[i]}'))
        # fig.update_layout(
        #     title='Error Distribution by Output Feature (Newtons)',
        #     yaxis_title='Prediction Error'
        # )
        # fig.show()

    def test_knn(self):
        X_train, X_val, X_test, y_train, y_val, y_test, scaler_X, scaler_y = self.load_data()
        # Predict each feature separately
        all_preds = []
        for i, knn in enumerate(self.knn_models):
            if i <= 2:
                preds = knn.predict(X_test)
            else:
                single_feature = X_test[:, (i % 3)].reshape(-1, 1)
                preds = knn.predict(single_feature)
            all_preds.append(preds.reshape(-1, 1))
        all_preds = np.concatenate(all_preds, axis=1)
        # Inverse transform to original scale
        all_preds_orig = scaler_y.inverse_transform(all_preds)
        y_test_orig = scaler_y.inverse_transform(y_test)
        # Compute errors for each feature
        errors = (all_preds_orig - y_test_orig) * 1000
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

        # torque vs angular velocity for each axis
        # X torque vs velocity around x
        wx = X_test[:, 0]  # angular velocity around x
        mx_true = y_test_orig[:, 3]  # true Mx
        mx_pred = all_preds_orig[:, 3]  # predicted Mx
        fig_x = go.Figure()
        fig_x.add_trace(go.Scatter(x=wx, y=mx_true, mode='markers', name='True Mx'))
        fig_x.add_trace(go.Scatter(x=wx, y=mx_pred, mode='markers', name='Predicted Mx'))
        fig_x.update_layout(title='Mx vs wx', xaxis_title='Angular Velocity wx', yaxis_title='Torque Mx')
        fig_x.show()

        # Y torque vs velocity around y
        wy = X_test[:, 1]
        my_true = y_test_orig[:, 4]
        my_pred = all_preds_orig[:, 4]
        fig_y = go.Figure()
        fig_y.add_trace(go.Scatter(x=wy, y=my_true, mode='markers', name='True My'))
        fig_y.add_trace(go.Scatter(x=wy, y=my_pred, mode='markers', name='Predicted My'))
        fig_y.update_layout(title='My vs wy', xaxis_title='Angular Velocity wy', yaxis_title='Torque My')
        fig_y.show()

        # Z torque vs velocity around z
        wz = X_test[:, 2]
        mz_true = y_test_orig[:, 5]
        mz_pred = all_preds_orig[:, 5]
        fig_z = go.Figure()
        fig_z.add_trace(go.Scatter(x=wz, y=mz_true, mode='markers', name='True Mz'))
        fig_z.add_trace(go.Scatter(x=wz, y=mz_pred, mode='markers', name='Predicted Mz'))
        fig_z.update_layout(title='Mz vs wz', xaxis_title='Angular Velocity wz', yaxis_title='Torque Mz')
        fig_z.show()


def main():
    DATA_PATH_LINEAR = ['dragData.txt', 'dragData2.txt', 'dragData3.csv']
    DATA_PATH_ANGULAR = ['AngVelAndTorque.csv', 'AngVelAndTorque2.csv']  # CSV/TXT files
    INPUT_SIZE = 3
    OUTPUT_SIZE = 6
    N_LINEAR_NEIGHBORS = 12 # Chosen via grid search on the set {i : 1 \le i \le 20}
    N_ANGULAR_NEIGHBORS = 15 # Chosen via grid search on the set {i : 1 \le i \le 20}
    VAL_SIZE = 0.10
    TEST_SIZE = .10/0.90

    # drag_knn_linear = DragKNNLinear(DATA_PATH_LINEAR, INPUT_SIZE, OUTPUT_SIZE, N_LINEAR_NEIGHBORS, VAL_SIZE, TEST_SIZE)
    # drag_knn_linear.train_knn()
    # drag_knn.validate_knn()
    # drag_knn.test_knn()

    drag_knn_angular = DragKNNAngular(DATA_PATH_ANGULAR, INPUT_SIZE, OUTPUT_SIZE, N_ANGULAR_NEIGHBORS, VAL_SIZE, TEST_SIZE)
    drag_knn_angular.train_knn()
    # drag_knn_angular.validate_knn()
    drag_knn_angular.test_knn()

    # for i in range(1, 20):
    #     N_ANGULAR_NEIGHBORS = i
    #     print(f"Number of neighbors used: {i}")
    #     drag_knn_angular = DragKNNAngular(DATA_PATH_ANGULAR, INPUT_SIZE, OUTPUT_SIZE, N_ANGULAR_NEIGHBORS, VAL_SIZE, TEST_SIZE)
    #     drag_knn_angular.train_knn()
    #     drag_knn_angular.validate_knn()
    #     # drag_knn_angular.test_knn()


if __name__ == "__main__":
    main()