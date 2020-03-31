import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, QuantileTransformer
from sklearn.metrics import classification_report, mean_squared_error, precision_score


def cal_ndcg(n, true, pred):
    """
    calculate the NDCG@n score

    :param n: top n elements to be considered
    :param true: ground truth
    :param pred: predicted values
    :return: ndcg score
    """

    _true = true.reshape(-1)
    _pred = pred.reshape(-1)

    _list = list(zip(_true, _pred))
    true_list = sorted(_list, key=lambda x: x[0], reverse=True)[:n]
    pred_list = sorted(_list, key=lambda x: x[1], reverse=True)[:n]

    true_array = np.zeros((len(true_list), 2))
    pred_array = np.zeros((len(pred_list), 2))

    for i, num in enumerate(true_list):
        true_array[i] = np.array([i + 1, num[0]])

    for i, num in enumerate(pred_list):
        pred_array[i] = np.array([i + 1, num[0]])

    # 使用工业界的算法
        ideal_score = np.sum((2 ** true_array[:, 1] - 1) / np.log2(true_array[:, 0] + 1))
        pred_score = np.sum((2 ** pred_array[:, 1] - 1) / np.log2(pred_array[:, 0] + 1))

    # 使用普通的算法
    # ideal_score = np.sum((true_array[:, 1]) / np.log2(true_array[:, 0] + 1))
    # pred_score = np.sum((pred_array[:, 1]) / np.log2(pred_array[:, 0] + 1))

    return pred_score / ideal_score


def eval_model(method, y_true_1, y_pred_1, scaler):
    """
    Model evaluation (after transforming to the original scale)

    :param method: evaluate method
    :param y_true_1: ground truth
    :param y_pred_1: predicted values
    :param scaler: the scaler used in data pre-processing
    :return:
    """
    assert y_true_1.shape == y_pred_1.shape

    if scaler is not None:
        # y_true = scaler.inverse_transform(y_true_1.reshape(-1, 1)).reshape(y_true_1.shape)
        # y_pred = scaler.inverse_transform(y_pred_1.reshape(-1, 1)).reshape(y_pred_1.shape)
        y_true = scaler.inverse_transform(y_true_1)
        y_pred = scaler.inverse_transform(y_pred_1)
    else:
        y_true = y_true_1.copy()
        y_pred = y_pred_1.copy()

    if method.lower() == 'mae':
        errors = np.abs(y_true - y_pred)
        annual_mae = np.mean(errors, axis=0)
        overall_mae = np.mean(errors, axis=None)

        return overall_mae, annual_mae

    if method.lower() == 'mse':
        errors = np.square(y_true - y_pred)
        annual_mse = np.mean(errors, axis=0)
        overall_mse = np.mean(errors, axis=None)

        return overall_mse, annual_mse

    if method.lower() == 'rmse':
        errors = np.square(y_true - y_pred)
        annual_rmse = np.sqrt(np.mean(errors, axis=0))
        overall_rmse = np.sqrt(np.mean(errors, axis=None))

        return overall_rmse, annual_rmse

    if method.lower() == 'mape':
        # 因为真实的y_true有0，会导致mape非常大，所以不还原了
        # errors = np.abs((y_true - y_pred) / y_true)
        errors = np.abs((y_true_1 - y_pred_1) / y_true_1)
        annual_mape = np.mean(errors, axis=0)
        overall_mape = np.mean(errors, axis=None)

        return overall_mape, annual_mape

    if method.lower() == 'ndcg':
        n = 20
        samples, years = y_true.shape
        annual_ndcg = []

        for year in range(years):
            annual_ndcg.append(cal_ndcg(n, y_true[:, year], y_pred[:, year]))

        overall_ndcg = cal_ndcg(n, np.sum(y_true, axis=1), np.sum(y_pred, axis=1))

        return overall_ndcg, annual_ndcg

    if method.lower() == 'classification':
        # 存在问题，因为是预测分数之后按排位来分类，所以最后的准确率、召回率、F1是一样的
        percentiles = [70, 85, 95]
        ps_true = np.percentile(y_true, percentiles, axis=0)
        ps_pred = np.percentile(y_pred, percentiles, axis=0)

        _y_true = y_true.copy()
        _y_pred = y_pred.copy()

        for i in range(_y_true.shape[0]):
            for j in range(_y_true.shape[1]):
                true_es = _y_true[i][j]
                pred_es = _y_pred[i][j]

                # if np.isclose(true_es, 0.0):
                #     _y_true[i][j] = 0
                # elif 0 < true_es < ps_true[0][j]:
                if true_es < ps_true[0][j]:
                    _y_true[i][j] = 1  # Below the 70th percentile
                elif ps_true[0][j] <= true_es < ps_true[1][j]:
                    _y_true[i][j] = 2  # Below the 85th percentile
                elif ps_true[1][j] <= true_es < ps_true[2][j]:
                    _y_true[i][j] = 3  # Below the 95th percentile
                else:
                    _y_true[i][j] = 4  # Top classes


                # if np.isclose(pred_es, 0.0):
                #     _y_pred[i][j] = 0
                # elif 0 < pred_es < ps_pred[0][j]:
                if pred_es < ps_pred[0][j]:
                    _y_pred[i][j] = 1  # Below the 70th percentile
                elif ps_pred[0][j] <= pred_es < ps_pred[1][j]:
                    _y_pred[i][j] = 2  # Below the 85th percentile
                elif ps_pred[1][j] <= pred_es < ps_pred[2][j]:
                    _y_pred[i][j] = 3  # Below the 95th percentile
                else:
                    _y_pred[i][j] = 4  # Top classes
        return _y_true, _y_pred
        # return precision_score(_y_true.reshape(-1), _y_pred.reshape(-1), average='macro')



def difference(data, interval=1):
    result = np.diff(data, interval, axis=1, prepend=0)
    return result


def add_diff(diff, original_data):
    result = np.zeros(diff.shape)

    for i in range(len(diff)):
        for j in range(len(diff[0])):
            if j == 0:
                result[i][j] = original_data[i][-1][-2] + diff[i][j]
            else:
                result[i][j] = result[i][j - 1] + diff[i][j]
    return result


def scale_data(data, scaler):
    data_reshaped = data.reshape(data.shape[0] * data.shape[1], data.shape[2])

    if scaler.lower() == 'none':
        return None, data
    elif scaler.lower() == 'standard':
        s = StandardScaler()
    elif scaler.lower() == 'minmax':
        s = MinMaxScaler()
    elif scaler.lower() == 'robust':
        s = RobustScaler()
    elif scaler.lower() == 'power':
        s = PowerTransformer()
    elif scaler.lower() == 'quantile':
        s = QuantileTransformer()
    else:
        return None, data

    data_rescaled = s.fit_transform(data_reshaped)
    s.fit(data_reshaped[:, -2].reshape(-1, 1))
    data_rescaled = data_rescaled.reshape(data.shape)

    # 返回es的scaler以便恢复原始数值（近似）
    return s, data_rescaled


def split_data(data, n_input, ratio):
    X, y = data[:, :n_input, :], data[:, n_input:, -2]
    return train_test_split(X, y, test_size=ratio, random_state=20200214, shuffle=True, stratify=data[:, n_input, -1])


def split_data_with_index(data, n_input, ratio):
    X, y = data[:, :n_input, :], data[:, n_input:, -2]
    ids = np.arange(len(X))
    return train_test_split(X, y, ids, test_size=ratio, random_state=20200214, shuffle=True, stratify=data[:, n_input, -1])


def split_data_by_time(data, n_input, n_output, multi_targets):
    if multi_targets:
        X_train, y_train = data[:, :n_input, :], data[:, n_input:-n_output, :]
        X_test, y_test = data[:, -(n_input + n_output):-n_output, :], data[:, -n_output:, :]
    else:
        X_train, y_train = data[:, :n_input, :], data[:, n_input:-n_output, -2]
        X_test, y_test = data[:, -(n_input + n_output):-n_output, :], data[:, -n_output:, -2]
    return X_train, X_test, y_train, y_test
