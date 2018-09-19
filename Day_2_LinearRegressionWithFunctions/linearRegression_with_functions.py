from random import seed, randrange
from csv import reader
from math import sqrt
from sklearn import datasets, linear_model

#Read CSV
def load_csv(filename):
    dataset = list()
    with open(filename, 'rU') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

#Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = row[column].replace(u'\ufeff', '')
        row[column] = float(row[column].strip())

# Split a dataset into a train and test set
def train_test_split(dataset, split):
    train = list()
    train_size = split * len(dataset)
    dataset_copy = list(dataset)
    while len(train) < train_size:
        index = randrange(len(dataset_copy))
        train.append(dataset_copy.pop(index))
    return train, dataset_copy

# Calculate root mean squared error
def rmse_metric(actual, predict):
    sum_error = 0.0
    for i in range(len(actual)):
        predict_error = predict[i] - actual[i]
        sum_error += (predict_error ** 2)
    mean_error = sum_error / float(len(actual))
    return sqrt(mean_error)

#Evaluate an algorithm using a train/test split
def evaluate_algorithm(dataset, algorithm, split, *args):
    train, test = train_test_split(dataset, split)
    test_set = list()
    for row in test:
        row_copy = list(row)
        row_copy[-1] = None
        test_set.append(row_copy)
    predicted = algorithm(train, test_set, *args)
    actual = [row[-1] for row in test]
    rmse = rmse_metric(actual, predicted)
    return rmse

#Calculate MEAN
def mean(values):
    return sum(values) / float(len(values))

#calculate covariance
def covariance(x, mean_x, y, mean_y):
    covar = 0.0
    for i in range(len(x)):
        covar += (x[i] - mean_x) * (y[i] - mean_y)
    return covar

#calculate variance
def variance(values, mean):
    return sum([(x-mean) ** 2 for x in values])

#calculate coefficients
def coefficients(dataset):
    x = [row[0] for row in dataset]
    y = [row[1] for row in dataset]
    x_mean, y_mean = mean(x), mean(y)

    b1 = covariance(x, x_mean, y, y_mean) / variance(x, x_mean)
    b0 = y_mean - b1 * x_mean

    return [b0, b1]

# Linear Regression
def linear_regression(train, test):
    predict = list()
    b0, b1 = coefficients(train)
    for row in test:
        yhat = b0 + b1 * row[0]
        predict.append(yhat)
    return predict

seed(1)

filename = 'insurance.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])):
    str_column_to_float(dataset, i)

split = 0.6
rmse = evaluate_algorithm(dataset, linear_regression, split)
print('RMSE %3f' % (rmse))


