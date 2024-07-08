import os
from PIL import Image
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import MinMaxScaler
import csv
import cv2
import pandas as pd
from imblearn.over_sampling import SMOTE

import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

train_normal_dir = 'train/NORMAL'
test_normal_dir = 'test/NORMAL'
train_pneumonia_dir = 'train/PNEUMONIA'
test_pneumonia_dir = 'test/PNEUMONIA'
from sklearn.metrics import roc_curve, auc


def plot_roc_curve(model, x_test, y_test):
    y_probs = model.predict_proba(x_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange',
             lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def timing_function(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"    [Time per function]{func.__name__} took {end - start:.2f} seconds")
        return result

    return wrapper


def report_data_distribution(x_train, x_val, x_test):
    total_count = len(x_train) + len(x_val) + len(x_test)
    train_percentage = len(x_train) / total_count * 100
    val_percentage = len(x_val) / total_count * 100
    test_percentage = len(x_test) / total_count * 100
    print(f"Training data: {len(x_train)} samples, {train_percentage:.2f}% of total data")
    print(f"Validation data: {len(x_val)} samples, {val_percentage:.2f}% of total data")
    print(f"Testing data: {len(x_test)} samples, {test_percentage:.2f}% of total data")


def crop_lungs_from_image(img):
    img_cv = np.array(img)
    _, thresh = cv2.threshold(img_cv, 15, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped_img_cv = img_cv[y:y + h, x:x + w]
    cropped_img = Image.fromarray(cropped_img_cv)
    return cropped_img


@timing_function
def load_images_from_folder(folder, label, size):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        with Image.open(img_path) as img:
            # Greyish colors = L, Normal colors = P
            img = img.convert('L')
            img = crop_lungs_from_image(img)
            # Size of the images
            img = img.resize((size, size))
            img_array = np.array(img)
            # Normalize pixels so the value will be either 0 or 1
            scaler = MinMaxScaler(feature_range=(0, 1))
            img_array = scaler.fit_transform(img_array.reshape(-1, 1))
            img_array = img_array.flatten()
            images.append(img_array.flatten())
            labels.append(label)

    return images, labels


def load_and_split_data(train_normal_path, train_pneumonia_path, virus_bacteria='false'):
    size = 250
    print("Loading images for train normal data")
    normal_images, normal_labels = load_images_from_folder(train_normal_path, 0, size)
    if virus_bacteria != 'false':
        print("Loading bacteria images for train pneumonia data")
        pneumonia_bacteria_images, pneumonia_bacteria_labels = load_images_from_folder(train_pneumonia_path, 1, size)

    else:
        print("Loading images for train pneumonia data")
        pneumonia_images, pneumonia_labels = load_images_from_folder(train_pneumonia_path, 1, size)
    x = normal_images + pneumonia_images
    print(f"Training data: {len(x)} samples")

    y = normal_labels + pneumonia_labels
    # splitting the number of train images
    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.2,
                                                        random_state=42)
    # adding the splitted images into the validation and test set
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5,
                                                    random_state=42)
    print("Loading images for test normal data")
    normal_test_images, normal_test_labels = load_images_from_folder(test_normal_dir, 0, size)
    print("Loading images for test pneumonia data")
    pneumonia_test_images, pneumonia_test_labels = load_images_from_folder(test_pneumonia_dir, 1, size)
    print(
        f"Testing data: We have {len(pneumonia_test_images)} pneumonia samples, {len(normal_test_images)}"
        f" normal samples, for a total of {len(x_test)} samples")
    x_test = x_test + normal_test_images + pneumonia_test_images
    print(f"Testing data: {len(x_test)} samples")
    y_test = y_test + normal_test_labels + pneumonia_test_labels
    report_data_distribution(x_train, x_val, x_test)
    return x_train, x_test, x_val, y_val, y_train, y_test


def process_new_image(image_path):
    size = 68
    with Image.open(image_path) as img:
        img = img.convert('L')
        img = crop_lungs_from_image(img)
        img = img.resize((size, size))
        img_array = np.array(img)
        scaler = MinMaxScaler(feature_range=(0, 1))
        img_array = scaler.fit_transform(img_array.reshape(-1, 1))
        img_array = img_array.flatten()
        return img_array.flatten()


@timing_function
def train_model(x_train_value, y_train_value):
    print("Starting training model")
    model = KNeighborsClassifier(algorithm= 'auto', metric= 'manhattan', n_neighbors= 15, weights= 'uniform')
    model.fit(x_train_value, y_train_value)
    print("Finished training model")
    return model



@timing_function
def test_predict_model(model, x_test_value, y_test_value):
    start_time = time.time()
    print("Starting test model")
    accuracy = model.score(x_test_value, y_test_value)
    print(f'Test Accuracy: {accuracy:.2f}')
    # Predict test dataset
    y_pred = model.predict(x_test_value)
    report = classification_report(y_test_value, y_pred, target_names=['Normal', 'Pneumonia'], output_dict=True)
    plot_confusion_matrix(y_test_value, y_pred, classes=np.array(['Normal', 'Pneumonia']), normalize=True)
    plt.show()

    print(report)
    print("Finished test model")
    testing_time = time.time() - start_time
    return accuracy, report, testing_time


@timing_function
def predict_function(image_path, model, X_train, y_train):
    start_time = time.time()
    print("Starting predicting model for the picture at", image_path)
    image = process_new_image(image_path)
    prediction = model.predict([image])
    print("Predicted Class:", "Pneumonia" if prediction[0] == 1 else "Normal")
    prediction_time = time.time() - start_time
    return prediction[0], prediction_time


@timing_function
def cross_validation():
    size = 68
    # x_train, x_test, x_val, y_val, y_train, y_test = load_and_split_data(train_normal_dir, train_pneumonia_dir)
    normal_images, normal_labels = load_images_from_folder(train_normal_dir, 0, size)
    pneumonia_images, pneumonia_labels = load_images_from_folder(train_pneumonia_dir, 1, size)
    x = normal_images + pneumonia_images
    y = normal_labels + pneumonia_labels

    # Split into train+val and test initially
    x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, test_size=0.20, stratify=y, random_state=42)
    param_grid = {'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15, 17],
                  'weights': ['uniform', 'distance'],
                  'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                  'metric': ['euclidean', 'manhattan']}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid, scoring='accuracy', cv=cv, verbose=3, n_jobs=-1)
    grid_search.fit(x_train_val, y_train_val)
    print("Best parameters found: ", grid_search.best_params_)
    best_knn = grid_search.best_estimator_

    y_test_pred = best_knn.predict(x_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Test Accuracy: {test_accuracy:.2f}")
    report = classification_report(y_test, y_test_pred, target_names=['Normal', 'Pneumonia'])
    print(report)
    return best_knn


def k_nearest_neighbors():
    x_train, x_test, x_val, y_val, y_train, y_test = load_and_split_data(train_normal_dir, train_pneumonia_dir)

    # Initialize KNN model
    knn = train_model(x_train, y_train)
    plot_roc_curve(knn, x_test, y_test)


    # Test the model
    accuracy, report, test_time = test_predict_model(knn, x_test, y_test)

    predict, predict_time = predict_function(
        'testito/person1952_bacteria_4883.jpeg', knn,
        x_train, y_train)
    if predict == 1:
        print("Prediction successful")
    else:
        print("Prediction unsuccessful")

    # to add to a previous file modify the w into a
    with open('train/Knn_metrics.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Prediction Time', f"{predict_time:.2f} seconds"])
        writer.writerow(['Test Accuracy', f"{accuracy:.2f}"])
        writer.writerow([])
        writer.writerow(['Detailed report'])
        writer.writerow(['Class', 'Precision', 'Recall', 'F1-Score', 'Support'])
        for key, metrics in report.items():
            if key in ["Normal", "Pneumonia"]:
                writer.writerow([key, f"{metrics['precision']:.2f}", f"{metrics['recall']:.2f}",
                                 f"{metrics['f1-score']:.2f}", metrics['support']])


# tests for the optimal size
def model_dimension_precision():
    i = 1
    response_array = []
    with open('Knn_Size_Precision.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Size', 'Accuracy', 'Normal Precision', 'Pneumonia precision', 'Time'])
    while i <= 500:
        print("I am at " + str(i))
        start_time = time.time()
        normal_images, normal_labels = load_images_from_folder(train_normal_dir, 0, i)
        pneumonia_images, pneumonia_labels = load_images_from_folder(train_pneumonia_dir, 1, i)
        x_train = normal_images + pneumonia_images
        y_train = normal_labels + pneumonia_labels
        normal_test_images, normal_test_labels = load_images_from_folder(test_normal_dir, 0, i)
        pneumonia_test_images, pneumonia_test_labels = load_images_from_folder(test_pneumonia_dir, 1, i)
        x_test = normal_test_images + pneumonia_test_images
        y_test = normal_test_labels + pneumonia_test_labels
        knn, train_time = train_model(x_train, y_train, n=5)
        accuracy, report, test_time = test_predict_model(knn, x_test, y_test)
        accuracy = report['accuracy']
        precision_normal = report['Normal']['precision']
        precision_pneumonia = report['Pneumonia']['precision']
        work_time = time.time() - start_time
        response_array.append([i, accuracy, precision_normal, precision_pneumonia, work_time])
        with open('train/size_precision.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([i, accuracy, precision_normal, precision_pneumonia, work_time])
        if i < 4:
            i += 1
        else:
            i = i + 4
    print("Data written to 'size_precision.csv'.")


def show_size_plot():
    data = pd.read_csv('train/size_precision.csv')
    x = data['Size'].values
    y1 = data['Accuracy'].values
    y2 = data['Pneumonia precision'].values
    y3 = data['Normal Precision'].values
    max_normal = y3.argmax()
    max_pneumonia = y2.argmax()
    max_precision = y2.argmax()
    plt.figure(figsize=(10, 6))
    plt.plot(x, y1, label=f'Global Accuracy(Max: {max_precision:.2f})', marker='o', linestyle='-')
    plt.plot(x, y2, label=f'Pneumonia Precision(Max: {max_pneumonia:.2f})', marker='o', linestyle='--')
    plt.plot(x, y3, label=f'Normal Precision(Max: {max_normal:.2f})', marker='o', linestyle=':')

    plt.scatter(x[max_normal], y3[max_normal], color='green', s=100, edgecolors='black',
                label='Max Normal Precision')
    plt.scatter(x[max_pneumonia], y2[max_pneumonia], color='red', s=100, edgecolors='black',
                label='Max Pneumonia Precision')
    plt.scatter(x[max_precision], y1[max_precision], color='blue', s=100, edgecolors='black',
                label='Max Precision')
    plt.xlabel('Size')
    plt.ylabel('%precision')
    plt.title('Precision by Image Size')
    plt.legend()
    plt.grid(True)
    plt.show()


def show_k_plot():
    data = pd.read_csv('train/Knn_Numberofk_Precision.csv')
    x = data['Number of neighbors'].values
    y1 = data['Accuracy'].values
    y2 = data['Pneumonia precision'].values
    y3 = data['Normal Precision'].values
    max_normal = y3.argmax()
    max_pneumonia = y2.argmax()
    max_precision = y2.argmax()
    plt.figure(figsize=(25, 15))
    plt.plot(x, y1, label=f'Global Accuracy(Max: {max_precision:.2f})', marker='o', linestyle='-')
    plt.plot(x, y2, label=f'Pneumonia Precision(Max: {max_pneumonia:.2f})', marker='o', linestyle='--')
    plt.plot(x, y3, label=f'Normal Precision(Max: {max_normal:.2f})', marker='o', linestyle=':')

    plt.scatter(x[max_normal], y3[max_normal], color='green', s=100, edgecolors='black',
                label='Max Normal Precision')
    plt.scatter(x[max_pneumonia], y2[max_pneumonia], color='red', s=100, edgecolors='black',
                label='Max Pneumonia Precision')
    plt.scatter(x[max_precision], y1[max_precision], color='blue', s=100, edgecolors='black',
                label='Max Precision')
    plt.xlabel('Size')
    plt.ylabel('%precision')
    plt.title('Precision based on the number on neighbors')
    plt.legend()
    plt.show()


# tests for the optimal size
def model_numberofk_precision():
    i = 1
    response_array = []
    x_train, x_test, y_train, y_test = load_and_split_data(train_normal_dir, train_pneumonia_dir)
    with open('train/Knn_Numberofk_Precision.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Number of neighbors', 'Accuracy', 'Normal Precision', 'Pneumonia precision', 'Time'])
    while i <= 150:
        print("I am at " + str(i))
        start_time = time.time()
        knn, train_time = train_model(x_train, y_train, n=i)
        accuracy, report, test_time = test_predict_model(knn, x_test, y_test)
        accuracy = report['accuracy']
        precision_normal = report['Normal']['precision']
        precision_pneumonia = report['Pneumonia']['precision']
        work_time = time.time() - start_time
        response_array.append([i, accuracy, precision_normal, precision_pneumonia, work_time])
        with open('train/Knn_Numberofk_Precision.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([i, accuracy, precision_normal, precision_pneumonia, work_time])
        i += 1

    print("Data written to 'size_precision.csv'.")


#cross_validation()
# model_numberofk_precision()
# show_k_plot()
# show_size_plot()
k_nearest_neighbors()
