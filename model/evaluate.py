import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import cv2


def print_plot_keras_metrics(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Disease Training and validation accuracy (Dev set)')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Disease Training and validation loss (Dev Set)')
    plt.legend()

    plt.show()


# def eval_model(model, test_labels, test_data):
   # validation of small network
    # test_loss, test_acc = model.evaluate(
    #     x=test_data, y=test_labels, batch_size=None, verbose=1)
    # print('test_accuracy for test set:', test_acc)

    # # print test images and results using matplotlib

    # CAD = {0: "Normal", 1: "Stenotic"}
    # Anatomy = {0: "Left Main or LAD/LCx", 1: "LAD", 2: "LCx", 3: "RCA"}
    # prediction = model.predict(test_data)

    # counter_stenosis = 0
    # counter_anatomy = 0

    # for i in range(test_data_np.shape[0]):
    #     print("test images #", i)
    #     plt.imshow(test_data[i], cmap='gray', interpolation='bicubic')
    #     plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    #     plt.show()

    #     prediction_row = prediction[i]
    #     print(prediction_row)

    #     cad = int(prediction_row[0] > 0.5)  # this is specific for one shot
    #     # this is specific for one shot
    #     anatomy = np.argmax(prediction_row[1:5])

    #     if (cad == int(test_label_np[i])):
    #         counter_stenosis = counter_stenosis + 1
    #     if (anatomy == int(test_label_np1[i])):
    #         counter_anatomy = counter_anatomy + 1

    #     print("deciphered:", cad, anatomy)
    #     print("Predicted", CAD[cad], Anatomy[anatomy])
    #     print("Actual", CAD[int(test_label_np[i])],
    #           Anatomy[int(test_label_np1[i])])
    #     print("--------")
    # correct_stenosis_ratio = counter_stenosis / test_data_np.shape[0]
    # correct_anatomy_ratio = counter_anatomy / test_data_np.shape[0]
    # print("stenosis correct ratio:", correct_stenosis_ratio,
    #       "; anatomy correct ratio:", correct_anatomy_ratio
