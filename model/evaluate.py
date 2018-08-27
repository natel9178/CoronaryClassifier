import numpy as np
import cv2


def print_plot_keras_metrics(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    # plt.plot(epochs, acc, 'bo', label='Training acc')
    # plt.plot(epochs, val_acc, 'b', label='Validation acc')
    # plt.title('Disease Training and validation accuracy (Dev set)')
    # plt.legend()

    # plt.figure()

    # plt.plot(epochs, loss, 'bo', label='Training loss')
    # plt.plot(epochs, val_loss, 'b', label='Validation loss')
    # plt.title('Disease Training and validation loss (Dev Set)')
    # plt.legend()

    # plt.show()


def eval_model(model, dev_labels, dev_data, dev_label_np, dev_data_np, dev_label_np1, dev_data_np1):
    dev_loss, dev_acc = model.evaluate(
        x=dev_data, y=dev_labels, batch_size=None, verbose=1)
    print('dev_accuracy for dev set:', dev_acc)

    # print test images and results using matplotlib

    CAD = {0: "Normal", 1: "Stenotic"}
    Anatomy = {0: "Left Main or LAD/LCx", 1: "LAD", 2: "LCx", 3: "RCA"}
    prediction = model.predict(dev_data)

    counter_stenosis = 0
    counter_anatomy = 0

    for i in range(dev_data_np.shape[0]):
        print("test images #", i)
        # plt.imshow(test_data[i], cmap='gray', interpolation='bicubic')
        # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        # plt.show()

        prediction_row = prediction[i]
        print(prediction_row)

        cad = int(prediction_row[0] > 0.5)  # this is specific for one shot
        # this is specific for one shot
        anatomy = np.argmax(prediction_row[1:5])

        if (cad == int(dev_label_np[i])):
            counter_stenosis = counter_stenosis + 1
        if (anatomy == int(dev_label_np1[i])):
            counter_anatomy = counter_anatomy + 1

        print("deciphered:", cad, anatomy)
        print("Predicted", CAD[cad], Anatomy[anatomy])
        print("Actual", CAD[int(dev_label_np[i])],
              Anatomy[int(dev_label_np1[i])])
        print("--------")
    correct_stenosis_ratio = counter_stenosis / dev_data_np.shape[0]
    correct_anatomy_ratio = counter_anatomy / dev_data_np.shape[0]
    print("stenosis correct ratio:", correct_stenosis_ratio,
          "; anatomy correct ratio:", correct_anatomy_ratio)
