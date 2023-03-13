import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.signal as signal

if __name__ == '__main__':

    img = cv2.imread('g.jpeg', cv2.IMREAD_GRAYSCALE)
    h1 = np.array([[1 / 6, 1 / 6, 1 / 6], [0, 0, 0], [1 / 6, 1 / 6, 1 / 6]])
    h2 = np.array([[1], [-1], [0]])
    h3 = np.array([[1, 1], [-1, -1], [2, 2]])

    output1 = signal.convolve2d(img, h1, mode='same')
    output2 = signal.convolve2d(output1, h2, mode='same')
    output3 = signal.convolve2d(output2, h3, mode='same')


    # Display the results
    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')

    plt.subplot(2, 2, 2)
    plt.imshow(output1, cmap='gray')
    plt.title('Convolution with h1')

    plt.subplot(2, 2, 3)
    plt.imshow(output2, cmap='gray')
    plt.title('Convolution with h1 & h2')

    plt.subplot(2, 2, 4)
    plt.imshow(output3, cmap='gray')
    plt.title('Convolution with h1 & h2 & h3')

    plt.show()


