import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image


def display_image(image):
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return


def calculate_histogram(image, range=256):
    histogram = [0] * range
    for row in image:
        for pixel in row:
            histogram[pixel] += 1
    return histogram


def plot_histogram(image, path, range=256):
    # histogram = cv2.calcHist([image], [0], None, [range], [0, range])
    histogram = calculate_histogram(image, range)

    plt.plot(histogram, color="black")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.title("Histogram")
    plt.savefig(f"{path}/histogram.png")
    plt.close()

    normalized_histogram = histogram / np.sum(histogram)

    plt.plot(normalized_histogram, color="black")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Normalized Frequency")
    plt.title("Normalized Histogram")
    plt.savefig(f"{path}/normalized_histogram.png")
    plt.close()


def histogram_equalization(image, r=256):
    histogram = calculate_histogram(image, range=r)

    # Compute cumulative distribution function (CDF)
    cdf = [sum(histogram[: i + 1]) for i in range(len(histogram))]

    # Compute histogram equalization transformation
    total_pixels = len(image) * len(image[0])
    equalized_image = [[0] * len(image[0]) for _ in range(len(image))]
    for i in range(len(image)):
        for j in range(len(image[0])):
            intensity = image[i][j]
            equalized_intensity = int(round((cdf[intensity] / total_pixels) * 255))
            equalized_image[i][j] = equalized_intensity

    equalized_image = np.array(equalized_image, dtype=np.uint8)
    return equalized_image


if __name__ == "__main__":
    # PART ONE
    image_path = "./plants.jpg"
    image = load_image(image_path)

    plot_histogram(image, path="./hist_1")

    # equalized_image = cv2.equalizeHist(image)
    equalized_image = histogram_equalization(image)
    cv2.imwrite("eq_image_1.jpg", equalized_image)

    # PART TWO
    array = np.array(
        [
            [1, 2, 2, 6, 5, 5, 6, 7, 3, 7],
            [1, 5, 6, 1, 2, 2, 2, 2, 7, 7],
            [2, 1, 6, 4, 2, 3, 4, 2, 4, 3],
            [3, 5, 3, 4, 4, 6, 6, 6, 6, 6],
            [4, 6, 7, 3, 4, 6, 5, 4, 5, 6],
            [6, 3, 2, 1, 7, 7, 4, 5, 6, 7],
            [2, 1, 6, 4, 2, 3, 4, 2, 4, 3],
            [1, 2, 2, 6, 5, 5, 6, 7, 3, 7],
            [4, 6, 7, 3, 4, 6, 5, 4, 5, 6],
            [6, 3, 2, 1, 7, 7, 4, 5, 6, 7],
        ]
    )
    image2 = np.uint8(array)
    cv2.imwrite("org_image.jpg", image2)

    # equalized_image = cv2.equalizeHist(image2)
    equalized_image2 = histogram_equalization(image2, r=8)
    cv2.imwrite("eq_image_2.jpg", equalized_image2)

    plot_histogram(image2, path="./hist_2/before", range=8)
    plot_histogram(equalized_image2, path="./hist_2/after")
