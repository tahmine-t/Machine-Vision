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
    print(histogram)

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
    print(cdf)
    print("------------------------")

    # Compute histogram equalization transformation
    total_pixels = len(image) * len(image[0])
    print(total_pixels)
    equalized_image = [[0] * len(image[0]) for _ in range(len(image))]
    for i in range(len(image)):
        for j in range(len(image[0])):
            intensity = image[i][j]
            equalized_intensity = int(round((cdf[intensity] / total_pixels) * 255))
            # print(equalized_intensity)
            equalized_image[i][j] = equalized_intensity

    equalized_image = np.array(equalized_image, dtype=np.uint8)
    print("------------------------")
    print(equalized_image)
    return equalized_image


if __name__ == "__main__":
    # PART TWO
    array = np.array(
        [
            [0, 0, 1, 2, 2, 2, 3, 3],
            [0, 0, 1, 2, 2, 3, 3, 3],
            [1, 1, 1, 2, 2, 3, 3, 3],
            [12, 12, 11, 11, 10, 10, 4, 4],
            [6, 12, 15, 7, 9, 4, 2, 2],
            [13, 12, 14, 3, 4, 9, 1, 1],
            [15, 10, 14, 4, 3, 9, 9, 1],
            [3, 5, 4, 4, 3, 2, 15, 0],
        ]
    )
    image2 = np.uint8(array)
    cv2.imwrite("org_image.jpg", image2)

    # equalized_image2 = cv2.equalizeHist(image2)
    equalized_image2 = histogram_equalization(image2, r=16)
    cv2.imwrite("quizz.jpg", equalized_image2)

    plot_histogram(image2, path="./now", range=16)
