import cv2
from matplotlib import pyplot as plt
import numpy as np


def load_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image


def display_image(image):
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return


def gamma_correction(image, gamma):
    # corrected_image = np.power(image / 255.0, 1 / gamma)
    corrected_image = np.power(image / 255.0, gamma)
    corrected_image = np.uint8(corrected_image * 255)
    return corrected_image


def plot_histogram(image, path, range=256):
    histogram = cv2.calcHist([image], [0], None, [range], [0, range])

    plt.plot(histogram, color="black")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.title("Histogram")
    plt.savefig(path)
    plt.close()


if __name__ == "__main__":
    # PART ONE
    image_path = "./rice.png"
    image = load_image(image_path)

    gamma = [0.1, 0.2, 0.8, 1.3, 1.5, 3, 5]

    for g in gamma:
        result = gamma_correction(image, g)
        cv2.imwrite(f"./results/rice_gamma_{g}.png", result)
        print(g, np.mean(result))
        # 0.1: 241.18221184248347
        # 0.2: 228.6210074734119
        # 0.8: 165.80338603046852
        # 1.3: 127.01247484909457
        # 1.5: 114.07638401839608
        # 3: 51.38579764300086
        # 5: 17.902375682667433

    # PART TWO
    image_path = "./plants.jpg"
    image = load_image(image_path)

    for g in gamma:
        result = gamma_correction(image, g)
        cv2.imwrite(f"./results/plants_gamma_{g}.png", result)

    gamma = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    for g in gamma:
        result = gamma_correction(image, g)
        cv2.imwrite(f"./results/plants_gamma_{g}.png", result)
        print(g, np.mean(result))
        # 0.1: 209.3303697630742
        # 0.2: 175.04046843544427
        # 0.3: 148.7740368284103
        # 0.4: 128.41293958030982
        # 0.5: 112.38959614630065
        # 0.6: 99.84492750001857

    # EVALUATE
    gamma = [0.1, 0.2, 0.8, 1.3, 1.5, 3, 5]
    for g in gamma:
        image_path = f"./results/rice_gamma_{g}.png"
        image = load_image(image_path)
        plot_histogram(image, path=f"./hist/rice_gamma_{g}.png")

    gamma = [0.1, 0.2, 0.3, 0.4, 0.5, 0.8]
    for g in gamma:
        image_path = f"./results/plants_gamma_{g}.png"
        image = load_image(image_path)
        plot_histogram(image, path=f"./hist/plants_gamma_{g}.png")
