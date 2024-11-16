import cv2
import numpy as np


def load_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image


def bcd_to_bin(image):
    binary_array = np.array(
        [[np.binary_repr(num, width=8) for num in row] for row in image]
    )
    return binary_array


def seperate_binary_pages(binary_arr):
    row = binary_arr.shape[0]
    col = binary_arr.shape[1]
    binary_pages = np.empty((8, row, col))
    for r in range(row):
        for c in range(col):
            for x in range(8):
                binary_pages[x][r][c] = binary_arr[r][c][x]

    # binary_pages = np.split(binary_arr, 8, axis=2)
    return binary_pages


def seperate_pages(image):
    row = image.shape[0]
    col = image.shape[1]
    pages = np.empty((8, row, col))
    for x in range(8):
        for r in range(row):
            for c in range(col):
                value = image[r][c]
                for j in reversed(range(8)):
                    if j > x:
                        if value >= 2**j:
                            value -= 2**j
                limit = 2**x
                pages[x][r][c] = 0 if value < limit else limit

    return pages


def zero_out(image, page_num):
    row = image.shape[0]
    col = image.shape[1]
    for r in range(row):
        for c in range(col):
            value = image[r][c]
            for j in reversed(range(8)):
                if j > page_num:
                    if value >= 2**j:
                        value -= 2**j
            limit = 2**page_num
            if value >= limit:
                image[r][c] -= limit

    return image


def display_image(image):
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return


if __name__ == "__main__":
    image_path = "./image.jpg"

    image = load_image(image_path)
    # binary_array = bcd_to_bin(image)
    # binary_pages = seperate_binary_pages(binary_array)

    pages = seperate_pages(image)
    print(pages[7])

    # # PART A
    # for page in pages:
    #     display_image(page)

    # # PART B
    # for i in range(8):
    #     out = zero_out(image, i)
    #     display_image(out)
