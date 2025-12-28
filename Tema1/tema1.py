import numpy as np, cv2
import matplotlib.pyplot as plt
from scipy.datasets import ascent, face
from scipy.fft import dctn, idctn

def mse(a, b):
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    return np.mean((a - b) ** 2)


X = ascent()
plt.imshow(X, cmap='gray')
plt.show()


Q_down = 10

X_jpeg = X.copy()
X_jpeg = Q_down*np.round(X_jpeg/Q_down);

fig, ax = plt.subplots(1, 2, figsize=(10, 4))

ax[0].imshow(X, cmap='gray')
ax[0].set_title('Original')

ax[1].imshow(X_jpeg, cmap='gray')
ax[1].set_title('Down-sampled')

plt.tight_layout()
plt.savefig('./Tema1/down_sampling.pdf', format='pdf')
plt.show()

# Exercitiul 1

Q_jpeg = [[16, 11, 10, 16, 24, 40, 51, 61],
          [12, 12, 14, 19, 26, 28, 60, 55],
          [14, 13, 16, 24, 40, 57, 69, 56],
          [14, 17, 22, 29, 51, 87, 80, 62],
          [18, 22, 37, 56, 68, 109, 103, 77],
          [24, 35, 55, 64, 81, 104, 113, 92],
          [49, 64, 78, 87, 103, 121, 120, 101],
          [72, 92, 95, 98, 112, 100, 103, 99]]


def dct_cuantization(image, Q_jpeg=Q_jpeg):
    
    h, w = image.shape
    image_encoded = np.zeros_like(image)

    for i in range(0, h, 8):
        for j in range(0, w, 8):
            magic_8_block = image[i:i+8, j:j+8]

            non_magic_dct_8_block = dctn(magic_8_block)
            quantized_8_block_but_not_the_fun_type_like_cuantum = Q_jpeg * np.round(non_magic_dct_8_block / Q_jpeg)
            image_encoded[i:i+8, j:j+8] = quantized_8_block_but_not_the_fun_type_like_cuantum

    return image_encoded

def jpeg_decoding(encoded_image, Q_jpeg=Q_jpeg):
    
    h, w = encoded_image.shape
    image_decoded = np.zeros_like(encoded_image)
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            encoded_8_block = encoded_image[i:i+8, j:j+8]
            non_magic_decoding_since_opposite = idctn(encoded_8_block)
            image_decoded[i:i+8, j:j+8] = non_magic_decoding_since_opposite
    
    return image_decoded


magic_encoding = dct_cuantization(X_jpeg, Q_jpeg)
non_magic_decoding = jpeg_decoding(magic_encoding, Q_jpeg)

mse_gray = mse(X, non_magic_decoding)
print("MSE (JPEG grayscale decoded) =", mse_gray)


fig, ax = plt.subplots(1, 3, figsize=(14, 4))

ax[0].imshow(X_jpeg, cmap='gray')
ax[0].set_title("Original image")

ax[1].imshow(magic_encoding, cmap='gray')
ax[1].set_title("Encoded image through the magic of math beyond me")

ax[2].imshow(non_magic_decoding, cmap='gray')
ax[2].set_title("Decoded")

plt.tight_layout()
plt.savefig('./Tema1/ex1.pdf', format='pdf')
plt.show()

# Exercitiul 2

def rgb_to_ycbcr(img):
    img = img.astype(np.float32)

    Y  =  0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]
    Cb = -0.1687*img[:,:,0] - 0.3313*img[:,:,1] + 0.5*img[:,:,2] + 128
    Cr =  0.5*img[:,:,0] - 0.4187*img[:,:,1] - 0.0813*img[:,:,2] + 128

    return np.stack([Y, Cb, Cr], axis=2)

def ycbcr_to_rgb(img):
    Y, Cb, Cr = img[:,:,0], img[:,:,1], img[:,:,2]

    R = Y + 1.402 * (Cr - 128)
    G = Y - 0.34414 * (Cb - 128) - 0.71414 * (Cr - 128)
    B = Y + 1.772 * (Cb - 128)

    rgb = np.stack([R, G, B], axis=2)
    return np.clip(rgb, 0, 255).astype(np.uint8)

def dct_cuantization_color(image_ycbcr, Q_jpeg=Q_jpeg):
    image_encoded = np.zeros_like(image_ycbcr)

    for c in range(3):   # Y, Cb, Cr
        image_encoded[:,:,c] = dct_cuantization(image_ycbcr[:,:,c], Q_jpeg)

    return image_encoded

def jpeg_decoding_color(encoded_image, Q_jpeg=Q_jpeg):
    image_decoded = np.zeros_like(encoded_image)

    for c in range(3):
        image_decoded[:,:,c] = jpeg_decoding(encoded_image[:,:,c], Q_jpeg)

    return image_decoded

X_rgb = face()
X_ycbr = rgb_to_ycbcr(X_rgb)
magic_encoding_coloured = dct_cuantization_color(X_ycbr, Q_jpeg)
non_magic_decoding_coloured = jpeg_decoding_color(magic_encoding_coloured, Q_jpeg)
X_decoded_rgb = ycbcr_to_rgb(non_magic_decoding_coloured)

mse_rgb = mse(X_rgb, X_decoded_rgb)
print("MSE (RGB total) =", mse_rgb)

magic_encoding_coloured_display = np.clip(magic_encoding_coloured, 0, 255)
X_decoded_rgb_display = np.clip(X_decoded_rgb, 0, 255)

# Convert to uint8 for safe RGB display
magic_encoding_coloured_display = magic_encoding_coloured_display.astype(np.uint8)
X_decoded_rgb_display = X_decoded_rgb_display.astype(np.uint8)

fig, ax = plt.subplots(1, 3, figsize=(14, 4))

ax[0].imshow(X_rgb)
ax[0].set_title("Original image with colour")

ax[1].imshow(magic_encoding_coloured_display)
ax[1].set_title("Encoded image through the magic of math beyond me")

ax[2].imshow(X_decoded_rgb_display)
ax[2].set_title("Decoded")

plt.tight_layout()
plt.savefig('./Tema1/ex2.pdf', format='pdf')
plt.show()

# Exercitiul 3

target_mse = float(input("Introdu pragul MSE dorit: "))

scale = 5
current_mse = 1e9
best_decoded = None

while scale > 0:
    Q_scaled = np.array(Q_jpeg) * scale

    encoded = dct_cuantization(X_jpeg, Q_scaled)
    decoded = jpeg_decoding(encoded)

    current_mse = mse(X_jpeg, decoded)

    print(f"scale={scale:.2f} with MSE={current_mse:.4f}")

    best_decoded = decoded

    if current_mse <= target_mse:
        break

    scale -= 0.01      # compresie in scadere

plt.imshow(best_decoded, cmap='gray')
plt.title(f"JPEG cu MSE aprox: {current_mse:.3f}")
plt.savefig('./Tema1/ex3.pdf', format='pdf')
plt.show()


# Exercitiul 4

cap = cv2.VideoCapture('./Tema1/video.mp4')

if not cap.isOpened():
    raise RuntimeError("Error")

fps   = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

video_compressed   = cv2.VideoWriter('./Tema1/video_compressed.mp4', fourcc, fps, (width, height))
video_decompressed = cv2.VideoWriter('./Tema1/video_decompressed.mp4', fourcc, fps, (width, height))

mse_video = []

while True:
    ret, frame_bgr = cap.read()
    if not ret:
        break

    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    ycbcr_frame = rgb_to_ycbcr(frame)
    compressed_frame = dct_cuantization_color(ycbcr_frame, Q_jpeg)

    compressed_rgb = ycbcr_to_rgb(compressed_frame)
    compressed_bgr = cv2.cvtColor(compressed_rgb, cv2.COLOR_RGB2BGR)
    video_compressed.write(compressed_bgr)

    frame_decompressed = jpeg_decoding_color(compressed_frame, Q_jpeg)
    frame_decompressed_rgb = ycbcr_to_rgb(frame_decompressed)
    frame_decompressed_bgr = cv2.cvtColor(frame_decompressed_rgb, cv2.COLOR_RGB2BGR)
    video_decompressed.write(frame_decompressed_bgr)

    mse_frame = mse(frame, frame_decompressed_rgb)
    mse_video.append(mse_frame)

cap.release()
video_compressed.release()
video_decompressed.release()

avg_mse = sum(mse_video) / len(mse_video)
print("Average MSE for video compression loss:", avg_mse)
