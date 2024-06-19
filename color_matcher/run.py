import cv2
import numpy as np


TEXT_FORMATS = (cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0xFF, 0), 1)
TEXT_SPACING = 18
WND_NAME = 'Color Matcher'
SAMPLE_COUNT = 20
WND_SIZE = (1280, 720)

# Create an array of shape (16^3, 3) with each channel ranging from 0 to 255 (step of 16)
l_values = np.arange(0, 256, 16)
a_values = np.arange(0, 256, 16)
b_values = np.arange(0, 256, 16)
L, A, B = np.meshgrid(l_values, a_values, b_values)
colors_array = np.stack((L, A, B), axis=-1).astype(np.uint8)
print(colors_array.shape)
REF_POINTS = cv2.cvtColor(colors_array.reshape(-1, 16, 3), cv2.COLOR_LAB2BGR).reshape(-1, 3)

CAM_PROPS = (
    (cv2.CAP_PROP_FPS, 'fps'),
    (cv2.CAP_PROP_BRIGHTNESS, 'brig'),
    (cv2.CAP_PROP_CONTRAST, 'contra'),
    (cv2.CAP_PROP_SATURATION, 'sat'),
    (cv2.CAP_PROP_HUE, 'hue'),
    (cv2.CAP_PROP_GAIN, 'g'),
    (cv2.CAP_PROP_EXPOSURE, 'e'),
    (cv2.CAP_PROP_AUTO_EXPOSURE, 'ae'),
    (cv2.CAP_PROP_GAMMA, 'gamma'),
    (cv2.CAP_PROP_ISO_SPEED, 'iso'),
    (cv2.CAP_PROP_AUTO_WB, 'auto_wb'),
    (cv2.CAP_PROP_WB_TEMPERATURE, 'wb_temp'),
    (cv2.CAP_PROP_BITRATE, 'bitrate'),
)

cap = cv2.VideoCapture()
# The device number might be 0 or 1 depending on the device and the webcam
cap.open(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.0)
cap.set(cv2.CAP_PROP_GAIN, 0.0)
cap.set(cv2.CAP_PROP_EXPOSURE, 0.0)


cv2.namedWindow(WND_NAME, cv2.WINDOW_KEEPRATIO)
neutral_ = np.ndarray((1, 1, 3), np.uint8)
neutral_[:, :] = (255 * 0.18, 128, 128)
neutral_ = cv2.cvtColor(neutral_, cv2.COLOR_LAB2BGR).reshape(3)
while(True):
    ret, frame = cap.read()
    clr_measured_ = np.array(cv2.mean(frame)).astype(np.uint8)[:3]
    clr_measured_ = cv2.cvtColor(clr_measured_.reshape(1, 1, -1), cv2.COLOR_BGR2LAB).reshape(-1, 3)

    h, w, *_ = frame.shape
    frame[h//2:, :] = neutral_

    cv2.putText(frame, f'Hold the measuring device to the display and wait for eposure and',
                (12, 1*TEXT_SPACING+8), *TEXT_FORMATS)
    cv2.putText(frame, f'white balance to stabilize. Press any key to continue...',
                (12, 2*TEXT_SPACING+8), *TEXT_FORMATS)
    cv2.putText(frame, f'measured={clr_measured_.ravel()}',
                (12, 4*TEXT_SPACING+8), *TEXT_FORMATS)
    cv2.imshow(WND_NAME, frame)
    cv2.resizeWindow(WND_NAME, *WND_SIZE)

    k_ = cv2.waitKey(1) & 0xFF
    if k_ > 0 and k_ < 0xFF:
        break

cindex = 0
samples = 0
while(True):
    ret, frame = cap.read()

    clr_target_ = REF_POINTS[cindex]
    clr_measured_ = np.array(cv2.mean(frame)).astype(np.uint8)[:3]
    h, w, *_ = frame.shape
    frame[h//2:, :] = clr_target_
    cv2.putText(frame, f'target={clr_target_}',     (12, 1*TEXT_SPACING+8), *TEXT_FORMATS)
    cv2.putText(frame, f'measured={clr_measured_}',   (12, 2*TEXT_SPACING+8), *TEXT_FORMATS)

    txt_pos_ = 3*TEXT_SPACING+8
    for pid_, pname_ in CAM_PROPS:
        v_ = cap.get(pid_)
        cv2.putText(frame, f'{pname_}={v_}',   (12, txt_pos_), *TEXT_FORMATS)
        txt_pos_ += TEXT_SPACING

    cv2.imshow(WND_NAME, frame)
    cv2.resizeWindow(WND_NAME, *WND_SIZE)

    k_ = cv2.waitKey(1) & 0xFF
    if k_  == 27:
        break
    elif k_ > 0 and k_ < 0xFF:
        print(k_)

    samples += 1
    if samples > SAMPLE_COUNT:
        samples = 0
        cindex += 1
        if cindex > len(REF_POINTS): cindex = 0

cap.release()
cv2.destroyAllWindows()