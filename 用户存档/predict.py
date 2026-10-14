# -*- coding: utf-8 -*-
import cv2

CAMERA_ID = 0
USE_STEREO = True
FRAME_WIDTH = 1280
FRAME_HEIGHT = 480
SAVE_PATH = "stereo_spliced.jpg"


def main():
    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)

    if USE_STEREO:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        # 双目高分辨率经常需要 MJPG
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    else:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("[ERROR] 无法打开摄像头")
        return

    print("[INFO] 按 q 退出，按 s 保存当前拼接图")

    first = True
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] 读取帧失败")
            break

        # 只打印一次分辨率，确认相机输出格式
        if first:
            print("[INFO] frame.shape =", frame.shape)
            first = False

        # 注意：这里不要裁剪左半，直接显示原始帧
        # 如果相机输出本来就是左右拼接，frame 就是拼接图
        cv2.imshow("Stereo Spliced Output", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(SAVE_PATH, frame)
            print(f"[INFO] 已保存: {SAVE_PATH}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()