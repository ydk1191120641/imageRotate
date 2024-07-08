# 代码示例

import os
import sys
import glob
import cv2
import numpy as np
import onnxruntime

def process(src_image_dir, output_filename):
    current_path = os.path.dirname(__file__)
    image_paths = glob.glob(os.path.join(src_image_dir, '*.jpg'))
    sess = onnxruntime.InferenceSession(path_or_bytes='onnx.onnx', providers=['CPUExecutionProvider'])
    norm_mean=np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    norm_std=np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))

    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    with open(os.path.join(current_path, output_filename), 'w') as f:
        for image_path in image_paths:
            image_name = image_path.split('/')[-1]
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_h, img_w = img.shape[:2]
            percent = 256 / min(img_w, img_h)
            w = int(round(img_w * percent))
            h = int(round(img_h * percent))
            img = cv2.resize(img,(w,h))
            w_start = (w - 224) // 2
            h_start = (h - 224) // 2
            w_end = w_start + 224
            h_end = h_start + 224
            img = img[h_start:h_end, w_start:w_end, :]
            img = np.transpose(img,[2,0,1])/255
            img = (img-norm_mean)/norm_std
            img = np.expand_dims(img.astype("float32"), axis=0)
            outputs = sess.run([output_name], {input_name: img})
            pred_label = np.argmax(outputs)
            f.write(f'{image_name} {pred_label}\n')
        f.close()


if __name__ == "__main__":
    # assert len(sys.argv) == 3
    #
    # src_image_dir = sys.argv[1]
    # output_filename = sys.argv[2]
    src_image_dir = './'
    output_filename = 'result_onnxruntime.txt'
    process(src_image_dir, output_filename)