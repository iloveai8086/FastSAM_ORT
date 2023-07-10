import onnxruntime
import numpy as np
from numpy import ndarray
from typing import List, Tuple, Union
import cv2
import argparse
import ast
import random
import time

random.seed(0)

# detection model classes
CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
           'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
           'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
           'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
           'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
           'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
           'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
           'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
           'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
           'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
           'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
           'scissors', 'teddy bear', 'hair drier', 'toothbrush')

# colors for per classes
COLORS = {
    cls: [random.randint(0, 255) for _ in range(3)]
    for i, cls in enumerate(CLASSES)
}

# colors for segment masks
MASK_COLORS = np.array([(255, 56, 56), (255, 157, 151), (255, 112, 31),
                        (255, 178, 29), (207, 210, 49), (72, 249, 10),
                        (146, 204, 23), (61, 219, 134), (26, 147, 52),
                        (0, 212, 187), (44, 153, 168), (0, 194, 255),
                        (52, 69, 147), (100, 115, 255), (0, 24, 236),
                        (132, 56, 255), (82, 0, 133), (203, 56, 255),
                        (255, 149, 200), (255, 55, 199)],
                       dtype=np.float32) / 255.

# alpha for segment masks
ALPHA = 0.5

def letterbox(im: ndarray,
              new_shape: Union[Tuple, List] = (640, 640),
              color: Union[Tuple, List] = (114, 114, 114)) \
        -> Tuple[ndarray, float, Tuple[float, float]]:
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    # new_shape: [width, height]

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[1], new_shape[1] / shape[0])
    # Compute padding [width, height]
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[0] - new_unpad[0], new_shape[1] - new_unpad[
        1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im,
                            top,
                            bottom,
                            left,
                            right,
                            cv2.BORDER_CONSTANT,
                            value=color)  # add border
    return im, r, (dw, dh)


def blob(im: ndarray, return_seg: bool = False) -> Union[ndarray, Tuple]:
    if return_seg:
        seg = im.astype(np.float32) / 255
    im = im.transpose([2, 0, 1])
    im = im[np.newaxis, ...]
    im = np.ascontiguousarray(im).astype(np.float32) / 255
    if return_seg:
        return im, seg
    else:
        return im

def crop_mask(masks: ndarray, bboxes: ndarray) -> ndarray:
    n, h, w = masks.shape
    x1, y1, x2, y2 = np.split(bboxes[:, :, None], [1, 2, 3],
                              1)  # x1 shape(1,1,n)
    r = np.arange(w, dtype=x1.dtype)[None, None, :]  # rows shape(1,w,1)
    c = np.arange(h, dtype=x1.dtype)[None, :, None]  # cols shape(h,1,1)

    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def seg_postprocess(
        data: Tuple[ndarray],
        shape: Union[Tuple, List],
        conf_thres: float = 0.25,
        iou_thres: float = 0.65) \
        -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    assert len(data) == 2
    h, w = shape[0] // 4, shape[1] // 4  # 4x downsampling
    outputs, proto = (i[0] for i in data)
    bboxes, scores, labels, maskconf = np.split(outputs, [4, 5, 6], 1)
    scores, labels = scores.squeeze(), labels.squeeze()
    idx = scores > conf_thres
    bboxes, scores, labels, maskconf = \
        bboxes[idx], scores[idx], labels[idx], maskconf[idx]
    cvbboxes = np.concatenate([bboxes[:, :2], bboxes[:, 2:] - bboxes[:, :2]],
                              1)
    labels = labels.astype(np.int32)
    v0, v1 = map(int, (cv2.__version__).split('.')[:2])
    assert v0 == 4, 'OpenCV version is wrong'
    if v1 > 6:
        idx = cv2.dnn.NMSBoxesBatched(cvbboxes, scores, labels, conf_thres,
                                      iou_thres)
    else:
        idx = cv2.dnn.NMSBoxes(cvbboxes, scores, conf_thres, iou_thres)
    bboxes, scores, labels, maskconf = \
        bboxes[idx], scores[idx], labels[idx], maskconf[idx]
    masks = sigmoid(maskconf @ proto).reshape(-1, h, w)
    masks = crop_mask(masks, bboxes / 4.)
    masks = masks.transpose([1, 2, 0])
    masks = cv2.resize(masks, (shape[1], shape[0]),
                       interpolation=cv2.INTER_LINEAR)
    masks = masks.transpose(2, 0, 1)
    masks = np.ascontiguousarray((masks > 0.5)[..., None], dtype=np.float32)
    return bboxes, scores, labels, masks


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./weights/FastSAM.pt', help='model')
    parser.add_argument('--img_path', type=str, default='./images/1634370875501404.jpg', help='path to image file')
    parser.add_argument('--imgsz', type=int, default=1024, help='image size')
    parser.add_argument('--iou', type=float, default=0.9, help='iou threshold for filtering the annotations')
    parser.add_argument('--text_prompt', type=str, default=None, help='use text prompt eg: "a dog"')
    parser.add_argument('--conf', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--output', type=str, default='./output/', help='image save path')
    parser.add_argument('--randomcolor', type=bool, default=True, help='mask random color')
    parser.add_argument('--point_prompt', type=str, default="[[0,0]]", help='[[x1,y1],[x2,y2]]')
    parser.add_argument('--point_label', type=str, default="[0]", help='[1,0] 0:background, 1:foreground')
    parser.add_argument('--box_prompt', type=str, default="[0,0,0,0]", help='[x,y,w,h]')
    parser.add_argument('--better_quality', type=str, default=False, help='better quality using morphologyEx')
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # parser.add_argument('--device', type=str, default=device, help="cuda:[0,1,2,3,4] or cpu")
    parser.add_argument('--retina', type=bool, default=True, help='draw high-resolution segmentation masks')
    parser.add_argument('--withContours', type=bool, default=False, help='draw the edges of the masks')
    parser.add_argument('--show',
                        action='store_true',
                        help='Show the detection results')
    return parser.parse_args()




if __name__ == '__main__':
    args = parse_args()
    args.point_prompt = ast.literal_eval(args.point_prompt)
    args.box_prompt = ast.literal_eval(args.box_prompt)
    args.point_label = ast.literal_eval(args.point_label)
    print(args.point_prompt)
    print(args.box_prompt)
    print(args.point_label)
    session = onnxruntime.InferenceSession("weights/FastSAM-x.onnx", providers=[
        "CUDAExecutionProvider"])  # CPU 后端cpu cuda trt的,可选“CPUExecutionProvider”
    print(session)
    W = 1024
    H = 1024
    bgr = cv2.imread(args.img_path)
    print(bgr.shape)
    draw = bgr.copy()
    bgr, ratio, dwdh = letterbox(bgr, (W, H))
    dw, dh = int(dwdh[0]), int(dwdh[1])
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    tensor, seg_img = blob(rgb, return_seg=True)
    dwdh = np.array(dwdh * 2, dtype=np.float32)
    tensor = np.ascontiguousarray(tensor)
    print(tensor.shape)
    t1 = time.time()
    for i in range(10):
        pred = session.run(["outputs", "proto"], {"images": tensor})
    t2 = time.time()
    print("infer costs:", (t2 - t1) / 10.0)
    print(pred[0].shape)
    print(pred[1].shape)

    seg_img = seg_img[dh:H - dh, dw:W - dw, [2, 1, 0]]
    bboxes, scores, labels, masks = seg_postprocess(
        pred, bgr.shape[:2], 0.4, 0.9)
    masks = masks[:, dh:H - dh, dw:W - dw, :]
    mask_colors = MASK_COLORS[labels % len(MASK_COLORS)]
    mask_colors = mask_colors.reshape(-1, 1, 1, 3) * ALPHA
    mask_colors = masks @ mask_colors
    inv_alph_masks = (1 - masks * 0.5).cumprod(0)
    mcs = (mask_colors * inv_alph_masks).sum(0) * 2
    seg_img = (seg_img * inv_alph_masks[-1] + mcs) * 255
    draw = cv2.resize(seg_img.astype(np.uint8), draw.shape[:2][::-1])

    bboxes -= dwdh
    bboxes /= ratio

    for (bbox, score, label) in zip(bboxes, scores, labels):
        bbox = bbox.round().astype(np.int32).tolist()
        cls_id = int(label)
        cls = CLASSES[cls_id]
        color = COLORS[cls]
        cv2.rectangle(draw, bbox[:2], bbox[2:], color, 2)
        cv2.putText(draw,
                    f'{cls}:{score:.3f}', (bbox[0], bbox[1] - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, [225, 255, 255],
                    thickness=2)
    if args.show:
        cv2.imshow('result', draw)
        cv2.waitKey(0)
    else:
        cv2.imwrite("res.jpg", draw)

