import torch
import cv2
import bbox_visualizer as bbv
import numpy as np
from collections import defaultdict
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tool_usage ={"Empty" : "T0",
            "Needle_driver": "T1",
            "Forceps": "T2",
            "Scissors":"T3"}
names = {int(v[-1]):k for k,v in tool_usage.items()}

def label_smoothing(preds):
    kernel = np.array(list(range(1, 10)) + list([10]) + list(range(1, 10))[::-1])
    k = len(kernel)
    dim = 4
    assert k % 2 == 1

    def to_one_hots(preds, dim):
        one_hots = []
        for p in preds:
            v = np.zeros(dim)
            if p is not None:
                v[p] = 1
            one_hots.append(v)
        return one_hots

    one_hots = to_one_hots(preds, dim)
    padding = np.array([np.zeros(dim) for _ in range(int(k / 2))])
    padded_preds = np.concatenate((padding, one_hots, padding), axis=0)
    output = []
    for i in range(len(padded_preds) - k + 1):
        weighted = padded_preds[i:i + k, :].T @ kernel
        output.append(np.argmax(weighted))
    return output



def get_preds_not_smooth(model, vid_path, cutoff = None):
    right_labels, left_labels = [], []
    cap = cv2.VideoCapture(vid_path)
    i = 0
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            boxes = model(frame).pred[0]
            labels = boxes[:, -1].int().tolist()
            confs = boxes[:, -2].tolist()

            confs_sum_right = defaultdict(lambda: 0)
            confs_sum_left = defaultdict(lambda: 0)
            for l, c in zip(labels, confs):
                if l % 2 == 0:
                    confs_sum_right[l]+=c
                else:
                    confs_sum_left[l]+=c
            right_pred = max((p, l) for l, p in confs_sum_right.items())[1] if len(confs_sum_right) else None
            left_pred = max((p, l) for l, p in confs_sum_left.items())[1] if len(confs_sum_left) else None
            
            right_pred = right_pred // 2 if right_pred is not None else None
            left_pred = left_pred // 2 if left_pred is not None else None

            if right_pred in [0, 3]:
                right_pred = 3 - right_pred
            if left_pred in [0, 3]:
                left_pred = 3 - left_pred 

            right_labels.append(right_pred)
            left_labels.append(left_pred)
            i += 1

            if cutoff is not None and i == cutoff:
                break
        else:
            break
    # When everything done, release the video capture object
    cap.release()
    cv2.destroyAllWindows()
    return left_labels, right_labels

def save_vid(model, vid_path, out_path, left_labels_smooth, right_labels_smooth, cutoff = None):
    cap = cv2.VideoCapture(vid_path)
    size = (int(cap.get(3)), int(cap.get(4)))
    out = cv2.VideoWriter(out_path,cv2.VideoWriter_fourcc(*'MP4V'), 30, size)
    frames = []
    i = 0
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            boxes = model(frame).pred[0]
            preds = boxes[:2, -1].int().tolist()
            confs = boxes[:2, -2].tolist()
            confs = [round(c, 2) for c in confs]
            boxes = boxes[:2, :-2].int().tolist()
            if len(boxes) == 1:
                if preds[0] % 2 == 0:
                    preds = [str(confs[0])+" R_"+names[right_labels_smooth[i]]]
                else:
                    preds = [str(confs[0])+" L_"+names[left_labels_smooth[i]]]
            elif len(boxes) and preds[0] % 2 == 0:
                preds = [str(confs[0])+" R_"+names[right_labels_smooth[i]], str(confs[1])+ " L_"+names[left_labels_smooth[i]]]

            elif len(boxes):
                preds = [str(confs[0])+" L_"+names[left_labels_smooth[i]], str(confs[1])+" R_"+names[right_labels_smooth[i]]]

            if len(boxes):
                frame = bbv.draw_multiple_rectangles(frame, boxes, bbox_color=(255,0,0), thickness=2)
                frame = add_multiple_labels(frame, preds, boxes,text_bg_color=(255,0,0))
            out.write(frame)
            i += 1
            if cutoff is not None and i == cutoff:
                break
        else:
            break
        
    # When everything done, release the video capture object
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def add_label(img,
              label,
              bbox,
              draw_bg=True,
              text_bg_color=(255, 255, 255),
              text_color=(0, 0, 0),
              top=True):

    text_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 1)[0][0]

    if top:
        label_bg = [bbox[0], bbox[1], bbox[0] + text_width, bbox[1] - 30]
        if draw_bg:
            cv2.rectangle(img, (label_bg[0], label_bg[1]),
                          (label_bg[2] + 5, label_bg[3]), text_bg_color, -1)
        cv2.putText(img, label, (bbox[0] + 5, bbox[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, text_color, 1)

    else:
        label_bg = [bbox[0], bbox[1], bbox[0] + text_width, bbox[1] + 30]
        if draw_bg:
            cv2.rectangle(img, (label_bg[0], label_bg[1]),
                          (label_bg[2] + 5, label_bg[3]), text_bg_color, -1)
        cv2.putText(img, label, (bbox[0] + 5, bbox[1] - 5 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 0.5)

    return img

def add_multiple_labels(img,
                        labels,
                        bboxes,
                        draw_bg=True,
                        text_bg_color=(255, 255, 255),
                        text_color=(0, 0, 0),
                        top=True):

    for label, bbox in zip(labels, bboxes):
        img = add_label(img, label, bbox, draw_bg, text_bg_color, text_color,
                        top)

    return img

def main(vid_path):
    print(vid_path)
    true_labels_right = []
    true_labels_left = []
    with open(f"/home/student/HW1/datasets/HW1_dataset/tool_usage/tools_left/{vid_path[:-4].split('/')[-1]}.txt") as f:
        for line in f.readlines():
            start, fin, lab = line.split()
            lab = int(lab[-1])
            start, fin = int(start), int(fin)
            true_labels_left.extend([lab]*(fin - start + 1))

    with open(f"/home/student/HW1/datasets/HW1_dataset/tool_usage/tools_right/{vid_path[:-4].split('/')[-1]}.txt") as f:
        for line in f.readlines():
            start, fin, lab = line.split()
            lab = int(lab[-1])
            start, fin = int(start), int(fin)
            true_labels_right.extend([lab]*(fin - start + 1))

    model = torch.hub.load('yolov5', 'custom', path=fr"/home/student/HW1/yolov5/runs/train/exp8/weights/best.pt", source='local') 
    model = model.to(device)
    model.eval()

    left_labels, right_labels = get_preds_not_smooth(model, vid_path, cutoff=None)
    left_labels_smooth = label_smoothing(left_labels)
    right_labels_smooth = label_smoothing(right_labels)
    out_path = vid_path[:-4].split('/')[-1]+"_pred.mp4"
    save_vid(model, vid_path, out_path, left_labels_smooth, right_labels_smooth, cutoff=None)
    with open(vid_path[:-4].split('/')[-1]+ "_lists", "wb") as fp:
        pickle.dump([left_labels_smooth, right_labels_smooth], fp)

    1 == 1
   

if __name__ == '__main__':
    main("/home/student/HW1/datasets/HW1_dataset/videos/P026_tissue1.wmv")