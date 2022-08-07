import cv2
import numpy as np
from alfred.vis.image.common import get_unique_color_by_id


def overlay_bbox_cv(img, dets, class_names, score_thresh):
    all_box = []
    for label in dets:
        for bbox in dets[label]:
            score = bbox[-1]
            if score > score_thresh:
                x0, y0, x1, y1 = [int(i) for i in bbox[:4]]
                all_box.append([label, x0, y0, x1, y1, score])
    all_box.sort(key=lambda v: v[5])
    for box in all_box:
        label, x0, y0, x1, y1, score = box
        # color = self.cmap(i)[:3]
        color = get_unique_color_by_id(label)
        text = "{}:{:.1f}%".format(class_names[label], score * 100)
        txt_color = (0, 0, 0) if np.mean(color) > 0.5 * 255 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        txt_size = cv2.getTextSize(text, font, 0.5, 2)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        cv2.rectangle(
            img,
            (x0, y0 - txt_size[1] - 1),
            (x0 + txt_size[0] + txt_size[1], y0 - 1),
            color,
            -1,
        )
        cv2.putText(img, text, (x0, y0 - 1), font, 0.5, txt_color, thickness=1)
    return img
