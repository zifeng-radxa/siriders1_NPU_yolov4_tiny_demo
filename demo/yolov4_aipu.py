import cv2
import numpy as np
import subprocess
import time
import glob
import argparse
import sys
import os 

class Yolov4Demo():
    def __init__(self):
        self.cap = None
        self.score_threshold = 0.45
        self.iou_threshold = 0.5
        self.sigma = 0
        self.method = 'nms'
        self.labels = self.make_label_list()

    def check_camera(self, ):
        video_devices = sorted(glob.glob('/dev/video*'))
        for id in range(len(video_devices)):
            cap = cv2.VideoCapture(id)
            if cap.isOpened():
                print("成功打开摄像头 {}".format(id))
                return cap
            else:
                cap.release()

        print("未检测到摄像头 Exit")
        exit()
        
    def make_label_list(self,):
        with open('labels.txt', 'r') as file:
            lines = [line.strip() for line in file]

        return lines

    def preprocess(self, ol_frame):
        frame = cv2.cvtColor(ol_frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (416, 416))
        frame = np.expand_dims(frame, axis=0)
        frame.tofile('aipu_input.bin')

    def postprocess(self, ol_frame, ol_h, ol_w, use_resize=True):
        opt_box = np.fromfile('output.bin.1', dtype=np.int8).reshape(1, 2535, 4).astype(np.float32) / 0.2820578516
        opt_score = np.fromfile('output.bin.0', dtype=np.uint8).reshape(1, 2535, 80).astype(np.float32) / 256.3258667

        # default batch = 0
        pred_xywh = opt_box[0]
        pred_prob = opt_score[0]

        # box shape (x, y, w, h) --> (xmin, ymin, xmax, ymax)
        pred_coor = np.concatenate([pred_xywh[..., :2] - pred_xywh[..., 2:] * 0.5,
                                    pred_xywh[..., :2] + pred_xywh[..., 2:] * 0.5], axis=-1)
        if use_resize:
            org_h = 416
            org_w = 416
            ol_frame = cv2.resize(ol_frame, (org_h, org_w))
        else:
            org_h, org_w = ol_h, ol_w

        input_h, input_w = 416, 416

        scale_h = org_h / input_h
        scale_w = org_w / input_w


        # handle boxes that exceed boundaries
        pred_coor = np.concatenate([np.maximum(pred_coor[..., :2], [0, 0]),
                                    np.minimum(pred_coor[..., 2:], [org_w - 1, org_h - 1])], axis=-1)

        # filter some boxes with higher scores
        classes = np.argmax(pred_prob, axis=-1)
        scores = pred_prob[np.arange(len(pred_coor)), classes]
        score_mask = scores > self.score_threshold
        mask = score_mask  # np.logical_and(valid_mask, score_mask)
        coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]
        bboxes = np.concatenate([coors, scores[..., np.newaxis], classes[..., np.newaxis]], axis=-1)

        def cal_iou_xyxy(bbox_det, bbox_gt):
            '''
            :param bbox_det: n * 4, [[ymin, xmin, ymax, xmax],  [ymin, xmin, ymax, xmax], ...]
            :param bbox_gt:
            :return:
            '''
            xmin = np.maximum(bbox_det[..., 0], bbox_gt[..., 0])
            ymin = np.maximum(bbox_det[..., 1], bbox_gt[..., 1])
            xmax = np.minimum(bbox_det[..., 2], bbox_gt[..., 2])
            ymax = np.minimum(bbox_det[..., 3], bbox_gt[..., 3])

            width = np.maximum(xmax - xmin, 0)
            height = np.maximum(ymax - ymin, 0)
            inter = height * width

            union = (bbox_det[..., 2] - bbox_det[..., 0]) * (bbox_det[..., 3] - bbox_det[..., 1]) + \
                    (bbox_gt[..., 2] - bbox_gt[..., 0]) * (bbox_gt[..., 3] - bbox_gt[..., 1]) - inter

            overlap = inter / union
            return overlap

        def nms(bboxes):
            """
            param bboxes: (xmin, ymin, xmax, ymax, score, class)
            """
            classes_in_img = list(set(bboxes[..., 5]))
            best_bboxes = []

            for cls in classes_in_img:
                cls_mask = (bboxes[..., 5] == cls)
                cls_bboxes = bboxes[cls_mask]

                while len(cls_bboxes) > 0:
                    max_ind = np.argmax(cls_bboxes[..., 4])
                    best_bbox = cls_bboxes[max_ind]
                    best_bboxes.append(best_bbox)
                    cls_bboxes = np.concatenate(
                        [cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
                    iou = cal_iou_xyxy(
                        best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
                    weight = np.ones((len(iou),), dtype=np.float32)
                    assert self.method in ['nms', 'soft-nms']
                    if self.method == 'nms':
                        iou_mask = iou > self.iou_threshold
                        weight[iou_mask] = 0.0

                    if self.method == 'soft-nms':
                        weight = np.exp(-(1.0 * iou ** 2 / self.sigma))

                    cls_bboxes[:, 4] = cls_bboxes[..., 4] * weight
                    score_mask = cls_bboxes[..., 4] > 0.
                    cls_bboxes = cls_bboxes[score_mask]

            return best_bboxes

        results = nms(bboxes)

        for result in results:
            box = result[:4].astype(np.int32)
            box = box - 20 # 有点偏移
            score = result[4]
            class_id = int(result[5])
            if True:
                box[0] = int(box[0] * scale_w) 
                box[1] = int(box[1] * scale_h)  
                box[2] = int(box[2] * scale_w)  
                box[3] = int(box[3] * scale_h)  

                box[0] = max(0, min(box[0], org_w - 1)) 
                box[1] = max(0, min(box[1], org_h - 1))  
                box[2] = max(0, min(box[2], org_w - 1)) 
                box[3] = max(0, min(box[3], org_h - 1))  

                
                label = "{}".format(self.labels[class_id])
                score_percentage = round(score * 100, 2)
                show_text = "{} {}%".format(label, score_percentage)
                font_scale = 0.5
                font_thickness = 1
                color_index = class_id % 3
                color = [0, 0, 0]
                color[color_index] = 255
                cv2.rectangle(ol_frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                text_size, _ = cv2.getTextSize(show_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

                print(show_text)

                text_x, text_y = box[0] + 1, box[1] - 5
                bg_x1, bg_y1 = box[0] - 1, text_y - text_size[1] - 2
                bg_x2, bg_y2 = box[0] + text_size[0] + 2, box[1]

                # # 绘制绿色背景矩形
                cv2.rectangle(ol_frame, (bg_x1, bg_y1), (bg_x2, bg_y2), color, cv2.FILLED)

                # # 绘制白色文字
                cv2.putText(ol_frame, show_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

        
        ol_frame = cv2.resize(ol_frame, (ol_w, ol_h))
        return ol_frame

    def inference(self, ):
        command = "./aipu_test"
        args = ["aipu_yolov4_tiny.bin", "aipu_input.bin"]
        try:
            result = subprocess.run([command] + args, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True, check=True)
            print("Command output:", result.stdout)  
        except subprocess.CalledProcessError as e:
            pass
            # print("Command failed with return code:", e.returncode)
            # print("Command stdout:", e.stdout)
            # print("Command stderr:", e.stderr) 

    def stream_forward(self, camera, args=None):
        if camera:
            self.cap = self.check_camera()
        else:
            self.cap = cv2.VideoCapture(args.input)

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if args.save:
            output_path = './output/output_video.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while True:
            # time0 = time.time()
            ret, frame = self.cap.read()


            if not ret:
                print("无法接收帧，结束程序")
                break

            h,w = frame.shape[0:2]


            self.preprocess(frame)
            self.inference()
            result_frame = self.postprocess(frame, ol_h=h, ol_w=w, use_resize=False)

            # print("{} ms".format(round((time.time() - time0) * 1000, 3)))
            if args.real_time:
                cv2.imshow('Camera Stream', result_frame)

            if args.save:
                out.write(result_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        self.cap.release()
        cv2.destroyAllWindows()

    def image_forward(self, args):
        frame = cv2.imread(args.input)
        h, w = frame.shape[:2]
        self.preprocess(frame)
        self.inference()
        result = self.postprocess(frame, ol_h=h, ol_w=w, use_resize=False)

        if args.real_time:
            while True:
                cv2.imshow("output", result)
                if args.save:
                    cv2.imwrite("./output/output.jpg", result)
                if cv2.waitKey(1000) & 0xFF == ord('q'):
                    break
            cv2.destroyAllWindows()


        if not args.real_time and args.save:
            cv2.imwrite("./output/output.jpg", result)



if __name__ == '__main__':
    print(f'argv: {sys.argv}')
    description = "Zhouyi Z2 AIPU Yolov4_tiny Demo"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-m', '--mode', choices=['camera', 'video', 'image'], default='camera', help="input mode")
    parser.add_argument('-s','--save', action='store_true', help='save result')
    parser.add_argument('-r','--real_time', action='store_true', help='real time preview')
    parser.add_argument('-i','--input', required=False, help='input source path')

    args = parser.parse_args()
    current_ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    new_library_path = "./linux/libs"
    os.environ["LD_LIBRARY_PATH"] = f"{current_ld_library_path}:{new_library_path}"

    if not os.path.exists('./output'):
        os.makedirs('./output', exist_ok=True)
    
    yolo = Yolov4Demo()

    if args.mode == "camera":
        yolo.stream_forward(camera=True, args=args)
    elif args.mode == "video":
        yolo.stream_forward(camera=False,args=args)
    elif args.mode == "image":
        yolo.image_forward(args=args)
