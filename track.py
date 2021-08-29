import sys
sys.path.insert(0, './yolov5')

from yolov5.utils.google_utils import attempt_download
from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, check_imshow, xyxy2xywh, save_one_box
from yolov5.utils.torch_utils import select_device, time_synchronized
from yolov5.utils.plots import plot_one_box
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import argparse
import os
import platform
import shutil
import time
import numpy as np
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from Centroid_Tracker import CentroidTracker
import json
import datetime
import matplotlib
matplotlib.use('Agg')

def compute_color_for_id(label):
    """
    Simple function that adds fixed color depending on the id
    """
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def correct_json_file(json_path):
    with open(json_path, 'r') as f:
        all_text = f.read()
        text = '[' + all_text[:-1] + ']'
    with open(json_path, 'w') as f:
        f.write(text)

def write_c_files(c_file_path, detections, trackings, frame_id, shape):
    '# Frame	 x(0-1024)	 y(0-1024)	 obj id	 bounding size(0-1024^2)	 sequence(may be normalized)	 angel (0-360)	 num objects	 convex size(0-1024^2)	 width -(0-1024)	 height |(0-1024)	 movie id	 current_time	 current_milli	 time from start	 hour in day	 label (person/car/other)	 x-vel(0-1024)	 y-vel(0-1024)	 vel(0-1024)	 x-acceleration(0-1024)	 y- acceleration(0-1024)	 acceleration(0-1024)	 place_holder	 place_holder	 boxScore(0-1024)	 labelScore(0-1024)	 place_holder	 place_holder	 place_holder	 place_holder	 obj-type'
    '22, 665.6, 246.125, 0, 2385.48, 1, 0, 1, 1327.57, 34.1333, 69.8872, -99, Thu Feb 21 06:45:15 2019, 719, -99, -99, -99, 0, 0, -1, -1, -1, -1, -999, -999, -1, -1, -999, -999, 1550731515719, 0, vp'

    """    _data = [('frame#', int, 0),
             ('x', float, 1),
             ('y', float, 2),
             ('object_id', int, 3),
             ('contour_size', float, 8),
             ('width', float, 9),
             ('height', float, 10),
             ('current_date', str, 12),
             ('current_milliseconds', int, 13),
             ('is_abn', str, 15),
             ('label', str, 16),
             ('total_milli', float, 29)
             """
    coef = shape[1] / 1024
    with open(c_file_path, 'w') as c_file:
        for box in range(len(trackings)):
            #bboxes = detections[0][box, :4].cpu().numpy()
            tracks = trackings[box]
            curr_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

            Frame = frame_id
            x = ((tracks[2] - tracks[0]) + tracks[0]) / coef
            y = ((tracks[3] - tracks[1]) + tracks[1]) / coef
            obj_id = tracks[-1]
            bounding_size = (tracks[2]-tracks[0]) * (tracks[3]-tracks[1])
            sequence = 1 # sequence(may be normalized)
            angle = 0 # angel (0-360)
            num_objects = 0
            convex_size = None  # convex size(0-1024^2)
            width = tracks[3] - tracks[1]  # width -(0-1024)
            height = tracks[2] - tracks[0]  # height |(0-1024)
            movie_id = -99
            current_time = curr_time[:-7]
            current_milli = curr_time[-3:]
            time_from_start = -99
            hour_in_day = -99
            label = names[int(detections[0][:, -1])]
            x_vel = 0 # x - vel(0 - 1024)
            y_vel = 0 # y - vel(0 - 1024)
            vel = -1 # vel(0 - 1024)
            x_acc = -1 # x - acceleration(0 - 1024)
            y_acc = -1 # y - acceleration(0 - 1024)
            acc = -1 # acceleration(0 - 1024)
            ph1 = -999 # place_holder
            ph2 = -999 # place_holder(0-1024)
            boxScore = -1 # boxScore(0 - 1024)
            label_score = -1 # labelScore(0 - 1024)
            ph3 = -999
            ph4 = -999
            ph5 = None
            ph6 = 0
            obj_type = 'vp'

            c_file.write(f'{Frame}, {x}, {y}, {obj_id}, {bounding_size}, {sequence}, {angle}, {num_objects}, {convex_size}'
                         f',{width}, {height}, {movie_id}, {current_time}, {current_milli}, {time_from_start}, {hour_in_day}, {label}, {x_vel}, '
                         f'{y_vel}, {vel}, {x_acc}, {y_acc}, {acc}, {ph1}, {ph2}, {boxScore},{label_score}, {ph3}, {ph4}, {ph5}, {ph6}, {obj_type} \n')

class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        self.palette = [self.hex2rgb(c) for c in matplotlib.colors.TABLEAU_COLORS.values()]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # create instance for 'from utils.plots import colors'


def detect(opt):
    out, source, yolo_weights, deep_sort_weights, json_path, show_vid, save_vid, save_txt, imgsz, evaluate = \
        opt.output, opt.source, opt.yolo_weights, opt.deep_sort_weights, opt.json, opt.show_vid, opt.save_vid, \
            opt.save_txt, opt.img_size, opt.evaluate,
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    source = '../../../../oddetect/videos/cut_king.mp4'
    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    attempt_download(deep_sort_weights, repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # initialize CentroidTracker
    ct = CentroidTracker()

    # Initialize
    device = select_device(opt.device)

    # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    # its own .txt file. Hence, in that case, the output folder is not restored
    if not evaluate:
        if os.path.exists(out):
            pass
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(yolo_weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    global names
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays
    if show_vid:
        show_vid = check_imshow()

    if webcam:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    save_path = str(Path(out))
    # extract what is in between the last '/' and last '.'
    txt_file_name = source.split('/')[-1].split('.')[0]
    txt_path = str(Path(out)) + '/' + txt_file_name + '.txt'

    # Detection tracking from json
    if os.path.exists(json_path):
        to_save_json = False
        json_path_detection = open(os.path.join(json_path, 'detection.json'), 'r')
        json_path_tracking = open(os.path.join(json_path, 'tracking.json'), 'r')

        json_data_det = json.load(json_path_detection)
        json_data_track = json.load(json_path_tracking)
        #img_ratio = json_data_track[0].get('ratio')
    else:
        to_save_json = True
        json_path_detection = open(os.path.join(save_path, 'detection.json'), 'wt')
        json_path_tracking = open(os.path.join(save_path, 'tracking.json'), 'wt')
        batch_size = 400
        batch_imgs = []
        batch_im0 = []

    tracker_runtime = 0
    fw_runtime = 0
    cuda_time = 0
    json_time = 0
    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        if to_save_json:
            if len(batch_imgs) % batch_size == 0 and frame_idx != 0:
                #t_img = time.time()
                #torch.from_numpy(img).to(device)
                #print('Image to cuda takes: ', time.time() - t_img)

                t_cuda = time.time()
                #batch_imgs = torch.tensor(batch_imgs, dtype=torch.uint8).to(device)
                batch_imgs = torch.from_numpy(np.array(batch_imgs)).to(device)
                cuda_time += time.time() - t_cuda
                #print('Batches to cuda takes: ', time.time() - t_cuda)

                batch_imgs = batch_imgs.half() if half else batch_imgs.float()  # uint8 to fp16/32
                batch_imgs /= 255.0  # 0 - 255 to 0.0 - 1.0

                # Inference
                t1 = time.time()
                batch_det = model(batch_imgs, augment=False)[0]
                fw_runtime += time.time() - t1
                # Apply NMS
                batch_det = non_max_suppression(
                    batch_det, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
                t2 = time_synchronized()

                # Process detections
                for batch_id, frame_det in enumerate(batch_det):  # detections per image
                    if webcam:  # batch_size >= 1
                        p, s, im0 = path[batch_id], '%g: ' % batch_id, im0s[batch_id].copy()
                    else:
                        p, s, im0 = path, '', batch_im0[batch_id]

                    s += '%gx%g ' % img.shape[1:]  # print string
                    save_path = str(Path(out) / Path(p).name)
                    #gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

                    if frame_det is not None and len(frame_det):
                        # Rescale boxes from img_size to im0 size
                        frame_det[:, :4] = scale_coords(
                            img.shape[1:], frame_det[:, :4], im0.shape).round()

                        # Print results
                        for c in frame_det[:, -1].unique():
                            n = (frame_det[:, -1] == c).sum()  # detections per class
                            s += '%g %ss, ' % (n, names[int(c)])  # add to string

                        # DeepSort
                        """xywhs = xyxy2xywh(det[:, 0:4])
                        confs = det[:, 4]
                        clss = det[:, 5]
    
                        # pass detections to deepsort
                        outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss, im0)"""

                        # keep only dynamics objects for Tracking
                        dynamics_objects = ['person', 'car', 'truck', 'bus','motorcycle','train','bicycle']
                        frame_lab_class = {id:names[d] for id,d in enumerate(frame_det[:, 5].cpu().type(torch.uint8).tolist())}
                        dynamics_batch_ids = [k for k, v in frame_lab_class.items() if v in dynamics_objects]

                        #CentroidTracker update
                        tt = time.time()
                        track_outputs = ct.update(frame_det[dynamics_batch_ids, 0:4].cpu())
                        track_outputs = [[int(v[0]), int(v[1]), k] for k, v in track_outputs.items()]
                        tracker_runtime += time.time() - tt

                        # To Json
                        jj = time.time()
                        det_dict_to_json = {'fi': frame_idx + batch_id - batch_size}
                        det_dict_to_json['outputs'] = frame_det.tolist()
                        track_dict_to_json = {'fi': frame_idx + batch_id - batch_size}
                        track_dict_to_json['outputs'] = track_outputs
                        json_time += time.time() - jj

                        # Visualization - Tracking
                        if False:
                            if len(track_outputs) > 0:
                                for j, track in enumerate(track_outputs):
                                    bboxes = track[0:2]
                                    id = track[2]
                                    label = f'{id}'
                                    color = compute_color_for_id(id)
                                    #plot_one_box(bboxes, im0, label=label, color=color, line_thickness=2)
                                    cv2.putText(im0, label, bboxes,cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

                            # Visualization - Detection
                            for *xyxy, conf, cls in reversed(frame_det):
                                if save_vid or opt.save_crop or show_vid:  # Add bbox to image
                                    c = int(cls)  # integer class
                                    label = f'{names[c]} {conf:.2f}'
                                    plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=1)
                    else:
                        deepsort.increment_ages()

                    # Print time (inference + NMS)
                    #print('%sDone. (%.3fs)' % (s, t2 - t1))

                    # Stream results
                    if False and show_vid:
                        cv2.imshow(p, im0)
                        if cv2.waitKey(1) == ord('q'):  # q to quit
                            raise StopIteration

                    # Save results (image with detections)
                    if False and save_vid:
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_path += '.mp4'

                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer.write(im0)

                    # Dumb to json
                    j0 = time.time()
                    json_path_detection.write(json.dumps(det_dict_to_json) + ",")
                    json_path_detection.flush()
                    json_path_tracking.write(json.dumps(track_dict_to_json) + ",")
                    json_path_tracking.flush()
                    json_time += time.time() - j0
                batch_imgs = []
                batch_imgs.append(img)
                batch_im0 = []
                batch_im0.append(im0s)
            else:
                batch_imgs.append(img)
                batch_im0.append(im0s)
        # Display from json saved results
        else:
            try:
                outputs = json_data_det[frame_idx].get('outputs')
                outputs = [torch.Tensor(outputs).to('cuda')]
                track_outputs = json_data_track[frame_idx].get('tracking')
                track_outputs = np.array(track_outputs) if track_outputs is not None else np.array([])
                # Visualization - Tracking
                if len(track_outputs) > 0:
                    for j, track in enumerate(track_outputs):
                        bboxes = track[0:2]
                        id = track[2]
                        label = f'{id}'
                        color = compute_color_for_id(id)
                        # plot_one_box(bboxes, im0, label=label, color=color, line_thickness=2)
                        cv2.putText(im0, label, bboxes, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

                # Visualization - Detection
                for *xyxy, conf, cls in reversed(outputs[0]):
                    if save_vid or opt.save_crop or show_vid:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = f'{names[c]} {conf:.2f}'
                        plot_one_box(xyxy, im0s, label=label, color=colors(c, True), line_thickness=1)
                        #save_one_box(xyxy, im0s, file=save_path / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                # Save results (image with detections)
                if save_vid:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'

                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            except IndexError:
                print(f'Frame {frame_idx} is missing')

    if save_txt or save_vid:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    # Correct json format
    if to_save_json:
        correct_json_file(os.path.join(str(Path(out)), 'detection.json'))
        correct_json_file(os.path.join(str(Path(out)), 'tracking.json'))

    print('Done. (%.3fs)' % (time.time() - t0))
    print("Tracker runtime %.2f, per frame %.4f for %d frames "%(tracker_runtime, tracker_runtime/frame_idx, frame_idx))
    print("FW runtime %.2f, per frame %.4f for %d frames " % ( fw_runtime, fw_runtime / frame_idx, frame_idx))
    print("Cuda runtime %.2f, per frame %.4f for %d frames " % (cuda_time, cuda_time / frame_idx, frame_idx))
    print("Json runtime %.2f, per frame %.4f for %d frames " % (json_time, json_time / frame_idx, frame_idx))
    print("Dataset runtime %.2f, per frame %.4f for %d frames " % (dataset.iter_time, dataset.iter_time / frame_idx, frame_idx))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_weights', type=str, default='yolov5/weights/yolov5s.pt', help='model.pt path')
    parser.add_argument('--deep_sort_weights', type=str, default='deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7', help='ckpt.t7 path')
    parser.add_argument('--json', type=str, default='', help='json path to save results')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str, default='0', help='source')
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-vid', default=False, action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort_pytorch/configs/deep_sort.yaml")
    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)

    with torch.no_grad():
        detect(args)
