# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import os
import platform
import sys
from pathlib import Path
import cv2
import mouse
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh,
                           detectLabel, getDriver)
from utils.plots_test import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

from time import sleep
from random import uniform
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support import expected_conditions as EC

import PIL
import numpy as np
# import core.utils as utils
import tensorflow as tf
from PIL import Image


def hover(element):
    hov = ActionChains(driver).move_to_element(element)
    hov.perform()


@smart_inference_mode()
def run(
        weights=ROOT / 'run/train/exp3/best.pt',  # model path or triton URL
        source=ROOT / 'file.png',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/custom_data.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=1,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        get_height='//*[@id="rc-imageselect-target"]/table/tbody/tr[1]/td[1]/div/div[1]',
):
    # img = cv2.imread(source)
    # img = img[0:600, 0:400]
    # cv2.imwrite(source, img)
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # detect label
    # detectlabel = 'label'
    # dataLabel = detectLabel(source, detectlabel)
    dataLabel = WebDriverWait(driver, 5).until(
        EC.presence_of_element_located((By.XPATH, '//*[@id="rc-imageselect"]/div[2]/div[1]/div[1]/div/strong'))
    ).text
    print(dataLabel)
    check_height_img = WebDriverWait(driver, 5).until(
        EC.presence_of_element_located((By.XPATH, get_height))
    ).size.get('height')

    if check_height_img > 120:
        isClicked = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    else:
        isClicked = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}, "  # add to string
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')

                        # Kiểm tra xem đối tượng detect được có phải nằm trên phần label không
                        # Nếu có thì kiểm tra tiếp số lượng detect được của label đó trong chuỗi kết quả s:
                        # + Nếu số lượng lớn hơn 1 thì giảm số lượng đi 1 trong chuỗi kết quả s
                        # + Nếu số lượng bằng 1 thì xóa label đó ra khỏi chuỗi kết quả s

                        # Chuỗi s có giá trị như sau:
                        # image 1/1 D:\Newfolder\captcha\file.png: 640x448 1 bicycles, 1 bridges, 9 cars, 250.0ms
                        if int(xyxy[3]) < 120:
                            text = s.split('640x448 ')[1].split(', ')
                            label1 = label.split(' 0')[0]
                            for j in range(0, len(text)):
                                if label1 in text[j]:
                                    num = int(text[j].split(' ')[0])
                                    if num > 1:
                                        s = s.replace(str(num) + ' ' + label1, str(num - 1) + ' ' + label1)
                                        break
                                    else:
                                        s = s.replace(str(num) + ' ' + label1, '')
                                        break

                        annotator.box_label(xyxy, label, dataLabel, isClicked,
                                            check_height_img, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

        # Nếu như không detect được nhãn (vd như boats)
        if dataLabel == '':
            mouse.move(140, 340)
            mouse.click('left')
            sleep(0.3)
            mouse.move(260, 460)
            mouse.click('left')
            sleep(0.3)
            mouse.move(420, 520)
            mouse.click('left')
            sleep(0.3)

        # Kiểm tra xem loại ảnh captcha chỉ cần click 1 lần hay phải click nhiều lần
        driver1 = getDriver(driver)
        checkpoint = WebDriverWait(driver1, 5).until(
            EC.presence_of_element_located((By.CLASS_NAME, 'rc-imageselect-tileselected')))

        # detectbutton = 'button'
        # databutton = detectLabel(source, detectbutton) #kiểm tra button xác thực
        databutton = WebDriverWait(driver1, 5).until(
            EC.presence_of_element_located((By.ID, 'recaptcha-verify-button'))
        ).accessible_name
        if databutton == 'VERIFY':

            if checkpoint:  # nếu chỉ cần tích 1 lần
                sleep(0.3)
                mouse.move(450, 750)
                mouse.click('left')

                # trường hợp yêu cầu thực hiện lại khi có thông báo: Please try again
                driver1 = getDriver(driver)
                checkpoint1 = WebDriverWait(driver1, 5).until(
                    EC.presence_of_element_located((By.CLASS_NAME, 'rc-imageselect-incorrect-response')))
                if checkpoint1.aria_role == 'generic': # có xuất hiện thông báo Please try again
                    opt = parse_opt()
                    main(opt)
                else:
                    pass

                # trường hợp yêu cầu thực hiện khi có thông báo: Please select all matching images.
                # thực hiện click vào 1 số ô ảnh nhất định để chuyển qua ảnh khác
                if ((WebDriverWait(driver1, 5).until(
                        EC.presence_of_element_located((By.CLASS_NAME, 'rc-imageselect-error-select-more'))
                )).aria_role == 'generic'):
                    position = {0: '150 380', 1: '285 380', 2: '420 380',
                                3: '150 515', 4: '285 515', 5: '420 515',
                                6: '150 650', 7: '285 650', 8: '420 650'}
                    count = 0
                    for i, j in enumerate(isClicked):
                        if j == 0:
                            mouse.move(position[i].split(' ')[0], position[i].split(' ')[1])
                            mouse.click('left')
                            sleep(0.3)
                            count += 1
                        if count == 3:
                            break
                    mouse.move(450, 750)
                    mouse.click('left')
                    driver1 = getDriver(driver)
                    opt = parse_opt()
                    main(opt)
                else:
                    pass

            # nếu là trường hợp phải xác thực 1 ô nhiều lần hoặc không detect được đối tượng
            else:
                # nếu datalabel có nằm trong chuỗi kết quả s, thì thực hiện lặp lại luồng thực hiện từ đầu: chụp ảnh - detect - click
                if dataLabel in s:
                    opt = parse_opt()
                    main(opt)
                else:
                    sleep(0.3)
                    mouse.move(450, 750)
                    mouse.click('left')
                    driver1 = getDriver(driver)
                    # trường hợp yêu cầu thực hiện khi có thông báo: Please select all matching images.
                    if ((WebDriverWait(driver1, 5).until(
                            EC.presence_of_element_located((By.CLASS_NAME, 'rc-imageselect-error-select-more'))
                    )).aria_role == 'generic'):
                        position = {0: '140 340', 1: '260 340', 2: '420 340', 3: '140 460', 4: '260 460',
                                    5: '420 460', 6: '140 640', 7: '260 640', 8: '420 640'}
                        count = 0
                        for i, j in enumerate(isClicked):
                            if j == 0:
                                mouse.move(position[i].split(' ')[0], position[i].split(' ')[1])
                                mouse.click('left')
                                sleep(0.3)
                                count += 1
                            if count == 3:
                                break
                        mouse.move(450, 750)
                        mouse.click('left')
                        driver1 = getDriver(driver)
                        opt = parse_opt()
                        main(opt)

                    # trường hợp yêu cầu thực hiện khi có thông báo: Please try again.
                    if ((WebDriverWait(driver1, 5).until(
                            EC.presence_of_element_located((By.CLASS_NAME, 'rc-imageselect-incorrect-response'))
                    )).aria_role == 'generic'):
                        driver1 = getDriver(driver)
                        opt = parse_opt()
                        main(opt)

                    # trường hợp yêu cầu thực hiện khi có thông báo: Please also check the new images..
                    if ((WebDriverWait(driver1, 5).until(
                        EC.presence_of_element_located((By.CLASS_NAME, 'rc-imageselect-error-dynamic-more'))
                    )).aria_role == 'generic'):
                        position = {0: '140 340', 1: '260 340', 2: '420 340', 3: '140 460', 4: '260 460',
                                    5: '420 460', 6: '140 640', 7: '260 640', 8: '420 640'}
                        count = 0
                        for i, j in enumerate(isClicked):
                            if j == 0:
                                mouse.move(position[i].split(' ')[0], position[i].split(' ')[1])
                                mouse.click('left')
                                sleep(0.3)
                                count += 1
                            if count == 3:
                                break
                        mouse.move(450, 750)
                        mouse.click('left')
                        driver1 = getDriver(driver)
                        opt = parse_opt()
                        main(opt)
                    else:
                        pass
        else:
            # databutton == 'SKIP' or databutton == 'NEXT'
            # tức sẽ chuyển sang ảnh captcha tiếp theo
            if ((WebDriverWait(driver1, 5).until(
                    EC.presence_of_element_located((By.CLASS_NAME, 'rc-imageselect-error-select-more'))
            )).aria_role == 'generic'):
                position = {0: '140 340', 1: '260 340', 2: '420 340', 3: '140 460', 4: '260 460',
                            5: '420 460', 6: '140 640', 7: '260 640', 8: '420 640'}
                count = 0
                for i, j in enumerate(isClicked):
                    if j == 0:
                        mouse.move(position[i].split(' ')[0], position[i].split(' ')[1])
                        mouse.click('left')
                        sleep(0.3)
                        count += 1
                    if count == 3:
                        break
                mouse.move(450, 750)
                mouse.click('left')
                driver1 = getDriver(driver)
                opt = parse_opt()
                main(opt)
            else:
                mouse.move(450, 750)
                mouse.click('left')
                driver1 = getDriver(driver)
                opt = parse_opt()
                main(opt)

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'run/train/exp3/best.pt',
                        help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'file.png', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/custom_data.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=1, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--get_height', type=str,
                        default='//*[@id="rc-imageselect-target"]/table/tbody/tr[1]/td[1]/div/div[1]')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    driver = webdriver.Chrome()
    driver.maximize_window()
    driver.get("https://www.google.com/recaptcha/api2/demo")

    recaptchaFrame = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.TAG_NAME, 'iframe')))
    frameName = recaptchaFrame.get_attribute('name')
    driver.switch_to.frame(frameName)
    CheckBox = WebDriverWait(driver, 5).until(
        EC.presence_of_element_located((By.ID, "recaptcha-anchor"))
    )

    rand = uniform(0.5, 1.0)
    sleep(rand)
    hover(CheckBox)

    rand = uniform(0.5, 0.7)
    sleep(rand)
    clickReturn = CheckBox.click()

    driver = getDriver(driver)
    CheckSize = WebDriverWait(driver, 5).until(
        EC.presence_of_element_located(
            (By.XPATH, '//*[@id="rc-imageselect-target"]/table/tbody/tr[1]/td[1]/div/div[1]'))
    )
    h = WebDriverWait(driver, 5).until(
        EC.presence_of_element_located(
            (By.XPATH, '//*[@id="rc-imageselect-target"]/table/tbody/tr[1]/td[1]/div/div[1]'))
    ).size.get('height')
    opt = parse_opt()
    main(opt)
