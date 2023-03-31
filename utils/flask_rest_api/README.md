# Flask REST API

[REST](https://en.wikipedia.org/wiki/Representational_state_transfer) [API](https://en.wikipedia.org/wiki/API)s are
commonly used to expose Machine Learning (ML)  models to other services. This folder contains an example REST API
created using Flask to expose the YOLOv5s model from [PyTorch Hub](https://pytorch.org/hub/ultralytics_yolov5/).

## Requirements

- [Flask](https://palletsprojects.com/p/flask/) is required. Install with:

  ```shell
  $ pip install Flask
  ```

## Run

- After Flask installation run:

  ```shell
  $ python3 testApi.py
  ```

- Then use [curl](https://curl.se/) to perform a request:

  ```shell
  $ curl -X POST -F image=@zidane.jpg 'http://localhost:5000/v1/object-detection/yolov5s'
  ```
  or `Swagger`. VD:
  ```shell
    http://localhost:8000/docs#/default/detect_obj_return_json_result_object_to_json_post
  ```

- The model inference results are returned as a JSON response:

  ```json
  {
    "result": [
      {
        "xmin": 115.6615829468,
        "ymin": 139.0968170166,
        "xmax": 222.3686828613,
        "ymax": 217.5043182373,
        "confidence": 0.7991174459,
        "class": 19,
        "name": "crosswalks"
      },
      {
        "xmin": 4.1690120697,
        "ymin": 258.2970581055,
        "xmax": 42.6604614258,
        "ymax": 324.5874328613,
        "confidence": 0.7145614028,
        "class": 18,
        "name": "cars"
      },
      {
        "xmin": 115.603302002,
        "ymin": 276.9156188965,
        "xmax": 220.9389648438,
        "ymax": 302.2738647461,
        "confidence": 0.6913132668,
        "class": 19,
        "name": "crosswalks"
      },
      {
        "xmin": 275.947845459,
        "ymin": 138.4297943115,
        "xmax": 326.040222168,
        "ymax": 169.8421630859,
        "confidence": 0.6490659714,
        "class": 18,
        "name": "cars"
      },
      {
        "xmin": 36.8217163086,
        "ymin": 110.7645568848,
        "xmax": 51.0918769836,
        "ymax": 144.0728149414,
        "confidence": 0.6366413236,
        "class": 41,
        "name": "traffic lights"
      },
      {
        "xmin": 114.8069458008,
        "ymin": 253.2816162109,
        "xmax": 135.1693725586,
        "ymax": 274.7538757324,
        "confidence": 0.5581126213,
        "class": 18,
        "name": "cars"
      },
      {
        "xmin": 303.2810668945,
        "ymin": 232.5967254639,
        "xmax": 328.0079040527,
        "ymax": 267.5518188477,
        "confidence": 0.5045660138,
        "class": 41,
        "name": "traffic lights"
      },
      {
        "xmin": 116.2076568604,
        "ymin": 403.8820800781,
        "xmax": 199.3947906494,
        "ymax": 431.4833068848,
        "confidence": 0.3864547014,
        "class": 19,
        "name": "crosswalks"
      },
      {
        "xmin": 99.1036376953,
        "ymin": 118.9832000732,
        "xmax": 110.2509689331,
        "ymax": 143.2490844727,
        "confidence": 0.3351151943,
        "class": 41,
        "name": "traffic lights"
      },
      {
        "xmin": 20.6060333252,
        "ymin": 384.1140136719,
        "xmax": 50.7255744934,
        "ymax": 405.3839416504,
        "confidence": 0.3094891012,
        "class": 16,
        "name": "bus"
      },
      {
        "xmin": 196.9251708984,
        "ymin": 356.4229125977,
        "xmax": 209.0809173584,
        "ymax": 376.5519714355,
        "confidence": 0.2737327814,
        "class": 41,
        "name": "traffic lights"
      }
    ]
  }
  ```

- Trong đó:
  - `(xmin, ymin), (xmax, ymax)` là tọa độ 2 điểm để tạo thành 1 hình chữ nhật vẽ bao quanh đối tượng mà model detect được.
  - `confidence`: độ chính xác khi detect 1 đối tượng
  - `class, name` là tên của đối tượng


