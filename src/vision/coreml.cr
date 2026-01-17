# CoreML FFI bindings for object detection
# Uses Vision framework with YOLOv8 model

@[Link(framework: "CoreML")]
@[Link(framework: "Vision")]
lib LibCoreML
  # Detection structure
  struct GSDetection
    x : Float32         # center x (0-1)
    y : Float32         # center y (0-1)
    width : Float32     # width (0-1)
    height : Float32    # height (0-1)
    confidence : Float32
    class_id : Int32
    track_id : Int32    # -1 if not tracked
  end

  struct GSDetectionArray
    detections : GSDetection*
    count : Int32
    capacity : Int32
  end

  # Kalman track structure
  struct GSKalmanTrack
    x : Float32
    y : Float32
    vx : Float32
    vy : Float32
    w : Float32
    h : Float32
    p : StaticArray(Float32, 16)  # covariance matrix
    id : Int32
    hits : Int32
    misses : Int32
    class_id : Int32
  end

  # Model loading
  fun gs_coreml_load_yolo(model_path : LibC::Char*) : Int32
  fun gs_coreml_is_loaded : Int32
  fun gs_coreml_release : Void

  # Detection
  fun gs_coreml_detect(
    rgb_data : UInt8*,
    width : Int32,
    height : Int32,
    confidence_threshold : Float32,
    nms_threshold : Float32,
    out_detections : GSDetectionArray*
  ) : Int32

  # Class names
  fun gs_coreml_class_name(class_id : Int32) : LibC::Char*

  # Kalman filter
  fun gs_kalman_init(
    track : GSKalmanTrack*,
    id : Int32,
    x : Float32, y : Float32,
    w : Float32, h : Float32,
    class_id : Int32
  ) : Void

  fun gs_kalman_predict(track : GSKalmanTrack*, dt : Float32) : Void

  fun gs_kalman_update(
    track : GSKalmanTrack*,
    mx : Float32, my : Float32,
    mw : Float32, mh : Float32
  ) : Void

  # Hungarian assignment
  fun gs_hungarian_assign(
    cost_matrix : Float32*,
    n_tracks : Int32,
    m_detections : Int32,
    assignments : Int32*,
    max_cost : Float32
  ) : Void

  # IoU computation
  fun gs_compute_iou(
    tx : Float32, ty : Float32, tw : Float32, th : Float32,
    dx : Float32, dy : Float32, dw : Float32, dh : Float32
  ) : Float32
end

module GS
  module Vision
    # COCO class names
    COCO_CLASSES = [
      "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
      "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
      "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
      "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
      "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
      "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
      "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
      "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
      "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
      "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
      "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
      "toothbrush"
    ]

    # Vehicle class IDs in COCO
    VEHICLE_CLASSES = [2, 3, 5, 6, 7]  # car, motorcycle, bus, train, truck

    # Detection result
    struct Detection
      property x : Float32        # center x (normalized)
      property y : Float32        # center y (normalized)
      property width : Float32
      property height : Float32
      property confidence : Float32
      property class_id : Int32
      property track_id : Int32

      def initialize(@x, @y, @width, @height, @confidence, @class_id, @track_id = -1)
      end

      def class_name : String
        COCO_CLASSES[@class_id]? || "unknown"
      end

      def is_vehicle? : Bool
        VEHICLE_CLASSES.includes?(@class_id)
      end

      # Convert to pixel coordinates
      def to_pixels(img_width : Int32, img_height : Int32) : {Int32, Int32, Int32, Int32}
        cx = (@x * img_width).to_i32
        cy = (@y * img_height).to_i32
        w = (@width * img_width).to_i32
        h = (@height * img_height).to_i32
        x1 = cx - w // 2
        y1 = cy - h // 2
        {x1, y1, w, h}
      end

      # Bounding box as tuple
      def bbox : {Float32, Float32, Float32, Float32}
        {@x - @width/2, @y - @height/2, @x + @width/2, @y + @height/2}
      end
    end
  end
end
