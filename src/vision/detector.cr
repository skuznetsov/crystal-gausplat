# YOLO object detector using CoreML
# Optimized for Apple Silicon (Neural Engine + GPU)

require "./coreml"
require "../core/tensor"

module GS
  module Vision
    # YOLO detector configuration
    struct DetectorConfig
      property confidence_threshold : Float32
      property nms_threshold : Float32
      property max_detections : Int32
      property filter_classes : Array(Int32)?  # nil = all classes

      def initialize(
        @confidence_threshold = 0.5_f32,
        @nms_threshold = 0.45_f32,
        @max_detections = 100,
        @filter_classes = nil
      )
      end

      # Preset for vehicle detection
      def self.vehicles : DetectorConfig
        DetectorConfig.new(
          confidence_threshold: 0.4_f32,
          nms_threshold: 0.5_f32,
          max_detections: 50,
          filter_classes: VEHICLE_CLASSES
        )
      end

      # Preset for high precision
      def self.high_precision : DetectorConfig
        DetectorConfig.new(
          confidence_threshold: 0.7_f32,
          nms_threshold: 0.4_f32,
          max_detections: 100
        )
      end

      # Preset for high recall (more detections)
      def self.high_recall : DetectorConfig
        DetectorConfig.new(
          confidence_threshold: 0.25_f32,
          nms_threshold: 0.6_f32,
          max_detections: 200
        )
      end
    end

    # YOLO object detector
    class Detector
      @config : DetectorConfig
      @loaded : Bool = false
      @detection_buffer : Pointer(LibCoreML::GSDetection)
      @detection_array : LibCoreML::GSDetectionArray

      def initialize(@config : DetectorConfig = DetectorConfig.new)
        # Allocate detection buffer
        @detection_buffer = Pointer(LibCoreML::GSDetection).malloc(@config.max_detections)
        @detection_array = LibCoreML::GSDetectionArray.new
        @detection_array.detections = @detection_buffer
        @detection_array.count = 0
        @detection_array.capacity = @config.max_detections
      end

      def finalize
        # Buffer will be garbage collected
      end

      # Load YOLO model from path
      def load(model_path : String) : Bool
        unless File.exists?(model_path)
          raise "Model file not found: #{model_path}"
        end

        result = LibCoreML.gs_coreml_load_yolo(model_path.to_unsafe)

        if result == 0
          @loaded = true
          true
        else
          @loaded = false
          false
        end
      end

      # Check if model is loaded
      def loaded? : Bool
        @loaded && LibCoreML.gs_coreml_is_loaded == 1
      end

      # Detect objects in RGB tensor [H, W, 3]
      def detect(image : Tensor) : Array(Detection)
        raise "Model not loaded" unless loaded?
        raise "Image must be on CPU" unless image.on_cpu?
        raise "Image must have 3 dimensions [H, W, 3]" unless image.shape.rank == 3

        height = image.shape[0]
        width = image.shape[1]
        channels = image.shape[2]
        raise "Image must have 3 channels" unless channels == 3

        # Convert float32 [0-1] to uint8 [0-255]
        float_data = image.cpu_data.not_nil!
        rgb_data = Pointer(UInt8).malloc(width * height * 3)

        (width * height * 3).times do |i|
          rgb_data[i] = (float_data[i].clamp(0.0_f32, 1.0_f32) * 255).to_u8
        end

        # Run detection
        @detection_array.count = 0
        result = LibCoreML.gs_coreml_detect(
          rgb_data,
          width, height,
          @config.confidence_threshold,
          @config.nms_threshold,
          pointerof(@detection_array)
        )

        if result != 0
          raise "Detection failed with code #{result}"
        end

        # Convert to Crystal objects
        detections = [] of Detection
        @detection_array.count.times do |i|
          d = @detection_buffer[i]

          # Apply class filter if specified
          if filter = @config.filter_classes
            next unless filter.includes?(d.class_id)
          end

          detections << Detection.new(
            x: d.x,
            y: d.y,
            width: d.width,
            height: d.height,
            confidence: d.confidence,
            class_id: d.class_id,
            track_id: d.track_id
          )
        end

        detections
      end

      # Detect objects in video frame
      def detect(frame : Video::Frame) : Array(Detection)
        detect(frame.data)
      end

      # Convenience method: detect only vehicles
      def detect_vehicles(image : Tensor) : Array(Detection)
        detections = detect(image)
        detections.select(&.is_vehicle?)
      end

      # Release model resources
      def release
        LibCoreML.gs_coreml_release
        @loaded = false
      end

      # Get class name
      def self.class_name(class_id : Int32) : String
        COCO_CLASSES[class_id]? || "unknown"
      end
    end

    # Helper to download YOLOv8 model
    module ModelDownloader
      YOLOV8_URLS = {
        "yolov8n" => "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.mlmodel",
        "yolov8s" => "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.mlmodel",
        "yolov8m" => "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.mlmodel",
      }

      def self.download(model_name : String, output_dir : String = "models") : String
        url = YOLOV8_URLS[model_name]?
        raise "Unknown model: #{model_name}" unless url

        Dir.mkdir_p(output_dir)
        output_path = File.join(output_dir, "#{model_name}.mlmodel")

        unless File.exists?(output_path)
          puts "Downloading #{model_name}..."
          # Use curl to download
          system("curl -L -o #{output_path} #{url}")
        end

        output_path
      end
    end
  end
end
