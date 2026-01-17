# SORT (Simple Online Realtime Tracking)
# Kalman filter + Hungarian algorithm for multi-object tracking

require "./coreml"
require "./detector"

module GS
  module Vision
    # Tracker configuration
    struct TrackerConfig
      property max_age : Int32           # Max frames without detection before track dies
      property min_hits : Int32          # Min detections before track is confirmed
      property iou_threshold : Float32   # Min IoU for association
      property dt : Float32              # Time step for Kalman prediction

      def initialize(
        @max_age = 30,
        @min_hits = 3,
        @iou_threshold = 0.3_f32,
        @dt = 1.0_f32 / 30.0_f32  # Assume 30 FPS
      )
      end
    end

    # Tracked object
    class Track
      property id : Int32
      property class_id : Int32
      property hits : Int32
      property misses : Int32
      property age : Int32
      property state : LibCoreML::GSKalmanTrack

      # History for 3D reconstruction
      property history : Array({Float64, Detection})  # timestamp, detection

      def initialize(@id : Int32, detection : Detection)
        @class_id = detection.class_id
        @hits = 1
        @misses = 0
        @age = 0
        @history = [] of {Float64, Detection}

        @state = LibCoreML::GSKalmanTrack.new
        LibCoreML.gs_kalman_init(
          pointerof(@state),
          @id,
          detection.x, detection.y,
          detection.width, detection.height,
          detection.class_id
        )
      end

      # Predict next state
      def predict(dt : Float32)
        LibCoreML.gs_kalman_predict(pointerof(@state), dt)
        @age += 1
      end

      # Update with matched detection
      def update(detection : Detection, timestamp : Float64)
        LibCoreML.gs_kalman_update(
          pointerof(@state),
          detection.x, detection.y,
          detection.width, detection.height
        )
        @hits += 1
        @misses = 0

        # Store in history
        tracked_detection = Detection.new(
          x: @state.x,
          y: @state.y,
          width: @state.w,
          height: @state.h,
          confidence: detection.confidence,
          class_id: @class_id,
          track_id: @id
        )
        @history << {timestamp, tracked_detection}

        # Limit history size
        if @history.size > 1000
          @history.shift
        end
      end

      # Mark as missed this frame
      def mark_missed
        @misses += 1
      end

      # Get current detection
      def current_detection : Detection
        Detection.new(
          x: @state.x,
          y: @state.y,
          width: @state.w,
          height: @state.h,
          confidence: 1.0_f32,
          class_id: @class_id,
          track_id: @id
        )
      end

      # Is track confirmed (enough hits)?
      def confirmed?(min_hits : Int32) : Bool
        @hits >= min_hits
      end

      # Should track be deleted?
      def dead?(max_age : Int32) : Bool
        @misses > max_age
      end

      # Get class name
      def class_name : String
        COCO_CLASSES[@class_id]? || "unknown"
      end

      # Compute IoU with detection
      def iou(detection : Detection) : Float32
        LibCoreML.gs_compute_iou(
          @state.x, @state.y, @state.w, @state.h,
          detection.x, detection.y, detection.width, detection.height
        )
      end
    end

    # SORT multi-object tracker
    class Tracker
      @config : TrackerConfig
      @tracks : Array(Track)
      @next_id : Int32
      @frame_count : Int32

      def initialize(@config : TrackerConfig = TrackerConfig.new)
        @tracks = [] of Track
        @next_id = 1
        @frame_count = 0
      end

      # Update tracker with new detections
      def update(detections : Array(Detection), timestamp : Float64 = 0.0) : Array(Detection)
        @frame_count += 1

        # Step 1: Predict all tracks
        @tracks.each { |t| t.predict(@config.dt) }

        # Step 2: Associate detections to tracks using IoU
        matched_tracks, matched_dets, unmatched_tracks, unmatched_dets =
          associate(detections)

        # Step 3: Update matched tracks
        matched_tracks.each_with_index do |track_idx, i|
          det_idx = matched_dets[i]
          @tracks[track_idx].update(detections[det_idx], timestamp)
        end

        # Step 4: Mark unmatched tracks as missed
        unmatched_tracks.each do |track_idx|
          @tracks[track_idx].mark_missed
        end

        # Step 5: Create new tracks for unmatched detections
        unmatched_dets.each do |det_idx|
          track = Track.new(@next_id, detections[det_idx])
          @tracks << track
          @next_id += 1
        end

        # Step 6: Remove dead tracks
        @tracks.reject! { |t| t.dead?(@config.max_age) }

        # Return confirmed track detections
        @tracks
          .select { |t| t.confirmed?(@config.min_hits) }
          .map(&.current_detection)
      end

      # Get all active tracks
      def active_tracks : Array(Track)
        @tracks.select { |t| t.confirmed?(@config.min_hits) }
      end

      # Get specific track by ID
      def get_track(id : Int32) : Track?
        @tracks.find { |t| t.id == id }
      end

      # Get track history for 3D reconstruction
      def get_track_history(id : Int32) : Array({Float64, Detection})
        if track = get_track(id)
          track.history
        else
          [] of {Float64, Detection}
        end
      end

      # Reset tracker
      def reset
        @tracks.clear
        @next_id = 1
        @frame_count = 0
      end

      # Number of active tracks
      def count : Int32
        @tracks.size
      end

      private def associate(detections : Array(Detection)) : {Array(Int32), Array(Int32), Array(Int32), Array(Int32)}
        n_tracks = @tracks.size
        n_dets = detections.size

        if n_tracks == 0
          return {[] of Int32, [] of Int32, [] of Int32, (0...n_dets).to_a}
        end

        if n_dets == 0
          return {[] of Int32, [] of Int32, (0...n_tracks).to_a, [] of Int32}
        end

        # Build cost matrix (1 - IoU)
        cost_matrix = Pointer(Float32).malloc(n_tracks * n_dets)

        n_tracks.times do |i|
          n_dets.times do |j|
            iou = @tracks[i].iou(detections[j])
            cost_matrix[i * n_dets + j] = 1.0_f32 - iou
          end
        end

        # Run assignment
        assignments = Pointer(Int32).malloc(n_tracks)
        max_cost = 1.0_f32 - @config.iou_threshold

        LibCoreML.gs_hungarian_assign(
          cost_matrix,
          n_tracks,
          n_dets,
          assignments,
          max_cost
        )

        # Parse results
        matched_tracks = [] of Int32
        matched_dets = [] of Int32
        unmatched_tracks = [] of Int32
        det_matched = Array(Bool).new(n_dets, false)

        n_tracks.times do |i|
          if assignments[i] >= 0
            matched_tracks << i
            matched_dets << assignments[i]
            det_matched[assignments[i]] = true
          else
            unmatched_tracks << i
          end
        end

        unmatched_dets = (0...n_dets).reject { |j| det_matched[j] }

        {matched_tracks, matched_dets, unmatched_tracks, unmatched_dets}
      end
    end

    # Combined detector + tracker for video processing
    class DetectorTracker
      @detector : Detector
      @tracker : Tracker
      @filter_classes : Array(Int32)?

      def initialize(
        detector_config : DetectorConfig = DetectorConfig.vehicles,
        tracker_config : TrackerConfig = TrackerConfig.new
      )
        @detector = Detector.new(detector_config)
        @tracker = Tracker.new(tracker_config)
        @filter_classes = detector_config.filter_classes
      end

      # Load YOLO model
      def load_model(path : String) : Bool
        @detector.load(path)
      end

      # Check if model is loaded
      def loaded? : Bool
        @detector.loaded?
      end

      # Process single frame
      def process(frame : Video::Frame) : Array(Detection)
        detections = @detector.detect(frame.data)

        # Filter by class if needed
        if filter = @filter_classes
          detections = detections.select { |d| filter.includes?(d.class_id) }
        end

        @tracker.update(detections, frame.timestamp)
      end

      # Process image tensor
      def process(image : Tensor, timestamp : Float64 = 0.0) : Array(Detection)
        detections = @detector.detect(image)

        if filter = @filter_classes
          detections = detections.select { |d| filter.includes?(d.class_id) }
        end

        @tracker.update(detections, timestamp)
      end

      # Get active tracks
      def active_tracks : Array(Track)
        @tracker.active_tracks
      end

      # Get track history for 3D reconstruction
      def get_track_history(track_id : Int32) : Array({Float64, Detection})
        @tracker.get_track_history(track_id)
      end

      # Get specific track
      def get_track(id : Int32) : Track?
        @tracker.get_track(id)
      end

      # Reset tracker state
      def reset
        @tracker.reset
      end

      # Release resources
      def release
        @detector.release
      end
    end
  end
end
