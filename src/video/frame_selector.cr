# Smart frame selection for 3D reconstruction
# Filters out blurry frames and ensures sufficient camera movement

require "./reader"
require "../core/tensor"

module GS
  module Video
    # Frame quality metrics
    struct FrameMetrics
      property sharpness : Float32     # Laplacian variance (higher = sharper)
      property brightness : Float32    # Mean brightness
      property motion : Float32        # Motion from previous frame

      def initialize(@sharpness, @brightness, @motion)
      end
    end

    # Frame with quality metrics
    struct QualifiedFrame
      property frame : Frame
      property metrics : FrameMetrics

      def initialize(@frame, @metrics)
      end
    end

    # Selection criteria
    struct SelectionCriteria
      property min_sharpness : Float32       # Minimum Laplacian variance
      property min_brightness : Float32      # Minimum mean brightness [0-1]
      property max_brightness : Float32      # Maximum mean brightness [0-1]
      property min_motion : Float32          # Minimum motion between frames
      property max_frames : Int32            # Maximum frames to select
      property uniform_distribution : Bool   # Try to distribute frames uniformly

      def initialize(
        @min_sharpness = 100.0_f32,
        @min_brightness = 0.1_f32,
        @max_brightness = 0.9_f32,
        @min_motion = 0.02_f32,
        @max_frames = 100,
        @uniform_distribution = true
      )
      end

      # Presets
      def self.high_quality : SelectionCriteria
        SelectionCriteria.new(
          min_sharpness: 150.0_f32,
          min_motion: 0.03_f32,
          max_frames: 50
        )
      end

      def self.fast : SelectionCriteria
        SelectionCriteria.new(
          min_sharpness: 50.0_f32,
          min_motion: 0.05_f32,
          max_frames: 30
        )
      end

      def self.thorough : SelectionCriteria
        SelectionCriteria.new(
          min_sharpness: 80.0_f32,
          min_motion: 0.015_f32,
          max_frames: 150
        )
      end
    end

    # Smart frame selector
    class FrameSelector
      @reader : Reader
      @criteria : SelectionCriteria
      @prev_gray : Array(Float32)?

      def initialize(@reader : Reader, @criteria : SelectionCriteria = SelectionCriteria.new)
        @prev_gray = nil
      end

      # Select best frames from video
      def select : Array(Frame)
        all_frames = [] of QualifiedFrame

        # First pass: collect all frames with metrics
        @reader.each_frame do |frame|
          metrics = compute_metrics(frame)

          # Apply basic filters
          next if metrics.sharpness < @criteria.min_sharpness
          next if metrics.brightness < @criteria.min_brightness
          next if metrics.brightness > @criteria.max_brightness

          all_frames << QualifiedFrame.new(frame, metrics)
        end

        return [] of Frame if all_frames.empty?

        # Second pass: filter by motion and limit count
        selected = filter_by_motion(all_frames)

        # Limit to max_frames, preferring high sharpness
        if selected.size > @criteria.max_frames
          selected = select_top_frames(selected, @criteria.max_frames)
        end

        selected.map(&.frame)
      end

      # Select frames with progress callback
      def select_with_progress(&block : Int32, Int32 -> _) : Array(Frame)
        all_frames = [] of QualifiedFrame
        total = @reader.info.frame_count.to_i32
        processed = 0

        @reader.each_frame do |frame|
          processed += 1
          block.call(processed, total) if processed % 10 == 0

          metrics = compute_metrics(frame)

          next if metrics.sharpness < @criteria.min_sharpness
          next if metrics.brightness < @criteria.min_brightness
          next if metrics.brightness > @criteria.max_brightness

          all_frames << QualifiedFrame.new(frame, metrics)
        end

        return [] of Frame if all_frames.empty?

        selected = filter_by_motion(all_frames)

        if selected.size > @criteria.max_frames
          selected = select_top_frames(selected, @criteria.max_frames)
        end

        selected.map(&.frame)
      end

      private def compute_metrics(frame : Frame) : FrameMetrics
        data = frame.data.cpu_data.not_nil!
        height = frame.data.shape[0]
        width = frame.data.shape[1]

        # Convert to grayscale
        gray = Array(Float32).new(height * width, 0.0_f32)
        (height * width).times do |i|
          r = data[i * 3 + 0]
          g = data[i * 3 + 1]
          b = data[i * 3 + 2]
          gray[i] = 0.299_f32 * r + 0.587_f32 * g + 0.114_f32 * b
        end

        # Compute sharpness using Laplacian variance
        sharpness = compute_laplacian_variance(gray, width, height)

        # Compute mean brightness
        brightness = gray.sum / gray.size

        # Compute motion from previous frame
        motion = 0.0_f32
        if prev = @prev_gray
          motion = compute_motion(prev, gray)
        end

        @prev_gray = gray

        FrameMetrics.new(sharpness, brightness, motion)
      end

      private def compute_laplacian_variance(gray : Array(Float32), width : Int32, height : Int32) : Float32
        # Laplacian kernel: [0, 1, 0; 1, -4, 1; 0, 1, 0]
        laplacian = Array(Float32).new((height - 2) * (width - 2), 0.0_f32)

        (1...height - 1).each do |y|
          (1...width - 1).each do |x|
            center = gray[y * width + x]
            top = gray[(y - 1) * width + x]
            bottom = gray[(y + 1) * width + x]
            left = gray[y * width + (x - 1)]
            right = gray[y * width + (x + 1)]

            lap = -4.0_f32 * center + top + bottom + left + right
            laplacian[(y - 1) * (width - 2) + (x - 1)] = lap
          end
        end

        # Variance of Laplacian
        mean = laplacian.sum / laplacian.size
        variance = laplacian.map { |x| (x - mean) ** 2 }.sum / laplacian.size

        variance
      end

      private def compute_motion(prev : Array(Float32), curr : Array(Float32)) : Float32
        return 0.0_f32 if prev.size != curr.size
        return 0.0_f32 if prev.empty?

        # Mean absolute difference
        diff_sum = 0.0_f32
        prev.size.times do |i|
          diff_sum += (prev[i] - curr[i]).abs
        end

        diff_sum / prev.size
      end

      private def filter_by_motion(frames : Array(QualifiedFrame)) : Array(QualifiedFrame)
        return frames if frames.size <= 1

        selected = [frames[0]]

        (1...frames.size).each do |i|
          frame = frames[i]

          # Check if there's enough motion from last selected frame
          if frame.metrics.motion >= @criteria.min_motion
            selected << frame
          end
        end

        selected
      end

      private def select_top_frames(frames : Array(QualifiedFrame), n : Int32) : Array(QualifiedFrame)
        if @criteria.uniform_distribution
          # Select uniformly distributed frames, preferring sharper ones
          select_uniform(frames, n)
        else
          # Just take the sharpest frames
          frames.sort_by { |f| -f.metrics.sharpness }.first(n)
        end
      end

      private def select_uniform(frames : Array(QualifiedFrame), n : Int32) : Array(QualifiedFrame)
        return frames if frames.size <= n

        # Divide into n buckets and take the sharpest from each
        bucket_size = frames.size.to_f32 / n
        selected = [] of QualifiedFrame

        n.times do |i|
          start_idx = (i * bucket_size).to_i32
          end_idx = ((i + 1) * bucket_size).to_i32.clamp(start_idx + 1, frames.size)

          bucket = frames[start_idx...end_idx]
          best = bucket.max_by { |f| f.metrics.sharpness }
          selected << best
        end

        selected
      end
    end

    # Convenience method to extract frames from video
    def self.extract_frames(
      path : String,
      criteria : SelectionCriteria = SelectionCriteria.new
    ) : Array(Frame)
      reader = Reader.new(path)
      selector = FrameSelector.new(reader, criteria)
      frames = selector.select
      reader.close
      frames
    end

    # Extract frames with simple sampling (no quality filtering)
    def self.extract_uniform(path : String, n_frames : Int32) : Array(Frame)
      reader = Reader.new(path)
      frames = reader.read_uniform(n_frames)
      reader.close
      frames
    end

    # Extract frames at fixed interval
    def self.extract_interval(path : String, interval_sec : Float64) : Array(Frame)
      reader = Reader.new(path)
      frames = reader.read_sampled(interval_sec)
      reader.close
      frames
    end
  end
end
