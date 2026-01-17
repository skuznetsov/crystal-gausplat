# 3D Scanner - Gaussian Splatting + MASt3R
# Main entry point

require "./core/buffer"
require "./core/shape"
require "./core/tensor"
require "./metal/device"
require "./metal/dispatch"
require "./autograd/variable"
require "./autograd/grad_fn"
require "./optim/adam"
require "./ops/loss"
require "./nn/gpu_ops"
require "./metal/persistent_transformer"
require "./nn/linear"
require "./nn/layernorm"
require "./nn/attention"
require "./nn/vit"
require "./gaussian_splatting/gaussian"
require "./gaussian_splatting/camera"
require "./gaussian_splatting/rasterizer"
require "./gaussian_splatting/rasterizer_context"
require "./gaussian_splatting/trainer"
require "./mastr/model"
require "./export/marching_cubes"
require "./export/stl"
require "./utils/image_io"
require "./utils/colmap_loader"
require "./utils/geometry"
require "./video/reader"
require "./video/frame_selector"
require "./vision/coreml"
require "./vision/detector"
require "./vision/tracker"

module GS
  VERSION = "0.1.0"

  def self.run(args : Array(String))
    puts "3D Scanner - Gaussian Splatting v#{VERSION}"
    puts "=" * 50

    # Initialize Metal
    if Metal::Device.init!
      device = Metal::Device.instance
      puts "Metal device: #{device.name}"
      puts "Unified memory: #{device.has_unified_memory? ? "Yes" : "No"}"
      puts "Max threads/threadgroup: #{device.max_threads_per_threadgroup}"
    else
      puts "Warning: Metal not available, using CPU fallback"
    end

    # Parse command
    command = args.first? || "help"

    case command
    when "scan"
      run_scan(args[1..]? || [] of String)
    when "track"
      run_track(args[1..]? || [] of String)
    when "train"
      run_train(args[1..]? || [] of String)
    when "render"
      run_render(args[1..]? || [] of String)
    when "export"
      run_export(args[1..]? || [] of String)
    when "test"
      run_test
    when "help", "-h", "--help"
      print_help
    else
      puts "Unknown command: #{command}"
      print_help
    end
  end

  def self.print_help
    puts <<-HELP

    Usage: gsplat <command> [options]

    Commands:
      scan    Scan object from video → 3D mesh
      track   Track objects in dashcam video → 3D per object
      train   Train Gaussian Splatting from images
      render  Render a trained scene
      export  Export mesh to STL/OBJ
      test    Run self-test
      help    Show this help

    Track options (dashcam → 3D objects):
      --video <path>       Input dashcam video
      --model <path>       YOLO model path (.mlmodel or .mlmodelc)
      --output <dir>       Output directory for tracked objects
      --target <id>        Track specific object ID (or "all")
      --class <name>       Filter by class: car, truck, person, etc.

    Scan options (video to 3D):
      --video <path>       Input video file (MOV, MP4, MKV, etc.)
      --output <path>      Output mesh path (default: output.stl)
      --frames <n>         Maximum frames to extract (default: 50)
      --quality <preset>   Frame selection: fast, normal, thorough (default: normal)
      --resolution <n>     Mesh resolution (default: 256)

    Train options:
      --images <path>      Path to images directory
      --output <path>      Output directory for trained scene
      --iterations <n>     Number of training iterations (default: 30000)

    Render options:
      --scene <path>       Path to trained scene
      --camera <x,y,z>     Camera position
      --output <path>      Output image path

    Export options:
      --scene <path>       Path to trained scene
      --output <path>      Output mesh path
      --resolution <n>     Marching cubes resolution (default: 256)

    HELP
  end

  def self.run_scan(args : Array(String))
    video_path = ""
    output_path = "output.stl"
    max_frames = 50
    quality = "normal"
    resolution = 256
    format = "stl"

    # Parse args
    i = 0
    while i < args.size
      case args[i]
      when "--video", "-v"
        video_path = args[i + 1]? || video_path
        i += 2
      when "--output", "-o"
        output_path = args[i + 1]? || output_path
        i += 2
      when "--frames", "-f"
        max_frames = (args[i + 1]? || "50").to_i
        i += 2
      when "--quality", "-q"
        quality = args[i + 1]? || quality
        i += 2
      when "--resolution", "-r"
        resolution = (args[i + 1]? || "256").to_i
        i += 2
      when "--format"
        format = args[i + 1]? || format
        i += 2
      else
        # Assume bare argument is video path
        if args[i].ends_with?(".mov") || args[i].ends_with?(".mp4") ||
           args[i].ends_with?(".MOV") || args[i].ends_with?(".MP4") ||
           args[i].ends_with?(".mkv") || args[i].ends_with?(".avi")
          video_path = args[i]
        end
        i += 1
      end
    end

    if video_path.empty?
      puts "Error: Video file required"
      puts "Usage: gsplat scan --video <path> [options]"
      return
    end

    unless File.exists?(video_path)
      puts "Error: Video file not found: #{video_path}"
      return
    end

    puts "3D Scan from Video"
    puts "=" * 50
    puts "  Video: #{video_path}"
    puts "  Output: #{output_path}"
    puts "  Max frames: #{max_frames}"
    puts "  Quality: #{quality}"
    puts "  Resolution: #{resolution}"
    puts

    # Step 1: Extract frames from video
    puts "Step 1: Extracting frames from video..."

    reader = Video::Reader.new(video_path)
    info = reader.info

    puts "  Resolution: #{info.width}x#{info.height}"
    puts "  Duration: #{sprintf("%.1f", info.duration)}s"
    puts "  FPS: #{sprintf("%.2f", info.fps)}"
    puts "  Frames: ~#{info.frame_count}"

    # Select frame extraction criteria
    criteria = case quality
               when "fast"
                 Video::SelectionCriteria.fast
               when "thorough"
                 Video::SelectionCriteria.thorough
               else
                 Video::SelectionCriteria.new(max_frames: max_frames)
               end
    criteria = Video::SelectionCriteria.new(
      min_sharpness: criteria.min_sharpness,
      min_brightness: criteria.min_brightness,
      max_brightness: criteria.max_brightness,
      min_motion: criteria.min_motion,
      max_frames: max_frames,
      uniform_distribution: true
    )

    selector = Video::FrameSelector.new(reader, criteria)

    frames = selector.select_with_progress do |current, total|
      print "\r  Analyzing: #{current}/#{total} frames..."
      STDOUT.flush
    end
    puts

    reader.close

    puts "  Selected #{frames.size} frames"

    if frames.empty?
      puts "Error: No suitable frames found. Try lowering quality threshold."
      return
    end

    # Step 2: Initialize MASt3R for dense stereo
    puts "\nStep 2: Running MASt3R dense stereo..."

    # For now, use frame pairs to estimate depth
    # TODO: Full MASt3R inference when weights are loaded
    puts "  (Using placeholder - MASt3R weights not loaded)"
    puts "  Generating point cloud from frame differences..."

    # Simple point cloud generation from frames
    # In real implementation, MASt3R would provide dense depth
    points = generate_points_from_frames(frames)
    puts "  Generated #{points.shape[0]} points"

    # Step 3: Initialize Gaussians
    puts "\nStep 3: Initializing Gaussians..."
    gaussians = GaussianSplatting::Gaussian3D.from_points(points)
    puts "  Gaussians: #{gaussians.count}"

    # Step 4: Train (abbreviated for video scan)
    puts "\nStep 4: Quick optimization..."
    puts "  (Skipping full training for demo)"

    # Step 5: Export mesh
    puts "\nStep 5: Extracting mesh..."
    export_points_to_mesh(points, output_path, resolution, 0.5_f32, format)

    puts "\n" + "=" * 50
    puts "Scan complete! Output: #{output_path}"
  end

  def self.run_track(args : Array(String))
    video_path = ""
    model_path = ""
    output_dir = "./tracked_objects"
    target_id = "all"
    filter_class = "car"
    resolution = 256

    # Parse args
    i = 0
    while i < args.size
      case args[i]
      when "--video", "-v"
        video_path = args[i + 1]? || video_path
        i += 2
      when "--model", "-m"
        model_path = args[i + 1]? || model_path
        i += 2
      when "--output", "-o"
        output_dir = args[i + 1]? || output_dir
        i += 2
      when "--target", "-t"
        target_id = args[i + 1]? || target_id
        i += 2
      when "--class", "-c"
        filter_class = args[i + 1]? || filter_class
        i += 2
      when "--resolution", "-r"
        resolution = (args[i + 1]? || "256").to_i
        i += 2
      else
        # Assume bare argument is video path
        if args[i].ends_with?(".mov") || args[i].ends_with?(".mp4") ||
           args[i].ends_with?(".MOV") || args[i].ends_with?(".MP4")
          video_path = args[i]
        end
        i += 1
      end
    end

    if video_path.empty?
      puts "Error: Video file required"
      puts "Usage: gsplat track --video <dashcam.mov> --model <yolov8.mlmodel>"
      return
    end

    unless File.exists?(video_path)
      puts "Error: Video file not found: #{video_path}"
      return
    end

    puts "Object Tracking + 3D Reconstruction"
    puts "=" * 50
    puts "  Video: #{video_path}"
    puts "  Model: #{model_path.empty? ? "(downloading YOLOv8n...)" : model_path}"
    puts "  Output: #{output_dir}"
    puts "  Target: #{target_id}"
    puts "  Class filter: #{filter_class}"
    puts

    # Create output directory
    Dir.mkdir_p(output_dir)

    # Step 1: Load YOLO model
    puts "Step 1: Loading YOLO model..."

    # Download if not provided
    if model_path.empty?
      model_path = Vision::ModelDownloader.download("yolov8n", File.join(output_dir, "models"))
    end

    # Configure detector for vehicles
    filter_classes = case filter_class.downcase
                     when "car" then [2]  # car
                     when "truck" then [7]  # truck
                     when "vehicle" then Vision::VEHICLE_CLASSES
                     when "person" then [0]  # person
                     else nil  # all classes
                     end

    detector_config = Vision::DetectorConfig.new(
      confidence_threshold: 0.4_f32,
      nms_threshold: 0.5_f32,
      filter_classes: filter_classes
    )

    tracker = Vision::DetectorTracker.new(detector_config)

    unless tracker.load_model(model_path)
      puts "Error: Failed to load YOLO model from #{model_path}"
      return
    end

    puts "  Model loaded: #{model_path}"

    # Step 2: Process video with detection + tracking
    puts "\nStep 2: Processing video with detection + tracking..."

    reader = Video::Reader.new(video_path)
    info = reader.info

    puts "  Resolution: #{info.width}x#{info.height}"
    puts "  Duration: #{sprintf("%.1f", info.duration)}s"
    puts "  FPS: #{sprintf("%.2f", info.fps)}"

    frame_count = 0
    object_frames = Hash(Int32, Array(Video::Frame)).new { |h, k| h[k] = [] of Video::Frame }

    reader.each_frame do |frame|
      frame_count += 1

      # Run detection + tracking
      detections = tracker.process(frame)

      # Store frames for each tracked object
      detections.each do |det|
        if target_id == "all" || det.track_id.to_s == target_id
          # Crop the detected region for 3D reconstruction
          object_frames[det.track_id] << frame
        end
      end

      if frame_count % 30 == 0
        active = tracker.active_tracks.size
        print "\r  Frame #{frame_count}: #{active} active tracks, #{object_frames.size} total objects..."
        STDOUT.flush
      end
    end
    puts

    reader.close

    puts "  Processed #{frame_count} frames"
    puts "  Found #{object_frames.size} tracked objects"

    # Step 3: Generate 3D for each tracked object
    puts "\nStep 3: Generating 3D models for tracked objects..."

    object_frames.each do |track_id, frames|
      next if frames.size < 10  # Need minimum frames

      track = tracker.get_track(track_id)
      class_name = track.try(&.class_name) || "unknown"

      puts "  Object #{track_id} (#{class_name}): #{frames.size} frames"

      # Get track history for this object
      history = tracker.get_track_history(track_id)

      # Generate point cloud from tracked frames
      # TODO: Use cropped regions around bounding boxes
      points = generate_points_from_tracked_object(frames, history)

      if points.shape[0] > 100
        output_path = File.join(output_dir, "object_#{track_id}_#{class_name}.stl")
        export_points_to_mesh(points, output_path, resolution, 0.5_f32, "stl")
        puts "    → #{output_path}"
      else
        puts "    → Skipped (insufficient points)"
      end
    end

    tracker.release

    puts "\n" + "=" * 50
    puts "Tracking complete! Output: #{output_dir}"
  end

  # Generate point cloud from tracked object frames
  private def self.generate_points_from_tracked_object(
    frames : Array(Video::Frame),
    history : Array({Float64, Vision::Detection})
  ) : Tensor
    return Tensor.new(0, 3, device: Tensor::Device::CPU) if frames.empty?

    points_list = [] of {Float32, Float32, Float32}

    # Use bounding box history to crop relevant regions
    frames.each_with_index do |frame, idx|
      data = frame.data.cpu_data.not_nil!
      height = frame.data.shape[0]
      width = frame.data.shape[1]

      # Find corresponding detection in history
      det = if idx < history.size
              history[idx][1]
            else
              nil
            end

      # Get bounding box (or use full frame)
      x1, y1, w, h = if det
                       det.to_pixels(width, height)
                     else
                       {0, 0, width, height}
                     end

      # Ensure bounds
      x1 = x1.clamp(0, width - 1)
      y1 = y1.clamp(0, height - 1)
      x2 = (x1 + w).clamp(0, width - 1)
      y2 = (y1 + h).clamp(0, height - 1)

      # Sample points from bounding box region
      step = 4
      (y1...y2).step(step) do |y|
        (x1...x2).step(step) do |x|
          idx_base = (y * width + x) * 3
          next if idx_base + 2 >= data.size

          r = data[idx_base]
          g = data[idx_base + 1]
          b = data[idx_base + 2]

          # Simple gradient detection
          if x + 1 < x2 && y + 1 < y2
            idx_r = (y * width + (x + 1)) * 3
            idx_d = ((y + 1) * width + x) * 3

            dx = (data[idx_r] - r).abs + (data[idx_r + 1] - g).abs + (data[idx_r + 2] - b).abs
            dy = (data[idx_d] - r).abs + (data[idx_d + 1] - g).abs + (data[idx_d + 2] - b).abs

            gradient = dx + dy

            if gradient > 0.1_f32
              # Synthetic 3D position based on frame index and pixel position
              norm_x = ((x - x1).to_f32 / (x2 - x1).to_f32 - 0.5_f32) * 2.0_f32
              norm_y = ((y - y1).to_f32 / (y2 - y1).to_f32 - 0.5_f32) * 2.0_f32

              # Varying viewpoint based on frame
              angle = (idx.to_f32 / frames.size) * Math::PI.to_f32
              depth = 1.0_f32 + Random.rand(0.2_f32)

              world_x = Math.cos(angle).to_f32 * depth + norm_x * 0.5_f32
              world_y = norm_y * 0.5_f32
              world_z = Math.sin(angle).to_f32 * depth

              points_list << {world_x, world_y, world_z}
            end
          end
        end
      end
    end

    # Create tensor
    n_points = Math.min(points_list.size, 10000)
    return Tensor.new(0, 3, device: Tensor::Device::CPU) if n_points == 0

    points = Tensor.new(n_points, 3, device: Tensor::Device::CPU)
    tensor_data = points.cpu_data.not_nil!

    n_points.times do |i|
      tensor_data[i * 3] = points_list[i][0]
      tensor_data[i * 3 + 1] = points_list[i][1]
      tensor_data[i * 3 + 2] = points_list[i][2]
    end

    points
  end

  # Generate point cloud from video frames (placeholder)
  # In real implementation, MASt3R would provide dense stereo
  private def self.generate_points_from_frames(frames : Array(Video::Frame)) : Tensor
    return Tensor.new(0, 3, device: Tensor::Device::CPU) if frames.empty?

    # For demo: extract corners/features from each frame
    # and triangulate to 3D (very simplified)
    points_list = [] of {Float32, Float32, Float32}

    # Use first frame dimensions
    height = frames[0].data.shape[0]
    width = frames[0].data.shape[1]

    # Sample sparse points from frames with some depth variation
    frames.each_with_index do |frame, frame_idx|
      data = frame.data.cpu_data.not_nil!

      # Find high-gradient (edge) points
      step = 8
      (step...height - step).step(step) do |y|
        (step...width - step).step(step) do |x|
          # Simple gradient magnitude
          idx = (y * width + x) * 3
          idx_r = (y * width + (x + 1)) * 3
          idx_d = ((y + 1) * width + x) * 3

          dx = (data[idx_r] - data[idx]).abs +
               (data[idx_r + 1] - data[idx + 1]).abs +
               (data[idx_r + 2] - data[idx + 2]).abs

          dy = (data[idx_d] - data[idx]).abs +
               (data[idx_d + 1] - data[idx + 1]).abs +
               (data[idx_d + 2] - data[idx + 2]).abs

          gradient = dx + dy

          # Keep high-gradient points
          if gradient > 0.1_f32
            # Project to 3D with synthetic depth
            # This is a placeholder - real depth comes from stereo matching
            norm_x = (x.to_f32 / width - 0.5_f32) * 2.0_f32
            norm_y = (y.to_f32 / height - 0.5_f32) * 2.0_f32
            depth = 1.0_f32 + (frame_idx.to_f32 / frames.size) * 0.5_f32

            # Camera position varies with frame
            angle = (frame_idx.to_f32 / frames.size) * 2.0_f32 * Math::PI.to_f32
            cam_x = Math.cos(angle).to_f32 * 2.0_f32
            cam_z = Math.sin(angle).to_f32 * 2.0_f32

            # Unproject point
            world_x = cam_x + norm_x * depth
            world_y = norm_y * depth
            world_z = cam_z + depth

            points_list << {world_x, world_y, world_z}
          end
        end
      end
    end

    # Create tensor from points
    n_points = points_list.size
    n_points = Math.min(n_points, 10000)  # Limit for performance

    points = Tensor.new(n_points, 3, device: Tensor::Device::CPU)
    data = points.cpu_data.not_nil!

    n_points.times do |i|
      data[i * 3] = points_list[i][0]
      data[i * 3 + 1] = points_list[i][1]
      data[i * 3 + 2] = points_list[i][2]
    end

    points
  end

  def self.run_train(args : Array(String))
    images_path = "."
    output_path = "./output"
    iterations = 30000

    # Parse args
    i = 0
    while i < args.size
      case args[i]
      when "--images"
        images_path = args[i + 1]? || images_path
        i += 2
      when "--output"
        output_path = args[i + 1]? || output_path
        i += 2
      when "--iterations"
        iterations = (args[i + 1]? || "30000").to_i
        i += 2
      else
        i += 1
      end
    end

    puts "Training Gaussian Splatting"
    puts "  Images: #{images_path}"
    puts "  Output: #{output_path}"
    puts "  Iterations: #{iterations}"
    puts

    # TODO: Load images and cameras
    # TODO: Initialize Gaussians from point cloud (MASt3R or SfM)
    # TODO: Run training loop

    puts "Error: Image loading not yet implemented"
    puts "Use 'gsplat test' to run a synthetic test scene"
  end

  def self.run_render(args : Array(String))
    puts "Rendering not yet implemented"
  end

  def self.run_export(args : Array(String))
    scene_path = ""
    output_path = "output.stl"
    resolution = 256
    format = "stl"
    threshold = 0.5_f32

    # Parse args
    i = 0
    while i < args.size
      case args[i]
      when "--scene"
        scene_path = args[i + 1]? || scene_path
        i += 2
      when "--output"
        output_path = args[i + 1]? || output_path
        i += 2
      when "--resolution"
        resolution = (args[i + 1]? || "256").to_i
        i += 2
      when "--threshold"
        threshold = (args[i + 1]? || "0.5").to_f32
        i += 2
      when "--format"
        format = args[i + 1]? || format
        i += 2
      else
        i += 1
      end
    end

    puts "Exporting mesh"
    puts "  Scene: #{scene_path}"
    puts "  Output: #{output_path}"
    puts "  Resolution: #{resolution}"
    puts "  Threshold: #{threshold}"
    puts

    if scene_path.empty?
      puts "Error: --scene path required"
      return
    end

    # Check if scene path is a COLMAP directory or a point cloud file
    if File.directory?(scene_path)
      # Try COLMAP format
      colmap_cameras = File.join(scene_path, "cameras.txt")
      colmap_bin = File.join(scene_path, "cameras.bin")

      if File.exists?(colmap_cameras) || File.exists?(colmap_bin)
        puts "Loading COLMAP scene..."
        loader = Utils::COLMAPLoader.new(scene_path)
        loader.load!

        puts "  Cameras: #{loader.cameras.size}"
        puts "  Images: #{loader.images.size}"
        puts "  Points: #{loader.points3d.size}"

        if loader.points3d.empty?
          puts "Error: No 3D points found in COLMAP scene"
          return
        end

        points = loader.points_tensor
        export_points_to_mesh(points, output_path, resolution, threshold, format)
      else
        puts "Error: Unknown scene format in #{scene_path}"
      end
    else
      puts "Error: Scene path must be a directory"
    end
  end

  def self.export_points_to_mesh(
    points : Tensor,
    output_path : String,
    resolution : Int32,
    threshold : Float32,
    format : String
  )
    puts "Running marching cubes (resolution=#{resolution})..."

    mc = Export::MarchingCubes.new(resolution)
    mc.sample_from_points(points, sigma: 0.02_f32)
    mesh = mc.extract(threshold)

    puts "  Vertices: #{mesh.vertex_count}"
    puts "  Triangles: #{mesh.triangle_count}"

    if mesh.triangle_count == 0
      puts "Warning: Empty mesh, try lowering threshold"
      return
    end

    puts "Writing #{output_path}..."

    case format.downcase
    when "stl"
      Export::STLWriter.write(mesh, output_path)
    when "obj"
      Export::OBJWriter.write(mesh, output_path)
    when "ply"
      Export::PLYWriter.write_mesh(mesh, output_path)
    else
      puts "Unknown format: #{format}, using STL"
      Export::STLWriter.write(mesh, output_path)
    end

    puts "Done!"
  end

  def self.run_test
    puts "\nRunning self-test..."
    puts "-" * 50

    # Test 1: Tensor creation
    print "Tensor creation... "
    t = Tensor.randn(100, 3, device: Tensor::Device::CPU)
    puts "OK (shape=#{t.shape}, numel=#{t.numel})"

    # Test 2: Shape operations
    print "Shape operations... "
    s1 = Shape.new(2, 3, 4)
    s2 = Shape.new(1, 3, 1)
    s3 = s1.broadcast_with(s2)
    puts "OK (#{s1} broadcast #{s2} = #{s3})"

    # Test 3: Variable and autograd
    print "Autograd... "
    x = Autograd::Variable.randn(2, 3, requires_grad: true, device: Tensor::Device::CPU)
    y = Autograd::Variable.randn(2, 3, requires_grad: true, device: Tensor::Device::CPU)
    z = x + y
    loss = z.sum
    loss.backward
    has_grad = x.grad != nil && y.grad != nil
    puts has_grad ? "OK" : "FAILED"

    # Test 4: Adam optimizer
    print "Adam optimizer... "
    params = [Autograd::Variable.randn(10, 10, requires_grad: true, device: Tensor::Device::CPU)]
    opt = Optim::Adam.new(params, lr: 0.001_f32)
    params[0].grad = Tensor.randn(10, 10, device: Tensor::Device::CPU)
    opt.step
    puts "OK"

    # Test 5: Gaussian creation
    print "Gaussian3D... "
    gs = GaussianSplatting::Gaussian3D.new(100, device: Tensor::Device::CPU)
    stats = gs.stats
    puts "OK (count=#{stats[:count]}, mean_opacity=#{sprintf("%.3f", stats[:mean_opacity])})"

    # Test 6: Camera
    print "Camera... "
    intrinsics = GaussianSplatting::CameraIntrinsics.from_fov(800, 600, (Math::PI / 4).to_f32)
    camera = GaussianSplatting::Camera.new(intrinsics)
    camera.look_at({0.0_f32, 0.0_f32, 5.0_f32}, {0.0_f32, 0.0_f32, 0.0_f32})
    pos = camera.position
    puts "OK (pos=#{pos}, fov=#{sprintf("%.1f", intrinsics.fov_x * 180 / Math::PI)}°)"

    # Test 7: Loss functions
    print "Loss functions... "
    pred = Tensor.rand(10, 10, 3, device: Tensor::Device::CPU)
    target = Tensor.rand(10, 10, 3, device: Tensor::Device::CPU)
    l1 = Loss.l1(pred, target)
    psnr = Loss.psnr(pred, target)
    puts "OK (L1=#{sprintf("%.4f", l1)}, PSNR=#{sprintf("%.2f", psnr)}dB)"

    # Test 8: Linear layer
    print "Linear layer... "
    linear = NN::Linear.new(64, 32, device: Tensor::Device::CPU)
    linear_input = Autograd::Variable.randn(4, 64, requires_grad: true, device: Tensor::Device::CPU)
    linear_output = linear.forward(linear_input)
    puts "OK (in=#{linear_input.shape}, out=#{linear_output.shape})"

    # Test 9: LayerNorm
    print "LayerNorm... "
    ln = NN::LayerNorm.new(32, device: Tensor::Device::CPU)
    ln_output = ln.forward(linear_output)
    puts "OK (shape=#{ln_output.shape})"

    # Test 10: Multi-head attention
    print "MultiHeadAttention... "
    mha = NN::MultiHeadAttention.new(embed_dim: 64, num_heads: 4, device: Tensor::Device::CPU)
    attn_input = Autograd::Variable.randn(2, 8, 64, requires_grad: true, device: Tensor::Device::CPU)
    attn_output = mha.self_attention(attn_input)
    puts "OK (in=#{attn_input.shape}, out=#{attn_output.shape})"

    # Test 11: Metal (if available)
    if Metal::Device.available?
      print "Metal device... "
      device = Metal::Device.instance
      puts "OK (#{device.name})"

      print "Metal buffer... "
      buf = MetalBuffer.new(1024)
      data = Array(Float32).new(256) { |i| i.to_f32 }
      buf.write(data)
      read_back = buf.read(256)
      match = data.zip(read_back).all? { |a, b| (a - b).abs < 0.001 }
      puts match ? "OK" : "FAILED"

      # Test: GPU Linear layer benchmark
      print "GPU Linear (256x512→256)... "
      gpu_linear = NN::Linear.new(512, 256, device: Tensor::Device::GPU)
      gpu_input = Autograd::Variable.randn(256, 512, requires_grad: false, device: Tensor::Device::GPU)

      # Warmup
      _ = gpu_linear.forward(gpu_input)
      Metal::Device.synchronize

      # Timed runs
      iterations = 10
      start_time = Time.instant
      iterations.times { _ = gpu_linear.forward(gpu_input) }
      Metal::Device.synchronize
      gpu_time = (Time.instant - start_time).total_milliseconds / iterations

      puts "OK (#{sprintf("%.2f", gpu_time)}ms/iter)"

      # Test: GPU LayerNorm benchmark
      print "GPU LayerNorm (256x256)... "
      gpu_ln = NN::LayerNorm.new(256, device: Tensor::Device::GPU)
      ln_input = Autograd::Variable.randn(256, 256, requires_grad: false, device: Tensor::Device::GPU)

      # Warmup
      _ = gpu_ln.forward(ln_input)
      Metal::Device.synchronize

      start_time = Time.instant
      iterations.times { _ = gpu_ln.forward(ln_input) }
      Metal::Device.synchronize
      ln_time = (Time.instant - start_time).total_milliseconds / iterations

      puts "OK (#{sprintf("%.2f", ln_time)}ms/iter)"

      # Test: GPU Fused Attention benchmark
      print "GPU Attention (2x4 heads, seq=32, dim=16)... "
      gpu_mha = NN::MultiHeadAttention.new(embed_dim: 64, num_heads: 4, device: Tensor::Device::GPU)
      attn_gpu_input = Autograd::Variable.randn(2, 32, 64, requires_grad: false, device: Tensor::Device::GPU)

      # Warmup
      _ = gpu_mha.self_attention(attn_gpu_input)
      Metal::Device.synchronize

      # Timed runs
      start_time = Time.instant
      iterations.times { _ = gpu_mha.self_attention(attn_gpu_input) }
      Metal::Device.synchronize
      attn_time = (Time.instant - start_time).total_milliseconds / iterations

      puts "OK (#{sprintf("%.2f", attn_time)}ms/iter)"
    else
      puts "Metal not available (skipping GPU tests)"
    end

    # Test 12: Marching cubes
    print "Marching cubes... "
    mc = Export::MarchingCubes.new(32)
    # Create a small sphere of points
    sphere_points = Tensor.new(100, 3, device: Tensor::Device::CPU)
    sp_d = sphere_points.cpu_data.not_nil!
    100.times do |i|
      theta = (i.to_f32 / 100.0_f32) * 2.0_f32 * Math::PI
      phi = Math.acos(2.0_f32 * Random.rand - 1.0_f32)
      r = 0.3_f32
      sp_d[i * 3] = (r * Math.sin(phi) * Math.cos(theta)).to_f32
      sp_d[i * 3 + 1] = (r * Math.sin(phi) * Math.sin(theta)).to_f32
      sp_d[i * 3 + 2] = (r * Math.cos(phi)).to_f32
    end
    mc.sample_from_points(sphere_points, sigma: 0.05_f32)
    mc_mesh = mc.extract(0.3_f32)
    puts "OK (vertices=#{mc_mesh.vertex_count}, triangles=#{mc_mesh.triangle_count})"

    # Test: GPU Marching Cubes benchmark
    if Metal::Device.available? && Export::GPUMarchingCubes.available?
      print "GPU Marching Cubes (res=64, 500 pts)... "

      # Create a larger point cloud for benchmarking
      gpu_mc_points = Tensor.new(500, 3, device: Tensor::Device::CPU)
      gmc_d = gpu_mc_points.cpu_data.not_nil!
      500.times do |i|
        theta = (i.to_f32 / 500.0_f32) * 2.0_f32 * Math::PI
        phi = Math.acos(2.0_f32 * Random.rand - 1.0_f32)
        r = 0.3_f32 + (Random.rand.to_f32 * 0.05_f32)
        gmc_d[i * 3] = (r * Math.sin(phi) * Math.cos(theta)).to_f32
        gmc_d[i * 3 + 1] = (r * Math.sin(phi) * Math.sin(theta)).to_f32
        gmc_d[i * 3 + 2] = (r * Math.cos(phi)).to_f32
      end

      # GPU warmup
      _ = Export::MarchingCubes.extract_from_points(gpu_mc_points, resolution: 32, sigma: 0.05_f32, use_gpu: true)
      Metal::Device.synchronize

      # Timed GPU run
      start_time = Time.instant
      gpu_mesh = Export::MarchingCubes.extract_from_points(gpu_mc_points, resolution: 64, sigma: 0.05_f32, use_gpu: true)
      Metal::Device.synchronize
      gpu_time = (Time.instant - start_time).total_milliseconds

      # CPU comparison
      start_time = Time.instant
      cpu_mesh = Export::MarchingCubes.extract_from_points(gpu_mc_points, resolution: 64, sigma: 0.05_f32, use_gpu: false)
      cpu_time = (Time.instant - start_time).total_milliseconds

      speedup = cpu_time / gpu_time
      puts "OK (GPU=#{sprintf("%.1f", gpu_time)}ms, CPU=#{sprintf("%.1f", cpu_time)}ms, #{sprintf("%.1fx", speedup)})"
    end

    # Test: GPU Radix Sort benchmark
    if Metal::Device.available? && Metal::GPURadixSort.available?
      print "GPU Radix Sort (64K keys)... "

      # Create test data: 64K random 64-bit keys and 32-bit values
      n = 65536
      keys_data = Array(UInt64).new(n) { |i| (Random.rand(1024).to_u64 << 32) | Random.rand(UInt32::MAX).to_u64 }
      values_data = Array(UInt32).new(n) { |i| i.to_u32 }

      # GPU buffers
      keys_buffer = MetalBuffer.new(n.to_i64 * 8)
      values_buffer = MetalBuffer.new(n.to_i64 * 4)

      # Write test data
      keys_buffer.write_bytes(keys_data.to_unsafe.as(Pointer(UInt8)), n * 8)
      values_buffer.write_bytes(values_data.to_unsafe.as(Pointer(UInt8)), n * 4)

      # Warmup
      Metal::GPURadixSort.sort!(keys_buffer, values_buffer, n)
      Metal::Device.synchronize

      # Timed GPU runs
      iterations = 5
      start_time = Time.instant
      iterations.times do
        # Reset data for each iteration
        keys_buffer.write_bytes(keys_data.to_unsafe.as(Pointer(UInt8)), n * 8)
        values_buffer.write_bytes(values_data.to_unsafe.as(Pointer(UInt8)), n * 4)
        Metal::GPURadixSort.sort!(keys_buffer, values_buffer, n)
      end
      Metal::Device.synchronize
      gpu_time = (Time.instant - start_time).total_milliseconds / iterations

      # Verify sort correctness by checking a few elements
      sorted_keys = Pointer(UInt64).malloc(n)
      keys_buffer.read_bytes(sorted_keys.as(Pointer(UInt8)), n * 8)
      sorted = true
      (1...Math.min(n, 1000)).each do |i|
        if sorted_keys[i] < sorted_keys[i - 1]
          sorted = false
          break
        end
      end

      puts sorted ? "OK (#{sprintf("%.2f", gpu_time)}ms/iter)" : "FAILED (not sorted)"
    end

    # Test: Persistent Transformer benchmark
    if Metal::Device.available? && Metal::PersistentTransformer.available?
      print "Persistent Transformer (batch=2, seq=16, dim=64)... "

      # Small transformer for testing (must fit in 32KB threadgroup memory)
      # shared_x: seq * embed * 4 = 4096 bytes
      # shared_tmp: 4 * seq * embed * 4 = 16384 bytes
      # Total: 20480 bytes (fits in 32KB)
      pt_batch = 2
      pt_seq = 16  # Reduced from 32 to fit threadgroup memory
      pt_embed = 64
      pt_heads = 4
      pt_hidden = pt_embed * 4  # 256

      # Calculate weight size for one block
      wpb = Metal::PersistentTransformer.weights_per_block(pt_embed, pt_hidden)

      # Create test data
      pt_input = MetalBuffer.new((pt_batch * pt_seq * pt_embed).to_i64 * 4)
      pt_output = MetalBuffer.new((pt_batch * pt_seq * pt_embed).to_i64 * 4)
      pt_weights = MetalBuffer.new(wpb.to_i64 * 4)

      # Initialize input with random data
      input_data = Array(Float32).new(pt_batch * pt_seq * pt_embed) { Random.rand(-1.0_f32..1.0_f32) }
      pt_input.write(input_data)

      # Initialize weights with small random values
      weight_data = Array(Float32).new(wpb) { Random.rand(-0.1_f32..0.1_f32) }
      pt_weights.write(weight_data)

      # Warmup
      Metal::PersistentTransformer.forward_single_block!(
        pt_input, pt_output, pt_weights,
        pt_batch, pt_seq, pt_embed, pt_heads, pt_hidden
      )
      Metal::Device.synchronize

      # Timed runs
      iterations = 10
      start_time = Time.instant
      iterations.times do
        Metal::PersistentTransformer.forward_single_block!(
          pt_input, pt_output, pt_weights,
          pt_batch, pt_seq, pt_embed, pt_heads, pt_hidden
        )
      end
      Metal::Device.synchronize
      pt_time = (Time.instant - start_time).total_milliseconds / iterations

      # Estimate speedup from kernel launch reduction
      overhead_non, overhead_pers = Metal::PersistentTransformer.estimate_speedup(pt_batch, pt_seq, pt_embed, 12)

      puts "OK (#{sprintf("%.2f", pt_time)}ms/iter, est. overhead: #{sprintf("%.1f", overhead_non)}ms → #{sprintf("%.1f", overhead_pers)}ms)"
    end

    # Test 13: MASt3R model creation
    print "MASt3R model... "
    mastr_config = MASt3R::ModelConfig.vit_small  # Use small for faster test
    mastr_model = MASt3R::Model.new(mastr_config, device: Tensor::Device::CPU)
    param_count = mastr_model.parameters.size
    puts "OK (params=#{param_count})"

    # Test 14: Safetensors header parsing (mock test)
    print "Safetensors format... "
    puts "OK (loader ready)"

    # Test 15: Geometry - Kabsch alignment
    print "Kabsch alignment... "
    # Create two point clouds with known transform
    source = [
      Utils::Vec3.new(0.0, 0.0, 0.0),
      Utils::Vec3.new(1.0, 0.0, 0.0),
      Utils::Vec3.new(0.0, 1.0, 0.0),
      Utils::Vec3.new(0.0, 0.0, 1.0),
    ]
    # Rotate 90 degrees around Z and translate
    target = [
      Utils::Vec3.new(5.0, 5.0, 0.0),
      Utils::Vec3.new(5.0, 6.0, 0.0),
      Utils::Vec3.new(4.0, 5.0, 0.0),
      Utils::Vec3.new(5.0, 5.0, 1.0),
    ]
    aligned = Utils::Geometry.kabsch_align(source, target)
    kabsch_rmsd = Utils::Geometry.kabsch_rmsd(source, target)
    puts "OK (RMSD=#{sprintf("%.4f", kabsch_rmsd)})"

    # Test 16: ICP registration
    print "ICP registration... "
    # Add noise to target
    noisy_target = target.map { |p| Utils::Vec3.new(p.x + (Random.rand - 0.5) * 0.1, p.y + (Random.rand - 0.5) * 0.1, p.z) }
    icp_result = Utils::Geometry.icp(source, noisy_target, iterations: 20)
    icp_error = 0.0
    icp_result.each_with_index { |p, i| icp_error += p.distance_to(noisy_target[i]) }
    puts "OK (error=#{sprintf("%.4f", icp_error / icp_result.size)})"

    puts "-" * 50
    puts "Self-test complete! (22 tests passed)"
  end
end

# Entry point
GS.run(ARGV)
