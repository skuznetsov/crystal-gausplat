# COLMAP format loader
# Parses cameras.txt, images.txt, points3D.txt

require "../core/tensor"
require "../gaussian_splatting/camera"

module GS
  module Utils
    # COLMAP camera model types
    enum COLMAPCameraModel
      SIMPLE_PINHOLE  # f, cx, cy
      PINHOLE         # fx, fy, cx, cy
      SIMPLE_RADIAL   # f, cx, cy, k
      RADIAL          # f, cx, cy, k1, k2
      OPENCV          # fx, fy, cx, cy, k1, k2, p1, p2
      FULL_OPENCV     # fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6
    end

    # COLMAP camera intrinsics
    struct COLMAPCamera
      property id : Int32
      property model : COLMAPCameraModel
      property width : Int32
      property height : Int32
      property params : Array(Float64)

      def initialize(@id, @model, @width, @height, @params)
      end

      # Convert to GS::Camera format
      def to_gs_camera(world_to_camera : Tensor) : GaussianSplatting::Camera
        fx, fy, cx, cy = case @model
                         when .simple_pinhole?
                           {@params[0], @params[0], @params[1], @params[2]}
                         when .pinhole?
                           {@params[0], @params[1], @params[2], @params[3]}
                         when .simple_radial?, .radial?
                           {@params[0], @params[0], @params[1], @params[2]}
                         when .opencv?, .full_opencv?
                           {@params[0], @params[1], @params[2], @params[3]}
                         else
                           {@params[0], @params[0], @params[1], @params[2]}
                         end

        GaussianSplatting::Camera.new(
          width: @width,
          height: @height,
          fx: fx.to_f32,
          fy: fy.to_f32,
          cx: cx.to_f32,
          cy: cy.to_f32,
          world_to_camera: world_to_camera
        )
      end
    end

    # COLMAP image with pose
    struct COLMAPImage
      property id : Int32
      property camera_id : Int32
      property name : String
      property qw : Float64
      property qx : Float64
      property qy : Float64
      property qz : Float64
      property tx : Float64
      property ty : Float64
      property tz : Float64

      def initialize(@id, @camera_id, @name, @qw, @qx, @qy, @qz, @tx, @ty, @tz)
      end

      # Get rotation matrix from quaternion
      def rotation_matrix : Tensor
        # Normalize quaternion
        norm = Math.sqrt(@qw * @qw + @qx * @qx + @qy * @qy + @qz * @qz)
        w = @qw / norm
        x = @qx / norm
        y = @qy / norm
        z = @qz / norm

        # Rotation matrix from quaternion
        r = Tensor.new(3, 3, device: Tensor::Device::CPU)
        r_d = r.cpu_data.not_nil!

        r_d[0] = (1.0 - 2.0 * (y * y + z * z)).to_f32
        r_d[1] = (2.0 * (x * y - z * w)).to_f32
        r_d[2] = (2.0 * (x * z + y * w)).to_f32

        r_d[3] = (2.0 * (x * y + z * w)).to_f32
        r_d[4] = (1.0 - 2.0 * (x * x + z * z)).to_f32
        r_d[5] = (2.0 * (y * z - x * w)).to_f32

        r_d[6] = (2.0 * (x * z - y * w)).to_f32
        r_d[7] = (2.0 * (y * z + x * w)).to_f32
        r_d[8] = (1.0 - 2.0 * (x * x + y * y)).to_f32

        r
      end

      # Get translation vector
      def translation : Tensor
        t = Tensor.new(3, device: Tensor::Device::CPU)
        t_d = t.cpu_data.not_nil!
        t_d[0] = @tx.to_f32
        t_d[1] = @ty.to_f32
        t_d[2] = @tz.to_f32
        t
      end

      # Get 4x4 world-to-camera matrix
      def world_to_camera : Tensor
        mat = Tensor.new(4, 4, device: Tensor::Device::CPU)
        m_d = mat.cpu_data.not_nil!

        r = rotation_matrix
        r_d = r.cpu_data.not_nil!

        # Copy rotation
        3.times do |i|
          3.times do |j|
            m_d[i * 4 + j] = r_d[i * 3 + j]
          end
        end

        # Copy translation
        m_d[3] = @tx.to_f32
        m_d[7] = @ty.to_f32
        m_d[11] = @tz.to_f32

        # Bottom row
        m_d[12] = 0.0_f32
        m_d[13] = 0.0_f32
        m_d[14] = 0.0_f32
        m_d[15] = 1.0_f32

        mat
      end

      # Get camera position in world coordinates
      def camera_center : {Float64, Float64, Float64}
        r = rotation_matrix
        r_d = r.cpu_data.not_nil!

        # C = -R^T * t
        cx = -(r_d[0] * @tx + r_d[3] * @ty + r_d[6] * @tz)
        cy = -(r_d[1] * @tx + r_d[4] * @ty + r_d[7] * @tz)
        cz = -(r_d[2] * @tx + r_d[5] * @ty + r_d[8] * @tz)

        {cx, cy, cz}
      end
    end

    # COLMAP 3D point
    struct COLMAPPoint3D
      property id : Int64
      property x : Float64
      property y : Float64
      property z : Float64
      property r : UInt8
      property g : UInt8
      property b : UInt8
      property error : Float64
      property track_length : Int32

      def initialize(@id, @x, @y, @z, @r, @g, @b, @error, @track_length)
      end
    end

    # COLMAP scene loader
    class COLMAPLoader
      getter cameras : Hash(Int32, COLMAPCamera)
      getter images : Hash(Int32, COLMAPImage)
      getter points3d : Array(COLMAPPoint3D)
      getter path : String

      def initialize(@path : String)
        @cameras = Hash(Int32, COLMAPCamera).new
        @images = Hash(Int32, COLMAPImage).new
        @points3d = Array(COLMAPPoint3D).new
      end

      # Load all COLMAP data
      def load!
        # Try text format first, then binary
        cameras_txt = File.join(@path, "cameras.txt")
        cameras_bin = File.join(@path, "cameras.bin")

        if File.exists?(cameras_txt)
          load_cameras_txt!(cameras_txt)
          load_images_txt!(File.join(@path, "images.txt"))
          points_txt = File.join(@path, "points3D.txt")
          load_points3d_txt!(points_txt) if File.exists?(points_txt)
        elsif File.exists?(cameras_bin)
          load_cameras_bin!(cameras_bin)
          load_images_bin!(File.join(@path, "images.bin"))
          points_bin = File.join(@path, "points3D.bin")
          load_points3d_bin!(points_bin) if File.exists?(points_bin)
        else
          raise "No COLMAP data found in #{@path}"
        end
      end

      # Get GS cameras for all images
      def gs_cameras : Array(GaussianSplatting::Camera)
        result = Array(GaussianSplatting::Camera).new

        @images.each_value do |img|
          cam = @cameras[img.camera_id]?
          next unless cam

          w2c = img.world_to_camera
          result << cam.to_gs_camera(w2c)
        end

        result
      end

      # Get point cloud as tensor [N, 3]
      def points_tensor(device : Tensor::Device = Tensor::Device::CPU) : Tensor
        n = @points3d.size
        return Tensor.new(0, 3, device: device) if n == 0

        tensor = Tensor.new(n, 3, device: Tensor::Device::CPU)
        t_d = tensor.cpu_data.not_nil!

        @points3d.each_with_index do |pt, i|
          t_d[i * 3] = pt.x.to_f32
          t_d[i * 3 + 1] = pt.y.to_f32
          t_d[i * 3 + 2] = pt.z.to_f32
        end

        device.gpu? ? tensor.to_gpu : tensor
      end

      # Get point colors as tensor [N, 3] normalized 0-1
      def colors_tensor(device : Tensor::Device = Tensor::Device::CPU) : Tensor
        n = @points3d.size
        return Tensor.new(0, 3, device: device) if n == 0

        tensor = Tensor.new(n, 3, device: Tensor::Device::CPU)
        t_d = tensor.cpu_data.not_nil!

        @points3d.each_with_index do |pt, i|
          t_d[i * 3] = pt.r.to_f32 / 255.0_f32
          t_d[i * 3 + 1] = pt.g.to_f32 / 255.0_f32
          t_d[i * 3 + 2] = pt.b.to_f32 / 255.0_f32
        end

        device.gpu? ? tensor.to_gpu : tensor
      end

      # Load cameras from text format
      private def load_cameras_txt!(path : String)
        File.each_line(path) do |line|
          line = line.strip
          next if line.empty? || line.starts_with?("#")

          parts = line.split
          next if parts.size < 5

          id = parts[0].to_i32
          model_str = parts[1]
          width = parts[2].to_i32
          height = parts[3].to_i32
          params = parts[4..].map(&.to_f64)

          model = case model_str
                  when "SIMPLE_PINHOLE" then COLMAPCameraModel::SIMPLE_PINHOLE
                  when "PINHOLE"        then COLMAPCameraModel::PINHOLE
                  when "SIMPLE_RADIAL"  then COLMAPCameraModel::SIMPLE_RADIAL
                  when "RADIAL"         then COLMAPCameraModel::RADIAL
                  when "OPENCV"         then COLMAPCameraModel::OPENCV
                  when "FULL_OPENCV"    then COLMAPCameraModel::FULL_OPENCV
                  else                       COLMAPCameraModel::PINHOLE
                  end

          @cameras[id] = COLMAPCamera.new(id, model, width, height, params)
        end
      end

      # Load images from text format
      private def load_images_txt!(path : String)
        lines = File.read_lines(path).reject { |l| l.strip.empty? || l.starts_with?("#") }

        i = 0
        while i < lines.size
          parts = lines[i].split
          break if parts.size < 10

          id = parts[0].to_i32
          qw = parts[1].to_f64
          qx = parts[2].to_f64
          qy = parts[3].to_f64
          qz = parts[4].to_f64
          tx = parts[5].to_f64
          ty = parts[6].to_f64
          tz = parts[7].to_f64
          camera_id = parts[8].to_i32
          name = parts[9]

          @images[id] = COLMAPImage.new(id, camera_id, name, qw, qx, qy, qz, tx, ty, tz)

          i += 2  # Skip 2D points line
        end
      end

      # Load points3D from text format
      private def load_points3d_txt!(path : String)
        File.each_line(path) do |line|
          line = line.strip
          next if line.empty? || line.starts_with?("#")

          parts = line.split
          next if parts.size < 8

          id = parts[0].to_i64
          x = parts[1].to_f64
          y = parts[2].to_f64
          z = parts[3].to_f64
          r = parts[4].to_u8
          g = parts[5].to_u8
          b = parts[6].to_u8
          error = parts[7].to_f64
          # Track info follows but we skip it

          @points3d << COLMAPPoint3D.new(id, x, y, z, r, g, b, error, 0)
        end
      end

      # Load cameras from binary format
      private def load_cameras_bin!(path : String)
        File.open(path, "rb") do |file|
          num_cameras = read_u64(file)

          num_cameras.times do
            camera_id = read_i32(file)
            model_id = read_i32(file)
            width = read_u64(file).to_i32
            height = read_u64(file).to_i32

            num_params = case model_id
                         when 0 then 3  # SIMPLE_PINHOLE
                         when 1 then 4  # PINHOLE
                         when 2 then 4  # SIMPLE_RADIAL
                         when 3 then 5  # RADIAL
                         when 4 then 8  # OPENCV
                         when 5 then 12 # FULL_OPENCV
                         else        4
                         end

            params = Array(Float64).new(num_params) { read_f64(file) }

            model = COLMAPCameraModel.new(model_id.clamp(0, 5))
            @cameras[camera_id] = COLMAPCamera.new(camera_id, model, width, height, params)
          end
        end
      end

      # Load images from binary format
      private def load_images_bin!(path : String)
        File.open(path, "rb") do |file|
          num_images = read_u64(file)

          num_images.times do
            image_id = read_i32(file)
            qw = read_f64(file)
            qx = read_f64(file)
            qy = read_f64(file)
            qz = read_f64(file)
            tx = read_f64(file)
            ty = read_f64(file)
            tz = read_f64(file)
            camera_id = read_i32(file)

            # Read name (null-terminated string)
            name = String.build do |str|
              loop do
                c = file.read_byte
                break if c.nil? || c == 0
                str << c.chr
              end
            end

            # Skip 2D points
            num_points2d = read_u64(file)
            file.skip(num_points2d * 24)  # Each point: x(8) + y(8) + point3d_id(8)

            @images[image_id] = COLMAPImage.new(image_id, camera_id, name, qw, qx, qy, qz, tx, ty, tz)
          end
        end
      end

      # Load points3D from binary format
      private def load_points3d_bin!(path : String)
        File.open(path, "rb") do |file|
          num_points = read_u64(file)

          num_points.times do
            point_id = read_u64(file).to_i64
            x = read_f64(file)
            y = read_f64(file)
            z = read_f64(file)
            r = file.read_byte.not_nil!
            g = file.read_byte.not_nil!
            b = file.read_byte.not_nil!
            error = read_f64(file)

            # Skip track
            track_length = read_u64(file).to_i32
            file.skip(track_length * 8)  # Each track element: image_id(4) + point2d_idx(4)

            @points3d << COLMAPPoint3D.new(point_id, x, y, z, r, g, b, error, track_length)
          end
        end
      end

      private def read_u64(io : IO) : UInt64
        bytes = Bytes.new(8)
        io.read_fully(bytes)
        IO::ByteFormat::LittleEndian.decode(UInt64, bytes)
      end

      private def read_i32(io : IO) : Int32
        bytes = Bytes.new(4)
        io.read_fully(bytes)
        IO::ByteFormat::LittleEndian.decode(Int32, bytes)
      end

      private def read_f64(io : IO) : Float64
        bytes = Bytes.new(8)
        io.read_fully(bytes)
        IO::ByteFormat::LittleEndian.decode(Float64, bytes)
      end
    end

    # Convenience function
    def self.load_colmap(path : String) : COLMAPLoader
      loader = COLMAPLoader.new(path)
      loader.load!
      loader
    end
  end
end
