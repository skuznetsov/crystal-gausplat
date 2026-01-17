# Camera model for Gaussian Splatting
# Pinhole camera with intrinsics and extrinsics

require "../core/tensor"

module GS
  module GaussianSplatting
    # Camera intrinsic parameters
    struct CameraIntrinsics
      property width : Int32
      property height : Int32
      property fx : Float32  # Focal length x (pixels)
      property fy : Float32  # Focal length y (pixels)
      property cx : Float32  # Principal point x
      property cy : Float32  # Principal point y

      def initialize(@width : Int32, @height : Int32, @fx : Float32, @fy : Float32, @cx : Float32, @cy : Float32)
      end

      # Create from field of view
      def self.from_fov(width : Int32, height : Int32, fov_x : Float32, fov_y : Float32? = nil) : CameraIntrinsics
        fx = width / (2.0_f32 * Math.tan(fov_x / 2.0_f32))
        fy = fov_y ? height / (2.0_f32 * Math.tan(fov_y / 2.0_f32)) : fx
        cx = width / 2.0_f32
        cy = height / 2.0_f32
        new(width, height, fx, fy, cx, cy)
      end

      # Field of view in radians
      def fov_x : Float32
        2.0_f32 * Math.atan(@width / (2.0_f32 * @fx))
      end

      def fov_y : Float32
        2.0_f32 * Math.atan(@height / (2.0_f32 * @fy))
      end

      # Tan of half FOV (used in frustum culling)
      def tan_half_fov_x : Float32
        Math.tan(fov_x / 2.0_f32)
      end

      def tan_half_fov_y : Float32
        Math.tan(fov_y / 2.0_f32)
      end

      # Projection matrix (3x3)
      def projection_matrix : Tensor
        mat = Tensor.zeros(3, 3, device: Tensor::Device::CPU)
        data = mat.cpu_data.not_nil!
        data[0] = @fx;  data[1] = 0;     data[2] = @cx
        data[3] = 0;    data[4] = @fy;   data[5] = @cy
        data[6] = 0;    data[7] = 0;     data[8] = 1
        mat
      end
    end

    # Full camera with pose
    class Camera
      property intrinsics : CameraIntrinsics

      # Camera extrinsics
      # world_to_camera: transforms world coordinates to camera coordinates
      # camera_to_world: transforms camera coordinates to world coordinates (camera pose)
      property world_to_camera : Tensor  # [4, 4]
      property camera_to_world : Tensor  # [4, 4]

      # Optional: associated image for training
      property image : Tensor?
      property image_path : String?

      def initialize(@intrinsics : CameraIntrinsics)
        # Default: identity transform (camera at origin looking down -Z)
        @world_to_camera = Tensor.eye(4, device: Tensor::Device::CPU)
        @camera_to_world = Tensor.eye(4, device: Tensor::Device::CPU)
        @image = nil
        @image_path = nil
      end

      # Width and height shortcuts
      def width : Int32
        @intrinsics.width
      end

      def height : Int32
        @intrinsics.height
      end

      # Set camera pose from position and look-at point
      def look_at(eye : {Float32, Float32, Float32}, target : {Float32, Float32, Float32}, up : {Float32, Float32, Float32} = {0.0_f32, 1.0_f32, 0.0_f32}) : Nil
        # Forward vector (camera looks down -Z in camera space)
        forward = normalize({target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]})

        # Right vector
        right = normalize(cross(forward, up))

        # Recompute up to ensure orthogonality
        up_corrected = cross(right, forward)

        # Build rotation matrix (columns are right, up, -forward)
        # camera_to_world rotation part
        @camera_to_world.to_cpu!
        data = @camera_to_world.cpu_data.not_nil!

        # Column 0: right
        data[0] = right[0]; data[4] = right[1]; data[8] = right[2]
        # Column 1: up
        data[1] = up_corrected[0]; data[5] = up_corrected[1]; data[9] = up_corrected[2]
        # Column 2: -forward (camera looks down -Z)
        data[2] = -forward[0]; data[6] = -forward[1]; data[10] = -forward[2]
        # Column 3: translation (eye position)
        data[3] = eye[0]; data[7] = eye[1]; data[11] = eye[2]
        # Row 3: [0, 0, 0, 1]
        data[12] = 0; data[13] = 0; data[14] = 0; data[15] = 1

        # Compute world_to_camera as inverse
        @world_to_camera = invert_rigid_transform(@camera_to_world)
      end

      # Set from world_to_camera matrix directly
      def set_world_to_camera(matrix : Tensor) : Nil
        @world_to_camera = matrix.clone
        @camera_to_world = invert_rigid_transform(matrix)
      end

      # Set from camera_to_world matrix directly
      def set_camera_to_world(matrix : Tensor) : Nil
        @camera_to_world = matrix.clone
        @world_to_camera = invert_rigid_transform(matrix)
      end

      # Get camera position in world coordinates
      def position : {Float32, Float32, Float32}
        @camera_to_world.to_cpu! if @camera_to_world.on_gpu?
        data = @camera_to_world.cpu_data.not_nil!
        {data[3], data[7], data[11]}
      end

      # Get camera forward direction in world coordinates
      def forward : {Float32, Float32, Float32}
        @camera_to_world.to_cpu! if @camera_to_world.on_gpu?
        data = @camera_to_world.cpu_data.not_nil!
        # Forward is -Z column
        {-data[2], -data[6], -data[10]}
      end

      # Full projection matrix (4x4) for rendering
      # Projects from world space to clip space
      def full_projection_matrix(near : Float32 = 0.1_f32, far : Float32 = 100.0_f32) : Tensor
        # Perspective projection matrix
        fx = @intrinsics.fx
        fy = @intrinsics.fy
        cx = @intrinsics.cx
        cy = @intrinsics.cy
        w = @intrinsics.width.to_f32
        h = @intrinsics.height.to_f32

        # OpenGL-style projection matrix
        proj = Tensor.zeros(4, 4, device: Tensor::Device::CPU)
        data = proj.cpu_data.not_nil!

        data[0] = 2.0_f32 * fx / w
        data[5] = 2.0_f32 * fy / h
        data[2] = -(2.0_f32 * cx / w - 1.0_f32)
        data[6] = -(2.0_f32 * cy / h - 1.0_f32)
        data[10] = -(far + near) / (far - near)
        data[14] = -1.0_f32
        data[11] = -2.0_f32 * far * near / (far - near)

        # Multiply with world_to_camera
        matmul_4x4(proj, @world_to_camera)
      end

      # Project 3D point to 2D
      def project(point_world : {Float32, Float32, Float32}) : {Float32, Float32, Float32}?
        # Transform to camera space
        @world_to_camera.to_cpu! if @world_to_camera.on_gpu?
        w2c = @world_to_camera.cpu_data.not_nil!

        x = point_world[0]
        y = point_world[1]
        z = point_world[2]

        # Apply world_to_camera transform
        cx = w2c[0]*x + w2c[1]*y + w2c[2]*z + w2c[3]
        cy = w2c[4]*x + w2c[5]*y + w2c[6]*z + w2c[7]
        cz = w2c[8]*x + w2c[9]*y + w2c[10]*z + w2c[11]

        # Behind camera
        return nil if cz <= 0

        # Project to image plane
        px = @intrinsics.fx * cx / cz + @intrinsics.cx
        py = @intrinsics.fy * cy / cz + @intrinsics.cy

        {px, py, cz}
      end

      private def normalize(v : {Float32, Float32, Float32}) : {Float32, Float32, Float32}
        len = Math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
        return {0.0_f32, 0.0_f32, 1.0_f32} if len < 1e-8_f32
        {v[0]/len, v[1]/len, v[2]/len}
      end

      private def cross(a : {Float32, Float32, Float32}, b : {Float32, Float32, Float32}) : {Float32, Float32, Float32}
        {
          a[1]*b[2] - a[2]*b[1],
          a[2]*b[0] - a[0]*b[2],
          a[0]*b[1] - a[1]*b[0]
        }
      end

      # Invert rigid transform (rotation + translation)
      private def invert_rigid_transform(mat : Tensor) : Tensor
        mat.to_cpu! if mat.on_gpu?
        m = mat.cpu_data.not_nil!

        result = Tensor.zeros(4, 4, device: Tensor::Device::CPU)
        r = result.cpu_data.not_nil!

        # R^T
        r[0] = m[0]; r[1] = m[4]; r[2] = m[8]
        r[4] = m[1]; r[5] = m[5]; r[6] = m[9]
        r[8] = m[2]; r[9] = m[6]; r[10] = m[10]

        # -R^T * t
        tx = m[3]; ty = m[7]; tz = m[11]
        r[3] = -(r[0]*tx + r[1]*ty + r[2]*tz)
        r[7] = -(r[4]*tx + r[5]*ty + r[6]*tz)
        r[11] = -(r[8]*tx + r[9]*ty + r[10]*tz)

        r[12] = 0; r[13] = 0; r[14] = 0; r[15] = 1

        result
      end

      private def matmul_4x4(a : Tensor, b : Tensor) : Tensor
        a.to_cpu! if a.on_gpu?
        b.to_cpu! if b.on_gpu?

        result = Tensor.zeros(4, 4, device: Tensor::Device::CPU)
        ad = a.cpu_data.not_nil!
        bd = b.cpu_data.not_nil!
        rd = result.cpu_data.not_nil!

        4.times do |i|
          4.times do |j|
            sum = 0.0_f32
            4.times { |k| sum += ad[i*4 + k] * bd[k*4 + j] }
            rd[i*4 + j] = sum
          end
        end

        result
      end
    end

    # Camera set for multi-view training
    class CameraSet
      property cameras : Array(Camera)

      def initialize
        @cameras = Array(Camera).new
      end

      def <<(camera : Camera) : self
        @cameras << camera
        self
      end

      def size : Int32
        @cameras.size
      end

      def [](index : Int32) : Camera
        @cameras[index]
      end

      def each(&block : Camera -> Nil) : Nil
        @cameras.each { |c| yield c }
      end

      def sample : Camera
        @cameras.sample
      end

      # Load cameras from COLMAP format
      def self.from_colmap(images_txt_path : String, cameras_txt_path : String) : CameraSet
        set = new

        # Parse cameras.txt for intrinsics
        intrinsics_map = Hash(Int32, CameraIntrinsics).new

        File.each_line(cameras_txt_path) do |line|
          next if line.starts_with?("#") || line.empty?
          parts = line.split
          next unless parts.size >= 5

          camera_id = parts[0].to_i
          model = parts[1]
          width = parts[2].to_i
          height = parts[3].to_i

          case model
          when "PINHOLE"
            fx = parts[4].to_f32
            fy = parts[5].to_f32
            cx = parts[6].to_f32
            cy = parts[7].to_f32
            intrinsics_map[camera_id] = CameraIntrinsics.new(width, height, fx, fy, cx, cy)
          when "SIMPLE_PINHOLE"
            f = parts[4].to_f32
            cx = parts[5].to_f32
            cy = parts[6].to_f32
            intrinsics_map[camera_id] = CameraIntrinsics.new(width, height, f, f, cx, cy)
          end
        end

        # Parse images.txt for poses
        File.each_line(images_txt_path) do |line|
          next if line.starts_with?("#") || line.empty?
          parts = line.split
          next unless parts.size >= 10

          # Format: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
          qw = parts[1].to_f32
          qx = parts[2].to_f32
          qy = parts[3].to_f32
          qz = parts[4].to_f32
          tx = parts[5].to_f32
          ty = parts[6].to_f32
          tz = parts[7].to_f32
          camera_id = parts[8].to_i
          image_name = parts[9]

          intrinsics = intrinsics_map[camera_id]? || next

          camera = Camera.new(intrinsics)
          camera.image_path = image_name

          # COLMAP stores world_to_camera: R and t such that X_cam = R * X_world + t
          # Build world_to_camera matrix from quaternion and translation
          mat = Tensor.zeros(4, 4, device: Tensor::Device::CPU)
          data = mat.cpu_data.not_nil!

          # Quaternion to rotation matrix
          r = quat_to_rotation_matrix(qw, qx, qy, qz)
          data[0] = r[0]; data[1] = r[1]; data[2] = r[2]; data[3] = tx
          data[4] = r[3]; data[5] = r[4]; data[6] = r[5]; data[7] = ty
          data[8] = r[6]; data[9] = r[7]; data[10] = r[8]; data[11] = tz
          data[12] = 0; data[13] = 0; data[14] = 0; data[15] = 1

          camera.set_world_to_camera(mat)
          set << camera
        end

        set
      end

      private def self.quat_to_rotation_matrix(w : Float32, x : Float32, y : Float32, z : Float32) : StaticArray(Float32, 9)
        # Normalize quaternion
        norm = Math.sqrt(w*w + x*x + y*y + z*z)
        w /= norm; x /= norm; y /= norm; z /= norm

        result = StaticArray(Float32, 9).new(0.0_f32)
        result[0] = 1 - 2*y*y - 2*z*z
        result[1] = 2*x*y - 2*z*w
        result[2] = 2*x*z + 2*y*w
        result[3] = 2*x*y + 2*z*w
        result[4] = 1 - 2*x*x - 2*z*z
        result[5] = 2*y*z - 2*x*w
        result[6] = 2*x*z - 2*y*w
        result[7] = 2*y*z + 2*x*w
        result[8] = 1 - 2*x*x - 2*y*y
        result
      end
    end
  end
end
