require "spec"
require "../src/core/buffer"
require "../src/core/shape"
require "../src/core/tensor"
require "../src/metal/device"
require "../src/metal/dispatch"
require "../src/autograd/variable"
require "../src/autograd/grad_fn"
require "../src/optim/adam"
require "../src/ops/loss"
require "../src/nn/gpu_ops"
require "../src/nn/linear"
require "../src/nn/layernorm"
require "../src/nn/attention"
require "../src/gaussian_splatting/gaussian"
require "../src/gaussian_splatting/camera"
require "../src/export/marching_cubes"
require "../src/utils/geometry"

# Initialize Metal once for all tests
GS::Metal::Device.init!

# Helper to check tensor values approximately equal
def tensor_approx_equal(a : GS::Tensor, b : GS::Tensor, eps : Float32 = 1e-5_f32) : Bool
  return false unless a.shape == b.shape
  a_data = a.on_cpu? ? a.cpu_data.not_nil! : a.to_cpu.cpu_data.not_nil!
  b_data = b.on_cpu? ? b.cpu_data.not_nil! : b.to_cpu.cpu_data.not_nil!
  a.numel.times do |i|
    return false if (a_data[i] - b_data[i]).abs > eps
  end
  true
end

# Helper to create random tensor
def random_tensor(shape : Array(Int32), device : GS::Tensor::Device = GS::Tensor::Device::CPU) : GS::Tensor
  t = GS::Tensor.new(GS::Shape.new(shape), GS::DType::Float32, device)
  if t.on_cpu?
    t.cpu_data.not_nil!.map! { Random.rand(-1.0_f32..1.0_f32) }
  end
  t
end
