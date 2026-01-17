require "../spec_helper"

describe GS::GaussianSplatting::Gaussian3D do
  describe "#initialize" do
    it "creates gaussians with correct count" do
      g = GS::GaussianSplatting::Gaussian3D.new(100, device: GS::Tensor::Device::CPU)
      g.count.should eq(100)
    end

    it "initializes all parameter tensors" do
      g = GS::GaussianSplatting::Gaussian3D.new(50, device: GS::Tensor::Device::CPU)
      g.position.shape.should eq(GS::Shape.new([50, 3]))
      g.scale.shape.should eq(GS::Shape.new([50, 3]))
      g.rotation.shape.should eq(GS::Shape.new([50, 4]))
      g.opacity.shape.should eq(GS::Shape.new([50, 1]))
      g.sh_coeffs.shape[0].should eq(50)
    end
  end

  describe ".from_points" do
    it "creates gaussians from point cloud" do
      points = GS::Tensor.new(200, 3, device: GS::Tensor::Device::CPU)
      points.cpu_data.not_nil!.map! { Random.rand(-1.0_f32..1.0_f32) }

      g = GS::GaussianSplatting::Gaussian3D.from_points(points)
      g.count.should eq(200)
    end
  end

  describe "#clone_gaussians" do
    it "creates copy with specified indices" do
      g = GS::GaussianSplatting::Gaussian3D.new(100, device: GS::Tensor::Device::CPU)
      indices = [0, 5, 10, 15]
      cloned = g.clone_gaussians(indices)
      cloned.count.should eq(4)
    end
  end

  describe "#concat!" do
    it "concatenates two gaussian sets" do
      g1 = GS::GaussianSplatting::Gaussian3D.new(50, device: GS::Tensor::Device::CPU)
      g2 = GS::GaussianSplatting::Gaussian3D.new(30, device: GS::Tensor::Device::CPU)
      g1.concat!(g2)
      g1.count.should eq(80)
    end
  end

  describe "#remove!" do
    it "removes gaussians at indices" do
      g = GS::GaussianSplatting::Gaussian3D.new(100, device: GS::Tensor::Device::CPU)
      to_remove = Set{0, 10, 20, 30, 40}
      g.remove!(to_remove)
      g.count.should eq(95)
    end
  end

  describe "#parameters" do
    it "returns all trainable parameters" do
      g = GS::GaussianSplatting::Gaussian3D.new(10, device: GS::Tensor::Device::CPU)
      params = g.parameters
      params.size.should eq(5)  # position, scale, rotation, opacity, sh_coeffs
    end
  end

  describe "#stats" do
    it "computes statistics including mean opacity" do
      g = GS::GaussianSplatting::Gaussian3D.new(10, device: GS::Tensor::Device::CPU)
      stats = g.stats
      stats[:mean_opacity].should be >= 0.0
      stats[:mean_opacity].should be <= 1.0
      stats[:count].should eq(10)
    end
  end
end
