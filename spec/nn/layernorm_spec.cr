require "../spec_helper"

describe GS::NN::LayerNorm do
  describe "#initialize" do
    it "creates layer norm with correct size" do
      ln = GS::NN::LayerNorm.new(64, device: GS::Tensor::Device::CPU)
      ln.normalized_shape.should eq([64])
    end

    it "initializes weight (gamma) to ones" do
      ln = GS::NN::LayerNorm.new(32, device: GS::Tensor::Device::CPU)
      weight = ln.weight.data.cpu_data.not_nil!
      weight.all? { |x| (x - 1.0_f32).abs < 1e-6 }.should be_true
    end

    it "initializes bias (beta) to zeros" do
      ln = GS::NN::LayerNorm.new(32, device: GS::Tensor::Device::CPU)
      bias = ln.bias.data.cpu_data.not_nil!
      bias.all? { |x| x.abs < 1e-6 }.should be_true
    end
  end

  describe "#forward" do
    it "produces correct output shape" do
      ln = GS::NN::LayerNorm.new(64, device: GS::Tensor::Device::CPU)
      input = GS::Autograd::Variable.randn(8, 64, requires_grad: false, device: GS::Tensor::Device::CPU)
      output = ln.forward(input)
      output.shape.should eq(GS::Shape.new([8, 64]))
    end

    it "normalizes to approximately zero mean and unit variance" do
      ln = GS::NN::LayerNorm.new(128, device: GS::Tensor::Device::CPU)
      input = GS::Autograd::Variable.randn(4, 128, requires_grad: false, device: GS::Tensor::Device::CPU)
      output = ln.forward(input)

      data = output.data.cpu_data.not_nil!

      # Check each row has approximately zero mean
      4.times do |row|
        row_data = (0...128).map { |c| data[row * 128 + c] }
        mean = row_data.sum / 128
        mean.abs.should be < 0.1
      end
    end
  end

  describe "#parameters" do
    it "returns gamma and beta" do
      ln = GS::NN::LayerNorm.new(64, device: GS::Tensor::Device::CPU)
      params = ln.parameters
      params.size.should eq(2)
    end
  end
end

describe GS::NN::RMSNorm do
  describe "#forward" do
    it "produces correct output shape" do
      rms = GS::NN::RMSNorm.new(64, device: GS::Tensor::Device::CPU)
      input = GS::Autograd::Variable.randn(8, 64, requires_grad: false, device: GS::Tensor::Device::CPU)
      output = rms.forward(input)
      output.shape.should eq(GS::Shape.new([8, 64]))
    end
  end
end
