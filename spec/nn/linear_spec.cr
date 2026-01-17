require "../spec_helper"

describe GS::NN::Linear do
  describe "#initialize" do
    it "creates linear layer with correct dimensions" do
      linear = GS::NN::Linear.new(64, 32, device: GS::Tensor::Device::CPU)
      linear.in_features.should eq(64)
      linear.out_features.should eq(32)
    end

    it "initializes weight with correct shape" do
      linear = GS::NN::Linear.new(10, 5, device: GS::Tensor::Device::CPU)
      linear.weight.shape.should eq(GS::Shape.new([5, 10]))  # [out, in]
    end

    it "initializes bias when enabled" do
      linear = GS::NN::Linear.new(10, 5, bias: true, device: GS::Tensor::Device::CPU)
      linear.bias.should_not be_nil
      linear.bias.not_nil!.shape.should eq(GS::Shape.new([5]))
    end

    it "has no bias when disabled" do
      linear = GS::NN::Linear.new(10, 5, bias: false, device: GS::Tensor::Device::CPU)
      linear.bias.should be_nil
    end
  end

  describe "#forward" do
    it "produces correct output shape" do
      linear = GS::NN::Linear.new(64, 32, device: GS::Tensor::Device::CPU)
      input = GS::Autograd::Variable.randn(8, 64, requires_grad: false, device: GS::Tensor::Device::CPU)
      output = linear.forward(input)
      output.shape.should eq(GS::Shape.new([8, 32]))
    end

    it "handles batch dimensions" do
      linear = GS::NN::Linear.new(16, 8, device: GS::Tensor::Device::CPU)
      input = GS::Autograd::Variable.randn(2, 4, 16, requires_grad: false, device: GS::Tensor::Device::CPU)
      output = linear.forward(input)
      output.shape.should eq(GS::Shape.new([2, 4, 8]))
    end
  end

  describe "#parameters" do
    it "returns weight and bias" do
      linear = GS::NN::Linear.new(10, 5, bias: true, device: GS::Tensor::Device::CPU)
      params = linear.parameters
      params.size.should eq(2)
    end

    it "returns only weight when no bias" do
      linear = GS::NN::Linear.new(10, 5, bias: false, device: GS::Tensor::Device::CPU)
      params = linear.parameters
      params.size.should eq(1)
    end
  end

  describe "GPU forward" do
    it "works on GPU" do
      if GS::Metal::Device.available?
        linear = GS::NN::Linear.new(64, 32, device: GS::Tensor::Device::GPU)
        input = GS::Autograd::Variable.randn(8, 64, requires_grad: false, device: GS::Tensor::Device::GPU)
        output = linear.forward(input)
        output.shape.should eq(GS::Shape.new([8, 32]))
        output.data.on_gpu?.should be_true
      end
    end
  end
end
