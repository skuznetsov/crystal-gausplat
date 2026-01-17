require "../spec_helper"

describe GS::Tensor do
  describe "#initialize" do
    it "creates tensor with shape" do
      t = GS::Tensor.new(2, 3, 4, device: GS::Tensor::Device::CPU)
      t.shape.should eq(GS::Shape.new([2, 3, 4]))
      t.numel.should eq(24)
    end

    it "creates tensor from shape object" do
      shape = GS::Shape.new([5, 10])
      t = GS::Tensor.new(shape, GS::DType::F32, GS::Tensor::Device::CPU)
      t.shape.should eq(shape)
    end
  end

  describe "#zeros" do
    it "creates zero-filled tensor" do
      t = GS::Tensor.zeros(3, 3, device: GS::Tensor::Device::CPU)
      data = t.cpu_data.not_nil!
      data.all? { |x| x == 0.0_f32 }.should be_true
    end
  end

  describe "#ones" do
    it "creates one-filled tensor" do
      t = GS::Tensor.ones(2, 4, device: GS::Tensor::Device::CPU)
      data = t.cpu_data.not_nil!
      data.all? { |x| x == 1.0_f32 }.should be_true
    end
  end

  describe "#randn" do
    it "creates tensor with random normal values" do
      t = GS::Tensor.randn(100, 100, device: GS::Tensor::Device::CPU)
      data = t.cpu_data.not_nil!

      # Check mean is approximately 0
      mean = data.sum / data.size
      mean.abs.should be < 0.1

      # Check we have both positive and negative values
      has_positive = data.any? { |x| x > 0 }
      has_negative = data.any? { |x| x < 0 }
      has_positive.should be_true
      has_negative.should be_true
    end
  end

  describe "device transfer" do
    it "transfers CPU to GPU and back" do
      if GS::Metal::Device.available?
        cpu_tensor = GS::Tensor.new(4, 4, device: GS::Tensor::Device::CPU)
        cpu_tensor.cpu_data.not_nil!.map_with_index! { |_, i| i.to_f32 }

        gpu_tensor = cpu_tensor.to_gpu
        gpu_tensor.on_gpu?.should be_true

        back_to_cpu = gpu_tensor.to_cpu
        back_to_cpu.on_cpu?.should be_true

        # Verify data integrity
        original = cpu_tensor.cpu_data.not_nil!
        restored = back_to_cpu.cpu_data.not_nil!
        16.times do |i|
          restored[i].should eq(original[i])
        end
      end
    end
  end
end
