require "../spec_helper"

describe GS::Export::MarchingCubes do
  describe "#initialize" do
    it "creates with specified resolution" do
      mc = GS::Export::MarchingCubes.new(64)
      mc.resolution.should eq(64)
    end
  end

  describe "#sample_from_points" do
    it "samples density field from points" do
      mc = GS::Export::MarchingCubes.new(32)
      points = GS::Tensor.new(100, 3, device: GS::Tensor::Device::CPU)
      data = points.cpu_data.not_nil!

      # Create sphere of points
      100.times do |i|
        theta = (i.to_f32 / 100.0_f32) * 2.0_f32 * Math::PI
        phi = Math.acos(2.0_f32 * Random.rand - 1.0_f32)
        r = 0.3_f32
        data[i * 3] = (r * Math.sin(phi) * Math.cos(theta)).to_f32
        data[i * 3 + 1] = (r * Math.sin(phi) * Math.sin(theta)).to_f32
        data[i * 3 + 2] = (r * Math.cos(phi)).to_f32
      end

      mc.sample_from_points(points, sigma: 0.05_f32)
      # Should have non-zero density values
    end
  end

  describe "#extract" do
    it "extracts mesh from density field" do
      mc = GS::Export::MarchingCubes.new(32)
      points = GS::Tensor.new(100, 3, device: GS::Tensor::Device::CPU)
      data = points.cpu_data.not_nil!

      # Create sphere
      100.times do |i|
        theta = (i.to_f32 / 100.0_f32) * 2.0_f32 * Math::PI
        phi = Math.acos(2.0_f32 * Random.rand - 1.0_f32)
        r = 0.3_f32
        data[i * 3] = (r * Math.sin(phi) * Math.cos(theta)).to_f32
        data[i * 3 + 1] = (r * Math.sin(phi) * Math.sin(theta)).to_f32
        data[i * 3 + 2] = (r * Math.cos(phi)).to_f32
      end

      mc.sample_from_points(points, sigma: 0.05_f32)
      mesh = mc.extract(0.3_f32)

      mesh.vertex_count.should be > 0
      mesh.triangle_count.should be > 0
    end
  end

  describe ".extract_from_points" do
    it "extracts mesh in one call" do
      points = GS::Tensor.new(50, 3, device: GS::Tensor::Device::CPU)
      data = points.cpu_data.not_nil!

      50.times do |i|
        theta = (i.to_f32 / 50.0_f32) * 2.0_f32 * Math::PI
        phi = Math.acos(2.0_f32 * Random.rand - 1.0_f32)
        r = 0.3_f32
        data[i * 3] = (r * Math.sin(phi) * Math.cos(theta)).to_f32
        data[i * 3 + 1] = (r * Math.sin(phi) * Math.sin(theta)).to_f32
        data[i * 3 + 2] = (r * Math.cos(phi)).to_f32
      end

      mesh = GS::Export::MarchingCubes.extract_from_points(points, resolution: 32, sigma: 0.05_f32, use_gpu: false)
      mesh.vertex_count.should be > 0
    end
  end
end
