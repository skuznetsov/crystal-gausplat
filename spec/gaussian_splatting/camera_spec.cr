require "../spec_helper"

describe GS::GaussianSplatting::CameraIntrinsics do
  describe "#initialize" do
    it "creates intrinsics with specified parameters" do
      intrinsics = GS::GaussianSplatting::CameraIntrinsics.new(800, 600, 500.0_f32, 500.0_f32, 400.0_f32, 300.0_f32)
      intrinsics.width.should eq(800)
      intrinsics.height.should eq(600)
      intrinsics.fx.should eq(500.0_f32)
      intrinsics.fy.should eq(500.0_f32)
    end
  end

  describe ".from_fov" do
    it "creates intrinsics from field of view" do
      intrinsics = GS::GaussianSplatting::CameraIntrinsics.from_fov(640, 480, 60.0_f32)
      intrinsics.width.should eq(640)
      intrinsics.height.should eq(480)
      intrinsics.cx.should eq(320.0_f32)  # width / 2
      intrinsics.cy.should eq(240.0_f32)  # height / 2
    end
  end
end

describe GS::GaussianSplatting::Camera do
  describe "#initialize" do
    it "creates camera from intrinsics" do
      intrinsics = GS::GaussianSplatting::CameraIntrinsics.new(800, 600, 500.0_f32, 500.0_f32, 400.0_f32, 300.0_f32)
      cam = GS::GaussianSplatting::Camera.new(intrinsics)
      cam.width.should eq(800)
      cam.height.should eq(600)
    end
  end

  describe "#position" do
    it "returns camera position" do
      intrinsics = GS::GaussianSplatting::CameraIntrinsics.new(640, 480, 320.0_f32, 320.0_f32, 320.0_f32, 240.0_f32)
      cam = GS::GaussianSplatting::Camera.new(intrinsics)
      pos = cam.position
      # Default identity transform means position at origin
      pos[0].should be_close(0.0_f32, 1e-5)  # x
      pos[1].should be_close(0.0_f32, 1e-5)  # y
      pos[2].should be_close(0.0_f32, 1e-5)  # z
    end
  end
end
