require "../spec_helper"

describe GS::Video::VideoInfo do
  it "stores video metadata" do
    info = GS::Video::VideoInfo.new(
      width: 1920,
      height: 1080,
      fps: 30.0,
      duration: 10.5,
      frame_count: 315_i64,
      codec_name: "h264"
    )

    info.width.should eq(1920)
    info.height.should eq(1080)
    info.fps.should eq(30.0)
    info.duration.should eq(10.5)
    info.frame_count.should eq(315)
  end
end

describe GS::Video::Frame do
  it "stores frame data and timestamp" do
    tensor = GS::Tensor.new(480, 640, 3, device: GS::Tensor::Device::CPU)
    frame = GS::Video::Frame.new(
      data: tensor,
      timestamp: 1.5,
      frame_number: 45_i64
    )

    frame.data.shape.should eq(GS::Shape.new([480, 640, 3]))
    frame.timestamp.should eq(1.5)
    frame.frame_number.should eq(45)
  end
end

describe GS::Video::SelectionCriteria do
  describe ".new" do
    it "creates criteria with default values" do
      criteria = GS::Video::SelectionCriteria.new
      criteria.min_sharpness.should eq(100.0_f32)
      criteria.max_frames.should eq(100)
    end

    it "accepts custom values" do
      criteria = GS::Video::SelectionCriteria.new(
        min_sharpness: 200.0_f32,
        max_frames: 50
      )
      criteria.min_sharpness.should eq(200.0_f32)
      criteria.max_frames.should eq(50)
    end
  end

  describe ".fast" do
    it "returns fast preset" do
      criteria = GS::Video::SelectionCriteria.fast
      criteria.min_sharpness.should eq(50.0_f32)
      criteria.max_frames.should eq(30)
    end
  end

  describe ".high_quality" do
    it "returns high quality preset" do
      criteria = GS::Video::SelectionCriteria.high_quality
      criteria.min_sharpness.should eq(150.0_f32)
      criteria.max_frames.should eq(50)
    end
  end

  describe ".thorough" do
    it "returns thorough preset" do
      criteria = GS::Video::SelectionCriteria.thorough
      criteria.min_sharpness.should eq(80.0_f32)
      criteria.max_frames.should eq(150)
    end
  end
end

describe GS::Video::FrameMetrics do
  it "stores frame quality metrics" do
    metrics = GS::Video::FrameMetrics.new(
      sharpness: 150.0_f32,
      brightness: 0.5_f32,
      motion: 0.03_f32
    )

    metrics.sharpness.should eq(150.0_f32)
    metrics.brightness.should eq(0.5_f32)
    metrics.motion.should eq(0.03_f32)
  end
end

describe "GS::Video module helpers" do
  describe ".av_error_string" do
    it "converts error code to string" do
      # Test with a common error code
      str = GS::Video.av_error_string(-2)  # ENOENT
      str.should be_a(String)
    end
  end

  describe ".rational_to_f" do
    it "converts rational to float" do
      r = LibAV::AVRationalStruct.new(num: 30000, den: 1001)
      GS::Video.rational_to_f(r).should be_close(29.97, 0.01)
    end

    it "handles zero denominator" do
      r = LibAV::AVRationalStruct.new(num: 30, den: 0)
      GS::Video.rational_to_f(r).should eq(0.0)
    end
  end
end
