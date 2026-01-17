require "../spec_helper"

describe GS::Vision::Detection do
  describe "#initialize" do
    it "creates detection with all properties" do
      det = GS::Vision::Detection.new(
        x: 0.5_f32,
        y: 0.5_f32,
        width: 0.2_f32,
        height: 0.3_f32,
        confidence: 0.9_f32,
        class_id: 2,  # car
        track_id: 5
      )

      det.x.should eq(0.5_f32)
      det.y.should eq(0.5_f32)
      det.width.should eq(0.2_f32)
      det.height.should eq(0.3_f32)
      det.confidence.should eq(0.9_f32)
      det.class_id.should eq(2)
      det.track_id.should eq(5)
    end
  end

  describe "#class_name" do
    it "returns correct class name for car" do
      det = GS::Vision::Detection.new(0.5_f32, 0.5_f32, 0.2_f32, 0.3_f32, 0.9_f32, 2)
      det.class_name.should eq("car")
    end

    it "returns correct class name for person" do
      det = GS::Vision::Detection.new(0.5_f32, 0.5_f32, 0.2_f32, 0.3_f32, 0.9_f32, 0)
      det.class_name.should eq("person")
    end
  end

  describe "#is_vehicle?" do
    it "returns true for car" do
      det = GS::Vision::Detection.new(0.5_f32, 0.5_f32, 0.2_f32, 0.3_f32, 0.9_f32, 2)
      det.is_vehicle?.should be_true
    end

    it "returns true for truck" do
      det = GS::Vision::Detection.new(0.5_f32, 0.5_f32, 0.2_f32, 0.3_f32, 0.9_f32, 7)
      det.is_vehicle?.should be_true
    end

    it "returns false for person" do
      det = GS::Vision::Detection.new(0.5_f32, 0.5_f32, 0.2_f32, 0.3_f32, 0.9_f32, 0)
      det.is_vehicle?.should be_false
    end
  end

  describe "#to_pixels" do
    it "converts normalized coords to pixels" do
      det = GS::Vision::Detection.new(0.5_f32, 0.5_f32, 0.2_f32, 0.1_f32, 0.9_f32, 2)
      x, y, w, h = det.to_pixels(1920, 1080)

      # center at 0.5 = 960, 540
      # width 0.2 = 384, height 0.1 = 108
      # top-left = 960-192, 540-54 = 768, 486
      x.should eq(768)
      y.should eq(486)
      w.should eq(384)
      h.should eq(108)
    end
  end

  describe "#bbox" do
    it "returns bounding box corners" do
      det = GS::Vision::Detection.new(0.5_f32, 0.5_f32, 0.2_f32, 0.2_f32, 0.9_f32, 2)
      x1, y1, x2, y2 = det.bbox

      x1.should eq(0.4_f32)
      y1.should eq(0.4_f32)
      x2.should eq(0.6_f32)
      y2.should eq(0.6_f32)
    end
  end
end

describe GS::Vision::DetectorConfig do
  describe ".new" do
    it "creates config with defaults" do
      config = GS::Vision::DetectorConfig.new
      config.confidence_threshold.should eq(0.5_f32)
      config.nms_threshold.should eq(0.45_f32)
      config.max_detections.should eq(100)
    end
  end

  describe ".vehicles" do
    it "returns vehicle detection preset" do
      config = GS::Vision::DetectorConfig.vehicles
      config.filter_classes.should eq(GS::Vision::VEHICLE_CLASSES)
    end
  end

  describe ".high_precision" do
    it "returns high precision preset" do
      config = GS::Vision::DetectorConfig.high_precision
      config.confidence_threshold.should eq(0.7_f32)
    end
  end
end
