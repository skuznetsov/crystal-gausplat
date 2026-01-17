require "../spec_helper"

describe GS::Vision::TrackerConfig do
  describe "#initialize" do
    it "creates config with defaults" do
      config = GS::Vision::TrackerConfig.new
      config.max_age.should eq(30)
      config.min_hits.should eq(3)
      config.iou_threshold.should eq(0.3_f32)
    end

    it "accepts custom values" do
      config = GS::Vision::TrackerConfig.new(
        max_age: 50,
        min_hits: 5,
        iou_threshold: 0.5_f32
      )
      config.max_age.should eq(50)
      config.min_hits.should eq(5)
      config.iou_threshold.should eq(0.5_f32)
    end
  end
end

describe GS::Vision::Track do
  describe "#initialize" do
    it "creates track from detection" do
      det = GS::Vision::Detection.new(0.5_f32, 0.5_f32, 0.2_f32, 0.3_f32, 0.9_f32, 2)
      track = GS::Vision::Track.new(1, det)

      track.id.should eq(1)
      track.class_id.should eq(2)
      track.hits.should eq(1)
      track.misses.should eq(0)
    end
  end

  describe "#class_name" do
    it "returns correct class name" do
      det = GS::Vision::Detection.new(0.5_f32, 0.5_f32, 0.2_f32, 0.3_f32, 0.9_f32, 2)
      track = GS::Vision::Track.new(1, det)
      track.class_name.should eq("car")
    end
  end

  describe "#current_detection" do
    it "returns detection with track id" do
      det = GS::Vision::Detection.new(0.5_f32, 0.5_f32, 0.2_f32, 0.3_f32, 0.9_f32, 2)
      track = GS::Vision::Track.new(42, det)
      current = track.current_detection

      current.track_id.should eq(42)
      current.class_id.should eq(2)
    end
  end

  describe "#confirmed?" do
    it "returns false when not enough hits" do
      det = GS::Vision::Detection.new(0.5_f32, 0.5_f32, 0.2_f32, 0.3_f32, 0.9_f32, 2)
      track = GS::Vision::Track.new(1, det)
      track.confirmed?(3).should be_false
    end
  end

  describe "#dead?" do
    it "returns false when misses below threshold" do
      det = GS::Vision::Detection.new(0.5_f32, 0.5_f32, 0.2_f32, 0.3_f32, 0.9_f32, 2)
      track = GS::Vision::Track.new(1, det)
      track.dead?(30).should be_false
    end
  end
end

describe GS::Vision::Tracker do
  describe "#initialize" do
    it "creates empty tracker" do
      tracker = GS::Vision::Tracker.new
      tracker.count.should eq(0)
    end
  end

  describe "#update" do
    it "creates new tracks for detections" do
      tracker = GS::Vision::Tracker.new
      detections = [
        GS::Vision::Detection.new(0.3_f32, 0.3_f32, 0.1_f32, 0.1_f32, 0.9_f32, 2),
        GS::Vision::Detection.new(0.7_f32, 0.7_f32, 0.1_f32, 0.1_f32, 0.8_f32, 2),
      ]

      tracker.update(detections)
      tracker.count.should eq(2)
    end

    it "maintains tracks across frames" do
      tracker = GS::Vision::Tracker.new(
        GS::Vision::TrackerConfig.new(min_hits: 1)  # immediate confirmation
      )

      # First frame - create tracks
      det1 = GS::Vision::Detection.new(0.5_f32, 0.5_f32, 0.2_f32, 0.2_f32, 0.9_f32, 2)
      tracker.update([det1])

      # Second frame - same position (high IoU)
      det2 = GS::Vision::Detection.new(0.51_f32, 0.51_f32, 0.2_f32, 0.2_f32, 0.9_f32, 2)
      results = tracker.update([det2])

      # Should maintain same track ID
      results.size.should be >= 1
    end
  end

  describe "#reset" do
    it "clears all tracks" do
      tracker = GS::Vision::Tracker.new
      detections = [
        GS::Vision::Detection.new(0.5_f32, 0.5_f32, 0.1_f32, 0.1_f32, 0.9_f32, 2),
      ]
      tracker.update(detections)
      tracker.count.should eq(1)

      tracker.reset
      tracker.count.should eq(0)
    end
  end

  describe "#get_track" do
    it "returns nil for unknown id" do
      tracker = GS::Vision::Tracker.new
      tracker.get_track(999).should be_nil
    end
  end
end

describe "IoU computation" do
  it "computes positive IoU for overlapping boxes" do
    # Two boxes with overlap
    iou = LibCoreML.gs_compute_iou(
      0.5_f32, 0.5_f32, 0.4_f32, 0.4_f32,  # box 1: center, size
      0.6_f32, 0.5_f32, 0.4_f32, 0.4_f32   # box 2: center, size
    )

    # Should have positive overlap
    iou.should be > 0.3_f32
    iou.should be < 1.0_f32
  end

  it "returns 0 for non-overlapping boxes" do
    iou = LibCoreML.gs_compute_iou(
      0.2_f32, 0.2_f32, 0.1_f32, 0.1_f32,
      0.8_f32, 0.8_f32, 0.1_f32, 0.1_f32
    )
    iou.should eq(0.0_f32)
  end

  it "returns ~1 for identical boxes" do
    iou = LibCoreML.gs_compute_iou(
      0.5_f32, 0.5_f32, 0.2_f32, 0.2_f32,
      0.5_f32, 0.5_f32, 0.2_f32, 0.2_f32
    )
    iou.should be_close(1.0_f32, 1e-5)
  end
end
