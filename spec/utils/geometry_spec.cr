require "../spec_helper"

describe GS::Utils::Vec3 do
  describe "#initialize" do
    it "creates vector with coordinates" do
      v = GS::Utils::Vec3.new(1.0, 2.0, 3.0)
      v.x.should eq(1.0)
      v.y.should eq(2.0)
      v.z.should eq(3.0)
    end
  end

  describe "#+" do
    it "adds vectors" do
      v1 = GS::Utils::Vec3.new(1.0, 2.0, 3.0)
      v2 = GS::Utils::Vec3.new(4.0, 5.0, 6.0)
      result = v1 + v2
      result.x.should eq(5.0)
      result.y.should eq(7.0)
      result.z.should eq(9.0)
    end
  end

  describe "#-" do
    it "subtracts vectors" do
      v1 = GS::Utils::Vec3.new(5.0, 7.0, 9.0)
      v2 = GS::Utils::Vec3.new(1.0, 2.0, 3.0)
      result = v1 - v2
      result.x.should eq(4.0)
      result.y.should eq(5.0)
      result.z.should eq(6.0)
    end
  end

  describe "#*" do
    it "scales vector" do
      v = GS::Utils::Vec3.new(1.0, 2.0, 3.0)
      result = v * 2.0
      result.x.should eq(2.0)
      result.y.should eq(4.0)
      result.z.should eq(6.0)
    end
  end

  describe "#dot" do
    it "computes dot product" do
      v1 = GS::Utils::Vec3.new(1.0, 2.0, 3.0)
      v2 = GS::Utils::Vec3.new(4.0, 5.0, 6.0)
      result = v1.dot(v2)
      result.should eq(32.0)  # 1*4 + 2*5 + 3*6
    end
  end

  describe "#cross" do
    it "computes cross product" do
      v1 = GS::Utils::Vec3.new(1.0, 0.0, 0.0)
      v2 = GS::Utils::Vec3.new(0.0, 1.0, 0.0)
      result = v1.cross(v2)
      result.x.should be_close(0.0, 1e-6)
      result.y.should be_close(0.0, 1e-6)
      result.z.should be_close(1.0, 1e-6)
    end
  end

  describe "#length" do
    it "computes vector length" do
      v = GS::Utils::Vec3.new(3.0, 4.0, 0.0)
      v.length.should eq(5.0)
    end
  end

  describe "#normalize" do
    it "returns unit vector" do
      v = GS::Utils::Vec3.new(3.0, 4.0, 0.0)
      n = v.normalize
      n.length.should be_close(1.0, 1e-6)
    end
  end

  describe "#distance_to" do
    it "computes distance between points" do
      v1 = GS::Utils::Vec3.new(0.0, 0.0, 0.0)
      v2 = GS::Utils::Vec3.new(3.0, 4.0, 0.0)
      v1.distance_to(v2).should eq(5.0)
    end
  end
end

describe GS::Utils::Geometry do
  describe ".kabsch_align" do
    it "aligns point clouds" do
      source = [
        GS::Utils::Vec3.new(0.0, 0.0, 0.0),
        GS::Utils::Vec3.new(1.0, 0.0, 0.0),
        GS::Utils::Vec3.new(0.0, 1.0, 0.0),
        GS::Utils::Vec3.new(0.0, 0.0, 1.0),
      ]

      # Target is rotated 90 degrees around Z and translated
      target = [
        GS::Utils::Vec3.new(5.0, 5.0, 0.0),
        GS::Utils::Vec3.new(5.0, 6.0, 0.0),
        GS::Utils::Vec3.new(4.0, 5.0, 0.0),
        GS::Utils::Vec3.new(5.0, 5.0, 1.0),
      ]

      aligned = GS::Utils::Geometry.kabsch_align(source, target)
      aligned.size.should eq(4)
    end
  end

  describe ".kabsch_rmsd" do
    it "computes RMSD between point clouds" do
      source = [
        GS::Utils::Vec3.new(0.0, 0.0, 0.0),
        GS::Utils::Vec3.new(1.0, 0.0, 0.0),
        GS::Utils::Vec3.new(0.0, 1.0, 0.0),
      ]

      target = [
        GS::Utils::Vec3.new(5.0, 5.0, 0.0),
        GS::Utils::Vec3.new(6.0, 5.0, 0.0),
        GS::Utils::Vec3.new(5.0, 6.0, 0.0),
      ]

      rmsd = GS::Utils::Geometry.kabsch_rmsd(source, target)
      rmsd.should be_close(0.0, 1e-3)
    end
  end

  describe ".icp" do
    it "performs iterative closest point" do
      source = [
        GS::Utils::Vec3.new(0.0, 0.0, 0.0),
        GS::Utils::Vec3.new(1.0, 0.0, 0.0),
        GS::Utils::Vec3.new(0.0, 1.0, 0.0),
        GS::Utils::Vec3.new(0.0, 0.0, 1.0),
      ]

      # Add small noise to target
      target = source.map { |p| GS::Utils::Vec3.new(p.x + Random.rand * 0.1, p.y + Random.rand * 0.1, p.z) }

      result = GS::Utils::Geometry.icp(source, target, iterations: 10)
      result.size.should eq(4)
    end
  end
end
