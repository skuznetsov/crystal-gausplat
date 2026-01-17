require "../spec_helper"

describe GS::Shape do
  describe "#initialize" do
    it "creates shape from dimensions" do
      s = GS::Shape.new([2, 3, 4])
      s.ndim.should eq(3)
      s[0].should eq(2)
      s[1].should eq(3)
      s[2].should eq(4)
    end
  end

  describe "#numel" do
    it "returns total number of elements" do
      s = GS::Shape.new([2, 3, 4])
      s.numel.should eq(24)
    end

    it "handles scalar shape" do
      s = GS::Shape.new([1])
      s.numel.should eq(1)
    end
  end

  describe "#==" do
    it "compares equal shapes" do
      s1 = GS::Shape.new([2, 3])
      s2 = GS::Shape.new([2, 3])
      (s1 == s2).should be_true
    end

    it "compares different shapes" do
      s1 = GS::Shape.new([2, 3])
      s2 = GS::Shape.new([3, 2])
      (s1 == s2).should be_false
    end
  end

  describe "#broadcast_with" do
    it "broadcasts compatible shapes" do
      s1 = GS::Shape.new([2, 3, 4])
      s2 = GS::Shape.new([1, 3, 1])
      result = s1.broadcast_with(s2)
      result.should eq(GS::Shape.new([2, 3, 4]))
    end

    it "broadcasts with different ranks" do
      s1 = GS::Shape.new([3, 4])
      s2 = GS::Shape.new([2, 3, 4])
      result = s2.broadcast_with(s1)
      result.should eq(GS::Shape.new([2, 3, 4]))
    end
  end

  describe "#broadcastable_with?" do
    it "returns true for compatible shapes" do
      s1 = GS::Shape.new([2, 3, 4])
      s2 = GS::Shape.new([1, 3, 1])
      s1.broadcastable_with?(s2).should be_true
    end

    it "returns false for incompatible shapes" do
      s1 = GS::Shape.new([2, 3])
      s2 = GS::Shape.new([4, 5])
      s1.broadcastable_with?(s2).should be_false
    end
  end
end
