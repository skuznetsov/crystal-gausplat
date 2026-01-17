require "../spec_helper"

describe GS::NN::MultiHeadAttention do
  describe "#initialize" do
    it "creates attention with correct dimensions" do
      mha = GS::NN::MultiHeadAttention.new(embed_dim: 64, num_heads: 4, device: GS::Tensor::Device::CPU)
      mha.embed_dim.should eq(64)
      mha.num_heads.should eq(4)
      mha.head_dim.should eq(16)  # 64 / 4
    end

    it "validates head_dim divides embed_dim" do
      # 64 / 4 = 16, valid
      mha = GS::NN::MultiHeadAttention.new(embed_dim: 64, num_heads: 4, device: GS::Tensor::Device::CPU)
      mha.head_dim.should eq(16)
    end
  end

  describe "#self_attention" do
    it "produces correct output shape" do
      mha = GS::NN::MultiHeadAttention.new(embed_dim: 64, num_heads: 4, device: GS::Tensor::Device::CPU)
      input = GS::Autograd::Variable.randn(2, 8, 64, requires_grad: false, device: GS::Tensor::Device::CPU)
      output = mha.self_attention(input)
      output.shape.should eq(GS::Shape.new([2, 8, 64]))
    end
  end

  describe "#forward" do
    it "computes cross-attention correctly" do
      mha = GS::NN::MultiHeadAttention.new(embed_dim: 32, num_heads: 2, device: GS::Tensor::Device::CPU)
      query = GS::Autograd::Variable.randn(1, 4, 32, requires_grad: false, device: GS::Tensor::Device::CPU)
      key = GS::Autograd::Variable.randn(1, 6, 32, requires_grad: false, device: GS::Tensor::Device::CPU)
      value = GS::Autograd::Variable.randn(1, 6, 32, requires_grad: false, device: GS::Tensor::Device::CPU)

      output = mha.forward(query, key, value)
      output.shape.should eq(GS::Shape.new([1, 4, 32]))  # Same seq_len as query
    end
  end

  describe "#parameters" do
    it "returns all projection weights and biases" do
      mha = GS::NN::MultiHeadAttention.new(embed_dim: 64, num_heads: 4, device: GS::Tensor::Device::CPU)
      params = mha.parameters
      # Wq, Wk, Wv, Wo × (weight + bias) = 8 parameters
      params.size.should eq(8)
    end
  end

  describe "GPU attention" do
    it "works on GPU" do
      if GS::Metal::Device.available?
        mha = GS::NN::MultiHeadAttention.new(embed_dim: 64, num_heads: 4, device: GS::Tensor::Device::GPU)
        input = GS::Autograd::Variable.randn(2, 8, 64, requires_grad: false, device: GS::Tensor::Device::GPU)
        output = mha.self_attention(input)
        output.shape.should eq(GS::Shape.new([2, 8, 64]))
      end
    end
  end
end
