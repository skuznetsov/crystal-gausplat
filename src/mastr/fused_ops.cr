# MASt3R Fused Operations
# Crystal wrappers for optimized Metal kernels

require "../metal/device"
require "../metal/dispatch"
require "../core/tensor"
require "../autograd/variable"

module GS
  module MASt3R
    # Fused operations dispatcher for MASt3R
    class FusedOps
      KERNEL_SOURCE_PATH = "src/metal/kernels/mastr_fused.metal"

      @device : Metal::Device
      @pipelines : Hash(String, Metal::ComputePipeline)

      def initialize
        @device = Metal::Device.instance
        @pipelines = Hash(String, Metal::ComputePipeline).new

        # Compile kernels on initialization
        compile_kernels!
      end

      # Compile all fused kernels
      private def compile_kernels!
        kernel_names = [
          "fused_layernorm_qkv",
          "fused_attention_scores",
          "softmax_rows",
          "fused_mlp",
          "fused_cross_attention",
          "apply_rope_2d",
          "fused_residual_layernorm",
        ]

        source = File.read(KERNEL_SOURCE_PATH)

        kernel_names.each do |name|
          @pipelines[name] = Metal::ComputePipeline.new(source, name, "")
        end

        puts "FusedOps: Compiled #{@pipelines.size} kernels"
      end

      # Fused LayerNorm + QKV projection
      # Input: x [batch, seq, embed_dim]
      # Output: qkv [batch, seq, 3 * embed_dim]
      def layernorm_qkv(
        x : Tensor,
        ln_weight : Tensor,
        ln_bias : Tensor,
        qkv_weight : Tensor,
        qkv_bias : Tensor,
        eps : Float32 = 1e-5_f32
      ) : Tensor
        batch = x.shape[0]
        seq_len = x.shape[1]
        embed_dim = x.shape[2]

        output = Tensor.new(batch, seq_len, 3 * embed_dim, device: Tensor::Device::GPU)

        pipeline = @pipelines["fused_layernorm_qkv"]
        cmd = Metal::CommandBuffer.new

        encoder = Metal::ComputeEncoder.new(cmd)
        encoder.set_pipeline(pipeline)
        encoder.set_buffer(x.metal_buffer.not_nil!, 0)
        encoder.set_buffer(ln_weight.metal_buffer.not_nil!, 1)
        encoder.set_buffer(ln_bias.metal_buffer.not_nil!, 2)
        encoder.set_buffer(qkv_weight.metal_buffer.not_nil!, 3)
        encoder.set_buffer(qkv_bias.metal_buffer.not_nil!, 4)
        encoder.set_buffer(output.metal_buffer.not_nil!, 5)
        encoder.set_bytes(pointerof(batch).as(Pointer(Void)), sizeof(Int32), 6)
        encoder.set_bytes(pointerof(seq_len).as(Pointer(Void)), sizeof(Int32), 7)
        encoder.set_bytes(pointerof(embed_dim).as(Pointer(Void)), sizeof(Int32), 8)
        encoder.set_bytes(pointerof(eps).as(Pointer(Void)), sizeof(Float32), 9)

        threads = {3 * embed_dim, seq_len, batch}
        threadgroups = {
          (3 * embed_dim + 15) // 16,
          (seq_len + 15) // 16,
          batch,
        }
        encoder.dispatch(threads, threadgroups)
        encoder.end_encoding

        cmd.commit_and_wait

        output
      end

      # Fused attention scores with optional RoPE
      # Returns: scores [batch, heads, seq, seq]
      def attention_scores(
        q : Tensor,
        k : Tensor,
        num_heads : Int32,
        rope_cos : Tensor? = nil,
        rope_sin : Tensor? = nil
      ) : Tensor
        batch = q.shape[0]
        seq_len = q.shape[1]
        embed_dim = q.shape[2]
        head_dim = embed_dim // num_heads

        # Reshape Q, K to [batch, heads, seq, head_dim]
        # For now, assume they're already in right format or do CPU reshape
        scores = Tensor.new(batch, num_heads, seq_len, seq_len, device: Tensor::Device::GPU)

        use_rope = (rope_cos && rope_sin) ? 1 : 0

        pipeline = @pipelines["fused_attention_scores"]
        cmd = Metal::CommandBuffer.new

        encoder = Metal::ComputeEncoder.new(cmd)
        encoder.set_pipeline(pipeline)
        encoder.set_buffer(q.metal_buffer.not_nil!, 0)
        encoder.set_buffer(k.metal_buffer.not_nil!, 1)
        encoder.set_buffer(scores.metal_buffer.not_nil!, 2)

        if rc = rope_cos
          encoder.set_buffer(rc.metal_buffer.not_nil!, 3)
        end
        if rs = rope_sin
          encoder.set_buffer(rs.metal_buffer.not_nil!, 4)
        end

        encoder.set_bytes(pointerof(batch).as(Pointer(Void)), sizeof(Int32), 5)
        encoder.set_bytes(pointerof(num_heads).as(Pointer(Void)), sizeof(Int32), 6)
        encoder.set_bytes(pointerof(seq_len).as(Pointer(Void)), sizeof(Int32), 7)
        encoder.set_bytes(pointerof(head_dim).as(Pointer(Void)), sizeof(Int32), 8)
        encoder.set_bytes(pointerof(use_rope).as(Pointer(Void)), sizeof(Int32), 9)

        threads = {seq_len, seq_len, batch * num_heads}
        threadgroups = {
          (seq_len + 15) // 16,
          (seq_len + 15) // 16,
          batch * num_heads,
        }
        encoder.dispatch(threads, threadgroups)
        encoder.end_encoding

        # Softmax
        softmax_inplace!(scores)

        cmd.commit_and_wait

        scores
      end

      # Softmax (in-place, row-wise)
      def softmax_inplace!(scores : Tensor)
        total_elements = scores.numel
        rows = total_elements // scores.shape[-1]
        cols = scores.shape[-1]

        pipeline = @pipelines["softmax_rows"]
        cmd = Metal::CommandBuffer.new

        encoder = Metal::ComputeEncoder.new(cmd)
        encoder.set_pipeline(pipeline)
        encoder.set_buffer(scores.metal_buffer.not_nil!, 0)
        encoder.set_bytes(pointerof(rows).as(Pointer(Void)), sizeof(Int32), 1)
        encoder.set_bytes(pointerof(cols).as(Pointer(Void)), sizeof(Int32), 2)

        threads = {rows, 1, 1}
        threadgroups = {(rows + 255) // 256, 1, 1}
        encoder.dispatch(threads, threadgroups)
        encoder.end_encoding

        cmd.commit_and_wait
      end

      # Fused MLP: GELU(x @ W1 + b1) @ W2 + b2
      def mlp(
        x : Tensor,
        w1 : Tensor,
        b1 : Tensor,
        w2 : Tensor,
        b2 : Tensor
      ) : Tensor
        batch_seq = x.shape[0] * x.shape[1]
        in_dim = x.shape[-1]
        hidden_dim = w1.shape[0]
        out_dim = w2.shape[0]

        output = Tensor.new(x.shape[0], x.shape[1], out_dim, device: Tensor::Device::GPU)

        pipeline = @pipelines["fused_mlp"]
        cmd = Metal::CommandBuffer.new

        encoder = Metal::ComputeEncoder.new(cmd)
        encoder.set_pipeline(pipeline)
        encoder.set_buffer(x.metal_buffer.not_nil!, 0)
        encoder.set_buffer(w1.metal_buffer.not_nil!, 1)
        encoder.set_buffer(b1.metal_buffer.not_nil!, 2)
        encoder.set_buffer(w2.metal_buffer.not_nil!, 3)
        encoder.set_buffer(b2.metal_buffer.not_nil!, 4)
        encoder.set_buffer(output.metal_buffer.not_nil!, 5)
        encoder.set_bytes(pointerof(batch_seq).as(Pointer(Void)), sizeof(Int32), 6)
        encoder.set_bytes(pointerof(in_dim).as(Pointer(Void)), sizeof(Int32), 7)
        encoder.set_bytes(pointerof(hidden_dim).as(Pointer(Void)), sizeof(Int32), 8)
        encoder.set_bytes(pointerof(out_dim).as(Pointer(Void)), sizeof(Int32), 9)

        threads = {out_dim, batch_seq, 1}
        threadgroups = {
          (out_dim + 15) // 16,
          (batch_seq + 15) // 16,
          1,
        }
        encoder.dispatch(threads, threadgroups)
        encoder.end_encoding

        cmd.commit_and_wait

        output
      end

      # Apply RoPE 2D (for image patches)
      def apply_rope_2d!(
        x : Tensor,
        freqs : Tensor,
        height : Int32,
        width : Int32,
        num_heads : Int32
      )
        batch = x.shape[0]
        seq_len = x.shape[1]
        embed_dim = x.shape[2]

        pipeline = @pipelines["apply_rope_2d"]
        cmd = Metal::CommandBuffer.new

        encoder = Metal::ComputeEncoder.new(cmd)
        encoder.set_pipeline(pipeline)
        encoder.set_buffer(x.metal_buffer.not_nil!, 0)
        encoder.set_buffer(freqs.metal_buffer.not_nil!, 1)
        encoder.set_bytes(pointerof(batch).as(Pointer(Void)), sizeof(Int32), 2)
        encoder.set_bytes(pointerof(height).as(Pointer(Void)), sizeof(Int32), 3)
        encoder.set_bytes(pointerof(width).as(Pointer(Void)), sizeof(Int32), 4)
        encoder.set_bytes(pointerof(embed_dim).as(Pointer(Void)), sizeof(Int32), 5)
        encoder.set_bytes(pointerof(num_heads).as(Pointer(Void)), sizeof(Int32), 6)

        threads = {embed_dim, seq_len, batch}
        threadgroups = {
          (embed_dim + 15) // 16,
          (seq_len + 15) // 16,
          batch,
        }
        encoder.dispatch(threads, threadgroups)
        encoder.end_encoding

        cmd.commit_and_wait
      end

      # Fused residual + LayerNorm
      def residual_layernorm(
        x : Tensor,
        residual : Tensor,
        weight : Tensor,
        bias : Tensor,
        eps : Float32 = 1e-5_f32
      ) : Tensor
        batch_seq = x.shape[0] * x.shape[1]
        dim = x.shape[-1]

        output = Tensor.new(x.shape, x.dtype, Tensor::Device::GPU)

        pipeline = @pipelines["fused_residual_layernorm"]
        cmd = Metal::CommandBuffer.new

        encoder = Metal::ComputeEncoder.new(cmd)
        encoder.set_pipeline(pipeline)
        encoder.set_buffer(x.metal_buffer.not_nil!, 0)
        encoder.set_buffer(residual.metal_buffer.not_nil!, 1)
        encoder.set_buffer(weight.metal_buffer.not_nil!, 2)
        encoder.set_buffer(bias.metal_buffer.not_nil!, 3)
        encoder.set_buffer(output.metal_buffer.not_nil!, 4)
        encoder.set_bytes(pointerof(batch_seq).as(Pointer(Void)), sizeof(Int32), 5)
        encoder.set_bytes(pointerof(dim).as(Pointer(Void)), sizeof(Int32), 6)
        encoder.set_bytes(pointerof(eps).as(Pointer(Void)), sizeof(Float32), 7)

        threads = {dim, batch_seq, 1}
        threadgroups = {
          (dim + 15) // 16,
          (batch_seq + 15) // 16,
          1,
        }
        encoder.dispatch(threads, threadgroups)
        encoder.end_encoding

        cmd.commit_and_wait

        output
      end

      # Singleton access
      @@instance : FusedOps?

      def self.instance : FusedOps
        @@instance ||= FusedOps.new
      end
    end
  end
end
