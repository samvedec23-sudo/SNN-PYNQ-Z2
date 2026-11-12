// -----------------------------------------------------------------------------
// lif_bn_unit.v  -  BatchNorm re-quantisation + LIF membrane update (one neuron per call)
//
// Bit-exact with golden/snn_golden.py (mode "fixed"):
//   y_q16   = ((acc * m_q24 + 128) >>> 8) + b_q16                  (BN folded with INT8 weight scale)
//   reset   = (mem_in > THR_Q)                                      (delayed subtractive reset)
//   mem_new = sat32( ((mem_in*BETA_Q + 32768) >>> 16) + y_q16 - (reset ? THR_Q : 0) )
//   spike   = (mem_new > THR_Q)
// Q16 fixed point: 1.0 = 65536.  BETA_Q = round(0.85*65536) = 55706.
//
// Widths come from the measured ranges of YOUR exported weights:
//   |acc|<=134K (20b signed), |Mq|<=305K (21b signed), product needs 48b.
// Two-stage pipeline: stage 1 = multiply/round/add bias, stage 2 = LIF update.  Latency = 2 clocks.
// Streams one neuron per clock (throughput 1/clk).  Instantiate N copies for N neurons in parallel.
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps
module lif_bn_unit #(
    parameter ACC_W = 20,
    parameter M_W   = 21
)(
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     in_valid,
    input  wire signed [ACC_W-1:0]  acc,       // conv accumulator (sum of spike * int8 weight)
    input  wire signed [M_W-1:0]    m_q24,     // per-channel multiplier  round(A*Sw*2^24)
    input  wire signed [31:0]       b_q16,     // per-channel offset      round(B*2^16)
    input  wire signed [31:0]       mem_in,    // membrane potential read from state memory
    output reg                      out_valid,
    output reg  signed [31:0]       mem_out,   // new membrane (write back to state memory)
    output reg                      spike
);
    localparam signed [63:0] THR_Q  = 64'sd65536;
    localparam signed [63:0] BETA_Q = 64'sd55706;
    localparam signed [63:0] MAXV   = 64'sd2147483647;
    localparam signed [63:0] MINV   = -64'sd2147483648;

    // ---------------- stage 1: BN requantisation ----------------
    wire signed [47:0] prod = $signed(acc) * $signed(m_q24);
    wire signed [47:0] rnd  = (prod + 48'sd128) >>> 8;
    wire signed [47:0] y_c  = rnd + $signed(b_q16);

    reg signed [47:0] y_r;
    reg signed [31:0] mem_r;
    reg               v1;

    always @(posedge clk) begin
        if (!rst_n) begin
            v1 <= 1'b0; y_r <= 48'sd0; mem_r <= 32'sd0;
        end else begin
            v1 <= in_valid;
            if (in_valid) begin y_r <= y_c; mem_r <= mem_in; end
        end
    end

    // ---------------- stage 2: LIF update ----------------
    wire               reset_f = ($signed({{32{mem_r[31]}}, mem_r}) > THR_Q);
    wire signed [63:0] mem64   = {{32{mem_r[31]}}, mem_r};
    wire signed [63:0] decay   = ((mem64 * BETA_Q) + 64'sd32768) >>> 16;
    wire signed [63:0] y64     = {{16{y_r[47]}}, y_r};
    wire signed [63:0] nxt     = decay + y64 - (reset_f ? THR_Q : 64'sd0);
    wire signed [63:0] sat     = (nxt > MAXV) ? MAXV : ((nxt < MINV) ? MINV : nxt);

    always @(posedge clk) begin
        if (!rst_n) begin
            out_valid <= 1'b0; mem_out <= 32'sd0; spike <= 1'b0;
        end else begin
            out_valid <= v1;
            if (v1) begin
                mem_out <= sat[31:0];
                spike   <= (sat > THR_Q);
            end
        end
    end
endmodule
