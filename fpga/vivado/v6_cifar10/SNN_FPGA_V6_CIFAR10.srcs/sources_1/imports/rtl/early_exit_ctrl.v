// -----------------------------------------------------------------------------
// early_exit_ctrl.v  -  Confidence-Gated Early Exit (CGEE)            [v2: timing-optimised]
//
// Counts output-layer spikes per class after every timestep. As soon as
//        steps >= t_min   AND   (top1_count - top2_count) >= margin
// the inference is finished: `done` goes high, `pred` / `steps_used` are frozen and the
// layer controller can skip the remaining timesteps (saving latency and energy).
// If the rule never fires, the run ends at t_max (normally 32) - same as the baseline network.
//
// Decisions are IDENTICAL to golden/early_exit.py (ties -> lowest class index; margin 0 is treated as 1).
//
// v2 change (why): v1 found the leader with a serial 10-step compare chain, then the runner-up with a second
// serial chain, then subtracted - ~35 ns of logic, which FAILED timing at 50 MHz (WNS -15.7 ns). Now:
//   * "is class i ahead of every other class by >= margin?" is evaluated for all 10 classes IN PARALLEL
//     (cnt[i] >= cnt[j] + margin for all j != i)  -> a shallow AND-of-compares, no serial chain, no subtractor
//   * lowest-index argmax is also a parallel compare network
//   * the decision is taken one clock AFTER the counters update (2-stage pipeline)
// Interface timing: `done`/`pred`/`steps_used` update 2 clocks after the `step_valid` pulse (v1: 1 clock).
// Back-to-back step_valid pulses are still handled correctly (pipelined).
//
// Pulse `start` once per image (clears counters). Pulse `step_valid` once per timestep with the
// 10 output spikes of that timestep on `out_spikes`.
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps
module early_exit_ctrl #(
    parameter NCLS  = 10,
    parameter CNT_W = 6                       // supports up to 63 timesteps
)(
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 start,        // clear for a new image
    input  wire                 step_valid,   // one pulse per timestep
    input  wire [NCLS-1:0]      out_spikes,
    input  wire [CNT_W-1:0]     t_min,
    input  wire [CNT_W-1:0]     margin,
    input  wire [CNT_W-1:0]     t_max,
    output reg                  done,
    output reg  [3:0]           pred,
    output reg  [CNT_W-1:0]     steps_used,
    output wire [NCLS*CNT_W-1:0] counts_flat  // for the dashboard / debug registers
);
    reg [CNT_W-1:0] cnt [0:NCLS-1];
    reg [CNT_W-1:0] t;
    reg             eval_v;                   // stage 2 must evaluate cnt/t (they were updated at the last edge)
    integer i, j;

    assign counts_flat = {cnt[9], cnt[8], cnt[7], cnt[6], cnt[5],
                          cnt[4], cnt[3], cnt[2], cnt[1], cnt[0]};

    // ---------------- stage 1: count (just an adder per class) ----------------
    always @(posedge clk) begin
        if (!rst_n || start) begin
            for (i = 0; i < NCLS; i = i + 1) cnt[i] <= {CNT_W{1'b0}};
            t      <= {CNT_W{1'b0}};
            eval_v <= 1'b0;
        end else begin
            eval_v <= step_valid && !done;
            if (step_valid && !done) begin
                for (i = 0; i < NCLS; i = i + 1)
                    cnt[i] <= cnt[i] + {{(CNT_W-1){1'b0}}, out_spikes[i]};
                t <= t + {{(CNT_W-1){1'b0}}, 1'b1};
            end
        end
    end

    // ---------------- stage 2: decide (parallel compare networks, no serial chains) ----------------
    wire [CNT_W:0] m_eff = (margin == {CNT_W{1'b0}}) ? {{CNT_W{1'b0}}, 1'b1} : {1'b0, margin};

    reg [CNT_W:0]  cm [0:NCLS-1];             // cnt[j] + margin
    reg [NCLS-1:0] lead;                      // class i is ahead of every other class by >= margin
    reg [NCLS-1:0] amax;                      // class i is the lowest-index maximum
    reg [3:0]      arg_c;
    reg            fire;

    always @* begin
        for (j = 0; j < NCLS; j = j + 1)
            cm[j] = {1'b0, cnt[j]} + m_eff;
        for (i = 0; i < NCLS; i = i + 1) begin
            lead[i] = 1'b1;
            amax[i] = 1'b1;
            for (j = 0; j < NCLS; j = j + 1) begin
                if (i != j) begin
                    if ({1'b0, cnt[i]} < cm[j]) lead[i] = 1'b0;
                    if (j < i) begin if (!(cnt[i] >  cnt[j])) amax[i] = 1'b0; end
                    else       begin if (!(cnt[i] >= cnt[j])) amax[i] = 1'b0; end
                end
            end
        end
        arg_c = 4'd0;                          // amax is one-hot -> binary
        for (i = 0; i < NCLS; i = i + 1)
            if (amax[i]) arg_c = arg_c | i[3:0];
        fire = ((t >= t_min) && (|lead)) || (t >= t_max);
    end

    always @(posedge clk) begin
        if (!rst_n || start) begin
            done <= 1'b0; pred <= 4'd0; steps_used <= {CNT_W{1'b0}};
        end else if (eval_v && !done && fire) begin
            done <= 1'b1; pred <= arg_c; steps_used <= t;
        end
    end
endmodule
