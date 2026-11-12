// -----------------------------------------------------------------------------
// ee_axil.v - AXI4-Lite register wrapper around early_exit_ctrl.
// Lets the ARM CPU (PYNQ/Python) feed the output-layer spikes of each timestep to the FPGA and
// read back the FPGA's early-exit decision.
//
// Register map (32-bit, byte offsets from the block's base address):
//   0x00  CTRL   (write)  [9:0]   out_spikes of this timestep (bit c = class c fired)
//                         [10]    step_valid  - write-1 = "count this timestep" (one-shot, self-clearing)
//                         [11]    start       - write-1 = clear for a new image (one-shot)
//                         [17:12] t_min       [23:18] margin       [29:24] t_max
//   0x04  STATUS (read)   [0] done   [4:1] pred   [10:5] steps_used   [31:16] 0xEE01 (magic = "this is my block")
//   --- v3 additions (LED feedback + time-to-first-spike) ---
//   0x08  FIRST  (read)   [5:0] first_step (1-based timestep of the first output spike, 0 = none yet)
//                         [6] first_seen   [15:8] 0xA5 (capability marker: "this build has LEDs")
//   0x0C  LED    (write/read) [3:0] value shown on the 4 user LEDs LD0..LD3 (software chooses, e.g. class code)
//   Hardware-driven outputs:
//     rgb0_g = first_seen      (RGB LED LD4 green  : "an output spike has arrived")
//     rgb1_g = done, exited early      (RGB LED LD5 green)
//     rgb1_r = done, ran the full t_max (RGB LED LD5 red : no early exit)
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps
module ee_axil (
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 s_axi_aclk CLK" *)
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF s_axi, ASSOCIATED_RESET s_axi_aresetn" *)
    input  wire        s_axi_aclk,
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 s_axi_aresetn RST" *)
    (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input  wire        s_axi_aresetn,
    input  wire [3:0]  s_axi_awaddr,
    input  wire        s_axi_awvalid,
    output reg         s_axi_awready,
    input  wire [31:0] s_axi_wdata,
    input  wire [3:0]  s_axi_wstrb,
    input  wire        s_axi_wvalid,
    output reg         s_axi_wready,
    output wire [1:0]  s_axi_bresp,
    output reg         s_axi_bvalid,
    input  wire        s_axi_bready,
    input  wire [3:0]  s_axi_araddr,
    input  wire        s_axi_arvalid,
    output reg         s_axi_arready,
    output reg  [31:0] s_axi_rdata,
    output wire [1:0]  s_axi_rresp,
    output reg         s_axi_rvalid,
    input  wire        s_axi_rready,
    // v3: board LEDs
    output wire [3:0]  led,
    output wire        rgb0_g,
    output wire        rgb1_g,
    output wire        rgb1_r
);
    assign s_axi_bresp = 2'b00;   // OKAY
    assign s_axi_rresp = 2'b00;

    // ---------------- write channel (one outstanding transaction) ----------------
    reg [3:0]  awaddr_q;
    reg        aw_got, w_got;
    reg [31:0] wdata_q;
    reg        wr_pend;                    // delays bvalid a few clocks so a following read sees fresh status
    wire       do_write = aw_got && w_got && !s_axi_bvalid && !wr_pend;
    reg [2:0]  wr_dly;

    reg [9:0]  spikes_r;
    reg [5:0]  tmin_r, margin_r, tmax_r;
    reg        step_pulse, start_pulse;
    reg [3:0]  led_sw;

    always @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            s_axi_awready <= 1'b0; s_axi_wready <= 1'b0; s_axi_bvalid <= 1'b0;
            aw_got <= 1'b0; w_got <= 1'b0; wr_pend <= 1'b0; wr_dly <= 3'd0;
            led_sw <= 4'd0; spikes_r <= 10'd0; tmin_r <= 6'd1; margin_r <= 6'd1; tmax_r <= 6'd32;
            step_pulse <= 1'b0; start_pulse <= 1'b0; awaddr_q <= 4'd0; wdata_q <= 32'd0;
        end else begin
            s_axi_awready <= 1'b0; s_axi_wready <= 1'b0;
            step_pulse <= 1'b0; start_pulse <= 1'b0;
            if (s_axi_awvalid && !aw_got && !wr_pend && !s_axi_bvalid) begin
                s_axi_awready <= 1'b1; aw_got <= 1'b1; awaddr_q <= s_axi_awaddr;
            end
            if (s_axi_wvalid && !w_got && !wr_pend && !s_axi_bvalid) begin
                s_axi_wready <= 1'b1; w_got <= 1'b1; wdata_q <= s_axi_wdata;
            end
            if (do_write) begin
                if (awaddr_q[3:2] == 2'd3) led_sw <= wdata_q[3:0];
                if (awaddr_q[3:2] == 2'd0) begin
                    spikes_r <= wdata_q[9:0];
                    tmin_r <= wdata_q[17:12]; margin_r <= wdata_q[23:18]; tmax_r <= wdata_q[29:24];
                    step_pulse <= wdata_q[10];       // one-shot, seen by the controller next clock
                    start_pulse <= wdata_q[11];
                end
                aw_got <= 1'b0; w_got <= 1'b0; wr_pend <= 1'b1; wr_dly <= 3'd0;
            end
            if (wr_pend) begin
                wr_dly <= wr_dly + 3'd1;
                if (wr_dly == 3'd3) begin wr_pend <= 1'b0; s_axi_bvalid <= 1'b1; end
            end
            if (s_axi_bvalid && s_axi_bready) s_axi_bvalid <= 1'b0;
        end
    end

    // ---------------- the real work: the early-exit controller ----------------
    wire        done;
    wire [3:0]  pred;
    wire [5:0]  steps_used;
    wire [59:0] counts_unused;

    early_exit_ctrl #(.NCLS(10), .CNT_W(6)) u_ee (
        .clk(s_axi_aclk), .rst_n(s_axi_aresetn), .start(start_pulse), .step_valid(step_pulse),
        .out_spikes(spikes_r), .t_min(tmin_r), .margin(margin_r), .t_max(tmax_r),
        .done(done), .pred(pred), .steps_used(steps_used), .counts_flat(counts_unused));

    // ---------------- v3: time-to-first-spike detector (hardware) ----------------
    reg [5:0] stepcnt, first_step;
    reg       first_seen;
    always @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn || start_pulse) begin
            stepcnt <= 6'd0; first_step <= 6'd0; first_seen <= 1'b0;
        end else if (step_pulse && !done) begin
            stepcnt <= stepcnt + 6'd1;
            if ((|spikes_r) && !first_seen) begin
                first_seen <= 1'b1;
                first_step <= stepcnt + 6'd1;
            end
        end
    end

    assign led    = led_sw;
    assign rgb0_g = first_seen;
    assign rgb1_g = done && (steps_used != tmax_r);
    assign rgb1_r = done && (steps_used == tmax_r);

    // ---------------- read channel ----------------
    wire [31:0] status = {16'hEE01, 5'd0, steps_used, pred, done};
    wire [31:0] ctrl_rb = {2'b0, tmax_r, margin_r, tmin_r, 2'b0, spikes_r};
    wire [31:0] first_rb = {16'd0, 8'hA5, 1'b0, first_seen, first_step};

    always @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            s_axi_arready <= 1'b0; s_axi_rvalid <= 1'b0; s_axi_rdata <= 32'd0;
        end else begin
            s_axi_arready <= s_axi_arvalid && !s_axi_arready && !s_axi_rvalid;       // 1-clock pulse
            if (s_axi_arready && s_axi_arvalid && !s_axi_rvalid) begin               // handshake happened -> respond
                s_axi_rvalid <= 1'b1;
                s_axi_rdata  <= (s_axi_araddr[3:2] == 2'd1) ? status :
                                (s_axi_araddr[3:2] == 2'd0) ? ctrl_rb :
                                (s_axi_araddr[3:2] == 2'd2) ? first_rb : {28'd0, led_sw};
            end
            if (s_axi_rvalid && s_axi_rready) s_axi_rvalid <= 1'b0;
        end
    end
endmodule
