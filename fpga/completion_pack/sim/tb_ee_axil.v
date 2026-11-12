`timescale 1ns / 1ps
// Drives ee_axil exactly like the ARM CPU will (AXI4-Lite writes/reads with random wait states),
// replays 3000 images from ee_vectors.txt, and checks the FPGA's decision against the golden model.
module tb_ee_axil;
    localparam NIMG = 3000, T = 32;
    reg clk = 0, rstn = 0;
    always #5 clk = ~clk;

    reg  [3:0]  awaddr = 0; reg awvalid = 0; wire awready;
    reg  [31:0] wdata = 0;  reg wvalid = 0;  wire wready;
    wire [1:0]  bresp; wire bvalid; reg bready = 0;
    reg  [3:0]  araddr = 0; reg arvalid = 0; wire arready;
    wire [31:0] rdata; wire [1:0] rresp; wire rvalid; reg rready = 0;

    ee_axil dut(.s_axi_aclk(clk), .s_axi_aresetn(rstn),
        .s_axi_awaddr(awaddr), .s_axi_awvalid(awvalid), .s_axi_awready(awready),
        .s_axi_wdata(wdata), .s_axi_wstrb(4'hF), .s_axi_wvalid(wvalid), .s_axi_wready(wready),
        .s_axi_bresp(bresp), .s_axi_bvalid(bvalid), .s_axi_bready(bready),
        .s_axi_araddr(araddr), .s_axi_arvalid(arvalid), .s_axi_arready(arready),
        .s_axi_rdata(rdata), .s_axi_rresp(rresp), .s_axi_rvalid(rvalid), .s_axi_rready(rready));

    integer seed = 7, protocol_errors = 0;

    task axi_write(input [3:0] a, input [31:0] d);
        integer gap, w_late;
        reg aw_done, w_done;
        begin
            gap = $urandom(seed) % 4;  repeat (gap) @(posedge clk);
            w_late = $urandom(seed) % 3;                       // W channel may arrive 0..2 clocks after AW
            awaddr <= a; awvalid <= 1; wdata <= d; wvalid <= 0;
            aw_done = 0; w_done = 0;
            repeat (w_late) begin @(posedge clk); if (awvalid && awready) aw_done = 1; end
            wvalid <= 1;
            while (!(aw_done && w_done)) begin
                @(posedge clk);
                if (awvalid && awready) aw_done = 1;
                if (wvalid && wready)   w_done = 1;
                #1;
                if (aw_done) awvalid <= 0;
                if (w_done)  wvalid <= 0;
            end
            bready <= ($urandom(seed) % 2);
            while (!(bvalid && bready)) begin @(posedge clk); if (bvalid) bready <= 1; end
            @(posedge clk); bready <= 0;
            if (bresp !== 2'b00) protocol_errors = protocol_errors + 1;
        end
    endtask

    // ---- AXI protocol monitors: responses must come AFTER the request handshakes ----
    reg ar_hs_d = 0, rvalid_d = 0, bvalid_d = 0, aw_seen = 0, w_seen = 0;
    always @(posedge clk) begin
        ar_hs_d <= arvalid && arready;  rvalid_d <= rvalid;  bvalid_d <= bvalid;
        if (rvalid && !rvalid_d && !ar_hs_d) begin protocol_errors = protocol_errors + 1; $display("PROTOCOL: RVALID before AR handshake"); end
        if (awvalid && awready) aw_seen = 1;
        if (wvalid && wready)   w_seen = 1;
        if (bvalid && !bvalid_d && !(aw_seen && w_seen)) begin protocol_errors = protocol_errors + 1; $display("PROTOCOL: BVALID before AW+W handshakes"); end
        if (bvalid && bready) begin aw_seen = 0; w_seen = 0; end
    end

    task axi_read(input [3:0] a, output [31:0] d);
        begin
            araddr <= a; arvalid <= 1; rready <= 0;
            @(posedge clk);
            while (!(arvalid && arready)) @(posedge clk);
            // handshake cycle: RVALID must NOT already be high before the handshake
            arvalid <= 0;
            rready <= ($urandom(seed) % 2);
            while (!rvalid) @(posedge clk);
            rready <= 1; @(posedge clk);
            d = rdata; rready <= 0;
        end
    endtask

    integer fd, img, t, rc, errors = 0, early = 0;
    reg [31:0] v, e_steps, e_pred, tm, mg, st;
    reg [9:0] sp [0:T-1];
    reg [31:0] cfg;

    initial begin
        fd = $fopen("ee_vectors.txt", "r");
        if (fd == 0) begin $display("ERROR: run gen_vectors.py first"); $finish; end
        repeat (4) @(posedge clk); rstn = 1; repeat (2) @(posedge clk);
        axi_read(4'h4, st);
        if (st[31:16] !== 16'hEE01) begin errors = errors + 1; $display("BAD MAGIC %h", st); end
        for (img = 0; img < NIMG; img = img + 1) begin
            rc = $fscanf(fd, "%h", tm); rc = $fscanf(fd, "%h", mg);
            for (t = 0; t < T; t = t + 1) begin rc = $fscanf(fd, "%h", v); sp[t] = v[9:0]; end
            rc = $fscanf(fd, "%h", e_steps); rc = $fscanf(fd, "%h", e_pred);
            cfg = (tm[5:0] << 12) | (mg[5:0] << 18) | (32'd32 << 24);
            axi_write(4'h0, cfg | (1 << 11));                          // START
            for (t = 0; t < T; t = t + 1) begin
                if (($urandom(seed) % 10) < 3)                         // config-only write (no step bit): must NOT count
                    axi_write(4'h0, cfg | ($urandom(seed) & 32'h3FF));
                axi_write(4'h0, cfg | (1 << 10) | sp[t]);             // one write per timestep
            end
            axi_read(4'h4, st);
            if (st[0] !== 1'b1 || st[10:5] !== e_steps[5:0] || st[4:1] !== e_pred[3:0] || st[31:16] !== 16'hEE01) begin
                errors = errors + 1;
                if (errors <= 5) $display("MISMATCH img %0d: status=%h exp steps=%0d pred=%0d", img, st, e_steps, e_pred);
            end
            if (e_steps < T) early = early + 1;
        end
        if (errors == 0 && protocol_errors == 0)
            $display("TEST PASSED: ee_axil (AXI4-Lite) matches golden on %0d images, %0d exited early, magic word OK", NIMG, early);
        else $display("TEST FAILED: %0d decision errors, %0d protocol errors", errors, protocol_errors);
        $finish;
    end
    // watchdog
    initial begin #2000000000; $display("TEST FAILED: timeout"); $finish; end
endmodule
