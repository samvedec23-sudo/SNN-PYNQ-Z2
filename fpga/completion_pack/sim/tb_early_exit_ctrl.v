`timescale 1ns / 1ps
// 3000 random images x 32 timesteps; compares exit step and prediction with golden/early_exit.py
module tb_early_exit_ctrl;
    localparam NIMG = 3000, T = 32;
    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;

    reg start = 0, step_valid = 0;
    reg [9:0] out_spikes = 0;
    reg [5:0] t_min = 1, margin = 1;
    wire done; wire [3:0] pred; wire [5:0] steps_used; wire [59:0] counts;

    early_exit_ctrl dut(.clk(clk), .rst_n(rst_n), .start(start), .step_valid(step_valid), .out_spikes(out_spikes),
                        .t_min(t_min), .margin(margin), .t_max(6'd32), .done(done), .pred(pred),
                        .steps_used(steps_used), .counts_flat(counts));

    integer fd, img, t, rc, errors = 0, early = 0, sum_steps = 0, seed = 11;
    reg [31:0] v, e_steps, e_pred, tm, mg;
    reg [9:0] sp [0:T-1];

    initial begin
        fd = $fopen("ee_vectors.txt", "r");
        if (fd == 0) begin $display("ERROR: cannot open ee_vectors.txt (run gen_vectors.py first)"); $finish; end
        repeat (3) @(posedge clk); rst_n = 1; @(posedge clk);
        for (img = 0; img < NIMG; img = img + 1) begin
            rc = $fscanf(fd, "%h", tm); rc = $fscanf(fd, "%h", mg);
            for (t = 0; t < T; t = t + 1) begin rc = $fscanf(fd, "%h", v); sp[t] = v[9:0]; end
            rc = $fscanf(fd, "%h", e_steps); rc = $fscanf(fd, "%h", e_pred);
            t_min <= tm[5:0]; margin <= mg[5:0];
            start <= 1; @(posedge clk); start <= 0; @(posedge clk);
            for (t = 0; t < T; t = t + 1) begin            // keep feeding even after done: result must stay frozen
                out_spikes <= sp[t]; step_valid <= 1; @(posedge clk);
                step_valid <= 0;
                repeat ($urandom(seed) % 3) @(posedge clk);       // 0..2 idle clocks: 0 = back-to-back steps
            end
            repeat (4) @(posedge clk);                             // let the 2-stage pipeline finish
            if (!done || steps_used !== e_steps[5:0] || pred !== e_pred[3:0]) begin
                errors = errors + 1;
                if (errors <= 5) $display("MISMATCH img %0d (t_min=%0d margin=%0d): got steps=%0d pred=%0d done=%b, exp steps=%0d pred=%0d",
                                          img, tm, mg, steps_used, pred, done, e_steps, e_pred);
            end
            if (e_steps < T) early = early + 1;
            sum_steps = sum_steps + e_steps;
        end
        if (errors == 0) $display("TEST PASSED: early_exit_ctrl matches golden on %0d images (%0d exited early, mean steps %0d.%0d of 32)",
                                  NIMG, early, sum_steps / NIMG, (sum_steps * 10 / NIMG) % 10);
        else             $display("TEST FAILED: %0d errors", errors);
        $finish;
    end
endmodule
