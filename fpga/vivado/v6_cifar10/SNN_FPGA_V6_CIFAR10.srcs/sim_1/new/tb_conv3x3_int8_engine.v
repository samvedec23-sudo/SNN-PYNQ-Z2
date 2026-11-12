`timescale 1ns / 1ps

module tb_conv3x3_int8_engine;

    reg clk;
    reg resetn;

    reg signed [7:0] in_data;
    reg in_valid;
    wire in_ready;

    reg start;
    wire busy;
    wire done;

    wire signed [31:0] out_data;
    wire out_valid;
    reg out_ready;
    wire out_last;

    integer i;
    integer output_count;

    reg signed [31:0] expected [0:3];

    // Small test configuration
    conv3x3_int8_engine #(
        .IN_CH(1),
        .OUT_CH(1),
        .IMG_W(2),
        .IMG_H(2),
        .IN_MEM_DEPTH(4),
        .W_MEM_DEPTH(9)
    ) dut (
        .clk(clk),
        .resetn(resetn),

        .in_data(in_data),
        .in_valid(in_valid),
        .in_ready(in_ready),

        .start(start),
        .busy(busy),
        .done(done),

        .out_data(out_data),
        .out_valid(out_valid),
        .out_ready(out_ready),
        .out_last(out_last)
    );

    // Clock: 10 ns period
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    // Expected outputs using the first 9 weights
    // from conv0_weights_int8.mem
    initial begin
        expected[0] = -94;
        expected[1] = -124;
        expected[2] = 76;
        expected[3] = -30;
    end

    // Monitor output transactions
    always @(posedge clk) begin
        if (resetn && out_valid && out_ready) begin

            $display("Output %0d = %0d",
                     output_count, $signed(out_data));

            if ($signed(out_data) !== expected[output_count]) begin
                $display("ERROR: Expected %0d, received %0d",
                         expected[output_count],
                         $signed(out_data));
                $finish;
            end

            output_count = output_count + 1;
        end
    end

    // Main test
    initial begin

        output_count = 0;

        resetn = 0;
        in_data = 0;
        in_valid = 0;
        start = 0;
        out_ready = 1;

        // Reset
        repeat (5) @(negedge clk);
        resetn = 1;

        // Load four input pixels: all -2
        for (i = 0; i < 4; i = i + 1) begin

            @(negedge clk);

            while (!in_ready)
                @(negedge clk);

            in_data = -2;
            in_valid = 1;

            @(negedge clk);
            in_valid = 0;

        end

        // Start convolution
        @(negedge clk);
        start = 1;

        @(negedge clk);
        start = 0;

        // Wait for completion
        wait(done == 1);

        #10;

        if (output_count == 4)
            $display("TEST PASSED: All 4 outputs match!");
        else
            $display("TEST FAILED: Only %0d outputs received",
                     output_count);

        $finish;

    end

endmodule