`timescale 1ns / 1ps
// Streams 20000 vectors through lif_bn_unit at 1 neuron/clock and compares with the Python golden model.
module tb_lif_bn_unit;
    localparam N = 20000;
    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;

    reg signed [19:0] acc_a [0:N-1];
    reg signed [20:0] m_a   [0:N-1];
    reg signed [31:0] b_a   [0:N-1];
    reg signed [31:0] mem_a [0:N-1];
    reg signed [31:0] exp_mem [0:N-1];
    reg               exp_spk [0:N-1];

    reg                in_valid = 0;
    reg  signed [19:0] acc; reg signed [20:0] m_q24; reg signed [31:0] b_q16, mem_in;
    wire               out_valid, spike;
    wire signed [31:0] mem_out;

    lif_bn_unit dut(.clk(clk), .rst_n(rst_n), .in_valid(in_valid), .acc(acc), .m_q24(m_q24),
                    .b_q16(b_q16), .mem_in(mem_in), .out_valid(out_valid), .mem_out(mem_out), .spike(spike));

    integer fd, i, rc, errors = 0, checked = 0;
    reg [19:0] t_acc; reg [20:0] t_m; reg [31:0] t_b, t_mem, t_emem; reg t_spk;

    initial begin
        fd = $fopen("lif_vectors.txt", "r");
        if (fd == 0) begin $display("ERROR: cannot open lif_vectors.txt (run gen_vectors.py first)"); $finish; end
        for (i = 0; i < N; i = i + 1) begin
            rc = $fscanf(fd, "%h %h %h %h %h %d\n", t_acc, t_m, t_b, t_mem, t_emem, t_spk);
            acc_a[i] = t_acc; m_a[i] = t_m; b_a[i] = t_b; mem_a[i] = t_mem; exp_mem[i] = t_emem; exp_spk[i] = t_spk;
        end
        $fclose(fd);
        repeat (3) @(posedge clk); rst_n = 1; @(posedge clk);
        for (i = 0; i < N; i = i + 1) begin           // back-to-back: one neuron per clock
            in_valid <= 1; acc <= acc_a[i]; m_q24 <= m_a[i]; b_q16 <= b_a[i]; mem_in <= mem_a[i];
            @(posedge clk);
        end
        in_valid <= 0;
        repeat (5) @(posedge clk);
        if (checked != N) begin errors = errors + 1; $display("ERROR: checked %0d of %0d outputs", checked, N); end
        if (errors == 0) $display("TEST PASSED: lif_bn_unit matches golden model on all %0d vectors (incl. saturation + threshold edges)", N);
        else             $display("TEST FAILED: %0d errors", errors);
        $finish;
    end

    integer k = 0;
    always @(posedge clk) if (out_valid) begin
        if (mem_out !== exp_mem[k] || spike !== exp_spk[k]) begin
            errors = errors + 1;
            if (errors <= 5) $display("MISMATCH #%0d: acc=%0d m=%0d b=%0d mem=%0d -> got mem=%0d spk=%b, exp mem=%0d spk=%b",
                                      k, acc_a[k], m_a[k], b_a[k], mem_a[k], mem_out, spike, exp_mem[k], exp_spk[k]);
        end
        k = k + 1; checked = checked + 1;
    end
endmodule
