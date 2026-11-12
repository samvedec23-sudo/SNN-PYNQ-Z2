`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 14.10.2026 22:04:24
// Design Name: 
// Module Name: cifar10_snn_accelerator
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////



module cifar10_snn_accelerator (
    input  wire        clk,
    input  wire        resetn,

    // Input stream from AXI DMA MM2S
    input  wire [31:0] s_axis_tdata,
    input  wire        s_axis_tvalid,
    output wire        s_axis_tready,
    input  wire        s_axis_tlast,

    // Output stream to AXI DMA S2MM
    output wire [31:0] m_axis_tdata,
    output wire        m_axis_tvalid,
    input  wire        m_axis_tready,
    output wire        m_axis_tlast,

    // Status
    output reg         busy,
    output reg         done
);

    // Temporary interface placeholder.
    // SNN computation is NOT implemented yet.
    // This module currently accepts and forwards stream data.

    assign s_axis_tready = m_axis_tready;
    assign m_axis_tdata  = s_axis_tdata;
    assign m_axis_tvalid = s_axis_tvalid;
    assign m_axis_tlast  = s_axis_tlast;

    always @(posedge clk) begin
        if (!resetn) begin
            busy <= 1'b0;
            done <= 1'b0;
        end else begin
            busy <= 1'b0;
            done <= 1'b0;
        end
    end

endmodule

module snn_mac_engine #(
    parameter DATA_W = 16,
    parameter ACC_W  = 40
)(
    input  wire                    clk,
    input  wire                    resetn,
    input  wire                    start,
    input  wire                    valid,
    input  wire                    last,
    input  wire signed [DATA_W-1:0] activation,
    input  wire signed [DATA_W-1:0] weight,

    output reg  signed [ACC_W-1:0] accumulator,
    output reg                     busy,
    output reg                     done
);

    wire signed [(2*DATA_W)-1:0] product;
    assign product = activation * weight;

    always @(posedge clk) begin
        if (!resetn) begin
            accumulator <= 0;
            busy        <= 0;
            done        <= 0;
        end else begin
            done <= 0;

            if (start && !busy) begin
                accumulator <= 0;
                busy        <= 1;
            end else if (busy && valid) begin
                if (last) begin
                    accumulator <= accumulator + product;
                    busy        <= 0;
                    done        <= 1;
                end else begin
                    accumulator <= accumulator + product;
                end
            end
        end
    end

endmodule
