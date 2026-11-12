`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 18.10.2026 13:40:51
// Design Name: 
// Module Name: conv0_lif_pool
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


module conv0_mac (
    input  wire               clk,
    input  wire               resetn,
    input  wire               clear_acc,
    input  wire               mac_valid,
    input  wire signed [7:0]  activation,
    input  wire signed [7:0]  weight,
    output reg  signed [31:0] accumulator
);

    wire signed [15:0] product;
    assign product = activation * weight;

    always @(posedge clk) begin
        if (!resetn)
            accumulator <= 32'sd0;
        else if (clear_acc)
            accumulator <= 32'sd0;
        else if (mac_valid)
            accumulator <= accumulator + {{16{product[15]}}, product};
    end

endmodule

