`timescale 1ps/100fs 


module rgb2gs(
    input         clk,
    input  [23:0] rgb,
    input         rgb_en,
    output [7:0]  gs,
    output        gs_valid
);
    reg gs_valid_reg = 0;
    reg [8:0] sum;

    assign gs_valid = gs_valid_reg; 
    assign gs = sum;
    
    integer n_transform_debug = 0;
    always @(posedge clk) begin
        if (rgb_en) begin
            sum = (rgb[23:16] + rgb[7:0]) >> 1;
            sum = (sum + rgb[15:8]) >> 1;
            gs_valid_reg = 1;
            $display("%d: RGB ([%d %d %d]) -> GS %d", n_transform_debug * 4, rgb[23:16], rgb[15:8], rgb[7:0], gs);
            n_transform_debug = n_transform_debug + 1;
        end
        else
            gs_valid_reg = 0;
    end
endmodule


module rgb2gs_tb;
    reg [7:0]  r   = 122;
    reg [7:0]  g   = 210;
    reg [7:0]  b   = 64;
    reg [23:0] rgb;
    reg [8:0]  sum;
    wire [7:0] sum_wire;

    assign sum_wire = sum;

    initial begin
        rgb = {r, g, b};
        sum = (rgb[23:16] + rgb[7:0]) >> 1 ;
        sum = (sum + rgb[15:8]) >> 1;
        $display("%d", sum_wire);
    end
endmodule