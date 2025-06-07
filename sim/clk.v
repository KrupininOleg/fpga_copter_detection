`timescale 1ps/100fs

module clk_generator #(
    parameter CLKIN_PERIOD = 3000,
    parameter REFCLK_FREQ  = 200.0
)
(
    output sys_clk,
    output ref_clk
);
    localparam real REFCLK_PERIOD = (1000000.0/(2*REFCLK_FREQ));

    reg sys_clk_reg = 1'b0;
    reg ref_clk_reg = 1'b0;
    assign sys_clk = sys_clk_reg;
    assign ref_clk = ref_clk_reg;
    
    initial
    begin
        forever
            #(CLKIN_PERIOD/2.0) sys_clk_reg = ~sys_clk_reg;
    end

    initial
    begin
        forever
            #REFCLK_PERIOD ref_clk_reg = ~ref_clk_reg;
    end
endmodule