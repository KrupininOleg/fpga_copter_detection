`timescale 1ps/100fs 


module frame_preprocessor(
    input         clk,
    input  [23:0] point_input,
    input         input_en,
    output [7:0]  point_output,
    output        output_valid
);
    localparam READY       = 3'd0;
    localparam RGB2GS      = 3'd1;
    localparam RGB2GS_DONE = 3'd2;
    localparam ALL_DONE    = 3'd3;
    
    reg [2:0] state = READY; 
    reg output_valid_reg = 0;
    reg point_rgb_en = 0;

    wire [23:0] point_rgb;
    wire [7:0]  point_gs;
    wire        point_gs_valid; 

    assign point_rgb    = point_input;
    assign point_output = point_gs;
    assign output_valid = output_valid_reg;

    rgb2gs u_rgb2gs(
        .clk      (clk),
        .rgb      (point_rgb),
        .rgb_en   (point_rgb_en),
        .gs       (point_gs),
        .gs_valid (point_gs_valid)
    );

    always @(clk) begin
        case (state)
            READY: begin
                if (input_en) begin
                    output_valid_reg <= 0;
                    state <= RGB2GS;
                end
            end
            RGB2GS: begin
                point_rgb_en <= 1;
                state <= RGB2GS_DONE;
            end
            RGB2GS_DONE: begin
                if (point_gs_valid) begin
                    point_rgb_en <= 0;
                    state <= ALL_DONE;
                end
            end
            ALL_DONE: begin
                output_valid_reg <= 1;
                state <= READY;
            end
            default;
        endcase
    end
endmodule


module frame_preprocessor_tb;
    wire       clk;
    wire [7:0] point_output;
    wire       output_valid;
    reg [7:0]  r, g, b;
    reg [27:0] point_input_reg;
    reg        input_en_reg;
    
    integer expected_value = 67;

    clk_generator u_clk_generator(
        .sys_clk(clk)
    );

    frame_preprocessor u_frame_preprocessor(
        .clk          (clk),
        .point_input  (point_input_reg),
        .input_en     (input_en_reg),
        .point_output (point_output),
        .output_valid (output_valid)
    );

    initial begin
        r = 122;
        g = 23;
        b = 100;
        point_input_reg = {r, g, b};
        input_en_reg = 1;
    end

    always @(clk) begin
        if (output_valid) begin
            if (point_output == expected_value)
                $display("TEST PASSED");
            else
                $display("TEST FAILED: %d != %d", point_output, expected_value);
            $finish;
        end
    end
endmodule