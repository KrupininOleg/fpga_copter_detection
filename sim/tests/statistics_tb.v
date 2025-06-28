`timescale 1ps/100fs 


module mean_tb;
    integer input_file, code;
    localparam byte = 8;
    localparam ws_i = 8;
    localparam ws_j = 8;
    localparam n_bits_window = ws_i * ws_j * byte;

    wire                       clk;
    reg                        rst_reg = 0;
    reg                        input_valid_reg = 0;
    wire                       output_valid;
    wire [7:0]                 mean_value;
    reg  [n_bits_window - 1:0] window_reg;

    clk_generator u_clk_generator(
        .sys_clk (clk)
    );

    mean u_mean (
        .clk          (clk),
        .rst          (rst_reg),
        .values       (window_reg),
        .input_valid  (input_valid_reg),
        .mean_value   (mean_value),
        .output_valid (output_valid)
    );

    initial begin
        input_file = $fopen("C:/Users/Public/OwnPrograms/stereo_vision/fpga/fpga_copter_detection/sim/data/window_8x8_mean=124_var=1_9.bin", "rb");
        code = $fread(window_reg, input_file, 0, ws_i * ws_j);
    end

    integer exp_mean_value = 124;

    always @(clk) begin
        if (code > 0) begin
            if (~input_valid_reg)
                input_valid_reg = 1;
            else if (output_valid) begin
                if (mean_value == exp_mean_value)
                    $display("TEST PASSED");
                else
                    $display("TEST FAILED (%d != %d)", mean_value, exp_mean_value);
                $finish;
            end
        end
    end
endmodule


module var_tb;
    integer input_file, code;
    localparam byte = 8;
    localparam ws_i = 8;
    localparam ws_j = 8;
    localparam n_bits_window = ws_i * ws_j * byte;

    wire                       clk;
    reg                        rst_reg = 0;
    reg                        input_valid_reg = 0;
    reg  [7:0]                 mean_value_reg = 8'd124;
    wire                       output_valid;
    wire [7:0]                 var_value;
    reg  [n_bits_window - 1:0] window_reg;

    clk_generator u_clk_generator(
        .sys_clk (clk)
    );

    var u_var (
        .clk          (clk),
        .rst          (rst_reg),
        .values       (window_reg),
        .mean_value   (mean_value_reg),
        .input_valid  (input_valid_reg),
        .var_value    (var_value),
        .output_valid (output_valid)
    );

    initial begin
        input_file = $fopen("C:/Users/Public/OwnPrograms/stereo_vision/fpga/fpga_copter_detection/sim/data/window_8x8_mean=124_var=1_9.bin", "rb");
        code = $fread(window_reg, input_file, 0, ws_i * ws_j);
    end

    integer exp_var_value = 2;
    always @(clk) begin
        if (code > 0) begin
            if (~input_valid_reg)
                input_valid_reg = 1;
            else if (output_valid) begin
                if (var_value == exp_var_value)
                    $display("TEST PASSED");
                else
                    $display("TEST FAILED (%d != %d)", var_value, exp_var_value);
                $finish;
            end
        end
    end
endmodule
