`timescale 1ps/100fs 

module camera #(
    parameter BUS_WIDTH = 96,
    parameter SHAPE_H   = 480,
    parameter SHAPE_W   = 848,
    parameter SHAPE_CH  = 3
)
(
    input                    clk,
    input                    recieve_ready,
    output                   in_progress,
    output [BUS_WIDTH - 1:0] data,
    output                   data_valid,
    output                   frame_end
);
    localparam N_FRAMES_BITS = BUS_WIDTH * SHAPE_H * SHAPE_W * SHAPE_CH;
    // localparam N_FRAMES_BITS = 5000;

    integer input_file;
    integer code;
    integer n_bits_read;

    reg                   data_valid_reg;
    reg [BUS_WIDTH - 1:0] data_reg;
    reg                   frame_end_reg;
    reg                   in_progress_reg;

    assign data_valid = data_valid_reg;
    assign data = data_reg;
    assign frame_end = frame_end_reg;
    assign in_progress = in_progress_reg;

    initial begin
        input_file = $fopen("C:/Users/Public/OwnPrograms/stereo_vision/fpga/fpga_copter_detection/sim/data/running_0_2.bin", "rb");
        data_valid_reg = 0;
        n_bits_read = 0;
        frame_end_reg = 0;
        in_progress_reg = 0;
    end

    always @(posedge clk) begin
        if (recieve_ready & ~in_progress) begin
            in_progress_reg <= 1;
            data_valid_reg <= 0;
        end
        else if (~recieve_ready & in_progress) begin
            code = $fread(data_reg, input_file, 0, BUS_WIDTH);
            data_valid_reg = code > 0;
            if (data_valid_reg) begin
                n_bits_read = n_bits_read + BUS_WIDTH;
                $display("Read %d bits from %d", n_bits_read, N_FRAMES_BITS);
                if (n_bits_read >= N_FRAMES_BITS) begin
                    frame_end_reg = 1;
                    n_bits_read = 0;
                end
                in_progress_reg <= 0;
            end
        end
    end
endmodule


module camera_tb;
    wire sys_clk;
    reg recieve_ready;
    wire [23:0] data; 
    wire data_valid;
    wire frame_end;

    clk_generator u_clk_generator(
        .sys_clk(sys_clk)
    );

    camera u_camera(
        sys_clk,
        recieve_ready,
        data,
        data_valid,
        frame_end
    );

    initial begin
        recieve_ready = 1;
    end

    always @(posedge sys_clk) begin
        if (~recieve_ready) begin
            #500 recieve_ready = 1;
        end
        else if (data_valid) begin
            recieve_ready = 0;
            $display("%d %d %d", data[23:16], data[15:8], data[7:0]);
        end
    end

endmodule
