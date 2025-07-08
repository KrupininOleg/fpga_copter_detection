`timescale 1ps/100fs 

module storage #(
    parameter BUS_WIDTH = 32
)
(
    input                   clk,
    input [BUS_WIDTH - 1:0] input_data,
    input                   input_valid,
    output                  ready
);
    localparam READY = 3'd0;
    localparam IN_PROGRESS = 3'd1;
    localparam DONE = 3'd2;

    integer file_uid;

    reg [2:0] state;
    reg ready_reg;
    reg in_progress_reg;

    assign ready = ready_reg;

    initial begin
        file_uid <= $fopen("C:/Users/Public/OwnPrograms/stereo_vision/fpga/fpga_copter_detection/sim/data/results/gs.bin", "wb");
        state <= READY;
    end

    always @(posedge clk) begin
        case (state)
            READY: begin
                if (input_valid) begin
                    // $display("STORAGE READY");
                    ready_reg <= 0;
                    state <= IN_PROGRESS;
                end
            end
            IN_PROGRESS: begin
                if (input_valid) begin
                    // $display("STORAGE IN_PROGRESS");
                    $fwrite(file_uid, "%h", input_data);
                    $display("write to file %b", input_data);
                    ready_reg <= 1;
                    state <= DONE;
                end
            end
            DONE: begin
                if (~input_valid) begin
                    // $display("STORAGE DONE");
                    state <= READY;
                end
            end
        endcase
    end
endmodule