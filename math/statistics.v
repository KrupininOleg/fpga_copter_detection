`timescale 1ps/100fs 


module mean #(
    parameter DEBUG = 0,
    parameter SIZE = 512,       // 64 * 8
    parameter LOG2_N_VALUES = 6 // 64 = 2 ^ 6 
)
(
    input               clk,
    input               rst,
    input  [SIZE - 1:0] values,
    input               input_valid,
    output [7:0]        mean_value,
    output              output_valid
);
    localparam N_VALUES = SIZE >> 3;
    localparam READY = 3'd0;
    localparam IN_PROGRESS = 3'd1;
    localparam DONE = 3'd2;

    reg [2:0] state = READY;
    
    // При каждом сложении может быть переполнение старшего разряда,
    // поэтому нужен размер 8 + (SIZE // 8 - 1)
    reg [7 + N_VALUES - 1:0] sum = 0;
    reg [SIZE - 1:0]         values_reg = 0;
    reg [7:0]                mean_value_reg = 0;
    reg                      output_valid_reg = 0;
    reg [9:0]                i = 0;

    assign mean_value = mean_value_reg;
    assign output_valid = output_valid_reg;

    always @(posedge clk) begin
        if (rst) begin
            output_valid_reg <= 0;
            sum <= 0;
            i <= 0;
            state <= READY;
        end
        case (state)
            READY: begin
                if (input_valid) begin
                    values_reg <= values;
                    state <= IN_PROGRESS;
                end
            end
            IN_PROGRESS: begin
                if (DEBUG) $display("UPDATE MEAN SUM %d + %d", sum,  values_reg[7:0]);
                sum <= sum + values_reg[7:0];
                values_reg <= values_reg >> 8;
                values_reg[SIZE - 1:SIZE - 8] <= values_reg[7:0];
                i = i + 8;

                if (i == SIZE)
                    state <= DONE;
            end
            DONE: begin
                if (~output_valid_reg) begin
                    sum = sum + {sum[LOG2_N_VALUES - 1], {(LOG2_N_VALUES - 1){1'b0}}}; // деление
                    mean_value_reg = sum >> LOG2_N_VALUES;                             // с округлением до целых
                    if (DEBUG) $display("COMPUTED MEAN %d from SUM %d", mean_value_reg, sum);
                    output_valid_reg = 1;
                end
            end
        endcase
    end
endmodule


module var #(
    parameter DEBUG = 0,
    parameter SIZE = 512,       // 64 * 8
    parameter LOG2_N_VALUES = 6 // 64 = 2 ^ 6 
)
(
    input               clk,
    input               rst,
    input  [SIZE - 1:0] values,
    input  [7:0]        mean_value,
    input               input_valid,
    output [13:0]       var_value,
    output              output_valid
);
    localparam READY = 3'd0;
    localparam IN_PROGRESS = 3'd1;
    localparam DONE = 3'd2;

    reg [2:0] state = READY;

    reg [7 + SIZE - 1:0] sum = 0;
    reg [15:0]           diff_reg = 0;
    reg [SIZE - 1:0]     values_reg = 0;
    reg [13:0]           var_value_reg = 0; // Максимальное значение (255 / 2) ^ 2 - 14 разрядов
    reg                  output_valid_reg = 0;
    reg [9:0]            i = 0;

    assign var_value = var_value_reg;
    assign output_valid = output_valid_reg;

    always @(posedge clk) begin
        if (rst) begin
            output_valid_reg <= 0;
            sum <= 0;
            i <= 0;
            state <= READY;
        end
        case (state)
            READY: begin
                if (input_valid) begin
                    values_reg <= values;
                    state <= IN_PROGRESS;
                end
            end
            IN_PROGRESS: begin
                if (DEBUG) $display("UPDATE VAR SUM %d + %d (%d)", sum, diff_reg, values_reg[7:0]);
                diff_reg = values_reg[7:0] - mean_value;
                diff_reg = diff_reg * diff_reg;
                sum = sum + diff_reg;

                values_reg <= values_reg >> 8;
                values_reg[SIZE - 1:SIZE - 8] <= values_reg[7:0];
                i = i + 8;

                if (i == SIZE)
                    state <= DONE;
            end
            DONE: begin
                if (~output_valid_reg) begin
                    sum = sum + {sum[LOG2_N_VALUES - 1], {(LOG2_N_VALUES - 1){1'b0}}}; // деление
                    var_value_reg = sum >> LOG2_N_VALUES;                              // с округлением до целых
                    if (DEBUG) $display("COMPUTED VAR %d from SUM %d", var_value_reg, sum);
                    output_valid_reg = 1;
                end
            end
        endcase
    end
endmodule


