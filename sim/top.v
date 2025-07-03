`timescale 1ps/100fs 

module top;
    localparam BYTE = 8;
    localparam N_BITS_ONE_POINT = BYTE * 3;
    localparam CAMERA_WIDTH_BUS_N_POINTS = 4;
    localparam CAMERA_WIDTH_BUS = N_BITS_ONE_POINT * CAMERA_WIDTH_BUS_N_POINTS;
    localparam STRAT_ADDRESS = 32'd0;
    localparam DDR3_BUS_WIDTH = 32;
    
    localparam INPUT_IMAGE_SHAPE_H   = 16;
    localparam INPUT_IMAGE_SHAPE_W   = 16;
    localparam INPUT_IMAGE_SHAPE_CH  = 3;

    reg sys_rst = 0;
    initial
      #1000 sys_rst = 1;

    // Переменные для чтения окон изображения
    reg [2:0] ddr3_reading_wnd_state;
    localparam DDR3_READING_WND_READY = 3'd0;
    localparam DDR3_READING_WND_REQUEST = 3'd1;
    localparam DDR3_READING_WND_WAIT = 3'd2;
    localparam DDR3_READING_WND_DONE = 3'd3;

    reg [9:0] wnd_i = 4, wnd_j = 4;
    reg [5:0] ddr3_reading_wnd_cur_i;
    reg       shfit_to_next_row;
    localparam HALF_WS = 4;
    localparam SHIFT_TO_START_WND = (HALF_WS * INPUT_IMAGE_SHAPE_W + HALF_WS) * BYTE;
    localparam SHIFT_TO_END_WND = ((HALF_WS - 1) * INPUT_IMAGE_SHAPE_W + (HALF_WS - 1)) * BYTE;
    localparam WS = HALF_WS * 4;

    reg [511:0] wnd_values_reg;

    // Переменные для вычисления статистики
    reg         statistics_rst_reg = 0;
    reg         mean_input_valid_reg = 0;
    reg         var_input_valid_reg = 0;
    wire [7:0]  mean_value;
    wire [13:0] var_value;
    wire        mean_output_valid;
    wire        var_output_valid;

    // Переменные для вычисления сдвигов точек между кадрами
    reg [2:0] compute_shifts_state;
    localparam COMPUTE_SHIFTS_IDLE = 3'd0;
    localparam COMPUTE_SHIFTS_START = 3'd1;
    localparam COMPUTE_SHIFTS_READ_WND = 3'd2;
    localparam COMPUTE_SHIFTS_COMPUTE_WND_STATISTICS = 3'd3;
    localparam COMPUTE_SHIFTS_SEARCH_PAIR_POINT = 3'd4;
    localparam COMPUTE_SHIFTS_TO_NEXT_POINT = 3'd7;

    localparam START_WND_I = 4;
    localparam START_WND_J = 4;
    localparam COMPUTE_SHIFTS_VAR_LIM = 100;


      ///////////////////////////////////////////////////////////////////////////////////////////
     //Тактовый генератор///////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////    
  
    wire sys_clk;
    wire ref_clk;

    clk_generator u_clk_generator(
        .sys_clk(sys_clk),
        .ref_clk(ref_clk)
    );
     
      ///////////////////////////////////////////////////////////////////////////////////////////
     //Память///////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////
    
    localparam DDR_WRITE_CMD = 3'b0;
    localparam DDR_READ_CMD = 3'b1;

    reg  [27:0]                 app_addr_reg;
    reg  [2:0]                  app_cmd_reg;
    reg                         app_en_reg;
    reg  [DDR3_BUS_WIDTH - 1:0] app_wdf_data_reg;
    reg  [3:0]                  app_wdf_mask_reg;
    reg                         app_wdf_wren_reg;
    reg                         app_wdf_end_reg;
    wire [DDR3_BUS_WIDTH - 1:0] app_rd_data;
    wire                        app_rd_data_end;
    wire                        app_rd_data_valid;
    wire                        app_rdy;
    wire                        app_wdf_rdy;
    wire                        init_calib_complete;
    wire                        ui_clk;
    wire                        ui_clk_sync_rst;

    ddr3_top u_ddr3_top (
        .app_addr              (app_addr_reg),
        .app_cmd               (app_cmd_reg),
        .app_en                (app_en_reg),
        .app_wdf_data          (app_wdf_data_reg),
        .app_wdf_mask          (app_wdf_mask_reg),
        .app_wdf_wren          (app_wdf_wren_reg),
        .app_wdf_end           (app_wdf_end_reg),
        .app_rd_data           (app_rd_data),
        .app_rd_data_end       (app_rd_data_end),
        .app_rd_data_valid     (app_rd_data_valid),
        .app_rdy               (app_rdy),
        .app_wdf_rdy           (app_wdf_rdy),
        .sys_clk_i             (sys_clk),
        .clk_ref_i             (ref_clk),
        .sys_rst               (sys_rst),
        .init_calib_complete   (init_calib_complete),
        .ui_clk                (ui_clk),
        .ui_clk_sync_rst       (ui_clk_sync_rst)
    );
    
    always @ (posedge ui_clk) begin
        if (ui_clk_sync_rst) begin
            app_en_reg <= 0;
            app_wdf_wren_reg <= 0;
        end 
    end
   
      ///////////////////////////////////////////////////////////////////////////////////////////
     //Камера///////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////

    reg                           camera_recieve_ready_reg;
    wire                          camera_recieve_ready;
    wire                          camera_in_progress;
    wire [CAMERA_WIDTH_BUS - 1:0] camera_data;
    wire                          camera_data_valid;
    wire                          camera_frame_end;

    assign camera_recieve_ready = camera_recieve_ready_reg;

    camera #(
        .BUS_WIDTH (CAMERA_WIDTH_BUS),
        .SHAPE_H   (INPUT_IMAGE_SHAPE_H),
        .SHAPE_W   (INPUT_IMAGE_SHAPE_W)
    ) 
    u_camera (
        .clk           (sys_clk),
        .recieve_ready (camera_recieve_ready),
        .in_progress   (camera_in_progress),
        .data          (camera_data),
        .data_valid    (camera_data_valid),
        .frame_end     (camera_frame_end)
    );
   
      ///////////////////////////////////////////////////////////////////////////////////////////
     //Тестовая запись в файл///////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////

    localparam STORAGE_BUS_WIDTH = 32;

    reg  [STORAGE_BUS_WIDTH - 1:0] storage_input_data_reg;
    reg  storage_input_valid_reg;
    wire [STORAGE_BUS_WIDTH - 1:0] storage_input_data;
    wire storage_input_valid;
    wire storage_ready;

    assign storage_input_data = storage_input_data_reg;
    assign storage_input_valid = storage_input_valid_reg;
   
    storage #(
        .BUS_WIDTH (STORAGE_BUS_WIDTH)
    )
    u_storage (
       .clk         (sys_clk),
       .input_data  (storage_input_data),
       .input_valid (storage_input_valid),
       .ready       (storage_ready)
    );

      ///////////////////////////////////////////////////////////////////////////////////////////
     //Предобработка данных с камеры////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////
    localparam PREPROCESSING_READY                   = 4'd0;
    localparam PREPROCESSING_WAIT_CAMERA_IN_PROGRESS = 4'd1;
    localparam PREPROCESSING_WAIT_CAMERA_DATA        = 4'd2;
    localparam PREPROCESSING_WAIT                    = 4'd3;
    localparam PREPROCESSING_WRITE_RESULT            = 4'd4;
    localparam PREPROCESSING_WAIT_WRITING            = 4'd5;

    localparam PREPROCESSING_READ_TEST               = 4'd11;
    localparam PREPROCESSING_WAIT_READING_TEST       = 4'd12;
    localparam PREPROCESSING_WRITE_STORAGE_TEST      = 4'd13;
    localparam PREPROCESSING_WRITE_STORAGE_DONE_TEST = 4'd14;
    localparam PREPROCESSING_IDLE                    = 4'd15;

    reg [3:0]  preprocessing_state = PREPROCESSING_READY;
    reg [31:0] cur_mem_addr = STRAT_ADDRESS;
    reg [31:0] n_bits_mem_read;

    reg                                           preprocessor_input_en_reg;
    reg  [CAMERA_WIDTH_BUS - 1:0]                 preprocessor_point_input_reg;
    wire [N_BITS_ONE_POINT - 1:0]                 preprocessor_point_input [CAMERA_WIDTH_BUS_N_POINTS - 1:0];
    wire                                          preprocessor_input_en;
    wire [CAMERA_WIDTH_BUS_N_POINTS * BYTE - 1:0] preprocessor_point_output;
    wire [CAMERA_WIDTH_BUS_N_POINTS - 1:0]        preprocessor_output_valid;

    assign preprocessor_input_en = preprocessor_input_en_reg;

    genvar i;
    integer j;

    generate 
        for (i = CAMERA_WIDTH_BUS_N_POINTS; i > 0; i = i - 1) begin:  gen_preprocessor_input
            assign preprocessor_point_input[i - 1] = preprocessor_point_input_reg[i * N_BITS_ONE_POINT - 1:(i - 1) * N_BITS_ONE_POINT];
        end
    endgenerate

    generate 
        for (i = CAMERA_WIDTH_BUS_N_POINTS - 1; i >= 0; i = i - 1) begin: gen_preprocessor_units
            frame_preprocessor u_frame_preprocessor(
                .clk                   (sys_clk),
                .point_input           (preprocessor_point_input[i]),
                .input_en              (preprocessor_input_en),
                .point_output          (preprocessor_point_output[(i + 1) * BYTE - 1:i * BYTE]),
                .output_valid          (preprocessor_output_valid[i])
            );
        end
    endgenerate

    always @(sys_clk) begin
        case (preprocessing_state)
            PREPROCESSING_READY: begin
                if (camera_frame_end) begin
                    $display("FRAME END!!!");
                    n_bits_mem_read <= cur_mem_addr - STRAT_ADDRESS;
                    preprocessing_state <= PREPROCESSING_IDLE;
                    compute_shifts_state <= COMPUTE_SHIFTS_START;
                end
                else if (init_calib_complete) begin // условие готовности следующих шагов
                    camera_recieve_ready_reg <= 1;
                    preprocessing_state <= PREPROCESSING_WAIT_CAMERA_IN_PROGRESS;
                end
            end
            PREPROCESSING_WAIT_CAMERA_IN_PROGRESS: begin
                if (camera_in_progress) begin
                    camera_recieve_ready_reg <= 0;
                    preprocessing_state <= PREPROCESSING_WAIT_CAMERA_DATA;
                end
            end
            PREPROCESSING_WAIT_CAMERA_DATA: begin
                if (camera_data_valid) begin
                    // TODO: Если данные с камер пришли, перепишим их в регистр,
                    //  и дадим камере передать следующие значения.
                    //  Для этого нужно разделить получение данных и препроцессинг
                    preprocessor_point_input_reg <= camera_data;
                    preprocessor_input_en_reg <= 1;
                    preprocessing_state <= PREPROCESSING_WAIT;
                end
            end
            PREPROCESSING_WAIT: begin
                // Все единицы
                if (&preprocessor_output_valid & preprocessor_input_en_reg) begin
                    preprocessor_input_en_reg <= 0;
                    preprocessing_state <= PREPROCESSING_WRITE_RESULT;
                    $display("%d %d %d %d", preprocessor_point_output[31:24], preprocessor_point_output[23:16], preprocessor_point_output[15:8], preprocessor_point_output[7:0]);
                    $display("preproc result %h", preprocessor_point_output);
                end
            end
            default;
        endcase
    end

      ///////////////////////////////////////////////////////////////////////////////////////////
     //Выбор точек для вычисления преобразования////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////
    mean u_mean (
        .clk          (sys_clk),
        .rst          (statistics_rst_reg),
        .values       (wnd_values_reg),
        .input_valid  (mean_input_valid_reg),
        .mean_value   (mean_value),
        .output_valid (mean_output_valid)
    );

    var u_var (
        .clk          (sys_clk),
        .rst          (statistics_rst_reg),
        .values       (wnd_values_reg),
        .mean_value   (mean_value),
        .input_valid  (var_input_valid_reg),
        .var_value    (var_value),
        .output_valid (var_output_valid)
    );

    always @(sys_clk) begin
        case (compute_shifts_state)
            COMPUTE_SHIFTS_START: begin
                $display("COMPUTE_SHIFTS_START");
                statistics_rst_reg <= 0;
                wnd_i <= START_WND_I;
                wnd_j <= START_WND_J;
                compute_shifts_state <= COMPUTE_SHIFTS_READ_WND;
            end
            COMPUTE_SHIFTS_READ_WND: begin
                $display("COMPUTE_SHIFTS_READ_WND cur_point (%d, %d)", wnd_i, wnd_j);
                ddr3_reading_wnd_state <= DDR3_READING_WND_READY;
                compute_shifts_state <= COMPUTE_SHIFTS_COMPUTE_WND_STATISTICS;
                statistics_rst_reg <= 0;
            end
            COMPUTE_SHIFTS_COMPUTE_WND_STATISTICS: begin
                // $display("COMPUTE_SHIFTS_COMPUTE_WND_STATISTICS");
                if (ddr3_reading_wnd_state == DDR3_READING_WND_DONE) begin
                    mean_input_valid_reg <= 1;
                    if (mean_output_valid) begin
                        var_input_valid_reg <= 1;
                        if (var_output_valid) begin
                            $display("Point (%d, %d) MEAN COMPUTED: %d", wnd_i, wnd_j, mean_value);
                            $display("Point (%d, %d) VAR COMPUTED: %d", wnd_i, wnd_j, var_value);
                            compute_shifts_state <= var_value >= COMPUTE_SHIFTS_VAR_LIM ? COMPUTE_SHIFTS_SEARCH_PAIR_POINT : COMPUTE_SHIFTS_TO_NEXT_POINT;
                            mean_input_valid_reg <= 0;
                            var_input_valid_reg <= 0;
                            statistics_rst_reg <= 1;
                        end
                    end
                end
            end
            COMPUTE_SHIFTS_SEARCH_PAIR_POINT: begin
                $display("Point (%d, %d) is accepted! Pair can be search! But not implemented yet", wnd_i, wnd_j);
                compute_shifts_state <= COMPUTE_SHIFTS_TO_NEXT_POINT;
            end
            COMPUTE_SHIFTS_TO_NEXT_POINT: begin
                compute_shifts_state = COMPUTE_SHIFTS_READ_WND;
                wnd_j = wnd_j + 8;
                if (wnd_j >= INPUT_IMAGE_SHAPE_W) begin
                    wnd_j = START_WND_J;
                    wnd_i = wnd_i + 8;
                    if (wnd_i >= INPUT_IMAGE_SHAPE_H)
                        compute_shifts_state = COMPUTE_SHIFTS_IDLE;
                end
            end
            COMPUTE_SHIFTS_IDLE: begin
                $display("All points are checked!");
                $finish;
            end
        endcase
    end

    // Запись
    always @ (posedge ui_clk) begin
        case (preprocessing_state)
            PREPROCESSING_WRITE_RESULT: begin
                if (app_rdy & app_wdf_rdy) begin
                    app_cmd_reg = DDR_WRITE_CMD; // Команда записать
                    app_addr_reg = cur_mem_addr; // Адрес
                    app_en_reg = 1; // Подтверждение адреса и команды
                    app_wdf_data_reg = preprocessor_point_output; // Данные
                    app_wdf_mask_reg = 0; // Маска данных
                    app_wdf_wren_reg = 1; // Подтверждение данных
                    app_wdf_end_reg = 1;
                    preprocessing_state = PREPROCESSING_WAIT_WRITING;
                end
            end
            PREPROCESSING_WAIT_WRITING: begin
                if (app_rdy & app_en_reg) begin
                    app_en_reg <= 0;
                end
                if (app_wdf_rdy & app_wdf_wren_reg) begin
                    app_wdf_wren_reg <= 0;
                end
                if (~app_en_reg & ~app_wdf_wren_reg) begin
                    $display("write %h to %h", app_wdf_data_reg, cur_mem_addr);
                    cur_mem_addr = cur_mem_addr + BYTE * CAMERA_WIDTH_BUS_N_POINTS;
                    preprocessing_state = PREPROCESSING_READY;
                end
            end
            default;
        endcase
    end 

    reg [2:0] ddr3_reading_state;

    localparam DDR3_READING_START = 3'd0;
    localparam DDR3_READING_WAIT = 3'd1;
    localparam DDR3_READING_DONE = 3'd2;

    function [27:0] ij_to_address (
        input [27:0] start_address,
        input [9:0]  i,
        input [9:0]  j,
        input [9:0]  shape_j,
        input [5:0]  size_of_type
    );
        begin
            ij_to_address = start_address + (i * shape_j + j) * size_of_type;
        end
    endfunction

    // Чтение
    always @ (posedge ui_clk) begin
        case (ddr3_reading_state)
            DDR3_READING_START: begin
                $display("DDR3_READING_START");
                if (app_rdy) begin
                    app_cmd_reg = DDR_READ_CMD; // Команда читать
                    app_addr_reg = cur_mem_addr; // Адрес
                    app_en_reg = 1;
                    ddr3_reading_state = DDR3_READING_WAIT;
                end
            end
            DDR3_READING_WAIT: begin
                $display("DDR3_READING_WAIT");
                if (app_rdy & app_en_reg) 
                    app_en_reg = 0;
                if (app_rd_data_valid) begin
                    storage_input_data_reg = app_rd_data;
                    ddr3_reading_state = DDR3_READING_DONE;
                end
            end
            default;
        endcase
    end

    reg [9:0] idx_i = 0;
    reg [9:0] idx_j = 0;

    // Почему-то не всегда корректно на sys_clk
    always @ (posedge ui_clk) begin
        case (preprocessing_state)
            PREPROCESSING_READ_TEST: begin
                $display("PREPROCESSING_READ_TEST");
                cur_mem_addr = ij_to_address(STRAT_ADDRESS, idx_i, idx_j, INPUT_IMAGE_SHAPE_W, BYTE);
                ddr3_reading_state = DDR3_READING_START;
                preprocessing_state = PREPROCESSING_WRITE_STORAGE_TEST;
            end
            PREPROCESSING_WRITE_STORAGE_TEST: begin
                $display("PREPROCESSING_WRITE_STORAGE_TEST");
                if (ddr3_reading_state == DDR3_READING_DONE) begin
                    storage_input_valid_reg <= 1;
                    preprocessing_state <= PREPROCESSING_WRITE_STORAGE_DONE_TEST;
                end
            end
            PREPROCESSING_WRITE_STORAGE_DONE_TEST: begin
                $display("PREPROCESSING_WRITE_STORAGE_DONE_TEST");
                if (storage_ready) begin
                    storage_input_valid_reg = 0;
                    idx_j = idx_j + 4;
                    if (idx_j > INPUT_IMAGE_SHAPE_W) begin
                        idx_i = idx_i + 1;
                        idx_j = idx_j - INPUT_IMAGE_SHAPE_W;
                    end
                    n_bits_mem_read = n_bits_mem_read - 32'd32;
                    if (n_bits_mem_read > 0)
                        preprocessing_state = PREPROCESSING_READ_TEST;
                    else
                        $finish;
                        // preprocessing_state <= PREPROCESSING_READY;
                end
            end
            default;
        endcase
    end

    // читаем окно
    always @ (posedge ui_clk) begin
        case (ddr3_reading_wnd_state)
            DDR3_READING_WND_READY: begin
                $display("DDR3_READING_WND_READY");
                cur_mem_addr <= ij_to_address(STRAT_ADDRESS, wnd_i, wnd_j, INPUT_IMAGE_SHAPE_W, BYTE) - SHIFT_TO_START_WND;
                ddr3_reading_wnd_cur_i <= 0;
                shfit_to_next_row <= 0;
                ddr3_reading_wnd_state <= DDR3_READING_WND_REQUEST;
            end
            DDR3_READING_WND_REQUEST: begin
                $display("DDR3_READING_WND_REQUEST");
                ddr3_reading_state <= DDR3_READING_START;
                ddr3_reading_wnd_state <= DDR3_READING_WND_WAIT;
            end
            DDR3_READING_WND_WAIT: begin
                $display("DDR3_READING_WND_WAIT: cur_i %d next_row %d", ddr3_reading_wnd_cur_i, shfit_to_next_row);
                if (ddr3_reading_state == DDR3_READING_DONE) begin
                    wnd_values_reg[511:512 - DDR3_BUS_WIDTH] = storage_input_data_reg;
                    // За раз читаем 4 точки. Для окна в 8 точек строчку читаем за 2 сдвига.
                    // Для окна не кратного 4 нужны дополнительные проверки
                    $display("ddr3_reading_wnd %d %d %d %d", storage_input_data_reg[31:24], storage_input_data_reg[23:16], storage_input_data_reg[15:8], storage_input_data_reg[7:0]);
                    wnd_values_reg <= wnd_values_reg << DDR3_BUS_WIDTH;
                    wnd_values_reg[DDR3_BUS_WIDTH - 1:0] <= wnd_values_reg[511:512 - DDR3_BUS_WIDTH];
                    ddr3_reading_wnd_cur_i = ddr3_reading_wnd_cur_i + 4;
                    if (~|ddr3_reading_wnd_cur_i)
                        ddr3_reading_wnd_state <= DDR3_READING_WND_DONE;
                    else begin
                        cur_mem_addr <= shfit_to_next_row ? cur_mem_addr + BYTE * INPUT_IMAGE_SHAPE_W - DDR3_BUS_WIDTH : cur_mem_addr +  DDR3_BUS_WIDTH;
                        ddr3_reading_wnd_state <= DDR3_READING_WND_REQUEST;
                        shfit_to_next_row <= ~shfit_to_next_row;
                    end
                end
            end
            // DDR3_READING_WND_DONE: begin
            //     $display("DDR3_READING_WND_DONE");
            //     for (j = 512; j > 0; j = j - 8) begin
            //         $display("%d: %d", j / 8, wnd_values_reg[511:504]);
            //         wnd_values_reg = wnd_values_reg << 8;
            //     end
            //     $finish;
            // end
            default;
        endcase
    end
endmodule