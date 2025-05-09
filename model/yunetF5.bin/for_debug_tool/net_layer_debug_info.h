struct cnn_one_layer_debug_info{
    uint8_t  layer_index; // start from 0
    uint8_t  mem_type; // 0: in ddr, 1: in iram
    uint8_t  int_type; // 1: feature map 8bit, 0: feature map 16bit
    int8_t   output_fraction;
    uint32_t output_start_addr;
    uint32_t output_channel;
    uint32_t output_height;
    uint32_t output_valid_elem_per_row; // how many elements(8bit or 16bit) in a row of feature map
    uint32_t output_size; // how many bytes occupied by output
};
struct cnn_one_layer_debug_info debug_info[] = {
    {0, 0, 1, -2, 25600, 3, 160, 160, 76800},
    {1, 0, 1, 4, 102400, 16, 80, 80, 122880},
    {2, 0, 1, 5, 225280, 16, 80, 80, 122880},
    {3, 0, 1, 4, 348160, 16, 80, 80, 122880},
    {4, 0, 1, 4, 25600, 16, 40, 40, 40960},
    {5, 0, 1, 4, 66560, 16, 40, 40, 40960},
    {6, 0, 1, 4, 107520, 16, 40, 40, 40960},
    {7, 0, 1, 5, 148480, 64, 40, 40, 163840},
    {8, 0, 1, 4, 312320, 64, 40, 40, 163840},
    {9, 0, 1, 5, 476160, 64, 40, 40, 163840},
    {10, 0, 1, 5, 25600, 64, 40, 40, 163840},
    {11, 0, 1, 6, 189440, 64, 40, 40, 163840},
    {12, 0, 1, 5, 353280, 64, 40, 40, 163840},
    {13, 1, 1, 5, 0, 64, 20, 20, 40960},
    {14, 1, 1, 6, 0, 64, 20, 20, 40960},
    {15, 1, 1, 4, 0, 64, 20, 20, 40960},
    {16, 1, 1, 5, 0, 64, 20, 20, 40960},
    {17, 0, 1, 5, 25600, 64, 20, 20, 40960},
    {18, 1, 1, 5, 0, 64, 10, 10, 20480},
    {19, 1, 1, 6, 0, 64, 10, 10, 20480},
    {20, 1, 1, 5, 0, 64, 10, 10, 20480},
    {21, 1, 1, 6, 0, 64, 10, 10, 20480},
    {22, 0, 1, 5, 66560, 64, 10, 10, 20480},
    {23, 1, 1, 5, 0, 64, 5, 5, 10240},
    {24, 1, 1, 7, 0, 64, 5, 5, 10240},
    {25, 1, 1, 5, 0, 64, 5, 5, 10240},
    {26, 1, 1, 7, 0, 64, 5, 5, 10240},
    {27, 1, 1, 6, 0, 64, 5, 5, 10240},
    {28, 1, 1, 7, 0, 64, 5, 5, 10240},
    {29, 1, 1, 4, 0, 64, 5, 5, 10240},
    {30, 1, 1, 4, 0, 34, 5, 5, 5440},
    {31, 0, 1, 3, 87040, 34, 5, 5, 5440},
    {32, 1, 1, 6, 0, 64, 10, 10, 20480},
    {33, 1, 1, 4, 0, 64, 10, 10, 20480},
    {34, 1, 1, 4, 0, 34, 10, 10, 10880},
    {35, 0, 1, 3, 66560, 34, 10, 10, 10880},
    {36, 1, 1, 6, 0, 64, 20, 20, 40960},
    {37, 1, 1, 4, 0, 64, 20, 20, 40960},
    {38, 1, 1, 4, 0, 51, 20, 20, 32640},
    {39, 0, 1, 3, 25600, 51, 20, 20, 32640}
};
uint32_t input_size = 25600;
uint32_t total_weight_size = 91904;
uint32_t features_total_size = 640000;
