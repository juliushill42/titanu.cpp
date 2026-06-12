#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <cstdint>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <pthread.h>
#include <sched.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

// ==========================================
// 1. SYSTEM-LEVEL OPTIMIZATIONS
// ==========================================

void lock_thread_to_realtime() {
    pthread_t this_thread = pthread_self();
    struct sched_param params{};
    params.sched_priority = sched_get_priority_max(SCHED_FIFO);
    if (pthread_setschedparam(this_thread, SCHED_FIFO, &params) != 0) {
        std::cerr << "[SYSTEM] Note: Run as root for strict real-time SCHED_FIFO priority.\n";
    }
}

void lock_thread_to_core(int core_id) {
    cpu_set_t cpuset;               // FIX: was cpu_set_set_t (typo, would not compile)
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
}

// ==========================================
// 2. HARDWARE MATH KERNELS
// ==========================================
//
// FIX: vdotq_s32 requires the ARM "dotprod" feature, which is NOT
// guaranteed by __ARM_NEON alone. Gate on __ARM_FEATURE_DOTPROD
// specifically. Provide a portable (non-dotprod) NEON fallback too,
// so the same binary can run on ARM cores without dotprod, and an
// x86 scalar fallback for dev-machine testing.

int32_t titan_vec_dot_int8(const int8_t* __restrict vec_a, const int8_t* __restrict vec_b, size_t len) {
#if defined(__ARM_FEATURE_DOTPROD)
    int32x4_t acc_vec = vdupq_n_s32(0);
    size_t i = 0;
    for (; i + 16 <= len; i += 16) {
        int8x16_t a = vld1q_s8(&vec_a[i]);
        int8x16_t b = vld1q_s8(&vec_b[i]);
        acc_vec = vdotq_s32(acc_vec, a, b);
    }
    int32_t buffer[4];
    vst1q_s32(buffer, acc_vec);
    int32_t total_sum = buffer[0] + buffer[1] + buffer[2] + buffer[3];
    for (; i < len; ++i) {
        total_sum += static_cast<int32_t>(vec_a[i]) * static_cast<int32_t>(vec_b[i]);
    }
    return total_sum;

#elif defined(__ARM_NEON)
    // Portable NEON fallback (no dotprod instruction available):
    // widen 8-bit lanes to 16-bit and accumulate with vmlal.
    int16x8_t acc_lo = vdupq_n_s16(0);
    int16x8_t acc_hi = vdupq_n_s16(0);
    size_t i = 0;
    for (; i + 16 <= len; i += 16) {
        int8x16_t a = vld1q_s8(&vec_a[i]);
        int8x16_t b = vld1q_s8(&vec_b[i]);
        int16x8_t prod_lo = vmull_s8(vget_low_s8(a), vget_low_s8(b));
        int16x8_t prod_hi = vmull_s8(vget_high_s8(a), vget_high_s8(b));
        acc_lo = vaddq_s16(acc_lo, prod_lo);
        acc_hi = vaddq_s16(acc_hi, prod_hi);
    }
    int32_t total_sum = 0;
    for (int lane = 0; lane < 8; ++lane) {
        total_sum += acc_lo[lane];
        total_sum += acc_hi[lane];
    }
    for (; i < len; ++i) {
        total_sum += static_cast<int32_t>(vec_a[i]) * static_cast<int32_t>(vec_b[i]);
    }
    return total_sum;

#else
    // Scalar fallback (x86 dev builds, or any non-NEON target).
    int32_t total_sum = 0;
    for (size_t i = 0; i < len; ++i) {
        total_sum += static_cast<int32_t>(vec_a[i]) * static_cast<int32_t>(vec_b[i]);
    }
    return total_sum;
#endif
}

// ==========================================
// 3. ZERO-COPY MODEL LOADER (.ttn)
// ==========================================

struct ttn_header {
    uint32_t magic;
    uint32_t version;
    uint32_t latent_dim;
    uint32_t tensor_count;
};

class TitanModel {
private:
    int fd;
    size_t file_size;
    void* mapped_data;

public:
    ttn_header* header;
    int8_t* weights_data; // Pointer to quantized weights
    size_t weights_size;  // FIX: track usable weight bytes for bounds checks

    TitanModel()
        : fd(-1), file_size(0), mapped_data(MAP_FAILED),
          header(nullptr), weights_data(nullptr), weights_size(0) {}

    bool load_mmap(const std::string& filepath) {
        fd = open(filepath.c_str(), O_RDONLY);
        if (fd < 0) {
            std::cerr << "[ERROR] Could not open model file.\n";
            return false;
        }

        struct stat sb;
        if (fstat(fd, &sb) == -1) {
            std::cerr << "[ERROR] fstat failed.\n";
            close(fd);
            fd = -1;
            return false;
        }
        file_size = static_cast<size_t>(sb.st_size);

        // FIX: minimum size check before touching header fields.
        if (file_size < sizeof(ttn_header)) {
            std::cerr << "[ERROR] File too small to contain a valid .ttn header.\n";
            close(fd);
            fd = -1;
            return false;
        }

        mapped_data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (mapped_data == MAP_FAILED) {
            std::cerr << "[ERROR] mmap failed.\n";
            close(fd);
            fd = -1;
            return false;
        }

        header = static_cast<ttn_header*>(mapped_data);
        if (header->magic != 0x4E544954) { // 'TITN'
            std::cerr << "[ERROR] Invalid .ttn magic number.\n";
            munmap(mapped_data, file_size);
            mapped_data = MAP_FAILED;
            close(fd);
            fd = -1;
            return false;
        }

        if (header->latent_dim == 0 || header->tensor_count == 0) {
            std::cerr << "[ERROR] Invalid .ttn header: zero latent_dim or tensor_count.\n";
            munmap(mapped_data, file_size);
            mapped_data = MAP_FAILED;
            close(fd);
            fd = -1;
            return false;
        }

        // Offset weights past header and metadata.
        size_t meta_offset = sizeof(ttn_header) + static_cast<size_t>(header->tensor_count) * 80;

        // FIX: validate that meta_offset and the expected weights region
        // actually fit inside the mapped file, before exposing the pointer.
        if (meta_offset > file_size) {
            std::cerr << "[ERROR] .ttn metadata offset exceeds file size (truncated/corrupt file).\n";
            munmap(mapped_data, file_size);
            mapped_data = MAP_FAILED;
            close(fd);
            fd = -1;
            return false;
        }

        weights_data = static_cast<int8_t*>(mapped_data) + meta_offset;
        weights_size = file_size - meta_offset;

        // FIX: ensure there's at least latent_dim bytes of weight data,
        // since titan_vec_dot_int8 reads latent_dim bytes from weights_data.
        if (weights_size < header->latent_dim) {
            std::cerr << "[ERROR] .ttn weights region smaller than latent_dim "
                         "(truncated/corrupt file).\n";
            munmap(mapped_data, file_size);
            mapped_data = MAP_FAILED;
            weights_data = nullptr;
            close(fd);
            fd = -1;
            return false;
        }

        std::cout << "[TITAN] Loaded World Model via mmap. Latent Dim: "
                  << header->latent_dim << "\n";
        return true;
    }

    ~TitanModel() {
        if (mapped_data != MAP_FAILED) munmap(mapped_data, file_size);
        if (fd >= 0) close(fd);
    }
};

// ==========================================
// 4. SPATIAL CONTEXT & VISION PIPELINE
// ==========================================

class TitanContext {
private:
    TitanModel* model;
    std::vector<int8_t> latent_state;
    std::vector<int8_t> input_buffer;

public:
    explicit TitanContext(TitanModel* m) : model(m) {
        latent_state.resize(model->header->latent_dim, 0);
        input_buffer.resize(model->header->latent_dim, 0);
    }

    int8_t* get_input_buffer() { return input_buffer.data(); }
    size_t input_buffer_size() const { return input_buffer.size(); }

    void observe_frame() {
        int32_t state_update = titan_vec_dot_int8(
            input_buffer.data(), model->weights_data, model->header->latent_dim);

        // FIX: previous version used `%` directly on a value that could be
        // negative (UB-adjacent / inconsistent sign behavior for an int8
        // "activation"). Clamp into the valid int8 range [-127, 127] instead
        // of relying on modulo of a possibly-negative sum.
        for (size_t i = 0; i < latent_state.size(); i++) {
            int32_t v = static_cast<int32_t>(latent_state[i]) + state_update;
            if (v > 127) v = 127;
            if (v < -127) v = -127;
            latent_state[i] = static_cast<int8_t>(v);
        }
    }

    int infer_action() {
        int32_t action_sum = 0;
        for (auto val : latent_state) action_sum += val;
        return std::abs(action_sum % 8); // 8 potential directional actions
    }
};

class TitanVision {
private:
    int8_t* target_tensor;
    size_t target_capacity; // FIX: bound writes to the actual buffer size

public:
    TitanVision(int8_t* buffer, size_t capacity)
        : target_tensor(buffer), target_capacity(capacity) {}

    void process_stream(const uint8_t* raw_packet, size_t packet_size, TitanContext& context) {
        // FIX: write_size must never exceed target_capacity, regardless of
        // packet_size or infer_action()'s output, or this overflows the
        // latent input buffer.
        size_t write_size = std::min(packet_size, target_capacity);
        if (write_size > 0) {
            std::memset(target_tensor, raw_packet[0], write_size);
        }
        context.observe_frame();
    }
};

// ==========================================
// 5. MAIN EXECUTION LOOP
// ==========================================

int main() {
    std::cout << "Booting Titan Edge Inference Engine...\n";

    lock_thread_to_realtime();
    lock_thread_to_core(0);

    const char* model_path = "drone_world_model_v1.ttn";
    if (access(model_path, F_OK) == -1) {
        std::cout << "[INIT] Generating zero-copy target model: " << model_path << "\n";
        FILE* f = fopen(model_path, "wb");
        if (!f) {
            std::cerr << "[ERROR] Could not create model file.\n";
            return 1;
        }
        ttn_header dummy_header = {0x4E544954, 1, 4096, 1}; // 4096 latent dim
        fwrite(&dummy_header, sizeof(ttn_header), 1, f);

        // FIX: original wrote 4096*2 bytes of weights but meta_offset is
        // sizeof(header) + tensor_count*80, and the loader expects at least
        // latent_dim (4096) bytes available *after* meta_offset. Write
        // tensor_count*80 padding bytes plus latent_dim weight bytes so the
        // generated file is self-consistent with load_mmap()'s checks.
        std::vector<int8_t> meta_padding(dummy_header.tensor_count * 80, 0);
        fwrite(meta_padding.data(), 1, meta_padding.size(), f);

        std::vector<int8_t> dummy_weights(dummy_header.latent_dim, 1);
        fwrite(dummy_weights.data(), 1, dummy_weights.size(), f);
        fclose(f);
    }

    TitanModel model;
    if (!model.load_mmap(model_path)) return 1;

    TitanContext context(&model);
    TitanVision vision(context.get_input_buffer(), context.input_buffer_size());

    std::cout << "[TITAN] System Armed. Entering real-time control loop.\n";

    uint8_t dummy_h264_packet[1024];
    std::memset(dummy_h264_packet, 255, sizeof(dummy_h264_packet));

    for (int frame = 0; frame < 100; ++frame) {
        vision.process_stream(dummy_h264_packet, sizeof(dummy_h264_packet), context);

        int action = context.infer_action();

        if (frame % 20 == 0) {
            std::cout << "Frame " << frame << " | Latent Update Processed | Executing Action ID: "
                      << action << "\n";
        }
    }

    std::cout << "[TITAN] Execution Complete. Clean shutdown.\n";
    return 0;
}
