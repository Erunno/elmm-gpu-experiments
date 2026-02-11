#include <iostream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <ctime>
#include <iomanip>
#include <sstream>

// INCLUDE THE EXACT SAME HEADER YOUR TOOL USES!
// This guarantees the struct padding and memory alignment are 100% identical.
#include "snapshot.hpp" 

// ---------------------------------------------------------
// Helper to generate dynamic file paths
// ---------------------------------------------------------
std::string get_snapshot_filepath(const char* suffix) {
    static std::string base_dir = "";
    static int call_id = 0;
    
    std::string suffix_str(suffix);
    
    if (base_dir.empty()) {
        std::time_t t = std::time(nullptr);
        char time_buf[100];
        std::strftime(time_buf, sizeof(time_buf), "%Y_%m_%d_%H%M%S", std::localtime(&t));
        
        mkdir("bin_snapshots", 0777);
        base_dir = "bin_snapshots/" + std::string(time_buf);
        mkdir(base_dir.c_str(), 0777);
    }
    
    if (suffix_str.find("input") != std::string::npos) {
        call_id++;
    }
    
    std::ostringstream oss;
    oss << base_dir << "/call_" << std::setfill('0') << std::setw(3) << call_id << "_" << suffix_str;
    
    return oss.str();
}

// ---------------------------------------------------------
// FORTRAN BINDING
// ---------------------------------------------------------

// Helper template to correctly deduce the type (float or double)
template <typename RealType>
void dump_cdv_snapshot_tmpl(const char* suffix, RealType* U, RealType* V, RealType* W, RealType* V2, std::size_t total_size) {
    snapshot::Dumper dumper;
    
    // Register the arrays. TypeNamer will now safely resolve.
    dumper.register_array("U", U, total_size);
    dumper.register_array("V", V, total_size);
    dumper.register_array("W", W, total_size);
    dumper.register_array("V2", V2, total_size);
    
    std::string full_path = get_snapshot_filepath(suffix);
    
    dumper.dump_to_file(full_path);
    std::cout << "Successfully dumped snapshot to: " << full_path << std::endl;
}

extern "C" {
    void dump_cdv_snapshot(const char* suffix, void* U, void* V, void* W, void* V2, std::size_t total_size, std::size_t knd_bytes) {
        if (knd_bytes == 4) {
            dump_cdv_snapshot_tmpl(suffix, (float*)U, (float*)V, (float*)W, (float*)V2, total_size);
        } else if (knd_bytes == 8) {
            dump_cdv_snapshot_tmpl(suffix, (double*)U, (double*)V, (double*)W, (double*)V2, total_size);
        } else {
            std::cerr << "Snapshot Error: Unsupported real kind size (" << knd_bytes << " bytes)" << std::endl;
        }
    }
}