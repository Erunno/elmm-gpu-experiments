#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include <exception>

#include "snapshot.hpp" 

// --- ANSI Color Codes for Colorful Output ---
const std::string RESET   = "\033[0m";
const std::string BOLD    = "\033[1m";
const std::string RED     = "\033[31m";
const std::string GREEN   = "\033[32m";
const std::string YELLOW  = "\033[33m";
const std::string BLUE    = "\033[34m";
const std::string MAGENTA = "\033[35m";
const std::string CYAN    = "\033[36m";

void print_usage(const char* program_name) {
    std::cout << BOLD << "Usage:" << RESET << "\n";
    std::cout << "  " << program_name << " info <path_to_file>\n";
    std::cout << "  " << program_name << " compare <path_to_file_1> <path_to_file_2>\n\n";
    std::cout << "Commands:\n";
    std::cout << "  " << GREEN << "info" << RESET << "      Displays information about all arrays in a snapshot file.\n";
    std::cout << "  " << GREEN << "compare" << RESET << "   Compares two snapshot files and calculates differences.\n";
}

void handle_info(const std::string& filename) {
    try {
        std::cout << BOLD << CYAN << "Reading Snapshot: " << RESET << filename << "\n\n";
        
        snapshot::Loader loader;
        loader.load_from_file(filename);
        
        auto info_list = loader.get_dump_info();

        if (info_list.empty()) {
            std::cout << YELLOW << "The snapshot is empty or contains no valid arrays." << RESET << "\n";
        } else {
            std::cout << BOLD << "Total Arrays: " << RESET << loader.get_total_arrays() << "\n\n";

            // --- Print Table Header ---
            std::cout << BOLD 
                      << std::left << std::setw(40) << "Array Name" 
                      << std::setw(15) << "Data Type" 
                      << std::setw(15) << "Size (Bytes)" 
                      << RESET << "\n";
            std::cout << std::string(70, '-') << "\n";

            // --- Print Table Rows ---
            for (const auto& item : info_list) {
                std::cout << std::left 
                          << MAGENTA << std::setw(40) << item.name << RESET
                          << CYAN    << std::setw(15) << item.type << RESET
                          << GREEN   << std::setw(15) << item.size_bytes << RESET << "\n";
            }
            std::cout << "\n";
        }
    } catch (const std::exception& e) {
        std::cerr << BOLD << RED << "Error loading info: " << RESET << e.what() << "\n";
    }
}

void handle_compare(const std::string& file1, const std::string& file2) {
    try {
        std::cout << BOLD << CYAN << "Comparing Dumps:" << RESET << "\n";
        std::cout << "  File 1: " << file1 << "\n";
        std::cout << "  File 2: " << file2 << "\n\n";

        auto results = snapshot::DumpsComparer::compare_dumps(file1, file2);

        if (results.empty()) {
            std::cout << YELLOW << "No arrays found to compare." << RESET << "\n";
            return;
        }

        for (const auto& res : results) {
            std::cout << BOLD << "- Array: " << MAGENTA << res.array_name << RESET 
                      << " [" << CYAN << res.type_name << RESET << "]\n";
            
            if (!res.type_match) {
                std::cout << "  " << RED << "✖ Type mismatch or array missing in second file." << RESET << "\n\n";
                continue;
            }
            if (!res.size_match) {
                std::cout << "  " << RED << "✖ Size mismatch." << RESET << "\n\n";
                continue;
            }
            
            std::cout << "  Items Count:    " << res.items_count << "\n";
            
            std::cout << "  Max Difference: ";
            if (res.max_difference == 0.0) {
                std::cout << GREEN << res.max_difference << " (Perfect Match) ✔" << RESET << "\n";
            } else {
                std::cout << YELLOW << std::fixed << std::setprecision(6) << res.max_difference << RESET << "\n";
            }
            
            std::cout << "  Min Abs Value:  " << res.min_absolute_value << "\n";
            std::cout << "  Max Abs Value:  " << res.max_absolute_value << "\n\n";
        }

    } catch (const std::exception& e) {
        std::cerr << BOLD << RED << "Error during comparison: " << RESET << e.what() << "\n";
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string command = argv[1];

    if (command == "info") {
        if (argc != 3) {
            std::cerr << RED << "Error: 'info' requires exactly 1 file path." << RESET << "\n";
            print_usage(argv[0]);
            return 1;
        }
        handle_info(argv[2]);
        
    } else if (command == "compare") {
        if (argc != 4) {
            std::cerr << RED << "Error: 'compare' requires exactly 2 file paths." << RESET << "\n";
            print_usage(argv[0]);
            return 1;
        }
        handle_compare(argv[2], argv[3]);
        
    } else {
        std::cerr << RED << "Unknown command: " << command << RESET << "\n";
        print_usage(argv[0]);
        return 1;
    }

    return 0;
}