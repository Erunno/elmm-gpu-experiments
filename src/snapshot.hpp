#ifndef SNAPSHOT_HPP
#define SNAPSHOT_HPP

#define NAME_MAX_LENGTH 255

#include <string>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <limits>

namespace snapshot {

#pragma pack(push, 1)
struct ArrayRecord {
    char name[NAME_MAX_LENGTH + 1];
    std::uint64_t byte_count;
    std::int32_t is_valid;    
    std::uint64_t offset;

    char type_name[16];

    ArrayRecord() : byte_count(0), is_valid(0), offset(0) {
        std::memset(name, 0, sizeof(name));
        std::memset(type_name, 0, sizeof(type_name));
    }
};


template<typename T>
struct TypeNamer {
    static const char* name() { return "unknown"; }
};

template<> class TypeNamer<float> { public: static const char* name() { return "float"; } };
template<> class TypeNamer<double> { public: static const char* name() { return "double"; } };
template<> class TypeNamer<std::int32_t> { public: static const char* name() { return "int32"; } };
template<> class TypeNamer<std::int64_t> { public: static const char* name() { return "int64"; } };
template<> class TypeNamer<std::uint32_t> { public: static const char* name() { return "uint32"; } };
template<> class TypeNamer<std::uint64_t> { public: static const char* name() { return "uint64"; } };

class Dumper {
public:
    template<typename ItemType>
    Dumper& register_array(const std::string& array_name, const std::vector<ItemType>& array_data) {
        return register_array(array_name, array_data.data(), array_data.size());
    }

    template<typename ItemType>
    Dumper& register_array(const std::string& array_name, const ItemType* array_data, std::size_t item_count) {
        if (array_name.length() >= NAME_MAX_LENGTH) {
            throw std::runtime_error("Array name exceeds maximum length of " + std::to_string(NAME_MAX_LENGTH));
        }

        ArrayRecord record;
        std::strncpy(record.name, array_name.c_str(), NAME_MAX_LENGTH - 1);
        std::strncpy(record.type_name, TypeNamer<ItemType>::name(), sizeof(record.type_name) - 1);

        record.byte_count = item_count * sizeof(ItemType);
        record.is_valid = 1;
        record.offset = total_byte_count;

        total_byte_count += record.byte_count;
        array_records.push_back(record);

        array_data_pointers.push_back(const_cast<ItemType*>(array_data));

        return *this;
    }

    Dumper& dump_to_file(const std::string& filename) {
        std::ofstream output_file(filename, std::ios::binary);

        if (!output_file) {
            throw std::runtime_error("Failed to open file for dumping: " + filename);
        }

        ArrayRecord last_array_record;
        last_array_record.is_valid = 0;
        array_records.push_back(last_array_record);

        for (const auto& record : array_records) {
            output_file.write(reinterpret_cast<const char*>(&record), sizeof(ArrayRecord));
        }

        for (std::size_t i = 0; i < array_data_pointers.size(); ++i) {
            if (array_records[i].is_valid) {
                output_file.write(reinterpret_cast<const char*>(array_data_pointers[i]), 
                                array_records[i].byte_count);
            }
        }

        array_records.clear();
        array_data_pointers.clear();
        total_byte_count = 0;

        output_file.close();

        return *this;
    }

private:
    std::vector<ArrayRecord> array_records;
    std::vector<void*> array_data_pointers;

    std::size_t total_byte_count = 0;
};

class Loader {
public:
    Loader& load_from_file(const std::string& filename) {
        std::ifstream input_file(filename, std::ios::binary);
        
        if (!input_file) {
            throw std::runtime_error("Failed to open file for loading: " + filename);
        }

        input_file.seekg(0, std::ios::end);
        std::size_t file_size = input_file.tellg();
        input_file.seekg(0, std::ios::beg);

        all_raw_data.resize(file_size);
        input_file.read(all_raw_data.data(), file_size);

        array_records = reinterpret_cast<ArrayRecord*>(all_raw_data.data());
        raw_array_data = all_raw_data.data() + sizeof(ArrayRecord) * (get_total_arrays() + 1);

        return *this;
    }

    template<typename ItemType>
    std::size_t get_array_size(const std::string& array_name) const {
        ArrayRecord* current_record = array_records;

        while (current_record->is_valid) {
            if (std::strncmp(current_record->name, array_name.c_str(), NAME_MAX_LENGTH) == 0) {
                return current_record->byte_count / sizeof(ItemType);
            }
            ++current_record;
        }

        throw std::runtime_error("Array not found: " + array_name);
    }

    template<typename ItemType>
    const Loader& load_array(const std::string& array_name, ItemType* array_data) const {
        ArrayRecord* current_record = array_records;

        while (current_record->is_valid) {
            if (std::strncmp(current_record->name, array_name.c_str(), NAME_MAX_LENGTH) == 0) {
                std::memcpy(array_data, raw_array_data + current_record->offset, current_record->byte_count);
                return *this;
            }
            ++current_record;
        }

        throw std::runtime_error("Array not found: " + array_name);
    }

    std::size_t get_total_arrays() const {
        ArrayRecord* current_record = array_records;
        std::size_t count = 0;

        while (current_record->is_valid) {
            ++count;
            ++current_record;
        }

        return count;
    }

    struct ArrayDumpInfo {
        std::string name;
        std::string type;
        std::size_t size_bytes;
    };

    std::vector<ArrayDumpInfo> get_dump_info() const {
        std::vector<ArrayDumpInfo> info;
        ArrayRecord* current_record = array_records;

        while (current_record->is_valid) {
            ArrayDumpInfo dump_info;
            dump_info.name = std::string(current_record->name);
            dump_info.type = std::string(current_record->type_name);
            dump_info.size_bytes = current_record->byte_count;
            info.push_back(dump_info);
            ++current_record;
        }

        return info;
    }

    std::vector<std::string> get_array_names() const {
        std::vector<std::string> names;
        ArrayRecord* current_record = array_records;

        while (current_record->is_valid) {
            names.push_back(std::string(current_record->name));
            ++current_record;
        }

        return names;
    }

    std::string get_array_type(const std::string& array_name) const {
        ArrayRecord* current_record = array_records;

        while (current_record->is_valid) {
            if (std::strncmp(current_record->name, array_name.c_str(), NAME_MAX_LENGTH) == 0) {
                return std::string(current_record->type_name);
            }
            ++current_record;
        }

        throw std::runtime_error("Array not found: " + array_name);
    }

private:
    ArrayRecord* array_records;
    char* raw_array_data;
    std::vector<char> all_raw_data;
};

struct ComparisonResult {
    std::string array_name;
    std::string type_name;
    std::size_t items_count;
    
    double max_difference;
    double max_absolute_value;
    double min_absolute_value;

    bool size_match;
    bool type_match;
    bool comparison_done;
};

class DumpsComparer {
public:
    static std::vector<ComparisonResult> compare_dumps(const std::string& file1, const std::string& file2) {
        Loader loader1, loader2;
        loader1.load_from_file(file1);
        loader2.load_from_file(file2);

        std::vector<ComparisonResult> results;

        for (const auto& name : loader1.get_array_names()) {
            auto result = handle_array_comparison(loader1, loader2, name);
            results.push_back(result);
        }

        return results;
    }
private:
    static ComparisonResult handle_array_comparison(const Loader& loader1, const Loader& loader2, const std::string& name) {
        ComparisonResult result;
        result.array_name = name;
        result.type_name = loader1.get_array_type(name);

        auto type1 = loader1.get_array_type(name);
        auto type2 = loader2.get_array_type(name);

        result.type_match = (type1 == type2);

        if (!result.type_match) {
            result.items_count = 0;
            result.max_difference = 0.0;
            result.size_match = false;
            return result;
        }
        
        if (type1 == "float") {
            compare_arrays<float>(loader1, loader2, name, result);
        } else if (type1 == "double") {
            compare_arrays<double>(loader1, loader2, name, result);
        } else if (type1 == "int32") {
            compare_arrays<std::int32_t>(loader1, loader2, name, result);
        } else if (type1 == "int64") {
            compare_arrays<std::int64_t>(loader1, loader2, name, result);
        } else if (type1 == "uint32") {
            compare_arrays<std::uint32_t>(loader1, loader2, name, result);
        } else if (type1 == "uint64") {
            compare_arrays<std::uint64_t>(loader1, loader2, name, result);
        } else {
            result.items_count = 0;
            result.max_difference = 0.0;
            result.size_match = false;
            result.type_match = false;
            result.comparison_done = false;
        }
        return result;
    }

    template<typename T>
    static void compare_arrays(const Loader& loader1, const Loader& loader2, const std::string& name, ComparisonResult& result) {
        std::size_t size1 = loader1.get_array_size<T>(name);
        std::size_t size2 = loader2.get_array_size<T>(name);

        if (size1 != size2) {
            result.items_count = 0;
            result.max_difference = 0.0;
            result.size_match = false;
            return;
        }

        std::vector<T> data1(size1);
        std::vector<T> data2(size2);

        loader1.load_array(name, data1.data());
        loader2.load_array(name, data2.data());

        double max_diff = 0.0;
        double max_total = std::numeric_limits<double>::min();
        double min_total = std::numeric_limits<double>::max();


        for (std::size_t i = 0; i < size1; ++i) {
            double abs_val1 = std::abs(static_cast<double>(data1[i]));
            double abs_val2 = std::abs(static_cast<double>(data2[i]));

            double diff = std::abs(abs_val1 - abs_val2);
 
            if (diff > max_diff) {
                max_diff = diff;
            }

            if (abs_val1 > max_total) {
                max_total = abs_val1;
            }
            if (abs_val2 > max_total) {
                max_total = abs_val2;
            }

            if (abs_val1 < min_total) {
                min_total = abs_val1;
            }
            if (abs_val2 < min_total) {
                min_total = abs_val2;
            }
        }
        
        result.items_count = size1;
        
        result.max_difference = max_diff;
        result.max_absolute_value = max_total;
        result.min_absolute_value = min_total;

        result.size_match = true;
    }
};

} // namespace snapshot

#endif // SNAPSHOT_HPP