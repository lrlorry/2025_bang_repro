BUILD_DIR ?= build

.PHONY: all clean

all: $(BUILD_DIR)/bang_plain

$(BUILD_DIR)/bang_plain: CMakeLists.txt plain/plain_search.cu plain/plain_main.cu
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake .. -DCMAKE_BUILD_TYPE=Release
	cmake --build $(BUILD_DIR) --target bang_plain -- -j$(shell nproc)

clean:
	rm -rf $(BUILD_DIR)
