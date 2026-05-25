BUILD_DIR ?= build

.PHONY: all official plain engineered clean

all: official

official:
	bash scripts/build_official_bang.sh

plain: $(BUILD_DIR)/bang_plain

engineered: $(BUILD_DIR)/bang_engineered

$(BUILD_DIR)/bang_plain $(BUILD_DIR)/bang_engineered: CMakeLists.txt
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake .. -DCMAKE_BUILD_TYPE=Release
	cmake --build $(BUILD_DIR) -- -j$(shell nproc)

clean:
	rm -rf $(BUILD_DIR)
