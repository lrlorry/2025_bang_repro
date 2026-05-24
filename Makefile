.PHONY: check build smoke estimate plot clean help

REPO_ROOT := $(shell pwd)
SCRIPTS   := $(REPO_ROOT)/scripts

help:
	@echo "BANG 复现项目 Makefile"
	@echo ""
	@echo "  make check     检查环境"
	@echo "  make build     编译 BANG_Base"
	@echo "  make smoke     SIFT10K smoke test（需 GPU）"
	@echo "  make estimate  估算各规模资源"
	@echo "  make plot      生成图表"
	@echo "  make clean     清除 results/ 和 figures/ 下的输出文件"

check:
	bash $(SCRIPTS)/check_env.sh

build:
	bash $(SCRIPTS)/build_bang_base.sh

smoke:
	bash $(SCRIPTS)/run_sift10k_smoke.sh

estimate:
	python3 $(SCRIPTS)/estimate_resources.py

plot:
	python3 $(SCRIPTS)/plot_results.py \
		--results_dir $(REPO_ROOT)/results \
		--figures_dir $(REPO_ROOT)/figures

clean:
	find $(REPO_ROOT)/results -type f ! -name '.gitkeep' -delete
	find $(REPO_ROOT)/figures -type f ! -name '.gitkeep' -delete
	@echo "清除完成（.gitkeep 保留）"
