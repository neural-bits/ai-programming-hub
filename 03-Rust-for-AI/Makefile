YELLOW=\033[1;33m
GREEN=\033[1;32m
BLUE=\033[1;34m
RED=\033[1;31m
NC=\033[0m

list:
	@echo "Legend:"
	@echo "$(YELLOW)install$(NC) - Install environemt for Rust and Python"
	@echo "$(GREEN)build_debug$(NC) - Build the rust library in debug mode (without optimizations)"
	@echo "$(BLUE)build_rel$(NC) - Build the Rust library in release"

install:
	@echo "$(YELLOW)Installing Rust and Python$(NC)"
	@curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
	@echo "$(YELLOW)Creating Python environment$(NC)"
	@conda create -n py38 python=3.8 -y
	@echo "$(YELLOW)Activating Python environment$(NC)"
	@conda activate py38

build_debug:
	@echo "$(BLUE)Building Rust Library in Debug Mode$(NC)"
	@cd rust-llm && maturin develop
	@cd rust-llm && if [ -d target/wheels ]; then \
		echo "$(GREEN)Installing the wheel package$(NC)"; \
		find target/wheels -type f -name '*.whl' -exec pip install {} \; ; \
	else \
		echo "$(RED)Directory target/wheels does not exist. No wheel files to install.$(NC)"; \
	fi

build_rel:
	@echo "$(GREEN)Building Rust Library in Release Mode$(NC)"
	@cd rust-llm && maturin build --release
	@cd rust-llm && if [ -d target/wheels ]; then \
		echo "$(GREEN)Installing the wheel package$(NC)"; \
		find target/wheels -type f -name '*.whl' -exec pip install {} \; ; \
	else \
		echo "$(RED)Directory target/wheels does not exist. No wheel files to install.$(NC)"; \
	fi