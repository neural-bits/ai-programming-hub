install:
	@echo "Installing Rust and Python"
	@curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
	@conda create -n py38 python=3.8 -y
	@conda activate py38

build_debug:
	@echo "Building Rust Library in Debug Mode"
	@cd rust-llm && maturin develop
	@cd rust-llm && if [ -d target/wheels ]; then \
		find target/wheels -type f -name '*.whl' -exec pip install {} \; ; \
	else \
		echo "Directory target/wheels does not exist. No wheel files to install."; \
	fi

build_rel:
	@echo "Building Rust Library in Release Mode"
	@cd rust-llm && maturin build --release
	@cd rust-llm && if [ -d target/wheels ]; then \
		find target/wheels -type f -name '*.whl' -exec pip install {} \; ; \
	else \
		echo "Directory target/wheels does not exist. No wheel files to install."; \
	fi