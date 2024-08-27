
install_mac:
	@echo "Installing Mojo"
	sudo brew update
	sudo brew install modular

	@echo "Creating Conda Environment Py311"
	conda create -n mojo python=3.11 -y && conda activate mojo

	@echo "Installing Mojo"
	modular install mojo

install_linux:
	@echo "Installing Mojo"
	sudo apt update
	sudo apt install modular

	@echo "Creating Conda Environment Py310"
	conda create -n mojo python=3.10 -y && conda activate mojo

	@echo "Installing Mojo"
	modular install mojo

run_bench_custom:
	@echo "Running the benchmark"
	@python 01_python_matmul_default.py
	@mojo 02_mojo_matmul_default.mojo
	@mojo 03_mojo_matmul_static_typed.mojo
	@mojo 04_mojo_matmul_advanced.mojo
