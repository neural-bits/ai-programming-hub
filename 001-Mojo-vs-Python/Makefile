
install:
	@echo "Installing Mojo"
	sudo apt update
	sudo apt install modular

	@echo "Creating Conda Environment Py311"
	conda create -n mojo python=3.11 -y && conda activate mojo

	@echo "Installing Mojo"
	modular install mojo

run_bench:
	@echo "Running the benchmark"
	@python 01_python_matmul_default.py
	@mojo 02_mojo_matmul_default.mojo
	@mojo 03_mojo_matmul_static_typed.mojo
	@mojo 04_mojo_matmul_advanced.mojo
