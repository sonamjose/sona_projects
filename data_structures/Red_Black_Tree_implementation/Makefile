# Makefile for creating an executable from Python script

# Variables
PYTHON = python3
SCRIPT = plateMgmt.py  # Your main Python script
EXE = plateMgmt  # Name of the output executable
BUILD_DIR = dist  # Directory for the output executable

# Define the path to pyinstaller (adjust this based on where pyinstaller is located)
PYINSTALLER_PATH := $(HOME)/.local/bin

# Add pyinstaller to the PATH so that make can find it
export PATH := $(PYINSTALLER_PATH):$(PATH)

# Default target
all: build

# Create an executable using PyInstaller
build:
	$(PYTHON) -m pip install --upgrade pyinstaller  # Ensure PyInstaller is installed
	pyinstaller --onefile --name $(EXE) $(SCRIPT)
	
# Run the executable
run: build
	./$(BUILD_DIR)/$(EXE)

# Clean up generated files
clean:
	rm -rf $(BUILD_DIR) build __pycache__ $(EXE).spec

