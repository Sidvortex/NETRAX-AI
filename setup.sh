#!/bin/bash

# NETRAX AI - Vision System Setup Script
# Automates installation and configuration

set -e

echo "🔴 NETRAX AI - Vision System Setup"
echo "===================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Detect OS
OS="unknown"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    OS="windows"
fi

echo -e "${GREEN}Detected OS: $OS${NC}"
echo ""

# Check Python version
echo "Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python 3 not found. Please install Python 3.10+${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo -e "${GREEN}Python version: $PYTHON_VERSION${NC}"

if (( $(echo "$PYTHON_VERSION < 3.10" | bc -l) )); then
    echo -e "${RED}Python 3.10+ required. Current: $PYTHON_VERSION${NC}"
    exit 1
fi
echo ""

# Create directory structure
echo "Creating directory structure..."
mkdir -p vision_engine
mkdir -p models
mkdir -p logs
touch vision_engine/__init__.py
echo -e "${GREEN}✓ Directories created${NC}"
echo ""

# Setup virtual environment
echo "Setting up virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${YELLOW}Virtual environment already exists${NC}"
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel
echo -e "${GREEN}✓ Pip upgraded${NC}"
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt
echo -e "${GREEN}✓ Dependencies installed${NC}"
echo ""

# Setup configuration
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    cp .env.example .env
    echo -e "${GREEN}✓ Configuration file created${NC}"
    echo -e "${YELLOW}⚠ Please edit .env with your settings${NC}"
else
    echo -e "${YELLOW}.env file already exists${NC}"
fi
echo ""

# Check camera access
echo "Checking camera access..."
if [[ "$OS" == "linux" ]]; then
    if [ -e "/dev/video0" ]; then
        echo -e "${GREEN}✓ Camera found at /dev/video0${NC}"
    else
        echo -e "${YELLOW}⚠ No camera found at /dev/video0${NC}"
        echo "Available video devices:"
        ls -la /dev/video* 2>/dev/null || echo "None found"
    fi
elif [[ "$OS" == "macos" ]]; then
    echo -e "${YELLOW}⚠ Camera access: Grant permissions in System Preferences${NC}"
fi
echo ""

# Test MediaPipe installation
echo "Testing MediaPipe installation..."
python3 -c "import mediapipe as mp; print('MediaPipe version:', mp.__version__)" && \
    echo -e "${GREEN}✓ MediaPipe working${NC}" || \
    echo -e "${RED}✗ MediaPipe installation failed${NC}"
echo ""

# Test OpenCV installation
echo "Testing OpenCV installation..."
python3 -c "import cv2; print('OpenCV version:', cv2.__version__)" && \
    echo -e "${GREEN}✓ OpenCV working${NC}" || \
    echo -e "${RED}✗ OpenCV installation failed${NC}"
echo ""

# Optional: Install object detection
read -p "Install object detection (YOLOv8)? This is heavy (~500MB). [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing ultralytics..."
    pip install ultralytics
    echo -e "${GREEN}✓ Object detection installed${NC}"
else
    echo -e "${YELLOW}Skipping object detection${NC}"
fi
echo ""

# Check GPU availability
echo "Checking GPU availability..."
python3 -c "
try:
    import torch
    if torch.cuda.is_available():
        print('✓ CUDA available')
        print('  Device:', torch.cuda.get_device_name(0))
        print('  CUDA Version:', torch.version.cuda)
    else:
        print('ℹ CUDA not available - using CPU')
except ImportError:
    print('ℹ PyTorch not installed - using CPU')
" 2>/dev/null || echo -e "${YELLOW}Using CPU mode${NC}"
echo ""

# Create systemd service (Linux only)
if [[ "$OS" == "linux" ]]; then
    read -p "Create systemd service for auto-start? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        SERVICE_FILE="/etc/systemd/system/netrax-vision.service"
        WORK_DIR=$(pwd)
        
        sudo tee $SERVICE_FILE > /dev/null <<EOF
[Unit]
Description=NETRAX AI Vision System
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$WORK_DIR
Environment="PATH=$WORK_DIR/venv/bin"
ExecStart=$WORK_DIR/venv/bin/python main.py
Restart=always

[Install]
WantedBy=multi-user.target
EOF
        
        sudo systemctl daemon-reload
        echo -e "${GREEN}✓ Service created${NC}"
        echo "Start with: sudo systemctl start netrax-vision"
        echo "Enable auto-start: sudo systemctl enable netrax-vision"
    fi
    echo ""
fi

# Docker setup
if command -v docker &> /dev/null; then
    echo -e "${GREEN}✓ Docker detected${NC}"
    read -p "Build Docker image? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Building Docker image..."
        docker-compose build
        echo -e "${GREEN}✓ Docker image built${NC}"
        echo "Run with: docker-compose up -d"
    fi
else
    echo -e "${YELLOW}Docker not found - skipping Docker setup${NC}"
fi
echo ""

# Final instructions
echo "===================================="
echo -e "${GREEN}🔴 Setup Complete!${NC}"
echo "===================================="
echo ""
echo "Next steps:"
echo "1. Edit .env with your configuration"
echo "2. Run the server:"
echo "   python main.py"
echo "   OR"
echo "   docker-compose up -d"
echo ""
echo "3. Open frontend:"
echo "   Open index.html in your browser"
echo ""
echo "4. Access API:"
echo "   http://localhost:8000"
echo "   ws://localhost:8000/ws"
echo ""
echo -e "${YELLOW}⚠ Remember to grant camera permissions!${NC}"
echo ""
echo "Documentation: README.md"
echo "Support: Check troubleshooting section"
echo ""
echo -e "${GREEN}NETRAX IS READY TO WATCH 👁️${NC}"