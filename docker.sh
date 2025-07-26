#!/bin/bash

# Docker Simulator Script
# This script simulates Docker commands locally for the MLOps assignment

IMAGE_NAME="mlops-assignment-3"
CONTAINER_NAME="mlops-container"

print_step() {
    echo "============================================================"
    echo "🐳 DOCKER SIMULATION: $1"
    echo "📝 $2"
    echo "============================================================"
}

case "$1" in
    "build")
        print_step "BUILD" "Building Docker image (simulated)"
        
        if [ ! -f "requirements.txt" ]; then
            echo "❌ Error: requirements.txt not found"
            exit 1
        fi
        
        if [ ! -d "src" ]; then
            echo "❌ Error: src directory not found"
            exit 1
        fi
        
        echo "📁 Copying requirements.txt..."
        echo "📁 Copying src/ directory..."
        
        if [ ! -d "models" ]; then
            echo "📁 Creating models/ directory..."
            mkdir -p models
        fi
        
        echo "📁 Copying models/ directory..."
        echo "🔧 Installing Python dependencies..."
        
        if pip install -r requirements.txt; then
            echo "✅ Dependencies installed successfully"
        else
            echo "❌ Error installing dependencies"
            exit 1
        fi
        
        echo "✅ Docker image built successfully (simulated)"
        ;;
        
    "run")
        print_step "RUN" "Running Docker container (simulated)"
        
        if [ ! -f "models/sklearn_model.joblib" ]; then
            echo "❌ Error: Model file not found. Please run train.py first."
            exit 1
        fi
        
        echo "🚀 Starting container..."
        echo "🔧 Setting environment variables..."
        export PYTHONPATH=$(pwd)
        export MODEL_PATH=$(pwd)/models/sklearn_model.joblib
        echo "   PYTHONPATH=$PYTHONPATH"
        echo "   MODEL_PATH=$MODEL_PATH"
        
        echo "📊 Running prediction script..."
        if python src/predict.py; then
            echo "✅ Container executed successfully"
        else
            echo "❌ Container execution failed"
            exit 1
        fi
        ;;
        
    "push")
        print_step "PUSH" "Pushing to registry (simulated)"
        
        REGISTRY=${2:-"localhost"}
        echo "📤 Tagging image as $REGISTRY/$IMAGE_NAME..."
        echo "📤 Pushing image to $REGISTRY..."
        echo "✅ Image pushed successfully (simulated)"
        ;;
        
    "full")
        print_step "FULL PIPELINE" "Running complete Docker simulation"
        
        echo "🚀 Starting Docker Simulation Pipeline"
        echo "This simulates the complete Docker workflow locally"
        
        # Build
        if ! $0 build; then
            exit 1
        fi
        
        # Run
        if ! $0 run; then
            exit 1
        fi
        
        # Push
        $0 push
        
        echo ""
        echo "🎉 Docker simulation completed successfully!"
        ;;
        
    *)
        echo "Usage: $0 {build|run|push|full}"
        echo ""
        echo "Commands:"
        echo "  build   - Simulate docker build"
        echo "  run     - Simulate docker run"
        echo "  push    - Simulate docker push"
        echo "  full    - Run complete pipeline"
        exit 1
        ;;
esac 