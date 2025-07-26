#!/usr/bin/env python3
"""
Docker Simulator - Simulates Docker operations locally
This script mimics Docker build and run operations for the MLOps assignment
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

class DockerSimulator:
    def __init__(self):
        self.image_name = "mlops-assignment-3"
        self.container_name = "mlops-container"
        self.work_dir = Path.cwd()
        
    def print_step(self, step, description):
        print(f"\n{'='*60}")
        print(f"🐳 DOCKER SIMULATION: {step}")
        print(f"📝 {description}")
        print(f"{'='*60}")
    
    def simulate_build(self):
        """Simulate docker build"""
        self.print_step("BUILD", "Building Docker image (simulated)")
        
        # Check if requirements.txt exists
        if not os.path.exists("requirements.txt"):
            print("❌ Error: requirements.txt not found")
            return False
            
        # Check if src directory exists
        if not os.path.exists("src"):
            print("❌ Error: src directory not found")
            return False
            
        # Simulate copying files
        print("📁 Copying requirements.txt...")
        print("📁 Copying src/ directory...")
        
        # Check if models directory exists, if not create it
        if not os.path.exists("models"):
            print("📁 Creating models/ directory...")
            os.makedirs("models", exist_ok=True)
            
        print("📁 Copying models/ directory...")
        
        # Simulate installing dependencies
        print("🔧 Installing Python dependencies...")
        try:
            result = subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Dependencies installed successfully")
            else:
                print(f"❌ Error installing dependencies: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
            
        print("✅ Docker image built successfully (simulated)")
        return True
    
    def simulate_run(self):
        """Simulate docker run"""
        self.print_step("RUN", "Running Docker container (simulated)")
        
        # Check if model exists
        model_path = "models/sklearn_model.joblib"
        if not os.path.exists(model_path):
            print("❌ Error: Model file not found. Please run train.py first.")
            return False
            
        # Set environment variables (simulate Docker ENV)
        os.environ['PYTHONPATH'] = str(self.work_dir)
        os.environ['MODEL_PATH'] = str(self.work_dir / model_path)
        
        print("🚀 Starting container...")
        print("🔧 Setting environment variables...")
        print(f"   PYTHONPATH={os.environ['PYTHONPATH']}")
        print(f"   MODEL_PATH={os.environ['MODEL_PATH']}")
        
        # Run the prediction script
        print("📊 Running prediction script...")
        try:
            result = subprocess.run([sys.executable, "src/predict.py"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Container executed successfully")
                print("📋 Output:")
                print(result.stdout)
            else:
                print(f"❌ Container execution failed: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
            
        return True
    
    def simulate_push(self, registry="localhost"):
        """Simulate docker push"""
        self.print_step("PUSH", f"Pushing to registry {registry} (simulated)")
        
        print(f"📤 Tagging image as {registry}/{self.image_name}...")
        print(f"📤 Pushing image to {registry}...")
        print("✅ Image pushed successfully (simulated)")
        return True
    
    def run_full_pipeline(self):
        """Run the complete Docker simulation pipeline"""
        print("🚀 Starting Docker Simulation Pipeline")
        print("This simulates the complete Docker workflow locally")
        
        # Build
        if not self.simulate_build():
            return False
            
        # Run
        if not self.simulate_run():
            return False
            
        # Push (optional)
        self.simulate_push()
        
        print("\n🎉 Docker simulation completed successfully!")
        return True

def main():
    simulator = DockerSimulator()
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "build":
            simulator.simulate_build()
        elif command == "run":
            simulator.simulate_run()
        elif command == "push":
            registry = sys.argv[2] if len(sys.argv) > 2 else "localhost"
            simulator.simulate_push(registry)
        else:
            print("Usage: python docker_simulator.py [build|run|push]")
    else:
        simulator.run_full_pipeline()

if __name__ == "__main__":
    main() 