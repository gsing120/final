#!/bin/bash

# Script to test the bundled application
echo "Testing the bundled Gemma Advanced Trading System application..."

# Check if the executable exists
if [ ! -f "/home/ubuntu/gemma_advanced/bundled_app/dist/GemmaAdvancedTrading" ]; then
    echo "Error: Executable not found at /home/ubuntu/gemma_advanced/bundled_app/dist/GemmaAdvancedTrading"
    exit 1
fi

# Create a test directory
TEST_DIR="/home/ubuntu/gemma_advanced/bundled_app/test"
mkdir -p "$TEST_DIR"

# Copy the executable to the test directory
cp "/home/ubuntu/gemma_advanced/bundled_app/dist/GemmaAdvancedTrading" "$TEST_DIR/"

# Change to the test directory
cd "$TEST_DIR"

# Test the executable (run in background with timeout)
echo "Starting the application in the background..."
./GemmaAdvancedTrading &
APP_PID=$!

# Wait for the application to start (5 seconds)
echo "Waiting for the application to start..."
sleep 5

# Check if the application is running
if ps -p $APP_PID > /dev/null; then
    echo "Application started successfully!"
    
    # Test if the web server is running by checking if port 5000 is open
    echo "Testing if the web server is running..."
    if netstat -tuln | grep ":5000 "; then
        echo "Web server is running on port 5000!"
        
        # Try to access the web server
        echo "Attempting to access the web server..."
        RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:5000)
        
        if [ "$RESPONSE" = "200" ]; then
            echo "Successfully accessed the web server! Response code: $RESPONSE"
            echo "Test PASSED!"
        else
            echo "Failed to access the web server. Response code: $RESPONSE"
            echo "Test FAILED!"
        fi
    else
        echo "Web server is not running on port 5000."
        echo "Test FAILED!"
    fi
    
    # Kill the application
    echo "Terminating the application..."
    kill $APP_PID
else
    echo "Application failed to start."
    echo "Test FAILED!"
fi

echo "Test completed."
