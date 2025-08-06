
#!/bin/bash
# run_app.sh - Switch between API or UI

MODE=$1

if [ "$MODE" = "api" ]; then
    echo "🚀 Starting FastAPI server..."
    uvicorn api_gptoss:app --host 0.0.0.0 --port 8000
elif [ "$MODE" = "ui" ]; then
    echo "💬 Launching Streamlit chatbot..."
    streamlit run streamlit_gptoss.py
else
    echo "Usage: ./run_app.sh [api|ui]"
fi
