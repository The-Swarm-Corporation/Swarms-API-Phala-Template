# Swarms API

Build, deploy, and orchestrate AI agents at scale with ease. Swarms API provides a comprehensive suite of endpoints for creating and managing multi-agent systems.

## Docker Setup

To run the Swarms API using Docker, follow these steps:

1. **Build the Docker Image**
   
   Navigate to the directory containing the `Dockerfile` and run the following command:
   
   ```bash
   docker build -t swarms-api .
   ```

2. **Run the Docker Container**

   Once the image is built, you can run the container with:

   ```bash
   docker run -p 8080:8080 swarms-api
   ```

   This will start the API server on port 8080.

## Installing Requirements Locally

If you prefer to run the API locally without Docker, you can install the required Python packages using:

```bash
pip install -r api/requirements.txt
```

## API Endpoints

### Core Endpoints

- **`GET /health`**: Check API health.
- **`GET /v1/swarms/available`**: List available swarm types.

### Swarm Operation Endpoints

- **`POST /v1/swarm/completions`**: Run a single swarm task.
- **`POST /v1/swarm/batch/completions`**: Run multiple swarm tasks.
- **`GET /v1/swarm/logs`**: Retrieve API request logs.

### Scheduling Endpoints

- **`POST /v1/swarm/schedule`**: Schedule a swarm task.
- **`GET /v1/swarm/schedule`**: List all scheduled jobs.
- **`DELETE /v1/swarm/schedule/{job_id}`**: Cancel a scheduled job.


## Example Usage

Here's a basic example of running a swarm with multiple agents:

```python
import os
import requests
from dotenv import load_dotenv
import json

load_dotenv()

BASE_URL = "https://api.swarms.world"

headers = {"Content-Type": "application/json"}

def run_single_swarm():
    payload = {
        "name": "Financial Analysis Swarm",
        "description": "Market analysis swarm",
        "agents": [
            {
                "agent_name": "Market Analyst",
                "description": "Analyzes market trends",
                "system_prompt": "You are a financial analyst expert.",
                "model_name": "gpt-4o",
                "role": "worker",
                "max_loops": 1,
                "max_tokens": 8192,
                "temperature": 0.5
            },
            {
                "agent_name": "Economic Forecaster",
                "description": "Predicts economic trends",
                "system_prompt": "You are an expert in economic forecasting.",
                "model_name": "gpt-4o",
                "role": "worker",
                "max_loops": 1,
                "max_tokens": 8192,
                "temperature": 0.7
            },
            {
                "agent_name": "Data Scientist",
                "description": "Performs data analysis",
                "system_prompt": "You are a data science expert.",
                "model_name": "gpt-4o",
                "role": "worker",
                "max_loops": 1,
                "max_tokens": 8192,
                "temperature": 0.3
            },
        ],
        "max_loops": 1,
        "swarm_type": "ConcurrentWorkflow",
        "task": "What are the best ETFs and index funds for AI and tech?",
        "return_history": True,
    }

    response = requests.post(
        f"{BASE_URL}/v1/swarm/completions",
        headers=headers,
        json=payload,
    )

    return json.dumps(response.json(), indent=4)

if __name__ == "__main__":
    result = run_single_swarm()
    print("Swarm Result:")
    print(result)
```

## Getting Support

For questions or support, check the [documentation](https://docs.swarms.world/en/latest/swarms_cloud/swarms_api/) or contact kye@swarms.world.